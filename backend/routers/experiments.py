from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from backend.database import get_db
from backend.schemas.experiment import (
    ExperimentComparison,
    ExperimentConfig,
    ExperimentDetail,
    ExperimentOut,
    ExperimentRunOut,
)

router = APIRouter()

# Track running experiment tasks so they can be cancelled
_running_tasks: dict[str, asyncio.Task] = {}


@router.post("", response_model=ExperimentOut)
async def create_experiment(config: ExperimentConfig, request: Request):
    """Create and start a new experiment."""
    # Extract OpenAI API key from header
    openai_api_key = request.headers.get("X-OpenAI-API-Key")
    if not openai_api_key:
        raise HTTPException(
            status_code=400,
            detail="OpenAI API key required. Please set it in Settings."
        )

    experiment_id = str(uuid.uuid4())

    db = await get_db()
    try:
        # Verify dataset exists
        cursor = await db.execute(
            "SELECT id, target_col, task_type FROM datasets WHERE id = ?",
            (config.dataset_id,),
        )
        dataset = await cursor.fetchone()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        await db.execute(
            """INSERT INTO experiments (id, dataset_id, status, config, strategy, epsilon)
               VALUES (?, ?, 'running', ?, ?, ?)""",
            (
                experiment_id,
                config.dataset_id,
                config.model_dump_json(),
                config.strategy,
                config.epsilon,
            ),
        )
        await db.commit()
    finally:
        await db.close()

    # Launch the orchestrator as a background task
    from backend.agents.orchestrator import run_experiment

    task = asyncio.create_task(run_experiment(experiment_id, config, openai_api_key))
    _running_tasks[experiment_id] = task

    return ExperimentOut(
        id=experiment_id,
        dataset_id=config.dataset_id,
        status="running",
        strategy=config.strategy,
        epsilon=config.epsilon,
    )


@router.get("", response_model=list[ExperimentOut])
async def list_experiments():
    """List all experiments."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, dataset_id, status, strategy, epsilon, created_at, finished_at FROM experiments ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()
        return [ExperimentOut(**dict(row)) for row in rows]
    finally:
        await db.close()


@router.get("/compare", response_model=ExperimentComparison)
async def compare_experiments(exp_id_1: str, exp_id_2: str):
    """Compare two experiments side-by-side."""
    # Fetch both experiments using the same logic as get_experiment (defined below)
    exp1 = await get_experiment(exp_id_1)
    exp2 = await get_experiment(exp_id_2)

    return ExperimentComparison(
        experiment_1=exp1,
        experiment_2=exp2,
    )


@router.get("/{experiment_id}", response_model=ExperimentDetail)
async def get_experiment(experiment_id: str):
    """Get experiment details including runs and best metrics."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Experiment not found")

        row_dict = dict(row)
        config_raw = row_dict.get("config")
        config = (
            ExperimentConfig.model_validate_json(config_raw) if config_raw else None
        )

        # Get runs
        cursor = await db.execute(
            "SELECT * FROM experiment_runs WHERE experiment_id = ? ORDER BY iteration",
            (experiment_id,),
        )
        run_rows = await cursor.fetchall()
        runs = []
        best_metrics: dict[str, float] = {}
        for r in run_rows:
            rd = dict(r)
            metrics = json.loads(rd.get("metrics") or "{}")
            hp = json.loads(rd.get("hyperparameters") or "{}")
            runs.append(
                ExperimentRunOut(
                    id=rd["id"],
                    experiment_id=rd["experiment_id"],
                    iteration=rd["iteration"],
                    model_name=rd["model_name"],
                    hyperparameters=hp,
                    metrics=metrics,
                    train_time_s=rd.get("train_time_s"),
                    ucb_score=rd.get("ucb_score"),
                    created_at=rd.get("created_at"),
                )
            )
            # Track best metric values
            for k, v in metrics.items():
                if k not in best_metrics or v > best_metrics[k]:
                    best_metrics[k] = v

        return ExperimentDetail(
            id=row_dict["id"],
            dataset_id=row_dict["dataset_id"],
            status=row_dict["status"],
            strategy=row_dict.get("strategy", "balanced"),
            epsilon=row_dict.get("epsilon", 0.3),
            created_at=row_dict.get("created_at"),
            finished_at=row_dict.get("finished_at"),
            config=config,
            runs=runs,
            best_metrics=best_metrics,
        )
    finally:
        await db.close()


@router.post("/{experiment_id}/stop")
async def stop_experiment(experiment_id: str):
    """Stop a running experiment."""
    task = _running_tasks.get(experiment_id)
    if task and not task.done():
        task.cancel()

    db = await get_db()
    try:
        await db.execute(
            "UPDATE experiments SET status = 'stopped', finished_at = datetime('now') WHERE id = ? AND status = 'running'",
            (experiment_id,),
        )
        await db.commit()
    finally:
        await db.close()

    return {"status": "stopped"}


@router.get("/{experiment_id}/export")
async def export_best_model(experiment_id: str):
    """Download the best model as a pickle file."""
    import os
    from backend.services.metrics_service import is_higher_better

    db = await get_db()
    try:
        # Get experiment config to determine primary metric
        cursor = await db.execute(
            "SELECT config FROM experiments WHERE id = ?", (experiment_id,)
        )
        exp_row = await cursor.fetchone()
        if not exp_row:
            raise HTTPException(status_code=404, detail="Experiment not found")

        config_json = exp_row["config"]
        config = ExperimentConfig.model_validate_json(config_json) if config_json else None
        primary_metric = config.primary_metric if config else "f1"

        # Get all runs with artifact paths
        cursor = await db.execute(
            """SELECT id, model_name, metrics, model_artifact_path
               FROM experiment_runs
               WHERE experiment_id = ? AND model_artifact_path IS NOT NULL
               ORDER BY iteration""",
            (experiment_id,),
        )
        rows = await cursor.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail="No model artifacts found")

        # Find best run by primary metric
        best_run = None
        best_metric_val = float("-inf") if is_higher_better(primary_metric) else float("inf")

        for row in rows:
            metrics = json.loads(row["metrics"] or "{}")
            val = metrics.get(primary_metric, 0)

            if is_higher_better(primary_metric):
                if val > best_metric_val:
                    best_metric_val = val
                    best_run = row
            else:
                if val < best_metric_val:
                    best_metric_val = val
                    best_run = row

        if not best_run or not best_run["model_artifact_path"]:
            raise HTTPException(status_code=404, detail="Best model artifact not found")

        artifact_path = best_run["model_artifact_path"]
        if not os.path.exists(artifact_path):
            raise HTTPException(status_code=404, detail="Model file not found on disk")

        filename = f"{best_run['model_name']}_best.pkl"
        return FileResponse(
            artifact_path,
            media_type="application/octet-stream",
            filename=filename,
        )

    finally:
        await db.close()


@router.get("/{experiment_id}/data-versions")
async def get_data_versions(experiment_id: str):
    """Get dataset version lineage for an experiment."""
    from backend.services.dataset_service import get_dataset_versions

    versions = await get_dataset_versions(experiment_id)
    return versions


@router.get("/{experiment_id}/export-pdf")
async def export_experiment_pdf(experiment_id: str):
    """Generate and download a comprehensive PDF report of the experiment."""
    from backend.services.pdf_export import generate_experiment_report

    # Fetch experiment details
    exp_detail = await get_experiment(experiment_id)

    # Convert to dict for PDF generation
    exp_data = {
        'id': exp_detail.id,
        'dataset_id': exp_detail.dataset_id,
        'status': exp_detail.status,
        'strategy': exp_detail.strategy,
        'epsilon': exp_detail.epsilon,
        'created_at': exp_detail.created_at,
        'finished_at': exp_detail.finished_at,
        'config': exp_detail.config.model_dump() if exp_detail.config else None,
        'best_metrics': exp_detail.best_metrics,
    }

    # Convert runs to dict
    runs_data = [
        {
            'iteration': run.iteration,
            'model_name': run.model_name,
            'hyperparameters': run.hyperparameters,
            'metrics': run.metrics,
            'train_time_s': run.train_time_s,
            'ucb_score': run.ucb_score,
        }
        for run in exp_detail.runs
    ]

    # Generate PDF
    pdf_bytes = generate_experiment_report(exp_data, runs_data)

    from fastapi.responses import Response

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=experiment_{experiment_id[:8]}_report.pdf"
        },
    )
