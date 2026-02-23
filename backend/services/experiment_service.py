from __future__ import annotations

import json
import uuid

from backend.database import get_db


async def save_run(
    experiment_id: str,
    iteration: int,
    model_name: str,
    hyperparameters: dict,
    metrics: dict,
    train_time_s: float,
    feature_config: dict | None = None,
    ucb_score: float | None = None,
    model_artifact_path: str | None = None,
) -> str:
    """Save a model run to the database."""
    run_id = str(uuid.uuid4())
    db = await get_db()
    try:
        await db.execute(
            """INSERT INTO experiment_runs
               (id, experiment_id, iteration, model_name, hyperparameters, metrics, train_time_s, feature_config, ucb_score, model_artifact_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                experiment_id,
                iteration,
                model_name,
                json.dumps(hyperparameters),
                json.dumps(metrics),
                train_time_s,
                json.dumps(feature_config) if feature_config else None,
                ucb_score,
                model_artifact_path,
            ),
        )
        await db.commit()
    finally:
        await db.close()
    return run_id


async def save_trace(
    experiment_id: str,
    iteration: int | None,
    agent_name: str,
    step_type: str,
    content: str,
    parent_trace_id: str | None = None,
    depth: int = 0,
    score: float | None = None,
    selected: bool = False,
    metadata: dict | None = None,
) -> str:
    """Save an agent trace entry to the database."""
    trace_id = str(uuid.uuid4())
    db = await get_db()
    try:
        await db.execute(
            """INSERT INTO agent_traces
               (id, experiment_id, iteration, agent_name, step_type, content, parent_trace_id, depth, score, selected, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trace_id,
                experiment_id,
                iteration,
                agent_name,
                step_type,
                content,
                parent_trace_id,
                depth,
                score,
                1 if selected else 0,
                json.dumps(metadata) if metadata else None,
            ),
        )
        await db.commit()
    finally:
        await db.close()
    return trace_id


async def get_run_history(experiment_id: str) -> list[dict]:
    """Get all runs for an experiment."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM experiment_runs WHERE experiment_id = ? ORDER BY iteration",
            (experiment_id,),
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            rd = dict(row)
            rd["metrics"] = json.loads(rd.get("metrics") or "{}")
            rd["hyperparameters"] = json.loads(rd.get("hyperparameters") or "{}")
            results.append(rd)
        return results
    finally:
        await db.close()


async def update_experiment_status(experiment_id: str, status: str) -> None:
    """Update experiment status."""
    db = await get_db()
    try:
        if status in ("completed", "failed", "stopped"):
            await db.execute(
                "UPDATE experiments SET status = ?, finished_at = datetime('now') WHERE id = ?",
                (status, experiment_id),
            )
        else:
            await db.execute(
                "UPDATE experiments SET status = ? WHERE id = ?",
                (status, experiment_id),
            )
        await db.commit()
    finally:
        await db.close()
