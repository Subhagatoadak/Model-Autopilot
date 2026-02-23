from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.config import settings
from backend.database import get_db
from backend.schemas.dataset import (
    ColumnInfo,
    DatasetCreate,
    DatasetDBConnect,
    DatasetDetail,
    DatasetOut,
    DatasetProfile,
)

router = APIRouter()


def _profile_dataframe(df: pd.DataFrame) -> tuple[list[ColumnInfo], DatasetProfile]:
    """Generate column info and profile from a DataFrame."""
    columns = []
    for col in df.columns:
        null_pct = float(df[col].isnull().mean() * 100)
        unique_count = int(df[col].nunique())
        sample_vals = [str(v) for v in df[col].dropna().head(5).tolist()]
        columns.append(
            ColumnInfo(
                name=col,
                dtype=str(df[col].dtype),
                null_pct=round(null_pct, 2),
                unique_count=unique_count,
                sample_values=sample_vals,
            )
        )
    profile = DatasetProfile(
        num_rows=len(df),
        num_cols=len(df.columns),
        columns=columns,
    )
    return columns, profile


@router.post("", response_model=DatasetOut)
@router.post("/upload", response_model=DatasetOut)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(""),
    target_col: str = Form(""),
    task_type: str = Form(""),
):
    """Upload a CSV file as a new dataset."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    dataset_id = str(uuid.uuid4())
    dataset_name = name or file.filename.rsplit(".", 1)[0]

    # Save file
    root = Path(__file__).resolve().parent.parent.parent
    upload_dir = root / settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{dataset_id}.csv"

    content = await file.read()
    if len(content) > settings.max_bytes:
        raise HTTPException(status_code=400, detail="File too large")

    file_path.write_bytes(content)

    # Profile
    df = pd.read_csv(file_path)
    columns, profile = _profile_dataframe(df)

    # Auto-detect task type
    detected_task = task_type or None
    detected_target = target_col or None
    if detected_target and not detected_task:
        col_data = df[detected_target]
        if col_data.dtype == "object" or col_data.nunique() <= 20:
            detected_task = "classification"
        else:
            detected_task = "regression"

    profile.target_col = detected_target
    profile.task_type = detected_task

    # Save to DB
    db = await get_db()
    try:
        await db.execute(
            """INSERT INTO datasets (id, name, file_path, num_rows, num_cols, columns, target_col, task_type, profile)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                dataset_id,
                dataset_name,
                str(file_path),
                len(df),
                len(df.columns),
                json.dumps([c.model_dump() for c in columns]),
                detected_target,
                detected_task,
                profile.model_dump_json(),
            ),
        )
        await db.commit()
    finally:
        await db.close()

    return DatasetOut(
        id=dataset_id,
        name=dataset_name,
        num_rows=len(df),
        num_cols=len(df.columns),
        target_col=detected_target,
        task_type=detected_task,
    )


@router.post("/connect", response_model=DatasetOut)
async def connect_database(req: DatasetDBConnect):
    """Connect to a PostgreSQL database and create a dataset from a query."""
    try:
        df = pd.read_sql(req.query, req.db_uri)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database connection failed: {e}")

    dataset_id = str(uuid.uuid4())
    columns, profile = _profile_dataframe(df)

    detected_task = req.task_type or None
    detected_target = req.target_col or None
    if detected_target and not detected_task:
        col_data = df[detected_target]
        if col_data.dtype == "object" or col_data.nunique() <= 20:
            detected_task = "classification"
        else:
            detected_task = "regression"

    profile.target_col = detected_target
    profile.task_type = detected_task

    # Save CSV copy for training
    root = Path(__file__).resolve().parent.parent.parent
    upload_dir = root / settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{dataset_id}.csv"
    df.to_csv(file_path, index=False)

    db = await get_db()
    try:
        await db.execute(
            """INSERT INTO datasets (id, name, file_path, db_uri, db_query, num_rows, num_cols, columns, target_col, task_type, profile)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                dataset_id,
                req.name,
                str(file_path),
                req.db_uri,
                req.query,
                len(df),
                len(df.columns),
                json.dumps([c.model_dump() for c in columns]),
                detected_target,
                detected_task,
                profile.model_dump_json(),
            ),
        )
        await db.commit()
    finally:
        await db.close()

    return DatasetOut(
        id=dataset_id,
        name=req.name,
        num_rows=len(df),
        num_cols=len(df.columns),
        target_col=detected_target,
        task_type=detected_task,
    )


@router.get("", response_model=list[DatasetOut])
async def list_datasets():
    """List all datasets."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, name, num_rows, num_cols, target_col, task_type, created_at FROM datasets ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()
        return [DatasetOut(**dict(row)) for row in rows]
    finally:
        await db.close()


@router.get("/{dataset_id}/columns")
async def get_dataset_columns(dataset_id: str):
    """Get column names and types for a dataset."""
    from backend.services.dataset_service import get_dataset_columns as _get_cols
    cols = await _get_cols(dataset_id)
    if not cols:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return cols


@router.get("/{dataset_id}/preview")
async def get_dataset_preview(dataset_id: str, n: int = 10):
    """Get first N rows of a dataset."""
    from backend.services.dataset_service import get_dataset_preview as _preview
    rows = await _preview(dataset_id, n)
    return rows


@router.get("/{dataset_id}", response_model=DatasetDetail)
async def get_dataset(dataset_id: str):
    """Get dataset details including column info and profile."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Dataset not found")

        row_dict = dict(row)
        columns = json.loads(row_dict.get("columns") or "[]")
        profile_raw = row_dict.get("profile")
        profile = DatasetProfile.model_validate_json(profile_raw) if profile_raw else None

        return DatasetDetail(
            id=row_dict["id"],
            name=row_dict["name"],
            num_rows=row_dict.get("num_rows"),
            num_cols=row_dict.get("num_cols"),
            target_col=row_dict.get("target_col"),
            task_type=row_dict.get("task_type"),
            created_at=row_dict.get("created_at"),
            columns=[ColumnInfo(**c) for c in columns],
            profile=profile,
            file_path=row_dict.get("file_path"),
            db_uri=row_dict.get("db_uri"),
        )
    finally:
        await db.close()
