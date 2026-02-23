from __future__ import annotations

from pydantic import BaseModel


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    null_pct: float = 0.0
    unique_count: int = 0
    sample_values: list[str] = []


class DatasetProfile(BaseModel):
    num_rows: int
    num_cols: int
    columns: list[ColumnInfo]
    target_col: str | None = None
    task_type: str | None = None
    summary: str = ""


class DatasetCreate(BaseModel):
    name: str
    target_col: str | None = None
    task_type: str | None = None


class DatasetDBConnect(BaseModel):
    name: str
    db_uri: str
    query: str
    target_col: str | None = None
    task_type: str | None = None


class DatasetOut(BaseModel):
    id: str
    name: str
    num_rows: int | None = None
    num_cols: int | None = None
    target_col: str | None = None
    task_type: str | None = None
    created_at: str | None = None


class DatasetDetail(DatasetOut):
    columns: list[ColumnInfo] = []
    profile: DatasetProfile | None = None
    file_path: str | None = None
    db_uri: str | None = None
