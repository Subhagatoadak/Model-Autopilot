from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from backend.database import get_db


async def load_dataset(dataset_id: str) -> tuple[pd.DataFrame, dict]:
    """Load a dataset from the database and return (DataFrame, metadata)."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        row = await cursor.fetchone()
        if not row:
            raise ValueError(f"Dataset {dataset_id} not found")

        row_dict = dict(row)
        file_path = row_dict.get("file_path")
        if not file_path or not Path(file_path).exists():
            raise ValueError(f"Dataset file not found: {file_path}")

        df = pd.read_csv(file_path)
        metadata = {
            "id": row_dict["id"],
            "name": row_dict["name"],
            "target_col": row_dict.get("target_col"),
            "task_type": row_dict.get("task_type"),
            "columns": json.loads(row_dict.get("columns") or "[]"),
        }
        return df, metadata
    finally:
        await db.close()


async def get_dataset_columns(dataset_id: str) -> list[dict]:
    """Get column info for a dataset."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT columns FROM datasets WHERE id = ?", (dataset_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return []
        return json.loads(dict(row).get("columns") or "[]")
    finally:
        await db.close()


async def get_dataset_preview(dataset_id: str, n_rows: int = 10) -> list[dict]:
    """Get first N rows of a dataset as a list of dicts."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT file_path FROM datasets WHERE id = ?", (dataset_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return []
        file_path = dict(row).get("file_path")
        if not file_path or not Path(file_path).exists():
            return []
        df = pd.read_csv(file_path, nrows=n_rows)
        return df.to_dict(orient="records")
    finally:
        await db.close()


def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    feature_cols: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Prepare data for model training.

    Args:
        feature_cols: If provided, only use these columns as features.
                      If empty or None, use all non-target columns.

    Returns:
        (X_train, X_test, y_train, y_test, feature_names)
    """
    # Separate features and target
    if feature_cols:
        # Use only selected feature columns
        available = [c for c in feature_cols if c in df.columns and c != target_col]
        if not available:
            available = [c for c in df.columns if c != target_col]
    else:
        available = [c for c in df.columns if c != target_col]

    X = df[available].copy()
    y = df[target_col].copy()

    # Handle categorical features: label encode
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Handle missing values: fill with median for numeric
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    # Convert target for classification if needed
    if task_type == "classification" and y.dtype == "object":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)))

    # Handle missing target values
    mask = ~y.isnull()
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, list(available)


async def save_dataset_version(
    dataset_id: str,
    experiment_id: str,
    version: int,
    parent_version: str | None,
    transforms: list[dict],
    num_features: int,
    feature_names: list[str],
) -> str:
    """Save a dataset version to track transform lineage."""
    import uuid

    version_id = str(uuid.uuid4())
    db = await get_db()
    try:
        await db.execute(
            """INSERT INTO dataset_versions
               (id, dataset_id, experiment_id, version, parent_version, transforms, num_features, feature_names)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                version_id,
                dataset_id,
                experiment_id,
                version,
                parent_version,
                json.dumps(transforms),
                num_features,
                json.dumps(feature_names),
            ),
        )
        await db.commit()
    finally:
        await db.close()
    return version_id


async def get_dataset_versions(experiment_id: str) -> list[dict]:
    """Get all dataset versions for an experiment."""
    db = await get_db()
    try:
        cursor = await db.execute(
            """SELECT * FROM dataset_versions
               WHERE experiment_id = ?
               ORDER BY version""",
            (experiment_id,),
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            rd = dict(row)
            rd["transforms"] = json.loads(rd.get("transforms") or "[]")
            rd["feature_names"] = json.loads(rd.get("feature_names") or "[]")
            results.append(rd)
        return results
    finally:
        await db.close()
