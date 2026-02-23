from __future__ import annotations

import os
from pathlib import Path

import aiosqlite

from backend.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    file_path   TEXT,
    db_uri      TEXT,
    db_query    TEXT,
    num_rows    INTEGER,
    num_cols    INTEGER,
    columns     TEXT,
    target_col  TEXT,
    task_type   TEXT,
    profile     TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS experiments (
    id          TEXT PRIMARY KEY,
    dataset_id  TEXT NOT NULL REFERENCES datasets(id),
    status      TEXT DEFAULT 'pending',
    config      TEXT,
    strategy    TEXT DEFAULT 'balanced',
    epsilon     REAL DEFAULT 0.3,
    created_at  TEXT DEFAULT (datetime('now')),
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    id                  TEXT PRIMARY KEY,
    experiment_id       TEXT NOT NULL REFERENCES experiments(id),
    iteration           INTEGER NOT NULL,
    model_name          TEXT NOT NULL,
    hyperparameters     TEXT,
    metrics             TEXT,
    train_time_s        REAL,
    feature_config      TEXT,
    ucb_score           REAL,
    model_artifact_path TEXT,
    created_at          TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS agent_traces (
    id              TEXT PRIMARY KEY,
    experiment_id   TEXT NOT NULL REFERENCES experiments(id),
    iteration       INTEGER,
    agent_name      TEXT NOT NULL,
    step_type       TEXT,
    content         TEXT,
    parent_trace_id TEXT,
    depth           INTEGER DEFAULT 0,
    score           REAL,
    selected        INTEGER DEFAULT 0,
    metadata        TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS dataset_versions (
    id              TEXT PRIMARY KEY,
    dataset_id      TEXT NOT NULL REFERENCES datasets(id),
    experiment_id   TEXT NOT NULL REFERENCES experiments(id),
    version         INTEGER NOT NULL,
    parent_version  TEXT REFERENCES dataset_versions(id),
    transforms      TEXT,
    num_features    INTEGER,
    feature_names   TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_runs_experiment ON experiment_runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_traces_experiment ON agent_traces(experiment_id);
CREATE INDEX IF NOT EXISTS idx_traces_parent ON agent_traces(parent_trace_id);
CREATE INDEX IF NOT EXISTS idx_dv_experiment ON dataset_versions(experiment_id);
"""


def _db_path() -> str:
    """Resolve the database path relative to the project root."""
    root = Path(__file__).resolve().parent.parent
    path = root / settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


async def get_db() -> aiosqlite.Connection:
    """Open a connection to the SQLite database."""
    db = await aiosqlite.connect(_db_path())
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db() -> None:
    """Create tables if they don't exist."""
    db = await get_db()
    try:
        await db.executescript(_SCHEMA)
        await db.commit()
    finally:
        await db.close()
