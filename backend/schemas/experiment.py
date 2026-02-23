from __future__ import annotations

from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    dataset_id: str
    target_col: str
    feature_cols: list[str] = []  # empty = use all non-target columns
    task_type: str | None = None  # auto-detect if None
    primary_metric: str | None = None  # auto-detect if None
    secondary_metrics: list[str] = []
    strategy: str = "balanced"  # explore | exploit | balanced
    epsilon: float = 0.3
    max_iterations: int = 10
    selected_models: list[str] = []  # empty = use all
    test_size: float = 0.2
    cross_validation: bool = False
    cv_folds: int = 5
    multi_objective: bool = False
    objective_weights: dict[str, float] = Field(default_factory=dict)  # e.g. {"f1": 0.7, "train_time_s": 0.3}


class ExperimentOut(BaseModel):
    id: str
    dataset_id: str
    status: str
    strategy: str = "balanced"
    epsilon: float = 0.3
    created_at: str | None = None
    finished_at: str | None = None


class ExperimentRunOut(BaseModel):
    id: str
    experiment_id: str
    iteration: int
    model_name: str
    hyperparameters: dict = {}
    metrics: dict[str, float] = {}
    train_time_s: float | None = None
    ucb_score: float | None = None
    created_at: str | None = None


class ExperimentDetail(ExperimentOut):
    config: ExperimentConfig | None = None
    runs: list[ExperimentRunOut] = []
    best_metrics: dict[str, float] = {}


class ExperimentComparison(BaseModel):
    """Side-by-side comparison of two experiments."""
    experiment_1: ExperimentDetail
    experiment_2: ExperimentDetail
