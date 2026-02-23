from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
REGRESSION_METRICS = ["mse", "rmse", "mae", "r2"]


def default_metrics(task_type: str) -> list[str]:
    """Return default metrics for a task type."""
    if task_type == "classification":
        return list(CLASSIFICATION_METRICS)
    return list(REGRESSION_METRICS)


def primary_metric(task_type: str) -> str:
    """Return the default primary metric for a task type."""
    if task_type == "classification":
        return "f1"
    return "r2"


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    metric_names: list[str] | None = None,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute a set of metrics for predictions vs ground truth."""
    if metric_names is None:
        metric_names = default_metrics(task_type)

    results: dict[str, float] = {}

    for name in metric_names:
        try:
            if name == "accuracy":
                results[name] = float(accuracy_score(y_true, y_pred))
            elif name == "precision":
                results[name] = float(
                    precision_score(y_true, y_pred, average="weighted", zero_division=0)
                )
            elif name == "recall":
                results[name] = float(
                    recall_score(y_true, y_pred, average="weighted", zero_division=0)
                )
            elif name == "f1":
                results[name] = float(
                    f1_score(y_true, y_pred, average="weighted", zero_division=0)
                )
            elif name == "roc_auc":
                if y_proba is not None:
                    try:
                        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                            results[name] = float(
                                roc_auc_score(y_true, y_proba[:, 1])
                            )
                        elif y_proba.ndim == 2:
                            results[name] = float(
                                roc_auc_score(
                                    y_true, y_proba, multi_class="ovr", average="weighted"
                                )
                            )
                    except ValueError:
                        results[name] = 0.0
                else:
                    results[name] = 0.0
            elif name == "mse":
                results[name] = float(mean_squared_error(y_true, y_pred))
            elif name == "rmse":
                results[name] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            elif name == "mae":
                results[name] = float(mean_absolute_error(y_true, y_pred))
            elif name == "r2":
                results[name] = float(r2_score(y_true, y_pred))
        except Exception:
            results[name] = 0.0

    return results


def is_higher_better(metric_name: str) -> bool:
    """Return True if higher values are better for the given metric."""
    return metric_name in {"accuracy", "precision", "recall", "f1", "roc_auc", "r2"}
