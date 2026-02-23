import numpy as np
import pytest
from backend.services.metrics_service import (
    compute_metrics,
    default_metrics,
    is_higher_better,
    primary_metric,
)


def test_default_metrics_classification():
    metrics = default_metrics("classification")
    assert "accuracy" in metrics
    assert "f1" in metrics


def test_default_metrics_regression():
    metrics = default_metrics("regression")
    assert "mse" in metrics
    assert "r2" in metrics


def test_primary_metric():
    assert primary_metric("classification") == "f1"
    assert primary_metric("regression") == "r2"


def test_is_higher_better():
    assert is_higher_better("accuracy") is True
    assert is_higher_better("f1") is True
    assert is_higher_better("mse") is False


def test_classification_metrics():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    results = compute_metrics(y_true, y_pred, "classification", ["accuracy", "f1"])
    assert "accuracy" in results
    assert "f1" in results
    assert results["accuracy"] == 0.8


def test_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1])
    results = compute_metrics(y_true, y_pred, "regression", ["mse", "r2", "mae"])
    assert "mse" in results
    assert "r2" in results
    assert results["r2"] > 0.9
