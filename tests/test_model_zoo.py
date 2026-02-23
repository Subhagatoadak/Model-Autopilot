import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

# Import to trigger registration
import backend.model_zoo.sklearn_models
import backend.model_zoo.xgboost_models
import backend.model_zoo.lightgbm_models

from backend.model_zoo.registry import get_model, list_models, available_model_names


@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    return X[:80], X[80:], y[:80], y[80:]


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    return X[:80], X[80:], y[:80], y[80:]


def test_list_models():
    models = list_models()
    assert len(models) > 0
    names = [m["name"] for m in models]
    assert "random_forest" in names
    assert "xgboost" in names
    assert "lightgbm" in names


def test_list_models_filtered():
    classification_models = list_models(task_type="classification")
    regression_models = list_models(task_type="regression")
    assert len(classification_models) > 0
    assert len(regression_models) > 0


def test_available_model_names():
    names = available_model_names("classification")
    assert "logistic_regression" in names
    assert "linear_regression" not in names


def test_random_forest_classification(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    model = get_model("random_forest", "classification", n_estimators=10)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (20,)
    proba = model.predict_proba(X_test)
    assert proba is not None
    assert proba.shape[0] == 20


def test_random_forest_regression(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    model = get_model("random_forest", "regression", n_estimators=10)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (20,)


def test_xgboost_classification(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    model = get_model("xgboost", "classification", n_estimators=10)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (20,)


def test_lightgbm_regression(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    model = get_model("lightgbm", "regression", n_estimators=10)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (20,)


def test_svm_classification(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    model = get_model("svm", "classification")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (20,)


def test_logistic_regression(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    model = get_model("logistic_regression", "classification")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (20,)


def test_knn_classification(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    model = get_model("knn", "classification", n_neighbors=3)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (20,)


def test_get_params():
    model = get_model("random_forest", "classification", n_estimators=50)
    params = model.get_params()
    assert params["n_estimators"] == 50


def test_invalid_model():
    with pytest.raises(ValueError, match="not found"):
        get_model("nonexistent_model", "classification")


def test_invalid_task_type():
    with pytest.raises(ValueError, match="does not support"):
        get_model("logistic_regression", "regression")
