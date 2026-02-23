from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from backend.model_zoo.base import ModelWrapper
from backend.model_zoo.registry import register_model


# ──────────────────────────────────────────
# Ensemble Models
# ──────────────────────────────────────────

@register_model("random_forest", task_types=["classification", "regression"])
class RandomForestWrapper(ModelWrapper):
    default_params = {"n_estimators": 100, "max_depth": None, "random_state": 42}

    def _build_model(self):
        if self.task_type == "classification":
            return RandomForestClassifier(**self.params)
        return RandomForestRegressor(**self.params)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        if self.task_type == "classification":
            return self._model.predict_proba(X)
        return None


@register_model("extra_trees", task_types=["classification", "regression"])
class ExtraTreesWrapper(ModelWrapper):
    default_params = {"n_estimators": 100, "max_depth": None, "random_state": 42}

    def _build_model(self):
        if self.task_type == "classification":
            return ExtraTreesClassifier(**self.params)
        return ExtraTreesRegressor(**self.params)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        if self.task_type == "classification":
            return self._model.predict_proba(X)
        return None


@register_model("decision_tree", task_types=["classification", "regression"])
class DecisionTreeWrapper(ModelWrapper):
    default_params = {"max_depth": None, "random_state": 42}

    def _build_model(self):
        if self.task_type == "classification":
            return DecisionTreeClassifier(**self.params)
        return DecisionTreeRegressor(**self.params)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        if self.task_type == "classification":
            return self._model.predict_proba(X)
        return None


@register_model("gradient_boosting", task_types=["classification", "regression"])
class GradientBoostingWrapper(ModelWrapper):
    default_params = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42}

    def _build_model(self):
        if self.task_type == "classification":
            return GradientBoostingClassifier(**self.params)
        return GradientBoostingRegressor(**self.params)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        if self.task_type == "classification":
            return self._model.predict_proba(X)
        return None


@register_model("adaboost", task_types=["classification", "regression"])
class AdaBoostWrapper(ModelWrapper):
    default_params = {"n_estimators": 50, "learning_rate": 1.0, "random_state": 42}

    def _build_model(self):
        if self.task_type == "classification":
            return AdaBoostClassifier(**self.params)
        return AdaBoostRegressor(**self.params)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        if self.task_type == "classification":
            return self._model.predict_proba(X)
        return None


@register_model("bagging", task_types=["classification", "regression"])
class BaggingWrapper(ModelWrapper):
    default_params = {"n_estimators": 10, "random_state": 42}

    def _build_model(self):
        if self.task_type == "classification":
            return BaggingClassifier(**self.params)
        return BaggingRegressor(**self.params)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        if self.task_type == "classification":
            return self._model.predict_proba(X)
        return None


# ──────────────────────────────────────────
# Linear Models
# ──────────────────────────────────────────

@register_model("logistic_regression", task_types=["classification"])
class LogisticRegressionWrapper(ModelWrapper):
    default_params = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs", "random_state": 42}

    def _build_model(self):
        return LogisticRegression(**self.params)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


@register_model("linear_regression", task_types=["regression"])
class LinearRegressionWrapper(ModelWrapper):
    default_params = {}

    def _build_model(self):
        return LinearRegression(**self.params)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)


@register_model("ridge", task_types=["regression"])
class RidgeWrapper(ModelWrapper):
    default_params = {"alpha": 1.0}

    def _build_model(self):
        return Ridge(**self.params)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)


@register_model("lasso", task_types=["regression"])
class LassoWrapper(ModelWrapper):
    default_params = {"alpha": 1.0, "max_iter": 5000}

    def _build_model(self):
        return Lasso(**self.params)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)


@register_model("elastic_net", task_types=["regression"])
class ElasticNetWrapper(ModelWrapper):
    default_params = {"alpha": 1.0, "l1_ratio": 0.5, "max_iter": 5000}

    def _build_model(self):
        return ElasticNet(**self.params)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)


@register_model("sgd", task_types=["classification", "regression"])
class SGDWrapper(ModelWrapper):
    default_params = {"max_iter": 1000, "random_state": 42}

    def _build_model(self):
        if self.task_type == "classification":
            params = dict(self.params)
            params.setdefault("loss", "modified_huber")  # supports predict_proba
            return SGDClassifier(**params)
        return SGDRegressor(**self.params)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        if self.task_type == "classification" and hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        return None


# ──────────────────────────────────────────
# Bayesian Linear Models
# ──────────────────────────────────────────

@register_model("bayesian_ridge", task_types=["regression"])
class BayesianRidgeWrapper(ModelWrapper):
    default_params = {"max_iter": 300, "alpha_1": 1e-6, "alpha_2": 1e-6, "lambda_1": 1e-6, "lambda_2": 1e-6}

    def _build_model(self):
        return BayesianRidge(**self.params)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)


@register_model("ard_regression", task_types=["regression"])
class ARDRegressionWrapper(ModelWrapper):
    default_params = {"max_iter": 300}

    def _build_model(self):
        return ARDRegression(**self.params)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)


# ──────────────────────────────────────────
# Multivariate / Partial Least Squares
# ──────────────────────────────────────────

@register_model("pls_regression", task_types=["regression"])
class PLSRegressionWrapper(ModelWrapper):
    default_params = {"n_components": 2, "max_iter": 500}

    def _build_model(self):
        return PLSRegression(**self.params)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X).ravel()


# ──────────────────────────────────────────
# Support Vector Machines
# ──────────────────────────────────────────

@register_model("svm", task_types=["classification", "regression"])
class SVMWrapper(ModelWrapper):
    default_params = {"C": 1.0, "kernel": "rbf", "random_state": 42}

    def _build_model(self):
        params = dict(self.params)
        if self.task_type == "classification":
            return SVC(probability=True, **params)
        params.pop("random_state", None)
        return SVR(**params)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        if self.task_type == "classification":
            return self._model.predict_proba(X)
        return None


# ──────────────────────────────────────────
# K-Nearest Neighbors
# ──────────────────────────────────────────

@register_model("knn", task_types=["classification", "regression"])
class KNNWrapper(ModelWrapper):
    default_params = {"n_neighbors": 5}

    def _build_model(self):
        if self.task_type == "classification":
            return KNeighborsClassifier(**self.params)
        return KNeighborsRegressor(**self.params)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        if self.task_type == "classification":
            return self._model.predict_proba(X)
        return None


# ──────────────────────────────────────────
# Naive Bayes
# ──────────────────────────────────────────

@register_model("naive_bayes", task_types=["classification"])
class NaiveBayesWrapper(ModelWrapper):
    default_params = {}

    def _build_model(self):
        return GaussianNB(**self.params)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


# ──────────────────────────────────────────
# Non-linear Models: Gaussian Process
# ──────────────────────────────────────────

@register_model("gaussian_process", task_types=["classification", "regression"])
class GaussianProcessWrapper(ModelWrapper):
    default_params = {"random_state": 42}

    def _build_model(self):
        params = dict(self.params)
        if self.task_type == "classification":
            return GaussianProcessClassifier(**params)
        params.pop("random_state", None)
        return GaussianProcessRegressor(**params)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        if self.task_type == "classification":
            return self._model.predict_proba(X)
        return None


# ──────────────────────────────────────────
# Non-linear Models: Kernel Ridge Regression
# ──────────────────────────────────────────

@register_model("kernel_ridge", task_types=["regression"])
class KernelRidgeWrapper(ModelWrapper):
    default_params = {"alpha": 1.0, "kernel": "rbf", "gamma": None}

    def _build_model(self):
        return KernelRidge(**self.params)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)


# ──────────────────────────────────────────
# Non-linear Models: MLP Neural Network
# ──────────────────────────────────────────

@register_model("mlp", task_types=["classification", "regression"])
class MLPWrapper(ModelWrapper):
    default_params = {
        "hidden_layer_sizes": (100,),
        "activation": "relu",
        "max_iter": 500,
        "random_state": 42,
    }

    def _build_model(self):
        if self.task_type == "classification":
            return MLPClassifier(**self.params)
        return MLPRegressor(**self.params)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        if self.task_type == "classification":
            return self._model.predict_proba(X)
        return None
