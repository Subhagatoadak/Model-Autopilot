from __future__ import annotations

from typing import Any

import numpy as np
from xgboost import XGBClassifier, XGBRegressor

from backend.model_zoo.base import ModelWrapper
from backend.model_zoo.registry import register_model


@register_model("xgboost", task_types=["classification", "regression"])
class XGBoostWrapper(ModelWrapper):
    default_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "random_state": 42,
        "verbosity": 0,
    }

    def _build_model(self):
        if self.task_type == "classification":
            return XGBClassifier(**self.params, use_label_encoder=False, eval_metric="logloss")
        return XGBRegressor(**self.params)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        if self.task_type == "classification":
            return self._model.predict_proba(X)
        return None
