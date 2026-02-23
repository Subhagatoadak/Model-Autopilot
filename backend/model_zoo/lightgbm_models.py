from __future__ import annotations

from typing import Any

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor

from backend.model_zoo.base import ModelWrapper
from backend.model_zoo.registry import register_model


@register_model("lightgbm", task_types=["classification", "regression"])
class LightGBMWrapper(ModelWrapper):
    default_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": -1,
        "random_state": 42,
        "verbosity": -1,
    }

    def _build_model(self):
        if self.task_type == "classification":
            return LGBMClassifier(**self.params)
        return LGBMRegressor(**self.params)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        if self.task_type == "classification":
            return self._model.predict_proba(X)
        return None
