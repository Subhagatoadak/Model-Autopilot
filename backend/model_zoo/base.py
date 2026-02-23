from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


class ModelWrapper(ABC):
    """Abstract base class for all models in the zoo."""

    name: str = ""
    task_type: Literal["classification", "regression"] = "classification"
    default_params: dict[str, Any] = {}

    def __init__(self, task_type: str = "classification", **params: Any):
        self.task_type = task_type
        self.params = {**self.default_params, **params}
        self._model = self._safe_build_model()

    def _safe_build_model(self) -> Any:
        """Build the model, filtering out invalid parameters if needed."""
        try:
            return self._build_model()
        except TypeError as e:
            # Filter params to only those accepted by the underlying model
            logger.warning(f"Build failed with params, retrying with filtered params: {e}")
            self._filter_valid_params()
            return self._build_model()

    def _filter_valid_params(self) -> None:
        """Remove params not accepted by the underlying sklearn-style model."""
        # Build with just defaults to inspect the model class
        saved = dict(self.params)
        self.params = dict(self.default_params)
        try:
            test_model = self._build_model()
            # Get valid params from the model's __init__ signature
            sig = inspect.signature(test_model.__class__.__init__)
            valid_keys = set(sig.parameters.keys()) - {"self"}
            # If the model has **kwargs, allow everything
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            if not has_var_keyword:
                filtered = {k: v for k, v in saved.items() if k in valid_keys}
                removed = set(saved.keys()) - set(filtered.keys())
                if removed:
                    logger.info(f"Filtered out invalid params: {removed}")
                self.params = filtered
            else:
                self.params = saved
        except Exception:
            # Fall back to default params only
            self.params = dict(self.default_params)

    @abstractmethod
    def _build_model(self) -> Any:
        """Construct the underlying model object."""
        ...

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> ModelWrapper:
        """Train the model on the given data."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Generate probability predictions (classification only). Returns None if not supported."""
        return None

    def get_params(self) -> dict[str, Any]:
        """Return current hyperparameters."""
        return dict(self.params)

    def set_params(self, **params: Any) -> None:
        """Update hyperparameters and rebuild the model."""
        self.params.update(params)
        self._model = self._build_model()

    def get_info(self) -> dict[str, Any]:
        """Return model metadata for the zoo catalog."""
        return {
            "name": self.name,
            "task_types": self._supported_task_types(),
            "default_params": self.default_params,
            "current_params": self.params,
        }

    @classmethod
    def _supported_task_types(cls) -> list[str]:
        return ["classification", "regression"]
