"""Feature engineering service that applies data transformations between iterations.

The recommendation agent suggests transformations as structured actions.
This service executes those actions on the training/test data.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)

logger = logging.getLogger(__name__)

# All supported transformation actions
AVAILABLE_TRANSFORMS = [
    "standard_scale",       # StandardScaler
    "minmax_scale",         # MinMaxScaler
    "robust_scale",         # RobustScaler
    "log_transform",        # log1p on positive-skewed features
    "sqrt_transform",       # sqrt on positive features
    "polynomial_features",  # degree-2 polynomial + interactions
    "interaction_features", # pairwise interaction only (no polynomial)
    "remove_features",      # drop specific feature columns
    "square_features",      # add x^2 for selected features
    "reciprocal_features",  # add 1/x for non-zero features
]


class FeatureEngineer:
    """Applies a sequence of data transformations.

    Tracks which transformations have been applied so the recommendation
    agent can see what's already been tried.
    """

    def __init__(self, feature_names: list[str]):
        self.original_feature_names = list(feature_names)
        self.current_feature_names = list(feature_names)
        self.applied_transforms: list[dict[str, Any]] = []
        self._scaler: Any = None

    def apply(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        action: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Apply a single transformation action.

        Args:
            X_train: Training features
            X_test: Test features
            action: Dict with 'type' key and optional parameters
                e.g. {"type": "standard_scale"}
                     {"type": "remove_features", "columns": [0, 2]}
                     {"type": "polynomial_features", "degree": 2}
                     {"type": "log_transform", "columns": [1, 3]}

        Returns:
            (X_train_transformed, X_test_transformed, updated_feature_names)
        """
        transform_type = action.get("type", "")
        logger.info(f"Applying transform: {transform_type} with params: {action}")

        try:
            if transform_type == "standard_scale":
                X_train, X_test = self._scale(X_train, X_test, StandardScaler())

            elif transform_type == "minmax_scale":
                X_train, X_test = self._scale(X_train, X_test, MinMaxScaler())

            elif transform_type == "robust_scale":
                X_train, X_test = self._scale(X_train, X_test, RobustScaler())

            elif transform_type == "log_transform":
                cols = action.get("columns", None)
                X_train, X_test = self._log_transform(X_train, X_test, cols)

            elif transform_type == "sqrt_transform":
                cols = action.get("columns", None)
                X_train, X_test = self._sqrt_transform(X_train, X_test, cols)

            elif transform_type == "polynomial_features":
                degree = action.get("degree", 2)
                X_train, X_test = self._polynomial(X_train, X_test, degree)

            elif transform_type == "interaction_features":
                X_train, X_test = self._interactions(X_train, X_test)

            elif transform_type == "remove_features":
                columns = action.get("columns", [])
                X_train, X_test = self._remove_features(X_train, X_test, columns)

            elif transform_type == "square_features":
                cols = action.get("columns", None)
                X_train, X_test = self._square_features(X_train, X_test, cols)

            elif transform_type == "reciprocal_features":
                cols = action.get("columns", None)
                X_train, X_test = self._reciprocal_features(X_train, X_test, cols)

            else:
                logger.warning(f"Unknown transform type: {transform_type}")
                return X_train, X_test, self.current_feature_names

            self.applied_transforms.append(action)

        except Exception as e:
            logger.error(f"Transform {transform_type} failed: {e}")
            # Return data unchanged on failure

        return X_train, X_test, self.current_feature_names

    def _scale(self, X_train: np.ndarray, X_test: np.ndarray, scaler: Any) -> tuple[np.ndarray, np.ndarray]:
        self._scaler = scaler
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test

    def _log_transform(
        self, X_train: np.ndarray, X_test: np.ndarray, cols: list[int] | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if cols is None:
            # Apply to all columns where all values >= 0
            cols = [i for i in range(X_train.shape[1]) if np.all(X_train[:, i] >= 0)]
        for c in cols:
            if c < X_train.shape[1]:
                X_train[:, c] = np.log1p(np.maximum(X_train[:, c], 0))
                X_test[:, c] = np.log1p(np.maximum(X_test[:, c], 0))
        return X_train, X_test

    def _sqrt_transform(
        self, X_train: np.ndarray, X_test: np.ndarray, cols: list[int] | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if cols is None:
            cols = [i for i in range(X_train.shape[1]) if np.all(X_train[:, i] >= 0)]
        for c in cols:
            if c < X_train.shape[1]:
                X_train[:, c] = np.sqrt(np.maximum(X_train[:, c], 0))
                X_test[:, c] = np.sqrt(np.maximum(X_test[:, c], 0))
        return X_train, X_test

    def _polynomial(
        self, X_train: np.ndarray, X_test: np.ndarray, degree: int
    ) -> tuple[np.ndarray, np.ndarray]:
        # Limit features to avoid explosion
        max_feats = min(X_train.shape[1], 10)
        X_sub_train = X_train[:, :max_feats]
        X_sub_test = X_test[:, :max_feats]

        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
        X_poly_train = poly.fit_transform(X_sub_train)
        X_poly_test = poly.transform(X_sub_test)

        # Append new polynomial features (skip original columns which are already there)
        n_orig = X_sub_train.shape[1]
        new_train = X_poly_train[:, n_orig:]
        new_test = X_poly_test[:, n_orig:]

        X_train = np.hstack([X_train, new_train])
        X_test = np.hstack([X_test, new_test])

        n_new = new_train.shape[1]
        self.current_feature_names = self.current_feature_names + [f"poly_{i}" for i in range(n_new)]
        return X_train, X_test

    def _interactions(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        max_feats = min(X_train.shape[1], 8)
        X_sub_train = X_train[:, :max_feats]
        X_sub_test = X_test[:, :max_feats]

        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        X_int_train = poly.fit_transform(X_sub_train)
        X_int_test = poly.transform(X_sub_test)

        n_orig = X_sub_train.shape[1]
        new_train = X_int_train[:, n_orig:]
        new_test = X_int_test[:, n_orig:]

        X_train = np.hstack([X_train, new_train])
        X_test = np.hstack([X_test, new_test])

        n_new = new_train.shape[1]
        self.current_feature_names = self.current_feature_names + [f"interact_{i}" for i in range(n_new)]
        return X_train, X_test

    def _remove_features(
        self, X_train: np.ndarray, X_test: np.ndarray, columns: list[int | str]
    ) -> tuple[np.ndarray, np.ndarray]:
        # Resolve column names to indices
        indices: list[int] = []
        for c in columns:
            if isinstance(c, int) and c < X_train.shape[1]:
                indices.append(c)
            elif isinstance(c, str) and c in self.current_feature_names:
                indices.append(self.current_feature_names.index(c))

        if not indices:
            return X_train, X_test

        keep = [i for i in range(X_train.shape[1]) if i not in indices]
        if not keep:
            return X_train, X_test  # Don't remove all features

        X_train = X_train[:, keep]
        X_test = X_test[:, keep]
        self.current_feature_names = [self.current_feature_names[i] for i in keep]
        return X_train, X_test

    def _square_features(
        self, X_train: np.ndarray, X_test: np.ndarray, cols: list[int] | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if cols is None:
            cols = list(range(min(X_train.shape[1], 5)))
        new_train_cols = []
        new_test_cols = []
        new_names = []
        for c in cols:
            if c < X_train.shape[1]:
                new_train_cols.append(X_train[:, c] ** 2)
                new_test_cols.append(X_test[:, c] ** 2)
                name = self.current_feature_names[c] if c < len(self.current_feature_names) else f"f{c}"
                new_names.append(f"{name}_sq")
        if new_train_cols:
            X_train = np.hstack([X_train, np.column_stack(new_train_cols)])
            X_test = np.hstack([X_test, np.column_stack(new_test_cols)])
            self.current_feature_names = self.current_feature_names + new_names
        return X_train, X_test

    def _reciprocal_features(
        self, X_train: np.ndarray, X_test: np.ndarray, cols: list[int] | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if cols is None:
            # Only apply to columns with no zeros
            cols = [i for i in range(min(X_train.shape[1], 5))
                    if np.all(X_train[:, i] != 0)]
        new_train_cols = []
        new_test_cols = []
        new_names = []
        for c in cols:
            if c < X_train.shape[1] and np.all(X_train[:, c] != 0):
                new_train_cols.append(1.0 / X_train[:, c])
                new_test_cols.append(1.0 / np.where(X_test[:, c] != 0, X_test[:, c], 1.0))
                name = self.current_feature_names[c] if c < len(self.current_feature_names) else f"f{c}"
                new_names.append(f"{name}_inv")
        if new_train_cols:
            X_train = np.hstack([X_train, np.column_stack(new_train_cols)])
            X_test = np.hstack([X_test, np.column_stack(new_test_cols)])
            self.current_feature_names = self.current_feature_names + new_names
        return X_train, X_test

    def get_state_summary(self) -> str:
        """Return a summary of current feature state for LLM context."""
        lines = [
            f"Features: {len(self.current_feature_names)} (originally {len(self.original_feature_names)})",
            f"Applied transforms: {len(self.applied_transforms)}",
        ]
        for t in self.applied_transforms:
            lines.append(f"  - {t.get('type', 'unknown')}")
        return "\n".join(lines)
