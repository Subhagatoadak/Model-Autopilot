from __future__ import annotations

import os
from typing import Any

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from backend.model_zoo.base import ModelWrapper
from backend.model_zoo.registry import register_model


@register_model("keras_mlp", task_types=["classification", "regression"])
class KerasMLPWrapper(ModelWrapper):
    default_params = {
        "hidden_dims": [128, 64],
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
    }

    def __init__(self, task_type: str = "classification", **params: Any):
        self.task_type = task_type
        self.params = {**self.default_params, **params}
        self._model = None
        self._classes = None

    def _build_model(self):
        return None  # Built dynamically in fit()

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> KerasMLPWrapper:
        import tensorflow as tf

        input_dim = X.shape[1]
        hidden_dims = self.params.get("hidden_dims", [128, 64])

        layers = [tf.keras.layers.Input(shape=(input_dim,))]
        for h in hidden_dims:
            layers.append(tf.keras.layers.Dense(h, activation="relu"))
            layers.append(tf.keras.layers.Dropout(0.2))

        if self.task_type == "classification":
            self._classes = np.unique(y)
            output_dim = len(self._classes)
            label_map = {v: i for i, v in enumerate(self._classes)}
            y_encoded = np.array([label_map[v] for v in y])

            layers.append(tf.keras.layers.Dense(output_dim, activation="softmax"))
            model = tf.keras.Sequential(layers)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.params.get("learning_rate", 0.001)),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            model.fit(
                X, y_encoded,
                epochs=self.params.get("epochs", 50),
                batch_size=self.params.get("batch_size", 32),
                verbose=0,
            )
        else:
            layers.append(tf.keras.layers.Dense(1))
            model = tf.keras.Sequential(layers)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.params.get("learning_rate", 0.001)),
                loss="mse",
            )
            model.fit(
                X, y,
                epochs=self.params.get("epochs", 50),
                batch_size=self.params.get("batch_size", 32),
                verbose=0,
            )

        self._model = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        output = self._model.predict(X, verbose=0)
        if self.task_type == "classification":
            indices = output.argmax(axis=1)
            return np.array([self._classes[i] for i in indices])
        return output.squeeze()

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self.task_type != "classification":
            return None
        return self._model.predict(X, verbose=0)
