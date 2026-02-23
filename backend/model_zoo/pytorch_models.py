from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from backend.model_zoo.base import ModelWrapper
from backend.model_zoo.registry import register_model


class _MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], task_type: str):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        if task_type == "classification" and output_dim > 1:
            layers.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_model("pytorch_mlp", task_types=["classification", "regression"])
class PyTorchMLPWrapper(ModelWrapper):
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
        return None  # Built dynamically in fit() based on input shape

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> PyTorchMLPWrapper:
        input_dim = X.shape[1]

        if self.task_type == "classification":
            self._classes = np.unique(y)
            output_dim = len(self._classes)
            # Map labels to 0..N-1
            label_map = {v: i for i, v in enumerate(self._classes)}
            y_encoded = np.array([label_map[v] for v in y])
            y_tensor = torch.LongTensor(y_encoded)
            criterion = nn.CrossEntropyLoss()
        else:
            output_dim = 1
            y_tensor = torch.FloatTensor(y).unsqueeze(1)
            criterion = nn.MSELoss()

        hidden_dims = self.params.get("hidden_dims", [128, 64])
        self._model = _MLP(input_dim, output_dim, hidden_dims, self.task_type)

        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.params.get("batch_size", 32), shuffle=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.params.get("learning_rate", 0.001))

        self._model.train()
        for epoch in range(self.params.get("epochs", 50)):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                output = self._model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            output = self._model(X_tensor)
            if self.task_type == "classification":
                indices = output.argmax(dim=1).numpy()
                return np.array([self._classes[i] for i in indices])
            return output.squeeze().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self.task_type != "classification":
            return None
        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            output = self._model(X_tensor)
            return output.numpy()
