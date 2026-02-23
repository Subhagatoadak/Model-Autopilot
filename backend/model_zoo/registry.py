from __future__ import annotations

from typing import Any

from backend.model_zoo.base import ModelWrapper

# Global registry: model_name -> (class, supported_task_types)
MODEL_REGISTRY: dict[str, tuple[type[ModelWrapper], list[str]]] = {}


def register_model(name: str, task_types: list[str] | None = None):
    """Decorator to register a model class in the zoo."""
    if task_types is None:
        task_types = ["classification", "regression"]

    def decorator(cls: type[ModelWrapper]) -> type[ModelWrapper]:
        cls.name = name
        MODEL_REGISTRY[name] = (cls, task_types)
        return cls

    return decorator


def get_model(name: str, task_type: str = "classification", **params: Any) -> ModelWrapper:
    """Instantiate a model by name from the registry."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}")

    cls, supported = MODEL_REGISTRY[name]
    if task_type not in supported:
        raise ValueError(f"Model '{name}' does not support task type '{task_type}'. Supported: {supported}")

    return cls(task_type=task_type, **params)


def list_models(task_type: str | None = None) -> list[dict[str, Any]]:
    """Return catalog of available models with metadata."""
    result = []
    for name, (cls, supported) in MODEL_REGISTRY.items():
        if task_type and task_type not in supported:
            continue
        result.append({
            "name": name,
            "task_types": supported,
            "default_params": cls.default_params,
        })
    return result


def available_model_names(task_type: str | None = None) -> list[str]:
    """Return list of model names, optionally filtered by task type."""
    names = []
    for name, (_, supported) in MODEL_REGISTRY.items():
        if task_type is None or task_type in supported:
            names.append(name)
    return names
