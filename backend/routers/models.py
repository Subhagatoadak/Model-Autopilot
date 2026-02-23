from __future__ import annotations

from fastapi import APIRouter

from backend.model_zoo.registry import list_models

router = APIRouter()


@router.get("/zoo")
async def get_model_zoo(task_type: str | None = None):
    """List all available models in the zoo, optionally filtered by task type."""
    return list_models(task_type=task_type)
