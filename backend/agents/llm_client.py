from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from backend.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Async OpenAI wrapper with retry and structured output support."""

    def __init__(self, api_key: str | None = None):
        # Use provided API key or fall back to settings (for backward compatibility)
        key = api_key or settings.openai_api_key or ""
        self.client = AsyncOpenAI(api_key=key) if key else None
        self.default_model = settings.openai_model  # gpt-4o-mini
        self.strong_model = settings.openai_matching_model  # gpt-4o

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        """Single completion with the specified or default model."""
        model = model or self.default_model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = await self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Complete and parse JSON response."""
        raw = await self.complete(
            messages, model=model, temperature=temperature, json_mode=True
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON: {raw[:200]}")
            return {"error": "Failed to parse response", "raw": raw}

    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
        model: str | None = None,
        temperature: float = 0.7,
    ) -> BaseModel:
        """Complete and parse into a Pydantic model."""
        result = await self.complete_json(messages, model=model, temperature=temperature)
        return response_model.model_validate(result)
