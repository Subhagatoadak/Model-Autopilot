from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.path.join(Path(__file__).resolve().parent.parent, ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_matching_model: str = "gpt-4o"

    # Application
    port: int = 4381
    repo_path: str = "/workspace"
    db_path: str = "data/events.db"
    max_bytes: int = 2_000_000
    upload_dir: str = "data/uploads"

    # Experiment defaults
    max_iterations: int = 10
    default_epsilon: float = 0.3

    # CORS
    cors_enabled: bool = True
    cors_origins: str = "http://localhost:5173,http://localhost:4381"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
