from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from backend.agents.llm_client import LLMClient
from backend.agents.prompts.data_agent import (
    ITERATION_PROMPT,
    PROFILE_PROMPT,
    SYSTEM_PROMPT,
)
from backend.services.experiment_service import save_trace

logger = logging.getLogger(__name__)


class DataAgent:
    """Analyzes data quality, profiling, and suggests feature engineering."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def profile(
        self,
        experiment_id: str,
        df: pd.DataFrame,
        metadata: dict,
    ) -> dict[str, Any]:
        """Run initial data profiling on a dataset."""
        # Build column details
        col_details = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_pct = f"{df[col].isnull().mean() * 100:.1f}%"
            unique = df[col].nunique()
            col_details.append(f"  - {col}: type={dtype}, nulls={null_pct}, unique={unique}")

        # Get sample data
        sample = df.head(5).to_string(index=False)

        prompt = PROFILE_PROMPT.format(
            dataset_name=metadata.get("name", "unknown"),
            num_rows=len(df),
            num_cols=len(df.columns),
            target_col=metadata.get("target_col", "N/A"),
            task_type=metadata.get("task_type", "unknown"),
            column_details="\n".join(col_details),
            sample_data=sample,
        )

        result = await self.llm.complete_json(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )

        # Save trace
        await save_trace(
            experiment_id=experiment_id,
            iteration=0,
            agent_name="data",
            step_type="observation",
            content=json.dumps(result),
            metadata={"type": "initial_profile"},
        )

        return result

    async def suggest_features(
        self,
        experiment_id: str,
        iteration: int,
        metadata: dict,
        run_history: list[dict],
        best_metrics: dict,
        feature_names: list[str],
    ) -> dict[str, Any]:
        """Suggest new feature engineering based on run history."""
        # Format run history
        history_str = ""
        for run in run_history[-5:]:  # Last 5 runs
            history_str += f"  Iter {run['iteration']}: {run['model_name']} -> {run['metrics']}\n"

        prompt = ITERATION_PROMPT.format(
            dataset_name=metadata.get("name", "unknown"),
            target_col=metadata.get("target_col", "N/A"),
            task_type=metadata.get("task_type", "unknown"),
            run_history=history_str or "No runs yet",
            best_metrics=json.dumps(best_metrics),
            feature_names=", ".join(feature_names),
        )

        result = await self.llm.complete_json(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )

        await save_trace(
            experiment_id=experiment_id,
            iteration=iteration,
            agent_name="data",
            step_type="observation",
            content=json.dumps(result),
            metadata={"type": "feature_suggestion"},
        )

        return result
