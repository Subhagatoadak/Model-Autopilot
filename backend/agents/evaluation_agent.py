from __future__ import annotations

import json
import logging
from typing import Any

from backend.agents.llm_client import LLMClient
from backend.agents.prompts.evaluation_agent import EVALUATE_PROMPT, SYSTEM_PROMPT
from backend.services.experiment_service import save_trace
from backend.services.metrics_service import is_higher_better

logger = logging.getLogger(__name__)


class EvaluationAgent:
    """Analyzes model performance and determines if experiment should continue."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def evaluate(
        self,
        experiment_id: str,
        iteration: int,
        model_name: str,
        hyperparameters: dict,
        current_metrics: dict[str, float],
        train_time_s: float,
        run_history: list[dict],
        best_metrics: dict[str, float],
        primary_metric: str,
    ) -> dict[str, Any]:
        """Analyze the latest run and compare to history."""
        higher = is_higher_better(primary_metric)

        # Format run history
        history_str = ""
        for run in run_history[-10:]:
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in run.get("metrics", {}).items())
            history_str += f"  Iter {run['iteration']}: {run['model_name']} ({metrics_str}) [{run.get('train_time_s', 0):.1f}s]\n"

        prompt = EVALUATE_PROMPT.format(
            iteration=iteration,
            model_name=model_name,
            hyperparameters=json.dumps(hyperparameters),
            current_metrics=json.dumps({k: round(v, 4) for k, v in current_metrics.items()}),
            train_time_s=train_time_s,
            run_history=history_str or "No previous runs",
            best_metrics=json.dumps({k: round(v, 4) for k, v in best_metrics.items()}) if best_metrics else "N/A",
            primary_metric=primary_metric,
            metric_direction="higher is better" if higher else "lower is better",
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
            agent_name="evaluation",
            step_type="observation",
            content=json.dumps(result),
            metadata={
                "model": model_name,
                "primary_metric": primary_metric,
                "value": current_metrics.get(primary_metric, 0),
            },
        )

        return result
