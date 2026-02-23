from __future__ import annotations

import json
import logging
from typing import Any

from backend.agents.llm_client import LLMClient
from backend.agents.prompts.model_agent import SELECT_MODEL_PROMPT, SYSTEM_PROMPT
from backend.model_zoo.registry import available_model_names
from backend.services.experiment_service import save_trace

logger = logging.getLogger(__name__)


class ModelAgent:
    """Selects models and hyperparameters using LLM guidance."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def select_model(
        self,
        experiment_id: str,
        iteration: int,
        task_type: str,
        target_col: str,
        num_rows: int,
        num_features: int,
        decision: str,
        suggested_model: str,
        run_history: list[dict],
        best_metrics: dict,
        data_insights: dict | None = None,
        selected_models: list[str] | None = None,
    ) -> dict[str, Any]:
        """Select model and hyperparameters for the next iteration."""
        if selected_models:
            available = selected_models
        else:
            available = available_model_names(task_type)

        # Format run history
        history_str = ""
        for run in run_history[-10:]:
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in run.get("metrics", {}).items())
            history_str += f"  Iter {run['iteration']}: {run['model_name']} ({metrics_str})\n"

        # Find best model name
        best_model = "none"
        if run_history:
            best_model = run_history[-1].get("model_name", "none")

        single_model_mode = len(available) == 1

        if single_model_mode:
            # User selected only one model - focus entirely on hyperparameter tuning
            mode_description = f"HYPERPARAMETER TUNING: Optimize {available[0]}"
            mode_instruction = (
                f"SINGLE MODEL MODE: The user has selected ONLY {available[0]}. "
                "You MUST use this exact model. Do NOT suggest any other model. "
                "Focus entirely on finding better hyperparameter configurations. "
                "Analyze previous runs to understand which hyperparameter changes "
                "improved performance and which didn't. Try systematic variations: "
                "adjust learning rates, regularization, tree depths, number of estimators, etc. "
                "Be creative with hyperparameter combinations but stay with this model."
            )
        elif decision == "explore":
            mode_description = "EXPLORE: Try something new and different"
            mode_instruction = (
                "EXPLORE MODE: Consider trying a model type or configuration we haven't "
                "used yet, or an unconventional hyperparameter setting. Be creative."
            )
        else:
            mode_description = "EXPLOIT: Refine what's working best"
            mode_instruction = (
                f"EXPLOIT MODE: Our best model so far is {best_model}. "
                "Suggest refined hyperparameters to squeeze out more performance."
            )

        prompt = SELECT_MODEL_PROMPT.format(
            decision=decision,
            mode_description=mode_description,
            mode_instruction=mode_instruction,
            suggested_model=suggested_model,
            num_rows=num_rows,
            num_features=num_features,
            task_type=task_type,
            target_col=target_col,
            available_models=", ".join(available),
            run_history=history_str or "No runs yet (first iteration)",
            best_metrics=json.dumps(best_metrics) if best_metrics else "N/A",
            data_insights=json.dumps(data_insights) if data_insights else "N/A",
        )

        result = await self.llm.complete_json(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )

        # Validate model name
        model_name = result.get("model_name", suggested_model)
        if model_name not in available:
            logger.warning(f"LLM suggested invalid model '{model_name}', falling back to '{suggested_model}'")
            result["model_name"] = suggested_model

        await save_trace(
            experiment_id=experiment_id,
            iteration=iteration,
            agent_name="model",
            step_type="action",
            content=json.dumps(result),
            metadata={"decision": decision, "suggested": suggested_model},
        )

        return result
