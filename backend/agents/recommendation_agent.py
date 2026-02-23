from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from backend.agents.llm_client import LLMClient
from backend.agents.prompts.recommendation_agent import (
    EVALUATE_THOUGHT_PROMPT,
    EXPAND_STRATEGY_PROMPT,
    GENERATE_STRATEGIES_PROMPT,
    PRIORITIZE_PROMPT,
    SYSTEM_PROMPT,
)
from backend.schemas.agent_trace import ThoughtNode, ThoughtTree
from backend.services.experiment_service import save_trace

logger = logging.getLogger(__name__)


class RecommendationAgent:
    """Uses Tree of Thought reasoning to generate improvement strategies.

    Integrates RL reward tracking and feature engineering context to
    produce actionable recommendations including data transformations.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def run(
        self,
        experiment_id: str,
        iteration: int,
        task_type: str,
        num_rows: int,
        num_features: int,
        primary_metric: str,
        run_history: list[dict],
        best_metrics: dict[str, float],
        data_profile: dict | None,
        evaluation_summary: str,
        rl_summary: str = "",
        feature_state: str = "",
        single_model_mode: bool = False,
    ) -> ThoughtTree:
        """Execute full Tree of Thought reasoning.

        Returns a ThoughtTree with the selected strategy. The selected
        strategy's 'transforms' field contains executable transformation
        actions for the FeatureEngineer.
        """

        # Find best run
        best_run = None
        best_val = None
        for run in run_history:
            val = run.get("metrics", {}).get(primary_metric, 0)
            if best_val is None or val > best_val:
                best_val = val
                best_run = run

        best_model = best_run["model_name"] if best_run else "none"
        best_iteration = best_run["iteration"] if best_run else 0
        best_metric_value = best_val or 0

        # Format run history
        history_str = ""
        for run in run_history:
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in run.get("metrics", {}).items())
            history_str += f"  Iter {run['iteration']}: {run['model_name']} ({metrics_str})\n"

        common_kwargs = {
            "best_model": best_model,
            "best_metric_value": best_metric_value,
            "primary_metric": primary_metric,
            "task_type": task_type,
            "rl_summary": rl_summary or "No RL data yet (first iteration).",
            "feature_state": feature_state or "No transformations applied yet.",
        }

        # STEP 1: Generate top-level strategies
        strategies = await self._generate_strategies(
            experiment_id=experiment_id,
            iteration=iteration,
            best_iteration=best_iteration,
            num_rows=num_rows,
            num_features=num_features,
            run_history=history_str,
            data_profile=json.dumps(data_profile) if data_profile else "N/A",
            evaluation_summary=evaluation_summary,
            single_model_mode="Yes - focus on feature engineering and hyperparameter tuning only" if single_model_mode else "No - can try different models",
            **common_kwargs,
        )

        # STEP 2: Score each strategy
        branches: list[ThoughtNode] = []
        for s in strategies:
            score = await self._evaluate_thought(
                experiment_id=experiment_id,
                iteration=iteration,
                thought=s,
                num_rows=num_rows,
                num_features=num_features,
                num_iterations=len(run_history),
                run_summary=history_str,
                **common_kwargs,
            )
            s["score"] = score
            branches.append(ThoughtNode(
                id=s.get("id", str(uuid.uuid4())[:8]),
                content=s.get("content", ""),
                reasoning=s.get("reasoning", ""),
                expected_impact=s.get("expected_impact", 0),
                confidence=s.get("confidence", 0),
                score=score,
                depth=0,
            ))

        # STEP 3: Expand top-2 branches
        sorted_branches = sorted(branches, key=lambda b: b.score, reverse=True)
        for branch in sorted_branches[:2]:
            matching = [s for s in strategies if s.get("id") == branch.id]
            strategy = matching[0] if matching else {"content": branch.content, "category": "general"}

            sub_strategies = await self._expand_strategy(
                experiment_id=experiment_id,
                iteration=iteration,
                parent_id=branch.id,
                strategy=strategy,
                run_history=history_str,
                **common_kwargs,
            )

            for sub in sub_strategies:
                sub_score = await self._evaluate_thought(
                    experiment_id=experiment_id,
                    iteration=iteration,
                    thought=sub,
                    num_rows=num_rows,
                    num_features=num_features,
                    num_iterations=len(run_history),
                    run_summary=history_str,
                    **common_kwargs,
                )
                sub["score"] = sub_score
                branch.children.append(ThoughtNode(
                    id=sub.get("id", str(uuid.uuid4())[:8]),
                    content=sub.get("content", ""),
                    reasoning=sub.get("reasoning", ""),
                    expected_impact=sub.get("expected_impact", 0),
                    confidence=sub.get("confidence", 0),
                    score=sub_score,
                    depth=1,
                ))

        # STEP 4: Select best leaf path
        selected_path = self._select_best_path(sorted_branches)
        for branch in sorted_branches:
            if branch.id in selected_path:
                branch.selected = True
            for child in branch.children:
                if child.id in selected_path:
                    child.selected = True

        root_thought = f"Iteration {iteration}: Best is {best_model} with {primary_metric}={best_metric_value:.4f}"

        tree = ThoughtTree(
            root_thought=root_thought,
            branches=sorted_branches,
            selected_path=selected_path,
            iteration=iteration,
        )

        # Save the full tree as a trace
        await save_trace(
            experiment_id=experiment_id,
            iteration=iteration,
            agent_name="recommendation",
            step_type="thought",
            content=tree.model_dump_json(),
            metadata={"type": "thought_tree"},
        )

        return tree

    def extract_selected_actions(
        self,
        tree: ThoughtTree,
        strategies: list[dict],
    ) -> dict[str, Any]:
        """Extract the transforms and hyperparameters from the selected path.

        Returns dict with 'transforms' list and 'hyperparameters' dict.
        """
        selected_ids = set(tree.selected_path)
        result: dict[str, Any] = {"transforms": [], "hyperparameters": {}}

        for s in strategies:
            sid = s.get("id", "")
            if sid in selected_ids:
                result["transforms"].extend(s.get("transforms", []))
                result["hyperparameters"].update(s.get("hyperparameters", {}))
            # Check sub-strategies
            for sub in s.get("sub_strategies", []):
                if sub.get("id", "") in selected_ids:
                    result["transforms"].extend(sub.get("transforms", []))
                    result["hyperparameters"].update(sub.get("hyperparameters", {}))

        return result

    async def _generate_strategies(self, experiment_id: str, iteration: int, **kwargs) -> list[dict]:
        """Generate top-level strategies using the LLM."""
        prompt = GENERATE_STRATEGIES_PROMPT.format(**kwargs)
        result = await self.llm.complete_json(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=self.llm.strong_model,
            temperature=0.8,
        )

        strategies = result.get("strategies", [])

        # Save trace for each strategy
        root_id = await save_trace(
            experiment_id=experiment_id,
            iteration=iteration,
            agent_name="recommendation",
            step_type="thought",
            content=result.get("root_thought", ""),
            depth=0,
            metadata={"type": "root"},
        )
        for s in strategies:
            await save_trace(
                experiment_id=experiment_id,
                iteration=iteration,
                agent_name="recommendation",
                step_type="thought",
                content=s.get("content", ""),
                parent_trace_id=root_id,
                depth=1,
                score=s.get("expected_impact", 0) * s.get("confidence", 0),
                metadata={"strategy": s},
            )

        return strategies

    async def _evaluate_thought(self, experiment_id: str, iteration: int, thought: dict, **kwargs) -> float:
        """Evaluate a single thought's potential impact."""
        prompt = EVALUATE_THOUGHT_PROMPT.format(
            thought_content=thought.get("content", ""),
            thought_reasoning=thought.get("reasoning", ""),
            **kwargs,
        )
        result = await self.llm.complete_json(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=self.llm.strong_model,
        )
        return float(result.get("score", 0))

    async def _expand_strategy(self, experiment_id: str, iteration: int, parent_id: str, strategy: dict, **kwargs) -> list[dict]:
        """Expand a strategy into sub-strategies."""
        prompt = EXPAND_STRATEGY_PROMPT.format(
            strategy_content=strategy.get("content", ""),
            strategy_category=strategy.get("category", "general"),
            strategy_score=strategy.get("score", 0),
            parent_id=parent_id,
            **kwargs,
        )
        result = await self.llm.complete_json(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=self.llm.strong_model,
        )
        return result.get("sub_strategies", [])

    def _select_best_path(self, branches: list[ThoughtNode]) -> list[str]:
        """Select the highest-scoring path from root to leaf."""
        best_score = -1.0
        best_path: list[str] = []

        for branch in branches:
            if branch.children:
                for child in branch.children:
                    score = child.score
                    if score > best_score:
                        best_score = score
                        best_path = [branch.id, child.id]
            else:
                if branch.score > best_score:
                    best_score = branch.score
                    best_path = [branch.id]

        return best_path
