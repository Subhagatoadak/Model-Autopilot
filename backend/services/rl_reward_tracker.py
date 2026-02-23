"""Reinforcement learning reward tracker for model tuning.

Tracks action -> reward pairs across iterations to learn which
transformations and hyperparameter changes improve model performance.
Uses a simple tabular approach: each action type accumulates rewards
based on metric improvement, and the tracker provides a ranked summary
to guide the recommendation agent.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RLRewardTracker:
    """Tracks actions and their rewards (metric improvement) across iterations.

    Each iteration records:
    - actions taken (transformations, hyperparameter changes)
    - the resulting metric change (reward)

    The tracker computes statistics per action type and provides a
    ranked summary for the LLM recommendation agent.
    """

    def __init__(self, primary_metric: str, higher_is_better: bool = True):
        self.primary_metric = primary_metric
        self.higher_is_better = higher_is_better
        self.history: list[dict[str, Any]] = []
        # action_type -> list of rewards
        self.action_rewards: dict[str, list[float]] = {}
        self.prev_metric_value: float | None = None

    def record_baseline(self, metric_value: float) -> None:
        """Record the baseline metric from the first iteration."""
        self.prev_metric_value = metric_value

    def record_iteration(
        self,
        iteration: int,
        actions: list[dict[str, Any]],
        metric_value: float,
        model_name: str,
        hyperparameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Record an iteration's actions and compute rewards.

        Args:
            iteration: Current iteration number
            actions: List of action dicts (transformations applied)
            metric_value: Current primary metric value
            model_name: Model used
            hyperparameters: Hyperparameters used

        Returns:
            Dict with reward info for this iteration
        """
        # Compute reward as metric improvement
        if self.prev_metric_value is not None:
            if self.higher_is_better:
                reward = metric_value - self.prev_metric_value
            else:
                reward = self.prev_metric_value - metric_value  # lower is better
        else:
            reward = 0.0

        # Normalize all actions into action types
        action_types = []
        for action in actions:
            action_type = action.get("type", "unknown")
            action_types.append(action_type)
            if action_type not in self.action_rewards:
                self.action_rewards[action_type] = []
            self.action_rewards[action_type].append(reward)

        # Also track hyperparameter changes as an action
        if hyperparameters:
            hp_type = f"hyperparam_{model_name}"
            if hp_type not in self.action_rewards:
                self.action_rewards[hp_type] = []
            self.action_rewards[hp_type].append(reward)

        entry = {
            "iteration": iteration,
            "model_name": model_name,
            "actions": action_types,
            "metric_value": metric_value,
            "reward": reward,
            "cumulative_improvement": (
                metric_value - (self.history[0]["metric_value"] if self.history else metric_value)
            ),
        }
        self.history.append(entry)
        self.prev_metric_value = metric_value

        return entry

    def get_action_rankings(self) -> list[dict[str, Any]]:
        """Get actions ranked by average reward.

        Returns list of dicts sorted by avg reward (best first):
        [{"action": str, "avg_reward": float, "times_used": int, "total_reward": float}]
        """
        rankings = []
        for action, rewards in self.action_rewards.items():
            avg = sum(rewards) / len(rewards) if rewards else 0.0
            rankings.append({
                "action": action,
                "avg_reward": round(avg, 6),
                "times_used": len(rewards),
                "total_reward": round(sum(rewards), 6),
                "best_reward": round(max(rewards), 6) if rewards else 0.0,
                "worst_reward": round(min(rewards), 6) if rewards else 0.0,
            })
        rankings.sort(key=lambda x: x["avg_reward"], reverse=True)
        return rankings

    def get_rl_summary(self) -> str:
        """Generate a summary string for the LLM recommendation agent.

        This provides the RL context so the agent can learn from past
        action-reward pairs and make better recommendations.
        """
        rankings = self.get_action_rankings()
        if not rankings:
            return "No actions recorded yet. This is the first iteration."

        lines = [
            "=== RL Reward Summary (Action -> Metric Improvement) ===",
            f"Primary metric: {self.primary_metric} ({'higher' if self.higher_is_better else 'lower'} is better)",
            f"Total iterations tracked: {len(self.history)}",
            "",
            "Action Rankings (by average reward):",
        ]

        for r in rankings:
            sign = "+" if r["avg_reward"] >= 0 else ""
            lines.append(
                f"  {r['action']}: avg={sign}{r['avg_reward']:.4f}, "
                f"used={r['times_used']}x, "
                f"best={'+' if r['best_reward'] >= 0 else ''}{r['best_reward']:.4f}"
            )

        # Recent trajectory
        if self.history:
            lines.append("")
            lines.append("Recent iteration rewards:")
            for entry in self.history[-5:]:
                sign = "+" if entry["reward"] >= 0 else ""
                lines.append(
                    f"  Iter {entry['iteration']}: {entry['model_name']} "
                    f"-> {self.primary_metric}={entry['metric_value']:.4f} "
                    f"(reward={sign}{entry['reward']:.4f}) "
                    f"actions={entry['actions']}"
                )

        # Recommendations based on rewards
        positive_actions = [r for r in rankings if r["avg_reward"] > 0]
        negative_actions = [r for r in rankings if r["avg_reward"] < 0]

        if positive_actions:
            lines.append("")
            lines.append("POSITIVE actions (repeat these): " +
                         ", ".join(r["action"] for r in positive_actions[:3]))
        if negative_actions:
            lines.append("NEGATIVE actions (avoid these): " +
                         ", ".join(r["action"] for r in negative_actions[:3]))

        return "\n".join(lines)
