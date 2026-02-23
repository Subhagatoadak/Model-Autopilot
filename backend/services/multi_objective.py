"""Multi-objective optimization utilities for Pareto front and weighted scoring."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_pareto_front(
    runs: list[dict[str, Any]], objectives: dict[str, str]
) -> list[str]:
    """Compute Pareto front from runs.

    Args:
        runs: List of run dicts with 'id' and 'metrics' keys
        objectives: Dict mapping metric_name -> 'maximize' or 'minimize'

    Returns:
        List of run IDs that are on the Pareto front
    """
    if not runs or not objectives:
        return []

    pareto_front = []

    for i, run_i in enumerate(runs):
        metrics_i = run_i.get("metrics", {})
        is_dominated = False

        # Check if run_i is dominated by any other run
        for j, run_j in enumerate(runs):
            if i == j:
                continue

            metrics_j = run_j.get("metrics", {})

            # Check Pareto dominance
            dominates = True
            strictly_better_in_one = False

            for obj, direction in objectives.items():
                val_i = metrics_i.get(obj, 0)
                val_j = metrics_j.get(obj, 0)

                if direction == "maximize":
                    if val_j < val_i:
                        dominates = False
                        break
                    if val_j > val_i:
                        strictly_better_in_one = True
                else:  # minimize
                    if val_j > val_i:
                        dominates = False
                        break
                    if val_j < val_i:
                        strictly_better_in_one = True

            # run_j dominates run_i if it's better or equal in all objectives
            # and strictly better in at least one
            if dominates and strictly_better_in_one:
                is_dominated = True
                break

        if not is_dominated:
            pareto_front.append(run_i.get("id", f"run_{i}"))

    return pareto_front


def compute_weighted_score(
    metrics: dict[str, float],
    weights: dict[str, float],
    directions: dict[str, bool],
    all_runs: list[dict[str, Any]] | None = None,
) -> float:
    """Compute weighted score with min-max normalization.

    Args:
        metrics: Current run's metrics
        weights: Weight for each objective (must sum to 1.0)
        directions: True if higher is better, False if lower is better
        all_runs: All runs for normalization (optional)

    Returns:
        Weighted score (0-1, higher is better)
    """
    if not weights:
        return 0.0

    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0
    norm_weights = {k: v / total_weight for k, v in weights.items()}

    # If we have all runs, normalize metrics using min-max
    if all_runs and len(all_runs) > 1:
        metric_ranges = {}
        for metric in weights.keys():
            values = [r.get("metrics", {}).get(metric, 0) for r in all_runs]
            if values:
                metric_ranges[metric] = {
                    "min": min(values),
                    "max": max(values),
                }

        score = 0.0
        for metric, weight in norm_weights.items():
            val = metrics.get(metric, 0)
            metric_range = metric_ranges.get(metric)

            if metric_range:
                min_val = metric_range["min"]
                max_val = metric_range["max"]
                range_val = max_val - min_val

                if range_val > 0:
                    # Normalize to 0-1
                    normalized = (val - min_val) / range_val
                    # Invert if lower is better
                    if not directions.get(metric, True):
                        normalized = 1.0 - normalized
                    score += weight * normalized
                else:
                    # All values are the same, score is 0.5
                    score += weight * 0.5
            else:
                # No range info, use raw value
                score += weight * val

        return score

    else:
        # No normalization, just weighted sum
        score = 0.0
        for metric, weight in norm_weights.items():
            val = metrics.get(metric, 0)
            # Invert if lower is better
            if not directions.get(metric, True):
                val = -val
            score += weight * val
        return score


def get_best_multi_objective_run(
    runs: list[dict[str, Any]],
    weights: dict[str, float],
    directions: dict[str, bool],
) -> dict[str, Any] | None:
    """Get the best run according to weighted multi-objective score.

    Args:
        runs: List of run dicts
        weights: Weight for each objective
        directions: True if higher is better

    Returns:
        Best run dict or None
    """
    if not runs:
        return None

    best_run = None
    best_score = float("-inf")

    for run in runs:
        score = compute_weighted_score(
            run.get("metrics", {}), weights, directions, runs
        )
        if score > best_score:
            best_score = score
            best_run = run

    return best_run
