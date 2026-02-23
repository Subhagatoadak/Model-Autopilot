from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from backend.agents.data_agent import DataAgent
from backend.agents.evaluation_agent import EvaluationAgent
from backend.agents.llm_client import LLMClient
from backend.agents.model_agent import ModelAgent
from backend.agents.recommendation_agent import RecommendationAgent
from backend.model_zoo.registry import get_model, available_model_names
from backend.routers.ws import broadcast
from backend.schemas.experiment import ExperimentConfig
from backend.services.dataset_service import load_dataset, prepare_data
from backend.services.experiment_service import (
    get_run_history,
    save_run,
    save_trace,
    update_experiment_status,
)
from backend.services.explore_exploit import ExploreExploitStrategy
from backend.services.feature_engineering import FeatureEngineer
from backend.services.metrics_service import (
    compute_metrics,
    default_metrics,
    is_higher_better,
    primary_metric as get_primary_metric,
)
from backend.services.rl_reward_tracker import RLRewardTracker

logger = logging.getLogger(__name__)


async def _broadcast_step(experiment_id: str, agent: str, step: str, content: str):
    """Broadcast an agent step via WebSocket."""
    await broadcast(experiment_id, {
        "type": "agent_step",
        "data": {"agent": agent, "step": step, "content": content},
    })


async def run_experiment(experiment_id: str, config: ExperimentConfig, openai_api_key: str | None = None) -> None:
    """Main orchestrator loop. Runs as an async background task."""
    llm = LLMClient(api_key=openai_api_key)
    data_agent = DataAgent(llm)
    model_agent = ModelAgent(llm)
    eval_agent = EvaluationAgent(llm)
    rec_agent = RecommendationAgent(llm)
    strategy = ExploreExploitStrategy(epsilon=config.epsilon)

    try:
        await _broadcast_step(experiment_id, "orchestrator", "start", "Loading dataset...")

        # Load and prepare data
        df, metadata = await load_dataset(config.dataset_id)
        target_col = config.target_col or metadata.get("target_col")
        if not target_col:
            raise ValueError("No target column specified")

        task_type = config.task_type or metadata.get("task_type")
        if not task_type:
            if df[target_col].dtype == "object" or df[target_col].nunique() <= 20:
                task_type = "classification"
            else:
                task_type = "regression"

        p_metric = config.primary_metric or get_primary_metric(task_type)
        metric_names = config.secondary_metrics or default_metrics(task_type)
        if p_metric not in metric_names:
            metric_names = [p_metric] + metric_names
        higher = is_higher_better(p_metric)

        X_train_orig, X_test_orig, y_train, y_test, feature_names = prepare_data(
            df, target_col, task_type,
            feature_cols=config.feature_cols or None,
            test_size=config.test_size,
        )

        await _broadcast_step(
            experiment_id, "orchestrator", "ready",
            f"Dataset loaded: {len(df)} rows, {len(feature_names)} features, task={task_type}",
        )

        # Initialize Feature Engineer and RL Reward Tracker
        feat_engineer = FeatureEngineer(feature_names)
        rl_tracker = RLRewardTracker(p_metric, higher_is_better=higher)

        # Working copies of data (transforms modify these)
        X_train = X_train_orig.copy()
        X_test = X_test_orig.copy()

        # Step 1: Data Agent - initial profiling
        await _broadcast_step(experiment_id, "data", "profiling", "Analyzing data quality...")
        data_profile = await data_agent.profile(experiment_id, df, metadata)
        await _broadcast_step(
            experiment_id, "data", "complete",
            data_profile.get("summary", "Data profiling complete"),
        )

        available = config.selected_models or available_model_names(task_type)
        single_model_mode = len(available) == 1
        best_metrics: dict[str, float] = {}
        data_insights = data_profile
        pending_transforms: list[dict[str, Any]] = []
        previous_version_id: str | None = None

        # Save initial dataset version (version 0 - original data)
        from backend.services.dataset_service import save_dataset_version
        previous_version_id = await save_dataset_version(
            dataset_id=config.dataset_id,
            experiment_id=experiment_id,
            version=0,
            parent_version=None,
            transforms=[],
            num_features=len(feature_names),
            feature_names=feature_names,
        )

        # Main iteration loop
        for iteration in range(1, config.max_iterations + 1):
            await broadcast(experiment_id, {
                "type": "experiment_status",
                "data": {"status": "running", "iteration": iteration, "max_iterations": config.max_iterations},
            })

            # --- Apply pending transforms from previous iteration's recommendation ---
            if pending_transforms:
                await _broadcast_step(
                    experiment_id, "data", "transforming",
                    f"Applying {len(pending_transforms)} data transformation(s)...",
                )
                for transform in pending_transforms:
                    t_type = transform.get("type", "unknown")
                    X_train, X_test, updated_names = feat_engineer.apply(
                        X_train, X_test, transform,
                    )
                    await _broadcast_step(
                        experiment_id, "data", "transform_applied",
                        f"Applied: {t_type} -> now {X_train.shape[1]} features",
                    )

                # Save dataset version after applying transforms
                previous_version_id = await save_dataset_version(
                    dataset_id=config.dataset_id,
                    experiment_id=experiment_id,
                    version=iteration,
                    parent_version=previous_version_id,
                    transforms=pending_transforms,
                    num_features=X_train.shape[1],
                    feature_names=feat_engineer.current_feature_names,
                )

                pending_transforms = []  # Clear after applying

            # Step 2: Explore/Exploit decision
            run_history = await get_run_history(experiment_id)
            run_dicts = [
                {"model_name": r["model_name"], "metrics": r["metrics"], "iteration": r["iteration"], "id": r.get("id", "")}
                for r in run_history
            ]

            # Multi-objective optimization
            if config.multi_objective and config.objective_weights:
                # Build directions dict (True if higher is better)
                directions = {}
                for obj in config.objective_weights.keys():
                    if obj == "train_time_s":
                        directions[obj] = False  # Lower training time is better
                    else:
                        directions[obj] = is_higher_better(obj)

                decision, suggested = strategy.decide_multi_objective(
                    run_dicts, available, config.objective_weights, directions
                )
                await _broadcast_step(
                    experiment_id, "strategy", f"{decision}_multi",
                    f"Iteration {iteration}: Multi-objective {decision.upper()} - suggesting {suggested}",
                )
            else:
                # Single-objective optimization
                decision, suggested = strategy.decide(run_dicts, available, p_metric, higher)
                await _broadcast_step(
                    experiment_id, "strategy", decision,
                    f"Iteration {iteration}: {decision.upper()} - suggesting {suggested}",
                )

            # Step 3: Model Agent - select model & hyperparameters
            await _broadcast_step(experiment_id, "model", "selecting", "Choosing model and hyperparameters...")
            model_selection = await model_agent.select_model(
                experiment_id=experiment_id,
                iteration=iteration,
                task_type=task_type,
                target_col=target_col,
                num_rows=len(df),
                num_features=X_train.shape[1],
                decision=decision,
                suggested_model=suggested,
                run_history=run_dicts,
                best_metrics=best_metrics,
                data_insights=data_insights,
                selected_models=config.selected_models or None,
            )

            model_name = model_selection.get("model_name", suggested)
            hyperparams = model_selection.get("hyperparameters", {})
            await _broadcast_step(
                experiment_id, "model", "selected",
                f"Selected {model_name}: {model_selection.get('justification', '')}",
            )

            # Step 4: Train model
            await _broadcast_step(experiment_id, "training", "start", f"Training {model_name}...")
            start_time = time.time()

            try:
                clean_params = {k: v for k, v in hyperparams.items() if isinstance(v, (int, float, str, bool, list))}
                model = get_model(model_name, task_type, **clean_params)
                await asyncio.to_thread(model.fit, X_train, y_train)
            except Exception as e:
                logger.error(f"Training failed for {model_name}: {e}")
                await _broadcast_step(experiment_id, "training", "error", f"Training failed: {e}")
                await save_run(
                    experiment_id=experiment_id,
                    iteration=iteration,
                    model_name=model_name,
                    hyperparameters=hyperparams,
                    metrics={},
                    train_time_s=time.time() - start_time,
                )
                continue

            train_time = time.time() - start_time

            # Serialize model for export
            import pickle
            from pathlib import Path
            artifacts_dir = Path(__file__).parent.parent.parent / "data" / "artifacts" / experiment_id
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            model_artifact_path = str(artifacts_dir / f"model_iter_{iteration}_{model_name}.pkl")
            try:
                with open(model_artifact_path, "wb") as f:
                    pickle.dump(model._model, f)
            except Exception as e:
                logger.warning(f"Failed to serialize model: {e}")
                model_artifact_path = None

            # Step 5: Predict and compute metrics
            y_pred = await asyncio.to_thread(model.predict, X_test)
            y_proba = None
            if task_type == "classification":
                y_proba = await asyncio.to_thread(model.predict_proba, X_test)

            metrics = compute_metrics(y_test, y_pred, task_type, metric_names, y_proba)

            await _broadcast_step(
                experiment_id, "training", "complete",
                f"Trained in {train_time:.1f}s - {p_metric}={metrics.get(p_metric, 0):.4f}",
            )

            # Update best metrics
            for k, v in metrics.items():
                if k not in best_metrics:
                    best_metrics[k] = v
                elif is_higher_better(k) and v > best_metrics[k]:
                    best_metrics[k] = v
                elif not is_higher_better(k) and v < best_metrics[k]:
                    best_metrics[k] = v

            # Compute UCB score
            ucb_scores = strategy.compute_ucb_scores(run_dicts + [{"model_name": model_name, "metrics": metrics}], p_metric, higher)
            ucb_score = ucb_scores.get(model_name, 0)

            # Save run
            await save_run(
                experiment_id=experiment_id,
                iteration=iteration,
                model_name=model_name,
                hyperparameters=hyperparams,
                metrics=metrics,
                train_time_s=train_time,
                ucb_score=ucb_score,
                model_artifact_path=model_artifact_path,
            )

            # Broadcast run complete
            await broadcast(experiment_id, {
                "type": "run_complete",
                "data": {
                    "iteration": iteration,
                    "model_name": model_name,
                    "metrics": metrics,
                    "train_time_s": train_time,
                    "ucb_score": ucb_score,
                },
            })

            await broadcast(experiment_id, {
                "type": "metrics_update",
                "data": {"iteration": iteration, "metrics": metrics},
            })

            # --- RL Reward Tracking ---
            current_metric_val = metrics.get(p_metric, 0)
            if iteration == 1:
                rl_tracker.record_baseline(current_metric_val)

            # Record this iteration's actions (transforms applied + hyperparams)
            iteration_actions = list(feat_engineer.applied_transforms[-len(pending_transforms):]) if pending_transforms else []
            rl_entry = rl_tracker.record_iteration(
                iteration=iteration,
                actions=iteration_actions,
                metric_value=current_metric_val,
                model_name=model_name,
                hyperparameters=hyperparams,
            )

            rl_reward = rl_entry.get("reward", 0)
            reward_sign = "+" if rl_reward >= 0 else ""
            await _broadcast_step(
                experiment_id, "rl_tracker", "reward",
                f"Iteration {iteration}: reward={reward_sign}{rl_reward:.4f} (cumulative={rl_entry.get('cumulative_improvement', 0):.4f})",
            )

            # Step 6: Evaluation Agent
            await _broadcast_step(experiment_id, "evaluation", "analyzing", "Analyzing results...")
            eval_result = await eval_agent.evaluate(
                experiment_id=experiment_id,
                iteration=iteration,
                model_name=model_name,
                hyperparameters=hyperparams,
                current_metrics=metrics,
                train_time_s=train_time,
                run_history=run_dicts,
                best_metrics=best_metrics,
                primary_metric=p_metric,
            )

            await _broadcast_step(
                experiment_id, "evaluation", "complete",
                eval_result.get("analysis", "Evaluation complete"),
            )

            # Check if we should stop early
            should_continue = eval_result.get("should_continue", True)
            if not should_continue and iteration >= 3:
                await _broadcast_step(
                    experiment_id, "evaluation", "converged",
                    "Model performance has converged. Stopping early.",
                )
                break

            # Step 7: Recommendation Agent (Tree of Thought) with RL context
            if iteration < config.max_iterations:
                await _broadcast_step(experiment_id, "recommendation", "thinking", "Generating improvement strategies (Tree of Thought)...")
                updated_history = await get_run_history(experiment_id)
                updated_dicts = [
                    {"model_name": r["model_name"], "metrics": r["metrics"], "iteration": r["iteration"]}
                    for r in updated_history
                ]

                # Get RL summary and feature state for the recommendation agent
                rl_summary = rl_tracker.get_rl_summary()
                feature_state = feat_engineer.get_state_summary()

                thought_tree = await rec_agent.run(
                    experiment_id=experiment_id,
                    iteration=iteration,
                    task_type=task_type,
                    num_rows=len(df),
                    num_features=X_train.shape[1],
                    primary_metric=p_metric,
                    run_history=updated_dicts,
                    best_metrics=best_metrics,
                    data_profile=data_profile,
                    evaluation_summary=eval_result.get("analysis", ""),
                    rl_summary=rl_summary,
                    feature_state=feature_state,
                    single_model_mode=single_model_mode,
                )

                # Broadcast thought tree nodes for visualization
                for branch in thought_tree.branches:
                    branch_data = branch.model_dump()
                    branch_data["iteration"] = iteration
                    await broadcast(experiment_id, {
                        "type": "thought_node",
                        "data": branch_data,
                    })

                await broadcast(experiment_id, {
                    "type": "thought_selected",
                    "data": {"path": thought_tree.selected_path, "iteration": iteration},
                })

                await _broadcast_step(
                    experiment_id, "recommendation", "complete",
                    f"Selected path: {' -> '.join(thought_tree.selected_path)}",
                )

                # Extract transforms from the selected strategy for next iteration
                pending_transforms = _extract_transforms_from_tree(thought_tree)
                if pending_transforms:
                    transform_names = [t.get("type", "unknown") for t in pending_transforms]
                    await _broadcast_step(
                        experiment_id, "recommendation", "transforms_queued",
                        f"Queued {len(pending_transforms)} transform(s) for next iteration: {', '.join(transform_names)}",
                    )

                # Broadcast RL action rankings
                rankings = rl_tracker.get_action_rankings()
                if rankings:
                    top_actions = rankings[:3]
                    ranking_str = ", ".join(
                        f"{r['action']}(avg={'+' if r['avg_reward'] >= 0 else ''}{r['avg_reward']:.4f})"
                        for r in top_actions
                    )
                    await _broadcast_step(
                        experiment_id, "rl_tracker", "rankings",
                        f"Top actions: {ranking_str}",
                    )

                # Data agent suggests features every 3rd iteration
                if iteration % 3 == 0:
                    data_insights = await data_agent.suggest_features(
                        experiment_id=experiment_id,
                        iteration=iteration,
                        metadata=metadata,
                        run_history=updated_dicts,
                        best_metrics=best_metrics,
                        feature_names=feat_engineer.current_feature_names,
                    )

        # Experiment complete
        await update_experiment_status(experiment_id, "completed")
        await broadcast(experiment_id, {
            "type": "experiment_status",
            "data": {"status": "completed", "best_metrics": best_metrics},
        })
        await _broadcast_step(
            experiment_id, "orchestrator", "complete",
            f"Experiment complete! Best {p_metric}: {best_metrics.get(p_metric, 0):.4f}",
        )

    except asyncio.CancelledError:
        await update_experiment_status(experiment_id, "stopped")
        await broadcast(experiment_id, {
            "type": "experiment_status",
            "data": {"status": "stopped"},
        })
    except Exception as e:
        logger.exception(f"Experiment {experiment_id} failed")
        await update_experiment_status(experiment_id, "failed")
        await broadcast(experiment_id, {
            "type": "experiment_status",
            "data": {"status": "failed", "error": str(e)},
        })


def _extract_transforms_from_tree(tree: Any) -> list[dict[str, Any]]:
    """Extract transform actions from the selected path in the thought tree.

    Walks the selected branches/children and collects any 'transforms' lists.
    Falls back to inferring transforms from the node content text.
    """
    transforms: list[dict[str, Any]] = []
    selected_ids = set(tree.selected_path)

    for branch in tree.branches:
        if branch.id in selected_ids:
            # Check if the branch content mentions specific transforms
            transforms.extend(_infer_transforms_from_content(branch.content))
            for child in branch.children:
                if child.id in selected_ids:
                    transforms.extend(_infer_transforms_from_content(child.content))

    return transforms


def _infer_transforms_from_content(content: str) -> list[dict[str, Any]]:
    """Infer transform actions from textual content of a thought node.

    Maps common keywords in recommendation text to transform actions.
    """
    transforms: list[dict[str, Any]] = []
    content_lower = content.lower()

    # Map keywords to transform types
    keyword_map = {
        "standard scal": {"type": "standard_scale"},
        "normalize": {"type": "standard_scale"},
        "standardize": {"type": "standard_scale"},
        "minmax": {"type": "minmax_scale"},
        "min-max": {"type": "minmax_scale"},
        "robust scal": {"type": "robust_scale"},
        "log transform": {"type": "log_transform"},
        "logarithm": {"type": "log_transform"},
        "log1p": {"type": "log_transform"},
        "sqrt": {"type": "sqrt_transform"},
        "square root": {"type": "sqrt_transform"},
        "polynomial": {"type": "polynomial_features", "degree": 2},
        "interaction": {"type": "interaction_features"},
        "feature interaction": {"type": "interaction_features"},
        "remove feature": {"type": "remove_features", "columns": []},
        "drop feature": {"type": "remove_features", "columns": []},
        "feature selection": {"type": "remove_features", "columns": []},
        "squared feature": {"type": "square_features"},
        "reciprocal": {"type": "reciprocal_features"},
    }

    seen_types = set()
    for keyword, transform in keyword_map.items():
        if keyword in content_lower and transform["type"] not in seen_types:
            transforms.append(dict(transform))
            seen_types.add(transform["type"])

    return transforms
