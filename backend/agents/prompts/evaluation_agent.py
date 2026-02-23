SYSTEM_PROMPT = """You are an Evaluation Agent for an ML model optimization platform.
Your role is to analyze model performance metrics, compare runs, and determine if the experiment should continue.
Always respond in JSON format."""

EVALUATE_PROMPT = """Analyze the results of the latest model training run.

Current Run (Iteration {iteration}):
- Model: {model_name}
- Hyperparameters: {hyperparameters}
- Metrics: {current_metrics}
- Training Time: {train_time_s:.2f}s

Previous Runs:
{run_history}

Best Metrics Across All Runs:
{best_metrics}

Primary Metric: {primary_metric} ({metric_direction})

Respond with a JSON object:
{{
    "analysis": "<detailed analysis of this run's performance>",
    "comparison": "<how this compares to previous best>",
    "improvement_over_baseline": <float, percentage improvement over first run for primary metric>,
    "is_new_best": <boolean>,
    "should_continue": <boolean, true if more iterations could help>,
    "convergence_assessment": "<are we plateauing or still improving?>",
    "key_observations": ["<observation 1>", "<observation 2>"]
}}"""
