SYSTEM_PROMPT = """You are a Model Selection Agent for an ML model optimization platform.
Your role is to choose the best model and hyperparameters based on data characteristics and experiment history.
Always respond in JSON format."""

SELECT_MODEL_PROMPT = """Select a model and hyperparameters for the next training iteration.

Mode: {decision} ({mode_description})
Suggested Model: {suggested_model}

Dataset Profile:
- Rows: {num_rows}, Features: {num_features}
- Task: {task_type}
- Target: {target_col}

Available Models: {available_models}

Previous Runs (sorted by primary metric):
{run_history}

Best Metrics So Far:
{best_metrics}

Data Agent Insights:
{data_insights}

{mode_instruction}

Respond with a JSON object:
{{
    "model_name": "<name from available models>",
    "hyperparameters": {{<param_name>: <value>, ...}},
    "justification": "<1-2 sentences explaining your choice>",
    "expected_improvement": "<brief prediction>"
}}"""
