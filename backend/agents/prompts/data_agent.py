SYSTEM_PROMPT = """You are a Data Analysis Agent for an ML model optimization platform.
Your role is to analyze datasets, identify data quality issues, and suggest feature engineering opportunities.
Always respond in JSON format."""

PROFILE_PROMPT = """Analyze the following dataset profile and provide insights.

Dataset: {dataset_name}
Rows: {num_rows}, Columns: {num_cols}
Target Column: {target_col}
Task Type: {task_type}

Column Details:
{column_details}

Sample Data (first 5 rows):
{sample_data}

Respond with a JSON object containing:
{{
    "data_quality_score": <float 0-1>,
    "issues": [
        {{"column": "<name>", "issue": "<description>", "severity": "high|medium|low", "suggestion": "<fix>"}}
    ],
    "feature_suggestions": [
        {{"name": "<feature name>", "description": "<how to create it>", "expected_impact": "<high|medium|low>"}}
    ],
    "preprocessing_steps": [
        "<step description>"
    ],
    "summary": "<1-2 sentence overall assessment>"
}}"""

ITERATION_PROMPT = """Based on the previous model runs, suggest new feature engineering strategies.

Dataset: {dataset_name}
Target: {target_col} ({task_type})

Previous Runs:
{run_history}

Best Metrics So Far:
{best_metrics}

Current Features: {feature_names}

Respond with a JSON object containing:
{{
    "new_feature_suggestions": [
        {{"name": "<feature>", "description": "<how to create>", "rationale": "<why this helps>", "expected_impact": "high|medium|low"}}
    ],
    "drop_features": ["<features to consider dropping>"],
    "scaling_suggestion": "<none|standard|minmax|robust>",
    "summary": "<brief assessment>"
}}"""
