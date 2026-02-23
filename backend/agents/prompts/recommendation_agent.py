SYSTEM_PROMPT = """You are a Recommendation Agent for an ML model optimization platform.
You use Tree of Thought reasoning to systematically explore improvement strategies.
Your role is to generate, evaluate, and select the best path to improve model performance.

IMPORTANT: Your recommendations must include concrete, executable DATA TRANSFORMATION actions.
The system can apply these transformations: standard_scale, minmax_scale, robust_scale,
log_transform, sqrt_transform, polynomial_features, interaction_features,
remove_features, square_features, reciprocal_features.

You also have access to Reinforcement Learning reward data that shows which past actions
improved or hurt performance. USE this data to avoid repeating failed strategies and
to double down on successful ones.

Always respond in JSON format."""

GENERATE_STRATEGIES_PROMPT = """You are using Tree of Thought reasoning to find the best improvement strategy.

STEP 1: GENERATE top-level strategies.

Current State:
- Best Model: {best_model} (Iteration {best_iteration})
- Best {primary_metric}: {best_metric_value}
- Task: {task_type}
- Dataset: {num_rows} rows, {num_features} features
- Single Model Mode: {single_model_mode}

Feature Engineering State:
{feature_state}

RL Reward History (learn from past actions):
{rl_summary}

Run History (all iterations):
{run_history}

Data Profile:
{data_profile}

Evaluation Agent's Assessment:
{evaluation_summary}

Available data transformations you can recommend:
- standard_scale: StandardScaler normalization
- minmax_scale: Scale features to [0,1]
- robust_scale: Scale using median/IQR (robust to outliers)
- log_transform: Apply log1p to reduce skewness (columns param optional)
- sqrt_transform: Apply sqrt to reduce skewness (columns param optional)
- polynomial_features: Add polynomial terms (degree param, default 2)
- interaction_features: Add pairwise interaction terms
- remove_features: Drop low-value features (columns param required)
- square_features: Add squared features (columns param optional)
- reciprocal_features: Add 1/x features (columns param optional)

Think step by step. Generate exactly 4 diverse strategies. Consider:
1. Feature Engineering (transformations, interactions, polynomial features, feature removal)
2. Data Preprocessing (scaling methods, handling skewness)
3. Hyperparameter Tuning (specific parameter changes for the model)
4. Combined approaches (transformation + hyperparameter change together)

CRITICAL: Each strategy MUST include a "transforms" field with concrete transformation
actions that the system will automatically execute.

If the RL reward history shows that certain actions were NEGATIVE (hurt performance),
AVOID those actions. If certain actions were POSITIVE, consider repeating or building on them.

Respond with:
{{
    "root_thought": "<current situation summary>",
    "strategies": [
        {{
            "id": "s1",
            "content": "<strategy description>",
            "reasoning": "<step-by-step reasoning why this could work>",
            "category": "<feature|preprocessing|hyperparameter|combined>",
            "expected_impact": <float 0.0-1.0>,
            "confidence": <float 0.0-1.0>,
            "transforms": [
                {{"type": "<transform_name>", ...optional params...}}
            ],
            "hyperparameters": {{<specific hyperparameters to try>}},
            "actionable_steps": ["<concrete step 1>", "<concrete step 2>"]
        }}
    ]
}}"""

EXPAND_STRATEGY_PROMPT = """You are using Tree of Thought reasoning.

STEP 2: EXPAND a promising strategy into specific sub-strategies.

Parent Strategy: {strategy_content}
Category: {strategy_category}
Parent Score: {strategy_score}

Current Best Model: {best_model}
Current {primary_metric}: {best_metric_value}
Task: {task_type}

Feature Engineering State:
{feature_state}

RL Reward History:
{rl_summary}

Run History:
{run_history}

Generate exactly 2 specific, actionable sub-strategies that implement the parent strategy differently.
Each MUST include concrete "transforms" that the system will execute.

Available transforms: standard_scale, minmax_scale, robust_scale, log_transform,
sqrt_transform, polynomial_features, interaction_features, remove_features,
square_features, reciprocal_features.

Respond with:
{{
    "sub_strategies": [
        {{
            "id": "{parent_id}_a",
            "content": "<specific implementation description>",
            "reasoning": "<why this specific approach>",
            "model_name": "<concrete model to use or 'keep_current'>",
            "hyperparameters": {{<specific params if applicable>}},
            "transforms": [
                {{"type": "<transform_name>", ...optional params...}}
            ],
            "expected_impact": <float 0.0-1.0>,
            "confidence": <float 0.0-1.0>
        }}
    ]
}}"""

EVALUATE_THOUGHT_PROMPT = """Evaluate this improvement strategy for an ML model.

Strategy: {thought_content}
Reasoning: {thought_reasoning}

Context:
- Current best: {best_model} with {primary_metric}={best_metric_value}
- Dataset: {num_rows} rows, {num_features} features, task={task_type}
- Iterations completed: {num_iterations}
- Run history summary: {run_summary}

RL Reward Context:
{rl_summary}

Score this strategy on two dimensions:
1. Expected Impact (0.0-1.0): How much improvement is likely?
2. Confidence (0.0-1.0): How sure are you about this estimate?

Consider the RL reward history: if this strategy involves actions that previously
had negative rewards, lower your confidence. If it builds on positive actions, increase it.

Respond with:
{{
    "expected_impact": <float>,
    "confidence": <float>,
    "score": <float, impact * confidence>,
    "justification": "<brief justification>"
}}"""

PRIORITIZE_PROMPT = """Given all the explored strategies and their scores, provide a final prioritized recommendation.

Explored Tree of Thought:
{thought_tree_summary}

Selected Path: {selected_path}

Best Strategy Details:
{best_strategy}

Provide a final recommendation that prioritizes changes by their expected impact on {primary_metric}.

Respond with:
{{
    "primary_recommendation": {{
        "action": "<what to do next>",
        "model_name": "<model to use>",
        "hyperparameters": {{<specific params>}},
        "transforms": [
            {{"type": "<transform_name>", ...optional params...}}
        ],
        "expected_metric_improvement": <float>,
        "priority": "high"
    }},
    "secondary_recommendations": [
        {{
            "action": "<what to try after>",
            "priority": "medium|low",
            "rationale": "<brief reason>"
        }}
    ],
    "insights": [
        "<key insight about the data/models from this analysis>"
    ]
}}"""
