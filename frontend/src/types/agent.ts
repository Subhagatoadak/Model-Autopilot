export interface ThoughtNode {
  id: string;
  content: string;
  reasoning: string;
  expected_impact: number;
  confidence: number;
  score: number;
  children: ThoughtNode[];
  selected: boolean;
  depth: number;
  iteration?: number;
}

export interface ThoughtTree {
  root_thought: string;
  branches: ThoughtNode[];
  selected_path: string[];
  iteration: number;
}

export interface AgentStep {
  agent: string;
  step: string;
  content: string;
  timestamp: number;
}

export type WSMessage =
  | { type: 'agent_step'; data: { agent: string; step: string; content: string } }
  | { type: 'thought_node'; data: ThoughtNode & { iteration: number } }
  | { type: 'thought_selected'; data: { path: string[]; iteration: number } }
  | { type: 'run_complete'; data: { iteration: number; model_name: string; metrics: Record<string, number>; train_time_s: number; ucb_score: number } }
  | { type: 'metrics_update'; data: { iteration: number; metrics: Record<string, number> } }
  | { type: 'experiment_status'; data: { status: string; iteration?: number; max_iterations?: number; best_metrics?: Record<string, number>; error?: string } };

export interface ModelInfo {
  name: string;
  task_types: string[];
  default_params: Record<string, unknown>;
}
