export interface ExperimentConfig {
  dataset_id: string;
  target_col: string;
  feature_cols?: string[];
  task_type?: string;
  primary_metric?: string;
  secondary_metrics?: string[];
  strategy?: string;
  epsilon?: number;
  max_iterations?: number;
  selected_models?: string[];
  test_size?: number;
  cross_validation?: boolean;
  cv_folds?: number;
  multi_objective?: boolean;
  objective_weights?: Record<string, number>;
}

export interface Experiment {
  id: string;
  dataset_id: string;
  status: string;
  strategy: string;
  epsilon: number;
  created_at: string | null;
  finished_at: string | null;
}

export interface ExperimentRun {
  id: string;
  experiment_id: string;
  iteration: number;
  model_name: string;
  hyperparameters: Record<string, unknown>;
  metrics: Record<string, number>;
  train_time_s: number | null;
  ucb_score: number | null;
  created_at: string | null;
}

export interface ExperimentDetail extends Experiment {
  config: ExperimentConfig | null;
  runs: ExperimentRun[];
  best_metrics: Record<string, number>;
}

export interface ExperimentComparison {
  experiment_1: ExperimentDetail;
  experiment_2: ExperimentDetail;
}
