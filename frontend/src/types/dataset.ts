export interface ColumnInfo {
  name: string;
  dtype: string;
  null_pct: number;
  unique_count: number;
  sample_values: string[];
}

export interface Dataset {
  id: string;
  name: string;
  num_rows: number | null;
  num_cols: number | null;
  target_col: string | null;
  task_type: string | null;
  created_at: string | null;
}

export interface DatasetDetail extends Dataset {
  columns: ColumnInfo[];
  file_path: string | null;
  db_uri: string | null;
}

export interface DatasetVersion {
  id: string;
  dataset_id: string;
  experiment_id: string;
  version: number;
  parent_version: string | null;
  transforms: Record<string, unknown>[];
  num_features: number;
  feature_names: string[];
  created_at: string | null;
}
