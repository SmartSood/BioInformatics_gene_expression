export interface Experiment {
  id: string;
  user_id: string;
  name: string;
  description: string;
  status: "queued" | "started" | "finished" | "failed" | "running" | "pending" | "completed";
  createdAt: string;
  updatedAt: string;
  // Optional fields returned by the detailed experiment API
  datasetUri?: string | null;
  modelPath?: string | null;
  resultsPath?: string | null;
}

export interface dataset_props {
  columnCount: number;
  createdAt: Date;
  description: string;
  filePath: string;
  id: string;
  name: string;
  rowCount: number;
  updatedAt: Date;
  user: string | null;
  userId: number;
}

export interface ExperimentParameters {
  id?: string;
  experiment_id?: string;
  preprocessing_steps: string[];
  model_type: string;
  problem_type?: "classification" | "regression";
  num_folds: number;
  train_test_split: number;
  feature_selection: string | null;
  hyperparameters: Record<string, any>;
  created_at?: string;
}

export interface Gene {
  symbol: string;
  // These may be null/undefined if the backend only knows that the
  // feature was selected (e.g. from feature selection) but does not
  // have per-gene statistics like expression, p-value or fold-change.
  expression?: number | null;
  pvalue?: number | null;
  foldChange?: number | null;
}

export interface ExperimentResults {
  id?: string;
  experiment_id?: string;
  problem_type?: "classification" | "regression";
  top_genes: Gene[];
  // Classification metrics
  accuracy?: number | null;
  precision_score?: number | null;
  recall_score?: number | null;
  f1_score?: number | null;
  roc_auc?: number | null;
  // Regression metrics
  r2_score?: number | null;
  mse?: number | null;
  rmse?: number | null;
  // Common metrics
  cv_mean?: number | null;
  cv_std?: number | null;
  n_features_original?: number | null;
  n_features_selected?: number | null;
  feature_selection?: any;
  warnings?: string[];
  warnings_count?: number;
  additional_metrics: Record<string, any>;
  created_at?: string;
}
