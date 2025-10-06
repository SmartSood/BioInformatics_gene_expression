
export interface Experiment {
  id: string;
  user_id: string;
  name: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  created_at: string;
  updated_at: string;
}

export interface ExperimentParameters {
  id: string;
  experiment_id: string;
  preprocessing_steps: string[];
  model_type: string;
  num_folds: number;
  train_test_split: number;
  feature_selection: string;
  hyperparameters: Record<string, any>;
  created_at: string;
}

export interface Gene {
  symbol: string;
  expression: number;
  pvalue: number;
  foldChange: number;
}

export interface ExperimentResults {
  id: string;
  experiment_id: string;
  top_genes: Gene[];
  accuracy: number;
  precision_score: number;
  recall_score: number;
  f1_score: number;
  roc_auc: number;
  additional_metrics: Record<string, any>;
  created_at: string;
}
