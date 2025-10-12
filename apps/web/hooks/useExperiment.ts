import { useState, useEffect } from "react";
import {
  Experiment,
  ExperimentParameters,
  ExperimentResults,
} from "../utils/scemma";

const MOCK_EXPERIMENTS: Experiment[] = [
  {
    id: "1",
    user_id: "00000000-0000-0000-0000-000000000000",
    name: "Cancer Drug Target Analysis",
    description: "Identifying potential drug targets in lung cancer",
    status: "completed",
    created_at: new Date(Date.now() - 86400000).toISOString(),
    updated_at: new Date(Date.now() - 86400000).toISOString(),
  },
  {
    id: "2",
    user_id: "00000000-0000-0000-0000-000000000000",
    name: "Alzheimer Gene Expression",
    description: "Analysis of differential gene expression in AD patients",
    status: "completed",
    created_at: new Date(Date.now() - 172800000).toISOString(),
    updated_at: new Date(Date.now() - 172800000).toISOString(),
  },
];

export function useExperiments() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchExperiments();
  }, []);

  const fetchExperiments = async () => {
    try {
      const stored = sessionStorage.getItem("experiments");
      if (stored) {
        setExperiments(JSON.parse(stored));
      } else {
        sessionStorage.setItem("experiments", JSON.stringify(MOCK_EXPERIMENTS));
        setExperiments(MOCK_EXPERIMENTS);
      }
    } catch (error) {
      console.error("Error fetching experiments:", error);
      setExperiments(MOCK_EXPERIMENTS);
    } finally {
      setLoading(false);
    }
  };

  return { experiments, loading, refetch: fetchExperiments };
}

const MOCK_PARAMETERS: Record<string, ExperimentParameters> = {
  "1": {
    id: "1",
    experiment_id: "1",
    preprocessing_steps: [
      "Normalization",
      "Outlier Removal",
      "Feature Scaling",
    ],
    model_type: "random_forest",
    num_folds: 5,
    train_test_split: 0.8,
    feature_selection: "LASSO",
    hyperparameters: {},
    created_at: new Date().toISOString(),
  },
  "2": {
    id: "2",
    experiment_id: "2",
    preprocessing_steps: ["Normalization", "Batch Correction"],
    model_type: "xgboost",
    num_folds: 10,
    train_test_split: 0.75,
    feature_selection: "Random Forest Importance",
    hyperparameters: {},
    created_at: new Date().toISOString(),
  },
};

const MOCK_RESULTS: Record<string, ExperimentResults> = {
  "1": {
    id: "1",
    experiment_id: "1",
    top_genes: [
      { symbol: "EGFR", expression: 8.5, pvalue: 0.001, foldChange: 3.2 },
      { symbol: "TP53", expression: 7.8, pvalue: 0.002, foldChange: 2.8 },
      { symbol: "KRAS", expression: 7.2, pvalue: 0.005, foldChange: 2.5 },
      { symbol: "ALK", expression: 6.9, pvalue: 0.008, foldChange: 2.3 },
      { symbol: "BRAF", expression: 6.5, pvalue: 0.01, foldChange: 2.1 },
      { symbol: "MET", expression: 6.1, pvalue: 0.012, foldChange: 1.9 },
      { symbol: "ROS1", expression: 5.8, pvalue: 0.015, foldChange: 1.8 },
      { symbol: "PIK3CA", expression: 5.5, pvalue: 0.02, foldChange: 1.7 },
      { symbol: "ERBB2", expression: 5.2, pvalue: 0.025, foldChange: 1.6 },
      { symbol: "NRAS", expression: 4.9, pvalue: 0.03, foldChange: 1.5 },
    ],
    accuracy: 0.89,
    precision_score: 0.87,
    recall_score: 0.85,
    f1_score: 0.86,
    roc_auc: 0.92,
    additional_metrics: {},
    created_at: new Date().toISOString(),
  },
  "2": {
    id: "2",
    experiment_id: "2",
    top_genes: [
      { symbol: "APP", expression: 9.1, pvalue: 0.0001, foldChange: 4.2 },
      { symbol: "APOE", expression: 8.7, pvalue: 0.0005, foldChange: 3.8 },
      { symbol: "PSEN1", expression: 8.3, pvalue: 0.001, foldChange: 3.5 },
      { symbol: "PSEN2", expression: 7.9, pvalue: 0.002, foldChange: 3.2 },
      { symbol: "MAPT", expression: 7.5, pvalue: 0.003, foldChange: 2.9 },
      { symbol: "TREM2", expression: 7.1, pvalue: 0.005, foldChange: 2.6 },
      { symbol: "CLU", expression: 6.7, pvalue: 0.008, foldChange: 2.4 },
      { symbol: "SORL1", expression: 6.3, pvalue: 0.01, foldChange: 2.2 },
      { symbol: "BIN1", expression: 5.9, pvalue: 0.015, foldChange: 2.0 },
      { symbol: "CD33", expression: 5.5, pvalue: 0.02, foldChange: 1.8 },
    ],
    accuracy: 0.91,
    precision_score: 0.89,
    recall_score: 0.88,
    f1_score: 0.885,
    roc_auc: 0.94,
    additional_metrics: {},
    created_at: new Date().toISOString(),
  },
};

export function useExperimentDetails(experimentId: string | null) {
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [parameters, setParameters] = useState<ExperimentParameters | null>(
    null
  );
  const [results, setResults] = useState<ExperimentResults | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (experimentId) {
      fetchExperimentDetails(experimentId);
    } else {
      setExperiment(null);
      setParameters(null);
      setResults(null);
    }
  }, [experimentId]);

  const fetchExperimentDetails = async (id: string) => {
    setLoading(true);
    try {
      const stored = sessionStorage.getItem("experiments");
      const experiments = stored ? JSON.parse(stored) : MOCK_EXPERIMENTS;
      const exp = experiments.find((e: Experiment) => e.id === id);

      const paramsKey = `params_${id}`;
      const resultsKey = `results_${id}`;

      const storedParams = sessionStorage.getItem(paramsKey);
      const storedResults = sessionStorage.getItem(resultsKey);

      setExperiment(exp || null);
      setParameters(
        storedParams ? JSON.parse(storedParams) : MOCK_PARAMETERS[id] || null
      );
      setResults(
        storedResults ? JSON.parse(storedResults) : MOCK_RESULTS[id] || null
      );
    } catch (error) {
      console.error("Error fetching experiment details:", error);
    } finally {
      setLoading(false);
    }
  };

  return { experiment, parameters, results, loading };
}
