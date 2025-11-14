import { useState } from "react";
import axios from "axios";
import { X, Sparkles, Beaker, ChevronDown, ChevronUp } from "lucide-react";
import { MODEL_BACKEND_URL } from "@repo/config";
import {dataset_props} from '../../../utils/scemma';
interface NewExperimentFormProps {
  datasetId: string;
  dataset: dataset_props;
  onClose: () => void;
  onSuccess: () => void;
}

/**
 * Options and labels derived from:
 * - scemma.model (Pydantic schema)
 * - pipeline.py (MODEL_MAP, feature selection builders, preprocessing expectations)
 *
 * We've kept your UI unchanged but use precise keys and defaults that the backend expects.
 */

// Preprocessing "steps" map to top-level Preprocessing fields in Pydantic schema
const PREPROCESSING_STEPS: { key: string; label: string }[] = [
  { key: "batch_correction", label: "Batch Correction" },
  { key: "missing_values", label: "Missing Value Imputation" },
  { key: "outlier_removal", label: "Outlier Removal" },
  { key: "scaling", label: "Feature Scaling" },
  { key: "log_transform", label: "Log Transformation" },
  { key: "qc_filtering", label: "Quality Control Filtering" },
  { key: "encoding", label: "Encoding" },
  { key: "feature_selection", label: "Feature Selection" },
];

// Models come from pipeline.MODEL_MAP keys
const MODEL_OPTIONS = [
  { value: "random_forest", label: "Random Forest" },
  { value: "svm", label: "Support Vector Machine" },
  { value: "neural_network", label: "Neural Network" },
  { value: "gradient_boosting", label: "Gradient Boosting" },
  { value: "logistic_regression", label: "Logistic Regression" },
  { value: "xgboost", label: "XGBoost" },
];

// Feature selection options follow FeatureSelectionMethod in schema + pipeline support
const FEATURE_SELECTION_OPTIONS = [
  { value: "none", label: "None" },
  { value: "variance_threshold", label: "Variance Threshold" },
  { value: "rfe", label: "RFE (recursive feature elimination)" },
  { value: "lasso", label: "LASSO (L1) selection" },
  { value: "random_forest_importance", label: "Random Forest Importance" },
  { value: "chi2", label: "Chi-Square Test (non-negative features only)" },
];

// Batch correction method options
const BATCH_CORRECTION_METHODS = [
  { value: "none", label: "None" },
  { value: "combat", label: "ComBat" },
  { value: "zscore", label: "Z-Score Normalization" },
  { value: "ratio", label: "Ratio Method" },
];

// Missing value imputation strategies
const NUMERIC_IMPUTATION_STRATEGIES = [
  { value: "mean", label: "Mean" },
  { value: "median", label: "Median" },
  { value: "most_frequent", label: "Most Frequent" },
  { value: "constant", label: "Constant" },
];

const CATEGORICAL_IMPUTATION_STRATEGIES = [
  { value: "most_frequent", label: "Most Frequent" },
  { value: "constant", label: "Constant" },
];

// Outlier removal methods
const OUTLIER_REMOVAL_METHODS = [
  { value: "none", label: "None" },
  { value: "iqr", label: "IQR (Interquartile Range)" },
  { value: "zscore", label: "Z-Score" },
  { value: "percentile", label: "Percentile" },
];

// Scaling methods
const SCALING_METHODS = [
  { value: "none", label: "None" },
  { value: "standard", label: "Standard (Z-score)" },
  { value: "minmax", label: "Min-Max" },
  { value: "robust", label: "Robust" },
  { value: "maxabs", label: "Max Absolute" },
];

// Encoding methods
const ENCODING_METHODS = [
  { value: "onehot", label: "One-Hot Encoding" },
  { value: "ordinal", label: "Ordinal Encoding" },
  { value: "none", label: "None" },
];

export function NewExperimentForm({
  datasetId,
  dataset,
  onClose,
  onSuccess,
}: NewExperimentFormProps) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [modelType, setModelType] = useState("random_forest");
  const [problemType, setProblemType] = useState<"classification" | "regression">("classification");
  const [numFolds, setNumFolds] = useState(5);
  const [trainTestSplit, setTrainTestSplit] = useState(80);
  const [featureSelection, setFeatureSelection] = useState("none");
  const [selectedPreprocessing, setSelectedPreprocessing] = useState<
    string[]
  >([]);
  const [targetVariable, setTargetVariable] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [expandedConfigs, setExpandedConfigs] = useState<Record<string, boolean>>({});
  const [showHyperparams, setShowHyperparams] = useState(false);
  const [hyperparams, setHyperparams] = useState<Array<{ key: string; value: string }>>([]);

  // Configuration states for each preprocessing step
  const [batchCorrectionConfig, setBatchCorrectionConfig] = useState({
    enabled: false,
    method: "combat" as "none" | "combat" | "zscore" | "ratio",
    batch_column: "",
  });

  const [missingValuesConfig, setMissingValuesConfig] = useState({
    strategy_numeric: "median" as "mean" | "median" | "most_frequent" | "constant",
    strategy_categorical: "most_frequent" as "most_frequent" | "constant",
    fill_value_numeric: "",
    fill_value_categorical: "",
    drop_rows: false,
  });

  const [outlierRemovalConfig, setOutlierRemovalConfig] = useState({
    method: "iqr" as "none" | "iqr" | "zscore" | "percentile",
    iqr_factor: 1.5,
    zscore_threshold: 3.0,
    percentile_min: 0.5,
    percentile_max: 99.5,
    cap_outliers: false,
  });

  const [scalingConfig, setScalingConfig] = useState({
    method: "standard" as "none" | "standard" | "minmax" | "robust" | "maxabs",
    feature_range_min: 0.0,
    feature_range_max: 1.0,
    apply_to: "numeric_only" as "numeric_only" | "all",
  });

  const [logTransformConfig, setLogTransformConfig] = useState({
    enabled: false,
    offset: 1.0,
    columns: "",
  });

  const [qcFilteringConfig, setQcFilteringConfig] = useState({
    enabled: false,
    max_missing_fraction: 0.2,
    numeric_range: "",
  });

  const [encodingConfig, setEncodingConfig] = useState({
    method: "onehot" as "onehot" | "ordinal" | "none",
    drop_first: false,
  });

  const [featureSelectionConfig, setFeatureSelectionConfig] = useState({
    method: "none" as string,
    k_features: "",
    variance_threshold: 0.0,
    alpha: 0.001,
    importance_threshold: "",
  });

  const togglePreprocessing = (step: string) => {
    setSelectedPreprocessing((prev) => {
      const isCurrentlySelected = prev.includes(step);
      const newSelection = isCurrentlySelected
        ? prev.filter((s) => s !== step)
        : [...prev, step];
      
      // Update enabled state for steps that have it
      if (step === "batch_correction") {
        setBatchCorrectionConfig(prev => ({ 
          ...prev, 
          enabled: !isCurrentlySelected,
          method: !isCurrentlySelected && prev.method === "none" ? "combat" : prev.method
        }));
      } else if (step === "log_transform") {
        setLogTransformConfig(prev => ({ ...prev, enabled: !isCurrentlySelected }));
      } else if (step === "qc_filtering") {
        setQcFilteringConfig(prev => ({ ...prev, enabled: !isCurrentlySelected }));
      }
      
      // Toggle expanded state when enabling
      if (!isCurrentlySelected) {
        setExpandedConfigs(prev => ({ ...prev, [step]: true }));
      }
      
      return newSelection;
    });
  };

  const toggleConfigExpanded = (step: string) => {
    setExpandedConfigs(prev => ({ ...prev, [step]: !prev[step] }));
  };

  /**
   * Build a preprocessing payload that matches the Pydantic model:
   * Preprocessing {
   *   missing_values: ImputationConfig,
   *   outlier_removal: OutlierRemovalConfig,
   *   scaling: ScalingConfig,
   *   log_transform: LogTransformConfig,
   *   batch_correction: BatchCorrectionConfig,
   *   qc_filtering: QCFilteringConfig,
   *   encoding: EncodingConfig,
   *   feature_selection: FeatureSelectionConfig
   * }
   *
   * The function uses the user's selected configuration values.
   */
  const buildPreprocessingPayload = () => {
    const payload: any = {
      missing_values: {
        strategy_numeric: missingValuesConfig.strategy_numeric,
        strategy_categorical: missingValuesConfig.strategy_categorical,
        fill_value_numeric: missingValuesConfig.fill_value_numeric 
          ? parseFloat(missingValuesConfig.fill_value_numeric) 
          : null,
        fill_value_categorical: missingValuesConfig.fill_value_categorical || null,
        drop_rows: missingValuesConfig.drop_rows,
      },
      outlier_removal: {
        method: selectedPreprocessing.includes("outlier_removal") 
          ? outlierRemovalConfig.method 
          : "none",
        iqr_factor: outlierRemovalConfig.iqr_factor,
        zscore_threshold: outlierRemovalConfig.zscore_threshold,
        percentile_min: outlierRemovalConfig.percentile_min,
        percentile_max: outlierRemovalConfig.percentile_max,
        cap_outliers: outlierRemovalConfig.cap_outliers,
      },
      scaling: {
        method: selectedPreprocessing.includes("scaling") 
          ? scalingConfig.method 
          : "none",
        feature_range: [scalingConfig.feature_range_min, scalingConfig.feature_range_max],
        apply_to: scalingConfig.apply_to,
      },
      log_transform: {
        enabled: logTransformConfig.enabled,
        offset: logTransformConfig.offset,
        columns: logTransformConfig.columns 
          ? logTransformConfig.columns.split(",").map(c => c.trim()).filter(c => c)
          : null,
      },
      batch_correction: {
        enabled: batchCorrectionConfig.enabled,
        method: batchCorrectionConfig.enabled 
          ? (batchCorrectionConfig.method === "none" ? "combat" : batchCorrectionConfig.method)
          : "none",
        batch_column: batchCorrectionConfig.enabled && batchCorrectionConfig.batch_column 
          ? batchCorrectionConfig.batch_column 
          : null,
      },
      qc_filtering: {
        enabled: qcFilteringConfig.enabled,
        max_missing_fraction: qcFilteringConfig.max_missing_fraction,
        numeric_range: qcFilteringConfig.numeric_range 
          ? JSON.parse(qcFilteringConfig.numeric_range)
          : null,
      },
      encoding: {
        method: encodingConfig.method,
        drop_first: encodingConfig.drop_first,
      },
      feature_selection: {
        method: featureSelection,
        k_features: featureSelectionConfig.k_features 
          ? parseInt(featureSelectionConfig.k_features) 
          : null,
        variance_threshold: featureSelectionConfig.variance_threshold,
        alpha: featureSelectionConfig.alpha,
        importance_threshold: featureSelectionConfig.importance_threshold 
          ? parseFloat(featureSelectionConfig.importance_threshold) 
          : null,
      },
    };

    return payload;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);

    try {
      // dataset_uri fallback: allow callers to set dataset-specific URI in sessionStorage,
      // otherwise use the datasetId itself (backend should resolve)
      
      const dataset_uri = dataset.filePath;

      // default target (you should add a UI field later to choose this)
      

      // Build hyperparams object - parse numeric values where appropriate
      const parsedHyperparams: Record<string, any> = {};
      for (const { key, value } of hyperparams) {
        if (!key.trim() || !value.trim()) continue; // Skip empty values
        
        // Try to parse as number, if fails keep as string
        const numValue = parseFloat(value);
        if (!isNaN(numValue) && isFinite(numValue)) {
          // Check if it's an integer
          if (Number.isInteger(numValue)) {
            parsedHyperparams[key.trim()] = parseInt(value, 10);
          } else {
            parsedHyperparams[key.trim()] = numValue;
          }
        } else if (value.toLowerCase() === "true") {
          parsedHyperparams[key.trim()] = true;
        } else if (value.toLowerCase() === "false") {
          parsedHyperparams[key.trim()] = false;
        } else {
          parsedHyperparams[key.trim()] = value;
        }
      }

      const trainRequest = {
        dataset_id: datasetId,
        dataset_uri,
        config: {
          target: targetVariable,
          problem_type: problemType,
          preprocessing: buildPreprocessingPayload(),
          model: modelType,
          hyperparams: parsedHyperparams,
          split: {
            // backend expects test_size as fraction (pipeline uses test_size)
            test_size: Number((1 - trainTestSplit / 100).toFixed(3)),
            cv_folds: numFolds,
            random_state: 42,
          },
        },
        name: name || `Experiment ${Date.now()}`,
        description: description || "",
      };

      // backend base URL (set via env var in your app) or default to localhost
      const MODEL_BACKEND_URL =
        (process.env.NEXT_PUBLIC_MODEL_BACKEND_URL as string) ||
        "http://localhost:8000";

      const token = sessionStorage.getItem("authToken") || undefined;

      const headers: any = { "Content-Type": "application/json" };
      if (token) headers["Authorization"] = `Bearer ${token}`;

      const resp = await axios.post(`${MODEL_BACKEND_URL}/train`, trainRequest, {
        headers,
      });
      //@ts-ignore
      const jobId = resp?.data?.job_id ?? Date.now().toString();

      // Persist experiment + params + placeholder results in sessionStorage (like before)
      const now = new Date().toISOString();
      const newExperiment = {
        id: jobId,
        user_id:
          sessionStorage.getItem("currentUser") ||
          "00000000-0000-0000-0000-000000000000",
        dataset_id: datasetId,
        name: trainRequest.name,
        description: trainRequest.description,
        status: "queued",
        created_at: now,
        updated_at: now,
      };

      const newParams = {
        id: jobId,
        experiment_id: jobId,
        preprocessing_steps: selectedPreprocessing,
        model_type: modelType,
        num_folds: numFolds,
        train_test_split: trainTestSplit / 100,
        feature_selection: featureSelection,
        hyperparameters: {},
        created_at: now,
        raw_train_request: trainRequest,
      };

      const newResults = {
        id: jobId,
        experiment_id: jobId,
        top_genes: [],
        accuracy: null,
        precision_score: null,
        recall_score: null,
        f1_score: null,
        roc_auc: null,
        additional_metrics: {},
        created_at: now,
      };

      const stored = sessionStorage.getItem("experiments");
      const experiments = stored ? JSON.parse(stored) : [];
      experiments.unshift(newExperiment);
      sessionStorage.setItem("experiments", JSON.stringify(experiments));
      sessionStorage.setItem(`params_${jobId}`, JSON.stringify(newParams));
      sessionStorage.setItem(`results_${jobId}`, JSON.stringify(newResults));

      onSuccess();
      onClose();
    } catch (error: any) {
      console.error("Error creating experiment:", error?.response?.data || error);
      alert(
        "Failed to create experiment. Check console for details and ensure the backend is reachable."
      );
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto border border-slate-700/50">
        <div className="sticky top-0 bg-gradient-to-r from-slate-800 to-slate-900 border-b border-slate-700/50 p-6 flex items-center justify-between z-10">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-teal-500/20 to-blue-500/20 rounded-lg">
              <Sparkles className="w-6 h-6 text-teal-400" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">New Analysis</h2>
              <p className="text-sm text-slate-400">
                Configure your drug target interaction experiment
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors"
          >
            <X className="w-6 h-6 text-slate-400" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          <div className="grid grid-cols-2 gap-6">
            <div className="col-span-2">
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Experiment Name
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500/50"
                placeholder="e.g., Cancer Drug Target Analysis"
              />
            </div>

            <div className="col-span-2">
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Description (Optional)
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={3}
                className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500/50 resize-none"
                placeholder="Describe your experiment..."
              />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Target Column
              </label>
              <input
                type="text"
                value={targetVariable}
                onChange={(e) => setTargetVariable(e.target.value)}
                className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500/50"
                placeholder="Enter the target column (e.g. Y)"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Problem Type
              </label>
              <select
                value={problemType}
                onChange={(e) => setProblemType(e.target.value as "classification" | "regression")}
                className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500/50"
              >
                <option value="classification">Classification</option>
                <option value="regression">Regression</option>
              </select>
            </div>
          </div>

          <div className="border-t border-slate-700/50 pt-6">
            <div className="flex items-center gap-2 mb-4">
              <Beaker className="w-5 h-5 text-purple-400" />
              <h3 className="text-lg font-semibold text-white">
                Preprocessing Steps
              </h3>
            </div>
            <div className="space-y-3">
              {PREPROCESSING_STEPS.map((step) => (
                <div
                  key={step.key}
                  className={`rounded-lg border transition-all ${
                    selectedPreprocessing.includes(step.key)
                      ? "bg-purple-500/10 border-purple-500/50"
                      : "bg-slate-700/30 border-slate-600/50"
                  }`}
                >
                  <label
                    className={`flex items-center gap-3 p-3 cursor-pointer ${
                      selectedPreprocessing.includes(step.key)
                        ? "text-purple-300"
                        : "text-slate-300 hover:bg-slate-700/50"
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={selectedPreprocessing.includes(step.key)}
                      onChange={() => togglePreprocessing(step.key)}
                      className="w-4 h-4 rounded border-slate-500 text-teal-500 focus:ring-teal-500/50 bg-slate-700"
                    />
                    <span className="text-sm font-medium flex-1">{step.label}</span>
                    {selectedPreprocessing.includes(step.key) && (
                      <button
                        type="button"
                        onClick={() => toggleConfigExpanded(step.key)}
                        className="p-1 hover:bg-slate-700/50 rounded transition-colors"
                      >
                        {expandedConfigs[step.key] ? (
                          <ChevronUp className="w-4 h-4" />
                        ) : (
                          <ChevronDown className="w-4 h-4" />
                        )}
                      </button>
                    )}
                  </label>
                  
                  {/* Configuration Panel */}
                  {selectedPreprocessing.includes(step.key) && expandedConfigs[step.key] && (
                    <div className="p-4 pt-0 border-t border-slate-600/50 space-y-4">
                      {/* Batch Correction Configuration */}
                      {step.key === "batch_correction" && (
                        <>
                          <div>
                            <label className="block text-sm font-medium text-slate-300 mb-2">
                              Method
                            </label>
                            <select
                              value={batchCorrectionConfig.method}
                              onChange={(e) =>
                                setBatchCorrectionConfig((prev) => ({
                                  ...prev,
                                  method: e.target.value as typeof prev.method,
                                }))
                              }
                              className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                            >
                              {BATCH_CORRECTION_METHODS.map((method) => (
                                <option key={method.value} value={method.value}>
                                  {method.label}
                                </option>
                              ))}
                            </select>
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-slate-300 mb-2">
                              Batch Column
                            </label>
                            <input
                              type="text"
                              value={batchCorrectionConfig.batch_column}
                              onChange={(e) =>
                                setBatchCorrectionConfig((prev) => ({
                                  ...prev,
                                  batch_column: e.target.value,
                                }))
                              }
                              placeholder="Enter batch column name"
                              className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                            />
                          </div>
                        </>
                      )}

                      {/* Missing Values Configuration */}
                      {step.key === "missing_values" && (
                        <>
                          <div className="grid grid-cols-2 gap-4">
                            <div>
                              <label className="block text-sm font-medium text-slate-300 mb-2">
                                Numeric Strategy
                              </label>
                              <select
                                value={missingValuesConfig.strategy_numeric}
                                onChange={(e) =>
                                  setMissingValuesConfig((prev) => ({
                                    ...prev,
                                    strategy_numeric: e.target.value as typeof prev.strategy_numeric,
                                  }))
                                }
                                className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                              >
                                {NUMERIC_IMPUTATION_STRATEGIES.map((strategy) => (
                                  <option key={strategy.value} value={strategy.value}>
                                    {strategy.label}
                                  </option>
                                ))}
                              </select>
                            </div>
                            <div>
                              <label className="block text-sm font-medium text-slate-300 mb-2">
                                Categorical Strategy
                              </label>
                              <select
                                value={missingValuesConfig.strategy_categorical}
                                onChange={(e) =>
                                  setMissingValuesConfig((prev) => ({
                                    ...prev,
                                    strategy_categorical: e.target.value as typeof prev.strategy_categorical,
                                  }))
                                }
                                className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                              >
                                {CATEGORICAL_IMPUTATION_STRATEGIES.map((strategy) => (
                                  <option key={strategy.value} value={strategy.value}>
                                    {strategy.label}
                                  </option>
                                ))}
                              </select>
                            </div>
                          </div>
                          <div className="grid grid-cols-2 gap-4">
                            <div>
                              <label className="block text-sm font-medium text-slate-300 mb-2">
                                Numeric Fill Value (optional)
                              </label>
                              <input
                                type="number"
                                step="any"
                                value={missingValuesConfig.fill_value_numeric}
                                onChange={(e) =>
                                  setMissingValuesConfig((prev) => ({
                                    ...prev,
                                    fill_value_numeric: e.target.value,
                                  }))
                                }
                                placeholder="For constant strategy"
                                className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                              />
                            </div>
                            <div>
                              <label className="block text-sm font-medium text-slate-300 mb-2">
                                Categorical Fill Value (optional)
                              </label>
                              <input
                                type="text"
                                value={missingValuesConfig.fill_value_categorical}
                                onChange={(e) =>
                                  setMissingValuesConfig((prev) => ({
                                    ...prev,
                                    fill_value_categorical: e.target.value,
                                  }))
                                }
                                placeholder="For constant strategy"
                                className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                              />
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <input
                              type="checkbox"
                              checked={missingValuesConfig.drop_rows}
                              onChange={(e) =>
                                setMissingValuesConfig((prev) => ({
                                  ...prev,
                                  drop_rows: e.target.checked,
                                }))
                              }
                              className="w-4 h-4 rounded border-slate-500 text-teal-500 focus:ring-teal-500/50 bg-slate-700"
                            />
                            <label className="text-sm text-slate-300">
                              Drop rows with missing values
                            </label>
                          </div>
                        </>
                      )}

                      {/* Outlier Removal Configuration */}
                      {step.key === "outlier_removal" && (
                        <>
                          <div>
                            <label className="block text-sm font-medium text-slate-300 mb-2">
                              Method
                            </label>
                            <select
                              value={outlierRemovalConfig.method}
                              onChange={(e) =>
                                setOutlierRemovalConfig((prev) => ({
                                  ...prev,
                                  method: e.target.value as typeof prev.method,
                                }))
                              }
                              className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                            >
                              {OUTLIER_REMOVAL_METHODS.map((method) => (
                                <option key={method.value} value={method.value}>
                                  {method.label}
                                </option>
                              ))}
                            </select>
                          </div>
                          <div className="grid grid-cols-2 gap-4">
                            <div>
                              <label className="block text-sm font-medium text-slate-300 mb-2">
                                IQR Factor
                              </label>
                              <input
                                type="number"
                                step="0.1"
                                value={outlierRemovalConfig.iqr_factor}
                                onChange={(e) =>
                                  setOutlierRemovalConfig((prev) => ({
                                    ...prev,
                                    iqr_factor: parseFloat(e.target.value) || 1.5,
                                  }))
                                }
                                className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                              />
                            </div>
                            <div>
                              <label className="block text-sm font-medium text-slate-300 mb-2">
                                Z-Score Threshold
                              </label>
                              <input
                                type="number"
                                step="0.1"
                                value={outlierRemovalConfig.zscore_threshold}
                                onChange={(e) =>
                                  setOutlierRemovalConfig((prev) => ({
                                    ...prev,
                                    zscore_threshold: parseFloat(e.target.value) || 3.0,
                                  }))
                                }
                                className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                              />
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <input
                              type="checkbox"
                              checked={outlierRemovalConfig.cap_outliers}
                              onChange={(e) =>
                                setOutlierRemovalConfig((prev) => ({
                                  ...prev,
                                  cap_outliers: e.target.checked,
                                }))
                              }
                              className="w-4 h-4 rounded border-slate-500 text-teal-500 focus:ring-teal-500/50 bg-slate-700"
                            />
                            <label className="text-sm text-slate-300">
                              Cap outliers instead of removing
                            </label>
                          </div>
                        </>
                      )}

                      {/* Scaling Configuration */}
                      {step.key === "scaling" && (
                        <>
                          <div>
                            <label className="block text-sm font-medium text-slate-300 mb-2">
                              Method
                            </label>
                            <select
                              value={scalingConfig.method}
                              onChange={(e) =>
                                setScalingConfig((prev) => ({
                                  ...prev,
                                  method: e.target.value as typeof prev.method,
                                }))
                              }
                              className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                            >
                              {SCALING_METHODS.map((method) => (
                                <option key={method.value} value={method.value}>
                                  {method.label}
                                </option>
                              ))}
                            </select>
                          </div>
                          {scalingConfig.method === "minmax" && (
                            <div className="grid grid-cols-2 gap-4">
                              <div>
                                <label className="block text-sm font-medium text-slate-300 mb-2">
                                  Min Value
                                </label>
                                <input
                                  type="number"
                                  step="0.1"
                                  value={scalingConfig.feature_range_min}
                                  onChange={(e) =>
                                    setScalingConfig((prev) => ({
                                      ...prev,
                                      feature_range_min: parseFloat(e.target.value) || 0.0,
                                    }))
                                  }
                                  className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                                />
                              </div>
                              <div>
                                <label className="block text-sm font-medium text-slate-300 mb-2">
                                  Max Value
                                </label>
                                <input
                                  type="number"
                                  step="0.1"
                                  value={scalingConfig.feature_range_max}
                                  onChange={(e) =>
                                    setScalingConfig((prev) => ({
                                      ...prev,
                                      feature_range_max: parseFloat(e.target.value) || 1.0,
                                    }))
                                  }
                                  className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                                />
                              </div>
                            </div>
                          )}
                          <div>
                            <label className="block text-sm font-medium text-slate-300 mb-2">
                              Apply To
                            </label>
                            <select
                              value={scalingConfig.apply_to}
                              onChange={(e) =>
                                setScalingConfig((prev) => ({
                                  ...prev,
                                  apply_to: e.target.value as typeof prev.apply_to,
                                }))
                              }
                              className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                            >
                              <option value="numeric_only">Numeric Only</option>
                              <option value="all">All Features</option>
                            </select>
                          </div>
                        </>
                      )}

                      {/* Log Transform Configuration */}
                      {step.key === "log_transform" && (
                        <>
                          <div>
                            <label className="block text-sm font-medium text-slate-300 mb-2">
                              Offset
                            </label>
                            <input
                              type="number"
                              step="0.1"
                              value={logTransformConfig.offset}
                              onChange={(e) =>
                                setLogTransformConfig((prev) => ({
                                  ...prev,
                                  offset: parseFloat(e.target.value) || 1.0,
                                }))
                              }
                              className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                            />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-slate-300 mb-2">
                              Columns (comma-separated, leave empty for all numeric)
                            </label>
                            <input
                              type="text"
                              value={logTransformConfig.columns}
                              onChange={(e) =>
                                setLogTransformConfig((prev) => ({
                                  ...prev,
                                  columns: e.target.value,
                                }))
                              }
                              placeholder="col1, col2, col3"
                              className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                            />
                          </div>
                        </>
                      )}

                      {/* QC Filtering Configuration */}
                      {step.key === "qc_filtering" && (
                        <>
                          <div>
                            <label className="block text-sm font-medium text-slate-300 mb-2">
                              Max Missing Fraction
                            </label>
                            <input
                              type="number"
                              step="0.01"
                              min="0"
                              max="1"
                              value={qcFilteringConfig.max_missing_fraction}
                              onChange={(e) =>
                                setQcFilteringConfig((prev) => ({
                                  ...prev,
                                  max_missing_fraction: parseFloat(e.target.value) || 0.2,
                                }))
                              }
                              className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                            />
                          </div>
                        </>
                      )}

                      {/* Encoding Configuration */}
                      {step.key === "encoding" && (
                        <>
                          <div>
                            <label className="block text-sm font-medium text-slate-300 mb-2">
                              Method
                            </label>
                            <select
                              value={encodingConfig.method}
                              onChange={(e) =>
                                setEncodingConfig((prev) => ({
                                  ...prev,
                                  method: e.target.value as typeof prev.method,
                                }))
                              }
                              className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                            >
                              {ENCODING_METHODS.map((method) => (
                                <option key={method.value} value={method.value}>
                                  {method.label}
                                </option>
                              ))}
                            </select>
                          </div>
                          <div className="flex items-center gap-2">
                            <input
                              type="checkbox"
                              checked={encodingConfig.drop_first}
                              onChange={(e) =>
                                setEncodingConfig((prev) => ({
                                  ...prev,
                                  drop_first: e.target.checked,
                                }))
                              }
                              className="w-4 h-4 rounded border-slate-500 text-teal-500 focus:ring-teal-500/50 bg-slate-700"
                            />
                            <label className="text-sm text-slate-300">
                              Drop first category (for one-hot encoding)
                            </label>
                          </div>
                        </>
                      )}

                      {/* Feature Selection Configuration */}
                      {step.key === "feature_selection" && (
                        <>
                          <div>
                            <label className="block text-sm font-medium text-slate-300 mb-2">
                              Method
                            </label>
                            <select
                              value={featureSelection}
                              onChange={(e) => setFeatureSelection(e.target.value)}
                              className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                            >
                              {FEATURE_SELECTION_OPTIONS.map((method) => (
                                <option key={method.value} value={method.value}>
                                  {method.label}
                                </option>
                              ))}
                            </select>
                          </div>
                          {(featureSelection === "rfe" || featureSelection === "chi2") && (
                            <div>
                              <label className="block text-sm font-medium text-slate-300 mb-2">
                                Number of Features (k_features)
                              </label>
                              <input
                                type="number"
                                value={featureSelectionConfig.k_features}
                                onChange={(e) =>
                                  setFeatureSelectionConfig((prev) => ({
                                    ...prev,
                                    k_features: e.target.value,
                                  }))
                                }
                                placeholder="Leave empty for auto"
                                className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                              />
                            </div>
                          )}
                          {featureSelection === "variance_threshold" && (
                            <div>
                              <label className="block text-sm font-medium text-slate-300 mb-2">
                                Variance Threshold
                              </label>
                              <input
                                type="number"
                                step="0.001"
                                value={featureSelectionConfig.variance_threshold}
                                onChange={(e) =>
                                  setFeatureSelectionConfig((prev) => ({
                                    ...prev,
                                    variance_threshold: parseFloat(e.target.value) || 0.0,
                                  }))
                                }
                                className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                              />
                            </div>
                          )}
                          {featureSelection === "lasso" && (
                            <div>
                              <label className="block text-sm font-medium text-slate-300 mb-2">
                                Alpha (L1 Regularization)
                              </label>
                              <input
                                type="number"
                                step="0.001"
                                value={featureSelectionConfig.alpha}
                                onChange={(e) =>
                                  setFeatureSelectionConfig((prev) => ({
                                    ...prev,
                                    alpha: parseFloat(e.target.value) || 0.001,
                                  }))
                                }
                                className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                              />
                            </div>
                          )}
                          {featureSelection === "random_forest_importance" && (
                            <div>
                              <label className="block text-sm font-medium text-slate-300 mb-2">
                                Importance Threshold
                              </label>
                              <input
                                type="number"
                                step="0.01"
                                value={featureSelectionConfig.importance_threshold}
                                onChange={(e) =>
                                  setFeatureSelectionConfig((prev) => ({
                                    ...prev,
                                    importance_threshold: e.target.value,
                                  }))
                                }
                                placeholder="Leave empty for auto"
                                className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                              />
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Model Type
              </label>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500/50"
              >
                {MODEL_OPTIONS.map((model) => (
                  <option key={model.value} value={model.value}>
                    {model.label}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Cross-Validation Folds
              </label>
              <input
                type="number"
                value={numFolds}
                onChange={(e) => setNumFolds(parseInt(e.target.value))}
                min={2}
                max={20}
                className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500/50"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Train/Test Split (%)
              </label>
              <input
                type="number"
                value={trainTestSplit}
                onChange={(e) => setTrainTestSplit(parseInt(e.target.value))}
                min={50}
                max={95}
                className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500/50"
              />
            </div>
          </div>

          {/* Hyperparameters Section */}
          <div className="border-t border-slate-700/50 pt-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <h3 className="text-lg font-semibold text-white">
                  Hyperparameters (Optional)
                </h3>
                <span className="text-xs text-slate-400">
                  Enter as key-value pairs (e.g., n_estimators: 100, max_depth: 10)
                </span>
              </div>
              <button
                type="button"
                onClick={() => setShowHyperparams(!showHyperparams)}
                className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors text-slate-300"
              >
                {showHyperparams ? (
                  <ChevronUp className="w-5 h-5" />
                ) : (
                  <ChevronDown className="w-5 h-5" />
                )}
              </button>
            </div>

            {showHyperparams && (
              <div className="space-y-3">
                <div className="text-sm text-slate-400 mb-4">
                  <p className="mb-2">Common hyperparameters by model:</p>
                  <ul className="list-disc list-inside space-y-1 ml-2">
                    <li><strong>Random Forest:</strong> n_estimators, max_depth, min_samples_split, min_samples_leaf</li>
                    <li><strong>SVM:</strong> C, kernel, gamma</li>
                    <li><strong>Neural Network:</strong> hidden_layer_sizes, activation, alpha, learning_rate</li>
                    <li><strong>Gradient Boosting:</strong> n_estimators, max_depth, learning_rate</li>
                    <li><strong>Logistic Regression:</strong> C, penalty, solver</li>
                    <li><strong>XGBoost:</strong> n_estimators, max_depth, learning_rate, subsample</li>
                  </ul>
                </div>

                {hyperparams.map((param, index) => (
                  <div key={index} className="grid grid-cols-[1fr_1fr_auto] gap-3 items-end">
                    <div>
                      <label className="block text-sm font-medium text-slate-300 mb-2">
                        Parameter Name
                      </label>
                      <input
                        type="text"
                        value={param.key}
                        onChange={(e) => {
                          const newHyperparams = [...hyperparams];
                          if (newHyperparams[index]) {
                            newHyperparams[index].key = e.target.value;
                            setHyperparams(newHyperparams);
                          }
                        }}
                        placeholder="e.g., n_estimators"
                        className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-slate-300 mb-2">
                        Value
                      </label>
                      <input
                        type="text"
                        value={param.value}
                        onChange={(e) => {
                          const newHyperparams = [...hyperparams];
                          if (newHyperparams[index]) {
                            newHyperparams[index].value = e.target.value;
                            setHyperparams(newHyperparams);
                          }
                        }}
                        placeholder="e.g., 100"
                        className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                      />
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        setHyperparams(hyperparams.filter((_, i) => i !== index));
                      }}
                      className="px-4 py-2 bg-red-600/50 hover:bg-red-600 text-white rounded-lg transition-colors text-sm"
                    >
                      Remove
                    </button>
                  </div>
                ))}

                <button
                  type="button"
                  onClick={() => {
                    setHyperparams([...hyperparams, { key: "", value: "" }]);
                  }}
                  className="w-full px-4 py-2 bg-slate-700/50 hover:bg-slate-700 text-white rounded-lg border border-slate-600/50 transition-colors text-sm"
                >
                  + Add Hyperparameter
                </button>
              </div>
            )}
          </div>

          <div className="flex gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-6 py-3 bg-slate-700/50 hover:bg-slate-700 text-white rounded-lg font-medium transition-colors border border-slate-600/50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting}
              className="flex-1 px-6 py-3 bg-gradient-to-r from-teal-600 to-blue-600 hover:from-teal-500 hover:to-blue-500 text-white rounded-lg font-medium transition-all shadow-lg shadow-teal-500/20 hover:shadow-teal-500/40 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {submitting ? "Creating..." : "Create Analysis"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
