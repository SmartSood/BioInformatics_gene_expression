import { useState } from "react";
import axios from "axios";
import { X, Sparkles, Beaker } from "lucide-react";
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

export function NewExperimentForm({
  datasetId,
  dataset,
  onClose,
  onSuccess,
}: NewExperimentFormProps) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [modelType, setModelType] = useState("random_forest");
  const [numFolds, setNumFolds] = useState(5);
  const [trainTestSplit, setTrainTestSplit] = useState(80);
  const [featureSelection, setFeatureSelection] = useState("none");
  const [selectedPreprocessing, setSelectedPreprocessing] = useState<
    string[]
  >([]);
  const [targetVariable, setTargetVariable] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const togglePreprocessing = (step: string) => {
    setSelectedPreprocessing((prev) =>
      prev.includes(step) ? prev.filter((s) => s !== step) : [...prev, step]
    );
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
   * The function sets safe defaults; toggling a step will enable/change fields appropriately.
   */
  const buildPreprocessingPayload = () => {
    // Defaults mirror your Pydantic defaults and pipeline expectations
    const payload: any = {
      missing_values: {
        strategy_numeric: "median", // ImputationConfig.strategy_numeric
        strategy_categorical: "most_frequent",
        fill_value_numeric: null,
        fill_value_categorical: null,
        drop_rows: false,
      },
      outlier_removal: {
        method: "none", // "none" | "iqr" | "zscore" | "percentile"
        iqr_factor: 1.5,
        zscore_threshold: 3.0,
        percentile_min: 0.5,
        percentile_max: 99.5,
        cap_outliers: false,
      },
      scaling: {
        method: "standard", // "none"|"standard"|"minmax"|"robust"|"maxabs"
        feature_range: [0.0, 1.0],
        apply_to: "numeric_only",
      },
      log_transform: {
        enabled: false,
        offset: 1.0,
        columns: null,
      },
      batch_correction: {
        enabled: false,
        method: "none", // "none" | "combat" | "zscore" | "ratio"
        batch_column: null,
      },
      qc_filtering: {
        enabled: false,
        max_missing_fraction: 0.2,
        numeric_range: null,
      },
      encoding: {
        method: "onehot", // "onehot" | "ordinal" | "none"
        drop_first: false,
      },
      feature_selection: {
        method: "none", // one of FeatureSelectionMethod
        k_features: null,
        variance_threshold: 0.0,
        alpha: 0.001,
        importance_threshold: null,
      },
    };

    // Now enable/adjust chosen steps to produce a payload compatible with the pipeline
    if (selectedPreprocessing.includes("missing_values")) {
      // keep defaults; let backend handle if drop_rows true vs impute
      payload.missing_values = { ...payload.missing_values };
    }

    if (selectedPreprocessing.includes("outlier_removal")) {
      // pick a reasonable default used by pipeline — "iqr"
      payload.outlier_removal.method = "iqr";
      payload.outlier_removal.cap_outliers = false;
      payload.outlier_removal.iqr_factor = 1.5;
    }

    if (selectedPreprocessing.includes("scaling")) {
      // choose "standard" as default; if user also picked "Normalization" earlier you might map to minmax,
      // but UI does not have separate "Normalization" now — scaling step enables StandardScaler by default.
      payload.scaling.method = "standard";
      payload.scaling.feature_range = [0.0, 1.0];
    }

    if (selectedPreprocessing.includes("log_transform")) {
      payload.log_transform.enabled = true;
      payload.log_transform.offset = 1.0;
      payload.log_transform.columns = null; // pipeline applies to all numeric if null
    }

    if (selectedPreprocessing.includes("batch_correction")) {
      payload.batch_correction.enabled = true;
      // pipeline supports "combat", "zscore", "ratio"; choose "combat" as a common default
      payload.batch_correction.method = "combat";
      payload.batch_correction.batch_column = null; // user may set later
    }

    if (selectedPreprocessing.includes("qc_filtering")) {
      payload.qc_filtering.enabled = true;
      payload.qc_filtering.max_missing_fraction = 0.2;
    }

    if (selectedPreprocessing.includes("encoding")) {
      // Most pipelines expect onehot for categorical features by default
      payload.encoding.method = "onehot";
      payload.encoding.drop_first = false;
    }

    if (selectedPreprocessing.includes("feature_selection")) {
      // Map the selected featureSelection choice into the structured object
      const sel = featureSelection;
      if (!sel || sel === "none") {
        payload.feature_selection.method = "none";
      } else if (sel === "variance_threshold") {
        payload.feature_selection.method = "variance_threshold";
        payload.feature_selection.variance_threshold = 0.0;
      } else if (sel === "rfe") {
        payload.feature_selection.method = "rfe";
        payload.feature_selection.k_features = 10; // reasonable default
      } else if (sel === "lasso") {
        payload.feature_selection.method = "lasso";
        payload.feature_selection.alpha = 0.001;
      } else if (sel === "random_forest_importance") {
        payload.feature_selection.method = "random_forest_importance";
        payload.feature_selection.importance_threshold = 0.01;
      } else if (sel === "chi2") {
        payload.feature_selection.method = "chi2";
        payload.feature_selection.k_features = 10;
      } else {
        payload.feature_selection.method = "none";
      }
    } else {
      // if user didn't enable feature_selection step, keep method "none"
      payload.feature_selection.method = "none";
    }

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
      

      // Determine problem_type fallback — default to classification (pipeline expects "classification"|"regression")
      const problem_type =
        sessionStorage.getItem(`dataset_problem_${datasetId}`) || "classification";

      const trainRequest = {
        dataset_id: datasetId,
        dataset_uri,
        config: {
          target: targetVariable,
          problem_type,
          preprocessing: buildPreprocessingPayload(),
          model: modelType,
          hyperparams: {}, // UI doesn't expose hyperparams — keep empty
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
          <div className="col-span-2">
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Target
              </label>
              <textarea
                value={targetVariable}
                onChange={(e) => setTargetVariable(e.target.value)}
                rows={1}
                className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500/50 resize-none"
                placeholder="Enter the target column (e.g. Y)"
              />
            </div>
          

          <div className="border-t border-slate-700/50 pt-6">
            <div className="flex items-center gap-2 mb-4">
              <Beaker className="w-5 h-5 text-purple-400" />
              <h3 className="text-lg font-semibold text-white">
                Preprocessing Steps
              </h3>
            </div>
            <div className="grid grid-cols-2 gap-3">
              {PREPROCESSING_STEPS.map((step) => (
                <label
                  key={step.key}
                  className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                    selectedPreprocessing.includes(step.key)
                      ? "bg-purple-500/20 border-purple-500/50 text-purple-300"
                      : "bg-slate-700/30 border-slate-600/50 text-slate-300 hover:bg-slate-700/50"
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={selectedPreprocessing.includes(step.key)}
                    onChange={() => togglePreprocessing(step.key)}
                    className="w-4 h-4 rounded border-slate-500 text-teal-500 focus:ring-teal-500/50 bg-slate-700"
                  />
                  <span className="text-sm font-medium">{step.label}</span>
                </label>
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
                Feature Selection
              </label>
              <select
                value={featureSelection}
                onChange={(e) => setFeatureSelection(e.target.value)}
                className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500/50"
              >
                {FEATURE_SELECTION_OPTIONS.map((method) => (
                  <option key={method.value} value={method.value}>
                    {method.label}
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
