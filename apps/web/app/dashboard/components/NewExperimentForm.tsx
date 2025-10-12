import { useState } from "react";
import { X, Sparkles, Beaker } from "lucide-react";

interface NewExperimentFormProps {
  datasetId: string;
  onClose: () => void;
  onSuccess: () => void;
}

const PREPROCESSING_OPTIONS = [
  "Normalization",
  "Missing Value Imputation",
  "Outlier Removal",
  "Feature Scaling",
  "Log Transformation",
  "Batch Correction",
  "Quality Control Filtering",
];

const MODEL_OPTIONS = [
  { value: "random_forest", label: "Random Forest" },
  { value: "svm", label: "Support Vector Machine" },
  { value: "neural_network", label: "Neural Network" },
  { value: "gradient_boosting", label: "Gradient Boosting" },
  { value: "logistic_regression", label: "Logistic Regression" },
  { value: "xgboost", label: "XGBoost" },
];

const FEATURE_SELECTION_OPTIONS = [
  "None",
  "Variance Threshold",
  "Recursive Feature Elimination",
  "LASSO",
  "Random Forest Importance",
  "Chi-Square Test",
];

export function NewExperimentForm({
  datasetId,
  onClose,
  onSuccess,
}: NewExperimentFormProps) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [modelType, setModelType] = useState("random_forest");
  const [numFolds, setNumFolds] = useState(5);
  const [trainTestSplit, setTrainTestSplit] = useState(80);
  const [featureSelection, setFeatureSelection] = useState("None");
  const [selectedPreprocessing, setSelectedPreprocessing] = useState<string[]>(
    []
  );
  const [submitting, setSubmitting] = useState(false);

  const togglePreprocessing = (step: string) => {
    setSelectedPreprocessing((prev) =>
      prev.includes(step) ? prev.filter((s) => s !== step) : [...prev, step]
    );
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);

    try {
      const newId = Date.now().toString();
      const now = new Date().toISOString();

      const currentUser =
        sessionStorage.getItem("currentUser") ||
        "00000000-0000-0000-0000-000000000000";

      const newExperiment = {
        id: newId,
        user_id: currentUser,
        dataset_id: datasetId,
        name,
        description,
        status: "completed" as const,
        created_at: now,
        updated_at: now,
      };

      const newParams = {
        id: newId,
        experiment_id: newId,
        preprocessing_steps: selectedPreprocessing,
        model_type: modelType,
        num_folds: numFolds,
        train_test_split: trainTestSplit / 100,
        feature_selection: featureSelection === "None" ? "" : featureSelection,
        hyperparameters: {},
        created_at: now,
      };

      const mockGenes = Array.from({ length: 10 }, (_, i) => ({
        symbol: `GENE${i + 1}`,
        expression: Math.random() * 10,
        pvalue: Math.random() * 0.05,
        foldChange: 1 + Math.random() * 3,
      })).sort((a, b) => b.expression - a.expression);

      const newResults = {
        id: newId,
        experiment_id: newId,
        top_genes: mockGenes,
        accuracy: 0.75 + Math.random() * 0.2,
        precision_score: 0.7 + Math.random() * 0.25,
        recall_score: 0.7 + Math.random() * 0.25,
        f1_score: 0.72 + Math.random() * 0.23,
        roc_auc: 0.8 + Math.random() * 0.15,
        additional_metrics: {},
        created_at: now,
      };

      const stored = sessionStorage.getItem("experiments");
      const experiments = stored ? JSON.parse(stored) : [];
      experiments.unshift(newExperiment);
      sessionStorage.setItem("experiments", JSON.stringify(experiments));
      sessionStorage.setItem(`params_${newId}`, JSON.stringify(newParams));
      sessionStorage.setItem(`results_${newId}`, JSON.stringify(newResults));

      onSuccess();
      onClose();
    } catch (error) {
      console.error("Error creating experiment:", error);
      alert("Failed to create experiment. Please try again.");
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

          <div className="border-t border-slate-700/50 pt-6">
            <div className="flex items-center gap-2 mb-4">
              <Beaker className="w-5 h-5 text-purple-400" />
              <h3 className="text-lg font-semibold text-white">
                Preprocessing Steps
              </h3>
            </div>
            <div className="grid grid-cols-2 gap-3">
              {PREPROCESSING_OPTIONS.map((step) => (
                <label
                  key={step}
                  className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                    selectedPreprocessing.includes(step)
                      ? "bg-purple-500/20 border-purple-500/50 text-purple-300"
                      : "bg-slate-700/30 border-slate-600/50 text-slate-300 hover:bg-slate-700/50"
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={selectedPreprocessing.includes(step)}
                    onChange={() => togglePreprocessing(step)}
                    className="w-4 h-4 rounded border-slate-500 text-teal-500 focus:ring-teal-500/50 bg-slate-700"
                  />
                  <span className="text-sm font-medium">{step}</span>
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
                  <option key={method} value={method}>
                    {method}
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
                min="2"
                max="20"
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
                min="50"
                max="95"
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
