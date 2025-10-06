import { FlaskConical, Sparkles } from 'lucide-react';

interface EmptyStateProps {
  onNewExperiment: () => void;
}

export function EmptyState({ onNewExperiment }: EmptyStateProps) {
  return (
    <div className="h-full flex items-center justify-center p-8">
      <div className="text-center max-w-md">
        <div className="mb-6 flex justify-center">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-teal-500/20 to-blue-500/20 blur-2xl"></div>
            <div className="relative p-6 bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl border border-slate-700/50">
              <FlaskConical className="w-16 h-16 text-teal-400" />
            </div>
          </div>
        </div>

        <h2 className="text-2xl font-bold text-white mb-3">
          Welcome to DrugTarget AI
        </h2>
        <p className="text-slate-400 mb-8">
          Start your first drug target interaction analysis to identify key genes and evaluate model performance.
        </p>

        <button
          onClick={onNewExperiment}
          className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-teal-600 to-blue-600 hover:from-teal-500 hover:to-blue-500 text-white rounded-lg font-medium transition-all shadow-lg shadow-teal-500/20 hover:shadow-teal-500/40"
        >
          <Sparkles className="w-5 h-5" />
          Create Your First Analysis
        </button>

        <div className="mt-12 grid grid-cols-3 gap-4 text-left">
          <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700/50">
            <div className="text-teal-400 font-semibold mb-1">Configure</div>
            <div className="text-xs text-slate-400">Set preprocessing and model parameters</div>
          </div>
          <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700/50">
            <div className="text-blue-400 font-semibold mb-1">Analyze</div>
            <div className="text-xs text-slate-400">Run ML models on your data</div>
          </div>
          <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700/50">
            <div className="text-purple-400 font-semibold mb-1">Discover</div>
            <div className="text-xs text-slate-400">Identify top expressed genes</div>
          </div>
        </div>
      </div>
    </div>
  );
}
