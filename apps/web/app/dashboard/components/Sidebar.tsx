import { useExperiments } from '../../../hooks/useExperiment';
import { Plus, Dna, Clock, CheckCircle, XCircle, Loader } from 'lucide-react';
import { Experiment } from '../../../utils/scemma';

interface SidebarProps {
  selectedExperimentId: string | null;
  onSelectExperiment: (id: string) => void;
  onNewExperiment: () => void;
}

export function Sidebar({ selectedExperimentId, onSelectExperiment, onNewExperiment }: SidebarProps) {
  const { experiments, loading } = useExperiments();

  const getStatusIcon = (status: Experiment['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-emerald-400" />;
      case 'running':
        return <Loader className="w-4 h-4 text-blue-400 animate-spin" />;
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-400" />;
      default:
        return <Clock className="w-4 h-4 text-amber-400" />;
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  };

  return (
    <div className="w-80 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border-r border-slate-700/50 flex flex-col h-screen">
      <div className="p-6 border-b border-slate-700/50">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 bg-gradient-to-br from-teal-500/20 to-blue-500/20 rounded-lg">
            <Dna className="w-6 h-6 text-teal-400" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">DrugTarget AI</h1>
            <p className="text-xs text-slate-400">Interaction Analysis</p>
          </div>
        </div>

        <button
          onClick={onNewExperiment}
          className="w-full bg-gradient-to-r from-teal-600 to-blue-600 hover:from-teal-500 hover:to-blue-500 text-white py-3 px-4 rounded-lg flex items-center justify-center gap-2 font-medium transition-all shadow-lg shadow-teal-500/20 hover:shadow-teal-500/40"
        >
          <Plus className="w-5 h-5" />
          New Analysis
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3 px-2">
          Recent Experiments
        </h2>

        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Loader className="w-6 h-6 text-teal-400 animate-spin" />
          </div>
        ) : experiments.length === 0 ? (
          <div className="text-center py-8 px-4">
            <p className="text-slate-500 text-sm">No experiments yet</p>
            <p className="text-slate-600 text-xs mt-1">Create your first analysis</p>
          </div>
        ) : (
          <div className="space-y-2">
            {experiments.map((experiment) => (
              <button
                key={experiment.id}
                onClick={() => onSelectExperiment(experiment.id)}
                className={`w-full text-left p-4 rounded-lg transition-all ${
                  selectedExperimentId === experiment.id
                    ? 'bg-gradient-to-r from-teal-600/30 to-blue-600/30 border border-teal-500/50 shadow-lg'
                    : 'bg-slate-800/50 hover:bg-slate-800 border border-transparent'
                }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <h3 className="font-semibold text-white text-sm line-clamp-1">
                    {experiment.name}
                  </h3>
                  {getStatusIcon(experiment.status)}
                </div>

                {experiment.description && (
                  <p className="text-xs text-slate-400 mb-2 line-clamp-2">
                    {experiment.description}
                  </p>
                )}

                <div className="flex items-center gap-2 text-xs text-slate-500">
                  <Clock className="w-3 h-3" />
                  {formatDate(experiment.created_at)}
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
