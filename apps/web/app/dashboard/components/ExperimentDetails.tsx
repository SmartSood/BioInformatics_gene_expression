import { Card } from '@repo/ui/card';
import { useExperimentDetails } from '../../../hooks/useExperiment';
import {
  Activity,
  BarChart3,
  Settings,
  TrendingUp,
  Loader,
  CheckCircle,
  XCircle,
  Clock
} from 'lucide-react';
import { Gene } from '../../../utils/scemma';

interface ExperimentDetailsProps {
  experimentId: string;
}

export function ExperimentDetails({ experimentId }: ExperimentDetailsProps) {
  const { experiment, parameters, results, loading } = useExperimentDetails(experimentId);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader className="w-8 h-8 text-teal-400 animate-spin" />
      </div>
    );
  }

  if (!experiment) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-slate-400">Experiment not found</p>
      </div>
    );
  }

  const getStatusBadge = () => {
    const statusConfig = {
      completed: { icon: CheckCircle, text: 'Completed', className: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/50' },
      running: { icon: Loader, text: 'Running', className: 'bg-blue-500/20 text-blue-400 border-blue-500/50' },
      failed: { icon: XCircle, text: 'Failed', className: 'bg-red-500/20 text-red-400 border-red-500/50' },
      pending: { icon: Clock, text: 'Pending', className: 'bg-amber-500/20 text-amber-400 border-amber-500/50' },
    };
    //@ts-ignore
    const config = statusConfig[experiment.status];
    const Icon = config.icon;

    return (
      <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full border ${config.className}`}>
        <Icon className={`w-4 h-4 ${experiment.status === 'started' ? 'animate-spin' : ''}`} />
        <span className="text-sm font-medium">{config.text}</span>
      </div>
    );
  };

  return (
    <div className="h-full overflow-y-auto p-8">
      <div className="max-w-6xl mx-auto space-y-6">
        <Card color="slate">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h1 className="text-2xl font-bold text-white mb-2">{experiment.name}</h1>
              {experiment.description && (
                <p className="text-slate-400">{experiment.description}</p>
              )}
            </div>
            {getStatusBadge()}
          </div>
          
          <div className="flex gap-6 text-sm">
            <div>
              <span className="text-slate-500">Created:</span>
              <span className="text-white ml-2">
                {new Date(experiment.createdAt).toLocaleString()}
              </span>
            </div>
            <div>
              <span className="text-slate-500">Updated:</span>
              <span className="text-white ml-2">
                {new Date(experiment.updatedAt).toLocaleString()}
              </span>
            </div>
          </div>
        </Card>

        {parameters && (
          <Card 
            title="Configuration Parameters" 
            icon={<Settings className="w-5 h-5" />}
            color="slate"
            iconColor="purple"
          >
            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-slate-400">Model Type</label>
                  <div className="mt-1 px-4 py-2 bg-slate-700/50 rounded-lg border border-slate-600/50">
                    <span className="text-white font-medium">{parameters.model_type}</span>
                  </div>
                </div>

                <div>
                  <label className="text-sm font-medium text-slate-400">Cross-Validation Folds</label>
                  <div className="mt-1 px-4 py-2 bg-slate-700/50 rounded-lg border border-slate-600/50">
                    <span className="text-white font-medium">{parameters.num_folds}</span>
                  </div>
                </div>

                <div>
                  <label className="text-sm font-medium text-slate-400">Train/Test Split</label>
                  <div className="mt-1 px-4 py-2 bg-slate-700/50 rounded-lg border border-slate-600/50">
                    <span className="text-white font-medium">{(parameters.train_test_split * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                {parameters.feature_selection && (
                  <div>
                    <label className="text-sm font-medium text-slate-400">Feature Selection</label>
                    <div className="mt-1 px-4 py-2 bg-slate-700/50 rounded-lg border border-slate-600/50">
                      <span className="text-white font-medium">{parameters.feature_selection}</span>
                    </div>
                  </div>
                )}

                <div>
                  <label className="text-sm font-medium text-slate-400">Preprocessing Steps</label>
                  <div className="mt-1 space-y-2">
                    {parameters.preprocessing_steps.length > 0 ? (
                      parameters.preprocessing_steps.map((step, index) => (
                        <div key={index} className="px-4 py-2 bg-slate-700/50 rounded-lg border border-slate-600/50">
                          <span className="text-white">{step}</span>
                        </div>
                      ))
                    ) : (
                      <div className="px-4 py-2 bg-slate-700/50 rounded-lg border border-slate-600/50">
                        <span className="text-slate-400">None</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </Card>
        )}

        {results && (
          <>
            <Card 
              title="Performance Metrics" 
              icon={<BarChart3 className="w-5 h-5" />}
              color="slate"
              iconColor="blue"
            >
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <MetricCard label="Accuracy" value={results.accuracy} color="teal" />
                <MetricCard label="Precision" value={results.precision_score} color="blue" />
                <MetricCard label="Recall" value={results.recall_score} color="purple" />
                <MetricCard label="F1 Score" value={results.f1_score} color="pink" />
                <MetricCard label="ROC AUC" value={results.roc_auc} color="emerald" />
              </div>
            </Card>

            <Card 
              title="Top Expressed Genes" 
              icon={<TrendingUp className="w-5 h-5" />}
              color="slate"
              iconColor="emerald"
            >
              <div className="space-y-3">
                {results.top_genes.length > 0 ? (
                  results.top_genes.map((gene: Gene, index: number) => (
                    <GeneCard key={index} gene={gene} rank={index + 1} />
                  ))
                ) : (
                  <div className="text-center py-8 text-slate-400">
                    No gene expression data available
                  </div>
                )}
              </div>
            </Card>
          </>
        )}

        {!results && experiment.status === 'finished' && (
          <Card className="text-center">
            <Activity className="w-12 h-12 text-slate-600 mx-auto mb-3" />
            <p className="text-slate-400">No results available for this experiment</p>
          </Card>
        )}
      </div>
    </div>
  );
}

function MetricCard({ label, value, color }: { label: string; value: number | null; color: string }) {
  const colorClasses = {
    teal: 'from-teal-500/20 to-teal-600/10 border-teal-500/30 text-teal-400',
    blue: 'from-blue-500/20 to-blue-600/10 border-blue-500/30 text-blue-400',
    purple: 'from-purple-500/20 to-purple-600/10 border-purple-500/30 text-purple-400',
    pink: 'from-pink-500/20 to-pink-600/10 border-pink-500/30 text-pink-400',
    emerald: 'from-emerald-500/20 to-emerald-600/10 border-emerald-500/30 text-emerald-400',
  };

  return (
    <div className={`bg-gradient-to-br ${colorClasses[color as keyof typeof colorClasses]} rounded-lg border p-4`}>
      <div className="text-sm text-slate-300 mb-1">{label}</div>
      <div className="text-2xl font-bold">
        {value !== null ? (value * 100).toFixed(1) + '%' : 'N/A'}
      </div>
    </div>
  );
}

function GeneCard({ gene, rank }: { gene: Gene; rank: number }) {
  const getExpressionColor = (expression: number) => {
    if (expression >= 0.8) return 'from-red-500/20 to-pink-500/20 border-red-500/40';
    if (expression >= 0.6) return 'from-orange-500/20 to-amber-500/20 border-orange-500/40';
    if (expression >= 0.4) return 'from-yellow-500/20 to-lime-500/20 border-yellow-500/40';
    return 'from-green-500/20 to-emerald-500/20 border-green-500/40';
  };

  const getExpressionText = (expression: number) => {
    if (expression >= 0.8) return 'text-red-400';
    if (expression >= 0.6) return 'text-orange-400';
    if (expression >= 0.4) return 'text-yellow-400';
    return 'text-green-400';
  };

  return (
    <div className={`bg-gradient-to-r ${getExpressionColor(gene.expression)} rounded-lg border p-4`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center justify-center w-8 h-8 rounded-full bg-slate-700/50 border border-slate-600/50">
            <span className="text-sm font-bold text-white">{rank}</span>
          </div>
          <div>
            <h3 className="text-lg font-bold text-white">{gene.symbol}</h3>
            <div className="flex gap-4 text-sm text-slate-300 mt-1">
              <span>Fold Change: <span className="font-medium">{gene.foldChange.toFixed(2)}</span></span>
              <span>p-value: <span className="font-medium">{gene.pvalue.toExponential(2)}</span></span>
            </div>
          </div>
        </div>
        <div className="text-right">
          <div className="text-sm text-slate-400">Expression</div>
          <div className={`text-2xl font-bold ${getExpressionText(gene.expression)}`}>
            {(gene.expression * 100).toFixed(1)}%
          </div>
        </div>
      </div>
    </div>
  );
}
