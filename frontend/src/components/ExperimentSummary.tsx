import type { ExperimentRun } from '../types/experiment';

interface Props {
  runs: ExperimentRun[];
  primaryMetric?: string;
}

export default function ExperimentSummary({ runs, primaryMetric }: Props) {
  if (runs.length === 0) return null;

  const metric = primaryMetric || Object.keys(runs[0]?.metrics || {})[0] || 'accuracy';

  // Find best run
  const bestRun = [...runs].sort((a, b) => (b.metrics[metric] || 0) - (a.metrics[metric] || 0))[0];

  // First run baseline
  const firstRun = runs[0];
  const firstMetric = firstRun?.metrics[metric] || 0;
  const bestMetric = bestRun?.metrics[metric] || 0;
  const improvement = firstMetric > 0 ? ((bestMetric - firstMetric) / firstMetric * 100) : 0;

  // Unique models tried
  const uniqueModels = new Set(runs.map((r) => r.model_name)).size;

  // Average training time
  const avgTime = runs.reduce((sum, r) => sum + (r.train_time_s || 0), 0) / runs.length;

  // Total training time
  const totalTime = runs.reduce((sum, r) => sum + (r.train_time_s || 0), 0);

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      <div className="bg-gray-800/50 rounded-lg p-3">
        <div className="text-xs text-gray-500 mb-1">Best {metric}</div>
        <div className="text-xl font-bold text-blue-400">{bestMetric.toFixed(4)}</div>
        <div className="text-xs text-gray-500 mt-0.5">{bestRun?.model_name}</div>
      </div>
      <div className="bg-gray-800/50 rounded-lg p-3">
        <div className="text-xs text-gray-500 mb-1">Improvement</div>
        <div className={`text-xl font-bold ${improvement >= 0 ? 'text-green-400' : 'text-red-400'}`}>
          {improvement >= 0 ? '+' : ''}{improvement.toFixed(1)}%
        </div>
        <div className="text-xs text-gray-500 mt-0.5">vs baseline</div>
      </div>
      <div className="bg-gray-800/50 rounded-lg p-3">
        <div className="text-xs text-gray-500 mb-1">Models Tried</div>
        <div className="text-xl font-bold text-purple-400">{uniqueModels}</div>
        <div className="text-xs text-gray-500 mt-0.5">{runs.length} total runs</div>
      </div>
      <div className="bg-gray-800/50 rounded-lg p-3">
        <div className="text-xs text-gray-500 mb-1">Total Time</div>
        <div className="text-xl font-bold text-orange-400">{totalTime.toFixed(1)}s</div>
        <div className="text-xs text-gray-500 mt-0.5">avg {avgTime.toFixed(1)}s/run</div>
      </div>
    </div>
  );
}
