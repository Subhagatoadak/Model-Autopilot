import { useEffect, useState } from 'react';
import { useExperimentStore } from '../stores/experimentStore';
import { compareExperiments } from '../api/experiments';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { ExperimentComparison } from '../types/experiment';

const statusColors: Record<string, string> = {
  running: 'bg-blue-600/20 text-blue-400 border-blue-600/30',
  completed: 'bg-green-600/20 text-green-400 border-green-600/30',
  failed: 'bg-red-600/20 text-red-400 border-red-600/30',
  stopped: 'bg-yellow-600/20 text-yellow-400 border-yellow-600/30',
  pending: 'bg-gray-600/20 text-gray-400 border-gray-600/30',
};

export default function ComparePage() {
  const { experiments, fetchExperiments } = useExperimentStore();
  const [expId1, setExpId1] = useState<string>('');
  const [expId2, setExpId2] = useState<string>('');
  const [comparison, setComparison] = useState<ExperimentComparison | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    fetchExperiments();
  }, [fetchExperiments]);

  const handleCompare = async () => {
    if (!expId1 || !expId2) {
      setError('Please select two experiments to compare');
      return;
    }
    if (expId1 === expId2) {
      setError('Please select two different experiments');
      return;
    }

    setLoading(true);
    setError('');
    try {
      const result = await compareExperiments(expId1, expId2);
      setComparison(result);
    } catch (err) {
      setError('Failed to compare experiments');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Build chart data for primary metric comparison
  const buildChartData = () => {
    if (!comparison) return [];

    const exp1 = comparison.experiment_1;
    const exp2 = comparison.experiment_2;
    const primaryMetric = exp1.config?.primary_metric || 'f1';

    const maxIterations = Math.max(exp1.runs.length, exp2.runs.length);
    const data = [];

    for (let i = 0; i < maxIterations; i++) {
      const point: Record<string, number | string> = { iteration: i + 1 };

      if (exp1.runs[i]) {
        point[`Exp 1 (${exp1.id.slice(0, 8)})`] = exp1.runs[i].metrics[primaryMetric] || 0;
      }

      if (exp2.runs[i]) {
        point[`Exp 2 (${exp2.id.slice(0, 8)})`] = exp2.runs[i].metrics[primaryMetric] || 0;
      }

      data.push(point);
    }

    return data;
  };

  // Get all unique metrics from both experiments
  const getAllMetrics = () => {
    if (!comparison) return [];
    const metricsSet = new Set<string>();

    comparison.experiment_1.runs.forEach(run => {
      Object.keys(run.metrics).forEach(m => metricsSet.add(m));
    });

    comparison.experiment_2.runs.forEach(run => {
      Object.keys(run.metrics).forEach(m => metricsSet.add(m));
    });

    return Array.from(metricsSet);
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <h2 className="text-2xl font-bold text-white">Compare Experiments</h2>

      {/* Selectors */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
        <div className="grid grid-cols-3 gap-4 items-end">
          <div>
            <label className="block text-sm text-gray-400 mb-2">Experiment 1</label>
            <select
              value={expId1}
              onChange={(e) => setExpId1(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
            >
              <option value="">Select experiment...</option>
              {experiments.map(exp => (
                <option key={exp.id} value={exp.id}>
                  {exp.id.slice(0, 8)} - {exp.status} - {exp.created_at}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-2">Experiment 2</label>
            <select
              value={expId2}
              onChange={(e) => setExpId2(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
            >
              <option value="">Select experiment...</option>
              {experiments.map(exp => (
                <option key={exp.id} value={exp.id}>
                  {exp.id.slice(0, 8)} - {exp.status} - {exp.created_at}
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={handleCompare}
            disabled={!expId1 || !expId2 || loading}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Comparing...' : 'Compare'}
          </button>
        </div>

        {error && (
          <div className="mt-3 text-sm text-red-400 bg-red-900/30 border border-red-800 rounded-lg px-3 py-2">
            {error}
          </div>
        )}
      </div>

      {/* Comparison Results */}
      {comparison && (
        <div className="space-y-6">
          {/* Config Comparison */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">Configuration Comparison</h3>
            <div className="grid grid-cols-2 gap-6">
              {/* Experiment 1 */}
              <div className="space-y-3">
                <div className="flex items-center justify-between pb-2 border-b border-gray-700">
                  <span className="font-mono text-xs text-gray-500">{comparison.experiment_1.id.slice(0, 8)}</span>
                  <span className={`text-xs px-2 py-0.5 rounded border ${statusColors[comparison.experiment_1.status]}`}>
                    {comparison.experiment_1.status}
                  </span>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Strategy:</span>
                    <span className="text-white font-medium">{comparison.experiment_1.strategy}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Epsilon:</span>
                    <span className="text-white font-medium">{comparison.experiment_1.epsilon}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Runs:</span>
                    <span className="text-white font-medium">{comparison.experiment_1.runs.length}</span>
                  </div>
                  {comparison.experiment_1.config && (
                    <>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Primary Metric:</span>
                        <span className="text-white font-medium">{comparison.experiment_1.config.primary_metric}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Max Iterations:</span>
                        <span className="text-white font-medium">{comparison.experiment_1.config.max_iterations}</span>
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* Experiment 2 */}
              <div className="space-y-3">
                <div className="flex items-center justify-between pb-2 border-b border-gray-700">
                  <span className="font-mono text-xs text-gray-500">{comparison.experiment_2.id.slice(0, 8)}</span>
                  <span className={`text-xs px-2 py-0.5 rounded border ${statusColors[comparison.experiment_2.status]}`}>
                    {comparison.experiment_2.status}
                  </span>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Strategy:</span>
                    <span className="text-white font-medium">{comparison.experiment_2.strategy}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Epsilon:</span>
                    <span className="text-white font-medium">{comparison.experiment_2.epsilon}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Runs:</span>
                    <span className="text-white font-medium">{comparison.experiment_2.runs.length}</span>
                  </div>
                  {comparison.experiment_2.config && (
                    <>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Primary Metric:</span>
                        <span className="text-white font-medium">{comparison.experiment_2.config.primary_metric}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Max Iterations:</span>
                        <span className="text-white font-medium">{comparison.experiment_2.config.max_iterations}</span>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Metrics Comparison Chart */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">
              Performance Comparison - {comparison.experiment_1.config?.primary_metric || 'Primary Metric'}
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={buildChartData()}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="iteration" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey={`Exp 1 (${comparison.experiment_1.id.slice(0, 8)})`}
                  stroke="#3B82F6"
                  strokeWidth={2}
                  dot={{ fill: '#3B82F6' }}
                />
                <Line
                  type="monotone"
                  dataKey={`Exp 2 (${comparison.experiment_2.id.slice(0, 8)})`}
                  stroke="#10B981"
                  strokeWidth={2}
                  dot={{ fill: '#10B981' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Best Metrics Table */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">Best Metrics Comparison</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800">
                    <th className="text-left py-2 px-3 text-gray-400 font-medium">Metric</th>
                    <th className="text-right py-2 px-3 text-gray-400 font-medium">Experiment 1</th>
                    <th className="text-right py-2 px-3 text-gray-400 font-medium">Experiment 2</th>
                    <th className="text-center py-2 px-3 text-gray-400 font-medium">Winner</th>
                  </tr>
                </thead>
                <tbody>
                  {getAllMetrics().map(metric => {
                    const val1 = comparison.experiment_1.best_metrics[metric] || 0;
                    const val2 = comparison.experiment_2.best_metrics[metric] || 0;
                    const winner = val1 > val2 ? 1 : val2 > val1 ? 2 : 0;

                    return (
                      <tr key={metric} className="border-b border-gray-800/50">
                        <td className="py-2 px-3 text-gray-300 font-medium">{metric}</td>
                        <td className={`text-right py-2 px-3 ${winner === 1 ? 'text-green-400 font-semibold' : 'text-gray-400'}`}>
                          {val1.toFixed(4)}
                        </td>
                        <td className={`text-right py-2 px-3 ${winner === 2 ? 'text-green-400 font-semibold' : 'text-gray-400'}`}>
                          {val2.toFixed(4)}
                        </td>
                        <td className="text-center py-2 px-3">
                          {winner === 1 && <span className="text-blue-400">←</span>}
                          {winner === 2 && <span className="text-green-400">→</span>}
                          {winner === 0 && <span className="text-gray-600">-</span>}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {!comparison && !loading && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-10 text-center">
          <p className="text-gray-500">Select two experiments above to compare them.</p>
        </div>
      )}
    </div>
  );
}
