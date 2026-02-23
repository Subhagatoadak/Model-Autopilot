import { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useExperimentStore } from '../stores/experimentStore';

const statusColors: Record<string, string> = {
  running: 'bg-blue-600/20 text-blue-400 border-blue-600/30',
  completed: 'bg-green-600/20 text-green-400 border-green-600/30',
  failed: 'bg-red-600/20 text-red-400 border-red-600/30',
  stopped: 'bg-yellow-600/20 text-yellow-400 border-yellow-600/30',
  pending: 'bg-gray-600/20 text-gray-400 border-gray-600/30',
};

export default function HistoryPage() {
  const { experiments, fetchExperiments, loadExperiment } = useExperimentStore();

  useEffect(() => {
    fetchExperiments();
  }, [fetchExperiments]);

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <h2 className="text-2xl font-bold text-white">Experiment History</h2>

      {experiments.length === 0 ? (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-10 text-center">
          <p className="text-gray-500">No experiments yet.</p>
          <Link
            to="/experiment"
            className="inline-block mt-3 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700"
          >
            Start your first experiment
          </Link>
        </div>
      ) : (
        <div className="space-y-2">
          {experiments.map((exp) => (
            <Link
              key={exp.id}
              to="/experiment"
              onClick={() => loadExperiment(exp.id)}
              className="block bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-gray-700 transition-colors"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="font-mono text-xs text-gray-500">{exp.id.slice(0, 8)}</span>
                  <span className={`text-xs px-2 py-0.5 rounded border ${statusColors[exp.status] || statusColors.pending}`}>
                    {exp.status}
                  </span>
                </div>
                <span className="text-xs text-gray-500">{exp.created_at}</span>
              </div>
              <div className="flex gap-4 mt-2 text-xs text-gray-500">
                <span>Strategy: {exp.strategy}</span>
                <span>Epsilon: {exp.epsilon}</span>
                <span>Dataset: {exp.dataset_id.slice(0, 8)}...</span>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
