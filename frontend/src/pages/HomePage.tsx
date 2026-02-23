import { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useDatasetStore } from '../stores/datasetStore';
import { useExperimentStore } from '../stores/experimentStore';

export default function HomePage() {
  const { datasets, fetchDatasets } = useDatasetStore();
  const { experiments, fetchExperiments } = useExperimentStore();

  useEffect(() => {
    fetchDatasets();
    fetchExperiments();
  }, [fetchDatasets, fetchExperiments]);

  const completedExps = experiments.filter((e) => e.status === 'completed');

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      <div className="text-center space-y-4 py-8">
        <h1 className="text-4xl font-bold text-white">Model Autopilot</h1>
        <p className="text-gray-400 text-lg">
          AI-powered ML model optimization using Tree of Thought reasoning
          and explore/exploit strategies
        </p>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 text-center">
          <p className="text-3xl font-bold text-blue-400">{datasets.length}</p>
          <p className="text-sm text-gray-500 mt-1">Datasets</p>
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 text-center">
          <p className="text-3xl font-bold text-green-400">{experiments.length}</p>
          <p className="text-sm text-gray-500 mt-1">Experiments</p>
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 text-center">
          <p className="text-3xl font-bold text-purple-400">{completedExps.length}</p>
          <p className="text-sm text-gray-500 mt-1">Completed</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <Link
          to="/datasets"
          className="block bg-gray-900 border border-gray-800 rounded-xl p-6 hover:border-blue-600/50 transition-colors"
        >
          <h3 className="text-lg font-semibold text-white mb-2">Upload Dataset</h3>
          <p className="text-sm text-gray-400">
            Upload a CSV file or connect to a PostgreSQL database to get started.
          </p>
        </Link>
        <Link
          to="/experiment"
          className="block bg-gray-900 border border-gray-800 rounded-xl p-6 hover:border-blue-600/50 transition-colors"
        >
          <h3 className="text-lg font-semibold text-white mb-2">New Experiment</h3>
          <p className="text-sm text-gray-400">
            Launch an AI-guided model optimization experiment with real-time visualization.
          </p>
        </Link>
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-3">How it works</h3>
        <div className="grid grid-cols-4 gap-4 text-center">
          {[
            { step: '1', title: 'Data Agent', desc: 'Profiles your data and identifies quality issues' },
            { step: '2', title: 'Model Agent', desc: 'Selects models using explore/exploit strategy' },
            { step: '3', title: 'Evaluation Agent', desc: 'Computes metrics and tracks performance' },
            { step: '4', title: 'Recommendation Agent', desc: 'Uses Tree of Thought to find improvements' },
          ].map((item) => (
            <div key={item.step} className="space-y-2">
              <div className="w-8 h-8 rounded-full bg-blue-600/20 text-blue-400 flex items-center justify-center mx-auto font-bold text-sm">
                {item.step}
              </div>
              <p className="text-sm font-medium text-gray-200">{item.title}</p>
              <p className="text-xs text-gray-500">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
