import { useEffect } from 'react';
import { useDatasetStore } from '../stores/datasetStore';
import DatasetUpload from '../components/DatasetUpload';

export default function DatasetPage() {
  const { datasets, selectedDataset, fetchDatasets, selectDataset, error } = useDatasetStore();

  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <h2 className="text-2xl font-bold text-white">Datasets</h2>

      {error && (
        <div className="bg-red-900/30 border border-red-800 text-red-300 px-4 py-3 rounded-lg text-sm">
          {error}
        </div>
      )}

      {/* Upload */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Upload Dataset</h3>
        <DatasetUpload />
      </div>

      {/* Dataset List */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Your Datasets</h3>
        {datasets.length === 0 ? (
          <p className="text-gray-500 text-sm">No datasets uploaded yet.</p>
        ) : (
          <div className="space-y-2">
            {datasets.map((ds) => (
              <button
                key={ds.id}
                onClick={() => selectDataset(ds.id)}
                className={`w-full text-left p-4 rounded-lg border transition-colors ${
                  selectedDataset?.id === ds.id
                    ? 'border-blue-600 bg-blue-600/10'
                    : 'border-gray-800 hover:border-gray-700 bg-gray-800/50'
                }`}
              >
                <div className="flex justify-between items-center">
                  <span className="font-medium text-gray-200">{ds.name}</span>
                  <span className="text-xs text-gray-500">{ds.created_at}</span>
                </div>
                <div className="flex gap-4 mt-1 text-xs text-gray-500">
                  <span>{ds.num_rows} rows</span>
                  <span>{ds.num_cols} cols</span>
                  {ds.target_col && <span>Target: {ds.target_col}</span>}
                  {ds.task_type && (
                    <span className="px-1.5 py-0.5 bg-gray-700 rounded text-gray-300">
                      {ds.task_type}
                    </span>
                  )}
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Dataset Detail */}
      {selectedDataset && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4">
            {selectedDataset.name} - Column Details
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700 text-gray-400">
                  <th className="text-left py-2 px-3">Column</th>
                  <th className="text-left py-2 px-3">Type</th>
                  <th className="text-right py-2 px-3">Null %</th>
                  <th className="text-right py-2 px-3">Unique</th>
                  <th className="text-left py-2 px-3">Samples</th>
                </tr>
              </thead>
              <tbody>
                {selectedDataset.columns.map((col) => (
                  <tr key={col.name} className="border-b border-gray-800">
                    <td className="py-2 px-3 font-medium text-gray-200">{col.name}</td>
                    <td className="py-2 px-3 text-gray-400">{col.dtype}</td>
                    <td className="text-right py-2 px-3 text-gray-400">{col.null_pct}%</td>
                    <td className="text-right py-2 px-3 text-gray-400">{col.unique_count}</td>
                    <td className="py-2 px-3 text-gray-500 text-xs">
                      {col.sample_values.join(', ')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
