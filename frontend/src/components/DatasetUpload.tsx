import { useState, useCallback } from 'react';
import { useDatasetStore } from '../stores/datasetStore';

export default function DatasetUpload() {
  const { upload, loading } = useDatasetStore();
  const [dragActive, setDragActive] = useState(false);
  const [name, setName] = useState('');
  const [targetCol, setTargetCol] = useState('');
  const [taskType, setTaskType] = useState('');

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      const file = e.dataTransfer.files[0];
      if (file && file.name.endsWith('.csv')) {
        await upload(file, name || undefined, targetCol || undefined, taskType || undefined);
        setName('');
        setTargetCol('');
        setTaskType('');
      }
    },
    [upload, name, targetCol, taskType],
  );

  const handleFileInput = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        await upload(file, name || undefined, targetCol || undefined, taskType || undefined);
        setName('');
        setTargetCol('');
        setTaskType('');
      }
    },
    [upload, name, targetCol, taskType],
  );

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3">
        <input
          type="text"
          placeholder="Dataset name (optional)"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-500"
        />
        <input
          type="text"
          placeholder="Target column"
          value={targetCol}
          onChange={(e) => setTargetCol(e.target.value)}
          className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-500"
        />
        <select
          value={taskType}
          onChange={(e) => setTaskType(e.target.value)}
          className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200"
        >
          <option value="">Auto-detect task type</option>
          <option value="classification">Classification</option>
          <option value="regression">Regression</option>
        </select>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-xl p-10 text-center transition-colors cursor-pointer ${
          dragActive
            ? 'border-blue-500 bg-blue-500/10'
            : 'border-gray-700 hover:border-gray-600'
        }`}
      >
        <p className="text-gray-400 mb-2">
          {loading ? 'Uploading...' : 'Drag & drop a CSV file here'}
        </p>
        <p className="text-gray-600 text-sm mb-4">or</p>
        <label className="inline-block px-4 py-2 bg-blue-600 text-white rounded-lg cursor-pointer hover:bg-blue-700 text-sm">
          Browse Files
          <input
            type="file"
            accept=".csv"
            onChange={handleFileInput}
            className="hidden"
          />
        </label>
      </div>
    </div>
  );
}
