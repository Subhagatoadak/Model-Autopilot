import { useState, useEffect } from 'react';
import type { ExperimentConfig as Config } from '../types/experiment';
import type { ModelInfo } from '../types/agent';
import { useDatasetStore } from '../stores/datasetStore';
import { getDatasetColumns, getDatasetPreview } from '../api/datasets';
import { getModelZoo } from '../api/experiments';

interface ColumnInfo {
  name: string;
  dtype: string;
  null_pct: number;
  unique_count: number;
  sample_values: string[];
}

interface Props {
  onStart: (config: Config) => void;
  disabled?: boolean;
}

export default function ExperimentConfigForm({ onStart, disabled }: Props) {
  const { datasets, fetchDatasets } = useDatasetStore();
  const [datasetId, setDatasetId] = useState('');
  const [columns, setColumns] = useState<ColumnInfo[]>([]);
  const [preview, setPreview] = useState<Record<string, unknown>[]>([]);
  const [targetCol, setTargetCol] = useState('');
  const [featureCols, setFeatureCols] = useState<string[]>([]);
  const [taskType, setTaskType] = useState('');
  const [primaryMetric, setPrimaryMetric] = useState('');
  const [epsilon, setEpsilon] = useState(0.3);
  const [maxIterations, setMaxIterations] = useState(10);
  const [testSize, setTestSize] = useState(0.2);
  const [multiObjective, setMultiObjective] = useState(false);
  const [metricWeight, setMetricWeight] = useState(0.7);  // weight for primary metric

  // Model Zoo
  const [allModels, setAllModels] = useState<ModelInfo[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [showModelZoo, setShowModelZoo] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  // Load columns when dataset changes
  useEffect(() => {
    if (!datasetId) {
      setColumns([]);
      setPreview([]);
      setTargetCol('');
      setFeatureCols([]);
      return;
    }

    const ds = datasets.find((d) => d.id === datasetId);
    if (ds?.target_col) setTargetCol(ds.target_col);
    if (ds?.task_type) setTaskType(ds.task_type || '');

    getDatasetColumns(datasetId).then(setColumns).catch(console.error);
    getDatasetPreview(datasetId, 5).then(setPreview).catch(console.error);
  }, [datasetId, datasets]);

  // Load model zoo when task type changes
  useEffect(() => {
    getModelZoo(taskType || undefined).then((models) => {
      setAllModels(models);
    }).catch(console.error);
  }, [taskType]);

  // Auto-select all features when columns load (excluding target)
  useEffect(() => {
    if (columns.length > 0 && featureCols.length === 0) {
      setFeatureCols(columns.filter((c) => c.name !== targetCol).map((c) => c.name));
    }
  }, [columns, targetCol, featureCols.length]);

  // Update feature cols when target changes
  useEffect(() => {
    if (targetCol && columns.length > 0) {
      setFeatureCols((prev) => prev.filter((c) => c !== targetCol));
    }
  }, [targetCol, columns]);

  const classificationMetrics = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc'];
  const regressionMetrics = ['r2', 'mse', 'rmse', 'mae'];
  const metricOptions = taskType === 'regression' ? regressionMetrics : classificationMetrics;

  const nonTargetCols = columns.filter((c) => c.name !== targetCol);
  const allFeaturesSelected = nonTargetCols.length > 0 && featureCols.length === nonTargetCols.length;

  const toggleFeature = (col: string) => {
    setFeatureCols((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  };

  const toggleAllFeatures = () => {
    if (allFeaturesSelected) {
      setFeatureCols([]);
    } else {
      setFeatureCols(nonTargetCols.map((c) => c.name));
    }
  };

  const toggleModel = (name: string) => {
    setSelectedModels((prev) =>
      prev.includes(name) ? prev.filter((m) => m !== name) : [...prev, name]
    );
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!datasetId || !targetCol) return;

    // Build objective weights if multi-objective is enabled
    const objectiveWeights: Record<string, number> = {};
    if (multiObjective) {
      const metric = primaryMetric || (taskType === 'regression' ? 'r2' : 'f1');
      objectiveWeights[metric] = metricWeight;
      objectiveWeights['train_time_s'] = 1 - metricWeight;
    }

    onStart({
      dataset_id: datasetId,
      target_col: targetCol,
      feature_cols: allFeaturesSelected ? [] : featureCols,
      task_type: taskType || undefined,
      primary_metric: primaryMetric || undefined,
      strategy: 'balanced',
      epsilon,
      max_iterations: maxIterations,
      selected_models: selectedModels.length > 0 ? selectedModels : undefined,
      test_size: testSize,
      multi_objective: multiObjective,
      objective_weights: Object.keys(objectiveWeights).length > 0 ? objectiveWeights : undefined,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Dataset */}
      <div>
        <label className="block text-sm text-gray-400 mb-1">Dataset</label>
        <select
          value={datasetId}
          onChange={(e) => setDatasetId(e.target.value)}
          className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200"
          required
        >
          <option value="">Select a dataset...</option>
          {datasets.map((ds) => (
            <option key={ds.id} value={ds.id}>
              {ds.name} ({ds.num_rows} rows, {ds.num_cols} cols)
            </option>
          ))}
        </select>
      </div>

      {/* Data Preview Toggle */}
      {datasetId && preview.length > 0 && (
        <div>
          <button
            type="button"
            onClick={() => setShowPreview(!showPreview)}
            className="text-xs text-blue-400 hover:text-blue-300"
          >
            {showPreview ? 'Hide' : 'Show'} Data Preview
          </button>
          {showPreview && (
            <div className="mt-2 overflow-x-auto max-h-48 border border-gray-700 rounded-lg">
              <table className="text-xs w-full">
                <thead>
                  <tr className="border-b border-gray-700 bg-gray-800/50">
                    {Object.keys(preview[0]).map((k) => (
                      <th key={k} className="text-left px-2 py-1 text-gray-400 whitespace-nowrap">{k}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.map((row, i) => (
                    <tr key={i} className="border-b border-gray-800">
                      {Object.values(row).map((v, j) => (
                        <td key={j} className="px-2 py-1 text-gray-300 whitespace-nowrap">
                          {String(v ?? '')}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Target Column (Dependent Variable) */}
      {columns.length > 0 && (
        <div>
          <label className="block text-sm text-gray-400 mb-1">
            Target Variable (Dependent)
          </label>
          <select
            value={targetCol}
            onChange={(e) => setTargetCol(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200"
            required
          >
            <option value="">Select target column...</option>
            {columns.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name} ({c.dtype}, {c.unique_count} unique)
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Feature Columns (Independent Variables) */}
      {columns.length > 0 && targetCol && (
        <div>
          <div className="flex items-center justify-between mb-1">
            <label className="text-sm text-gray-400">
              Features (Independent) - {featureCols.length}/{nonTargetCols.length}
            </label>
            <button
              type="button"
              onClick={toggleAllFeatures}
              className="text-xs text-blue-400 hover:text-blue-300"
            >
              {allFeaturesSelected ? 'Deselect All' : 'Select All'}
            </button>
          </div>
          <div className="max-h-40 overflow-y-auto border border-gray-700 rounded-lg p-2 space-y-1">
            {nonTargetCols.map((c) => (
              <label
                key={c.name}
                className="flex items-center gap-2 py-0.5 px-1 rounded hover:bg-gray-800/50 cursor-pointer"
              >
                <input
                  type="checkbox"
                  checked={featureCols.includes(c.name)}
                  onChange={() => toggleFeature(c.name)}
                  className="rounded border-gray-600 bg-gray-800 text-blue-500"
                />
                <span className="text-sm text-gray-200 flex-1">{c.name}</span>
                <span className="text-xs text-gray-500">{c.dtype}</span>
                {c.null_pct > 0 && (
                  <span className="text-xs text-yellow-500">{(c.null_pct * 100).toFixed(0)}% null</span>
                )}
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Task Type & Metric */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-sm text-gray-400 mb-1">Task Type</label>
          <select
            value={taskType}
            onChange={(e) => setTaskType(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200"
          >
            <option value="">Auto-detect</option>
            <option value="classification">Classification</option>
            <option value="regression">Regression</option>
          </select>
        </div>
        <div>
          <label className="block text-sm text-gray-400 mb-1">Primary Metric</label>
          <select
            value={primaryMetric}
            onChange={(e) => setPrimaryMetric(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200"
          >
            <option value="">Auto</option>
            {metricOptions.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Model Zoo */}
      <div>
        <button
          type="button"
          onClick={() => setShowModelZoo(!showModelZoo)}
          className="flex items-center justify-between w-full text-sm text-gray-400 hover:text-gray-300 mb-1"
        >
          <span>
            Model Zoo {selectedModels.length > 0 ? `(${selectedModels.length} selected)` : '(all models)'}
          </span>
          <span className="text-xs">{showModelZoo ? '\u25B2' : '\u25BC'}</span>
        </button>
        {showModelZoo && (
          <div className="border border-gray-700 rounded-lg p-2 space-y-1 max-h-52 overflow-y-auto">
            {selectedModels.length > 0 && (
              <button
                type="button"
                onClick={() => setSelectedModels([])}
                className="text-xs text-blue-400 hover:text-blue-300 mb-1"
              >
                Clear selection (use all)
              </button>
            )}
            {allModels.map((model) => (
              <label
                key={model.name}
                className="flex items-center gap-2 py-1 px-2 rounded hover:bg-gray-800/50 cursor-pointer"
              >
                <input
                  type="checkbox"
                  checked={selectedModels.length === 0 || selectedModels.includes(model.name)}
                  onChange={() => toggleModel(model.name)}
                  className="rounded border-gray-600 bg-gray-800 text-blue-500"
                />
                <div className="flex-1 min-w-0">
                  <span className="text-sm text-gray-200">{model.name}</span>
                  <div className="flex gap-1 mt-0.5">
                    {model.task_types.map((t) => (
                      <span key={t} className="text-[10px] px-1.5 py-0.5 rounded bg-gray-800 text-gray-400 border border-gray-700">
                        {t}
                      </span>
                    ))}
                  </div>
                </div>
              </label>
            ))}
            {allModels.length === 0 && (
              <p className="text-xs text-gray-500 italic px-2 py-1">Loading models...</p>
            )}
          </div>
        )}
      </div>

      {/* Strategy */}
      <div>
        <label className="block text-sm text-gray-400 mb-1">
          Explore/Exploit (Epsilon: {epsilon.toFixed(2)})
        </label>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={epsilon}
          onChange={(e) => setEpsilon(Number(e.target.value))}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-gray-500">
          <span>Exploit (refine best)</span>
          <span>Explore (try new)</span>
        </div>
      </div>

      {/* Max Iterations */}
      <div>
        <label className="block text-sm text-gray-400 mb-1">Max Iterations: {maxIterations}</label>
        <input
          type="range"
          min={3}
          max={30}
          step={1}
          value={maxIterations}
          onChange={(e) => setMaxIterations(Number(e.target.value))}
          className="w-full"
        />
      </div>

      {/* Advanced Settings */}
      <div>
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-xs text-gray-500 hover:text-gray-400"
        >
          {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
        </button>
        {showAdvanced && (
          <div className="mt-2 space-y-3 border border-gray-700 rounded-lg p-3">
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Test Size: {(testSize * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min={0.1}
                max={0.5}
                step={0.05}
                value={testSize}
                onChange={(e) => setTestSize(Number(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>10% test</span>
                <span>50% test</span>
              </div>
            </div>

            {/* Multi-Objective Optimization */}
            <div className="border-t border-gray-700 pt-3">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={multiObjective}
                  onChange={(e) => setMultiObjective(e.target.checked)}
                  className="rounded border-gray-600 bg-gray-800 text-blue-500"
                />
                <span className="text-sm text-gray-300">Multi-Objective Optimization</span>
              </label>
              <p className="text-xs text-gray-500 mt-1">Optimize for both accuracy and training time</p>

              {multiObjective && (
                <div className="mt-2">
                  <label className="block text-sm text-gray-400 mb-1">
                    Accuracy Weight: {(metricWeight * 100).toFixed(0)}% / Time: {((1 - metricWeight) * 100).toFixed(0)}%
                  </label>
                  <input
                    type="range"
                    min={0.1}
                    max={0.9}
                    step={0.1}
                    value={metricWeight}
                    onChange={(e) => setMetricWeight(Number(e.target.value))}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>Favor speed</span>
                    <span>Favor accuracy</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      <button
        type="submit"
        disabled={disabled || !datasetId || !targetCol || featureCols.length === 0}
        className="w-full py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium text-sm"
      >
        {disabled ? 'Running...' : 'Start Experiment'}
      </button>
    </form>
  );
}
