import { useCallback, useEffect, useState } from 'react';
import { useExperimentStore } from '../stores/experimentStore';
import { useExperimentWebSocket } from '../api/websocket';
import { getDatasetVersions } from '../api/experiments';
import ExperimentConfigForm from '../components/ExperimentConfig';
import AgentTimeline from '../components/AgentTimeline';
import ThoughtTreeViz from '../components/ThoughtTree';
import MetricsChart from '../components/MetricsChart';
import ModelLeaderboard from '../components/ModelLeaderboard';
import ExperimentSummary from '../components/ExperimentSummary';
import ModelComparison from '../components/ModelComparison';
import TrainingTimeChart from '../components/TrainingTimeChart';
import DataVersionTimeline from '../components/DataVersionTimeline';
import type { ExperimentConfig } from '../types/experiment';
import type { DatasetVersion } from '../types/dataset';

type Tab = 'overview' | 'thought_tree' | 'timeline' | 'data_versions';

export default function ExperimentPage() {
  const {
    currentExperiment,
    runs,
    agentSteps,
    thoughtNodes,
    thoughtsByIteration,
    selectedPathByIteration,
    selectedPath,
    selectedThoughtIteration,
    setSelectedThoughtIteration,
    status,
    currentIteration,
    maxIterations,
    startExperiment,
    stopCurrentExperiment,
    handleWSMessage,
    error,
  } = useExperimentStore();

  const [activeTab, setActiveTab] = useState<Tab>('overview');
  const [dataVersions, setDataVersions] = useState<DatasetVersion[]>([]);

  const onMessage = useCallback(handleWSMessage, [handleWSMessage]);

  useExperimentWebSocket(
    currentExperiment?.id || null,
    onMessage,
  );

  // Fetch dataset versions when experiment changes
  useEffect(() => {
    if (currentExperiment?.id) {
      getDatasetVersions(currentExperiment.id)
        .then(setDataVersions)
        .catch((err) => console.error('Failed to fetch dataset versions:', err));
    }
  }, [currentExperiment?.id]);

  const handleStart = async (config: ExperimentConfig) => {
    await startExperiment(config);
  };

  const isRunning = status === 'running';
  const isGeneratingThought = agentSteps.some(
    (s) => s.agent === 'recommendation' && s.step === 'thinking'
  ) && thoughtNodes.length === 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">Experiment</h2>
        <div className="flex items-center gap-3">
          {isRunning && (
            <>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                <span className="text-sm text-gray-400">
                  Iteration {currentIteration}/{maxIterations}
                </span>
              </div>
              <div className="w-32 bg-gray-800 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all"
                  style={{ width: `${(currentIteration / maxIterations) * 100}%` }}
                />
              </div>
              <button
                onClick={stopCurrentExperiment}
                className="px-3 py-1.5 bg-red-600/20 text-red-400 border border-red-600/30 rounded-lg text-sm hover:bg-red-600/30"
              >
                Stop
              </button>
            </>
          )}
          {status === 'completed' && (
            <>
              <span className="px-3 py-1.5 bg-green-600/20 text-green-400 border border-green-600/30 rounded-lg text-sm">
                Completed
              </span>
              <button
                onClick={() => window.open(`/api/experiments/${currentExperiment?.id}/export`, '_blank')}
                className="px-3 py-1.5 bg-blue-600/20 text-blue-400 border border-blue-600/30 rounded-lg text-sm hover:bg-blue-600/30 flex items-center gap-1.5"
                title="Download best model"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Download Model
              </button>
              <button
                onClick={() => window.open(`/api/experiments/${currentExperiment?.id}/export-pdf`, '_blank')}
                className="px-3 py-1.5 bg-purple-600/20 text-purple-400 border border-purple-600/30 rounded-lg text-sm hover:bg-purple-600/30 flex items-center gap-1.5"
                title="Download comprehensive PDF report"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Export PDF
              </button>
            </>
          )}
          {status === 'failed' && (
            <span className="px-3 py-1.5 bg-red-600/20 text-red-400 border border-red-600/30 rounded-lg text-sm">
              Failed
            </span>
          )}
          {status === 'stopped' && (
            <span className="px-3 py-1.5 bg-yellow-600/20 text-yellow-400 border border-yellow-600/30 rounded-lg text-sm">
              Stopped
            </span>
          )}
        </div>
      </div>

      {error && (
        <div className="bg-red-900/30 border border-red-800 text-red-300 px-4 py-3 rounded-lg text-sm">
          {error}
        </div>
      )}

      {/* Summary Cards */}
      {runs.length > 0 && (
        <ExperimentSummary
          runs={runs}
          primaryMetric={currentExperiment?.config?.primary_metric}
        />
      )}

      {/* Main Grid */}
      <div className="grid grid-cols-12 gap-6">
        {/* Left: Config */}
        <div className="col-span-3">
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <h3 className="text-sm font-semibold text-gray-400 uppercase mb-3">Configuration</h3>
            <ExperimentConfigForm onStart={handleStart} disabled={isRunning} />
          </div>
        </div>

        {/* Center: Tabbed View */}
        <div className="col-span-5 space-y-4">
          {/* Tabs */}
          <div className="flex gap-1 bg-gray-900 border border-gray-800 rounded-xl p-1">
            {([
              ['overview', 'Overview'],
              ['thought_tree', 'Tree of Thought'],
              ['timeline', 'Agent Timeline'],
              ['data_versions', 'Data Versions'],
            ] as const).map(([key, label]) => (
              <button
                key={key}
                onClick={() => setActiveTab(key)}
                className={`flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-colors ${
                  activeTab === key
                    ? 'bg-gray-800 text-white'
                    : 'text-gray-500 hover:text-gray-300'
                }`}
              >
                {label}
                {key === 'thought_tree' && thoughtNodes.length > 0 && (
                  <span className="ml-1.5 text-xs bg-pink-600/30 text-pink-400 px-1.5 py-0.5 rounded-full">
                    {thoughtNodes.length}
                  </span>
                )}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          {activeTab === 'overview' && (
            <div className="space-y-4">
              {/* Metrics Chart */}
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
                <h3 className="text-sm font-semibold text-gray-400 uppercase mb-3">
                  Performance Trajectory
                </h3>
                <MetricsChart
                  runs={runs}
                  primaryMetric={currentExperiment?.config?.primary_metric}
                />
              </div>

              {/* Model Comparison */}
              {runs.length > 1 && (
                <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
                  <h3 className="text-sm font-semibold text-gray-400 uppercase mb-3">
                    Model Comparison
                  </h3>
                  <ModelComparison
                    runs={runs}
                    primaryMetric={currentExperiment?.config?.primary_metric}
                  />
                </div>
              )}

              {/* Training Time */}
              {runs.length > 1 && (
                <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
                  <h3 className="text-sm font-semibold text-gray-400 uppercase mb-3">
                    Training Time per Iteration
                  </h3>
                  <TrainingTimeChart runs={runs} />
                </div>
              )}
            </div>
          )}

          {activeTab === 'thought_tree' && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-gray-400 uppercase">
                  Tree of Thought - Recommendation Strategies
                </h3>
                {Object.keys(thoughtsByIteration).length > 0 && (
                  <select
                    value={selectedThoughtIteration ?? ''}
                    onChange={(e) => setSelectedThoughtIteration(e.target.value ? Number(e.target.value) : null)}
                    className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                  >
                    {Object.keys(thoughtsByIteration).sort((a, b) => Number(b) - Number(a)).map((iter) => (
                      <option key={iter} value={iter}>
                        Iteration {iter}
                      </option>
                    ))}
                  </select>
                )}
              </div>
              <ThoughtTreeViz
                thoughtNodes={selectedThoughtIteration !== null ? (thoughtsByIteration[selectedThoughtIteration] || []) : thoughtNodes}
                selectedPath={selectedThoughtIteration !== null ? (selectedPathByIteration[selectedThoughtIteration] || []) : selectedPath}
                rootThought={currentExperiment ? `Optimizing model performance` : undefined}
                isGenerating={isGeneratingThought}
              />
              {((selectedThoughtIteration !== null ? selectedPathByIteration[selectedThoughtIteration] : selectedPath) || []).length > 0 && (
                <div className="mt-3 p-3 bg-blue-950/30 border border-blue-800/30 rounded-lg">
                  <div className="text-xs text-blue-400 font-semibold mb-1">Selected Strategy Path</div>
                  <div className="text-sm text-gray-300">
                    {(selectedThoughtIteration !== null ? selectedPathByIteration[selectedThoughtIteration] : selectedPath).join(' → ')}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'timeline' && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <h3 className="text-sm font-semibold text-gray-400 uppercase mb-3">
                Agent Activity Log ({agentSteps.length} events)
              </h3>
              <AgentTimeline steps={agentSteps} />
            </div>
          )}

          {activeTab === 'data_versions' && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <h3 className="text-sm font-semibold text-gray-400 uppercase mb-3">
                Dataset Version Lineage ({dataVersions.length} versions)
              </h3>
              <DataVersionTimeline versions={dataVersions} />
            </div>
          )}
        </div>

        {/* Right: Agent Timeline (compact) + Leaderboard */}
        <div className="col-span-4 space-y-4">
          {/* Live Agent Feed */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <h3 className="text-sm font-semibold text-gray-400 uppercase mb-3">
              Live Feed
              {isRunning && (
                <span className="ml-2 inline-block w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              )}
            </h3>
            <div className="max-h-[300px] overflow-y-auto">
              <AgentTimeline steps={agentSteps.slice(-15)} />
            </div>
          </div>

          {/* Mini Leaderboard */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <h3 className="text-sm font-semibold text-gray-400 uppercase mb-3">
              Top Models
            </h3>
            <ModelLeaderboard
              runs={runs}
              primaryMetric={currentExperiment?.config?.primary_metric}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
