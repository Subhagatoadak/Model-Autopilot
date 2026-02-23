import { create } from 'zustand';
import type { Experiment, ExperimentConfig, ExperimentDetail, ExperimentRun } from '../types/experiment';
import type { AgentStep, ThoughtNode, WSMessage } from '../types/agent';
import { createExperiment, listExperiments, getExperiment, stopExperiment } from '../api/experiments';

interface ExperimentStore {
  experiments: Experiment[];
  currentExperiment: ExperimentDetail | null;
  runs: ExperimentRun[];
  agentSteps: AgentStep[];
  thoughtNodes: ThoughtNode[];
  thoughtsByIteration: Record<number, ThoughtNode[]>;
  selectedPathByIteration: Record<number, string[]>;
  selectedPath: string[];
  selectedThoughtIteration: number | null;
  status: string;
  currentIteration: number;
  maxIterations: number;
  loading: boolean;
  error: string | null;

  fetchExperiments: () => Promise<void>;
  loadExperiment: (id: string) => Promise<void>;
  startExperiment: (config: ExperimentConfig) => Promise<Experiment>;
  stopCurrentExperiment: () => Promise<void>;
  handleWSMessage: (msg: WSMessage) => void;
  setSelectedThoughtIteration: (iteration: number | null) => void;
  reset: () => void;
}

export const useExperimentStore = create<ExperimentStore>((set, get) => ({
  experiments: [],
  currentExperiment: null,
  runs: [],
  agentSteps: [],
  thoughtNodes: [],
  thoughtsByIteration: {},
  selectedPathByIteration: {},
  selectedPath: [],
  selectedThoughtIteration: null,
  status: 'idle',
  currentIteration: 0,
  maxIterations: 10,
  loading: false,
  error: null,

  fetchExperiments: async () => {
    set({ loading: true });
    try {
      const experiments = await listExperiments();
      set({ experiments, loading: false });
    } catch (e: any) {
      set({ error: e.message, loading: false });
    }
  },

  loadExperiment: async (id: string) => {
    set({ loading: true });
    try {
      const detail = await getExperiment(id);
      set({
        currentExperiment: detail,
        runs: detail.runs,
        status: detail.status,
        loading: false,
      });
    } catch (e: any) {
      set({ error: e.message, loading: false });
    }
  },

  startExperiment: async (config: ExperimentConfig) => {
    set({ loading: true, agentSteps: [], thoughtNodes: [], selectedPath: [], runs: [] });
    try {
      const experiment = await createExperiment(config);
      set({
        currentExperiment: { ...experiment, config, runs: [], best_metrics: {} },
        status: 'running',
        maxIterations: config.max_iterations || 10,
        loading: false,
      });
      return experiment;
    } catch (e: any) {
      set({ error: e.message, loading: false });
      throw e;
    }
  },

  stopCurrentExperiment: async () => {
    const exp = get().currentExperiment;
    if (!exp) return;
    try {
      await stopExperiment(exp.id);
      set({ status: 'stopped' });
    } catch (e: any) {
      set({ error: e.message });
    }
  },

  handleWSMessage: (msg: WSMessage) => {
    switch (msg.type) {
      case 'agent_step': {
        const step = msg.data;
        // Clear thought tree when a new recommendation phase starts
        if (step.agent === 'recommendation' && step.step === 'thinking') {
          set((state) => ({
            agentSteps: [...state.agentSteps, { ...step, timestamp: Date.now() }],
            thoughtNodes: [],
            selectedPath: [],
          }));
        } else {
          set((state) => ({
            agentSteps: [...state.agentSteps, { ...step, timestamp: Date.now() }],
          }));
        }
        break;
      }

      case 'thought_node': {
        const iteration = msg.data.iteration;
        set((state) => {
          const iterThoughts = state.thoughtsByIteration[iteration] || [];
          return {
            thoughtNodes: [...state.thoughtNodes, msg.data],
            thoughtsByIteration: {
              ...state.thoughtsByIteration,
              [iteration]: [...iterThoughts, msg.data],
            },
            selectedThoughtIteration: iteration, // Auto-select latest iteration
          };
        });
        break;
      }

      case 'thought_selected': {
        const iteration = msg.data.iteration;
        set((state) => ({
          selectedPath: msg.data.path,
          selectedPathByIteration: {
            ...state.selectedPathByIteration,
            [iteration]: msg.data.path,
          },
        }));
        break;
      }

      case 'run_complete':
        set((state) => ({
          runs: [
            ...state.runs,
            {
              id: `run-${msg.data.iteration}`,
              experiment_id: state.currentExperiment?.id || '',
              iteration: msg.data.iteration,
              model_name: msg.data.model_name,
              hyperparameters: {},
              metrics: msg.data.metrics,
              train_time_s: msg.data.train_time_s,
              ucb_score: msg.data.ucb_score,
              created_at: null,
            },
          ],
        }));
        break;

      case 'metrics_update':
        set({ currentIteration: msg.data.iteration });
        break;

      case 'experiment_status':
        set((state) => ({
          status: msg.data.status,
          currentIteration: msg.data.iteration ?? state.currentIteration,
          maxIterations: msg.data.max_iterations ?? state.maxIterations,
        }));
        break;
    }
  },

  setSelectedThoughtIteration: (iteration: number | null) => {
    set({ selectedThoughtIteration: iteration });
  },

  reset: () => {
    set({
      currentExperiment: null,
      runs: [],
      agentSteps: [],
      thoughtNodes: [],
      thoughtsByIteration: {},
      selectedPathByIteration: {},
      selectedPath: [],
      selectedThoughtIteration: null,
      status: 'idle',
      currentIteration: 0,
      error: null,
    });
  },
}));
