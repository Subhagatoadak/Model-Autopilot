import api from './client';
import type { Experiment, ExperimentComparison, ExperimentConfig, ExperimentDetail } from '../types/experiment';
import type { ModelInfo } from '../types/agent';
import type { DatasetVersion } from '../types/dataset';

export async function createExperiment(config: ExperimentConfig): Promise<Experiment> {
  const { data } = await api.post<Experiment>('/experiments', config);
  return data;
}

export async function listExperiments(): Promise<Experiment[]> {
  const { data } = await api.get<Experiment[]>('/experiments');
  return data;
}

export async function getExperiment(id: string): Promise<ExperimentDetail> {
  const { data } = await api.get<ExperimentDetail>(`/experiments/${id}`);
  return data;
}

export async function stopExperiment(id: string): Promise<void> {
  await api.post(`/experiments/${id}/stop`);
}

export async function getModelZoo(taskType?: string): Promise<ModelInfo[]> {
  const params = taskType ? { task_type: taskType } : {};
  const { data} = await api.get<ModelInfo[]>('/models/zoo', { params });
  return data;
}

export async function compareExperiments(id1: string, id2: string): Promise<ExperimentComparison> {
  const { data } = await api.get<ExperimentComparison>('/experiments/compare', {
    params: { exp_id_1: id1, exp_id_2: id2 },
  });
  return data;
}

export async function getDatasetVersions(experimentId: string): Promise<DatasetVersion[]> {
  const { data } = await api.get<DatasetVersion[]>(`/experiments/${experimentId}/data-versions`);
  return data;
}
