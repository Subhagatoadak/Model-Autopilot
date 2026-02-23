import api from './client';
import type { Dataset, DatasetDetail } from '../types/dataset';

export async function uploadDataset(file: File, name?: string, targetCol?: string, taskType?: string): Promise<Dataset> {
  const form = new FormData();
  form.append('file', file);
  if (name) form.append('name', name);
  if (targetCol) form.append('target_col', targetCol);
  if (taskType) form.append('task_type', taskType);
  const { data } = await api.post<Dataset>('/datasets/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

export async function listDatasets(): Promise<Dataset[]> {
  const { data } = await api.get<Dataset[]>('/datasets');
  return data;
}

export async function getDataset(id: string): Promise<DatasetDetail> {
  const { data } = await api.get<DatasetDetail>(`/datasets/${id}`);
  return data;
}

export async function getDatasetColumns(id: string): Promise<{ name: string; dtype: string; null_pct: number; unique_count: number; sample_values: string[] }[]> {
  const { data } = await api.get(`/datasets/${id}/columns`);
  return data;
}

export async function getDatasetPreview(id: string, n: number = 10): Promise<Record<string, unknown>[]> {
  const { data } = await api.get(`/datasets/${id}/preview`, { params: { n } });
  return data;
}

export async function connectDatabase(params: {
  name: string;
  db_uri: string;
  query: string;
  target_col?: string;
  task_type?: string;
}): Promise<Dataset> {
  const { data } = await api.post<Dataset>('/datasets/connect', params);
  return data;
}
