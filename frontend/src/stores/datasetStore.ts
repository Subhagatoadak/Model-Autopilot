import { create } from 'zustand';
import type { Dataset, DatasetDetail } from '../types/dataset';
import { listDatasets, getDataset, uploadDataset } from '../api/datasets';

interface DatasetStore {
  datasets: Dataset[];
  selectedDataset: DatasetDetail | null;
  loading: boolean;
  error: string | null;

  fetchDatasets: () => Promise<void>;
  selectDataset: (id: string) => Promise<void>;
  upload: (file: File, name?: string, targetCol?: string, taskType?: string) => Promise<Dataset>;
  clearError: () => void;
}

export const useDatasetStore = create<DatasetStore>((set) => ({
  datasets: [],
  selectedDataset: null,
  loading: false,
  error: null,

  fetchDatasets: async () => {
    set({ loading: true, error: null });
    try {
      const datasets = await listDatasets();
      set({ datasets, loading: false });
    } catch (e: any) {
      set({ error: e.message, loading: false });
    }
  },

  selectDataset: async (id: string) => {
    set({ loading: true, error: null });
    try {
      const detail = await getDataset(id);
      set({ selectedDataset: detail, loading: false });
    } catch (e: any) {
      set({ error: e.message, loading: false });
    }
  },

  upload: async (file, name, targetCol, taskType) => {
    set({ loading: true, error: null });
    try {
      const dataset = await uploadDataset(file, name, targetCol, taskType);
      set((state) => ({
        datasets: [dataset, ...state.datasets],
        loading: false,
      }));
      return dataset;
    } catch (e: any) {
      set({ error: e.message, loading: false });
      throw e;
    }
  },

  clearError: () => set({ error: null }),
}));
