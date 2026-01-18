/**
 * React Query hooks for metrics and data fetching
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import apiClient from '@/api/client';
import type {
  ModelMetrics,
  Experiment,
  ExperimentMetrics,
  FunnelData,
  SystemMetrics,
  StatisticalResult,
} from '@/types';

// Query keys
export const queryKeys = {
  modelMetrics: ['modelMetrics'] as const,
  experiments: ['experiments'] as const,
  experiment: (name: string) => ['experiment', name] as const,
  experimentMetrics: (name: string) => ['experimentMetrics', name] as const,
  funnelData: ['funnelData'] as const,
  systemMetrics: ['systemMetrics'] as const,
  statisticalAnalysis: (name: string) => ['statisticalAnalysis', name] as const,
};

// Fetch model metrics
export function useModelMetrics() {
  return useQuery({
    queryKey: queryKeys.modelMetrics,
    queryFn: async (): Promise<ModelMetrics[]> => {
      const { data } = await apiClient.get('/v1/metrics/models');
      return data;
    },
    // Mock data for development
    placeholderData: [
      {
        model: 'Popular',
        precision_at_5: 0.042, precision_at_10: 0.045, precision_at_20: 0.041,
        recall_at_5: 0.028, recall_at_10: 0.032, recall_at_20: 0.038,
        ndcg_at_5: 0.048, ndcg_at_10: 0.051, ndcg_at_20: 0.054,
        hit_rate: 0.28, mrr: 0.15, coverage: 0.02, diversity: 0.12,
      },
      {
        model: 'ALS',
        precision_at_5: 0.068, precision_at_10: 0.072, precision_at_20: 0.065,
        recall_at_5: 0.048, recall_at_10: 0.054, recall_at_20: 0.062,
        ndcg_at_5: 0.082, ndcg_at_10: 0.089, ndcg_at_20: 0.094,
        hit_rate: 0.42, mrr: 0.28, coverage: 0.18, diversity: 0.45,
      },
      {
        model: 'NCF',
        precision_at_5: 0.071, precision_at_10: 0.075, precision_at_20: 0.068,
        recall_at_5: 0.052, recall_at_10: 0.057, recall_at_20: 0.065,
        ndcg_at_5: 0.088, ndcg_at_10: 0.094, ndcg_at_20: 0.098,
        hit_rate: 0.45, mrr: 0.31, coverage: 0.22, diversity: 0.48,
      },
      {
        model: 'GRU4Rec',
        precision_at_5: 0.074, precision_at_10: 0.078, precision_at_20: 0.071,
        recall_at_5: 0.055, recall_at_10: 0.061, recall_at_20: 0.068,
        ndcg_at_5: 0.091, ndcg_at_10: 0.097, ndcg_at_20: 0.102,
        hit_rate: 0.48, mrr: 0.33, coverage: 0.25, diversity: 0.52,
      },
      {
        model: 'SASRec',
        precision_at_5: 0.076, precision_at_10: 0.081, precision_at_20: 0.074,
        recall_at_5: 0.058, recall_at_10: 0.064, recall_at_20: 0.071,
        ndcg_at_5: 0.094, ndcg_at_10: 0.101, ndcg_at_20: 0.106,
        hit_rate: 0.51, mrr: 0.35, coverage: 0.28, diversity: 0.55,
      },
      {
        model: 'Hybrid',
        precision_at_5: 0.078, precision_at_10: 0.084, precision_at_20: 0.077,
        recall_at_5: 0.059, recall_at_10: 0.066, recall_at_20: 0.074,
        ndcg_at_5: 0.096, ndcg_at_10: 0.105, ndcg_at_20: 0.110,
        hit_rate: 0.54, mrr: 0.38, coverage: 0.32, diversity: 0.58,
      },
    ],
  });
}

// Fetch experiments
export function useExperiments() {
  return useQuery({
    queryKey: queryKeys.experiments,
    queryFn: async (): Promise<Experiment[]> => {
      const { data } = await apiClient.get('/v1/experiments');
      return data;
    },
    placeholderData: [
      {
        name: 'hybrid_vs_sasrec',
        description: 'Compare Hybrid model with SASRec',
        variants: [
          { name: 'control', percentage: 50, model_name: 'hybrid', config: {}, description: 'Hybrid model' },
          { name: 'treatment', percentage: 50, model_name: 'sasrec', config: {}, description: 'SASRec model' },
        ],
        status: 'running' as const,
        start_date: '2025-01-15T10:00:00',
        end_date: null,
        target_metric: 'ndcg@10',
        min_sample_size: 1000,
        created_at: '2025-01-14T09:00:00',
        metadata: {},
      },
    ],
  });
}

// Fetch experiment metrics
export function useExperimentMetrics(experimentName: string) {
  return useQuery({
    queryKey: queryKeys.experimentMetrics(experimentName),
    queryFn: async (): Promise<ExperimentMetrics> => {
      const { data } = await apiClient.get(`/v1/experiments/${experimentName}/metrics`);
      return data;
    },
    enabled: !!experimentName,
  });
}

// Fetch funnel data
export function useFunnelData() {
  return useQuery({
    queryKey: queryKeys.funnelData,
    queryFn: async (): Promise<FunnelData> => {
      const { data } = await apiClient.get('/v1/metrics/funnel');
      return data;
    },
    placeholderData: {
      stages: [
        { name: 'View', count: 2660000, percentage: 100 },
        { name: 'Add to Cart', count: 69000, percentage: 2.6 },
        { name: 'Transaction', count: 22000, percentage: 0.83 },
      ],
      conversion_rates: {
        view_to_cart: 0.026,
        cart_to_transaction: 0.319,
        overall: 0.0083,
      },
    },
  });
}

// Fetch system metrics
export function useSystemMetrics() {
  return useQuery({
    queryKey: queryKeys.systemMetrics,
    queryFn: async (): Promise<SystemMetrics> => {
      const { data } = await apiClient.get('/v1/metrics/system');
      return data;
    },
    refetchInterval: 30000, // Refresh every 30 seconds
    placeholderData: {
      total_users: 1407580,
      total_items: 235061,
      total_events: 2756101,
      active_experiments: 1,
      recommendations_served: 125430,
      avg_latency_ms: 45,
    },
  });
}

// Statistical analysis
export function useStatisticalAnalysis(experimentName: string) {
  return useQuery({
    queryKey: queryKeys.statisticalAnalysis(experimentName),
    queryFn: async (): Promise<StatisticalResult> => {
      const { data } = await apiClient.get(`/v1/experiments/${experimentName}/analysis`);
      return data;
    },
    enabled: !!experimentName,
    placeholderData: {
      control_mean: 0.102,
      treatment_mean: 0.108,
      relative_lift: 0.059,
      relative_lift_percent: 5.9,
      p_value: 0.034,
      confidence_level: 0.95,
      is_significant: true,
      confidence_interval: [0.001, 0.011],
      control_n: 5420,
      treatment_n: 5380,
      test_type: 'welch_t_test',
    },
  });
}

// Mutations

// Start experiment
export function useStartExperiment() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (experimentName: string) => {
      const { data } = await apiClient.post(`/v1/experiments/${experimentName}/start`);
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.experiments });
    },
  });
}

// Stop experiment
export function useStopExperiment() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (experimentName: string) => {
      const { data } = await apiClient.post(`/v1/experiments/${experimentName}/stop`);
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.experiments });
    },
  });
}

// Create experiment
export function useCreateExperiment() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (experiment: Partial<Experiment>) => {
      const { data } = await apiClient.post('/v1/experiments', experiment);
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.experiments });
    },
  });
}
