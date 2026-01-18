/**
 * Type definitions for the RecSys Dashboard
 */

// Model metrics
export interface ModelMetrics {
  model: string;
  precision_at_5: number;
  precision_at_10: number;
  precision_at_20: number;
  recall_at_5: number;
  recall_at_10: number;
  recall_at_20: number;
  ndcg_at_5: number;
  ndcg_at_10: number;
  ndcg_at_20: number;
  hit_rate: number;
  mrr: number;
  coverage: number;
  diversity: number;
}

// Experiment types
export interface ExperimentVariant {
  name: string;
  percentage: number;
  model_name: string | null;
  config: Record<string, unknown>;
  description: string;
}

export interface Experiment {
  name: string;
  description: string;
  variants: ExperimentVariant[];
  status: 'draft' | 'running' | 'paused' | 'completed' | 'archived';
  start_date: string | null;
  end_date: string | null;
  target_metric: string;
  min_sample_size: number;
  created_at: string;
  metadata: Record<string, unknown>;
}

export interface ExperimentMetrics {
  experiment_name: string;
  metrics: Record<string, VariantMetrics>;
}

export interface VariantMetrics {
  variants: Record<string, AggregatedMetric>;
  total_count: number;
  total_users: number;
}

export interface AggregatedMetric {
  variant_name: string;
  metric_name: string;
  count: number;
  sum: number;
  mean: number;
  min: number;
  max: number;
  unique_users: number;
}

// Statistical analysis
export interface StatisticalResult {
  control_mean: number;
  treatment_mean: number;
  relative_lift: number;
  relative_lift_percent: number;
  p_value: number;
  confidence_level: number;
  is_significant: boolean;
  confidence_interval: [number, number];
  control_n: number;
  treatment_n: number;
  test_type: string;
}

// Funnel data
export interface FunnelStage {
  name: string;
  count: number;
  percentage: number;
}

export interface FunnelData {
  stages: FunnelStage[];
  conversion_rates: {
    view_to_cart: number;
    cart_to_transaction: number;
    overall: number;
  };
}

// User behavior
export interface UserBehavior {
  user_id: number;
  total_events: number;
  views: number;
  add_to_carts: number;
  transactions: number;
  funnel_stage: string;
  last_activity: string;
}

// System metrics
export interface SystemMetrics {
  total_users: number;
  total_items: number;
  total_events: number;
  active_experiments: number;
  recommendations_served: number;
  avg_latency_ms: number;
}

// Real-time metrics
export interface RealTimeMetric {
  timestamp: string;
  value: number;
  metric_name: string;
}

// API response types
export interface ApiResponse<T> {
  data: T;
  status: 'success' | 'error';
  message?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}
