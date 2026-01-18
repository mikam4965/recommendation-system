/**
 * Model Comparison Chart Component
 */

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Card, CardHeader } from './Card';
import { useModelMetrics } from '@/hooks/useMetrics';

const METRIC_COLORS = {
  precision: '#3b82f6', // blue
  recall: '#10b981',    // green
  ndcg: '#f59e0b',      // amber
};

interface ModelComparisonProps {
  metricK?: number;
}

export function ModelComparison({ metricK = 10 }: ModelComparisonProps) {
  const { data: metrics, isLoading, error } = useModelMetrics();

  if (isLoading) {
    return (
      <Card>
        <CardHeader title="Model Comparison" />
        <div className="h-80 flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader title="Model Comparison" />
        <div className="h-80 flex items-center justify-center text-red-500">
          Error loading metrics
        </div>
      </Card>
    );
  }

  const chartData = metrics?.map((m) => ({
    model: m.model,
    precision: m[`precision_at_${metricK}` as keyof typeof m] as number,
    recall: m[`recall_at_${metricK}` as keyof typeof m] as number,
    ndcg: m[`ndcg_at_${metricK}` as keyof typeof m] as number,
  }));

  return (
    <Card>
      <CardHeader
        title="Model Comparison"
        subtitle={`Metrics @${metricK}`}
      />
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="model"
              tick={{ fontSize: 12 }}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 12 }}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
              }}
              formatter={(value: number) => value.toFixed(4)}
            />
            <Legend />
            <Bar
              dataKey="precision"
              fill={METRIC_COLORS.precision}
              name={`Precision@${metricK}`}
              radius={[4, 4, 0, 0]}
            />
            <Bar
              dataKey="recall"
              fill={METRIC_COLORS.recall}
              name={`Recall@${metricK}`}
              radius={[4, 4, 0, 0]}
            />
            <Bar
              dataKey="ndcg"
              fill={METRIC_COLORS.ndcg}
              name={`NDCG@${metricK}`}
              radius={[4, 4, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

export function ModelMetricsTable() {
  const { data: metrics, isLoading } = useModelMetrics();

  if (isLoading || !metrics) {
    return null;
  }

  return (
    <Card>
      <CardHeader title="Detailed Metrics" />
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Model
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                P@10
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                R@10
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                NDCG@10
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Hit Rate
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                MRR
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Coverage
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {metrics.map((m, idx) => (
              <tr key={m.model} className={idx === metrics.length - 1 ? 'bg-primary-50' : ''}>
                <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900">
                  {m.model}
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                  {m.precision_at_10.toFixed(4)}
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                  {m.recall_at_10.toFixed(4)}
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                  {m.ndcg_at_10.toFixed(4)}
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                  {(m.hit_rate * 100).toFixed(1)}%
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                  {m.mrr.toFixed(4)}
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                  {(m.coverage * 100).toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
