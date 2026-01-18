/**
 * System Metrics Dashboard Cards
 */

import { useSystemMetrics } from '@/hooks/useMetrics';
import {
  Users,
  Package,
  Activity,
  FlaskConical,
  Sparkles,
  Clock,
} from 'lucide-react';
import { clsx } from 'clsx';

interface MetricCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  change?: number;
  subtitle?: string;
  color?: 'blue' | 'green' | 'amber' | 'purple';
}

function MetricCard({
  title,
  value,
  icon,
  change,
  subtitle,
  color = 'blue',
}: MetricCardProps) {
  const colorClasses = {
    blue: 'bg-blue-50 text-blue-600',
    green: 'bg-green-50 text-green-600',
    amber: 'bg-amber-50 text-amber-600',
    purple: 'bg-purple-50 text-purple-600',
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-500">{title}</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">
            {typeof value === 'number' ? value.toLocaleString() : value}
          </p>
          {subtitle && (
            <p className="text-xs text-gray-400 mt-1">{subtitle}</p>
          )}
          {change !== undefined && (
            <p
              className={clsx(
                'text-xs font-medium mt-1',
                change >= 0 ? 'text-green-600' : 'text-red-600'
              )}
            >
              {change >= 0 ? '+' : ''}
              {change.toFixed(1)}% vs last period
            </p>
          )}
        </div>
        <div className={clsx('p-3 rounded-lg', colorClasses[color])}>
          {icon}
        </div>
      </div>
    </div>
  );
}

export function SystemMetrics() {
  const { data: metrics, isLoading } = useSystemMetrics();

  if (isLoading || !metrics) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        {Array.from({ length: 6 }).map((_, i) => (
          <div
            key={i}
            className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 animate-pulse"
          >
            <div className="h-4 bg-gray-200 rounded w-1/2 mb-2" />
            <div className="h-8 bg-gray-200 rounded w-3/4" />
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
      <MetricCard
        title="Total Users"
        value={metrics.total_users}
        icon={<Users className="w-6 h-6" />}
        color="blue"
        subtitle="Unique visitors"
      />
      <MetricCard
        title="Total Items"
        value={metrics.total_items}
        icon={<Package className="w-6 h-6" />}
        color="green"
        subtitle="Product catalog"
      />
      <MetricCard
        title="Total Events"
        value={metrics.total_events}
        icon={<Activity className="w-6 h-6" />}
        color="amber"
        subtitle="User interactions"
      />
      <MetricCard
        title="Active Experiments"
        value={metrics.active_experiments}
        icon={<FlaskConical className="w-6 h-6" />}
        color="purple"
        subtitle="A/B tests running"
      />
      <MetricCard
        title="Recommendations"
        value={metrics.recommendations_served}
        icon={<Sparkles className="w-6 h-6" />}
        color="blue"
        subtitle="Served today"
        change={12.5}
      />
      <MetricCard
        title="Avg Latency"
        value={`${metrics.avg_latency_ms}ms`}
        icon={<Clock className="w-6 h-6" />}
        color="green"
        subtitle="P50 response time"
      />
    </div>
  );
}
