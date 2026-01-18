/**
 * Funnel Visualization Component
 */

import { Card, CardHeader } from './Card';
import { useFunnelData } from '@/hooks/useMetrics';
import { clsx } from 'clsx';

const FUNNEL_COLORS = [
  'bg-blue-500',
  'bg-amber-500',
  'bg-green-500',
];

export function FunnelVisualization() {
  const { data: funnelData, isLoading } = useFunnelData();

  if (isLoading || !funnelData) {
    return (
      <Card>
        <CardHeader title="Conversion Funnel" />
        <div className="h-64 flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
        </div>
      </Card>
    );
  }

  const maxCount = funnelData.stages[0]?.count || 1;

  return (
    <Card>
      <CardHeader
        title="Conversion Funnel"
        subtitle="User journey from view to purchase"
      />
      <div className="space-y-4">
        {funnelData.stages.map((stage, idx) => {
          const width = (stage.count / maxCount) * 100;
          return (
            <div key={stage.name} className="relative">
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium text-gray-700">
                  {stage.name}
                </span>
                <span className="text-sm text-gray-500">
                  {stage.count.toLocaleString()} ({stage.percentage.toFixed(2)}%)
                </span>
              </div>
              <div className="h-8 bg-gray-100 rounded-lg overflow-hidden">
                <div
                  className={clsx(
                    'h-full transition-all duration-500 rounded-lg',
                    FUNNEL_COLORS[idx] || 'bg-gray-400'
                  )}
                  style={{ width: `${width}%` }}
                />
              </div>
              {idx < funnelData.stages.length - 1 && (
                <div className="flex justify-center my-2">
                  <svg
                    className="w-4 h-4 text-gray-400"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 14l-7 7m0 0l-7-7m7 7V3"
                    />
                  </svg>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Conversion rates */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <h4 className="text-sm font-medium text-gray-700 mb-3">
          Conversion Rates
        </h4>
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {(funnelData.conversion_rates.view_to_cart * 100).toFixed(2)}%
            </div>
            <div className="text-xs text-gray-500">View → Cart</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-amber-600">
              {(funnelData.conversion_rates.cart_to_transaction * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-gray-500">Cart → Purchase</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {(funnelData.conversion_rates.overall * 100).toFixed(2)}%
            </div>
            <div className="text-xs text-gray-500">Overall</div>
          </div>
        </div>
      </div>
    </Card>
  );
}
