/**
 * A/B Test Results Component
 */

import { useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Card, CardHeader } from './Card';
import {
  useExperiments,
  useStatisticalAnalysis,
  useStartExperiment,
  useStopExperiment,
} from '@/hooks/useMetrics';
import { clsx } from 'clsx';

const STATUS_COLORS = {
  draft: 'bg-gray-100 text-gray-800',
  running: 'bg-green-100 text-green-800',
  paused: 'bg-yellow-100 text-yellow-800',
  completed: 'bg-blue-100 text-blue-800',
  archived: 'bg-gray-100 text-gray-500',
};

export function ABTestResults() {
  const { data: experiments, isLoading } = useExperiments();
  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null);

  if (isLoading) {
    return (
      <Card>
        <CardHeader title="A/B Test Experiments" />
        <div className="h-64 flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader
          title="A/B Test Experiments"
          subtitle="Active and completed experiments"
        />
        <div className="space-y-4">
          {experiments?.map((exp) => (
            <ExperimentCard
              key={exp.name}
              experiment={exp}
              isSelected={selectedExperiment === exp.name}
              onSelect={() =>
                setSelectedExperiment(
                  selectedExperiment === exp.name ? null : exp.name
                )
              }
            />
          ))}
          {experiments?.length === 0 && (
            <div className="text-center text-gray-500 py-8">
              No experiments found
            </div>
          )}
        </div>
      </Card>

      {selectedExperiment && (
        <ExperimentDetails experimentName={selectedExperiment} />
      )}
    </div>
  );
}

interface ExperimentCardProps {
  experiment: {
    name: string;
    description: string;
    status: keyof typeof STATUS_COLORS;
    variants: Array<{
      name: string;
      percentage: number;
      model_name: string | null;
    }>;
    start_date: string | null;
    target_metric: string;
  };
  isSelected: boolean;
  onSelect: () => void;
}

function ExperimentCard({ experiment, isSelected, onSelect }: ExperimentCardProps) {
  const startMutation = useStartExperiment();
  const stopMutation = useStopExperiment();

  const handleStart = (e: React.MouseEvent) => {
    e.stopPropagation();
    startMutation.mutate(experiment.name);
  };

  const handleStop = (e: React.MouseEvent) => {
    e.stopPropagation();
    stopMutation.mutate(experiment.name);
  };

  return (
    <div
      className={clsx(
        'p-4 border rounded-lg cursor-pointer transition-all',
        isSelected
          ? 'border-primary-500 bg-primary-50'
          : 'border-gray-200 hover:border-gray-300'
      )}
      onClick={onSelect}
    >
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h4 className="font-medium text-gray-900">{experiment.name}</h4>
            <span
              className={clsx(
                'px-2 py-0.5 text-xs font-medium rounded-full',
                STATUS_COLORS[experiment.status]
              )}
            >
              {experiment.status}
            </span>
          </div>
          <p className="text-sm text-gray-500 mt-1">{experiment.description}</p>
          <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
            <span>Target: {experiment.target_metric}</span>
            {experiment.start_date && (
              <span>
                Started: {new Date(experiment.start_date).toLocaleDateString()}
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {experiment.status === 'draft' && (
            <button
              onClick={handleStart}
              disabled={startMutation.isPending}
              className="px-3 py-1 text-sm bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
            >
              Start
            </button>
          )}
          {experiment.status === 'running' && (
            <button
              onClick={handleStop}
              disabled={stopMutation.isPending}
              className="px-3 py-1 text-sm bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
            >
              Stop
            </button>
          )}
        </div>
      </div>

      {/* Variants */}
      <div className="flex gap-2 mt-3">
        {experiment.variants.map((variant) => (
          <div
            key={variant.name}
            className="flex-1 px-3 py-2 bg-gray-50 rounded text-sm"
          >
            <div className="font-medium text-gray-700">{variant.name}</div>
            <div className="text-gray-500">
              {variant.model_name || 'default'} ({variant.percentage}%)
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

interface ExperimentDetailsProps {
  experimentName: string;
}

function ExperimentDetails({ experimentName }: ExperimentDetailsProps) {
  const { data: analysis, isLoading } = useStatisticalAnalysis(experimentName);

  if (isLoading) {
    return (
      <Card>
        <CardHeader title="Statistical Analysis" />
        <div className="h-48 flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
        </div>
      </Card>
    );
  }

  if (!analysis) {
    return null;
  }

  // Mock timeline data
  const timelineData = Array.from({ length: 7 }, (_, i) => ({
    day: `Day ${i + 1}`,
    control: 0.098 + Math.random() * 0.01,
    treatment: 0.102 + Math.random() * 0.01,
  }));

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader
          title="Statistical Analysis"
          subtitle={experimentName}
        />
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-gray-900">
              {analysis.control_mean.toFixed(4)}
            </div>
            <div className="text-sm text-gray-500">Control Mean</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-gray-900">
              {analysis.treatment_mean.toFixed(4)}
            </div>
            <div className="text-sm text-gray-500">Treatment Mean</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div
              className={clsx(
                'text-2xl font-bold',
                analysis.relative_lift_percent >= 0
                  ? 'text-green-600'
                  : 'text-red-600'
              )}
            >
              {analysis.relative_lift_percent >= 0 ? '+' : ''}
              {analysis.relative_lift_percent.toFixed(2)}%
            </div>
            <div className="text-sm text-gray-500">Relative Lift</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div
              className={clsx(
                'text-2xl font-bold',
                analysis.is_significant ? 'text-green-600' : 'text-gray-600'
              )}
            >
              {analysis.p_value.toFixed(4)}
            </div>
            <div className="text-sm text-gray-500">p-value</div>
          </div>
        </div>

        {/* Significance indicator */}
        <div
          className={clsx(
            'p-4 rounded-lg text-center',
            analysis.is_significant
              ? 'bg-green-50 text-green-800 border border-green-200'
              : 'bg-yellow-50 text-yellow-800 border border-yellow-200'
          )}
        >
          <div className="font-medium">
            {analysis.is_significant
              ? 'Statistically Significant'
              : 'Not Yet Significant'}
          </div>
          <div className="text-sm mt-1">
            95% CI: [{analysis.confidence_interval[0].toFixed(4)},{' '}
            {analysis.confidence_interval[1].toFixed(4)}]
          </div>
        </div>

        {/* Sample sizes */}
        <div className="mt-4 flex justify-center gap-8 text-sm text-gray-500">
          <span>Control: n={analysis.control_n.toLocaleString()}</span>
          <span>Treatment: n={analysis.treatment_n.toLocaleString()}</span>
        </div>
      </Card>

      {/* Timeline chart */}
      <Card>
        <CardHeader title="Metric Over Time" />
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={timelineData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="day" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} domain={['auto', 'auto']} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'white',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                }}
                formatter={(value: number) => value.toFixed(4)}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="control"
                stroke="#6b7280"
                name="Control"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="treatment"
                stroke="#3b82f6"
                name="Treatment"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  );
}
