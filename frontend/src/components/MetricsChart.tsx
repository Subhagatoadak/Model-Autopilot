import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { ExperimentRun } from '../types/experiment';

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

interface Props {
  runs: ExperimentRun[];
  primaryMetric?: string;
}

export default function MetricsChart({ runs, primaryMetric }: Props) {
  if (runs.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-gray-500 text-sm">
        No data yet
      </div>
    );
  }

  // Get all metric names from runs
  const metricNames = Array.from(
    new Set(runs.flatMap((r) => Object.keys(r.metrics))),
  );

  const data = runs.map((run) => ({
    iteration: run.iteration,
    model: run.model_name,
    ...run.metrics,
  }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis
          dataKey="iteration"
          stroke="#9ca3af"
          label={{ value: 'Iteration', position: 'insideBottom', offset: -5, fill: '#9ca3af' }}
        />
        <YAxis stroke="#9ca3af" />
        <Tooltip
          contentStyle={{
            backgroundColor: '#1f2937',
            border: '1px solid #374151',
            borderRadius: '0.5rem',
            color: '#e5e7eb',
          }}
          formatter={(value: number | undefined) => value?.toFixed(4) ?? ''}
          labelFormatter={(label) => {
            const run = runs.find((r) => r.iteration === label);
            return `Iteration ${label} (${run?.model_name || ''})`;
          }}
        />
        <Legend />
        {metricNames.map((name, i) => (
          <Line
            key={name}
            type="monotone"
            dataKey={name}
            stroke={COLORS[i % COLORS.length]}
            strokeWidth={name === primaryMetric ? 3 : 1.5}
            dot={{ r: name === primaryMetric ? 5 : 3 }}
            name={name}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
