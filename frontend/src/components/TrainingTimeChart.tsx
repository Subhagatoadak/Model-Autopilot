import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { ExperimentRun } from '../types/experiment';

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

interface Props {
  runs: ExperimentRun[];
}

export default function TrainingTimeChart({ runs }: Props) {
  if (runs.length === 0) {
    return <p className="text-gray-500 text-sm italic">No runs yet</p>;
  }

  const data = runs.map((run) => ({
    iteration: `#${run.iteration}`,
    time: Number((run.train_time_s || 0).toFixed(2)),
    model: run.model_name,
  }));

  return (
    <ResponsiveContainer width="100%" height={180}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis dataKey="iteration" stroke="#9ca3af" tick={{ fontSize: 11 }} />
        <YAxis stroke="#9ca3af" label={{ value: 'seconds', angle: -90, position: 'insideLeft', fill: '#9ca3af', fontSize: 11 }} />
        <Tooltip
          contentStyle={{
            backgroundColor: '#1f2937',
            border: '1px solid #374151',
            borderRadius: '0.5rem',
            color: '#e5e7eb',
          }}
          formatter={(value: number | undefined) => [`${value ?? 0}s`, 'Training Time']}
          labelFormatter={(_, payload) => {
            const item = payload?.[0]?.payload;
            return item ? `${item.model} (Iter ${item.iteration})` : '';
          }}
        />
        <Bar dataKey="time" radius={[4, 4, 0, 0]}>
          {data.map((_, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} fillOpacity={0.8} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
