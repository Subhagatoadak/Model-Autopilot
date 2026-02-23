import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { ExperimentRun } from '../types/experiment';

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];

interface Props {
  runs: ExperimentRun[];
  primaryMetric?: string;
}

export default function ModelComparison({ runs, primaryMetric }: Props) {
  if (runs.length === 0) {
    return <p className="text-gray-500 text-sm italic">No runs yet</p>;
  }

  const metric = primaryMetric || Object.keys(runs[0]?.metrics || {})[0] || 'accuracy';

  // Group by model, take best score per model
  const modelBest: Record<string, { score: number; time: number; count: number }> = {};
  for (const run of runs) {
    const val = run.metrics[metric] || 0;
    if (!modelBest[run.model_name] || val > modelBest[run.model_name].score) {
      modelBest[run.model_name] = {
        score: val,
        time: run.train_time_s || 0,
        count: (modelBest[run.model_name]?.count || 0) + 1,
      };
    } else {
      modelBest[run.model_name].count++;
    }
  }

  const data = Object.entries(modelBest)
    .sort(([, a], [, b]) => b.score - a.score)
    .map(([name, info]) => ({
      name: name.length > 15 ? name.slice(0, 12) + '...' : name,
      fullName: name,
      [metric]: Number(info.score.toFixed(4)),
      time: Number(info.time.toFixed(1)),
      runs: info.count,
    }));

  return (
    <div>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} layout="vertical" margin={{ left: 10, right: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis type="number" stroke="#9ca3af" domain={['auto', 'auto']} />
          <YAxis dataKey="name" type="category" stroke="#9ca3af" width={100} tick={{ fontSize: 11 }} />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1f2937',
              border: '1px solid #374151',
              borderRadius: '0.5rem',
              color: '#e5e7eb',
            }}
            formatter={(value: number | undefined) => [(value ?? 0).toFixed(4), metric]}
            labelFormatter={(label) => {
              const item = data.find((d) => d.name === label);
              return `${item?.fullName || label} (${item?.runs || 1} run${(item?.runs || 1) > 1 ? 's' : ''})`;
            }}
          />
          <Bar dataKey={metric} radius={[0, 4, 4, 0]}>
            {data.map((_, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
