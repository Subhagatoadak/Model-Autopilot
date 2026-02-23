import type { ExperimentRun } from '../types/experiment';

interface Props {
  runs: ExperimentRun[];
  primaryMetric?: string;
}

export default function ModelLeaderboard({ runs, primaryMetric }: Props) {
  if (runs.length === 0) {
    return <p className="text-gray-500 text-sm italic">No runs yet</p>;
  }

  // Sort by primary metric descending
  const metric = primaryMetric || Object.keys(runs[0]?.metrics || {})[0] || 'accuracy';
  const sorted = [...runs].sort(
    (a, b) => (b.metrics[metric] || 0) - (a.metrics[metric] || 0),
  );

  const metricNames = Array.from(
    new Set(runs.flatMap((r) => Object.keys(r.metrics))),
  );

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-700 text-gray-400">
            <th className="text-left py-2 px-3">#</th>
            <th className="text-left py-2 px-3">Iter</th>
            <th className="text-left py-2 px-3">Model</th>
            {metricNames.map((m) => (
              <th key={m} className={`text-right py-2 px-3 ${m === metric ? 'text-blue-400' : ''}`}>
                {m}
              </th>
            ))}
            <th className="text-right py-2 px-3">Time</th>
            <th className="text-right py-2 px-3">UCB</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((run, i) => (
            <tr
              key={run.id || i}
              className={`border-b border-gray-800 hover:bg-gray-800/50 ${
                i === 0 ? 'bg-blue-600/10' : ''
              }`}
            >
              <td className="py-2 px-3 text-gray-500">{i + 1}</td>
              <td className="py-2 px-3">{run.iteration}</td>
              <td className="py-2 px-3 font-medium">{run.model_name}</td>
              {metricNames.map((m) => (
                <td
                  key={m}
                  className={`text-right py-2 px-3 ${m === metric ? 'text-blue-400 font-semibold' : 'text-gray-300'}`}
                >
                  {run.metrics[m]?.toFixed(4) || '-'}
                </td>
              ))}
              <td className="text-right py-2 px-3 text-gray-400">
                {run.train_time_s?.toFixed(1)}s
              </td>
              <td className="text-right py-2 px-3 text-gray-400">
                {run.ucb_score?.toFixed(3) || '-'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
