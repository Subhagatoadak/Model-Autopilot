import type { AgentStep } from '../types/agent';

const agentColors: Record<string, string> = {
  orchestrator: 'text-purple-400',
  data: 'text-green-400',
  strategy: 'text-yellow-400',
  model: 'text-blue-400',
  training: 'text-orange-400',
  evaluation: 'text-cyan-400',
  recommendation: 'text-pink-400',
  rl_tracker: 'text-amber-400',
};

const agentIcons: Record<string, string> = {
  orchestrator: '\u25B6',
  data: '\u2630',
  strategy: '\u2696',
  model: '\u2699',
  training: '\u26A1',
  evaluation: '\u2714',
  recommendation: '\u2728',
  rl_tracker: '\uD83C\uDFAF',
};

const stepBadgeColors: Record<string, string> = {
  transform_applied: 'bg-green-900 text-green-300',
  transforming: 'bg-green-900 text-green-300',
  transforms_queued: 'bg-purple-900 text-purple-300',
  reward: 'bg-amber-900 text-amber-300',
  rankings: 'bg-amber-900 text-amber-300',
  error: 'bg-red-900 text-red-300',
  complete: 'bg-blue-900 text-blue-300',
  converged: 'bg-emerald-900 text-emerald-300',
};

interface Props {
  steps: AgentStep[];
}

export default function AgentTimeline({ steps }: Props) {
  return (
    <div className="space-y-1 max-h-[500px] overflow-y-auto pr-2">
      {steps.length === 0 && (
        <p className="text-gray-500 text-sm italic">Waiting for agent activity...</p>
      )}
      {steps.map((step, i) => (
        <div
          key={i}
          className="flex items-start gap-3 py-2 px-3 rounded-lg bg-gray-800/50 animate-in fade-in"
        >
          <span className={`text-lg ${agentColors[step.agent] || 'text-gray-400'}`}>
            {agentIcons[step.agent] || '\u25CB'}
          </span>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className={`text-xs font-semibold uppercase ${agentColors[step.agent] || 'text-gray-400'}`}>
                {step.agent}
              </span>
              <span className={`text-xs px-1.5 py-0.5 rounded ${stepBadgeColors[step.step] || 'text-gray-500'}`}>
                {step.step}
              </span>
            </div>
            <p className="text-sm text-gray-300 mt-0.5 break-words">{step.content}</p>
          </div>
          <span className="text-xs text-gray-600 whitespace-nowrap">
            {new Date(step.timestamp).toLocaleTimeString()}
          </span>
        </div>
      ))}
    </div>
  );
}
