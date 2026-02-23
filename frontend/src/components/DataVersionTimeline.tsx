import type { DatasetVersion } from '../types/dataset';

interface Props {
  versions: DatasetVersion[];
}

export default function DataVersionTimeline({ versions }: Props) {
  return (
    <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2">
      {versions.length === 0 && (
        <p className="text-gray-500 text-sm italic">No dataset versions tracked yet...</p>
      )}
      {versions.map((version, i) => {
        const isInitial = version.version === 0;
        const featureChange = i > 0 ? version.num_features - versions[i - 1].num_features : 0;

        return (
          <div
            key={version.id}
            className="relative flex items-start gap-4 py-3 px-4 rounded-lg bg-gray-800/50 border-l-4 border-blue-500/50"
          >
            {/* Timeline connector */}
            {i < versions.length - 1 && (
              <div className="absolute left-6 top-12 w-0.5 h-full bg-gray-700" />
            )}

            {/* Version badge */}
            <div className="flex flex-col items-center gap-1 z-10">
              <div className="w-12 h-12 rounded-full bg-blue-600 flex items-center justify-center text-white font-bold shadow-lg">
                v{version.version}
              </div>
              <span className="text-xs text-gray-500">
                {new Date(version.created_at || '').toLocaleTimeString()}
              </span>
            </div>

            {/* Content */}
            <div className="flex-1 min-w-0">
              {isInitial ? (
                <>
                  <h4 className="text-sm font-semibold text-gray-200">Initial Dataset</h4>
                  <p className="text-xs text-gray-400 mt-1">
                    {version.num_features} features loaded
                  </p>
                </>
              ) : (
                <>
                  <h4 className="text-sm font-semibold text-gray-200">
                    Transform Applied
                    {featureChange !== 0 && (
                      <span className={`ml-2 text-xs ${featureChange > 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {featureChange > 0 ? '+' : ''}{featureChange} features
                      </span>
                    )}
                  </h4>

                  {/* Transform details */}
                  <div className="mt-2 space-y-1">
                    {version.transforms.map((transform, idx) => {
                      let action = 'transform';
                      let paramsStr = '';

                      if (typeof transform === 'string') {
                        action = transform;
                      } else if (typeof transform === 'object' && transform !== null) {
                        const transformObj = transform as Record<string, unknown>;
                        action = (transformObj.action as string) || 'transform';
                        if (transformObj.params) {
                          paramsStr = JSON.stringify(transformObj.params);
                        }
                      }

                      return (
                        <div
                          key={idx}
                          className="text-xs bg-gray-900/50 rounded px-2 py-1 text-gray-300"
                        >
                          <span className="font-mono text-blue-400">
                            {action}
                          </span>
                          {paramsStr && (
                            <span className="text-gray-500 ml-2">
                              {paramsStr}
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>

                  <p className="text-xs text-gray-500 mt-2">
                    Total features: {version.num_features}
                  </p>
                </>
              )}

              {/* Feature names preview (first 5) */}
              {version.feature_names.length > 0 && (
                <details className="mt-2">
                  <summary className="text-xs text-gray-400 cursor-pointer hover:text-gray-300">
                    View features ({version.feature_names.length})
                  </summary>
                  <div className="mt-1 flex flex-wrap gap-1">
                    {version.feature_names.slice(0, 10).map((name, idx) => (
                      <span
                        key={idx}
                        className="text-xs bg-gray-900/70 px-1.5 py-0.5 rounded text-gray-400 font-mono"
                      >
                        {name}
                      </span>
                    ))}
                    {version.feature_names.length > 10 && (
                      <span className="text-xs text-gray-500 px-1.5">
                        +{version.feature_names.length - 10} more
                      </span>
                    )}
                  </div>
                </details>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
