import { useEffect, useState } from 'react';
import { ChevronDown, Loader2 } from 'lucide-react';

import { api } from '../lib/api.ts';
import { RecommendationPill } from '../components/ui.tsx';
import type { ApiError, FailureModesResponse, Variant } from '../types.ts';


interface FailureModesViewProps {
  variant: Variant;
  onVariantChange: (v: Variant) => void;
  onGoToSingle: (pdb: string) => void;
}


export function FailureModesView({ variant, onGoToSingle }: FailureModesViewProps) {
  const [data, setData] = useState<FailureModesResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    (async () => {
      const res = await api.failureModes(variant);
      if (cancelled) return;
      setLoading(false);
      if (!res.ok) {
        setError(res.error);
        return;
      }
      setData(res.data);
    })();
    return () => { cancelled = true; };
  }, [variant]);

  return (
    <div className="flex flex-col gap-8 animate-in fade-in duration-300">
      <div className="flex flex-col border border-slate-200 rounded-lg overflow-hidden bg-white shadow-sm">
        <div className="bg-slate-50 px-6 py-5 border-b border-slate-200 flex justify-between items-start">
          <div>
            <h2 className="text-lg font-semibold tracking-tight text-slate-900 mb-1">Where the model fails</h2>
            <p className="text-slate-500 text-sm max-w-2xl leading-relaxed">
              Predictions the trained model made confidently, where the independent agent
              found contradicting evidence. Top {data?.rows.length ?? 10} systems by |model − mm-gbsa|.
            </p>
          </div>
          <div className="flex border border-slate-200 rounded divide-x divide-slate-200 bg-white font-medium text-sm text-slate-600 shadow-sm shrink-0">
            <span className="px-3 py-1.5">Variant: <span className="font-mono">{variant}</span></span>
          </div>
        </div>

        <div className="p-6">
          {loading && (
            <div className="flex items-center gap-2 text-slate-500 text-sm py-6">
              <Loader2 size={16} className="animate-spin" /> loading precomputed failure modes…
            </div>
          )}

          {error && (
            <div className="border border-amber-200 bg-amber-50 text-amber-800 text-sm px-4 py-3 rounded">
              No precomputed failure modes for variant <span className="font-mono">{variant}</span> yet
              {error.status === 404 ? '' : ` — ${error.message}`}.
              <div className="text-xs text-amber-700 mt-1">
                Run <code className="bg-amber-100 px-1 rounded">make precompute</code> inside the inference container
                to populate <code className="bg-amber-100 px-1 rounded">data/failure_modes_{variant}.json</code>.
              </div>
            </div>
          )}

          {data && (
            <>
              <div className="flex items-center mb-4 text-sm font-medium text-slate-600">
                <button className="flex items-center gap-1.5 hover:text-slate-900 px-2 py-1 rounded border border-slate-200 bg-slate-50 shadow-sm">
                  Sort: |model − mmgbsa| <ChevronDown size={14}/>
                </button>
                <span className="ml-auto text-xs font-mono text-slate-400">
                  generated_at: {new Date(data.generated_at).toLocaleString()}
                </span>
              </div>

              <table className="w-full text-sm text-left border-collapse table-fixed">
                <thead>
                  <tr className="border-y border-slate-200 bg-slate-50/50 text-slate-500">
                    <th className="font-semibold py-3 px-4 font-sans border-r border-slate-200 w-24">PDB</th>
                    <th className="font-semibold py-3 px-4 font-sans text-right w-20">model</th>
                    <th className="font-semibold py-3 px-4 font-sans text-right w-20 border-r border-slate-200">vina</th>
                    <th className="font-semibold py-3 px-4 font-sans text-right w-24 border-r border-slate-200">mm-gbsa</th>
                    <th className="font-semibold py-3 px-4 font-sans border-r border-slate-200 w-32">agent</th>
                    <th className="font-semibold py-3 px-4 font-sans">why the model is wrong</th>
                  </tr>
                </thead>
                <tbody className="font-mono text-[13px]">
                  {data.rows.map(row => (
                    <tr
                      key={row.pdb}
                      onClick={() => onGoToSingle(row.pdb)}
                      className="border-b border-slate-100 hover:bg-slate-50 cursor-pointer"
                    >
                      <td className="py-3 px-4 font-semibold text-slate-700 border-r border-slate-200">{row.pdb}</td>
                      <td className="py-3 px-4 text-right text-slate-600">{row.model.toFixed(1)}</td>
                      <td className="py-3 px-4 text-right text-slate-500 border-r border-slate-200">{row.vina.toFixed(1)}</td>
                      <td className="py-3 px-4 text-right text-slate-500 border-r border-slate-200">{row.mmgbsa.toFixed(1)}</td>
                      <td className="py-3 px-4 border-r border-slate-200 align-top pt-2.5">
                        <RecommendationPill type={row.agent} />
                      </td>
                      <td className="py-3 px-4 text-slate-600 font-sans text-sm leading-relaxed pr-8">
                        {row.reason}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-4 text-sm text-slate-500 italic">
                Click any row → opens the full prediction + agent trace in the Single tab.
              </p>
            </>
          )}
        </div>
      </div>

      {data && data.patterns.length > 0 && (
        <div className="flex flex-col border border-slate-200 rounded-lg overflow-hidden bg-white shadow-sm">
          <div className="bg-slate-50 px-6 py-4 border-b border-slate-200">
            <h2 className="text-sm font-semibold tracking-wide text-slate-600 uppercase">Aggregate failure pattern analysis</h2>
          </div>
          <div className="p-6">
            <table className="w-full text-sm text-left border-collapse max-w-4xl">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500 font-sans">
                  <th className="font-medium py-2 px-2 w-[40%]">Failure cluster</th>
                  <th className="font-medium py-2 px-2 text-center w-[15%]">Count</th>
                  <th className="font-medium py-2 px-2">Affected systems</th>
                </tr>
              </thead>
              <tbody className="font-sans text-[14px]">
                {data.patterns.map((p, i) => (
                  <tr key={i} className="border-b border-slate-100 last:border-0 hover:bg-slate-50/50">
                    <td className="py-3 px-2 text-slate-700 font-medium">{p.cluster}</td>
                    <td className="py-3 px-2 text-center font-mono text-slate-500">{p.count}</td>
                    <td className="py-3 px-2 font-mono text-slate-600 text-xs tracking-tight">{p.systems}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <div className="mt-6 p-4 bg-indigo-50/50 border border-indigo-100 rounded-md text-sm text-indigo-800 font-medium flex gap-3 items-start">
              <span className="text-indigo-500">→</span>
              suggests v2 should add a "pose stability" auxiliary supervision signal.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
