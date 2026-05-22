import { useEffect, useState } from 'react';
import { ChevronDown, Loader2, TriangleAlert } from 'lucide-react';

import { api } from '../lib/api.ts';
import { RecommendationPill } from '../components/ui.tsx';
import { GlowCard, CardHeader } from '../components/GlowCard.tsx';
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
    <div className="flex flex-col gap-6">
      <GlowCard className="fx-fade-up">
        <div className="px-6 py-5 border-b flex flex-wrap items-start justify-between gap-3"
             style={{ borderColor: 'var(--color-line)' }}>
          <div>
            <span className="text-[11px] font-mono tracking-[0.22em] uppercase"
                  style={{ color: 'var(--color-ink-dim)' }}>
              Failure modes
            </span>
            <h2 className="font-display text-2xl font-semibold tracking-[-0.02em] mt-1 mb-1"
                style={{ color: 'var(--color-ink)' }}>
              Where the model fails
            </h2>
            <p className="text-sm max-w-2xl"
               style={{ color: 'var(--color-ink-mute)' }}>
              Predictions the trained model made confidently, where the independent agent
              found contradicting evidence. Top {data?.rows.length ?? 10} systems by |model − mm-gbsa|.
            </p>
          </div>
          <div className="glass rounded-full px-3 py-1.5 text-[12px] font-mono tracking-wider"
               style={{ color: 'var(--color-ink-2)' }}>
            Variant · <span style={{ color: '#5467F2' }}>{variant}</span>
          </div>
        </div>

        <div className="p-6">
          {loading && (
            <div className="flex items-center gap-2 text-sm py-6"
                 style={{ color: 'var(--color-ink-mute)' }}>
              <Loader2 size={16} className="animate-spin" /> loading precomputed failure modes…
            </div>
          )}

          {error && (
            <div className="rounded-xl px-4 py-3 text-sm flex items-start gap-3"
                 style={{
                   background: 'rgba(255, 153, 0, 0.1)',
                   border: '1px solid rgba(255, 153, 0, 0.4)',
                   color: '#a55a00',
                 }}>
              <TriangleAlert size={16} className="shrink-0 mt-0.5" />
              <div>
                No precomputed failure modes for variant <span className="font-mono">{variant}</span> yet
                {error.status === 404 ? '' : ` — ${error.message}`}.
                <div className="text-xs mt-1" style={{ color: '#b86d10' }}>
                  Run <code className="px-1 rounded font-mono" style={{ background: 'rgba(255, 153, 0, 0.18)' }}>make precompute</code>
                  {' '}inside the inference container to populate
                  {' '}<code className="px-1 rounded font-mono" style={{ background: 'rgba(255, 153, 0, 0.18)' }}>data/failure_modes_{variant}.json</code>.
                </div>
              </div>
            </div>
          )}

          {data && (
            <>
              <div className="flex items-center mb-4 text-sm">
                <button className="btn-ghost rounded-lg px-2.5 py-1 text-[12px] flex items-center gap-1.5">
                  Sort: |model − mmgbsa| <ChevronDown size={12} />
                </button>
                <span className="ml-auto text-[11px] font-mono"
                      style={{ color: 'var(--color-ink-dim)' }}>
                  generated_at: {new Date(data.generated_at).toLocaleString()}
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-sm text-left border-collapse">
                  <thead>
                    <tr className="text-[10px] font-mono tracking-[0.18em] uppercase"
                        style={{ color: 'var(--color-ink-dim)' }}>
                      <th className="font-medium py-3 px-4 border-b w-24" style={{ borderColor: 'var(--color-line)' }}>PDB</th>
                      <th className="font-medium py-3 px-4 text-right border-b w-20" style={{ borderColor: 'var(--color-line)' }}>model</th>
                      <th className="font-medium py-3 px-4 text-right border-b w-20" style={{ borderColor: 'var(--color-line)' }}>vina</th>
                      <th className="font-medium py-3 px-4 text-right border-b w-24" style={{ borderColor: 'var(--color-line)' }}>mm-gbsa</th>
                      <th className="font-medium py-3 px-4 border-b w-32" style={{ borderColor: 'var(--color-line)' }}>agent</th>
                      <th className="font-medium py-3 px-4 border-b" style={{ borderColor: 'var(--color-line)' }}>why the model is wrong</th>
                    </tr>
                  </thead>
                  <tbody className="font-mono text-[13px]">
                    {data.rows.map((row, i) => (
                      <tr
                        key={row.pdb}
                        onClick={() => onGoToSingle(row.pdb)}
                        className="cursor-pointer fx-rise hover:bg-[rgba(84,103,242,0.06)] transition-colors group"
                        style={{
                          animationDelay: `${i * 30}ms`,
                          borderBottom: '1px solid var(--color-line)',
                        }}
                      >
                        <td className="py-3 px-4 font-semibold transition-colors"
                            style={{ color: 'var(--color-ink)' }}>
                          <span className="group-hover:text-gradient-cool">{row.pdb}</span>
                        </td>
                        <td className="py-3 px-4 text-right" style={{ color: 'var(--color-ink-2)' }}>{row.model.toFixed(1)}</td>
                        <td className="py-3 px-4 text-right" style={{ color: 'var(--color-ink-mute)' }}>{row.vina.toFixed(1)}</td>
                        <td className="py-3 px-4 text-right" style={{ color: 'var(--color-ink-mute)' }}>{row.mmgbsa.toFixed(1)}</td>
                        <td className="py-3 px-4 align-top pt-3">
                          <RecommendationPill type={row.agent} />
                        </td>
                        <td className="py-3 px-4 text-sm leading-relaxed pr-8 font-sans"
                            style={{ color: 'var(--color-ink-2)' }}>
                          {row.reason}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <p className="mt-4 text-sm italic"
                 style={{ color: 'var(--color-ink-dim)' }}>
                Click any row → opens the full prediction + agent trace in the Inspect tab.
              </p>
            </>
          )}
        </div>
      </GlowCard>

      {data && data.patterns.length > 0 && (
        <GlowCard className="fx-fade-up" style={{ animationDelay: '80ms' } as React.CSSProperties}>
          <CardHeader>
            Aggregate failure pattern analysis
          </CardHeader>
          <div className="p-6">
            <table className="w-full text-sm text-left border-collapse max-w-4xl">
              <thead>
                <tr className="text-[10px] font-mono tracking-[0.18em] uppercase"
                    style={{ color: 'var(--color-ink-dim)' }}>
                  <th className="font-medium py-2 px-2 w-[40%] border-b"
                      style={{ borderColor: 'var(--color-line)' }}>Failure cluster</th>
                  <th className="font-medium py-2 px-2 text-center w-[15%] border-b"
                      style={{ borderColor: 'var(--color-line)' }}>Count</th>
                  <th className="font-medium py-2 px-2 border-b"
                      style={{ borderColor: 'var(--color-line)' }}>Affected systems</th>
                </tr>
              </thead>
              <tbody className="text-[14px]">
                {data.patterns.map((p, i) => (
                  <tr key={i}
                      className="fx-rise"
                      style={{
                        animationDelay: `${i * 60}ms`,
                        borderBottom: '1px solid var(--color-line)',
                      }}>
                    <td className="py-3 px-2 font-medium" style={{ color: 'var(--color-ink-2)' }}>{p.cluster}</td>
                    <td className="py-3 px-2 text-center font-mono" style={{ color: 'var(--color-ink-mute)' }}>{p.count}</td>
                    <td className="py-3 px-2 font-mono text-xs tracking-tight"
                        style={{ color: 'var(--color-ink-mute)' }}>{p.systems}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <div className="mt-6 p-4 rounded-xl text-sm font-medium flex gap-3 items-start"
                 style={{
                   background: 'rgba(84, 103, 242, 0.1)',
                   border: '1px solid rgba(84, 103, 242, 0.35)',
                   color: '#3a4cd6',
                 }}>
              <span style={{ color: '#5467F2' }}>→</span>
              suggests v2 should add a “pose stability” auxiliary supervision signal.
            </div>
          </div>
        </GlowCard>
      )}
    </div>
  );
}
