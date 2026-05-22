import { useEffect, useMemo, useState } from 'react';
import {
  ChevronDown, Plus, X as XIcon, Download, SlidersHorizontal, Loader2,
  Sparkles, Search,
} from 'lucide-react';

import { api } from '../lib/api.ts';
import { RecommendationPill, DarkInput } from '../components/ui.tsx';
import { GlowCard, CardHeader } from '../components/GlowCard.tsx';
import type {
  ApiError,
  PredictResponse,
  RecommendationLabel,
  Variant,
  Verdict,
} from '../types.ts';


interface BatchViewProps {
  variant: Variant;
  onVariantChange: (v: Variant) => void;
  onGoToSingle: (pdb: string) => void;
}


interface BatchRow {
  pdb_id: string;
  prediction?: PredictResponse;
  verdict?: Verdict;
  predictError?: string;
  verdictError?: string;
  predicting: boolean;
  evaluating: boolean;
}


const AGENT_USD_PER_CALL = 0.30;
const PARALLEL_AGENT_CALLS = 5;


export function BatchView({ variant, onGoToSingle }: BatchViewProps) {
  const [whitelist, setWhitelist] = useState<string[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [includeAgent, setIncludeAgent] = useState(true);
  const [rows, setRows] = useState<Record<string, BatchRow>>({});
  const [running, setRunning] = useState(false);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [pickerFilter, setPickerFilter] = useState('');
  const [error, setError] = useState<ApiError | null>(null);
  const [filterTrust, setFilterTrust] = useState(true);
  const [filterReview, setFilterReview] = useState(true);
  const [filterDiscard, setFilterDiscard] = useState(false);
  const [sortBy, setSortBy] = useState<'recommendation' | 'pred' | 'delta'>('recommendation');

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const res = await api.pdbIds({ limit: 50 });
      if (cancelled) return;
      if (res.ok) {
        setWhitelist(res.data);
        if (selected.length === 0 && res.data.length > 0) {
          setSelected(res.data.slice(0, 12));
        }
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!pickerOpen) return;
    const controller = new AbortController();
    const t = setTimeout(async () => {
      const res = await api.pdbIds({ q: pickerFilter, limit: 50, signal: controller.signal });
      if (res.ok) setWhitelist(res.data);
    }, 250);
    return () => {
      clearTimeout(t);
      controller.abort();
    };
  }, [pickerFilter, pickerOpen]);

  const pickerOptions = useMemo(() => {
    const q = pickerFilter.toUpperCase();
    const pool = whitelist.filter(id => !selected.includes(id));
    return (q ? pool.filter(id => id.includes(q)) : pool).slice(0, 50);
  }, [whitelist, selected, pickerFilter]);

  const estCostUsd = includeAgent ? selected.length * AGENT_USD_PER_CALL : 0;
  const estLatencyMin = includeAgent
    ? Math.max(1, Math.ceil((selected.length * 20) / PARALLEL_AGENT_CALLS / 60))
    : Math.ceil(selected.length / 60);

  async function handleRun() {
    if (selected.length === 0 || running) return;
    setRunning(true);
    setError(null);
    setRows(Object.fromEntries(selected.map(id => [id, { pdb_id: id, predicting: true, evaluating: false }])));

    const batchRes = await api.predictBatch(selected, variant);
    if (!batchRes.ok) {
      setError(batchRes.error);
      setRunning(false);
      return;
    }

    const newRows: Record<string, BatchRow> = { ...rows };
    for (const r of batchRes.data.results) {
      newRows[r.pdb_id] = { pdb_id: r.pdb_id, prediction: r, predicting: false, evaluating: includeAgent };
    }
    for (const f of batchRes.data.failed) {
      newRows[f.pdb_id] = { pdb_id: f.pdb_id, predictError: f.error, predicting: false, evaluating: false };
    }
    setRows(newRows);

    if (!includeAgent) {
      setRunning(false);
      return;
    }

    const queue = batchRes.data.results.map(r => r.pdb_id);
    const workers: Promise<void>[] = [];
    for (let w = 0; w < PARALLEL_AGENT_CALLS; w++) {
      workers.push((async () => {
        while (queue.length > 0) {
          const id = queue.shift();
          if (!id) break;
          const res = await api.evaluate(id, variant);
          setRows(prev => {
            const cur = prev[id] ?? { pdb_id: id, predicting: false, evaluating: false };
            if (!res.ok) {
              return { ...prev, [id]: { ...cur, verdictError: res.error.message, evaluating: false } };
            }
            return { ...prev, [id]: { ...cur, verdict: res.data, evaluating: false } };
          });
        }
      })());
    }
    await Promise.all(workers);
    setRunning(false);
  }

  function exportCsv() {
    const headers = ['pdb', 'pred_pK', 'hidden_pK', 'abs_delta', 'regex_fraction', 'recommendation', 'evidence'];
    const lines = [headers.join(',')];
    for (const id of selected) {
      const r = rows[id];
      const p = r?.prediction;
      const v = r?.verdict;
      const delta = p && p.hidden_pK != null ? Math.abs(p.pK - p.hidden_pK).toFixed(2) : '';
      const regex = p ? `${p.regex_verifier.verified}/${p.regex_verifier.total}` : '';
      const evidence = v ? (v.verified_claims[0]?.evidence ?? '').replace(/,/g, ';') : '';
      lines.push([
        id,
        p?.pK.toFixed(2) ?? '',
        p?.hidden_pK?.toFixed(2) ?? '',
        delta,
        regex,
        v?.recommendation ?? '',
        evidence,
      ].join(','));
    }
    const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `batch_${variant}_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const displayedRows = useMemo(() => {
    const rs = selected.map(id => rows[id]).filter(Boolean);
    const allowed = new Set<RecommendationLabel>([
      ...(filterTrust ? (['trust'] as const) : []),
      ...(filterReview ? (['review'] as const) : []),
      ...(filterDiscard ? (['discard'] as const) : []),
    ]);
    const filtered = rs.filter(r => !r.verdict || allowed.has(r.verdict.recommendation));
    return filtered.sort((a, b) => {
      if (sortBy === 'pred') return (b.prediction?.pK ?? 0) - (a.prediction?.pK ?? 0);
      if (sortBy === 'delta') {
        const da = a.prediction?.hidden_pK != null ? Math.abs(a.prediction.pK - a.prediction.hidden_pK) : 99;
        const db = b.prediction?.hidden_pK != null ? Math.abs(b.prediction.pK - b.prediction.hidden_pK) : 99;
        return da - db;
      }
      const rank: Record<string, number> = { trust: 0, review: 1, discard: 2 };
      const ra = a.verdict ? rank[a.verdict.recommendation] : 3;
      const rb = b.verdict ? rank[b.verdict.recommendation] : 3;
      return ra - rb;
    });
  }, [rows, selected, filterTrust, filterReview, filterDiscard, sortBy]);

  const stats = useMemo(() => {
    const completed = displayedRows.filter(r => r.prediction);
    const trust    = completed.filter(r => r.verdict?.recommendation === 'trust').length;
    const review   = completed.filter(r => r.verdict?.recommendation === 'review').length;
    const discard  = completed.filter(r => r.verdict?.recommendation === 'discard').length;
    return { trust, review, discard, total: completed.length };
  }, [displayedRows]);

  return (
    <div className="flex flex-col gap-6">
      {/* ---------- Header card ---------- */}
      <GlowCard className="fx-fade-up">
        <div className="px-6 py-5 border-b" style={{ borderColor: 'var(--color-line)' }}>
          <span className="text-[11px] font-mono tracking-[0.22em] uppercase"
                style={{ color: 'var(--color-ink-dim)' }}>
            Batch triage
          </span>
          <h2 className="font-display text-2xl font-semibold tracking-[-0.02em] mt-1 mb-2"
              style={{ color: 'var(--color-ink)' }}>
            Rank a set of test-split PDBs by their defensible predicted pK.
          </h2>
          <p className="text-sm" style={{ color: 'var(--color-ink-mute)' }}>
            Each row gets a regex-grounded rationale and (optionally) an independent agent verdict.
          </p>
        </div>

        <div className="px-6 py-5 flex flex-col gap-5">
          <div>
            <div className="text-[10px] font-mono tracking-[0.2em] uppercase mb-3"
                 style={{ color: 'var(--color-ink-dim)' }}>
              Selected PDB IDs · {selected.length}
            </div>
            <div className="flex flex-wrap gap-2 text-sm font-mono relative">
              {selected.map(id => (
                <span
                  key={id}
                  className="px-2 py-0.5 rounded-full flex items-center gap-1 transition-colors"
                  style={{
                    background: '#f3f5fb',
                    border: '1px solid var(--color-line)',
                    color: 'var(--color-ink-2)',
                  }}
                >
                  {id}
                  <button
                    onClick={() => setSelected(selected.filter(x => x !== id))}
                    className="ml-1 transition-colors hover:text-[#f4625f]"
                    style={{ color: 'var(--color-ink-dim)' }}
                  >
                    ×
                  </button>
                </span>
              ))}
              <button
                onClick={() => setPickerOpen(o => !o)}
                className="flex items-center gap-1 px-3 py-0.5 rounded-full text-[12px] transition"
                style={{
                  background: 'rgba(84, 103, 242, 0.12)',
                  color: '#5467F2',
                  border: '1px solid rgba(84, 103, 242, 0.35)',
                }}
              >
                <Plus size={12} /> Add
              </button>
              <button
                onClick={() => setSelected([])}
                className="flex items-center gap-1 px-2 py-0.5 rounded-full text-[12px] transition-colors hover:text-ink"
                style={{ color: 'var(--color-ink-dim)' }}
              >
                <XIcon size={12} /> Clear
              </button>

              {pickerOpen && (
                <div className="absolute top-full left-0 mt-2 z-30 w-80 max-h-72 overflow-hidden flex flex-col glass-strong rounded-xl bevel-border fx-fade-soft">
                  <div className="relative">
                    <Search size={12}
                            className="absolute left-3 top-1/2 -translate-y-1/2"
                            style={{ color: 'var(--color-ink-dim)' }} />
                    <DarkInput
                      autoFocus
                      value={pickerFilter}
                      onChange={e => setPickerFilter(e.target.value)}
                      placeholder={`search ${whitelist.length} PDBs`}
                      className="!rounded-none !border-0 !border-b !pl-9"
                    />
                  </div>
                  <div className="overflow-y-auto">
                    {pickerOptions.length === 0 && (
                      <div className="px-3 py-2 text-xs italic font-mono"
                           style={{ color: 'var(--color-ink-dim)' }}>
                        no matches
                      </div>
                    )}
                    {pickerOptions.map(id => (
                      <button
                        key={id}
                        onClick={() => { setSelected([...selected, id]); setPickerFilter(''); }}
                        className="block w-full text-left px-3 py-1.5 text-sm font-mono transition-colors hover:bg-[rgba(84,103,242,0.1)]"
                        style={{ color: 'var(--color-ink-2)' }}
                      >
                        {id}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="flex items-center justify-between flex-wrap gap-3">
            <div className="flex flex-wrap gap-3 items-center">
              <button
                onClick={() => {
                  const shuffled = [...whitelist].sort(() => Math.random() - 0.5);
                  setSelected(shuffled.slice(0, 20));
                }}
                className="btn-ghost rounded-lg px-3 py-1.5 text-[13px] flex items-center gap-2"
              >
                <Plus size={13} /> Pick 20 random
              </button>
              <label className="flex items-center gap-2 text-[13px] cursor-pointer pl-2"
                     style={{ color: 'var(--color-ink-2)' }}>
                <input
                  type="checkbox"
                  checked={includeAgent}
                  onChange={e => setIncludeAgent(e.target.checked)}
                  className="rounded"
                />
                Include agent evaluation
                <span className="text-[11px] font-mono"
                      style={{ color: 'var(--color-ink-dim)' }}>
                  ~${AGENT_USD_PER_CALL.toFixed(2)} ea
                </span>
              </label>
            </div>
            <div className="flex items-center gap-4 text-sm">
              <span className="font-mono text-[12px]"
                    style={{ color: 'var(--color-ink-dim)' }}>
                Est cost: ${estCostUsd.toFixed(2)} · ~{estLatencyMin} min
              </span>
              <button
                onClick={handleRun}
                disabled={selected.length === 0 || running}
                className="btn-primary rounded-lg px-4 py-2 text-sm flex items-center gap-2"
              >
                {running ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
                Run batch
              </button>
            </div>
          </div>

          {error && (
            <div className="rounded-lg px-3 py-2 text-sm"
                 style={{
                   background: 'rgba(244, 98, 95, 0.12)',
                   color: '#a8332f',
                   border: '1px solid rgba(244, 98, 95, 0.4)',
                 }}>
              {error.message} (status {error.status})
            </div>
          )}
        </div>
      </GlowCard>

      {/* ---------- Verdict count strip ---------- */}
      {stats.total > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 fx-fade-up">
          <StatChip label="Predicted" count={stats.total} tone="cyan" />
          <StatChip label="Trust"   count={stats.trust}   tone="lime" />
          <StatChip label="Review"  count={stats.review}  tone="amber" />
          <StatChip label="Discard" count={stats.discard} tone="rose" />
        </div>
      )}

      {/* ---------- Results card ---------- */}
      <GlowCard className="fx-fade-up" style={{ animationDelay: '80ms' } as React.CSSProperties}>
        <CardHeader
          right={
            <span className="text-[11px] font-mono"
                  style={{ color: 'var(--color-ink-dim)' }}>
              {displayedRows.length}/{selected.length} shown
            </span>
          }
        >
          Results
        </CardHeader>

        <div className="px-5 py-4 flex items-center gap-4 text-sm border-b"
             style={{ borderColor: 'var(--color-line)' }}>
          <button
            onClick={() => {
              const next = sortBy === 'recommendation' ? 'pred' : sortBy === 'pred' ? 'delta' : 'recommendation';
              setSortBy(next);
            }}
            className="btn-ghost rounded-lg px-2.5 py-1 text-[12px] flex items-center gap-1.5"
          >
            <SlidersHorizontal size={12} />
            Sort: {sortBy} <ChevronDown size={12} />
          </button>
          <div className="flex gap-3 pl-3 border-l" style={{ borderColor: 'var(--color-line)' }}>
            {[
              { k: 'trust',   v: filterTrust,   set: setFilterTrust,   color: '#4ec07a' },
              { k: 'review',  v: filterReview,  set: setFilterReview,  color: '#FF9900' },
              { k: 'discard', v: filterDiscard, set: setFilterDiscard, color: '#f4625f' },
            ].map(it => (
              <label key={it.k}
                     className="flex items-center gap-1.5 cursor-pointer text-[12px] font-mono uppercase tracking-wider"
                     style={{ color: it.v ? it.color : 'var(--color-ink-dim)' }}>
                <input
                  type="checkbox"
                  checked={it.v}
                  onChange={e => it.set(e.target.checked)}
                  className="rounded"
                  style={{ accentColor: it.color }}
                />
                {it.k}
              </label>
            ))}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left border-collapse">
            <thead>
              <tr className="text-[10px] font-mono tracking-[0.18em] uppercase"
                  style={{ color: 'var(--color-ink-dim)' }}>
                <th className="font-medium py-3 px-4 border-b" style={{ borderColor: 'var(--color-line)' }}>PDB</th>
                <th className="font-medium py-3 px-4 text-right border-b" style={{ borderColor: 'var(--color-line)' }}>pred</th>
                <th className="font-medium py-3 px-4 text-right border-b" style={{ borderColor: 'var(--color-line)' }}>|Δ|</th>
                <th className="font-medium py-3 px-4 border-b" style={{ borderColor: 'var(--color-line)' }}>regex</th>
                <th className="font-medium py-3 px-4 border-b" style={{ borderColor: 'var(--color-line)' }}>agent</th>
                <th className="font-medium py-3 px-4 border-b" style={{ borderColor: 'var(--color-line)' }}>supporting evidence</th>
              </tr>
            </thead>
            <tbody className="font-mono text-[13px]">
              {displayedRows.length === 0 && (
                <tr>
                  <td colSpan={6} className="py-10 px-4 text-center italic"
                      style={{ color: 'var(--color-ink-dim)' }}>
                    No rows yet — pick PDBs and Run batch.
                  </td>
                </tr>
              )}
              {displayedRows.map((row, i) => {
                const p = row.prediction;
                const v = row.verdict;
                const delta = p && p.hidden_pK != null ? Math.abs(p.pK - p.hidden_pK) : null;
                const regex = p ? `${p.regex_verifier.verified}/${p.regex_verifier.total}` : '';
                const evidence = row.verdictError ?? row.predictError
                  ?? v?.verified_claims[0]?.evidence
                  ?? (row.evaluating ? 'evaluating…' : row.predicting ? 'predicting…' : '');
                return (
                  <tr
                    key={row.pdb_id}
                    onClick={() => onGoToSingle(row.pdb_id)}
                    className="cursor-pointer fx-rise hover:bg-[rgba(84,103,242,0.06)] transition-colors group"
                    style={{
                      animationDelay: `${i * 30}ms`,
                      borderBottom: '1px solid var(--color-line)',
                    }}
                  >
                    <td className="py-3 px-4 font-semibold tracking-tight transition-colors"
                        style={{ color: 'var(--color-ink)' }}>
                      <span className="group-hover:text-gradient-cool">{row.pdb_id}</span>
                    </td>
                    <td className="py-3 px-4 text-right" style={{ color: 'var(--color-ink-2)' }}>
                      {p ? p.pK.toFixed(2) : '—'}
                    </td>
                    <td className="py-3 px-4 text-right" style={{ color: 'var(--color-ink-mute)' }}>
                      {delta != null ? delta.toFixed(2) : '—'}
                    </td>
                    <td className="py-3 px-4 tracking-tight" style={{ color: 'var(--color-ink-mute)' }}>
                      {regex || '—'}
                    </td>
                    <td className="py-3 px-4">
                      {v
                        ? <RecommendationPill type={v.recommendation} />
                        : row.evaluating
                          ? <span className="text-xs italic" style={{ color: 'var(--color-ink-dim)' }}>evaluating…</span>
                          : <span style={{ color: 'var(--color-ink-faint)' }}>—</span>}
                    </td>
                    <td className="py-3 px-4 text-sm truncate max-w-[280px]"
                        style={{ color: 'var(--color-ink-2)' }}>
                      {evidence}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="px-5 py-4 flex flex-wrap gap-3 border-t"
             style={{ borderColor: 'var(--color-line)' }}>
          <button
            onClick={exportCsv}
            disabled={displayedRows.length === 0}
            className="btn-ghost rounded-lg px-4 py-2 text-sm flex items-center gap-2 disabled:opacity-50"
          >
            <Download size={14} /> Export selected as CSV
          </button>
          <button
            onClick={() => {
              const trusted = displayedRows.filter(r => r.verdict?.recommendation === 'trust').map(r => r.pdb_id);
              alert(`(mock action) would queue ${trusted.length} systems for assay: ${trusted.join(', ')}`);
            }}
            className="rounded-lg px-4 py-2 text-sm flex items-center gap-2 transition-colors"
            style={{
              background: 'rgba(78, 192, 122, 0.12)',
              color: '#4ec07a',
              border: '1px solid rgba(78, 192, 122, 0.35)',
            }}
          >
            Send only “trust” to assay queue (mock)
          </button>
        </div>
      </GlowCard>
    </div>
  );
}

function StatChip({ label, count, tone }: { label: string; count: number; tone: 'cyan' | 'lime' | 'amber' | 'rose' }) {
  const color =
    tone === 'cyan'  ? '#5467F2' :
    tone === 'lime'  ? '#4ec07a' :
    tone === 'amber' ? '#FF9900'  :
                       '#f4625f';
  return (
    <div className="glass rounded-xl bevel-border px-4 py-3 flex items-center justify-between">
      <span className="text-[10px] font-mono tracking-[0.2em] uppercase"
            style={{ color: 'var(--color-ink-dim)' }}>
        {label}
      </span>
      <span className="font-display text-xl font-semibold tracking-tight"
            style={{ color }}>
        {count}
      </span>
    </div>
  );
}
