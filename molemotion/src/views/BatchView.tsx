import { useEffect, useMemo, useState } from 'react';
import { ChevronDown, Plus, X as XIcon, Download, SlidersHorizontal, Loader2 } from 'lucide-react';

import { api } from '../lib/api.ts';
import { RecommendationPill } from '../components/ui.tsx';
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

  // Refetch as user types in the picker (server-side autocomplete, supports tunnel mode).
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

    // Fan-out /evaluate with bounded concurrency.
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
      // recommendation: trust > review > discard > undefined
      const rank: Record<string, number> = { trust: 0, review: 1, discard: 2 };
      const ra = a.verdict ? rank[a.verdict.recommendation] : 3;
      const rb = b.verdict ? rank[b.verdict.recommendation] : 3;
      return ra - rb;
    });
  }, [rows, selected, filterTrust, filterReview, filterDiscard, sortBy]);

  return (
    <div className="flex flex-col gap-8 animate-in fade-in duration-300">
      <div className="flex flex-col border border-slate-200 rounded-lg overflow-hidden bg-white shadow-sm">
        <div className="bg-slate-50 px-6 py-5 border-b border-slate-200">
          <h2 className="text-lg font-semibold tracking-tight text-slate-900 mb-1">Batch triage</h2>
          <p className="text-slate-500 text-sm mb-6">Rank a set of test-split PDBs by their defensible predicted pK.</p>

          <div className="flex items-start gap-4 mb-4">
            <div className="flex-1 relative">
              <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
                Selected PDB IDs ({selected.length})
              </div>
              <div className="flex flex-wrap gap-2 text-sm font-mono">
                {selected.map(id => (
                  <span key={id} className="bg-white border border-slate-200 px-2 flex items-center gap-1 rounded text-slate-700">
                    [{id}]
                    <button
                      onClick={() => setSelected(selected.filter(x => x !== id))}
                      className="text-slate-400 hover:text-rose-600 ml-1"
                    >
                      ×
                    </button>
                  </span>
                ))}
                <button
                  onClick={() => setPickerOpen(o => !o)}
                  className="text-indigo-600 hover:text-indigo-700 flex items-center gap-1 font-sans font-medium px-2 ml-2"
                >
                  <Plus size={14} /> Add
                </button>
                <button
                  onClick={() => setSelected([])}
                  className="text-slate-500 hover:text-slate-700 flex items-center gap-1 font-sans font-medium px-2"
                >
                  <XIcon size={14} /> Clear
                </button>
              </div>

              {pickerOpen && (
                <div className="absolute top-full left-0 mt-1 z-30 bg-white border border-slate-200 rounded-md shadow-lg w-72 max-h-72 overflow-hidden flex flex-col">
                  <input
                    autoFocus
                    value={pickerFilter}
                    onChange={e => setPickerFilter(e.target.value)}
                    placeholder={`search ${whitelist.length} PDBs`}
                    className="px-3 py-2 border-b border-slate-200 text-sm focus:outline-none font-mono"
                  />
                  <div className="overflow-y-auto">
                    {pickerOptions.length === 0 && (
                      <div className="px-3 py-2 text-slate-400 text-xs italic">no matches</div>
                    )}
                    {pickerOptions.map(id => (
                      <button
                        key={id}
                        onClick={() => { setSelected([...selected, id]); setPickerFilter(''); }}
                        className="block w-full text-left px-3 py-1.5 text-sm font-mono hover:bg-indigo-50 hover:text-indigo-700"
                      >
                        {id}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="flex border border-slate-200 rounded divide-x divide-slate-200 bg-white font-medium text-sm text-slate-600 shadow-sm shrink-0">
              <span className="px-3 py-1.5">Variant: <span className="font-mono">{variant}</span></span>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex gap-4 text-sm font-medium">
              <button
                onClick={() => {
                  const shuffled = [...whitelist].sort(() => Math.random() - 0.5);
                  setSelected(shuffled.slice(0, 20));
                }}
                className="text-indigo-600 border border-indigo-200 bg-indigo-50 px-3 py-1.5 rounded hover:bg-indigo-100 transition-colors flex items-center gap-2"
              >
                <Plus size={14} /> Pick 20 random
              </button>
              <label className="flex items-center gap-2 text-slate-600 cursor-pointer pl-4">
                <input
                  type="checkbox"
                  checked={includeAgent}
                  onChange={e => setIncludeAgent(e.target.checked)}
                  className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-600"
                />
                Include agent evaluation <span className="text-slate-400 font-mono text-xs">(~${AGENT_USD_PER_CALL.toFixed(2)} ea)</span>
              </label>
            </div>
            <div className="flex items-center gap-4 text-sm">
              <span className="text-slate-500 font-mono">
                Est cost: ${estCostUsd.toFixed(2)}, ~{estLatencyMin} min
              </span>
              <button
                onClick={handleRun}
                disabled={selected.length === 0 || running}
                className="bg-slate-900 text-white px-4 py-2 rounded-md hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed font-semibold shadow-sm flex items-center gap-2"
              >
                {running && <Loader2 size={14} className="animate-spin" />}
                Run batch
              </button>
            </div>
          </div>

          {error && (
            <div className="mt-3 border border-rose-200 bg-rose-50 text-rose-800 text-sm px-3 py-2 rounded">
              {error.message} (status {error.status})
            </div>
          )}
        </div>

        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-4 text-sm font-medium text-slate-600">
              <button
                onClick={() => {
                  const next = sortBy === 'recommendation' ? 'pred' : sortBy === 'pred' ? 'delta' : 'recommendation';
                  setSortBy(next);
                }}
                className="flex items-center gap-1.5 hover:text-slate-900 px-2 py-1 rounded border border-transparent hover:border-slate-200 hover:bg-slate-50"
              >
                <SlidersHorizontal size={14} />
                Sort: {sortBy} <ChevronDown size={14} />
              </button>
              <div className="flex gap-4 border-l border-slate-200 pl-4">
                <label className="flex items-center gap-1.5 cursor-pointer">
                  <input type="checkbox" checked={filterTrust} onChange={e => setFilterTrust(e.target.checked)} className="rounded border-slate-300 text-emerald-600 focus:ring-emerald-600" /> trust
                </label>
                <label className="flex items-center gap-1.5 cursor-pointer">
                  <input type="checkbox" checked={filterReview} onChange={e => setFilterReview(e.target.checked)} className="rounded border-slate-300 text-amber-600 focus:ring-amber-600" /> review
                </label>
                <label className="flex items-center gap-1.5 cursor-pointer">
                  <input type="checkbox" checked={filterDiscard} onChange={e => setFilterDiscard(e.target.checked)} className="rounded border-slate-300 text-rose-600 focus:ring-rose-600" /> discard
                </label>
              </div>
            </div>
            <div className="text-sm text-slate-500 font-mono">{displayedRows.length}/{selected.length} shown</div>
          </div>

          <table className="w-full text-sm text-left border-collapse">
            <thead>
              <tr className="border-y border-slate-200 bg-slate-50/50 text-slate-500">
                <th className="font-semibold py-3 px-4 font-sans border-r border-slate-200">PDB</th>
                <th className="font-semibold py-3 px-4 font-sans text-right">pred</th>
                <th className="font-semibold py-3 px-4 font-sans text-right border-r border-slate-200">|Δ|</th>
                <th className="font-semibold py-3 px-4 font-sans border-r border-slate-200">regex</th>
                <th className="font-semibold py-3 px-4 font-sans border-r border-slate-200">agent</th>
                <th className="font-semibold py-3 px-4 font-sans">supporting evidence</th>
              </tr>
            </thead>
            <tbody className="font-mono text-[13px]">
              {displayedRows.length === 0 && (
                <tr><td colSpan={6} className="py-8 px-4 text-center text-slate-400 italic">No rows yet — pick PDBs and Run batch.</td></tr>
              )}
              {displayedRows.map(row => {
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
                    className="border-b border-slate-100 hover:bg-indigo-50/50 cursor-pointer group"
                  >
                    <td className="py-2.5 px-4 font-semibold text-slate-700 border-r border-slate-200 group-hover:text-indigo-700">{row.pdb_id}</td>
                    <td className="py-2.5 px-4 text-right text-slate-600">{p ? p.pK.toFixed(2) : '—'}</td>
                    <td className="py-2.5 px-4 text-right text-slate-500 border-r border-slate-200">{delta != null ? delta.toFixed(2) : '—'}</td>
                    <td className="py-2.5 px-4 text-slate-500 border-r border-slate-200 tracking-tight">{regex || '—'}</td>
                    <td className="py-2.5 px-4 border-r border-slate-200">
                      {v ? <RecommendationPill type={v.recommendation} />
                        : row.evaluating ? <span className="text-slate-400 italic text-xs">…</span>
                        : <span className="text-slate-300">—</span>}
                    </td>
                    <td className="py-2.5 px-4 text-slate-600 font-sans text-sm truncate max-w-[280px]">
                      {evidence}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <div className="mt-4 flex flex-col gap-6">
            <div className="flex gap-4">
              <button
                onClick={exportCsv}
                disabled={displayedRows.length === 0}
                className="border border-slate-200 bg-white text-slate-600 font-medium text-sm px-4 py-2 rounded-md hover:bg-slate-50 disabled:opacity-50 flex items-center gap-2 shadow-sm"
              >
                <Download size={14} /> Export selected as CSV
              </button>
              <button
                onClick={() => {
                  const trusted = displayedRows.filter(r => r.verdict?.recommendation === 'trust').map(r => r.pdb_id);
                  alert(`(mock action) would queue ${trusted.length} systems for assay: ${trusted.join(', ')}`);
                }}
                className="font-medium text-sm px-4 py-2 rounded-md hover:bg-indigo-50 text-indigo-700 bg-indigo-50/50 border border-indigo-100 flex items-center gap-2"
              >
                Send only "trust" to assay queue (mock)
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
