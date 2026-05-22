import { useEffect, useRef, useState } from 'react';
import { ChevronDown, Play, Pause, Box, Activity, Loader2 } from 'lucide-react';
import { LineChart, Line, ReferenceLine, ResponsiveContainer, YAxis } from 'recharts';

import { RecommendationPill, VerifierMark, Citation, ScoreBar, AgentTrace } from '../components/ui.tsx';
import { StructureViewer } from '../components/StructureViewer.tsx';
import { api } from '../lib/api.ts';
import { MOCK_CHART_DATA } from '../data.ts';
import type {
  ApiError,
  ChannelFrame,
  EvaluateAgentResponse,
  HealthResponse,
  PredictResponse,
  TraceStep,
  Variant,
  VerifierClaim,
} from '../types.ts';


interface SingleViewProps {
  pdb: string;
  variant: Variant;
  onPdbChange: (pdb: string) => void;
  onVariantChange: (v: Variant) => void;
}


// Regex-verifier status → UI verifier mark.
function statusToMark(s: VerifierClaim['status']): 'pass' | 'fail' | 'warn' {
  if (s === 'verified') return 'pass';
  if (s === 'contradicted') return 'fail';
  return 'warn';
}

const ESTIMATED_AGENT_COST_USD = 0.30;


export function SingleView({ pdb, variant, onPdbChange, onVariantChange }: SingleViewProps) {
  const [pdbIds, setPdbIds] = useState<string[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [pdbSearchOpen, setPdbSearchOpen] = useState(false);
  const [pdbFilter, setPdbFilter] = useState('');

  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [predictLoading, setPredictLoading] = useState(false);
  const [predictError, setPredictError] = useState<ApiError | null>(null);

  const [agent, setAgent] = useState<EvaluateAgentResponse | null>(null);
  const [agentLoading, setAgentLoading] = useState(false);
  const [agentError, setAgentError] = useState<ApiError | null>(null);
  const [showCostModal, setShowCostModal] = useState(false);

  // 3D viewer state — driven by a shared currentFrame so the channel-chart cursor
  // and the structure animation stay in sync.
  const [pdbString, setPdbString] = useState<string | null>(null);
  const [pdbLoading, setPdbLoading] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const playTimerRef = useRef<number | null>(null);

  // Per-frame channels — real (rmsd/energy/dist/bsasa) for the chart.
  const [channels, setChannels] = useState<ChannelFrame[] | null>(null);

  // Mount: fetch service health + initial PDB list (empty q = all in local mode,
  // empty list in tunnel mode — user typing kicks off autocomplete).
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const [idsRes, healthRes] = await Promise.all([api.pdbIds({ limit: 50 }), api.health()]);
      if (cancelled) return;
      if (idsRes.ok) setPdbIds(idsRes.data);
      if (healthRes.ok) setHealth(healthRes.data);
    })();
    return () => { cancelled = true; };
  }, []);

  // Server-side autocomplete: refetch when the picker filter changes.
  // Debounced 250ms + AbortController so stale responses don't overwrite fresh ones.
  useEffect(() => {
    if (!pdbSearchOpen) return;
    const controller = new AbortController();
    const t = setTimeout(async () => {
      const res = await api.pdbIds({ q: pdbFilter, limit: 50, signal: controller.signal });
      if (res.ok) setPdbIds(res.data);
    }, 250);
    return () => {
      clearTimeout(t);
      controller.abort();
    };
  }, [pdbFilter, pdbSearchOpen]);

  // Changing PDB or variant clears stale results.
  useEffect(() => {
    setPrediction(null);
    setAgent(null);
    setPredictError(null);
    setAgentError(null);
  }, [pdb, variant]);

  // Fetch the multi-MODEL PDB whenever the PDB changes.
  useEffect(() => {
    let cancelled = false;
    setPdbString(null);
    setCurrentFrame(0);
    setPlaying(false);
    setPdbLoading(true);
    (async () => {
      const res = await api.pdbString(pdb);
      if (cancelled) return;
      setPdbLoading(false);
      if (res.ok) setPdbString(res.data);
    })();
    return () => { cancelled = true; };
  }, [pdb]);

  // Fetch real per-frame channels for the chart.
  useEffect(() => {
    let cancelled = false;
    setChannels(null);
    (async () => {
      const res = await api.channels(pdb);
      if (cancelled) return;
      if (res.ok) setChannels(res.data.frames);
    })();
    return () => { cancelled = true; };
  }, [pdb]);

  // Frame slider operates in raw MD-frame space (0..99) so the channel chart
  // cursor and the image-mode counter line up. The live-3D viewer translates
  // to its own stride-5 frame space internally.
  const nFrames = channels?.length ?? 100;

  // Play loop: advance the frame at ~10 fps.
  useEffect(() => {
    if (!playing) {
      if (playTimerRef.current != null) {
        window.clearInterval(playTimerRef.current);
        playTimerRef.current = null;
      }
      return;
    }
    playTimerRef.current = window.setInterval(() => {
      setCurrentFrame(f => (f + 1) % nFrames);
    }, 100);
    return () => {
      if (playTimerRef.current != null) {
        window.clearInterval(playTimerRef.current);
        playTimerRef.current = null;
      }
    };
  }, [playing, nFrames]);

  // Server now does the filtering — just take what it returned.
  const filteredPdbs = pdbIds.slice(0, 50);

  async function handlePredict() {
    setPredictLoading(true);
    setPredictError(null);
    const res = await api.predict(pdb, variant);
    setPredictLoading(false);
    if (!res.ok) {
      setPredictError(res.error);
      return;
    }
    setPrediction(res.data);
  }

  function requestAgentRun() {
    if (!prediction) return;
    setShowCostModal(true);
  }

  async function handleAgentRun(force = false) {
    setShowCostModal(false);
    setAgentLoading(true);
    setAgentError(null);
    const res = await api.evaluateAgent(pdb, variant, force);
    setAgentLoading(false);
    if (!res.ok) {
      setAgentError(res.error);
      return;
    }
    setAgent(res.data);
  }

  return (
    <div className="flex flex-col gap-6 animate-in fade-in duration-300">

      {/* ---------- Top Toolbar ---------- */}
      <div className="flex items-center gap-6 border border-slate-200 bg-white rounded-md px-4 py-3 shadow-sm text-sm relative">
        <label className="flex items-center gap-3 font-semibold text-slate-700">
          PDB ID
          <div
            className="border border-slate-200 bg-slate-50 rounded px-2 py-1 flex items-center font-mono font-medium text-slate-600 cursor-pointer hover:border-slate-300"
            onClick={() => setPdbSearchOpen(o => !o)}
          >
            {pdb} <ChevronDown size={14} className="ml-2 text-slate-400" />
          </div>
        </label>

        {pdbSearchOpen && (
          <div className="absolute top-full left-16 mt-1 z-30 bg-white border border-slate-200 rounded-md shadow-lg w-72 max-h-80 overflow-hidden flex flex-col">
            <input
              autoFocus
              value={pdbFilter}
              onChange={e => setPdbFilter(e.target.value)}
              placeholder={`search ${pdbIds.length} PDBs`}
              className="px-3 py-2 border-b border-slate-200 text-sm focus:outline-none font-mono"
            />
            <div className="overflow-y-auto">
              {filteredPdbs.length === 0 && (
                <div className="px-3 py-2 text-slate-400 text-xs italic">
                  no matches{pdbIds.length === 0 ? ' — backend not loaded' : ''}
                </div>
              )}
              {filteredPdbs.map(id => (
                <button
                  key={id}
                  onClick={() => { onPdbChange(id); setPdbSearchOpen(false); setPdbFilter(''); }}
                  className="block w-full text-left px-3 py-1.5 text-sm font-mono hover:bg-indigo-50 hover:text-indigo-700"
                >
                  {id}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="w-px h-6 bg-slate-200"></div>

        <label className="flex items-center gap-3 font-semibold text-slate-700">
          Variant
          <div className="flex border border-slate-200 rounded divide-x divide-slate-200 font-mono text-xs overflow-hidden">
            {(['v1a', 'v1b'] as const).map(v => (
              <button
                key={v}
                onClick={() => onVariantChange(v)}
                className={
                  v === variant
                    ? 'px-3 py-1.5 bg-indigo-50 text-indigo-700 font-semibold shadow-[0_2px_2px_0_inset_#c7d2fe]'
                    : 'px-3 py-1.5 bg-slate-50 hover:bg-slate-100 text-slate-500'
                }
              >
                {v}
              </button>
            ))}
          </div>
        </label>

        <div className="ml-auto flex items-center gap-3">
          {health && health.status !== 'ready' && (
            <span className="text-xs text-amber-700 font-mono bg-amber-50 border border-amber-200 px-2 py-1 rounded">
              backend: {health.status}
            </span>
          )}
          <button
            disabled={predictLoading}
            onClick={handlePredict}
            className="bg-slate-900 text-white px-5 py-1.5 rounded-md hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed font-semibold shadow-sm text-sm flex items-center gap-2"
          >
            {predictLoading && <Loader2 size={14} className="animate-spin" />}
            Predict
          </button>
        </div>
      </div>

      {predictError && (
        <div className="border border-rose-200 bg-rose-50 text-rose-800 text-sm px-4 py-2 rounded">
          Predict failed: {predictError.message} (status {predictError.status})
        </div>
      )}

      {/* ---------- Prediction + 3D + channels ---------- */}
      <div className="grid grid-cols-[1fr_minmax(0,1.2fr)] gap-6">
        {/* Left: prediction + rationale */}
        <div className="flex flex-col gap-6">
          <div className="border border-slate-200 bg-white rounded-md shadow-sm overflow-hidden">
            <div className="border-b border-slate-200 bg-slate-50 px-4 py-2 flex items-center gap-2">
              <Activity size={14} className="text-slate-500" />
              <span className="font-semibold text-slate-700 text-sm">Prediction</span>
            </div>
            <div className="p-4 grid grid-cols-[140px_1fr] gap-y-2 text-sm font-sans">
              <span className="text-slate-500">Predicted pK</span>
              <span className="font-mono font-semibold text-slate-900">
                {prediction ? prediction.pK.toFixed(2) : '—'}
                {prediction?.affinity != null && (
                  <span className="text-slate-400 font-normal text-xs ml-2">
                    (ΔG = {prediction.affinity.toFixed(2)} kcal/mol)
                  </span>
                )}
              </span>
              <span className="text-slate-500">Actual pK</span>
              <span className="font-mono text-slate-600">
                {prediction?.hidden_pK != null ? prediction.hidden_pK.toFixed(2) : '—'}
              </span>
              <span className="text-slate-500">|Δ|</span>
              <span className={`font-mono ${prediction && prediction.hidden_pK != null && Math.abs(prediction.pK - prediction.hidden_pK) < 0.3 ? 'text-emerald-600' : 'text-slate-600'}`}>
                {prediction && prediction.hidden_pK != null
                  ? Math.abs(prediction.pK - prediction.hidden_pK).toFixed(2)
                  : '—'}
              </span>
              {prediction?.verdict && (
                <>
                  <span className="text-slate-500">Verdict</span>
                  <span className={`font-mono text-xs px-2 py-0.5 rounded inline-block w-fit ${
                    prediction.verdict === 'CONFIRMED'
                      ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                      : prediction.verdict === 'INCONCLUSIVE'
                        ? 'bg-amber-50 text-amber-700 border border-amber-200'
                        : 'bg-rose-50 text-rose-700 border border-rose-200'
                  }`}>
                    {prediction.verdict}
                    {prediction.confidence && (
                      <span className="ml-2 opacity-70">conf={prediction.confidence}</span>
                    )}
                  </span>
                </>
              )}
              {prediction?.disagreement_z != null && (
                <>
                  <span className="text-slate-500">Disagreement z</span>
                  <span className="font-mono text-slate-600">
                    {prediction.disagreement_z.toFixed(2)}
                    {prediction.independent_energy != null && (
                      <span className="text-slate-400 font-normal text-xs ml-2">
                        (indep E = {prediction.independent_energy.toFixed(2)} kcal/mol)
                      </span>
                    )}
                  </span>
                </>
              )}
              <span className="text-slate-500">Backbone</span>
              <span className="font-mono text-slate-600 text-xs">
                {prediction?.model_version ?? '—'}
              </span>
            </div>
            {prediction?.verdict_reason && (
              <div className="border-t border-slate-200 bg-amber-50/40 px-4 py-2 text-xs text-amber-800 font-sans italic">
                {prediction.verdict_reason}
              </div>
            )}
          </div>

          <RationalePanel prediction={prediction} />
        </div>

        {/* Right: 3D viewer + channels */}
        <div className="flex flex-col gap-6">
          <StructurePanel
            pdb={pdb}
            pdbString={pdbString}
            loading={pdbLoading}
            currentFrame={currentFrame}
            nFrames={nFrames}
            playing={playing}
            onTogglePlay={() => setPlaying(p => !p)}
            onFrameChange={setCurrentFrame}
          />
          <div className="border border-slate-200 bg-white rounded-md shadow-sm overflow-hidden flex-1 flex flex-col">
            <div className="border-b border-slate-200 bg-slate-50 px-4 py-2 flex items-center justify-between">
              <span className="font-semibold text-slate-700 text-sm tracking-tight">Per-frame channels</span>
              <span className="text-xs font-mono text-indigo-500">
                MD {currentFrame}/{nFrames - 1}
              </span>
            </div>
            <div className="p-4 grow font-mono text-xs text-slate-500 relative min-h-[260px]">
              <div className="h-full w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={channels ?? MOCK_CHART_DATA} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                    <ReferenceLine x={currentFrame} stroke="#6366f1" strokeOpacity={0.6} strokeWidth={1} />
                    {/* Three independent y-axes — bSASA at ~950 would otherwise crush RMSD at ~1 */}
                    <Line type="monotone" yAxisId="rmsd" dataKey="rmsd" stroke="#f43f5e" dot={false} strokeWidth={1.5} />
                    <Line type="monotone" yAxisId="energy" dataKey="energy" stroke="#3b82f6" dot={false} strokeWidth={1.5} />
                    <Line type="monotone" yAxisId="bsasa" dataKey="bsasa" stroke="#8b5cf6" dot={false} strokeWidth={1.5} />
                    <YAxis yAxisId="rmsd" hide domain={['dataMin - 0.2', 'dataMax + 0.2']} />
                    <YAxis yAxisId="energy" hide domain={['dataMin - 5', 'dataMax + 5']} />
                    <YAxis yAxisId="bsasa" hide domain={['dataMin - 20', 'dataMax + 20']} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="flex gap-4 mt-2 justify-center">
                <span className="text-rose-500">── RMSD</span>
                <span className="text-blue-500">── energy</span>
                <span className="text-violet-500">── bSASA</span>
                {!channels && <span className="text-slate-400 italic">(channels endpoint loading…)</span>}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ---------- Agent verdict ---------- */}
      <AgentVerdictPanel
        prediction={prediction}
        agent={agent}
        agentLoading={agentLoading}
        agentError={agentError}
        onRequestRun={requestAgentRun}
      />

      <div className="flex gap-4">
        <button className="border border-slate-200 bg-white text-slate-600 font-medium text-sm px-4 py-2 rounded-md hover:bg-slate-50 flex items-center gap-2 shadow-sm">
          Show baselines <ChevronDown size={14} className="opacity-50" />
        </button>
        <button
          onClick={() => onVariantChange(variant === 'v1a' ? 'v1b' : 'v1a')}
          className="border border-slate-200 bg-white text-slate-600 font-medium text-sm px-4 py-2 rounded-md hover:bg-slate-50 flex items-center gap-2 shadow-sm"
        >
          Compare to {variant === 'v1a' ? 'v1b' : 'v1a'} ⇄
        </button>
        <button
          disabled={!agent}
          onClick={() => {
            if (!agent) return;
            const blob = new Blob([JSON.stringify(agent, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${pdb}_${variant}_trace.json`;
            a.click();
            URL.revokeObjectURL(url);
          }}
          className="ml-auto font-medium text-sm px-4 py-2 rounded-md hover:bg-indigo-50 disabled:opacity-40 disabled:cursor-not-allowed text-indigo-700 border border-indigo-100 flex items-center bg-indigo-50/50"
        >
          Export trace as JSON
        </button>
      </div>

      {/* ---------- Cost confirmation modal ---------- */}
      {showCostModal && (
        <CostModal
          health={health}
          estimateUsd={ESTIMATED_AGENT_COST_USD}
          onCancel={() => setShowCostModal(false)}
          onConfirm={() => handleAgentRun(false)}
        />
      )}
    </div>
  );
}


// --------------------------------------------------------------------------
// Subcomponents
// --------------------------------------------------------------------------


function RationalePanel({ prediction }: { prediction: PredictResponse | null }) {
  if (!prediction) {
    return (
      <div className="border border-slate-200 bg-white rounded-md shadow-sm overflow-hidden flex-1 flex flex-col">
        <div className="border-b border-slate-200 bg-slate-50 px-4 py-2 flex justify-between items-center">
          <span className="font-semibold text-slate-700 text-sm tracking-tight">Rationale (regex verified)</span>
          <span className="text-xs text-slate-500 font-mono bg-slate-200 px-1.5 py-0.5 rounded">— / —</span>
        </div>
        <div className="p-5 text-sm text-slate-400 italic grow flex items-center justify-center">
          Press Predict to generate a rationale.
        </div>
      </div>
    );
  }

  const v = prediction.regex_verifier;
  const grounded = v.verified + v.contradicted;
  const pct = grounded > 0 ? Math.round((v.verified / grounded) * 100) : 0;
  const passing = grounded > 0 ? `${v.verified}/${grounded} (${pct}%)` : `0 / ${v.total}`;

  return (
    <div className="border border-slate-200 bg-white rounded-md shadow-sm overflow-hidden flex-1 flex flex-col">
      <div className="border-b border-slate-200 bg-slate-50 px-4 py-2 flex justify-between items-center">
        <span className="font-semibold text-slate-700 text-sm tracking-tight">Rationale (regex verified)</span>
        <span className="text-xs text-slate-500 font-mono bg-slate-200 px-1.5 py-0.5 rounded">{passing}</span>
      </div>
      <div className="p-5 text-sm leading-relaxed text-slate-700 font-sans flex flex-col gap-4 bg-slate-50/30 grow">
        <p className="whitespace-pre-wrap">{prediction.rationale}</p>
        {v.claims.length > 0 && (
          <ul className="border-t border-slate-200 pt-3 flex flex-col gap-1.5">
            {v.claims.map((c, i) => (
              <li key={i} className="flex items-start gap-2 text-xs font-mono">
                <VerifierMark status={statusToMark(c.status)} />
                <span className="text-slate-600">{c.text}</span>
                {c.evidence && <span className="text-slate-400 ml-auto whitespace-nowrap">{c.evidence}</span>}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}


function StructurePanel({
  pdb, pdbString, loading, currentFrame, nFrames, playing, onTogglePlay, onFrameChange,
}: {
  pdb: string;
  pdbString: string | null;
  loading: boolean;
  currentFrame: number;
  nFrames: number;
  playing: boolean;
  onTogglePlay: () => void;
  onFrameChange: (f: number) => void;
}) {
  const [mode, setMode] = useState<'image' | 'live'>('image');
  const isMobile = typeof window !== 'undefined' && window.innerWidth < 768;
  // currentFrame is now in raw MD-frame space (0..99) for everything.
  // The image endpoint takes the raw frame directly.
  const mdFrame = Math.max(0, Math.min(currentFrame, 99));
  const imgSrc = `/api/frame_image/${encodeURIComponent(pdb)}?frame=${mdFrame}&width=600`;

  return (
    <div className="border border-slate-200 bg-white rounded-md shadow-sm overflow-hidden h-[340px] flex flex-col">
      <div className="border-b border-slate-200 bg-slate-50 px-4 py-2 flex items-center justify-between">
        <span className="font-semibold text-slate-700 text-sm flex items-center gap-2">
          <Box size={14} className="text-slate-500" /> 3D pocket view ({pdb})
        </span>
        <div className="flex gap-2">
          {/* Image (static, server-rendered) vs Live (3Dmol, client-side) */}
          <div className="flex border border-slate-200 rounded divide-x divide-slate-200 text-xs font-mono overflow-hidden bg-white">
            <button
              onClick={() => setMode('image')}
              className={mode === 'image' ? 'px-2 py-1 bg-slate-200 text-slate-800' : 'px-2 py-1 text-slate-500 hover:bg-slate-50'}
            >image</button>
            <button
              onClick={() => setMode('live')}
              className={mode === 'live' ? 'px-2 py-1 bg-slate-200 text-slate-800' : 'px-2 py-1 text-slate-500 hover:bg-slate-50'}
            >live 3D</button>
          </div>
          <button
            onClick={onTogglePlay}
            className="flex items-center gap-1.5 text-xs font-semibold px-2 py-1 rounded bg-slate-200 hover:bg-slate-300 text-slate-700 transition"
          >
            {playing ? <><Pause size={12} className="fill-slate-700" /> pause</> : <><Play size={12} className="fill-slate-700" /> play</>}
          </button>
        </div>
      </div>
      <div className="grow bg-[#1a1b26] relative overflow-hidden">
        {mode === 'image' && (
          <img
            key={`${pdb}-${mdFrame}`}
            src={imgSrc}
            alt={`${pdb} frame ${mdFrame}`}
            className="absolute inset-0 w-full h-full object-contain"
            onError={e => { (e.currentTarget as HTMLImageElement).style.display = 'none'; }}
          />
        )}
        {mode === 'live' && (
          <>
            {loading && (
              <div className="absolute inset-0 flex items-center justify-center text-slate-400 text-xs font-mono">
                <Loader2 size={14} className="animate-spin mr-2" /> loading trajectory…
              </div>
            )}
            {!loading && !pdbString && (
              <div className="absolute inset-0 flex items-center justify-center text-amber-300 text-xs font-mono italic px-4 text-center">
                backend /pdb_string unavailable (HDF5 not mounted?)
              </div>
            )}
            {isMobile && pdbString && (
              <div className="absolute top-2 left-2 right-2 bg-slate-900/80 text-slate-200 text-[10px] font-mono p-2 rounded">
                best viewed on desktop — 3D viewer disabled on small screens
              </div>
            )}
            {!isMobile && pdbString && (
              <StructureViewer
                pdbString={pdbString}
                currentFrame={Math.floor(currentFrame / 5)}
                className="absolute inset-0"
              />
            )}
          </>
        )}
      </div>
      <div className="bg-slate-900 h-10 px-4 flex items-center justify-between text-xs font-mono text-slate-400">
        <button
          onClick={() => onFrameChange(Math.max(0, currentFrame - 1))}
          className="cursor-pointer hover:text-white px-1"
        >◀</button>
        <input
          type="range"
          min={0}
          max={Math.max(0, nFrames - 1)}
          value={currentFrame}
          onChange={e => onFrameChange(parseInt(e.target.value, 10))}
          className="flex-1 mx-4 h-1 accent-indigo-500"
        />
        <button
          onClick={() => onFrameChange(Math.min(nFrames - 1, currentFrame + 1))}
          className="cursor-pointer hover:text-white px-1"
        >▶</button>
        <span className="ml-4 w-14 text-right">
          {mode === 'image' ? `MD ${mdFrame}/99` : `${currentFrame + 1}/${nFrames}`}
        </span>
      </div>
    </div>
  );
}


function AgentVerdictPanel({
  prediction, agent, agentLoading, agentError, onRequestRun,
}: {
  prediction: PredictResponse | null;
  agent: EvaluateAgentResponse | null;
  agentLoading: boolean;
  agentError: ApiError | null;
  onRequestRun: () => void;
}) {
  if (!prediction) {
    return (
      <div className="border border-slate-200 bg-slate-50/50 rounded-md text-slate-500 text-sm italic px-6 py-10 text-center">
        Run Predict first; the independent agent panel will appear here.
      </div>
    );
  }

  if (!agent) {
    return (
      <div className="border border-indigo-200 bg-indigo-50/30 rounded-md p-6 flex flex-col gap-3 items-start">
        <h3 className="font-semibold text-slate-800">Independent agent verdict</h3>
        <p className="text-sm text-slate-600 max-w-2xl">
          Re-checks this prediction against orthogonal evidence (raw coordinates,
          Vina, label-filtered literature). Takes ~20 s and costs ~$0.30 per run.
          Cached results are free.
        </p>
        {agentError && (
          <div className="text-sm text-rose-700 bg-rose-50 border border-rose-200 px-3 py-1 rounded">
            {agentError.message} (status {agentError.status})
          </div>
        )}
        <button
          onClick={onRequestRun}
          disabled={agentLoading}
          className="bg-indigo-600 text-white text-sm font-semibold px-4 py-2 rounded-md hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {agentLoading && <Loader2 size={14} className="animate-spin" />}
          {agentLoading ? 'Running…' : 'Run deep evaluation'}
        </button>
      </div>
    );
  }

  const v = agent.verdict;
  const scoreRows = [
    { label: 'Structural', score: v.scores.structural_consistency, description: 'cluster_poses, clash_check' },
    { label: 'Physical', score: v.scores.physical_consistency, description: 'vina_rescore, hbond_persistence' },
    { label: 'Literature', score: v.scores.literature_consistency, description: 'rag_query (label-filtered)' },
    { label: 'Chemical', score: v.scores.chemical_plausibility, description: 'ligand_descriptors' },
  ];

  return (
    <div className="border border-indigo-200 bg-[#fbfbfe] rounded-md shadow-sm overflow-hidden relative">
      <div className="absolute top-0 left-0 bottom-0 w-1 bg-indigo-500"></div>
      <div className="p-6 pl-8">
        <h3 className="font-semibold text-slate-800 mb-6 flex items-center gap-2">
          Independent agent verdict
          {v.cached && <span className="text-xs font-mono bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded">cached</span>}
        </h3>

        <div className="mb-6 flex gap-4 items-center">
          <span className="text-slate-500 text-sm font-medium">Recommendation:</span>
          <RecommendationPill type={v.recommendation} />
        </div>

        <div className="flex flex-col gap-2.5 mb-8 max-w-3xl">
          {scoreRows.map(r => (
            <ScoreBar key={r.label} label={r.label} score={r.score} description={r.description} />
          ))}
        </div>

        <AgentTrace steps={agent.trace.map((s: TraceStep) => ({
          tool: s.tool,
          result: typeof s.result === 'string' ? s.result : JSON.stringify(s.result).slice(0, 160),
        }))} />

        {v.citations.length > 0 && (
          <div className="mt-8 mb-8 text-sm font-sans">
            <h4 className="font-semibold text-slate-700 mb-3 tracking-tight flex items-center">
              <ChevronDown size={14} className="mr-1 -ml-1 text-slate-400" /> Citations
            </h4>
            <ul className="pl-5 space-y-2 text-slate-600">
              {v.citations.map((c, i) => (
                <li key={i}>
                  <Citation>{c.chunk_id}</Citation>{' '}
                  <span className="text-xs text-slate-400 font-mono">score={c.score.toFixed(2)}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {v.independence_caveats.length > 0 && (
          <div className="mb-8 p-4 bg-slate-50 border border-slate-200 rounded-md text-sm text-slate-700 font-sans shadow-sm">
            <div className="font-semibold mb-2">Caveats:</div>
            <ul className="space-y-1 text-slate-600 pl-4 list-disc marker:text-slate-400">
              {v.independence_caveats.map((c, i) => (
                <li key={i}>{c}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="text-xs font-mono text-slate-500">
          {v.agent_trace.tool_calls} tool calls · {(v.agent_trace.latency_ms / 1000).toFixed(1)} s ·
          {' '}{v.agent_trace.input_tokens} in / {v.agent_trace.output_tokens} out tokens
        </div>
      </div>
    </div>
  );
}


function CostModal({
  health, estimateUsd, onCancel, onConfirm,
}: {
  health: HealthResponse | null;
  estimateUsd: number;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const remaining = health?.remaining_cap_usd;
  const blocked = remaining != null && remaining < estimateUsd;

  return (
    <div className="fixed inset-0 z-50 bg-slate-900/40 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-2xl max-w-md w-full p-6">
        <h3 className="font-semibold text-slate-900 text-lg mb-2">Run independent agent</h3>
        <p className="text-sm text-slate-600 mb-4 leading-relaxed">
          This call will spend ~<span className="font-mono">${estimateUsd.toFixed(2)}</span> on the
          OpenRouter API (Claude Opus 4.7 + ~6 tool calls).
        </p>
        {remaining != null && (
          <p className="text-xs font-mono text-slate-500 mb-4">
            Daily cap remaining: ${remaining.toFixed(2)}{blocked && ' — insufficient'}
          </p>
        )}
        <div className="flex gap-3 justify-end">
          <button
            onClick={onCancel}
            className="px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-50 rounded-md"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={blocked}
            className="px-4 py-2 text-sm font-semibold text-white bg-indigo-600 rounded-md hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Confirm — run agent
          </button>
        </div>
      </div>
    </div>
  );
}
