import { useEffect, useRef, useState } from 'react';
import {
  ChevronDown, Play, Pause, Box, Activity, Loader2, Sparkles, Search,
} from 'lucide-react';
import { LineChart, Line, ReferenceLine, ResponsiveContainer, YAxis } from 'recharts';

import {
  RecommendationPill, VerifierMark, Citation, ScoreBar, AgentTrace,
  DarkInput, Segmented,
} from '../components/ui.tsx';
import { StructureViewer } from '../components/StructureViewer.tsx';
import { GlowCard, CardHeader } from '../components/GlowCard.tsx';
import { AnimatedNumber } from '../components/AnimatedNumber.tsx';
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

  const [pdbString, setPdbString] = useState<string | null>(null);
  const [pdbLoading, setPdbLoading] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const playTimerRef = useRef<number | null>(null);

  const [channels, setChannels] = useState<ChannelFrame[] | null>(null);

  // Mount: pdb list + health
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const [idsRes, healthRes] = await Promise.all([
        api.pdbIds({ limit: 50 }),
        api.health(),
      ]);
      if (cancelled) return;
      if (idsRes.ok) setPdbIds(idsRes.data);
      if (healthRes.ok) setHealth(healthRes.data);
    })();
    return () => { cancelled = true; };
  }, []);

  // Picker autocomplete
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

  // Reset stale results on PDB/variant change
  useEffect(() => {
    setPrediction(null);
    setAgent(null);
    setPredictError(null);
    setAgentError(null);
  }, [pdb, variant]);

  // Trajectory string
  useEffect(() => {
    let cancelled = false;
    setPdbString(null);
    setCurrentFrame(0);
    setPlaying(true);
    setPdbLoading(true);
    (async () => {
      const res = await api.pdbString(pdb);
      if (cancelled) return;
      setPdbLoading(false);
      if (res.ok) setPdbString(res.data);
    })();
    return () => { cancelled = true; };
  }, [pdb]);

  // Channels
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

  const nFrames = channels?.length ?? 100;

  // Play loop
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
    <div className="flex flex-col gap-6">
      {/* ---------- Top Toolbar ---------- */}
      <GlowCard className="px-4 py-3 sm:px-5 sm:py-3.5 fx-fade-up">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-3 sm:gap-5 relative">
          <div className="flex items-center gap-2 sm:gap-3">
            <span className="text-[10px] font-mono tracking-[0.2em] uppercase"
                  style={{ color: 'var(--color-ink-dim)' }}>
              PDB
            </span>
            <button
              onClick={() => setPdbSearchOpen(o => !o)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg font-mono text-sm transition-colors"
              style={{
                background: '#f3f5fb',
                border: '1px solid var(--color-line)',
                color: 'var(--color-ink)',
              }}
            >
              <Search size={12} style={{ color: 'var(--color-ink-dim)' }} />
              {pdb}
              <ChevronDown size={12} style={{ color: 'var(--color-ink-dim)' }} />
            </button>

            {pdbSearchOpen && (
              <div className="absolute top-full left-0 sm:left-12 mt-2 z-30 w-[min(20rem,calc(100vw-2rem))] max-h-80 overflow-hidden flex flex-col glass-strong rounded-xl bevel-border fx-fade-soft">
                <DarkInput
                  autoFocus
                  value={pdbFilter}
                  onChange={e => setPdbFilter(e.target.value)}
                  placeholder={`search ${pdbIds.length} PDBs`}
                  className="!rounded-none !border-0 !border-b"
                />
                <div className="overflow-y-auto">
                  {filteredPdbs.length === 0 && (
                    <div className="px-3 py-2 text-xs italic font-mono"
                         style={{ color: 'var(--color-ink-dim)' }}>
                      no matches{pdbIds.length === 0 ? ' — backend not loaded' : ''}
                    </div>
                  )}
                  {filteredPdbs.map(id => (
                    <button
                      key={id}
                      onClick={() => {
                        onPdbChange(id);
                        setPdbSearchOpen(false);
                        setPdbFilter('');
                      }}
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

          <div className="hidden sm:block h-6 w-px" style={{ background: 'var(--color-line)' }} />

          <div className="flex items-center gap-2 sm:gap-3">
            <span className="text-[10px] font-mono tracking-[0.2em] uppercase"
                  style={{ color: 'var(--color-ink-dim)' }}>
              Variant
            </span>
            <Segmented value={variant} options={['v1a', 'v1b'] as const} onChange={onVariantChange} />
          </div>

          <div className="ml-auto flex items-center gap-2 sm:gap-3 w-full sm:w-auto justify-end">
            {health && health.status !== 'ready' && (
              <span className="text-[11px] font-mono px-2 py-1 rounded"
                    style={{
                      color: '#FF9900',
                      background: 'rgba(255, 153, 0, 0.12)',
                      border: '1px solid rgba(255, 153, 0, 0.4)',
                    }}>
                backend · {health.status}
              </span>
            )}
            <button
              disabled={predictLoading}
              onClick={handlePredict}
              className="btn-primary rounded-lg px-4 sm:px-5 py-2 text-sm flex items-center gap-2"
            >
              {predictLoading ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
              {predictLoading ? 'Predicting…' : 'Predict'}
            </button>
          </div>
        </div>
      </GlowCard>

      {predictError && (
        <div className="rounded-lg px-4 py-2 text-sm fx-fade-in"
             style={{
               background: 'rgba(244, 98, 95, 0.12)',
               border: '1px solid rgba(244, 98, 95, 0.4)',
               color: '#a8332f',
             }}>
          Predict failed: {predictError.message} (status {predictError.status})
        </div>
      )}

      {/* ---------- Prediction + 3D viewer ---------- */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_minmax(0,1.25fr)] gap-5 fx-fade-up" style={{ animationDelay: '80ms' }}>
        {/* Left: prediction + rationale */}
        <div className="flex flex-col gap-5">
          <PredictionCard prediction={prediction} />
          <RationalePanel prediction={prediction} />
        </div>

        {/* Right: 3D viewer + channels */}
        <div className="flex flex-col gap-5">
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
          <ChannelsPanel channels={channels} currentFrame={currentFrame} nFrames={nFrames} />
        </div>
      </div>

      {/* ---------- Agent verdict ---------- */}
      <div className="fx-fade-up" style={{ animationDelay: '160ms' }}>
        <AgentVerdictPanel
          prediction={prediction}
          agent={agent}
          agentLoading={agentLoading}
          agentError={agentError}
          onRequestRun={requestAgentRun}
        />
      </div>

      {/* ---------- Action row ---------- */}
      <div className="flex flex-wrap gap-3 fx-fade-up" style={{ animationDelay: '220ms' }}>
        <button className="btn-ghost rounded-lg px-4 py-2 text-sm flex items-center gap-2">
          Show baselines <ChevronDown size={14} className="opacity-50" />
        </button>
        <button
          onClick={() => onVariantChange(variant === 'v1a' ? 'v1b' : 'v1a')}
          className="btn-ghost rounded-lg px-4 py-2 text-sm flex items-center gap-2"
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
          className="sm:ml-auto rounded-lg px-4 py-2 text-sm flex items-center gap-2 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          style={{
            background: 'rgba(84, 103, 242, 0.12)',
            color: '#5467F2',
            border: '1px solid rgba(84, 103, 242, 0.4)',
          }}
        >
          Export trace as JSON
        </button>
      </div>

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


// ---------------------------------------------------------------------------
// Subcomponents
// ---------------------------------------------------------------------------

function PredictionCard({ prediction }: { prediction: PredictResponse | null }) {
  return (
    <GlowCard tone={prediction ? 'cyan' : 'default'}>
      <CardHeader icon={<Activity size={14} style={{ color: '#5467F2' }} />}>
        Prediction
      </CardHeader>
      <div className="p-4 sm:p-5 grid grid-cols-[110px_1fr] sm:grid-cols-[140px_1fr] gap-y-3 gap-x-3 text-sm">
        <Label>Predicted pK</Label>
        <span className="font-mono font-semibold text-2xl tracking-tight"
              style={{ color: 'var(--color-ink)' }}>
          {prediction ? (
            <>
              <AnimatedNumber value={prediction.pK} decimals={2} className="text-gradient-cool" />
              {prediction.affinity != null && (
                <span className="text-xs font-normal ml-2"
                      style={{ color: 'var(--color-ink-dim)' }}>
                  ΔG = {prediction.affinity.toFixed(2)} kcal/mol
                </span>
              )}
            </>
          ) : <span style={{ color: 'var(--color-ink-faint)' }}>—</span>}
        </span>

        <Label>Actual pK</Label>
        <span className="font-mono" style={{ color: 'var(--color-ink-2)' }}>
          {prediction?.hidden_pK != null ? prediction.hidden_pK.toFixed(2) : '—'}
        </span>

        <Label>|Δ|</Label>
        <span
          className="font-mono"
          style={{
            color: prediction && prediction.hidden_pK != null && Math.abs(prediction.pK - prediction.hidden_pK) < 0.3
              ? '#4ec07a'
              : 'var(--color-ink-2)',
          }}
        >
          {prediction && prediction.hidden_pK != null
            ? Math.abs(prediction.pK - prediction.hidden_pK).toFixed(2)
            : '—'}
        </span>

        {prediction?.verdict && (
          <>
            <Label>Verdict</Label>
            <span
              className="font-mono text-[11px] px-2 py-0.5 rounded-full inline-block w-fit tracking-wider uppercase"
              style={{
                color:
                  prediction.verdict === 'CONFIRMED' ? '#4ec07a'
                  : prediction.verdict === 'INCONCLUSIVE' ? '#FF9900'
                  : '#f4625f',
                background:
                  prediction.verdict === 'CONFIRMED' ? 'rgba(78, 192, 122, 0.12)'
                  : prediction.verdict === 'INCONCLUSIVE' ? 'rgba(255, 153, 0, 0.12)'
                  : 'rgba(244, 98, 95, 0.12)',
                border:
                  prediction.verdict === 'CONFIRMED' ? '1px solid rgba(78, 192, 122, 0.4)'
                  : prediction.verdict === 'INCONCLUSIVE' ? '1px solid rgba(255, 153, 0, 0.4)'
                  : '1px solid rgba(244, 98, 95, 0.4)',
              }}
            >
              {prediction.verdict}
              {prediction.confidence && (
                <span className="ml-2 opacity-70">conf={prediction.confidence}</span>
              )}
            </span>
          </>
        )}

        {prediction?.disagreement_z != null && (
          <>
            <Label>Disagreement z</Label>
            <span className="font-mono" style={{ color: 'var(--color-ink-2)' }}>
              {prediction.disagreement_z.toFixed(2)}
              {prediction.independent_energy != null && (
                <span className="text-xs font-normal ml-2"
                      style={{ color: 'var(--color-ink-dim)' }}>
                  indep E = {prediction.independent_energy.toFixed(2)} kcal/mol
                </span>
              )}
            </span>
          </>
        )}

        <Label>Backbone</Label>
        <span className="font-mono text-xs" style={{ color: 'var(--color-ink-dim)' }}>
          {prediction?.model_version ?? '—'}
        </span>
      </div>
      {prediction?.verdict_reason && (
        <div className="px-5 py-3 text-xs italic border-t"
             style={{
               background: 'rgba(255, 153, 0, 0.08)',
               color: '#a55a00',
               borderColor: 'var(--color-line)',
             }}>
          {prediction.verdict_reason}
        </div>
      )}
    </GlowCard>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-[11px] font-mono tracking-[0.18em] uppercase self-center"
          style={{ color: 'var(--color-ink-dim)' }}>
      {children}
    </span>
  );
}

function RationalePanel({ prediction }: { prediction: PredictResponse | null }) {
  if (!prediction) {
    return (
      <GlowCard className="flex-1 flex flex-col">
        <CardHeader
          right={
            <span className="text-[11px] font-mono px-2 py-0.5 rounded-full"
                  style={{
                    background: '#f3f5fb',
                    color: 'var(--color-ink-dim)',
                  }}>
              — / —
            </span>
          }
        >
          Rationale (regex verified)
        </CardHeader>
        <div className="p-6 text-sm italic flex-1 flex items-center justify-center"
             style={{ color: 'var(--color-ink-dim)' }}>
          Press Predict to generate a rationale.
        </div>
      </GlowCard>
    );
  }

  const v = prediction.regex_verifier;
  const grounded = v.verified + v.contradicted;
  const pct = grounded > 0 ? Math.round((v.verified / grounded) * 100) : 0;
  const passing = grounded > 0 ? `${v.verified}/${grounded} (${pct}%)` : `0 / ${v.total}`;

  return (
    <GlowCard className="flex-1 flex flex-col">
      <CardHeader
        right={
          <span
            className="text-[11px] font-mono px-2 py-0.5 rounded-full"
            style={{
              background: pct >= 75 ? 'rgba(78, 192, 122, 0.12)' : 'rgba(255, 153, 0, 0.12)',
              color:      pct >= 75 ? '#4ec07a'        : '#FF9900',
              border: '1px solid ' + (pct >= 75 ? 'rgba(78, 192, 122, 0.4)' : 'rgba(255, 153, 0, 0.4)'),
            }}
          >
            {passing}
          </span>
        }
      >
        Rationale · regex verified
      </CardHeader>
      <div className="p-5 text-[14px] leading-relaxed flex flex-col gap-4 flex-1"
           style={{ color: 'var(--color-ink-2)' }}>
        <p className="whitespace-pre-wrap">{prediction.rationale}</p>
        {v.claims.length > 0 && (
          <ul className="border-t pt-3 flex flex-col gap-2 fx-stagger"
              style={{ borderColor: 'var(--color-line)' }}>
            {v.claims.map((c, i) => (
              <li key={i} className="flex items-start gap-2 text-xs font-mono">
                <VerifierMark status={statusToMark(c.status)} />
                <span style={{ color: 'var(--color-ink-2)' }}>{c.text}</span>
                {c.evidence && (
                  <span className="ml-auto whitespace-nowrap"
                        style={{ color: 'var(--color-ink-dim)' }}>
                    {c.evidence}
                  </span>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </GlowCard>
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
  const [mode, setMode] = useState<'image' | 'live'>('live');
  const [imageAvailable, setImageAvailable] = useState(true);
  // Reset availability when pdb changes — backend may serve a different system.
  useEffect(() => { setImageAvailable(true); }, [pdb]);
  const isMobile = typeof window !== 'undefined' && window.innerWidth < 768;
  const mdFrame = Math.max(0, Math.min(currentFrame, 99));
  const imgSrc = `/api/frame_image/${encodeURIComponent(pdb)}?frame=${mdFrame}&width=600`;

  return (
    <GlowCard className="h-[380px] flex flex-col">
      <CardHeader
        icon={<Box size={14} style={{ color: '#5467F2' }} />}
        right={
          <div className="flex gap-2 items-center">
            <Segmented value={mode} options={['image', 'live'] as const} onChange={setMode} />
            <button
              onClick={onTogglePlay}
              className="flex items-center gap-1.5 text-[11px] font-mono font-semibold px-2.5 py-1 rounded-full transition"
              style={{
                background: 'rgba(84, 103, 242, 0.15)',
                color: '#5467F2',
                border: '1px solid rgba(84, 103, 242, 0.4)',
              }}
            >
              {playing
                ? <><Pause size={11} /> pause</>
                : <><Play size={11} /> play</>}
            </button>
          </div>
        }
      >
        3D pocket · <span className="font-mono">{pdb}</span>
      </CardHeader>
      <div className="viewer-frame flex-1 relative">
        <span className="viewer-corner tl" />
        <span className="viewer-corner tr" />
        <span className="viewer-corner bl" />
        <span className="viewer-corner br" />

        {mode === 'image' && imageAvailable && (
          <img
            key={`${pdb}-${mdFrame}`}
            src={imgSrc}
            alt={`${pdb} frame ${mdFrame}`}
            className="absolute inset-0 w-full h-full object-contain"
            onError={() => { setImageAvailable(false); }}
          />
        )}
        {mode === 'image' && !imageAvailable && (
          <div className="absolute inset-0 flex items-center justify-center text-xs font-mono italic px-4 text-center"
               style={{ color: '#FF9900' }}>
            backend /frame_image unavailable — try the live viewer
          </div>
        )}
        {mode === 'live' && (
          <>
            {loading && (
              <div className="absolute inset-0 flex items-center justify-center text-xs font-mono"
                   style={{ color: 'var(--color-ink-mute)' }}>
                <Loader2 size={14} className="animate-spin mr-2" /> loading trajectory…
              </div>
            )}
            {!loading && !pdbString && (
              <div className="absolute inset-0 flex items-center justify-center text-xs font-mono italic px-4 text-center"
                   style={{ color: '#FF9900' }}>
                backend /pdb_string unavailable (HDF5 not mounted?)
              </div>
            )}
            {isMobile && pdbString && (
              <div className="absolute top-2 left-2 right-2 glass-strong rounded-md text-[10px] font-mono p-2"
                   style={{ color: 'var(--color-ink-2)' }}>
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
        {/* HUD readout */}
        <div className="absolute bottom-3 left-3 z-10 font-mono text-[10px] tracking-wider"
             style={{ color: '#5467F2' }}>
          MD {mdFrame}/99
        </div>
      </div>
      <div className="h-10 px-4 flex items-center gap-3 text-[11px] font-mono"
           style={{
             background: '#f6f8fc',
             borderTop: '1px solid var(--color-line)',
             color: 'var(--color-ink-mute)',
           }}>
        <button
          onClick={() => onFrameChange(Math.max(0, currentFrame - 1))}
          className="cursor-pointer hover:text-ink px-1 transition-colors"
        >◀</button>
        <input
          type="range"
          min={0}
          max={Math.max(0, nFrames - 1)}
          value={currentFrame}
          onChange={e => onFrameChange(parseInt(e.target.value, 10))}
          className="flex-1 h-1 accent-cyan-400"
          style={{ accentColor: '#5467F2' }}
        />
        <button
          onClick={() => onFrameChange(Math.min(nFrames - 1, currentFrame + 1))}
          className="cursor-pointer hover:text-ink px-1 transition-colors"
        >▶</button>
        <span className="w-14 text-right">
          {mode === 'image' ? `MD ${mdFrame}/99` : `${currentFrame + 1}/${nFrames}`}
        </span>
      </div>
    </GlowCard>
  );
}


function ChannelsPanel({ channels, currentFrame, nFrames }: { channels: ChannelFrame[] | null; currentFrame: number; nFrames: number }) {
  return (
    <GlowCard className="flex-1 flex flex-col">
      <CardHeader
        right={
          <span className="text-[10px] font-mono tracking-wider"
                style={{ color: '#5467F2' }}>
            MD {currentFrame}/{nFrames - 1}
          </span>
        }
      >
        Per-frame channels
      </CardHeader>
      <div className="p-4 flex-1 min-h-[240px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={channels ?? MOCK_CHART_DATA} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
            <defs>
              <linearGradient id="rmsdStroke" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%"  stopColor="#f4625f" />
                <stop offset="100%" stopColor="#FF9900" />
              </linearGradient>
              <linearGradient id="energyStroke" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%"  stopColor="#5467F2" />
                <stop offset="100%" stopColor="#8a96f5" />
              </linearGradient>
              <linearGradient id="bsasaStroke" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%"  stopColor="#8a96f5" />
                <stop offset="100%" stopColor="#4ec07a" />
              </linearGradient>
            </defs>
            <ReferenceLine x={currentFrame} stroke="#5467F2" strokeOpacity={0.7} strokeWidth={1.2} />
            <Line type="monotone" yAxisId="rmsd"   dataKey="rmsd"   stroke="url(#rmsdStroke)"   dot={false} strokeWidth={1.6} />
            <Line type="monotone" yAxisId="energy" dataKey="energy" stroke="url(#energyStroke)" dot={false} strokeWidth={1.6} />
            <Line type="monotone" yAxisId="bsasa"  dataKey="bsasa"  stroke="url(#bsasaStroke)"  dot={false} strokeWidth={1.6} />
            <YAxis yAxisId="rmsd"   hide domain={['dataMin - 0.2', 'dataMax + 0.2']} />
            <YAxis yAxisId="energy" hide domain={['dataMin - 5',   'dataMax + 5']} />
            <YAxis yAxisId="bsasa"  hide domain={['dataMin - 20',  'dataMax + 20']} />
          </LineChart>
        </ResponsiveContainer>
        <div className="flex gap-4 mt-2 justify-center text-[11px] font-mono">
          <span style={{ color: '#f4625f' }}>── RMSD</span>
          <span style={{ color: '#5467F2' }}>── energy</span>
          <span style={{ color: '#8a96f5' }}>── bSASA</span>
          {!channels && <span className="italic" style={{ color: 'var(--color-ink-dim)' }}>(channels endpoint loading…)</span>}
        </div>
      </div>
    </GlowCard>
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
      <div className="glass rounded-2xl bevel-border text-sm italic px-6 py-10 text-center"
           style={{ color: 'var(--color-ink-dim)' }}>
        Run Predict first; the independent agent panel will appear here.
      </div>
    );
  }

  if (!agent) {
    return (
      <GlowCard tone="violet" className="p-7 flex flex-col gap-4 items-start relative overflow-hidden">
        <div
          className="absolute -right-12 -top-12 w-72 h-72 rounded-full opacity-50 pointer-events-none"
          style={{
            background: 'radial-gradient(circle, rgba(138, 150, 245, 0.5), transparent 60%)',
            filter: 'blur(30px)',
          }}
        />
        <h3 className="relative font-display text-lg font-semibold tracking-tight"
            style={{ color: 'var(--color-ink)' }}>
          Independent agent verdict
        </h3>
        <p className="relative text-sm max-w-2xl leading-relaxed"
           style={{ color: 'var(--color-ink-mute)' }}>
          Re-checks this prediction against orthogonal evidence — raw coordinates, Vina,
          label-filtered literature. ~20 s and ~$0.30 per run. Cached results are free.
        </p>
        {agentError && (
          <div className="relative text-sm px-3 py-1.5 rounded"
               style={{
                 background: 'rgba(244, 98, 95, 0.12)',
                 color: '#c83b37',
                 border: '1px solid rgba(244, 98, 95, 0.4)',
               }}>
            {agentError.message} (status {agentError.status})
          </div>
        )}
        <button
          onClick={onRequestRun}
          disabled={agentLoading}
          className="relative btn-primary rounded-lg px-5 py-2.5 text-sm flex items-center gap-2"
        >
          {agentLoading && <Loader2 size={14} className="animate-spin" />}
          {agentLoading ? 'Running…' : 'Run deep evaluation'}
        </button>
      </GlowCard>
    );
  }

  const v = agent.verdict;
  const scoreRows = [
    { label: 'Structural', score: v.scores.structural_consistency, description: 'cluster_poses, clash_check' },
    { label: 'Physical',   score: v.scores.physical_consistency,    description: 'vina_rescore, hbond_persistence' },
    { label: 'Literature', score: v.scores.literature_consistency,  description: 'rag_query (label-filtered)' },
    { label: 'Chemical',   score: v.scores.chemical_plausibility,   description: 'ligand_descriptors' },
  ];

  return (
    <GlowCard tone="violet" className="p-7 relative overflow-hidden">
      <div
        className="absolute -right-10 -top-10 w-80 h-80 rounded-full opacity-40 pointer-events-none"
        style={{
          background: 'radial-gradient(circle, rgba(138, 150, 245, 0.45), transparent 60%)',
          filter: 'blur(28px)',
        }}
      />
      <h3 className="relative flex items-center gap-3 mb-6">
        <span className="font-display text-lg font-semibold tracking-tight"
              style={{ color: 'var(--color-ink)' }}>
          Independent agent verdict
        </span>
        {v.cached && (
          <span className="text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded-full"
                style={{
                  background: 'rgba(78, 192, 122, 0.12)',
                  color: '#4ec07a',
                  border: '1px solid rgba(78, 192, 122, 0.4)',
                }}>
            cached
          </span>
        )}
      </h3>

      <div className="relative mb-6 flex gap-4 items-center">
        <span className="text-[11px] font-mono tracking-[0.18em] uppercase"
              style={{ color: 'var(--color-ink-dim)' }}>
          Recommendation
        </span>
        <RecommendationPill type={v.recommendation} />
      </div>

      <div className="relative flex flex-col gap-3 mb-8 max-w-3xl fx-stagger">
        {scoreRows.map(r => (
          <ScoreBar key={r.label} label={r.label} score={r.score} description={r.description} />
        ))}
      </div>

      <AgentTrace steps={agent.trace.map((s: TraceStep) => ({
        tool: s.tool,
        result: typeof s.result === 'string' ? s.result : JSON.stringify(s.result).slice(0, 160),
      }))} />

      {v.citations.length > 0 && (
        <div className="relative mt-7 text-sm">
          <h4 className="text-[11px] font-mono tracking-[0.18em] uppercase mb-3"
              style={{ color: 'var(--color-ink-dim)' }}>
            Citations
          </h4>
          <ul className="space-y-2">
            {v.citations.map((c, i) => (
              <li key={i} className="flex items-baseline gap-2 text-[13px]"
                  style={{ color: 'var(--color-ink-2)' }}>
                <Citation>{c.chunk_id}</Citation>
                <span className="text-xs font-mono"
                      style={{ color: 'var(--color-ink-dim)' }}>
                  score={c.score.toFixed(2)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {v.independence_caveats.length > 0 && (
        <div className="relative mt-7 p-4 rounded-xl text-sm"
             style={{
               background: '#fafbfd',
               border: '1px solid var(--color-line)',
               color: 'var(--color-ink-2)',
             }}>
          <div className="text-[11px] font-mono tracking-[0.18em] uppercase mb-2"
               style={{ color: 'var(--color-ink-dim)' }}>
            Caveats
          </div>
          <ul className="space-y-1 pl-4 list-disc marker:text-[#9a9fb3]">
            {v.independence_caveats.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="relative mt-5 text-[11px] font-mono tracking-wider"
           style={{ color: 'var(--color-ink-dim)' }}>
        {v.agent_trace.tool_calls} tool calls · {(v.agent_trace.latency_ms / 1000).toFixed(1)}s ·{' '}
        {v.agent_trace.input_tokens} in / {v.agent_trace.output_tokens} out tokens
      </div>
    </GlowCard>
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
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 fx-fade-soft"
         style={{ background: 'rgba(20, 22, 43, 0.4)', backdropFilter: 'blur(8px)' }}>
      <GlowCard tone="cyan" className="max-w-md w-full p-7">
        <h3 className="font-display font-semibold text-lg mb-2 tracking-tight"
            style={{ color: 'var(--color-ink)' }}>
          Run independent agent
        </h3>
        <p className="text-sm mb-4 leading-relaxed"
           style={{ color: 'var(--color-ink-mute)' }}>
          This call will spend ~<span className="font-mono" style={{ color: 'var(--color-ink)' }}>
            ${estimateUsd.toFixed(2)}</span> on the OpenRouter API (Claude Opus 4.7 + ~6 tool calls).
        </p>
        {remaining != null && (
          <p className="text-xs font-mono mb-4"
             style={{ color: 'var(--color-ink-dim)' }}>
            Daily cap remaining: ${remaining.toFixed(2)}{blocked && ' — insufficient'}
          </p>
        )}
        <div className="flex gap-3 justify-end">
          <button
            onClick={onCancel}
            className="btn-ghost rounded-lg px-4 py-2 text-sm"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={blocked}
            className="btn-primary rounded-lg px-4 py-2 text-sm"
          >
            Confirm · run agent
          </button>
        </div>
      </GlowCard>
    </div>
  );
}
