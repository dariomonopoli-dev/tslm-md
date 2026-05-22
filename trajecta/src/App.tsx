import { useEffect, useState } from 'react';
import { clsx } from 'clsx';

import { SingleView } from './views/SingleView.tsx';
import { BatchView } from './views/BatchView.tsx';
import { FailureModesView } from './views/FailureModesView.tsx';
import { KnowledgeView } from './views/KnowledgeView.tsx';
import { AboutView } from './views/AboutView.tsx';
import { Brand } from './components/Brand.tsx';
import { BackgroundFX } from './components/BackgroundFX.tsx';
import { api } from './lib/api.ts';
import type { HealthResponse, Variant } from './types.ts';

type ViewState = 'about' | 'single' | 'batch' | 'failure' | 'knowledge';

const DEFAULT_PDB = '1A1B';
const HEALTH_POLL_MS = 5_000;

const TABS: Array<{ id: ViewState; label: string }> = [
  { id: 'about',     label: 'Overview' },
  { id: 'single',    label: 'Inspect' },
  { id: 'batch',     label: 'Triage' },
  { id: 'failure',   label: 'Failure modes' },
  { id: 'knowledge', label: 'Knowledge' },
];

export default function App() {
  const [view, setView] = useState<ViewState>('about');
  const [selectedPDB, setSelectedPDB] = useState<string>(DEFAULT_PDB);
  const [variant, setVariant] = useState<Variant>('v1b');
  const [transitionKey, setTransitionKey] = useState(0);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    setTransitionKey(k => k + 1);
  }, [view]);

  // Poll /health: fast while booting (status !== ready), then back off to
  // a longer cadence so we still notice transitions to degraded/starting.
  useEffect(() => {
    let cancelled = false;
    let timerId: number | null = null;

    const tick = async () => {
      const res = await api.health();
      if (cancelled) return;
      if (res.ok) {
        setHealth(res.data);
        const next = res.data.status === 'ready' ? 30_000 : HEALTH_POLL_MS;
        timerId = window.setTimeout(tick, next);
      } else {
        timerId = window.setTimeout(tick, HEALTH_POLL_MS);
      }
    };
    tick();

    return () => {
      cancelled = true;
      if (timerId != null) window.clearTimeout(timerId);
    };
  }, []);

  // Once /health reports the loaded variants, switch off any default
  // that isn't actually available.
  useEffect(() => {
    if (!health?.variants_loaded?.length) return;
    if (!health.variants_loaded.includes(variant)) {
      setVariant(health.variants_loaded[0] as Variant);
    }
  }, [health, variant]);

  const goToSingle = (pdb?: string) => {
    if (pdb) setSelectedPDB(pdb);
    setView('single');
  };

  return (
    <div className="min-h-screen relative" style={{ color: 'var(--color-ink)' }}>
      <BackgroundFX />

      {/* ---------- Pitch-deck nav: pill tabs, active = bordered ink ---------- */}
      <header className="sticky top-0 z-40 px-4 sm:px-6 pt-4 sm:pt-6 pb-2 sm:pb-3">
        <div className="max-w-[1320px] mx-auto flex items-center justify-between gap-3 sm:gap-6">
          <Brand onClick={() => setView('about')} />

          <nav className="hidden md:flex items-center gap-2">
            {TABS.map((tab) => {
              const active = view === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setView(tab.id)}
                  className={clsx('tab-pill', active && 'on')}
                >
                  {tab.label}
                </button>
              );
            })}
          </nav>

          <HealthPill health={health} />
        </div>

        {/* mobile: horizontal pill strip below the brand row */}
        <nav
          className="md:hidden mt-3 -mx-4 px-4 flex items-center gap-1 overflow-x-auto scrollbar-none"
          aria-label="Sections"
        >
          {TABS.map((tab) => {
            const active = view === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setView(tab.id)}
                className={clsx('tab-chip', active && 'on')}
              >
                {tab.label}
              </button>
            );
          })}
        </nav>
      </header>

      {/* ---------- View body ---------- */}
      <main className="px-4 sm:px-6 pb-20 sm:pb-24 pt-2">
        <div key={transitionKey} className="max-w-[1320px] mx-auto fx-fade-in">
          {view === 'about' && <AboutView onGoToSingle={goToSingle} onGoToTab={setView as (v: string) => void} />}
          {view === 'single' && (
            <SingleView
              pdb={selectedPDB}
              variant={variant}
              health={health}
              onPdbChange={setSelectedPDB}
              onVariantChange={setVariant}
            />
          )}
          {view === 'batch' && (
            <BatchView
              variant={variant}
              health={health}
              onVariantChange={setVariant}
              onGoToSingle={goToSingle}
            />
          )}
          {view === 'failure' && (
            <FailureModesView
              variant={variant}
              health={health}
              onVariantChange={setVariant}
              onGoToSingle={goToSingle}
            />
          )}
          {view === 'knowledge' && <KnowledgeView health={health} />}
        </div>
      </main>

      {/* ---------- Footer ---------- */}
      <footer className="border-t" style={{ borderColor: 'var(--color-line)' }}>
        <div className="max-w-[1320px] mx-auto px-4 sm:px-6 py-5 sm:py-6 flex flex-wrap items-center justify-between gap-3 sm:gap-4 text-[10px] sm:text-[11px] font-mono tracking-wider"
             style={{ color: 'var(--color-ink-dim)' }}>
          <span>TRAJECTA · COLOSSEUM IDEARUM 2026 · DEMO ARTIFACT</span>
          <span className="flex items-center gap-3">
            <span className="hidden sm:inline">MISATO · OPENTSLM-SP · ANTHROPIC</span>
            <span className="dot-live" />
          </span>
        </div>
      </footer>
    </div>
  );
}


// ---------------------------------------------------------------------------
// HealthPill — replaces the static "v0.1 · live" tag. Surfaces backend
// readiness, model + corpus version (audit-ability is the demo's whole point),
// and remaining-cap spend so users see when the agent is gated.
// ---------------------------------------------------------------------------

function HealthPill({ health }: { health: HealthResponse | null }) {
  const status = health?.status ?? 'starting';
  const dotColor =
    status === 'ready'   ? '#4ec07a' :
    status === 'degraded' ? '#f4625f' :
                            '#FF9900';
  const spendLine =
    health?.spend_today_usd != null && health?.remaining_cap_usd != null
      ? `$${health.spend_today_usd.toFixed(2)} spent · $${health.remaining_cap_usd.toFixed(2)} left`
      : null;
  const tooltip = health
    ? [
        `status: ${status}`,
        health.judge_model && `judge: ${health.judge_model}`,
        health.rag_corpus_version && `corpus: ${health.rag_corpus_version}`,
        health.inference_backend && `backend: ${health.inference_backend}`,
        spendLine,
      ].filter(Boolean).join('\n')
    : 'connecting to backend…';

  return (
    <span
      title={tooltip}
      className="flex items-center gap-2 rounded-full px-3 py-1.5 sm:px-4 sm:py-2 text-[11px] sm:text-[12px] font-mono tracking-wider shadow-card"
      style={{
        color: 'var(--color-ink-2)',
        background: '#ffffff',
        border: '1.5px solid var(--color-line)',
      }}
    >
      <span
        className="inline-block w-1.5 h-1.5 rounded-full"
        style={{ background: dotColor, boxShadow: `0 0 6px ${dotColor}` }}
      />
      <span>{status === 'ready' ? 'live' : status}</span>
      {health?.inference_backend && (
        <span className="hidden sm:inline" style={{ color: 'var(--color-ink-dim)' }}>
          · {health.inference_backend}
        </span>
      )}
    </span>
  );
}
