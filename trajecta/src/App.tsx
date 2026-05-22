import { useEffect, useState } from 'react';
import { clsx } from 'clsx';

import { SingleView } from './views/SingleView.tsx';
import { BatchView } from './views/BatchView.tsx';
import { FailureModesView } from './views/FailureModesView.tsx';
import { KnowledgeView } from './views/KnowledgeView.tsx';
import { AboutView } from './views/AboutView.tsx';
import { Brand } from './components/Brand.tsx';
import { BackgroundFX } from './components/BackgroundFX.tsx';

type ViewState = 'about' | 'single' | 'batch' | 'failure' | 'knowledge';
type Variant = 'v1a' | 'v1b';

const DEFAULT_PDB = '1A1B';

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

  useEffect(() => {
    setTransitionKey(k => k + 1);
  }, [view]);

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

          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-2 rounded-full px-3 py-1.5 sm:px-4 sm:py-2 text-[11px] sm:text-[12px] font-mono tracking-wider transition shadow-card"
            style={{
              color: 'var(--color-ink-2)',
              background: '#ffffff',
              border: '1.5px solid var(--color-line)',
            }}
          >
            <span className="dot-live" />
            v0.1 · live
          </a>
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
              onPdbChange={setSelectedPDB}
              onVariantChange={setVariant}
            />
          )}
          {view === 'batch' && (
            <BatchView
              variant={variant}
              onVariantChange={setVariant}
              onGoToSingle={goToSingle}
            />
          )}
          {view === 'failure' && (
            <FailureModesView
              variant={variant}
              onVariantChange={setVariant}
              onGoToSingle={goToSingle}
            />
          )}
          {view === 'knowledge' && <KnowledgeView />}
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
