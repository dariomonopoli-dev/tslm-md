import { useState } from 'react';
import { clsx } from 'clsx';
import { SingleView } from './views/SingleView.tsx';
import { BatchView } from './views/BatchView.tsx';
import { FailureModesView } from './views/FailureModesView.tsx';
import { KnowledgeView } from './views/KnowledgeView.tsx';
import { AboutView } from './views/AboutView.tsx';
import { Database } from 'lucide-react';

type ViewState = 'single' | 'batch' | 'failure' | 'knowledge' | 'about';
type Variant = 'v1a' | 'v1b';

const DEFAULT_PDB = '1A1B';

export default function App() {
  const [view, setView] = useState<ViewState>('batch');
  const [selectedPDB, setSelectedPDB] = useState<string>(DEFAULT_PDB);
  const [variant, setVariant] = useState<Variant>('v1b');

  const goToSingle = (pdb?: string) => {
    if (pdb) setSelectedPDB(pdb);
    setView('single');
  };

  const tabs = [
    { id: 'single', label: 'Single' },
    { id: 'batch', label: 'Batch' },
    { id: 'failure', label: 'Failure modes' },
    { id: 'knowledge', label: 'Knowledge' },
    { id: 'about', label: 'About' },
  ];

  return (
    <div className="min-h-screen bg-slate-100/50 text-slate-900 font-sans p-[4vh] flex justify-center">

      {/* Outer Browser/App Window Shell */}
      <div className="w-full max-w-6xl bg-white shadow-xl shadow-slate-200/50 border border-slate-200/60 rounded-xl overflow-hidden flex flex-col">

        {/* Header Navigation */}
        <header className="flex items-center justify-between px-6 bg-[#f8fafc] border-b border-slate-200 shrink-0">

          <div className="flex items-center gap-3 py-4 text-slate-800 cursor-pointer" onClick={() => setView('about')}>
            <div className="p-1.5 bg-blue-500 rounded">
               <Database className="text-white" size={16} />
            </div>
            <div className="flex flex-col">
              <div className="font-bold tracking-tight text-[18px] leading-none mb-1">
                 <span className="text-blue-500">Mole</span><span className="text-slate-900">Motion</span>
              </div>
              <span className="text-[11px] text-slate-500 italic leading-none tracking-wide">motion tells what structure hides.</span>
            </div>
          </div>

          <nav className="flex gap-1 h-full items-end">
            {tabs.map((tab) => {
              const active = view === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setView(tab.id as ViewState)}
                  className={clsx(
                    "relative px-4 py-3.5 text-[14px] font-medium transition-colors hover:text-slate-900 hover:bg-white bottom-[-1px]",
                    active
                      ? "text-indigo-700 bg-white border-x border-t border-slate-200 border-b border-b-transparent rounded-t-md z-10"
                      : "text-slate-500 border border-transparent"
                  )}
                >
                  {active && <span className="absolute top-0 left-0 right-0 h-[2px] bg-indigo-600 rounded-t-md" />}
                  {tab.label}
                </button>
              );
            })}
          </nav>

        </header>

        {/* Content Area */}
        <main className="p-8 grow bg-white overflow-y-auto">
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
           {view === 'about' && <AboutView />}
        </main>

      </div>

    </div>
  );
}
