import { FileCode, ShieldCheck, Database, ScanSearch, Activity } from 'lucide-react';

export function AboutView() {
  return (
    <div className="flex flex-col gap-10 animate-in fade-in duration-300 max-w-5xl">
      
      {/* Header Area */}
      <div>
        <h1 className="text-4xl font-bold tracking-tight mb-2">
          <span className="text-blue-500">Mole</span><span className="text-slate-900">Motion</span>
        </h1>
        <p className="text-lg text-slate-500 font-medium tracking-tight mb-8 italic">motion tells what structure hides.</p>
        
        <p className="text-slate-600 leading-relaxed font-sans max-w-3xl text-[15px]">
          This demo applies OpenTSLM-SoftPrompt (Stanford BDHG, arXiv 2510.02410) 
          to a new modality: protein-ligand molecular dynamics trajectories from 
          the MISATO dataset. Every prediction is paired with a natural-language 
          rationale, which is then independently audited by a tool-using agent.
        </p>
      </div>

      <div className="border border-slate-200 rounded-lg bg-white overflow-hidden shadow-sm">
         <div className="bg-slate-50 px-6 py-4 border-b border-slate-200">
           <h2 className="text-sm font-semibold tracking-wide text-slate-600 uppercase">Try these worked examples</h2>
         </div>
         <div className="p-6 flex flex-col gap-3 font-mono text-[13px]">
            <div className="flex items-center gap-4 hover:bg-slate-50 p-2 rounded cursor-pointer group">
               <span className="text-indigo-600 font-semibold group-hover:underline">[1A1B  — easy, trust]</span>
               <span className="text-slate-500 font-sans text-sm">Stable binder, all 4 evidence sources agree</span>
            </div>
            <div className="flex items-center gap-4 hover:bg-slate-50 p-2 rounded cursor-pointer group">
               <span className="text-indigo-600 font-semibold group-hover:underline">[4QZL  — hard but right]</span>
               <span className="text-slate-500 font-sans text-sm">Model beats Vina by 1.8 pK, lit confirms</span>
            </div>
            <div className="flex items-center gap-4 hover:bg-slate-50 p-2 rounded cursor-pointer group">
               <span className="text-rose-600 font-semibold group-hover:underline">[2X3K  — failure caught]</span>
               <span className="text-slate-500 font-sans text-sm">Model overshoots; agent catches the gap</span>
            </div>
         </div>
      </div>

      {/* Architecture Block Map */}
      <div className="pt-2 border-t border-slate-200">
        <h2 className="text-lg font-semibold tracking-tight text-slate-900 mb-8">How it works</h2>

        <div className="flex items-start gap-4 text-center font-sans">
          
          <div className="flex flex-col items-center">
             <div className="border border-slate-300 shadow-sm rounded-md bg-white w-32 h-24 flex flex-col items-center justify-center p-2 mb-3">
               <Database className="text-slate-400 mb-2" size={24} />
               <div className="font-semibold text-slate-700 text-sm">MISATO</div>
               <div className="text-xs text-slate-500 font-mono">MD HDF5</div>
             </div>
             <div className="text-xs text-slate-500 max-w-[100px] leading-relaxed">4 channels / system</div>
          </div>

          <div className="text-slate-300 self-center -mt-8 font-mono text-xl tracking-tighter">──</div>

          <div className="flex flex-col items-center">
             <div className="border border-indigo-200 shadow-sm rounded-md bg-indigo-50 w-32 h-24 flex flex-col items-center justify-center p-2 mb-3">
               <Activity className="text-indigo-500 mb-2" size={24} />
               <div className="font-semibold text-indigo-900 text-sm">trained</div>
               <div className="text-xs text-indigo-700 font-mono">TSLM</div>
             </div>
             <div className="text-xs text-slate-500 max-w-[100px] leading-relaxed">pK + rationale</div>
          </div>

          <div className="text-slate-300 self-center -mt-8 font-mono text-xl tracking-tighter">──</div>

          <div className="flex flex-col items-center">
             <div className="border border-amber-200 shadow-sm rounded-md bg-amber-50 w-32 h-24 flex flex-col items-center justify-center p-2 mb-3">
               <FileCode className="text-amber-500 mb-2" size={24} />
               <div className="font-semibold text-amber-900 text-sm">regex</div>
               <div className="text-xs text-amber-700 font-mono">verifier</div>
             </div>
             <div className="text-xs text-slate-500 max-w-[120px] leading-relaxed">rationale vs.<br/>channels</div>
          </div>

          <div className="text-slate-300 self-center -mt-8 font-mono text-xl tracking-tighter">──</div>

          <div className="flex flex-col items-center">
           <div className="border border-emerald-200 shadow-sm rounded-md bg-emerald-50 w-36 h-24 flex flex-col items-center justify-center p-2 mb-3">
               <ShieldCheck className="text-emerald-500 mb-2" size={24} />
               <div className="font-semibold text-emerald-900 text-sm">independent</div>
               <div className="text-xs text-emerald-700 font-mono">agent</div>
             </div>
             <div className="text-xs text-slate-500 max-w-[120px] leading-relaxed">physics +<br/>structure +<br/>literature</div>
          </div>

        </div>
      </div>

      <div className="grid grid-cols-2 gap-10 border-t border-slate-200 pt-10">
        <div>
          <h3 className="text-sm font-semibold tracking-wide text-slate-600 uppercase mb-4 w-full border-b border-slate-200 pb-2">What this demo IS</h3>
          <ul className="space-y-4 text-[14px] text-slate-700 leading-relaxed font-sans">
            <li className="flex gap-3"><span className="text-emerald-500 font-bold shrink-0">✓</span> Application of OpenTSLM-SP to a new modality (MD trajectories)</li>
            <li className="flex gap-3"><span className="text-emerald-500 font-bold shrink-0">✓</span> Grounded rationales — every claim checkable against either the input channels or independent physics</li>
            <li className="flex gap-3"><span className="text-emerald-500 font-bold shrink-0">✓</span> An auditable AI — the agent shows its work, cites its sources, and respects independence boundaries from the trained model</li>
          </ul>
        </div>
        <div>
          <h3 className="text-sm font-semibold tracking-wide text-slate-600 uppercase mb-4 w-full border-b border-slate-200 pb-2">What this demo is NOT</h3>
          <ul className="space-y-4 text-[14px] text-slate-700 leading-relaxed font-sans">
            <li className="flex gap-3"><span className="text-rose-500 font-bold shrink-0">✗</span> A production drug-discovery tool. Demo-grade artifact.</li>
            <li className="flex gap-3"><span className="text-rose-500 font-bold shrink-0">✗</span> A replacement for wet-lab assays. Use as a triage layer.</li>
            <li className="flex gap-3"><span className="text-rose-500 font-bold shrink-0">✗</span> A regulatory or clinical decision tool.</li>
            <li className="flex gap-3"><span className="text-rose-500 font-bold shrink-0">✗</span> A model that beats experimental accuracy. 10 ns of MD limits resolution to ~±0.3 pK.</li>
          </ul>
        </div>
      </div>

      <div className="border border-slate-200 bg-slate-50 rounded-lg p-6 mb-8">
        <h3 className="font-semibold text-slate-900 mb-4 flex items-center gap-2 tracking-tight">
           <ScanSearch size={18} className="text-indigo-500" /> Independence guarantees
        </h3>
        <p className="text-sm text-slate-600 mb-3 font-medium">The agent that audits each prediction:</p>
        <ul className="space-y-2 text-sm text-slate-600 font-sans list-disc pl-5 marker:text-slate-400">
           <li>Uses only tools that operate on data the trained model did not see (raw atomic coordinates, external force fields, label-filtered RAG)</li>
           <li>Cannot "look up the answer" — RAG chunks containing the system's experimental Kd are excluded at retrieval time for the system under test</li>
           <li>Refuses to use prior knowledge: every factual claim must cite either a tool output or a retrieved evidence chunk</li>
        </ul>
      </div>

      <div className="flex gap-4 text-sm font-semibold text-slate-500 items-center justify-center pb-12">
        <span className="cursor-pointer hover:text-slate-800 transition">Authors</span> <span className="opacity-40">•</span>
        <span className="cursor-pointer hover:text-slate-800 transition">Source code</span> <span className="opacity-40">•</span>
        <span className="cursor-pointer hover:text-slate-800 transition">Paper draft</span> <span className="opacity-40">•</span>
        <span className="cursor-pointer font-mono text-indigo-600 hover:text-indigo-800 transition">PROJECT_BRIEF.md</span> <span className="opacity-40 font-sans">•</span>
        <span className="cursor-pointer hover:text-slate-800 transition">Contact</span>
      </div>

    </div>
  );
}
