import { RecommendationLabel } from '../types.ts';
import { cn } from '../lib/utils.ts';
import { Check, X, AlertTriangle, ChevronDown, ChevronRight } from 'lucide-react';
import React, { useState } from 'react';

export function RecommendationPill({ type }: { type: RecommendationLabel }) {
  return (
    <span className={cn(
      "inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-sm text-xs font-medium uppercase tracking-wider",
      type === 'trust' && "bg-emerald-50 text-emerald-700 border border-emerald-200",
      type === 'review' && "bg-amber-50 text-amber-700 border border-amber-200",
      type === 'discard' && "bg-rose-50 text-rose-700 border border-rose-200",
    )}>
      {type === 'trust' && <Check size={12} className="stroke-[3]" />}
      {type === 'review' && <AlertTriangle size={12} className="stroke-[2.5]" />}
      {type === 'discard' && <X size={12} className="stroke-[3]" />}
      {type}
    </span>
  );
}

export function ScoreBar({ label, score, description }: { label: string; score: number; description: string }) {
  const filled = Math.round(score * 10);
  const blocks = '█'.repeat(filled) + '░'.repeat(10 - filled);
  
  return (
    <div className="flex gap-4 font-mono text-sm leading-6 items-start">
      <div className="w-28 text-slate-600 font-sans tracking-tight">{label}</div>
      <div className="w-32 flex whitespace-pre text-slate-400">
        <span className="text-slate-800 tracking-[-2px] mr-3">{blocks}</span>
        <span className="tabular-nums">{score.toFixed(2)}</span>
      </div>
      <div className="flex-1 text-slate-500 font-sans">({description})</div>
    </div>
  );
}

export function VerifierMark({ status }: { status: 'pass' | 'fail' | 'warn' }) {
  if (status === 'pass') return <span className="text-emerald-500 font-bold ml-1">✓</span>;
  if (status === 'fail') return <span className="text-rose-500 font-bold ml-1">✗</span>;
  return <span className="text-amber-500 font-bold ml-1">?</span>;
}

export function Citation({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-indigo-600 cursor-pointer hover:underline decoration-indigo-300 underline-offset-2">
      [{children}]
    </span>
  );
}

export function AgentTrace({ steps }: { steps: Array<{ tool: string; result: string }> }) {
  const [open, setOpen] = useState(true);
  
  return (
    <div className="mt-4 border border-slate-200/60 rounded-md bg-slate-50/50">
      <button 
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 text-slate-600 text-sm font-medium w-full p-2.5 hover:bg-slate-100/50 transition-colors"
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        Agent trace (6 steps, 18 s, $0.22)
      </button>
      
      {open && (
        <div className="px-4 pb-3 pt-1 border-t border-slate-200/60 font-mono text-xs leading-relaxed flex flex-col gap-1.5">
          {steps.map((s, idx) => (
            <div key={idx} className="flex">
              <span className="text-slate-400 w-6 shrink-0">{idx + 1}.</span>
              <span className="text-indigo-600 font-semibold w-40 shrink-0">{s.tool}</span>
              <span className="text-slate-400 px-2 shrink-0">→</span>
              <span className="text-slate-700 truncate">{s.result}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
