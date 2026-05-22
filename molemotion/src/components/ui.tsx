import { useState } from 'react';
import { Check, X, AlertTriangle, ChevronDown, ChevronRight } from 'lucide-react';
import React from 'react';

import { RecommendationLabel } from '../types.ts';
import { cn } from '../lib/utils.ts';

// ---------------------------------------------------------------------------
// Recommendation pill
// ---------------------------------------------------------------------------

const PILL_STYLE: Record<RecommendationLabel, { fg: string; bg: string; bd: string }> = {
  trust:   { fg: '#4ec07a', bg: 'rgba(78, 192, 122, 0.12)', bd: 'rgba(78, 192, 122, 0.5)' },
  review:  { fg: '#FF9900',  bg: 'rgba(255, 153, 0, 0.12)',  bd: 'rgba(255, 153, 0, 0.5)'  },
  discard: { fg: '#f4625f',  bg: 'rgba(244, 98, 95, 0.12)',  bd: 'rgba(244, 98, 95, 0.5)'  },
};

export function RecommendationPill({ type }: { type: RecommendationLabel }) {
  const s = PILL_STYLE[type];
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-mono font-semibold uppercase tracking-[0.15em]"
      style={{ color: s.fg, background: s.bg, border: `1px solid ${s.bd}` }}
    >
      {type === 'trust' && <Check size={11} className="stroke-[3]" />}
      {type === 'review' && <AlertTriangle size={11} className="stroke-[2.5]" />}
      {type === 'discard' && <X size={11} className="stroke-[3]" />}
      {type}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Score bar — used in agent verdict
// ---------------------------------------------------------------------------

export function ScoreBar({ label, score, description }: { label: string; score: number; description: string }) {
  const pct = Math.max(0, Math.min(1, score));
  const hue = 130 + (1 - pct) * (22 - 130); // green at 1.0, red at 0.0
  return (
    <div className="flex items-center gap-4 text-sm">
      <div className="w-28 shrink-0 font-medium tracking-tight"
           style={{ color: 'var(--color-ink-2)' }}>
        {label}
      </div>
      <div className="flex items-center gap-3 flex-1">
        <div
          className="relative h-1.5 rounded-full overflow-hidden flex-1 max-w-[200px]"
          style={{ background: '#eef1fa' }}
        >
          <div
            className="absolute inset-y-0 left-0 rounded-full transition-all duration-700"
            style={{
              width: `${pct * 100}%`,
              background: `linear-gradient(90deg, #5467F2, hsl(${hue}, 60%, 55%))`,
              boxShadow: `0 0 12px rgba(84, 103, 242, 0.5)`,
            }}
          />
        </div>
        <span className="font-mono text-xs tabular-nums w-10 text-right"
              style={{ color: 'var(--color-ink)' }}>
          {score.toFixed(2)}
        </span>
      </div>
      <div className="flex-1 text-xs font-mono tracking-tight"
           style={{ color: 'var(--color-ink-dim)' }}>
        {description}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Verifier mark
// ---------------------------------------------------------------------------

export function VerifierMark({ status }: { status: 'pass' | 'fail' | 'warn' }) {
  if (status === 'pass') return <span className="font-bold" style={{ color: '#4ec07a' }}>✓</span>;
  if (status === 'fail') return <span className="font-bold" style={{ color: '#f4625f' }}>✗</span>;
  return <span className="font-bold" style={{ color: '#FF9900' }}>?</span>;
}

// ---------------------------------------------------------------------------
// Citation chip
// ---------------------------------------------------------------------------

export function Citation({ children }: { children: React.ReactNode }) {
  return (
    <span
      className="cursor-pointer transition-colors"
      style={{ color: '#5467F2' }}
    >
      [{children}]
    </span>
  );
}

// ---------------------------------------------------------------------------
// Agent trace — terminal-style fade-in
// ---------------------------------------------------------------------------

interface AgentTraceProps {
  steps: Array<{ tool: string; result: string }>;
}

export function AgentTrace({ steps }: AgentTraceProps) {
  const [open, setOpen] = useState(true);

  return (
    <div
      className="mt-4 rounded-xl overflow-hidden"
      style={{
        background: '#fafbfd',
        border: '1px solid var(--color-line)',
      }}
    >
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 text-sm font-medium w-full p-3 hover:bg-[#f3f5fb] transition-colors"
        style={{ color: 'var(--color-ink-2)' }}
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <span>Agent trace</span>
        <span className="font-mono text-[11px]" style={{ color: 'var(--color-ink-dim)' }}>
          · {steps.length} steps · 18s · $0.22
        </span>
        <span
          className="ml-auto inline-block w-1.5 h-1.5 rounded-full"
          style={{
            background: '#4ec07a',
            boxShadow: '0 0 6px #4ec07a',
          }}
        />
      </button>

      {open && (
        <div
          className="px-4 pb-3 pt-2 border-t font-mono text-xs leading-relaxed flex flex-col gap-1"
          style={{ borderColor: 'var(--color-line)' }}
        >
          {steps.map((s, idx) => (
            <div
              key={idx}
              className="flex gap-2 fx-rise"
              style={{ animationDelay: `${idx * 60}ms` }}
            >
              <span className="w-6 shrink-0" style={{ color: 'var(--color-ink-dim)' }}>
                {String(idx + 1).padStart(2, '0')}
              </span>
              <span className="w-44 shrink-0 font-semibold tracking-tight"
                    style={{ color: '#5467F2' }}>
                {s.tool}
              </span>
              <span className="shrink-0" style={{ color: 'var(--color-ink-dim)' }}>→</span>
              <span className="truncate" style={{ color: 'var(--color-ink-2)' }}>
                {s.result}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tiny input — used for picker filters etc.
// ---------------------------------------------------------------------------

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {}

export function DarkInput({ className, ...rest }: InputProps) {
  return (
    <input
      {...rest}
      className={cn(
        'rounded-md px-3 py-2 text-sm font-mono focus:outline-none transition-colors w-full',
        className,
      )}
      style={{
        background: '#ffffff',
        border: '1px solid var(--color-line)',
        color: 'var(--color-ink)',
      }}
    />
  );
}

// ---------------------------------------------------------------------------
// Segmented control (variants etc.)
// ---------------------------------------------------------------------------

interface SegmentedProps<T extends string> {
  value: T;
  options: ReadonlyArray<T>;
  onChange: (next: T) => void;
  className?: string;
}

export function Segmented<T extends string>({ value, options, onChange, className }: SegmentedProps<T>) {
  return (
    <div className={cn('seg', className)}>
      {options.map(opt => {
        const active = opt === value;
        return (
          <button
            key={opt}
            onClick={() => onChange(opt)}
            className={cn(
              'relative px-3 py-1 text-[12px] font-mono font-medium rounded-full transition-colors',
              active ? '' : 'hover:text-ink',
            )}
            style={{
              color: active ? 'var(--color-ink)' : 'var(--color-ink-mute)',
              background: active
                ? 'linear-gradient(135deg, rgba(84, 103, 242, 0.2), rgba(138, 150, 245, 0.2))'
                : 'transparent',
              boxShadow: active
                ? 'inset 0 0 0 1px rgba(84, 103, 242, 0.4)'
                : 'none',
            }}
          >
            {opt}
          </button>
        );
      })}
    </div>
  );
}
