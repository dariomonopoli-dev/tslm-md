import { cn } from '../lib/utils.ts';
import React from 'react';

interface GlowCardProps {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
  tone?: 'default' | 'cyan' | 'violet' | 'lime' | 'amber' | 'rose';
  ticks?: boolean;
}

const TONE_CLASS: Record<NonNullable<GlowCardProps['tone']>, string> = {
  default: '',
  cyan: 'glow-brand',
  violet: 'glow-brand',
  lime: 'glow-ok',
  amber: '',
  rose: 'glow-warn',
};

// Specimen-sheet card. Sharp hairline border, optional cobalt corner-ticks.
export function GlowCard({
  children, className, style, tone = 'default', ticks = false,
}: GlowCardProps) {
  return (
    <div
      className={cn(
        'panel relative shadow-card',
        ticks && 'ticks',
        TONE_CLASS[tone],
        className,
      )}
      style={style}
    >
      {ticks && (
        <>
          <span className="tick-tr" />
          <span className="tick-bl" />
        </>
      )}
      {children}
    </div>
  );
}

interface CardHeaderProps {
  children: React.ReactNode;
  right?: React.ReactNode;
  icon?: React.ReactNode;
  refId?: string;     // e.g. "REF 03.1"
  className?: string;
}

// Card header with optional ref-id stamp on the right — specimen-sheet feel.
export function CardHeader({ children, right, icon, refId, className }: CardHeaderProps) {
  return (
    <div
      className={cn(
        'px-5 py-3 flex items-center justify-between gap-3',
        className,
      )}
      style={{ borderBottom: '1.5px solid var(--color-line)' }}
    >
      <div className="flex items-center gap-2.5"
           style={{ color: 'var(--color-ink-2)' }}>
        {icon}
        <span className="font-display font-semibold text-[13px] tracking-tight">
          {children}
        </span>
      </div>
      <div className="flex items-center gap-3">
        {right}
        {refId && (
          <span className="stamp">{refId}</span>
        )}
      </div>
    </div>
  );
}
