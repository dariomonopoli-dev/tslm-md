import { useEffect, useState } from 'react';
import { ArrowRight, ArrowUpRight } from 'lucide-react';

import { MoleculeHero } from '../components/MoleculeHero.tsx';

interface AboutViewProps {
  onGoToSingle?: (pdb: string) => void;
  onGoToTab?: (tab: string) => void;
}

const HERO_PDBS = ['1A1B', '1A28', '4QZL'];

export function AboutView({ onGoToSingle, onGoToTab }: AboutViewProps) {
  const [heroIdx, setHeroIdx] = useState(0);
  useEffect(() => {
    const id = window.setInterval(() => setHeroIdx(i => (i + 1) % HERO_PDBS.length), 12_000);
    return () => window.clearInterval(id);
  }, []);
  const heroPdb = HERO_PDBS[heroIdx];

  return (
    <div className="flex flex-col gap-16 sm:gap-24 lg:gap-28 pb-16">
      {/* ============================================================
         HERO: asymmetric editorial composition
         ============================================================ */}
      <section className="pt-4 sm:pt-10 relative">
        <div className="grid grid-cols-12 gap-x-8 gap-y-10 sm:gap-y-12">
          <div className="col-span-12 lg:col-span-7 flex flex-col gap-6 sm:gap-7 fx-fade-up">
            <div className="flex flex-wrap items-center gap-2 sm:gap-3">
              <span className="sec-id"><span className="num">01</span> &nbsp;/&nbsp; OVERVIEW</span>
              <span className="rule flex-1 max-w-[140px] fx-rule-draw hidden sm:inline-block" />
              <span className="stamp">COLOSSEUM IDEARUM · 2026</span>
            </div>

            <h1 className="font-display font-bold tracking-[-0.045em] text-[clamp(2.8rem,1rem+5.5vw,6rem)] leading-[0.92]"
                style={{ color: 'var(--color-ink)' }}>
              Motion tells what{' '}
              <span className="serif" style={{ color: 'var(--color-brand)' }}>structure</span>{' '}
              <span className="serif" style={{ color: 'var(--color-brand)' }}>hides.</span>
            </h1>

            <p className="text-[16px] sm:text-[18px] leading-[1.55] max-w-[44ch]"
               style={{ color: 'var(--color-ink-2)' }}>
              A language model that reads protein–ligand{' '}
              <span className="serif" style={{ color: 'var(--color-ink)' }}>molecular dynamics</span>{' '}
              to predict binding affinity, paired with an independent
              agent that audits every claim before it ships.
            </p>

            <div className="flex flex-wrap items-center gap-3 pt-2">
              <button
                onClick={() => onGoToTab?.('single')}
                className="group btn-primary px-5 py-3 text-[14px] flex items-center gap-2"
              >
                Inspect a system
                <ArrowRight size={15} className="transition-transform group-hover:translate-x-0.5" />
              </button>
              <button
                onClick={() => onGoToTab?.('batch')}
                className="btn-ghost px-5 py-3 text-[14px]"
              >
                Run a batch triage
              </button>
              <button
                onClick={() => onGoToTab?.('failure')}
                className="btn-ghost px-5 py-3 text-[14px]"
              >
                Where it fails
              </button>
            </div>
          </div>

          <div className="col-span-12 lg:col-span-5 relative fx-fade-up lg:translate-y-12"
               style={{ animationDelay: '120ms' }}>
            <div className="relative h-[320px] sm:h-[440px] lg:h-[520px]">
              <MoleculeHero pdb={heroPdb} className="absolute inset-0" />
            </div>

            <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
              <span className="stamp">SYSTEM SELECTOR</span>
              <div className="flex gap-2 flex-wrap">
                {HERO_PDBS.map((id, i) => (
                  <button
                    key={id}
                    onClick={() => setHeroIdx(i)}
                    className={`px-3 py-1 text-[11px] font-mono tracking-[0.16em] uppercase rounded-full transition-all ${
                      i === heroIdx ? 'panel-tint' : 'panel hover:bg-[#f6f8fc]'
                    }`}
                    style={{
                      color: i === heroIdx ? 'var(--color-brand)' : 'var(--color-ink-mute)',
                    }}
                  >
                    {id}
                  </button>
                ))}
              </div>
            </div>
          </div>

        </div>
      </section>

      {/* ============================================================
         WORKED EXAMPLES
         ============================================================ */}
      <section className="flex flex-col gap-8 fx-fade-up">
        <SectionLockup id="02" eyebrow="Try these" title="Three worked examples."
                       subtitle="Curated to show what the audit catches, and what it can’t." />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 fx-stagger">
          <ExampleCard n="01" pdb="1A1B" tone="ok"    tag="easy / trust"
            title="Stable binder."
            blurb="All four evidence sources agree: model, regex, physics, literature."
            onClick={() => onGoToSingle?.('1A1B')} />
          <ExampleCard n="02" pdb="4QZL" tone="brand" tag="hard / right"
            title="Model beats Vina by 1.8 pK."
            blurb="The audit lands on trust after literature confirms a non-obvious pose."
            onClick={() => onGoToSingle?.('4QZL')} />
          <ExampleCard n="03" pdb="2X3K" tone="warn"  tag="failure caught"
            title="Confident, and wrong."
            blurb="Model overshoots. The agent catches it with ligand-efficiency reasoning."
            onClick={() => onGoToSingle?.('2X3K')} />
        </div>
      </section>

      {/* ============================================================
         INDEPENDENCE: printed specimen sheet
         ============================================================ */}
      <section className="fx-fade-up">
        <div className="panel-tint relative p-6 sm:p-9 ticks">
          <span className="tick-tr" />
          <span className="tick-bl" />
          <div className="flex items-start justify-between mb-7 gap-6 flex-wrap">
            <div>
              <span className="sec-id"><span className="num">03</span> &nbsp;/&nbsp; GUARDRAILS</span>
              <h3 className="font-display font-bold tracking-[-0.025em] text-[clamp(1.8rem,1rem+1.6vw,2.6rem)] mt-3"
                  style={{ color: 'var(--color-ink)' }}>
                Independence <span className="serif" style={{ color: 'var(--color-brand)' }}>guarantees.</span>
              </h3>
            </div>
            <div className="text-right">
              <div className="stamp">REF 03.0 · I.S.O. · TRAJECTA</div>
              <div className="stamp mt-1" style={{ color: 'var(--color-brand)' }}>v0.1 · audited</div>
            </div>
          </div>

          <p className="text-[15px] mb-5 max-w-[60ch]"
             style={{ color: 'var(--color-ink-mute)' }}>
            The agent that audits each prediction operates under three hard rules.
          </p>

          <ol className="space-y-0 max-w-[64ch]">
            <Guarantee n="01" body="Uses only tools that operate on data the trained model did not see: raw atomic coordinates, external force fields, label-filtered RAG." />
            <Guarantee n="02" body="Cannot “look up the answer.” RAG chunks containing the system’s experimental Kd are excluded at retrieval time for the system under test." />
            <Guarantee n="03" body="Refuses to use prior knowledge. Every factual claim must cite either a tool output or a retrieved evidence chunk." />
          </ol>
        </div>
      </section>

      {/* ============================================================
         FOOTER: colophon
         ============================================================ */}
      <section className="flex flex-col gap-3 items-center pt-6">
        <span className="rule max-w-[400px]" />
        <div className="flex flex-wrap items-center justify-center gap-x-5 gap-y-2 text-[12px] font-mono tracking-wider"
             style={{ color: 'var(--color-ink-dim)' }}>
          <FooterLink>Authors</FooterLink>
          <FooterLink>Source code</FooterLink>
          <FooterLink>Paper draft</FooterLink>
          <FooterLink accent>PROJECT_BRIEF.md ↗</FooterLink>
          <FooterLink>Contact</FooterLink>
        </div>
      </section>
    </div>
  );
}

// ---------------------------------------------------------------------------

function SectionLockup({ id, eyebrow, title, subtitle }:
  { id: string; eyebrow: string; title: string; subtitle?: string }) {
  return (
    <div className="flex flex-col gap-3 max-w-3xl">
      <div className="flex items-center gap-3">
        <span className="sec-id"><span className="num">{id}</span> &nbsp;/&nbsp; {eyebrow.toUpperCase()}</span>
        <span className="rule flex-1 max-w-[80px]" />
      </div>
      <h2 className="font-display font-bold tracking-[-0.03em] text-[clamp(2rem,1rem+2.2vw,3rem)] leading-[1.02]"
          style={{ color: 'var(--color-ink)' }}>
        {title}
      </h2>
      {subtitle && (
        <p className="text-[15.5px] max-w-[58ch]" style={{ color: 'var(--color-ink-mute)' }}>
          {subtitle}
        </p>
      )}
    </div>
  );
}

function ExampleCard({
  n, pdb, tone, tag, title, blurb, onClick,
}: {
  n: string; pdb: string; tone: 'ok' | 'brand' | 'warn';
  tag: string; title: string; blurb: string; onClick?: () => void;
}) {
  const c = tone === 'ok' ? '#4ec07a' : tone === 'brand' ? '#5467F2' : '#f4625f';
  return (
    <button
      onClick={onClick}
      className="text-left panel ticks p-6 flex flex-col gap-4 transition-all hover:-translate-y-1 hover:shadow-card-lg group relative"
    >
      <span className="tick-tr" />
      <span className="tick-bl" />
      <div className="flex items-start justify-between">
        <span className="stamp" style={{ color: c }}>{n} / {tag}</span>
        <ArrowUpRight size={16}
          className="transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5"
          style={{ color: 'var(--color-ink-dim)' }} />
      </div>
      <MiniSchematic tone={c} />
      <div className="flex items-baseline gap-3 mt-1">
        <span className="font-display font-bold tracking-[-0.03em] text-[32px] leading-none"
              style={{ color: 'var(--color-ink)' }}>
          {pdb}
        </span>
      </div>
      <h4 className="font-display font-semibold text-[17px] tracking-tight leading-snug"
          style={{ color: 'var(--color-ink-2)' }}>
        {title}
      </h4>
      <p className="text-[13.5px] leading-[1.55] -mt-2"
         style={{ color: 'var(--color-ink-mute)' }}>
        {blurb}
      </p>
    </button>
  );
}

function MiniSchematic({ tone }: { tone: string }) {
  return (
    <svg viewBox="0 0 120 80" className="w-full h-20" aria-hidden>
      <ellipse cx="60" cy="40" rx="50" ry="32" className="mol-pocket" />
      <line x1="60" y1="40" x2="32" y2="22" className="mol-bond" />
      <line x1="60" y1="40" x2="88" y2="22" className="mol-bond" />
      <line x1="60" y1="40" x2="46" y2="64" className="mol-bond" />
      <g className="mol-lig">
        <circle cx="60" cy="40" r="10" fill={tone} />
      </g>
      <circle cx="32" cy="22" r="6" fill="none" stroke="var(--color-mol-stroke)" strokeWidth="2.5" />
      <circle cx="88" cy="22" r="6" fill="none" stroke="var(--color-mol-stroke)" strokeWidth="2.5" />
      <circle cx="46" cy="64" r="5" fill="none" stroke="var(--color-mol-stroke)" strokeWidth="2.5" />
    </svg>
  );
}

function Guarantee({ n, body }: { n: string; body: string }) {
  return (
    <li className="flex items-start gap-5 py-4"
        style={{ borderBottom: '1.5px solid rgba(84, 103, 242, 0.25)' }}>
      <span className="font-display font-semibold text-[15px] tracking-[0.04em] shrink-0 w-8 pt-0.5"
            style={{ color: 'var(--color-brand)' }}>
        {n}
      </span>
      <span className="text-[14.5px] leading-[1.55]"
            style={{ color: 'var(--color-ink-2)' }}>
        {body}
      </span>
    </li>
  );
}

function FooterLink({ children, accent }: { children: React.ReactNode; accent?: boolean }) {
  return (
    <span
      className="cursor-pointer transition-colors"
      style={{ color: accent ? 'var(--color-brand)' : 'var(--color-ink-dim)' }}
    >
      {children}
    </span>
  );
}
