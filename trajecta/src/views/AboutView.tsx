import { useEffect, useState } from 'react';
import { ArrowRight, ArrowUpRight } from 'lucide-react';

import { MoleculeHero } from '../components/MoleculeHero.tsx';
import { AnimatedNumber } from '../components/AnimatedNumber.tsx';

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
    <div className="flex flex-col gap-28 pb-16">
      {/* ============================================================
         HERO — asymmetric editorial composition
         ============================================================ */}
      <section className="pt-10 relative">
        <div className="grid grid-cols-12 gap-x-8 gap-y-12">
          <div className="col-span-12 lg:col-span-7 flex flex-col gap-7 fx-fade-up">
            <div className="flex items-center gap-3">
              <span className="sec-id"><span className="num">01</span> &nbsp;/&nbsp; OVERVIEW</span>
              <span className="rule flex-1 max-w-[140px] fx-rule-draw" />
              <span className="stamp">COLOSSEUM IDEARUM · 2026</span>
            </div>

            <h1 className="font-display font-bold tracking-[-0.045em] text-[clamp(2.8rem,1rem+5.5vw,6rem)] leading-[0.92]"
                style={{ color: 'var(--color-ink)' }}>
              Motion tells what{' '}
              <span className="serif" style={{ color: 'var(--color-brand)' }}>structure</span>{' '}
              <span className="serif" style={{ color: 'var(--color-brand)' }}>hides.</span>
            </h1>

            <p className="text-[18px] leading-[1.55] max-w-[44ch]"
               style={{ color: 'var(--color-ink-2)' }}>
              A language model that reads protein–ligand{' '}
              <span className="serif" style={{ color: 'var(--color-ink)' }}>molecular dynamics</span>{' '}
              to predict binding affinity — paired with an independent
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
            <div className="relative h-[460px] lg:h-[520px]">
              <MoleculeHero pdb={heroPdb} className="absolute inset-0" />
            </div>

            <div className="mt-4 flex items-center justify-between">
              <span className="stamp">SYSTEM SELECTOR</span>
              <div className="flex gap-2">
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

          <div className="col-span-12 grid grid-cols-3 gap-x-8 mt-4 fx-fade-up"
               style={{ animationDelay: '300ms' }}>
            <Stat label="Test systems"        value={130}  decimals={0} />
            <Stat label="Channels per frame" value={4}    decimals={0} />
            <Stat label="Avg pK error"       value={0.43} decimals={2} suffix=" pK" />
          </div>
        </div>
      </section>

      {/* ============================================================
         WORKED EXAMPLES
         ============================================================ */}
      <section className="flex flex-col gap-8 fx-fade-up">
        <SectionLockup id="02" eyebrow="Try these" title="Three worked examples."
                       subtitle="Curated to show what the audit catches — and what it can’t." />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 fx-stagger">
          <ExampleCard n="01" pdb="1A1B" tone="ok"    tag="easy / trust"
            title="Stable binder."
            blurb="All four evidence sources agree — model, regex, physics, literature."
            onClick={() => onGoToSingle?.('1A1B')} />
          <ExampleCard n="02" pdb="4QZL" tone="brand" tag="hard / right"
            title="Model beats Vina by 1.8 pK."
            blurb="The audit lands on trust after literature confirms a non-obvious pose."
            onClick={() => onGoToSingle?.('4QZL')} />
          <ExampleCard n="03" pdb="2X3K" tone="warn"  tag="failure caught"
            title="Confident — and wrong."
            blurb="Model overshoots. The agent catches it with ligand-efficiency reasoning."
            onClick={() => onGoToSingle?.('2X3K')} />
        </div>
      </section>

      {/* ============================================================
         PIPELINE — deck architecture slide
         ============================================================ */}
      <section className="flex flex-col gap-8 fx-fade-up">
        <SectionLockup id="03" eyebrow="The pipeline" title="A four-stage audit."
                       subtitle="Every claim has to defend itself against orthogonal evidence." />
        <PipelineDiagram />
      </section>

      {/* ============================================================
         IS / IS NOT
         ============================================================ */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-6 fx-fade-up">
        <SpecimenList refId="REF 04.1" eyebrow="IS"
          title="What this demo is."
          tone="brand"
          items={[
            'An application of OpenTSLM-Flamingo to a new modality — MD trajectories.',
            'Grounded rationales — every claim checkable against the input channels or independent physics.',
            'An auditable AI — the agent shows its work, cites its sources, respects independence.',
          ]} />
        <SpecimenList refId="REF 04.2" eyebrow="IS NOT"
          title="What this demo is not."
          tone="warn"
          items={[
            'A production drug-discovery tool. Demo-grade artifact.',
            'A replacement for wet-lab assays. Use as a triage layer.',
            'A regulatory or clinical decision tool.',
            'A model that beats experimental accuracy — 10 ns of MD limits resolution to ~±0.3 pK.',
          ]} />
      </section>

      {/* ============================================================
         INDEPENDENCE — printed specimen sheet
         ============================================================ */}
      <section className="fx-fade-up">
        <div className="panel-tint relative p-9 ticks">
          <span className="tick-tr" />
          <span className="tick-bl" />
          <div className="flex items-start justify-between mb-7 gap-6 flex-wrap">
            <div>
              <span className="sec-id"><span className="num">05</span> &nbsp;/&nbsp; GUARDRAILS</span>
              <h3 className="font-display font-bold tracking-[-0.025em] text-[clamp(1.8rem,1rem+1.6vw,2.6rem)] mt-3"
                  style={{ color: 'var(--color-ink)' }}>
                Independence <span className="serif" style={{ color: 'var(--color-brand)' }}>guarantees.</span>
              </h3>
            </div>
            <div className="text-right">
              <div className="stamp">REF 05.0 · I.S.O. · TRAJECTA</div>
              <div className="stamp mt-1" style={{ color: 'var(--color-brand)' }}>v0.1 · audited</div>
            </div>
          </div>

          <p className="text-[15px] mb-5 max-w-[60ch]"
             style={{ color: 'var(--color-ink-mute)' }}>
            The agent that audits each prediction operates under three hard rules.
          </p>

          <ol className="space-y-0 max-w-[64ch]">
            <Guarantee n="01" body="Uses only tools that operate on data the trained model did not see — raw atomic coordinates, external force fields, label-filtered RAG." />
            <Guarantee n="02" body="Cannot “look up the answer.” RAG chunks containing the system’s experimental Kd are excluded at retrieval time for the system under test." />
            <Guarantee n="03" body="Refuses to use prior knowledge — every factual claim must cite either a tool output or a retrieved evidence chunk." />
          </ol>
        </div>
      </section>

      {/* ============================================================
         FOOTER — colophon
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

function Stat({ label, value, decimals = 0, suffix = '' }:
  { label: string; value: number; decimals?: number; suffix?: string }) {
  return (
    <div className="flex flex-col gap-3">
      <span className="rule-strong fx-rule-draw" />
      <div className="flex items-baseline justify-between">
        <span className="text-[10px] font-mono tracking-[0.22em] uppercase"
              style={{ color: 'var(--color-ink-dim)' }}>
          {label}
        </span>
        <span className="stamp" style={{ color: 'var(--color-brand)' }}>↑</span>
      </div>
      <div className="display-num text-[clamp(2.4rem,1rem+3vw,4rem)]"
           style={{ color: 'var(--color-ink)' }}>
        <AnimatedNumber value={value} decimals={decimals} suffix={suffix} duration={1400} />
      </div>
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

const PIPELINE: Array<{
  step: string; title: string; sub: string; kind: 'ours' | 'base';
}> = [
  { step: '01', title: 'MD trajectory',   sub: '10 ch × 30 frames',     kind: 'ours' },
  { step: '02', title: 'Chronos-2',       sub: 'time-series encoder',   kind: 'ours' },
  { step: '03', title: 'Perceiver',       sub: 'resampler → latents',   kind: 'base' },
  { step: '04', title: 'Llama-3.2-1B',    sub: '❄ frozen · cross-attn', kind: 'base' },
  { step: '05', title: 'Regression head', sub: 'L1 loss on ΔG',         kind: 'ours' },
];

function PipelineDiagram() {
  return (
    <div className="panel p-7 relative overflow-hidden">
      <div className="flex items-stretch gap-3 overflow-x-auto no-scrollbar pt-3">
        {PIPELINE.map((p, i) => (
          <PipelineStep key={p.step} {...p} last={i === PIPELINE.length - 1} />
        ))}
        <PipelineOut />
      </div>
      <div className="flex items-center gap-6 mt-6 pt-6 border-t" style={{ borderColor: 'var(--color-line)' }}>
        <LegendSwatch tone="base" label="OpenTSLM-Flamingo (the paper)" />
        <LegendSwatch tone="ours" label="What we added for molecular dynamics" />
      </div>
    </div>
  );
}

function PipelineStep({ step, title, sub, kind, last }:
  { step: string; title: string; sub: string; kind: 'ours' | 'base'; last?: boolean }) {
  const isOurs = kind === 'ours';
  return (
    <div className="flex items-center gap-3 shrink-0">
      <div
        className="relative w-[180px] h-[88px] rounded-xl flex flex-col items-center justify-center text-center px-3"
        style={{
          background: isOurs ? 'var(--color-brand-tint)' : 'var(--color-bg-soft)',
          border: isOurs ? '2px solid var(--color-brand)' : '1.5px solid var(--color-line)',
        }}
      >
        {isOurs && (
          <span
            className="absolute -top-3 left-1/2 -translate-x-1/2 px-2.5 py-0.5 rounded-full text-[9.5px] font-mono font-bold tracking-[0.14em] uppercase"
            style={{ background: 'var(--color-brand)', color: '#fff' }}
          >
            + ours
          </span>
        )}
        <span className="stamp" style={{ color: 'var(--color-ink-dim)' }}>{step}</span>
        <div className="font-display font-semibold text-[15px] mt-1 tracking-tight"
             style={{ color: isOurs ? 'var(--color-ink)' : 'var(--color-ink-2)' }}>
          {title}
        </div>
        <div className="text-[11px] font-mono mt-0.5"
             style={{ color: 'var(--color-ink-mute)' }}>
          {sub}
        </div>
      </div>
      {!last && (
        <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden>
          <path d="M2 10 L16 10 M11 5 L16 10 L11 15" stroke="var(--color-mol-stroke)"
                strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      )}
    </div>
  );
}

function PipelineOut() {
  return (
    <div className="flex items-center gap-3 shrink-0">
      <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden>
        <path d="M2 10 L16 10 M11 5 L16 10 L11 15" stroke="var(--color-mol-stroke)"
              strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <div
        className="w-[140px] h-[64px] rounded-full flex flex-col items-center justify-center"
        style={{
          background: 'var(--color-brand)',
          color: '#fff',
          boxShadow: '0 8px 20px -8px rgba(84, 103, 242, 0.55)',
        }}
      >
        <div className="font-display font-semibold text-[15px] tracking-tight">binding ΔG</div>
        <div className="text-[11px] font-mono mt-0.5" style={{ color: '#dfe4ff' }}>kcal/mol</div>
      </div>
    </div>
  );
}

function LegendSwatch({ tone, label }: { tone: 'ours' | 'base'; label: string }) {
  return (
    <div className="flex items-center gap-2.5 text-[13px]"
         style={{ color: 'var(--color-ink-mute)' }}>
      <span
        className="block w-6 h-4 rounded-[4px]"
        style={
          tone === 'ours'
            ? { background: 'var(--color-brand-tint)', border: '2px solid var(--color-brand)' }
            : { background: 'var(--color-bg-soft)',    border: '1.5px solid var(--color-line)' }
        }
      />
      {label}
    </div>
  );
}

function SpecimenList({ refId, eyebrow, title, tone, items }:
  { refId: string; eyebrow: string; title: string; tone: 'brand' | 'warn'; items: string[] }) {
  const c = tone === 'brand' ? 'var(--color-brand)' : 'var(--color-warn)';
  return (
    <div className="panel ticks p-7 relative">
      <span className="tick-tr" />
      <span className="tick-bl" />
      <div className="flex items-start justify-between mb-4">
        <span className="eyebrow" style={{ color: c }}>{eyebrow}</span>
        <span className="stamp">{refId}</span>
      </div>
      <h3 className="font-display font-bold tracking-[-0.025em] text-[26px] leading-[1.05] mb-6"
          style={{ color: 'var(--color-ink)' }}>
        {title}
      </h3>
      <ol className="space-y-0">
        {items.map((it, i) => (
          <li key={i}
              className="flex items-start gap-4 py-3.5"
              style={{
                borderTop: i === 0 ? '1.5px solid var(--color-line)' : 'none',
                borderBottom: '1.5px solid var(--color-line)',
              }}>
            <span className="font-mono text-[11px] tracking-[0.18em] pt-1 shrink-0 w-6"
                  style={{ color: c }}>
              {String(i + 1).padStart(2, '0')}
            </span>
            <span className="text-[14.5px] leading-[1.55]"
                  style={{ color: 'var(--color-ink-2)' }}>
              {it}
            </span>
          </li>
        ))}
      </ol>
    </div>
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
