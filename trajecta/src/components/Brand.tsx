interface BrandProps {
  size?: number;
  showTagline?: boolean;
  onClick?: () => void;
}

// Editorial molecule mark: dashed pocket, cobalt core, two satellite atoms,
// and a slow-traced orbital arc. Pure SVG/CSS so it ships in <2kb and has no
// runtime cost on Vercel.
export function Brand({ size = 40, showTagline = true, onClick }: BrandProps) {
  return (
    <div
      onClick={onClick}
      className="flex items-center gap-2.5 sm:gap-3 cursor-pointer select-none group"
    >
      <MoleculeMark size={size} />
      <div className="flex flex-col leading-none">
        <span
          className="font-display font-bold tracking-[-0.025em] text-[18px] sm:text-[20px] transition-colors"
          style={{ color: 'var(--color-brand)' }}
        >
          Trajecta
        </span>
        {showTagline && (
          <span
            className="hidden sm:inline-block text-[10px] font-mono tracking-[0.18em] uppercase mt-1.5"
            style={{ color: 'var(--color-ink-dim)' }}
          >
            motion · audited
          </span>
        )}
      </div>
    </div>
  );
}

function MoleculeMark({ size }: { size: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      aria-hidden
      className="shrink-0 brand-mark"
    >
      <defs>
        <radialGradient id="brand-core" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#7c8cf7" />
          <stop offset="65%" stopColor="#5467F2" />
          <stop offset="100%" stopColor="#3a4cd6" />
        </radialGradient>
        <linearGradient id="brand-bond" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#5467F2" stopOpacity="0.85" />
          <stop offset="100%" stopColor="#9aa0b8" stopOpacity="0.45" />
        </linearGradient>
      </defs>

      {/* outer orbital ring — traced */}
      <circle
        cx="32"
        cy="32"
        r="29"
        fill="none"
        stroke="var(--color-brand)"
        strokeOpacity="0.35"
        strokeWidth="1.2"
        strokeDasharray="3 6"
        className="brand-orbit"
      />

      {/* binding-pocket ellipse — dashed */}
      <ellipse
        cx="32"
        cy="33"
        rx="22"
        ry="20"
        className="mol-pocket"
      />

      {/* bonds */}
      <g className="mol-lig brand-lig">
        <line x1="32" y1="32" x2="18" y2="20" stroke="url(#brand-bond)" strokeWidth="2.5" strokeLinecap="round" />
        <line x1="32" y1="32" x2="48" y2="22" stroke="url(#brand-bond)" strokeWidth="2.5" strokeLinecap="round" />
        <line x1="32" y1="32" x2="40" y2="48" stroke="url(#brand-bond)" strokeWidth="2.5" strokeLinecap="round" />

        {/* central atom — cobalt core with subtle highlight */}
        <circle cx="32" cy="32" r="8" fill="url(#brand-core)" />
        <circle cx="29.5" cy="29.5" r="2.2" fill="#ffffff" opacity="0.55" />

        {/* satellite atoms — outline */}
        <circle cx="18" cy="20" r="4.5" fill="#ffffff" stroke="var(--color-mol-stroke)" strokeWidth="2" />
        <circle cx="48" cy="22" r="4.5" fill="#ffffff" stroke="var(--color-mol-stroke)" strokeWidth="2" />
        <circle cx="40" cy="48" r="3.5" fill="#ffffff" stroke="var(--color-mol-stroke)" strokeWidth="2" />
      </g>
    </svg>
  );
}
