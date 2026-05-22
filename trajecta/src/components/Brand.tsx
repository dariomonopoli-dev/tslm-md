interface BrandProps {
  size?: number;
  showTagline?: boolean;
  onClick?: () => void;
}

// Schematic molecule motif matching the pitch deck: dashed circular pocket,
// central ligand sphere, two bonded satellite atoms — gently wobbling.
export function Brand({ size = 36, showTagline = true, onClick }: BrandProps) {
  return (
    <div
      onClick={onClick}
      className="flex items-center gap-3 cursor-pointer select-none group"
    >
      <MoleculeGlyph size={size} />
      <div className="flex flex-col leading-none">
        <span
          className="font-display font-bold tracking-[-0.02em] text-[20px]"
          style={{ color: 'var(--color-brand)' }}
        >
          Trajecta
        </span>
        {showTagline && (
          <span
            className="text-[10px] font-mono tracking-[0.18em] uppercase mt-1.5"
            style={{ color: 'var(--color-ink-dim)' }}
          >
            motion · audited
          </span>
        )}
      </div>
    </div>
  );
}

function MoleculeGlyph({ size }: { size: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      aria-hidden
      className="shrink-0"
    >
      {/* dashed pocket — fingerprint of the binding cavity */}
      <ellipse
        cx="32"
        cy="32"
        rx="28"
        ry="28"
        className="mol-pocket"
      />
      <g className="mol-lig">
        {/* bonds */}
        <line x1="32" y1="32" x2="20" y2="20" className="mol-bond" />
        <line x1="32" y1="32" x2="46" y2="22" className="mol-bond" />
        {/* central atom — cobalt */}
        <circle cx="32" cy="32" r="8" fill="var(--color-brand)" />
        {/* satellite atoms — outline */}
        <circle cx="20" cy="20" r="5" fill="none" stroke="var(--color-mol-stroke)" strokeWidth="2.5" />
        <circle cx="46" cy="22" r="5" fill="none" stroke="var(--color-mol-stroke)" strokeWidth="2.5" />
      </g>
    </svg>
  );
}
