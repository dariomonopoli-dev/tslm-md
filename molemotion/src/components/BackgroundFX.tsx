// Editorial-scientific ambient background. Hairline grid + one big architectural
// cobalt arc + a single slow brand-tint blob. No dots, no maximalist blobs.
export function BackgroundFX() {
  return (
    <div
      aria-hidden
      className="fixed inset-0 -z-10 overflow-hidden pointer-events-none"
      style={{ background: 'var(--color-bg)' }}
    >
      {/* cobalt sky-wash up top */}
      <div
        className="absolute inset-x-0 top-0 h-[58vh]"
        style={{
          background:
            'linear-gradient(180deg, rgba(84, 103, 242, 0.09) 0%, rgba(84, 103, 242, 0.025) 35%, transparent 100%)',
        }}
      />

      {/* horizontal baseline-rules — scientific paper feel */}
      <div className="absolute inset-0 bg-baseline-rules opacity-100" />

      {/* one large architectural cobalt arc, top-right corner */}
      <svg
        className="bg-arc"
        viewBox="0 0 720 720"
        fill="none"
      >
        <defs>
          <linearGradient id="arc-fade" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%"  stopColor="#5467F2" stopOpacity="0.55" />
            <stop offset="60%" stopColor="#5467F2" stopOpacity="0.12" />
            <stop offset="100%" stopColor="#5467F2" stopOpacity="0" />
          </linearGradient>
        </defs>
        <circle cx="500" cy="220" r="380" stroke="url(#arc-fade)" strokeWidth="1.5" />
        <circle cx="500" cy="220" r="280" stroke="url(#arc-fade)" strokeWidth="1.5" strokeDasharray="3 6" opacity="0.6" />
        <circle cx="500" cy="220" r="180" stroke="url(#arc-fade)" strokeWidth="1.5" opacity="0.4" />
      </svg>

      {/* a single VERY slow drifting blob, bottom-left, mostly to add depth */}
      <div
        className="bg-blob"
        style={{
          width: 520,
          height: 520,
          bottom: '-18%',
          left: '-8%',
          background:
            'radial-gradient(circle, rgba(78, 192, 122, 0.10), transparent 65%)',
          animation: 'drift 40s ease-in-out infinite',
        }}
      />
    </div>
  );
}
