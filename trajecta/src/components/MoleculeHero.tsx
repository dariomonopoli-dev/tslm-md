import { useEffect, useRef, useState } from 'react';
import * as $3Dmol from '3dmol';

import { api } from '../lib/api.ts';

interface MoleculeHeroProps {
  pdb: string;
  className?: string;
  // When provided, also cycles the server-rendered frame image as a fallback
  // / cinematic film-strip behind the 3D viewer.
  filmstrip?: boolean;
}

// Cinematic auto-rotating protein hero. Loads a multi-MODEL PDB, plays it as
// an MD movie, and slowly spins the camera at the same time. Falls back to a
// frame-image film strip if 3Dmol can't parse or HDF5 isn't mounted.
export function MoleculeHero({ pdb, className, filmstrip = false }: MoleculeHeroProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<ReturnType<typeof $3Dmol.createViewer> | null>(null);
  const rafRef = useRef<number | null>(null);
  const frameRef = useRef(0);
  const [parsed, setParsed] = useState<boolean | null>(null);
  const [pdbString, setPdbString] = useState<string | null>(null);
  const [filmFrame, setFilmFrame] = useState(0);

  // Fetch PDB string.
  useEffect(() => {
    let cancelled = false;
    setParsed(null);
    setPdbString(null);
    (async () => {
      const res = await api.pdbString(pdb);
      if (cancelled) return;
      if (res.ok) setPdbString(res.data);
      else setParsed(false);
    })();
    return () => { cancelled = true; };
  }, [pdb]);

  // Boot 3Dmol viewer + auto-rotate + auto-advance frame loop.
  useEffect(() => {
    if (!hostRef.current || !pdbString) return;

    if (viewerRef.current) {
      try { viewerRef.current.removeAllModels(); } catch { /* ignore */ }
    }

    const viewer = $3Dmol.createViewer(hostRef.current, {
      backgroundColor: 'rgba(0,0,0,0)',
      antialias: true,
    });

    try {
      viewer.addModelsAsFrames(pdbString, 'pdb');
      viewer.setStyle({ chain: 'A' }, { cartoon: { color: 'spectrum', thickness: 0.4 } });
      viewer.setStyle({ chain: 'L' }, { stick: { colorscheme: 'cyanCarbon', radius: 0.2 } });
      viewer.zoomTo();
      viewer.zoom(1.05);
      viewer.render();
      viewerRef.current = viewer;
      setParsed(true);

      // Smooth combined animation: rotate camera + step frame.
      let last = performance.now();
      const tick = (t: number) => {
        const dt = t - last;
        last = t;
        try {
          viewer.rotate(dt * 0.012, 'y');
          if (Math.floor(t / 80) !== Math.floor((t - dt) / 80)) {
            frameRef.current = (frameRef.current + 1) % 50;
            viewer.setFrame(frameRef.current);
          }
          viewer.render();
        } catch {
          /* viewer torn down */
        }
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    } catch {
      setParsed(false);
    }

    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      if (viewerRef.current) {
        try { viewerRef.current.removeAllModels(); } catch { /* ignore */ }
        viewerRef.current = null;
      }
    };
  }, [pdbString]);

  // Filmstrip cycler — runs whenever 3Dmol failed OR explicitly requested.
  useEffect(() => {
    if (!filmstrip && parsed !== false) return;
    const id = window.setInterval(() => {
      setFilmFrame(f => (f + 1) % 100);
    }, 110);
    return () => window.clearInterval(id);
  }, [filmstrip, parsed]);

  const showFilmstrip = filmstrip || parsed === false;

  return (
    <div className={`viewer-frame rounded-2xl ${className ?? ''}`}>
      {/* corner brackets */}
      <span className="viewer-corner tl" />
      <span className="viewer-corner tr" />
      <span className="viewer-corner bl" />
      <span className="viewer-corner br" />

      {/* image filmstrip fallback (and optional overlay) */}
      {showFilmstrip && (
        <img
          key={`${pdb}-${filmFrame}`}
          src={`/api/frame_image/${encodeURIComponent(pdb)}?frame=${filmFrame}&width=900`}
          alt={`${pdb} frame ${filmFrame}`}
          className="absolute inset-0 w-full h-full object-contain opacity-90"
          onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = 'none'; }}
        />
      )}

      {/* live 3D viewer */}
      <div
        ref={hostRef}
        className="absolute inset-0"
        style={{ pointerEvents: 'none' }}
      />

      {/* HUD overlay — top-left readout */}
      <div className="absolute top-4 left-4 flex flex-col gap-1 z-10 font-mono text-[10px] tracking-wider"
           style={{ color: '#5467F2' }}>
        <span className="flex items-center gap-2">
          <span
            className="inline-block w-1.5 h-1.5 rounded-full"
            style={{ background: '#4ec07a', boxShadow: '0 0 6px #4ec07a' }}
          />
          MD-LIVE · {pdb}
        </span>
        <span style={{ color: '#9a9fb3' }}>MISATO TRAJECTORY · 10 NS</span>
      </div>

      {/* HUD overlay — bottom-right scale */}
      <div className="absolute bottom-4 right-4 z-10 font-mono text-[10px] tracking-wider"
           style={{ color: '#9a9fb3' }}>
        AUTO-ROTATING · 4 CHANNELS
      </div>

      {/* loading state */}
      {parsed === null && !showFilmstrip && (
        <div className="absolute inset-0 flex items-center justify-center text-[11px] font-mono tracking-wider"
             style={{ color: '#6a7088' }}>
          decoding trajectory…
        </div>
      )}
    </div>
  );
}
