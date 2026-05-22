import { useEffect, useRef } from 'react';
import * as $3Dmol from '3dmol';


interface StructureViewerProps {
  pdbString: string | null;
  currentFrame: number;
  highlightResidue?: number | null;
  className?: string;
}


// Renders protein cartoon + ligand sticks + optional highlighted residue.
// Multi-MODEL PDB strings drive frame-by-frame animation via viewer.setFrame().
export function StructureViewer({ pdbString, currentFrame, highlightResidue, className }: StructureViewerProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<ReturnType<typeof $3Dmol.createViewer> | null>(null);
  const lastPdbRef = useRef<string | null>(null);

  // Init / reload model when pdbString changes.
  useEffect(() => {
    if (!hostRef.current) return;
    if (pdbString == null) return;
    if (pdbString === lastPdbRef.current && viewerRef.current) return;

    // Tear down any prior viewer to avoid WebGL leaks.
    if (viewerRef.current) {
      try { viewerRef.current.removeAllModels(); } catch { /* ignore */ }
    }

    const viewer = $3Dmol.createViewer(hostRef.current, {
      backgroundColor: '#1a1b26',
      antialias: true,
    });

    try {
      // addModelsAsFrames consumes a multi-MODEL block and exposes setFrame().
      viewer.addModelsAsFrames(pdbString, 'pdb');
      viewer.setStyle({ chain: 'A' }, { cartoon: { color: 'spectrum' } });
      viewer.setStyle({ chain: 'L' }, { stick: { colorscheme: 'orangeCarbon', radius: 0.18 } });
      if (highlightResidue != null) {
        viewer.addStyle(
          { chain: 'A', resi: highlightResidue },
          { stick: { colorscheme: 'yellowCarbon', radius: 0.2 } },
        );
      }
      viewer.zoomTo();
      viewer.render();
      viewerRef.current = viewer;
      lastPdbRef.current = pdbString;
    } catch (e) {
      // 3Dmol parse failure → fall back to a placeholder message.
      console.error('3Dmol parse failed', e);
      if (hostRef.current) {
        hostRef.current.innerHTML =
          '<div style="color:#94a3b8;font-family:monospace;font-size:11px;padding:1rem;text-align:center;">' +
          '3Dmol could not parse this PDB. Falling back to static frame.' +
          '</div>';
      }
    }

    return () => {
      if (viewerRef.current) {
        try { viewerRef.current.removeAllModels(); } catch { /* ignore */ }
      }
    };
  }, [pdbString, highlightResidue]);

  // Frame change: cheap setFrame + render, no rebuild.
  useEffect(() => {
    const v = viewerRef.current;
    if (!v) return;
    try {
      v.setFrame(currentFrame);
      v.render();
    } catch {
      /* viewer not ready */
    }
  }, [currentFrame]);

  return <div ref={hostRef} className={className} style={{ position: 'relative', width: '100%', height: '100%' }} />;
}
