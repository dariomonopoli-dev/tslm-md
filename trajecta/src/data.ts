import { TriageRow, FailureModeRow, FailurePattern } from './types.ts';

export const ALPHABET_PDBS = [
  '1A1B', '1A28', '1A30', '1B6H', '1F0R', '1F0S', '1G2K', '1KE5',
  '1NHU', '1QPE', '2BR1', '2X3K'
];

export const BATCH_DATA: TriageRow[] = [
  { pdb: '1A28', pred: 8.91, delta: 0.12, regex: '4/4 ✓✓✓✓', recommendation: 'trust', evidence: 'vina ✓, hbond Lys145 94%' },
  { pdb: '1F0R', pred: 7.84, delta: 0.21, regex: '4/4 ✓✓✓✓', recommendation: 'trust', evidence: 'vina ✓, lit [PDBbind_..]' },
  { pdb: '1A1B', pred: 6.42, delta: 0.11, regex: '3/4 ✓✓✓✗', recommendation: 'trust', evidence: 'vina ✓, 1 contradiction' },
  { pdb: '1B6H', pred: 7.21, delta: 0.59, regex: '4/4 ✓✓✓✓', recommendation: 'review', evidence: 'vina disagrees by 1.4' },
  { pdb: '1G2K', pred: 5.40, delta: 0.31, regex: '2/4 ✓✗✓✗', recommendation: 'review', evidence: 'pose unstable (3 clust.)' },
  { pdb: '2X3K', pred: 7.50, delta: 1.50, regex: '4/4 ✓✓✓✓', recommendation: 'discard', evidence: 'LE implausible, no lit.' },
  { pdb: '1QPE', pred: 4.80, delta: 0.42, regex: '3/4 ✓✓✗✓', recommendation: 'trust', evidence: 'weak binder, agent agrees' },
  { pdb: '1A30', pred: 6.10, delta: 0.15, regex: '4/4 ✓✓✓✓', recommendation: 'trust', evidence: 'vina ✓, stable trajectory' },
  { pdb: '1F0S', pred: 8.12, delta: 0.30, regex: '3/4 ✓✓✗✓', recommendation: 'trust', evidence: 'hbond network preserved' },
  { pdb: '1KE5', pred: 5.90, delta: 0.88, regex: '4/4 ✓✓✓✓', recommendation: 'review', evidence: 'ligand strains > 3kcal/mol' },
  { pdb: '1NHU', pred: 7.45, delta: 0.19, regex: '4/4 ✓✓✓✓', recommendation: 'trust', evidence: 'vina ✓, high consensus' },
  { pdb: '2BR1', pred: 6.80, delta: 0.25, regex: '3/4 ✓✗✓✓', recommendation: 'trust', evidence: 'minor fluctuation tolerated' },
];

export const FAILURE_MODES_DATA: FailureModeRow[] = [
  { pdb: '2X3K', model: 7.5, vina: 5.1, mmgbsa: 5.3, agent: 'discard', reason: 'Ligand efficiency 0.71 — implausible for a 320-Da ligand. No known binder in this LE band [PubMed:..].' },
  { pdb: '4HHB', model: 8.2, vina: 6.7, mmgbsa: 6.5, agent: 'review', reason: 'Interaction-energy channel dominated by 1 outlier frm; clash_check flagged frm 73.' },
  { pdb: '1RPE', model: 7.0, vina: 5.2, mmgbsa: 5.4, agent: 'discard', reason: 'Pose splits into 3 clusters of equal size — not stable; rationale missed this.' },
  { pdb: '1B6H', model: 7.2, vina: 5.8, mmgbsa: 5.9, agent: 'review', reason: 'Agent physically contradicts model; pose flips halfway through MD.' },
  { pdb: '1F8B', model: 6.9, vina: 4.5, mmgbsa: 4.8, agent: 'discard', reason: 'Unstable trajectory. Rationalizer hallucinated stability despite RMSD > 4.5.' },
];

export const FAILURE_PATTERNS_DATA: FailurePattern[] = [
  { cluster: 'Implausible ligand efficiency', count: 3, systems: '2X3K, 1ZZB, 2P2N' },
  { cluster: 'Pose unstable (>2 clusters)', count: 4, systems: '1RPE, 1ABE, 1B6H, 1F8B' },
  { cluster: 'Single-frame outlier dominates', count: 2, systems: '4HHB, 1MZK' },
  { cluster: 'Literature contradicts binding mode', count: 1, systems: '2X3K' },
];

export const MOCK_CHART_DATA = Array.from({ length: 99 }).map((_, i) => ({
  frame: i,
  rmsd: 1.5 + Math.sin(i / 10) * 0.5 + (i > 50 ? Math.random() * 0.4 : 0),
  energy: -35 + Math.cos(i / 15) * 5 - (i > 20 ? 8 : 0) + Math.random() * 2,
  dist: 2.8 + Math.cos(i / 8) * 0.2 + Math.random() * 0.1,
  bsasa: 550 + Math.sin(i / 20) * 20 + Math.random() * 10
}));
