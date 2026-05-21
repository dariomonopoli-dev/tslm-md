"""Deterministic grounded rationale generator.

Given a [6, F] feature tensor, emit a paragraph of sentences whose every claim
references a number actually present in the trajectory. No LLM hallucination
surface.

Stub: implement during hour 8-12 in parallel with training.

Output template (one sentence per channel, plus a concluding line):
  "Min pocket distance tightened from {ch0[0]:.1f} A to {ch0[-1]:.1f} A and
   {plateau_phrase}. Ligand RMSD {plateau_phrase_ch3} at {ch3.mean():.1f} A.
   Contact count {trend_phrase_ch2}. Radius of gyration {trend_phrase_ch4}.
   {n_buried} ligand atoms buried (proxy SASA). These signals are
   {consistent_or_inconsistent} with a stable bound pose."
"""

from __future__ import annotations

import torch


def deterministic_rationale(feats: torch.Tensor) -> str:
    """Return a single-paragraph grounded summary of the feature trajectory."""
    # TODO(hour 8-12): implement per template above
    return ""
