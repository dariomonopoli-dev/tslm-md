"""Evaluation metrics for TSLM-MD.

Primary metric: Pearson r between predicted affinity and PDBbind ground truth
on held-out PDB ids.

Secondary: abstention precision/recall — given the verdict, did we abstain on
the worst-quartile of true errors?

Stub: implement during hour 10-14 in parallel with training.
"""

from __future__ import annotations

# TODO(hour 10-14): implement eval()
