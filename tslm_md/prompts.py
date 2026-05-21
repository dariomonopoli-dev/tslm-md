"""Prompt templates for TSLM-MD stage-6 fine-tuning + inference.

The Chronos-2 encoder is univariate per call, so each of our 6 channels
becomes its OWN media chunk in the prompt. The LM sees 6 named chunks of
time-series tokens interleaved with their text descriptors, then is asked
for the affinity.
"""

from __future__ import annotations

PRE_PROMPT_TEMPLATE = """You are a computational chemist analysing a molecular dynamics trajectory of a protein-ligand complex.

You will be given six per-frame time series (each ~30 subsampled frames) describing different aspects of the binding interface.

Your task is to predict the binding affinity (PDBbind -logKd/Ki, converted to kcal/mol) for this complex.

Typical range: roughly -12 kcal/mol (very strong) to -2 kcal/mol (weak).

Signatures of stable binding:
  - low and flat min pocket-ligand distance (tight contact maintained)
  - low and flat ligand RMSD plateau (ligand not wandering)
  - saturated close-contact count (contact network stable)

Signatures of weak / unbinding behaviour:
  - rising or fluctuating min pocket-ligand distance
  - drifting upward ligand RMSD
  - dropping close-contact count"""

POST_PROMPT = """Output exactly one line in the following format and nothing else:
Answer: <x> kcal/mol. Confidence: <high|medium|low>."""

# One descriptor per channel — used as the natural-language anchor next to each
# Chronos-encoded chunk in the multi-modal prompt. Order MUST match featurize.py.
CHANNEL_DESCRIPTIONS = [
    "Minimum protein-pocket / ligand-atom distance per frame, in Angstroms",
    "Mean pocket-ligand distance under a 4 A mask per frame, in Angstroms",
    "Number of close contacts within 4 A per frame",
    "Ligand RMSD from the first frame after pocket alignment, per frame, in Angstroms",
    "Ligand radius of gyration per frame, in Angstroms",
    "Interface buriedness proxy: count of ligand atoms with at most two protein neighbours within 5 A",
]


def build_prompts(pdb_id: str) -> tuple[str, str]:
    """Return (pre_prompt, post_prompt) for a given PDB id."""
    return PRE_PROMPT_TEMPLATE, POST_PROMPT


def channel_descriptors() -> list[str]:
    """The six text labels that flank the six Chronos-encoded chunks, in order."""
    return list(CHANNEL_DESCRIPTIONS)
