"""Prompt templates for TSLM-MD stage-6 fine-tuning + inference."""

from __future__ import annotations

PRE_PROMPT_TEMPLATE = """You are a computational chemist analysing a molecular dynamics trajectory of a protein-ligand complex.

The time-series carries six features computed per frame:
  ch0  minimum protein-pocket / ligand-atom distance (Å)
  ch1  mean pocket-ligand distance within 4 Å (Å)
  ch2  number of close contacts within 4 Å
  ch3  ligand RMSD from frame 0 after pocket alignment (Å)
  ch4  ligand radius of gyration (Å)
  ch5  interface buriedness proxy (count of buried ligand atoms)

Your task is to predict the binding affinity (PDBbind -logKd/Ki, converted to kcal/mol) for this complex.

Typical range: roughly -12 kcal/mol (very strong) to -2 kcal/mol (weak).

Signatures of stable binding:
  - low and flat ch0 (tight contact maintained)
  - low ch3 plateau (ligand not wandering)
  - saturated ch2 (contact network stable)

Signatures of weak / unbinding behaviour:
  - ch0 rising or fluctuating
  - ch3 drifting upward
  - ch2 dropping
"""

POST_PROMPT = """Output exactly one line in the following format and nothing else:
Answer: <x> kcal/mol. Confidence: <high|medium|low>."""


def build_prompts(pdb_id: str) -> tuple[str, str]:
    """Return (pre_prompt, post_prompt) for a given PDB id.

    The pre_prompt currently does not vary by PDB id, but we keep the
    signature ready so we can inject e.g. target-family or ligand-class
    context later without churning callers.
    """
    return PRE_PROMPT_TEMPLATE, POST_PROMPT
