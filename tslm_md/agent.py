"""TSLM-MD agent loop.

Orchestrates the trained model + deterministic rationale + independent
physics verifier, with an abstention rule that triggers INCONCLUSIVE when
the model and physics disagree (or when confidence is low).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import h5py
import torch

from tslm_md.featurize import featurize, normalise
from tslm_md.prompts import build_prompts
from tslm_md.parse import parse_answer
from tslm_md.rationale import deterministic_rationale, channel_summary_dict
from tslm_md.verifier import VerifierStats, combined_independent_energy

DEFAULT_TAU_HIGH = 1.5  # |z(affinity) - z(independent)| above this -> INCONCLUSIVE


@dataclass
class Report:
    pdb_id: str
    affinity: Optional[float]
    confidence: Optional[str]
    independent_energy: Optional[float]
    disagreement_z: Optional[float]
    rationale: str
    channel_summary: dict
    verdict: str          # "CONFIRMED" | "INCONCLUSIVE" | "PARSE_FAILED"
    verdict_reason: str
    raw_pred: str

    def to_dict(self) -> dict:
        return asdict(self)


def run_agent(
    pdb_id: str,
    misato_h5: h5py.File,
    model,                                            # OpenTSLMFlamingo instance
    stats: Optional[VerifierStats] = None,
    feature_stats_mean: Optional[torch.Tensor] = None,
    feature_stats_std: Optional[torch.Tensor] = None,
    tau_high: float = DEFAULT_TAU_HIGH,
    max_new_tokens: int = 50,
) -> Report:
    """Run the full TSLM-MD agent loop for a single PDB id.

    Args:
        pdb_id: key into the open misato_h5 file
        misato_h5: an opened h5py.File on MISATO MD.hdf5
        model: an instantiated OpenTSLMFlamingo (with trained adapter loaded)
        stats: VerifierStats for z-scoring. If None, no abstention threshold check.
        feature_stats_{mean,std}: optional [6, 1] tensors for normalisation
        tau_high: abstention threshold in absolute z-score units
        max_new_tokens: generation length cap
    """
    if pdb_id not in misato_h5:
        raise KeyError(f"PDB id {pdb_id!r} not found in MISATO HDF5")
    group = misato_h5[pdb_id]

    # 1) featurize
    feats = featurize(group)
    feats_norm = normalise(feats, mean=feature_stats_mean, std=feature_stats_std)

    # 2) call the trained TSLM
    pre_prompt, post_prompt = build_prompts(pdb_id)
    batch_item = {
        "time_series": feats_norm,
        "time_series_text": ["MD trajectory features per frame"],
        "pre_prompt": pre_prompt,
        "post_prompt": post_prompt,
        "answer": "",   # not used at generation time
    }
    pred_texts = model.generate([batch_item], max_new_tokens=max_new_tokens)
    raw_pred = pred_texts[0] if pred_texts else ""
    affinity, confidence = parse_answer(raw_pred)

    # 3) deterministic rationale (always available, never hallucinated)
    rationale = deterministic_rationale(feats)
    channel_summary = channel_summary_dict(feats)

    # 4) independent verifier
    independent = combined_independent_energy(group)

    # 5) decide verdict
    if affinity is None:
        verdict = "PARSE_FAILED"
        reason = "could not parse 'Answer: <x> kcal/mol. Confidence: <y>.' from LM output"
        disagreement = None
    elif stats is None or independent is None:
        verdict = "CONFIRMED"
        reason = "no verifier signal available — accepting model prediction"
        disagreement = None
    else:
        disagreement = stats.disagreement(affinity, independent)
        if confidence == "low":
            verdict = "INCONCLUSIVE"
            reason = "model reported low confidence — escalating"
        elif disagreement > tau_high:
            verdict = "INCONCLUSIVE"
            reason = (
                f"disagreement |z(model) - z(physics)| = {disagreement:.2f} > tau_high = {tau_high:.2f}"
            )
        else:
            verdict = "CONFIRMED"
            reason = (
                f"agreement within threshold (|delta_z| = {disagreement:.2f} <= {tau_high:.2f})"
            )

    return Report(
        pdb_id=pdb_id,
        affinity=affinity,
        confidence=confidence,
        independent_energy=independent,
        disagreement_z=disagreement,
        rationale=rationale,
        channel_summary=channel_summary,
        verdict=verdict,
        verdict_reason=reason,
        raw_pred=raw_pred,
    )
