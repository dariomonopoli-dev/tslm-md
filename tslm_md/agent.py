"""The TSLM-MD agent loop.

Stub: implement during hour 8-14 in parallel with training.

Pseudocode:
    def agent(pdb_id: str, model, misato_h5, train_stats) -> Report:
        complex     = misato_h5[pdb_id]
        features    = featurize(complex)
        pred_text   = model.generate({...features, prompts...})
        affinity, confidence = parse_answer(pred_text)

        rationale   = deterministic_rationale(features)
        independent = verifier.mean_frame_energy(complex)
        disagreement = abs(z(affinity, train_stats) - z(independent, train_stats))

        verdict = "INCONCLUSIVE" if (disagreement > TAU_HIGH or confidence == "low") \
                                  else "CONFIRMED"
        return Report(pdb_id, affinity, confidence, independent, rationale, verdict, pred_text)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Report:
    pdb_id: str
    affinity: Optional[float]
    confidence: Optional[str]
    independent: Optional[float]
    rationale: str
    verdict: str       # "CONFIRMED" | "INCONCLUSIVE"
    raw_pred: str


# TODO(hour 8-14): implement agent()
