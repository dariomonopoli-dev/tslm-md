"""Independent physics-based verifier.

Reads MISATO's `frames_*` per-frame energy components (EPtot / EELEC / EVDW),
averages across frames, and returns a scalar comparable to predicted affinity
via z-score against the train-set distribution.

CRITICAL property: the trained TSLM NEVER sees `frames_*` values during
training — only the featurised coordinates — so this signal is genuinely
INDEPENDENT. Disagreement is therefore meaningful, not a tautology.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

DEFAULT_COMPONENT = "EPtot"
ENERGY_COMPONENTS = ("EPtot", "EELEC", "EVDW")


def mean_frame_energy(
    h5_group: h5py.Group,
    component: str = DEFAULT_COMPONENT,
) -> Optional[float]:
    """Mean of `frames_<component>` across the trajectory. None if key missing."""
    key = f"frames_{component}"
    if key not in h5_group:
        return None
    return float(np.asarray(h5_group[key]).mean())


def combined_independent_energy(
    h5_group: h5py.Group,
    weights: dict | None = None,
) -> Optional[float]:
    """Optional combined verifier: weighted sum of multiple components.

    Default weights match the spec §6 R2 suggestion: 0.5 EPtot + 0.3 EELEC + 0.2 EVDW.
    """
    weights = weights or {"EPtot": 0.5, "EELEC": 0.3, "EVDW": 0.2}
    total = 0.0
    found_any = False
    for comp, w in weights.items():
        v = mean_frame_energy(h5_group, comp)
        if v is not None:
            total += w * v
            found_any = True
    return total if found_any else None


class VerifierStats:
    """Holds train-set z-score stats for affinity and independent-energy."""

    def __init__(
        self,
        affinity_mean: float,
        affinity_std: float,
        independent_mean: float,
        independent_std: float,
    ):
        self.affinity_mean = affinity_mean
        self.affinity_std = max(affinity_std, 1e-6)
        self.independent_mean = independent_mean
        self.independent_std = max(independent_std, 1e-6)

    @classmethod
    def from_file(cls, path: str | Path) -> "VerifierStats":
        with Path(path).open() as f:
            d = json.load(f)
        return cls(
            affinity_mean=d["affinity_mean"],
            affinity_std=d["affinity_std"],
            independent_mean=d["independent_mean"],
            independent_std=d["independent_std"],
        )

    def to_file(self, path: str | Path) -> None:
        with Path(path).open("w") as f:
            json.dump(
                {
                    "affinity_mean": self.affinity_mean,
                    "affinity_std": self.affinity_std,
                    "independent_mean": self.independent_mean,
                    "independent_std": self.independent_std,
                },
                f,
                indent=2,
            )

    def z_affinity(self, x: float) -> float:
        return (x - self.affinity_mean) / self.affinity_std

    def z_independent(self, x: float) -> float:
        return (x - self.independent_mean) / self.independent_std

    def disagreement(self, predicted_affinity: float, independent_energy: float) -> float:
        """Absolute z-score gap between predicted affinity and independent energy."""
        return abs(self.z_affinity(predicted_affinity) - self.z_independent(independent_energy))
