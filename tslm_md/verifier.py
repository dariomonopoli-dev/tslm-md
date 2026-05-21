"""Independent physics-based verifier.

Reads MISATO's `frames_*` per-frame energy components (EPtot / EELEC / EVDW),
averages across frames, returns a scalar comparable to predicted affinity via
z-score against train-set distribution.

Crucially: the trained TSLM NEVER sees `frames_*` values at training time —
only the featurised coordinates — so this signal is independent.

Stub: implement during hour 8-10 (in parallel with training).
"""

from __future__ import annotations

from typing import Optional

import h5py
import numpy as np


def mean_frame_energy(
    h5_group: h5py.Group,
    component: str = "EPtot",
) -> Optional[float]:
    """Mean of `frames_<component>` across the trajectory.

    Returns None if the key is missing.
    """
    key = f"frames_{component}"
    if key not in h5_group:
        return None
    return float(np.asarray(h5_group[key]).mean())


# TODO(hour 8-10): add z-score helper using train-set stats
