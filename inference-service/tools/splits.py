"""Split metadata + ground-truth lookup tools — task #15.

Both tools share the test-split whitelist file, so this module is the
single source for split membership across the service (also used by
inference.predict()'s whitelist check).
"""

from __future__ import annotations

import csv
import math
import os
from functools import lru_cache
from pathlib import Path

from . import register


# --------------------------------------------------------------------------
# Split membership
# --------------------------------------------------------------------------


def _read_text_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _read_npz_pdb_ids(path: Path) -> list[str]:
    import numpy as np
    arr = np.load(path, allow_pickle=True)["pdb_ids"]
    return [str(p) for p in arr]


@lru_cache(maxsize=1)
def _splits() -> dict[str, str]:
    """Build pdb_id → split ('train'|'val'|'test') from whichever source exists.

    Priority order per split:
      1. $X_SPLIT_PATH env var (text file, one pdb per line)
      2. /app/data/{split}_MD.txt (text)
      3. /app/data/preprocessed/features_{split}.npz (pdb_ids array)
      4. preprocessed/features_{split}.npz (dev fallback)
    """
    candidates = {
        "train": [
            (Path(os.getenv("TRAIN_SPLIT_PATH", "/app/data/train_MD.txt")), _read_text_split),
            (Path("/app/data/preprocessed/features_train.npz"), _read_npz_pdb_ids),
            (Path("preprocessed/features_train.npz"), _read_npz_pdb_ids),
            (Path("../preprocessed/features_train.npz"), _read_npz_pdb_ids),
        ],
        "val": [
            (Path(os.getenv("VAL_SPLIT_PATH", "/app/data/val_MD.txt")), _read_text_split),
            (Path("/app/data/preprocessed/features_val.npz"), _read_npz_pdb_ids),
            (Path("preprocessed/features_val.npz"), _read_npz_pdb_ids),
            (Path("../preprocessed/features_val.npz"), _read_npz_pdb_ids),
        ],
        "test": [
            (Path(os.getenv("TEST_SPLIT_PATH", "/app/data/test_MD.txt")), _read_text_split),
            (Path("/app/data/preprocessed/features_test.npz"), _read_npz_pdb_ids),
            (Path("preprocessed/features_test.npz"), _read_npz_pdb_ids),
            (Path("../preprocessed/features_test.npz"), _read_npz_pdb_ids),
        ],
    }
    out: dict[str, str] = {}
    for split, options in candidates.items():
        for path, reader in options:
            if path.exists():
                for pid in reader(path):
                    out[pid] = split
                break
    return out


@register({
    "name": "lookup_split",
    "description": (
        "Return whether this PDB is in train / val / test of the MISATO MD "
        "split. Always call this first — predictions on training data are "
        "uninformative."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"pdb_id": {"type": "string"}},
        "required": ["pdb_id"],
    },
})
def lookup_split(pdb_id: str) -> dict:
    s = _splits()
    return {
        "pdb_id": pdb_id,
        "split": s.get(pdb_id, "unknown"),
        "is_test": s.get(pdb_id) == "test",
    }


# --------------------------------------------------------------------------
# Ground-truth pK lookup
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _affinity_pK() -> dict[str, float]:
    """pdb_id → pK from misato-affinity/data/affinity_data.csv.

    Uses the same priority as the training preprocessor: Kd > Ki > IC50.
    pK = 9 − log10(Kd|Ki|IC50 nM).
    """
    candidates = [
        Path(os.getenv("AFFINITY_CSV_PATH", "/app/data/affinity_data.csv")),
        Path("misato-affinity/data/affinity_data.csv"),
        Path("../misato-affinity/data/affinity_data.csv"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if not path:
        return {}

    out: dict[str, float] = {}
    with path.open() as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            pid = (row.get("PDBid") or "").strip()
            if not pid:
                continue
            for col in ("Kd (nM)", "Ki (nM)", "IC50 (nM)"):
                val = (row.get(col) or "").strip()
                if val and val not in ("nan", "NA"):
                    try:
                        nM = float(val)
                        if nM > 0:
                            out[pid] = 9.0 - math.log10(nM)
                            break
                    except ValueError:
                        continue
    return out


@register({
    "name": "actual_pK_lookup",
    "description": (
        "Return the experimental pK for this PDB if it is in the test split. "
        "Returns the literal string '[redacted]' on train/val — do not use "
        "low prediction error on non-test splits as a signal."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"pdb_id": {"type": "string"}},
        "required": ["pdb_id"],
    },
})
def actual_pK_lookup(pdb_id: str) -> dict:
    split = _splits().get(pdb_id, "unknown")
    if split != "test":
        return {"pdb_id": pdb_id, "split": split, "actual_pK": "[redacted]"}
    pK = _affinity_pK().get(pdb_id)
    return {"pdb_id": pdb_id, "split": split, "actual_pK": pK}
