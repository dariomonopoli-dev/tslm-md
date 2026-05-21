"""Build "Answer: <x> kcal/mol. Confidence: <y>." target strings per PDB id.

Inputs:
  - PDBbind v2020 index file (--pdbbind-index), pointing at INDEX_general_PL_data.2020
    (or refined/core variants). Columns: pdb_id resolution year -logKd/Ki Kd/Ki ref ligand_name
  - Optional refined+core lists for confidence-tier assignment.

Outputs:
  - data/targets.json  : {pdb_id: {"answer": "...", "affinity_kcal_mol": float, "confidence": "high|medium|low"}}
  - data/splits/{train,val,test}.txt  if --build-splits

Confidence:
  - "high"   = PDBbind 2020 CORE set (PDBbind-CN's most-trusted ~290 complexes)
  - "medium" = REFINED set (~5000)
  - "low"    = the rest of GENERAL set
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Set


def neglog_kx_to_kcal_per_mol(pK: float) -> float:
    """-logKd/Ki -> binding free energy in kcal/mol at 298 K.

    dG = -RT ln K = -RT (-pK * ln 10) = RT * pK * ln 10
    At T=298 K: RT = 0.59248 kcal/mol; ln 10 = 2.302585
    So dG = -0.59248 * 2.302585 * pK = -1.3642 * pK
    """
    return -1.3642 * pK


def read_index_file(path: Path) -> dict[str, float]:
    """Return {pdb_id: pK_value} from a PDBbind INDEX file."""
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            pid, _res, _yr, pK = parts[0], parts[1], parts[2], parts[3]
            try:
                out[pid.lower()] = float(pK)
            except ValueError:
                continue
    return out


def read_id_list(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {line.strip().lower() for line in f if line.strip() and not line.startswith("#")}


def main(args: argparse.Namespace) -> None:
    general = read_index_file(Path(args.pdbbind_index))
    refined_ids = read_id_list(Path(args.refined_ids)) if args.refined_ids else set()
    core_ids = read_id_list(Path(args.core_ids)) if args.core_ids else set()

    print(f"general: {len(general)} | refined: {len(refined_ids)} | core: {len(core_ids)}")

    def tier(pid: str) -> str:
        if pid in core_ids:
            return "high"
        if pid in refined_ids:
            return "medium"
        return "low"

    targets: dict[str, dict] = {}
    for pid, pK in general.items():
        dG = neglog_kx_to_kcal_per_mol(pK)
        conf = tier(pid)
        answer = f"Answer: {dG:.2f} kcal/mol. Confidence: {conf}."
        targets[pid] = {
            "answer": answer,
            "affinity_kcal_mol": round(dG, 4),
            "pK": pK,
            "confidence": conf,
        }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(targets, f, indent=2)
    print(f"wrote {len(targets)} targets to {out_path}")

    # Spot-check 5 samples
    for pid in list(targets.keys())[:5]:
        print(f"  {pid}: {targets[pid]['answer']}")

    if args.build_splits:
        # Random 80/10/10 split keyed by pdb id (deterministic via seed)
        ids = sorted(targets.keys())
        rng = random.Random(args.seed)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        train_ids = ids[:n_train]
        val_ids = ids[n_train : n_train + n_val]
        test_ids = ids[n_train + n_val :]
        splits_dir = Path(args.splits_dir)
        splits_dir.mkdir(parents=True, exist_ok=True)
        for name, lst in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            with (splits_dir / f"{name}.txt").open("w") as f:
                f.write("\n".join(lst))
            print(f"  split {name}: {len(lst)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pdbbind-index", required=True,
                   help="path to e.g. INDEX_general_PL_data.2020")
    p.add_argument("--refined-ids", default=None,
                   help="optional path to a list of refined-set PDB ids (one per line)")
    p.add_argument("--core-ids", default=None,
                   help="optional path to a list of CASF/core-set PDB ids")
    p.add_argument("--out-json", default="data/targets.json")
    p.add_argument("--build-splits", action="store_true",
                   help="also write train/val/test splits to --splits-dir")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
