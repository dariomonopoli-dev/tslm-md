"""Build "Answer: <x> kcal/mol. Confidence: <y>." target strings per PDB id.

Two input formats supported:

(A) --kierandidi-csv data/misato_affinity/affinity_data.csv  (RECOMMENDED)
    Columns: PDBid;Kd (nM);Ki (nM);IC50 (nM);type;ligand;Uniprot;Protein
    Confidence tier inferred from data quality (Kd > Ki > IC50 priority).

(B) --pdbbind-index INDEX_general_PL_data.2020
    Columns: pdb_id resolution year -logKd/Ki Kd/Ki ref ligand_name
    Confidence from PDBbind tier (core/refined/general).

Outputs:
  - data/targets.json  : {pdb_id: {"answer": "...", "affinity_kcal_mol": float, "confidence": "high|medium|low"}}
  - data/splits/{train,val,test}.txt  if --build-splits

Confidence (kierandidi mode):
  - "high"   = Kd present (most direct binding measurement)
  - "medium" = Ki present (inhibition constant, well-correlated with Kd)
  - "low"    = IC50 only (assay-dependent, noisier proxy)
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


def read_kierandidi_csv(path: Path) -> dict[str, dict]:
    """Parse data/misato_affinity/affinity_data.csv.

    Columns: PDBid;Kd (nM);Ki (nM);IC50 (nM);type;ligand;Uniprot;Protein
    Returns {pdb_id_lower: {"value_nM": float, "kind": "Kd"|"Ki"|"IC50", "ligand": str, "protein": str}}
    Skips rows where Kd, Ki, and IC50 are all 0/missing.
    """
    import math
    out: dict[str, dict] = {}
    with path.open() as f:
        header = f.readline()  # skip column header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 4:
                continue
            try:
                pid, kd_s, ki_s, ic50_s = parts[0], parts[1], parts[2], parts[3]
                kd = float(kd_s) if kd_s else 0.0
                ki = float(ki_s) if ki_s else 0.0
                ic50 = float(ic50_s) if ic50_s else 0.0
            except ValueError:
                continue
            ligand = parts[5] if len(parts) > 5 else ""
            protein = parts[7] if len(parts) > 7 else ""
            # Priority: Kd > Ki > IC50; tier from which measurement is available
            if kd > 0:
                value, kind, conf = kd, "Kd", "high"
            elif ki > 0:
                value, kind, conf = ki, "Ki", "medium"
            elif ic50 > 0:
                value, kind, conf = ic50, "IC50", "low"
            else:
                continue
            # Convert nM -> M -> pK = -log10(M)
            try:
                pK = -math.log10(value * 1e-9)
            except (ValueError, OverflowError):
                continue
            out[pid.lower()] = {
                "pK": pK,
                "value_nM": value,
                "kind": kind,
                "confidence": conf,
                "ligand": ligand,
                "protein": protein,
            }
    return out


def main(args: argparse.Namespace) -> None:
    targets: dict[str, dict] = {}

    if args.kierandidi_csv:
        rows = read_kierandidi_csv(Path(args.kierandidi_csv))
        print(f"kierandidi CSV: {len(rows)} valid rows")
        for pid, row in rows.items():
            dG = neglog_kx_to_kcal_per_mol(row["pK"])
            answer = f"Answer: {dG:.2f} kcal/mol. Confidence: {row['confidence']}."
            targets[pid] = {
                "answer": answer,
                "affinity_kcal_mol": round(dG, 4),
                "pK": round(row["pK"], 4),
                "kind": row["kind"],
                "confidence": row["confidence"],
                "ligand": row["ligand"],
                "protein": row["protein"],
            }
    else:
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
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--kierandidi-csv", default=None,
                     help="path to data/misato_affinity/affinity_data.csv (recommended)")
    src.add_argument("--pdbbind-index", default=None,
                     help="path to e.g. INDEX_general_PL_data.2020")
    p.add_argument("--refined-ids", default=None,
                   help="optional path to a list of refined-set PDB ids (pdbbind mode only)")
    p.add_argument("--core-ids", default=None,
                   help="optional path to a list of CASF/core-set PDB ids (pdbbind mode only)")
    p.add_argument("--out-json", default="data/targets.json")
    p.add_argument("--build-splits", action="store_true",
                   help="also write train/val/test splits to --splits-dir (random 80/10/10)")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
