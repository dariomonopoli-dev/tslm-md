"""Offline preprocessor: MD.hdf5 + affinity_data.csv -> SageMaker-ready artifacts.

Produces, per split:
  features_{split}.npz : channels (N,100,4) float32, pK (N,) float32,
                        pdb_ids (N,) str, dissociated (N,) bool,
                        unstable (N,) bool, multi_ligand (N,) bool
  samples_{split}.jsonl: one JSON line per PDB with pK, facts, rationale, flags

Plus once:
  norm_stats.json : per-channel train mean/std after physical clipping
  metadata.json   : channel order, clip bounds, counts, version

The training job ships only these artifacts (~80 MB total) to SageMaker.
The 124 GB MD.hdf5 is never opened on the training instance.

Reference: PROJECT_BRIEF.md sections 4, 6-7; DATASET.md sections 2-5, 9.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

CHANNEL_ORDER = ["rmsd_ligand", "interaction_energy", "distance", "bSASA"]
CHANNEL_KEYS = {
    "rmsd_ligand": "frames_rmsd_ligand",
    "interaction_energy": "frames_interaction_energy",
    "distance": "frames_distance",
    "bSASA": "frames_bSASA",
}
CLIP_BOUNDS = {
    "rmsd_ligand": (0.0, 50.0),
    "interaction_energy": (-500.0, 50.0),
    "distance": (0.0, 50.0),
    "bSASA": (0.0, 2500.0),
}
N_FRAMES = 100
POCKET_THRESHOLD_A = 20.0         # MISATO frames_distance is CoM-CoM; median bound ~13.5 A
ENERGY_SPIKE_KCAL = 30.0          # frame-to-frame IE jump above this = spike
DRIFT_WINDOW = 20                 # last-N frames used for dissociation check
DRIFT_RMSD_A = 5.0
DRIFT_DIST_A = 30.0
UNSTABLE_RMSD_A = 10.0
MULTI_LIGAND_PDB = "6CC9"
UNLABELLED_TRAIN_PDBS = {"4DGO", "4OTW", "4V1C", "5V8H", "5V8J", "6FIM", "6H7K"}


@dataclass
class SystemRecord:
    pdb_id: str
    split: str
    pK: float
    channels_raw: np.ndarray   # (100, 4) float32, pre-clip
    channels: np.ndarray       # (100, 4) float32, post-clip
    dissociated: bool
    unstable: bool
    multi_ligand: bool


def load_splits(splits_dir: Path) -> dict[str, str]:
    pdb_to_split: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = splits_dir / f"{split}_MD.txt"
        for line in path.read_text().splitlines():
            pdb = line.strip()
            if pdb:
                pdb_to_split[pdb] = split
    return pdb_to_split


def load_affinity(csv_path: Path) -> dict[str, float]:
    """Parse manually: protein-name column can contain unquoted semicolons.

    Columns: PDBid;Kd (nM);Ki (nM);IC50 (nM);type;ligand;Uniprot;Protein
    """
    pk: dict[str, float] = {}
    with csv_path.open() as f:
        header = f.readline()  # noqa: F841
        for line in f:
            parts = line.rstrip("\n").split(";", maxsplit=7)
            if len(parts) < 4:
                continue
            pdb = parts[0]
            for raw in parts[1:4]:
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    v = 0.0
                if v > 0:
                    pk[pdb] = 9.0 - math.log10(v)
                    break
    return pk


def read_system_channels(h5: h5py.File, pdb: str) -> np.ndarray | None:
    """Return (100, 4) float32 raw channels for one PDB, or None on shape mismatch."""
    group = h5[pdb]
    cols: list[np.ndarray] = []
    for name in CHANNEL_ORDER:
        arr = np.asarray(group[CHANNEL_KEYS[name]], dtype=np.float32)
        if pdb == MULTI_LIGAND_PDB and arr.shape == (400,):
            arr = arr[:N_FRAMES]
        if arr.shape != (N_FRAMES,):
            return None
        cols.append(arr)
    return np.stack(cols, axis=1)


def compute_tags(raw: np.ndarray) -> tuple[bool, bool, bool]:
    """Returns (dissociated, unstable, _) computed on UNCLIPPED arrays."""
    rmsd = raw[:, 0]
    dist = raw[:, 2]
    last = rmsd[-DRIFT_WINDOW:].mean()
    last_d = dist[-DRIFT_WINDOW:].mean()
    dissociated = bool(last > DRIFT_RMSD_A or last_d > DRIFT_DIST_A)
    unstable = bool(rmsd.max() > UNSTABLE_RMSD_A and not dissociated)
    return dissociated, unstable, False


def clip_channels(raw: np.ndarray) -> np.ndarray:
    out = raw.copy()
    for i, name in enumerate(CHANNEL_ORDER):
        lo, hi = CLIP_BOUNDS[name]
        out[:, i] = np.clip(out[:, i], lo, hi)
    return out


def extract_facts(rec: SystemRecord) -> dict:
    """Build a fact dict the verifier and rationale template can both read.

    Uses post-clip channels for summary stats (matches what the model sees),
    but tags from compute_tags() reflect the unclipped truth.
    """
    ch = rec.channels
    rmsd = ch[:, 0]
    ie = ch[:, 1]
    dist = ch[:, 2]
    bsasa = ch[:, 3]
    frames = np.arange(N_FRAMES, dtype=np.float32)

    def slope(y: np.ndarray) -> float:
        a, _ = np.polyfit(frames, y, 1)
        return float(a)

    summary = {
        "rmsd_mean": float(rmsd.mean()),
        "rmsd_std": float(rmsd.std()),
        "rmsd_max": float(rmsd.max()),
        "energy_mean": float(ie.mean()),
        "energy_std": float(ie.std()),
        "energy_range": float(ie.max() - ie.min()),
        "energy_slope": slope(ie),
        "distance_mean": float(dist.mean()),
        "distance_max": float(dist.max()),
        "pocket_residence_fraction": float((dist < POCKET_THRESHOLD_A).mean()),
        "bsasa_mean": float(bsasa.mean()),
        "bsasa_slope": slope(bsasa),
        "bsasa_max": float(bsasa.max()),
        "contacts_persistent": bool(bsasa.mean() > 200 and bsasa.min() > 50),
        "ligand_drift": bool(rec.dissociated or rmsd.max() - rmsd.min() > 3.0),
    }

    events: list[dict] = []
    ie_diffs = np.diff(ie)
    spike_idx = int(np.argmax(np.abs(ie_diffs)))
    if abs(ie_diffs[spike_idx]) > ENERGY_SPIKE_KCAL:
        events.append({
            "type": "energy_spike",
            "frame": spike_idx + 1,
            "from": float(ie[spike_idx]),
            "to": float(ie[spike_idx + 1]),
        })
    if rec.dissociated:
        events.append({
            "type": "ligand_drift",
            "frame_range": [N_FRAMES - DRIFT_WINDOW, N_FRAMES - 1],
            "delta_rmsd": float(rmsd[-DRIFT_WINDOW:].mean() - rmsd[:DRIFT_WINDOW].mean()),
        })
    bsasa_start = bsasa[:DRIFT_WINDOW].mean()
    bsasa_end = bsasa[-DRIFT_WINDOW:].mean()
    if bsasa_start - bsasa_end > 100.0:
        events.append({
            "type": "contact_drop",
            "frame_range": [N_FRAMES - DRIFT_WINDOW, N_FRAMES - 1],
            "from": float(bsasa_start),
            "to": float(bsasa_end),
        })

    return {"pdb_id": rec.pdb_id, "summary": summary, "events": events}


def render_rationale(facts: dict, pK: float) -> str:
    s = facts["summary"]
    parts = [
        f"Mean interaction energy was {s['energy_mean']:.1f} kcal/mol "
        f"with a swing of {s['energy_range']:.1f} kcal/mol.",
        f"Ligand RMSD averaged {s['rmsd_mean']:.2f} A (max {s['rmsd_max']:.2f}).",
        f"Buried SASA averaged {s['bsasa_mean']:.0f} A^2.",
        f"The ligand stayed within {POCKET_THRESHOLD_A:.0f} A of the pocket "
        f"for {100*s['pocket_residence_fraction']:.0f}% of frames.",
    ]
    for e in facts["events"]:
        if e["type"] == "energy_spike":
            parts.append(
                f"Interaction energy jumps from {e['from']:.1f} to {e['to']:.1f} "
                f"kcal/mol at frame {e['frame']}."
            )
        elif e["type"] == "ligand_drift":
            parts.append(
                f"Between frames {e['frame_range'][0]} and {e['frame_range'][1]} "
                f"the ligand drifts by {e['delta_rmsd']:.1f} A."
            )
        elif e["type"] == "contact_drop":
            parts.append(
                f"Buried SASA drops from {e['from']:.0f} to {e['to']:.0f} A^2 "
                f"between frames {e['frame_range'][0]} and {e['frame_range'][1]}."
            )
    if s["contacts_persistent"] and not s["ligand_drift"]:
        parts.append("The pose remains stable throughout the trajectory.")
    parts.append(f"Answer: {pK:.2f}")
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md-hdf5", type=Path, required=True)
    ap.add_argument("--affinity-csv", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--subset", type=int, default=None,
                    help="Optional: cap total systems for smoke testing.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pdb_to_split = load_splits(args.splits_dir)
    pk_lookup = load_affinity(args.affinity_csv)
    target_pdbs = sorted(p for p in pdb_to_split if p in pk_lookup and p not in UNLABELLED_TRAIN_PDBS)
    if args.subset:
        target_pdbs = target_pdbs[: args.subset]
    print(f"targets: {len(target_pdbs)} (after dropping unlabelled + UNLABELLED_TRAIN_PDBS)")

    records: list[SystemRecord] = []
    skipped: list[tuple[str, str]] = []

    with h5py.File(args.md_hdf5, "r") as h5:
        for pdb in tqdm(target_pdbs, desc="reading h5"):
            if pdb not in h5:
                skipped.append((pdb, "missing_in_h5"))
                continue
            raw = read_system_channels(h5, pdb)
            if raw is None:
                skipped.append((pdb, "shape_mismatch"))
                continue
            diss, unst, _ = compute_tags(raw)
            records.append(SystemRecord(
                pdb_id=pdb,
                split=pdb_to_split[pdb],
                pK=float(pk_lookup[pdb]),
                channels_raw=raw,
                channels=clip_channels(raw),
                dissociated=diss,
                unstable=unst,
                multi_ligand=(pdb == MULTI_LIGAND_PDB),
            ))

    print(f"kept: {len(records)}   skipped: {len(skipped)}")

    by_split: dict[str, list[SystemRecord]] = {"train": [], "val": [], "test": []}
    for r in records:
        by_split[r.split].append(r)
    for split, rs in by_split.items():
        print(f"  {split}: {len(rs)}")

    train_stack = np.stack([r.channels for r in by_split["train"]], axis=0) if by_split["train"] else np.zeros((0, N_FRAMES, 4), np.float32)
    norm_stats = {
        "channel_order": CHANNEL_ORDER,
        "clip_bounds": CLIP_BOUNDS,
        "train_mean": train_stack.reshape(-1, 4).mean(axis=0).tolist() if train_stack.size else [0.0] * 4,
        "train_std": (train_stack.reshape(-1, 4).std(axis=0) + 1e-6).tolist() if train_stack.size else [1.0] * 4,
        "n_train_systems": int(len(by_split["train"])),
    }
    (args.out_dir / "norm_stats.json").write_text(json.dumps(norm_stats, indent=2))
    print(f"train mean: {norm_stats['train_mean']}")
    print(f"train std:  {norm_stats['train_std']}")

    for split, rs in by_split.items():
        if not rs:
            print(f"skip {split} (empty)")
            continue
        channels = np.stack([r.channels for r in rs], axis=0).astype(np.float32)
        pK = np.array([r.pK for r in rs], dtype=np.float32)
        pdb_ids = np.array([r.pdb_id for r in rs])
        dissociated = np.array([r.dissociated for r in rs], dtype=bool)
        unstable = np.array([r.unstable for r in rs], dtype=bool)
        multi_ligand = np.array([r.multi_ligand for r in rs], dtype=bool)
        npz_path = args.out_dir / f"features_{split}.npz"
        np.savez_compressed(
            npz_path,
            channels=channels, pK=pK, pdb_ids=pdb_ids,
            dissociated=dissociated, unstable=unstable, multi_ligand=multi_ligand,
        )

        jsonl_path = args.out_dir / f"samples_{split}.jsonl"
        with jsonl_path.open("w") as f:
            for r in rs:
                facts = extract_facts(r)
                facts["dissociated"] = r.dissociated
                facts["unstable"] = r.unstable
                facts["multi_ligand"] = r.multi_ligand
                rationale = render_rationale(facts, r.pK)
                f.write(json.dumps({
                    "pdb_id": r.pdb_id,
                    "pK": r.pK,
                    "facts": facts,
                    "rationale": rationale,
                    "dissociated": r.dissociated,
                    "unstable": r.unstable,
                    "multi_ligand": r.multi_ligand,
                }) + "\n")
        print(f"wrote {npz_path.name} ({channels.nbytes/1e6:.1f} MB) and {jsonl_path.name}")

    metadata = {
        "version": "v1",
        "channel_order": CHANNEL_ORDER,
        "n_frames": N_FRAMES,
        "clip_bounds": CLIP_BOUNDS,
        "pocket_threshold_a": POCKET_THRESHOLD_A,
        "energy_spike_kcal": ENERGY_SPIKE_KCAL,
        "drift_window": DRIFT_WINDOW,
        "drift_rmsd_a": DRIFT_RMSD_A,
        "drift_dist_a": DRIFT_DIST_A,
        "unstable_rmsd_a": UNSTABLE_RMSD_A,
        "counts": {s: len(rs) for s, rs in by_split.items()},
        "skipped": skipped,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"wrote metadata.json")


if __name__ == "__main__":
    main()
