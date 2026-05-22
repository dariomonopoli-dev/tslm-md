"""v2 preprocessor: expanded channels + Cheng-Prusoff label correction.

Changes from v1 (preprocess_misato.py):

  Channels (4 → 12):
    Kept:   rmsd_ligand, interaction_energy, distance, bSASA
    Added:  pocket_rmsd                 — binding-site protein flexibility
            ligand_rgyr                 — ligand compactness
            min_contact_distance        — closest protein-ligand atom
            n_contacts_4A               — interface density (any element)
            n_polar_contacts_35A        — H-bond count proxy (N/O-N/O within 3.5 Å)
            n_hydrophobic_contacts_45A  — hydrophobic contact count (C-C within 4.5 Å)
            ligand_internal_rmsd        — ligand internal flexibility (self-aligned)
            com_dist_velocity           — frame-to-frame ligand-protein CoM motion

  Labels (Cheng-Prusoff gold-standard correction, all 17K systems kept):
    Kd available    → pK = pKd                        (label_source = "Kd")
    Ki  (no Kd)     → pK = pKi                        (label_source = "Ki")
    IC50 only       → pK = pIC50 + 0.301              (label_source = "IC50_CP")
    (0.301 = log10(2); standard Cheng-Prusoff offset under [S] ≈ Km.)

  Rationale: new template covers the additional channels and reads pK from the
    corrected value. Existing "Answer: X.XX" suffix is preserved.

  Output: preprocessed_v2/
    features_{split}.npz   channels (N,100,12), pK (N,), pdb_ids (N,),
                           label_source (N,), dissociated/unstable/multi_ligand (N,)
    samples_{split}.jsonl
    norm_stats.json   (12-channel mean/std on train)
    metadata.json     (v2 with channel order + clip bounds + Cheng-Prusoff note)
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────── #
# Channel definitions                                                    #
# ────────────────────────────────────────────────────────────────────── #
ORIG_CHANNEL_KEYS = {
    "rmsd_ligand": "frames_rmsd_ligand",
    "interaction_energy": "frames_interaction_energy",
    "distance": "frames_distance",
    "bSASA": "frames_bSASA",
}
CHANNEL_ORDER = [
    "rmsd_ligand", "interaction_energy", "distance", "bSASA",
    "pocket_rmsd", "ligand_rgyr", "min_contact_distance",
    "n_contacts_4A", "n_polar_contacts_35A", "n_hydrophobic_contacts_45A",
    "ligand_internal_rmsd", "com_dist_velocity",
]
CLIP_BOUNDS = {
    "rmsd_ligand":                 (0.0,   50.0),
    "interaction_energy":          (-500.0, 50.0),
    "distance":                    (0.0,   50.0),
    "bSASA":                       (0.0,   2500.0),
    "pocket_rmsd":                 (0.0,   10.0),
    "ligand_rgyr":                 (0.0,   25.0),
    "min_contact_distance":        (0.0,   20.0),
    "n_contacts_4A":               (0.0,   1000.0),
    "n_polar_contacts_35A":        (0.0,   50.0),
    "n_hydrophobic_contacts_45A":  (0.0,   500.0),
    "ligand_internal_rmsd":        (0.0,   10.0),
    "com_dist_velocity":           (0.0,   5.0),
}
N_FRAMES = 100
N_CHANNELS = len(CHANNEL_ORDER)

# Pocket selection: any protein heavy atom within this radius of any ligand
# heavy atom in frame 0.
POCKET_RADIUS_A = 8.0

# Contact thresholds (Å)
CONTACT_R = 4.0       # generic close-contact
POLAR_R = 3.5         # H-bond proxy
HYDROPHOBIC_R = 4.5   # hydrophobic contact

# Element codes in atoms_element (decoded from MISATO data)
ELEM_C, ELEM_H, ELEM_N, ELEM_O = 1, 2, 3, 4
POLAR_ELEMS = {ELEM_N, ELEM_O}

# Existing v1 tags
POCKET_THRESHOLD_A = 20.0
DRIFT_WINDOW = 20
DRIFT_RMSD_A = 5.0
DRIFT_DIST_A = 30.0
UNSTABLE_RMSD_A = 10.0
MULTI_LIGAND_PDB = "6CC9"
UNLABELLED_TRAIN_PDBS = {"4DGO", "4OTW", "4V1C", "5V8H", "5V8J", "6FIM", "6H7K"}

# Cheng-Prusoff offset for IC50 -> Ki (under [S] ≈ Km)
CHENG_PRUSOFF_OFFSET = math.log10(2)  # ≈ 0.30103


# ────────────────────────────────────────────────────────────────────── #
# Label parsing                                                          #
# ────────────────────────────────────────────────────────────────────── #
@dataclass
class Label:
    pK: float
    source: str          # "Kd" | "Ki" | "IC50_CP"
    raw_nM: float        # raw measured value (after priority)
    raw_assay: str       # "Kd" | "Ki" | "IC50"


def load_affinity_v2(csv_path: Path) -> dict[str, Label]:
    """Returns gold-standard-corrected pK per PDB.

    Priority: Kd > Ki > IC50. For IC50-derived labels, applies Cheng-Prusoff
    offset (+log10(2)) under the [S] ≈ Km assumption.
    """
    out: dict[str, Label] = {}
    with csv_path.open() as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split(";", maxsplit=7)
            if len(parts) < 4:
                continue
            pdb = parts[0]
            kd, ki, ic50 = (
                _safe_float(parts[1]),
                _safe_float(parts[2]),
                _safe_float(parts[3]),
            )
            if kd > 0:
                out[pdb] = Label(
                    pK=9.0 - math.log10(kd),
                    source="Kd", raw_nM=kd, raw_assay="Kd",
                )
            elif ki > 0:
                out[pdb] = Label(
                    pK=9.0 - math.log10(ki),
                    source="Ki", raw_nM=ki, raw_assay="Ki",
                )
            elif ic50 > 0:
                pIC50 = 9.0 - math.log10(ic50)
                out[pdb] = Label(
                    pK=pIC50 + CHENG_PRUSOFF_OFFSET,
                    source="IC50_CP", raw_nM=ic50, raw_assay="IC50",
                )
    return out


def _safe_float(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


# ────────────────────────────────────────────────────────────────────── #
# Per-system feature extraction                                          #
# ────────────────────────────────────────────────────────────────────── #
@dataclass
class SystemRecord:
    pdb_id: str
    split: str
    label: Label
    channels_raw: np.ndarray   # (100, 12) float32 pre-clip
    channels: np.ndarray       # (100, 12) float32 post-clip
    dissociated: bool
    unstable: bool
    multi_ligand: bool
    facts_extra: dict = field(default_factory=dict)  # per-system summary stats


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """RMSD between two (N,3) atom sets after optimal rotation+translation."""
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    Pr = Pc @ R.T
    return float(np.sqrt(((Pr - Qc) ** 2).sum() / len(P)))


def radius_of_gyration(coords: np.ndarray) -> float:
    """Rg of a (N,3) point cloud."""
    com = coords.mean(0)
    return float(np.sqrt(((coords - com) ** 2).sum(1).mean()))


def extract_v2_channels(group: h5py.Group, pdb: str) -> tuple[np.ndarray, dict] | None:
    """Read original 4 channels + compute 8 new ones.

    Returns:
      (channels_raw: (100, 12) float32, extra_facts: dict)
      or None on shape mismatch / missing data.
    """
    # Original 4 channels from precomputed frames_* arrays
    orig_cols = []
    for name in ["rmsd_ligand", "interaction_energy", "distance", "bSASA"]:
        arr = np.asarray(group[ORIG_CHANNEL_KEYS[name]], dtype=np.float32)
        if pdb == MULTI_LIGAND_PDB and arr.shape == (400,):
            arr = arr[:N_FRAMES]
        if arr.shape != (N_FRAMES,):
            return None
        orig_cols.append(arr)

    # Atomic data for derived channels
    coords = np.asarray(group["trajectory_coordinates"], dtype=np.float32)  # (T, N, 3)
    if coords.shape[0] != N_FRAMES:
        # Some systems (multi-ligand) may have more frames concatenated;
        # take the first N_FRAMES so the rest of the pipeline matches.
        if coords.shape[0] > N_FRAMES:
            coords = coords[:N_FRAMES]
        else:
            return None

    elements = np.asarray(group["atoms_element"])
    mbi = np.asarray(group["molecules_begin_atom_index"])
    n_atoms = coords.shape[1]
    if elements.shape[0] != n_atoms or mbi.size == 0:
        return None

    # Ligand = atoms from mbi[-1] to end, protein = everything before.
    ligand_start = int(mbi[-1])
    if ligand_start <= 0 or ligand_start >= n_atoms:
        return None
    ligand_idx = np.arange(ligand_start, n_atoms)
    protein_idx = np.arange(0, ligand_start)

    # Heavy-atom masks (exclude hydrogens)
    lig_heavy_mask = elements[ligand_idx] != ELEM_H
    prot_heavy_mask = elements[protein_idx] != ELEM_H
    if lig_heavy_mask.sum() < 3 or prot_heavy_mask.sum() < 3:
        return None
    lig_heavy = ligand_idx[lig_heavy_mask]
    prot_heavy = protein_idx[prot_heavy_mask]

    # Pocket: protein heavy atoms within POCKET_RADIUS_A of any ligand heavy
    # atom in frame 0.
    lig_f0 = coords[0, lig_heavy]
    prot_f0 = coords[0, prot_heavy]
    tree = cKDTree(prot_f0)
    pairs = tree.query_ball_point(lig_f0, r=POCKET_RADIUS_A)
    pocket_local = sorted({i for sub in pairs for i in sub})
    if len(pocket_local) < 5:
        # Pocket too small — likely a low-quality/dissociated system.
        # Fall back to nearest 50 protein heavy atoms to the ligand CoM at f0.
        com0 = lig_f0.mean(0)
        d2 = ((prot_f0 - com0) ** 2).sum(1)
        pocket_local = list(np.argsort(d2)[:50])
    pocket_atoms = prot_heavy[pocket_local]              # global indices into N_atoms
    pocket_elem = elements[pocket_atoms]                 # per-pocket-atom element codes
    lig_elem = elements[lig_heavy]                       # per-ligand-heavy-atom element

    # Frame-by-frame derived channels
    pocket_rmsd     = np.zeros(N_FRAMES, dtype=np.float32)
    ligand_rgyr     = np.zeros(N_FRAMES, dtype=np.float32)
    min_contact     = np.zeros(N_FRAMES, dtype=np.float32)
    n_contacts_4    = np.zeros(N_FRAMES, dtype=np.float32)
    n_polar         = np.zeros(N_FRAMES, dtype=np.float32)
    n_hydrophobic   = np.zeros(N_FRAMES, dtype=np.float32)
    lig_internal    = np.zeros(N_FRAMES, dtype=np.float32)
    com_velocity    = np.zeros(N_FRAMES, dtype=np.float32)

    pocket0 = coords[0, pocket_atoms]
    lig0 = coords[0, lig_heavy]

    # Precompute element-pair masks (ligand × pocket)
    lig_is_C = (lig_elem == ELEM_C)
    pocket_is_C = (pocket_elem == ELEM_C)
    cc_mask = lig_is_C[:, None] & pocket_is_C[None, :]      # (Lh, Ph)

    lig_is_polar = np.isin(lig_elem, list(POLAR_ELEMS))
    pocket_is_polar = np.isin(pocket_elem, list(POLAR_ELEMS))
    polar_mask = lig_is_polar[:, None] & pocket_is_polar[None, :]

    prev_com_dist: float | None = None
    for t in range(N_FRAMES):
        pocket_t = coords[t, pocket_atoms]
        lig_t = coords[t, lig_heavy]

        # 1. pocket RMSD vs frame 0 (Kabsch-aligned)
        pocket_rmsd[t] = kabsch_rmsd(pocket_t, pocket0)

        # 2. ligand Rg
        ligand_rgyr[t] = radius_of_gyration(lig_t)

        # 3-6. pairwise distance matrix (Lh × Ph)
        d = np.linalg.norm(lig_t[:, None, :] - pocket_t[None, :, :], axis=-1)
        min_contact[t] = float(d.min())
        n_contacts_4[t] = float((d < CONTACT_R).sum())
        n_polar[t] = float(((d < POLAR_R) & polar_mask).sum())
        n_hydrophobic[t] = float(((d < HYDROPHOBIC_R) & cc_mask).sum())

        # 7. ligand internal RMSD (self-aligned)
        lig_internal[t] = kabsch_rmsd(lig_t, lig0)

        # 8. CoM velocity (|Δ ligand-pocket CoM distance|)
        lig_com = lig_t.mean(0)
        pocket_com = pocket_t.mean(0)
        com_dist = float(np.linalg.norm(lig_com - pocket_com))
        if prev_com_dist is None:
            com_velocity[t] = 0.0
        else:
            com_velocity[t] = abs(com_dist - prev_com_dist)
        prev_com_dist = com_dist

    new_cols = [
        pocket_rmsd, ligand_rgyr, min_contact,
        n_contacts_4, n_polar, n_hydrophobic,
        lig_internal, com_velocity,
    ]
    all_cols = orig_cols + new_cols
    raw = np.stack(all_cols, axis=1).astype(np.float32)   # (100, 12)

    extra_facts = {
        "n_protein_heavy": int(prot_heavy.size),
        "n_ligand_heavy": int(lig_heavy.size),
        "n_pocket_atoms": int(pocket_atoms.size),
    }
    return raw, extra_facts


def compute_tags(raw: np.ndarray) -> tuple[bool, bool, bool]:
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


# ────────────────────────────────────────────────────────────────────── #
# Rationale template (v2)                                                 #
# ────────────────────────────────────────────────────────────────────── #
def extract_facts(rec: SystemRecord) -> dict:
    ch = rec.channels
    rmsd, ie, dist, bsasa = ch[:, 0], ch[:, 1], ch[:, 2], ch[:, 3]
    pocket_rmsd, lig_rgyr, min_contact = ch[:, 4], ch[:, 5], ch[:, 6]
    n_contacts4, n_polar, n_hphobic = ch[:, 7], ch[:, 8], ch[:, 9]
    lig_internal, com_vel = ch[:, 10], ch[:, 11]

    return {
        "pdb_id": rec.pdb_id,
        "label_source": rec.label.source,
        "summary": {
            "rmsd_mean": float(rmsd.mean()),
            "rmsd_max": float(rmsd.max()),
            "energy_mean": float(ie.mean()),
            "energy_range": float(ie.max() - ie.min()),
            "distance_mean": float(dist.mean()),
            "pocket_residence_fraction": float((dist < POCKET_THRESHOLD_A).mean()),
            "bsasa_mean": float(bsasa.mean()),
            "pocket_rmsd_mean": float(pocket_rmsd.mean()),
            "pocket_rmsd_max": float(pocket_rmsd.max()),
            "ligand_rgyr_mean": float(lig_rgyr.mean()),
            "min_contact_mean": float(min_contact.mean()),
            "n_contacts_4A_mean": float(n_contacts4.mean()),
            "n_polar_contacts_mean": float(n_polar.mean()),
            "n_hydrophobic_contacts_mean": float(n_hphobic.mean()),
            "ligand_internal_rmsd_mean": float(lig_internal.mean()),
            "com_velocity_mean": float(com_vel.mean()),
            "contacts_persistent": bool(bsasa.mean() > 200 and bsasa.min() > 50),
            "ligand_drift": bool(rec.dissociated or rmsd.max() - rmsd.min() > 3.0),
        },
    }


def render_rationale(facts: dict, pK: float) -> str:
    s = facts["summary"]
    parts = [
        f"Ligand RMSD averaged {s['rmsd_mean']:.2f} A (max {s['rmsd_max']:.2f}).",
        f"Mean interaction energy was {s['energy_mean']:.1f} kcal/mol "
        f"with a swing of {s['energy_range']:.1f} kcal/mol.",
        f"Buried SASA averaged {s['bsasa_mean']:.0f} A^2.",
        f"Binding-site flexibility (pocket RMSD) averaged "
        f"{s['pocket_rmsd_mean']:.2f} A (max {s['pocket_rmsd_max']:.2f}).",
        f"Ligand internal RMSD (after self-alignment) averaged "
        f"{s['ligand_internal_rmsd_mean']:.2f} A.",
        f"Closest protein-ligand atomic distance averaged "
        f"{s['min_contact_mean']:.2f} A across the trajectory.",
        f"Hydrogen-bond contacts averaged {s['n_polar_contacts_mean']:.1f} per frame; "
        f"hydrophobic contacts averaged {s['n_hydrophobic_contacts_mean']:.1f} per frame.",
        f"Total close protein-ligand contacts (< 4 A) averaged "
        f"{s['n_contacts_4A_mean']:.0f} per frame.",
        f"The ligand stayed within {POCKET_THRESHOLD_A:.0f} A of the pocket "
        f"for {100 * s['pocket_residence_fraction']:.0f}% of frames.",
    ]
    if s["contacts_persistent"] and not s["ligand_drift"]:
        parts.append("The pose remains stable throughout the trajectory.")
    parts.append(f"Answer: {pK:.2f}")
    return " ".join(parts)


# ────────────────────────────────────────────────────────────────────── #
# Driver                                                                  #
# ────────────────────────────────────────────────────────────────────── #
def load_splits(splits_dir: Path) -> dict[str, str]:
    pdb_to_split: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for line in (splits_dir / f"{split}_MD.txt").read_text().splitlines():
            pdb = line.strip()
            if pdb:
                pdb_to_split[pdb] = split
    return pdb_to_split


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
    labels = load_affinity_v2(args.affinity_csv)
    target_pdbs = sorted(
        p for p in pdb_to_split
        if p in labels and p not in UNLABELLED_TRAIN_PDBS
    )
    if args.subset:
        target_pdbs = target_pdbs[: args.subset]
    print(f"targets: {len(target_pdbs):,} (after drop unlabelled + UNLABELLED_TRAIN_PDBS)")
    src_counts = {"Kd": 0, "Ki": 0, "IC50_CP": 0}
    for p in target_pdbs:
        src_counts[labels[p].source] += 1
    print(f"label sources: {src_counts}")

    records: list[SystemRecord] = []
    skipped: list[tuple[str, str]] = []

    with h5py.File(args.md_hdf5, "r") as h5:
        for pdb in tqdm(target_pdbs, desc="reading h5"):
            if pdb not in h5:
                skipped.append((pdb, "missing_in_h5"))
                continue
            try:
                result = extract_v2_channels(h5[pdb], pdb)
            except Exception as e:
                skipped.append((pdb, f"extract_error:{type(e).__name__}"))
                continue
            if result is None:
                skipped.append((pdb, "shape_or_geometry_mismatch"))
                continue
            raw, extra = result
            diss, unst, _ = compute_tags(raw)
            records.append(SystemRecord(
                pdb_id=pdb,
                split=pdb_to_split[pdb],
                label=labels[pdb],
                channels_raw=raw,
                channels=clip_channels(raw),
                dissociated=diss,
                unstable=unst,
                multi_ligand=(pdb == MULTI_LIGAND_PDB),
                facts_extra=extra,
            ))

    print(f"kept: {len(records):,}   skipped: {len(skipped):,}")
    by_split: dict[str, list[SystemRecord]] = {"train": [], "val": [], "test": []}
    for r in records:
        by_split[r.split].append(r)
    for split, rs in by_split.items():
        print(f"  {split}: {len(rs):,}")

    # Normalization stats from train
    if by_split["train"]:
        train_stack = np.stack([r.channels for r in by_split["train"]], axis=0)
    else:
        train_stack = np.zeros((0, N_FRAMES, N_CHANNELS), np.float32)
    norm_stats = {
        "channel_order": CHANNEL_ORDER,
        "clip_bounds": CLIP_BOUNDS,
        "train_mean": (train_stack.reshape(-1, N_CHANNELS).mean(0).tolist()
                       if train_stack.size else [0.0] * N_CHANNELS),
        "train_std": ((train_stack.reshape(-1, N_CHANNELS).std(0) + 1e-6).tolist()
                      if train_stack.size else [1.0] * N_CHANNELS),
        "n_train_systems": int(len(by_split["train"])),
    }
    (args.out_dir / "norm_stats.json").write_text(json.dumps(norm_stats, indent=2))

    for split, rs in by_split.items():
        if not rs:
            continue
        channels = np.stack([r.channels for r in rs], axis=0).astype(np.float32)
        pK = np.array([r.label.pK for r in rs], dtype=np.float32)
        pdb_ids = np.array([r.pdb_id for r in rs])
        label_source = np.array([r.label.source for r in rs])
        dissociated = np.array([r.dissociated for r in rs], dtype=bool)
        unstable = np.array([r.unstable for r in rs], dtype=bool)
        multi_ligand = np.array([r.multi_ligand for r in rs], dtype=bool)
        np.savez_compressed(
            args.out_dir / f"features_{split}.npz",
            channels=channels, pK=pK, pdb_ids=pdb_ids,
            label_source=label_source,
            dissociated=dissociated, unstable=unstable, multi_ligand=multi_ligand,
        )

        with (args.out_dir / f"samples_{split}.jsonl").open("w") as f:
            for r in rs:
                facts = extract_facts(r)
                facts["dissociated"] = r.dissociated
                facts["unstable"] = r.unstable
                facts["multi_ligand"] = r.multi_ligand
                facts.update(r.facts_extra)
                rationale = render_rationale(facts, r.label.pK)
                f.write(json.dumps({
                    "pdb_id": r.pdb_id,
                    "pK": r.label.pK,
                    "label_source": r.label.source,
                    "raw_assay": r.label.raw_assay,
                    "raw_nM": r.label.raw_nM,
                    "facts": facts,
                    "rationale": rationale,
                    "dissociated": r.dissociated,
                    "unstable": r.unstable,
                    "multi_ligand": r.multi_ligand,
                }) + "\n")
        print(f"wrote features_{split}.npz ({channels.nbytes / 1e6:.1f} MB) "
              f"and samples_{split}.jsonl")

    metadata = {
        "version": "v2",
        "channel_order": CHANNEL_ORDER,
        "n_channels": N_CHANNELS,
        "n_frames": N_FRAMES,
        "clip_bounds": CLIP_BOUNDS,
        "pocket_radius_a": POCKET_RADIUS_A,
        "contact_thresholds_a": {
            "close": CONTACT_R, "polar": POLAR_R, "hydrophobic": HYDROPHOBIC_R,
        },
        "label_correction": {
            "rule": "Cheng-Prusoff under [S]≈Km",
            "offset_pK_ic50": CHENG_PRUSOFF_OFFSET,
            "priority": "Kd > Ki > IC50(corrected)",
        },
        "counts": {s: len(rs) for s, rs in by_split.items()},
        "label_source_counts_train": _src_counts(by_split["train"]),
        "label_source_counts_val": _src_counts(by_split["val"]),
        "label_source_counts_test": _src_counts(by_split["test"]),
        "skipped": skipped,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("wrote metadata.json")


def _src_counts(records: list[SystemRecord]) -> dict[str, int]:
    out = {"Kd": 0, "Ki": 0, "IC50_CP": 0}
    for r in records:
        out[r.label.source] += 1
    return out


if __name__ == "__main__":
    main()
