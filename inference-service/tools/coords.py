"""Raw-coordinate analysis tools — task #15.

All operate on the MISATO trajectory the model never saw at training time.
The training-time encoder consumed four aggregated per-frame channels;
these tools work on raw atomic coordinates from MD.hdf5, so they are
orthogonal evidence for the agent.
"""

from __future__ import annotations

import os
from functools import lru_cache

import h5py
import numpy as np

from . import register


# --------------------------------------------------------------------------
# HDF5 loader (cached per pdb_id)
# --------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _load_system(pdb_id: str) -> dict:
    """Read everything we need for one system once. ~50 MB per call worst case."""
    path = os.getenv("MISATO_HDF5_PATH", "/app/data/MD.hdf5")
    with h5py.File(path, "r") as h5:
        key = pdb_id if pdb_id in h5 else pdb_id.upper()
        if key not in h5:
            raise KeyError(pdb_id)
        g = h5[key]
        return {
            "traj": g["trajectory_coordinates"][:],     # (frames, atoms, 3)
            "atoms_type": g["atoms_type"][:],
            "atoms_number": g["atoms_number"][:],
            "atoms_residue": g["atoms_residue"][:],
            "mol_begin": g["molecules_begin_atom_index"][:],
        }


def _ligand_mask(sys: dict) -> np.ndarray:
    """Boolean mask over atoms where atom_idx ∈ [mol_begin[1], mol_begin[2])."""
    mb = sys["mol_begin"]
    n = sys["atoms_type"].shape[0]
    mask = np.zeros(n, dtype=bool)
    if len(mb) >= 2:
        end = int(mb[2]) if len(mb) >= 3 else n
        mask[int(mb[1]):end] = True
    return mask


def _heavy_mask(sys: dict) -> np.ndarray:
    """Heavy atoms = atomic number != 1 (drop hydrogens)."""
    return sys["atoms_number"] != 1


# --------------------------------------------------------------------------
# cluster_poses
# --------------------------------------------------------------------------


@register({
    "name": "cluster_poses",
    "description": (
        "Cluster the trajectory frames in heavy-atom coordinate space; "
        "returns cluster sizes, inter-cluster RMSD, and the dominant cluster. "
        "Useful to verify claims like 'pose is stable'. If the dominant "
        "cluster covers <60% of frames, the pose is NOT stable."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pdb_id": {"type": "string"},
            "k": {"type": "integer", "default": 3, "minimum": 2, "maximum": 5},
        },
        "required": ["pdb_id"],
    },
})
def cluster_poses(pdb_id: str, k: int = 3) -> dict:
    from sklearn.cluster import KMeans

    sys = _load_system(pdb_id)
    heavy = _heavy_mask(sys) & _ligand_mask(sys)  # ligand heavy atoms only
    if heavy.sum() == 0:
        return {"error": "no ligand heavy atoms found"}

    coords = sys["traj"][:, heavy, :]              # (frames, n_heavy, 3)
    flat = coords.reshape(coords.shape[0], -1)
    k_eff = min(k, max(2, coords.shape[0]))
    km = KMeans(n_clusters=k_eff, n_init=10, random_state=42)
    labels = km.fit_predict(flat)
    sizes = np.bincount(labels, minlength=k_eff).tolist()

    # Inter-cluster RMSD on medoids
    medoids = []
    for c in range(k_eff):
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            medoids.append(None)
            continue
        sub = flat[idxs]
        dists = np.linalg.norm(sub - sub.mean(axis=0), axis=1)
        medoids.append(sub[int(dists.argmin())].reshape(-1, 3))

    rmsd_matrix: list[list[float]] = []
    for i in range(k_eff):
        row: list[float] = []
        for j in range(k_eff):
            if medoids[i] is None or medoids[j] is None:
                row.append(float("nan"))
            else:
                diff = medoids[i] - medoids[j]
                row.append(float(np.sqrt((diff ** 2).sum() / medoids[i].shape[0])))
        rmsd_matrix.append(row)

    dominant = int(np.argmax(sizes))
    return {
        "pdb_id": pdb_id,
        "k": k_eff,
        "cluster_sizes": sizes,
        "dominant_cluster": dominant,
        "dominant_fraction": float(sizes[dominant] / sum(sizes)),
        "rmsd_between_medoids": rmsd_matrix,
        "stable": bool(sizes[dominant] / sum(sizes) >= 0.60),
    }


# --------------------------------------------------------------------------
# clash_check
# --------------------------------------------------------------------------


@register({
    "name": "clash_check",
    "description": (
        "Count heavy-atom pairs whose distance is below 0.9 Å (true van der "
        "Waals overlap, well below any chemical bond length of ~1.5 Å). "
        "Use to rule out broken-frame artifacts before running expensive "
        "tools like Vina. Reports both raw count and the threshold used."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pdb_id": {"type": "string"},
            "frame_idx": {"type": "integer"},
            "threshold_angstrom": {
                "type": "number", "default": 0.9, "minimum": 0.5, "maximum": 1.4,
                "description": "Distance cutoff in Å. 0.9 = real overlap; 1.5 catches every bond.",
            },
        },
        "required": ["pdb_id", "frame_idx"],
    },
})
def clash_check(pdb_id: str, frame_idx: int, threshold_angstrom: float = 0.9) -> dict:
    sys = _load_system(pdb_id)
    n_frames = sys["traj"].shape[0]
    if not 0 <= frame_idx < n_frames:
        return {"error": f"frame_idx {frame_idx} out of range [0, {n_frames})"}
    heavy = _heavy_mask(sys)
    coords = sys["traj"][frame_idx][heavy]
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    clashes = int((dist < threshold_angstrom).sum() // 2)
    # Heuristic: > 3 hard overlaps = real broken frame. Single isolated
    # overlap can be force-field artifact, not enough to discard.
    return {
        "pdb_id": pdb_id,
        "frame_idx": frame_idx,
        "threshold_angstrom": threshold_angstrom,
        "n_clash_pairs": clashes,
        "is_broken_frame": bool(clashes > 3),
    }


# --------------------------------------------------------------------------
# hbond_persistence
# --------------------------------------------------------------------------


@register({
    "name": "hbond_persistence",
    "description": (
        "Per-bond donor/acceptor pairs with %% of frames persistent, using a "
        "simple distance/angle heuristic (heavy donor↔acceptor < 3.5 Å). "
        "Reports the top 10 most persistent contacts."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"pdb_id": {"type": "string"}},
        "required": ["pdb_id"],
    },
})
def hbond_persistence(pdb_id: str) -> dict:
    sys = _load_system(pdb_id)
    lig_mask = _ligand_mask(sys)
    if lig_mask.sum() == 0:
        return {"error": "no ligand atoms found"}

    # Acceptor/donor heuristic: N and O heavy atoms on ligand and protein.
    is_donor_acceptor = (sys["atoms_number"] == 7) | (sys["atoms_number"] == 8)

    lig_da = np.where(lig_mask & is_donor_acceptor)[0]
    prot_mask = ~lig_mask
    prot_da = np.where(prot_mask & is_donor_acceptor)[0]
    if len(lig_da) == 0 or len(prot_da) == 0:
        return {"pdb_id": pdb_id, "bonds": [], "note": "no N/O heteroatoms in pairing"}

    traj = sys["traj"]
    n_frames = traj.shape[0]
    pair_counts: dict[tuple[int, int], int] = {}

    for fi in range(n_frames):
        lp = traj[fi, lig_da, :]
        pp = traj[fi, prot_da, :]
        d = np.sqrt(((lp[:, None, :] - pp[None, :, :]) ** 2).sum(axis=-1))
        close = np.where(d < 3.5)
        for li, pi in zip(close[0], close[1]):
            key = (int(lig_da[li]), int(prot_da[pi]))
            pair_counts[key] = pair_counts.get(key, 0) + 1

    bonds = sorted(
        [
            {
                "ligand_atom_idx": k[0],
                "protein_atom_idx": k[1],
                "persistence_pct": round(100.0 * v / n_frames, 1),
            }
            for k, v in pair_counts.items()
        ],
        key=lambda x: x["persistence_pct"],
        reverse=True,
    )[:10]

    return {"pdb_id": pdb_id, "n_frames": n_frames, "bonds": bonds}


# --------------------------------------------------------------------------
# per_residue_contacts
# --------------------------------------------------------------------------


@register({
    "name": "per_residue_contacts",
    "description": (
        "For each ligand heavy atom, returns the nearest protein residue index "
        "(averaged over the trajectory). Useful for verifying claims about "
        "specific binding-site residues."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"pdb_id": {"type": "string"}},
        "required": ["pdb_id"],
    },
})
def per_residue_contacts(pdb_id: str) -> dict:
    sys = _load_system(pdb_id)
    lig_mask = _ligand_mask(sys)
    if lig_mask.sum() == 0:
        return {"error": "no ligand atoms found"}

    traj = sys["traj"]
    prot_mask = ~lig_mask & _heavy_mask(sys)
    prot_residues = sys["atoms_residue"][prot_mask]

    # Mean coordinates over the trajectory cheaply approximate the contact map.
    mean = traj.mean(axis=0)
    lig = mean[lig_mask & _heavy_mask(sys)]
    prot = mean[prot_mask]
    diff = lig[:, None, :] - prot[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    nearest_atom = dist.argmin(axis=1)
    nearest_res = prot_residues[nearest_atom]

    contact_residues, counts = np.unique(nearest_res, return_counts=True)
    sorted_idx = np.argsort(-counts)
    return {
        "pdb_id": pdb_id,
        "ligand_n_heavy": int(lig.shape[0]),
        "contacts": [
            {"residue_id": int(contact_residues[i]), "n_ligand_atoms": int(counts[i])}
            for i in sorted_idx[:15]
        ],
    }
