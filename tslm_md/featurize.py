"""Featurise a MISATO MD trajectory into a [10, F_sub] tensor.

Six geometric channels recomputed from `trajectory_coordinates`:
  ch0  min pocket-ligand distance (Å)
  ch1  mean pocket-ligand distance under 4 Å mask (Å)
  ch2  number of close (<= 4 Å) ligand-protein contacts
  ch3  ligand RMSD from frame 0 after pocket-Kabsch alignment (Å)
  ch4  ligand radius of gyration (Å)
  ch5  interface buriedness proxy: count of ligand atoms with <= 2 protein neighbours within 5 Å

Four MISATO-precomputed per-frame channels (subsampled at the same indices):
  ch6  frames_interaction_energy (kcal/mol)
  ch7  frames_distance (Å)
  ch8  frames_rmsd_ligand (Å)
  ch9  frames_bSASA (Å²)
"""

from __future__ import annotations

from typing import Optional

import h5py
import numpy as np
import torch

N_CHANNELS = 10
F_SUB = 30
PRECOMPUTED_CHANNELS = (
    "frames_interaction_energy",
    "frames_distance",
    "frames_rmsd_ligand",
    "frames_bSASA",
)
POCKET_CUTOFF_A = 6.0
CONTACT_CUTOFF_A = 4.0
SASA_PROXY_CUTOFF_A = 5.0
SASA_PROXY_MAX_NEIGHBOURS = 2


def kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Optimal rotation matrix to align centered P onto centered Q. Both (N, 3)."""
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    return U @ D @ Vt


def featurize(
    h5_group: h5py.Group,
    n_frames_out: int = F_SUB,
) -> torch.Tensor:
    """Convert one MISATO PDB-id trajectory to a [6, n_frames_out] feature tensor.

    Args:
        h5_group: open h5py group for one PDB id, e.g. f["11GS"].
        n_frames_out: number of subsampled frames.
    Returns:
        torch.FloatTensor of shape [N_CHANNELS, n_frames_out].
    """
    coords = h5_group["trajectory_coordinates"][:]  # (F, N, 3)
    molecules_begin = h5_group["molecules_begin_atom_index"][:]
    ligand_start = int(molecules_begin[-1])

    F_raw = coords.shape[0]
    if F_raw < n_frames_out:
        indices = np.concatenate([
            np.arange(F_raw),
            np.full(n_frames_out - F_raw, F_raw - 1),
        ])
    else:
        indices = np.linspace(0, F_raw - 1, n_frames_out).astype(int)
    coords = coords[indices]  # (T, N, 3)
    T = coords.shape[0]

    protein = coords[:, :ligand_start, :]
    ligand = coords[:, ligand_start:, :]

    # Pocket = protein atoms within POCKET_CUTOFF_A of any ligand atom in frame 0.
    p0 = protein[0]
    l0 = ligand[0]
    d0 = np.linalg.norm(p0[:, None, :] - l0[None, :, :], axis=-1)  # (N_p, N_l)
    pocket_mask = d0.min(axis=1) <= POCKET_CUTOFF_A
    pocket = protein[:, pocket_mask, :]

    feats = np.zeros((N_CHANNELS, T), dtype=np.float32)

    # ch6-9: MISATO-precomputed per-frame scalars, subsampled at the same indices.
    # These do not depend on the geometric pocket calculation, so fill them
    # before the degenerate-pocket early return.
    for offset, name in enumerate(PRECOMPUTED_CHANNELS):
        if name in h5_group:
            arr = h5_group[name][:]
            if arr.shape[0] >= F_raw:
                feats[6 + offset] = arr[indices].astype(np.float32)

    if pocket.shape[1] == 0:
        # Degenerate (e.g., ligand far from any protein atom). Geometric channels
        # stay zero; precomputed channels (if available) are still populated.
        return torch.from_numpy(feats)

    ref_ligand = ligand[0]
    ref_pocket = pocket[0]

    for t in range(T):
        l_t = ligand[t]
        p_t = pocket[t]
        d_pl = np.linalg.norm(p_t[:, None, :] - l_t[None, :, :], axis=-1)  # (N_pkt, N_l)

        # ch0 min distance
        feats[0, t] = d_pl.min()

        # ch1 mean masked distance
        mask4 = d_pl <= CONTACT_CUTOFF_A
        feats[1, t] = d_pl[mask4].mean() if mask4.any() else CONTACT_CUTOFF_A

        # ch2 contact count
        feats[2, t] = float(mask4.sum())

        # ch3 ligand RMSD after pocket Kabsch alignment
        com_pt = p_t.mean(0)
        com_pref = ref_pocket.mean(0)
        R = kabsch(p_t - com_pt, ref_pocket - com_pref)
        l_t_aligned = (l_t - com_pt) @ R + com_pref
        feats[3, t] = float(np.sqrt(((l_t_aligned - ref_ligand) ** 2).sum(-1).mean()))

        # ch4 ligand radius of gyration
        com_l = l_t.mean(0)
        feats[4, t] = float(np.sqrt(((l_t - com_l) ** 2).sum(-1).mean()))

        # ch5 buriedness proxy
        d_lp = np.linalg.norm(l_t[:, None, :] - p_t[None, :, :], axis=-1)  # (N_l, N_pkt)
        n_neigh = (d_lp <= SASA_PROXY_CUTOFF_A).sum(axis=1)
        feats[5, t] = float((n_neigh <= SASA_PROXY_MAX_NEIGHBOURS).sum())

    return torch.from_numpy(feats)


def normalise(
    feats: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Z-score per channel. If mean/std omitted, use per-feature stats (do this at train-set level)."""
    if mean is None:
        mean = feats.mean(dim=1, keepdim=True)
    if std is None:
        std = feats.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (feats - mean) / std
