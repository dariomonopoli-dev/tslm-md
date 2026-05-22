"""Render MD trajectory frames to PNG — task #30.

One frame at a time. The 3D pocket view panel in the UI calls
GET /frame_image/{pdb_id}?frame=N to display these.

Renderer: matplotlib 3D scatter (no extra dependencies beyond what's
already installed). Protein in muted gray, ligand in orange — same color
language as the live 3Dmol view. Caches PNGs to disk so repeated views
are instant.

For higher-quality output (publication figures, residue labels), swap in
PyMOL by setting RENDER_BACKEND=pymol — requires
`pip install pymol-open-source` (~50 MB). matplotlib fallback ships free.
"""

from __future__ import annotations

import hashlib
import io
import os
from functools import lru_cache
from pathlib import Path

import numpy as np


_BACKEND = os.getenv("RENDER_BACKEND", "matplotlib").lower()


def _cache_path(pdb_id: str, frame_idx: int, width: int, height: int) -> Path:
    cache_dir = Path(os.getenv("FRAME_IMAGE_CACHE", "data/frame_images"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{pdb_id}|{frame_idx}|{width}x{height}|{_BACKEND}".encode()).hexdigest()[:12]
    return cache_dir / f"{pdb_id}_f{frame_idx}_{key}.png"


@lru_cache(maxsize=64)
def _load_frame(pdb_id: str, frame_idx: int) -> dict:
    """Pull the per-frame coords + per-atom metadata. Cached per (pdb, frame)."""
    import h5py
    path = os.getenv("MISATO_HDF5_PATH", "/app/data/MD.hdf5")
    with h5py.File(path, "r") as h5:
        key = pdb_id if pdb_id in h5 else pdb_id.upper()
        if key not in h5:
            raise KeyError(pdb_id)
        g = h5[key]
        n_frames = g["trajectory_coordinates"].shape[0]
        if not 0 <= frame_idx < n_frames:
            raise IndexError(f"frame {frame_idx} out of range [0, {n_frames})")
        return {
            "coords": g["trajectory_coordinates"][frame_idx][:],   # (n_atoms, 3)
            "atoms_number": g["atoms_number"][:],
            "mol_begin": g["molecules_begin_atom_index"][:],
            "n_frames": n_frames,
        }


def _chain_mask(mol_begin: np.ndarray, n_atoms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (protein_mask, ligand_mask, water_mask)."""
    prot = np.zeros(n_atoms, dtype=bool)
    lig = np.zeros(n_atoms, dtype=bool)
    wat = np.zeros(n_atoms, dtype=bool)
    if len(mol_begin) >= 1:
        prot_end = int(mol_begin[1]) if len(mol_begin) >= 2 else n_atoms
        prot[int(mol_begin[0]):prot_end] = True
    if len(mol_begin) >= 2:
        lig_end = int(mol_begin[2]) if len(mol_begin) >= 3 else n_atoms
        lig[int(mol_begin[1]):lig_end] = True
    if len(mol_begin) >= 3:
        wat[int(mol_begin[2]):] = True
    return prot, lig, wat


def _render_matplotlib(pdb_id: str, frame_idx: int, width: int, height: int) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

    data = _load_frame(pdb_id, frame_idx)
    coords = data["coords"]
    nums = data["atoms_number"]
    n_atoms = coords.shape[0]
    prot, lig, _ = _chain_mask(data["mol_begin"], n_atoms)

    # Heavy atoms only — skip hydrogens for clarity
    heavy = nums != 1
    prot_h = prot & heavy
    lig_h = lig & heavy

    # Center on the ligand if it exists, otherwise on the protein
    focus = coords[lig_h] if lig_h.any() else coords[prot_h]
    center = focus.mean(axis=0)
    coords_c = coords - center

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor="#1a1b26")
    ax = fig.add_subplot(111, projection="3d", facecolor="#1a1b26")

    # Protein backbone — thinned scatter, muted gray
    pc = coords_c[prot_h]
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=2, c="#64748b", alpha=0.35, depthshade=True)

    # Ligand sticks — show as larger orange points connected by lines
    lc = coords_c[lig_h]
    if len(lc) > 0:
        ax.scatter(lc[:, 0], lc[:, 1], lc[:, 2], s=42, c="#fb923c",
                   edgecolors="#fdba74", linewidths=0.8, depthshade=True)
        # Connect nearest neighbors to suggest bonds
        for i, p in enumerate(lc):
            for j in range(i + 1, len(lc)):
                if np.linalg.norm(lc[j] - p) < 1.8:  # heavy-atom bond cutoff
                    ax.plot([p[0], lc[j][0]], [p[1], lc[j][1]], [p[2], lc[j][2]],
                            color="#fdba74", linewidth=1.2, alpha=0.85)

    ax.set_axis_off()
    ax.set_facecolor("#1a1b26")
    # Frame-the-ligand zoom
    if len(lc) > 0:
        m = max(8.0, np.max(np.abs(lc)) * 1.4)
    else:
        m = 25.0
    ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_zlim(-m, m)

    # Annotate frame in top-left
    fig.text(0.02, 0.96, f"{pdb_id}  frame {frame_idx}/{data['n_frames'] - 1}",
             color="#cbd5e1", fontsize=10, family="monospace")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="#1a1b26", edgecolor="none",
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return buf.getvalue()


def _render_pymol(pdb_id: str, frame_idx: int, width: int, height: int) -> bytes:
    """Optional higher-quality path. Requires pip install pymol-open-source."""
    try:
        from pymol import cmd
    except ImportError as e:
        raise RuntimeError(
            "RENDER_BACKEND=pymol but pymol-open-source not installed"
        ) from e
    import tempfile
    from hdf5_to_pdb import hdf5_to_pdb

    pdb_text = hdf5_to_pdb(pdb_id, stride=max(1, frame_idx + 1), drop_water=True)
    # The above returns a multi-MODEL PDB starting at frame 0; for one frame
    # we just take MODEL 1. Simpler: extract a single-frame PDB ourselves.
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tf:
        tf.write(pdb_text)
        tmp_pdb = tf.name
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tmp_png = tf.name

    cmd.reinitialize()
    cmd.bg_color("#1a1b26")
    cmd.load(tmp_pdb, "system")
    cmd.frame(1)
    cmd.hide("everything")
    cmd.show("cartoon", "chain A")
    cmd.color("gray60", "chain A")
    cmd.show("sticks", "chain L")
    cmd.color("orange", "chain L")
    cmd.zoom("chain L", 6)
    cmd.ray(width, height)
    cmd.png(tmp_png, dpi=120)
    data = Path(tmp_png).read_bytes()
    Path(tmp_pdb).unlink(missing_ok=True)
    Path(tmp_png).unlink(missing_ok=True)
    return data


def render_frame(pdb_id: str, frame_idx: int, width: int = 600, height: int = 450) -> bytes:
    """Return PNG bytes. Cached per (pdb, frame, dims, backend)."""
    cache = _cache_path(pdb_id, frame_idx, width, height)
    if cache.exists():
        return cache.read_bytes()
    if _BACKEND == "pymol":
        data = _render_pymol(pdb_id, frame_idx, width, height)
    else:
        data = _render_matplotlib(pdb_id, frame_idx, width, height)
    cache.write_bytes(data)
    return data
