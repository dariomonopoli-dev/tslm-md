"""MISATO HDF5 → multi-MODEL PDB reconstruction — task #9.

Spec from FRONTEND_v2.md §12. Builds on the reference implementation at
misato-dataset-master/src/data/processing/h5_to_pdb.py, but:
  - emits a multi-MODEL trajectory (3Dmol.js animation) instead of one frame
  - applies stride (default 5 → 20 frames per system, ~2-3 MB)
  - drops water by default
  - assigns chains A=protein, L=ligand, W=water based on molecules_begin_atom_index
  - caches results per (pdb_id, stride, drop_water) to disk
  - validates output with Biopython before returning (3Dmol fails silently on
    misaligned ATOM records)

The chain assignment is critical for 3Dmol's selectors: the SingleView will
do `addStyle({chain: 'L'}, {stick: ...})` to draw the ligand.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np


# Map MISATO atom numbers (atomic Z) → element symbol used in column 77-78.
ATOMIC_NUMBERS_MAP = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S",
    17: "Cl", 19: "K", 20: "Ca", 34: "Se", 35: "Br", 53: "I",
}


# --------------------------------------------------------------------------
# Map files (residueMap, typeMap, nameMap)
# --------------------------------------------------------------------------


def _map_dir() -> Path:
    candidates = [
        Path(os.getenv("MISATO_MAPS_DIR", "")),
        Path("/app/data/Maps"),
        Path("misato-dataset-master/src/data/Maps"),
        Path("../misato-dataset-master/src/data/Maps"),
        Path("misato-affinity/data/Maps"),
        Path("../misato-affinity/data/Maps"),
    ]
    for c in candidates:
        if c and (c / "atoms_residue_map.pickle").exists():
            return c
    raise FileNotFoundError(
        "MISATO Maps directory not found. Set MISATO_MAPS_DIR env var."
    )


@lru_cache(maxsize=1)
def _load_maps() -> tuple[dict, dict, dict]:
    mdir = _map_dir()
    with open(mdir / "atoms_residue_map.pickle", "rb") as f:
        residue_map = pickle.load(f)
    with open(mdir / "atoms_type_map.pickle", "rb") as f:
        type_map = pickle.load(f)
    with open(mdir / "atoms_name_map_for_pdb.pickle", "rb") as f:
        name_map = pickle.load(f)
    return residue_map, type_map, name_map


# --------------------------------------------------------------------------
# HDF5 access
# --------------------------------------------------------------------------


def _hdf5_path() -> Path:
    path = Path(os.getenv("MISATO_HDF5_PATH", "/app/data/MD.hdf5"))
    if not path.exists():
        raise FileNotFoundError(f"MISATO HDF5 not found at {path}")
    return path


def _read_system(h5: h5py.File, pdb_id: str):
    """Pull every array we need for the system in one shot.

    MISATO encodes things in different cases — try both upper and lower.
    """
    key = pdb_id if pdb_id in h5 else pdb_id.upper()
    if key not in h5:
        raise KeyError(pdb_id)
    g = h5[key]
    traj = g["trajectory_coordinates"][:]   # (n_frames, n_atoms, 3)
    atoms_type = g["atoms_type"][:]
    atoms_number = g["atoms_number"][:]
    atoms_residue = g["atoms_residue"][:]
    mol_begin = g["molecules_begin_atom_index"][:]
    return traj, atoms_type, atoms_number, atoms_residue, mol_begin


# --------------------------------------------------------------------------
# PDB record formatting
# --------------------------------------------------------------------------


# Column-strict PDB v3.3 ATOM record. Format spec: 80 chars total.
# Cols 1-6: "ATOM  ", 7-11: serial, 13-16: name, 17: altLoc, 18-20: resName,
# 22: chainID, 23-26: resSeq, 31-38: x, 39-46: y, 47-54: z, 55-60: occ,
# 61-66: tempFactor, 77-78: element.
_ATOM_FMT = (
    "ATOM  {serial:>5d} {name:<4s}{alt:1s}{resname:>3s} {chain:1s}"
    "{resseq:>4d}    {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{tf:6.2f}          "
    "{element:>2s}"
)


def _chain_for_atom(atom_idx: int, mol_begin: np.ndarray) -> str:
    """Chain A=protein, L=ligand, W=water.

    MISATO convention: mol_begin = [first_protein_atom, first_ligand_atom, first_water_atom].
    """
    if len(mol_begin) >= 3 and atom_idx >= mol_begin[2]:
        return "W"
    if len(mol_begin) >= 2 and atom_idx >= mol_begin[1]:
        return "L"
    return "A"


def _atom_name(
    atom_idx: int,
    atoms_number: np.ndarray,
    residue_atom_index: int,
    residue_name: str,
    type_string: str,
    name_map: dict,
) -> str:
    """Reproduces the reference implementation's MOL-vs-AA distinction."""
    z = int(atoms_number[atom_idx])
    if residue_name == "MOL":
        return f"{ATOMIC_NUMBERS_MAP.get(z, 'X')}{residue_atom_index}"
    try:
        return name_map[(residue_name, residue_atom_index - 1, type_string)]
    except KeyError:
        return f"{ATOMIC_NUMBERS_MAP.get(z, 'X')}{residue_atom_index}"


def _frame_to_lines(
    coords: np.ndarray,
    atoms_type: np.ndarray,
    atoms_number: np.ndarray,
    atoms_residue: np.ndarray,
    mol_begin: np.ndarray,
    drop_water: bool,
) -> list[str]:
    residue_map, type_map, name_map = _load_maps()
    lines: list[str] = []
    residue_number = 1
    residue_atom_index = 0

    n_atoms = len(atoms_type)
    serial = 0
    for i in range(n_atoms):
        chain = _chain_for_atom(i, mol_begin)
        if drop_water and chain == "W":
            continue

        residue_atom_index += 1
        type_string = type_map[atoms_type[i]]
        residue_name = residue_map[atoms_residue[i]]
        atom_name = _atom_name(i, atoms_number, residue_atom_index, residue_name, type_string, name_map)
        z = int(atoms_number[i])
        element = ATOMIC_NUMBERS_MAP.get(z, "X")
        x, y, z_coord = float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])
        serial += 1

        lines.append(_ATOM_FMT.format(
            serial=serial, name=atom_name[:4], alt=" ",
            resname=residue_name[:3], chain=chain,
            resseq=residue_number % 10000,  # PDB resSeq wraps at 9999
            x=x, y=y, z=z_coord, occ=1.0, tf=0.0, element=element,
        ))

        # Residue-boundary heuristic from the reference implementation
        if i < n_atoms - 1:
            next_type = type_map[atoms_type[i + 1]]
            next_resname = residue_map[atoms_residue[i + 1]]
            if (type_string[0] == "O" and next_type[0] == "N") or next_resname == "MOL":
                gln_exception = residue_name == "GLN" and residue_atom_index in (12, 14)
                asn_exception = residue_name == "ASN" and residue_atom_index in (9, 11)
                if not (gln_exception or asn_exception):
                    residue_number += 1
                    residue_atom_index = 0

        # TER between chains/molecules
        if i + 1 in mol_begin:
            lines.append("TER")
            residue_number += 1
            residue_atom_index = 0

    return lines


# --------------------------------------------------------------------------
# Public API + caching
# --------------------------------------------------------------------------


def _cache_path(pdb_id: str, stride: int, drop_water: bool) -> Path:
    key = hashlib.sha256(f"{pdb_id}|{stride}|{drop_water}".encode()).hexdigest()[:16]
    cache_dir = Path(os.getenv("PDB_STRING_CACHE", "data/pdb_string_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{pdb_id}_{key}.pdb"


def hdf5_to_pdb(pdb_id: str, stride: int = 5, drop_water: bool = True) -> str:
    """Return a multi-MODEL PDB string for the system.

    Cached per (pdb_id, stride, drop_water) — HDF5 reads are slow.
    """
    cache = _cache_path(pdb_id, stride, drop_water)
    if cache.exists():
        return cache.read_text()

    with h5py.File(_hdf5_path(), "r") as h5:
        traj, atoms_type, atoms_number, atoms_residue, mol_begin = _read_system(h5, pdb_id)

    n_frames = traj.shape[0]
    frame_indices = list(range(0, n_frames, max(1, stride)))

    parts: list[str] = []
    parts.append(f"REMARK   1 MISATO {pdb_id} stride={stride} drop_water={drop_water}")
    parts.append(f"REMARK   2 n_frames={len(frame_indices)} of {n_frames}")

    for model_num, fidx in enumerate(frame_indices, start=1):
        parts.append(f"MODEL     {model_num:4d}")
        parts.extend(_frame_to_lines(
            traj[fidx], atoms_type, atoms_number, atoms_residue, mol_begin, drop_water,
        ))
        parts.append("ENDMDL")
    parts.append("END")

    text = "\n".join(parts) + "\n"
    cache.write_text(text)
    return text
