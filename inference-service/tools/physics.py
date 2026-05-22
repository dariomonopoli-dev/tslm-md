"""External-physics tools — task #16.

AutoDock Vina is invoked as a subprocess in scoring-only mode. The trajectory
frame is extracted from MD.hdf5, split into protein-receptor + ligand PDBs,
converted to PDBQT with Meeko, then scored.

Vina is non-deterministic by default — we force `--seed 42 --cpu 1`. The
kcal/mol → pK conversion is approximate (no entropy correction); we surface
the assumption in the result dict so the agent reports it honestly.

Graceful degradation: if Meeko or the Vina binary is missing, the tool
returns `{error: "..."}` so the agent can route around it instead of
crashing the loop.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import register


# kcal/mol → pK conversion at 298 K. ΔG = -RT ln Kd; pK = -log10(Kd) = -ΔG / (RT ln 10).
# RT ln 10 ≈ 1.36 kcal/mol at 298 K.
_RT_LN10_KCAL = 1.3633


def _frame_pdb_lines(pdb_id: str, frame_idx: int) -> tuple[list[str], list[str]]:
    """Return (protein_lines, ligand_lines) for one frame.

    Reuses the hdf5_to_pdb chain assignment so chain A = protein, L = ligand.
    """
    import h5py
    from hdf5_to_pdb import (
        _ATOM_FMT, _atom_name, _chain_for_atom, _load_maps, ATOMIC_NUMBERS_MAP,
    )

    path = os.getenv("MISATO_HDF5_PATH", "/app/data/MD.hdf5")
    with h5py.File(path, "r") as h5:
        key = pdb_id if pdb_id in h5 else pdb_id.upper()
        if key not in h5:
            raise KeyError(pdb_id)
        g = h5[key]
        traj = g["trajectory_coordinates"][frame_idx]  # (n_atoms, 3)
        atoms_type = g["atoms_type"][:]
        atoms_number = g["atoms_number"][:]
        atoms_residue = g["atoms_residue"][:]
        mol_begin = g["molecules_begin_atom_index"][:]

    residue_map, type_map, name_map = _load_maps()
    prot, lig = [], []
    residue_number = 1
    residue_atom_index = 0
    prot_serial = lig_serial = 0

    for i in range(len(atoms_type)):
        chain = _chain_for_atom(i, mol_begin)
        if chain == "W":
            continue
        residue_atom_index += 1
        type_string = type_map[atoms_type[i]]
        resname = residue_map[atoms_residue[i]]
        aname = _atom_name(i, atoms_number, residue_atom_index, resname, type_string, name_map)
        z = int(atoms_number[i])
        elem = ATOMIC_NUMBERS_MAP.get(z, "X")
        x, y, zc = float(traj[i, 0]), float(traj[i, 1]), float(traj[i, 2])
        if chain == "A":
            prot_serial += 1
            prot.append(_ATOM_FMT.format(
                serial=prot_serial, name=aname[:4], alt=" ", resname=resname[:3],
                chain="A", resseq=residue_number % 10000, x=x, y=y, z=zc,
                occ=1.0, tf=0.0, element=elem,
            ))
        else:  # 'L'
            lig_serial += 1
            lig.append(_ATOM_FMT.format(
                serial=lig_serial, name=aname[:4], alt=" ", resname="LIG",
                chain="L", resseq=1, x=x, y=y, z=zc,
                occ=1.0, tf=0.0, element=elem,
            ))

        if i < len(atoms_type) - 1:
            next_type = type_map[atoms_type[i + 1]]
            next_resname = residue_map[atoms_residue[i + 1]]
            if (type_string[0] == "O" and next_type[0] == "N") or next_resname == "MOL":
                gln_exc = resname == "GLN" and residue_atom_index in (12, 14)
                asn_exc = resname == "ASN" and residue_atom_index in (9, 11)
                if not (gln_exc or asn_exc):
                    residue_number += 1
                    residue_atom_index = 0

    prot.append("END")
    lig.append("END")
    return prot, lig


def _pdb_to_pdbqt_protein(pdb_path: Path, out_path: Path) -> str | None:
    """Use Meeko's receptor prep. Returns None on success, error string on failure."""
    try:
        from meeko import PDBQTReceptor
    except ImportError:
        # Fallback: minimal manual PDB → PDBQT by appending charges of 0
        # (Vina will still score; charges affect electrostatics only).
        try:
            lines = pdb_path.read_text().splitlines()
            out_lines = []
            for ln in lines:
                if ln.startswith("ATOM"):
                    elem = ln[76:78].strip() or "X"
                    out_lines.append(ln[:66] + "  0.000 " + f"{elem:<2s}")
                else:
                    out_lines.append(ln)
            out_path.write_text("\n".join(out_lines) + "\n")
            return None
        except Exception as e:
            return f"Meeko unavailable and fallback failed: {e}"

    try:
        rec = PDBQTReceptor(str(pdb_path))
        rec.write_pdbqt_file(str(out_path))
        return None
    except Exception as e:
        return f"Meeko receptor prep failed: {e}"


def _pdb_to_pdbqt_ligand(pdb_path: Path, out_path: Path) -> str | None:
    try:
        from meeko import MoleculePreparation
        from rdkit import Chem
    except ImportError as e:
        return f"Meeko/RDKit not installed: {e}"

    try:
        mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False)
        if mol is None:
            return f"RDKit could not parse {pdb_path}"
        mol = Chem.AddHs(mol, addCoords=True)
        prep = MoleculePreparation()
        prep.prepare(mol)
        out_path.write_text(prep.write_pdbqt_string())
        return None
    except Exception as e:
        return f"Meeko ligand prep failed: {e}"


def _run_vina(receptor: Path, ligand: Path) -> tuple[float | None, str]:
    vina_bin = os.getenv("VINA_BIN", shutil.which("vina") or "vina")
    if not shutil.which(vina_bin):
        return None, f"vina binary not found (tried '{vina_bin}')"

    cmd = [
        vina_bin, "--score_only",
        "--receptor", str(receptor),
        "--ligand", str(ligand),
        "--seed", "42",
        "--cpu", "1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return None, "vina timed out (>15 s)"
    if proc.returncode != 0:
        return None, f"vina exit {proc.returncode}: {proc.stderr[:400]}"

    # Vina prints "Affinity: X.XXX (kcal/mol)" — grab the float.
    import re
    m = re.search(r"Affinity:\s*(-?\d+(?:\.\d+)?)", proc.stdout)
    if not m:
        return None, f"could not parse vina output: {proc.stdout[:400]}"
    return float(m.group(1)), proc.stdout


def _cache_key(pdb_id: str, frame_idx: int) -> Path:
    cache_dir = Path(os.getenv("VINA_CACHE_DIR", "data/vina_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(f"{pdb_id}|{frame_idx}".encode()).hexdigest()[:16]
    return cache_dir / f"{pdb_id}_{frame_idx}_{h}.json"


@register({
    "name": "vina_rescore",
    "description": (
        "AutoDock Vina --score_only on a single trajectory frame. Returns "
        "kcal/mol and an approximate pK conversion. Run clash_check first on "
        "the frame to avoid scoring broken poses."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pdb_id": {"type": "string"},
            "frame_idx": {"type": "integer"},
        },
        "required": ["pdb_id", "frame_idx"],
    },
})
def vina_rescore(pdb_id: str, frame_idx: int) -> dict:
    import json

    cache = _cache_key(pdb_id, frame_idx)
    if cache.exists():
        return json.loads(cache.read_text())

    try:
        prot_lines, lig_lines = _frame_pdb_lines(pdb_id, frame_idx)
    except KeyError:
        return {"error": f"{pdb_id} not in MISATO HDF5"}
    except Exception as e:
        return {"error": f"frame extraction failed: {e}"}

    with tempfile.TemporaryDirectory() as tdir:
        td = Path(tdir)
        prot_pdb = td / "protein.pdb"
        lig_pdb = td / "ligand.pdb"
        prot_pdbqt = td / "protein.pdbqt"
        lig_pdbqt = td / "ligand.pdbqt"

        prot_pdb.write_text("\n".join(prot_lines))
        lig_pdb.write_text("\n".join(lig_lines))

        if err := _pdb_to_pdbqt_protein(prot_pdb, prot_pdbqt):
            return {"error": err}
        if err := _pdb_to_pdbqt_ligand(lig_pdb, lig_pdbqt):
            return {"error": err}

        kcal, stdout = _run_vina(prot_pdbqt, lig_pdbqt)
        if kcal is None:
            return {"error": stdout}

    approx_pK = -kcal / _RT_LN10_KCAL  # ΔG (-) → pK (+)
    result = {
        "pdb_id": pdb_id,
        "frame_idx": frame_idx,
        "vina_kcal_mol": round(kcal, 3),
        "approx_pK": round(approx_pK, 2),
        "tool_version": "vina-1.2",
        "approximation_note": (
            "pK ≈ -ΔG / (RT ln 10) at 298 K; no entropy/solvation correction. "
            "Compare to model_pK only as a soft prior."
        ),
    }
    cache.write_text(json.dumps(result, indent=2))
    return result
