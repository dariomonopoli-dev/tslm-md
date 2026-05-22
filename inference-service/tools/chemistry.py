"""Chemistry tools — RDKit on ligand SMILES — task #15.

The TSLM did not see SMILES during training, so MW/LogP/LE are orthogonal.
"""

from __future__ import annotations

import csv
import os
from functools import lru_cache
from pathlib import Path

from . import register


@lru_cache(maxsize=1)
def _smiles_map() -> dict[str, str]:
    """pdb_id → ligand SMILES from PDBbind/MISATO sidecar files.

    Falls back to an empty map if no source is configured — the tool then
    returns {error: 'no SMILES'} for unknown systems.
    """
    candidates = [
        Path(os.getenv("LIGAND_SMILES_CSV", "/app/data/ligand_smiles.csv")),
        Path("misato-affinity/data/ligand_smiles.csv"),
        Path("../misato-affinity/data/ligand_smiles.csv"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if not path:
        return {}
    out: dict[str, str] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("pdb_id") or row.get("PDBid") or "").strip()
            smi = (row.get("smiles") or row.get("SMILES") or "").strip()
            if pid and smi:
                out[pid] = smi
    return out


@register({
    "name": "ligand_descriptors",
    "description": (
        "RDKit-derived ligand descriptors: molecular weight, LogP, ligand "
        "efficiency (assuming model_pK). The model did not see SMILES; this "
        "is independent chemistry."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pdb_id": {"type": "string"},
            "model_pK": {"type": "number", "description": "needed for LE"},
        },
        "required": ["pdb_id", "model_pK"],
    },
})
def ligand_descriptors(pdb_id: str, model_pK: float) -> dict:
    smi = _smiles_map().get(pdb_id)
    if not smi:
        return {"error": f"no SMILES for {pdb_id}"}

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Crippen, Descriptors
    except ImportError as e:
        return {"error": f"RDKit not installed: {e}"}

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {"error": f"could not parse SMILES: {smi}"}

    mw = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    heavy_n = int(mol.GetNumHeavyAtoms())
    le = float(model_pK * 1.36 / heavy_n) if heavy_n else float("nan")  # ΔG/HA → kcal/mol/atom

    return {
        "pdb_id": pdb_id,
        "smiles": smi,
        "mw": round(mw, 2),
        "logp": round(logp, 2),
        "n_heavy_atoms": heavy_n,
        "ligand_efficiency": round(le, 3),
        "le_plausible": bool(0.20 <= le <= 0.50),
    }
