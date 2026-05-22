"""Generate a tiny synthetic MISATO-style HDF5 for dry-run / architecture testing.

The real MD.hdf5 takes hours to download. This script builds a tiny file
(~1 MB) with the SAME SCHEMA so we can run the dry-run end-to-end NOW
to verify the architecture wires up, then re-run on the real file once
the download finishes.

What it tests:
  - featurize.py reads trajectory_coordinates + molecules_begin_atom_index
  - Model loads, forward + backward passes
  - Generate / parse pipeline works

What it does NOT test:
  - That training converges on real data
  - That our featurization captures real binding signal

Run:
    python scripts/make_dummy_md.py
    python scripts/dry_run.py --misato-h5 data/misato/MD_dummy.hdf5 --pdb-id TEST
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def main(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    n_frames = args.frames
    n_protein = args.n_protein
    n_ligand = args.n_ligand
    n_total = n_protein + n_ligand

    # Build a plausible-ish trajectory: protein atoms randomly distributed
    # in a 20A box; ligand atoms cluster near the protein center.
    protein_coords_frame0 = rng.uniform(-10, 10, size=(n_protein, 3)).astype(np.float64)
    ligand_center = protein_coords_frame0.mean(axis=0) + rng.uniform(-2, 2, size=3)
    ligand_coords_frame0 = ligand_center + rng.normal(0, 1.5, size=(n_ligand, 3)).astype(np.float64)

    trajectory = np.zeros((n_frames, n_total, 3), dtype=np.float64)
    for t in range(n_frames):
        # Add small thermal jitter to the whole system
        protein_jitter = rng.normal(0, 0.15, size=(n_protein, 3))
        ligand_drift = rng.normal(0, 0.1, size=(n_ligand, 3)).cumsum(axis=0) * 0.01
        trajectory[t, :n_protein, :] = protein_coords_frame0 + protein_jitter
        trajectory[t, n_protein:, :] = ligand_coords_frame0 + ligand_drift

    atoms_element = np.concatenate([
        rng.choice([6, 7, 8], size=n_protein, p=[0.7, 0.2, 0.1]).astype(np.int8),  # C,N,O
        rng.choice([6, 7, 8], size=n_ligand, p=[0.7, 0.2, 0.1]).astype(np.int8),
    ])
    # MISATO stores molecules_begin_atom_index as an array; the LAST entry is
    # where the ligand starts.
    molecules_begin_atom_index = np.array([0, n_protein], dtype=np.int64)

    with h5py.File(out, "w") as f:
        for pid in args.ids:
            g = f.create_group(pid)
            g.create_dataset("trajectory_coordinates", data=trajectory, compression="gzip")
            g.create_dataset("atoms_element", data=atoms_element, compression="gzip")
            g.create_dataset(
                "molecules_begin_atom_index", data=molecules_begin_atom_index
            )
            # frames_* energies (for the verifier later)
            g.create_dataset(
                "frames_EPtot",
                data=rng.normal(-100.0, 5.0, size=n_frames).astype(np.float64),
            )
            g.create_dataset(
                "frames_EELEC",
                data=rng.normal(-50.0, 3.0, size=n_frames).astype(np.float64),
            )
            g.create_dataset(
                "frames_EVDW",
                data=rng.normal(-30.0, 2.0, size=n_frames).astype(np.float64),
            )

    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out} ({size_mb:.2f} MB) with ids: {args.ids}")
    print(
        f"  trajectory shape per id: ({n_frames}, {n_total}, 3)  "
        f"(protein 0..{n_protein-1}, ligand {n_protein}..{n_total-1})"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/misato/MD_dummy.hdf5")
    p.add_argument("--ids", nargs="+", default=["TEST", "11GS"],
                   help="PDB id keys to create at top level")
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--n-protein", type=int, default=500)
    p.add_argument("--n-ligand", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
