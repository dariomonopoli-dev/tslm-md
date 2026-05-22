"""Selective MISATO download via h5py ROS3 driver (HTTP range reads).

Instead of downloading the full 132 GB MD.hdf5, fetch ONLY the PDB ids you
need into a local subset HDF5. Typical: 500-2000 ids = ~2-5 GB, minutes not
hours.

Requires h5py compiled with the ROS3 driver:
    python -c "import h5py; print('ros3' in h5py.registered_drivers())"
Should print True. If False, fall back to scripts/download_misato_direct.sh.

Usage:
    # First grab the official split files (tiny):
    wget -P data/misato/splits/ https://zenodo.org/records/7711953/files/train_MD.txt
    wget -P data/misato/splits/ https://zenodo.org/records/7711953/files/val_MD.txt
    wget -P data/misato/splits/ https://zenodo.org/records/7711953/files/test_MD.txt

    # Then subset:
    python scripts/download_misato_subset.py --max-train 500 --max-val 100 --max-test 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
from tqdm import tqdm

ZENODO_MD_URL = "https://zenodo.org/records/7711953/files/MD.hdf5"


def read_ids(path: Path, cap: int | None) -> list[str]:
    if not path.exists():
        print(f"  skip {path.name} (not found)")
        return []
    with path.open() as f:
        ids = [line.strip() for line in f if line.strip()]
    if cap and len(ids) > cap:
        ids = ids[:cap]
    return ids


def main(args: argparse.Namespace) -> None:
    if "ros3" not in h5py.registered_drivers():
        raise SystemExit(
            "h5py was not built with the ROS3 driver. "
            "Reinstall: `pip install --no-binary=h5py h5py` "
            "or fall back to scripts/download_misato_direct.sh."
        )

    splits_dir = Path(args.splits_dir)
    out_path = Path(args.out_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_ids = read_ids(splits_dir / "train_MD.txt", args.max_train)
    val_ids = read_ids(splits_dir / "val_MD.txt", args.max_val)
    test_ids = read_ids(splits_dir / "test_MD.txt", args.max_test)
    wanted = list(dict.fromkeys(train_ids + val_ids + test_ids))  # de-dupe, preserve order
    print(f"requesting {len(wanted)} pdb ids (train {len(train_ids)} / val {len(val_ids)} / test {len(test_ids)})")

    fetched = 0
    skipped = 0
    print(f"opening {ZENODO_MD_URL} via ROS3 driver — this can take ~20 s to handshake...")
    with h5py.File(ZENODO_MD_URL, mode="r", driver="ros3") as src, \
         h5py.File(out_path, "w") as dst:
        for pid in tqdm(wanted, desc="fetching", unit="pdb"):
            if pid not in src:
                skipped += 1
                continue
            grp = dst.create_group(pid)
            for k in src[pid].keys():
                # Reading [:] forces a download of THAT key only.
                grp.create_dataset(k, data=src[pid][k][:], compression="gzip")
            fetched += 1

    print(f"done. fetched {fetched} ids, skipped {skipped}, wrote {out_path}")
    print(f"out file size: {out_path.stat().st_size / 1e9:.2f} GB")

    # Also write local split files reflecting what we actually have
    local_splits = Path("data/splits")
    local_splits.mkdir(parents=True, exist_ok=True)
    with (local_splits / "train.txt").open("w") as f:
        f.write("\n".join(train_ids))
    with (local_splits / "val.txt").open("w") as f:
        f.write("\n".join(val_ids))
    with (local_splits / "test.txt").open("w") as f:
        f.write("\n".join(test_ids))
    print(f"wrote local split files to {local_splits}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir", default="data/misato/splits",
                   help="directory holding the official train/val/test _MD.txt files from Zenodo")
    p.add_argument("--out-h5", default="data/misato/MD_subset.hdf5")
    p.add_argument("--max-train", type=int, default=500)
    p.add_argument("--max-val", type=int, default=100)
    p.add_argument("--max-test", type=int, default=200)
    main(p.parse_args())
