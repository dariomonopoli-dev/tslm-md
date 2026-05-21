#!/usr/bin/env bash
# Pull the small affinity labels + official splits from kierandidi/misato-affinity.
#
# This gets us in seconds what would otherwise require parsing PDBbind ourselves:
#   - data/affinity_data.h5     (~67 MB)  : binding affinity labels keyed by pdb_id
#   - data/train_pairs.pickle              : official train split
#   - data/val_pairs.pickle                : official val split
#   - data/test_pairs.pickle               : official test split
#   - data/affinity_data.csv               : human-readable label table
#
# The MD trajectories still need to come from Zenodo (MD.hdf5).

set -euo pipefail

REPO="kierandidi/misato-affinity"
DEST_DIR="data/misato_affinity"
mkdir -p "$DEST_DIR"

FILES=(
  data/affinity_data.h5
  data/affinity_data.csv
  data/train_pairs.pickle
  data/val_pairs.pickle
  data/test_pairs.pickle
  data/affinity_structs.pickle
)

for f in "${FILES[@]}"; do
  out="${DEST_DIR}/$(basename "$f")"
  url="https://raw.githubusercontent.com/${REPO}/main/${f}"
  echo "==> $url -> $out"
  curl -sL --fail -o "$out" "$url" || curl -sL --fail -o "$out" "${url/main/master}"
done

echo
echo "==> Done. Files in $DEST_DIR:"
ls -lh "$DEST_DIR"
echo
echo "Note: MD trajectories still required from Zenodo (MD.hdf5)."
echo "Either:"
echo "  - bash scripts/download_misato_direct.sh         (full 133 GiB download)"
echo "  - python scripts/download_misato_subset.py        (selective via h5py ROS3)"
