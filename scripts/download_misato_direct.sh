#!/usr/bin/env bash
# Direct MISATO download from Zenodo to local disk.
#
# Recommended on any datacenter-network box (vast.ai, paperspace, etc).
# Faster + simpler than the EC2 → S3 detour, which is only useful when
# the download endpoint is a home Wi-Fi connection.
#
# After the download completes, optionally `aws s3 sync` to S3 for durability
# (in case the vast.ai instance is destroyed and you need to rebuild).
#
# USAGE:
#   bash scripts/download_misato_direct.sh
#   # optional backup:
#   aws s3 sync data/misato s3://your-bucket-name/misato/

set -euo pipefail

DATA_DIR="${DATA_DIR:-data/misato}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "==> Installing aria2 (parallel-connection downloader)"
if ! command -v aria2c >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y aria2
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y aria2
  elif command -v brew >/dev/null 2>&1; then
    brew install aria2
  else
    echo "Please install aria2 manually for your distribution."
    exit 1
  fi
fi

echo "==> Downloading MD.hdf5 (~133 GiB) from Zenodo"
aria2c -x 8 -s 8 --continue=true \
  'https://zenodo.org/records/7711953/files/MD.hdf5'

echo "==> Downloading QM.hdf5 (~0.3 GiB) from Zenodo"
aria2c -x 8 -s 8 --continue=true \
  'https://zenodo.org/records/7711953/files/QM.hdf5'

echo
echo "==> Done. Files in $DATA_DIR:"
ls -lh

cat <<'EOF'

==> Optional: back up to S3 for durability (in case the GPU box dies)
    aws s3 sync data/misato s3://<your-bucket>/misato/
EOF
