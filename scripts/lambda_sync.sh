#!/usr/bin/env bash
# Push OpenTSLM source + preprocessed features to the Lambda box.
# Run from your laptop, after editing ~/.ssh/config to set HostName for `lambda`.
#
#   bash scripts/lambda_sync.sh
set -euo pipefail

PROJECT_DIR="/home/mxlk/Documents/AIproject /"   # trailing space before slash is real
REMOTE="lambda:~/tslm-md"

ssh lambda 'mkdir -p ~/tslm-md'

# Code (OpenTSLM fork only — that's all the trainer needs)
rsync -avz --delete \
    --exclude '.venv' --exclude '__pycache__' --exclude 'wandb' \
    --exclude '.git' --exclude 'runs' \
    "${PROJECT_DIR}OpenTSLM/" "${REMOTE}/OpenTSLM/"

# Preprocessed features (26 MB total)
rsync -avz "${PROJECT_DIR}preprocessed/" "${REMOTE}/preprocessed/"

# Bootstrap script
rsync -avz "${PROJECT_DIR}scripts/lambda_bootstrap.sh" "${REMOTE}/"

echo "Sync complete. Now: ssh lambda, then 'HF_TOKEN=hf_... bash lambda_bootstrap.sh'"
