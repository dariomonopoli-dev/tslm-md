#!/usr/bin/env bash
# Launch stage6 training in the background with the juncliu starting checkpoint.
# Single command, no multi-line backslashes — avoids zsh quoting traps.
#
# Usage:
#   bash scripts/launch_train.sh            # foreground, see output live
#   bash scripts/launch_train.sh nohup      # background, log to train.log

set -euo pipefail

CKPT="$HOME/.cache/huggingface/hub/models--juncliu--llama-3.2-1b-ecg-flamingo-epoch-35/snapshots/cfcdf8f7141b729ae50da4e1ef4e3bdc2b638674/best_model.pt"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: juncliu checkpoint not found at:"
  echo "  $CKPT"
  exit 1
fi

CMD="python tslm_md/train_stage6.py --config configs/stage6_md_cot.yaml --starting-checkpoint $CKPT"

if [[ "${1:-}" == "nohup" ]]; then
  echo "==> launching in background, log -> train.log"
  echo "    PID will be printed below. Tail with: tail -f train.log"
  nohup $CMD > train.log 2>&1 &
  echo "training PID: $!"
  echo "Sleeping 60s, then showing tail of train.log..."
  sleep 60
  tail -40 train.log
else
  echo "==> launching in foreground (use 'bash scripts/launch_train.sh nohup' for background)"
  exec $CMD
fi
