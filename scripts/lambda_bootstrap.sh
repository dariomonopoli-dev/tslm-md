#!/usr/bin/env bash
# Run on the Lambda Labs box (Ubuntu 22.04, CUDA pre-installed):
#   bash ~/lambda_bootstrap.sh
# Idempotent — safe to re-run after a partial failure.
set -euo pipefail

cd "$HOME/tslm-md"

# Install uv (manages Python 3.12 + uv_build backend that OpenTSLM requires)
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# shellcheck disable=SC1091
source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"

sudo apt-get update -qq
sudo apt-get install -y -qq tmux htop

# Fresh venv on Python 3.12 (OpenTSLM pyproject demands >=3.12)
rm -rf .venv
uv venv --python 3.12
# shellcheck disable=SC1091
source .venv/bin/activate

uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install -e OpenTSLM
uv pip install wandb

# HF auth — token either pre-placed at ~/.cache/huggingface/token or via $HF_TOKEN
if [[ ! -s "$HOME/.cache/huggingface/token" ]]; then
    if [[ -n "${HF_TOKEN:-}" ]]; then
        mkdir -p "$HOME/.cache/huggingface"
        printf '%s' "$HF_TOKEN" > "$HOME/.cache/huggingface/token"
        chmod 600 "$HOME/.cache/huggingface/token"
    else
        echo "No HF token. scp ~/.cache/huggingface/token from laptop, or set HF_TOKEN."
        exit 1
    fi
fi

python -c "from huggingface_hub import whoami; print('HF user:', whoami()['name'])"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi -L

cat <<'NEXT'

Bootstrap done. To train v1b:

  tmux new -s train
  source ~/tslm-md/.venv/bin/activate
  export OPENTSLM_MISATO_DATA=$HOME/tslm-md/preprocessed
  cd ~/tslm-md/OpenTSLM
  python train_misato.py --variant v1b --epochs 5 --batch-size 16 --lambda-reg 0.5 \
      --wandb-project misato-opentslm --wandb-tags v1b lambda a100 \
      2>&1 | tee ~/v1b_run.log

Detach: Ctrl-b d   |   Reattach: ssh lambda; tmux a -t train
NEXT
