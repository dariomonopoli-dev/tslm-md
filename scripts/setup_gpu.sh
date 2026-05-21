#!/usr/bin/env bash
# One-shot environment setup on any single-GPU CUDA box.
# Tested on: vast.ai A100 80GB, vast.ai H100 80GB, local A30 24GB.
# Idempotent — safe to re-run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> 1/5 Creating Python venv (.venv)"
# On vast.ai's PyTorch image, system PyTorch is already installed — inherit it
# with --system-site-packages so we don't re-download ~2 GB.
VENV_FLAGS=""
if python3 -c "import torch" 2>/dev/null; then
  echo "    System torch detected — venv will inherit system site-packages"
  VENV_FLAGS="--system-site-packages"
fi
if [ ! -d .venv ]; then
  # shellcheck disable=SC2086
  python3.10 -m venv $VENV_FLAGS .venv 2>/dev/null || python3 -m venv $VENV_FLAGS .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
echo "    Python: $(python --version)"
if python -c "import torch; print(f'    torch: {torch.__version__}, cuda: {torch.cuda.is_available()}')"; then
  :
fi

echo "==> 2/5 Cloning third_party deps (OpenTSLM + MiSaTo-dataset)"
mkdir -p third_party
if [ ! -d third_party/OpenTSLM ]; then
  git clone --depth 1 https://github.com/StanfordBDHG/OpenTSLM third_party/OpenTSLM
fi
if [ ! -d third_party/MiSaTo-dataset ]; then
  git clone --depth 1 https://github.com/sab148/MiSaTo-dataset third_party/MiSaTo-dataset
fi

echo "==> 3/5 Installing this package (tslm_md)"
pip install --upgrade pip wheel
pip install -e .

echo "==> 4/5 Installing OpenTSLM"
echo "    THIS IS THE KEY DEP-HELL CHECK — pulls open_flamingo as transitive dep."
echo "    If this fails: stop, diagnose, do not proceed to dry_run."
pip install -e third_party/OpenTSLM
echo "    OK."

echo "==> 5/5 HuggingFace login + pre-warm cache"
echo "    You will be prompted for a HF token (read scope)."
echo "    Get one at https://huggingface.co/settings/tokens"
if ! huggingface-cli whoami >/dev/null 2>&1; then
  huggingface-cli login
fi

echo
echo "==> Pre-warming HF cache (Llama-3.2-1B + OpenTSLM stage-5 checkpoint)"
huggingface-cli download meta-llama/Llama-3.2-1B
huggingface-cli download OpenTSLM/llama-3.2-1b-ecg-flamingo

echo
echo "============================================================"
echo "✅ Setup complete."
echo
echo "Activate the venv:        source .venv/bin/activate"
echo "Run the dry-run:          python scripts/dry_run.py"
echo "============================================================"
