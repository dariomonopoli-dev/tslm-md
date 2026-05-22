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

echo "==> 2/5 Cloning third_party deps (OpenTSLM Chronos fork + MiSaTo-dataset)"
# Per OpenTSLM team's direct recommendation: use the liu-jc fork's
# add-chronos2-encoder branch, which uses Amazon Chronos-2 as the time-series
# encoder and pairs with the better juncliu/llama-3.2-1b-ecg-flamingo-epoch-35
# checkpoint. Falls back to upstream main if the fork is unreachable.
mkdir -p third_party
if [ ! -d third_party/OpenTSLM ]; then
  git clone --depth 1 -b add-chronos2-encoder \
    https://github.com/liu-jc/OpenTSLM third_party/OpenTSLM \
  || git clone --depth 1 https://github.com/StanfordBDHG/OpenTSLM third_party/OpenTSLM
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

# The Chronos branch's pyproject pins requires-python = ">=3.12" but the code
# itself runs on 3.11. Auto-patch on systems where Python is 3.11.x so we
# don't hit the version wall.
PY_MINOR=$(python -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MINOR" -lt 12 ] && [ -f third_party/OpenTSLM/pyproject.toml ]; then
  echo "    Python is 3.${PY_MINOR}; patching OpenTSLM pyproject requires-python ->= 3.11"
  sed -i.bak 's/requires-python = ">=3.12"/requires-python = ">=3.11"/' \
    third_party/OpenTSLM/pyproject.toml
fi

pip install -e third_party/OpenTSLM

# Bug in chronos branch: model.vision_encoder is wrapped in a SimpleNamespace,
# but OpenTSLMFlamingo.__init__ calls .requires_grad_(True) on it directly.
# The actual encoder lives at .visual. Auto-patch.
FLAM_FILE="third_party/OpenTSLM/src/opentslm/model/llm/OpenTSLMFlamingo.py"
if [ -f "$FLAM_FILE" ] && grep -q 'model\.vision_encoder\.requires_grad_(True)' "$FLAM_FILE"; then
  echo "    Patching SimpleNamespace.requires_grad_ bug in OpenTSLMFlamingo.py"
  sed -i.bak 's/model\.vision_encoder\.requires_grad_(True)/model.vision_encoder.visual.requires_grad_(True)/' "$FLAM_FILE"
fi
echo "    OK."

echo "==> 5/5 HuggingFace login + pre-warm cache"
echo "    You will be prompted for a HF token (read scope)."
echo "    Get one at https://huggingface.co/settings/tokens"

# HF Hub 1.x renamed `huggingface-cli` -> `hf`. Pick whichever is on PATH.
HF_CLI=""
if command -v hf >/dev/null 2>&1; then
  HF_CLI="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI="huggingface-cli"
else
  echo "    Installing huggingface_hub[cli]..."
  pip install -U "huggingface_hub[cli]"
  HF_CLI="hf"
fi
echo "    Using HF CLI: $HF_CLI"

if [ "$HF_CLI" = "hf" ]; then
  if ! hf auth whoami >/dev/null 2>&1; then
    hf auth login
  fi
else
  if ! huggingface-cli whoami >/dev/null 2>&1; then
    huggingface-cli login
  fi
fi

echo
echo "==> Pre-warming HF cache"
echo "    1) Llama-3.2-1B backbone (gated — accept Meta license on HF first)"
$HF_CLI download meta-llama/Llama-3.2-1B
echo "    2) Chronos-2 time-series encoder (Amazon, public)"
$HF_CLI download amazon/chronos-2 || echo "    (chronos-2 may also auto-download on first model init)"
echo "    3) Pretrained adapter checkpoint (juncliu Chronos-encoder, public)"
$HF_CLI download juncliu/llama-3.2-1b-ecg-flamingo-epoch-35

echo
echo "============================================================"
echo "✅ Setup complete."
echo
echo "Activate the venv:        source .venv/bin/activate"
echo "Run the dry-run:          python scripts/dry_run.py"
echo "============================================================"
