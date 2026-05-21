#!/usr/bin/env bash
# Apply all known patches to the liu-jc/OpenTSLM Chronos branch.
# Idempotent — safe to re-run.
#
# Patches:
#   1) SimpleNamespace.requires_grad_(True) bug in OpenTSLMFlamingo.__init__
#      (vision_encoder is wrapped in SimpleNamespace; need to dereference .visual)
#   2) requires-python = ">=3.12" -> ">=3.11" so it installs on noctua

set -euo pipefail

FLAM="third_party/OpenTSLM/src/opentslm/model/llm/OpenTSLMFlamingo.py"
TSF="third_party/OpenTSLM/src/opentslm/model/llm/TimeSeriesFlamingoWithTrainableEncoder.py"
PYPROJ="third_party/OpenTSLM/pyproject.toml"

echo "==> patching $FLAM (init)"
if [ ! -f "$FLAM" ]; then
  echo "    ERROR: $FLAM not found. Did you clone third_party/OpenTSLM?"
  exit 1
fi
if grep -q 'model\.vision_encoder\.requires_grad_(True)' "$FLAM"; then
  sed -i.bak 's/model\.vision_encoder\.requires_grad_(True)/model.vision_encoder.visual.requires_grad_(True)/' "$FLAM"
  echo "    OK — init patch applied"
else
  echo "    already patched (or upstream changed)"
fi

echo "==> patching $TSF (forward)"
if [ ! -f "$TSF" ]; then
  echo "    ERROR: $TSF not found"
  exit 1
fi
if grep -q 'vision_x = self\.vision_encoder(vision_x)' "$TSF"; then
  sed -i.bak 's/vision_x = self\.vision_encoder(vision_x)/vision_x = self.vision_encoder.visual(vision_x)/' "$TSF"
  echo "    OK — forward patch applied"
else
  echo "    already patched (or upstream changed)"
fi

echo "==> patching $PYPROJ for Python 3.11+"
if [ -f "$PYPROJ" ] && grep -q 'requires-python = ">=3.12"' "$PYPROJ"; then
  sed -i.bak 's/requires-python = ">=3.12"/requires-python = ">=3.11"/' "$PYPROJ"
  echo "    OK — Python requirement loosened"
else
  echo "    already patched"
fi

echo
echo "Verify with:"
echo "  grep -n 'vision_encoder' $FLAM $TSF | head -10"
