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
CHRONOS="third_party/OpenTSLM/src/opentslm/model/encoder/Chronos2Encoder.py"
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

echo "==> patching $CHRONOS (move projection/norm to device)"
if [ ! -f "$CHRONOS" ]; then
  echo "    ERROR: $CHRONOS not found"
  exit 1
fi
# Chronos2Encoder only moves self.chronos_model to device, leaving
# self.projection / self.output_norm / self.output_dropout on CPU.
# Insert a line BEFORE the final print() that moves them.
if ! grep -q '# device-move patch applied' "$CHRONOS"; then
  python - <<'PY'
import io, os, re
p = "third_party/OpenTSLM/src/opentslm/model/encoder/Chronos2Encoder.py"
with open(p) as f:
    src = f.read()
marker = 'print(f"Chronos2Encoder initialized:'
insert = (
    "if device is not None:  # device-move patch applied\n"
    "            self.projection = self.projection.to(device)\n"
    "            self.output_norm = self.output_norm.to(device)\n"
    "            self.output_dropout = self.output_dropout.to(device)\n"
    "        "
)
src = src.replace(marker, insert + marker, 1)
with open(p, "w") as f:
    f.write(src)
print("    OK — device-move patch applied")
PY
else
  echo "    already patched"
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
