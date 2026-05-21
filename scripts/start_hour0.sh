#!/usr/bin/env bash
# Kick off the real pipeline once the dry-run is green.
#
# Stub: implement during hour 0 once we know which steps are auto-pipelinable.
# Will likely orchestrate:
#   1. aws s3 cp s3://${S3_BUCKET}/misato/MD.hdf5 data/misato/
#   2. python scripts/preprocess_features.py
#   3. python scripts/train_gbm_baseline.py  (R1 gate)
#   4. python scripts/build_training_targets.py
#   5. python tslm_md/train_stage6.py
#   6. python scripts/train_cmapss_fallback.py &   (background insurance)

set -euo pipefail
echo "TODO(hour 0): implement orchestration"
exit 1
