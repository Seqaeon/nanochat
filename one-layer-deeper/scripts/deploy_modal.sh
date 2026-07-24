#!/bin/sh
set -eu

echo "[modal-deploy] deploying one-layer-benchmark-runner"
modal deploy modal_runner.py

echo "[modal-deploy] deployment succeeded; starting Easy e1 smoke evaluation"
python modal_runner.py smoke \
  --submission-file submissions/baseline_adamw/submission.py \
  --manifest-filename h100_easy_e1.json

echo "[modal-deploy] smoke evaluation succeeded"
