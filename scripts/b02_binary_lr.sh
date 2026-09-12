#!/usr/bin/env bash
# ============================================================================
# B02: learning-rate search for the binary arm.
#
# WHY THIS EXISTS
#   The first R5 run compared a DENSE model at dense-tuned LRs against a BINARY
#   model at those same LRs, which is not an architectural comparison. It landed
#   at +0.6418 bpb (1.5993 against dense 0.9575) and its 200-step loss deltas went
#   UP three separate times late in training:
#       -4.815 +0.139 -0.141 -0.521 +0.116 +0.230 -0.643 -0.061 +0.003
#   A converging model does not do that. Meanwhile O6, at plain AdamW 1e-3 on
#   everything, reached +0.1957. The optimiser configuration, not the
#   architecture, is the leading explanation for the difference.
#
#   O2's one durable finding says the same thing from the other side: binary
#   needs materially different LRs, because only sign CROSSINGS change the
#   function, so a latent weight must traverse the whole STE clip window before
#   the function moves at all. Nominal LR and effective step size in function
#   space are decoupled in a way they are not for dense.
#
#   Muon carries the matrices and is the prime suspect: it orthogonalises the
#   update to a LATENT weight whose only job is to cross zero, which is not
#   obviously the right thing to do.
#
#   An arm is ~7 minutes at depth 8 on an H100, so this is cheap.
#
#   bash scripts/b02_binary_lr.sh 8
#   MATRIX_LRS="0.02 0.005" bash scripts/b02_binary_lr.sh 8
# ============================================================================
set -o pipefail
DEPTH="${1:-8}"
MATRIX_LRS="${MATRIX_LRS:-0.08 0.04 0.02 0.01 0.005}"
export OUT_BASE="${OUT_BASE:-out/b02_binary_lr}"
export SEEDS=1
export SWEEP_LOG="${SWEEP_LOG:-b02_binary_lr.log}"

echo "############ B02 binary LR search, depth=$DEPTH ############" | tee -a "$SWEEP_LOG"
echo "reference: dense R0 = 0.957500 bpb; R5 at dense LRs = 1.599251 (+0.6418)" | tee -a "$SWEEP_LOG"
for lr in $MATRIX_LRS; do
    echo "==== matrix_lr $lr ====" | tee -a "$SWEEP_LOG"
    MATRIX_LR="$lr" bash scripts/b01_binary_ladder.sh --rungs R5 --force "$DEPTH"
done
echo "############ B02 done ############" | tee -a "$SWEEP_LOG"
grep -hE "^==== matrix_lr|Minimum validation bpb" "$SWEEP_LOG" | tail -40
