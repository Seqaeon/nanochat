#!/usr/bin/env bash
# ============================================================================
# B03: the v3 bet. Binary at MATCHED BYTES, which means WIDE.
#
# WHY THIS SWEEP EXISTS
#   Phase 1 built the wrong model and measured it carefully. A d=512 W1A1
#   transformer is dense's architecture with quantised weights: it pays
#   binarisation's accumulation cost and declines the width that is supposed to
#   pay for it. It cost +0.62 bpb and the gap GREW with data. That is the single
#   worst point in the design space, and section 3.8 v3 says why with a formula
#   that also predicts where the good point is.
#
#   A binary neuron summing n terms emits an output whose entropy is
#       H = 0.5*log2(2*pi*e*n) - 1      (measured: 5.54 at n=512, 7.55 at n=8192)
#   Per NEURON binary always loses to fp's ~7.9. Per LAYER it wins, because equal
#   bytes buys more neurons:
#
#       dense fp16 d=512        512 x 7.91 =  4,050 bits/layer   1.00x
#       binary d=512  (Phase 1) 512 x 5.55 =  2,840 bits/layer   0.70x  <- lost
#       binary d=3584 (matched) 3584 x 6.95 = 24,912 bits/layer   6.15x  <- the bet
#
#   The 0.70x is why Phase 1 failed, and the same formula says 6.15x is available
#   for the same memory. That is the whole experiment.
#
# WHAT IS AND IS NOT BEING TESTED
#   WIDTH is the accumulation argument. NATIVE is the consistency argument (O5's
#   negative in-situ costs: an fp component feeding binary consumers is worse than
#   a consistent binary one). They are separate claims and this sweep crosses them
#   so they can be attributed separately:
#       BINARY_NATIVE=0  quantised block: fp softmax attention, fp residual
#       BINARY_NATIVE=1  BinaryBlock: Hamming retrieval, binary KV-memory FFN,
#                        bundled residual, no normalisation
#
# NO DENSE ARM. The reference already exists and is not re-run:
#       dense d=512 depth 8, 240.0 MiB -> val bpb 0.957500   (b01, seed 1)
#   Every arm here is pinned to the same token budget, so the comparison holds.
#
#   bash scripts/b03_binary_width.sh 8
#   WIDTHS="3584 2048 1024" bash scripts/b03_binary_width.sh 8
# ============================================================================
set -o pipefail
DEPTH="${1:-8}"
export OUT_BASE="${OUT_BASE:-out/b03_binary_width}"
export SEEDS="${SEEDS:-1}"
export SWEEP_LOG="${SWEEP_LOG:-b03_binary_width.log}"
export MATRIX_LR="${MATRIX_LR:-0.08}"   # B02: monotone to the top of a 0.005-0.08 grid
# Two reference points already measured. NEITHER is re-run.
DENSE_REF="${DENSE_REF:-0.957500}"     # b01: dense fp16 d=512 depth 8, 240.0 MiB
QUANT_REF="${QUANT_REF:-1.578246}"     # b02: QUANTISED binary d=512, matrix_lr 0.08
# Native only. The quantised arm is the thing we already know loses (+0.62), and its
# d=512 point above is what the d=512 NATIVE arm below is compared against.
NATIVE_MODES="${NATIVE_MODES:-1}"

# Derived, never typed. A hand-matched width is right once and then silently
# compares two budgets.
if [ -z "${WIDTHS:-}" ]; then
    MATCHED=$(python -m scripts.binary_width_budget --depth "$DEPTH" --quiet 2>/dev/null | tail -1)
    if [ -z "$MATCHED" ]; then
        echo "FATAL: could not derive the matched-bytes width; refusing to type one" ; exit 1
    fi
    # The matched point, two rungs below it for the width trend, and 512 -- the width
    # where the QUANTISED number is already known, so native-vs-quantised is
    # attributable without running a single quantised arm.
    WIDTHS="$MATCHED $((MATCHED/2)) $((MATCHED/4)) 512"
fi

echo "############ B03 width sweep, depth=$DEPTH ############" | tee -a "$SWEEP_LOG"
echo "references (NEITHER re-run):" | tee -a "$SWEEP_LOG"
echo "  dense fp16 d=512      $DENSE_REF bpb   240.0 MiB" | tee -a "$SWEEP_LOG"
echo "  quantised bin d=512   $QUANT_REF bpb    15.0 MiB" | tee -a "$SWEEP_LOG"
echo "what each arm answers:" | tee -a "$SWEEP_LOG"
echo "  native d=512   vs $QUANT_REF -> does NATIVE beat QUANTISED at equal width?" | tee -a "$SWEEP_LOG"
echo "  native d=3584  vs $DENSE_REF -> the v3 bet: does binary beat dense at equal BYTES?" | tee -a "$SWEEP_LOG"
echo "  the 896/1792 rungs give the width trend between them" | tee -a "$SWEEP_LOG"
echo "widths: $WIDTHS   matrix_lr: $MATRIX_LR" | tee -a "$SWEEP_LOG"
python - "$DEPTH" $WIDTHS <<'PYEOF' | tee -a "$SWEEP_LOG"
import math, sys
d_list = [int(x) for x in sys.argv[2:]]
FP, DW = 7.91, 512
acc = lambda n: 0.5*math.log2(2*math.pi*math.e*n) - 1.0
print(f"{'width':>7}{'bits/neuron':>13}{'bits/layer':>13}{'vs dense layer':>16}")
print(f"{DW:>7}{FP:>13.2f}{DW*FP:>13,.0f}{1.0:>15.2f}x   <- dense fp16")
for d in d_list:
    print(f"{d:>7}{acc(d):>13.2f}{d*acc(d):>13,.0f}{d*acc(d)/(DW*FP):>15.2f}x")
PYEOF

for NAT in ${NATIVE_MODES:-0 1}; do
for W in $WIDTHS; do
    echo "==== width $W native $NAT ====" | tee -a "$SWEEP_LOG"
    MODEL_DIM="$W" BINARY_NATIVE="$NAT" bash scripts/b01_binary_ladder.sh --rungs R5 --force "$DEPTH"
done
done

echo "############ B03 done ############" | tee -a "$SWEEP_LOG"
echo "dense fp16 d=512 reference: $DENSE_REF   quantised binary d=512: $QUANT_REF" | tee -a "$SWEEP_LOG"
grep -hE "^==== width|Minimum validation bpb" "$SWEEP_LOG" | tail -40
