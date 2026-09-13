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
#
# THE BUDGET IS PINNED BY CAPACITY, NOT BY TOKEN COUNT.  The first H100 attempt
# pinned every arm to the dense arm's 440M tokens, which is the right control when
# the arms differ only in precision and the wrong one the moment an arm changes
# width. Parameters grow as d^2 while bytes per parameter fall only 16x, so the
# matched-bytes width at depth 8 is 32x the parameters of dense, and a pinned token
# count hands it 0.33 tokens per parameter against dense's 10.5. The whole point of
# the accumulation-entropy argument is CAPACITY, and a starved model cannot exhibit
# capacity, so neither outcome there is attributable.
#
# But charging a binary arm 10.5 tokens per PARAMETER is also wrong, and it is the
# plan contradicting itself. Section 3.8 sets the capacity of a bf16 parameter at
# about USEFUL_BITS task-relevant bits and a binary parameter at 1. Using that ratio
# to claim the width advantage and then ignoring it when buying data charges the
# binary arm four times the data its parameters can absorb. One convention, used in
# both places: match TOKENS PER CAPACITY BIT.
#
#   dense d=512: 10.5 tokens/param / 4 useful bits = 2.625 tokens per capacity bit
#
# At USEFUL_BITS=4, and time measured against the ~7-minute dense d=512 run:
#
#   width   params    tokens   cost     time     bits/layer vs dense
#     512    41.9M      110M    0.2x    2 min      0.70x
#     896   106.4M      279M    1.6x   11 min      1.32x
#    1792   367.0M      963M   19.1x  134 min      2.85x
#    3584 1,350.6M    3,545M  259.2x   30.2 h      6.15x   <- over the cap
#
# This is one assumption used consistently and it is falsifiable in both directions:
# a reviewer who rejects 4 useful bits for the data budget has to reject the 6.15x
# bandwidth claim too. Set USEFUL_BITS=1 for the conservative reading, which is the
# old behaviour and puts even width 1792 at 8.9 hours.
RATIO="${RATIO:-10.5}"                 # the repo's dense tokens:params ratio
USEFUL_BITS="${USEFUL_BITS:-4}"        # task-relevant bits in a bf16 parameter (section 3.8)
MAX_COST="${MAX_COST:-4}"              # refuse an arm costing more than this many dense runs.
                                       # 4 buys the anchor plus two slope points in ~23 min.
                                       # Width 1792 is 26.8x (~3.1 h): buy it with MAX_COST=32
                                       # only after the 512 -> 896 slope comes back positive.
BIN_RATIO=$(python3 -c "print($RATIO / $USEFUL_BITS)")
if [ -z "${WIDTHS:-}" ]; then
    MATCHED=$(python -m scripts.binary_width_budget --depth "$DEPTH" --quiet 2>/dev/null | tail -1)
    if [ -z "$MATCHED" ]; then
        echo "FATAL: could not derive the matched-bytes width; refusing to type one" ; exit 1
    fi
    WIDTHS="512 $((MATCHED/4)) $((MATCHED/2))"   # cheapest first: the trend before the bill
fi

echo "############ B03 width sweep, depth=$DEPTH ############" | tee -a "$SWEEP_LOG"
echo "references (NEITHER re-run):" | tee -a "$SWEEP_LOG"
echo "  dense fp16 d=512      $DENSE_REF bpb   240.0 MiB" | tee -a "$SWEEP_LOG"
echo "  quantised bin d=512   $QUANT_REF bpb    15.0 MiB" | tee -a "$SWEEP_LOG"
echo "what each arm answers:" | tee -a "$SWEEP_LOG"
echo "  ANCHOR d=512 at the dense token budget:" | tee -a "$SWEEP_LOG"
echo "    vs $QUANT_REF -> does NATIVE beat QUANTISED at equal width and equal data?" | tee -a "$SWEEP_LOG"
echo "    vs $DENSE_REF -> how far native binary is from dense before any width is spent" | tee -a "$SWEEP_LOG"
echo "  SLOPE arms, all at equal tokens per capacity bit, so width is the only difference:" | tee -a "$SWEEP_LOG"
echo "    does bpb FALL as width rises? that is the v3 bet, and it is the only part of it" | tee -a "$SWEEP_LOG"
echo "    that is affordable. The matched-bytes point itself is a 3.5B-token run." | tee -a "$SWEEP_LOG"
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

# ---------------------------------------------------------------------------
# The batch plan, DERIVED. Two failures on the first H100 attempt, both from
# reusing the d=512 batch settings at seven times the width:
#
#   1. AssertionError at the grad-accum check. --model-dim was reaching
#      build_model_meta(12), so widening the arm widened the d12 REFERENCE too,
#      inflating D_REF ~21x and shrinking the auto batch from 2**19 to 2**17
#      while the micro-batch stayed at 128 x 2048. Fixed in base_train.py, but
#      this sweep now pins --total-batch-size outright so the auto path cannot
#      participate at all.
#   2. CUDA OOM at width 1792. Fixed properly in HammingAttention (each query
#      chunk is recomputed in backward instead of stored), and hedged here by
#      scaling the micro-batch down as the width goes up.
#
# Activation bytes are linear in width, so hold width x device_batch constant.
# Then clamp so the micro-batch divides the total batch on however many GPUs
# are present, which is what assertion 1 was really about.
# ---------------------------------------------------------------------------
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
BASE_DBS="${BASE_DBS:-128}"          # the value validated at width 512
WORLD="${WORLD:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 1)}"
[ "$WORLD" -ge 1 ] 2>/dev/null || WORLD=1
TOK="${TOKENIZER_DIR:-tokenizer}"

budget () { python3 -m scripts.code_head_budget --depth "$DEPTH" --ratio "$2" \
                --model-dim "$1" --tokenizer-dir "$TOK" 2>/dev/null; }

REF_TOKENS=$(budget 512 "$RATIO")     # the DENSE arm's budget, which produced $DENSE_REF
D_REF_TOKENS=$(python3 -m scripts.code_head_budget --depth 12 --ratio "$RATIO" \
    --tokenizer-dir "$TOK" 2>/dev/null)
if [ -z "$REF_TOKENS" ] || [ -z "$D_REF_TOKENS" ]; then
    echo "FATAL: could not derive the token budgets; refusing to type one" ; exit 1
fi

# The ANCHOR arm is deliberately iso-TOKEN, not iso-capacity: it is the only arm
# directly comparable to the two numbers already measured, so it answers "does native
# beat quantised, and how far is it from dense" for about seven minutes.
[ "${ANCHOR:-1}" = "1" ] && WIDTHS="512@$REF_TOKENS $WIDTHS"

# The plan, printed and costed BEFORE anything launches. The first attempt ran a
# 235-minute arm that nobody had costed, which is how the data-ratio confound got in:
# nothing in the loop was forced to look at tokens per parameter.
echo "plan: dense ratio $RATIO, binary ratio $BIN_RATIO (= $RATIO / $USEFUL_BITS useful bits)" | tee -a "$SWEEP_LOG"
printf "%8s %14s %10s %9s %9s  %s\n" width tokens tok/param cost est_min role | tee -a "$SWEEP_LOG"
for SPEC in $WIDTHS; do
    W="${SPEC%@*}"; T="${SPEC#*@}"; ROLE="slope (iso-capacity)"
    if [ "$T" = "$SPEC" ]; then T=$(budget "$W" "$BIN_RATIO"); else ROLE="anchor (iso-token)"; fi
    python3 -c "
import sys
w, t, ref = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
sp = 12*w*w*8 + 32768*w
c = t / ref * (w / 512) ** 2
print(f'{w:>8}{t:>15,}{t/sp:>10.2f}{c:>8.1f}x{7*c:>9.0f}  {sys.argv[4]}')" \
        "$W" "$T" "$REF_TOKENS" "$ROLE" | tee -a "$SWEEP_LOG"
done

for NAT in ${NATIVE_MODES:-0 1}; do
for SPEC in $WIDTHS; do
    W="${SPEC%@*}"
    ARM_TOKENS="${SPEC#*@}"
    [ "$ARM_TOKENS" = "$SPEC" ] && ARM_TOKENS=$(budget "$W" "$BIN_RATIO")
    COST=$(python3 -c "print(f'{int($ARM_TOKENS)/int($REF_TOKENS)*($W/512)**2:.2f}')")
    if [ "$(python3 -c "print(1 if float($COST) > float($MAX_COST) else 0)")" = "1" ]; then
        echo "[skip] width $W costs ${COST}x the reference run, over MAX_COST=$MAX_COST." | tee -a "$SWEEP_LOG"
        echo "       Set MAX_COST higher to run it, but read the header first." | tee -a "$SWEEP_LOG"
        continue
    fi
    # The batch plan is per-arm because the token budget is now per-arm.
    TBS=$(python3 -c "
import math, sys
T, D = int(sys.argv[1]), int(sys.argv[2])
print(2 ** round(math.log2(2**19 * (T / D) ** 0.383)))" "$ARM_TOKENS" "$D_REF_TOKENS")
    DBS=$(python3 -c "
import math, sys
w, base, total, seq, world = (int(x) for x in sys.argv[1:])
cap = max(1, base * 512 // w)                     # activation bytes are linear in width
fit = max(1, total // (seq * world))              # micro-batch must divide the total batch
print(max(1, 2 ** int(math.floor(math.log2(min(cap, fit))))))" \
        "$W" "$BASE_DBS" "$TBS" "$MAX_SEQ_LEN" "$WORLD")
    GA=$(( TBS / (DBS * MAX_SEQ_LEN * WORLD) ))
    echo "==== width $W native $NAT | ${ARM_TOKENS} tokens | cost ${COST}x | batch $TBS = $DBS x $MAX_SEQ_LEN x $WORLD x $GA ====" | tee -a "$SWEEP_LOG"
    MODEL_DIM="$W" BINARY_NATIVE="$NAT" DEVICE_BATCH_SIZE="$DBS" \
        TOTAL_BATCH_SIZE="$TBS" MAX_SEQ_LEN="$MAX_SEQ_LEN" TARGET_TOKENS="$ARM_TOKENS" \
        TAG_SUFFIX="t$((ARM_TOKENS / 1000000))M" \
        bash scripts/b01_binary_ladder.sh --rungs R5 --force "$DEPTH"
done
done

echo "############ B03 done ############" | tee -a "$SWEEP_LOG"
echo "dense fp16 d=512 reference: $DENSE_REF   quantised binary d=512: $QUANT_REF" | tee -a "$SWEEP_LOG"
grep -hE "^==== width|Minimum validation bpb" "$SWEEP_LOG" | tail -40
