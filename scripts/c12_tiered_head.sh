#!/usr/bin/env bash
# ============================================================================
# C12: tiered-capacity head. Which words get how many dimensions.
#
# WHAT EVERY EARLIER SWEEP ASSUMED WITHOUT TESTING
#   c05 through c11 gave all V words the same per-word capacity and varied the
#   STRUCTURE around it: interaction order, block count, block membership,
#   private-versus-shared split, clustering. At V=131,072 the top 1,024 tokens
#   carry 65.7% of the probability mass and the bottom 65,536 carry 1.7%, so
#   uniform capacity spends the same dimensions separating the commonest words as
#   the rarest. Nothing in eleven sweeps varied that.
#
#   Scored offline against the trained dense head at the exact cost of the arms
#   already run (scripts/head_oracle.py, frequency-weighted captured energy over a
#   cost-matched pure low-rank head):
#
#     32 equal blocks, uniform capacity (Monarch, as trained)   +0.0055
#     32 equal blocks, uniform capacity, clustered              +0.0364
#     5 tiers, capacity solved                                  +0.1323
#
#   Twenty-four times the structural work, on the axis nobody varied.
#
# WHY THERE IS NO SHARED RESIDUAL HERE
#   A shared rank-r term costs r*(d + V) because it is charged against every word;
#   a tier's capacity is charged only against its own words. Dropping the residual
#   in favour of tier capacity was worth +0.0202 capture AND a third of the budget,
#   so this head has none. It also has no block-diagonal factor and no feature
#   permutation: the tiers are T ordinary dense GEMMs, which is why none of the
#   Monarch kernel problems (the clone, the XBLOCK assertion, K=32 tensor cores)
#   appear.
#
# THE ARMS, AND WHAT THEY SETTLE
#   The capacities are SOLVED, not swept: captured energy is additive across
#   disjoint tiers and concave in each tier's capacity, so the optimum is a greedy
#   over marginal eigenvalue per MAC. scripts/solve_tiers.py returns it in seconds.
#
#   The one real uncertainty is the metric. Weighting word w by p(w) is correct to
#   second order, but the two weightings disagree sharply about the tail:
#
#     frequency-weighted   caps [512, 512, 511, 480,  18]   +0.1323
#     unweighted           caps [187, 240, 293, 280, 233]   +0.0109
#
#   The whole 24x comes from the weighting permitting 65,536 words to be cut to 18
#   dimensions. So both are trained. WEIGHTED >> UNWEIGHTED means the tail really
#   is nearly free and the result is large; WEIGHTED ~ UNWEIGHTED means p(w)
#   over-discounts the tail and the honest gain is the modest 1.8x. Either way the
#   answer says which metric to design against, which outlives this sweep.
#
#   ID_ORDER is the control: the same solved capacities applied in token-id order
#   instead of frequency order. If it matches WEIGHTED, the gain is the tier
#   structure and not the frequency ordering, and the story is wrong.
#
#   bash scripts/c12_tiered_head.sh                        # d8, V=131,072
#   HEAD_CKPT=head_d12_v131k.pt bash scripts/c12_tiered_head.sh 12
#   BOUNDS="512,2048,8192,32768" bash scripts/c12_tiered_head.sh 8
# ============================================================================
set -o pipefail

FORCE=0
SEEDS=1
RUN_WEIGHTED=1
RUN_UNWEIGHTED=1
RUN_IDORDER=1
RUN_LOWRANK=0
DEPTHS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)          FORCE=1; shift ;;
        --seeds)          SEEDS="$2"; shift 2 ;;
        --weighted-only)  RUN_UNWEIGHTED=0; RUN_IDORDER=0; shift ;;
        --no-control)     RUN_IDORDER=0; shift ;;
        --with-lowrank)   RUN_LOWRANK=1; shift ;;
        [0-9]*)           DEPTHS+=("$1"); shift ;;
        *) echo "unknown arg: $1"
           echo "usage: $0 [--force] [--seeds N] [--weighted-only] [--no-control]"
           echo "       [--with-lowrank] [DEPTH ...]"
           exit 1 ;;
    esac
done
[ ${#DEPTHS[@]} -eq 0 ] && DEPTHS=(8)

VOCAB="${VOCAB:-131072}"
BOUNDS="${BOUNDS:-1024,4096,16384,65536}"
# The budget, from the Monarch arm already trained: d*M + V*m1 + r*(d+V).
REF_M="${REF_M:-1024}"; REF_M1="${REF_M1:-32}"; REF_RANK="${REF_RANK:-224}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
OUT_BASE="${OUT_BASE:-out/c12_tiered_head}"
RANK_CONTEXTS="${RANK_CONTEXTS:-16384}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
# Dense head the capacities are solved against. Same vocabulary as the runs.
HEAD_CKPT="${HEAD_CKPT:-head_d8_v131k.pt}"

if [ "$VOCAB" -eq 131072 ]; then
    TOK="${TOKENIZER_DIR:-${TOKENIZER_DIR_131K:-tokenizer_131k}}"
else
    TOK="${TOKENIZER_DIR:-tokenizer}"
fi
if ! python3 -m scripts.ensure_tokenizer --vocab-size "$VOCAB" --tokenizer-dir "$TOK" \
        --data-dir "${DATA_DIR:-data}" ${MAX_SHARDS:+--max-shards "$MAX_SHARDS"}; then
    echo "could not prepare the tokenizer at '${TOK}'; nothing was run."
    exit 1
fi
FREQ="${TOK}/freq_table.pt"
for f in "$HEAD_CKPT" "$FREQ"; do
    [ -f "$f" ] || { echo "missing '${f}'. The capacities are solved against a dense"
                     echo "head at V=${VOCAB} and its frequency table; extract the head"
                     echo "on the training box (see scripts/head_oracle.py) first."
                     exit 1; }
done
mkdir -p "$OUT_BASE"

# Solved, never typed. A hand-set allocation is right once and then silently
# compares two different budgets the moment M, m1, r or the boundaries move.
solve_caps() {   # $1 = "weighted" | "unweighted"
    local extra=""
    [ "$1" = "weighted" ] && extra="--freq-table $FREQ"
    python3 -m scripts.solve_tiers "$HEAD_CKPT" --bounds "$BOUNDS" \
        --match-monarch "$REF_M" "$REF_M1" "$REF_RANK" --vocab-size "$VOCAB" $extra \
        2>/dev/null | grep -- "--sch-tier-caps" | awk '{print $2}'
}
CAPS_W=$(solve_caps weighted)
CAPS_U=$(solve_caps unweighted)
[ -n "$CAPS_W" ] || { echo "solve_tiers produced no capacities; run it directly to see why."; exit 1; }

done_already() {
    [ "$FORCE" -eq 1 ] && return 1
    python3 -c "
import json,sys
sys.exit(0 if '$1' in json.load(open('$STATE')).get('completed',{}) else 1)" 2>/dev/null
}
mark_done() {
    python3 -c "
import json,datetime
s=json.load(open('$STATE'))
s.setdefault('completed',{})['$1']=datetime.datetime.now().isoformat()
json.dump(s,open('$STATE','w'),indent=2)"
}

for DEPTH in "${DEPTHS[@]}"; do

MODEL_DIM=$(( ((DEPTH * ASPECT_RATIO + 127) / 128) * 128 ))
LOGFILE="${SWEEP_LOG:-${OUT_BASE}/c12_d${DEPTH}.log}"
STATE="${OUT_BASE}/c12_state_d${DEPTH}.json"
[ "$FORCE" -eq 1 ] && rm -f "$STATE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"

TARGET_TOKENS="${TARGET_TOKENS:-$(python3 -m scripts.code_head_budget --depth "$DEPTH" --ratio "${RATIO:-10.5}" --tokenizer-dir "$TOK")}"

COMMON="--device-batch-size $DEVICE_BATCH_SIZE --total-batch-size -1 \
  --use-onecycle 0 --log-every ${LOG_EVERY:-200} --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir $TOK \
  --sequence-len ${SEQ_LEN:-2048} --target-tokens $TARGET_TOKENS \
  --target-param-data-ratio -1 \
  --warmup-ratio 0.005 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --research-dim -1 --target-active-params 0 \
  --save-every 200 --eval-every -1 \
  --sch-decile-metrics 1 --sch-eval-steps ${EVAL_STEPS:-100}"
[ -n "${MAX_SHARDS:-}" ] && COMMON="$COMMON --max-shards $MAX_SHARDS"
PROBE="--sch-phi-dtype fp32 --sch-rank-probe $RANK_CONTEXTS --sch-bias 1"
TIER="--models base --use-code-head 1 --sch-head-type tiered --sch-tier-bounds $BOUNDS $PROBE"

run() {
    local tag="$1"; shift
    for s in $(seq 1 "$SEEDS"); do
        local t="${tag}_s${s}"
        if done_already "$t"; then echo "SKIP  $t (already completed)"; continue; fi
        echo ""
        echo "--- $t  (depth $DEPTH, V=${VOCAB}, device-batch ${DEVICE_BATCH_SIZE}) ---"
        local dir="${OUT_BASE}/d${DEPTH}/${t}"
        [ "$FORCE" -eq 1 ] && rm -rf "$dir"
        if bash scripts/research_sweep.sh $COMMON --out-dir "$dir" --seed "$s" \
               "$@" "$DEPTH" 2>&1 | tee -a "$LOGFILE"; then
            mark_done "$t"; echo "OK    $t"
        else
            echo "FAIL  $t (will retry on the next invocation)"
        fi
    done
}

echo "============================================================"
echo "  C12: tiered-capacity head, V=${VOCAB}, depth ${DEPTH}, d=${MODEL_DIM}"
echo "  budget from the Monarch arm M=${REF_M} m1=${REF_M1} r=${REF_RANK}"
echo "  tier boundaries ${BOUNDS}   solved against ${HEAD_CKPT}"
echo "    frequency-weighted caps  ${CAPS_W}"
echo "    unweighted caps          ${CAPS_U}"
echo "  target tokens ${TARGET_TOKENS}   device-batch ${DEVICE_BATCH_SIZE}"
echo "============================================================"

[ "$RUN_WEIGHTED" -eq 1 ] && run "TIER_weighted" $TIER \
    --sch-tier-caps "$CAPS_W" --sch-tier-order freq
[ "$RUN_UNWEIGHTED" -eq 1 ] && run "TIER_unweighted" $TIER \
    --sch-tier-caps "$CAPS_U" --sch-tier-order freq
# Same capacities, wrong order: separates "tiers help" from "frequency helps".
[ "$RUN_IDORDER" -eq 1 ] && run "TIER_weighted_idorder" $TIER \
    --sch-tier-caps "$CAPS_W" --sch-tier-order none
if [ "$RUN_LOWRANK" -eq 1 ]; then
    R=$(python3 -c "print(int((${MODEL_DIM}*${REF_M} + ${VOCAB}*${REF_M1} + ${REF_RANK}*(${MODEL_DIM}+${VOCAB})) / (${MODEL_DIM}+${VOCAB})))")
    run "LOWRANK_M${R}" --models base --use-code-head 1 \
        --sch-phi-mode learned --sch-max-m "$R" $PROBE
fi

done

echo ""
echo "============================================================"
echo "  C12 complete."
echo ""
echo "    python -m scripts.sweep_report ${OUT_BASE}/d<DEPTH>"
echo ""
echo "  Read against the arms already at this budget, d8 V=131,072:"
echo "    LOWRANK_M259 (pure low-rank)                       0.918721"
echo "    MON m2=32 clustered (best structured so far)       0.912857"
echo "    MON m2=128 clustered                               0.911994"
echo ""
echo "  Calibration from two measured points is ~0.17 bpb per unit captured"
echo "  energy, which puts TIER_weighted near 0.903 -- dense's own bpb at depth 8"
echo "  (0.902908) for two thirds of the FLOPs. Treat that as an extrapolation:"
echo "  it is 24x further out than any point the calibration was fitted on."
echo ""
echo "  BEFORE BELIEVING ANY OF IT, check the noise floor. Two runs of an"
echo "  identical config at depth 8 differed by 0.0031 (c10 0.914580 against c11"
echo "  0.917674). Effects below that are not measurements."
echo "============================================================"
