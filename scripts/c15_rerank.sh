#!/usr/bin/env bash
# ============================================================================
# C15: stop making the head cheaper, make it better.
#
# WHY THE COST SIDE IS OVER
#   Every head direction in this project has been cost-side. At V=32,768, which is
#   the vocabulary we can afford to run, that is capped by arithmetic. A head costing
#   NOTHING AT ALL is worth:
#
#     depth 4   head share 64.6%   ratio 0.354x   break-even +0.0762 bpb
#     depth 8   head share 35.2%   ratio 0.648x   break-even +0.0318 bpb
#     depth 12  head share 20.4%   ratio 0.796x   break-even +0.0168 bpb
#
#   And every approximation measured here costs MORE than the whole prize:
#   factorised distribution 0.125, hierarchical softmax 0.122 (c05, it ran, it lost),
#   cost-matched low rank 0.15+. Cost-side head work cannot produce a result at this
#   vocabulary, and that includes the binary-code seed, which is inherently cost-side
#   and whose own motivating hypothesis is refuted by c05: dense's tail-minus-head gap
#   is 0.1901 and all 17 code arms land between 0.2802 and 1.0636, i.e. codes made the
#   tail WORSE, not better.
#
# THE ASYMMETRY THIS EXPLOITS
#   A correction costing 6% of the head is +2.1% of TOTAL FLOPs at depth 8, so it has
#   to GAIN only 0.0015 bpb to be Pareto-positive. That is a thousand times easier
#   than the cost side offers, on the same hardware, at the same vocabulary. And the
#   headline arm costs 1.0011x, so its hurdle is 0.0001 bpb.
#
# WHY A GAIN SHOULD BE THERE
#   The dense head is rank-saturated: c05 depth 4 reports rank_ceiling=257 against
#   rank_effective_rank=256, so the logit matrix uses every direction it is allowed.
#   Godey et al. (2024) tie exactly that mismatch, hidden dimension against the rank
#   of the target distribution, to small-model saturation. And the mass is
#   concentrated: top-1 0.677, top-64 0.9805, effective support 9.3 tokens, so a
#   correction applied only where the mass is costs almost nothing and covers it.
#
# THE MECHANISM
#     z     = W h            dense, exact, unchanged
#     K     = topk(z)        the model's OWN top-k, no corpus statistics
#     z[K] += U[K] . f(h)    rank-r nonlinear correction, on K only
#     p     = softmax(z)     over all V, exact
#
#   Cost k*r rather than V*r: 2,048 MACs/token at k=64 r=32 against the head's 16.8M.
#   f is nonlinear and K is data-dependent, so the log-prob matrix is not confined to
#   rank d+1, which is what Mixture of Softmaxes buys at R times dense cost.
#
# PRIOR ART, and the difference in one line each
#   MoS (Yang 2018) lifts rank with R full-vocabulary softmaxes at R x cost.
#   Sigsoftmax (Kanai 2018) and Ganea (2019) apply a monotone POINTWISE function of
#   z_w alone, so they cannot move two words with equal logits relative to each other.
#   AS-Softmax (2508.03175) DISCARDS low-scoring classes from the normaliser, making
#   the loss approximate; this keeps it exact over all V and only adds capacity.
#
#   VOCAB=32768 bash scripts/c15_rerank.sh 8
#   TK=256 TR=32 bash scripts/c15_rerank.sh 8
#   bash scripts/c15_rerank.sh --with-control 8      # adds the r=0 == dense check
# ============================================================================
set -o pipefail

FORCE=0; SEEDS=1; RUN_CONTROL=0; DEPTHS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)        FORCE=1; shift ;;
        --seeds)        SEEDS="$2"; shift 2 ;;
        --with-control) RUN_CONTROL=1; shift ;;
        [0-9]*)         DEPTHS+=("$1"); shift ;;
        *) echo "unknown arg: $1"
           echo "usage: $0 [--force] [--seeds N] [--with-control] [DEPTH ...]"; exit 1 ;;
    esac
done
[ ${#DEPTHS[@]} -eq 0 ] && DEPTHS=(8)

VOCAB="${VOCAB:-32768}"
# Plain scalars, no loop. tests/test_code_head.py extracts arms statically and cannot
# resolve a variable inside a `for` list, and its `$VAR` substitution is by word
# boundary, so a loop variable that PREFIXES an environment variable silently
# corrupts the arm (that is how `--sch-monarch-m1` once came out as "10241S").
# Sweeping means invoking the script twice, which is how every other knob here works.
TK="${TK:-64}"
TR="${TR:-32}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
OUT_BASE="${OUT_BASE:-out/c15_rerank}"
RANK_CONTEXTS="${RANK_CONTEXTS:-16384}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
SLOPE="${SLOPE:-0.169}"
if [ "$VOCAB" -ge 131072 ]; then SLOPE="${SLOPE_131K:-0.186}"; fi

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
mkdir -p "$OUT_BASE"

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
LOGFILE="${SWEEP_LOG:-${OUT_BASE}/c15_d${DEPTH}.log}"
STATE="${OUT_BASE}/c15_state_d${DEPTH}.json"
[ "$FORCE" -eq 1 ] && rm -f "$STATE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"

# The dense arm's budget, pinned on every arm. The correction adds parameters, so on
# base_train's default sizing the rerank arms would draw a LARGER budget than dense
# and the comparison would be confounded by data. Same argument as OPEN_QUESTIONS Q10,
# in the opposite direction to every previous head here.
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
# The rank probe is the point here, not a diagnostic: the claim is that dense
# saturates its d+1 ceiling and the correction escapes it.
PROBE="--sch-phi-dtype fp32 --sch-rank-probe $RANK_CONTEXTS"
RR="--models base --use-code-head 1 --sch-head-type rerank $PROBE"

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
echo "  C15: rerank head, V=${VOCAB}, depth ${DEPTH}, d=${MODEL_DIM}"
python3 -c "
import math
V,d,D = $VOCAB,$MODEL_DIM,$DEPTH
share = {4:0.646, 8:0.352, 12:0.204}.get(D, V/(V+12*d*D))
print(f'  head share {share:.3f}   a FREE head would be worth '
      f'{-math.log10((1-share))*$SLOPE:+.4f} bpb: that is the ENTIRE cost-side prize')
for k, r in [($TK, $TR)]:
    if True:
        head = 6*V*d + 6*d*r + 6*k*r
        ratio = (1-share) + share*head/(6*V*d)
        print(f'  topk k={k:<5} r={r:<4} head {head/(6*V*d):.4f}x   total {ratio:.4f}x   '
              f'needs a GAIN of {math.log10(ratio)*$SLOPE:+.5f} bpb to break even')
        full = 6*V*d + 6*d*r + 6*V*r
        ratio = (1-share) + share*full/(6*V*d)
        print(f'  full      r={r:<4} head {full/(6*V*d):.4f}x   total {ratio:.4f}x   '
              f'needs a GAIN of {math.log10(ratio)*$SLOPE:+.5f} bpb to break even')"
echo "  target tokens ${TARGET_TOKENS}   device-batch ${DEVICE_BATCH_SIZE}"
echo "============================================================"

run "DENSE" --models base $PROBE

run "RERANK_k${TK}_r${TR}" $RR --sch-rerank-mode topk \
    --sch-rerank-k "$TK" --sch-rerank-rank "$TR"
# the same correction over all V. Prices what restricting to the top-k buys, and is
# the honest cost comparator at r(d+V).
run "RERANK_full_r${TR}" $RR --sch-rerank-mode full --sch-rerank-rank "$TR"
# the published thing this has to beat: a two-component mixture of softmaxes with a
# low-rank second component.
run "MOS2_r${TR}" $RR --sch-rerank-mode mos2 --sch-rerank-rank "$TR"

# a per-token logit scale. Rank +1, d MACs, the cheapest possible rank lift and the
# floor the mechanism has to clear. If this captures most of the gain then the effect
# is one free parameter and not a paper.
run "TEMP" $RR --sch-rerank-mode temp

# paranoia: rank 0 goes down the rerank code path and must land on DENSE exactly.
if [ "$RUN_CONTROL" -eq 1 ]; then
    run "RERANK_r0" $RR --sch-rerank-mode topk --sch-rerank-rank 0
fi

done

echo ""
echo "============================================================"
echo "  C15 complete."
echo ""
echo "    python -m scripts.sweep_report ${OUT_BASE}/d<DEPTH>"
echo ""
echo "  Read in this order and stop at the first failure."
echo ""
echo "    1. RERANK_k64_r32 against DENSE. The arm costs 1.0011x, so ANY gain beyond"
echo "       the +/-0.004 noise floor is Pareto-positive. No gain means the rank"
echo "       ceiling is not binding at d=${MODEL_DIM} and the direction closes."
echo "    2. TEMP. If a per-token scalar captures most of the gain, the effect is one"
echo "       free parameter, not a mechanism, and not a paper."
echo "    3. RERANK_full_r32. If it matches the top-k arm, the restriction is not doing"
echo "       the work and this is a cheap-MoS paper, which is incremental."
echo "    4. MOS2_r32. The published comparator. The mechanism has to beat it."
echo "    5. rank_effective_rank across the arms. DENSE should sit at its d+1 ceiling"
echo "       (c05 measured 256 of 257); the correction should visibly exceed it."
echo "============================================================"
