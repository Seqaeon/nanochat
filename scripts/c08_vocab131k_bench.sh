#!/usr/bin/env bash
# ============================================================================
# C08: does the Monarch head's quality gap survive a 4x larger vocabulary?
#
# TWO ARMS PER DEPTH: dense and MON_M1024, both at V=131072. The dense arm is
# NOT optional here. There is no V=131072 baseline anywhere in this repo, so it
# has to be measured alongside.
#
# WHY THIS RUN EXISTS, AND WHAT IT IS ALLOWED TO CONCLUDE
#   c07 killed the direction at V=32768: Monarch sits ABOVE the dense Pareto
#   curve from depth 8 onward and moves further off it with depth (1.22x, 1.55x,
#   1.73x more FLOPs than dense needs for the same bpb). The gap shrinks with
#   depth, but the dense curve flattens faster, so a shrinking gap is a growing
#   penalty.
#
#   The only surviving argument is that a larger vocabulary keeps the head a
#   large share of the model and therefore keeps the FLOPs saving alive. That
#   half is arithmetic. The other half is an assumption that has never been
#   tested: that the bpb gap does not itself get worse when V grows.
#
#   c07's own data argues against it. Across that ladder the gap tracked head
#   share as gap ~ share^0.294 with R^2 = 0.9975. But head share, depth and width
#   were collinear there, so depth^-0.338 fits equally well (R^2 = 0.9919) and the
#   ladder cannot separate them. Raising V moves head share WITHOUT moving depth
#   or width, which is the one manipulation that can.
#
# THE COMPARISON THAT SETTLES IT NEEDS ONLY THE DEPTH-8 PAIR
#   V=32768 depth 8 is already measured: head share 35.2%, gap +0.0788.
#   V=131072 depth 8 is this run:        head share 68.4%, gap = ?
#   Same depth, same width, same budget rule. Only the vocabulary moves.
#
# PRE-REGISTERED, BEFORE LOOKING
#   depth 8, V=131072: dense 5.8825e8, MON 2.1417e8, ratio 0.364x, rank 513 = d+1
#     break-even gap (at the V=32768 dense slope of 0.3368/decade):  +0.1478
#     if the gap is V-invariant:                                     +0.0788  wins
#     if the gap tracks share^0.294:                                 +0.0958  wins
#   depth 4, V=131072: dense 2.2885e8, MON 5.4527e7, ratio 0.238x, rank 257 = d+1
#     break-even gap: +0.2098   V-invariant: +0.0963   share^0.294: +0.1055
#
#   So BOTH of my models predict a comfortable win, and the gap has to come in
#   roughly 1.5x worse than the pessimistic one to break even. If it does, both
#   models are wrong and the mechanism is the one neither captures: at 4x the
#   vocabulary, 4x as many tokens share the same d x M map, and per-token
#   capacity (m1 = 32 against d) is unchanged by V. That would end the direction.
#
#   Write the number down before you look at it.
#
# THE SLOPE IS BORROWED AND MUST BE REPLACED
#   The break-even figures above use the dense bpb-per-decade slope measured at
#   V=32768. The V=131072 dense curve is a different curve. Running depth 4 as
#   well as depth 8 gives two dense points at this vocabulary and therefore a
#   local slope, which replaces the borrowed one. Do not quote a Pareto verdict
#   off the borrowed slope.
#
#   bash scripts/c08_vocab131k_bench.sh          # depth 8, the decisive pair
#   bash scripts/c08_vocab131k_bench.sh 4 8      # adds the V=131072 dense slope
# ============================================================================
set -o pipefail

FORCE=0
SEEDS=1
RUN_DENSE=1
RUN_SCH=1
DEPTHS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)      FORCE=1; shift ;;
        --seeds)      SEEDS="$2"; shift 2 ;;
        --dense-only) RUN_DENSE=1; RUN_SCH=0; shift ;;
        --sch-only)   RUN_DENSE=0; RUN_SCH=1; shift ;;
        [0-9]*)       DEPTHS+=("$1"); shift ;;
        *) echo "unknown arg: $1"
           echo "usage: $0 [--force] [--seeds N] [--dense-only] [--sch-only] [DEPTH ...]"; exit 1 ;;
    esac
done
[ ${#DEPTHS[@]} -eq 0 ] && DEPTHS=(8)

M="${M:-1024}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
OUT_BASE="${OUT_BASE:-out/c08_vocab131k}"
RANK_CONTEXTS="${RANK_CONTEXTS:-16384}"
TOK131="${TOKENIZER_DIR_131K:-tokenizer_131k}"

# The tokenizer is the whole prerequisite. Section 8 of the plan requires a
# separate tokenizer trained per vocabulary size on the same corpus; padding a
# 32k tokenizer up to 131k would hand the head 98k rows that never receive a
# gradient and make its job artificially easy.
# Builds the tokenizer, its token_bytes and its frequency table if any are
# missing, and is a no-op otherwise. The loader tokenises on the fly, so nothing
# has to be re-processed to disk; only the tokenizer itself is new. It also
# refuses a directory whose vocab_size is not 131072, because a sweep pinned to
# one vocabulary must not silently run at another.
if ! python3 -m scripts.ensure_tokenizer --vocab-size 131072 --tokenizer-dir "$TOK131" \
        --data-dir "${DATA_DIR:-data}" ${MAX_SHARDS:+--max-shards "$MAX_SHARDS"}; then
    echo "could not prepare the tokenizer at '${TOK131}'; nothing was run."
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

# M >= d keeps the rank ceiling at d+1, which is the property that made Monarch
# beat a plain low-rank head in c06. Below it the arm is a different architecture.
if [ "$M" -lt "$MODEL_DIM" ]; then
    echo "REFUSING depth ${DEPTH}: M=${M} < d=${MODEL_DIM} would make the head rank-limited."
    echo "  Re-run with M=$(python3 -c "
import math;print(1 << max(0, math.ceil(math.log2($MODEL_DIM))))") or higher."
    continue
fi

LOGFILE="${SWEEP_LOG:-${OUT_BASE}/c08_d${DEPTH}.log}"
STATE="${OUT_BASE}/c08_state_d${DEPTH}.json"
[ "$FORCE" -eq 1 ] && rm -f "$STATE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"

# The dense budget at THIS vocabulary. code_head_budget reads the vocab size off
# the tokenizer, so pointing it at the 131k directory is all that is needed.
# Pinning matters more here than anywhere: at V=131072 a dense head carries
# 4x the parameters it did at 32k, so the unpinned budgets would diverge further.
VAR="TARGET_TOKENS_${DEPTH}"
TARGET_TOKENS="${!VAR:-${TARGET_TOKENS:-$(python3 -m scripts.code_head_budget --depth "$DEPTH" --ratio "${RATIO:-10.5}" --tokenizer-dir "$TOK131")}}"

COMMON="--device-batch-size ${DEVICE_BATCH_SIZE:-32} --total-batch-size -1 \
  --use-onecycle 0 --log-every ${LOG_EVERY:-200} --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir $TOK131 \
  --sequence-len ${SEQ_LEN:-2048} --target-tokens $TARGET_TOKENS \
  --target-param-data-ratio -1 \
  --warmup-ratio 0.005 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --research-dim -1 --target-active-params 0 \
  --save-every 200 --eval-every -1 \
  --sch-decile-metrics 1 --sch-eval-steps ${EVAL_STEPS:-100}"
[ -n "${MAX_SHARDS:-}" ] && COMMON="$COMMON --max-shards $MAX_SHARDS"

PROBE="--sch-phi-dtype fp32 --sch-rank-probe $RANK_CONTEXTS --sch-bias 1"

run() {                                   # run <tag> <depth> <flags...>
    local tag="$1"; shift
    local depth="$1"; shift
    for s in $(seq 1 "$SEEDS"); do
        local t="${tag}_s${s}"
        if done_already "$t"; then echo "SKIP  $t (already completed)"; continue; fi
        echo ""
        echo "--- $t  (depth $depth, V=131072) ---"
        local dir="${OUT_BASE}/d${depth}/${t}"
        [ "$FORCE" -eq 1 ] && rm -rf "$dir"
        if bash scripts/research_sweep.sh $COMMON --out-dir "$dir" --seed "$s" \
               "$@" "$depth" 2>&1 | tee -a "$LOGFILE"; then
            mark_done "$t"; echo "OK    $t"
        else
            echo "FAIL  $t (will retry on the next invocation)"
        fi
    done
}

echo "============================================================"
echo "  C08: V=131072 benchmark, depth ${DEPTH}, d=${MODEL_DIM}, M=${M}"
echo "  tokenizer ${TOK131}"
echo "  TARGET TOKENS: ${TARGET_TOKENS}   (pinned from the dense arm)"
echo "  out ${OUT_BASE}"
echo "============================================================"

[ "$RUN_DENSE" -eq 1 ] && run BASE_dense "$DEPTH" --models base --sch-rank-probe $RANK_CONTEXTS
[ "$RUN_SCH"   -eq 1 ] && run "MON_M${M}" "$DEPTH" --models base --use-code-head 1 \
    --sch-head-type monarch --sch-max-m "$M" $PROBE

done

echo ""
echo "============================================================"
echo "  C08 complete."
echo ""
echo "    python -m scripts.sweep_report ${OUT_BASE}/d<DEPTH>"
echo ""
echo "  THE ONE NUMBER: gap = bpb_MON - bpb_dense at depth 8."
echo "    V=32768  depth 8, head share 35.2%:  +0.0788  (already known)"
echo "    V=131072 depth 8, head share 68.4%:  this run"
echo ""
echo "    below +0.0788  the gap IMPROVED with vocabulary. Unexpected, and the"
echo "                   strongest possible result. Go to depth 12 and 16."
echo "    near  +0.0788  V-invariant. Monarch wins the Pareto read at depth 8"
echo "                   and 12; re-check 16."
echo "    near  +0.0958  tracks head share, as c07 suggested. Wins at depth 8,"
echo "                   loses at 12. The viable window is one rung wide."
echo "    above +0.1478  BREAK-EVEN EXCEEDED. Monarch is above the dense curve"
echo "                   even in its most favourable regime. That ends it, and"
echo "                   the reason is the mechanism neither model captured:"
echo "                   4x the tokens sharing one d x M map, with per-token"
echo "                   capacity m1=32 against d unchanged by V."
echo ""
echo "  The break-even figure uses the dense slope measured at V=32768. Run"
echo "  depth 4 as well to get the real V=131072 slope before quoting it."
echo "============================================================"
