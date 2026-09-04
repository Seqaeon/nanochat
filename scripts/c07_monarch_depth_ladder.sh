#!/usr/bin/env bash
# ============================================================================
# C07: OPEN_QUESTIONS Q10. Does the Monarch head's quality gap grow with depth?
#
# ONE ARM PER DEPTH. Nothing else. The dense legs are assumed to exist already.
#
# WHY THIS IS THE QUESTION THAT DECIDES THE DIRECTION
#   c06 settled that the structure earns its place at fixed cost: at matched
#   FLOPs a Monarch head beat a plain learned low-rank head on all three pairs
#   (+0.045 to +0.063 bpb), and the margin GREW as cost fell, which is what the
#   softmax bottleneck predicts, because Monarch keeps the dense head's full rank
#   ceiling where low rank throws 94% of it away.
#
#   Everything measured so far is at depth 4, where the head is 64.6% of FLOPs.
#   It is not there at depth 16:
#
#     depth   d     MON/dense FLOPs   head cost removed   rank ceiling
#         4   256          0.455x             84.2%        257 = d+1
#         8   512          0.682x             90.6%        513 = d+1
#        12   768          0.811x             92.7%        769 = d+1
#        16  1024          0.878x             93.7%       1025 = d+1
#
#   The head reduction gets BETTER with depth (84.2% to 93.7%) while the
#   end-to-end saving collapses (0.455x to 0.878x), because the head is an
#   additive constant and the backbone grows around it. At depth 16 a 93.7% cut
#   to the head is 12.2% of the model.
#
#   So the direction lives or dies on the OTHER curve. If the bpb gap shrinks
#   with depth faster than the FLOPs advantage does, the method improves with
#   scale and there is a paper. If the gap holds at +0.096 while the advantage
#   decays to 12%, there is not, and no amount of head engineering fixes it.
#
# WHY M=1024 IS HELD FIXED
#   Not arbitrary: M >= d is what keeps the rank ceiling at d+1, which is the
#   whole reason Monarch beat low rank in c06. M=1024 satisfies that at depths 4
#   through 16 and FAILS at depth 20 (ceiling 1025 against d+1 = 1281), where the
#   arm would silently stop being the thing c06 validated. The script refuses that
#   rather than running it. Past depth 16, set M to the next power of two >= d.
#
# WHY THE BUDGET IS PINNED, WHICH IS NOT OPTIONAL AND IS NOT PARANOIA
#   base_train sizes the horizon from `transformer_matrices + lm_head` (chosen
#   empirically in dev/LOG.md: the Kaplan-style count held the ratio near 10.5
#   across 1e18 to 1e19 FLOPs where the all-parameter count drifted 3.0 to 4.0).
#   That rule was fit on models where head size is a FUNCTION OF d. It stops
#   meaning anything the moment the head is the thing being varied, because the
#   budget then shrinks as a reward for making the head smaller.
#
#   Unpinned, this arm would train on a fraction of its dense counterpart's data:
#
#     depth   dense budget      MON_M1024 budget    ratio
#         4    121,111,872          47,138,112     0.389x
#         8    440,407,296         281,105,664     0.638x
#        12  1,156,067,136         911,437,632     0.788x
#        16  2,466,272,256       2,136,314,880     0.866x
#
#   Read the ratio column. The shortfall SHRINKS with depth, so an unpinned
#   ladder would show the gap closing with scale for a reason that has nothing to
#   do with the architecture. That is precisely the effect Q10 exists to measure,
#   and the confound points the same way as the hoped-for result.
#
#   Excluding the head instead of pinning does NOT work here: it gives both arms
#   264,246,528 at depth 8, which matches neither the dense legs (440,401,920)
#   nor anything else, and would strand every existing dense run.
#
#   VERIFY THE PRINTED NUMBER against your dense leg's "Total number of training
#   tokens" line before trusting any gap. Override with TARGET_TOKENS_<depth>.
#
#   bash scripts/c07_monarch_depth_ladder.sh 8 12 16
#   M=2048 bash scripts/c07_monarch_depth_ladder.sh 20        # past depth 16
#   TARGET_TOKENS_8=123456789 bash scripts/c07_monarch_depth_ladder.sh 8
# ============================================================================
set -o pipefail

FORCE=0
SEEDS=1
DEPTHS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)  FORCE=1; shift ;;
        --seeds)  SEEDS="$2"; shift 2 ;;
        [0-9]*)   DEPTHS+=("$1"); shift ;;
        *) echo "unknown arg: $1"
           echo "usage: $0 [--force] [--seeds N] DEPTH [DEPTH ...]"; exit 1 ;;
    esac
done
[ ${#DEPTHS[@]} -eq 0 ] && DEPTHS=(8 12 16)

M="${M:-1024}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
OUT_BASE="${OUT_BASE:-out/c07_monarch_depth}"
RANK_CONTEXTS="${RANK_CONTEXTS:-16384}"
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

# A Monarch head reaches rank min(M, d) + 1. Below d + 1 the arm stops being the
# full-rank head c06 validated and becomes a low-rank one wearing its name, which
# would read as a depth trend and be a configuration error.
if [ "$M" -lt "$MODEL_DIM" ]; then
    echo ""
    echo "REFUSING depth ${DEPTH}: M=${M} < d=${MODEL_DIM}, so the rank ceiling would be"
    echo "  $((M + 1)) instead of $((MODEL_DIM + 1)). Monarch beat low rank in c06 BECAUSE it"
    echo "  keeps full rank; running it rank-limited here would measure a different"
    echo "  architecture and look like a depth effect."
    echo "  Re-run with M=$(python3 -c "
import math;print(1 << max(0, math.ceil(math.log2($MODEL_DIM))))") or higher."
    continue
fi

LOGFILE="${SWEEP_LOG:-${OUT_BASE}/c07_d${DEPTH}.log}"
STATE="${OUT_BASE}/c07_state_d${DEPTH}.json"
[ "$FORCE" -eq 1 ] && rm -f "$STATE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"

# The dense-equivalent budget at this depth. base_train would otherwise size the
# budget from THIS arm's head parameters, and a Monarch head has 6x fewer of them
# than a dense one, so the arm would silently train on less data than the dense
# leg it is being compared against.
VAR="TARGET_TOKENS_${DEPTH}"
TARGET_TOKENS="${!VAR:-${TARGET_TOKENS:-$(python3 -m scripts.code_head_budget --depth "$DEPTH" --ratio "${RATIO:-10.5}" --tokenizer-dir "${TOKENIZER_DIR:-tokenizer}")}}"

COMMON="--device-batch-size ${DEVICE_BATCH_SIZE:-32} --total-batch-size -1 \
  --use-onecycle 0 --log-every ${LOG_EVERY:-200} --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len ${SEQ_LEN:-2048} --target-tokens $TARGET_TOKENS \
  --target-param-data-ratio -1 \
  --warmup-ratio 0.005 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --research-dim -1 --target-active-params 0 \
  --save-every 200 --eval-every -1 \
  --sch-decile-metrics 1 --sch-eval-steps ${EVAL_STEPS:-100}"
[ -n "${MAX_SHARDS:-}" ] && COMMON="$COMMON --max-shards $MAX_SHARDS"

# --sch-bias 1 because every c05 and c06 Monarch arm carried it. Dropping it here
# would change the architecture between the depth-4 point and the rest of the
# ladder, which is the same class of mistake as changing the budget.
PROBE="--sch-phi-dtype fp32 --sch-rank-probe $RANK_CONTEXTS --sch-bias 1"

run() {                                   # run <tag> <depth> <flags...>
    local tag="$1"; shift
    local depth="$1"; shift
    for s in $(seq 1 "$SEEDS"); do
        local t="${tag}_s${s}"
        if done_already "$t"; then echo "SKIP  $t (already completed)"; continue; fi
        echo ""
        echo "--- $t  (depth $depth) ---"
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
echo "  C07 Q10: Monarch depth ladder, one arm"
echo "  depth ${DEPTH}   d=${MODEL_DIM}   M=${M}   rank ceiling $((M < MODEL_DIM ? M : MODEL_DIM))+1"
echo "  seeds ${SEEDS}"
echo ""
echo "  TARGET TOKENS: ${TARGET_TOKENS}"
echo "  ^ CHECK THIS against your dense depth-${DEPTH} leg's"
echo "    'Total number of training tokens' line. If they differ, stop: the"
echo "    ladder would be measuring the budget and it would look like a trend."
echo "    Override with TARGET_TOKENS_${DEPTH}=<n>."
echo "  out ${OUT_BASE}"
echo "============================================================"

run "MON_M${M}" "$DEPTH" --models base --use-code-head 1 \
    --sch-head-type monarch --sch-max-m "$M" $PROBE

done

echo ""
echo "============================================================"
echo "  C07 complete."
echo ""
echo "  READ IT WITH:"
echo "    python -m scripts.sweep_report ${OUT_BASE}/d<DEPTH> --no-subspace"
echo "  (--no-subspace: there is no dense arm in this sweep to measure against.)"
echo ""
echo "  THE VERTICAL READ, which is the direct answer to Q10:"
echo "    gap(D) = bpb_MON(D) - bpb_dense(D)   at each depth, using YOUR dense legs."
echo "    depth 4 is already known: 1.2559 - 1.1596 = +0.0963."
echo "    If gap(16) < gap(4) the method improves with scale. If it holds or grows"
echo "    while the FLOPs advantage decays from 0.455x to 0.878x, it does not."
echo ""
echo "  THE HORIZONTAL READ, which is the number that goes in an abstract:"
echo "    You have dense at 4/8/12/16, so you have the dense bpb-vs-FLOPs curve."
echo "    For MON at depth D, find where that curve reaches the SAME bpb and"
echo "    report the FLOPs ratio. That is 'Monarch reaches this quality at Nx"
echo "    fewer FLOPs', and it needs no protocol defence. A single pair of points"
echo "    is a comparison; only the curve gives a multiplier."
echo ""
echo "  BEFORE BELIEVING ANY OF IT: confirm every leg trained on the same token"
echo "  budget as its dense counterpart, and that the dense legs came from a"
echo "  comparable commit. sweep_report prints a FLOPs-drift warning when a sweep"
echo "  straddles a code change; there is no such warning across sweeps."
echo "============================================================"
