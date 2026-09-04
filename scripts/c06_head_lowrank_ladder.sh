#!/usr/bin/env bash
# ============================================================================
# C06: does a Monarch head beat a plain learned low-rank head at matched cost?
#
# WHY THIS SWEEP EXISTS
#   c05 produced one genuinely good arm. A Monarch head reached 1.2559 bpb at
#   0.455x dense FLOPs and 0.762x dense wall clock, removing 84.2% of the head's
#   cost and parameters, with every parameter trained so alignment never enters.
#
#   But c05 compared THREE Monarch points against ONE low-rank point, and that
#   one point was at the wrong width. A learned Phi of width M costs
#   6*(V+d)*M = 198,144*M FLOPs at V=32768, d=256, so:
#
#       MON M1024 m1=8  total 3.074e7  cost-matched learned_W is M = 16
#       MON M256        total 3.113e7  cost-matched learned_W is M = 18
#       MON M1024       total 3.546e7  cost-matched learned_W is M = 40
#
#   c05 ran learned_W at M=120, which costs 5.130e7. It was never a matched
#   comparison: the low-rank arm was handed 1.45x the compute of the Monarch arm
#   it was being read against. The cost-matched widths are far SMALLER than the
#   one that ran, and nobody has measured them.
#
#   This sweep runs both curves so the comparison can be read horizontally.
#
# THE DECISION RULE, WRITTEN BEFORE THE RUN
#   At 3.546e7 FLOPs, MON_M1024 scored 1.2559. If LOWRANK_M40 scores 1.2559 or
#   better, the Monarch structure buys NOTHING that a plain rank reduction does
#   not, and the direction is finished. Say so and stop.
#   If LOWRANK_M40 is clearly worse, the block-diagonal structure is doing real
#   work at fixed cost, and only then is the depth ladder (OPEN_QUESTIONS Q10)
#   worth the compute.
#
# WHY BOTH CURVES RUN HERE RATHER THAN REUSING c05
#   The c05 arms straddle a code change: `sweep_report` reports a FLOPs drift for
#   them, and the product arms alone span two implementations that differ 70x in
#   cost at the same width. A comparison this sharp has to come from one commit.
#
# THE CONFOUND c05 CONTAINED, NOW REMOVED
#   Every MON arm in c05 ran with --sch-bias 1 and BASE_learned_W ran without it.
#   A per-token bias costs 2V FLOPs and was worth 0.177 bpb on the code arms
#   (FREE_order2_bias against BASE_code_order2), so the two were never on equal
#   footing. Every arm here carries --sch-bias 1.
#
#   bash scripts/c06_head_lowrank_ladder.sh                    # depth 8, all groups
#   bash scripts/c06_head_lowrank_ladder.sh 4                  # depth 4, matching c05
#   bash scripts/c06_head_lowrank_ladder.sh --group lowrank 4  # the low-rank curve alone
#   bash scripts/c06_head_lowrank_ladder.sh --seeds 3 4        # with error bars
# ============================================================================
set -o pipefail

FORCE=0
RUN_GROUPS="baseline lowrank monarch"
SEEDS=1
DEPTHS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)  FORCE=1; shift ;;
        --group)  RUN_GROUPS="$2"; shift 2 ;;
        --seeds)  SEEDS="$2"; shift 2 ;;
        [0-9]*)   DEPTHS+=("$1"); shift ;;
        *) echo "unknown arg: $1"
           echo "usage: $0 [--force] [--group G] [--seeds N] [DEPTH ...]"; exit 1 ;;
    esac
done
[ ${#DEPTHS[@]} -eq 0 ] && DEPTHS=("${DEPTH:-8}")

ASPECT_RATIO="${ASPECT_RATIO:-64}"
OUT_BASE="${OUT_BASE:-out/c06_head_lowrank}"
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

has() { echo " $RUN_GROUPS " | grep -q " $1 "; }

# ==== per-depth body ========================================================
for DEPTH in "${DEPTHS[@]}"; do

MODEL_DIM=$(( ((DEPTH * ASPECT_RATIO + 127) / 128) * 128 ))
LOGFILE="${SWEEP_LOG:-${OUT_BASE}/c06_d${DEPTH}.log}"
STATE="${OUT_BASE}/c06_state_d${DEPTH}.json"
[ "$FORCE" -eq 1 ] && rm -f "$STATE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"

# Same pinning as c00 and c05. base_train sizes the token budget from head
# parameters, and these heads differ by 15x in that count, so per-arm Chinchilla
# budgeting would hand every arm a different budget and confound the sweep.
# Compute the DENSE budget once and pin it everywhere.
TARGET_TOKENS="${TARGET_TOKENS:-$(python3 -m scripts.code_head_budget --depth "$DEPTH" --ratio "${RATIO:-10.5}" --tokenizer-dir "${TOKENIZER_DIR:-tokenizer}")}"

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

# Every arm carries the bias, so the c05 confound cannot recur here.
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
echo "  C06: learned low-rank head against Monarch, at matched cost"
echo "  depth ${DEPTH}   d=${MODEL_DIM}   seeds ${SEEDS}"
echo "  target tokens ${TARGET_TOKENS}   rank contexts ${RANK_CONTEXTS}"
echo "  groups: ${RUN_GROUPS}"
echo "  out ${OUT_BASE}"
echo "============================================================"

# ---------------------------------------------------------------- baseline
# The anchor both curves are read against, on this commit rather than c05's.
if has baseline; then
    echo ""; echo "### BASELINE: dense softmax"
    run BASE_dense "$DEPTH" --models base --sch-rank-probe $RANK_CONTEXTS
fi

# ---------------------------------------------------------------- lowrank
# The curve c05 never measured. M=16, 18 and 40 are the cost matches to the
# three Monarch arms; 32 and 64 fill the gap; 120 reproduces c05's single point
# so the two sweeps can be checked against each other; 256 is the width at which
# a learned low-rank head costs the same as the dense softmax it replaces and is
# there to show the curve turning over rather than to win anything.
if has lowrank; then
    echo ""; echo "### LOWRANK: learned Phi of width M. Cost is 198,144*M at V=32768, d=256."
    for M in 16 18 32 40 64 120 256; do
        run "LOWRANK_M${M}" "$DEPTH" --models base --use-code-head 1 \
            --sch-phi-mode learned --sch-max-m "$M" $PROBE
    done
fi

# ---------------------------------------------------------------- monarch
# Re-run on this commit, with the bias, so the two curves are comparable. M512
# and M2048 fill in around the three points c05 already has.
if has monarch; then
    echo ""; echo "### MONARCH: two block-diagonal factors, cost d*M + V*m1"
    for M in 256 512 1024 2048; do
        run "MON_M${M}" "$DEPTH" --models base --use-code-head 1 \
            --sch-head-type monarch --sch-max-m "$M" $PROBE
    done
    # The cheapest Monarch point in c05, kept because it is the one that matches
    # LOWRANK_M16 on cost and so decides the low-cost end of the comparison.
    run MON_M1024_m1_8 "$DEPTH" --models base --use-code-head 1 \
        --sch-head-type monarch --sch-max-m 1024 --sch-monarch-m1 8 $PROBE
fi

echo ""
echo "============================================================"
echo "  C06 depth ${DEPTH} complete."
echo ""
echo "  READ IT WITH:"
echo "    python -m scripts.sweep_report ${OUT_BASE}/d${DEPTH}"
echo ""
echo "  THE COMPARISON IS HORIZONTAL, AT MATCHED COST. At V=32768 and d=256:"
echo "    LOWRANK_M16  ~ MON_M1024_m1_8   (3.07e7 vs 3.07e7)"
echo "    LOWRANK_M18  ~ MON_M256         (3.11e7 vs 3.11e7)"
echo "    LOWRANK_M40  ~ MON_M1024        (3.55e7 vs 3.55e7)"
echo "  Those pairs are the whole sweep. Everything else fills the curves."
echo ""
echo "  THE DECISION RULE, FIXED BEFORE THE RUN: MON_M1024 scored 1.2559 in c05."
echo "  If LOWRANK_M40 reaches 1.2559 or better, the block-diagonal structure"
echo "  buys nothing a plain rank reduction does not, and the direction is over."
echo "  Only if LOWRANK_M40 is clearly worse is the depth ladder (Q10) worth it."
echo ""
echo "  WATCH THE WALL CLOCK, NOT ONLY THE FLOPS. A learned Phi is one dense"
echo "  matmul and runs at tensor-core speed; Monarch is two smaller ones plus a"
echo "  transpose. c05 measured Monarch at 0.762x dense time against learned_W's"
echo "  0.968x, so Monarch's advantage there was larger than its FLOPs advantage."
echo "  If that inverts at small M the FLOPs axis is the wrong one to argue on."
echo "============================================================"

done
