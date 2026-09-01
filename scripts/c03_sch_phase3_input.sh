#!/usr/bin/env bash
# ============================================================================
# C03: Phase 3 of the Structured Code Output Heads plan. The input side, with
#      the output head held fixed.
#
# THE FIVE ARMS
#
#     arm        form                  flag                          expected
#     ---------  --------------------  ----------------------------  --------------------------
#     control    learned table         --sch-input-mode table        baseline
#     linear     E = C U               --sch-input-mode linear       rank <= B, SEVERE collapse
#     expanded   E = phi_k(c) U        --sch-input-mode expanded     partial recovery
#     nonlinear  E = MLP(c)            --sch-input-mode nonlinear    near baseline
#     tied       shared C both sides   --sch-input-mode tied         "coded weight tying"
#
# WHY THE ARM THAT SHOULD FAIL IS THE POINT
#   The linear arm is predicted to collapse, and it is run precisely for that
#   reason. E = C U cannot exceed rank B no matter how wide d is, so at V=32768
#   with the minimal code the entire model is fed a 15-dimensional embedding
#   dressed up as a 512-dimensional one. Predicting the failure mode and the
#   exact rank in advance, then measuring both, confirms the mechanism on BOTH
#   sides of the model and earns a figure. A vague "it was worse" earns nothing.
#   The predicted number is measurable directly: the input embedding matrix has
#   rank exactly B, which tests/test_code_head.py already asserts on a tiny model.
#
# THE ECONOMICS ARE DIFFERENT HERE AND THE PAPER MUST SAY SO
#   An input embedding is a GATHER: O(1) compute, costing parameters only. So the
#   efficiency argument that motivates a coded output head is weaker on the input
#   side. The rank damage, meanwhile, is WORSE, because the constraint propagates
#   through the whole network with no normalisation to hide behind. ALBERT uses an
#   intermediate dimension of 128, not 17, for exactly this reason. Note also that
#   the 'expanded' and 'tied' arms are NOT free: they gather a row of Phi and then
#   multiply by U, so they cost M*d MACs per token where a table costs none. The
#   FLOP accounting in GPT.estimate_flops prices that rather than hiding it.
#
#   Prior art to cite here: hash embeddings and Bloom-style compositional
#   embeddings, long standard in large-vocabulary recommender systems.
#
# WHY THIS PHASE IS MANDATORY IF EXTENSION IS THE HEADLINE
#   Coding only the output means a new token can be GENERATED but not CONSUMED,
#   which is half a capability, and reviewers will spot it. So every arm here runs
#   the held-out-vocabulary instrumentation in BOTH modes:
#     --sch-holdout-mode target  masks the held-out ids as prediction targets only.
#                                The head gets no gradient for producing them,
#                                which isolates the output-side claim, while their
#                                input embeddings still train.
#     --sch-holdout-mode full    also rewrites them in the inputs, so the model
#                                never sees them at all. This is the honest
#                                end-to-end extension test, and it is the one the
#                                input side exists to support. It does perturb the
#                                context distribution, so quote both.
#
# CONFIGURATION
#   The output head is held at the Phase 1 winner and the Phase 2 mitigation.
#   Override from the c01 and c02 results rather than editing:
#     SCH_BITS=64 SCH_ORDER=2 SCH_MIXTURE=4 bash scripts/c03_sch_phase3_input.sh 8
#
#   bash scripts/c03_sch_phase3_input.sh                 # depth 8, everything
#   bash scripts/c03_sch_phase3_input.sh --group mechanism 8  # control + linear only
#   bash scripts/c03_sch_phase3_input.sh --seeds 1 8     # fast first pass
# ============================================================================
set -o pipefail

FORCE=0
RUN_GROUPS="mechanism recovery extension full"
SEEDS=3
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
OUT_BASE="${OUT_BASE:-out/c03_sch_input}"
HOLDOUT="${HOLDOUT:-2000}"

# Output-side configuration, held fixed across every arm in this sweep.
SCH_BITS="${SCH_BITS:-64}"
SCH_ORDER="${SCH_ORDER:-2}"
SCH_GTYPE="${SCH_GTYPE:-mlp}"
SCH_CODE_MODE="${SCH_CODE_MODE:-random}"
SCH_CODE_PATH="${SCH_CODE_PATH:-}"
SCH_MIXTURE="${SCH_MIXTURE:-1}"
SCH_RESIDUAL="${SCH_RESIDUAL:-0}"
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
LOGFILE="${SWEEP_LOG:-${OUT_BASE}/c03_d${DEPTH}.log}"
STATE="${OUT_BASE}/c03_state_d${DEPTH}.json"
[ "$FORCE" -eq 1 ] && rm -f "$STATE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"

TARGET_TOKENS="${TARGET_TOKENS:-$(python3 -m scripts.code_head_budget --depth "$DEPTH" --ratio "${RATIO:-10.5}" --tokenizer-dir "${TOKENIZER_DIR:-tokenizer}")}"

# Note: no --sch-holdout-mode here. Each arm sets it explicitly, because running
# both modes on the same output head is half the point of this phase.
COMMON="--device-batch-size ${DEVICE_BATCH_SIZE:-32} --total-batch-size -1 \
  --use-onecycle 0 --log-every ${LOG_EVERY:-200} --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len ${SEQ_LEN:-2048} --target-tokens $TARGET_TOKENS \
  --target-param-data-ratio -1 \
  --warmup-ratio 0.005 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --research-dim -1 --target-active-params 0 \
  --save-every 200 --eval-every -1 \
  --sch-holdout-tokens $HOLDOUT --sch-holdout-seed 7 \
  --sch-decile-metrics 1 --sch-rank-probe ${RANK_CONTEXTS:-8192} \
  --sch-eval-steps ${EVAL_STEPS:-100}"
[ -n "${MAX_SHARDS:-}" ] && COMMON="$COMMON --max-shards $MAX_SHARDS"

HEAD="--models base --use-code-head 1 --sch-bits $SCH_BITS --sch-order $SCH_ORDER \
  --sch-g-type $SCH_GTYPE --sch-g-hidden $MODEL_DIM --sch-code-mode $SCH_CODE_MODE \
  --sch-mixture $SCH_MIXTURE --sch-residual-rank $SCH_RESIDUAL"
if [ -n "$SCH_CODE_PATH" ] && [ -f "$SCH_CODE_PATH" ]; then
    HEAD="--models base --use-code-head 1 --sch-order $SCH_ORDER \
      --sch-g-type $SCH_GTYPE --sch-g-hidden $MODEL_DIM \
      --sch-code-mode file --sch-code-path $SCH_CODE_PATH \
      --sch-mixture $SCH_MIXTURE --sch-residual-rank $SCH_RESIDUAL"
fi

# The 'tied' arm shares the output head's final projection transposed, which is
# only well defined for a linear g. Fall back and say so rather than failing.
TIED_HEAD="$HEAD"
TIED_NOTE=""
if [ "$SCH_GTYPE" != "linear" ]; then
    TIED_HEAD="${HEAD/--sch-g-type $SCH_GTYPE/--sch-g-type linear}"
    TIED_NOTE=" (g forced to linear: the transpose is only defined for a linear g)"
fi

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
echo "  C03 Phase 3: the input side"
echo "  depth ${DEPTH}   d=${MODEL_DIM}   seeds ${SEEDS}"
echo "  output head held at: B=${SCH_BITS} k=${SCH_ORDER} g=${SCH_GTYPE}"
echo "                       mixture=${SCH_MIXTURE} residual=${SCH_RESIDUAL}"
echo "  target tokens ${TARGET_TOKENS}"
echo "  groups: ${RUN_GROUPS}"
echo "  out ${OUT_BASE}"
echo "============================================================"

# ---------------------------------------------------------------- mechanism
# The control and the predicted failure, first, because together they carry the
# mechanism claim. Everything else in this phase is a recovery curve between
# these two points.
if has mechanism; then
    echo ""; echo "### MECHANISM: learned table versus E = C U (rank <= B)"
    run IN_control_table "$DEPTH" $HEAD --sch-input-mode table --sch-holdout-mode target
    run IN_linear_CU     "$DEPTH" $HEAD --sch-input-mode linear --sch-holdout-mode target
fi

# ---------------------------------------------------------------- recovery
# How much of the collapse the expansion and a nonlinear map buy back. The
# ordering of these three against the control is the figure.
if has recovery; then
    echo ""; echo "### RECOVERY: expanded, nonlinear, and coded weight tying${TIED_NOTE}"
    run IN_expanded_phiU "$DEPTH" $HEAD --sch-input-mode expanded --sch-holdout-mode target
    run IN_nonlinear_MLP "$DEPTH" $HEAD --sch-input-mode nonlinear --sch-holdout-mode target
    run IN_tied          "$DEPTH" $TIED_HEAD --sch-input-mode tied --sch-holdout-mode target
fi

# ---------------------------------------------------------------- extension
# The same arms under FULL removal: the held-out ids never appear as inputs
# either. This is the end-to-end capability test. A coded input side should
# degrade far less here than a learned table, because a table row for an unseen
# id is still at its initialisation while a coded embedding is composed from
# parameters that were trained.
if has extension; then
    echo ""; echo "### EXTENSION: the same arms with --sch-holdout-mode full"
    run EXT_control_table "$DEPTH" $HEAD --sch-input-mode table     --sch-holdout-mode full
    run EXT_nonlinear_MLP "$DEPTH" $HEAD --sch-input-mode nonlinear --sch-holdout-mode full
    run EXT_expanded_phiU "$DEPTH" $HEAD --sch-input-mode expanded  --sch-holdout-mode full
    run EXT_tied          "$DEPTH" $TIED_HEAD --sch-input-mode tied --sch-holdout-mode full
    # Dense softmax under the same removal: the thing a code head has to beat on
    # the capability axis, and the arm whose held-out rows never received a
    # gradient at all.
    run EXT_dense_softmax "$DEPTH" --models base --sch-holdout-mode full
fi

# ---------------------------------------------------------------- full
# The complete method: coded input, coded output, and the Phase 2 mitigation
# together. This is the row the paper reports as "ours".
if has full; then
    echo ""; echo "### FULL METHOD: coded input plus the mitigated code head"
    run FULL_method_target "$DEPTH" $HEAD --sch-input-mode nonlinear \
        --sch-mixture "${SCH_MIXTURE_BEST:-4}" --sch-holdout-mode target
    run FULL_method_removal "$DEPTH" $HEAD --sch-input-mode nonlinear \
        --sch-mixture "${SCH_MIXTURE_BEST:-4}" --sch-holdout-mode full
fi

echo ""
echo "============================================================"
echo "  C03 depth ${DEPTH} complete."
echo ""
echo "  READ: sch_results.csv."
echo "    val_bpb across IN_* is the recovery curve. Expected ordering:"
echo "      table  <=  nonlinear  <  expanded  <<  linear"
echo "    IN_linear_CU should be dramatically worse. Verify the MECHANISM rather"
echo "    than just the loss: the input embedding matrix should have rank exactly"
echo "    B. Measure it directly on the checkpoint rather than inferring it."
echo ""
echo "    bpb_holdout and holdout_mean_rank across EXT_* is the capability claim."
echo "    EXT_dense_softmax is the arm to beat: its held-out rows never received"
echo "    a gradient, so it should sit near chance (mean rank about V/2)."
echo ""
echo "  SCOPE NOTE for the write-up: the input side is only MANDATORY if"
echo "  vocabulary extension is the headline. Coding only the output means new"
echo "  tokens can be generated but not consumed, and that is half a capability."
echo "============================================================"

done
