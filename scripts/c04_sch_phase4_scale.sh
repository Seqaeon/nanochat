#!/usr/bin/env bash
# ============================================================================
# C04: Phase 4 of the Structured Code Output Heads plan. Scale confirmation.
#      Run this ONLY after Phases 0 to 3 have cleared their gates.
#
# THE PURPOSE, STATED PRECISELY SO IT IS NOT OVERCLAIMED
#   This sweep confirms that the ORDERING of methods holds at scale. It does not
#   produce headline numbers, and it is not a bid for a state-of-the-art result at
#   85M parameters. A scale run that reorders the methods invalidates the
#   small-scale conclusions and the paper has to be rebuilt around whatever the
#   new ordering is. A scale run that preserves the ordering is what licenses the
#   cheap d=512 grid to stand in for the general claim.
#
# CONFIGURATION: d=1024, 12 layers (about 85M parameters), V=131072
#   In this repo the width comes from depth * aspect_ratio rounded to a multiple
#   of head_dim, so depth 12 would give d=768, not d=1024. Every arm therefore
#   passes --model-dim 1024 explicitly. That combination is also what makes the
#   width-cap test meaningful a second time: at d=1024 the linear-g ceiling moves
#   from 512 to 1024, so an arm that was capped at d=512 and is still capped at
#   d=1024 is genuinely saturated, while one that improves was never saturated at
#   all. That is the entire reason the plan asks for a second width.
#
# VOCABULARY: 131072, WITH ITS OWN TOKENIZER
#   Section 8 requires a separate tokenizer trained per vocabulary size on the
#   same corpus, and requires BITS PER BYTE rather than token perplexity when
#   comparing across tokenizers, because token-level perplexity is not comparable
#   across them and reviewers catch it immediately. The (gate) group below refuses
#   to run and prints the exact commands if the tokenizer is missing.
#
#   131k is also the vocabulary where the thesis can actually be tested. At 32k
#   even the rarest tokens occur thousands of times in 600M tokens, so the
#   data-scarcity regime the hypothesis lives in does not exist there. At 131k the
#   rarest deciles drop to tens or hundreds of occurrences.
#
# COST
#   These runs are long. At 85M parameters and a Chinchilla budget the arms here
#   are several hours each on one H100, so the default is ONE seed. Anything
#   reported in the paper needs three: section 9 is explicit that small-scale
#   output-layer papers die in review over missing error bars more often than over
#   any other single flaw. Budget for a three-seed rerun of whichever arms make
#   the final table rather than three seeds of everything.
#
#   The completion state file means a killed sweep resumes where it stopped.
#
#   bash scripts/c04_sch_phase4_scale.sh                    # depth 12, d=1024
#   bash scripts/c04_sch_phase4_scale.sh --group gate       # preflight only
#   bash scripts/c04_sch_phase4_scale.sh --group extension  # the flagship result
#   SCH_BITS=64 SCH_ORDER=2 bash scripts/c04_sch_phase4_scale.sh
# ============================================================================
set -o pipefail

FORCE=0
RUN_GROUPS="gate dense best ladder extension"
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
[ ${#DEPTHS[@]} -eq 0 ] && DEPTHS=("${DEPTH:-12}")

MODEL_DIM="${MODEL_DIM:-1024}"
OUT_BASE="${OUT_BASE:-out/c04_sch_scale}"
HOLDOUT="${HOLDOUT:-2000}"
TOKENIZER_DIR_131K="${TOKENIZER_DIR_131K:-tokenizer_131k}"

# The winning configuration from Phases 1 to 3. Override from those results.
SCH_BITS="${SCH_BITS:-64}"
SCH_ORDER="${SCH_ORDER:-2}"
SCH_GTYPE="${SCH_GTYPE:-mlp}"
SCH_CODE_MODE="${SCH_CODE_MODE:-random}"
SCH_CODE_PATH="${SCH_CODE_PATH:-}"
SCH_MIXTURE="${SCH_MIXTURE:-1}"
SCH_RESIDUAL="${SCH_RESIDUAL:-0}"
SCH_INPUT_MODE="${SCH_INPUT_MODE:-table}"
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

# ---------------------------------------------------------------- gate
# Preflight. Runs before anything expensive, and refuses rather than silently
# falling back to the 32k tokenizer, which would produce numbers that look fine
# and answer a different question.
if has gate; then
    echo ""
    echo "### PREFLIGHT"
    MISSING=0
    if [ ! -f "${TOKENIZER_DIR_131K}/tokenizer.pkl" ]; then
        echo "MISSING: no 131k tokenizer at '${TOKENIZER_DIR_131K}'."
        echo "  Section 8 requires a separate tokenizer per vocabulary size, trained on the"
        echo "  same corpus, and bits per byte rather than token perplexity for comparison."
        echo "  Build it with:"
        echo "    python -m scripts.tok_train --vocab-size 131072 --tokenizer-dir ${TOKENIZER_DIR_131K}"
        echo "    python -m scripts.tok_eval  --tokenizer-dir ${TOKENIZER_DIR_131K}"
        MISSING=1
    fi
    if [ ! -f "${TOKENIZER_DIR_131K}/freq_table.pt" ]; then
        echo "MISSING: no freq_table.pt at '${TOKENIZER_DIR_131K}'."
        echo "  The frequency deciles and the Huffman baseline both read it. Build it with:"
        echo "    python -m scripts.code_assign --build-freq-table --tokenizer-dir ${TOKENIZER_DIR_131K}"
        MISSING=1
    fi
    if [ "$MISSING" -eq 1 ]; then
        echo ""
        echo "Refusing to run Phase 4 against an incomplete 131k setup."
        exit 1
    fi
    echo "OK: 131k tokenizer and frequency table present."
fi

# ==== per-depth body ========================================================
for DEPTH in "${DEPTHS[@]}"; do

LOGFILE="${SWEEP_LOG:-${OUT_BASE}/c04_d${DEPTH}.log}"
STATE="${OUT_BASE}/c04_state_d${DEPTH}.json"
[ "$FORCE" -eq 1 ] && rm -f "$STATE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"

# Matched budget from the DENSE arm at this size, as in every other SCH sweep.
TARGET_TOKENS="${TARGET_TOKENS:-$(python3 -m scripts.code_head_budget --depth "$DEPTH" --model-dim "$MODEL_DIM" --ratio "${RATIO:-10.5}" --tokenizer-dir "$TOKENIZER_DIR_131K")}"

COMMON="--device-batch-size ${DEVICE_BATCH_SIZE:-8} --total-batch-size -1 \
  --use-onecycle 0 --log-every ${LOG_EVERY:-200} --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR_131K} \
  --model-dim ${MODEL_DIM} \
  --sequence-len ${SEQ_LEN:-2048} --target-tokens $TARGET_TOKENS \
  --target-param-data-ratio -1 \
  --warmup-ratio 0.005 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --research-dim -1 --target-active-params 0 \
  --save-every 500 --eval-every -1 \
  --sch-decile-metrics 1 --sch-rank-probe ${RANK_CONTEXTS:-8192} \
  --sch-eval-steps ${EVAL_STEPS:-200}"
[ -n "${MAX_SHARDS:-}" ] && COMMON="$COMMON --max-shards $MAX_SHARDS"

BEST="--models base --use-code-head 1 --sch-bits $SCH_BITS --sch-order $SCH_ORDER \
  --sch-g-type $SCH_GTYPE --sch-g-hidden $MODEL_DIM --sch-code-mode $SCH_CODE_MODE \
  --sch-mixture $SCH_MIXTURE --sch-residual-rank $SCH_RESIDUAL \
  --sch-input-mode $SCH_INPUT_MODE"
if [ -n "$SCH_CODE_PATH" ] && [ -f "$SCH_CODE_PATH" ]; then
    BEST="--models base --use-code-head 1 --sch-order $SCH_ORDER \
      --sch-g-type $SCH_GTYPE --sch-g-hidden $MODEL_DIM \
      --sch-code-mode file --sch-code-path $SCH_CODE_PATH \
      --sch-mixture $SCH_MIXTURE --sch-residual-rank $SCH_RESIDUAL \
      --sch-input-mode $SCH_INPUT_MODE"
fi

run() {                                   # run <tag> <depth> <flags...>
    local tag="$1"; shift
    local depth="$1"; shift
    for s in $(seq 1 "$SEEDS"); do
        local t="${tag}_s${s}"
        if done_already "$t"; then echo "SKIP  $t (already completed)"; continue; fi
        echo ""
        echo "--- $t  (depth $depth, d=$MODEL_DIM) ---"
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
echo "  C04 Phase 4: scale confirmation"
echo "  depth ${DEPTH}   d=${MODEL_DIM}   V=131072   seeds ${SEEDS}"
echo "  best config: B=${SCH_BITS} k=${SCH_ORDER} g=${SCH_GTYPE}"
echo "               mixture=${SCH_MIXTURE} residual=${SCH_RESIDUAL} input=${SCH_INPUT_MODE}"
echo "  target tokens ${TARGET_TOKENS}"
echo "  groups: ${RUN_GROUPS}"
echo "  out ${OUT_BASE}"
echo "============================================================"

# ---------------------------------------------------------------- dense
# First, always. Without a same-sweep dense arm every delta below is being
# measured against a number from a different sweep at a different size, which is
# exactly how the MST paper lost its Pareto claim.
if has dense; then
    echo ""; echo "### DENSE: the softmax baseline at this size"
    run SCALE_dense_softmax "$DEPTH" --models base
fi

# ---------------------------------------------------------------- best
# The winner from Phases 1 to 3, unchanged except for the size.
if has best; then
    echo ""; echo "### BEST: the Phase 1 to 3 configuration at d=${MODEL_DIM}"
    run SCALE_best "$DEPTH" $BEST
    # The matched-capacity control travels with it. If the learned-W control
    # closes the gap at scale but did not at d=512, the structure claim is
    # size-dependent and the paper has to say so.
    run SCALE_learned_W "$DEPTH" --models base --use-code-head 1 \
        --sch-phi-mode learned --sch-max-m 2080 \
        --sch-g-type "$SCH_GTYPE" --sch-g-hidden "$MODEL_DIM"
fi

# ---------------------------------------------------------------- ladder
# Two neighbouring rungs, so the paper can show the saturation point MOVES with
# B, k and d as predicted rather than merely existing at one setting. That
# movement is contribution 2 and it needs at least two widths to be visible.
if has ladder; then
    echo ""; echo "### LADDER: neighbouring rungs, plus the width-cap check at d=${MODEL_DIM}"
    run SCALE_B17_k2 "$DEPTH" --models base --use-code-head 1 --sch-order 2 \
        --sch-g-type mlp --sch-g-hidden "$MODEL_DIM"
    run SCALE_B17_k3 "$DEPTH" --models base --use-code-head 1 --sch-order 3 \
        --sch-g-type mlp --sch-g-hidden "$MODEL_DIM"
    # Linear g at the larger width. An arm that was capped at d=512 and is still
    # capped here is genuinely saturated; one that improves never was. This is
    # the second-width test section 3.4 asks for, and it cannot be run at d=512.
    run SCALE_B17_k3_lin "$DEPTH" --models base --use-code-head 1 --sch-order 3 \
        --sch-g-type linear
fi

# ---------------------------------------------------------------- extension
# The flagship. Section 11 is explicit that this is the result that turns the
# work from an efficiency paper into an expressivity paper, and section 7 is
# equally explicit that if it fails there is no A-star paper here.
if has extension; then
    echo ""; echo "### EXTENSION: zero-shot vocabulary extension against the dense control"
    EXT="--sch-holdout-tokens $HOLDOUT --sch-holdout-seed 7"
    run EXT_dense_target "$DEPTH" --models base $EXT --sch-holdout-mode target
    run EXT_code_target  "$DEPTH" $BEST          $EXT --sch-holdout-mode target
    # Full removal: the held-out ids never appear as inputs either. Only
    # meaningful when the input side is coded too, which is why it is paired
    # with SCH_INPUT_MODE from Phase 3.
    run EXT_dense_full "$DEPTH" --models base $EXT --sch-holdout-mode full
    run EXT_code_full  "$DEPTH" $BEST          $EXT --sch-holdout-mode full
fi

echo ""
echo "============================================================"
echo "  C04 depth ${DEPTH} complete."
echo ""
echo "  READ: sch_results.csv. What this sweep is allowed to conclude:"
echo ""
echo "    ORDERING. Compare the rank order of SCALE_* against the same arms at"
echo "    d=512 in c01. Preserved ordering licenses the cheap grid. Reordering"
echo "    means the small-scale conclusions do not transfer and the paper has to"
echo "    be rebuilt around what happens here."
echo ""
echo "    SATURATION MOVES. SCALE_B17_k2 vs SCALE_B17_k3 against the same pair at"
echo "    d=512. Contribution 2 is that the saturation point moves predictably"
echo "    with B, k and d, and one width cannot show movement."
echo ""
echo "    WIDTH CAP AT A SECOND WIDTH. SCALE_B17_k3_lin should now reach rank"
echo "    min(833, 1024) = 833 rather than being pinned at 512. If it is still"
echo "    pinned, something other than d is binding and the ladder reading in c01"
echo "    needs revisiting."
echo ""
echo "    THE FLAGSHIP. bpb_holdout and holdout_mean_rank for EXT_code_* against"
echo "    EXT_dense_*. The dense arm's held-out rows never received a gradient, so"
echo "    it should sit near chance (mean rank about 65536 at V=131072)."
echo "    Section 7: if zero-shot extension fails, say so and move on."
echo ""
echo "  Report bits per byte, never token perplexity: this vocabulary uses a"
echo "  different tokenizer from the d=512 sweeps."
echo "  Anything that reaches the paper needs three seeds. This defaults to one."
echo "============================================================"

done
