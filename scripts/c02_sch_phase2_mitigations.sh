#!/usr/bin/env bash
# ============================================================================
# C02: Phase 2 of the Structured Code Output Heads plan. The rank mitigations,
#      run at the best Phase 1 configuration.
#
# THE STRUCTURAL FACT THAT DECIDES WHAT IS WORTH RUNNING
#   Head ARCHITECTURE cannot fix rank. The span of {phi_k(c(w))} is fixed before
#   h enters the head, so attention inside the head, extra layers on top of it,
#   or any other elaboration of how h is processed raises nothing: the logit
#   matrix still factors through that fixed span. Exactly three things change the
#   bound, and this sweep runs exactly those three. Anything else that looks like
#   a mitigation is a way of spending compute without moving the quantity the
#   theory says is binding.
#
#     1. MIXTURE OF CODE HEADS.  log P = log sum_m pi_m(h) P_m(w|h), each P_m a
#        code head. The log-sum-exp is nonlinear in the component logits, so the
#        result is not in the span of Phi at all and the rank bound stops applying
#        rather than merely rising. This is the most publishable of the three, so
#        it runs first among the singles.
#     2. POINTWISE NONLINEARITY on the code logits (sigsoftmax, or a learnable
#        monotonic map). Cheap, elementwise, and it breaks the linear factorisation
#        for the same reason.
#     3. DENSE RESIDUAL HYBRID.  logit = phi(c)^T g(h) + v_w^T W_r h with
#        v_w in R^r. Buys exactly r rank for rV parameters.
#
# WHAT WE EXPECT, WRITTEN DOWN BEFORE THE RUNS
#   The hybrid is expected to win on perplexity, and it is the LEAST interesting
#   result, because it partially reintroduces the softmax it was meant to replace.
#   Section 7 is explicit: if the dense-residual hybrid dominates everything by a
#   wide margin, the pure-code thesis is weak and the paper has to be reframed as
#   a hybrid method. Better to know in week three than month six. So the hybrid
#   runs at several ranks specifically to find where its advantage starts, and
#   r is kept small (around 32) so "it wins" does not just mean "we rebuilt a
#   softmax".
#
#   The mixture and nonlinearity arms carry --sch-rank-probe: their measured rank
#   should EXCEED M, which is the direct evidence that these two escape the bound
#   rather than raising it. If measured rank stays pinned at M, the mitigation is
#   not doing what the derivation says it does and the result is not reportable.
#
# CONFIGURATION
#   The base configuration must come from the c01 results, not from a guess. Set
#   it through the environment:
#
#     SCH_BITS=64 SCH_ORDER=2 SCH_GTYPE=mlp bash scripts/c02_sch_phase2_mitigations.sh 8
#
#   Defaults below are the plan's prediction (B=64, order 2, MLP g), which is what
#   c01's redundancy group is designed to confirm or refute. If c01 selected
#   something else, override rather than editing.
#
#   bash scripts/c02_sch_phase2_mitigations.sh                # depth 8, everything
#   bash scripts/c02_sch_phase2_mitigations.sh --group combo 8  # the go/no-go alone
#   bash scripts/c02_sch_phase2_mitigations.sh --seeds 1 8    # fast first pass
# ============================================================================
set -o pipefail

FORCE=0
RUN_GROUPS="control combo mixture nonlin hybrid"
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
OUT_BASE="${OUT_BASE:-out/c02_sch_mitigations}"
HOLDOUT="${HOLDOUT:-2000}"

# The Phase 1 winner. Override these from the c01 results.
SCH_BITS="${SCH_BITS:-64}"
SCH_ORDER="${SCH_ORDER:-2}"
SCH_GTYPE="${SCH_GTYPE:-mlp}"
SCH_CODE_MODE="${SCH_CODE_MODE:-random}"
SCH_CODE_PATH="${SCH_CODE_PATH:-}"
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
LOGFILE="${SWEEP_LOG:-${OUT_BASE}/c02_d${DEPTH}.log}"
STATE="${OUT_BASE}/c02_state_d${DEPTH}.json"
[ "$FORCE" -eq 1 ] && rm -f "$STATE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"

# Identical token budget on every arm, taken from the dense model. See c01.
TARGET_TOKENS="${TARGET_TOKENS:-$(python3 -m scripts.code_head_budget --depth "$DEPTH" --ratio "${RATIO:-10.5}" --tokenizer-dir "${TOKENIZER_DIR:-tokenizer}")}"

COMMON="--device-batch-size ${DEVICE_BATCH_SIZE:-32} --total-batch-size -1 \
  --use-onecycle 0 --log-every ${LOG_EVERY:-200} --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len ${SEQ_LEN:-2048} --target-tokens $TARGET_TOKENS \
  --target-param-data-ratio -1 \
  --warmup-ratio 0.005 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --research-dim -1 --target-active-params 0 \
  --save-every 200 --eval-every -1 \
  --sch-holdout-tokens $HOLDOUT --sch-holdout-seed 7 --sch-holdout-mode target \
  --sch-decile-metrics 1 --sch-rank-probe ${RANK_CONTEXTS:-8192} \
  --sch-eval-steps ${EVAL_STEPS:-100}"
[ -n "${MAX_SHARDS:-}" ] && COMMON="$COMMON --max-shards $MAX_SHARDS"

# The Phase 1 winner, as a flag string every arm below starts from.
BASECFG="--models base --use-code-head 1 --sch-bits $SCH_BITS --sch-order $SCH_ORDER \
  --sch-g-type $SCH_GTYPE --sch-g-hidden $MODEL_DIM --sch-code-mode $SCH_CODE_MODE"
if [ -n "$SCH_CODE_PATH" ] && [ -f "$SCH_CODE_PATH" ]; then
    BASECFG="--models base --use-code-head 1 --sch-order $SCH_ORDER \
      --sch-g-type $SCH_GTYPE --sch-g-hidden $MODEL_DIM \
      --sch-code-mode file --sch-code-path $SCH_CODE_PATH"
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
echo "  C02 Phase 2: rank mitigations"
echo "  depth ${DEPTH}   d=${MODEL_DIM}   seeds ${SEEDS}"
echo "  base config: B=${SCH_BITS} k=${SCH_ORDER} g=${SCH_GTYPE} code=${SCH_CODE_MODE}"
echo "  target tokens ${TARGET_TOKENS}"
echo "  groups: ${RUN_GROUPS}"
echo "  out ${OUT_BASE}"
echo "============================================================"

# ---------------------------------------------------------------- combo
# The go/no-go, first, as in p08 and p32. If the best mixture plus a small
# residual does not beat the unmitigated arm, the decomposition below is not
# worth the GPU hours and you want that answer in one run rather than twelve.
if has combo; then
    echo ""; echo "### COMBO: best mixture plus a small dense residual (go/no-go)"
    run COMBO_mix4_res32 "$DEPTH" $BASECFG --sch-mixture 4 --sch-residual-rank 32
fi

# ---------------------------------------------------------------- control
# Anchors. Without a same-sweep unmitigated arm and a same-sweep dense arm, the
# deltas below are being measured against numbers from a different sweep, which
# is exactly the mistake that cost the MST paper its Pareto claim.
if has control; then
    echo ""; echo "### CONTROL: the unmitigated Phase 1 winner and the dense baseline"
    run CTRL_unmitigated "$DEPTH" $BASECFG
    run CTRL_dense       "$DEPTH" --models base
fi

# ---------------------------------------------------------------- mixture
# Mitigation 1. Each component is a code head; the log-sum-exp over components is
# where the rank escapes. Order 1 components are the cleanest demonstration
# (each is provably rank B on its own, so any measured rank above B comes from
# the mixing), and one arm at the Phase 1 order shows whether it still helps when
# the components are already wide.
if has mixture; then
    echo ""; echo "### MIXTURE of code heads: rank escapes the bound entirely"
    for M in 2 4 8; do
        run "MIX_order1_m${M}" "$DEPTH" --models base --use-code-head 1 \
            --sch-order 1 --sch-code-mode binary --sch-mixture "$M"
    done
    run MIX_best_m4 "$DEPTH" $BASECFG --sch-mixture 4
fi

# ---------------------------------------------------------------- nonlin
# Mitigation 2. Elementwise and nonlinear, so the log-probability matrix leaves
# the span of Phi. sigsoftmax is the published form; 'monotonic' is a learnable
# variant that starts at the identity, so it cannot hurt at initialisation.
if has nonlin; then
    echo ""; echo "### POINTWISE NONLINEARITY on the code logits"
    run NL_sigsoftmax_order1 "$DEPTH" --models base --use-code-head 1 \
        --sch-order 1 --sch-code-mode binary --sch-logit-act sigsoftmax
    run NL_sigsoftmax_best   "$DEPTH" $BASECFG --sch-logit-act sigsoftmax
    run NL_monotonic_best    "$DEPTH" $BASECFG --sch-logit-act monotonic
fi

# ---------------------------------------------------------------- hybrid
# Mitigation 3. Expected to win, and expected to be the least interesting win.
# The sweep over r is what tells you whether it wins because it added a little
# rank or because it quietly rebuilt the softmax: if quality keeps climbing with
# r, the pure-code thesis is in trouble and section 7 says reframe.
if has hybrid; then
    echo ""; echo "### DENSE RESIDUAL HYBRID: r rank for rV parameters"
    for R in 8 32 128; do
        run "HYB_r${R}" "$DEPTH" $BASECFG --sch-residual-rank "$R"
    done
    # Order 1 plus a residual: the cheapest possible head that still has real
    # rank. If this matches the full ladder, the ladder is not what is working.
    run HYB_order1_r32 "$DEPTH" --models base --use-code-head 1 \
        --sch-order 1 --sch-code-mode binary --sch-residual-rank 32
fi

echo ""
echo "============================================================"
echo "  C02 depth ${DEPTH} complete."
echo ""
echo "  READ: sch_results.csv."
echo "    rank_effective_rank vs phi_width_M"
echo "        MIX_* and NL_* must measure rank ABOVE M. That is the direct"
echo "        evidence that they escape the bound rather than raising it."
echo "        If they stay pinned at M, the mitigation is not working."
echo "    val_bpb"
echo "        every arm against CTRL_unmitigated and CTRL_dense from this sweep."
echo "    head_params, head_flops_per_token"
echo "        the hybrid buys rank with rV parameters. Price it here, because a"
echo "        mitigation that costs as much as the softmax is not a mitigation."
echo ""
echo "  THE DECISION (section 7): if HYB_* dominates MIX_* and NL_* by a wide"
echo "  margin, the pure-code thesis is weak. Reframe as a hybrid method and say"
echo "  so, rather than reporting the hybrid as a code-head result."
echo "============================================================"

done
