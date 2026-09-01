#!/usr/bin/env bash
# ============================================================================
# C00: Phase 0 of the Structured Code Output Heads plan. Validate the theory
#      before spending anything on the ladder.
#
# WHY THIS SWEEP EXISTS
#   The whole project rests on one claim: an independent-bit code head has logit
#   rank exactly B. The derivation is four lines. Because
#   log sigma(s) - log sigma(-s) = s,
#
#       log P(w|h) = A(h) + sum_b c_b(w) s_b(h)
#
#   and A(h) does not depend on w, so normalisation removes it and the effective
#   logit matrix is exactly C S. At V=32768 with the minimal code that is rank 15,
#   two orders of magnitude below the ~1000 head-rank threshold Godey et al. (2024)
#   report. Everything downstream (the ladder, the redundancy claim, the choice of
#   mitigations) is a consequence of that bound. If the implementation does not
#   reproduce it, nothing measured later means anything.
#
#   Section 7 of the plan therefore makes it a hard stop:
#       "Phase 0 achieved rank != B  ->  Stop. Implementation bug."
#
#   This sweep also measures the DENSE softmax baseline's ACHIEVED rank. It is
#   expected to sit well below d, because spectral collapse late in training is
#   documented, and that number changes what "matched capacity" means. It belongs
#   in the paper, not in a footnote.
#
# THE SECOND THING THIS SETTLES: THE WIDTH CAP
#   If g is a linear map R^d -> R^M then the logit matrix is Phi G H and its rank
#   is min(M, d), NOT M. At d=512 with B=15 that makes orders 3 (M=575) and 4
#   (M=1940) rank-identical. Running the ladder without knowing this would show
#   saturation at order 3 and produce a false conclusion that is plausible enough
#   to survive review and then collapse at scale. The (ladder) and (mlp) groups
#   below are the direct measurement of that confound: expect 15 / 120 / 512 with
#   a linear g, and 575 with an MLP g at the same order.
#
# fp32 IS NOT OPTIONAL HERE
#   Every arm carrying a rank probe sets --sch-phi-dtype fp32. With a bf16 Phi the
#   singular values below the true rank sit at roughly 1e-3 of the leading one
#   instead of at zero, and the gate misreads as full rank. This was measured
#   during implementation, not assumed: a dense d=64 head reported rank 276 with a
#   bf16 probe and exactly 64 with an fp32 one.
#
# WHAT THIS SWEEP DOES NOT SETTLE
#   Nothing about quality. Every arm here is short and structural. Do not read the
#   bpb column as a comparison; that is what c01 is for.
#
#   bash scripts/c00_sch_phase0_rank.sh                  # depth 8, all groups
#   bash scripts/c00_sch_phase0_rank.sh 8                # depth positionally
#   bash scripts/c00_sch_phase0_rank.sh --group gate 8   # the go/no-go alone
#   bash scripts/c00_sch_phase0_rank.sh --seeds 3 8      # with error bars
#   bash scripts/c00_sch_phase0_rank.sh --force 8        # ignore completion state
# ============================================================================
set -o pipefail

FORCE=0
RUN_GROUPS="gate dense ladder mlp controls"
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
OUT_BASE="${OUT_BASE:-out/c00_sch_phase0}"
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
LOGFILE="${SWEEP_LOG:-${OUT_BASE}/c00_d${DEPTH}.log}"
STATE="${OUT_BASE}/c00_state_d${DEPTH}.json"
[ "$FORCE" -eq 1 ] && rm -f "$STATE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"

# Matched token budget. base_train sizes the budget from head parameters, and a
# code head has up to 17x fewer of them, so per-arm Chinchilla budgeting would
# hand the code arms a much smaller budget than the dense control and confound
# every comparison. Compute the DENSE arm's budget once and pin it everywhere.
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

# Every code-head arm rides the dense backbone; only the head changes.
PROBE="--sch-phi-dtype fp32 --sch-rank-probe $RANK_CONTEXTS"

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
echo "  C00 Phase 0: rank validation"
echo "  depth ${DEPTH}   d=${MODEL_DIM}   seeds ${SEEDS}"
echo "  target tokens ${TARGET_TOKENS}   rank contexts ${RANK_CONTEXTS}"
echo "  groups: ${RUN_GROUPS}"
echo "  out ${OUT_BASE}"
echo "============================================================"

# ---------------------------------------------------------------- gate
# THE GO/NO-GO. Runs first, on purpose. B=15 exactly at V=32768 makes the check
# unambiguous: the code is a bijection onto {0,1}^15, so measured rank must be
# exactly 15. Anything else is an implementation bug and the sweep should stop.
if has gate; then
    echo ""; echo "### GATE: order-1, minimal code. Measured rank must equal B."
    run GATE_order1_Bmin "$DEPTH" --models base --use-code-head 1 \
        --sch-order 1 --sch-code-mode binary $PROBE
fi

# ---------------------------------------------------------------- dense
# The baseline's ACHIEVED rank, not its ceiling. A softmax can never exceed
# rank d+1 at any parameter count, but late in training it typically sits well
# below that, which is the honest number to compare a code head's M against.
if has dense; then
    echo ""; echo "### DENSE: the softmax baseline's achieved rank and decile profile"
    run DENSE_softmax "$DEPTH" --models base --sch-rank-probe $RANK_CONTEXTS
fi

# ---------------------------------------------------------------- ladder
# The width cap, measured. With a linear g, rank = min(M, d). At d=512 that is
# 15, 120, then 512 rather than 575. Seeing exactly 512 at order 3 is the
# evidence that section 3.4 describes a real effect and not a worry.
if has ladder; then
    echo ""; echo "### LADDER (linear g): expect rank 15, 120, then d rather than 575"
    run LADDER_k1_lin "$DEPTH" --models base --use-code-head 1 --sch-order 1 --sch-g-type linear $PROBE
    run LADDER_k2_lin "$DEPTH" --models base --use-code-head 1 --sch-order 2 --sch-g-type linear $PROBE
    run LADDER_k3_lin "$DEPTH" --models base --use-code-head 1 --sch-order 3 --sch-g-type linear $PROBE
fi

# ---------------------------------------------------------------- mlp
# The other half of the same experiment. A nonlinear g has an image that is not
# contained in any d-dimensional subspace, so the ceiling returns to M. This
# pair is what separates "the ladder saturated" from "we hit d", and it is the
# reason every quality sweep from c01 onward carries an MLP-g slice.
if has mlp; then
    echo ""; echo "### MLP g: same order 3, expect rank 575 rather than d"
    run LADDER_k3_mlp "$DEPTH" --models base --use-code-head 1 --sch-order 3 \
        --sch-g-type mlp --sch-g-hidden "$MODEL_DIM" $PROBE
    run LADDER_k4_mlp "$DEPTH" --models base --use-code-head 1 --sch-order 4 \
        --sch-g-type mlp --sch-g-hidden "$MODEL_DIM" $PROBE
fi

# ---------------------------------------------------------------- controls
# Rank is a property of the WIDTH of Phi, not of the monomial structure. These
# two arms should measure the same rank as the monomial arm at the same M. If
# they do not, the probe is measuring something else and the ladder readings
# cannot be trusted.
if has controls; then
    echo ""; echo "### CONTROLS: same M, different Phi. Rank should not care."
    run CTRL_random_binary "$DEPTH" --models base --use-code-head 1 \
        --sch-phi-mode random_binary --sch-max-m 120 $PROBE
    run CTRL_learned_W "$DEPTH" --models base --use-code-head 1 \
        --sch-phi-mode learned --sch-max-m 120 $PROBE
fi

echo ""
echo "============================================================"
echo "  C00 depth ${DEPTH} complete."
echo ""
echo "  READ:  sch_results.csv, column rank_effective_rank"
echo ""
echo "  THE GATE: GATE_order1_Bmin must measure rank exactly 15 at V=32768."
echo "            Anything else is an implementation bug. Section 7 of the plan"
echo "            says stop and fix it before running c01."
echo "            Diagnose in this order: were the logits captured before the"
echo "            softcap, were they mean-centred across the vocabulary axis,"
echo "            was the probe run in fp32."
echo ""
echo "  THE WIDTH CAP: LADDER_k3_lin should measure d (${MODEL_DIM}), not 575,"
echo "                 while LADDER_k3_mlp measures 575. If both measure the same"
echo "                 thing, the MLP g is not doing its job and the whole ladder"
echo "                 sweep in c01 would be uninterpretable."
echo ""
echo "  Re-measure any arm afterwards with:"
echo "    python -m scripts.code_head_diagnostics --checkpoint-dir ${OUT_BASE}/d${DEPTH}/<TAG>"
echo "============================================================"

done
