#!/usr/bin/env bash
# ============================================================================
# B00: Phase 0 of the fully binary transformer. Three oracles, no training.
#
# WHY THIS SWEEP EXISTS
#   The direction claims a W1A1 transformer with the vocabulary interfaces and
#   the optimiser included, and it claims it on energy and bytes rather than on
#   FLOPs. None of that can be scored by this repo's existing machinery, because
#   a W1A1 model has IDENTICAL FLOPs/token to its dense twin: binarisation
#   changes what an operation costs, not how many there are. Phase 0 builds the
#   instruments and fires the three cheap shots that can close the direction
#   before it costs a training run.
#
#   (cost)        O4. The three axes: bit-operations, energy, training-state
#                 bytes. Its gate reproduces this repo's parameter counts and
#                 FLOPs/token exactly at depths 4 and 8 before anything is
#                 scored against it. Already found two wrong numbers in the
#                 plan: this repo's optimiser measures 60.4 bits/param, not the
#                 assumed 96, and binarising every weight while leaving the
#                 optimiser alone moves training state 1.37x, not 16x, because
#                 the optimiser holds 39.3 of those 60.4 bits.
#
#                 The number that matters comes out of here: at MATCHED
#                 INFERENCE BYTES the binary model is depth 27 / 1.98B params
#                 against dense depth 8 / 126M, so it issues 24.5x the MACs.
#
#   (kernel)      O3. Therefore a b1 GEMM must be >= 24.5x faster per MAC than
#                 bf16 for the binary model merely to TIE on wall clock. That is
#                 38% of the Ampere spec ceiling. Measured so far: 1.93x to
#                 4.19x, unoptimised kernel, on a device hard-capped at 20 W
#                 where the governor trades clock for watts differently per
#                 kernel. THE MAGNITUDE IS UNRESOLVED AND THIS GROUP EXISTS TO
#                 RESOLVE IT ON A CARD WITH A STABLE POWER BUDGET.
#
#                 Architecture is not a free choice. b1 with both AND and XOR
#                 lives on Turing (sm_75) and Ampere (sm_80/86). Ada dropped
#                 INT1; the script refuses to run there. Hopper and Blackwell
#                 removed the XOR operand from hardware and emulate it up to 5x
#                 slower, so they must not carry a headline number. Use T4,
#                 A10G or A100.
#
#   (sensitivity) O5. Where in a whole transformer is floating point actually
#                 load-bearing? One component class at a time, everything else
#                 bf16, measure val bpb. Two axes, because "binarise" is easy to
#                 get wrong: weights only (W1A16, which changes NO operation and
#                 is only an upper bound), activations only, and both (W1A1,
#                 the only arm matching the claim). Crossed with scale none
#                 (strict) against a per-row float scale (a Phase 1 ladder rung
#                 that breaks the title if it survives).
#
#                 Read the asymmetry before reading the numbers: this is a
#                 PROJECTION oracle, so it penalises. A component that survives
#                 is strong evidence; one that fails is weak, because a model
#                 trained binary shapes its own weights for sign() and this one
#                 never did. The opposite of how K7/K8 read a free-fit oracle.
#
#   Phase 0 has two more items not in this script: O2, the sign-agreement probe,
#   and O6, the tiny from-scratch W1A1 run, which needs the architecture written
#   first.
#
# WHAT THIS MACHINE CANNOT DO
#   The dev laptop completed ZERO optimizer steps in 34 minutes at depth 4, and
#   O5 OOMs above batch 1 because the logit tensor is 512 MiB at seq 2048. Run
#   (cost) anywhere; run (kernel) and (sensitivity) on a rented card.
#
#   bash scripts/b00_binary_phase0.sh                          # all three, depth 8
#   bash scripts/b00_binary_phase0.sh --group cost
#   bash scripts/b00_binary_phase0.sh --group kernel
#   DEVICE_BATCH_SIZE=8 EVAL_STEPS=40 bash scripts/b00_binary_phase0.sh --group sensitivity
#   CKPT=out/dense_d8_V32k_model_001014.pt VOCAB_SIZE=32768 \
#       bash scripts/b00_binary_phase0.sh 8
#   MAX_SHARDS=300 SWEEP_LOG=b00.log bash scripts/b00_binary_phase0.sh
# ============================================================================
set -o pipefail

FORCE=0
RUN_GROUPS="cost kernel sensitivity"
DEPTHS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)  FORCE=1; shift ;;
        --group)  RUN_GROUPS="$2"; shift 2 ;;
        [0-9]*)   DEPTHS+=("$1"); shift ;;
        *) echo "unknown arg: $1"
           echo "usage: $0 [--force] [--group cost|kernel|sensitivity] [DEPTH ...]"; exit 1 ;;
    esac
done
[ ${#DEPTHS[@]} -eq 0 ] && DEPTHS=("${DEPTH:-8}")

OUT_BASE="${OUT_BASE:-out/b00_binary_phase0}"
CKPT="${CKPT:-out/dense_d8_V32k_model_001014.pt}"
VOCAB_SIZE="${VOCAB_SIZE:-32768}"
SEQ_LEN="${SEQ_LEN:-2048}"
WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"
TOKENIZER_DIR="${TOKENIZER_DIR:-tokenizer}"
DATA_DIR="${DATA_DIR:-}"
MAX_SHARDS="${MAX_SHARDS:-}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
EVAL_STEPS="${EVAL_STEPS:-40}"
KERNEL_ROUNDS="${KERNEL_ROUNDS:-6}"
SWEEP_LOG="${SWEEP_LOG:-}"
STATE="${STATE:-${OUT_BASE}/state.json}"
mkdir -p "$OUT_BASE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"

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
log() { if [ -n "$SWEEP_LOG" ]; then tee -a "$SWEEP_LOG"; else cat; fi; }

for DEPTH in "${DEPTHS[@]}"; do
    echo "############ B00 depth=$DEPTH V=$VOCAB_SIZE ############" | log

    # ---- O4: the cost model. No GPU, no data. ------------------------------
    if has cost; then
        ARM="cost_d${DEPTH}_V${VOCAB_SIZE}"
        if done_already "$ARM"; then
            echo "[skip] $ARM" | log
        else
            python -m scripts.o4_cost_model --depths "$DEPTH" --vocab "$VOCAB_SIZE" \
                2>&1 | tee "${OUT_BASE}/${ARM}.txt" | log
            [ "${PIPESTATUS[0]}" -eq 0 ] && mark_done "$ARM"
        fi
    fi

    # ---- O3: the kernel gate. Needs nvcc and a b1-capable card. ------------
    if has kernel; then
        ARM="kernel_$(python3 -c "
import torch;p=torch.cuda.get_device_properties(0);print(f'sm{p.major}{p.minor}')" 2>/dev/null || echo nocuda)"
        if done_already "$ARM"; then
            echo "[skip] $ARM" | log
        else
            python -m scripts.o3_kernel_gate --rounds "$KERNEL_ROUNDS" --outdir "$OUT_BASE" \
                2>&1 | tee "${OUT_BASE}/${ARM}.txt" | log
            [ "${PIPESTATUS[0]}" -eq 0 ] && mark_done "$ARM"
        fi
    fi

    # ---- O5: the sensitivity scan. Forward-only on a trained checkpoint. ---
    if has sensitivity; then
        ARM="sensitivity_d${DEPTH}_V${VOCAB_SIZE}"
        if [ ! -f "$CKPT" ]; then
            echo "[skip] $ARM: CKPT not found at $CKPT" | log
        elif done_already "$ARM"; then
            echo "[skip] $ARM" | log
        else
            EXTRA=""
            [ -n "$DATA_DIR" ]   && EXTRA="$EXTRA --data-dir $DATA_DIR"
            [ -n "$MAX_SHARDS" ] && EXTRA="$EXTRA --max-shards $MAX_SHARDS"
            python -m scripts.o5_sensitivity \
                --ckpt "$CKPT" --depth "$DEPTH" --vocab "$VOCAB_SIZE" \
                --seq "$SEQ_LEN" --window-pattern "$WINDOW_PATTERN" \
                --tokenizer-dir "$TOKENIZER_DIR" \
                --batch "$DEVICE_BATCH_SIZE" --eval-steps "$EVAL_STEPS" \
                --binarise weights acts both --scale row none $EXTRA \
                2>&1 | tee "${OUT_BASE}/${ARM}.txt" | log
            [ "${PIPESTATUS[0]}" -eq 0 ] && mark_done "$ARM"
        fi
    fi
done
echo "############ B00 done. results in ${OUT_BASE} ############" | log
