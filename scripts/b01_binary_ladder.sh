#!/usr/bin/env bash
# ============================================================================
# B01: Phase 1 of the fully binary transformer. The ladder.
#
# WHY THIS SWEEP EXISTS
#   Phase 0 produced one from-scratch W1A1 number: +0.1957 bpb against dense at
#   depth 8, all 58 modules binary including wte and all four value_embeds. It
#   came from a standalone script at 400 steps on 6.6M tokens. It is not
#   budget-pinned, not seeded, not on the repo's Pareto curve, and not
#   reproducible through the normal sweep machinery. This turns it into a result.
#
#   It also asks a question Phase 0 could not. O5 ranked every component by its
#   IN-SITU cost, measured by projecting a TRAINED checkpoint onto sign():
#
#       lm_head +0.0909 | wte +0.0254 | mlp.c_proj +0.0078 | value_embeds +0.0048
#       attn.c_proj -0.0045 | mlp.c_fc -0.0111 | attn.qkv -0.0123 | ve_gate -0.0159
#
#   Four of those are NEGATIVE: restoring the component to bf16 makes the model
#   WORSE, which is the measured form of "consistency beats precision" and is the
#   spine of the design argument. The rungs below are ordered by that table.
#   DOES THE ORDERING SURVIVE FROM-SCRATCH TRAINING? If it does not, every
#   in-situ conclusion in LEARNINGS.md needs re-reading.
#
# THE RUNGS
#   R0  dense control
#   R1  body weights only, W1A16. Changes NO operation: still a bf16 GEMM over a
#       matrix holding two values per row. An upper bound, not a binary model.
#   R2  body W1A1, all three interfaces fp. The BitNet-comparable point.
#   R3  R2 + value_embeds. 53.3% of all parameters for a predicted +0.0048.
#   R4  R3 + wte.
#   R5  R4 + lm_head. Equals O6's configuration; should reproduce +0.1957.
#   R6  R5 with the threshold replacing per-channel scales. Section 3.3's arm.
#       Target: recover most of the 0.4358 bpb that strict no-scale costs.
#
# WHAT THIS SWEEP IS NOT MEASURING
#   Speed. Every arm here trains SLOWER than R0: BinaryLinear does sign_ste, an
#   fp scale multiply, and then an ordinary bf16 GEMM. Phase 1 uses no binary
#   arithmetic at all. That is how quantisation-aware training works, and the
#   speed story belongs to O3 (1.13-1.58x at matched architecture, on Ampere,
#   with a kernel at 4-8% of b1 peak). Reading a step-time regression here as a
#   failure is an axis confusion.
#
# HARDWARE
#   H100 for this sweep: it is purely a faster bf16 machine and that is all
#   Phase 1 needs. Keep EVERY arm on one GPU type. H100 enables FlashAttention-3
#   which Ampere falls back from, and mixing would put an attention-implementation
#   difference inside the bpb deltas. O3 kernel work still requires Ampere.
#
#   bash scripts/b01_binary_ladder.sh                       # depth 8, all rungs
#   bash scripts/b01_binary_ladder.sh --rungs "R0 R5" 8     # reproduce O6 only
#   bash scripts/b01_binary_ladder.sh --seeds 3 8
#   TARGET_TOKENS=100000000 bash scripts/b01_binary_ladder.sh 8
# ============================================================================
set -o pipefail

FORCE=0
RUN_RUNGS="R0 R1 R2 R3 R4 R5 R6"
SEEDS="${SEEDS:-3}"
DEPTHS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)  FORCE=1; shift ;;
        --rungs)  RUN_RUNGS="$2"; shift 2 ;;
        --seeds)  SEEDS="$2"; shift 2 ;;
        [0-9]*)   DEPTHS+=("$1"); shift ;;
        *) echo "unknown arg: $1"
           echo "usage: $0 [--force] [--rungs \"R0 R5\"] [--seeds N] [DEPTH ...]"; exit 1 ;;
    esac
done
[ ${#DEPTHS[@]} -eq 0 ] && DEPTHS=("${DEPTH:-8}")

OUT_BASE="${OUT_BASE:-out/b01_binary_ladder}"
VOCAB_SIZE="${VOCAB_SIZE:-32768}"
TOKENIZER_DIR="${TOKENIZER_DIR:-tokenizer}"
DATA_DIR="${DATA_DIR:-data}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
# -1 = auto-compute from the token budget (B_REF * (D/D_REF)^0.383, rounded to a
# power of two), which is what every other sweep uses. Override only to pin the
# batch across arms; note the LRs and weight decay are BOTH derived from it
# (base_train.py:1749, :1799), so changing it changes more than the batch.
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:--1}"
# Match the repo's standard config (p13, and every existing dense leg) so these
# arms land on the SAME Pareto curve. 1024/L was carried over from the Ampere
# probes, where SDPA has no sliding-window support and warns about it; on H100
# FlashAttention-3 handles windows natively, and full attention on every layer
# costs 1.05x the FLOPs while making the results incomparable to the repo's
# existing d8 dense baselines.
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"
# These were being silently ignored: base_train's --log-every DEFAULTS TO 1, so an
# unwired LOG_EVERY prints a line per step, and an unwired MAX_SHARDS scans every
# shard. Mirrors the set p13_isodata.sh:297-305 passes.
LOG_EVERY="${LOG_EVERY:-200}"
EVAL_EVERY="${EVAL_EVERY:--1}"
SAVE_EVERY="${SAVE_EVERY:-200}"
MAX_SHARDS="${MAX_SHARDS:-}"
COMPILE_REGIONAL="${COMPILE_REGIONAL:-0}"
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
has() { echo " $RUN_RUNGS " | grep -q " $1 "; }
log() { if [ -n "$SWEEP_LOG" ]; then tee -a "$SWEEP_LOG"; else cat; fi; }

# Rung -> extra base_train flags. `binary_skip` holds name substrings left in fp,
# so the rungs are built by SUBTRACTION from the fully binary model.
rung_flags() {
    case "$1" in
        R0) echo "" ;;
        R1) echo "--use-binary 1 --binary-acts 0 --binary-embeddings 0 --binary-skip lm_head" ;;
        R2) echo "--use-binary 1 --binary-embeddings 0 --binary-skip lm_head" ;;
        R3) echo "--use-binary 1 --binary-skip lm_head,transformer.wte" ;;
        R4) echo "--use-binary 1 --binary-skip lm_head" ;;
        R5) echo "--use-binary 1" ;;
        R6) echo "--use-binary 1 --binary-weight-scale threshold" ;;
        *)  echo "UNKNOWN" ;;
    esac
}

for DEPTH in "${DEPTHS[@]}"; do
    # NOTE ON --target-param-data-ratio: it stays POSITIVE even though the budget is
    # pinned explicitly. --target-tokens wins the budget at base_train.py:1769, but
    # D_REF at :1785 is `target_param_data_ratio * get_scaling_params(d12_ref)` and is
    # used UNCONDITIONALLY for the batch-size derivation. Passing -1 there (as c05
    # does) makes D_REF negative, so batch_size_ratio is negative and
    # `B_REF * ratio**0.383` returns a COMPLEX number, which dies at
    # `math.log2(...)` with "TypeError: must be real number, not complex".
    # p13_isodata.sh has this right: pin with --target-tokens and leave the ratio at
    # 10.5, where it is inert for the budget and sane for D_REF.
    #
    # Pin every arm to the DENSE arm's token budget. Binary arms carry
    # extra per-channel scale parameters, and get_scaling_params is
    # transformer_matrices + lm_head, so per-arm Chinchilla would hand each rung a
    # slightly different budget and confound data with architecture.
    if [ -z "${TARGET_TOKENS:-}" ]; then
        TARGET_TOKENS=$(python3 -m scripts.code_head_budget --depth "$DEPTH" --ratio 10.5 \
            --tokenizer-dir "$TOKENIZER_DIR" 2>/dev/null)
    fi
    if [ -z "$TARGET_TOKENS" ]; then
        echo "FATAL: could not compute a token budget; refusing to run unpinned arms" | log
        exit 1
    fi
    echo "############ B01 depth=$DEPTH V=$VOCAB_SIZE tokens=$TARGET_TOKENS seeds=$SEEDS ############" | log

    for RUNG in R0 R1 R2 R3 R4 R5 R6; do
        has "$RUNG" || continue
        FLAGS=$(rung_flags "$RUNG")
        [ "$FLAGS" = "UNKNOWN" ] && { echo "unknown rung $RUNG" | log; exit 1; }
        for s in $(seq 1 "$SEEDS"); do
            TAG="${RUNG}_s${s}"
            ARM="d${DEPTH}_${TAG}"
            if done_already "$ARM"; then
                echo "[skip] $ARM" | log
                continue
            fi
            echo "---- $ARM : $RUNG ${FLAGS:-(dense)} ----" | log
            # -u is not optional here. Step logs go through print0 -> stdout, which
            # Python BLOCK-BUFFERS when piped through tee, while checkpoint messages go
            # through logging -> stderr unbuffered. Without it the log shows only the
            # periodic saves and looks like training has stalled.
            python -u -m scripts.base_train \
                --depth "$DEPTH" --tokenizer-dir "$TOKENIZER_DIR" --data-dir "$DATA_DIR" \
                --device-batch-size "$DEVICE_BATCH_SIZE" --max-seq-len "$MAX_SEQ_LEN" \
                --total-batch-size "$TOTAL_BATCH_SIZE" \
                --window-pattern "$WINDOW_PATTERN" \
                --log-every "$LOG_EVERY" --eval-every "$EVAL_EVERY" \
                --save-every "$SAVE_EVERY" --compile-regional "$COMPILE_REGIONAL" \
                ${MAX_SHARDS:+--max-shards $MAX_SHARDS} \
                --target-tokens "$TARGET_TOKENS" --target-param-data-ratio 10.5 \
                --seed "$s" --model-tag "$ARM" $FLAGS \
                2>&1 | tee "${OUT_BASE}/${ARM}.log" | log
            [ "${PIPESTATUS[0]}" -eq 0 ] && mark_done "$ARM"
        done
    done
done
echo "############ B01 done. results in ${OUT_BASE} ############" | log
