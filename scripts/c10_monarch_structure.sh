#!/usr/bin/env bash
# ============================================================================
# C10: the two structural moves on the Monarch head, at the depth where the
#      current configuration loses.
#
# WHY THIS SWEEP EXISTS
#   Re-scored against the MEASURED V=131072 dense curve (-0.186 bpb/decade from
#   d4 to d8, -0.132 from d8 to d12; the 0.3672 figure quoted in c08 and c09 was
#   carried over from another vocabulary and is far too steep):
#
#     arm             bpb      dense at the same FLOPs   verdict   horizontal
#     d4  m1=32    1.1608                       1.1691      WIN         0.901x
#     d8  m1=32    1.0140                       0.9844     LOSE         1.442x
#     d8  m1=128   0.9398                       0.9601      WIN         0.778x
#     d12 m1=128   0.8616                       0.8535     LOSE         1.151x
#
#   And m1 cannot fix depth 12. The dense head is 50.7% of forward FLOPs there,
#   so a head costing NOTHING earns a break-even of +0.0405 bpb, and m1=128
#   already spends +0.0391 of it. Projecting the measured gap law (m1^-0.795)
#   across m1 gives margins of -0.0080, -0.0013, +0.0008, +0.0001, -0.0027 at
#   m1 = 128, 192, 256, 384, 512. The whole curve sits inside +/-0.003 bpb of
#   break-even. The factorisation axis is exhausted; the capacity axis is not.
#
# Q12: WHICH WORDS SHARE A BLOCK  (--sch-monarch-perm)
#   Word w gets only the m1 features of block w // block_out. Today that is token
#   id, i.e. BPE merge order: roughly frequency-stratified, never semantically
#   coherent. A block only has to separate the words inside its own shard, so a
#   coherent shard should need fewer than m1 dimensions. Costs nothing: no
#   parameters, no FLOPs, one index buffer, so it moves bpb at a fixed x-axis and
#   any gain is pure Pareto movement.
#
#   RANDOM IS NOT FILLER. If freq or cluster wins and random wins as much, the
#   effect is not coherence and the story is wrong. If random LOSES to none, then
#   token-id order was already doing useful work and that is worth knowing too.
#
# Q13: SHARED DIRECTIONS  (--sch-residual-rank)
#   Block-diagonality gives every word m1 PRIVATE directions and no shared ones,
#   so a global unigram or syntactic-class direction has to be relearned in all
#   m2 blocks. logits += C (A h) buys r shared ones for r(d + V) MACs.
#
#     r     head MACs   model vs dense   break-even   m1 that costs the same
#     0         17.6M           0.582x      +0.0310                      128
#    16         19.7M           0.592x      +0.0299                      144
#    32         21.8M           0.603x      +0.0289                      160
#    64         26.0M           0.624x      +0.0270                      192
#
#   Read it against the m1 column: r=32 and m1=160 cost the same, so the question
#   the sweep answers is whether a shared direction is worth more than a private
#   one. Projected from m1^-0.795, m1=160 buys 0.0063 bpb; the residual has to
#   buy 0.0102 to pay for itself outright.
#
# WHAT WOULD END THE DIRECTION
#   Every arm here lands within +/-0.003 bpb of the baseline. Then per-block
#   capacity is not the constraint either, the depth-12 ceiling is real rather
#   than an artifact of a bad partition, and the paper scopes itself to the
#   regime the ceiling allows: head share is about V / (V + 12 d L), so large
#   vocabulary against a small body, which is on-device and multilingual.
#
#   bash scripts/c10_monarch_structure.sh                 # V=131072 depth 12
#   bash scripts/c10_monarch_structure.sh --q12-only 12
#   RANKS="32" PERMS="freq" bash scripts/c10_monarch_structure.sh 8
#   PERM_FILE=perms/cluster_m2_8.pt bash scripts/c10_monarch_structure.sh
# ============================================================================
set -o pipefail

FORCE=0
SEEDS=1
RUN_DENSE=0
RUN_Q12=1
RUN_Q13=1
RUN_LOWRANK=0
DEPTHS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)      FORCE=1; shift ;;
        --seeds)      SEEDS="$2"; shift 2 ;;
        --with-dense) RUN_DENSE=1; shift ;;
        --q12-only)   RUN_Q12=1; RUN_Q13=0; shift ;;
        --q13-only)   RUN_Q12=0; RUN_Q13=1; shift ;;
        --with-lowrank) RUN_LOWRANK=1; shift ;;
        [0-9]*)       DEPTHS+=("$1"); shift ;;
        *) echo "unknown arg: $1"
           echo "usage: $0 [--force] [--seeds N] [--with-dense] [--with-lowrank] [--q12-only|--q13-only] [DEPTH ...]"
           exit 1 ;;
    esac
done
[ ${#DEPTHS[@]} -eq 0 ] && DEPTHS=(12)

M="${M:-1024}"
M1="${M1:-128}"
PERMS="${PERMS:-freq random}"
RANKS="${RANKS:-32}"
VOCAB="${VOCAB:-131072}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
OUT_BASE="${OUT_BASE:-out/c10_monarch_structure}"
RANK_CONTEXTS="${RANK_CONTEXTS:-16384}"
# Same value for every arm, as in c09: wall clock is not comparable across
# device-batch settings, and the Monarch head holds two full (N, V) tensors.
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
# Dense checkpoint to cluster the vocabulary from. Its lm_head IS the matrix
# Monarch replaces, so its row similarity is the structure a block must reproduce.
CLUSTER_CKPT="${CLUSTER_CKPT:-}"
PERM_FILE="${PERM_FILE:-}"

if [ "$VOCAB" -eq 131072 ]; then
    TOK="${TOKENIZER_DIR:-${TOKENIZER_DIR_131K:-tokenizer_131k}}"
else
    TOK="${TOKENIZER_DIR:-tokenizer}"
fi
# Every vocabulary, not just 131k: the Q12 freq arm reads <TOK>/freq_table.pt, and
# a missing table is an assertion inside the head after torchrun has already spun up.
if ! python3 -m scripts.ensure_tokenizer --vocab-size "$VOCAB" --tokenizer-dir "$TOK" \
        --data-dir "${DATA_DIR:-data}" ${MAX_SHARDS:+--max-shards "$MAX_SHARDS"}; then
    echo "could not prepare the tokenizer at '${TOK}'; nothing was run."
    exit 1
fi
mkdir -p "$OUT_BASE"

# Build the clustered permutation if one was asked for and is not on disk yet.
# m2 = M / m1 is the number of blocks, so the clustering has to know it.
M2=$(( M / M1 ))
if [ -z "$PERM_FILE" ] && [ -n "$CLUSTER_CKPT" ]; then
    PERM_FILE="${OUT_BASE}/perm_cluster_m2_${M2}_v${VOCAB}.pt"
fi
if [ -n "$PERM_FILE" ] && [ ! -f "$PERM_FILE" ]; then
    if [ -z "$CLUSTER_CKPT" ]; then
        echo "PERM_FILE=${PERM_FILE} does not exist and CLUSTER_CKPT is unset;"
        echo "  set CLUSTER_CKPT to a dense model_*.pt at V=${VOCAB}, or build the"
        echo "  file yourself with scripts/build_vocab_permutation.py."
        exit 1
    fi
    echo "[c10] clustering ${VOCAB} lm_head rows into ${M2} balanced blocks"
    if ! python3 -m scripts.build_vocab_permutation --mode cluster \
            --checkpoint "$CLUSTER_CKPT" --source lm_head \
            --vocab-size "$VOCAB" --blocks "$M2" --out "$PERM_FILE"; then
        echo "could not build ${PERM_FILE}; nothing was run."
        exit 1
    fi
fi
[ -n "$PERM_FILE" ] && PERMS="$PERMS file"

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
if [ "$M" -lt "$MODEL_DIM" ] && [ "${ALLOW_LOW_RANK:-0}" -eq 0 ]; then
    echo "REFUSING depth ${DEPTH}: M=${M} < d=${MODEL_DIM} would drop the rank ceiling"
    echo "  below d+1 and stop this being the full-rank head c06 validated."
    echo "  Set ALLOW_LOW_RANK=1 if that is deliberate, as it is for a screen that"
    echo "  matches V/M to the target instead of matching M to d."
    continue
fi

LOGFILE="${SWEEP_LOG:-${OUT_BASE}/c10_d${DEPTH}.log}"
STATE="${OUT_BASE}/c10_state_d${DEPTH}.json"
[ "$FORCE" -eq 1 ] && rm -f "$STATE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"

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

PROBE="--sch-phi-dtype fp32 --sch-rank-probe $RANK_CONTEXTS --sch-bias 1"
MON="--models base --use-code-head 1 --sch-head-type monarch \
  --sch-max-m $M --sch-monarch-m1 $M1 $PROBE"

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
echo "  C10: Monarch structure, V=${VOCAB}, depth ${DEPTH}, d=${MODEL_DIM}"
echo "  M=${M}  m1=${M1}  m2=${M2}  block_out=$(( VOCAB / M2 ))"
echo "  Q12 permutations: ${PERMS}"
echo "  Q13 residual ranks: ${RANKS}"
[ -n "$PERM_FILE" ] && echo "  clustered permutation: ${PERM_FILE}"
echo "  target tokens ${TARGET_TOKENS}   device-batch ${DEVICE_BATCH_SIZE}"
echo "============================================================"

# The in-sweep reference. Every other arm is read against this one and not
# against c09's number, because device batch and tokenizer must match.
run "MON_base" $MON
[ "$RUN_DENSE" -eq 1 ] && run BASE_dense --models base --sch-rank-probe $RANK_CONTEXTS

if [ "$RUN_Q12" -eq 1 ]; then
    for p in $PERMS; do
        EXTRA=""
        [ "$p" = "file" ] && EXTRA="--sch-monarch-perm-path $PERM_FILE"
        run "MON_perm_${p}" $MON --sch-monarch-perm "$p" $EXTRA
    done
fi

if [ "$RUN_Q13" -eq 1 ]; then
    for r in $RANKS; do
        run "MON_res_${r}" $MON --sch-residual-rank "$r"
    done
fi

# The control that decides whether the Monarch factorisation is still earning its
# keep. Once the residual is most of the head's cost (79% at r=128), this stops
# being a Monarch head with a residual and becomes a low-rank head with a Monarch
# term, and c06's answer to "is this just low-rank" was established at r=0.
#
# A pure low-rank head of rank R costs R(d + V), so the rank that spends exactly
# what MonarchHead+residual spends is
#
#     R = (d*M + V*m1) / (d + V) + r
#
# computed, never typed, because a hand-matched control drifts the moment M, m1 or
# r changes and then compares two different budgets while looking correct.
if [ "$RUN_LOWRANK" -eq 1 ]; then
    for r in $RANKS; do
        R=$(python3 -c "print(int(($MODEL_DIM*$M + $VOCAB*$M1) / ($MODEL_DIM + $VOCAB)) + $r)")
        run "LOWRANK_M${R}_vs_r${r}" --models base --use-code-head 1 \
            --sch-phi-mode learned --sch-max-m "$R" $PROBE
    done
fi

# The combination, only when both axes ran: if each move helps alone, the
# question is whether they help for the same reason (and so do not add up).
if [ "$RUN_Q12" -eq 1 ] && [ "$RUN_Q13" -eq 1 ]; then
    BEST_PERM="${BEST_PERM:-freq}"
    BEST_RANK="${BEST_RANK:-$(echo $RANKS | awk '{print $NF}')}"
    EXTRA=""
    [ "$BEST_PERM" = "file" ] && EXTRA="--sch-monarch-perm-path $PERM_FILE"
    run "MON_perm_${BEST_PERM}_res_${BEST_RANK}" $MON \
        --sch-monarch-perm "$BEST_PERM" $EXTRA --sch-residual-rank "$BEST_RANK"
fi

done

echo ""
echo "============================================================"
echo "  C10 complete."
echo ""
echo "    python -m scripts.sweep_report ${OUT_BASE}/d<DEPTH>"
echo ""
echo "  READ IT AGAINST MON_base, NOT AGAINST DENSE. Both moves are about"
echo "  per-block capacity, and MON_base is the arm that has neither."
echo ""
echo "  Q12 arms cost exactly the same FLOPs as MON_base, so any bpb they buy is"
echo "  free Pareto movement. Check MON_perm_random FIRST: if it moves as much as"
echo "  freq or file, coherence is not the mechanism and the result is about"
echo "  something else."
echo ""
echo "  Q13 arms cost more, so compare against the break-even in the header:"
echo "    r=16  +0.0299    r=32  +0.0289    r=64  +0.0270"
echo "  and against the equal-cost m1 (144 / 160 / 192), whose projected gain is"
echo "  0.0032 / 0.0063 / 0.0108 bpb. A residual that buys less than the m1 it"
echo "  displaces means shared directions are worth less than private ones, and"
echo "  the answer to depth 12 is simply a bigger m1."
echo "============================================================"
