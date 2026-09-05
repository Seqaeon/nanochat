#!/usr/bin/env bash
# ============================================================================
# C11: block COUNT at fixed cost, and whether the blocks are chosen or arbitrary.
#
# THE AXIS EVERY EARLIER SWEEP MISSED
#   c09 and c10 held M fixed and moved m1, which forces m2 = M/m1. The number of
#   blocks was never a free variable, and it turns out to be the dominant one.
#   Scoring architectures offline against the trained dense head at depth 4
#   (scripts/head_oracle.py, captured energy of the row-centred head):
#
#     block_out    token-id gain over pure low-rank
#           512    +0.0319
#         4,096    +0.0063     5x worse
#        16,384    +0.0015    21x worse
#
#   A block of 16,384 arbitrary words spans very nearly the same subspace as the
#   whole vocabulary, so its private capacity duplicates the shared basis and buys
#   nothing. That single fact predicts everything measured so far: the apparent m1
#   saturation, the collapse from V=32,768 to V=131,072, and why the frequency and
#   random permutations did nothing (neither changes subspace structure).
#
# THE TRADE THIS SWEEP PRICES
#   More blocks means larger M = m1*m2, and d*M is paid out of the shared rank r.
#   Iso-cost at depth 4, m1=32, against the arm actually trained (m2=8, bpb 1.1757):
#
#     m2      M   block_out    r   token-id  clustered  ratio
#      8    256       4,096  128    +0.0063    +0.0131  2.07x   <- what we ran
#     16    512       2,048  126    +0.0071    +0.0159  2.24x
#     32   1024       1,024  122    +0.0071    +0.0183  2.58x
#     64   2048         512  114    +0.0048    +0.0191  3.94x   <- best
#    128   4096         256   98    -0.0020    +0.0153     --
#
#   Two things to read there. The trained arm is the worst row. And the token-id
#   column goes NEGATIVE past m2=128: with an arbitrary assignment small blocks are
#   pure waste, so shared rank is given up for nothing. Clustering is what makes
#   small blocks pay, which makes the two changes complements rather than
#   alternatives -- hence both arms, not just the clustered one.
#
# WHY THE CONTROL IS NOT OPTIONAL
#   If token-id also improves at m2=64, the win is block count and the clustering
#   is decoration. If only the clustered arm improves, subspace coherence is the
#   mechanism and the offline oracle is trustworthy for designing what comes next.
#
#   bash scripts/c11_monarch_blocks.sh                    # d4, V=32,768, m2=64
#   M2S="16 32 64" bash scripts/c11_monarch_blocks.sh 4
#   VOCAB=131072 CLUSTER_CKPT=<dense head .pt> bash scripts/c11_monarch_blocks.sh 8
# ============================================================================
set -o pipefail

FORCE=0
SEEDS=1
RUN_ID=1
RUN_CLU=1
RUN_REF=0
DEPTHS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)     FORCE=1; shift ;;
        --seeds)     SEEDS="$2"; shift 2 ;;
        --id-only)   RUN_ID=1; RUN_CLU=0; shift ;;
        --clu-only)  RUN_ID=0; RUN_CLU=1; shift ;;
        --with-ref)  RUN_REF=1; shift ;;
        [0-9]*)      DEPTHS+=("$1"); shift ;;
        *) echo "unknown arg: $1"
           echo "usage: $0 [--force] [--seeds N] [--id-only|--clu-only] [--with-ref] [DEPTH ...]"
           exit 1 ;;
    esac
done
[ ${#DEPTHS[@]} -eq 0 ] && DEPTHS=(4)

M1="${M1:-32}"                  # per-word block capacity, held fixed across the sweep
M2S="${M2S:-64}"                # block counts to price
REF_M2="${REF_M2:-8}"           # the configuration whose cost defines the budget
REF_RANK="${REF_RANK:-128}"
VOCAB="${VOCAB:-32768}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
OUT_BASE="${OUT_BASE:-out/c11_monarch_blocks}"
RANK_CONTEXTS="${RANK_CONTEXTS:-16384}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
CLUSTER_ITERS="${CLUSTER_ITERS:-25}"
# Dense checkpoint the clustering reads. Its lm_head IS the matrix being replaced,
# so its row directions are the structure a block has to reproduce. Must be at the
# SAME vocabulary as the runs.
CLUSTER_CKPT="${CLUSTER_CKPT:-out/c00_sch_phase0/d4/DENSE_softmax_s1/depth_4/ckpt_base/base/model_000462.pt}"
PERM_DIR="${PERM_DIR:-perms}"

if [ "$VOCAB" -eq 131072 ]; then
    TOK="${TOKENIZER_DIR:-${TOKENIZER_DIR_131K:-tokenizer_131k}}"
else
    TOK="${TOKENIZER_DIR:-tokenizer}"
fi
if ! python3 -m scripts.ensure_tokenizer --vocab-size "$VOCAB" --tokenizer-dir "$TOK" \
        --data-dir "${DATA_DIR:-data}" ${MAX_SHARDS:+--max-shards "$MAX_SHARDS"}; then
    echo "could not prepare the tokenizer at '${TOK}'; nothing was run."
    exit 1
fi
mkdir -p "$OUT_BASE" "$PERM_DIR"

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
LOGFILE="${SWEEP_LOG:-${OUT_BASE}/c11_d${DEPTH}.log}"
STATE="${OUT_BASE}/c11_state_d${DEPTH}.json"
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

# The budget every arm spends, from the reference configuration. Derived, never
# typed: a hand-matched budget is right once and then silently compares two
# different costs the moment m1, m2 or the reference rank moves.
budget_rank() {   # $1 = m2 -> the residual rank that keeps this arm iso-cost, or -1
    python3 -c "
d, V, m1 = $MODEL_DIM, $VOCAB, $M1
budget = d*(m1*$REF_M2) + V*m1 + $REF_RANK*(d + V)
r = (budget - d*(m1*$1) - V*m1) / float(d + V)
print(-1 if r < 0 else round(r))"
}

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
echo "  C11: block count at fixed cost, V=${VOCAB}, depth ${DEPTH}, d=${MODEL_DIM}"
echo "  m1=${M1} fixed   reference m2=${REF_M2}, r=${REF_RANK}   block counts: ${M2S}"
echo "  target tokens ${TARGET_TOKENS}   device-batch ${DEVICE_BATCH_SIZE}"
printf "  %-6s %-7s %-11s %-6s %s\n" "m2" "M" "block_out" "r" "arms"
for m2 in $M2S; do
    R=$(budget_rank "$m2")
    [ "$R" -lt 0 ] && { printf "  %-6s %-7s %-11s %-6s over budget, will skip\n" \
        "$m2" "$(( M1 * m2 ))" "$(( VOCAB / m2 ))" "-"; continue; }
    A=""
    [ "$RUN_ID" -eq 1 ]  && A="${A}id "
    [ "$RUN_CLU" -eq 1 ] && A="${A}clustered"
    printf "  %-6s %-7s %-11s %-6s %s\n" "$m2" "$(( M1 * m2 ))" "$(( VOCAB / m2 ))" "$R" "$A"
done
echo "============================================================"

[ "$RUN_REF" -eq 1 ] && run "MON_m2_${REF_M2}_ref" --models base --use-code-head 1 \
    --sch-head-type monarch --sch-max-m "$(( M1 * REF_M2 ))" --sch-monarch-m1 "$M1" \
    --sch-residual-rank "$REF_RANK" $PROBE

for m2 in $M2S; do
    R=$(budget_rank "$m2")
    if [ "$R" -lt 0 ]; then
        echo "SKIP  m2=${m2}: d*M alone exceeds the head budget; no residual rank"
        echo "      makes this arm iso-cost. Lower m2, or raise REF_RANK."
        continue
    fi
    M=$(( M1 * m2 ))
    MON="--models base --use-code-head 1 --sch-head-type monarch \
      --sch-max-m $M --sch-monarch-m1 $M1 --sch-residual-rank $R $PROBE"

    [ "$RUN_ID" -eq 1 ] && run "MON_m2_${m2}_id" $MON

    if [ "$RUN_CLU" -eq 1 ]; then
        PERM="${PERM_DIR}/cluster_v${VOCAB}_m2_${m2}.pt"
        if [ ! -f "$PERM" ]; then
            if [ ! -f "$CLUSTER_CKPT" ]; then
                echo "SKIP  MON_m2_${m2}_clu: no ${PERM} and CLUSTER_CKPT='${CLUSTER_CKPT}'"
                echo "      does not exist. Point it at a DENSE checkpoint at V=${VOCAB}."
                continue
            fi
            echo "[c11] clustering ${VOCAB} lm_head rows into ${m2} balanced blocks"
            if ! python3 -m scripts.build_vocab_permutation --mode cluster \
                    --checkpoint "$CLUSTER_CKPT" --source lm_head \
                    --vocab-size "$VOCAB" --blocks "$m2" --iters "$CLUSTER_ITERS" \
                    --out "$PERM"; then
                echo "SKIP  MON_m2_${m2}_clu: could not build ${PERM}"
                continue
            fi
        fi
        run "MON_m2_${m2}_clu" $MON --sch-monarch-perm file --sch-monarch-perm-path "$PERM"
    fi
done

done

echo ""
echo "============================================================"
echo "  C11 complete."
echo ""
echo "    python -m scripts.sweep_report ${OUT_BASE}/d<DEPTH>"
echo ""
echo "  Read every arm against the reference at the same budget, not against dense:"
echo "  the whole sweep is one cost with one variable moving."
echo ""
echo "  d4 V=32,768 reference (m2=8, m1=32, r=128):  bpb 1.175721"
echo "  cost-matched pure low-rank at that budget:   bpb 1.217850"
echo ""
echo "  The oracle predicts, in captured energy over pure low-rank:"
echo "    m2= 8  token-id +0.0063   clustered +0.0131"
echo "    m2=64  token-id +0.0048   clustered +0.0191   <- 3.0x the trained arm"
echo ""
echo "  If both m2=64 arms improve, block count is the win and clustering is"
echo "  decoration. If only the clustered arm improves, subspace coherence is the"
echo "  mechanism, and scripts/head_oracle.py can be trusted to design the next one"
echo "  without spending a run on it."
echo "============================================================"
