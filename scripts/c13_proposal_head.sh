#!/usr/bin/env bash
# ============================================================================
# C13: exact dense head, approximate normalisation.
#
# WHAT IS DIFFERENT ABOUT THIS ONE
#   c00 through c12 all approximate the head MATRIX and normalise exactly: codes,
#   Monarch blocks, low-rank, frequency tiers. Every one of them trades capacity for
#   FLOPs, and every one is bounded by how much capacity the head can lose. This
#   trades nothing. The V x d weight is unchanged and unfactorised; only a few
#   thousand of its rows are read per token.
#
#     z_prop = B (A h)          rank-c proposal over all V            V*c MACs
#     K      = topK(z_prop) U {target}
#     z_K    = weight[K] . h    EXACT logits from the full-rank head  K*d MACs
#     tail   = importance sample S from q ∝ exp(z_prop) off K         S*d MACs
#     loss   = logaddexp(lse(z_K), tail) - z_target
#
#   The only approximation is log Z, and since the loss is z_target - log Z, an error
#   of e nats per token is exactly e/(ln2 * bytes_per_token) bits per byte. That is
#   measurable rather than inferred, and it was measured offline on the trained
#   V=131,072 heads before any of this was built (scripts/proposal_probe.py):
#
#     depth   proposal   K       error      head FLOPs   margin over dense
#        4    rank 16    4096    0.045 nats      9.8x        +0.1130
#        8    rank 32    4096    0.036 nats      9.8x        +0.0666
#       12    rank 32    8192    0.035 nats      8.9x        +0.0243
#
#   For scale: at depth 12 the best Monarch arm was +0.0016 and a completely FREE
#   head would be +0.0405. This captures 60-72% of that ceiling at every depth.
#
# THE TWO THINGS THAT MAKE IT WORK, BOTH OF WHICH ARE ARMS HERE
#   The tail must be SAMPLED, not substituted. Plugging the cheap logits in for the
#   non-selected words is biased: -0.03 nats at depth 4 and -1.6 at depth 12, enough
#   to lose everything the FLOPs bought. PROP_plugin (S=0) is that ablation.
#
#   The proposal must be trained to RANK, not to reconstruct. A truncated SVD is the
#   optimal rank-c approximation and a poor ranker: at depth 12, rank 32 by SVD keeps
#   the true top-1 inside K 76.1% of the time against 98.6% trained, a 14x difference
#   in log-Z error. That is why the SVD's rank requirement grew with d (0.06d, 0.13d,
#   0.33d) and why a trained one does not (16 to 32 at every depth). Here the proposal
#   is trained jointly by the model's own loss, so the rank sweep is the evidence.
#
# WHAT WOULD KILL IT
#   Rows never selected never receive gradient. Importance sampling covers them
#   stochastically, which is how sampled softmax has always trained, but whether that
#   suffices for 131,072 rows is the open question and this is the run that answers
#   it. Watch bpb against LOWRANK_M259 and dense, and watch the warmup boundary: a
#   step change there means the proposal was not ready when the exact path stopped.
#
#   bash scripts/c13_proposal_head.sh                    # d8, V=131,072
#   RANKS="16 32" TOPKS="4096 8192" bash scripts/c13_proposal_head.sh 8
#   bash scripts/c13_proposal_head.sh --with-plugin --with-lowrank 12
# ============================================================================
set -o pipefail

FORCE=0
SEEDS=1
RUN_MAIN=1
RUN_PLUGIN=0
RUN_LOWRANK=0
DEPTHS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)        FORCE=1; shift ;;
        --seeds)        SEEDS="$2"; shift 2 ;;
        --with-plugin)  RUN_PLUGIN=1; shift ;;
        --with-lowrank) RUN_LOWRANK=1; shift ;;
        --plugin-only)  RUN_MAIN=0; RUN_PLUGIN=1; shift ;;
        [0-9]*)         DEPTHS+=("$1"); shift ;;
        *) echo "unknown arg: $1"
           echo "usage: $0 [--force] [--seeds N] [--with-plugin] [--with-lowrank]"
           echo "       [--plugin-only] [DEPTH ...]"
           exit 1 ;;
    esac
done
[ ${#DEPTHS[@]} -eq 0 ] && DEPTHS=(8)

VOCAB="${VOCAB:-131072}"
RANKS="${RANKS:-32}"
TOPKS="${TOPKS:-4096}"
SAMPLES="${SAMPLES:-1024}"
WARMUP="${WARMUP:-200}"          # steps of exact softmax; the proposal is noise at init
PCHUNK="${PCHUNK:-128}"          # tokens per gather: weight[idx] is (chunk, K, d)
VCHUNK="${VCHUNK:-16384}"
# Budget reference, the Monarch arm already trained: d*M + V*m1 + r*(d+V).
REF_M="${REF_M:-1024}"; REF_M1="${REF_M1:-32}"; REF_RANK="${REF_RANK:-224}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
OUT_BASE="${OUT_BASE:-out/c13_proposal_head}"
RANK_CONTEXTS="${RANK_CONTEXTS:-16384}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"

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
LOGFILE="${SWEEP_LOG:-${OUT_BASE}/c13_d${DEPTH}.log}"
STATE="${OUT_BASE}/c13_state_d${DEPTH}.json"
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
PROP="--models base --use-code-head 1 --sch-head-type proposal \
  --sch-proposal-warmup $WARMUP --sch-proposal-chunk $PCHUNK \
  --sch-proposal-vocab-chunk $VCHUNK $PROBE"

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
echo "  C13: proposal head, V=${VOCAB}, depth ${DEPTH}, d=${MODEL_DIM}"
echo "  ranks ${RANKS}   K ${TOPKS}   S ${SAMPLES}   warmup ${WARMUP} steps"
printf "  %-8s %-7s %-7s %-11s %s\n" "rank" "K" "head" "vs dense" "vs the trained arms"
for c in $RANKS; do for K in $TOPKS; do
    python3 -c "
V,d,c,K,S = $VOCAB,$MODEL_DIM,$c,$K,$SAMPLES
head = V*c + c*d + (K+S)*d
ref  = d*$REF_M + V*$REF_M1 + $REF_RANK*(d+V)
print(f'  {c:<8} {K:<7} {head/1e6:6.2f}M {V*d/head:10.1f}x  {head/ref:.3f}x their budget')"
done; done
echo "  target tokens ${TARGET_TOKENS}   device-batch ${DEVICE_BATCH_SIZE}"
echo "============================================================"

if [ "$RUN_MAIN" -eq 1 ]; then
    for c in $RANKS; do for K in $TOPKS; do
        run "PROP_c${c}_K${K}" $PROP --sch-proposal-rank "$c" \
            --sch-proposal-topk "$K" --sch-proposal-samples "$SAMPLES"
    done; done
fi
# The estimator ablation: substitute the cheap logits for the tail instead of
# sampling it. Offline this was biased by -0.03 nats at d4 and -1.6 at d12.
if [ "$RUN_PLUGIN" -eq 1 ]; then
    for c in $RANKS; do for K in $TOPKS; do
        run "PROP_plugin_c${c}_K${K}" $PROP --sch-proposal-rank "$c" \
            --sch-proposal-topk "$K" --sch-proposal-samples 0
    done; done
fi
if [ "$RUN_LOWRANK" -eq 1 ]; then
    R=$(python3 -c "print(int((${MODEL_DIM}*${REF_M} + ${VOCAB}*${REF_M1} + ${REF_RANK}*(${MODEL_DIM}+${VOCAB})) / (${MODEL_DIM}+${VOCAB})))")
    run "LOWRANK_M${R}" --models base --use-code-head 1 \
        --sch-phi-mode learned --sch-max-m "$R" $PROBE
fi

done

echo ""
echo "============================================================"
echo "  C13 complete."
echo ""
echo "    python -m scripts.sweep_report ${OUT_BASE}/d<DEPTH>"
echo ""
echo "  These arms are NOT at the budget of c10-c12: they are far cheaper, which is"
echo "  the point. Read them on the FLOPs-bpb curve, not against a fixed budget."
echo "  d8 V=131,072 reference points:"
echo "    dense                          0.902908 at 98.13M MACs/token"
echo "    LOWRANK_M259                   0.918721 at 65.03M"
echo "    best structured (c11)          0.911994 at 65.03M"
echo "    PROP rank 32 K=4096            predicted ~0.906 at 37.85M"
echo ""
echo "  Then check the three things that decide whether the mechanism is sound:"
echo "    1. the loss curve across step ${WARMUP}, where the exact path switches off."
echo "       A step change means the proposal was not ready."
echo "    2. PROP against PROP_plugin. The gap IS the tail estimator, and offline it"
echo "       was the difference between working and not."
echo "    3. rank 16 against rank 32. If 16 holds, the proposal really is constant"
echo "       cost in d and the saving does not decay with model size."
echo "============================================================"
