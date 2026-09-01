#!/usr/bin/env bash
# ============================================================================
# P10: iso-token control at d8. MST vs MoL vs dense, same token budget.
#
# WHY
#   The default budget is 10.5 * total matmul params, so at d8 the three arms
#   draw wildly different amounts of data:
#       MoL   0.59B    dense 0.44B    MST 0.28B      (MoL = 2.10x MST)
#   MoL has 3.75x total/active params, so the rule hands it the most. The
#   FLOPs/token axis never charges for that, which is why MoL scores 1.178x
#   there while tying MST exactly on training FLOPs (0.665x vs 0.666x).
#   Ternovtsii & Bilak trained every 1.3B arm at 20B tokens -- iso-token -- so
#   their setup gave MoL no data advantage, which is why they report no gain
#   of this size. This run removes the confound.
#
# BUDGET = 0.6B
#   Chosen as MoL's own compute-optimal point (0.59B), so MoL is not
#   undertrained and cannot be dismissed on that ground. MST gets 2.1x its
#   usual allocation; if its bpb moves a lot, that is itself the answer.
#   1B was considered and rejected: it overtrains everything relative to
#   compute-optimal (3.6x MST's budget), and the Pareto claim lives at the
#   compute-optimal point.
#
# NO VALUE-EMBEDDING VARIANTS. Both arms run the plain shared table, so the
# only difference is the architecture. G3 / per-block VE are measured separately.
#
# PREDICTION, stated before the run so it can be wrong:
#   If MoL's edge was data, its bpb rises from 0.983 toward MST's ~1.03 and the
#   two land close. If MoL still wins clearly at equal tokens, the gain is real
#   and our MoL implementation needs auditing -- start with OPEN_QUESTIONS Q2
#   (shared block inside vs outside the routing softmax), then whether routed
#   attention actually restricts.
#
#   bash scripts/p10_isotoken.sh            # d8, 0.6B tokens
#   TOKENS=1000000000 bash scripts/p10_isotoken.sh
#   bash scripts/p10_isotoken.sh 8 16
# ============================================================================
set -o pipefail

FORCE=0
SEEDS=1
DEPTHS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        [0-9]*)  DEPTHS+=("$1"); shift ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done
[ ${#DEPTHS[@]} -eq 0 ] && DEPTHS=(8)

TOKENS="${TOKENS:-600000000}"
N_SUBS="${N_SUBS:-4}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
OUT_BASE="${OUT_BASE:-out/p10_isotoken}"
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

# Same as p08 except --target-tokens, which overrides the param-data ratio.
COMMON="--device-batch-size ${DEVICE_BATCH_SIZE:-32} --total-batch-size -1 \
  --use-onecycle 0 --log-every ${LOG_EVERY:-200} --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 --target-param-data-ratio 10.5 \
  --warmup-ratio 0.005 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --research-dim -1 --target-active-params 0 \
  --save-every 200 --eval-every -1 --target-tokens ${TOKENS}"
[ -n "${MAX_SHARDS:-}" ] && COMMON="$COMMON --max-shards $MAX_SHARDS"

run() {
    local tag="$1"; shift
    local depth="$1"; shift
    for s in $(seq 1 "$SEEDS"); do
        local t="${tag}_s${s}"
        if done_already "$t"; then echo "SKIP $t"; continue; fi
        echo ""; echo "=== $t (depth $depth, ${TOKENS} tokens) ==="
        local dir="${OUT_BASE}/d${depth}/${t}"
        [ "$FORCE" -eq 1 ] && rm -rf "$dir"
        if bash scripts/research_sweep.sh $COMMON --out-dir "$dir" --seed "$s" \
               "$@" "$depth" 2>&1 | tee -a "$LOGFILE"; then
            mark_done "$t"; echo "OK $t"
        else
            echo "FAIL $t"
        fi
    done
}

for DEPTH in "${DEPTHS[@]}"; do
MODEL_DIM=$(( ((DEPTH * ASPECT_RATIO + 127) / 128) * 128 ))
SUB_DIM=$(( MODEL_DIM / N_SUBS ))
LOGFILE="${OUT_BASE}/p10_d${DEPTH}.log"
STATE="${OUT_BASE}/p10_state_d${DEPTH}.json"
[ "$FORCE" -eq 1 ] && rm -f "$STATE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"

echo "============================================================"
echo "  P10 iso-token   depth ${DEPTH}  D=${MODEL_DIM}  d=${SUB_DIM}"
echo "  tokens ${TOKENS}   seeds ${SEEDS}   out ${OUT_BASE}"
echo "============================================================"

# MST, plain value embeddings (no G3)
run ISO_mst "$DEPTH" \
    --use-mst 1 --models base --mst-n-subs "$N_SUBS" --mst-sub-dim "$SUB_DIM" \
    --mst-head-dim 0 --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 0 --mst-ffn-mode standard \
    --mst-transition-mode aggregate_distribute \
    --mst-final-mode concat_proj --mst-final-topk 0 \
    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0 \
    --mst-grad-equalize 1 --mst-block-diagonal-muon 1 \
    --mst-transition-width-mult 4.0 --mst-sub-lr-scale 2.0 \
    --mst-multi-scale-windows 1 \
    --mst-sub-head-dim 64 --mst-compose-windows 1 --mst-wo-mode dense \
    --mst-stream-topk 1 --mst-stream-router-noise 1.0

# MoL, plain shared value-embedding table (no per-block VE)
run ISO_mol "$DEPTH" \
    --use-mol 1 --models base --mol-n-blocks 15 --mol-n-shared 1 --mol-topk 3 \
    --mol-thin-dim "$SUB_DIM" --mol-head-dim 64 --mol-ffn-mult 4.0 \
    --mol-router-aux 0.05 --mol-routed-attn softmax --mol-dispatch 1

# Dense at the same budget. Makes it a real triple and costs one short run.
run ISO_dense "$DEPTH" --models base

echo ""
echo "============================================================"
echo "  done depth ${DEPTH}: ${OUT_BASE}/d${DEPTH}/"
echo "  All three saw ${TOKENS} tokens, so bpb is directly comparable."
echo "  FLOPs/token still differs; divide by it for the per-token axis."
echo "============================================================"
done
