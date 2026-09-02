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
# ATTENTION-GATING ARMS were added 2026-09-01: ISO_mst_ve (control), ISO_mst_ve_gattn,
# ISO_mst_ve_gattn_s1, ISO_mst_ve_s1. They share the iso-token budget for the same reason
# the original three do: gating attention changes the active parameter count, so an
# own-budget run would confound the architecture change with a data change. Unlike the
# three above they DO use per-stream value embeddings, because they compare MST against
# MST and the headline uses them; see the block above those arms for why that is not
# cosmetic.
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
#run ISO_mst "$DEPTH" \
#    --use-mst 1 --models base --mst-n-subs "$N_SUBS" --mst-sub-dim "$SUB_DIM" \
#    --mst-head-dim 0 --mst-input-mode learned_proj \
#    --mst-routing-mode soft_weighted --mst-routing-topk 0 --mst-ffn-mode standard \
#    --mst-transition-mode aggregate_distribute \
#    --mst-final-mode concat_proj --mst-final-topk 0 \
#    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0 \
#    --mst-grad-equalize 1 --mst-block-diagonal-muon 1 \
#    --mst-transition-width-mult 4.0 --mst-sub-lr-scale 2.0 \
#    --mst-multi-scale-windows 1 \
#    --mst-sub-head-dim 64 --mst-compose-windows 1 --mst-wo-mode dense \
#    --mst-stream-topk 1 --mst-stream-router-noise 1.0
#
# MoL, plain shared value-embedding table (no per-block VE)
#run ISO_mol "$DEPTH" \
#    --use-mol 1 --models base --mol-n-blocks 15 --mol-n-shared 1 --mol-topk 3 \
#    --mol-thin-dim "$SUB_DIM" --mol-head-dim 64 --mol-ffn-mult 4.0 \
#    --mol-router-aux 0.05 --mol-routed-attn softmax --mol-dispatch 1

# ---------------------------------------------------------------- gate_attn
# Does gating ATTENTION as well as the FFN pay, and does it need a shared stream?
#
# WHY THIS IS THE LEVER. mst_stream_topk currently gates the FFN only, because a
# skipped token stops being a key/value for that stream and the attention semantics
# change per stream. Measured at L=24 (scripts/p11_active_params.py), the FFN is 44.4%
# of a layer's matmul parameters and attention is 38.9%, so k=1-of-4 on the FFN alone
# removes 33.3% of a layer and 66.7% keeps running. Adding attention takes it to 62.5%
# removed, 37.5% active.
#
# That is the difference between the two ends of MoL's own speedup curve (their 2.3):
# 1.53x forward speedup at 57% active, 2.85x at 20%. At 66.7% active there is nothing
# to win, which is why the p08 `dispatch` group could not have shown a speedup no matter
# how the gather/scatter was implemented.
#
# WHY THE SHARED ARM IS NOT OPTIONAL. Gating attention is exactly what creates MoL's
# "attention coverage problem" (their 3.1): at 3-of-15 each block sees 20% of the
# sequence, and softmax-only 3-of-15 scores WORSE than 3-of-5 (34.73 vs 32.04 PPL)
# despite 2.2x the parameters. Their fix is 3.2's Shared + Routed topology, an
# always-active block carrying global context at every layer. So gate_attn ALONE tests
# whether the coverage problem bites us, and gate_attn + shared tests whether their fix
# transfers. Running only the first would confound "attention gating is bad" with
# "attention gating without coverage is bad".
#
# NOTE ON SPARSITY LEVELS. These arms are deliberately NOT iso-active-FLOPs with each
# other; S=1,k=1 runs two of four streams against gate_attn's one. That is the point:
# the axis is bpb against active FLOPs, and each arm places itself on it. estimate_flops
# now discounts by 1 - (S+k)/N, so the shared stream is charged for (it was NOT before
# 2026-09-01, which made S=1,k=1 look identical in cost to S=0,k=1; see
# tests/test_mst_parity_fixes.py::test_shared_streams_are_not_free_in_the_flops_accounting).
#
# What each arm actually costs, from scripts/p11_active_params.py. The 33.3% / 62.5%
# figures above are shares of a LAYER's matmul parameters; these are whole-model active
# FLOPs per token, which also carry embeddings, the head, and the attention QK/AV term:
#
#            arm            S+k   d8 act.FLOPs  act/total   d24 act.FLOPs  act/total
#   ISO_mst (control)        1      1.562e8      0.892        1.481e9      0.744
#   ISO_mst_gattn            1      1.382e8      0.790        1.192e9      0.599
#   ISO_mst_gattn_s1         2      1.505e8      0.860        1.459e9      0.733
#   ISO_mst_s1               2      1.625e8      0.928        1.651e9      0.829
#
# So gate_attn buys 11.5% of active FLOPs at d8 and 19.5% at d24: the saving GROWS with
# depth, because attention's share of a layer grows with d. Read the d8 result as a
# lower bound on what it is worth at the depths the paper reports.
# ══ THESE ARMS RUN PER-STREAM VALUE EMBEDDINGS. THE THREE ABOVE DO NOT. ══
# The "no value-embedding variants" rule at the top exists to keep MST vs MoL vs dense
# an architecture-only comparison. It does not apply here, because these arms compare
# MST against MST, and the headline SP2_k1 config uses --mst-per-stream-ve 1 (G3). A
# delta measured on plain VE would not transfer, and NOT for a bookkeeping reason:
#
#   VE enters the model through v, inside attention:
#       v5 = v5 + gates.unsqueeze(-1) * ve_heads       # mst.py, attention block
#       if stream_w is not None and self._stream_gate_attn:
#           attn_out = stream_w.unsqueeze(-1) * attn_out
#
#   so gating attention discards the gated stream's VE contribution along with its
#   attention output. Under a SHARED table every stream injects the same vector, so a
#   token still receives that content through whichever stream it did select. Under
#   per-stream VE each stream reads its OWN slice, and gating stream j destroys slice j
#   for that token with nothing else carrying it. Attention gating therefore costs
#   STRICTLY MORE under G3, and a plain-VE measurement is an optimistic estimate of what
#   the headline config would pay. That is the failure mode this ordering avoids:
#   adopting gate_attn on a plain-VE result and having it regress at the headline.
#
# This is why ISO_mst_ve below exists and the plain-VE ISO_mst above is NOT the control
# for these arms. It costs one extra run; without it there is no valid delta.
#
# FLOPs are unaffected by the VE choice (value embeddings are lookups and are excluded
# from the FLOPs formula), so the cost table above holds for these arms too. Only the
# total parameter count moves, 60.3M to 110.6M at d8, and only bpb is at stake.
MST_ISO_VE="--use-mst 1 --models base --mst-n-subs $N_SUBS --mst-sub-dim $SUB_DIM \
    --mst-head-dim 0 --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 0 --mst-ffn-mode standard \
    --mst-transition-mode aggregate_distribute \
    --mst-final-mode concat_proj --mst-final-topk 0 \
    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0 \
    --mst-grad-equalize 1 --mst-block-diagonal-muon 1 \
    --mst-transition-width-mult 4.0 --mst-sub-lr-scale 2.0 \
    --mst-multi-scale-windows 1 \
    --mst-sub-head-dim 64 --mst-compose-windows 1 --mst-wo-mode dense \
    --mst-per-stream-ve 1 --target-tokens -1"

# The control every arm below is read against: the headline SP2_k1 config at this
# budget. Not optional.
run ISO_mst_ve "$DEPTH" $MST_ISO_VE \
    --mst-stream-topk 1 --mst-stream-router-noise 1.0 --target-tokens 3657905664

# Attention gated too, no shared stream. 1 of 4 streams active.
#run ISO_mst_ve_gattn "$DEPTH" $MST_ISO_VE \
#    --mst-stream-topk 1 --mst-stream-router-noise 1.0 \
#    --mst-stream-gate-attn 1 --target-active-params 1

# MoL's Shared + Routed topology: stream 0 always on, top-1 over the remaining 3.
# 2 of 4 streams active. Costs more than the arm above and is expected to recover the
# coverage it loses.
#run ISO_mst_ve_gattn_s1 "$DEPTH" $MST_ISO_VE \
#    --mst-stream-shared 1 --mst-stream-topk 1 --mst-stream-router-noise 1.0 \
#    --mst-stream-gate-attn 1 --target-active-params 1

# DROP THIS ONE FIRST if compute is tight. Shared stream WITHOUT attention gating, to
# attribute a gattn_s1 win to coverage rather than to the extra always-on capacity. Only
# worth running once gattn_s1 has come back looking good.
#run ISO_mst_ve_s1 "$DEPTH" $MST_ISO_VE \
#    --mst-stream-shared 1 --mst-stream-topk 1 --mst-stream-router-noise 1.0 --target-active-params 1

# Dense at the same budget. Makes it a real triple and costs one short run.
#run ISO_dense "$DEPTH" --models base

echo ""
echo "============================================================"
echo "  done depth ${DEPTH}: ${OUT_BASE}/d${DEPTH}/"
echo "  Every arm saw ${TOKENS} tokens, so bpb is directly comparable."
echo "  FLOPs/token still differs; divide by it for the per-token axis."
echo "============================================================"
done
