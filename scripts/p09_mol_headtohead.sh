#!/usr/bin/env bash
# ============================================================================
# P09: MST versus MoL, head to head.
#
# WHY THIS SWEEP EXISTS
#   "Mixture of Layers with Hybrid Attention: Parallel Thin Blocks for Sparse
#   Transformer Compute" (Ternovtsii & Bilak, arXiv:2605.09516v1, 10 May 2026)
#   is prior art for MST, not adjacent work: parallel narrow transformer blocks,
#   top-k routing over them, a load-balance aux, gather/scatter dispatch, and
#   d_head pinned at 64. At four to twelve months before any realistic deadline it
#   has to be cited and beaten, and it is now the baseline that matters more than
#   dense. There is no official implementation; nanochat/mol.py is recovered from
#   their equations and validated against their published parameter counts.
#
#   The structural difference, and the whole thesis:
#     MoL  wraps each thin block in its own W_down/W_up, plumbing = D/(D+6*d_thin)
#     MST  partitions the residual stream,             plumbing = (N+8)/(13N+8)
#   MoL's grows as blocks narrow (their own §2.3 quotes 40/57/73% at d_thin
#   256/128/64); MST's depends only on the stream count and FALLS as you add
#   streams. scripts/p09_projection_overhead.py derives all of this with no GPU,
#   and reproduces their three published figures on the way. Run that first; this
#   sweep asks whether the parameter argument shows up as bpb.
#
# MATCHING
#   At L=8, D=512, MST runs N=4 streams of d=128. MoL's own design rule
#   (K_active x d_expert ~ d_model, their Appendix I) gives d_thin=128 at 4 active
#   blocks, so MOL_1plus3of5 is the matched-active-width arm. The d_thin sweep puts
#   their projection cliff on the same axis as our measured bpb.
#
# NOT THEIR HEADLINE CONFIG. Routed blocks use softmax, not Gated DeltaNet. Their
# §5.3 reports a dense DeltaNet control matching dense softmax within 0.01 PPL, so
# the structural gain is not the attention swap, and they run a MoL all-softmax
# control themselves. But their Table 2 prices DeltaNet at 0.85 PPL inside MoL at
# d_thin=256, so this is their architecture, not their best number. Say so in the
# paper rather than letting a reviewer find it.
#
#   bash scripts/p09_mol_headtohead.sh 8                  # default depth 8
#   bash scripts/p09_mol_headtohead.sh --group mol 8      # one group
#   bash scripts/p09_mol_headtohead.sh --seeds 2 8
#   bash scripts/p09_mol_headtohead.sh --force 8
# ============================================================================
set -o pipefail

FORCE=0
RUN_GROUPS="control mol width sparsity"
SEEDS=1
DEPTHS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)  FORCE=1; shift ;;
        --group)  RUN_GROUPS="$2"; shift 2 ;;
        --seeds)  SEEDS="$2"; shift 2 ;;
        [0-9]*)   DEPTHS+=("$1"); shift ;;
        *) echo "unknown arg: $1"; echo "usage: $0 [--force] [--group G] [--seeds N] [DEPTH ...]"; exit 1 ;;
    esac
done
[ ${#DEPTHS[@]} -eq 0 ] && DEPTHS=("${DEPTH:-8}")

N_SUBS="${N_SUBS:-4}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
OUT_BASE="${OUT_BASE:-out/p09_mol_headtohead}"
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

# Byte-identical to p08's COMMON so these numbers sit on the same axes as the MST
# results without any refitting.
COMMON="--device-batch-size ${DEVICE_BATCH_SIZE:-32} --total-batch-size -1 \
  --use-onecycle 0 --log-every ${LOG_EVERY:-200} --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 --target-param-data-ratio 10.5 \
  --warmup-ratio 0.005 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --research-dim -1 --target-tokens -1 --target-active-params 0 \
  --save-every 200 --eval-every -1"
[ -n "${MAX_SHARDS:-}" ] && COMMON="$COMMON --max-shards $MAX_SHARDS"

mst_config() {                            # mst_config <sub_dim> <n_subs>
    echo "--use-mst 1 --models base --mst-n-subs $2 --mst-sub-dim $1 \
      --mst-head-dim 0 --mst-input-mode learned_proj \
      --mst-routing-mode soft_weighted --mst-routing-topk 0 --mst-ffn-mode standard \
      --mst-transition-mode aggregate_distribute \
      --mst-final-mode concat_proj --mst-final-topk 0 \
      --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0 \
      --mst-grad-equalize 1 --mst-block-diagonal-muon 1 \
      --mst-transition-width-mult $2.0 --mst-sub-lr-scale 2.0 \
      --mst-multi-scale-windows 1"
}

mol_config() {                            # mol_config <n_blocks> <n_shared> <topk> <thin_dim>
    echo "--use-mol 1 --models base --mol-n-blocks $1 --mol-n-shared $2 \
      --mol-topk $3 --mol-thin-dim $4 --mol-head-dim 64 --mol-ffn-mult 4.0 \
      --mol-router-aux 0.05 --mol-routed-attn softmax --mol-dispatch 0"
}

run() {                                   # run <tag> <depth> <flags...>
    local tag="$1"; shift
    local depth="$1"; shift
    for s in $(seq 1 "$SEEDS"); do
        local t="${tag}_s${s}"
        if done_already "$t"; then echo "⏭  $t"; continue; fi
        echo ""
        echo "━━━ $t  (depth $depth) ━━━"
        local dir="${OUT_BASE}/d${depth}/${t}"
        [ "$FORCE" -eq 1 ] && rm -rf "$dir"
        if bash scripts/research_sweep.sh $COMMON --out-dir "$dir" --seed "$s" \
               "$@" "$depth" 2>&1 | tee -a "$LOGFILE"; then
            mark_done "$t"; echo "✅  $t"
        else
            echo "❌  $t failed; will retry on the next invocation"
        fi
    done
}

has() { echo " $RUN_GROUPS " | grep -q " $1 "; }

for DEPTH in "${DEPTHS[@]}"; do

MODEL_DIM=$(( ((DEPTH * ASPECT_RATIO + 127) / 128) * 128 ))
SUB_DIM=$(( MODEL_DIM / N_SUBS ))
LOGFILE="${SWEEP_LOG:-${OUT_BASE}/p09_d${DEPTH}.log}"
STATE="${OUT_BASE}/p09_state_d${DEPTH}.json"
[ "$FORCE" -eq 1 ] && rm -f "$STATE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"

MST_FULL="$(mst_config "$SUB_DIM" "$N_SUBS")"
BEST="--mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1 --mst-wo-mode dense"

echo "════════════════════════════════════════════════════════════"
echo "  P09 MST versus MoL"
echo "  depth ${DEPTH}   D=${MODEL_DIM}   MST N=${N_SUBS} d=${SUB_DIM}"
echo "  seeds ${SEEDS}   groups: ${RUN_GROUPS}"
echo "  out ${OUT_BASE}"
echo "════════════════════════════════════════════════════════════"

# ---------------------------------------------------------------- control
# Our side of the comparison, so every MoL delta is read against the same sweep
# rather than against p08 numbers from a different run.
if has control; then
    echo ""; echo "### CONTROL: dense, and MST's current best"
    run CTRL_dense    "$DEPTH" --models base
    run CTRL_mst_best "$DEPTH" $MST_FULL $BEST
fi

# ---------------------------------------------------------------- mol
# The matched-active-width arm. 1 shared + top-3 of 4 routed = 4 active blocks at
# d_thin=128, against MST's 4 streams at d=128.
if has mol; then
    echo ""; echo "### MOL: matched active width against MST"
    run MOL_1plus3of5 "$DEPTH" $(mol_config 5 1 3 "$SUB_DIM")
    # No shared block: their Table 1 configuration, and the arm that shows why
    # §3.2 had to introduce one (softmax-only sparse routing loses coverage).
    run MOL_0plus3of5 "$DEPTH" $(mol_config 5 0 3 "$SUB_DIM")
    # All-active. Their Table 1 prices selective activation at 0.99 PPL over uniform
    # composition, so this separates "narrow blocks" from "routing between them".
    run MOL_allactive "$DEPTH" $(mol_config 4 4 1 "$SUB_DIM")
fi

# ---------------------------------------------------------------- width
# THE decisive group. Their projection overhead is 40% at d_thin=D/4 and 57% at
# D/8; MST's coupling goes the other way (21% at N=4, 15% at N=8). If the parameter
# argument is real, MoL should degrade faster than MST as both narrow.
if has width; then
    echo ""; echo "### WIDTH: does the projection cliff show up as bpb?"
    HALF=$(( SUB_DIM / 2 ))
    QUARTER=$(( SUB_DIM / 4 ))
    if [ "$HALF" -ge 64 ] && [ $(( HALF % 64 )) -eq 0 ]; then
        run MOL_thin_half "$DEPTH" $(mol_config 9 1 7 "$HALF")
        run MST_n8        "$DEPTH" $(mst_config "$HALF" 8) $BEST
    else
        echo "⚠  skipping half-width: ${HALF} is not a usable head-dim-64 width"
    fi
    if [ "$QUARTER" -ge 64 ] && [ $(( QUARTER % 64 )) -eq 0 ]; then
        run MOL_thin_quarter "$DEPTH" $(mol_config 17 1 15 "$QUARTER")
        run MST_n16          "$DEPTH" $(mst_config "$QUARTER" 16) $BEST
    else
        echo "⚠  skipping quarter-width: ${QUARTER} is not a usable head-dim-64 width"
    fi
fi

# ---------------------------------------------------------------- sparsity
# Both architectures at their own best sparse setting, which is the number the
# paper's headline comparison actually rests on.
if has sparsity; then
    echo ""; echo "### SPARSITY: each architecture's conditional-execution arm"
    run MST_sp2_k1 "$DEPTH" $MST_FULL $BEST \
        --mst-stream-topk 1 --mst-stream-router-noise 1.0
    run MOL_1plus1of5 "$DEPTH" $(mol_config 5 1 1 "$SUB_DIM")
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  P09 complete for depth ${DEPTH}"
echo "  results: ${OUT_BASE}/d${DEPTH}/mst_results*.csv   log: ${LOGFILE}"
echo ""
echo "  Read it on FLOPs-vs-bpb, and report iso-active-params alongside, because"
echo "  iso-active is the axis MoL itself claims (their §5.5: they LOSE to dense"
echo "  by 3.01 PPL at iso-total, and win only at iso-active)."
echo "════════════════════════════════════════════════════════════"

done
