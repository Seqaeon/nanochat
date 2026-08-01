#!/usr/bin/env bash
# ============================================================================
# P32: the ablation grid for the MST paper (Group C).
#
#   C1  main ablation, Table 7          14 conditions
#   C2  appendix extras, Table 9         5 conditions
#   C3  coupling variants that failed   12 conditions
#   C4  transfer check at d12            2 conditions
#
# Run at L=8 where a full grid is affordable (~7 min/run). The one intervention
# measurable at two depths is repeated at d12 to support the transfer argument
# in Section 6.
#
# ORDERING IS DELIBERATE. The dense head_dim=32 control runs FIRST, before the
# other 41 runs. If a dense model with the same small head dimension closes the
# gap, the paper's structural claim is in trouble and you want to know that for
# 20 minutes of compute rather than 12 hours.
#
#   bash scripts/p32_paper_ablations.sh              # everything, 3 seeds
#   bash scripts/p32_paper_ablations.sh --group c1   # one group
#   bash scripts/p32_paper_ablations.sh --seeds 1    # single seed, quick pass
#   bash scripts/p32_paper_ablations.sh --force      # ignore completion state
# ============================================================================
set -o pipefail

FORCE=0
RUN_GROUPS="control c1 c2 c3 c4"
SEEDS=3
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)  FORCE=1; shift ;;
        --group)  RUN_GROUPS="$2"; shift 2 ;;
        --seeds)  SEEDS="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

DEPTH="${DEPTH:-8}"
N_SUBS="${N_SUBS:-4}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
MODEL_DIM=$(( ((DEPTH * ASPECT_RATIO + 127) / 128) * 128 ))
SUB_DIM=$(( MODEL_DIM / N_SUBS ))

OUT_BASE="${OUT_BASE:-out/p32_paper_ablations}"
LOGFILE="${SWEEP_LOG:-${OUT_BASE}/p32_d${DEPTH}.log}"
STATE="${OUT_BASE}/p32_state_d${DEPTH}.json"
mkdir -p "$OUT_BASE"
[ "$FORCE" -eq 1 ] && rm -f "$STATE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"

echo "════════════════════════════════════════════════════════════"
echo "  P32 paper ablations"
echo "  depth ${DEPTH}   D=${MODEL_DIM}   N=${N_SUBS}   d=${SUB_DIM}"
echo "  seeds ${SEEDS}   groups: ${RUN_GROUPS}"
echo "  out ${OUT_BASE}"
echo "════════════════════════════════════════════════════════════"

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

COMMON="--device-batch-size ${DEVICE_BATCH_SIZE:-32} --total-batch-size -1 \
  --use-onecycle 0 --log-every ${LOG_EVERY:-200} --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 --target-param-data-ratio 10.5 \
  --warmup-ratio 0.005 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --research-dim -1 --target-tokens -1 --target-active-params 0 \
  --save-every 200 --eval-every -1"
[ -n "${MAX_SHARDS:-}" ] && COMMON="$COMMON --max-shards $MAX_SHARDS"

# The MST partition without any of the Section 3.3 recipe.
MST_PLAIN="--use-mst 1 --models base --mst-n-subs ${N_SUBS} --mst-sub-dim ${SUB_DIM} \
  --mst-head-dim 0 --mst-input-mode learned_proj \
  --mst-routing-mode soft_weighted --mst-routing-topk 0 --mst-ffn-mode standard \
  --mst-transition-mode aggregate_distribute \
  --mst-final-mode concat_proj --mst-final-topk 0 \
  --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0"

# The full recipe, i.e. the model the paper proposes.
RECIPE="--mst-grad-equalize 1 --mst-block-diagonal-muon 1 \
  --mst-transition-width-mult ${N_SUBS}.0 --mst-sub-lr-scale 2.0"
MST_FULL="$MST_PLAIN $RECIPE --mst-multi-scale-windows 1"

run() {                       # run <tag> <depth> <flags...>
    local tag="$1"; shift
    local depth="$1"; shift
    for s in $(seq 1 "$SEEDS"); do
        local t="${tag}_s${s}"
        if done_already "$t"; then echo "⏭  $t"; continue; fi
        echo ""
        echo "━━━ $t  (depth $depth) ━━━"
        local dir="${OUT_BASE}/${t}"
        [ "$FORCE" -eq 1 ] && rm -rf "$dir"
        if bash scripts/research_sweep.sh $COMMON --out-dir "$dir" \
               "$@" "$depth" 2>&1 | tee -a "$LOGFILE"; then
            mark_done "$t"; echo "✅  $t"
        else
            echo "❌  $t failed; will retry on the next invocation"
        fi
    done
}

has() { echo " $RUN_GROUPS " | grep -q " $1 "; }

# ---------------------------------------------------------------- control
# Run this before anything else. See the header.
if has control; then
    echo ""; echo "### CONTROL: is this just a head_dim effect?"
    run CTRL_dense_hd128 "$DEPTH" --models base
    run CTRL_dense_hd32  "$DEPTH" --models base --head-dim 32
fi

# ---------------------------------------------------------------- C1
if has c1; then
    echo ""; echo "### C1: main ablation (Table 7)"
    run C1_coupling_none  "$DEPTH" $MST_PLAIN $RECIPE --mst-transition-mode parallel
    run C1_coupling_mean  "$DEPTH" $MST_PLAIN $RECIPE --mst-mean-transition 1
    run C1_coupling_ours  "$DEPTH" $MST_FULL

    run C1_recipe_none    "$DEPTH" $MST_PLAIN
    run C1_recipe_gradeq  "$DEPTH" $MST_PLAIN --mst-grad-equalize 1
    run C1_recipe_muonlr  "$DEPTH" $MST_PLAIN --mst-block-diagonal-muon 1 --mst-sub-lr-scale 2.0
    run C1_recipe_widetr  "$DEPTH" $MST_PLAIN --mst-transition-width-mult ${N_SUBS}.0
    run C1_recipe_all     "$DEPTH" $MST_PLAIN $RECIPE
    run C1_full_msw       "$DEPTH" $MST_FULL

    for n in 2 4 8; do
        run "C1_N${n}" "$DEPTH" --use-mst 1 --models base \
            --mst-n-subs "$n" --mst-sub-dim $(( MODEL_DIM / n )) --mst-head-dim 0 \
            --mst-input-mode learned_proj --mst-routing-mode soft_weighted \
            --mst-routing-topk 0 --mst-ffn-mode standard \
            --mst-transition-mode aggregate_distribute \
            --mst-final-mode concat_proj --mst-final-topk 0 \
            --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0 \
            --mst-grad-equalize 1 --mst-block-diagonal-muon 1 \
            --mst-transition-width-mult "${n}.0" --mst-sub-lr-scale 2.0 \
            --mst-multi-scale-windows 1
    done
fi

# ---------------------------------------------------------------- C2
# Block-diagonal Muon and the per-sub LR are ablated jointly in C1 because
# blocking divides the effective rate by sqrt(N) and the multiplier restores it.
# These two runs separate them, to show the confound rather than assume it.
if has c2; then
    echo ""; echo "### C2: appendix extras (Table 9)"
    run C2_blockdiag_only "$DEPTH" $MST_PLAIN --mst-block-diagonal-muon 1
    run C2_sublr_only     "$DEPTH" $MST_PLAIN --mst-sub-lr-scale 2.0
    for w in 1.0 2.0 ${N_SUBS}.0; do
        run "C2_width_${w%%.*}x" "$DEPTH" $MST_PLAIN --mst-grad-equalize 1 \
            --mst-block-diagonal-muon 1 --mst-sub-lr-scale 2.0 \
            --mst-transition-width-mult "$w"
    done
fi

# ---------------------------------------------------------------- C3
# Twelve ways of enriching the coupling, all on top of the full recipe.
# If you already have these from the current environment, populate Table 8 from
# those logs instead and skip this group; it is ~4 GPU-hours.
if has c3; then
    echo ""; echo "### C3: coupling variants that did not help (Table 8)"
    run C3_nonlinear   "$DEPTH" $MST_FULL --mst-transition-nonlinear 1
    run C3_gated       "$DEPTH" $MST_FULL --mst-transition-gated 1
    run C3_mlp         "$DEPTH" $MST_FULL --mst-transition-mlp 1
    run C3_bilinear    "$DEPTH" $MST_FULL --mst-bilinear-transition 1
    run C3_slice       "$DEPTH" $MST_FULL --mst-slice-transition 4
    run C3_lookback    "$DEPTH" $MST_FULL --mst-lookback-layers 2
    run C3_hyper       "$DEPTH" $MST_FULL --mst-hyper-connect 1
    run C3_crossgate   "$DEPTH" $MST_FULL --mst-cross-sub-gate 32
    run C3_crosskv     "$DEPTH" $MST_FULL --mst-cross-kv-inject 1
    run C3_qmod        "$DEPTH" $MST_FULL --mst-cross-sub-qmod 16
    run C3_featcycle   "$DEPTH" $MST_FULL --mst-feature-cycle 1
    run C3_globalres   "$DEPTH" $MST_FULL --mst-global-residual 1
fi

# ---------------------------------------------------------------- C4
# The transfer check behind Table 6: the one intervention measurable at two
# depths. Single seed is enough, we only need the sign and rough magnitude.
if has c4; then
    echo ""; echo "### C4: transfer check at d12"
    D12=12
    MD12=$(( ((D12 * ASPECT_RATIO + 127) / 128) * 128 ))
    SD12=$(( MD12 / N_SUBS ))
    P12="--use-mst 1 --models base --mst-n-subs ${N_SUBS} --mst-sub-dim ${SD12} \
      --mst-head-dim 0 --mst-input-mode learned_proj \
      --mst-routing-mode soft_weighted --mst-routing-topk 0 --mst-ffn-mode standard \
      --mst-transition-mode aggregate_distribute \
      --mst-final-mode concat_proj --mst-final-topk 0 \
      --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0"
    SEEDS_SAVE="$SEEDS"; SEEDS=1
    run C4_d12_plain "$D12" $P12
    run C4_d12_full  "$D12" $P12 $RECIPE --mst-multi-scale-windows 1
    SEEDS="$SEEDS_SAVE"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  done. state: $STATE"
echo "  collect results with:  python -m scripts.paper_collect --dir $OUT_BASE"
echo "════════════════════════════════════════════════════════════"
