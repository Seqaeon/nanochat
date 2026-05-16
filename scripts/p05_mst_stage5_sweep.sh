#!/usr/bin/env bash
# ============================================================================
# P05 MST Stage 5 Sweep — Specialization & Structural Pivots
# ============================================================================
# Motivated by Stage 4 diagnostics:
#   - FFA: subs collapse (cosine sim 0.94), routing learns fixed permutations
#   - AggDist: subs diverse (sim ~0.0) but routing near-uniform (no specialization)
#   - Neither achieves meaningful specialization
#
# Three experiments on top of AggDist baseline:
#   H3:  Per-sub auxiliary LM heads (direct specialization signal per sub)
#   T1:  Micro-attention transition (selective communication, not averaging)
#   N1:  Progressive sub-merging (pyramid: 8→4→2→1 subs)
#
# Usage:
#   bash scripts/p05_mst_stage5_sweep.sh [--force] [DEPTH]
#   bash scripts/p05_mst_stage5_sweep.sh 8
#   bash scripts/p05_mst_stage5_sweep.sh --force 12
#
# Default depth: 8
# ============================================================================

set -euo pipefail

# ── Parse arguments ──────────────────────────────────────────────────────────
FORCE=0
DEPTH=8
for arg in "$@"; do
    case $arg in
        --force) FORCE=1 ;;
        *)       DEPTH=$arg ;;
    esac
done

# ── Configuration ────────────────────────────────────────────────────────────
ASPECT_RATIO="${ASPECT_RATIO:-64}"
N_SUBS="${N_SUBS:-8}"

# Compute model_dim and sub_dim from depth (same logic as p04)
MODEL_DIM=$(( DEPTH * ASPECT_RATIO ))
# Round up to nearest 128 (head_dim alignment)
MODEL_DIM=$(( ((MODEL_DIM + 127) / 128) * 128 ))
SUB_DIM=$(( MODEL_DIM / N_SUBS ))

# ── Output directory ─────────────────────────────────────────────────────────
MST_OUT_BASE="${MST_OUT_BASE:-out/p05_mst_stage5}"
LOGFILE="${SWEEP_LOG:-${MST_OUT_BASE}/sweep_p05_d${DEPTH}.log}"
STATE_FILE="${MST_OUT_BASE}/sweep_state_d${DEPTH}.json"
mkdir -p "$MST_OUT_BASE"

echo "═══════════════════════════════════════════════════════════════"
echo "  P05 MST Stage 5: Specialization & Structural Pivots"
echo "  Depth:       ${DEPTH}"
echo "  Model dim:   ${MODEL_DIM} (${DEPTH} × ${ASPECT_RATIO})"
echo "  Sub dim:     ${SUB_DIM} (${MODEL_DIM} / ${N_SUBS})"
echo "  N subs:      ${N_SUBS}"
echo "  Output:      ${MST_OUT_BASE}"
echo "  State:       ${STATE_FILE}"
echo "  Log:         ${LOGFILE}"
echo "═══════════════════════════════════════════════════════════════"

# ── State management (JSON) ──────────────────────────────────────────────────
init_state() {
    if [ ! -f "$STATE_FILE" ]; then
        echo '{"completed":{},"started":{}}' > "$STATE_FILE"
    fi
}

check_completed() {
    local tag="$1"
    [ "$FORCE" -eq 1 ] && return 1
    python3 -c "
import json, sys
state = json.load(open('$STATE_FILE'))
sys.exit(0 if '$tag' in state.get('completed', {}) else 1)
" 2>/dev/null
}

mark_started() {
    local tag="$1"
    local run_dir="$2"
    python3 -c "
import json, datetime
state = json.load(open('$STATE_FILE'))
state.setdefault('started', {})['$tag'] = {
    'run_dir': '$run_dir',
    'started_at': datetime.datetime.now().isoformat()
}
json.dump(state, open('$STATE_FILE', 'w'), indent=2)
"
}

mark_completed() {
    local tag="$1"
    python3 -c "
import json, datetime
state = json.load(open('$STATE_FILE'))
state.setdefault('completed', {})['$tag'] = {
    'completed_at': datetime.datetime.now().isoformat()
}
state.get('started', {}).pop('$tag', None)
json.dump(state, open('$STATE_FILE', 'w'), indent=2)
"
}

print_header() {
    local id="$1" tag="$2" desc="$3"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$id] $tag"
    echo "  $desc"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

init_state

# ── Common flags ─────────────────────────────────────────────────────────────
MST_COMMON="--models base \
  --device-batch-size 128 --total-batch-size -1 --use-onecycle 0 --log-every 20 --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 \
  --target-param-data-ratio 10.5 \
  --warmup-ratio 0.005 \
  --warmdown-ratio 0.65 \
  --final-lr-frac 0.05 \
  --research-dim -1 \
  --target-tokens 0 \
  --target-active-params 0 \
  --save-every 200 \
  --eval-every -1 \
  --use-mst 1 \
  --mst-n-subs ${N_SUBS} --mst-head-dim 0 \
  --mst-sub-dim ${SUB_DIM}"

# Add optional env-based flags
[ -n "${MAX_SHARDS:-}" ]    && MST_COMMON="$MST_COMMON --max-shards $MAX_SHARDS"
[ "${USE_FP8:-0}" = "1" ]   && MST_COMMON="$MST_COMMON --fp8"

run_experiment() {
    local tag="$1"
    shift
    local desc="$1"
    shift

    if check_completed "$tag"; then
        echo "⏭  Skipping $tag (already completed)"
        return 0
    fi

    print_header "$tag" "$tag" "$desc"
    local run_dir="${MST_OUT_BASE}/${tag}"

    # --force: clean old run directory to prevent checkpoint resumption
    if [ "$FORCE" -eq 1 ] && [ -d "$run_dir" ]; then
        echo "🗑  --force: removing old run directory: $run_dir"
        rm -rf "$run_dir"
    fi

    mark_started "$tag" "$run_dir"

    if bash scripts/research_sweep.sh $MST_COMMON \
      --out-dir "$run_dir" \
      "$@" \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $tag done"
        mark_completed "$tag"
    else
        echo "❌  $tag FAILED — will retry next run"
    fi
}

# ============================================================================
# Stage 5 Experiments (all on AggDist baseline)
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Depth: ${DEPTH} (model_dim=${MODEL_DIM}, sub_dim=${SUB_DIM})"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# S5-0: AggDist baseline (control — same as P4B for comparison)
#run_experiment "S5_0_AGGDIST_BASE_D${DEPTH}" \
#    "AggDist baseline (control)" \
#    --mst-input-mode learned_proj \
#    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
#    --mst-transition-mode aggregate_distribute \
#    --mst-final-mode concat_proj --mst-final-topk 0 \
#    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0

# S5-H3: AggDist + Per-Sub Auxiliary LM Heads (specialization via per-sub prediction)
#run_experiment "S5_H3_SUB_AUX_D${DEPTH}" \
#    "AggDist + per-sub aux prediction heads (weight=0.3)" \
#    --mst-input-mode learned_proj \
#    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
#    --mst-transition-mode aggregate_distribute \
#    --mst-final-mode concat_proj --mst-final-topk 0 \
#    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0 \
#    --mst-sub-aux-weight 0.3

# S5-T1: AggDist-style routing + Micro-Attention Transition (selective, not averaging)
#run_experiment "S5_T1_MICRO_ATTN_D${DEPTH}" \
#    "Micro-attention transition (N-way self-attn over subs)" \
#    --mst-input-mode learned_proj \
#    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
#    --mst-transition-mode micro_attention \
#    --mst-final-mode concat_proj --mst-final-topk 0 \
#    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0


# S5-T2: AggDist-style routing + Micro-Attention Transition (selective, not averaging) 4 subs
run_experiment "S5_T2_MICRO_ATTN_D${DEPTH}" \
    "Micro-attention transition (N-way self-attn over 4 subs)" \
    --mst-input-mode learned_proj --mst-n-subs 4 --mst-sub-dim 128 \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-transition-mode micro_attention \
    --mst-final-mode concat_proj --mst-final-topk 0 \
    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0

# Variant A: FFA + concat_proj 4 subs
#run_experiment "P4A_FFA_D${DEPTH}" \
#    "FFA + concat_proj" \
#    --mst-input-mode learned_proj \
#    --mst-routing-mode soft_weighted --mst-routing-topk 0 --mst-ffn-mode standard \
#    --mst-transition-mode free_for_all --mst-n-subs 4  --mst-sub-dim 128\
#    --mst-final-mode concat_proj --mst-final-topk 0 \
#    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0

# S5-N1: Progressive Sub-Merging (Pyramid: 8→4→2→1)
# Needs smaller device-batch-size: merged layers have d=512 FFN (4× activation memory)
run_experiment "S5_N1_PYRAMID_D${DEPTH}" \
    "Progressive sub-merging pyramid (8→4→2→1)" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-transition-mode aggregate_distribute \
    --mst-final-mode concat_proj --mst-final-topk 0 \
    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0 \
    --mst-progressive-merge 1 \
    --device-batch-size 64

# S5-H3-T1: Combined — Micro-attention + per-sub aux loss
#run_experiment "S5_H3T1_COMBO_D${DEPTH}" \
#    "Micro-attention + per-sub aux (combined best candidates)" \
#    --mst-input-mode learned_proj \
#    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
#    --mst-transition-mode micro_attention \
#    --mst-final-mode concat_proj --mst-final-topk 0 \
#    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0 \
#    --mst-sub-aux-weight 0.3

# S5-T1b-4: Shared-KV Micro-Attention, N=4 (per-sub Q, shared K/V — query-based specialization)
run_experiment "S5_T1B_SHARED_KV_4SUB_D${DEPTH}" \
    "Shared-KV micro-attention N=4 (per-sub Q, shared K/V)" \
    --mst-input-mode learned_proj --mst-n-subs 4 --mst-sub-dim 128 \
    --mst-routing-mode soft_weighted --mst-routing-topk 0 --mst-ffn-mode standard \
    --mst-transition-mode micro_attention_shared_kv \
    --mst-final-mode concat_proj --mst-final-topk 0 \
    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0

# S5-T1b-8: Shared-KV Micro-Attention, N=8 (same but with original 8 subs)
run_experiment "S5_T1B_SHARED_KV_8SUB_D${DEPTH}" \
    "Shared-KV micro-attention N=8 (per-sub Q, shared K/V)" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 0 --mst-ffn-mode standard \
    --mst-transition-mode micro_attention_shared_kv \
    --mst-final-mode concat_proj --mst-final-topk 0 \
    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0

# S5-W1-4: Multi-scale windows, N=4 + micro-attention (each sub sees different context range)
run_experiment "S5_W1_MULTISCALE_4SUB_D${DEPTH}" \
    "Multi-scale windows N=4 + micro-attn (local→global per sub)" \
    --mst-input-mode learned_proj --mst-n-subs 4 --mst-sub-dim 128 \
    --mst-routing-mode soft_weighted --mst-routing-topk 0 --mst-ffn-mode standard \
    --mst-transition-mode micro_attention \
    --mst-final-mode concat_proj --mst-final-topk 0 \
    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0 \
    --mst-multi-scale-windows 1

# S5-W1-8: Multi-scale windows, N=8 + micro-attention
#run_experiment "S5_W1_MULTISCALE_8SUB_D${DEPTH}" \
#    "Multi-scale windows N=8 + micro-attn (local→global per sub)" \
#    --mst-input-mode learned_proj \
#    --mst-routing-mode soft_weighted --mst-routing-topk 0 --mst-ffn-mode standard \
#    --mst-transition-mode micro_attention \
#    --mst-final-mode concat_proj --mst-final-topk 0 \
#    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0 \
#    --mst-multi-scale-windows 1

# S5-W1-4-AGG: Multi-scale windows, N=4 + aggdist (compare transition methods)
run_experiment "S5_W1_MULTISCALE_4SUB_AGG_D${DEPTH}" \
    "Multi-scale windows N=4 + aggdist" \
    --mst-input-mode learned_proj --mst-n-subs 4 --mst-sub-dim 128 \
    --mst-routing-mode soft_weighted --mst-routing-topk 0 --mst-ffn-mode standard \
    --mst-transition-mode aggregate_distribute \
    --mst-final-mode concat_proj --mst-final-topk 0 \
    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0 \
    --mst-multi-scale-windows 1

echo ""
echo "  ✓ Depth ${DEPTH} complete"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  P05 Stage 5 Sweep Complete"
echo "  Depth:    ${DEPTH}"
echo "  Variants: 0=AggDist baseline, H3=per-sub aux, T1=micro-attn,"
echo "            N1=pyramid, H3T1=micro-attn+aux combo"
echo "═══════════════════════════════════════════════════════════════"
