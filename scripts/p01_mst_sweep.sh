#!/usr/bin/env bash
# ============================================================================
# P01 MST Sweep — Modular Sub-Transformer Architecture Experiments
# ============================================================================
# Staged factorial design over 5 axes:
#   Axis 1 (Input):      fixed_slice | learned_proj | rotated_slice | per_sub_embed | stem
#   Axis 2 (Routing):    soft_weighted | topk_hard (k=1,4) | sequence_path
#   Axis 3 (FFN):        standard | no_downproj
#   Axis 4 (Transition): parallel | aggregate_distribute | cross_attend
#   Axis 5 (Final):      aggregate_proj | weighted_logits
#
# Usage:
#   bash scripts/p01_mst_sweep.sh [DEPTH]
#   bash scripts/p01_mst_sweep.sh --force [DEPTH]    # re-run completed experiments
#
# Default depth: 8 (D=512 with aspect_ratio=64, N=8, d=64)
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

# ── Output directory ─────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MST_OUT_BASE="${MST_OUT_BASE:-out/p01_mst_sweep}"
LOGFILE="${MST_OUT_BASE}/sweep_${DEPTH}_${TIMESTAMP}.log"
STATE_FILE="${MST_OUT_BASE}/sweep_state_d${DEPTH}.json"
mkdir -p "$MST_OUT_BASE"

echo "═══════════════════════════════════════════════════════════════"
echo "  P01 MST Sweep — Depth ${DEPTH}"
echo "  Output:  ${MST_OUT_BASE}"
echo "  State:   ${STATE_FILE}"
echo "  Log:     ${LOGFILE}"
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
  --device-batch-size 128 --total-batch-size -1 --use-onecycle 0 --log-every 200 --skip-core \
  --sequence-len 2048 \
  --target-param-data-ratio 10.5 \
  --warmup-ratio 0.005 \
  --warmdown-ratio 0.65 \
  --final-lr-frac 0.05 \
  --research-dim -1 \
  --target-tokens 0 \
  --target-active-params 0 \
  --save-every 200 \
  --eval-every 250 \
  --use-mst 1 \
  --mst-n-subs 8 \
  --mst-sub-dim 64"

# Add optional env-based flags
[ -n "${DATA_DIR:-}" ]      && MST_COMMON="$MST_COMMON --data-dir $DATA_DIR"
[ -n "${TOKENIZER_DIR:-}" ] && MST_COMMON="$MST_COMMON --tokenizer-dir $TOKENIZER_DIR"
[ -n "${MAX_SHARDS:-}" ]    && MST_COMMON="$MST_COMMON --max-shards $MAX_SHARDS"
[ "${USE_FP8:-0}" = "1" ]   && MST_COMMON="$MST_COMMON --fp8"

run_experiment() {
    local tag="$1"
    shift
    local desc="$1"
    shift
    # Remaining args are the MST-specific flags

    if check_completed "$tag"; then
        echo "⏭  Skipping $tag (already completed)"
        return 0
    fi

    print_header "$tag" "$tag" "$desc"
    local run_dir="${MST_OUT_BASE}/${tag}"
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
# Stage 0: Sanity Baseline
# ============================================================================
# S0A: MST baseline = fixed_slice + soft_weighted + standard + parallel + aggregate_proj

run_experiment "S0A_MST_BASELINE_D${DEPTH}" \
    "Stage 0: MST sanity baseline (I-A + R-A + F-A + T-B + V-B)" \
    --mst-input-mode fixed_slice \
    --mst-routing-mode soft_weighted \
    --mst-ffn-mode standard \
    --mst-transition-mode parallel \
    --mst-final-mode aggregate_proj

# ============================================================================
# Stage 1: Independent Axis Sweeps
# ============================================================================
# Each experiment varies ONE axis from the baseline, keeping all others at default.
# Baseline: fixed_slice | soft_weighted | standard | parallel | aggregate_proj

# ── Axis 1: Input Mode ──────────────────────────────────────────────────────

run_experiment "S1_INPUT_B_LEARNED_PROJ_D${DEPTH}" \
    "Stage 1: Input = learned_proj (rest baseline)" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-ffn-mode standard \
    --mst-transition-mode parallel --mst-final-mode aggregate_proj

run_experiment "S1_INPUT_C_ROTATED_SLICE_D${DEPTH}" \
    "Stage 1: Input = rotated_slice (frozen orthogonal)" \
    --mst-input-mode rotated_slice --mst-rotated-slice-learned 0 \
    --mst-routing-mode soft_weighted --mst-ffn-mode standard \
    --mst-transition-mode parallel --mst-final-mode aggregate_proj

run_experiment "S1_INPUT_C2_ROTATED_LEARNED_D${DEPTH}" \
    "Stage 1: Input = rotated_slice (learned rotation)" \
    --mst-input-mode rotated_slice --mst-rotated-slice-learned 1 \
    --mst-routing-mode soft_weighted --mst-ffn-mode standard \
    --mst-transition-mode parallel --mst-final-mode aggregate_proj

run_experiment "S1_INPUT_D_PER_SUB_EMBED_D${DEPTH}" \
    "Stage 1: Input = per_sub_embed (N separate embedding tables)" \
    --mst-input-mode per_sub_embed \
    --mst-routing-mode soft_weighted --mst-ffn-mode standard \
    --mst-transition-mode parallel --mst-final-mode aggregate_proj

run_experiment "S1_INPUT_E_STEM_D${DEPTH}" \
    "Stage 1: Input = stem (shared mini-transformer)" \
    --mst-input-mode stem \
    --mst-routing-mode soft_weighted --mst-ffn-mode standard \
    --mst-transition-mode parallel --mst-final-mode aggregate_proj

# ── Axis 2: Routing Mode ────────────────────────────────────────────────────

run_experiment "S1_ROUTE_B_TOPK4_D${DEPTH}" \
    "Stage 1: Routing = topk_hard (k=4)" \
    --mst-input-mode fixed_slice \
    --mst-routing-mode topk_hard --mst-routing-topk 4 \
    --mst-ffn-mode standard --mst-transition-mode parallel --mst-final-mode aggregate_proj

run_experiment "S1_ROUTE_B_TOPK1_D${DEPTH}" \
    "Stage 1: Routing = topk_hard (k=1, most sparse)" \
    --mst-input-mode fixed_slice \
    --mst-routing-mode topk_hard --mst-routing-topk 1 \
    --mst-ffn-mode standard --mst-transition-mode parallel --mst-final-mode aggregate_proj

run_experiment "S1_ROUTE_C_SEQUENCE_D${DEPTH}" \
    "Stage 1: Routing = sequence_path (one path per sequence)" \
    --mst-input-mode fixed_slice \
    --mst-routing-mode sequence_path \
    --mst-ffn-mode standard --mst-transition-mode parallel --mst-final-mode aggregate_proj

# ── Axis 3: FFN Mode ────────────────────────────────────────────────────────

run_experiment "S1_FFN_B_NO_DOWNPROJ_D${DEPTH}" \
    "Stage 1: FFN = no_downproj (d->4d, no compress back)" \
    --mst-input-mode fixed_slice --mst-routing-mode soft_weighted \
    --mst-ffn-mode no_downproj \
    --mst-transition-mode parallel --mst-final-mode aggregate_proj

# ── Axis 4: Transition Mode ─────────────────────────────────────────────────

run_experiment "S1_TRANS_A_AGGDIST_D${DEPTH}" \
    "Stage 1: Transition = aggregate_distribute" \
    --mst-input-mode fixed_slice --mst-routing-mode soft_weighted --mst-ffn-mode standard \
    --mst-transition-mode aggregate_distribute \
    --mst-final-mode aggregate_proj

run_experiment "S1_TRANS_C_CROSS_D${DEPTH}" \
    "Stage 1: Transition = cross_attend (lightweight cross-attention)" \
    --mst-input-mode fixed_slice --mst-routing-mode soft_weighted --mst-ffn-mode standard \
    --mst-transition-mode cross_attend \
    --mst-final-mode aggregate_proj

# ── Axis 5: Final Output Mode ───────────────────────────────────────────────

run_experiment "S1_FINAL_A_WEIGHTED_LOGITS_D${DEPTH}" \
    "Stage 1: Final = weighted_logits (N independent heads)" \
    --mst-input-mode fixed_slice --mst-routing-mode soft_weighted --mst-ffn-mode standard \
    --mst-transition-mode parallel \
    --mst-final-mode weighted_logits

# ============================================================================
# Stage 2: Interaction Grids (filled after Stage 1 analysis)
# ============================================================================
# Uncomment and fill these after analyzing Stage 1 results.
# Focus on the top-performing options from each axis.
#
# run_experiment "S2_BEST_INPUT_x_BEST_ROUTE_D${DEPTH}" \
#     "Stage 2: Best input × best routing" \
#     --mst-input-mode <BEST> --mst-routing-mode <BEST> ...

# ============================================================================
# Stage 3/4: Scaling & Analysis (deferred)
# ============================================================================
# Multi-seed validation, LogitLens analysis, routing visualization.

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  P01 MST Sweep Complete — Depth ${DEPTH}"
echo "═══════════════════════════════════════════════════════════════"
