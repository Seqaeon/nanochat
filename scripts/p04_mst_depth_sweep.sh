#!/usr/bin/env bash
# ============================================================================
# P04 MST Depth-Scaling Sweep
# ============================================================================
# Tests top 2 MST configs across depths to build scaling curves.
# model_dim = depth × aspect_ratio (default 64), sub_dim = model_dim / n_subs.
#
# Variant A: FFA + concat_proj (best FFA: 1.111 BPB at d8)
# Variant B: Aggdist + concat_proj + residual (best overall: 1.114 BPB at d8)
#
# Usage:
#   bash scripts/p04_mst_depth_sweep.sh [--force] DEPTH [DEPTH ...]
#   bash scripts/p04_mst_depth_sweep.sh 4 8 12 16
#   bash scripts/p04_mst_depth_sweep.sh --force 12
#
# Default depths: 4 8 12 16
# ============================================================================

set -euo pipefail

# ── Parse arguments ──────────────────────────────────────────────────────────
FORCE=0
DEPTHS=()
for arg in "$@"; do
    case $arg in
        --force) FORCE=1 ;;
        *)       DEPTHS+=("$arg") ;;
    esac
done
[ ${#DEPTHS[@]} -eq 0 ] && DEPTHS=(8 12 16)

# ── Configuration ────────────────────────────────────────────────────────────
ASPECT_RATIO="${ASPECT_RATIO:-64}"
N_SUBS="${N_SUBS:-8}"

# ── Output directory ─────────────────────────────────────────────────────────
MST_OUT_BASE="${MST_OUT_BASE:-out/p04_mst_depth}"
LOGFILE="${SWEEP_LOG:-${MST_OUT_BASE}/sweep_p04.log}"
STATE_FILE="${MST_OUT_BASE}/sweep_state_p04.json"
mkdir -p "$MST_OUT_BASE"

echo "═══════════════════════════════════════════════════════════════"
echo "  P04 MST Depth-Scaling Sweep"
echo "  Depths:       ${DEPTHS[*]}"
echo "  Aspect ratio: ${ASPECT_RATIO}"
echo "  N subs:       ${N_SUBS}"
echo "  Output:       ${MST_OUT_BASE}"
echo "  State:        ${STATE_FILE}"
echo "  Log:          ${LOGFILE}"
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

run_experiment() {
    local tag="$1"
    shift
    local desc="$1"
    shift
    local depth="$1"
    shift

    if check_completed "$tag"; then
        echo "⏭  Skipping $tag (already completed)"
        return 0
    fi

    # Compute model_dim and sub_dim from depth
    local model_dim=$(( depth * ASPECT_RATIO ))
    # Round up to nearest 128 (head_dim alignment)
    model_dim=$(( ((model_dim + 127) / 128) * 128 ))
    local sub_dim=$(( model_dim / N_SUBS ))

    print_header "$tag" "$tag" "$desc (model_dim=${model_dim}, sub_dim=${sub_dim})"
    local run_dir="${MST_OUT_BASE}/${tag}"
    mark_started "$tag" "$run_dir"

    # Common training flags
    local MST_COMMON="--models base \
      --device-batch-size 128 --total-batch-size -1 --use-onecycle 0 --log-every 20 --skip-core \
      --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
      --sequence-len 2048 \
      --target-param-data-ratio 10.5 \
      --warmup-ratio 0.005 \
      --warmdown-ratio 0.20 \
      --final-lr-frac 0.05 \
      --research-dim -1 \
      --target-tokens 0 \
      --target-active-params 0 \
      --save-every 200 \
      --eval-every -1 \
      --use-mst 1 \
      --mst-n-subs ${N_SUBS} --mst-head-dim 0 \
      --mst-sub-dim ${sub_dim}"

    # Add optional env-based flags
    [ -n "${MAX_SHARDS:-}" ]    && MST_COMMON="$MST_COMMON --max-shards $MAX_SHARDS"
    [ "${USE_FP8:-0}" = "1" ]   && MST_COMMON="$MST_COMMON --fp8"

    if bash scripts/research_sweep.sh $MST_COMMON \
      --out-dir "$run_dir" \
      "$@" \
      $depth 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $tag done"
        mark_completed "$tag"
    else
        echo "❌  $tag FAILED — will retry next run"
    fi
}

# ============================================================================
# Run both variants at each depth
# ============================================================================

for DEPTH in "${DEPTHS[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Depth: ${DEPTH} (model_dim=$(( ((DEPTH * ASPECT_RATIO + 127) / 128) * 128 )))"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"



    # Variant B: Aggdist + concat_proj + residual
    run_experiment "P4B_AGGDIST_D${DEPTH}" \
        "Aggdist + concat_proj + residual" \
        "$DEPTH" \
        --mst-input-mode learned_proj \
        --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
        --mst-transition-mode aggregate_distribute \
        --mst-final-mode concat_proj --mst-final-topk 0 \
        --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0

    # Variant A: FFA + concat_proj
    run_experiment "P4A_FFA_D${DEPTH}" \
        "FFA + concat_proj" \
        "$DEPTH" \
        --mst-input-mode learned_proj \
        --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
        --mst-transition-mode free_for_all \
        --mst-final-mode concat_proj --mst-final-topk 0 \
        --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0

    echo ""
    echo "  ✓ Depth ${DEPTH} complete"
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  P04 Depth-Scaling Sweep Complete"
echo "  Depths tested: ${DEPTHS[*]}"
echo "  Variants: A=FFA+concat_proj, B=Aggdist+concat_proj+residual"
echo "═══════════════════════════════════════════════════════════════"
