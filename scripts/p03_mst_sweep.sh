#!/usr/bin/env bash
# ============================================================================
# P03 MST Sweep — Stage 3: Transition Residual + Normalization
# ============================================================================
# Based on P02 Stage 2 results. Key finding:
#   - FFA + concat_proj_final is the dominant combo (1.11 val_bpb)
#   - aggdist + concat_proj_final is 2nd best (1.158)
#   - Transition had NO residual connection or pre-norm — now fixed
#
# Stage 3 experiments:
#   Group A: Re-test S2 top configs with transition residual + pre-norm
#   Group B: Head dimension exploration (head_dim=64)
#
# Key code change: MSTLayer.forward now wraps non-parallel transitions as:
#   normed = norm(sub_output)
#   transitioned = transition(normed)
#   output = sub_output + transitioned   ← residual around transition
#
# Usage:
#   bash scripts/p03_mst_sweep.sh [DEPTH]
#   bash scripts/p03_mst_sweep.sh --force [DEPTH]
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

# ── Output directory ─────────────────────────────────────────────────────────
MST_OUT_BASE="${MST_OUT_BASE:-out/p03_mst_sweep}"
LOGFILE="${SWEEP_LOG:-${MST_OUT_BASE}/sweep_mst_p03_d${DEPTH}.log}"
STATE_FILE="${MST_OUT_BASE}/sweep_state_d${DEPTH}.json"
mkdir -p "$MST_OUT_BASE"

echo "═══════════════════════════════════════════════════════════════"
echo "  P03 MST Sweep — Stage 3: Transition Residual — Depth ${DEPTH}"
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
  --mst-n-subs 8 --mst-head-dim 0 \
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
# Group A: Re-test S2 winners with transition residual + pre-norm
# ============================================================================
# S2 reference results (no residual):
#   learned+FFA+concat_proj      → 1.110 val_bpb  (row 26/28)
#   fixed+FFA+concat_proj        → 1.148 val_bpb  (row 27)
#   learned+aggdist+concat_proj  → 1.158 val_bpb  (row 25)
#   learned+FFA+aggregate_proj   → 1.194 val_bpb  (row 21)

# A1: Best S2 config + transition residual (S2 baseline: 1.110)
run_experiment "S3A1_FFA_CONCAT_RESID_D${DEPTH}" \
    "Stage 3: learned_proj + FFA(soft) + concat_proj [+transition residual+norm]" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode concat_proj \
    --mst-final-topk 0 \
    --mst-routing-aux-weight 0.01 \
    --mst-diversity-weight 0.0

# A2: Aggdist + concat_proj + transition residual (S2 baseline: 1.158)
run_experiment "S3A2_AGGDIST_CONCAT_RESID_D${DEPTH}" \
    "Stage 3: learned_proj + aggdist + concat_proj [+transition residual+norm]" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-transition-mode aggregate_distribute \
    --mst-final-mode concat_proj \
    --mst-final-topk 0 \
    --mst-routing-aux-weight 0.01 \
    --mst-diversity-weight 0.0

# A3: Fixed slice + FFA + concat_proj + transition residual (S2 baseline: 1.148)
run_experiment "S3A3_FIXED_FFA_CONCAT_RESID_D${DEPTH}" \
    "Stage 3: fixed_slice + FFA(soft) + concat_proj [+transition residual+norm]" \
    --mst-input-mode fixed_slice \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode concat_proj \
    --mst-final-topk 0 \
    --mst-routing-aux-weight 0.01 \
    --mst-diversity-weight 0.0

# A4: FFA + aggregate_proj + transition residual (S2 baseline: 1.194)
#     Tests whether residual helps even without the concat_proj final head
run_experiment "S3A4_FFA_AGGPROJ_RESID_D${DEPTH}" \
    "Stage 3: learned_proj + FFA(soft) + aggregate_proj [+transition residual+norm]" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode aggregate_proj \
    --mst-final-topk -1 \
    --mst-routing-aux-weight 0.01 \
    --mst-diversity-weight 0.0

# ============================================================================
# Group B: Head Dimension (head_dim=64 → 4 heads × 64-dim = 256-dim attention)
# ============================================================================
# Adds ~3M params, stays under dense FLOPs budget (≤2.86e8)
# Tests whether richer per-sub attention (wider heads) improves quality

# B1: FFA + concat_proj + head_dim=64 [+transition residual]
run_experiment "S3B1_FFA_CONCAT_HD64_D${DEPTH}" \
    "Stage 3: learned_proj + FFA + concat_proj + head_dim=64 [+residual]" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode concat_proj \
    --mst-final-topk 0 \
    --mst-head-dim 64 \
    --mst-routing-aux-weight 0.01 \
    --mst-diversity-weight 0.0

# ============================================================================
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  P03 MST Stage 3 Sweep Complete — Depth ${DEPTH}"
echo "  Total experiments: 5 (4 Group A + 1 Group B)"
echo "  Key change vs S2: transition residual + pre-transition normalization"
echo "═══════════════════════════════════════════════════════════════"
