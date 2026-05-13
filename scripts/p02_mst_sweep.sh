#!/usr/bin/env bash
# ============================================================================
# P02 MST Sweep — Stage 2: Interaction Grid & New Modes
# ============================================================================
# Based on P01 Stage 1 results. Key findings:
#   - Transition mode is the dominant axis (aggregate_distribute: 1.1934 bpb)
#   - learned_proj is the best input mode (1.2043 bpb)
#   - Routing mode has minimal impact at this scale
#
# Stage 2 experiments:
#   Group A: Best transition × best input combinations
#   Group B: New concat_proj transition mode
#   Group C: concat_proj final head mode
#   Group D: Cosine diversity penalty (specialization pressure)
#
# Usage:
#   bash scripts/p02_mst_sweep.sh [DEPTH]
#   bash scripts/p02_mst_sweep.sh --force [DEPTH]
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
MST_OUT_BASE="${MST_OUT_BASE:-out/p02_mst_sweep}"
LOGFILE="${SWEEP_LOG:-${MST_OUT_BASE}/sweep_mst_p02_d${DEPTH}.log}"
STATE_FILE="${MST_OUT_BASE}/sweep_state_d${DEPTH}.json"
mkdir -p "$MST_OUT_BASE"

echo "═══════════════════════════════════════════════════════════════"
echo "  P02 MST Sweep — Stage 2 Interaction Grid — Depth ${DEPTH}"
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
# Group A: Best Transition × Best Input (4 experiments)
# ============================================================================
# S1 winners: aggregate_distribute (1.1934) and cross_attend (1.1967) for transition
#             learned_proj (1.2043) for input

# A1: Best input × Best transition — strongest combo candidate
run_experiment "S2A1_LEARNED_x_AGGDIST_D${DEPTH}" \
    "Stage 2: learned_proj + aggregate_distribute" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-ffn-mode standard \
    --mst-transition-mode aggregate_distribute \
    --mst-final-mode aggregate_proj

# A2: Best input × 2nd best transition
run_experiment "S2A2_LEARNED_x_CROSS_D${DEPTH}" \
    "Stage 2: learned_proj + cross_attend" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-ffn-mode standard \
    --mst-transition-mode cross_attend \
    --mst-final-mode aggregate_proj

# A3: Learned rotation (cheaper alternative) × best transition
run_experiment "S2A3_ROTLEARN_x_AGGDIST_D${DEPTH}" \
    "Stage 2: rotated_slice_learned + aggregate_distribute" \
    --mst-input-mode rotated_slice --mst-rotated-slice-learned 1 \
    --mst-routing-mode soft_weighted --mst-ffn-mode standard \
    --mst-transition-mode aggregate_distribute \
    --mst-final-mode aggregate_proj

# A4: Sparse routing + bottleneck transition
run_experiment "S2A4_FIXED_TOPK1_x_AGGDIST_D${DEPTH}" \
    "Stage 2: fixed_slice + topk_hard(k=1) + aggregate_distribute" \
    --mst-input-mode fixed_slice \
    --mst-routing-mode topk_hard --mst-routing-topk 1 \
    --mst-ffn-mode standard \
    --mst-transition-mode aggregate_distribute \
    --mst-final-mode aggregate_proj

# ============================================================================
# Group B: New concat_proj Transition Mode (2 experiments)
# ============================================================================
# concat_proj: concat all N sub outputs → Linear(N*d, d) → redistribute
# Preserves cross-sub information that weighted sum destroys.

# B1: concat_proj transition vs aggregate_distribute (direct comparison)
run_experiment "S2B1_FIXED_x_CONCAT_D${DEPTH}" \
    "Stage 2: fixed_slice + concat_proj transition (vs agg_dist)" \
    --mst-input-mode fixed_slice \
    --mst-routing-mode soft_weighted --mst-ffn-mode standard \
    --mst-transition-mode concat_proj \
    --mst-final-mode aggregate_proj

# B2: Best input × concat_proj
run_experiment "S2B2_LEARNED_x_CONCAT_D${DEPTH}" \
    "Stage 2: learned_proj + concat_proj transition" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-ffn-mode standard \
    --mst-transition-mode concat_proj \
    --mst-final-mode aggregate_proj

# ============================================================================
# Group E: free_for_all Transition Mode (2 experiments)
# ============================================================================
# free_for_all: each sub has an independent router that dynamically decides
# which target sub(s) to send its output to. Creates input-dependent wiring
# patterns — more expressive than cross_attend (which uses static weights).

# E1: free_for_all transition (direct comparison vs agg_dist and cross_attend)
run_experiment "S2E1_FIXED_x_FFA_D${DEPTH}" \
    "Stage 2: fixed_slice + free_for_all transition" \
    --mst-input-mode fixed_slice \
    --mst-routing-mode soft_weighted --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode aggregate_proj

# E2: Best input × free_for_all
run_experiment "S2E2_LEARNED_x_FFA_D${DEPTH}" \
    "Stage 2: learned_proj + free_for_all transition" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode aggregate_proj

# E3: Best input × free_for_all + concat_proj final head
run_experiment "S2E3_LEARNED_x_FFA__x_CONCAT_FINAL_D${DEPTH}" \
    "Stage 2: learned_proj + free_for_all transition + concat_proj final head" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode concat_proj

# E4: Fixed Slice × free_for_all + concat_proj final head
run_experiment "S2E4_FIXED_x_FFA__x_CONCAT_FINAL_D${DEPTH}" \
    "Stage 2: learned_proj + free_for_all transition + concat_proj final head" \
    --mst-input-mode fixed_slice \
    --mst-routing-mode soft_weighted --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode concat_proj

# E5: Best input × free_for_all + TOPK + concat_proj final head
run_experiment "S2E5_LEARNED_x_FFA_TOP1_x_CONCAT_FINAL_D${DEPTH}" \
    "Stage 2: learned_proj + free_for_all transition +TOP1 + concat_proj final head" \
    --mst-input-mode learned_proj \
    --mst-routing-mode topk_hard --mst-routing-topk 1 --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode concat_proj

# E6: FFA topk=1 transition + concat_proj final (ALL subs at final head)
run_experiment "S2E6_LEARNED_FFA_TOP1_CONCAT_ALL_D${DEPTH}" \
    "Stage 2: learned_proj + FFA topk=1 transition + concat_proj final (all subs)" \
    --mst-input-mode learned_proj \
    --mst-routing-mode topk_hard --mst-routing-topk 1 --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode concat_proj \
    --mst-final-topk 0

# E7: FFA topk=1 transition + concat_proj final (topk=1 at final head too)
run_experiment "S2E7_LEARNED_FFA_TOP1_CONCAT_TOP1_D${DEPTH}" \
    "Stage 2: learned_proj + FFA topk=1 transition + concat_proj final (topk=1)" \
    --mst-input-mode learned_proj \
    --mst-routing-mode topk_hard --mst-routing-topk 1 --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode concat_proj \
    --mst-final-topk 1
# ============================================================================
# Group C: concat_proj Final Head (2 experiments)
# ============================================================================
# concat_proj final: concat all N sub outputs → Linear(N*d, D) → lm_head
# More expressive than router-weighted sum for combining sub representations.

# C1: aggregate_distribute + concat_proj final head
run_experiment "S2C1_AGGDIST_x_CONCAT_FINAL_D${DEPTH}" \
    "Stage 2: aggregate_distribute transition + concat_proj final head" \
    --mst-input-mode fixed_slice \
    --mst-routing-mode soft_weighted --mst-ffn-mode standard \
    --mst-transition-mode aggregate_distribute \
    --mst-final-mode concat_proj

# C2: Best input × aggregate_distribute + concat_proj final head
run_experiment "S2C2_LEARNED_AGGDIST_x_CONCAT_FINAL_D${DEPTH}" \
    "Stage 2: learned_proj + aggregate_distribute + concat_proj final head" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-ffn-mode standard \
    --mst-transition-mode aggregate_distribute \
    --mst-final-mode concat_proj


# C3: Best input × aggregate_distribute + concat_proj final head
run_experiment "S2C3_LEARNED_AGGDIST_TOPK1_x_CONCAT_FINAL_D${DEPTH}" \
    "Stage 2: learned_proj + aggregate_distribute TOPK1_+ concat_proj final head" \
    --mst-input-mode learned_proj \
    --mst-routing-mode topk_hard --mst-routing-topk 1 --mst-ffn-mode standard \
    --mst-transition-mode aggregate_distribute \
    --mst-final-mode concat_proj

# ============================================================================
# Group D: Specialization Pressure (4 experiments)
# ============================================================================
# The load balance loss (aux_weight=0.01) actively fights specialization in
# soft_weighted mode because there's no expert starvation risk — all subs
# always process all tokens. The quadratic penalty pushes routing weights
# back to uniform 1/N, counteracting any natural specialization signal.
#
# We test: (1) diversity penalty, (2) reduced aux, (3) both, (4) zero aux.

# D1: Diversity penalty with default load balance (0.01)
run_experiment "S2D1_AGGDIST_DIVERSITY_D${DEPTH}" \
    "Stage 2: aggregate_distribute + diversity=0.01 + aux=0.01 (default)" \
    --mst-input-mode fixed_slice \
    --mst-routing-mode soft_weighted --mst-ffn-mode standard \
    --mst-transition-mode aggregate_distribute \
    --mst-final-mode aggregate_proj \
    --mst-diversity-weight 0.01

# D2: Reduced load balance (10x lower) — let router specialize naturally
run_experiment "S2D2_AGGDIST_LOW_AUX_D${DEPTH}" \
    "Stage 2: aggregate_distribute + aux=0.001 (reduced load balance)" \
    --mst-input-mode fixed_slice \
    --mst-routing-mode soft_weighted --mst-ffn-mode standard \
    --mst-transition-mode aggregate_distribute \
    --mst-final-mode aggregate_proj \
    --mst-routing-aux-weight 0.001

 D3: Diversity penalty + reduced load balance (best of both)
run_experiment "S2D3_AGGDIST_DIV_LOW_AUX_D${DEPTH}" \
    "Stage 2: aggregate_distribute + diversity=0.01 + aux=0.001" \
    --mst-input-mode fixed_slice \
    --mst-routing-mode soft_weighted --mst-ffn-mode standard \
    --mst-transition-mode aggregate_distribute \
    --mst-final-mode aggregate_proj \
    --mst-diversity-weight 0.01 \
    --mst-routing-aux-weight 0.001

# D4: No load balance at all — maximum specialization freedom
run_experiment "S2D4_AGGDIST_NO_AUX_D${DEPTH}" \
    "Stage 2: aggregate_distribute + aux=0.0 (no load balance)" \
    --mst-input-mode fixed_slice \
    --mst-routing-mode soft_weighted --mst-ffn-mode standard \
    --mst-transition-mode aggregate_distribute \
    --mst-final-mode aggregate_proj \
    --mst-routing-aux-weight 0.0

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  P02 MST Sweep Complete — Depth ${DEPTH}"
echo "  Total experiments: 21 (4 Group A + 2 Group B + 7 Group E + 2 Group C + 4 Group D + 2 Group F)"
echo "═══════════════════════════════════════════════════════════════"

