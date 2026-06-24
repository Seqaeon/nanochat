#!/usr/bin/env bash
# ============================================================================
# P07 MST Scaling Improvements Sweep
# ============================================================================
# Motivated by deep analysis of MST scaling bottlenecks at d32:
#   - 7.2× gradient imbalance across subs (sub 0 dominates)
#   - Muon Newton-Schulz cross-contamination in stacked (N*out, in) weights
#   - Aggregate-distribute bottleneck (75% info loss per layer)
#   - Diversity loss in deep layers (sub_sim rises from -0.31 to -0.21)
#
# Experiments (all on AggDist baseline with concat_proj final head):
#   P0: 1A — Per-sub gradient equalization
#   P0: 1B — Block-diagonal Muon Newton-Schulz
#   P1: 1C — Wider transition bottleneck (D-dim)
#   P1: LR — Per-sub LR scaling (√N boost)
#   P2: 2B — Longer warmup (3%)
#   P2: 2A — DeepSeek-style shared expert
#   P2: 2C — Router entropy regularization
#   P3: 3A — Shared K/V attention
#   P3: 3B — Contrastive diversity loss
#   COMBO: Best P0 + P1 combination
#
# Usage:
#   bash scripts/p07_mst_scaling_sweep.sh [--force] [DEPTH]
#   bash scripts/p07_mst_scaling_sweep.sh 8          # quick d8 iteration
#   bash scripts/p07_mst_scaling_sweep.sh 12 16      # multi-depth
#
# Default depth: 8
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
if [ ${#DEPTHS[@]} -eq 0 ]; then
    DEPTHS=(8)
fi

# ── Configuration ────────────────────────────────────────────────────────────
ASPECT_RATIO="${ASPECT_RATIO:-64}"
N_SUBS="${N_SUBS:-4}"

for DEPTH in "${DEPTHS[@]}"; do

# Compute model_dim and sub_dim from depth (same logic as p04/p05)
MODEL_DIM=$(( DEPTH * ASPECT_RATIO ))
# Round up to nearest 128 (head_dim alignment)
MODEL_DIM=$(( ((MODEL_DIM + 127) / 128) * 128 ))
SUB_DIM=$(( MODEL_DIM / N_SUBS ))

# ── Output directory ─────────────────────────────────────────────────────────
MST_OUT_BASE="${MST_OUT_BASE:-out/p07_mst_scaling}"
LOGFILE="${SWEEP_LOG:-${MST_OUT_BASE}/sweep_p07_d${DEPTH}.log}"
STATE_FILE="${MST_OUT_BASE}/sweep_state_d${DEPTH}.json"
mkdir -p "$MST_OUT_BASE"

echo "═══════════════════════════════════════════════════════════════"
echo "  P07 MST Scaling Improvements Sweep"
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

# ── Common flags (matching p05 AggDist baseline) ─────────────────────────────
MST_COMMON="--models base \
  --device-batch-size ${DEVICE_BATCH_SIZE:-32} --total-batch-size -1 --use-onecycle 0 --log-every 200 --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 \
  --target-param-data-ratio 10.5 \
  --warmup-ratio 0.005 \
  --warmdown-ratio 0.65 \
  --final-lr-frac 0.05 \
  --research-dim -1 \
  --target-tokens -1 \
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

# ── AggDist baseline flags (shared across all experiments) ───────────────────
AGGDIST_BASE="--mst-input-mode learned_proj \
  --mst-routing-mode soft_weighted --mst-routing-topk 0 --mst-ffn-mode standard \
  --mst-transition-mode aggregate_distribute \
  --mst-final-mode concat_proj --mst-final-topk 0 \
  --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0"

# ============================================================================
# P0: Critical fixes (high impact, low risk)
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  P0: Critical Fixes — Depth ${DEPTH}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# S7-0: AggDist baseline (control — same as p05 for comparison)
#run_experiment "S7_0_BASELINE_D${DEPTH}" \
#    "AggDist baseline (control for P07)" \
#    $AGGDIST_BASE

# S7-1A: Per-sub gradient equalization
#run_experiment "S7_1A_GRAD_EQ_D${DEPTH}" \
#    "1A: Per-sub gradient equalization (fixes 7.2× imbalance)" \
#    $AGGDIST_BASE \
#    --mst-grad-equalize 1

# S7-1B: Block-diagonal Muon Newton-Schulz
#run_experiment "S7_1B_BLOCK_DIAG_D${DEPTH}" \
#    "1B: Block-diagonal Muon (per-sub orthogonalization, fixes LR scale)" \
#    $AGGDIST_BASE \
#    --mst-block-diagonal-muon 1

# S7-P0: Combined 1A + 1B (expected best P0 result)
#run_experiment "S7_P0_GRADEQ_BLOCKDIAG_D${DEPTH}" \
#    "P0 combined: grad equalization + block-diagonal Muon" \
#    $AGGDIST_BASE \
#    --mst-grad-equalize 1 \
#    --mst-block-diagonal-muon 1

# ============================================================================
# P1: High priority
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  P1: High Priority — Depth ${DEPTH}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# S7-1C: Wider transition bottleneck (aggregate to D instead of d)
#run_experiment "S7_1C_WIDE_TRANS_D${DEPTH}" \
#    "1C: Wider transition (tw_mult=${N_SUBS}.0 → D-dim bottleneck)" \
#    $AGGDIST_BASE \
#    --mst-transition-width-mult ${N_SUBS}.0

# S7-LR: Per-sub LR scaling (√N boost)
# For N=4, √N = 2.0 — compensates for μP scaling mismatch
#run_experiment "S7_LR_SUB_SCALE_D${DEPTH}" \
#    "LR: Per-sub Muon LR × 2.0 (√N correction for μP mismatch)" \
#    $AGGDIST_BASE \
#    --mst-sub-lr-scale 2.0

# ============================================================================
# P2: Medium priority
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  P2: Medium Priority — Depth ${DEPTH}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# S7-2B: Longer warmup (3% instead of 0.5%)
#run_experiment "S7_2B_LONG_WARMUP_D${DEPTH}" \
#    "2B: Longer warmup (3% instead of 0.5%)" \
#    $AGGDIST_BASE \
#    --warmup-ratio 0.03

# S7-2A: DeepSeek-style shared expert (sub 0 always fully weighted)
run_experiment "S7_2A_SHARED_EXPERT_D${DEPTH}" \
    "2A: Shared expert (sub 0 always-on, subs 1-3 routed)" \
    $AGGDIST_BASE \
    --mst-shared-expert 1

# S7-2C: Router entropy regularization
run_experiment "S7_2C_ROUTE_ENTROPY_D${DEPTH}" \
    "2C: Router entropy regularization (weight=0.1)" \
    $AGGDIST_BASE \
    --mst-router-entropy-weight 0.1

# ============================================================================
# P3: Speculative
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  P3: Speculative — Depth ${DEPTH}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# S7-3A: Shared K/V attention (per-sub Q, shared K/V across subs)
run_experiment "S7_3A_SHARED_KV_D${DEPTH}" \
    "3A: Shared K/V attention (per-sub Q, shared K/V)" \
    $AGGDIST_BASE \
    --mst-shared-kv-attn 1

# S7-3B: Contrastive diversity loss on sub representations
run_experiment "S7_3B_CONTRASTIVE_D${DEPTH}" \
    "3B: Contrastive diversity loss (weight=0.01)" \
    $AGGDIST_BASE \
    --mst-contrastive-diversity-weight 0.01

# ============================================================================
# COMBO: Best interventions combined
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  COMBO: Combined Best — Depth ${DEPTH}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# S7-COMBO-A: P0 + wider transition + sub LR scaling
run_experiment "S7_COMBO_A_D${DEPTH}" \
    "COMBO-A: grad_eq + block_diag + wide_trans + sub_lr" \
    $AGGDIST_BASE \
    --mst-grad-equalize 1 \
    --mst-block-diagonal-muon 1 \
    --mst-transition-width-mult ${N_SUBS}.0 \
    --mst-sub-lr-scale 2.0

# S7-COMBO-B: P0 + longer warmup + entropy reg
run_experiment "S7_COMBO_B_D${DEPTH}" \
    "COMBO-B: grad_eq + block_diag + warmup 3% + entropy reg" \
    $AGGDIST_BASE \
    --mst-grad-equalize 1 \
    --mst-block-diagonal-muon 1 \
    --warmup-ratio 0.03 \
    --mst-router-entropy-weight 0.1

# S7-COMBO-FULL: All P0+P1+P2 interventions combined
run_experiment "S7_COMBO_FULL_D${DEPTH}" \
    "COMBO-FULL: all P0+P1+P2 combined" \
    $AGGDIST_BASE \
    --mst-grad-equalize 1 \
    --mst-block-diagonal-muon 1 \
    --mst-transition-width-mult ${N_SUBS}.0 \
    --mst-sub-lr-scale 2.0 \
    --warmup-ratio 0.03 \
    --mst-router-entropy-weight 0.1

echo ""
echo "  ✓ Depth ${DEPTH} P07 sweep complete"

echo "═══════════════════════════════════════════════════════════════"
echo "  P07 MST Scaling Improvements Sweep Complete"
echo "  Depth:    ${DEPTH}"
echo "  Experiments: 13 total"
echo "    P0: baseline, grad_eq, block_diag, combined"
echo "    P1: wide_trans, sub_lr_scale"
echo "    P2: long_warmup, shared_expert, router_entropy"
echo "    P3: shared_kv, contrastive_div"
echo "    COMBO: A (P0+P1), B (P0+P2), FULL (all)"
echo "═══════════════════════════════════════════════════════════════"

done
