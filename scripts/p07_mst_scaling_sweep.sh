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
  --device-batch-size ${DEVICE_BATCH_SIZE:-128} --total-batch-size -1 --use-onecycle 0 --log-every 200 --skip-core \
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
#run_experiment "S7_2A_SHARED_EXPERT_D${DEPTH}" \
#    "2A: Shared expert (sub 0 always-on, subs 1-3 routed)" \
#    $AGGDIST_BASE \
#    --mst-shared-expert 1
#
# S7-2C: Router entropy regularization
#run_experiment "S7_2C_ROUTE_ENTROPY_D${DEPTH}" \
#    "2C: Router entropy regularization (weight=0.1)" \
#    $AGGDIST_BASE \
#    --mst-router-entropy-weight 0.1

# ============================================================================
# P3: Speculative
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  P3: Speculative — Depth ${DEPTH}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#
# S7-3A: Shared K/V attention (per-sub Q, shared K/V across subs)
#run_experiment "S7_3A_SHARED_KV_D${DEPTH}" \
#    "3A: Shared K/V attention (per-sub Q, shared K/V)" \
#    $AGGDIST_BASE \
#    --mst-shared-kv-attn 1
#
# S7-3B: Contrastive diversity loss on sub representations
#run_experiment "S7_3B_CONTRASTIVE_D${DEPTH}" \
#    "3B: Contrastive diversity loss (weight=0.01)" \
#    $AGGDIST_BASE \
#    --mst-contrastive-diversity-weight 0.01

# ============================================================================
# COMBO: Best interventions combined
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  COMBO: Combined Best — Depth ${DEPTH}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# S7-COMBO-A: P0 + wider transition + sub LR scaling
#run_experiment "S7_COMBO_A_D${DEPTH}" \
#    "COMBO-A: grad_eq + block_diag + wide_trans + sub_lr" \
#    $AGGDIST_BASE \
#    --mst-grad-equalize 1 \
#    --mst-block-diagonal-muon 1 \
#    --mst-transition-width-mult ${N_SUBS}.0 \
#    --mst-sub-lr-scale 2.0
#
# S7-COMBO-B: P0 + longer warmup + entropy reg
#run_experiment "S7_COMBO_B_D${DEPTH}" \
#    "COMBO-B: grad_eq + block_diag + warmup 3% + entropy reg" \
#    $AGGDIST_BASE \
#    --mst-grad-equalize 1 \
#    --mst-block-diagonal-muon 1 \
#    --warmup-ratio 0.03 \
#    --mst-router-entropy-weight 0.1

# S7-COMBO-FULL: All P0+P1+P2 interventions combined
#run_experiment "S7_COMBO_FULL_D${DEPTH}" \
#    "COMBO-FULL: all P0+P1+P2 combined" \
#    $AGGDIST_BASE \
#    --mst-grad-equalize 1 \
#    --mst-block-diagonal-muon 1 \
#    --mst-transition-width-mult ${N_SUBS}.0 \
#    --mst-sub-lr-scale 2.0 \
#    --warmup-ratio 0.03 \
#    --mst-router-entropy-weight 0.1
#
echo ""
echo "  ✓ Depth ${DEPTH} P07 sweep complete"

# ============================================================================
# Stage 8: Transition expressivity (builds on COMBO_A baseline)
# ============================================================================
# COMBO_A (1.051 bpp) is the best so far. It already includes:
#   - grad_equalize + block_diagonal_muon (optimizer fixes)
#   - transition_width_mult=4.0 (D-width bottleneck with relu²)
#   - sub_lr_scale=2.0 (√N LR correction)
#
# Note: wide_trans already has relu² nonlinearity, so plain --mst-transition-nonlinear
# is redundant here. Stage 8 tests whether different transition ARCHITECTURES improve
# on the AggDist+wide_trans combo.

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Stage 8: Transition Expressivity — Depth ${DEPTH}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# COMBO_A flags as base (the current best)
COMBO_A_BASE="$AGGDIST_BASE \
  --mst-grad-equalize 1 \
  --mst-block-diagonal-muon 1 \
  --mst-transition-width-mult ${N_SUBS}.0 \
  --mst-sub-lr-scale 2.0 \
  --mst-multi-scale-windows 1"

# S8-1: Gated transition — input-dependent routing via concat→gate
# Replaces mean-based router with concat(all subs)→Linear(D,N).
# Each token's routing depends on the FULL cross-sub representation.
# Extra params: only ~12K (N × N*d = 4×512 per layer)
#run_experiment "S8_GATED_D${DEPTH}" \
#    "S8: Gated routing (concat→gate, input-dependent)" \
#    $COMBO_A_BASE \
#    --mst-transition-gated 1
#
# S8-2: MLP transition — concat→Linear→SiLU→Linear→split
# Replaces entire AggDist with a nonlinear MLP over all subs.
# Strictly more expressive than AggDist: can compose nonlinear functions of sub outputs.
# Extra params: ~3.7M (two D×D = 512×512 matrices per layer)
# Note: replaces wide_trans bottleneck — uses its own D→D→D path.
#run_experiment "S8_MLP_D${DEPTH}" \
#    "S8: MLP transition (concat→SiLU MLP→split)" \
#    $AGGDIST_BASE \
#    --mst-grad-equalize 1 \
#    --mst-block-diagonal-muon 1 \
#    --mst-sub-lr-scale 2.0 \
#    --mst-multi-scale-windows 1 \
#    --mst-transition-mlp 1

# S8-3: Gated + nonlinear — gated routing + SiLU at bottleneck
# Tests whether adding SiLU ON TOP of the wide_trans relu² helps.
# Two nonlinearities in the transition: relu² in the expanded path + SiLU after.
#run_experiment "S8_GATED_NL_D${DEPTH}" \
#    "S8: Gated + nonlinear (concat→gate + SiLU)" \
#    $COMBO_A_BASE \
#    --mst-transition-gated 1 \
#    --mst-transition-nonlinear 1
#
# S8-4: FFA hard routing — each sub routes to exactly 1 target per token
# Uses STE for gradient flow. Tests whether explicit sub→sub wiring helps vs AggDist.
#run_experiment "S8_FFA_HARD_D${DEPTH}" \
#    "S8: FFA hard routing (STE, topk=1)" \
#    --mst-input-mode learned_proj \
#    --mst-routing-mode soft_weighted --mst-routing-topk 1 --mst-ffn-mode standard \
#    --mst-transition-mode free_for_all \
#    --mst-final-mode concat_proj --mst-final-topk 0 \
#    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0 \
#    --mst-multi-scale-windows 1 \
#    --mst-grad-equalize 1 \
#    --mst-block-diagonal-muon 1 \
#    --mst-sub-lr-scale 2.0
#
# S8-5: FFA soft routing (control — compare soft vs hard FFA)
#run_experiment "S8_FFA_SOFT_D${DEPTH}" \
#    "S8: FFA soft routing (topk=0, control for S8-4)" \
#    --mst-input-mode learned_proj \
#    --mst-routing-mode soft_weighted --mst-routing-topk 0 --mst-ffn-mode standard \
#    --mst-transition-mode free_for_all \
#    --mst-final-mode concat_proj --mst-final-topk 0 \
#    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0 \
#    --mst-multi-scale-windows 1 \
#    --mst-grad-equalize 1 \
#    --mst-block-diagonal-muon 1 \
#    --mst-sub-lr-scale 2.0

echo ""
echo "  ✓ Depth ${DEPTH} Stage 8 sweep complete"

# ============================================================================
# Stage 9: Cross-sub expressivity (builds on COMBO_A baseline)
# ============================================================================
# Root cause: within each sub, features interact nonlinearly (attn + FFN).
# But ACROSS subs, features only interact linearly (transition).
# Stage 9 addresses this by enabling nonlinear cross-sub interaction at
# different points in the layer: FFN gating (A), residual lookback (B),
# and attention-level mixing (C).

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Stage 9: Cross-Sub Expressivity — Depth ${DEPTH}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# S9-A: Cross-sub FFN gating (rank=32)
# Gate each sub's FFN hidden state with a signal from ALL subs.
# Multiplicative interaction: cross-sub info flows THROUGH relu² nonlinearity.
# Cost: ~82K params/layer (rank-32 bottleneck: D→32→N*4d)
run_experiment "S9_CSGATE_R32_D${DEPTH}" \
    "S9-A: Cross-sub FFN gate rank=32" \
    $COMBO_A_BASE \
    --mst-cross-sub-gate 32

# S9-A2: Cross-sub FFN gating (rank=64) — more expressive gate
run_experiment "S9_CSGATE_R64_D${DEPTH}" \
    "S9-A: Cross-sub FFN gate rank=64" \
    $COMBO_A_BASE \
    --mst-cross-sub-gate 64

# S9-B: Hyper-connected sub residuals
# Each transition sees previous layer's pre-transition state via learned EMA.
# Near-zero extra params. Addresses depth-compounding bottleneck.
run_experiment "S9_HYPER_D${DEPTH}" \
    "S9-B: Hyper-connected sub residuals" \
    $COMBO_A_BASE \
    --mst-hyper-connect 1

# S9-C: Cross-sub KV injection (per-token N×N attention across subs)
# Softmax-based nonlinear cross-sub mixing at the attention level.
# Cost: ~41K params/layer (per-sub Q, shared K/V, per-sub proj)
run_experiment "S9_CROSSKV_D${DEPTH}" \
    "S9-C: Cross-sub KV injection attention" \
    $COMBO_A_BASE \
    --mst-cross-kv-inject 1

# S9-AB: Combo: cross-sub gate + hyper-connect
run_experiment "S9_GATE_HYPER_D${DEPTH}" \
    "S9-AB: Cross-sub gate (r=32) + hyper-connect" \
    $COMBO_A_BASE \
    --mst-cross-sub-gate 32 \
    --mst-hyper-connect 1

# S9-ABC: Full combo: gate + hyper + cross-KV
run_experiment "S9_FULL_D${DEPTH}" \
    "S9-ABC: All cross-sub expressivity" \
    $COMBO_A_BASE \
    --mst-cross-sub-gate 32 \
    --mst-hyper-connect 1 \
    --mst-cross-kv-inject 1

echo ""
echo "  ✓ Depth ${DEPTH} Stage 9 sweep complete"

# ============================================================================
# Stage 10: Structural transition improvements (builds on COMBO_A baseline)
# ============================================================================
# These address the STRUCTURE of the transition, not just adding nonlinearity.
# A: SliceMoE — fine-grained per-feature routing (different features routed differently)
# B: DenseFormer lookback — transition sees earlier layers lost through sequential bottleneck
# C: Bilinear — 2nd-order cross-sub interactions (products) impossible for linear transitions

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Stage 10: Structural Transition — Depth ${DEPTH}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# S10-A: SliceMoE with 4 slices (d=128 → 4 slices of 32)
run_experiment "S10_SLICE4_D${DEPTH}" \
    "S10-A: SliceMoE 4 slices" \
    $COMBO_A_BASE \
    --mst-slice-transition 4

# S10-A2: SliceMoE with 8 slices (d=128 → 8 slices of 16)
run_experiment "S10_SLICE8_D${DEPTH}" \
    "S10-A: SliceMoE 8 slices" \
    $COMBO_A_BASE \
    --mst-slice-transition 8

# S10-B: DenseFormer lookback K=2
run_experiment "S10_LOOKBACK2_D${DEPTH}" \
    "S10-B: DenseFormer lookback K=2" \
    $COMBO_A_BASE \
    --mst-lookback-layers 2

# S10-B2: DenseFormer lookback K=4
run_experiment "S10_LOOKBACK4_D${DEPTH}" \
    "S10-B: DenseFormer lookback K=4" \
    $COMBO_A_BASE \
    --mst-lookback-layers 4

# S10-C: Bilinear transition
run_experiment "S10_BILINEAR_D${DEPTH}" \
    "S10-C: Bilinear 2nd-order transition" \
    $COMBO_A_BASE \
    --mst-bilinear-transition 1

# S10-AC: SliceMoE + Bilinear (best structural combo?)
run_experiment "S10_SLICE_BILIN_D${DEPTH}" \
    "S10-AC: SliceMoE(4) + Bilinear" \
    $COMBO_A_BASE \
    --mst-slice-transition 4 \
    --mst-bilinear-transition 1

# S10-BC: Lookback + Bilinear
run_experiment "S10_LOOK_BILIN_D${DEPTH}" \
    "S10-BC: Lookback(2) + Bilinear" \
    $COMBO_A_BASE \
    --mst-lookback-layers 2 \
    --mst-bilinear-transition 1

echo ""
echo "  ✓ Depth ${DEPTH} Stage 10 sweep complete"

echo "═══════════════════════════════════════════════════════════════"
echo "  P07+S8+S9+S10 MST Scaling Sweep Complete"
echo "  Depth:    ${DEPTH}"
echo "  P07: 13 experiments (baseline through COMBO)"
echo "  S8:  5 experiments (gated, mlp, gated+nl, ffa_hard, ffa_soft)"
echo "  S9:  6 experiments (csgate_r32, csgate_r64, hyper, crosskv, gate+hyper, full)"
echo "  S10: 7 experiments (slice4, slice8, lookback2, lookback4, bilinear, slice+bilin, look+bilin)"
echo "═══════════════════════════════════════════════════════════════"

done
