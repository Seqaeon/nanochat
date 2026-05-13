#!/usr/bin/env bash
# ============================================================================
# P03 MST Stage 3 Sweep: Transition Residual + Normalization
# ============================================================================
#
# Goal: Test the impact of transition residual + pre-transition normalization
# on the winning FFA+concat_proj architecture.
#
# FLOPs budget: ≤ 2.862643e8 (dense d8 baseline)
# Dense baseline: 125M params, 0.969 val_bpb
#
# The transition residual (x = x + transition(norm(x))) is now baked into
# the code for ALL non-parallel transition modes. No new CLI flags needed.
#
# Usage:
#   bash scripts/p03_mst_sweep.sh 8
# ============================================================================

set -euo pipefail

DEPTH=${1:-8}

run_experiment() {
    local NAME="$1"
    local DESC="$2"
    shift 2

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Starting: $NAME"
    echo "  $DESC"
    echo "═══════════════════════════════════════════════════════════════"

    python -m scripts.research_compare \
        --output-dir "out/p03_mst_sweep/$NAME" \
        --depth "$DEPTH" \
        --aspect-ratio 64 \
        --head-dim 128 \
        --model-dim 512 \
        --max-seq-len 2048 \
        --total-batch-size 262144 \
        --target-param-data-ratio 10.5 \
        --eval-every -1 \
        --log-every 200 \
        --core-metric-every 0 \
        --save-every 200 \
        --warmup-ratio 0.005 \
        --warmdown-ratio 0.65 \
        --final-lr-frac 0.05 \
        --adam-beta2 0.99 \
        --research-warmup-ratio 0.05 \
        --compile \
        --use-mst 1 \
        --mst-n-subs 8 \
        --mst-sub-dim 64 \
        --mst-head-dim 0 \
        --mst-routing-aux-weight 0.01 \
        --mst-diversity-weight 0.0 \
        "$@"
}

# ============================================================================
# Group A: Transition Residual Impact (re-test S2 winners with residual+norm)
# ============================================================================
# These directly compare to S2 rows 26, 25, 27 to isolate the effect.

# A1: Best S2 config + transition residual (was 1.11 val_bpb)
run_experiment "S3A1_FFA_CONCAT_RESID_D${DEPTH}" \
    "Stage 3: learned_proj + FFA(soft) + concat_proj [+transition residual]" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 0 --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode concat_proj \
    --mst-final-topk -1

# A2: Aggdist + concat_proj + transition residual (was 1.158 val_bpb)
run_experiment "S3A2_AGGDIST_CONCAT_RESID_D${DEPTH}" \
    "Stage 3: learned_proj + aggregate_distribute + concat_proj [+transition residual]" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 0 --mst-ffn-mode standard \
    --mst-transition-mode aggregate_distribute \
    --mst-final-mode concat_proj \
    --mst-final-topk -1

# A3: Fixed slice + FFA + concat_proj + transition residual (was 1.148 val_bpb)
run_experiment "S3A3_FIXED_FFA_CONCAT_RESID_D${DEPTH}" \
    "Stage 3: fixed_slice + FFA(soft) + concat_proj [+transition residual]" \
    --mst-input-mode fixed_slice \
    --mst-routing-mode soft_weighted --mst-routing-topk 0 --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode concat_proj \
    --mst-final-topk -1

# A4: FFA + aggregate_proj + transition residual (was 1.194 — does residual help here too?)
run_experiment "S3A4_FFA_AGGPROJ_RESID_D${DEPTH}" \
    "Stage 3: learned_proj + FFA(soft) + aggregate_proj [+transition residual]" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 0 --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode aggregate_proj \
    --mst-final-topk -1

# ============================================================================
# Group B: Head Dimension (test head_dim=d=64 with 1 effective head)
# ============================================================================
# head_dim=64 with n_head=4 → 256-dim attention per sub
# Adds ~3M params but stays under FLOPs budget (~2.74e8 → ~2.82e8 estimated)

# B1: FFA + concat_proj + head_dim=64
run_experiment "S3B1_FFA_CONCAT_HD64_D${DEPTH}" \
    "Stage 3: learned_proj + FFA + concat_proj + head_dim=64 [+residual]" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 0 --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode concat_proj \
    --mst-final-topk -1 \
    --mst-head-dim 64

# ============================================================================
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  P03 MST Stage 3 Sweep Complete — Depth ${DEPTH}"
echo "  Total experiments: 5"
echo "  Key change: transition residual + pre-transition normalization"
echo "═══════════════════════════════════════════════════════════════"
