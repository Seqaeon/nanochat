#!/bin/bash
# Phase 20: Context-Conditioned Dynamic Weight Computation — Full Sweep (A–J)
# Each proposal runs independently against the dense baseline (model="base")
# "base" = standard dense transformer with Block + MLP
#
# Features:
#   - Resume: skips proposals that already have results in sweep.log
#   - Demarcation: clear separators between each proposal run
#
# Usage:
#   bash scripts/p20_sweeps.sh           # run all (skipping completed)
#   bash scripts/p20_sweeps.sh --force   # re-run everything

set -e

LOGFILE="${SWEEP_LOG:-sweep.log}"
FORCE=0
if [[ "$1" == "--force" ]]; then
    FORCE=1
    shift
fi

COMMON="--fp8 --max-shards 170 --models base \
  --device-batch-size 64 --use-onecycle 0 --log-every 200 --skip-core \
  --data-dir /root/nanochat/data --tokenizer-dir /root/nanochat/tokenizer \
  --sequence-len 2048 --mu-p-mode base_only --model-dim 128 \
  --modulation-diagnostics 1"
DEPTH=4

# ─────────────────────────────────────────
# Resume helper: check if a proposal already has results
# ─────────────────────────────────────────
check_completed() {
    local tag="$1"
    if [[ "$FORCE" -eq 1 ]]; then
        return 1  # force re-run
    fi
    if [[ ! -f "$LOGFILE" ]]; then
        return 1  # no log file yet
    fi
    # Check if both the start marker AND a "Minimum validation bpb" exist
    # after it (before the next start marker)
    if grep -q "▶▶▶ $tag ◀◀◀" "$LOGFILE" && \
       awk "/▶▶▶ $tag ◀◀◀/,/▶▶▶/{if(/Minimum validation bpb/) found=1} END{exit !found}" "$LOGFILE"; then
        return 0  # completed
    fi
    return 1  # not completed
}

# Demarcation helper
print_header() {
    local num="$1"
    local tag="$2"
    local desc="$3"
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  [$num/10]  $tag"
    echo "║  $desc"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo "▶▶▶ $tag ◀◀◀"  # machine-readable marker for resume detection
    echo ""
}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Phase 20 Full Sweep (A–J)                         ║"
echo "║  Baseline: base model, model_dim=128, depth=$DEPTH             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────
# PHASE 1 PROPOSALS (train from scratch)
# ─────────────────────────────────────────

# 20A: Hash-Routed Column Selection (×4 scale)
TAG="20A_HRCS"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "1" "$TAG" "Hash-Routed Column Selection (scale=4)"
    bash scripts/research_sweep.sh $COMMON \
      --p20-hrcs-scale 4 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
fi

# 20B: LSH Weight Routing (×4 scale, 8 planes)
TAG="20B_LSWR"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "2" "$TAG" "LSH Weight Routing (scale=4, planes=8)"
    bash scripts/research_sweep.sh $COMMON \
      --p20-lswr-scale 4 --p20-lswr-planes 8 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
fi

# 20C: Frozen Content-Routed Branches (K=4)
TAG="20C_LRCFB"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "3" "$TAG" "Frozen Content-Routed Branches (K=4)"
    bash scripts/research_sweep.sh $COMMON \
      --p20-lrcfb-branches 4 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
fi

# 20D: Detached-Gradient Content Routing (K=4)
TAG="20D_DGCR"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "4" "$TAG" "Detached-Gradient Content Routing (K=4)"
    bash scripts/research_sweep.sh $COMMON \
      --p20-dgcr-branches 4 --p20-dgcr-aux-weight 0.01 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
fi

# 20F: Mixture of Narrow Experts (K=4, same total params)
TAG="20F_MoNE"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "5" "$TAG" "Mixture of Narrow Experts (K=4)"
    bash scripts/research_sweep.sh $COMMON \
      --p20-mone-experts 4 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
fi

# 20H: Noise-Contrastive Expert Assignment (K=4)
TAG="20H_NCEA"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "6" "$TAG" "Noise-Contrastive Expert Assignment (K=4, eps=0.1)"
    bash scripts/research_sweep.sh $COMMON \
      --p20-ncea-branches 4 --p20-ncea-eps 0.1 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
fi

# 20I: Attention-Derived Weight Interpolation
TAG="20I_ADWI"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "7" "$TAG" "Attention-Derived Weight Interpolation"
    bash scripts/research_sweep.sh $COMMON \
      --p20-adwi 1 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
fi

# ─────────────────────────────────────────
# PHASE 2 PROPOSALS (pretrain then convert)
# ─────────────────────────────────────────

# 20E: Progressive Weight Unfreezing (K=4 branches, Phase 2: router only)
TAG="20E_PWU"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "8" "$TAG" "Progressive Weight Unfreezing (K=4, phase=2)"
    bash scripts/research_sweep.sh $COMMON \
      --p20-pwu-branches 4 --p20-pwu-phase 2 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
fi

# 20G: Frozen-SVD σ Gating
TAG="20G_FSVD"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "9" "$TAG" "Frozen-SVD Sigma Gating"
    bash scripts/research_sweep.sh $COMMON \
      --p20-fsvd-gate 1 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
fi

# 20J: Weight Bank Frozen Clustering (K=8 clusters, M=2 active)
TAG="20J_WBFC"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "10" "$TAG" "Weight Bank Frozen Clustering (K=8, M=2)"
    bash scripts/research_sweep.sh $COMMON \
      --p20-wbfc-clusters 8 --p20-wbfc-active 2 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Phase 20 Sweep Complete (All 10)                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Check $LOGFILE for results."
