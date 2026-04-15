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
STATEFILE="${LOGFILE%.log}.state"
FORCE=0
if [[ "$1" == "--force" ]]; then
    FORCE=1
    rm -f "$STATEFILE"  # clear state on force 
    shift
fi

COMMON="--fp8 --max-shards 170 --models base \
  --device-batch-size 64 --use-onecycle 0 --log-every 200 --skip-core \
  --data-dir /root/nanochat/data --tokenizer-dir /root/nanochat/tokenizer \
  --sequence-len 2048 --mu-p-mode base_only \
  --modulation-diagnostics 1"
DEPTH=4

# ─────────────────────────────────────────
# Resume helper: uses a .state file (not the log, which gets truncated by tee)
# ─────────────────────────────────────────
check_completed() {
    local tag="$1"
    if [[ "$FORCE" -eq 1 ]]; then
        return 1  # force re-run
    fi
    if [[ ! -f "$STATEFILE" ]]; then
        return 1  # no state file yet
    fi
    if grep -qx "$tag" "$STATEFILE" 2>/dev/null; then
        return 0  # completed
    fi
    return 1  # not completed
}

mark_completed() {
    local tag="$1"
    echo "$tag" >> "$STATEFILE"
}

# Demarcation helper
print_header() {
    local num="$1"
    local tag="$2"
    local desc="$3"
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  [$num]  $tag"
    echo "║  $desc"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Phase 20+21+22 Full Sweep                         ║"
echo "║  Baseline: normal base model calculations for depth=$DEPTH ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────
# BASELINE
# ─────────────────────────────────────────
TAG="00_BASELINE"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "0" "$TAG" "Normal Base Model Baseline"
    bash scripts/research_sweep.sh $COMMON $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# ─────────────────────────────────────────
# PHASE 1 PROPOSALS (train from scratch)
# ─────────────────────────────────────────

# 20A: Hash-Routed Column Selection (×4 scale)
#TAG="20A_HRCS"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "1" "$TAG" "Hash-Routed Column Selection (scale=4)"
#    bash scripts/research_sweep.sh $COMMON \
#      --p20-hrcs-scale 4 \
#      $DEPTH
#    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
#fi
#
# 20B: LSH Weight Routing (×4 scale, 8 planes)
#TAG="20B_LSWR"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "2" "$TAG" "LSH Weight Routing (scale=4, planes=8)"
#    bash scripts/research_sweep.sh $COMMON \
#      --p20-lswr-scale 4 --p20-lswr-planes 8 \
#      $DEPTH
#    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
#fi
#
# 20C: Frozen Content-Routed Branches (K=4, full-size — already proven best)
#TAG="20C_LRCFB"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "3" "$TAG" "Frozen Content-Routed Branches (K=4, full-size)"
#    bash scripts/research_sweep.sh $COMMON \
#      --p20-lrcfb-branches 8 \
#      $DEPTH
#    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
#fi

# 20C-narrow: Frozen routing, narrow branches (param parity!)
#TAG="20C_NARROW"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "3b" "$TAG" "Frozen Content-Routed, Narrow Branches (K=4, param parity)"
#    bash scripts/research_sweep.sh $COMMON \
#      --p20-lrcfb-branches 8 --p20-lrcfb-narrow 1 \
#      $DEPTH
#    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
#fi
#
# 20C-learned: Learned routing, narrow branches
#TAG="20C_LEARNED"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "3c" "$TAG" "Learned Content-Routed, Narrow Branches (K=4)"
#    bash scripts/research_sweep.sh $COMMON \
#      --p20-lrcfb-branches 8 --p20-lrcfb-narrow 1 --p20-lrcfb-learned 1 \
#      $DEPTH
#    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
#fi
#
# 20C-topk1: Frozen routing, narrow branches, top-1 sparse (minimum FLOPs!)
#TAG="20C_TOPK1"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "3d" "$TAG" "Frozen Narrow + Top-1 Sparse (K=4, only 1 active)"
#    bash scripts/research_sweep.sh $COMMON \
#      --p20-lrcfb-branches 8 --p20-lrcfb-narrow 1 --p20-lrcfb-topk 1 \
#      $DEPTH
#    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
#fi
#
# 20D: Detached-Gradient Content Routing (K=4)
#TAG="20D_DGCR"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "4" "$TAG" "Detached-Gradient Content Routing (K=4)"
#    bash scripts/research_sweep.sh $COMMON \
#      --p20-dgcr-branches 4 --p20-dgcr-aux-weight 0.01 \
#      $DEPTH
#    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
#fi

# 20F: Mixture of Narrow Experts (K=4, same total params)
#TAG="20F_MoNE"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "5" "$TAG" "Mixture of Narrow Experts (K=4)"
#    bash scripts/research_sweep.sh $COMMON \
#      --p20-mone-experts 8 \
#      $DEPTH
#    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
#fi
#
# 20H: Noise-Contrastive Expert Assignment (K=4)
#TAG="20H_NCEA"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "6" "$TAG" "Noise-Contrastive Expert Assignment (K=4, eps=0.1)"
#    bash scripts/research_sweep.sh $COMMON \
#      --p20-ncea-branches 4 --p20-ncea-eps 0.1 \
#      $DEPTH
#    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
#fi
#
# 20I: Attention-Derived Weight Interpolation
#TAG="20I_ADWI"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "7" "$TAG" "Attention-Derived Weight Interpolation"
#    bash scripts/research_sweep.sh $COMMON \
#      --p20-adwi 1 \
#      $DEPTH
#    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
#fi
#
# ─────────────────────────────────────────
# PHASE 2 PROPOSALS (pretrain then convert)
# ─────────────────────────────────────────
#
# 20E: Progressive Weight Unfreezing (K=4 branches, Phase 2: router only)
#TAG="20E_PWU"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "8" "$TAG" "Progressive Weight Unfreezing (K=4, phase=2)"
#    bash scripts/research_sweep.sh $COMMON \
#      --p20-pwu-branches 4 --p20-pwu-phase 2 \
#      $DEPTH
#    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
#fi
#
# 20G: Frozen-SVD σ Gating
#TAG="20G_FSVD"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "9" "$TAG" "Frozen-SVD Sigma Gating"
#    bash scripts/research_sweep.sh $COMMON \
#      --p20-fsvd-gate 1 \
#      $DEPTH
#    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
#fi
#
# 20J: Weight Bank Frozen Clustering (K=8 clusters, M=2 active)
#TAG="20J_WBFC"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "10" "$TAG" "Weight Bank Frozen Clustering (K=8, M=2)"
#    bash scripts/research_sweep.sh $COMMON \
#      --p20-wbfc-clusters 8 --p20-wbfc-active 2 \
#      $DEPTH
#    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
#fi

# ─────────────────────────────────────────
# PHASE 21: Pervasive Expert Routing (PER)
# MoELinear replaces standard Linear layers
# ─────────────────────────────────────────

# 21-MLP-soft: COMMENTED OUT — equivalent to MoNE (20F) which already tests MLP-only MoE
#TAG="21_PER_MLP_SOFT"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "11" "$TAG" "PER: MLP-only, K=4 experts, soft routing"
#    bash scripts/research_sweep.sh $COMMON \
#      --p21-per-experts 4 \
#      $DEPTH
#    echo "════════════════ $TAG COMPLETE ════════════════"
#    mark_completed "$TAG"
#fi

# 21-MLP-top1: COMMENTED OUT — equivalent to MoNE (20F) with top-1
#TAG="21_PER_MLP_TOP1"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "12" "$TAG" "PER: MLP-only, K=4 experts, top-1 routing"
#    bash scripts/research_sweep.sh $COMMON \
#      --p21-per-experts 4 --p21-per-topk 1 \
#      $DEPTH
#    echo "════════════════ $TAG COMPLETE ════════════════"
#    mark_completed "$TAG"
#fi

# 21-ALL-soft: MoELinear everywhere (MLP + attention), soft routing
TAG="21_PER_ALL_SOFT"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "13" "$TAG" "PER: MLP+Attention, K=4 experts, soft routing"
    bash scripts/research_sweep.sh $COMMON \
      --p21-per-experts 4 --p21-per-attn 1 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# 21-ALL-top1: MoELinear everywhere, top-1 routing (maximum FLOP savings)
TAG="21_PER_ALL_TOP1"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "14" "$TAG" "PER: MLP+Attention, K=4 experts, top-1 routing"
    bash scripts/research_sweep.sh $COMMON \
      --p21-per-experts 4 --p21-per-topk 1 --p21-per-attn 1 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# ─────────────────────────────────────────
# PHASE 22: True MoE (Full-Size Experts)
# Each expert is full-width (4×D hidden), K× total params
# ─────────────────────────────────────────

# 22A: LRCFB Full — soft routing, frozen
TAG="22_LRCFB_FULL_SOFT_FROZEN"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "15" "$TAG" "LRCFB Full: K=8 full experts, soft, frozen routing"
    bash scripts/research_sweep.sh $COMMON \
      --p20-lrcfb-branches 8 --p20-lrcfb-narrow 0 --p20-lrcfb-learned 0 --p20-lrcfb-topk 0 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# 22B: LRCFB Full — soft routing, learned
TAG="22_LRCFB_FULL_SOFT_LEARNED"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "16" "$TAG" "LRCFB Full: K=8 full experts, soft, learned routing"
    bash scripts/research_sweep.sh $COMMON \
      --p20-lrcfb-branches 8 --p20-lrcfb-narrow 0 --p20-lrcfb-learned 1 --p20-lrcfb-topk 0 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# 22C: LRCFB Full — hard top-1, frozen
TAG="22_LRCFB_FULL_HARD_FROZEN"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "17" "$TAG" "LRCFB Full: K=8 full experts, hard top-1, frozen routing"
    bash scripts/research_sweep.sh $COMMON \
      --p20-lrcfb-branches 8 --p20-lrcfb-narrow 0 --p20-lrcfb-learned 0 --p20-lrcfb-topk 1 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# 22D: LRCFB Full — hard top-1, learned
TAG="22_LRCFB_FULL_HARD_LEARNED"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "18" "$TAG" "LRCFB Full: K=8 full experts, hard top-1, learned routing"
    bash scripts/research_sweep.sh $COMMON \
      --p20-lrcfb-branches 8 --p20-lrcfb-narrow 0 --p20-lrcfb-learned 1 --p20-lrcfb-topk 1 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# 22E: MoNE Full — soft routing, learned
TAG="22_MONE_FULL_SOFT_LEARNED"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "19" "$TAG" "MoNE Full: K=8 full experts, soft, learned router"
    bash scripts/research_sweep.sh $COMMON \
      --p20-mone-experts 8 --p20-mone-narrow 0 --p20-mone-frozen 0 --p20-mone-topk 0 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# 22F: MoNE Full — soft routing, frozen
TAG="22_MONE_FULL_SOFT_FROZEN"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "20" "$TAG" "MoNE Full: K=8 full experts, soft, frozen routing"
    bash scripts/research_sweep.sh $COMMON \
      --p20-mone-experts 8 --p20-mone-narrow 0 --p20-mone-frozen 1 --p20-mone-topk 0 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# 22G: MoNE Full — hard top-1, learned
TAG="22_MONE_FULL_HARD_LEARNED"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "21" "$TAG" "MoNE Full: K=8 full experts, hard top-1, learned router"
    bash scripts/research_sweep.sh $COMMON \
      --p20-mone-experts 8 --p20-mone-narrow 0 --p20-mone-frozen 0 --p20-mone-topk 1 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# 22H: MoNE Full — hard top-1, frozen
TAG="22_MONE_FULL_HARD_FROZEN"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "22" "$TAG" "MoNE Full: K=8 full experts, hard top-1, frozen routing"
    bash scripts/research_sweep.sh $COMMON \
      --p20-mone-experts 8 --p20-mone-narrow 0 --p20-mone-frozen 1 --p20-mone-topk 1 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# ─────────────────────────────────────────
# PHASE 22: RemixedLinear MoE Template Routing
# K template_mixing matrices with content routing
# Uses --models remixed-linear (CCLBlock path)
# Common: norm_x context, selective stream, boundary aux
# ─────────────────────────────────────────

REMIX_COMMON="--fp8 --max-shards 170 --models remixed-linear \
  --device-batch-size 64 --use-onecycle 0 --log-every 200 --skip-core \
  --data-dir /root/nanochat/data --tokenizer-dir /root/nanochat/tokenizer \
  --sequence-len 2048 \
  --modulation-diagnostics 1 \
  --cclblock-context-source norm_x \
  --cclblock-context-stream selective \
  --cclblock-aux-objective boundary --cclblock-aux-lambda 0.2"

# 22I: RemixedLinear weight modulation, K=8 templates, frozen routing
TAG="22_REMIX_WEIGHT_4T_FROZEN"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "23" "$TAG" "RemixedLinear weight mod, 8 templates, frozen routing"
    bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation weight \
      --p22-n-templates 8 --p22-template-routing-learned 0 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# 22J: RemixedLinear weight modulation, K=8 templates, learned routing
TAG="22_REMIX_WEIGHT_4T_LEARNED"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "24" "$TAG" "RemixedLinear weight mod, 8 templates, learned routing"
    bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation weight \
      --p22-n-templates 8 --p22-template-routing-learned 1 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# 22K: RemixedLinear householder modulation, K=8 templates, frozen routing
TAG="22_REMIX_HOUSE_4T_FROZEN"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "25" "$TAG" "RemixedLinear householder mod, 8 templates, frozen routing"
    bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation householder \
      --p22-n-templates 8 --p22-template-routing-learned 0 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# 22L: RemixedLinear householder modulation, K=8 templates, learned routing
TAG="22_REMIX_HOUSE_4T_LEARNED"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "26" "$TAG" "RemixedLinear householder mod, 8 templates, learned routing"
    bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation householder \
      --p22-n-templates 8 --p22-template-routing-learned 1 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# 22M: CKR (already MoE with K=8 full-rank branches, position+content routing)
# n_templates doesn't apply — CKR uses CausalKernelLinear, not RemixedLinear
TAG="22_CKR_FULL"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "27" "$TAG" "CKR K=8 branches (already overparameterized MoE)"
    bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation ckr \
      --cclblock-ckr-branches 8 \
      $DEPTH
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Phase 20+21+22 Sweep Complete                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Check $LOGFILE for results."
