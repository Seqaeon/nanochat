#!/bin/bash
# Phase 23: Tiny-Experts RemixedLinear + Standard MoE Baseline
#
# Experiments:
#
#   ── DENSE BASELINES ────────────────────────────────────────────────────────
#   23_BASE_DENSE      — plain dense transformer (no MoE, no remix)
#   23_REMIX_WEIGHT    — dense RemixedLinear, weight mod, no template bank
#   23_REMIX_HOUSE     — dense RemixedLinear, householder mod, no template bank
#   23_REMIX_CKR       — dense RemixedLinear, CKR (causal kernel) mod
#
#   ── STANDARD MOE (param-parity experts, learned router) ───────────────────
#   23_STD_MOE_TOP1    — StandardMoE K=8, topk=1  (max sparsity)
#   23_STD_MOE_TOP_OPT — StandardMoE K=8, topk=optimal (E^0.7≈5 for E=8, c=0.3)
#   (expert_dim = 4*D // E for both — param parity with dense)
#
#   ── TINY EXPERTS — weight modulation ──────────────────────────────────────
#   23_TINY_WEIGHT_4T_FROZEN   — K_total=64, topk=16, frozen route
#   23_TINY_WEIGHT_4T_LEARNED  — K_total=64, topk=16, learned route
#   23_TINY_WEIGHT_TOP1_FROZEN — K_total=64, topk=1,  frozen route (full expert_dim)
#   23_TINY_WEIGHT_TOP1_LEARNED
#
#   ── TINY EXPERTS — householder modulation ─────────────────────────────────
#   23_TINY_HOUSE_4T_FROZEN    — K_total=64, topk=16, frozen route
#   23_TINY_HOUSE_4T_LEARNED   — K_total=64, topk=16, learned route
#
#   ── TINY EXPERTS — CKR modulation ─────────────────────────────────────────
#   23_TINY_CKR_4T_FROZEN      — K_total=64, topk=16, frozen route
#   23_TINY_CKR_4T_LEARNED     — K_total=64, topk=16, learned route
#
# Key design:
#   expert_dim = basis_size // topk  (compute parity with dense baseline)
#   FFN layers:  per-token routing
#   Attn layers: per-sequence routing (mean-pool x over T)
#
# Usage (from nanochat root):
#   bash scripts/p23_sweep.sh           # run all (resumes via .state file)
#   bash scripts/p23_sweep.sh --force   # re-run everything
#   SWEEP_LOG=p23.log bash scripts/p23_sweep.sh

set -e
set -o pipefail  # pipefail: pipeline exit code = first failing command, not tee's

LOGFILE="${SWEEP_LOG:-sweep.log}"
STATEFILE="${LOGFILE%.log}.state"
FORCE=0
if [[ "$1" == "--force" ]]; then
    FORCE=1
    rm -f "$STATEFILE"
    shift
fi

# ─────────────────────────────────────────
# Resume helpers
# ─────────────────────────────────────────
check_completed() {
    local tag="$1"
    if [[ "$FORCE" -eq 1 ]]; then return 1; fi
    if [[ ! -f "$STATEFILE" ]]; then return 1; fi
    grep -qx "$tag" "$STATEFILE" 2>/dev/null && return 0 || return 1
}

mark_completed() {
    echo "$1" >> "$STATEFILE"
}

print_header() {
    local num="$1" tag="$2" desc="$3"
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  [$num]  $tag"
    echo "║  $desc"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
}

# ─────────────────────────────────────────
# Shared settings (mirrors p20_sweeps.sh)
# ─────────────────────────────────────────
DEPTH=4

# Dense base-model flags (for baseline and StandardMoE runs)
BASE_COMMON="--fp8 --max-shards 170 --models base \
  --device-batch-size 8 --use-onecycle 0 --log-every 200 --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 --mu-p-mode base_only \
  --warmup-ratio 0.15 \
  --research-dim -1"

# RemixedLinear flags shared by all remix experiments
# --no-compile: 64 experts = 3180 Linear layers; torch.compile traces every one
#   and hangs for 20+ minutes before step 1. Eager mode is fine for 462 steps.
REMIX_COMMON="--fp8 --no-compile --max-shards 170 --models remixed-linear \
  --device-batch-size 2 --use-onecycle 0 --log-every 200 --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 \
  --warmup-ratio 0.15 \
  --research-dim -1 \
  --modulation-diagnostics 1 \
  --cclblock-context-source norm_x \
  --cclblock-context-stream selective \
  --cclblock-aux-objective boundary --cclblock-aux-lambda 0.2"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Phase 23: Tiny Experts Sweep (14 experiments)     ║"
echo "║  depth=$DEPTH  | log=$LOGFILE                               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ══════════════════════════════════════════════════════
# DENSE BASELINES
# ══════════════════════════════════════════════════════
#
# 1: Plain dense transformer (no MoE, no remix) — anchor reference
#TAG="23_BASE_DENSE"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "1" "$TAG" "Dense baseline (plain transformer, no MoE)"
#    bash scripts/research_sweep.sh $BASE_COMMON \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"
#    echo "════════════════ $TAG COMPLETE ════════════════"
#    mark_completed "$TAG"
#fi

# 2: Tiny Expert RemixedLinear, weight modulation — K=8, topk=1 (no expert bank)
TAG="23_REMIX_WEIGHT"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "2" "$TAG" "Tiny RemixedLinear, weight mod, K=8, top-1"
    bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation weight \
      --p23-tiny-expert 1 \
      --p23-use-shared-block-router 1 \
      --p23-n-experts 8 \
      --p23-topk 1 \
      --p23-learned-route 1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# 3: Tiny Expert RemixedLinear, householder modulation — K=8, topk=1
TAG="23_REMIX_HOUSE"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "3" "$TAG" "Tiny RemixedLinear, householder mod, K=8, top-1"
    bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation householder \
      --p23-tiny-expert 1 \
      --p23-use-shared-block-router 1 \
      --p23-n-experts 8 \
      --p23-topk 1 \
      --p23-learned-route 1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# 4: Tiny Expert RemixedLinear, CKR (causal kernel regression) modulation — K=8, topk=1
TAG="23_REMIX_CKR"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "4" "$TAG" "Tiny RemixedLinear, CKR mod, K=8 branches, K_exp=8, top-1"
    bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation ckr \
      --cclblock-ckr-branches 8 \
      --p23-tiny-expert 1 \
      --p23-use-shared-block-router 1 \
      --p23-n-experts 8 \
      --p23-topk 1 \
      --p23-learned-route 1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# ══════════════════════════════════════════════════════
# STANDARD MOE BASELINE (full-size experts, learned router)
# ══════════════════════════════════════════════════════

# 5: StandardMoE K=8, top-1 routing
#TAG="23_STD_MOE_TOP1"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "5" "$TAG" "StandardMoE K=8 full-size experts, top-1 routing"
#    bash scripts/research_sweep.sh $BASE_COMMON \
#      --p23-std-moe-experts 8 \
#      --p23-std-moe-topk 1 \
#      --p23-std-moe-aux-weight 0.01 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"
#    echo "════════════════ $TAG COMPLETE ════════════════"
#    mark_completed "$TAG"
#fi
#
# 6: StandardMoE K=8, optimal-sparsity topk (= E^(1-c) = 8^0.7 ≈ 5 for c=0.3)
#    Pass topk=-1 → StandardMoE_MLP resolves to _moe_optimal_topk(8, c=0.3)=5
#TAG="23_STD_MOE_TOP_OPT"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "6" "$TAG" "StandardMoE K=8 param-parity experts, optimal-sparsity topk (c=0.3 scaling law)"
#    bash scripts/research_sweep.sh $BASE_COMMON \
#      --p23-std-moe-experts 8 \
#      --p23-std-moe-topk -1 \
#      --p23-std-moe-aux-weight 0.01 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"
#    echo "════════════════ $TAG COMPLETE ════════════════"
#    mark_completed "$TAG"
#fi
#
# ══════════════════════════════════════════════════════

# 6.5: LinearMoE K=8, top-1 routed blending
TAG="23_LINEAR_MOE_TOP1"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "18" "$TAG" "LinearMoE K=8 weight matrices, top-1 blending"
    bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation weight \
      --p23-linear-moe-experts 8 \
      --p23-linear-moe-topk 1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# ══════════════════════════════════════════════════════
# TINY EXPERTS — weight modulation
# expert_dim = basis_size // topk  (compute parity)
# ══════════════════════════════════════════════════════

# 7: Tiny Expert weight mod — K_total=64, topk=16, frozen routing
#    expert_dim = basis_size // 16  e.g. 256//16 = 16 per expert
# TAG="23_TINY_WEIGHT_4T_FROZEN"
# if check_completed "$TAG"; then
#     echo "⏭  Skipping $TAG (already completed)"
# else
#     print_header "7" "$TAG" "TinyExpert weight mod, K_total=64, topk=16, frozen routing"
#     bash scripts/research_sweep.sh $REMIX_COMMON \
#       --cclblock-modulation weight \
#       --p23-tiny-expert 1 \
#       --p23-use-shared-block-router 1 \
#       --p23-n-experts 64 \
#       --p23-topk 16 \
#       --p23-learned-route 0 \
#       $DEPTH 2>&1 | tee -a "$LOGFILE"
#     echo "════════════════ $TAG COMPLETE ════════════════"
#     mark_completed "$TAG"
# fi

# 8: Tiny Expert weight mod — K_total=64, topk=16, learned routing
TAG="23_TINY_WEIGHT_4T_LEARNED"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "8" "$TAG" "TinyExpert weight mod, K_total=64, topk=16, learned routing"
    bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation weight \
      --p23-tiny-expert 1 \
      --p23-use-shared-block-router 1 \
      --p23-n-experts 64 \
      --p23-topk 16 \
      --p23-learned-route 1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# 9: Tiny Expert weight mod — K_total=64, topk=1, frozen routing
#    expert_dim = basis_size (full-rank per expert) — max specialization
# TAG="23_TINY_WEIGHT_TOP1_FROZEN"
# if check_completed "$TAG"; then
#     echo "⏭  Skipping $TAG (already completed)"
# else
#     print_header "9" "$TAG" "TinyExpert weight mod, K_total=64, topk=1 (full expert_dim), frozen routing"
#     bash scripts/research_sweep.sh $REMIX_COMMON \
#       --cclblock-modulation weight \
#       --p23-tiny-expert 1 \
#       --p23-use-shared-block-router 1 \
#       --p23-n-experts 64 \
#       --p23-topk 1 \
#       --p23-learned-route 0 \
#       $DEPTH 2>&1 | tee -a "$LOGFILE"
#     echo "════════════════ $TAG COMPLETE ════════════════"
#     mark_completed "$TAG"
# fi

# 10: Tiny Expert weight mod — K_total=64, topk=1, learned routing
TAG="23_TINY_WEIGHT_TOP1_LEARNED"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "10" "$TAG" "TinyExpert weight mod, K_total=64, topk=1, learned routing"
    bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation weight \
      --p23-tiny-expert 1 \
      --p23-use-shared-block-router 1 \
      --p23-n-experts 64 \
      --p23-topk 1 \
      --p23-learned-route 1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# ══════════════════════════════════════════════════════
# TINY EXPERTS — householder modulation
# ══════════════════════════════════════════════════════

# 11: Tiny Expert householder mod — K_total=64, topk=16, frozen routing
# TAG="23_TINY_HOUSE_4T_FROZEN"
# if check_completed "$TAG"; then
#     echo "⏭  Skipping $TAG (already completed)"
# else
#     print_header "11" "$TAG" "TinyExpert householder mod, K_total=64, topk=16, frozen routing"
#     bash scripts/research_sweep.sh $REMIX_COMMON \
#       --cclblock-modulation householder \
#       --p23-tiny-expert 1 \
#       --p23-use-shared-block-router 1 \
#       --p23-n-experts 64 \
#       --p23-topk 16 \
#       --p23-learned-route 0 \
#       $DEPTH 2>&1 | tee -a "$LOGFILE"
#     echo "════════════════ $TAG COMPLETE ════════════════"
#     mark_completed "$TAG"
# fi

# 12: Tiny Expert householder mod — K_total=64, topk=16, learned routing
TAG="23_TINY_HOUSE_4T_LEARNED"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "12" "$TAG" "TinyExpert householder mod, K_total=64, topk=16, learned routing"
    bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation householder \
      --p23-tiny-expert 1 \
      --p23-use-shared-block-router 1 \
      --p23-n-experts 64 \
      --p23-topk 16 \
      --p23-learned-route 1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# ══════════════════════════════════════════════════════
# TINY EXPERTS — CKR (Causal Kernel Regression) modulation
# CKR uses position-aware causal kernels for basis modulation.
# The Tiny Expert expert bank replaces the template_bank in the output path.
# ══════════════════════════════════════════════════════

# 13: Tiny Expert CKR mod — K_total=64, topk=16, frozen routing
# TAG="23_TINY_CKR_4T_FROZEN"
# if check_completed "$TAG"; then
#     echo "⏭  Skipping $TAG (already completed)"
# else
#     print_header "13" "$TAG" "TinyExpert CKR mod, K_total=64, topk=16, K_ckr=8 branches, frozen routing"
#     bash scripts/research_sweep.sh $REMIX_COMMON \
#       --cclblock-modulation ckr \
#       --cclblock-ckr-branches 8 \
#       --p23-tiny-expert 1 \
#       --p23-use-shared-block-router 1 \
#       --p23-n-experts 64 \
#       --p23-topk 16 \
#       --p23-learned-route 0 \
#       $DEPTH 2>&1 | tee -a "$LOGFILE"
#     echo "════════════════ $TAG COMPLETE ════════════════"
#     mark_completed "$TAG"
# fi

# 14: Tiny Expert CKR mod — K_total=64, topk=16, learned routing
TAG="23_TINY_CKR_4T_LEARNED"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "14" "$TAG" "TinyExpert CKR mod, K_total=64, topk=16, K_ckr=8 branches, learned routing"
    bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation ckr \
      --cclblock-ckr-branches 8 \
      --p23-tiny-expert 1 \
      --p23-use-shared-block-router 1 \
      --p23-n-experts 64 \
      --p23-topk 16 \
      --p23-learned-route 1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# ══════════════════════════════════════════════════════
# LoKR EXPERTS — Iso-Parameter Low-Rank MoE
# ══════════════════════════════════════════════════════

# 15: LoKR weight mod — K_total=64, rank=4, topk=16, learned routing
TAG="23_LOKR_WEIGHT_4T_LEARNED"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "15" "$TAG" "LoKR weight mod, K_total=64, topk=16, rank=4, learned routing"
    bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation weight \
      --p23-lokr 1 \
      --p23-n-experts 64 \
      --p23-topk 16 \
      --p23-lokr-rank 4 \
      --p23-learned-route 1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# 16: LoKR householder mod — K_total=64, rank=4, topk=16, learned routing
TAG="23_LOKR_HOUSE_4T_LEARNED"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "16" "$TAG" "LoKR householder mod, K_total=64, topk=16, rank=4, learned routing"
    bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation householder \
      --p23-lokr 1 \
      --p23-n-experts 64 \
      --p23-topk 16 \
      --p23-lokr-rank 4 \
      --p23-learned-route 1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

# 17: LoKR CKR mod — K_total=64, rank=4, topk=16, learned routing
TAG="23_LOKR_CKR_4T_LEARNED"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "17" "$TAG" "LoKR CKR mod, K_total=64, topk=16, rank=4, learned routing"
    bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation ckr \
      --cclblock-ckr-branches 8 \
      --p23-lokr 1 \
      --p23-n-experts 64 \
      --p23-topk 16 \
      --p23-lokr-rank 4 \
      --p23-learned-route 1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"
    echo "════════════════ $TAG COMPLETE ════════════════"
    mark_completed "$TAG"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Phase 23 Sweep Complete (14 experiments)          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Check $LOGFILE for results."
