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
# Shared settings
# ─────────────────────────────────────────
DEPTH=4

# Dense base-model flags
BASE_COMMON="--fp8 --max-shards 170 --models base \
  --device-batch-size 8 --use-onecycle 0 --log-every 200 --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 --mu-p-mode base_only \
  --warmup-ratio 0.15 \
  --research-dim -1"

# RemixedLinear flags — context conditioning ENABLED with SharedContextGates
# (batches both FFN layer gate MLPs into 3 shared matmuls per block; attn layers use local gates).
# Half the device-batch-size vs dense to give MoE layers VRAM headroom.
REMIX_COMMON="--fp8 --max-shards 170 --models remixed-linear \
  --device-batch-size 2 --use-onecycle 0 --log-every 1 --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 \
  --warmup-ratio 0.15 \
  --research-dim -1 \
  --remix-use-context 1 \
  --remix-shared-context-gates 1"

# ══════════════════════════════════════════════════════
# 1: Dense baseline — anchor reference
# ══════════════════════════════════════════════════════
#TAG="23_BASE_DENSE"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "1" "$TAG" "Dense baseline (plain transformer, no MoE)"
#    if bash scripts/research_sweep.sh $BASE_COMMON \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "════════════════ $TAG COMPLETE ════════════════"
#        mark_completed "$TAG"
#    else
#        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
#    fi
#fi

# ══════════════════════════════════════════════════════
# 1B: Dense RemixedLinear, weight mod (no template bank)
# ══════════════════════════════════════════════════════
TAG="23_REMIX_WEIGHT"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "1B" "$TAG" "Dense RemixedLinear, weight mod (no template bank)"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation weight \
      --p22-n-templates 1 \
      --remix-use-context 1 \
      --remix-shared-context-gates 1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "════════════════ $TAG COMPLETE ════════════════"
        mark_completed "$TAG"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi

# ══════════════════════════════════════════════════════
# 2: TinyExpert K=8, top-1
# ══════════════════════════════════════════════════════
#TAG="23_TINY_K8_TOP1"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "2" "$TAG" "TinyExpert K=8, top-1, no context, compile enabled"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --p23-tiny-expert 1 \
#      --p23-use-shared-block-router 1 \
#      --p23-n-experts 8 \
#      --p23-topk 1 \
#      --p23-learned-route 1 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "════════════════ $TAG COMPLETE ════════════════"
#        mark_completed "$TAG"
#    else
#        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
#    fi
#fi
#
# ══════════════════════════════════════════════════════
# 3: TinyExpert K=64, top-16
# ══════════════════════════════════════════════════════
#TAG="23_TINY_K64_TOP16"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "3" "$TAG" "TinyExpert K=64, top-16, no context, compile enabled"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --p23-tiny-expert 1 \
#      --p23-use-shared-block-router 1 \
#      --p23-n-experts 64 \
#      --p23-topk 16 \
#      --p23-learned-route 1 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "════════════════ $TAG COMPLETE ════════════════"
#        mark_completed "$TAG"
#    else
#        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
#    fi
#fi
#
# ══════════════════════════════════════════════════════
# 4: LoKR K=64, rank=4, top-16
# ══════════════════════════════════════════════════════
#TAG="23_LOKR_K64_TOP16"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "4" "$TAG" "LoKR K=64, top-16, rank=4, no context, compile enabled"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --p23-lokr 1 \
#      --p23-n-experts 64 \
#      --p23-topk 16 \
#      --p23-lokr-rank 4 \
#      --p23-learned-route 1 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "════════════════ $TAG COMPLETE ════════════════"
#        mark_completed "$TAG"
#    else
#        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
#    fi
#fi

# ══════════════════════════════════════════════════════
# 5: LinearMoE K=8, top-1
# ══════════════════════════════════════════════════════
TAG="23_LINEAR_MOE_K8_TOP1"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "5" "$TAG" "LinearMoE K=8, top-1, no context, compile enabled"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p23-linear-moe-experts 8 \
      --p23-linear-moe-topk 1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "════════════════ $TAG COMPLETE ════════════════"
        mark_completed "$TAG"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi

# ══════════════════════════════════════════════════════
# 6: LinearMoE K=8, top-16
# ══════════════════════════════════════════════════════
TAG="23_LINEAR_MOE_K8_TOP16"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "6" "$TAG" "LinearMoE K=8, top-16, no context, compile enabled"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p23-linear-moe-experts 8 \
      --p23-linear-moe-topk 16 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "════════════════ $TAG COMPLETE ════════════════"
        mark_completed "$TAG"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi

# ══════════════════════════════════════════════════════
# ARCHIVED (context-modulation dependent)
# ══════════════════════════════════════════════════════
# 23_REMIX_WEIGHT/HOUSE/CKR, 23_STD_MOE_TOP1/TOP_OPT
# 23_TINY_WEIGHT/HOUSE/CKR variants, 23_LOKR_WEIGHT/HOUSE/CKR

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Phase 23 Sweep Complete (6 experiments)           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Check $LOGFILE for results."
