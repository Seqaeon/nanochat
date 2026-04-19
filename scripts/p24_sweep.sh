#!/bin/bash
# Phase 24: LinearMoE Variant Sweep (planning scaffold)
#
# Goal:
#   Add a dedicated sweep script for the new p24 linear-layer variants while
#   preserving p23 unchanged. This mirrors p23's resume-friendly structure.
#
# Variants covered:
#   - LinearMoE2 / SlicedWeightLinear-style bank slicing (quantile/Product-Key routing)
#   - LinearMoE3 / FoldedModulationLinear (sum consecutive groups + modulation gate)
#   - SequenceGatedLinear (dense baseline weights + sequence-level gating)
#
# Usage:
#   bash scripts/p24_sweep.sh
#   bash scripts/p24_sweep.sh --force
#   SWEEP_LOG=sweep_p24.log bash scripts/p24_sweep.sh

set -o pipefail

LOGFILE="${SWEEP_LOG:-sweep_p24.log}"
STATEFILE="${LOGFILE%.log}.state"
FORCE=0
if [[ "$1" == "--force" ]]; then
    FORCE=1
    rm -f "$STATEFILE"
    shift
fi

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

# Shared settings
DEPTH=4

# Keep this aligned with p23 remixed runs for apples-to-apples comparisons.
REMIX_COMMON="--fp8 --max-shards 170 --models remixed-linear \
  --device-batch-size 2 --use-onecycle 0 --log-every 1 --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 \
  --warmup-ratio 0.150 \
  --research-dim -1 \
  --remix-use-context 1 \
  --remix-shared-context-gates 1"

# -----------------------------------------------------------------------------
# NOTE:
# p24 flags are wired through research_sweep -> research_compare -> base_train.
# This script runs concrete p24 variant implementations.
# -----------------------------------------------------------------------------

# ══════════════════════════════════════════════════════
# 1) LinearMoE2: weight-bank slicing (Product-Key / quantile-routed)
#    reduction_scale=8 + minimum selected dim clamp=128
# ══════════════════════════════════════════════════════
TAG="24_LINEARMOE2_RS8_MIN128_TOKEN"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "1" "$TAG" "LinearMoE2 (per-token): select C//8 with min 128 via Product-Key router"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p24-use-sliced-weight 1 \
      --p24-sliced-weight-reduction-scale 8 \
      --p24-sliced-weight-min-select 128 \
      --p24-sliced-weight-scope per_token \
      --p24-sliced-weight-balance-coeff 0.01 \
      --p24-quantile-route 2 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "════════════════ $TAG COMPLETE ════════════════"
        mark_completed "$TAG"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi

TAG="24_LINEARMOE2_RS8_MIN128_BLOCK"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "2" "$TAG" "LinearMoE2 (per-block): one route decision per block"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p24-use-sliced-weight 1 \
      --p24-sliced-weight-reduction-scale 8 \
      --p24-sliced-weight-min-select 128 \
      --p24-sliced-weight-scope per_block \
      --p24-sliced-weight-balance-coeff 0.01 \
      --p24-quantile-route 2 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "════════════════ $TAG COMPLETE ════════════════"
        mark_completed "$TAG"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi

TAG="24_LINEARMOE2_RS8_MIN128_GLOBAL"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "3" "$TAG" "LinearMoE2 (global): one route decision per forward pass"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p24-use-sliced-weight 1 \
      --p24-sliced-weight-reduction-scale 8 \
      --p24-sliced-weight-min-select 128 \
      --p24-sliced-weight-scope global \
      --p24-sliced-weight-balance-coeff 0.01 \
      --p24-quantile-route 2 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "════════════════ $TAG COMPLETE ════════════════"
        mark_completed "$TAG"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi

# ══════════════════════════════════════════════════════
# 2) LinearMoE3: folded-sum groups then modulate grouped channels
#    (sum groups of reduction_scale=8 => C//8 channels)
# ══════════════════════════════════════════════════════
TAG="24_LINEARMOE3_FOLDED_RS8_GLOBAL_GATE"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "4" "$TAG" "LinearMoE3 (global gate): one modulation gate per forward pass"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p24-use-folded-mod 1 \
      --p24-folded-mod-reduction-scale 8 \
      --p24-folded-mod-scope global \
      --p24-folded-mod-gate-act sigmoid \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "════════════════ $TAG COMPLETE ════════════════"
        mark_completed "$TAG"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi

TAG="24_LINEARMOE3_FOLDED_RS8_BLOCK_GATE"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "5" "$TAG" "LinearMoE3 (per-block gate): one modulation gate per block"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p24-use-folded-mod 1 \
      --p24-folded-mod-reduction-scale 8 \
      --p24-folded-mod-scope per_block \
      --p24-folded-mod-gate-act sigmoid \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "════════════════ $TAG COMPLETE ════════════════"
        mark_completed "$TAG"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi

# ══════════════════════════════════════════════════════
# 3) Dense modulation baseline: full-dim sequence gating
# ══════════════════════════════════════════════════════
TAG="24_SEQUENCE_GATED_DENSE_GLOBAL"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "6" "$TAG" "SequenceGatedLinear (global): dense dims gated once per forward"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p24-use-sequence-gated-linear 1 \
      --p24-sequence-gated-scope global \
      --p24-sequence-gated-act sigmoid \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "════════════════ $TAG COMPLETE ════════════════"
        mark_completed "$TAG"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi

TAG="24_SEQUENCE_GATED_DENSE_BLOCK"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "7" "$TAG" "SequenceGatedLinear (per-block): dense dims gated once per block"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p24-use-sequence-gated-linear 1 \
      --p24-sequence-gated-scope per_block \
      --p24-sequence-gated-act sigmoid \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "════════════════ $TAG COMPLETE ════════════════"
        mark_completed "$TAG"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Phase 24 Sweep Scaffold Complete (7 runs)         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Check $LOGFILE for results."
