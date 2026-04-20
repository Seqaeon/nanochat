#!/bin/bash
# Phase 25: RemixedLinear Component Ablation Sweep
#
# Goal: identify which gates drive performance and which drive latency overhead.
#
# Background:
#   RemixedLinear (weight mode) costs ~1.625C² FLOPs vs 2C² for dense.
#   The bottleneck is basis_modulator MLP (C→C/2→B = 0.625C²).
#   This sweep ablates the gate stack from nothing → full MLP to isolate value.
#
# Variants:
#   1. NO_CONTEXT   — pure structure (basis + mixing, no gates at all)
#   2. OUTPUT_ONLY  — only low-rank output gate from context (cheapest gated)
#   3. LINEAR_GATE  — single-linear basis gate + output gate  (75% of dense)
#   4. ATTN_GATE    — bilinear attention gate + output gate   (81% of dense)
#   5. MLP_GATE     — full MLP basis gate + output gate (current, 162% of dense)
#
# Usage:
#   bash scripts/p25_sweep.sh
#   bash scripts/p25_sweep.sh --force
#   SWEEP_LOG=sweep_p25.log bash scripts/p25_sweep.sh

set -o pipefail

LOGFILE="${SWEEP_LOG:-sweep_p25.log}"
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

# Shared settings — apples-to-apples with p23/p24 for fair comparison.
DEPTH=4

REMIX_COMMON="--fp8 --max-shards 170 --models remixed-linear \
  --device-batch-size 2 --use-onecycle 0 --log-every 1 --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 \
  --warmup-ratio 0.05 \
  --warmdown-ratio 0.50 \
  --final-lr-frac 0.10 \
  --research-dim -1 \
  --remix-shared-context-gates 0"
# NOTE: shared-context-gates disabled here — we want each variant to use its own
# gate modules so the ablation cleanly isolates basis_gate_mode.

# ══════════════════════════════════════════════════════
# 1) NO_CONTEXT — pure structure: y = W_m · LN(W_b · x)
#    No context, no gates. Measures what the basis+mixing structure gives alone.
#    Expected FLOPs: ~1.0C² (50% of dense)
# ══════════════════════════════════════════════════════
TAG="25_NO_CONTEXT"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "1" "$TAG" "No context, no gates — pure basis+mixing structure (cheapest)"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
        $DEPTH \
        --remix-use-context 0 \
        --remix-use-basis-gate 0 \
        --remix-use-output-gate 0 \
        --remix-basis-gate-mode none \
        >> "$LOGFILE" 2>&1; then
        mark_completed "$TAG"
        echo "════════════════ $TAG DONE ════════════════"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi

# ══════════════════════════════════════════════════════
# 2) OUTPUT_ONLY — low-rank output gate, no basis gate
#    gate_out = 1 + tanh(s · (W_oc·ctx) @ G)
#    Context enabled but basis gate disabled.
#    Expected FLOPs: ~1.02C² (51% of dense, output gate is negligible cost)
# ══════════════════════════════════════════════════════
TAG="25_OUTPUT_ONLY"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "2" "$TAG" "Output gate only — low-rank context gate on output, no basis gate"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
        $DEPTH \
        --remix-use-context 1 \
        --remix-use-basis-gate 0 \
        --remix-use-output-gate 1 \
        --remix-basis-gate-mode none \
        >> "$LOGFILE" 2>&1; then
        mark_completed "$TAG"
        echo "════════════════ $TAG DONE ════════════════"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi

# ══════════════════════════════════════════════════════
# 3) LINEAR_GATE — single linear basis gate + output gate
#    gate = σ(W_g · ctx) elementwise on basis; W_g: C → B (no hidden layer)
#    Expected FLOPs: ~1.27C² (63% of dense) vs MLP gate at 1.625C²
# ══════════════════════════════════════════════════════
TAG="25_LINEAR_GATE"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "3" "$TAG" "Linear basis gate — single projection C→B, no MLP hidden layer"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
        $DEPTH \
        --remix-use-context 1 \
        --remix-use-basis-gate 1 \
        --remix-use-output-gate 1 \
        --remix-basis-gate-mode linear \
        >> "$LOGFILE" 2>&1; then
        mark_completed "$TAG"
        echo "════════════════ $TAG DONE ════════════════"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi

# ══════════════════════════════════════════════════════
# 4) ATTN_GATE — bilinear attention gate + output gate
#    gate = σ(W_content·h ⊙ W_context·ctx); jointly conditioned on content + context
#    Expected FLOPs: ~1.33C² (67% of dense)
# ══════════════════════════════════════════════════════
TAG="25_ATTN_GATE"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "4" "$TAG" "Attention (bilinear) gate — content × context joint gating"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
        $DEPTH \
        --remix-use-context 1 \
        --remix-use-basis-gate 1 \
        --remix-use-output-gate 1 \
        --remix-basis-gate-mode attn \
        >> "$LOGFILE" 2>&1; then
        mark_completed "$TAG"
        echo "════════════════ $TAG DONE ════════════════"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi

# ══════════════════════════════════════════════════════
# 5) MLP_GATE — full 2-layer MLP basis gate (current default)
#    gate = σ(MLP(ctx))  MLP: C → C/2 → B (most expensive, the baseline)
#    Expected FLOPs: ~1.625C² (81% of dense) — upper bound for this study
# ══════════════════════════════════════════════════════
TAG="25_MLP_GATE"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "5" "$TAG" "MLP basis gate — 2-layer C→C/2→B (current default, most expensive)"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
        $DEPTH \
        --remix-use-context 1 \
        --remix-use-basis-gate 1 \
        --remix-use-output-gate 1 \
        --remix-basis-gate-mode mlp \
        >> "$LOGFILE" 2>&1; then
        mark_completed "$TAG"
        echo "════════════════ $TAG DONE ════════════════"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         Phase 25 Ablation Sweep Complete (5 runs)           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Check $LOGFILE for results."
echo ""
echo "Reading order (fastest to slowest, best wins):"
echo "  25_NO_CONTEXT   → 25_OUTPUT_ONLY → 25_LINEAR_GATE → 25_ATTN_GATE → 25_MLP_GATE"
