#!/bin/bash
# Phase 26: Streamlined Full-Rank RemixedLinear
#
# Goal: Find the minimal configuration that beats dense at ≤1.1× FLOPs.
#
# Background:
#   P25 showed full-rank RemixedLinear (B=C) beats dense (1.1604 vs 1.167 BPB).
#   P25 OUTPUT_ONLY (W_b@W_m + output gate, no basis gate) also beat dense (1.1655).
#   The W_b@W_m factorization = 2× FLOPs vs dense. Can we get the same quality
#   with a single W + output gate only (~1.01× FLOPs)?
#
# Experiments:
#   26A  23_BASE_DENSE          — anchor baseline
#   26B  26_OUTPUT_GATED        — single W + output gate (no factorization) ← KEY TEST
#   26C  26_FACTORED_NO_CTX     — W_b@W_m + LN, no context (does factorization help alone?)
#   26D  26_FACTORED_OUT_GATE   — W_b@W_m + output gate only (≈ P25 OUTPUT_ONLY, confirm)
#   26E  26_LOWRANK_BASIS_GATE  — W_b@W_m + lowrank basis gate + output gate (~free gate)
#   26F  26_MLP_GATE            — W_b@W_m + MLP basis gate + output gate (P25 winner, sanity)
#
# Usage (from nanochat root):
#   bash scripts/p26_sweep.sh           # run all (resumes via .state file)
#   bash scripts/p26_sweep.sh --force   # re-run everything
#   SWEEP_LOG=p26.log bash scripts/p26_sweep.sh

set -o pipefail

LOGFILE="${SWEEP_LOG:-sweep_p26.log}"
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
DEPTH=8

CCL_MOD="${CCL_MOD:-weight}"
CCL_STREAM="${CCL_STREAM:-selective}"

# Dense base-model flags
BASE_COMMON="--fp8 --max-shards 170 --models base \
  --device-batch-size 4 --use-onecycle 0 --log-every 200 --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 --mu-p-mode base_only \
  --warmup-ratio 0.05 \
  --research-dim -1"

# RemixedLinear / OutputGatedLinear shared flags
# research-dim -1 → full-rank basis (basis_size = min(in,out))
REMIX_COMMON="--fp8 --max-shards 170 --models remixed-linear \
  --device-batch-size 2 --use-onecycle 0 --log-every 1 --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 \
  --warmup-ratio 0.20 \
  --warmdown-ratio 0.50 \
  --research-dim -1 \
  --cclblock-modulation $CCL_MOD \
  --cclblock-context-stream $CCL_STREAM \
  --cclblock-gate-temperature 2.0 \
  --remix-shared-context-gates 0 \
  --remix-use-context 1"

# ══════════════════════════════════════════════════════
# 26A: Dense baseline — anchor reference
# ══════════════════════════════════════════════════════
TAG="26_BASE_DENSE"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "26A" "$TAG" "Dense baseline (plain transformer, no MoE)"
    if bash scripts/research_sweep.sh $BASE_COMMON \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "════════════════ $TAG COMPLETE ════════════════"
        mark_completed "$TAG"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi


# ══════════════════════════════════════════════════════
# 26B: OutputGatedLinear — KEY EXPERIMENT
#     Single W + low-rank centered output gate.
#     No W_b/W_m factorization, no basis gate, no LN.
#     FLOPs ≈ 1.01× dense. Params ≈ 1.01× dense.
#     If this beats dense: the W_b@W_m factorization is unnecessary.
#     If not: the factorization (spectral bias / intermediate LN) is load-bearing.
# ══════════════════════════════════════════════════════
#TAG="26_OUTPUT_GATED_${CCL_MOD^^}"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "26B" "$TAG" "OutputGatedLinear: single W + low-rank output gate, ~1.01× FLOPs"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --p26-output-gated-linear 1 \
#      --remix-use-basis-gate 0 \
#      --remix-use-output-gate 1 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "════════════════ $TAG COMPLETE ════════════════"
#        mark_completed "$TAG"
#    else
#        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
#    fi
#fi
#
#
# ══════════════════════════════════════════════════════
# 26C: Factored, No Context — W_b@W_m + LN, no gating
#     Tests whether the factorization structure alone (spectral
#     bias, intermediate LayerNorm) helps vs a plain dense W.
#     If yes: factorization provides an inductive bias.
#     If no: all quality comes from gating, not factorization.
# ══════════════════════════════════════════════════════
#TAG="26_FACTORED_NO_CTX_${CCL_MOD^^}"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "26C" "$TAG" "W_b@W_m + LN only, no context/gate — does factorization help?"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --remix-use-basis-gate 0 \
#      --remix-use-output-gate 0 \
#      --remix-use-context 0 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "════════════════ $TAG COMPLETE ════════════════"
#        mark_completed "$TAG"
#    else
#        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
#    fi
#fi
#
#
# ══════════════════════════════════════════════════════
# 26D: Factored + Output Gate only — confirmation of P25 OUTPUT_ONLY
#     W_b@W_m + output gate, no basis gate.
#     P25 got 1.1655 BPB with this setup (beats dense ~1.167).
#     Confirm that result here with the same schedule as the others.
# ══════════════════════════════════════════════════════
#TAG="26_FACTORED_OUT_GATE_${CCL_MOD^^}"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "26D" "$TAG" "W_b@W_m + output gate only (≈ P25 OUTPUT_ONLY) — confirm 1.1655"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --remix-use-basis-gate 0 \
#      --remix-use-output-gate 1 \
#      --remix-basis-gate-mode none \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "════════════════ $TAG COMPLETE ════════════════"
#        mark_completed "$TAG"
#    else
#        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
#    fi
#fi
#
#
# ══════════════════════════════════════════════════════
# 26E: Factored + Low-Rank Basis Gate + Output Gate
#     Replaces the MLP basis gate (~1.5C² FLOPs) with a low-rank
#     gate (r=8 vectors, ~16C FLOPs — same overhead as output gate).
#     If this matches MLP gate quality: MLP gate is wasteful.
#     If not: MLP gate's hidden dim provides something irreducible.
# ══════════════════════════════════════════════════════
#TAG="26_LOWRANK_GATE_${CCL_MOD^^}"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "26E" "$TAG" "W_b@W_m + lowrank basis gate (r=8) + output gate — gate at ~0 FLOPs"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --remix-use-basis-gate 1 \
#      --remix-use-output-gate 1 \
#      --remix-basis-gate-mode lowrank \
#      --remix-basis-gate-rank 8 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "════════════════ $TAG COMPLETE ════════════════"
#        mark_completed "$TAG"
#    else
#        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
#    fi
#fi
#
#
# ══════════════════════════════════════════════════════
# 26F: Factored + MLP Basis Gate + Output Gate — P25 winner
#     Reproduces the P25 best result (1.1604 BPB) in the P26
#     sweep setup for a fair apples-to-apples comparison.
# ══════════════════════════════════════════════════════
#TAG="26_MLP_GATE_${CCL_MOD^^}"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "26F" "$TAG" "W_b@W_m + MLP basis gate + output gate — P25 winner sanity check"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --remix-use-basis-gate 1 \
#      --remix-use-output-gate 1 \
#      --remix-basis-gate-mode mlp \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "════════════════ $TAG COMPLETE ════════════════"
#        mark_completed "$TAG"
#    else
#        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
#    fi
#fi


echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Phase 26 Sweep Complete (6 experiments)            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Check $LOGFILE for results."
echo ""
echo "Key comparison table:"
echo "  26A (dense)          → ~1.167 BPB  (1.00× FLOPs)"
echo "  26B (single W+gate)  → ?           (~1.01× FLOPs)  ← KEY"
echo "  26C (factored,no ctx)→ ?           (~2.00× FLOPs)"
echo "  26D (factored+outgate)→ ~1.165?    (~2.01× FLOPs)"
echo "  26E (factored+lr gate)→ ?          (~2.01× FLOPs)"
echo "  26F (factored+mlpgate)→ ~1.160?    (~2.75× FLOPs)"
