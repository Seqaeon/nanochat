#!/bin/bash
# Phase 28: FLOPs-Efficient Template Routing
# 7 experiments testing sparse routing, shared basis, chunk routing, and global template bank
#
# Usage (from nanochat root):
#   bash scripts/p28_sweep.sh           # run all (resumes via .state file)
#   bash scripts/p28_sweep.sh --force   # re-run everything
#   SWEEP_LOG=p28.log bash scripts/p28_sweep.sh
#
# Experiment layout:
#   28A  — Top-1 of 8 templates (max sparsity)
#   28B  — Top-2 of 8 templates (balanced sparsity)
#   28C  — Shared W_b across attn Q/K/V/O per block
#   28D1 — Amortized chunk routing, chunk_size=64
#   28D2 — Amortized chunk routing, chunk_size=256
#   28E  — Global template bank, FFN-only
#   28F  — Global template bank, FFN + Attn

set -o pipefail

LOGFILE="${SWEEP_LOG:-sweep_p28.log}"
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

DEPTH=4
MODEL_DIM=$(python3 -c "d=$DEPTH; h=128; print(((d*64+h-1)//h)*h)")

CCL_MOD="${CCL_MOD:-weight}"
CCL_STREAM="${CCL_STREAM:-selective}"

# Shared base flags: full-rank RemixedLinear with 8 templates + output gate (best P27 config)
REMIX_COMMON="--fp8 --max-shards 170 --models remixed-linear \
  --device-batch-size 128 --use-onecycle 0 --log-every 1 --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 \
  --warmup-ratio 0.20 \
  --warmdown-ratio 0.50 \
  --research-dim -1 \
  --remix-basis-size $MODEL_DIM \
  --cclblock-modulation $CCL_MOD \
  --cclblock-context-stream $CCL_STREAM \
  --cclblock-gate-temperature 2.0 \
  --remix-shared-context-gates 0 \
  --remix-use-context 1 \
  --remix-use-output-gate 1 \
  --remix-use-basis-gate 0 \
  --p22-n-templates 8 \
  --p22-template-routing-learned 1 \
  --target-tokens -1"

# ══════════════════════════════════════════════════════
# 28A: Top-1 of 8 learned templates (hard sparse routing)
# FLOPs target: ~dense (1 template active = 1 matmul vs K weighted matmuls)
# ══════════════════════════════════════════════════════
TAG="28A_8T_TOP1"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "28A" "$TAG" "Top-1 sparse routing over 8 learned templates"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p22-template-topk 1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        mark_completed "$TAG"
        echo "✅  $TAG done"
    else
        echo "❌  $TAG FAILED" | tee -a "$LOGFILE"; exit 1
    fi
fi

# ══════════════════════════════════════════════════════
# 28B: Top-2 of 8 learned templates (balanced sparsity)
# FLOPs target: ~1.25× dense (2 active templates)
# ══════════════════════════════════════════════════════
TAG="28B_8T_TOP2"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "28B" "$TAG" "Top-2 sparse routing over 8 learned templates"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p22-template-topk 2 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        mark_completed "$TAG"
        echo "✅  $TAG done"
    else
        echo "❌  $TAG FAILED" | tee -a "$LOGFILE"; exit 1
    fi
fi

# ══════════════════════════════════════════════════════
# 28C: Shared W_b across attn Q/K/V/O per block
# Computes h_basis once for all 4 attn projections → 3/4 basis FLOPs saved in attn
# ══════════════════════════════════════════════════════
TAG="28C_SHARED_WB_ATTN"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "28C" "$TAG" "Single shared W_b projection across attn Q/K/V/O per block"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p28-shared-basis 1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        mark_completed "$TAG"
        echo "✅  $TAG done"
    else
        echo "❌  $TAG FAILED" | tee -a "$LOGFILE"; exit 1
    fi
fi

# ══════════════════════════════════════════════════════
# 28D1: Amortized chunk routing, chunk_size=64
# Route once per 64 tokens → 64× fewer routing FLOPs, materialize 1 W_eff per chunk
# ══════════════════════════════════════════════════════
TAG="28D1_CHUNK64"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "28D1" "$TAG" "Amortized template routing per 64-token chunk"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p28-chunk-routing-size 64 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        mark_completed "$TAG"
        echo "✅  $TAG done"
    else
        echo "❌  $TAG FAILED" | tee -a "$LOGFILE"; exit 1
    fi
fi

# ══════════════════════════════════════════════════════
# 28D2: Amortized chunk routing, chunk_size=256
# Coarser amortization → even fewer routing FLOPs
# ══════════════════════════════════════════════════════
TAG="28D2_CHUNK256"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "28D2" "$TAG" "Amortized template routing per 256-token chunk"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p28-chunk-routing-size 256 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        mark_completed "$TAG"
        echo "✅  $TAG done"
    else
        echo "❌  $TAG FAILED" | tee -a "$LOGFILE"; exit 1
    fi
fi

# ══════════════════════════════════════════════════════
# 28E: Global template bank, FFN only
# Single shared K templates for all FFN layers with tiny per-layer router
# Params: K×(4D+D)×B instead of n_layers×K×(4D+D)×B
# ══════════════════════════════════════════════════════
TAG="28E_GLOBAL_BANK_FFN"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "28E" "$TAG" "Global template bank: FFN-only (shared K templates across all layers)"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p28-global-template-bank ffn \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        mark_completed "$TAG"
        echo "✅  $TAG done"
    else
        echo "❌  $TAG FAILED" | tee -a "$LOGFILE"; exit 1
    fi
fi

# ══════════════════════════════════════════════════════
# 28F: Global template bank, FFN + Attention
# Extends global bank to attn Q/K/V/O projections as well
# Max parameter compression across all RemixedLinear layers
# ══════════════════════════════════════════════════════
TAG="28F_GLOBAL_BANK_ALL"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "28F" "$TAG" "Global template bank: FFN + Attn (all RemixedLinear layers)"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p28-global-template-bank all \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        mark_completed "$TAG"
        echo "✅  $TAG done"
    else
        echo "❌  $TAG FAILED" | tee -a "$LOGFILE"; exit 1
    fi
fi

echo ""
echo "════════════════════════════════════════"
echo "  Phase 28 sweep complete ✅"
echo "════════════════════════════════════════"
