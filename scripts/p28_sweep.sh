#!/bin/bash
# Phase 28 Redux: Compute-Correct Template Routing Efficiency
#
# Fixes vs. original p28:
#   - --target-active-params 1  → token budget based on active (not total) params
#   - Causal chunk routing       → first token of chunk used for routing (no look-ahead)
#   - Chunk FLOPs estimator fix  → active_flops correctly reflects chunk amortization
#   - Per-module K overrides     → attn c_proj / c_q / c_k can have different n_templates
#
# Variants:
#   28A  — 8T top-1 sparse routing           (active-params token budget)
#   28C0 — shared W_b only                   (original 28C, no template reduction)
#   28C2 — shared W_b + c_proj K=2           (redundant proj templates removed)
#   28C3 — shared W_b + c_q/c_k K=2         (redundant Q/K templates removed)
#   28C5 — shared W_b + c_proj K=2 + qk K=2 (combined)
#   28D1 — chunk routing N=64  (causal-fixed, active-params token budget)
#   28D2 — chunk routing N=256 (causal-fixed, active-params token budget)
#
# Usage (from nanochat root):
#   bash scripts/p28_sweep.sh           # run all (resumes via .state file)
#   bash scripts/p28_sweep.sh --force   # re-run everything
#   SWEEP_LOG=my.log bash scripts/p28_sweep.sh

set -o pipefail

LOGFILE="${SWEEP_LOG:-sweep_p28r.log}"
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

# Common flags shared by all variants.
# Notes:
#   --target-active-params 1  → sparse variants get token budget = ratio × active_params
#   --p22-template-routing-learned 1 → learned (gradient-driven) routing weights
REMIX_COMMON="--fp8 --max-shards 170 --models remixed-linear \
  --device-batch-size 64 --use-onecycle 0 --log-every 1 --skip-core \
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
  --p22-n-templates 8 \
  --p22-template-routing-learned 1 \
  --remix-use-basis-gate 1 \
  --remix-use-output-gate 1 \
  --remix-basis-gate-mode centered \
  --target-tokens -1 \
  --target-active-params 1"

# ══════════════════════════════════════════════════════
# 28A: 8T top-1 sparse routing  [active-params token budget]
#   - 8 templates, hard top-1 routing per token
#   - token budget = ratio × active_params  (~39M, not 55M)
# ══════════════════════════════════════════════════════
TAG="28A_8T_TOP1"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "28A" "$TAG" "8T top-1 sparse routing with active-params token budget"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p22-template-topk 1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $TAG done"
        mark_completed "$TAG"
    else
        echo "❌  $TAG FAILED — will retry next run"
    fi
fi

# ══════════════════════════════════════════════════════
# 28C0: Shared W_b only  (no template count reduction)
#   - Shares single basis projection across all attn Q/K/V/O
#   - Baseline for understanding C-series: shared W_b alone worth it?
# ══════════════════════════════════════════════════════
TAG="28C0_SHARED_WB"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "28C0" "$TAG" "Shared W_b across all attn layers (baseline for C variants)"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p28-shared-basis 1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $TAG done"
        mark_completed "$TAG"
    else
        echo "❌  $TAG FAILED — will retry next run"
    fi
fi

# ══════════════════════════════════════════════════════
# 28C2: Shared W_b + attn c_proj K=2
#   - Motivated by similarity analysis: attn_c_proj mean sim 0.40–0.59
#   - Reduces c_proj from K=8 to K=2 (saves 75% of proj template bank params)
#   - All other sub-layers keep K=8
# ══════════════════════════════════════════════════════
TAG="28C2_SHARED_WB_PROJ_K2"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "28C2" "$TAG" "Shared W_b + c_proj reduced to K=2 templates"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p28-shared-basis 1 \
      --p28-attn-proj-templates 2 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $TAG done"
        mark_completed "$TAG"
    else
        echo "❌  $TAG FAILED — will retry next run"
    fi
fi

# ══════════════════════════════════════════════════════
# 28C3: Shared W_b + attn c_q/c_k K=2
#   - Motivated by: attn_c_q mean sim 0.30–0.51, attn_c_k mean sim 0.17–0.49
#   - c_v and c_proj keep K=8 (c_v is diverse, max sim only 0.26–0.36)
# ══════════════════════════════════════════════════════
TAG="28C3_SHARED_WB_QK_K2"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "28C3" "$TAG" "Shared W_b + c_q/c_k reduced to K=2 templates"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p28-shared-basis 1 \
      --p28-attn-qk-templates 2 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $TAG done"
        mark_completed "$TAG"
    else
        echo "❌  $TAG FAILED — will retry next run"
    fi
fi

# ══════════════════════════════════════════════════════
# 28C5: Shared W_b + c_proj K=2 + c_q/c_k K=2  (combined)
#   - Maximum template reduction for attn based on similarity data
#   - Only c_v keeps K=8 (genuinely diverse: max sim 0.26–0.36)
# ══════════════════════════════════════════════════════
TAG="28C5_SHARED_WB_PROJ_QK_K2"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "28C5" "$TAG" "Shared W_b + c_proj K=2 + c_q/c_k K=2 (combined max-reduction)"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p28-shared-basis 1 \
      --p28-attn-proj-templates 2 \
      --p28-attn-qk-templates 2 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $TAG done"
        mark_completed "$TAG"
    else
        echo "❌  $TAG FAILED — will retry next run"
    fi
fi

# ══════════════════════════════════════════════════════
# 28D1: Chunk routing N=64  [causal-fixed + active-params budget]
#   - Routing decision shared across each 64-token block
#   - Causal: routes from chunk FIRST token only (no future look-ahead)
#   - FLOPs estimator now correctly credits (1 - 1/64) savings
# ══════════════════════════════════════════════════════
TAG="28D1_CHUNK64"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "28D1" "$TAG" "Chunk routing N=64, causal fix + correct FLOPs"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p28-chunk-routing-size 64 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $TAG done"
        mark_completed "$TAG"
    else
        echo "❌  $TAG FAILED — will retry next run"
    fi
fi

# ══════════════════════════════════════════════════════
# 28D2: Chunk routing N=256  [causal-fixed + active-params budget]
#   - Larger chunks → stronger amortization (saves 255/256 of routing ops)
#   - But coarser routing: one weight blend per 256 tokens
# ══════════════════════════════════════════════════════
TAG="28D2_CHUNK256"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "28D2" "$TAG" "Chunk routing N=256, causal fix + correct FLOPs"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p28-chunk-routing-size 256 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $TAG done"
        mark_completed "$TAG"
    else
        echo "❌  $TAG FAILED — will retry next run"
    fi
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Phase 28 Redux Sweep Complete                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
