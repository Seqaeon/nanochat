#!/bin/bash
# Phase 27: Full-Rank Remixed Weight vs 4T Learned
# Usage (from nanochat root):
#   bash scripts/p27_sweep.sh           # run all (resumes via .state file)
#   bash scripts/p27_sweep.sh --force   # re-run everything
#   SWEEP_LOG=p27.log bash scripts/p27_sweep.sh

set -o pipefail

LOGFILE="${SWEEP_LOG:-sweep_p27.log}"
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

CCL_MOD="${CCL_MOD:-weight}"
CCL_STREAM="${CCL_STREAM:-selective}"

# --research-dim -1 tells research_compare.py to use model_dim as target_dim AND
# automatically appends --remix-basis-size model_dim, ensuring truly full-rank
# basis at any depth without hardcoding.
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
  --remix-use-context 1 \
  --target-tokens -1"


# ══════════════════════════════════════════════════════
# 27A: 27_FULLRANK_REMIX_WEIGHT
# ══════════════════════════════════════════════════════
TAG="27_FULLRANK_REMIX_WEIGHT"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "27A" "$TAG" "Standard full-rank remixed linear"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p22-n-templates 1 \
      --remix-use-basis-gate 1 \
      --remix-use-output-gate 1 \
      --remix-basis-gate-mode centered \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "════════════════ $TAG COMPLETE ════════════════"
        mark_completed "$TAG"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi

# ══════════════════════════════════════════════════════
# 27B: 27_REMIX_WEIGHT_4T_LEARNED
# ══════════════════════════════════════════════════════
TAG="27_REMIX_WEIGHT_4T_LEARNED"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "27B" "$TAG" "Full-rank remixed linear with 4 learned templates"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --p22-n-templates 4 \
      --p22-template-routing-learned 1 \
      --remix-use-basis-gate 1 \
      --remix-use-output-gate 1 \
      --remix-basis-gate-mode centered \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "════════════════ $TAG COMPLETE ════════════════"
        mark_completed "$TAG"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Phase 27 Sweep Complete                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
