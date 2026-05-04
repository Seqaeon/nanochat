#!/bin/bash

set -o pipefail

# ── Phase 30: LayerNorm Ablation & MoE Retrain Sweep ──────────────────────────
#
# Purpose: Address NeurIPS reviewer weaknesses by:
#   30A: Dense + intermediate LayerNorm (isolate LN confound from K=1 ablation)
#   30B: RemixedLinear K=1, no LN (isolate factorization effect from LN)
#   30C: Standard MoE top-all retrained with TOTAL param budget (not active)
#   30D: Standard MoE top-1 retrained with TOTAL param budget
#
# Usage:
#   bash scripts/p30_sweep.sh [--force] [depth...]
#   DEPTH=4 bash scripts/p30_sweep.sh
#
# ──────────────────────────────────────────────────────────────────────────────

# ── Startup acceleration ───────────────────────────────────────────────────────
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-out/.triton_cache}"
export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}"

# Parse --force
FORCE=0
if [[ "${1:-}" == "--force" ]]; then FORCE=1; shift; fi

# Collect all numeric positional args as depths
DEPTHS=()
while [[ -n "${1:-}" && "$1" =~ ^[0-9]+$ ]]; do
    DEPTHS+=("$1"); shift
done
[[ ${#DEPTHS[@]} -eq 0 ]] && DEPTHS=("${DEPTH:-4}")

# Multi-depth: re-invoke self for each depth sequentially.
if [[ ${#DEPTHS[@]} -gt 1 ]]; then
    echo "P30 multi-depth sweep: ${DEPTHS[*]}"
    for _d in "${DEPTHS[@]}"; do
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  ▶ Starting depth ${_d}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        bash "$0" $([[ "$FORCE" == 1 ]] && echo "--force") "$_d" \
            || echo "❌  Depth ${_d} sweep failed — continuing with next depth"
    done
    exit 0
fi
DEPTH="${DEPTHS[0]}"

# Log and state files auto-named per depth.
LOGFILE="${SWEEP_LOG:-sweep_p30_d${DEPTH}.log}"
STATEFILE="${LOGFILE%.log}_state.json"

if [[ "$FORCE" == 1 ]]; then
    rm -f "$STATEFILE"
fi

## ---------------------------------------------------------------------------
# JSON state helpers (same as p29_sweep.sh)
# ---------------------------------------------------------------------------
_EMPTY_STATE='{"completed":[],"unfinished":{},"output_dir":{}}'

_state_init() {
    if [[ ! -f "$STATEFILE" ]]; then
        echo "$_EMPTY_STATE" > "${STATEFILE}.tmp" && mv "${STATEFILE}.tmp" "$STATEFILE"
    fi
}

_state_read() {
    python3 -c "
import json, sys
try:
    with open('$STATEFILE') as f:
        s = json.load(f)
    print(json.dumps(s))
except (json.JSONDecodeError, FileNotFoundError) as e:
    print('WARNING: state file corrupt or missing, resetting: ' + str(e), file=sys.stderr)
    s = {'completed': [], 'unfinished': {}, 'output_dir': {}}
    with open('${STATEFILE}.tmp', 'w') as f:
        json.dump(s, f, indent=2)
    import os; os.rename('${STATEFILE}.tmp', '$STATEFILE')
    print(json.dumps(s))
"
}

check_completed() {
    local tag="$1"
    if [[ "$FORCE" -eq 1 ]]; then return 1; fi
    _state_init
    python3 -c "
import json, sys
try:
    with open('$STATEFILE') as f: s = json.load(f)
except (json.JSONDecodeError, FileNotFoundError):
    sys.exit(1)
sys.exit(0 if '$tag' in s.get('completed', []) else 1)
" 2>/dev/null && return 0 || return 1
}

mark_started() {
    local tag="$1" ckpt_dir="$2" out_dir="$3"
    _state_init
    python3 - <<PYEOF
import json, os
try:
    with open('$STATEFILE') as f: s = json.load(f)
except (json.JSONDecodeError, FileNotFoundError):
    s = {'completed': [], 'unfinished': {}, 'output_dir': {}}
s.setdefault('unfinished', {})['$tag'] = '$ckpt_dir'
s.setdefault('output_dir', {})['$tag'] = '$out_dir'
with open('${STATEFILE}.tmp', 'w') as f: json.dump(s, f, indent=2)
os.rename('${STATEFILE}.tmp', '$STATEFILE')
PYEOF
}

mark_completed() {
    local tag="$1"
    _state_init
    python3 - <<PYEOF
import json, os
try:
    with open('$STATEFILE') as f: s = json.load(f)
except (json.JSONDecodeError, FileNotFoundError):
    s = {'completed': [], 'unfinished': {}, 'output_dir': {}}
if '$tag' not in s.get('completed', []):
    s.setdefault('completed', []).append('$tag')
s.get('unfinished', {}).pop('$tag', None)
with open('${STATEFILE}.tmp', 'w') as f: json.dump(s, f, indent=2)
os.rename('${STATEFILE}.tmp', '$STATEFILE')
PYEOF
}

get_out_dir() {
    local tag="$1"
    _state_init
    python3 -c "
import json, sys
try:
    with open('$STATEFILE') as f: s = json.load(f)
except (json.JSONDecodeError, FileNotFoundError):
    print('', end='')
    sys.exit(0)
print(s.get('output_dir', {}).get('$tag', ''), end='')
" 2>/dev/null
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

ASPECT_RATIO="${ASPECT_RATIO:-64}"
MODEL_DIM=$(python3 -c "d=$DEPTH; ar=$ASPECT_RATIO; h=128; print(((d*ar+h-1)//h)*h)")

CCL_MOD="${CCL_MOD:-weight}"
CCL_STREAM="${CCL_STREAM:-selective}"
P30_OUT_BASE="${P30_OUT_BASE:-out/sweep_p30}"

# ── Common flags for RemixedLinear experiments ────────────────────────────────
# Mirrors P29 canonical config but with specific overrides per experiment.
REMIX_COMMON="--fp8 --max-shards 170 --models remixed-linear \
  --device-batch-size 16 --total-batch-size -1 --use-onecycle 0 --log-every 200 --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 --aspect-ratio $ASPECT_RATIO \
  --target-param-data-ratio 10.5 \
  --warmup-ratio 0.005 \
  --warmdown-ratio 0.65 \
  --final-lr-frac 0.05 \
  --research-dim -1 \
  --remix-basis-size $MODEL_DIM \
  --cclblock-modulation $CCL_MOD \
  --cclblock-context-stream $CCL_STREAM \
  --cclblock-gate-temperature 2.0 \
  --remix-shared-context-gates 0 \
  --remix-use-context 1 \
  --p22-template-routing-learned 1 \
  --remix-use-basis-gate 0 \
  --remix-use-output-gate 1 \
  --remix-basis-gate-mode centered \
  --target-tokens -1 \
  --target-active-params 0 \
  --save-every 200 \
  --p23-quantile-route 1"

# ── Common flags for dense / MoE baselines ────────────────────────────────────
# NOTE: --target-active-params 0 ensures token budget = ratio × TOTAL params.
# This fixes the previous issue where MoE was undertrained due to
# active-param-based budget calculation.
BASE_COMMON="--fp8 --max-shards 170 --models base \
  --device-batch-size 128 --total-batch-size -1 --use-onecycle 0 --log-every 200 --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 \
  --target-param-data-ratio 10.5 \
  --warmup-ratio 0.20 \
  --warmdown-ratio 0.50 \
  --research-dim -1 \
  --target-tokens -1 \
  --target-active-params 0 \
  --save-every 200"


# ══════════════════════════════════════════════════════
# 30A: Dense Baseline + Intermediate LayerNorm
#   - Standard dense transformer with LN after c_fc
#   - Isolates the LayerNorm confound from K=1 ablation
#   - If this matches K=1 no-context (1.168), the 0.002
#     improvement is from LN, not the factored structure
# ══════════════════════════════════════════════════════
TAG="30A_DENSE_LN_D${DEPTH}"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "30A" "$TAG" "Dense baseline + intermediate LayerNorm (LN confound isolation)"
    _SAVED=$(get_out_dir "$TAG")
    _RUN_DIR="${_SAVED:-${P30_OUT_BASE}/${TAG}}"
    mark_started "$TAG" "${_RUN_DIR}/depth_${DEPTH}/ckpt_base/base" "$_RUN_DIR"
    if bash scripts/research_sweep.sh $BASE_COMMON \
      --out-dir "$_RUN_DIR" \
      --dense-intermediate-ln 1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $TAG done"
        mark_completed "$TAG"
    else
        echo "❌  $TAG FAILED — will retry next run"
    fi
fi

# ══════════════════════════════════════════════════════
# 30B: RemixedLinear K=1, No Context, No Gates, No LN
#   - Single template, no modulation, but WITHOUT ln_basis
#   - Isolates pure factorization effect (W_b @ T_1)
#   - Compare with K=1 no-context WITH LN (from P29: 1.168)
# ══════════════════════════════════════════════════════
TAG="30B_REMIX_K1_NO_LN_D${DEPTH}"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "30B" "$TAG" "RemixedLinear K=1, no context/gates, no intermediate LN"
    _SAVED=$(get_out_dir "$TAG")
    _RUN_DIR="${_SAVED:-${P30_OUT_BASE}/${TAG}}"
    mark_started "$TAG" "${_RUN_DIR}/depth_${DEPTH}/ckpt_remixed-linear/remixed-linear" "$_RUN_DIR"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --out-dir "$_RUN_DIR" \
      --p22-n-templates 1 \
      --remix-use-context 0 \
      --remix-use-basis-gate 0 \
      --remix-use-output-gate 0 \
      --remix-disable-ln-basis 1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $TAG done"
        mark_completed "$TAG"
    else
        echo "❌  $TAG FAILED — will retry next run"
    fi
fi

# ══════════════════════════════════════════════════════
# 30C: Standard MoE top-all RETRAINED
#   - K=8 full-size experts, all active (top-all)
#   - Token budget from TOTAL params (not active)
#   - Fixes the undertrained MoE_all from P29
# ══════════════════════════════════════════════════════
TAG="30C_MOE_ALL_RETRAIN_D${DEPTH}"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "30C" "$TAG" "Standard MoE top-all, retrained with total-param token budget"
    _SAVED=$(get_out_dir "$TAG")
    _RUN_DIR="${_SAVED:-${P30_OUT_BASE}/${TAG}}"
    mark_started "$TAG" "${_RUN_DIR}/depth_${DEPTH}/ckpt_base/base" "$_RUN_DIR"
    if bash scripts/research_sweep.sh $BASE_COMMON \
      --out-dir "$_RUN_DIR" \
      --p23-std-moe-experts 8 \
      --p23-std-moe-topk 8 \
      --p23-std-moe-aux-weight 0.01 \
      --target-active-params 0 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $TAG done"
        mark_completed "$TAG"
    else
        echo "❌  $TAG FAILED — will retry next run"
    fi
fi

# ══════════════════════════════════════════════════════
# 30D: Standard MoE top-1 RETRAINED
#   - K=8 full-size experts, top-1 sparse routing
#   - Token budget from TOTAL params (not active)
# ══════════════════════════════════════════════════════
TAG="30D_MOE_TOP1_RETRAIN_D${DEPTH}"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "30D" "$TAG" "Standard MoE top-1, retrained with total-param token budget"
    _SAVED=$(get_out_dir "$TAG")
    _RUN_DIR="${_SAVED:-${P30_OUT_BASE}/${TAG}}"
    mark_started "$TAG" "${_RUN_DIR}/depth_${DEPTH}/ckpt_base/base" "$_RUN_DIR"
    if bash scripts/research_sweep.sh $BASE_COMMON \
      --out-dir "$_RUN_DIR" \
      --p23-std-moe-experts 8 \
      --p23-std-moe-topk 1 \
      --p23-std-moe-aux-weight 0.01 \
      --target-active-params 0 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $TAG done"
        mark_completed "$TAG"
    else
        echo "❌  $TAG FAILED — will retry next run"
    fi
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Phase 30 Sweep Complete                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
