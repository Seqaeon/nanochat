#!/bin/bash
# p35_ablation_sweep.sh — Diagnose RemixedLinear D8 underperformance
#
# All arms are FLOP-matched at 6.058865e+16 (the dense D8 budget).
# This isolates which component of RemixedLinear wastes the most FLOPs.
#
# Arms:
#   A. K=1 (single template)     — is the factorization itself the problem?
#   B. No context/gates          — is the gate/context overhead wasted FLOPs?
#   C. No intermediate LN        — does LN(W_b·x) before W_m hurt expressiveness?
#   D. Dense + intermediate LN   — does adding LN to dense cause the same gap?
#   E. Current 29C baseline      — the full RemixedLinear for reference
#
# Usage:
#   bash scripts/p35_ablation_sweep.sh [--force] [depth]
#   DEPTH=8 bash scripts/p35_ablation_sweep.sh
#
set -o pipefail

# ── Startup acceleration ───────────────────────────────────────────────────────
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-out/.triton_cache}"
export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}"

# Parse --force
FORCE=0
if [[ "${1:-}" == "--force" ]]; then FORCE=1; shift; fi

DEPTH="${1:-8}"
TARGET_FLOPS="6.058865e+16"

LOGFILE="sweep_p35_ablation_d${DEPTH}.log"
STATEFILE="${LOGFILE%.log}_state.json"

if [[ "$FORCE" == 1 ]]; then
    rm -f "$STATEFILE"
fi

## ---------------------------------------------------------------------------
# JSON state helpers (same as p29_sweep.sh)
## ---------------------------------------------------------------------------
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
    local tag="$1" ckpt_dir="$2" run_dir="$3"
    _state_init
    python3 -c "
import json, os
with open('$STATEFILE') as f: s = json.load(f)
s.setdefault('unfinished', {})['$tag'] = '$ckpt_dir'
s.setdefault('output_dir', {})['$tag'] = '$run_dir'
with open('${STATEFILE}.tmp', 'w') as f: json.dump(s, f, indent=2)
os.rename('${STATEFILE}.tmp', '$STATEFILE')
"
}

mark_completed() {
    local tag="$1"
    _state_init
    python3 -c "
import json, os
with open('$STATEFILE') as f: s = json.load(f)
s.setdefault('completed', []).append('$tag')
s.get('unfinished', {}).pop('$tag', None)
with open('${STATEFILE}.tmp', 'w') as f: json.dump(s, f, indent=2)
os.rename('${STATEFILE}.tmp', '$STATEFILE')
"
}

get_out_dir() {
    local tag="$1"
    _state_init
    python3 -c "
import json, sys
try:
    with open('$STATEFILE') as f: s = json.load(f)
    d = s.get('output_dir', {}).get('$tag', '')
    if d: print(d)
except: pass
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
P35_OUT_BASE="${P35_OUT_BASE:-out/sweep_p35_ablation}"

# ── Common flags for remixed arms (same as p29_sweep.sh REMIX_COMMON) ────────
REMIX_COMMON="--fp8 --max-shards ${MAX_SHARDS:-170} --models remixed-linear \
  --device-batch-size ${REMIX_DEVICE_BATCH_SIZE:-${DEVICE_BATCH_SIZE:-16}} --total-batch-size -1 --use-onecycle 0 --log-every ${LOG_EVERY:-200} --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 --aspect-ratio $ASPECT_RATIO \
  --target-param-data-ratio 20 \
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
  --p23-quantile-route 1 \
  --target-flops $TARGET_FLOPS"

# ── Common flags for dense arms ──────────────────────────────────────────────
BASE_COMMON="--fp8 --max-shards ${MAX_SHARDS:-170} --models base \
  --device-batch-size ${BASE_DEVICE_BATCH_SIZE:-${DEVICE_BATCH_SIZE:-128}} --total-batch-size -1 --use-onecycle 0 --log-every ${LOG_EVERY:-200} --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 \
  --target-param-data-ratio 20 \
  --warmup-ratio 0.005 \
  --warmdown-ratio 0.65 \
  --final-lr-frac 0.05 \
  --research-dim -1 \
  --target-tokens -1 \
  --target-active-params 0 \
  --save-every 200 \
  --target-flops $TARGET_FLOPS"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Phase 35: RemixedLinear Ablation Sweep (D${DEPTH})          ║"
echo "║     Target FLOPs: ${TARGET_FLOPS}                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ══════════════════════════════════════════════════════
# ARM A: K=1 Single Template (No Routing)
#   - RemixedLinear with n_templates=1
#   - This makes it y = W_m @ LN(W_b @ x) + bias + output_gate
#   - Isolates: is the 2-matmul factorization + LN the problem?
#   - If A ≈ full RemixedLinear → routing adds nothing
#   - If A ≪ dense → factorization itself is the problem
# ══════════════════════════════════════════════════════
TAG="35A_K1_NO_ROUTE_D${DEPTH}"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "35A" "$TAG" "K=1 single template — no routing, pure factorized linear"
    _SAVED=$(get_out_dir "$TAG")
    _RUN_DIR="${_SAVED:-${P35_OUT_BASE}/${TAG}}"
    if [[ "$FORCE" == 1 ]] && [[ -d "$_RUN_DIR" ]]; then
        echo "🗑  --force: removing old run directory: $_RUN_DIR"
        rm -rf "$_RUN_DIR"
    fi
    mark_started "$TAG" "${_RUN_DIR}/depth_${DEPTH}/ckpt_remixed-linear/remixed-linear" "$_RUN_DIR"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --out-dir "$_RUN_DIR" \
      --p22-n-templates 1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $TAG done"
        mark_completed "$TAG"
    else
        echo "❌  $TAG FAILED — will retry next run"
    fi
fi

# ══════════════════════════════════════════════════════
# ARM B: No Context / No Gates
#   - RemixedLinear with 8 templates, chunk routing, but NO context
#     stream, NO basis gate, NO output gate
#   - This makes it y = W_eff @ LN(W_b @ x) + bias (pure factored + route)
#   - Isolates: how much FLOP budget does the gate infrastructure waste?
#   - If B > full RemixedLinear → gates actively hurt
#   - If B ≈ full RemixedLinear → gates are useless overhead
#   - If B < full RemixedLinear → gates help, problem is elsewhere
# ══════════════════════════════════════════════════════
TAG="35B_NO_GATES_8T_D${DEPTH}"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "35B" "$TAG" "8T chunk routing, NO context/gates — pure routing overhead"
    _SAVED=$(get_out_dir "$TAG")
    _RUN_DIR="${_SAVED:-${P35_OUT_BASE}/${TAG}}"
    if [[ "$FORCE" == 1 ]] && [[ -d "$_RUN_DIR" ]]; then
        echo "🗑  --force: removing old run directory: $_RUN_DIR"
        rm -rf "$_RUN_DIR"
    fi
    mark_started "$TAG" "${_RUN_DIR}/depth_${DEPTH}/ckpt_remixed-linear/remixed-linear" "$_RUN_DIR"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --out-dir "$_RUN_DIR" \
      --p22-n-templates 8 --p23-quantile-route 0 --p31-template-delta-rank 0 \
      --p28-chunk-routing-size 256 --p22-template-topk 0 --p31-drop-basis-proj 1 \
      --p31-route-side narrow --p31-basis-side-templates -1 \
      --remix-use-context 0 --remix-use-output-gate 0 --remix-use-basis-gate 0 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $TAG done"
        mark_completed "$TAG"
    else
        echo "❌  $TAG FAILED — will retry next run"
    fi
fi

# ══════════════════════════════════════════════════════
# ARM C: No Intermediate LayerNorm
#   - RemixedLinear with 8 templates, chunk routing, full gates,
#     but the LN between W_b and W_m is removed
#   - Isolates: does LN(W_b·x) destroy magnitude information?
#   - If C > full RemixedLinear → LN is actively harmful
#   - If C ≈ full RemixedLinear → LN is neutral
# ══════════════════════════════════════════════════════
TAG="35C_NO_LN_8T_D${DEPTH}"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "35C" "$TAG" "8T chunk routing, NO intermediate LN — test LN impact"
    _SAVED=$(get_out_dir "$TAG")
    _RUN_DIR="${_SAVED:-${P35_OUT_BASE}/${TAG}}"
    if [[ "$FORCE" == 1 ]] && [[ -d "$_RUN_DIR" ]]; then
        echo "🗑  --force: removing old run directory: $_RUN_DIR"
        rm -rf "$_RUN_DIR"
    fi
    mark_started "$TAG" "${_RUN_DIR}/depth_${DEPTH}/ckpt_remixed-linear/remixed-linear" "$_RUN_DIR"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --out-dir "$_RUN_DIR" \
      --p22-n-templates 8 --p23-quantile-route 0 --p31-template-delta-rank 0 \
      --p28-chunk-routing-size 256 --p22-template-topk 0 --p31-drop-basis-proj 1 \
      --p31-route-side narrow --p31-basis-side-templates -1 \
      --remix-disable-ln-basis 1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $TAG done"
        mark_completed "$TAG"
    else
        echo "❌  $TAG FAILED — will retry next run"
    fi
fi

# ══════════════════════════════════════════════════════
# ARM D: Dense Baseline + Intermediate LN
#   - Standard dense transformer, but with a LayerNorm added
#     between c_fc and the activation (mirrors RemixedLinear's LN)
#   - Isolates: does intermediate LN degrade ANY architecture?
#   - If D < plain dense → LN itself is the problem
#   - If D ≈ plain dense → LN is fine, RemixedLinear has other issues
# ══════════════════════════════════════════════════════
TAG="35D_DENSE_WITH_LN_D${DEPTH}"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "35D" "$TAG" "Dense baseline + intermediate LN — LN control experiment"
    _SAVED=$(get_out_dir "$TAG")
    _RUN_DIR="${_SAVED:-${P35_OUT_BASE}/${TAG}}"
    if [[ "$FORCE" == 1 ]] && [[ -d "$_RUN_DIR" ]]; then
        echo "🗑  --force: removing old run directory: $_RUN_DIR"
        rm -rf "$_RUN_DIR"
    fi
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
# ARM E: Full 29C RemixedLinear (Reference)
#   - Exact same config as the 29C sweep, but FLOP-matched
#   - Provides the direct comparison point for arms A-D
# ══════════════════════════════════════════════════════
TAG="35E_FULL_REMIX_8T_D${DEPTH}"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "35E" "$TAG" "Full 29C RemixedLinear — FLOP-matched reference"
    _SAVED=$(get_out_dir "$TAG")
    _RUN_DIR="${_SAVED:-${P35_OUT_BASE}/${TAG}}"
    if [[ "$FORCE" == 1 ]] && [[ -d "$_RUN_DIR" ]]; then
        echo "🗑  --force: removing old run directory: $_RUN_DIR"
        rm -rf "$_RUN_DIR"
    fi
    mark_started "$TAG" "${_RUN_DIR}/depth_${DEPTH}/ckpt_remixed-linear/remixed-linear" "$_RUN_DIR"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --out-dir "$_RUN_DIR" \
      --p22-n-templates 8 --p23-quantile-route 0 --p31-template-delta-rank 0 \
      --p28-chunk-routing-size 256 --p22-template-topk 0 --p31-drop-basis-proj 1 \
      --p31-route-side narrow --p31-basis-side-templates -1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $TAG done"
        mark_completed "$TAG"
    else
        echo "❌  $TAG FAILED — will retry next run"
    fi
fi


echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Phase 35 Ablation Sweep Complete                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Results summary — compare final val BPB across:"
echo "  35A: K=1 no routing      → factorization overhead"
echo "  35B: No gates/context    → gate overhead"
echo "  35C: No intermediate LN  → LN harm"
echo "  35D: Dense + LN          → LN control"
echo "  35E: Full RemixedLinear  → reference"
echo "  Dense D8 baseline        → 1.058 BPB (from previous run)"
