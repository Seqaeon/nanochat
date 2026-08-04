#!/bin/bash

set -o pipefail

# ── Startup acceleration ───────────────────────────────────────────────────────
# Cache torch.compile (Inductor) kernels in a stable dir on the Modal volume.
# First run still compiles; every subsequent run reuses the cache.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-out/.triton_cache}"
# Also cache the Dynamo FX graph (the Python graph capture step).
export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}"
# NOTE: HF_HUB_OFFLINE is intentionally NOT set here — research_sweep.sh has
# set -euo pipefail, and any HF Hub call during the setup phase (report reset,
# nanochat.report imports) would raise OfflineModeIsEnabled and kill the script.
# ──────────────────────────────────────────────────────────────────────────────

# Parse --force
FORCE=0
if [[ "${1:-}" == "--force" ]]; then FORCE=1; shift; fi

# Collect all numeric positional args as depths
DEPTHS=()
while [[ -n "${1:-}" && "$1" =~ ^[0-9]+$ ]]; do
    DEPTHS+=("$1"); shift
done
[[ ${#DEPTHS[@]} -eq 0 ]] && DEPTHS=("${DEPTH:-8}")

# Multi-depth: re-invoke self for each depth sequentially.
# Each depth gets its own log + state file; all other env vars are inherited.
if [[ ${#DEPTHS[@]} -gt 1 ]]; then
    echo "P29 multi-depth sweep: ${DEPTHS[*]}"
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

# Log and state files auto-named per depth unless SWEEP_LOG is set explicitly.
LOGFILE="${SWEEP_LOG:-sweep_p29_d${DEPTH}.log}"
STATEFILE="${LOGFILE%.log}_state.json"

if [[ "$FORCE" == 1 ]]; then
    rm -f "$STATEFILE"
fi

## ---------------------------------------------------------------------------
# JSON state helpers — state file format:
# {
#   "completed":  ["TAG1", "TAG2", ...],
#   "unfinished": { "TAG": "/path/to/ckpt_dir" },
#   "output_dir": { "TAG": "/path/to/run_dir" }
# }
# All writes are atomic (write tmp → rename) to survive crashes mid-write.
# All reads recover gracefully from corrupt JSON.
# ---------------------------------------------------------------------------
_EMPTY_STATE='{"completed":[],"unfinished":{},"output_dir":{}}'

_state_init() {
    if [[ ! -f "$STATEFILE" ]]; then
        echo "$_EMPTY_STATE" > "${STATEFILE}.tmp" && mv "${STATEFILE}.tmp" "$STATEFILE"
    fi
}

# _state_read: prints the state JSON to stdout, recovering from corrupt files.
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

# Call BEFORE launching an experiment so a crash mid-run is tracked
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

# Call on success — moves tag from unfinished → completed
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

# Read the output_dir stored by mark_started for a given tag.
# Returns the stored path if found, or empty string if not.
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
MODEL_DIM_C4=$(( MODEL_DIM / 4 ))
MODEL_DIM_C2=$(( MODEL_DIM / 2 ))

CCL_MOD="${CCL_MOD:-weight}"
CCL_STREAM="${CCL_STREAM:-selective}"
# Default root for output dirs on first run of each tag.
# On re-runs, get_out_dir reads the actual path from the state file instead.
P29_OUT_BASE="${P29_OUT_BASE:-out/sweep_p29}"

# Common flags shared by all variants.
# Notes:
#   --target-active-params 1  → sparse variants get token budget = ratio × active_params
#   --p22-template-routing-learned 1 → learned (gradient-driven) routing weights
REMIX_COMMON="--fp8 --max-shards ${MAX_SHARDS:-170} --models remixed-linear \
  --device-batch-size ${REMIX_DEVICE_BATCH_SIZE:-${DEVICE_BATCH_SIZE:-16}} --total-batch-size 524288 --use-onecycle 0 --log-every ${LOG_EVERY:-200} --skip-core \
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
  --target-active-params 1 \
  --save-every 200 \
  --p23-quantile-route 1"

# Common flags for dense baseline and Standard MoE experiments.
# Uses --models base (standard transformer, no remix-linear).
# Higher device-batch-size (128) since there is no MoE/remix overhead.
BASE_COMMON="--fp8 --max-shards ${MAX_SHARDS:-170} --models base \
  --device-batch-size ${BASE_DEVICE_BATCH_SIZE:-${DEVICE_BATCH_SIZE:-128}} --total-batch-size 524288 --use-onecycle 0 --log-every ${LOG_EVERY:-200} --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 \
  --target-param-data-ratio 10.5 \
  --warmup-ratio 0.005 \
  --warmdown-ratio 0.65 \
  --final-lr-frac 0.05 \
  --research-dim -1 \
  --target-tokens -1 \
  --target-active-params 1 \
  --save-every 200"


# ══════════════════════════════════════════════════════
# 29A: 8T Top-1 (Full Rank Baseline)
#   - 8 templates, hard top-1 routing per token
#   - Basis size = MODEL_DIM (Full rank)
#   - Token budget dynamically scaled by active params
# ══════════════════════════════════════════════════════
#TAG="29A_8T_TOP1_BASELINE_D${DEPTH}"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "29A" "$TAG" "8T top-1 sparse routing (Full rank baseline)"
#    # Use stored output_dir if we've run this tag before; otherwise use default.
#    _SAVED=$(get_out_dir "$TAG")
#    _RUN_DIR="${_SAVED:-${P29_OUT_BASE}/${TAG}}"
#    mark_started "$TAG" "${_RUN_DIR}/depth_${DEPTH}/ckpt_remixed-linear/remixed-linear" "$_RUN_DIR"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --out-dir "$_RUN_DIR" \
#      --p22-n-templates 8 \
#      --p22-template-topk 1 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "✅  $TAG done"
#        mark_completed "$TAG"
#    else
#        echo "❌  $TAG FAILED — will retry next run"
#    fi
#fi
#

# ══════════════════════════════════════════════════════
# 29B: 8T Top-1 (C//4 Compressed Basis)
#   - 8 templates, hard top-1 routing per token
#   - Basis size = MODEL_DIM // 4 (testing basis compression)
# ══════════════════════════════════════════════════════
#TAG="29B_8T_TOP1_C4"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "29B" "$TAG" "8T top-1 sparse routing with C//4 basis compression"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --p22-n-templates 8 \
#      --p22-template-topk 1 \
#      --remix-basis-size $MODEL_DIM_C4 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "✅  $TAG done"
#        mark_completed "$TAG"
#    else
#        echo "❌  $TAG FAILED — will retry next run"
#    fi
#fi
#



#      --remix-basis-size $MODEL_DIM_C4 \
# ══════════════════════════════════════════════════════
# 29C: Chunk Routing N=64 (Full Rank Baseline)
#   - Soft routing over 8 templates, amortized over 64 tokens
#   - Basis size = MODEL_DIM (Full rank)
# ══════════════════════════════════════════════════════
TAG="29C_CHUNK64_BASELINE_8T_D${DEPTH}"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "29C" "$TAG" "Chunk routing N=64 (Full rank baseline)"
    _SAVED=$(get_out_dir "$TAG")
    _RUN_DIR="${_SAVED:-${P29_OUT_BASE}/${TAG}}"
    if [[ "$FORCE" == 1 ]] && [[ -d "$_RUN_DIR" ]]; then
        echo "🗑  --force: removing old run directory: $_RUN_DIR"
        rm -rf "$_RUN_DIR"
    fi
    mark_started "$TAG" "${_RUN_DIR}/depth_${DEPTH}/ckpt_remixed-linear/remixed-linear" "$_RUN_DIR"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --out-dir "$_RUN_DIR" \
      --p22-n-templates 8 --p23-quantile-route 1 --p31-template-delta-rank 0 \
      --p28-chunk-routing-size 512 --p22-template-topk 0 --p31-drop-basis-proj 1 --p29-grad-equalize 1 \
      --p31-route-side narrow --p31-basis-side-templates -1 --remix-template-lr-scale 1.0 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $TAG done"
        mark_completed "$TAG"
    else
        echo "❌  $TAG FAILED — will retry next run"
    fi
fi

# ══════════════════════════════════════════════════════
# 29C-G: chunk routing, HARD TOP-1, grouped fast path
#
# A DIFFERENT EXPERIMENT from 29C above, not a faster version of it:
# --p22-template-topk 1 selects one template per chunk instead of mixing all 8,
# so BPB is not comparable to the soft baseline. Separate TAG accordingly.
#
#   --p22-template-topk 1        one template per chunk => at most K distinct
#                                W_eff exist, so they never need materializing
#   --p31-chunk-route-impl grouped
#                                permute chunks by template, K dense GEMMs
#                                against the bank. Declines and falls back to
#                                compose for anything that is not top-1.
#   --p31-top1-gate switch       coefficient from the full softmax. WITHOUT IT
#                                THE ROUTER DOES NOT TRAIN: softmax over one
#                                unmasked logit is the constant 1.0, so the
#                                gradient to template_route is exactly zero.
#
# WHEN THIS PAYS. Compose materializes out*basis per chunk, i.e. out*(basis/chunk)
# per token; grouped's overhead is two gathers and is independent of chunk. So the
# ratio basis/chunk decides it. Compiled single-projection measurements:
#     basis/chunk = 1  (d4,  chunk 256)   0.67x   grouped LOSES
#     basis/chunk = 3  (d12, chunk 256)   1.01x   break-even
#     basis/chunk = 12 (d12, chunk 64)    2.25x   attn
#     basis/chunk = 12 (d12 c_fc, chunk 64) 3.65x
# At d4 with chunk 256, basis == chunk == MODEL_DIM, which is the worst case and
# is expected to be slower. The informative run is d12+, or chunk 64 where
# grouped costs the same as chunk 256 does (measured 15.06ms/67.1M vs
# 15.29ms/66.9M at d12) while giving 4x finer routing.
#
# Set P29G_CHUNK to override the chunk size for this arm alone.
# ══════════════════════════════════════════════════════
#TAG="29CG_GROUPED_TOP1_8T_N${P29G_CHUNK:-256}_D${DEPTH}"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "29C-G" "$TAG" "Hard top-1, grouped fast path (N=${P29G_CHUNK:-256})"
#    _SAVED=$(get_out_dir "$TAG")
#    _RUN_DIR="${_SAVED:-${P29_OUT_BASE}/${TAG}}"
#    if [[ "$FORCE" == 1 ]] && [[ -d "$_RUN_DIR" ]]; then
#        echo "🗑  --force: removing old run directory: $_RUN_DIR"
#        rm -rf "$_RUN_DIR"
#    fi
#    mark_started "$TAG" "${_RUN_DIR}/depth_${DEPTH}/ckpt_remixed-linear/remixed-linear" "$_RUN_DIR"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --out-dir "$_RUN_DIR" \
#      --p22-n-templates 8 --p23-quantile-route 0 --p31-template-delta-rank 0 \
#      --p28-chunk-routing-size ${P29G_CHUNK:-256} --p22-template-topk 1 \
#      --p31-chunk-route-impl grouped --p31-top1-gate switch \
#      --p31-drop-basis-proj 1 \
#      --p31-route-side narrow --p31-basis-side-templates -1 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "✅  $TAG done"
#        mark_completed "$TAG"
#    else
#        echo "❌  $TAG FAILED — will retry next run"
#    fi
#fi

# ══════════════════════════════════════════════════════
# P34: does per-token weight mixing replace the FFN's 4x expansion?
#
# The dense FFN is D -> 4D -> D. The expansion exists because a *static* matrix
# has to serve every context with one set of weights, so it buys capacity with
# width. If a per-token effective operator supplies that capacity instead, the
# expansion is redundant and the FFN gets much cheaper. Nothing in the paper
# tests this: every RemixedLinear result so far keeps the 4x shape and remixes
# all six projections, which is the most expensive possible configuration.
#
# Four arms, chosen so each isolates one variable against the 29C baseline:
#   A  remix FFN only, 4x shape      -> does remixing ATTENTION earn its cost?
#   B  remix FFN only, D->D->D       -> does the 4x expansion earn its cost?
#   C  remix FFN only, single D->D   -> is one remixed layer the whole FFN?
#   D  remix everything, D->D->D     -> is the expansion redundant even with
#                                       attention remixed (separates the two)
#
# Projected at d12 against the dense baseline (7.385e8 active FLOPs, 286.3M params):
#   current 29C (all six, 4x)  779.3M params  1.065e9 active FLOPs  1.44x dense
#   A  remix FFN only, 4x      551.7M         8.694e8               1.18x
#   D  all six, D->D->D        587.5M         7.917e8               1.07x
#   B  FFN only, D->D->D       360.0M         5.958e8               0.81x
#   C  FFN only, single D->D   296.0M         5.044e8               0.68x
#
# C is the interesting one: 3% more total parameters than dense, 16% fewer active
# parameters, 32% fewer active FLOPs, and one remixed projection per block instead
# of six, which cuts the W_eff bandwidth that dominates the runtime by the same
# factor. If quality holds anywhere near the 29C level, the claim stops being
# "better at matched FLOPs" (which the AC rejects, since FLOPs do not predict
# wall-clock here) and becomes "better with fewer FLOPs and fewer active params",
# which needs no FLOP-accounting argument at all.
#
# Run at d4 first; it is hours, not days, and it decides whether to go further.
# ══════════════════════════════════════════════════════
#P34_ARMS=(
#  "A_FFNONLY_4X|--p34-dense-attn 1"
#  "B_FFNONLY_1X|--p34-dense-attn 1 --p34-ffn-mult 1.0"
#  "C_FFNONLY_SINGLE|--p34-dense-attn 1 --p34-ffn-single 1"
#  "D_ALL_1X|--p34-ffn-mult 1.0"
#)
#for _arm in "${P34_ARMS[@]}"; do
#    _name="${_arm%%|*}"
#    _flags="${_arm#*|}"
#    TAG="34${_name}_8T_N${P34_CHUNK:-256}_D${DEPTH}"
#    if check_completed "$TAG"; then
#        echo "⏭  Skipping $TAG (already completed)"
#        continue
#    fi
#    print_header "P34" "$TAG" "FFN-shape study: ${_flags}"
#    _SAVED=$(get_out_dir "$TAG")
#    _RUN_DIR="${_SAVED:-${P29_OUT_BASE}/${TAG}}"
#    if [[ "$FORCE" == 1 ]] && [[ -d "$_RUN_DIR" ]]; then
#        echo "🗑  --force: removing old run directory: $_RUN_DIR"
#        rm -rf "$_RUN_DIR"
#    fi
#    mark_started "$TAG" "${_RUN_DIR}/depth_${DEPTH}/ckpt_remixed-linear/remixed-linear" "$_RUN_DIR"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --out-dir "$_RUN_DIR" \
#      --p22-n-templates 8 --p23-quantile-route 0 --p31-template-delta-rank 0 \
#      --p28-chunk-routing-size ${P34_CHUNK:-256} --p22-template-topk 0 \
#      $_flags \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "✅  $TAG done"
#        mark_completed "$TAG"
#    else
#        echo "❌  $TAG FAILED — will retry next run"
#    fi
#done




# ══════════════════════════════════════════════════════
# 29C: Chunk Routing N=64 (Full Rank Baseline)
#   - Soft routing over 8 templates, amortized over 64 tokens
#   - Basis size = MODEL_DIM (Full rank)
# ══════════════════════════════════════════════════════
#TAG="29C_CHUNK64_BASELINE_8T_D${DEPTH}"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "29C" "$TAG" "Chunk routing N=64 (Full rank baseline)"
#    _SAVED=$(get_out_dir "$TAG")
#    _RUN_DIR="${_SAVED:-${P29_OUT_BASE}/${TAG}}"
#    if [[ "$FORCE" == 1 ]] && [[ -d "$_RUN_DIR" ]]; then
#        echo "🗑  --force: removing old run directory: $_RUN_DIR"
#        rm -rf "$_RUN_DIR"
#    fi
#    mark_started "$TAG" "${_RUN_DIR}/depth_${DEPTH}/ckpt_remixed-linear/remixed-linear" "$_RUN_DIR"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --out-dir "$_RUN_DIR" \
#      --p22-n-templates 8 --p23-quantile-route 0 --p31-template-delta-rank 0 \
#      --p28-chunk-routing-size 256 --p22-template-topk 0 --p31-drop-basis-proj 1 \
#      --p31-route-side narrow --p31-basis-side-templates -1 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "✅  $TAG done"
#        mark_completed "$TAG"
#    else
#        echo "❌  $TAG FAILED — will retry next run"
#    fi
#fi


# ══════════════════════════════════════════════════════
# 29C: Chunk Routing N=64 (Full Rank Baseline)
#   - Soft routing over 8 templates, amortized over 64 tokens
#   - Basis size = MODEL_DIM (Full rank)
# ══════════════════════════════════════════════════════
#TAG="29C_CHUNK64_BASELINE_8T_AS32_D${DEPTH}"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "29C" "$TAG" "Chunk routing N=64 (Full rank baseline)"
#    _SAVED=$(get_out_dir "$TAG")
#    _RUN_DIR="${_SAVED:-${P29_OUT_BASE}/${TAG}}"
#    mark_started "$TAG" "${_RUN_DIR}/depth_${DEPTH}/ckpt_remixed-linear/remixed-linear" "$_RUN_DIR"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --out-dir "$_RUN_DIR" \
#      --p22-n-templates 8 \
#      --p28-chunk-routing-size 64 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "✅  $TAG done"
#        mark_completed "$TAG"
#    else
#        echo "❌  $TAG FAILED — will retry next run"
#    fi
#fi

# ══════════════════════════════════════════════════════
# 29D: Chunk Routing N=64 (C//4 Compressed Basis)
#   - Soft routing over 8 templates, amortized over 64 tokens
#   - Basis size = MODEL_DIM // 4 (testing basis compression)
# ══════════════════════════════════════════════════════
#TAG="29D_CHUNK64_C4"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "29D" "$TAG" "Chunk routing N=64 with C//4 basis compression"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --p22-n-templates 8 \
#      --p28-chunk-routing-size 64 \
#      --remix-basis-size $MODEL_DIM_C4 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "✅  $TAG done"
#        mark_completed "$TAG"
#    else
#        echo "❌  $TAG FAILED — will retry next run"
#    fi
#fi
#
# ══════════════════════════════════════════════════════
# 29E: Top-1 AND Chunk Routing N=64 combined
#   - 8 templates, hard top-1 routing BUT amortized over 64 tokens
#   - Tests if picking 1 expert per chunk works as well as soft-mixing
# ══════════════════════════════════════════════════════
#TAG="29E_8T_TOP1_CHUNK64_D${DEPTH}"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "29E" "$TAG" "Combining Top-1 sparse routing AND Chunk N=64 routing"
#    _SAVED=$(get_out_dir "$TAG")
#    _RUN_DIR="${_SAVED:-${P29_OUT_BASE}/${TAG}}"
#    mark_started "$TAG" "${_RUN_DIR}/depth_${DEPTH}/ckpt_remixed-linear/remixed-linear" "$_RUN_DIR"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --out-dir "$_RUN_DIR" \
#      --p22-n-templates 8 \
#      --p28-chunk-routing-size 64 \
#      --p22-template-topk 1 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "✅  $TAG done"
#        mark_completed "$TAG"
#    else
#        echo "❌  $TAG FAILED — will retry next run"
#    fi
#fi

# ══════════════════════════════════════════════════════
# 29E2: Top-1 AND Chunk Routing N=256 combined
#   - 8 templates, hard top-1 routing BUT amortized over 256 tokens
#   - Tests if picking 1 expert per chunk works as well as soft-mixing
# ══════════════════════════════════════════════════════
#TAG="29E2_8T_TOP1_CHUNK256_D${DEPTH}"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "29E" "$TAG" "Combining Top-1 sparse routing AND Chunk N=64 routing"
#    _SAVED=$(get_out_dir "$TAG")
#    _RUN_DIR="${_SAVED:-${P29_OUT_BASE}/${TAG}}"
#    mark_started "$TAG" "${_RUN_DIR}/depth_${DEPTH}/ckpt_remixed-linear/remixed-linear" "$_RUN_DIR"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --out-dir "$_RUN_DIR" \
#      --p22-n-templates 8 \
#      --p28-chunk-routing-size 256 \
#      --p22-template-topk 1 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "✅  $TAG done"
#        mark_completed "$TAG"
#    else
#        echo "❌  $TAG FAILED — will retry next run"
#    fi
#fi
#

# ══════════════════════════════════════════════════════
# 29F: Dense Mixture 8T (C//4 Compressed Basis)
#   - 8 learned templates, fully dense mixture (no top-k or chunking)
#   - Tests if dense mixture survives aggressive C//4 basis compression
# ══════════════════════════════════════════════════════
#TAG="29F_8T_DENSE_C4"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "29F" "$TAG" "8T Dense mixture with C//4 basis compression"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --p22-n-templates 8 \
#      --remix-basis-size $MODEL_DIM_C4 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "✅  $TAG done"
#        mark_completed "$TAG"
#    else
#        echo "❌  $TAG FAILED — will retry next run"
#    fi
#fi
#
#
# ══════════════════════════════════════════════════════
# 29G: Dense Mixture 4T (C//4 Compressed Basis)
#   - 4 learned templates, fully dense mixture
#   - Basis size = MODEL_DIM // 4
# ══════════════════════════════════════════════════════
#TAG="29G_4T_DENSE_C4"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "29G" "$TAG" "4T Dense mixture with C//4 basis compression"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --p22-n-templates 4 \
#      --remix-basis-size $MODEL_DIM_C4 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "✅  $TAG done"
#        mark_completed "$TAG"
#    else
#        echo "❌  $TAG FAILED — will retry next run"
#    fi
#fi
#
# ══════════════════════════════════════════════════════
# 29H: Dense Mixture 4T (C//2 Compressed Basis)
#   - 4 learned templates, fully dense mixture
#   - Basis size = MODEL_DIM // 2
# ══════════════════════════════════════════════════════
#TAG="29H_4T_DENSE_C2"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "29H" "$TAG" "4T Dense mixture with C//2 basis compression"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --p22-n-templates 4 \
#      --remix-basis-size $MODEL_DIM_C2 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "✅  $TAG done"
#        mark_completed "$TAG"
#    else
#        echo "❌  $TAG FAILED — will retry next run"
#    fi
#fi
#
#BASE_COMMON="--fp8 --max-shards 170 --models base \
#  --device-batch-size 128 --total-batch-size 262144 --use-onecycle 0 --log-every 200 --skip-core \
#  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
#  --sequence-len 2048 \
#  --warmup-ratio 0.20 \
#  --warmdown-ratio 0.65 \
#  --research-dim -1 \
#  --target-tokens -1 \
#  --p23-quantile-route 1 \
#  --target-active-params 0"
#
# ══════════════════════════════════════════════════════
# 29I: Standard MoE baseline — K=8 full-size experts, top-1 routing
# ══════════════════════════════════════════════════════
#TAG="29I_STD_MOE_TOP1"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "29I" "$TAG" "StandardMoE K=8 full-size experts, top-1 routing (baseline)"
#    if bash scripts/research_sweep.sh $BASE_COMMON \
#      --p23-std-moe-experts 8 \
#      --p23-std-moe-topk 1 \
#      --p23-std-moe-aux-weight 0.01 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "✅  $TAG done"
#        mark_completed "$TAG"
#    else
#        echo "❌  $TAG FAILED — will retry next run"
#    fi
#fi
#
# ══════════════════════════════════════════════════════
# 29J: Standard MoE baseline — K=8 full-size experts, top-optimal routing
# ══════════════════════════════════════════════════════
#TAG="29J_STD_MOE_TOP_OPT"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "29J" "$TAG" "StandardMoE K=8 full-size experts, top-optimal routing (baseline)"
#    if bash scripts/research_sweep.sh $BASE_COMMON \
#      --p23-std-moe-experts 8 \
#      --p23-std-moe-topk 8 \
#      --p23-std-moe-aux-weight 0.01 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "✅  $TAG done"
#        mark_completed "$TAG"
#    else
#        echo "❌  $TAG FAILED — will retry next run"
#    fi
#fi

# ══════════════════════════════════════════════════════
# 29BASE: Dense transformer baseline
#   - Standard transformer (no MoE, no remix-linear)
#   - Chinchilla-optimal token budget from total params
#   - Provides reference curve for all other variants
# ══════════════════════════════════════════════════════
#TAG="29BASE_DENSE_D${DEPTH}"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "29BASE" "$TAG" "Dense baseline — standard transformer (depth ${DEPTH})"
#    _SAVED=$(get_out_dir "$TAG")
#    _RUN_DIR="${_SAVED:-${P29_OUT_BASE}/${TAG}}"
#    mark_started "$TAG" "${_RUN_DIR}/depth_${DEPTH}/ckpt_base/base" "$_RUN_DIR"
#    if bash scripts/research_sweep.sh $BASE_COMMON \
#      --out-dir "$_RUN_DIR" \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "✅  $TAG done"
#        mark_completed "$TAG"
#    else
#        echo "❌  $TAG FAILED — will retry next run"
#    fi
#fi
#
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Phase 29 Sweep Complete                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
