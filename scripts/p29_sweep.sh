#!/bin/bash

set -o pipefail

LOGFILE="${SWEEP_LOG:-sweep_p29.log}"
STATEFILE="${LOGFILE%.log}_state.json"
FORCE=0
if [[ "$1" == "--force" ]]; then
    FORCE=1
    rm -f "$STATEFILE"
    shift
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

DEPTH=8
MODEL_DIM=$(python3 -c "d=$DEPTH; h=128; print(((d*64+h-1)//h)*h)")
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
REMIX_COMMON="--fp8 --max-shards 170 --models remixed-linear \
  --device-batch-size 32 --total-batch-size 262144 --use-onecycle 0 --log-every 200 --skip-core \
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
  --p22-template-routing-learned 1 \
  --remix-use-basis-gate 0 \
  --remix-use-output-gate 1 \
  --remix-basis-gate-mode centered \
  --target-tokens -1 \
  --target-active-params 0 \
  --save-every 200 \
  --p23-quantile-route 1"


# ══════════════════════════════════════════════════════
# 29A: 8T Top-1 (Full Rank Baseline)
#   - 8 templates, hard top-1 routing per token
#   - Basis size = MODEL_DIM (Full rank)
#   - Token budget dynamically scaled by active params
# ══════════════════════════════════════════════════════
TAG="29A_8T_TOP1_BASELINE_D8"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "29A" "$TAG" "8T top-1 sparse routing (Full rank baseline)"
    # Use stored output_dir if we've run this tag before; otherwise use default.
    _SAVED=$(get_out_dir "$TAG")
    _RUN_DIR="${_SAVED:-${P29_OUT_BASE}/${TAG}}"
    mark_started "$TAG" "${_RUN_DIR}/depth_${DEPTH}/ckpt_remixed-linear/remixed-linear" "$_RUN_DIR"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --out-dir "$_RUN_DIR" \
      --p22-n-templates 8 \
      --p22-template-topk 1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $TAG done"
        mark_completed "$TAG"
    else
        echo "❌  $TAG FAILED — will retry next run"
    fi
fi


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


# ══════════════════════════════════════════════════════
# 29C: Chunk Routing N=64 (Full Rank Baseline)
#   - Soft routing over 8 templates, amortized over 64 tokens
#   - Basis size = MODEL_DIM (Full rank)
# ══════════════════════════════════════════════════════
TAG="29C_CHUNK64_BASELINE_D8"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "29C" "$TAG" "Chunk routing N=64 (Full rank baseline)"
    _SAVED=$(get_out_dir "$TAG")
    _RUN_DIR="${_SAVED:-${P29_OUT_BASE}/${TAG}}"
    mark_started "$TAG" "${_RUN_DIR}/depth_${DEPTH}/ckpt_remixed-linear/remixed-linear" "$_RUN_DIR"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --out-dir "$_RUN_DIR" \
      --p22-n-templates 8 \
      --p28-chunk-routing-size 64 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $TAG done"
        mark_completed "$TAG"
    else
        echo "❌  $TAG FAILED — will retry next run"
    fi
fi


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
TAG="29E_8T_TOP1_CHUNK64_D8"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "29E" "$TAG" "Combining Top-1 sparse routing AND Chunk N=64 routing"
    _SAVED=$(get_out_dir "$TAG")
    _RUN_DIR="${_SAVED:-${P29_OUT_BASE}/${TAG}}"
    mark_started "$TAG" "${_RUN_DIR}/depth_${DEPTH}/ckpt_remixed-linear/remixed-linear" "$_RUN_DIR"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --out-dir "$_RUN_DIR" \
      --p22-n-templates 8 \
      --p28-chunk-routing-size 64 \
      --p22-template-topk 1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $TAG done"
        mark_completed "$TAG"
    else
        echo "❌  $TAG FAILED — will retry next run"
    fi
fi



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
#  --warmdown-ratio 0.50 \
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

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Phase 29 Sweep Complete                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
