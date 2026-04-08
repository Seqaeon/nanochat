#!/bin/bash
# scripts/warmup_sweep.sh
#
# Warmup-ratio sweep for CCL research model variants across multiple depths.
# Learning rates are fixed to LR-sweep winners (configured in warmup_sweep.py).
# Each (model, warmup_frac) pair is trained for a short 100M token run.
#
# Usage:
#   bash scripts/warmup_sweep.sh [flags] <depth1> [depth2] ...
#
# Examples:
#   bash scripts/warmup_sweep.sh --fp8 --max-shards 170 8 16
#   bash scripts/warmup_sweep.sh --fp8 --max-shards 170 --target-tokens 100000000 8
#   bash scripts/warmup_sweep.sh --fp8 --max-shards 170 --warmup-fracs "0.0 0.01 0.05 0.10" 8
#   bash scripts/warmup_sweep.sh --models "moe_perm remixed-linear" 8 16
#
# Note: --target-tokens is the FULL training budget (e.g. 20B) used for warmup step
# calculation. --early-stop-tokens (or legacy --run-tokens) controls per-run early stop.
#
# Env-var overrides:
#   NPROC_PER_NODE      — DDP worker count (default: 8, auto-capped to GPU count)
#   NANOCHAT_SKIP_ENV_SETUP=1  — skip uv/venv setup if already activated

export OMP_NUM_THREADS=8

# ── GPU count guard ───────────────────────────────────────────────────────────
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
if [[ "${AVAILABLE_GPUS}" =~ ^[0-9]+$ ]] && [ "$AVAILABLE_GPUS" -gt 0 ] && [ "$NPROC_PER_NODE" -gt "$AVAILABLE_GPUS" ]; then
    echo "Requested NPROC_PER_NODE=${NPROC_PER_NODE}, but only ${AVAILABLE_GPUS} GPU(s) visible. Capping."
    NPROC_PER_NODE="${AVAILABLE_GPUS}"
fi
export NPROC_PER_NODE
echo "Using NPROC_PER_NODE=${NPROC_PER_NODE}"

export NANOCHAT_BASE_DIR="out"
mkdir -p $NANOCHAT_BASE_DIR
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: bash scripts/warmup_sweep.sh [flags] <depth1> [depth2] ..."
    echo "Flags:"
    echo "  --fp8                      Enable FP8 training"
    echo "  --no-compile               Disable torch.compile"
    echo "  --use-onecycle 0|1         Enable OneCycleLR scheduler (default: 1)"
    echo "  --max-shards N             Maximum number of dataset shards (default: 170)"
    echo "  --target-tokens N          Full training budget for warmup step calc (default: 20B)"
    echo "  --early-stop-tokens N      Per-run early stop length in tokens (recommended)"
    echo "  --run-tokens N             Legacy alias for --early-stop-tokens"
    echo "  --warmup-fracs '0.0 0.01'  Space-separated warmup fractions of full budget"
    echo "  --log-every N              Print logs every N steps (default: 1)"
    echo "  --models 'moe_perm ...'    Models to sweep"
    echo "  --tokenizer-dir PATH       Explicit tokenizer directory"
    echo "  --data-dir PATH            Explicit data directory"
    echo ""
    echo "Examples:"
    echo "  bash scripts/warmup_sweep.sh --fp8 --max-shards 170 8 16"
    echo "  bash scripts/warmup_sweep.sh --fp8 --target-tokens 20000000000 8"
    echo "  bash scripts/warmup_sweep.sh --fp8 --warmup-fracs '0.0 0.01 0.05' 8"
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ROOT_OUT_DIR="out/warmup_sweep_${TIMESTAMP}"

EXTRA_ARGS=()
MAX_SHARDS=170
TARGET_TOKENS=20000000000
DATA_DIR_FLAG=""
TOKENIZER_DIR_FLAG=""

# ── Parse flags ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fp8)
            EXTRA_ARGS+=("--fp8")
            shift
            ;;
        --no-compile)
            EXTRA_ARGS+=("--no-compile")
            shift
            ;;
        --max-shards)
            MAX_SHARDS=$2
            EXTRA_ARGS+=("--max-shards" "$2")
            shift 2
            ;;
        --target-tokens)
            EXTRA_ARGS+=("--target-tokens" "$2")
            shift 2
            ;;
        --run-tokens)
            EXTRA_ARGS+=("--early-stop-tokens" "$2")
            shift 2
            ;;
        --early-stop-tokens)
            EXTRA_ARGS+=("--early-stop-tokens" "$2")
            shift 2
            ;;
        --use-onecycle)
            EXTRA_ARGS+=("--use-onecycle" "$2")
            shift 2
            ;;
        --warmup-fracs)
            # nargs="+" expects: --warmup-fracs v1 v2 v3 (one flag, multiple values)
            EXTRA_ARGS+=("--warmup-fracs")
            for frac in $2; do
                EXTRA_ARGS+=("$frac")
            done
            shift 2
            ;;
        --log-every)
            EXTRA_ARGS+=("--log-every" "$2")
            shift 2
            ;;
        --models)
            # nargs="+" expects: --models base moe_perm ... (one flag, multiple values)
            EXTRA_ARGS+=("--models")
            for model in $2; do
                EXTRA_ARGS+=("$model")
            done
            shift 2
            ;;
        --tokenizer-dir)
            TOKENIZER_DIR_FLAG="$2"
            EXTRA_ARGS+=("--tokenizer-dir" "$2")
            shift 2
            ;;
        --data-dir)
            DATA_DIR_FLAG="$2"
            EXTRA_ARGS+=("--data-dir" "$2")
            shift 2
            ;;
        --research-dim)
            EXTRA_ARGS+=("--research-dim" "$2")
            shift 2
            ;;
        --cclblock-modulation)
            EXTRA_ARGS+=("--cclblock-modulation" "$2")
            shift 2
            ;;
        --cclblock-context-stream)
            EXTRA_ARGS+=("--cclblock-context-stream" "$2")
            shift 2
            ;;
        --cclblock-ema-factor)
            EXTRA_ARGS+=("--cclblock-ema-factor" "$2")
            shift 2
            ;;
        --cclblock-stale-ctx-lag)
            EXTRA_ARGS+=("--cclblock-stale-ctx-lag" "$2")
            shift 2
            ;;
        --cclblock-sparse-gate-k)
            EXTRA_ARGS+=("--cclblock-sparse-gate-k" "$2")
            shift 2
            ;;
        --cclblock-gate-temperature)
            EXTRA_ARGS+=("--cclblock-gate-temperature" "$2")
            shift 2
            ;;
        --cclblock-context-bank-size)
            EXTRA_ARGS+=("--cclblock-context-bank-size" "$2")
            shift 2
            ;;
        --cclblock-per-head-ctx)
            EXTRA_ARGS+=("--cclblock-per-head-ctx" "$2")
            shift 2
            ;;
        *)
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                break   # first numeric arg = depth, stop parsing flags
            else
                echo "Unknown argument: $1"
                exit 1
            fi
            ;;
    esac
done

if [ $# -eq 0 ]; then
    echo "Error: at least one depth argument is required."
    exit 1
fi

echo "===== GPU Info ====="
nvidia-smi || true

# ── Environment setup ─────────────────────────────────────────────────────────
SKIP_ENV_SETUP="${NANOCHAT_SKIP_ENV_SETUP:-0}"

if [[ "$SKIP_ENV_SETUP" == "1" ]]; then
    echo "Skipping env setup (NANOCHAT_SKIP_ENV_SETUP=1)"
    if [[ ! -d ".venv" ]]; then
        echo "Error: .venv not found. Unset NANOCHAT_SKIP_ENV_SETUP or create .venv first."
        exit 1
    fi
else
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
fi

source .venv/bin/activate

echo "Starting Warmup Sweep. Output directory: ${ROOT_OUT_DIR}"
echo "Full-budget target tokens: ${TARGET_TOKENS}"
mkdir -p "${ROOT_OUT_DIR}"

# ── One-time data + tokenizer setup ──────────────────────────────────────────
python -m nanochat.report reset

# Resolve directories for one-time calls
# Default data dir: $NANOCHAT_BASE_DIR/base_data_climbmix
CHECK_DATA_DIR="${DATA_DIR_FLAG:-$NANOCHAT_BASE_DIR/base_data_climbmix}"
DATA_OPT=""
[ -n "$DATA_DIR_FLAG" ] && DATA_OPT="--data-dir $DATA_DIR_FLAG"

TOK_OPT=""
[ -n "$TOKENIZER_DIR_FLAG" ] && TOK_OPT="--tokenizer-dir $TOKENIZER_DIR_FLAG"
TOKENIZER_CHECK_DIR="${TOKENIZER_DIR_FLAG:-$NANOCHAT_BASE_DIR/tokenizer}"

# 1. Dataset existence check
LAST_SHARD_IDX=$((MAX_SHARDS - 1))
LAST_SHARD_FILE=$(printf "shard_%05d.parquet" $LAST_SHARD_IDX)
VAL_SHARD_FILE="shard_06542.parquet"

if [ -f "$CHECK_DATA_DIR/$LAST_SHARD_FILE" ] && [ -f "$CHECK_DATA_DIR/$VAL_SHARD_FILE" ]; then
    echo "Dataset (up to $MAX_SHARDS shards) already exists in $CHECK_DATA_DIR, skipping download."
    DATASET_DOWNLOAD_PID=""
else
    echo "Dataset incomplete. Downloading $MAX_SHARDS shards to $CHECK_DATA_DIR..."
    mkdir -p "$CHECK_DATA_DIR"
    # Download 8 shards immediately so training can start while the rest come in
    python -m nanochat.dataset -n 8 $DATA_OPT
    python -m nanochat.dataset -n $MAX_SHARDS $DATA_OPT &
    DATASET_DOWNLOAD_PID=$!
fi

# 2. Tokenizer existence check
if [ ! -f "$TOKENIZER_CHECK_DIR/tokenizer.pkl" ]; then
    echo "Tokenizer not found in $TOKENIZER_CHECK_DIR, training..."
    python -m scripts.tok_train $TOK_OPT $DATA_OPT
    python -m scripts.tok_eval $TOK_OPT $DATA_OPT
else
    echo "Tokenizer already exists in $TOKENIZER_CHECK_DIR, skipping training."
fi

if [ -n "$DATASET_DOWNLOAD_PID" ]; then
    echo "Waiting for background dataset download to complete..."
    wait $DATASET_DOWNLOAD_PID
fi

# ── Per-depth sweep ───────────────────────────────────────────────────────────
for DEPTH in "$@"; do
    echo "================================================================"
    echo "Warmup Sweep for Depth: ${DEPTH}"
    echo "================================================================"

    RUN_DIR="${ROOT_OUT_DIR}/depth_${DEPTH}"
    mkdir -p "${RUN_DIR}"

    python -m scripts.warmup_sweep \
        --depth "${DEPTH}" \
        --run-dir "${RUN_DIR}" \
        "${EXTRA_ARGS[@]}"

    if [ $? -ne 0 ]; then
        echo "Error: Warmup sweep failed for depth ${DEPTH}. Check logs in ${RUN_DIR}."
        exit 1
    fi
done

python -m nanochat.report generate

echo "================================================================"
echo "Warmup Sweep Complete! Results saved to ${ROOT_OUT_DIR}"
echo "================================================================"
