#!/bin/bash
# scripts/lr_sweep.sh
#
# Learning-rate sweep for base and CCL (MoE perm) models across multiple depths.
# Trains each (model, lr_scale) for a fixed token budget and plots loss curves.
#
# Usage:
#   bash scripts/lr_sweep.sh [flags] <depth1> [depth2] ...
#
# Examples:
#   bash scripts/lr_sweep.sh --fp8 --max-shards 170 8 16 24
#   bash scripts/lr_sweep.sh --fp8 --max-shards 170 --target-tokens 1000000000 8 16 24
#   bash scripts/lr_sweep.sh --fp8 --max-shards 170 --models "base" --lr-scale-factors "1.0 3.0 5.0" 8
#
# Env-var overrides:
#   NPROC_PER_NODE      — DDP worker count (default: 8, auto-capped to GPU count)
#   NANOCHAT_SKIP_ENV_SETUP=1  — skip uv/venv setup if already activated

export OMP_NUM_THREADS=8

# ── GPU count guard ──────────────────────────────────────────────────────────
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
    echo "Usage: bash scripts/lr_sweep.sh [--fp8] [--no-compile] [--max-shards N]"
    echo "       [--target-tokens N] [--lr-scale-factors '1.0 3.0 5.0 7.0 10.0']"
    echo "       [--models 'base moe_perm'] [--eval-every N]"
    echo "       <depth1> [depth2] ..."
    echo ""
    echo "Examples:"
    echo "  bash scripts/lr_sweep.sh --fp8 --max-shards 170 8 16 24"
    echo "  bash scripts/lr_sweep.sh --fp8 --max-shards 170 --target-tokens 1000000000 --models 'base moe_perm' 8"
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ROOT_OUT_DIR="out/lr_sweep_${TIMESTAMP}"

# Flags forwarded to lr_sweep.py
EXTRA_ARGS=()
MAX_SHARDS=170
TARGET_TOKENS=1000000000
LR_SCALE_FACTORS=""
MODELS=""
DATA_DIR_FLAG=""
TOKENIZER_DIR_FLAG=""

# ── Parse flags ──────────────────────────────────────────────────────────────
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
            TARGET_TOKENS=$2
            EXTRA_ARGS+=("--target-tokens" "$2")
            shift 2
            ;;
        --use-onecycle)
            EXTRA_ARGS+=("--use-onecycle" "$2")
            shift 2
            ;;
        --lr-scale-factors)
            # Accept space-separated list quoted in one arg: --lr-scale-factors "1.0 3.0 5.0"
            EXTRA_ARGS+=("--lr-scale-factors")
            for scale in $2; do
                EXTRA_ARGS+=("$scale")
            done
            shift 2
            ;;
        --models)
            # --models "base moe_perm" — nargs="+" needs single flag then all values
            EXTRA_ARGS+=("--models")
            for model in $2; do
                EXTRA_ARGS+=("$model")
            done
            shift 2
            ;;
        --eval-every)
            EXTRA_ARGS+=("--eval-every" "$2")
            shift 2
            ;;
        --sequence-len)
            EXTRA_ARGS+=("--sequence-len" "$2")
            shift 2
            ;;
        --device-batch-size)
            EXTRA_ARGS+=("--device-batch-size" "$2")
            shift 2
            ;;
        --total-batch-size)
            EXTRA_ARGS+=("--total-batch-size" "$2")
            shift 2
            ;;
        --router-context-window)
            EXTRA_ARGS+=("--router-context-window" "$2")
            shift 2
            ;;
        --research-dim)
            EXTRA_ARGS+=("--research-dim" "$2")
            shift 2
            ;;
        --remix-use-basis-gate)
            EXTRA_ARGS+=("--remix-use-basis-gate" "$2")
            shift 2
            ;;
        --remix-use-output-gate)
            EXTRA_ARGS+=("--remix-use-output-gate" "$2")
            shift 2
            ;;
        --remix-use-context)
            EXTRA_ARGS+=("--remix-use-context" "$2")
            shift 2
            ;;
        --cclblock-modulation)
            EXTRA_ARGS+=("--cclblock-modulation" "$2")
            shift 2
            ;;
        --cclblock-orth-lambda)
            EXTRA_ARGS+=("--cclblock-orth-lambda" "$2")
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
        --cclblock-context-source)
            EXTRA_ARGS+=("--cclblock-context-source" "$2")
            shift 2
            ;;
        --cclblock-chunk-size)
            EXTRA_ARGS+=("--cclblock-chunk-size" "$2")
            shift 2
            ;;
        --cclblock-aux-objective)
            EXTRA_ARGS+=("--cclblock-aux-objective" "$2")
            shift 2
            ;;
        --cclblock-aux-lambda)
            EXTRA_ARGS+=("--cclblock-aux-lambda" "$2")
            shift 2
            ;;
        --cclblock-boundary-token-id)
            EXTRA_ARGS+=("--cclblock-boundary-token-id" "$2")
            shift 2
            ;;
        --use-ral)
            EXTRA_ARGS+=("--use-ral" "$2")
            shift 2
            ;;
        --ral-rank)
            EXTRA_ARGS+=("--ral-rank" "$2")
            shift 2
            ;;
        --cclblock-film-gate)
            EXTRA_ARGS+=("--cclblock-film-gate" "$2")
            shift 2
            ;;
        --cclblock-attn-shadow-dim)
            EXTRA_ARGS+=("--cclblock-attn-shadow-dim" "$2")
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

# ── Environment setup ────────────────────────────────────────────────────────
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

echo "Starting LR Sweep. Output directory: ${ROOT_OUT_DIR}"
echo "Target tokens per run: ${TARGET_TOKENS}"
mkdir -p "${ROOT_OUT_DIR}"

# ── One-time data + tokenizer setup ─────────────────────────────────────────
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

# ── Per-depth sweep ──────────────────────────────────────────────────────────
for DEPTH in "$@"; do
    echo "================================================================"
    echo "LR Sweep for Depth: ${DEPTH}  |  Target tokens: ${TARGET_TOKENS}"
    echo "================================================================"

    RUN_DIR="${ROOT_OUT_DIR}/depth_${DEPTH}"
    mkdir -p "${RUN_DIR}"

    python -m scripts.lr_sweep \
        --depth "${DEPTH}" \
        --run-dir "${RUN_DIR}" \
        "${EXTRA_ARGS[@]}"

    if [ $? -ne 0 ]; then
        echo "Error: LR sweep failed for depth ${DEPTH}. Check logs in ${RUN_DIR}."
        exit 1
    fi
done

python -m nanochat.report generate

echo "================================================================"
echo "LR Sweep Complete! Results saved to ${ROOT_OUT_DIR}"
echo "================================================================"
