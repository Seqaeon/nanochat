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
EXTRA_ARGS=""
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
            EXTRA_ARGS="$EXTRA_ARGS --fp8"
            shift
            ;;
        --no-compile)
            EXTRA_ARGS="$EXTRA_ARGS --no-compile"
            shift
            ;;
        --max-shards)
            MAX_SHARDS=$2
            EXTRA_ARGS="$EXTRA_ARGS --max-shards $2"
            shift 2
            ;;
        --target-tokens)
            TARGET_TOKENS=$2
            EXTRA_ARGS="$EXTRA_ARGS --target-tokens $2"
            shift 2
            ;;
        --use-onecycle)
            EXTRA_ARGS="$EXTRA_ARGS --use-onecycle $2"
            shift 2
            ;;
        --lr-scale-factors)
            # Accept space-separated list quoted in one arg: --lr-scale-factors "1.0 3.0 5.0"
            LR_SCALE_FACTORS="$2"
            EXTRA_ARGS="$EXTRA_ARGS --lr-scale-factors $2"
            shift 2
            ;;
        --models)
            # --models "base moe_perm"
            MODELS="$2"
            EXTRA_ARGS="$EXTRA_ARGS --models $2"
            shift 2
            ;;
        --eval-every)
            EXTRA_ARGS="$EXTRA_ARGS --eval-every $2"
            shift 2
            ;;
        --tokenizer-dir)
            TOKENIZER_DIR_FLAG="$2"
            EXTRA_ARGS="$EXTRA_ARGS --tokenizer-dir $2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR_FLAG="$2"
            EXTRA_ARGS="$EXTRA_ARGS --data-dir $2"
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
DATA_OPT=""
[ -n "$DATA_DIR_FLAG" ] && DATA_OPT="--data-dir $DATA_DIR_FLAG"
TOK_OPT=""
[ -n "$TOKENIZER_DIR_FLAG" ] && TOK_OPT="--tokenizer-dir $TOKENIZER_DIR_FLAG"

# Download 8 shards immediately so training can start while the rest come in
python -m nanochat.dataset -n 8 $DATA_OPT
python -m nanochat.dataset -n $MAX_SHARDS $DATA_OPT &
DATASET_DOWNLOAD_PID=$!

TOKENIZER_CHECK_DIR="${TOKENIZER_DIR_FLAG:-$NANOCHAT_BASE_DIR/tokenizer}"
if [ ! -f "$TOKENIZER_CHECK_DIR/tokenizer.pkl" ]; then
    python -m scripts.tok_train $TOK_OPT $DATA_OPT
    python -m scripts.tok_eval $TOK_OPT $DATA_OPT
else
    echo "Tokenizer already exists in $TOKENIZER_CHECK_DIR, skipping training."
fi

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

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
        $EXTRA_ARGS

    if [ $? -ne 0 ]; then
        echo "Error: LR sweep failed for depth ${DEPTH}. Check logs in ${RUN_DIR}."
        exit 1
    fi
done

python -m nanochat.report generate

echo "================================================================"
echo "LR Sweep Complete! Results saved to ${ROOT_OUT_DIR}"
echo "================================================================"
