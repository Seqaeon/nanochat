#!/bin/bash
# research_sweep.sh
# Automates the research comparison script across multiple depths
export OMP_NUM_THREADS=8

# ── 0. Clone or update the repo ───────────────────────────────────────────────
#REPO_URL="https://github.com/Seqaeon/nanochat.git"
#REPO_DIR="nanochat"
#
#if [ ! -d "$REPO_DIR" ]; then
#    git clone "$REPO_URL"
#else
#    echo "Repo already exists, pulling latest..."
#    git -C "$REPO_DIR" pull origin master
#fi
#
#cd "$REPO_DIR"

export NANOCHAT_BASE_DIR="out"
mkdir -p $NANOCHAT_BASE_DIR
set -euo pipefail

# Default to 8-way launch (for full 8xGPU nodes), but allow override via env
# and cap to locally visible GPUs when fewer are allocated.
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
if [[ "${AVAILABLE_GPUS}" =~ ^[0-9]+$ ]] && [ "$AVAILABLE_GPUS" -gt 0 ] && [ "$NPROC_PER_NODE" -gt "$AVAILABLE_GPUS" ]; then
    echo "Requested NPROC_PER_NODE=${NPROC_PER_NODE}, but only ${AVAILABLE_GPUS} GPU(s) visible. Capping."
    NPROC_PER_NODE="${AVAILABLE_GPUS}"
fi
echo "Using NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "NANOCHAT_SKIP_ENV_SETUP=${NANOCHAT_SKIP_ENV_SETUP:-0}"

echo "===== GPU Info ====="
nvidia-smi || true

if [ $# -eq 0 ]; then
    echo "Usage: ./scripts/research_sweep.sh [flags] <depth1> [depth2] ..."
    echo "Flags:"
    echo "  --fp8               Enable FP8 training"
    echo "  --no-compile        Disable torch.compile"
    echo "  --target-tokens N   Explicit token budget per run (default: auto-estimated)"
    echo "  --max-shards N      Step through up to N data shards"
    echo ""
    echo "Example: ./scripts/research_sweep.sh --fp8 --target-tokens 100000000 8"
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ROOT_OUT_DIR="out/research_sweep_${TIMESTAMP}"

EXTRA_ARGS=""
MAX_SHARDS=170

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
            EXTRA_ARGS="$EXTRA_ARGS --target-tokens $2"
            shift 2
            ;;
        --tokenizer-dir)
            EXTRA_ARGS="$EXTRA_ARGS --tokenizer-dir $2"
            shift 2
            ;;
        --data-dir)
            EXTRA_ARGS="$EXTRA_ARGS --data-dir $2"
            shift 2
            ;;
        *)
            # Stop parsing flags when we hit a depth
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                break
            else
                echo "Unknown argument: $1"
                exit 1
            fi
            ;;
    esac
done

# Check for depths again after potential shift
if [ $# -eq 0 ]; then
    echo "Usage: ./scripts/research_sweep.sh [flags] <depth1> [depth2] ..."
    exit 1
fi

echo "Starting Research Sweep. Output directory: ${ROOT_OUT_DIR}"
mkdir -p "${ROOT_OUT_DIR}"

# ── 1. Install uv if missing ──────────────────────────────────────────────────
SKIP_ENV_SETUP="${NANOCHAT_SKIP_ENV_SETUP:-0}"

if [[ "$SKIP_ENV_SETUP" == "1" ]]; then
    echo "Skipping environment setup because NANOCHAT_SKIP_ENV_SETUP=1"
    if [[ ! -d ".venv" ]]; then
        echo "Error: .venv not found, cannot skip setup. Unset NANOCHAT_SKIP_ENV_SETUP or create .venv first."
        exit 1
    fi
else
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

    # ── 2. Create venv and install dependencies ───────────────────────────────
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
fi

# ── 3. Activate venv ──────────────────────────────────────────────────────────
source .venv/bin/activate

# ── 4. Resolve torchrun (prefers venv, falls back to system) ──────────────────
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
if command -v torchrun &> /dev/null; then
    RUNNER="torchrun --standalone --nproc_per_node=${NPROC_PER_NODE}"
else
    echo "Warning: torchrun not found, falling back to python -m torch.distributed.run"
    RUNNER="python -m torch.distributed.run --standalone --nproc_per_node=${NPROC_PER_NODE}"
fi

python -m nanochat.report reset
python -m nanochat.dataset -n 8 

python -m nanochat.dataset -n $MAX_SHARDS &
DATASET_DOWNLOAD_PID=$!
#
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    python -m scripts.tok_train
    # evaluate the tokenizer (report compression ratio etc.)
    python -m scripts.tok_eval
else
    echo "Tokenizer already exists in $NANOCHAT_BASE_DIR/tokenizer, skipping training."
fi

#
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID


# Use the current python or fallback to venv if it exists locally
#if [ -n "$VIRTUAL_ENV" ]; then
#    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
#elif [ -d ".venv" ]; then
#    PYTHON_BIN=".venv/bin/python"
#else
#    PYTHON_BIN="python3"
#fi

for DEPTH in "$@"; do
    echo "================================================================"
    echo "Processing Depth: ${DEPTH}"
    echo "================================================================"
    
    RUN_DIR="${ROOT_OUT_DIR}/depth_${DEPTH}"
    mkdir -p "${RUN_DIR}"

    # research_compare orchestrates multiple training subprocesses itself.
    # Running it under torchrun causes nested distributed launches and rank/env collisions.
    python -m scripts.research_compare --depth "${DEPTH}" --run-dir "${RUN_DIR}" $EXTRA_ARGS
    
#    $PYTHON_BIN -m scripts.research_compare --depth "${DEPTH}" --run-dir "${RUN_DIR}" $EXTRA_ARGS
    
    if [ $? -ne 0 ]; then
        echo "Error: Sweep failed for depth ${DEPTH}. Check logs in ${RUN_DIR}."
        exit 1
    fi
done
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run SFT and eval the model
#torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- --device-batch-size=16 #--run=$WANDB_RUN
#torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i sft
python -m nanochat.report generate
echo "================================================================"
echo "Sweep Complete! Results saved to ${ROOT_OUT_DIR}"
echo "================================================================"
