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
# calculation. --run-tokens is the per-run early-stop length (e.g. 100M default).
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
    echo "  --max-shards N             Maximum number of dataset shards (default: 170)"
    echo "  --target-tokens N          Full training budget for warmup step calc (default: 20B)"
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

EXTRA_ARGS=""
MAX_SHARDS=170
TARGET_TOKENS=100000000

# ── Parse flags ───────────────────────────────────────────────────────────────
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
        --warmup-fracs)
            # Accept space-separated list quoted in one arg: --warmup-fracs "0.0 0.01 0.05"
            EXTRA_ARGS="$EXTRA_ARGS --warmup-fracs $2"
            shift 2
            ;;
        --log-every)
            EXTRA_ARGS="$EXTRA_ARGS --log-every $2"
            shift 2
            ;;
        --models)
            EXTRA_ARGS="$EXTRA_ARGS --models $2"
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
echo "Target tokens per run: ${TARGET_TOKENS}"
mkdir -p "${ROOT_OUT_DIR}"

# ── One-time data + tokenizer setup ──────────────────────────────────────────
python -m nanochat.report reset

# Download 8 shards immediately so training can start while the rest come in
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n $MAX_SHARDS &
DATASET_DOWNLOAD_PID=$!

if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    python -m scripts.tok_train
    python -m scripts.tok_eval
else
    echo "Tokenizer already exists in $NANOCHAT_BASE_DIR/tokenizer, skipping training."
fi

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

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
        $EXTRA_ARGS

    if [ $? -ne 0 ]; then
        echo "Error: Warmup sweep failed for depth ${DEPTH}. Check logs in ${RUN_DIR}."
        exit 1
    fi
done

python -m nanochat.report generate

echo "================================================================"
echo "Warmup Sweep Complete! Results saved to ${ROOT_OUT_DIR}"
echo "================================================================"
