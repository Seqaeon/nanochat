#!/bin/bash
# scripts/actual_lr_research_sweep.sh
#
# Phased coordinate-descent LR sweep for research model variants (base, MoE, remixed-linear).
# Wraps actual_lr_research_sweep.py — each depth is swept independently.
#
# Usage:
#   bash scripts/actual_lr_research_sweep.sh [flags] <depth1> [depth2] ...
#
# Examples:
#   bash scripts/actual_lr_research_sweep.sh --fp8 --max-shards 170 8
#   bash scripts/actual_lr_research_sweep.sh --fp8 --max-shards 170 --early-stop-tokens 100000000 8 16
#   bash scripts/actual_lr_research_sweep.sh --phase 1 --models "base moe_perm" 8
#   bash scripts/actual_lr_research_sweep.sh --generate-only 8
#
# Phase semantics:
#   --phase 0  (default) — run all three phases sequentially
#   --phase 1            — uniform scale sweep around each model's base LR profile
#   --phase 2            — coordinate descent per LR group around s* center
#   --phase 3            — random log-space refinement around phase-2 prior
#
# Env-var overrides:
#   NPROC_PER_NODE           — DDP worker count (default: 8, auto-capped to GPU count)
#   NANOCHAT_SKIP_ENV_SETUP=1 — skip uv/venv setup if already activated

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
    echo "Usage: bash scripts/actual_lr_research_sweep.sh [flags] <depth1> [depth2] ..."
    echo ""
    echo "Flags:"
    echo "  --fp8                           Enable FP8 training"
    echo "  --no-compile                    Disable torch.compile"
    echo "  --max-shards N                  Maximum number of dataset shards (default: 170)"
    echo "  --phase N                       Phase to run: 1, 2, 3, or 0=all (default: 0)"
    echo "  --generate-only                 Only emit JSON run configs, no training"
    echo "  --models 'base moe_perm ...'    Space-separated model list"
    echo "  --phase1-scales '0.03 0.1 ...'  Phase-1 uniform scale sweep values"
    echo "  --s-star F                      Phase-2/3 center scale factor (default: 0.1)"
    echo "  --phase2-multipliers '0.3 1 3'  Phase-2 per-group multipliers"
    echo "  --phase3-samples N              Phase-3 random samples (default: 10)"
    echo "  --phase3-log-radius F           Phase-3 log10 radius (default: 0.3)"
    echo "  --seed N                        RNG seed (default: 1337)"
    echo "  --target-tokens N               Full training budget in tokens (default: 20B)"
    echo "  --early-stop-tokens N           Per-run early-stop token count (default: 100M)"
    echo "  --warmup-ratio F                LR warmup ratio (default: 0.0)"
    echo "  --research-warmup-ratio F       Research warmup ratio (default: 0.0)"
    echo "  --use-onecycle 0|1              Enable OneCycleLR scheduler (default: 1)"
    echo "  --device-batch-size N           Micro-batch size per GPU (default: 16)"
    echo "  --total-batch-size N            Global batch size in tokens (default: 524288)"
    echo "  --max-seq-len N                 Sequence length (default: 2048)"
    echo "  --num-experts N                 MoE expert count (default: 8)"
    echo "  --log-every N                   Log every N steps (default: 200)"
    echo "  --eval-every N                  Eval every N steps (default: 0=off)"
    echo "  --tokenizer-dir PATH            Explicit tokenizer directory"
    echo "  --data-dir PATH                 Explicit data directory"
    echo ""
    echo "Examples:"
    echo "  bash scripts/actual_lr_research_sweep.sh --fp8 --max-shards 170 --early-stop-tokens 100000000 8"
    echo "  bash scripts/actual_lr_research_sweep.sh --phase 1 --models 'base moe_perm' 8 16"
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ROOT_OUT_DIR="out/actual_lr_research_sweep_${TIMESTAMP}"

EXTRA_ARGS=""
MAX_SHARDS=170

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
        --generate-only)
            EXTRA_ARGS="$EXTRA_ARGS --generate-only"
            shift
            ;;
        --max-shards)
            MAX_SHARDS=$2
            EXTRA_ARGS="$EXTRA_ARGS --max-shards $2"
            shift 2
            ;;
        --phase)
            EXTRA_ARGS="$EXTRA_ARGS --phase $2"
            shift 2
            ;;
        --models)
            # Accept space-separated list quoted as one arg: --models "base moe_perm"
            EXTRA_ARGS="$EXTRA_ARGS --models $2"
            shift 2
            ;;
        --phase1-scales)
            EXTRA_ARGS="$EXTRA_ARGS --phase1-scales $2"
            shift 2
            ;;
        --s-star)
            EXTRA_ARGS="$EXTRA_ARGS --s-star $2"
            shift 2
            ;;
        --phase2-multipliers)
            EXTRA_ARGS="$EXTRA_ARGS --phase2-multipliers $2"
            shift 2
            ;;
        --phase3-samples)
            EXTRA_ARGS="$EXTRA_ARGS --phase3-samples $2"
            shift 2
            ;;
        --phase3-log-radius)
            EXTRA_ARGS="$EXTRA_ARGS --phase3-log-radius $2"
            shift 2
            ;;
        --seed)
            EXTRA_ARGS="$EXTRA_ARGS --seed $2"
            shift 2
            ;;
        --target-tokens)
            EXTRA_ARGS="$EXTRA_ARGS --target-tokens $2"
            shift 2
            ;;
        --early-stop-tokens)
            EXTRA_ARGS="$EXTRA_ARGS --early-stop-tokens $2"
            shift 2
            ;;
        --warmup-ratio)
            EXTRA_ARGS="$EXTRA_ARGS --warmup-ratio $2"
            shift 2
            ;;
        --research-warmup-ratio)
            EXTRA_ARGS="$EXTRA_ARGS --research-warmup-ratio $2"
            shift 2
            ;;
        --use-onecycle)
            EXTRA_ARGS="$EXTRA_ARGS --use-onecycle $2"
            shift 2
            ;;
        --device-batch-size)
            EXTRA_ARGS="$EXTRA_ARGS --device-batch-size $2"
            shift 2
            ;;
        --total-batch-size)
            EXTRA_ARGS="$EXTRA_ARGS --total-batch-size $2"
            shift 2
            ;;
        --max-seq-len)
            EXTRA_ARGS="$EXTRA_ARGS --max-seq-len $2"
            shift 2
            ;;
        --num-experts)
            EXTRA_ARGS="$EXTRA_ARGS --num-experts $2"
            shift 2
            ;;
        --log-every)
            EXTRA_ARGS="$EXTRA_ARGS --log-every $2"
            shift 2
            ;;
        --eval-every)
            EXTRA_ARGS="$EXTRA_ARGS --eval-every $2"
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

echo "Starting Actual-LR Research Sweep. Output directory: ${ROOT_OUT_DIR}"
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
    echo "Actual-LR Research Sweep for Depth: ${DEPTH}"
    echo "================================================================"

    RUN_DIR="${ROOT_OUT_DIR}/depth_${DEPTH}"
    mkdir -p "${RUN_DIR}"

    # actual_lr_research_sweep.py orchestrates multiple training subprocesses itself.
    # Run under plain python (not torchrun) to avoid nested distributed launches.
    python -m scripts.actual_lr_research_sweep \
        --depth "${DEPTH}" \
        --run-dir "${RUN_DIR}" \
        $EXTRA_ARGS

    if [ $? -ne 0 ]; then
        echo "Error: Actual-LR research sweep failed for depth ${DEPTH}. Check logs in ${RUN_DIR}."
        exit 1
    fi
done

python -m nanochat.report generate

echo "================================================================"
echo "Actual-LR Research Sweep Complete! Results saved to ${ROOT_OUT_DIR}"
echo "================================================================"
