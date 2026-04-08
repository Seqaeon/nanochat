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
    echo "  --phase 1|2|3               Run only a specific phase (default: all)"
    echo "  --models 'M1 M2 ...'        Space-quoted list of models to sweep"
    echo "                              Choices: base moe_no_perm moe_perm remixed-linear"
    echo "                              Default: moe_no_perm moe_perm remixed-linear"
    echo "  --generate-only             Print run configs without training"
    echo "  --disable-base-mu-p         Also disable \u03bcP LR scaling for base model"
    echo "                              (research models always have \u03bcP disabled)"
    echo ""
    echo "Phase 1 — Absolute LR grid sweep (all groups same value):"
    echo "  --phase1-lrs 'LR1 LR2 ...'  Space-quoted absolute LR values to sweep"
    echo "                              Default: 0.0003 0.001 0.003 0.01 0.03 0.1 0.3"
    echo ""
    echo "Phase 2 — Coordinate descent around the phase-1 winner:"
    echo "  --lr-star F                 Absolute LR winner from phase 1 (center for p2/p3)"
    echo "  --lr-star-json PATH         JSON file with per-model lr_star dict"
    echo "  --phase2-multipliers 'M1 M2 ...'  Per-group multipliers (default: 0.3 1.0 3.0)"
    echo ""
    echo "Phase 3 — Optional refinement:"
    echo "  --phase3-samples N          Number of random samples (default: 10)"
    echo "  --phase3-log-radius F       Log10 search radius (default: 0.3)"
    echo "  --seed N                    RNG seed (default: 1337)"
    echo ""
    echo "Training flags:"
    echo "  --target-tokens N           Full training budget in tokens (default: auto)"
    echo "  --early-stop-tokens N       Per-run early-stop token count (default: 100M)"
    echo "  --warmup-ratio F            LR warmup ratio (default: 0.0)"
    echo "  --use-onecycle 0|1          Enable OneCycleLR scheduler (default: 1)"
    echo "  --device-batch-size N       Micro-batch size per GPU (default: 16)"
    echo "  --moe-num-experts N         MoE expert count (default: 8)"
    echo "  --log-every N               Log every N steps (default: 200)"
    echo "  --tokenizer-dir PATH        Explicit tokenizer directory"
    echo "  --data-dir PATH             Explicit data directory"
    echo "  --max-shards N              Max data shards to use (default: 170)"
    echo ""
    echo "Phase 4 — Bayesian Optimisation (Optuna/TPE):"
    echo "  --phase4-trials N                 Number of Optuna trials (default: 50)"
    echo "  --phase4-lr-min F                 LR search space lower bound (default: 1e-4)"
    echo "  --phase4-lr-max F                 LR search space upper bound (default: 0.5)"
    echo "  --phase4-warm-start/--no-...      Seed Optuna with Phase 1-3 results (default: warm-start)"
    echo ""
    echo "Resumption and Indexing:"
    echo "  --run-dir PATH              Override the root output directory (re-uses existing results)"
    echo "  --start-index N             Start execution from this configuration index"
    echo "  --end-index N               Stop execution after this index (-1: end)"
    echo "  --resume / --no-resume      Skip already-finished training runs (default: --resume)"
    echo ""
    echo "Examples:"
    echo "  # Phase 1: sweep LRs for MoE models at depth 8"
    echo "  bash scripts/actual_lr_research_sweep.sh \\"
    echo "      --phase 1 \\"
    echo "      --models 'moe_perm moe_no_perm remixed-linear' \\"
    echo "      --phase1-lrs '0.001 0.003 0.01 0.03 0.1' \\"
    echo "      --early-stop-tokens 100000000 \\"
    echo "      --fp8 8"
    echo ""
    echo "  # Phase 2: coordinate descent around best LR from phase 1"
    echo "  bash scripts/actual_lr_research_sweep.sh \\"
    echo "      --phase 2 --lr-star 0.01 \\"
    echo "      --models 'moe_perm remixed-linear' \\"
    echo "      --early-stop-tokens 100000000 \\"
    echo "      --fp8 8"
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ROOT_OUT_DIR=""  # Set via --run-dir flag below

EXTRA_ARGS=()
MAX_SHARDS=170
DATA_DIR_FLAG=""
TOKENIZER_DIR_FLAG=""

# ── Parse flags ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            # Trigger help by setting argc to 0 and re-running this script 
            # (or just call the help function if we had one, but it's just top-level script)
            # Simplest is to just exit and let the top check handle it if we reset args
            exec bash "$0"
            ;;
        --fp8)
            EXTRA_ARGS+=("--fp8")
            shift
            ;;
        --no-compile)
            EXTRA_ARGS+=("--no-compile")
            shift
            ;;
        --generate-only)
            EXTRA_ARGS+=("--generate-only")
            shift
            ;;
        --max-shards)
            MAX_SHARDS=$2
            EXTRA_ARGS+=("--max-shards" "$2")
            shift 2
            ;;
        --run-dir)
            ROOT_OUT_DIR="$2"
            shift 2
            ;;
        --start-index)
            EXTRA_ARGS+=("--start-index" "$2")
            shift 2
            ;;
        --end-index)
            EXTRA_ARGS+=("--end-index" "$2")
            shift 2
            ;;
        --resume)
            EXTRA_ARGS+=("--resume")
            shift
            ;;
        --no-resume)
            EXTRA_ARGS+=("--no-resume")
            shift
            ;;
        --phase)
            EXTRA_ARGS+=("--phase" "$2")
            shift 2
            ;;
        --models)
            EXTRA_ARGS+=("--models")
            for model in $2; do
                EXTRA_ARGS+=("$model")
            done
            shift 2
            ;;
        --phase1-lrs)
            EXTRA_ARGS+=("--phase1-lrs")
            for lr in $2; do
                EXTRA_ARGS+=("$lr")
            done
            shift 2
            ;;
        --lr-star)
            EXTRA_ARGS+=("--lr-star" "$2")
            shift 2
            ;;
        --lr-star-json)
            EXTRA_ARGS+=("--lr-star-json" "$2")
            shift 2
            ;;
        --disable-base-mu-p)
            EXTRA_ARGS+=("--disable-base-mu-p")
            shift
            ;;
        --phase2-multipliers)
            EXTRA_ARGS+=("--phase2-multipliers")
            for mult in $2; do
                EXTRA_ARGS+=("$mult")
            done
            shift 2
            ;;
        --phase3-samples)
            EXTRA_ARGS+=("--phase3-samples" "$2")
            shift 2
            ;;
        --phase3-log-radius)
            EXTRA_ARGS+=("--phase3-log-radius" "$2")
            shift 2
            ;;
        --phase4-trials)
            EXTRA_ARGS+=("--phase4-trials" "$2")
            shift 2
            ;;
        --phase4-lr-min)
            EXTRA_ARGS+=("--phase4-lr-min" "$2")
            shift 2
            ;;
        --phase4-lr-max)
            EXTRA_ARGS+=("--phase4-lr-max" "$2")
            shift 2
            ;;
        --phase4-warm-start)
            EXTRA_ARGS+=("--phase4-warm-start")
            shift
            ;;
        --no-phase4-warm-start)
            EXTRA_ARGS+=("--no-phase4-warm-start")
            shift
            ;;
        --seed)
            EXTRA_ARGS+=("--seed" "$2")
            shift 2
            ;;
        --target-tokens)
            EXTRA_ARGS+=("--target-tokens" "$2")
            shift 2
            ;;
        --early-stop-tokens)
            EXTRA_ARGS+=("--early-stop-tokens" "$2")
            shift 2
            ;;
        --warmup-ratio)
            EXTRA_ARGS+=("--warmup-ratio" "$2")
            shift 2
            ;;
        --research-warmup-ratio)
            EXTRA_ARGS+=("--research-warmup-ratio" "$2")
            shift 2
            ;;
        --use-onecycle)
            EXTRA_ARGS+=("--use-onecycle" "$2")
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
        --max-seq-len)
            EXTRA_ARGS+=("--max-seq-len" "$2")
            shift 2
            ;;
        --moe-num-experts)
            EXTRA_ARGS+=("--moe-num-experts" "$2")
            shift 2
            ;;
        --log-every)
            EXTRA_ARGS+=("--log-every" "$2")
            shift 2
            ;;
        --eval-every)
            EXTRA_ARGS+=("--eval-every" "$2")
            shift 2
            ;;
        --tokenizer-dir)
            TOKENIZER_DIR_FLAG="$2"
            EXTRA_ARGS+=("--tokenizer-dir" "$2")
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
        --cclblock-context-source)
            EXTRA_ARGS+=("--cclblock-context-source" "$2")
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

if [ -z "$ROOT_OUT_DIR" ]; then
    ROOT_OUT_DIR="out/actual_lr_research_sweep_${TIMESTAMP}"
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
    echo "Actual-LR Research Sweep for Depth: ${DEPTH}"
    echo "================================================================"

    RUN_DIR="${ROOT_OUT_DIR}/depth_${DEPTH}"
    mkdir -p "${RUN_DIR}"

    # actual_lr_research_sweep.py orchestrates multiple training subprocesses itself.
    # Run under plain python (not torchrun) to avoid nested distributed launches.
    python -m scripts.actual_lr_research_sweep \
        --depth "${DEPTH}" \
        --run-dir "${RUN_DIR}" \
        "${EXTRA_ARGS[@]}"

    if [ $? -ne 0 ]; then
        echo "Error: Actual-LR research sweep failed for depth ${DEPTH}. Check logs in ${RUN_DIR}."
        exit 1
    fi
done

python -m nanochat.report generate

echo "================================================================"
echo "Actual-LR Research Sweep Complete! Results saved to ${ROOT_OUT_DIR}"
echo "================================================================"
