#!/bin/bash
# research_sweep.sh
# Automates the research comparison script across multiple depths

if [ $# -eq 0 ]; then
    echo "Usage: ./scripts/research_sweep.sh <depth1> [depth2] [depth3] ..."
    echo "Example: ./scripts/research_sweep.sh 2 4 8"
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ROOT_OUT_DIR="out/research_sweep_${TIMESTAMP}"

FP8_FLAG=""
case "$1" in
    --fp8)
        FP8_FLAG="--fp8"
        shift
        ;;
esac

# Check for depths again after potential shift
if [ $# -eq 0 ]; then
    echo "Usage: ./scripts/research_sweep.sh [--fp8] <depth1> [depth2] ..."
    exit 1
fi

echo "Starting Research Sweep. Output directory: ${ROOT_OUT_DIR}"
mkdir -p "${ROOT_OUT_DIR}"

# Use the absolute path to the venv python to be safe
PYTHON_BIN="/home/seqaeon/Downloads/nanochat/.venv/bin/python"

for DEPTH in "$@"; do
    echo "================================================================"
    echo "Processing Depth: ${DEPTH}"
    echo "================================================================"
    
    RUN_DIR="${ROOT_OUT_DIR}/depth_${DEPTH}"
    mkdir -p "${RUN_DIR}"
    
    $PYTHON_BIN -m scripts.research_compare --depth "${DEPTH}" --run-dir "${RUN_DIR}" $FP8_FLAG
    
    if [ $? -ne 0 ]; then
        echo "Error: Sweep failed for depth ${DEPTH}. Check logs in ${RUN_DIR}."
        exit 1
    fi
done

echo "================================================================"
echo "Sweep Complete! Results saved to ${ROOT_OUT_DIR}"
echo "================================================================"
