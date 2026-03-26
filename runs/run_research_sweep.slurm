#!/bin/bash
#SBATCH --account=ufdatastudios
#SBATCH --job-name=nanochat-research-sweep
#SBATCH --output=out/slurm/research_sweep_%j.out
#SBATCH --error=out/slurm/research_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --partition=hpg-b200
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# Make sure we execute from repository root regardless of submit directory.
REPO_DIR="/orange/ufdatastudios/c.okocha/nanochat"
cd "$REPO_DIR"

mkdir -p out/slurm

echo "===== Job Context ====="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Working dir: $PWD"
echo "Args: $*"

echo "===== GPU Info ====="
nvidia-smi || true

# Keep tokenizer/OpenMP threads predictable on shared CPU resources.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# Reuse existing environment by default to avoid reinstalling dependencies each run.
export NANOCHAT_SKIP_ENV_SETUP="${NANOCHAT_SKIP_ENV_SETUP:-1}"

# Match torch distributed world size to GPUs reserved by Slurm.
if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  export NPROC_PER_NODE="$SLURM_GPUS_ON_NODE"
elif [[ -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
  export NPROC_PER_NODE="${SLURM_GPUS_PER_NODE%%(*}"
else
  export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
fi

echo "Using NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "NANOCHAT_SKIP_ENV_SETUP=${NANOCHAT_SKIP_ENV_SETUP}"

if [[ $# -eq 0 ]]; then
  SWEEP_ARGS=(--fp8 --max-shards 170 8 16 24)
  echo "No args provided, using default sweep args: ${SWEEP_ARGS[*]}"
else
  SWEEP_ARGS=("$@")
fi

bash scripts/research_sweep.sh "${SWEEP_ARGS[@]}"
