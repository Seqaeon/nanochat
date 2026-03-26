#!/bin/bash
#SBATCH --account=ufdatastudios
#SBATCH --job-name=nanochat-b200-smoke
#SBATCH --output=out/slurm/b200_smoke_%j.out
#SBATCH --error=out/slurm/b200_smoke_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:10:00
#SBATCH --partition=hpg-b200
#SBATCH --mail-type=END,FAIL

set -euo pipefail

REPO_DIR="/orange/ufdatastudios/c.okocha/nanochat"
cd "$REPO_DIR"
mkdir -p out/slurm

echo "===== B200 Smoke Test ====="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Repo: $PWD"

echo "===== GPU visibility ====="
nvidia-smi || true
nvidia-smi -L || true

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# Ensure uv and dependencies are available in this job environment.
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d .venv ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

echo "===== Python/CUDA sanity ====="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    x = torch.randn(1024, 1024, device="cuda")
    y = torch.matmul(x, x)
    print("cuda matmul ok, mean:", float(y.mean().item()))
PY

echo "===== nanochat entrypoint sanity ====="
python -m scripts.research_compare --help >/dev/null
python -m scripts.base_train --help >/dev/null

echo "===== Smoke test finished successfully ====="
