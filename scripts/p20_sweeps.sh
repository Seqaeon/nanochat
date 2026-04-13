#!/bin/bash
# Phase 20: Context-Conditioned Dynamic Weight Computation — Full Sweep (A–J)
# Each proposal runs independently against the dense baseline (model="base")
# "base" = standard dense transformer with Block + MLP
#
# Phase 1 proposals (A,B,C,D,F,H,I): train from scratch
# Phase 2 proposals (E,G,J): first pretrain base, then convert + continue
#
# Usage: bash scripts/p20_sweeps.sh

set -e

COMMON="--fp8 --max-shards 170 --models base \
  --device-batch-size 64 --use-onecycle 0 --log-every 200 --skip-core \
  --data-dir /root/nanochat/data --tokenizer-dir /root/nanochat/tokenizer \
  --sequence-len 2048 --mu-p-mode base_only --model-dim 128 \
  --modulation-diagnostics 1"
DEPTH=4

echo "======================================"
echo "  Phase 20 Full Sweep (A–J)"
echo "======================================"
echo "Baseline: base model, model_dim=128, depth=$DEPTH"
echo ""

# ─────────────────────────────────────────
# PHASE 1 PROPOSALS (train from scratch)
# ─────────────────────────────────────────

# 20A: Hash-Routed Column Selection (×4 scale)
echo "[1/10] Running 20A (HRCS, scale=4)..."
bash scripts/research_sweep.sh $COMMON \
  --p20-hrcs-scale 4 \
  $DEPTH

# 20B: LSH Weight Routing (×4 scale, 8 planes)
echo "[2/10] Running 20B (LSWR, scale=4, planes=8)..."
bash scripts/research_sweep.sh $COMMON \
  --p20-lswr-scale 4 --p20-lswr-planes 8 \
  $DEPTH

# 20C: Frozen Content-Routed Branches (K=4)
echo "[3/10] Running 20C (LRCFB, K=4)..."
bash scripts/research_sweep.sh $COMMON \
  --p20-lrcfb-branches 4 \
  $DEPTH

# 20D: Detached-Gradient Content Routing (K=4)
echo "[4/10] Running 20D (DGCR, K=4)..."
bash scripts/research_sweep.sh $COMMON \
  --p20-dgcr-branches 4 --p20-dgcr-aux-weight 0.01 \
  $DEPTH

# 20F: Mixture of Narrow Experts (K=4, same total params)
echo "[5/10] Running 20F (MoNE, K=4)..."
bash scripts/research_sweep.sh $COMMON \
  --p20-mone-experts 4 \
  $DEPTH

# 20H: Noise-Contrastive Expert Assignment (K=4)
echo "[6/10] Running 20H (NCEA, K=4, eps=0.1)..."
bash scripts/research_sweep.sh $COMMON \
  --p20-ncea-branches 4 --p20-ncea-eps 0.1 \
  $DEPTH

# 20I: Attention-Derived Weight Interpolation
echo "[7/10] Running 20I (ADWI)..."
bash scripts/research_sweep.sh $COMMON \
  --p20-adwi 1 \
  $DEPTH

# ─────────────────────────────────────────
# PHASE 2 PROPOSALS (pretrain then convert)
# ─────────────────────────────────────────
# These first train a base model (Phase 1), then convert the MLP weights
# and continue training. The convert_to_phase2() function in GPT handles
# the weight conversion automatically when the flags are set.
#
# NOTE: For Phase 2, the base model is trained with default settings first.
# The Phase 2 flags trigger conversion of trained MLP weights before
# optimizer setup. The research_sweep.sh runs a fresh training from random
# init, so E/G/J will train base→convert→continue in a single run.

# 20E: Progressive Weight Unfreezing (K=4 branches, Phase 2: router only)
echo "[8/10] Running 20E (PWU, K=4, phase=2)..."
bash scripts/research_sweep.sh $COMMON \
  --p20-pwu-branches 4 --p20-pwu-phase 2 \
  $DEPTH

# 20G: Frozen-SVD σ Gating
echo "[9/10] Running 20G (FSVD gate)..."
bash scripts/research_sweep.sh $COMMON \
  --p20-fsvd-gate 1 \
  $DEPTH

# 20J: Weight Bank Frozen Clustering (K=8 clusters, M=2 active)
echo "[10/10] Running 20J (WBFC, K=8, M=2)..."
bash scripts/research_sweep.sh $COMMON \
  --p20-wbfc-clusters 8 --p20-wbfc-active 2 \
  $DEPTH

echo ""
echo "======================================"
echo "  Phase 20 Sweep Complete (All 10)"
echo "======================================"
echo "Check sweep.log for results."
