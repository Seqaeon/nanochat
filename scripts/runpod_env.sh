#!/usr/bin/env bash
# ============================================================================
# RunPod environment for nanochat. Contains NO secrets; safe to commit.
#
#   source /workspace/nanochat/scripts/runpod_env.sh
#
# Make it permanent on a pod (once, not in git):
#   echo 'source /workspace/nanochat/scripts/runpod_env.sh' >> ~/.bashrc
#
# Why this file exists: RunPod gives you a small container disk mounted at /
# (enforced with an XFS project quota, which is why you get "Disk quota
# exceeded" rather than "No space left on device", and why df and the dashboard
# both look fine) and a large persistent volume at /workspace. Only /workspace
# survives a pod restart. Everything below points caches and outputs at the
# volume instead of the container disk.
# ============================================================================

# Root of the persistent volume. Override if your pod mounts it elsewhere.
: "${NANOCHAT_VOLUME:=/workspace}"

if [ ! -d "$NANOCHAT_VOLUME" ]; then
    echo "runpod_env: WARNING — $NANOCHAT_VOLUME does not exist." >&2
    echo "runpod_env: set NANOCHAT_VOLUME to your mount point and re-source." >&2
fi

# Warn if the volume is actually on the container disk, i.e. no volume attached.
if [ -d "$NANOCHAT_VOLUME" ] \
   && [ "$(stat -c %d / 2>/dev/null)" = "$(stat -c %d "$NANOCHAT_VOLUME" 2>/dev/null)" ]; then
    echo "runpod_env: WARNING — $NANOCHAT_VOLUME is on the same filesystem as /." >&2
    echo "runpod_env: no persistent volume is attached; you will hit the container" >&2
    echo "runpod_env: disk quota again. Attach a network volume to this pod." >&2
fi

export NANOCHAT_CACHE="$NANOCHAT_VOLUME/.cache"

# ── nanochat: datasets, tokenizer, checkpoints (see nanochat/common.py) ──────
export NANOCHAT_BASE_DIR="$NANOCHAT_VOLUME/nanochat/out"

# ── generic caches that otherwise land in ~ (container disk) ─────────────────
export XDG_CACHE_HOME="$NANOCHAT_CACHE"
export HF_HOME="$NANOCHAT_CACHE/huggingface"
export TORCH_HOME="$NANOCHAT_CACHE/torch"
export UV_CACHE_DIR="$NANOCHAT_CACHE/uv"
export PIP_CACHE_DIR="$NANOCHAT_CACHE/pip"

# ── torch.compile / Triton: large, and default to /tmp on the container disk ─
export TRITON_CACHE_DIR="$NANOCHAT_CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$NANOCHAT_CACHE/inductor"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1

# ── /tmp is on the container disk; dataloaders and ffmpeg can fill it ────────
export TMPDIR="$NANOCHAT_VOLUME/tmp"

mkdir -p "$NANOCHAT_BASE_DIR" "$NANOCHAT_CACHE" "$TMPDIR" \
         "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# ============================================================================
# CAVEAT: scripts/research_sweep.sh line 6 does an unconditional
#     export NANOCHAT_BASE_DIR="out"
# which overrides the value set above with a path relative to your working
# directory. Either launch sweeps from $NANOCHAT_VOLUME/nanochat, so that "out"
# resolves onto the volume, or change that line to
#     export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-out}"
# ============================================================================

# Secrets do NOT belong in this file. See scripts/runpod_env.sh notes in the
# README, or use RunPod's secrets feature. For the record, the training path
# needs no HuggingFace token: nanochat/dataset.py fetches the public ClimbMix
# shards over plain HTTPS with no auth header.

echo "runpod_env: NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo "runpod_env: caches -> $NANOCHAT_CACHE, TMPDIR=$TMPDIR"
