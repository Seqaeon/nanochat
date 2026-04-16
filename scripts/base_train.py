"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
import torch
torch.set_num_threads(8)
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import contextmanager

# import wandb
import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 1000
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized, wrap_model
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
parser.add_argument("--data-dir", type=str, default=None, help="dataset parquet directory (default: nanochat.dataset.DATA_DIR)")
parser.add_argument("--checkpoints-dir", type=str, default=None, help="base checkpoint root directory (default: <base_dir>/base_checkpoints)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--parallel", type=str, default="ddp", choices=["ddp", "dp"], help="ddp: DistributedDataParallel (via torchrun), dp: nn.DataParallel (for Kaggle/notebooks)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--model-dim", type=int, default=0, help="explicit model_dim override (0 = use aspect-ratio)")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# Research branches
parser.add_argument("--use-moe", action="store_true", help="enable research MoE embedding branch")
parser.add_argument("--use-perm", action="store_true", help="use permutation MoE branch (only with --use-moe)")
parser.add_argument("--moe-num-experts", type=int, default=8, help="number of experts for research MoE branch")
parser.add_argument("--moe-router-dim", type=int, default=64, help="router dim for MoE embedding branch")
parser.add_argument("--moe-embed-dim", type=int, default=64, help="embedding dim for MoE branch (must match n_embd)")
parser.add_argument("--use-remix-linear", action="store_true", help="enable remixed linear blocks")
parser.add_argument("--remix-context-dim", type=int, default=64, help="context dim for remixed linear control")
parser.add_argument("--remix-basis-size", type=int, default=64, help="basis size for remixed linear")
parser.add_argument("--use-pos-embed", action="store_true", help="add learned absolute positional embeddings on top of token/research embeddings")
parser.add_argument("--moe-use-abs-pos-embed", type=int, default=0, choices=[0, 1], help="use learned absolute positional embeddings inside permutation MoE embeddings (1/0)")
parser.add_argument("--remix-use-basis-gate", type=int, default=1, choices=[0, 1], help="enable basis gating in remixed linear (1/0)")
parser.add_argument("--remix-use-output-gate", type=int, default=1, choices=[0, 1], help="enable output gating in remixed linear (1/0)")
parser.add_argument("--remix-use-context", type=int, default=1, choices=[0, 1], help="enable context modulation in remixed linear (1/0)")
# Phase 22: MoE-style overparameterized template mixing in RemixedLinear
parser.add_argument("--p22-n-templates", type=int, default=1, help="22: number of template_mixing matrices (1=standard, K>1=MoE routing)")
parser.add_argument("--p22-template-routing-learned", type=int, default=0, choices=[0, 1], help="22: learned template routing (0=frozen, 1=learned)")
parser.add_argument("--p22-attn-moe-route", type=str, default="none", choices=["none", "sequence", "token"], help="22: MoE routing for attention Q/K/V/Proj ('none'=off, 'sequence'=per-seq, 'token'=per-tok)")
# Phase 23: Tiny-Experts RemixedLinear + Standard MoE baseline
parser.add_argument("--p23-tiny-expert", type=int, default=0, choices=[0, 1], help="23: enable Tiny Experts mode in RemixedLinear (each expert_dim=basis_size//topk for compute parity)")
parser.add_argument("--p23-n-experts", type=int, default=64, help="23: total experts in the Tiny Expert bank (K_total)")
parser.add_argument("--p23-topk", type=int, default=16, help="23: active experts per forward pass (expert_dim=basis_size//topk); 0=soft all-expert routing")
parser.add_argument("--p23-learned-route", type=int, default=0, choices=[0, 1], help="23: learned routing projection in Tiny Expert (0=frozen, 1=learned)")
parser.add_argument("--p23-std-moe-experts", type=int, default=0, help="23: enable StandardMoE_MLP with K full-size experts (0=off)")
parser.add_argument("--p23-std-moe-topk", type=int, default=1, help="23: top-k active experts for StandardMoE_MLP (0=all/soft)")
parser.add_argument("--p23-std-moe-aux-weight", type=float, default=0.01, help="23: load-balance auxiliary loss weight for StandardMoE_MLP")
# CCL block modulation (only active when --use-remix-linear is set)

parser.add_argument("--cclblock-modulation", type=str, default="weight",
                    choices=["weight", "normalization", "householder", "spectral", "ocd", "lie", "polynomial", "grassmann", "decoupled", "tucker", "svs", "vq", "dcu", "fsi", "aesp", "ckr", "ckr_ffn", "com", "giad", "psg", "splitstream", "lokr", "pgr", "cil", "prb", "arg", "kfl"],
                    help="CCL block strategy")
parser.add_argument("--cclblock-orth-lambda", type=float, default=0.0,
                    help="OCD overlap penalty weight (0 disables)")
parser.add_argument("--cclblock-context-stream", type=str, default="local", 
                    choices=["local", "shifted", "ema", "selective", "multiscale", "ssm", "boundary", "chunk", "predictive_chunk", "evidence_ssm", "dacs", "prefix", "warmup_ema", "dacs_ema", "decay_prefix"],
                    help="Context stream: 'local', 'shifted', 'ema', 'selective', 'multiscale', 'ssm', 'boundary', 'chunk', 'predictive_chunk', 'evidence_ssm', 'dacs', 'prefix', 'warmup_ema', 'dacs_ema', 'decay_prefix'")
parser.add_argument("--cclblock-ema-factor", type=float, default=0.99,
                    help="EMA factor for the legacy EMAContextStream")
parser.add_argument("--cclblock-stale-ctx-lag", type=int, default=0,
                    help="Design C stale context lag (0=disabled)")
# Novel ablation designs
parser.add_argument("--cclblock-sparse-gate-k", type=int, default=0,
                    help="Design 3: sparse top-k basis gate (0=soft sigmoid, N=activate top-N basis functions)")
parser.add_argument("--cclblock-gate-temperature", type=float, default=1.0,
                    help="Design 6: basis gate temperature (<1=sharper, >1=softer, 1.0=standard sigmoid)")
parser.add_argument("--cclblock-context-bank-size", type=int, default=0,
                    help="Design 4: context prototype bank size (0=disabled, e.g. 16=16 learned prototypes)")
parser.add_argument("--cclblock-per-head-ctx", type=int, default=0, choices=[0, 1],
                    help="Design 7: separate ctx projections for attn vs ffn (0=off, 1=on)")
parser.add_argument("--cclblock-context-source", type=str, default="norm_x",
                    choices=["norm_x", "attn_heads", "attn_geometry"],
                    help="Context source for FFN gate/router ('norm_x', 'attn_heads', 'attn_geometry')")
# Phase 8: Boundary-Gated / Chunk Context / Auxiliary Objective
parser.add_argument("--cclblock-chunk-size", type=int, default=0,
                    help="Design 9: hard chunk pooling stride in tokens (0=off, e.g. 64)")
parser.add_argument("--cclblock-aux-objective", type=str, default="none",
                    choices=["none", "boundary", "entropy"],
                    help="Design 10: aux context objective ('none'=off, 'boundary'=boundary BCE, 'entropy'=difficulty MSE)")
parser.add_argument("--cclblock-aux-lambda", type=float, default=0.1,
                    help="Design 10: weight of auxiliary context loss (default 0.1)")
parser.add_argument("--cclblock-boundary-token-id", type=int, default=198,
                    help="Design 10: token ID for boundary detection (default 198=newline in many tokenizers)")
# Phase 9: RAL & FiLM
parser.add_argument("--use-ral", type=int, default=0, choices=[0, 1], help="Proposal A: Use ResidualAdaptiveLinear instead of RemixedLinear")
parser.add_argument("--ral-rank", type=int, default=32, help="Proposal A: Rank for the RAL context delta")
parser.add_argument("--cclblock-film-gate", type=int, default=0, choices=[0, 1], help="Proposal C: Use FiLM affine basis gate in RemixedLinear")
parser.add_argument("--cclblock-attn-shadow-dim", type=int, default=0, help="Dual-V shadow routing width (0=off)")
parser.add_argument("--cclblock-dynamic-ratio", type=float, default=0.25, help="Paradigm 1 (decoupled): fraction of channels routed to dynamic path")
parser.add_argument("--cclblock-gate-rank", type=int, default=8, help="Paradigm 1 (decoupled): low-rank context gate rank")
parser.add_argument("--cclblock-num-regimes", type=int, default=8, help="Paradigm 2 (evidence_ssm): number of latent regimes K")
parser.add_argument("--cclblock-regime-temperature", type=float, default=1.0, help="Paradigm 2 (evidence_ssm): softmax temperature over regimes")
parser.add_argument("--cclblock-poly-order", type=int, default=2)
parser.add_argument("--cclblock-lie-generators", type=int, default=4)
parser.add_argument("--cclblock-grassmann-bank-size", type=int, default=4)
parser.add_argument("--cclblock-tucker-rank", type=int, default=32)
parser.add_argument("--cclblock-tucker-modes", type=int, default=8)
parser.add_argument("--cclblock-svs-rank", type=int, default=64)
parser.add_argument("--cclblock-svs-eps", type=float, default=0.1)
parser.add_argument("--cclblock-vq-codes", type=int, default=8)
parser.add_argument("--cclblock-vq-temperature", type=float, default=1.0)
parser.add_argument("--cclblock-dcu-warmup-steps", type=int, default=0)
# Phase 12: FSI / AESP / CKR
parser.add_argument("--cclblock-fsi-rotations", type=int, default=8, help="FSI: number of frozen orthogonal rotations")
parser.add_argument("--cclblock-fsi-selector-dim", type=int, default=64, help="FSI: frozen routing projection dim")
parser.add_argument("--cclblock-aesp-strata", type=int, default=4, help="AESP: number of entropy strata")
parser.add_argument("--cclblock-aesp-delta-rank", type=int, default=4, help="AESP: rank of per-stratum low-rank deltas")
parser.add_argument("--cclblock-ckr-branches", type=int, default=4, help="CKR: number of parallel dense branches")
parser.add_argument("--cclblock-ckr-kernel-size", type=int, default=64, help="CKR: causal conv1d kernel size")
# Phase 13: CKR enhancements
parser.add_argument("--cclblock-ckr-pos-channels", type=int, default=1, help="CKR: multi-channel position signal (1=original, 3=multi-scale)")
parser.add_argument("--cclblock-ckr-dual-optim", type=int, default=0, choices=[0, 1], help="CKR: route gate params to dedicated conservative AdamW")
parser.add_argument("--cclblock-ckr-content-bias", type=float, default=0.0, help="CKR: frozen content hash bias scale (0=pure position)")
# Phase 14: Gradient-isolated content conditioning
parser.add_argument("--cclblock-giad-rank", type=int, default=32, help="GIAD: low-rank bottleneck dimension")
parser.add_argument("--cclblock-psg-kernel-size", type=int, default=64, help="PSG: causal conv kernel size")
parser.add_argument("--cclblock-ss-dynamic-ratio", type=float, default=0.25, help="SplitStream: dynamic channel fraction")
parser.add_argument("--cclblock-ss-branches", type=int, default=2, help="SplitStream: CKR branches on dynamic path")
parser.add_argument("--cclblock-ss-kernel-size", type=int, default=64, help="SplitStream: causal conv kernel size")
# Phase 15: LoKR + diagnostics
parser.add_argument("--cclblock-lokr-branches", type=int, default=8, help="LoKR: number of low-rank perturbation branches")
parser.add_argument("--cclblock-lokr-rank", type=int, default=16, help="LoKR: rank of each perturbation")
# Phase 16: CKR-Anneal / COM
parser.add_argument("--cclblock-ckr-temp-start", type=float, default=2.0, help="CKR-Anneal: initial softmax temperature")
parser.add_argument("--cclblock-ckr-temp-end", type=float, default=0.3, help="CKR-Anneal: final softmax temperature")
parser.add_argument("--cclblock-com-kernel-size", type=int, default=32, help="COM: causal output mixer kernel size")
# Phase 17: CKR enhancements + new architectures
parser.add_argument("--cclblock-ckr-ortho-init", type=int, default=0, choices=[0, 1], help="17D: orthogonal branch init")
parser.add_argument("--cclblock-ckr-branch-dropout", type=float, default=0.0, help="17E: branch dropout probability")
parser.add_argument("--cclblock-ckr-diversity-lambda", type=float, default=0.0, help="17H: branch diversity loss weight")
parser.add_argument("--cclblock-pgr-kernel-size", type=int, default=64, help="17C: PGR causal conv kernel size")
parser.add_argument("--cclblock-cil-kernel-size", type=int, default=64, help="17I: CIL causal conv kernel size")
parser.add_argument("--cclblock-prb-kernel-size", type=int, default=64, help="17J: PRB causal conv kernel size")
parser.add_argument("--modulation-diagnostics", type=int, default=0, choices=[0, 1], help="enable modulation layer diagnostics logging")
# Phase 18: Beyond CKR — orthogonal improvements
parser.add_argument("--p18-layer-drop", type=float, default=0.0, help="18E: stochastic depth drop probability (0=off)")
parser.add_argument("--p18-dynamic-activation", type=int, default=0, choices=[0, 1], help="18I: learned activation mix (ReLU²+GELU+SiLU)")
parser.add_argument("--p18-mixture-norm", type=int, default=0, choices=[0, 1], help="18H: learned RMSNorm+LayerNorm mixture")
parser.add_argument("--p18-aux-sim-lambda", type=float, default=0.0, help="18G: layer similarity penalty weight (0=off)")
parser.add_argument("--p18-gradient-penalty", type=float, default=0.0, help="18B: gradient penalty weight for Lipschitz regularization (0=off)")
parser.add_argument("--p18-per-channel-scale", type=int, default=0, choices=[0, 1], help="18F: learnable per-channel output scale")
# Phase 19: Zero-overhead indirect modulation
parser.add_argument("--p19-residual-gate", type=int, default=0, choices=[0, 1], help="19A: per-layer learned scalar on block output (0/1)")
parser.add_argument("--p19-head-importance", type=int, default=0, choices=[0, 1], help="19B: per-head learned scalar on attn output (0/1)")
parser.add_argument("--p19-residual-mix-groups", type=int, default=0, help="19C: grouped 1x1 conv between blocks (0=off, N=group_size)")
parser.add_argument("--p19-attn-logit-bias", type=int, default=0, choices=[0, 1], help="19D: per-head learned QK temperature (0/1)")
parser.add_argument("--p19-residual-decay", type=int, default=0, choices=[0, 1], help="19E: learned depth-dependent x0 decay (0/1)")
parser.add_argument("--p19-grad-equilibrium", type=float, default=0.0, help="19F: gradient equilibrium regularization lambda (0=off)")
parser.add_argument("--p19-spectral-reparam", type=int, default=0, choices=[0, 1, 2], help="19G: spectral reparameterization (0=off, 1=c_proj, 2=c_fc+c_proj)")
parser.add_argument("--p19-weight-anticollapse", type=float, default=0.0, help="19H: weight anti-collapse penalty lambda (0=off)")
parser.add_argument("--p19-ve-bias", type=int, default=0, choices=[0, 1], help="19I: add learnable bias to VE gate (0/1)")
parser.add_argument("--p19-weight-noise", type=float, default=0.0, help="19J: training-time weight perturbation epsilon (0=off)")
# Phase 20: Context-conditioned dynamic weight computation
parser.add_argument("--p20-hrcs-scale", type=int, default=0, help="20A: Hash-routed column selection (0=off, scale=D_stored/D_active)")
parser.add_argument("--p20-lswr-scale", type=int, default=0, help="20B: LSH weight routing (0=off, scale factor)")
parser.add_argument("--p20-lswr-planes", type=int, default=8, help="20B: number of LSH hash planes")
parser.add_argument("--p20-lrcfb-branches", type=int, default=0, help="20C: Content-routed branches (0=off, K=branches)")
parser.add_argument("--p20-lrcfb-narrow", type=int, default=0, choices=[0, 1], help="20C: narrow branches (0=full-size, 1=H//K param parity)")
parser.add_argument("--p20-lrcfb-learned", type=int, default=0, choices=[0, 1], help="20C: learned routing (0=frozen, 1=learnable)")
parser.add_argument("--p20-lrcfb-topk", type=int, default=0, help="20C: top-k sparse routing (0=soft/all)")
parser.add_argument("--p20-dgcr-branches", type=int, default=0, help="20D: Detached-gradient content-routed branches (0=off, K=branches)")
parser.add_argument("--p20-dgcr-aux-weight", type=float, default=0.01, help="20D: auxiliary routing loss weight")
parser.add_argument("--p20-mone-experts", type=int, default=0, help="20F: Mixture of Narrow Experts (0=off, K=num experts)")
parser.add_argument("--p20-mone-topk", type=int, default=0, help="20F: top-k expert routing (0=compute all, K=top-k sparse)")
parser.add_argument("--p20-mone-narrow", type=int, default=1, choices=[0, 1], help="20F: narrow experts (1=4D/K, 0=full 4D each)")
parser.add_argument("--p20-mone-frozen", type=int, default=0, choices=[0, 1], help="20F: frozen routing (0=learned, 1=frozen random proj)")
parser.add_argument("--p20-ncea-branches", type=int, default=0, help="20H: Noise-contrastive expert assignment (0=off, K=branches)")
parser.add_argument("--p20-ncea-eps", type=float, default=0.1, help="20H: perturbation magnitude")
parser.add_argument("--p20-adwi", type=int, default=0, choices=[0, 1], help="20I: Attention-derived weight interpolation (0=off, 1=on)")
# Phase 20 — Phase 2 proposals (require pre-trained checkpoint)
parser.add_argument("--p20-pwu-branches", type=int, default=0, help="20E: Progressive weight unfreezing (0=off, K=branches)")
parser.add_argument("--p20-pwu-phase", type=int, default=1, choices=[1, 2, 3], help="20E: training phase (1=pretrain, 2=router only, 3=joint)")
parser.add_argument("--p20-fsvd-gate", type=int, default=0, choices=[0, 1], help="20G: Frozen-SVD σ gating (0=off, 1=on)")
parser.add_argument("--p20-wbfc-clusters", type=int, default=0, help="20J: Weight bank frozen clustering (0=off, K=clusters)")
parser.add_argument("--p20-wbfc-active", type=int, default=0, help="20J: active clusters per token (0=auto K//4)")
# Phase 21: Pervasive Expert Routing
parser.add_argument("--p21-per-experts", type=int, default=0, help="21: MoELinear experts per layer (0=off, K=experts)")
parser.add_argument("--p21-per-topk", type=int, default=0, help="21: top-k routing (0=soft/all)")
parser.add_argument("--p21-per-learned", type=int, default=0, choices=[0, 1], help="21: learned routing (0=frozen, 1=learnable)")
parser.add_argument("--p21-per-attn", type=int, default=0, choices=[0, 1], help="21: also replace attention Q/K/V/O (0=MLP only, 1=all)")
# Fix 1A: per-layer context updaters
parser.add_argument("--use-layer-context", type=int, default=1, choices=[0, 1], help="per-layer context deltas for remix_linear: 1=enable (Fix 1A), 0=static base context")
parser.add_argument("--router-context-window", type=int, default=-1, help="sliding window size for GlobalContextManager (-1 for full)")
# Fix 1B: basis scaling
parser.add_argument("--scale-basis-size", type=int, default=1, choices=[0, 1], help="auto-scale RemixedLinear basis_size to max(basis_size, in_features//4) (Fix 1B)")
# Fix 1D: PermutationMoE expert mode
parser.add_argument("--perm-expert-mode", type=str, default="low_rank", choices=["full", "low_rank", "factored"], help="PermutationMoE expert mode: 'full' (original D×D), 'low_rank', or 'factored' (Fix 1D)")
parser.add_argument("--perm-rank", type=int, default=16, help="rank divisor for 'low_rank' mode or block size for 'factored' mode (Fix 1D)")
# Fix 4C: gradient clipping
parser.add_argument("--max-grad-norm", type=float, default=1.0, help="gradient norm clip threshold (-1 to disable, Fix 4C)")
# Fix 1H: PermutationMoE temperature scheduling
parser.add_argument("--perm-temp-start", type=float, default=5.0, help="initial PermutationMoE temperature (decays to 1.0 over first 50%% of training, Fix 1H)")
parser.add_argument("--research-onecycle", type=int, default=1, choices=[0, 1], help="for research runs: 1=use OneCycle LR schedule, 0=fallback to base warmup/flat/warmdown")
parser.add_argument("--use-onecycle", type=int, default=None, choices=[0, 1], help="alias for --research-onecycle")
parser.add_argument("--research-warmup-ratio", type=float, default=0.05, help="research-only warmup ratio/pct_start for OneCycle")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-tokens", type=int, default=-1, help="explicit number of tokens to train for (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=10.5, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.8, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.2, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding")
parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2 for embedding/unembedding")
parser.add_argument("--disable-mu-p", action="store_true",
                    help="disable μP-style LR scaling (model_dim/768)^-0.5 for AdamW params. "
                         "Use when sweeping absolute LRs directly for research models.")
parser.add_argument("--mu-p-scale-override", type=float, default=-1.0, help="force a specific mu-P scale")
parser.add_argument("--warmup-ratio", type=float, default=0.05, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.7, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = only at end, 0 = disable)")
parser.add_argument("--log-every", type=int, default=1, help="print step log to console every N steps")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = only at end, 0 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="enable/disable torch.compile")
parser.add_argument("--tokenizer-dir", type=str, default=None, help="explicit tokenizer directory (overrides default)")
parser.add_argument("--max-shards", type=int, default=-1, help="maximum number of dataset shards to use (-1 = all)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
parser.add_argument("--early-stop-tokens", type=int, default=-1, help="terminate training after this many tokens without affecting the LR schedule (-1 = disabled)")
parser.add_argument("--step-loss-file", type=str, default="", help="optional JSONL file to write per-step training loss for external sweep plotting")
args = parser.parse_args()
if args.use_onecycle is not None:
    args.research_onecycle = args.use_onecycle
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

# nn.DataParallel optimization (multi-GPU without torchrun)
is_dp = args.parallel == "dp" and device_type == "cuda" and torch.cuda.device_count() > 1
if is_dp:
    # When using DP, we act like a single world but with larger device_batch_size
    ddp_world_size = torch.cuda.device_count()
    ddp_rank = 0
    ddp_local_rank = 0
    print0(f"✓ Using nn.DataParallel (detected {ddp_world_size} GPUs)")
else:
    if args.parallel == "dp":
        print0(f"i DataParallel requested but suppressed: device_type={device_type}, gpu_count={torch.cuda.device_count()}")

master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
if is_dp:
    print0(f"DataParallel enabled: world_size={ddp_world_size}")

# wandb logging init
use_dummy_wandb = True # args.run == "dummy" or not master_process
wandb_run = DummyWandb() # if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Flash Attention status
from nanochat.flash_attention import USE_FA3
using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer(tokenizer_dir=args.tokenizer_dir)
token_bytes = get_token_bytes(device=device, tokenizer_dir=args.tokenizer_dir)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    if getattr(args, 'model_dim', 0) > 0:
        base_model_dim = args.model_dim
    else:
        base_dim = depth * args.aspect_ratio
        base_model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    base_num_heads = base_model_dim // args.head_dim
    if args.use_moe or args.use_remix_linear:
        model_dim = args.moe_embed_dim
        # Keep research branches compatible with repo attention constraints while
        # preferring a head count close to the base model's count.
        def _choose_research_heads(embed_dim: int, preferred_heads: int) -> int:
            pow2 = [1 << i for i in range(0, 12)]  # up to 2048 heads, way above practical usage
            valid_pow2 = [h for h in pow2 if h <= embed_dim and embed_dim % h == 0 and (embed_dim // h) % 8 == 0]
            if valid_pow2:
                return min(valid_pow2, key=lambda h: (abs(h - preferred_heads), -h))
            # Fallback: any divisor that keeps integer head_dim.
            valid_any = [h for h in range(1, embed_dim + 1) if embed_dim % h == 0]
            return min(valid_any, key=lambda h: abs(h - preferred_heads)) if valid_any else 1

        num_heads = _choose_research_heads(model_dim, base_num_heads)
        assert model_dim % num_heads == 0, f"moe_embed_dim must be divisible by n_head ({num_heads}), got {model_dim}"
    else:
        model_dim = base_model_dim
        num_heads = base_num_heads
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
        use_moe=args.use_moe,
        use_perm=args.use_perm,
        moe_num_experts=args.moe_num_experts,
        moe_router_dim=args.moe_router_dim,
        moe_embed_dim=args.moe_embed_dim,
        use_remix_linear=args.use_remix_linear,
        remix_context_dim=args.remix_context_dim,
        remix_basis_size=args.remix_basis_size,
        use_pos_embed=args.use_pos_embed,
        moe_use_abs_pos_embed=bool(args.moe_use_abs_pos_embed),
        remixed_linear_kwargs=dict(
            use_basis_gate=bool(args.remix_use_basis_gate),
            use_output_gate=bool(args.remix_use_output_gate),
            use_context=bool(args.remix_use_context),
            sparse_gate_k=getattr(args, 'cclblock_sparse_gate_k', 0),
            gate_temperature=getattr(args, 'cclblock_gate_temperature', 1.0),
            # n_templates = K_total (total experts in the bank):
            # When p23_tiny_expert=1, p23_n_experts sets K_total.
            # Otherwise falls back to p22_n_templates (legacy template bank).
            n_templates=(
                getattr(args, 'p23_n_experts', 64)
                if getattr(args, 'p23_tiny_expert', 0)
                else getattr(args, 'p22_n_templates', 1)
            ),
            template_routing_learned=bool(
                getattr(args, 'p23_learned_route', 0)
                if getattr(args, 'p23_tiny_expert', 0)
                else getattr(args, 'p22_template_routing_learned', 0)
            ),
        ),

        # Fix 1A
        use_layer_context=bool(getattr(args, 'use_layer_context', 1)),
        # Fix 1B
        scale_basis_size=bool(getattr(args, 'scale_basis_size', 1)),
        # Fix 1D
        perm_expert_mode=getattr(args, 'perm_expert_mode', 'low_rank'),
        perm_rank=getattr(args, 'perm_rank', 16),
        router_context_window=getattr(args, 'router_context_window', -1),
        # CCL block redesign
        cclblock_modulation=getattr(args, 'cclblock_modulation', 'weight'),
        cclblock_orth_lambda=getattr(args, 'cclblock_orth_lambda', 0.0),
        cclblock_context_stream=getattr(args, 'cclblock_context_stream', 'local'),
        cclblock_ema_factor=getattr(args, 'cclblock_ema_factor', 0.99),
        cclblock_stale_ctx_lag=getattr(args, 'cclblock_stale_ctx_lag', 0),
        # Novel ablation designs
        cclblock_sparse_gate_k=getattr(args, 'cclblock_sparse_gate_k', 0),
        cclblock_gate_temperature=getattr(args, 'cclblock_gate_temperature', 1.0),
        cclblock_context_bank_size=getattr(args, 'cclblock_context_bank_size', 0),
        cclblock_per_head_ctx=bool(getattr(args, 'cclblock_per_head_ctx', 0)),
        cclblock_context_source=getattr(args, 'cclblock_context_source', 'norm_x'),
        # Phase 8
        cclblock_chunk_size=getattr(args, 'cclblock_chunk_size', 0),
        cclblock_aux_objective=getattr(args, 'cclblock_aux_objective', 'none'),
        cclblock_aux_lambda=getattr(args, 'cclblock_aux_lambda', 0.1),
        cclblock_boundary_token_id=getattr(args, 'cclblock_boundary_token_id', 198),
        use_ral=bool(getattr(args, 'use_ral', 0)),
        ral_rank=getattr(args, 'ral_rank', 32),
        cclblock_film_gate=bool(getattr(args, 'cclblock_film_gate', 0)),
        cclblock_attn_shadow_dim=getattr(args, 'cclblock_attn_shadow_dim', 0),
        cclblock_dynamic_ratio=getattr(args, 'cclblock_dynamic_ratio', 0.25),
        cclblock_gate_rank=getattr(args, 'cclblock_gate_rank', 8),
        cclblock_num_regimes=getattr(args, 'cclblock_num_regimes', 8),
        cclblock_regime_temperature=getattr(args, 'cclblock_regime_temperature', 1.0),
        cclblock_poly_order=getattr(args, 'cclblock_poly_order', 2),
        cclblock_lie_generators=getattr(args, 'cclblock_lie_generators', 4),
        cclblock_grassmann_bank_size=getattr(args, 'cclblock_grassmann_bank_size', 4),
        cclblock_tucker_rank=getattr(args, 'cclblock_tucker_rank', 32),
        cclblock_tucker_modes=getattr(args, 'cclblock_tucker_modes', 8),
        cclblock_svs_rank=getattr(args, 'cclblock_svs_rank', 64),
        cclblock_svs_eps=getattr(args, 'cclblock_svs_eps', 0.1),
        cclblock_vq_codes=getattr(args, 'cclblock_vq_codes', 8),
        cclblock_vq_temperature=getattr(args, 'cclblock_vq_temperature', 1.0),
        cclblock_dcu_warmup_steps=getattr(args, 'cclblock_dcu_warmup_steps', 0),
        # Phase 12: FSI/AESP/CKR
        cclblock_fsi_rotations=getattr(args, 'cclblock_fsi_rotations', 8),
        cclblock_fsi_selector_dim=getattr(args, 'cclblock_fsi_selector_dim', 64),
        cclblock_aesp_strata=getattr(args, 'cclblock_aesp_strata', 4),
        cclblock_aesp_delta_rank=getattr(args, 'cclblock_aesp_delta_rank', 4),
        cclblock_ckr_branches=getattr(args, 'cclblock_ckr_branches', 4),
        cclblock_ckr_kernel_size=getattr(args, 'cclblock_ckr_kernel_size', 64),
        # Phase 13: CKR enhancements
        cclblock_ckr_pos_channels=getattr(args, 'cclblock_ckr_pos_channels', 1),
        cclblock_ckr_dual_optim=getattr(args, 'cclblock_ckr_dual_optim', 0),
        cclblock_ckr_content_bias=getattr(args, 'cclblock_ckr_content_bias', 0.0),
        # Phase 14: Gradient-isolated content conditioning
        cclblock_giad_rank=getattr(args, 'cclblock_giad_rank', 32),
        cclblock_psg_kernel_size=getattr(args, 'cclblock_psg_kernel_size', 64),
        cclblock_ss_dynamic_ratio=getattr(args, 'cclblock_ss_dynamic_ratio', 0.25),
        cclblock_ss_branches=getattr(args, 'cclblock_ss_branches', 2),
        cclblock_ss_kernel_size=getattr(args, 'cclblock_ss_kernel_size', 64),
        # Phase 15: LoKR
        cclblock_lokr_branches=getattr(args, 'cclblock_lokr_branches', 8),
        cclblock_lokr_rank=getattr(args, 'cclblock_lokr_rank', 16),
        # Phase 18: Beyond CKR
        p18_layer_drop=getattr(args, 'p18_layer_drop', 0.0),
        p18_dynamic_activation=getattr(args, 'p18_dynamic_activation', 0),
        p18_mixture_norm=getattr(args, 'p18_mixture_norm', 0),
        p18_aux_sim_lambda=getattr(args, 'p18_aux_sim_lambda', 0.0),
        p18_gradient_penalty=getattr(args, 'p18_gradient_penalty', 0.0),
        p18_per_channel_scale=getattr(args, 'p18_per_channel_scale', 0),
        # Phase 19: Zero-overhead indirect modulation
        p19_residual_gate=getattr(args, 'p19_residual_gate', 0),
        p19_head_importance=getattr(args, 'p19_head_importance', 0),
        p19_residual_mix_groups=getattr(args, 'p19_residual_mix_groups', 0),
        p19_attn_logit_bias=getattr(args, 'p19_attn_logit_bias', 0),
        p19_residual_decay=getattr(args, 'p19_residual_decay', 0),
        p19_grad_equilibrium=getattr(args, 'p19_grad_equilibrium', 0.0),
        p19_spectral_reparam=getattr(args, 'p19_spectral_reparam', 0),
        p19_weight_anticollapse=getattr(args, 'p19_weight_anticollapse', 0.0),
        p19_ve_bias=getattr(args, 'p19_ve_bias', 0),
        p19_weight_noise=getattr(args, 'p19_weight_noise', 0.0),
        # Phase 20: Context-conditioned dynamic weight computation
        p20_hrcs_scale=getattr(args, 'p20_hrcs_scale', 0),
        p20_lswr_scale=getattr(args, 'p20_lswr_scale', 0),
        p20_lswr_planes=getattr(args, 'p20_lswr_planes', 8),
        p20_lrcfb_branches=getattr(args, 'p20_lrcfb_branches', 0),
        p20_lrcfb_narrow=getattr(args, 'p20_lrcfb_narrow', 0),
        p20_lrcfb_learned=getattr(args, 'p20_lrcfb_learned', 0),
        p20_lrcfb_topk=getattr(args, 'p20_lrcfb_topk', 0),
        p20_dgcr_branches=getattr(args, 'p20_dgcr_branches', 0),
        p20_dgcr_aux_weight=getattr(args, 'p20_dgcr_aux_weight', 0.01),
        p20_mone_experts=getattr(args, 'p20_mone_experts', 0),
        p20_mone_topk=getattr(args, 'p20_mone_topk', 0),
        p20_mone_narrow=getattr(args, 'p20_mone_narrow', 1),
        p20_mone_frozen=getattr(args, 'p20_mone_frozen', 0),
        p20_ncea_branches=getattr(args, 'p20_ncea_branches', 0),
        p20_ncea_eps=getattr(args, 'p20_ncea_eps', 0.1),
        p20_adwi=getattr(args, 'p20_adwi', 0),
        # Phase 2 proposals
        p20_pwu_branches=getattr(args, 'p20_pwu_branches', 0),
        p20_pwu_phase=getattr(args, 'p20_pwu_phase', 1),
        p20_fsvd_gate=getattr(args, 'p20_fsvd_gate', 0),
        p20_wbfc_clusters=getattr(args, 'p20_wbfc_clusters', 0),
        p20_wbfc_active=getattr(args, 'p20_wbfc_active', 0),
        # Phase 21
        p21_per_experts=getattr(args, 'p21_per_experts', 0),
        p21_per_topk=getattr(args, 'p21_per_topk', 0),
        p21_per_learned=getattr(args, 'p21_per_learned', 0),
        p21_per_attn=getattr(args, 'p21_per_attn', 0),
        # Phase 22
        p22_attn_moe_route=getattr(args, 'p22_attn_moe_route', 'none'),
        # Phase 23: Tiny Expert RemixedLinear + Standard MoE baseline
        p23_tiny_expert=getattr(args, 'p23_tiny_expert', 0),
        p23_n_experts=getattr(args, 'p23_n_experts', 64),
        p23_topk=getattr(args, 'p23_topk', 16),
        p23_learned_route=getattr(args, 'p23_learned_route', 0),
        p23_std_moe_experts=getattr(args, 'p23_std_moe_experts', 0),
        p23_std_moe_topk=getattr(args, 'p23_std_moe_topk', 1),
        p23_std_moe_aux_weight=getattr(args, 'p23_std_moe_aux_weight', 0.01),
    )

    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

# Build the model, move to device, init the weights
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized

# Phase 17: Auto-enable modulation diagnostics for any research model
# (always on for research, no need for --modulation-diagnostics flag)
mod_diag = None
diag_metrics = None  # Track last-collected metrics for wandb logging
if args.use_remix_linear:
    from nanochat.gpt import ModulationDiagnostics
    mod_diag = ModulationDiagnostics(model)
    if mod_diag._layers:
        print0(f"Modulation diagnostics enabled: tracking {len(mod_diag._layers)} conditioning layers")
    else:
        print0(f"Modulation diagnostics: no position-conditioned layers found (mode={args.cclblock_modulation})")
        mod_diag = None

# If we are resuming, overwrite the model parameters with those of the checkpoint
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
if args.checkpoints_dir:
    checkpoints_root = os.path.abspath(args.checkpoints_dir)
else:
    # default to a "base_checkpoints" folder inside the nanochat base directory
    checkpoints_root = os.path.join(get_base_dir(), "base_checkpoints")

checkpoint_dir = os.path.abspath(os.path.join(checkpoints_root, output_dirname))
print0(f"Checkpoints directory: {checkpoint_dir}")
if args.step_loss_file and master_process:
    step_loss_dir = os.path.dirname(os.path.abspath(args.step_loss_file))
    if step_loss_dir:
        os.makedirs(step_loss_dir, exist_ok=True)
    # Fresh file per run.
    with open(args.step_loss_file, "w", encoding="utf-8"):
        pass
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
        args.fp8 = False
    else:
        # Check compute capability (requires 8.9+ for L4/4090 or 9.0+ for H100)
        major, minor = torch.cuda.get_device_capability()
        if major < 8 or (major == 8 and minor < 9):
            print0(f"Warning: FP8 training requires compute capability >= 8.9 (e.g. H100, L4, 4090), but detected {major}.{minor}. Disabling FP8.")
            args.fp8 = False

if args.fp8:
    # our custom fp8 is simpler than torchao, written for exact API compatibility
    from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
    # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    import torch.nn as nn

    # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
    def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
        if min(mod.in_features, mod.out_features) < 128:
            return False
        return True

    fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
    num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
    num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
    num_skipped = num_linear - num_fp8
    print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device=fp8_module.weight.device,
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = wrap_model(model, parallel_type=args.parallel, compile=args.compile, device=device)

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
param_counts = orig_model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token, num_active_flops_per_token = orig_model.estimate_flops()
print0(f"Estimated FLOPs per token (total):  {num_flops_per_token:e}")
print0(f"Estimated FLOPs per token (active): {num_active_flops_per_token:e}")


# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(orig_model)
if args.target_tokens > 0:
    target_tokens = args.target_tokens
else:
    target_tokens = int(args.target_param_data_ratio * num_scaling_params) # optimal tokens for the model we are about to train

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
d12_ref = build_model_meta(12) # creates the model on meta device
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# Phase 20 (E/G/J): Convert standard MLP to Phase 2 variant if flags are set
# This must happen AFTER weights are initialized/loaded but BEFORE optimizer setup
_p20_pwu = getattr(args, 'p20_pwu_branches', 0)
_p20_fsvd = getattr(args, 'p20_fsvd_gate', 0)
_p20_wbfc = getattr(args, 'p20_wbfc_clusters', 0)
if _p20_pwu > 0 or _p20_fsvd > 0 or _p20_wbfc > 0:
    n_converted = orig_model.convert_to_phase2()
    print0(f"Phase 2 conversion complete: {n_converted} MLP modules converted")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = orig_model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    adam_betas=(args.adam_beta1, args.adam_beta2),
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
    # μP
    disable_mu_p=args.disable_mu_p,
    mu_p_scale_override=args.mu_p_scale_override,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer,
    args.device_batch_size * (ddp_world_size if is_dp else 1),
    args.max_seq_len,
    split="train",
    device=device,
    resume_state_dict=dataloader_resume_state_dict,
    data_dir=args.data_dir,
    max_shards=args.max_shards,
)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer,
    args.device_batch_size * (ddp_world_size if is_dp else 1),
    args.max_seq_len,
    split="val",
    device=device,
    data_dir=args.data_dir,
    max_shards=args.max_shards,
)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Research branches use a OneCycle-style schedule; base keeps the original warmup/flat/warmdown schedule
use_research_mode = args.use_moe or args.use_perm or args.use_remix_linear
use_research_scheduler = use_research_mode and bool(args.research_onecycle)
if use_research_scheduler:
    print0("Using research scheduler: OneCycle-style LR multiplier")
elif use_research_mode:
    print0("Research mode with OneCycle disabled: using base warmup/flat/warmdown schedule")
else:
    print0("Using base scheduler: linear warmup/flat/linear warmdown")

# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac


def get_lr_multiplier_onecycle(it):
    """
    OneCycle-style multiplier in [final_lr_frac, 1.0].
    - rise phase: cosine from final_lr_frac -> 1.0
    - decay phase: cosine from 1.0 -> final_lr_frac
    Uses warmup_ratio as pct_start for the peak.
    """
    if num_iterations <= 1:
        return 1.0
    t = min(max(it, 0), num_iterations - 1)
    pct = t / (num_iterations - 1)
    warmup_src = args.warmup_ratio if args.research_warmup_ratio < 0 else args.research_warmup_ratio
    pct_start = min(max(warmup_src, 0.01), 0.99)
    low = args.final_lr_frac
    if pct <= pct_start:
        phase = pct / pct_start
        return low + (1.0 - low) * (1 - math.cos(math.pi * phase)) * 0.5
    phase = (pct - pct_start) / (1 - pct_start)
    return low + (1.0 - low) * (1 + math.cos(math.pi * phase)) * 0.5

# Momentum scheduler for Muon optimizer (warms up to 0.95 over the first 300 steps)
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# Weight decay scheduler for Muon optimizer (linearly decays to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)

# Fix 1H: PermutationMoE temperature scheduler
# Exponentially decays from perm_temp_start -> 1.0 over the first 50% of training
# Prevents early routing collapse from logit saturation and hard-argmax-like softmax behavior
def get_perm_temperature(it):
    t_start = args.perm_temp_start
    t_end = 1.0
    frac = min(it / max(num_iterations * 0.5, 1), 1.0)
    return t_start * (t_end / t_start) ** frac

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
effective_device_batch_size = args.device_batch_size * (ddp_world_size if is_dp else 1)
tokens_per_fwdbwd = effective_device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * (1 if is_dp else ddp_world_size) # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
EMA_BETA = 0.9

# Go!
while True:
    last_step = step == num_iterations # normal end

    # early stop: force this to be the last step if we hit the token limit
    if args.early_stop_tokens > 0 and step * total_batch_size >= args.early_stop_tokens:
        print0(f"[early stop] Reached {step * total_batch_size:,} tokens (limit: {args.early_stop_tokens:,}). Initiating final eval/save.")
        last_step = True

    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    do_eval = (args.eval_every > 0 and (last_step or step % args.eval_every == 0)) or (last_step and args.eval_every == -1)
    if do_eval:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model):
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if (args.core_metric_every != 0) and (last_step or (args.core_metric_every > 0 and step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            optimizer.state_dict(), # optimizer state
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # termination conditions
    if last_step:
        # Ensure sweep parsers always see an end-of-run loss line, even when
        # early-stop triggers before hitting a log_every boundary.
        debiased_at_step = smooth_train_loss / (1 - EMA_BETA**max(step, 1))
        print0(f"step {step:05d}/{num_iterations:05d} (final) | loss: {debiased_at_step:.6f} | early_stop: {int(args.early_stop_tokens > 0)}")
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        if is_dp:
            loss = loss.mean()
        train_loss = loss.detach() # for logging
        # 18B: Gradient penalty — weight Frobenius norm regularization
        # Penalizes large weight norms to constrain Lipschitz constant
        gp_lambda = float(getattr(args, 'p18_gradient_penalty', 0.0))
        if gp_lambda > 0:
            gp_loss = sum(
                p.float().square().mean() for p in model.parameters()
                if p.ndim >= 2 and p.requires_grad
            )
            loss = loss + gp_lambda * gp_loss
        # Phase 20: Auxiliary routing loss for DGCR (20D) and NCEA (20H)
        # These train the router to predict which branch is best, separate from main loss
        _p20_dgcr = getattr(args, 'p20_dgcr_branches', 0)
        _p20_ncea = getattr(args, 'p20_ncea_branches', 0)
        if _p20_dgcr > 0 or _p20_ncea > 0:
            aux_loss_total = torch.tensor(0.0, device=loss.device)
            for block in orig_model.transformer.h:
                if hasattr(block, 'mlp') and hasattr(block.mlp, 'compute_aux_loss'):
                    aux_loss_total = aux_loss_total + block.mlp.compute_aux_loss()
            if aux_loss_total.item() > 0:
                loss = loss + aux_loss_total
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # step the optimizer
    lrm = get_lr_multiplier_onecycle(step) if use_research_scheduler else get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    if scaler is not None:
        scaler.unscale_(optimizer)
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        # Fix 4C: gradient clipping before optimizer step (protects against large gradients
        # in adaptive gate pathways early in training, e.g. PermutationMoE, RemixedLinear)
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(orig_model.parameters(), args.max_grad_norm)
        # 19F: Gradient Equilibrium Regularization (gradient modifier)
        # Equalizes per-block gradient norms to prevent gradient starvation/domination
        ger_lambda = float(getattr(args, 'p19_grad_equilibrium', 0.0))
        if ger_lambda > 0:
            with torch.no_grad():
                block_grad_norms = []
                block_key_params = []
                for block in orig_model.transformer.h:
                    mlp = block.mlp if hasattr(block, 'mlp') else (block.ffwd if hasattr(block, 'ffwd') else None)
                    if mlp is not None and hasattr(mlp, 'c_fc') and mlp.c_fc.weight.grad is not None:
                        gn = mlp.c_fc.weight.grad.float().norm()
                        block_grad_norms.append(gn)
                        block_key_params.append(mlp.c_fc.weight)
                if len(block_grad_norms) >= 2:
                    gn_stack = torch.stack(block_grad_norms)
                    gn_mean = gn_stack.mean()
                    # Correction factor: scale each layer's gradients toward the mean
                    for idx, (gn, param) in enumerate(zip(block_grad_norms, block_key_params)):
                        if gn > 1e-12:
                            correction = (gn_mean / gn).clamp(1.0 - ger_lambda, 1.0 + ger_lambda)
                            param.grad.mul_(correction.to(param.grad.dtype))
        optimizer.step()
    # Fix 1H: update PermutationMoE temperature each step
    if use_research_mode:
        perm_temp = get_perm_temperature(step)
        for module in orig_model.modules():
            if hasattr(module, 'temperature') and isinstance(getattr(module, 'temperature'), torch.Tensor):
                module.temperature.fill_(perm_temp)
    # Phase 16A: CKR temperature annealing (exponential: temp_start → temp_end)
    if getattr(args, 'cclblock_modulation', '') in ('ckr', 'ckr_ffn'):
        temp_start = getattr(args, 'cclblock_ckr_temp_start', 1.0)
        temp_end = getattr(args, 'cclblock_ckr_temp_end', 1.0)
        if temp_start != temp_end and num_iterations > 1:
            progress = min(step / (num_iterations - 1), 1.0)
            current_temp = temp_start * (temp_end / temp_start) ** progress
            from nanochat.gpt import CausalKernelLinear
            for module in orig_model.modules():
                if isinstance(module, CausalKernelLinear):
                    module._temperature.fill_(current_temp)
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    smooth_train_loss = EMA_BETA * smooth_train_loss + (1 - EMA_BETA) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - EMA_BETA**(step + 1)) # debias the EMA
    if args.step_loss_file and master_process:
        with open(args.step_loss_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "step": int(step),
                "tokens": int(step * total_batch_size),
                "loss": float(debiased_smooth_loss),
            }) + "\n")
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    adamw_lrs = [g["lr"] for g in optimizer.param_groups if g.get("kind") == "adamw"]
    muon_lrs = [g["lr"] for g in optimizer.param_groups if g.get("kind") == "muon"]
    lr_msg = f"lr(adamw:{(sum(adamw_lrs)/len(adamw_lrs)) if adamw_lrs else 0:.3e}, muon:{(sum(muon_lrs)/len(muon_lrs)) if muon_lrs else 0:.3e})"
    if step % args.log_every == 0 or step == num_iterations - 1 or last_step:
        print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | {lr_msg} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
        # Phase 17: Modulation diagnostics at log intervals
        if mod_diag is not None:
            diag_metrics = mod_diag.collect()
            if diag_metrics:
                print0(mod_diag.format(diag_metrics))
                if master_process:
                    diag_file = os.path.join(checkpoint_dir, "modulation_diagnostics.jsonl")
                    mod_diag.save_to_file(diag_metrics, step, diag_file)
            else:
                diag_metrics = None
            # Phase 19: Expanded diagnostics (always collected, even without CKR layers)
            p19_metrics = mod_diag.collect_p19(orig_model)
            if p19_metrics:
                p19_log = mod_diag.format_p19(p19_metrics)
                if p19_log:
                    print0(p19_log)
            # Phase 20: Dynamic weight computation diagnostics
            p20_metrics = mod_diag.collect_p20(orig_model)
            if p20_metrics:
                p20_log = mod_diag.format_p20(p20_metrics)
                if p20_log:
                    print0(p20_log)
        else:
            p19_metrics = None
            p20_metrics = None
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        # Add modulation diagnostics to wandb if available
        if mod_diag is not None and diag_metrics is not None:
            log_data.update(mod_diag.to_dict(diag_metrics))
        # Phase 19: expanded diagnostics
        if mod_diag is not None and p19_metrics:
            log_data.update(mod_diag.to_dict_p19(p19_metrics))
        # Phase 20: dynamic weight diagnostics
        if mod_diag is not None and p20_metrics:
            log_data.update(mod_diag.to_dict_p20(p20_metrics))
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# Log to report
from nanochat.report import get_report
section_name = "Base model training"
if args.model_tag:
    section_name += f" ({args.model_tag})"
get_report().log(section=section_name, data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": args.warmup_ratio,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
        "research_warmup_ratio": args.research_warmup_ratio,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
