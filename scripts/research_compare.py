import argparse
import sys
import os
import json
import glob
import shutil
from pathlib import Path
import subprocess

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scripts._sweep_utils import resolve_runner, estimate_tokens_from_base, model_dims, check_and_prepare_env
from nanochat.checkpoint_manager import find_last_step


RUNNER = resolve_runner()

def run_training_sweep(args):
    # Ensure environment is ready
    check_and_prepare_env(args)
    
    depth = args.depth
    run_dir = args.run_dir
    run_dir_path = Path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)
    
    target_tokens = args.target_tokens if args.target_tokens and args.target_tokens > 0 else estimate_tokens_from_base(depth, tokenizer_dir=args.tokenizer_dir)
    print("=" * 64)
    print(f"Starting Sweep for Depth {depth}")
    print(f"Calculated Target Tokens: {target_tokens:,}")
    print("=" * 64)
    
    aspect_ratio, head_dim, model_dim, target_dim = model_dims(depth)
    if getattr(args, "research_dim", None) and args.research_dim > 0:
        print(f"  Overriding default target_dim ({target_dim}) with --research-dim {args.research_dim}")
        target_dim = args.research_dim
    max_seq_len = args.sequence_len
    
    device_batch_size = args.device_batch_size if args.device_batch_size > 0 else {4: 8, 8: 32, 16: 16, 24: 8}.get(depth, 16)
    total_batch_size = args.total_batch_size if args.total_batch_size > 0 else 262144
    eval_every = args.eval_every
    log_every = args.log_every
    
    warm_up_ratio = args.warmup_ratio
    adam_beta2 = 0.99
    
    # Common kwargs for all models
    common_args = [
        "--depth", str(depth),
        "--aspect-ratio", str(aspect_ratio),
        "--head-dim", str(head_dim),
        "--max-seq-len", str(max_seq_len),
        "--device-batch-size", str(device_batch_size),
        "--total-batch-size", str(total_batch_size), # standard for reference
        "--target-tokens", str(target_tokens),
        "--eval-every", str(eval_every),        
        "--log-every", str(log_every),
        "--core-metric-every", "0" if args.skip_core else str(args.core_metric_every),
        "--save-every", str(args.save_every),
        "--warmup-ratio", str(warm_up_ratio),    # Safer for research models
        "--adam-beta2", str(adam_beta2),     # Matches notebook
        "--research-warmup-ratio", str(args.research_warmup_ratio),
        "--use-onecycle", str(args.use_onecycle),
        "--router-context-window", str(args.router_context_window),
        "--remix-use-basis-gate", str(getattr(args, 'remix_use_basis_gate', 1)),
        "--remix-use-output-gate", str(getattr(args, 'remix_use_output_gate', 1)),
        "--remix-use-context", str(getattr(args, 'remix_use_context', 1)),
        "--cclblock-modulation", str(args.cclblock_modulation),
        "--cclblock-orth-lambda", str(getattr(args, 'cclblock_orth_lambda', 0.0)),
        "--cclblock-context-stream", str(args.cclblock_context_stream),
        "--cclblock-ema-factor", str(args.cclblock_ema_factor),
        "--cclblock-stale-ctx-lag", str(args.cclblock_stale_ctx_lag),
        # Novel ablation designs
        "--cclblock-sparse-gate-k", str(getattr(args, 'cclblock_sparse_gate_k', 0)),
        "--cclblock-gate-temperature", str(getattr(args, 'cclblock_gate_temperature', 1.0)),
        "--cclblock-context-bank-size", str(getattr(args, 'cclblock_context_bank_size', 0)),
        "--cclblock-per-head-ctx", str(getattr(args, 'cclblock_per_head_ctx', 0)),
        "--cclblock-context-source", str(getattr(args, 'cclblock_context_source', 'norm_x')),
        # Phase 8
        "--cclblock-chunk-size",        str(getattr(args, 'cclblock_chunk_size', 0)),
        "--cclblock-aux-objective",     str(getattr(args, 'cclblock_aux_objective', 'none')),
        "--cclblock-aux-lambda",        str(getattr(args, 'cclblock_aux_lambda', 0.1)),
        "--cclblock-boundary-token-id", str(getattr(args, 'cclblock_boundary_token_id', 198)),
        "--use-ral", str(getattr(args, 'use_ral', 0)),
        "--ral-rank", str(getattr(args, 'ral_rank', 32)),
        "--cclblock-film-gate", str(getattr(args, 'cclblock_film_gate', 0)),
        "--cclblock-attn-shadow-dim", str(getattr(args, 'cclblock_attn_shadow_dim', 0)),
        "--cclblock-dynamic-ratio", str(getattr(args, 'cclblock_dynamic_ratio', 0.25)),
        "--cclblock-gate-rank", str(getattr(args, 'cclblock_gate_rank', 8)),
        "--cclblock-num-regimes", str(getattr(args, 'cclblock_num_regimes', 8)),
        "--cclblock-regime-temperature", str(getattr(args, 'cclblock_regime_temperature', 1.0)),
        "--cclblock-poly-order", str(getattr(args, 'cclblock_poly_order', 2)),
        "--cclblock-lie-generators", str(getattr(args, 'cclblock_lie_generators', 4)),
        "--cclblock-grassmann-bank-size", str(getattr(args, 'cclblock_grassmann_bank_size', 4)),
        "--cclblock-tucker-rank", str(getattr(args, 'cclblock_tucker_rank', 32)),
        "--cclblock-tucker-modes", str(getattr(args, 'cclblock_tucker_modes', 8)),
        "--cclblock-svs-rank", str(getattr(args, 'cclblock_svs_rank', 64)),
        "--cclblock-svs-eps", str(getattr(args, 'cclblock_svs_eps', 0.1)),
        "--cclblock-vq-codes", str(getattr(args, 'cclblock_vq_codes', 8)),
        "--cclblock-vq-temperature", str(getattr(args, 'cclblock_vq_temperature', 1.0)),
        "--cclblock-dcu-warmup-steps", str(getattr(args, 'cclblock_dcu_warmup_steps', 0)),
        # Phase 12: FSI/AESP/CKR
        "--cclblock-fsi-rotations", str(getattr(args, 'cclblock_fsi_rotations', 8)),
        "--cclblock-fsi-selector-dim", str(getattr(args, 'cclblock_fsi_selector_dim', 64)),
        "--cclblock-aesp-strata", str(getattr(args, 'cclblock_aesp_strata', 4)),
        "--cclblock-aesp-delta-rank", str(getattr(args, 'cclblock_aesp_delta_rank', 4)),
        "--cclblock-ckr-branches", str(getattr(args, 'cclblock_ckr_branches', 4)),
        "--cclblock-ckr-kernel-size", str(getattr(args, 'cclblock_ckr_kernel_size', 64)),
        # Phase 13: CKR enhancements
        "--cclblock-ckr-pos-channels", str(getattr(args, 'cclblock_ckr_pos_channels', 1)),
        "--cclblock-ckr-dual-optim", str(getattr(args, 'cclblock_ckr_dual_optim', 0)),
        "--cclblock-ckr-content-bias", str(getattr(args, 'cclblock_ckr_content_bias', 0.0)),
        # Phase 14: GIAD/PSG/SplitStream
        "--cclblock-giad-rank", str(getattr(args, 'cclblock_giad_rank', 32)),
        "--cclblock-psg-kernel-size", str(getattr(args, 'cclblock_psg_kernel_size', 64)),
        "--cclblock-ss-dynamic-ratio", str(getattr(args, 'cclblock_ss_dynamic_ratio', 0.25)),
        "--cclblock-ss-branches", str(getattr(args, 'cclblock_ss_branches', 2)),
        "--cclblock-ss-kernel-size", str(getattr(args, 'cclblock_ss_kernel_size', 64)),
        # Phase 15: LoKR
        "--cclblock-lokr-branches", str(getattr(args, 'cclblock_lokr_branches', 8)),
        "--cclblock-lokr-rank", str(getattr(args, 'cclblock_lokr_rank', 16)),
        # Phase 16: CKR-Anneal / COM
        "--cclblock-ckr-temp-start", str(getattr(args, 'cclblock_ckr_temp_start', 2.0)),
        "--cclblock-ckr-temp-end", str(getattr(args, 'cclblock_ckr_temp_end', 0.3)),
        "--cclblock-com-kernel-size", str(getattr(args, 'cclblock_com_kernel_size', 32)),
        # Phase 17: CKR enhancements + new architectures
        "--cclblock-ckr-ortho-init", str(getattr(args, 'cclblock_ckr_ortho_init', 0)),
        "--cclblock-ckr-branch-dropout", str(getattr(args, 'cclblock_ckr_branch_dropout', 0.0)),
        "--cclblock-ckr-diversity-lambda", str(getattr(args, 'cclblock_ckr_diversity_lambda', 0.0)),
        "--cclblock-pgr-kernel-size", str(getattr(args, 'cclblock_pgr_kernel_size', 64)),
        "--cclblock-cil-kernel-size", str(getattr(args, 'cclblock_cil_kernel_size', 64)),
        "--cclblock-prb-kernel-size", str(getattr(args, 'cclblock_prb_kernel_size', 64)),
        # Phase 18: Beyond CKR
        "--p18-layer-drop", str(getattr(args, 'p18_layer_drop', 0.0)),
        "--p18-dynamic-activation", str(getattr(args, 'p18_dynamic_activation', 0)),
        "--p18-mixture-norm", str(getattr(args, 'p18_mixture_norm', 0)),
        "--p18-aux-sim-lambda", str(getattr(args, 'p18_aux_sim_lambda', 0.0)),
        "--p18-gradient-penalty", str(getattr(args, 'p18_gradient_penalty', 0.0)),
        "--p18-per-channel-scale", str(getattr(args, 'p18_per_channel_scale', 0)),
    ]
    if args.compile:
        common_args.append("--compile")
    else:
        common_args.append("--no-compile")
    if getattr(args, "fp8", False):
        common_args.append("--fp8")
    if getattr(args, "tokenizer_dir", None):
        common_args.extend(["--tokenizer-dir", args.tokenizer_dir])
    if getattr(args, "data_dir", None):
        common_args.extend(["--data-dir", args.data_dir])
    if getattr(args, "max_shards", -1) != -1:
        common_args.extend(["--max-shards", str(args.max_shards)])
    
    # --- Optimal LR Configurations (from actual_lr_research_sweep) ---
    BEST_LRS = {
        "moe_no_perm": {
            "embedding_lr":   0.104074,
            "unembedding_lr": 0.0245175,
            "matrix_lr":      0.0329274,
            "scalar_lr":      0.152507,
        },
        "moe_perm": {
            "embedding_lr":   0.104074,
            "unembedding_lr": 0.0245175,
            "matrix_lr":      0.0329274,
            "scalar_lr":      0.152507,
        },
        "remixed-linear": {
            "embedding_lr":   0.104074,
            "unembedding_lr": 0.0245175,
            "matrix_lr":      0.0329274,
            "scalar_lr":      0.152507,
        },
    }
    # ---------------------------------------------------------------

    models = {
        "base": [], # Base model relies on base_train.py defaults
    }

    # Research branch configurations (architecture only)
    research_configs = {
        "moe_no_perm": [
            "--use-moe",
            "--moe-num-experts", "8",
            "--moe-router-dim", str(target_dim),
            "--moe-embed-dim",  str(target_dim),
        ],
        "moe_perm": [
            "--use-moe",
            "--use-perm",
            "--moe-num-experts", "8",
            "--moe-router-dim", str(target_dim),
            "--moe-embed-dim",  str(target_dim),
        ],
        "remixed-linear": [
            "--use-remix-linear",
            "--moe-embed-dim",    str(target_dim),
            "--moe-router-dim",   str(target_dim),
            "--remix-context-dim", str(target_dim),
            "--remix-basis-size",  str(target_dim),
        ]
    }

    for name, flags in research_configs.items():
        lrs = BEST_LRS[name]
        models[name] = flags + [
            "--embedding-lr",   str(lrs["embedding_lr"]),
            "--unembedding-lr", str(lrs["unembedding_lr"]),
            "--matrix-lr",      str(lrs["matrix_lr"]),
            "--scalar-lr",      str(lrs["scalar_lr"]),
        ]
    
    # Filter models if requested
    target_models = args.models.split(",") if args.models != "all" else models.keys()
    filtered_models = {k: v for k, v in models.items() if k in target_models}
    if not filtered_models:
        print(f"No matching models found in {list(models.keys())} for selection '{args.models}'")
        return

    results = {}
    
    for model_name, extra_args in filtered_models.items():
        print(f"\n--- Training {model_name} ---")
        
        ckpt_dir = (run_dir_path / f"ckpt_{model_name}").resolve()

        train_cmd_args = common_args + extra_args + [
            "--checkpoints-dir", str(ckpt_dir),
            "--model-tag", model_name
        ]
        
        # Handle mu-P scaling based on the new mode system
        if args.mu_p_mode == "disable":
            if model_name != "base":
                train_cmd_args.append("--disable-mu-p")
        elif args.mu_p_mode == "base_only":
            if model_name != "base":
                # Force research models to use the exact same multiplier base would've used
                base_multiplier = (model_dim / 768) ** -0.5
                train_cmd_args.extend(["--mu-p-scale-override", str(base_multiplier)])
        # if "enable", neither flag is passed; models calculate their own inherently
            
        # Check for resumption
        actual_model_ckpt_dir = ckpt_dir / model_name
        try:
            last_step = find_last_step(str(actual_model_ckpt_dir))
            print(f"  Found existing checkpoints in {actual_model_ckpt_dir}, resuming from step {last_step}")
            train_cmd_args.extend(["--resume-from-step", str(last_step)])
        except FileNotFoundError:
            pass # Starting fresh
        
        # Need to preserve environment variables, especially LD_LIBRARY_PATH for cusparseLt
        env = os.environ.copy()

        # Each model is trained as a proper DDP job via torchrun.
        cmd = RUNNER + ["-m", "scripts.base_train"] + train_cmd_args
        print(f"Running: {' '.join(cmd)}")
        
        try:
            # We stream stdout so user isn't stuck waiting blindly
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    print(line, end="", flush=True)
            process.communicate()
            
            if process.returncode != 0:
                print(f"Error training {model_name}. Skipping to next.")
                continue
                
            # Extract final checkpoint val_bpb
            # Checkpoint format is usually checkpoints_dir/model/state_*.pt etc
            # Base train saves it with step index. We find the largest one.
            model_ckpt_dir = ckpt_dir / "model"
            if model_ckpt_dir.exists():
                meta_files = glob.glob(str(model_ckpt_dir / "meta_*.json"))
                if meta_files:
                    meta_files.sort()
                    last_meta = meta_files[-1]
                    try:
                        with open(last_meta, "r") as f:
                            meta_data = json.load(f)
                        if "val_bpb" in meta_data and meta_data["val_bpb"] is not None:
                            val_bpb = float(meta_data["val_bpb"])
                            results[model_name] = {"val_bpb": val_bpb, "checkpoint": last_meta}
                            print(f"Final Validation BPB for {model_name}: {val_bpb:.4f}")
                        else:
                            print(f"No val_bpb found in {last_meta}")
                    except Exception as e:
                        print(f"Failed to load metadata {last_meta}: {e}")
                else:
                    print(f"No meta_*.json files found in {model_ckpt_dir}")
            else:
                 print(f"Checkpoint directory {model_ckpt_dir} does not exist.")
                 
        except Exception as e:
            print(f"Exception during {model_name}: {e}")
            
    # --- Generate Report and Plot ---
    if not results:
        print("No results collected to plot.")
        return
        
    print("\n--- Generating Report ---")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    names = list(results.keys())
    # Ensure all collected BPBs are floats for math
    bpbs = [float(results[n]["val_bpb"]) for n in names]
    
    bars = plt.bar(names, bpbs, color=sns.color_palette("husl", len(names)))
    
    plt.title(f"Validation BPB Comparison at Depth {depth} ({target_tokens:,} tokens)", fontsize=14)
    plt.ylabel("Validation Bits Per Byte (lower is better)", fontsize=12)
    plt.ylim(float(min(bpbs)) * 0.95, float(max(bpbs)) * 1.05) # Zoom in for better contrast

    # Add exact values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center', fontsize=10)
        
    plt.tight_layout()
    plot_path = run_dir_path / f"comparison_depth_{depth}.png"
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    
    # Save TSV data
    tsv_path = run_dir_path / f"results_depth_{depth}.tsv"
    with open(tsv_path, "w") as f:
        f.write("model_name\tval_bpb\n")
        for name, data in results.items():
            f.write(f"{name}\t{data['val_bpb']}\n")
    print(f"Saved TSV data to {tsv_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--fp8", action="store_true", help="Enable FP8 training (Blackwell optimization)")
    parser.add_argument("--tokenizer-dir", type=str, default=None, help="explicit tokenizer directory")
    parser.add_argument("--data-dir", type=str, default=None, help="explicit data directory")
    parser.add_argument("--max-shards", type=int, default=-1, help="maximum number of dataset shards to use")
    parser.add_argument("--target-tokens", type=int, default=-1, help="explicit number of tokens to train for per model")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="enable/disable torch.compile")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="base warmup ratio passed to all runs")
    parser.add_argument("--models", type=str, default="all", help="Comma-separated list of models to run (e.g. 'base,remixed-linear'), or 'all'")
    parser.add_argument("--research-warmup-ratio", type=float, default=0.05, help="research-branch warmup ratio for OneCycle")
    parser.add_argument("--use-onecycle", type=int, default=1, choices=[0, 1], help="research branches: 1=OneCycle, 0=use base schedule")
    
    # New flags for run configuration
    parser.add_argument("--device-batch-size", type=int, default=-1, help="override per-device batch size")
    parser.add_argument("--total-batch-size", type=int, default=-1, help="override total batch size")
    parser.add_argument("--log-every", type=int, default=1, help="logging frequency")
    parser.add_argument("--eval-every", type=int, default=-1, help="evaluation frequency (-1 = at end)")
    parser.add_argument("--save-every", type=int, default=-1, help="checkpoint frequency")
    parser.add_argument("--core-metric-every", type=int, default=-1, help="core metric frequency")
    parser.add_argument("--skip-core", action="store_true", help="completely disable CORE metric evaluation")
    parser.add_argument("--mu-p-mode", type=str, default="base_only", choices=["disable", "base_only", "enable"], help="mu-P scaling logic")
    parser.add_argument("--sequence-len", type=int, default=2048, help="override max sequence length")
    parser.add_argument("--router-context-window", type=int, default=-1, help="override sliding window size for contextual router (-1 for full sequence)")
    # Research dimension override
    parser.add_argument("--research-dim", type=int, default=0, help="override default 1/8th model_dim for research branches (MoE/Remix)")
    # Remixed-linear components
    parser.add_argument("--remix-use-basis-gate", type=int, default=1, choices=[0, 1], help="enable basis gating in remixed linear (1/0)")
    parser.add_argument("--remix-use-output-gate", type=int, default=1, choices=[0, 1], help="enable output gating in remixed linear (1/0)")
    parser.add_argument("--remix-use-context", type=int, default=1, choices=[0, 1], help="enable context modulation in remixed linear (1/0)")
    # CCL block modulation
    parser.add_argument("--cclblock-modulation", type=str, default="weight",
                        choices=["weight", "normalization", "householder", "spectral", "ocd", "lie", "polynomial", "grassmann", "decoupled", "tucker", "svs", "vq", "dcu", "fsi", "aesp", "ckr", "ckr_ffn", "com", "giad", "psg", "splitstream", "lokr", "pgr", "cil", "prb", "arg", "kfl"],
                        help="CCL block strategy: 'weight' (RemixedLinear+SelectiveContextStream) "
                             "or 'normalization' (CCLBlock with AdaRMSNorm)")
    parser.add_argument("--cclblock-orth-lambda", type=float, default=0.0,
                        help="OCD overlap penalty weight (0 disables)")
    parser.add_argument("--cclblock-context-stream", type=str, default="local", 
                        choices=["local", "shifted", "ema", "selective", "multiscale", "ssm", "boundary", "chunk", "predictive_chunk", "evidence_ssm", "dacs", "prefix", "warmup_ema", "dacs_ema", "decay_prefix"],
                        help="Context stream type")
    parser.add_argument("--cclblock-ema-factor", type=float, default=0.99,
                        help="EMA factor for the legacy EMAContextStream")
    parser.add_argument("--cclblock-stale-ctx-lag", type=int, default=0,
                        help="Design C stale context lag (0=disabled, k>=1 = context from k blocks ago)")
    # Novel ablation designs
    parser.add_argument("--cclblock-sparse-gate-k", type=int, default=0,
                        help="Design 3: sparse top-k basis gate (0=off, N=top-N)")
    parser.add_argument("--cclblock-gate-temperature", type=float, default=1.0,
                        help="Design 6: gate temperature (<1=sharper, >1=softer)")
    parser.add_argument("--cclblock-context-bank-size", type=int, default=0,
                        help="Design 4: context prototype bank size (0=off, e.g. 16)")
    parser.add_argument("--cclblock-per-head-ctx", type=int, default=0, choices=[0, 1],
                        help="Design 7: separate attn/ffn context projections (0=off, 1=on)")
    parser.add_argument("--cclblock-context-source", type=str, default="norm_x",
                        choices=["norm_x", "attn_heads", "attn_geometry"],
                        help="Design 2: context source ('norm_x'=residual, 'attn_heads'=query vectors)")
    # Phase 8
    parser.add_argument("--cclblock-chunk-size", type=int, default=0)
    parser.add_argument("--cclblock-aux-objective", type=str, default="none", choices=["none", "boundary", "entropy"])
    parser.add_argument("--cclblock-aux-lambda", type=float, default=0.1)
    parser.add_argument("--cclblock-boundary-token-id", type=int, default=198)
    # Phase 9
    parser.add_argument("--use-ral", type=int, default=0, choices=[0, 1])
    parser.add_argument("--ral-rank", type=int, default=32)
    parser.add_argument("--cclblock-film-gate", type=int, default=0, choices=[0, 1])
    parser.add_argument("--cclblock-attn-shadow-dim", type=int, default=0)
    parser.add_argument("--cclblock-dynamic-ratio", type=float, default=0.25)
    parser.add_argument("--cclblock-gate-rank", type=int, default=8)
    parser.add_argument("--cclblock-num-regimes", type=int, default=8)
    parser.add_argument("--cclblock-regime-temperature", type=float, default=1.0)
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
    # Phase 12: FSI/AESP/CKR
    parser.add_argument("--cclblock-fsi-rotations", type=int, default=8)
    parser.add_argument("--cclblock-fsi-selector-dim", type=int, default=64)
    parser.add_argument("--cclblock-aesp-strata", type=int, default=4)
    parser.add_argument("--cclblock-aesp-delta-rank", type=int, default=4)
    parser.add_argument("--cclblock-ckr-branches", type=int, default=4)
    parser.add_argument("--cclblock-ckr-kernel-size", type=int, default=64)
    # Phase 13: CKR enhancements
    parser.add_argument("--cclblock-ckr-pos-channels", type=int, default=1)
    parser.add_argument("--cclblock-ckr-dual-optim", type=int, default=0, choices=[0, 1])
    parser.add_argument("--cclblock-ckr-content-bias", type=float, default=0.0)
    # Phase 14: GIAD/PSG/SplitStream
    parser.add_argument("--cclblock-giad-rank", type=int, default=32)
    parser.add_argument("--cclblock-psg-kernel-size", type=int, default=64)
    parser.add_argument("--cclblock-ss-dynamic-ratio", type=float, default=0.25)
    parser.add_argument("--cclblock-ss-branches", type=int, default=2)
    parser.add_argument("--cclblock-ss-kernel-size", type=int, default=64)
    # Phase 15: LoKR
    parser.add_argument("--cclblock-lokr-branches", type=int, default=8)
    parser.add_argument("--cclblock-lokr-rank", type=int, default=16)
    # Phase 16: CKR-Anneal / COM
    parser.add_argument("--cclblock-ckr-temp-start", type=float, default=2.0)
    parser.add_argument("--cclblock-ckr-temp-end", type=float, default=0.3)
    parser.add_argument("--cclblock-com-kernel-size", type=int, default=32)
    # Phase 17
    parser.add_argument("--cclblock-ckr-ortho-init", type=int, default=0, choices=[0, 1])
    parser.add_argument("--cclblock-ckr-branch-dropout", type=float, default=0.0)
    parser.add_argument("--cclblock-ckr-diversity-lambda", type=float, default=0.0)
    parser.add_argument("--cclblock-pgr-kernel-size", type=int, default=64)
    parser.add_argument("--cclblock-cil-kernel-size", type=int, default=64)
    parser.add_argument("--cclblock-prb-kernel-size", type=int, default=64)
    parser.add_argument("--modulation-diagnostics", type=int, default=0, choices=[0, 1])
    # Phase 18
    parser.add_argument("--p18-layer-drop", type=float, default=0.0)
    parser.add_argument("--p18-dynamic-activation", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p18-mixture-norm", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p18-aux-sim-lambda", type=float, default=0.0)
    parser.add_argument("--p18-gradient-penalty", type=float, default=0.0)
    parser.add_argument("--p18-per-channel-scale", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()
    
    run_training_sweep(args)
