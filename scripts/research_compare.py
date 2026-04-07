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
    max_seq_len = 2048
    device_batch_size = {4: 8, 8: 32, 16: 16, 24: 8}.get(depth, 16)
    total_batch_size = 524288
    eval_every = 1000
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
        "--eval-every", "-1",        # We only evaluate at end for speed
        "--core-metric-every", "-1",
#        "--sample-every", "-1",
        "--warmup-ratio", str(warm_up_ratio),    # Safer for research models
        "--adam-beta2", str(adam_beta2),     # Matches notebook
        "--research-warmup-ratio", str(args.research_warmup_ratio),
        "--use-onecycle", str(args.use_onecycle),
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
            "embedding_lr":   0.1,
            "unembedding_lr": 0.3,
            "matrix_lr":      1.0,
            "scalar_lr":      0.1,
        },
        "moe_perm": {
            "embedding_lr":   0.5,
            "unembedding_lr": 0.5,
            "matrix_lr":      0.5,
            "scalar_lr":      2.5,
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
                    print(line, end="")
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
    parser.add_argument("--warmup-ratio", type=float, default=0.0, help="base warmup ratio passed to all runs")
    parser.add_argument("--models", type=str, default="all", help="Comma-separated list of models to run (e.g. 'base,remixed-linear'), or 'all'")
    parser.add_argument("--research-warmup-ratio", type=float, default=0.05, help="research-branch warmup ratio for OneCycle")
    parser.add_argument("--use-onecycle", type=int, default=1, choices=[0, 1], help="research branches: 1=OneCycle, 0=use base schedule")
    args = parser.parse_args()
    
    run_training_sweep(args)
