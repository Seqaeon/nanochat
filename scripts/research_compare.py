import argparse
import sys
import os
import json
import glob
from pathlib import Path
import subprocess

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def estimate_tokens_from_base(depth: int, target_ratio: float = 10.5, tokenizer_dir: str = None) -> int:
    # Replica of base_train's logic
    from nanochat.gpt import GPT, GPTConfig
    
    vocab_size = 32768 # fallback
    try:
        from nanochat.tokenizer import get_tokenizer
        tokenizer = get_tokenizer(tokenizer_dir=tokenizer_dir)
        vocab_size = tokenizer.get_vocab_size()
    except Exception:
        pass
        
    aspect_ratio = 4
    head_dim = 16
    
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    
    config = GPTConfig(
        sequence_len=64, # fixed max_seq_len for sweep
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
    )
    with torch.device("meta"):
        model = GPT(config)
    
    params_counts = model.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    
    return int(scaling_params * target_ratio)

def check_and_prepare_env(args):
    # 1. Resolve data_dir and tokenizer_dir
    from nanochat.common import get_base_dir
    from nanochat.dataset import resolve_data_dir, list_parquet_files
    
    data_dir = args.data_dir if args.data_dir else resolve_data_dir()
    tokenizer_dir = args.tokenizer_dir if args.tokenizer_dir else os.path.join(get_base_dir(), "tokenizer")
    
    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    # 2. Check for data shards
    shards = list_parquet_files(data_dir=data_dir)
    if not shards:
        print(f"No data shards found in {data_dir}. Downloading...")
        # If max_shards is set, use it. Otherwise download at least 2 (1 train, 1 val).
        num_files = args.max_shards if (args.max_shards and args.max_shards > 0) else 2
        cmd = [sys.executable, "-m", "nanochat.dataset", "-n", str(num_files), "--data-dir", data_dir]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    
    # 3. Check for tokenizer
    tokenizer_pkl = os.path.join(tokenizer_dir, "tokenizer.pkl")
    if not os.path.exists(tokenizer_pkl):
        print(f"Tokenizer not found at {tokenizer_pkl}. Training...")
        # Train on a small subset for speed.
        cmd = [sys.executable, "-m", "scripts.tok_train", "--max-chars", "10000000", "--data-dir", data_dir, "--tokenizer-dir", tokenizer_dir]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

def run_training_sweep(args):
    # Ensure environment is ready
    check_and_prepare_env(args)
    
    depth = args.depth
    run_dir = args.run_dir
    run_dir_path = Path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)
    
    target_tokens = estimate_tokens_from_base(depth, tokenizer_dir=args.tokenizer_dir)
    print("=" * 64)
    print(f"Starting Sweep for Depth {depth}")
    print(f"Calculated Target Tokens: {target_tokens:,}")
    print("=" * 64)
    
    aspect_ratio = 4
    head_dim = 16
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    target_dim = model_dim // 8
    
    # Common kwargs for all models
    common_args = [
        "--depth", str(depth),
        "--aspect-ratio", "4",
        "--head-dim", "16",
        "--max-seq-len", "64",
        "--device-batch-size", "16",
        "--total-batch-size", "524288", # standard for reference
        "--target-tokens", str(target_tokens),
        "--eval-every", "-1",        # We only evaluate at end for speed
        "--core-metric-every", "-1",
        "--sample-every", "-1",
        "--compile",
    ]
    if getattr(args, "fp8", False):
        common_args.append("--fp8")
    if getattr(args, "tokenizer_dir", None):
        common_args.extend(["--tokenizer-dir", args.tokenizer_dir])
    if getattr(args, "data_dir", None):
        common_args.extend(["--data-dir", args.data_dir])
    if getattr(args, "max_shards", -1) != -1:
        common_args.extend(["--max-shards", str(args.max_shards)])
    
    models = {
        "base": [],
        "moe_no_perm": [
            "--use-moe",
            "--num-experts", "8",
            "--router-dim", "64",
            "--target-dim", str(target_dim),
            "--embedding-lr", "0.05",
        ],
        "moe_perm": [
            "--use-moe",
            "--use-perm",
            "--num-experts", "8",
            "--router-dim", "64",
            "--target-dim", str(target_dim),
            "--selection-mode", "soft",
            "--embedding-lr", "0.05",
        ],
        "remixed_linear": [
            "--use-remixed-linear",
            "--target-dim", str(target_dim),
            "--router-dim", "64",
            "--context-dim", "32",
            "--linear-basis-size", "16",
            "--embedding-lr", "0.05",
        ]
    }
    
    results = {}
    
    for model_name, extra_args in models.items():
        print(f"\n--- Training {model_name} ---")
        
        ckpt_dir = run_dir_path / f"ckpt_{model_name}"
        
        args = common_args + extra_args + [
            "--checkpoints-dir", str(ckpt_dir),
            "--model-tag", "model"
        ]
        
        
        # Need to preserve environment variables, especially LD_LIBRARY_PATH for cusparseLt
        env = os.environ.copy()
        
        cmd = [sys.executable, "-m", "scripts.base_train"] + args
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
                ckpts = glob.glob(str(model_ckpt_dir / "checkpoint_*.pt"))
                if ckpts:
                    ckpts.sort()
                    last_ckpt = ckpts[-1]
                    try:
                        checkpoint = torch.load(last_ckpt, map_location="cpu", weights_only=False)
                        if "metadata" in checkpoint:
                            val_bpb = float(checkpoint["metadata"].get("val_bpb", float("inf")))
                            results[model_name] = {"val_bpb": val_bpb, "checkpoint": last_ckpt}
                            print(f"Final Validation BPB for {model_name}: {val_bpb:.4f}")
                        else:
                            print(f"No metadata found in {last_ckpt}")
                    except Exception as e:
                        print(f"Failed to load checkpoint {last_ckpt}: {e}")
                else:
                    print(f"No checkpoints found in {model_ckpt_dir}")
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
    args = parser.parse_args()
    
    run_training_sweep(args)
