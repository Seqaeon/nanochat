import sys
import os
import torch
import shlex

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanochat.gpt import GPT, GPTConfig
from scripts._sweep_utils import model_dims
from scripts.base_train import get_args

def parse_args_string(cmd):
    parser = get_args()
    return parser.parse_args(shlex.split(cmd))

def build_model(args):
    _, head_dim, model_dim, _ = model_dims(args.depth)
    num_heads = model_dim // head_dim
    vocab_size = 100288 # dummy for test
    config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=vocab_size,
        n_layer=args.depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        use_remixed_linear=args.use_remix_linear,
        remixed_linear_kwargs={
            'use_basis_gate': args.remix_use_basis_gate == 1,
            'use_output_gate': args.remix_use_output_gate == 1,
            'use_context': args.remix_use_context == 1,
            'basis_gate_mode': args.remix_basis_gate_mode,
            'output_gate_rank': 8,
            'film_gate': False,
        },
        remix_basis_size=args.remix_basis_size,
        remix_context_dim=args.remix_context_dim,
    )
    
    # add all the cclblock args
    for k, v in vars(args).items():
        if k.startswith('cclblock_'):
            setattr(config, k, v)
            
    model = GPT(config)
    return model

# Dense cmd
cmd_dense = "--depth 12 --aspect-ratio 64 --head-dim 128 --model-dim 768 --max-seq-len 2048"
args_dense = parse_args_string(cmd_dense)
model_dense = build_model(args_dense)
print(f"Dense: {sum(p.numel() for p in model_dense.parameters()):,}")

# Remix cmd (from the log)
cmd_remix = "--depth 12 --aspect-ratio 64 --head-dim 128 --model-dim 768 --max-seq-len 2048 --remix-use-basis-gate 1 --remix-use-output-gate 1 --remix-use-context 1 --remix-basis-gate-mode mlp --use-remix-linear --remix-basis-size 256 --remix-context-dim 256"
args_remix = parse_args_string(cmd_remix)
model_remix = build_model(args_remix)
print(f"Remixed: {sum(p.numel() for p in model_remix.parameters()):,}")

# Let's count where the params are
def print_param_summary(model):
    from collections import defaultdict
    counts = defaultdict(int)
    for name, p in model.named_parameters():
        if 'attn' in name: counts['attn'] += p.numel()
        elif 'mlp' in name: counts['mlp'] += p.numel()
        elif 'cclblock' in name: counts['cclblock'] += p.numel()
        elif 'wte' in name or 'lm_head' in name: counts['embed'] += p.numel()
        else: counts['other'] += p.numel()
    for k, v in counts.items():
        print(f"  {k}: {v:,}")

print("\nDense breakdown:")
print_param_summary(model_dense)

print("\nRemixed breakdown:")
print_param_summary(model_remix)
