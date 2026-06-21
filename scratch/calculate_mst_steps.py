import os
import sys
import math
import torch

# Add workspace directory to path
sys.path.append(os.path.abspath('/home/seqaeon/Downloads/nanochat'))

from nanochat.gpt import GPTConfig
from nanochat.mst import MST
from scripts.base_train import get_tokenizer

# Initialize tokenizer to get actual vocab_size
tokenizer = get_tokenizer(tokenizer_dir='/home/seqaeon/Downloads/nanochat/tokenizer')
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size}")

def get_scaling_params(m):
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params

# Setup config template
def get_mst_config(depth):
    # Setup dims following the sweep script logic
    aspect_ratio = 64
    n_subs = 4
    model_dim = depth * aspect_ratio
    # Round up to nearest 128 (head_dim alignment)
    model_dim = ((model_dim + 127) // 128) * 128
    sub_dim = model_dim // n_subs
    
    # We choose head_dim = 64 (standard for these depths in nanochat)
    num_heads = model_dim // 64
    
    config = GPTConfig(
        sequence_len=2048,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern="SSSSL",
        use_moe=False,
        use_perm=False,
        use_mst=True,
        mst_n_subs=n_subs,
        mst_sub_dim=sub_dim,
        mst_head_dim=0,
        mst_input_mode="learned_proj",
        mst_routing_mode="soft_weighted",
        mst_routing_topk=0,
        mst_routing_aux_weight=0.01,
        mst_diversity_weight=0.0,
        mst_ffn_mode="standard",
        mst_transition_mode="aggregate_distribute",
        mst_final_mode="concat_proj",
        mst_final_topk=0,
        mst_multi_scale_windows=1
    )
    return config

# Get reference model d12_ref
d12_config = get_mst_config(12)
with torch.device("meta"):
    d12_ref = MST(d12_config)
    
target_param_data_ratio = 10.5
D_REF = target_param_data_ratio * get_scaling_params(d12_ref)
B_REF = 2**19

print(f"D_REF (d12 tokens): {D_REF:,}")
print(f"B_REF (d12 batch size): {B_REF:,}")

for depth in [8, 12, 16, 24]:
    config = get_mst_config(depth)
    with torch.device("meta"):
        model = MST(config)
    
    num_scaling_params = get_scaling_params(model)
    target_tokens = int(target_param_data_ratio * num_scaling_params)
    
    # Auto-compute total_batch_size
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * (batch_size_ratio ** 0.383)
    total_batch_size = 2 ** round(math.log2(predicted_batch_size))
    
    num_iterations = target_tokens // total_batch_size
    
    print(f"\nDepth {depth}:")
    print(f"  Model Dim: {config.n_embd}")
    print(f"  Sub Dim: {config.mst_sub_dim}")
    print(f"  Num Scaling Params: {num_scaling_params:,}")
    print(f"  Target Tokens: {target_tokens:,}")
    print(f"  Total Batch Size: {total_batch_size:,}")
    print(f"  Num Iterations (Steps): {num_iterations:,}")
