import argparse
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanochat.gpt import GPT, GPTConfig
from scripts._sweep_utils import model_dims

def get_model_params(model_type, depth, gate_mode='mlp'):
    n_layer, n_head, n_embd, max_seq_len = model_dims(depth)
    
    config = GPTConfig(
        sequence_len=max_seq_len,
        vocab_size=100277,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )
    
    if model_type == 'dense':
        config.use_remixed_linear = False
    elif model_type == 'remixed':
        config.use_remixed_linear = True
        config.remix_basis_size = 256
        config.remix_context_dim = 256
        config.remixed_linear_kwargs = {
            'use_basis_gate': True,
            'use_output_gate': True,
            'use_context': True,
            'basis_gate_mode': gate_mode,
            'output_gate_rank': 8,
        }
        
    model = GPT(config)
    # also print parameter table to find the bloat
    if depth == 12:
        from scripts.base_train import print_params_table
        print(f"\n--- {model_type} ({gate_mode}) ---")
        print_params_table(model)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

for d in [4, 12]:
    print(f"Depth {d} (C={model_dims(d)[2]}):")
    print(f"  Dense:         {get_model_params('dense', d):,}")
    print(f"  Remixed (MLP): {get_model_params('remixed', d, 'mlp'):,}")
    print(f"  Remixed (Lin): {get_model_params('remixed', d, 'linear'):,}")
