import torch
from nanochat.flash_attention import flash_attn_with_kvcache, flash_attn_func
q = torch.randn(1, 1, 2, 64, dtype=torch.bfloat16, device='cuda')
k = torch.randn(1, 1, 2, 64, dtype=torch.bfloat16, device='cuda')
v = torch.randn(1, 1, 2, 72, dtype=torch.bfloat16, device='cuda')
k_cache = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16, device='cuda')
v_cache = torch.randn(1, 128, 2, 72, dtype=torch.bfloat16, device='cuda')
seqlens = torch.tensor([0], dtype=torch.int32, device='cuda')

print("Testing FA3 training...")
try:
    flash_attn_func(q, k, v, causal=True)
    print("Success training")
except Exception as e:
    print("Error training:", e)

print("Testing FA3 kvcache...")
try:
    flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, cache_seqlens=seqlens, causal=True)
    print("Success kvcache")
except Exception as e:
    print("Error kvcache:", e)
