import torch
from nanochat.gpt import QuantileCrossAttentionRouter

router = QuantileCrossAttentionRouter(in_features=256, n_experts=8, topk=2)
x = torch.randn(2, 64, 256)
w = router(x)
print(f"Train output shape: {w.shape}")

router.eval()
w_eval = router(x)
print(f"Eval output shape: {w_eval.shape}")

kv_state = {}
w_kv = router(x[:, :1, :], kv_state)
print(f"Eval KV step 1 shape: {w_kv.shape}")
print(f"KV keys: {kv_state.keys()}")

# Test compile
compiled_router = torch.compile(router)
w_comp = compiled_router(x)
print(f"Compiled output shape: {w_comp.shape}")
