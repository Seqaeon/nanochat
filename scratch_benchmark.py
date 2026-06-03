import torch
import time

def benchmark():
    B = 4
    T = 1024
    C = 768
    K_cur = 512
    rope_dim = 64
    ve_dim = 768
    device = "cuda"
    dtype = torch.bfloat16

    x = torch.randn(B, T, C, device=device, dtype=dtype)
    x0 = torch.randn(B, T, C, device=device, dtype=dtype)
    x_out = torch.randn(B, K_cur, C, device=device, dtype=dtype)
    active_idx = torch.randint(0, T, (B, K_cur), device=device)
    cos_full = torch.randn(1, T, 1, rope_dim, device=device, dtype=dtype)
    sin_full = torch.randn(1, T, 1, rope_dim, device=device, dtype=dtype)
    ve_full = torch.randn(B, T, ve_dim, device=device, dtype=dtype)

    # Method 1: Old way (always gather/scatter)
    def method_old(x, x0, active_idx, cos_full, sin_full, ve_full, x_out):
        idx3 = active_idx.unsqueeze(-1)
        # Scatter block output
        x_new = x.scatter(1, idx3.expand(-1, -1, C), x_out)
        # Gather for next block
        x_active = torch.gather(x_new, 1, idx3.expand(-1, -1, C))
        # Gather RoPE
        idx4 = idx3.unsqueeze(-1)
        cos_act = torch.gather(cos_full.expand(B, -1, -1, -1), 1, idx4.expand(-1, -1, 1, rope_dim))
        sin_act = torch.gather(sin_full.expand(B, -1, -1, -1), 1, idx4.expand(-1, -1, 1, rope_dim))
        # Gather VE
        ve = torch.gather(ve_full, 1, idx3.expand(-1, -1, ve_dim))
        return x_active, cos_act, sin_act, ve, x_new

    # Method 2: New way (non-routing block: zero gather/scatter of hidden states)
    # Since active_idx doesn't change, we just reuse the gathered tensors or gather VE/RoPE
    # if necessary, but we do ZERO gather/scatter on x!
    def method_new(x_active, cos_act, sin_act, ve_full, active_idx):
        # We don't gather x_active, we don't scatter it either!
        # We only need to gather VE if it changes per-block
        idx3 = active_idx.unsqueeze(-1)
        ve = torch.gather(ve_full, 1, idx3.expand(-1, -1, ve_dim))
        return x_active, cos_act, sin_act, ve

    compiled_old = torch.compile(method_old, fullgraph=True)
    compiled_new = torch.compile(method_new, fullgraph=True)

    # Warmup
    for _ in range(30):
        _ = compiled_old(x, x0, active_idx, cos_full, sin_full, ve_full, x_out)
        _ = compiled_new(x_out, cos_full[:, :K_cur], sin_full[:, :K_cur], ve_full, active_idx)

    torch.cuda.synchronize()

    # Time Old
    t0 = time.time()
    for _ in range(1000):
        _ = compiled_old(x, x0, active_idx, cos_full, sin_full, ve_full, x_out)
    torch.cuda.synchronize()
    t_old = time.time() - t0

    # Time New
    t0 = time.time()
    for _ in range(1000):
        _ = compiled_new(x_out, cos_full[:, :K_cur], sin_full[:, :K_cur], ve_full, active_idx)
    torch.cuda.synchronize()
    t_new = time.time() - t0

    print(f"Non-routing block Compiled Old (full scatter/gather): {t_old * 1000:.2f} ms")
    print(f"Non-routing block Compiled New (zero scatter/gather): {t_new * 1000:.2f} ms")

if __name__ == "__main__":
    benchmark()
