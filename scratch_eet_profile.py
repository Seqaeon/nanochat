"""
Profile EET compute-skip vs dense path to find the 5X slowdown.
Run: PYTHONPATH=. .venv/bin/python scratch_eet_profile.py
"""
import torch
import time
from nanochat.gpt import GPTConfig
from nanochat.eet import EarlyExitGPT

device = "cuda" if torch.cuda.is_available() else "cpu"
B, T = 4, 1024
n_layer = 8

def make_config(**overrides):
    defaults = dict(
        n_head=6, n_kv_head=6, n_embd=384, vocab_size=32768,
        sequence_len=T, use_eet=True, eet_min_exit_layer=1,
        eet_frozen_kv=False, eet_reenter_final=True,
        eet_loss_variant='ce_guided', eet_ce_guided_lambda=1.0,
        eet_gumbel_temp_start=1.0, eet_gumbel_temp_end=0.1,
        eet_gumbel_hard=1, eet_global_router=True,
        eet_warmup_frac=0.0, eet_explore_frac=0.0,
        eet_target_active_frac=0.10,
        n_layer=n_layer,
    )
    defaults.update(overrides)
    return GPTConfig(**defaults)

# === Test 1: Verify hard exit is working ===
print("=" * 70)
print("TEST 1: Verify hard exit physically drops tokens")
print("=" * 70)

config_skip = make_config(eet_compute_skip=True)
model_skip = EarlyExitGPT(config_skip).to(device)
model_skip.train()

x = torch.randint(0, config_skip.vocab_size, (B, T), device=device)
y = torch.randint(0, config_skip.vocab_size, (B, T), device=device)

# Monkey-patch blocks to print their input shapes
original_forwards = {}
for i, block in enumerate(model_skip.transformer.h):
    original_forwards[i] = block.forward
    def make_hook(idx, orig_fwd):
        def hooked_forward(x_input, *args, **kwargs):
            print(f"  Block {idx}: input shape = {x_input.shape} ({x_input.shape[1]} tokens)")
            return orig_fwd(x_input, *args, **kwargs)
        return hooked_forward
    block.forward = make_hook(i, block.forward)

loss = model_skip(x, y, eet_do_route=True, eet_phase=3)
print(f"\nLoss: {loss.item():.4f}")

diag = model_skip._eet_diagnostics
print(f"Diagnostics: phase={diag['phase']}, active={diag['active_frac']:.3f}, "
      f"exit_frac={diag['total_exit_frac']:.3f}")

# Restore original forwards
for i, block in enumerate(model_skip.transformer.h):
    block.forward = original_forwards[i]

loss.backward()
print("\nBackward pass OK")

# Check router gradients
router_grads = {}
for name, p in model_skip.named_parameters():
    if 'eet_router' in name and p.grad is not None:
        router_grads[name] = p.grad.abs().mean().item()
print(f"Router parameters with gradients: {len(router_grads)}")
for name, g in list(router_grads.items())[:3]:
    print(f"  {name}: grad_norm = {g:.6e}")

# === Test 2: Profile with and without candidate_states ===
print("\n" + "=" * 70)
print("TEST 2: Profile compute_skip forward pass (compiled)")
print("=" * 70)

# Fresh model for profiling (no hooks)
model_prof = EarlyExitGPT(make_config(eet_compute_skip=True)).to(device)
model_prof.train()
model_compiled = torch.compile(model_prof)

x = torch.randint(0, config_skip.vocab_size, (B, T), device=device)
y = torch.randint(0, config_skip.vocab_size, (B, T), device=device)

# Warmup (trigger compilation)
print("Warming up compiled model (compute_skip=True)...")
for _ in range(3):
    loss = model_compiled(x, y, eet_do_route=True, eet_phase=3)
    loss.backward()
    model_prof.zero_grad()

torch.cuda.synchronize() if device == "cuda" else None
t0 = time.perf_counter()
N = 20
for _ in range(N):
    loss = model_compiled(x, y, eet_do_route=True, eet_phase=3)
    loss.backward()
    model_prof.zero_grad()
torch.cuda.synchronize() if device == "cuda" else None
dt_skip = (time.perf_counter() - t0) / N * 1000
print(f"  compute_skip=True: {dt_skip:.1f} ms/step")

# === Test 3: Profile without compute_skip (dense masking path) ===
# The dense path uses eet_compute_skip=False, but with global_router=True
# the code auto-enables compute_skip. So test with global_router=False
# to get the old dense masking path.
print("\nProfiling dense masking path (no compute_skip, per-layer routers)...")
model_dense = EarlyExitGPT(make_config(
    eet_compute_skip=False, eet_global_router=False,
)).to(device)
model_dense.train()
model_dense_compiled = torch.compile(model_dense)

for _ in range(3):
    loss = model_dense_compiled(x, y, eet_do_route=True, eet_phase=3)
    loss.backward()
    model_dense.zero_grad()

torch.cuda.synchronize() if device == "cuda" else None
t0 = time.perf_counter()
for _ in range(N):
    loss = model_dense_compiled(x, y, eet_do_route=True, eet_phase=3)
    loss.backward()
    model_dense.zero_grad()
torch.cuda.synchronize() if device == "cuda" else None
dt_dense = (time.perf_counter() - t0) / N * 1000
print(f"  dense masking path: {dt_dense:.1f} ms/step")

print(f"\n  Ratio: compute_skip is {dt_skip/dt_dense:.2f}x vs dense")

print("\nDone!")
