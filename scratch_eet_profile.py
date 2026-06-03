"""
Profile EET compute-skip vs dense path to measure the fix.
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

x = torch.randint(0, 32768, (B, T), device=device)
y = torch.randint(0, 32768, (B, T), device=device)

# === Test 1: Verify hard exit shapes ===
print("=" * 70)
print("TEST 1: Verify hard exit physically drops tokens")
print("=" * 70)

config_skip = make_config(eet_compute_skip=True)
model_skip = EarlyExitGPT(config_skip).to(device)
model_skip.train()

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
loss.backward()
print("Backward OK")

# Restore
for i, block in enumerate(model_skip.transformer.h):
    block.forward = original_forwards[i]
del model_skip
torch.cuda.empty_cache() if device == "cuda" else None

# === Test 2: Profile compute_skip WITH task grad (eet_router_task_grad=True) ===
print("\n" + "=" * 70)
print("TEST 2: Compiled compute_skip WITH task grad (eet_router_task_grad=True)")
print("=" * 70)

model_a = EarlyExitGPT(make_config(eet_compute_skip=True, eet_router_task_grad=True)).to(device)
model_a.train()
model_a_c = torch.compile(model_a)

print("Warming up...")
for _ in range(3):
    loss = model_a_c(x, y, eet_do_route=True, eet_phase=3)
    loss.backward()
    model_a.zero_grad()

if device == "cuda": torch.cuda.synchronize()
N = 20
t0 = time.perf_counter()
for _ in range(N):
    loss = model_a_c(x, y, eet_do_route=True, eet_phase=3)
    loss.backward()
    model_a.zero_grad()
if device == "cuda": torch.cuda.synchronize()
dt_skip_grad = (time.perf_counter() - t0) / N * 1000
print(f"  compute_skip + task_grad=True: {dt_skip_grad:.1f} ms/step")

del model_a, model_a_c
torch.cuda.empty_cache() if device == "cuda" else None

# === Test 3: Profile compute_skip WITHOUT task grad (eet_router_task_grad=False) ===
print("\n" + "=" * 70)
print("TEST 3: Compiled compute_skip WITHOUT task grad (eet_router_task_grad=False)")
print("=" * 70)

model_b = EarlyExitGPT(make_config(eet_compute_skip=True, eet_router_task_grad=False)).to(device)
model_b.train()
model_b_c = torch.compile(model_b)

print("Warming up...")
for _ in range(3):
    loss = model_b_c(x, y, eet_do_route=True, eet_phase=3)
    loss.backward()
    model_b.zero_grad()

if device == "cuda": torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    loss = model_b_c(x, y, eet_do_route=True, eet_phase=3)
    loss.backward()
    model_b.zero_grad()
if device == "cuda": torch.cuda.synchronize()
dt_skip_nograd = (time.perf_counter() - t0) / N * 1000
print(f"  compute_skip + task_grad=False (optimized): {dt_skip_nograd:.1f} ms/step")

del model_b, model_b_c
torch.cuda.empty_cache() if device == "cuda" else None

# === Test 4: Profile TRUE Dense Baseline (use_eet=False) ===
print("\n" + "=" * 70)
print("TEST 4: Compiled TRUE Dense Baseline (use_eet=False)")
print("=" * 70)

model_dense = EarlyExitGPT(make_config(use_eet=False)).to(device)
model_dense.train()
model_dense_c = torch.compile(model_dense)

print("Warming up...")
for _ in range(3):
    loss = model_dense_c(x, y)
    loss.backward()
    model_dense.zero_grad()

if device == "cuda": torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    loss = model_dense_c(x, y)
    loss.backward()
    model_dense.zero_grad()
if device == "cuda": torch.cuda.synchronize()
dt_dense = (time.perf_counter() - t0) / N * 1000
print(f"  true dense baseline: {dt_dense:.1f} ms/step")

print(f"\nSummary of Step Times (B={B}, T={T}, C={384}, layers={n_layer}):")
print(f"  True Dense Baseline              : {dt_dense:.1f} ms")
print(f"  Compute Skip (task_grad=True)    : {dt_skip_grad:.1f} ms (ratio vs dense: {dt_skip_grad/dt_dense:.2f}x)")
print(f"  Compute Skip (task_grad=False)   : {dt_skip_nograd:.1f} ms (ratio vs dense: {dt_skip_nograd/dt_dense:.2f}x)")

print("\nDone!")
