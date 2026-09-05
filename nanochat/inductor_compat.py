"""Keep Inductor from crashing on a Triton block size it chose itself.

`triton_config` builds autotune candidates and then asserts each block size is
within `TRITON_MAX_BLOCK`. It can violate its own bound: the `min_elem_per_thread`
expansion multiplies XBLOCK *after* the grid clamp, so a large fused pointwise
kernel can be handed XBLOCK=8192 against a documented maximum of 4096 and die with

    AssertionError: 'XBLOCK' too large. Maximum: 4096. Actual: 8192.

The Monarch head reaches this through `y.reshape(-1, V)`, the unavoidable clone
from `(m2, N, block_out)` to `(N, m2, block_out)` with the bias add fused in. It
is shape dependent: at V=32768 the failing arms are the ones whose `block_out`
exceeds 4096, meaning m2 = V/block_out of 4 or less.

Clamping is safe where raising the bound is not. The kernel source was generated
while `max_block_size()` reported 4096, so 4096 is a block size that source is
already written to handle; the value only picks how the x range is tiled, and the
generated code masks with `xindex < xnumel` for any of them. Raising the bound
instead would make the selected block disagree with the assumptions codegen made.

The cost is one autotune candidate tiled at 4096 rather than 8192 on affected
kernels. That is a tiling choice among several the autotuner already races, not a
loss of compilation, which is what `--no-compile` would cost instead.

Compilation happens in worker processes (`worker_start_method = "subprocess"`),
which import torch fresh and so do not inherit this patch. Reach them with either

    TORCHINDUCTOR_COMPILE_THREADS=1     compile in process, slower compile, once
    PYTHONPATH=<dir with sitecustomize.py that calls this>   keeps parallel compile
"""
_PATCHED = False
_REPORTED = set()


def guard_triton_block_size(verbose: bool = True) -> bool:
    """Replace Inductor's block-size assertion with a clamp. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return True
    try:
        from torch._inductor.runtime import triton_heuristics as th
        from torch._inductor.runtime.hints import TRITON_MAX_BLOCK
    except Exception:
        return False

    def check_max_block(cfg):
        for var, val in cfg.items():
            if "BLOCK" not in var:
                continue
            cap = TRITON_MAX_BLOCK.get(var.removesuffix("BLOCK"))
            if cap is not None and val > cap:
                cfg[var] = cap
                key = (var, val, cap)
                if verbose and key not in _REPORTED:
                    _REPORTED.add(key)
                    print(f"[inductor] clamped {var} {val} -> {cap} "
                          f"(Inductor exceeded its own TRITON_MAX_BLOCK; "
                          f"see nanochat/inductor_compat.py)", flush=True)

    th.check_max_block = check_max_block
    _PATCHED = True
    return True
