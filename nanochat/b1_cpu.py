"""Python binding for the CPU bit kernel, and the equivalence check that matters.

nanochat/binary.py simulates binary arithmetic in bf16. This runs the same function
with real bit operations, and `check_equivalence` asserts they agree EXACTLY. Until
that assertion passes, "a fully binary transformer" describes a simulation.
"""
import ctypes
import os
import subprocess

import numpy as np
import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_HERE, "kernels", "b1_cpu.c")
_LIB = os.path.join(_HERE, "kernels", "libb1cpu.so")
_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIB) or os.path.getmtime(_SRC) > os.path.getmtime(_LIB):
        subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC", _SRC,
                        "-o", _LIB], check=True)
    lib = ctypes.CDLL(_LIB)
    u64p, f32p, i32p = (ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_int32))
    lib.b1_pack.argtypes = [f32p, u64p, ctypes.c_int, ctypes.c_int]
    lib.b1_linear.argtypes = [u64p, u64p, i32p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.b1_bundle.argtypes = [u64p, u64p, i32p, ctypes.c_int, ctypes.c_int]
    _lib = lib
    return lib


def pack(x: torch.Tensor) -> np.ndarray:
    """(rows, cols) of +-1 -> packed uint64 bits, 1 = positive."""
    lib = _load()
    a = np.ascontiguousarray(x.detach().float().cpu().numpy())
    rows, cols = a.shape
    out = np.zeros((rows, (cols + 63) // 64), dtype=np.uint64)
    lib.b1_pack(a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), rows, cols)
    return out


def linear(a_packed: np.ndarray, b_packed: np.ndarray, K: int) -> np.ndarray:
    """Integer popcount form: out[m][n] = K - 2*hamming = <sign(a), sign(b)>."""
    lib = _load()
    M, N = a_packed.shape[0], b_packed.shape[0]
    out = np.zeros((M, N), dtype=np.int32)
    lib.b1_linear(a_packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                  b_packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                  out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), M, N, K)
    return out


def bundle(packed: np.ndarray, bits: int, threshold=None) -> np.ndarray:
    lib = _load()
    n = packed.shape[0]
    out = np.zeros(((bits + 63) // 64,), dtype=np.uint64)
    th = (np.zeros(bits, dtype=np.int32) if threshold is None
          else np.ascontiguousarray(threshold, dtype=np.int32))
    lib.b1_bundle(packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                  out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                  th.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), n, bits)
    return out


def check_equivalence(M=64, N=128, K=512, seed=0):
    """The kernel and the fp simulation must compute the SAME function.

    Returns (ok, max_abs_diff). Exactness, not tolerance: both compute an integer
    count, so any disagreement is a bug rather than rounding.
    """
    from nanochat.binary import BinaryLinear, sign_ste
    torch.manual_seed(seed)
    lin = BinaryLinear(K, N, weight_scale="none", act_scale="none")
    lin.reset_parameters()
    x = torch.randn(M, K)
    with torch.no_grad():
        sim = lin(x)                                    # the bf16/fp simulation
        xb = sign_ste(x, lin.clip)
        wb = sign_ste(lin.weight, lin.clip)
    got = linear(pack(xb), pack(wb), K).astype(np.float32)
    diff = float(np.abs(got - sim.numpy()).max())
    return diff == 0.0, diff
