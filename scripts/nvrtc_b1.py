"""Runtime-compiled b1 GEMM: the no-nvcc path for O3.

`pip install nvidia-cuda-nvcc-cu12` DOES NOT give you nvcc. Every published version
of that wheel, 12.0 through 12.9, ships `ptxas` and nothing else. A CUDA *runtime*
container therefore cannot obtain the nvcc frontend from pip at all, which is the
whole reason this module exists.

What IS available, in the wheels torch already depends on:
  nvidia/cuda_nvrtc/lib/libnvrtc.so.12      the runtime compiler
  nvidia/cuda_runtime/include/mma.h         the WMMA header
plus libcuda.so.1 from the driver, which is present wherever a GPU is.

So: compile kernels/b1_gemm.cuh with NVRTC to PTX, load it with the CUDA driver API,
and launch it on torch's own stream against torch's own tensors. No toolkit, no
build step, no new dependency.
"""
import ctypes
import glob
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL = os.path.join(REPO, "kernels", "b1_gemm.cuh")


def _roots():
    import site
    rs = list(site.getsitepackages())
    try:
        rs.append(site.getusersitepackages())
    except Exception:
        pass
    rs.append(os.path.join(sys.prefix, "lib",
                           f"python{sys.version_info.major}.{sys.version_info.minor}",
                           "site-packages"))
    return [r for r in dict.fromkeys(rs) if os.path.isdir(r)]


def find_nvrtc():
    for r in _roots():
        hits = sorted(glob.glob(os.path.join(r, "nvidia", "*", "lib", "libnvrtc.so*")))
        hits = [h for h in hits if "builtins" not in os.path.basename(h)]
        if hits:
            return hits[-1]
    for name in ("libnvrtc.so", "libnvrtc.so.12", "libnvrtc.so.11.2"):
        try:
            ctypes.CDLL(name)
            return name
        except OSError:
            pass
    return None


def find_cuda_include():
    """Return every include root NVRTC needs, as a list.

    `mma.h` and the `crt/mma.h` it includes live in DIFFERENT wheels:
      nvidia/cuda_runtime/include/mma.h
      nvidia/cuda_nvcc/include/crt/mma.h     <- yes, from the nvcc wheel
    so one -I is never enough. And the cu13 bundle must not be mixed with a cu12
    nvrtc, so anything under nvidia/cu13 goes last and only as a fallback.
    """
    cand = []
    for r in _roots():
        cand += sorted(glob.glob(os.path.join(r, "nvidia", "*", "include")))
    cand += sorted(glob.glob("/usr/local/cuda*/include"))
    cand = [d for d in cand
            if os.path.exists(os.path.join(d, "mma.h"))
            or os.path.exists(os.path.join(d, "crt", "mma.h"))]
    same_major = [d for d in cand if "cu13" not in d]
    other = [d for d in cand if "cu13" in d]
    dirs = same_major + other
    have_mma = any(os.path.exists(os.path.join(d, "mma.h")) for d in dirs)
    have_crt = any(os.path.exists(os.path.join(d, "crt", "mma.h")) for d in dirs)
    if not (have_mma and have_crt):
        return None
    return dirs


def _check(fn, rc, lib=None, prog=None):
    if rc == 0:
        return
    msg = f"{fn} failed with code {rc}"
    if lib is not None and prog is not None:
        n = ctypes.c_size_t()
        lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
        buf = ctypes.create_string_buffer(n.value)
        lib.nvrtcGetProgramLog(prog, buf)
        msg += "\n" + buf.value.decode(errors="replace")
    raise RuntimeError(msg)


def compile_ptx(arch, use_xor=False):
    """NVRTC-compile the shared kernel header into PTX for sm_<arch>."""
    so = find_nvrtc()
    if so is None:
        raise RuntimeError("libnvrtc not found; pip install nvidia-cuda-nvrtc-cu12")
    inc = find_cuda_include()
    if inc is None:
        raise RuntimeError(
            "CUDA headers incomplete. mma.h and crt/mma.h come from different wheels:\n"
            "  pip install nvidia-cuda-runtime-cu12 nvidia-cuda-nvcc-cu12\n"
            "(the nvcc wheel is needed for its crt/ headers even though it ships no nvcc)")

    src = open(KERNEL).read()
    if use_xor:
        src = "#define USE_XOR 1\n" + src
    src = src.replace("#pragma once", "")

    lib = ctypes.CDLL(so)
    prog = ctypes.c_void_p()
    _check("nvrtcCreateProgram", lib.nvrtcCreateProgram(
        ctypes.byref(prog), src.encode(), b"b1_gemm.cu", 0, None, None))

    opts = [f"--gpu-architecture=compute_{arch}".encode(), b"--std=c++17",
            b"-default-device"] + [f"-I{d}".encode() for d in inc]
    arr = (ctypes.c_char_p * len(opts))(*opts)
    _check("nvrtcCompileProgram", lib.nvrtcCompileProgram(prog, len(opts), arr), lib, prog)

    n = ctypes.c_size_t()
    lib.nvrtcGetPTXSize(prog, ctypes.byref(n))
    buf = ctypes.create_string_buffer(n.value)
    lib.nvrtcGetPTX(prog, buf)
    lib.nvrtcDestroyProgram(ctypes.byref(prog))
    return buf.value


class B1Kernel:
    """PTX loaded through the driver API and launched on torch's stream."""

    BM = BN = 64
    THREADS = 256

    def __init__(self, arch, use_xor=False):
        import torch
        self.torch = torch
        ptx = compile_ptx(arch, use_xor)
        self.cu = ctypes.CDLL("libcuda.so.1")
        self.cu.cuInit(0)
        ctx = ctypes.c_void_p()
        self.cu.cuCtxGetCurrent(ctypes.byref(ctx))
        if not ctx.value:
            torch.zeros(1, device="cuda")  # force torch to create a context
            self.cu.cuCtxGetCurrent(ctypes.byref(ctx))
        mod = ctypes.c_void_p()
        rc = self.cu.cuModuleLoadData(ctypes.byref(mod), ptx)
        if rc != 0:
            raise RuntimeError(f"cuModuleLoadData failed: {rc}")
        self.fn = ctypes.c_void_p()
        rc = self.cu.cuModuleGetFunction(ctypes.byref(self.fn), mod, b"b1_gemm")
        if rc != 0:
            raise RuntimeError(f"cuModuleGetFunction failed: {rc}")

    def __call__(self, A, B, C, M, N, K):
        stream = ctypes.c_void_p(self.torch.cuda.current_stream().cuda_stream)
        pA = ctypes.c_void_p(A.data_ptr())
        pB = ctypes.c_void_p(B.data_ptr())
        pC = ctypes.c_void_p(C.data_ptr())
        cM, cN, cK = ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)
        args = (ctypes.c_void_p * 6)(
            ctypes.cast(ctypes.byref(pA), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(pB), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(pC), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(cM), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(cN), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(cK), ctypes.c_void_p))
        rc = self.cu.cuLaunchKernel(self.fn,
                                    N // self.BN, M // self.BM, 1,
                                    self.THREADS, 1, 1,
                                    0, stream, args, None)
        if rc != 0:
            raise RuntimeError(f"cuLaunchKernel failed: {rc}")


def benchmark(arch, shapes, rounds=6, use_xor=False, verbose=True):
    """Interleaved A/B of the b1 kernel against torch bf16, inside one process.

    Interleaving is not a nicety. On a power-capped device the same binary measured
    4.19x and 1.93x in two states, because the governor trades clock for watts
    differently per kernel. Two separate runs measure the machine, not the kernel.
    """
    import torch
    k = B1Kernel(arch, use_xor=use_xor)
    results = {}
    for label, (M, N, K) in shapes.items():
        kw = K // 32
        A = torch.randint(-2**31, 2**31 - 1, (M, kw), device="cuda", dtype=torch.int32)
        B = torch.randint(-2**31, 2**31 - 1, (N, kw), device="cuda", dtype=torch.int32)
        C = torch.zeros((M, N), device="cuda", dtype=torch.int32)

        k(A, B, C, M, N, K)
        torch.cuda.synchronize()

        # correctness against a CPU popcount reference on a few entries
        Ah, Bh, Ch = A.cpu().numpy(), B.cpu().numpy(), C.cpu().numpy()
        bad = 0
        for t in range(16):
            m, n = (t * 37) % M, (t * 53) % N
            au = Ah[m].astype("uint32")
            bu = Bh[n].astype("uint32")
            v = (au ^ bu) if use_xor else (au & bu)
            ref = int(sum(int(x).bit_count() for x in v))
            if ref != int(Ch[m, n]):
                bad += 1
        ok = "PASS" if bad == 0 else f"FAIL ({bad}/16)"

        fa = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        fb = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

        def time_b1():
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(20):
                k(A, B, C, M, N, K)
            e.record(); torch.cuda.synchronize()
            return 2.0 * M * N * K * 20 / (s.elapsed_time(e) * 1e-3) / 1e12

        def time_bf16():
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(20):
                fa @ fb
            e.record(); torch.cuda.synchronize()
            return 2.0 * M * N * K * 20 / (s.elapsed_time(e) * 1e-3) / 1e12

        time_b1(); time_bf16()
        ratios = []
        for r in range(rounds):
            t1 = time_b1(); t2 = time_bf16(); t1b = time_b1()
            b1 = (t1 + t1b) / 2
            ratios.append(b1 / t2)
            if verbose:
                print(f"  {label:<16} round {r}  b1 {b1:7.2f} TOPS  bf16 {t2:7.2f} TFLOPS  "
                      f"{b1/t2:6.2f}x")
        med = sorted(ratios)[len(ratios) // 2]
        op = "XOR" if use_xor else "AND"
        print(f"  {label:<16} {op} correctness={ok}  median ratio: {med:.2f}x")
        results[label + ("" if not use_xor else " (XOR)")] = med
        del A, B, C, fa, fb
        torch.cuda.empty_cache()
    return results
