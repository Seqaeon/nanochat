"""O3: is a 1-bit GEMM actually faster than bf16 on this device, and by how much?

The gate the binary direction pivots on.  From scripts/o4_cost_model.py, at matched
inference bytes the binary model issues 24.5x the MACs of the dense one, so a b1
kernel must reach >= 24.5x over bf16 -- 38% of the Ampere spec ceiling -- for the
binary model merely to TIE on wall clock.  Below that the direction is
memory-and-energy only and the abstract has to say so.

ARCHITECTURE MATTERS AND IS CHECKED AT RUNTIME.  `b1` MMA with both the AND and XOR
operands exists on Turing (sm_75) and Ampere (sm_80/sm_86).  Ada (sm_89) dropped
INT1.  Hopper (sm_90+) removed the XOR operand from the hardware and emulates it,
measured up to 5x slower on GH200.  A headline number must come from T4, A10G or
A100, not from L4/L40S/H100/H200/B200.

Every comparison is interleaved inside one process.  Two separate runs are not
comparable: on a power-capped device the same binary here measured 4.19x and 1.93x
in two states, because the governor trades clock for watts differently per kernel.

Run:
  python -m scripts.o3_kernel_gate
  python -m scripts.o3_kernel_gate --rounds 8 --shapes square ffn head
"""
import argparse
import glob
import json
import os
import shutil
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "kernels", "b1_probe.cu")

# label -> "M N K".  M and N must be multiples of 64, K a multiple of 512.
SHAPES = {
    "square":     [("square 4096", "4096 4096 4096")],
    "ffn":        [("ffn d=512", "8192 2048 512"),
                   ("ffn d=1024", "8192 4096 1024"),
                   ("ffn d=2048", "8192 8192 2048")],
    "head":       [("head V=32k", "8192 32768 512"),
                   ("head V=131k", "4096 131072 512")],
}
BREAK_EVEN = 24.5  # from scripts/o4_cost_model.py, matched inference bytes at depth 8


def find_toolchain():
    """Locate nvcc and cuBLAS.

    A CUDA *runtime* image (which is what most GPU containers ship) has no nvcc.
    The pip wheels do: `nvidia-cuda-nvcc-cu12` carries nvcc and `nvidia-cublas-cu12`
    carries both the header and the library, and torch already pulls the latter in.
    So search PATH, then CUDA_HOME, then site-packages, and build the include and
    link flags from whatever is found.  Returns (nvcc, extra_flags) or (None, hint).
    """
    import site
    roots = list(site.getsitepackages())
    try:
        roots.append(site.getusersitepackages())
    except Exception:
        pass

    roots.append(os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}."
                              f"{sys.version_info.minor}", "site-packages"))
    roots = [r for r in dict.fromkeys(roots) if os.path.isdir(r)]

    cands = []
    w = shutil.which("nvcc")
    if w:
        cands.append(w)
    for env in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(env):
            cands.append(os.path.join(os.environ[env], "bin", "nvcc"))
    try:
        from torch.utils.cpp_extension import CUDA_HOME as TORCH_CUDA_HOME
        if TORCH_CUDA_HOME:
            cands.append(os.path.join(TORCH_CUDA_HOME, "bin", "nvcc"))
    except Exception:
        pass
    for r in roots:
        cands += sorted(glob.glob(os.path.join(r, "nvidia", "*", "bin", "nvcc")))
    cands += sorted(glob.glob("/usr/local/cuda*/bin/nvcc"))
    nvcc = next((c for c in cands if c and os.path.exists(c)), None)
    if nvcc is None:
        return None, ("nvcc not found. This is a CUDA runtime image without the compiler.\n"
                      "  pip install -q nvidia-cuda-nvcc-cu12\n"
                      "then re-run in a FRESH process (a notebook kernel that already\n"
                      "imported torch will not see the new package on sys.path).\n"
                      "Searched: PATH, CUDA_HOME, CUDA_PATH, torch's CUDA_HOME,\n"
                      "site-packages/nvidia/*/bin/nvcc, /usr/local/cuda*/bin/nvcc.\n"
                      "No system CUDA toolkit is required.")

    # Find the library FIRST, then take the header from the SAME package, so a
    # cu13 header is never paired with a cu12 library.
    flags = []
    libpath = None
    for r in roots:
        hits = sorted(glob.glob(os.path.join(r, "nvidia", "*", "lib", "libcublas.so*")))
        if hits:
            libpath = hits[-1]
            break

    incdir = None
    if libpath:
        sibling = os.path.join(os.path.dirname(os.path.dirname(libpath)), "include")
        if os.path.exists(os.path.join(sibling, "cublas_v2.h")):
            incdir = sibling
    if incdir is None:
        for r in roots:
            for d in sorted(glob.glob(os.path.join(r, "nvidia", "*", "include"))):
                if os.path.exists(os.path.join(d, "cublas_v2.h")):
                    incdir = d
                    break
            if incdir:
                break
    if incdir:
        flags.append(f"-I{incdir}")

    if libpath:
        # pip wheels ship only the versioned soname, so plain -lcublas will not
        # resolve, and nvcc refuses a bare ".so.12" path as an input file.  Go
        # through the host linker by soname and bake in an rpath so it runs.
        libdir, soname = os.path.dirname(libpath), os.path.basename(libpath)
        flags.append(f"-L{libdir}")
        flags.append(f"-Xlinker -l:{soname}")
        flags.append(f"-Xlinker -rpath={libdir}")
    else:
        flags.append("-lcublas")
    return nvcc, " ".join(flags)


def sh(cmd, quiet=False):
    if not quiet:
        print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout:
        print(r.stdout, end="", flush=True)
    if r.returncode != 0 and r.stderr:
        print("STDERR:", r.stderr[:2000], file=sys.stderr, flush=True)
    return r


RATIO_RE = re.compile(r"median ratio .*?:\s*([0-9.]+)x")


def parse_ratio(stdout):
    m = RATIO_RE.search(stdout or "")
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--shapes", nargs="+", default=["square", "ffn", "head"],
                    choices=list(SHAPES))
    ap.add_argument("--outdir", default="out/b00_binary_phase0")
    a = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("no CUDA device; O3 cannot run"); return 1
    p = torch.cuda.get_device_properties(0)
    arch = p.major * 10 + p.minor
    print(f"device: {p.name}  sm_{arch}  {p.total_memory/2**30:.1f} GiB  "
          f"{p.multi_processor_count} SMs")

    if arch == 89:
        print("REFUSING: sm_89 (Ada) dropped INT1 tensor ops. Use T4, A10G or A100.")
        return 1
    xor_native = arch < 90
    if not xor_native:
        print("WARNING: sm_90+ removed the b1 XOR operand from hardware and emulates it")
        print("         (up to 5x slower on GH200). AND mode only, and this device must")
        print("         NOT carry the paper's headline number.")

    nvcc, extra = find_toolchain()
    if nvcc is None:
        print(extra)
        return 1
    print(f"nvcc: {nvcc}")
    print(f"link: {extra}")

    os.makedirs(a.outdir, exist_ok=True)
    and_bin = os.path.join(a.outdir, "b1_and")
    xor_bin = os.path.join(a.outdir, "b1_xor")
    if sh(f"{nvcc} -O3 -arch=sm_{arch} {SRC} {extra} -o {and_bin}").returncode:
        return 1
    if xor_native:
        sh(f"{nvcc} -O3 -arch=sm_{arch} -DUSE_XOR {SRC} {extra} -o {xor_bin}")

    print()
    sh("nvidia-smi -q -d POWER | grep -iE 'current power limit|default power limit'", quiet=True)
    sh("nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu --format=csv", quiet=True)
    print("If current and default power limits differ, the governor is trading clock for")
    print("watts and the ratios below are a property of the cap, not of the kernel.")
    print()

    measured = {}
    for group in a.shapes:
        for label, dims in SHAPES[group]:
            print(f"=== {label}  ({dims})  AND ===")
            r = sh(f"{and_bin} {dims} {a.rounds}", quiet=True)
            v = parse_ratio(r.stdout)
            if v is not None:
                measured[label] = v
            if xor_native and group == "square":
                print(f"=== {label}  ({dims})  XOR ===")
                rx = sh(f"{xor_bin} {dims} {a.rounds}", quiet=True)
                vx = parse_ratio(rx.stdout)
                if vx is not None:
                    measured[label + " (XOR)"] = vx
            print()

    # Write what was MEASURED so scripts/o4_cost_model.py stops reprinting stale
    # placeholders from a different machine.
    jpath = os.path.join(a.outdir, "o3_kernel_gate.json")
    with open(jpath, "w") as f:
        json.dump({"device": p.name, "sm": arch, "xor_native": xor_native,
                   "rounds": a.rounds, "ratios": measured}, f, indent=2)
    print(f"wrote {jpath}")
    if measured:
        best = max(measured.values())
        print(f"best measured ratio on THIS device: {best:.2f}x  "
              f"({'CLEARS' if best >= BREAK_EVEN else 'BELOW'} the {BREAK_EVEN}x break-even)")

    print("=" * 72)
    print(f"GATE: a b1 kernel needs >= {BREAK_EVEN}x over bf16 for the matched-bytes binary")
    print("model to tie on wall clock. Below that: drop the speed claim, keep memory and")
    print("energy, and say so in the abstract rather than burying it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
