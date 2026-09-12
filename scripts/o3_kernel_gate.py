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
import os
import shutil
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


def sh(cmd, quiet=False):
    if not quiet:
        print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout:
        print(r.stdout, end="", flush=True)
    if r.returncode != 0 and r.stderr:
        print("STDERR:", r.stderr[:2000], file=sys.stderr, flush=True)
    return r


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

    if shutil.which("nvcc") is None:
        print("nvcc not found; install the CUDA toolkit (the runtime alone is not enough)")
        return 1

    os.makedirs(a.outdir, exist_ok=True)
    and_bin = os.path.join(a.outdir, "b1_and")
    xor_bin = os.path.join(a.outdir, "b1_xor")
    if sh(f"nvcc -O3 -arch=sm_{arch} {SRC} -lcublas -o {and_bin}").returncode:
        return 1
    if xor_native:
        sh(f"nvcc -O3 -arch=sm_{arch} -DUSE_XOR {SRC} -lcublas -o {xor_bin}")

    print()
    sh("nvidia-smi -q -d POWER | grep -iE 'current power limit|default power limit'", quiet=True)
    sh("nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu --format=csv", quiet=True)
    print("If current and default power limits differ, the governor is trading clock for")
    print("watts and the ratios below are a property of the cap, not of the kernel.")
    print()

    for group in a.shapes:
        for label, dims in SHAPES[group]:
            print(f"=== {label}  ({dims})  AND ===")
            sh(f"{and_bin} {dims} {a.rounds}", quiet=True)
            if xor_native and group == "square":
                print(f"=== {label}  ({dims})  XOR ===")
                sh(f"{xor_bin} {dims} {a.rounds}", quiet=True)
            print()

    print("=" * 72)
    print(f"GATE: a b1 kernel needs >= {BREAK_EVEN}x over bf16 for the matched-bytes binary")
    print("model to tie on wall clock. Below that: drop the speed claim, keep memory and")
    print("energy, and say so in the abstract rather than burying it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
