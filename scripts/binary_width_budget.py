"""Solve for the binary model width that matches a dense model's BYTES.

Derived, never typed: a hand-matched width is correct once and then silently
compares two budgets. Mirrors scripts/code_head_budget.py, which does the same job
for token budgets.

The answer is NOT 16x dense's width. The vocabulary interfaces (wte, lm_head,
value_embeds) scale LINEARLY in d while the body scales quadratically, so at
V=32,768 the interfaces eat roughly a third of the budget and the reachable width
is ~7x. That matters: plan section 3.8 needs n ~= 13,000 to close the accumulation
entropy gap, and this says how close equal bytes actually gets.

Run: python -m scripts.binary_width_budget --depth 8 --vocab 32768
"""
import argparse
import math

import torch

from nanochat.gpt import GPT, GPTConfig

HEAD_DIM = 128


def build(depth, d, vocab, seq=2048, wp="SSSL", binary=False):
    cfg = GPTConfig(sequence_len=seq, vocab_size=vocab, n_layer=depth,
                    n_head=max(1, d // HEAD_DIM), n_kv_head=max(1, d // HEAD_DIM),
                    n_embd=d, window_pattern=wp, use_binary=binary)
    with torch.device("meta"):
        return GPT(cfg)


def bytes_of(model, bits_per_param):
    n = sum(p.numel() for p in model.parameters())
    return n * bits_per_param / 8.0, n


def accum_bits(n):
    """H = 0.5*log2(2*pi*e*n) - 1, verified against measurement at n=512 and 8192."""
    return 0.5 * math.log2(2 * math.pi * math.e * n) - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--dense-dim", type=int, default=512)
    ap.add_argument("--dense-bits", type=int, default=16)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    dense = build(a.depth, a.dense_dim, a.vocab, a.seq)
    budget, dn = bytes_of(dense, a.dense_bits)

    best = None
    rows = []
    for d in range(HEAD_DIM, 200 * HEAD_DIM + 1, HEAD_DIM):
        m = build(a.depth, d, a.vocab, a.seq, binary=True)
        b, n = bytes_of(m, 1)
        if b > budget:
            break
        best = (d, n, b)
        rows.append((d, n, b))

    if not a.quiet:
        print(f"dense: depth {a.depth}, d={a.dense_dim}, V={a.vocab:,}, "
              f"{dn:,} params at {a.dense_bits} bits = {budget/2**20:.1f} MiB")
        print(f"  accumulation entropy at n={a.dense_dim}: fp is not fan-in limited")
        print()
        print(f"{'binary d':>9}{'params':>15}{'MiB':>8}{'x dense d':>11}{'accum bits':>12}")
        for d, n, b in rows[-6:]:
            print(f"{d:>9}{n:>15,}{b/2**20:>8.1f}{d/a.dense_dim:>10.1f}x{accum_bits(d):>12.2f}")
        d, n, b = best
        print()
        print(f"MATCHED-BYTES WIDTH: d={d} ({d/a.dense_dim:.1f}x dense), "
              f"{n:,} params, {b/2**20:.1f} MiB of {budget/2**20:.1f}")
        print(f"  accumulation entropy {accum_bits(d):.2f} bits "
              f"(fp measured ~7.9; parity needs n ~= 13,000)")
        print(f"  NOT 16x: the interfaces scale linearly in d and take "
              f"{100*(2*a.vocab*d + (a.depth//2)*a.vocab*d)/n:.0f}% of the budget")
    print(best[0])


if __name__ == "__main__":
    main()
