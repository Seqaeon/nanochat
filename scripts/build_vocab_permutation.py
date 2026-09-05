"""Build a vocabulary permutation for ``--sch-monarch-perm=file``.

``MonarchHead`` gives word ``w`` only the ``m1`` features of block ``w // block_out``.
Which words share a block is therefore a real architectural choice, and today it is
made by token id, which is BPE merge order: roughly frequency-stratified, never
semantically coherent. A block only has to separate the words inside its own shard,
so a coherent shard should need fewer than ``m1`` dimensions to do it.

The permutation costs nothing at run time: no parameters, no FLOPs, one buffer. This
script writes ``perm`` with ``perm[p] = t``, meaning block position ``p`` carries
token ``t``.

Modes
  freq     descending unigram frequency; block 0 owns the head of the distribution
  cluster  balanced k-means over a trained checkpoint's vocabulary rows
  random   the control, and the arm that decides what a cluster win means

Blocks must come out exactly equal, so ``cluster`` does not use plain k-means: an
unbalanced assignment would change the head's geometry and confound the comparison.
It solves a capacitated assignment instead, described at ``balanced_assign``.

Examples
  python -m scripts.build_vocab_permutation --mode freq  --vocab-size 131072 \
      --tokenizer-dir tokenizer_131k --out perms/freq_131k.pt
  python -m scripts.build_vocab_permutation --mode cluster --blocks 8 \
      --checkpoint out/c08_vocab131k/d12/BASE_dense_s1/.../model_007434.pt \
      --source lm_head --out perms/cluster_m2_8.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanochat.code_head import load_freq_table


def vocab_rows(checkpoint: str, source: str, vocab_size: int) -> torch.Tensor:
    """Pull the (V, d) matrix whose row geometry we are clustering.

    ``lm_head`` is the honest choice: it is the matrix Monarch replaces, so its row
    similarity is exactly the structure a block has to reproduce. ``wte`` is the
    fallback for a tied or non-dense checkpoint.
    """
    sd = torch.load(checkpoint, weights_only=True, map_location="cpu")
    sd = sd.get("model", sd)
    keys = {"lm_head": ("lm_head.weight", "_orig_mod.lm_head.weight"),
            "wte": ("transformer.wte.weight", "_orig_mod.transformer.wte.weight")}[source]
    for k in keys:
        if k in sd:
            rows = sd[k].float()
            assert rows.shape[0] >= vocab_size, (
                f"{k} has {rows.shape[0]} rows, fewer than vocab_size={vocab_size}")
            return rows[:vocab_size]
    raise SystemExit(
        f"no {source} matrix in {checkpoint}; tried {keys}, found e.g. "
        f"{sorted(sd)[:6]}. A code or Monarch checkpoint has no dense lm_head, so "
        "cluster from a dense run.")


def balanced_assign(rows: torch.Tensor, blocks: int, iters: int, seed: int) -> torch.Tensor:
    """k-means, then a capacitated assignment that forces exactly V/blocks per block.

    Plain k-means gives clusters of wildly different sizes and the head needs equal
    ones, so the last step re-assigns under capacity: tokens are ordered by how much
    they prefer their best centroid over their second best, and the most opinionated
    are placed first. A token whose first choice is full falls through to its next
    available one, so the tokens that pay for the balancing are the ones that cared
    least about the outcome.
    """
    V = rows.shape[0]
    assert V % blocks == 0, f"vocab_size={V} is not divisible by blocks={blocks}"
    cap = V // blocks
    x = torch.nn.functional.normalize(rows, dim=1)     # direction, not magnitude
    gen = torch.Generator().manual_seed(seed)
    centroids = x[torch.randperm(V, generator=gen)[:blocks]].clone()

    for it in range(iters):
        assign = (x @ centroids.t()).argmax(dim=1)
        for j in range(blocks):
            sel = x[assign == j]
            if sel.numel():
                centroids[j] = torch.nn.functional.normalize(sel.mean(0), dim=0)
        print(f"  k-means iter {it + 1}/{iters}  sizes "
              f"{torch.bincount(assign, minlength=blocks).tolist()}")

    sim = x @ centroids.t()                             # (V, blocks)
    order_pref = sim.argsort(dim=1, descending=True)
    top2 = sim.topk(2, dim=1).values
    confidence = top2[:, 0] - top2[:, 1]
    final = torch.full((V,), -1, dtype=torch.long)
    room = [cap] * blocks
    for t in torch.argsort(confidence, descending=True).tolist():
        for j in order_pref[t].tolist():
            if room[j]:
                final[t], room[j] = j, room[j] - 1
                break
    assert (final >= 0).all()
    return final


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("freq", "cluster", "random"), required=True)
    ap.add_argument("--out", required=True, help="destination .pt")
    ap.add_argument("--vocab-size", type=int, default=0,
                    help="0 = infer from the frequency table or checkpoint")
    ap.add_argument("--blocks", type=int, default=0, help="m2, required for --mode=cluster")
    ap.add_argument("--checkpoint", default="", help="model_*.pt to cluster from")
    ap.add_argument("--source", choices=("lm_head", "wte"), default="lm_head")
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if args.mode == "cluster":
        assert args.blocks > 1, "--mode=cluster needs --blocks (the head's m2)"
        assert args.checkpoint, "--mode=cluster needs --checkpoint"
        assert args.vocab_size, "--mode=cluster needs an explicit --vocab-size"
        rows = vocab_rows(args.checkpoint, args.source, args.vocab_size)
        block_of = balanced_assign(rows, args.blocks, args.iters, args.seed)
        # perm[p] = t: walk the blocks in order and lay their members out.
        perm = torch.argsort(block_of, stable=True)
    elif args.mode == "freq":
        freqs = load_freq_table(args.vocab_size or 1, args.tokenizer_dir)
        assert freqs is not None, (
            f"no freq_table.pt under {args.tokenizer_dir or 'the default tokenizer dir'}; "
            "scripts/ensure_tokenizer.py builds it")
        V = args.vocab_size or freqs.numel()
        perm = torch.argsort(freqs[:V], descending=True, stable=True)
    else:
        assert args.vocab_size, "--mode=random needs --vocab-size"
        perm = torch.randperm(args.vocab_size,
                              generator=torch.Generator().manual_seed(args.seed))

    V = perm.numel()
    assert torch.equal(torch.sort(perm).values, torch.arange(V)), "not a permutation"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(perm, args.out)
    moved = int((perm != torch.arange(V)).sum())
    print(f"[perm] wrote {args.out}: {V:,} entries, {moved:,} moved ({moved / V:.1%})")


if __name__ == "__main__":
    main()
