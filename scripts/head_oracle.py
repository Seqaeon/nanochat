"""Score output-head architectures against a trained dense head, without training.

Every head in this family constrains where a word's row vector may live:

    dense       u_w is free in R^d
    low-rank    all u_w share one R-dim subspace
    Monarch     u_w lies in block(w)'s m1-dim coordinate slice
    + residual  u_w lies in block(w)'s m1-dim slice PLUS an r-dim shared subspace

So the architecture is a hypothesis about how a trained head's V row vectors cluster
into shared subspaces, and a trained dense head is the evidence. This measures, for a
fixed per-word capacity, how much of that head's logit energy each hypothesis can
reproduce -- in seconds, against runs that cost hours.

The metric is captured Frobenius energy of the row-centred head. Centring removes the
component identical across all words, which softmax invariance makes free, so it is
not credit any architecture should get.

Run:  python -m scripts.head_oracle <model_*.pt> [--capacity 160] [--blocks 8]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_head(path: str) -> torch.Tensor:
    """Accept a full checkpoint, a one-key dict, or a bare (V, d) tensor.

    The bare form matters: at V=131,072 a full checkpoint is gigabytes while the head
    alone is a few hundred megabytes, so extracting it before moving it off the
    training box is the difference between a convenient analysis and an impossible one.
    """
    obj = torch.load(path, weights_only=True, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj.float()
    sd = obj.get("model", obj)
    for k in ("lm_head.weight", "_orig_mod.lm_head.weight"):
        if k in sd:
            return sd[k].float()
    raise SystemExit(
        f"no dense lm_head in {path}; found {sorted(sd)[:6]}. A code or Monarch "
        "checkpoint has no dense head, so use a dense run.")


def topk_basis(X: torch.Tensor, k: int) -> torch.Tensor:
    """Right singular vectors of X spanning its k dominant row directions."""
    if k <= 0:
        return X.new_zeros(X.shape[1], 0)
    k = min(k, X.shape[0], X.shape[1])
    return torch.linalg.svd(X, full_matrices=False).Vh[:k].T


def captured(U: torch.Tensor, blocks: torch.Tensor, m1: int, r: int, n_blocks: int) -> float:
    """Fraction of energy an (r shared + m1 per-block) head can reproduce.

    Greedy in the same order the architecture is: the shared basis is global and the
    block bases explain what it leaves. That is a slight underestimate against joint
    training, and identical across the assignments being compared, which is what the
    comparison needs.
    """
    S = topk_basis(U, r)
    R = U - (U @ S) @ S.T if r else U.clone()
    err = 0.0
    for j in range(n_blocks):
        Rj = R[blocks == j]
        if Rj.numel() == 0:
            continue
        B = topk_basis(Rj, m1)
        resid = Rj - (Rj @ B) @ B.T if m1 else Rj
        err += resid.pow(2).sum().item()
    return 1.0 - err / U.pow(2).sum().item()


def balanced_clusters(U: torch.Tensor, n_blocks: int, iters: int, seed: int) -> torch.Tensor:
    """k-means on row DIRECTION, then a capacitated assignment to equal blocks.

    Direction, not magnitude: two words with the same direction and different norms
    share a one-dimensional subspace, which is exactly what a block must supply.
    """
    V = U.shape[0]
    cap = V // n_blocks
    X = torch.nn.functional.normalize(U, dim=1)
    g = torch.Generator().manual_seed(seed)
    C = X[torch.randperm(V, generator=g)[:n_blocks]].clone()
    for _ in range(iters):
        a = (X @ C.T).argmax(1)
        for j in range(n_blocks):
            sel = X[a == j]
            if sel.numel():
                C[j] = torch.nn.functional.normalize(sel.mean(0), dim=0)
    sim = X @ C.T
    pref = sim.argsort(1, descending=True)
    top2 = sim.topk(2, dim=1).values
    out = torch.full((V,), -1, dtype=torch.long)
    room = [cap] * n_blocks
    for t in torch.argsort(top2[:, 0] - top2[:, 1], descending=True).tolist():
        for j in pref[t].tolist():
            if room[j]:
                out[t], room[j] = j, room[j] - 1
                break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--capacity", type=int, default=160, help="per-word budget m1 + r")
    ap.add_argument("--blocks", type=int, default=8, help="m2")
    ap.add_argument("--splits", type=int, nargs="*", default=None, help="m1 values to try")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    W = load_head(args.checkpoint)
    U = W - W.mean(dim=0, keepdim=True)          # drop the softmax-invariant direction
    V, d = U.shape
    c, m2 = args.capacity, args.blocks
    print(f"head {V:,} x {d}   per-word capacity {c}   blocks {m2}   "
          f"block_out {V // m2:,}\n")

    full = captured(U, torch.zeros(V, dtype=torch.long), min(c, d), 0, 1)
    print(f"  {'assignment':<12} {'m1':>4} {'r':>4} {'captured':>9}   vs global rank-{c}")
    print(f"  {'global':<12} {'-':>4} {c:>4} {full:9.4f}   (pure low-rank ceiling)")

    ids = torch.arange(V) // (V // m2)
    g = torch.Generator().manual_seed(args.seed)
    rnd = torch.randperm(V, generator=g) // (V // m2)
    clu = balanced_clusters(U, m2, args.iters, args.seed)
    splits = args.splits or [s for s in (8, 16, 32, 64, 128) if s < c]
    for name, blocks in (("token-id", ids), ("random", rnd), ("clustered", clu)):
        for m1 in splits:
            cap = captured(U, blocks, m1, c - m1, m2)
            print(f"  {name:<12} {m1:>4} {c-m1:>4} {cap:9.4f}   {cap - full:+.4f}")
    print("\n  the last column is what the block structure buys over pure low-rank at")
    print("  the same per-word cost. If it is near zero, blocks of that assignment")
    print("  share the global subspace and their private capacity is wasted.")


if __name__ == "__main__":
    main()
