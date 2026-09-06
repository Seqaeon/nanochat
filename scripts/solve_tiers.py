"""Solve the optimal per-tier capacity for a tiered output head.

With disjoint frequency tiers, captured energy is ADDITIVE across tiers and CONCAVE
in each tier's capacity: the k-th dimension given to tier t buys that tier's k-th
eigenvalue and costs (d + V_t) MACs. A separable concave problem under one budget is
solved exactly by a greedy over marginal value per MAC, so this is not a search --
it returns the optimum for the given boundaries in one pass.

The evidence it reads is a trained DENSE head: the architecture is a hypothesis about
how that head's V row vectors share subspaces, and this measures which hypothesis the
budget can afford.

Weighting matters more than any other choice here. Plain Frobenius counts all 131,072
words equally; the loss weights word w by roughly p(w), and at V=131,072 the bottom
half of the vocabulary carries 1.7% of the mass. The two weightings disagree sharply
about the tail (weighted wants 18 dims there, unweighted wants 233), so run both and
train the pair: which one wins is itself the finding.

  python -m scripts.solve_tiers head_d8_v131k.pt --match-monarch 1024 32 224 \
      --bounds 1024,4096,16384,65536 --freq-table tokenizer_131k/freq_table.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.head_oracle import load_head


def solve(U: torch.Tensor, order: torch.Tensor, edges: list[int], budget: float, d: int):
    """Greedy over marginal eigenvalue per MAC. Exact for this problem."""
    evs, sizes = [], []
    for i in range(len(edges) - 1):
        Ri = U[order[edges[i]:edges[i + 1]]]
        ev = torch.linalg.eigvalsh(Ri.T.double() @ Ri.double()).flip(0).clamp_min(0)
        evs.append(ev.cpu())
        sizes.append(edges[i + 1] - edges[i])
    cand = []
    for t, (ev, n) in enumerate(zip(evs, sizes)):
        for k in range(min(len(ev), d)):
            cand.append((float(ev[k]) / (d + n), t, float(ev[k])))
    cand.sort(reverse=True)
    caps = [0] * len(sizes)
    spent = gained = 0.0
    for _, t, val in cand:
        step = d + sizes[t]
        if spent + step > budget:
            continue
        caps[t] += 1
        spent += step
        gained += val
    return caps, spent, gained


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="dense lm_head: full checkpoint or bare (V, d) tensor")
    ap.add_argument("--bounds", default="1024,4096,16384,65536",
                    help="cumulative vocabulary edges, frequency-ordered")
    ap.add_argument("--budget", type=float, default=0.0, help="MACs/token for the head")
    ap.add_argument("--match-monarch", type=int, nargs=3, metavar=("M", "M1", "R"),
                    help="take the budget from a Monarch arm: d*M + V*m1 + r*(d+V)")
    ap.add_argument("--freq-table", default=None,
                    help="weight each word by how often it occurs (what the loss does)")
    ap.add_argument("--vocab-size", type=int, default=0, help="0 = infer from the head")
    args = ap.parse_args()

    W = load_head(args.checkpoint)
    V = args.vocab_size or W.shape[0]
    W, d = W[:V], W.shape[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if args.match_monarch:
        M, m1, r = args.match_monarch
        budget = d * M + V * m1 + r * (d + V)
    elif args.budget:
        budget = args.budget
    else:
        raise SystemExit("give --budget or --match-monarch")

    if args.freq_table:
        ft = torch.load(args.freq_table, weights_only=True, map_location="cpu").float()
        if ft.numel() < V:
            ft = torch.cat([ft, torch.zeros(V - ft.numel())])
        p = ft[:V] / ft[:V].sum()
        w = p.sqrt()
        U = ((W - (W * w[:, None]).sum(0) / w.sum()) * w[:, None]).to(dev)
        order = torch.argsort(p, descending=True)
    else:
        # Without frequencies the tiers cannot be frequency-ordered, so this scores
        # the boundaries as given in token-id order. Useful as the control, not as a
        # design: the ordering is where the structure is.
        p = None
        U = (W - W.mean(0, keepdim=True)).to(dev)
        order = torch.arange(V)
    order = order.to(dev)

    edges = [0] + [int(b) for b in args.bounds.replace(" ", ",").split(",") if b] + [V]
    assert all(edges[i] < edges[i + 1] for i in range(len(edges) - 1)), f"bad edges {edges}"

    total = float(U.pow(2).sum().double())
    lr = int(budget / (d + V))
    ev_all = torch.linalg.eigvalsh(U.T.double() @ U.double()).flip(0).clamp_min(0)
    base = 1.0 - (total - float(ev_all[:min(lr, d)].sum())) / total

    caps, spent, gained = solve(U, order, edges, budget, d)
    capture = 1.0 - (total - gained) / total

    print(f"head {V:,} x {d}   budget {budget / 1e6:.3f}M MACs/token"
          f"   weighting: {'frequency' if p is not None else 'uniform'}")
    print(f"cost-matched pure low-rank rank {lr} captures {base:.4f}\n")
    print(f"  {'tier':>5} {'words':>9} {'dims':>6} {'MACs':>11} {'share':>7}")
    for i, c in enumerate(caps):
        n = edges[i + 1] - edges[i]
        mac = c * (d + n)
        mass = f"{float(p[order[edges[i]:edges[i+1]].cpu()].sum()):6.2%}" if p is not None else "     -"
        print(f"  {i:>5} {n:>9,} {c:>6} {mac / 1e6:>10.3f}M {mass:>7}")
    print(f"\n  spent {spent / 1e6:.3f}M ({spent / budget:.4f}x budget)"
          f"   captures {capture:.4f}   vs low-rank {capture - base:+.4f}\n")
    print("  --sch-head-type tiered \\")
    print(f"  --sch-tier-bounds {','.join(str(e) for e in edges[1:-1])} \\")
    print(f"  --sch-tier-caps {','.join(map(str, caps))} \\")
    print(f"  --sch-tier-order {'freq' if p is not None else 'none'}")


if __name__ == "__main__":
    main()
