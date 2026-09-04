"""Subspace diagnostics for structured code output heads.

The rank probe in ``nanochat/code_metrics.py`` answers "how many dimensions can
this head reach". This script answers the question that turned out to matter
more: "are they the *right* dimensions".

A code head can only ever emit logit vectors inside ``colspace(Phi)``, an
M-dimensional subspace of R^V that is fixed before training. A dense softmax
emits inside ``colspace(W)``, a d-dimensional subspace that SGD is free to
place anywhere. Two heads can therefore have identical rank and wildly
different loss. This script measures the overlap directly, on the CPU, from a
trained dense checkpoint, so that code assignments can be screened before any
of them costs GPU time.

Definitions, for a trained dense head W (V x d):

  Wc          W with its vocabulary-axis mean removed, since a constant shift
              across the vocabulary is free under the softmax.
  capture     ||P_Phi Wc||_F^2 / ||Wc||_F^2, the fraction of the dense head's
              logit energy that lies inside colspace(Phi).
  oracle      the same quantity for the best possible M-dimensional subspace,
              which is the span of the top M left singular vectors of Wc.
  +bias       capture of colspace(Phi) augmented with one freely chosen
              direction, which is what a learnable per-token bias buys.
  residual    capture of Wc after its single dominant direction is removed.
              That dominant direction is close to unigram frequency and a bias
              term captures it exactly and for free, so the residual is where
              all conditional structure lives, and it is the honest number.

Usage:
    python -m scripts.code_head_subspace --checkpoint path/to/model_XXXX.pt
"""

import argparse
import sys

import torch

from nanochat.code_head import (
    build_codes,
    build_phi_monomial,
    build_phi_random_binary,
    enumerate_monomials,
)


def learned_direction_count(S: torch.Tensor, V: int, d: int, seed: int = 0) -> tuple[int, float]:
    """How many of the head's directions are learned rather than left at init.

    An untrained head is isotropic: its singular values follow the
    Marchenko-Pastur bulk. Training concentrates energy into a few directions
    and leaves the rest near their initial values. Comparing the spectrum
    against a random matrix of the *same Frobenius norm* therefore separates
    signal from initialisation, and the count of directions above that bulk is
    the number this checkpoint can support conclusions about.

    This gate exists because without it the tool is actively misleading. A
    100-step checkpoint has two learned directions and 254 of noise; every
    candidate basis then scores at the random-subspace baseline M/V, and a
    basis *fitted* to the noise scores far higher, which reads exactly like a
    real result and is not one.
    """
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(V, d, generator=g, dtype=torch.float64)
    R *= (S.norm() / R.norm())
    Sr = torch.linalg.svdvals(R - R.mean(dim=0, keepdim=True))
    n = int((S > Sr).sum())
    share = float((S[:n] ** 2).sum() / (S ** 2).sum()) if n else 0.0
    return n, share


def load_dense_head(path: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return (lm_head.weight, transformer.wte.weight) as fp32 on the CPU."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if "lm_head.weight" not in sd:
        raise SystemExit(
            f"{path} has no 'lm_head.weight'. This tool needs a *dense* baseline "
            f"checkpoint: a code head stores 'codes' and g's weights instead."
        )
    W = sd["lm_head.weight"].float()
    if not torch.isfinite(W).all():
        raise SystemExit(
            f"{path} lm_head is not finite. nanochat poisons unused parameters "
            f"with NaN, so this checkpoint is from a diverged or aborted run."
        )
    wte = sd.get("transformer.wte.weight")
    return W, (wte.float() if wte is not None else None)


def orthonormal_basis(phi: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis of colspace(phi), centred on the vocabulary axis."""
    P = phi.float()
    P = (P - P.mean(dim=0, keepdim=True)).double()
    Q, _ = torch.linalg.qr(P)
    return Q


def analyse(name, C, bits, orders, Wc, Wr, total, total_res, cum, cum_res, rows):
    for order in orders:
        groups = enumerate_monomials(bits, order)
        phi = build_phi_monomial(C, groups, torch.float32, "cpu")
        M = phi.shape[1]
        Q = orthonormal_basis(phi)

        capture = float((Q.T @ Wc).pow(2).sum() / total)
        residual = float((Q.T @ Wr).pow(2).sum() / total_res)

        # A learnable per-token bias adds exactly one freely chosen direction to
        # the reachable vocabulary subspace. Give it the best one available.
        R = Wc - Q @ (Q.T @ Wc)
        u = torch.linalg.svd(R, full_matrices=False)[0][:, :1]
        Qb, _ = torch.linalg.qr(torch.cat([Q, u], dim=1))
        with_bias = float((Qb.T @ Wc).pow(2).sum() / total)

        oracle = float(cum[min(M, len(cum)) - 1])
        oracle_res = float(cum_res[min(M, len(cum_res)) - 1])
        rows.append((name, order, M, capture, with_bias, residual, oracle, oracle_res))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="a trained DENSE baseline checkpoint (model_*.pt)")
    ap.add_argument("--bits", type=int, default=15)
    ap.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--tokenizer-dir", default="tokenizer")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--force", action="store_true",
                    help="report even when the checkpoint is too early to be meaningful")
    args = ap.parse_args()

    W, wte = load_dense_head(args.checkpoint)
    V, d = W.shape
    Wc = (W - W.mean(dim=0, keepdim=True)).double()
    U, S, _ = torch.linalg.svd(Wc, full_matrices=False)
    energy = S ** 2
    total = energy.sum()
    cum = torch.cumsum(energy, 0) / total

    # Residual after the single dominant direction, which a bias captures free.
    Wr = Wc - U[:, :1] @ (U[:, :1].T @ Wc)
    total_res = (Wr ** 2).sum()
    cum_res = torch.cumsum(energy[1:], 0) / energy[1:].sum()

    n_learned, learned_share = learned_direction_count(S, V, d)

    print(f"checkpoint     {args.checkpoint}")
    print(f"dense head     V={V}  d={d}  rank={int((S > 1e-5 * S[0]).sum())}")
    print(f"learned dirs   {n_learned} of {d} rise above the random bulk, "
          f"carrying {learned_share * 100:.2f}% of the energy")
    if n_learned < 8 and not args.force:
        print()
        print(f"REFUSING TO REPORT. Only {n_learned} direction(s) in this head are learned; "
              f"the rest sit at\ntheir initialisation. Capture measured against them is capture "
              f"of noise, every basis\nwill score at the random baseline M/V, and any basis fitted "
              f"to this matrix will score high\nfor the wrong reason. Use a checkpoint trained to "
              f"convergence, or pass --force if you\nspecifically want the noise numbers.")
        return 2
    print(f"top-1 direction carries {float(energy[0] / total) * 100:.2f}% of the logit energy; "
          f"the residual {float(total_res / total) * 100:.2f}% holds all conditional structure.")
    print(f"cumulative energy: top-15 {float(cum[14]) * 100:.2f}%  "
          f"top-120 {float(cum[119]) * 100:.2f}%  top-{d} 100.00%\n")

    B = args.bits
    candidates: list[tuple[str, torch.Tensor]] = [
        ("binary (token-id bits)", build_codes(V, B, mode="binary")),
        ("random", build_codes(V, B, mode="random", seed=args.seed)),
    ]
    try:
        from nanochat.code_head import load_freq_table
        freqs = load_freq_table(V, args.tokenizer_dir)
        candidates.append(("frequency-ranked",
                           build_codes(V, B, mode="frequency", freqs=freqs)))
    except Exception as exc:
        print(f"note: frequency codes skipped ({type(exc).__name__}: {exc})\n")

    # Codes derived from the dense head's own spectrum. These are privileged:
    # they use the very matrix they are being scored against, so they act as an
    # upper bound on what any binary code assignment could achieve.
    order_u1 = torch.argsort(U[:, 0])
    rank_of = torch.empty(V, dtype=torch.long)
    rank_of[order_u1] = torch.arange(V)
    dyadic = ((rank_of.unsqueeze(1) >> torch.arange(B)) & 1).to(torch.uint8)
    thresh = (U[:, :B] > U[:, :B].median(dim=0, keepdim=True).values).to(torch.uint8)
    n_dy = B // 2
    hybrid = torch.cat(
        [dyadic[:, -n_dy:],
         (U[:, 1:1 + B - n_dy] > U[:, 1:1 + B - n_dy].median(dim=0, keepdim=True).values).to(torch.uint8)],
        dim=1)
    candidates += [
        ("SVD threshold (privileged)", thresh),
        ("dyadic on u1 (privileged)", dyadic),
        ("hybrid dyadic+threshold (privileged)", hybrid),
    ]

    rows: list[tuple] = []
    for name, C in candidates:
        analyse(name, C, B, args.orders, Wc, Wr, total, total_res, cum, cum_res, rows)

    hdr = f"{'code assignment':38s} {'k':>2s} {'M':>5s} {'capture':>8s} {'+bias':>8s} {'residual':>9s} {'oracle':>8s} {'orc.res':>8s}"
    print(hdr)
    print("-" * len(hdr))
    last = None
    for name, k, M, cap, bias, res, orc, orcr in rows:
        label = name if name != last else ""
        last = name
        print(f"{label:38s} {k:2d} {M:5d} {cap*100:7.2f}% {bias*100:7.2f}% "
              f"{res*100:8.2f}% {orc*100:7.2f}% {orcr*100:7.2f}%")

    print("\nRead the 'residual' column. 'capture' and '+bias' are dominated by the "
          "unigram direction,\nwhich a per-token bias supplies for free, so they "
          "overstate how much a code head actually\nlearns. A residual column that "
          "does not grow with k means the monomial ladder is adding\nrank without "
          "adding reach, and no interaction order will rescue that code.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
