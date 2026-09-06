"""Measure whether a cheap proposal can replace a dense output head's normalisation.

Every structured head in this project approximates the head matrix and normalises
exactly. The alternative is to keep the head EXACTLY dense and approximate the
normalisation instead:

    z_prop = U_c h              rank-c proposal over all V words        V*c MACs
    K      = topK(z_prop)       plus the target during training
    z_K    = U[K] h             exact logits from the full-rank head    K*d MACs
    tail   = importance sample S from q ∝ exp(z_prop) off K             S*d MACs
    log Z  = logaddexp(lse(z_K), tail)

No capacity is lost: U is the full V x d matrix and only its evaluation is sparse.
The one approximation is log Z, and because the loss is z_target - log Z, an error of
e nats per token is a straight e/(ln2 * bytes_per_token) bits per byte.

Two estimators are reported because the difference decides the idea. The plug-in
("use the cheap logits for the tail") is BIASED -- measured at -0.027 nats, enough to
eat the whole saving. Sampling the tail from the proposal instead is unbiased, and its
variance is small precisely where the proposal is good, which the top-K step has
already arranged.

  python -m scripts.proposal_probe acts_d8_v131k.pt --head head_d8_v131k.pt \\
      --bytes-per-token 5.02 --slope 0.132 --dense-model-macs 98145792
"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.head_oracle import load_head

SOFTCAP = 20.0        # base_train squashes logits before the loss; match it


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("acts", help="(N, d) activations, or a dict with 'acts' and 'lm_head'")
    ap.add_argument("--head", default=None, help="dense lm_head if not inside --acts")
    ap.add_argument("--ranks", default="8,16,32", help="proposal ranks to try")
    ap.add_argument("--topk", default="1024,4096", help="K values")
    ap.add_argument("--samples", default="256,1024", help="S values for the tail")
    ap.add_argument("--bytes-per-token", type=float, default=0.0,
                    help="val_loss / (ln2 * val_bpb) from the run's own log; enables bpb")
    ap.add_argument("--slope", type=float, default=0.0,
                    help="dense bpb/decade at this vocabulary; enables the margin column")
    ap.add_argument("--dense-model-macs", type=float, default=0.0,
                    help="whole-model MACs/token of the dense arm; enables the margin")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    obj = torch.load(args.acts, weights_only=True, map_location="cpu")
    if isinstance(obj, dict):
        H, U = obj["acts"].float(), obj["lm_head"].float()
    else:
        H = obj.float()
        assert args.head, "--head is required when the acts file holds only activations"
        U = load_head(args.head)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    H, U = H.to(dev), U.to(dev)
    N, d = H.shape
    V = U.shape[0]
    torch.manual_seed(args.seed)
    print(f"{N:,} activations, head {V:,} x {d}")

    ints = lambda s: [int(t) for t in s.replace(" ", ",").split(",") if t]
    ranks, Ks, Ss = ints(args.ranks), ints(args.topk), ints(args.samples)
    bpb = (1.0 / (math.log(2) * args.bytes_per_token)) if args.bytes_per_token else None
    margin_ok = bool(bpb and args.slope and args.dense_model_macs)
    body = args.dense_model_macs - V * d if margin_ok else 0.0

    with torch.no_grad():
        Z = SOFTCAP * torch.tanh((H @ U.T) / SOFTCAP)
        logZ = torch.logsumexp(Z, dim=-1)
        Vh = torch.linalg.svd(U, full_matrices=False).Vh
        hdr = f"  {'rank':>5} {'K':>6} {'S':>5} {'top-1':>7} {'massK':>7} " \
              f"{'plug-in':>9} {'sampled':>9}"
        if bpb:
            hdr += f" {'bpb cost':>9}"
        if margin_ok:
            hdr += f" {'head':>8} {'vs dense':>9} {'margin':>8}"
        print(hdr)
        for c in ranks:
            Uc = (U @ Vh[:c].T) @ Vh[:c]
            Zp = SOFTCAP * torch.tanh((H @ Uc.T) / SOFTCAP)
            for K in Ks:
                idx = Zp.topk(K, dim=-1).indices
                hi = torch.logsumexp(Z.gather(-1, idx), dim=-1)
                kept = (idx == Z.argmax(-1, keepdim=True)).any(-1).float().mean()
                mass = Z.softmax(-1).gather(-1, idx).sum(-1).mean()
                off = Zp.masked_fill(
                    torch.zeros_like(Zp, dtype=torch.bool).scatter_(-1, idx, True), -1e30)
                logC = torch.logsumexp(off, dim=-1)
                plug = (torch.logaddexp(hi, logC) - logZ).abs().mean()
                for S in Ss:
                    q = (off - logC.unsqueeze(-1)).exp()
                    s_idx = torch.multinomial(q, S, replacement=True)
                    ratio = (Z.gather(-1, s_idx) - Zp.gather(-1, s_idx)).exp()
                    tail = logC + ratio.mean(-1).clamp_min(1e-30).log()
                    err = float((torch.logaddexp(hi, tail) - logZ).abs().mean())
                    row = (f"  {c:>5} {K:>6} {S:>5} {kept:>6.1%} {mass:>7.4f} "
                           f"{plug:>9.5f} {err:>9.5f}")
                    if bpb:
                        row += f" {err*bpb:>9.5f}"
                    if margin_ok:
                        head = V*c + (K+S)*d
                        be = args.slope * -math.log10((body+head)/args.dense_model_macs)
                        row += (f" {head/1e6:7.2f}M {V*d/head:8.1f}x "
                                f"{be - err*bpb:+8.4f}")
                    print(row)
    if margin_ok:
        print("\n  margin = what dense gives up for the same FLOPs cut, minus what the")
        print("  approximate normalisation costs. Positive means the head is above the")
        print("  dense curve. Every logit in the top-K is EXACT: the head is not")
        print("  approximated at all, only its normalisation is.")


if __name__ == "__main__":
    main()
