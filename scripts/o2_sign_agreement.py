"""O2: does quality in a binary network track gradient SIGN agreement or magnitude?

Prediction P1 (plan section 3.2): a binary weight's update is a flip decision, so the
backward pass need only deliver sign(dL/dw). Final quality should then be a function
of the sign-agreement rate with exact backprop and roughly invariant to magnitude
error. If it instead tracks gradient MSE, the backward contribution collapses to
"low-bit backward with error feedback", which is published, and contribution 2 of the
plan is cut.

THE CONFOUND, and the reason optimiser is an axis here. Adam divides by the square
root of the second moment, so its update is already approximately sign-like (Balles
and Hennig, 2018). Run only under Adam, magnitude corruption would look harmless for
dense AND binary and the experiment would prove nothing about binarity. SGD is the
magnitude-sensitive control, and the result that matters is the INTERACTION: does
magnitude corruption hurt dense under SGD while leaving binary intact?

The corruptions are chosen to move sign agreement and magnitude error INDEPENDENTLY:
  exact       control
  signonly    g <- sign(g)          agreement 1.00, magnitude destroyed
  lognormal   g <- g*exp(N(0,s))    agreement 1.00, magnitude corrupted
  flip_p      flip a fraction p     agreement 1-p,  magnitude preserved

Corruption is applied to p.grad after backward, which is exactly the claim in section
3.2: what the WEIGHT update needs. It does not cheapen activation-gradient
propagation; that is a separate question and a separate experiment.

Run:  python -m scripts.o2_sign_agreement --steps 300
"""
import argparse
import math
import time

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.binary import binarise_model_
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb


def corrupt(g, kind, param):
    """Corrupt a weight gradient, and report agreement over NONZERO entries only.

    Two traps, both of which produced wrong readings before they were fixed.

    1. `sign(0) == sign(0)` counts as agreement. A binary model's STE zeroes the
       gradient of every latent weight outside the clip window, which is most of
       them, so a naive agreement metric reported 0.946 where dense reported 0.742
       for the SAME nominal corruption: the binary arm received about 5x less sign
       corruption and then looked robust to it. Agreement is therefore measured over
       the support of g.
    2. Parameterising by flip PROBABILITY delivers different corruption strengths to
       different architectures for the same reason. `flipa_<a>` targets an agreement
       level instead, so the arms are matched on the quantity the hypothesis is about.
    """
    nz = g != 0
    n_nz = int(nz.sum())
    if kind == "exact":
        return g, 1.0, 0.0, n_nz / max(g.numel(), 1)
    if kind == "signonly":
        # Also norm-preserved, for the same reason.
        g2 = torch.sign(g)
        g2 = g2 * (g.norm() / g2.norm().clamp(min=1e-12))
    elif kind.startswith("lognormal"):
        # Norm-preserving. Raw multiplicative lognormal noise has mean exp(s^2/2) and a
        # heavy tail, so it inflates the update and the arm diverges: the dense cell hit
        # +1.7464 bpb with a loss drop of -3.1014, which measures instability and not
        # magnitude sensitivity. Rescaling to the original norm isolates the precision
        # of the magnitudes from the size of the step.
        sd = float(kind.split("_")[1])
        g2 = g * torch.randn_like(g).mul(sd).exp()
        n0, n1 = g.norm(), g2.norm().clamp(min=1e-12)
        g2 = g2 * (n0 / n1)
    elif kind.startswith("flipa"):
        # flip (1 - target agreement) of the NONZERO entries
        target = float(kind.split("_")[1])
        p = max(0.0, 1.0 - target)
        sel = (torch.rand_like(g) < p) & nz
        g2 = torch.where(sel, -g, g)
    elif kind.startswith("flip"):
        p = float(kind.split("_")[1])
        sel = (torch.rand_like(g) < p) & nz
        g2 = torch.where(sel, -g, g)
    else:
        raise ValueError(kind)
    if n_nz:
        agree = (torch.sign(g2[nz]) == torch.sign(g[nz])).to(torch.float32).mean().item()
    else:
        agree = 1.0
    denom = g.pow(2).sum().item() + 1e-12
    relmse = (g2 - g).pow(2).sum().item() / denom
    return g2, agree, relmse, n_nz / max(g.numel(), 1)


def build(depth, vocab, seq, device, binary):
    n_embd = 128 * max(1, depth * 64 // 128)
    cfg = GPTConfig(sequence_len=seq, vocab_size=vocab, n_layer=depth,
                    n_head=n_embd // 128, n_kv_head=n_embd // 128, n_embd=n_embd,
                    window_pattern="L")
    with torch.device("meta"):
        m = GPT(cfg)
    m.to_empty(device=device)
    m.init_weights()
    if binary:
        binarise_model_(m, binarise_acts=True)
    return m, cfg


_LOADERS = {}


def get_loader(split, args, device, tok):
    """One dataloader per split, reused across every arm.

    Constructing a tokenizing loader scans the shard directory and spins up tokenizer
    threads. Doing that once per run, ~46 times, dominated the wall clock: a d8 sweep
    that was costed at 10 minutes of arithmetic took 90.
    """
    key = (split, args.batch, args.seq)
    if key not in _LOADERS:
        _LOADERS[key] = tokenizing_distributed_data_loader_bos_bestfit(
            tok, args.batch, args.seq, split=split, device=device,
            data_dir=args.data_dir, max_shards=args.max_shards)
    return _LOADERS[key]


def run_arm(kind, binary, opt_name, args, device, tok, token_bytes):
    t_run = time.time()
    torch.manual_seed(args.seed)
    model, cfg = build(args.depth, args.vocab, args.seq, device, binary)
    params = [p for p in model.parameters() if p.requires_grad]
    if opt_name == "sgd":
        opt = torch.optim.SGD(params, lr=args.sgd_lr, momentum=0.9)
    else:
        opt = torch.optim.AdamW(params, lr=args.adam_lr, betas=(0.9, 0.95),
                                weight_decay=0.0)
    loader = get_loader("train", args, device, tok)
    agrees, mses, dens = [], [], []
    from nanochat.binary import BinaryLinear, BinaryEmbedding
    bmods = [m for m in model.modules() if isinstance(m, (BinaryLinear, BinaryEmbedding))]
    first_losses, last_losses = [], []
    model.train()
    for step in range(args.steps):
        x, y = next(loader)
        out = model(x, targets=y)
        loss = out[0] if isinstance(out, tuple) else out
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for p in params:
            if p.grad is None:
                continue
            g2, a, m, dn = corrupt(p.grad, kind, p)
            p.grad.copy_(g2)
            if step % max(1, args.steps // 20) == 0:
                agrees.append(a)
                mses.append(m)
                dens.append(dn)
        opt.step()
        if step < 20:
            first_losses.append(float(loss))
        if step >= args.steps - 20:
            last_losses.append(float(loss))
    model.eval()
    val = get_loader("val", args, device, tok)
    # Dead bits: latent weights outside the STE clip window receive no gradient and can
    # never flip again. This exists because the run contains a contradiction: for binary,
    # `signonly` cost +0.4940 and stopped learning while `lognormal` cost -0.0252 and did
    # not, yet BOTH are magnitude corruptions. Hypothesis: signonly hands every nonzero
    # gradient the same magnitude, which shoves latent weights near the boundary out of
    # the window, so it measures dead-bit creation rather than loss of magnitude
    # information. If dead% spikes under signonly and not under lognormal, that is it.
    dead = float("nan")
    if bmods:
        d = sum(int((m.weight.detach().abs() > m.clip).sum()) for m in bmods)
        t = sum(m.weight.numel() for m in bmods)
        dead = 100.0 * d / max(t, 1)
    bpb, _ = evaluate_bpb(model, val, args.eval_steps, token_bytes)
    trained = (sum(first_losses) / max(1, len(first_losses))
               - sum(last_losses) / max(1, len(last_losses)))
    del model, opt
    torch.cuda.empty_cache()
    run_arm.last_seconds = time.time() - t_run
    return (float(bpb), sum(agrees) / len(agrees), sum(mses) / len(mses), trained,
            sum(dens) / max(1, len(dens)), dead)


def tune_lr(binary, opt_name, args, device, tok, token_bytes, grid):
    """Pick the LR per (arch, optimiser) on the UNCORRUPTED arm.

    Running every arm at one LR is how the first version of this experiment compared
    a trained dense model against an untrained binary one and read the binary column
    as "insensitive to corruption". A model that is not learning is insensitive to
    everything. Tune first, then corrupt.
    """
    best, best_lr = None, grid[0]
    full_steps = args.steps
    args.steps = args.tune_steps or max(50, full_steps // 3)
    for lr in grid:
        if opt_name == "sgd":
            args.sgd_lr = lr
        else:
            args.adam_lr = lr
        bpb, _, _, trained, dn, _dead = run_arm("exact", binary, opt_name, args, device,
                                                tok, token_bytes)
        tag = "binary" if binary else "dense"
        print(f"    lr {lr:<9g} bpb {bpb:.4f}  loss drop {trained:+.4f}  "
              f"nonzero-grad {100*dn:.1f}%  [{run_arm.last_seconds:.0f}s]", flush=True)
        if lr == grid[0]:
            n_runs = len(grid) + 6
            print(f"    ({run_arm.last_seconds:.0f}s per run x ~{n_runs} runs in this cell "
                  f"= ~{run_arm.last_seconds*n_runs/60:.0f} min; x4 cells)", flush=True)
        if best is None or bpb < best:
            best, best_lr = bpb, lr
    # An optimum at the edge of the grid means the real optimum is probably OUTSIDE
    # it, so the arm is still undertrained and every corruption delta measured at
    # that LR is suspect. Binary needs far larger steps than dense under SGD, because
    # only sign CROSSINGS change the function and a latent weight must traverse the
    # whole clip window to produce one.
    args.steps = full_steps
    if best_lr in (max(grid), min(grid)):
        edge = "MAX" if best_lr == max(grid) else "MIN"
        print(f"    WARNING: best lr {best_lr:g} is the {edge} of the grid. The optimum is")
        print(f"             likely outside it; extend --lr-grid before trusting this cell.")
    return best_lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--tune-steps", type=int, default=0,
                    help="steps for the LR search (0 = steps//3). LR tuning is ~22 of the "
                         "~46 runs in a full sweep and does not need the full budget to "
                         "rank learning rates.")
    ap.add_argument("--eval-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sgd-lr", type=float, default=0.05)
    ap.add_argument("--adam-lr", type=float, default=3e-3)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max-shards", type=int, default=8,
                    help="scanning 300 shards per loader construction is pure overhead "
                         "for a 300-step probe")
    ap.add_argument("--optimizers", nargs="+", default=["sgd", "adamw"])
    ap.add_argument("--sgd-lr-grid", type=float, nargs="+",
                    default=[10, 3, 1, 0.3, 0.1, 0.03],
                    help="SGD LRs to tune over. Binary needs ~10x dense: only sign "
                         "CROSSINGS change the function, so a latent weight must cross "
                         "the whole clip window before anything moves.")
    ap.add_argument("--adam-lr-grid", type=float, nargs="+",
                    default=[1e-2, 3e-3, 1e-3, 3e-4, 1e-4],
                    help="Adam LRs. A SINGLE shared --lr-grid is a bug: tuning Adam over "
                         "an SGD-scale grid (0.1 to 30) makes every Adam cell garbage, "
                         "which is exactly what happened on the first A100 run.")
    ap.add_argument("--min-loss-drop", type=float, default=0.10,
                    help="the exact arm must learn at least this much or the whole "
                         "(arch, optimiser) cell is reported as INCONCLUSIVE")
    ap.add_argument("--kinds", nargs="+",
                    default=["exact", "signonly", "lognormal_1.0", "flip_0.05",
                             "flip_0.15", "flip_0.30"])
    ap.add_argument("--arch", nargs="+", default=["dense", "binary"])
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = get_tokenizer("tokenizer")
    token_bytes = get_token_bytes(device=device, tokenizer_dir="tokenizer")

    print(f"O2: depth={a.depth} V={a.vocab} seq={a.seq} batch={a.batch} steps={a.steps}")
    print("P1: bpb should track AGREEMENT and not MSE, and that should be true for")
    print("    BINARY under SGD specifically. Adam is already sign-like, so an effect")
    print("    visible under Adam alone says nothing about binarity.")
    print()
    results = {}
    for opt_name in a.optimizers:
        for arch in a.arch:
            print(f"=== {arch.upper()} / {opt_name.upper()} ===")
            grid = a.sgd_lr_grid if opt_name == "sgd" else a.adam_lr_grid
            print(f"  tuning lr on the uncorrupted arm over {grid}")
            lr = tune_lr(arch == "binary", opt_name, a, device, tok, token_bytes, grid)
            if opt_name == "sgd":
                a.sgd_lr = lr
            else:
                a.adam_lr = lr
            print(f"  chosen lr = {lr:g}")
            print(f"{'corruption':<16}{'agreement':>11}{'rel MSE':>11}{'bpb':>10}"
                  f"{'vs exact':>11}{'loss drop':>11}{'nonzero':>10}{'dead':>9}")
            base, base_trained = None, None
            for kind in a.kinds:
                bpb, agree, mse, trained, dn, dead = run_arm(kind, arch == "binary",
                                                             opt_name, a, device, tok,
                                                             token_bytes)
                if base is None:
                    base, base_trained = bpb, trained
                results[(arch, opt_name, kind)] = (bpb, agree, mse, trained)
                print(f"{kind:<16}{agree:>11.3f}{mse:>11.3f}{bpb:>10.4f}"
                      f"{bpb-base:>+11.4f}{trained:>+11.4f}{100*dn:>9.1f}%"
                      + ("      -" if dead != dead else f"{dead:>8.2f}%"))
            if base_trained is not None and base_trained < a.min_loss_drop:
                print(f"  INCONCLUSIVE: the uncorrupted arm learned only {base_trained:+.4f}")
                print(f"  (< --min-loss-drop {a.min_loss_drop}). A model that is not learning is")
                print("  insensitive to every corruption, so these deltas mean nothing.")
                results[("SKIP", arch, opt_name)] = True
            print()

    print("=" * 72)
    print("VERDICT INPUTS")
    for opt_name in a.optimizers:
        for arch in a.arch:
            ks = [k for (ar, o, k) in results if ar == arch and o == opt_name]
            if not ks:
                continue
            if ("SKIP", arch, opt_name) in results:
                print(f"  {arch:<7}/{opt_name:<6} INCONCLUSIVE (uncorrupted arm did not train)")
                continue
            ex = results[(arch, opt_name, "exact")][0]
            mag = [k for k in ks if k in ("signonly",) or k.startswith("lognormal")]
            flip = [k for k in ks if k.startswith("flip")]
            dm = max((results[(arch, opt_name, k)][0] - ex) for k in mag) if mag else float("nan")
            df = max((results[(arch, opt_name, k)][0] - ex) for k in flip) if flip else float("nan")
            print(f"  {arch:<7}/{opt_name:<6} worst magnitude-only damage {dm:+.4f}   "
                  f"worst sign-flip damage {df:+.4f}")
    print()
    print("P1 SURVIVES if, under SGD, magnitude-only damage is small for BINARY while")
    print("sign-flip damage is large, AND magnitude-only damage is larger for DENSE.")
    print("P1 FAILS if magnitude-only damage is comparable for both, or if binary is")
    print("hurt by magnitude corruption as much as by sign flips.")


if __name__ == "__main__":
    main()
