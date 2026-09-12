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

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.binary import binarise_model_
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb


def corrupt(g, kind, param):
    if kind == "exact":
        return g, 1.0, 0.0
    if kind == "signonly":
        g2 = torch.sign(g) * g.abs().mean()
    elif kind.startswith("lognormal"):
        s = float(kind.split("_")[1])
        g2 = g * torch.randn_like(g).mul(s).exp()
    elif kind.startswith("flip"):
        p = float(kind.split("_")[1])
        mask = (torch.rand_like(g) < p).to(g.dtype) * -2 + 1
        g2 = g * mask
    else:
        raise ValueError(kind)
    agree = (torch.sign(g2) == torch.sign(g)).to(torch.float32).mean().item()
    denom = g.pow(2).sum().item() + 1e-12
    relmse = (g2 - g).pow(2).sum().item() / denom
    return g2, agree, relmse


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


def run_arm(kind, binary, opt_name, args, device, tok, token_bytes):
    torch.manual_seed(args.seed)
    model, cfg = build(args.depth, args.vocab, args.seq, device, binary)
    params = [p for p in model.parameters() if p.requires_grad]
    if opt_name == "sgd":
        opt = torch.optim.SGD(params, lr=args.sgd_lr, momentum=0.9)
    else:
        opt = torch.optim.AdamW(params, lr=args.adam_lr, betas=(0.9, 0.95),
                                weight_decay=0.0)
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tok, args.batch, args.seq, split="train", device=device, data_dir=args.data_dir)
    agrees, mses = [], []
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
            g2, a, m = corrupt(p.grad, kind, p)
            p.grad.copy_(g2)
            if step % max(1, args.steps // 20) == 0:
                agrees.append(a)
                mses.append(m)
        opt.step()
        if step < 20:
            first_losses.append(float(loss))
        if step >= args.steps - 20:
            last_losses.append(float(loss))
    model.eval()
    val = tokenizing_distributed_data_loader_bos_bestfit(
        tok, args.batch, args.seq, split="val", device=device, data_dir=args.data_dir)
    bpb, _ = evaluate_bpb(model, val, args.eval_steps, token_bytes)
    trained = (sum(first_losses) / max(1, len(first_losses))
               - sum(last_losses) / max(1, len(last_losses)))
    del model, opt
    torch.cuda.empty_cache()
    return float(bpb), sum(agrees) / len(agrees), sum(mses) / len(mses), trained


def tune_lr(binary, opt_name, args, device, tok, token_bytes, grid):
    """Pick the LR per (arch, optimiser) on the UNCORRUPTED arm.

    Running every arm at one LR is how the first version of this experiment compared
    a trained dense model against an untrained binary one and read the binary column
    as "insensitive to corruption". A model that is not learning is insensitive to
    everything. Tune first, then corrupt.
    """
    best, best_lr = None, grid[0]
    for lr in grid:
        if opt_name == "sgd":
            args.sgd_lr = lr
        else:
            args.adam_lr = lr
        bpb, _, _, trained = run_arm("exact", binary, opt_name, args, device, tok, token_bytes)
        tag = "binary" if binary else "dense"
        print(f"    lr {lr:<9g} bpb {bpb:.4f}  loss drop {trained:+.4f}")
        if best is None or bpb < best:
            best, best_lr = bpb, lr
    # An optimum at the edge of the grid means the real optimum is probably OUTSIDE
    # it, so the arm is still undertrained and every corruption delta measured at
    # that LR is suspect. Binary needs far larger steps than dense under SGD, because
    # only sign CROSSINGS change the function and a latent weight must traverse the
    # whole clip window to produce one.
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
    ap.add_argument("--eval-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sgd-lr", type=float, default=0.05)
    ap.add_argument("--adam-lr", type=float, default=3e-3)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--optimizers", nargs="+", default=["sgd", "adamw"])
    ap.add_argument("--lr-grid", type=float, nargs="+", default=None,
                    help="LRs to tune over per (arch, optimiser) on the exact arm")
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
            grid = a.lr_grid or ([0.5, 0.15, 0.05, 0.015] if opt_name == "sgd"
                                 else [3e-3, 1e-3, 3e-4])
            print(f"  tuning lr on the uncorrupted arm over {grid}")
            lr = tune_lr(arch == "binary", opt_name, a, device, tok, token_bytes, grid)
            if opt_name == "sgd":
                a.sgd_lr = lr
            else:
                a.adam_lr = lr
            print(f"  chosen lr = {lr:g}")
            print(f"{'corruption':<16}{'agreement':>11}{'rel MSE':>11}{'bpb':>10}"
                  f"{'vs exact':>11}{'loss drop':>11}")
            base, base_trained = None, None
            for kind in a.kinds:
                bpb, agree, mse, trained = run_arm(kind, arch == "binary", opt_name, a,
                                                   device, tok, token_bytes)
                if base is None:
                    base, base_trained = bpb, trained
                results[(arch, opt_name, kind)] = (bpb, agree, mse, trained)
                print(f"{kind:<16}{agree:>11.3f}{mse:>11.3f}{bpb:>10.4f}"
                      f"{bpb-base:>+11.4f}{trained:>+11.4f}")
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
