#!/usr/bin/env python3
"""How much conditioning capacity exists at each layer, before training anything.

Every design in this repo asks how to make W depend on x. This asks the prior
question: is there anything at this layer for conditioning to exploit?

There is an exact answer. The per-token weight gradient of a linear layer is
rank one, g_t = a_t x_t^T with a_t = dL/dy_t. The mean gbar over tokens is what a
dense layer learns. The DISPERSION of g_t around gbar is exactly the capacity
available to any input-conditioned scheme, of any design: if tokens do not
disagree about which way the weight should move, no conditional layer can beat
dense at that layer, and no amount of K or R changes that.

The dispersion lives in d^2 dimensions, but it never has to be formed. The N x N
Gram matrix of the per-token gradients factors:

    <g_s, g_t>_F = tr(x_s a_s^T a_t x_t^T) = (a_s . a_t)(x_s . x_t)

so  G = (A A^T) * (X X^T),  a Hadamard product of two N x N inner-product
matrices. (Standard NTK algebra; what is new here is using it as a per-layer
conditioning-capacity diagnostic rather than as a kernel.) Center it and take
the eigenvalues. Three numbers come out per layer:

    headroom     lambda_disp / (lambda_disp + N*|gbar|^2)
                 the fraction of gradient signal a single static operator
                 cannot capture. 0 means every token wants the same update and
                 conditioning is provably useless here. Near 1 means the tokens
                 disagree almost entirely.

    dof          participation ratio (sum L)^2 / sum L^2 of the dispersion
                 spectrum. The number of independent directions the tokens
                 disagree along, i.e. the largest K or R that can pay for
                 itself at this layer. Compare against K-1 for a template bank
                 and R for ConditionedLinear.

    top1         largest dispersion eigenvalue over the total. If this is near
                 1 the disagreement is one-dimensional, which is the regime
                 where a rank-1 conditioned delta is enough and a bank is waste.

The estimator is Wishart-biased: from N samples of a rank-D object it reports
roughly D/(1 + D/N) rather than D. --tokens sets N; the printed dof is a lower
bound and the bias shrinks as N grows.

Usage:
    python -m scripts.conditioning_headroom --checkpoint out/sweep_p33/A1_.../
    python -m scripts.conditioning_headroom --depth 4 --steps 0   # random init
    python -m scripts.conditioning_headroom --checkpoint DIR --tokens 8192 --json out/headroom.json
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="run directory to load (uses nanochat.checkpoint_manager). "
                        "Omit to analyze a freshly initialized model.")
    p.add_argument("--step", type=int, default=-1, help="checkpoint step (-1 = latest)")
    p.add_argument("--depth", type=int, default=4, help="depth when no checkpoint is given")
    p.add_argument("--aspect-ratio", type=int, default=64)
    p.add_argument("--tokens", type=int, default=4096,
                   help="N, the number of token samples. Cost is O(N^2 d) and O(N^3); "
                        "the dof estimate is biased low by 1/(1 + dof/N)")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--tokenizer-dir", type=str, default="tokenizer")
    p.add_argument("--max-shards", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--json", type=str, default=None, help="write results here")
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="optimizer steps to take before measuring. A freshly initialized "
                        "model has zero-init output projections, so no gradient reaches "
                        "q/k/v/c_fc at step 0 and their headroom is undefined rather than "
                        "zero. Irrelevant when --checkpoint is given.")
    p.add_argument("--random-data", action="store_true",
                   help="use random token ids instead of the corpus (smoke test only: "
                        "headroom on random data is meaningless)")
    return p.parse_args()


# ── the measurement ──────────────────────────────────────────────────────────

def headroom_from_grams(X, A, eps=1e-12):
    """Spectrum of the per-token weight-gradient dispersion, from activations alone.

    X: (N, d_in)  layer inputs
    A: (N, d_out) gradients of the loss wrt the layer outputs

    Returns (headroom, dof, top1, n_eff). Never forms the (d_out, d_in) gradient
    of any single token, let alone their covariance.
    """
    X = X.double()
    A = A.double()
    N = X.shape[0]
    # G[s,t] = <g_s, g_t>_F for the rank-1 per-token gradients g_t = a_t x_t^T
    G = (A @ A.T) * (X @ X.T)
    # Both outputs are scale-invariant ratios, so normalize by the mean squared
    # gradient norm first. Gram entries are products of two Gram entries, so raw
    # magnitudes can span many orders of magnitude across layers and the squares
    # in the participation ratio are the first thing to overflow.
    scale = G.diagonal().mean().clamp(min=eps)
    G = G / scale
    # Center: subtract the mean gradient, which is what a dense layer learns.
    # H G H with H = I - 11^T/N does this without forming gbar in weight space.
    row = G.mean(dim=1, keepdim=True)
    tot = G.mean()
    Gc = G - row - row.T + tot
    # |gbar|^2 * N is the trace the mean direction accounts for
    mean_energy = tot * N
    evals = torch.linalg.eigvalsh(Gc).clamp(min=0.0)
    disp = evals.sum()
    total = disp + mean_energy.clamp(min=0)
    if total <= eps:
        # No gradient reaches this layer at all, which is NOT the same finding as
        # "the tokens agree". At init the dense recipe zero-inits attn.c_proj and
        # mlp.c_proj, so nothing flows back to q/k/v/c_fc on step 0. Reported
        # separately so it cannot be misread as "conditioning cannot help".
        return float('nan'), float('nan'), float('nan'), N
    if disp <= eps:
        return 0.0, 0.0, 0.0, N
    headroom = (disp / (disp + mean_energy.clamp(min=0) + eps)).item()
    dof = ((disp ** 2) / (evals.square().sum() + eps)).item()
    top1 = (evals.max() / disp).item()
    return headroom, dof, top1, N


@torch.no_grad()
def _subsample(t, n):
    flat = t.reshape(-1, t.shape[-1])
    if flat.shape[0] <= n:
        return flat
    stride = flat.shape[0] // n
    return flat[::stride][:n]


def collect(model, batches, target_modules, n_tokens, device):
    """One forward and backward per batch, capturing (input, grad_output) per layer."""
    store = {name: {'x': [], 'a': []} for name in target_modules}
    handles = []

    def mk_hook(name):
        def fwd_hook(mod, inp, out):
            store[name]['x'].append(inp[0].detach())

        def bwd_hook(mod, grad_in, grad_out):
            store[name]['a'].append(grad_out[0].detach())
        return fwd_hook, bwd_hook

    for name, mod in target_modules.items():
        f, b = mk_hook(name)
        handles.append(mod.register_forward_hook(f))
        handles.append(mod.register_full_backward_hook(b))

    per_batch = max(1, n_tokens // max(1, len(batches)))
    kept = {name: {'x': [], 'a': []} for name in target_modules}
    for ids in batches:
        model.zero_grad(set_to_none=True)
        loss = model(ids, ids)
        loss.backward()
        for name in target_modules:
            if not store[name]['x'] or not store[name]['a']:
                continue
            x = store[name]['x'][-1]
            a = store[name]['a'][-1]
            kept[name]['x'].append(_subsample(x, per_batch).float().cpu())
            kept[name]['a'].append(_subsample(a, per_batch).float().cpu())
            store[name]['x'].clear()
            store[name]['a'].clear()

    for h in handles:
        h.remove()
    out = {}
    for name in target_modules:
        if kept[name]['x']:
            out[name] = (torch.cat(kept[name]['x'])[:n_tokens],
                         torch.cat(kept[name]['a'])[:n_tokens])
    return out


def main():
    args = parse_args()
    device = args.device
    torch.manual_seed(0)

    from nanochat.gpt import GPT, GPTConfig

    # ── model ────────────────────────────────────────────────────────────────
    if args.checkpoint:
        from nanochat.checkpoint_manager import load_model_from_dir
        model, _tok, _meta = load_model_from_dir(args.checkpoint, device, phase="eval",
                                                 step=None if args.step < 0 else args.step)
        model = model.module if hasattr(model, 'module') else model
        cfg = model.config
        print(f"loaded {args.checkpoint} (n_layer={cfg.n_layer}, n_embd={cfg.n_embd})")
    else:
        dim = ((args.depth * args.aspect_ratio + 127) // 128) * 128
        cfg = GPTConfig(sequence_len=args.seq_len, vocab_size=65536, n_layer=args.depth,
                        n_head=max(1, dim // 128), n_kv_head=max(1, dim // 128), n_embd=dim)
        with torch.device('meta'):
            model = GPT(cfg)
        model.to_empty(device=device)
        model.init_weights()
        print(f"freshly initialized dense model (n_layer={cfg.n_layer}, n_embd={cfg.n_embd})")
        print("NOTE: headroom at random init is not the headroom of a trained model. "
              "Point --checkpoint at a real run for a number you can act on.")
    model.train()

    # ── data ─────────────────────────────────────────────────────────────────
    n_batches = max(1, args.tokens // (args.batch * args.seq_len))
    if args.random_data:
        print("WARNING: --random-data, headroom on random ids measures nothing real")
        batches = [torch.randint(0, cfg.vocab_size, (args.batch, args.seq_len), device=device)
                   for _ in range(n_batches)]
    else:
        from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
        from nanochat.tokenizer import get_tokenizer
        tok = get_tokenizer(args.tokenizer_dir) if os.path.isdir(args.tokenizer_dir) else None
        loader = tokenizing_distributed_data_loader_bos_bestfit(
            tok, args.batch, args.seq_len, split="val", device=device,
            data_dir=args.data_dir, max_shards=args.max_shards)
        batches = [next(loader)[0] for _ in range(n_batches)]

    # ── which layers ─────────────────────────────────────────────────────────
    # Any module with a single (in -> out) linear action is measurable. We take
    # every attention/MLP projection, whatever class currently implements it.
    targets = {}
    for name, mod in model.named_modules():
        leaf = name.rsplit('.', 1)[-1]
        if leaf in ('c_q', 'c_k', 'c_v', 'c_proj', 'c_fc') and 'transformer.h.' in name:
            targets[name] = mod
    if not targets:
        print("no projections found")
        return
    print(f"measuring {len(targets)} projections over {args.tokens} tokens "
          f"({n_batches} batches of {args.batch}x{args.seq_len})")

    if args.warmup_steps > 0:
        opt = model.setup_optimizer(unembedding_lr=4e-3, embedding_lr=0.2, matrix_lr=0.02)
        for i in range(args.warmup_steps):
            opt.zero_grad()
            model(batches[i % len(batches)], batches[i % len(batches)]).backward()
            opt.step()
        print(f"took {args.warmup_steps} warmup steps so gradients reach every projection")

    data = collect(model, batches, targets, args.tokens, device)

    # ── report ───────────────────────────────────────────────────────────────
    rows = []
    for name, (X, A) in data.items():
        h, dof, top1, n = headroom_from_grams(X.to(device), A.to(device))
        parts = name.split('.')
        layer = int(parts[parts.index('h') + 1])
        proj = '.'.join(parts[parts.index('h') + 2:])
        rows.append(dict(layer=layer, proj=proj, headroom=h, dof=dof, top1=top1, n=n))

    rows.sort(key=lambda r: (r['layer'], r['proj']))
    print("\n" + "=" * 78)
    print("PER LAYER  (mean over projections)")
    print("=" * 78)
    print(f"  {'layer':>5s}  {'headroom':>9s}  {'dof':>8s}  {'top1':>6s}   verdict")
    by_layer = {}
    for r in rows:
        by_layer.setdefault(r['layer'], []).append(r)
    for layer in sorted(by_layer):
        rs = by_layer[layer]
        ok = [x for x in rs if x['headroom'] == x['headroom']]
        if not ok:
            print(f"  {layer:>5d}  {'n/a':>9s}  {'n/a':>8s}  {'n/a':>6s}   "
                  f"no gradient reaches this layer (see --warmup-steps)")
            continue
        h = sum(x['headroom'] for x in ok) / len(ok)
        d = sum(x['dof'] for x in ok) / len(ok)
        t = sum(x['top1'] for x in ok) / len(ok)
        if h != h:  # NaN
            verdict = "no gradient reaches this layer (see --warmup-steps)"
        elif d > 0.5 * args.tokens:
            verdict = f"dof unresolvable at N={args.tokens}, raise --tokens"
        elif h < 0.05:
            verdict = "no headroom: conditioning cannot help here"
        elif t > 0.6:
            verdict = "one-dimensional: a rank-1 delta is enough, a bank is waste"
        elif d < 4:
            verdict = f"narrow: K or R above ~{max(2, int(d))} buys nothing"
        else:
            verdict = f"real headroom for up to ~{int(d)} conditioning directions"
        print(f"  {layer:>5d}  {h:>9.4f}  {d:>8.1f}  {t:>6.3f}   {verdict}")

    print("\n" + "=" * 78)
    print("PER PROJECTION  (mean over layers)")
    print("=" * 78)
    print(f"  {'projection':18s}  {'headroom':>9s}  {'dof':>8s}  {'top1':>6s}")
    by_proj = {}
    for r in rows:
        by_proj.setdefault(r['proj'], []).append(r)
    for proj in sorted(by_proj):
        rs = by_proj[proj]
        ok = [x for x in rs if x['headroom'] == x['headroom']]
        if not ok:
            print(f"  {proj:18s}  {'n/a':>9s}  {'n/a':>8s}  {'n/a':>6s}")
            continue
        print(f"  {proj:18s}  {sum(x['headroom'] for x in ok) / len(ok):>9.4f}"
              f"  {sum(x['dof'] for x in ok) / len(ok):>8.1f}"
              f"  {sum(x['top1'] for x in ok) / len(ok):>6.3f}")

    ok_rows = [r for r in rows if r['headroom'] == r['headroom']]
    all_h = sum(r['headroom'] for r in ok_rows) / max(1, len(ok_rows))
    all_d = sum(r['dof'] for r in ok_rows) / max(1, len(ok_rows))
    if len(ok_rows) < len(rows):
        print(f"\n  {len(rows) - len(ok_rows)} of {len(rows)} projections received no "
              f"gradient and are excluded; pass --warmup-steps 20")
    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("  headroom  fraction of the per-token weight-gradient signal that a single")
    print("            static operator cannot represent. This is the ceiling on what")
    print("            ANY input-conditioned design can win at that layer, before")
    print("            architecture, routing or optimization enter the picture.")
    print("  dof       independent directions the tokens disagree along. Compare to")
    print("            K-1 for a template bank and to R for ConditionedLinear. Biased")
    print(f"            low by 1/(1 + dof/N); at N={args.tokens} a true value of 64 reads as")
    print(f"            {64 / (1 + 64 / args.tokens):.0f}.")
    print("  top1      share of the dispersion in its leading direction. Near 1 means")
    print("            the disagreement is essentially rank-1.")
    print(f"\n  model-wide: headroom {all_h:.4f}, dof {all_d:.1f}")
    if all_h < 0.05:
        print("  => the tokens broadly agree on the weight update. No conditional linear")
        print("     layer of any design can beat dense by much here, and the honest move")
        print("     is to spend the parameters on something else.")
    else:
        print(f"  => there is real headroom. Size K or R against the per-layer dof column")
        print("     rather than uniformly, and spend nothing where headroom is ~0.")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump({'rows': rows, 'tokens': args.tokens,
                       'checkpoint': args.checkpoint}, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
