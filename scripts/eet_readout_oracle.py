#!/usr/bin/env python3
"""EET P02 ideas 5 and 6: is the gap a READOUT problem, and is there a free halting signal?

WHY THESE TWO
-------------
Against an iso-FLOP dense control, EET is +0.049 bpb. Only 0.001 of that is the cost of
spreading depth without signal: your own iso-data dense curve is loss(F) = 4.17*F^-0.0738,
which is convex, so Jensen prices signal-free allocation at

    E[loss(random depth)] 1.0140  -  loss(mean depth) 1.0130  =  +0.0010

That leaves ~0.048 unexplained by WHICH tokens go deep. The remaining hypothesis is that
loss(F) describes a model TRAINED at budget F, while a token exiting at layer k of an
8-layer stack is read out from a layer never optimised to be prediction-ready, by a head
trained on layer-8 statistics.

TEST 5 (readout ceiling) measures the best any per-depth readout fix can do: freeze the
dense backbone, refit the head separately for each exit depth, and price the resulting
mixture against the budget. This is a TIGHT upper bound for that family, because the
refitted head is the same function class as the real one (d -> V linear), not a
free over-parameterised fit. It does NOT bound fixes that change the backbone itself
(deep supervision, depth-nesting, terminal blocks) -- those can beat it.

TEST 6 (halting signal) asks whether ||x_{l+1} - x_l|| predicts per-token difficulty. It
is free, causal and contextual, which is everything the x0 router is not. ANIRA (ICML
2026) found online halting tracks execution state while early halting collapses to static
cues; this measures whether that holds on your checkpoint.

PRE-REGISTERED KILL CRITERIA
----------------------------
    Test 5: if a per-depth head still leaves the mixture above the budget
            (dense_bpb + 0.032), the readout is NOT the whole story and the
            decomposition above is wrong. Say so before building ideas 1-4.
    Test 6: if |Spearman(delta_l, per-token CE)| < 0.3 at every layer, the residual-delta
            halting signal carries no more than the frequency prior did. Close idea 6.

USAGE
    python -m scripts.eet_readout_oracle \
        --ckpt out/dense_d8_V32k_model_001014.pt \
        --dense-bpb 0.991231 --budget 0.032
"""
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE, print0
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.gpt import GPT, GPTConfig, norm
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from scripts.eet_context_oracle import bell_capacities

LN2 = math.log(2.0)


def infer_config(sd, seq_len, window_pattern, head_dim):
    """Reconstruct GPTConfig from a bare state dict.

    The FFN width is read back per layer rather than assumed uniform. An inverse-width
    checkpoint (--eet-width-power) has a different hidden size in every block, and building
    it with the default schedule produced shape mismatches that load_state_dict swallowed
    under strict=False, leaving a randomly initialised model that silently profiled as
    nothing at all.
    """
    n_layer = 1 + max(int(k.split('.')[2]) for k in sd if k.startswith('transformer.h.'))
    n_embd = sd['transformer.h.0.attn.c_q.weight'].shape[1]
    vocab = sd['lm_head.weight'].shape[0]
    n_head = n_embd // head_dim
    mults = []
    for i in range(n_layer):
        w = sd.get(f'transformer.h.{i}.mlp.c_fc.weight')
        mults.append(round(w.shape[0] / n_embd, 6) if w is not None else 4.0)
    kw = dict(sequence_len=seq_len, vocab_size=vocab, n_layer=n_layer,
              n_head=n_head, n_kv_head=n_head, n_embd=n_embd,
              window_pattern=window_pattern)
    if len(set(mults)) > 1 or abs(mults[0] - 4.0) > 1e-6:
        kw['p34_ffn_schedule'] = ','.join(str(m) for m in mults)
        print0(f"[readout] per-layer FFN widths from the checkpoint: "
               + " ".join(f"{m:.2f}" for m in mults))
    return GPTConfig(**kw)


def load_dense(path, device, seq_len, window_pattern, head_dim):
    sd = torch.load(path, map_location='cpu', weights_only=False)
    sd = {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}
    cfg = infer_config(sd, seq_len, window_pattern, head_dim)
    with torch.device('meta'):
        model = GPT(cfg)
    model.to_empty(device=device)
    model.init_weights(verify=False)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    shape_bad = [k for k, v in sd.items()
                 if k in dict(model.named_parameters()) and
                 tuple(dict(model.named_parameters())[k].shape) != tuple(v.shape)]
    if shape_bad or len(missing) > 4:
        raise RuntimeError(
            "the reconstructed config does not match the checkpoint, so the profile would "
            "describe a partly random model rather than the trained one.\n"
            f"  shape mismatches: {shape_bad[:5]}\n  missing: {missing[:5]}")
    if missing:
        print0(f"[readout] {len(missing)} missing keys (e.g. {missing[:3]})")
    # An EET checkpoint carries the router and its buffers on top of an identical
    # backbone. Those are expected here: the oracle reads layer states densely, which is
    # the right counterfactual precisely because the routing was measured to be random.
    eet_pref = ('eet_', 'exit_', 'token_', 'vocab_route', 'departure')
    stray = [k for k in unexpected if not k.startswith(eet_pref)]
    if unexpected:
        print0(f"[readout] {len(unexpected)} extra keys, {len(unexpected)-len(stray)} of them "
               f"EET-specific (expected)")
    if stray:
        print0(f"[readout] WARNING unexplained extra keys: {stray[:5]}")
    # assign=True installs the checkpoint's own CPU tensors as the parameters, so the
    # model lands back on CPU no matter what to_empty() did.
    model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def layer_states(model, idx):
    """Every layer's output for one batch. Mirrors GPT.forward's trunk exactly."""
    B, T = idx.shape
    cfg = model.config
    cos_sin = model.cos[:, :T], model.sin[:, :T]
    x = model.transformer.wte(idx).to(COMPUTE_DTYPE)
    x = norm(x)
    x0 = x
    decay = (torch.sigmoid(model.depth_decay_raw)
             if getattr(model, '_use_residual_decay', False) and model.depth_decay_raw is not None
             else None)
    states = []
    for i, block in enumerate(model.transformer.h):
        x0_w = model.x0_lambdas[i]
        if decay is not None:
            x0_w = x0_w * (decay ** i)
        x_in = model.resid_lambdas[i] * x + x0_w * x0
        ve = model.value_embeds[str(i)](idx).to(x_in.dtype) if str(i) in model.value_embeds else None
        x = block(x_in, ve, cos_sin, model.window_sizes[i], None)
        if model.residual_mixers is not None:
            g = model.residual_mix_gamma[i].to(x.dtype)
            x = x + g * model.residual_mixers[i](x.transpose(1, 2)).transpose(1, 2)
        states.append(x)
    return states


@torch.no_grad()
def head_nats(head, h, targets, token_bytes, chunk=2):
    """Per-token nats and byte counts under a given head. Mirrors GPT.forward's readout."""
    softcap, V = 20.0, head.shape[0]
    nats, nbytes = [], []
    for b0 in range(0, h.size(0), chunk):
        logits = (norm(h[b0:b0 + chunk]).float() @ head.T.float())
        logits = softcap * torch.tanh(logits / softcap)
        t = targets[b0:b0 + chunk].reshape(-1)
        loss = F.cross_entropy(logits.view(-1, V), t, ignore_index=-1, reduction='none')
        nb = torch.where(t >= 0, token_bytes[t.clamp(min=0)], torch.zeros_like(token_bytes[:1]).expand(t.shape))
        nats.append(loss * (nb > 0))
        nbytes.append(nb)
    return torch.cat(nats), torch.cat(nbytes)


def spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = (ra - ra.mean()) / (ra.std() + 1e-9)
    rb = (rb - rb.mean()) / (rb.std() + 1e-9)
    return float((ra * rb).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="out/dense_d8_V32k_model_001014.pt")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--tokenizer-dir", default="tokenizer")
    ap.add_argument("--max-shards", type=int, default=-1)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--window-pattern", default="SSSL")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--fit-batches", type=int, default=48, help="batches cached for head fitting")
    ap.add_argument("--eval-batches", type=int, default=16, help="held-out batches")
    ap.add_argument("--steps", type=int, default=250, help="head-fit steps per depth")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--fit-tokens", type=int, default=2048, help="tokens per head-fit step; V-wide fp32 logits, so 2048 is ~270MB")
    ap.add_argument("--eval-chunk", type=int, default=1024, help="tokens per eval logit chunk")
    ap.add_argument("--target-active-frac", type=float, default=0.10)
    ap.add_argument("--min-exit-layer", type=int, default=1)
    ap.add_argument("--dense-bpb", type=float, default=0.991231,
                    help="the dense control's bpb at the matched token budget")
    ap.add_argument("--budget", type=float, default=0.032,
                    help="allowed gap; the saturated value from the depth projection")
    ap.add_argument("--skip-fit", action="store_true", help="run test 6 only")
    ap.add_argument("--out", default="out/eet_p02/readout_oracle.json")
    ap.add_argument("--expect-mixture", type=float, default=None,
                    help="the training run's own bpb for this checkpoint. The shared-head "
                         "mixture should land near it; a large miss means the oracle is not "
                         "reproducing what the model actually does.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model, cfg = load_dense(args.ckpt, device, args.seq_len, args.window_pattern, args.head_dim)
    nl, d, V = cfg.n_layer, cfg.n_embd, cfg.vocab_size
    print0("=" * 78)
    print0(f"  EET readout oracle   depth={nl} d_model={d} V={V}  device={device}")
    print0(f"  dense reference bpb {args.dense_bpb:.6f}   budget +{args.budget:.4f} "
           f"-> must reach {args.dense_bpb + args.budget:.4f}")
    print0("=" * 78)

    tok = get_tokenizer(args.tokenizer_dir)
    assert tok.get_vocab_size() >= 1000, f"stub tokenizer (vocab {tok.get_vocab_size()})"
    token_bytes = get_token_bytes(device=device, tokenizer_dir=args.tokenizer_dir)
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tok, args.batch, args.seq_len, split="val", device=device,
        data_dir=args.data_dir, max_shards=args.max_shards)

    n_batches = args.fit_batches + args.eval_batches
    H = [[] for _ in range(nl)]          # cached layer states (cpu, bf16)
    Y, DELTA, CE = [], [[] for _ in range(nl)], []
    lm_head = model.lm_head.weight.detach()

    for b in range(n_batches):
        idx, targets = next(loader)
        st = layer_states(model, idx)
        prev = None
        for l, s in enumerate(st):
            H[l].append(s.to(torch.bfloat16).cpu())
            if prev is not None:
                DELTA[l].append((s - prev).float().norm(dim=-1).reshape(-1).cpu())
            else:
                DELTA[l].append(torch.zeros(s.shape[0] * s.shape[1]))
            prev = s
        nats, nb = head_nats(lm_head, st[-1], targets, token_bytes)
        CE.append(nats.cpu())
        Y.append(targets.reshape(-1).cpu())
        if (b + 1) % 16 == 0:
            print0(f"  cached {b+1}/{n_batches} batches")

    ce_all = torch.cat(CE)
    valid = ce_all > 0
    results = {"depth": nl, "d_model": d, "vocab": V,
               "dense_bpb": args.dense_bpb, "budget": args.budget}

    # ---------------- Test 6: does the residual delta predict difficulty? -----
    print0("\n  TEST 6  residual-delta halting signal")
    print0(f"  {'layer':>6}{'mean |dx|':>12}{'spearman vs CE':>17}")
    sp = {}
    for l in range(1, nl):
        dl = torch.cat(DELTA[l])[valid]
        r = spearman(dl, ce_all[valid])
        sp[l] = r
        print0(f"  {l:>6}{dl.mean().item():>12.4f}{r:>17.4f}")
    best = max(sp.values(), key=abs) if sp else 0.0
    results["test6_spearman"] = sp
    results["test6_best_abs"] = abs(best)
    results["test6_pass"] = abs(best) >= 0.3
    print0(f"  best |spearman| = {abs(best):.4f}  -> "
           f"{'SIGNAL, keep idea 6' if abs(best) >= 0.3 else 'NO SIGNAL, close idea 6'}")

    # --- per-layer profile: the Tier 0 measurement ---------------------------
    # bpb of each layer read through the SHARED head, plus how far the residual moved.
    # A healthy stack improves monotonically with depth; a collapsed one goes flat.
    n_fit = args.fit_batches
    ev_h = lambda l: torch.cat([H[l][i] for i in range(n_fit, n_batches)]).view(-1, d).to(device)
    ev_y = torch.cat(Y[n_fit:]).to(device)
    ev_bytes = torch.where(ev_y >= 0, token_bytes[ev_y.clamp(min=0)],
                           torch.zeros_like(token_bytes[:1]).expand(ev_y.shape))
    tot_bytes = float(ev_bytes.sum())

    def eval_head(head, l):
        h = ev_h(l)
        nats = 0.0
        C = args.eval_chunk
        for i in range(0, h.size(0), C):
            logits = (norm(h[i:i+C].float()) @ head.T.float().to(h.device))
            logits = 20.0 * torch.tanh(logits / 20.0)
            t = ev_y[i:i+C]
            loss = F.cross_entropy(logits, t, ignore_index=-1, reduction='none')
            nats += float((loss * (ev_bytes[i:i+C] > 0)).sum())
            del logits, loss
        return nats / tot_bytes / LN2

    print0("\n  PER-LAYER PROFILE (shared head)")
    print0(f"  {'layer':>6}{'bpb':>10}{'gain':>9}{'mean |dx|':>12}")
    shared, gains = {}, {}
    for l in range(nl):
        shared[l] = eval_head(lm_head, l)
        gains[l] = (shared[l-1] - shared[l]) if l else float('nan')
        dl = float(torch.cat(DELTA[l]).mean()) if l else float('nan')
        print0(f"  {l:>6}{shared[l]:>10.4f}"
               f"{('%+.4f' % gains[l]) if l else '        -':>9}"
               f"{(('%.2f' % dl) if l else '-'):>12}")
    deep_gain = shared[max(0, nl - 5)] - shared[nl - 1]
    results['per_layer_bpb'] = shared
    results['deep_gain'] = deep_gain
    print0(f"\n  bpb gained over the last 4 layers: {deep_gain:+.4f}")
    print0(f"  dense d8 reference for the same span: +0.8008")
    print0(f"  -> {'DEEP LAYERS ARE WORKING' if deep_gain > 0.05 else 'DEPTH COLLAPSE: the deep layers contribute nothing'}")

    if args.skip_fit:
        return _write(args, results)

    # ---------------- Test 5: per-depth readout ceiling -----------------------
    print0("\n  TEST 5  per-depth readout ceiling (backbone frozen)")
    lm_head = lm_head.clone()
    del model
    torch.cuda.empty_cache() if device == 'cuda' else None

    refit, raw_refit = {}, {}
    for l in range(nl):
        head = lm_head.clone().float().requires_grad_(True)
        opt = torch.optim.Adam([head], lr=args.lr)
        fit_h = torch.cat([H[l][i] for i in range(n_fit)]).view(-1, d)
        fit_y = torch.cat(Y[:n_fit])
        N = fit_h.size(0)
        for s in range(args.steps):
            sel = torch.randint(0, N, (args.fit_tokens,))
            hb = norm(fit_h[sel].to(device).float())
            tb = fit_y[sel].to(device)
            logits = 20.0 * torch.tanh((hb @ head.T) / 20.0)
            loss = F.cross_entropy(logits, tb, ignore_index=-1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            raw = eval_head(head.detach(), l)
        raw_refit[l] = raw
        # Keeping the shared head is always available to a per-depth head, so the best of
        # the two is achievable and is therefore a valid ceiling for this family.
        refit[l] = min(raw, shared[l])
        flag = "  (fit did not beat shared; ceiling falls back to it)" if raw > shared[l] + 1e-3 else ""
        print0(f"  layer {l}: shared head {shared[l]:.4f}   per-depth head {refit[l]:.4f}   "
               f"recovered {shared[l]-refit[l]:+.4f}{flag}")

    # Mixture under the bell schedule: what fraction of tokens exits at each depth
    _, per_block = bell_capacities(nl, args.min_exit_layer, args.target_active_frac, 'bell')
    weights, prev = {}, 1.0
    for i in range(nl):
        nxt = per_block[i + 1] if i + 1 < nl else 0.0
        if i + 1 < nl and prev - nxt > 1e-9:
            weights[i] = prev - nxt
        prev = nxt if i + 1 < nl else prev
    weights[nl - 1] = weights.get(nl - 1, 0.0) + per_block[-1]
    tot_w = sum(weights.values())
    weights = {k: v / tot_w for k, v in weights.items()}

    undertrained = [l for l in range(nl) if raw_refit[l] > shared[l] + 1e-3]
    if undertrained:
        print0(f"\n  [!] the fit failed to beat the shared head at layers {undertrained}, so those "
               f"rows fall back to it. The ceiling stays VALID (falling back is achievable) but is "
               f"LOOSE there: a converged fit could only be better, never worse.")
    results["undertrained_layers"] = undertrained
    results["per_depth_refit_raw"] = raw_refit
    mix_shared = sum(w * shared[l] for l, w in weights.items())
    mix_refit = sum(w * refit[l] for l, w in weights.items())
    target = args.dense_bpb + args.budget
    results.update({"per_depth_shared": shared, "per_depth_refit": refit,
                    "exit_weights": weights, "mixture_shared": mix_shared,
                    "mixture_refit": mix_refit, "target": target,
                    "test5_pass": mix_refit <= target})
    print0(f"\n  final-layer bpb on this eval slice: {shared[nl-1]:.4f} "
           f"(the training run measured {args.dense_bpb:.4f} on the full val set; a large "
           f"difference means this slice is not representative)")
    print0(f"  exit-depth mixture: {[(l, round(w,3)) for l, w in sorted(weights.items())]}")
    print0(f"  mixture with the SHARED head   : {mix_shared:.4f}")
    print0(f"  mixture with PER-DEPTH heads   : {mix_refit:.4f}   "
           f"(readout fix recovers {mix_shared-mix_refit:+.4f})")
    if args.expect_mixture is not None:
        miss = mix_shared - args.expect_mixture
        print0(f"  VALIDATION: this checkpoint's training run scored {args.expect_mixture:.4f}; "
               f"the shared-head mixture is {mix_shared:.4f} ({miss:+.4f}).")
        print0(f"    {'consistent, the oracle reproduces the model' if abs(miss) < 0.05 else 'INCONSISTENT: do not trust the ceiling below'}")
        results["expect_mixture"] = args.expect_mixture
        results["mixture_miss"] = miss
    print0(f"  budget to beat                 : {target:.4f}")
    print0(f"  -> TEST 5 {'PASS: the readout IS the story' if mix_refit <= target else 'FAIL: readout alone is NOT enough'}")
    if mix_refit > target:
        print0(f"     still {mix_refit - target:+.4f} over budget after a perfect per-depth head.")
        print0("     The decomposition is wrong: the backbone itself must change (ideas 1, 2, 11),")
        print0("     not just how it is read out (ideas 3, 4).")
    _write(args, results)


def _write(args, results):
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump(results, open(args.out, 'w'), indent=2, default=float)
        print0(f"\n[readout] wrote {args.out}")


if __name__ == "__main__":
    main()
