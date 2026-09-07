#!/usr/bin/env python3
"""EET P02 Test 0A: the context-destruction oracle.

WHAT THIS DECIDES
-----------------
EET's compute-skip path removes an exited token from every later layer's attention, not
just from its own computation. At d8 with the bell schedule, layers 5-7 attend over
26.7%, 14.6% and 12.5% of the sequence. The surviving tokens are by construction the
hard ones, and they are denied most of their context.

None of the thirteen abandoned EET experiments touched this. They all addressed gradient
flow, learning rates, representation alignment, distillation or scheduling. This script
measures the size of the effect on a TRAINED DENSE checkpoint, with no training at all,
by imposing EET's key masking while holding depth constant. If the number is small, the
context hypothesis is dead and the ``--eet-kv-mode`` training arms should not be launched.

PRE-REGISTERED GATE
-------------------
    delta_bpb(ctx) < 0.02   ->  context destruction is NOT the dominant cause.
                                Do not run Test 1. Close Defect 1.
    delta_bpb(ctx) >= 0.02  ->  run Test 1 (``--eet-kv-mode fresh``).

Write the number down before interpreting it.

WHAT THE THREE ABLATIONS SEPARATE
---------------------------------
    ctx     every token still runs every layer, but attention keys at layer L are
            restricted to the tokens EET would still have active there. Isolates the
            cost of losing context, with depth held constant. THIS IS THE GATE.
    depth   tokens genuinely exit and are read out from their exit-layer state, but
            attention keys are never restricted. Isolates the cost of losing depth.
    both    EET's actual inference behaviour, for reference.

Note that these are upper-bound-ish estimates on an un-co-trained model: a dense
checkpoint was never trained to tolerate either ablation. The point is the RATIO between
ctx and depth, which says which defect to spend GPU time on.

ROUTER PROXIES
--------------
EET's global router reads only the token embedding (``use_pos_embed`` is off), so the
exit depth it learns is close to a per-vocabulary-item lookup table correlated with
frequency. ``--rank freq`` reproduces that. ``--rank random`` is the control that says
how much of the effect is just "fewer keys". ``--rank ce`` is a best-case router that
exits whatever the model already predicts well, and bounds how much a better router
could help.

USAGE
-----
    python -m scripts.eet_context_oracle \
        --ckpt-dir out/eet_p02/DENSE_D8/ckpt_base/base \
        --target-active-frac 0.125 --min-exit-layer 1 \
        --eval-tokens 2000000 --out out/eet_p02/oracle_d8.json
"""
import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

from nanochat.checkpoint_manager import find_last_step, load_checkpoint
from nanochat.common import COMPUTE_DTYPE, print0
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.gpt import GPT, GPTConfig, norm
from nanochat.tokenizer import get_tokenizer, get_token_bytes


# ---------------------------------------------------------------------------
# EET's capacity schedule, reproduced exactly (see nanochat/eet.py compute_skip)
# ---------------------------------------------------------------------------
READOUT_CHUNK = 2   # sequences per lm_head call; see readout_nats


def bell_capacities(n_layer, min_exit_layer, target_frac, schedule='bell'):
    """Return (routing_layers, per_block_active_frac).

    per_block_active_frac[i] is the fraction of tokens still active when block i runs,
    which is what determines both the FLOP saving and the size of the key set.
    """
    routing_layers = list(range(min_exit_layer, n_layer - 1))
    n_rl = len(routing_layers)
    if n_rl <= 0:
        return [], [1.0] * n_layer

    if schedule == 'uniform':
        fracs = [(1.0 - target_frac) / n_rl] * n_rl
    elif schedule == 'linear':
        w = [k + 1 for k in range(n_rl)]
        fracs = [x / sum(w) * (1.0 - target_frac) for x in w]
    elif schedule == 'geometric':
        per = 1.0 - target_frac ** (1.0 / n_rl)
        fracs = [per] * n_rl
    else:  # bell
        mid = (n_rl - 1) / 2.0
        sigma = max(1.0, n_rl / 4.0)
        w = [math.exp(-((k - mid) / sigma) ** 2) for k in range(n_rl)]
        fracs = [x / sum(w) * (1.0 - target_frac) for x in w]

    survivor, caps = 1.0, []
    for f in fracs:
        survivor -= f
        caps.append(max(1e-6, survivor))

    per_block = []
    rl = 0
    cur = 1.0
    for i in range(n_layer):
        per_block.append(cur)
        if i in routing_layers and rl < n_rl:
            cur = caps[rl]
            rl += 1
    return routing_layers, per_block


# ---------------------------------------------------------------------------
def load_dense(ckpt_dir, device, step=None):
    step = step if step is not None else find_last_step(ckpt_dir)
    model_data, _, meta = load_checkpoint(ckpt_dir, step, device, load_optimizer=False)
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    cfg_kw = dict(meta["model_config"])
    cfg_kw.setdefault("window_pattern", "L")
    # The oracle always runs the dense path, even on an EET checkpoint.
    cfg_kw["use_eet"] = False
    config = GPTConfig(**cfg_kw)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    missing, unexpected = model.load_state_dict(model_data, strict=False, assign=True)
    if missing:
        print0(f"[oracle] {len(missing)} missing keys (first few: {missing[:4]})")
    model.eval()
    return model, config, meta, step


def build_rank_scores(mode, idx, model, freq_table, per_token_ce):
    """Higher score = exits earlier. Shape (B, T)."""
    if mode == 'freq':
        # Frequent tokens leave first, which is what a router reading only the token
        # embedding converges to.
        return freq_table[idx].float()
    if mode == 'random':
        return torch.rand(idx.shape, device=idx.device)
    if mode == 'ce':
        # Best-case router: whatever the full model already predicts well can go.
        return -per_token_ce
    raise ValueError(mode)


@torch.no_grad()
def oracle_forward(model, idx, targets, per_block_frac, rank_score, ablation, token_bytes):
    """One forward pass under a given ablation. Returns (sum_nats, sum_bytes).

    ablation:
        'dense'  untouched reference
        'ctx'    keys restricted to the active set, depth held constant
        'depth'  tokens read out at their exit layer, keys unrestricted
        'both'   both
    """
    B, T = idx.shape
    device = idx.device
    config = model.config
    cos_sin = model.cos[:, :T], model.sin[:, :T]

    # Per-layer active sets. Ranking is per sequence so each row keeps exactly K_i tokens,
    # which is what EET's fixed-capacity top-K does.
    order = torch.argsort(rank_score, dim=-1, descending=True)  # earliest-exit first
    rank_of = torch.empty_like(order)
    rank_of.scatter_(1, order, torch.arange(T, device=device).expand(B, -1))

    # Mirror GPT.forward exactly. Any divergence here would show up as a constant offset
    # on every ablation, which mostly cancels in the deltas but would still make the
    # oracle's 'dense' row disagree with the checkpoint's own val_bpb. --check compares
    # them so a drift is visible rather than silent.
    x = model.transformer.wte(idx) if model.embedding_model is None else model.embedding_model(idx)[0]
    x = x.to(COMPUTE_DTYPE)
    x = norm(x)
    x0 = x
    x_exit = x0.clone()          # readout state per token, for the depth ablations
    prev_active = torch.ones(B, T, dtype=torch.bool, device=device)
    decay_base = (torch.sigmoid(model.depth_decay_raw)
                  if getattr(model, '_use_residual_decay', False) and model.depth_decay_raw is not None
                  else None)

    for i, block in enumerate(model.transformer.h):
        frac = per_block_frac[i]
        k_keep = max(1, int(frac * T))   # int() truncation, matching eet.py
        active = rank_of >= (T - k_keep)   # keep the LAST k_keep in exit order

        x0_w = model.x0_lambdas[i]
        if decay_base is not None:
            x0_w = x0_w * (decay_base ** i)
        x_input = model.resid_lambdas[i] * x + x0_w * x0
        ve = None
        if str(i) in model.value_embeds:
            ve = model.value_embeds[str(i)](idx).to(x_input.dtype)

        key_mask = active if ablation in ('ctx', 'both') else None
        if key_mask is None:
            x_new = block(x_input, ve, cos_sin, model.window_sizes[i], None)
        else:
            attn_norm = block.norm_attn if block.norm_attn is not None else norm
            k, v = block.attn._project_kv(attn_norm(x_input), ve, cos_sin[0], cos_sin[1])
            q_pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            x_new = block.forward_split(x_input, cos_sin, q_pos, k, v,
                                        model.window_sizes[i], key_mask=key_mask)

        if model.residual_mixers is not None:
            gamma = model.residual_mix_gamma[i].to(x_new.dtype)
            mixed = model.residual_mixers[i](x_new.transpose(1, 2)).transpose(1, 2)
            x_new = x_new + gamma * mixed

        if ablation in ('depth', 'both'):
            # A token that just left is frozen: record its state and stop updating it.
            just_left = prev_active & ~active
            x_exit = torch.where(just_left.unsqueeze(-1), x_new, x_exit)
            x = torch.where(active.unsqueeze(-1), x_new, x)
            prev_active = active
        else:
            x = x_new

    if ablation in ('depth', 'both'):
        x = torch.where(prev_active.unsqueeze(-1), x, x_exit)

    return readout_nats(model, norm(x), targets, token_bytes, chunk=READOUT_CHUNK)


@torch.no_grad()
def readout_nats(model, hidden, targets, token_bytes, chunk=2):
    """Mirror of GPT.forward's dense readout: lm_head, depad, fp32, softcap 20.

    Chunked over the batch because the fp32 logits are (chunk, T, V): at T=2048 and
    V=32768 that is 0.5 GB per sequence, and the oracle runs 12 of these forwards per
    batch (3 router proxies x 4 ablations).
    """
    device = hidden.device
    sum_nats = torch.zeros((), dtype=torch.float32, device=device)
    sum_bytes = torch.zeros((), dtype=torch.int64, device=device)
    softcap = 20.0
    for b0 in range(0, hidden.size(0), chunk):
        logits = model.lm_head(hidden[b0:b0 + chunk])[..., :model.config.vocab_size].float()
        logits = softcap * torch.tanh(logits / softcap)
        t = targets[b0:b0 + chunk].reshape(-1)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), t,
                               ignore_index=-1, reduction='none')
        nb = torch.where(t >= 0, token_bytes[t.clamp(min=0)], torch.zeros_like(token_bytes[:1]).expand(t.shape))
        sum_nats += (loss * (nb > 0)).sum()
        sum_bytes += nb.sum()
    return sum_nats, sum_bytes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", type=str, required=True, help="dense checkpoint directory")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--tokenizer-dir", type=str, default=None)
    ap.add_argument("--max-shards", type=int, default=-1)
    ap.add_argument("--device-batch-size", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--eval-tokens", type=int, default=2_000_000)
    ap.add_argument("--target-active-frac", type=float, default=0.125)
    ap.add_argument("--min-exit-layer", type=int, default=1)
    ap.add_argument("--capacity-schedule", type=str, default="bell",
                    choices=["bell", "uniform", "linear", "geometric"])
    ap.add_argument("--rank", type=str, default="freq,random",
                    help="comma-separated router proxies: freq, random, ce")
    ap.add_argument("--ablation", type=str, default="ctx,depth,both",
                    help="comma-separated ablations to run alongside the dense reference")
    ap.add_argument("--gate-delta-bpb", type=float, default=0.02,
                    help="pre-registered threshold on delta_bpb(ctx)")
    ap.add_argument("--out", type=str, default=None, help="write JSON results here")
    ap.add_argument("--check", action="store_true", default=True,
                    help="compare the oracle's unablated 'dense' row against the model's own "
                         "forward on the first batch (on by default; it costs one batch)")
    ap.add_argument("--no-check", dest="check", action="store_false")
    ap.add_argument("--readout-chunk", type=int, default=2,
                    help="sequences per lm_head call; lower it if the readout OOMs")
    args = ap.parse_args()
    global READOUT_CHUNK
    READOUT_CHUNK = args.readout_chunk

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model, config, meta, step = load_dense(args.ckpt_dir, device, args.step)
    assert not getattr(config, 'use_code_head', False), (
        "the oracle replicates the standard dense readout; rerun the control with the "
        "default lm_head or extend readout_nats()")
    assert not getattr(config, 'use_remix_linear', False), (
        "remix-linear blocks return (x, ctx) and thread context block-to-block; the "
        "oracle's loop does not, and would mis-measure rather than fail")
    n_layer = config.n_layer
    routing_layers, per_block = bell_capacities(
        n_layer, args.min_exit_layer, args.target_active_frac, args.capacity_schedule)
    avg_active = sum(per_block) / len(per_block)

    print0("=" * 74)
    print0(f"  EET P02 Test 0A: context oracle   depth={n_layer}  step={step}")
    print0(f"  schedule={args.capacity_schedule}  target_active_frac={args.target_active_frac}")
    print0("  per-block active fraction: " + " ".join(f"{f:.3f}" for f in per_block))
    print0(f"  average active fraction:   {avg_active:.4f}  "
           f"(that is a {100*(1-avg_active):.1f}% FLOP cut)")
    print0(f"  PRE-REGISTERED GATE: run Test 1 only if delta_bpb(ctx) >= {args.gate_delta_bpb}")
    print0("=" * 74)

    tokenizer = get_tokenizer(args.tokenizer_dir) if args.tokenizer_dir else get_tokenizer()
    token_bytes = get_token_bytes(device=device, tokenizer_dir=args.tokenizer_dir)

    freq_table = None
    if 'freq' in args.rank:
        from nanochat.eet import FrequencyPrior
        # FrequencyPrior._load_or_compute map_location's the cached table to CPU and
        # ignores its device argument, so move it explicitly.
        freq_table = FrequencyPrior(config.vocab_size, args.tokenizer_dir, device).freq_bias.to(device)

    steps = max(1, args.eval_tokens // (args.device_batch_size * args.max_seq_len))
    ranks = [r.strip() for r in args.rank.split(',') if r.strip()]
    ablations = ['dense'] + [a.strip() for a in args.ablation.split(',') if a.strip()]

    totals = {(r, a): [0.0, 0] for r in ranks for a in ablations}
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, args.max_seq_len, split="val",
        device=device, data_dir=args.data_dir, max_shards=args.max_shards)
    it = iter(loader)

    check_done = not args.check
    for s in range(steps):
        idx, targets = next(it)
        if not check_done:
            check_done = True
            with torch.no_grad():
                ref_loss = model(idx, targets, loss_reduction='none').view(-1)
                nb = token_bytes[targets.reshape(-1).clamp(min=0)]
                ref_bpb = float((ref_loss * (nb > 0)).sum()) / float(nb.sum()) / math.log(2.0)
            on, ob = oracle_forward(model, idx, targets, per_block,
                                    torch.rand(idx.shape, device=idx.device), 'dense', token_bytes)
            orc_bpb = float(on) / int(ob) / math.log(2.0)
            print0(f"[check] model forward bpb={ref_bpb:.4f}  oracle 'dense' bpb={orc_bpb:.4f}"
                   f"  diff={orc_bpb - ref_bpb:+.4f}")
            if abs(orc_bpb - ref_bpb) > 2e-3:
                print0("[check] WARNING: the oracle's dense reimplementation does not match the "
                       "model's own forward. Deltas below are still self-consistent, but fix "
                       "this before quoting the absolute numbers.")
        # Per-token CE from the untouched model, needed by the 'ce' proxy.
        per_token_ce = None
        if 'ce' in ranks:
            per_token_ce = model(idx, targets, loss_reduction='none').detach().view(idx.shape)
        for r in ranks:
            score = build_rank_scores(r, idx, model, freq_table, per_token_ce)
            for a in ablations:
                nats, nbytes = oracle_forward(model, idx, targets, per_block, score, a, token_bytes)
                totals[(r, a)][0] += float(nats)
                totals[(r, a)][1] += int(nbytes)
        if (s + 1) % 10 == 0 or s == steps - 1:
            print0(f"  [{s+1}/{steps}] batches done")

    results = {
        'depth': n_layer, 'step': step,
        'target_active_frac': args.target_active_frac,
        'min_exit_layer': args.min_exit_layer,
        'capacity_schedule': args.capacity_schedule,
        'per_block_active_frac': per_block,
        'avg_active_frac': avg_active,
        'flop_cut': 1.0 - avg_active,
        'gate_delta_bpb': args.gate_delta_bpb,
        'bpb': {}, 'delta_bpb': {},
    }
    ln2 = math.log(2.0)
    print0("")
    print0(f"{'rank':<8}{'ablation':<10}{'bpb':>10}{'delta':>10}")
    print0("-" * 38)
    for r in ranks:
        base = totals[(r, 'dense')]
        base_bpb = base[0] / base[1] / ln2
        for a in ablations:
            t = totals[(r, a)]
            bpb = t[0] / t[1] / ln2
            results['bpb'][f"{r}/{a}"] = bpb
            results['delta_bpb'][f"{r}/{a}"] = bpb - base_bpb
            print0(f"{r:<8}{a:<10}{bpb:>10.4f}{bpb - base_bpb:>+10.4f}")

    # --- the gate -----------------------------------------------------------
    print0("")
    key = 'freq/ctx' if 'freq/ctx' in results['delta_bpb'] else None
    if key is None:
        print0("[oracle] no freq/ctx measurement; gate not evaluated")
    else:
        d = results['delta_bpb'][key]
        results['gate_passed'] = bool(d >= args.gate_delta_bpb)
        verdict = "RUN TEST 1" if results['gate_passed'] else "DO NOT RUN TEST 1"
        print0(f"[GATE] delta_bpb(freq/ctx) = {d:+.4f}   threshold {args.gate_delta_bpb:.4f}"
               f"   ->  {verdict}")
        if 'freq/depth' in results['delta_bpb']:
            dd = results['delta_bpb']['freq/depth']
            if abs(dd) > 1e-9:
                print0(f"[GATE] context/depth cost ratio = {d/dd:.2f} "
                       f"(>1 means context loss dominates depth loss)")
        if 'random/ctx' in results['delta_bpb']:
            print0(f"[GATE] control: random ranking gives {results['delta_bpb']['random/ctx']:+.4f}; "
                   "a similar number means the effect is 'fewer keys', not 'wrong keys'")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print0(f"[oracle] wrote {args.out}")


if __name__ == "__main__":
    main()
