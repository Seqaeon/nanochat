"""Group B2-B5 of the MST paper experiments: inference-time probes on a trained
MST checkpoint. No training, no gradient steps.

  B2  zero each stream in turn, measure delta bpb        -> Fig. 4(a)
  B3  per-layer pairwise cosine similarity between streams -> Fig. 4(b)
  B4  router entropy vs layer                            -> Fig. 4(c)
  B5  shrink each stream's attention window, measure delta bpb -> Fig. 4(d)
  B6  per-token loss attribution: which tokens does each stream matter for?
  B7  stream logit-lens: what does each stream predict on its own?
  B8  weight-space similarity between the per-stream matrices

B6-B8 are the closest available analogue to MoE expert-specialisation analysis.
The analogy is imperfect and worth stating: in MoE the router *assigns* tokens to
experts, so "which tokens go to expert 3" is well posed. In MST every stream
processes every token and the router only decides how to summarise them, so there
is no assignment to inspect. These probes instead ask which tokens each stream is
load-bearing for (B6), what it would predict alone (B7), and whether the
symmetry-breaking matrices actually diverged (B8).

B2 is the one that answers the reviewer question the paper invites by reporting
a 7.2x gradient imbalance: does a partitioned model silently collapse into a
narrower one? If zeroing any single stream costs about the same, it does not.

    python -m scripts.paper_probe --ckpt-dir out/.../S7_COMBO_A_D32 \\
        --data-dir data --tokenizer-dir tokenizer --out scratch/probe_d32.json
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanochat.loss_eval import evaluate_bpb
from nanochat.tokenizer import get_token_bytes
from scripts.paper_lib import load_any_model, build_val_batches


def measure_bpb(model, tokenizer, token_bytes, device, batch, seq, steps,
                data_dir, max_shards):
    batches = build_val_batches(tokenizer, batch, seq, device,
                                data_dir=data_dir, max_shards=max_shards)
    with torch.inference_mode():
        bpb, _ = evaluate_bpb(model, batches, steps, token_bytes)
    return float(bpb)


# --------------------------------------------------------------- B2
class _ZeroStream:
    """Zero one sub-stream at the output of every MST layer.

    The batched path carries state as (B, T, N, d), so zeroing index j at every
    layer removes that stream's contribution everywhere, not just at the head.
    """
    def __init__(self, model, j):
        self.model, self.j, self.handles = model, j, []

    def __enter__(self):
        j = self.j
        def hook(_mod, _inp, output):
            if isinstance(output, tuple):
                states, aux = output[0], output[1:]
                states = states.clone()
                states[:, :, j, :] = 0
                return (states,) + aux
            out = output.clone()
            out[:, :, j, :] = 0
            return out
        for layer in self.model.layers:
            self.handles.append(layer.register_forward_hook(hook))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


def b2_stream_ablation(model, cfg, ev):
    base = ev(model)
    print(f"  baseline bpb {base:.4f}")
    rows = []
    for j in range(cfg.mst_n_subs):
        with _ZeroStream(model, j):
            b = ev(model)
        rows.append(dict(stream=j, bpb=b, delta=b - base))
        print(f"  zero stream {j}: bpb {b:.4f}   delta {b-base:+.4f}")
    d = [r["delta"] for r in rows]
    print(f"  spread: min {min(d):+.4f}  max {max(d):+.4f}  ratio {max(d)/max(min(d),1e-9):.2f}x")
    return dict(baseline=base, per_stream=rows)


# --------------------------------------------------------------- B3 / B4
def b3b4_diagnostics(model, tokenizer, device, batch, seq, data_dir, max_shards,
                     n_batches=4):
    """Cosine similarity between streams and router entropy, per layer."""
    batches = build_val_batches(tokenizer, batch, seq, device,
                                data_dir=data_dir, max_shards=max_shards)
    it = iter(batches)
    model._diag_enabled = True
    agg = {}
    with torch.inference_mode():
        for _ in range(n_batches):
            x, y = next(it)
            model(x, y)
            d = model.compute_diagnostics()
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    agg.setdefault(k, []).append(v)
    model._diag_enabled = False
    mean = {k: sum(v) / len(v) for k, v in agg.items()}
    n = model.config.mst_n_subs
    floor = -1.0 / (n - 1)
    sims = {int(k.split("_L")[1].split("_")[0]): v
            for k, v in mean.items() if k.startswith("sub_sim_L") and k.endswith("_mean")}
    ents = {int(k.split("_L")[1]): v for k, v in mean.items() if k.startswith("route_entropy_L")}
    print(f"  cosine floor for N={n} is {floor:+.4f}; log N = {torch.log(torch.tensor(float(n))):.4f}")
    for l in sorted(sims):
        e = ents.get(l)
        print(f"    layer {l:2d}: sub_sim {sims[l]:+.4f}"
              + (f"   route_entropy {e:.4f}" if e is not None else ""))
    return dict(cosine_floor=floor, sub_sim=sims, route_entropy=ents, raw=mean)


# --------------------------------------------------------------- B5
def b5_window_ablation(model, cfg, ev, shrink_to=32):
    """Shrink one stream's window at a time and see whether the damage tracks
    the scale that stream was assigned."""
    windows = getattr(model, "sub_window_sizes", None)
    if not windows:
        print(f"  this checkpoint has mst_multi_scale_windows="
              f"{getattr(cfg, 'mst_multi_scale_windows', '?')}, so the model was "
              f"built with sub_window_sizes={windows}.")
        print("  if you believe multi-scale WAS enabled for this run, the "
              "checkpoint's saved config disagrees, and every other probe here "
              "is measuring a model without it. Check the run's config before "
              "trusting B2. Re-run with --force-multi-scale to override.")
        return None
    orig = list(model.sub_window_sizes)
    base = ev(model)
    print(f"  baseline bpb {base:.4f}   windows {orig}")
    rows = []
    for j in range(len(orig)):
        model.sub_window_sizes = list(orig)
        model.sub_window_sizes[j] = (shrink_to, 0)
        b = ev(model)
        rows.append(dict(stream=j, orig_window=orig[j][0], bpb=b, delta=b - base))
        print(f"  stream {j} window {orig[j][0]:>5} -> {shrink_to}: "
              f"bpb {b:.4f}   delta {b-base:+.4f}")
    model.sub_window_sizes = orig
    return dict(baseline=base, original=orig, per_stream=rows)


# --------------------------------------------------------------- B6
def b6_token_attribution(model, cfg, tokenizer, token_bytes, device, batch, seq,
                         steps, data_dir, max_shards, freq_table=None):
    """Per-token loss attribution: bucket the damage from zeroing each stream.

    B2 asks how much a stream matters overall. This asks *which tokens* it
    matters for, which is the question MoE specialisation analyses answer by
    inspecting routing assignments. Buckets are token frequency (quartiles of the
    corpus unigram distribution) and position in the sequence.
    """
    N = cfg.mst_n_subs

    def per_token_loss(m):
        """Returns (losses, targets, positions) concatenated over eval steps."""
        batches = build_val_batches(tokenizer, batch, seq, device,
                                    data_dir=data_dir, max_shards=max_shards)
        it = iter(batches)
        L, Y, P = [], [], []
        with torch.inference_mode():
            for _ in range(steps):
                x, y = next(it)
                loss = m(x, y, loss_reduction='none').reshape(y.shape)
                L.append(loss.float().cpu()); Y.append(y.cpu())
                P.append(torch.arange(y.shape[1]).expand_as(y).cpu())
        return torch.cat([t.reshape(-1) for t in L]), \
               torch.cat([t.reshape(-1) for t in Y]), \
               torch.cat([t.reshape(-1) for t in P])

    base_l, y, pos = per_token_loss(model)
    valid = y >= 0
    base_l, y, pos = base_l[valid], y[valid], pos[valid]
    print(f"  baseline mean token loss {base_l.mean():.4f} over {len(y):,} tokens")

    # frequency quartiles: bucket 0 = rarest
    if freq_table is not None:
        tf = freq_table.cpu()[y.clamp(min=0)]
        q = torch.quantile(tf.float(), torch.tensor([0.25, 0.5, 0.75]))
        freq_bucket = torch.bucketize(tf.float(), q)
        fnames = ["rarest 25%", "25-50%", "50-75%", "commonest 25%"]
    else:
        freq_bucket, fnames = None, []
    pos_bucket = torch.bucketize(pos.float(),
                                 torch.tensor([seq * 0.125, seq * 0.5]))
    pnames = [f"pos 0-{seq//8}", f"pos {seq//8}-{seq//2}", f"pos {seq//2}-{seq}"]

    rows = []
    for j in range(N):
        with _ZeroStream(model, j):
            lj, _, _ = per_token_loss(model)
        lj = lj[valid]
        d = lj - base_l
        r = dict(stream=j, mean_delta=float(d.mean()))
        if freq_bucket is not None:
            r["by_freq"] = {fnames[b]: float(d[freq_bucket == b].mean())
                            for b in range(4) if (freq_bucket == b).any()}
        r["by_pos"] = {pnames[b]: float(d[pos_bucket == b].mean())
                       for b in range(3) if (pos_bucket == b).any()}
        rows.append(r)
        fs = "  ".join(f"{k} {v:+.3f}" for k, v in r.get("by_freq", {}).items())
        ps = "  ".join(f"{k} {v:+.3f}" for k, v in r["by_pos"].items())
        print(f"  stream {j}: mean {r['mean_delta']:+.3f} | {fs} | {ps}")

    # a stream "specialises" if its damage profile deviates from the average
    if rows and "by_freq" in rows[0]:
        print("  relative profile (stream delta / mean delta, by frequency):")
        for r in rows:
            rel = {k: v / r["mean_delta"] for k, v in r["by_freq"].items()}
            print(f"    stream {r['stream']}: "
                  + "  ".join(f"{k} {v:.2f}" for k, v in rel.items()))
    return rows


# --------------------------------------------------------------- B7
def b7_logit_lens(model, cfg, tokenizer, device, batch, seq, data_dir,
                  max_shards, topk=10):
    """What does each stream predict on its own?

    Zero every stream but one in the final concatenation, push it through the
    output head, and read the resulting distribution. This asks what stream j
    would say if it were the only one speaking.
    """
    N, d = cfg.mst_n_subs, cfg.mst_sub_dim
    captured = {}

    def hook(_m, _i, output):
        captured["h"] = output[0] if isinstance(output, tuple) else output
    handle = model.layers[-1].register_forward_hook(hook)

    batches = build_val_batches(tokenizer, batch, seq, device,
                                data_dir=data_dir, max_shards=max_shards)
    x, y = next(iter(batches))
    with torch.inference_mode():
        model(x)
        H = captured["h"]                                   # (B, T, N, d)
        from nanochat.gpt import norm
        Hn = norm(H)
        full = model.lm_head(model.final_head.proj(Hn.reshape(*Hn.shape[:2], N * d)))
        full_top = full[..., :cfg.vocab_size].argmax(-1)

        out = []
        for j in range(N):
            m = torch.zeros_like(Hn); m[:, :, j, :] = 1.0
            lj = model.lm_head(model.final_head.proj((Hn * m).reshape(*Hn.shape[:2], N * d)))
            lj = lj[..., :cfg.vocab_size]
            agree = float((lj.argmax(-1) == full_top).float().mean())
            ids = lj.float().mean(dim=(0, 1)).topk(topk).indices.tolist()
            try:
                toks = [repr(tokenizer.decode([i]))[:14] for i in ids]
            except Exception:
                toks = [str(i) for i in ids]
            ent = float(-(lj.float().softmax(-1)
                          * lj.float().log_softmax(-1)).sum(-1).mean())
            out.append(dict(stream=j, top1_agreement=agree, entropy=ent,
                            top_tokens=toks))
            print(f"  stream {j}: top-1 agreement with full model {agree:6.2%}   "
                  f"entropy {ent:.3f}")
            print(f"    favours: {' '.join(toks[:8])}")
    handle.remove()
    return out


# --------------------------------------------------------------- B8
def b8_weight_similarity(model, cfg):
    """Did the per-stream matrices actually diverge?

    Unlike the activation-similarity statistic, this comparison is well posed:
    the N distribute matrices all map from the *same* aggregate vector, so
    cosine between them measures whether the streams receive different messages.
    Near-identical W^D_i would mean the symmetry breaking never took.
    """
    N, d = cfg.mst_n_subs, cfg.mst_sub_dim
    names = ("distribute_w", "c_q_w", "c_k_w", "c_v_w", "fc_w")
    res = {}
    for nm in names:
        per_layer = []
        for layer in model.layers:
            w = getattr(layer, nm, None)
            if w is None or w.shape[0] % N:
                continue
            W = w.detach().float().view(N, -1)
            W = W / W.norm(dim=1, keepdim=True).clamp_min(1e-8)
            C = W @ W.T
            off = C[~torch.eye(N, dtype=torch.bool, device=C.device)]
            per_layer.append(float(off.mean()))
        if per_layer:
            res[nm] = per_layer
            print(f"  {nm:14s} mean pairwise cosine: "
                  f"first {per_layer[0]:+.4f}  mid {per_layer[len(per_layer)//2]:+.4f}  "
                  f"last {per_layer[-1]:+.4f}  (overall {sum(per_layer)/len(per_layer):+.4f})")
    print(f"  note: 0 means the {N} streams hold unrelated matrices; "
          f"+1 would mean the symmetry breaking failed")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval-steps", type=int, default=40)
    ap.add_argument("--only", default=None, help="b2,b3,b5,b6,b7,b8")
    ap.add_argument("--force-multi-scale", action="store_true",
                    help="rebuild per-stream windows even if the checkpoint "
                         "config says multi-scale was off (diagnostic only)")
    ap.add_argument("--out", default="scratch/paper_probe.json")
    args = ap.parse_args()

    # Validate the probe selection before loading a multi-GB checkpoint, and
    # fail loudly on a name this build does not have: silently running nothing
    # looks identical to running everything and finding nothing.
    KNOWN = ("b2", "b3", "b5", "b6", "b7", "b8")
    sel = [k.strip() for k in args.only.split(",")] if args.only else list(KNOWN)
    unknown = [k for k in sel if k not in KNOWN]
    if unknown:
        print(f"unknown probe(s) {unknown}; this build knows {list(KNOWN)}.")
        print("if you expected one of these to exist, the checkout is stale: "
              "git pull and re-run.")
        sys.exit(1)
    print(f"probes: {', '.join(sel)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, cfg, _ = load_any_model(args.ckpt_dir, device, step=args.step,
                                              tokenizer_dir=args.tokenizer_dir)
    if not getattr(cfg, "use_mst", False):
        print("this checkpoint is a dense model; these probes are MST-only")
        sys.exit(1)
    if args.force_multi_scale and not getattr(model, "sub_window_sizes", None):
        import math
        N, T = cfg.mst_n_subs, cfg.sequence_len
        model.sub_window_sizes = [
            (-1, 0) if j == N - 1
            else (min(int(32 * (T / 32) ** (j / max(1, N - 1))), T), 0)
            for j in range(N)]
        print(f"  --force-multi-scale: {model.sub_window_sizes}")
    token_bytes = get_token_bytes(device=device, tokenizer_dir=args.tokenizer_dir)
    seq = cfg.sequence_len

    ev = lambda m: measure_bpb(m, tokenizer, token_bytes, device, args.batch, seq,
                               args.eval_steps, args.data_dir, args.max_shards)
    want = lambda k: k in sel
    out = dict(ckpt=args.ckpt_dir, n_layer=cfg.n_layer, n_subs=cfg.mst_n_subs,
               sub_dim=cfg.mst_sub_dim, eval_steps=args.eval_steps)

    if want("b2"):
        print("\n[B2] zero each stream at inference")
        out["b2"] = b2_stream_ablation(model, cfg, ev)
    if want("b3"):
        print("\n[B3/B4] per-layer cosine similarity and router entropy")
        out["b3b4"] = b3b4_diagnostics(model, tokenizer, device, args.batch, seq,
                                       args.data_dir, args.max_shards)
    if want("b5"):
        print("\n[B5] shrink each stream's window")
        out["b5"] = b5_window_ablation(model, cfg, ev)
    if want("b6"):
        print("\n[B6] per-token loss attribution")
        try:
            ft = torch.load(os.path.join(args.tokenizer_dir or "tokenizer",
                                         "freq_table.pt"), map_location="cpu",
                            weights_only=False)
        except Exception as e:
            print(f"  no freq_table.pt ({e}); position buckets only")
            ft = None
        out["b6"] = b6_token_attribution(model, cfg, tokenizer, token_bytes,
                                         device, args.batch, seq,
                                         max(4, args.eval_steps // 4),
                                         args.data_dir, args.max_shards, ft)
    if want("b7"):
        print("\n[B7] stream logit-lens")
        out["b7"] = b7_logit_lens(model, cfg, tokenizer, device, args.batch, seq,
                                  args.data_dir, args.max_shards)
    if want("b8"):
        print("\n[B8] weight-space similarity between streams")
        out["b8"] = b8_weight_similarity(model, cfg)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
