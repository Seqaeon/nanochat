"""Group B2-B5 of the MST paper experiments: inference-time probes on a trained
MST checkpoint. No training, no gradient steps.

  B2  zero each stream in turn, measure delta bpb        -> Fig. 4(a)
  B3  per-layer pairwise cosine similarity between streams -> Fig. 4(b)
  B4  router entropy vs layer                            -> Fig. 4(c)
  B5  shrink each stream's attention window, measure delta bpb -> Fig. 4(d)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval-steps", type=int, default=40)
    ap.add_argument("--only", default=None, help="b2,b3,b5")
    ap.add_argument("--force-multi-scale", action="store_true",
                    help="rebuild per-stream windows even if the checkpoint "
                         "config says multi-scale was off (diagnostic only)")
    ap.add_argument("--out", default="scratch/paper_probe.json")
    args = ap.parse_args()

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
    want = lambda k: args.only is None or k in args.only.split(",")
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

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
