"""O5: whole-model binarisation sensitivity scan.

Where in a trained transformer is floating point actually load-bearing?  Binarise
one component class at a time, hold everything else in bf16, and measure val bpb.

Two variants, and the difference decides what a result MEANS (see LEARNINGS,
"A projection oracle and a free-fit oracle have OPPOSITE pass/fail asymmetries"):

  projection  sign(W) * scale, zero degrees of freedom.  PENALISES, because the
              weights were never shaped for the constraint.  A pass is informative,
              a failure is weak.
  fitted      not implemented here; it needs an optimisation loop against the
              frozen teacher and belongs on a real GPU.

WHAT "BINARISE" MEANS HERE, because it is easy to measure the wrong thing.  The
plan's section 1 definition is: every learned parameter one bit, EVERY MATMUL
OPERAND one bit, the operation itself XNOR + popcount into an integer accumulator.
This script therefore has two independent axes and reports them separately:

  --binarise weights   sign(W), activations untouched.  This is W1A16.  It is an
                       UPPER BOUND on a binary model's quality and it changes no
                       operation: still a bf16 GEMM over a matrix holding two
                       distinct values per row.
  --binarise acts      sign(x) on the layer input, weights untouched.  A16W1
                       inverted; isolates the half BitNet refuses to go below 8 bits.
  --binarise both      W1A1.  The only arm that corresponds to the plan's claim.

And the scale is a second axis, because a per-row float scale is a float parameter:
  --scale none         strict.  Nothing but signs.
  --scale row          XNOR-Net L1-optimal per-output-channel scale (weights) and
                       per-token scale (activations).  Permitted only as a Phase 1
                       ladder rung; section 4.1 says a surviving scalar breaks the
                       title.
The gap between them prices exactly what the scales are worth.

Run:
  python -m scripts.o5_sensitivity --ckpt out/dense_d8_V32k_model_001014.pt
"""
import argparse
import glob
import os
import time

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb

HEAD_DIM, ASPECT = 128, 64

# Substring patterns into the state dict.  Verified against
# out/dense_d8_V32k_model_001014.pt, which has exactly 60 keys.
COMPONENTS = {
    "attn.qkv":     [".attn.c_q.", ".attn.c_k.", ".attn.c_v."],
    "attn.c_proj":  [".attn.c_proj."],
    "attn.ve_gate": [".attn.ve_gate."],
    "mlp.c_fc":     [".mlp.c_fc."],
    "mlp.c_proj":   [".mlp.c_proj."],
    "lm_head":      ["lm_head."],
    "wte":          ["transformer.wte."],
    "value_embeds": ["value_embeds."],
}
AGGREGATES = {
    "ALL body matmuls": [".attn.c_q.", ".attn.c_k.", ".attn.c_v.", ".attn.c_proj.",
                         ".mlp.c_fc.", ".mlp.c_proj."],
    "ALL interfaces":   ["transformer.wte.", "lm_head.", "value_embeds."],
    "EVERYTHING":       [".attn.c_q.", ".attn.c_k.", ".attn.c_v.", ".attn.c_proj.",
                         ".mlp.c_fc.", ".mlp.c_proj.", ".attn.ve_gate.",
                         "transformer.wte.", "lm_head.", "value_embeds."],
}
# NOTE: this model has NO learned normalisation parameters.  nanochat's `norm` is a
# parameterless F.rms_norm (gpt.py:911), so there is no "binarise the norms" arm and
# section 3.3 of the plan is about removing an OPERATION, not parameters.


def build_config(depth, vocab_size, seq_len, wp):
    base = depth * ASPECT
    n_embd = ((base + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    n_head = n_embd // HEAD_DIM
    return GPTConfig(sequence_len=seq_len, vocab_size=vocab_size, n_layer=depth,
                     n_head=n_head, n_kv_head=n_head, n_embd=n_embd, window_pattern=wp)


def binarise_(t: torch.Tensor, scale: str) -> torch.Tensor:
    """sign(W) times the L1-optimal per-row scalar (XNOR-Net), or bare sign."""
    s = torch.sign(t)
    s[s == 0] = 1.0
    if scale == "row":
        s = s * t.abs().mean(dim=-1, keepdim=True)
    return s.to(t.dtype)


def _sign(t, scale, dim=-1):
    s = torch.sign(t)
    s = torch.where(s == 0, torch.ones_like(s), s)
    if scale == "row":
        s = s * t.abs().mean(dim=dim, keepdim=True)
    return s.to(t.dtype)


def install_act_hooks(model, patterns, scale):
    """Binarise the INPUT to every matched Linear: the other matmul operand."""
    handles = []

    def hook(mod, args):
        if not args:
            return None
        x = args[0]
        return (_sign(x, scale),) + tuple(args[1:])

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and any(p.strip(".") in name for p in patterns):
            handles.append(mod.register_forward_pre_hook(hook))
    return handles


def apply_binarisation(model, base_sd, patterns, scale):
    sd = {}
    touched = 0
    for k, v in base_sd.items():
        if v.dim() >= 2 and any(p in k for p in patterns):
            sd[k] = binarise_(v, scale)
            touched += v.numel()
        else:
            sd[k] = v
    model.load_state_dict(sd, strict=True)
    return touched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/dense_d8_V32k_model_001014.pt")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--window-pattern", default="SSSL")
    ap.add_argument("--tokenizer-dir", default="tokenizer")
    ap.add_argument("--data-dir", default=None,
                    help="parquet shard dir; val split is the LAST shard. Defaults to ./data "
                         "when that exists, because the ~/.cache/nanochat fallback may hold "
                         "only dummy_val.parquet")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--eval-steps", type=int, default=20)
    ap.add_argument("--scale", nargs="+", default=["row", "none"])
    ap.add_argument("--binarise", nargs="+", default=["weights", "acts", "both"],
                    choices=["weights", "acts", "both"],
                    help="which matmul operand(s) to binarise; 'both' is the only arm "
                         "matching the plan's definition")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these component names")
    ap.add_argument("--direction", default="add", choices=["add", "remove", "both-ways"],
                    help="add: dense baseline, binarise one component (overstates any "
                         "component that receives full-precision input). remove: fully "
                         "binary baseline, RESTORE one component to bf16, which measures "
                         "what that component's binarisation actually costs IN SITU. The "
                         "interfaces cost +0.90 measured by 'add' and +0.16 marginal once "
                         "the body is already binary, so 'remove' is the one that should "
                         "order the Phase 1 ladder.")
    ap.add_argument("--calibrate", action="store_true", default=True,
                    help="fit ONE scalar gain on the head output per arm before scoring. "
                         "Without it the scan measures logit-scale drift, not information "
                         "loss: binarising the head shrinks logits ~2.5x, binarising the body "
                         "inflates activations ~3.5x, and doing both partially cancels, which "
                         "makes a union look CHEAPER than its parts. A from-scratch binary "
                         "model learns its own output scale, so that drift is not a cost of "
                         "binarisation and must not be charged to it.")
    ap.add_argument("--no-calibrate", dest="calibrate", action="store_false")
    ap.add_argument("--diagnose", action="store_true",
                    help="also report PRE-softcap logit statistics per arm. The forward path "
                         "applies logits = 20*tanh(logits/20) (gpt.py:11656), which is a "
                         "nonlinearity between the head and the loss and can make a union of "
                         "components look CHEAPER than its parts.")
    a = ap.parse_args()

    # Prefer the repo's own data/ symlink over the cache fallback: a fresh cache can
    # contain nothing but dummy_val.parquet, which is 100 rows of "hello world this is
    # a dummy dataset" and silently produces meaningless bpb.
    if a.data_dir is None and os.path.isdir("data"):
        import glob as _g
        if _g.glob(os.path.join("data", "*.parquet")):
            a.data_dir = "data"
            print("[o5] --data-dir defaulted to ./data")

    # Bootstrap data and tokenizer exactly as the other sweeps do, so a fresh
    # clone with no ~/.cache/nanochat does not die inside the dataloader with
    # "No dataset parquet files found".  Downloads 2 shards if none are present;
    # 2 is the minimum, because the val split is parquet_paths[-1:].
    try:
        from scripts._sweep_utils import check_and_prepare_env
        check_and_prepare_env(a, label="o5")
    except Exception as e:
        print(f"[o5] data bootstrap skipped ({type(e).__name__}: {e})")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = build_config(a.depth, a.vocab, a.seq, a.window_pattern)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=dev)
    base_sd = torch.load(a.ckpt, map_location=dev, weights_only=False)
    base_sd = base_sd.get("model", base_sd) if isinstance(base_sd, dict) and "model" in base_sd else base_sd
    model.load_state_dict(base_sd, strict=True)
    model.eval()

    tok = get_tokenizer(a.tokenizer_dir) if a.tokenizer_dir else get_tokenizer()
    token_bytes = get_token_bytes(device=dev, tokenizer_dir=a.tokenizer_dir)

    def val_batches():
        return tokenizing_distributed_data_loader_bos_bestfit(
            tok, a.batch, a.seq, split="val", device=dev, data_dir=a.data_dir,
            max_shards=a.max_shards)

    # Refuse to report a number measured on the placeholder corpus.
    from nanochat.dataset import list_parquet_files
    _shards = list_parquet_files(data_dir=a.data_dir, max_shards=a.max_shards)
    _val = _shards[-1] if _shards else None
    print(f"[o5] {len(_shards)} shard(s); val split = {_val}")
    if _val and "dummy" in os.path.basename(_val).lower():
        raise SystemExit(
            "REFUSING TO RUN: the val shard is " + os.path.basename(_val) + ", the placeholder "
            "corpus (100 rows of repeated 'hello world this is a dummy dataset'). Any bpb from "
            "it is meaningless. Point --data-dir at real shards, or run "
            "`python -m nanochat.dataset -n 2` to download some.")

    # A mutable gain on the head output, plus pre-softcap logit statistics.
    _stats = {}
    _gain = [1.0]

    def _gain_hook(mod, args, out):
        return out if _gain[0] == 1.0 else out * _gain[0]

    model.lm_head.register_forward_hook(_gain_hook)

    def _logit_hook(mod, args, out):
        z = out.detach().float()
        z = z[..., :cfg.vocab_size]
        _stats["rms"] = float(z.pow(2).mean().sqrt())
        _stats["absmax"] = float(z.abs().max())
        # tanh(x/20) is >0.96 saturated by |x|=40; count how much mass is past that
        _stats["frac_sat"] = float((z.abs() > 40.0).float().mean())
        _stats["mean_abs"] = float(z.abs().mean())

    def score():
        bpb, _ = evaluate_bpb(model, val_batches(), a.eval_steps, token_bytes)
        return float(bpb)

    # One calibration batch, held out from scoring by construction: it is the FIRST
    # batch of a fresh iterator, and scoring always builds its own fresh iterator.
    _cal_x, _cal_y = next(iter(tokenizing_distributed_data_loader_bos_bestfit(
        tok, 1, min(512, a.seq), split="val", device=dev, data_dir=a.data_dir,
        max_shards=a.max_shards)))

    def fit_gain():
        """Fit one scalar on the head output by minimising CE on the calibration batch.

        Captures pre-softcap logits once, then optimises offline, so this costs one
        extra forward per arm rather than one eval sweep per candidate value.
        """
        if not a.calibrate:
            return 1.0
        cap = {}

        def grab(mod, args, out):
            cap["z"] = out.detach().float()[..., :cfg.vocab_size]

        h = model.lm_head.register_forward_hook(grab)
        old, _gain[0] = _gain[0], 1.0
        with torch.no_grad():
            model(_cal_x, targets=_cal_y, loss_reduction="mean")
        h.remove()
        _gain[0] = old
        z = cap["z"].reshape(-1, cfg.vocab_size)
        y = _cal_y.reshape(-1)
        keep = y >= 0
        z, y = z[keep], y[keep]
        logg = torch.zeros((), device=z.device, requires_grad=True)
        opt = torch.optim.LBFGS([logg], lr=0.5, max_iter=40)

        def closure():
            opt.zero_grad()
            zz = 20.0 * torch.tanh(z * logg.exp() / 20.0)
            loss = torch.nn.functional.cross_entropy(zz, y)
            loss.backward()
            return loss

        opt.step(closure)
        g = float(logg.detach().exp())
        del cap, z
        torch.cuda.empty_cache()
        return max(min(g, 1e3), 1e-3)

    _hook_handle = model.lm_head.register_forward_hook(_logit_hook) if a.diagnose else None

    t0 = time.time()
    _gain[0] = fit_gain()
    base_gain = _gain[0]
    baseline = score()
    base_stats = dict(_stats)
    print(f"baseline dense bpb = {baseline:.4f}   ({time.time()-t0:.0f}s per eval, "
          f"{a.eval_steps} steps x {a.batch} x {a.seq} tokens)")
    print(f"calibration: {'ON' if a.calibrate else 'OFF'}"
          + (f", baseline fitted gain = {base_gain:.3f}" if a.calibrate else ""))
    print()

    items = list(COMPONENTS.items()) + list(AGGREGATES.items())
    if a.only:
        items = [(k, v) for k, v in items if k in a.only]

    ALL_PATS = AGGREGATES["EVERYTHING"]
    if a.direction in ("remove", "both-ways"):
        # Re-reference: the baseline becomes the FULLY BINARY model, and each arm
        # restores one component to bf16.
        print()
        print("=" * 70)
        print("LEAVE-ONE-OUT FROM THE FULLY BINARY MODEL")
        print("=" * 70)
        for sc in a.scale:
            apply_binarisation(model, base_sd, ALL_PATS, sc)
            handles = install_act_hooks(model, ALL_PATS, sc)
            _gain[0] = fit_gain()
            all_bin = score()
            for h in handles:
                h.remove()
            print(f"fully binary baseline (scale={sc}): bpb {all_bin:.4f} "
                  f"({all_bin - baseline:+.4f} vs dense)")
            print(f"{'restored to bf16':<20}{'params':>13}{'bpb':>10}{'in-situ cost':>14}")
            print("-" * 57)
            loo = []
            for name, pats in items:
                if name in AGGREGATES:
                    continue
                keep = [q for q in ALL_PATS if q not in pats]
                apply_binarisation(model, base_sd, keep, sc)
                handles = install_act_hooks(model, keep, sc)
                _gain[0] = fit_gain()
                b = score()
                for h in handles:
                    h.remove()
                n = sum(v.numel() for k, v in base_sd.items()
                        if v.dim() >= 2 and any(q in k for q in pats))
                loo.append((name, n, b, all_bin - b))
                print(f"{name:<20}{n:>13,d}{b:>10.4f}{all_bin - b:>+14.4f}")
            print("-" * 57)
            print("in-situ cost = what binarising that component costs INSIDE a fully binary")
            print("model. This is the number that orders the Phase 1 ladder, not the 'add'")
            print("column, which charges a component for receiving bf16 input it will never see.")
            loo.sort(key=lambda r: -r[3])
            print()
            print(f"ladder order (most expensive to binarise last), scale={sc}:")
            for name, n, b, d in loo:
                print(f"  {d:+8.4f}  {name:<20} ({n:,} params)")
        model.load_state_dict(base_sd, strict=True)
        _gain[0] = base_gain
        if a.direction == "remove":
            return

    combos = [(m, s) for m in a.binarise for s in a.scale]
    hdr = f"{'component':<20}{'params':>13}"
    for m, s in combos:
        hdr += f"{m[:4]+'/'+s[:4]:>13}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    diag = []
    for name, pats in items:
        line = ""
        deltas = {}
        n = 0
        for mode, sc in combos:
            handles = []
            if mode in ("weights", "both"):
                n = apply_binarisation(model, base_sd, pats, sc)
            else:
                model.load_state_dict(base_sd, strict=True)
                n = sum(v.numel() for k, v in base_sd.items()
                        if v.dim() >= 2 and any(p in k for p in pats))
            if mode in ("acts", "both"):
                handles = install_act_hooks(model, pats, sc)
            _gain[0] = fit_gain()
            arm_gain = _gain[0]
            b = score()
            arm_stats = dict(_stats)
            arm_stats["gain"] = arm_gain
            for h in handles:
                h.remove()
            deltas[(mode, sc)] = b - baseline
            line += f"{b-baseline:+13.4f}"
            if a.diagnose:
                diag.append((name, mode, sc, b - baseline, arm_stats))
        rows.append((name, n, deltas))
        print(f"{name:<20}{n:13,d}{line}")
    model.load_state_dict(base_sd, strict=True)

    if a.diagnose:
        print()
        print("PRE-SOFTCAP LOGIT STATISTICS  (softcap=20, so tanh is ~saturated past |z|=40)")
        print(f"{'arm':<20}{'mode':>9}{'scale':>7}{'d bpb':>10}{'rms':>10}{'mean|z|':>10}"
              f"{'max|z|':>11}{'frac>40':>10}{'gain':>9}")
        print("-" * 87)
        print(f"{'BASELINE dense':<20}{'-':>9}{'-':>7}{0.0:>10.4f}{base_stats.get('rms',0):>10.2f}"
              f"{base_stats.get('mean_abs',0):>10.2f}{base_stats.get('absmax',0):>11.1f}"
              f"{base_stats.get('frac_sat',0):>10.4f}{base_gain:>9.3f}")
        for name, mode, sc, d, st in diag:
            print(f"{name:<20}{mode:>9}{sc:>7}{d:>10.4f}{st.get('rms',0):>10.2f}"
                  f"{st.get('mean_abs',0):>10.2f}{st.get('absmax',0):>11.1f}"
                  f"{st.get('frac_sat',0):>10.4f}{st.get('gain',1.0):>9.3f}")
        print("-" * 87)
        print("If a UNION shows lower frac>40 than its parts, the softcap is compensating and")
        print("the bpb deltas are not additive. That is a property of the measurement, not of")
        print("binarisation, and the scan must be read with it or run with the cap disabled.")

    # A union of components must cost at least as much as any of its parts. When it
    # does not, the deltas are not composable and the ranking cannot be read as one.
    SUBSETS = {
        "ALL body matmuls": ["attn.qkv", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
        "ALL interfaces": ["lm_head", "wte", "value_embeds"],
        "EVERYTHING": ["ALL body matmuls", "ALL interfaces", "lm_head", "mlp.c_proj"],
    }
    by_name = {n: d for n, _, d in rows}
    violations = []
    for union, parts in SUBSETS.items():
        if union not in by_name:
            continue
        for combo in combos:
            u = by_name[union].get(combo)
            if u is None:
                continue
            for part in parts:
                if part in by_name and by_name[part].get(combo) is not None:
                    v = by_name[part][combo]
                    if v > u + 1e-9:
                        violations.append((union, part, combo, u, v))
    print()
    if violations:
        print("MONOTONICITY VIOLATIONS: a union costs LESS than one of its parts.")
        print(f"{'union':<20}{'part':<20}{'mode/scale':<16}{'union':>10}{'part':>10}")
        for union, part, combo, u, v in violations:
            print(f"{union:<20}{part:<20}{combo[0]+'/'+combo[1]:<16}{u:>10.4f}{v:>10.4f}")
        print("Deltas are NOT composable. Two causes, and only the first is a measurement")
        print("artifact: (1) logit-scale drift, removed by --calibrate; (2) sign() is")
        print("idempotent-ish, so binarising an input that an upstream binarised layer has")
        print("already coarsened costs less than binarising a full-precision one. (2) is a")
        print("real property of the architecture and means single-component deltas are an")
        print("UPPER BOUND on their marginal cost inside a fully binary model.")
    else:
        print("monotonicity: OK, every union costs at least as much as each of its parts")

    print()
    key = ("both", a.scale[0]) if "both" in a.binarise else combos[0]
    print(f"sensitivity ranking by delta bpb, mode={key[0]} scale={key[1]}, worst first:")
    for name, n, d in sorted(rows, key=lambda r: -r[2][key]):
        print(f"  {d[key]:+8.4f}  {name:<20} ({n:,} params)")
    if "weights" in a.binarise and "both" in a.binarise:
        print()
        print("weights-only vs both: how much of the damage is the ACTIVATIONS")
        for name, n, d in rows:
            w, bo = d[("weights", a.scale[0])], d[("both", a.scale[0])]
            share = (bo - w) / bo * 100 if abs(bo) > 1e-9 else float("nan")
            print(f"  {name:<20} W1A16 {w:+8.4f}   W1A1 {bo:+8.4f}   activations are {share:5.1f}% of it")
    print()
    print("W1A16 IS NOT BINARISATION. It changes no operation: still a bf16 GEMM over a")
    print("matrix holding two values per row. Only the 'both' column corresponds to the")
    print("plan's claim, and only --scale none is strict; a per-row float scale is a float.")
    print()
    print("READ THIS BEFORE USING THE NUMBERS: projection PENALISES. A component that")
    print("survives here is strong evidence it survives from-scratch training; one that")
    print("fails here is weak evidence, because a model trained binary shapes its own")
    print("weight distribution for sign() and this one never did.")


if __name__ == "__main__":
    main()
