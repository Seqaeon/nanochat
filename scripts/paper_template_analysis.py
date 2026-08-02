"""Template specialization analysis for a trained RemixedLinear checkpoint.

R1, R2 and R3 all asked for this and the metareview lists it. The question behind
all three is the same: **does routing do anything?** If alpha is effectively
constant, the gain comes from the extra parameters and the intermediate
LayerNorm, not from conditioning, and that changes what the paper claims.

The report answers that in five parts, per projection and per layer:

  1. Router identity      which router class actually ran, and with what topk
  2. Utilization          usage histogram, load balance, collapse detection
  3. Where alpha varies   variance decomposition: within-sequence (position)
                          vs across-sequence (input). This is the decisive one.
  4. Effective weight     how far W_eff(chunk) actually moves from its mean,
                          in relative Frobenius norm
  5. Bank geometry        pairwise cosine between templates. Routing between
                          templates that have converged to the same matrix is
                          decorative regardless of how varied alpha looks.

(3) and (4) are the two numbers to look at first. Item 4 is computed exactly and
cheaply without ever materialising W_eff, via the bank Gram matrix
G[k,j] = <T_k, T_j>:

    || sum_k c_k T_k ||_F^2 = c^T G c

so a full distribution over chunks costs one KxK matrix per module.

    python -m scripts.paper_template_analysis --ckpt out/.../base_checkpoints/xxx
    python -m scripts.paper_template_analysis --ckpt DIR --batches 20 --plot out/tmpl
    python -m scripts.paper_template_analysis --ckpt DIR --synthetic   # smoke test
"""
import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanochat.gpt import GPT, GPTConfig, RemixedLinear
from nanochat.checkpoint_manager import load_checkpoint, _patch_missing_config_keys
from nanochat.tokenizer import get_tokenizer
from scripts.paper_lib import find_last_step, build_val_batches


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def _stack_legacy_templates(state, model):
    """Rebuild `X.template_bank` from a checkpoint that stored templates separately.

    The template bank used to be K separate 2-D parameters per projection; P29
    stacked them into one (K, out, basis) tensor so Muon could run block-diagonal
    Newton-Schulz over it. Checkpoints written before that stacking load into a
    current model with every `template_bank` key missing and nothing else wrong,
    which is exactly the failure this repairs.

    Returns (patched_state, notes). Anything it cannot repair is left alone so the
    caller can report it.
    """
    notes = []
    want = {k for k in model.state_dict() if k.endswith(".template_bank")}
    missing = sorted(want - set(state))
    if not missing:
        return state, notes
    state = dict(state)
    for key in missing:
        prefix = key[: -len("template_bank")]
        target = model.state_dict()[key]
        K, per_template = target.shape[0], target.shape[1:]
        # Per-template keys under this module, in index order. The shape filter
        # matters: template_route also lives here, is 2-D, and has "template" in
        # its name, so a name match alone picks up K+1 candidates and repairs
        # nothing.
        cands = sorted(
            (k for k in state
             if k.startswith(prefix)
             and re.fullmatch(r"template[_.]?\d+(\.weight)?", k[len(prefix):])
             and tuple(state[k].shape) == tuple(per_template)),
            key=lambda k: int(re.findall(r"\d+", k[len(prefix):])[0]),
        )
        if len(cands) == K:
            state[key] = torch.stack([state.pop(k) for k in cands], dim=0)
            notes.append(f"stacked {K} legacy tensors -> {key}")
        elif len(cands) == 1 and state[cands[0]].shape == model.state_dict()[key].shape[1:]:
            # Single shared matrix (a K=1 checkpoint being read as K>1).
            notes.append(f"FOUND ONLY ONE template matrix for {key}: {cands[0]}")
        else:
            notes.append(f"CANNOT REPAIR {key}: candidates under this module = "
                         + (", ".join(k[len(prefix):] for k in cands) or "(none)"))
    return state, notes


def load_remix_model(ckpt_dir, device, step=None, tokenizer_dir=None):
    """Load a RemixedLinear checkpoint, repairing known key-layout changes.

    Deliberately not paper_lib.load_any_model: that loads strictly and reports only
    the first missing key, which turns a mechanical layout change into an opaque
    failure. Here a mismatch prints what the checkpoint actually contains.
    """
    step = find_last_step(ckpt_dir) if step is None else step
    device = torch.device(device) if isinstance(device, str) else device
    state, _, meta = load_checkpoint(ckpt_dir, step, device, load_optimizer=False)
    state = {k.removeprefix("_orig_mod.").removeprefix("module."): v for k, v in state.items()}
    if device.type in {"cpu", "mps"}:
        state = {k: (v.float() if v.dtype == torch.bfloat16 else v) for k, v in state.items()}

    cfg_kwargs = dict(meta["model_config"])
    _patch_missing_config_keys(cfg_kwargs)
    config = GPTConfig(**cfg_kwargs)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()

    state, notes = _stack_legacy_templates(state, model)
    for n in notes:
        print(f"[analysis] {n}")
    try:
        model.load_state_dict(state, strict=True, assign=True)
    except RuntimeError as e:
        want, have = set(model.state_dict()), set(state)
        miss, extra = sorted(want - have), sorted(have - want)
        print(f"\n[analysis] state_dict mismatch: {len(miss)} missing, {len(extra)} unexpected")
        print(f"[analysis] missing (first 8): {miss[:8]}")
        print(f"[analysis] unexpected (first 8): {extra[:8]}")
        if miss:
            pre = miss[0].rsplit(".", 1)[0] + "."
            print(f"[analysis] everything the checkpoint has under {pre}:")
            for k in sorted(k for k in state if k.startswith(pre)):
                print(f"             {k[len(pre):]:32s} {tuple(state[k].shape)}")
        raise SystemExit(
            "Refusing to analyse a partially-loaded model. The listing above shows "
            "what the checkpoint stores where the current code expects template_bank; "
            "if those are the templates under another name, the remap in "
            "_stack_legacy_templates needs one more case."
        ) from e
    model.eval()
    print(f"[analysis] loaded {ckpt_dir} step {step}: L={config.n_layer} D={config.n_embd} "
          f"K={config.remixed_linear_kwargs.get('n_templates', 1)} "
          f"chunk={getattr(config, 'p28_chunk_routing_size', 0)} "
          f"quantile_route={getattr(config, 'p23_quantile_route', 0)}")
    return model, get_tokenizer(tokenizer_dir=tokenizer_dir), config, meta


# ---------------------------------------------------------------------------
# Routing capture
# ---------------------------------------------------------------------------
def routing_weights(mod, x):
    """The routing weights this module would use for input x, shape (B, N, K).

    Mirrors RemixedLinear.forward rather than calling it, so the analysis sees
    the same numbers the model computes. Two paths, and they are NOT equivalent:

      chunk_routing_size > 0
          anchor = first token of each chunk, then _template_weights(), which is
          exactly what the compose branch does.

      chunk_routing_size == 0 (legacy per-token)
          the inline branch feeds the router output through *another* softmax.
          With a quantile router that output is already a normalised weight
          vector, so the second softmax flattens it. We reproduce that here
          rather than 'fix' it, because reproducing it is the point: it means a
          chunk-vs-per-token ablation compared two different routing functions.
          `double_softmax` in the output marks the modules where it applies.
    """
    K = mod.n_templates
    chunk = int(getattr(mod, "chunk_routing_size", 0) or 0)
    qrouter = getattr(mod, "_qrouter", None)
    if chunk > 0:
        B, T, C = x.shape
        n = (T + chunk - 1) // chunk
        pad = n * chunk - T
        xp = F.pad(x, (0, 0, 0, pad)) if pad > 0 else x
        sig = xp.reshape(B, n, chunk, C)[:, :, 0, :].float()
        return mod._template_weights(sig, torch.float32).float(), False
    # legacy per-token branch
    if qrouter is not None:
        route_logits = qrouter(x)
    elif getattr(mod, "template_route", None) is not None:
        route_logits = x.float() @ mod.template_route.float()
    else:
        return torch.full(x.shape[:-1] + (K,), 1.0 / K, device=x.device), False
    topk = int(getattr(mod, "template_topk", 0) or 0)
    if 0 < topk < K:
        vals, idx = route_logits.topk(topk, dim=-1)
        w = (F.one_hot(idx, K).float() * F.softmax(vals.float(), -1).unsqueeze(-1)).sum(-2)
        return w, False
    return F.softmax(route_logits.float(), -1), qrouter is not None


class Capture:
    """Forward-pre-hooks that record routing weights and chunk anchor tokens."""

    def __init__(self, model):
        self.mods = {}
        self.alpha = defaultdict(list)   # name -> list of (B, N, K) cpu tensors
        self.handles = []
        for name, m in model.named_modules():
            if isinstance(m, RemixedLinear) and m.n_templates > 1 \
                    and getattr(m, "template_bank", None) is not None:
                self.mods[name] = m
                self.handles.append(
                    m.register_forward_pre_hook(self._make_hook(name)))
        if not self.mods:
            raise SystemExit(
                "No RemixedLinear modules with a template bank found. This "
                "checkpoint is either dense, K=1, or uses the tiny-expert / "
                "LoKR / global-bank variant, none of which this script covers.")

    def _make_hook(self, name):
        def hook(mod, args):
            x = args[0]
            with torch.no_grad():
                w, dbl = routing_weights(mod, x)
            self.alpha[name].append(w.detach().float().cpu())
            mod._analysis_double_softmax = dbl
            return None
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


# ---------------------------------------------------------------------------
# Per-module statistics
# ---------------------------------------------------------------------------
def bank_gram(mod):
    """G[k,j] = <T_k, T_j> over the flattened templates, float64 for stability."""
    T = mod.template_bank.detach().float().reshape(mod.n_templates, -1).double()
    return T @ T.T


def analyse_module(name, mod, alphas, gram):
    """alphas: (S, N, K) — S sequences (concatenated over batches), N routing units."""
    a = torch.cat(alphas, dim=0).double()       # (S, N, K)
    S, N, K = a.shape
    mean = a.reshape(-1, K).mean(0)             # (K,)

    # --- utilization -------------------------------------------------------
    hard = a.argmax(-1).reshape(-1)
    usage = torch.bincount(hard, minlength=K).double() / hard.numel()
    ent = -(a.clamp_min(1e-12).log() * a).sum(-1)              # (S, N)
    n_eff = float(torch.exp(ent.mean()))                        # perplexity of alpha

    # --- variance decomposition -------------------------------------------
    # total = across-sequence (input dependence) + within-sequence (position
    # dependence). A model whose routing ignores position has within == 0.
    per_seq = a.mean(1)                                         # (S, K)
    var_total = a.reshape(-1, K).var(0, unbiased=False).sum()
    var_across = per_seq.var(0, unbiased=False).sum() if S > 1 else torch.tensor(0.)
    var_within = (a - per_seq[:, None, :]).reshape(-1, K).var(0, unbiased=False).sum()

    # --- effective-weight deviation ---------------------------------------
    # ||W_eff(c) - Wbar||_F / ||Wbar||_F, exactly, via the bank Gram matrix.
    dev = (a.reshape(-1, K) - mean)                             # (S*N, K)
    num = torch.einsum("ik,kj,ij->i", dev, gram, dev).clamp_min(0).sqrt()
    den = float((mean @ gram @ mean).clamp_min(1e-30).sqrt())
    rel_dev = num / den

    # --- bank geometry -----------------------------------------------------
    d = gram.diagonal().clamp_min(1e-30).sqrt()
    cos = gram / torch.outer(d, d)
    off = cos[~torch.eye(K, dtype=torch.bool)]
    Tb = mod.template_bank.detach().float().reshape(K, -1).double()
    Tbar = Tb.mean(0)
    spread = float((Tb - Tbar).norm(dim=1).mean() / Tbar.norm().clamp_min(1e-30))

    return dict(
        name=name, K=K, n_seq=S, n_units=N,
        chunk=int(getattr(mod, "chunk_routing_size", 0) or 0),
        topk=int(getattr(mod, "template_topk", 0) or 0),
        router=type(getattr(mod, "_qrouter", None)).__name__
               if getattr(mod, "_qrouter", None) is not None else "learned_linear",
        double_softmax=bool(getattr(mod, "_analysis_double_softmax", False)),
        route_side=getattr(mod, "route_side", "output"),
        bank_shape=list(mod.template_bank.shape),
        mean_alpha=[round(v, 5) for v in mean.tolist()],
        usage=[round(v, 5) for v in usage.tolist()],
        usage_cv=float(usage.std(unbiased=False) / usage.mean().clamp_min(1e-12)),
        usage_max=float(usage.max()), usage_min=float(usage.min()),
        n_templates_used=int((usage > 0.01).sum()),
        entropy=float(ent.mean()), entropy_norm=float(ent.mean() / math.log(K)),
        n_eff_templates=n_eff,
        var_total=float(var_total),
        var_within_seq=float(var_within), var_across_seq=float(var_across),
        frac_var_within_seq=float(var_within / var_total.clamp_min(1e-30)),
        weff_rel_dev_mean=float(rel_dev.mean()),
        weff_rel_dev_p95=float(rel_dev.quantile(0.95)),
        bank_cos_mean=float(off.mean()), bank_cos_max=float(off.max()),
        bank_spread=spread,
    )


def proj_kind(name):
    """attn.c_q -> 'attn.c_q'; strips the transformer.h.<i>. prefix."""
    m = re.search(r"transformer\.h\.(\d+)\.(.+)$", name)
    return (int(m.group(1)), m.group(2)) if m else (-1, name)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report(stats, fh=sys.stdout):
    p = lambda *a: print(*a, file=fh)
    by_kind = defaultdict(list)
    for s in stats:
        by_kind[proj_kind(s["name"])[1]].append(s)

    r0 = stats[0]
    p("\n" + "=" * 96)
    p("ROUTER IDENTITY  (what actually ran, not what the flags suggest)")
    p("=" * 96)
    p(f"  router class        : {r0['router']}")
    p(f"  template_topk flag  : {r0['topk']}   (0 conventionally means 'soft over all K')")
    p(f"  chunk_routing_size  : {r0['chunk']}")
    p(f"  K                   : {r0['K']}")
    p(f"  route_side          : {r0['route_side']}")
    if r0["double_softmax"]:
        p("  !! per-token branch re-softmaxes an already-normalised router output")
    hard = [s for s in stats if s["n_eff_templates"] < 1.05]
    if hard:
        p(f"  !! {len(hard)}/{len(stats)} modules route to a SINGLE template "
          f"(effective count < 1.05) despite topk={r0['topk']}")
    flat = [s for s in stats if s["frac_var_within_seq"] < 1e-6]
    if flat:
        p(f"  !! {len(flat)}/{len(stats)} modules have ZERO within-sequence variance:")
        p("     alpha is constant across all chunks of a sequence, so chunk-amortised")
        p("     routing is numerically identical to per-sequence routing in these modules.")

    p("\n" + "=" * 96)
    p("PER PROJECTION (mean over layers)")
    p("=" * 96)
    p(f"  {'projection':16s} {'H/logK':>7s} {'K_eff':>6s} {'used':>5s} {'usage CV':>9s} "
      f"{'var within':>11s} {'var across':>11s} {'|dW|/|W|':>9s} {'bank cos':>9s} {'spread':>7s}")
    for kind, ss in sorted(by_kind.items()):
        avg = lambda k: sum(s[k] for s in ss) / len(ss)
        p(f"  {kind:16s} {avg('entropy_norm'):7.4f} {avg('n_eff_templates'):6.3f} "
          f"{avg('n_templates_used'):5.2f} {avg('usage_cv'):9.4f} "
          f"{avg('var_within_seq'):11.3e} {avg('var_across_seq'):11.3e} "
          f"{avg('weff_rel_dev_mean'):9.4f} {avg('bank_cos_mean'):9.4f} {avg('bank_spread'):7.4f}")

    p("\n" + "=" * 96)
    p("PER LAYER (mean over projections)")
    p("=" * 96)
    by_layer = defaultdict(list)
    for s in stats:
        by_layer[proj_kind(s["name"])[0]].append(s)
    p(f"  {'layer':>5s} {'H/logK':>7s} {'K_eff':>6s} {'usage CV':>9s} {'|dW|/|W|':>9s} {'bank cos':>9s}")
    for li in sorted(by_layer):
        ss = by_layer[li]
        avg = lambda k: sum(s[k] for s in ss) / len(ss)
        p(f"  {li:5d} {avg('entropy_norm'):7.4f} {avg('n_eff_templates'):6.3f} "
          f"{avg('usage_cv'):9.4f} {avg('weff_rel_dev_mean'):9.4f} {avg('bank_cos_mean'):9.4f}")

    p("\n" + "=" * 96)
    p("HOW TO READ THIS")
    p("=" * 96)
    p("  H/logK      1.0 = uniform mixing, 0.0 = hard one-template routing.")
    p("  K_eff       exp(H): how many templates a routing decision effectively blends.")
    p("  usage CV    0.0 = perfectly balanced load; sqrt(K-1) = total collapse to one.")
    p("  var within  variance of alpha across chunks *inside* a sequence. If this is 0,")
    p("              routing is position-independent and chunk size cannot matter.")
    p("  var across  variance of the per-sequence mean alpha. If this is also ~0, routing")
    p("              is input-independent and the layer is a fixed linear map.")
    p("  |dW|/|W|    relative Frobenius distance from W_eff(chunk) to its own mean.")
    p("              This is the honest 'how dynamic is the weight' number: <0.05 means")
    p("              the effective weight barely moves no matter what alpha looks like.")
    p("  bank cos    mean pairwise cosine between templates. Near 1.0 means the bank has")
    p("              collapsed and routing between its entries is close to a no-op.")


def make_plots(stats, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    by_kind = defaultdict(dict)
    for s in stats:
        li, kind = proj_kind(s["name"])
        by_kind[kind][li] = s
    kinds = sorted(by_kind)
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.1 * len(kinds), 3.6), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        layers = sorted(by_kind[kind])
        M = np.array([by_kind[kind][li]["usage"] for li in layers])
        im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=max(M.max(), 1e-9))
        ax.set_title(kind, fontsize=9)
        ax.set_xlabel("template"); ax.set_ylabel("layer")
        ax.set_yticks(range(len(layers))); ax.set_yticklabels(layers, fontsize=6)
    fig.colorbar(im, ax=axes[0].tolist(), label="usage fraction", shrink=.8)
    fig.suptitle("template utilization", fontsize=11)
    fig.savefig(base + "_usage.png", dpi=160, bbox_inches="tight")

    fig2, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for kind in kinds:
        layers = sorted(by_kind[kind])
        ax[0].plot(layers, [by_kind[kind][l]["entropy_norm"] for l in layers], marker="o", label=kind)
        ax[1].plot(layers, [by_kind[kind][l]["weff_rel_dev_mean"] for l in layers], marker="o", label=kind)
        ax[2].plot(layers, [by_kind[kind][l]["bank_cos_mean"] for l in layers], marker="o", label=kind)
    for a, t in zip(ax, ["routing entropy H/logK", "|W_eff - mean| / |mean|", "mean pairwise template cosine"]):
        a.set_xlabel("layer"); a.set_title(t, fontsize=10); a.grid(alpha=.3)
    ax[0].set_ylim(-0.02, 1.02); ax[0].legend(fontsize=7)
    fig2.tight_layout()
    fig2.savefig(base + "_specialization.png", dpi=160)
    print(f"wrote {base}_usage.png and {base}_specialization.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint directory (contains model_*.pt)")
    ap.add_argument("--step", type=int, default=None, help="default: last step in the dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--synthetic", action="store_true",
                    help="random token ids instead of val data. Utilization and bank "
                         "geometry stay meaningful; anything about *what* routing "
                         "keys on does not. Smoke tests only.")
    ap.add_argument("--train-mode", action="store_true",
                    help="run the router in train() mode. QuantileBalancedRouter takes a "
                         "different branch there (per-batch quantile thresholds instead "
                         "of plain top-k), so this is the routing the model was fit with.")
    ap.add_argument("--out", default="out/template_analysis.json")
    ap.add_argument("--plot", default=None, help="path prefix for the figures")
    args = ap.parse_args()

    model, tokenizer, config, meta = load_remix_model(
        args.ckpt, args.device, step=args.step, tokenizer_dir=args.tokenizer_dir)
    model.train(args.train_mode)
    cap = Capture(model)
    print(f"[analysis] {len(cap.mods)} RemixedLinear modules with a template bank")

    vocab = config.vocab_size
    if args.synthetic:
        batches = ((torch.randint(0, vocab, (args.batch_size, args.seq), device=args.device),
                    None) for _ in range(args.batches))
    else:
        batches = build_val_batches(tokenizer, args.batch_size, args.seq, args.device,
                                    data_dir=args.data_dir, max_shards=args.max_shards)

    with torch.no_grad():
        for i in range(args.batches):
            try:
                b = next(batches)
            except StopIteration:
                print(f"[analysis] data exhausted after {i} batches"); break
            x = b[0] if isinstance(b, (tuple, list)) else b
            model(x)
            print(f"  batch {i + 1}/{args.batches}", end="\r", flush=True)
    cap.close()
    print()

    stats = []
    for name, mod in cap.mods.items():
        if not cap.alpha[name]:
            continue
        stats.append(analyse_module(name, mod, cap.alpha[name], bank_gram(mod)))
    stats.sort(key=lambda s: proj_kind(s["name"]))

    report(stats)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(dict(ckpt=args.ckpt, step=args.step, train_mode=args.train_mode,
                   synthetic=args.synthetic, n_modules=len(stats), modules=stats),
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    with open(os.path.splitext(args.out)[0] + ".txt", "w") as f:
        report(stats, fh=f)
    if args.plot:
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        make_plots(stats, args.plot)


if __name__ == "__main__":
    main()
