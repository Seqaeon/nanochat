"""Measure every arm of a finished sweep in one pass.

Replaces running these by hand, once per arm, and then collating the output:

    python -m scripts.code_head_diagnostics --checkpoint-dir out/c05_sch_phase5/d4/<TAG>
    python -m scripts.code_head_subspace --checkpoint <dense arm>/model_XXXX.pt

Point it at the depth directory instead:

    python -m scripts.sweep_report out/c05_sch_phase5/d4

It walks every arm, runs the same diagnostics `base_train` runs at the end of a
run, measures each frozen head's subspace against the sweep's OWN dense arm, and
prints one table plus a CSV. The dense baseline is found automatically: it is the
arm whose config has no code head, and it is also the reference the capture
column is measured against, so nothing has to be named on the command line.

    --cached       harvest sch_metrics.json instead of recomputing (fast)
    --arms REGEX   restrict to matching arm tags
    --no-subspace  skip the capture column (it needs a converged dense arm)
    --out PATH     CSV destination (default <dir>/sweep_report.csv)

Capture is the fraction of the dense arm's logit energy that lies inside a
head's reachable subspace. It answers the question rank cannot: a head can have
ample rank and still point it in the wrong directions. Measured at V=32768 on
c00, a binary order-2 code captured 1.79% where the best possible subspace of
the same width captured 74.21%.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import traceback
from glob import glob

import torch

from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, print0
from nanochat.checkpoint_manager import load_model_from_dir
from nanochat.code_metrics import run_all_diagnostics
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.tokenizer import get_token_bytes

RANK_THRESHOLD = 1000        # Godey et al. (2024) empirical head-rank threshold


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_arms(sweep_dir: str, pattern: str = "") -> list[tuple[str, str]]:
    """Return [(tag, checkpoint_dir)] for every arm that produced a checkpoint.

    A sweep writes <dir>/<TAG>_s<seed>/depth_<D>/ckpt_base/base/model_*.pt.
    Arms that crashed leave the directory without the checkpoint, so keying on
    the checkpoint rather than the directory silently skips the failures instead
    of dying on the first one.
    """
    out = []
    rx = re.compile(pattern) if pattern else None
    for ckpt in sorted(glob(os.path.join(sweep_dir, "*", "depth_*", "ckpt_base"))):
        if not glob(os.path.join(ckpt, "*", "model_*.pt")):
            continue
        tag = os.path.basename(os.path.dirname(os.path.dirname(ckpt)))
        if rx and not rx.search(tag):
            continue
        out.append((tag, ckpt))
    return out


def cached_metrics(ckpt_dir: str) -> dict:
    """Whatever base_train already wrote for this arm, flattened, or {}.

    The file nests under "row" (the CSV line, which carries val_bpb and the cost
    columns) and "metrics" (the probe output). Both are wanted and neither alone
    is enough: val_bpb is the number the run actually trained to, while `bpb`
    only exists if the decile pass ran.
    """
    for path in (os.path.join(ckpt_dir, "base", "sch_metrics.json"),
                 os.path.join(ckpt_dir, "sch_metrics.json")):
        if not os.path.exists(path):
            continue
        with open(path) as f:
            blob = json.load(f)
        flat = {}
        for section in ("row", "metrics"):
            for k, v in (blob.get(section) or {}).items():
                if v is not None and not isinstance(v, (list, dict)):
                    flat[k] = v
        return flat or {k: v for k, v in blob.items()
                        if not isinstance(v, (list, dict))}
    return {}


def quality(row):
    """The bits-per-byte to rank on.

    `bpb` comes from the diagnostics pass and only exists when the decile
    metrics ran; `val_bpb` is what the run itself measured and is always there.
    Preferring the former keeps every arm on the same measurement when it is
    available, and falling back means an arm is never dropped from the table for
    a metric it was never asked to produce.
    """
    for k in ("bpb", "val_bpb", "min_val_bpb"):
        v = row.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


# ---------------------------------------------------------------------------
# Subspace capture
# ---------------------------------------------------------------------------

def read_user_config(ckpt_dir: str, step=None) -> dict:
    """The args the run was actually launched with, from its own meta_*.json.

    Two things depend on this and both bit on the first real use. The probes
    need the run's data and tokenizer directories, and defaulting them to None
    makes every arm without cached metrics die on "No dataset parquet files
    found". And the sch_* settings must be read from what RAN, not from a
    freshly constructed GPTConfig: a field added since the run is absent from
    the checkpoint, so the dataclass default silently fills in and the report
    describes the current code rather than the experiment.
    """
    metas = sorted(glob(os.path.join(ckpt_dir, "*", "meta_*.json")))
    if not metas:
        return {}
    path = metas[-1]
    if step is not None:
        want = [m for m in metas if f"{step:06d}" in os.path.basename(m)]
        if want:
            path = want[0]
    with open(path) as f:
        return (json.load(f) or {}).get("user_config", {}) or {}


def inherit(args, user_cfg):
    """CLI wins, then the run's own setting if it still resolves, then None."""
    out = {}
    for key in ("data_dir", "tokenizer_dir"):
        chosen = getattr(args, key, None)
        if chosen is None:
            candidate = user_cfg.get(key)
            if candidate and os.path.exists(candidate):
                chosen = candidate
        out[key] = chosen
    out["max_shards"] = (args.max_shards if args.max_shards is not None
                         else user_cfg.get("max_shards"))
    return out


def head_basis(model) -> torch.Tensor | None:
    """The (V, M) matrix whose column space the head can reach, or None.

    A dense or Monarch head learns its own subspace, so "capture" is 100% by
    construction and the question is not interesting for them. A tree head never
    forms a logit vector at all.
    """
    head = getattr(model, "lm_head", None)
    if head is None or not getattr(model.config, "use_code_head", False):
        return None
    if getattr(head, "custom_loss", False):                 # hierarchical softmax
        return None
    learned = getattr(head, "phi_learned", None)
    if learned is not None:
        return learned.detach().float().cpu()
    phi = getattr(head, "phi", None)
    if phi is not None and phi.numel():
        return (phi[0] if phi.dim() == 3 else phi).detach().float().cpu()
    assign = getattr(head, "assign", None)
    if assign is not None and assign.numel():
        # product code with the gather implementation: Phi is never materialised
        a = assign.detach().cpu().long()
        V, g = a.shape
        K = head.cfg["product_codebook"]
        out = torch.zeros(V, g * K)
        rows = torch.arange(V)
        for j in range(g):
            out[rows, j * K + a[:, j]] = 1.0
        return out
    return None


def capture_of(basis: torch.Tensor, Wc: torch.Tensor, total, Wr, total_res):
    """Fraction of the dense head's logit energy inside colspace(basis).

    Centred on the vocabulary axis, because a constant shift across the
    vocabulary is free under the softmax. Reported twice: against all of it, and
    against the residual after the single dominant direction, which a per-token
    bias supplies for nothing and which would otherwise flatter every arm.
    """
    P = basis.float()
    P = P - P.mean(dim=0, keepdim=True)
    G = (P.T @ P).double()
    A = (P.T @ Wc.float()).double()
    sol = torch.linalg.lstsq(G + 1e-9 * torch.eye(G.shape[0], dtype=torch.float64), A).solution
    cap = float((A * sol).sum() / total)
    Ar = (P.T @ Wr.float()).double()
    solr = torch.linalg.lstsq(G + 1e-9 * torch.eye(G.shape[0], dtype=torch.float64), Ar).solution
    return cap, float((Ar * solr).sum() / total_res)


# ---------------------------------------------------------------------------
# Per-arm measurement
# ---------------------------------------------------------------------------

def measure(tag, ckpt_dir, args, device, ref):
    user_cfg = read_user_config(ckpt_dir, args.step)
    env = inherit(args, user_cfg)
    model, tokenizer, meta = load_model_from_dir(
        ckpt_dir, device, phase="eval", model_tag="base",
        step=args.step, tokenizer_dir=env["tokenizer_dir"])
    cfg = model.config
    head = model.lm_head
    vocab_size = tokenizer.get_vocab_size()

    row = {"arm": tag, "depth": cfg.n_layer, "d": cfg.n_embd, "vocab": vocab_size}
    row["head"] = (getattr(cfg, "sch_head_type", "code")
                   if getattr(cfg, "use_code_head", False) else "dense")
    if getattr(cfg, "use_code_head", False):
        # From the launch args, not the rebuilt config: a flag added after the
        # run is missing from the checkpoint, and reading it off a fresh
        # GPTConfig would report the current default as though the run had used
        # it. "?" says "this run predates the flag", which is the truth.
        def ran(key, default="?"):
            return user_cfg.get(key, default)
        row["phi_mode"] = ran("sch_phi_mode")
        row["order"] = ran("sch_order")
        row["mixture"] = ran("sch_mixture")
        row["topk"] = ran("sch_mixture_topk")
        # Without this the FLOPs column is uninterpretable: two product arms at
        # the same M differ by up to 70x depending on whether Phi was
        # materialised and multiplied (4*V*M) or gathered (4*V*g). Same function
        # either way, so bpb stays comparable and cost does not.
        row["product_impl"] = ran("sch_product_impl") if row["phi_mode"] == "product" else ""
        row["bias"] = ran("sch_bias")
    row["M"] = getattr(head, "width", "")
    row["head_params"] = sum(p.numel() for p in head.parameters())
    if hasattr(head, "flops_per_token"):
        row["head_flops"] = head.flops_per_token()
    est = model.estimate_flops()
    row["flops_recomputed"] = float(est[0] if isinstance(est, tuple) else est)
    if hasattr(head, "rank_ceiling"):
        row["rank_ceiling"] = head.rank_ceiling()

    metrics = cached_metrics(ckpt_dir)
    if not args.cached or not metrics:
        token_bytes = get_token_bytes(device=device, tokenizer_dir=env["tokenizer_dir"])
        seq_len = args.seq_len or meta["model_config"]["sequence_len"]

        def build_val_loader():
            return tokenizing_distributed_data_loader_bos_bestfit(
                tokenizer, args.batch_size, seq_len, split="val", device=device,
                data_dir=env["data_dir"], max_shards=env["max_shards"])

        holdout = os.path.join(ckpt_dir, "sch_holdout_ids.pt")
        holdout_ids = torch.load(holdout, weights_only=True, map_location="cpu") \
            if os.path.exists(holdout) else None
        fresh = run_all_diagnostics(
            model, build_val_loader=build_val_loader, token_bytes=token_bytes,
            vocab_size=vocab_size, steps=args.steps, decile=not args.no_decile,
            rank_contexts=args.rank_contexts, holdout_ids=holdout_ids,
            tokenizer_dir=env["tokenizer_dir"], device=device)
        metrics = {**metrics, **fresh}
    row.update({k: v for k, v in metrics.items() if not isinstance(v, (list, dict))})
    # The recorded cost is what the run actually paid; the recomputed one is what
    # today's code would pay. A gap means the sweep straddles a code change, and
    # then the FLOPs column is not comparable across arms.
    if not isinstance(row.get("flops_per_token"), (int, float)):
        row["flops_per_token"] = row["flops_recomputed"]
    recorded, fresh = row["flops_per_token"], row["flops_recomputed"]
    row["flops_drift"] = fresh / recorded if recorded else 1.0

    if ref is not None:
        basis = head_basis(model)
        if basis is not None and basis.shape[0] == ref["Wc"].shape[0]:
            cap, cap_res = capture_of(basis, ref["Wc"], ref["total"], ref["Wr"], ref["total_res"])
            row["capture"], row["capture_residual"] = cap, cap_res
            M = basis.shape[1]
            row["capture_oracle"] = float(ref["cum"][min(M, len(ref["cum"])) - 1])
    del model
    return row


def build_reference(arms, args, device, no_subspace):
    """The sweep's own dense arm, used as the capture target.

    Refuses an under-trained one. On a 100-step checkpoint only two directions
    rise above the noise floor, every unfitted basis then scores at the random
    baseline M/V, and every fitted one scores high: it reads exactly like a
    result and is not one.
    """
    if no_subspace:
        return None
    from scripts.code_head_subspace import learned_direction_count
    for tag, ckpt in arms:
        try:
            tok_dir = inherit(args, read_user_config(ckpt, args.step))["tokenizer_dir"]
            model, _tok, _meta = load_model_from_dir(ckpt, device, phase="eval",
                                                     model_tag="base", step=args.step,
                                                     tokenizer_dir=tok_dir)
        except Exception:
            continue
        if getattr(model.config, "use_code_head", False) or not hasattr(model.lm_head, "weight"):
            del model
            continue
        W = model.lm_head.weight.detach().float().cpu()
        del model
        if not torch.isfinite(W).all():
            continue
        Wc = (W - W.mean(dim=0, keepdim=True)).double()
        U, S, _ = torch.linalg.svd(Wc, full_matrices=False)
        n_learned, share = learned_direction_count(S, *W.shape)
        print0(f"[report] capture reference: {tag}, {n_learned} of {W.shape[1]} directions "
               f"learned, carrying {share * 100:.1f}% of the energy")
        if n_learned < 8:
            print0("[report] that arm is too early to measure against; skipping capture")
            return None
        Wr = Wc - U[:, :1] @ (U[:, :1].T @ Wc)
        return {"tag": tag, "Wc": Wc, "total": (S ** 2).sum(),
                "Wr": Wr, "total_res": (Wr ** 2).sum(),
                "cum": torch.cumsum(S ** 2, 0) / (S ** 2).sum()}
    print0("[report] no dense arm found in this sweep; skipping the capture column")
    return None


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep_dir", help="e.g. out/c05_sch_phase5/d4")
    p.add_argument("--arms", default="", help="regex to restrict which arms run")
    p.add_argument("--cached", action="store_true",
                   help="harvest sch_metrics.json instead of recomputing the probes")
    p.add_argument("--no-subspace", action="store_true")
    p.add_argument("--no-decile", action="store_true")
    p.add_argument("--rank-contexts", type=int, default=16384, help="0 skips the rank probe")
    p.add_argument("--steps", type=int, default=100, help="validation batches per probe")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=0)
    p.add_argument("--step", type=int, default=None, help="checkpoint step (default: last)")
    p.add_argument("--tokenizer-dir", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--max-shards", type=int, default=None)
    p.add_argument("--device-type", default="")
    p.add_argument("--out", default="", help="CSV path (default <sweep_dir>/sweep_report.csv)")
    args = p.parse_args()

    arms = find_arms(args.sweep_dir, args.arms)
    if not arms:
        raise SystemExit(f"no completed arms under {args.sweep_dir}")
    print0(f"[report] {len(arms)} arm(s) under {args.sweep_dir}")
    probe = inherit(args, read_user_config(arms[0][1], args.step))
    print0(f"[report] data_dir={probe['data_dir']!r} tokenizer_dir={probe['tokenizer_dir']!r} "
           f"max_shards={probe['max_shards']!r} (inherited from the runs unless overridden)")

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _ddp, _rank, _local, _world, device = compute_init(device_type)

    try:
        ref = build_reference(arms, args, device, args.no_subspace)
        rows, failed = [], []
        for i, (tag, ckpt) in enumerate(arms, 1):
            print0(f"\n[report] ({i}/{len(arms)}) {tag}")
            try:
                rows.append(measure(tag, ckpt, args, device, ref))
            except Exception as exc:
                # One bad arm must not cost the other twenty their measurements.
                failed.append((tag, f"{type(exc).__name__}: {exc}"))
                print0(f"[report] FAILED {tag}: {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=3, file=sys.stderr)

        out_path = args.out or os.path.join(args.sweep_dir, "sweep_report.csv")
        cols, seen = [], set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    cols.append(k)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        report(rows, failed, out_path, ref)
    finally:
        compute_cleanup()


def report(rows, failed, out_path, ref):
    if not rows:
        print0("[report] nothing measured")
        return
    rows = sorted(rows, key=lambda r: quality(r) if quality(r) is not None else float("inf"))
    dense = next((r for r in rows if r["head"] == "dense"), None)
    d_bpb0 = quality(dense) if dense else None
    d_fl0 = dense.get("flops_per_token") if dense else None

    def num(v, fmt):
        return format(v, fmt) if isinstance(v, (int, float)) else ""

    def pct(v):
        return f"{v * 100:.2f}%" if isinstance(v, (int, float)) else ""

    print0("")
    print0("=" * 108)
    title = "SWEEP REPORT"
    if ref:
        title += f"   (capture measured against {ref['tag']})"
    print0(title)
    print0("=" * 108)
    hdr = (f"{'arm':26s} {'head':9s} {'impl':>6s} {'M':>6s} {'bpb':>8s} {'vs dense':>9s} "
           f"{'FLOPs/tok':>11s} {'vs dense':>9s} {'rank':>7s} {'ceil':>7s} "
           f"{'capture':>8s} {'resid':>7s}")
    print0(hdr)
    print0("-" * len(hdr))
    for r in rows:
        bpb, fl = quality(r), r.get("flops_per_token")
        dbpb = num(bpb - d_bpb0, "+.4f") if (bpb and d_bpb0) else ""
        dfl = f"{fl / d_fl0:.3f}x" if (fl and d_fl0) else ""
        ceil = r.get("rank_ceiling")
        ceil_s = "inf" if ceil == float("inf") else num(ceil, ".0f")
        impl = str(r.get("product_impl", ""))[:6]
        print0(f"{r['arm'][:26]:26s} {r['head']:9s} {impl:>6s} {str(r.get('M', '')):>6s} "
               f"{num(bpb, '.5f'):>8s} {dbpb:>9s} "
               f"{num(fl, '.4e'):>11s} {dfl:>9s} "
               f"{num(r.get('rank_effective_rank'), '.0f'):>7s} {ceil_s:>7s} "
               f"{pct(r.get('capture')):>8s} {pct(r.get('capture_residual')):>7s}")

    print0("")
    print0("  Read bpb against the LEARNED low-rank head, not only against dense. A plain")
    print0("  learned rank-M head is the arm a structured head has to beat, and it already")
    print0("  reaches within 0.07 bpb of dense at two thirds of the FLOPs.")
    print0("  Read `capture` against `resid`. The first is dominated by the single unigram")
    print0("  direction a per-token bias supplies for free; the second is where every")
    print0("  context-dependent contrast lives, and it is the honest number.")
    if any(isinstance(r.get("rank_effective_rank"), (int, float)) for r in rows):
        print0(f"  Ranks are against the ~{RANK_THRESHOLD} empirical head-rank threshold.")
    drifted = [r for r in rows
               if isinstance(r.get("flops_drift"), float) and abs(r["flops_drift"] - 1.0) > 0.01]
    if drifted:
        print0("")
        print0(f"  WARNING: {len(drifted)} arm(s) would cost a different number of FLOPs under")
        print0("  today's code than they did when they ran. The sweep straddles a code change,")
        print0("  so the FLOPs column is not comparable across arms. bpb still is.")
        for r in sorted(drifted, key=lambda r: -abs(r["flops_drift"] - 1.0))[:6]:
            print0(f"    {r['arm'][:30]:30s} ran {r['flops_per_token']:.3e}, "
                   f"today {r['flops_recomputed']:.3e}  ({r['flops_drift']:.2f}x)")

    impls = {r.get("product_impl") for r in rows if r.get("product_impl")}
    if len(impls) > 1:
        print0("")
        print0(f"  WARNING: this sweep mixes product implementations {sorted(impls)}.")
        print0("  `gather` costs 4*V*g and `dense` costs 4*V*M, up to 70x apart at the same M,")
        print0("  so the FLOPs column is NOT comparable across those arms. bpb still is: the")
        print0("  two implementations compute the same function. Re-run the cost comparison on")
        print0("  one implementation before reading anything off the FLOPs axis.")
    if failed:
        print0("")
        print0(f"  {len(failed)} arm(s) failed to measure:")
        for tag, err in failed:
            print0(f"    {tag:30s} {err}")
    print0("")
    print0(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
