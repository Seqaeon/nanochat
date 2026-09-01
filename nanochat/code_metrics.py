"""
Metrics for the Structured Code Output Heads project.

Section 6 of ``structured-code-output-heads-plan.md`` lists six things that must
be logged on *every* run.  This module implements all of them behind one entry
point so that an arm is either fully instrumented or not run at all:

  1. **Perplexity by token-frequency decile.**  The money plot.  The hypothesis
     is a crossover: the code head loses on head tokens and wins on tail tokens,
     because a code forces parameter sharing while a softmax gives every token a
     free independent row.  No crossover in any configuration kills the
     tail-generalisation framing.
  2. **Achieved logit-matrix rank**, by SVD after mean-centring across the
     vocabulary axis.  Also measured for the softmax baseline, whose *achieved*
     rank is well below ``d`` late in training, which changes what "matched
     capacity" means.
  3. **Zero-shot vocabulary extension.**  Tokens held out of training, scored at
     validation time against an untrained softmax row.
  4. **Head wall-clock and peak memory**, reported separately from the model.
  5. **Final-layer hidden-state anisotropy**, the documented symptom of this
     class of bottleneck.
  6. **Bits-per-byte alongside perplexity**, mandatory across vocab sizes.

Two measurement traps are handled here rather than left to the caller:

  * **Mean-centre before the SVD.**  Forgetting it leaves the rank-1 ``A(h)``
    term and inflates the measured rank by exactly 1, which is small enough to
    look like noise and large enough to make an order-1 head read as rank B+1.
  * **Probe in fp32.**  With a bf16 ``Phi`` the singular values below the true
    rank sit at ~1e-3 of the leading one instead of at zero, so a
    fixed 1e-6 threshold reports full rank for a genuinely rank-15 matrix.  The
    probe temporarily promotes ``Phi`` to fp32; the trained weights are untouched.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import asdict, is_dataclass

import torch
import torch.nn.functional as F

from nanochat.common import print0


# ---------------------------------------------------------------------------
# Frequency deciles
# ---------------------------------------------------------------------------

def frequency_deciles(freqs: torch.Tensor, vocab_size: int, n_bins: int = 10) -> torch.Tensor:
    """Assign each token id to a frequency bin, balanced by *corpus mass*.

    Bin 0 holds the most frequent types, bin ``n_bins - 1`` the rarest.  Bins are
    cut so each holds roughly the same number of token *occurrences*, not the
    same number of types.  Rank-balanced bins would put 10% of the vocabulary in
    the rarest bin and give it a handful of validation samples; mass-balanced
    bins give every decile comparable statistical power, which is what a
    crossover claim needs.  The number of *types* per bin is reported alongside,
    since that is the quantity the sharing argument is actually about.
    """
    f = freqs[:vocab_size].double().clamp_min(0.0)
    order = torch.argsort(f, descending=True)
    sorted_f = f[order]
    cum = torch.cumsum(sorted_f, dim=0)
    total = cum[-1].clamp_min(1.0)
    edges = torch.linspace(0, 1, n_bins + 1)[1:-1] * total
    positions = torch.searchsorted(cum.contiguous(), edges.contiguous())
    bins_sorted = torch.zeros(vocab_size, dtype=torch.int64)
    start = 0
    for b, end in enumerate(positions.tolist() + [vocab_size]):
        end = max(end, start)
        bins_sorted[start:end] = b
        start = end
    bins = torch.empty(vocab_size, dtype=torch.int64)
    bins[order] = bins_sorted
    return bins


# ---------------------------------------------------------------------------
# Grouped bits-per-byte
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb_grouped(model, batches, steps, token_bytes, groups, n_groups,
                         extra_mask=None):
    """Bits-per-byte and mean loss, overall and per group.

    ``groups`` is a ``(vocab_size,)`` int64 tensor mapping token id -> group.
    ``extra_mask`` is an optional ``(vocab_size,)`` bool selecting an additional
    subset to report separately (used for the held-out vocabulary).

    Uses the same sum-nats / sum-bytes construction as ``loss_eval.evaluate_bpb``
    so the numbers are directly comparable to the training log's val bpb, and so
    they remain comparable across tokenizers with different vocabulary sizes.
    """
    device = model.get_device()
    groups = groups.to(device)
    tb = token_bytes.to(device)
    nats = torch.zeros(n_groups + 1, dtype=torch.float64, device=device)
    byts = torch.zeros(n_groups + 1, dtype=torch.float64, device=device)
    toks = torch.zeros(n_groups + 1, dtype=torch.float64, device=device)
    sub_nats = torch.zeros(2, dtype=torch.float64, device=device)
    sub_byts = torch.zeros(2, dtype=torch.float64, device=device)
    sub_toks = torch.zeros(2, dtype=torch.float64, device=device)

    it = iter(batches)
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            break
        loss2d = model(x, y, loss_reduction='none').view(-1).double()
        yf = y.view(-1)
        valid = yf >= 0
        safe = torch.where(valid, yf, torch.zeros_like(yf))
        nb = torch.where(valid, tb[safe].double(), torch.zeros_like(loss2d))
        counted = nb > 0                      # excludes special tokens, as bpb does
        g = torch.where(counted, groups[safe], torch.full_like(safe, n_groups))
        nats.index_add_(0, g, loss2d * counted)
        byts.index_add_(0, g, nb)
        toks.index_add_(0, g, counted.double())
        if extra_mask is not None:
            em = extra_mask.to(device)[safe] & counted
            sel = em.long()
            sub_nats.index_add_(0, sel, loss2d * counted)
            sub_byts.index_add_(0, sel, nb)
            sub_toks.index_add_(0, sel, counted.double())

    def _pack(n, b, t):
        n, b, t = float(n), float(b), float(t)
        if b <= 0:
            return {"bpb": None, "loss": None, "tokens": 0}
        return {"bpb": n / (math.log(2) * b), "loss": n / max(t, 1.0), "tokens": int(t)}

    out = {"overall": _pack(nats[:n_groups].sum(), byts[:n_groups].sum(), toks[:n_groups].sum())}
    out["by_group"] = [_pack(nats[i], byts[i], toks[i]) for i in range(n_groups)]
    if extra_mask is not None:
        out["subset_in"] = _pack(sub_nats[1], sub_byts[1], sub_toks[1])
        out["subset_out"] = _pack(sub_nats[0], sub_byts[0], sub_toks[0])
    return out


# ---------------------------------------------------------------------------
# Logit-matrix rank
# ---------------------------------------------------------------------------

class _Fp32Phi:
    """Context manager promoting a frozen ``Phi`` to fp32 for the probe.

    Rank is a property of the *span* of Phi, so measuring it in the storage
    dtype that training happened to use measures the rounding, not the model.
    """

    def __init__(self, head):
        self.head = head
        self.saved = None

    def __enter__(self):
        phi = getattr(self.head, "phi", None)
        if isinstance(phi, torch.Tensor) and phi.numel() > 0 and phi.dtype != torch.float32:
            self.saved = phi
            self.head.phi = phi.float()
        return self

    def __exit__(self, *exc):
        if self.saved is not None:
            self.head.phi = self.saved
        return False


@torch.no_grad()
def measure_logit_rank(model, batches, steps, vocab_size, n_rows=8192, n_cols=8192,
                       seed=0, tol_rel=1e-5):
    """Effective rank of the pre-softcap logit matrix.

    Returns ``effective_rank`` (singular values above ``tol_rel`` of the
    leading one), ``rank_99`` (smallest k carrying 99% of the spectral energy),
    ``stable_rank`` (``||L||_F^2 / sigma_1^2``, threshold-free) and the leading
    spectrum.

    Two approximations, both harmless and both deliberate:
      * rows are capped at ``n_rows`` contexts, because an ``N x V`` fp32 matrix
        at N=50k, V=32k is 6.5 GB;
      * columns are subsampled to ``n_cols``, which preserves rank up to
        ``n_cols`` almost surely for a random subset.
    Centring is done over the FULL vocabulary axis before the column subsample,
    so removing ``A(h)`` stays exact.
    """
    head = model.lm_head
    if getattr(head, "custom_loss", False):
        return {"effective_rank": None, "rank_99": None, "stable_rank": None,
                "note": "hierarchical softmax has no materialised logit matrix"}
    device = model.get_device()
    g = torch.Generator().manual_seed(seed)
    cols = torch.randperm(vocab_size, generator=g)[:min(n_cols, vocab_size)].sort().values.to(device)

    captured = []
    n_seen = 0

    def hook(_mod, _inp, out):
        nonlocal n_seen
        if n_seen >= n_rows:
            return
        L = out.detach().reshape(-1, out.shape[-1])[:, :vocab_size].float()
        L = L - L.mean(dim=1, keepdim=True)       # remove the rank-1 A(h) term
        take = min(n_rows - n_seen, L.shape[0])
        captured.append(L[:take].index_select(1, cols).cpu())
        n_seen += take

    # Force the head to run in fp32 for the probe. The backbone still runs in
    # bf16, which is fine: the logit matrix's row space is fixed by the head's
    # weight (or by Phi), so only the head's arithmetic can manufacture rank.
    # Without this the DENSE baseline reads as rank ~276 at d=64, because bf16
    # rounding lifts every singular value past d off the floor -- the same trap
    # that would make an order-1 code head look full-rank.
    fp32_in = head.register_forward_pre_hook(lambda _m, inp: (inp[0].float(),) + inp[1:])
    handle = head.register_forward_hook(hook)
    it = iter(batches)
    try:
        with _Fp32Phi(head):
            for _ in range(steps):
                if n_seen >= n_rows:
                    break
                try:
                    x, _y = next(it)
                except StopIteration:
                    break
                model(x)
    finally:
        handle.remove()
        fp32_in.remove()

    if not captured:
        return {"effective_rank": None, "rank_99": None, "stable_rank": None}
    L = torch.cat(captured, dim=0)
    s = torch.linalg.svdvals(L.double())
    s2 = s.pow(2)
    total = s2.sum().clamp_min(1e-300)
    cum = torch.cumsum(s2, dim=0) / total
    return {
        "effective_rank": int((s > s[0] * tol_rel).sum()),
        "rank_99": int((cum < 0.99).sum()) + 1,
        "stable_rank": float(total / s2[0].clamp_min(1e-300)),
        "rows": int(L.shape[0]),
        "cols": int(L.shape[1]),
        "spectrum_head": [float(v) for v in s[:32]],
    }


# ---------------------------------------------------------------------------
# Anisotropy
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_anisotropy(model, batches, steps, n_vectors=4096, seed=0):
    """Mean pairwise cosine similarity of the final hidden states.

    Representation collapse (all hidden states pointing the same way) is the
    documented symptom of squeezing the output layer, so it is measured rather
    than assumed.  Captured on the *input* to the head, which is the normalised
    final hidden state.
    """
    device = model.get_device()
    buf, n_seen = [], 0

    def pre_hook(_mod, inp):
        nonlocal n_seen
        if n_seen >= n_vectors:
            return
        h = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
        take = min(n_vectors - n_seen, h.shape[0])
        buf.append(h[:take].cpu())
        n_seen += take

    handle = model.lm_head.register_forward_pre_hook(pre_hook)
    it = iter(batches)
    try:
        for _ in range(steps):
            if n_seen >= n_vectors:
                break
            try:
                x, _y = next(it)
            except StopIteration:
                break
            model(x)
    finally:
        handle.remove()
    if not buf:
        return {"anisotropy": None}
    H = F.normalize(torch.cat(buf, dim=0), dim=-1)
    g = torch.Generator().manual_seed(seed)
    n = H.shape[0]
    i = torch.randint(0, n, (100_000,), generator=g)
    j = torch.randint(0, n, (100_000,), generator=g)
    keep = i != j
    cos = (H[i[keep]] * H[j[keep]]).sum(dim=-1)
    return {"anisotropy": float(cos.mean()), "anisotropy_std": float(cos.std()),
            "hidden_vectors": int(n)}


# ---------------------------------------------------------------------------
# Zero-shot vocabulary extension
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_holdout_rank(model, batches, steps, holdout_mask, max_positions=20000):
    """Where the true token lands in the ranking, for held-out targets.

    bits-per-byte answers "how surprised was the model"; this answers "would it
    ever generate the token".  A softmax row that never received a gradient sits
    at its initialisation, so its logit is arbitrary and the true token's rank is
    near-uniform.  A code head composes the token's logit from monomial
    coefficients that *were* trained, so the rank should be far better than
    chance.  That gap is the capability separation, and it does not depend on
    calibration the way perplexity does.
    """
    device = model.get_device()
    hm = holdout_mask.to(device)
    ranks, recips, n = [], [], 0
    it = iter(batches)
    for _ in range(steps):
        if n >= max_positions:
            break
        try:
            x, y = next(it)
        except StopIteration:
            break
        logits = model(x)
        V = logits.shape[-1]
        lf = logits.reshape(-1, V)
        yf = y.reshape(-1)
        sel = (yf >= 0) & hm[yf.clamp_min(0)]
        if not bool(sel.any()):
            continue
        lf, yf = lf[sel], yf[sel]
        take = min(max_positions - n, lf.shape[0])
        lf, yf = lf[:take], yf[:take]
        true_logit = lf.gather(1, yf.unsqueeze(1))
        r = (lf > true_logit).sum(dim=1) + 1
        ranks.append(r.cpu())
        recips.append((1.0 / r.double()).cpu())
        n += take
    if not ranks:
        return {"holdout_positions": 0}
    r = torch.cat(ranks).double()
    return {
        "holdout_positions": int(r.numel()),
        "holdout_mean_rank": float(r.mean()),
        "holdout_median_rank": float(r.median()),
        "holdout_mrr": float(torch.cat(recips).mean()),
        "holdout_top10_acc": float((r <= 10).double().mean()),
        "holdout_top100_acc": float((r <= 100).double().mean()),
    }


# ---------------------------------------------------------------------------
# Head cost, measured rather than derived
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_head_cost(model, n_embd, batch=8, seq=1024, iters=20, warmup=5):
    """Wall-clock and peak memory of the head alone.

    Reported separately from the model total because the plan's cost argument is
    about the head and a whole-model number would bury a 4x head difference
    under the backbone.
    """
    head = model.lm_head
    if getattr(head, "custom_loss", False):
        return {}
    device = model.get_device()
    try:
        x = torch.randn(batch, seq, n_embd, device=device,
                        dtype=next(head.parameters()).dtype)
    except StopIteration:
        return {}
    cuda = device.type == "cuda" if hasattr(device, "type") else "cuda" in str(device)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        head(x)
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        head(x)
    if cuda:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    out = {"head_forward_ms": dt * 1e3,
           "head_forward_us_per_token": dt * 1e6 / (batch * seq)}
    if cuda:
        out["head_peak_mem_mib"] = torch.cuda.max_memory_allocated() / 1024 / 1024
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_all_diagnostics(model, build_val_loader, token_bytes, vocab_size, steps=100,
                        decile=True, rank_contexts=0, holdout_ids=None,
                        tokenizer_dir=None, device=None, n_bins=10):
    """Run every section-6 metric and return one flat-ish dict.

    ``build_val_loader`` is a zero-argument factory, because each probe consumes
    its own independent pass over the validation stream.
    """
    from nanochat.code_head import load_freq_table

    model.eval()
    metrics = {}
    freqs = load_freq_table(vocab_size, tokenizer_dir)
    if freqs is None:
        if decile:
            print0("[SCH] no tokenizer/freq_table.pt: skipping frequency deciles. "
                   "Build it with `python -m scripts.code_assign --build-freq-table`.")
        decile = False

    holdout_mask = None
    if holdout_ids is not None and len(holdout_ids) > 0:
        holdout_mask = torch.zeros(vocab_size, dtype=torch.bool)
        holdout_mask[holdout_ids.cpu()] = True

    if decile or holdout_mask is not None:
        groups = frequency_deciles(freqs, vocab_size, n_bins) if decile \
            else torch.zeros(vocab_size, dtype=torch.int64)
        n_groups = n_bins if decile else 1
        res = evaluate_bpb_grouped(model, build_val_loader(), steps, token_bytes,
                                   groups, n_groups, extra_mask=holdout_mask)
        metrics["bpb"] = res["overall"]["bpb"]
        metrics["loss"] = res["overall"]["loss"]
        if decile:
            counts = torch.bincount(groups, minlength=n_bins)
            for i, d in enumerate(res["by_group"]):
                metrics[f"bpb_decile{i}"] = d["bpb"]
                metrics[f"tokens_decile{i}"] = d["tokens"]
                metrics[f"types_decile{i}"] = int(counts[i])
            head_b, tail_b = metrics.get("bpb_decile0"), metrics.get(f"bpb_decile{n_bins - 1}")
            if head_b and tail_b:
                # The crossover statistic. Positive means the tail costs more
                # than the head, which every model shows; what matters is how
                # this number moves between the code head and the softmax.
                metrics["bpb_tail_minus_head"] = tail_b - head_b
        if holdout_mask is not None:
            metrics["bpb_holdout"] = res["subset_in"]["bpb"]
            metrics["bpb_seen"] = res["subset_out"]["bpb"]
            metrics["holdout_eval_tokens"] = res["subset_in"]["tokens"]
            metrics.update(measure_holdout_rank(model, build_val_loader(), steps, holdout_mask))

    if rank_contexts and rank_contexts > 0:
        metrics.update({f"rank_{k}": v for k, v in
                        measure_logit_rank(model, build_val_loader(), steps, vocab_size,
                                           n_rows=min(rank_contexts, 16384)).items()})
    metrics.update(measure_anisotropy(model, build_val_loader(), steps))

    head = model.lm_head
    metrics["head_params"] = int(sum(p.numel() for p in head.parameters()))
    if hasattr(head, "flops_per_token"):
        metrics["head_flops_per_token"] = int(head.flops_per_token())
    else:
        # Dense softmax: V*d MACs per token, 6 FLOPs per MAC. Recorded on the
        # baseline rows too, so the cost table is one column, not two.
        metrics["head_flops_per_token"] = int(6 * vocab_size * model.config.n_embd)
        metrics["rank_ceiling"] = model.config.n_embd + 1
    if hasattr(head, "rank_ceiling"):
        metrics["rank_ceiling"] = head.rank_ceiling()
        metrics["phi_width_M"] = head.width
        metrics["code_bits_B"] = head.bits
        metrics["code_order_k"] = head.order
    metrics.update(measure_head_cost(model, model.config.n_embd))
    model.train()
    return metrics


SCH_CSV_COLUMNS = [
    "run_tag", "depth", "n_embd", "vocab_size", "seed",
    "use_code_head", "head_type", "phi_mode", "code_mode", "code_ecc_bits",
    "code_bits_B", "code_order_k", "phi_width_M", "g_type", "g_hidden",
    "mixture", "residual_rank", "logit_act", "bias", "input_mode",
    "rank_ceiling", "rank_effective_rank", "rank_rank_99", "rank_stable_rank",
    "val_bpb", "min_val_bpb", "bpb", "loss",
    "bpb_decile0", "bpb_decile1", "bpb_decile2", "bpb_decile3", "bpb_decile4",
    "bpb_decile5", "bpb_decile6", "bpb_decile7", "bpb_decile8", "bpb_decile9",
    "bpb_tail_minus_head",
    "bpb_holdout", "bpb_seen", "holdout_eval_tokens", "holdout_mean_rank",
    "holdout_median_rank", "holdout_mrr", "holdout_top10_acc", "holdout_top100_acc",
    "anisotropy", "head_params", "head_flops_per_token", "head_forward_ms",
    "head_peak_mem_mib",
    "num_params", "scaling_params", "flops_per_token", "total_tokens",
    "train_time_min",
]


def write_sch_row(run_dir, args, config, metrics, val_bpb, min_val_bpb,
                  num_flops_per_token, num_params, total_tokens,
                  total_training_time, scaling_params):
    """Append one row per run to ``sch_results.csv`` next to the run directory.

    Mirrors the MST tracker's convention (one CSV shared by a whole sweep, plus a
    cwd copy) so the phase scripts can be read as tables without log scraping.
    A JSON sidecar carries the full metric dict, including the spectrum, which
    does not fit a CSV cell.
    """
    sp = scaling_params or {}
    row = {
        "run_tag": getattr(args, "model_tag", "") or f"d{getattr(args, 'depth', 0)}",
        "depth": getattr(config, "n_layer", None),
        "n_embd": getattr(config, "n_embd", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "seed": getattr(args, "seed", -1),
        "use_code_head": int(bool(getattr(config, "use_code_head", False))),
        "head_type": getattr(config, "sch_head_type", "code"),
        "phi_mode": getattr(config, "sch_phi_mode", ""),
        "code_mode": getattr(config, "sch_code_mode", ""),
        "code_ecc_bits": getattr(config, "sch_code_ecc_bits", 0),
        "g_type": getattr(config, "sch_g_type", ""),
        "g_hidden": getattr(config, "sch_g_hidden", 0),
        "mixture": getattr(config, "sch_mixture", 1),
        "residual_rank": getattr(config, "sch_residual_rank", 0),
        "logit_act": getattr(config, "sch_logit_act", "none"),
        "bias": getattr(config, "sch_bias", 0),
        "input_mode": getattr(config, "sch_input_mode", "table"),
        "val_bpb": val_bpb,
        "min_val_bpb": min_val_bpb,
        "num_params": num_params,
        "scaling_params": sp.get("transformer_matrices", 0) + sp.get("lm_head", 0),
        "flops_per_token": num_flops_per_token,
        "total_tokens": total_tokens,
        "train_time_min": total_training_time / 60.0 if total_training_time else None,
    }
    for k, v in metrics.items():
        if k in ("rank_spectrum_head", "rank_note"):
            continue
        row.setdefault(k, v)
    row = {k: row.get(k) for k in SCH_CSV_COLUMNS}

    targets = [os.path.normpath(os.path.join(run_dir, "..", "sch_results.csv")),
               os.path.join(os.getcwd(), "sch_results.csv")]
    for path in dict.fromkeys(targets):
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            new = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=SCH_CSV_COLUMNS)
                if new:
                    w.writeheader()
                w.writerow(row)
        except Exception as e:  # pragma: no cover
            print0(f"[SCH] could not write {path}: {e}")

    try:
        side = os.path.join(run_dir, "sch_metrics.json")
        os.makedirs(run_dir, exist_ok=True)
        payload = {"row": row, "metrics": metrics,
                   "config": asdict(config) if is_dataclass(config) else str(config)}
        with open(side, "w") as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception as e:  # pragma: no cover
        print0(f"[SCH] could not write the metrics sidecar: {e}")

    print0("[SCH] " + "  ".join(
        f"{k}={row[k]:.4f}" if isinstance(row[k], float) else f"{k}={row[k]}"
        for k in ("run_tag", "phi_width_M", "rank_ceiling", "rank_effective_rank",
                  "val_bpb", "bpb_holdout", "bpb_tail_minus_head")
        if row.get(k) is not None))
    return row
