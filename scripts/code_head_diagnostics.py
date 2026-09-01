"""
Post-hoc diagnostics for a trained Structured Code Head model.

``scripts/base_train.py`` already writes every section-6 metric at the end of a
run.  This script runs the same measurements against a checkpoint, for the cases
that come up constantly in practice: a run that finished before a metric was
added, an arm whose diagnostics were skipped for speed, or a dense baseline you
want to re-measure with exactly the code that measured the code-head arms.

Its other job is the PHASE 0 GATE.  Section 7 of the plan makes one thing a hard
stop: an order-1 code head whose achieved logit rank is not exactly B means the
implementation is wrong, and nothing downstream is meaningful.  This script
prints that verdict as a single unambiguous PASS or FAIL line.

Examples:

    # gate an order-1 run
    python -m scripts.code_head_diagnostics --model-tag c00_gate_order1_s1 --rank-contexts 16384

    # full diagnostics including the held-out vocabulary, from an explicit directory
    python -m scripts.code_head_diagnostics \\
        --checkpoint-dir out/c00_sch_phase0/d8/GATE_order1_B15_s1 \\
        --holdout-ids out/.../sch_holdout_ids.pt --out diag.json

The rank probe promotes Phi to fp32 internally (see
``nanochat.code_metrics._Fp32Phi``).  Do not "optimise" that away: with a bf16
Phi the singular values below the true rank sit at roughly 1e-3 of the leading
one rather than at zero, and a genuinely rank-15 head reads as full rank.  This
was measured, not guessed: a dense d=64 head reported rank 276 before the fix and
exactly 64 after.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import load_model, load_model_from_dir
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.code_metrics import run_all_diagnostics

# Godey et al. (2024): below roughly this logit-matrix rank, head quality degrades
# regardless of model size. Printed alongside every measured rank so one glance
# says whether the arm is above or below the line the ladder has to cross.
RANK_THRESHOLD = 1000


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", type=str, default=None,
                   help="explicit directory holding model_*.pt (overrides --model-tag)")
    p.add_argument("--model-tag", type=str, default=None, help="tag under base_checkpoints")
    p.add_argument("--step", type=int, default=None, help="checkpoint step (default: last)")
    p.add_argument("--tokenizer-dir", type=str, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--max-shards", type=int, default=None)
    p.add_argument("--device-type", type=str, default="")
    p.add_argument("--batch-size", type=int, default=8, help="per-device validation batch size")
    p.add_argument("--seq-len", type=int, default=0, help="0 = the checkpoint's sequence_len")
    p.add_argument("--steps", type=int, default=100, help="validation batches per probe")
    p.add_argument("--rank-contexts", type=int, default=16384,
                   help="rows for the logit-rank SVD (0 = skip the rank probe)")
    p.add_argument("--no-decile", action="store_true", help="skip the frequency-decile breakdown")
    p.add_argument("--holdout-ids", type=str, default=None,
                   help="sch_holdout_ids.pt written by base_train into the checkpoint directory")
    p.add_argument("--out", type=str, default="", help="write the full metric dict here as JSON")
    args = p.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _ddp, _rank, _local_rank, _world, device = compute_init(device_type)

    if args.checkpoint_dir:
        model, tokenizer, meta = load_model_from_dir(
            args.checkpoint_dir, device, phase="eval",
            model_tag=args.model_tag, step=args.step, tokenizer_dir=args.tokenizer_dir)
    else:
        model, tokenizer, meta = load_model(
            "base", device, phase="eval", model_tag=args.model_tag, step=args.step,
            tokenizer_dir=args.tokenizer_dir)

    vocab_size = tokenizer.get_vocab_size()
    seq_len = args.seq_len or meta["model_config"]["sequence_len"]
    token_bytes = get_token_bytes(device=device, tokenizer_dir=args.tokenizer_dir)

    # A factory, not an iterator: each probe consumes its own independent pass.
    def build_val_loader():
        return tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, args.batch_size, seq_len, split="val", device=device,
            data_dir=args.data_dir, max_shards=args.max_shards)

    holdout_ids = None
    if args.holdout_ids:
        holdout_ids = torch.load(args.holdout_ids, weights_only=True, map_location="cpu")
        print0(f"[SCH] scoring {holdout_ids.numel():,} held-out token ids from {args.holdout_ids}")
    elif args.checkpoint_dir:
        guess = os.path.join(args.checkpoint_dir, "sch_holdout_ids.pt")
        if os.path.exists(guess):
            holdout_ids = torch.load(guess, weights_only=True, map_location="cpu")
            print0(f"[SCH] found {holdout_ids.numel():,} held-out token ids at {guess}")

    metrics = run_all_diagnostics(
        model, build_val_loader=build_val_loader, token_bytes=token_bytes,
        vocab_size=vocab_size, steps=args.steps, decile=not args.no_decile,
        rank_contexts=args.rank_contexts, holdout_ids=holdout_ids,
        tokenizer_dir=args.tokenizer_dir, device=device)

    head = model.lm_head
    config = model.config
    is_code = bool(getattr(config, "use_code_head", False))

    print0("")
    print0("=" * 78)
    print0("STRUCTURED CODE HEAD DIAGNOSTICS")
    print0("=" * 78)
    if is_code and hasattr(head, "extra_repr"):
        print0(f"head: {head.extra_repr()}")
    else:
        print0(f"head: dense softmax, V={vocab_size:,} d={config.n_embd}")

    def show(label, key, fmt="{:.5f}"):
        v = metrics.get(key)
        if v is None:
            return
        print0(f"  {label:34s} " + (fmt.format(v) if isinstance(v, float) else f"{v}"))

    print0("\nquality")
    show("bits per byte", "bpb")
    show("mean loss (nats/token)", "loss")

    if not args.no_decile and metrics.get("bpb_decile0") is not None:
        print0("\nbits per byte by frequency decile (0 = most frequent types)")
        print0("  decile   bpb        eval tokens     vocab types")
        for i in range(10):
            b = metrics.get(f"bpb_decile{i}")
            if b is None:
                continue
            print0(f"    {i:<6d} {b:<10.5f} {metrics.get(f'tokens_decile{i}', 0):<15,} "
                   f"{metrics.get(f'types_decile{i}', 0):,}")
        show("tail minus head", "bpb_tail_minus_head", "{:+.5f}")
        print0("  The tail-generalisation thesis predicts this gap SHRINKS for a code head")
        print0("  relative to the dense baseline. No crossover in any configuration kills it.")

    if metrics.get("bpb_holdout") is not None:
        print0("\nzero-shot vocabulary extension")
        show("bpb on held-out ids", "bpb_holdout")
        show("bpb on trained ids", "bpb_seen")
        show("held-out eval tokens", "holdout_eval_tokens")
        show("mean rank of the true token", "holdout_mean_rank", "{:.1f}")
        show("median rank of the true token", "holdout_median_rank", "{:.1f}")
        show("mean reciprocal rank", "holdout_mrr")
        show("true token in top 10", "holdout_top10_acc")
        show("true token in top 100", "holdout_top100_acc")
        print0(f"  Chance-level mean rank at this vocabulary is about {vocab_size // 2:,}.")
        print0("  A dense softmax row that never received a gradient is still at its")
        print0("  initialisation, so it should sit near chance. A code head composes the")
        print0("  logit from monomial coefficients that WERE trained.")

    print0("\nrank")
    ceiling = metrics.get("rank_ceiling")
    measured = metrics.get("rank_effective_rank")
    show("rank ceiling (theory)", "rank_ceiling", "{:.0f}")
    show("effective rank (measured)", "rank_effective_rank")
    show("rank carrying 99% of energy", "rank_rank_99")
    show("stable rank", "rank_stable_rank", "{:.2f}")
    if is_code and hasattr(head, "width"):
        print0(f"  B={head.bits}  k={head.order}  M={head.width}  d={config.n_embd}")
    if measured is not None:
        verdict = "ABOVE" if measured >= RANK_THRESHOLD else "BELOW"
        print0(f"  measured rank is {verdict} the ~{RANK_THRESHOLD} empirical head-rank threshold")

    print0("\nrepresentation")
    show("anisotropy (mean pairwise cosine)", "anisotropy")

    print0("\nhead cost")
    show("head parameters", "head_params")
    show("head FLOPs per token", "head_flops_per_token")
    show("head forward (ms)", "head_forward_ms", "{:.3f}")
    show("head peak memory (MiB)", "head_peak_mem_mib", "{:.1f}")
    dense_params = vocab_size * config.n_embd
    if metrics.get("head_params"):
        print0(f"  dense equivalent: {dense_params:,} parameters, "
               f"{6 * dense_params:,} FLOPs per token")

    # ---- the Phase 0 gate ----------------------------------------------------
    print0("\n" + "=" * 78)
    if is_code and getattr(head, "order", None) == 1 and getattr(head, "n_mixture", 1) == 1 \
            and getattr(head, "logit_act", "none") == "none" and getattr(head, "residual_rank", 0) == 0:
        if measured is None:
            print0("PHASE 0 GATE: NOT RUN (pass --rank-contexts > 0)")
        elif measured == head.bits:
            print0(f"PHASE 0 GATE: PASS. Effective rank {measured} == B {head.bits}, exactly as "
                   f"the independent-bit derivation predicts.")
        else:
            print0(f"PHASE 0 GATE: FAIL. Effective rank {measured} != B {head.bits}.")
            print0("  Section 7 of the plan makes this a hard stop: the implementation is wrong.")
            print0("  Check, in this order: (1) were the logits captured BEFORE the softcap,")
            print0("  (2) were they mean-centred across the vocabulary axis (forgetting it inflates")
            print0("  the rank by exactly 1), (3) was the probe run in fp32.")
    elif is_code:
        print0("PHASE 0 GATE: not applicable. It is defined for a plain order-1 head; this "
               "configuration carries a higher order or a rank mitigation.")
    else:
        print0("PHASE 0 GATE: not applicable to a dense softmax head. Its measured rank above is "
               "the baseline's ACHIEVED rank, which is expected to sit well below d and which "
               "changes what 'matched capacity' means.")
    print0("=" * 78)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"metrics": metrics, "step": meta.get("step"),
                       "model_config": meta.get("model_config")}, f, indent=2, default=str)
        print0(f"wrote {args.out}")

    compute_cleanup()


if __name__ == "__main__":
    main()
