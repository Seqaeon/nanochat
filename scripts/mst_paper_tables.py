"""
Recompute every cost cell for the MST paper from the *current* model code, so the
MST and dense arms are internally consistent (both FLOP counts come from the same
estimate_flops implementation at the same commit).

Usage:
    python -m scripts.mst_paper_tables            # print tables + write JSON/CSV
    python -m scripts.mst_paper_tables --latex    # also emit the LaTeX table body

Models are built on the meta device: shapes only, no allocation, no GPU needed.
"""
import argparse
import contextlib
import io
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanochat.gpt import GPT, GPTConfig
from nanochat.mst import MST

# ── the ladder, matching scripts/p07_mst_scaling_sweep.sh ────────────────────
ASPECT_RATIO = 64          # model_dim = depth * aspect_ratio, rounded up to head_dim
HEAD_DIM = 128             # --head-dim default in base_train.py
N_SUBS = 4                 # --mst-n-subs
SEQ_LEN = 2048             # --sequence-len
VOCAB = 32768              # tokenizer/ vocab size
WINDOW_PATTERN = "SSSL"    # --window-pattern default
PARAM_DATA_RATIO = 10.5    # --target-param-data-ratio

MST_DEPTHS = [8, 9, 16, 18, 20, 22, 24, 26, 32]
DENSE_DEPTHS = [8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

# Measured validation bits-per-byte. None => not run.
BPB = {
    ("mst", 8): 1.050999, ("mst", 9): 1.000036, ("mst", 16): 0.8809507,
    ("mst", 18): 0.8543874, ("mst", 20): 0.8349362, ("mst", 22): 0.8103885,
    ("mst", 24): 0.7945918, ("mst", 26): 0.7804194,
    ("mst", 32): 0.7433941,
    ("dense", 8): 0.969126, ("dense", 12): 0.9030, ("dense", 14): 0.8688,
    ("dense", 16): 0.8364, ("dense", 18): 0.8131, ("dense", 20): 0.7906,
    ("dense", 22): 0.7714, ("dense", 24): 0.7545, ("dense", 26): 0.7382,
    ("dense", 28): 0.7246, ("dense", 30): 0.7120,
}


def dims(depth):
    """Replicates build_model_meta() in scripts/base_train.py."""
    base = depth * ASPECT_RATIO
    D = ((base + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    n_head = D // HEAD_DIM
    return D, n_head, D // N_SUBS


def dense_config(depth):
    D, n_head, _ = dims(depth)
    return GPTConfig(sequence_len=SEQ_LEN, vocab_size=VOCAB, n_layer=depth,
                     n_head=n_head, n_kv_head=n_head, n_embd=D,
                     window_pattern=WINDOW_PATTERN)


def mst_config(depth):
    """S7_COMBO_A + multi-scale windows, i.e. COMBO_A_BASE in p07_mst_scaling_sweep.sh."""
    D, n_head, d = dims(depth)
    return GPTConfig(
        sequence_len=SEQ_LEN, vocab_size=VOCAB, n_layer=depth,
        n_head=n_head, n_kv_head=n_head, n_embd=D, window_pattern=WINDOW_PATTERN,
        use_mst=True, mst_n_subs=N_SUBS, mst_sub_dim=d, mst_head_dim=0,
        mst_input_mode='learned_proj',
        mst_routing_mode='soft_weighted', mst_routing_topk=0,
        mst_ffn_mode='standard',
        mst_transition_mode='aggregate_distribute',
        mst_final_mode='concat_proj', mst_final_topk=0,
        mst_routing_aux_weight=0.01, mst_diversity_weight=0.0,
        mst_grad_equalize=1, mst_block_diagonal_muon=1,
        mst_transition_width_mult=float(N_SUBS), mst_sub_lr_scale=2.0,
        mst_multi_scale_windows=1,
    )


def measure(arm, depth):
    cfg = mst_config(depth) if arm == "mst" else dense_config(depth)
    cls = MST if arm == "mst" else GPT
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), torch.device('meta'):
        model = cls(cfg)
    if arm == "mst":
        assert model._use_batched, f"depth {depth}: expected BatchedMSTLayer path"
    counts = model.num_scaling_params()
    flops_per_token, active_flops, active_params = model.estimate_flops()
    # Token budget: the ladder trains every model at PARAM_DATA_RATIO tokens per
    # scaling parameter, where scaling params = transformer matrices + lm_head.
    scaling_params = counts['transformer_matrices'] + counts['lm_head']
    tokens = int(PARAM_DATA_RATIO * scaling_params)
    D, n_head, d = dims(depth)
    return dict(
        arm=arm, depth=depth, d_model=D, n_head=n_head, sub_dim=d if arm == "mst" else None,
        head_dim=(d // n_head) if arm == "mst" else HEAD_DIM,
        total_params=counts['total'],
        matrices=counts['transformer_matrices'],
        wte=counts['wte'], lm_head=counts['lm_head'],
        value_embeds=counts['value_embeds'],
        scaling_params=scaling_params,
        active_params=active_params,
        flops_per_token=flops_per_token,
        active_flops_per_token=active_flops,
        tokens=tokens,
        train_flops=flops_per_token * tokens,
        bpb=BPB.get((arm, depth)),
    )


def fmt(x, sig=4):
    return f"{x:.{sig}e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="emit LaTeX table bodies")
    ap.add_argument("--out", default="scratch/mst_paper_tables.json")
    args = ap.parse_args()

    rows = [measure("mst", D) for D in MST_DEPTHS] + \
           [measure("dense", D) for D in DENSE_DEPTHS]

    hdr = (f"{'arm':>6} {'L':>3} {'D':>5} {'d':>5} {'hd':>4} {'total':>15} "
           f"{'matrices':>13} {'VE':>13} {'scaling':>13} {'flops/tok':>11} "
           f"{'tokens':>15} {'train_flops':>11} {'bpb':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        bpb_s = f"{r['bpb']:.4f}" if r['bpb'] is not None else "-"
        print(f"{r['arm']:>6} {r['depth']:>3} {r['d_model']:>5} "
              f"{str(r['sub_dim'] or '-'):>5} {r['head_dim']:>4} "
              f"{r['total_params']:>15,} {r['matrices']:>13,} {r['value_embeds']:>13,} "
              f"{r['scaling_params']:>13,} {fmt(r['flops_per_token']):>11} "
              f"{r['tokens']:>15,} {fmt(r['train_flops']):>11} {bpb_s:>9}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {args.out}")

    # Sanity checks that would catch a config regression.
    for r in rows:
        if r['arm'] == 'mst':
            assert r['head_dim'] == HEAD_DIM // N_SUBS, r
        assert r['active_params'] == r['total_params'] or r['arm'] == 'dense', \
            f"MST should be fully dense (active == total): {r}"

    if args.latex:
        print("\n% ---- Table 1 body ----")
        for arm, label in (("mst", "MST"), ("dense", "Dense")):
            sub = [r for r in rows if r['arm'] == arm and r['bpb'] is not None]
            for r in sub:
                print(f" & {r['depth']} & {r['total_params']:,} & "
                      f"{r['matrices']/1e6:.1f}M & {fmt(r['flops_per_token'],3)} & "
                      f"{fmt(r['train_flops'],3)} & {r['bpb']:.4f} \\\\")
            print(r"\midrule")


if __name__ == "__main__":
    main()
