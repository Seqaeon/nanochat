"""Active parameters and active matrix parameters for every arm in the paper's tables.

WHY THIS EXISTS
    Tables 1 and 5 report total parameters and total matrix parameters. For dense those
    are also the active counts, but MST's headline arm gates streams (mst_stream_topk=1
    of N=4), so its active counts are strictly smaller and are the ones a reader should
    compare against dense. Reporting only totals overstates MST's cost on exactly the
    axis the sparsity is supposed to help.

WHAT COUNTS AS GATED
    The N streams span the WHOLE layer: attention, FFN and the transition are all
    per-stream. The top-k GATE, however, is applied only to the FFN output unless
    mst_stream_gate_attn is set, because gating attention would stop a skipped token
    being a key/value for that stream and change the attention semantics. So the gated
    fraction is the FFN's share of the layer, not the layer.

    This script prints that share explicitly, so the "MST gates a third of a layer"
    claim is a measured number rather than an estimate. estimate_flops() is the single
    source of truth for what is discounted; this reads its output rather than
    re-deriving it.

Usage:
    python -m scripts.p11_active_params
    python -m scripts.p11_active_params --gate-attn     # what gating attention would buy
"""

import argparse
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.mst import MST, BatchedMSTLayer

VOCAB, SEQ, ASPECT, N_SUBS, HEAD_DIM = 32768, 2048, 64, 4, 128
# base_train.py defaults --window-pattern to SSSL; GPTConfig's own default is SSSSL.
# The runs used base_train's, and the pattern changes the attention FLOPs term, so it
# has to be set explicitly or the dense FLOPs will not match the recorded ones.
WINDOW = "SSSL"


def model_dim(depth):
    return ((depth * ASPECT + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM


def dense_cfg(depth):
    D = model_dim(depth)
    return GPTConfig(sequence_len=SEQ, vocab_size=VOCAB, n_layer=depth,
                     n_head=D // HEAD_DIM, n_kv_head=D // HEAD_DIM, n_embd=D,
                     window_pattern=WINDOW)


def mst_cfg(depth, topk=1, gate_attn=0):
    """The headline SP2_k1 arm: mst_config() + BEST + K1 from p08_mst_parity_sweep.sh."""
    D = model_dim(depth)
    return GPTConfig(
        sequence_len=SEQ, vocab_size=VOCAB, n_layer=depth,
        n_head=D // HEAD_DIM, n_kv_head=D // HEAD_DIM, n_embd=D,
        window_pattern=WINDOW,
        use_mst=True, mst_n_subs=N_SUBS, mst_sub_dim=D // N_SUBS,
        mst_head_dim=0, mst_input_mode='learned_proj',
        mst_routing_mode='soft_weighted', mst_routing_topk=0,
        mst_ffn_mode='standard', mst_transition_mode='aggregate_distribute',
        mst_final_mode='concat_proj', mst_final_topk=0,
        mst_routing_aux_weight=0.01, mst_diversity_weight=0.0,
        mst_grad_equalize=1, mst_block_diagonal_muon=1,
        mst_transition_width_mult=float(N_SUBS), mst_sub_lr_scale=2.0,
        mst_multi_scale_windows=1,
        mst_sub_head_dim=64, mst_per_stream_ve=1, mst_compose_windows=1,
        mst_wo_mode='dense',
        mst_stream_topk=topk, mst_stream_router_noise=1.0,
        mst_stream_gate_attn=gate_attn,
    )


def build(cfg, cls):
    with torch.device("meta"):
        return cls(cfg)


def layer_breakdown(m):
    """Per-layer matmul parameters by role, from the first coupling layer."""
    layer = next(l for l in m.layers if isinstance(l, BatchedMSTLayer))
    def n(name):
        p = getattr(layer, name, None)
        return p.numel() if p is not None else 0
    attn = n('c_q_w') + n('c_k_w') + n('c_v_w') + n('c_proj_w') + n('c_proj_dense_w')
    ffn = n('fc_w') + n('fc_proj_w')
    trans = (n('router_w') + n('gate_w') + n('agg_up_w') + n('agg_down_w')
             + n('distribute_w') + n('stream_router_w'))
    return attn, ffn, trans


def report(depths, gate_attn):
    print(f"{'arm':<10} {'L':>3} {'total':>14} {'active':>14} {'matrices':>12} "
          f"{'act.matrices':>13} {'FLOPs/tok':>11} {'act.FLOPs':>11}")
    rows = []
    for depth in depths:
        for name, cls, cfg in (("dense", GPT, dense_cfg(depth)),
                               ("mst", MST, mst_cfg(depth, gate_attn=gate_attn))):
            try:
                m = build(cfg, cls)
            except AssertionError as e:
                # MST needs mst_sub_head_dim | (D/N), i.e. D a multiple of 256. Depths
                # whose D=64L rounded to 128 is not (18 -> 1152, 22 -> 1408) have no MST
                # arm at all; the dense row is still wanted as a curve point.
                print(f"{name:<10} {depth:3d}   (no MST arm: {str(e).split(';')[0]})")
                continue
            counts = m.num_scaling_params()
            total_flops, active_flops, active_params = m.estimate_flops()
            total = counts['total']
            matrices = counts['transformer_matrices']
            # estimate_flops discounts active_params by exactly the gated matmul
            # parameters, so the same delta is the matrix-parameter discount.
            act_matrices = matrices - (total - active_params)
            print(f"{name:<10} {depth:3d} {total:14,d} {active_params:14,d} "
                  f"{matrices/1e6:11.1f}M {act_matrices/1e6:12.1f}M "
                  f"{total_flops:11.3e} {active_flops:11.3e}")
            rows.append((name, depth, total, active_params, matrices, act_matrices,
                         total_flops, active_flops))
        print()
    return rows


def emit(arm, depth, field, gate_attn=0, shared=0):
    """Print one number for a shell script to consume.

    p12_isoflop.sh needs active FLOPs per token BEFORE the run, to turn a training-FLOPs
    budget into a token count. It cannot use base_train's --target-flops, because that
    divides by estimate_flops()[0], the TOTAL count, while the paper's training-FLOPs
    axis uses [1], the active one. For MST those differ by 1.34x at L=24, so the two
    routes would put MST on a different isoFLOP contour than dense.
    """
    cls, cfg = ((GPT, dense_cfg(depth)) if arm == "dense"
                else (MST, mst_cfg(depth, gate_attn=gate_attn)))
    if arm == "mst" and shared:
        from dataclasses import replace
        cfg = replace(cfg, mst_stream_shared=shared)
    m = build(cfg, cls)
    counts = m.num_scaling_params()
    total_flops, active_flops, active_params = m.estimate_flops()
    vals = {"active_flops": active_flops, "total_flops": total_flops,
            "matrices": counts["transformer_matrices"], "total": counts["total"],
            "active": active_params,
            "scaling": counts["transformer_matrices"] + counts["lm_head"]}
    print(f"{vals[field]:.10g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", nargs=3, metavar=("ARM", "DEPTH", "FIELD"),
                    help="print one value and exit, e.g. --emit mst 16 active_flops. "
                         "Fields: active_flops total_flops matrices total active scaling")
    ap.add_argument("--shared", type=int, default=0, help="--emit: mst_stream_shared")
    ap.add_argument("--depths", type=int, nargs="+", default=[8, 16, 18, 20, 22, 24])
    ap.add_argument("--gate-attn", action="store_true",
                    help="also gate attention (mst_stream_gate_attn=1)")
    args = ap.parse_args()

    if args.emit:
        arm, depth, field = args.emit
        emit(arm, int(depth), field, int(args.gate_attn), args.shared)
        return

    print(f"MST headline arm: N={N_SUBS}, mst_stream_topk=1, "
          f"gate_attn={int(args.gate_attn)}\n")
    report(args.depths, int(args.gate_attn))

    # What fraction of a layer the top-k gate actually removes.
    m = build(mst_cfg(24, gate_attn=int(args.gate_attn)), MST)
    attn, ffn, trans = layer_breakdown(m)
    layer_total = attn + ffn + trans
    frac = 1.0 - 1.0 / N_SUBS
    gated = (ffn + attn) * frac if args.gate_attn else ffn * frac
    print(f"Per-layer matmul parameters at L=24 (D={model_dim(24)}, d={model_dim(24)//N_SUBS}):")
    print(f"  attention (incl. dense W_O) {attn/1e6:8.2f}M  {attn/layer_total:6.1%}")
    print(f"  FFN                         {ffn/1e6:8.2f}M  {ffn/layer_total:6.1%}")
    print(f"  transition + routers        {trans/1e6:8.2f}M  {trans/layer_total:6.1%}")
    print(f"  total                       {layer_total/1e6:8.2f}M")
    print(f"\n  top-k=1 of {N_SUBS} removes {gated/layer_total:.1%} of the layer's matmul "
          f"parameters; {1 - gated/layer_total:.1%} stays active.")
    if not args.gate_attn:
        print("  (the gate reaches the FFN only; --gate-attn shows the alternative)")


if __name__ == "__main__":
    main()
