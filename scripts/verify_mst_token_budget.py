"""
Compare: if we used MST's own scaling params for target_tokens (vs current dense baseline),
would it train for MORE or FEWER tokens?

Formula (base_train.py:922-939):
    scaling_params = model.num_scaling_params()['transformer_matrices'] + ['lm_head']
    target_tokens  = target_param_data_ratio * scaling_params   (default ratio = 10.5)

Currently estimate_tokens_from_base() always builds a GPT (dense) model —
so MST is trained for 10.5 × dense_scaling_params regardless of its own size.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.mst import MST

VOCAB_SIZE  = 32768
SEQ_LEN     = 2048
RATIO       = 10.5   # default target_param_data_ratio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def model_dims(depth, aspect_ratio=64):
    head_dim = 128
    model_dim = ((depth * aspect_ratio + head_dim - 1) // head_dim) * head_dim
    return head_dim, model_dim

def dense_scaling_params(depth, aspect_ratio=64):
    head_dim, model_dim = model_dims(depth, aspect_ratio)
    num_heads = model_dim // head_dim
    cfg = GPTConfig(
        sequence_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
    )
    with torch.device("meta"):
        m = GPT(cfg)
    counts = m.num_scaling_params()
    return counts['transformer_matrices'] + counts['lm_head'], counts

def mst_scaling_params(depth, n_subs=8, sub_dim=64, aspect_ratio=64):
    head_dim, model_dim = model_dims(depth, aspect_ratio)
    assert model_dim == n_subs * sub_dim, \
        f"n_embd ({model_dim}) must equal n_subs ({n_subs}) × sub_dim ({sub_dim}). " \
        f"Got {n_subs * sub_dim}. Adjust sub_dim."
    num_heads = model_dim // head_dim   # same as dense
    cfg = GPTConfig(
        sequence_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        use_mst=True,
        mst_n_subs=n_subs,
        mst_sub_dim=sub_dim,
        mst_input_mode='fixed_slice',
        mst_routing_mode='soft_weighted',
        mst_transition_mode='aggregate_distribute',
        mst_final_mode='aggregate_proj',
        mst_routing_topk=4,
    )
    with torch.device("meta"):
        m = MST(cfg)
    counts = m.num_scaling_params()
    return counts['transformer_matrices'] + counts['lm_head'], counts


# ---------------------------------------------------------------------------
# Main: compare budgets
# ---------------------------------------------------------------------------

def main():
    TARGET_RATIO = RATIO

    print("=" * 78)
    print("  MST vs Dense Baseline — Token Budget Comparison")
    print(f"  Formula: target_tokens = {TARGET_RATIO} × (transformer_matrices + lm_head)")
    print("=" * 78)

    # Standard MST config from p05 stage5 sweep: N=8, d=64 → D=512
    # That fixes the depth to 8 (since 8*64=512 = depth*64 → depth=8)
    # For other depths we need to adjust n_subs or sub_dim.
    # The AggDist runs from the chart use d8 and d16.
    # depth=8:  model_dim=512  → N=8, d=64  ✓
    # depth=16: model_dim=1024 → N=8, d=128  (or N=16, d=64)

    configs = [
        # (depth, n_subs, sub_dim)  — chosen to match model_dim = depth*64
        (4,  8,  32),   # model_dim=256 → N=8, d=32
        (8,  8,  64),   # model_dim=512 → N=8, d=64  (the AggDist_d8 run)
        (12, 8,  96),   # model_dim=768 → N=8, d=96
        (16, 8, 128),   # model_dim=1024 → N=8, d=128 (AggDist_d16 run)
        (24, 8, 192),   # model_dim=1536 → N=8, d=192
    ]

    print()
    hdr = f"{'depth':>5} | {'Dense scaling':>15} | {'MST scaling':>13} | {'Ratio MST/Dense':>16} | {'Dense tokens':>14} | {'MST tokens':>12} | {'Δ tokens':>12} | Direction"
    print(hdr)
    print("-" * len(hdr))

    for depth, n_subs, sub_dim in configs:
        try:
            d_sp, d_counts = dense_scaling_params(depth)
            m_sp, m_counts = mst_scaling_params(depth, n_subs=n_subs, sub_dim=sub_dim)
        except AssertionError as e:
            print(f"  depth={depth}: SKIPPED — {e}")
            continue

        d_tokens = int(TARGET_RATIO * d_sp)
        m_tokens = int(TARGET_RATIO * m_sp)
        delta    = m_tokens - d_tokens
        ratio    = m_sp / d_sp
        direction = "MORE ↑" if delta > 0 else "LESS ↓"

        print(f"{depth:>5} | {d_sp:>15,} | {m_sp:>13,} | {ratio:>16.4f} | {d_tokens:>14,} | {m_tokens:>12,} | {delta:>+12,} | {direction}")

    # -----------------------------------------------------------------------
    # Detailed breakdown for depth=8 (the AggDist_d8 run)
    # -----------------------------------------------------------------------
    print()
    print("=" * 78)
    print("  Detailed Breakdown — depth=8 (AggDist config: N=8, d=64)")
    print("=" * 78)

    d_sp, d_counts = dense_scaling_params(8)
    m_sp, m_counts = mst_scaling_params(8, n_subs=8, sub_dim=64)

    def fmt(v): return f"{v:>14,}"

    print(f"\n  {'Component':<30} {'Dense':>14} {'MST':>14}")
    print(f"  {'-'*58}")
    for key in ['wte', 'value_embeds', 'lm_head', 'transformer_matrices', 'scalars', 'total']:
        dv = d_counts.get(key, 0)
        mv = m_counts.get(key, 0)
        marker = " ← used for budget" if key in ('transformer_matrices', 'lm_head') else ""
        print(f"  {key:<30} {fmt(dv)} {fmt(mv)}{marker}")
    print(f"  {'-'*58}")
    print(f"  {'scaling_params (tm + lm_head)':<30} {fmt(d_sp)} {fmt(m_sp)}")
    print(f"  {'target_tokens @10.5x':<30} {fmt(int(10.5*d_sp))} {fmt(int(10.5*m_sp))}")
    print()

    # breakdown of MST transformer_matrices
    print("  MST transformer_matrices breakdown:")
    cfg = GPTConfig(
        sequence_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
        n_layer=8, n_head=4, n_kv_head=4, n_embd=512,
        use_mst=True, mst_n_subs=8, mst_sub_dim=64,
        mst_input_mode='fixed_slice',
        mst_routing_mode='soft_weighted',
        mst_transition_mode='aggregate_distribute',
        mst_final_mode='aggregate_proj',
        mst_routing_topk=4,
    )
    with torch.device("meta"):
        mst_model = MST(cfg)
    layers_p     = sum(p.numel() for p in mst_model.layers.parameters())
    input_p      = sum(p.numel() for p in mst_model.input_layer.parameters())
    final_head_p = sum(p.numel() for p in mst_model.final_head.parameters())
    lm_head_p    = sum(p.numel() for p in mst_model.lm_head.parameters())
    print(f"    layers (sub-blocks + transitions): {layers_p:>12,}")
    print(f"    input_layer:                       {input_p:>12,}")
    print(f"    final_head:                        {final_head_p:>12,}")
    print(f"    → transformer_matrices total:      {layers_p+input_p+final_head_p:>12,}")
    print(f"    lm_head:                           {lm_head_p:>12,}")
    print(f"    → scaling_params total:            {layers_p+input_p+final_head_p+lm_head_p:>12,}")

    # compare to dense d8
    d_sp8, _ = dense_scaling_params(8)
    pct = (m_sp - d_sp8) / d_sp8 * 100
    print()
    print(f"  MST scaling_params vs dense: {pct:+.1f}%")
    print(f"  → MST would train for {abs(pct):.1f}% {'MORE' if pct > 0 else 'LESS'} tokens than current (dense-budget) approach")
    print()
    print("  NOTE: current code always calls estimate_tokens_from_base() which builds")
    print("  a GPT (dense) model, so MST always trains on the DENSE token budget.")


if __name__ == "__main__":
    main()
