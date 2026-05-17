"""
Verify FLOP and parameter calculations for RemixedLinear (chunk64 variant) vs Dense baseline.

This script instantiates both models on meta device and compares:
1. What estimate_flops() reports vs manual calculation
2. Whether value_embeds are excluded consistently for both models
3. Whether n_head/n_kv_head change with depth for the dense baseline
4. Detailed per-component param breakdown
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from nanochat.gpt import GPT, GPTConfig, RemixedLinear

VOCAB_SIZE = 32768
SEQ_LEN = 2048

def model_dims(depth, aspect_ratio=64):
    """Mirror _sweep_utils.model_dims"""
    head_dim = 128
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    return aspect_ratio, head_dim, model_dim

def build_dense_model(depth, aspect_ratio=64, std_moe_experts=0, std_moe_topk=0):
    """Build dense baseline model on meta device."""
    _, head_dim, model_dim = model_dims(depth, aspect_ratio)
    num_heads = model_dim // head_dim
    config = GPTConfig(
        sequence_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        p23_std_moe_experts=std_moe_experts,
        p23_std_moe_topk=std_moe_topk,
    )
    with torch.device("meta"):
        model = GPT(config)
    return model, config

def build_remix_model(depth, aspect_ratio=64, n_templates=8, chunk_routing_size=64):
    """Build RemixedLinear chunk64 model on meta device (mirrors P29/P30 config)."""
    _, head_dim, model_dim = model_dims(depth, aspect_ratio)
    num_heads = model_dim // head_dim

    config = GPTConfig(
        sequence_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        use_remix_linear=True,
        remix_basis_size=model_dim,  # full-rank basis (P29 canonical)
        remix_context_dim=64,
        remix_context_dim_ratio=6,
        remix_output_gate_rank=16,
        remixed_linear_kwargs=dict(
            use_basis_gate=False,
            use_output_gate=True,
            use_context=True,
            basis_gate_mode='centered',
            gate_temperature=2.0,
            basis_scale_factor=4,
            n_templates=n_templates,
            template_routing_learned=True,
            template_topk=0,  # soft routing over all templates
            tiny_expert=False,
            lokr_expert=False,
            output_gate_rank=16,
            operator_modulation='none',
            use_quantile_route=1,
            disable_ln_basis=False,
        ),
        cclblock_modulation='weight',
        cclblock_context_stream='selective',
        cclblock_gate_temperature=2.0,
        p28_chunk_routing_size=chunk_routing_size,
        p23_quantile_route=1,
        scale_basis_size=True,
    )
    with torch.device("meta"):
        model = GPT(config)
    return model, config


def analyze_model(model, config, label):
    """Detailed analysis of a model's FLOP and param calculation."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  depth={config.n_layer}, n_embd={config.n_embd}, n_head={config.n_head}, n_kv_head={config.n_kv_head}")
    print(f"{'='*70}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Breakdown
    wte_params = model.transformer.wte.weight.numel()
    wpe_params = model.transformer.wpe.weight.numel() if "wpe" in model.transformer else 0
    ve_params = sum(ve.weight.numel() for ve in model.value_embeds.values())
    lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
    resid_lambda_params = model.resid_lambdas.numel()
    x0_lambda_params = model.x0_lambdas.numel()
    transformer_h_params = sum(p.numel() for p in model.transformer.h.parameters())

    print(f"\n--- Parameter Breakdown ---")
    print(f"  wte (token embed):      {wte_params:>12,}")
    print(f"  wpe (pos embed):        {wpe_params:>12,}")
    print(f"  value_embeds:           {ve_params:>12,}  ({len(model.value_embeds)} layers)")
    print(f"  lm_head:                {lm_head_params:>12,}")
    print(f"  resid_lambdas:          {resid_lambda_params:>12,}")
    print(f"  x0_lambdas:             {x0_lambda_params:>12,}")
    print(f"  transformer.h (blocks): {transformer_h_params:>12,}")
    accounted = wte_params + wpe_params + ve_params + lm_head_params + resid_lambda_params + x0_lambda_params + transformer_h_params
    other = total_params - accounted
    if other > 0:
        print(f"  other (embedding_model, etc): {other:>12,}")
    print(f"  TOTAL:                  {total_params:>12,}")

    # estimate_flops
    total_flops, active_flops, active_params = model.estimate_flops()
    print(f"\n--- estimate_flops() ---")
    print(f"  total_flops:  {total_flops:>15,}  ({total_flops:.6e})")
    print(f"  active_flops: {active_flops:>15,}  ({active_flops:.6e})")
    print(f"  active_params:{active_params:>15,}")

    # Manual FLOP calculation
    nparams_exclude = wte_params + wpe_params + ve_params + resid_lambda_params + x0_lambda_params
    matmul_params = total_params - nparams_exclude
    print(f"\n--- Manual Verification ---")
    print(f"  Excluded from matmul count: {nparams_exclude:,}")
    print(f"    wte:            {wte_params:,}")
    print(f"    wpe:            {wpe_params:,}")
    print(f"    value_embeds:   {ve_params:,}")
    print(f"    resid_lambdas:  {resid_lambda_params:,}")
    print(f"    x0_lambdas:     {x0_lambda_params:,}")
    print(f"  Matmul params (total - excluded): {matmul_params:,}")

    # Attention FLOPs
    h = config.n_head
    q = config.n_embd // config.n_head
    t = config.sequence_len
    attn_flops = 0
    for ws in model.window_sizes:
        w = ws[0]
        eff_seq = t if w < 0 else min(w, t)
        attn_flops += 12 * h * q * eff_seq
    print(f"  Attention FLOPs: {attn_flops:,}")

    manual_total_flops = 6 * matmul_params + attn_flops
    print(f"  Manual total_flops = 6*{matmul_params:,} + {attn_flops:,} = {manual_total_flops:,}")
    match = "✅ MATCH" if manual_total_flops == total_flops else f"❌ MISMATCH (diff={manual_total_flops - total_flops:,})"
    print(f"  vs estimate_flops total: {match}")

    # Count inactive expert params
    inactive_expert_params = 0
    remix_details = []
    from nanochat.gpt import RemixedLinear, StandardMoE_MLP
    for name, submod in model.named_modules():
        if isinstance(submod, RemixedLinear):
            chunk = getattr(submod, 'chunk_routing_size', 0)
            n_templates = getattr(submod, 'n_templates', 1)
            if n_templates > 1 and submod.template_bank is not None and chunk > 0:
                template_params = sum(t.numel() for t in submod.template_bank)
                route_params = submod.template_route.numel() if submod.template_route is not None else 0
                topk = getattr(submod, 'template_topk', 0)
                K = n_templates
                active_templates = topk if (0 < topk < K) else K
                active_template_params = template_params * (active_templates / K)
                active_per_token = (active_template_params + route_params) / chunk
                inactive = (template_params + route_params) - active_per_token
                inactive_expert_params += int(inactive)
                remix_details.append({
                    'name': name,
                    'chunk': chunk,
                    'K': K,
                    'topk': topk,
                    'template_params': template_params,
                    'route_params': route_params,
                    'active_per_token': active_per_token,
                    'inactive': int(inactive),
                })
        elif isinstance(submod, StandardMoE_MLP):
            K = submod.n_experts
            topk = submod.topk if submod.topk > 0 else K
            if topk < K:
                expert_params = submod.fc_w.numel() + submod.proj_w.numel()
                inactive_frac = 1.0 - (topk / K)
                inactive_expert_params += int(expert_params * inactive_frac)

    if remix_details:
        print(f"\n--- RemixedLinear Template Bank Details ---")
        for rd in remix_details[:4]:  # show first 4
            print(f"  {rd['name']}: K={rd['K']}, chunk={rd['chunk']}, topk={rd['topk']}")
            print(f"    template_params={rd['template_params']:,}, route_params={rd['route_params']:,}")
            print(f"    active_per_token={rd['active_per_token']:,.1f}, inactive={rd['inactive']:,}")
        if len(remix_details) > 4:
            print(f"  ... and {len(remix_details)-4} more RemixedLinear layers")

    manual_active_flops = manual_total_flops - 6 * inactive_expert_params
    manual_active_params = total_params - inactive_expert_params
    print(f"\n  Inactive expert params (manual): {inactive_expert_params:,}")
    print(f"  Manual active_flops = {manual_active_flops:,}")
    match_active = "✅ MATCH" if manual_active_flops == active_flops else f"❌ MISMATCH (diff={manual_active_flops - active_flops:,})"
    print(f"  vs estimate_flops active: {match_active}")
    print(f"  Manual active_params = {manual_active_params:,}")
    match_aparams = "✅ MATCH" if manual_active_params == active_params else f"❌ MISMATCH (diff={manual_active_params - active_params:,})"
    print(f"  vs estimate_flops active_params: {match_aparams}")

    return {
        'total_params': total_params,
        'active_params': active_params,
        'total_flops': total_flops,
        'active_flops': active_flops,
        've_params': ve_params,
        'n_head': config.n_head,
        'n_kv_head': config.n_kv_head,
        'n_embd': config.n_embd,
    }


def main():
    print("=" * 70)
    print("  FLOP/Param Verification: Dense Baseline vs RemixedLinear (chunk64)")
    print("=" * 70)

    # ==========================================================================
    # Q1: Does the dense baseline change n_head/n_kv_head with depth?
    # ==========================================================================
    print("\n\n" + "#" * 70)
    print("# Q1: Dense baseline n_head/n_kv_head across depths")
    print("#" * 70)
    for depth in [4, 6, 8, 12, 16, 24]:
        _, head_dim, model_dim = model_dims(depth, aspect_ratio=64)
        n_head = model_dim // head_dim
        print(f"  depth={depth:>2}: model_dim={model_dim:>5}, head_dim={head_dim}, "
              f"n_head={n_head}, n_kv_head={n_head}")

    # ==========================================================================
    # Q2: Detailed FLOP verification at depth=8
    # ==========================================================================
    print("\n\n" + "#" * 70)
    print("# Q2: Detailed FLOP/Param verification at depth=8 (aspect_ratio=64)")
    print("#" * 70)

    dense_model, dense_cfg = build_dense_model(8, aspect_ratio=64)
    dense_info = analyze_model(dense_model, dense_cfg, "Dense Baseline (depth=8)")

    remix_model, remix_cfg = build_remix_model(8, aspect_ratio=64, n_templates=8, chunk_routing_size=64)
    remix_info = analyze_model(remix_model, remix_cfg, "RemixedLinear chunk64 K=8 (depth=8)")

    # ==========================================================================
    # Q3: Value embed exclusion consistency
    # ==========================================================================
    print("\n\n" + "#" * 70)
    print("# Q3: Value embed exclusion consistency")
    print("#" * 70)
    print(f"  Dense  VE params excluded: {dense_info['ve_params']:,}")
    print(f"  Remix  VE params excluded: {remix_info['ve_params']:,}")
    if dense_info['ve_params'] == remix_info['ve_params']:
        print("  ✅ Both models exclude the same number of VE params from FLOPs")
    else:
        print("  ❌ VE param exclusion differs!")
        print(f"     Dense excludes {dense_info['ve_params']:,} VE params")
        print(f"     Remix excludes {remix_info['ve_params']:,} VE params")

    # ==========================================================================
    # Q4: Cross-depth verification
    # ==========================================================================
    print("\n\n" + "#" * 70)
    print("# Q4: Cross-depth FLOP comparison")
    print("#" * 70)
    print(f"{'depth':>5} | {'Dense total':>15} | {'Dense active':>15} | {'Remix total':>15} | {'Remix active':>15} | {'Ratio (active)':>15}")
    print("-" * 100)
    for depth in [4, 8, 12]:
        d_model, d_cfg = build_dense_model(depth, aspect_ratio=64)
        d_tf, d_af, d_ap = d_model.estimate_flops()
        r_model, r_cfg = build_remix_model(depth, aspect_ratio=64, n_templates=8, chunk_routing_size=64)
        r_tf, r_af, r_ap = r_model.estimate_flops()
        ratio = r_af / d_af if d_af > 0 else float('inf')
        print(f"{depth:>5} | {d_tf:>15,} | {d_af:>15,} | {r_tf:>15,} | {r_af:>15,} | {ratio:>15.4f}")

    # ==========================================================================
    # Q5: What's excluded and what's included in BOTH models
    # ==========================================================================
    print("\n\n" + "#" * 70)
    print("# Q5: What's included/excluded in FLOPs for BOTH models (depth=8)")
    print("#" * 70)
    for label, model, cfg in [("Dense", dense_model, dense_cfg), ("Remix", remix_model, remix_cfg)]:
        total_p = sum(p.numel() for p in model.parameters())
        wte = model.transformer.wte.weight.numel()
        wpe = model.transformer.wpe.weight.numel() if "wpe" in model.transformer else 0
        ve = sum(ve.weight.numel() for ve in model.value_embeds.values())
        rl = model.resid_lambdas.numel()
        xl = model.x0_lambdas.numel()
        excluded = wte + wpe + ve + rl + xl
        included = total_p - excluded
        print(f"\n  {label} model:")
        print(f"    Total params:     {total_p:>12,}")
        print(f"    EXCLUDED:         {excluded:>12,}")
        print(f"      wte:            {wte:>12,}")
        print(f"      wpe:            {wpe:>12,}")
        print(f"      value_embeds:   {ve:>12,}")
        print(f"      resid_lambdas:  {rl:>12,}")
        print(f"      x0_lambdas:     {xl:>12,}")
        print(f"    INCLUDED (matmul):{included:>12,}")
        print(f"      lm_head:        {sum(p.numel() for p in model.lm_head.parameters()):>12,}")
        print(f"      transformer.h:  {sum(p.numel() for p in model.transformer.h.parameters()):>12,}")


if __name__ == "__main__":
    main()
