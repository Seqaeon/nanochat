import torch
from nanochat.gpt import GPTConfig
from nanochat.eet import EarlyExitGPT

def test_eet_loss_variants():
    device = "cpu"
    print(f"Running EET Loss Variants smoke test on {device}...")
    
    # 1. Create a tiny EET config
    config = GPTConfig(
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        vocab_size=128,
        sequence_len=32,
        use_eet=True,
        eet_min_exit_layer=0,  # allow exiting early from layer 0
        eet_exit_threshold=0.0, # force exits to test the exit loss pathways
        eet_topk_vocab=16,
    )
    
    # Instantiate the model
    model = EarlyExitGPT(config).to(device)
    model.train()
    
    # Create tiny dummy input and target tensors
    B, T = 2, 8
    x = torch.randint(0, config.vocab_size, (B, T), device=device)
    y = torch.randint(0, config.vocab_size, (B, T), device=device)
    
    variants = ['reconstruct', 'entropy_surprise', 'adversarial', 'quality']
    
    for variant in variants:
        print(f"\n--- Testing EET Loss Variant: {variant} ---")
        config.eet_loss_variant = variant
        
        # Zero out existing gradients
        model.zero_grad(set_to_none=True)
        
        # Run forward pass simulating Phase 2
        loss = model(
            x, y,
            eet_do_route=True,
            eet_phase=2,
            eet_lambda_r=torch.tensor(1.0, device=device),
            eet_lambda_e=torch.tensor(0.1, device=device)
        )
        
        # Verify the loss is not NaN or Infinite
        loss_val = loss.item()
        print(f"Loss computed successfully: {loss_val}")
        assert not torch.isnan(loss), f"Loss is NaN for variant {variant}"
        assert not torch.isinf(loss), f"Loss is Infinite for variant {variant}"
        
        # Verify the backward pass works
        loss.backward()
        print(f"Backward pass successful for variant {variant}!")
        
        # Check that we populated gradients on router parameters
        router_has_grad = False
        for name, p in model.named_parameters():
            if "eet_router" in name and p.grad is not None:
                router_has_grad = True
                assert not torch.isnan(p.grad).any(), f"NaN gradient detected in {name} for variant {variant}"
                assert not torch.isinf(p.grad).any(), f"Infinite gradient detected in {name} for variant {variant}"
        
        # Note: If no token exited because of threshold initialization, router gradients might not flow,
        # but we successfully backwarded the main cross-entropy and general pathways.
        print(f"Gradient check complete for variant {variant}. Router parameters updated: {router_has_grad}")


def test_eet_phase3_quality_losses():
    device = "cpu"
    print(f"\n--- Testing EET Loss Phase 3 Quality Hybrid Gradient Flow ---")
    
    config = GPTConfig(
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        vocab_size=128,
        sequence_len=32,
        use_eet=True,
        eet_min_exit_layer=0,
        eet_exit_threshold=0.5, # standard threshold
        eet_topk_vocab=16,
        eet_loss_variant='quality',
        eet_quality_entropy_bonus=0.1,
    )
    
    model = EarlyExitGPT(config).to(device)
    model.train()
    
    B, T = 2, 8
    x = torch.randint(0, config.vocab_size, (B, T), device=device)
    y = torch.randint(0, config.vocab_size, (B, T), device=device)
    
    model.zero_grad(set_to_none=True)
    
    # Run forward pass simulating Phase 3 (hard routing)
    loss = model(
        x, y,
        eet_do_route=True,
        eet_phase=3,
        eet_lambda_r=torch.tensor(1.0, device=device),
        eet_lambda_e=torch.tensor(0.1, device=device)
    )
    
    loss_val = loss.item()
    print(f"Phase 3 Loss computed successfully: {loss_val}")
    assert not torch.isnan(loss), "Phase 3 Loss is NaN"
    
    loss.backward()
    print("Backward pass successful for Phase 3!")
    
    # Verify that the router parameters got gradients populated!
    router_has_grad = False
    for name, p in model.named_parameters():
        if "eet_router" in name and p.grad is not None:
            router_has_grad = True
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
            print(f"  {name} has grad. norm: {p.grad.norm().item():.6f}")
            
    assert router_has_grad, "Error: Router did not receive gradients in Phase 3 training!"
    print("✓ Phase 3 hybrid gradient path is fully functional!")

def test_eet_gumbel_routing():
    device = "cpu"
    print(f"\n--- Testing EET Gumbel-Softmax Routing with Annealing & Commitment ---")
    
    config = GPTConfig(
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        vocab_size=128,
        sequence_len=32,
        use_eet=True,
        eet_min_exit_layer=0,
        eet_gumbel_temp_start=2.0,
        eet_gumbel_temp_end=0.5,
        eet_gumbel_hard=True,
        eet_commitment_beta=0.1,
        eet_loss_variant='quality',
    )
    
    model = EarlyExitGPT(config).to(device)
    model.train()
    
    B, T = 2, 8
    x = torch.randint(0, config.vocab_size, (B, T), device=device)
    y = torch.randint(0, config.vocab_size, (B, T), device=device)
    
    model.zero_grad(set_to_none=True)
    
    # Run forward pass with Gumbel temp 1.5
    loss = model(
        x, y,
        eet_do_route=True,
        eet_phase=2,
        eet_lambda_r=torch.tensor(1.0, device=device),
        eet_lambda_e=torch.tensor(0.1, device=device),
        eet_gumbel_temp=1.5
    )
    
    loss_val = loss.item()
    print(f"Gumbel Loss computed successfully: {loss_val}")
    assert not torch.isnan(loss), "Gumbel Loss is NaN"
    
    loss.backward()
    print("Backward pass successful for Gumbel!")
    
    # Verify that the router parameters got gradients populated!
    router_has_grad = False
    for name, p in model.named_parameters():
        if "eet_router" in name and p.grad is not None:
            router_has_grad = True
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
            print(f"  {name} has grad. norm: {p.grad.norm().item():.6f}")
            
    assert router_has_grad, "Error: Router did not receive gradients in Gumbel training!"
    print("✓ Gumbel-Softmax routing path is fully functional!")


def test_eet_layer_weighted_routing():
    device = "cpu"
    print(f"\n--- Testing EET Per-Token Layer-Weighted Loss & Commitment ---")
    
    config = GPTConfig(
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        vocab_size=128,
        sequence_len=32,
        use_eet=True,
        eet_min_exit_layer=0,
        eet_commitment_beta=0.1,
        eet_loss_variant='layer_weighted',
    )
    
    model = EarlyExitGPT(config).to(device)
    model.train()
    
    B, T = 2, 8
    x = torch.randint(0, config.vocab_size, (B, T), device=device)
    y = torch.randint(0, config.vocab_size, (B, T), device=device)
    
    model.zero_grad(set_to_none=True)
    
    # Run forward pass simulating layer-weighted loss
    loss = model(
        x, y,
        eet_do_route=True,
        eet_phase=2,
        eet_lambda_r=torch.tensor(1.0, device=device),
        eet_lambda_e=torch.tensor(0.1, device=device)
    )
    
    loss_val = loss.item()
    print(f"Layer-Weighted Loss computed successfully: {loss_val}")
    assert not torch.isnan(loss), "Layer-Weighted Loss is NaN"
    
    loss.backward()
    print("Backward pass successful for Layer-Weighted!")
    
    # Verify that BOTH the router parameters and backbone got gradients populated!
    router_has_grad = False
    backbone_has_grad = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            if "eet_router" in name:
                router_has_grad = True
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
                print(f"  Router {name} has grad. norm: {p.grad.norm().item():.6f}")
            elif "transformer.h.0" in name:
                backbone_has_grad = True
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
                print(f"  Backbone {name} has grad. norm: {p.grad.norm().item():.6f}")
            
    assert router_has_grad, "Error: Router did not receive gradients in Layer-Weighted training!"
    assert backbone_has_grad, "Error: Backbone did not receive gradients in Layer-Weighted training!"
    print("✓ Per-Token Layer-Weighted routing path is fully functional!")


def test_eet_global_router():
    device = "cpu"
    print(f"\n--- Testing EET Global Upfront Exit Router ---")
    
    # Test 1: Global Router + Gumbel-Softmax + Quality Loss
    config = GPTConfig(
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        vocab_size=128,
        sequence_len=32,
        use_eet=True,
        eet_min_exit_layer=0,
        eet_gumbel_temp_start=2.0,
        eet_gumbel_temp_end=0.5,
        eet_gumbel_hard=True,
        eet_commitment_beta=0.1,
        eet_loss_variant='quality',
        eet_global_router=True,
    )
    
    model = EarlyExitGPT(config).to(device)
    model.train()
    
    B, T = 2, 8
    x = torch.randint(0, config.vocab_size, (B, T), device=device)
    y = torch.randint(0, config.vocab_size, (B, T), device=device)
    
    model.zero_grad(set_to_none=True)
    loss = model(
        x, y,
        eet_do_route=True,
        eet_phase=2,
        eet_lambda_r=torch.tensor(1.0, device=device),
        eet_lambda_e=torch.tensor(0.1, device=device),
        eet_gumbel_temp=1.5
    )
    
    assert not torch.isnan(loss), "Global Router + Quality Loss is NaN"
    loss.backward()
    
    router_has_grad = False
    for name, p in model.named_parameters():
        if "eet_router" in name and p.grad is not None:
            router_has_grad = True
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
            print(f"  [Quality] Router {name} has grad. norm: {p.grad.norm().item():.6f}")
            
    assert router_has_grad, "Error: Global Router did not receive gradients in Quality mode!"
    print("✓ Global Upfront Router with Gumbel/Quality is fully functional!")
    
    # Test 2: Global Router + Layer-Weighted Loss
    config_lw = GPTConfig(
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        vocab_size=128,
        sequence_len=32,
        use_eet=True,
        eet_min_exit_layer=0,
        eet_commitment_beta=0.1,
        eet_loss_variant='layer_weighted',
        eet_global_router=True,
    )
    
    model_lw = EarlyExitGPT(config_lw).to(device)
    model_lw.train()
    
    model_lw.zero_grad(set_to_none=True)
    loss_lw = model_lw(
        x, y,
        eet_do_route=True,
        eet_phase=2,
        eet_lambda_r=torch.tensor(1.0, device=device),
        eet_lambda_e=torch.tensor(0.1, device=device)
    )
    
    assert not torch.isnan(loss_lw), "Global Router + Layer-Weighted Loss is NaN"
    loss_lw.backward()
    
    router_has_grad = False
    backbone_has_grad = False
    for name, p in model_lw.named_parameters():
        if p.grad is not None:
            if "eet_router" in name:
                router_has_grad = True
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
                print(f"  [Layer-Weighted] Router {name} has grad. norm: {p.grad.norm().item():.6f}")
            elif "transformer.h.0" in name:
                backbone_has_grad = True
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
                print(f"  [Layer-Weighted] Backbone {name} has grad. norm: {p.grad.norm().item():.6f}")
                
    assert router_has_grad, "Error: Global Router did not receive gradients in Layer-Weighted mode!"
    assert backbone_has_grad, "Error: Backbone did not receive gradients in Layer-Weighted mode!"
    print("✓ Global Upfront Router with Layer-Weighted loss is fully functional!")


def test_eet_freq_efficiency_and_diversity():
    """Test per-token frequency-scaled efficiency loss and exit diversity pressure."""
    from nanochat.eet import compute_efficiency_and_diversity
    device = "cpu"
    print(f"\nRunning frequency-scaled efficiency & diversity pressure test on {device}...")
    
    B, T, n_exits = 4, 16, 5
    
    # Create fake p_exits that sum to ~1 per token
    raw = torch.randn(B, T, n_exits).softmax(dim=-1)
    p_exits = [raw[:, :, k] for k in range(n_exits)]
    
    # Create fake freq_bias: half tokens are high-freq, half low-freq
    freq_bias = torch.zeros(B, T)
    freq_bias[:, :T//2] = 0.9   # high frequency
    freq_bias[:, T//2:] = 0.1   # low frequency
    
    # Test 1: Uniform efficiency (baseline, no freq scaling, no diversity)
    config_baseline = GPTConfig(
        n_head=2, n_kv_head=2, n_embd=16, vocab_size=128, sequence_len=32,
        use_eet=True,
        eet_freq_efficiency_alpha=0.0,
        eet_diversity_lambda=0.0,
    )
    eet_lambda_e = torch.tensor(0.1)
    
    loss_baseline, diag_baseline = compute_efficiency_and_diversity(
        p_exits, n_exits, freq_bias, config_baseline, eet_lambda_e
    )
    assert not torch.isnan(loss_baseline), "Baseline efficiency loss is NaN"
    print(f"  Baseline efficiency loss: {loss_baseline.item():.6f}")
    print(f"  Expected exit: {diag_baseline['expected_exit'].item():.4f}")
    
    # Test 2: With freq scaling (alpha=2.0)
    config_freq = GPTConfig(
        n_head=2, n_kv_head=2, n_embd=16, vocab_size=128, sequence_len=32,
        use_eet=True,
        eet_freq_efficiency_alpha=2.0,
        eet_diversity_lambda=0.0,
    )
    
    loss_freq, diag_freq = compute_efficiency_and_diversity(
        p_exits, n_exits, freq_bias, config_freq, eet_lambda_e
    )
    assert not torch.isnan(loss_freq), "Freq-scaled efficiency loss is NaN"
    assert loss_freq > loss_baseline, \
        f"Freq-scaled loss ({loss_freq.item():.6f}) should be > baseline ({loss_baseline.item():.6f}) since freq_weight > 1"
    print(f"  Freq-scaled efficiency loss: {loss_freq.item():.6f} (> baseline ✓)")
    
    # Test 3: With diversity pressure (lambda=0.1)
    config_div = GPTConfig(
        n_head=2, n_kv_head=2, n_embd=16, vocab_size=128, sequence_len=32,
        use_eet=True,
        eet_freq_efficiency_alpha=0.0,
        eet_diversity_lambda=0.1,
    )
    
    loss_div, diag_div = compute_efficiency_and_diversity(
        p_exits, n_exits, freq_bias, config_div, eet_lambda_e
    )
    assert not torch.isnan(loss_div), "Diversity-augmented loss is NaN"
    assert 'diversity' in diag_div, "Diversity diagnostic missing"
    print(f"  Diversity-augmented loss: {loss_div.item():.6f}")
    print(f"  Exit std (diversity): {diag_div['diversity'].item():.4f}")
    
    # Test 4: Gradient check — diversity loss should flow back through p_exits
    p_exits_grad = [p.clone().requires_grad_(True) for p in p_exits]
    config_both = GPTConfig(
        n_head=2, n_kv_head=2, n_embd=16, vocab_size=128, sequence_len=32,
        use_eet=True,
        eet_freq_efficiency_alpha=2.0,
        eet_diversity_lambda=0.1,
    )
    loss_both, _ = compute_efficiency_and_diversity(
        p_exits_grad, n_exits, freq_bias, config_both, eet_lambda_e
    )
    loss_both.backward()
    
    grads_ok = all(p.grad is not None and not torch.isnan(p.grad).any() for p in p_exits_grad)
    assert grads_ok, "Gradients through freq-scaled efficiency + diversity are invalid"
    print(f"  Combined loss: {loss_both.item():.6f} — gradients flow correctly ✓")
    
    # Test 5: End-to-end with a real model using layer_weighted + freq scaling + diversity
    config_e2e = GPTConfig(
        n_head=2, n_kv_head=2, n_embd=16, vocab_size=128, sequence_len=32,
        use_eet=True,
        eet_min_exit_layer=0,
        eet_loss_variant='layer_weighted',
        eet_commitment_beta=0.1,
        eet_freq_efficiency_alpha=2.0,
        eet_diversity_lambda=0.1,
    )
    model = EarlyExitGPT(config_e2e).to(device)
    model.train()
    
    x = torch.randint(0, config_e2e.vocab_size, (2, 8), device=device)
    y = torch.randint(0, config_e2e.vocab_size, (2, 8), device=device)
    
    model.zero_grad(set_to_none=True)
    loss = model(x, y, eet_do_route=True, eet_phase=2,
                 eet_lambda_r=torch.tensor(1.0, device=device),
                 eet_lambda_e=torch.tensor(0.1, device=device))
    
    assert not torch.isnan(loss), "E2E layer_weighted + freq + diversity loss is NaN"
    loss.backward()
    
    router_has_grad = any(
        p.grad is not None and p.grad.norm() > 0
        for name, p in model.named_parameters() if "eet_router" in name
    )
    assert router_has_grad, "Router should have gradients with freq-scaled efficiency + diversity"
    print(f"  E2E loss: {loss.item():.6f} — router receives differentiated gradients ✓")
    
    print("✓ Frequency-scaled efficiency + exit diversity pressure fully functional!")


def test_eet_sigmoid_temp_annealing():
    """Verify that sigmoid temperature annealing produces expected soft vs sharp behavior."""
    from nanochat.eet import EarlyExitRouter
    device = "cpu"
    print(f"\nRunning sigmoid temperature annealing test on {device}...")
    
    router = EarlyExitRouter(n_embd=16, router_type='linear').to(device)
    
    # Create fake hidden state
    h = torch.randn(2, 4, 16, device=device)
    
    # Under high temp (e.g. 10.0), sigmoid should be closer to 0.5 (flatter)
    p_high = router(h, temp=10.0)
    dev_high = torch.abs(p_high - 0.5).mean().item()
    print(f"  Mean deviation from 0.5 under High Temp (10.0): {dev_high:.4f}")
    
    # Under low temp (e.g. 0.1), sigmoid should be pushed closer to 0.0 or 1.0 (sharper)
    p_low = router(h, temp=0.1)
    dev_low = torch.abs(p_low - 0.5).mean().item()
    print(f"  Mean deviation from 0.5 under Low Temp (0.1):  {dev_low:.4f}")
    
    assert dev_low > dev_high, "Low temperature should sharpen decisions (larger deviation from 0.5)"
    
    # E2E forward test
    config = GPTConfig(
        n_head=2, n_kv_head=2, n_embd=16, vocab_size=128, sequence_len=32,
        use_eet=True,
        eet_min_exit_layer=0,
        eet_loss_variant='entropy_surprise',
        eet_gumbel_temp_start=5.0,
        eet_gumbel_temp_end=0.1,
    )
    model = EarlyExitGPT(config).to(device)
    model.train()
    
    x = torch.randint(0, config.vocab_size, (2, 8), device=device)
    y = torch.randint(0, config.vocab_size, (2, 8), device=device)
    
    # Run with low temp
    loss_low_temp = model(x, y, eet_do_route=True, eet_phase=2, eet_gumbel_temp=0.1)
    assert not torch.isnan(loss_low_temp), "E2E forward with low temp failed"
    
    # Run with high temp
    loss_high_temp = model(x, y, eet_do_route=True, eet_phase=2, eet_gumbel_temp=10.0)
    assert not torch.isnan(loss_high_temp), "E2E forward with high temp failed"
    
    print("✓ Sigmoid temperature annealing is functional and correct!")


if __name__ == "__main__":
    test_eet_loss_variants()
    test_eet_phase3_quality_losses()
    test_eet_gumbel_routing()
    test_eet_layer_weighted_routing()
    test_eet_global_router()
    test_eet_freq_efficiency_and_diversity()
    test_eet_sigmoid_temp_annealing()

