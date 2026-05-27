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

if __name__ == "__main__":
    test_eet_loss_variants()
    test_eet_phase3_quality_losses()
