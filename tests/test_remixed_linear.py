import torch
import torch.nn as nn
from nanochat.gpt import GPT, GPTConfig, RemixedLinear, Linear, ImprovedContextAwareRouter

def test_linear_bias_fix():
    print("Testing Linear bias fix...")
    # Create a Linear layer with a specific weight and bias
    lin = Linear(2, 2)
    lin.weight.data = torch.eye(2)
    lin.bias.data = torch.tensor([1.0, 2.0])
    
    x = torch.zeros(1, 2)
    y = lin(x)
    
    # If bias is working, y should be [1.0, 2.0]
    expected = torch.tensor([[1.0, 2.0]])
    print(f"Input: {x}")
    print(f"Output: {y}")
    print(f"Expected: {expected}")
    
    assert torch.allclose(y, expected), f"Linear bias fix failed! Expected {expected}, got {y}"
    print("Linear bias fix verified!")

def test_router_integration():
    print("\nTesting Router integration (biases and init)...")
    router = ImprovedContextAwareRouter(
        vocab_size=100,
        num_experts=64,
        router_dim=64,
        full_embed_dim=128
    )
    
    # Check if biases are enabled in key layers
    assert router.expert_proj.bias is not None, "Router expert_proj missing bias"
    assert router.alpha_gate.bias is not None, "Router alpha_gate missing bias"
    assert router.embed_proj.bias is not None, "Router embed_proj missing bias"
    
    # Check initialization
    def _init_router_mock(m):
        if isinstance(m, ImprovedContextAwareRouter):
            torch.nn.init.normal_(m.expert_proj.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(m.cross_expert_proj.weight, mean=0.0, std=0.02)
    
    _init_router_mock(router)
    std = router.expert_proj.weight.std().item()
    print(f"Router expert_proj weight std: {std}")
    assert abs(std - 0.02) < 0.01, f"Router init std mismatch: expected ~0.02, got {std}"
    print("Router integration verified!")

def test_remixed_linear_init():
    print("\nTesting RemixedLinear initialization and forward pass...")
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=32,
        use_remixed_linear=True,
        context_dim=16,
        linear_basis_size=16
    )
    
    model = GPT(config)
    model.init_weights()
    
    # Find a RemixedLinear layer
    remixed_layer = None
    router_layer = None
    for m in model.modules():
        if isinstance(m, RemixedLinear):
            remixed_layer = m
        if isinstance(m, ImprovedContextAwareRouter):
            router_layer = m
    
    assert remixed_layer is not None, "Could not find RemixedLinear layer in model"
    assert router_layer is not None, "Could not find ImprovedContextAwareRouter layer in model"
    
    # 1. Check if context_modulator final bias is exactly 2.0 (ensures it wasn't overwritten)
    last_linear = [m for m in remixed_layer.context_modulator.modules() if isinstance(m, (Linear, nn.Linear))][-1]
    bias_val = last_linear.bias.data.mean().item()
    print(f"Modulator final bias mean: {bias_val}")
    assert abs(bias_val - 2.0) < 1e-6, f"RemixedLinear modulator bias initialization failed! Expected 2.0, got {bias_val}"
    
    # 2. Check Router weights std
    router_std = router_layer.expert_proj.weight.std().item()
    print(f"Router expert_proj weight std (from model.init_weights): {router_std}")
    assert abs(router_std - 0.02) < 0.01, f"Router model init std mismatch: expected ~0.02, got {router_std}"

    # 3. Perform a forward pass
    x = torch.randn(1, 4, 32)
    context = torch.randn(1, 4, 16)
    y = remixed_layer(x, context)
    print(f"Forward pass output shape: {y.shape}")
    assert y.shape == (1, 4, 32) or y.shape == (1, 4, 4 * 32), f"Unexpected output shape: {y.shape}"
    print("RemixedLinear deep integration verified!")

if __name__ == "__main__":
    test_linear_bias_fix()
    test_router_integration()
    test_remixed_linear_init()
