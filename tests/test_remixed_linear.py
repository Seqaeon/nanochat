import torch
import torch.nn as nn
from nanochat.gpt import GPT, GPTConfig, RemixedLinear, Linear

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

def test_remixed_linear_init():
    print("\nTesting RemixedLinear initialization...")
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
    for m in model.modules():
        if isinstance(m, RemixedLinear):
            remixed_layer = m
            break
    
    assert remixed_layer is not None, "Could not find RemixedLinear layer in model"
    
    # Check if context_modulator final bias is 2.0
    last_linear = [m for m in remixed_layer.context_modulator.modules() if isinstance(m, (Linear, nn.Linear))][-1]
    bias_val = last_linear.bias.data.mean().item()
    print(f"Modulator final bias mean: {bias_val}")
    assert abs(bias_val - 2.0) < 1e-5, f"RemixedLinear modulator bias initialization failed! Expected 2.0, got {bias_val}"
    
    # Perform a forward pass
    x = torch.randn(1, 4, 32)
    context = torch.randn(1, 4, 16)
    y = remixed_layer(x, context)
    print(f"Forward pass output shape: {y.shape}")
    assert y.shape == (1, 4, 4 * 32) or y.shape == (1, 4, 32), f"Unexpected output shape: {y.shape}"
    print("RemixedLinear initialization and forward pass verified!")

if __name__ == "__main__":
    test_linear_bias_fix()
    test_remixed_linear_init()
