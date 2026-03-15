# NVIDIA Blackwell (B200) Compatibility & Optimization Guide

`nanochat` is designed to scale with high-end hardware. Here is how to ensure your setup is optimized for the NVIDIA Blackwell architecture.

## 1. Hardware Detection & MFU
`nanochat` already recognizes Blackwell GPUs in `nanochat/common.py`. Your Model Flops Utilization (MFU) will be calculated against the following peak BF16/FP8 ratings:
- **GB200**: 2.5 PetaFLOPS
- **B200**: 2.25 PetaFLOPS
- **B100**: 1.8 PetaFLOPS

## 2. Flash Attention 3 (FA3)
Blackwell (SM100) and Hopper (SM90) both support the TMA (Tensor Memory Accelerator) required by FA3. 
- `nanochat` now allows Blackwell to attempt loading FA3 kernels.
- If specific Blackwell-optimized kernels are missing, it will gracefully fall back to **PyTorch SDPA**, which is highly optimized via the CUDA driver for new architectures.

## 3. FP8 Training
Blackwell offers **2x higher FP8 throughput** than Hopper.
- **Enabled by default**: Use the `--fp8` flag.
- **Implementation**: Our `nanochat/fp8.py` uses `torch._scaled_mm`, which utilizes the Blackwell-enhanced Tensor Cores automatically.
- **Tip**: Ensure you use `E4M3` for weights/activations and `E5M2` for gradients (this is our default).

## 4. The Next Frontier: FP4
Blackwell introduces a dedicated **FP4 engine**.
- Current Status: `nanochat` does not yet have native FP4 kernels.
- Future Path: When `torchao` or `TensorRT-Model-Optimizer` release stable FP4 quantization for training, it can be integrated into the `Float8Linear` pattern we've established.

## 5. Software Requirements
For full Blackwell support, ensure your environment meets these minimums:
- **CUDA**: 12.4+
- **PyTorch**: 2.4+
- **Driver**: 550.x+

> [!TIP]
> Use `torch.compile(model)` (enabled in our `wrap_model` utility) as it allows the Inductor backend to fuse operations specifically for the Blackwell memory hierarchy.
