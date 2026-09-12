"""Tests for the binary layers and their base_train integration.

Each test here exists because the corresponding failure actually happened and was
expensive, not because the behaviour seemed worth asserting in the abstract.
"""
import math

import pytest
import torch
import torch.nn as nn

from nanochat.binary import BinaryLinear, BinaryEmbedding, binarise_model_
from nanochat.gpt import GPT, GPTConfig


def tiny_config(**kw):
    base = dict(sequence_len=128, vocab_size=128, n_layer=2, n_head=1, n_kv_head=1,
                n_embd=128, window_pattern="L")
    base.update(kw)
    return GPTConfig(**base)


def build(cfg, device="cpu"):
    with torch.device("meta"):
        m = GPT(cfg)
    m.to_empty(device=device)
    m.init_weights()
    if cfg.use_binary:
        binarise_model_(m, binarise_acts=cfg.binary_acts,
                        weight_scale=cfg.binary_weight_scale,
                        skip=tuple(x for x in cfg.binary_skip.split(",") if x))
    return m


def test_log_alpha_is_1d_so_it_does_not_route_to_muon():
    """(out,1) is ndim==2 and setup_optimizer sends every ndim==2 param to Muon.

    Newton-Schulz on an (N,1) matrix gives every channel the same update magnitude,
    optim.py:448 inflates the LR by sqrt(out/in) (32x at out=1024), and Muon applies
    weight decay while every AdamW group here uses 0.0.
    """
    lin = BinaryLinear(64, 32)
    assert lin.log_alpha.ndim == 1
    assert lin.log_alpha.shape == (32,)


def test_binary_params_land_in_adamw_not_muon():
    cfg = tiny_config(use_binary=True)
    m = build(cfg)
    opts = m.setup_optimizer()
    opts = opts if isinstance(opts, (list, tuple)) else [opts]
    scale_ids = {id(p) for n, p in m.named_parameters()
                 if n.rsplit(".", 1)[-1] in ("log_alpha", "theta", "log_g")}
    assert scale_ids, "no binary scale parameters found"
    # The optimiser is a FUSED MuonAdamW, so its class name is not the test: each
    # param_group carries kind='adamw' or 'muon' (nanochat/optim.py:268).
    offenders = []
    for o in opts:
        for g in o.param_groups:
            if g.get("kind") != "muon":
                continue
            for p in g["params"]:
                if id(p) in scale_ids:
                    offenders.append(next(n for n, q in m.named_parameters() if q is p))
    assert not offenders, f"binary scale parameters reached Muon: {offenders}"


def test_every_tensor_receives_gradient_within_ten_steps():
    """The O6 gate. Checking at step 0 is wrong: nanochat zero-initialises every
    c_proj, which blocks gradient to c_q/c_k/c_v/c_fc inside that block for exactly
    one backward. What matters is whether a tensor EVER receives one."""
    cfg = tiny_config(use_binary=True)
    m = build(cfg)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    ever = set()
    for _ in range(10):
        x = torch.randint(0, cfg.vocab_size, (2, 64))
        out = m(x, targets=x)
        loss = out[0] if isinstance(out, tuple) else out
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for n, p in m.named_parameters():
            if p.requires_grad and p.grad is not None and p.grad.any():
                ever.add(n)
        opt.step()
    never = [n for n, p in m.named_parameters() if p.requires_grad and n not in ever]
    assert not never, f"never received gradient: {never[:8]}"


def test_zero_initialised_rows_are_not_frozen():
    """A detached mean|W| scale makes a zero row output zero AND receive zero
    gradient, so every c_proj was dead from step 0 and unrecoverable."""
    donor = nn.Linear(32, 16, bias=False)
    with torch.no_grad():
        donor.weight.zero_()
    holder = nn.Sequential(donor)
    binarise_model_(holder)
    lin = holder[0]
    x = torch.randn(4, 32)
    lin(x).sum().backward()
    assert lin.weight.grad is not None and lin.weight.grad.any(), "zero row still frozen"
    assert lin.binary_weight().abs().sum() > 0, "zero row still emits zeros"


def test_num_scaling_params_assert_holds_with_binary():
    m = build(tiny_config(use_binary=True))
    counts = m.num_scaling_params()  # asserts internally
    assert counts["total"] == sum(p.numel() for p in m.parameters())


def test_scales_are_excluded_from_the_flops_proxy():
    """log_alpha is an elementwise rescale of the output, not a matmul weight, so
    the 6N proxy would overcharge it."""
    dense = build(tiny_config())
    binary = build(tiny_config(use_binary=True))
    assert binary.estimate_flops()[0] == pytest.approx(dense.estimate_flops()[0])


def test_binary_flops_equal_dense_flops():
    """The whole reason section 3.1 changes the cost axis: binarisation changes what
    an operation costs, not how many there are."""
    dense = build(tiny_config())
    binary = build(tiny_config(use_binary=True))
    assert binary.estimate_flops()[0] == pytest.approx(dense.estimate_flops()[0])


@pytest.mark.parametrize("mode", ["row", "none", "threshold"])
def test_scale_modes_run_and_only_row_and_threshold_control_magnitude(mode):
    torch.manual_seed(0)
    lin = BinaryLinear(256, 256, weight_scale=mode)
    lin.reset_parameters()
    y = lin(torch.randn(8, 256))
    std = float(y.std().detach())
    if mode == "none":
        assert std > 5.0, "strict no-scale should blow the output up; that is the point"
    else:
        assert std < 2.0, f"{mode} failed to control output magnitude: std {std}"


def test_ladder_skip_leaves_named_modules_dense():
    cfg = tiny_config(use_binary=True, binary_skip="lm_head")
    m = build(cfg)
    assert not isinstance(m.lm_head, BinaryLinear), "skip did not spare lm_head"
    inner = [mod for mod in m.transformer.h.modules() if isinstance(mod, BinaryLinear)]
    assert inner, "skip spared the body too"
