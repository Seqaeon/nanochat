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


def test_binary_scale_params_get_a_sane_learning_rate():
    """Avoiding Muon was not enough: the AdamW LR they landed on was the real bug.

    log_alpha and log_g are LOG-parameterised. In research_adamw_params they got
    embedding_lr * dmodel_lr_scale ~= 0.245, and Adam's step magnitude is ~lr, so the
    scale multiplied by exp(0.245) ~= 1.28 EVERY STEP. R5 learned nothing.
    """
    cfg = tiny_config(use_binary=True)
    m = build(cfg)
    opts = m.setup_optimizer()
    opts = opts if isinstance(opts, (list, tuple)) else [opts]
    scale_ids = {id(p) for n, p in m.named_parameters()
                 if n.rsplit(".", 1)[-1] in ("log_alpha", "theta", "log_g")}
    assert scale_ids
    seen = []
    for o in opts:
        for g in o.param_groups:
            if any(id(p) in scale_ids for p in g["params"]):
                seen.append(g["lr"])
    assert seen, "binary scale parameters are in no optimiser group"
    worst = max(seen)
    # exp(lr) is the per-step multiplier on a log-parameterised scale.
    assert worst < 0.05, (
        f"binary scale LR {worst:.4f} multiplies the scale by exp({worst:.4f})="
        f"{math.exp(worst):.3f} per step")


def test_a_collapsed_scale_does_not_kill_the_latent_weight():
    """R5's actual death: loss fell to 7.28, reversed, then pinned at exactly
    ln(32768)=10.3972 forever. w = sign(W)*alpha, so dL/dW is proportional to alpha;
    once a scale reaches zero the latent weight gets no gradient and never recovers.
    SCALE_FLOOR was applied only at init, so nothing stopped log_alpha -> -inf."""
    lin = BinaryLinear(32, 16)
    lin.reset_parameters()
    with torch.no_grad():
        lin.log_alpha.fill_(-60.0)   # exp(-60)=8.8e-27: not zero, but dead in practice
    x = torch.randn(4, 32)
    lin(x).sum().backward()
    # The guarantee is the floor, not an underflow accident: alpha never drops below
    # SCALE_FLOOR however far log_alpha runs, so dL/dW keeps a usable magnitude.
    alpha = lin.log_alpha.exp() + lin.SCALE_FLOOR
    assert float(alpha.min()) == pytest.approx(lin.SCALE_FLOOR, rel=1e-5)
    assert float(lin.binary_weight().abs().min()) >= lin.SCALE_FLOOR * 0.99
    g = lin.weight.grad
    assert g is not None and float(g.abs().max()) > 1e-8, \
        f"latent weight gradient is {float(g.abs().max()):.3e}: layer is dead"


def test_native_block_every_param_gets_gradient():
    """bundle() originally used a hard torch.where and severed the graph outright;
    then the residual threshold was unused at width=1 and sat dead."""
    from nanochat.binary import nativise_model_, BinaryBlock
    cfg = tiny_config(use_binary=True, binary_native=True)
    with torch.device("meta"):
        m = GPT(cfg)
    m.to_empty(device="cpu")
    m.init_weights()
    nativise_model_(m, cfg)
    binarise_model_(m, binarise_acts=cfg.binary_acts)
    assert sum(1 for x in m.modules() if isinstance(x, BinaryBlock)) == cfg.n_layer
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    out = m(x, targets=x)
    (out[0] if isinstance(out, tuple) else out).backward()
    dead = [n for n, p in m.named_parameters()
            if p.requires_grad and (p.grad is None or not p.grad.any())]
    assert not dead, f"no gradient: {dead[:6]}"


def test_native_block_output_is_binary_and_has_no_normalisation():
    from nanochat.binary import BinaryBlock
    cfg = tiny_config()
    blk = BinaryBlock(cfg, 0)
    blk.reset_parameters()
    y = blk(torch.randn(2, 16, cfg.n_embd), None, None, -1, None)
    assert set(y.flatten().tolist()) <= {-1.0, 1.0}, "block output is not binary"
    names = [n for n, _ in blk.named_modules()]
    assert not any("norm" in n.lower() for n in names), \
        "a native binary block should carry no normalisation: sign is scale-free"


def test_chunk_recomputation_is_exact_in_both_directions():
    """Chunking HammingAttention caps the forward peak but not the training peak.

    Softmax backward needs its own output and the value einsum needs the same weights,
    so autograd saves w for every chunk and the chunks sum back to the full
    (B, H, T, T) matrix. That is 30 GB for one layer at B=64, H=28, T=2048, and it
    OOMed an 80 GB H100 on the first b03 attempt. Each chunk is now recomputed in
    backward, which is only legitimate if it changes no number.
    """
    from nanochat.binary import HammingAttention
    torch.manual_seed(0)
    B, T, H, D = 2, 64, 4, 16
    q, k, v = (torch.randn(B, T, H, D, requires_grad=True) for _ in range(3))

    def run(chunk, training):
        for t in (q, k, v):
            t.grad = None
        attn = HammingAttention(H * D, H, tau=1.0, chunk=chunk)
        attn.train(training)
        out = attn(q, k, v)
        out.sum().backward()
        return out.detach().clone(), q.grad.clone(), k.grad.clone()

    o_full, gq_full, gk_full = run(0, True)     # unchunked reference
    o_ck, gq_ck, gk_ck = run(16, True)          # chunked, recomputed in backward
    o_st, gq_st, gk_st = run(16, False)         # chunked, weights stored (no recompute)

    # Recomputation is the thing under test, and it must be bit-exact: same chunking,
    # only the stored-versus-recomputed choice differs.
    assert torch.equal(o_ck, o_st)
    assert torch.equal(gq_ck, gq_st)
    assert torch.equal(gk_ck, gk_st)

    # Chunking itself is NOT bit-exact against the unchunked path, and never was: the
    # key gradient accumulates one contribution per chunk, so the summation order
    # differs. Forward and the query gradient are per-chunk and stay exact.
    assert torch.equal(o_ck, o_full)
    assert torch.equal(gq_ck, gq_full)
    assert torch.allclose(gk_ck, gk_full, atol=1e-6)


def test_model_dim_override_does_not_reach_the_d12_reference():
    """--model-dim must widen the arm under test, never the d12 anchor.

    base_train derives D_REF (and through it the auto batch size and the LR scaling)
    from build_model_meta(12). Letting --model-dim widen that too moved the reference
    and the arm together: at width 3584 it inflated D_REF ~21x, shrank the auto batch
    from 2**19 to 2**17, and killed b03 at the grad-accum assert.
    """
    import ast, pathlib
    src = pathlib.Path("scripts/base_train.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "build_model_meta")
    assert "apply_dim_override" in [a.arg for a in fn.args.args], \
        "build_model_meta lost its override switch"
    call = next(n for n in ast.walk(tree)
                if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "build_model_meta"
                and n.args and getattr(n.args[0], "value", None) == 12)
    kw = {k.arg: k.value.value for k in call.keywords}
    assert kw.get("apply_dim_override") is False, \
        "the d12 reference is being built with --model-dim applied"


def test_soft_gates_are_not_argmax_and_anneal_to_the_hard_path():
    """The native gates divided +-1 dot products by tau with no scale correction.

    Those scores have standard deviation sqrt(fan_in), so tau=1.0 retrieved 1.4 of
    2048 FFN slots and attended 1.8 of 256 keys: every Kanerva read returned a single
    stored vector. Softmax also made the slots COMPETE, so the soft branch never
    converged to the hard one as tau fell; they were different functions.
    """
    import math
    from nanochat.binary import BinaryKVFFN, HammingAttention

    def effective(w):
        p = (w / w.sum(-1, keepdim=True).clamp_min(1e-30)).clamp_min(1e-30)
        return float(torch.exp(-(p * p.log()).sum(-1)).mean())

    torch.manual_seed(0)
    d, slots = 256, 1024
    ffn = BinaryKVFFN(d, slots, tau=1.0)
    ffn.reset_parameters()
    with torch.no_grad():
        xb = torch.sign(torch.randn(2, 32, d))
        scores = torch.nn.functional.linear(xb, torch.sign(ffn.keys)) - ffn.threshold
        soft = torch.sigmoid(scores / math.sqrt(d))
        hard = (scores > 0).float()
    assert effective(soft) > 0.2 * slots, "the FFN is still an argmax lookup table"
    assert effective(hard) > 0.2 * slots

    # tau -> 0 must reach the hard path, which is what makes annealing meaningful.
    q, k, v = (torch.randn(2, 32, 2, 128) for _ in range(3))
    cold = HammingAttention(256, 2, tau=1e-4, hard=False, chunk=0).eval()
    hard_attn = HammingAttention(256, 2, hard=True, chunk=0).eval()
    with torch.no_grad():
        agree = (cold(q, k, v) == hard_attn(q, k, v)).float().mean()
    assert agree > 0.99, f"soft(tau->0) disagrees with hard on {(1-agree)*100:.1f}% of bits"


def test_the_residual_accumulator_carries_the_stream_across_depth():
    """resid_width was a no-op and width=1 destroyed the residual path.

    x and branch are both +-1, so (x + branch).clamp(-width, width) never left
    [-2, 2] and every width behaved as width 1, which is a majority-of-two with a
    constant tiebreak. Measured over eight blocks at depth 8, the output sign agreed
    with the input on 49.8% of bits: chance. The embedding had no path to the head.
    """
    from nanochat.gpt import GPTConfig
    from nanochat.binary import BinaryBlock

    d, layers = 128, 8
    cfg = GPTConfig(sequence_len=32, vocab_size=512, n_layer=layers, n_head=1,
                    n_kv_head=1, n_embd=d, binary_clip=1.0, binary_attn_chunk=0)
    torch.manual_seed(0)
    x0 = torch.sign(torch.randn(2, 32, d))

    def survival(width):
        torch.manual_seed(1)
        x = x0
        with torch.no_grad():
            for i in range(layers):
                blk = BinaryBlock(cfg, i, tau=1.0, resid_width=width)
                blk.reset_parameters()
                x = blk(x, None, None, None, None)
        return float((torch.sign(x) == x0).float().mean())

    assert survival(1) < 0.6, "width=1 is expected to destroy the stream; it is the baseline"
    wide = survival(max(2, layers // 2))
    assert wide > 0.7, f"the derived width only carries {wide*100:.1f}% of the stream"
    assert wide < 0.99, "the stream is frozen: no branch can change a saturated channel"
