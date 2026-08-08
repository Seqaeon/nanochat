"""Smoke tests for the MST dense-parity fixes (G1/G2/G3) and overhead cuts (O1/O2).

Each fix is meant to close a gap between MST and the dense baseline without
changing the FLOP budget. The properties worth proving:

  G1 (mst_sub_head_dim): head_dim moves 32 -> 64/128 while matrix parameters and
      FLOPs/token stay put, and the batched layer path is still selected.
  G2 (mst_final_norm):   the unembedding input is normalized, so its RMS is ~1.
  G3 (mst_per_stream_ve): the value-embedding table widens to N*d and each stream
      reads a different slice, instead of all streams sharing one vector.

The Stage 13 cuts are the opposite: they are meant to REDUCE FLOPs while leaving
the trunk's matrix parameters alone, because MST is at parity with dense per
matrix parameter and its whole FLOPs deficit is D-proportional overhead.

  O1 (mst_lm_head_dim):    the output head factorizes as D -> Dh -> V.
  O2 (mst_compose_windows): per-sub windows intersect the layer window pattern
      instead of replacing it, so the widest stream is not full-context at every
      layer while dense gets full context on one layer in four.

Run:  pytest tests/test_mst_parity_fixes.py -q
"""
import contextlib
import io

import pytest
import torch

from nanochat.gpt import GPTConfig
from nanochat.mst import MST, resolve_sub_heads

N_SUBS = 4
VOCAB = 512
# Must exceed the 256-token "short" window that _compute_window_sizes hardcodes,
# or every layer is effectively full-context and O2 has nothing to compose.
SEQ = 512


def make_config(depth=4, D=256, **overrides):
    d = D // N_SUBS
    cfg = dict(
        sequence_len=SEQ, vocab_size=VOCAB, n_layer=depth,
        n_head=D // 128, n_kv_head=D // 128, n_embd=D, window_pattern="SSSL",
        use_mst=True, mst_n_subs=N_SUBS, mst_sub_dim=d, mst_head_dim=0,
        mst_input_mode='learned_proj',
        mst_routing_mode='soft_weighted', mst_routing_topk=0,
        mst_ffn_mode='standard', mst_transition_mode='aggregate_distribute',
        mst_final_mode='concat_proj', mst_final_topk=0,
        mst_routing_aux_weight=0.01, mst_diversity_weight=0.0,
        mst_grad_equalize=1, mst_block_diagonal_muon=1,
        mst_transition_width_mult=float(N_SUBS), mst_sub_lr_scale=2.0,
        mst_multi_scale_windows=1,
    )
    cfg.update(overrides)
    return GPTConfig(**cfg)


def build_meta(**overrides):
    """Shapes only: no allocation, no GPU."""
    with contextlib.redirect_stdout(io.StringIO()), torch.device('meta'):
        return MST(make_config(**overrides))


# ── G1 ───────────────────────────────────────────────────────────────────────

def test_g1_default_head_dim_is_the_narrow_legacy_one():
    cfg = make_config()
    n_head, head_dim = resolve_sub_heads(cfg)
    # D=256, n_head=2, d=64 -> 2 heads of 32. The 32 is what dense pays 128 for.
    assert (n_head, head_dim) == (2, 32)
    assert n_head * head_dim == cfg.mst_sub_dim


@pytest.mark.parametrize("sub_head_dim", [16, 32, 64])
def test_g1_pins_head_dim_and_keeps_qkv_width(sub_head_dim):
    cfg = make_config(mst_sub_head_dim=sub_head_dim)
    n_head, head_dim = resolve_sub_heads(cfg)
    assert head_dim == sub_head_dim
    assert n_head * head_dim == cfg.mst_sub_dim, "qkv_dim must stay == d to be FLOP-neutral"


def test_g1_is_parameter_and_flop_neutral():
    """The whole premise of G1: better head geometry for free."""
    base = build_meta()
    wide = build_meta(mst_sub_head_dim=64)

    b_matrices = base.num_scaling_params()['transformer_matrices']
    w_matrices = wide.num_scaling_params()['transformer_matrices']
    b_flops = base.estimate_flops()[0]
    w_flops = wide.estimate_flops()[0]

    # Only the VE gates change shape ((N*n_head, 32)), which is negligible.
    assert abs(w_matrices - b_matrices) / b_matrices < 1e-3
    assert abs(w_flops - b_flops) / b_flops < 1e-3
    assert base._use_batched and wide._use_batched, "batched path must survive G1"


def test_g1_rejects_indivisible_head_dim():
    with pytest.raises(AssertionError, match="must divide"):
        resolve_sub_heads(make_config(mst_sub_head_dim=48))  # d=64 not divisible by 48


# ── G2 ───────────────────────────────────────────────────────────────────────

def test_g2_normalizes_the_unembedding_input():
    """With the fix, lm_head sees unit-RMS rows; without it, an unconstrained scale."""
    captured = {}

    def run(final_norm):
        torch.manual_seed(0)
        with contextlib.redirect_stdout(io.StringIO()):
            model = MST(make_config(mst_final_norm=final_norm))
            model.init_weights()
        model.eval()
        hook = model.lm_head.register_forward_pre_hook(
            lambda _m, args: captured.__setitem__('h', args[0].detach().float()))
        with torch.no_grad():
            model(torch.randint(0, VOCAB, (2, SEQ)))
        hook.remove()
        h = captured['h']
        return h.pow(2).mean(dim=-1).sqrt()

    rms_on = run(1)
    assert torch.allclose(rms_on, torch.ones_like(rms_on), atol=1e-2), \
        f"G2 on: expected unit RMS into lm_head, got {rms_on.mean():.4f}"
    rms_off = run(0)
    assert not torch.allclose(rms_off, torch.ones_like(rms_off), atol=1e-2), \
        "G2 off should leave the legacy unnormalized scale"


# ── G3 ───────────────────────────────────────────────────────────────────────

def test_g3_widens_ve_table_without_touching_flops():
    base = build_meta()
    per_stream = build_meta(mst_per_stream_ve=1)

    b_ve = base.num_scaling_params()['value_embeds']
    p_ve = per_stream.num_scaling_params()['value_embeds']
    assert p_ve == N_SUBS * b_ve, "each stream should get its own d-wide slice"

    # VE is a lookup, so widening it must not move FLOPs/token.
    assert abs(per_stream.estimate_flops()[0] - base.estimate_flops()[0]) < 1.0


def test_g3_gives_streams_different_value_embeddings():
    """Legacy broadcasts one vector to all N streams; G3 must not."""
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        model = MST(make_config(mst_per_stream_ve=1))
        model.init_weights()
    ve = model.value_embeds[str(model.config.n_layer - 1)]
    d = model.config.mst_sub_dim
    assert ve.weight.shape[-1] == N_SUBS * d

    row = ve.weight[7].detach().float()
    slices = row.split(d)
    for j in range(1, N_SUBS):
        assert not torch.allclose(slices[0], slices[j]), \
            f"stream {j} received the same VE vector as stream 0"


# ── O1: output-head bottleneck ───────────────────────────────────────────────

def test_o1_factorizes_the_output_head():
    D = 256
    base = build_meta(D=D)
    thin = build_meta(D=D, mst_lm_head_dim=D // 2)

    assert base.lm_head.weight.shape[1] == D
    assert thin.lm_head.weight.shape[1] == D // 2
    assert thin.final_head.proj.weight.shape[0] == D // 2, "proj must land at the bottleneck"

    b, t = base.num_scaling_params(), thin.num_scaling_params()
    # Head cost is D*Dh + V*Dh against D*D + V*D, so halving Dh halves both pieces.
    b_head = b['lm_head'] + D * D
    t_head = t['lm_head'] + D * (D // 2)
    assert t_head == b_head // 2

    # Trunk untouched, FLOPs strictly down.
    assert t['transformer_matrices'] < b['transformer_matrices']   # proj shrank
    assert thin.estimate_flops()[0] < base.estimate_flops()[0]


def test_o1_rejects_conflict_with_global_residual():
    with pytest.raises(AssertionError, match="global residual"):
        build_meta(mst_lm_head_dim=64, mst_global_residual=1)


# ── O2: window composition ───────────────────────────────────────────────────

def test_o2_legacy_replaces_the_layer_pattern():
    """The defect being fixed: the widest stream runs full context at every layer."""
    m = build_meta(depth=8)
    full = [[w for w in layer if w[0] < 0 or w[0] >= m.config.sequence_len]
            for layer in m.layer_sub_windows]
    assert all(len(f) == 1 for f in full), \
        "legacy: exactly one full-context stream, on every single layer"


def test_o2_composition_narrows_windows_on_short_layers():
    m = build_meta(depth=8, mst_compose_windows=1)
    short_layers = [i for i, ws in enumerate(m.window_sizes)
                    if 0 < ws[0] < m.config.sequence_len]
    assert short_layers, "need a short layer in the pattern to test against"
    for i in short_layers:
        cap = m.window_sizes[i][0]
        for w in m.layer_sub_windows[i]:
            assert w[0] >= 0 and w[0] <= cap, \
                f"layer {i} sub window {w} exceeds the layer cap {cap}"
    # Long layers keep the per-sub schedule, including the full-context stream.
    long_layers = [i for i, ws in enumerate(m.window_sizes) if i not in short_layers]
    assert any(w[0] < 0 or w[0] >= m.config.sequence_len
               for i in long_layers for w in m.layer_sub_windows[i])


def test_o2_cuts_attention_flops_only():
    base = build_meta(depth=8)
    comp = build_meta(depth=8, mst_compose_windows=1)
    b, c = base.num_scaling_params(), comp.num_scaling_params()
    assert b['transformer_matrices'] == c['transformer_matrices'], "params must not move"
    assert comp.estimate_flops()[0] < base.estimate_flops()[0]


def test_o2_is_a_noop_without_multi_scale_windows():
    """With one window per layer there is nothing to compose."""
    off = build_meta(mst_multi_scale_windows=0)
    on = build_meta(mst_multi_scale_windows=0, mst_compose_windows=1)
    assert off.layer_sub_windows == on.layer_sub_windows
    assert off.estimate_flops()[0] == on.estimate_flops()[0]


# ── all fixes together ───────────────────────────────────────────────────────

def test_all_three_train_step():
    """Forward, backward and a real loss with every fix on."""
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        model = MST(make_config(mst_sub_head_dim=64, mst_final_norm=1, mst_per_stream_ve=1,
                                mst_lm_head_dim=128, mst_compose_windows=1))
        model.init_weights()
    assert model._use_batched

    idx = torch.randint(0, VOCAB, (2, SEQ))
    targets = torch.randint(0, VOCAB, (2, SEQ))
    loss = model(idx, targets=targets)

    assert torch.isfinite(loss), f"loss is {loss}"
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
