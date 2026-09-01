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
from nanochat.mst import MST, mix_channels, resolve_sub_heads

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


# ── Stage 14: free cross-stream mixing ───────────────────────────────────────

@pytest.mark.parametrize("mode", ["roll", "shuffle"])
def test_mix_channels_roundtrips(mode):
    x = torch.arange(2 * 3 * N_SUBS * 8, dtype=torch.float32).view(2, 3, N_SUBS, 8)
    y = mix_channels(x, mode, N_SUBS, 8)
    assert not torch.equal(x, y), "the permutation must actually move channels"
    assert torch.equal(mix_channels(y, mode, N_SUBS, 8, inverse=True), x)


@pytest.mark.parametrize("mode", ["roll", "shuffle"])
def test_mix_channels_regroups_channels(mode):
    """The property the old feature_cycle lacked.

    Rolling by exactly d maps stream n to stream n+1 intact, so the *set* of channels
    travelling together never changes and nothing mixes. A real mixer has to put
    channels from different original streams into the same stream.
    """
    d = 8
    chan = torch.arange(N_SUBS * d).view(1, 1, N_SUBS, d).float()   # channel -> its index
    origin = lambda t: [set((v // d).long().tolist()) for v in t[0, 0]]

    assert origin(chan) == [{i} for i in range(N_SUBS)], "before: each stream is pure"
    mixed = origin(mix_channels(chan, mode, N_SUBS, d))
    assert all(len(s) > 1 for s in mixed), f"{mode} left some stream unmixed: {mixed}"

    # Contrast: feature_cycle's roll by exactly d leaves every stream pure.
    cycled = torch.roll(chan.reshape(1, 1, N_SUBS * d), shifts=d, dims=-1).view(1, 1, N_SUBS, d)
    assert all(len(s) == 1 for s in origin(cycled)), \
        "roll-by-d should be a pure relabelling, which is why it never mixed"


def test_mix_is_free():
    base = build_meta()
    for site in ("layer", "ffn", "both"):
        m = build_meta(mst_channel_mix="roll", mst_channel_mix_site=site)
        assert m.num_scaling_params() == base.num_scaling_params()
        assert m.estimate_flops()[0] == base.estimate_flops()[0]
        assert m._use_batched


def _wake_residual_branches(model, seed=1):
    """Make the layers compute something.

    MST zero-inits every residual output projection (attention c_proj, FFN fc_proj,
    the transition's agg_down), so a freshly initialized model is *exactly* the
    identity through its layers. Any change of basis is then unobservable: a
    permuted layer gives bit-identical output, because P^-1 . identity . P is the
    identity. Filling these is what makes the mixing measurable at init.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in model.layers:
            for name in ("c_proj_w", "fc_proj_w", "agg_down_w"):
                p = getattr(layer, name, None)
                if p is not None:
                    p.copy_(torch.randn(p.shape, generator=g) * 0.02)


def test_mix_changes_the_computation():
    """Same weights, permuted partition: the outputs must differ."""
    def logits(**ov):
        torch.manual_seed(0)
        with contextlib.redirect_stdout(io.StringIO()):
            model = MST(make_config(**ov))
            model.init_weights()
        _wake_residual_branches(model)
        model.eval()
        with torch.no_grad():
            return model(torch.arange(SEQ).remainder(VOCAB).view(1, SEQ))

    off = logits()
    assert off.abs().max() > 1e-3, "sanity: the layers should be doing something"
    for mode in ("roll", "shuffle"):
        for site in ("layer", "ffn"):
            on = logits(mst_channel_mix=mode, mst_channel_mix_site=site)
            assert not torch.allclose(off, on, atol=1e-6), f"{mode}/{site} changed nothing"


def test_mix_rejects_bad_settings():
    with pytest.raises(AssertionError, match="none\\|roll\\|shuffle"):
        build_meta(mst_channel_mix="bogus")
    with pytest.raises(AssertionError, match="layer\\|ffn\\|both"):
        build_meta(mst_channel_mix="roll", mst_channel_mix_site="bogus")


# ── Stage 15: coupling optimization + attention cross-stream mixing ──────────

def _muon_groups(model):
    """{id(param): (lr, block_diagonal_or_None)} over the Muon groups."""
    out = {}
    for g in model.setup_optimizer().param_groups:
        if g.get('kind') != 'muon':
            continue
        for p in g['params']:
            out[id(p)] = (g['lr'], g.get('block_diagonal'))
    return out


def test_f1_distribute_w_is_excluded_by_default():
    """The defect: distribute_w is (N*d, d) like c_proj_w but never got block-diag Muon."""
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        model = MST(make_config())
    g = _muon_groups(model)
    layer = model.layers[0]
    assert layer.distribute_w.shape[0] % N_SUBS == 0, "same stacked shape as c_proj_w"
    assert g[id(layer.distribute_w)][1] is None, "legacy: no block_diagonal"
    assert g[id(layer.c_proj_w)][1] == N_SUBS, "but the per-stream weights do get it"


def test_f1_puts_distribute_w_on_the_same_footing():
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        model = MST(make_config(mst_distribute_block_muon=1))
    g = _muon_groups(model)
    layer = model.layers[0]
    dist_lr, dist_blk = g[id(layer.distribute_w)]
    proj_lr, proj_blk = g[id(layer.c_proj_w)]
    assert dist_blk == N_SUBS and dist_lr == proj_lr, \
        f"distribute_w should match c_proj_w, got lr={dist_lr} block={dist_blk}"


def test_f2_gives_the_transition_matrices_spectral_lrs():
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        base = MST(make_config())
        spec = MST(make_config(mst_trans_spectral_lr=1))
    gb, gs = _muon_groups(base), _muon_groups(spec)
    lb, ls = base.layers[0], spec.layers[0]
    # Legacy: both share one lr.
    assert gb[id(lb.agg_up_w)][0] == gb[id(lb.agg_down_w)][0]
    # F2: they differ by exactly N, agg_up above and agg_down below.
    up, down = gs[id(ls.agg_up_w)][0], gs[id(ls.agg_down_w)][0]
    assert up / down == pytest.approx(N_SUBS), f"expected ratio {N_SUBS}, got {up/down}"
    assert up > gb[id(lb.agg_up_w)][0] > down


# ── F3: couple less often ────────────────────────────────────────────────────

@pytest.mark.parametrize("every,expected", [(1, [0, 1, 2, 3]), (2, [0, 2, 3]), (4, [0, 3])])
def test_f3_selects_the_coupling_layers(every, expected):
    m = build_meta(depth=4, mst_transition_every=every)
    assert [i for i, l in enumerate(m.layers) if l._couples] == expected
    assert m.layers[-1]._couples, "the last layer must always couple"


def test_f3_drops_params_and_flops():
    base = build_meta(depth=4)
    every2 = build_meta(depth=4, mst_transition_every=2)
    b, e = base.num_scaling_params(), every2.num_scaling_params()
    assert e['transformer_matrices'] < b['transformer_matrices']
    assert every2.estimate_flops()[0] < base.estimate_flops()[0]
    # Non-coupling layers allocate no transition weights at all.
    assert not hasattr(every2.layers[1], 'distribute_w')
    assert hasattr(every2.layers[0], 'distribute_w')
    assert every2._use_batched


def test_f3_is_no_longer_a_silent_noop():
    """It was ignored on the batched path for its whole life; that is the bug being fixed."""
    base = build_meta(depth=4)
    every2 = build_meta(depth=4, mst_transition_every=2)
    assert base.estimate_flops()[0] != every2.estimate_flops()[0]


# ── F4: talking heads ────────────────────────────────────────────────────────

def test_f4_is_identity_at_init_and_nearly_free():
    base = build_meta()
    talk = build_meta(mst_talking_heads=1)
    b, t = base.num_scaling_params(), talk.num_scaling_params()
    extra = t['transformer_matrices'] - b['transformer_matrices']
    H = N_SUBS * resolve_sub_heads(make_config())[0]
    assert extra == H * H * base.config.n_layer, "one (Nh, Nh) matrix per layer"
    assert extra / b['transformer_matrices'] < 0.01, "must be negligible"

    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        model = MST(make_config(mst_talking_heads=1))
        model.init_weights()
    assert torch.equal(model.layers[0].talking_w,
                       torch.eye(H, dtype=model.layers[0].talking_w.dtype))


def test_f4_changes_the_computation_once_trained_away_from_identity():
    def logits(**ov):
        torch.manual_seed(0)
        with contextlib.redirect_stdout(io.StringIO()):
            model = MST(make_config(**ov))
            model.init_weights()
        _wake_residual_branches(model)
        if ov:  # perturb off identity, as a step of training would
            with torch.no_grad():
                g = torch.Generator().manual_seed(3)
                for layer in model.layers:
                    layer.talking_w.add_(torch.randn(layer.talking_w.shape, generator=g) * 0.1)
        model.eval()
        with torch.no_grad():
            return model(torch.arange(SEQ).remainder(VOCAB).view(1, SEQ))

    assert not torch.allclose(logits(), logits(mst_talking_heads=1), atol=1e-6)


# ── F5: dense output projection ──────────────────────────────────────────────

def test_f5_dense_wo_costs_the_predicted_amount():
    D = 256
    base = build_meta(D=D)
    dense = build_meta(D=D, mst_wo_mode='dense')
    d, L = D // N_SUBS, base.config.n_layer
    b, x = base.num_scaling_params(), dense.num_scaling_params()
    # block c_proj is N*d*qkv = D^2/N; dense is D^2. Delta = D^2 * (1 - 1/N) per layer.
    assert x['transformer_matrices'] - b['transformer_matrices'] == \
        L * (D * D - N_SUBS * d * d)
    assert dense.estimate_flops()[0] > base.estimate_flops()[0]
    assert dense._use_batched


def test_f5_dense_wo_is_not_treated_as_block_structured():
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        model = MST(make_config(mst_wo_mode='dense'))
    layer = model.layers[0]
    assert not hasattr(layer, 'c_proj_w'), "the block-diagonal projection should be gone"
    g = _muon_groups(model)
    assert g[id(layer.c_proj_dense_w)][1] is None, \
        "a dense W_O must not get block-diagonal Newton-Schulz"


def test_f5_rejects_a_bad_mode():
    with pytest.raises(AssertionError, match="block\\|dense"):
        build_meta(mst_wo_mode='bogus')


# ── Stage 16: conditional stream execution ───────────────────────────────────

def _gated_layer(topk, **ov):
    """A layer whose router has been perturbed off its zero init, so top-k is meaningful."""
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        model = MST(make_config(mst_stream_topk=topk, **ov))
        model.init_weights()
    _wake_residual_branches(model)
    with torch.no_grad():
        for layer in model.layers:
            layer.stream_router_w.normal_(0, 0.5)
    return model


@pytest.mark.parametrize("k", [1, 2, 3])
def test_s16_gate_selects_exactly_k_streams_and_is_hard(k):
    """The forward value must be genuinely 0/1, or we are measuring a soft blend."""
    model = _gated_layer(k)
    layer = model.layers[0]
    with torch.no_grad():
        w, _ = layer._stream_gate(torch.randn(2, 8, N_SUBS, model.config.mst_sub_dim))
    assert set(w.flatten().tolist()) <= {0.0, 1.0}, "gate must be exactly 0 or 1"
    assert torch.equal(w.sum(-1), torch.full(w.shape[:-1], float(k))), \
        f"every token must activate exactly {k} streams"


def test_s16_init_is_not_already_collapsed():
    """Zero-init IS the collapsed state for a top-k gate, which is how it shipped first.

    With every logit at exactly 0, torch.topk breaks ties by index, so the same k streams
    win for every token before any training happens -- and the losers' FFNs then get
    exactly zero gradient, so they can never earn their way back in.
    """
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        model = MST(make_config(mst_stream_topk=2))
        model.init_weights()
    with torch.no_grad():
        w, _ = model.layers[0]._stream_gate(
            torch.randn(2, 64, N_SUBS, model.config.mst_sub_dim))
    load = w.mean(dim=(0, 1))
    assert (load > 0.05).all(), \
        f"some stream is never selected at init, so its FFN can never train: {load.tolist()}"


def test_s16_aux_loss_is_minimized_at_uniform_load():
    """The property the first version lacked.

    Switch's N*sum(f_i*P_i) only means anything when both factors are on the simplex:
    then uniform gives 1 and full concentration gives N. With independent sigmoids there
    is no simplex, and the same expression is minimized by driving every gate to zero --
    it balances nothing and shrinks the router's gradient.
    """
    model = _gated_layer(2)
    layer = model.layers[0]
    d = model.config.mst_sub_dim

    torch.manual_seed(3)
    balanced = torch.randn(4, 256, N_SUBS, d)
    with torch.no_grad():
        # Force a collapsed router by making one stream's logits dominate every token.
        layer.stream_router_w.zero_()
        layer.stream_router_w[0] = 50.0
        layer.stream_router_w[1] = 25.0
    _, aux_collapsed = layer._stream_gate(balanced)
    with torch.no_grad():
        layer.stream_router_w.zero_()          # uniform softmax over streams
    _, aux_uniform = layer._stream_gate(balanced)

    assert aux_collapsed > aux_uniform, \
        f"aux must penalize collapse: collapsed={aux_collapsed:.5f} uniform={aux_uniform:.5f}"


def test_s16_router_noise_explores_in_training_only():
    """Noisy top-k is what breaks the unselected-stream death spiral."""
    model = _gated_layer(2, mst_stream_router_noise=1.0)
    layer = model.layers[0]
    x = torch.randn(2, 64, N_SUBS, model.config.mst_sub_dim)

    layer.train()
    with torch.no_grad():
        a, _ = layer._stream_gate(x)
        b, _ = layer._stream_gate(x)
    assert not torch.equal(a, b), "noise should make training-time selection stochastic"

    layer.eval()
    with torch.no_grad():
        c, _ = layer._stream_gate(x)
        e, _ = layer._stream_gate(x)
    assert torch.equal(c, e), "evaluation must be deterministic"


def test_s16_router_receives_gradient_through_the_hard_gate():
    """Without the STE the router only hears the aux loss and collapses (eet.py:1994)."""
    model = _gated_layer(2)
    layer = model.layers[0]
    layer.stream_router_w.grad = None
    w, _ = layer._stream_gate(torch.randn(2, 8, N_SUBS, model.config.mst_sub_dim))
    w.sum().backward()
    assert layer.stream_router_w.grad is not None
    assert layer.stream_router_w.grad.abs().sum() > 0, "STE is not passing gradient"


def test_s16_routing_is_causal():
    """A token's stream choice must not depend on later tokens."""
    model = _gated_layer(2)
    layer = model.layers[0]
    with torch.no_grad():
        a = torch.randn(1, 6, N_SUBS, model.config.mst_sub_dim)
        b = a.clone()
        b[:, 3:] = torch.randn_like(b[:, 3:])      # perturb only the future
        wa, _ = layer._stream_gate(a)
        wb, _ = layer._stream_gate(b)
    assert torch.equal(wa[:, :3], wb[:, :3]), "gate for early tokens changed with later ones"


@pytest.mark.parametrize("k,expected", [(3, 0.75), (2, 0.50), (1, 0.25)])
def test_s16_active_flops_track_the_gated_fraction(k, expected):
    """The whole claim is on active FLOPs, so the accounting has to be right."""
    base = build_meta()
    sparse = build_meta(mst_stream_topk=k)
    tot, act, act_p = sparse.estimate_flops()
    b_tot, b_act, _ = base.estimate_flops()

    assert b_act == b_tot, "dense MST must not claim any sparsity"
    assert act < tot, "gated MST must discount active FLOPs"
    # FFN-only gating: the saving is 6 * (1-k/N) * (fc_w + fc_proj_w)
    ffn = sum(l.fc_w.numel() + l.fc_proj_w.numel() for l in sparse.layers)
    assert tot - act == 6 * int(ffn * (1 - k / N_SUBS) / len(sparse.layers)) * len(sparse.layers) \
        or abs((tot - act) - 6 * ffn * (1 - k / N_SUBS)) / (tot - act) < 0.01
    assert act_p < sparse.num_scaling_params()['total']


def test_s18_monarch_halves_the_sparsity_saving():
    """Monarch makes fc_w unskippable, so the FFN saving must drop to exactly half.

    Stream j's down-projection reads hidden units from every stream's up-projection, so
    a sparse kernel cannot skip fc_w and only fc_proj_w can be dropped. Gating both
    reported MON_shuffle_k1 at 0.797x when the honest figure is 0.752x, i.e. worse than
    doing nothing at all. Monarch is only asserted against stream_dispatch, so this
    combination is reachable and the accounting has to be right rather than assumed away.
    """
    k1 = build_meta(mst_stream_topk=1)
    mk1 = build_meta(mst_stream_topk=1, mst_ffn_monarch='shuffle')
    mon = build_meta(mst_ffn_monarch='shuffle')

    t_k1, a_k1, _ = k1.estimate_flops()
    t_m, a_m, _ = mk1.estimate_flops()
    t_o, a_o, _ = mon.estimate_flops()

    assert a_o == t_o, "Monarch alone is a permutation and must claim no saving"
    assert (1 - a_m / t_m) == pytest.approx((1 - a_k1 / t_k1) / 2, rel=1e-3), \
        "Monarch must halve the sparsity saving, not leave it untouched"


def test_s16_attention_gating_discounts_attention_flops_too():
    ffn_only = build_meta(mst_stream_topk=2)
    with_attn = build_meta(mst_stream_topk=2, mst_stream_gate_attn=1)
    assert with_attn.estimate_flops()[1] < ffn_only.estimate_flops()[1], \
        "gating attention should discount the QK term as well as the projections"


def test_s16_is_off_by_default_and_costs_nothing():
    base = build_meta()
    assert not base.layers[0]._stream_sparse
    assert not hasattr(base.layers[0], 'stream_router_w')
    # topk == N is dense, not sparse: no router, no discount.
    full = build_meta(mst_stream_topk=N_SUBS)
    assert not full.layers[0]._stream_sparse
    assert full.estimate_flops()[1] == full.estimate_flops()[0]


def test_s16_router_is_negligible_and_rejects_bad_k():
    base = build_meta()
    sparse = build_meta(mst_stream_topk=2)
    extra = (sparse.num_scaling_params()['transformer_matrices']
             - base.num_scaling_params()['transformer_matrices'])
    assert extra == N_SUBS * (N_SUBS * base.config.mst_sub_dim) * base.config.n_layer
    assert extra / base.num_scaling_params()['transformer_matrices'] < 0.02
    with pytest.raises(AssertionError, match="mst_stream_topk"):
        build_meta(mst_stream_topk=N_SUBS + 1)


def test_s16_changes_the_computation():
    def logits(**ov):
        model = _gated_layer(ov.pop('topk')) if 'topk' in ov else None
        if model is None:
            torch.manual_seed(0)
            with contextlib.redirect_stdout(io.StringIO()):
                model = MST(make_config(**ov))
                model.init_weights()
            _wake_residual_branches(model)
        model.eval()
        with torch.no_grad():
            return model(torch.arange(SEQ).remainder(VOCAB).view(1, SEQ))
    assert not torch.allclose(logits(), logits(topk=2), atol=1e-6)


# ── Stage 16 Phase B: real gather/scatter dispatch ───────────────────────────

def _dispatch_pair(cap, k=2, depth=2, noise=0.0):
    """Two models with identical weights and routers, one masked and one dispatched."""
    out = []
    for dispatch in (0, 1):
        torch.manual_seed(0)
        with contextlib.redirect_stdout(io.StringIO()):
            m = MST(make_config(depth=depth, D=256, mst_stream_topk=k,
                                mst_stream_router_noise=noise,
                                mst_stream_dispatch=dispatch,
                                mst_stream_capacity_factor=cap))
            m.init_weights()
        _wake_residual_branches(m)
        with torch.no_grad():
            g = torch.Generator().manual_seed(5)
            for layer in m.layers:
                layer.stream_router_w.copy_(
                    torch.randn(layer.stream_router_w.shape, generator=g) * 0.3)
        m.eval()
        # _last_stream_drop is gated on _diag_enabled: computing it costs a device sync
        # and a graph break per layer, so it is off in the hot path.
        m._diag_enabled = True
        out.append(m)
    return out


@pytest.mark.parametrize("cap", [1.5, 2.0])
def test_phase_b_matches_masking_when_nothing_overflows(cap):
    """Dispatch must be an optimization, not a different model.

    With capacity headroom every selected token is kept, so gather/FFN/scatter has to
    reproduce the masked path exactly. Any drift here means the indexing is wrong.
    """
    masked, dispatched = _dispatch_pair(cap)
    idx = torch.arange(256).remainder(VOCAB).view(1, 256)
    with torch.no_grad():
        ref, out = masked(idx), dispatched(idx)
    assert max(l._last_stream_drop for l in dispatched.layers) == 0.0
    assert torch.equal(ref, out), f"dispatch diverged from masking at capacity {cap}"


def test_phase_b_drops_on_overflow_and_reports_it():
    """At capacity 1.0 routing imbalance must overflow, and that must be visible."""
    masked, dispatched = _dispatch_pair(1.0)
    idx = torch.arange(256).remainder(VOCAB).view(1, 256)
    with torch.no_grad():
        ref, out = masked(idx), dispatched(idx)
    drop = max(l._last_stream_drop for l in dispatched.layers)
    assert drop > 0, "perfectly balanced routing at cap=1.0 would be suspicious"
    assert not torch.equal(ref, out), "dropped tokens must actually change the output"


def test_phase_b_selection_is_causal():
    """Capacity is resolved in POSITION order, so token t cannot be evicted by token t+1.

    This is why the dispatch keeps token-choice routing. Expert-choice, where each stream
    takes its own top-K tokens, gives exact load balance but is non-causal and would leak
    future information into an autoregressive model.
    """
    _, m = _dispatch_pair(1.0)
    layer = m.layers[0]
    d = m.config.mst_sub_dim
    torch.manual_seed(11)
    x = torch.randn(1, 64, N_SUBS, d)
    fc, fcp = layer.fc_w.view(N_SUBS, -1, d), layer.fc_proj_w.view(N_SUBS, d, -1)

    with torch.no_grad():
        w, _ = layer._stream_gate(x)
        a = layer._ffn_dispatched(x, w, fc, fcp)
        x2 = x.clone()
        x2[:, 32:] = torch.randn_like(x2[:, 32:])       # perturb only the future
        w2, _ = layer._stream_gate(x2)
        b = layer._ffn_dispatched(x2, w2, fc, fcp)
    assert torch.equal(w[:, :32], w2[:, :32]), "routing for early tokens changed"
    assert torch.equal(a[:, :32], b[:, :32]), \
        "a later token displaced an earlier one from its stream: capacity is not causal"


def test_phase_b_keeps_every_stream_trainable():
    """Dispatch removes the router's 'what if I picked this' gradient, so check the FFNs."""
    _, m = _dispatch_pair(1.0, noise=1.0)
    m.train()
    idx = torch.arange(256).remainder(VOCAB).view(1, 256)
    m(idx, targets=torch.randint(0, VOCAB, (1, 256))).backward()
    layer = m.layers[0]
    per_stream = layer.fc_w.grad.reshape(N_SUBS, -1).abs().sum(-1)
    assert (per_stream > 0).all(), f"a stream's FFN got no gradient: {per_stream.tolist()}"
    assert layer.stream_router_w.grad.abs().sum() > 0


def test_phase_b_rejects_incompatible_cross_sub_gate():
    with pytest.raises(AssertionError, match="mst_cross_sub_gate"):
        build_meta(mst_stream_topk=2, mst_stream_dispatch=1, mst_cross_sub_gate=32)


# ── Stage 18: Monarch-structured FFN ─────────────────────────────────────────

def _monarch_model(mode, **ov):
    torch.manual_seed(0)                      # reset per model, or two builds diverge
    with contextlib.redirect_stdout(io.StringIO()):
        m = MST(make_config(mst_ffn_monarch=mode, **ov))
        m.init_weights()
    return m


@pytest.mark.parametrize("mode,expect_full", [("shuffle", True), ("roll", False)])
def test_monarch_makes_the_ffn_mix_across_streams(mode, expect_full):
    """The whole point: stream j's down-projection must read other streams' hidden units.

    Without the permutation, fc_w and fc_proj_w are two independently block-diagonal maps
    and stream j only ever sees its own 4d hidden units -- a Monarch factorization with P
    set to identity.
    """
    inner = 4 * (256 // N_SUBS)
    h = torch.arange(N_SUBS * inner).view(1, 1, N_SUBS, inner).float()
    mixed = mix_channels(h, mode, N_SUBS, inner)
    origins = [set((v // inner).long().tolist()) for v in mixed[0, 0]]

    assert all(len(o) > 1 for o in origins), \
        f"{mode} left a stream reading only its own up-projection: {origins}"
    if expect_full:
        assert all(len(o) == N_SUBS for o in origins), \
            "shuffle is the Monarch transpose: every stream should draw from every other"
    else:
        assert all(len(o) == 2 for o in origins), \
            "roll trades with one neighbour only"


def test_monarch_permutation_commutes_with_the_nonlinearity():
    """relu^2 is elementwise, so the permutation may sit on either side of it.

    Worth pinning: it means the placement in forward is a readability choice, not a
    semantic one, and a future refactor that moves it cannot silently change the model.
    """
    inner = 4 * (256 // N_SUBS)
    torch.manual_seed(0)
    z = torch.randn(2, 8, N_SUBS, inner)
    after = mix_channels(torch.relu(z).square(), 'shuffle', N_SUBS, inner)
    before = torch.relu(mix_channels(z, 'shuffle', N_SUBS, inner)).square()
    assert torch.equal(after, before)


@pytest.mark.parametrize("mode", ["shuffle", "roll"])
def test_monarch_is_free(mode):
    base = build_meta()
    mon = build_meta(mst_ffn_monarch=mode)
    assert mon.num_scaling_params() == base.num_scaling_params()
    assert mon.estimate_flops()[0] == base.estimate_flops()[0]
    assert mon._use_batched


@pytest.mark.parametrize("mode", ["shuffle", "roll"])
def test_monarch_changes_the_computation(mode):
    """Must wake the residual branches first.

    fc_proj_w is zero-initialized, so at init the FFN output is exactly zero whatever the
    hidden units are, and this test passes vacuously without _wake_residual_branches --
    verified: the two models are bit-identical at init and differ once woken.
    """
    idx = torch.arange(SEQ).remainder(VOCAB).view(1, SEQ)
    off, on = _monarch_model('none'), _monarch_model(mode)
    with torch.no_grad():
        assert torch.equal(off(idx), on(idx)), \
            "sanity: at init the zero-init FFN branch should make these identical"
    for m in (off, on):
        _wake_residual_branches(m)
    with torch.no_grad():
        assert not torch.allclose(off(idx), on(idx), atol=1e-6)


def test_monarch_rejects_bad_mode_and_dispatch():
    with pytest.raises(AssertionError, match="none\\|shuffle\\|roll"):
        build_meta(mst_ffn_monarch='bogus')
    # Monarch needs every stream's up-projection; the dispatch exists not to compute them.
    # And in the dispatched path a buffer slot holds a different token per stream, so
    # permuting across streams would mix hidden units from different tokens.
    with pytest.raises(AssertionError, match="mst_stream_dispatch"):
        build_meta(mst_ffn_monarch='shuffle', mst_stream_topk=1, mst_stream_dispatch=1)


def test_monarch_composes_with_masked_sparsity():
    """Masked sparsity is allowed (all up-projections still run), unlike the dispatch."""
    m = build_meta(mst_ffn_monarch='shuffle', mst_stream_topk=1)
    assert m._use_batched and m.layers[0]._ffn_monarch == 'shuffle'


# ── CLI plumbing ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("script", ["scripts/base_train.py", "scripts/research_compare.py"])
def test_no_duplicate_cli_flags(script):
    """Two add_argument calls for one option string is an ArgumentError at import.

    It kills every run in a sweep before a single step, and the sweep dry-run does not
    catch it because that stubs research_sweep.sh and never reaches these parsers.
    `--mst-transition-every` was declared twice this way: once as the Stage 3 flag and
    once again when Stage 15 made it functional.
    """
    import ast, collections, pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    tree = ast.parse((root / script).read_text())
    seen = collections.Counter(
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    )
    dupes = {flag: n for flag, n in seen.items() if n > 1}
    assert not dupes, f"{script} declares these option strings more than once: {dupes}"


def test_stage_15_flags_are_wired_end_to_end():
    """Every new flag needs a config field, a CLI arg, and a passthrough, or it no-ops."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    base_train = (root / "scripts/base_train.py").read_text()
    compare = (root / "scripts/research_compare.py").read_text()
    sweep = (root / "scripts/research_sweep.sh").read_text()
    cfg = make_config()

    for field in ("mst_distribute_block_muon", "mst_trans_spectral_lr",
                  "mst_talking_heads", "mst_wo_mode", "mst_transition_every",
                  "mst_stream_topk", "mst_stream_router_aux", "mst_stream_gate_attn",
                  "mst_stream_router_noise", "mst_stream_dispatch",
                  "mst_stream_capacity_factor",
                  "mst_shampoo", "mst_precond_every", "mst_shampoo_beta",
                  "mst_ffn_monarch"):
        flag = "--" + field.replace("_", "-")
        assert hasattr(cfg, field), f"{field} missing from GPTConfig"
        assert f'add_argument("{flag}"' in base_train, f"{flag} has no base_train CLI arg"
        assert f"{field}=getattr(args" in base_train, f"{flag} never reaches the model config"
        assert f'"{flag}", str(' in compare, f"{flag} is not forwarded by research_compare"
        # research_sweep.sh whitelists args and hard-exits on anything unknown.
        assert flag + "|" in sweep or flag + ")" in sweep, f"{flag} missing from the sweep whitelist"


# ── all fixes together ───────────────────────────────────────────────────────

def test_all_three_train_step():
    """Forward, backward and a real loss with every fix on."""
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        model = MST(make_config(mst_sub_head_dim=64, mst_final_norm=1, mst_per_stream_ve=1,
                                mst_lm_head_dim=128, mst_compose_windows=1,
                                mst_channel_mix='roll', mst_channel_mix_site='both',
                                mst_distribute_block_muon=1, mst_trans_spectral_lr=1,
                                mst_transition_every=2, mst_talking_heads=1))
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


# ---------------------------------------------------------------- G3-cheap: VE map

@pytest.mark.parametrize("rank", [0, 32])
def test_ve_map_is_identity_at_init(rank):
    """Both map forms must start EXACTLY at the plain-VE baseline.

    Full rank is eye-initialised, low rank is identity + V U with V zero. Without this
    the arm would not be a clean ablation of plain VE, it would be a different model.
    """
    torch.manual_seed(0)
    plain = MST(make_config(mst_sub_head_dim=64, mst_compose_windows=1,
                            mst_wo_mode='dense'))
    plain.init_weights()
    torch.manual_seed(0)
    mapped = MST(make_config(mst_sub_head_dim=64, mst_compose_windows=1,
                             mst_wo_mode='dense', mst_ve_map=1, mst_ve_map_rank=rank))
    mapped.init_weights()
    plain.eval(); mapped.eval()
    idx = torch.randint(0, make_config().vocab_size, (2, 32))
    with torch.no_grad():
        assert torch.equal(plain(idx), mapped(idx))


def test_ve_map_gives_each_stream_a_distinct_vector():
    """The whole point: one shared table, N different per-stream views of it."""
    m = MST(make_config(mst_sub_head_dim=64, mst_compose_windows=1,
                        mst_wo_mode='dense', mst_ve_map=1))
    m.init_weights()
    N, d = m.config.mst_n_subs, m.config.mst_sub_dim
    key = next(iter(m.value_embeds.keys()))
    with torch.no_grad():
        torch.nn.init.normal_(m.ve_map_w[key], std=0.02)   # wake it from identity
        ve = torch.randn(1, 3, d)
        out = m._apply_ve_map(ve, key, 1, 3, N, d).view(1, 3, N, d)
    for j in range(1, N):
        assert not torch.allclose(out[..., 0, :], out[..., j, :]), \
            f"stream {j} got the same VE vector as stream 0"


def test_ve_map_is_counted_in_params_and_flops():
    """It is a matmul, not a lookup, so it must be charged on both axes.

    num_scaling_params enumerates explicitly (layers + input + final) rather than
    subtracting, so a model-level parameter is invisible to it unless added by hand.
    That also sets the token budget, so an uncounted map would train on free data.
    """
    plain = build_meta(mst_sub_head_dim=64, mst_compose_windows=1, mst_wo_mode='dense')
    mapped = build_meta(mst_sub_head_dim=64, mst_compose_windows=1, mst_wo_mode='dense',
                        mst_ve_map=1)
    cp, cm = plain.num_scaling_params(), mapped.num_scaling_params()
    extra = sum(p.numel() for n, p in mapped.named_parameters() if 've_map' in n)
    assert extra > 0
    assert cm['total'] - cp['total'] == extra
    assert cm['transformer_matrices'] - cp['transformer_matrices'] == extra, \
        "the VE map belongs in transformer_matrices; it sets the token budget"
    assert cm['value_embeds'] == cp['value_embeds'], "it is not a lookup"
    assert mapped.estimate_flops()[0] - plain.estimate_flops()[0] == 6 * extra


def test_ve_map_rejects_conflicting_and_useless_settings():
    with pytest.raises(AssertionError, match="pick one"):
        build_meta(mst_ve_map=1, mst_per_stream_ve=1)
    with pytest.raises(AssertionError, match="must be <"):
        build_meta(mst_ve_map=1, mst_ve_map_rank=999)


# ── MoL's S+KofN topology for MST ────────────────────────────────────────────

def test_stream_shared_keeps_the_first_S_always_active():
    """MoL's 1+3of15: S always-active streams, top-k over the remaining N-S.

    MST has no attention-coverage problem to fix here (it never gates attention),
    so a shared stream is pure always-on capacity, not a repair. The arm exists to
    measure whether that is worth anything.
    """
    torch.manual_seed(0)
    N, k, S = 8, 2, 1
    with contextlib.redirect_stdout(io.StringIO()):
        m = MST(make_config(D=512, mst_n_subs=N, mst_sub_dim=512 // N,
                            mst_sub_head_dim=64, mst_wo_mode='dense',
                            mst_compose_windows=1, mst_stream_topk=k,
                            mst_stream_shared=S, mst_stream_router_noise=1.0))
        m.init_weights()
    m.train()
    w, _ = m.layers[0]._stream_gate(torch.randn(2, 8, N, 512 // N))
    act = (w > 0).float()
    assert act.sum(-1).unique().tolist() == [float(S + k)]
    assert act[..., :S].min() == 1.0, "shared streams must always be on"
    assert act[..., S:].sum(-1).unique().tolist() == [float(k)], \
        "top-k must run over the routed pool only"
    assert set(w.unique().tolist()) <= {0.0, 1.0}, "the gate value stays hard 0/1"


def test_stream_shared_rejects_impossible_splits():
    with pytest.raises(AssertionError, match="mst_stream_shared"):
        build_meta(mst_n_subs=N_SUBS, mst_stream_shared=N_SUBS)
    with pytest.raises(AssertionError, match="routed pool"):
        build_meta(mst_n_subs=N_SUBS, mst_stream_shared=2, mst_stream_topk=3)


def test_stream_shared_is_off_by_default():
    m = build_meta(mst_stream_topk=2)
    assert m.layers[0]._stream_shared == 0


# ═══════════════════════════════════════════════════════════════════════════
# Stage 19: non-GEMM overhead cuts. Measurement (scripts/p10_mfu_microbench.py,
# H100) showed MST's block-diagonal GEMMs clear their wall-clock crossover at
# every D >= 512, and that the 2.4x training gap against dense is non-GEMM time:
# 36.7 ms of a 47.7 ms step at depth 12, against dense's 10.6 of 20.2. These three
# cuts attack that without touching the mathematics.
#
#   O3 (_rope_streams):  RoPE and QK-norm run once over (B, T, N, H, hd) instead
#       of N times over per-stream slices. Both are elementwise and identical
#       across streams, so the result must be exactly the per-stream one.
#   O4 (fused QKV):      one bmm with a 3x wider output instead of three, because
#       the reduction dimension d is what makes these GEMMs narrow. Exact in the
#       forward; the backward reassociates, so it agrees only to bf16 rounding.
#   O5 (gated diagnostics): the routing reductions are graph outputs under
#       torch.compile, so they cost on every step rather than on log steps.
# ═══════════════════════════════════════════════════════════════════════════

from nanochat.gpt import apply_rotary_emb as _apply_rotary_emb
from nanochat.mst import _rope_streams


def _real_model(**overrides):
    """An allocated, non-identity MST. build_meta gives shapes only."""
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        m = MST(make_config(**overrides))
        m.init_weights()
    _wake_residual_branches(m)
    m.eval()
    return m


def test_o3_rope_streams_matches_the_per_stream_form():
    """The whole point of hoisting: same arithmetic, one call instead of N."""
    B, T, N, H, hd = 2, 8, N_SUBS, 3, 16
    torch.manual_seed(0)
    x = torch.randn(B, T, N, H, hd)
    cos = torch.randn(1, T, 1, hd // 2)
    sin = torch.randn(1, T, 1, hd // 2)

    fused = _rope_streams(x, cos, sin)
    for j in range(N):
        per_stream = _apply_rotary_emb(x[:, :, j], cos, sin)
        assert torch.equal(fused[:, :, j], per_stream), \
            f"stream {j} diverges from gpt.apply_rotary_emb"


def test_o4_fused_qkv_is_exact_in_the_forward():
    """One (N, 3*qkv, d) bmm must give the same q/k/v as three (N, qkv, d) ones.

    Compared against the model's own unfused path rather than a reimplementation,
    by flipping _fuse_qkv, so the test cannot drift from the code it checks.
    """
    m = _real_model(mst_stream_topk=1, mst_stream_router_noise=0.0)
    idx = torch.arange(SEQ).remainder(VOCAB).view(1, SEQ)
    tgt = torch.arange(1, SEQ + 1).remainder(VOCAB).view(1, SEQ)

    assert all(layer._fuse_qkv for layer in m.layers), "fusion should be on by default"
    with torch.no_grad():
        fused = m(idx, targets=tgt)
    for layer in m.layers:
        layer._fuse_qkv = False
    with torch.no_grad():
        unfused = m(idx, targets=tgt)

    a = fused[0] if isinstance(fused, tuple) else fused
    b = unfused[0] if isinstance(unfused, tuple) else unfused
    assert torch.equal(a, b), f"fused QKV changed the forward: {a.item()} vs {b.item()}"


def test_o4_fusion_is_off_when_kv_is_shared():
    """Shared K/V stores c_k_w at (qkv, d), so there is no stream axis to concat."""
    assert build_meta(mst_shared_kv_attn=1).layers[0]._fuse_qkv is False
    assert build_meta(mst_shared_kv_attn=0).layers[0]._fuse_qkv is True


def test_o5_routing_diagnostics_are_off_in_the_hot_path():
    """Populated on log steps, absent otherwise, and compute_diagnostics still works.

    The reductions write to module attributes, which makes them graph outputs under
    torch.compile: they pin the routing mask alive and block dead-code elimination on
    every step. Gating them on _diag_enabled reuses the two-graph mechanism that
    _diag_sub_states already relies on.
    """
    m = _real_model(mst_stream_topk=2, mst_stream_router_noise=0.0)
    idx = torch.arange(SEQ).remainder(VOCAB).view(1, SEQ)
    layer = m.layers[0]

    m._diag_enabled = False
    with torch.no_grad():
        m(idx)
    assert layer._last_stream_load is None, "stream load captured off a log step"
    assert layer._last_route_entropy is None, "route entropy captured off a log step"

    m._diag_enabled = True
    with torch.no_grad():
        m(idx)
    assert layer._last_stream_load is not None and layer._last_stream_load.numel() == N_SUBS
    assert layer._last_route_entropy is not None
    diag = m.compute_diagnostics()
    assert 'stream_load_L0_S0' in diag and 'route_entropy_L0' in diag
