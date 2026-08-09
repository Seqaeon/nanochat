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
                  "mst_shampoo", "mst_precond_every", "mst_shampoo_beta"):
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
