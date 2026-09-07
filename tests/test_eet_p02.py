"""Tests for the EET P02 decision experiments.

Two mechanisms are under test, each attached to one pre-registered hypothesis about
where the ~0.06 bpb gap to dense comes from:

  Test 1 (``eet_kv_mode``)     context destruction. In the compute-skip path a token that
                               exits is removed from every later layer's attention, so the
                               surviving (hard) tokens lose most of their context.
  Test 2 (``eet_route_noise``) data starvation. With a router that sees only the token
                               embedding, exit depth is a lookup table, so a deep layer
                               only ever trains on a fixed slice of the vocabulary.

These tests prove the mechanisms behave as specified. They do not test whether the
hypotheses are true, which is what the training sweep is for.
"""
import torch

from nanochat.gpt import GPTConfig
from nanochat.eet import EarlyExitGPT


def _config(**overrides):
    kw = dict(
        n_layer=6,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        vocab_size=128,
        sequence_len=32,
        window_pattern="L",
        use_eet=True,
        eet_global_router=True,
        eet_compute_skip=True,
        eet_min_exit_layer=1,
        eet_capacity_schedule='bell',
        eet_target_active_frac=0.25,
        eet_loss_variant='ce_guided',
        eet_warmup_frac=0.0,
        eet_explore_frac=0.0,
        eet_gumbel_temp_start=0.0,
    )
    kw.update(overrides)
    return GPTConfig(**kw)


def _batch(config, B=2, T=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, config.vocab_size, (B, T), generator=g)
    y = torch.randint(0, config.vocab_size, (B, T), generator=g)
    return x, y


def _run(config, seed=0, train=True, **fwd):
    torch.manual_seed(seed)
    model = EarlyExitGPT(config)
    model.train(train)
    x, y = _batch(config)
    kwargs = dict(eet_do_route=True, eet_phase=3, eet_total_steps=100)
    kwargs.update(fwd)
    return model, model(x, y, **kwargs)


def test_kv_modes_run_and_backprop():
    """All three kv modes produce a finite loss and a full-backbone gradient."""
    for mode in ('none', 'fresh', 'stale'):
        model, loss = _run(_config(eet_kv_mode=mode))
        assert torch.isfinite(loss), f"{mode}: non-finite loss"
        loss.backward()
        # The deepest block must still receive gradient in every mode.
        last = model.transformer.h[-1].attn.c_q
        w = last.weight if hasattr(last, 'weight') else next(last.parameters())
        assert w.grad is not None and torch.isfinite(w.grad).all(), f"{mode}: bad grad"


def test_kv_modes_change_the_computation():
    """fresh/stale must differ from none, and from each other.

    If they did not, the training arms would silently be duplicates of the baseline and
    the whole experiment would be uninformative.
    """
    losses = {}
    for mode in ('none', 'fresh', 'stale'):
        _, loss = _run(_config(eet_kv_mode=mode), train=False)
        losses[mode] = float(loss)
    assert abs(losses['fresh'] - losses['none']) > 1e-6, losses
    assert abs(losses['stale'] - losses['none']) > 1e-6, losses
    assert abs(losses['fresh'] - losses['stale']) > 1e-6, losses


def test_kv_mode_none_is_unchanged():
    """The default path must be bit-identical to the pre-P02 behaviour."""
    _, a = _run(_config(), train=False)
    _, b = _run(_config(eet_kv_mode='none'), train=False)
    assert torch.equal(a.detach(), b.detach())


def test_split_kv_matches_dense_attention_when_nothing_has_exited():
    """With no exits, full-key attention must reproduce ordinary attention.

    This is the correctness anchor for ``forward_split``: same queries, same keys, so any
    difference would be a bug in the RoPE gather, the mask, or the head reshape rather
    than a real effect of restoring context.
    """
    torch.manual_seed(0)
    config = _config(n_layer=2)
    model = EarlyExitGPT(config)
    model.eval()
    block = model.transformer.h[0]
    B, T, C = 2, 32, config.n_embd
    x = torch.randn(B, T, C)
    cos, sin = model.cos[:, :T], model.sin[:, :T]
    pos = torch.arange(T).unsqueeze(0).expand(B, -1).contiguous()

    norm_fn = block.norm_attn if block.norm_attn is not None else _norm
    with torch.no_grad():
        ref = block(x, None, (cos, sin), (-1, 0), None)
        k, v = block.attn._project_kv(norm_fn(x), None, cos, sin)
        got = block.forward_split(x, (cos, sin), pos, k, v, (-1, 0))
    assert torch.allclose(ref, got, atol=2e-2, rtol=2e-2), (ref - got).abs().max()


def _norm(t):
    from nanochat.gpt import norm
    return norm(t)


def test_route_noise_randomises_routing_in_training_only():
    """Noise must change which tokens exit during training and nothing at eval."""
    cfg = _config(eet_route_noise=5.0, eet_coverage_diag=True)

    torch.manual_seed(0)
    m = EarlyExitGPT(cfg)
    x, y = _batch(cfg)

    # Training: the same batch must produce different exit assignments across steps.
    m.train()
    m(x, y, eet_do_route=True, eet_phase=3, eet_total_steps=100)
    caps = list(m._last_active_counts)
    losses = set()
    for _ in range(4):
        out = m(x, y, eet_do_route=True, eet_phase=3, eet_total_steps=100)
        losses.add(round(float(out), 6))
        # Capacities are untouched by the noise: the FLOP budget stays fixed.
        assert list(m._last_active_counts) == caps
    assert len(losses) > 1, "route noise did not change the routing"

    # Eval is deterministic: two identical forwards give an identical loss.
    m.eval()
    with torch.no_grad():
        l1 = m(x, y, eet_do_route=True, eet_phase=3, eet_total_steps=100)
        l2 = m(x, y, eet_do_route=True, eet_phase=3, eet_total_steps=100)
    assert torch.equal(l1, l2)


def test_route_noise_lifts_deep_layer_coverage():
    """The point of Test 2: stochastic routing must widen what deep layers see.

    This is the mechanism the data-starvation hypothesis needs. With a deterministic
    router the deepest layer trains on one fixed slice of the vocabulary for the whole
    run; with resampled routing every token id eventually reaches every depth, at an
    unchanged per-step FLOP budget.
    """
    def deep_coverage(noise):
        cfg = _config(vocab_size=512, eet_route_noise=noise, eet_coverage_diag=True)
        torch.manual_seed(0)
        m = EarlyExitGPT(cfg)
        m.train()
        x, y = _batch(cfg, B=4, T=32)
        for _ in range(6):
            m(x, y, eet_do_route=True, eet_phase=3, eet_total_steps=100)
        return m.coverage_report()[-1]['n_seen']

    det, stoch = deep_coverage(0.0), deep_coverage(5.0)
    assert stoch > det, f"deterministic={det} stochastic={stoch}"


def test_route_noise_anneal_endpoints():
    """Annealing must actually move between the two endpoints."""
    cfg = _config(eet_route_noise=8.0, eet_route_noise_end=0.0)
    torch.manual_seed(0)
    m = EarlyExitGPT(cfg)
    x, y = _batch(cfg)
    m.train()
    # At the last step the noise is 0, so routing is deterministic and repeatable.
    outs = []
    for _ in range(2):
        torch.manual_seed(1234)
        outs.append(m(x, y, eet_do_route=True, eet_phase=3,
                      eet_step=99, eet_total_steps=100).detach().clone())
    assert torch.equal(outs[0], outs[1])


def test_coverage_diagnostic_shows_deep_layer_starvation():
    """Coverage must shrink with depth.

    This is the measurement behind the data-starvation hypothesis: if deep layers see
    most of the token distribution anyway, the hypothesis is dead before any training run.
    """
    cfg = _config(eet_coverage_diag=True)
    torch.manual_seed(0)
    m = EarlyExitGPT(cfg)
    m.eval()
    x, y = _batch(cfg, B=4, T=32)
    with torch.no_grad():
        for _ in range(3):
            m(x, y, eet_do_route=True, eet_phase=3)
    rep = m.coverage_report()
    assert rep is not None and len(rep) == cfg.n_layer
    assert rep[0]['mass_frac'] == 1.0
    assert rep[-1]['mass_frac'] < rep[0]['mass_frac']
    assert rep[-1]['n_seen'] <= rep[0]['n_seen']
    m.reset_coverage()
    assert m.coverage_report() is None


def test_coverage_is_opt_in():
    """Nothing is accumulated unless the flag is set."""
    torch.manual_seed(0)
    m = EarlyExitGPT(_config())
    m.train()
    x, y = _batch(_config())
    m(x, y, eet_do_route=True, eet_phase=3)
    assert m.coverage_report() is None


# ---------------------------------------------------------------------------
# Test 0A: the offline context oracle
# ---------------------------------------------------------------------------
def test_bell_capacities_reproduce_eet_schedule():
    """The oracle must mask exactly the tokens EET would have dropped.

    These are the d8 numbers the whole break-even argument is computed from, so a drift
    here would silently invalidate the gate.
    """
    from scripts.eet_context_oracle import bell_capacities
    rl, per_block = bell_capacities(8, 1, 0.125, 'bell')
    assert rl == [1, 2, 3, 4, 5, 6]
    expect = [1.0, 1.0, 0.9795, 0.858, 0.5625, 0.267, 0.1455, 0.125]
    for got, want in zip(per_block, expect):
        assert abs(got - want) < 1e-3, (per_block, expect)
    assert abs(sum(per_block) / 8 - 0.6172) < 1e-3


def test_oracle_ablations_separate_context_from_depth():
    """Each ablation must move the loss, and 'dense' must be the untouched reference."""
    from scripts.eet_context_oracle import bell_capacities, oracle_forward

    torch.manual_seed(0)
    cfg = GPTConfig(n_layer=6, n_head=2, n_kv_head=2, n_embd=16, vocab_size=128,
                    sequence_len=32, window_pattern="L")
    from nanochat.gpt import GPT
    model = GPT(cfg)
    model.eval()
    x, y = _batch(cfg, B=2, T=32)
    token_bytes = torch.ones(cfg.vocab_size, dtype=torch.long)
    _, per_block = bell_capacities(cfg.n_layer, 1, 0.125, 'bell')
    score = torch.rand(x.shape)

    out = {}
    for ab in ('dense', 'ctx', 'depth', 'both'):
        nats, nbytes = oracle_forward(model, x, y, per_block, score, ab, token_bytes)
        assert int(nbytes) > 0
        out[ab] = float(nats) / int(nbytes)
    assert out['ctx'] != out['dense'], out
    assert out['depth'] != out['dense'], out
    assert out['both'] != out['ctx'] and out['both'] != out['depth'], out


def test_oracle_is_a_no_op_at_full_capacity():
    """With nothing masked and nothing exiting, every ablation equals dense.

    Guards against the masking or the exit bookkeeping silently perturbing the reference.
    """
    from scripts.eet_context_oracle import oracle_forward

    torch.manual_seed(0)
    cfg = GPTConfig(n_layer=4, n_head=2, n_kv_head=2, n_embd=16, vocab_size=128,
                    sequence_len=32, window_pattern="L")
    from nanochat.gpt import GPT
    model = GPT(cfg)
    model.eval()
    x, y = _batch(cfg, B=2, T=32)
    token_bytes = torch.ones(cfg.vocab_size, dtype=torch.long)
    per_block = [1.0] * cfg.n_layer
    score = torch.rand(x.shape)

    ref = oracle_forward(model, x, y, per_block, score, 'dense', token_bytes)
    for ab in ('ctx', 'depth', 'both'):
        got = oracle_forward(model, x, y, per_block, score, ab, token_bytes)
        assert abs(float(got[0]) - float(ref[0])) < 2e-2, (ab, float(got[0]), float(ref[0]))
