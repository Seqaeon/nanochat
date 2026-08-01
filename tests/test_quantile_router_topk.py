"""Regression tests for the quantile routers' topk convention.

Background. Everywhere else in nanochat/gpt.py, `template_topk = 0` means "mix
all K templates softly": RemixedLinear._template_weights and both compose
branches fall through to a plain softmax when topk is 0 or >= K, and that is what
--p22-template-topk 0 means in scripts/p29_sweep.sh.

The quantile routers disagreed. They clamped with `max(1, min(topk, K))`, which
maps 0 to 1, so every run with --p23-quantile-route 1 --p22-template-topk 0 did
hard top-1 selection with coefficient exactly 1.0 rather than soft mixing. These
tests pin the corrected convention and pin the escape hatch: --p22-template-topk 1
still reproduces the old behaviour bit for bit.

The last test does not check a fix. It documents the routers' *scope*: the
affinity signal is mean-pooled over dim 1, so one weight vector is broadcast to
every position and routing is per-sequence rather than per-token or per-chunk.
That is load-bearing for how the paper describes chunk-amortised routing, so it
should fail loudly if anyone changes it by accident.

    python -m pytest tests/test_quantile_router_topk.py -v
"""
import math
import sys
import os

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanochat.gpt import (QuantileBalancedRouter, QuantileCrossAttentionRouter,
                          _resolve_topk)

K = 8
D = 64
ROUTERS = ["balanced", "cross_attention"]


def make_router(kind, topk, seed=0):
    torch.manual_seed(seed)
    if kind == "balanced":
        return QuantileBalancedRouter(D, K, topk, learned=True)
    return QuantileCrossAttentionRouter(D, K, topk, head_dim=16)


def signal(seed=1, B=4, T=32):
    torch.manual_seed(seed)
    return torch.randn(B, T, D)


# ---------------------------------------------------------------------------
def test_resolve_topk_semantics():
    assert _resolve_topk(0, K) == K, "0 must mean 'all K', not 1"
    assert _resolve_topk(-1, K) == K
    assert _resolve_topk(1, K) == 1
    assert _resolve_topk(3, K) == 3
    assert _resolve_topk(K, K) == K
    assert _resolve_topk(K + 5, K) == K, "must clamp above K"


@pytest.mark.parametrize("kind", ROUTERS)
@pytest.mark.parametrize("training", [False, True])
def test_topk_zero_is_soft_over_all_templates(kind, training):
    """The bug: topk=0 produced a one-hot vector instead of a mixture."""
    r = make_router(kind, topk=0).train(training)
    w = r(signal())
    nz = (w > 0).sum(-1)
    assert nz.min().item() == K, (
        f"topk=0 must activate all {K} templates, got min {nz.min().item()}")
    ent = -(w.clamp_min(1e-12).log() * w).sum(-1).mean()
    assert ent / math.log(K) > 0.5, (
        f"topk=0 must be a genuine mixture; normalised entropy {ent / math.log(K):.3f}")


@pytest.mark.parametrize("kind", ROUTERS)
@pytest.mark.parametrize("training", [False, True])
def test_topk_zero_equals_topk_K(kind, training):
    """Same instance, so any difference is the code path, not the init."""
    r = make_router(kind, topk=0).train(training)
    x = signal()
    r.topk = 0
    w0 = r(x)
    r.topk = K
    wK = r(x)
    assert torch.equal(w0, wK), f"max diff {(w0 - wK).abs().max().item():.3e}"


@pytest.mark.parametrize("kind", ROUTERS)
@pytest.mark.parametrize("topk", [1, 2, 4])
@pytest.mark.parametrize("training", [False, True])
def test_explicit_topk_still_sparse(kind, topk, training):
    """The escape hatch. --p22-template-topk 1 reproduces pre-fix behaviour."""
    r = make_router(kind, topk=topk).train(training)
    w = r(signal())
    nz = (w > 0).sum(-1)
    assert nz.max().item() == topk and nz.min().item() == topk, (
        f"topk={topk} should select exactly {topk}, got {nz.min()}-{nz.max()}")


@pytest.mark.parametrize("kind", ROUTERS)
@pytest.mark.parametrize("topk", [0, 1, 3])
@pytest.mark.parametrize("training", [False, True])
def test_weights_are_a_distribution(kind, topk, training):
    r = make_router(kind, topk=topk).train(training)
    w = r(signal())
    assert (w >= 0).all(), "negative routing weight"
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)


@pytest.mark.parametrize("kind", ROUTERS)
def test_router_receives_gradient_under_topk_zero(kind):
    """Soft mixing must keep the router trainable; a masked one-hot does not."""
    r = make_router(kind, topk=0).train(True)
    x = signal()
    w = r(x)
    # A target that actually depends on which template wins, unlike w.sum().
    target = torch.zeros(K)
    target[0] = 1.0
    (w - target).pow(2).mean().backward()
    grads = [p.grad for p in r.gate_parameters() if p.grad is not None]
    assert grads, "no gate parameter received a gradient"
    assert max(float(g.abs().max()) for g in grads) > 0, "gradient is identically zero"


@pytest.mark.parametrize("kind", ROUTERS)
@pytest.mark.parametrize("training", [False, True])
def test_routing_varies_with_position(kind, training):
    """Each position gets its own routing decision.

    This is what makes chunk-amortised routing mean anything: the chunk path
    hands the router one anchor per chunk, so a router that scores positions
    independently produces one template per chunk. The router previously
    mean-pooled dim 1, which is the chunk axis, and broadcast one decision to the
    whole sequence.
    """
    r = make_router(kind, topk=0).train(training)
    w = r(signal())
    spread = (w.amax(dim=1) - w.amin(dim=1)).max().item()
    assert spread > 0.0, (
        "routing is constant across positions, so every chunk of a sequence gets "
        "the same template and chunk size cannot matter")


@pytest.mark.parametrize("training", [False, True])
def test_routing_is_causal(training):
    """A position's routing must not depend on later positions.

    The pooled variant averaged the whole sequence, so chunk 0's template
    depended on tokens at position T-1. For an autoregressive LM that is an
    information leak, and it undercuts describing the method as causal segment
    routing.
    """
    r = make_router("balanced", topk=0).train(training)
    x = signal()
    w = r(x)
    x2 = x.clone()
    x2[:, -1, :] += 50.0                      # perturb only the final position
    w2 = r(x2)
    early = (w[:, :-1] - w2[:, :-1]).abs().max().item()
    assert early == 0.0, (
        f"perturbing the last position changed earlier routing by {early:.3e}")
    assert (w[:, -1] - w2[:, -1]).abs().max().item() > 0, "the perturbation did nothing"


@pytest.mark.parametrize("training", [False, True])
def test_legacy_sequence_scope_restores_pooling(training):
    """route_scope='sequence' reproduces the pre-fix behaviour for old runs."""
    torch.manual_seed(0)
    r = QuantileBalancedRouter(D, K, 0, learned=True, route_scope="sequence").train(training)
    x = signal()
    w = r(x)
    assert (w.amax(dim=1) - w.amin(dim=1)).max().item() == 0.0, (
        "legacy scope must stay position-invariant")
    x2 = x.clone()
    x2[:, -1, :] += 50.0
    assert (w[:, 0] - r(x2)[:, 0]).abs().max().item() > 0, (
        "legacy scope must stay non-causal; if this changed, the escape hatch no "
        "longer reproduces the runs it exists for")


@pytest.mark.parametrize("topk", [0, 1, 2, 4])
@pytest.mark.parametrize("training", [False, True])
def test_quantile_balancing_does_not_change_any_routing_decision(topk, training):
    """The balancing mechanism is a no-op, and this pins that it still is.

    `mask = q_mask | hard_mask` unions the quantile-thresholded set with the
    per-position top-k. Because the top-k entries hold the largest scores, the
    following `masked_scores.topk(topk)` re-selects exactly those same entries no
    matter what the threshold admitted. The threshold can only add lower-scoring
    entries to the mask, and adding them never displaces the top-k.

    So QuantileBalancedRouter is bit-identical to plain top-k plus softmax. If
    this test ever fails, the balancing has started doing something and the
    paper's claim about replacing the auxiliary loss can finally be supported.
    """
    r = make_router("balanced", topk=topk).train(training)
    x = signal()
    scores = F.linear(x.float(), r.route_proj.float())
    eff = K if topk <= 0 else topk
    vals, idx = scores.topk(eff, dim=-1)
    plain = torch.zeros_like(scores).scatter_(-1, idx, F.softmax(vals, dim=-1))
    w = r(x)
    assert torch.equal(w > 0, plain > 0), "selected set differs from plain top-k"
    assert (w - plain).abs().max().item() < 1e-6, (
        f"weights differ from plain top-k by {(w - plain).abs().max().item():.3e}")


def test_end_to_end_remixed_linear_mixes_templates():
    """The configuration the paper ships: quantile routing with topk=0.

    Before the fix this produced a one-hot alpha, so W_eff was exactly one bank
    entry and 'template mixing' never happened.
    """
    from nanochat.gpt import RemixedLinear
    torch.manual_seed(0)
    layer = RemixedLinear(
        D, D, context_dim=16, basis_size=D,
        remixed_linear_kwargs=dict(
            n_templates=K, template_topk=0, template_routing_learned=True,
            use_quantile_route=1, use_output_gate=True, use_basis_gate=False,
            use_context=False, chunk_routing_size=8,
        ),
    ).eval()
    assert layer._qrouter is not None, "expected the quantile router to be active"
    x = torch.randn(2, 32, D)
    w = layer._template_weights(x.reshape(2, 4, 8, D)[:, :, 0, :].float(), torch.float32)
    assert (w > 0).sum(-1).min().item() == K, (
        f"29C configuration still routes to {(w > 0).sum(-1).min().item()} template(s)")
    layer(x, None)  # must not raise


# ---------------------------------------------------------------------------
# Grouped fast path with the quantile router
# ---------------------------------------------------------------------------
def make_layer(topk, impl, chunk=8, quantile=1, seed=0):
    from nanochat.gpt import RemixedLinear
    torch.manual_seed(seed)
    return RemixedLinear(
        D, D, context_dim=16, basis_size=D,
        remixed_linear_kwargs=dict(
            n_templates=K, template_topk=topk, template_routing_learned=True,
            use_quantile_route=quantile, use_output_gate=True, use_basis_gate=False,
            use_context=False, chunk_routing_size=chunk, chunk_route_impl=impl,
        ),
    ).eval()


@pytest.mark.parametrize("quantile", [0, 1])
def test_effective_topk_sees_through_the_router(quantile):
    """template_topk alone does not tell you whether routing is hard top-1."""
    assert make_layer(0, "compose", quantile=quantile)._effective_topk() == K
    assert make_layer(1, "compose", quantile=quantile)._effective_topk() == 1
    assert make_layer(3, "compose", quantile=quantile)._effective_topk() == 3


@pytest.mark.parametrize("quantile", [0, 1])
def test_grouped_only_engages_for_top1(quantile):
    """Soft mixing has no grouped equivalent, so the path must decline it."""
    sig = torch.randn(2, 4, D)
    assert make_layer(0, "grouped", quantile=quantile)._grouped_route(sig, torch.float32) is None
    assert make_layer(3, "grouped", quantile=quantile)._grouped_route(sig, torch.float32) is None
    assert make_layer(1, "compose", quantile=quantile)._grouped_route(sig, torch.float32) is None
    assert make_layer(1, "grouped", quantile=quantile)._grouped_route(sig, torch.float32) is not None


@pytest.mark.parametrize("quantile", [0, 1])
def test_grouped_matches_compose(quantile):
    """The fast path must be a pure throughput change, including for _qrouter=1."""
    a = make_layer(1, "compose", quantile=quantile)
    b = make_layer(1, "grouped", quantile=quantile)
    b.load_state_dict(a.state_dict())
    x = torch.randn(3, 32, D)
    ya, yb = a(x, None), b(x, None)
    err = (ya - yb).abs().max().item() / max(ya.abs().max().item(), 1e-9)
    assert err < 1e-5, f"grouped diverges from compose by rel {err:.3e}"


@pytest.mark.parametrize("quantile", [0, 1])
def test_grouped_keeps_the_router_in_the_graph(quantile):
    """DDP raises on a parameter that requires grad and receives None.

    The coefficient is numerically 1.0 in the legacy gate, so it is tempting to
    skip the multiply. Skipping it detaches the router.
    """
    layer = make_layer(1, "grouped", quantile=quantile)
    layer(torch.randn(2, 32, D), None).sum().backward()
    router_params = [p for n, p in layer.named_parameters()
                     if "route" in n and p.requires_grad]
    assert router_params, "no router parameter found"
    for p in router_params:
        assert p.grad is not None, "router parameter received no gradient at all"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
