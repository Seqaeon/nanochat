"""Tier 2 idea 5: the inverse-width stack.

Depth collapse is not starvation. Tier 0 showed the model simply stops building hierarchy
past the first exit: moving --eet-min-exit-layer from 1 to 4 moved the flat region from
layer 3 to layer 5 and took bpb from 1.0622 to 1.0194. And T0B showed the readability
pressure cannot be routed around, because a DENSE model asked to be readable at every
depth loses 0.033 bpb and 94% of its depth hierarchy with no exits involved at all.

So instead of fighting the pressure, accept it and give the deep layers something the
shallow ones structurally cannot have: width. A block running 10% of the tokens can be
many times wider at the same FLOPs, because block cost scales with (tokens x width^2).
"""
import math

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig, eet_active_fractions, resolve_ffn_schedule
from nanochat.eet import EarlyExitGPT


def _cfg(**over):
    kw = dict(n_layer=8, n_head=4, n_kv_head=4, n_embd=512, vocab_size=1024,
              sequence_len=128, window_pattern="L", use_eet=True,
              eet_global_router=True, eet_compute_skip=True, eet_min_exit_layer=1,
              eet_capacity_schedule='bell', eet_target_active_frac=0.10,
              eet_warmup_frac=0.0, eet_explore_frac=0.0, eet_loss_variant='none',
              eet_router_task_grad=False, eet_gumbel_temp_start=0.0)
    kw.update(over)
    return GPTConfig(**kw)


def test_active_fractions_match_the_forward_pass():
    """The construction-time schedule must equal the one the forward pass enforces.

    The FFN widths are chosen before any forward has run, from a copy of the capacity
    schedule that lives in gpt.py. If that copy drifts from eet.py's inline version, every
    layer is sized for the wrong token count and the FLOP accounting silently lies.
    """
    for min_exit in (1, 2, 4):
        for target in (0.10, 0.25):
            cfg = _cfg(eet_min_exit_layer=min_exit, eet_target_active_frac=target)
            predicted = eet_active_fractions(cfg.n_layer, min_exit, target, 'bell')
            torch.manual_seed(0)
            m = EarlyExitGPT(cfg)
            m.train()
            x = torch.randint(0, cfg.vocab_size, (2, cfg.sequence_len))
            y = torch.randint(0, cfg.vocab_size, (2, cfg.sequence_len))
            m(x, y, eet_do_route=True, eet_phase=3)
            actual = [c / cfg.sequence_len for c in m._last_active_counts]
            assert len(actual) == len(predicted)
            for i, (p_, a_) in enumerate(zip(predicted, actual)):
                assert abs(p_ - a_) < 0.02, (min_exit, target, i, p_, a_)


def test_width_grows_where_tokens_are_scarce():
    """The multiplier must rise monotonically with depth and be capped."""
    a = eet_active_fractions(8, 1, 0.10, 'bell')
    mult = resolve_ffn_schedule(_cfg(eet_width_power=1.0, eet_width_cap=16.0), 8)
    assert all(mult[i] <= mult[i + 1] + 1e-9 for i in range(7)), mult
    assert max(mult) <= 16.0 * 4.0 + 1e-6
    # Layer 7 sees a tenth of the tokens, so at power 1 it should be ~10x the base width.
    assert mult[-1] / mult[0] > 8.0, mult
    # power 0 must leave the stack exactly uniform.
    assert resolve_ffn_schedule(_cfg(eet_width_power=0.0), 8) == [4.0] * 8


def test_power_one_is_flop_neutral_per_layer():
    """At power 1 every layer's FFN costs the same, however few tokens reach it.

    That is the whole claim: the routing saving is spent on width at depth rather than
    banked. Anything less than power 1 banks part of it.
    """
    a = eet_active_fractions(8, 1, 0.10, 'bell')
    for power, want in ((0.0, 0.606), (1.0, 1.0)):
        mult = resolve_ffn_schedule(_cfg(eet_width_power=power, eet_width_cap=1e9), 8)
        cost = [m * ai for m, ai in zip(mult, a)]
        rel = sum(cost) / (4.0 * 8)
        assert abs(rel - want) < 0.01, (power, rel, want)
        if power == 1.0:
            assert max(cost) - min(cost) < 1e-6, cost


def test_the_cap_binds_and_is_reported_in_the_flop_ratio():
    """A cap makes the stack cheaper than power alone implies; it must not be silent."""
    a = eet_active_fractions(8, 1, 0.10, 'bell')
    uncapped = resolve_ffn_schedule(_cfg(eet_width_power=1.0, eet_width_cap=1e9), 8)
    capped = resolve_ffn_schedule(_cfg(eet_width_power=1.0, eet_width_cap=4.0), 8)
    assert capped[-1] < uncapped[-1]
    rel = lambda m: sum(mi * ai for mi, ai in zip(m, a)) / (4.0 * 8)
    assert rel(capped) < rel(uncapped)


@pytest.mark.parametrize("power", [0.5, 1.0])
def test_the_model_builds_and_trains(power):
    cfg = _cfg(eet_width_power=power, eet_width_cap=16.0)
    torch.manual_seed(0)
    m = EarlyExitGPT(cfg)
    m.train()
    x = torch.randint(0, cfg.vocab_size, (2, cfg.sequence_len))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.sequence_len))
    loss = m(x, y, eet_do_route=True, eet_phase=3)
    assert torch.isfinite(loss)
    loss.backward()
    # The widened deep block must actually be wider and must receive gradient.
    deep = m.transformer.h[-1].mlp.c_fc
    shallow = m.transformer.h[0].mlp.c_fc
    assert deep.weight.shape[0] > shallow.weight.shape[0], (deep.weight.shape, shallow.weight.shape)
    assert deep.weight.grad is not None and torch.isfinite(deep.weight.grad).all()


def test_width_power_is_inert_without_eet():
    """A dense run must be untouched: the schedule keys off the capacity schedule."""
    cfg = GPTConfig(n_layer=8, n_head=4, n_kv_head=4, n_embd=512, vocab_size=1024,
                    sequence_len=128, window_pattern="L", eet_width_power=1.0)
    assert resolve_ffn_schedule(cfg, 8) == [4.0] * 8
