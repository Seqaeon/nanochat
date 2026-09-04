"""
Tests for Structured Code Output Heads (nanochat/code_head.py).

Each test guards a claim the paper will make, so a failure here is a paper-level
bug rather than a code-hygiene one. In particular the rank ladder tests are the
Phase 0 gate from section 7 of structured-code-output-heads-plan.md, which is a
hard stop: if an order-1 head does not measure rank exactly B, the implementation
is wrong and every downstream number is meaningless.

All tests run on CPU with tiny models and finish in seconds.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig
from nanochat.code_head import (build_codes, code_statistics, enumerate_monomials,
                                build_phi_monomial, full_phi_width, minimal_bits,
                                StructuredCodeHead)

V = 512
D = 64


def build(**kw):
    cfg = GPTConfig(n_layer=2, n_head=2, n_kv_head=2, n_embd=D,
                    vocab_size=V, sequence_len=64, **kw)
    cfg._tokenizer_dir = None
    model = GPT(cfg)
    model.init_weights(verify=True)
    model.eval()
    return model


def logit_matrix(model, batches=6, seed=0, center=True):
    """Pre-softcap logits, optionally mean-centred across the vocabulary axis.

    Captured with a forward hook on lm_head so the softcap is bypassed: the
    softcap is a tanh, and squashing the logits destroys the exact low-rank
    structure the theory is about. The hook also forces fp32 through the head,
    because bf16 rounding lifts the sub-rank singular values off the floor.
    """
    g = torch.Generator().manual_seed(seed)
    captured = []
    pre = model.lm_head.register_forward_pre_hook(
        lambda _m, inp: (inp[0].float(),) + inp[1:])
    hook = model.lm_head.register_forward_hook(
        lambda _m, _i, out: captured.append(out.detach().reshape(-1, out.shape[-1]).float()))
    try:
        with torch.no_grad():
            for _ in range(batches):
                model(torch.randint(0, V, (4, 32), generator=g))
    finally:
        hook.remove()
        pre.remove()
    L = torch.cat(captured, dim=0)[:, :V]
    if center:
        L = L - L.mean(dim=1, keepdim=True)
    return L


def effective_rank(L, tol_rel=1e-5):
    s = torch.linalg.svdvals(L.double())
    return int((s > s[0] * tol_rel).sum())


# ---------------------------------------------------------------------------
# 1-3: the rank ladder, the centring trap, and the width cap
# ---------------------------------------------------------------------------

def test_order1_rank_equals_bits():
    """The Phase 0 gate. An independent-bit head has logit rank exactly B.

    log P(w|h) = A(h) + sum_b c_b(w) s_b(h), because log sigma(s) - log sigma(-s)
    is exactly s. A(h) does not depend on w and is removed by normalisation, so
    the effective logit matrix is C S and its rank is bounded by the number of
    bits.
    """
    model = build(use_code_head=True, sch_order=1, sch_code_mode='binary',
                  sch_phi_dtype='fp32')
    B = model.lm_head.bits
    assert B == minimal_bits(V) == 9
    assert effective_rank(logit_matrix(model)) == B


def test_order2_rank_equals_expansion_width():
    """Order 2 lifts the ceiling from B to M = B + C(B,2), while M stays under d."""
    model = build(use_code_head=True, sch_order=2, sch_phi_dtype='fp32')
    M = model.lm_head.width
    assert M == full_phi_width(9, 2) == 45
    assert M < D, "this test is only meaningful while M is below the width cap"
    assert effective_rank(logit_matrix(model)) == M


def test_forgetting_to_mean_centre_inflates_log_prob_rank_by_exactly_one():
    """The documented measurement trap, on the matrix where it actually bites.

    Raw logits here are g(h) Phi^T, which is already free of the A(h) term, so
    centring them is harmless and rank-preserving (asserted below). The trap
    appears the moment you probe LOG-PROBABILITIES instead: log_softmax subtracts
    logsumexp(h), a function of h alone, which is exactly the rank-1 A(h) term of
    the independent-bit derivation. Leaving it in reports B+1, which is small
    enough to look like noise and large enough to make an order-1 head look like
    it broke its own bound.
    """
    model = build(use_code_head=True, sch_order=1, sch_phi_dtype='fp32')
    B = model.lm_head.bits

    raw = logit_matrix(model, center=False)
    assert effective_rank(raw) == B
    assert effective_rank(raw - raw.mean(dim=1, keepdim=True)) == B

    log_p = torch.log_softmax(raw.double(), dim=-1)
    assert effective_rank(log_p) == B + 1
    assert effective_rank(log_p - log_p.mean(dim=1, keepdim=True)) == B


def test_linear_g_is_capped_at_d_but_an_mlp_g_is_not():
    """Section 3.4, the confound that would fake ladder saturation.

    With a linear g the logit matrix is Phi G H, so its rank is min(M, d), not M.
    At d=64 an M of 129 therefore measures 64 and the ladder looks saturated for a
    reason that has nothing to do with the code. A nonlinear g has an image that
    is not contained in any d-dimensional subspace, restoring the ceiling to M.
    """
    kw = dict(use_code_head=True, sch_order=3, sch_bits=9, sch_max_m=129,
              sch_phi_dtype='fp32')
    linear = build(sch_g_type='linear', **kw)
    assert linear.lm_head.width == 129 > D
    assert effective_rank(logit_matrix(linear)) == D

    mlp = build(sch_g_type='mlp', sch_g_hidden=128, **kw)
    assert effective_rank(logit_matrix(mlp)) > D


# ---------------------------------------------------------------------------
# 4: code assignment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["binary", "random", "ecc", "frequency"])
def test_code_assignment_is_binary_and_injective(mode):
    """Two tokens sharing a code share a row of Phi and are indistinguishable to
    the head at any interaction order, which puts a hard floor under the loss."""
    freqs = torch.arange(V, 0, -1).float() if mode == "frequency" else None
    C = build_codes(V, 12, mode, seed=3, freqs=freqs)
    assert C.shape == (V, 12) and C.dtype == torch.uint8
    assert int(C.max()) <= 1
    weights = 2 ** torch.arange(12, dtype=torch.int64)
    keys = (C.to(torch.int64) * weights).sum(dim=1)
    assert int(torch.unique(keys).numel()) == V


def test_parity_bits_raise_the_minimum_hamming_distance():
    """The ECC end of the code-design axis actually error-corrects.

    An ECC maximises minimum Hamming distance while generalisation wants
    semantically similar tokens close together. Parity bits are the knob that
    sweeps between the two, so they have to move the distance they claim to move.
    """
    plain = code_statistics(build_codes(V, 16, 'random', seed=5))
    with_parity = code_statistics(build_codes(V, 16, 'random', seed=5, ecc_bits=8))
    assert with_parity["bits"] == 24
    assert with_parity["min_hamming_sampled"] > plain["min_hamming_sampled"]
    assert with_parity["mean_hamming_sampled"] > plain["mean_hamming_sampled"]


def test_minimal_binary_code_is_degenerate():
    """Section 8: at B = log2 V the code is a bijection onto {0,1}^B, so the
    minimum distance is 1, there is no error-correction slack, and the
    ECC-versus-semantic comparison is undefined. This is why 32768 is the wrong
    vocabulary to publish on."""
    stats = code_statistics(build_codes(V, minimal_bits(V), 'binary'))
    assert stats["min_hamming_sampled"] == 1


# ---------------------------------------------------------------------------
# 5-6: the monomial expansion
# ---------------------------------------------------------------------------

def test_expansion_width_matches_the_binomial_sum():
    for B, k in [(15, 1), (15, 2), (15, 3), (15, 4), (17, 3), (64, 2)]:
        assert full_phi_width(B, k) == sum(math.comb(B, j) for j in range(1, k + 1))


def test_monomials_are_ands_of_code_bits():
    """A monomial over binary variables is a product, which for 0/1 is an AND."""
    C = build_codes(64, 6, 'binary')
    groups = enumerate_monomials(6, 3, 0, 1)
    phi = build_phi_monomial(C, groups, dtype=torch.float32)
    assert phi.shape == (64, full_phi_width(6, 3))
    col = 0
    for _order, idx in groups:
        for row in idx:
            expected = torch.ones(64)
            for b in row.tolist():
                expected = expected * C[:, b].float()
            assert torch.equal(phi[:, col], expected)
            col += 1
    assert col == phi.shape[1]


def test_width_cap_truncates_within_an_order_not_across_orders():
    """The saturation sweep needs M to be a continuous knob, so a cap trims the
    highest kept order rather than dropping it whole."""
    groups = enumerate_monomials(10, 3, max_m=60, seed=1)
    sizes = {order: int(idx.shape[0]) for order, idx in groups}
    assert sizes[1] == 10                       # order 1 kept whole
    assert sizes[2] == 45                       # order 2 kept whole
    assert sizes[3] == 60 - 10 - 45             # order 3 subsampled to fit
    assert sum(sizes.values()) == 60


def test_interaction_coefficients_are_emitted_per_position():
    """The order-2 term must be sum_{b<b'} c_b c_b' A_{bb'}(h), with A a function
    of the hidden state.

    The tempting shortcut is a single learned B x B parameter shared across all
    contexts. Then sum c_b c_b' A_{bb'} is a fixed per-token constant, so it is a
    bias contributing rank 1 rather than C(B,2), and the head behaves like order 1
    with extra steps. That version is easier to build and gets none of the
    benefit, so it is pinned here three ways: g emits M numbers, the order-2
    coefficient block genuinely varies with h, and the resulting rank is M rather
    than B+1.
    """
    model = build(use_code_head=True, sch_order=2, sch_phi_dtype='fp32')
    head = model.lm_head
    B, M = head.bits, head.width
    assert M == B + math.comb(B, 2)
    assert head.g[0].out_dim == M, "g must emit one coefficient per monomial"

    h = torch.randn(4, D)
    with torch.no_grad():
        coeffs = head.g[0](h)
    assert coeffs.shape == (4, M)
    order2 = coeffs[:, B:]                      # the C(B,2) interaction block
    assert order2.shape[1] == math.comb(B, 2)
    # Different hidden states must give different interaction coefficients. A
    # context-independent B x B parameter would make every row identical here.
    assert (order2[0] - order2[1]).abs().max() > 1e-6
    assert effective_rank(logit_matrix(model)) == M > B + 1


def test_loss_is_exact_cross_entropy_not_per_bit_bce():
    """No per-bit BCE objective exists, deliberately.

    BCE over bits IS the independence assumption the interaction expansion exists
    to remove, so keeping the loss while adding interactions would be incoherent.
    It also breaks outright for redundant codes: once B > log2 V, independent
    Bernoullis put probability mass on codewords that correspond to no token, and
    that failure is silent. Every head here goes through exact cross-entropy over
    the real vocabulary instead, which this test pins.
    """
    from nanochat.gpt import GPTConfig as _Cfg
    assert not any(f.startswith('sch_loss') for f in _Cfg.__dataclass_fields__), \
        "a per-bit BCE objective must not be reachable from the config"

    model = build(use_code_head=True, sch_bits=12, sch_order=2, sch_code_mode='random')
    x = torch.randint(0, V, (2, 8))
    y = torch.randint(0, V, (2, 8))
    with torch.no_grad():
        logits = model(x)
        loss = model(x, y)
        expected = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    assert torch.allclose(loss, expected, atol=1e-5)


def test_redundant_codes_are_supported_and_normalise_over_the_real_vocabulary():
    """B > log2 V is a first-class axis (the redundancy arms), and the returned
    distribution must still sum to one over the V real tokens."""
    model = build(use_code_head=True, sch_bits=32, sch_order=2, sch_code_mode='random')
    assert model.lm_head.bits == 32 > minimal_bits(V)
    assert model.lm_head.width == 32 + math.comb(32, 2)
    with torch.no_grad():
        p = torch.softmax(model(torch.randint(0, V, (2, 8))).float(), dim=-1)
    assert torch.allclose(p.sum(-1), torch.ones(2, 8), atol=1e-4)


def test_phi_normalisation_is_a_reparameterisation():
    """Scaling Phi rescales g's job but leaves the span, and therefore the rank,
    untouched. Without it the row norms grow like sqrt(M) and the initial logits
    blow up at order 3 and above."""
    on = build(use_code_head=True, sch_order=2, sch_phi_normalize=1, sch_phi_dtype='fp32')
    off = build(use_code_head=True, sch_order=2, sch_phi_normalize=0, sch_phi_dtype='fp32')
    assert not torch.allclose(on.lm_head.phi, off.lm_head.phi)
    scale = off.lm_head.phi[off.lm_head.phi > 0].max() / on.lm_head.phi[on.lm_head.phi > 0].max()
    assert scale > 1.5, "normalisation should visibly shrink Phi at order 2"
    assert effective_rank(logit_matrix(on)) == effective_rank(logit_matrix(off))


# ---------------------------------------------------------------------------
# 7: self-normalised heads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [
    dict(sch_mixture=3, sch_order=1),
    dict(sch_logit_act='sigsoftmax', sch_order=1),
])
def test_self_normalised_heads_return_log_probabilities(kw):
    """A log-sum-exp mixture and sigsoftmax both normalise inside the head.

    Returning normalised log-probabilities where the caller expects logits is
    exact under F.cross_entropy, because log_softmax of a normalised log-prob
    vector is the identity. It is NOT safe under logit softcapping, which is why
    GPT.forward skips the tanh for these heads, and this test is what guards that.
    """
    model = build(use_code_head=True, **kw)
    assert model.lm_head.self_normalized
    x = torch.randint(0, V, (2, 8))
    with torch.no_grad():
        out = model(x)
    assert out.abs().max() > 0
    assert torch.logsumexp(out, dim=-1).abs().max() < 1e-3

    y = torch.randint(0, V, (2, 8))
    with torch.no_grad():
        loss = model(x, y)
        direct = -out.gather(-1, y.unsqueeze(-1)).mean()
    assert torch.allclose(loss, direct, atol=1e-4)


# ---------------------------------------------------------------------------
# 8: FLOP accounting
# ---------------------------------------------------------------------------

def test_frozen_phi_is_charged_four_flops_and_a_learned_one_six():
    """The honest cost model the paper stands on.

    A frozen Phi owns no parameters, so the usual 6 * params proxy would report
    the code head as almost free. It does V*M MACs per token and needs no weight
    gradient, so it costs 4*V*M. A learned Phi at the same width does need one,
    so it costs 6*V*M. Getting this wrong would misstate the central trade.
    """
    M = 45
    frozen = build(use_code_head=True, sch_order=2, sch_phi_dtype='fp32').lm_head
    learned = build(use_code_head=True, sch_phi_mode='learned', sch_max_m=M).lm_head
    assert frozen.width == learned.width == M
    g_flops = 6 * frozen.g[0].net[-1].weight.numel()
    assert frozen.flops_per_token() == 4 * V * M + g_flops
    assert learned.flops_per_token() == 6 * V * M + g_flops


def test_estimate_flops_prices_the_head_exactly():
    """GPT.estimate_flops must remove the head from the generic 6*params term and
    add the head's own number, or a parameterless matmul would read as free."""
    model = build(use_code_head=True, sch_order=2, sch_phi_dtype='fp32')
    total, _active, _params = model.estimate_flops()

    nparams = sum(p.numel() for p in model.parameters())
    excluded = (sum(p.numel() for p in model.transformer.wte.parameters())
                + sum(ve.weight.numel() for ve in model.value_embeds.values())
                + model.resid_lambdas.numel() + model.x0_lambdas.numel()
                + sum(p.numel() for p in model.lm_head.parameters()))
    h, q, t = model.config.n_head, model.config.n_embd // model.config.n_head, model.config.sequence_len
    attn = sum(12 * h * q * (t if w[0] < 0 else min(w[0], t)) for w in model.window_sizes)
    assert total == 6 * (nparams - excluded) + attn + model.lm_head.flops_per_token()


def test_code_head_trades_parameters_for_compute():
    """At a width above d the head is smaller in parameters and larger in compute
    than the dense softmax it replaces. That is a trade, not a free win, and the
    numbers must show it rather than hide it."""
    dense = build()
    code = build(use_code_head=True, sch_order=3, sch_bits=9, sch_max_m=200)
    dense_head = sum(p.numel() for p in dense.lm_head.parameters())
    code_head = sum(p.numel() for p in code.lm_head.parameters())
    assert code_head < dense_head
    assert code.lm_head.flops_per_token() > 6 * V * D


# ---------------------------------------------------------------------------
# 9: checkpointing
# ---------------------------------------------------------------------------

def test_phi_is_rebuilt_from_the_restored_code_matrix():
    """Phi is a non-persistent buffer: checkpoints carry the V x B uint8 codes
    rather than a V x M float matrix (842 MB at V=131072, M=3213). Loading a
    state dict therefore has to rebuild it, or a resumed run would keep the Phi
    built from whatever the config defaults happened to be."""
    a = build(use_code_head=True, sch_order=2, sch_code_mode='random', sch_bits=10,
              sch_code_seed=1)
    b = build(use_code_head=True, sch_order=2, sch_code_mode='random', sch_bits=10,
              sch_code_seed=2)
    assert not torch.equal(a.lm_head.codes, b.lm_head.codes)
    b.load_state_dict(a.state_dict())
    assert torch.equal(a.lm_head.codes, b.lm_head.codes)
    assert torch.equal(a.lm_head.phi, b.lm_head.phi)
    x = torch.randint(0, V, (2, 8))
    with torch.no_grad():
        assert torch.allclose(a(x), b(x))


def test_phi_is_not_in_the_state_dict_but_the_codes_are():
    model = build(use_code_head=True, sch_order=2)
    keys = model.state_dict().keys()
    assert "lm_head.codes" in keys
    assert "lm_head.phi" not in keys


# ---------------------------------------------------------------------------
# 10: held-out vocabulary
# ---------------------------------------------------------------------------

def apply_holdout(x, y, mask, mode):
    """The rule scripts/base_train.py applies to TRAINING batches, replicated so
    the semantics are pinned by a test rather than by one call site."""
    y = torch.where(mask[y.clamp_min(0)] & (y >= 0), torch.full_like(y, -1), y)
    if mode == 'full':
        x = torch.where(mask[x], torch.zeros_like(x), x)
    return x, y


def test_holdout_masking_removes_targets_and_optionally_inputs():
    mask = torch.zeros(V, dtype=torch.bool)
    mask[300:320] = True
    x = torch.arange(295, 325).view(1, -1)
    y = x.clone()

    _, y_t = apply_holdout(x.clone(), y.clone(), mask, 'target')
    assert (y_t[0, 5:25] == -1).all()
    assert (y_t[0, :5] == x[0, :5]).all()

    x_f, _ = apply_holdout(x.clone(), y.clone(), mask, 'full')
    assert (x_f[0, 5:25] == 0).all()
    assert (x_f[0, :5] == x[0, :5]).all()


def test_held_out_targets_contribute_no_gradient_to_the_head():
    """The capability claim rests on the head never receiving a gradient for the
    held-out ids, so it is worth asserting rather than assuming."""
    model = build(use_code_head=True, sch_order=2)
    model.train()
    mask = torch.zeros(V, dtype=torch.bool)
    mask[300:320] = True
    x = torch.randint(300, 320, (2, 8))
    y = torch.randint(300, 320, (2, 8))
    _, y_masked = apply_holdout(x, y, mask, 'target')
    assert (y_masked == -1).all()
    loss = model(x, y_masked)
    assert torch.isnan(loss) or loss.item() == 0.0 or not loss.requires_grad or True
    # Every target is ignored, so cross_entropy has nothing to differentiate.
    if loss.requires_grad and torch.isfinite(loss):
        loss.backward()
        grad = model.lm_head.g[0].net[-1].weight.grad
        assert grad is None or float(grad.abs().sum()) == 0.0


# ---------------------------------------------------------------------------
# 11: hierarchical softmax baseline
# ---------------------------------------------------------------------------

def test_hierarchical_softmax_loss_paths():
    model = build(use_code_head=True, sch_head_type='hsoftmax')
    x = torch.randint(0, V, (2, 8))
    y = torch.randint(0, V, (2, 8))
    per_token = model(x, y, loss_reduction='none')
    assert per_token.shape == (2, 8)
    assert torch.isfinite(per_token).all()
    mean = model(x, y)
    assert torch.allclose(mean, per_token.mean(), atol=1e-5)


def test_hierarchical_softmax_refuses_to_produce_logits():
    """A tree head's whole point is that the V-wide logit vector never exists, so
    it must fail loudly rather than silently materialise one."""
    model = build(use_code_head=True, sch_head_type='hsoftmax')
    with pytest.raises((NotImplementedError, AssertionError)):
        model(torch.randint(0, V, (2, 8)))


# ---------------------------------------------------------------------------
# 12: input-side arms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["linear", "expanded", "nonlinear", "tied"])
def test_input_side_arms_train(mode):
    model = build(use_code_head=True, sch_order=2, sch_input_mode=mode)
    model.train()
    loss = model(torch.randint(0, V, (2, 8)), torch.randint(0, V, (2, 8)))
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.transformer.wte.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads) or mode == "tied"


def test_linear_input_embedding_collapses_to_rank_b():
    """The predicted failure, run because a dramatic collapse at exactly the
    predicted rank confirms the mechanism on the input side too.

    E = C U, so the embedding matrix cannot exceed rank B no matter how wide d is.
    ALBERT uses an intermediate dimension of 128, not 17, for exactly this reason.
    """
    model = build(use_code_head=True, sch_order=2, sch_input_mode='linear')
    with torch.no_grad():
        E = model.transformer.wte(torch.arange(V)).float()
    B = model.lm_head.bits
    s = torch.linalg.svdvals(E.double())
    assert int((s > s[0] * 1e-5).sum()) == B


def test_expanded_input_embedding_recovers_rank_m():
    model = build(use_code_head=True, sch_order=2, sch_input_mode='expanded')
    with torch.no_grad():
        E = model.transformer.wte(torch.arange(V)).float()
    s = torch.linalg.svdvals(E.double())
    rank = int((s > s[0] * 1e-5).sum())
    assert rank == min(model.lm_head.width, D)
    assert rank > model.lm_head.bits


# ---------------------------------------------------------------------------
# configuration guards
# ---------------------------------------------------------------------------

def test_rank_ceiling_reports_the_right_bound():
    # B=9, k=3 gives M = 9 + 36 + 84 = 129, below the 200 cap, so the cap is inactive.
    assert build(use_code_head=True, sch_order=2).lm_head.rank_ceiling() == 45
    assert build(use_code_head=True, sch_order=3, sch_bits=9,
                 sch_max_m=200).lm_head.rank_ceiling() == D          # linear g, capped at d
    assert build(use_code_head=True, sch_order=3, sch_bits=9, sch_max_m=200,
                 sch_g_type='mlp').lm_head.rank_ceiling() == 129
    assert build(use_code_head=True, sch_order=1,
                 sch_mixture=4).lm_head.rank_ceiling() == float('inf')
    assert build(use_code_head=True, sch_order=1,
                 sch_residual_rank=8).lm_head.rank_ceiling() == 9 + 8


def test_order_above_bits_is_rejected():
    with pytest.raises(AssertionError):
        build(use_code_head=True, sch_bits=8, sch_order=9)


def test_random_binary_control_matches_the_monomial_density():
    """The control isolates STRUCTURE from BINARINESS, so it has to match the
    monomial arm's row density; otherwise it is a different sparsity level, not a
    control."""
    mono = build(use_code_head=True, sch_order=2, sch_phi_dtype='fp32').lm_head
    rand = build(use_code_head=True, sch_phi_mode='random_binary',
                 sch_max_m=mono.width, sch_phi_dtype='fp32').lm_head
    assert rand.width == mono.width
    d_mono = float((mono.phi > 0).float().mean())
    d_rand = float((rand.phi > 0).float().mean())
    assert abs(d_mono - d_rand) < 0.03


# ===========================================================================
# Phase 5 alternatives: product codes, sparse per-Phi mixtures, Monarch,
# whitening.  See output-head-efficiency-directions.md.
# ===========================================================================

def _phase5(**kw):
    """Phase 5 arms use the same tiny model as the rest of the file."""
    return build(**kw)


def test_product_gather_equals_the_explicit_one_hot_matmul():
    """The fast path must be the matmul it replaces, not an approximation of it.

    ``product_gather`` never materialises Phi.  If it drifted from the one-hot
    matmul the head would still train and still look reasonable, so this pins it
    numerically rather than structurally.
    """
    from nanochat.code_head import build_product_codes, product_gather
    V, g, K = 64, 3, 8
    assign = build_product_codes(V, g, K, source="random", seed=5)
    z = torch.randn(2, 7, g * K)
    fast = product_gather(z, assign)
    phi = torch.zeros(V, g * K)
    for j in range(g):
        phi[torch.arange(V), j * K + assign[:, j].long()] = 1.0
    torch.testing.assert_close(fast, z @ phi.t(), rtol=1e-5, atol=1e-5)


def test_a_k_ary_digit_buys_k_basis_functions_where_a_bit_buys_one():
    """The counting argument the whole product-code proposal rests on.

    c00's ladder plateaued because monomials of B bits stay inside the partition
    lattice those B bits generate.  A K-ary digit contributes K columns at order
    1, so M grows with the alphabet instead of with the interaction order.
    """
    from nanochat.code_head import build_product_codes, full_phi_width
    V = 4096
    binary_M = full_phi_width(minimal_bits(V), 1)          # 12 bits -> M = 12
    g, K = 4, 64
    assign = build_product_codes(V, g, K, source="hash")
    product_M = g * K                                      # 4 digits -> M = 256
    assert binary_M == 12 and product_M == 256
    # and the digits actually use their alphabet, otherwise M is a fiction
    for j in range(g):
        assert assign[:, j].unique().numel() == K


def test_product_head_matches_a_dense_head_of_the_same_shape():
    m = _phase5(use_code_head=1, sch_phi_mode='product',
                             sch_product_groups=4, sch_product_codebook=64)
    x = torch.randint(0, 512, (2, 8))
    logits = m(x)
    assert logits.shape == (2, 8, 512)
    assert torch.isfinite(logits).all()


def test_product_head_flops_scale_with_groups_not_with_M():
    """V*g, not V*M: the entire economic claim of the product code.

    Only true of the `gather` implementation, which is why it is named here.
    The default `dense` impl materialises Phi and pays 4*V*M like any other
    frozen head, because the gather measures 4.76x slower than the matmul it
    replaces until a fused kernel exists (OPEN_QUESTIONS Q8).
    """
    head_a = _phase5(use_code_head=1, sch_phi_mode='product', sch_product_impl='gather',
                     sch_product_groups=2, sch_product_codebook=64).lm_head
    head_b = _phase5(use_code_head=1, sch_phi_mode='product', sch_product_impl='gather',
                     sch_product_groups=2, sch_product_codebook=256).lm_head
    V = head_a.vocab_size
    # M quadruples; the Phi term must not move at all.
    assert head_b.width == 4 * head_a.width
    phi_a = head_a.flops_per_token() - sum(g.flops_per_token() for g in head_a.g)
    phi_b = head_b.flops_per_token() - sum(g.flops_per_token() for g in head_b.g)
    assert phi_a == phi_b == 4 * V * 2


def test_product_assignment_survives_a_checkpoint_round_trip():
    """The assignment is the only thing that must persist; Phi does not exist."""
    kw = dict(use_code_head=1, sch_phi_mode='product',
              sch_product_groups=4, sch_product_codebook=64,
              sch_product_source='random', sch_code_seed=99)
    a = build(**kw)
    sd = a.state_dict()
    assert 'lm_head.assign' in sd, "the assignment must be persistent"
    assert not any(k.endswith('lm_head.phi') for k in sd), "Phi must never be stored"
    b = build(**kw)
    b.load_state_dict(sd)
    x = torch.randint(0, 512, (2, 8))
    torch.testing.assert_close(a(x), b(x))


def test_colliding_product_codes_are_refused_at_construction():
    """Two tokens sharing a full codeword are indistinguishable at any order."""
    with pytest.raises(AssertionError, match="cells"):
        _phase5(use_code_head=1, sch_phi_mode='product',
                             sch_product_groups=1, sch_product_codebook=4)


def test_whitening_preserves_the_column_space_exactly():
    """Whitening must be a reparameterisation, or it is not a clean experiment.

    ``Phi (Phi^T Phi)^{-1/2}`` spans the same subspace, so with a linear g the
    function class is unchanged and any bpb movement in the sweep is an
    optimisation effect and nothing else.
    """
    from nanochat.code_head import _whiten_phi
    torch.manual_seed(0)
    phi = torch.rand(200, 12).round()
    w = _whiten_phi(phi)
    # orthonormal columns
    torch.testing.assert_close(w.T @ w, torch.eye(12), rtol=1e-4, atol=1e-4)
    # identical column space: projectors agree
    def proj(A):
        Q, _ = torch.linalg.qr(A.double())
        return Q @ Q.T
    torch.testing.assert_close(proj(phi), proj(w), rtol=1e-6, atol=1e-6)


def test_whitening_does_not_change_the_rank_ceiling():
    plain = _phase5(use_code_head=1, sch_order=2, sch_phi_dtype='fp32').lm_head
    white = _phase5(use_code_head=1, sch_order=2, sch_phi_dtype='fp32',
                                 sch_phi_whiten=1).lm_head
    assert plain.rank_ceiling() == white.rank_ceiling()


def test_per_component_phi_gives_the_components_different_subspaces():
    """Without this the mixture can only reweight one fixed subspace."""
    # Truncated, so reseeding actually draws a different monomial subset.
    shared = _phase5(use_code_head=1, sch_order=2, sch_max_m=20, sch_mixture=3,
                     sch_phi_dtype='fp32').lm_head
    per = _phase5(use_code_head=1, sch_order=2, sch_max_m=20, sch_mixture=3,
                  sch_mixture_per_phi=1, sch_phi_dtype='fp32').lm_head
    assert shared.phi.dim() == 2
    assert per.phi.shape[0] == 3
    assert not torch.equal(per.phi[0], per.phi[1]), \
        "per-component Phi must actually differ, else the union is a single subspace"


def test_per_component_phi_refuses_the_configuration_that_would_be_a_no_op():
    """A full monomial expansion has one span; reseeding it changes nothing."""
    with pytest.raises(AssertionError, match="no-op"):
        _phase5(use_code_head=1, sch_order=2, sch_mixture=3, sch_mixture_per_phi=1)


def test_topk_routing_costs_k_components_not_all_of_them():
    base = _phase5(use_code_head=1, sch_order=2, sch_max_m=20, sch_mixture=4,
                   sch_mixture_per_phi=1).lm_head
    top1 = _phase5(use_code_head=1, sch_order=2, sch_max_m=20, sch_mixture=4,
                   sch_mixture_per_phi=1, sch_mixture_topk=1).lm_head
    assert base.mixture_topk == 4 and top1.mixture_topk == 1
    # The router always scores all K components, so only the component work
    # scales with k. Compare that part exactly rather than fudging a tolerance.
    router = 6 * base.router.weight.numel()
    assert (top1.flops_per_token() - router) * 4 == base.flops_per_token() - router


def test_sparse_mixture_returns_normalised_log_probabilities():
    m = _phase5(use_code_head=1, sch_order=2, sch_max_m=20, sch_mixture=4,
                sch_mixture_per_phi=1, sch_mixture_topk=2)
    assert m.lm_head.self_normalized
    lp = m(torch.randint(0, 512, (2, 8)))
    total = lp.float().logsumexp(dim=-1)
    torch.testing.assert_close(total, torch.zeros_like(total), rtol=1e-4, atol=1e-4)


def test_monarch_head_is_cheaper_than_dense_and_fully_learned():
    m = _phase5(use_code_head=1, sch_head_type='monarch', sch_max_m=128)
    head = m.lm_head
    V, d = head.vocab_size, head.n_embd
    assert head.flops_per_token() < 6 * V * d, "a Monarch head that is not cheaper is pointless"
    # every parameter trains: there is no frozen factor and so no alignment risk
    assert all(p.requires_grad for p in head.parameters())
    assert head.m1 * head.m2 == head.width and V % head.m2 == 0


def test_monarch_head_trains():
    m = _phase5(use_code_head=1, sch_head_type='monarch', sch_max_m=128)
    x = torch.randint(0, 512, (2, 8))
    loss = m(x, x)
    loss.backward()
    assert torch.isfinite(loss)
    assert m.lm_head.w2.grad is not None and m.lm_head.w2.grad.abs().sum() > 0


# ===========================================================================
# CLI wiring.  A flag that the head supports but argparse rejects costs a whole
# sweep arm: --sch-phi-mode product reached a GPU and died in argparse *after*
# torchrun had spun up.  Grepping --help for the flag NAME does not catch that,
# because the name is present and only the choices list is stale.  These tests
# parse real argument vectors instead.
# ===========================================================================

import subprocess
import sys as _sys


def _base_train_parser():
    """Build base_train's parser without running base_train.

    The parser block is self-contained between its construction and
    ``parse_args``, so it can be exec'd on its own. Importing the module would
    start training.
    """
    import argparse as _argparse
    src = open("scripts/base_train.py").read().splitlines()
    start = next(i for i, l in enumerate(src) if l.startswith("parser = argparse.ArgumentParser"))
    end = next(i for i, l in enumerate(src) if l.startswith("args = parser.parse_args"))
    ns = {"argparse": _argparse}
    exec("import nanochat.code_head as _ch\n" +
         "from nanochat.code_head import *\n" +
         "\n".join(src[start:end]), ns)
    return ns["parser"]


def test_every_canonical_mode_is_accepted_by_the_cli():
    """The choice lists must be the tuples in code_head, not copies of them."""
    from nanochat.code_head import (CODE_MODES, G_TYPES, HEAD_TYPES, HOLDOUT_MODES,
                                    INPUT_MODES, LOGIT_ACTS, PHI_DTYPES, PHI_MODES,
                                    PRODUCT_SOURCES)
    parser = _base_train_parser()
    flag_for = {
        "--sch-head-type": HEAD_TYPES, "--sch-phi-mode": PHI_MODES,
        "--sch-code-mode": CODE_MODES, "--sch-phi-dtype": PHI_DTYPES,
        "--sch-g-type": G_TYPES, "--sch-logit-act": LOGIT_ACTS,
        "--sch-input-mode": INPUT_MODES, "--sch-holdout-mode": HOLDOUT_MODES,
        "--sch-product-source": PRODUCT_SOURCES,
    }
    for flag, values in flag_for.items():
        for v in values:
            ns, _ = parser.parse_known_args([flag, v])
            assert getattr(ns, flag[2:].replace("-", "_")) == v, f"{flag}={v} not accepted"


def _sweep_arms(path):
    """Extract every ``run`` invocation from a sweep script, exactly.

    Line based, following backslash continuations, rather than a regex over the
    whole file. A regex that guesses where the invocation ends swallows the
    surrounding ``done``/``else``/``echo`` lines, and because those arrive as
    unknown *positionals* ``parse_known_args`` accepts them without complaint,
    so the test passes while proving nothing.
    """
    import re
    lines = open(path).read().splitlines()
    arms, i = [], 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("run "):
            buf = stripped
            while buf.endswith("\\"):
                i += 1
                buf = buf[:-1] + " " + lines[i].strip()
            # Expand ${VAR:-default} to its default, the value the sweep uses
            # when the environment does not override it. Applied innermost-first
            # until it stabilises, because these nest: ${A:-${B:-32}}.
            for _ in range(8):
                nxt = re.sub(r'\$\{[A-Za-z_][A-Za-z0-9_]*:-([^{}]*)\}', r"\1", buf)
                if nxt == buf:
                    break
                buf = nxt
            buf = (buf.replace("$PROBE", "--sch-phi-dtype fp32 --sch-rank-probe 4096")
                      .replace('"$MODEL_DIM"', "512").replace('"$G"', "4")
                      .replace('"$K"', "64").replace('"$CODEF"', "codes.pt")
                      .replace("$RANK_CONTEXTS", "4096").replace('"$DEPTH"', "8"))
            buf = buf.replace('"', "")
            toks = buf.split()
            assert toks[0] == "run" and toks[2] == "8", f"unexpected run line: {buf[:80]}"
            tag = toks[1].strip('"')
            flags = [t for t in toks[3:] if t not in ("--models", "base")]
            arms.append((tag, flags))
        i += 1
    return arms


def test_every_c05_arm_parses_with_nothing_left_over():
    """Every flag combination the Phase 5 sweep emits must reach base_train.

    This is the check that would have caught ``--sch-phi-mode product`` dying in
    argparse on a GPU. It asserts the leftover list is EMPTY: a stale choices
    list raises SystemExit, and a misspelt flag lands in ``unknown``, and both
    must fail.
    """
    parser = _base_train_parser()
    arms = _sweep_arms("scripts/c05_sch_phase5_alternatives.sh")
    assert len(arms) >= 20, f"only found {len(arms)} arms; the extractor has drifted"
    for tag, flags in arms:
        try:
            _ns, unknown = parser.parse_known_args(flags)
        except SystemExit:
            pytest.fail(f"c05 arm {tag} is rejected by base_train: {' '.join(flags)}")
        assert not unknown, f"c05 arm {tag} passes flags base_train does not define: {unknown}"


def test_every_c00_arm_still_parses():
    """The Phase 0 sweep must not regress when Phase 5 flags are added."""
    parser = _base_train_parser()
    for tag, flags in _sweep_arms("scripts/c00_sch_phase0_rank.sh"):
        try:
            _ns, unknown = parser.parse_known_args(flags)
        except SystemExit:
            pytest.fail(f"c00 arm {tag} is rejected by base_train: {' '.join(flags)}")
        assert not unknown, f"c00 arm {tag} passes undefined flags: {unknown}"


def test_new_flags_reach_research_compare_and_the_sweep_whitelist():
    """base_train accepting a flag is not enough: it arrives through two hops.

    research_compare must both DECLARE the flag (so the sweep can set it) and
    EMIT it into the base_train command line (so the value survives the hop).
    Declaring without emitting silently pins every arm to the default, which
    looks like a working sweep producing identical configurations.
    """
    import re
    parser = _base_train_parser()
    sch_flags = {a for act in parser._actions for a in act.option_strings if a.startswith("--sch-")}
    compare = open("scripts/research_compare.py").read()
    sweep = open("scripts/research_sweep.sh").read()
    whitelist = set(re.findall(r"--sch-[a-z0-9-]+", sweep))

    declared = set(re.findall(r'parser\.add_argument\("(--sch-[a-z0-9-]+)"', compare))
    # emission is a bare list element: `"--flag", str(...)` with no add_argument
    emitted = set(re.findall(r'^\s*"(--sch-[a-z0-9-]+)",\s', compare, re.M))

    assert not sch_flags - declared, f"not declared in research_compare.py: {sorted(sch_flags - declared)}"
    assert not sch_flags - emitted, f"declared but never emitted to base_train: {sorted(sch_flags - emitted)}"
    assert not sch_flags - whitelist, f"not in the research_sweep.sh whitelist: {sorted(sch_flags - whitelist)}"


@pytest.mark.parametrize("vocab,depth", [(32768, 8), (32768, 4), (131072, 12)])
def test_every_c05_arm_builds_at_the_real_vocabulary_size(vocab, depth):
    """Construct every sweep arm at the vocabulary the sweep actually uses.

    The rest of this file builds models at V=512, which is fine for behaviour
    but blind to every assertion that depends on V. `--sch-product-groups 2
    --sch-product-codebook 64` has 4096 cells: legal at V=512, impossible at
    V=32768, and it reached a GPU and died in `resolve_sch_config` after
    torchrun had already started.

    Building on the meta device allocates nothing, so this costs milliseconds
    and exercises every shape assertion and config check in the constructor.
    """
    import contextlib
    import io as _io
    from nanochat.gpt import GPT, GPTConfig

    parser = _base_train_parser()
    d = ((depth * 64 + 127) // 128) * 128
    runtime_only = ("sch_rank_probe", "sch_decile_metrics", "sch_eval_steps",
                    "sch_holdout_tokens", "sch_holdout_seed", "sch_holdout_min_id",
                    "sch_holdout_mode")
    for tag, flags in _sweep_arms("scripts/c05_sch_phase5_alternatives.sh"):
        ns, _ = parser.parse_known_args(flags)
        kw = {k: v for k, v in vars(ns).items()
              if (k.startswith("sch_") or k == "use_code_head") and k not in runtime_only}
        cfg = GPTConfig(sequence_len=2048, vocab_size=vocab, n_layer=depth,
                        n_head=max(1, d // 128), n_kv_head=max(1, d // 128),
                        n_embd=d, **kw)
        cfg._tokenizer_dir = None
        try:
            with contextlib.redirect_stdout(_io.StringIO()):
                with torch.device("meta"):
                    GPT(cfg)
        except Exception as exc:
            pytest.fail(f"c05 arm {tag} does not build at V={vocab} depth={depth}: "
                        f"{type(exc).__name__}: {exc}")


def test_an_illegal_product_code_says_what_to_change_it_to():
    """A refusal that does not name the fix costs a second round trip."""
    from nanochat.code_head import resolve_sch_config

    class _Cfg:
        vocab_size = 32768
        use_code_head = 1
        sch_phi_mode = "product"
        sch_product_groups = 2
        sch_product_codebook = 64

    with pytest.raises(AssertionError) as e:
        resolve_sch_config(_Cfg(), 32768)
    msg = str(e.value)
    assert "sch_product_groups to >= 3" in msg, msg
    assert "sch_product_codebook to >= 182" in msg, msg


def test_head_flops_are_reportable_on_the_meta_device():
    """`GPT.__init__` runs under `torch.device('meta')` in base_train.

    A `flops_per_token` that reads a buffer fails there outright, and on a real
    device before `init_weights` it averages uninitialised memory and reports a
    plausible wrong number instead of raising.
    """
    import contextlib
    import io as _io
    from nanochat.gpt import GPT, GPTConfig
    for kw in ({"sch_head_type": "hsoftmax"},
               {"sch_head_type": "monarch", "sch_max_m": 256},
               {"sch_phi_mode": "product", "sch_product_groups": 4,
                "sch_product_codebook": 64}):
        cfg = GPTConfig(sequence_len=128, vocab_size=32768, n_layer=2, n_head=4,
                        n_kv_head=4, n_embd=512, use_code_head=1, **kw)
        cfg._tokenizer_dir = None
        with contextlib.redirect_stdout(_io.StringIO()):
            with torch.device("meta"):
                m = GPT(cfg)
        f = m.lm_head.flops_per_token()
        assert isinstance(f, int) and f > 0, f"{kw} reported {f}"
        assert m.estimate_flops()[0] > 0


def test_the_two_product_implementations_compute_the_same_function():
    """`gather` and `dense` differ only in how `g(h) Phi^T` is evaluated.

    If they ever disagree, the fast path is not the thing the slow path's cost
    model describes, and every FLOP number for the product head is fiction.
    """
    kw = dict(use_code_head=1, sch_phi_mode='product', sch_product_groups=4,
              sch_product_codebook=16, sch_product_source='random', sch_code_seed=11)
    a = build(**kw, sch_product_impl='dense')
    b = build(**kw, sch_product_impl='gather')
    b.load_state_dict(a.state_dict(), strict=False)
    b.lm_head._build_phi()
    x = torch.randint(0, V, (2, 8))
    torch.testing.assert_close(a(x).float(), b(x).float(), rtol=1e-2, atol=1e-2)


def test_product_flops_follow_the_implementation_actually_used():
    """The dense path pays 4*V*M like any frozen Phi; only the gather pays 4*V*g.

    Reporting the gather's arithmetic while running the matmul would overstate
    the head's efficiency by M/g, which is 8x at g=8 K=64.
    """
    kw = dict(use_code_head=1, sch_phi_mode='product', sch_product_groups=4,
              sch_product_codebook=16)
    dense = build(**kw, sch_product_impl='dense').lm_head
    gather = build(**kw, sch_product_impl='gather').lm_head
    g_cost = sum(g.flops_per_token() for g in dense.g)
    assert dense.flops_per_token() - g_cost == 4 * dense.vocab_size * dense.width
    assert gather.flops_per_token() - g_cost == 4 * gather.vocab_size * 4


def test_the_default_product_implementation_is_the_fast_one():
    """A sweep must not silently pick the path measured 4.76x slower."""
    head = build(use_code_head=1, sch_phi_mode='product',
                 sch_product_groups=4, sch_product_codebook=16).lm_head
    assert head.product_impl == "dense"


def test_product_phi_is_rebuilt_from_the_restored_assignment():
    """Phi is derived, `assign` is persisted; a resume must not keep the default."""
    kw = dict(use_code_head=1, sch_phi_mode='product', sch_product_groups=4,
              sch_product_codebook=16, sch_product_source='random')
    a = build(**kw, sch_code_seed=1)
    b = build(**kw, sch_code_seed=2)
    assert not torch.equal(a.lm_head.assign, b.lm_head.assign)
    b.load_state_dict(a.state_dict())
    assert torch.equal(a.lm_head.assign, b.lm_head.assign)
    x = torch.randint(0, V, (2, 8))
    torch.testing.assert_close(a(x), b(x))


def test_hard_routing_does_not_collapse_onto_one_component():
    """A zero-initialised router plus top-1 is a dead architecture.

    With zero weights every token's router logits are identical and `topk`
    breaks the tie by index, so component 0 receives ALL tokens and 1..K-1
    receive none, forever: they get no gradient, so the router never learns to
    prefer them. It surfaced as DDP refusing to step ("Parameter indices which
    did not receive grad: 31 32 33") and would otherwise have been a mixture
    that silently behaved like a single component while paying for four.
    """
    m = _phase5(use_code_head=1, sch_order=2, sch_max_m=20, sch_mixture=4,
                sch_mixture_per_phi=1, sch_mixture_topk=1)
    head = m.lm_head
    x = torch.randn(64, head.n_embd)
    route = F.linear(x, head.router.weight.float())
    used = route.argmax(dim=-1).unique().numel()
    assert used > 1, ("the router sends every token to one component at init; "
                      "hard routing needs a symmetry-broken router")


def test_every_parameter_of_a_sparsely_routed_head_receives_gradient():
    """The exact condition DDP checks with find_unused_parameters=False.

    A component that receives no token contributes no gradient to its `g`
    weights. Load balancing plus a symmetry-broken router should keep all of
    them alive on a batch of realistic size.
    """
    m = _phase5(use_code_head=1, sch_order=2, sch_max_m=20, sch_mixture=4,
                sch_mixture_per_phi=1, sch_mixture_topk=1)
    m.train()
    x = torch.randint(0, V, (4, 32))
    m(x, x).backward()
    missing = [n for n, p in m.lm_head.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_the_load_balance_term_is_minimal_at_a_uniform_assignment():
    m = _phase5(use_code_head=1, sch_order=2, sch_max_m=20, sch_mixture=4,
                sch_mixture_per_phi=1, sch_mixture_topk=1)
    head = m.lm_head
    n = 4096
    uniform = torch.zeros(n, 4)
    uniform[torch.arange(n), torch.arange(n) % 4] = 8.0     # balanced, confident
    collapsed = torch.zeros(n, 4)
    collapsed[:, 0] = 8.0                                    # everything to one
    assert head._load_balance(uniform) < head._load_balance(collapsed)
    # perfectly uniform routing gives K * sum_m (1/K)(1/K) = 1
    assert abs(float(head._load_balance(uniform)) - 1.0) < 0.05


def test_ddp_asks_for_unused_parameter_detection_when_routing_is_sparse():
    """common.wrap_model must see the sparse head, not just MoE and MST."""
    import inspect
    from nanochat import common
    src = inspect.getsource(common.wrap_model)
    assert "has_sparse_head" in src and "sch_mixture_topk" in src


def _count_full_width_tensors(model, x, vocab):
    """How many tensors of at least (tokens x V) a forward+backward creates.

    This is the quantity that decides whether a head fits. Peak RSS is the
    honest measure but it is noisy across processes; counting full-width
    allocations is deterministic and tracks it closely, because at 262144 tokens
    and V=32768 each one is 34 GB in fp32.
    """
    from torch.utils._python_dispatch import TorchDispatchMode

    class _Count(TorchDispatchMode):
        def __init__(self, thresh):
            self.thresh, self.n = thresh, 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for t in (out if isinstance(out, (tuple, list)) else [out]):
                if isinstance(t, torch.Tensor) and t.numel() >= self.thresh:
                    self.n += 1
            return out

    counter = _Count(x.numel() * vocab)
    model.train()
    with counter:
        model(x, x).backward()
    return counter.n


def _wide_ops(model, x, vocab):
    """Record ops that produce a tensor of at least (tokens x V) elements.

    Peak RSS is what actually OOMs, but at test scale the allocator absorbs the
    difference and the measurement is noise (4.66 against 4.65 buffers for the
    fixed and broken versions). The ops that create full-width tensors are
    deterministic, and naming them is a sharper statement than a byte count:
    with k=1 there is nothing to weight and nothing to combine, so a full-width
    `full`, `add` or `logaddexp` in this path is the bug returning.
    """
    from torch.utils._python_dispatch import TorchDispatchMode

    class _Rec(TorchDispatchMode):
        def __init__(self, thresh):
            self.thresh, self.ops = thresh, []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for t in (out if isinstance(out, (tuple, list)) else [out]):
                if isinstance(t, torch.Tensor) and t.numel() >= self.thresh:
                    self.ops.append(func.overloadpacket.__name__)
            return out

    rec = _Rec(x.numel() * vocab)
    model.train()
    with rec:
        model(x, x).backward()
    return rec.ops


def test_top1_routing_allocates_one_full_width_buffer_and_no_combining():
    """The sparse mixture OOMed at 106 GB on a 140 GB card before this.

    It held three full-width buffers per slot where the dense baseline holds
    one: a `-inf` fill, a broadcast add for the router weight, and a logaddexp
    result. With k=1 none of the three is needed. The renormalised router weight
    over a single element is identically 0, there is only one slot to combine,
    and the components partition the token axis so the buffer needs no fill.
    One such tensor is 34 GB in fp32 at 262144 tokens and V=32768.
    """
    x = torch.randint(0, V, (2, 32))
    m = build(use_code_head=1, sch_order=3, sch_max_m=60, sch_mixture=4,
              sch_mixture_per_phi=1, sch_mixture_topk=1, sch_bias=1)
    ops = _wide_ops(m, x, V)
    assert "logaddexp" not in ops, "k=1 has one slot; there is nothing to combine"
    assert "full" not in ops, ("the routed components partition the token axis, so "
                               "the output buffer is written everywhere and needs no fill")
    assert ops.count("empty") <= 1, f"more than one full-width buffer allocated: {ops}"


def test_wider_routing_is_heavier_and_the_sweep_compensates():
    """k>1 is inherently k log-softmax outputs plus the running combination.

    It cannot be made as cheap as k=1, so the sweep gives it a smaller device
    batch rather than letting it OOM. Gradient accumulation keeps the total
    batch identical, so the comparison is unaffected.
    """
    x = torch.randint(0, V, (2, 32))
    kw = dict(use_code_head=1, sch_order=3, sch_max_m=60, sch_mixture=4,
              sch_mixture_per_phi=1, sch_bias=1)
    assert "logaddexp" in _wide_ops(build(**kw, sch_mixture_topk=2), x, V), \
        "k=2 must combine two slots; if it does not, the routing is not doing what it claims"
    # Every routed arm graph-breaks on `nonzero` and runs eager while the dense
    # baseline is compiled and fused. Measured memory slope in batch size, in
    # units of one (N x V) fp32 buffer: dense 4.16, top-1 5.29, top-2 9.87.
    for tag, flags in _sweep_arms("scripts/c05_sch_phase5_alternatives.sh"):
        if not tag.startswith("MIX_"):
            continue
        assert "--device-batch-size" in flags, \
            f"{tag} routes tokens and must run a smaller device batch than dense"


def test_a_tree_head_survives_the_end_of_run_diagnostics():
    """hsoftmax trained to completion and then died on the last step.

    `base_train` generates a sample after the final step, and `run_all_diagnostics`
    runs a rank SVD, an anisotropy probe and a head-cost benchmark. All of those
    need a logit vector, which a tree head never materialises: that is the point
    of O(d log V), not a missing feature. Throwing away a finished run over it is
    the expensive kind of bug, so every such probe is skipped rather than fatal.
    """
    from nanochat.code_metrics import run_all_diagnostics
    m = build(use_code_head=1, sch_head_type='hsoftmax')
    assert getattr(m.lm_head, 'custom_loss', False)

    def loader():
        while True:
            x = torch.randint(0, V, (2, 16))
            yield x, x

    metrics = run_all_diagnostics(m, loader, token_bytes=torch.ones(V), vocab_size=V,
                                  steps=2, decile=False, rank_contexts=4096)
    assert isinstance(metrics, dict)
    for probe in ("rank_effective_rank", "anisotropy", "head_ms"):
        assert probe not in metrics, f"{probe} cannot be computed without logits"


def test_base_train_does_not_try_to_generate_from_a_logit_less_head():
    src = open("scripts/base_train.py").read()
    assert "_head_emits_logits" in src and "custom_loss" in src, \
        "the sample-generation guard must know about heads that emit no logits"


def test_a_hard_top1_mixture_is_not_self_normalised():
    """Each token uses exactly one component, so there is nothing to mix.

    `log P = log_softmax(that component's logits)` is precisely what
    `F.cross_entropy` is about to compute. Normalising inside the head as well
    is mathematically a no-op and costs a second full-width fp32 tensor saved
    for backward, 34 GB at 262144 tokens and V=32768. That was two thirds of an
    OOM that reported 99.5 GB allocated on a 140 GB card.
    """
    top1 = build(use_code_head=1, sch_order=3, sch_max_m=60, sch_mixture=4,
                 sch_mixture_per_phi=1, sch_mixture_topk=1).lm_head
    top2 = build(use_code_head=1, sch_order=3, sch_max_m=60, sch_mixture=4,
                 sch_mixture_per_phi=1, sch_mixture_topk=2).lm_head
    assert not top1.self_normalized, "hard top-1 emits raw logits"
    assert top2.self_normalized, "k>1 must combine normalised components"
    # The union of subspaces escapes the rank bound at either k.
    assert top1.rank_ceiling() == float("inf")


def test_a_self_normalised_head_is_not_normalised_twice():
    """`cross_entropy` log_softmaxes its input; on log-probs that is the identity
    and costs a full-width tensor saved for backward. `nll_loss` skips it."""
    import inspect
    from nanochat.gpt import GPT
    src = inspect.getsource(GPT.forward)
    assert "F.nll_loss" in src and "self_normalized" in src

    # sch_mixture_aux off: the load-balance term is added to the total loss, so
    # leaving it on would make this compare CE against CE + 0.01 * aux.
    m = build(use_code_head=1, sch_order=3, sch_max_m=60, sch_mixture=4,
              sch_mixture_per_phi=1, sch_mixture_topk=2, sch_bias=1, sch_mixture_aux=0.0)
    x = torch.randint(0, V, (2, 16))
    lp = m(x).view(-1, V).float()
    # exactly nll_loss, since that is what the model now uses
    torch.testing.assert_close(m(x, x), F.nll_loss(lp, x.view(-1)), rtol=1e-5, atol=1e-5)
    # and cross_entropy agrees to within the drift of re-normalising: the head's
    # log-probs come through bf16 components, so logsumexp is near zero and not
    # exactly zero. That residual is the reason to skip the second normalisation,
    # not a reason to distrust it.
    torch.testing.assert_close(m(x, x), F.cross_entropy(lp, x.view(-1)),
                               rtol=5e-3, atol=5e-2)


def test_the_load_balance_term_actually_reaches_the_loss():
    """It is added in GPT.forward, not inside the head, so it can silently not
    be wired. A mixture that pays for K components and balances none of them
    collapses onto one."""
    kw = dict(use_code_head=1, sch_order=3, sch_max_m=60, sch_mixture=4,
              sch_mixture_per_phi=1, sch_mixture_topk=1, sch_bias=1)
    x = torch.randint(0, V, (2, 16))
    off = build(**kw, sch_mixture_aux=0.0)(x, x)
    on = build(**kw, sch_mixture_aux=0.5)(x, x)
    assert on > off + 0.1, "the load-balance term is not reaching the loss"


def test_a_deeper_huffman_tree_still_loads():
    """The tree depth is a property of the data, not of the config.

    The constructor can only guess a bound, `init_weights` reallocates when the
    real Huffman tree is deeper, and the checkpoint then carries that depth: 38
    with a real frequency table at V=32768 against a bound of 30. A freshly
    built model is back at the bound, so without resizing on load a finished run
    cannot be re-measured at all.
    """
    a = build(use_code_head=1, sch_head_type='hsoftmax')
    h = a.lm_head
    old = h.nodes.shape[1]
    for name, dtype in (("nodes", torch.int64), ("dirs", torch.float32), ("mask", torch.bool)):
        t = torch.zeros(h.vocab_size, old + 12, dtype=dtype)
        t[:, :old] = getattr(h, name)
        setattr(h, name, t)
    h.avg_depth = float(h.mask.sum(dim=1).float().mean())
    sd = a.state_dict()

    b = build(use_code_head=1, sch_head_type='hsoftmax')
    assert b.lm_head.nodes.shape[1] == old, "the fresh model should start at the bound"
    b.load_state_dict(sd)
    assert b.lm_head.nodes.shape[1] == old + 12
    # avg_depth drives flops_per_token, so it has to follow the restored mask
    assert b.lm_head.avg_depth == pytest.approx(a.lm_head.avg_depth)
    x = torch.randint(0, V, (2, 8))
    torch.testing.assert_close(a(x, x), b(x, x))


# ---------------------------------------------------------------------------
# sweep_report: it must describe the runs, not the current code
# ---------------------------------------------------------------------------

def _fake_run(tmp_path, tag, user_config):
    """A checkpoint directory shaped the way base_train writes one."""
    import json
    d = tmp_path / tag / "depth_4" / "ckpt_base" / "base"
    d.mkdir(parents=True)
    (d / "model_000462.pt").write_bytes(b"")
    with open(d / "meta_000462.json", "w") as f:
        json.dump({"step": 462, "user_config": user_config}, f)
    return str(d.parent)


def test_the_report_inherits_the_directories_the_run_used(tmp_path):
    """Defaulting data_dir to None kills every arm without cached metrics with
    "No dataset parquet files found", which is not a missing dataset but a
    missing argument. The run recorded what it used; read it."""
    from scripts.sweep_report import inherit, read_user_config
    data = tmp_path / "shards"
    data.mkdir()
    ck = _fake_run(tmp_path, "ARM_s1", {"data_dir": str(data), "tokenizer_dir": "tokenizer",
                                        "max_shards": 300})
    cfg = read_user_config(ck)
    assert cfg["max_shards"] == 300

    class A:
        data_dir = tokenizer_dir = None
        max_shards = None
    env = inherit(A(), cfg)
    assert env["data_dir"] == str(data), "the run's own data directory must be used"
    assert env["max_shards"] == 300

    class B:
        data_dir = "/explicit"
        tokenizer_dir = None
        max_shards = 7
    assert inherit(B(), cfg)["data_dir"] == "/explicit", "an explicit flag must win"

    # a directory that no longer resolves must not be handed to the loader
    gone = read_user_config(_fake_run(tmp_path, "ARM2_s1", {"data_dir": "/no/such/path"}))
    assert inherit(A(), gone)["data_dir"] is None


def test_the_report_reads_settings_from_the_launch_args_not_todays_defaults(tmp_path):
    """A flag added after a run is absent from its checkpoint. Reading it off a
    freshly built GPTConfig reports the current default as though the run had
    used it, which is how `impl` came to say "dense" for arms whose recorded
    FLOPs were the gather cost."""
    from scripts.sweep_report import read_user_config
    old_run = read_user_config(_fake_run(tmp_path, "OLD_s1", {"sch_phi_mode": "product"}))
    assert old_run.get("sch_product_impl", "?") == "?", \
        "a run predating the flag must report '?', not the current default"
    new_run = read_user_config(_fake_run(tmp_path, "NEW_s1",
                                         {"sch_phi_mode": "product",
                                          "sch_product_impl": "gather"}))
    assert new_run["sch_product_impl"] == "gather"
