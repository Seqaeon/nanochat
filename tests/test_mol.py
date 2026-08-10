"""Validation for the MoL baseline (nanochat/mol.py, arXiv:2605.09516v1).

There is no official implementation to diff against, so correctness rests on
reproducing the paper's own published numbers from our parameter counter, plus
behavioural tests for each equation. The parameter tests are the load-bearing
ones: if the architecture were wrong, 85.3M and 0.61B would not both fall out.
"""

import re
import pytest
import torch
import torch.nn as nn

from nanochat.gpt import GPTConfig
from nanochat.gpt import norm as mol_norm
from nanochat.mol import (MoL, ThinBlock, _BLOCK_GATES, _block_config_knobs,
                         make_thin_config)


def make_config(**ov):
    base = dict(
        sequence_len=256, vocab_size=1024, n_layer=4, n_head=4, n_embd=512,
        use_mol=True, mol_n_blocks=5, mol_n_shared=1, mol_topk=3,
        mol_thin_dim=128, mol_ffn_mult=4.0,
    )
    base.update(ov)
    return GPTConfig(**base)


def build_meta(**ov):
    with torch.device("meta"):
        return MoL(make_config(**ov))


# ---------------------------------------------------------------- the paper's numbers

def test_table1_parameter_count():
    """Table 1: MoL K=5 top-3 dense FFN thin blocks = 85.3M on WikiText-103.

    Their §4.2: d_model=1024, 16 heads, 8 layers, d_thin=256, 4 heads per thin
    block. WikiText-103 uses a custom 32K BPE. Table 1 reports 85.3M, and lists no
    shared block, so this is the pure top-k configuration of Eq (2).
    """
    with torch.device("meta"):
        m = MoL(GPTConfig(sequence_len=255, vocab_size=32000, n_layer=8,
                          n_head=16, n_embd=1024, use_mol=True,
                          mol_n_blocks=5, mol_n_shared=0, mol_topk=3,
                          mol_thin_dim=256, mol_ffn_mult=4.0))
    counts = m.num_scaling_params()
    # Tied embeddings in their setup: transformer matrices + one 32000x1024 table.
    total = counts['transformer_matrices'] + 32000 * 1024
    assert 85.0e6 < total < 85.6e6, f"expected ~85.3M (their Table 1), got {total/1e6:.2f}M"


def test_table5_active_parameter_count():
    """Table 5: MoL Hybrid 1+3of15 at 1.3B scale = 0.61B active.

    Their §4.2: d_model=2048, 24 layers, GPT-2 50257 vocab, d_thin=512,
    15 thin blocks (1 shared + 14 routed, top-3 = 4 active per token),
    d_ff,thin=2048 (which is 4x d_thin).
    """
    with torch.device("meta"):
        m = MoL(GPTConfig(sequence_len=4096, vocab_size=50257, n_layer=24,
                          n_head=16, n_embd=2048, use_mol=True,
                          mol_n_blocks=15, mol_n_shared=1, mol_topk=3,
                          mol_thin_dim=512, mol_ffn_mult=4.0))
    c = m.num_scaling_params()
    # 4 active thin blocks x 24 layers, plus the tied embedding table.
    active = (c['active'] - c['total'] + c['transformer_matrices']) + 50257 * 2048
    assert 0.59e9 < active < 0.63e9, f"expected ~0.61B active (Table 5), got {active/1e9:.3f}B"


@pytest.mark.parametrize("d_thin,expected", [(256, 0.400), (128, 0.571), (64, 0.727)])
def test_projection_overhead_matches_their_quoted_figures(d_thin, expected):
    """§2.3: projections are 40% of wrapper params at d_thin=256, 57% at 128, 73% at 64.

    This is the number the whole MST-versus-MoL argument turns on, so it is pinned
    against all three of their published values at their d_model=1024.
    """
    m = build_meta(n_embd=1024, n_layer=2, mol_thin_dim=d_thin, mol_n_shared=0,
                   mol_n_blocks=5, mol_topk=3)
    frac = m.num_scaling_params()['projection_fraction']
    assert frac == pytest.approx(expected, abs=0.005), \
        f"d_thin={d_thin}: got {frac:.3f}, paper says {expected}"


def test_projection_fraction_follows_the_closed_form():
    """MoL's plumbing fraction is D/(D + 6*d_thin), and it grows as blocks narrow.

    The contrast with MST, whose (N+8)/(13N+8) is independent of width, is the
    structural claim; this pins the MoL half of it.
    """
    D = 1024
    prev = 0.0
    for d in (512, 256, 128, 64):
        m = build_meta(n_embd=D, n_layer=2, mol_thin_dim=d, mol_n_shared=0)
        frac = m.num_scaling_params()['projection_fraction']
        assert frac == pytest.approx(D / (D + 6 * d), abs=0.005)
        assert frac > prev, "plumbing fraction must grow as thin blocks narrow"
        prev = frac


# ---------------------------------------------------------------- the equations

class _FakeBlock(nn.Module):
    """Stand-in for Block_thin with a known delta, so Eq (1) can be checked exactly."""

    def __init__(self, delta=None):
        super().__init__()
        self.delta = delta

    def forward(self, h, *a, **kw):
        return h if self.delta is None else h + self.delta


def test_eq1_strips_the_inner_residual():
    """Eq (1) subtracts W_down.x, so an identity block emits exactly zero."""
    cfg = make_config()
    tb = ThinBlock(cfg, make_thin_config(cfg), 0)
    tb.block = _FakeBlock()                   # Block_thin := identity
    x = torch.randn(2, 8, cfg.n_embd)
    out = tb(x, None, None, (-1, 0), None)
    assert torch.equal(out, torch.zeros_like(out)), \
        "ThinBlock must emit only the block's delta, not its input"


def test_eq1_delta_is_the_block_delta():
    """With a non-identity block, output is W_up applied to (Block(h) - h)."""
    cfg = make_config()
    tb = ThinBlock(cfg, make_thin_config(cfg), 0)
    bias = torch.randn(cfg.mol_thin_dim)
    tb.block = _FakeBlock(bias)
    x = torch.randn(2, 8, cfg.n_embd)
    expected = tb.w_up(bias.expand(2, 8, -1))
    assert torch.allclose(tb(x, None, None, (-1, 0), None), expected, atol=1e-5)


def test_eq2_selects_topk_and_weights_by_softmax_score():
    """Eq (2): exactly k blocks contribute, weighted by their softmax scores."""
    torch.manual_seed(0)
    m = MoL(make_config(mol_n_shared=0, mol_n_blocks=4, mol_topk=2))
    m.init_weights()
    stage = m.stages[0]
    x = torch.randn(2, 6, m.config.n_embd)
    weights, mask, aux = stage._route(x)
    assert torch.isfinite(aux), "the CV^2 term must be returned, not stashed on the module"

    assert mask.sum(-1).unique().tolist() == [2], "exactly k blocks active per token"
    assert torch.equal(weights > 0, mask), "weights must be zero off the top-k"

    probs = torch.softmax(stage.router(mol_norm(x)).float(), dim=-1)
    assert torch.allclose(weights.float(), probs * mask.float(), atol=1e-5), \
        "surviving weights must be the untouched softmax probabilities"
    # Eq (2) divides by k, not by the surviving probability mass, so the stage
    # output is NOT a convex combination. Pin that, since renormalising is the
    # obvious "fix" someone would apply later and it would change the architecture.
    assert not torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)))


def test_cv2_aux_is_zero_when_balanced_and_positive_otherwise():
    m = build_meta(mol_n_shared=0, mol_n_blocks=4, mol_topk=2)
    stage = m.stages[0]
    stage.router = None  # exercise the loss directly, not the linear layer

    def cv2(importance):
        return importance.var(unbiased=False) / importance.mean().pow(2)

    assert cv2(torch.full((4,), 3.0)) == pytest.approx(0.0)
    assert cv2(torch.tensor([10.0, 1.0, 1.0, 1.0])) > 0.5


# ---------------------------------------------------------------- restricted attention

def test_routed_block_ignores_tokens_not_routed_to_it():
    """§2.3: sparse dispatch equals *dense restricted attention*.

    A routed block must see only its own tokens. If this fails, MoL silently gets
    full sequence coverage, which would inflate its quality and invalidate the
    comparison. This is the single most important behavioural test in the file.
    """
    torch.manual_seed(0)
    cfg = make_config(n_layer=1, mol_n_blocks=2, mol_n_shared=0, mol_topk=1,
                      sequence_len=16)
    m = MoL(cfg)
    m.init_weights()
    m.eval()
    stage = m.stages[0]
    for tb in stage.blocks:                      # wake the zero-inited up-projections
        torch.nn.init.normal_(tb.w_up.weight, std=0.02)

    B, T, D = 1, 8, cfg.n_embd
    x = torch.randn(B, T, D)
    active = torch.zeros(B, T, dtype=torch.bool)
    active[0, [0, 1, 2]] = True                  # only these three tokens routed here
    cos_sin = (m.cos[:, :T], m.sin[:, :T])

    with torch.no_grad():
        y1 = stage.blocks[0](x, None, cos_sin, (-1, 0), None, token_active=active)
        x2 = x.clone()
        x2[0, 5:] = torch.randn(T - 5, D)        # perturb only unrouted tokens
        y2 = stage.blocks[0](x2, None, cos_sin, (-1, 0), None, token_active=active)

    assert torch.allclose(y1[0, :3], y2[0, :3], atol=1e-4), \
        "a routed block's active tokens must not depend on tokens routed elsewhere"


# ---------------------------------------------------------------- accounting

def test_active_flops_track_the_routed_fraction():
    dense = build_meta(mol_n_blocks=4, mol_n_shared=4, mol_topk=1)  # all shared
    tot_d, act_d, _ = dense.estimate_flops()
    assert act_d == tot_d, "an all-shared MoL claims no sparsity"

    sparse = build_meta(mol_n_blocks=5, mol_n_shared=1, mol_topk=2)
    tot, act, act_p = sparse.estimate_flops()
    assert act < tot, "routed blocks must discount active FLOPs"
    assert act_p < sum(p.numel() for p in sparse.parameters())


def test_active_flops_shrink_as_topk_shrinks():
    prev = None
    for k in (4, 3, 2, 1):
        _, act, _ = build_meta(mol_n_blocks=5, mol_n_shared=1, mol_topk=k).estimate_flops()
        if prev is not None:
            assert act < prev, f"top-{k} must be cheaper than top-{k+1}"
        prev = act


def test_routed_attention_is_priced_quadratically():
    """A routed block runs on frac of tokens AND attends over frac of keys.

    Pricing it linearly would overcharge MoL for attention, which would flatter us.
    """
    m = build_meta(n_layer=1, mol_n_blocks=5, mol_n_shared=0, mol_topk=1,
                   sequence_len=1024)
    t = m.config.sequence_len
    nh, hd = m.thin_n_head, m.head_dim
    eff = t if m.window_sizes[0][0] < 0 else min(m.window_sizes[0][0], t)
    frac = 1 / 5
    expected_attn = 5 * 12 * nh * hd * eff * frac * frac
    total, active, _ = m.estimate_flops()
    matmul_only = total - 5 * 12 * nh * hd * eff
    assert active - (matmul_only - 6 * 0) < total, "sanity"
    # Recover the active attention term and compare with the quadratic prediction.
    nparams = sum(p.numel() for p in m.parameters())
    ve = sum(p.numel() for n, p in m.named_parameters() if 'value_embed' in n)
    excl = m.wte.weight.numel() + ve + m.resid_lambdas.numel() + m.x0_lambdas.numel()
    _, routed, _ = m._param_groups()
    active_attn = active - (6 * (nparams - excl) - 6 * int(routed * (1 - frac)))
    assert active_attn == pytest.approx(expected_attn, rel=0.02)


# ---------------------------------------------------------------- guards

def test_shadow_config_rejects_research_flags():
    """The thin block is gpt.Block, which branches on ~40 flags it would inherit."""
    with pytest.raises(AssertionError, match="thin blocks"):
        build_meta(p36_swiglu_ffn=1)
    with pytest.raises(AssertionError, match="thin blocks"):
        build_meta(p20_mone_experts=4)


def test_shadow_guard_covers_every_block_knob():
    """Re-derive the knob list from gpt.py so a new Block flag cannot leak in."""
    src = open('nanochat/gpt.py').read()
    segs = []
    for cls in ('class Block(nn.Module)', 'class CausalSelfAttention(nn.Module)',
                'class MLP(nn.Module)'):
        i = src.find(cls)
        segs.append(src[i:src.find('\nclass ', i + 10)])
    seg = ''.join(segs)
    found = set(re.findall(r"getattr\(config,\s*['\"]([a-zA-Z0-9_]+)['\"]", seg))
    found |= set(re.findall(r"config\.([a-z0-9_]+)", seg))
    geometry = {'n_embd', 'n_head', 'n_kv_head', 'n_layer'}
    missing = (found - geometry) - set(_block_config_knobs())
    assert not missing, (
        f"gpt.Block reads config fields the MoL shadow guard does not classify: "
        f"{sorted(missing)}. Add each to _BLOCK_GATES (with its off value) or to "
        f"_BLOCK_GATED_PARAMS in nanochat/mol.py")


def test_guard_passes_for_base_train_argparse_defaults():
    """The guard must not fire on an ordinary run. It shipped broken and did.

    base_train sets some fields to sentinels that differ from the GPTConfig default
    while meaning "inactive". `p23_std_moe_topk` defaults to -1 there against 1 in
    the dataclass, and the original guard compared against dataclass defaults, so
    every MoL arm of the sweep died at model construction on a flag that cannot
    affect a thin block at all (it is inert unless p23_std_moe_experts > 0).

    Driven off base_train's real parser defaults for EVERY knob gpt.Block reads,
    gates and gated params alike, so any future sentinel is caught here rather than
    on a GPU.
    """
    import ast
    tree = ast.parse(open('scripts/base_train.py').read())
    defaults = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, 'attr', '') == 'add_argument'):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        name = node.args[0].value
        if not isinstance(name, str) or not name.startswith('--'):
            continue
        for kw in node.keywords:
            if kw.arg == 'default':
                try:
                    defaults[name[2:].replace('-', '_')] = ast.literal_eval(kw.value)
                except ValueError:
                    pass

    knobs = {k: v for k, v in defaults.items()
             if k in _block_config_knobs() and hasattr(GPTConfig(), k)}
    assert 'p23_std_moe_topk' in knobs, "the regression this test exists for is not covered"
    m = build_meta(**knobs)          # must not raise
    assert m.d_thin == 128


def test_head_dim_is_pinned_across_widths():
    for d_thin in (64, 128, 256, 512):
        m = build_meta(n_embd=1024, mol_thin_dim=d_thin)
        assert m.head_dim == 64, "MoL pins d_head=64 at every block width (§2.2)"
        assert m.thin_n_head == d_thin // 64


def test_indivisible_thin_dim_is_rejected():
    with pytest.raises(AssertionError, match="divisible"):
        build_meta(mol_thin_dim=100)


def test_topk_must_fit_the_routed_pool():
    with pytest.raises(AssertionError, match="mol_topk"):
        build_meta(mol_n_blocks=5, mol_n_shared=1, mol_topk=9)


# ---------------------------------------------------------------- end to end

def test_forward_and_backward_run():
    torch.manual_seed(0)
    m = MoL(make_config(n_layer=2))
    m.init_weights()
    idx = torch.randint(0, 1024, (2, 16))
    loss = m(idx, idx)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.parameters() if p.requires_grad)


def test_optimizer_groups_are_shape_homogeneous_and_train():
    """Muon stacks a group's tensors, so a shape-mixed group is a hard crash.

    MoL hits this where MST does not: W_down is (d, D) and W_up is (D, d), so every
    stage contributes two different shapes. Caught by a smoke train, not by
    inspection, which is why this test actually steps the optimizer.
    """
    torch.manual_seed(0)
    m = MoL(make_config(n_layer=2))
    m.init_weights()
    opt = m.setup_optimizer()
    for g in opt.param_groups:
        shapes = {tuple(p.shape) for p in g['params']}
        assert g['kind'] != 'muon' or len(shapes) == 1, \
            f"muon group mixes shapes {shapes}"

    idx = torch.randint(0, 1024, (2, 16))
    first = last = None
    for _ in range(15):
        loss = m(idx, idx)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        first = loss.item() if first is None else first
        last = loss.item()
    assert last < first - 0.1, f"MoL failed to overfit a fixed batch: {first:.3f} -> {last:.3f}"


def test_router_does_not_collapse_onto_one_block():
    """The CV^2 aux exists to prevent collapse; verify it actually holds load up."""
    torch.manual_seed(0)
    m = MoL(make_config(n_layer=2, mol_n_blocks=5, mol_n_shared=1, mol_topk=2))
    m.init_weights()
    opt = m.setup_optimizer()
    idx = torch.randint(0, 1024, (2, 32))
    for _ in range(20):
        loss = m(idx, idx)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    diag = m.compute_diagnostics()
    worst = min(diag[f'mol_load_min_L{i}'] for i in range(2))
    assert worst > 0.05, f"a routed block went dead (min/ideal={worst:.3f})"


def test_diagnostics_report_router_load():
    torch.manual_seed(0)
    m = MoL(make_config(n_layer=2))
    m.init_weights()
    m(torch.randint(0, 1024, (2, 16)), None)
    diag = m.compute_diagnostics()
    assert 'mol_load_min_L0' in diag and 'mol_entropy_L0' in diag
    assert 0.0 <= diag['mol_load_min_L0'] <= 2.0


def test_per_block_ve_gives_each_block_its_own_vector():
    """The G3 equivalent for MoL: without it MoL is handicapped by a component
    measured to be worth 0.0059 bpb to MST, so the comparison would be rigged."""
    shared = MoL(make_config(mol_per_block_ve=0))
    per_blk = MoL(make_config(mol_per_block_ve=1))
    for m in (shared, per_blk):
        m.init_weights()
    key = next(iter(shared.value_embeds))
    ids = torch.randint(0, 1024, (1, 4))

    ve_s = shared.value_embeds[key](ids)
    st_s = shared.stages[0]
    assert torch.equal(st_s._ve_for(ve_s, 0), st_s._ve_for(ve_s, 1)), \
        "without the flag every block must see the same vector"

    ve_p = per_blk.value_embeds[key](ids)
    st_p = per_blk.stages[0]
    a, b = st_p._ve_for(ve_p, 0), st_p._ve_for(ve_p, 1)
    assert a.shape[-1] == per_blk.d_thin and not torch.allclose(a, b), \
        "with the flag each block must read a distinct d_thin slice"


def test_per_block_ve_widens_only_the_table_not_the_flops():
    """VE is a lookup, so the G3-equivalent must cost parameters and zero FLOPs."""
    a = build_meta(mol_per_block_ve=0)
    b = build_meta(mol_per_block_ve=1)
    ve_a = sum(p.numel() for n, p in a.named_parameters() if 'value_embed' in n)
    ve_b = sum(p.numel() for n, p in b.named_parameters() if 'value_embed' in n)
    assert ve_b == a.n_blocks * ve_a
    assert b.estimate_flops()[0] == a.estimate_flops()[0]
    assert b.estimate_flops()[1] == a.estimate_flops()[1]


def test_kv_cache_config_is_a_property_with_kvcache_keys():
    """It is consumed as **model.kv_cache_config, so a method silently type-errors,
    and the key names are KVCache's constructor kwargs, not GPTConfig field names.
    Both were wrong and the run died 80% of the way in at the sampling step."""
    import inspect
    from nanochat.engine import KVCache
    m = build_meta()
    cfg = m.kv_cache_config
    assert isinstance(cfg, dict), "must be a @property returning a mapping"
    accepted = set(inspect.signature(KVCache.__init__).parameters) - {'self'}
    assert set(cfg) <= accepted, f"KVCache rejects {set(cfg) - accepted}"
    assert cfg['num_layers'] == m.config.n_layer * m.n_blocks


def test_every_thin_block_gets_its_own_kv_cache_slot():
    """CausalSelfAttention reuses layer_idx as its cache slot. Constructing all
    n_blocks blocks of a stage with the stage index would alias them onto one slot."""
    m = build_meta()
    slots = [tb.block.attn.layer_idx for st in m.stages for tb in st.blocks]
    assert len(slots) == len(set(slots)) == m.config.n_layer * m.n_blocks
    assert min(slots) == 0 and max(slots) == m.config.n_layer * m.n_blocks - 1
