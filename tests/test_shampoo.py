"""Block-diagonal Shampoo (Stage 17).

The architecture claim expressed as an optimizer: preconditioning a dense D x D weight is
O(D^3), but MST's stacked per-stream weights are N blocks of d x d, so exact block
preconditioning costs N*(D/N)^3 = D^3/N^2, i.e. N^2 = 16x cheaper at N=4. K-FAC and Shampoo
both *approximate* block-diagonality; MST makes it exact by construction, so at equal
optimizer cost MST affords a stronger preconditioner than dense can.

These tests pin the properties that claim depends on, plus the three numerical hazards that
actually bit during development (see _inverse_fourth_root's docstring).

Run:  pytest tests/test_shampoo.py -q
"""
import pytest
import torch

from nanochat.optim import MuonAdamW

N, OUT_B, IN_D, P = 4, 8, 6, 2          # N blocks of (OUT_B x IN_D), P params per group


def make(kind, dtype=torch.float32, **extra):
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(N * OUT_B, IN_D, dtype=dtype) * 0.1)
              for _ in range(P)]
    group = dict(kind=kind, params=params, lr=0.02, momentum=0.95, ns_steps=5,
                 beta2=0.95, weight_decay=0.0, block_diagonal=N)
    group.update(extra)
    return params, MuonAdamW([group])


def _problem():
    """Ill-conditioned least squares: the regime preconditioning is supposed to help."""
    torch.manual_seed(1)
    A = torch.randn(128, N * OUT_B) @ torch.diag(torch.logspace(0, 1, N * OUT_B))
    target = torch.randn(128, IN_D) * 0.1
    return lambda ps: sum(((A @ p - target) ** 2).mean() for p in ps)


@pytest.mark.parametrize("every", [1, 10])
def test_shampoo_optimizes(every):
    loss_of = _problem()
    params, opt = make("shampoo", precond_every=every)
    start = loss_of(params).item()
    for _ in range(100):
        opt.zero_grad()
        loss_of(params).backward()
        opt.step()
    end = loss_of(params).item()
    assert torch.isfinite(params[0]).all(), "shampoo diverged to non-finite params"
    assert end < start / 10, f"loss barely moved: {start:.4f} -> {end:.4f}"


def test_shampoo_is_competitive_with_muon():
    """Not a benchmark, just a guard that the preconditioner is not actively harmful."""
    loss_of = _problem()
    finals = {}
    for kind, extra in (("muon", {}), ("shampoo", {"precond_every": 1})):
        params, opt = make(kind, **extra)
        for _ in range(100):
            opt.zero_grad()
            loss_of(params).backward()
            opt.step()
        finals[kind] = loss_of(params).item()
    assert finals["shampoo"] < finals["muon"] * 1.5, finals


def test_preconditioner_is_block_shaped_not_dense():
    """The N^2 saving is exactly this: L is (Kb, d, d), never (K, N*d, N*d)."""
    loss_of = _problem()
    params, opt = make("shampoo", precond_every=1)
    opt.zero_grad()
    loss_of(params).backward()
    opt.step()

    state = opt.state[params[0]]
    assert state["L"].shape == (P * N, OUT_B, OUT_B)
    assert state["R"].shape == (P * N, IN_D, IN_D)

    block_cost = P * N * OUT_B ** 3
    dense_cost = P * (N * OUT_B) ** 3
    assert dense_cost / block_cost == pytest.approx(N ** 2), \
        "the eigh cost ratio is the whole claim and it must be N^2"


def test_blocks_are_independent():
    """A gradient in one stream must not enter another stream's preconditioner."""
    params, opt = make("shampoo", precond_every=1)
    for p in params:
        p.grad = torch.zeros_like(p)
    params[0].grad[:OUT_B] = torch.randn(OUT_B, IN_D)   # only block 0 of param 0
    opt.step()

    L = opt.state[params[0]]["L"]
    assert L[0].abs().sum() > 0, "the touched block should have accumulated statistics"
    assert L[1].abs().sum() == 0, "an untouched block picked up statistics from another"


def test_state_survives_a_bf16_round_trip():
    """torch.optim casts state to the param dtype on load, which would ruin fp32 L/R.

    MuonAdamW.load_state_dict restores them. Without that, resuming a bf16 run silently
    downcasts the preconditioner statistics and the inverse fourth root is garbage.
    """
    params, opt = make("shampoo", dtype=torch.bfloat16, precond_every=1)
    params[0].grad = torch.randn_like(params[0])
    opt.step()
    assert opt.state[params[0]]["L"].dtype == torch.float32

    _, opt2 = make("shampoo", dtype=torch.bfloat16, precond_every=1)
    opt2.load_state_dict(opt.state_dict())
    for key in ("L", "R", "QL", "QR"):
        assert opt2.state[list(opt2.state)[0]][key].dtype == torch.float32, \
            f"{key} was downcast on resume"


def test_survives_a_degenerate_first_step():
    """After one step L is rank 1. An absolute ridge left it singular and eigh raised."""
    params, opt = make("shampoo", precond_every=1)
    params[0].grad = torch.zeros_like(params[0])
    params[1].grad = torch.zeros_like(params[1])
    params[0].grad[0, 0] = 1.0            # rank-1, almost entirely zero
    opt.step()                            # must not raise
    assert torch.isfinite(params[0]).all()
    assert torch.isfinite(opt.state[params[0]]["QL"]).all()


@pytest.mark.parametrize("grad_scale", [1e-4, 1.0, 1e3])
def test_update_norm_matches_muons_convention_at_any_gradient_scale(grad_scale):
    """The bug that made the first shampoo sweep useless.

    Polar Express emits an approximately semi-orthogonal matrix, so Muon's update norm is
    ~sqrt(min(m,n)) no matter the gradient scale, and the LR is tuned for that. Shampoo
    originally normalized to ||g||_F, which is not scale-free: on a real MST layer
    ||g||_F was 2.6e-4 against Muon's 8.07, so the effective LR was ~3e4 too small. It
    surfaced as a uniform +0.07 bpb with near-identical results across a 50x range of
    refresh cadences, which is the signature of a step size, not a preconditioner.

    The toy convergence tests above did NOT catch it, because there ||g|| happened to sit
    near sqrt(min(m,n)). Hence this test pins the ratio directly.
    """
    params, opt = make("shampoo", precond_every=1)
    before = [p.detach().clone() for p in params]
    for p in params:
        p.grad = torch.randn_like(p) * grad_scale
    opt.step()

    lr = opt.param_groups[0]["lr"]
    # _step_shampoo applies Muon's spectral LR correction max(1, out/in)**0.5 on top,
    # so the per-block update norm is sqrt(min(m,n)) * that factor. The point of the test
    # is that this is a constant, independent of how big the gradient was.
    expected = float(min(OUT_B, IN_D)) ** 0.5 * max(1.0, OUT_B / IN_D) ** 0.5
    for p, b in zip(params, before):
        delta = (b - p.detach()).reshape(N, OUT_B, IN_D)
        norms = delta.norm(dim=(-2, -1)) / lr
        assert torch.allclose(norms, torch.full_like(norms, expected), rtol=0.05), \
            f"update norm {norms.mean():.3f} should be {expected:.3f} independent of the " \
            f"gradient scale, but the gradient was scaled by {grad_scale}"


def test_unknown_kind_still_raises():
    """The dispatch has nine hardcoded sites; a typo must fail loudly, not silently."""
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = MuonAdamW([dict(kind='shampoo_typo', params=[p], lr=0.01, momentum=0.95,
                          ns_steps=5, beta2=0.95, weight_decay=0.0)])
    p.grad = torch.randn_like(p)
    with pytest.raises(ValueError, match="Unknown optimizer kind"):
        opt.step()
