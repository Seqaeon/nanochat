#!/usr/bin/env python3
"""How much conditioning capacity exists at each layer, before training anything.

Every design in this repo asks how to make W depend on x. This asks the prior
question: is there anything at this layer for conditioning to exploit?

There is an exact answer. The per-token weight gradient of a linear layer is
rank one, g_t = a_t x_t^T with a_t = dL/dy_t. The mean gbar over tokens is what a
dense layer learns. The DISPERSION of g_t around gbar is exactly the capacity
available to any input-conditioned scheme, of any design: if tokens do not
disagree about which way the weight should move, no conditional layer can beat
dense at that layer, and no amount of K or R changes that.

The dispersion lives in d^2 dimensions, but it never has to be formed. The N x N
Gram matrix of the per-token gradients factors:

    <g_s, g_t>_F = tr(x_s a_s^T a_t x_t^T) = (a_s . a_t)(x_s . x_t)

so  G = (A A^T) * (X X^T),  a Hadamard product of two N x N inner-product
matrices. (Standard NTK algebra; what is new here is using it as a per-layer
conditioning-capacity diagnostic rather than as a kernel.) Center it and take
the eigenvalues. Three numbers come out per layer:

    headroom     lambda_disp / (lambda_disp + N*|gbar|^2)
                 the fraction of gradient signal a single static operator
                 cannot capture. 0 means every token wants the same update and
                 conditioning is provably useless here. Near 1 means the tokens
                 disagree almost entirely.

    dof          participation ratio (sum L)^2 / sum L^2 of the dispersion
                 spectrum. The number of independent directions the tokens
                 disagree along, i.e. the largest K or R that can pay for
                 itself at this layer. Compare against K-1 for a template bank
                 and R for ConditionedLinear.

    top1         largest dispersion eigenvalue over the total. If this is near
                 1 the disagreement is one-dimensional, which is the regime
                 where a rank-1 conditioned delta is enough and a bank is waste.

The estimator is Wishart-biased: from N samples of a rank-D object it reports
roughly D/(1 + D/N) rather than D. --tokens sets N; the printed dof is a lower
bound and the bias shrinks as N grows.

Usage:
    python -m scripts.conditioning_headroom --checkpoint out/sweep_p33/A1_.../
    python -m scripts.conditioning_headroom --depth 4 --steps 0   # random init
    python -m scripts.conditioning_headroom --checkpoint DIR --tokens 8192 --json out/headroom.json

    # is the reach% ceiling the router's linearity or its capacity?
    python -m scripts.conditioning_headroom --checkpoint DIR \
        --tokens 49152 --gram-tokens 4096 \
        --nonlinear-reach --reach-rank 64 --json out/reach.json

--nonlinear-reach needs far more tokens than the rest of the report, and the two
are separate flags for that reason. Everything else is the spectrum of an N x N
Gram, which is fine at N=4096 and is what headroom_results.log already used; at
N=32768 that matrix is 8.6 GB in float64 and cusolver's syevd workspace query
fails outright. The router fit is a regression in d_in features scored on held
out tokens, and it needs roughly 32x d_in to resolve. Raise --tokens, leave
--gram-tokens alone, and read the UNDERPOWERED guard before the gain column.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="run directory to load (uses nanochat.checkpoint_manager). "
                        "Omit to analyze a freshly initialized model.")
    p.add_argument("--step", type=int, default=-1, help="checkpoint step (-1 = latest)")
    p.add_argument("--depth", type=int, default=4, help="depth when no checkpoint is given")
    p.add_argument("--aspect-ratio", type=int, default=64)
    p.add_argument("--tokens", type=int, default=4096,
                   help="N, the number of token samples. Cost is O(N^2 d) and O(N^3); "
                        "the dof estimate is biased low by 1/(1 + dof/N)")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--tokenizer-dir", type=str, default="tokenizer")
    p.add_argument("--max-shards", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--json", type=str, default=None, help="write results here")
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="optimizer steps to take before measuring. A freshly initialized "
                        "model has zero-init output projections, so no gradient reaches "
                        "q/k/v/c_fc at step 0 and their headroom is undefined rather than "
                        "zero. Irrelevant when --checkpoint is given.")
    p.add_argument("--random-data", action="store_true",
                   help="use random token ids instead of the corpus (smoke test only: "
                        "headroom on random data is meaningless)")
    p.add_argument("--nonlinear-reach", action="store_true",
                   help="also fit a rank-r LINEAR and a rank-r NONLINEAR router per "
                        "projection and score both on held out tokens. The reach%% column "
                        "everywhere else is a full-rank linear fit, which the estimator's "
                        "own docstring calls a lower bound; this measures how much of that "
                        "bound is the router's linearity rather than its capacity. Costs a "
                        "few minutes of fitting and no training FLOPs.")
    p.add_argument("--reach-rank", type=int, default=64,
                   help="r for the two matched routers. The default matches the "
                        "--cond-router-rank 64 arm that was already trained, so mlp_r minus "
                        "lin_r prices the nonlinearity at a router cost we have a run for.")
    p.add_argument("--reach-steps", type=int, default=400,
                   help="Adam steps per router fit")
    p.add_argument("--reach-holdout", type=float, default=0.25,
                   help="fraction of tokens held out. Scores are computed only on these.")
    p.add_argument("--reach-score-tokens", type=int, default=4096,
                   help="cap on tokens used to SCORE (the Gram is quadratic in this). All "
                        "held-out tokens beyond the cap are dropped from scoring only; the "
                        "fit still uses every training token.")
    p.add_argument("--reach-signals", type=str, default="own",
                   help="comma list of signals the router reads, from own,late,label. 'own' is "
                        "the layer's own input, which is what every router in the p35 sweep and "
                        "every earlier version of this measurement used. 'late' is the last "
                        "block's output, standing in for any lookahead or two-pass design. "
                        "'label' is the target token embedding and is a strict ORACLE, since "
                        "dL/dy depends on the label and no causal router can have it. Only the "
                        "nonlinear fit and its control are run for the extra signals; the "
                        "lin_r/mlp_r pair is already settled on 'own'.")
    p.add_argument("--gram-tokens", type=int, default=4096,
                   help="cap on tokens used for every Gram-spectrum measurement (headroom, "
                        "dof, top1, supply, alignment, reach%%). These are N x N eigen"
                        "decompositions costing O(N^2) memory and O(N^3) time in float64, and "
                        "they converge long before the router fit does: at N=32768 the matrix "
                        "is 8.6 GB and cusolver's syevd workspace query fails outright. Only "
                        "--nonlinear-reach needs the full --tokens, so raising --tokens for it "
                        "does not have to drag this along. The 4096 default reproduces the "
                        "numbers already in headroom_results.log.")
    return p.parse_args()


# ── the measurement ──────────────────────────────────────────────────────────

def _eigvalsh(M):
    """eigvalsh that reports a bad matrix instead of dying inside cusolver.

    cusolver raises CUSOLVER_STATUS_INVALID_VALUE for two unrelated reasons, a
    non-finite entry and a workspace query it cannot satisfy, and its message
    blames NaN in both cases. Checking finiteness first separates them, and
    falling back to the CPU covers the size case. Returns None if the matrix is
    genuinely not finite, which callers already treat as "no gradient here".
    """
    if not torch.isfinite(M).all():
        return None
    try:
        return torch.linalg.eigvalsh(M).clamp(min=0.0)
    except Exception:
        return torch.linalg.eigvalsh(M.cpu()).clamp(min=0.0).to(M.device)


def headroom_from_grams(X, A, eps=1e-12, want_gram=False):
    """Spectrum of the per-token weight-gradient dispersion, from activations alone.

    X: (N, d_in)  layer inputs
    A: (N, d_out) gradients of the loss wrt the layer outputs

    Returns (headroom, dof, top1, n_eff). Never forms the (d_out, d_in) gradient
    of any single token, let alone their covariance.
    """
    X = X.double()
    A = A.double()
    N = X.shape[0]
    # G[s,t] = <g_s, g_t>_F for the rank-1 per-token gradients g_t = a_t x_t^T
    G = (A @ A.T) * (X @ X.T)
    # Both outputs are scale-invariant ratios, so normalize by the mean squared
    # gradient norm first. Gram entries are products of two Gram entries, so raw
    # magnitudes can span many orders of magnitude across layers and the squares
    # in the participation ratio are the first thing to overflow.
    scale = G.diagonal().mean().clamp(min=eps)
    G = G / scale
    # Center: subtract the mean gradient, which is what a dense layer learns.
    # H G H with H = I - 11^T/N does this without forming gbar in weight space.
    row = G.mean(dim=1, keepdim=True)
    tot = G.mean()
    Gc = G - row - row.T + tot
    # |gbar|^2 * N is the trace the mean direction accounts for
    mean_energy = tot * N
    evals = _eigvalsh(Gc)
    if evals is None:
        return (float('nan'),) * 3 + (N,) + ((Gc,) if want_gram else ())
    disp = evals.sum()
    total = disp + mean_energy.clamp(min=0)
    if total <= eps:
        # No gradient reaches this layer at all, which is NOT the same finding as
        # "the tokens agree". At init the dense recipe zero-inits attn.c_proj and
        # mlp.c_proj, so nothing flows back to q/k/v/c_fc on step 0. Reported
        # separately so it cannot be misread as "conditioning cannot help".
        return (float('nan'),) * 3 + (N,) + ((Gc,) if want_gram else ())
    if disp <= eps:
        return (0.0, 0.0, 0.0, N) + ((Gc,) if want_gram else ())
    headroom = (disp / (disp + mean_energy.clamp(min=0) + eps)).item()
    dof = ((disp ** 2) / (evals.square().sum() + eps)).item()
    top1 = (evals.max() / disp).item()
    return (headroom, dof, top1, N) + ((Gc,) if want_gram else ())


def supply_from_grams(Dact, W1, W2, eps=1e-12):
    """Spectrum of the per-token JACOBIAN dispersion an FFN already supplies.

    headroom_from_grams measures DEMAND: how many independent directions the
    tokens disagree along about what the weight should become. This measures
    SUPPLY: how many independent directions the layer's effective operator
    already varies along, for free, because it is nonlinear.

    That distinction is the one the p35 results turned on. A dense linear
    projection has J_t = W for every token, so it supplies exactly zero
    diversity, and everything ConditionedLinear did was an expensive way to buy
    some. An FFN with y = W2 g(W1 x) has J_t = W2 diag(g'(W1 x_t)) W1, which
    already varies per token at no extra cost. If supply already covers demand,
    conditioning is redundant no matter how it is implemented.

    Never forms a single Jacobian. With d_t = g'(W1 x_t),

        <J_s, J_t>_F = tr(D_s W2^T W2 D_t W1 W1^T) = d_s^T M d_t
        M = (W2^T W2) * (W1 W1^T)          Hadamard, h x h, built once

    so the whole N x N Gram is D M D^T. Verified against explicit Jacobians to
    2e-16 relative.

    Returns (supply_dof, Gram) with the Gram centered, so it is the dispersion
    around the mean Jacobian, which is what a dense layer would have used.
    """
    Dact = Dact.double()
    M = (W2.double().T @ W2.double()) * (W1.double() @ W1.double().T)
    Dc = Dact - Dact.mean(dim=0, keepdim=True)
    G = Dc @ M @ Dc.T
    G = G / G.diagonal().mean().clamp(min=eps)
    ev = _eigvalsh(G)
    if ev is None:
        return 0.0, G
    tot = ev.sum()
    if tot <= eps:
        return 0.0, G
    return ((tot ** 2) / (ev.square().sum() + eps)).item(), G


def alignment(G_demand, G_supply, eps=1e-12):
    """Centered kernel alignment between demand and supply, in token space.

    Matching dimension COUNTS is not the same as matching directions: a layer
    could vary its operator a great deal, in directions no token wants. Both
    Grams live over the same tokens, so CKA (Kornblith et al. 2019) asks the
    question that matters: is the variation the layer already supplies pointed
    where the gradients are pulling?

    1.0 = the free variation is exactly the wanted variation, and conditioning
    has nothing to add. 0.0 = none of the demand is met, which is the analytic
    value for any plain linear projection since it supplies no variation at all.
    """
    a, b = G_demand.double(), G_supply.double()
    a = a - a.mean(0, keepdim=True) - a.mean(1, keepdim=True) + a.mean()
    b = b - b.mean(0, keepdim=True) - b.mean(1, keepdim=True) + b.mean()
    den = a.norm() * b.norm()
    return ((a * b).sum() / den).item() if den > eps else 0.0


def alignment_null(G_demand, G_supply, draws=8, seed=0):
    """CKA under a random re-pairing of tokens: the null the raw number needs.

    Both Grams have diagonals dominated by per-token magnitude (gradient norm on
    one side, Jacobian deviation on the other), and those magnitudes correlate
    for trivial reasons: an unusual token is unusual in both. Two UNRELATED
    random Grams score ~0.5 on this measure for exactly that reason. Permuting
    the token index on one side destroys the per-token correspondence while
    preserving both spectra, so it isolates how much alignment is real.

    Report alignment - null. A value at the null means the layer's free operator
    variation carries no information about what the gradients want.
    """
    g = torch.Generator(device='cpu').manual_seed(seed)
    N = G_supply.shape[0]
    vals = []
    for _ in range(draws):
        idx = torch.randperm(N, generator=g).to(G_supply.device)
        vals.append(alignment(G_demand, G_supply[idx][:, idx]))
    return sum(vals) / len(vals)


def predictable_demand(X, A, ridge=1e-3):
    """How much of the demand ANY x-conditioned scheme could possibly reach.

    The per-token gradient is g_t = a_t x_t^T. x_t is known to a router by
    definition, so the only part that can be unpredictable is a_t. Much of a_t
    is minibatch and label noise, which no operator conditioned on x can
    anticipate, and that noise inflates the demand spectrum without offering
    anything to exploit. Projecting A onto the column space of X keeps only the
    part a linear router could have predicted and discards the rest.

    This is a LOWER bound on reachable demand: a nonlinear router could do
    better than a ridge fit. It is still the right first bound, because every
    router in the p35 sweep was linear in x.

    Returns (dof_reachable, energy_fraction).

    The fraction is an ENERGY ratio, tr(G_pred)/tr(G_full) on the centered Grams,
    not a ratio of participation ratios. A participation ratio is not monotone in
    rank: projecting onto a smaller subspace can flatten the spectrum and RAISE
    the PR, which is why a PR ratio produced impossible values above 100%. Energy
    is the quantity that actually partitions.
    """
    Xd, Ad = X.double(), A.double()
    n, d = Xd.shape
    Gx = Xd.T @ Xd
    Gx = Gx + ridge * Gx.diagonal().mean() * torch.eye(d, device=Xd.device, dtype=Xd.dtype)
    A_hat = Xd @ torch.linalg.solve(Gx, Xd.T @ Ad)

    def centered_gram(A_):
        G = (A_ @ A_.T) * (Xd @ Xd.T)
        r = G.mean(dim=1, keepdim=True)
        return G - r - r.T + G.mean()

    Gf, Gp = centered_gram(Ad), centered_gram(A_hat)
    tf = Gf.diagonal().sum()
    if not torch.isfinite(tf) or tf <= 0:
        return float('nan'), float('nan')
    ev = _eigvalsh(Gp / Gp.diagonal().mean().clamp(min=1e-12))
    if ev is None:
        return float('nan'), float('nan')
    tot = ev.sum()
    pr = ((tot ** 2) / (ev.square().sum() + 1e-12)).item() if tot > 0 else 0.0
    return pr, (Gp.diagonal().sum() / tf).clamp(0, 1).item()


def reach_by_router(X, A, rank=64, steps=400, lr=3e-3, holdout=0.25, seed=0,
                    ridge=1e-3, eps=1e-12, score_tokens=4096, S=None):
    """Reach as a function of the ROUTER rather than of the operator.

    predictable_demand() answers "how much of the demand could a full-rank LINEAR
    map from x reach", and its own docstring flags that as a lower bound. Every
    router in the p35 sweep was linear, and the answer came back between 44 and
    52 percent at every depth, every layer and every projection across four
    checkpoints. That invariance is why this function exists. If the cap is the
    router's linearity and not its rank, then the programme has been buying
    operator capacity (K templates, rank R, block width) to serve a selector that
    cannot address more than half of what the loss is asking for, and the cheap
    move is a nonlinear router rather than a bigger operator.

    Three predictors are fit on one train split and scored on one held out split,
    so the numbers are directly comparable:

        lin_full   ridge, full rank, closed form. The existing measurement.
        lin_r      x -> Linear(d_in, r) -> Linear(r, d_out)
        mlp_r      x -> Linear(d_in, r) -> GELU -> Linear(r, d_out)

    lin_r and mlp_r have identical parameter counts and identical FLOPs, so
    mlp_r minus lin_r is the value of the nonlinearity at fixed router cost.
    That difference is the decision this measurement exists to make: it is what a
    cheap nonlinear router would buy, priced before a training run is spent on one.

    THE SCORE IS A RESIDUAL AGAINST A STATIC BASELINE, NOT AN ENERGY.
    predictable_demand() reports tr(G_pred)/tr(G_full), which is a fraction of
    explained signal only when the predictor is an orthogonal projection. Ridge
    is; a trained net is not, and a net emitting large uncorrelated predictions
    would score near 1 on that formula while explaining nothing. Everything here
    is scored instead as

        1 - tr(cg(A - A_hat)) / tr(cg(A - abar))       cg = double-centered Gram

    where abar is the TRAIN mean. The abar denominator is not cosmetic. A
    constant predictor still shrinks tr(cg(A - c)) because the predicted gradient
    c x_t^T varies with x_t even when c does not, so without it a net that had
    learned nothing but the mean scored +0.27 on the permuted control. Against
    this baseline a mean-predictor scores exactly 0, which is what makes the
    permutation control readable. The quantity is therefore the fraction of the
    TOKEN-VARYING demand a router of this form explains beyond a static one,
    which is the question the conditioning programme is actually asking.

    HOLDOUT IS NOT OPTIONAL. r*(d_in + d_out) parameters can memorize N tokens.
    Every score is on tokens the fit never saw, and mlp_null repeats the
    nonlinear fit against a permuted target as a second control. A large mlp_r
    next to a large mlp_null is a fitting artifact, not headroom.

    UNDERFITTING IS THE OTHER FAILURE. If the fit does not converge, the gain
    reads as zero and the verdict wrongly says the direction is dead. mlp_train
    is returned for exactly that: a low mlp_train means the optimizer failed and
    the run says nothing, not that the headroom is absent.

    S IS THE SIGNAL THE ROUTER READS, and defaults to X. Every router in the p35
    sweep, and every earlier version of this measurement, fed the router the
    layer's own input. Group K then showed that at fixed cost it does not matter
    how the conditioning budget is split between the operator and the router, so
    the remaining question is not what the router COMPUTES but what it SEES.
    Passing S separately answers that: the fits use S, while the Grams keep using
    X because the demand is a property of the layer's real geometry and does not
    change with the router's input. Note that a_t = dL/dy_t depends on the label,
    so a signal carrying label information is an ORACLE and bounds what any
    causal router could ever reach; it is there to tell an unlucky architecture
    apart from an impossible one.

    Returns a dict of held out scores.
    """
    import torch.nn.functional as F

    dev = X.device
    if S is None:
        S = X
    N, d_in = S.shape          # d_in is the ROUTER's input width, which may differ from X's
    d_out = A.shape[1]
    n_te = max(64, int(round(holdout * N)))
    if N - n_te < 128:
        return None
    idx = torch.randperm(N, generator=torch.Generator().manual_seed(seed)).to(dev)
    te, tr = idx[:n_te], idx[n_te:]
    Xtr, Atr, Xte, Ate = X[tr], A[tr], X[te], A[te]
    Str, Ste = S[tr], S[te]

    # Scoring is quadratic in the number of tokens it scores on, so it is capped
    # independently of the fit. All n_train tokens are used to FIT; at most
    # score_tokens of each split are used to SCORE. The Gram trace converges in
    # the scoring sample long before the regression converges in the fitting one,
    # and without the cap a --tokens 32768 run allocates gigabytes per projection.
    def cap(idx_len, t):
        return t[:min(idx_len, score_tokens)]

    Xte_s, Ate_s = cap(n_te, Xte), cap(n_te, Ate)
    n_s = Xte_s.shape[0]
    Xd = Xte_s.double()
    Kx = Xd @ Xd.T
    Xtr_s, Atr_s = cap(N - n_te, Xtr), cap(N - n_te, Atr)
    Ktr = Xtr_s.double() @ Xtr_s.double().T

    def centered_trace(T, K):
        # tr(cg(G)) = tr(G) - sum(G)/n for the double centering
        # G - r1^T - 1r^T + mean(G), so none of those terms need materializing.
        G = (T.double() @ T.double().T).mul_(K)
        return G.diagonal().sum() - G.sum() / G.shape[0]

    abar = Atr.mean(0, keepdim=True)
    t_full = centered_trace(Ate_s, Kx)
    base = centered_trace(Ate_s - abar, Kx)
    base_tr = centered_trace(Atr_s - abar, Ktr)
    if not (torch.isfinite(base) and base > 0 and torch.isfinite(t_full) and t_full > 0):
        return None

    def score(A_hat):
        return float((1.0 - centered_trace(Ate_s - A_hat[:n_s], Kx) / base).clamp(-1, 1))

    def score_train(A_hat):
        return float((1.0 - centered_trace(Atr_s - A_hat[:Atr_s.shape[0]], Ktr)
                      / base_tr).clamp(-1, 1))

    # lin_full: the closed form ridge, fit on the train split only so it is
    # scored on the same tokens as the other two. Fit FROM the signal S, which is
    # X unless the caller is asking what a differently-informed router could do.
    Xt, At = Str.double(), Atr.double()
    Gx = Xt.T @ Xt
    Gx = Gx + ridge * Gx.diagonal().mean() * torch.eye(d_in, device=dev, dtype=Xt.dtype)
    B = torch.linalg.solve(Gx, Xt.T @ At)
    Ahat_full = Ste.double()[:n_s] @ B
    G_pred = (Ahat_full @ Ahat_full.T).mul_(Kx)
    energy_legacy = float((((G_pred.diagonal().sum() - G_pred.sum() / n_s))
                           / t_full).clamp(0, 1))
    del G_pred

    # Standardizing x is absorbed by the first layer and scaling the target by
    # one global scalar is absorbed by the last, so neither changes the function
    # class of either predictor. Both keep Adam in a sane range.
    xm, xs = Str.mean(0), Str.std(0).clamp(min=eps)
    Xtr_n, Xte_n = (Str - xm) / xs, (Ste - xm) / xs

    def fit(nonlinear, target):
        torch.manual_seed(seed)
        layers = [nn.Linear(d_in, rank)]
        if nonlinear:
            layers.append(nn.GELU())
        layers.append(nn.Linear(rank, d_out, bias=False))
        net = nn.Sequential(*layers).to(dev)
        s = target.std().clamp(min=eps)
        tgt = target / s
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        n, bs = Xtr_n.shape[0], min(1024, Xtr_n.shape[0])
        for _ in range(steps):
            j = torch.randint(0, n, (bs,), device=dev)
            opt.zero_grad(set_to_none=True)
            F.mse_loss(net(Xtr_n[j]), tgt[j]).backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            return net(Xte_n) * s, net(Xtr_n) * s

    shuffled = Atr[torch.randperm(Atr.shape[0], generator=torch.Generator().manual_seed(seed + 1)).to(dev)]
    lin_te, _ = fit(False, Atr)
    mlp_te, mlp_tr = fit(True, Atr)
    null_te, _ = fit(True, shuffled)
    return dict(lin_full=score(Ahat_full.float()), lin_full_energy=energy_legacy,
                lin_r=score(lin_te), mlp_r=score(mlp_te), mlp_train=score_train(mlp_tr),
                mlp_null=score(null_te), rank=rank, n_test=n_te,
                n_train=int(N - n_te), d_in=int(d_in))


@torch.no_grad()
def _subsample(t, n):
    flat = t.reshape(-1, t.shape[-1])
    if flat.shape[0] <= n:
        return flat
    stride = flat.shape[0] // n
    return flat[::stride][:n]


def collect_signals(model, batches, n_tokens, device):
    """Per-token signals a router could read INSTEAD of the layer's own input.

    Returned on the same token subsample as collect(), because _subsample is a
    deterministic stride over (B*T) and every tensor here has the same (B, T)
    layout. Two signals, chosen to bracket the question rather than to scan:

      late    the last block's output. Not causally available to layer L in this
              architecture, so it stands in for any two-pass or lookahead design.
      label   the embedding of the TARGET token. a_t = dL/dy_t is a function of
              the label, so this is a strict ORACLE: no causal router can have
              it. It exists to separate "our routers read the wrong thing" from
              "the demand is not a function of anything a forward pass knows".

    If reach does not move under `late`, no rearrangement of forward information
    helps. If it does not move under `label` either, the unreached demand is
    minibatch noise and the ceiling is not an architecture problem at all.
    """
    out = {'late': [], 'label': []}
    grab = {}
    h = model.transformer.h[-1].register_forward_hook(
        lambda m, i, o: grab.__setitem__('late', (o[0] if isinstance(o, tuple) else o).detach()))
    per_batch = max(1, n_tokens // max(1, len(batches)))
    with torch.no_grad():
        for ids in batches:
            model(ids, ids)
            out['late'].append(_subsample(grab['late'], per_batch).float().cpu())
            # targets are the next-token ids, so shift and pad the last position
            tgt = torch.cat([ids[:, 1:], ids[:, -1:]], dim=1)
            out['label'].append(_subsample(model.transformer.wte(tgt), per_batch).float().cpu())
    h.remove()
    return {k: torch.cat(v)[:n_tokens] for k, v in out.items()}


def collect(model, batches, target_modules, n_tokens, device):
    """One forward and backward per batch, capturing (input, grad_output) per layer."""
    store = {name: {'x': [], 'a': [], 'pre': []} for name in target_modules}
    handles = []

    def mk_hook(name):
        def fwd_hook(mod, inp, out):
            store[name]['x'].append(inp[0].detach())
            # For an FFN's first projection this output IS the pre-activation,
            # which is all the supply side needs.
            store[name]['pre'].append(out.detach())

        def bwd_hook(mod, grad_in, grad_out):
            store[name]['a'].append(grad_out[0].detach())
        return fwd_hook, bwd_hook

    for name, mod in target_modules.items():
        f, b = mk_hook(name)
        handles.append(mod.register_forward_hook(f))
        handles.append(mod.register_full_backward_hook(b))

    per_batch = max(1, n_tokens // max(1, len(batches)))
    kept = {name: {'x': [], 'a': [], 'pre': []} for name in target_modules}
    for ids in batches:
        model.zero_grad(set_to_none=True)
        loss = model(ids, ids)
        loss.backward()
        for name in target_modules:
            if not store[name]['x'] or not store[name]['a']:
                continue
            x = store[name]['x'][-1]
            a = store[name]['a'][-1]
            kept[name]['x'].append(_subsample(x, per_batch).float().cpu())
            kept[name]['a'].append(_subsample(a, per_batch).float().cpu())
            if store[name]['pre']:
                kept[name]['pre'].append(_subsample(store[name]['pre'][-1], per_batch).float().cpu())
                store[name]['pre'].clear()
            store[name]['x'].clear()
            store[name]['a'].clear()

    for h in handles:
        h.remove()
    out = {}
    for name in target_modules:
        if kept[name]['x']:
            pre = torch.cat(kept[name]['pre'])[:n_tokens] if kept[name]['pre'] else None
            out[name] = (torch.cat(kept[name]['x'])[:n_tokens],
                         torch.cat(kept[name]['a'])[:n_tokens], pre)
    return out


def main():
    args = parse_args()
    device = args.device
    torch.manual_seed(0)

    from nanochat.gpt import GPT, GPTConfig

    # ── model ────────────────────────────────────────────────────────────────
    if args.checkpoint:
        from nanochat.checkpoint_manager import load_model_from_dir
        model, _tok, _meta = load_model_from_dir(args.checkpoint, device, phase="eval",
                                                 step=None if args.step < 0 else args.step)
        model = model.module if hasattr(model, 'module') else model
        cfg = model.config
        print(f"loaded {args.checkpoint} (n_layer={cfg.n_layer}, n_embd={cfg.n_embd})")
    else:
        dim = ((args.depth * args.aspect_ratio + 127) // 128) * 128
        cfg = GPTConfig(sequence_len=args.seq_len, vocab_size=65536, n_layer=args.depth,
                        n_head=max(1, dim // 128), n_kv_head=max(1, dim // 128), n_embd=dim)
        with torch.device('meta'):
            model = GPT(cfg)
        model.to_empty(device=device)
        model.init_weights()
        print(f"freshly initialized dense model (n_layer={cfg.n_layer}, n_embd={cfg.n_embd})")
        print("NOTE: headroom at random init is not the headroom of a trained model. "
              "Point --checkpoint at a real run for a number you can act on.")
    model.train()

    # ── data ─────────────────────────────────────────────────────────────────
    n_batches = max(1, args.tokens // (args.batch * args.seq_len))
    if args.random_data:
        print("WARNING: --random-data, headroom on random ids measures nothing real")
        batches = [torch.randint(0, cfg.vocab_size, (args.batch, args.seq_len), device=device)
                   for _ in range(n_batches)]
    else:
        from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
        from nanochat.tokenizer import get_tokenizer
        tok = get_tokenizer(args.tokenizer_dir) if os.path.isdir(args.tokenizer_dir) else None
        loader = tokenizing_distributed_data_loader_bos_bestfit(
            tok, args.batch, args.seq_len, split="val", device=device,
            data_dir=args.data_dir, max_shards=args.max_shards)
        batches = [next(loader)[0] for _ in range(n_batches)]

    # ── which layers ─────────────────────────────────────────────────────────
    # Any module with a single (in -> out) linear action is measurable. We take
    # every attention/MLP projection, whatever class currently implements it.
    targets, ffn_of = {}, {}
    for name, mod in model.named_modules():
        leaf = name.rsplit('.', 1)[-1]
        if leaf in ('c_q', 'c_k', 'c_v', 'c_proj', 'c_fc') and 'transformer.h.' in name:
            targets[name] = mod
            if leaf == 'c_fc':
                parent = model.get_submodule(name.rsplit('.', 1)[0])
                # Supply is only defined where a nonlinearity sits between two
                # known projections. ReLU^2 gives g'(z) = 2 relu(z).
                if hasattr(parent, 'c_proj') and hasattr(parent.c_proj, 'weight') \
                        and hasattr(mod, 'weight'):
                    ffn_of[name] = (mod.weight, parent.c_proj.weight,
                                    lambda z: 2.0 * torch.relu(z))
    if not targets:
        print("no projections found")
        return
    print(f"measuring {len(targets)} projections over {args.tokens} tokens "
          f"({n_batches} batches of {args.batch}x{args.seq_len})")

    if args.warmup_steps > 0:
        opt = model.setup_optimizer(unembedding_lr=4e-3, embedding_lr=0.2, matrix_lr=0.02)
        for i in range(args.warmup_steps):
            opt.zero_grad()
            model(batches[i % len(batches)], batches[i % len(batches)]).backward()
            opt.step()
        print(f"took {args.warmup_steps} warmup steps so gradients reach every projection")

    data = collect(model, batches, targets, args.tokens, device)

    sig_names = [s.strip() for s in args.reach_signals.split(',') if s.strip()]
    bad = [s for s in sig_names if s not in ('own', 'late', 'label')]
    if bad:
        raise SystemExit(f"unknown --reach-signals {bad}, pick from own,late,label")
    alt = {}
    if args.nonlinear_reach and [s for s in sig_names if s != 'own']:
        print(f"capturing alternate router signals: {[s for s in sig_names if s != 'own']}")
        alt = collect_signals(model, batches, args.tokens, device)

    # ── report ───────────────────────────────────────────────────────────────
    rows = []
    ng = args.gram_tokens
    if args.tokens > ng:
        print(f"Gram measurements use the first {ng} tokens (--gram-tokens); "
              f"only --nonlinear-reach uses all {args.tokens}")
    for name, (X, A, PRE) in data.items():
        # Two different sample sizes on purpose. The Gram spectra are N x N
        # eigendecompositions and converge at a few thousand tokens; the router
        # fit is a regression in d_in features and does not.
        Xg, Ag = X[:ng].to(device), A[:ng].to(device)
        h, dof, top1, n, Gd = headroom_from_grams(Xg, Ag, want_gram=True)
        parts = name.split('.')
        layer = int(parts[parts.index('h') + 1])
        proj = '.'.join(parts[parts.index('h') + 2:])
        sup, ali, null = 0.0, 0.0, 0.0
        if name in ffn_of and PRE is not None and h == h:
            W1, W2, gprime = ffn_of[name]
            sup, Gs = supply_from_grams(gprime(PRE[:ng].to(device)),
                                        W1.to(device), W2.to(device))
            ali = alignment(Gd, Gs)
            null = alignment_null(Gd, Gs)
        reach, reach_frac = (float('nan'), float('nan'))
        router = None
        if h == h:
            reach, reach_frac = predictable_demand(Xg, Ag)
            if args.nonlinear_reach:
                Xf, Af = X.to(device), A.to(device)
                router = reach_by_router(Xf, Af, rank=args.reach_rank,
                                         steps=args.reach_steps, holdout=args.reach_holdout,
                                         score_tokens=args.reach_score_tokens)
                for s in sig_names:
                    if s == 'own' or router is None:
                        continue
                    # Same X for the Grams, same split, same rank, same steps:
                    # only the router's INPUT changes, so the difference is
                    # attributable to the signal and to nothing else.
                    r2 = reach_by_router(Xf, Af, rank=args.reach_rank,
                                         steps=args.reach_steps, holdout=args.reach_holdout,
                                         score_tokens=args.reach_score_tokens,
                                         S=alt[s][:Xf.shape[0]].to(device))
                    if r2 is not None:
                        router[f'{s}_mlp'] = r2['mlp_r']
                        router[f'{s}_null'] = r2['mlp_null']
        rows.append(dict(layer=layer, proj=proj, headroom=h, dof=dof, top1=top1, n=n,
                         supply=sup, align=ali, align_null=null,
                         reach=reach, reach_frac=reach_frac, router=router,
                         unmet=(dof * (1.0 - ali)) if dof == dof else float('nan')))

    rows.sort(key=lambda r: (r['layer'], r['proj']))
    print("\n" + "=" * 78)
    print("PER LAYER  (mean over projections)")
    print("=" * 78)
    print(f"  {'layer':>5s}  {'headroom':>9s}  {'dof':>8s}  {'top1':>6s}   verdict")
    by_layer = {}
    for r in rows:
        by_layer.setdefault(r['layer'], []).append(r)
    for layer in sorted(by_layer):
        rs = by_layer[layer]
        ok = [x for x in rs if x['headroom'] == x['headroom']]
        if not ok:
            print(f"  {layer:>5d}  {'n/a':>9s}  {'n/a':>8s}  {'n/a':>6s}   "
                  f"no gradient reaches this layer (see --warmup-steps)")
            continue
        h = sum(x['headroom'] for x in ok) / len(ok)
        d = sum(x['dof'] for x in ok) / len(ok)
        t = sum(x['top1'] for x in ok) / len(ok)
        if h != h:  # NaN
            verdict = "no gradient reaches this layer (see --warmup-steps)"
        elif d > 0.5 * ng:
            verdict = f"dof unresolvable at N={ng}, raise --gram-tokens"
        elif h < 0.05:
            verdict = "no headroom: conditioning cannot help here"
        elif t > 0.6:
            verdict = "one-dimensional: a rank-1 delta is enough, a bank is waste"
        elif d < 4:
            verdict = f"narrow: K or R above ~{max(2, int(d))} buys nothing"
        else:
            verdict = f"real headroom for up to ~{int(d)} conditioning directions"
        print(f"  {layer:>5d}  {h:>9.4f}  {d:>8.1f}  {t:>6.3f}   {verdict}")

    print("\n" + "=" * 78)
    print("SUPPLY vs DEMAND   (the number the whole conditioning programme turned on)")
    print("=" * 78)
    print(f"  {'layer':>5s}  {'demand':>7s}  {'supply':>7s}  {'align':>6s} {'null':>6s} {'excess':>7s}  "
          f"{'reach':>7s} {'reach%':>6s}   what it means")
    for layer in sorted(by_layer):
        ffn = [r for r in by_layer[layer] if r['supply'] > 0]
        att = [r for r in by_layer[layer] if r['proj'].startswith('attn') and r['headroom'] == r['headroom']]
        if ffn:
            r = ffn[0]
            ex = r['align'] - r['align_null']
            v = ("supply covers demand: conditioning is redundant here" if ex > 0.30 else
                 "partly aligned: some of the free variation is useful" if ex > 0.05 else
                 "AT THE NULL: the free variation carries no information about demand"
                 if ex > -0.05 else "below the null")
            print(f"  {layer:>5d}  {r['dof']:7.1f}  {r['supply']:7.1f}  {r['align']:6.3f} "
                  f"{r['align_null']:6.3f} {ex:+7.3f}  {r['reach']:7.1f} {r['reach_frac']:6.1%}   {v}")
        if att:
            d = sum(x['dof'] for x in att) / len(att)
            rf = [x['reach_frac'] for x in att if x['reach_frac'] == x['reach_frac']]
            print(f"  {'':>5s}  {d:7.1f}  {0.0:7.1f}  {0.0:6.3f} {0.0:6.3f} {0.0:+7.3f}  "
                  f"{sum(x['reach'] for x in att)/len(att):7.1f} {(sum(rf)/len(rf) if rf else 0):6.1%}   "
                  f"attn projections: linear, supply is ZERO by construction")

    fa = [r for r in rows if r['supply'] > 0]
    if fa:
        print(f"\n  FFN mean: demand {sum(r['dof'] for r in fa)/len(fa):6.1f}   "
              f"supply {sum(r['supply'] for r in fa)/len(fa):6.1f}   "
              f"alignment {sum(r['align'] for r in fa)/len(fa):.3f}")
        rf = [r['reach_frac'] for r in fa if r['reach_frac'] == r['reach_frac']]
        print(f"  excess over null {sum(r['align']-r['align_null'] for r in fa)/len(fa):+.3f}   "
              f"reachable demand {sum(rf)/len(rf):.1%} of measured")
        print("\n  TWO CONTROLS, read them before the raw alignment.")
        print("  null    CKA after permuting the token index on the supply side. Both Grams")
        print("          have magnitude-dominated diagonals, so unrelated Grams score high")
        print("          for trivial reasons. Only alignment MINUS null is signal.")
        print("  reach%  fraction of the demand a linear router could even in principle")
        print("          predict from x. The rest is minibatch and label noise that inflates")
        print("          the spectrum and offers nothing to condition on. If this is small,")
        print("          there is no headroom for ANY x-conditioned design and the measured")
        print("          demand was never the opportunity it looked like.")
        print("\n  Read it this way. A plain linear projection has J_t = W for every token,")
        print("  so it supplies zero operator variation and its entire demand is unmet.")
        print("  An FFN supplies variation for free because it is nonlinear. Where alignment")
        print("  is high the free variation already points where the gradients pull, and no")
        print("  conditioning scheme can add anything there at any price. Where alignment is")
        print("  low there is room, and the cheap way to buy it is more nonlinearity per FLOP")
        print("  (depth multiplies linear regions, width only adds them), not more parameters.")

    rt = [r for r in rows if r.get('router')]
    if rt:
        r0 = rt[0]['router']
        print("\n" + "=" * 78)
        print(f"ROUTER CEILING   (held out on {r0['n_test']} tokens, r={r0['rank']})")
        print("=" * 78)
        print("  Every reach% above is a FULL RANK LINEAR fit. These four columns ask")
        print("  whether that ceiling is the router's linearity or its capacity. lin_r and")
        print("  mlp_r have the same parameters and the same FLOPs, so their difference is")
        print("  the price of the nonlinearity at a router cost we already have a run for.")
        print(f"  Scores are 1 - tr(cg(A - Ahat))/tr(cg(A - abar)), so a predictor no better")
        print(f"  than a static one scores 0 and a worse one scores negative.")
        print("  A projection resolves only when its own permuted control is near zero.")
        print("  d_in differs 4x across projection types here (mlp.c_proj reads the FFN")
        print("  hidden width), so they do not all resolve at the same --tokens and one")
        print("  unresolved projection must not be allowed to speak for the rest.")
        print(f"\n  {'projection':18s} {'d_in':>6s} {'lin_full':>9s} {'lin_r':>8s} "
              f"{'mlp_r':>8s} {'mlp_tr':>7s} {'mlp_null':>9s} {'gain':>8s}  ok")

        def agg(rs, key):
            v = [x['router'][key] for x in rs if x['router'][key] == x['router'][key]]
            return sum(v) / len(v) if v else float('nan')

        def resolved(x):
            return x['router']['mlp_null'] > -0.10

        by_p = {}
        for r in rt:
            by_p.setdefault(r['proj'], []).append(r)
        for proj in sorted(by_p):
            rs = by_p[proj]
            lr, mr = agg(rs, 'lin_r'), agg(rs, 'mlp_r')
            nok = sum(1 for x in rs if resolved(x))
            print(f"  {proj:18s} {rs[0]['router']['d_in']:6d} {agg(rs, 'lin_full'):9.3f} "
                  f"{lr:8.3f} {mr:8.3f} {agg(rs, 'mlp_train'):7.3f} "
                  f"{agg(rs, 'mlp_null'):9.3f} {mr - lr:+8.3f}  {nok}/{len(rs)}")
        ok = [x for x in rt if resolved(x)]
        lf, lr, mr, mn = (agg(rt, 'lin_full'), agg(rt, 'lin_r'),
                          agg(rt, 'mlp_r'), agg(rt, 'mlp_null'))
        mtr = agg(rt, 'mlp_train')
        gain = (agg(ok, 'mlp_r') - agg(ok, 'lin_r')) if ok else float('nan')
        print(f"  {'-' * 80}")
        print(f"  {'all':18s} {'':6s} {lf:9.3f} {lr:8.3f} {mr:8.3f} {mtr:7.3f} "
              f"{mn:9.3f} {'':8s}  {len(ok)}/{len(rt)}")
        if ok:
            print(f"  {'resolved only':18s} {'':6s} {agg(ok, 'lin_full'):9.3f} "
                  f"{agg(ok, 'lin_r'):8.3f} {agg(ok, 'mlp_r'):8.3f} "
                  f"{agg(ok, 'mlp_train'):7.3f} {agg(ok, 'mlp_null'):9.3f} {gain:+8.3f}")
        print("\n  The gain is taken over RESOLVED projections only. Everything else is")
        print("  reported so an unresolved row is visible rather than silently averaged in.")

        extra = [s for s in sig_names if s != 'own' and any(f'{s}_mlp' in x['router'] for x in rt)]
        if extra and ok:
            print("\n  WHAT THE ROUTER READS   (same Grams, same split, same rank; only the")
            print("  router's input differs. Resolved projections only.)")
            hdr = "  " + f"{'signal':10s} {'mlp_r':>8s} {'null':>8s} {'vs own':>8s}   what it is"
            print(hdr)
            own = agg(ok, 'mlp_r')
            print(f"  {'own':10s} {own:8.3f} {agg(ok, 'mlp_null'):8.3f} {0.0:+8.3f}   "
                  f"the layer's own input, what every arm used")
            desc = {'late': "last block output: any lookahead or two-pass design",
                    'label': "target embedding: a strict ORACLE, no causal router has it"}
            best = own
            for s in extra:
                v, n_ = agg(ok, f'{s}_mlp'), agg(ok, f'{s}_null')
                best = max(best, v)
                print(f"  {s:10s} {v:8.3f} {n_:8.3f} {v - own:+8.3f}   {desc.get(s, '')}")
            print("\n  READ IT THIS WAY")
            if best - own < 0.03:
                print(f"  Nothing moves it. The best alternate signal gains {best - own:+.3f}, and")
                print("  that includes the oracle if you ran it. The unreached demand is not a")
                print("  function of anything in the forward pass, so it is minibatch and label")
                print("  noise rather than information a better-informed router could exploit.")
                print("  No routing scheme can reach it, at any cost, from any signal. Combined")
                print("  with group K, that closes per-token operator conditioning: neither the")
                print("  router's function class, nor its capacity, nor its input is the cap.")
            elif 'label' in extra and agg(ok, 'label_mlp') - own > 0.10 and \
                    all(agg(ok, f'{s}_mlp') - own < 0.05 for s in extra if s != 'label'):
                print("  Only the ORACLE moves it. The demand is predictable, but only from the")
                print("  label, which no causal router can see. That is a clean impossibility")
                print("  result rather than a design failure, and it is the honest end of the")
                print("  direction: the headroom is real and structurally unreachable.")
            else:
                print(f"  A forward-available signal gains {best - own:+.3f} over the layer's own")
                print("  input. The routers were reading the wrong thing, which is a different")
                print("  and fixable problem from everything groups I through K ruled out.")
                print("  Next: feed that signal to ConditionedLinear (--cond-gate-source ctx is")
                print("  the existing hook) and re-run the matched pair.")
        print(f"\n  lin_full also scores {agg(rt, 'lin_full_energy'):.3f} on the legacy energy")
        print("  formula tr(G_pred)/tr(G_full), which is what headroom_results.log reports.")
        print("  The two are not comparable and the residual form is the honest one: a trained")
        print("  net is not an orthogonal projection, and the energy form would reward it for")
        print("  emitting large predictions that explain nothing.")
        print("\n  VERDICT")
        n_tr, d_in_max = r0['n_train'], max(x['router']['d_in'] for x in rt)
        dof_seen = [x['dof'] for x in rt if x['dof'] == x['dof']]
        dof_mean = sum(dof_seen) / max(1, len(dof_seen))
        if len(ok) < 0.5 * len(rt):
            worst = max((x for x in rt if not resolved(x)),
                        key=lambda x: x['router']['d_in'])
            print(f"  UNDERPOWERED, no verdict. Only {len(ok)} of {len(rt)} projections have a")
            print(f"  permuted control near zero, so most of the table is fitting noise. The")
            print(f"  widest unresolved one is {worst['proj']} at d_in={worst['router']['d_in']} "
                  f"with a control of {worst['router']['mlp_null']:+.3f}.")
            print(f"  You fit {n_tr} train tokens against up to {d_in_max} input features with a")
            print(f"  demand of ~{dof_mean:.0f} dof. Re-run with --tokens {max(32768, 32 * d_in_max)}")
            print("  or higher. On a synthetic target with a known nonlinear component, N = 32x")
            print("  d_in is where the permuted control reaches zero and the gain stops moving;")
            print("  below it the gain is not merely noisy, it comes out with the WRONG SIGN.")
        elif mn > 0.10:
            print(f"  mlp_null is {mn:+.3f}, not ~0. A fit on permuted targets should not")
            print("  generalize at all, so the held out split is leaking and the gain column")
            print("  is not trustworthy. Raise --tokens or --reach-holdout and re-run.")
        elif mtr < 0.05:
            print(f"  The fit did not converge: mlp_train is only {mtr:+.3f}, so the network")
            print("  never learned the training split either. The gain column says nothing")
            print("  about headroom. Raise --reach-steps, or lower --reach-rank.")
        elif gain > 0.10:
            print(f"  The nonlinearity buys {gain:+.3f} of reach at ZERO extra router FLOPs.")
            print("  The ceiling was the router's linearity, not the operator's capacity, and")
            print("  every arm in the p35 sweep was spending on the wrong side of the layer.")
            print(f"  Next: one d8 arm with a rank-{r0['rank']} nonlinear router on the two")
            print("  highest-demand projections. That is ~5% overhead against the ~20% that")
            print("  H3 and H4 paid, and it needs only their quality gain to clear 1.0x.")
        elif gain > 0.03:
            print(f"  The nonlinearity buys {gain:+.3f}, real but small. A nonlinear router")
            print("  alone will not close the 6% gap to dense at d8. Worth one arm only if it")
            print("  stacks with the targeting that H3 and H4 already found.")
        else:
            print(f"  The nonlinearity buys {gain:+.3f}, nothing. The unreachable half of the")
            print("  demand is not a function of the current token at all, so no router of any")
            print("  design can address it and per-token operator conditioning cannot pay at")
            print("  any price. That is a real finding and it is the end of this direction.")

    print("\n" + "=" * 78)
    print("PER PROJECTION  (mean over layers)")
    print("=" * 78)
    print(f"  {'projection':18s}  {'headroom':>9s}  {'dof':>8s}  {'top1':>6s}")
    by_proj = {}
    for r in rows:
        by_proj.setdefault(r['proj'], []).append(r)
    for proj in sorted(by_proj):
        rs = by_proj[proj]
        ok = [x for x in rs if x['headroom'] == x['headroom']]
        if not ok:
            print(f"  {proj:18s}  {'n/a':>9s}  {'n/a':>8s}  {'n/a':>6s}")
            continue
        print(f"  {proj:18s}  {sum(x['headroom'] for x in ok) / len(ok):>9.4f}"
              f"  {sum(x['dof'] for x in ok) / len(ok):>8.1f}"
              f"  {sum(x['top1'] for x in ok) / len(ok):>6.3f}")

    ok_rows = [r for r in rows if r['headroom'] == r['headroom']]
    all_h = sum(r['headroom'] for r in ok_rows) / max(1, len(ok_rows))
    all_d = sum(r['dof'] for r in ok_rows) / max(1, len(ok_rows))
    if len(ok_rows) < len(rows):
        print(f"\n  {len(rows) - len(ok_rows)} of {len(rows)} projections received no "
              f"gradient and are excluded; pass --warmup-steps 20")
    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("  headroom  fraction of the per-token weight-gradient signal that a single")
    print("            static operator cannot represent. This is the ceiling on what")
    print("            ANY input-conditioned design can win at that layer, before")
    print("            architecture, routing or optimization enter the picture.")
    print("  dof       independent directions the tokens disagree along. Compare to")
    print("            K-1 for a template bank and to R for ConditionedLinear. Biased")
    print(f"            low by 1/(1 + dof/N); at N={ng} a true value of 64 reads as")
    print(f"            {64 / (1 + 64 / ng):.0f}.")
    print("  top1      share of the dispersion in its leading direction. Near 1 means")
    print("            the disagreement is essentially rank-1.")
    print(f"\n  model-wide: headroom {all_h:.4f}, dof {all_d:.1f}")
    if all_h < 0.05:
        print("  => the tokens broadly agree on the weight update. No conditional linear")
        print("     layer of any design can beat dense by much here, and the honest move")
        print("     is to spend the parameters on something else.")
    else:
        print(f"  => there is real headroom. Size K or R against the per-layer dof column")
        print("     rather than uniformly, and spend nothing where headroom is ~0.")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump({'rows': rows, 'tokens': args.tokens, 'gram_tokens': ng,
                       'checkpoint': args.checkpoint}, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
