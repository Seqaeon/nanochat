# Research direction: addressability, not capacity

## The claim

A conditional linear layer has two halves. The **operator** is the set of weight
matrices the layer can express. The **router** is the map from a token to a
choice within that set. Every design in this project spent its FLOPs on the
operator and almost nothing on the router, and every one of them lost to dense
on FLOPs per unit of bits-per-byte.

The hypothesis of this direction is that we were spending on the wrong half. The
binding constraint is not how many distinct operators a layer can express. It is
how many of them the router can actually address from the information available
to it at that point in the network.

If that is right, the object worth building is not a better operator. It is a
better selector, which is a far smaller and far cheaper thing.

## Motivation

A dense linear layer applies one operator to every token. The transformer's
compute is dominated by these layers. If a token-dependent operator can be built
for less compute than the quality it buys, that is a strict improvement to every
transformer, and it compounds with scale.

The metric is FLOPs per unit of bits-per-byte, expressed as an **effective
compute multiplier**: the training compute a dense model would need to reach the
same BPB, divided by the compute the arm actually spent. Above 1.0 is a win.
Quality at fixed parameters is not the metric, and neither is quality at fixed
steps. Both flattered every arm we ran, which is why neither is used here.

## What we built, and what it cost

Two families of conditional operator, wired through the full sweep with matched
token budgets scaled by active parameters:

```
RemixedLinear       W_eff(x) = sum_k alpha_k(x) T_k        K templates, softmax router
ConditionedLinear   W_eff(x) = W0 + U diag(c(x)) V^T       rank R additive delta
                    plus a multiplicative variant, z_i = z_{i-1} + c_i(x) u_i (v_i^T z_{i-1})
```

Plus a group of experiments on where FFN capacity should sit, driven by a
per-layer demand measurement taken on a trained dense d22 checkpoint.

Effective compute multipliers against the dense curve in the same protocol
(dense d8 to d12 gives a local exponent of -0.0572; the noise floor across arms
that should be equivalent to dense is plus or minus 0.0013 BPB, which is plus or
minus 2.4 percent on the multiplier):

| arm | what it is | bpb | dC vs dense | mult |
|---|---|---|---|---|
| G1 ffn_measured | reallocate FFN width by the d22 profile, FLOP-neutral | 0.9592 | 0% | 1.019 |
| G2 ffn_shrink | shrink only the over-provisioned layers | 0.9692 | -13% | 0.981 |
| H1 ffn_native | same as G1 with the *correct* depth-native profile | 0.9617 | 0% | 0.973 |
| H3 cond_early_half | conditioning on attention, first half of layers | 0.9539 | +20% | 0.938 |
| G5 ffn_every2_iso | same params, FFN in half the layers at 2x width | 0.9645 | 0% | 0.926 |
| H4 cond_qo_only | conditioning on c_q and c_proj only | 0.9547 | +20% | 0.925 |
| E4 cond_tied_attn | conditioning on all attention projections | 0.9458 | +41% | 0.923 |
| H2 cond_skip_v | conditioning on q, k, proj | 0.9519 | +30% | 0.895 |
| D1 cond_tied | conditioning everywhere, R=256 | 0.9337 | +102% | 0.807 |
| G7 ffn_every4_iso | same params, FFN in a quarter of layers at 4x width | 0.9771 | 0% | 0.738 |
| G3 ffn_last_only | no FFN except the last layer | 1.0443 | -55% | 0.513 |
| **F1 cond_attn_R192 (d12)** | | 0.8583 | +26% | **0.834** |
| **D1 cond_tied (d12)** | | 0.8464 | +86% | **0.720** |

## What we found

**1. Conditioning works. It just does not pay.** D1 at d8 reached 0.9337 against
a dense 0.9602. That 0.027 BPB is one of the largest single-change improvements
in this codebase. It cost 102 percent more compute, and 102 percent more compute
spent on plain dense reaches 0.922. The quality is real and the price is wrong.

**2. The price gets worse with scale, not better.** The same arm goes 0.807 at
d8 and 0.720 at d12. This was the finding that ruled out waiting for scale to
rescue the idea. It also matches the earlier RemixedLinear result, where a K=1
variant beat dense at d4 by 0.002 BPB and lost at d8 by 0.014.

**3. You cannot fund conditioning by shrinking the FFN.** Every arm that removed
FFN compute lost more than it saved (G3 at 0.513, G6 at 0.668). Holding
parameters fixed and concentrating the same FFN width into fewer blocks is
monotonically worse (G5 0.926, G7 0.738). And reallocating FFN width by measured
per-layer demand is a wash in both directions: G1 used a profile that was
anti-correlated with actual d8 demand and scored 1.019, H1 used the correctly
measured profile and scored 0.973. The correct profile did no better than the
wrong one, and both are inside the noise floor. **The per-layer demand
measurement does not predict allocation.** That was the centerpiece of the
previous plan and it failed its first real test.

**4. Template diversity was never the binding constraint.** The collapse was
diagnosed to three mechanisms: softmax gradient starvation, where
`dL/dz_k = alpha_k (g_k - sum_j alpha_j g_j)` makes a losing template's gradient
proportional to how much it is already used; the operator-norm trap, where a
uniform mixture of K near-orthogonal unit templates has Frobenius norm
`1/sqrt(K)`, so the only route to full operator scale on a simplex is to
concentrate the router or align the bank, both of which were measured; and
shared-base redundancy. An explicit diversity penalty did nothing (1.084 with and
without). The strongest of the three fixes is to give the templates a shared base
and independent low-rank deltas, which removes the norm trap entirely because the
shared part never cancels. That construction is exactly ConditionedLinear. It was
built, it scores better than the template bank, and it still loses. Fixing the
other two mechanisms raises a ceiling that ConditionedLinear already sits above.

## The pivot

One number does not move.

`reach%` is the fraction of the measured per-token demand that a linear map from
the layer's input can predict. Across four trained checkpoints at d8, d12, d20
and d22, at every layer and every projection, it comes back between 44 and 52
percent. At d22 the FFN aggregate is demand 307.2, supply 126.7, and reachable
demand 51.3 percent of measured.

Three results line up behind reading that as a router limit rather than an
operator limit:

- **Router rank is not the cap.** Dropping the router to rank 64 was slightly
  compute-*positive* against the full-rank router (1.039x). If the selector were
  starved for capacity, shrinking it would have hurt.
- **More operator moves along the curve, not across it.** The rank sweep at d4
  (R = 64, 128, 256, 512 giving 1.1307, 1.1127, 1.0878, 1.0539) improves
  monotonically with a local exponent between -0.051 and -0.068, against dense's
  -0.057. It is a curve parallel to dense with a worse offset. Scaling R never
  closes the gap because it is not the gap.
- **Targeting helps by deleting operator, not by adding it.** H3 and H4 beat E4
  while spending less, by removing conditioning from projections the router was
  not usefully selecting over.

So the reframing:

> **Addressability** is the fraction of the token-varying weight-gradient
> dispersion at a layer that a router of a given cost can predict from that
> layer's input. Capacity is what the operator could express. Addressability is
> what the selector can reach. The measurements say the models have ample
> capacity, roughly matched supply, and an addressability ceiling that no arm has
> ever tried to move.

This also reinterprets the collapse results without contradicting them. Templates
collapse in attention because attention is already a data-dependent operator, so
a second selector on top of it addresses little that the first one does not. FFN
templates diversify because the FFN's own activation mask is a different kind of
selector. Neither observation is about how many operators the layer can hold.

## The decisive test, and it costs no training FLOPs

`reach%` was always a linear fit, and `predictable_demand()`'s own docstring
called that a lower bound. `reach_by_router()` in
`scripts/conditioning_headroom.py` closes it. On one train split and one held out
split it fits three predictors:

```
lin_full   ridge, full rank, closed form            the existing measurement
lin_r      x -> Linear(d_in, r) -> Linear(r, d_out)
mlp_r      x -> Linear(d_in, r) -> GELU -> Linear(r, d_out)
```

`lin_r` and `mlp_r` have identical parameters and identical FLOPs, so
`mlp_r - lin_r` is the value of the nonlinearity at fixed router cost, priced
before a single training run is spent on one. The default r=64 matches the
`--cond-router-rank 64` arm we already have a result for.

```
python -m scripts.conditioning_headroom --checkpoint DIR \
    --tokens 49152 --gram-tokens 4096 \
    --nonlinear-reach --reach-rank 64 --json out/reach.json
```

`--tokens` and `--gram-tokens` are separate on purpose. The Gram spectra are
N x N eigendecompositions that converge at a few thousand tokens and are what
`headroom_results.log` already used; at N=32768 that matrix is 8.6 GB in float64
and cusolver's workspace query fails. Only the router fit needs the large sample.

Two methodological points that the implementation had to get right, both found by
testing the estimator against synthetic targets with known answers:

- The score is `1 - tr(cg(A - Ahat)) / tr(cg(A - abar))` against a **static**
  baseline, not the legacy `tr(G_pred)/tr(G_full)`. A trained net is not an
  orthogonal projection, so the energy form rewards it for emitting large
  predictions that explain nothing. And without the `abar` baseline, a predictor
  that had learned only the training mean scored +0.27 on the permutation
  control, because the predicted gradient `c x_t^T` varies with `x_t` even when
  `c` does not.
- **Projections do not all resolve at once.** `d_in` varies by 4x across
  projection types, because `mlp.c_proj` reads the FFN hidden width rather than
  the model dimension. At any given `--tokens` the narrow projections resolve and
  that one does not, so resolution is tested per projection using its own
  permutation control, the gain is averaged over resolved projections only, and
  unresolved rows are printed rather than silently folded in.
- **The measurement needs about 32x d_in tokens.** On a synthetic target with a
  known nonlinear component, the recovered gain is +0.25 at N = 16384 and stable
  through N = 65536, but at N = 4096 it comes back at **-0.29**, the wrong sign.
  The default `--tokens 4096` would have produced a confident false negative. The
  permutation control is the power indicator: it goes -0.42, -0.16, -0.07, -0.03,
  -0.01 as N grows, and the script refuses to issue a verdict until it is near
  zero.

### Pre-registered outcomes

**Gain above 0.10.** The ceiling was the router's linearity. The next arm is a
rank-64 nonlinear router on the two highest-demand projections at d8. A rank-64
two-layer router costs roughly 5 percent overhead against the 20 percent that H3
and H4 paid, and it needs only their quality gain to clear 1.0x. Concretely,
1.123 / 1.05 is about 1.07.

**Gain between 0.03 and 0.10.** Real but not sufficient alone. Worth one arm only
stacked with the targeting that H3 and H4 already found.

**Gain below 0.03.** The unreachable half of the demand is not a function of the
current token at all. No router of any design can address it, per-token operator
conditioning cannot pay at any price, and this direction ends. That is a real
finding and it is worth writing down, because it is a general statement about
conditional linear layers rather than a statement about our particular ones.

Note that this last branch is a genuine possibility and the direction is staked
on an unrun measurement. That is the point: it is one measurement on checkpoints
that already exist, it costs no training compute, and it resolves in hours rather
than in another round of d12 sweeps.

## Open questions the framing raises

1. **Is the demand addressable from a different input?** Every router here reads
   the layer's own input `x`. The unreachable half might be predictable from the
   residual stream at a different depth, from the attention output, or from a
   short window of previous tokens. Addressability is a property of the router's
   *information*, not only of its function class, and we have only ever varied
   the function class.
2. **Does addressability have a scaling law?** If `reach%` is flat in depth, as
   the four checkpoints suggest, then the addressable fraction is a constant of
   the architecture and the returns to conditioning are bounded independently of
   model size. That would be a clean and citable negative law.
3. **Where is the demand nonlinear?** The per-projection breakdown of
   `mlp_r - lin_r` says which sites would benefit from a nonlinear selector.
   Unlike the per-layer width profile, which failed to predict anything, this is a
   prediction about the router and it has never been tested.

## Status

- `reach_by_router()` implemented in `scripts/conditioning_headroom.py`, behind
  `--nonlinear-reach`, with the holdout, the permutation control, the
  convergence diagnostic and the underpower guard.
- Validated on synthetic targets: recovers +0.57 gain on a purely nonlinear
  target, zero on a linear one, zero on noise, and is immune to the prediction
  magnitude trap.
- Not yet run on a real checkpoint. That is the next action, and it decides
  whether the rest of this document is a research direction or a negative result.
