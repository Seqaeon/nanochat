# EET — Early Exit Transformer

## Project Summary

EET is a Mixture-of-Depths (MoD) style early exit architecture for autoregressive transformers. A **global router** examines all tokens upfront and assigns each an exit depth via a fixed-capacity bell-curve schedule. Tokens physically leave the computation at their assigned layer via gather/scatter, so later layers process fewer tokens for real FLOP savings. All shapes are static Python ints (no `.item()` calls), making the architecture fully compatible with `torch.compile`. At d8/768-dim, EET achieves ~25% wallclock speedup over dense training with a ~0.06 val_bpb quality gap.

---

## 🚫 Catalog of Unsuccessful Ideas (Do Not Retry)

### 1. Phase 1 Warmup & Phase 2 Exploration (Reconstruction Loss)
- **Concept:** Warm up the backbone dense, then train with a reconstruction loss where a translator predicts final layer states from early exit states.
- **Outcome:** Made performance worse and did not improve routing compared to training Phase 3 (committed routing) directly from scratch.
- **Status:** **Abandoned**. We now run Phase 3 only (`--eet-warmup-frac 0.0 --eet-explore-frac 0.0`).

### 2. Exit Adapters (`--eet-exit-adapter-rank > 0`)
- **Concept:** Low-rank adapters (rank 16–32) to map early-exit hidden representations to the shared LM head subspace.
- **Outcome:** Slowed the model down significantly and resulted in worse val_bpb performance. The representation gap between intermediate and final layers is too large for a low-rank linear projection to bridge.
- **Status:** **Abandoned** (set to `0`).

### 3. Router After Block 1 (`--eet-router-after-block 1`)
- **Concept:** Let the global router see post-attention/FFN context from block 1 instead of raw embeddings (x₀).
- **Outcome:** Did not improve performance.
- **Status:** **Abandoned** (set to `0`).

### 4. Layer-Weighted Losses
- **Concept:** Weigh the per-token CE loss by the exit depth of each token (linear, EMA, sqrt strategies).
- **Outcome:** Did not improve performance.
- **Status:** **Abandoned**.

### 5. FFN-Skip / A³D Mode (`--eet-ffn-skip 1`)
- **Concept:** Skip FFNs while retaining full attention across all tokens at all layers.
- **Outcome:** Increased step time (dt) by ~150ms and performed significantly worse (val_bpb of ~1.40 vs ~1.03 for full block skip).
- **Status:** **Abandoned** (set to `0`).

### 6. Depth-Scaled Learning Rate (`--eet-depth-lr-scale 1`)
- **Concept:** Scale per-layer Muon LR by the inverse of the surviving token fraction (e.g., layer 7 at 10% surviving → 10× LR). Motivated by diagnostic finding that deep layers are gradient-starved.
- **Outcome:** Made things **worse** (final loss 3.529 vs 3.477 baseline). High LR on small-batch gradients from few final-layer tokens caused noisy, destabilizing updates.
- **Status:** **Abandoned**.

### 7. Depth Gradient Scaling (`--eet-depth-grad-scale 1`)
- **Concept:** Scale per-token CE loss by inverse of active fraction at the token's exit layer (from capacity schedule), so deeper exits get amplified gradient.
- **Outcome:** **Zero measurable effect** (final loss 3.477 — identical to baseline). The scaling amplifies gradients that already exist but doesn't create new gradient paths to deeper layers for tokens that exited early.
- **Status:** **Abandoned**.

### 8. Detach Auxiliary Losses from Backbone (`--eet-detach-aux-from-backbone 1`)
- **Concept:** Detach `stacked`, `routing_weights`, and hidden states before passing to CE-guided and surprise aux losses, so only the router (not the backbone) gets gradient from aux losses.
- **Outcome:** **Zero measurable effect** (final loss 3.477 — identical to baseline). The main CE loss (not aux losses) is the dominant gradient source; detaching aux losses changes essentially nothing.
- **Status:** **Abandoned**.

### 9. Detach Exit Representations from Backbone (`--eet-detach-exit-from-backbone 1`)
- **Concept:** Detach exiting token representations before scattering into x_final, so the backbone only receives gradient from final-layer tokens. Intended to prevent conflicting gradient objectives (early layers pulled toward "prediction-ready" and "good intermediate features" simultaneously).
- **Outcome:** Made things **significantly worse**. The backbone is starved of training signal:
  - 10% active: 1.277 val_bpb (vs ~1.09 baseline)
  - 30% active: 1.200 val_bpb
  - 50% active: 1.153 val_bpb
- **Conclusion:** The backbone needs gradient from ALL tokens (including early-exit tokens) to train properly. Data starvation is far more damaging than any gradient conflict.
- **Status:** **Abandoned**.

### 10. Dense Distillation (`--eet-dense-distill-interval > 0`)
- **Concept:** Periodically run a dense forward pass and use the logits as a KL distillation target for the EET model.
- **Outcome:** Did not improve performance. The dense forward uses the same (already-degraded) backbone, so it's distilling from a degraded teacher into a degraded student.
- **Status:** **Abandoned**.

### 11. Capacity Annealing (`--eet-capacity-anneal-frac > 0`)
- **Concept:** Start with 50% active tokens and gradually anneal down to the target (e.g., 10%) over training. Progressive introduction of exits.
- **Outcome:** Did not improve the quality gap.
- **Status:** **Abandoned**.

### 12. Depth Affine / Per-Exit γ,β (`--eet-depth-affine 1`)
- **Concept:** Learned per-exit-depth scale (γ) and shift (β) applied to hidden states before the LM head. Cheap alignment of different-depth representations.
- **Outcome:** Did not improve performance.
- **Status:** **Abandoned**.

### 13. Reentry at Final Layer (`--eet-reenter-final 1`)
- **Concept:** Restore all exited tokens at the final layer so every token goes through the last transformer block.
- **Outcome:** Did not improve performance.
- **Status:** **Abandoned**.

---

## 🚀 Active Configurations & Hyperparameter Tuning

### 1. Global Router & CE-Guided Loss (Phase 3 Only)
- **Flags:** `--use-eet 1 --eet-global-router 1 --eet-loss-variant ce_guided --eet-compute-skip 1`
- **Schedule:** `--eet-capacity-schedule bell`
- **Warmup/Explore:** `--eet-warmup-frac 0.0 --eet-explore-frac 0.0`
- **Current Baseline Gap (d8):** ~0.06 val_bpb gap from dense baseline.
- **Speed:** ~25% wallclock speedup over dense at d8 (when compile-breaking features are disabled).

### 2. Router Learning Rate Tuning (`--eet-router-lr-mult`)
- **Concept:** Gating/routing networks often need a decoupled learning rate from the rest of the network to break the constant-function equilibrium.
- **Implementation:** Decoupled and independent from backbone LR. Set via `--eet-router-lr-mult <float>` (defaults to `5.0` relative to `gate_lr`).

### 3. EET Backbone Learning Rate Tuning (`--eet-model-lr-mult`)
- **Concept:** When capacity is restricted (e.g. 10% active tokens at deep layers), the optimal backbone learning rate might differ from the dense baseline's optimal LR.
- **Implementation:** Scales all non-router LRs (Muon + AdamW parameters) when EET is active. Set via `--eet-model-lr-mult <float>` (defaults to `1.0`).

---

## 📊 Key Diagnostic Findings

- **Gradient imbalance:** Early layers receive 3–5× more gradient than a dense model; deep layers (especially layer 7) receive only ~13% of normal gradient. This is a symptom, not the cause — fixing it doesn't close the gap.
- **Backbone divergence:** Weight cosine similarity between Dense and EET backbones is nearly zero (0.01–0.09). The EET backbone learns fundamentally different representations.
- **Training gap vs Routing gap:** 78% of the quality gap stems from backbone co-training degradation, not inference-time routing decisions. However, attempts to fix the training gap (detaching, scaling, adapters) all failed.
- **The gap appears architectural:** The ~0.06 bpb cost is the inherent price of routing 90% of tokens to early exits with a shared LM head. At d24 scale, the gap persists.

---

## 🧪 P02 — The Three Decision Tests (pre-registered)

The "Key Diagnostic Findings" section above concludes that the remaining ~0.06 bpb is
architectural. That conclusion is not yet supported: all thirteen abandoned ideas targeted
gradient flow, learning rates, representation alignment, distillation or scheduling, and
none targeted the two things that actually differ from a dense model.

### Break-even, stated first

On the repo's iso-data dense curve (`mst_isodata.html`, d10–d16), the local log-log slope
near d8 is about **-0.085**. The LM head is roughly **29% of active FLOPs per token** at
d8/512/V=32k and routing does not shrink it, so the bell schedule's 39% saving on the
blocks is only a **~31% saving overall**.

| what EET spends | allowed bpb gap |
|---|---|
| 0.69× dense active FLOPs | **+0.034** |
| 0.75× dense wallclock    | **+0.027** |

EET is at +0.06, so it is 2.2× over budget and must give back **0.026–0.033 bpb**.
`scripts/eet_p02_report.py` recomputes this from the sweep's own dense controls rather
than from a borrowed exponent.

### The two untested defects

**Defect 1 — context destruction.** In the `compute_skip` path an exited token loses its
keys and values in every later layer, so at d8 layers 5–7 attend over 25%, 12% and 10% of
the sequence. The survivors are by construction the hard tokens, and they are denied most
of their context. MoD avoids this by interleaving full-capacity blocks; EET has no full
block after layer 1. `early_exit_architecture_idea.md` called Option B (frozen KV) "the one
worth pursuing", but the fast path shipped Option A because Option B was not
`torch.compile`-static. Restoring full-context reads costs about **1.6% of a dense layer**.

**Defect 2 — data starvation.** `use_pos_embed` is off, so the global router's input is
`norm(wte(idx))` and exit depth is a per-vocabulary-item lookup table. Layer 7 therefore
trains on a fixed ~10% slice of the vocabulary for the whole run and never sees `the`. This
explains the measured "78% of the gap is backbone co-training", and it explains why
depth-LR-scale (#6) made things worse and depth-grad-scale (#7) did nothing: a layer short
of *samples* cannot be fixed with a learning rate.

### Mechanisms added

| flag | values | what it does |
|---|---|---|
| `--eet-kv-mode` | `none` / `fresh` / `stale` | `fresh` re-projects keys and values at every layer for every position (quality upper bound, +12% FLOPs). `stale` reuses the keys and values banked at each token's exit layer (near free). `none` is the historical behaviour and is bit-identical to it. |
| `--eet-route-noise` | float | Gumbel noise on the exit score before top-K, **training only**. Capacities and therefore the FLOP budget are unchanged; only the assignment is resampled, so every token id visits every depth over training. 0 = deterministic, ~0.3 mild, ~1.0 strong, ≥10 effectively uniform-random. |
| `--eet-route-noise-end` | float | linear anneal of the above (`<0` holds it constant). |
| `--eet-coverage-diag` | 0/1 | per-layer vocabulary and token-mass coverage. Breaks the compile graph, so use it on short diagnostic runs with `--compile 0`. |

### Pre-registered criteria (do not edit after seeing results)

| test | criterion |
|---|---|
| T0A gate | `delta_bpb(freq/ctx)` on the dense checkpoint ≥ **0.020**, else Defect 1 is closed and the T1 arms are skipped automatically. |
| T0B gate | token mass reaching the last layer ≤ **0.50** under deterministic routing, else Defect 2 is closed and the T2 arms are skipped. |
| T1 pass | best `--eet-kv-mode` arm reaches gap ≤ **0.045** bpb. |
| T2 pass | best `--eet-route-noise` arm reaches gap ≤ **0.045** bpb. |
| T3 pass | combined arm reaches gap ≤ break-even wallclock (≈0.027 at d8), at two depths, with the gap not widening from d8 to d16. |

**If T1 and T2 both fail, the "architectural" verdict is confirmed. Close the direction
rather than sweeping more flags.**

### Files

- `scripts/eet_context_oracle.py` — T0A. Imposes EET's key masking on a trained *dense*
  checkpoint with depth held constant, so it separates the cost of losing context from the
  cost of losing depth without any training. Ranks tokens by frequency (the router proxy),
  at random (control) and by the model's own CE (best-case router).
- `scripts/eet_p02_tests.sh` — the sweep. Runs the dense control first, then enforces both
  gates automatically before spending GPU time on the arms behind them. Every arm after
  `DENSE` is pinned to the dense run's exact token count, so the whole sweep is iso-data.
  It also runs the **iso-FLOP iso-data dense controls at d5 and d6**, which no earlier EET
  sweep had and which the Pareto claim cannot be made without.
- `scripts/eet_p02_report.py` — break-even arithmetic and pass/fail. Charges the LM head at
  full price to every arm and fits the dense exponent from the sweep's own controls.
- `tests/test_eet_p02.py` — proves the mechanisms do what they claim, including that
  `kv_mode none` is bit-identical to the old path and that route noise leaves capacities
  untouched.

### Novelty note

Closing the gap is necessary but not sufficient. "Matches dense, 25% faster" is not a
main-track result in 2026: **Mixture-of-Recursions** (NeurIPS 2025) already owns learned
per-token depth with capacity routing and explicit KV strategies for tokens that stop
early, and **N-vium** (2026) reports 57.9% wallclock speedup at 1.5B with no perplexity
cost. If P02 succeeds, the paper framing has to change with it: the defensible primitive is
**decoupling write-depth from read-depth** (how deep a token is refined versus how deep it
stays readable by others), which no published work routes as two separate budgets.
MoD is symmetric skip, MoR is symmetric recursion, and CALM/SkipDecode patch the KV as an
implementation detail rather than as a routed resource.
