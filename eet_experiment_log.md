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

### P02 run gotchas (cost real GPU time, both now guarded)

**RESOLVED: `tokenizer/` stays tracked and is now pinned.** `tokenizer/PINNED.json` records
the V=32768 tokenizer's SHA-256s and `tests/test_tokenizer_pin.py` fails if the tracked
files stop matching it. Tracking is deliberate: untracking makes `git pull` DELETE the
tokenizer on every box whose copy is clean, and refuse the pull outright on a box whose
copy differs. Tracked and pinned, a pull repairs a box sitting on a bad copy. The
paragraph below is the history that motivated the pin.

**The repo's `./tokenizer` was a 265-token stub.** `tokenizer.pkl` is 1.9 KB and
`get_tokenizer('tokenizer')` returns `vocab_size=265`. base_train prints
`Vocab size: 265`, pads it to 320, and trains to completion without complaint. The first
P02 attempt trained `DENSE_D8` and both iso-FLOP dense controls that way; their bpb
numbers are not comparable with anything trained at V=32768 and had to be discarded. The
real tokenizer is at `~/.cache/nanochat/tokenizer` (411 KB, V=32768); `tokenizer_131k/` is
the V=131072 one. `scripts/eet_p02_tests.sh` now aborts before the first run if the
resolved vocabulary is under 1000, and `scripts/eet_p02_report.py` refuses to print a
table if any arm trained at a stub vocabulary or if arms disagree.

**`GPT.estimate_flops` does not count a tied LM head.** It subtracts `wte` from the
parameter count, so when `wte` and `lm_head` share a tensor the head's `6*d*V` disappears
from the printed active-FLOPs figure entirely. At d8/512/V=32768 that is 1.007e8 per
token, **32% of the honest total**, and routing never reduces it because every token is
still predicted. A Pareto claim computed on the printed axis therefore overstates the
saving of any token-routing method. The P02 report prices the head back in and prints
both axes; the same correction applies to any MST or Remix curve plotted against active
FLOPs per token.

### Initialization bugs found and fixed during P02 (all pre-existing)

All three were latent in the `meta` -> `to_empty()` -> `init_weights()` path base_train
uses, and all three were allocator-dependent: NaN storage crashed, zero storage passed
silently while being wrong.

1. **The uninitialized-tensor tripwire crashed on its own findings.** `init_weights`
   called `get_submodule(name.rsplit('.', 1)[0])`, which for a dot-less name like
   `exit_freq_ema` hands `get_submodule` the tensor's own name and raises
   "`exit_freq_ema` is not an nn.Module". It could only ever report tensors living inside
   a submodule. This killed a training run, and was also the cause of the pre-existing
   `tests/test_eet_losses.py::test_eet_global_router` failure.
2. **`EarlyExitGPT`'s own buffers were never re-initialized.** `exit_freq_ema`,
   `vocab_route_ema`, `token_ce_sum`, `token_ce_count`, `token_difficulty` and
   `eet_phase_tracker` get their values in `__init__`, which `to_empty()` discards, and
   nothing wrote them again. With zero storage, `--eet-depth-weight-type ema` computed
   `1/clamp(0, min=0.05) = 20` for every exit and normalised to exactly 1.0, so the EMA
   depth weighting was a silent no-op in every run that used it.
3. **Only the last linear of each router was initialized.** `GPT.init_weights` does not
   reach `eet_routers`, and the EET override set just the output layer, so an `mlp1` or
   `mlp2` router's hidden layer held whatever `to_empty()` returned. All-zero means the
   router emits an identical score for every token and the exit assignment at
   initialization is decided by sort tie-breaking rather than by content. `mlp1` is the
   router in the P01 configuration that produced the 0.06 gap.

`init_weights` now runs its verification pass on the FINAL state (a subclass initializes
after `super().init_weights(verify=False)` returns), so the banner no longer fires on
tensors that are about to be set correctly. `tests/test_eet_p02.py` poisons the storage
with NaN before `init_weights` so these tests pin the worst case instead of sampling the
allocator.

### More P02 run gotchas

**The sweep log is append-only, so the report must read each arm's LAST section.** It read
the first, so an arm rerun against the real tokenizer was still judged on the original
run's `vocab_size`, `val_bpb` and `dt`. That is why the stub abort kept firing after
everything had been retrained. Fixed; the abort now names exactly which arms are stale and
which to rerun.

**Reruns without `--force`.** `--force` wipes every completed arm. `--redo <TAG>` clears
one arm from the state file, and `--redo-oracle` deletes the oracle result so it is
recomputed. The sweep prints the resolved oracle path on startup, because the gate is a
file check (`out/eet_p02/oracle_d<D>.json`) and deleting a copy at the repo root does
nothing.

**The oracle's frequency proxy needs a matching `freq_table.pt`.** `FrequencyPrior` loads
it from the tokenizer directory, and a table left over from a different vocabulary changes
which tokens the oracle masks without failing. The sweep now aborts if its entry count
does not equal the tokenizer's vocabulary size.

**`torch.compile` stride guard on the split-KV mask.** T3 died in the BACKWARD with

    assert_size_stride(constant_pad_nd, (64, 1, s0, 2056), (3691776, 3691776, 2112, 1))
    AssertionError: expected size 64==64, stride 430848==3691776 at dim=0

430848 = 204*2112 and 3691776 = 1748*2112: two different layers' capacities sharing one
compiled backward. `forward_split` was reading the query count off `x_q.size(1)`, which
torch.compile can carry symbolically, and the capacities are `int(survivor * T)` so a
symbolic sequence length makes every per-layer count symbolic too. Fixed by passing the
capacity in as a plain `int` (`n_q=K_cur`), marking the sequence dim static, and handing
SDPA a contiguous mask. `--eet-kv-eager 1` runs that attention outside the compiled graph
if it ever recurs: slower, so never use it for a wallclock claim, but the T1 quality gate
still returns a bpb.

---

## 🔴 P02 RESULT: both hypotheses failed. EET closed.

All EET arms at 265,814,016 tokens, d8/512, V=32768, `target_active_frac=0.10`, bell schedule.

| arm | val bpb | vs EET base | dt (ms) | vs dense dt | MFU |
|---|---|---|---|---|---|
| EET base (kv none, deterministic) | 1.06433 | — | 227.6 | **1.12x slower** | 33.4 |
| T1 fresh (per-layer KV restored) | 1.05617 | **-0.00816** | 289.6 | 1.42x slower | 26.2 |
| T1 stale (banked exit KV) | 1.06448 | +0.00015 | 283.0 | 1.39x slower | 26.9 |
| T2 random (uniform routing) | 1.06487 | +0.00054 | 244.0 | 1.20x slower | 31.1 |
| T2 anneal (1.0 -> 0.0) | 1.06258 | -0.00175 | 437.6 | 2.15x slower | 17.4 |
| dense d5 control | 1.04243 | — | 174.2 | | 23.0 |
| dense d6 control | 1.03339 | — | 176.5 | | 24.5 |

**The dense control, measured.** `DENSE_D8` first ran at 440,401,920 tokens rather than
265,814,016, because the state file still held the budget measured during the V=265 stub
runs. Rerun at the matched budget it scores **0.991231**. (A two-point extrapolation from
d5 and d6 had predicted 0.968; the measured number is 0.023 worse, so the gaps below are
smaller than first reported and the allowed gap is smaller too. The measured value stands.)

Iso-data dense curve, three measured points at 265,814,016 tokens:

| depth | active FLOPs/token | bpb |
|---|---|---|
| 5 | 1.509967e8 | 1.042433 |
| 6 | 1.627932e8 | 1.033392 |
| 8 | 2.862643e8 | **0.991231** |

Local log-log slope d6 to d8 is **-0.0738** (three-point fit -0.0769). EET's FLOP ratio with
the head priced in is **0.722**, so the allowed gap is **+0.024**.

| arm | bpb | gap vs dense | over budget |
|---|---|---|---|
| EET base | 1.06433 | +0.0731 | 3.03x |
| T1 fresh | 1.05617 | **+0.0649** | 2.70x |
| T1 stale | 1.06448 | +0.0732 | 3.04x |
| T2 random | 1.06487 | +0.0736 | 3.06x |
| T2 anneal | 1.06258 | +0.0713 | 2.96x |

Every arm fails the pre-registered 0.045 threshold, and there is no wallclock budget at all
because EET is slower than dense here.

### Verdict against the pre-registered criteria

| test | threshold | result |
|---|---|---|
| T1 (context restoration) | gap <= 0.045 | best arm +0.065. **FAIL** |
| T2 (routing coverage) | gap <= 0.045 | best arm +0.071. **FAIL** |

Pre-registered consequence, written before the runs: *if T1 and T2 both fail, the
"architectural" verdict is confirmed; close the direction rather than sweeping more flags.*
**Both failed. EET is closed.**

### What the runs actually established

**Defect 2 (data starvation) was wrong, and the coverage diagnostic falsified it directly.**
The prediction was that a router reading only `norm(wte(idx))` makes exit depth a
per-vocabulary lookup, so deep layers only ever train on a fixed slice of the vocabulary.
Measured coverage at layer 7:

    L0:100.0%/100.0%  L1:100.0%/100.0%  L2:100.0%/97.9%  L3:100.0%/85.4%
    L4:100.0%/55.0%   L5:99.9%/24.6%    L6:99.8%/12.1%   L7:99.7%/10.0%
    (vocabulary fraction / token-mass fraction)

Layer 7 sees **99.7% of the vocabulary**, 32,575 of 32,666 distinct ids. Only the token
*mass* is 10%. Routing is a within-sequence top-K, so a token that loses the competition in
one sequence wins it in another and every id reaches every depth. The T0B gate as
pre-registered watched token mass, which was the wrong quantity; vocabulary coverage was
the right one and it kills the hypothesis outright.

**The learned router is worth nothing.** T2 random (Gumbel noise 10.0, effectively uniform
routing) scores 1.06487 against the learned router's 1.06433: a difference of +0.0005.
Replacing the router with a coin flip costs nothing measurable. Whatever the global router
on `x0` is doing, it is not selecting better than chance, which retires the
"interpretable, prior-informed adaptive computation" framing entirely.

**Context restoration is real but tiny, and co-training absorbs almost all of it.** The T0A
oracle measured +0.234 bpb for context destruction on a model never trained to tolerate it.
Restoring fresh per-layer keys and values in training recovers **0.008**, about 8% of the
gap and about 3% of what the oracle predicted. Banked (stale) KV recovers nothing, so what
little there is comes from the keys being in the right per-layer subspace, not from their
presence. The write-depth / read-depth decoupling framing rests on this effect and it is
too small to carry a paper.

**There is no speedup to trade quality against.** EET is 1.12x *slower* than dense in
wallclock at d8/512, and MFU drops from 37.3 to 33.4. The historical "25% speedup" does not
reproduce at this configuration. The reason is visible in the FLOP split: at V=32768 the
LM head is **35.2%** of active FLOPs per token and routing never touches it, blocks are
52.7% and attention 12.1%. Routing the blocks down to a 0.606 average active fraction only
buys a **0.72x** overall FLOP ratio, worth **+0.037 bpb** on the measured dense curve, and
gather/scatter overhead plus the MFU loss eats even that.

### Do not retry

- Restoring exited tokens' keys and values, in any variant. Fresh per-layer KV is the
  upper bound and it is worth 0.008.
- Stochastic or exploratory routing schedules. Uniform-random routing already matches the
  learned router, so there is nothing for exploration to discover.
- Router architecture work on the `x0` global router. It performs at chance.


### Why there is no speedup at this configuration, and where it would come from

The routing path did run: the effective config records `use_eet: true`,
`eet_compute_skip: true`, `eet_global_router: true`, `eet_target_active_frac: 0.1`. The
problem is the FLOP split, on one H200 at d8/512/V=32768:

| component | share of active FLOPs/token | routed? |
|---|---|---|
| transformer blocks | 52.7% | yes, to 0.606 average active |
| **LM head** | **35.2%** | **no** |
| attention kernel | 12.1% | yes, as a^2 |

The best possible overall ratio is therefore **0.722**, a 28% FLOP cut, and gather/scatter
on (128, 2048, 512) tensors plus the aux-loss machinery eats it: EET base is **1.12x
slower** than dense in wallclock.

That share is strongly depth-dependent. At d24 (d_model 1536) the head falls to **6.4%**
and blocks rise to **86.9%**, and the same bell schedule gives a ratio of **0.586** rather
than 0.722. So EET's economics genuinely improve with depth. That is not a plan on its own:
the quality gap would have to close by roughly 3x at the same time, and nothing tried here
moved it by more than 8%.

Two per-step costs in the P01 configuration are worth separating from the architecture
before quoting any timing: `--eet-depth-weight-type ema` runs a Python loop over the exits
with scalar buffer writes inside the forward, and `--eet-loss-variant ce_guided
--eet-surprise-lambda 0.1` adds a top-k vocabulary entropy term. A timing probe with
`--eet-depth-weight-type none --eet-capacity-alignment-lambda 0 --eet-surprise-lambda 0`
isolates backbone routing cost from aux-loss cost.

**Fixed here:** `eet_route_noise`'s anneal recomputed a Python float from `eet_step` every
step and used it inside the compiled graph, so dynamo guarded on its value and recompiled
every step. That is why T2 anneal took 437ms against 244ms for the same arm at constant
noise. The scale is now a tensor.
