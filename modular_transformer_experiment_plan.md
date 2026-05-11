# Modular Sub-Transformer Architecture: Experimental Plan

## Overview

### The Core Idea

Standard transformers stack L layers, each consisting of a single attention block and FFN operating at a fixed embedding dimension D. Every layer is monolithic — a single set of weights processes the entire D-dimensional token representation. This architecture makes no structural distinction between different aspects of what a token might encode; all D dimensions are entangled and processed together at every layer.

The Modular Sub-Transformer (MST) architecture proposes a different structural prior: instead of one large transformer block per layer, each layer contains N smaller transformer blocks (sub-transformers), each operating at a reduced dimension d = D/N. Rather than one model processing everything, you have N parallel specialists — and the question the architecture is built around is whether those specialists can be made to meaningfully diverge, each learning to process a different aspect of the representation.

### What It Is Not

MST is related to but distinct from Mixture of Experts (MoE). In standard FFN-MoE (e.g. Switch Transformer), experts all receive the **same full D-dimensional token representation** and the routing selects which expert's FFN output to use. The experts differ in what transformation they apply, but they all see the same thing.

MST differs in a key structural way: sub-transformers operate on **lower-dimensional representations** (d-dim, not D-dim), which means they are forced to either process a partition or a projection of the full representation — not the whole thing. This is not just a routing choice; it is a decomposition of the representation space itself. The inductive bias is that different aspects of meaning can be processed more efficiently by independent pipelines than by a single entangled computation.

### What It Is Trying to Do

The architecture has three potential payoffs that are not mutually exclusive and that the experiments are designed to distinguish:

**1. Efficiency at iso-performance.** With N=8 sub-transformers at d=64, attention scales as d² not D², dramatically reducing per-layer parameter count and FLOPs. If sparse routing (top-k selection) can achieve performance comparable to a dense transformer of equivalent total parameters, this yields a meaningful inference efficiency gain. The goal is: fewer active parameters per forward pass, same downstream quality.

**2. Structured specialization.** Because each sub-transformer only ever processes a d-dim slice or projection of the representation, they are structurally encouraged to specialize in ways that FFN-MoE experts — which all see the same full representation — cannot. If the specialization is structured and interpretable (e.g. sub-transformers diverge by syntactic role, token frequency, or semantic category), this is both a scientific finding and a practical one for understanding what transformer layers compute.

**3. Modularity and composability.** If sub-transformers specialize reliably, the architecture opens the door to modular reuse — swapping or adding sub-transformers for new domains without disturbing existing ones. This is not the focus of the current experiments but is a downstream motivation for the design.

### Architecture Summary

The base configuration uses D=512, N=8, d=64 (so 8 sub-transformers per layer, each at dimension 64). The architecture has five independently variable design axes:

- **How sub-transformers receive input at layer 1** — via fixed embedding slices, learned projections, rotated slices, per-sub embedding tables, or a shared stem transformer
- **How sub-transformer outputs are routed and combined between layers** — soft weighted sum, hard top-k selection, sequence-level free-for-all path routing, or logit-level ensemble at the output only
- **What the FFN inside each sub-transformer outputs** — d-dim (standard) or 4d-dim (no downprojection, with the next layer's sub-transformers doing the projection back to d individually)
- **How information flows between layers** — aggregate-then-distribute with per-sub learned projections, or direct parallel columns with no inter-layer mixing
- **How the final layer produces vocabulary logits** — weighted sum of N independent logit heads, or a single shared projection from the aggregated hidden state

These axes interact in non-trivial ways and the experimental plan is designed to surface those interactions without running the full combinatorial grid.

---

## Design Space: All Axes


### Axis 1 — First-Layer Input (how sub-transformers receive the initial token embeddings)

| ID | Name | Description |
|---|---|---|
| I-A | Fixed Slice | Global embedding of dim D is sliced into N chunks of dim d. Sub-transformer i always receives dims [i*d : (i+1)*d]. |
| I-B | Learned Projection | Each sub-transformer i has its own linear projection W_i ∈ R^{D×d} that projects the full D-dim embedding down to d. Each sees the full embedding but through a learned lens. |
| I-C | Rotated Slice | Apply a fixed or learned rotation matrix R ∈ R^{DxD} to the embedding first, then slice as in I-A. The rotation mixes dimensions before partitioning so sub-transformers see structured combinations rather than raw dimension ranges. Fixed R = random orthogonal; Learned R = parameterized via Cayley or Householder. |
| I-D | Per-Sub Embedding Table | Each sub-transformer i has its own embedding table of size vocab_size × d. No shared embedding. |
| I-E | Stem Transformer | The embedding table emits d-dim vectors directly. One small transformer (single attention block) with dim d processes these raw d-dim embeddings and its output is passed identically to all N sub-transformers in layer 1. No differentiation at layer 1 input — all sub-transformers start from the same d-dim representation. |

**Notes:**
- I-A is the strictest inductive bias (true dimensional partition). Requires D-dim embedding table; each sub-transformer receives a non-overlapping slice.
- I-B and I-C are softer: sub-transformers can in principle access all information but are forced to compress. Also require D-dim embedding table.
- I-D removes the shared representation entirely — most parameter-expensive (N separate embedding tables).
- I-E uses a d-dim embedding table throughout. The stem is a shared pre-processing step; sub-transformers are forced to differentiate via their own weights from layer 1 onward, not from their input.

---

### Axis 2 — Routing Mechanism (how sub-transformer outputs are combined or selected)

| ID | Name | Description |
|---|---|---|
| R-A | Soft Weighted Sum | Router produces scores s ∈ R^N via softmax. Output = Σ s_i * f_i(x) where f_i is sub-transformer i's output. All sub-transformers active. |
| R-B | Top-K Hard | Router selects top-k sub-transformers. Only those k run (or their outputs are used). Others contribute zero. k is a hyperparameter. Train with hard top-k + auxiliary load balancing loss. |
| R-C | Free-For-All (Sequence-Level Path Routing) | Each sub-transformer i in layer l routes its entire sequence tensor (seq_len × d) to one sub-transformer in layer l+1 based on learned routing scores. The whole sequence travels together, preserving the causal mask. Creates paths like T2→T5→T4→T1 across layers at the sequence level. Multiple sub-transformers can route to the same destination — when this happens, their d-dim contributions are summed before the destination sub-transformer's attention. Final layer aggregates all sub-transformer outputs. |
| R-D | Logit-Level Ensemble | Sub-transformers are N independent transformers that each produce their own logits over vocab_size. Final prediction = weighted sum of logits (not hidden states). Weights can be learned scalars, softmax scores, or uniform. |

**Notes:**
- R-A, R-B, R-C operate on hidden states between layers
- R-D is architecturally different — sub-transformers run fully independently and are only combined at the output. This means no inter-sub-transformer information exchange at all between layers
- R-C uses sequence-level (not token-level) routing to preserve causal mask validity. The routing decision is per sub-transformer per layer, not per token. Router scores can be computed as a learned function of the mean-pooled sequence representation.
- R-C collision handling: when two sub-transformers route their (seq_len × d) tensors to the same destination, contributions are summed element-wise before the destination's attention — equivalent to overlapping residual contributions, no special handling needed.
- Top-k variants to test for R-B: k=1 of N=8, k=4 of N=8, k=16 of N=64

---

### Axis 3 — FFN Internal Transition (what the FFN inside each sub-transformer outputs)

| ID | Name | Description |
|---|---|---|
| F-A | d → 4d → d | Standard FFN shape but at sub-transformer dim. Output is d-dim. Ready for direct routing/aggregation. |
| F-B | d → 4d (no downproject) | FFN expands but does not compress back. Output is 4d-dim. Aggregation (weighted sum) happens in 4d space. Each next-layer sub-transformer then has its own projection 4d → d before its attention. |

**Notes:**
- F-B increases the aggregation dimensionality which may help preserve information across sub-transformers
- F-B only makes sense with R-A or R-B, not R-C or R-D
- F-B adds a per-layer projection parameter in each sub-transformer of the next layer

---

### Axis 4 — Layer-to-Layer Transition (how sub-transformer outputs flow between layers)

| ID | Name | Description |
|---|---|---|
| T-A | Aggregate then Distribute (Learned) | After routing (R-A or R-B), outputs are combined via weighted sum. With F-A (d→4d→d), the aggregate is d-dim. With F-B (d→4d), the aggregate is 4d-dim. Each sub-transformer i in the *next* layer has its own learned projection W_i ∈ R^{d×d} (F-A) or R^{4d×d} (F-B) back to d before its attention. The per-sub-transformer W_i is structurally necessary — without it, all sub-transformers in layer l+1 receive identical input and only diverge by initialization noise. W_i being learned and independent per sub-transformer is the symmetry-breaking mechanism. |
| T-B | Direct Pass (Parallel Columns) | Each sub-transformer i in layer l passes its output directly to sub-transformer i in layer l+1. No aggregation between layers. Routing only happens at the final layer. Simplest topology — sub-transformers are fully independent columns. |
| T-C | Free-For-All (see R-C) | No fixed topology. Routing is dynamic per sub-transformer per layer. The sequence-level routing in R-C defines this transition. |

**⚠️ T-A Symmetry Warning:** If all sub-transformers in layer l+1 receive the same aggregate (d-dim or 4d-dim) with no per-sub projection, differentiation relies entirely on weight initialization noise. This is fragile and likely leads to representational collapse over training. The individual W_i projections are not optional for T-A — they are the mechanism that makes each sub-transformer in layer l+1 see a genuinely different linear combination of the aggregated representation.

---

### Axis 5 — Final Layer → Vocabulary

| ID | Name | Description |
|---|---|---|
| V-A | Weighted Sum of Logits | Each final-layer sub-transformer produces logits of shape vocab_size via its own FFN head. Final logits = weighted sum (weights from router or learned scalars). Same as R-D but only at the last layer. |
| V-B | Aggregate then Project | Weighted sum or concatenation of final sub-transformer hidden states → single vector of dim D or d*k → one shared FFN head projects to vocab_size. |

---

## Clarification: What "Rotated Slicing" Means

A rotation matrix R ∈ R^{DxD} is applied to the full embedding e ∈ R^D before slicing:

```
e' = R @ e          # rotate
sub_i_input = e'[i*d : (i+1)*d]   # then slice
```

- **Fixed R**: sampled once (random orthogonal matrix, e.g. via QR decomposition) and frozen. Computationally free at train time (can be fused). Ensures sub-transformers see mixed combinations of all original dimensions.
- **Learned R**: parameterized rotation, trained jointly. More expressive but harder to keep orthogonal — requires constrained optimization (Cayley map or Householder reflections).

The motivation: fixed slicing may give sub-transformers access to semantically unrelated dimension groups by accident. A rotation redistributes information more evenly across all sub-transformer inputs.

---

## Experiment Structure

### The Core Problem: Combinatorial Explosion

With 5 input options × 4 routing options × 2 FFN options × 3 transition options × 2 final-layer options = 240 combinations. You cannot run all of them. But you also cannot run one-factor-at-a-time (OFAT) naively because some axes have strong interactions (e.g., I-A + R-C may behave very differently from I-B + R-C).

### Strategy: Staged Factorial Design

#### Stage 0 — Sanity Baseline (1 run)

Establish that the architecture can train at all and roughly what performance to expect.

**Config:** I-A + R-A + F-A + T-B + V-B
(Fixed slice | Soft weighted sum | Standard FFN | Parallel columns | Aggregate then project)

This is the simplest coherent configuration. Compare against:
- Iso-parameter dense transformer (match total param count, not n_embd)
- Iso-FLOP dense transformer (match active FLOPs per forward pass, not params)

**Metrics to track from day 1:**
- Validation loss / perplexity
- Router entropy per layer (is routing collapsing?)
- Sub-transformer load distribution (are all being used?)
- Training stability (loss spikes, gradient norms)

---

#### Stage 1 — Independent Axis Sweeps (N runs per axis)

Hold all other axes at the Stage 0 baseline config. Vary one axis at a time.

| Sweep | Configs to run | Purpose |
|---|---|---|
| Input sweep | I-A, I-B, I-C, I-D, I-E (vs baseline I-A) | Which input mechanism is best in isolation |
| Routing sweep | R-A, R-B (k=1), R-B (k=4), R-C, R-D (vs baseline R-A) | Which routing paradigm is best in isolation |
| FFN sweep | F-A, F-B (vs baseline F-A) | Whether expanding without downprojecting helps |
| Final layer sweep | V-A, V-B (vs baseline V-B) | Whether logit-level ensemble vs hidden-state aggregation matters |

**Total Stage 1 runs:** ~12-15 configs (some axes have fewer options)

**Decision rule:** Rank each axis option by validation loss. Keep top-2 per axis for Stage 2. Do not eliminate anything based on Stage 1 alone if the margin is under 0.5 perplexity points — interaction effects may reverse rankings.

---

#### Stage 2 — Targeted Interaction Grid (focused combinations)

Based on Stage 1 results, identify the top-2 options per axis and run a partial factorial over the axes most likely to interact. Strongly interacting pairs (run these regardless of Stage 1 rank):

| Pair | Why they interact |
|---|---|
| (Input axis) × (Routing axis) | How information is partitioned determines what routing can meaningfully select |
| (Routing axis) × (Final layer axis) | R-D makes V-A redundant; R-C changes what the final layer even looks like |
| (FFN axis) × (Transition axis) | F-B only makes sense with certain transition modes |

For each interacting pair, run the 2×2 or 2×3 grid of top options from Stage 1.

**Total Stage 2 runs:** ~20-30 configs

---

#### Stage 3 — Best Config Validation and Scaling

Take the top-3 overall configs from Stage 2. For each:
- Run at 2 model scales (e.g. d=64/N=8 and d=128/N=16)
- Run with multiple seeds (3 seeds minimum) to verify stability
- Run extended training to check if rankings are stable or reverse at longer horizons

This is where you produce the main results table.

---

#### Stage 4 — Analysis Experiments (not ablations, but insight runs)

These are not about finding the best config — they're about understanding what the architecture is doing. Run these on the best config from Stage 3.

| Experiment | What you learn |
|---|---|
| LogitLens per sub-transformer | Does each sub-transformer specialize in different token prediction patterns? |
| Router score visualization | Do routing weights cluster by token type, position, or semantic class? |
| Path analysis (R-C only) | What paths do different token types take through the layer graph? |
| Ablate individual sub-transformers at inference | Which sub-transformers are most critical? Is there redundancy? |
| Cross-layer routing entropy | Does specialization increase in later layers? |

These are your paper's analysis section and they justify *why* the architecture works, not just *that* it works.

---

## Tracking: How to Avoid Discarding Good Combinations Early

### Log everything, decide nothing early

For every run, log:
```
config_id, input_mode, routing_mode, ffn_mode, transition_mode, final_mode,
val_loss_1k_steps, val_loss_5k_steps, val_loss_final,
router_entropy_mean, router_entropy_min,
load_balance_score, grad_norm_mean, wall_time
```

Do not prune a config from consideration in Stage 2 based on val_loss_1k_steps alone — some routing mechanisms (especially R-C) may be slow to learn good routing but converge to better solutions.

### Use a results matrix, not a leaderboard

Maintain a table where rows = configs and columns = metrics. The goal is to understand the *structure* of the space, not to find one winner. Configs that underperform on val loss but have unusually high router entropy or interesting specialization patterns are still worth understanding.

### Stage 2 inclusion rule

A config is included in Stage 2 interaction experiments if:
- It is top-2 on val loss in its axis sweep, OR
- It shows qualitatively different routing behavior than the current top configs (even if slightly worse loss)

The second rule is important — R-C (free-for-all) may underperform in Stage 1 but the interaction with certain input modes may be what unlocks it.

---

## Compute Budget Estimate

All estimates assume a small-scale experiment setup (GPT-2 small equivalent, ~10M-50M params, trained on ~100M tokens).

| Stage | Runs | Est. GPU-hours each | Total |
|---|---|---|---|
| Stage 0 | 3 (+ 2 baselines) | 2h | ~10h |
| Stage 1 | ~15 | 2h | ~30h |
| Stage 2 | ~25 | 2h | ~50h |
| Stage 3 | ~9 (3 configs × 3 seeds) | 6h | ~54h |
| Stage 4 | 5-8 analysis runs | 2h | ~16h |
| **Total** | | | **~160h** |

On a single A100, this is roughly 1 week of continuous training. Parallelizable across multiple GPUs by running Stage 1 sweeps simultaneously.

---

## Resolved Design Decisions

These were open questions now settled — treat as fixed across all experiments unless explicitly ablating.

### 1. Positional Encoding (RoPE) — RESOLVED
The architecture uses RoPE, which is applied inside attention to Q and K projections, not to the input tensor. Each sub-transformer has its own Q, K projections and applies RoPE independently. Use the same RoPE rotation frequencies (same θ schedule) across all sub-transformers since they all process the same sequence positions. No separate decision needed.

### 2. Residual Connections — RESOLVED (follows transition type)
Residual connections are not a separate axis — they naturally follow the transition mode:

- **T-B (parallel columns):** Each sub-transformer has its own d-dim residual stream. `out_i = x_i + sub_i(x_i)`. Fully independent.
- **T-A / T-A2 (aggregate then distribute):** Two valid sub-options, ablate within T-A experiments:
  - **Per-sub residual (default):** Each sub-transformer adds its d-dim residual internally before aggregation. Residual lives at d-dim.
  - **Shared D-dim residual:** Aggregate all outputs to D-dim, add to a persistent D-dim residual stream, then redistribute. Closer to standard transformer residual stream but more expensive. Try as a secondary ablation if T-A shows promise.
- **R-C (free-for-all):** Each sub-transformer cell has its own d-dim residual. The residual follows the sequence along its path through the layer graph.

### 3. Layer Normalization — ABLATE (pre vs post)
Ablate pre-norm vs post-norm within Stage 1. Default to **pre-norm RMSNorm** — consistent with modern LLM stacks using RoPE (LLaMA, Mistral, etc.) and more training-stable than post-norm LayerNorm.

Placement rules:
- RMSNorm applies inside each sub-transformer at d-dim, before attention and before FFN (pre-norm)
- For T-A with shared residual variant: an additional RMSNorm applies to the aggregate (d-dim or 4d-dim) before the per-sub W_i projection
- For V-A (logit-level ensemble): RMSNorm applies inside each sub-transformer before its output projection

### 4. Causal Attention Mask in R-C — RESOLVED
Token-level routing in R-C breaks the causal mask (tokens from different positions may co-locate in the same sub-transformer cell). **Use sequence-level routing instead** (see R-C definition): the entire sequence tensor from sub-transformer i routes together to one sub-transformer in layer l+1. The causal mask applies normally within the destination since all tokens are co-located. This is the only implementation of R-C that is both tractable and mask-correct.

### 5. Load Balancing Loss — RESOLVED
Start with coefficient 0.01 (standard MoE default). Treat as fixed — do not tune per experiment. If routing collapse is observed consistently, increase to 0.02 as a diagnostic, but do not use this as an experimental variable.

**On quantile routing:** Deferred to Stage 4. If Stage 1/2 results show strong sensitivity to the choice of k in R-B, quantile routing (adaptive k based on score distribution percentile) is a motivated follow-up. Not a core ablation — adds dynamic-graph complexity that complicates batching and load balancing.

### 6. Weight Tying — DEFERRED
Hold off. Do not tie input embeddings and output projection weights for now. Revisit after Stage 2 if parameter budget becomes a concern.

---

## Suggested Paper Framing (Based on What the Experiments Could Show)

The strongest framing that distinguishes this from MoE:

> "Standard MoE applies expert routing to the FFN sublayer over full-dimensional token representations. We propose a fundamentally different decomposition: partitioning the representation space itself, such that each expert operates on an independent dimensional subspace. This induces a qualitatively different form of specialization — not over token types (as in FFN MoE), but over representational subspaces — and we show that this specialization is structured, measurable, and beneficial."

The analysis experiments in Stage 4 are what make this framing defensible. The architecture result alone is insufficient — the story is only complete if you can show *what* the subspaces specialize in.
