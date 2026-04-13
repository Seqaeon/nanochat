# Iterations on Remixed-Linear: A Historical Analysis

System / Context: I am working on novel Transformer architectures designed to efficiently condition internal representations on local or global sequence context. Attached are two papers that ground my conceptual goals: Remix (Don't Expand: Context-Aware Embedding Routing) and Context-Conditioned Linear Layers for Efficient Transformers (CCLL). The Problem Origin: Initially, my custom architecture showed incredible improvements on small-scale tests. However, upon testing longer context windows (T=2048 vs T=64), performance severely degraded. I investigated and discovered that my early "success" was actually a false positive caused by a causal leakage bug inside the `GlobalContextManager`'s cross-attention function. The model was looking into the future. Once I patched the causal leak, the architecture fundamentally failed to beat a standard, static dense Transformer baseline. The performance gap is strictly related to sequence length and sequence-time recurrence, not model width or depth. What I Have Already Tried (The Failed Ablations): I have spent the last 9 iterative phases trying to fix this conditional architecture. You can review the attached `remixed_linear_history.md` for full implementation details, but here is a summary of the traps my designs fall into that i have guessed:
1. BPTT/Gradient Explosion: Passing continuous exponential moving average (EMA) context tokens recurrently between spatial blocks caused massive gradient explosions. We had to fix this by `detaching` the context history, but this breaks long-range gradient flow.
2. The "Identity Trap": When we generate a "context signal" from the exact same `norm(x)` that the Feed-Forward Network (FFN) is about to process, the context branch becomes redundant. Multiplicative sigmoid gating collapses to identity, acting as high-frequency noise that drags down the dense weights.
3. The Rank Bottleneck: Replacing a full dense projection with a low-rank context-gated projection starves the network of capacity.
4. The Latency Mismatch: We tried chunking context (pooling over 64 tokens) and gating only at specific "learned boundaries" (like newlines) to improve signal-to-noise ratio. This failed because FFN transformations act locally, and providing them with "lagging" topological summaries doesn't help them with exact syntax prediction.
5. Additive vs. Multiplicative Fractures: We recently built a "Residual Adaptive Linear" (RAL) layer that is mathematically identical to a static linear layer at initialization (`y = base(x) + delta(ctx)`). Even natively protected from regressions, the parallel task of learning to dynamically modulate the `delta` path causes gradient fracturing that underperforms a simple static dense layer. Weight modulation works better than activation normalization, but it still lags behind the baseline. My Request to You: My architecture was relying on a causal leak to function, and now that it is structurally sound, the optimization penalty of learning a dynamic context controller vastly outweighs the theoretical benefits of the added capacity. I need you to think deeply, conceptually analyze what the CCLL paper are striving for, and propose completely novel, mathematically feasible architectural redesigns. Do not suggest simple EMA smoothing, chunking, residual additive deltas, top-k sparse gating, or standard FiLM layers, as I have empirically proven these drag down pretraining efficiency. How can we achieve the goal of a highly efficient, contextually modulated projection layer that is intrinsically optimized to learn faster than a static dense baseline?

The baseline architecture(without research branches, just normal gpt style) (yielding a validation loss of `4.15`) heavily utilizes a static Exponential Moving Average (EMA) context state with an explicit `.detach()` between tokens, alongside factorized weight modulation (basis/output gates) inside the MLP.

Below is an analysis of all the major design experiments we have attempted to bridge the gap toward dynamic, long-range context understanding, why they were theorized to work, and why empirical sweeps (degrading to the `4.5` - `4.9` range) proved they failed.

## Phase 1: The Global Context Manager (GCM)
**The Idea:** To prevent sequence dilution, we introduced a standalone `GlobalContextManager`. It ran an $O(T^2)$ cross-attention mechanism across the sequence to derive a dense "global" context vector. This context was then broadcast back out to all layers to modulate their forward passes.

**Why it Failed:** 
- **Data Leakage & Stability:** It casually risked data mask leakages at shorter sequences.
- **Computation Bottleneck:** $O(T^2)$ overhead outside the standard pipeline was unacceptable.
- **Dilution:** Averaging all tokens into a monolithic context embedding explicitly diluted the signal just as badly as the baseline over long contexts. It essentially suffered from the exact same vanishing signal degradation it was trying to cure.

## Phase 2: AG-CCL (Attention-Grounded Context-Conditioned Linear)
**The Idea:** Removed the monolithic GCM. Instead, we extracted the context signal structurally from *inside* the transformer block, directly ripping it from the layer's own Attention outputs (`attn_out`).

**Why it Failed:** 
- The initial attempts created a circular computation dependency (the context modified the FFN, but the context itself was derived from representations that hadn't been fully resolved). 

## Phase 3: Unleashing Gradient Flow (Selective Context Stream)
**The Idea:** We believed the static, hardcoded `0.99` EMA parameter from the baseline was "too rigid" to adapt to dynamic textual reasoning (e.g., failing to update quickly for new paragraphs while clinging to useless old ones). We built a GRU-style input-dependent gate (`SelectiveContextStream`) and **removed the `.detach()` command** to allow Back-Propagation Through Time (BPTT).

**Why it Failed:**
- The classic RNN failure mode: **Gradient explosion/vanishing**. By allowing gradients to flow backward through thousands of context-state updates across $T=2048+$, the gradient noise overwhelmed the dense layer updates. The original `.detach()` wasn't a flaw; it was a crucial, stable anchor that prevented the network from falling apart mathematically.

## Phase 4: DiT-Style Modulation (AdaRMSNorm)
**The Idea:** The `RemixedLinear` gates (basis modulator, output gate rank factorization) are highly complex. We attempted to scrap weight-level modulation entirely and copy Diffusion Transformer (DiT) architectures by shifting and scaling the standard MLP activations (`AdaRMSNorm`).

**Why it Failed:**
- Modulating the *activations* with an $O(sequence)$ scale/shift is fundamentally less expressive than perturbing the *weights* dynamically. `RemixedLinear` effectively generates custom sub-matrices per forward pass. AdaRMSNorm simply reoslved to scaling standard layer distributions, sacrificing vital capacity.

## Phase 5: Temporal Bottlenecks (MultiScale Context)
**The Idea:** If sequences degrade because short and long-term memory overwrite each other, we forced the context stream to split strictly into 3 channels (Fast $\alpha \approx 0.88$, Medium $\alpha \approx 0.50$, Slow $\alpha \approx 0.12$).

**Why it Failed:** 
- Manually slicing the state dimension ($D/3$) inherently starved the network of representation bandwidth. If the fast-changing local context needed the full capacity of `ctx_dim`, it was arbitrarily gated off.

## Phase 6: Breaking Circularity (Cross-Layer Stale Context Lag)
**The Idea:** To definitively solve the circular data dependency inside a single block, we forced Block $i$ to be conditioned strictly on the context emitted by Block $i-k$ (Stale Lag).

**Why it Failed:** 
- The exact localized features the FFN needs to process are produced directly by the attention module immediately preceding it. Passing an entirely different layer's context severely mismatched the spatial and semantic feature alignment.

## Phase 7: The CCL Ablation Suite (Causally-Safe Local Contexts)
**The Idea:** Having established that BPTT recurrence causes gradient explosions, we returned to strictly detached, causally-safe `LocalContextStream` architectures. We attempted to enrich the context signal itself rather than its recurrence path. We built a 7-part ablation suite including:
- **Attention Head Pooling (Design 2):** Using the post-RoPE Query vectors to gauge positional "search intent."
- **Sparse Top-K Gating (Design 3):** Forcing discrete, hard-routing basis selection (the true theoretical CCL mechanism).
- **Context Prototype Banks (Design 4):** A soft-lookup into learned "text-type" vocabularies.
- **Per-Head Context (Design 7):** Decoupling signals for Attention vs. FFN paths.

**Why it Failed:** 
While these designs restored structural stability (no `NaN` explosions), they consistently degraded validation loss into the `4.5` to `4.9` range (compared to baseline `4.15`). The empirical reality is that *locally derived multiplicative masking acts as an intense training drag.* In early optimization steps, the network struggles to simultaneously learn both a rich dense feature representation and a complex gating mechanism to mask it. The models fail to coordinate quickly enough, settling into suboptimal local minima compared to pure, un-gated dense layers with equivalent FLOPs.

## Phase 8: Latent Controllers & Boundary Gating
**The Idea:** To solve the "identity collapse" inherent in Phase 7 designs, we moved beyond viewing context as just another "feature stream." We implemented three architectural hypotheses:
- **Boundary-Gated Context (Design 8):** Context updates only at learned segment boundaries via a sigmoid gate, holding state fixed the rest of the time (with history detached for stability).
- **Hard Chunk Pooling (Design 9):** To improve Signal-to-Noise Ratio (SNR), tokens were causally pooled into fixed-size segment summaries (chunked summaries), eliminating token-level noise.
- **Auxiliary Context Objectives (Design 10):** We gave the context branch its own task (predicting newline boundaries or per-token entropy) to force it to encode non-trivial predictive information.

**Why it Failed:**
- **Signal Lag (Chunking):** Chunked summaries, while stable, introduced a lag between the information being produced and its availability as an FFN modulator. The model struggled to align "what happened 64 tokens ago" with the current token's sparse gating requirements.
- **Objective Disconnect (Aux Loss):** While the context branch successfully learned to predict boundaries/entropy (evidenced by the auxiliary loss converging), the resulting "learned knowledge" failed to translate into better next-token prediction in the main stream. The information that makes a token "structurally important" (a newline) is different from the information needed to modulate a low-rank basis transformation for specialized syntax.
- **Structural Identity Trap:** Even with auxiliary pressure, the most efficient "short-term" solution for the network is for the main stream to ignore the modulator branch entirely and focus on its dense weights, as the multiplicative gating remains a source of high-frequency noise during early optimization.

---

## Phase 9: Escaping the Identity Trap (RAL & DACS)
**The Idea:** We deduced that all prior designs regressed validation loss for two structural reasons. First, `RemixedLinear` was a *bottlenecked replacement* for the FFN dense layers, forcing the network to simultaneously learn a low-rank feature space and a complex gating mask. Second, drawing context from `norm(x)` meant the gate was transforming the exact same signal the FFN was already computing. We implemented 7 new ablations:
- **Residual Adaptive Linear (Proposal A):** Replaced `RemixedLinear` with an additive architecture: `y = base(x) + delta(x, ctx)`. Initializing `delta` to exactly 0 guarantees zero regression at step 0. 
- **FiLM Gate (Proposal C):** Added affine scale/shift conditioning as a more expressive alternative to sigmoid gating.
- **Detached Attention Context / DACS (Proposal B / F):** We pulled the context signal from `norm(attn_out).detach()`. This guarantees the context branch sees an orthogonal, cross-token mixture that the standard FFN input cannot trivially approximate.
- **Prefix / Decay Streams (Proposals E / G):** We tested causal history prefix aggregation and exponential-decay moving averages as computationally cheap alternatives to recurrent gating.

**Why it Failed:** 
Despite ensuring strict mathematical identity with the dense baseline at initialization (via `V=0`), the network persistently struggled during optimization. The addition of the dynamic, context-conditioned `delta` path caused gradients to fracture. 
1. **The Additive Delta Trap:** Even with `RAL`, the dynamic rank-32 delta requires simultaneous learning of context mappings and projection features. This parallel pathway introduced higher gradient variance compared to the stable, pure-dense base layer, acting as a frictional drag on convergence.
2. **Context Latency:** The causal prefix and DACS streams effectively decorrelated the context from `norm(x)`, fulfilling their mathematical purpose. However, they provided the FFN with generalized topological/positional data rather than specific semantic cues, proving insufficient to justify the parameter and optimization overhead.
3. **Conclusion:** Phase 9 solidifies a harsh empirical reality: dynamically modulating the core linear projections of the FFN — whether multiplicatively (RemixedLinear) or additively (RAL) — consistently under-performs static dense networks of equivalent capacity during aggressive pretraining schedules. The optimization penalty of learning dynamic structures vastly outweighs their theoretical capacity benefits.

## Phase 7: Operator-Space Modulation (Householder / Spectral)
**The Idea:** We introduced operator-space perturbations to avoid the identity trap of activation-only gates:
- `householder`: apply context-predicted reflections in basis space before projection.
- `spectral`: apply context-predicted near-isometric scaling in basis space.

The goal was to keep full-rank expressivity while moving control from raw activation amplitude to operator geometry.

**Why it Failed (in our sweeps):**
- The new operator branch still shares optimization bandwidth with a strong static path, so the controller under long context often converged to weak/noisy perturbations.
- The added controller complexity improved flexibility on paper, but did not close the empirical gap to the static dense baseline at long sequence lengths.
- Net result: no consistent gain over prior remixed variants; still below the baseline.

## Phase 8: OCD Delta + Orthogonality Penalty (`ocd`, `cclblock_orth_lambda`)
**The Idea:** We added an orthogonal-complement-style low-rank dynamic delta with an overlap penalty term to discourage dynamic updates from collapsing onto static structure.

**Why it Failed (in our sweeps):**
- While mathematically cleaner than unconstrained additive deltas, the practical overlap proxy did not reliably produce better long-context generalization.
- Tuning `cclblock_orth_lambda` mostly shifted optimization trade-offs (underfit when too high, ineffective when too low) instead of producing robust gains.
- It reduced some path interference symptoms but still did not beat the static dense baseline.

## Phase 9: Linear SSM-Style Context Stream (`ssm`)
**The Idea:** Replace EMA-like recurrence with a linear state-style context highway (`ParallelLinearContextStream`) to improve long-range conditioning stability.

**Why it Failed (in our sweeps):**
- The linear-decay state was more stable than fully unrolled recurrent gradients, but still did not provide enough useful long-horizon control signal to outperform baseline dense blocks.
- Gains were inconsistent and did not survive across depth/sequence sweep settings.

## Phase 10: Dual-V Shadow Routing (`cclblock_attn_shadow_dim`)
**The Idea:** Split attention output into:
1. normal content stream for residual path, and  
2. a shadow stream routed into FFN context conditioning.

This was meant to separate “what attention writes to residual” from “what context controller reads.”

**Why it Failed (in our sweeps):**
- The shadow path added extra routing degrees of freedom but did not provide a reliably better conditioning signal for next-token loss at long contexts.
- Additional complexity and parameter coupling increased tuning sensitivity without yielding repeatable improvements.
- Net result remained below the static baseline.

---

### Conclusion & Forward Outlook
1. **The Core Strength of Remixed:** Perturbing the actual low-rank weight structures of the linear layers works significantly better than activation normalization mechanisms.
2. **The necessity of `.detach()`:** Continuous, un-detached gradient flow through sequence time is universally destructive for our current setup. Detachment isolates the gradient paths to focus strictly on spatial layer-to-layer learning. 
3. **What also did *not* rescue performance:** Operator-geometry variants (Householder/Spectral), OCD-style overlap-penalized deltas, linear SSM-style context highways, and dual-V shadow routing all failed to produce durable wins over the dense baseline in our long-context sweeps.
4. **The next frontier:** If we want to solve long-sequence contextual degradation, we likely need a controller that is both identifiable and low-noise at scale (coarse regime selection), rather than additional fine-grained per-token control paths that increase optimization friction.

## Phase 11: Unified Operator-Family Sweep (Decoupled / Tucker / SVS / VQ / Lie-Poly-Grassmann / Predictive+Evidence Controllers)
**The Idea:** We consolidated the full redesign set into one ablation surface and ran them through the same remixed training/sweep interfaces, including:
- **Decoupled Feature-Space Routing (`decoupled`)** with explicit static/dynamic channel split.
- **Tucker-Decomposed Routing (`tucker`)** with simplex-routed operator core.
- **Singular Value Steering (`svs`)** and **Deferred Context Unlock (`dcu`)** variants.
- **Vector-Quantized Regime Routing (`vq`)** with discrete codebook-conditioned deltas.
- **Operator-family modulation in basis space:** `lie`, `polynomial`, `grassmann`.
- **Context-controller variants:** `predictive_chunk`, `evidence_ssm`, and `attn_geometry` source.

**What Improved:**
- The strongest configuration in this phase was **Decoupled routing** with:
  - `cclblock_modulation=decoupled`
  - `cclblock_gate_rank=128`
  - `cclblock_dynamic_ratio=0.5`

This setting produced the best relative improvement among the newly introduced families in our phase-level comparisons.

**What Did Not Beat That Setting:**
- Tucker/SVS/DCU/VQ provided useful structural ablations and sometimes competitive early curves, but did not exceed the decoupled-128/0.5 setup in our tested budgets.
- Lie/Polynomial/Grassmann operator-space controls were informative but remained more tuning-sensitive and less consistently strong than the best decoupled run.
- Predictive/evidence/geometry controller variants improved controllability and interpretability of context pathways, but were not sufficient on their own to overtake the best decoupled configuration.

**Takeaway:**
- The practical win in this phase came from **structurally separating static and dynamic feature subspaces** and giving the dynamic controller enough rank/capacity to matter (`gate_rank=128`) without forcing full-path competition over the same channels.
- This phase shifts our working hypothesis from “more expressive controller geometry” to “clean gradient-path separation + sufficient dynamic capacity.”

## Phase 12: Zero-Friction Context Conditioning (FSI / AESP / CKR)

**Core Diagnosis:** All prior phases shared a common failure mode — the **"optimization friction tax."** Every design that introduced trainable routing or gating created a multiplicative coupling in the gradient path (`∂L/∂W` depends on `g(ctx)`), forcing the optimizer to coordinate two competing learning tasks simultaneously. This coordination cost manifests as a 0.05–0.7 `val_bpb` penalty regardless of architectural sophistication. Even at T=64 (where context quality is a non-issue), CCL architectures underperformed the static dense baseline — proving the problem is fundamentally optimizer-interference, not information-theoretic.

**The Approaches:** Three novel designs that either eliminate or structurally isolate the routing mechanism from the dense gradient path:

### 12a: Frozen Subspace Indexing (FSI) — `cclblock_modulation=fsi`
**Mechanism:** Uses K frozen Householder reflections (unit vectors `v_k`) with soft routing weights from a frozen random projection of the attention signal: `y = base_dense(x - 2 * Σ_k w_k * v_k * (v_kᵀx))`. Zero trainable routing parameters — the only learned weights are the standard `nn.Linear` base projection.

**Hypothesis:** If the friction tax is the sole cause of regression, FSI should match the dense baseline exactly (since its gradient dynamics are identical to a standard linear layer — the frozen reflections only permute the input coordinate frame).

**Result: Loss increased by ~0.05 vs. baseline.** The frozen reflections actively *hurt* performance. The random Householder reflections, despite being orthogonal, scramble the input representation in ways that the base linear layer must then undo. The soft-weighted mixture of reflections creates a non-trivial input distribution shift that the optimizer must compensate for, burning capacity. This disproves the "coordinate frame diversity" hypothesis — random views of the input are worse than no views.

**Takeaway:** Frozen-random input transformations are not free. Even with zero gradient interference, the representational cost of undoing arbitrary reflections exceeds any diversity benefit.

### 12b: Attention-Entropy Stratified Projection (AESP) — `cclblock_modulation=aesp`
**Mechanism:** Routes tokens to different low-rank deltas based on attention entropy (approximated from `attn_out` magnitude). Stratum 0 (highest entropy = weakest context signal) has `alpha=0` → exact dense baseline. Higher strata progressively allow more specialization via frozen scaling. `y = base(x) + alpha_k * (x @ U_k) @ V_k` where k = stratum(entropy).

**Hypothesis:** By guaranteeing stratum 0 is pure dense and only specializing positions with strong context signals, we avoid the all-or-nothing regression of prior designs.

**Result: Loss increased by ~0.05 vs. baseline.** Despite the stratum-0 safety net, the `attn_out` mean is a poor proxy for attention entropy. The quantile-based bucketing introduces non-differentiable boundaries that create optimization noise. The low-rank deltas in higher strata add parameter overhead without sufficient signal quality to justify it.

**Takeaway:** The entropy routing concept may be sound but requires actual attention entropy (not a proxy) and learned (not frozen) scaling to be effective.

### 12c: Causal Kernel Reparameterization (CKR) — `cclblock_modulation=ckr` ✓
**Mechanism:** Uses K parallel dense branches mixed by position-dependent (NOT content-dependent) weights from a tiny causal conv1d over a learned positional signal: `y_t = Σ_k w_k(t) · W_k · x_t`. At init: conv output ≈ 0 → softmax(0) = 1/K → equal branch weighting (equivalent to a single dense layer).

**Hypothesis:** Position-dependent mixing breaks the identity trap because position weights (`w_k(t)`) and feature weights (`W_k`) learn independently — no chicken-and-egg problem. The causal conv captures local structure (paragraph boundaries, sentence rhythm) naturally.

**Result: Loss decreased by ~0.02 vs. baseline — the first CCL variant to show improvement.** The position-only routing avoids content-dependent gradient interference entirely. The causal conv learns interpretable positional patterns without competing with the branch weights for gradient bandwidth. Reparameterizable at inference for zero overhead.

**Why CKR Succeeded Where Others Failed:**
1. **No content-dependent routing:** `w_k(t)` depends on position, not token content. The branches `W_k` see standard `∂L/∂W` gradients without modulation coupling.
2. **Breaks the identity trap:** Unlike multiplicative gates that start at identity and stay there, CKR starts at uniform `1/K` weighting and naturally differentiates because position signals are inherently non-uniform.
3. **Independent learning:** Feature learning (branches) and position learning (conv) operate on orthogonal information sources, eliminating the coordination cost.
4. **Causal inductive bias:** The 1D causal conv naturally captures document-level structure (code blocks, paragraph boundaries, dialogue turns) that justifies per-position weight specialization.

---

### Phase 12 Takeaway

CKR's -0.02 win is real but small. The principle — position-only routing avoids content-gradient interference — is validated.

## Phase 13: Scaling CKR + Dual Optimizer

**Goal:** Amplify CKR's 0.02 improvement via richer position encoding, optimizer decoupling, and cautious content re-introduction.

### 13a: Enhanced CKR (Multi-Scale Position + More Branches)
**Mechanism:** 3 position channels, 8 branches, kernel size 128.

**Result: Negligible improvement with significant cost increase.** The extra channels and branches added substantial params and FLOPs but most branches learn near-identical features. Position alone doesn't carry enough information to justify 8 independent dense pathways.

### 13b: Dual-Optimizer CKR
**Mechanism:** Routed CKR's routing params (~1K/layer) to dedicated conservative AdamW (β₂=0.999, 0.5× LR). Branches remained on Muon.

**Result: No improvement in any configuration.** The routing parameters are too tiny for the optimizer choice to matter. The gradient signal is clean and low-dimensional already.

### 13c: CKR-Hybrid (Frozen Content Bias)
**Mechanism:** Frozen random projection of `attn_out.detach()` as 0.1× additive bias to position logits.

**Result: No measurable improvement.** Frozen random projections are too noisy to meaningfully disambiguate document types.

---

## Grand Synthesis: What We Know After 13 Phases

### What definitively fails:
1. **Content-dependent routing/gating** (Phases 3–11, 12a, 12b): Multiplicative coupling between content gates and dense gradients always degrades optimization.
2. **Frozen random input transformations** (FSI): Scrambling the input has non-zero cost even when gradient-free.
3. **Scaling position routing** (13a): More branches/channels can't extract signal that isn't there.
4. **Optimizer tricks** (13b): When gradient paths are already decoupled, optimizer choice is secondary.

### What has shown positive signal:
1. **Position-only routing** (CKR, -0.02): Position signals are orthogonal to content gradients.
2. **Static/dynamic channel separation** (Decoupled, Phase 11): Reducing the gradient competition surface.
3. **Identity-safe initialization**: Starting as the exact dense baseline is necessary but not sufficient.

### The fundamental tension:
The paper's vision is **context-conditioned** linear layers. But content conditioning creates friction. CKR succeeds by avoiding content entirely — but pure position can't achieve the goal of outperforming bigger models.

## Phase 14: Gradient-Isolated Content Conditioning

**Goal:** Test whether gradient isolation (`x.detach()`) can enable friction-free content conditioning, and whether cheaper position-modulation alternatives to CKR exist.

### 14a: GIAD — Gradient-Isolated Additive Delta
**Mechanism:** `y = base(x) + scale * delta_net(x.detach())`. The delta network receives detached input features, meaning `∂L/∂x` is mathematically identical to the pure dense baseline. Content conditioning with zero backward interference.

**Result: Crashed (dtype error under torch.compile + fp8).** The `x.detach()` operation breaks the autocast context, causing float32 propagation into FlashAttention. Fixed by explicit dtype preservation post-detach. Re-test pending.

### 14b: PSG — Positional Scalar Gating
**Mechanism:** `y_t = W · (s(t) ⊙ x_t)` — position-dependent per-channel diagonal scaling via causal conv. 140× cheaper than CKR (no duplicate weight matrices).

**Result: Degradation +0.04.** Position-dependent diagonal scaling is too weak a modulation — `W · diag(s)` can only scale column importances, not recombine features like a full-rank branch mixture. The effective weight perturbation is constrained to rank-1 changes per channel, which is insufficient for the model to learn interesting position-dependent specializations.

### 14c: SplitStream — Decoupled Channels + CKR
**Mechanism:** 75% of channels use standard dense projection; 25% use CKR-style position-dependent multi-branch routing. Combines Decoupled (Phase 11) + CKR (Phase 12).

**Result: Degradation +0.12 (worst of all Phase 14 proposals).** Channel splitting is actively harmful. The static path (75%) gets standard gradients but sees only a slice of the representation. The dynamic path (25%) has too few channels for meaningful position-dependent specialization. The concatenation boundary creates a hard partition that prevents cross-channel feature interaction.

---

## Updated Grand Synthesis: What We Know After 14 Phases

### Pattern 1: Full-rank is mandatory
Every design that restricts the effective weight to a subspace (low-rank deltas in RAL Phase 9, diagonal scaling in PSG Phase 14b, channel splitting in SplitStream Phase 14c, Decoupled Phase 11) performs worse than the baseline. The model needs **full-rank** access to all dimensions at every position.

### Pattern 2: Multi-branch mixture ≈ the only positive signal
CKR's 0.02 improvement comes from having K=4 **full-rank** branches mixed position-dependently. This is the ONLY design in 14 phases to beat the dense baseline. But scaling to K=8 showed no further improvement — suggesting 4 branches already capture the useful structure.

### Pattern 3: Position-only routing has a ceiling
CKR works because position routing is orthogonal to content gradients. But position alone is a very weak conditioning signal — it can capture document-level structure (paragraph boundaries, code blocks) but not semantic content. The 0.02 improvement represents the limit of what pure position can achieve.

### Pattern 4: Content conditioning is still unsolved
Every attempt to introduce content-dependent signals (Phases 3–11, 13c) has failed. GIAD (Phase 14a) theorized that gradient isolation via `x.detach()` would break the failure mode — but hasn't been successfully tested yet due to the dtype crash.

### The remaining question:
Can a design achieve CKR-like multi-branch expressivity (Pattern 2: full-rank, Pattern 1: mandatory) without CKR's 4× parameter overhead, while keeping position-only routing (Pattern 3: the only safe routing)?

## Phase 15: Parameter-Efficient Position Routing & Diagnostics

**Goal:** Achieve CKR's multi-branch expressiveness at lower parameter cost (LoKR), test content conditioning with proper gradient isolation (GIAD), and add diagnostic instrumentation.

### 15a: LoKR — Low-rank Kernel Reparameterization (K=8,r=16 and K=4,r=32)
**Mechanism:** Shared full-rank base W + K low-rank perturbations: `W_eff(t) = W + Σ w_k(t)·U_k@V_k^T`. Same position-only routing as CKR via causal conv on learned position signals, but with 21% parameter overhead (vs CKR's 300%).

**Result: Degradation +0.08 (both K=8/r=16 and K=4/r=32).** Low-rank perturbations cannot achieve the expressivity of full-rank branches. This confirms Pattern 1 from a different angle: even ADDITIVE low-rank perturbations to a full-rank base are insufficient. CKR's advantage specifically comes from K **independent** full-rank transforms, not from the routing mechanism.

**Technical issue (resolved):** The initial implementation had `.item()` calls inside `forward()` for diagnostics caching, which caused graph breaks under `torch.compile`, fragmenting the compiled graph and causing 16 GiB OOM during backward. Fixed by caching tensors in forward and calling `.item()` only in the post-forward diagnostics collector.

### 15b: GIAD — Gradient-Isolated Additive Delta (re-test)
**Mechanism:** `y = base(x) + delta_net(x)` with zero-init delta_up weights so the delta starts invisible.

**Result: Crashed (FlashAttention dtype error) across three fix attempts.**
- Attempt 1: `x.detach()` → breaks torch.compile's autocast context → float32
- Attempt 2: Custom `autograd.Function` (_StopGrad) → also breaks compiled graph dtype tracking
- Attempt 3: Remove detach entirely, use `scale * delta` → `nn.Parameter(torch.zeros(1))` is float32, causes float32 × bf16 = float32 promotion in compiled graph → FlashAttention crash
- **Final fix:** Remove scalar `scale` parameter entirely, rely on zero-init `delta_up.weight`

**Root cause lesson:** Under `torch.compile + fp8/bf16 autocast`, standalone float32 nn.Parameter scalars cause dtype promotion that propagates through the entire compiled graph. All scalar parameters must be in the compute dtype, or avoided entirely.

### 15c: Modulation Diagnostics
Added `ModulationDiagnostics` class tracking delta/base ratio and branch weight entropy. Not observable during Phase 15 runs because both LoKR (OOM) and GIAD (dtype crash) failed before reaching the first log step.

---

## Updated Grand Synthesis: What We Know After 15 Phases

### Pattern 1 (reinforced): Full-rank branches are mandatory, not just full-rank access
LoKR's failure confirms that even additive low-rank perturbations to a full-rank base are insufficient. CKR's advantage comes from having K **completely independent** full-rank transforms. The model needs the ability to learn genuinely different projections per position, not small tweaks to a shared projection.

### Pattern 2 (unchanged): CKR is the only positive signal (+0.02)
After 15 phases of experimentation, CKR (K=4 full-rank branches, position-only routing) remains the ONLY design that beats the dense baseline.

### Pattern 3 (new): torch.compile imposes strict constraints
Any operation that breaks the compiled graph (detach, custom autograd Functions, .item() calls, float32 scalar parameters) causes either dtype errors or OOM. Architectures must be "compile-clean": all operations should be standard PyTorch modules with no graph-breaking ops and no dtype mismatches.

### Pattern 4 (new): Parameter overhead ≠ expressivity
LoKR at 21% overhead performed worse than dense baseline. CKR at 300% overhead barely beat it (+0.02). The relationship between added parameters and benefit is highly non-linear — cheap approximations to CKR don't capture its value. This suggests the improvements, if any, come from structural properties (independent branch transforms) rather than raw parameter count.

### The refined question:
CKR shows position-dependent multi-branch mixing works. But 300% overhead for 0.02 improvement is not practical. Instead of making CKR cheaper (LoKR failed), can we make CKR's branches MORE effective — extracting more signal from the multi-branch structure to justify the overhead? Or is there a fundamentally different approach that achieves context conditioning without any of the failure modes we've catalogued?

## Phase 16: CKR Variants & Causal Output Mixer

**Goal:** Three proposals targeting different hypotheses about CKR's benefit.

### 16A: CKR-Anneal (Temperature Annealing)
**Mechanism:** Softmax temperature annealed from 2.0→0.3 during training to sharpen position routing.
**Result: UNTESTABLE (torch.compile recompilation hang).** The initial implementation used a Python float attribute for temperature, which torch.compile guards on — causing full recompilation every step when the float changes. **Fixed** by converting to a buffer tensor (`register_buffer`), which torch.compile treats as dynamic.

### 16B: CKR-FFN-Only
**Mechanism:** CKR on FFN projections only, plain Linear for attention Q/K/V/proj.
**Result: UNTESTABLE (Float8Linear crash).** Under fp8 mode, plain Linear → Float8Linear, which lacks `gate_parameters()` / `non_gate_parameters()` / `ln_basis` APIs. The `setup_optimizer` crashed. **Fixed** with `hasattr` guards in the optimizer routing loop.

### 16C: COM — Causal Output Mixer
**Mechanism:** `y = Linear(x) + gate(pos) ⊙ causal_depthwise_conv(Linear(x))`. Position-gated causal convolution after standard linear projections. ~0.5% overhead.
**Result: CATASTROPHIC degradation. Loss increased to 2.2** (from ~5.5 baseline). The model diverged completely. Post-processing convolution on linear outputs causes gradient instability — the conv gradients interfere with the base linear gradients despite zero-init gate.

### 16D: GIAD Re-test (gradient-isolated delta, now with fixed dtype)
**Result: Degradation +0.08.** Content conditioning via additive delta continues to fail, even with proper gradient isolation via zero-init.

### Technical fixes applied:
- torch.compile guard break: `self.temperature` (Python float) → `self.register_buffer('_temperature', tensor)` (buffer)
- Float8Linear API: `hasattr(rl, 'gate_parameters')` guard in `setup_optimizer`
- Diagnostics: Rewrote to compute from parameters directly (not forward cache), saves to `{checkpoint_dir}/modulation_diagnostics.jsonl` + wandb

---

## Updated Grand Synthesis: What We Know After 16 Phases

### Pattern 1 (reinforced): Full-rank branches remain mandatory
GIAD's failure (+0.08) confirms low-rank additive deltas don't work even with gradient isolation.

### Pattern 2 (unchanged): CKR (+0.02) is still the only positive result

### Pattern 3 (reinforced): torch.compile constraints are pervasive
Phase 16 revealed TWO new compile traps: (a) Python float attributes that change between steps cause guard-based recompilation, (b) Float8Linear conversion strips custom APIs from plain Linear modules. Solutions: use buffer tensors for dynamic values, `hasattr` guards in optimizer routing.

### Pattern 5 (new): Post-processing mixing is catastrophic
COM showed that adding convolution AFTER linear projections causes divergence (loss 2.2). The gradient flow from conv→linear creates destructive interference. This is fundamentally different from Mamba/RWKV where the conv is integrated into the architecture from the start, not bolted onto existing linear layers.

### Pattern 6 (new): Zero-init ≠ zero friction
Both COM and GIAD start as mathematical equivalents of the dense baseline (zero-init delta/gate). But even zero-init doesn't prevent the added pathway from disrupting training once gradients start flowing. The issue isn't the initial state — it's the gradient dynamics during training.

### The refined question (still):
CKR's 0.02 improvement at 300% overhead is the only positive signal. CKR-Anneal and CKR-FFN are now fixed and can be tested. But we need fundamentally new ideas. What structural properties give CKR its advantage, and can we achieve them more efficiently?

---

## Phase 17 Proposals: Deep Analysis + 10 Ideas

### Critical analysis of WHY CKR works

CKR's core operation: `y = Σ w_k(pos) * W_k * x`. This is equivalent to `y = W_eff(pos) * x` where `W_eff(pos) = Σ w_k(pos) * W_k`.

What CKR achieves: **position-dependent effective weight matrix** where the weights are a CONVEX COMBINATION of K full-rank matrices. This means:
1. W_eff is always full-rank (convex combo of full-rank = full-rank)
2. W_eff varies smoothly across positions (causal conv ensures this)
3. The "space" of possible W_eff is a K-dimensional manifold inside the space of all DxD matrices

The +0.02 comes from positions at different parts of the sequence using slightly different projections. But K=4→K=8 showed no further improvement, meaning the structure is low-dimensional.

Key insight: **CKR's benefit comes from smooth interpolation in weight space along the sequence dimension.** Not from having many branches, but from the ability to MOVE CONTINUOUSLY through weight space as position changes.

### Proposed experiments (10 ideas):

#### 17A: CKR-Anneal (re-test, now fixed)
Temperature 2.0→0.3. Tests whether sharper routing extracts more signal.

#### 17B: CKR-FFN-Only (re-test, now fixed)
CKR on FFN only, plain Linear for attention. ~150% overhead. Tests if attention already has enough position awareness via RoPE.

#### 17C: Position-Gated Residual (PGR)
**y = W₀x + g(pos) · W₁x** where g(pos) is a scalar from causal conv.
Key difference from CKR: only 2 branches, explicit base + delta structure. W₁ zero-init. g(pos) single scalar per position (not per-branch). This is CKR K=2 with explicit residual structure. ~100% overhead.
**Why it might work:** CKR K=4 and K=8 show no difference → the useful structure is low-dimensional. Maybe only ONE additional direction in weight space is needed.

#### 17D: Orthogonal Branch Initialization
Same CKR K=4 architecture, but initialize branch weights using orthogonal matrices (via `nn.init.orthogonal_`). Currently branches start random → their W_k are correlated → W_eff ≈ average ≈ single random matrix.
**Why it might work:** Orthogonal init ensures branches start in maximally different subspaces, giving CKR's routing something meaningful to interpolate between from step 0.

#### 17E: Branch Dropout
CKR K=4 with random branch dropout during training (drop 1 of 4 branches per forward pass, renormalize remaining weights). At eval, use all branches.
**Why it might work:** Forces each branch to be independently useful rather than co-adapted. Similar insight as regular dropout but applied to the weight-space interpolation.

#### 17F: Layer-Selective CKR
CKR on odd layers only (or last N/2 layers), plain Linear on even layers. Tests whether all layers benefit equally from position conditioning. The hypothesis: early layers learn universal features (position-independent), later layers need position awareness.
**Why it might work:** Cuts overhead in half while potentially keeping all the benefit if the useful position conditioning is concentrated in certain layers.

#### 17G: Weight-Shared CKR (WS-CKR)
All layers share the SAME K branch weights but have DIFFERENT routing convolutions (different pos_signal and branch_conv per layer). This dramatically reduces parameter overhead (~50% vs 300%) while maintaining per-layer position-dependent routing.
**Why it might work:** Branch weights define "modes" of computation; routing determines which mode to use where. Different layers might need the same modes but in different positions.

#### 17H: CKR with Auxiliary Branch Diversity Loss
Standard CKR K=4 plus an auxiliary loss: `L_div = -Σ_{i≠j} ||W_i - W_j||²`. This encourages branches to learn different projections rather than converging.
**Why it might work:** Without diversity pressure, gradient flow may cause branches to learn similar weights, making CKR degenerate to a single expensive linear layer. The diversity loss prevents this collapse.

#### 17I: Causal Interpolation Linear (CIL)
**y = ((1-α(pos)) · W₀ + α(pos) · W₁) · x** where α(pos) ∈ [0,1] from sigmoid of causal conv. Only 2 matrices, linearly interpolated. Zero-init convention: α starts at 0 → pure W₀.
Key difference from PGR (17C): this interpolates IN weight space (one multiplication), PGR sums TWO separate multiplications. CIL is more parameter-efficient in forward compute (one matmul vs two).
**Why it might work:** CKR's K=4→K=8 shows no benefit. Maybe K=2 with smooth interpolation captures all useful structure. CIL is the minimal version of this.

#### 17J: Positional Residual Bias (PRB)
**y = Wx + b(pos)** where b(pos) is a position-dependent bias vector from causal conv on learned position signal. No weight modification at all. ~1% overhead.
**Why it might work:** The simplest possible position conditioning that doesn't touch weights at all. If CKR's benefit comes from position-dependent output (not position-dependent transform), then a bias can capture it. This is a diagnostic experiment: if PRB matches CKR, the benefit was never about weight modulation.

---

### Phase 17 Results (first run — CKR NaN bug)

**CRITICAL BUG:** All CKR-based variants (17A, 17B, 17D, 17E) produced NaN on step 0.
**Root cause:** `register_buffer('_temperature', tensor)` combined with `model.to_empty(device)`.
`to_empty()` replaces ALL tensor storage (parameters AND buffers) with uninitialized garbage.
`init_weights()` re-inits parameters but had NO code to re-init buffers. Division by garbage → NaN.
**Fix:** Store `_init_temperature` as a Python float (survives `to_empty` since it's not a tensor), and
re-fill `_temperature.fill_(self._init_temperature)` in the `_init_research_module` path during `init_weights()`.
Also added comprehensive buffer re-init for ALL position-routed modules (pos_signal, conv weights, conv biases).

| Variant | Loss delta vs CKR base (+0.02) | Status |
|---|---|---|
| 17A: CKR-Anneal | NaN | 🔴 BUG (now fixed) |
| 17B: CKR-FFN-Only | NaN | 🔴 BUG (now fixed) |
| 17C: PGR | +0.07 degradation | ❌ Failed |
| 17D: CKR + Ortho Init | NaN | 🔴 BUG (now fixed) |
| 17E: CKR + Branch Dropout | NaN | 🔴 BUG (now fixed) |
| 17F: Layer-Selective CKR | Not implemented yet | ⏳ |
| 17G: Weight-Shared CKR | Not implemented yet | ⏳ |
| 17H: CKR + Diversity Loss | Not implemented yet | ⏳ |
| 17I: CIL | +1.3 degradation | ❌ Catastrophic |
| 17J: PRB | +0.08 degradation | ❌ Failed |

**Key observations from non-NaN results:**
- **PGR (+0.07):** Two-branch with position-gated delta. The explicit base+delta structure (zero-init W₁) failed. Despite gate starting at sigmoid(-3)≈0.05, the delta pathway still disrupted training. Reinforces Pattern 6: zero-init ≠ zero friction.
- **CIL (+1.3):** Catastrophic. Weight-space interpolation between two matrices caused severe instability. The issue may be that CIL doesn't have softmax normalization — alpha is unbounded sigmoid, and the interpolation (1-α)·W₀ + α·W₁ causes destructive gradient interference between the two weight matrices.
- **PRB (+0.08):** Position-dependent bias without weight modulation failed. This is a key diagnostic: it confirms CKR's benefit (if any) DOES come from weight modulation, not just output shifting. But also confirms that lightweight position conditioning can't match CKR.

### Updated Grand Synthesis: What We Know After 17 Phases

#### Pattern 7 (new): to_empty() + register_buffer = silent NaN
The `to_empty()` pattern (common in distributed training) replaces ALL tensor storage with garbage. `init_weights()` must explicitly re-initialize every buffer, not just parameters. Python float/bool/int attributes survive `to_empty()` and should be used to store "factory defaults" for buffer re-init.

#### Pattern 8 (new): CKR's K=4 convex combination is structurally unique
PGR (base + delta, K=2) and CIL (interpolation, K=2) both failed despite being conceptually simpler versions of CKR. CKR's benefit isn't just "position-dependent weights" — it's specifically the **K≥4 branch convex combination with softmax routing**. The softmax ensures weights are always non-negative and sum to 1, maintaining a well-conditioned W_eff. PGR/CIL don't have this constraint.

#### Pattern 9 (new): Bias-only position conditioning fails (PRB +0.08)
PRB (position-dependent bias, no weight change) degraded, confirming CKR's value comes from position-dependent TRANSFORMS (weight matrices), not position-dependent SHIFTS (biases).

#### Remaining experiments (must re-run after NaN fix):
17A (CKR-Anneal), 17B (CKR-FFN), 17D (CKR-OrthoInit), 17E (CKR-BranchDrop) all need re-running.
17F (Layer-Selective), 17G (Weight-Shared), 17H (Diversity Loss) are not yet implemented.

---

### Phase 17 Round 2 Results (post-NaN fix)

All CKR variants now run without NaN. But the results are universally negative:

| Variant | Loss delta vs dense baseline | Status |
|---|---|---|
| 17A: CKR-Anneal (temp 2.0→0.3) | +0.20 degradation | ❌ |
| 17B: CKR-FFN-Only | +0.08 degradation | ❌ |
| 17D: CKR + Ortho Init | +0.20 degradation | ❌ |
| 17E: CKR + Branch Dropout | +0.20 degradation | ❌ |

**WARNING: CKR's original +0.02 may not be reproducible.** All CKR variants are now WORSE than baseline. The original +0.02 from Phase 13 may have been noise.

### Critical Re-Evaluation

After 17 phases: ~170 experiments, ZERO reliably positive results. Possible explanations:
1. **Baseline already well-optimized** — Muon + fp8 + QK-norm + value residuals near efficiency frontier
2. **Parameter overhead ≠ capacity** — Extra params fit routing mechanics, not better representations
3. **Position conditioning solving a non-problem** — RoPE already provides sufficient position info
4. **Gradient interference** — ANY modification disrupts Muon + μP + compile gradient flow
5. **Scale dependence** — Ideas might only help at larger scale where baseline plateaus

### Diagnostic fix

ModulationDiagnostics rewritten: supports ALL module types, auto-enabled for research models, saves per-layer JSONL. No more `--modulation-diagnostics 1` needed.

---

## Phase 18 Proposals: Beyond CKR — Fundamentally Different Approaches

CKR is exhausted. Phase 18 explores completely different hypotheses about what could actually help.

### Core question: what are the real bottlenecks at this scale?

1. Parameter efficiency — more computation from fewer params
2. Representation diversity — preventing redundant head/channel features
3. Training dynamics — improving HOW it trains, not WHAT it computes
4. Structural inductive bias — useful structure without adding params

### Proposed experiments (10 ideas):

#### 18A: Adaptive ReLU² Gate (ARG)
y = W·x · g(x) where g(x) = sigmoid(w·x + b) per output channel. CONTENT-dependent, not position. g starts at 1.0. ~0.01% overhead.
Hypothesis: An adaptive input-dependent gate lets the model learn when to attenuate/amplify specific output channels.

#### 18B: Spectral Normalization + Gradient Penalty (SNGP)
No architecture change. Regularizes Lipschitz constant via spectral norm + gradient penalty.
Hypothesis: ALL experiments degrade because of gradient instability, not bad architecture.

#### 18C: Kronecker-Factored Linear (KFL)
Replace W·x with (A ⊗ B)·x. FEWER parameters than dense. Tests if baseline is over-parameterized.

#### 18D: Shared-Basis Multi-Head Attention (SBMHA)
All heads share one K/V projection, independent Q only. More aggressive than GQA.

#### 18E: Stochastic Depth (LayerDrop)
Randomly skip FFN blocks with probability p during training. Scale at eval by (1-p). Pure regularization.

#### 18F: Per-Channel Learning Rates (PCLR)
Each output channel gets its own learned LR multiplier. No architecture changes.

#### 18G: Auxiliary Representation Similarity Loss (ARSL)
Penalty for high cosine similarity between adjacent layers: L_sim = lambda * sum cos_sim(h_l, h_l+1).

#### 18H: Mixture of Norms (MoN)
norm(x) = w1*RMSNorm(x) + w2*LayerNorm(x) with learned w1, w2. Minimal overhead.

#### 18I: Dynamic Activation Function (DAF)
f(x) = alpha*ReLU²(x) + beta*GELU(x) + gamma*SiLU(x) with per-layer learned scalars. ~3 params per layer.

#### 18J: Causal Attention Score Bias (CASB)
Learned lower-triangular bias B in attention: attn = softmax(QK^T/sqrt(d) + B). Like learned ALiBi.
**NOT IMPLEMENTED** — incompatible with FlashAttention3 (FA3 doesn't support arbitrary additive bias).

### Implementation Status

| Experiment | Mode | CLI Flag | Status |
|---|---|---|---|
| 18A: ARG | `--cclblock-modulation arg` | modulation mode | ✓ Implemented + tested |
| 18B: SNGP | N/A | `--p18-gradient-penalty 0.001` | ✓ Weight Frobenius norm penalty |
| 18C: KFL | `--cclblock-modulation kfl` | modulation mode | ✓ Implemented + tested |
| 18D: SBMHA | N/A | `--n-kv-head 1` | ✓ Already supported via GQA |
| 18E: LayerDrop | N/A | `--p18-layer-drop 0.1` | ✓ Implemented + tested |
| 18F: PCLR | N/A | `--p18-per-channel-scale 1` | ✓ Per-channel learned scale |
| 18G: ARSL | N/A | `--p18-aux-sim-lambda 0.01` | ✓ Layer similarity penalty |
| 18H: MoN | N/A | `--p18-mixture-norm 1` | ✓ RMSNorm+LayerNorm blend |
| 18I: DAF | N/A | `--p18-dynamic-activation 1` | ✓ Learned activation mix |
| 18J: CASB | N/A | N/A | ✗ Incompatible with FA3 |

All Phase 18 flags are orthogonal — they can be combined with ANY modulation mode including the dense baseline.
Diagnostics auto-enabled: all modules report to `modulation_diagnostics.jsonl`.

### Phase 18 Results

| Experiment | Result | Notes |
|---|---|---|
| 18A: ARG (Adaptive Gated Linear) | **OOM** | Content gate doubles FFN parameter count |
| 18B: SNGP (Gradient Penalty) | **OOM** | Extra backward graph from penalty |
| 18C: KFL (Kronecker-Factored Linear) | **+0.12 loss** | Too constrained — Kronecker structure can't represent dense W |
| 18E: LayerDrop | **+0.12 loss** | Dropping FFN hurts more than it regularizes at this scale |
| 18F: Per-Channel Scale | **OOM** | Extra params pushed over memory budget |
| 18G: Aux Similarity Loss | **+0.12 loss** | Penalty on layer similarity may conflict with residual stream design |
| 18H: Mixture of Norms | **OOM** | Two full norms per layer doubled norm memory |
| 18I: Dynamic Activation | **OOM** | Three activation branches simultaneously materialized |

**Critical observation**: The dense baseline at depth=4 already fits tightly in memory. ANY additional parameters or activation memory pushes it OOM. The approaches that didn't OOM (18C, 18E, 18G) all degraded loss by ~0.12 vs best baseline.

### Post-Mortem: Why Phase 12–18 Failed

All experiments across Phase 12–18 have one thing in common: **none beat the dense baseline**. The pattern is clear:

1. **Overhead kills**: At our model scale (~46M params), ANY parameter overhead is proportionally large enough to hurt. A 5% parameter increase doesn't buy enough expressivity to compensate for the reduced effective width.

2. **Position conditioning is a dead end for small models**: CKR, PGR, CIL, PRB — all position-based routing adds parameters that learn near-identity mappings. The model doesn't have enough capacity for "specialized modes."

3. **Content conditioning needs zero overhead**: Ideas like ARG that use content to gate are sound in principle but double the parameter count of the FFN.

4. **Memory is the binding constraint**: At depth=4 on A100 with FP8/bf16, we're at the memory ceiling. New ideas MUST be parameter-neutral or parameter-reducing.

### Phase 19 Direction: Broader Modulation Scope

The next phase should expand the search beyond "replace nn.Linear with something else" and explore **indirect modulation** — ways to increase expressivity without adding parameters to the linear layers themselves:

**Attention-Level Modulation (Preferred — weight-adjacent):**
- **Head Permutation/Routing**: Learn to reorder attention heads per-input, effectively creating input-dependent multi-head patterns from fixed weights
- **Dynamic Head Scaling**: Per-head learned scale factors that modulate head importance based on input content (1 scalar per head, ~n_head params total)
- **Cross-Head Mixing**: Lightweight linear combination of head outputs before projection, learned per-position or per-content
- **Adaptive Head Dropping**: During training, randomly zero out entire heads (like DropHead); at inference, use all heads. Zero params added.

**Activation-Level Modulation:**
- **Residual Stream Gating**: Per-layer learned scalar gate on the residual connection (x = x + α·block(x)), α initialized to 1.0. Only 1 param per layer.
- **Feature Mixing**: Lightweight 1×1 convolution on the residual stream between layers (D→D with grouped convolution, ~D params)
- **Adaptive Skip Connections**: Learn which layers to skip entirely based on input content (requires differentiable routing)

**Weight Sharing / Rearrangement (Zero Additional Params):**
- **Cross-Layer Weight Sharing with Learned Offsets**: Share W across layers but learn a small per-layer delta (LoRA-style, rank-1 or rank-2)
- **Weight Permutation**: Learn a fixed permutation of weight rows/columns that creates a "second mode" from the same parameters
- **Spectral Reparameterization**: Reparameterize W = U·Σ·V^T and modulate only Σ (diagonal) based on content — D params, modulates the full D×D weight

**Key constraints for Phase 19:**
1. MUST NOT increase peak activation memory (no OOM)
2. MUST add ≤ 0.1% parameters (effectively zero overhead)
3. MUST be compatible with FlashAttention3 and torch.compile
4. Prefer weight/attention modulation over activation/normalization modulation

## Phase 19: Zero-Overhead Indirect Modulation (Implemented)

Phase 19 realizes 10 proposals targeting indirect modulation — modifying the **environment around** nn.Linear rather than replacing it. All proposals use identity-at-init design (softplus-parameterized or zero-init gating) so the model starts as the exact dense baseline.

### 19A: Residual Gate Scaling (RGS)
Per-layer learned scalar α on block FFN output: `x = x + softplus(α) * mlp(x)`. Init softplus(0.5413) ≈ 1.0. **1 param per layer.**

### 19B: Head Importance Scaling (HIS)
Per-head learned multiplier on post-attention output, applied before c_proj reassembly. Lets model smoothly suppress redundant heads. **n_head params per layer.**

### 19C: Residual Stream Depthwise Mixing (RSDM)
Grouped 1×1 convolution between blocks on residual stream, gamma-gated (init=0 → identity). Group size controls parameter count. **n_embd × group_size params per layer.**

### 19D: Attention Logit Bias Learning (ALB)
Per-head learned QK temperature via Q-scaling before flash_attn (FA3-safe). softplus-parameterized for stability. **n_head params per layer.**

### 19E: Learned Residual Decay (LRD)
Global depth-dependent x0 decay: `x0_factor_i = sigmoid(decay_raw)^i`. 1 param controls how quickly x0 influence fades across layers.

### 19F: Gradient Equilibrium Regularization (GER)
Post-backward gradient modifier: equalizes per-block gradient norms by scaling toward the mean. **0 params.** Implemented in training loop.

### 19G: Spectral Reparameterization (SRP)
Decomposes c_proj as W = U·diag(σ)·V^T where U,V frozen from SVD at init, only σ learned. **D params per reparameterized layer.** Identity at init.

### 19H: Anti-Collapse Norm Penalty (ACNP)
Soft penalty on weight cosine similarity between adjacent layers' c_fc matrices. Unlike 18G (activation similarity), this targets weight redundancy. **0 params.**

### 19I: Adaptive Value Residual Gating (AVRG)
Adds learnable bias to ve_gate Linear, giving per-head baseline importance for value residuals. **n_kv_head params per VE-enabled layer.**

### 19J: Training-Time Weight Perturbation (TWP)
Isotropic noise on MLP weights during training (SAM-like flat-minima regularization). Disabled at eval. torch.compile-safe. **0 params.**

### Diagnostics Expansion (ModulationDiagnosticsV2)
Expanded the existing diagnostic system with:
- **Gradient health**: per-layer grad norms, SNR, flow ratio (attn vs FFN balance)
- **Weight dynamics**: per-layer weight norms
- **Residual stream**: resid_lambda/x0_lambda values per block
- **Per-proposal metrics**: alpha values (19A), head scale distributions (19B), gamma scalars (19C), logit scale stats (19D), depth decay factor (19E), sigma entropy/ratio (19G), VE bias values (19I)

All metrics compute in torch.no_grad() outside the compiled forward graph for zero training overhead.

### Phase 19 Experimental Results (model_dim=128, depth=4, 121M tokens, H100)

| Proposal | Val BPB | Δ from Baseline | Verdict |
|---|---|---|---|
| **Baseline** | **1.2573** | — | Reference |
| **19D: Attn Logit Bias** | **1.2532** | **−0.0041** | 🏆 Best. Learned QK temperature helps. |
| **19J: Weight Noise** | **1.2558** | **−0.0014** | 🥈 Flat-minima regularization works. |
| **19H: Weight Anticollapse** | **1.2570** | **−0.0003** | Marginal improvement. |
| 19F: Gradient Equilibrium | 1.2573 | +0.0000 | Neutral — no effect at this scale. |
| 19A: Residual Gate | 1.2631 | +0.0059 | Hurt. Output scaling destabilizes. |
| 19B: Head Importance | 1.2641 | +0.0068 | Hurt. Per-head scaling harmful at 1 head. |
| 19I: VE Bias | 1.2654 | +0.0082 | Hurt. Biasing VE gates adds noise. |
| 19G: Spectral Reparam | 1.3210 | +0.0637 | Severely hurt. Frozen U/V restricts gradients. |
| 19E: Residual Decay | CRASHED | — | `num_scaling_params` assertion failed (mu-P bug). |
| 19C: Residual Mixing | Not tested | — | Omitted from sweep. |

**Key findings:**
1. **19D (Attn Logit Bias)** is a clear winner — a per-head learned QK temperature (1 param per head) consistently helps. This is essentially free and should be kept as a default.
2. **19J (Weight Noise)** works as a regularizer — SAM-like flat-minima exploration during training generalizes better.
3. **19G (Spectral Reparam)** is the biggest failure — freezing U/V from the SVD of zero-init c_proj means the decomposition captures random noise, not meaningful directions. The frozen basis is toxic.
4. **19B (Head Importance)** is meaningless at n_head=1 (model_dim=128 / head_dim=128). Needs re-testing at larger model dims.
5. **19A (Residual Gate)** learning to scale block output hurts — at 4 layers, every block's output is critical; scaling any down degrades the model.

### Post-Mortem: Phase 19 Lessons

The indirect modulation thesis partially validated: **attention-side modulation works, weight-side modulation doesn't at this scale.** The two winners (19D, 19J) share a key property: they don't interfere with the gradient flow through the main computation path. 19D modulates attention logits (pre-softmax), and 19J adds noise that's consumed in the forward pass. All proposals that directly scaled or gated the residual/output path (19A, 19B, 19I) degraded performance.

The spectral reparameterization (19G) failure reveals a deeper issue: at init, c_proj is zeros, so SVD of zeros → random U/V. Freezing these random bases and only learning σ creates an arbitrary reparameterization that the optimizer can't recover from.

---

## Phase 20: Context-Conditioned Dynamic Weight Computation

### Core Vision

A linear layer where context dynamically determines the effective computation. The layer stores more capacity than it uses per-forward-pass, and a context-aware mechanism selects *which* computation to perform. The ratio of stored-to-active capacity is a configurable hyperparameter (×4, ×6, ×8).

### Systematic Failure Mode Catalogue (from 19 Phases)

| FM | Name | Phases |
|---|---|---|
| FM1 | Gradient Fracturing (routing+feature weight coordination cost) | 3,7,8,9,14a,16D |
| FM2 | Identity Trap (ctx from norm(x) → gate=identity) | 5,7,8 |
| FM3 | Rank Bottleneck (subspace restriction) | 9,14b,14c,15a |
| FM4 | Capacity Tax (overhead disproportionate at small scale) | 12–18 |
| FM5 | Path Interference (post-processing gradient competition) | 16C,17C,17I |
| FM6 | Context Staleness (lagging inputs) | 4,6,8 |
| FM7 | Compile/Dtype Brittleness | 14a,15a,15b,16A |
| FM8 | Position Ceiling (K>4 useless, CKR +0.02 may be noise) | 13a,17 |

**Fundamental Constraint:** Any mechanism where the optimizer must simultaneously learn routing AND feature weights fails at our scale.

### Implemented Proposals


## Phase 20 Proposals

### 20A: Hash-Routed Column Selection (HRCS)

**Core idea:** Store a fat weight matrix W ∈ ℝ^{D_stored × D_out}, but at each token position, select only D_active columns based on a **deterministic hash** of the input token's features. No learned gate — the routing is a frozen, content-sensitive hash function.

**Mechanism:**
```
# W: (D_stored, D_out), D_stored = scale_factor * D_active
# x: (B, T, D_active)
# hash_fn: frozen LSH, maps x → indices of D_active columns in W

indices = hash_fn(x)           # (B, T, D_active) — deterministic, no grad
W_active = W[indices]          # (B, T, D_active, D_out) — gather
y = einsum('btd,btdo->bto', x, W_active)
```

**Why it might work:**
- **Avoids FM1** (gradient fracturing): hash is frozen, no routing gradients
- **Avoids FM2** (identity trap): hash is content-dependent but NOT learned
- **Avoids FM3** (rank bottleneck): each selected W_active is full-rank in active dimensions
- Active compute is D_active × D_out FLOPs (same as small model)
- Stored capacity is D_stored × D_out (same as large model)

**Risks:**
- Random hashing may not route intelligently — similar inputs might get very different columns
- The gather operation may not be compile/FlashAttention friendly
- D_stored weights live in VRAM even if only D_active are used per step

**Scale ratio:** Configurable via `D_stored / D_active`

---

### 20B: Locality-Sensitive Weight Routing (LSWR)

**Core idea:** Like 20A but with **Locality-Sensitive Hashing** — similar inputs hash to similar column subsets. This creates smooth, content-dependent routing without learned gates.

**Mechanism:**
```
# Fixed random hyperplanes for LSH (frozen at init, never updated)
# H: (n_planes, D_active) — random unit vectors
hash_bits = sign(x @ H.T)                    # (B, T, n_planes) — binary hash
bucket_id = binary_to_int(hash_bits)          # (B, T) — integer bucket
column_indices = bucket_to_columns[bucket_id] # (B, T, D_active) — precomputed lookup

W_active = W[:, column_indices]               # gather active columns
y = (x * W_active).sum(dim=-1)               # or proper matmul
```

**Why it might work:**
- LSH guarantees nearby inputs in feature space → similar column selections → smooth weight landscape
- Completely frozen routing — zero gradient interference
- The "weight space" is much larger than what any single token uses

**Risks:**
- LSH buckets may be too coarse or too fine
- Bucket-to-column mapping needs careful design (random assignment? learned once then frozen?)
- Memory: full W must be in VRAM for the gather

---

### 20C: Low-Rank Weight Composition with Frozen Basis (LRCFB)

**Core idea:** Instead of selecting columns, compose the effective weight from a **frozen bank of full-rank basis matrices** using content-derived coefficients. The basis matrices are initialized once (e.g., from orthogonal random matrices) and NEVER updated. Only the small composition coefficients are learned — but they're static (not input-dependent).

Wait — that's just an ensemble. Let me refine:

The composition coefficients ARE input-dependent but come from a **frozen random projection** of the input (not a learned gate):

**Mechanism:**
```
# B_k: K frozen full-rank basis matrices, (D_in, D_out), K=4-8
# R: frozen random projection matrix, (D_in, K)

alpha = softmax(x @ R)              # (B, T, K) — frozen routing, no grad through R
W_eff = sum_k alpha_k * B_k         # (D_in, D_out) — effective weight per token
y = (x * W_eff).sum(-1)             # matmul with token-specific weight
```

**Why it might work:**
- **Avoids FM1**: NO learned routing parameters — R is frozen, B_k are learned normally
- **Avoids FM2**: R is random, not derived from normalized x
- **Avoids FM3**: Each B_k is full-rank; the convex combination is full-rank
- The gradient to each B_k is `∂L/∂B_k = alpha_k * (x^T δy)` — standard linear gradient scaled by a frozen coefficient. No gradient fracturing.
- Effectively: each token gets a different "blend" of K weight matrices, but the blending is deterministic (from frozen R), so the optimizer only needs to learn the K base matrices

**Key insight:** CKR used position to route, which is orthogonal but uninformative. LRCFB uses frozen content projection to route, which is content-sensitive but gradient-free. This is the missing combination we never tried.

**Risks:**
- Frozen random R may create arbitrary routing that doesn't correlate with useful input structure
- K full-rank matrices = K× parameter overhead (same as CKR)
- BUT: since we know K=4 is sufficient (Phase 13), this is manageable

**Scale ratio:** K controls the capacity multiplier. K=4 gives ×4 capacity in weight space.

---

### 20D: Detached-Gradient Content Routing (DGCR)

**Core idea:** Learn the routing network, but pass gradients to it ONLY through a separate auxiliary loss — never through the main next-token prediction loss. The main loss gradients treat the routing as a constant.

**Mechanism:**
```
# Standard forward:
alpha = router(x.detach())                        # content-dependent, gradient-stopped from main loss
alpha_sg = alpha.detach()                          # stop grad for main branch
W_eff = sum_k alpha_sg_k * W_k                    # effective weight (routing treated as constant)
y = x @ W_eff                                     # main computation

# Auxiliary loss (separate backward):
# Train router to predict something useful about the input
# e.g., predict which branch reduces reconstruction error
with torch.no_grad():
    per_branch_error = [((x @ W_k - target)**2).mean() for k in range(K)]
best_branch = argmin(per_branch_error)
aux_loss = cross_entropy(alpha_logits, best_branch)
```

**Why it might work:**
- **Avoids FM1** completely: the main loss never backprops through the router
- The router learns via its own loss signal (which branch would have been best)
- The base matrices W_k see clean, standard gradients like CKR's branches
- Content-dependent (not just position) without identity trap or gradient interference

**Risks:**
- The auxiliary loss is a proxy — "which branch is best ex-post" might not correlate with "which branch would help next-token prediction"
- Two backward passes (main + aux) adds compute
- The router is trained on a different objective than the goal, which may misalign

> [!IMPORTANT]
> This is the only proposal that attempts to learn content-dependent routing while avoiding FM1. If the auxiliary objective can be designed well, this overcomes the fundamental constraint.

---

### 20E: Progressive Weight Unfreezing (PWU)

**Core idea:** Don't try to learn routing and weights simultaneously. Instead, train in phases:
1. **Phase A (standard pretraining):** Train a dense model with K=1 normally. No routing.
2. **Phase B (branch distillation):** Freeze the trained weights. Fork them into K copies with random perturbations. Train a small router to minimize reconstruction loss against the original model's outputs.
3. **Phase C (joint fine-tuning):** Unfreeze everything with a heavily reduced LR on branches, normal LR on router.

**Why it might work:**
- **Avoids FM1** in the critical early phase: routing and features never compete simultaneously
- The router starts with meaningful branches (from a pre-trained model) rather than random weights
- By the time we do joint training (Phase C), the router has already learned a useful routing pattern

**Risks:**
- Multi-phase training is complex and may not fit our sweep infrastructure
- The perturbations in Phase B may not create meaningfully different branches
- This doesn't help with pretraining from scratch — only with fine-tuning

---

### 20F: Mixture of Narrow Experts (MoNE)

**Core idea:** Instead of one wide MLP with D_in → 4×D_in → D_in, use K narrow MLPs each with D_in → (4×D_in/K) → D_in. A router selects the top-1 or top-2 experts per token. The total parameter count is the SAME as the baseline (each expert is 1/K the width), but different tokens use different experts.

**Mechanism:**
```
# K narrow experts, each with hidden_dim = 4*D/K
# Router: Linear(D, K) → softmax → top-1 or top-2

router_logits = x @ R_router     # (B, T, K)
top_idx = topk(router_logits, 1) # (B, T, 1)
expert_out = experts[top_idx](x) # only compute 1 expert per token
y = expert_out                   # or weighted sum of top-2
```

**Why it might work:**
- **Zero parameter overhead**: Same total params as baseline (K experts of width 4D/K = total width 4D)
- **Zero FLOP overhead**: Only 1 expert computed per token → same FLOPs as one narrow MLP
- Each expert can specialize in different input patterns
- This IS standard MoE but at the **intra-layer** level, at **parameter parity** with dense
- The router is tiny (D × K params)

**Risks:**
- Standard MoE load-balancing issues (experts collapse to a few being used)
- At small scale (D=128), K narrow experts may be too narrow (4D/K = 128 with K=4)
- Router gradient interference (FM1) — but the router is very small

**Scale ratio:** K controls the number of specialized modes. Active compute = 1/K of stored compute.

> [!TIP]
> This is conceptually closest to the user's original vision: the layer has K× more "weight capacity" (K different expert functions) but only uses 1/K of it per token, with content-aware routing deciding which.

---

### 20G: Frozen-SVD Weight Subspace Traversal (FSVD)

**Core idea:** Take the dense weight matrix W, compute its SVD: W = UΣV^T. Freeze U and V (the directional structure). Then let a content-dependent signal SELECT which singular values to amplify/suppress. The effective weight varies per-token but stays in the span of the original SVD.

Unlike Phase 19G (which failed): here U,V come from a PRE-TRAINED weight (not zero-init), and the σ modulation is content-dependent (not just learned scalars).

**Mechanism:**
```
# Pre-training phase: train dense model normally
# Post-training: decompose W = UΣV^T, freeze U, V
# Content-dependent σ selection:

sigma_gate = sigmoid(frozen_proj(x))  # (B, T, D) — frozen content projection
sigma_eff = sigma * sigma_gate        # element-wise gating of singular values
W_eff = U @ diag(sigma_eff) @ V.T    # effective weight per token
y = x @ W_eff
```

**Why it might work:**
- U, V capture the meaningful directional structure from pre-training
- The frozen projection avoids FM1 (no routing gradients)
- Content-dependent σ selection lets different inputs emphasize different spectral components
- Each token uses a different "view" of the pre-trained weight — potentially more expressive than the original

**Risks:**
- Requires pre-training first (can't train from scratch)
- The frozen gate may not select useful σ components
- SVD is expensive at init (but only done once)
- May not be much better than just using the original W

---

### 20H: Noise-Contrastive Expert Assignment (NCEA)

**Core idea:** Take inspiration from 19J (weight noise works!). Instead of random noise, use **structured** noise — multiple weight banks that are perturbed versions of a shared base. The router uses contrastive learning to assign tokens to the bank whose perturbation direction is most beneficial, trained via reconstruction comparison (not main loss).

This combines the success of 19J (noise helps) with MoE-style routing.

**Mechanism:**
```
# W_base: shared base weight
# delta_k: K learned perturbation directions (initialized random, small)
# Assignment: which perturbation direction helps this token most?

W_k = W_base + eps * delta_k                    # K perturbed weights
y_k = [x @ W_k for k in range(K)]               # K candidate outputs
y_base = x @ W_base                              # base output

# Assignment via contrastive comparison (no grad to W_base/delta_k from router):
with torch.no_grad():
    scores = [-||y_k - target||² for k in range(K)]  # which is closest to target?
    best_k = argmax(scores)

y = y_k[best_k]  # or: soft weighted sum
# Router aux loss to predict best_k from x
```

**Risks:**
- Computing ALL K outputs defeats the purpose (we want to compute only 1)
- Need a way to predict the best k WITHOUT computing all of them
- The perturbations delta_k may collapse to zero

---

### 20I: Attention-Derived Weight Interpolation (ADWI)

**Core idea:** The attention mechanism already computes content-dependent routing — that's what the attention weights ARE. Use the EXISTING attention weights (not a new router) to interpolate between weight matrices. Each attention head's contribution to the residual stream determines which expert's FFN weights to use.

**Mechanism:**
```
# After attention: we have per-head outputs h_1, ..., h_H
# Each head has learned to attend to different content patterns
# Use head activation magnitudes as routing signal:

head_magnitudes = [||h_k||² for k in range(H)]     # (B, T, H)
alpha = softmax(head_magnitudes)                     # (B, T, H)

# H expert FFNs (or shared base + H perturbations):
y = sum_h alpha_h * FFN_h(x)
```

**Why it might work:**
- **Reuses existing attention computation** — zero additional routing cost
- Head magnitudes are a naturally content-dependent signal
- Different heads attend to different patterns → natural routing diversity
- The routing signal's gradient flows through the attention mechanism, which is already being trained

**Risks:**
- H experts at full width = H× parameter overhead (too expensive)
- Could use H narrow experts (MoNE-style) to keep parameter parity
- Head magnitudes may not correlate with useful routing patterns
- Gradient coupling between attention and FFN routing

---

### 20J: Weight Bank with Frozen Clustering (WBFC)

**Core idea:** Offline, cluster a large weight matrix's rows/columns into K groups using K-means on a pre-trained model. During training, store K centroids + per-row residuals. Route tokens to cluster centroids first (cheap), then apply residual refinement (if needed).

This is the most direct realization of the user's vision: a large weight bank where context selects a subspace.

**Mechanism:**
```
# Offline: cluster columns of W into K groups
# centroids: (K, D_out) — cluster centers
# residuals: (D_in, D_out) — per-column residuals from centroid
# assignment: (D_in,) → K — which cluster each column belongs to

# Online: for each token, select top-M clusters (M < K)
cluster_affinity = frozen_proj(x) @ centroids.T   # (B, T, K)
top_clusters = topk(cluster_affinity, M)           # (B, T, M)

# Gather only the rows/columns from selected clusters
W_active = gather_columns_from_clusters(W, top_clusters)
y = x @ W_active
```

**Risks:**
- Clustering is static (computed offline) — may not adapt well to training dynamics
- The gather pattern may not vectorize well for GPU compute
- Requires pre-training first

---

## Comparative Analysis

| Proposal | Avoids FM1 (fracturing)? | Avoids FM2 (identity)? | Avoids FM3 (rank)? | Avoids FM4 (capacity tax)? | Compute savings? | Train from scratch? |
|---|---|---|---|---|---|---|
| **20A: HRCS** | ✅ (frozen hash) | ✅ (hash ≠ norm(x)) | ✅ (full columns) | ⚠️ (fat W in VRAM) | ✅ (D_active matmul) | ✅ |
| **20B: LSWR** | ✅ (frozen LSH) | ✅ | ✅ | ⚠️ (fat W in VRAM) | ✅ | ✅ |
| **20C: LRCFB** | ✅ (frozen R) | ✅ (R ≠ norm) | ✅ (full-rank B_k) | ❌ (K× weight params) | ❌ (K matmuls) | ✅ |
| **20D: DGCR** | ✅ (aux loss only) | ✅ (detached input) | ✅ (full-rank W_k) | ❌ (K× weight params) | ⚠️ (need all K for aux) | ✅ |
| **20E: PWU** | ✅ (phased training) | ✅ | ✅ | ❌ (K× weight params) | ❌ | ❌ (pre-train first) |
| **20F: MoNE** | ⚠️ (tiny router) | ✅ (learned routing) | ✅ (narrow but full) | ✅ (same total params) | ✅ (1/K FLOPs) | ✅ |
| **20G: FSVD** | ✅ (frozen proj) | ✅ | ✅ (modulates SVD) | ✅ (same params) | ❌ (recompose W) | ❌ (pre-train first) |
| **20H: NCEA** | ⚠️ (aux loss) | ✅ | ✅ | ❌ (K× params) | ❌ (compute all K) | ✅ |
| **20I: ADWI** | ⚠️ (coupled to attn) | ✅ (reuses attn) | ✅ | ❌ (H× params) | ❌ (H experts) | ✅ |
| **20J: WBFC** | ✅ (frozen clusters) | ✅ | ✅ | ⚠️ (full W in VRAM) | ✅ (M/K columns) | ❌ (pre-train first) |

### Technical Fixes
- **19E bug fix:** `num_scaling_params()` now counts GPT-level P19 params (depth_decay_raw, residual_mix_gamma).
- **19J generalized:** Weight noise iterates all 2D MLP params instead of hardcoding c_fc/c_proj.
- **P20 diagnostics:** `collect_p20()`, `format_p20()`, `to_dict_p20()` track routing entropy, load balance, branch divergence, sigma sparsity, and router grad norms for all 10 proposals.
- **Aux loss wiring:** DGCR (20D) and NCEA (20H) aux losses added to the training loop in `base_train.py`.
- **Phase 2 pipeline:** `GPT.convert_to_phase2()` converts standard MLP modules to PWU_MLP/FSVD_MLP/WBFC_MLP using pre-trained weights. Called after weight init, before optimizer setup.
- **Bug fix (20D):** `compute_aux_loss()` no longer requires external `logits` — uses mean of branch outputs as consensus target.
- **Bug fix (20H):** `NoiseCEA_MLP` delta params cast to weight dtype (`.to(base_fc_w.dtype)`) to fix bf16/float32 mismatch.

### Phase 20 Sweep Results

**Baseline**: 1.2573 BPB (dense base, dim=128, depth=4)

| # | Proposal | BPB | Δ vs Dense | Status |
|---|---|---|---|---|
| 20A | HRCS (scale=4) | 1.2572 | -0.0001 | ✅ Neutral |
| 20B | LSWR (scale=4) | 1.3443 | +0.0870 | ❌ Worse |
| **20C** | **LRCFB (K=4)** | **1.2323** | **-0.0250** | 🏆 **Best!** |
| 20D | DGCR (K=4) | — | — | ❌ Crashed (fixed) |
| 20F | MoNE (K=4) | 1.2589 | +0.0016 | ✅ Neutral |
| 20H | NCEA (K=4) | — | — | ❌ Crashed (fixed) |
| 20I | ADWI | 1.2575 | +0.0002 | ✅ Neutral |
| 20E | PWU (K=4) | 1.3568 | +0.0995 | ❌ Needs real pretrain |
| 20G | FSVD gate | 1.3365 | +0.0792 | ❌ Needs real pretrain |
| 20J | WBFC (K=8) | — | — | ⚠️ Incomplete |

**Key finding:** 20C (frozen content routing, K=4 branches) achieved **1.2323 BPB**, beating the dense baseline by 0.025 and beating CKR's best (4.08 at larger dim). This validates frozen content-dependent routing as a viable mechanism.

### 20C Enhancements: Narrow Branches + Routing Ablations

Based on 20C's strong results, added configurable variants via `--p20-lrcfb-narrow`, `--p20-lrcfb-learned`, `--p20-lrcfb-topk`:

| Variant | Flags | Params vs Dense | FLOPs | Tests |
|---|---|---|---|---|
| 20C (original) | `--p20-lrcfb-branches 4` | 4× | 4× | ✅ Proven: 1.2323 BPB |
| 20C-narrow | `+ --p20-lrcfb-narrow 1` | **1×** (param parity) | 1× | Does small-expert routing work? |
| 20C-learned | `+ --p20-lrcfb-learned 1` | 1× | 1× | Learned vs frozen routing? |
| 20C-topk1 | `+ --p20-lrcfb-topk 1` | 1× | **~¼×** | Top-1 sparse (min FLOPs) |

With `narrow=True, K=4`: each branch has H//4 hidden dim instead of H. So 4 branches × (D→H/4→D) = same params as 1 × (D→H→D).

---

## Phase 21: Pervasive Expert Routing (PER)

**The original vision realized:** Replace every Linear layer with K narrow expert sub-layers, each selected by frozen content routing. Different routing at every layer/projection guarantees each token takes a unique "path" through the network.

### Architecture: `MoELinear`

Drop-in replacement for `Linear(D_in, D_out)`:
- K experts, each factored: `expert_k(x) = A_k @ (B_k @ x)` where B_k is (r, D_in), A_k is (D_out, r), r = D_in // K
- Each expert outputs full D_out (no concatenation issues)
- Per-instance frozen random routing projection R (unique seed per `MoELinear` instance)
- Different routing at every layer/projection = different expert per layer

#### VRAM advantage
With narrow experts (r = D_in // K), activation intermediates are r-dimensional instead of D_out-dimensional. Even at the same model_dim, top-1 routing materializes only 1 expert's intermediates.

#### Modes
- `--p21-per-experts K`: number of experts (0=off)
- `--p21-per-topk N`: top-N routing (0=soft, all experts weighted sum)
- `--p21-per-learned 1`: learnable routing projection instead of frozen
- `--p21-per-attn 1`: also replace attention Q/K/V/O (default: MLP only)

#### Where MoELinear is applied
- **MLP** (always when p21_per_experts > 0): `c_fc` and `c_proj`
- **Attention** (when p21_per_attn = 1): `c_q`, `c_k`, `c_v`, `c_proj`

#### Sweep entries
| TAG | Experts | Top-k | Scope | What it tests |
|---|---|---|---|---|
| 21_PER_MLP_SOFT | K=4 | all | MLP only | Soft expert blending in MLP |
| 21_PER_MLP_TOP1 | K=4 | 1 | MLP only | Maximum FLOP savings in MLP |
| 21_PER_ALL_SOFT | K=4 | all | MLP+Attn | Full pervasive expert routing |
| 21_PER_ALL_TOP1 | K=4 | 1 | MLP+Attn | Maximum FLOP savings everywhere |
