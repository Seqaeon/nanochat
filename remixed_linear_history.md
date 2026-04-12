# Iterations on Remixed-Linear: A Historical Analysis

The core objective of the past several architectural iterations has been to solve **long-sequence context degradation (T ≥ 2048)** while maintaining a stable, low loss. 

The baseline architecture (yielding a validation loss of `4.15`) heavily utilizes a static Exponential Moving Average (EMA) context state with an explicit `.detach()` between tokens, alongside factorized weight modulation (basis/output gates) inside the MLP.

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

