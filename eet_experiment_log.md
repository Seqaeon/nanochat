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
