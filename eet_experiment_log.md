# EET Experiment Log & Catalog of Tried Ideas

This log catalogs the architectural changes and tuning attempts for the Early Exit Transformer (EET) to track what has been tried, why it was tried, and the outcomes.

---

## 🚫 Catalog of Unsuccessful Ideas (Do Not Suggest/Retry)

### 1. Phase 1 Warmup & Phase 2 Exploration (Reconstruction Loss)
- **Concept:** Warm up the backbone dense, then train with a reconstruction loss where a translator predicts final layer states from early exit states.
- **Outcome:** Made performance worse and did not improve routing compared to training Phase 3 (committed routing) directly from scratch.
- **Status:** **Abandoned**. We now run Phase 3 only (`--eet-warmup-frac 0.0 --eet-explore-frac 0.0`).

### 2. Exit Adapters (`--eet-exit-adapter-rank > 0`)
- **Concept:** Low-rank adapters to map early-exit hidden representations to the shared LM head subspace.
- **Outcome:** Slowed the model down significantly and resulted in worse val_bpp performance.
- **Status:** **Abandoned** (set to `0`).

### 3. Router After Block 1 (`--eet-router-after-block 1`)
- **Concept:** Let the global router see post-attention/FFN context from block 1 instead of raw embeddings ($x_0$).
- **Outcome:** Did not improve performance.
- **Status:** **Abandoned** (set to `0`).

### 4. Layer-Weighted Losses
- **Concept:** Weigh the loss by the exit depth of each token.
- **Outcome:** Did not improve performance.
- **Status:** **Abandoned**.

### 5. FFN-Skip (A³D Mode) (`--eet-ffn-skip 1`)
- **Concept:** Skip FFNs while retaining full attention across all tokens at all layers.
- **Outcome:** Increased step time (`dt`) by ~150ms and performed significantly worse (val_bpp of ~1.40 vs ~1.03 for full block skip).
- **Status:** **Abandoned** (set to `0`).

---

## 🚀 Active Configurations & Hyperparameter Tuning

### 1. Global Router & CE-Guided Loss (Phase 3 Only)
- **Flags:** `--use-eet 1 --eet-global-router 1 --eet-loss-variant ce_guided --eet-compute-skip 1`
- **Schedule:** `--eet-capacity-schedule bell`
- **Warmup/Explore:** `--eet-warmup-frac 0.0 --eet-explore-frac 0.0`
- **Current Baseline Gap (d8):** ~0.06 val_bpp gap from dense baseline.

### 2. Router Learning Rate Tuning (`--eet-router-lr-mult`)
- **Concept:** Gating/routing networks often need a decoupled learning rate from the rest of the network to break the constant-function equilibrium.
- **Implementation:** Decoupled and independent from backbone LR. Set via `--eet-router-lr-mult <float>` (defaults to `5.0` relative to `gate_lr`).

### 3. EET Backbone Learning Rate Tuning (`--eet-model-lr-mult`)
- **Concept:** When capacity is restricted (e.g. 10% active tokens at deep layers), the optimal backbone learning rate might differ from the dense baseline's optimal LR.
- **Implementation:** Scales all non-router LRs (Muon + AdamW parameters) when EET is active. Set via `--eet-model-lr-mult <float>` (defaults to `1.0`).
