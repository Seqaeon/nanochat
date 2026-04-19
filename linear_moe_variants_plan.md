# Linear Layer Variants — Implementation Plan

## Overview

Three new experimental linear layer variants, all comparable to the same dense baseline.

| Class | Mechanism | Weight change | Gate scope |
|---|---|---|---|
| `SlicedWeightLinear` | Product Key router selects columns from a wide bank | Wide bank, column selection | per_token / per_block / global |
| `FoldedModulationLinear` | Sum consecutive groups, gate the folded dims | None (dense weight on reduced input) | per_layer / per_block / global |
| `SequenceGatedLinear` | Sequence-pooled gate on full residual stream | None (identical to dense) | per_layer / per_block / global |

---

## 1. `SlicedWeightLinear`

**Concept:** One wide weight bank `(out_features, big_dim)` where `big_dim = in_features * reduction_scale`. A Product Key router selects `n_selected` columns per token to form the effective weight slice. No separate expert matrices — just one bank that gets indexed into.

### Parameter Shapes

```
weight_bank:  (out_features, big_dim)        big_dim = in_features * reduction_scale
router_A:     Linear(in_features, √big_dim)
router_B:     Linear(in_features, √big_dim)
```

Router cost: `in × 2√big_dim` vs `in × big_dim` for a full gate — roughly `√reduction_scale`× cheaper.

### Forward Pass

```
1. logits_A = router_A(route_input)                    # (B, T, √big_dim) or (B, 1, √big_dim)
   logits_B = router_B(route_input)                    # (B, T, √big_dim) or (B, 1, √big_dim)

2. topk_A, topk_B → outer product → flat column indices
   n_selected = max(min_select, big_dim // reduction_scale)
   topk_A = isqrt(n_selected)
   topk_B = n_selected // topk_A

3. W_slice = weight_bank[:, selected_indices]           # (B, T, out, n_selected)

4. out = einsum('bti,btoi->bto', x, W_slice)
```

`n_selected` must equal `in_features` so that x and W_slice are compatible — this holds by construction since `big_dim = in_features * R` and we select `in_features` columns out of it.

### Scope Variants

The Product Key router naturally produces per-token selections, but the routing input (`route_input`) can be scoped to reduce cost and study coarser routing granularity.

**`per_token`** — default. `route_input = x`, shape `(B, T, C)`. Each position selects its own weight slice independently.

**`per_block`** — `route_input` is the sequence-pooled block input, computed once at block entry and injected. All positions in the block share the same column selection.

```python
# In Block.forward / RemixedBlock.forward:
if self.sliced_weight_scope == 'per_block':
    pool = x.mean(dim=1, keepdim=True)              # (B, 1, C)
    self._block_route_input = pool                  # injected into SlicedWeightLinear.forward
# Each SlicedWeightLinear expands selection back to (B, T, ...):
logits_A = router_A(block_route_input).expand(-1, T, -1)
```

**`global`** — `route_input` computed once in `GPT.forward` from the token embedding output, shared across all blocks and all layers. Maximum sharing, minimum routing overhead.

```python
# In GPT.forward, after embedding:
if self.sliced_weight_scope == 'global':
    global_route = x_embed.mean(dim=1, keepdim=True)          # (B, 1, C)
    global_route = self.global_route_proj(global_route)        # optional projection
# Thread through block.forward(x, global_route=global_route)
```

For `per_block` and `global`, the EMA balance loss uses the shared (B, 1) logits instead of (B, T) — cheaper to compute and still enforces uniform bank coverage over training steps.

### Minimum Selection Clamping

```python
n_select_raw = big_dim // reduction_scale   # = in_features
n_selected   = max(config.sliced_weight_min_select, n_select_raw)
# Refactorize for Product Key
topk_A = int(math.isqrt(n_selected))
topk_B = n_selected // topk_A
```

The clamp lets you enforce a floor (e.g. 128) regardless of model size, preventing degenerate selection at very high `reduction_scale`.

### Quantile Balance Auxiliary Loss

Track EMA usage at the sub-router level — 2√big_dim counters total, not 512.

```python
# EMA update (detached, no gradient)
mean_A = softmax(logits_A).mean(dim=(0,1))         # (√big_dim,)
usage_A = 0.99 * usage_A + 0.01 * mean_A.detach()

# Imbalance: log-ratio from uniform
imbalance_A = (usage_A / (1 / n_keys_A)).log()

# Aux loss: penalise routing weight to already-overused keys
aux_A = (softmax(logits_A) * imbalance_A.detach()).mean()
aux_loss = aux_A + aux_B
```

Apply independently to both sub-routers. Balanced A × balanced B ≈ uniform coverage over all `big_dim` combinations.

**Optional independence penalty** (catches correlated A/B selections):

```python
expected_joint = mean_A.unsqueeze(1) * mean_B.unsqueeze(0)   # (√D, √D)
actual_joint   = scatter-accumulate from selected combo indices
independence_loss = (actual_joint - expected_joint.detach()).pow(2).mean()
```

In practice this is mild if aux loss is on; include it as an optional flag.

### Initialization

| Parameter | Init |
|---|---|
| `weight_bank` | Kaiming uniform |
| `router_A`, `router_B` weights | **Zeros** → uniform softmax at step 0 |
| `usage_A`, `usage_B` buffers | `1 / n_keys` (perfect balance at start) |

### Config Fields

```python
use_sliced_weight: bool = False
sliced_weight_reduction_scale: int = 8
sliced_weight_min_select: int = 128
sliced_weight_scope: str = 'per_token'               # 'per_token' | 'per_block' | 'global'
sliced_weight_balance_coeff: float = 0.01
sliced_weight_balance_ema: float = 0.99
sliced_weight_independence_penalty: float = 0.0      # 0 = disabled
```

### Integration Points

- Add as option in `RemixedFeedForward.__init__` alongside `LinearMoE`
- Collect `_last_balance_loss` in `GPT.forward` the same way `_last_orth_loss` is aggregated via `modules()` scan
- Aux loss only added when `loss_reduction == 'mean'`
- `Block.forward` / `RemixedBlock.forward`: if scope is `per_block`, compute `x.mean(dim=1, keepdim=True)` before attn and store as `_block_route_input`
- `GPT.forward`: if scope is `global`, compute from embedding output and thread through all block calls via `global_route` kwarg
- Fall back to `per_token` (local x) when no external route input is provided

---

## 2. `FoldedModulationLinear`

**Concept:** Fold `x` by summing consecutive groups of `reduction_scale` dims — cheap compression, no extra weight. Gate the folded representation (B, T, C//R) to produce modulation weights that scale each group's contribution before the actual dense linear. The dense weight matrix is unchanged.

### Folding Operation

```
x: (B, T, C)
x_fold = x.view(B, T, C//R, R).sum(dim=-1)   # (B, T, C//R)
```

Index 0 of `x_fold` = sum of x dims [0..R-1], index 1 = sum of x dims [R..2R-1], etc.

### Gate Prediction

```
gate = sigmoid(gate_proj(pool))               # (B, C//R)
pool = x_fold.mean(dim=1)                     # (B, C//R) — sequence average of folded x
```

Gate index `i` multiplies with `x_fold[:, :, i]` — directly controlling how much each group of R input dims contributes.

### Full Forward

```
1. x_fold = x.view(B, T, C//R, R).sum(-1)    # (B, T, C//R)
2. pool    = x_fold.mean(dim=1)               # (B, C//R)
3. gate    = sigmoid(gate_proj(pool))          # (B, C//R)
4. x_mod   = x_fold * gate.unsqueeze(1)       # (B, T, C//R)  — gated folded input
5. out     = F.linear(x_mod, weight, bias)    # weight: (out_features, C//R)
```

Note: `weight` is `(out_features, C//R)`, not `(out_features, C)`. This is a genuine parameter reduction from the folding. The dense baseline has `(out, C)` — this has `(out, C//R)` plus a small gate predictor.

### Scope Variants

**`per_layer`** — `gate_proj` lives inside each `FoldedModulationLinear`. Gate computed locally.

**`per_block`** — Gate computed once at block entry from `norm(x_block_input)`, stored on block, injected into all `FoldedModulationLinear` layers in that block via a `block_gate` kwarg.

```python
# In Block.forward / RemixedBlock.forward:
if self.folded_mod_scope == 'per_block':
    pool = norm(x).mean(dim=1)              # (B, C)
    self._block_gate = self.gate_proj(pool) # (B, C//R) — shared across layers
# Pass to ffn:
ffn(x, block_gate=self._block_gate)
```

**`global`** — Gate computed once in `GPT.forward` after the token embedding + norm, threaded through all block calls. Each layer uses the same gate for the entire forward pass.

```python
# In GPT.forward, after embedding:
if self.folded_mod_scope == 'global':
    global_gate = self.global_gate_proj(x_embed.mean(dim=1))  # (B, C//R)
# Pass into every block.forward(x, global_gate=global_gate)
```

### Initialization

| Parameter | Init |
|---|---|
| `weight` | Kaiming uniform (same as dense) |
| `gate_proj` last layer weight | **Zeros** → gate = sigmoid(0) = 0.5 at init |
| `gate_proj` last layer bias | **Zeros** |

Gate starting at 0.5 means x_mod = 0.5 * x_fold — a small uniform attenuation. Training departs cleanly from there.

### Config Fields

```python
use_folded_mod: bool = False
folded_mod_reduction_scale: int = 8
folded_mod_scope: str = 'per_layer'        # 'per_layer' | 'per_block' | 'global'
folded_mod_gate_act: str = 'sigmoid'       # 'sigmoid' | 'tanh_centered' (= 1 + tanh(x))
folded_mod_gate_bottleneck: int = 0        # 0 = direct C//R → C//R; N = bottleneck dim
```

### Integration Points

- `RemixedFeedForward.__init__`: swap `c_fc` / `c_proj` when `use_folded_mod=True`
- `Block.forward` / `RemixedBlock.forward`: add `_block_gate` computation pre-attn if scope is `per_block`
- `GPT.forward`: add `global_gate` computation post-embedding if scope is `global`
- Inject gate via `block_gate` / `global_gate` optional kwargs; fall back to local if `None`

---

## 3. `SequenceGatedLinear`

**Concept:** Standard dense linear, weights completely unchanged from baseline. Before the linear, compute a `(B, C)` gate from the sequence-pooled input and broadcast across T. Pure activation-space modulation — no weight-space changes at all.

### Architecture

```
weight:     (out_features, in_features)     # identical to dense baseline
gate_proj:  Linear(in_features, in_features, bias=True)
```

Optional cheap variant with bottleneck:
```
gate_proj:  Sequential(Linear(C, C//8), GELU, Linear(C//8, C))
```

### Forward Pass

```
1. pool  = x.mean(dim=1)                          # (B, C) — sequence average
2. gate  = sigmoid(gate_proj(pool)).unsqueeze(1)  # (B, 1, C)
3. out   = F.linear(x * gate, weight, bias)       # gate broadcast over T
```

**`seq_gate_position='post'` variant:**
```
3. out  = F.linear(x, weight, bias)               # normal linear first
4. out  = out * gate_out                          # gate on output channels
   gate_out: gate_proj maps to out_features dim
```

Pre-gating controls which input channels matter per sequence. Post-gating controls which output channels to amplify. Both are worth running.

### Scope Variants

Same three scopes as `FoldedModulationLinear`:

- **`per_layer`**: Each `SequenceGatedLinear` computes its own pool and gate
- **`per_block`**: One gate per block, shared by attn projections and ffn in that block
- **`global`**: One gate for the entire forward pass, computed from embeddings

For `per_block` and `global`, the gate is full `C`-dim and is simply passed in as an optional argument — no architectural surgery needed since it's just a multiplicative factor.

### Initialization

| Parameter | Init |
|---|---|
| `weight` | Kaiming uniform (same as dense) |
| `gate_proj` last layer weight | **Zeros** → gate = 0.5 at init |
| `gate_proj` last layer bias | **Zeros** |

### Config Fields

```python
use_seq_gate: bool = False
seq_gate_scope: str = 'per_layer'          # 'per_layer' | 'per_block' | 'global'
seq_gate_bottleneck: int = 0               # 0 = no bottleneck; N = bottleneck dim
seq_gate_act: str = 'sigmoid'              # 'sigmoid' | 'tanh_centered'
seq_gate_position: str = 'pre'            # 'pre' | 'post'
```

### Integration Points

- Swaps in directly for any `Linear` in `MLP` / `FeedForward` — identical interface
- For `per_block`: store `_block_gate` on the block object, pass to ffn and optionally attn projections
- For `global`: thread through `GPT.forward` same as `FoldedModulationLinear`

---

## Experiment Matrix

| Variant | Ablations | Key comparison |
|---|---|---|
| `SlicedWeightLinear` | scope × {per_token, per_block, global}, reduction_scale × {4, 8, 16}, min_select clamp on/off, independence penalty on/off | vs dense, vs LinearMoE |
| `FoldedModulationLinear` | scope × {per_layer, per_block, global}, gate_act × {sigmoid, tanh_centered} | vs dense, vs SequenceGatedLinear |
| `SequenceGatedLinear` | scope × {per_layer, per_block, global}, position × {pre, post}, bottleneck on/off | vs dense — cleanest ablation since weights are identical |

---

## Shared Implementation Notes

### Gate Initialization Rule
Always zero-init the **last linear layer** of any gate predictor:
- `sigmoid(0) = 0.5` → mild uniform attenuation at step 0
- `1 + tanh(0) = 1.0` → identity at step 0 (better for post-gating)

This ensures training departs from a near-baseline state regardless of variant.

### Scope Injection Pattern
Avoid threading gate tensors through every function signature. Preferred pattern:

```python
# Store on block object
block._block_gate = computed_gate    # set in Block.forward before sub-layers
# Each layer reads it optionally
def forward(self, x, block_gate=None):
    gate = block_gate if block_gate is not None else self._compute_local_gate(x)
```

This keeps sub-layer signatures clean while supporting all three scope modes.

### Aux Loss Collection
Follow the existing `_last_orth_loss` pattern — store on the module, collect in `GPT.forward` via `modules()` scan:

```python
# In GPT.forward, after main loss:
if self.sliced_weight_balance_coeff > 0:
    bal_terms = [m._last_balance_loss for m in self.modules()
                 if isinstance(m, SlicedWeightLinear) and m._last_balance_loss is not None]
    if bal_terms:
        loss = loss + self.config.sliced_weight_balance_coeff * torch.stack(bal_terms).mean()
```

Only add when `loss_reduction == 'mean'`.
