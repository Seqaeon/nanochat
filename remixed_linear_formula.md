# RemixedLinear — Per-Token Formula & Ablation Plan

## What it computes (weight mode, single token, no MoE)

Let:
- `x ∈ ℝᶜ` — input token
- `ctx ∈ ℝᴰ` — context state (from `LocalContextStream`, D = C in practice)
- `B` — basis_size = `max(default, min(in,out)//4)`, typically **C//4**
- `r` — output gate rank, default **8**

```
Step 1  h  = LayerNorm(W_b x)         W_b ∈ ℝᴮˣᶜ          "basis projection"

Step 2  γ  = σ( MLP_gate(ctx) )       MLP_gate: C → C/2 → B  "basis gate"
        h_g = h ⊙ γ                   element-wise mask on basis dims

Step 3  y₀ = W_m h_g + b             W_m ∈ ℝᴼˣᴮ             "template mixing"

Step 4  c  = W_oc ctx                 W_oc ∈ ℝʳˣᴰ            "low-rank output gate"
        G_out = 1 + tanh(s · c Gᵀ)   G ∈ ℝʳˣᴼ, s scalar
        y  = y₀ ⊙ G_out                                       "output gate"
```

**Full formula:**
```
y = [W_m · (LN(W_b x) ⊙ σ(MLP(ctx)))] ⊙ [1 + tanh(s · (W_oc ctx) G)] + b
```

### Householder variant (+ one extra step between 1 and 2)
```
v = normalize(W_H ctx)          W_H ∈ ℝᴮˣᴰ
h = h - 2(h·v)v                 Householder reflection in basis space
```

### CKR: completely different — replaces the whole layer with a causal convolution kernel. Not derived here.

---

## FLOPs per token (square layer, in=out=C, D_ctx=C)

| Component | Op | FLOPs |
|---|---|---|
| Basis `W_b x` | C × B = C × C/4 | **0.5 C²** |
| LayerNorm | ~B | ε |
| `MLP_gate`: `Linear(C, C/2)` | C × C/2 | **0.5 C²** |
| `MLP_gate`: `Linear(C/2, B)` | C/2 × C/4 | **0.125 C²** |
| Basis gate multiply | B | ε |
| Template mixing `W_m h_g` | B × O = C/4 × C | **0.5 C²** |
| Output gate coeffs | C × r ≈ 8C | ε |
| Output gate matmul | r × O ≈ 8C | ε |
| Output gate multiply | O | ε |
| **Total RemixedLinear** | | **≈ 1.625 C²** (×2 fwd+bwd = **3.25 C²**) |
| **Dense baseline** | `W x` | **2 C²** (fwd+bwd **4 C²**) |

> [!CAUTION]
> **RemixedLinear is ~62.5% more expensive than a dense linear per matmul call.**
> The bottleneck is `MLP_gate` — the hidden layer (`C → C/2 → B`) alone costs **0.625 C²**, which is nearly as much as the structural matmuls (0.5 + 0.5 = 1.0 C²) combined.

With Householder: add `Linear(C, B)` = **+0.5 C²** → **2.125 C²**, even worse.

---

## Proposed ablations (fastest → slowest, per linear layer)

| Variant | Formula | Cost vs Dense | What's removed |
|---|---|---|---|
| **A — BasisOnly** | `y = W_m LN(W_b x)` | **50%** | both gates |
| **B — + OutputGate** | `y = W_m LN(W_b x) ⊙ (1+tanh(s·cGᵀ))` | **52%** | basis gate only |
| **C — + LinearBasisGate** | `y = W_m (LN(W_b x) ⊙ σ(W_g ctx)) ⊙ G_out` | **75%** | MLP hidden layer in gate |
| **D — Current (MLPBasisGate)** | full formula above | **162%** | nothing |
| **E — + Householder** | D + Householder reflection | **200%** | — |

> [!IMPORTANT]
> **C (LinearBasisGate)** is the prime simplification target: replace `MLP_gate(ctx) = Linear(C,C/2)→GELU→Linear(C/2,B)` with a single `Linear(C, B)`. This halves gate cost and brings total to 75% of dense (vs 162% now).

---

## Implementation plan

### Simplification: replace `basis_modulator` MLP with single linear

In `RemixedLinear.__init__`, change:
```python
# CURRENT (expensive)
self.basis_modulator = nn.Sequential(
    Linear(context_dim, basis_hidden, bias=True),
    nn.GELU(),
    Linear(basis_hidden, _gate_out_size, bias=True),
)
```
to a flag-controlled option `basis_gate_mode ∈ {mlp, linear, none}`:
```python
# NEW: linear mode
self.basis_modulator = Linear(context_dim, _gate_out_size, bias=True)
```

### Ablation sweep (p25)

Run `research_sweep.sh` with `--models remixed-linear` at depth ∈ {4, 8, 12}:

| Run name | Flags |
|---|---|
| `25_BASIS_ONLY` | `--remix-use-basis-gate 0 --remix-use-output-gate 0 --remix-use-context 0` |
| `25_OUTPUT_GATE_ONLY` | `--remix-use-basis-gate 0 --remix-use-output-gate 1 --remix-use-context 1` |
| `25_LINEAR_BASIS_GATE` | `--remix-use-basis-gate 1 --remix-use-output-gate 1 --remix-basis-gate-mode linear` |
| `25_MLP_BASIS_GATE` | current default — full RemixedLinear |

### Open questions

1. Does the basis gate (γ) actually help vs output gate alone (variant B vs D)?  
   If B ≈ D in loss, the 0.625 C² gate MLP is pure overhead.
2. Is householder worth the extra 0.5 C²? Needs direct A/B vs D.
3. At what depth does the context stream signal become strong enough to actually drive the basis gate?
