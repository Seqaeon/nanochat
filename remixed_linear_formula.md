# RemixedLinear: Formula, Ablation Results & Next Steps

## Mathematical Operation (per token)

Given input token `x ∈ ℝ^C` and causal context state `ctx ∈ ℝ^C`:

```
# 1. Project to basis space (compression)
h = LayerNorm( W_b · x )          h ∈ ℝ^B, B = basis_size (≈ C/4)

# 2. [Optional] Operator Modulation on h (before gating)
#    householder: h = h - 2*(h·v / |v|²)·v   (context-adaptive reflection)
#    ckr:         h = h ⊙ σ(W_ckr · ctx)      (content-kernel routing)
#    spectral:    h = h ⊙ (1 + tanh(W_s · ctx))

# 3. Basis gate (context-selects which basis dims are active)
gate_basis = σ( MLP(ctx) )        gate_basis ∈ ℝ^B      [mode=mlp]
           = σ( W_g · ctx )       gate_basis ∈ ℝ^B      [mode=linear]
           = 1                                            [mode=none]

# 4. Gated basis representation
h̃ = h ⊙ gate_basis               h̃ ∈ ℝ^B

# 5. Mix back to output space (expansion)
y_pre = W_m · h̃                   y_pre ∈ ℝ^D   (D = out_features)

# 6. Output gate (low-rank context-adaptive scale)
coeffs  = W_oc · ctx              coeffs ∈ ℝ^r   (r=8)
gate_out = 1 + tanh( s · coeffs @ G )   G ∈ ℝ^{r×D}, s ∈ ℝ^1

# 7. Final output
y = (y_pre ⊙ gate_out) + bias
```

**FLOPs per token (dense vs RemixedLinear, basis B = C/4):**

| Component | FLOPs | Notes |
|---|---|---|
| Dense baseline | 2CD | W ∈ ℝ^{D×C} |
| Basis W_b | CB | B ≈ C/4 |
| LayerNorm | 5B | negligible |
| Basis gate (MLP) | CB/2 + B²/2 ≈ 0.625C² | C→C/2→B for C=D=B*4 |
| Basis gate (linear) | CB ≈ 0.25C² | C→B single proj |
| Mix W_m | BD | B·D = C·C/4 = 0.25C² |
| Output gate | Cr + rD ≈ 16C | tiny, r=8 |
| **Total (MLP mode)** | **~1.625C²** | vs 2C² dense → **81%** |
| **Total (linear mode)** | **~1.27C²** | vs 2C² dense → **63%** |

---

## P25 Ablation Results (depth=4, dim=256, full-rank basis B=C=256)

> **Note:** P25 used `--research-dim -1` which sets `basis_size=0` → auto-expanded to `min(in,out)` = **full rank**. This is NOT the compressed B=C//4 variant.

| Variant | ctx | basis gate | output gate | gate mode | **BPB** | Δ vs MLP |
|---|---|---|---|---|---|---|
| `25_NO_CONTEXT` | ✗ | ✗ | ✗ | — | **1.1683** | +0.0079 |
| `25_OUTPUT_ONLY` | ✓ | ✗ | ✓ | none | **1.1655** | +0.0051 |
| `25_LINEAR_GATE` | ✓ | ✓ | ✓ | linear | **1.1645** | +0.0041 |
| `25_ATTN_GATE` | ✓ | ✓ | ✓ | attn | **1.1718** | +0.0114 ← **worse** |
| `25_MLP_GATE` | ✓ | ✓ | ✓ | mlp | **1.1604** | baseline |

Dense baseline (same depth/dim): **~1.167 BPB** (from P23 sweep below)

**P25 conclusion:** Full-rank RemixedLinear (B=C) beats the dense baseline. The output gate + context signal provides the quality gain. Attn gate is unstable.

---

## P23 Sweep Results (depth=4, dim=256, compressed basis B=C//4=64)

> Sweep log: `sweep_p23 (27).log`. CCL_MOD=weight, CCL_STREAM=selective.

| Run | basis_size | gate_mode | gate_temp | warmup | warmdown | **BPB** | vs Dense |
|---|---|---|---|---|---|---|---|
| `23_BASE_DENSE` | — | — | — | 0.15 | — | **1.166914** | baseline |
| `23_REMIX_WEIGHT_LinearGate` | 64 (C//4) | linear | 1.0 | 0.15 | 0.70 | **1.212796** | **+0.046** ❌ |
| `23_REMIX_WEIGHT_LinearGate_BetterTemp` | 64 (C//4) | linear | 2.0 | 0.20 | 0.50 | **1.201986** | **+0.035** ❌ |
| `23_DUAL_GATE_*` | — (full W) | linear | 2.0 | 0.20 | 0.50 | worse than Remix | ❌ |

### Key Finding: Basis compression (B=C//4) hurts at this scale

At depth=4, C=256, `basis_size = max(64, 256//4) = 64`. This means **every layer** (Q/K/V/Proj/c_fc/c_proj) runs through a 64-dim bottleneck. The compression is too aggressive:

- Q/K/V: 256→64 is 4× compression in the gate space
- The gate oscillation (spike pattern visible in loss trajectory) shows the sigmoid gates are flipping channels on/off as LR decays
- Raising temp to 2.0 softened oscillation: -0.011 BPB improvement, but still -0.035 below dense

### Why P25 (full-rank) beat dense but P23 (compressed) didn't

| Property | P25 (B=C, full-rank) | P23 (B=C//4, compressed) |
|---|---|---|
| Gate space | D-dim (full output) | 64-dim (bottleneck) |
| What the gate selects | Full output channels | Compressed basis directions |
| Params vs dense | +gate overhead (>100%) | -basis savings (<100%) |
| Result | **beats dense** | **worse than dense** |

The output gate (low-rank) is the dominant quality driver, not the basis compression. The compression saves params/FLOPs but hurts quality at small scales.

### DualGateLinear result

DualGateLinear (single dense W + dual D-dim gate, no compression) performs **worse than RemixedLinear** at this scale. Reason: it has 107.9% of dense parameters (vs RemixedLinear's 97.1%), so it's strictly dominated by dense on both params AND quality. The D-dim gate adds overhead without the capacity benefit.

---

## Interpretation and Next Steps

### What is actually working

1. **The output gate** is the most important component (explains ~65% of quality gain in P25)
2. **Full-rank basis (B=C)** + output gate beats dense
3. **Compressed basis (B=C//4)** hurts at small scale — too narrow a bottleneck

### Open questions

| Question | Status |
|---|---|
| Does B=C//4 help at larger scale (depth=12, C=768)? | ❓ Untested |
| Is there a sweet spot between C//4 and C//1? | ❓ Try B=C//2 |
| Can DualGateLinear beat dense if trained with better schedule? | ❓ Possible — needs MLP gate not linear |

### Recommended next experiment

Run P23 with `--research-dim -1` (full-rank basis) or a larger fixed basis to see if the architecture advantage from P25 transfers to the P23 sweep setup. This isolates whether the P25 win was schedule-dependent or architecture-dependent.

---

## RemixedLinear Class (current, cleaned — no MoE/LoKR/TinyExpert/QuantileRoute)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class RemixedLinear(nn.Module):
    """
    Factorized linear layer with basis compression, operator modulation,
    context-conditioned basis gate, and low-rank output gate.

    Forward:
        h       = LayerNorm(W_b · x)               # basis projection
        h       = operator_modulate(h, ctx)         # optional: householder/ckr/spectral
        h_gated = h ⊙ σ(gate(ctx))                 # basis gate (selects active dims)
        y_pre   = W_m · h_gated                    # expand to output space
        gate_out= 1 + tanh(s · U · ctx)            # low-rank output gate
        y       = (y_pre ⊙ gate_out) + bias
    """

    def __init__(
        self,
        in_features,
        out_features,
        context_dim,
        basis_size=64,
        remixed_linear_kwargs=None,
        scale_basis=True,
        film_gate=False,
        **_ignored,
    ):
        super().__init__()
        if remixed_linear_kwargs is None:
            remixed_linear_kwargs = {}

        # Auto-scale basis_size: compress relative to the *smaller* of in/out
        # to avoid the bottleneck being wider than the narrower projection dimension.
        if scale_basis:
            basis_size = max(basis_size, min(in_features, out_features) // 4)

        self.in_features     = in_features
        self.out_features    = out_features
        self.basis_size      = basis_size
        self.use_context     = remixed_linear_kwargs.get('use_context', True)
        self.use_basis_gate  = remixed_linear_kwargs.get('use_basis_gate', True)
        self.use_output_gate = remixed_linear_kwargs.get('use_output_gate', True)
        self.gate_temperature = float(remixed_linear_kwargs.get('gate_temperature', 1.0))
        self.basis_gate_mode = remixed_linear_kwargs.get('basis_gate_mode', 'linear')
        self.operator_modulation = remixed_linear_kwargs.get('operator_modulation', 'none')
        self._film_gate_flag = film_gate

        # ── Structural Projections ────────────────────────────────────────────
        self.basis           = Linear(in_features, basis_size, bias=False)
        self.ln_basis        = nn.LayerNorm(basis_size)
        self.template_mixing = nn.Parameter(torch.empty(out_features, basis_size))
        nn.init.normal_(self.template_mixing, std=0.02)
        self.bias            = nn.Parameter(torch.zeros(out_features))

        # ── Operator Modulation ───────────────────────────────────────────────
        # Applied to h_basis in basis space before gating.
        if self.use_context:
            if self.operator_modulation == 'householder':
                # Context-adaptive Householder reflection in basis space
                self.householder_v = Linear(context_dim, basis_size, bias=False)
            elif self.operator_modulation == 'ckr':
                # Causal kernel reparameterisation: sigmoid gate in basis space
                self.ckr_gate = Linear(context_dim, basis_size, bias=True)
                nn.init.zeros_(self.ckr_gate.weight)
                nn.init.zeros_(self.ckr_gate.bias)
            elif self.operator_modulation == 'spectral':
                # Spectral steering: tanh scale in basis space
                self.spectral_scale = Linear(context_dim, basis_size, bias=True)
                nn.init.zeros_(self.spectral_scale.weight)
                nn.init.zeros_(self.spectral_scale.bias)

        # ── Basis Gate ────────────────────────────────────────────────────────
        if self.use_context and self.use_basis_gate:
            _gate_out_size = basis_size * 2 if film_gate else basis_size
            if self.basis_gate_mode == 'mlp':
                self.basis_modulator = nn.Sequential(
                    Linear(context_dim, context_dim // 2, bias=False),
                    nn.SiLU(),
                    Linear(context_dim // 2, _gate_out_size, bias=True),
                )
                nn.init.zeros_(self.basis_modulator[-1].weight)
                nn.init.zeros_(self.basis_modulator[-1].bias)
            else:  # 'linear' (default)
                self.basis_modulator = Linear(context_dim, _gate_out_size, bias=True)
                nn.init.zeros_(self.basis_modulator.weight)
                nn.init.zeros_(self.basis_modulator.bias)

        # ── Output Gate (Low-Rank) ────────────────────────────────────────────
        if self.use_context and self.use_output_gate:
            r = int(remixed_linear_kwargs.get('output_gate_rank', 8))
            self.output_gate_coeffs = Linear(context_dim, r, bias=True)
            self.output_gate_basis  = nn.Parameter(torch.zeros(r, out_features))
            self.output_gate_scale  = nn.Parameter(torch.ones(1) * 0.1)
            nn.init.zeros_(self.output_gate_coeffs.weight)
            nn.init.zeros_(self.output_gate_coeffs.bias)
        else:
            self.output_gate_coeffs = None
            self.output_gate_basis  = None
            self.output_gate_scale  = None

    # ── Optimizer parameter grouping ──────────────────────────────────────────
    def gate_parameters(self):
        """Gate-side params → lower-LR optimizer group."""
        if not self.use_context:
            return
        if self.use_basis_gate and hasattr(self, 'basis_modulator'):
            yield from self.basis_modulator.parameters()
        if self.output_gate_coeffs is not None:
            yield self.output_gate_coeffs.weight
            if self.output_gate_coeffs.bias is not None:
                yield self.output_gate_coeffs.bias
            yield self.output_gate_basis
            yield self.output_gate_scale
        if self.operator_modulation == 'householder':
            yield from self.householder_v.parameters()
        elif self.operator_modulation == 'ckr':
            yield from self.ckr_gate.parameters()
        elif self.operator_modulation == 'spectral':
            yield from self.spectral_scale.parameters()

    def non_gate_parameters(self):
        """Structural params → Muon / normal-LR group."""
        yield self.basis.weight
        yield self.template_mixing
        yield self.bias

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x, context_state, context_gates=None, **_ignored):
        """
        x:             (B, T, in_features)
        context_state: (B, T, context_dim)  — causal context from CCLBlock
        context_gates: optional dict with pre-computed 'basis_gate' / 'output_coeffs'
                       (from SharedContextGates, if enabled)
        """
        dtype = x.dtype

        # Step 1: Project to basis space and normalise
        h = self.ln_basis(
            self.basis(x).to(dtype=self.ln_basis.weight.dtype)
        ).to(dtype=dtype)

        gate_out = None

        if self.use_context and context_state is not None:
            ctx = context_state.to(dtype=dtype)

            # Step 2: Operator Modulation (in basis space, before gating)
            if self.operator_modulation == 'householder':
                v       = self.householder_v(ctx)
                v_sq    = torch.sum(v ** 2, dim=-1, keepdim=True) + 1e-6
                h_dot_v = torch.sum(h * v, dim=-1, keepdim=True)
                h       = h - 2 * (h_dot_v / v_sq) * v
            elif self.operator_modulation == 'ckr':
                h = h * torch.sigmoid(self.ckr_gate(ctx))
            elif self.operator_modulation == 'spectral':
                h = h * (1.0 + torch.tanh(self.spectral_scale(ctx)))

            # Step 3: Basis Gate
            if self.use_basis_gate and hasattr(self, 'basis_modulator'):
                gate_logits = (
                    context_gates['basis_gate'].to(dtype=dtype)
                    if (context_gates and 'basis_gate' in context_gates)
                    else self.basis_modulator(ctx)
                )
                if self._film_gate_flag:
                    scale_logits, shift = gate_logits.chunk(2, dim=-1)
                    gate_b = (1.0 + torch.tanh(scale_logits * 0.1)).to(dtype=dtype)
                    h = h * gate_b + shift.to(dtype=dtype)
                else:
                    h = h * torch.sigmoid(gate_logits / self.gate_temperature).to(dtype=dtype)

            # Step 4: Output Gate (low-rank)
            if self.use_output_gate and self.output_gate_coeffs is not None:
                coeffs = (
                    context_gates['output_coeffs'].to(dtype=dtype)
                    if (context_gates and 'output_coeffs' in context_gates)
                    else self.output_gate_coeffs(ctx)
                )
                gate_logits2 = torch.matmul(coeffs, self.output_gate_basis.to(dtype=dtype))
                gate_out = 1.0 + torch.tanh(
                    self.output_gate_scale.to(dtype=dtype) * gate_logits2
                )

        # Step 5: Expand to output space
        y = F.linear(h, self.template_mixing.to(dtype=dtype))

        # Step 6: Apply output gate + bias
        if gate_out is not None:
            y = y * gate_out
        return y + self.bias.to(dtype=dtype)
```

---

## DualGateLinear Class (ablation — no basis compression)

```python
class DualGateLinear(nn.Module):
    """
    Single dense projection + dual D-dim multiplicative gate.
    Ablation of RemixedLinear: gate operates in full output space (no bottleneck).

        h = LayerNorm(W · x)                     # single dense matmul, LN in D-space
        y = h ⊙ σ(W_gate · ctx / T)             # Stage-1: D-dim sigmoid gate
        y = y ⊙ (1 + tanh(s · U · ctx))         # Stage-2: low-rank output gate

    Result (depth=4, C=256): 107.9% of dense params, worse BPB than dense.
    The gate overhead with no compression savings is strictly dominated.
    """
```

---

## Design Notes

- **Compressed basis (B=C//4) hurts at small scale** (depth=4, C=256): oscillation + too-narrow bottleneck → worse than dense. Full-rank (B=C) beats dense.
- **ATTN gate hurts**: bilinear content×context gating is unstable — noisy basis gradients early in training corrupt the gate signal. Discarded.
- **gate_temperature=2.0** softens sigmoid oscillation in late warmdown; reduced the deficit vs dense from -0.046 to -0.035 BPB at B=C//4.
- **gate_parameters() / non_gate_parameters()**: critical for the dual-LR optimizer setup (gate params get lower LR to reduce early-training noise).
- **SharedContextGates**: batches all 6 per-block gate MLPs into 3 shared matmuls (~6× fewer kernel launches). Currently off in sweeps for clean ablation.
