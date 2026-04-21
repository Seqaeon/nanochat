# RemixedLinear: Formula, Ablation Results & Next Steps

## Mathematical Operation (per token)

Given input token `x ∈ ℝ^C` and causal context state `ctx ∈ ℝ^C`:

```
# 1. Project to basis space (compression)
h = LayerNorm( W_b · x )          h ∈ ℝ^B, B = basis_size (≈ C/4)

# 2. Basis gate (context-modulates which basis dims are active)
gate_basis = σ( MLP(ctx) )        gate_basis ∈ ℝ^B      [mode=mlp]
           = σ( W_g · ctx )       gate_basis ∈ ℝ^B      [mode=linear]
           = σ( W_c·h ⊙ W_k·ctx )                       [mode=attn]
           = 1                                            [mode=none]

# 3. Gated basis representation
h̃ = h ⊙ gate_basis               h̃ ∈ ℝ^B

# 4. Mix back to output space (expansion)
y_pre = W_m · h̃                   y_pre ∈ ℝ^D   (D = out_features)

# 5. Output gate (low-rank context-adaptive scale)
coeffs  = W_oc · ctx              coeffs ∈ ℝ^r   (r=8)
gate_out = 1 + tanh( s · coeffs @ G )   G ∈ ℝ^{r×D}, s ∈ ℝ^1

# 6. Final output
y = (y_pre ⊙ gate_out) + bias
```

**FLOPs per token (dense vs RemixedLinear):**

| Component | FLOPs | Notes |
|---|---|---|
| Dense baseline | 2CD | W ∈ ℝ^{D×C} |
| Basis W_b | CB | B ≈ C/4 |
| LayerNorm | 5B | negligible |
| Basis gate (MLP) | CB/2 + B²/2 ≈ 0.625C² | C→C/2→B for C=D=B*4 |
| Basis gate (linear) | CB ≈ 0.25C² | C→B single proj |
| Basis gate (attn) | 2CB ≈ 0.5C² | content + context proj |
| Mix W_m | BD | B·D = C·C/4 = 0.25C² |
| Output gate | Cr + rD ≈ 16C | tiny, r=8 |
| **Total (MLP mode)** | **~1.625C²** | vs 2C² dense → **81%** |
| **Total (linear mode)** | **~1.27C²** | vs 2C² dense → **63%** |
| **Total (none mode)** | **~0.5C²** | vs 2C² dense → **25%** |

---

## P25 Ablation Results (depth=4, dim=256, weight modulation)

| Variant | ctx | basis gate | output gate | gate mode | **BPB** | Δ vs MLP |
|---|---|---|---|---|---|---|
| `25_NO_CONTEXT` | ✗ | ✗ | ✗ | — | **1.1683** | +0.0079 |
| `25_OUTPUT_ONLY` | ✓ | ✗ | ✓ | none | **1.1655** | +0.0051 |
| `25_LINEAR_GATE` | ✓ | ✓ | ✓ | linear | **1.1645** | +0.0041 |
| `25_ATTN_GATE` | ✓ | ✓ | ✓ | attn | **1.1718** | +0.0114 ← **worse** |
| `25_MLP_GATE` | ✓ | ✓ | ✓ | mlp | **1.1604** | baseline |

### Interpretation

**What these results tell us:**

1. **MLP_GATE is best, but only marginally** — the full MLP gate achieves 1.1604 BPB vs 1.1683 for NO_CONTEXT, a 0.68% relative gap. At depth=4 this likely overestimates the gap vs deeper models (context has more signal to learn from at scale).

2. **ATTN_GATE is WORSE than NO_CONTEXT** (1.1718 vs 1.1683) — the bilinear content×context gating adds instability at this scale. The content branch (h_basis) adds noise from the partially-learned basis early in training, making the gate harder to learn than a pure context gate. **Discard attn mode.**

3. **LINEAR_GATE ≈ MLP_GATE − 0.0041 BPB** — dropping the MLP hidden layer (C→C/2→B becomes C→B) costs only 0.0041 BPB while saving ~0.375C² FLOPs per linear layer. At depth=12+ with 6 RemixedLinear per block this is a meaningful latency win.

4. **OUTPUT_ONLY costs 0.0051 BPB** — the output gate alone (with no basis selection) accounts for ~65% of the total quality gain from MLP gating. The basis gate adds only 0.001 additional BPB gain over OUTPUT_ONLY.

5. **Context itself (NO_CONTEXT→OUTPUT_ONLY) = 0.0028 BPB gain** — context information alone, even without basis gating, helps the model adapt its output scale to the token position.

### Key Insight: What this ablation is NOT measuring

The P25 ablation only probes **context modulation** (gate architecture). It does NOT ablate:
- The basis projection itself (`W_b`, `W_m`)
- The LayerNorm on the basis
- The number of basis dimensions (basis_size)
- Whether the basis+mixing structure itself helps vs dense

These are the **structural** components. To know if the basis/mixing factorization pays off vs dense, you need a direct comparison against the p23 dense baseline.

---

## Recommended Next Steps

### Immediate: Make `linear` the new default gate mode
- **Motivation**: LINEAR_GATE loses only 0.0041 BPB vs MLP_GATE but saves ~23% of total FLOPs.
- **At depth=16**: 6 layers/block × 16 blocks = 96 RemixedLinear ops; each saves 0.375C² FLOPs.
- Change default `basis_gate_mode` from `'mlp'` to `'linear'` in `base_train.py`.

### Follow-up: Ablate the structural components
The ablation scope was limited to gate architecture. The **remaining unknowns**:

| Question | Experiment |
|---|---|
| Does basis_size matter? | Sweep B = C/8, C/4, C/2 |
| Does the LayerNorm on basis help? | Remove `ln_basis` and compare |
| Does W_b + W_m beat dense W at same params? | Iso-param comparison (see P23 dense baseline) |
| Is `weight` modulation the right choice? | Compare vs householder/ckr at linear gate |

### Deeper: Reduce basis_size
If `linear` gate is adopted, the next largest cost is the `W_b: C→B` + `W_m: B→D` pair.
Currently B=C/4=64 (for dim=256). FLOPs = CB + BD = 2CB = 0.5C² (both projections).
- Halving B to C/8 saves ~0.25C² additional FLOPs → total ~1.0C² (50% of dense!)
- This DOES reduce model capacity and would need loss comparison.

---

## Full RemixedLinear Class Source (gpt.py, lines 1246–1813)

```python
class RemixedLinear(nn.Module):
    def __init__(self, in_features, out_features, context_dim, basis_size=64, remixed_linear_kwargs=None, scale_basis=True, film_gate=False, routing_scope='per_sequence'):
        super().__init__()
        self._film_gate_flag = film_gate
        # routing_scope: 'per_token' (FFN layers) or 'per_sequence' (attention layers)
        self.routing_scope = routing_scope
        if remixed_linear_kwargs is None:
            remixed_linear_kwargs = dict(use_basis_gate=True, use_output_gate=True, use_context=True)
        # Clamp basis_size to avoid rank bottleneck at asymmetric in/out dims
        if scale_basis:
            basis_size = max(basis_size, min(in_features, out_features) // 4)
        self.basis_size = basis_size
        self.use_basis_gate = remixed_linear_kwargs.get('use_basis_gate', True)
        self.use_output_gate = remixed_linear_kwargs.get('use_output_gate', True)
        self.use_context = remixed_linear_kwargs.get('use_context', True)
        self.sparse_gate_k = remixed_linear_kwargs.get('sparse_gate_k', 0)
        self.gate_temperature = max(remixed_linear_kwargs.get('gate_temperature', 1.0), 1e-6)
        self.operator_modulation = remixed_linear_kwargs.get('operator_modulation', 'none')
        self._last_orth_loss = None
        self.n_templates = remixed_linear_kwargs.get('n_templates', 1)
        self.template_routing_learned = remixed_linear_kwargs.get('template_routing_learned', False)
        self.tiny_expert = remixed_linear_kwargs.get('tiny_expert', False)
        self.tiny_expert_topk = remixed_linear_kwargs.get('tiny_expert_topk', 16)
        self.use_quantile_route = int(remixed_linear_kwargs.get('use_quantile_route', 0))
        self.lokr_expert = remixed_linear_kwargs.get('lokr_expert', False)
        self.lokr_n_experts = remixed_linear_kwargs.get('lokr_n_experts', 64)
        self.lokr_topk = remixed_linear_kwargs.get('lokr_topk', 16)
        self.lokr_rank = remixed_linear_kwargs.get('lokr_rank', 4)
        self.lokr_learned = remixed_linear_kwargs.get('lokr_learned', False)
        self._use_shared_route = remixed_linear_kwargs.get('use_shared_route', False)

        # Initialise all optional attrs to None
        self.template_bank = None
        self.template_mixing = None
        self.expert_up_w = None
        self.expert_down_w = None
        self.template_route = None
        self.lokr_down_w = None
        self.lokr_up_w = None
        self.lokr_route_proj = None

        if self.lokr_expert:
            K, R = self.lokr_n_experts, self.lokr_rank
            b_shrunk = basis_size - K * R
            if b_shrunk <= 0:
                import warnings
                R_old = R
                R = max(1, (basis_size - 1) // K)
                b_shrunk = basis_size - K * R
                self.lokr_rank = R
                warnings.warn(f"LoKR: auto-clamped rank {R_old}→{R} (basis={basis_size}, K={K})")
            assert b_shrunk > 0
            basis_size = b_shrunk
            self.basis_size = b_shrunk

        self.basis = Linear(in_features, basis_size, bias=False)
        if self.lokr_expert:
            K, R = self.lokr_n_experts, self.lokr_rank
            self.lokr_down_w = nn.Parameter(torch.empty(K, R, in_features))
            self.lokr_up_w   = nn.Parameter(torch.zeros(K, out_features, R))
            if self.lokr_learned:
                self.lokr_route_proj = nn.Parameter(torch.empty(K, in_features))
            else:
                self.lokr_route_proj = nn.Parameter(torch.empty(K, in_features), requires_grad=False)
            self.template_mixing = nn.Parameter(torch.empty(out_features, basis_size))
            self._lokr_basis_size = basis_size
        elif self.tiny_expert and self.n_templates > 1:
            topk = self.tiny_expert_topk
            assert topk > 0
            raw_expert_dim = basis_size // topk
            self.expert_dim = raw_expert_dim
            K = self.n_templates
            self.expert_up_w   = nn.Parameter(torch.empty(K, self.expert_dim, basis_size))
            self.expert_down_w = nn.Parameter(torch.empty(K, out_features, self.expert_dim))
            self.template_bank = None
            self.template_mixing = None
            if not self._use_shared_route:
                if self.use_quantile_route == 2:
                    self._qrouter = QuantileCrossAttentionRouter(in_features, K, topk)
                    self.template_route = None
                elif self.use_quantile_route == 1:
                    self._qrouter = QuantileBalancedRouter(in_features, K, topk, learned=self.template_routing_learned)
                    self.template_route = None
                else:
                    route_init = torch.randn(in_features, K) / (in_features ** 0.5)
                    if self.template_routing_learned:
                        self.template_route = nn.Parameter(route_init)
                    else:
                        self.register_buffer('template_route', route_init)
                    self._qrouter = None
            else:
                self._qrouter = None
            self.register_buffer('_template_entropy_buf', torch.zeros(1), persistent=False)
        elif self.n_templates > 1:
            self.template_bank = nn.ParameterList([
                nn.Parameter(torch.randn(out_features, basis_size)) for _ in range(self.n_templates)
            ])
            self.template_mixing = None
            route_init = torch.randn(in_features, self.n_templates) / (in_features ** 0.5)
            if self.template_routing_learned:
                self.template_route = nn.Parameter(route_init)
            else:
                self.register_buffer('template_route', route_init)
            self.register_buffer('_template_entropy_buf', torch.zeros(1), persistent=False)
        else:
            self.template_mixing = nn.Parameter(torch.randn(out_features, basis_size))
        self.ln_basis = nn.LayerNorm(basis_size)
        self.bias = nn.Parameter(torch.zeros(out_features))

        if self.use_context:
            basis_hidden = max(context_dim // 2, min(basis_size, context_dim * 2))
            film_gate = getattr(self, '_film_gate_flag', False)
            _gate_out_size = 2 * basis_size if self.use_basis_gate and film_gate else basis_size
            self.basis_gate_mode = remixed_linear_kwargs.get('basis_gate_mode', 'mlp')
            self.basis_modulator = None
            self.basis_gate_content = None
            self.basis_gate_context = None
            if self.use_basis_gate:
                if self.basis_gate_mode == 'mlp':
                    self.basis_modulator = nn.Sequential(
                        Linear(context_dim, basis_hidden, bias=True),
                        nn.GELU(),
                        Linear(basis_hidden, _gate_out_size, bias=True),
                    )
                    nn.init.zeros_(self.basis_modulator[-1].weight)
                    if self.basis_modulator[-1].bias is not None:
                        nn.init.zeros_(self.basis_modulator[-1].bias)
                elif self.basis_gate_mode == 'linear':
                    self.basis_modulator = Linear(context_dim, _gate_out_size, bias=True)
                    nn.init.zeros_(self.basis_modulator.weight)
                    nn.init.zeros_(self.basis_modulator.bias)
                elif self.basis_gate_mode == 'attn':
                    # gate = σ(W_content·h ⊙ W_context·ctx)
                    self.basis_gate_content = Linear(basis_size, _gate_out_size, bias=False)
                    self.basis_gate_context = Linear(context_dim, _gate_out_size, bias=True)
                    nn.init.normal_(self.basis_gate_content.weight, std=0.02)
                    nn.init.zeros_(self.basis_gate_context.weight)
                    nn.init.zeros_(self.basis_gate_context.bias)
                # else 'none': gate_basis = ones in forward
            r = remixed_linear_kwargs.get('output_gate_rank', 8)
            self.output_gate_coeffs = Linear(context_dim, r, bias=True)
            self.output_gate_basis  = nn.Parameter(torch.zeros(r, out_features))
            self.output_gate_scale  = nn.Parameter(torch.ones(1) * 0.1)
            # operator_modulation variants omitted for brevity (householder/spectral/ocd/lie/polynomial/grassmann)

    def gate_parameters(self):
        """Yield gate-specific parameters for lower-LR optimizer group."""
        if self.use_context:
            if self.basis_modulator is not None:
                yield from self.basis_modulator.parameters()
            if self.basis_gate_content is not None:
                yield from self.basis_gate_content.parameters()
            if self.basis_gate_context is not None:
                yield from self.basis_gate_context.parameters()
            yield self.output_gate_coeffs.weight
            if self.output_gate_coeffs.bias is not None:
                yield self.output_gate_coeffs.bias
            yield self.output_gate_basis
            yield self.output_gate_scale
            # ... operator_modulation params ...
            if self.template_route is not None and isinstance(self.template_route, nn.Parameter):
                yield self.template_route
            if self.lokr_route_proj is not None and self.lokr_route_proj.requires_grad:
                yield self.lokr_route_proj

    def non_gate_parameters(self):
        """Yield structural parameters for Muon/normal-LR group."""
        yield self.basis.weight
        if self.template_mixing is not None: yield self.template_mixing
        if self.template_bank is not None:
            for t in self.template_bank: yield t
        if self.expert_up_w is not None: yield self.expert_up_w
        if self.expert_down_w is not None: yield self.expert_down_w
        if self.lokr_down_w is not None: yield self.lokr_down_w
        if self.lokr_up_w is not None: yield self.lokr_up_w
        yield self.bias

    def forward(self, x, context_state, route_weights=None, context_gates=None, **kwargs):
        """
        x:              (B, T, in_features)
        context_state:  (B, T, context_dim) — causal context from CCLBlock
        route_weights:  optional (B, T, K) pre-computed routing tensor
        context_gates:  optional dict with 'basis_gate' and 'output_coeffs'
                        from SharedContextGates (skips local gate MLP when provided)
        """
        dtype = x.dtype
        # ── Step 1: Project to basis + normalise ──────────────────────────────
        h_basis = self.ln_basis(self.basis(x).to(dtype=self.ln_basis.weight.dtype)).to(dtype=dtype)

        if self.use_context and context_state is not None:
            ctx = context_state.to(dtype=dtype)
            # Optional operator modulation (householder/spectral/lie/polynomial/grassmann)
            # ... (applies in-place transform to h_basis) ...

            # ── Step 2: Basis gate ────────────────────────────────────────────
            if self.use_basis_gate:
                if context_gates is not None and 'basis_gate' in context_gates:
                    gate_logits = context_gates['basis_gate'].to(dtype=dtype)
                elif self.basis_gate_mode == 'attn':
                    gate_logits = self.basis_gate_content(h_basis) * self.basis_gate_context(ctx)
                elif self.basis_gate_mode == 'none':
                    gate_logits = None
                else:  # 'mlp' or 'linear'
                    gate_logits = self.basis_modulator(ctx)

                if gate_logits is None:
                    gate_basis = torch.ones_like(h_basis)
                elif self._film_gate_flag:
                    scale_logits, shift = gate_logits.chunk(2, dim=-1)
                    gate_basis = (1.0 + torch.tanh(scale_logits * 0.1)).to(dtype=dtype)
                    h_basis = h_basis * gate_basis + shift.to(dtype=dtype)
                    gate_basis = None
                elif self.sparse_gate_k > 0:
                    k = min(self.sparse_gate_k, self.basis_size)
                    topk_vals, topk_idx = torch.topk(gate_logits, k=k, dim=-1)
                    sparse = torch.zeros_like(gate_logits).scatter_(-1, topk_idx, F.softmax(topk_vals, dim=-1))
                    soft = torch.sigmoid(gate_logits / self.gate_temperature)
                    gate_basis = (sparse + (soft - soft.detach())).to(dtype=dtype)
                else:
                    gate_basis = torch.sigmoid(gate_logits / self.gate_temperature).to(dtype=dtype)
            else:
                gate_basis = torch.ones_like(h_basis)

            # ── Step 3: Output gate (low-rank) ────────────────────────────────
            if self.use_output_gate:
                if context_gates is not None and 'output_coeffs' in context_gates:
                    coeffs = context_gates['output_coeffs'].to(dtype=dtype)
                else:
                    coeffs = self.output_gate_coeffs(ctx)              # (B, T, r)
                gate_logits = torch.matmul(coeffs, self.output_gate_basis.to(dtype=dtype))
                gate_out = 1.0 + torch.tanh(self.output_gate_scale.to(dtype=dtype) * gate_logits)
            else:
                gate_out = None
        else:
            gate_basis = torch.ones_like(h_basis)
            gate_out = None

        # ── Step 4: Apply basis gate ──────────────────────────────────────────
        h_gated = (h_basis * gate_basis).to(dtype=dtype)

        # ── Step 5: Mix to output space ───────────────────────────────────────
        # (single-template fast path; MoE/TinyExpert/LoKR paths omitted)
        pre_output = F.linear(h_gated, self.template_mixing.to(dtype=dtype))

        # ── Step 6: Apply output gate + bias ─────────────────────────────────
        if gate_out is not None:
            pre_output = pre_output * gate_out
        return (pre_output + self.bias.to(dtype=dtype)).to(dtype=dtype) \
               if self.bias is not None else pre_output.to(dtype=dtype)
```

---

## Design Notes / Known Issues

- **ATTN gate hurts**: bilinear content×context gating is unstable at this scale. The content projection from a partially-trained basis creates noisy gradients early in training.
- **gate_parameters() / non_gate_parameters()**: Critical for the dual-LR optimizer setup. Gate params get a lower LR to reduce early-training noise.
- **use_output_gate unconditionally creates output_gate_coeffs** when `use_context=True`, even when `use_output_gate=False` is passed. This was by design (always init for init_weights) but means `gate_parameters()` always yields them — potential optimization.
- **SharedContextGates integration**: When enabled, the gate MLP is computed once per block (not per-layer), amortizing its cost across Q/K/V/Proj. Currently disabled in P25 sweep (`--remix-shared-context-gates 0`) for clean ablation.
