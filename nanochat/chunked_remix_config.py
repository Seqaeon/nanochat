"""
ChunkedRemixConfig — canonical P29 RemixedLinear architecture configuration.

Standardized class that encapsulates the best-known P29 architecture defaults:
  - Full-rank basis (basis_size = model_dim)
  - 8 templates with learned routing
  - N=64 chunk routing (amortized routing cost)
  - Quantile-balanced routing (no aux loss)
  - Selective context stream
  - Centered output gate (no basis gate)

Usage
-----
In a Python experiment::

    from nanochat.chunked_remix_config import ChunkedRemixConfig

    cfg = ChunkedRemixConfig()                     # canonical defaults
    cfg = ChunkedRemixConfig(n_templates=4)        # override a field
    cfg = ChunkedRemixConfig(chunk_routing_size=0) # disable chunk routing

    # Plug into GPTConfig:
    from nanochat.gpt import GPTConfig
    model_cfg = GPTConfig(
        **cfg.to_gpt_config_overrides(),
        n_layer=12, n_embd=768, ...
    )

    # Or obtain the remixed_linear_kwargs dict directly:
    rl_kwargs = cfg.to_remixed_linear_kwargs()

In a sweep script, activate with ``--use-chunked-remix 1`` in REMIX_COMMON.
Individual overrides (e.g. ``--p22-n-templates 4``) still take effect on top
of the config because ``research_compare.py`` applies them *after* the config
defaults are set.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChunkedRemixConfig:
    """Canonical P29 RemixedLinear configuration with chunk routing.

    Field names use clean, prefix-free names.  The mapping back to the
    internal nanochat conventions is documented for each field.
    """

    # ── Template / Routing ────────────────────────────────────────────
    # internal: p22_n_templates  (remixed_linear_kwargs['n_templates'])
    n_templates: int = 8
    """Number of template_mixing matrices (experts).  8 = canonical P29."""

    # internal: p28_chunk_routing_size  (remixed_linear_kwargs['chunk_routing_size'])
    chunk_routing_size: int = 64
    """Amortize template routing over N-token chunks.  0 = per-token (expensive)."""

    # internal: p23_quantile_route  (GPTConfig.p23_quantile_route)
    quantile_route: int = 1
    """Quantile-balanced routing mode.  1 = EMA quantile (no aux loss), 0 = off."""

    # internal: p22_template_routing_learned  (remixed_linear_kwargs['template_routing_learned'])
    routing_learned: bool = True
    """Gradient-driven learned routing weights (True) vs frozen random projection."""

    # ── Basis / Gates ─────────────────────────────────────────────────
    # internal: remix_basis_size  (GPTConfig.remix_basis_size)
    basis_size: int = 0
    """Explicit basis size.  0 = inherit from model_dim passed at call-site (full rank)."""

    # internal: remix_use_basis_gate  (remixed_linear_kwargs['use_basis_gate'])
    basis_gate: bool = False
    """Enable context-conditioned basis gating.  False = cleaner gradient at P29 scale."""

    # internal: remix_use_output_gate  (remixed_linear_kwargs['use_output_gate'])
    output_gate: bool = True
    """Enable low-rank output gate (1 + tanh(s·coeff·basis))."""

    # internal: remix_use_context  (remixed_linear_kwargs['use_context'])
    use_context: bool = True
    """Enable context modulation pathway."""

    # internal: remix_basis_gate_mode  (remixed_linear_kwargs['basis_gate_mode'])
    basis_gate_mode: str = "centered"
    """Basis gate architecture when basis_gate=True.  'centered' = 1+tanh."""

    # internal: remix_shared_context_gates  (GPTConfig.remix_shared_context_gates)
    shared_context_gates: bool = False
    """Batch all per-layer context gate matmuls into 3 block-level ops.  Saves memory."""

    # ── CCL Block / Context Stream ────────────────────────────────────
    # internal: cclblock_modulation  (GPTConfig.cclblock_modulation)
    modulation: str = "weight"
    """CCL block strategy.  'weight' = RemixedLinear gates in activation space."""

    # internal: cclblock_context_stream  (GPTConfig.cclblock_context_stream)
    context_stream: str = "selective"
    """Context threading mechanism.  'selective' = GRU-style input-dependent gating."""

    # internal: cclblock_gate_temperature  (GPTConfig.cclblock_gate_temperature)
    gate_temperature: float = 2.0
    """Basis gate temperature.  >1 = softer/more uniform selection."""

    # ─────────────────────────────────────────────────────────────────
    def to_remixed_linear_kwargs(self) -> dict:
        """Return a dict suitable for ``RemixedLinear(remixed_linear_kwargs=...)``.

        These are the fields that live *inside* the RemixedLinear constructor
        rather than on GPTConfig.
        """
        return {
            "n_templates":             self.n_templates,
            "chunk_routing_size":      self.chunk_routing_size,
            "template_routing_learned": self.routing_learned,
            "use_basis_gate":          self.basis_gate,
            "use_output_gate":         self.output_gate,
            "use_context":             self.use_context,
            "basis_gate_mode":         self.basis_gate_mode,
            # quantile_route lives on GPTConfig, not here; see to_gpt_config_overrides()
        }

    def to_gpt_config_overrides(self) -> dict:
        """Return a dict to ``**spread`` into ``GPTConfig(...)`` or model_config_kwargs.

        Only includes fields that map to top-level GPTConfig attributes (not the
        nested remixed_linear_kwargs dict).
        """
        overrides = {
            "cclblock_modulation":       self.modulation,
            "cclblock_context_stream":   self.context_stream,
            "cclblock_gate_temperature": self.gate_temperature,
            "remix_shared_context_gates": int(self.shared_context_gates),
            "p23_quantile_route":        self.quantile_route,
        }
        if self.basis_size > 0:
            overrides["remix_basis_size"] = self.basis_size
        return overrides

    def to_cli_args(self, model_dim: int = 0) -> list[str]:
        """Return a flat list of CLI flag strings for ``base_train.py`` / ``research_compare.py``.

        This is useful for constructing subprocess commands in tests or scripts.

        Args:
            model_dim: When non-zero, passes ``--remix-basis-size model_dim`` to
                       force full-rank basis regardless of ``self.basis_size``.
        """
        bs = model_dim if model_dim > 0 else self.basis_size
        args = [
            "--p22-n-templates",             str(self.n_templates),
            "--p28-chunk-routing-size",       str(self.chunk_routing_size),
            "--p23-quantile-route",           str(self.quantile_route),
            "--p22-template-routing-learned", str(int(self.routing_learned)),
            "--remix-use-basis-gate",         str(int(self.basis_gate)),
            "--remix-use-output-gate",        str(int(self.output_gate)),
            "--remix-use-context",            str(int(self.use_context)),
            "--remix-basis-gate-mode",        self.basis_gate_mode,
            "--remix-shared-context-gates",   str(int(self.shared_context_gates)),
            "--cclblock-modulation",          self.modulation,
            "--cclblock-context-stream",      self.context_stream,
            "--cclblock-gate-temperature",    str(self.gate_temperature),
        ]
        if bs > 0:
            args += ["--remix-basis-size", str(bs)]
        return args

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"ChunkedRemixConfig("
            f"n_templates={self.n_templates}, "
            f"chunk={self.chunk_routing_size}, "
            f"quantile_route={self.quantile_route}, "
            f"stream={self.context_stream}, "
            f"T={self.gate_temperature}, "
            f"basis_gate={self.basis_gate}, "
            f"output_gate={self.output_gate})"
        )

    def __repr__(self) -> str:
        return self.summary()
