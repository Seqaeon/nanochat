"""Cost accounting for sub-8-bit architectures.

O4 of ``fully-binary-transformer-plan.md``.

FLOPs cannot score a binary model.  A W1A1 transformer has *identical* FLOPs per
token to its dense twin, because binarisation changes what an operation costs and
not how many there are.  ``GPT.estimate_flops`` is therefore blind to the entire
claim, and it is additionally blind to ``wte``/``wpe``/``value_embeds``, which it
correctly excludes as lookups rather than matmuls but which are 80% of the
parameters at depth 8, V=32,768.

This module supplies the three axes the plan claims on:

  1. **bit-operations** -- reported under two conventions, which disagree, so both
     are printed and labelled rather than one being silently chosen.
  2. **energy** -- op energy plus memory-access energy.
  3. **training-state bytes** -- weights plus optimiser state.  Hardware
     independent, exactly measurable, and the axis that decides what fits.

Nothing here invents a number.  Quantities that must come from a citation or a
measurement are marked PROVISIONAL and carry the source they must be pinned from.
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants that must be pinned before any number leaves this repo.
# ---------------------------------------------------------------------------

# PROVISIONAL.  Pin from Horowitz, "Computing's Energy Problem (and what we can do
# about it)", ISSCC 2014, 45nm.  Values in picojoules.  Do not quote a joule figure
# in a paper until these are read off the paper itself.
ENERGY_PJ_PROVISIONAL = {
    "fp16_mac": 1.5,     # ~1.1 pJ multiply + ~0.4 pJ add
    "fp32_mac": 4.6,     # ~3.7 pJ multiply + ~0.9 pJ add
    "int8_mac": 0.23,    # ~0.2 pJ multiply + ~0.03 pJ add
    "int1_mac": 0.005,   # AND + popcount contribution; the weakest number here
    "dram_bit": 20.0,    # per bit of DRAM traffic; the dominant term
    "sram_bit": 0.16,    # per bit of on-chip access
}

# PROVISIONAL.  Ampere tensor-core throughput ratios relative to 1-bit, from the
# GA10x spec sheet.  O3 measured 1.93x-4.19x for b1 against cuBLAS bf16 on sm_86
# and could not resolve the magnitude under a 20 W power cap, so this column is a
# spec-sheet ceiling and NOT a measured speedup.
HW_OPS_PER_MAC_PROVISIONAL = {1: 1, 4: 4, 8: 16, 16: 64, 32: 256}


@dataclass(frozen=True)
class PrecisionSpec:
    """Bit widths.  ``None`` for a component means "not present in this model"."""
    w_bits: int = 16          # matmul weights
    a_bits: int = 16          # matmul activations
    embed_bits: int = 16      # wte / value_embeds table entries
    head_bits: int = 16       # lm_head weights
    accum_bits: int = 32      # integer/float accumulator, named not hidden

    @property
    def name(self):
        return f"W{self.w_bits}A{self.a_bits}E{self.embed_bits}H{self.head_bits}"


DENSE_BF16 = PrecisionSpec()
BITNET_LIKE = PrecisionSpec(w_bits=2, a_bits=8, embed_bits=16, head_bits=16)
FULLY_BINARY = PrecisionSpec(w_bits=1, a_bits=1, embed_bits=1, head_bits=1)


@dataclass(frozen=True)
class OptimizerSpec:
    """Bits of optimiser STATE per parameter, excluding the parameter itself.

    Training state per parameter is ``weight_bits + state_bits``; the two add, they
    do not overlap.  An earlier version of this module took ``max()`` of them, which
    made binary weights look worth exactly nothing during training.
    """
    label: str
    state_bits_per_param: float


# MEASURED on this repo, not assumed.  scripts/o4_cost_model.py --validate builds a
# real depth-4 model, takes one step, and sums the optimiser's state tensors:
# 21.0 bits/param of parameters + 39.3 bits/param of state = 60.4 total.
# The textbook "AdamW = fp32 master + 2 fp32 moments = 96" does NOT describe this
# repo, which runs a single fused MuonAdamW over mixed-dtype parameters.
NANOCHAT_MUONADAMW = OptimizerSpec("MuonAdamW (measured on this repo)", 39.3)
ADAMW_FP32_TEXTBOOK = OptimizerSpec("AdamW fp32, 2 moments (textbook, NOT this repo)", 64)
BOP_FP32 = OptimizerSpec("Bop: fp32 inertia, one real per weight", 32)


def counter_rule(k_bits: int) -> OptimizerSpec:
    """The plan's section 4.6 rule: a k-bit saturating counter, no master copy."""
    return OptimizerSpec(f"{k_bits}-bit saturating counter", k_bits)


@dataclass
class CostReport:
    params: dict
    flops_per_token: float
    macs_per_token: float
    bops_arithmetic: float
    bops_hardware: float
    energy_pj_per_token: float
    state_bytes: dict
    precision: PrecisionSpec
    inference_bytes: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def total_state_bytes(self):
        return sum(self.state_bytes.values())

    @property
    def total_inference_bytes(self):
        return sum(self.inference_bytes.values())


# ---------------------------------------------------------------------------

# Parameter groups that are LOOKUPS: they cost storage and memory traffic but
# contribute no MACs.  ``GPT.estimate_flops`` excludes exactly these at gpt.py:10852.
LOOKUP_GROUPS = ("wte", "wpe", "value_embeds")
# Groups that are matmul weights.
MATMUL_GROUPS = ("transformer_matrices", "research")
HEAD_GROUPS = ("lm_head",)


def _bits_for_group(group: str, prec: PrecisionSpec) -> int:
    if group in LOOKUP_GROUPS:
        return prec.embed_bits
    if group in HEAD_GROUPS:
        return prec.head_bits
    if group == "scalars":
        return 32  # scalars stay fp32 in every arm; counted, never hidden
    return prec.w_bits


def cost_report(params: dict, flops_per_token: float, prec: PrecisionSpec,
                optimizer: OptimizerSpec, seq_len: int = 2048,
                binary_optimizer_groups=()) -> CostReport:
    """Build the three-axis report.

    ``params`` is the dict from ``GPT.num_scaling_params()``.
    ``flops_per_token`` is from ``GPT.estimate_flops()``.
    ``binary_optimizer_groups`` names the groups the low-bit optimiser applies to;
    everything else keeps ``optimizer``.
    """
    notes = []
    # FLOPs are 2 per MAC by construction of the 6N + attention model.
    macs = flops_per_token / 2.0

    bops_arith = macs * prec.w_bits * prec.a_bits
    hw_w = HW_OPS_PER_MAC_PROVISIONAL.get(max(prec.w_bits, prec.a_bits))
    if hw_w is None:
        hw_w = max(prec.w_bits, prec.a_bits) ** 2
        notes.append(f"no spec-sheet ratio for {max(prec.w_bits, prec.a_bits)} bits; squared bits used")
    bops_hw = macs * hw_w

    # Energy: arithmetic plus weight traffic.  Weight traffic assumes each parameter
    # is read once per token, which is the decode/batch-1 regime and the honest
    # worst case; large-batch training amortises it and that is a separate number.
    if prec.w_bits == 1:
        mac_key = "int1_mac"
    elif prec.w_bits <= 8:
        mac_key = "int8_mac"
    elif prec.w_bits <= 16:
        mac_key = "fp16_mac"
    else:
        mac_key = "fp32_mac"
    arith_pj = macs * ENERGY_PJ_PROVISIONAL[mac_key]
    weight_bits = sum(n * _bits_for_group(g, prec) for g, n in params.items()
                      if g not in ("total",))
    traffic_pj = weight_bits * ENERGY_PJ_PROVISIONAL["dram_bit"] / max(seq_len, 1)
    notes.append("energy uses PROVISIONAL Horowitz-class figures; pin before quoting")
    notes.append(f"weight traffic amortised over seq_len={seq_len}")

    # Two different byte counts, and conflating them hides the main result.
    #  inference_bytes: weights at their own precision.  Binarisation is everything.
    #  state_bytes:     training state, max(weight, optimiser).  The optimiser
    #                   dominates at 96 bits/param, so weight precision is worth
    #                   ~nothing here unless the optimiser changes too.
    state_bytes, inference_bytes = {}, {}
    for g, n in params.items():
        if g == "total":
            continue
        opt = counter_rule(4) if g in binary_optimizer_groups else optimizer
        w_bits = _bits_for_group(g, prec)
        inference_bytes[g] = n * w_bits / 8.0
        # weight bits and optimiser-state bits ADD; they are different storage
        state_bytes[g] = n * (w_bits + opt.state_bits_per_param) / 8.0

    return CostReport(params=params, flops_per_token=flops_per_token, macs_per_token=macs,
                      bops_arithmetic=bops_arith, bops_hardware=bops_hw,
                      energy_pj_per_token=arith_pj + traffic_pj,
                      state_bytes=state_bytes, precision=prec, notes=notes,
                      inference_bytes=inference_bytes)
