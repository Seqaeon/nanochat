"""O4: the three-axis cost model, with its validation gate.

Gate (fully-binary-transformer-plan.md section 6): the axes must reproduce known
quantities on the dense baseline -- parameter count, FLOPs/token, optimiser bytes --
before any arm is scored against them.

Run: python -m scripts.o4_cost_model
"""
import argparse
import json
import os

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.bitcost import (DENSE_BF16, BITNET_LIKE, FULLY_BINARY,
                              NANOCHAT_MUONADAMW, counter_rule, cost_report)

# Ground truth, V=32,768.  Each entry carries the EXACT config that produced it;
# comparing against a number from a different config is how the first run of this
# gate "failed".  depth 4 is the local probe (seq 1024, full attention); depth 8 is
# the dense arm quoted throughout LEARNINGS.md (seq 2048, default sliding windows).
KNOWN = {
    4: dict(seq=1024, wp="L", flops=8.178970e+07, params=36_700_296),
    8: dict(seq=2048, wp="SSSL", flops=2.862643e+08, params=125_829_648),
}

HEAD_DIM = 128
ASPECT = 64


def build_config(depth, vocab_size=32768, seq_len=2048, wp="SSSL"):
    base = depth * ASPECT
    n_embd = ((base + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    n_head = n_embd // HEAD_DIM
    return GPTConfig(sequence_len=seq_len, vocab_size=vocab_size, n_layer=depth,
                     n_head=n_head, n_kv_head=n_head, n_embd=n_embd, window_pattern=wp)


def measure(depth, vocab_size=32768, seq_len=2048, wp="SSSL"):
    with torch.device("meta"):
        model = GPT(build_config(depth, vocab_size, seq_len, wp))
    params = model.num_scaling_params()
    total_flops, _active, _ap = model.estimate_flops()
    return params, total_flops


def validate():
    print("=" * 78)
    print("GATE: reproduce known dense quantities")
    print("=" * 78)
    ok = True
    for depth, ref in KNOWN.items():
        params, flops = measure(depth, seq_len=ref["seq"], wp=ref["wp"])
        p_ok = params["total"] == ref["params"]
        f_ok = abs(flops - ref["flops"]) / ref["flops"] < 1e-6
        ok = ok and p_ok and f_ok
        print(f"  depth {depth:2d} (seq={ref['seq']}, wp={ref['wp']})")
        print(f"    params {params['total']:12,d} vs {ref['params']:12,d}  {'PASS' if p_ok else 'FAIL'}")
        print(f"    flops  {flops:.6e} vs {ref['flops']:.6e}  {'PASS' if f_ok else 'FAIL'}")
    return ok


def validate_optimizer_bytes(depth=4):
    """Build a real model on CPU, take one step, and count actual optimiser state."""
    print("-" * 78)
    print(f"GATE: optimiser bytes, real model at depth {depth}")
    cfg = build_config(depth)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    opts = model.setup_optimizer()
    opts = opts if isinstance(opts, (list, tuple)) else [opts]
    x = torch.randint(0, cfg.vocab_size, (1, 64))
    loss = model(x, targets=x)
    loss = loss[0] if isinstance(loss, tuple) else loss
    loss.backward()
    for o in opts:
        o.step()
    state_bytes = 0
    for o in opts:
        for st in o.state.values():
            for v in st.values():
                if torch.is_tensor(v):
                    state_bytes += v.numel() * v.element_size()
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    nparams = sum(p.numel() for p in model.parameters())
    print(f"  parameters        {nparams:12,d}")
    print(f"  param bytes       {param_bytes/2**20:10.1f} MiB  ({8*param_bytes/nparams:.1f} bits/param)")
    print(f"  optimiser state   {state_bytes/2**20:10.1f} MiB  ({8*state_bytes/nparams:.1f} bits/param)")
    total_bits = 8 * (param_bytes + state_bytes) / nparams
    print(f"  MEASURED TOTAL    {total_bits:10.1f} bits/param  "
          f"(model uses measured {NANOCHAT_MUONADAMW.state_bits_per_param} state bits/param)")
    return total_bits


def report_table(depth, vocab_size=32768):
    params, flops = measure(depth, vocab_size)
    print("=" * 78)
    print(f"THREE AXES  depth={depth}  V={vocab_size:,}  n_embd={build_config(depth).n_embd}")
    print("=" * 78)
    rows = [
        ("dense bf16 + MuonAdamW", DENSE_BF16, NANOCHAT_MUONADAMW, ()),
        ("BitNet-like W1.58A8", BITNET_LIKE, NANOCHAT_MUONADAMW, ()),
        ("fully binary + MuonAdamW", FULLY_BINARY, NANOCHAT_MUONADAMW, ()),
        ("fully binary + counter", FULLY_BINARY, NANOCHAT_MUONADAMW,
         ("transformer_matrices", "lm_head", "wte", "value_embeds", "research")),
    ]
    print(f"{'arm':<24}{'FLOPs/tok':>12}{'BOPs arith':>13}{'pJ/tok':>12}"
          f"{'infer MiB':>11}{'train MiB':>11}")
    print("-" * 80)
    base = None
    for label, prec, opt, bin_groups in rows:
        r = cost_report(params, flops, prec, opt, seq_len=build_config(depth).sequence_len,
                        binary_optimizer_groups=bin_groups)
        mib = r.total_state_bytes / 2**20
        if base is None:
            base = (r.bops_arithmetic, r.energy_pj_per_token, mib)
        print(f"{label:<24}{r.flops_per_token:12.3e}{r.bops_arithmetic:13.3e}"
              f"{r.energy_pj_per_token:12.3e}"
              f"{r.total_inference_bytes/2**20:11.1f}{mib:11.1f}")
    print("-" * 80)
    r_d = cost_report(params, flops, DENSE_BF16, NANOCHAT_MUONADAMW, binary_optimizer_groups=())
    r_b = cost_report(params, flops, FULLY_BINARY, NANOCHAT_MUONADAMW,
                      binary_optimizer_groups=("transformer_matrices", "lm_head", "wte",
                                               "value_embeds", "research"))
    print(f"binary vs dense:  FLOPs {r_b.flops_per_token/r_d.flops_per_token:.2f}x   "
          f"BOPs(arith) {r_d.bops_arithmetic/r_b.bops_arithmetic:.0f}x cheaper   "
          f"energy {r_d.energy_pj_per_token/r_b.energy_pj_per_token:.1f}x cheaper")
    print(f"                  inference bytes {r_d.total_inference_bytes/r_b.total_inference_bytes:.1f}x smaller   "
          f"training state {r_d.total_state_bytes/r_b.total_state_bytes:.1f}x smaller")
    r_bw = cost_report(params, flops, FULLY_BINARY, NANOCHAT_MUONADAMW, binary_optimizer_groups=())
    print(f"  binary weights WITHOUT the counter optimiser: training state "
          f"{r_d.total_state_bytes/r_bw.total_state_bytes:.2f}x smaller   "
          f"(the optimiser is {NANOCHAT_MUONADAMW.state_bits_per_param}/60.4 of the baseline)")
    print()


def load_measured_ratios(path="out/b00_binary_phase0/o3_kernel_gate.json"):
    """Ratios MEASURED by scripts/o3_kernel_gate.py on the device it ran on.

    Without this the break-even table reprints numbers from a 20 W-capped laptop on
    whatever machine you happen to be using, which is worse than printing nothing.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def spending_curve(dense_depth=8, vocab_size=32768, max_depth=200, ratios_path=None):
    """How much of the 16x memory saving do you SPEND on parameters?

    "Matched inference bytes" is one endpoint of a curve, and it is the endpoint most
    hostile to binary: it spends the entire saving on more model, which maximises the
    FLOPs burden.  Nothing forces that choice.  At the other endpoint, same
    architecture, binary is 16x smaller at 1.00x the FLOPs with every MAC cheaper,
    and any kernel above 1x is a win.  Printing only the hostile endpoint, as an
    earlier version of this script did, is not an analysis.
    """
    ref = KNOWN.get(dense_depth, dict(seq=2048, wp="SSSL"))
    p_d, f_d = measure(dense_depth, vocab_size, ref["seq"], ref["wp"])
    r_d = cost_report(p_d, f_d, DENSE_BF16, NANOCHAT_MUONADAMW, seq_len=ref["seq"])
    budget = r_d.total_inference_bytes

    print("=" * 82)
    print(f"SPENDING CURVE  (dense depth {dense_depth}, V={vocab_size:,}, "
          f"{budget/2**20:.0f} MiB, {p_d['total']/1e6:.0f}M params)")
    print("=" * 82)
    print(f"{'binary depth':>13}{'params':>15}{'x params':>10}{'MiB':>8}{'x bytes':>9}"
          f"{'x FLOPs':>9}{'kernel needed':>15}")
    print("-" * 82)
    for depth in range(dense_depth, max_depth + 1):
        p_b, f_b = measure(depth, vocab_size, ref["seq"], ref["wp"])
        r_b = cost_report(p_b, f_b, FULLY_BINARY, counter_rule(4), seq_len=ref["seq"])
        mib = r_b.total_inference_bytes / 2**20
        if r_b.total_inference_bytes > budget:
            break
        if depth == dense_depth or depth % 4 == 0 or depth >= max_depth:
            print(f"{depth:>13}{p_b['total']:>15,d}{p_b['total']/p_d['total']:>9.1f}x"
                  f"{mib:>8.0f}{budget/r_b.total_inference_bytes:>8.1f}x"
                  f"{f_b/f_d:>8.1f}x{f_b/f_d:>14.1f}x")
    print("-" * 82)
    print("  'kernel needed' is the b1-over-bf16 speedup required to TIE on wall clock.")
    print("  The top row spends NOTHING on extra parameters: 16x smaller, same FLOPs,")
    print("  so any kernel above 1.0x is already a wall-clock win. The bottom row spends")
    print("  everything. The paper picks a point on this curve and defends it; it does")
    print("  not get to quote the memory of one end and the speed of the other.")
    print()


def matched_bytes_arm(dense_depth=8, vocab_size=32768, max_depth=200, ratios_path=None):
    """The comparison that actually matters: what does binary BUY at equal bytes?

    Same-architecture-lower-precision rows all share one FLOPs number by construction.
    At a matched inference-byte budget the binary model is a DIFFERENT, larger model,
    and then FLOPs, BOPs and energy all move.
    """
    ref = KNOWN.get(dense_depth, dict(seq=2048, wp="SSSL"))
    p_d, f_d = measure(dense_depth, vocab_size, ref["seq"], ref["wp"])
    r_d = cost_report(p_d, f_d, DENSE_BF16, NANOCHAT_MUONADAMW, seq_len=ref["seq"])
    budget = r_d.total_inference_bytes

    best = None
    for depth in range(dense_depth, max_depth + 1):
        p_b, f_b = measure(depth, vocab_size, ref["seq"], ref["wp"])
        r_b = cost_report(p_b, f_b, FULLY_BINARY, counter_rule(4), seq_len=ref["seq"],
                          binary_optimizer_groups=())
        if r_b.total_inference_bytes > budget:
            break
        best = (depth, p_b, f_b, r_b)
    if best is None:
        print("no matched-bytes depth found"); return
    depth, p_b, f_b, r_b = best

    print("=" * 80)
    print(f"MATCHED INFERENCE BYTES  (budget = dense depth {dense_depth} = "
          f"{budget/2**20:.1f} MiB, V={vocab_size:,})")
    print("=" * 80)
    print(f"{'arm':<26}{'depth':>7}{'params':>14}{'FLOPs/tok':>12}{'BOPs arith':>13}{'MiB':>8}")
    print("-" * 80)
    print(f"{'dense bf16':<26}{dense_depth:>7}{p_d['total']:>14,d}{f_d:12.3e}"
          f"{r_d.bops_arithmetic:13.3e}{r_d.total_inference_bytes/2**20:8.1f}")
    print(f"{'fully binary':<26}{depth:>7}{p_b['total']:>14,d}{f_b:12.3e}"
          f"{r_b.bops_arithmetic:13.3e}{r_b.total_inference_bytes/2**20:8.1f}")
    print("-" * 80)
    print(f"  at equal bytes the binary model has {p_b['total']/p_d['total']:.1f}x the parameters, "
          f"{f_b/f_d:.1f}x the FLOPs,")
    print(f"  and still {r_d.bops_arithmetic/r_b.bops_arithmetic:.1f}x FEWER bit-operations.")
    print(f"  THE BET: does a {p_b['total']/1e6:.0f}M-parameter 1-bit model beat a "
          f"{p_d['total']/1e6:.0f}M-parameter bf16 one?")
    print()
    # The wall-clock consequence, which the BOPs column hides.
    need = f_b / f_d
    print(f"  WALL-CLOCK BREAK-EVEN: at equal bytes the binary model issues {need:.1f}x the MACs,")
    print(f"  so a b1 kernel must be >= {need:.1f}x faster per MAC than bf16 just to TIE on time.")
    meas = load_measured_ratios(ratios_path or "out/b00_binary_phase0/o3_kernel_gate.json")
    rows = []
    if meas and meas.get("ratios"):
        for label, r in sorted(meas["ratios"].items(), key=lambda kv: -kv[1]):
            rows.append((f"MEASURED {label} on {meas.get('device','?')}", r))
    else:
        print("    NO MEASURED KERNEL RATIO ON THIS DEVICE.")
        print("    Run:  python -m scripts.o3_kernel_gate     (writes o3_kernel_gate.json)")
        print("    The numbers below are STALE PLACEHOLDERS from a 20 W power-capped")
        print("    laptop where the same binary gave 1.93x and 4.19x in two states.")
        print("    They describe that laptop and nothing else. Do not read them as a")
        print("    property of the hardware you are on now.")
        rows = [("STALE laptop, state B", 1.93), ("STALE laptop, state A", 4.19)]
    rows.append(("Ampere b1 spec ceiling", 64.0))
    for label, ratio in rows:
        v = need / ratio
        verdict = f"{v:.1f}x SLOWER" if v > 1 else f"{1 / v:.1f}x faster"
        print(f"    at {ratio:6.2f}x ({label:<44}) -> binary is {verdict}")
    print(f"  So the kernel must reach {100*need/64:.0f}% of the Ampere b1 spec ceiling to break even.")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--vocab", type=int, nargs="+", default=[32768])
    ap.add_argument("--skip-optimizer", action="store_true")
    ap.add_argument("--ratios", default=None,
                    help="o3_kernel_gate.json with ratios measured on THIS device")
    a = ap.parse_args()
    passed = validate()
    if not a.skip_optimizer:
        validate_optimizer_bytes(4)
    print()
    for v in a.vocab:
        for d in a.depths:
            report_table(d, v)
    spending_curve(8)
    matched_bytes_arm(8, ratios_path=a.ratios)
    print("gate:", "PASS" if passed else "FAIL")
