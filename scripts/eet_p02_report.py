#!/usr/bin/env python3
"""EET P02 report: break-even arithmetic and the pre-registered pass/fail.

Turns the P02 sweep log into the only three numbers that matter:

  1. the gap between each EET arm and the dense control,
  2. the gap each arm is ALLOWED, given what it actually saves, read off a dense
     iso-data curve measured in this same sweep,
  3. whether that clears the criteria written down before the runs started.

Two accounting traps this script exists to avoid:

  * The LM head does not shrink. Every token is still predicted, so the head's
    2*d*V per token is untouched by routing. At d8/768 with V=32768 the head is
    roughly 28% of active FLOPs per token, which turns a 39% saving on the blocks
    into a 28% saving overall and shrinks the allowed gap accordingly. The report
    prints both ratios and uses the total for the verdict.
  * Attention keys. With kv_mode 'none' a layer at active fraction a costs a^2 of
    the dense attention matmul because both queries and keys shrink; restoring
    context makes it a. That difference is small (about 1.6% of a dense layer at
    a=0.1) but it is real and it is charged to the arms that use it.

Usage:
    python -m scripts.eet_p02_report --out-base out/eet_p02 --depth 8 \
        --target-active-frac 0.10 --oracle out/eet_p02/oracle_d8.json \
        --log out/eet_p02/sweep_eet_p02_d8.log
"""
import argparse
import json
import math
import os
import re
import statistics
import sys

from scripts.eet_context_oracle import bell_capacities


# Only print_header's own banner counts as a run boundary. The training logs are full of
# other bracketed prefixes ([sweep], [SCH], [EET], [ok], [GATE], [report], [oracle]) that
# match this shape, and any one of them would silently reassign every following bpb line
# to a run that does not exist -- which is how DENSE_D8 came back with "no validation bpb".
# print_header always emits a dashed rule immediately above the tag, so require it.
HEADER_RE = re.compile(r"^\s+\[([A-Za-z0-9_]+)\]\s+(.*)$")
RULE_RE = re.compile(r"^\s*-{10,}\s*$")
BPB_RE = re.compile(r"Validation bpb:\s*([0-9.]+)")
MINBPB_RE = re.compile(r"Minimum validation bpb:\s*([0-9.]+)")
FLOPS_ACT_RE = re.compile(r"FLOPs per token \(active\):\s*([0-9.e+\-]+)")
FLOPS_TOT_RE = re.compile(r"FLOPs per token \(total\):\s*([0-9.e+\-]+)")
TOKENS_RE = re.compile(r"Total number of training tokens:\s*([0-9,]+)")
DT_RE = re.compile(r"\bdt:\s*([0-9.]+)ms")
VOCAB_RE = re.compile(r'^\s*"vocab_size":\s*(\d+)')
NEMBD_CFG_RE = re.compile(r'^\s*"n_embd":\s*(\d+)')
NLAYER_CFG_RE = re.compile(r'^\s*"n_layer":\s*(\d+)')
DEPTH_RE = re.compile(r"depth of the Transformer model|--depth\s+(\d+)")
NEMBD_RE = re.compile(r"model_dim[^0-9]*(\d+)|n_embd[^0-9]*(\d+)")


# ---------------------------------------------------------------------------
def parse_log(path):
    """Split the concatenated sweep log into one record per run tag."""
    runs, cur, prev_was_rule = {}, None, False
    with open(path, errors='replace') as f:
        for line in f:
            m = HEADER_RE.match(line) if prev_was_rule else None
            prev_was_rule = bool(RULE_RE.match(line))
            if m and m.group(1) not in ('GATE', 'REPORT', 'T0A'):
                cur = m.group(1)
                # Overwrite, do not setdefault: a rerun of the same arm appends a fresh
                # section to the same log and must replace the earlier one entirely.
                runs[cur] = {'tag': cur, 'bpb': [], 'dt': [],
                             'flops_active': None, 'flops_total': None,
                             'tokens': None, 'vocab_size': None,
                             'n_embd': None, 'n_layer': None}
                continue
            if cur is None:
                continue
            r = runs[cur]
            if (m2 := BPB_RE.search(line)):
                r['bpb'].append(float(m2.group(1)))
            if (m2 := MINBPB_RE.search(line)):
                r['bpb'].append(float(m2.group(1)))
            if (m2 := FLOPS_ACT_RE.search(line)):
                r['flops_active'] = float(m2.group(1))
            if (m2 := FLOPS_TOT_RE.search(line)):
                r['flops_total'] = float(m2.group(1))
            if (m2 := TOKENS_RE.search(line)):
                r['tokens'] = int(m2.group(1).replace(',', ''))
            if (m2 := DT_RE.search(line)):
                r['dt'].append(float(m2.group(1)))
            for key, rx in (('vocab_size', VOCAB_RE), ('n_embd', NEMBD_CFG_RE),
                            ('n_layer', NLAYER_CFG_RE)):
                if r[key] is None and (m2 := rx.match(line)):
                    r[key] = int(m2.group(1))
    for r in runs.values():
        r['val_bpb'] = min(r['bpb']) if r['bpb'] else None
        if r['val_bpb'] is None:
            r['failed'] = True
        # Skip the first steps: they carry compile and warmup time.
        tail = r['dt'][10:] if len(r['dt']) > 20 else r['dt']
        r['dt_ms'] = statistics.median(tail) if tail else None
    return runs


# ---------------------------------------------------------------------------
def effective_seq(T, window):
    """Attention span used by GPT.estimate_flops (PaLM convention, no causal halving)."""
    return T if (window is None or window < 0) else min(window, T)


def block_flops_per_token(d, T, window, ffn_mult=4.0):
    """Per-token FLOPs for one block, in GPT.estimate_flops's units.

    That means 6 FLOPs per matmul parameter (2 forward, 4 backward) plus the PaLM
    attention-kernel term 12*h*q*effective_seq = 12*d*effective_seq. Matching the
    convention is what makes ``printed_total - blocks`` a meaningful head cost; mixing
    a forward-only model with a forward+backward printout would overstate the head's
    share by about 3x and inflate every allowed gap below.
    """
    proj = 6.0 * 4.0 * d * d                    # q, k, v, o
    attn = 12.0 * d * effective_seq(T, window)
    mlp = 6.0 * 2.0 * ffn_mult * d * d          # up and down
    return proj, attn, mlp


def eet_flop_ratio(per_block_active, d, T, windows, kv_mode, ffn_mult=4.0,
                   head_flops_per_token=0.0):
    """Ratio of EET active FLOPs to dense active FLOPs, per token of the full sequence.

    per_block_active[i] is the fraction of tokens still active when block i runs.
    Returns (ratio_blocks_only, ratio_including_head).
    """
    dense = eet = 0.0
    for i, a in enumerate(per_block_active):
        proj, attn, mlp = block_flops_per_token(d, T, windows[i % len(windows)], ffn_mult)
        dense += proj + attn + mlp
        key_frac = a if kv_mode == 'none' else 1.0
        cost = a * (proj + mlp) + a * key_frac * attn
        if kv_mode == 'fresh':
            # k and v are re-projected for the tokens that already left.
            cost += (1.0 - a) * (6.0 * 2.0 * d * d)
        eet += cost
    blocks_only = eet / dense if dense else 1.0
    total = ((eet + head_flops_per_token) / (dense + head_flops_per_token)
             if (dense + head_flops_per_token) else 1.0)
    return blocks_only, total


# ---------------------------------------------------------------------------
def fit_exponent(points):
    """Least-squares slope of log(bpb) against log(active FLOPs/token).

    points: list of (flops_per_token, bpb) from ISO-DATA dense runs. Two points give
    the local slope exactly; more are fitted.
    """
    pts = [(math.log(f), math.log(b)) for f, b in points if f and b]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    return num / den if den else None


def allowed_gap(bpb_dense, ratio, exponent):
    """bpb a model is allowed to lose for spending `ratio` of dense's compute."""
    if ratio is None or ratio <= 0 or exponent is None:
        return None
    return bpb_dense * (ratio ** exponent - 1.0)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-base", default="out/eet_p02")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--target-active-frac", type=float, default=0.10)
    ap.add_argument("--min-exit-layer", type=int, default=1)
    ap.add_argument("--capacity-schedule", default="bell")
    ap.add_argument("--n-embd", type=int, default=0,
                    help="model width; 0 = infer from nanochat's depth->width rule (64*depth)")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--window-pattern", default="SSSL")
    ap.add_argument("--ffn-mult", type=float, default=4.0)
    ap.add_argument("--fallback-exponent", type=float, default=-0.085,
                    help="iso-data log-log slope to use when the dense controls are missing "
                         "(default measured from mst_isodata.html near d8)")
    ap.add_argument("--pass-t1t2", type=float, default=0.045, help="pre-registered T1/T2 threshold")
    ap.add_argument("--oracle", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    log = args.log or os.path.join(args.out_base, f"sweep_eet_p02_d{args.depth}.log")
    if not os.path.exists(log):
        print(f"[report] no log at {log}")
        return 1
    runs = parse_log(log)

    d = args.n_embd or 64 * args.depth
    T = args.seq_len
    windows = [(-1 if c.upper() == 'L' else T // 2) for c in args.window_pattern]
    _, per_block = bell_capacities(args.depth, args.min_exit_layer,
                                   args.target_active_frac, args.capacity_schedule)

    dense_tag = f"DENSE_D{args.depth}"
    dense = runs.get(dense_tag)
    if not dense or dense['val_bpb'] is None:
        print(f"[report] dense control {dense_tag} has no validation bpb yet; nothing to compare")
        return 1
    bpb_dense = dense['val_bpb']
    flops_dense = dense['flops_active'] or dense['flops_total']

    # bpb is only comparable at one vocabulary, and the repo ships a 265-token stub at
    # ./tokenizer that trains without complaint. Refuse rather than print a table.
    vocabs = {t: r['vocab_size'] for t, r in runs.items() if r['vocab_size']}
    if vocabs and len(set(vocabs.values())) > 1 or (vocabs and min(vocabs.values()) < 1000):
        good = max(vocabs.values())
        stale = sorted(t for t, v in vocabs.items() if v != good)
        print(f"[report] ABORT: arms did not all train at the same vocabulary, so their bpb "
              f"is not comparable.")
        for t, v in sorted(vocabs.items()):
            flag = "  <-- RERUN THIS ARM" if v != good else ""
            print(f"           {t:<28} vocab_size={v}{flag}")
        if min(vocabs.values()) < 1000:
            print("[report] The small one is the byte-level stub tokenizer.")
        print(f"[report] Each arm is read from its LAST section in the log, so an arm listed "
              f"here has not been rerun since. Rerun: {' '.join(stale)}")
        return 1
    if dense['n_embd'] and not args.n_embd:
        args.n_embd = dense['n_embd']

    # Split the printed per-token FLOPs into blocks and everything else (the LM head and
    # embeddings), so the head can be charged at full price to every arm.
    blocks_analytic = sum(sum(block_flops_per_token(d, T, windows[i % len(windows)], args.ffn_mult))
                          for i in range(args.depth))
    counted_head = max(0.0, (flops_dense or blocks_analytic) - blocks_analytic)
    # GPT.estimate_flops subtracts wte from the parameter count, so a TIED lm_head
    # vanishes from the printed figure entirely. The head is still 6*d*V per token at
    # runtime and routing never touches it, so a Pareto claim that ignores it overstates
    # the saving. Price it explicitly and report both axes.
    v_pad = ((args.vocab_size + 63) // 64) * 64
    true_head = 6.0 * d * v_pad
    head_flops = max(counted_head, true_head)

    print("=" * 78)
    print(f"  EET P02 REPORT   depth={args.depth}  d_model={d}  target_active={args.target_active_frac}")
    print(f"  dense control:   {dense_tag}  val_bpb={bpb_dense:.4f}"
          + (f"  dt={dense['dt_ms']:.1f}ms" if dense['dt_ms'] else ""))
    print(f"  active FLOPs/token: printed={flops_dense:.4e}  blocks={blocks_analytic:.4e}")
    print(f"  LM head: counted by estimate_flops={counted_head:.4e}, actually run={true_head:.4e} "
          f"({100*true_head/(blocks_analytic+true_head):.1f}% of the honest total, NOT reduced by routing)")
    if counted_head < true_head * 0.5:
        print("  NOTE: the repo's FLOP counter drops the tied LM head. The 'flop_r' column")
        print("        below prices it back in, so it is larger (and the allowed gap smaller)")
        print("        than the same ratio computed off the printed axis.")
    print("  per-block active fraction: " + " ".join(f"{f:.3f}" for f in per_block))
    print("=" * 78)

    # --- the dense iso-data curve, measured in this sweep -------------------
    ctrl_points = [(flops_dense, bpb_dense)]
    for tag, r in runs.items():
        if tag.startswith("DENSE_ISOFLOP") and r['val_bpb'] is not None:
            f = r['flops_active'] or r['flops_total']
            if f:
                ctrl_points.append((f, r['val_bpb']))
    exponent = fit_exponent(ctrl_points)
    if exponent is None:
        exponent = args.fallback_exponent
        print(f"\n[!] Only one dense point. Falling back to exponent {exponent:+.4f} from")
        print("    mst_isodata.html. Every allowed-gap number below is then borrowed, not")
        print("    measured: run the DENSE_ISOFLOP controls before quoting any of this.")
    else:
        print(f"\n  dense iso-data curve: bpb ~ F^{exponent:+.4f}  "
              f"from {len(ctrl_points)} points at {dense['tokens']:,} tokens" if dense['tokens']
              else f"\n  dense iso-data curve: bpb ~ F^{exponent:+.4f} from {len(ctrl_points)} points")

    # --- per-arm table ------------------------------------------------------
    kv_of = {'T1_FRESH': 'fresh', 'T1_STALE': 'stale', 'T3_BOTH': 'stale'}
    rows = []
    print("")
    print(f"{'arm':<22}{'bpb':>8}{'gap':>8}{'flop_r':>9}{'wall_r':>8}"
          f"{'allow_F':>9}{'allow_W':>9}{'verdict':>10}")
    print("-" * 83)
    for tag in sorted(runs):
        if tag.startswith("DENSE"):
            continue
        r = runs[tag]
        if r['val_bpb'] is None:
            print(f"{tag.rsplit('_D', 1)[0]:<22}{'-- most recent run produced no validation bpb (failed) --':>60}")
            continue
        base = tag.rsplit('_D', 1)[0]
        kv = next((v for k, v in kv_of.items() if base.startswith(k)), 'none')
        r_blocks, r_total = eet_flop_ratio(per_block, d, T, windows, kv,
                                           args.ffn_mult, head_flops)
        wall_r = (r['dt_ms'] / dense['dt_ms']) if (r['dt_ms'] and dense['dt_ms']) else None
        gap = r['val_bpb'] - bpb_dense
        allow_f = allowed_gap(bpb_dense, r_total, exponent)
        allow_w = allowed_gap(bpb_dense, wall_r, exponent)

        if base.startswith(('T1_', 'T2_')):
            ok = gap <= args.pass_t1t2
            verdict = "PASS" if ok else "fail"
        elif base.startswith('T3_'):
            ok = allow_w is not None and gap <= allow_w
            verdict = "PASS" if ok else "fail"
        else:
            verdict = "-"
        rows.append(dict(tag=tag, bpb=r['val_bpb'], gap=gap, flop_ratio=r_total,
                         flop_ratio_blocks=r_blocks, wall_ratio=wall_r,
                         allowed_flops=allow_f, allowed_wall=allow_w, verdict=verdict))
        print(f"{base:<22}{r['val_bpb']:>8.4f}{gap:>+8.4f}{r_total:>9.3f}"
              f"{(f'{wall_r:.3f}' if wall_r else '   n/a'):>8}"
              f"{(f'{allow_f:+.4f}' if allow_f is not None else '  n/a'):>9}"
              f"{(f'{allow_w:+.4f}' if allow_w is not None else '  n/a'):>9}"
              f"{verdict:>10}")

    # --- the oracle ---------------------------------------------------------
    oracle = None
    if args.oracle and os.path.exists(args.oracle):
        oracle = json.load(open(args.oracle))
        print("")
        print("  T0A context oracle (dense checkpoint, no training):")
        for k, v in sorted(oracle.get('delta_bpb', {}).items()):
            if k.endswith('/dense'):
                continue
            print(f"    delta_bpb[{k:<14}] = {v:+.4f}")
        g = oracle.get('delta_bpb', {}).get('freq/ctx')
        if g is not None:
            print(f"    gate: {'PASSED' if oracle.get('gate_passed') else 'FAILED'} "
                  f"(threshold {oracle.get('gate_delta_bpb', 0.02)})")

    # --- verdict ------------------------------------------------------------
    print("")
    print("  PRE-REGISTERED VERDICT")
    t1 = [r for r in rows if r['tag'].startswith('T1_')]
    t2 = [r for r in rows if r['tag'].startswith('T2_')]
    t3 = [r for r in rows if r['tag'].startswith('T3_')]
    base_gap = next((r['gap'] for r in rows if r['tag'].startswith('EET_BASE')), None)
    if base_gap is not None:
        print(f"    EET baseline gap reproduced: {base_gap:+.4f} bpb")
    for name, group in (("T1 (context)", t1), ("T2 (coverage)", t2)):
        if not group:
            print(f"    {name}: not run")
            continue
        best = min(group, key=lambda r: r['gap'])
        ok = best['gap'] <= args.pass_t1t2
        print(f"    {name}: best {best['tag']} gap {best['gap']:+.4f} "
              f"vs threshold {args.pass_t1t2:.3f} -> {'PASS' if ok else 'FAIL'}")
    if t3:
        b = min(t3, key=lambda r: r['gap'])
        if b['allowed_wall'] is not None:
            ok = b['gap'] <= b['allowed_wall']
            print(f"    T3 (Pareto): gap {b['gap']:+.4f} vs allowed {b['allowed_wall']:+.4f} "
                  f"at wallclock ratio {b['wall_ratio']:.3f} -> {'PASS' if ok else 'FAIL'}")
            print("    T3 also requires a repeat at a second depth with the gap not widening.")
    else:
        print("    T3 (Pareto): not run")

    if t1 and t2:
        if (min(r['gap'] for r in t1) > args.pass_t1t2 and
                min(r['gap'] for r in t2) > args.pass_t1t2):
            print("")
            print("    BOTH MECHANISMS FAILED. The 'the gap is architectural' verdict in")
            print("    eet_experiment_log.md is confirmed on its two strongest remaining")
            print("    challengers. Close the direction; do not sweep further flags.")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or '.', exist_ok=True)
        json.dump({'depth': args.depth, 'bpb_dense': bpb_dense, 'exponent': exponent,
                   'head_flop_share': head_flops / (flops_dense or 1),
                   'per_block_active': per_block, 'arms': rows, 'oracle': oracle},
                  open(args.json_out, 'w'), indent=2)
        print(f"\n[report] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
