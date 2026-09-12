"""O5: whole-model binarisation sensitivity scan.

Where in a trained transformer is floating point actually load-bearing?  Binarise
one component class at a time, hold everything else in bf16, and measure val bpb.

Two variants, and the difference decides what a result MEANS (see LEARNINGS,
"A projection oracle and a free-fit oracle have OPPOSITE pass/fail asymmetries"):

  projection  sign(W) * scale, zero degrees of freedom.  PENALISES, because the
              weights were never shaped for the constraint.  A pass is informative,
              a failure is weak.
  fitted      not implemented here; it needs an optimisation loop against the
              frozen teacher and belongs on a real GPU.

WHAT "BINARISE" MEANS HERE, because it is easy to measure the wrong thing.  The
plan's section 1 definition is: every learned parameter one bit, EVERY MATMUL
OPERAND one bit, the operation itself XNOR + popcount into an integer accumulator.
This script therefore has two independent axes and reports them separately:

  --binarise weights   sign(W), activations untouched.  This is W1A16.  It is an
                       UPPER BOUND on a binary model's quality and it changes no
                       operation: still a bf16 GEMM over a matrix holding two
                       distinct values per row.
  --binarise acts      sign(x) on the layer input, weights untouched.  A16W1
                       inverted; isolates the half BitNet refuses to go below 8 bits.
  --binarise both      W1A1.  The only arm that corresponds to the plan's claim.

And the scale is a second axis, because a per-row float scale is a float parameter:
  --scale none         strict.  Nothing but signs.
  --scale row          XNOR-Net L1-optimal per-output-channel scale (weights) and
                       per-token scale (activations).  Permitted only as a Phase 1
                       ladder rung; section 4.1 says a surviving scalar breaks the
                       title.
The gap between them prices exactly what the scales are worth.

Run:
  python -m scripts.o5_sensitivity --ckpt out/dense_d8_V32k_model_001014.pt
"""
import argparse
import time

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb

HEAD_DIM, ASPECT = 128, 64

# Substring patterns into the state dict.  Verified against
# out/dense_d8_V32k_model_001014.pt, which has exactly 60 keys.
COMPONENTS = {
    "attn.qkv":     [".attn.c_q.", ".attn.c_k.", ".attn.c_v."],
    "attn.c_proj":  [".attn.c_proj."],
    "attn.ve_gate": [".attn.ve_gate."],
    "mlp.c_fc":     [".mlp.c_fc."],
    "mlp.c_proj":   [".mlp.c_proj."],
    "lm_head":      ["lm_head."],
    "wte":          ["transformer.wte."],
    "value_embeds": ["value_embeds."],
}
AGGREGATES = {
    "ALL body matmuls": [".attn.c_q.", ".attn.c_k.", ".attn.c_v.", ".attn.c_proj.",
                         ".mlp.c_fc.", ".mlp.c_proj."],
    "ALL interfaces":   ["transformer.wte.", "lm_head.", "value_embeds."],
    "EVERYTHING":       [".attn.c_q.", ".attn.c_k.", ".attn.c_v.", ".attn.c_proj.",
                         ".mlp.c_fc.", ".mlp.c_proj.", ".attn.ve_gate.",
                         "transformer.wte.", "lm_head.", "value_embeds."],
}
# NOTE: this model has NO learned normalisation parameters.  nanochat's `norm` is a
# parameterless F.rms_norm (gpt.py:911), so there is no "binarise the norms" arm and
# section 3.3 of the plan is about removing an OPERATION, not parameters.


def build_config(depth, vocab_size, seq_len, wp):
    base = depth * ASPECT
    n_embd = ((base + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    n_head = n_embd // HEAD_DIM
    return GPTConfig(sequence_len=seq_len, vocab_size=vocab_size, n_layer=depth,
                     n_head=n_head, n_kv_head=n_head, n_embd=n_embd, window_pattern=wp)


def binarise_(t: torch.Tensor, scale: str) -> torch.Tensor:
    """sign(W) times the L1-optimal per-row scalar (XNOR-Net), or bare sign."""
    s = torch.sign(t)
    s[s == 0] = 1.0
    if scale == "row":
        s = s * t.abs().mean(dim=-1, keepdim=True)
    return s.to(t.dtype)


def _sign(t, scale, dim=-1):
    s = torch.sign(t)
    s = torch.where(s == 0, torch.ones_like(s), s)
    if scale == "row":
        s = s * t.abs().mean(dim=dim, keepdim=True)
    return s.to(t.dtype)


def install_act_hooks(model, patterns, scale):
    """Binarise the INPUT to every matched Linear: the other matmul operand."""
    handles = []

    def hook(mod, args):
        if not args:
            return None
        x = args[0]
        return (_sign(x, scale),) + tuple(args[1:])

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and any(p.strip(".") in name for p in patterns):
            handles.append(mod.register_forward_pre_hook(hook))
    return handles


def apply_binarisation(model, base_sd, patterns, scale):
    sd = {}
    touched = 0
    for k, v in base_sd.items():
        if v.dim() >= 2 and any(p in k for p in patterns):
            sd[k] = binarise_(v, scale)
            touched += v.numel()
        else:
            sd[k] = v
    model.load_state_dict(sd, strict=True)
    return touched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/dense_d8_V32k_model_001014.pt")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--window-pattern", default="SSSL")
    ap.add_argument("--tokenizer-dir", default="tokenizer")
    ap.add_argument("--data-dir", default=None, help="parquet shard dir; val split is the LAST shard")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--eval-steps", type=int, default=20)
    ap.add_argument("--scale", nargs="+", default=["row", "none"])
    ap.add_argument("--binarise", nargs="+", default=["weights", "acts", "both"],
                    choices=["weights", "acts", "both"],
                    help="which matmul operand(s) to binarise; 'both' is the only arm "
                         "matching the plan's definition")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these component names")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = build_config(a.depth, a.vocab, a.seq, a.window_pattern)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=dev)
    base_sd = torch.load(a.ckpt, map_location=dev, weights_only=False)
    base_sd = base_sd.get("model", base_sd) if isinstance(base_sd, dict) and "model" in base_sd else base_sd
    model.load_state_dict(base_sd, strict=True)
    model.eval()

    tok = get_tokenizer(a.tokenizer_dir) if a.tokenizer_dir else get_tokenizer()
    token_bytes = get_token_bytes(device=dev, tokenizer_dir=a.tokenizer_dir)

    def val_batches():
        return tokenizing_distributed_data_loader_bos_bestfit(
            tok, a.batch, a.seq, split="val", device=dev, data_dir=a.data_dir)

    def score():
        bpb, _ = evaluate_bpb(model, val_batches(), a.eval_steps, token_bytes)
        return float(bpb)

    t0 = time.time()
    baseline = score()
    print(f"baseline dense bpb = {baseline:.4f}   ({time.time()-t0:.0f}s per eval, "
          f"{a.eval_steps} steps x {a.batch} x {a.seq} tokens)")
    print()

    items = list(COMPONENTS.items()) + list(AGGREGATES.items())
    if a.only:
        items = [(k, v) for k, v in items if k in a.only]

    combos = [(m, s) for m in a.binarise for s in a.scale]
    hdr = f"{'component':<20}{'params':>13}"
    for m, s in combos:
        hdr += f"{m[:4]+'/'+s[:4]:>13}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for name, pats in items:
        line = ""
        deltas = {}
        n = 0
        for mode, sc in combos:
            handles = []
            if mode in ("weights", "both"):
                n = apply_binarisation(model, base_sd, pats, sc)
            else:
                model.load_state_dict(base_sd, strict=True)
                n = sum(v.numel() for k, v in base_sd.items()
                        if v.dim() >= 2 and any(p in k for p in pats))
            if mode in ("acts", "both"):
                handles = install_act_hooks(model, pats, sc)
            b = score()
            for h in handles:
                h.remove()
            deltas[(mode, sc)] = b - baseline
            line += f"{b-baseline:+13.4f}"
        rows.append((name, n, deltas))
        print(f"{name:<20}{n:13,d}{line}")
    model.load_state_dict(base_sd, strict=True)

    print()
    key = ("both", a.scale[0]) if "both" in a.binarise else combos[0]
    print(f"sensitivity ranking by delta bpb, mode={key[0]} scale={key[1]}, worst first:")
    for name, n, d in sorted(rows, key=lambda r: -r[2][key]):
        print(f"  {d[key]:+8.4f}  {name:<20} ({n:,} params)")
    if "weights" in a.binarise and "both" in a.binarise:
        print()
        print("weights-only vs both: how much of the damage is the ACTIVATIONS")
        for name, n, d in rows:
            w, bo = d[("weights", a.scale[0])], d[("both", a.scale[0])]
            share = (bo - w) / bo * 100 if abs(bo) > 1e-9 else float("nan")
            print(f"  {name:<20} W1A16 {w:+8.4f}   W1A1 {bo:+8.4f}   activations are {share:5.1f}% of it")
    print()
    print("W1A16 IS NOT BINARISATION. It changes no operation: still a bf16 GEMM over a")
    print("matrix holding two values per row. Only the 'both' column corresponds to the")
    print("plan's claim, and only --scale none is strict; a per-row float scale is a float.")
    print()
    print("READ THIS BEFORE USING THE NUMBERS: projection PENALISES. A component that")
    print("survives here is strong evidence it survives from-scratch training; one that")
    print("fails here is weak evidence, because a model trained binary shapes its own")
    print("weight distribution for sign() and this one never did.")


if __name__ == "__main__":
    main()
