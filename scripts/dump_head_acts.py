"""Dump the input to ``lm_head`` from a trained checkpoint.

The output head's job is to turn one d-dimensional vector per token into V logits,
so every question about head architecture is a question about the joint distribution
of those vectors and the head's rows. The head weights alone answer half of it; the
activations answer the other half, and they are tiny: 4,096 tokens at d=768 is 6 MB
in fp16 against a checkpoint of several gigabytes.

This exists because the natural one-liner does not work -- ``torch.load`` on a
nanochat checkpoint returns a state dict, not a module, so there is nothing to hook.
The model has to be rebuilt from the ``model_config`` recorded beside it.

  python -m scripts.dump_head_acts \\
      out/c08_vocab131k/d8/BASE_dense_s1/depth_8/ckpt_base/base \\
      --tokenizer-dir tokenizer_131k --data-dir data --tokens 4096 \\
      --out acts_d8_v131k.pt
"""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer


class _Stop(Exception):
    """Raised from the pre-hook so the V-wide logits are never materialised."""


def build(ckpt_dir: str, step: int | None):
    models = sorted(glob.glob(os.path.join(ckpt_dir, "model_*.pt")))
    assert models, f"no model_*.pt under {ckpt_dir}"
    path = (os.path.join(ckpt_dir, f"model_{step:06d}.pt") if step is not None
            else models[-1])
    meta_path = path.replace("model_", "meta_").replace(".pt", ".json")
    assert os.path.exists(meta_path), (
        f"{meta_path} is missing; the model_config recorded beside the weights is "
        "what says how to rebuild the module")
    meta = json.load(open(meta_path))
    cfg = GPTConfig(**meta["model_config"])
    cfg._tokenizer_dir = None
    model = GPT(cfg)
    sd = torch.load(path, weights_only=True, map_location="cpu")
    sd = sd.get("model", sd)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not [k for k in missing if "lm_head" in k or "transformer" in k], \
        f"checkpoint does not fill the model: missing {missing[:5]}"
    return model, cfg, meta, path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint_dir")
    ap.add_argument("--step", type=int, default=None, help="default: the latest")
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--out", default="acts.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--also-head", action="store_true",
                    help="save lm_head.weight alongside, so one file carries both")
    args = ap.parse_args()

    model, cfg, meta, path = build(args.checkpoint_dir, args.step)
    model.eval().to(args.device)
    print(f"[dump] {os.path.basename(path)}  L={cfg.n_layer} d={cfg.n_embd} "
          f"V={cfg.vocab_size}  val_bpb={meta.get('val_bpb')}")

    tok = get_tokenizer(args.tokenizer_dir)
    shards = sorted(glob.glob(os.path.join(args.data_dir, "*.parquet")))
    assert shards, f"no parquet shards under {args.data_dir}"
    import pyarrow.parquet as pq
    col = pq.read_table(shards[0]).column("text")
    ids: list[int] = []
    for i in range(len(col)):
        ids.extend(tok.encode(str(col[i])))
        if len(ids) >= args.tokens:
            break
    n = (args.tokens // args.seq_len) or 1
    x = torch.tensor(ids[:n * args.seq_len], dtype=torch.long,
                     device=args.device).view(n, args.seq_len)

    grabbed = {}
    def grab(_m, inp):
        grabbed["x"] = inp[0].detach()
        raise _Stop
    handle = model.lm_head.register_forward_pre_hook(grab)
    with torch.no_grad():
        try:
            model(x)
        except _Stop:
            pass
    handle.remove()
    H = grabbed["x"].reshape(-1, cfg.n_embd).half().cpu()

    payload = H if not args.also_head else {
        "acts": H, "lm_head": model.lm_head.weight.detach().half().cpu()}
    torch.save(payload, args.out)
    mb = os.path.getsize(args.out) / 1e6
    print(f"[dump] wrote {args.out}: {tuple(H.shape)} activations, {mb:.1f} MB"
          + ("  (+ lm_head)" if args.also_head else ""))


if __name__ == "__main__":
    main()
