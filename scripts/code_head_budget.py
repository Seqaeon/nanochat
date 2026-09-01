"""
Matched token budget for Structured Code Head sweeps.

Prints ONE integer to stdout: the training token budget that every arm of an SCH
sweep must be given.

WHY THIS EXISTS.  ``scripts/base_train.py`` sizes the token budget from
``transformer_matrices + lm_head`` parameters times ``--target-param-data-ratio``.
That is the right default when the arms differ in the backbone, and exactly the
wrong one here: the whole point of a code head is that it has far fewer head
parameters than a dense softmax (17x fewer at V=32768, d=512, order 4).  Left on
the default, the code arms would silently receive a much smaller token budget
than the dense baseline and every quality comparison in the paper would be
confounded by data, not architecture.

So the SCH sweeps compute the DENSE arm's Chinchilla budget once, with this
script, and pin it on every arm with ``--target-tokens``.

    TARGET_TOKENS=$(python3 -m scripts.code_head_budget --depth 8 --ratio 10.5)

Diagnostics go to stderr so command substitution stays clean.
"""

from __future__ import annotations

import argparse
import math
import sys

import torch


def build_dense_meta(depth: int, vocab_size: int, aspect_ratio: int, head_dim: int,
                     model_dim: int, seq_len: int, window_pattern: str):
    """Build the DENSE baseline on the meta device, using base_train's sizing.

    Mirrors ``build_model_meta`` in scripts/base_train.py: model_dim is nudged up
    to the nearest multiple of head_dim so that head_dim comes out exactly as
    requested.  Kept as a small copy rather than an import because base_train is
    a script with side effects at import time.
    """
    from nanochat.gpt import GPT, GPTConfig
    if model_dim > 0:
        base_model_dim = model_dim
    else:
        base_dim = depth * aspect_ratio
        base_model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = max(1, base_model_dim // head_dim)
    config = GPTConfig(
        sequence_len=seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=base_model_dim,
        window_pattern=window_pattern,
    )
    with torch.device("meta"):
        model = GPT(config)
    return model, config


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depth", type=int, required=True, help="number of transformer layers")
    p.add_argument("--ratio", type=float, default=10.5,
                   help="Chinchilla tokens:params ratio (base_train's default is 10.5)")
    p.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--model-dim", type=int, default=0, help="explicit width override (0 = aspect ratio)")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--window-pattern", type=str, default="SSSL")
    p.add_argument("--vocab-size", type=int, default=0,
                   help="0 = read it from the tokenizer at --tokenizer-dir")
    p.add_argument("--tokenizer-dir", type=str, default=None)
    p.add_argument("--round-to", type=int, default=262144,
                   help="round the budget down to a multiple of this (the standard total batch size, "
                        "so num_iterations comes out whole and identical across arms)")
    p.add_argument("--verbose", action="store_true", help="print the breakdown to stderr")
    args = p.parse_args()

    vocab_size = args.vocab_size
    if vocab_size <= 0:
        from nanochat.tokenizer import get_tokenizer
        vocab_size = get_tokenizer(tokenizer_dir=args.tokenizer_dir).get_vocab_size()

    model, config = build_dense_meta(args.depth, vocab_size, args.aspect_ratio,
                                     args.head_dim, args.model_dim, args.seq_len,
                                     args.window_pattern)
    sp = model.num_scaling_params()
    scaling = sp["transformer_matrices"] + sp["lm_head"]
    tokens = int(args.ratio * scaling)
    if args.round_to > 1:
        tokens = (tokens // args.round_to) * args.round_to

    if args.verbose:
        flops, _active_flops, _active_params = model.estimate_flops()
        print(f"depth={config.n_layer} d={config.n_embd} heads={config.n_head} "
              f"V={vocab_size}", file=sys.stderr)
        for k, v in sp.items():
            print(f"  {k:22s} {v:>14,}", file=sys.stderr)
        print(f"  {'scaling params':22s} {scaling:>14,}", file=sys.stderr)
        print(f"  {'ratio':22s} {args.ratio:>14.2f}", file=sys.stderr)
        print(f"  {'dense FLOPs/token':22s} {flops:>14,}", file=sys.stderr)
        print(f"  {'target tokens':22s} {tokens:>14,}", file=sys.stderr)
        print(f"  {'iterations @ 262144':22s} {tokens // 262144:>14,}", file=sys.stderr)

    print(tokens)


if __name__ == "__main__":
    main()
