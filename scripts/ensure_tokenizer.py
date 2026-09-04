"""Make a tokenizer directory usable, building whatever is missing.

A sweep at a different vocabulary size needs three artifacts, and discovering
that one of them is absent after the runs have been queued wastes the queue.
This builds them, skips whatever already exists, and is safe to call on every
invocation of a sweep script.

    python -m scripts.ensure_tokenizer --vocab-size 131072 --tokenizer-dir tokenizer_131k

What it ensures:
  tokenizer.pkl    the tokenizer itself             (scripts.tok_train)
  token_bytes.pt   bytes per token, for bits-per-byte (written by tok_train)
  freq_table.pt    unigram counts over the corpus   (scripts.code_assign)

The frequency table is not needed to train, but without it the decile breakdown
silently turns itself off and a hierarchical-softmax arm falls back to a
balanced tree instead of a Huffman one, so a sweep that wanted either would
finish and quietly not have it.

Training a tokenizer reads billions of characters. It is a one-time cost per
vocabulary size and it is why this prints what it is about to do.
"""

from __future__ import annotations

import argparse
import os
import functools
import subprocess
import sys

# Unbuffered: subprocess output interleaves with ours, and a message that arrives
# after the thing it describes is worse than no message.
print = functools.partial(__builtins__["print"] if isinstance(__builtins__, dict)
                          else __builtins__.print, flush=True)


def _exists(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def ensure(vocab_size: int, tokenizer_dir: str, data_dir: str | None = None,
           max_shards: int | None = None, max_chars: int | None = None,
           force: bool = False, run_eval: bool = False) -> str:
    """Build the missing pieces. Returns the tokenizer directory."""
    pkl = os.path.join(tokenizer_dir, "tokenizer.pkl")
    tb = os.path.join(tokenizer_dir, "token_bytes.pt")
    freq = os.path.join(tokenizer_dir, "freq_table.pt")

    if force or not (_exists(pkl) and _exists(tb)):
        print(f"[tokenizer] training vocab_size={vocab_size:,} into {tokenizer_dir}")
        print(f"[tokenizer] this reads a large slice of the corpus; one-time cost per size")
        cmd = [sys.executable, "-m", "scripts.tok_train",
               "--vocab-size", str(vocab_size), "--tokenizer-dir", tokenizer_dir]
        if data_dir:
            cmd += ["--data-dir", data_dir]
        if max_chars:
            cmd += ["--max-chars", str(max_chars)]
        if force:
            cmd += ["--force"]
        subprocess.run(cmd, check=True)
    else:
        print(f"[tokenizer] {tokenizer_dir} already has tokenizer.pkl and token_bytes.pt")

    if not (_exists(pkl) and _exists(tb)):
        raise SystemExit(f"[tokenizer] tok_train did not produce {pkl} and {tb}")

    if force or not _exists(freq):
        print(f"[tokenizer] building freq_table.pt (unigram counts over the corpus)")
        # Imported rather than shelled out: code_assign's main() would also parse
        # a mode we do not want, and the builder is a plain function.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from code_assign import build_freq_table
        from nanochat.tokenizer import get_tokenizer
        v = get_tokenizer(tokenizer_dir=tokenizer_dir).get_vocab_size()
        build_freq_table(v, tokenizer_dir, data_dir, max_shards)
    else:
        print(f"[tokenizer] {freq} already present")

    if run_eval:
        print(f"[tokenizer] evaluating compression (diagnostics only)")
        cmd = [sys.executable, "-m", "scripts.tok_eval", "--tokenizer-dir", tokenizer_dir]
        subprocess.run(cmd, check=False)     # never fatal: it measures, it does not build

    from nanochat.tokenizer import get_tokenizer
    got = get_tokenizer(tokenizer_dir=tokenizer_dir).get_vocab_size()
    if got != vocab_size:
        raise SystemExit(
            f"[tokenizer] {tokenizer_dir} has vocab_size {got:,}, expected {vocab_size:,}.\n"
            f"[tokenizer] A sweep pinned to one vocabulary must not silently run at another.\n"
            f"[tokenizer] If {got:,} is far below the target, BPE ran out of corpus: raise\n"
            f"[tokenizer] --max-chars, or check --data-dir points at the full shard set.\n"
            f"[tokenizer] If the directory holds a tokenizer of a different size, pass\n"
            f"[tokenizer] --force to rebuild it or point --tokenizer-dir somewhere else.")
    print(f"[tokenizer] ready: {tokenizer_dir}, vocab_size {got:,}")
    return tokenizer_dir


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--tokenizer-dir", type=str, required=True)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--max-shards", type=int, default=None)
    p.add_argument("--max-chars", type=int, default=None,
                   help="passed to tok_train; lower it to build a cheap tokenizer for a smoke test")
    p.add_argument("--force", action="store_true", help="rebuild even if present")
    p.add_argument("--eval", action="store_true", help="also run tok_eval (diagnostics only)")
    a = p.parse_args()
    ensure(a.vocab_size, a.tokenizer_dir, a.data_dir, a.max_shards,
           a.max_chars, a.force, a.eval)


if __name__ == "__main__":
    main()
