"""
Build the token codes that a Structured Code Head consumes.

A code head's code assignment is a *prior over which tokens should share
statistical strength*.  A softmax gives every token a free, independent row, so
every token needs enough occurrences to fit its own row.  A code head forces
sharing according to c(w), which is why the assignment is a first-class
experimental variable and not an implementation detail.

Two of the arms this script produces are directly opposed, and the tension is a
claimed contribution rather than an accident:

  * an **error-correcting code** maximises the minimum Hamming distance, so that
    no two tokens are confusable;
  * **generalisation** wants semantically similar tokens at *small* Hamming
    distance, so that a monomial trained on one token transfers to its neighbours.

``--mode semantic_ecc`` with ``--ecc-bits P`` sweeps that tension on a single
axis: a semantic base code plus P random GF(2) parity bits, which raise minimum
distance without disturbing the base assignment.

Usage:

    # semantic codes from a trained checkpoint's input embeddings
    python -m scripts.code_assign --mode semantic --semantic-method itq --bits 24 \\
        --from-checkpoint out/base_checkpoints --out out/codes/semantic_b24.pt --report

    # the same, plus 8 parity bits (B becomes 32)
    python -m scripts.code_assign --mode semantic_ecc --semantic-method itq --bits 24 \\
        --ecc-bits 8 --from-checkpoint out/base_checkpoints --out out/codes/sem_ecc_b32.pt

    # non-semantic arms, for the code-assignment comparison
    python -m scripts.code_assign --mode ecc    --bits 32 --out out/codes/ecc_b32.pt --report
    python -m scripts.code_assign --mode random --bits 32 --out out/codes/rand_b32.pt --report

    # the frequency table the decile metrics and the Huffman baseline both need
    python -m scripts.code_assign --build-freq-table --max-shards 8

Feed the result to training with ``--sch-code-mode file --sch-code-path <PATH>``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

from nanochat.code_head import (build_codes, code_statistics, full_phi_width,
                                minimal_bits, load_freq_table)

# Godey et al. (2024) report that head performance degrades below a logit-matrix
# rank of roughly this value, regardless of model size.  It is the line the
# expansion width M has to clear, so --report prints every M against it.
RANK_THRESHOLD = 1000


# ---------------------------------------------------------------------------
# Embedding sources
# ---------------------------------------------------------------------------

def load_embeddings(args, vocab_size: int) -> torch.Tensor:
    """Return a (V, d) float32 embedding matrix to derive semantic codes from."""
    if args.embeddings:
        E = torch.load(args.embeddings, weights_only=True, map_location="cpu").float()
    elif args.from_checkpoint:
        from nanochat.checkpoint_manager import load_model_from_dir
        model, _tok, _meta = load_model_from_dir(
            args.from_checkpoint, "cpu", "eval",
            model_tag=args.model_tag, step=args.step, tokenizer_dir=args.tokenizer_dir)
        if getattr(args, "output_embedding", False):
            # The output head, not the input table.  For product codes this is
            # what matters: measured on the c00 dense head, fitting the code to
            # the OUTPUT embedding captures 46.5% of its logit energy at M=2048
            # against 21.7% when fitted to the input table, because untied input
            # and output embeddings do not share a column space.
            head = model.lm_head
            assert hasattr(head, "weight"), \
                "this checkpoint's head has no dense weight to read; use a dense proxy"
            E = head.weight.detach().float().cpu()
        else:
            wte = model.transformer.wte
            assert hasattr(wte, "weight"), \
                "the checkpoint has a coded input embedding, so it carries no table to read"
            E = wte.weight.detach().float().cpu()
    else:
        raise SystemExit("semantic modes need --from-checkpoint or --embeddings")
    assert E.shape[0] >= vocab_size, \
        f"embedding table has {E.shape[0]} rows, need at least {vocab_size}"
    return E[:vocab_size].contiguous()


# ---------------------------------------------------------------------------
# K-ary product codes
# ---------------------------------------------------------------------------

def _kmeans(X: torch.Tensor, K: int, iters: int = 25, seed: int = 0):
    """Lloyd's algorithm with k-means++-ish seeding by random distinct rows."""
    N = X.shape[0]
    gen = torch.Generator().manual_seed(seed)
    C = X[torch.randperm(N, generator=gen)[:K]].clone()
    a = torch.zeros(N, dtype=torch.long)
    for _ in range(iters):
        # Chunk the distance matrix: N x K in one go is 32768 x 256 floats per
        # group, fine here, but V=131072 with K=1024 is not.
        for s in range(0, N, 8192):
            a[s:s + 8192] = torch.cdist(X[s:s + 8192], C).argmin(1)
        for k in range(K):
            m = a == k
            if m.any():
                C[k] = X[m].mean(0)
            else:
                # Re-seed an empty cluster on the worst-served point, otherwise
                # the codebook silently shrinks and M is smaller than advertised.
                far = torch.cdist(X, C).min(1).values.argmax()
                C[k] = X[far]
    return a, C


def product_codes(E: torch.Tensor, groups: int, codebook: int, seed: int = 0,
                  iters: int = 25):
    """Fit a ``groups``-digit code over a ``codebook``-symbol alphabet.

    Split the embedding's feature axis into ``groups`` contiguous blocks and
    k-means each block independently.  This is product quantisation (Jegou et
    al., 2011) used as a *basis* rather than as a compressor: the resulting
    one-hot-per-group ``Phi`` has ``groups * codebook`` columns whose cells
    follow the embedding geometry, instead of the token-id lattice a binary code
    inherits.

    Measured on the c00 dense head at V=32768: a binary order-2 code captures
    1.79% of the head's logit energy at M=120, while this captures 21.93% at
    M=128 and keeps growing with M (46.52% at M=2048) where the monomial ladder
    saturates.

    Returns ``(assign, distortion)`` with ``assign`` of shape (V, groups) int32.
    """
    V, d = E.shape
    assert d % groups == 0, \
        f"embedding width {d} is not divisible by groups={groups}"
    sub = d // groups
    Ec = E - E.mean(0, keepdim=True)
    out = torch.empty(V, groups, dtype=torch.int32)
    dist = 0.0
    for j in range(groups):
        block = Ec[:, j * sub:(j + 1) * sub].contiguous()
        a, C = _kmeans(block, codebook, iters=iters, seed=seed + j)
        out[:, j] = a.to(torch.int32)
        dist += float((block - C[a]).pow(2).sum() / block.pow(2).sum().clamp_min(1e-9))
    return out, dist / groups


# ---------------------------------------------------------------------------
# Semantic code assignment
# ---------------------------------------------------------------------------

def itq_codes(E: torch.Tensor, bits: int, iters: int = 50, seed: int = 0):
    """Iterative Quantisation (ITQ) binary hashing.

    Centre, project to ``bits`` principal directions, then find the rotation R
    minimising the quantisation error ``|| sign(XR) - XR ||_F`` by alternating a
    sign step with an orthogonal Procrustes step.  ITQ is the standard cheap
    choice for turning a continuous space into balanced binary codes that
    preserve cosine neighbourhoods, which is exactly the property the
    tail-generalisation hypothesis needs: a monomial trained on one token should
    mean something for tokens near it.

    Returns ``(codes uint8 (V, bits), projections float (V, bits))``.  The
    projections are kept because their magnitudes rank how confident each bit is,
    which is what collision repair flips first.
    """
    torch.manual_seed(seed)
    X = E - E.mean(dim=0, keepdim=True)
    # PCA by SVD on the centred matrix; take the top `bits` directions.
    q = min(bits + 16, min(X.shape))
    _U, _S, Vh = torch.svd_lowrank(X, q=q, niter=8)
    W = Vh[:, :bits] if Vh.shape[1] >= bits else Vh
    if W.shape[1] < bits:
        raise SystemExit(f"embedding dimension {E.shape[1]} cannot supply {bits} PCA directions")
    Z = X @ W                                   # (V, bits)
    R = torch.linalg.qr(torch.randn(bits, bits))[0]
    for _ in range(iters):
        Bsign = torch.sign(Z @ R)
        Bsign[Bsign == 0] = 1.0
        # Procrustes: the rotation closest to mapping Z onto the signs.
        U, _S2, Vh2 = torch.linalg.svd(Bsign.t() @ Z, full_matrices=False)
        R = (U @ Vh2).t()
    P = Z @ R
    return (P > 0).to(torch.uint8), P


def rqvae_codes(E: torch.Tensor, bits: int, seed: int = 0):
    """Residual balanced binary partition, in the spirit of RQ-VAE semantic IDs.

    At each level, split the current cluster along its top principal component at
    the MEDIAN, so the tree stays balanced and every code prefix is reachable,
    then recurse on the residual within each half.  The dependencies between
    successive bits are the point: in recsys and audio RVQ those dependencies are
    load-bearing, and an order-1 head is precisely the model that cannot use them.

    Returns ``(codes uint8 (V, bits), margins float (V, bits))`` where margins are
    the signed distances to each split, used for collision repair.
    """
    torch.manual_seed(seed)
    V = E.shape[0]
    codes = torch.zeros(V, bits, dtype=torch.uint8)
    margins = torch.zeros(V, bits)
    clusters = [torch.arange(V)]
    for b in range(bits):
        next_clusters = []
        for idx in clusters:
            if idx.numel() <= 1:
                next_clusters.append(idx)
                continue
            X = E[idx]
            X = X - X.mean(dim=0, keepdim=True)
            # Top principal direction of this cluster.
            _U, _S, Vh = torch.svd_lowrank(X, q=min(4, min(X.shape)), niter=4)
            d = Vh[:, 0]
            proj = X @ d
            thresh = proj.median()
            right = proj > thresh
            # Median splits can tie; break ties by index so the halves stay balanced.
            if int(right.sum()) == 0 or int(right.sum()) == idx.numel():
                order = torch.argsort(proj)
                right = torch.zeros_like(right)
                right[order[idx.numel() // 2:]] = True
            codes[idx, b] = right.to(torch.uint8)
            margins[idx, b] = proj - thresh
            next_clusters.append(idx[right])
            next_clusters.append(idx[~right])
        clusters = [c for c in next_clusters if c.numel() > 0]
    return codes, margins


def repair_collisions(codes: torch.Tensor, confidence: torch.Tensor, max_flips: int = 3):
    """Make the assignment injective by moving each duplicate to its nearest free code.

    Two tokens sharing a code share a row of Phi and are therefore
    indistinguishable to the head at *any* interaction order, which puts a hard
    floor under the loss no matter how much the model trains.  So collisions are
    repaired rather than tolerated.

    Each duplicate keeps its position and moves to the closest unoccupied code in
    Hamming distance, searching flips in increasing order of |confidence| so the
    bits that carry the least semantic signal move first.  Repeated flipping of a
    fixed bit index would not work: it just moves collisions around.  Returns
    ``(codes, n_repaired)``.
    """
    import itertools as _it
    V, B = codes.shape
    assert B <= 63, "collision repair packs codes into int64"
    weights = 2 ** torch.arange(B, dtype=torch.int64)
    keys = (codes.to(torch.int64) * weights).sum(dim=1)

    occupied, dupes = set(), []
    for i, k in enumerate(keys.tolist()):
        if k in occupied:
            dupes.append(i)
        else:
            occupied.add(k)
    if not dupes:
        return codes, 0

    order = torch.argsort(confidence.abs(), dim=1).tolist()   # least confident bit first
    repaired = 0
    for i in dupes:
        base = int(keys[i])
        placed = None
        for r in range(1, max_flips + 1):
            for combo in _it.combinations(order[i], r):
                cand = base
                for b in combo:
                    cand ^= (1 << b)
                if cand not in occupied:
                    placed = cand
                    break
            if placed is not None:
                break
        if placed is None:
            raise SystemExit(
                f"could not place token {i} within {max_flips} bit flips at B={B}. "
                f"The code space is too crowded: B must exceed {minimal_bits(V)} with slack "
                f"for a semantic assignment. Try --bits {max(B + 4, minimal_bits(V) + 4)}")
        occupied.add(placed)
        keys[i] = placed
        repaired += 1

    shifts = torch.arange(B, dtype=torch.int64)
    codes = ((keys.unsqueeze(1) >> shifts) & 1).to(torch.uint8)
    assert len(occupied) == V
    return codes, repaired


def append_parity(codes: torch.Tensor, n_parity: int, seed: int) -> torch.Tensor:
    """Append ``n_parity`` random GF(2) parity bits (the ECC end of the axis)."""
    if n_parity <= 0:
        return codes
    gen = torch.Generator().manual_seed(seed + 7919)
    P = torch.randint(0, 2, (codes.shape[1], n_parity), generator=gen, dtype=torch.uint8)
    parity = (codes.to(torch.int32) @ P.to(torch.int32)) % 2
    return torch.cat([codes, parity.to(torch.uint8)], dim=1)


# ---------------------------------------------------------------------------
# Frequency table
# ---------------------------------------------------------------------------

def build_freq_table(vocab_size: int, tokenizer_dir: str | None, data_dir: str | None,
                     max_shards: int | None) -> str:
    """Count token occurrences over the corpus and cache ``freq_table.pt``.

    Uses the same construction as ``nanochat.eet.FrequencyPrior._load_or_compute``
    (pyarrow read, batched tokenizer.encode, bincount) so that the frequency
    signal the deciles use, the one the Huffman baseline builds its tree from,
    and the one EET's router reads are all literally the same table.
    """
    import pyarrow.parquet as pq
    import numpy as np
    from nanochat.dataset import resolve_data_dir, list_parquet_files
    from nanochat.tokenizer import get_tokenizer
    from nanochat.common import get_base_dir

    if tokenizer_dir is None:
        tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")
    data_dir = data_dir or resolve_data_dir()
    shards = list_parquet_files(data_dir=data_dir)
    if max_shards:
        shards = shards[:max_shards]
    tokenizer = get_tokenizer(tokenizer_dir=tokenizer_dir)
    freq = torch.zeros(vocab_size, dtype=torch.float32)
    for i, shard in enumerate(shards):
        print(f"[code_assign] shard {i + 1}/{len(shards)}: {os.path.basename(shard)}", file=sys.stderr)
        texts = [t for t in pq.read_table(shard, columns=["text"]).column("text").to_pylist() if t]
        if not texts:
            continue
        ids = tokenizer.encode(texts)
        flat = np.concatenate([np.array(t, dtype=np.int32) for t in ids if len(t)])
        freq += torch.bincount(torch.from_numpy(flat).long(), minlength=vocab_size)[:vocab_size]
    os.makedirs(tokenizer_dir, exist_ok=True)
    path = os.path.join(tokenizer_dir, "freq_table.pt")
    torch.save(freq, path)
    nz = int((freq > 0).sum())
    print(f"[code_assign] wrote {path}: {nz:,}/{vocab_size:,} ids seen, "
          f"{int(freq.sum()):,} occurrences", file=sys.stderr)
    return path


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def semantic_coherence(codes: torch.Tensor, E: torch.Tensor, n_pairs: int = 200_000,
                       seed: int = 0) -> dict:
    """Correlate code Hamming distance with embedding cosine distance.

    This is what makes the word "semantic" mean something instead of being a
    label on a filename.  A code that is semantically organised should show a
    clearly positive Spearman correlation; a random code shows approximately zero,
    and reporting both is what turns the code-assignment arm into a measurement.
    """
    V = codes.shape[0]
    g = torch.Generator().manual_seed(seed)
    i = torch.randint(0, V, (n_pairs,), generator=g)
    j = torch.randint(0, V, (n_pairs,), generator=g)
    keep = i != j
    i, j = i[keep], j[keep]
    ham = (codes[i].to(torch.int16) ^ codes[j].to(torch.int16)).sum(dim=1).float()
    En = torch.nn.functional.normalize(E, dim=-1)
    cos = 1.0 - (En[i] * En[j]).sum(dim=-1)
    def _spearman(a, b):
        ra = torch.argsort(torch.argsort(a)).float()
        rb = torch.argsort(torch.argsort(b)).float()
        ra = ra - ra.mean(); rb = rb - rb.mean()
        return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-9))
    return {"pairs": int(ham.numel()),
            "spearman_hamming_vs_cosine": _spearman(ham, cos),
            "pearson_hamming_vs_cosine": float(
                ((ham - ham.mean()) * (cos - cos.mean())).sum()
                / (ham.std() * cos.std() * ham.numel()).clamp_min(1e-9))}


def report(codes: torch.Tensor, E: torch.Tensor | None, vocab_size: int) -> dict:
    stats = code_statistics(codes)
    B = codes.shape[1]
    widths = {}
    for k in range(1, 5):
        if k > B:
            break
        M = full_phi_width(B, k)
        widths[k] = {"M": M, "clears_rank_threshold": M >= RANK_THRESHOLD,
                     "exceeds_vocab": M >= vocab_size}
    out = {"vocab_size": vocab_size, "bits": B, "statistics": stats, "widths": widths}
    if E is not None:
        out["semantic_coherence"] = semantic_coherence(codes, E)

    print(f"\ncode matrix: V={vocab_size:,}  B={B}", file=sys.stderr)
    print(f"  density                {stats['density']:.4f}", file=sys.stderr)
    print(f"  mean Hamming (sampled) {stats['mean_hamming_sampled']:.2f}", file=sys.stderr)
    print(f"  min  Hamming (sampled) {stats['min_hamming_sampled']}", file=sys.stderr)
    if stats["min_hamming_sampled"] <= 1:
        print("  NOTE: minimum distance 1 means the code has no error-correction slack, "
              "so the ECC-versus-semantic comparison is undefined at this B "
              f"(need B >= {minimal_bits(vocab_size) + 3})", file=sys.stderr)
    print(f"  expansion width M by interaction order (threshold {RANK_THRESHOLD}):", file=sys.stderr)
    for k, w in widths.items():
        flag = "clears" if w["clears_rank_threshold"] else "below "
        cap = "  (>= V, no longer a compression)" if w["exceeds_vocab"] else ""
        print(f"    k={k}  M={w['M']:>7,}  {flag} threshold{cap}", file=sys.stderr)
    if E is not None:
        sc = out["semantic_coherence"]
        print(f"  semantic coherence: Spearman(Hamming, cosine distance) = "
              f"{sc['spearman_hamming_vs_cosine']:+.4f} over {sc['pairs']:,} pairs "
              f"(a random code scores about 0)", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", type=str, default="semantic",
                   choices=["semantic", "semantic_ecc", "ecc", "random", "binary", "frequency",
                            "product"])
    p.add_argument("--semantic-method", type=str, default="itq", choices=["itq", "rqvae"])
    p.add_argument("--bits", type=int, default=0, help="B of the BASE code (0 = ceil(log2 V))")
    p.add_argument("--ecc-bits", type=int, default=0,
                   help="parity bits appended after the base code; final B = bits + ecc-bits")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--itq-iters", type=int, default=50)
    p.add_argument("--product-groups", type=int, default=8,
                   help="g: digits in the K-ary product code (--mode product)")
    p.add_argument("--product-codebook", type=int, default=256,
                   help="K: symbols per digit; the head's M is g*K")
    p.add_argument("--kmeans-iters", type=int, default=25)
    p.add_argument("--output-embedding", action="store_true",
                   help="fit the code to the checkpoint's lm_head instead of its wte")
    p.add_argument("--vocab-size", type=int, default=0, help="0 = read from the tokenizer")
    p.add_argument("--tokenizer-dir", type=str, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--max-shards", type=int, default=None)
    p.add_argument("--from-checkpoint", type=str, default=None,
                   help="checkpoint directory whose input embeddings define semantics")
    p.add_argument("--model-tag", type=str, default=None)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--embeddings", type=str, default=None, help=".pt with a (V, d) embedding matrix")
    p.add_argument("--out", type=str, default="", help="destination .pt for the (V, B) uint8 codes")
    p.add_argument("--report", action="store_true")
    p.add_argument("--build-freq-table", action="store_true",
                   help="compute and cache <tokenizer-dir>/freq_table.pt, then exit")
    args = p.parse_args()

    vocab_size = args.vocab_size
    if vocab_size <= 0:
        from nanochat.tokenizer import get_tokenizer
        vocab_size = get_tokenizer(tokenizer_dir=args.tokenizer_dir).get_vocab_size()

    if args.build_freq_table:
        build_freq_table(vocab_size, args.tokenizer_dir, args.data_dir, args.max_shards)
        return

    bits = args.bits if args.bits > 0 else minimal_bits(vocab_size)
    E, provenance = None, {}

    if args.mode == "product":
        # K-ary product code.  Written as int32 (V, g), not uint8 (V, B): the
        # symbols are not bits, so none of the binary checks below apply.
        E = load_embeddings(args, vocab_size)
        assign, distortion = product_codes(E, args.product_groups, args.product_codebook,
                                           seed=args.seed, iters=args.kmeans_iters)
        used = [int(assign[:, j].unique().numel()) for j in range(args.product_groups)]
        cells = args.product_codebook ** args.product_groups
        print(f"product code: g={args.product_groups} K={args.product_codebook} "
              f"M={args.product_groups * args.product_codebook}  "
              f"residual distortion {distortion * 100:.2f}%  "
              f"symbols used per group min/max {min(used)}/{max(used)}  "
              f"cells {cells:.3e} for {vocab_size} tokens")
        if not args.out:
            return
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        torch.save(assign, args.out)
        with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
            json.dump({"mode": "product", "groups": args.product_groups,
                       "codebook": args.product_codebook, "width": args.product_groups * args.product_codebook,
                       "seed": args.seed, "vocab_size": vocab_size,
                       "residual_distortion": distortion,
                       "symbols_used_per_group": used,
                       "embedding_source": args.embeddings or args.from_checkpoint,
                       "embedding_side": "lm_head" if args.output_embedding else "wte",
                       "embedding_dim": int(E.shape[1])}, f, indent=2, default=str)
        print(f"wrote {args.out}  (use --sch-phi-mode product --sch-product-source file "
              f"--sch-code-path {args.out})")
        return

    if args.mode in ("semantic", "semantic_ecc"):
        E = load_embeddings(args, vocab_size)
        if args.semantic_method == "itq":
            codes, confidence = itq_codes(E, bits, iters=args.itq_iters, seed=args.seed)
        else:
            codes, confidence = rqvae_codes(E, bits, seed=args.seed)
        codes, rounds = repair_collisions(codes, confidence)
        provenance["collision_repairs"] = rounds
        provenance["embedding_source"] = args.embeddings or args.from_checkpoint
        provenance["embedding_dim"] = int(E.shape[1])
        if args.mode == "semantic_ecc":
            codes = append_parity(codes, args.ecc_bits, args.seed)
    else:
        freqs = load_freq_table(vocab_size, args.tokenizer_dir) if args.mode == "frequency" else None
        if args.mode == "frequency" and freqs is None:
            raise SystemExit("--mode frequency needs freq_table.pt; run --build-freq-table first")
        codes = build_codes(vocab_size, bits, args.mode, args.seed,
                            freqs=freqs, ecc_bits=args.ecc_bits)

    assert codes.dtype == torch.uint8 and codes.dim() == 2
    assert codes.shape[0] == vocab_size
    assert int(codes.max()) <= 1, "codes must be binary"
    if codes.shape[1] <= 63:
        weights = 2 ** torch.arange(codes.shape[1], dtype=torch.int64)
        keys = (codes.to(torch.int64) * weights).sum(dim=1)
        n_unique = int(torch.unique(keys).numel())
        assert n_unique == vocab_size, \
            f"code assignment is not injective: {vocab_size - n_unique} tokens collide"

    info = report(codes, E, vocab_size) if args.report else {}

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        torch.save(codes, args.out)
        meta = {
            "mode": args.mode, "semantic_method": args.semantic_method,
            "base_bits": bits, "ecc_bits": args.ecc_bits,
            "total_bits": int(codes.shape[1]), "seed": args.seed,
            "vocab_size": vocab_size, "tokenizer_dir": args.tokenizer_dir,
            **provenance, **info,
        }
        side = os.path.splitext(args.out)[0] + ".json"
        with open(side, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        # A code matrix with no provenance is unreproducible, and the assignment
        # is an experimental variable, so the sidecar is not optional.
        print(f"[code_assign] wrote {args.out} ({vocab_size} x {codes.shape[1]} uint8) "
              f"and {side}", file=sys.stderr)
        print(args.out)


if __name__ == "__main__":
    main()
