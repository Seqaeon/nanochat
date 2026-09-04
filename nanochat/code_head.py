"""
Structured Code Output Heads (SCH).

The dense softmax head computes ``logit(w | h) = W_w . h`` with a learned row
``W_w`` per vocabulary item.  This module replaces that lookup with an inner
product against a *frozen, binary, structured* expansion of a per-token binary
code::

    logit(w | h) = phi_k(c(w))^T g(h)

``c(w) in {0,1}^B`` is a code assigned to token ``w`` before training, and
``phi_k`` is the vector of all monomials (products of bits) of that code up to
interaction order ``k``::

    phi_k(c) = [ c_1, ..., c_B,                       order 1: B terms
                 c_1 c_2, c_1 c_3, ...,               order 2: C(B,2) terms
                 ...                                  up to order k ]

    M = |phi_k| = sum_{j<=k} C(B, j)

Stacking the expansions row-wise gives ``Phi in {0,1}^{V x M}``, so the whole
family is exactly "a softmax whose output embedding matrix is frozen, binary and
structured".  Order 1 recovers independent-bit code prediction (Oda et al., ACL
2017); order ``B`` recovers the exact full softmax.

Why the ladder matters (this is the theory the experiments test):

  * With independent bits the log-probability is
    ``log P(w|h) = A(h) + sum_b c_b(w) s_b(h)``, because
    ``log sigma(s) - log sigma(-s) = s``.  ``A(h)`` does not depend on ``w`` and
    is removed by normalisation, so the effective logit matrix is exactly
    ``C S`` and has **rank <= B**.  At V=32768 that is rank 15, two orders of
    magnitude below the ~1000 head-rank threshold reported by Godey et al.
    (2024).
  * Order-k monomials contribute ``c_{b1}...c_{bk} s_{b1..bk}(h)`` terms, so the
    rank ceiling rises to ``M`` and reaches ``2^B = V`` at ``k = B``.
  * **The width cap is the confound.**  If ``g`` is linear ``R^d -> R^M`` then
    the logit matrix is ``Phi G H`` and ``rank <= min(M, d)``, not ``M``.  At
    d=512 orders 3 and 4 are rank-identical and the ladder looks saturated when
    it is not.  Hence ``sch_g_type=mlp`` (a nonlinear ``g`` spans more than
    ``d`` dimensions) and at least one arm at larger ``d``.

Two design decisions here are load-bearing, and both are easy to get wrong in a
way that silently produces order-1 results with extra machinery.

**The interaction coefficients are emitted per position.**  At order 2 the term
is ``sum_{b<b'} c_b c_b' A_{bb'}(h)``, and ``A`` is a function of the hidden
state.  ``g`` therefore maps ``R^d -> R^M`` and the head produces M numbers *per
token position* (120 of them at B=15).  The tempting alternative, a single
learned ``B x B`` parameter shared across all contexts, makes
``sum c_b c_b' A_{bb'}`` a fixed per-token constant: a bias term adding rank 1,
not C(B,2).  That version is far easier to build and gets none of the benefit,
and it would look like order 1 with extra steps.  ``CodeProjection`` is
constructed with ``out_dim = M`` and the constructor asserts it.

**There is no per-bit BCE objective, deliberately.**  Binary-cross-entropy over
bits *is* the independence assumption this work exists to remove, so keeping the
loss while adding interactions would be incoherent.  Every head here computes
``g(h) @ Phi^T`` and hands the result to ordinary cross-entropy over the whole
vocabulary.  That is affordable exactly in the regime that matters: at M=120 and
V=32768 the logit matmul is about 3.9M MACs per token against the softmax's
16.8M, so order 2 is roughly 4x cheaper than the softmax it replaces while being
exact and properly normalised.  Per-bit BCE only ever existed to dodge the O(V)
computation, and below ``M ~ d`` there is nothing to dodge.

It also matters for the redundancy arms specifically.  At exactly ``B = log2 V``
with bijective codes, independent Bernoullis happen to normalise correctly over
the ``2^B`` codewords.  As soon as ``B > log2 V`` (the B in {24, 32, 64} arms)
they do not, because probability mass lands on codewords that correspond to no
token.  Exact cross-entropy over the real vocabulary sidesteps that entirely,
and the failure it avoids is silent rather than loud.

Cost model, stated honestly (see also ``StructuredCodeHead.flops_per_token``):

  * dense softmax: ``V d`` MACs/token, ``V d`` head parameters
  * code head:     ``V M`` MACs/token for the frozen ``Phi`` product plus the
    ``g`` cost, and only ``O(d M)`` head parameters.

So the code head is cheaper in *compute* only when ``M < d``.  It is cheaper in
*parameters* essentially always.  Order 4 at B=15 uses 17x fewer head
parameters and ~4x more compute than the softmax.  That is a trade, not a free
win, and the FLOP accounting below prices it rather than hiding it.

Everything here is built once at ``init_weights`` time (never in ``__init__``,
which runs under ``torch.device("meta")`` in ``scripts/base_train.py``), and
``Phi`` is a *non-persistent* buffer rebuilt deterministically from the
persistent code matrix ``C``, so checkpoints carry ``V x B`` uint8 rather than
``V x M`` floats.

Reference points this file deliberately reduces to a single implementation:

  Oda et al. 2017 (binary code prediction)  -> order 1, ``sch_loss=bce``
  VQ-Logits 2025 (K-vector codebook)        -> ``sch_phi_mode=onehot`` + bias
  learned dense W at width M (the control)  -> ``sch_phi_mode=learned``
  frozen random binary Phi (structure test) -> ``sch_phi_mode=random_binary``
  Huffman hierarchical softmax              -> ``sch_head_type=hsoftmax``
"""

from __future__ import annotations

import heapq
import itertools
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import print0

# Hard ceiling on the expansion width.  M is quadratic-to-exponential in B and a
# typo in a sweep script (order 4 at B=64 is 679,120 columns) would otherwise
# allocate hundreds of gigabytes before failing.
MAX_PHI_WIDTH = 65536

CODE_MODES = ("binary", "random", "ecc", "frequency", "file")
PHI_MODES = ("monomial", "random_binary", "onehot", "learned", "gaussian", "product")
PRODUCT_SOURCES = ("hash", "random", "file")
PRODUCT_IMPLS = ("dense", "gather")
G_TYPES = ("linear", "mlp")
LOGIT_ACTS = ("none", "sigsoftmax", "monotonic")
INPUT_MODES = ("table", "linear", "expanded", "nonlinear", "tied")
HEAD_TYPES = ("code", "hsoftmax", "monarch")
PHI_DTYPES = ("bf16", "fp32")
HOLDOUT_MODES = ("target", "full")

# These tuples are the single source of truth for every ``--sch-*`` choice list.
# scripts/base_train.py imports them rather than restating them: a hand-copied
# list silently rejects a mode that the head supports, which is how
# ``--sch-phi-mode product`` reached a GPU and died in argparse after the model
# had already been configured for it.


# ---------------------------------------------------------------------------
# Code assignment
# ---------------------------------------------------------------------------

def minimal_bits(vocab_size: int) -> int:
    """Smallest B with 2^B >= vocab_size."""
    return max(1, math.ceil(math.log2(max(vocab_size, 2))))


def _binary_expansion(ids: torch.Tensor, bits: int) -> torch.Tensor:
    """(N,) int64 -> (N, bits) uint8, little-endian bit b of each id."""
    shifts = torch.arange(bits, dtype=torch.int64)
    return ((ids.unsqueeze(1) >> shifts) & 1).to(torch.uint8)


def _codes_binary(vocab_size: int, bits: int) -> torch.Tensor:
    """Bijective binary expansion.  Minimum Hamming distance 1 by construction.

    At ``bits == minimal_bits(V)`` this is the degenerate code the plan warns
    about: every single-bit error lands on a different real token, so there is
    no error-correction slack and the ECC-vs-semantic comparison is undefined.
    It is still the cleanest configuration for the Phase 0 rank check.
    """
    assert 2 ** bits >= vocab_size, f"binary codes need 2^{bits} >= {vocab_size}"
    if bits > minimal_bits(vocab_size):
        print0(f"[SCH] warning: code_mode=binary with B={bits} > "
               f"{minimal_bits(vocab_size)} leaves the high bits identically "
               f"zero; redundancy is wasted.  Use code_mode=random or ecc.")
    return _binary_expansion(torch.arange(vocab_size, dtype=torch.int64), bits)


def _codes_random(vocab_size: int, bits: int, seed: int) -> torch.Tensor:
    """Uniformly random *distinct* codes.

    Distinctness is not cosmetic: two tokens sharing a code share a row of Phi
    and are therefore indistinguishable to the head at any order, which puts a
    hard floor under the loss.
    """
    assert 2 ** bits >= vocab_size, f"random codes need 2^{bits} >= {vocab_size}"
    gen = torch.Generator().manual_seed(seed)
    if bits <= 24:
        # Exact sampling without replacement: permute the whole codebook.
        perm = torch.randperm(2 ** bits, generator=gen)[:vocab_size]
        return _binary_expansion(perm, bits)
    # bits > 24: enumerating 2^B is infeasible, but collisions are vanishingly
    # rare, so draw and repair.
    C = torch.randint(0, 2, (vocab_size, bits), generator=gen, dtype=torch.uint8)
    if bits > 63:
        # Beyond int64 packing the birthday probability is < 2^-40 at V = 2^18;
        # a collision check would cost more than it can ever catch.
        return C
    for _ in range(64):
        packed = _pack_rows(C)
        uniq, inverse, counts = torch.unique(packed, return_inverse=True, return_counts=True)
        dup = counts[inverse] > 1
        # Keep the first occurrence of each code, resample the rest.
        seen = set()
        redraw = torch.zeros(vocab_size, dtype=torch.bool)
        for i in torch.nonzero(dup).flatten().tolist():
            key = int(packed[i])
            if key in seen:
                redraw[i] = True
            else:
                seen.add(key)
        n = int(redraw.sum())
        if n == 0:
            break
        C[redraw] = torch.randint(0, 2, (n, bits), generator=gen, dtype=torch.uint8)
    return C


def _pack_rows(C: torch.Tensor) -> torch.Tensor:
    """(V, B<=63) uint8 -> (V,) int64 key, for uniqueness checks."""
    B = C.shape[1]
    assert B <= 63, "uniqueness check only supports B <= 63"
    weights = (2 ** torch.arange(B, dtype=torch.int64))
    return (C.to(torch.int64) * weights).sum(dim=1)


def _codes_ecc(vocab_size: int, bits: int, seed: int) -> torch.Tensor:
    """Systematic random linear code over GF(2): ``c = [u | u P]``.

    ``u`` is the k-bit binary expansion of the token id (k = minimal_bits), and
    ``P`` is a random ``k x (B-k)`` parity matrix.  The systematic prefix makes
    the map injective for free, and random linear codes sit near the
    Gilbert-Varshamov bound, so the minimum distance grows with the number of
    parity bits.  This is the "maximise minimum Hamming distance" arm.
    """
    k = minimal_bits(vocab_size)
    assert bits >= k, f"ecc codes need B >= {k} information bits, got {bits}"
    u = _binary_expansion(torch.arange(vocab_size, dtype=torch.int64), k)
    if bits == k:
        print0("[SCH] warning: code_mode=ecc with no parity bits is identical "
               "to code_mode=binary (minimum distance 1).")
        return u
    gen = torch.Generator().manual_seed(seed)
    P = torch.randint(0, 2, (k, bits - k), generator=gen, dtype=torch.uint8)
    parity = (u.to(torch.int32) @ P.to(torch.int32)) % 2
    return torch.cat([u, parity.to(torch.uint8)], dim=1)


def _codes_frequency(vocab_size: int, bits: int, freqs: torch.Tensor) -> torch.Tensor:
    """Rank-order code: token of frequency rank r gets the binary expansion of r.

    Frequent tokens land on small integers, hence on codes with few bits set,
    hence on few active monomials.  This is the frequency/Huffman-derived arm of
    the code-assignment axis.  (The Huffman *tree* baseline is a different
    object entirely, see ``HierarchicalSoftmaxHead``.)
    """
    assert freqs is not None and freqs.numel() >= vocab_size, \
        "code_mode=frequency needs a token frequency table (tokenizer/freq_table.pt)"
    order = torch.argsort(freqs[:vocab_size].float(), descending=True)
    ranks = torch.empty(vocab_size, dtype=torch.int64)
    ranks[order] = torch.arange(vocab_size, dtype=torch.int64)
    return _binary_expansion(ranks, bits)


def _append_parity(C: torch.Tensor, n_parity: int, seed: int) -> torch.Tensor:
    """Append ``n_parity`` random GF(2) parity bits to an existing code.

    This is the knob that interpolates the two objectives that Contribution 4
    claims are opposed: an ECC wants large minimum Hamming distance, while
    generalisation wants semantically similar tokens *close* in Hamming space.
    Parity bits raise the former without disturbing the base assignment, so
    sweeping ``n_parity`` sweeps the tension on one axis.
    """
    if n_parity <= 0:
        return C
    gen = torch.Generator().manual_seed(seed + 7919)
    P = torch.randint(0, 2, (C.shape[1], n_parity), generator=gen, dtype=torch.uint8)
    parity = (C.to(torch.int32) @ P.to(torch.int32)) % 2
    return torch.cat([C, parity.to(torch.uint8)], dim=1)


def load_freq_table(vocab_size: int, tokenizer_dir: str | None) -> torch.Tensor | None:
    """Load ``<tokenizer_dir>/freq_table.pt`` if present (shared with EET)."""
    if tokenizer_dir is None:
        try:
            from nanochat.common import get_base_dir
            tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")
        except Exception:
            return None
    path = os.path.join(tokenizer_dir, "freq_table.pt")
    if not os.path.exists(path):
        return None
    try:
        ft = torch.load(path, weights_only=True, map_location="cpu").float()
    except Exception as e:  # pragma: no cover - corrupt cache
        print0(f"[SCH] could not read {path}: {e}")
        return None
    if ft.numel() < vocab_size:
        ft = torch.cat([ft, torch.zeros(vocab_size - ft.numel())])
    return ft


def build_codes(vocab_size: int, bits: int, mode: str = "binary", seed: int = 1234,
                freqs: torch.Tensor | None = None, path: str = "",
                ecc_bits: int = 0) -> torch.Tensor:
    """Return the (vocab_size, B_total) uint8 code matrix.  Always on CPU.

    ``ecc_bits`` appends parity bits *after* the base assignment, so the final
    width is ``bits + ecc_bits``.
    """
    assert mode in CODE_MODES, f"unknown code mode {mode!r}, expected one of {CODE_MODES}"
    if mode == "file":
        assert path, "code_mode=file requires sch_code_path"
        C = torch.load(path, weights_only=True, map_location="cpu")
        assert C.dim() == 2 and C.shape[0] >= vocab_size, \
            f"{path}: expected a (>= {vocab_size}, B) code matrix, got {tuple(C.shape)}"
        C = C[:vocab_size].to(torch.uint8)
        if bits > 0 and C.shape[1] != bits:
            print0(f"[SCH] {path} supplies B={C.shape[1]}; overriding sch_bits={bits}")
    elif mode == "binary":
        C = _codes_binary(vocab_size, bits)
    elif mode == "random":
        C = _codes_random(vocab_size, bits, seed)
    elif mode == "ecc":
        C = _codes_ecc(vocab_size, bits, seed)
    elif mode == "frequency":
        C = _codes_frequency(vocab_size, bits, freqs)
    C = _append_parity(C, ecc_bits, seed)
    if C.shape[1] <= 63:
        n_unique = int(torch.unique(_pack_rows(C)).numel())
        assert n_unique == vocab_size, (
            f"code assignment is not injective: {vocab_size - n_unique} tokens share a "
            f"code with another token, which puts a hard floor under the loss")
    return C.contiguous()


def code_statistics(C: torch.Tensor, max_pairs: int = 200_000, seed: int = 0) -> dict:
    """Descriptive statistics for a code matrix, for logging and the paper.

    Minimum Hamming distance over all pairs is O(V^2 B); we sample pairs instead
    and report the sampled minimum, which is an upper bound on the true minimum.
    """
    V, B = C.shape
    Cf = C.to(torch.int16)
    gen = torch.Generator().manual_seed(seed)
    n = min(max_pairs, V * (V - 1) // 2)
    i = torch.randint(0, V, (n,), generator=gen)
    j = torch.randint(0, V, (n,), generator=gen)
    keep = i != j
    i, j = i[keep], j[keep]
    d = (Cf[i] ^ Cf[j]).sum(dim=1)
    return {
        "bits": B,
        "density": float(C.float().mean()),
        "mean_hamming_sampled": float(d.float().mean()),
        "min_hamming_sampled": int(d.min()),
    }


# ---------------------------------------------------------------------------
# Monomial expansion
# ---------------------------------------------------------------------------

def full_phi_width(bits: int, order: int) -> int:
    """M = sum_{j<=k} C(B, j), the uncapped expansion width."""
    return sum(math.comb(bits, j) for j in range(1, order + 1))


def enumerate_monomials(bits: int, order: int, max_m: int = 0, seed: int = 1234):
    """Return ``[(j, LongTensor(n_j, j)), ...]``, the monomial index sets.

    Orders are filled greedily from the bottom: every order below the truncation
    point is kept whole, and the first order that would overflow ``max_m`` is
    subsampled uniformly (seeded).  Truncating within an order rather than
    dropping the order keeps the width knob continuous, which is what the
    saturation sweep needs.
    """
    assert 1 <= order <= bits, f"order must be in [1, {bits}], got {order}"
    budget = max_m if max_m > 0 else MAX_PHI_WIDTH
    assert budget <= MAX_PHI_WIDTH, f"max_m {budget} exceeds MAX_PHI_WIDTH {MAX_PHI_WIDTH}"
    gen = torch.Generator().manual_seed(seed)
    groups, used = [], 0
    for j in range(1, order + 1):
        n_full = math.comb(bits, j)
        room = budget - used
        if room <= 0:
            break
        if n_full <= room:
            idx = torch.tensor(list(itertools.combinations(range(bits), j)), dtype=torch.int64)
            groups.append((j, idx.view(n_full, j)))
            used += n_full
        else:
            idx = _sample_combinations(bits, j, room, gen)
            groups.append((j, idx))
            used += room
            break
    return groups


def _sample_combinations(bits: int, j: int, n: int, gen: torch.Generator) -> torch.Tensor:
    """``n`` distinct size-``j`` subsets of ``range(bits)``, sampled uniformly."""
    n_full = math.comb(bits, j)
    if n_full <= 4_000_000:
        all_idx = torch.tensor(list(itertools.combinations(range(bits), j)), dtype=torch.int64)
        pick = torch.randperm(n_full, generator=gen)[:n]
        return all_idx.view(n_full, j)[pick].contiguous()
    # Rejection sampling for the combinatorially huge cases.
    seen, rows = set(), []
    while len(rows) < n:
        cand = torch.argsort(torch.rand(bits, generator=gen))[:j]
        key = tuple(sorted(cand.tolist()))
        if key in seen:
            continue
        seen.add(key)
        rows.append(torch.tensor(key, dtype=torch.int64))
    return torch.stack(rows)


def build_phi_monomial(C: torch.Tensor, groups, dtype=torch.float32,
                       device="cpu") -> torch.Tensor:
    """Materialise ``Phi in {0,1}^{V x M}`` from the code matrix and index sets.

    A monomial over binary variables is an AND, so ``phi_S(c) = min_{b in S} c_b``.
    The gather is chunked over columns because the intermediate is
    ``V x chunk x order`` bytes and V is up to 131k.
    """
    V = C.shape[0]
    M = sum(int(idx.shape[0]) for _, idx in groups)
    phi = torch.empty(V, M, dtype=dtype, device=device)
    Cd = C.to(device)
    col = 0
    for j, idx in groups:
        idx = idx.to(device)
        n = idx.shape[0]
        chunk = max(16, int(2 ** 26 // max(V * j, 1)))
        for s in range(0, n, chunk):
            sub = idx[s:s + chunk]                       # (c, j)
            c = sub.shape[0]
            gathered = Cd[:, sub.reshape(-1)].view(V, c, j)
            phi[:, col:col + c] = gathered.min(dim=-1).values.to(dtype)
            col += c
    assert col == M
    return phi


def build_phi_random_binary(V: int, M: int, density: float, seed: int,
                            dtype=torch.float32, device="cpu") -> torch.Tensor:
    """Frozen random binary Phi at a matched density.

    This is the control that separates *structure* from *binariness*: it has the
    same shape, the same dtype and (by construction) the same expected number of
    ones per row as the monomial expansion, but no interaction structure at all.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    phi = torch.empty(V, M, dtype=dtype)
    step = max(1, 2 ** 22 // max(V, 1))     # bound the fp32 staging chunk
    for s in range(0, M, step):
        w = min(step, M - s)
        phi[:, s:s + w] = (torch.rand(V, w, generator=gen) < density).to(dtype)
    return phi.to(device)


def build_product_codes(vocab_size: int, groups: int, codebook: int,
                        source: str = "hash", seed: int = 1234,
                        path: str = "") -> torch.Tensor:
    """Assign every token a K-ary codeword of length ``groups``.

    A binary digit contributes exactly one column to ``Phi``, so ``B`` digits buy
    ``M = B`` basis functions at order 1 and the only way to widen is interaction
    order, which multiplies columns *inside* the partition lattice the same ``B``
    digits already generate.  That is the ladder plateau measured in c00: order 3
    to order 4 costs 2.7x the FLOPs and buys nothing.

    A K-ary digit contributes ``K`` columns, one per symbol, so ``g`` digits give
    ``M = g*K`` at order 1 with no interactions at all.  ``phi_mode='onehot'`` is
    the ``g=1`` corner of this and LightRNN (Li et al., 2016) is the ``g=2``
    corner; the general case is what this builds.

    Sources:
      ``hash``    deterministic per-group hash of the token id.  Cheap control,
                  and the K-ary analogue of ``code_mode='binary'``.
      ``random``  independent uniform symbols.  The null control.
      ``file``    load an assignment fitted to an embedding, from
                  ``scripts/code_assign.py --mode product``.  This is the arm
                  that is supposed to work: the partitions come from k-means on
                  a token embedding, so the cells follow the geometry instead of
                  the token ids.
    """
    assert source in PRODUCT_SOURCES, f"product source {source!r} not in {PRODUCT_SOURCES}"
    assert groups >= 1 and codebook >= 2
    if source == "file":
        assert path, "sch_phi_mode=product with source=file needs --sch-code-path"
        A = torch.load(path, map_location="cpu")
        if isinstance(A, dict):
            A = A["assign"]
        A = A.long()
        assert A.shape == (vocab_size, groups), \
            f"assignment file has shape {tuple(A.shape)}, expected {(vocab_size, groups)}"
        assert int(A.max()) < codebook, \
            f"assignment file uses {int(A.max()) + 1} symbols, head built for {codebook}"
        return A.to(torch.int32)
    if source == "random":
        gen = torch.Generator().manual_seed(seed + 5003)
        return torch.randint(0, codebook, (vocab_size, groups),
                             generator=gen, dtype=torch.int32)
    # hash: mix the token id per group with a cheap odd multiplier, then reduce.
    ids = torch.arange(vocab_size, dtype=torch.int64)
    mult = torch.tensor([2654435761 + 2 * j for j in range(groups)], dtype=torch.int64)
    A = ((ids.unsqueeze(1) * mult.unsqueeze(0)) >> 11) % codebook
    return A.to(torch.int32)


def product_gather(z, assign):
    """``g(h) @ Phi^T`` when ``Phi`` is one-hot within each group.

    ``Phi`` is never materialised.  Being one-hot per group makes the product a
    gather and add: each vocabulary entry sums ``g`` looked-up scalars, so the
    ARITHMETIC is ``V*g`` additions rather than ``V*M`` multiply-accumulates.
    Open question Q1 asked whether a fast transform exists for the truncated
    monomial expansion; for one-hot codes it exists and this is it.

    WALL CLOCK DOES NOT FOLLOW, AND THIS IMPLEMENTATION IS THE SLOW ONE.
    Two separate things stand between the FLOP count and a speedup:

    1. Both this and a dense head write the same ``N x V`` logit tensor, 17.2 GB
       per forward at V=131072 with 65536 tokens.  The dense head is compute
       bound at 768 FLOP/byte, so removing its arithmetic lands on that write
       and not on zero.  H100 roofline: dense 13.3 ms against a *fused* gather
       at 5.1 ms.  The FLOP ratio is 96x; the achievable ratio is 2.6x.
    2. The loop below is not fused.  Each ``index_select`` materialises a full
       ``(..., V)`` tensor and each add allocates another, so this makes about
       ``2g`` passes over the output instead of one: 275 GB instead of 17, which
       roofline puts at 82 ms, roughly 6x SLOWER than the dense matmul it
       replaces.  ``flops_per_token`` will still report the 96x reduction.

    So this is correct and cheap in arithmetic, and it is a regression in time
    until a single-pass kernel accumulates the ``g`` lookups in registers before
    one write.  ``torch.compile`` may fuse the chain (the sweeps pass
    ``--compile``); Triton is the fallback.  Benchmark before quoting a ratio.

      z       (..., groups * codebook), the output of ``g``
      assign  (V, groups) int64, the codeword of each token

    MEASURED SLOWER THAN THE MATMUL IT REPLACES.  Do not use this path for a
    training sweep; ``product_impl='dense'`` is the default for that reason.
    On CPU at N=2048, V=32768, g=8, K=64 this is 4.76x the time of the dense
    ``z @ Phi^T``, which performs 64x more arithmetic, and on a GPU it is worse
    still: the backward of ``index_select`` is an ``index_add`` scattering V
    values into K slots per position, so at V=32768 and K=64 every slot takes
    512-way atomic contention.  The forward is also g separate passes over a
    full ``(..., V)`` tensor rather than one.

    Kept because the arithmetic claim is real and only the kernel is missing:
    see OPEN_QUESTIONS Q8.  Benchmark any replacement against the dense path.
    """
    groups = assign.shape[1]
    K = z.shape[-1] // groups
    zg = z.view(*z.shape[:-1], groups, K)
    out = None
    for j in range(groups):
        part = zg[..., j, :].index_select(-1, assign[:, j])
        # in place: saves the g extra allocations the adds would make.
        # Halves the traffic; it does not make this single-pass.
        out = part if out is None else out.add_(part)
    return out


def build_phi_onehot(V: int, M: int, seed: int, dtype=torch.float32,
                     device="cpu") -> torch.Tensor:
    """VQ-Logits-style scatter: each token points at one of M codebook vectors.

    VQ-Logits reduces to a code head whose Phi has exactly one bit set per row,
    which makes it the ``k=1``, ``B=M``, one-hot corner of this design space.
    Its per-token bias is what recovers most of the lost quality, hence the
    pairing with ``sch_bias=1`` in the baseline table.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    assign = torch.randint(0, M, (V,), generator=gen)
    phi = torch.zeros(V, M, dtype=dtype)
    phi[torch.arange(V), assign] = 1.0
    return phi.to(device)


# ---------------------------------------------------------------------------
# g: the learned projection R^d -> R^M
# ---------------------------------------------------------------------------

class CodeProjection(nn.Module):
    """``g(h)``: linear, or an MLP when the width cap has to be escaped.

    A *linear* ``g`` caps the logit-matrix rank at ``min(M, d)``, so orders 3 and
    4 at d=512 are rank-identical and the ladder appears to saturate for a
    reason that has nothing to do with the code.  A nonlinear ``g`` has an image
    that is not contained in any d-dimensional subspace, restoring the ceiling to
    ``M`` and giving the head a rank that is decoupled from the model width
    (something a dense softmax can never have: it is capped at ``d + 1`` at any
    parameter count).
    """

    def __init__(self, n_embd: int, out_dim: int, g_type: str = "linear",
                 hidden: int = 0, layers: int = 2):
        super().__init__()
        assert g_type in G_TYPES, f"unknown g type {g_type!r}"
        self.g_type = g_type
        self.out_dim = out_dim
        self.n_embd = n_embd
        if g_type == "linear":
            self.net = nn.ModuleList([nn.Linear(n_embd, out_dim, bias=False)])
            self.hidden = 0
        else:
            hidden = hidden if hidden > 0 else n_embd
            self.hidden = hidden
            assert layers >= 2, "mlp g needs at least 2 layers"
            mods = [nn.Linear(n_embd, hidden, bias=False)]
            for _ in range(layers - 2):
                mods.append(nn.Linear(hidden, hidden, bias=False))
            mods.append(nn.Linear(hidden, out_dim, bias=False))
            self.net = nn.ModuleList(mods)

    def forward(self, x):
        if self.g_type == "linear":
            w = self.net[0].weight
            return F.linear(x, w.to(dtype=x.dtype))
        for i, lin in enumerate(self.net):
            x = F.linear(x, lin.weight.to(dtype=x.dtype))
            if i < len(self.net) - 1:
                x = F.relu(x).square()   # relu^2, as the backbone MLP uses
        return x

    def init_weights(self, out_std: float = 0.001):
        """Small final layer so initial logits are near-uniform, as lm_head is."""
        s = 3 ** 0.5 * self.n_embd ** -0.5
        for lin in self.net[:-1]:
            torch.nn.init.uniform_(lin.weight, -s, s)
        torch.nn.init.normal_(self.net[-1].weight, mean=0.0, std=out_std)

    def flops_per_token(self) -> int:
        return 6 * sum(l.weight.numel() for l in self.net)


# ---------------------------------------------------------------------------
# The head
# ---------------------------------------------------------------------------

def _whiten_phi(phi):
    """``Phi (Phi^T Phi)^{-1/2}``: same column space, orthonormal columns.

    For a *linear* ``g`` this is provably a reparameterisation.  If
    ``Phi_w = Phi R`` for invertible ``R`` then ``g(h) Phi_w^T = g(h) R^T Phi^T``
    and ``g(h) R^T = h (R W_g)^T``, so the function class is identical and only
    the optimisation geometry changes.  That is exactly what makes it a clean
    experiment: any movement in final bpb is an optimisation effect and nothing
    else, which separates the conditioning hypothesis from the alignment one in
    a single run.  Measured ``cond(Phi^T Phi)`` after row normalisation is 396 at
    order 2 and 3225 at order 3; whitening sets it to 1.
    """
    single = phi.dim() == 2
    P = phi.unsqueeze(0) if single else phi
    out = []
    for k in range(P.shape[0]):
        A = P[k].double()
        G = A.T @ A
        ev, U = torch.linalg.eigh(G)
        inv_sqrt = U @ torch.diag(ev.clamp_min(1e-10).rsqrt()) @ U.T
        out.append((A @ inv_sqrt).to(phi.dtype))
    W = torch.stack(out, 0)
    return W[0] if single else W


class StructuredCodeHead(nn.Module):
    """Drop-in replacement for ``lm_head`` computing ``g(h) Phi^T``.

    Shapes match ``Linear(n_embd, padded_vocab_size)`` exactly so the caller does
    not change: input ``(..., d)``, output ``(..., padded_vocab_size)``.

    Attributes the caller reads:
      ``self_normalized``  the head already returns log-probabilities, so the
                           caller must not apply logit softcapping (softcap is a
                           monotone squash of *logits*, and re-squashing a
                           normalised log-prob vector silently changes the
                           distribution).
      ``custom_loss``      False here; True only for hierarchical softmax.
    """

    custom_loss = False

    def __init__(self, config, padded_vocab_size: int, n_embd: int):
        super().__init__()
        cfg = resolve_sch_config(config, padded_vocab_size)
        self.cfg = cfg
        self.vocab_size = padded_vocab_size
        self.n_embd = n_embd
        self.bits = cfg["bits_total"]
        self.order = cfg["order"]
        self.width = cfg["width"]          # M
        self.phi_mode = cfg["phi_mode"]
        self.product_impl = cfg["product_impl"]
        self.n_mixture = cfg["mixture"]
        self.logit_act = cfg["logit_act"]

        M = self.width
        # g, one per mixture component. Its output width is M, never B: the
        # interaction coefficients A_{bb'}(h) are functions of the hidden state
        # and must be emitted PER POSITION. A shared B x B parameter instead
        # would make the order-2 term a fixed per-token constant, contributing
        # rank 1 rather than C(B,2), and the whole head would behave like order 1.
        self.g = nn.ModuleList([
            CodeProjection(n_embd, M, cfg["g_type"], cfg["g_hidden"], cfg["g_layers"])
            for _ in range(self.n_mixture)
        ])
        assert all(g.out_dim == M for g in self.g), \
            "g must emit one coefficient per monomial per position"
        # Mixture router over components.  The log-sum-exp of component softmaxes
        # is not a linear function of any fixed feature map, so the rank bound
        # does not apply to it at all -- this is the one mitigation that escapes
        # the ceiling rather than raising it.
        self.router = nn.Linear(n_embd, self.n_mixture, bias=False) if self.n_mixture > 1 else None
        # Sparse routing over components.  0 or >= n_mixture means the dense
        # mixture (every component computed, cost K * 4VM).  With top-k the cost
        # is k/K of that, which is what makes a *union* of subspaces affordable:
        # reach up to K*M dimensions at the per-token price of k.
        tk = cfg["mixture_topk"]
        self.mixture_topk = self.n_mixture if tk <= 0 else min(tk, self.n_mixture)
        # Per-component Phi.  Without this every component draws from the SAME
        # M-dimensional subspace and the mixture can only reweight it; the
        # log-sum-exp still escapes the rank bound, but the reach does not grow.
        self.per_phi = bool(cfg["mixture_per_phi"]) and self.n_mixture > 1
        if self.per_phi:
            # Guard against a silent no-op.  Distinct components need distinct
            # column spaces or the "union of subspaces" is one subspace and the
            # arm measures nothing while costing full price.  For the *full*
            # monomial expansion the span is fixed by the code, so reseeding
            # returns the same set of monomials; only a truncated expansion, or
            # a mode whose Phi is drawn at random, actually differs.
            assert self.phi_mode not in ("learned", "product"), (
                f"sch_mixture_per_phi is not supported with phi_mode={self.phi_mode!r}: "
                f"a learned Phi already places its own subspace, and a product code "
                f"stores one assignment")
            if self.phi_mode == "monomial":
                assert 0 < cfg["max_m"] < full_phi_width(self.bits, self.order), (
                    f"sch_mixture_per_phi with the full order-{self.order} monomial "
                    f"expansion is a no-op: every component would get the same "
                    f"M={self.width} basis. Set --sch-max-m below "
                    f"{full_phi_width(self.bits, self.order)} so the components draw "
                    f"different monomial subsets, or use a random Phi mode.")
        self.n_phi = self.n_mixture if self.per_phi else 1

        # Phi: frozen. Non-persistent, rebuilt from `codes` (see init_weights).
        phi_dtype = torch.bfloat16 if cfg["phi_dtype"] == "bf16" else torch.float32
        self.phi_dtype = phi_dtype
        if self.phi_mode == "learned":
            # The matched-capacity control reviewers ask for: identical width,
            # identical g, but a *learned* real-valued output embedding.
            self.phi_learned = nn.Parameter(torch.empty(padded_vocab_size, M))
            self.register_buffer("phi", torch.empty(0), persistent=False)
        elif self.phi_mode == "product":
            # Only the assignment is checkpointed; Phi is derived from it.
            # int64 because index_select requires it and converting V values on
            # every forward is pure waste.
            self.phi_learned = None
            self.register_buffer("assign",
                                 torch.empty(padded_vocab_size, cfg["product_groups"],
                                             dtype=torch.int64), persistent=True)
            # The gather path does not materialise Phi; the dense path does,
            # because the one-hot matmul on tensor cores beats the gather by a
            # wide margin until a fused kernel exists (see product_gather).
            # V x M in bf16 is 33 MB at V=32768 M=512 and 537 MB at V=131072
            # M=2048, which is affordable; the gather's g full-width
            # intermediates are not.
            self.register_buffer("phi",
                                 torch.empty(0) if self.product_impl == "gather"
                                 else torch.empty(padded_vocab_size, M, dtype=phi_dtype),
                                 persistent=False)
        else:
            self.phi_learned = None
            self.register_buffer("phi",
                                 torch.empty(self.n_phi, padded_vocab_size, M, dtype=phi_dtype)
                                 if self.per_phi else
                                 torch.empty(padded_vocab_size, M, dtype=phi_dtype),
                                 persistent=False)
        # The code matrix is the thing worth checkpointing: V x B uint8.
        if self.phi_mode != "product":
            self.register_buffer("codes",
                                 torch.empty(padded_vocab_size, self.bits, dtype=torch.uint8),
                                 persistent=True)
        else:
            self.register_buffer("codes", torch.empty(0, dtype=torch.uint8), persistent=False)

        # Per-token bias.  Costs V parameters and adds one rank, and it makes
        # zero-shot vocabulary extension impossible (a token added after
        # training has no bias), so it is off by default and reported as such.
        self.bias = nn.Parameter(torch.empty(padded_vocab_size)) if cfg["bias"] else None

        # Dense residual hybrid: buys r rank for rV parameters.  Expected to win
        # on perplexity and expected to be the least interesting result, because
        # it partially reintroduces the softmax it is meant to replace.
        r = cfg["residual_rank"]
        self.residual_rank = r
        if r > 0:
            self.res_down = nn.Linear(n_embd, r, bias=False)
            self.res_emb = nn.Parameter(torch.empty(padded_vocab_size, r))
        else:
            self.res_down, self.res_emb = None, None

        # Learnable pointwise nonlinearity on code logits (mitigation 2).
        if self.logit_act == "monotonic":
            self.act_alpha = nn.Parameter(torch.empty(1))
            self.act_beta = nn.Parameter(torch.empty(1))
        else:
            self.act_alpha = self.act_beta = None

        # The caller slices logits to config.vocab_size *before* the softmax, so
        # padding rows never compete. A self-normalised head normalises inside
        # forward(), so it has to know where the padding starts or the padded
        # rows would silently steal probability mass that the caller then throws
        # away, leaving the returned vector sub-normalised.
        self.active_vocab = int(getattr(config, "vocab_size", padded_vocab_size))
        self._phi_scale = 1.0
        self._materialized = False
        self._aux_loss = None
        # Name differs across torch versions (the private form predates 2.2).
        hook_api = getattr(self, "register_load_state_dict_post_hook", None) or \
            getattr(self, "_register_load_state_dict_post_hook")
        hook_api(self._post_load)

    # -- properties the caller branches on ---------------------------------
    @property
    def self_normalized(self) -> bool:
        """True when forward() already returns normalised log-probabilities.

        A *hard* top-1 mixture is not one of those. Each token uses exactly one
        component, so ``log P = log_softmax(that component's logits)``, which is
        precisely the normalisation ``F.cross_entropy`` is about to perform.
        Doing it inside the head as well costs a second full ``(N, V)`` fp32
        tensor saved for backward, 34 GB at 262144 tokens and V=32768, for a
        mathematically identical result. Returning raw logits instead makes the
        k=1 path hold exactly what the dense baseline holds.
        """
        return (self.n_mixture > 1 and self.mixture_topk > 1) \
            or self.logit_act == "sigsoftmax"

    # -- construction -------------------------------------------------------
    def init_weights(self):
        """Materialise Phi and initialise the learned parts.

        Called from ``GPT.init_weights`` *after* the NaN-poison pass, and never
        from ``__init__``: the model is built under ``torch.device("meta")`` and
        then ``to_empty()``-ed, so any tensor computed in the constructor would
        be a meta tensor and any buffer would hold garbage.
        """
        cfg = self.cfg
        if self.phi_mode == "product":
            A = build_product_codes(self.vocab_size, cfg["product_groups"],
                                    cfg["product_codebook"], cfg["product_source"],
                                    cfg["code_seed"], cfg["code_path"])
            self.assign.copy_(A.to(dtype=torch.int64, device=self.assign.device))
            self._build_phi()
            self._init_learned_parts()
            return
        if self.phi_mode == "learned":
            # Nothing here reads `codes`: the output is bit-identical under any
            # code mode or seed. Leaving it unbuilt keeps the checkpoint honest
            # about what the arm actually is.
            self._materialized = True
            self._init_learned_parts()
            return
        device = self.codes.device
        freqs = load_freq_table(self.vocab_size, cfg["tokenizer_dir"]) \
            if cfg["code_mode"] == "frequency" else None
        C = build_codes(self.vocab_size, cfg["bits"], cfg["code_mode"], cfg["code_seed"],
                        freqs=freqs, path=cfg["code_path"], ecc_bits=cfg["ecc_bits"])
        assert C.shape[1] == self.bits, (
            f"code width {C.shape[1]} does not match the width {self.bits} the head was "
            f"built for; sch_code_path files must agree with sch_bits + sch_code_ecc_bits")
        self.codes.copy_(C.to(device))
        self._build_phi()
        self._init_learned_parts()

    def _init_learned_parts(self):
        cfg = self.cfg
        for g in self.g:
            g.init_weights(out_std=cfg["g_out_std"])
        if self.router is not None:
            if self.mixture_topk >= self.n_mixture:
                torch.nn.init.zeros_(self.router.weight)   # uniform mixture at init
            else:
                # Hard routing cannot start from a symmetric router. With zero
                # weights every token's router logits are identical and topk
                # breaks the tie by index, so component 0 receives ALL tokens
                # and components 1..K-1 receive none, forever: they get no
                # gradient, so the router never learns to prefer them. That
                # shows up first as DDP refusing to step ("parameters that were
                # not used in producing loss") and would otherwise show up as a
                # mixture that silently behaves like a single component.
                torch.nn.init.normal_(self.router.weight, mean=0.0,
                                      std=self.n_embd ** -0.5)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
        if self.res_emb is not None:
            torch.nn.init.normal_(self.res_emb, mean=0.0, std=0.001)
            torch.nn.init.uniform_(self.res_down.weight,
                                   -(3 ** 0.5) * self.n_embd ** -0.5,
                                   (3 ** 0.5) * self.n_embd ** -0.5)
        if self.act_alpha is not None:
            torch.nn.init.zeros_(self.act_alpha)       # identity at init
            torch.nn.init.ones_(self.act_beta)
        if self.phi_learned is not None:
            torch.nn.init.normal_(self.phi_learned, mean=0.0, std=0.02)

    def _build_phi(self):
        """(Re)build the frozen Phi from ``self.codes``."""
        if self.phi_mode == "learned":
            self._materialized = True
            return
        if self.phi_mode == "product":
            # Derived from `assign`, which IS persisted, so a resumed run must
            # rebuild Phi from the restored assignment rather than from the
            # config default.  The gather path has no Phi to rebuild.
            if self.product_impl == "dense":
                phi = torch.zeros(self.vocab_size, self.width, dtype=self.phi_dtype)
                rows = torch.arange(self.vocab_size)
                A = self.assign.detach().cpu().long()
                for j in range(self.cfg["product_groups"]):
                    phi[rows, j * self.cfg["product_codebook"] + A[:, j]] = 1
                self.phi = phi.to(device=self.assign.device, dtype=self.phi_dtype)
            self._materialized = True
            return
        device = self.codes.device
        V, M = self.vocab_size, self.width
        # Build straight into the storage dtype. At V=131072 and M=3213 an fp32
        # staging copy is 1.7 GB of host RAM for a matrix of zeros and ones, and
        # the AND products are exact in bf16 anyway. Centring is the exception:
        # it produces genuine fractions, so it stages in fp32.
        build_dtype = torch.float32 if self.cfg["phi_center"] else self.phi_dtype
        mats = [self._build_one_phi(V, M, build_dtype, k) for k in range(self.n_phi)]
        phi = torch.stack(mats, 0) if self.per_phi else mats[0]

        if self.cfg["phi_center"]:
            phi = phi - phi.mean(dim=-2, keepdim=True)
        if self.cfg["phi_whiten"]:
            phi = _whiten_phi(phi)
        if self.cfg["phi_normalize"]:
            # Pure reparameterisation (absorbed into g), and rank-preserving, but
            # without it the row norms grow like sqrt(M) and the initial logits
            # blow up at order 3 and above. The row norms are accumulated in
            # fp32 over column chunks so a bf16 Phi does not bias the scale.
            acc = torch.zeros(phi.shape[:-1], dtype=torch.float32)
            step = max(1, 2 ** 22 // max(V, 1))
            for s in range(0, M, step):
                acc += phi[..., s:s + step].float().pow(2).sum(dim=-1)
            rms = acc.mean().clamp_min(1e-8).sqrt()
            self._phi_scale = float(1.0 / rms)
            phi.mul_(self._phi_scale)   # in place: a full-precision copy is 1.7 GB at V=131k
        self.phi = phi.to(device=device, dtype=self.phi_dtype)
        self._materialized = True

    def _build_one_phi(self, V, M, build_dtype, k: int):
        """Build component ``k``'s Phi.  Distinct seeds give distinct subspaces."""
        off = 1013 * k
        if self.phi_mode == "monomial":
            # Different monomial subsets per component, so the components span
            # genuinely different corners of the lattice rather than the same one.
            groups = enumerate_monomials(self.bits, self.order, self.cfg["max_m"],
                                         self.cfg["mono_seed"] + off)
            return build_phi_monomial(self.codes.cpu(), groups, dtype=build_dtype)
        if self.phi_mode == "random_binary":
            return build_phi_random_binary(V, M, self.cfg["phi_density"],
                                           self.cfg["code_seed"] + 31 + off, dtype=build_dtype)
        if self.phi_mode == "onehot":
            return build_phi_onehot(V, M, self.cfg["code_seed"] + 61 + off, dtype=build_dtype)
        if self.phi_mode == "gaussian":
            gen = torch.Generator().manual_seed(self.cfg["code_seed"] + 97 + off)
            return (torch.randn(V, M, generator=gen) / math.sqrt(M)).to(build_dtype)
        raise ValueError(self.phi_mode)  # pragma: no cover - guarded in resolve_sch_config

    def _post_load(self, module, incompatible_keys):
        """Rebuild Phi after ``load_state_dict`` restores a (possibly different)
        code assignment.  Phi is not persisted, so without this a resumed run
        would keep the Phi built from the *config default* codes."""
        try:
            self._build_phi()
        except Exception as e:  # pragma: no cover
            print0(f"[SCH] warning: could not rebuild Phi after load_state_dict: {e}")

    # -- forward ------------------------------------------------------------
    def _phi_matmul(self, z, m: int = 0):
        if self.phi_mode == "product":
            if self.product_impl == "gather":
                # g gathers and g-1 adds: V*g arithmetic instead of V*M, and
                # measurably slower than the matmul until it is fused. Q8.
                return product_gather(z, self.assign)
            return F.linear(z.to(dtype=self.phi.dtype), self.phi)
        if self.phi_learned is not None:
            # Cast the weight, exactly as nanochat's Linear does for lm_head, so
            # the learned-W control and the dense baseline pay the same cast.
            return F.linear(z, self.phi_learned.to(dtype=z.dtype))
        # Cast the *activation* instead: Phi is frozen and up to V x M, so
        # casting it every forward would copy hundreds of MB per step.
        phi = self.phi[m] if self.per_phi else self.phi
        return F.linear(z.to(dtype=phi.dtype), phi)

    def _component_logits(self, x, m: int):
        logits = self._phi_matmul(self.g[m](x), m)
        if self.res_emb is not None:
            r = F.linear(x, self.res_down.weight.to(dtype=x.dtype))
            logits = logits + F.linear(r, self.res_emb.to(dtype=logits.dtype)).to(logits.dtype)
        if self.bias is not None:
            logits = logits + self.bias.to(dtype=x.dtype)
        if self.logit_act == "monotonic":
            # z + alpha * silu(beta z): monotone for alpha >= 0, identity at init.
            logits = logits + self.act_alpha.to(logits.dtype) * F.silu(
                self.act_beta.to(logits.dtype) * logits)
        return logits

    def _mask_padding(self, logits):
        """Mask the padded vocabulary rows before a self-normalising softmax.

        Masks in place when the caller owns the tensor. The clone is a full
        ``(..., V)`` copy, 34 GB in fp32 at 262144 tokens and V=32768, and it is
        only needed when someone else may still be holding the input.
        """
        if self.active_vocab < self.vocab_size:
            logits[..., self.active_vocab:] = float("-inf")
        return logits

    def forward(self, x):
        if self.n_mixture == 1:
            logits = self._component_logits(x, 0)
            if self.logit_act == "sigsoftmax":
                logits = self._mask_padding(logits)
                # sigsoftmax(z)_w ∝ exp(z_w) sigma(z_w).  Elementwise and
                # nonlinear, so the resulting log-prob matrix is not in the span
                # of Phi and the rank bound stops applying.
                return torch.log_softmax(logits.float() + F.logsigmoid(logits.float()), dim=-1)
            return logits
        # Mixture of code heads.  log P = log sum_m pi_m(h) P_m(w|h); the
        # log-sum-exp is nonlinear in the component logits, so the rank of the
        # resulting log-prob matrix is not bounded by M.
        route = F.linear(x, self.router.weight.to(dtype=x.dtype)).float()
        self._aux_loss = self._load_balance(route) if self.mixture_topk < self.n_mixture else None
        if self.mixture_topk >= self.n_mixture:
            # Accumulate with logaddexp instead of stacking. A (..., V) tensor is
            # 34 GB in fp32 at 262144 tokens and V=32768, so `torch.stack` over K
            # components asks for K times that in one allocation, and the dense
            # baseline only ever holds one.
            log_pi = torch.log_softmax(route, dim=-1)
            out = None
            for m in range(self.n_mixture):
                term = self._mixture_term(x, m) + log_pi[..., m:m + 1]
                out = term if out is None else torch.logaddexp(out, term)
            return out
        return self._sparse_mixture(x, route)

    def _load_balance(self, route):
        """Switch-Transformer load-balancing term, ``K * sum_m f_m * P_m``.

        ``f_m`` is the fraction of tokens routed to component ``m`` and ``P_m``
        the mean router probability for it. Minimised at a uniform assignment,
        where it equals 1. Without it, top-1 routing concentrates: the component
        that happens to win early gets all the gradient, gets better, and wins
        more, and the sweep measures a single frozen subspace while paying for K.
        """
        flat = route.reshape(-1, self.n_mixture)
        probs = torch.softmax(flat, dim=-1)
        top1 = flat.argmax(dim=-1)
        frac = torch.zeros(self.n_mixture, device=flat.device, dtype=probs.dtype)
        frac.scatter_add_(0, top1, torch.ones_like(top1, dtype=probs.dtype))
        frac = frac / max(flat.shape[0], 1)
        return self.n_mixture * (frac * probs.mean(dim=0)).sum()

    def _mixture_term(self, x, m: int):
        # `.float()` on a bf16 head output already copies, so `lg` is owned here
        # and `_mask_padding` may write into it.
        lg = self._component_logits(x, m).float()
        if self.logit_act == "sigsoftmax":
            lg = lg + F.logsigmoid(lg)
        return torch.log_softmax(self._mask_padding(lg), dim=-1)

    def _sparse_mixture(self, x, route):
        """Top-k routing over components, each with its own Phi.

        Only the selected components are evaluated, so the per-token cost is
        ``k/K`` of the dense mixture while the reachable set is the *union* of
        the K subspaces rather than any one of them.  Dispatch is a gather on the
        flattened token axis, the same shape of trick the MoE blocks in this repo
        use, so the saving is real rather than a mask over wasted work.

        MEMORY IS THE BINDING CONSTRAINT HERE, NOT COMPUTE.  One ``(N, V)`` fp32
        tensor is 34 GB at N=262144 and V=32768, which is already what the dense
        baseline holds.  A first version allocated three of those per slot (a
        ``-inf`` fill, a broadcast add, and a logaddexp result) and died on
        `loss.backward()` with 106 GB resident on a 140 GB card.

        ``k=1`` now holds exactly one, matching the dense baseline: the
        renormalised router weight over a single element is identically 0, so
        there is nothing to weight and nothing to combine, and the components
        partition the token axis so the buffer needs no fill.  ``k>1`` is
        inherently ``k`` log-softmax outputs plus the running combination, so it
        needs a smaller ``--device-batch-size``; gradient accumulation keeps the
        total batch identical and the comparison valid.
        """
        shape = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        n = flat.shape[0]
        k = self.mixture_topk
        top_v, top_i = route.reshape(-1, self.n_mixture).topk(k, dim=-1)
        # With k=1 the renormalised router weight is log_softmax of a single
        # element, which is exactly 0, so the whole mixture collapses to "each
        # token uses its own component" and needs no weighting and no combining.
        # k=1 emits RAW logits and lets the caller's cross-entropy do the single
        # normalisation; k>1 has to combine normalised components, so it cannot.
        raw = (k == 1)
        log_pi = None if raw else torch.log_softmax(top_v, dim=-1)
        out = None
        for slot in range(k):
            idx = top_i[:, slot]
            # Uninitialised, not -inf filled: the components partition the token
            # axis, so every row is written exactly once. Asserted below.
            # Match the components' own dtype on the raw path. Promoting each
            # component's (n_m, V) slice to fp32 before the copy costs one full
            # (N, V) fp32 tensor in transients, and the caller is about to call
            # .float() on the result anyway, so it is a conversion done twice.
            # The dense baseline returns bf16 from its matmul for the same reason.
            acc_dtype = flat.dtype if raw else torch.float32
            acc = torch.empty(n, self.vocab_size, device=flat.device, dtype=acc_dtype)
            covered = 0
            for m in range(self.n_mixture):
                sel = (idx == m).nonzero(as_tuple=True)[0]
                if sel.numel() == 0:
                    continue
                covered += sel.numel()
                sub = flat.index_select(0, sel)
                term = self._component_logits(sub, m) if raw \
                    else self._mixture_term(sub, m)
                acc.index_copy_(0, sel, term.to(dtype=acc_dtype))
            assert covered == n, f"routing left {n - covered} tokens unassigned"
            if log_pi is not None:
                acc += log_pi[:, slot].unsqueeze(-1)          # in place
            out = acc if out is None else torch.logaddexp(out, acc)
        return out.view(*shape, self.vocab_size)

    # -- accounting ---------------------------------------------------------
    def flops_per_token(self) -> int:
        """Exact head FLOPs per token (forward + backward).

        The frozen ``Phi`` product costs 4 rather than 6 per MAC: 2 in the
        forward, 2 for the gradient with respect to ``g(h)``, and *none* for a
        weight gradient, because ``Phi`` has none.  Every learned matrix costs
        the usual 6.  ``GPT.estimate_flops`` removes this head's parameters from
        its generic ``6 * params`` term and adds this number instead, so the
        "17x fewer parameters, 4x more compute" trade shows up in the FLOPs
        column instead of hiding in it.
        """
        V, M = self.vocab_size, self.width
        # Only the routed components run, so cost tracks k rather than K.
        active = self.mixture_topk
        f = sum(g.flops_per_token() for g in self.g) * active // max(self.n_mixture, 1)
        if self.phi_mode == "product" and self.product_impl == "gather":
            # A gather and add, not a matmul: g lookups and g-1 adds per
            # vocabulary entry forward, and a scatter-add of the same shape in
            # the backward.  2 * V * groups per pass, no weight gradient because
            # the assignment is frozen.  This is the ARITHMETIC; the wall clock
            # does not follow it without a fused kernel (Q8), so a sweep run on
            # this path must publish step times beside this column.
            f += active * 4 * V * self.cfg["product_groups"]
        elif self.phi_learned is not None:
            f += active * 6 * V * M
        else:
            f += active * 4 * V * M
        if self.router is not None:
            f += 6 * self.router.weight.numel()
        if self.res_emb is not None:
            f += 6 * (self.res_down.weight.numel() + V * self.residual_rank)
        if self.bias is not None:
            f += 2 * V
        return int(f)

    def extra_repr(self):
        # A learned Phi uses no code at all, so reporting B, the interaction
        # order and the code mode for it is a lie the startup line was telling on
        # every low-rank control arm. Only the modes that read `codes` name them.
        coded = self.phi_mode in ("monomial", "random_binary", "onehot")
        code_bits = (f"B={self.bits}, order={self.order}, " if coded else "")
        code_mode = (f"code={self.cfg['code_mode']}, " if coded else "")
        return (f"V={self.vocab_size}, {code_bits}M={self.width}, "
                f"phi={self.phi_mode}, {code_mode}g={self.cfg['g_type']}, "
                + (f"prod={self.cfg['product_groups']}x{self.cfg['product_codebook']}"
                   f"/{self.cfg['product_source']}/{self.cfg['product_impl']}, "
                   if self.phi_mode == "product" else "")
                + (f"whiten=1, " if self.cfg["phi_whiten"] else "")
                + f"mixture={self.n_mixture}"
                + (f"(top{self.mixture_topk}"
                   f"{',per-phi' if self.per_phi else ''})" if self.n_mixture > 1 else "")
                + f", bias={int(self.bias is not None)}, residual_rank={self.residual_rank}, "
                f"act={self.logit_act}, rank_ceiling={self.rank_ceiling()}")

    def rank_ceiling(self) -> int | float:
        """The theoretical rank ceiling of this configuration's logit matrix.

        ``min(M, d)`` for a linear ``g``; ``M`` for a nonlinear one; unbounded
        (reported as ``inf``) once a log-sum-exp mixture or a pointwise
        nonlinearity is in play.  Printed at startup so a sweep row can be read
        against the ~1000 empirical threshold without recomputing it by hand.
        """
        if self.self_normalized or (self.n_mixture > 1 and self.per_phi):
            # Per-component Phi with routing reaches a UNION of subspaces, so the
            # logit matrix over a batch is not confined to any one of them and
            # the bound does not apply, hard routing included.
            return float("inf")
        base = self.width if self.cfg["g_type"] == "mlp" else min(self.width, self.n_embd)
        if self.phi_mode == "product":
            # One-hot per group means the g group blocks each contain the
            # all-ones vector, so g-1 of those directions are redundant.
            base = min(base, self.width - (self.cfg["product_groups"] - 1))
        return base + self.residual_rank


# ---------------------------------------------------------------------------
# Hierarchical softmax baseline
# ---------------------------------------------------------------------------

def build_huffman_paths(freqs: torch.Tensor, vocab_size: int):
    """Huffman tree over token frequencies -> (nodes, dirs, mask, depth).

    ``nodes[w, i]`` is the internal-node index at step ``i`` of token ``w``'s
    root-to-leaf path, ``dirs[w, i] in {+1, -1}`` the branch taken, and
    ``mask[w, i]`` marks the used prefix.  Loss for token ``w`` is
    ``-sum_i log sigma(dirs * (u_{nodes} . h))``, which touches O(log V) node
    vectors instead of V rows.
    """
    f = freqs[:vocab_size].double().clamp_min(1e-9)
    heap = [(float(f[i]), i) for i in range(vocab_size)]   # (weight, node id)
    heapq.heapify(heap)
    # Leaves are 0..V-1; internal nodes are V..2V-2 and index into U as (id - V).
    children = {}
    next_id = vocab_size
    while len(heap) > 1:
        w1, n1 = heapq.heappop(heap)
        w2, n2 = heapq.heappop(heap)
        children[next_id] = (n1, n2)
        heapq.heappush(heap, (w1 + w2, next_id))
        next_id += 1
    root = heap[0][1]

    depth = 0
    paths = [None] * vocab_size
    stack = [(root, [])]
    while stack:
        node, path = stack.pop()
        if node < vocab_size:
            paths[node] = path
            depth = max(depth, len(path))
            continue
        left, right = children[node]
        internal = node - vocab_size
        stack.append((left, path + [(internal, +1)]))
        stack.append((right, path + [(internal, -1)]))

    depth = max(depth, 1)
    nodes = torch.zeros(vocab_size, depth, dtype=torch.int64)
    dirs = torch.zeros(vocab_size, depth, dtype=torch.float32)
    mask = torch.zeros(vocab_size, depth, dtype=torch.bool)
    for w, path in enumerate(paths):
        for i, (n, d) in enumerate(path):
            nodes[w, i], dirs[w, i], mask[w, i] = n, float(d), True
    return nodes, dirs, mask, depth


class HierarchicalSoftmaxHead(nn.Module):
    """Huffman hierarchical softmax (Morin & Bengio 2005), the tree baseline.

    Included because a reviewer will ask for it: it is the classical way to make
    the output layer cost ``log V``, and it is *structurally different* from an
    interaction expansion (a tree, not a feature map).  It cannot produce a full
    logit vector cheaply, so it supports the loss path only; generation and the
    rank probe raise.
    """

    custom_loss = True
    self_normalized = True

    def __init__(self, config, padded_vocab_size: int, n_embd: int):
        super().__init__()
        self.vocab_size = padded_vocab_size
        self.n_embd = n_embd
        self.tokenizer_dir = getattr(config, "_tokenizer_dir", None)
        self.chunk = 8192
        # Depth is data-dependent; allocate the worst case (a degenerate chain is
        # impossible with positive weights, but ceil(log2 V) * 3 is a safe bound
        # for the Huffman depth of a Zipf distribution) and trim after build.
        self.node_emb = nn.Parameter(torch.empty(max(padded_vocab_size - 1, 1), n_embd))
        # Average root-to-leaf path length, which is what the head actually costs.
        # Until the Huffman tree is built the balanced-tree depth is the right
        # estimate, and it is a plain float so it survives the meta device.
        self.avg_depth = float(minimal_bits(padded_vocab_size))
        depth_bound = max(2 * minimal_bits(padded_vocab_size), 8)
        self.register_buffer("nodes", torch.empty(padded_vocab_size, depth_bound, dtype=torch.int64))
        self.register_buffer("dirs", torch.empty(padded_vocab_size, depth_bound, dtype=torch.float32))
        self.register_buffer("mask", torch.empty(padded_vocab_size, depth_bound, dtype=torch.bool))

    def init_weights(self):
        device = self.node_emb.device
        freqs = load_freq_table(self.vocab_size, self.tokenizer_dir)
        if freqs is None:
            print0("[SCH] hsoftmax: no freq_table.pt, falling back to a uniform "
                   "(balanced-tree) Huffman code")
            freqs = torch.ones(self.vocab_size)
        nodes, dirs, mask, depth = build_huffman_paths(freqs, self.vocab_size)
        bound = self.nodes.shape[1]
        if depth > bound:
            # Reallocate rather than truncate: a truncated path is a wrong model.
            self.nodes = torch.empty(self.vocab_size, depth, dtype=torch.int64, device=device)
            self.dirs = torch.empty(self.vocab_size, depth, dtype=torch.float32, device=device)
            self.mask = torch.empty(self.vocab_size, depth, dtype=torch.bool, device=device)
        else:
            pad = bound - depth
            V = self.vocab_size
            nodes = torch.cat([nodes, torch.zeros(V, pad, dtype=nodes.dtype)], dim=1)
            dirs = torch.cat([dirs, torch.zeros(V, pad, dtype=dirs.dtype)], dim=1)
            mask = torch.cat([mask, torch.zeros(V, pad, dtype=torch.bool)], dim=1)
        self.nodes.copy_(nodes.to(device))
        self.dirs.copy_(dirs.to(device))
        self.mask.copy_(mask.to(device))
        # Huffman is shorter than a balanced tree on a Zipf distribution, so this
        # is the number the FLOP column should carry, not ceil(log2 V).
        self.avg_depth = float(self.mask.sum(dim=1).float().mean().item())
        torch.nn.init.normal_(self.node_emb, mean=0.0, std=0.001)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Resize the path buffers to the checkpoint before copying into them.

        The Huffman depth is a property of the token-frequency distribution, not
        of the config, so the constructor can only guess a bound and
        ``init_weights`` reallocates when the real tree is deeper. A checkpoint
        therefore carries whatever depth that run happened to produce (38 with a
        real frequency table at V=32768, against the constructor's bound of 30),
        and a freshly built model is back at the bound. Without this the load
        fails on a shape mismatch and a completed run cannot be re-measured.
        """
        for name in ("nodes", "dirs", "mask"):
            key = prefix + name
            if key not in state_dict:
                continue
            want = state_dict[key].shape
            cur = getattr(self, name)
            if tuple(cur.shape) != tuple(want):
                setattr(self, name, torch.empty(want, dtype=cur.dtype, device=cur.device))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        # avg_depth is derived from the mask, so it has to follow it or the FLOP
        # column would report the constructor's guess for a restored run.
        try:
            if self.mask.numel() and not self.mask.is_meta:
                self.avg_depth = float(self.mask.sum(dim=1).float().mean().item())
        except (NotImplementedError, RuntimeError):   # pragma: no cover - meta device
            pass

    def forward(self, x):
        raise NotImplementedError(
            "hierarchical softmax cannot materialise a full logit vector cheaply; "
            "use head.loss(x, targets) for training/eval and a different head for "
            "generation or the rank probe")

    def loss(self, x, targets, reduction: str = "mean"):
        """Per-token NLL of ``targets`` under the tree."""
        B, T = targets.shape
        xf = x.reshape(-1, self.n_embd)
        tf = targets.reshape(-1)
        valid = tf >= 0
        safe = torch.where(valid, tf, torch.zeros_like(tf))
        out = torch.zeros(tf.shape[0], device=x.device, dtype=torch.float32)
        for s in range(0, tf.shape[0], self.chunk):
            sl = slice(s, s + self.chunk)
            n = self.nodes[safe[sl]]                       # (c, L)
            d = self.dirs[safe[sl]]
            m = self.mask[safe[sl]]
            u = self.node_emb[n]                           # (c, L, d)
            s_logit = torch.einsum("cld,cd->cl", u.float(), xf[sl].float())
            nll = -F.logsigmoid(d * s_logit) * m
            out[sl] = nll.sum(dim=-1)
        out = out * valid.float()
        if reduction == "none":
            return out.view(B, T)
        return out.sum() / valid.float().sum().clamp_min(1.0)

    def flops_per_token(self) -> int:
        # Read the cached Python float, never the buffer. Reading `mask` here
        # fails outright on the meta device and, before `init_weights` has run,
        # silently reports the average of uninitialised memory on a real one.
        return int(6 * max(self.avg_depth, 1.0) * self.n_embd)


class MonarchHead(nn.Module):
    """Monarch-factorised output head: two block-diagonal factors with a
    transpose between them.

    The code head freezes ``Phi`` and pays for it in alignment.  This attacks the
    same FLOP bill from the other side: keep the map fully *learned*, so there is
    no alignment question at all, and make it cheap structurally instead.

        z  = W1 h                        (d -> m1*m2, dense but small)
        Z  = z.view(m1, m2).T            (m2, m1)
        y_i = B_i Z[i]                   for each of m2 blocks, m1 -> V/m2
        out = concat(y_0..y_{m2-1})      (V,)

    Cost is ``d*M + V*m1`` instead of ``V*d``, with ``M = m1*m2``.  At V=131072,
    d=768, M=1024, m1=32 that is 5.0M MACs against 100.7M, a 20x reduction, and
    every parameter is trained.  Monarch (Dao et al., 2022) is the general class;
    this is the rectangular instance that fits an output layer.

    ``m2`` must divide ``V``.  The head reports the factorisation it chose.
    """

    custom_loss = False
    self_normalized = False

    def __init__(self, config, padded_vocab_size: int, n_embd: int):
        super().__init__()
        V = self.vocab_size = padded_vocab_size
        self.n_embd = n_embd
        M = int(getattr(config, "sch_max_m", 0)) or 1024
        m1 = int(getattr(config, "sch_monarch_m1", 0))
        if m1 <= 0:
            m1 = max(1, int(round(math.sqrt(M))))
        # m2 must divide V and m1*m2 must be the width we advertise.
        m2 = max(1, M // m1)
        while m2 > 1 and V % m2 != 0:
            m2 -= 1
        assert m2 >= 1 and V % m2 == 0, f"could not factor V={V} for M={M}"
        self.m1, self.m2, self.width = m1, m2, m1 * m2
        self.block_out = V // m2
        self.w1 = nn.Linear(n_embd, self.width, bias=False)
        # Block-diagonal second factor, stored stacked: (m2, block_out, m1).
        self.w2 = nn.Parameter(torch.empty(m2, self.block_out, m1))
        self.bias = nn.Parameter(torch.empty(V)) if int(getattr(config, "sch_bias", 0)) else None

    def init_weights(self):
        s = 3 ** 0.5 * self.n_embd ** -0.5
        torch.nn.init.uniform_(self.w1.weight, -s, s)
        torch.nn.init.normal_(self.w2, mean=0.0, std=0.001)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        shape = x.shape[:-1]
        # Cast the weight to the activation dtype, as nanochat's Linear does.
        z = F.linear(x, self.w1.weight.to(dtype=x.dtype)).view(*shape, self.m1, self.m2)
        z = z.transpose(-1, -2)                              # (..., m2, m1)
        # (m2, block_out, m1) x (..., m2, m1) -> (..., m2, block_out)
        y = torch.einsum("obi,...oi->...ob", self.w2.to(dtype=z.dtype), z)
        # `reshape` copies here and cannot avoid it: bmm requires the batch axis
        # first, so the result is (m2, N, block_out) while the caller needs
        # (N, m2, block_out), and those two orders cannot share memory. That copy
        # is a second full (N, V) tensor, 34 GB at V=131072 with 131072 tokens,
        # and it is why this head needs a smaller device batch than dense for the
        # same model. Removing it needs a fused grouped-GEMM kernel that writes
        # the output transposed; see OPEN_QUESTIONS Q11.
        out = y.reshape(*shape, self.vocab_size)
        if self.bias is not None:
            # In place. `out` is the fresh tensor that reshape just copied into,
            # so nobody else holds it, and the alternative allocates a THIRD
            # full-width tensor for what is a per-token constant.
            out += self.bias.to(dtype=out.dtype)
        return out

    def rank_ceiling(self) -> int:
        return min(self.width, self.n_embd) + 1

    def flops_per_token(self) -> int:
        f = 6 * (self.w1.weight.numel() + self.w2.numel())
        if self.bias is not None:
            f += 2 * self.vocab_size
        return int(f)

    def extra_repr(self):
        return (f"V={self.vocab_size}, M={self.width}, m1={self.m1}, m2={self.m2}, "
                f"block_out={self.block_out}, rank_ceiling={self.rank_ceiling()}")


# ---------------------------------------------------------------------------
# Input side (Phase 3)
# ---------------------------------------------------------------------------

class CodeInputEmbedding(nn.Module):
    """Coded token embeddings: ``E = C U``, ``phi_k(c) U`` or ``MLP(c)``.

    The economics differ from the output side and the paper must say so: an
    input embedding is a *gather*, O(1) compute, costing parameters only, so the
    efficiency argument is weaker here.  The rank damage, however, is worse: the
    constraint propagates through the entire network with no normalisation to
    hide behind.  The ``linear`` arm (rank <= B) is expected to fail badly and is
    run precisely because a dramatic failure at exactly the predicted rank
    confirms the mechanism on both sides.
    """

    def __init__(self, mode: str, codes_source, n_embd: int, hidden: int = 0):
        super().__init__()
        assert mode in INPUT_MODES and mode != "table"
        self.mode = mode
        self.n_embd = n_embd
        self._src = [codes_source]     # list -> not registered as a submodule
        if mode in ("expanded", "tied"):
            assert isinstance(codes_source, StructuredCodeHead) and \
                codes_source.phi_mode != "learned", (
                "input modes 'expanded'/'tied' read the output head's frozen Phi, so they "
                "require a structured (non-learned) output head")
        in_dim = codes_source.bits if mode in ("linear", "nonlinear") else codes_source.width
        self.in_dim = in_dim
        if mode == "nonlinear":
            hidden = hidden if hidden > 0 else 4 * n_embd
            self.up = nn.Linear(in_dim, hidden, bias=False)
            self.down = nn.Linear(hidden, n_embd, bias=False)
            self.proj = None
        elif mode == "tied":
            # "Coded weight tying": the input map is the transpose of the output
            # head's final projection, so one matrix serves both directions and
            # the code is genuinely shared rather than merely duplicated.
            assert codes_source.cfg["g_type"] == "linear", \
                "input mode 'tied' needs a linear g so the transpose is well defined"
            self.proj = self.up = self.down = None
        else:
            self.proj = nn.Linear(in_dim, n_embd, bias=False)
            self.up = self.down = None

    @property
    def source(self):
        return self._src[0]

    @property
    def weight(self):
        """Stand-in for ``nn.Embedding.weight``.

        ``GPT`` reads ``transformer.wte.weight`` in three places purely to learn a
        device or a dtype (``get_device``, the COMPUTE_DTYPE cast, the RoPE cache
        build). Exposing the first real matrix keeps those call sites working
        without a type check at each one.
        """
        if self.mode == "tied":
            return self.source.g[0].net[-1].weight
        return self.proj.weight if self.proj is not None else self.up.weight

    def _features(self, idx):
        src = self.source
        if self.mode in ("linear", "nonlinear"):
            return src.codes[idx].to(dtype=self.dtype_ref)
        return src.phi[idx].to(dtype=self.dtype_ref)

    @property
    def dtype_ref(self):
        if self.mode == "tied":
            return self.source.g[0].net[-1].weight.dtype
        w = self.proj.weight if self.proj is not None else self.up.weight
        return w.dtype

    def forward(self, idx):
        f = self._features(idx)
        if self.mode == "nonlinear":
            return self.down(F.relu(self.up(f)).square())
        if self.mode == "tied":
            w = self.source.g[0].net[-1].weight        # (M, d)
            return F.linear(f, w.t().to(dtype=f.dtype))
        return self.proj(f)

    def init_weights(self):
        s = 3 ** 0.5 * self.in_dim ** -0.5
        for lin in [m for m in (self.proj, self.up, self.down) if m is not None]:
            torch.nn.init.uniform_(lin.weight, -s, s)

    def flops_per_token(self) -> int:
        if self.mode == "nonlinear":
            return 6 * (self.up.weight.numel() + self.down.weight.numel())
        if self.mode == "tied":
            return 6 * self.source.g[0].net[-1].weight.numel()
        return 6 * self.proj.weight.numel()


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def resolve_sch_config(config, padded_vocab_size: int) -> dict:
    """Read the ``sch_*`` fields off a GPTConfig and resolve derived quantities.

    Single source of truth for B, M and the width cap, so the head, the FLOP
    estimate, the sweep scripts and the diagnostics all agree.
    """
    def g(name, default):
        return getattr(config, name, default)

    code_mode = g("sch_code_mode", "binary")
    phi_mode = g("sch_phi_mode", "monomial")
    g_type = g("sch_g_type", "linear")
    logit_act = g("sch_logit_act", "none")
    assert code_mode in CODE_MODES, f"sch_code_mode={code_mode!r} not in {CODE_MODES}"
    assert phi_mode in PHI_MODES, f"sch_phi_mode={phi_mode!r} not in {PHI_MODES}"
    product_source = g("sch_product_source", "hash")
    assert product_source in PRODUCT_SOURCES, \
        f"sch_product_source={product_source!r} not in {PRODUCT_SOURCES}"
    assert g_type in G_TYPES, f"sch_g_type={g_type!r} not in {G_TYPES}"
    assert logit_act in LOGIT_ACTS, f"sch_logit_act={logit_act!r} not in {LOGIT_ACTS}"

    bits = int(g("sch_bits", 0)) or minimal_bits(padded_vocab_size)
    ecc_bits = int(g("sch_code_ecc_bits", 0))
    bits_total = bits + ecc_bits
    order = int(g("sch_order", 2))
    max_m = int(g("sch_max_m", 0))

    if phi_mode == "product":
        # K-ary product code: M = groups * codebook at order 1, no interactions.
        groups = int(g("sch_product_groups", 8))
        codebook = int(g("sch_product_codebook", 256))
        assert groups >= 1 and codebook >= 2, "sch_product_groups>=1, sch_product_codebook>=2"
        if codebook ** groups < padded_vocab_size:
            min_g = math.ceil(math.log(padded_vocab_size) / math.log(codebook))
            min_k = math.ceil(padded_vocab_size ** (1.0 / groups))
            raise AssertionError(
                f"a {groups}-digit code over {codebook} symbols has {codebook ** groups} "
                f"cells for {padded_vocab_size} tokens. Collisions make tokens "
                f"indistinguishable at any interaction order, which puts a hard floor "
                f"under the loss. At V={padded_vocab_size} either raise "
                f"sch_product_groups to >= {min_g} at K={codebook}, or raise "
                f"sch_product_codebook to >= {min_k} at g={groups}.")
        width = groups * codebook
        assert width <= MAX_PHI_WIDTH, f"M={width} exceeds MAX_PHI_WIDTH={MAX_PHI_WIDTH}"
        density = 1.0 / codebook
        impl = g("sch_product_impl", "dense")
        assert impl in PRODUCT_IMPLS, f"sch_product_impl={impl!r} not in {PRODUCT_IMPLS}"
    elif phi_mode == "monomial":
        assert 1 <= order <= bits_total, \
            f"sch_order={order} must be in [1, B={bits_total}]"
        width = full_phi_width(bits_total, order)
        if max_m > 0:
            width = min(width, max_m)
        width = min(width, MAX_PHI_WIDTH)
        density = None
    else:
        # For the non-monomial arms M is set directly: it is the matched-capacity
        # width, so the sweep can hold M fixed while swapping what fills it.
        width = max_m if max_m > 0 else full_phi_width(bits_total, order)
        width = min(width, MAX_PHI_WIDTH)
        # Match the monomial arm's expected row density so "frozen random binary"
        # differs from "frozen structured binary" only in structure.
        if phi_mode == "random_binary":
            density = float(g("sch_phi_density", 0.0)) or _expected_monomial_density(bits_total, order)
        else:
            density = 0.0

    assert width >= 1
    if width > padded_vocab_size:
        print0(f"[SCH] M={width} exceeds V={padded_vocab_size}; the expansion is no "
               f"longer a compression and the exact softmax is reached at M=V")

    return {
        "bits": bits,
        "ecc_bits": ecc_bits,
        "bits_total": bits_total,
        "order": order,
        "max_m": max_m,
        "width": width,
        "phi_mode": phi_mode,
        "phi_density": density if density is not None else 0.0,
        "product_groups": int(g("sch_product_groups", 8)),
        "product_codebook": int(g("sch_product_codebook", 256)),
        "product_source": product_source,
        "product_impl": g("sch_product_impl", "dense"),
        "phi_whiten": bool(int(g("sch_phi_whiten", 0))),
        "mixture_per_phi": bool(int(g("sch_mixture_per_phi", 0))),
        "mixture_topk": int(g("sch_mixture_topk", 0)),
        "phi_dtype": g("sch_phi_dtype", "bf16"),
        "phi_normalize": bool(int(g("sch_phi_normalize", 1))),
        "phi_center": bool(int(g("sch_phi_center", 0))),
        "code_mode": code_mode,
        "code_path": g("sch_code_path", ""),
        "code_seed": int(g("sch_code_seed", 1234)),
        "mono_seed": int(g("sch_code_seed", 1234)) + 17,
        "g_type": g_type,
        "g_hidden": int(g("sch_g_hidden", 0)),
        "g_layers": int(g("sch_g_layers", 2)),
        "g_out_std": float(g("sch_g_out_std", 0.001)),
        "mixture": max(1, int(g("sch_mixture", 1))),
        "residual_rank": int(g("sch_residual_rank", 0)),
        "logit_act": logit_act,
        "bias": bool(int(g("sch_bias", 0))),
        "tokenizer_dir": getattr(config, "_tokenizer_dir", None),
    }


def _expected_monomial_density(bits: int, order: int) -> float:
    """E[phi_S(c)] for a uniform random code, averaged over the kept monomials.

    A size-j monomial is 1 with probability 2^-j, so the density of the whole
    expansion is ``sum_j C(B,j) 2^-j / sum_j C(B,j)``.  Matching it is what makes
    the random-binary control a control rather than a different sparsity level.
    """
    num = sum(math.comb(bits, j) * (0.5 ** j) for j in range(1, order + 1))
    den = sum(math.comb(bits, j) for j in range(1, order + 1))
    return float(num / max(den, 1))


def build_code_head(config, padded_vocab_size: int, n_embd: int) -> nn.Module:
    """Factory used by ``GPT.__init__``."""
    head_type = getattr(config, "sch_head_type", "code")
    assert head_type in HEAD_TYPES, f"sch_head_type={head_type!r} not in {HEAD_TYPES}"
    if head_type == "hsoftmax":
        return HierarchicalSoftmaxHead(config, padded_vocab_size, n_embd)
    if head_type == "monarch":
        return MonarchHead(config, padded_vocab_size, n_embd)
    assert head_type == "code", f"unknown sch_head_type={head_type!r}"
    return StructuredCodeHead(config, padded_vocab_size, n_embd)


def describe_head(head: nn.Module) -> str:
    """One-line startup summary, printed by GPT.__init__."""
    if isinstance(head, HierarchicalSoftmaxHead):
        return (f"[SCH] head=hsoftmax V={head.vocab_size} d={head.n_embd}, "
                f"avg_depth={head.avg_depth:.2f} | head params "
                f"{sum(p.numel() for p in head.parameters()):,} "
                f"| head FLOPs/token {head.flops_per_token():,} "
                f"| dense equivalent {6 * head.vocab_size * head.n_embd:,}")
    params = sum(p.numel() for p in head.parameters())
    kind = "monarch" if isinstance(head, MonarchHead) else "code"
    return (f"[SCH] head={kind} {head.extra_repr()} | head params {params:,} "
            f"| head FLOPs/token {head.flops_per_token():,} "
            f"| dense equivalent {6 * head.vocab_size * head.n_embd:,}")
