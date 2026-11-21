# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Canonical dataflow of paged-KV attention: the classes ``attn.bf16.d<D>`` (causal), ``attn.bf16.d<D>.nc`` (non-causal),
``attn.bf16.d<D>.br`` (causal with a bidirectional key range per row) and ``attn.bf16.d<D>.mask`` (an explicit key set per row).

One dataflow, four ways of saying which keys a row sees (``MASK_MODES``). The mask mode is a class dimension, not a per-call
input. A row has a set of visible keys of its request; the tiles, the chunks, the online softmax and the merge are stated over
that set alone (``RowSpec.visible``): a key outside it is an absent leaf, a tile or a chunk without a visible key is dropped, and a
row without one outputs ``+0`` (``output_elem_or_zero``). The term of a row is a function of its own visible set, so two rows with
the same set have the same term whatever mode stated it (the receipts differ: a receipt names what a row MAY see):

* causal: the keys ``[lo, p + 1)`` of the row's position ``p`` (bottom-right aligned, ``lo`` from the optional sliding window);
* non-causal: every key of the row's request, ``[0, kv_len)`` (no window) -- the causal row at the request's last position;
* key ranges: the causal keys united with one key range ``[a, b)`` of the request -- a row with an empty range is the causal row,
  a row whose range is ``[0, kv_len)`` is the non-causal row;
* key set: the set is data (a custom attention mask) -- a row whose set is a causal range is the causal row, one whose set is
  every key of the request is the non-causal row, and one whose set is ``causal ∪ [a, b)`` is the key-range row
  (``RowSpec.as_key_set``).

The class is stated for finite bf16 operands: a non-finite Q / K / V value, or scaled scores outside the binary32 range, give
the NaN that fp32 softmax arithmetic gives (bit for bit, like every other value) and are not guarded against.

Everything numeric lives here once, parameterised by an ``fp`` backend (``TermAlgebra`` for the
symbolic spec and the Lean coupling, ``CuteF32`` inside the kernel). Per logical row ``(p, h)`` with visible keys ``[lo, L)``:

* the score of key ``k`` is the GEMM's ``kfold`` over ``D`` (opaque ``MMA16`` K-steps, ascending);
* a KV **tile** (``T`` keys at an absolute multiple of ``T``) is one step of a ``fold`` on the state
  ``(m, l, O[d])``: the tile's row maximum and row sum are the engine's ``pairwise`` reductions over
  the key digit (``rowmax_reduction``, ``rowsum_reduction``), the rescale ``alpha`` and the
  probabilities are ``EX2`` of ``.rn`` terms (``tile_step``), and ``O`` advances by the opaque PV MMA
  fold from ``mul(O, alpha)`` (``pv_fold``);
* a **chunk** (``C`` keys at an absolute multiple of ``C``) is the tile fold from the init state; the
  chunks are folded in index order with ``merge`` from the same init state (the first present chunk
  is merged with it; absent chunks are dropped);
* the epilogue is ``cvt_bf16(add(mul(O, div(1, l)), +0))``.

Tile and chunk boundaries are absolute, ``T`` and ``C`` are functions of ``D`` only, so decode at
position ``p`` and prefill of the token at position ``p`` build the same term (R4), and nothing in
the term depends on the batch, the page table or the configuration (R2, R3).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from ...batch_invariant import canon
from ...batch_invariant.fp_interface.term import to_f32
from ...batch_invariant.canon import Digit, Reduction
from ...gemm.batch_invariant.spec import MMA_OP as _GEMM_MMA_OP
from ...gemm.batch_invariant.spec import reduction as gemm_kfold
from ...batch_invariant.fp_interface.term import Term, TermAlgebra

# ----------------------------------------------------------------------------- the class constants

TILE_KEYS = 128  # T: keys per KV tile (the N extent of one QK^T MMA tile)
MMA_K = 16  # bf16 K extent of one tcgen05.mma step (shared with the GEMM)
FLT_MAX = 3.4028234663852886e38  # the largest binary32 number
MMA_OP = _GEMM_MMA_OP  # "mma_bf16_k16": the same opaque instruction family as the GEMM
HEAD_DIMS = (
    64,
    128,
    256,
)  # the classes attn.bf16.d64 / d128 / d256 (one class per (D, C(D)))
NEG_INF = float("-inf")

# what a row sees (``mask_mode``): the declared values of the one mode switch every signature of the kernels, the planner and the
# fold takes by keyword. A mode names where a row's visible keys come from; the tiles, the chunks, the tile step and the merge are
# the same dataflow under every mode
MASK_CAUSAL = "causal"  # the keys [max(0, p - window_left), p] of the row's position
MASK_NON_CAUSAL = "non_causal"  # every key [0, kv_len) of the row's request (no window)
MASK_KEY_RANGES = "key_ranges"  # the causal keys united with one key range [a, b) per row (data: bidirectional ranges)
MASK_KEY_SET = "key_set"  # an explicit key set per row (data: a custom mask)
MASK_MODES = (MASK_CAUSAL, MASK_NON_CAUSAL, MASK_KEY_RANGES, MASK_KEY_SET)


def check_mask_mode(mask_mode: str) -> str:
    if mask_mode not in MASK_MODES:
        raise ValueError(f"mask_mode {mask_mode!r}: one of {MASK_MODES}")
    return mask_mode


def mask_mode_of_causal(causal: bool) -> str:
    """The mode a ``causal`` flag names: the two modes whose visible keys follow from the request metadata alone. The callers'
    ``causal`` / ``is_causal`` argument is mapped here once; below the wrappers every signature takes ``mask_mode``."""
    return MASK_CAUSAL if causal else MASK_NON_CAUSAL


CHUNK_LEN = 512  # C: keys per chunk (4 tiles) -- a class constant (another C would be a different class)
CHUNK_LEN_OF = {
    64: CHUNK_LEN,
    128: CHUNK_LEN,
    256: CHUNK_LEN,
}  # C(D): the chunk length per head dim (a class parameter per D; all 512)


def chunk_len(D: int, L: Optional[int] = None) -> int:
    """The chunk length of the class: a function of the head dim ONLY (I5). ``L``, the token's own
    kv length ``p + 1``, is accepted so that a KV-length bucket rule can be introduced later as a
    *new class* without changing call sites; the rule ignores it. Never a function of the
    batch, the request's total length, the SM count or the configuration.

    ``C = 512`` (rather than ``1024``): a B = 1 decode step has ``Hkv x ceil(L / C)``
    work units, so halving ``C`` doubles its parallelism at the cost of one more chunk merge per 1024 keys
    (prefill) and twice the split workspace. Every output bit of the class changes with ``C``
    (``RLK.Attn.output_depends_on_C``); the fingerprint records it. ``C`` is a class parameter PER HEAD DIM
    (``CHUNK_LEN_OF``): D = 64 and D = 256 take the same 512 (the D = 256 body's merged chunk state
    lives in a per-CTA global scratch because TMEM holds no second 256-column accumulator, which made a longer
    C a candidate there; the common constant was kept)."""
    assert D in HEAD_DIMS, D
    return CHUNK_LEN_OF[D]


def tiles_per_chunk(D: int) -> int:
    return chunk_len(D) // TILE_KEYS


def d_steps(D: int) -> int:
    return D // MMA_K


def scale_log2(sm_scale: float) -> float:
    """``c = fp32(sm_scale * log2(e))``: formed on the host in double precision and rounded ONCE to
    binary32; the kernel and the CPU model receive exactly this value (and ``-c``). ``OverflowError`` when the product is not a
    binary32 number (``struct.pack`` returns ``inf`` for it on some CPython versions and raises on others)."""
    c = float(sm_scale) * (1.0 / math.log(2.0))
    if not abs(c) <= FLT_MAX:
        raise OverflowError(
            f"sm_scale {sm_scale}: sm_scale * log2(e) is outside the binary32 range"
        )
    return to_f32(c)


# ----------------------------------------------------------------------------- the row and its tiles


@dataclass(frozen=True)
class RowSpec:
    """One logical row and its visible keys, stated in one of the four modes (``mask_mode``): the causal keys
    ``[lo_causal, p + 1)`` of the query position ``p`` (0-based, absolute in the KV sequence; ``lo_causal = max(0, p -
    window_left)``); every key ``[0, kv_len)`` of the request (``causal=False``: the position only names the row and a window is not
    a parameter of that class); the causal keys united with the bidirectional range ``brange = (a, b)`` (the keys ``[a, b)``;
    ``a >= b``: no range, the causal row); or the explicit set ``keys`` over the request's ``kv_len`` keys. ``[lo, L)`` is the hull
    of the keys the row may see; ``row`` names the row (query token and head) in the leaf terms."""

    p: int
    window_left: int = -1
    row: int = 0  # the row's identity in the leaf terms (query token / head index)
    causal: bool = True
    kv_len: Optional[int] = (
        None  # the request's key count (non-causal: the visible range is [0, kv_len); with ``keys``: the set's universe)
    )
    keys: Optional[FrozenSet[int]] = (
        None  # the key-set mode: the row's visible keys; ``p`` and ``window_left`` are then unused
    )
    brange: Optional[Tuple[int, int]] = (
        None  # the key-ranges mode: the bidirectional range [a, b) next to the causal keys
    )

    def __post_init__(self) -> None:
        assert self.keys is None or self.brange is None, (
            "a row states its keys as a set or as causal keys with a range, not both"
        )
        if not self.causal:
            assert self.keys is None and self.brange is None, (
                "a non-causal row sees every key of its request: it takes no key set and no range"
            )
            assert self.kv_len is not None and self.kv_len >= 1, (
                "a non-causal row needs the request's kv_len (at least one key)"
            )
            assert self.window_left < 0, (
                "window_left is not a parameter of the non-causal class"
            )
        if self.keys is not None:
            assert self.kv_len is not None and self.kv_len >= 1, (
                "a row with an explicit key set needs the request's kv_len (at least one key)"
            )
            assert all(0 <= k < self.kv_len for k in self.keys), (
                "the key set is a subset of [0, kv_len)"
            )

    @property
    def mask_mode(self) -> str:
        """The mode that states this row's keys (``MASK_MODES``)."""
        if self.keys is not None:
            return MASK_KEY_SET
        if self.brange is not None:
            return MASK_KEY_RANGES
        return MASK_CAUSAL if self.causal else MASK_NON_CAUSAL

    @property
    def range_keys(self) -> Optional[Tuple[int, int]]:
        """The row's range ``(a, b)`` when it holds a key, else ``None``."""
        if self.brange is None:
            return None
        a, b = self.brange
        assert a >= 0, self.brange
        return (a, b) if a < b else None

    @property
    def lo_causal(self) -> int:
        return 0 if self.window_left < 0 else max(0, self.p - self.window_left)

    @property
    def L(self) -> int:
        """One past the last key the row may see (the hull's end; the request's ``kv_len`` for a non-causal row and a key set)."""
        if not self.causal or self.keys is not None:
            assert self.kv_len is not None
            return self.kv_len
        r = self.range_keys
        return self.p + 1 if r is None else max(self.p + 1, r[1])

    @property
    def lo(self) -> int:
        """The first key the row may see (the hull's start)."""
        if not self.causal or self.keys is not None:
            return 0
        r = self.range_keys
        return self.lo_causal if r is None else min(self.lo_causal, r[0])

    def visible(self, k: int) -> bool:
        """THE visibility predicate: every mode's rows enter the dataflow through it alone."""
        if self.keys is not None:
            return k in self.keys
        if not self.causal:
            return 0 <= k < self.L
        r = self.range_keys
        return self.lo_causal <= k <= self.p or (r is not None and r[0] <= k < r[1])

    def key_set(self) -> FrozenSet[int]:
        """The row's visible keys as a set."""
        return frozenset(k for k in range(self.lo, self.L) if self.visible(k))

    def as_key_set(self, kv_len: Optional[int] = None) -> "RowSpec":
        """The key-set row with this row's visible set (over ``kv_len`` keys; the hull's end by default): the same term, since the
        dataflow reads ``visible`` alone."""
        return RowSpec(
            p=self.p,
            window_left=self.window_left,
            row=self.row,
            kv_len=self.L if kv_len is None else kv_len,
            keys=self.key_set(),
        )


def visible_keys(row: RowSpec, t: int, T: int = TILE_KEYS) -> List[int]:
    """The visible absolute keys of tile ``t`` for ``row`` (empty = the tile is absent)."""
    a, b = max(row.lo, t * T), min(row.L, (t + 1) * T)
    return [k for k in range(a, b) if row.visible(k)]


def present_chunks(row: RowSpec, D: int) -> List[int]:
    C = chunk_len(D, row.L)
    return [
        j
        for j in range(row.lo // C, -(-row.L // C))
        if any(
            row.visible(k) for k in range(max(row.lo, j * C), min(row.L, (j + 1) * C))
        )
    ]


# ----------------------------------------------------------------------------- the engine reductions


def rowmax_reduction(T: int, leaf: Callable[[Any, int], Any]) -> Reduction:
    """``fmax`` over the keys of one tile. ``leaf(fp, k)`` returns the score term/value of key
    ``k`` (0 .. T-1 inside the tile) or ``None`` when the key is masked."""
    key = Digit(
        "key",
        T,
        "pairwise",
        combine=lambda fp, a, b: fp.fmaxc(a, b),
        identity_note="fmax(v, -inf) == v for every non-NaN v: -inf is a global identity of the join "
        "(max.f32 is the join of the total order -inf < .. < -0 < +0 < .. < inf on sm_107)",
    )
    return Reduction(
        f"attn.rowmax[T={T}]", (key,), lambda fp, idx: leaf(fp, idx["key"])
    )


def rowsum_reduction(T: int, leaf: Callable[[Any, int], Any]) -> Reduction:
    """``add`` over the keys of one tile. ``leaf(fp, k)`` returns the probability term/value
    ``p_k`` or ``None`` when the key is masked."""
    key = Digit(
        "key",
        T,
        "pairwise",
        combine=lambda fp, a, b: fp.add(a, b),
        identity_note="add(v, +0) == v for v != -0; every leaf is ex2(.) >= +0 and no partial sum is -0",
    )
    return Reduction(
        f"attn.rowsum[T={T}]", (key,), lambda fp, idx: leaf(fp, idx["key"])
    )


def reduce_keys(
    red: Reduction,
    fp,
    placement: Optional[canon.Placement] = None,
    peers: Optional[canon.Peers] = None,
    loader: Optional[Callable[[tuple], Any]] = None,
):
    """The tile's key reduction: the canonical term when no placement is given (spec, CPU model),
    or the engine's plan executed with the caller's loader and peers (the kernel)."""
    if placement is None:
        return red.canonical(fp)
    p = canon.plan(red, placement)
    assert loader is not None and peers is not None
    state, start = canon.init_state(p, fp, loader)
    return canon.run(p, state, start, peers, fp)


# ----------------------------------------------------------------------------- the arithmetic (written once)


@dataclass(frozen=True)
class State:
    """The online-softmax state of one row for one output element ``d``: running maximum ``m``
    (unscaled), running sum ``l``, accumulator ``O[d]``. ``m = -inf, l = +0, O = +0`` is the init."""

    m: Any
    l: Any
    o: Any


def init_state(fp) -> State:
    return State(fp.const(NEG_INF), fp.const(0.0), fp.const(0.0))


def score_term(fp, row: RowSpec, k: int, D: int):
    """``s_k``: the GEMM's K-fold over ``D`` (``D/16`` opaque ``MMA16`` steps from +0) for A row
    ``row.row`` and B column ``k``: the same reduction object as ``gemm.bf16.kfold[K=D]``."""
    return gemm_kfold(D, row.row, k).canonical(fp)


def alpha_term(fp, m_old, m_new, c, nc):
    """``alpha = EX2(add(mul(m_old, c), mul(m_new, nc)))``: the rescale from the two ROUNDED products
    (``nc = -c`` exactly, so this is ``ms_old - ms_new``); an unchanged maximum gives ``add = +0`` and
    ``alpha = 1.0`` exactly, ``m_old = -inf`` gives ``EX2(-inf) = +0``."""
    return fp.ex2(fp.add(fp.mul(m_old, c), fp.mul(m_new, nc)))


def prob_term(fp, s, m_new, c, nc):
    """``p_k = EX2(fma(s_k, c, mul(m_new, nc)))``."""
    return fp.ex2(fp.fma(s, c, fp.mul(m_new, nc)))


def tile_stats(
    fp,
    m,
    l,
    scores: Sequence[Any],
    c,
    nc,
    placement=None,
    peers=None,
    loader_max=None,
    loader_sum=None,
):
    """The statistics part of one tile step for one row: ``scores[k]`` is the score of key ``k``
    of the tile or ``None`` when masked (at least one present). Returns
    ``(m_new, alpha, probs, l_new)`` with ``probs[k] = +0`` for masked keys."""
    T = len(scores)
    m_t = reduce_keys(
        rowmax_reduction(T, lambda fp_, k: scores[k]), fp, placement, peers, loader_max
    )
    m_new = fp.fmaxc(m, m_t)
    alpha = alpha_term(fp, m, m_new, c, nc)
    nms = fp.mul(m_new, nc)
    probs = [None if s is None else fp.ex2(fp.fma(s, c, nms)) for s in scores]
    r_t = reduce_keys(
        rowsum_reduction(T, lambda fp_, k: probs[k]), fp, placement, peers, loader_sum
    )
    l_new = fp.fma(l, alpha, r_t)
    return m_new, alpha, [fp.const(0.0) if q is None else q for q in probs], l_new


def pv_fold(fp, o_scaled, probs_bf16: Sequence[Any], v_slice: Callable[[int], Any]):
    """``O' = MMA16-fold over the tile's key-steps from ``o_scaled = mul(O, alpha)``: step ``j`` takes
    the bf16 P slice of keys ``16j .. 16j+15`` (masked keys carry ``+0``) and ``v_slice(j)``."""
    acc = o_scaled
    for j in range(len(probs_bf16) // MMA_K):
        p_slice = fp.op("p_slice", *probs_bf16[j * MMA_K : (j + 1) * MMA_K])
        acc = fp.op(MMA_OP, acc, fp.op("pv_kslice", p_slice, v_slice(j)))
    return acc


def tile_step(fp, st: State, tile: Dict[str, Any], c, nc, D: int, d: int) -> State:
    """One tile of the per-chunk fold (the ``tile`` digit's step). ``tile`` = ``{"t": abs tile index,
    "keys": visible absolute keys, "row": RowSpec}``."""
    row, t, keys, T = tile["row"], tile["t"], tile["keys"], tile["T"]
    scores = [
        score_term(fp, row, t * T + k, D) if (t * T + k) in keys else None
        for k in range(T)
    ]
    m_new, alpha, probs, l_new = tile_stats(fp, st.m, st.l, scores, c, nc)
    probs_bf16 = [fp.cvt_out(q, "bf16") for q in probs]
    o_new = pv_fold(
        fp,
        fp.mul(st.o, alpha),
        probs_bf16,
        lambda j: fp.op("v_slice", t * (T // MMA_K) + j, d),
    )
    return State(m_new, l_new, o_new)


def merge(fp, a: State, b: State, c, nc) -> State:
    """The ``chunk`` digit's step: merge the running state ``a`` with the finished chunk ``b``."""
    m = fp.fmaxc(a.m, b.m)
    nms = fp.mul(m, nc)
    a1 = fp.ex2(fp.add(fp.mul(a.m, c), nms))
    a2 = fp.ex2(fp.add(fp.mul(b.m, c), nms))
    return State(m, fp.fma(b.l, a2, fp.mul(a.l, a1)), fp.fma(b.o, a2, fp.mul(a.o, a1)))


def output_elem(fp, st: State, out_fmt: str = "bf16"):
    """``cvt(add(mul(O, div(1, l)), +0))``: a correctly rounded reciprocal, one multiply, and an
    ``add +0`` that makes the sign of an exact-zero accumulator unobservable."""
    inv = fp.div(fp.const(1.0), st.l)
    return fp.cvt_out(fp.add(fp.mul(st.o, inv), fp.const(0.0)), out_fmt)


def output_elem_or_zero(fp, st: Optional[State], out_fmt: str = "bf16"):
    """The output element of the explicit-key-set class: ``output_elem`` of the row's state, or ``cvt(+0)`` for a row without a
    visible key (``st`` is ``None``: no chunk is present, so there is no state to normalise). The rule is declared, not derived:
    the init state's ``mul(+0, div(1, +0))`` would be a NaN. The kernels take it from the row's own ``l`` (``l == +0`` exactly
    when no key of the row is visible: every present tile adds the probability of its maximum, ``ex2`` of a rounding
    residual), replacing the reciprocal by ``+0``."""
    if st is None:
        return fp.cvt_out(fp.const(0.0), out_fmt)
    return output_elem(fp, st, out_fmt)


# ----------------------------------------------------------------------------- the reduction object


def attention_reduction(
    row: RowSpec, D: int, d: int, c, nc, T: int = TILE_KEYS, C: Optional[int] = None
) -> Reduction:
    """The per-row dataflow as an engine ``Reduction`` with two ``fold`` digits: ``tile`` (radix
    ``C/T``, the online softmax inside a chunk) and ``chunk`` (radix = chunks of the prefix, the
    merge). Leaves are tile descriptors; absent tiles (no visible key) are ``None`` and dropped, an
    all-absent chunk folds to ``None`` and is dropped by the chunk fold. ``T`` and ``C`` may be
    shrunk for the Lean cases; the class uses the module constants."""
    C = chunk_len(D, row.L) if C is None else C
    assert C % T == 0
    n_chunks = max(1, -(-row.L // C))

    def leaf(fp, idx):
        t = idx["chunk"] * (C // T) + idx["tile"]
        keys = visible_keys(row, t, T)
        return {"t": t, "keys": keys, "row": row, "T": T} if keys else None

    tile = Digit(
        "tile",
        C // T,
        "fold",
        init=init_state,
        step=lambda fp, st, tl: tile_step(fp, st, tl, c, nc, D, d),
        identity_note="absent tiles are dropped; a kernel's null step (P row = 0, alpha = 1) is exact on (m, l) "
        "and exact on O up to the sign of an exact zero, which the epilogue's add(+0) erases",
    )
    chunk = Digit(
        "chunk",
        n_chunks,
        "fold",
        init=init_state,
        step=lambda fp, a, b: merge(fp, a, b, c, nc),
        identity_note="absent chunks are dropped (never merged)",
    )
    return Reduction(f"attn.bf16.d{D}[T={T},C={C}]", (tile, chunk), leaf)


def fold_placement(red: Reduction) -> canon.Placement:
    """Both folds are register-level: one thread (the softmax thread of the row) folds tiles, one
    merge agent (the correction warp or the fixup thread) folds chunks."""
    return canon.Placement.make({d.name: [(d.radix, "reg")] for d in red.digits})


def verify_plan(red: Reduction, fp) -> Any:
    """``canon.verify`` for a state-valued reduction (``State`` compares field-wise; terms by identity)."""
    p = canon.plan(red, fold_placement(red))
    ref = red.canonical(fp)
    got = canon.trace(p, fp)
    bad = [u for u, v in got.items() if not (v is ref or v == ref)]
    assert not bad, f"{red.name}: {len(bad)} units differ under {p.describe()}"
    return ref


# ----------------------------------------------------------------------------- symbolic spec + fingerprint


class AttentionSpec:
    """Symbolic view: ``output(row, d)`` is the term of one output element."""

    def __init__(
        self,
        D: int = 128,
        alg: Optional[TermAlgebra] = None,
        T: int = TILE_KEYS,
        C: Optional[int] = None,
        out_fmt: str = "bf16",
    ):
        self.D, self.T, self.out_fmt = D, T, out_fmt
        self.C = chunk_len(D) if C is None else C
        self.alg = alg or TermAlgebra()
        self.c, self.nc = self.alg.sym("c"), self.alg.sym("nc")

    def reduction(self, row: RowSpec, d: int) -> Reduction:
        return attention_reduction(row, self.D, d, self.c, self.nc, self.T, self.C)

    def state(self, row: RowSpec, d: int) -> State:
        return self.reduction(row, d).canonical(self.alg)

    def output(self, row: RowSpec, d: int) -> Term:
        # one epilogue for every mode: a row without a visible key (possible under the key-set mode alone) is +0
        return output_elem_or_zero(self.alg, self.state(row, d), self.out_fmt)


_FINGERPRINTS: Dict[tuple, str] = {}
# the recorded fingerprint of the published class (D = 128, no window) at C = 512 (with C = 1024 it was
# attn-bf16-d128-3384db6839e66672). ``tests/attention/test_batch_invariant_attention.py`` pins the three head dims' fingerprints.
FINGERPRINT_D128 = "attn-bf16-d128-027b0379515b82e2"
# the D = 64 and D = 256 classes: the same term with D as a parameter (D/16 MMA K-steps per
# score, a D-wide PV accumulator), C(D) = 512 for both; pinned by the same test
FINGERPRINT_D64 = "attn-bf16-d64-9a05233ec9351b0d"
FINGERPRINT_D256 = "attn-bf16-d256-b8cb59dab4eb6c0f"
FINGERPRINTS = {64: FINGERPRINT_D64, 128: FINGERPRINT_D128, 256: FINGERPRINT_D256}
# the explicit-key-set class per head dim (``mask_fingerprint``); pinned by the same test
MASK_FINGERPRINTS = {
    64: "attn-bf16-d64-mask-a6c0fb6237f4f1da",
    128: "attn-bf16-d128-mask-79713bfb8fb3c998",
    256: "attn-bf16-d256-mask-f94e81a34c536168",
}


# the receipts' infix per mask mode (the causal receipts have none)
FINGERPRINT_INFIX = {
    MASK_CAUSAL: "",
    MASK_NON_CAUSAL: "nc-",
    MASK_KEY_RANGES: "br-",
    MASK_KEY_SET: "mask-",
}


def sample_key_set(T: int = TILE_KEYS, C: int = CHUNK_LEN) -> FrozenSet[int]:
    """The declared key set the explicit-key-set class's fingerprint is sampled on, over ``kv_len = C + T // 2 + 1`` keys: key ``k``
    is visible unless ``k % 3 == 1`` or ``k`` lies in tile 1 -- masked leaves inside every other tile, a tile without a visible key
    inside a chunk, the chunk merge with the init state and a partial last tile are all in the sample row's term."""
    return frozenset(
        k for k in range(C + T // 2 + 1) if k % 3 != 1 and not T <= k < 2 * T
    )


def ranged_sample_row(T: int, C: int) -> RowSpec:
    """The ranged class's sample row: a windowed causal part in the third chunk and a range that starts inside the first tile and ends
    inside the second chunk, so the row has a masked head and tail of a tile on both parts, a chunk of which only the first tile is
    present, tiles and chunk-interior keys absent between the two parts, and three merged chunks."""
    return RowSpec(p=2 * C + T // 2, window_left=T // 4, brange=(T // 2, C + T // 4))


def spec_fingerprint(
    D: int = 128,
    window_left: int = -1,
    causal: Optional[bool] = None,
    *,
    mask_mode: Optional[str] = None,
) -> str:
    """Identifier of the class: hashes the terms of output elements 0 and 1 of a row whose visible keys
    span one full chunk plus one tile (so the tile fold, the chunk merge with the init state, the
    masking of the last tile and the epilogue are all inside), together with ``D``, ``T``, ``C`` and the mask mode. The window is a
    per-call input of the causal class (like the KV position), not a class dimension: the public receipt
    (``flashinfer.batch_invariant.fingerprint``) is the digest of the no-window sample row; ``window_left`` here only selects
    another sample row.

    The mask mode IS a class dimension with one receipt per value, stated once: as ``causal`` (``True`` / omitted: the causal
    class; ``False``: the non-causal class) or as ``mask_mode`` (keyword-only, any of ``MASK_MODES``):

    * ``"non_causal"`` (``attn-bf16-d<D>-nc-*``): a token sees ``[0, kv_len)`` of its request where the causal class gives it
      ``[0, p + 1)``, so the same token has a different term under the two classes; the sample row's term is the same dataflow
      over the same visible keys in both modes, and the mode enters the digest explicitly.
    * ``"key_ranges"`` (``attn-bf16-d<D>-br-*``): rows take a bidirectional key range next to the causal keys. The receipt digests
      ``ranged_sample_row`` and records the mode; a row without a range has the causal class's term under it, so the two receipts
      name the same bits for such rows and differ as identifiers of what a row may see.
    * ``"key_set"`` (``attn-bf16-d<D>-mask-*``): rows take an explicit key set. The receipt digests the row with the declared
      set ``sample_key_set`` and a row without a visible key. The set is DATA of the class, like the KV position of the causal
      class: one receipt covers every mask and the mask's values never enter it. The class contains the other three as functions:
      a row whose set is a causal range, every key of its request, or a causal range united with a key range, has that row's
      term."""
    if causal is not None and not isinstance(causal, bool):
        raise TypeError(
            f"causal {causal!r}: True or False (a mode name goes to the keyword `mask_mode`)"
        )
    if mask_mode is None:
        mask_mode = mask_mode_of_causal(causal is None or causal)
    elif causal is not None:
        raise ValueError("state the mask mode once: `causal` or `mask_mode`")
    check_mask_mode(mask_mode)
    if mask_mode == MASK_NON_CAUSAL:
        if window_left >= 0:
            raise ValueError("window_left is not a parameter of the non-causal class")
    elif mask_mode != MASK_CAUSAL and window_left != -1:
        raise ValueError(f"the {mask_mode} class's receipt has one sample")
    key = (D, window_left, mask_mode)
    if key not in _FINGERPRINTS:
        from ...batch_invariant.fp_interface.term import fingerprint

        spec = AttentionSpec(D)
        parts = [f"D={D}", f"T={spec.T}", f"C={spec.C}"]
        if mask_mode == MASK_KEY_SET:
            kv_len = spec.C + spec.T // 2 + 1
            row = RowSpec(p=0, kv_len=kv_len, keys=sample_key_set(spec.T, spec.C))
            empty = RowSpec(p=0, kv_len=kv_len, keys=frozenset())
            parts += [
                "mask=explicit",
                fingerprint(spec.output(row, 0)),
                fingerprint(spec.output(row, 1)),
                fingerprint(spec.output(empty, 0)),
            ]
        else:
            L = spec.C + spec.T // 2 + 1
            if mask_mode == MASK_KEY_RANGES:
                row = ranged_sample_row(spec.T, spec.C)
                parts.append("mask=causal+ranges")
            elif mask_mode == MASK_NON_CAUSAL:
                row = RowSpec(p=0, causal=False, kv_len=L)
                parts.append("mask=noncausal")
            else:
                row = RowSpec(p=L - 1, window_left=window_left)
            parts += [
                f"win={window_left}",
                fingerprint(spec.output(row, 0)),
                fingerprint(spec.output(row, 1)),
            ]
        _FINGERPRINTS[key] = (
            f"attn-bf16-d{D}-"
            + FINGERPRINT_INFIX[mask_mode]
            + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
        )
    return _FINGERPRINTS[key]


def mask_fingerprint(D: int = 128) -> str:
    """The explicit-key-set class's identifier at head dim ``D``: ``spec_fingerprint(D, mask_mode="key_set")``."""
    return spec_fingerprint(D, mask_mode=MASK_KEY_SET)
