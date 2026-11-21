# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Canonical dataflow of paged-KV attention: the classes ``attn.bf16.d<D>`` (causal) and ``attn.bf16.d<D>.nc`` (non-causal).

The mask mode is a class dimension, not a per-call input: a causal row at position ``p`` sees the keys ``[lo, p + 1)`` (bottom-right
aligned, ``lo`` from the optional sliding window), a non-causal row sees every key of its request, ``[0, kv_len)`` (no window). The
tiles, the chunks, the online softmax and the merge are the same dataflow in both modes; only the visible set a token gets from the
request metadata differs, so a token's term (and the fingerprint) differs between the two classes.

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
* the epilogue is ``cvt_bf16(add(mul(O, div(1, l)), +0))``;
* the LSE member (``lse_term``), a second output of the same row state: ``lse2 = add(mul(m, c), LG2(l))`` in base 2 and
  ``mul(lse2, LN2)`` in base e, with ``LG2 = lg2.approx.ftz.f32`` (opaque, like ``EX2``). It reads the ``(m, l)`` that
  normalise the output and has its own fingerprints (``lse_fingerprint``); the output terms do not contain it;
* attention sinks (``sink_merge``), the member of a call that passes ``sinks``: one more merge after the last chunk merge
  and before the epilogue, of the row state with a virtual key of head ``h`` whose logit is ``sinks[h]`` (natural-log units
  of the scaled score; ``sm_scale`` is not applied to it) and whose value is zero. In base-2 units, with ``LOG2E = fp32(log2 e)``:
  ``ms = mul(m, c)``, ``s2 = mul(sink, LOG2E)``, ``m' = fmax(ms, s2)``, ``nm = mul(m', -1)`` (exact),
  ``a = EX2(add(ms, nm))``, ``b = EX2(add(s2, nm))``, ``l' = fma(l, a, b)``; the epilogue is
  ``cvt_bf16(add(mul(O, mul(a, div(1, l'))), +0))`` and the LSE member ``add(m', LG2(l'))`` (``m'`` is already scaled).
  The step is taken once per row whatever the tiling, the unit kind and the kernel body; a row without sinks does not
  take it (the terms above, unchanged), so the sink member has its own fingerprints (``sink_fingerprint``).
  ``sink = -inf`` gives ``b = +0``, ``a = 1`` and the bits of the row without sinks; a finite sink so large that ``a``
  underflows gives ``+0`` outputs, as long as its base-2 value ``s2`` is finite; a ``+inf`` or NaN sink gives NaN
  (``b = EX2(add(+inf, -inf))``), like any non-finite operand. The step works on ``s2``, so a finite sink whose ``s2``
  overflows binary32 (a magnitude above about ``FLT_MAX / log2 e``, ``2.36e38``) is the sink of that infinity: NaN above,
  the bits of the row without sinks below.

Both members read a row's final ``(m, l)`` and nothing else, so they exist in both mask modes: under the non-causal class
every query row of a request folds the same keys ``[0, kv_len)``, its LSE depends on the request's key count, its head and its
own query alone, and the sink joins that key set. The members of the two classes are different terms with their own
fingerprints (``attn-bf16-d<D>-nc-lse-<base>-...``, ``attn-bf16-d<D>-nc-sink-...``, ``attn-bf16-d<D>-nc-sink-lse-<base>-...``).

Tile and chunk boundaries are absolute, ``T`` and ``C`` are functions of ``D`` only, so decode at
position ``p`` and prefill of the token at position ``p`` build the same term (R4), and nothing in
the term depends on the batch, the page table or the configuration (R2, R3).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

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
LN2 = to_f32(math.log(2.0))  # fp32(ln 2): the constant of the base-e LSE member
LOG2E = to_f32(
    1.0 / math.log(2.0)
)  # fp32(log2 e): a sink logit in the fold's base-2 units
LSE_BASES = ("log2", "ln")


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
    """One logical row: query position ``p`` (0-based, absolute in the KV sequence) and its visible keys ``[lo, L)``. Causal
    (the default): ``L = p + 1`` and ``lo = max(0, p - window_left)``. Non-causal (``causal=False``): ``L = kv_len`` (every key of
    the request) and ``lo = 0``; the position only names the row and a window is not a parameter of that class. ``row`` names the row
    (query token and head) in the leaf terms."""

    p: int
    window_left: int = -1
    row: int = 0  # the row's identity in the leaf terms (query token / head index)
    causal: bool = True
    kv_len: Optional[int] = (
        None  # the request's key count (non-causal: the visible range is [0, kv_len))
    )

    def __post_init__(self) -> None:
        if not self.causal:
            assert self.kv_len is not None and self.kv_len >= 1, (
                "a non-causal row needs the request's kv_len (at least one key)"
            )
            assert self.window_left < 0, (
                "window_left is not a parameter of the non-causal class"
            )

    @property
    def L(self) -> int:
        if not self.causal:
            assert self.kv_len is not None
            return self.kv_len
        return self.p + 1

    @property
    def lo(self) -> int:
        if not self.causal:
            return 0
        return 0 if self.window_left < 0 else max(0, self.p - self.window_left)

    def visible(self, k: int) -> bool:
        return self.lo <= k < self.L


def visible_keys(row: RowSpec, t: int, T: int = TILE_KEYS) -> List[int]:
    """The visible absolute keys of tile ``t`` for ``row`` (empty = the tile is absent)."""
    a, b = max(row.lo, t * T), min(row.L, (t + 1) * T)
    return list(range(a, b)) if a < b else []


def present_chunks(row: RowSpec, D: int) -> List[int]:
    C = chunk_len(D, row.L)
    return list(range(row.lo // C, -(-row.L // C))) if row.lo < row.L else []


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


@dataclass(frozen=True)
class SinkState:
    """A row's statistics after the sink merge: the maximum ``m`` over the keys and the sink in scaled base-2 units, the
    sum ``l`` relative to it, and ``a``, the rescale of the keys' accumulator to it."""

    m: Any
    l: Any
    a: Any


def sink_merge(fp, st: State, sink, c) -> SinkState:
    """The sink step on the row's final ``(m, l)``: the merge with a virtual key of logit ``sink`` (natural-log units,
    never divided by the scale) and zero value. ``m' = fmax(mul(m, c), mul(sink, LOG2E))``; the two exponentials are of the
    exact differences to ``m'`` (one of them ``EX2(+0) = 1``); ``l' = fma(l, a, b)``. Where ``mul(sink, LOG2E)`` is ``+inf``
    (a ``+inf`` sink, or a finite one whose base-2 value overflows binary32) ``b`` is ``EX2(add(+inf, -inf))``, a NaN,
    and so are ``l'``, the output and the LSE."""
    ms = fp.mul(st.m, c)
    s2 = fp.mul(sink, fp.const(LOG2E))
    m = fp.fmaxc(ms, s2)
    nm = fp.mul(m, fp.const(-1.0))
    a = fp.ex2(fp.add(ms, nm))
    b = fp.ex2(fp.add(s2, nm))
    return SinkState(m, fp.fma(st.l, a, b), a)


def sink_normaliser(fp, sk: SinkState):
    """``mul(a, div(1, l'))``: the row's normaliser under sinks, in the place of the epilogue's ``div(1, l)``."""
    return fp.mul(sk.a, fp.div(fp.const(1.0), sk.l))


def output_elem_sink(fp, st: State, sink, c, out_fmt: str = "bf16"):
    """The epilogue of a row with sinks: ``cvt(add(mul(O, mul(a, div(1, l'))), +0))``."""
    inv = sink_normaliser(fp, sink_merge(fp, st, sink, c))
    return fp.cvt_out(fp.add(fp.mul(st.o, inv), fp.const(0.0)), out_fmt)


def lg2(fp, x):
    """``LG2(x) = lg2.approx.ftz.f32``: ``fp.lg2`` on the backends that name it, the opaque ``op`` on the term algebra
    (printed ``(op:lg2 x)``)."""
    if hasattr(fp, "lg2"):
        return fp.lg2(x)
    return fp.op("lg2", x)


def lse_term(fp, st: State, c, base: str = "log2"):
    """The LSE member from the row's final statistics: ``lse2 = add(mul(m, c), LG2(l))`` (``m`` is in raw score units,
    ``c = fp32(sm_scale * log2 e)``) for ``base="log2"``, ``mul(lse2, LN2)`` for ``base="ln"``. At the init state
    ``(-inf, +0)`` both are ``-inf`` (``mul(-inf, c) = -inf`` for the class's ``c > 0``, ``LG2(+0) = -inf``)."""
    assert base in LSE_BASES, base
    lse2 = fp.add(fp.mul(st.m, c), lg2(fp, st.l))
    return lse2 if base == "log2" else fp.mul(lse2, fp.const(LN2))


def lse_term_sink(fp, sk: SinkState, base: str = "log2"):
    """The LSE member of a row with sinks, the sink included: ``lse2 = add(m', LG2(l'))`` on the merged statistics (``m'``
    is in scaled base-2 units already), ``mul(lse2, LN2)`` for ``base="ln"``."""
    assert base in LSE_BASES, base
    lse2 = fp.add(sk.m, lg2(fp, sk.l))
    return lse2 if base == "log2" else fp.mul(lse2, fp.const(LN2))


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
    """Symbolic view: ``output(row, d)`` is the term of one output element, ``lse(row, base)`` the LSE member;
    ``sinks=True`` gives the terms of a call with attention sinks (the row's sink is the leaf ``sink``)."""

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
        self.sink = self.alg.sym("sink")

    def reduction(self, row: RowSpec, d: int) -> Reduction:
        return attention_reduction(row, self.D, d, self.c, self.nc, self.T, self.C)

    def state(self, row: RowSpec, d: int) -> State:
        return self.reduction(row, d).canonical(self.alg)

    def output(self, row: RowSpec, d: int, sinks: bool = False) -> Term:
        st = self.state(row, d)
        if sinks:
            return output_elem_sink(self.alg, st, self.sink, self.c, self.out_fmt)
        return output_elem(self.alg, st, self.out_fmt)

    def lse(self, row: RowSpec, base: str = "log2", sinks: bool = False) -> Term:
        st = self.state(row, 0)
        if sinks:
            return lse_term_sink(
                self.alg, sink_merge(self.alg, st, self.sink, self.c), base
            )
        return lse_term(self.alg, st, self.c, base)


_FINGERPRINTS: Dict[tuple, str] = {}
# the recorded fingerprint of the published class (D = 128, no window) at C = 512 (with C = 1024 it was
# attn-bf16-d128-3384db6839e66672). ``tests/attention/test_batch_invariant_attention.py`` pins the three head dims' fingerprints.
FINGERPRINT_D128 = "attn-bf16-d128-027b0379515b82e2"
# the D = 64 and D = 256 classes: the same term with D as a parameter (D/16 MMA K-steps per
# score, a D-wide PV accumulator), C(D) = 512 for both; pinned by the same test
FINGERPRINT_D64 = "attn-bf16-d64-9a05233ec9351b0d"
FINGERPRINT_D256 = "attn-bf16-d256-b8cb59dab4eb6c0f"
FINGERPRINTS = {64: FINGERPRINT_D64, 128: FINGERPRINT_D128, 256: FINGERPRINT_D256}


def spec_fingerprint(D: int = 128, window_left: int = -1, causal: bool = True) -> str:
    """Identifier of the class: hashes the terms of output elements 0 and 1 of a row whose visible keys
    span one full chunk plus one tile (so the tile fold, the chunk merge with the init state, the
    masking of the last tile and the epilogue are all inside), together with ``D``, ``T``, ``C`` and the mask mode. The window is a
    per-call input of the causal class (like the KV position), not a class dimension: the public receipt
    (``flashinfer.batch_invariant.fingerprint``) is the digest of the no-window sample row; ``window_left`` here only selects
    another sample row. The mask mode IS a class dimension: the non-causal class (``causal=False``, name ``attn-bf16-d<D>-nc-*``)
    gives a token the visible set ``[0, kv_len)`` of its request where the causal class gives ``[0, p + 1)``, so the same token
    has a different term under the two classes; the sample row's term is the same dataflow over the same visible keys in both
    modes, and the mode enters the digest explicitly."""
    if not causal and window_left >= 0:
        raise ValueError("window_left is not a parameter of the non-causal class")
    key = (D, window_left, bool(causal))
    if key not in _FINGERPRINTS:
        from ...batch_invariant.fp_interface.term import fingerprint

        spec = AttentionSpec(D)
        L = spec.C + spec.T // 2 + 1
        row = (
            RowSpec(p=L - 1, window_left=window_left)
            if causal
            else RowSpec(p=0, causal=False, kv_len=L)
        )
        parts = [
            f"D={D}",
            f"T={spec.T}",
            f"C={spec.C}",
            f"win={window_left}",
            fingerprint(spec.output(row, 0)),
            fingerprint(spec.output(row, 1)),
        ]
        if not causal:
            parts.insert(3, "mask=noncausal")
        _FINGERPRINTS[key] = (
            f"attn-bf16-d{D}-"
            + ("" if causal else "nc-")
            + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
        )
    return _FINGERPRINTS[key]


_LSE_FINGERPRINTS: Dict[tuple, str] = {}
# the recorded fingerprints of the LSE member per (head dim, base); pinned by tests/attention/test_batch_invariant_attention.py
LSE_FINGERPRINTS = {
    (64, "log2"): "attn-bf16-d64-lse-log2-a8c548ec287e23bd",
    (64, "ln"): "attn-bf16-d64-lse-ln-d94f46b73185b273",
    (128, "log2"): "attn-bf16-d128-lse-log2-c9084818e797cacf",
    (128, "ln"): "attn-bf16-d128-lse-ln-8e4f2608bce73f6c",
    (256, "log2"): "attn-bf16-d256-lse-log2-ad95fa9071f86c5c",
    (256, "ln"): "attn-bf16-d256-lse-ln-54b51dc083581fce",
}


# the recorded fingerprints of the sink member (a call with ``sinks``): its output per head dim, its LSE per (head dim, base)
SINK_FINGERPRINTS = {
    64: "attn-bf16-d64-sink-a239bf124352c39c",
    128: "attn-bf16-d128-sink-08ebe33f73cbe897",
    256: "attn-bf16-d256-sink-97a726a039ac102d",
}
SINK_LSE_FINGERPRINTS = {
    (64, "log2"): "attn-bf16-d64-sink-lse-log2-ce5540cea8c9060a",
    (64, "ln"): "attn-bf16-d64-sink-lse-ln-411e595f84dacdf6",
    (128, "log2"): "attn-bf16-d128-sink-lse-log2-d06a4faff7897362",
    (128, "ln"): "attn-bf16-d128-sink-lse-ln-911ada9a2df50aef",
    (256, "log2"): "attn-bf16-d256-sink-lse-log2-fcdaa847d25ab16a",
    (256, "ln"): "attn-bf16-d256-sink-lse-ln-8bd8ede7bafb8ca0",
}
# the recorded fingerprints of the non-causal class and of its members: the mode precedes the member in the name
NC_FINGERPRINTS = {
    64: "attn-bf16-d64-nc-5c64fc377eb66759",
    128: "attn-bf16-d128-nc-421e018610a7f93c",
    256: "attn-bf16-d256-nc-603c24eeb8691c0a",
}
NC_LSE_FINGERPRINTS = {
    (64, "log2"): "attn-bf16-d64-nc-lse-log2-f287f9e1fd18384f",
    (64, "ln"): "attn-bf16-d64-nc-lse-ln-f1552997d639d2ce",
    (128, "log2"): "attn-bf16-d128-nc-lse-log2-66ad5536ca59cb76",
    (128, "ln"): "attn-bf16-d128-nc-lse-ln-8953d00478ded858",
    (256, "log2"): "attn-bf16-d256-nc-lse-log2-9bfafd1d8cbeb36d",
    (256, "ln"): "attn-bf16-d256-nc-lse-ln-67f2e98028fae0ef",
}
NC_SINK_FINGERPRINTS = {
    64: "attn-bf16-d64-nc-sink-e2b883b827085097",
    128: "attn-bf16-d128-nc-sink-f380a0e936c103f3",
    256: "attn-bf16-d256-nc-sink-f0a0bf2c7f86e1b7",
}
NC_SINK_LSE_FINGERPRINTS = {
    (64, "log2"): "attn-bf16-d64-nc-sink-lse-log2-e973b28b732c6827",
    (64, "ln"): "attn-bf16-d64-nc-sink-lse-ln-2a3b2f4f2e221fe6",
    (128, "log2"): "attn-bf16-d128-nc-sink-lse-log2-760246bf1153b0c1",
    (128, "ln"): "attn-bf16-d128-nc-sink-lse-ln-43249ae085aa93a0",
    (256, "log2"): "attn-bf16-d256-nc-sink-lse-log2-edc47c7137f23613",
    (256, "ln"): "attn-bf16-d256-nc-sink-lse-ln-5e999137d1b8c7e3",
}


def _sample_row(spec: "AttentionSpec", causal: bool) -> RowSpec:
    """``spec_fingerprint``'s no-window sample row (one full chunk plus one tile of visible keys) under a mask mode."""
    L = spec.C + spec.T // 2 + 1
    return RowSpec(p=L - 1) if causal else RowSpec(p=0, causal=False, kv_len=L)


def lse_fingerprint(
    D: int = 128, base: str = "log2", sinks: bool = False, *, causal: bool = True
) -> str:
    """Identifier of the LSE member of the class at head dim ``D`` in ``base`` (``"log2"`` | ``"ln"``): hashes the LSE term of
    ``spec_fingerprint``'s sample row together with ``D``, ``T``, ``C``, the mask mode and the base. A receipt of its own: the
    class fingerprints (``spec_fingerprint``) hash the output terms alone and do not move with it. ``sinks``: the LSE of the sink
    member (the sink included), ``attn-bf16-d<D>-sink-lse-<base>-...``. ``causal=False``: the member of the non-causal class
    (``attn-bf16-d<D>-nc-lse-<base>-...``, ``attn-bf16-d<D>-nc-sink-lse-<base>-...``; the class dimension precedes the
    members in the name)."""
    if base not in LSE_BASES:
        raise ValueError(f"lse_base must be one of {LSE_BASES}, got {base!r}")
    key = (D, base, bool(sinks), bool(causal))
    if key not in _LSE_FINGERPRINTS:
        from ...batch_invariant.fp_interface.term import fingerprint

        spec = AttentionSpec(D)
        row = _sample_row(spec, bool(causal))
        parts = [
            f"D={D}",
            f"T={spec.T}",
            f"C={spec.C}",
            f"base={base}",
            fingerprint(spec.lse(row, base, sinks=bool(sinks))),
        ]
        if sinks:
            parts.insert(3, "sinks")
        if not causal:
            parts.insert(3, "mask=noncausal")
        _LSE_FINGERPRINTS[key] = (
            f"attn-bf16-d{D}-{'' if causal else 'nc-'}{'sink-' if sinks else ''}lse-{base}-"
            + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
        )
    return _LSE_FINGERPRINTS[key]


def sink_fingerprint(D: int = 128, *, causal: bool = True) -> str:
    """Identifier of the sink member of the class at head dim ``D`` (a call with ``sinks``): hashes the output terms of
    ``spec_fingerprint``'s sample row with the sink step, together with ``D``, ``T``, ``C`` and the mask mode. The sink values
    are per-call inputs, like the scale; the fingerprints of the class without sinks do not move with it. ``causal=False``:
    the member of the non-causal class, ``attn-bf16-d<D>-nc-sink-...``."""
    key = (D, "sink", bool(causal))
    if key not in _FINGERPRINTS:
        from ...batch_invariant.fp_interface.term import fingerprint

        spec = AttentionSpec(D)
        row = _sample_row(spec, bool(causal))
        parts = [
            f"D={D}",
            f"T={spec.T}",
            f"C={spec.C}",
            "sinks",
            fingerprint(spec.output(row, 0, sinks=True)),
            fingerprint(spec.output(row, 1, sinks=True)),
        ]
        if not causal:
            parts.insert(3, "mask=noncausal")
        _FINGERPRINTS[key] = (
            f"attn-bf16-d{D}-{'' if causal else 'nc-'}sink-"
            + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
        )
    return _FINGERPRINTS[key]
