# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.cute.arch.nvvm_wrappers import inline_ptx

from .constants import MASK_CAUSAL, MASK_NONE, MASK_PADDED, MASK_SWA  # noqa: F401

_NEG_INF_BITS = -3.4028235e38


# ---------------------------------------------------------------------------
# Per-cell mask: one keep-word per 32 columns, register-to-predicate select
# ---------------------------------------------------------------------------
#
# `apply_mask_chunk` below masks a register chunk of S in the BIT-WORD form.  The
# information content of a padded / causal / sliding-window mask is one or two
# band EDGES per row, so that is where the instructions go:
#
#   1. per 32-column word, build the KEEP mask from the band edges with two
#      saturating shifts (`shr.u32` for the upper edge, `shl.b32` for the lower;
#      PTX clamps a shift amount >= 32 to a zero result, which is exactly the
#      "edge is outside this word" case) -- ~7 integer ops per word, per lane;
#   2. per cell, test one bit of the word and feed it to an `arith.select`.
#      ptxas turns 32 consecutive bit tests into `R2P` (register -> 7 predicates)
#      + one `FSEL` per cell, so the per-cell cost is 1.4-1.6 instructions
#      regardless of how many mask terms are on (sm_107a / sm_100a micro-kernel
#      listings, 2026-09-22; the same idiom the block-sparse-attention SM90
#      backward uses, `predicate_bitmask_below`).
#
# The form it replaced (shipped until PR #1192 / #1197 behind a per-kernel
# `MASK_FORM` constant, collapsed since) compared EVERY cell against every active
# bound and ORed the terms: per cell one IADD (kv_col_base + i, never folded into
# the compare), one ISETP per term and one FSEL per term (the i1 OR lowers to
# nested selects), so a causal+SWA tile cost 5 instructions per cell and a padded
# causal+SWA one 7 -- 51-72 % of a masked softmax tile's instructions, serialized
# in front of the exp burst (sm_107a listings, 2026-09-22: a masked KV tile ran
# 805-1602 instructions per lane against 358-451 for a dense one; one build had
# spilled its 128 live i1 values into GPR bits).  Both forms mask the same set
# with the same sentinel through the same `arith.select`, so O / LSE were bitwise
# identical across the switch; `test_tile_dsl_mask_bits.py` keeps a host emulation
# of the per-cell formula as the reference this op is pinned against, and the
# sm_107a / sm_100a SASS pins hold the lowering (R2P > 0, ISETP within the
# measured ceiling).
#
# Semantics contract:
#   - a masked cell becomes `mask_value` (default: the finite fp32-min sentinel;
#     pass float("-inf") for the true -inf form); an unmasked cell is passed
#     through untouched (NaN payloads included);
#   - bounds are PER LANE (one lane = one q row): `lo`/`hi` are Int32 values
#     derived from that lane's q row, `kv_col_base` is the chunk's first
#     absolute kv column; a fully-masked row yields `mask_value` in every cell,
#     so the caller's `row_max_for_exp2` guard applies exactly as before;
#   - a `tcgen05.ld 32x32b` chunk is contiguous along columns per lane (register
#     k = column k), so NO column remap is needed here -- unlike the WGMMA
#     accumulator layout the BSA kernel remaps with `sm90_col_to_predicate_idx`.

MASK_WORD_COLS = 32  # columns per keep-word = bits per register

# The band arithmetic (`lo - kv_col_base`, `hi - kv_col_base`, the per-word shift count) is Int32.
# With every absolute row / column index below 2**28 (a TMA coordinate keeps S far under that) a
# window bound below this limit cannot wrap it; a bound at or past it can, and a wrapped `lo` makes
# the op mask EVERYTHING where the per-cell formula masks nothing.  Enforced at trace time on the
# Python-int bounds (no instruction); pinned by `test_bits_form_domain_guard`.
MASK_BOUND_LIMIT = 1 << 30


def keep_below_word(hi_rel, s: int):
    """Keep-word of 32-column word ``s``: bit ``i`` is set iff column ``32 s + i < hi_rel``.

    ``hi_rel`` is the exclusive upper bound RELATIVE to the chunk base (an Int32, per lane;
    any value, including negative or past the chunk).  Spelled in PTX because the
    saturating shift is the whole point: LLVM treats a shift by >= 32 as poison, PTX
    ``shr.u32`` clamps it and yields 0 (= every column of this word is at or past ``hi``).
    The ``max.s32`` clamps the other side (``hi`` past the word -> shift 0 -> all kept)."""
    n_masked = cutlass.Int32(MASK_WORD_COLS * (s + 1)) - hi_rel
    return inline_ptx(
        "{ .reg .b32 n, ones; mov.b32 ones, 0xFFFFFFFF; max.s32 n, {$r0}, 0; shr.u32 {$w0}, ones, n; }",
        write_only_types=[cutlass.Uint32],
        read_only_args=[n_masked],
    )


def keep_from_word(lo_rel, s: int):
    """Keep-word of 32-column word ``s``: bit ``i`` is set iff column ``32 s + i >= lo_rel``.

    The lower-edge twin of :func:`keep_below_word`: ``shl.b32`` of all-ones by the number
    of columns below ``lo`` in this word, clamped at 0 (``lo`` left of the word -> all kept)
    and saturating at >= 32 (``lo`` right of the word -> 0, all masked)."""
    n_masked = lo_rel - cutlass.Int32(MASK_WORD_COLS * s)
    return inline_ptx(
        "{ .reg .b32 n, ones; mov.b32 ones, 0xFFFFFFFF; max.s32 n, {$r0}, 0; shl.b32 {$w0}, ones, n; }",
        write_only_types=[cutlass.Uint32],
        read_only_args=[n_masked],
    )


def band_mask_words(lo, hi, kv_col_base, n_cols: int):
    """Per-lane keep-words for the ``n_cols`` columns starting at absolute kv column
    ``kv_col_base``: bit ``i`` of word ``s`` is set iff column ``c = kv_col_base + 32 s + i``
    satisfies ``keep(c) = (lo is None or c >= lo) & (hi is None or c < hi)``.

    ``lo`` / ``hi`` are per-lane Int32 absolute column bounds (``hi`` exclusive), or ``None``
    for a side no mask term defines -- that side folds out at trace time (one shift per word
    for a one-sided mask, two for a band), never an INT_MIN/INT_MAX sentinel, so ``lo - base``
    cannot wrap as long as the caller keeps every bound within ``MASK_BOUND_LIMIT`` of the chunk
    (:func:`apply_mask_chunk` refuses a wider window).  Returns a tuple of
    ``ceil(n_cols / 32)`` ``Uint32`` words.  Trace-time helper (plain Python over traced
    values)."""
    if lo is None and hi is None:
        raise ValueError("band_mask_words: at least one of lo / hi must be given (a mask with no edge is no mask)")
    n_words = (n_cols + MASK_WORD_COLS - 1) // MASK_WORD_COLS
    hi_rel = None if hi is None else hi - kv_col_base
    lo_rel = None if lo is None else lo - kv_col_base
    words = []
    for s in range(n_words):
        word = None
        if hi_rel is not None:
            word = keep_below_word(hi_rel, s)
        if lo_rel is not None:
            above = keep_from_word(lo_rel, s)
            word = above if word is None else (word & above)
        words.append(word)
    return tuple(words)


def apply_mask_words(reg_S, words, mask_value: float = _NEG_INF_BITS, n_cols: int = None):
    """``reg_S[c] if keep-bit(c) else mask_value`` over an ``n_cols``-wide register chunk.

    ``words`` is the tuple :func:`band_mask_words` returns (bit ``i`` of word ``s`` = column
    ``32 s + i`` is KEPT).  The inner loop is a Python ``range`` over a compile-time bit
    index, which is what lets ptxas see 32 consecutive single-bit tests of one register and
    emit ``R2P`` + one ``FSEL`` per cell.  One ``arith.select`` per cell, so a kept cell passes
    through bit-exactly (NaN payloads included) and a masked one is exactly ``mask_value``,
    for either sentinel -- the same select the per-cell compare form used, only the predicate
    derivation differs."""
    if n_cols is None:
        n_cols = MASK_WORD_COLS * len(words)
    if len(words) * MASK_WORD_COLS < n_cols:
        raise ValueError(f"apply_mask_words: {len(words)} word(s) cover {len(words) * MASK_WORD_COLS} columns, chunk has {n_cols}")
    neg_inf = cutlass.Float32(mask_value)
    elems = []
    for s, word in enumerate(words):
        for i in range(MASK_WORD_COLS):
            c = MASK_WORD_COLS * s + i
            if c >= n_cols:
                break
            keep = cutlass.Boolean(word & cutlass.Uint32(1 << i))
            elems.append(cutlass.Float32(arith.select(keep.ir_value(), reg_S[c].ir_value(), neg_inf.ir_value())))
    return cutlass.Vector.from_elements(tuple(elems), cutlass.Float32)


def apply_mask_chunk(
    reg_S,
    q_abs,
    kv_col_base,
    seq_kv_len,
    window_left: int,
    mask_flags: int,
    N: int = 64,
    bottom_right: int = 0,
    causal_diag=None,
    mask_value: float = _NEG_INF_BITS,
    window_right: int = 0,
    keep_words=None,
):
    """Mask the ``N``-wide fp32 register chunk ``reg_S`` (one lane = one q row ``q_abs``, columns
    ``kv_col_base .. kv_col_base + N``) for the compile-time ``mask_flags`` set, in the bit-word form.

    The three terms map onto one band ``[lo, hi)`` per lane:

    - CAUSAL masks ``kv > q_caus_lim``  ->  ``hi = q_caus_lim + 1`` with
      ``q_caus_lim = q_abs (+ causal_diag under bottom_right) (+ window_right)``.  ``window_right``
      is the compile-time diagonal-band right bound (cuDNN diagonal_band_right_bound): kv columns
      up to ``q + window_right`` (plus the bottom-right diagonal offset) stay unmasked; 0 = plain causal;
    - PADDED masks ``kv >= seq_kv_len``  ->  ``hi = min(hi, seq_kv_len)``;
    - SWA masks ``kv < q_minus_w``  ->  ``lo = q_minus_w = swa_base - window_left`` with the same
      bottom-right anchor (``swa_base = q_abs + causal_diag`` under bottom_right, else ``q_abs``) --
      the whole band shifts with the diagonal.

    A side no term defines is ``None`` and folds out.  ``mask_value`` is what a masked score becomes:
    the default is the legacy finite sentinel; the f16 prefill kernels pass ``float("-inf")`` so a
    fully-masked row's max stays -inf under any scale and the canonical ``max == -inf -> substitute 0``
    guard (``row_max_for_exp2``) applies.  ``MASK_NONE`` without ``keep_words`` returns ``reg_S`` itself.

    ``keep_words`` (optional): a tuple of ``ceil(N / 32)`` per-32-column KEEP words from a DATA mask -- the
    per-key boolean ``kv_mask`` assembled by :func:`kv_mask_keep_words` -- ANDed into the band's words (or
    used alone under ``MASK_NONE``), so a data-masked key costs the same one select per cell as a banded one
    and the whole mask stays ONE ``R2P`` + ``FSEL`` pass; a keep word is warp-uniform (every lane of the warp
    holds the same word), the band words are per lane.

    Domain: a compile-time ``window_left`` / ``window_right`` at or past ``MASK_BOUND_LIMIT`` raises
    at trace time (see the constant): the band arithmetic is Int32 and a wrapped ``lo`` would mask the
    whole chunk.  Callers that invent a "no sequence reaches this" band (the dense per-tensor fp8 d192
    lowering) must stay below it -- `config_sm100.canonicalize_d192_lowering` uses ``MASK_BOUND_LIMIT - 1``."""
    n_words = (N + MASK_WORD_COLS - 1) // MASK_WORD_COLS
    if keep_words is not None and len(keep_words) != n_words:
        raise ValueError(f"apply_mask_chunk: keep_words has {len(keep_words)} word(s); an N={N} chunk needs {n_words}")
    if cutlass.const_expr(mask_flags == MASK_NONE):
        if keep_words is None:
            return reg_S
        return apply_mask_words(reg_S, tuple(keep_words), mask_value=mask_value, n_cols=N)

    if (mask_flags & MASK_SWA) and isinstance(window_left, int) and window_left >= MASK_BOUND_LIMIT:
        raise ValueError(f"apply_mask_chunk: window_left must be < {MASK_BOUND_LIMIT} (got {window_left}); the Int32 band arithmetic would wrap")
    if (mask_flags & MASK_CAUSAL) and isinstance(window_right, int) and window_right >= MASK_BOUND_LIMIT:
        raise ValueError(f"apply_mask_chunk: window_right must be < {MASK_BOUND_LIMIT} (got {window_right}); the Int32 band arithmetic would wrap")

    lo = None
    hi = None
    if mask_flags & MASK_SWA:
        swa_base = (q_abs + causal_diag) if bottom_right else q_abs
        lo = swa_base - cutlass.Int32(window_left)
    if mask_flags & MASK_CAUSAL:
        q_caus_lim = (q_abs + causal_diag) if bottom_right else q_abs
        if window_right != 0:
            q_caus_lim = q_caus_lim + cutlass.Int32(window_right)
        hi = q_caus_lim + cutlass.Int32(1)
    if mask_flags & MASK_PADDED:
        hi = seq_kv_len if hi is None else cute.math.min(hi, seq_kv_len)

    words = band_mask_words(lo, hi, kv_col_base, N)
    if keep_words is not None:
        words = tuple(w & k for w, k in zip(words, keep_words))
    return apply_mask_words(reg_S, words, mask_value=mask_value, n_cols=N)


def kv_mask_keep_words(kv_mask_tensor, batch_idx, kv_col_base, seq_kv_len, n_cols: int):
    """The per-key boolean mask's KEEP words for the ``n_cols`` kv columns starting at ``kv_col_base``: one
    warp-uniform ``Uint32`` per 32 columns, bit ``i`` of word ``s`` set iff key ``kv_col_base + 32 s + i`` is
    inside the sequence (``< seq_kv_len``) AND its mask byte is non-zero.

    Trace-time helper for the softmax warps (plain Python over traced values): lane ``i`` of the warp loads
    the byte of key ``32 s + i`` from ``kv_mask_tensor[batch_idx, key]`` (a ``(B, S_kv)`` Int8 view over the
    graph's BOOLEAN tensor; the key index is clamped into ``[0, seq_kv_len)`` so the load never leaves the
    row and an out-of-range key votes 0) and a ``vote.ballot`` assembles the word -- 4 byte loads + 4 ballots
    per 128-column tile per warp, no SMEM staging, nothing packed on the host.  Every lane of the warp holds
    the same words, which is what lets them AND into :func:`apply_mask_chunk`'s per-lane band words."""
    from cutlass.experimental.primitives import VoteSync, vote_sync

    n_words = (n_cols + MASK_WORD_COLS - 1) // MASK_WORD_COLS
    lane = cute.arch.thread_idx()[0] & cutlass.Int32(MASK_WORD_COLS - 1)
    last = cute.math.max(seq_kv_len - cutlass.Int32(1), cutlass.Int32(0))
    mask_arr = cutlass.make_array_view(kv_mask_tensor)
    words = []
    for s in range(n_words):
        col = kv_col_base + cutlass.Int32(MASK_WORD_COLS * s) + lane
        in_range = col < seq_kv_len
        col_c = cute.math.min(col, last)
        byte = mask_arr[batch_idx, col_c]
        keep = in_range & (byte != cutlass.Int8(0))
        words.append(cutlass.Uint32(vote_sync(0xFFFFFFFF, keep, VoteSync.BALLOT)))
    return tuple(words)


# ---------------------------------------------------------------------------
# Tile-level mask bounds: which kv TILES a q tile has to visit at all.
# ---------------------------------------------------------------------------
#
# `apply_mask_chunk` above is the per-CELL mask. This is the per-TILE one, and
# it is where the work actually gets saved: under a causal band a q tile only
# intersects the kv tiles up to its own diagonal, so the whole upper triangle of
# tiles is never issued -- roughly half the MMAs at causal. The returned range
# also splits into a middle sub-range `[unmasked_lo, unmasked_hi)` where no cell
# can be masked, so only the diagonal (and padding-tail) tiles pay for
# `apply_mask_chunk` at all.
#
# Lives here rather than beside one pass's kernels because both the forward and
# the backward need exactly this arithmetic, and it is pure: it reads mask
# parameters and tile geometry, nothing pass-specific.


class KvLoopBounds(NamedTuple):
    left: object
    unmasked_lo: object
    unmasked_hi: object
    right: object


def _div_up(a, b):
    return (a + cutlass.Int32(b - 1)) // cutlass.Int32(b)


def swa_kv_lo_tile(anchor_row, window_left: int, tile_n: int):
    """First KV tile a left window of ``window_left`` keeps for ``anchor_row``
    (the q row plus the bottom-right diagonal); 0 while the window still
    reaches kv 0.  The forward's KV-loop lower bound.
    """
    cond = anchor_row > cutlass.Int32(window_left)
    delta = anchor_row - cutlass.Int32(window_left)
    return cutlass.Int32(
        arith.select(
            cond.ir_value(),
            (delta // cutlass.Int32(tile_n)).ir_value(),
            cutlass.Int32(0).ir_value(),
        )
    )


def compute_kv_loop_bounds(
    q_row_coord,
    seqlen_q,
    seq_kv_len,
    window_left: int,
    mask_flags: int,
    tile_n: int,
    cga_tile_m: int,
    bottom_right: bool = False,
    window_right: int = 0,
) -> KvLoopBounds:
    # window_right: compile-time diagonal-band right bound (cuDNN
    # diagonal_band_right_bound) — the causal upper limit is widened by
    # window_right columns. 0 = plain causal; folds out entirely.
    left = cutlass.Int32(0)
    right = _div_up(seq_kv_len, tile_n)

    if cutlass.const_expr(bottom_right):
        causal_diag = seq_kv_len - seqlen_q
    else:
        causal_diag = cutlass.Int32(0)

    if cutlass.const_expr(mask_flags & MASK_CAUSAL):
        kv_hi_caus = _div_up(q_row_coord + cutlass.Int32(cga_tile_m + window_right) + causal_diag, tile_n)
        right = cute.math.min(right, kv_hi_caus)

    if cutlass.const_expr(mask_flags & MASK_SWA):
        # The whole band shifts with the diagonal: under BOTTOM_RIGHT the SWA
        # lower bound is q + (S_kv - S_q) - W, same anchor the causal upper
        # bound uses (causal_diag folds to 0 for top-left).
        left = cute.math.max(left, swa_kv_lo_tile(q_row_coord + causal_diag, window_left, tile_n))

    unmasked_hi = right
    if cutlass.const_expr(mask_flags & MASK_PADDED):
        unaligned = (seq_kv_len % cutlass.Int32(tile_n)) != cutlass.Int32(0)
        lo_pad = cutlass.Int32(
            arith.select(
                unaligned.ir_value(),
                (right - cutlass.Int32(1)).ir_value(),
                right.ir_value(),
            )
        )
        unmasked_hi = cute.math.min(unmasked_hi, lo_pad)
    if cutlass.const_expr(mask_flags & MASK_CAUSAL):
        lo_caus = (q_row_coord + cutlass.Int32(window_right) + causal_diag) // cutlass.Int32(tile_n)
        unmasked_hi = cute.math.min(unmasked_hi, lo_caus)
    unmasked_hi = cute.math.max(unmasked_hi, left)

    unmasked_lo = left
    if cutlass.const_expr(mask_flags & MASK_SWA):
        anchor = q_row_coord + causal_diag + cutlass.Int32(cga_tile_m - 1 - window_left)
        swa_unmasked_lo = _div_up(anchor, tile_n)
        cond = anchor > cutlass.Int32(0)
        swa_unmasked_lo = cutlass.Int32(
            arith.select(
                cond.ir_value(),
                swa_unmasked_lo.ir_value(),
                cutlass.Int32(0).ir_value(),
            )
        )
        unmasked_lo = cute.math.max(unmasked_lo, swa_unmasked_lo)

    unmasked_lo = cute.math.min(unmasked_lo, unmasked_hi)

    return KvLoopBounds(
        left=left,
        unmasked_lo=unmasked_lo,
        unmasked_hi=unmasked_hi,
        right=right,
    )


# The KV-major dual: a backward CTA owns one KV tile and loops over Q tiles, so
# it needs the range of q tiles that attend its kv rows.  Causal trims from
# BELOW (q rows above the diagonal never see this kv tile), the sliding window
# trims from ABOVE (rows more than window_left past the tile never see it).
# Same inputs and conventions as compute_kv_loop_bounds; the causal diagonal is
# derived the same way, and the right-band widening may be a runtime value
# (the backward keeps it dynamic).


class QLoopBounds(NamedTuple):
    lo: object  # first q tile attending the kv tile (inclusive)
    hi: object  # one past the last; hi <= lo means the kv tile has no work


def compute_q_loop_bounds(
    kv_row_coord,
    seqlen_q,
    seq_kv_len,
    n_q_tiles,
    window_left: int,
    mask_flags: int,
    tile_q: int,
    tile_kv: int,
    bottom_right: bool = False,
    window_right=0,
) -> QLoopBounds:
    if cutlass.const_expr(bottom_right):
        causal_diag = seq_kv_len - seqlen_q
    else:
        causal_diag = cutlass.Int32(0)

    lo = cutlass.Int32(0)
    if cutlass.const_expr(mask_flags & MASK_CAUSAL):
        # kv rows [kv_row_coord, +tile_kv) are attended by q whenever
        # kv <= q + causal_diag + window_right: the first such q is the tile's
        # first row minus the diagonal minus the band widening (the widening
        # is subtracted so the straddling rows of the previous tile keep their
        # dQ and this tile keeps their dK/dV contribution).
        q_lo_abs = kv_row_coord - causal_diag - window_right
        lo = cute.math.max(q_lo_abs // cutlass.Int32(tile_q), cutlass.Int32(0))

    hi = n_q_tiles
    if cutlass.const_expr(mask_flags & MASK_SWA):
        # The window keeps kv >= q + causal_diag - window_left, so no q at or
        # past kv_row_coord + tile_kv + window_left - causal_diag attends the tile.
        q_hi_abs = kv_row_coord + cutlass.Int32(tile_kv + window_left) - causal_diag
        q_hi_abs = cute.math.max(q_hi_abs, cutlass.Int32(0))
        hi = cute.math.min(_div_up(q_hi_abs, tile_q), n_q_tiles)

    return QLoopBounds(lo=lo, hi=hi)


@cute.jit
def q_tile_kv_hi(
    q_tile: cutlass.Int32,
    seqlen_q: cutlass.Int32,
    seq_kv_len: cutlass.Int32,
    kv_last: cutlass.Int32,
    mask_flags: cutlass.Constexpr[int],
    tile_q: cutlass.Constexpr[int],
    tile_kv: cutlass.Constexpr[int],
    bottom_right: cutlass.Constexpr[bool],
    window_right: cutlass.Int32,
) -> cutlass.Int32:
    """Highest kv tile whose compute_q_loop_bounds range contains ``q_tile``,
    capped at ``kv_last`` (the last kv tile that runs).

    Inverts the causal lower bound: kv tile k visits q tile j iff
    ``(k * tile_kv - diag - window_right) // tile_q <= j``.  The window only
    removes LOW kv tiles, so it does not move this bound.  The backward's
    deterministic dQ relay counts turns down from it.  Meaningless for a q tile
    no kv tile visits (nobody relays on it).  ``@cute.jit`` because the relay
    calls it inside the kernel's dynamic q loop.
    """
    hi = kv_last
    if cutlass.const_expr(mask_flags & MASK_CAUSAL):
        if cutlass.const_expr(bottom_right):
            causal_diag = seq_kv_len - seqlen_q
        else:
            causal_diag = cutlass.Int32(0)
        q_end = (q_tile + cutlass.Int32(1)) * cutlass.Int32(tile_q)
        hi = cute.math.min(hi, (q_end + causal_diag + window_right - cutlass.Int32(1)) // cutlass.Int32(tile_kv))
    return hi


# ---------------------------------------------------------------------------
# Membership mask: a list-gathered KV tile has no band; each cell is either one
# of THIS row's keys or not, and the decision is a bit the caller pre-computed.
# ---------------------------------------------------------------------------


def _membership_bit(word, i: int):
    """The single-bit test mask for bit ``i`` in ``word``'s own integer type: a ``Uint32`` word takes ``1 << i`` as is; an
    ``Int32`` word (the ``ldg_int32x4`` / ``ld.global.nc.v4.s32`` membership words) takes the SIGNED encoding, so bit 31 is
    ``-2**31`` and never an out-of-range i32 literal."""
    if isinstance(word, cutlass.Uint32):
        return cutlass.Uint32(1 << i)
    return cutlass.Int32((1 << i) - (1 << 32) if i == 31 else (1 << i))


def apply_membership_words(reg_S, words, *, n_cols: int = None, mask_value: float = float("-inf")):
    """``reg_S[c] if keep-bit(c) else mask_value`` over an ``n_cols``-wide register chunk whose mask is a MEMBERSHIP bit
    table: ``words`` is a tuple of 32-bit words, bit ``i`` of word ``s`` = column ``32 s + i`` is one of THIS row's keys
    (kept); a zero bit is "not one of this row's keys" (masked).  The bit-word twin of :func:`apply_mask_words` for a
    list-gathered KV tile, whose per-cell decision is a bit the caller pre-computed rather than a band edge -- the words
    arrive from GMEM already in mask-word form, so there is no ``band_mask_words`` step: ONE bit test + ONE
    ``arith.select`` per cell, spelled as ``apply_mask_words`` spells it (a Python ``range`` over a compile-time bit index
    of one register) so ptxas emits ``R2P`` (register -> 7 predicates) + one ``FSEL`` per cell.

    Words may be ``Int32`` (``ldg_int32x4`` slot words) or ``Uint32``; the bit-31 test uses the word's own signed /
    unsigned encoding (:func:`_membership_bit`).  The default ``mask_value`` is a TRUE ``-inf`` (not the finite
    ``_NEG_INF_BITS`` sentinel): a fully-masked chunk's max is then ``-inf`` under any scale and the consumer's
    ``max == -inf -> substitute`` guard applies (sdpa-invariants.md section 2); a kept cell passes through bit-exactly
    (NaN payloads included).  Truth table identical to the per-cell spelling ``((word >> i) & 1) == 0 -> mask_value`` --
    and on sm_107a ptxas ALREADY lowered that spelling to R2P + FSEL (the pre-upstream d512 index-list kernel read 16 R2P,
    no per-cell ISETP), so this is the shared idiom, not a SASS change: O / LSE bitwise either way.  Trace-time helper
    (plain Python over traced values)."""
    if n_cols is None:
        n_cols = MASK_WORD_COLS * len(words)
    if len(words) * MASK_WORD_COLS < n_cols:
        raise ValueError(f"apply_membership_words: {len(words)} word(s) cover {len(words) * MASK_WORD_COLS} columns, chunk has {n_cols}")
    neg_inf = cutlass.Float32(mask_value)
    elems = []
    for s, word in enumerate(words):
        for i in range(MASK_WORD_COLS):
            c = MASK_WORD_COLS * s + i
            if c >= n_cols:
                break
            keep = cutlass.Boolean(word & _membership_bit(word, i))
            elems.append(cutlass.Float32(arith.select(keep.ir_value(), reg_S[c].ir_value(), neg_inf.ir_value())))
    return cutlass.Vector.from_elements(tuple(elems), cutlass.Float32)


def apply_membership_fragment(reg_S, word, shift, bit_of_cell, *, mask_value: float = float("-inf")):
    """``reg_S[c] if bit bit_of_cell[c] of (word >> shift) else mask_value`` over a lane's FRAGMENT of one 32-column membership WORD.

    The ``mma.sync`` m16n8 C layout hands a lane the column pair ``2 t, 2 t + 1`` of every n8 block (rows ``g`` and ``g + 8``), so its cells of a
    32-key tile are the word's bits ``8 nb + 2 t + e`` -- not the contiguous column order :func:`apply_membership_words` assumes.  ``shift`` is
    each lane's RUNTIME part of the bit index (``8 nb + 2 t``, or ``2 t`` with the n-block folded into ``bit_of_cell``), ``bit_of_cell[c]`` the
    compile-time remainder (``e``, or ``8 nb + e``).  Same op as the words form: ONE bit test + ONE ``arith.select`` per cell, the same
    ``-inf`` default (a fully-masked tile's max is ``-inf`` under any scale), the same truth table, the same R2P + FSEL lowering; the word
    itself is the caller's (the SM107 lean sparse decode tile builds ONE per 32-key sub-tile from the gathered block ids).  Trace-time helper
    (plain Python over traced values); returns a Python list of ``Float32`` (constexpr-indexed) so the caller keeps its fragment slots."""
    bits = word >> shift
    neg_inf = cutlass.Float32(mask_value)
    out = []
    for c, b in enumerate(bit_of_cell):
        keep = cutlass.Boolean(bits & _membership_bit(bits, int(b)))
        out.append(cutlass.Float32(arith.select(keep.ir_value(), reg_S[c].ir_value(), neg_inf.ir_value())))
    return out


def apply_membership_chunk(reg_S, word_lo, word_hi, *, n: int = 64, mask_value: float = float("-inf")):
    """Per-CELL membership mask for a ``n``-wide chunk of scores: cell ``i`` KEEPS ``reg_S[i]``
    iff bit ``i`` of the 64-bit pair ``(word_lo, word_hi)`` is set, else becomes ``mask_value``.

    Bit ``i`` lives in ``word_lo`` for ``i < 32`` and in ``word_hi`` (bit ``i - 32``) above;
    ``word_hi`` is unused at ``n <= 32``.  A ZERO bit says "this gathered column is none of this
    row's keys" (under a multi-token UNION tile a duplicate-free row sees every copy but its own as
    zero bits; a ``-1`` / sanitised id is a zero-filled row AND a zero bit, so no real logit of 0
    survives).

    The two-word convenience form of :func:`apply_membership_words` (same select, same ``-inf``
    default, same truth table); a kernel with wider chunks calls the words form directly.
    """
    if cutlass.const_expr(not 1 <= n <= 64):
        raise ValueError(f"apply_membership_chunk: n must be in 1..64 (two 32-bit words), got {n}")
    words = (word_lo,) if n <= MASK_WORD_COLS else (word_lo, word_hi)
    return apply_membership_words(reg_S, words, n_cols=n, mask_value=mask_value)
