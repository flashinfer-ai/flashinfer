# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The selection passes of ``support.row.v1`` (``support_spec.py``) as the truncated draw kernel
(``draw_support_kernel.py``) runs them: the row class's passes, the per-row thresholds by radix selection over the
weights' bit patterns, and the kept set's mass and count. The class has no kernel or entry of its own: this module holds
the device code the composition's kernel imports.

Structure: the CTA's share of the row is staged in shared memory once -- by the loader's cp.async path when the config
stages, otherwise through its predicated register loads (``draw_kernel._DrawLoader.stage_from_regs``) -- and every pass
reads it from there: the trees through the ``lse.row`` loader and engine (the same plans, the same chains), the radix
passes as compact dynamic loops over the tile's 32-bit words (eight bf16 or four fp32 elements per thread per iteration,
lane-contiguous words). Each radix pass adds one to a 256-bin count histogram per candidate element (``red.shared.add.u32
[bin], 1``: the hardware aggregates a warp's same-address increments; a non-candidate lane increments a dummy slot, so
there is no branch) and, for top-p, the element's fixed-point mass ``trunc(mul(w, div(2^44, S)))`` split into
32-bit-safe chunks (``red.shared.add.u32`` of each chunk into one of ``RM`` lane-spread replicas of the bin; the only
shared atomics native on these parts are 32-bit -- 64-bit integer and fp64 adds are CAS loops). Integer sums in any
order are exact. Every warp then scans the 256 bins from the top (8 per lane, an exact integer prefix over the lanes) to
the bin where the running count (mass) reaches the goal and extends the prefix by that bin's digit (8 / 8 / 8 / 7 bits).
A cluster's CTAs each histogram their share and read all the shares' bins through ``mapa`` + ``ld.shared::cluster``.

**The early finish** (a placement of the same selection, bit-identical by construction): after the second pass the
selected bin holds the candidates whose 16 top pattern bits are the prefix; when their cluster-wide count is at most
``CAP`` the remaining two passes are replaced by one compaction sweep (the candidates' patterns appended to a small
shared list, warp-aggregated) and an exact rank selection over the list (the ``rem_k``-th largest pattern, or the
first pattern in descending order at which the running fixed-point mass reaches ``rem_m``) -- the same integer
procedure the passes compute, on the same candidates. Otherwise the two passes run as before. The early finish pays
on the mass selection (top-p: its passes carry the mass chunks) and not on the count selection, so the default
applies it to the mass selection only (``early_finish="mass"``; ``"all"`` and ``""`` for the study).

**The pass count by input kind** (``n_passes_for``): a bf16 probability widened to fp32 has 16 zero low pattern
bits and an f16 one 13 (the widening is a shift), so after two (three) passes every remaining digit of every
candidate is zero and the threshold pattern is the prefix shifted -- those inputs run two (three) passes. fp32
inputs and the ``ex2`` weights of every logits input run four.

The kept set is ``w >= threshold`` with the threshold the largest of the applied filters' (ties kept); the kept mass is
the canonical sum tree over the kept weights (the others ``+0``), the count an integer sum, the log-probabilities
``mul(add(z_j, neg(lg2(mass))), LN2)``.

The passes are built on the row kernel's plumbing (``kernel.py``: the peers, the predicates) and run over the draw
kernel's staged tile (``draw_kernel._DrawLoader``: the pad an absent element reads and the register-path staging); they
belong to the truncated draw kernel's cache module (an edit here recompiles it and nothing else).
"""

from __future__ import annotations


import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, Int64
from cutlass._mlir.dialects import arith, llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .. import canon
from .cute_ext import CuteF32
from .draw_kernel import INT_MAX, _imax, _imin, _ld_shared_cluster_i32
from .kernel import (
    _map_shared_rank,
    _select,
)
from .schedule import BLOCK, LANES
from .spec import (
    shifted,
)
from .support_spec import FIX_BITS

BINS = 256
BINS_PER_LANE = BINS // LANES
PASS_SHIFTS = (
    23,
    15,
    7,
    0,
)  # the digit of pass p is bits [shift, shift + width) of the 31-bit pattern
PASS_WIDTHS = (8, 8, 8, 7)
INF_BITS = 0x7F800000
I64_MASK = (1 << 32) - 1
MASS_BITS = (
    FIX_BITS + 1
)  # a row's fixed-point masses sum below 2^45 (the tree sum's rounding slack included)
RM = 4  # lane-spread replicas of every bin for the mass chunks (same-address adds serialise)
WPT = 4  # 32-bit words a thread reads per sweep iteration
#: The early finish: the candidates of the selected bin the compaction list holds (two per lane of a warp).
CAP = 64
#: The count selection's list mode: the selected bin's candidates after the first pass that fit the gathered list.
CAP_L = 1024
#: The list mode pays from this staged share on, and only for the four-pass kernels (measured 2026-11-13).
EARLY_AFTER = (
    2  # the passes before the early finish decides (the prefix then has 16 bits)
)


# ---- device helpers (bit moves, shared-memory integer adds, cluster reads) ------------------------------------------
@dsl_user_op
def _f32_bits(a: Float32, *, loc=None, ip=None) -> Int32:
    """``mov.b32 r, f``: the bit pattern of an fp32 (no arithmetic)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _neg_or_nan(we) -> Boolean:
    """``we < 0`` or NaN in ONE compare (``setp.ltu.f32``, the unordered less-than): a weight outside the class -- the sign bit
    with a non-zero magnitude (``-0.0`` is a zero: ``-0.0 < 0`` is false), or a NaN of either sign (a logits input's ``ex2`` is
    never negative, so there it is the NaN test); ``+inf`` is not flagged (its row sum is, as before)."""
    return Boolean(
        arith.cmpf(
            arith.CmpFPredicate.ULT, Float32(we).ir_value(), Float32(0.0).ir_value()
        )
    )


@dsl_user_op
def _bitcast_i32_to_f32(a: Int32, *, loc=None, ip=None) -> Float32:
    """``mov.b32 f, r``: the fp32 of a bit pattern (no arithmetic)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(a).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _red_inc_u32(addr: Int32, *, loc=None, ip=None) -> None:
    """``red.shared.add.u32 [addr], 1`` (the hardware aggregates a warp's same-address increments)."""
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "red.shared.add.u32 [$0], 1;",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _red_add_u32(addr: Int32, v: Int32, *, loc=None, ip=None) -> None:
    """``red.shared.add.u32 [addr], v``."""
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
        "red.shared.add.u32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _atom_add_u32(addr: Int32, v: Int32, *, loc=None, ip=None) -> Int32:
    """``atom.shared.add.u32 d, [addr], v``: the old value (the compaction's slot base)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
            "atom.shared.add.u32 $0, [$1], $2;",
            "=r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _popc(a: Int32, *, loc=None, ip=None) -> Int32:
    """``popc.b32``: the number of set bits (a ballot's candidate count, a lane's rank within it). Inline asm, so the
    MLIR stays free of the math dialect the IR pin forbids (``cute.arch.popc`` lowers to ``math.ctpop``)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(a).ir_value(loc=loc, ip=ip)],
            "popc.b32 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _ld_shared_cluster_i64(addr: Int32, *, loc=None, ip=None) -> Int64:
    """``ld.shared::cluster.b64``: one 64-bit word from a peer CTA's shared memory (bits move; no arithmetic)."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared::cluster.b64 $0, [$1];",
            "=l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _cvt_rzi_s64_f32(a: Float32, *, loc=None, ip=None) -> Int64:
    """``cvt.rzi.s64.f32``: the truncated fixed-point value (saturates at 2^63 - 1), the class's ``trunc``."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "cvt.rzi.s64.f32 $0, $1;",
            "=l,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _f16_lo(w: Int32, *, loc=None, ip=None) -> Float32:
    """The low fp16 half of a word, widened exactly (``cvt.f32.f16``)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(w).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b16 lo, hi;\nmov.b32 {lo, hi}, $1;\ncvt.f32.f16 $0, lo;\n}",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _f16_hi(w: Int32, *, loc=None, ip=None) -> Float32:
    """The high fp16 half of a word, widened exactly."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(w).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b16 lo, hi;\nmov.b32 {lo, hi}, $1;\ncvt.f32.f16 $0, hi;\n}",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _shfl_up_i64(v, d):
    """``shfl.up`` of a 64-bit value by its halves."""
    lo = cute.arch.shuffle_sync_up((v & Int64(I64_MASK)).to(Int32), offset=d)
    hi = cute.arch.shuffle_sync_up((v >> 32).to(Int32), offset=d)
    return (hi.to(Int64) << 32) | (lo.to(Int64) & Int64(I64_MASK))


def _shfl_idx_i64(v, src):
    """``shfl.idx`` of a 64-bit value by its halves."""
    lo = cute.arch.shuffle_sync((v & Int64(I64_MASK)).to(Int32), offset=src)
    hi = cute.arch.shuffle_sync((v >> 32).to(Int32), offset=src)
    return (hi.to(Int64) << 32) | (lo.to(Int64) & Int64(I64_MASK))


def _weight_bits(we):
    """The 31-bit pattern of a non-negative weight (``+0`` and ``-0`` both 0)."""
    return _f32_bits(we) & Int32(0x7FFFFFFF)


def _smem_base(tensor) -> Int32:
    """The shared-memory byte address of a tensor's first element."""
    return Int32(tensor.iterator.toint())


def chunking(N: int):
    """The mass chunks: ``chunk_bits`` wide so that a slot's sum over at most ``N`` elements stays below 2^32, and enough
    of them for the 45-bit masses."""
    chunk_bits = min(16, 31 - (N - 1).bit_length())
    assert chunk_bits >= 4, N
    return chunk_bits, -(-MASS_BITS // chunk_bits)


class _IntReducer:
    """Exact integer reductions (min / add) of one value per thread over the lanes, the CTA's warps and the cluster's
    CTAs."""

    def __init__(
        self, W, C, warp_id, lane_id, ibuf=None, icbuf=None, ibuf64=None, icbuf64=None
    ):
        self.W, self.C, self.warp_id, self.lane_id = W, C, warp_id, lane_id
        self.ibuf, self.icbuf, self.ibuf64, self.icbuf64 = ibuf, icbuf, ibuf64, icbuf64
        self.n_cluster = 0

    def reduce64(self, v, peers):
        """The exact 64-bit integer sum of one value per thread (lanes by shuffles of the halves, then the 64-bit
        buffers)."""
        for mask in (1, 2, 4, 8, 16):
            lo = cute.arch.shuffle_sync_bfly(
                (v & Int64(I64_MASK)).to(Int32), offset=mask
            )
            hi = cute.arch.shuffle_sync_bfly((v >> 32).to(Int32), offset=mask)
            v = v + ((hi.to(Int64) << 32) | (lo.to(Int64) & Int64(I64_MASK)))
        if self.W > 1:
            cute.arch.barrier()
            self.ibuf64[self.warp_id, self.lane_id] = v
            cute.arch.barrier()
            acc = self.ibuf64[0, self.lane_id]
            for sid in range(1, self.W):
                acc = acc + self.ibuf64[sid, self.lane_id]
            v = acc
        if self.C > 1:
            if self.n_cluster:
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
            self.icbuf64[self.lane_id] = v
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            self.n_cluster += 1
            peers.n_cluster_exchanges += 1
            off = self.lane_id * 8
            acc = _ld_shared_cluster_i64(
                _map_shared_rank(self.icbuf64.iterator, Int32(0)) + off
            )
            for sid in range(1, self.C):
                acc = acc + _ld_shared_cluster_i64(
                    _map_shared_rank(self.icbuf64.iterator, Int32(sid)) + off
                )
            v = acc
        return v

    def reduce(self, v, op: str, peers):
        """The exact 32-bit integer reduction (``"min"`` or ``"add"``) of one value per thread."""
        comb = _imin if op == "min" else (lambda a, b: a + b)
        for mask in (1, 2, 4, 8, 16):
            v = comb(v, cute.arch.shuffle_sync_bfly(v, offset=mask))
        if self.W > 1:
            cute.arch.barrier()  # the previous reduction's readers are done
            self.ibuf[self.warp_id, self.lane_id] = v
            cute.arch.barrier()
            acc = self.ibuf[0, self.lane_id]
            for sid in range(1, self.W):
                acc = comb(acc, self.ibuf[sid, self.lane_id])
            v = acc
        if self.C > 1:
            if self.n_cluster:
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
            self.icbuf[self.lane_id] = v
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            self.n_cluster += 1
            peers.n_cluster_exchanges += (
                1  # so that ``peers.finish()`` holds the CTA for the peers' reads
            )
            off = self.lane_id * 4
            acc = _ld_shared_cluster_i32(
                _map_shared_rank(self.icbuf.iterator, Int32(0)) + off
            )
            for sid in range(1, self.C):
                acc = comb(
                    acc,
                    _ld_shared_cluster_i32(
                        _map_shared_rank(self.icbuf.iterator, Int32(sid)) + off
                    ),
                )
            v = acc
        return v


class _Radix:
    """What the radix passes need: the staged words, their count and global column origin, the histograms, and the early
    finish's buffers: the candidate list (``cand``), its counter box (``cbox``)."""

    def __init__(
        self,
        words,
        n_words,
        col0,
        N,
        nm,
        c,
        hist,
        mslot,
        mred,
        kind,
        from_logits,
        gcnt=None,
        gmass=None,
        cand=None,
        lst=None,
        cbox=None,
        lpos=None,
    ):
        self.words, self.n_words, self.col0, self.N, self.nm, self.c = (
            words,
            n_words,
            col0,
            N,
            nm,
            c,
        )
        self.hist, self.mslot, self.mred, self.kind, self.from_logits = (
            hist,
            mslot,
            mred,
            kind,
            from_logits,
        )
        self.gcnt, self.gmass = gcnt, gmass
        self.cand, self.cbox, self.lst, self.lpos = (
            cand,
            cbox,
            lst,
            lpos,
        )


def _kind(dtype) -> int:
    """The staged words' element kind: 0 fp32, 1 bf16, 2 f16."""
    if dtype.width == 32:
        return 0
    return 1 if dtype is cutlass.BFloat16 else 2


def count_list_default(n_passes: int, tile_bytes: int) -> bool:
    """The count selection's list mode by default: every kernel with two passes or more. Since 2026-11-17 the list carries the
    kept set (the compaction with positions of every candidate at or above the selected bin), so the mode replaces the later
    passes AND the row-wide kept compaction the tail needed otherwise (22k cycles per row); until then it was the four-pass
    kernels at a staged share of 128 KB or more only (2026-11-13: f32 262144 x 4096 rows 7126 -> 6600 us, bf16
    3081 -> 3537 us, f32 at one row 34.5 -> 39.4 us), and the M = 64 picks (T256 members, short shares) ran the four passes
    plus the kept compaction: 121k cycles per row against 89k on the list path."""
    return n_passes >= 2


def n_passes_for(kind: int, from_logits: bool) -> int:
    """The radix passes that determine the threshold pattern: four for fp32 weights (fp32 inputs and the ``ex2`` weights
    of every logits input), two for bf16 probabilities and three for f16 ones -- their fp32 patterns' low 16 / 13 bits
    are zero (the widening is a shift; an f16 subnormal has at most 10 significant bits), so every remaining digit of
    every candidate is zero and the pattern is the prefix shifted."""
    if from_logits or kind == 0:
        return 4
    return 2 if kind == 1 else 3


def _early_finish_of(k) -> str:
    """The kernel's early-finish policy ("" / "mass" / "all"; the support draw's kernel takes the default)."""
    return getattr(k, "early_finish", "mass")


def _smem_layout(k):
    """Byte sizes of the tile, the count histograms (two buffers, a dummy slot each), the mass slots, the reduced masses and
    the early finish's buffers."""
    n_share = k.cfg.blocks_per_cta(k.N) * BLOCK
    tile = n_share * k.elem_bytes
    hist = 2 * (BINS + 1) * 4 if (k.use_k or k.use_p) else 0
    mslot = (BINS * RM + LANES) * k.n_chunks * 4 if k.use_p else 0
    mred = 2 * BINS * 8 if k.use_p else 0
    if k.C > 1:
        hist += 2 * BINS * 4
        mred += 2 * BINS * 8 if k.use_p else 0
    cand = (
        (CAP + 1) * 4 + 2 * 4 if (k.use_k or k.use_p) else 0
    )  # the compaction list and its counter box
    if k.use_k and not k.use_p:
        cand += (
            2 * (CAP_L + 1) * 4
        )  # the count selection's gathered list and its positions (they alias the mass slots otherwise)
    return n_share, tile, hist, mslot, mred, cand


# ---- trace-time helpers (plain Python: static loops, selects, no staged control flow) --------------------------------
def _alloc_hist(k, smem):
    """The count histograms (two buffers, a dummy slot each), the mass chunk slots, the per-bin masses (two buffers), for a
    cluster the gathered cluster-wide bins (counts and masses, two buffers), and for the early finish the candidate list,
    its counter box and the rank-ordered masses / patterns (each ``CAP`` + a dummy slot), and the count selection's list with
    its positions (``CAP_L`` + a dummy slot each; inside the mass slots when there are any)."""
    hist = mslot = mred = gcnt = gmass = cand = cbox = lst = lpos = None
    if k.use_k or k.use_p:
        hist = smem.allocate_tensor(
            Int32, cute.make_layout((2 * (BINS + 1),)), byte_alignment=16
        )
        if k.C > 1:
            gcnt = smem.allocate_tensor(
                Int32, cute.make_layout((2 * BINS,)), byte_alignment=16
            )
        cand = smem.allocate_tensor(
            Int32, cute.make_layout((CAP + 1,)), byte_alignment=16
        )
        cbox = smem.allocate_tensor(Int32, cute.make_layout((2,)), byte_alignment=16)
    if k.use_p:
        mslot = smem.allocate_tensor(
            Int32,
            cute.make_layout(((BINS * RM + LANES) * k.n_chunks,)),
            byte_alignment=16,
        )
        mred = smem.allocate_tensor(
            Int64, cute.make_layout((2 * BINS,)), byte_alignment=16
        )
        if k.C > 1:
            gmass = smem.allocate_tensor(
                Int64, cute.make_layout((2 * BINS,)), byte_alignment=16
            )
        # the count selection's gathered list and its positions alias the mass slots (never live together: the count selection
        # completes before the mass selection clears its slots, and a cluster barrier ends the gather)
        assert (BINS * RM + LANES) * k.n_chunks >= 2 * (CAP_L + 1), (
            "the mass slots hold the list and its positions"
        )
        lst = cute.make_tensor(mslot.iterator, cute.make_layout((CAP_L + 1,)))
        lpos = cute.domain_offset(
            (CAP_L + 1,),
            cute.make_tensor(mslot.iterator, cute.make_layout((2 * (CAP_L + 1),))),
        )
    elif k.use_k:
        lst = smem.allocate_tensor(
            Int32, cute.make_layout((CAP_L + 1,)), byte_alignment=16
        )
        lpos = smem.allocate_tensor(
            Int32, cute.make_layout((CAP_L + 1,)), byte_alignment=16
        )
    return hist, mslot, mred, gcnt, gmass, cand, cbox, lst, lpos


def _clear(tensor, start: int, size: int, tid, zero, n: int):
    """Zero ``size`` entries of a shared tensor from ``start`` (a static range), ``n`` threads at a time."""
    for i in range(-(-size // n)):
        idx = tid + i * n
        idx = _select(
            idx < Int32(size), idx, Int32(size - 1)
        )  # the tail: the last entry zeroed again, harmless
        tensor[Int32(start) + idx] = zero


def _staged_weight(wv, h: int, kind: int, from_logits: bool, nm, c):
    """Element ``h`` of a staged word as the weight: the word itself (fp32), the bf16 halves widened by a shift, the f16
    halves by ``cvt.f32.f16``; a logit through the class's ``ex2(shifted(x))``."""
    if kind == 0:
        x = _bitcast_i32_to_f32(wv)
    elif kind == 1:
        x = (
            _bitcast_i32_to_f32(wv << 16)
            if h == 0
            else _bitcast_i32_to_f32(wv & Int32(-65536))
        )
    else:
        x = _f16_lo(wv) if h == 0 else _f16_hi(wv)
    if from_logits:
        return CuteF32.ex2(shifted(CuteF32, x, nm, c))
    return x


@cute.jit
def _radix_sweep(
    words: cute.Tensor,
    n_words: Int32,
    col0: Int32,
    N: Int32,
    tid: Int32,
    lane: Int32,
    nm: Float32,
    c: Float32,
    scale_fix: Float32,
    prefix: Int32,
    base_bits: Int32,
    cnt_base: Int32,
    m_base: Int32,
    p: cutlass.Constexpr[int],
    shift: cutlass.Constexpr[int],
    width: cutlass.Constexpr[int],
    kind: cutlass.Constexpr[int],
    from_logits: cutlass.Constexpr[bool],
    by_mass: cutlass.Constexpr[bool],
    n_chunks: cutlass.Constexpr[int],
    chunk_bits: cutlass.Constexpr[int],
    nthreads: cutlass.Constexpr[int],
    n_iters: cutlass.Constexpr[int],
    epw: cutlass.Constexpr[int],
) -> Int32:
    """One radix pass over the staged words: every candidate element adds one to its digit's bin (and its mass chunks); the
    smallest candidate pattern (pass 0 of a mass selection) is returned. (No Python-object locals before the runtime loop:
    the DSL carries every local through it.)"""
    dmask = Int32((1 << width) - 1)
    min_pos = Int32(INT_MAX)
    rep = lane & Int32(RM - 1)
    dummy_c = cnt_base + Int32(BINS * 4)
    dummy_m = m_base + (Int32(BINS * RM * n_chunks) + lane * n_chunks) * 4
    # the candidate test is on the VALUE: ``we >= base`` (a mass selection: ``>=`` the smallest positive pattern, so zeros are
    # out) -- for the class's non-negative weights exactly the pattern test ``bits >= base_bits`` (one compare, as before, and no
    # magnitude mask), and a NEGATIVE weight or a NaN is no candidate at all, so pass 0's cluster-wide count falls short of N
    # exactly when the row holds one: the kernel's input-domain test on the list paths, at no cost (2026-11-18, docs 21.9)
    if cutlass.const_expr(by_mass):
        base_f = _bitcast_i32_to_f32(_imax(base_bits, Int32(1)))
    else:
        base_f = _bitcast_i32_to_f32(base_bits)
    for it in range(n_iters):
        for m in cutlass.range_constexpr(WPT):
            idx = Int32(it * (nthreads * WPT) + m * nthreads) + tid
            valid = idx < n_words
            idxc = _select(valid, idx, n_words - 1)
            wv = words[idxc]
            for h in cutlass.range_constexpr(epw):
                we = _staged_weight(wv, h, kind, from_logits, nm, c)
                bits = _f32_bits(
                    we
                )  # a candidate's pattern (non-negative: the value test admits no other)
                g = col0 + idxc * epw + h
                ok = valid & (g < N) & (we >= base_f)
                if cutlass.const_expr(p > 0):
                    ok = ok & ((bits >> (shift + width)) == prefix)
                digit = (bits >> shift) & dmask
                _red_inc_u32(_select(ok, cnt_base + digit * 4, dummy_c))
                if cutlass.const_expr(by_mass):
                    # (2026-11-17: grouping the lanes by digit with match.any + redux.sync and issuing the reds from the group
                    # leaders measured 4x SLOWER than these per-lane reds -- the hardware aggregates same-address lanes itself)
                    q = _cvt_rzi_s64_f32(CuteF32.mul(we, scale_fix))
                    slot = m_base + ((digit * RM + rep) * n_chunks) * 4
                    for ch in cutlass.range_constexpr(n_chunks):
                        part = (
                            (q >> (ch * chunk_bits)) & Int64((1 << chunk_bits) - 1)
                        ).to(Int32)
                        _red_add_u32(
                            _select(ok, slot + ch * 4, dummy_m + ch * 4),
                            _select(ok, part, Int32(0)),
                        )
                    if cutlass.const_expr(p == 0):
                        min_pos = _imin(min_pos, _select(ok, bits, Int32(INT_MAX)))
    return min_pos


def _reduce_mass(mslot, mred, buf, tid, n: int, n_chunks: int, chunk_bits: int):
    """Every bin's chunk replicas summed into the 64-bit mass the scans read (an integer sum; order immaterial)."""
    for i in range(max(1, BINS // n)):
        bin_ = (tid + i * n) & Int32(
            BINS - 1
        )  # more threads than bins: the same value written twice, harmless
        tot = Int64(0)
        for ch in range(n_chunks):
            parts = [mslot[(bin_ * RM + r) * n_chunks + ch] for r in range(RM)]
            acc = parts[0]
            for r in range(1, RM):
                acc = acc + parts[r]
            tot = tot + ((acc.to(Int64) & Int64(I64_MASK)) << (ch * chunk_bits))
        mred[Int32(buf * BINS) + bin_] = tot


def _scan_bins(hist, mred, gcnt, gmass, buf, lane, by_mass: bool, rem_k, rem_m, C: int):
    """Every warp: the bins from the top, 8 per lane, with an exact integer prefix over the lanes. Returns ``(any, bin,
    count above, mass above, total count, total mass, the selected bin's count)`` -- the first bin (from the top) with
    elements whose running count (mass) reaches the remaining goal."""
    cs, ms, bins = [], [], []
    for i in range(BINS_PER_LANE):
        b = Int32(BINS - 1 - i) - lane * BINS_PER_LANE
        bins.append(b)
        if C > 1:  # the cluster-wide bins, gathered into this CTA's smem
            cnt = gcnt[Int32(buf * BINS) + b]
            m = gmass[Int32(buf * BINS) + b] if by_mass else Int64(0)
        else:
            cnt = hist[Int32(buf * (BINS + 1)) + b]
            m = mred[Int32(buf * BINS) + b] if by_mass else Int64(0)
        cs.append(cnt)
        ms.append(m)
    tc = cs[0]
    tm = ms[0]
    for i in range(1, BINS_PER_LANE):
        tc = tc + cs[i]
        tm = tm + ms[i]
    inc_c, inc_m = tc, tm
    for d in (1, 2, 4, 8, 16):
        oc = cute.arch.shuffle_sync_up(inc_c, offset=d)
        inc_c = _select(lane >= Int32(d), inc_c + oc, inc_c)
        if by_mass:
            om = _shfl_up_i64(inc_m, d)
            inc_m = _select(lane >= Int32(d), inc_m + om, inc_m)
    cum_c = inc_c - tc
    cum_m = inc_m - tm
    tot_c = cute.arch.shuffle_sync(inc_c, offset=LANES - 1)
    tot_m = _shfl_idx_i64(inc_m, LANES - 1) if by_mass else Int64(0)
    found = Boolean(False)
    sel = Int32(0)
    sel_c = Int32(0)
    ab_c = Int32(0)
    ab_m = Int64(0)
    for i in range(BINS_PER_LANE):
        nc = cum_c + cs[i]
        nmv = cum_m + ms[i]
        cond = (nmv >= rem_m) if by_mass else (nc >= rem_k)
        hit = _select(found, Boolean(False), (cs[i] > Int32(0)) & cond)
        sel = _select(hit, bins[i], sel)
        sel_c = _select(hit, cs[i], sel_c)
        ab_c = _select(hit, cum_c, ab_c)
        ab_m = _select(hit, cum_m, ab_m)
        found = found | hit
        cum_c = nc
        cum_m = nmv
    hit_lane = _select(found, lane, Int32(LANES))
    for mask in (1, 2, 4, 8, 16):
        hit_lane = _imin(hit_lane, cute.arch.shuffle_sync_bfly(hit_lane, offset=mask))
    anyhit = hit_lane < Int32(LANES)
    src = _select(anyhit, hit_lane, Int32(0))
    sel = cute.arch.shuffle_sync(sel, offset=src)
    sel_c = cute.arch.shuffle_sync(sel_c, offset=src)
    ab_c = cute.arch.shuffle_sync(ab_c, offset=src)
    if by_mass:
        ab_m = _shfl_idx_i64(ab_m, src)
    return anyhit, sel, ab_c, ab_m, tot_c, tot_m, sel_c


def _cluster_reduce(
    hist, mred, gcnt, gmass, buf, tid, by_mass: bool, C: int, n: int, c_rank
):
    """The cluster-wide bins in ONE round trip: every CTA publishes its bins (the barrier: release / acquire) and every thread
    sums its bins over the ``C`` peers' shared memories through ``mapa`` + ``ld.shared::cluster`` (``C`` remote loads per bin,
    256 bins over the CTA's threads) into its own ``gcnt`` / ``gmass`` of this buffer; the scans then read local shared memory.
    (Until 2026-11-17 an owner-sum round and a gather round: two round trips per pass; the sums are the same integers.)"""
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
    hb = [
        _map_shared_rank(hist.iterator, Int32(sid)) + Int32(buf * (BINS + 1) * 4)
        for sid in range(C)
    ]
    mb = (
        [
            _map_shared_rank(mred.iterator, Int32(sid)) + Int32(buf * BINS * 8)
            for sid in range(C)
        ]
        if by_mass
        else None
    )
    for i in range(max(1, BINS // n)):
        b = (tid + i * n) & Int32(
            BINS - 1
        )  # more threads than bins: a bin summed twice, the same value
        parts = [_ld_shared_cluster_i32(hb[sid] + b * 4) for sid in range(C)]
        acc = parts[0]
        for sid in range(1, C):
            acc = acc + parts[sid]
        gcnt[Int32(buf * BINS) + b] = acc
        if by_mass:
            mparts = [_ld_shared_cluster_i64(mb[sid] + b * 8) for sid in range(C)]
            macc = mparts[0]
            for sid in range(1, C):
                macc = macc + mparts[sid]
            gmass[Int32(buf * BINS) + b] = macc
    cute.arch.barrier()


def _pass(
    p: int,
    words,
    n_words,
    col0,
    N,
    tid,
    lane,
    nm,
    c,
    scale_fix,
    base_bits,
    prefix,
    rem_k,
    rem_m,
    hist,
    mslot,
    mred,
    gcnt,
    gmass,
    hbase,
    mbase,
    c_rank,
    C: int,
    n: int,
    n_iters: int,
    n_chunks: int,
    chunk_bits: int,
    kind: int,
    from_logits: bool,
    by_mass: bool,
    dbg=None,
):
    """One radix pass: clear, sweep, (reduce the masses,) the cluster reduction, the scan. Returns ``(prefix, rem_k, rem_m, the
    sweep's smallest candidate pattern, total count, total mass, the selected bin's count)``; ``dbg`` the debug trace's
    ``(tensor, row)`` (the plain passes only)."""
    shift, width = PASS_SHIFTS[p], PASS_WIDTHS[p]
    buf = p % 2
    _clear(hist, buf * (BINS + 1), BINS + 1, tid, Int32(0), n)
    if by_mass:
        _clear(mslot, 0, (BINS * RM + LANES) * n_chunks, tid, Int32(0), n)
    cute.arch.barrier()
    mp = _radix_sweep(
        words,
        n_words,
        col0,
        N,
        tid,
        lane,
        nm,
        c,
        scale_fix,
        prefix,
        base_bits,
        hbase + Int32(buf * (BINS + 1) * 4),
        mbase,
        p,
        shift,
        width,
        kind,
        from_logits,
        by_mass,
        n_chunks,
        chunk_bits,
        n,
        n_iters,
        1 if kind == 0 else 2,
    )
    cute.arch.barrier()
    if by_mass:
        _reduce_mass(mslot, mred, buf, tid, n, n_chunks, chunk_bits)
        cute.arch.barrier()
    if C > 1:
        _cluster_reduce(hist, mred, gcnt, gmass, buf, tid, by_mass, C, n, c_rank)
    if dbg is not None:
        dbg[0][dbg[1], 48 + p] = hist[Int32(buf * (BINS + 1)) + Int32(127)]
        dbg[0][dbg[1], 52 + p] = hist[Int32(buf * (BINS + 1)) + Int32(254)]
        dbg[0][dbg[1], 56 + p] = hist[Int32(buf * (BINS + 1)) + Int32(0)]
    anyhit, sel, ab_c, ab_m, tot_c, tot_m, sel_c = _scan_bins(
        hist, mred, gcnt, gmass, buf, lane, by_mass, rem_k, rem_m, C
    )
    if dbg is not None:
        dbg[0][dbg[1], p * 8 + 0] = tot_c
        dbg[0][dbg[1], p * 8 + 1] = sel
        dbg[0][dbg[1], p * 8 + 2] = _select(anyhit, Int32(1), Int32(0))
        dbg[0][dbg[1], p * 8 + 3] = ab_c
        dbg[0][dbg[1], p * 8 + 4] = rem_k
        dbg[0][dbg[1], p * 8 + 5] = prefix
    prefix = (prefix << width) | _select(anyhit, sel, Int32(0))
    rem_k = _select(anyhit, rem_k - ab_c, rem_k)
    if by_mass:
        rem_m = _select(anyhit, rem_m - ab_m, rem_m)
    return prefix, rem_k, rem_m, mp, tot_c, tot_m, _select(anyhit, sel_c, Int32(0))


@cute.jit
def _compact_sweep(
    words: cute.Tensor,
    n_words: Int32,
    col0: Int32,
    N: Int32,
    tid: Int32,
    lane: Int32,
    nm: Float32,
    c: Float32,
    prefix: Int32,
    base_bits: Int32,
    cand: cute.Tensor,
    cnt_addr: Int32,
    shift: cutlass.Constexpr[int],
    kind: cutlass.Constexpr[int],
    from_logits: cutlass.Constexpr[bool],
    by_mass: cutlass.Constexpr[bool],
    nthreads: cutlass.Constexpr[int],
    n_iters: cutlass.Constexpr[int],
    epw: cutlass.Constexpr[int],
    hi_bits: Int32,
    cap: cutlass.Constexpr[int] = CAP,
    cpos: cute.Tensor = None,
    with_pos: cutlass.Constexpr[bool] = False,
):
    """The compaction: the patterns of the candidates (``base_bits <= bits < hi_bits`` (``INT_MAX``: no bound) and ``bits >> shift
    == prefix``; for the mass goal positive) appended to ``cand`` -- warp-aggregated over one iteration's ``WPT x epw`` sub-batches: the ballots of every
    sub-batch first, then -- behind one warp-uniform branch per iteration, rarely taken -- ONE ``atom.shared.add`` by lane 0 for
    their total (the other lanes add 0 to the box's second word) and the slots from the running prefix of the ballots (a store
    in every iteration would order the next iteration's loads behind it: 16.5k cycles per sweep against the histogram sweep's
    9.7k, measured 2026-11-17); a candidate past the list's capacity goes to the dummy slot ``cap`` (the
    caller takes the list path only when the count fits); with ``with_pos`` the candidates' positions go to ``cpos`` at the
    same slots. The list ORDER is a placement: the selections rank by pattern value (ties by position pick the same value)
    and the list tail sorts by position. (Until 2026-11-17 every sub-batch ended in a dynamic ``if n_w > 0`` region with its
    own atomic: 20-25k cycles per sweep against the histogram sweep's 9k.)"""
    lane_lt = (Int32(1) << lane) - Int32(1)
    lead = lane == Int32(0)
    dummy = cnt_addr + Int32(4)
    if cutlass.const_expr(
        by_mass
    ):  # the value test (_radix_sweep's note): negatives and NaN out; a mass selection: zeros out
        base_f = _bitcast_i32_to_f32(_imax(base_bits, Int32(1)))
    else:
        base_f = _bitcast_i32_to_f32(base_bits)
    for it in cutlass.range_constexpr(n_iters):
        bl = []
        gl = []
        okl = []
        bal = []
        for m in cutlass.range_constexpr(WPT):
            idx = Int32(it * (nthreads * WPT) + m * nthreads) + tid
            valid = idx < n_words
            idxc = _select(valid, idx, n_words - 1)
            wv = words[idxc]
            for h in cutlass.range_constexpr(epw):
                we = _staged_weight(wv, h, kind, from_logits, nm, c)
                bits = _f32_bits(we)
                g = col0 + idxc * epw + h
                ok = (
                    valid
                    & (g < N)
                    & (we >= base_f)
                    & (bits < hi_bits)
                    & ((bits >> shift) == prefix)
                )
                bl.append(bits)
                gl.append(g)
                okl.append(ok)
                bal.append(cute.arch.vote_ballot_sync(ok))
        tot = _popc(bal[0])
        for j in cutlass.range_constexpr(1, WPT * epw):
            tot = tot + _popc(bal[j])
        if (
            tot > Int32(0)
        ):  # warp-uniform, rarely taken: the stores do not order the next iteration's loads
            old = _atom_add_u32(
                _select(lead, cnt_addr, dummy), _select(lead, tot, Int32(0))
            )
            run = cute.arch.shuffle_sync(old, offset=0)
            for j in cutlass.range_constexpr(WPT * epw):
                slot = run + _popc(bal[j] & lane_lt)
                slot = _select(okl[j] & (slot < Int32(cap)), slot, Int32(cap))
                cand[slot] = bl[j]
                if cutlass.const_expr(with_pos):
                    cpos[slot] = gl[j]
                run = run + _popc(bal[j])


@cute.jit
def _compact_sweep_eager(
    words: cute.Tensor,
    n_words: Int32,
    col0: Int32,
    N: Int32,
    tid: Int32,
    lane: Int32,
    nm: Float32,
    c: Float32,
    prefix: Int32,
    base_bits: Int32,
    cand: cute.Tensor,
    cnt_addr: Int32,
    shift: cutlass.Constexpr[int],
    kind: cutlass.Constexpr[int],
    from_logits: cutlass.Constexpr[bool],
    by_mass: cutlass.Constexpr[bool],
    nthreads: cutlass.Constexpr[int],
    n_iters: cutlass.Constexpr[int],
    epw: cutlass.Constexpr[int],
    cap: cutlass.Constexpr[int] = CAP,
):
    """The early finish's compaction, the EAGER form: every sub-batch of words whose ballot has candidates reserves its slots
    behind its own warp-uniform branch -- the right form when almost every sub-batch is empty (the selected bin's few
    candidates over the whole share): the branch-free form (`_compact_sweep`, the count tail's and the kept set's, where
    many candidates are compacted) measured 56 us against this form's 32 us for top-p alone at bf16 32768 on the one-CTA
    member (2026-11-17 ablation, docs 21.9). The patterns of the selected bin's candidates (``bits >> shift == prefix``, the same
    candidates the pass counted) appended to ``cand`` -- warp-aggregated: a word batch with candidates reserves its slots with
    one ``atom.shared.add`` by lane 0, the lanes' ranks within the ballot place them; a candidate past the list's capacity
    goes to the dummy slot ``CAP`` (the caller takes this path only when the count fits)."""
    lane_lt = (Int32(1) << lane) - Int32(1)
    if cutlass.const_expr(by_mass):  # the value test (_radix_sweep's note)
        base_f = _bitcast_i32_to_f32(_imax(base_bits, Int32(1)))
    else:
        base_f = _bitcast_i32_to_f32(base_bits)
    for it in range(n_iters):
        for m in cutlass.range_constexpr(WPT):
            idx = Int32(it * (nthreads * WPT) + m * nthreads) + tid
            valid = idx < n_words
            idxc = _select(valid, idx, n_words - 1)
            wv = words[idxc]
            for h in cutlass.range_constexpr(epw):
                we = _staged_weight(wv, h, kind, from_logits, nm, c)
                bits = _f32_bits(we)
                g = col0 + idxc * epw + h
                ok = valid & (g < N) & (we >= base_f) & ((bits >> shift) == prefix)
                ballot = cute.arch.vote_ballot_sync(ok)
                n_w = _popc(ballot)
                if n_w > Int32(0):
                    old = Int32(0)
                    if lane == Int32(0):
                        old = _atom_add_u32(cnt_addr, n_w)
                    base = cute.arch.shuffle_sync(old, offset=0)
                    slot = base + _popc(ballot & lane_lt)
                    slot = _select(ok & (slot < Int32(cap)), slot, Int32(cap))
                    cand[slot] = bits


@cute.jit
def _bitonic64_desc(k0: Int32, k1: Int32, lane: Int32):
    """The 64 keys of a warp (two per lane: index ``lane`` and ``lane + 32``) sorted ASCENDING by the bitonic network (the
    partner of index ``i`` is ``i ^ j``; the lower index keeps the minimum when ``(i & k) == 0``; ``j == 32`` pairs a lane's two
    registers, the other steps pair lanes of one register through xor shuffles). Returns the sorted ``(k0, k1)``."""
    i0 = lane
    i1 = lane + Int32(LANES)
    for k in cutlass.range_constexpr(1, 7):  # k = 2 .. 64
        kk = 1 << k
        for jj in cutlass.range_constexpr(k):  # j = kk / 2 .. 1
            j = kk >> (jj + 1)
            if cutlass.const_expr(j == LANES):
                lo = _imin(k0, k1)
                hi = _imax(k0, k1)
                k0 = lo
                k1 = hi
            else:
                q0 = cute.arch.shuffle_sync_bfly(k0, offset=j)
                q1 = cute.arch.shuffle_sync_bfly(k1, offset=j)
                lower = (lane & Int32(j)) == Int32(0)
                keep0 = lower == ((i0 & Int32(kk)) == Int32(0))
                keep1 = lower == ((i1 & Int32(kk)) == Int32(0))
                k0 = _select(keep0, _imin(k0, q0), _imax(k0, q0))
                k1 = _select(keep1, _imin(k1, q1), _imax(k1, q1))
    return k0, k1


@cute.jit
def _select_small(
    cand: cute.Tensor,
    n_all: Int32,
    lane: Int32,
    warp: Int32,
    rem_k: Int32,
    rem_m: Int64,
    scale_fix: Float32,
    by_mass: cutlass.Constexpr[bool],
) -> Int32:
    """The exact selection over the compacted candidates (at most ``CAP``, two per lane): the candidates sorted by pattern,
    descending (the bitonic network over shuffles; equal patterns in any order: the result is a pattern value), then the
    ``rem_k``-th largest pattern, or -- the masses in that order with an exact 64-bit prefix over the lanes -- the first pattern
    at which the running mass reaches ``rem_m`` (the first candidate when ``rem_m`` is 0, as the bin scan selects the first
    non-empty bin). Every warp computes the same value from the same list; no candidate (``n_all == 0``) gives 0."""
    i0 = lane
    i1 = lane + Int32(LANES)
    v0 = i0 < n_all
    v1 = i1 < n_all
    p0 = cand[_select(v0, i0, Int32(CAP))]
    p1 = cand[_select(v1, i1, Int32(CAP))]
    k0 = _select(
        v0, Int32(INT_MAX) - p0, Int32(INT_MAX)
    )  # ascending keys = descending patterns; a non-candidate sorts last
    k1 = _select(v1, Int32(INT_MAX) - p1, Int32(INT_MAX))
    k0, k1 = _bitonic64_desc(k0, k1, lane)
    s0 = Int32(INT_MAX) - k0  # the pattern at sorted index lane (0 past the candidates)
    s1 = Int32(INT_MAX) - k1  # ... and at lane + 32
    res = Int32(0)
    if cutlass.const_expr(by_mass):
        # the masses of the candidates (the same op on the same bits as the passes), in sorted order
        q0 = _select(
            v0,
            _cvt_rzi_s64_f32(CuteF32.mul(_bitcast_i32_to_f32(s0), scale_fix)),
            Int64(0),
        )
        q1 = _select(
            v1,
            _cvt_rzi_s64_f32(CuteF32.mul(_bitcast_i32_to_f32(s1), scale_fix)),
            Int64(0),
        )
        inc0 = q0
        for d in cutlass.range_constexpr(5):
            o = _shfl_up_i64(inc0, 1 << d)
            inc0 = _select(lane >= Int32(1 << d), inc0 + o, inc0)
        tot0 = _shfl_idx_i64(inc0, LANES - 1)
        inc1 = q1
        for d in cutlass.range_constexpr(5):
            o = _shfl_up_i64(inc1, 1 << d)
            inc1 = _select(lane >= Int32(1 << d), inc1 + o, inc1)
        inc1 = inc1 + tot0
        hit0 = v0 & (inc0 >= rem_m) & ((inc0 - q0 < rem_m) | (i0 == Int32(0)))
        hit1 = v1 & (inc1 >= rem_m) & (inc1 - q1 < rem_m)
        res = _select(hit0, s0, _select(hit1, s1, Int32(0)))
    else:
        hit0 = v0 & (i0 == rem_k - Int32(1))
        hit1 = v1 & (i1 == rem_k - Int32(1))
        res = _select(hit0, s0, _select(hit1, s1, Int32(0)))
    for mask in cutlass.range_constexpr(5):
        res = _imax(res, cute.arch.shuffle_sync_bfly(res, offset=1 << mask))
    return res


@cute.jit
def _passes_tail(
    words: cute.Tensor,
    n_words: Int32,
    col0: Int32,
    N: Int32,
    tid: Int32,
    lane: Int32,
    nm: Float32,
    c: Float32,
    scale_fix: Float32,
    base_bits: Int32,
    prefix: Int32,
    rem_k: Int32,
    rem_m: Int64,
    hist: cute.Tensor,
    mslot: cute.Tensor,
    mred: cute.Tensor,
    gcnt: cute.Tensor,
    gmass: cute.Tensor,
    hbase: Int32,
    mbase: Int32,
    c_rank: Int32,
    C: cutlass.Constexpr[int],
    nthreads: cutlass.Constexpr[int],
    n_iters: cutlass.Constexpr[int],
    n_chunks: cutlass.Constexpr[int],
    chunk_bits: cutlass.Constexpr[int],
    kind: cutlass.Constexpr[int],
    from_logits: cutlass.Constexpr[bool],
    by_mass: cutlass.Constexpr[bool],
    n_pass: cutlass.Constexpr[int],
) -> Int32:
    """The remaining radix passes (the early finish's other branch): the passes the plain form runs; the pattern is the
    final prefix shifted by the last pass's digit position."""
    for p in cutlass.range_constexpr(EARLY_AFTER, n_pass):
        prefix, rem_k, rem_m, _, _, _, _ = _pass(
            p,
            words,
            n_words,
            col0,
            N,
            tid,
            lane,
            nm,
            c,
            scale_fix,
            base_bits,
            prefix,
            rem_k,
            rem_m,
            hist,
            mslot,
            mred,
            gcnt,
            gmass,
            hbase,
            mbase,
            c_rank,
            C,
            nthreads,
            n_iters,
            n_chunks,
            chunk_bits,
            kind,
            from_logits,
            by_mass,
        )
    return prefix << PASS_SHIFTS[n_pass - 1]


@cute.jit
def _finish_small(
    words: cute.Tensor,
    n_words: Int32,
    col0: Int32,
    N: Int32,
    tid: Int32,
    lane: Int32,
    warp: Int32,
    nm: Float32,
    c: Float32,
    scale_fix: Float32,
    base_bits: Int32,
    prefix: Int32,
    rem_k: Int32,
    rem_m: Int64,
    cand: cute.Tensor,
    cbox: cute.Tensor,
    c_rank: Int32,
    C: cutlass.Constexpr[int],
    nthreads: cutlass.Constexpr[int],
    n_iters: cutlass.Constexpr[int],
    kind: cutlass.Constexpr[int],
    from_logits: cutlass.Constexpr[bool],
    by_mass: cutlass.Constexpr[bool],
) -> Int32:
    """The early finish: compact the selected bin's candidates (every CTA its share; a cluster's CTAs append the peers' lists
    through ``mapa``, so every CTA selects over the same set), then the exact rank selection over the list."""
    if tid == Int32(0):
        cbox[0] = Int32(0)
    cute.arch.barrier()
    _compact_sweep_eager(
        words,
        n_words,
        col0,
        N,
        tid,
        lane,
        nm,
        c,
        prefix,
        base_bits,
        cand,
        _smem_base(cbox),
        PASS_SHIFTS[EARLY_AFTER - 1],
        kind,
        from_logits,
        by_mass,
        nthreads,
        n_iters,
        1 if kind == 0 else 2,
    )
    cute.arch.barrier()
    n_all = cbox[0]
    if cutlass.const_expr(C > 1):
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        off = n_all
        for sid in cutlass.range_constexpr(C):
            cnt_s = _ld_shared_cluster_i32(_map_shared_rank(cbox.iterator, Int32(sid)))
            cnt_s = _select(Int32(sid) == c_rank, Int32(0), cnt_s)
            base_s = _map_shared_rank(cand.iterator, Int32(sid))
            if warp == Int32(0):
                for s in cutlass.range_constexpr(CAP // LANES):
                    j = lane + Int32(s * LANES)
                    ok = j < cnt_s
                    v = _ld_shared_cluster_i32(base_s + _select(ok, j, Int32(0)) * 4)
                    dst = off + j
                    cand[_select(ok & (dst < Int32(CAP)), dst, Int32(CAP))] = v
            off = off + cnt_s
        n_all = off
        cute.arch.barrier()
    n_all = _imin(n_all, Int32(CAP))
    return _select_small(cand, n_all, lane, warp, rem_k, rem_m, scale_fix, by_mass)


@cute.jit
def _select_tail(
    small: Boolean,
    words: cute.Tensor,
    n_words: Int32,
    col0: Int32,
    N: Int32,
    tid: Int32,
    lane: Int32,
    warp: Int32,
    nm: Float32,
    c: Float32,
    scale_fix: Float32,
    base_bits: Int32,
    prefix: Int32,
    rem_k: Int32,
    rem_m: Int64,
    hist: cute.Tensor,
    mslot: cute.Tensor,
    mred: cute.Tensor,
    gcnt: cute.Tensor,
    gmass: cute.Tensor,
    cand: cute.Tensor,
    cbox: cute.Tensor,
    hbase: Int32,
    mbase: Int32,
    c_rank: Int32,
    C: cutlass.Constexpr[int],
    nthreads: cutlass.Constexpr[int],
    n_iters: cutlass.Constexpr[int],
    n_chunks: cutlass.Constexpr[int],
    chunk_bits: cutlass.Constexpr[int],
    kind: cutlass.Constexpr[int],
    from_logits: cutlass.Constexpr[bool],
    by_mass: cutlass.Constexpr[bool],
    n_pass: cutlass.Constexpr[int],
) -> Int32:
    """After the first two passes: the early finish when the selected bin's candidates fit the list (``small``, a
    cluster-uniform value), else the remaining passes."""
    result = Int32(0)
    if small:
        result = _finish_small(
            words,
            n_words,
            col0,
            N,
            tid,
            lane,
            warp,
            nm,
            c,
            scale_fix,
            base_bits,
            prefix,
            rem_k,
            rem_m,
            cand,
            cbox,
            c_rank,
            C,
            nthreads,
            n_iters,
            kind,
            from_logits,
            by_mass,
        )
    else:
        result = _passes_tail(
            words,
            n_words,
            col0,
            N,
            tid,
            lane,
            nm,
            c,
            scale_fix,
            base_bits,
            prefix,
            rem_k,
            rem_m,
            hist,
            mslot,
            mred,
            gcnt,
            gmass,
            hbase,
            mbase,
            c_rank,
            C,
            nthreads,
            n_iters,
            n_chunks,
            chunk_bits,
            kind,
            from_logits,
            by_mass,
            n_pass,
        )
    return result


# ---- the top_k_first shortcut: the mass selection over the top-k kept set ------------------------------------------
def _kept_shortcut(k) -> bool:
    """Whether the top-p selection runs over the top-k kept set (both filters, order ``top_k_first``): the kept
    candidates are compacted once and, when they fit the list (the usual case: about ``k`` of them), the exact rank
    selection over the list replaces the mass passes -- the early finish's procedure on the same candidates, so
    bit-identical by construction; a kept set beyond the list runs the passes as before."""
    return bool(k.use_k and k.use_p and k.filters.order == "top_k_first")


def _list_totals(cand, n_all, lane, scale_fix):
    """The compacted candidates' total fixed-point mass (an exact 64-bit sum, order immaterial) and smallest pattern
    (INT_MAX when the list is empty): what the mass selection's first pass reports for its fallbacks."""
    i0 = lane
    i1 = lane + Int32(LANES)
    v0 = i0 < n_all
    v1 = i1 < n_all
    p0 = cand[_select(v0, i0, Int32(CAP))]
    p1 = cand[_select(v1, i1, Int32(CAP))]
    q0 = _select(
        v0, _cvt_rzi_s64_f32(CuteF32.mul(_bitcast_i32_to_f32(p0), scale_fix)), Int64(0)
    )
    q1 = _select(
        v1, _cvt_rzi_s64_f32(CuteF32.mul(_bitcast_i32_to_f32(p1), scale_fix)), Int64(0)
    )
    tot = q0 + q1
    mn = _imin(_select(v0, p0, Int32(INT_MAX)), _select(v1, p1, Int32(INT_MAX)))
    for mask in (1, 2, 4, 8, 16):
        lo = cute.arch.shuffle_sync_bfly((tot & Int64(I64_MASK)).to(Int32), offset=mask)
        hi = cute.arch.shuffle_sync_bfly((tot >> 32).to(Int32), offset=mask)
        tot = tot + ((hi.to(Int64) << 32) | (lo.to(Int64) & Int64(I64_MASK)))
        mn = _imin(mn, cute.arch.shuffle_sync_bfly(mn, offset=mask))
    return tot, mn


# ---- the count selection's list mode -------------------------------------------------------------------------------
def _list_pass(
    p: int, lst, n_all, tid, lane, prefix, rem_k, hist, hbase, nthreads: int
):
    """One count pass over the gathered list (CTA-local: every CTA holds the same list, so every CTA finds the same
    digit): clear, the list's candidates of the prefix counted per digit, the scan. Returns ``(prefix, rem_k)``."""
    shift, width = PASS_SHIFTS[p], PASS_WIDTHS[p]
    buf = p % 2
    _clear(hist, buf * (BINS + 1), BINS + 1, tid, Int32(0), nthreads)
    cute.arch.barrier()
    cnt_base = hbase + Int32(buf * (BINS + 1) * 4)
    dummy = cnt_base + Int32(BINS * 4)
    dmask = Int32((1 << width) - 1)
    for it in range(-(-CAP_L // nthreads)):
        i = Int32(it * nthreads) + tid
        ok = i < n_all
        bits = lst[_select(ok, i, Int32(CAP_L))]
        ok = ok & ((bits >> (shift + width)) == prefix)
        digit = (bits >> shift) & dmask
        _red_inc_u32(_select(ok, cnt_base + digit * 4, dummy))
    cute.arch.barrier()
    anyhit, sel, ab_c, _, _, _, _ = _scan_bins(
        hist, hist, hist, hist, buf, lane, False, rem_k, Int64(0), 1
    )
    prefix = (prefix << width) | _select(anyhit, sel, Int32(0))
    rem_k = _select(anyhit, rem_k - ab_c, rem_k)
    return prefix, rem_k


@cute.jit
def _count_tail(
    small: Boolean,
    all_fit: Boolean,
    words: cute.Tensor,
    n_words: Int32,
    col0: Int32,
    N: Int32,
    tid: Int32,
    lane: Int32,
    warp: Int32,
    nm: Float32,
    c: Float32,
    base_bits: Int32,
    prefix: Int32,
    rem_k: Int32,
    k_all: Int32,
    hist: cute.Tensor,
    gcnt: cute.Tensor,
    lst: cute.Tensor,
    lpos: cute.Tensor,
    cbox: cute.Tensor,
    hbase: Int32,
    c_rank: Int32,
    C: cutlass.Constexpr[int],
    nthreads: cutlass.Constexpr[int],
    n_iters: cutlass.Constexpr[int],
    n_chunks: cutlass.Constexpr[int],
    chunk_bits: cutlass.Constexpr[int],
    kind: cutlass.Constexpr[int],
    from_logits: cutlass.Constexpr[bool],
    n_pass: cutlass.Constexpr[int],
):
    """After the count selection's first pass: when the selected bin's candidates fit the list (``small``, cluster-uniform) they
    are compacted WITH their positions -- and when the candidates ABOVE the bin fit too (``all_fit``) those as well, so that
    the kept set (every element whose pattern is at or above any later threshold) is on ``lst`` / ``lpos`` and the kernel
    filters it into the tail's list instead of sweeping the row again -- every CTA its share, the cluster's lists gathered
    into every CTA (the loops over the peers' actual counts: a dynamic trip count); the remaining digits are found on the list:
    by exact rank when at most ``CAP`` entries (the ``k_all``-th largest over the whole list with the entries above, the
    ``rem_k``-th within the bin alone), else by CTA-local histogram passes over the list (the entries above the bin do not
    match the prefix and are not counted); otherwise the remaining passes run over the words. Returns ``(the threshold
    pattern, the list's count when it holds the kept set, else -1)``."""
    pattern = Int32(0)
    n_list = Int32(-1)
    if small:
        if tid == Int32(0):
            cbox[0] = Int32(0)
        cute.arch.barrier()
        lo = _imax(base_bits, prefix << PASS_SHIFTS[0])
        hi = _select(
            all_fit | (prefix >= Int32(BINS - 1)),
            Int32(INT_MAX),
            (prefix + Int32(1)) << PASS_SHIFTS[0],
        )
        _compact_sweep(
            words,
            n_words,
            col0,
            N,
            tid,
            lane,
            nm,
            c,
            Int32(0),
            lo,
            lst,
            _smem_base(cbox),
            31,
            kind,
            from_logits,
            False,
            nthreads,
            n_iters,
            1 if kind == 0 else 2,
            hi,
            CAP_L,
            lpos,
            True,
        )
        cute.arch.barrier()
        n_all = cbox[0]
        if cutlass.const_expr(C > 1):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            off = n_all
            for sid in cutlass.range_constexpr(C):
                cnt_s = _ld_shared_cluster_i32(
                    _map_shared_rank(cbox.iterator, Int32(sid))
                )
                cnt_s = _select(Int32(sid) == c_rank, Int32(0), cnt_s)
                base_s = _map_shared_rank(lst.iterator, Int32(sid))
                base_p = _map_shared_rank(lpos.iterator, Int32(sid))
                n_s = (cnt_s + Int32(nthreads - 1)) // Int32(nthreads)
                for s in range(n_s):
                    j = tid + s * Int32(nthreads)
                    ok = j < cnt_s
                    v = _ld_shared_cluster_i32(base_s + _select(ok, j, Int32(0)) * 4)
                    g = _ld_shared_cluster_i32(base_p + _select(ok, j, Int32(0)) * 4)
                    dst = off + j
                    slot = _select(ok & (dst < Int32(CAP_L)), dst, Int32(CAP_L))
                    lst[slot] = v
                    lpos[slot] = g
                off = off + cnt_s
            n_all = off
            cute.arch.cluster_arrive()  # every CTA has read the peers' lists (the list may alias the mass slots)
            cute.arch.cluster_wait()
        n_all = _imin(n_all, Int32(CAP_L))
        n_list = _select(all_fit, n_all, Int32(-1))
        if n_all <= Int32(CAP):
            pattern = _select_small(
                lst,
                n_all,
                lane,
                warp,
                _select(all_fit, k_all, rem_k),
                Int64(0),
                Float32(0.0),
                False,
            )
        else:
            pre = prefix
            rk = rem_k
            for p in cutlass.range_constexpr(1, n_pass):
                pre, rk = _list_pass(
                    p, lst, n_all, tid, lane, pre, rk, hist, hbase, nthreads
                )
            pattern = pre << PASS_SHIFTS[n_pass - 1]
    else:
        pre = prefix
        rk = rem_k
        for p in cutlass.range_constexpr(1, n_pass):
            pre, rk, _, _, _, _, _ = _pass(
                p,
                words,
                n_words,
                col0,
                N,
                tid,
                lane,
                nm,
                c,
                Float32(0.0),
                base_bits,
                pre,
                rk,
                Int64(0),
                hist,
                hist,
                hist,
                gcnt,
                hist,
                hbase,
                Int32(0),
                c_rank,
                C,
                nthreads,
                n_iters,
                n_chunks,
                chunk_bits,
                kind,
                from_logits,
                False,
            )
        pattern = pre << PASS_SHIFTS[n_pass - 1]
    return pattern, n_list


@cute.jit
def _list_filter(
    lst: cute.Tensor,
    lpos: cute.Tensor,
    n_list: Int32,
    thr_bits: Int32,
    cand: cute.Tensor,
    cpos: cute.Tensor,
    cbox: cute.Tensor,
    tid: Int32,
    lane: Int32,
    nthreads: cutlass.Constexpr[int],
) -> Int32:
    """The kept entries of the count tail's list (pattern ``>= thr_bits``, positive) compacted with their positions into the
    tail's list (every CTA holds the same count-tail list, so every CTA builds a list of the same entries); returns the count
    (CTA-uniform; past ``CAP`` the entries went to the dummy slot and the caller takes the dense path)."""
    if tid == Int32(0):
        cbox[0] = Int32(0)
    cute.arch.barrier()
    lane_lt = (Int32(1) << lane) - Int32(1)
    lead = lane == Int32(0)
    box = _smem_base(cbox)
    n_s = (n_list + Int32(nthreads - 1)) // Int32(nthreads)
    for s in range(n_s):
        j = tid + s * Int32(nthreads)
        ok = j < n_list
        jc = _select(ok, j, Int32(CAP_L))
        b = lst[jc]
        g = lpos[jc]
        ok = ok & (b >= thr_bits) & (b > Int32(0))
        ballot = cute.arch.vote_ballot_sync(ok)
        old = _atom_add_u32(
            _select(lead, box, box + Int32(4)), _select(lead, _popc(ballot), Int32(0))
        )
        base = cute.arch.shuffle_sync(old, offset=0)
        slot = base + _popc(ballot & lane_lt)
        slot = _select(ok & (slot < Int32(CAP)), slot, Int32(CAP))
        cand[slot] = b
        cpos[slot] = g
    cute.arch.barrier()
    return cbox[0]


def _radix_select(
    k,
    rx,
    tid,
    lane,
    by_mass,
    goal_k,
    goal_m,
    scale_fix,
    base_bits,
    peers,
    red,
    dbg=None,
):
    """The threshold pattern among the candidates (present, pattern ``>= base_bits``; for the mass goal also positive): the
    ``goal_k``-th largest pattern, or the largest pattern whose upward-closed fixed-point mass reaches ``goal_m``. Returns
    ``(pattern, candidate count, candidate mass, smallest candidate pattern, list count)`` -- the middle three from the first
    pass (INT_MAX when there is no candidate), for the mass goal's fallback; the last the count tail's list count when the
    kept set is on ``rx.lst`` / ``rx.lpos`` (the count selection's list mode), else -1. The pass count is the input kind's (``n_passes_for``:
    the pattern is the final prefix shifted); with the early finish (the kernel's policy: the mass selection, every
    selection, or none) the passes after the second are replaced by the compaction and the rank selection when the
    selected bin's candidates fit the list."""
    n_pass = n_passes_for(rx.kind, rx.from_logits)
    ef = _early_finish_of(k)
    use_ef = bool(ef) and (ef == "all" or by_mass) and n_pass > EARLY_AFTER
    C = k.C
    n = k.num_threads
    n_iters = -(-(k.cfg.blocks_per_cta(k.N) * BLOCK * k.elem_bytes // 4) // (n * WPT))
    hbase = _smem_base(rx.hist)
    mbase = _smem_base(rx.mslot) if by_mass else Int32(0)
    c_rank = peers.cta_rank
    # the tensors the jit tail takes (never dereferenced when unused: gated by their constexpr flags)
    mslot = rx.mslot if by_mass else rx.hist
    mred = rx.mred if by_mass else rx.hist
    gcnt = rx.gcnt if C > 1 else rx.hist
    gmass = rx.gmass if (C > 1 and by_mass) else rx.hist
    prefix = Int32(0)
    rem_k, rem_m = goal_k, goal_m
    tot_c0, tot_m0 = Int32(0), Int64(0)
    min_pos = Int32(INT_MAX)
    sel_c = Int32(0)
    if not by_mass and getattr(k, "count_list", True) and n_pass >= 2:
        # the list mode: the first pass over the words, then the selected bin's candidates on the list when they fit
        prefix, rem_k, rem_m, mp, tot_c, tot_m, sel_c = _pass(
            0,
            rx.words,
            rx.n_words,
            rx.col0,
            rx.N,
            tid,
            lane,
            rx.nm,
            rx.c,
            scale_fix,
            base_bits,
            prefix,
            rem_k,
            rem_m,
            rx.hist,
            mslot,
            mred,
            gcnt,
            gmass,
            hbase,
            mbase,
            c_rank,
            C,
            n,
            n_iters,
            k.n_chunks,
            k.chunk_bits,
            rx.kind,
            rx.from_logits,
            False,
            dbg,
        )
        small = sel_c <= Int32(CAP_L)
        all_fit = (sel_c + (goal_k - rem_k)) <= Int32(
            CAP_L
        )  # the bin's candidates plus every candidate above it
        prefix, n_list = _count_tail(
            small,
            all_fit,
            rx.words,
            rx.n_words,
            rx.col0,
            rx.N,
            tid,
            lane,
            peers.warp_id,
            rx.nm,
            rx.c,
            base_bits,
            prefix,
            rem_k,
            goal_k,
            rx.hist,
            gcnt,
            rx.lst,
            rx.lpos,
            rx.cbox,
            hbase,
            c_rank,
            C,
            n,
            n_iters,
            k.n_chunks,
            k.chunk_bits,
            rx.kind,
            rx.from_logits,
            n_pass,
        )
        if C > 1:
            peers.n_cluster_exchanges += n_pass + 2
        return prefix, tot_c, tot_m, min_pos, n_list
    for p in range(EARLY_AFTER if use_ef else n_pass):
        prefix, rem_k, rem_m, mp, tot_c, tot_m, sel_c = _pass(
            p,
            rx.words,
            rx.n_words,
            rx.col0,
            rx.N,
            tid,
            lane,
            rx.nm,
            rx.c,
            scale_fix,
            base_bits,
            prefix,
            rem_k,
            rem_m,
            rx.hist,
            mslot,
            mred,
            gcnt,
            gmass,
            hbase,
            mbase,
            c_rank,
            C,
            n,
            n_iters,
            k.n_chunks,
            k.chunk_bits,
            rx.kind,
            rx.from_logits,
            by_mass,
            dbg,
        )
        if C > 1:
            peers.n_cluster_exchanges += 1
        if p == 0:
            tot_c0, tot_m0 = tot_c, tot_m
            if by_mass:
                min_pos = mp
    if use_ef:
        small = sel_c <= Int32(
            CAP
        )  # cluster-uniform: every CTA scanned the same reduced bins
        # the early finish's list is the count selection's (it aliases the mass slots, free after the passes), never the tail's
        # `cand`: a top-p selection over the row runs AFTER the kept set was filtered into `cand` / `cpos` (the joint order)
        ef_list = rx.lst if rx.lst is not None else rx.cand
        prefix = _select_tail(
            small,
            rx.words,
            rx.n_words,
            rx.col0,
            rx.N,
            tid,
            lane,
            peers.warp_id,
            rx.nm,
            rx.c,
            scale_fix,
            base_bits,
            prefix,
            rem_k,
            rem_m,
            rx.hist,
            mslot,
            mred,
            gcnt,
            gmass,
            ef_list,
            rx.cbox,
            hbase,
            mbase,
            c_rank,
            C,
            n,
            n_iters,
            k.n_chunks,
            k.chunk_bits,
            rx.kind,
            rx.from_logits,
            by_mass,
            n_pass,
        )
        if C > 1:
            peers.n_cluster_exchanges += 1
    else:
        prefix = prefix << PASS_SHIFTS[n_pass - 1]
    if by_mass:
        min_pos = red.reduce(min_pos, "min", peers)
    return prefix, tot_c0, tot_m0, min_pos, Int32(-1)


def _masked_sum(k, leaf, peers, thr_bits):
    """The canonical sum tree over the weights with pattern ``>= thr_bits`` (the others ``+0``): the kept test on the value
    (the same set for the class's non-negative weights; a negative weight or a NaN is ``+0`` here and flagged elsewhere)."""
    fp = CuteF32
    thr = _bitcast_i32_to_f32(thr_bits)

    def masked_leaf(key):
        we = leaf(key)
        return _select(we >= thr, we, Float32(0.0))

    state, start = canon.init_state(k.plan_sum, fp, masked_leaf)
    return canon.run(k.plan_sum, state, start, peers, fp)
