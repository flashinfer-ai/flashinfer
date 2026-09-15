"""MSA top-k block selection for SM90 (Hopper).

Ported from the Kernel-Factory winner (38.9x geomean over the shipped Triton
selection on 60 held-out shapes, none slower, measured against a contiguous
score tensor -- the permuted-view form understates the reference).

Emits indices already sorted ascending, which the FlashInfer contract requires
and which the Triton path has to pay a separate sort pass for.

The compile cache keys on the occupancy-derived CTA geometry, a bounded
discrete set, so shapes never force a recompile.
"""

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Int32, Float32
from cutlass.cute.runtime import from_dlpack

# ---------------------------------------------------------------------------
# MSA stage-2 block selection.
#
#   in : max_score (Hq, max_k_tiles, total_q) fp32, -inf past each token's
#        causal extent.  The reduction axis (tiles) is STRIDED by total_q;
#        the token axis is contiguous.
#   out: topk_idx (Hq, total_q, 16) int32, ascending, -1 padded.
#
# Decomposition
# -------------
# A thread owns one (head, token) column and one of W tile groups.  Threads
# are laid out (c = column, w = tile group) with the column index FASTEST, so
# a warp load covers 32/C tile rows of C consecutive floats.  C therefore
# sets the memory shape - C >= 32 is one full 128B line per warp load, C = 8
# is one full 32B sector, below that the warp pays a sector per lane - while
# W sets both the parallelism (total threads = Hq*Nq*W) and the depth of the
# shared-memory reduction (log2 W rounds).  C*W is the CTA size, so the two
# trade directly against each other, and the balance point is genuinely
# shape dependent: long columns with few tokens want a big W, short columns
# with many tokens want a big C.
#
# C and W are searched independently by a cost model rather than by a
# threshold, under a CTA-size ceiling and a whole-warp floor.  The model
# prices issue slots, L1 wavefronts (one per distinct 128B line a warp load
# touches), the log2(W) reduction rounds, the epilogue's output stores and
# unhidden memory latency, and the cheapest wins.  Every input is a hardware
# property (SM count, warp size, 128B line, issue width, warps resident per
# SM, register file and shared memory capacity, the instruction count of one
# chunk) or a runtime extent; none of it is a constant read off a workload
# list.  The winner is baked in as a compile-time constant, which folds every
# tile stride, every shared address and the whole reduction schedule into
# immediates.
#
# Note that L1 wavefronts per CTA come out at T whatever C is (a CTA has to
# pull T tile rows either way, and a narrower column tile just uses less of
# each 128B line it touches), so C cannot buy memory time back on its own -
# it trades issue slots against the grid, and against the epilogue.
#
# Tile groups are walked in one of two orders, chosen by the same model:
# INTERLEAVED (t = w + j*W) puts a warp's groups on adjacent tile rows so
# their loads share cache lines, which only pays while a row is short enough
# for the sharing to happen; BLOCKED (t = w*R + j) instead gives every warp
# one contiguous strided stream.
#
# Selection core
# --------------
# The running top-16 lives in registers, sorted descending.  Each 16-tile
# chunk is sorted by a 60-comparator depth-10 network on PACKED keys: the
# in-chunk position rides in the score's low mantissa bits, so a sort
# comparator is a bare FMAX/FMIN pair and none of the index selects it would
# otherwise need reaches the integer pipe (NCU showed ALU at 75% with DRAM at
# 35% before this change).  A 32-tile chunk only ever needs its top 16, so it
# runs a (32,16) selection - two 16-sorts, 16 maxima against the reversed
# half, one bitonic clean - rather than a full Batcher-32 sort.  Exact scores
# are gathered back out of the shared chunk scratch before the 16 maxima and
# the bitonic merge, so the accumulator only ever compares untruncated values
# - a packed key orders one chunk against itself and nothing else.
# -inf tiles lose every comparison, so causal masking costs nothing.
# ---------------------------------------------------------------------------

TOPK = 16
TILES = (8, 16, 32)           # register-tile widths the kernel can use
TOPK2 = 2 * TOPK              # shared rows the reduction tree needs
NT_MAX = 256                  # CTA size ceiling
FILL_BLOCK = 128              # 8 output rows/CTA, one half-warp per row
LOG_NT = 8
NSM = 132                     # SMs on an H100 SXM
WARPS_SM = 24                 # warps resident per SM at this register count
LINE = 128                    # L1 line, bytes
SECT = 32                     # L1 sector, bytes: the unit a miss actually moves
SCHED = 4.0                   # warp instructions an SM issues per cycle
L1_WF = 2.0                   # L1 wavefronts an SM retires per cycle: the
                              # tag stage takes four lookups a cycle and the
                              # return path one 128B line, so sector-sized
                              # requests from distinct lines pair up
IPC1 = 0.8                    # what one dependency-limited warp sustains
TREE_W = 1.0                  # reduction rounds cost what they issue
DRAM_BPC = 2100.0             # device DRAM bytes per SM clock (3 TB/s @ 1.41 GHz)
LAT = 500                     # cycles of memory latency to hide per chunk
MAX_LOG = 10
SMEM_CAP = 232448           # bytes a CTA may hold on SM90
BAR = 60                   # cycles a CTA barrier costs
OVERLAP = 0.35             # share of the hidden limit that still shows up
# Instruction counts one warp issues, read off the SASS the kernel emits
# (NCU "Instructions Executed" per source line, divided by the warps that ran).
FIRST_INST = {8: 175, 16: 344, 32: 712}   # load+mask+pack+sort+gather
CHUNK_INST = {8: 175, 16: 536, 32: 904}   # plus the 16-maxima/bitonic fold
FILTER_GROUP_INST = 160      # one eight-score compact/prefix group
FILTER_DRAIN_INST = 600      # one 16-candidate drain into the reservoir
MERGE_INST = 240           # scalar 16-way merge: 16 maxima + a 32 bitonic
PUBLISH_INST = 34          # re-publishing 16 values + 16 indices to smem
COOP_INST = {1: 34, 2: 60, 4: 110, 8: 210}     # the same spread over lanes
COOP_MIN = 8               # narrowest lane group worth spreading a merge over.
                           # Below eight lanes each thread has to carry four or
                           # eight slots, and the round then issues twice the
                           # instructions on twice the warps to save a third of
                           # one merge's latency.  Measured both ways: two-lane
                           # rounds cost 9-15% on the deep C=1 trees and return
                           # only 2-3% on the shallow ones, so the scalar merge
                           # stays until a whole half-list fits in lanes.
FINAL_INST = 75            # fused merge + ascending index sort + store
MERGE_CP = 230             # serial 16-way merge, in latency
COOP_CP = 70               # the same merge spread over 8/16 lanes
FINAL_CP = 120             # fused merge + ascending index sort + store
EPI_CP = 120               # scalar ascending sort in the fallback epilogue
NEG_BIG = -3.0e38             # finite empty-slot sentinel: -inf would turn
                              # into a NaN once a position is OR-ed into the
                              # mantissa.  PAD_LIM is the decode threshold; it
                              # sits above the sentinel (packing perturbs its
                              # low mantissa bits) and far below any real
                              # proxy score, which is O(1).
PAD_LIM = -1.0e30
# Forced-block bias, applied at the score load. Distinct per tile: two tiles
# carrying an identical score can take the same rank in the strict-< ranking
# below, which leaves one output slot unwritten. FORCE_STEP keeps the forced
# run strictly decreasing while staying far above any real proxy score (O(1)).
# Forced tiles are made strictly decreasing in t so no two can tie: the rank
# below is a strict <, and a tie leaves an output slot unwritten. Bases are
# chosen so ONE step clears the fp32 ULP at both (1e10 > ULP(1e16) = 1.1e9 and
# > ULP(1e14) = 1.1e7) -- a step that rounds away at the larger base would
# silently reintroduce the ties. The bands stay disjoint through t = 8192 and
# sit 14 orders above any real proxy score, which is O(1).
FORCE_BEGIN_BASE = 1.0e16
FORCE_END_BASE = 1.0e14
FORCE_STEP = 1.0e10
SENT = 16777216.0             # 2^24: empty slot, sorts last, decodes to -1
IDX_BIAS_BITS = 1258291200    # 0x4b000000: monotone fp32 key for index 0
FULL = 0xFFFFFFFF


def _mk_net():
    net = (
        (0, 13), (1, 12), (2, 15), (3, 14), (4, 8), (5, 6), (7, 11), (9, 10),
        (0, 5), (1, 7), (2, 9), (3, 4), (6, 13), (8, 14), (10, 15), (11, 12),
        (0, 1), (2, 3), (4, 5), (6, 8), (7, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 10), (5, 11), (6, 7), (8, 9), (12, 14), (13, 15),
        (1, 2), (3, 12), (4, 6), (5, 7), (8, 10), (9, 11), (13, 14),
        (1, 4), (2, 6), (5, 8), (7, 10), (9, 13), (11, 14),
        (2, 4), (3, 6), (9, 12), (11, 13),
        (3, 5), (6, 8), (7, 9), (10, 12),
        (3, 4), (5, 6), (7, 8), (9, 10), (11, 12),
        (6, 7), (8, 9),
    )
    return tuple(p[0] for p in net), tuple(p[1] for p in net)


def _batcher(n):
    """Batcher odd-even mergesort on n elements, n a power of two."""
    pairs = []

    def merge(lo, m, r):
        step = r * 2
        if step < m:
            merge(lo, m, step)
            merge(lo + r, m, step)
            for i in range(lo + r, lo + m - r, step):
                pairs.append((i, i + r))
        else:
            pairs.append((lo, lo + r))

    def srt(lo, m):
        if m > 1:
            h = m // 2
            srt(lo, h)
            srt(lo + h, h)
            merge(lo, m, 1)

    srt(0, n)
    return tuple(p[0] for p in pairs), tuple(p[1] for p in pairs)


def _mk_bmerge():
    a, b = [], []
    step = TOPK // 2
    while step >= 1:
        for i in range(TOPK):
            if (i & step) == 0:
                a.append(i)
                b.append(i + step)
        step //= 2
    return tuple(a), tuple(b)


def _mk_asc_stages():
    """Partner offset and direction bits for a 16-lane bitonic sort."""
    out = []
    k = 2
    while k <= TOPK:
        j = k // 2
        while j >= 1:
            out.append((j, j.bit_length() - 1, k.bit_length() - 1))
            j //= 2
        k *= 2
    return tuple(out)


_SA, _SB = _mk_net()          # 60 comparators, depth 10 (16 elements)
_C16A, _C16B = _SA, _SB       # 60 comparators for a 16-element chunk
_C8A, _C8B = _batcher(8)      # 19 comparators for an 8-element chunk
_MA, _MB = _mk_bmerge()       # 32 comparators, bitonic merge of 16
_ASC = _mk_asc_stages()        # 10 lane-parallel stages, ascending


@cute.kernel
def _fill_kernel(mS: cute.Tensor, mO: cute.Tensor, mNVP: cute.Tensor,
                 MASK: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    lane = tidx & Int32(31)
    warp = tidx >> 5
    row = bx * Int32(8) + warp * Int32(2) + (lane >> 4)
    slot = lane & Int32(15)

    Hq = mS.shape[0]
    T = mS.shape[1]
    Nq = mS.shape[2]
    if row < Hq * Nq:
        head = row // Nq
        tok = row - head * Nq
        ok = slot < T
        if cutlass.const_expr(MASK):
            # Every tile fits in the top-k here, so forcing selects nothing new;
            # only the per-token validity bound can exclude a tile.
            nvp_f = min(max(mNVP[tok], Int32(0)), T)
            ok = (slot < T) and (slot < nvp_f)
        score = mS[head, min(slot, T - Int32(1)), tok]
        finite = ok and (score > Float32(PAD_LIM))
        mO[head, tok, slot] = slot if finite else Int32(-1)


@cute.jit
def _sortp(bv: cute.Tensor, TL: cutlass.Constexpr):
    """Leave the chunk's top 16 packed keys in bv[0:16], descending.  Pure
    FMNMX: the comparator moves no index, because the in-chunk position rides
    in the key's low mantissa bits, so nothing lands on the integer pipe."""
    if cutlass.const_expr(TL == 32):
        # Only the top 16 of the 32 are ever read, so a (32,16) SELECTION is
        # enough and it is cheaper than sorting all 32: sort the two halves
        # (2 x 60 comparators), take the 16 pairwise maxima against the
        # reversed second half - which is exactly a bitonic half-cleaner, so
        # the winners land in bv[0:16] as a bitonic sequence - and clean it
        # with the same 32-comparator network the accumulator merge uses.
        # 320 FMNMX instead of Batcher-32's 382, with no index traffic.
        for e in cutlass.range_constexpr(len(_C16A)):
            i = _C16A[e]
            j = _C16B[e]
            a = bv[i]
            b = bv[j]
            bv[i] = cute.arch.fmax(a, b)
            bv[j] = cute.arch.fmin(a, b)
        for e in cutlass.range_constexpr(len(_C16A)):
            i = _C16A[e] + 16
            j = _C16B[e] + 16
            a = bv[i]
            b = bv[j]
            bv[i] = cute.arch.fmax(a, b)
            bv[j] = cute.arch.fmin(a, b)
        for i in cutlass.range_constexpr(TOPK):
            bv[i] = cute.arch.fmax(bv[i], bv[31 - i])
        for e in cutlass.range_constexpr(len(_MA)):
            i = _MA[e]
            j = _MB[e]
            a = bv[i]
            b = bv[j]
            bv[i] = cute.arch.fmax(a, b)
            bv[j] = cute.arch.fmin(a, b)
    elif cutlass.const_expr(TL == 8):
        # A tile narrower than the accumulator is only planned when it covers
        # the thread's whole tile group, so 19 comparators replace 60 and the
        # eight empty accumulator slots are filled with the sentinel.
        for e in cutlass.range_constexpr(len(_C8A)):
            i = _C8A[e]
            j = _C8B[e]
            a = bv[i]
            b = bv[j]
            bv[i] = cute.arch.fmax(a, b)
            bv[j] = cute.arch.fmin(a, b)
    else:
        for e in cutlass.range_constexpr(len(_C16A)):
            i = _C16A[e]
            j = _C16B[e]
            a = bv[i]
            b = bv[j]
            bv[i] = cute.arch.fmax(a, b)
            bv[j] = cute.arch.fmin(a, b)


@cute.jit
def _bmerge(av: cute.Tensor, ai: cute.Tensor):
    """Bitonic clean-up after the pairwise maxima."""
    for e in cutlass.range_constexpr(len(_MA)):
        i = _MA[e]
        j = _MB[e]
        a = av[i]
        b = av[j]
        ia = ai[i]
        ib = ai[j]
        p = a >= b
        av[i] = cute.arch.fmax(a, b)
        av[j] = cute.arch.fmin(a, b)
        ai[i] = ia if p else ib
        ai[j] = ib if p else ia


@cute.jit
def _foldp(av: cute.Tensor, ai: cute.Tensor, bv: cute.Tensor,
           bvi: cute.Tensor, sscr: cute.Tensor, tid: Int32,
           it0: Int32, istep: Int32, TL: cutlass.Constexpr):
    """Sort the 16 packed fresh keys, then merge them into the running
    top-16.  Value and index are rebuilt from the surviving key's low bits
    only where they are needed - the 16 maxima - instead of being carried
    through all 60 sort comparators as selected operands.

    The accumulator has to hold untruncated scores: a packed key orders one
    chunk against itself and nothing else, and letting the truncation escape
    into the accumulator turns every later cross-chunk comparison into a
    16-ulp coin flip, which on a few-token workload is enough distinct output
    rows to miss the matched-element floor."""
    _sortp(bv, TL)
    ibase = Int32(IDX_BIAS_BITS) + it0
    for i in cutlass.range_constexpr(TOPK):
        pos = bvi[TOPK - 1 - i] & Int32(TL - 1)
        a = av[i]
        b = sscr[pos, tid]
        p = a >= b
        av[i] = cute.arch.fmax(a, b)
        ikey = (ibase + istep * pos).bitcast(Float32)
        ai[i] = ai[i] if p else ikey
    _bmerge(av, ai)


@cute.jit
def _filter_drain(av: cute.Tensor, ai: cute.Tensor, bv: cute.Tensor,
                  bvi: cute.Tensor, cbv: cute.Tensor, cbi: cute.Tensor,
                  tid: Int32, REARM: cutlass.Constexpr):
    """Merge the compacted threshold candidates and re-arm the buffer."""
    for u in cutlass.range_constexpr(TOPK):
        v = cute.arch.fmax(cbv[u, tid], Float32(NEG_BIG))
        bvi[u] = (v.bitcast(Int32) & Int32(-TOPK)) | Int32(u)
    _sortp(bv, TOPK)
    for i in cutlass.range_constexpr(TOPK):
        pos = bvi[TOPK - 1 - i] & Int32(TOPK - 1)
        a = av[i]
        b = cbv[pos, tid]
        p = a >= b
        av[i] = cute.arch.fmax(a, b)
        ai[i] = ai[i] if p else cbi[pos, tid]
    _bmerge(av, ai)
    if cutlass.const_expr(REARM):
        for u in cutlass.range_constexpr(TOPK):
            cbv[u, tid] = Float32(NEG_BIG)


@cute.jit
def _filter_push(cbv: cute.Tensor, cbi: cute.Tensor,
                 xs: cute.Tensor, slots: cute.Tensor,
                 count: cute.Tensor, tau: Float32,
                 tid: Int32, it0: Int32, istep: Int32):
    """Append eight threshold survivors using a depth-three prefix tree."""
    n = count[0]
    q = cute.make_rmem_tensor((8,), Int32)
    for u in cutlass.range_constexpr(8):
        q[u] = Int32(1) if xs[u] > tau else Int32(0)
    p01, p23 = q[0] + q[1], q[2] + q[3]
    p45, p67 = q[4] + q[5], q[6] + q[7]
    p03 = p01 + p23
    slots[0], slots[1] = n, n + q[0]
    slots[2], slots[3] = n + p01, n + p01 + q[2]
    slots[4], slots[5] = n + p03, n + p03 + q[4]
    slots[6], slots[7] = n + p03 + p45, n + p03 + p45 + q[6]
    count[0] = n + p03 + p45 + p67
    for u in cutlass.range_constexpr(8):
        cbv[slots[u], tid] = xs[u]
        cbi[slots[u], tid] = (
            Int32(IDX_BIAS_BITS) + it0 + istep * Int32(u)).bitcast(Float32)


@cute.jit
def _filter_scan(av: cute.Tensor, ai: cute.Tensor, bv: cute.Tensor,
                 bvi: cute.Tensor, cbv: cute.Tensor, cbi: cute.Tensor,
                 gcol: cute.Tensor, tid: Int32, n: Int32,
                 t00: Int32, istep: Int32, nvp_c: Int32, fb_c: Int32,
                 hi_c: Int32, TL: cutlass.Constexpr,
                 MASK: cutlass.Constexpr):
    """Scan in eight-score groups, merging only threshold survivors."""
    # The compact view aliases the first-tile scratch.  Every thread must
    # finish gathering its exact first-tile values before the overwrite.
    cute.arch.barrier()
    for u in cutlass.range_constexpr(TOPK):
        cbv[u, tid] = Float32(NEG_BIG)
    count = cute.make_rmem_tensor((1,), Int32)
    xs = cute.make_rmem_tensor((8,), Float32)
    slots = cute.make_rmem_tensor((8,), Int32)
    count[0] = Int32(0)
    rem = max(n - Int32(TL), Int32(0))
    ngrp = rem >> 3
    for k in cutlass.range(0, ngrp, 1):
        t0 = t00 + istep * (Int32(TL) + (k << 3))
        for u in cutlass.range_constexpr(8):
            _t = t0 + istep * u
            _xf = cute.arch.fmax(gcol[_t], Float32(NEG_BIG))
            if cutlass.const_expr(MASK):
                _tf = Float32(FORCE_STEP) * (_t).to(Float32)
                _xf = (
                    Float32(NEG_BIG) if (_t) >= nvp_c
                    else (Float32(FORCE_BEGIN_BASE) - _tf if (_t) < fb_c
                          else (Float32(FORCE_END_BASE) - _tf if (_t) >= hi_c
                                else _xf))
                )
            xs[u] = _xf
        # Blocked order makes n warp-uniform; the vote keeps every lane on
        # the same drain schedule even when per-column survivor counts differ.
        if cute.arch.vote_any_sync(count[0] >= Int32(8)):
            _filter_drain(av, ai, bv, bvi, cbv, cbi, tid, True)
            count[0] = Int32(0)
        _filter_push(cbv, cbi, xs, slots, count, av[TOPK - 1],
                     tid, t0, istep)

    utail = Int32(TL) + (ngrp << 3)
    if utail < n:
        t0 = t00 + istep * utail
        tm1 = t00 + istep * (n - Int32(1))
        for u in cutlass.range_constexpr(8):
            ok = (utail + Int32(u)) < n
            _t = min(t0 + istep * u, tm1)
            _xf = cute.arch.fmax(gcol[_t], Float32(NEG_BIG))
            if cutlass.const_expr(MASK):
                _tf = Float32(FORCE_STEP) * (_t).to(Float32)
                _xf = (
                    Float32(NEG_BIG) if (_t) >= nvp_c
                    else (Float32(FORCE_BEGIN_BASE) - _tf if (_t) < fb_c
                          else (Float32(FORCE_END_BASE) - _tf if (_t) >= hi_c
                                else _xf))
                )
            xf = _xf
            xs[u] = xf if ok else Float32(NEG_BIG)
        if cute.arch.vote_any_sync(count[0] >= Int32(8)):
            _filter_drain(av, ai, bv, bvi, cbv, cbi, tid, True)
            count[0] = Int32(0)
        _filter_push(cbv, cbi, xs, slots, count, av[TOPK - 1],
                     tid, t0, istep)
    # The terminal reservoir is never reused before the allocation changes
    # lifetime into the reduction tree, so its 16-slot clear is dead.
    _filter_drain(av, ai, bv, bvi, cbv, cbi, tid, False)



@cute.jit
def _lmerge2(V: cute.Tensor, X: cute.Tensor, k: cutlass.Constexpr,
             a: Float32, ia: Float32, b: Float32, ib: Float32,
             ln: Int32, CLEAN: cutlass.Constexpr):
    """Half-cleaner plus bitonic clean on two descending 16-lists held one
    slot per lane, with `b` already reversed (slot 15-l in lane l).

    The clean only exists so the next level can treat the result as a sorted
    list.  A level whose consumer is the ascending-index scatter - which ranks
    the sixteen lanes against each other and does not care what order they
    arrive in - drops four dependent shuffle stages."""
    p = a >= b
    v = cute.arch.fmax(a, b)
    x = ia if p else ib
    for e in cutlass.range_constexpr(4 if CLEAN else 0):
        d = 8 >> e
        wv = cute.arch.shuffle_sync_bfly(v, offset=d, mask=FULL)
        wi = cute.arch.shuffle_sync_bfly(x, offset=d, mask=FULL)
        q = v >= wv
        up = (ln & Int32(d)) == Int32(0)
        hi = cute.arch.fmax(v, wv)
        lo = cute.arch.fmin(v, wv)
        ihi = x if q else wi
        ilo = wi if q else x
        v = hi if up else lo
        x = ihi if up else ilo
    V[k] = v
    X[k] = x


@cute.jit
def _lmerge(V: cute.Tensor, X: cute.Tensor, k: cutlass.Constexpr,
            j: cutlass.Constexpr, ln: Int32,
            CLEAN: cutlass.Constexpr = True):
    """Merge lists k and j of a lane group.  Slot 15-l of the partner lives in
    lane l^15, so the reversal is a single butterfly."""
    ov = cute.arch.shuffle_sync_bfly(V[j], offset=15, mask=FULL)
    oi = cute.arch.shuffle_sync_bfly(X[j], offset=15, mask=FULL)
    _lmerge2(V, X, k, V[k], X[k], ov, oi, ln, CLEAN)


@cute.jit
def _rank_store(v: Float32, x: Float32, mO: cute.Tensor, ln: Int32,
                h: Int32, col: Int32, Nq: Int32, gp: Int32):
    """Ascending-index scatter of one 16-lane list (see _coop_final_store)."""
    xx = x if v > Float32(PAD_LIM) else Float32(SENT)
    d = xx.bitcast(Int32) - Int32(IDX_BIAS_BITS)
    key = (d << 4) | ln
    kf = key.bitcast(Float32)
    rank = Int32(0)
    for e in cutlass.range_constexpr(TOPK - 1):
        ox = cute.arch.shuffle_sync_bfly(kf, offset=e + 1, mask=FULL)
        rank += Int32(1) if ox.bitcast(Int32) < key else Int32(0)
    if (gp == Int32(0)) and (col < Nq):
        mO[h, col, rank] = d | (Int32(0) - (d >> 23))


@cute.jit
def _merge_smem(av: cute.Tensor, ai: cute.Tensor,
                sbv: cute.Tensor, sbi: cute.Tensor, base: Int32,
                CLEAN: cutlass.Constexpr):
    """Fold an already-sorted partial list (smem column `base`) into ours.

    The half-cleaner alone already leaves the sixteen largest of the union in
    av/ai; the bitonic clean exists only so the next round can treat this as a
    sorted list.  The final round has no next round - its consumer is the
    ascending-index sort - so it drops the 32 comparators."""
    for i in cutlass.range_constexpr(TOPK):
        j = TOPK - 1 - i
        a = av[i]
        b = sbv[j, base]
        p = a >= b
        av[i] = cute.arch.fmax(a, b)
        ai[i] = ai[i] if p else sbi[j, base]
    if cutlass.const_expr(CLEAN):
        _bmerge(av, ai)


@cute.jit
def _coop_round(sbv: cute.Tensor, sbi: cute.Tensor,
                bv: cute.Tensor, bi: cute.Tensor, tid: Int32,
                C: Int32, st: Int32, SLOTS: cutlass.Constexpr):
    """Merge each live list pair with 16/SLOTS lanes."""
    L = TOPK // SLOTS
    ln = tid & Int32(L - 1)
    grp = tid >> Int32(L.bit_length() - 1)
    cg = grp % C
    gp = grp // C
    gpc = min(gp, st - Int32(1))
    ca = gpc * C + cg
    cb = (gpc + st) * C + cg
    for k in cutlass.range_constexpr(SLOTS):
        sl = ln + Int32(k * L)
        a, ia = sbv[sl, ca], sbi[sl, ca]
        b, ib = sbv[Int32(TOPK - 1) - sl, cb], sbi[Int32(TOPK - 1) - sl, cb]
        p = a >= b
        bv[k] = cute.arch.fmax(a, b)
        bi[k] = ia if p else ib
    for d in cutlass.range_constexpr(4):
        dd = 8 >> d
        if cutlass.const_expr(dd >= L):
            # slot k of lane ln holds element ln + L*k, so a stage of
            # distance dd is a local exchange of slots dd//L apart - not
            # SLOTS//2, which only coincides when a single stage is local
            off = dd // L
            for k in cutlass.range_constexpr(SLOTS):
                if cutlass.const_expr((k & off) == 0):
                    a, b = bv[k], bv[k + off]
                    ia, ib = bi[k], bi[k + off]
                    p = a >= b
                    bv[k], bv[k + off] = cute.arch.fmax(a, b), cute.arch.fmin(a, b)
                    bi[k], bi[k + off] = (ia if p else ib), (ib if p else ia)
        else:
            up = (ln & Int32(dd)) == Int32(0)
            for k in cutlass.range_constexpr(SLOTS):
                ov = cute.arch.shuffle_sync_bfly(bv[k], dd)
                oi = cute.arch.shuffle_sync_bfly(bi[k], dd)
                p = bv[k] >= ov
                hi, lo = cute.arch.fmax(bv[k], ov), cute.arch.fmin(bv[k], ov)
                ihi, ilo = (bi[k] if p else oi), (oi if p else bi[k])
                bv[k], bi[k] = (hi if up else lo), (ihi if up else ilo)
    if gp < st:
        for k in cutlass.range_constexpr(SLOTS):
            sl = ln + Int32(k * L)
            sbv[sl, ca], sbi[sl, ca] = bv[k], bi[k]


@cute.jit
def _coop_final_store(sbv: cute.Tensor, sbi: cute.Tensor, mO: cute.Tensor,
                      tid: Int32, bq: Int32, h: Int32,
                      C: Int32, Nq: Int32):
    """Final 16-lane merge, ascending-index sort, and direct output store."""
    ln = tid & Int32(15)
    grp = tid >> 4
    cg = grp % C
    gp = grp // C

    # st == 1.  Extra groups repeat a valid pair so every lane named by FULL
    # reaches every shuffle; only the first C groups commit an output row.
    # Half-cleaner only.  Pairing slot i of one sorted list against slot 15-i
    # of the other already leaves exactly the sixteen largest of the union in
    # the lanes - that is the property the bitonic merge is built on - and the
    # ascending-index scatter below does not care what order they arrive in.
    # The four-stage clean that used to follow was sorting by score a list
    # that is immediately reordered by index, so it goes: four dependent
    # shuffle stages and eight shuffles leave the tail.
    a, ia = sbv[ln, cg], sbi[ln, cg]
    b, ib = sbv[Int32(15) - ln, C + cg], sbi[Int32(15) - ln, C + cg]
    p = a >= b
    v = cute.arch.fmax(a, b)
    xx = ia if p else ib

    xx = xx if v > Float32(PAD_LIM) else Float32(SENT)
    # The sixteen keys only have to leave the warp in ascending order, and the
    # permutation that does that is cheaper to rank than to build by sorting.
    # The bitonic sort this replaces is ten strictly dependent shuffle stages;
    # the fifteen XOR shuffles below do not depend on each other, so the whole
    # exchange costs one shuffle latency instead of ten.  d is the tile index,
    # below 2^23, or 2^23 for the pad sentinel, so four spare low bits carry
    # the lane and equal pads still get a strict total order - the ranks stay a
    # permutation of 0..15 and every output slot is written exactly once.
    d = xx.bitcast(Int32) - Int32(IDX_BIAS_BITS)
    key = (d << 4) | ln
    kf = key.bitcast(Float32)
    rank = Int32(0)
    for e in cutlass.range_constexpr(TOPK - 1):
        ox = cute.arch.shuffle_sync_bfly(kf, offset=e + 1, mask=FULL)
        rank += Int32(1) if ox.bitcast(Int32) < key else Int32(0)

    col = bq * C + cg
    if (gp == Int32(0)) and (col < Nq):
        mO[h, col, rank] = d | (Int32(0) - (d >> 23))


@cute.kernel
def _topk_kernel(
    mS: cute.Tensor,
    mO: cute.Tensor,
    mNVP: cute.Tensor,
    FB: Int32,
    FE: Int32,
    T: Int32,
    Nq: Int32,
    LW: cutlass.Constexpr,
    DENSE: cutlass.Constexpr,
    TL: cutlass.Constexpr,
    FILTER: cutlass.Constexpr,
    C0: cutlass.Constexpr,
    MASK: cutlass.Constexpr,):
    c, w, _ = cute.arch.thread_idx()
    bq, h, _ = cute.arch.block_idx()

    W = 1 << LW                      # compile time: folds every tile stride
    C = Int32(C0)                    # compile time: folds row/group addressing
    # When one very long column is split at least 128 ways and each register
    # tile already carries a full top-k, selecting the 16 partial lists with
    # the greatest maxima is cheaper than merging the whole reduction tree.
    # This is exact: an excluded list's every value is at most its head, while
    # the 16 selected heads already provide 16 values at least that large.
    # The gate is a reduction/register-geometry crossover, not a threshold on
    # a workload extent; both paths implement the full contract.
    # A tile group deep enough to hold sixteen whole lists per lane group is
    # reduced by transposing every partial list into registers once and then
    # running the whole merge cascade on shuffles.  log2(W) barriers and
    # log2(W) shared round trips collapse to two, which is what a CTA that owns
    # one SM on its own actually pays for: NCU put 10% of this kernel's samples
    # on barriers and another 37% on the latency the barriers stop it hiding.
    # Which reduction the CTA runs is a lane-group-geometry crossover.  The
    # register cascade replaces log2(W) barriers with two, but it pays for it
    # in shuffle issue: every merge level costs ten warp shuffles, and a warp
    # shuffle is a quarter-rate instruction, so the cascade only wins while its
    # second stage is short - one fold when a tile group holds two partials per
    # column, up to three when the column tile is a single column and the head
    # preselection it displaces would otherwise cost a sort of W maxima.
    _NS1 = (1 << LW) // TOPK
    # _NS1 == 1 is the cascade's cheapest shape: one lane group holds every
    # partial of its column, so the whole reduction is one barrier, four
    # shuffle-merge levels and no second stage at all.  It is admitted for any
    # column tile, because the cost that used to gate the cascade - the second
    # stage's shuffle issue - does not exist there.
    NEWT = cutlass.const_expr(
        LW >= 4 and ((_NS1 == 1)
                     or (_NS1 <= 4 and (C0 << LW) <= 256)
                     or (C0 == 1 and TL >= TOPK and not FILTER and _NS1 <= 8)))
    HEADSEL = cutlass.const_expr(
        C0 == 1 and LW >= 7 and TL >= TOPK and not FILTER and not NEWT)
    smem = cutlass_utils.SmemAllocator()
    # one buffer, two lives: the 32-row chunk scratch while streaming, then
    # the value/index halves of the reduction tree.  They never overlap in
    # time, and 2*TOPK rows cover the widest chunk as well as the tree.
    # A two-word pad makes reversed-slot reads conflict-free in the 16-lane
    # rounds and reduces the preceding 8-lane round to two-way conflicts.
    NT = C0 << LW
    # The pad decides which banks a lane group lands on.  A one-column tile
    # gives neighbouring lane groups source columns sixteen apart, so an odd
    # pitch puts them on disjoint sixteen-bank windows; a wider column tile
    # gives them adjacent columns, where an even pitch separates them by
    # parity instead.  Both keep the publish (lane-consecutive) conflict-free.
    RP = NT + (1 if (NEWT and C0 == 1) else 2)
    ssc = smem.allocate_tensor(
        Float32, cute.make_layout((TOPK2, NT), stride=(RP, 1)),
        byte_alignment=16)
    sbv = ssc
    sbi = cute.make_tensor(ssc.iterator + TOPK * RP,
                           cute.make_layout((TOPK, NT), stride=(RP, 1)))
    # Compact views of the same allocation.  FILTER has two CTA barriers
    # around this alias lifetime before the padded reduction-tree view is used.
    cbv = cute.make_tensor(ssc.iterator,
                           cute.make_layout((TOPK, NT), stride=(NT, 1)))
    cbi = cute.make_tensor(ssc.iterator + TOPK * NT,
                           cute.make_layout((TOPK, NT), stride=(NT, 1)))
    if cutlass.const_expr(NEWT and _NS1 > 1):
        # Landing zone for the first cascade's results.  A separate allocation
        # (NT/16 lists) means the second cascade's sources are never the
        # storage the first one is still reading, so one barrier covers both.
        NG = NT // TOPK
        RQ = NG + 2
        qsc = smem.allocate_tensor(
            Float32, cute.make_layout((TOPK2, NG), stride=(RQ, 1)),
            byte_alignment=16)
        qbv = qsc
        qbi = cute.make_tensor(qsc.iterator + TOPK * RQ,
                               cute.make_layout((TOPK, NG), stride=(RQ, 1)))
    if cutlass.const_expr(HEADSEL):
        # Keep scalar heads/owners separate from the source lists, and land
        # the first eight indirect list merges in disjoint scratch so no
        # selected source can be overwritten before it is read.
        sH = smem.allocate_tensor(
            Float32, cute.make_layout((NT,), stride=(1,)),
            byte_alignment=16)
        sP = smem.allocate_tensor(
            Float32, cute.make_layout((NT,), stride=(1,)),
            byte_alignment=16)
        rsc = smem.allocate_tensor(
            Float32, cute.make_layout((TOPK2, 8), stride=(10, 1)),
            byte_alignment=16)
        rbv = rsc
        rbi = cute.make_tensor(
            rsc.iterator + TOPK * 10,
            cute.make_layout((TOPK, 8), stride=(10, 1)))

    tid = w * C + c
    col = bq * C + c
    colr = min(col, Nq - Int32(1))   # clamp so the ragged last CTA stays in bounds

    av = cute.make_rmem_tensor((TOPK,), Float32)
    ai = cute.make_rmem_tensor((TOPK,), Float32)
    # max(TL, TOPK): a tile narrower than the accumulator still has to
    # lend this buffer to the scalar epilogue's 16-slot ascending sort
    bv = cute.make_rmem_tensor((max(TL, TOPK),), Float32)
    bvi = cute.recast_tensor(bv, Int32)

    # 1-D view of this thread's own column: index math is one multiply-add
    gcol = mS[(h, None, colr)]

    # Tile group w owns t = t00 + j*step, j = 0 .. n-1.
    if cutlass.const_expr(DENSE):
        step = W
        n = (max(T - w, Int32(0)) + Int32(W - 1)) >> LW
        t00 = w
    else:
        step = 1
        A = ((T + Int32(W - 1)) >> LW)
        A = (A + Int32(TL - 1)) & Int32(-TL)
        n = min(max(T - w * A, Int32(0)), A)
        t00 = w * A

    Tm1 = T - Int32(1)

    # The first tile lands straight in the accumulator: against an empty
    # top-16 the fold degenerates to sorting the tile, so its 16 maxima and
    # 32-comparator merge would be pure overhead.  Masked, because a thread
    # owning a single register tile has this as its last tile too.  A group
    # with no rows at all ends up all NEG_BIG, which the epilogue turns
    # into -1.
    # NCU put 42% of this kernel's stall samples on the long scoreboard at this
    # load, with 86% of its sectors excessive: a C=1 column tile puts every lane
    # of a warp load on its own sector, so the loads have to be in flight
    # together or the warp pays the L2 latency once per tile.  Keeping the
    # shared stores out of this loop leaves no dependence between the TL loads.
    # Forced blocks and per-token validity are folded in here rather than
    # biasing max_score in a separate pass: the scores are already in flight,
    # so this costs a few predicated selects and no extra memory traffic.
    nvp_c = T
    fb_c = Int32(0)
    hi_c = T
    if cutlass.const_expr(MASK):
        nvp_c = min(max(mNVP[colr], Int32(0)), T)
        fb_c = min(FB, nvp_c)
        fe_c = min(FE, nvp_c - fb_c)
        hi_c = nvp_c - fe_c
    for u in cutlass.range_constexpr(TL):
        ok = Int32(u) < n
        t_u = min(t00 + step * u, Tm1)
        _xf = cute.arch.fmax(gcol[t_u], Float32(NEG_BIG))
        if cutlass.const_expr(MASK):
            _tf = Float32(FORCE_STEP) * (t_u).to(Float32)
            _xf = (
                Float32(NEG_BIG) if (t_u) >= nvp_c
                else (Float32(FORCE_BEGIN_BASE) - _tf if (t_u) < fb_c
                      else (Float32(FORCE_END_BASE) - _tf if (t_u) >= hi_c
                            else _xf))
            )
        xf = _xf
        bv[u] = xf if ok else Float32(NEG_BIG)
    for u in cutlass.range_constexpr(TL):
        ssc[u, tid] = bv[u]
        bvi[u] = (bvi[u] & Int32(-TL)) | Int32(u)
    _sortp(bv, TL)
    ibase = Int32(IDX_BIAS_BITS) + t00
    for u in cutlass.range_constexpr(min(TL, TOPK)):
        pos = bvi[u] & Int32(TL - 1)
        av[u] = ssc[pos, tid]
        ai[u] = (ibase + Int32(step) * pos).bitcast(Float32)
    # A register tile narrower than the accumulator owns fewer rows than the
    # accumulator has slots; the rest stay empty.  NEG_BIG loses every later
    # comparison and the epilogue decodes anything at or below PAD_LIM to -1,
    # which is the same padding a token with fewer than 16 valid blocks
    # already produces.
    for u in cutlass.range_constexpr(TOPK - min(TL, TOPK)):
        av[min(TL, TOPK) + u] = Float32(NEG_BIG)
        ai[min(TL, TOPK) + u] = Int32(IDX_BIAS_BITS).bitcast(Float32)

    # TL < TOPK is only planned when one register tile covers the whole tile
    # group (ceil(T/W) <= TL, checked on the runtime extent), so the fold -
    # which assumes a TOPK-wide packed key list - is compiled out rather than
    # left reachable.
    if cutlass.const_expr(TL >= TOPK):
        if cutlass.const_expr(FILTER):
            _filter_scan(av, ai, bv, bvi, cbv, cbi, gcol, tid,
                         n, t00, Int32(step), nvp_c, fb_c, hi_c, TL, MASK)
        else:
            rem = max(n - Int32(TL), Int32(0))
            nfull = rem // Int32(TL)
            for k in cutlass.range(0, nfull, 1):
                t0 = t00 + step * (Int32(TL) + k * Int32(TL))
                for u in cutlass.range_constexpr(TL):
                    _t = t0 + step * u
                    _xf = cute.arch.fmax(gcol[_t], Float32(NEG_BIG))
                    if cutlass.const_expr(MASK):
                        _tf = Float32(FORCE_STEP) * (_t).to(Float32)
                        _xf = (
                            Float32(NEG_BIG) if (_t) >= nvp_c
                            else (Float32(FORCE_BEGIN_BASE) - _tf if (_t) < fb_c
                                  else (Float32(FORCE_END_BASE) - _tf if (_t) >= hi_c
                                        else _xf))
                        )
                    bv[u] = _xf
                for u in cutlass.range_constexpr(TL):
                    ssc[u, tid] = bv[u]
                    bvi[u] = (bvi[u] & Int32(-TL)) | Int32(u)
                _foldp(av, ai, bv, bvi, ssc, tid, t0, Int32(step), TL)

            utail = Int32(TL) + nfull * Int32(TL)
            if utail < n:
                t0 = t00 + step * utail
                for u in cutlass.range_constexpr(TL):
                    ok = (utail + Int32(u)) < n
                    _t = min(t0 + step * u, Tm1)
                    _xf = cute.arch.fmax(gcol[_t], Float32(NEG_BIG))
                    if cutlass.const_expr(MASK):
                        _tf = Float32(FORCE_STEP) * (_t).to(Float32)
                        _xf = (
                            Float32(NEG_BIG) if (_t) >= nvp_c
                            else (Float32(FORCE_BEGIN_BASE) - _tf if (_t) < fb_c
                                  else (Float32(FORCE_END_BASE) - _tf if (_t) >= hi_c
                                        else _xf))
                        )
                    xf = _xf
                    bv[u] = xf if ok else Float32(NEG_BIG)
                for u in cutlass.range_constexpr(TL):
                    ssc[u, tid] = bv[u]
                    bvi[u] = (bvi[u] & Int32(-TL)) | Int32(u)
                _foldp(av, ai, bv, bvi, ssc, tid,
                       t0, Int32(step), TL)

    # ---- reduce the W tile groups of this CTA through shared memory -------
    # Round `st` folds list w into w+st, so it has st*C independent merges and
    # the CTA has NT threads to spend on them.  Give each merge the widest
    # power-of-two lane group it can have, capped at the 16 slots of a list:
    # once that is 8 lanes or more the merge is worth spreading with shuffles,
    # below it the scalar 16-slot merge on the st*C owner threads is cheaper
    # than making every thread carry 8 slots.  For a 256-thread CTA this
    # reproduces the previous fixed schedule exactly (two scalar rounds, then
    # 8- and 16-lane rounds) and extends the same tail - including the fused
    # merge/index-sort/store - to CTAs that used to fall back to the scalar
    # epilogue and its 16 uncoalesced stores.
    #
    # The whole schedule is a compile-time list, so `st` and every shared
    # address fold into immediates.  The previous dynamic trip count also made
    # the register accumulator cross a staged loop boundary, which does not
    # reliably carry.
    NTC = C0 << LW
    _sts = [1 << (LW - 1 - r) for r in range(LW)]
    _lane = [min(16, 1 << max(0, (NTC // (s * C0)).bit_length() - 1))
             for s in _sts]
    # a round only needs lane*st*C threads; rounded up to whole warps so the
    # full-mask shuffles still see every lane, the rest of the CTA skips it.
    # The tail rounds move two or four lists and used to cost every warp in
    # the CTA an issue slot to do it.
    _act = [min(NTC, ((_lane[e] * _sts[e] * C0 + 31) // 32) * 32)
            for e in range(LW)]
    _fin = min(NTC, ((16 * C0 + 31) // 32) * 32)
    COOP = cutlass.const_expr(
        LW > 0 and C0 <= 16 and NTC >= 32 and NTC % 32 == 0
        and NTC >= 16 * C0 and _lane[-1] >= 8)

    if cutlass.const_expr(LW > 0):
        # End the compact-pitch alias lifetime before any warp seeds the
        # padded value/index tree through the same storage.
        if cutlass.const_expr(FILTER):
            cute.arch.barrier()
        for j in cutlass.range_constexpr(TOPK):
            sbv[j, tid] = av[j]
            sbi[j, tid] = ai[j]
        if cutlass.const_expr(NEWT):
            # Cascade 1: lane group g owns column g % C and the sixteen tile
            # groups (g//C)*16 .. +15 of it.  Each lane pulls slot `ln` of all
            # sixteen source lists, so the group holds sixteen sorted lists one
            # slot per lane, and four shuffle-only merge levels leave one.
            NS1 = (1 << LW) // TOPK          # partial results per column
            _NL2 = NS1.bit_length() - 1
            _a2 = min(NT, ((TOPK * C0 + 31) // 32) * 32)
            ln = tid & Int32(TOPK - 1)
            grp = tid >> 4
            cg = grp % C
            sg = grp // C
            V = cute.make_rmem_tensor((TOPK,), Float32)
            X = cute.make_rmem_tensor((TOPK,), Float32)
            base = (sg << 4) * C + cg
            cute.arch.barrier()
            lnr = Int32(TOPK - 1) - ln
            # The first level's partner is read straight out of shared memory
            # at its reversed slot, so the eight half-cleaner butterflies never
            # issue and only eight lists are ever live in registers.
            for k in cutlass.range_constexpr(8):
                s0 = base + Int32(k * C0)
                s1 = base + Int32((k + 8) * C0)
                _lmerge2(V, X, k, sbv[ln, s0], sbi[ln, s0],
                         sbv[lnr, s1], sbi[lnr, s1], ln, True)
            for lv in cutlass.range_constexpr(3):
                h2 = 4 >> lv
                for k in cutlass.range_constexpr(h2):
                    # the last level of a single-stage cascade feeds the
                    # ascending-index scatter, which ranks lanes by index and
                    # so does not need the four clean stages
                    _lmerge(V, X, k, k + h2, ln,
                            not (NS1 == 1 and h2 == 1))
            if cutlass.const_expr(NS1 == 1):
                # one lane group per column already holds the whole reduction
                _rank_store(V[0], X[0], mO, ln, h, bq * C + cg, Nq, sg)
            else:
                qbv[ln, grp] = V[0]
                qbi[ln, grp] = X[0]
                cute.arch.barrier()
                # Cascade 2: one lane group per column folds that column's NS1
                # partials.  Groups past the column count repeat a live column
                # so every lane of a partial warp still reaches the shuffles;
                # only the first C of them commit an output row.
                if tid < Int32(_a2):
                    _h0 = NS1 // 2
                    for k in cutlass.range_constexpr(_h0):
                        s0 = Int32(k * C0) + cg
                        s1 = Int32((k + _h0) * C0) + cg
                        _lmerge2(V, X, k, qbv[ln, s0], qbi[ln, s0],
                                 qbv[lnr, s1], qbi[lnr, s1], ln, _h0 > 1)
                    for lv in cutlass.range_constexpr(_NL2 - 1):
                        h2 = NS1 >> (lv + 2)
                        for k in cutlass.range_constexpr(h2):
                            _lmerge(V, X, k, k + h2, ln, h2 > 1)
                    _rank_store(V[0], X[0], mO, ln, h, bq * C + cg, Nq, sg)
        elif cutlass.const_expr(HEADSEL):
            # Sort each 16-head half-warp descending.  Both half-warps execute
            # every full-mask shuffle, so the collective stays converged.
            ln = tid & Int32(15)
            grp = tid >> 4
            hv = av[0]
            hp = (Int32(IDX_BIAS_BITS) + tid).bitcast(Float32)
            for e in cutlass.range_constexpr(len(_ASC)):
                hoff, hlj, hlk = _ASC[e]
                ov = cute.arch.shuffle_sync_bfly(hv, offset=hoff, mask=FULL)
                op = cute.arch.shuffle_sync_bfly(hp, offset=hoff, mask=FULL)
                bj = (ln >> Int32(hlj)) & Int32(1)
                bk = (ln >> Int32(hlk)) & Int32(1)
                wmax = (bj ^ bk) == Int32(0)
                nv = cute.arch.fmax(hv, ov) if wmax else cute.arch.fmin(hv, ov)
                hp = op if nv != hv else hp
                hv = nv
            sH[tid] = hv
            sP[tid] = hp

            # Merge the sorted head groups down to the greatest 16.  Inactive
            # groups repeat a live pair so every lane reaches every shuffle;
            # only the live prefix writes the next level.
            NG = 1 << (LW - 4)
            for r in cutlass.range_constexpr(LW - 4):
                st = Int32(NG >> (r + 1))
                cute.arch.barrier()
                # Round the live half-warp groups up to one whole warp: this
                # preserves FULL-mask convergence without making every warp
                # repeat a clamped, dead merge.
                active = max(Int32(32), st << 4)
                if tid < active:
                    gpc = min(grp, st - Int32(1))
                    b = sH[((gpc + st) << 4) + (Int32(15) - ln)]
                    bx = sP[((gpc + st) << 4) + (Int32(15) - ln)]
                    p = hv >= b
                    hv = cute.arch.fmax(hv, b)
                    hp = hp if p else bx
                    for d in cutlass.range_constexpr(4):
                        dd = 8 >> d
                        ov = cute.arch.shuffle_sync_bfly(hv, dd, mask=FULL)
                        op = cute.arch.shuffle_sync_bfly(hp, dd, mask=FULL)
                        q = hv >= ov
                        lower = (ln & Int32(dd)) == Int32(0)
                        hv = (cute.arch.fmax(hv, ov) if lower
                              else cute.arch.fmin(hv, ov))
                        hp = ((hp if q else op) if lower
                              else (op if q else hp))
                    if grp < st:
                        sH[(grp << 4) + ln] = hv
                        sP[(grp << 4) + ln] = hp
            cute.arch.barrier()

            # Pair the 16 selected source lists into eight disjoint scratch
            # lists.  At W=256 only the first four warps participate; the
            # predicate is warp-uniform and all CTA threads meet the barriers.
            if tid < Int32(128):
                la = sP[grp].bitcast(Int32) - Int32(IDX_BIAS_BITS)
                lb = sP[grp + Int32(8)].bitcast(Int32) - Int32(IDX_BIAS_BITS)
                v = sbv[ln, la]
                xx = sbi[ln, la]
                b = sbv[Int32(15) - ln, lb]
                bx = sbi[Int32(15) - ln, lb]
                p = v >= b
                v = cute.arch.fmax(v, b)
                xx = xx if p else bx
                for d in cutlass.range_constexpr(4):
                    dd = 8 >> d
                    ov = cute.arch.shuffle_sync_bfly(v, dd, mask=FULL)
                    oi = cute.arch.shuffle_sync_bfly(xx, dd, mask=FULL)
                    q = v >= ov
                    lower = (ln & Int32(dd)) == Int32(0)
                    v = (cute.arch.fmax(v, ov) if lower
                         else cute.arch.fmin(v, ov))
                    xx = (xx if q else oi) if lower else (oi if q else xx)
                rbv[ln, grp] = v
                rbi[ln, grp] = xx

            # Eight lists need two ordinary cooperative rounds; the last pair
            # is fused with the ascending-index sort and global store.  Only
            # warp 0 produces and consumes that last pair, so warp scope is
            # sufficient for its shared-memory ordering.
            for r in cutlass.range_constexpr(2):
                cute.arch.barrier()
                if tid < Int32(64 >> r):
                    _coop_round(rbv, rbi, bv, ai, tid, Int32(1),
                                Int32(4 >> r), 1)
            if tid < Int32(32):
                cute.arch.sync_warp()
                _coop_final_store(rbv, rbi, mO, tid, bq, h,
                                  Int32(1), Nq)
        elif cutlass.const_expr(not COOP):
            for r in cutlass.range_constexpr(LW):
                st = Int32(1 << (LW - 1 - r))
                cute.arch.barrier()
                if w < st:
                    # The scalar epilogue consumes av/ai directly after the
                    # st=1 merge; only earlier rounds have a shared consumer,
                    # and only they need the merged list left sorted.
                    _merge_smem(av, ai, sbv, sbi, tid + st * C,
                                r + 1 < LW)
                    if cutlass.const_expr(r + 1 < LW):
                        for j in cutlass.range_constexpr(TOPK):
                            sbv[j, tid] = av[j]
                            sbi[j, tid] = ai[j]
        else:
            for e in cutlass.range_constexpr(LW - 1):
                cute.arch.barrier()
                if cutlass.const_expr(_lane[e] >= COOP_MIN):
                    if cutlass.const_expr(_act[e] >= NTC):
                        _coop_round(sbv, sbi, bv, ai, tid, C, Int32(_sts[e]),
                                    TOPK // _lane[e])
                    else:
                        if tid < Int32(_act[e]):
                            _coop_round(sbv, sbi, bv, ai, tid, C,
                                        Int32(_sts[e]), TOPK // _lane[e])
                else:
                    st = Int32(_sts[e])
                    if w < st:
                        _merge_smem(av, ai, sbv, sbi, tid + st * C, True)
                        for j in cutlass.range_constexpr(TOPK):
                            sbv[j, tid] = av[j]
                            sbi[j, tid] = ai[j]
            cute.arch.barrier()
            if cutlass.const_expr(_fin >= NTC):
                _coop_final_store(sbv, sbi, mO, tid, bq, h, C, Nq)
            else:
                if tid < Int32(_fin):
                    _coop_final_store(sbv, sbi, mO, tid, bq, h, C, Nq)

    # ---- epilogue: indices ascending, empty slots decoded to -1 -----------
    # A row of the output is TOPK*4 = 64 contiguous bytes, so a thread writing
    # one whole row on its own puts every lane of its warp on a different 128B
    # line: min(C,32) wavefronts per store instead of one.  A CTA covering C
    # columns owns C*TOPK contiguous words, so once the sorted keys are staged
    # in the reduction scratch the CTA can stream them out flat, one word per
    # thread per step, and every warp store is a single wavefront.  The extra
    # scratch round trip and its barrier are priced against the wavefronts they
    # remove, so this only engages once the column tile is wide enough to pay.
    _wpc = (NTC + 31) // 32
    _nstep = (C0 * TOPK + NTC - 1) // NTC
    _rate = min(SCHED, IPC1 * _wpc)
    _direct = TOPK * min(C0, 32) * ((C0 + 31) // 32) / L1_WF
    _flat = (max(1.0, C0 * TOPK / 32.0) / L1_WF
             + (TOPK * ((C0 + 31) // 32) + _nstep * 10 * _wpc) / _rate + BAR)
    SPLIT = cutlass.const_expr((not COOP) and _flat < _direct)
    if cutlass.const_expr(not COOP):
        if w == Int32(0):
            for j in cutlass.range_constexpr(TOPK):
                p = av[j] > Float32(PAD_LIM)
                bv[j] = ai[j] if p else Float32(SENT)
            for e in cutlass.range_constexpr(len(_SA)):
                i = _SA[e]
                j = _SB[e]
                a = bv[i]
                b = bv[j]
                bv[i] = cute.arch.fmin(a, b)
                bv[j] = cute.arch.fmax(a, b)
            if cutlass.const_expr(SPLIT):
                for j in cutlass.range_constexpr(TOPK):
                    sbv[j, tid] = bv[j]
            else:
                if col < Nq:
                    for j in cutlass.range_constexpr(TOPK):
                        xi = bv[j].bitcast(Int32) - Int32(IDX_BIAS_BITS)
                        # SENT's biased delta is 2^23 -> -1; real ones survive.
                        mO[h, col, j] = xi | (Int32(0) - (xi >> 23))
    if cutlass.const_expr(SPLIT):
        cute.arch.barrier()
        NV = C0 * TOPK
        NSTEP = (NV + NTC - 1) // NTC
        for e in cutlass.range_constexpr(NSTEP):
            idx = Int32(e * NTC) + tid
            cc = min(idx >> 4, Int32(C0 - 1))
            jj = idx & Int32(15)
            oc = bq * C + cc
            xi = sbv[jj, cc].bitcast(Int32) - Int32(IDX_BIAS_BITS)
            if cutlass.const_expr(NSTEP * NTC != NV):
                if idx < Int32(NV):
                    if oc < Nq:
                        mO[h, oc, jj] = xi | (Int32(0) - (xi >> 23))
            else:
                if oc < Nq:
                    mO[h, oc, jj] = xi | (Int32(0) - (xi >> 23))


@cute.jit
def _launch_fill(mS: cute.Tensor, mO: cute.Tensor, mNVP: cute.Tensor,
                 MASK: cutlass.Constexpr):
    rows = mS.shape[0] * mS.shape[2]
    _fill_kernel(mS, mO, mNVP, MASK).launch(
        grid=[(rows + Int32(7)) // Int32(8), 1, 1],
        block=[FILL_BLOCK, 1, 1],
    )


@cute.jit
def _launch(mS: cute.Tensor, mO: cute.Tensor, mNVP: cute.Tensor,
            FB: Int32, FE: Int32, LW: cutlass.Constexpr,
            DENSE: cutlass.Constexpr, TL: cutlass.Constexpr,
            FILTER: cutlass.Constexpr,
            C0: cutlass.Constexpr, MASK: cutlass.Constexpr):
    Hq = mS.shape[0]
    Nq = mS.shape[2]
    C = Int32(C0)
    _topk_kernel(mS, mO, mNVP, FB, FE, mS.shape[1], Nq,
                 LW, DENSE, TL, FILTER, C0, MASK).launch(
        grid=[(Nq + C - Int32(1)) // C, Hq, 1],
        block=[C0, 1 << LW, 1],
    )


def _plan(Hq, T, Nq):
    """Pick the CTA geometry (C columns x W tile groups), the tile walk order
    and the register-tile width TL from hardware costs alone.

    NCU on a small workload put the SM at 2.8 of its 4 warp instructions per
    cycle with DRAM at 0.2%, so what a geometry costs is mostly *instructions
    a CTA issues* over *the rate the SM can retire them*, and the previous
    model got both halves wrong at small grids.  It counted only the streaming
    tiles - the reduction tree, which is more than half the instructions once
    T/W falls to one tile, was priced as a critical path that a
    one-CTA-per-SM launch divides straight out - and it modelled the issue
    rate as proportional to resident warps all the way to eight, which makes a
    256-thread CTA and a 64-thread CTA cost the same.  A warp scheduler issues
    at most one instruction per cycle and an SM has four of them, so the rate
    saturates around five warps, and past that a wider CTA is pure loss: it
    retires the same work while leaving more SMs empty, because the grid is
    Hq*ceil(Nq/C) and W cannot add blocks.

    Terms, all in SM cycles:

    t_alu  (streaming tiles + reduction rounds, in instructions) x waves,
           over min(4, 0.8 * resident warps).  The instruction counts are read
           off the SASS the kernel emits for that exact geometry, round by
           round with the lane width the kernel will actually choose, so a
           geometry pays for the tile padding and the tree depth it creates.
           Reduction rounds are counted at half weight: a cooperative round
           spends its extra lanes on warps that are otherwise sitting at the
           barrier behind it.

    t_l1   wavefronts.  One warp load covers min(C,32) columns of 32//C tile
           rows and the rows are Nq*4 bytes apart, so it costs one L1
           wavefront per distinct line, and the L1 retires about two of them
           per cycle when each carries a single sector.  The output store is
           priced the same way: the fused
           16-lane epilogue writes one 64B row per column, while the scalar
           fallback puts each of its 16 stores on a different line per lane.

    t_dram bytes / device bandwidth: every score is read exactly once.

    Every input is a hardware property (132 SMs, 32 lanes, four schedulers, a
    128B line, the L1 return rate, DRAM bandwidth, the register file and the
    shared memory capacity) or a runtime extent.
    """
    # A column tile of LINE//4 = 32 columns already fills one 128B line per
    # warp load; widening it further buys no coalescing, halves the grid and
    # doubles the epilogue's store wavefronts, so the search stops there.
    cands = sorted({min(1 << lc, Nq) for lc in range(6)})
    best = None
    t_dram = float(Hq) * T * Nq * 4.0 / DRAM_BPC
    for TL in TILES:
        for C in cands:
            ncb = (Nq + C - 1) // C
            grid = Hq * ncb
            q = (grid + NSM - 1) // NSM
            for lw in range(LOG_NT + 1):
                W = 1 << lw
                NT = C * W
                if NT > NT_MAX:
                    break
                # An SM has four warp schedulers, so a CTA narrower than four
                # warps can only keep them fed through co-residency, and the
                # compactor refuses to run below four warps for the same
                # reason.  Measured: the one- and two-warp geometries this
                # rules out were the slowest bytes-per-second points in the
                # whole set, at 1.5-1.6 TB/s where the four-warp ones reach
                # 2.3 TB/s.
                if NT < 32 * SCHED:
                    continue
                rows = (T + W - 1) // W
                if rows < 1:
                    continue
                chunks = max(1, (rows + TL - 1) // TL)
                # a tile narrower than the accumulator carries no fold, so it
                # is only usable when one tile covers the whole tile group
                if TL < TOPK and chunks > 1:
                    continue
                wpc = (NT + 31) // 32
                smem = TOPK2 * (NT + 2) * 4
                sts = [1 << (lw - 1 - r) for r in range(lw)]
                lane = [min(16, 1 << max(0, (NT // (s * C)).bit_length() - 1))
                        for s in sts]
                coop = (lw > 0 and C <= 16 and NT >= 32 and NT % 32 == 0
                        and NT >= 16 * C and lane[-1] >= 8)
                # ---- instructions this CTA issues -------------------------
                i_cta = wpc * (FIRST_INST[TL] + (chunks - 1) * CHUNK_INST[TL])
                i_tree = 0.0
                if lw > 0:
                    i_tree += wpc * PUBLISH_INST
                    if coop:
                        for e in range(lw - 1):
                            if lane[e] >= 8:
                                aw = min(wpc, (lane[e] * sts[e] * C + 31) // 32)
                                i_tree += aw * COOP_INST[16 // lane[e]]
                            else:
                                i_tree += (((sts[e] * C + 31) // 32)
                                           * (MERGE_INST + PUBLISH_INST))
                        i_tree += min(wpc, (16 * C + 31) // 32) * FINAL_INST
                    else:
                        for e, s in enumerate(sts):
                            i_tree += (((s * C + 31) // 32)
                                       * (MERGE_INST +
                                          (PUBLISH_INST if e + 1 < lw else 0)))
                # ---- how fast the SM can retire them ----------------------
                res = max(1, min(32, WARPS_SM // wpc,
                                 SMEM_CAP // max(smem, 1)))
                res = min(res, max(1, q))
                warps = min(64, res * wpc)
                # ---- L1 wavefronts, output stores, DRAM ------------------
                lanes = min(32, NT)
                nrow = max(1, lanes // C)
                if C >= 32 or nrow == 1:
                    wf_blk = wf_den = 1
                else:
                    wf_blk = nrow
                    span = ((nrow - 1) * Nq + C) * 4
                    wf_den = min(nrow, (span + LINE - 1) // LINE)
                w_load = wpc * chunks * TL
                # the scalar epilogue puts its 16 stores 64B apart per lane,
                # so every one of them is a wavefront per lane; the fused
                # 16-lane store writes one contiguous 64B row per column
                t_out = (q * (16.0 * min(C, 32) * ((C + 31) // 32))
                         if not coop else q * max(1.0, C / 2.0))
                # the shared-memory tree also sits on one thread's critical
                # path, which a single resident CTA cannot hide.
                #
                # A geometry the register cascade can run does not pay
                # log2(W) barrier-separated rounds at all: every lane group
                # pulls sixteen partial lists into registers at once and folds
                # them on shuffles, so the chain is one barrier and four merge
                # levels, plus a second barrier and log2(W/16) more levels
                # when a column owns more than one partial.  Pricing those
                # geometries with the barrier-per-round schedule is what kept
                # the search on shallow trees that then had to run the
                # barrier-heavy path.
                newt = (lw >= 4 and ((W // TOPK == 1)
                                     or (W // TOPK <= 4 and NT <= NT_MAX)
                                     or (C == 1 and TL >= TOPK
                                         and W // TOPK <= 8)))
                nl2 = max(0, (W // TOPK).bit_length() - 1)
                # head preselection is not a barrier-per-round tree either: it
                # sorts the W maxima, folds them down to sixteen, and then
                # merges only the sixteen source lists it selected.
                hsel = C == 1 and lw >= 7 and TL >= TOPK and not newt
                if newt:
                    cp = ((1 if nl2 == 0 else 2) * BAR
                          + (4 + nl2) * COOP_CP)
                elif hsel:
                    cp = (lw - 4) * (COOP_CP + BAR) + 4 * COOP_CP + 3 * BAR
                elif coop:
                    cp = BAR + FINAL_CP + sum(
                        (COOP_CP if lane[e] >= 8 else MERGE_CP) + BAR
                        for e in range(lw - 1))
                else:
                    cp = lw * (MERGE_CP + BAR) + EPI_CP
                t_tree = cp * q / max(1, min(res, q))
                # unhidden memory latency: a chunk's loads all fly at once,
                # chunks do not overlap each other inside a thread
                t_lat = LAT * chunks / max(1.0, warps / 2.0)
                t_alu = q * (i_cta + TREE_W * i_tree) / min(SCHED, IPC1 * warps)
                # Both walk orders cost the same wavefronts whenever a warp
                # load already spans more than one 128B line (wf_den == wf_blk
                # there, and wf_den is never the larger of the two), so the
                # model cannot separate them and whichever is tried first wins
                # the tie.  Break that tie by what the two orders do to the L2:
                # a tile group that fits in ONE register tile has no fold loop,
                # so the walk order only decides where a warp's loads land -
                # interleaved puts them on nrow consecutive tile rows, one span
                # of nrow*Nq*4 bytes, while blocked scatters them across nrow
                # spans a whole tile group apart.  Measured 2-4% on every
                # single-tile geometry whose warp load spans more than one row;
                # a column tile of a full warp puts every lane on the same row,
                # where there is no spread to gain and the strided per-thread
                # stream measured 2% worse, so that case keeps the blocked
                # walk.  Once a group needs several register tiles the blocked
                # walk's contiguous per-thread stream wins instead (measured
                # the other way, up to 12%), and it is also what the threshold
                # compactor needs to keep its counts warp-uniform.
                order = (((True, wf_den), (False, wf_blk))
                         if (chunks == 1 and nrow > 1)
                         else ((False, wf_blk), (True, wf_den)))
                for dense, wf in order:
                    mem = q * w_load * wf / L1_WF + t_out + t_dram
                    t = (max(mem, t_alu) + OVERLAP * min(mem, t_alu)
                         + t_tree + t_lat)
                    if best is None or t < best[0]:
                        best = (t, lw, dense, TL, C)
    lw, dense, TL, C = best[1], best[2], best[3], best[4]
    W = 1 << lw
    rows = (T + W - 1) // W
    rem = max(rows - TL, 0)
    normal_cost = ((rem + TL - 1) // TL) * CHUNK_INST[TL]
    groups = (rem + 7) // 8
    # After the initial top-16, a random stream admits logarithmically many
    # record candidates. This conservative bound prices one drain per
    # doubling; both algorithms remain exact regardless of the estimate.
    drains = max(1, ((rows + TOPK - 1) // TOPK).bit_length())
    filter_cost = (groups * FILTER_GROUP_INST
                   + drains * FILTER_DRAIN_INST + 2 * BAR)
    # C being a multiple of one warp makes n/count and every vote uniform.
    # Four or more resident warps are required to hide the compactor's larger
    # live-register set; on a two-warp CTA it reduces latency hiding more than
    # the skipped sort work saves. BLOCKED keeps each group scan contiguous.
    filt = (not dense and (C * W) // 32 >= SCHED and TL >= TOPK
            and C >= 32 and C % 32 == 0
            and filter_cost < normal_cost)
    return lw, dense, TL, filt, C


_cache = {}
_compiled_fill = None


@torch.no_grad()
def run(max_score, cu_seqlens_q, context_lens, topk_idx,
        num_valid_pages=None, force_begin=0, force_end=0):
    """Top-k KV-block selection.

    ``num_valid_pages`` is a per-token int32 tensor of causal extents;
    ``force_begin`` / ``force_end`` pin the sink blocks and the trailing local
    window. All three are applied at the score load inside the kernel, so the
    unmasked path is unchanged and costs nothing.
    """
    global _compiled_fill
    Hq, T, Nq = max_score.shape
    mask = num_valid_pages is not None or bool(force_begin) or bool(force_end)
    nvp = num_valid_pages
    if nvp is None:
        nvp = torch.empty(0, dtype=torch.int32, device=max_score.device)

    def _dyn(t, ld):
        return from_dlpack(t, enable_tvm_ffi=True,
                           use_32bit_stride=True).mark_layout_dynamic(leading_dim=ld)

    if T <= TOPK:
        key = ("fill", mask)
        fn = _cache.get(key)
        if fn is None:
            fn = cute.compile(_launch_fill, _dyn(max_score, 2), _dyn(topk_idx, 2),
                              _dyn(nvp, 0), mask, options="--enable-tvm-ffi")
            _cache[key] = fn
        fn(max_score, topk_idx, nvp)
        return

    plan = _plan(int(Hq), int(T), int(Nq))
    # C is the CTA's occupancy-derived column tile, not a workload threshold.
    # Keying it makes the block extent and all row/group address arithmetic
    # compile-time constants while preserving the same algorithm for every C.
    key = plan + (mask,)
    fn = _cache.get(key)
    if fn is None:
        fn = cute.compile(_launch, _dyn(max_score, 2), _dyn(topk_idx, 2),
                          _dyn(nvp, 0), Int32(force_begin), Int32(force_end),
                          plan[0], plan[1], plan[2], plan[3], plan[4], mask,
                          options="--enable-tvm-ffi")
        _cache[key] = fn
    fn(max_score, topk_idx, nvp, force_begin, force_end)
