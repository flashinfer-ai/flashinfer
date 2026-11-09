# SPDX-License-Identifier: Apache-2.0
"""GDN ReplaySSM MTP decode step (1 <= T <= 8) on the UMMA / TMEM design (Blackwell).

Opt-in sibling of ``gdn_replay_mtp_umma`` (``backend="umma_ring64"`` of ``gated_delta_rule_mtp_ucache_flush``):
the same fp16-state arm (bf16 q/k/v/a/b/out, fp16 state pool, bf16 k ring, bf16 or fp16 u ring, fp32 g ring), but a 64-slot ring
with a 32-row window (RING_SLOTS = 64, W_RING = 32) instead of the 32-slot ring with a 16-row window. The caller
allocates 64-slot rings (+0.63 MiB per request and layer); the default ``flush_min`` is nu - T + 1 (T = 4: 29,
T = 8: 25) and 16 at T = 1. The ring contents are not interchangeable with the 32-slot kernel's.

* live window ``[cache_base, cache_base + hist_len) mod 64``, P = hist_len <= nu = min(W_RING, RING_SLOTS - T) = 32,
  g ring = window-relative cumulative log-decay, ``w_j = e^{G_P - g_j}``;
* every row appends (k̂_s, u_s, g) at ``(base + P + s) & 63``; g restarts at 0 on flush rows;
* rows with ``P >= flush_min`` also write ``S0 <- e^{G_P} S0 + sum_{j<P} w_j u_j k̂_j^T``
  (the T current tokens are not folded);
* padded rows (state index < 0) exit and write nothing; cursor commits are host-side;
* optional stochastic rounding of the flushed fp16 state (``stochastic_rounding``), off by
  default (round-to-nearest). "philox": FlashInfer's Mamba scheme, one Philox4x32 draw (key =
  seed, counter = flat state-pool offset of the 8-element chunk) feeds 4 ``cvt.rs.f16x2``
  pairs. "lcg": one LCG step per pair (s ^ (s >> 16) dither, the ameyn/gdn-ucache-sr recipe)
  from a hashed per-row seed (``rand_seed`` or %clock).

Math per (request, value head) with ``X = [k̂_0..k̂_{T-1}; s q̂_0..s q̂_{T-1}]``:

    S_h x = e^{G_P} S0 x + sum_j (w_j u_j)(k̂_j . x)                  (x in X)
    u_t   = b_t v_t - b_t e^{g_t} S_h k̂_t - sum_{s<t} b_t e^{g_t-g_s}(k̂_s.k̂_t) u_s
    y_t   = e^{g_t} S_h q̂_t + sum_{s<=t} e^{g_t-g_s}(k̂_s.q̂_t) u_s

CTA = (key head, request) = 4 value heads in sequence; 256 threads (two warpgroups); grid (H, B) (key head
fastest); 4 CTAs / SM (SMEM ~57 KB, 64 registers per thread at launch, 128 TMEM columns).
  warps 0-3 (consumer warpgroup, 96 registers after setmaxnreg): 128 lanes = 128 V rows (TMEM lane = row,
             row = lane quadrant ^ 64); prologue + per-row math.
  warp 4   : TMA producer — each value head's fp16 state as two 16 KB 128B-swizzled K halves
             through two stages; per-head u windows bulk-copied
             into a two-head SMEM double buffer; flush rows TMA-store the folded halves.
  warp 5   : allocates TMEM at CTA entry (the next CTA on the SM is not launched before the
             allocation permit is relinquished), only UMMA issuer.
  warp 6   : copies each next head's u window once every consumer has released the buffer.
  warp 7   : idle (the producer warpgroup gives its registers up: 32 per thread after setmaxnreg).
UMMAs (kind::f16, fp32 accumulate in TMEM):
  readout  R = S0 . X^T            M128 N16 K128  A: state stage   B: X tile (K-major)
  history  Y = D . C^T             M128 N16 K16   A: D (TMEM)      B: C tile
  fold     F = D . K_hist (halves) M128 N64 K16   A: D (TMEM)      B: K_hist tile, MN-major
  with D[v, j] = w_j u_j[v], C[x, j] = k̂_j . x, K_hist[j] = k̂_j (j < P).
The per-CTA Gram [k̂_ring; X] . X^T (all dots + the norms) runs on
mma.sync with fragments loaded straight from global memory.  The intra-step causal solve is
a T-step forward substitution per V row on CUDA cores.  Flush rows issue fold half 0
before the state lands, so each half's RMW follows its readout and the epilogue overlaps the
next head's state load.

Specialised for K == V == 128, HV % H == 0, HV // H == 4 (Qwen3.5 GDN geometry).
"""

from __future__ import annotations

import math
import warnings
from functools import cache
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float16, Float32, Int32, Int64, Uint32, Uint64
from cutlass.cute.nvgpu import cpasync

try:
    from . import _umma_helpers as U
    from .device_target import gdn_compile_options
except (
    ImportError
):  # tests and benchmarks may load this file by path, outside the package
    from flashinfer.gdn_kernels import _umma_helpers as U
    from flashinfer.gdn_kernels.device_target import gdn_compile_options

RING_SLOTS = 64  # physical ring per request (a power of two: indices use & RING_MASK)
RING_MASK = RING_SLOTS - 1
# window capacity in rows (<= RING_SLOTS): sizes the window tiles (W_CHUNKS tiles of W_CHUNK rows), the D columns in TMEM and
# the coefficient block. The u window in shared memory holds nu <= W_RING rows, a compile-time depth that window_rows may reduce.
W_RING = 32
W_CHUNK = 16  # history/fold/Gram window tiles: W_CHUNKS independent W_CHUNK-row tiles
W_CHUNKS = W_RING // W_CHUNK
assert (
    W_RING % W_CHUNK == 0
    and W_CHUNK <= W_RING <= RING_SLOTS
    and RING_SLOTS & RING_MASK == 0
)
K_DIM = 128
V_DIM = 128
HPK = 4  # value heads per key head (CTA)
_EPS = 1e-6
_TMEM_COLS = 128
# coefficient block of one value head (fp32 words). The intra-step tables A[t][s] (s < t, t = 1..7) and Bq[t][s] (s <= t) are
# stored as triangles whose rows are padded to 4 or 8 words (16 B aligned rows: the epilogue's row reads stay vector loads)
# instead of as 8 x 8 tables (see _a_off / _bq_off): 156 words per head instead of 192
_CO_A, _CO_BQ, _CO_BETA, _CO_BG, _CO_EG, _CO_GAM, _CO_W = 0, 40, 88, 96, 104, 112, 120
_CO_EGP = _CO_W + W_RING  # e^{G_P}
_COEF = (
    _CO_EGP + 1 + 3
) & ~3  # fp32 coefficient words per head (see _CO_*): a multiple of 4,
#                                                         so every head's block is 16 B aligned (vector loads at a runtime head)
_RN_OFF = (
    W_RING + 16
) * 16  # s_rn (16 fp32) sits right after s_gram inside the aliased block
# CTAs per SM the shared-memory budget allows (233,472 B per SM, 1,024 B reserved per CTA). The dynamic shared memory of a CTA is
# _SMEM_FIXED + 256 B per u-window row nu: state 32 KB, X tile 4 KB, Gram / K_hist block and C tile (W_RING rows), the coefficient
# block, barriers. 57,248 B at nu = 32 (the 8 x 8 coefficient tables would need 57,824 B) against the per-CTA budget of
# 4 CTAs/SM of 57,344 B. The launch bound (min_blocks_per_mp) follows it: asking ptxas for more resident CTAs than shared
# memory can hold only caps the registers (64 at 4 CTAs/SM of 256 threads; setmaxnreg then moves them between the warpgroups:
# _NREG_CONS / _NREG_OTHER, which assume a launch cap of at most 96 registers per thread).
_SMEM_FIXED = 32_768 + 4_096 + W_RING * (256 + 32) + 4 * _COEF * 4 + 480
_SMEM_PER_SM = 233_472
_SMEM_RESERVED_PER_CTA = 1_024
_SMEM_CTAS_PER_SM = 4
_MIN_BLOCKS_PER_SM = _SMEM_CTAS_PER_SM
# registers per thread of the consumer / the other warpgroup after setmaxnreg (a 256-thread CTA at 4 CTAs per SM
# launches with 64 each; the two warpgroups together keep that budget)
_NREG_CONS = 96
_NREG_OTHER = 128 - _NREG_CONS


class _MbArr:
    """Contiguous mbarrier array; ``[]`` takes a static or a runtime index."""

    def __init__(self, base):
        self.base = base

    def __getitem__(self, i):
        return self.base + i


def _sel(cond, a, b, ty):
    """Branch-free typed select (arith.select) for use outside AST-preprocessed scopes."""
    return ty(cutlass.select_(cond, ty(a), ty(b)))


def _pos(x):
    """max(x, 0) for a Python int or a runtime Int32."""
    return max(x, 0) if isinstance(x, int) else _sel(x > 0, x, Int32(0), Int32)


def _a_off(t, s):
    """Word offset of A[t][s] (s < t) in a head's coefficient block: rows 1..4 take 4 words, rows 5..7 take 8."""
    return _CO_A + 4 * _pos(t - 1) + 4 * _pos(t - 5) + s


def _bq_off(t, s):
    """Word offset of Bq[t][s] (s <= t) in a head's coefficient block: rows 0..3 take 4 words, rows 4..7 take 8."""
    return _CO_BQ + 4 * t + 4 * _pos(t - 4) + s


def _sw128_off(row, k):
    """Byte offset of element (row, k) in a [rows x 128] fp16 K-major tile stored as two
    128B-swizzled K groups of 64 (group stride rows * 128 B is added by the caller)."""
    chunk = (k % 64) // 8
    return row * 128 + ((chunk ^ (row % 8)) * 16) + (k % 8) * 2


class GdnReplayMtpUmma:
    def __init__(
        self,
        T: int,
        h: int,
        hv: int,
        u_bf16: bool = True,
        use_row_order: bool = False,
        hpc: int = HPK,
        sr: str = "",
        sr_rounds: int = 0,
        pmax: int = 0,
    ):
        assert 1 <= T <= 8
        assert hv % h == 0 and hv // h == HPK
        # hpc = value heads per CTA (4: one CTA per key head; 2 / 1 split a key head over 2 / 4
        # CTAs, which repeat the per-key-head prologue but shorten the serial head chain)
        assert hpc in (1, 2, 4)
        self.hpc = hpc
        self.nsub = HPK // hpc
        self.T = T
        self.h = h
        self.hv = hv
        self.u_bf16 = u_bf16
        self.use_row_order = use_row_order
        # window rows that can ever be live: P <= W_RING (the window tiles hold W_RING rows) and P <= RING_SLOTS - T
        # (the T appends of the call land at slots P..P+T-1 of the ring), so rows >= nu are never read and are not even
        # copied. pmax (compile-time, the wrapper's flush_min - 1 + T) is the deepest window the caller will present:
        # the shared-memory window holds nu = pmax rows (at least one W_CHUNK tile), 0 = the whole window
        nu_max = min(W_RING, RING_SLOTS - T)
        assert pmax <= nu_max, (pmax, nu_max)
        self.nu = max(pmax, W_CHUNK) if pmax else nu_max
        assert (
            _SMEM_FIXED + 256 * self.nu
            <= _SMEM_PER_SM // _SMEM_CTAS_PER_SM - _SMEM_RESERVED_PER_CTA
        ), (
            f"{_SMEM_FIXED + 256 * self.nu} B of shared memory per CTA do not fit {_SMEM_CTAS_PER_SM} CTAs/SM"
        )
        # stochastic rounding of the flush store: "" (round-to-nearest), "philox" (sr_rounds
        # rounds), "lcg" (seeded by rand_seed) or "lcg_clock" (seeded by %clock)
        assert sr in ("", "philox", "lcg", "lcg_clock")
        self.sr = sr
        self.sr_rounds = sr_rounds if sr == "philox" else 0
        self.sr_lcg = sr in ("lcg", "lcg_clock")
        # consumer head loop: a runtime loop over the CTA's 4 heads instead of 4 unrolled copies of the body (a
        # quarter of the consumer head code: it stays in the instruction cache while the CTAs of an SM run
        # different heads and flush / verify rows). hpc < 4 bodies are short and stay unrolled. The producer / issuer
        # use the same runtime head loop (their per-head code was ~25% of the kernel's SASS unrolled).
        self.hloop = hpc == HPK
        self.num_threads = 256  # warps 0-3 consumers, 4 producer, 5 issuer, 6 window copies, 7 idle (register donor)
        # T >= 5: the D builds' v loads sit on the epilogue-bound chain; below that they hide behind the readout wait
        self.pfv = T >= 5

    @cute.jit
    def _make_state_tma(self, state: cute.Tensor, op):
        elems = 64
        smem_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                (1, 1, V_DIM, (elems, 1), 1),
                stride=(0, 0, elems, (1, V_DIM * elems), V_DIM * elems),
            ),
        )
        return cpasync.make_tiled_tma_atom(
            op,
            cute.logical_divide(state, (None, None, None, elems)),
            smem_layout,
            cta_tiler=(1, 1, V_DIM, elems),
        )

    @cute.jit
    def __call__(
        self,
        q,
        k,
        v,
        a,
        b,
        a_log,
        dt_bias,
        state,
        state_idx,
        k_cache,
        u_cache,
        g_cache,
        hist_len,
        cache_base,
        row_order,
        out,
        rand_seed,
        scale: Float32,
        flush_min: Int32,
        stream,
    ):
        tma_ld = self._make_state_tma(state, cpasync.CopyBulkTensorTileG2SOp())
        tma_st = self._make_state_tma(state, cpasync.CopyBulkTensorTileS2GOp())
        grid = (self.h * self.nsub, q.shape[0], 1)
        self.kernel(
            q,
            k,
            v,
            a,
            b,
            a_log,
            dt_bias,
            state,
            state_idx,
            k_cache,
            u_cache,
            g_cache,
            hist_len,
            cache_base,
            row_order,
            out,
            rand_seed,
            scale,
            flush_min,
            tma_ld,
            tma_st,
        ).launch(
            grid=grid,
            block=(self.num_threads, 1, 1),
            min_blocks_per_mp=_MIN_BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        a_log: cute.Tensor,
        dt_bias: cute.Tensor,
        state: cute.Tensor,
        state_idx_t: cute.Tensor,
        k_cache: cute.Tensor,
        u_cache: cute.Tensor,
        g_cache: cute.Tensor,
        hist_len: cute.Tensor,
        cache_base: cute.Tensor,
        row_order: cute.Tensor,
        out: cute.Tensor,
        rand_seed: cute.Tensor,
        scale: Float32,
        flush_min: Int32,
        tma_ld: cpasync.TmaInfo,
        tma_st: cpasync.TmaInfo,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bx, blk_row, _ = cute.arch.block_idx()
        _HPC = self.hpc
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32

        # ------------------------------------------------------------------ SMEM
        smem = cutlass.utils.SmemAllocator()
        st_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout((V_DIM, (64, 2)), stride=(64, (1, V_DIM * 64))),
        )
        s_state = smem.allocate_tensor(
            Float16, st_layout.outer, byte_alignment=1024, swizzle=st_layout.inner
        )
        s_x = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(16 * 128), byte_alignment=1024
        )
        # K_hist tile: window keys k̂_j (j < P, else 0), as W_CHUNKS independent W_CHUNK-row
        # tiles (chunk c covers window rows c*W_CHUNK .. c*W_CHUNK+W_CHUNK-1), each row-major
        # [W_CHUNK slots x 128] fp16 in the X-tile SW128 layout; the fold reads each chunk as
        # an MN-major B operand (N = k index). Two independent W_CHUNK-row tiles side by side
        # (not one wider swizzled tile) so the proven 16-row swizzle math is reused unchanged.
        # Flush rows only. Its bytes double as s_gram (+ s_rn): rows [0, W_RING) ring x X (Gram,
        # two W_CHUNK-row blocks), [W_RING, W_RING+16) X x X (Gram), then s_rn. Both are dead once
        # the C tile is built; flush rows add a barrier between the C-tile fill (reads s_gram)
        # and the K_hist fill (overwrites it). Verify rows never write the K_hist tile.
        s_gram = smem.allocate_array(
            Float32, W_CHUNKS * W_CHUNK * 128 // 2, byte_alignment=1024
        )
        s_kt = s_gram
        s_c = smem.allocate_tensor(
            cutlass.Uint16,
            cute.make_layout(W_CHUNKS * W_CHUNK * 16),
            byte_alignment=256,
        )
        # u window of the head being processed: [logical slot j < nu][V] (ring dtype), bulk-copied
        # (head 0 by the producer warp at entry, head h + 1 by warp 6 once every consumer has read
        # head h's window). Single-buffered: rows >= nu = min(W_RING, RING_SLOTS - T) are never live.
        s_uwin = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(self.nu * V_DIM), byte_alignment=128
        )
        s_coef = smem.allocate_array(Float32, HPK * _COEF)
        s_rn = s_gram
        u_full = _MbArr(smem.allocate_array(cutlass.Int64, HPK))
        u_free = [smem.allocate_array(cutlass.Int64, 1)]
        st_full = _MbArr(smem.allocate_array(cutlass.Int64, 2 * HPK))
        mma_slot = _MbArr(smem.allocate_array(cutlass.Int64, 2 * HPK))
        mma_done = _MbArr(smem.allocate_array(cutlass.Int64, HPK))
        d_ready = _MbArr(smem.allocate_array(cutlass.Int64, HPK))
        epi_done = _MbArr(smem.allocate_array(cutlass.Int64, HPK))
        f_mma_a = _MbArr(smem.allocate_array(cutlass.Int64, HPK))
        f_mma_b = _MbArr(smem.allocate_array(cutlass.Int64, HPK))
        f_epi_a = _MbArr(smem.allocate_array(cutlass.Int64, HPK))
        rd0 = _MbArr(smem.allocate_array(cutlass.Int64, HPK))
        f_gdone = _MbArr(smem.allocate_array(cutlass.Int64, 2 * HPK))
        x_ready = smem.allocate_array(cutlass.Int64, 1)
        tmem_done = smem.allocate_array(cutlass.Int64, 1)
        tmem_addr = smem.allocate(Int32, 4)
        sink = smem.allocate(
            Int32, 4
        )  # dummy word: _d_build's ordering store before the window release

        # TMEM first: the next CTA on this SM is not launched before this one relinquishes
        # its TMEM allocation permit (measured: allocating after the prologue spaced the
        # CTAs of one SM ~2.5 us apart and left a quarter of the CTA slots empty)
        if warp == 5:
            U.tmem_alloc(tmem_addr, _TMEM_COLS)
            cute.arch.relinquish_tmem_alloc_permit()

        st_addr = s_state.iterator.toint()
        x_addr = s_x.iterator.toint()
        kt_addr = s_kt.toint()
        c_addr = s_c.iterator.toint()
        uw_addr = s_uwin.iterator.toint()
        u_g = u_cache.iterator.toint()
        S = dict(
            s_state=s_state,
            s_x=s_x,
            s_kt=s_kt,
            s_c=s_c,
            s_gram=s_gram,
            s_coef=s_coef,
            s_rn=s_rn,
            s_uwin=s_uwin,
            u_full=u_full,
            u_free=u_free,
            st_full=st_full,
            mma_slot=mma_slot,
            mma_done=mma_done,
            d_ready=d_ready,
            epi_done=epi_done,
            f_mma_a=f_mma_a,
            f_mma_b=f_mma_b,
            f_epi_a=f_epi_a,
            rd0=rd0,
            f_gdone=f_gdone,
            x_ready=x_ready,
            tmem_done=tmem_done,
            tmem_addr=tmem_addr,
            sink=sink,
            st_addr=st_addr,
            x_addr=x_addr,
            kt_addr=kt_addr,
            c_addr=c_addr,
            uw_addr=uw_addr,
            u_g=u_g,
        )

        self._item(
            blk_row * (self.h * self.nsub) + bx,
            0,
            S,
            tid,
            warp,
            lane,
            q,
            k,
            v,
            a,
            b,
            a_log,
            dt_bias,
            state,
            state_idx_t,
            k_cache,
            u_cache,
            g_cache,
            hist_len,
            cache_base,
            row_order,
            out,
            rand_seed,
            scale,
            flush_min,
            tma_ld,
            tma_st,
        )

    @cute.jit
    def _item(
        self,
        item,
        ph,
        S,
        tid,
        warp,
        lane,
        q,
        k,
        v,
        a,
        b,
        a_log,
        dt_bias,
        state,
        state_idx_t,
        k_cache,
        u_cache,
        g_cache,
        hist_len,
        cache_base,
        row_order,
        out,
        rand_seed,
        scale,
        flush_min,
        tma_ld,
        tma_st,
    ):
        """One work item (request x key head): the whole per-CTA body. Warps 0-3 (consumers) and 4-7
        (producer, issuer, two idle warps) are separate warpgroups with their own register budget."""
        # the loop-carried item index must be provably warp-uniform, otherwise everything derived
        # from it (row, state index, head base, addresses) leaves the uniform register file
        item = cute.arch.make_warp_uniform(item)
        HPC = self.hpc
        T = self.T
        T2 = 2 * T
        RB = self.u_bf16
        _ring_t = BFloat16 if RB else Float16
        s_state = S["s_state"]
        _s_x = S["s_x"]
        _s_kt = S["s_kt"]
        _s_c = S["s_c"]
        s_gram = S["s_gram"]
        s_coef = S["s_coef"]
        _s_rn = S["s_rn"]
        _s_uwin = S["s_uwin"]
        u_full = S["u_full"]
        u_free = S["u_free"]
        st_full = S["st_full"]
        mma_slot = S["mma_slot"]
        mma_done = S["mma_done"]
        d_ready = S["d_ready"]
        epi_done = S["epi_done"]
        f_mma_a = S["f_mma_a"]
        f_mma_b = S["f_mma_b"]
        f_epi_a = S["f_epi_a"]
        rd0 = S["rd0"]
        f_gdone = S["f_gdone"]
        x_ready = S["x_ready"]
        tmem_done = S["tmem_done"]
        tmem_addr = S["tmem_addr"]
        sink = S["sink"]
        st_addr = S["st_addr"]
        x_addr = S["x_addr"]
        kt_addr = S["kt_addr"]
        c_addr = S["c_addr"]
        uw_addr = S["uw_addr"]
        u_g = S["u_g"]
        nblk = self.h * self.nsub
        blk_row = item // nblk
        bx = item % nblk

        kh = bx >> (self.nsub.bit_length() - 1)  # key head
        sub = bx & (self.nsub - 1)  # which hpc-head group of it
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        if cutlass.const_expr(self.use_row_order):
            brow = Int32(row_order[blk_row])
        else:
            brow = blk_row
        hv0 = kh * HPK + sub * HPC
        if warp < 4:
            # ======================================================== consumer warpgroup (warps 0-3)
            cute.arch.warpgroup_reg_alloc(_NREG_CONS)
            # Inputs that depend on the batch row only are issued at entry (before the cursor loads,
            # the padded-row exit and the CTA barrier), so their latency overlaps the cursors'.
            T_ = self.T
            grp_e = lane >> 2
            c4_e = lane & 3
            cw_e = warp ^ 2  # consumer index: TMEM lane quadrant = warp % 4
            nt_e = cw_e & 1
            q_g = q.iterator.toint()
            k_g = k.iterator.toint()
            xr_b = nt_e * 8 + grp_e
            x_own = self._x_row_addr(k, q, k_g, q_g, brow, kh, xr_b)
            x_oth = self._x_row_addr(k, q, k_g, q_g, brow, kh, (1 - nt_e) * 8 + grp_e)
            wb = []
            wo = []
            is_oth = Int32(cw_e >= 2)  # only the X x X warps read the other X half
            for p in cutlass.range_constexpr(4):
                koff = Int64((32 * p + 8 * c4_e) * 2)
                wb.append(U.ldg_v4(x_own + koff))
                wo.append(U.ldg_v4_if(is_oth, x_oth + koff))
            ct_e = (tid ^ 64) & 127
            gh = (ct_e >> 3) & 3
            ghc = gh & (HPC - 1)  # in-range head for loads
            gt = ct_e & 7
            tt = _sel(gt < T_, gt, 0, Int32)
            ab_off = Int64(cute.crd2idx((brow, tt, hv0 + ghc), a.layout)) * 2
            av_raw = U.ldg_u16(a.iterator.toint() + ab_off)
            bv_raw = U.ldg_u16(b.iterator.toint() + ab_off)

            sidx = Int32(state_idx_t[brow])
            P = Int32(
                hist_len[brow]
            )  # the cursors do not depend on the state index: one round trip
            base = Int32(cache_base[brow])
            U.exit_if_negative(sidx)
            is_flush = P >= flush_min
            # window g values (gate threads: G_P of their head; weight threads: g_j)
            wh = ((ct_e - 32) >> 4) & 3
            whc = wh & (HPC - 1)
            wj = (ct_e - 32) & 15
            gp_g = Float32(g_cache[sidx, hv0 + ghc, (base + P - 1) & RING_MASK])
            gp_w = Float32(g_cache[sidx, hv0 + whc, (base + P - 1) & RING_MASK])
            gj_w = Float32(g_cache[sidx, hv0 + whc, (base + wj) & RING_MASK])
            # window rows [P, nu) are never copied: zero them once (this CTA's SMEM holds another CTA's
            # data) so the D build needs no per-row masks (its weights are 0 there)
            for r in cutlass.range_constexpr((self.nu * 16 + 127) // 128):
                zc = ct_e + 128 * r  # 16 B chunk of the window, 16 per row
                if zc < self.nu * 16:
                    if (zc >> 4) >= P:
                        U.sts_v4(
                            uw_addr + zc * 16,
                            Uint32(0),
                            Uint32(0),
                            Uint32(0),
                            Uint32(0),
                        )
            cute.arch.sync_threads()
            # ============================================================ consumers
            cw = warp ^ 2
            ct = cw * 32 + lane  # TMEM lane address / k index for staging
            row = ct ^ 64  # V row held by this thread (= TMEM lane)
            grp = lane >> 2
            c4 = lane & 3

            kc_g = k_cache.iterator.toint()

            # -------- Gram on mma.sync: rows [k̂_ring(16); X(16)] x cols X(16), K = 128.
            # Fragments come straight from global memory: per 32-wide k block a lane loads 16 B
            # (8 contiguous k) of each of its rows; the k order inside the m16n8k16 fragment is
            # permuted identically for A and B, which a dot product does not see.
            # Warps mt == 1 own the X x X tiles (norms, intra-step dots): they depend on the
            # batch row only, so the X tile and x_ready follow ONE global round trip.  Warps
            # mt == 0 own the ring x X tiles (need the cursors: a second round trip); they stage
            # the X tile from their B fragments first and run the ring tiles after x_ready.
            mt = cw >> 1
            nt = cw & 1
            gc = nt * 8 + 2 * c4
            if mt == 1:
                # A rows grp / grp + 8 of the X x X tile: this lane's own row (wb) and the
                # other half's (wo), both loaded at entry
                accx = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                for p in cutlass.range_constexpr(4):
                    for hs in cutlass.range_constexpr(2):
                        w0a = _sel(nt == 0, wb[p][2 * hs], wo[p][2 * hs], Uint32)
                        w1a = _sel(nt == 0, wo[p][2 * hs], wb[p][2 * hs], Uint32)
                        w0b = _sel(
                            nt == 0, wb[p][2 * hs + 1], wo[p][2 * hs + 1], Uint32
                        )
                        w1b = _sel(
                            nt == 0, wo[p][2 * hs + 1], wb[p][2 * hs + 1], Uint32
                        )
                        accx = U.mma_bf16_16816(
                            w0a, w1a, w0b, w1b, wb[p][2 * hs], wb[p][2 * hs + 1], accx
                        )
                s_gram[(W_RING + grp) * 16 + gc] = accx[0]
                s_gram[(W_RING + grp) * 16 + gc + 1] = accx[1]
                s_gram[(W_RING + 8 + grp) * 16 + gc] = accx[2]
                s_gram[(W_RING + 8 + grp) * 16 + gc + 1] = accx[3]
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                # Everything below needs only the X x X Gram and the gate inputs (one round
                # trip), so it overlaps warps mt == 0 waiting for the ring rows.
                cl = ct - 64
                if cl < T2:
                    rn_own = cute.math.rsqrt(
                        s_gram[(W_RING + cl) * 16 + cl] + Float32(_EPS)
                    )
                    if cl >= T:
                        rn_own = rn_own * scale
                    s_gram[_RN_OFF + cl] = rn_own
                if cl < 32:
                    # gates: lane = head gh x token gt
                    hvg = hv0 + ghc
                    av = U.u16_as_f32(av_raw, True)
                    bv = U.u16_as_f32(bv_raw, True)
                    alog = Float32(a_log[hvg])
                    dtb = Float32(dt_bias[hvg])
                    xx = av + dtb
                    sp = xx
                    if xx <= Float32(20.0):
                        sp = cute.math.log(
                            Float32(1.0) + cute.math.exp(xx, fastmath=True),
                            fastmath=True,
                        )
                    la = -cute.math.exp(alog, fastmath=True) * sp
                    if gt >= T:
                        la = Float32(0.0)
                    beta = Float32(1.0) / (
                        Float32(1.0) + cute.math.exp(-bv, fastmath=True)
                    )
                    gam = la
                    for off in (1, 2, 4):
                        lower = cute.arch.shuffle_sync_up(gam, off, mask_and_clamp=0)
                        if gt >= off:
                            gam = gam + lower
                    eg = cute.math.exp(gam, fastmath=True)
                    cbg = gh * _COEF
                    s_coef[cbg + _CO_BETA + gt] = beta
                    s_coef[cbg + _CO_BG + gt] = beta * eg
                    s_coef[cbg + _CO_EG + gt] = eg
                    s_coef[cbg + _CO_GAM + gt] = gam
                cute.arch.barrier(barrier_id=2, number_of_threads=64)
                # intra-step coefficient tables A[t][s], Bq[t][s] per head (64 threads)
                NT = HPC * T * T
                for it in cutlass.range_constexpr((NT + 63) // 64):
                    idx = cl + it * 64
                    if idx < NT:
                        ch = idx // (T * T)
                        t = (idx // T) % T
                        s = idx % T
                        cb = ch * _COEF
                        dec = cute.math.exp(
                            s_coef[cb + _CO_GAM + t] - s_coef[cb + _CO_GAM + s],
                            fastmath=True,
                        )
                        if s < t:
                            kk = (
                                s_gram[(W_RING + s) * 16 + t]
                                * s_gram[_RN_OFF + s]
                                * s_gram[_RN_OFF + t]
                            )
                            s_coef[cb + _a_off(t, s)] = (
                                s_coef[cb + _CO_BETA + t] * dec * kk
                            )
                        if s <= t:
                            qk = (
                                s_gram[(W_RING + s) * 16 + T + t]
                                * s_gram[_RN_OFF + s]
                                * s_gram[_RN_OFF + T + t]
                            )
                            s_coef[cb + _bq_off(t, s)] = dec * qk
            else:
                # ring rows j >= P are dead (their Gram rows are dropped by the C-tile
                # fill, MMA rows are independent): predicate their loads off, so a row
                # beyond the live window costs no DRAM traffic
                # Dead rows are redirected to the last live row (a duplicate address inside the
                # CTA: L1/L2 hit, no new DRAM bytes) instead of predicated off, which keeps the
                # loads plain and straight-line like the full-window kernel's (predicated asm
                # with undefined outputs cost spills at the 80-register cap).
                _wlim = _sel(P > 0, P - 1, Int32(0), Int32)
                # Ring rows of chunk 0 are loaded here (dead rows j >= P are redirected to the last
                # live row: a duplicate address inside the CTA, an L1/L2 hit with no new DRAM bytes,
                # which keeps the loads plain; predicated asm with undefined outputs spilled at the
                # 80-register cap). Chunks >= 1 are loaded and multiplied only while the live window
                # reaches them (CTA-uniform branch, after chunk 0 so the registers are never live
                # together): a shallow row (P <= 16) does no chunk-1 Gram work at all.
                wr0c0, wr1c0 = self._ring_k_loads(
                    0, kc_g, k_cache, sidx, kh, base, P, grp, c4
                )
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                # X tile (fp16, SW128) from the B fragments: lane = row xr_b, 4 x 16 B of k
                rn_b = cute.math.rsqrt(
                    s_gram[(W_RING + xr_b) * 16 + xr_b] + Float32(_EPS)
                )
                rn_b = _sel(xr_b >= T, rn_b * scale, rn_b, Float32)
                for p in cutlass.range_constexpr(4):
                    kc = 4 * p + c4
                    xw = []
                    for i in cutlass.range_constexpr(4):
                        xw.append(
                            U.pack_f16x2(
                                U.bf16x2_lo_f32(wb[p][i]) * rn_b,
                                U.bf16x2_hi_f32(wb[p][i]) * rn_b,
                            )
                        )
                    xb = (kc // 8) * 2048 + xr_b * 128 + (((kc % 8) ^ (xr_b % 8)) * 16)
                    U.sts_v4(x_addr + xb, xw[0], xw[1], xw[2], xw[3])
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(x_ready)
                # ring x X tiles
                self._gram_chunk(0, wr0c0, wr1c0, wb, s_gram, grp, gc)
                for wc in cutlass.range_constexpr(1, W_CHUNKS):
                    if P > wc * W_CHUNK:
                        wr0c, wr1c = self._ring_k_loads(
                            wc, kc_g, k_cache, sidx, kh, base, P, grp, c4
                        )
                        self._gram_chunk(wc, wr0c, wr1c, wb, s_gram, grp, gc)
                # k̂ ring appends (bf16, 16 B per lane) after the proxy fence above (it lowers
                # to MEMBAR.ALL.CTA, which would wait for these stores to be acknowledged)
                if xr_b < T:
                    if sub == 0:
                        kdst = (
                            kc_g
                            + Int64(
                                cute.crd2idx(
                                    (sidx, kh, (base + P + xr_b) & RING_MASK, 0),
                                    k_cache.layout,
                                )
                            )
                            * 2
                        )
                        for p in cutlass.range_constexpr(4):
                            kw4 = []
                            for i in cutlass.range_constexpr(4):
                                kw4.append(
                                    U.pack_bf16x2(
                                        U.bf16x2_lo_f32(wb[p][i]) * rn_b,
                                        U.bf16x2_hi_f32(wb[p][i]) * rn_b,
                                    )
                                )
                            U.stg_v4(
                                kdst + Int64((32 * p + 8 * c4) * 2),
                                kw4[0],
                                kw4[1],
                                kw4[2],
                                kw4[3],
                            )
            cute.arch.barrier(barrier_id=1, number_of_threads=128)

            # -------- after the ring tiles: C tile (warps mt == 1), window weights, g-ring
            # appends, Kt (flush rows)
            # C tile as W_CHUNKS independent W_CHUNK-row 32B-swizzled tiles (chunk wc at byte offset
            # wc * (W_CHUNK * 16 * 2)): reuses the proven 16-row swizzle formula unchanged per chunk, with jl the row
            # local to its own chunk. One thread per (x column, row pair): 128 threads x 2 rows, one 32-bit store
            # each, the column's norm once.
            cxc = ct >> 3
            crn = Float32(0.0)
            if cxc < T2:
                crn = cute.math.rsqrt(s_gram[(W_RING + cxc) * 16 + cxc] + Float32(_EPS))
                if cxc >= T:
                    crn = crn * scale
            for wc in cutlass.range_constexpr(W_CHUNKS):
                if P > (
                    -1 if wc == 0 else wc * W_CHUNK
                ):  # chunk wc >= 1 is read only while P reaches it
                    jl0 = (ct & 7) * 2
                    j0 = wc * W_CHUNK + jl0
                    c0 = Float32(0.0)
                    c1 = Float32(0.0)
                    if cxc < T2:
                        if j0 < P:
                            c0 = s_gram[j0 * 16 + cxc] * crn
                        if j0 + 1 < P:
                            c1 = s_gram[(j0 + 1) * 16 + cxc] * crn
                    cbyte = (
                        wc * (W_CHUNK * 16 * 2)
                        + cxc * 32
                        + (((jl0 // 8) ^ ((cxc >> 2) & 1)) * 16)
                        + (jl0 % 8) * 2
                    )
                    U.sts_u32(c_addr + cbyte, U.pack_f16x2(c0, c1))
            if is_flush:
                # the C-tile fill above reads s_gram, whose bytes the K_hist tile reuses
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                # K_hist tile (fold B operand): thread -> (slot row jr = ct / 8, 16 B chunk c8) of both
                # K groups; ring rows were just read by the Gram (L1/L2 hits); slots j >= P zeroed.
                # W_CHUNKS independent W_CHUNK-row tiles, chunk wc at byte offset
                # wc * (W_CHUNK * 128 * 2) -- same per-chunk layout / formula as the W=16 kernel.
                jr = ct >> 3
                c8 = ct & 7
                for wc in cutlass.range_constexpr(W_CHUNKS):
                    if P > (
                        -1 if wc == 0 else wc * W_CHUNK
                    ):  # chunk wc >= 1 is read only while P reaches it
                        j = wc * W_CHUNK + jr
                        krow = (
                            kc_g
                            + Int64(
                                cute.crd2idx(
                                    (sidx, kh, (base + j) & RING_MASK, c8 * 8),
                                    k_cache.layout,
                                )
                            )
                            * 2
                        )
                        kv = Int32(
                            j < P
                        )  # dead window rows (j >= P) are zero-filled, never loaded
                        kr = [U.ldg_v4_if(kv, krow), U.ldg_v4_if(kv, krow + Int64(128))]
                        kt_chunk = kt_addr + wc * (W_CHUNK * 128 * 2)
                        for gk in cutlass.range_constexpr(2):
                            kw4 = []
                            for i in cutlass.range_constexpr(4):
                                lo = _sel(
                                    j < P, U.bf16x2_lo_f32(kr[gk][i]), 0.0, Float32
                                )
                                hi = _sel(
                                    j < P, U.bf16x2_hi_f32(kr[gk][i]), 0.0, Float32
                                )
                                kw4.append(U.pack_f16x2(lo, hi))
                            U.sts_v4(
                                kt_chunk
                                + gk * 2048
                                + jr * 128
                                + ((c8 ^ (jr % 8)) * 16),
                                kw4[0],
                                kw4[1],
                                kw4[2],
                                kw4[3],
                            )
            # C / Kt are read by the history / fold UMMAs, which the issuer warp issues only after
            # d_ready, arrived by these same threads after this fence
            cute.arch.fence_view_async_shared()
            if ct >= 64:
                if ct < 96:
                    # g appends of the gate lanes (G_P needs the cursors: second round trip)
                    if gt < T:
                        if gh < HPC:
                            gnew = s_coef[gh * _COEF + _CO_GAM + gt]
                            if not is_flush:
                                if P > 0:
                                    gnew = gnew + gp_g
                            g_cache[sidx, hv0 + gh, (base + P + gt) & RING_MASK] = gnew
            if ct >= 32:
                if ct < 96:
                    # window weights w_j = e^{G_P - g_j}, j < P. 64 threads (4 heads x 16
                    # rows) cover one W_CHUNK; W_CHUNKS passes over the same threads cover
                    # the full W_RING window (second+ passes fetch g_j at wj + wc*W_CHUNK
                    # directly).
                    gpw = gp_w
                    for wc in cutlass.range_constexpr(W_CHUNKS):
                        wjg = wj + wc * W_CHUNK
                        if cutlass.const_expr(wc == 0):
                            gjw = gj_w
                            w = Float32(0.0)
                            if wjg < P:
                                w = cute.math.exp(gpw - gjw, fastmath=True)
                            s_coef[wh * _COEF + _CO_W + wjg] = w
                        else:
                            # chunk wc >= 1: the g ring rows are read only for live
                            # rows, and the whole pass is skipped while P stays inside chunk 0
                            if P > wc * W_CHUNK:
                                w = Float32(0.0)
                                if wjg < P:
                                    gjw = Float32(
                                        g_cache[
                                            sidx, hv0 + whc, (base + wjg) & RING_MASK
                                        ]
                                    )
                                    w = cute.math.exp(gpw - gjw, fastmath=True)
                                s_coef[wh * _COEF + _CO_W + wjg] = w
                    if wj == 0:
                        egp = Float32(1.0)
                        if P > 0:
                            egp = cute.math.exp(gpw, fastmath=True)
                        s_coef[wh * _COEF + _CO_EGP] = egp
            cute.arch.barrier(barrier_id=1, number_of_threads=128)

            tbase = tmem_addr[0]
            ctx = dict(
                sidx=sidx,
                base=base,
                P=P,
                brow=brow,
                hv0=hv0,
                ct=ct,
                row=row,
                tbase=tbase,
                st_addr=st_addr,
                u_cache=u_cache,
                v=v,
                out=out,
                s_coef=s_coef,
                d_ready=d_ready,
                mma_done=mma_done,
                epi_done=epi_done,
                uw_addr=uw_addr,
                sink=sink.toint(),
                u_full=u_full,
                u_free=u_free,
                f_mma_a=f_mma_a,
                f_mma_b=f_mma_b,
                f_epi_a=f_epi_a,
                f_gdone=f_gdone,
                rd0=rd0,
                ph=ph,
                is_flush=is_flush,
            )
            # one head body for flush and verify rows (runtime is_flush inside it): the D build and the epilogue
            # exist once, the flush-only waits + RMW halves are a CTA-uniform branch in the middle
            self._load_seed(ctx, rand_seed)
            if cutlass.const_expr(self.hloop):
                for hh in cutlass.range(HPC, unroll=1):
                    self._head_u(cute.arch.make_warp_uniform(hh), ctx)
            else:
                for hh in cutlass.range_constexpr(HPC):
                    self._head_u(hh, ctx)
            U.fence_before_sync()
            cute.arch.mbarrier_arrive(tmem_done)
        else:
            # ================================================= producer / issuer warpgroup (warps 4-7)
            cute.arch.warpgroup_reg_dealloc(_NREG_OTHER)
            sidx = Int32(state_idx_t[brow])
            P = Int32(hist_len[brow])
            base = Int32(cache_base[brow])
            if warp == 5:
                if sidx < 0:
                    # padded row: hand the TMEM columns back before the CTA exits
                    cute.arch.sync_warp()
                    U.tmem_dealloc(tmem_addr[0], _TMEM_COLS)
            U.exit_if_negative(sidx)
            is_flush = P >= flush_min
            if warp == 7:
                # the idle warp primes L1 / L2 (pure hints, off every critical path) with what the consumers load
                # right after the CTA barrier: the live k-ring rows of this key head (Gram chunks 0 and 1; replaces
                # the producer warp's L2-only prefetch of chunks >= 1), and for T >= 5 verify rows the v rows of the
                # call (D builds; flush rows are DRAM-bound and gain nothing)
                for rr in cutlass.range_constexpr(W_CHUNKS):
                    self._ring_k_prefetch_l1(
                        k_cache.iterator.toint(),
                        k_cache,
                        sidx,
                        kh,
                        base,
                        P,
                        lane,
                        rr * W_CHUNK,
                    )
                if cutlass.const_expr(self.pfv):
                    for rr in cutlass.range_constexpr(2):
                        pv, va = self._v_prefetch_l1(v, brow, hv0, lane, rr)
                        U.prefetch_l1_line_if(_sel(is_flush, Int32(0), pv, Int32), va)
            if warp == 4:
                # first two state halves before the bulk of the barrier inits
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(st_full[0], 1)
                    cute.arch.mbarrier_init(st_full[1], 1)
                    cute.arch.mbarrier_init(u_full[0], 1)
                    cute.arch.mbarrier_init(u_full[1], 1)
                    cute.arch.mbarrier_init_fence()
                cpasync.prefetch_descriptor(tma_ld.atom)
                for g in cutlass.range_constexpr(2):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(st_full[g], 16384)
                    U.simple_tma_copy(
                        tma_ld.atom,
                        tma_ld.tma_tensor[sidx, hv0, None, (None, g)],
                        s_state[None, (None, g)],
                        st_full[g],
                    )
                # single-buffered window: head 0 here (producer warp), head h + 1 by warp 6 once every
                # consumer has read head h's window (_window_warp)
                with cute.arch.elect_one():
                    self._u_window_copy(
                        u_cache, u_g, uw_addr, u_full[0], sidx, hv0, base, P
                    )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(u_full[2], 1)
                    cute.arch.mbarrier_init(u_full[3], 1)
                    cute.arch.mbarrier_init(u_free[0], 128)
                    for s in cutlass.range_constexpr(2, 2 * HPK):
                        cute.arch.mbarrier_init(st_full[s], 1)
                    for s in cutlass.range_constexpr(2 * HPK):
                        cute.arch.mbarrier_init(mma_slot[s], 1)
                    for i in cutlass.range_constexpr(HPK):
                        cute.arch.mbarrier_init(mma_done[i], 1)
                        cute.arch.mbarrier_init(d_ready[i], 128)
                        cute.arch.mbarrier_init(epi_done[i], 128)
                    cute.arch.mbarrier_init(x_ready, 64)
                    cute.arch.mbarrier_init(tmem_done, 128)
                    if is_flush:
                        for i in cutlass.range_constexpr(HPK):
                            cute.arch.mbarrier_init(f_mma_a[i], 1)
                            cute.arch.mbarrier_init(f_mma_b[i], 1)
                            cute.arch.mbarrier_init(f_epi_a[i], 128)
                            cute.arch.mbarrier_init(rd0[i], 1)
                        for s in cutlass.range_constexpr(2 * HPK):
                            cute.arch.mbarrier_init(f_gdone[s], 128)
                    cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()

            if warp == 4:
                # ============================================================ producer
                if is_flush:
                    cpasync.prefetch_descriptor(tma_st.atom)
                    pc = dict(
                        tma_ld=tma_ld,
                        tma_st=tma_st,
                        s_state=s_state,
                        sidx=sidx,
                        hv0=hv0,
                        ph=ph,
                        f_gdone=f_gdone,
                        st_full=st_full,
                        mma_slot=mma_slot,
                    )
                    if cutlass.const_expr(self.hloop):
                        for hh in cutlass.range(1, HPC, unroll=1):
                            self._produce_flush(hh, pc)
                    else:
                        for hh in cutlass.range_constexpr(1, HPC):
                            self._produce_flush(hh, pc)
                    for prev in cutlass.range_constexpr(2 * HPC - 2, 2 * HPC):
                        cute.arch.mbarrier_wait(f_gdone[prev], ph)
                        U.simple_tma_copy(
                            tma_st.atom,
                            s_state[None, (None, prev % 2)],
                            tma_st.tma_tensor[
                                sidx, hv0 + prev // 2, None, (None, prev % 2)
                            ],
                        )
                        cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0)
                else:
                    pc = dict(
                        tma_ld=tma_ld,
                        tma_st=tma_st,
                        s_state=s_state,
                        sidx=sidx,
                        hv0=hv0,
                        ph=ph,
                        f_gdone=f_gdone,
                        st_full=st_full,
                        mma_slot=mma_slot,
                    )
                    if cutlass.const_expr(self.hloop):
                        for hh in cutlass.range(1, HPC, unroll=1):
                            self._produce_verify(hh, pc)
                    else:
                        for hh in cutlass.range_constexpr(1, HPC):
                            self._produce_verify(hh, pc)
            elif warp == 5:
                # ============================================================ UMMA issuer
                tbase = tmem_addr[0]
                sd = U.sdesc_sw128(V_DIM * 128)
                x_base = sd | Uint64(x_addr >> 4)
                stage_base = [
                    sd | Uint64(st_addr >> 4),
                    sd | Uint64((st_addr + 16384) >> 4),
                ]
                # C / K_hist tiles: W_CHUNKS independent W_CHUNK-row tiles side by side (chunk
                # wc at its own byte offset); one descriptor per chunk, each built exactly like
                # the W_CHUNK=16 kernel's single tile. MN-major SW128: SBO = 1024 B between
                # 8-slot K atoms, N halves 2048 B apart.
                c_desc = [
                    U.sdesc_sw32(Uint64((c_addr + wc * (W_CHUNK * 16 * 2)) >> 4))
                    for wc in range(W_CHUNKS)
                ]
                kt_desc = [
                    U.sdesc_mn_sw128(2048, 1024)
                    | Uint64((kt_addr + wc * (W_CHUNK * 128 * 2)) >> 4)
                    for wc in range(W_CHUNKS)
                ]
                d_step = W_CHUNK // 2  # TMEM columns per window chunk (fp16x2-packed)
                d_width = (
                    W_RING // 2
                )  # total D width (all chunks); verify-path parity stride
                idesc16 = U.f16_idesc(128, 16)
                idesc64mn = U.f16_idesc(128, 64, b_mn_major=True)
                cute.arch.mbarrier_wait(x_ready, ph)
                U.fence_after_sync()
                if is_flush:
                    # per head: fold half 0 (needs only D, Kt) -> readout half 0 (rd0: the
                    # consumers' RMW of stage 0 may start) -> readout half 1 + history
                    # (mma_done) -> fold half 1 once the RMW has read fold half 0 (f_epi_a).
                    # R/Y (cols 0..31) and F (64..127) of head h + 1 wait for head h's epilogue.
                    ic = dict(
                        tbase=tbase,
                        stage_base=stage_base,
                        x_base=x_base,
                        idesc16=idesc16,
                        idesc64mn=idesc64mn,
                        c_desc=c_desc,
                        kt_desc=kt_desc,
                        d_step=d_step,
                        d_width=d_width,
                        P=P,
                        ph=ph,
                        mma_slot=mma_slot,
                        st_full=st_full,
                        d_ready=d_ready,
                        mma_done=mma_done,
                        epi_done=epi_done,
                        f_mma_a=f_mma_a,
                        f_mma_b=f_mma_b,
                        f_epi_a=f_epi_a,
                        rd0=rd0,
                    )
                    if cutlass.const_expr(self.hloop):
                        for hh in cutlass.range(HPC, unroll=1):
                            self._issue_flush_head(hh, ic)
                    else:
                        for hh in cutlass.range_constexpr(HPC):
                            self._issue_flush_head(hh, ic)
                else:
                    ic = dict(
                        tbase=tbase,
                        stage_base=stage_base,
                        x_base=x_base,
                        idesc16=idesc16,
                        idesc64mn=idesc64mn,
                        c_desc=c_desc,
                        kt_desc=kt_desc,
                        d_step=d_step,
                        d_width=d_width,
                        P=P,
                        ph=ph,
                        mma_slot=mma_slot,
                        st_full=st_full,
                        d_ready=d_ready,
                        mma_done=mma_done,
                        epi_done=epi_done,
                        f_mma_a=f_mma_a,
                        f_mma_b=f_mma_b,
                        f_epi_a=f_epi_a,
                        rd0=rd0,
                    )
                    if cutlass.const_expr(self.hloop):
                        for hh in cutlass.range(HPC, unroll=1):
                            self._issue_verify_head(hh, ic)
                    else:
                        for hh in cutlass.range_constexpr(HPC):
                            self._issue_verify_head(hh, ic)
                cute.arch.mbarrier_wait(tmem_done, ph)
                U.fence_after_sync()
                U.tmem_dealloc(tbase, _TMEM_COLS)
            elif warp == 6:
                # ============================================================ next-head window copies
                if cutlass.const_expr(HPC > 1):
                    self._window_warp(
                        dict(
                            P=P,
                            base=base,
                            u_free=u_free,
                            u_full=u_full,
                            u_g=u_g,
                            sidx=sidx,
                            hv0=hv0,
                            u_cache=u_cache,
                            uw_addr=uw_addr,
                        )
                    )

    @cute.jit
    def _produce_flush(self, hh, c):
        """Producer warp, flush row: both halves of head hh - 1 are folded: store each, then load the same stage with the
        same half of head hh."""
        tma_ld, tma_st, s_state, sidx, hv0, ph = (
            c["tma_ld"],
            c["tma_st"],
            c["s_state"],
            c["sidx"],
            c["hv0"],
            c["ph"],
        )
        for g in cutlass.range_constexpr(2):
            prev = 2 * (hh - 1) + g
            slot = 2 * hh + g
            cute.arch.mbarrier_wait(c["f_gdone"][prev], ph)
            U.simple_tma_copy(
                tma_st.atom,
                s_state[None, (None, g)],
                tma_st.tma_tensor[sidx, hv0 + hh - 1, None, (None, g)],
            )
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(c["st_full"][slot], 16384)
            U.simple_tma_copy(
                tma_ld.atom,
                tma_ld.tma_tensor[sidx, hv0 + hh, None, (None, g)],
                s_state[None, (None, g)],
                c["st_full"][slot],
            )

    @cute.jit
    def _produce_verify(self, hh, c):
        """Producer warp, verify row: head hh's two halves, each into the stage the readout of head hh - 1 has freed."""
        tma_ld, s_state, sidx, hv0, ph = (
            c["tma_ld"],
            c["s_state"],
            c["sidx"],
            c["hv0"],
            c["ph"],
        )
        for g in cutlass.range_constexpr(2):
            slot = 2 * hh + g
            cute.arch.mbarrier_wait(c["mma_slot"][slot - 2], ph)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(c["st_full"][slot], 16384)
            U.simple_tma_copy(
                tma_ld.atom,
                tma_ld.tma_tensor[sidx, hv0 + hh, None, (None, g)],
                s_state[None, (None, g)],
                c["st_full"][slot],
            )

    @cute.jit
    def _issue_verify_head(self, hh, c):
        """Issuer warp, verify row, head hh: readout of both state halves into TMEM columns (hh % 2) * 32, then the
        history Y = D . C^T."""
        tbase, stage_base, x_base, idesc16, c_desc = (
            c["tbase"],
            c["stage_base"],
            c["x_base"],
            c["idesc16"],
            c["c_desc"],
        )
        d_step, d_width, P, ph = c["d_step"], c["d_width"], c["P"], c["ph"]
        rcol = (hh & 1) * 32
        if hh >= 2:
            cute.arch.mbarrier_wait(c["epi_done"][hh - 2], ph)
            U.fence_after_sync()
        for g in cutlass.range_constexpr(2):
            slot = 2 * hh + g
            cute.arch.mbarrier_wait(c["st_full"][slot], ph)
            U.fence_after_sync()
            for ka in cutlass.range_constexpr(4):
                U.umma_ss(
                    tbase + rcol,
                    stage_base[g] | Uint64((ka * 32) >> 4),
                    x_base | Uint64((g * 2048 + ka * 32) >> 4),
                    idesc16,
                    (g > 0) or (ka > 0),
                )
            U.umma_commit(c["mma_slot"][slot])
        cute.arch.mbarrier_wait(c["d_ready"][hh], ph)
        U.fence_after_sync()
        for ka in cutlass.range_constexpr(W_CHUNKS):
            if P > (-1 if ka == 0 else ka * W_CHUNK):
                U.umma_ts(
                    tbase + rcol + 16,
                    tbase + 64 + (hh & 1) * d_width + ka * d_step,
                    c_desc[ka],
                    idesc16,
                    ka > 0,
                )
        U.umma_commit(c["mma_done"][hh])

    @cute.jit
    def _issue_flush_head(self, hh, c):
        """Issuer warp, flush row, head hh: fold half 0 -> readout half 0 (rd0) -> readout half 1 + history (mma_done)
        -> fold half 1 once the RMW has read fold half 0 (f_epi_a)."""
        tbase, stage_base, x_base, idesc16, idesc64mn = (
            c["tbase"],
            c["stage_base"],
            c["x_base"],
            c["idesc16"],
            c["idesc64mn"],
        )
        c_desc, kt_desc, d_step, P, ph = (
            c["c_desc"],
            c["kt_desc"],
            c["d_step"],
            c["P"],
            c["ph"],
        )
        cute.arch.mbarrier_wait(c["d_ready"][hh], ph)
        if hh > 0:
            cute.arch.mbarrier_wait(c["epi_done"][hh - 1], ph)
        U.fence_after_sync()
        for ka in cutlass.range_constexpr(W_CHUNKS):
            # window chunk ka holds rows j in [ka*W_CHUNK, (ka+1)*W_CHUNK): when P
            # does not reach it, D and K_hist are exact zeros there (chunk 0 always runs)
            if P > (-1 if ka == 0 else ka * W_CHUNK):
                U.umma_ts(
                    tbase + 64, tbase + 32 + ka * d_step, kt_desc[ka], idesc64mn, ka > 0
                )
        U.umma_commit(c["f_mma_a"][hh])
        for g in cutlass.range_constexpr(2):
            slot = 2 * hh + g
            cute.arch.mbarrier_wait(c["st_full"][slot], ph)
            U.fence_after_sync()
            for ka in cutlass.range_constexpr(4):
                U.umma_ss(
                    tbase + 0,
                    stage_base[g] | Uint64((ka * 32) >> 4),
                    x_base | Uint64((g * 2048 + ka * 32) >> 4),
                    idesc16,
                    (g > 0) or (ka > 0),
                )
            if cutlass.const_expr(g == 0):
                U.umma_commit(c["rd0"][hh])
        for ka in cutlass.range_constexpr(W_CHUNKS):
            if P > (-1 if ka == 0 else ka * W_CHUNK):
                U.umma_ts(
                    tbase + 16, tbase + 32 + ka * d_step, c_desc[ka], idesc16, ka > 0
                )
        U.umma_commit(c["mma_done"][hh])
        cute.arch.mbarrier_wait(c["f_epi_a"][hh], ph)
        U.fence_after_sync()
        for ka in cutlass.range_constexpr(W_CHUNKS):
            if P > (-1 if ka == 0 else ka * W_CHUNK):
                U.umma_ts(
                    tbase + 64,
                    tbase + 32 + ka * d_step,
                    kt_desc[ka] + Uint64(2048 >> 4),
                    idesc64mn,
                    ka > 0,
                )
        U.umma_commit(c["f_mma_b"][hh])

    def _x_row_addr(self, k, q, k_g, q_g, brow, kh, r):
        """Global byte address of X row r (k_t, then q_t; rows >= 2T -> k_0, never read)."""
        T = self.T
        r = Int32(r)
        is_q = (r >= T) & (r < 2 * T)
        t_k = _sel(r < T, r, 0, Int32)
        t_q = _sel(is_q, r - T, 0, Int32)
        ak = k_g + Int64(cute.crd2idx((brow, t_k, kh, 0), k.layout)) * 2
        aq = q_g + Int64(cute.crd2idx((brow, t_q, kh, 0), q.layout)) * 2
        return _sel(is_q, aq, ak, Int64)

    def _ring_k_rows(self, wc, k_cache, sidx, kh, base, P, grp, kc_g):
        off = wc * W_CHUNK
        wlim = _sel(P > 0, P - 1, Int32(0), Int32)
        r0 = _sel(grp + off < wlim, grp + off, wlim, Int32)
        r1 = _sel(grp + 8 + off < wlim, grp + 8 + off, wlim, Int32)
        ra0 = (
            kc_g
            + Int64(
                cute.crd2idx((sidx, kh, (base + r0) & RING_MASK, 0), k_cache.layout)
            )
            * 2
        )
        ra1 = (
            kc_g
            + Int64(
                cute.crd2idx((sidx, kh, (base + r1) & RING_MASK, 0), k_cache.layout)
            )
            * 2
        )
        return ra0, ra1

    def _ring_k_loads(self, wc, kc_g, k_cache, sidx, kh, base, P, grp, c4):
        ra0, ra1 = self._ring_k_rows(wc, k_cache, sidx, kh, base, P, grp, kc_g)
        wr0c = []
        wr1c = []
        for p in range(4):
            koff = Int64((32 * p + 8 * c4) * 2)
            wr0c.append(U.ldg_v4(ra0 + koff))
            wr1c.append(U.ldg_v4(ra1 + koff))
        return wr0c, wr1c

    def _gram_chunk(self, wc, wr0c, wr1c, wb, s_gram, grp, gc):
        off = wc * W_CHUNK
        accr = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
        for p in range(4):
            for hs in range(2):
                accr = U.mma_bf16_16816(
                    wr0c[p][2 * hs],
                    wr1c[p][2 * hs],
                    wr0c[p][2 * hs + 1],
                    wr1c[p][2 * hs + 1],
                    wb[p][2 * hs],
                    wb[p][2 * hs + 1],
                    accr,
                )
        s_gram[(grp + off) * 16 + gc] = accr[0]
        s_gram[(grp + off) * 16 + gc + 1] = accr[1]
        s_gram[(grp + 8 + off) * 16 + gc] = accr[2]
        s_gram[(grp + 8 + off) * 16 + gc + 1] = accr[3]

    def _ring_k_prefetch_l1(self, kc_g, k_cache, sidx, kh, base, P, lane, row0):
        """One warp, 32 lanes: prefetch.global.L1 of the two 128 B halves of the 16 window rows row0 .. row0 + 15 that
        lie below min(P, nu) (lane -> (row, half))."""
        j = row0 + (lane >> 1)
        half = lane & 1
        nl = _sel(P < self.nu, P, Int32(self.nu), Int32)
        ra = (
            kc_g
            + Int64(cute.crd2idx((sidx, kh, (base + j) & RING_MASK, 0), k_cache.layout))
            * 2
            + Int64(half * 128)
        )
        U.prefetch_l1_line_if(Int32(j < nl), ra)

    def _v_prefetch_l1(self, v, brow, hv0, lane, rr):
        """One warp, 32 lanes: prefetch.global.L1 of the 128 B lines of v[brow, t, hv0 + h, :] (256 B = two lines) for
        t < T, h < hpc; lines 32 * rr .. 32 * rr + 31 of at most 64 (lane -> (t, h, half)). Returns the predicate
        of the lanes whose line exists."""
        li = rr * 32 + lane
        half = li & 1
        h = (li >> 1) & (self.hpc - 1)
        t = li >> self.hpc.bit_length()
        va = (
            v.iterator.toint()
            + Int64(
                cute.crd2idx(
                    (brow, _sel(t < self.T, t, Int32(0), Int32), hv0 + h, 0), v.layout
                )
            )
            * 2
            + Int64(half * 128)
        )
        return Int32(t < self.T), va

    def _u_window_copy(self, u_cache, u_g, uw_addr, mbar, sidx, hv, base, P):
        """Bulk-copy the live logical window slots j = 0..min(P, nu)-1 (physical (base + j) & RING_MASK)
        of one value head's u ring into the single window buffer as [j][V]. Rows j >= P are not copied (they are
        zero-filled once per CTA, and their weights are +0). Call under elect_one."""
        # at most two contiguous copies (the window wraps the ring at most once) instead of one per row
        src_head = u_g + Int64(cute.crd2idx((sidx, hv, 0, 0), u_cache.layout)) * 2
        self._window_copy_pair(
            ctx_P=P,
            base=base,
            src_head=src_head,
            dst=uw_addr,
            rowb=V_DIM * 2,
            mbar=mbar,
        )

    @cute.jit
    def _window_bulk(self, nn0, dst, src_head, base, n1, n2, d2, s2, rowb, mbar):
        """The two contiguous window copies (nothing at all while no row is live)."""
        if nn0 > 0:
            U.bulk_g2s(dst, src_head + Int64(base * rowb), n1 * rowb, mbar.toint())
            U.bulk_g2s(
                dst + d2 * rowb, src_head + Int64(s2 * rowb), n2 * rowb, mbar.toint()
            )

    def _window_copy_pair(self, ctx_P, base, src_head, dst, rowb, mbar):
        """Live window rows j < min(P, nu) (physical (base + j) & RING_MASK) as at most two contiguous bulk copies (the window
        wraps the ring at most once); without a wrap the second copy re-copies the last row (same bytes), so
        the transaction count is (n1 + n2) rows either way and no copy has size 0. P == 0: no rows, the arrival alone
        completes the barrier and the zero-filled window stays zero. Call under elect_one / from one lane."""
        nn0 = _sel(ctx_P < self.nu, ctx_P, Int32(self.nu), Int32)
        nn = _sel(nn0 > 0, nn0, Int32(1), Int32)
        wrap = (base + nn) > RING_SLOTS
        n1 = _sel(wrap, Int32(RING_SLOTS) - base, nn, Int32)
        n2 = _sel(wrap, nn - n1, Int32(1), Int32)
        d2 = _sel(wrap, n1, nn - 1, Int32)
        s2 = _sel(wrap, Int32(0), base + nn - 1, Int32)
        cute.arch.mbarrier_arrive_and_expect_tx(
            mbar, _sel(nn0 > 0, (n1 + n2) * rowb, Int32(0), Int32)
        )
        self._window_bulk(nn0, dst, src_head, base, n1, n2, d2, s2, rowb, mbar)

    @cute.jit
    def _u_release(self, hh, ctx, uv):
        """Head hh's window rows are in registers: release the buffer (the next head's copy is issued by warp 6). The
        release must come after the rows were actually read: an LDS still queued in the shared-memory pipe when the
        arrive (a different unit) takes effect lets the bulk copy's rows land first (a launch-to-launch race). Storing
        a value that depends on all rows to a dummy word forces their data to be back first (a predicate or ALU use
        would be folded: the rows are known to be < 2^16)."""
        if hh < self.hpc - 1:
            dep = uv[0]
            for u_ in uv[1:]:
                dep = dep | u_
            U.sts_u32(ctx["sink"], dep)
            cute.arch.mbarrier_arrive(ctx["u_free"][0])

    @cute.jit
    def _window_warp(self, c):
        """Warp 6 (otherwise idle): once every consumer has released the window buffer (u_free completion hh), copy head
        hh + 1's LIVE window rows j < min(P, nu) (slot (base + j) & RING_MASK -> row j) into it, with no consumer on the
        path: two contiguous copies from one lane."""
        rowb = V_DIM * 2
        P, base = c["P"], c["base"]
        for hh in cutlass.range(self.hpc - 1, unroll=1):
            cute.arch.mbarrier_wait(c["u_free"][0], hh & 1)
            mbar = c["u_full"][hh + 1]
            src_head = (
                c["u_g"]
                + Int64(
                    cute.crd2idx(
                        (c["sidx"], c["hv0"] + hh + 1, 0, 0), c["u_cache"].layout
                    )
                )
                * 2
            )
            with cute.arch.elect_one():
                self._window_copy_pair(P, base, src_head, c["uw_addr"], rowb, mbar)

    @cute.jit
    def _d_chunks_deep(self, hh, dcol, ctx, ub):
        """D words of the window chunks >= 1 (rows j >= W_CHUNK) of head hh into TMEM columns
        dcol + wc * W_CHUNK / 2 ...; skipped (CTA-uniform branch) for a chunk the live window
        P does not reach -- its history / fold MMAs are skipped too, so nothing reads it."""
        P, ct, tbase, s_coef = ctx["P"], ctx["ct"], ctx["tbase"], ctx["s_coef"]
        cb = hh * _COEF
        for wc in cutlass.range_constexpr(1, W_CHUNKS):
            if P > wc * W_CHUNK:
                nrow = min(
                    W_CHUNK, self.nu - wc * W_CHUNK
                )  # rows >= nu are never live (read as 0)
                uvc = [
                    U.lds_u16(ub + (wc * W_CHUNK + j) * (V_DIM * 2))
                    for j in range(nrow)
                ]
                words = []
                for c in cutlass.range_constexpr(W_CHUNK // 2):
                    j0 = wc * W_CHUNK + 2 * c
                    lo = Float32(0.0)
                    hi = Float32(0.0)
                    if cutlass.const_expr(2 * c < nrow):
                        lo = (
                            U.u16_as_f32(uvc[2 * c], self.u_bf16)
                            * s_coef[cb + _CO_W + j0]
                        )
                    if cutlass.const_expr(2 * c + 1 < nrow):
                        hi = (
                            U.u16_as_f32(uvc[2 * c + 1], self.u_bf16)
                            * s_coef[cb + _CO_W + j0 + 1]
                        )
                    words.append(U.pack_f16x2(lo, hi))
                U.tmem_st_u32(ct, tbase + dcol + wc * (W_CHUNK // 2), words)

    def _d_build(self, hh, dcol, ctx):
        """D row w_j u_j (j < P, fp16 pairs) of head hh into TMEM columns dcol..dcol+W_RING/2-1;
        arrives d_ready[hh].  Also returns this row's raw v values of head hh (issued first
        so their latency overlaps the D build and the readout wait)."""
        T = self.T
        row, ct, tbase, s_coef = (ctx["row"], ctx["ct"], ctx["tbase"], ctx["s_coef"])
        v = ctx["v"]
        cb = hh * _COEF
        vv = [v[ctx["brow"], t, ctx["hv0"] + hh, row] for t in range(T)]
        cute.arch.mbarrier_wait(ctx["u_full"][hh], ctx["ph"])
        ub = ctx["uw_addr"] + row * 2
        uv = [U.lds_u16(ub + j * (V_DIM * 2)) for j in range(W_CHUNK)]
        self._d_chunks_deep(
            hh, dcol, ctx, ub
        )  # chunks >= 1: loads, words and TMEM store
        # single-buffered window: release it now (all rows are in registers); warp 6 refills it with the next head's
        # live rows, so their DRAM round trip overlaps this head's D math, the readout wait and the epilogue
        self._u_release(hh, ctx, uv)
        words = []
        for c in range(W_CHUNK // 2):
            lo = U.u16_as_f32(uv[2 * c], self.u_bf16) * s_coef[cb + _CO_W + 2 * c]
            hi = (
                U.u16_as_f32(uv[2 * c + 1], self.u_bf16)
                * s_coef[cb + _CO_W + 2 * c + 1]
            )
            words.append(U.pack_f16x2(lo, hi))
        U.tmem_st_u32(ct, tbase + dcol, words)
        U.tmem_wait_st()
        U.fence_before_sync()
        cute.arch.mbarrier_arrive(ctx["d_ready"][hh])
        return vv

    def _epilogue(self, hh, rcol, ctx, vv):
        """Outputs and u appends of head hh from the readout R (cols rcol..) and history Y
        (cols rcol + 16..) TMEM columns; arrives epi_done[hh] once its TMEM reads are done."""
        T = self.T
        sidx, base, P, brow, hv0 = (
            ctx["sidx"],
            ctx["base"],
            ctx["P"],
            ctx["brow"],
            ctx["hv0"],
        )
        ct, row, tbase = ctx["ct"], ctx["row"], ctx["tbase"]
        u_cache, out, s_coef = ctx["u_cache"], ctx["out"], ctx["s_coef"]
        hv = hv0 + hh
        cb = hh * _COEF
        nld = 1 << (T - 1).bit_length()  # tcgen05.ld widths are powers of two
        rk = U.tmem_ld(ct, tbase + rcol, nld)
        yk = U.tmem_ld(ct, tbase + rcol + 16, nld)
        U.tmem_wait_ld()
        egp = s_coef[cb + _CO_EGP]
        # all four column groups first, then release the head buffer (the readout two heads ahead waits on it)
        hks = [egp * rk[t] + yk[t] for t in range(T)]
        rq = U.tmem_ld(ct, tbase + rcol + T, nld)
        yq = U.tmem_ld(ct, tbase + rcol + 16 + T, nld)
        U.tmem_wait_ld()
        U.fence_before_sync()
        cute.arch.mbarrier_arrive(ctx["epi_done"][hh])
        uu = []
        for t in range(T):
            hk = hks[t]
            acc = (
                s_coef[cb + _CO_BETA + t] * Float32(vv[t])
                - s_coef[cb + _CO_BG + t] * hk
            )
            for s_ in range(t):
                acc = acc - s_coef[cb + _a_off(t, s_)] * uu[s_]
            uu.append(acc)

        ring_t = BFloat16 if self.u_bf16 else Float16
        for t in range(T):
            hq = egp * rq[t] + yq[t]
            y = s_coef[cb + _CO_EG + t] * hq
            for s_ in range(t + 1):
                y = y + s_coef[cb + _bq_off(t, s_)] * uu[s_]
            out[brow, t, hv, row] = BFloat16(y)
            u_cache[sidx, hv, (base + P + t) & RING_MASK, row] = ring_t(uu[t])

    def _load_seed(self, ctx, rand_seed):
        """SR variants, flush rows only: the Philox key / LCG mixing key (verify rows never
        read the seed)."""
        if self.sr_rounds > 0 or self.sr == "lcg":
            seed = Int64(rand_seed[0])
            ctx["seed_lo"] = Uint32(seed & Int64(0xFFFFFFFF))
            ctx["seed_hi"] = Uint32(seed >> Int64(32))
        if self.sr == "lcg":
            ctx["lcg_k"] = U.fmix32(
                ctx["seed_lo"] ^ U.fmix32(ctx["seed_hi"] ^ Uint32(0x9E3779B9))
            )
        elif self.sr == "lcg_clock":
            ctx["lcg_k"] = U.fmix32(U.clock32())

    @cute.jit
    def _f_epi_release(self, hh, g, ctx):
        if g == 0:
            U.fence_before_sync()
            cute.arch.mbarrier_arrive(
                ctx["f_epi_a"][hh]
            )  # F may be overwritten (fold half 1)

    @cute.jit
    def _flush_halves(self, hh, ctx):
        for g in cutlass.range(2, unroll=1):
            self._flush_half(hh, g, ctx)

    @cute.jit
    def _flush_half(self, hh, g, ctx):
        """Wait for half g of head hh (g = 0: fold half 0 and the half-0 readout; g = 1: fold half 1 and the
        history), then fold it into the resident stage."""
        if g == 0:
            cute.arch.mbarrier_wait(ctx["f_mma_a"][hh], ctx["ph"])
            cute.arch.mbarrier_wait(ctx["rd0"][hh], ctx["ph"])
        else:
            cute.arch.mbarrier_wait(ctx["f_mma_b"][hh], ctx["ph"])
            cute.arch.mbarrier_wait(ctx["mma_done"][hh], ctx["ph"])
        U.fence_after_sync()
        self._rmw(hh, g, ctx)

    def _rmw(self, hh, g, ctx):
        """Flush rows: S_half = e^{G_P} S0_half + F (TMEM cols 64..127) in place on state stage
        g (= half g of head hh), then publish it to the producer warp's TMA store (f_gdone)."""
        ct, row, tbase, s_coef = ctx["ct"], ctx["row"], ctx["tbase"], ctx["s_coef"]
        egp = s_coef[hh * _COEF + _CO_EGP]
        srow = ctx["st_addr"] + g * 16384 + row * 128
        if self.sr_rounds > 0:
            # flat element offset of S[sidx, hv, row, 64 g] in the [pool, HV, V, K] state pool
            # (FlashInfer Mamba's Philox counter: state offset + row * K + column)
            off_row = (
                (Int64(ctx["sidx"]) * self.hv + Int64(ctx["hv0"] + hh)) * V_DIM
                + Int64(row)
            ) * K_DIM + g * 64
        # TMEM columns per load (and Philox streams per call = its 8-element chunks): 32 for
        # RN and for Philox with 4 heads per CTA (many CTAs, issue throughput); 16 for Philox
        # with fewer heads per CTA (few CTAs, the RMW latency is the kernel time and the
        # smaller live set halves it: B = 1, 5 rounds +8% vs +17%) and for LCG always (fewer
        # spills; T = 8, B = 512, 100% fold +6% vs +9%)
        ldn = 16 if (self.sr_rounds > 0 and self.hpc < HPK) or self.sr_lcg else 32
        nch = ldn // 8  # 8-element chunks per load
        nst = nch  # Philox streams per call
        if self.sr_lcg:
            # LCG dither (the ameyn/gdn-ucache-sr recipe, 3 instructions per pair): this
            # (state row, half)'s seed = murmur3 of its flat id and the per-flush key; pair n
            # (= 4 chunk + i) takes LCG state n folded as s ^ (s >> 16), run as 4 jump-ahead
            # chains (x_i -> A^4 x_i + C4) so the 4 pairs of a 16-B word are independent
            # instructions. Known weakness (same as that branch): the two 13-bit fields of a
            # word XOR to the state's low bits and consecutive words are consecutive LCG
            # states, so dithers within one (row, half) are not jointly independent. A PCG
            # RXS-M-XS output fixes that but costs ~7 instructions per word -- as much as
            # 5-round Philox (measured), so use "philox" when that matters.
            cid = (
                (Int64(ctx["sidx"]) * self.hv + Int64(ctx["hv0"] + hh)) * V_DIM
                + Int64(row)
            ) * 2 + g
            x0 = U.fmix32(
                Uint32(cid & Int64(0xFFFFFFFF)) * Uint32(0x9E3779B9)
                + Uint32(cid >> Int64(32)) * Uint32(0x85EBCA6B)
                + ctx["lcg_k"]
            )
            lx = [x0]
            for _ in range(3):
                lx.append(lx[-1] * Uint32(U.LCG_A) + Uint32(U.LCG_C))
        for grp in range(64 // ldn):
            fv = U.tmem_ld(ct, tbase + 64 + grp * ldn, ldn)
            if self.sr_rounds > 0:
                # one Philox draw per 8-element chunk (word i -> pair i), all of the load's
                # chunks in one call, issued under the TMEM load. off_row is a multiple of 64, so
                # the chunk offsets never carry into the high word.
                off = off_row + grp * ldn
                off_lo, off_hi = (
                    Uint32(off & Int64(0xFFFFFFFF)),
                    Uint32(off >> Int64(32)),
                )
                rws = U.philox4x32_streams(
                    ctx["seed_lo"], ctx["seed_hi"], off_lo, off_hi, nst, self.sr_rounds
                )
            U.tmem_wait_ld()
            if grp == 64 // ldn - 1:
                self._f_epi_release(hh, g, ctx)
            for cc in range(nch):
                chunk = grp * nch + cc
                sa = srow + ((chunk ^ (row % 8)) * 16)
                ow = U.lds_v4(sa)
                if self.sr_lcg:
                    rw = [
                        x ^ (x >> Uint32(16)) for x in lx
                    ]  # 16-bit fold (ameyn/gdn-ucache-sr)
                    lx = [x * Uint32(U.LCG_A4) + Uint32(U.LCG_C4) for x in lx]
                nw = []
                for i in range(4):
                    lo = egp * U.f16x2_lo_f32(ow[i]) + fv[cc * 8 + 2 * i]
                    hi = egp * U.f16x2_hi_f32(ow[i]) + fv[cc * 8 + 2 * i + 1]
                    if self.sr_rounds > 0:
                        nw.append(U.pack_f16x2_rs(lo, hi, rws[cc % nst][i]))
                    elif self.sr_lcg:
                        nw.append(U.pack_f16x2_rs(lo, hi, rw[i]))
                    else:
                        nw.append(U.pack_f16x2(lo, hi))
                U.sts_v4(sa, nw[0], nw[1], nw[2], nw[3])
        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_arrive(ctx["f_gdone"][2 * hh + g])

    @cute.jit
    def _head_u(self, hh, ctx):
        """Head hh of a flush or a verify row (runtime is_flush): D build, then the flush row's two RMW halves or the
        verify row's wait for the history, then the epilogue with the early TMEM release for both."""
        is_flush = ctx["is_flush"]
        dcol = _sel(is_flush, Int32(32), 64 + (hh & 1) * (W_RING // 2), Int32)
        rcol = _sel(is_flush, Int32(0), (hh & 1) * 32, Int32)
        vv = self._d_build(hh, dcol, ctx)
        if is_flush:
            self._flush_halves(hh, ctx)
        else:
            cute.arch.mbarrier_wait(ctx["mma_done"][hh], ctx["ph"])
            U.fence_after_sync()
        self._epilogue(hh, rcol, ctx, vv)

    # ------------------------------------------------------------------ compile
    @cache
    @staticmethod
    def compile(
        device: torch.device,
        T: int,
        h: int,
        hv: int,
        u_bf16: bool,
        use_row_order: bool,
        hpc: int = HPK,
        sr: str = "",
        sr_rounds: int = 0,
        pmax: int = 0,
    ):
        batch = cute.sym_int()
        pool = cute.sym_int()
        ft = U.make_fake_tensor_lead_dyn
        uring = BFloat16 if u_bf16 else Float16  # u ring element type
        args = [
            ft(BFloat16, (batch, T, h, K_DIM), 8),  # q
            ft(BFloat16, (batch, T, h, K_DIM), 8),  # k
            ft(BFloat16, (batch, T, hv, V_DIM), 8),  # v
            ft(BFloat16, (batch, T, hv), 1),  # a
            ft(BFloat16, (batch, T, hv), 1),  # b
            ft(Float32, (hv,), 1),  # A_log (fp32)
            ft(Float32, (hv,), 1),  # dt_bias (fp32)
            ft(Float16, (pool, hv, V_DIM, K_DIM), 8),  # state
            ft(Int32, (batch,), 1),  # state indices
            ft(BFloat16, (pool, h, RING_SLOTS, K_DIM), 8),  # k ring (bf16)
            ft(uring, (pool, hv, RING_SLOTS, V_DIM), 8),  # u ring (bf16 / fp16)
            ft(Float32, (pool, hv, RING_SLOTS), 1),  # g ring
            ft(Int32, (batch,), 1),  # hist_len
            ft(Int32, (batch,), 1),  # cache_base
            ft(Int32, (batch,), 1),  # row order
            ft(BFloat16, (batch, T, hv, V_DIM), 8),  # out
            ft(Int64, (1,), 1),  # rand_seed (read by "philox" / "lcg")
        ]
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile[gdn_compile_options(device, cute.EnableTVMFFI(True))](
            GdnReplayMtpUmma(
                T,
                h,
                hv,
                u_bf16,
                use_row_order,
                hpc,
                sr,
                sr_rounds,
                pmax,
            ),
            *args,
            Float32(1.0),
            Int32(1),
            stream,
        )


# ============================================================================ host side
_F32_CACHE: dict = {}
_ROW_ID: dict = {}


def _cached_f32(t: torch.Tensor) -> torch.Tensor:
    """fp32 copy of a per-layer constant (A_log / dt_bias), cached by object identity."""
    if t.dtype == torch.float32 and t.is_contiguous():
        return t
    import weakref

    key = id(t)
    c = _F32_CACHE.get(key)
    if c is None:
        c = t.float().contiguous()
        _F32_CACHE[key] = c
        weakref.finalize(t, _F32_CACHE.pop, key, None)
    return c


_CVT_RS_CC = (
    (10, 0),
    (10, 3),
    (10, 7),
)  # archs whose ptxas accepts cvt.rs (see mamba/conversion.cuh)
_CTA_SLOTS_PER_SM = (
    _SMEM_CTAS_PER_SM  # 128 of 512 TMEM columns; SMEM per CTA decides the rest
)
# hpc = 2 up to this many waves of the hpc = 2 grid (see auto_heads_per_cta). Measured B200 crossovers (cold L2,
# fold 0-40 %, T = 4 and 8, verify-first consumer dispatch): hpc = 2 up to B ~ 52-54, hpc = 4 beyond (3 waves of the
# hpc = 2 grid = B <= 55 at 148 SMs).
_HPC2_MAX_WAVES = 3
_NUM_SMS: dict = {}


def auto_heads_per_cta(batch: int, n_key_heads: int, num_sms: int, T: int = 4) -> int:
    """Heads per CTA for a batch: 1 while the hpc = 1 grid fits in one wave of CTA slots,
    2 while the hpc = 2 grid fits in a few waves (``_HPC2_MAX_WAVES``), else 4.

    One CTA runs its value heads one after another, so splitting a key head over 2 or 4
    CTAs (each repeating the small per-key-head prologue) shortens the per-CTA chain and
    gives the scheduler finer-grained work. hpc = 1 only pays off while everything runs in
    one wave; hpc = 2 keeps winning over hpc = 4 well past one wave. B200 (148 SMs, 4 CTAs/SM,
    the prologue is twice as long as the 16-row-window kernel's): 1 for B <= 9, 2 for B <= 55
    (3 waves of the hpc = 2 grid), else 4, for every T."""
    slots = _CTA_SLOTS_PER_SM * num_sms
    if batch * n_key_heads * HPK <= slots:
        return 1
    if batch * n_key_heads * (HPK // 2) <= _HPC2_MAX_WAVES * slots:
        return 2
    return HPK


def _num_sms(device: torch.device) -> int:
    key = str(device)
    if key not in _NUM_SMS:
        _NUM_SMS[key] = torch.cuda.get_device_properties(device).multi_processor_count
    return _NUM_SMS[key]


def flush_first_row_order(hist_len: torch.Tensor, flush_min: int) -> torch.Tensor:
    """Permutation of batch rows with the flushing rows (hist_len >= flush_min) first."""
    return torch.argsort((hist_len < flush_min).to(torch.int32), stable=True).to(
        torch.int32
    )


def gated_delta_rule_mtp_ucache_flush_umma_ring64(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    initial_state_source: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    k_cache: Optional[torch.Tensor] = None,
    u_cache: Optional[torch.Tensor] = None,
    g_cache: Optional[torch.Tensor] = None,
    hist_len: Optional[torch.Tensor] = None,
    cache_base: Optional[torch.Tensor] = None,
    flush_min: Optional[int] = None,
    restart_hist_on_flush: bool = True,
    row_order: Optional[torch.Tensor] = None,
    heads_per_cta: Optional[int] = None,
    stochastic_rounding: Optional[str] = None,
    rand_seed: Optional[torch.Tensor] = None,
    philox_rounds: int = 10,
    window_rows: Optional[int] = None,
) -> torch.Tensor:
    """UMMA drop-in for ``gated_delta_rule_mtp_ucache_flush`` (fp16-state arm), 1 <= T <= 8.

    ``heads_per_cta`` (1, 2 or 4; None = by batch size, see ``auto_heads_per_cta``) sets how many
    of a key head's 4 value heads one CTA runs.

    ``window_rows`` (compile-time, part of the compile-cache key; default the full window
    nu = min(W_RING, RING_SLOTS - T)): rows of the shared-memory u window, i.e. the deepest live
    window P a call may present. The default holds any history the ring can carry, including rings
    advanced at a larger T than this call's. A smaller explicit value saves shared memory but silently
    truncates deeper histories.

    ``stochastic_rounding`` rounds the fp16 state written by flush rows stochastically
    (``cvt.rs.f16x2``, sm_100a / sm_103a / sm_107a); outputs, rings and verify rows are unchanged:

    * None (default): off; ``rand_seed`` alone does not turn it on.
    * "philox": FlashInfer Mamba's scheme and defaults -- Philox4x32 with ``philox_rounds``
      rounds, key = ``rand_seed``, counter = flat offset of the 8-element chunk in the state
      pool, one draw per chunk feeding 4 pairs. Reproducible per seed.
    * "lcg": the ``ameyn/gdn-ucache-sr`` dither -- one LCG step per pair, bits s ^ (s >> 16)
      -- started from a murmur3 hash of the (state row, half) id and the seed. Cheapest; each
      element is unbiased and flushes are independent, but dithers within one state row are
      not jointly independent (see ``_rmw``). With ``rand_seed`` it is reproducible per seed;
      without, the seed is %clock (not reproducible, like that branch).
    * "none": off.

    ``rand_seed`` is a single-element int64 CUDA tensor read on device, so a captured CUDA
    graph picks up updates; advance it every step (and use a different seed per layer), or
    repeated flushes of a slot reuse the same random bits.

    ``flush_min`` (default: the largest value that is safe for any acceptance pattern,
    ``nu - T + 1`` with ``nu = min(W_RING, RING_SLOTS - T)``, i.e. ``min(W_RING - T + 1, RING_SLOTS - 2 T + 1)``:
    13 / 9 at W_RING = 16 for T = 4 / 8, 25 / 17 at W_RING = RING_SLOTS = 32, 29 / 25 at W_RING = 32 on the 64-slot
    ring): a row with ``hist_len >= flush_min`` flushes. Values up to ``W_RING - T + 1`` are still accepted
    (single calls whose rows all have hist_len <= nu) but warn above the safe default: a verify row can then
    advance past nu, whose appends overwrite the live window (never the case at W_RING = 32 on the 64-slot ring,
    where the two bounds coincide). Contract: every call's hist_len <= nu (a deeper window is silently
    truncated): the flush threshold of the kernel AND of the host-side cursor commit must be the same flush_min.

    Same tensors, ring/flush contract and host-side cursor commit as FlashInfer's kernel
    (see the module docstring).  ``row_order`` (int32 [B], a permutation of rows, e.g.
    ``flush_first_row_order``) only changes which CTA runs which request.
    """
    assert softplus_beta == 1.0 and softplus_threshold == 20.0
    assert use_qk_l2norm_in_kernel
    assert q is not None and k is not None and v is not None and b is not None
    h0 = initial_state_source
    assert h0 is not None and h0.dtype == torch.float16, "fp16 state pool required"
    B, T, H, K_ = q.shape
    HV, V_ = v.shape[2], v.shape[3]
    assert K_ == K_DIM and V_ == V_DIM and HV // H == HPK and HV % H == 0
    assert 1 <= T <= 8, f"T={T} unsupported (1 <= T <= 8)"
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert (
        k_cache is not None
        and u_cache is not None
        and g_cache is not None
        and hist_len is not None
    )
    # the per-CTA Gram runs bf16 mma.sync on raw ring keys: a bf16 k ring; the u ring is read and appended through
    # dtype-aware conversions, so it may be bf16 or fp16 (vLLM's u-cache dtype)
    assert k_cache.dtype == torch.bfloat16, "bf16 k ring required"
    assert u_cache.dtype in (torch.bfloat16, torch.float16), (
        "bf16 or fp16 u ring required"
    )
    pool = h0.shape[0]
    assert tuple(k_cache.shape) == (pool, H, RING_SLOTS, K_DIM)
    assert tuple(u_cache.shape) == (pool, HV, RING_SLOTS, V_DIM)
    assert (
        tuple(g_cache.shape) == (pool, HV, RING_SLOTS)
        and g_cache.dtype == torch.float32
    )
    if scale is None:
        scale = 1.0 / math.sqrt(K_)
    # A verify row at P = flush_min - 1 appends up to T more rows (the caller advances P by 1..T per call), and
    # P must never exceed nu = min(W_RING, RING_SLOTS - T): the window tiles hold W_RING rows, the T appends of a
    # call land at slots P..P+T-1 of the RING_SLOTS-slot ring (beyond it they wrap onto the live window) and the
    # kernel only reads the first nu window rows. The flush_min that is safe for any acceptance pattern is
    # therefore nu - T + 1 = min(W_RING - T + 1, RING_SLOTS - 2 T + 1). At W_RING == 16 and at W_RING = 32 on the
    # 64-slot ring it is W_RING - T + 1 (the ring has headroom). At W_RING == RING_SLOTS == 32 the default
    # follows the ring (29 at T = 4 would be unsafe).
    nu = min(W_RING, RING_SLOTS - T)
    fm_safe = nu - T + 1  # = min(W_RING - T + 1, RING_SLOTS - 2 T + 1)
    fm_max = (
        W_RING - T + 1
    )  # unchanged upper bound: single calls may use up to it (e.g. all rows verify)
    if flush_min is None:
        # T = 1 appends one ring row per step, so a row flushes about every F + 1 steps: past one 16-row chunk the extra
        # window-row reads cost more than the flush writes they save (measured: F = 16 is 3 % faster than F = 32).
        flush_min = min(fm_safe, W_CHUNK) if T == 1 else fm_safe
    assert 1 <= flush_min <= fm_max, (
        f"flush_min={flush_min} outside [1, {fm_max}] (W_RING={W_RING}, RING_SLOTS={RING_SLOTS}, T={T}). "
        f"Omit flush_min for the safe default {fm_safe}."
    )
    if flush_min > fm_safe:
        warnings.warn(
            f"flush_min={flush_min} > {fm_safe} (= nu - T + 1, nu = min(W_RING, RING_SLOTS - T)) is only safe if "
            f"every call advances the history by exactly T tokens; with partial acceptance a verify row can reach "
            f"P > nu = {nu}, whose appends overwrite the live window.",
            stacklevel=2,
        )
    if window_rows is None:
        window_rows = nu
    assert 1 <= window_rows <= nu, (
        f"window_rows={window_rows} outside [1, {nu}] (W_RING={W_RING}, RING_SLOTS={RING_SLOTS}, T={T})"
    )
    pmax = (
        0 if window_rows == nu else window_rows
    )  # 0: the whole window (the compile key of a full build)
    if initial_state_indices is None:
        initial_state_indices = torch.arange(B, dtype=torch.int32, device=q.device)
    if cache_base is None:
        assert not restart_hist_on_flush, (
            "restart_hist_on_flush=True needs a caller-owned cache_base"
        )
        cache_base = torch.zeros_like(hist_len)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    a, b = a.contiguous().to(torch.bfloat16), b.contiguous().to(torch.bfloat16)
    if output is None:
        output = torch.empty(B, T, HV, V_, dtype=torch.bfloat16, device=q.device)
    use_ro = row_order is not None
    if not use_ro:
        ro_key = (B, str(q.device))
        row_order = _ROW_ID.get(ro_key)
        if row_order is None:
            row_order = torch.arange(B, dtype=torch.int32, device=q.device)
            _ROW_ID[ro_key] = row_order
    hpc = (
        auto_heads_per_cta(B, H, _num_sms(q.device), T)
        if heads_per_cta is None
        else int(heads_per_cta)
    )
    assert hpc in (1, 2, 4)
    mode = stochastic_rounding
    if mode is None:
        mode = "none"
    if mode not in ("none", "philox", "lcg"):
        raise ValueError(
            f"stochastic_rounding={stochastic_rounding!r}: use None, 'none', 'philox' or 'lcg'"
        )
    if mode == "philox" and rand_seed is None:
        raise ValueError("stochastic_rounding='philox' needs rand_seed")
    if mode != "none" and torch.cuda.get_device_capability(q.device) not in _CVT_RS_CC:
        raise NotImplementedError(
            "stochastic rounding needs cvt.rs (sm_100a / sm_103a / sm_107a)"
        )
    if mode == "philox" and philox_rounds < 2:
        raise ValueError(f"philox_rounds must be >= 2, got {philox_rounds}")
    sr = {
        "none": "",
        "philox": "philox",
        "lcg": "lcg" if rand_seed is not None else "lcg_clock",
    }[mode]
    sr_rounds = int(philox_rounds) if sr == "philox" else 0
    if rand_seed is not None and sr in ("philox", "lcg"):
        if not (
            isinstance(rand_seed, torch.Tensor)
            and rand_seed.numel() == 1
            and rand_seed.dtype == torch.int64
            and rand_seed.is_cuda
        ):
            raise ValueError("rand_seed must be a single-element int64 CUDA tensor")
        rand_seed = rand_seed.reshape(1)
    else:
        seed_key = ("seed", str(q.device))
        rand_seed = _ROW_ID.get(seed_key)
        if rand_seed is None:
            rand_seed = torch.zeros(
                1, dtype=torch.int64, device=q.device
            )  # unread placeholder
            _ROW_ID[seed_key] = rand_seed
    fn = GdnReplayMtpUmma.compile(
        q.device,
        T,
        H,
        HV,
        u_cache.dtype == torch.bfloat16,
        use_ro,
        hpc,
        sr,
        sr_rounds,
        pmax,
    )
    fn(
        q,
        k,
        v,
        a,
        b,
        _cached_f32(A_log),
        _cached_f32(dt_bias),
        h0,
        initial_state_indices,
        k_cache,
        u_cache,
        g_cache,
        hist_len,
        cache_base,
        row_order,
        output,
        rand_seed,
        float(scale),
        int(flush_min),
    )
    if restart_hist_on_flush:
        flushed = hist_len >= flush_min
        cache_base.copy_((cache_base + hist_len * flushed) & RING_MASK)
        hist_len.masked_fill_(flushed, 0)
    return output
