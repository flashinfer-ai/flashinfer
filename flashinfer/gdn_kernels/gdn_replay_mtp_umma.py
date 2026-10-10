# SPDX-License-Identifier: Apache-2.0
"""GDN ReplaySSM MTP decode step (1 <= T <= 8) on the UMMA / TMEM design (Blackwell).

The SM100 backend of ``gated_delta_rule_mtp_ucache_flush`` (fp16-state arm: bf16 q/k/v/a/b and
outputs, fp16 state pool, bf16 k ring, bf16 or fp16 u ring, fp32 g ring), same 32-slot
ring contract:

* live window ``[cache_base, cache_base + hist_len) mod 32``, P = hist_len <= 16,
  g ring = window-relative cumulative log-decay, ``w_j = e^{G_P - g_j}``;
* every row appends (k̂_s, u_s, g) at ``(base + P + s) & 31``; g restarts at 0 on flush rows;
* rows with ``P >= flush_min`` also write ``S0 <- e^{G_P} S0 + sum_{j<P} w_j u_j k̂_j^T``
  (the T current tokens are not folded);
* padded rows (state index < 0) exit and write nothing; cursor commits are host-side;
* optional stochastic rounding of the flushed fp16 state (``stochastic_rounding``), off by
  default (round-to-nearest). "philox": FlashInfer's Mamba scheme, one Philox4x32 draw (key =
  seed, counter = flat state-pool offset of the 8-element chunk) feeds 4 ``cvt.rs.f16x2``
  pairs. "lcg": one LCG step per pair (s ^ (s >> 16) dither) from a hashed per-row seed
  (``rand_seed`` or %clock).

Math per (request, value head) with ``X = [k̂_0..k̂_{T-1}; s q̂_0..s q̂_{T-1}]``:

    S_h x = e^{G_P} S0 x + sum_j (w_j u_j)(k̂_j . x)                  (x in X)
    u_t   = b_t v_t - b_t e^{g_t} S_h k̂_t - sum_{s<t} b_t e^{g_t-g_s}(k̂_s.k̂_t) u_s
    y_t   = e^{g_t} S_h q̂_t + sum_{s<=t} e^{g_t-g_s}(k̂_s.q̂_t) u_s

CTA = (key head, request): the key head's value heads in sequence (all HV / H of them, 1 to
4, or a slice of 1 / 2 for small batches, ``heads_per_cta``); grid (H x (HV / H) /
heads_per_cta, B), key head fastest; 4 CTAs / SM
(SMEM ~56 KB, 80 registers, 128 TMEM columns); 192 threads in three roles -- 256 with the
register split of the builds that would otherwise spill (``_use_regsplit``), where the
math warps are warps 4-7 and warps 2-3 idle:
  warp 0   : TMA producer — each value head's fp16 state as two 16 KB 128B-swizzled K halves
             through two stages; per-head u windows bulk-copied into a two-head SMEM double
             buffer; flush rows TMA-store the folded halves.
  warp 1   : allocates TMEM at CTA entry (the next CTA on the SM is not launched before the
             allocation permit is relinquished), only UMMA issuer.
  warps 2-5: the math warps ("consumers"): 128 lanes = 128 V rows (TMEM lane = row);
             prologue + per-row math.
UMMAs (kind::f16, fp32 accumulate in TMEM):
  readout  R = S0 . X^T            M128 N16 K128  A: state stage   B: X tile (K-major)
  history  Y = D . C^T             M128 N16 K16   A: D (TMEM)      B: C tile
  fold     F = D . K_hist (halves) M128 N64 K16   A: D (TMEM)      B: K_hist tile, MN-major
  with D[v, j] = w_j u_j[v], C[x, j] = k̂_j . x, K_hist[j] = k̂_j (j < P).
The per-CTA Gram [k̂_ring; X] . X^T (all dots + the norms) runs on mma.sync with fragments
loaded straight from global memory.  The intra-step causal solve is a T-step forward
substitution per V row on CUDA cores.  Flush rows issue fold half 0 before the state
lands, so each half's RMW follows its readout and the epilogue overlaps the next head's
state load.

Specialised for K == V == 128, HV % H == 0, HV // H == 4 (Qwen3.5 GDN geometry).
"""

from __future__ import annotations

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float16, Float32, Int32, Int64, Uint32, Uint64
from cutlass.cute.nvgpu import cpasync

try:
    from . import _umma_helpers as U
    from .device_target import gdn_compile_options, gdn_device_target
except (
    ImportError
):  # tests and benchmarks may load this file by path, outside the package
    from flashinfer.gdn_kernels import _umma_helpers as U
    from flashinfer.gdn_kernels.device_target import (
        gdn_compile_options,
        gdn_device_target,
    )

RING_SLOTS = 32
RING_MASK = RING_SLOTS - 1
W_RING = 16
K_DIM = 128
V_DIM = 128
HPK = 4  # most value heads a CTA runs: per-head SMEM / TMEM / barrier arrays are sized for 4
# The value heads per key head, HV / H, is 1..4 per kernel instance (``self.hpk``); a CTA
# runs ``heads_per_cta`` of them (a divisor of HV / H) with the per-head arrays sized for 4.
_EPS = 1e-6
_TMEM_COLS = 128
_COEF = 192  # fp32 coefficient words per head (see _CO_*)
_CO_A, _CO_BQ, _CO_BETA, _CO_BG, _CO_EG, _CO_GAM, _CO_W, _CO_EGP = (
    0,
    64,
    128,
    136,
    144,
    152,
    160,
    176,
)
_MIN_BLOCKS_PER_SM = (
    4  # 128 of the 512 TMEM columns, ~56 KB SMEM and 80 registers per CTA
)
# Register split: 256 threads, the math warps on warps 4-7 (one warpgroup, setmaxnreg.inc
# 104), warps 0-3 (TMA producer, UMMA issuer, two idle warps) setmaxnreg.dec 24; the
# 64-register launch budget keeps 4 CTAs/SM. It costs the two extra warps and the allocation
# wait (+1-2 % median where nothing spills), so it is used only where the 192-thread,
# 80-register layout spills on the math warps' path (B200, static STL / LDL): with 4 heads per
# CTA, round-to-nearest at T >= 5 (T = 8: 20 / 26, +8 % kernel time at B = 512, 0 % fold),
# LCG at T = 8 (16 / 16) and Philox-10 (T = 4: 3 / 9, T = 8: 27 / 43); with 2 heads per CTA
# the Philox-5 and LCG builds at T = 8 (1 / 1 each). The 4-head Philox-5 build and every other
# 1- and 2-head build fit without spilling. With 3 heads per CTA (HV / H = 3) round-to-nearest
# spills at T = 5 / 6 (16 / 22, 19 / 27; the split measured 1-8 % faster at B = 64..512) and
# only lightly at T = 7 / 8 (5 / 7, 3 / 5; the split measured 0-2.5 % slower at T = 8), so
# the 3-head builds split at T = 5 and 6 only; their stochastic-rounding cells were not
# swept (T = 8: Philox-5 7 / 13, Philox-10 3 / 5, LCG 1 / 1 in the 192-thread layout). The
# 192-thread layout cannot split: setmaxnreg is warpgroup-wide and a 2-warp warpgroup
# deadlocks.
_REGSPLIT = (24, 104)


def _use_regsplit(T: int, hpc: int, sr: str, sr_rounds: int) -> bool:
    if hpc == 2:
        return T == 8 and sr != ""
    if hpc == 3:
        return sr == "" and T in (5, 6)
    if hpc != HPK:
        return False
    if sr == "philox" and sr_rounds <= 5:
        return False
    return T >= 5 or sr_rounds > 5


def _sel(cond, a, b, ty):
    """Branch-free typed select (arith.select) for use outside AST-preprocessed scopes."""
    return ty(cutlass.select_(cond, ty(a), ty(b)))


class GdnReplayMtpUmma:
    def __init__(
        self,
        T: int,
        h: int,
        hv: int,
        u_bf16: bool = True,
        use_row_order: bool = False,
        hpc: Optional[int] = None,
        sr: str = "",
        sr_rounds: int = 0,
    ):
        assert 1 <= T <= 8
        assert hv % h == 0 and 1 <= hv // h <= HPK, (h, hv)
        self.hpk = hv // h  # value heads per key head
        # hpc = value heads per CTA (hpk: one CTA per key head; a smaller divisor of hpk splits
        # a key head over hpk / hpc CTAs, which repeat the per-key-head prologue but shorten
        # the serial head chain)
        hpc = self.hpk if hpc is None else int(hpc)
        assert 1 <= hpc <= self.hpk and self.hpk % hpc == 0, (hpc, self.hpk)
        self.hpc = hpc
        self.nsub = self.hpk // hpc
        self.T = T
        self.h = h
        self.hv = hv
        # u ring dtype (bf16 / fp16): read and appended through dtype-aware conversions.
        # The k ring is bf16 by construction (the Gram runs bf16 mma.sync on it).
        self.u_bf16 = u_bf16
        self.use_row_order = use_row_order
        # stochastic rounding of the flush store: "" (round-to-nearest), "philox" (sr_rounds
        # rounds), "lcg" (seeded by rand_seed) or "lcg_clock" (seeded by %clock)
        assert sr in ("", "philox", "lcg", "lcg_clock")
        self.sr = sr
        self.sr_rounds = sr_rounds if sr == "philox" else 0
        self.sr_lcg = sr in ("lcg", "lcg_clock")
        self.regsplit = (
            _REGSPLIT if _use_regsplit(T, hpc, self.sr, self.sr_rounds) else ()
        )
        self.cw0 = 4 if self.regsplit else 2  # first consumer warp
        self.num_threads = 256 if self.regsplit else 192

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
        self.kernel(
            q,
            k,
            v,
            a,
            b,
            a_log,
            dt_bias,
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
            grid=(self.h * self.nsub, q.shape[0], 1),
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
        T = self.T
        T2 = 2 * T

        tid, _, _ = cute.arch.thread_idx()
        bx, blk_row, _ = cute.arch.block_idx()
        HPC = self.hpc
        if cutlass.const_expr(self.nsub == 1):
            kh = bx  # key head
            sub = Int32(0)
        else:
            kh = bx // self.nsub  # key head
            sub = bx % self.nsub  # which hpc-head slice of it
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        if cutlass.const_expr(self.use_row_order):
            brow = Int32(row_order[blk_row])
        else:
            brow = blk_row
        hv0 = kh * self.hpk + sub * HPC

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
        # K_hist tile: window keys k̂_j (j < P, else 0) row-major [16 slots x 128] fp16 in the
        # X-tile SW128 layout; the fold reads it as an MN-major B operand (N = k index)
        s_kt = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(16 * 128), byte_alignment=1024
        )
        s_c = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(16 * 16), byte_alignment=256
        )
        s_gram = smem.allocate_array(Float32, 32 * 16)
        s_coef = smem.allocate_array(Float32, HPK * _COEF)
        s_rn = smem.allocate_array(Float32, 16)
        # u windows of two heads: [buf][logical slot j][V] (ring dtype), bulk-copied by warp 0
        s_uwin = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(2 * W_RING * V_DIM), byte_alignment=128
        )
        u_full = [smem.allocate_array(cutlass.Int64, 1) for _ in range(HPK)]
        u_free = [smem.allocate_array(cutlass.Int64, 1) for _ in range(2)]
        st_full = [smem.allocate_array(cutlass.Int64, 1) for _ in range(2 * HPK)]
        mma_slot = [smem.allocate_array(cutlass.Int64, 1) for _ in range(2 * HPK)]
        mma_done = [smem.allocate_array(cutlass.Int64, 1) for _ in range(HPK)]
        d_ready = [smem.allocate_array(cutlass.Int64, 1) for _ in range(HPK)]
        epi_done = [smem.allocate_array(cutlass.Int64, 1) for _ in range(HPK)]
        f_mma_a = [smem.allocate_array(cutlass.Int64, 1) for _ in range(HPK)]
        f_mma_b = [smem.allocate_array(cutlass.Int64, 1) for _ in range(HPK)]
        f_epi_a = [smem.allocate_array(cutlass.Int64, 1) for _ in range(HPK)]
        rd0 = [smem.allocate_array(cutlass.Int64, 1) for _ in range(HPK)]
        f_gdone = [smem.allocate_array(cutlass.Int64, 1) for _ in range(2 * HPK)]
        x_ready = smem.allocate_array(cutlass.Int64, 1)
        tmem_done = smem.allocate_array(cutlass.Int64, 1)
        tmem_addr = smem.allocate(Int32, 4)

        # TMEM first: the next CTA on this SM is not launched before this one relinquishes
        # its TMEM allocation permit (measured: allocating after the prologue spaced the
        # CTAs of one SM ~2.5 us apart and left a quarter of the CTA slots empty)
        if warp == 1:
            U.tmem_alloc(tmem_addr, _TMEM_COLS)
            cute.arch.relinquish_tmem_alloc_permit()

        # Cursors first (one global round trip, issued by every thread), then every mbarrier
        # while that round trip is in flight. Warp 0 publishes the inits through one named
        # barrier per role (3: warp 1; 4: the X-tile warps; 5: the ring-row warps), each
        # synced at that role's first mbarrier use -- no CTA-wide sync, which a warp only
        # reaches once its outstanding loads have landed.
        sidx = Int32(state_idx_t[brow])
        P = Int32(hist_len[brow])
        base = Int32(cache_base[brow])
        if warp == 0:
            with cute.arch.elect_one():
                for s_ in cutlass.range_constexpr(2 * HPK):
                    cute.arch.mbarrier_init(st_full[s_], 1)
                    cute.arch.mbarrier_init(mma_slot[s_], 1)
                    cute.arch.mbarrier_init(f_gdone[s_], 128)
                for i in cutlass.range_constexpr(HPK):
                    cute.arch.mbarrier_init(u_full[i], 1)
                    cute.arch.mbarrier_init(mma_done[i], 1)
                    cute.arch.mbarrier_init(d_ready[i], 128)
                    cute.arch.mbarrier_init(epi_done[i], 128)
                    cute.arch.mbarrier_init(f_mma_a[i], 1)
                    cute.arch.mbarrier_init(f_mma_b[i], 1)
                    cute.arch.mbarrier_init(f_epi_a[i], 128)
                    cute.arch.mbarrier_init(rd0[i], 1)
                cute.arch.mbarrier_init(u_free[0], 128)
                cute.arch.mbarrier_init(u_free[1], 128)
                cute.arch.mbarrier_init(x_ready, 64)
                cute.arch.mbarrier_init(tmem_done, 128)
                cute.arch.mbarrier_init_fence()
            cute.arch.fence_acq_rel_cta()
            cute.arch.barrier_arrive(barrier_id=3, number_of_threads=64)
            cute.arch.barrier_arrive(barrier_id=4, number_of_threads=96)
            cute.arch.barrier_arrive(barrier_id=5, number_of_threads=96)

        # Inputs that depend on the batch row only are issued at entry by EVERY thread (warps
        # 0/1 compute in-range dummy indices and drop the values), ahead of the padded-row
        # exit, so their latency overlaps the cursors'.
        T_ = self.T
        grp_e = lane >> 2
        c4_e = lane & 3
        CW0 = self.cw0
        cw_e = (warp - CW0) & 3
        nt_e = cw_e & 1
        q_g = q.iterator.toint()
        k_g = k.iterator.toint()
        xr_b = nt_e * 8 + grp_e
        is_cons = Int32(warp >= CW0)
        is_xt = Int32(
            warp >= CW0 + 2
        )  # X-tile warps (mt == 1); the ring-row warps never read wo
        x_own = self._x_row_addr(k, q, k_g, q_g, brow, kh, xr_b)
        x_oth = self._x_row_addr(k, q, k_g, q_g, brow, kh, (1 - nt_e) * 8 + grp_e)
        wb = []
        wo = []
        for p in cutlass.range_constexpr(4):
            koff = Int64((32 * p + 8 * c4_e) * 2)
            wb.append(U.ldg_v4_if(is_cons, x_own + koff))
            wo.append(U.ldg_v4_if(is_xt, x_oth + koff))
        ct_e = (tid - 32 * CW0) & 127
        gh = (ct_e >> 3) & 3
        ghc = gh % HPC  # in-range head for loads
        gt = ct_e & 7
        tt = _sel(gt < T_, gt, 0, Int32)
        ab_off = Int64(cute.crd2idx((brow, tt, hv0 + ghc), a.layout)) * 2
        av_raw = U.ldg_u16(a.iterator.toint() + ab_off)
        bv_raw = U.ldg_u16(b.iterator.toint() + ab_off)

        if warp == 1:
            if sidx < 0:
                # padded row: hand the TMEM columns back before the CTA exits
                cute.arch.sync_warp()
                U.tmem_dealloc(tmem_addr[0], _TMEM_COLS)
        U.exit_if_negative(sidx)
        is_flush = flush_min <= P
        # Cursors known: the Gram's ring rows (ring-row warps; row grp goes into the registers
        # that hold the other X half in the X-tile warps) and the gate values, so their round
        # trips overlap the entry loads and warp 0's window / state issue.
        is_rr = Int32(warp == CW0) + Int32(warp == CW0 + 1)  # ring-row warps (mt == 0)
        kc_g = k_cache.iterator.toint()
        ra0_e = (
            kc_g
            + Int64(
                cute.crd2idx((sidx, kh, (base + grp_e) & RING_MASK, 0), k_cache.layout)
            )
            * 2
        )
        ra1_e = (
            kc_g
            + Int64(
                cute.crd2idx(
                    (sidx, kh, (base + grp_e + 8) & RING_MASK, 0), k_cache.layout
                )
            )
            * 2
        )
        wr1_e = []
        for p in cutlass.range_constexpr(4):
            koff = Int64((32 * p + 8 * c4_e) * 2)
            wo[p] = U.ldg_v4_if_keep(
                is_rr, ra0_e + koff, wo[p][0], wo[p][1], wo[p][2], wo[p][3]
            )
            wr1_e.append(U.ldg_v4_if(is_rr, ra1_e + koff))
        # window g values (gate threads: G_P of their head; weight threads: g_j)
        wh = ((ct_e - 32) >> 4) & 3
        whc = wh % HPC
        wj = (ct_e - 32) & 15
        gp_g = Float32(g_cache[sidx, hv0 + ghc, (base + P - 1) & RING_MASK])
        gp_w = Float32(g_cache[sidx, hv0 + whc, (base + P - 1) & RING_MASK])
        gj_w = Float32(g_cache[sidx, hv0 + whc, (base + wj) & RING_MASK])

        st_addr = s_state.iterator.toint()
        x_addr = s_x.iterator.toint()
        kt_addr = s_kt.iterator.toint()
        c_addr = s_c.iterator.toint()
        uw_addr = s_uwin.iterator.toint()
        u_g = u_cache.iterator.toint()

        if warp == 0:
            cpasync.prefetch_descriptor(tma_ld.atom)
            # the two u windows (4 KB) ahead of the two state halves (32 KB): head 0's D
            # build needs its window long before the readout needs the state
            for hh in cutlass.range_constexpr(min(2, HPC)):
                with cute.arch.elect_one():
                    self._u_window_copy(
                        u_cache, u_g, uw_addr, u_full[hh], sidx, hv0 + hh, base, hh
                    )
            for g in cutlass.range_constexpr(2):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(st_full[g], 16384)
                U.simple_tma_copy(
                    tma_ld.atom,
                    tma_ld.tma_tensor[sidx, hv0, None, (None, g)],
                    s_state[None, (None, g)],
                    st_full[g],
                )

        # Warpgroup-level split first: with the register split each setmaxnreg must
        # dominate its warpgroup's code (ptxas uses the minimum after a merge). This shape
        # (and row from warp & 3) also lets ptxas fit Philox-5 at 4 heads per CTA in 80
        # registers, where a flat if / elif / else over the three roles spilled 6 values
        # whose reloads (16 % L1 misses: 233 KB of the SM's 256 KB is shared memory) sat on
        # the flush critical path (T = 8, B = 512, every row flushing: 451 vs 495 us).
        # Allocation-sensitive: re-check STL / LDL over the (T, heads per CTA, rounding)
        # cells (CUTE_DSL_KEEP=sass) after edits to the math warps' path.
        if warp < CW0:
            if cutlass.const_expr(self.regsplit):
                cute.arch.setmaxregister_decrease(self.regsplit[0])
            if warp == 0:
                # ============================================================ producer
                if is_flush:
                    cpasync.prefetch_descriptor(tma_st.atom)
                    for slot in cutlass.range_constexpr(2, 2 * HPC):
                        prev = slot - 2
                        cute.arch.mbarrier_wait(f_gdone[prev], 0)
                        U.simple_tma_copy(
                            tma_st.atom,
                            s_state[None, (None, prev % 2)],
                            tma_st.tma_tensor[
                                sidx, hv0 + prev // 2, None, (None, prev % 2)
                            ],
                        )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                st_full[slot], 16384
                            )
                        U.simple_tma_copy(
                            tma_ld.atom,
                            tma_ld.tma_tensor[
                                sidx, hv0 + slot // 2, None, (None, slot % 2)
                            ],
                            s_state[None, (None, slot % 2)],
                            st_full[slot],
                        )
                        if cutlass.const_expr(slot % 2 == 0 and slot // 2 >= 2):
                            hh = slot // 2
                            cute.arch.mbarrier_wait(u_free[hh - 2], 0)
                            with cute.arch.elect_one():
                                self._u_window_copy(
                                    u_cache,
                                    u_g,
                                    uw_addr,
                                    u_full[hh],
                                    sidx,
                                    hv0 + hh,
                                    base,
                                    hh % 2,
                                )
                    for prev in cutlass.range_constexpr(2 * HPC - 2, 2 * HPC):
                        cute.arch.mbarrier_wait(f_gdone[prev], 0)
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
                    for slot in cutlass.range_constexpr(2, 2 * HPC):
                        cute.arch.mbarrier_wait(mma_slot[slot - 2], 0)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                st_full[slot], 16384
                            )
                        U.simple_tma_copy(
                            tma_ld.atom,
                            tma_ld.tma_tensor[
                                sidx, hv0 + slot // 2, None, (None, slot % 2)
                            ],
                            s_state[None, (None, slot % 2)],
                            st_full[slot],
                        )
                        if cutlass.const_expr(slot % 2 == 0 and slot // 2 >= 2):
                            hh = slot // 2
                            cute.arch.mbarrier_wait(u_free[hh - 2], 0)
                            with cute.arch.elect_one():
                                self._u_window_copy(
                                    u_cache,
                                    u_g,
                                    uw_addr,
                                    u_full[hh],
                                    sidx,
                                    hv0 + hh,
                                    base,
                                    hh % 2,
                                )
            elif warp == 1:
                # ============================================================ UMMA issuer
                tbase = tmem_addr[0]
                sd = U.sdesc_sw128(V_DIM * 128)
                x_base = sd | Uint64(x_addr >> 4)
                stage_base = [
                    sd | Uint64(st_addr >> 4),
                    sd | Uint64((st_addr + 16384) >> 4),
                ]
                c_desc = U.sdesc_sw32(Uint64(c_addr >> 4))
                # MN-major SW128: SBO = 1024 B between 8-slot K atoms, N halves 2048 B apart
                kt_desc = U.sdesc_mn_sw128(2048, 1024) | Uint64(kt_addr >> 4)
                idesc16 = U.f16_idesc(128, 16)
                idesc64mn = U.f16_idesc(128, 64, b_mn_major=True)
                cute.arch.barrier(barrier_id=3, number_of_threads=64)  # warp 0's inits
                cute.arch.mbarrier_wait(x_ready, 0)
                U.fence_after_sync()
                if is_flush:
                    # per head: fold half 0 (needs only D, Kt) -> readout half 0 (rd0: the
                    # consumers' RMW of stage 0 may start) -> readout half 1 + history
                    # (mma_done) -> fold half 1 once the RMW has read fold half 0 (f_epi_a).
                    # R/Y (cols 0..31) and F (64..127) of head h + 1 wait for head h's epilogue.
                    for hh in cutlass.range_constexpr(HPC):
                        cute.arch.mbarrier_wait(d_ready[hh], 0)
                        if cutlass.const_expr(hh > 0):
                            cute.arch.mbarrier_wait(epi_done[hh - 1], 0)
                        U.fence_after_sync()
                        U.umma_ts(tbase + 64, tbase + 32, kt_desc, idesc64mn, False)
                        U.umma_commit(f_mma_a[hh])
                        for g in cutlass.range_constexpr(2):
                            slot = 2 * hh + g
                            cute.arch.mbarrier_wait(st_full[slot], 0)
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
                                U.umma_commit(rd0[hh])
                        U.umma_ts(tbase + 16, tbase + 32, c_desc, idesc16, False)
                        U.umma_commit(mma_done[hh])
                        cute.arch.mbarrier_wait(f_epi_a[hh], 0)
                        U.fence_after_sync()
                        U.umma_ts(
                            tbase + 64,
                            tbase + 32,
                            kt_desc + Uint64(2048 >> 4),
                            idesc64mn,
                            False,
                        )
                        U.umma_commit(f_mma_b[hh])
                else:
                    for hh in cutlass.range_constexpr(HPC):
                        rcol = (hh % 2) * 32
                        if cutlass.const_expr(hh >= 2):
                            cute.arch.mbarrier_wait(epi_done[hh - 2], 0)
                            U.fence_after_sync()
                        for g in cutlass.range_constexpr(2):
                            slot = 2 * hh + g
                            cute.arch.mbarrier_wait(st_full[slot], 0)
                            U.fence_after_sync()
                            for ka in cutlass.range_constexpr(4):
                                U.umma_ss(
                                    tbase + rcol,
                                    stage_base[g] | Uint64((ka * 32) >> 4),
                                    x_base | Uint64((g * 2048 + ka * 32) >> 4),
                                    idesc16,
                                    (g > 0) or (ka > 0),
                                )
                            U.umma_commit(mma_slot[slot])
                        cute.arch.mbarrier_wait(d_ready[hh], 0)
                        U.fence_after_sync()
                        U.umma_ts(
                            tbase + rcol + 16,
                            tbase + 64 + (hh % 2) * 8,
                            c_desc,
                            idesc16,
                            False,
                        )
                        U.umma_commit(mma_done[hh])
                cute.arch.mbarrier_wait(tmem_done, 0)
                U.fence_after_sync()
                U.tmem_dealloc(tbase, _TMEM_COLS)
        else:
            if cutlass.const_expr(self.regsplit):
                cute.arch.setmaxregister_increase(self.regsplit[1])
            # ============================================================ consumers
            cw = warp - CW0
            ct = cw * 32 + lane  # TMEM lane address / k index for staging
            row = ((warp & 3) << 5) | lane  # V row held by this thread (= TMEM lane)
            grp = lane >> 2
            c4 = lane & 3

            # -------- Gram on mma.sync: rows [k̂_ring(16); X(16)] x cols X(16), K = 128.
            # Fragments come straight from global memory: per 32-wide k block a lane loads 16 B
            # (8 contiguous k) of each of its rows; the k order inside the m16n8k16 fragment is
            # permuted identically for A and B, which a dot product does not see.
            # Warps mt == 1 own the X x X tiles (norms, intra-step dots): they depend on the
            # batch row only, so the X tile and x_ready follow ONE global round trip.  Warps
            # mt == 0 own the ring x X tiles from the rows issued once the cursors landed.
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
                s_gram[(16 + grp) * 16 + gc] = accx[0]
                s_gram[(16 + grp) * 16 + gc + 1] = accx[1]
                s_gram[(24 + grp) * 16 + gc] = accx[2]
                s_gram[(24 + grp) * 16 + gc + 1] = accx[3]
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                # X tile (fp16, SW128) and x_ready from these warps' own rows: they hold the
                # same rows as the ring-row warps and have nothing in flight, so the readout
                # can start once the X x X Gram is done instead of after the ring rows
                rn_x = cute.math.rsqrt(s_gram[(16 + xr_b) * 16 + xr_b] + Float32(_EPS))
                rn_x = _sel(xr_b >= T, rn_x * scale, rn_x, Float32)
                for p in cutlass.range_constexpr(4):
                    kc = 4 * p + c4
                    xw = []
                    for i in cutlass.range_constexpr(4):
                        xw.append(
                            U.pack_f16x2(
                                U.bf16x2_lo_f32(wb[p][i]) * rn_x,
                                U.bf16x2_hi_f32(wb[p][i]) * rn_x,
                            )
                        )
                    xb = (kc // 8) * 2048 + xr_b * 128 + (((kc % 8) ^ (xr_b % 8)) * 16)
                    U.sts_v4(x_addr + xb, xw[0], xw[1], xw[2], xw[3])
                cute.arch.barrier(barrier_id=4, number_of_threads=96)  # warp 0's inits
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(x_ready)
                # Everything below needs only the X x X Gram and the gate inputs (one round
                # trip), so it overlaps warps mt == 0 waiting for the ring rows.
                cl = ct - 64
                if cl < T2:
                    rn_own = cute.math.rsqrt(
                        s_gram[(16 + cl) * 16 + cl] + Float32(_EPS)
                    )
                    if cl >= T:
                        rn_own = rn_own * scale
                    s_rn[cl] = rn_own
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
                        kk = s_gram[(16 + s) * 16 + t] * s_rn[s] * s_rn[t]
                        qk = s_gram[(16 + s) * 16 + T + t] * s_rn[s] * s_rn[T + t]
                        av_ = Float32(0.0)
                        if s < t:
                            av_ = s_coef[cb + _CO_BETA + t] * dec * kk
                        bq_ = Float32(0.0)
                        if s <= t:
                            bq_ = dec * qk
                        s_coef[cb + _CO_A + t * 8 + s] = av_
                        s_coef[cb + _CO_BQ + t * 8 + s] = bq_
            else:
                wr0 = wo  # ring rows grp / grp + 8, issued when the cursors landed
                wr1 = wr1_e
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                # norm of this lane's X row (for the k̂ ring append below)
                rn_b = cute.math.rsqrt(s_gram[(16 + xr_b) * 16 + xr_b] + Float32(_EPS))
                rn_b = _sel(xr_b >= T, rn_b * scale, rn_b, Float32)
                # ring x X tiles
                accr = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                for p in cutlass.range_constexpr(4):
                    for hs in cutlass.range_constexpr(2):
                        accr = U.mma_bf16_16816(
                            wr0[p][2 * hs],
                            wr1[p][2 * hs],
                            wr0[p][2 * hs + 1],
                            wr1[p][2 * hs + 1],
                            wb[p][2 * hs],
                            wb[p][2 * hs + 1],
                            accr,
                        )
                s_gram[grp * 16 + gc] = accr[0]
                s_gram[grp * 16 + gc + 1] = accr[1]
                s_gram[(grp + 8) * 16 + gc] = accr[2]
                s_gram[(grp + 8) * 16 + gc + 1] = accr[3]
                # warp 0's inits, before this role's first mbarrier use (the coefficients)
                cute.arch.barrier(barrier_id=5, number_of_threads=96)
                # k̂ ring appends (bf16, 16 B per lane)
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
            if mt == 1:
                cl = ct - 64
                for i in cutlass.range_constexpr(4):
                    e = cl * 4 + i
                    xc = e >> 4
                    j = e & 15
                    cval = Float32(0.0)
                    if j < P:
                        if xc < T2:
                            rnx = cute.math.rsqrt(
                                s_gram[(16 + xc) * 16 + xc] + Float32(_EPS)
                            )
                            if xc >= T:
                                rnx = rnx * scale
                            cval = s_gram[j * 16 + xc] * rnx
                    cbyte = xc * 32 + (((j // 8) ^ ((xc >> 2) & 1)) * 16) + (j % 8) * 2
                    U.sts_u16(c_addr + cbyte, U.f32_to_u16(cval, False))
            if is_flush:
                # K_hist tile (fold B operand): thread -> (slot row jr = ct / 8, 16 B chunk c8) of both
                # K groups; ring rows were just read by the Gram (L1/L2 hits); slots j >= P zeroed
                jr = ct >> 3
                c8 = ct & 7
                krow = (
                    kc_g
                    + Int64(
                        cute.crd2idx(
                            (sidx, kh, (base + jr) & RING_MASK, c8 * 8), k_cache.layout
                        )
                    )
                    * 2
                )
                kr = [U.ldg_v4(krow), U.ldg_v4(krow + Int64(128))]
                for gk in cutlass.range_constexpr(2):
                    kw4 = []
                    for i in cutlass.range_constexpr(4):
                        lo = _sel(jr < P, U.bf16x2_lo_f32(kr[gk][i]), 0.0, Float32)
                        hi = _sel(jr < P, U.bf16x2_hi_f32(kr[gk][i]), 0.0, Float32)
                        kw4.append(U.pack_f16x2(lo, hi))
                    U.sts_v4(
                        kt_addr + gk * 2048 + jr * 128 + ((c8 ^ (jr % 8)) * 16),
                        kw4[0],
                        kw4[1],
                        kw4[2],
                        kw4[3],
                    )
            # C / Kt are read by the history / fold UMMAs, which warp 1 issues only after
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
                    w = Float32(0.0)
                    if wj < P:
                        w = cute.math.exp(gp_w - gj_w, fastmath=True)
                    s_coef[wh * _COEF + _CO_W + wj] = w
                    if wj == 0:
                        egp = Float32(1.0)
                        if P > 0:
                            egp = cute.math.exp(gp_w, fastmath=True)
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
                u_full=u_full,
                u_free=u_free,
                f_mma_a=f_mma_a,
                f_mma_b=f_mma_b,
                f_epi_a=f_epi_a,
                f_gdone=f_gdone,
                rd0=rd0,
            )
            if is_flush:
                self._load_seed(ctx, rand_seed)
                for hh in cutlass.range_constexpr(HPC):
                    self._head(hh, True, ctx)
            else:
                for hh in cutlass.range_constexpr(HPC):
                    self._head(hh, False, ctx)
            U.fence_before_sync()
            cute.arch.mbarrier_arrive(tmem_done)

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

    def _u_window_copy(self, u_cache, u_g, uw_addr, mbar, sidx, hv, base, buf):
        """Bulk-copy logical window slots j = 0..15 (physical (base + j) & 31) of one value
        head's u ring into SMEM buffer ``buf`` as [j][V]. Call under elect_one.

        The window's 16 ring slots wrap at most once: two contiguous copies instead of
        sixteen single-row ones (saves ~0.5 us of warp 0's serial issue per head pair).
        Without a wrap the second copy re-copies row 15 (the same bytes), so the transaction
        count stays expression-only."""
        rowb = V_DIM * 2
        src_head = u_g + Int64(cute.crd2idx((sidx, hv, 0, 0), u_cache.layout)) * 2
        dst = uw_addr + buf * (W_RING * rowb)
        n1 = _sel(
            base + W_RING <= RING_SLOTS, Int32(W_RING), Int32(RING_SLOTS) - base, Int32
        )
        n2 = _sel(n1 < W_RING, Int32(W_RING) - n1, Int32(1), Int32)
        d2 = _sel(n1 < W_RING, n1, Int32(W_RING - 1), Int32)
        s2 = _sel(n1 < W_RING, Int32(0), (base + Int32(W_RING - 1)) & RING_MASK, Int32)
        cute.arch.mbarrier_arrive_and_expect_tx(mbar, (n1 + n2) * rowb)
        U.bulk_g2s(dst, src_head + Int64(base * rowb), n1 * rowb, mbar.toint())
        U.bulk_g2s(
            dst + d2 * rowb, src_head + Int64(s2 * rowb), n2 * rowb, mbar.toint()
        )

    def _d_build(self, hh, dcol, ctx):
        """D row w_j u_j (j < P, fp16 pairs) of head hh into TMEM columns dcol..dcol+7;
        arrives d_ready[hh].  Also returns this row's raw v values of head hh (issued first
        so their latency overlaps the D build and the readout wait)."""
        T = self.T
        P, row, ct, tbase, s_coef = (
            ctx["P"],
            ctx["row"],
            ctx["ct"],
            ctx["tbase"],
            ctx["s_coef"],
        )
        v = ctx["v"]
        cb = hh * _COEF
        vv = []
        for t in range(T):
            vv.append(v[ctx["brow"], t, ctx["hv0"] + hh, row])
        cute.arch.mbarrier_wait(ctx["u_full"][hh], 0)
        ub = ctx["uw_addr"] + (hh % 2) * (W_RING * V_DIM * 2) + row * 2
        uv = [U.lds_u16(ub + j * (V_DIM * 2)) for j in range(W_RING)]
        if hh < 2:
            cute.arch.mbarrier_arrive(ctx["u_free"][hh])
        words = []
        for c in range(8):
            lo = _sel(
                2 * c < P,
                U.u16_as_f32(uv[2 * c], self.u_bf16) * s_coef[cb + _CO_W + 2 * c],
                0.0,
                Float32,
            )
            hi = _sel(
                2 * c + 1 < P,
                U.u16_as_f32(uv[2 * c + 1], self.u_bf16)
                * s_coef[cb + _CO_W + 2 * c + 1],
                0.0,
                Float32,
            )
            words.append(U.pack_f16x2(lo, hi))
        U.tmem_st_u32(ct, tbase + dcol, words)
        U.tmem_wait_st()
        U.fence_before_sync()
        cute.arch.mbarrier_arrive(ctx["d_ready"][hh])
        return vv

    def _epilogue(self, hh, rcol, ctx, vv, early_release):
        """Outputs and u appends of head hh from the readout R (cols rcol..) and history Y
        (cols rcol + 16..) TMEM columns; arrives epi_done[hh] once its TMEM reads are done."""
        T = self.T
        ring_t = BFloat16 if self.u_bf16 else Float16  # u ring element type
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
        if early_release:
            # verify rows: all four column groups first, then release the head buffer
            # (the readout two heads ahead waits on it)
            hks = [egp * rk[t] + yk[t] for t in range(T)]
            rq = U.tmem_ld(ct, tbase + rcol + T, nld)
            yq = U.tmem_ld(ct, tbase + rcol + 16 + T, nld)
            U.tmem_wait_ld()
            U.fence_before_sync()
            cute.arch.mbarrier_arrive(ctx["epi_done"][hh])
        uu = []
        for t in range(T):
            if early_release:
                hk = hks[t]
            else:
                hk = egp * rk[t] + yk[t]
            acc = (
                s_coef[cb + _CO_BETA + t] * Float32(vv[t])
                - s_coef[cb + _CO_BG + t] * hk
            )
            for s_ in range(t):
                acc = acc - s_coef[cb + _CO_A + t * 8 + s_] * uu[s_]
            uu.append(acc)
        if not early_release:
            # flush rows (register-bound): the u solve runs before the q columns are read,
            # so the k-side (hk, v) and q-side (rq, yq) values are never live together
            rq = U.tmem_ld(ct, tbase + rcol + T, nld)
            yq = U.tmem_ld(ct, tbase + rcol + 16 + T, nld)
            U.tmem_wait_ld()
            U.fence_before_sync()
            cute.arch.mbarrier_arrive(ctx["epi_done"][hh])

        # u appends at (base + P + t) & 31 wrap at most once: two base rows (s0, s0 - 32)
        # plus immediate offsets address all T appends
        s0 = (base + P) & RING_MASK
        u_a = (
            u_cache.iterator.toint()
            + Int64(cute.crd2idx((sidx, hv, s0, row), u_cache.layout)) * 2
        )
        u_b = u_a - Int64(RING_SLOTS * V_DIM * 2)
        for t in range(T):
            hq = egp * rq[t] + yq[t]
            y = s_coef[cb + _CO_EG + t] * hq
            for s_ in range(t + 1):
                y = y + s_coef[cb + _CO_BQ + t * 8 + s_] * uu[s_]
            out[brow, t, hv, row] = BFloat16(y)
            if early_release:
                u_cache[sidx, hv, (base + P + t) & RING_MASK, row] = ring_t(uu[t])
            else:
                ua = _sel(s0 + t < RING_SLOTS, u_a, u_b, Int64)
                U.stg_u16(ua + Int64(t * V_DIM * 2), U.f32_to_u16(uu[t], self.u_bf16))

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

    def _rmw(self, hh, g, ctx):
        """Flush rows: S_half = e^{G_P} S0_half + F (TMEM cols 64..127) in place on state stage
        g (= half g of head hh), then publish it to warp 0's TMA store (f_gdone)."""
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
            # LCG dither (3 instructions per pair): this
            # (state row, half)'s seed = murmur3 of its flat id and the per-flush key; pair n
            # (= 4 chunk + i) takes LCG state n folded as s ^ (s >> 16), run as 4 jump-ahead
            # chains (x_i -> A^4 x_i + C4) so the 4 pairs of a 16-B word are independent
            # instructions. Known weakness: the two 13-bit fields of a
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
            if g == 0 and grp == 64 // ldn - 1:
                U.fence_before_sync()
                cute.arch.mbarrier_arrive(
                    ctx["f_epi_a"][hh]
                )  # F may be overwritten (fold half 1)
            for cc in range(nch):
                chunk = grp * nch + cc
                sa = srow + ((chunk ^ (row % 8)) * 16)
                ow = U.lds_v4(sa)
                if self.sr_lcg:
                    rw = [x ^ (x >> Uint32(16)) for x in lx]  # 16-bit fold
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

    def _head(self, hh, flush, ctx):
        if flush:
            # Flush rows: the fold F = D . Kt does not need the state, so warp 1 issues fold
            # half 0 before the state lands; each half's RMW runs as soon as the readout has
            # consumed that stage, so the stage is stored and reloaded early and this head's
            # epilogue overlaps the next head's state load.
            vv = self._d_build(hh, 32, ctx)
            cute.arch.mbarrier_wait(ctx["f_mma_a"][hh], 0)
            cute.arch.mbarrier_wait(ctx["rd0"][hh], 0)
            U.fence_after_sync()
            self._rmw(hh, 0, ctx)
            cute.arch.mbarrier_wait(ctx["f_mma_b"][hh], 0)
            cute.arch.mbarrier_wait(ctx["mma_done"][hh], 0)
            U.fence_after_sync()
            self._rmw(hh, 1, ctx)
            self._epilogue(hh, 0, ctx, vv, False)
        else:
            vv = self._d_build(hh, 64 + (hh % 2) * 8, ctx)
            cute.arch.mbarrier_wait(ctx["mma_done"][hh], 0)
            U.fence_after_sync()
            self._epilogue(hh, (hh % 2) * 32, ctx, vv, True)

    # ------------------------------------------------------------------ compile
    @staticmethod
    def compile(
        device: torch.device,
        T: int,
        h: int,
        hv: int,
        u_bf16: bool,
        use_row_order: bool,
        hpc: Optional[int] = None,
        sr: str = "",
        sr_rounds: int = 0,
    ):
        """Compile one specialisation for ``device``'s architecture (callers use ``_get_kernel``)."""
        batch = cute.sym_int()
        pool = cute.sym_int()
        ft = U.make_fake_tensor_lead_dyn
        uring = BFloat16 if u_bf16 else Float16
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
            GdnReplayMtpUmma(T, h, hv, u_bf16, use_row_order, hpc, sr, sr_rounds),
            *args,
            Float32(1.0),
            Int32(1),
            stream,
        )


_KERNELS: dict = {}


def _get_kernel(
    device: torch.device,
    T: int,
    h: int,
    hv: int,
    u_bf16: bool,
    use_row_order: bool,
    hpc: int,
    sr: str,
    sr_rounds: int,
):
    """The compiled kernel for this specialisation on ``device``'s architecture: one per
    (compile target, parameters); batch and pool sizes are dynamic."""
    key = (
        gdn_device_target(device).compile_key,
        T,
        h,
        hv,
        u_bf16,
        use_row_order,
        hpc,
        sr,
        sr_rounds,
    )
    fn = _KERNELS.get(key)
    if fn is None:
        fn = GdnReplayMtpUmma.compile(
            device, T, h, hv, u_bf16, use_row_order, hpc, sr, sr_rounds
        )
        _KERNELS[key] = fn
    return fn


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
_CTA_SLOTS_PER_SM = _MIN_BLOCKS_PER_SM
_HPC2_MAX_WAVES = 8  # hpc = 2 up to this many waves (see auto_heads_per_cta)


def auto_heads_per_cta(
    batch: int, n_key_heads: int, num_sms: int, heads_per_key: int = HPK
) -> int:
    """Heads per CTA for a batch: 1 while the hpc = 1 grid fits in one wave of CTA slots,
    2 (when it divides ``heads_per_key``) while the hpc = 2 grid fits in 8 waves, else all
    ``heads_per_key`` value heads of a key head in one CTA.

    One CTA runs its value heads one after another, so splitting a key head over 2 or 4
    CTAs (each repeating the small per-key-head prologue) shortens the per-CTA chain and
    gives the scheduler finer-grained work. hpc = 1 only pays off while everything runs in
    one wave; hpc = 2 keeps winning over hpc = 4 well past one wave (5-25% at B = 16..128,
    cold clean L2), and the repeated prologue only catches up around B = 160-192, where
    the two are within 1.5%. B200 (148 SMs), 4 value heads per key head: 1 for B <= 9, 2
    for B <= 148, else 4 (3 per key head: 1 for B <= 12, else 3)."""
    slots = _CTA_SLOTS_PER_SM * num_sms
    hpk = heads_per_key
    if batch * n_key_heads * hpk <= slots:
        return 1
    if hpk % 2 == 0 and batch * n_key_heads * (hpk // 2) <= _HPC2_MAX_WAVES * slots:
        return 2
    return hpk


def flush_first_row_order(hist_len: torch.Tensor, flush_min: int) -> torch.Tensor:
    """Permutation of batch rows with the flushing rows (hist_len >= flush_min) first."""
    return torch.argsort((hist_len < flush_min).to(torch.int32), stable=True).to(
        torch.int32
    )


def gated_delta_rule_mtp_ucache_flush_umma(
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
) -> torch.Tensor:
    """The UMMA (SM100) kernel behind ``gated_delta_rule_mtp_ucache_flush`` for the fp16-state
    arm, 1 <= T <= 8; callable directly with the same tensors and contract.

    ``heads_per_cta`` (a divisor of HV / H; None = by batch size, see ``auto_heads_per_cta``)
    sets how many
    of a key head's 4 value heads one CTA runs.

    ``stochastic_rounding`` rounds the fp16 state written by flush rows stochastically
    (``cvt.rs.f16x2``, sm_100a / sm_103a / sm_107a); outputs, rings and verify rows are unchanged:

    * None (default) or "none": off, round to nearest; ``rand_seed`` is not read.
    * "philox": FlashInfer Mamba's scheme and defaults -- Philox4x32 with ``philox_rounds``
      rounds, key = ``rand_seed``, counter = flat offset of the 8-element chunk in the state
      pool, one draw per chunk feeding 4 pairs. Reproducible per seed.
    * "lcg": an LCG dither -- one LCG step per pair, bits s ^ (s >> 16)
      -- started from a murmur3 hash of the (state row, half) id and the seed. Cheapest; each
      element is unbiased and flushes are independent, but dithers within one state row are
      not jointly independent (see ``_rmw``). With ``rand_seed`` it is reproducible per seed;
      without, the seed is %clock (not reproducible).

    ``rand_seed`` is a single-element int64 CUDA tensor read on device, so a captured CUDA
    graph picks up updates; advance it every step (and use a different seed per layer), or
    repeated flushes of a slot reuse the same random bits.

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
    assert K_ == K_DIM and V_ == V_DIM, f"K == V == {K_DIM} required"
    assert HV % H == 0 and 1 <= HV // H <= HPK, (
        f"1 to {HPK} value heads per key head required, got HV={HV}, H={H}"
    )
    assert 1 <= T <= 8, f"T={T} unsupported (1 <= T <= 8)"
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert (
        k_cache is not None
        and u_cache is not None
        and g_cache is not None
        and hist_len is not None
    )
    # the per-CTA Gram runs bf16 mma.sync on the ring keys and q/k: bf16 k ring. The u
    # ring is only read and appended through dtype-aware conversions: bf16 or fp16.
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
    if flush_min is None:
        flush_min = W_RING - T + 1
    assert 1 <= flush_min <= W_RING - T + 1
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
        key = (B, str(q.device))
        row_order = _ROW_ID.get(key)
        if row_order is None:
            row_order = torch.arange(B, dtype=torch.int32, device=q.device)
            _ROW_ID[key] = row_order
    target = gdn_device_target(q.device)
    hpk = HV // H
    hpc = (
        auto_heads_per_cta(B, H, target.num_sms, hpk)
        if heads_per_cta is None
        else int(heads_per_cta)
    )
    if not (1 <= hpc <= hpk and hpk % hpc == 0):
        raise ValueError(
            f"heads_per_cta={heads_per_cta!r}: must divide the {hpk} value heads per key head"
        )
    mode = "none" if stochastic_rounding is None else stochastic_rounding
    if mode not in ("none", "philox", "lcg"):
        raise ValueError(
            f"stochastic_rounding={stochastic_rounding!r}: use None, 'none', 'philox' or 'lcg'"
        )
    if mode == "philox" and rand_seed is None:
        raise ValueError("stochastic_rounding='philox' needs rand_seed")
    if mode != "none" and (target.major, target.minor) not in _CVT_RS_CC:
        raise NotImplementedError(
            "stochastic rounding needs cvt.rs (sm_100a / sm_103a / sm_107a)"
        )
    if mode == "philox" and philox_rounds <= 0:
        raise ValueError(f"philox_rounds must be > 0, got {philox_rounds}")
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
        key = ("seed", str(q.device))
        rand_seed = _ROW_ID.get(key)
        if rand_seed is None:
            rand_seed = torch.zeros(
                1, dtype=torch.int64, device=q.device
            )  # unread placeholder
            _ROW_ID[key] = rand_seed
    fn = _get_kernel(
        q.device, T, H, HV, u_cache.dtype == torch.bfloat16, use_ro, hpc, sr, sr_rounds
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
