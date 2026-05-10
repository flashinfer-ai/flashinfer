# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
"""
TGV GEMM (low-latency Blackwell GEMM) — CuTe Ext (``cute_ext``)
implementation. Companion to ``include/flashinfer/gemm/tgv_gemm.cuh``
(C++).

Computes `C = A @ B` for batched bf16 → fp32-accum → bf16 GEMM:
  A:  (M, K, L)  K-major,  bf16
  B:  (N, K, L)  K-major,  bf16
  C:  (M, N, L)  M-major,  bf16

## Warp specialization (8 warps, 256 threads/CTA; warp 3 idle)
  Warp 0     DMA_A   TMA-loads A tiles into sA[..., stage]
  Warp 1     DMA_B   TMA-loads B tiles into sB[..., stage]; PDL
                     griddepcontrol.wait
  Warp 2     MMA     tcgen05.mma into TMEM; owns alloc/dealloc
  Warps 4-7  EPILOG  TMEM → RMEM → bf16 cast → STG (no TMA store)

## Mbarriers
  bar_full[stage]   2 arrivals (DMA_A+DMA_B mbarrier.arrive.expect_tx)
  bar_empty[stage]  1 arrival  (MMA tcgen05.commit, elect_one'd)
  bar_tma_epilog    32 arrivals (DMA_B warp; "B/activations issued")
  bar_mma_epilog    1 arrival   (MMA tcgen05.commit; "acc ready")
  bar_tmem_alloc    160 arrivals (32 MMA + 128 EPILOG); two-phase:
                      phase 0→1 = MMA finished alloc_tmem
                      phase 1→0 = EPILOG finished tcgen05.ld (safe to dealloc)

A 1-element Int32 SMEM slot (tmem_base_ptr) is written by alloc_tmem and read
by retrieve_tmem_ptr in MMA + EPILOG.

## Naming convention (matches include/flashinfer/gemm/tgv_gemm.cuh)
Partitioned tensors are named `tXyZ`:
  tX = partition pattern  (tC=MMA, tA=TMA, tD=t2r-copy)
  yZ = underlying tensor  (gA/gB/gC = gmem; sA/sB = smem; tAcc = TMEM;
                           rAcc/rD = rmem)
e.g. `tDgC` = GMEM C viewed as the per-thread destination of the t2r copy.

## Default-config shapes (CTA_M=64, CTA_N=8, CTA_K=128, DMA_Stage=8, bf16)
  Mma instruction:  Mma_M=(16,4)=64 lanes, Mma_N=8, Mma_K=16
                    NumMma_K = CTA_K/Mma_K = 8  (inner-K trip count)
                    Tiles_K  = Gemm_K/CTA_K = 12  (K=1536)
  sA staged:  ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
           = (((16,4), 16), 1, 8, 8)  — Sw<3,4,3>
  sB staged:  ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
           = ((8, 16), 1, 8, 8)       — Sw<3,4,3>
  tAcc:       ((Mma_M, Mma_N), NumMma_M, NumMma_N)
           = (((16,4), 8), 1, 1)      — fp32 in TMEM

## Integration with FlashInfer
This module is wired up as the default ``"tgv"`` backend in
:func:`flashinfer.mm_bf16` / :func:`flashinfer.bmm_bf16` (the C++ TGV path
is reachable by flipping ``_TGV_USE_CPP`` in ``flashinfer/gemm/gemm_base.py``).
The 11 ``cute_ext`` tactics here mirror the 11 C++ tactics in
``include/flashinfer/gemm/tgv_gemm_configs.h`` 1:1, so tactic ids are
interchangeable across implementations and the same default
(tactic 1 = 64×8 / 8 stages) is used when autotune is off.

The kernel writes M-contiguous output, so the runner does the same A↔B
swap trick as the C++ runner (``gemm_fn(b.t(), a.t(), …)``) to make the
output naturally land in row-major (M, N) PyTorch tensors.

## Environment
  pip install 'nvidia-cutlass-dsl[cu13]'   # quote brackets in tcsh
"""

from typing import List, NamedTuple, Optional, Tuple, Type

import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute import experimental as cute_ext
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cutlass.cute.nvgpu import tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils


# Same configurations as ``getAllTgvConfigs()`` in
# ``include/flashinfer/gemm/tgv_gemm_configs.h``. ``cta_k`` is fixed at 128
# in both the C++ and cute_ext kernels. Tactic 1 (64×8 stages=8) is the default.
_TGV_CUTE_EXT_CTA_K: int = 128
_TGV_CUTE_EXT_DEFAULT_TACTIC: int = 1
_TGV_CUTE_EXT_TACTIC_CONFIGS: List[Tuple[int, int, int]] = [
    (64, 8, 6),     # 0
    (64, 8, 8),     # 1 (default)
    (64, 8, 10),    # 2
    (64, 8, 12),    # 3
    (64, 16, 6),    # 4
    (64, 16, 8),    # 5
    (64, 16, 10),   # 6
    (64, 32, 6),    # 7
    (64, 32, 8),    # 8
    (64, 64, 6),    # 9
    (128, 16, 6),   # 10
]


def get_tgv_cute_ext_tactic_num() -> int:
    return len(_TGV_CUTE_EXT_TACTIC_CONFIGS)


def get_tgv_cute_ext_default_tactic() -> int:
    return _TGV_CUTE_EXT_DEFAULT_TACTIC


class WorkTileInfo(NamedTuple):
    """Which output tile this CTA processes. Mirrors the original WorkTileInfo."""
    M_idx: cutlass.Int32
    N_idx: cutlass.Int32
    L_idx: cutlass.Int32
    K_idx_start: cutlass.Int32
    K_idx_end: cutlass.Int32


class TgvGemmCuteExtKernel:
    """
    Low-latency Blackwell GEMM kernel rewritten with cute_ext primitives,
    keeping the raw-mbarrier 7-warp specialization from the C++ kernel
    in ``include/flashinfer/gemm/tgv_gemm.cuh``.
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        cta_m: int = 64,
        cta_n: int = 8,
        cta_k: int = _TGV_CUTE_EXT_CTA_K,
        num_ab_stage: int = 8,
        use_pdl: bool = False,
        pdl_launch: Optional[bool] = None,
        pdl_count: int = -1,
        has_bias: bool = False,
    ):
        self.acc_dtype = acc_dtype
        self.cta_m = cta_m
        self.cta_n = cta_n
        self.cta_k = cta_k
        self.num_ab_stage = num_ab_stage
        self.use_pdl = use_pdl
        self.pdl_launch = pdl_launch if pdl_launch is not None else use_pdl
        self.pdl_count = pdl_count
        # has_bias: when True, kernel reads a (Gemm_M, Gemm_N, Gemm_L):(1,0,0)
        # bias tensor (M-broadcast over N,L), converts it to fp32 in RMEM, and
        # adds it to the accumulator before the bf16 cast. When False, all
        # bias-related code is elided via cutlass.const_expr.
        self.has_bias = has_bias

        # Fixed configuration matching the C++ / DSL kernels.
        self.threads_per_cta = 256          # 8 warps (warp 3 unused)
        self.cluster_shape = (1, 1, 1)      # 1 SM mode, 1x1 cluster, no multicast.

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,    # (Gemm_M, Gemm_K, Gemm_L), K-major
        b: cute.Tensor,    # (Gemm_N, Gemm_K, Gemm_L), K-major
        c: cute.Tensor,    # (Gemm_M, Gemm_N, Gemm_L), M-major
        bias: cute.Tensor, # (Gemm_M, Gemm_N, Gemm_L):(1,0,0) — unused when has_bias=False
        stream: cuda.CUstream,
    ):
        # Each CTA processes one (CTA_M, CTA_N) output tile, no persistence.
        grid = (
            cute.ceil_div(c.layout.shape[0], self.cta_m),
            cute.ceil_div(c.layout.shape[1], self.cta_n),
            c.layout.shape[2],
        )
        self.kernel(a, b, c, bias).launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
            stream=stream,
            use_pdl=self.pdl_launch,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,    # (Gemm_M, Gemm_K, Gemm_L), K-major
        mB: cute.Tensor,    # (Gemm_N, Gemm_K, Gemm_L), K-major
        mC: cute.Tensor,    # (Gemm_M, Gemm_N, Gemm_L), M-major
        mBias: cute.Tensor, # (Gemm_M, Gemm_N, Gemm_L):(1,0,0) — unused when has_bias=False
    ):
        """
        Device-side dispatcher: build MMA descriptor, allocate SMEM/TMEM/
        mbarriers, init barriers, dispatch into dma_a/dma_b/mma/epilog warps.
        """
        DMA_Stage = self.num_ab_stage

        # ---- Tiled MMA ----
        # make_trivial_tiled_mma picks the largest tcgen05.mma atom that fits
        # CTA_M×CTA_N. For bf16 (64,8): Mma_M=(16,4)=64, Mma_N=8, Mma_K=16.
        cta_group = tcgen05.CtaGroup.ONE     # 1-SM mode
        a_major = utils.LayoutEnum.from_tensor(mA).mma_major_mode()
        b_major = utils.LayoutEnum.from_tensor(mB).mma_major_mode()
        ab_dtype = mA.element_type           # bf16
        c_dtype = mC.element_type            # bf16
        d_layout = utils.LayoutEnum.from_tensor(mC)

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, a_major, b_major,
            self.acc_dtype, cta_group, (self.cta_m, self.cta_n),
        )

        # NumMma_K = CTA_K/Mma_K — inner-K trip count for the MMA warp.
        # bf16 default: Mma_K=16, CTA_K=128 → mma_inst_tile_k = 8.
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = self.cta_k // mma_inst_shape_k

        mnk_tiler  = (self.cta_m, self.cta_n, self.cta_k)   # full MNK tile
        a_tiler_mk = (self.cta_m, self.cta_k)               # CTA tile of A
        b_tiler_nk = (self.cta_n, self.cta_k)               # CTA tile of B
        c_tiler_mn = (self.cta_m, self.cta_n)               # CTA tile of C

        # ---- WorkTileInfo (static 1-tile-per-CTA scheduler) ----
        # blockIdx maps directly to (M_tile, N_tile, batch). K_idx_start/end
        # carry over from C++ for future split-K; here every CTA reduces the
        # full K range (= ceil(K/CTA_K) tiles, e.g. 12 for K=1536).
        bidx, bidy, bidz = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        work_tile_info = WorkTileInfo(
            M_idx=bidx,
            N_idx=bidy,
            L_idx=bidz,
            K_idx_start=cutlass.Int32(0),
            K_idx_end=cute.ceil_div(cute.size(mA, mode=[1]), self.cta_k),
        )
        k_tile_count = work_tile_info.K_idx_end - work_tile_info.K_idx_start

        # ---- SMEM A/B (DMA_Stage-staged, swizzled ComposedLayout) ----
        # alignment=1024 covers TMA's natural alignment requirements.
        a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, mnk_tiler, ab_dtype, DMA_Stage,
        )  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage) = (((16,4),16), 1, 8, 8)
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, mnk_tiler, ab_dtype, DMA_Stage,
        )  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage) = ((8,16), 1, 8, 8)

        sA = cute_ext.allocate(
            ab_dtype, cute.AddressSpace.smem, a_smem_layout_staged, alignment=1024,
        )
        sB = cute_ext.allocate(
            ab_dtype, cute.AddressSpace.smem, b_smem_layout_staged, alignment=1024,
        )

        # ---- TMEM accumulator layout (manual alloc, see MMA warp) ----
        # `make_fragment_C(acc_shape).layout` is a *fake* TMEM tensor — layout
        # is right, ptr is unset. MMA's alloc_tmem will fill in the ptr; we
        # use cute.make_tensor(tmem_ptr, acc_layout) to bind it. We don't use
        # cute_ext.allocate(tmem) because it doesn't emit
        # relinquish_tmem_alloc_permit (~3% PDL win — see MMA warp).
        # acc_shape = ((Mma_M, Mma_N), NumMma_M, NumMma_N) = (((16,4),8), 1, 1)
        acc_shape = tiled_mma.partition_shape_C((self.cta_m, self.cta_n))
        acc_layout = tiled_mma.make_fragment_C(acc_shape).layout

        # ---- Raw mbarriers (Int64 SMEM arrays) + tmem_base_ptr Int32 slot ----
        # Two flavors of barriers exist on Hopper/Blackwell:
        #   1. Named barriers (bar.arv/bar.sync) — Ampere-era, 16 hardware
        #      barriers per SM, only handle intra-CTA thread sync.
        #   2. Mbarriers (mbarrier.* PTX) — 64-bit in SMEM per barrier,
        #      software-programmable. Used here for (a) thread sync within a
        #      CTA, (b) TMA transaction-count tracking (TMA→SM signaling),
        #      and (c) 1-arrival "wake the consumer warp" patterns.
        # Allocated as 1D tensors; .iterator gives a Pointer[Int64] supporting
        # `bar + stage` arithmetic. Arrival counts are in the module docstring.
        bar_full_arr       = cute_ext.allocate(cutlass.Int64, cute.AddressSpace.smem,
                                               cute.make_layout(DMA_Stage), alignment=8)
        bar_empty_arr      = cute_ext.allocate(cutlass.Int64, cute.AddressSpace.smem,
                                               cute.make_layout(DMA_Stage), alignment=8)
        bar_tma_epilog_arr = cute_ext.allocate(cutlass.Int64, cute.AddressSpace.smem,
                                               cute.make_layout(1), alignment=8)
        bar_mma_epilog_arr = cute_ext.allocate(cutlass.Int64, cute.AddressSpace.smem,
                                               cute.make_layout(1), alignment=8)
        bar_tmem_alloc_arr = cute_ext.allocate(cutlass.Int64, cute.AddressSpace.smem,
                                               cute.make_layout(1), alignment=8)
        # alloc_tmem writes the TMEM base address to this slot; both MMA and
        # EPILOG read it back via retrieve_tmem_ptr.
        tmem_base_arr      = cute_ext.allocate(cutlass.Int32, cute.AddressSpace.smem,
                                               cute.make_layout(1), alignment=4)

        bar_full       = bar_full_arr.iterator        # Pointer[Int64], DMA_Stage
        bar_empty      = bar_empty_arr.iterator       # Pointer[Int64], DMA_Stage
        bar_tma_epilog = bar_tma_epilog_arr.iterator  # Pointer[Int64]
        bar_mma_epilog = bar_mma_epilog_arr.iterator  # Pointer[Int64]
        bar_tmem_alloc = bar_tmem_alloc_arr.iterator  # Pointer[Int64]
        tmem_base_ptr  = tmem_base_arr.iterator       # Pointer[Int32]

        # ---- Barrier init (1 thread, PTX mbarrier.init is single-thread) ----
        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(DMA_Stage):
                    cute.arch.mbarrier_init(bar_full + i, 2)   # DMA_A + DMA_B
                for i in range(DMA_Stage):
                    cute.arch.mbarrier_init(bar_empty + i, 1)  # MMA tcgen05.commit
                cute.arch.mbarrier_init(bar_tma_epilog, 32)    # whole DMA_B warp
                cute.arch.mbarrier_init(bar_mma_epilog, 1)     # MMA tcgen05.commit
                cute.arch.mbarrier_init(bar_tmem_alloc, 32 + 128)  # MMA + 4 EPILOG warps

        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        # ---- CTA-to-Value maps for TMA (constexpr) ----
        # Encodes "which cluster CTA is responsible for which slice", i.e.
        # the CTA-coord → logical-multicast-id layout. For our 1×1 cluster
        # (no multicast) it's an identity layout; cute_ext.tma_load still
        # needs this explicit layout to construct the TMA descriptor. The
        # underlying group_modes<0,3> on the MMA partition is what tells
        # the TMA "everything in mode-0 is one bulk copy"; with NumTma_M/N
        # = 1 the partitioned shape coalesces to ((TMA, NumTma_K), Tiles_K).
        a_cta_v_map = cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A")
        b_cta_v_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")

        # ---- Per-CTA tile views ----
        # local_tile(t, tiler, coord) = zipped_divide(t, tiler)[coord]; modes
        # passed `None` stay free.
        gA_tile = cute.local_tile(             # (CTA_M, CTA_K, Tiles_K) = (64,128,12)
            mA, a_tiler_mk,
            (work_tile_info.M_idx, None, work_tile_info.L_idx),
        )
        gB_tile = cute.local_tile(             # (CTA_N, CTA_K, Tiles_K) = (8,128,12)
            mB, b_tiler_nk,
            (work_tile_info.N_idx, None, work_tile_info.L_idx),
        )
        gD_tile = cute.local_tile(             # (CTA_M, CTA_N) = (64,8)
            mC, c_tiler_mn,
            (work_tile_info.M_idx, work_tile_info.N_idx, work_tile_info.L_idx),
        )
        # gBias_tile: same (CTA_M, CTA_N) shape as gD_tile but stride (1, 0)
        # — the same M element is repeated across N. local_tile preserves the
        # (1, 0, 0) stride from mBias, so this works automatically.
        gBias_tile = cute.local_tile(          # (CTA_M, CTA_N) with stride (1, 0)
            mBias, c_tiler_mn,
            (work_tile_info.M_idx, work_tile_info.N_idx, work_tile_info.L_idx),
        )

        # ---- Warp dispatch (warp 3 idle, kept for 256-thread parity) ----
        if warp_idx == 0:
            self.dma_a_warp(
                bar_full, bar_empty,
                gA_tile, sA, a_cta_v_map,
                k_tile_count,
            )
        elif warp_idx == 1:
            self.dma_b_warp(
                bar_full, bar_empty, bar_tma_epilog,
                gB_tile, sB, b_cta_v_map,
                k_tile_count,
            )
        elif warp_idx == 2:
            self.mma_warp(
                bar_full, bar_empty, bar_mma_epilog, bar_tmem_alloc,
                tiled_mma, sA, sB, tmem_base_ptr, acc_layout,
                mma_inst_tile_k, k_tile_count,
            )
        elif warp_idx >= 4:
            # Epilog tid is 128..255 in the CTA; offset to 0..127 for partition.
            epi_tid = tidx - 128
            self.epilog_warp(
                bar_tma_epilog, bar_mma_epilog, bar_tmem_alloc,
                tmem_base_ptr, acc_layout, gD_tile, gBias_tile,
                epi_tid, c_dtype, d_layout,
            )

        # Final cluster barrier — ensures all warps finish before TMEM dealloc
        # / kernel exit.
        cute.arch.barrier()

    # ====================================================================
    # DMA_A WARP — TMA-loads A tiles into sA[..., stage], one per K-iter.
    # ====================================================================
    @cute.experimental.jit
    def dma_a_warp(
        self,
        bar_full,                # Pointer[Int64], DMA_Stage entries
        bar_empty,               # Pointer[Int64], DMA_Stage entries
        gA_tile: cute.Tensor,    # (CTA_M, CTA_K, Tiles_K) — this CTA's A strip
        sA: cute.Tensor,         # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        a_cta_v_map: cute.Layout,
        k_tile_count: cutlass.Int32,
    ):
        DMA_Stage = self.num_ab_stage

        # Phase-bit walkthrough for `bar_empty` (DMA_Stage=2 example).
        # Init phase = 0; phase flips on each arrival-count round. We wait
        # on the *old* phase about to flip away.
        #   k_tile  stage  old empty_phase  → wait for flip
        #     0       0          1                0→1 (init=0≠1, passes)
        #     1       1          1                0→1
        #     2       0          0                1→0 (slot reused)
        #     3       1          0                1→0
        #     4       0          1                0→1
        # Flip empty_phase once per DMA_Stage iters.
        empty_phase = cutlass.Int32(1)
        pdl_count = self.pdl_count

        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)

            # cute_ext.tma_load only manages TX bytes, NEVER the arrival
            # count (in the pipeline path producer_commit handles that).
            # bar_full[stage] needs 2 arrivals (DMA_A + DMA_B), so we arrive
            # ourselves via the fused mbarrier.arrive.expect_tx PTX, paired
            # with update_expect_tx=False so TX bytes aren't double-counted.
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    # 1 stage of A SMEM = sizeof(bf16) × CTA_M × CTA_K = 16 KB
                    cute.size_in_bytes(
                        sA.element_type,
                        cute.slice_(sA.layout, (None, None, None, 0)),
                    ),
                )
            cute_ext.tma_load(
                gA_tile[None, None, k_tile],     # (CTA_M, CTA_K) GMEM slice
                sA[None, None, None, stage],     # ((Mma_M,Mma_K),NumMma_M,NumMma_K)
                (bar_full + stage).value,        # Pointer→ir.Value bridge
                cta_v_map=a_cta_v_map,
                tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                update_expect_tx=False,
            )

            if (k_tile % DMA_Stage) == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1

            # PDL: launch dependent grid at pdl_count (default -1 = end).
            if cutlass.const_expr(self.use_pdl):
                if k_tile == pdl_count:
                    cute.arch.griddepcontrol_launch_dependents()

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

    # ====================================================================
    # DMA_B WARP — same as DMA_A but for B; also drives the PDL
    # griddepcontrol.wait and signals the epilog when all activation loads
    # are issued.
    # ====================================================================
    @cute.experimental.jit
    def dma_b_warp(
        self,
        bar_full,                # Pointer[Int64], DMA_Stage entries
        bar_empty,               # Pointer[Int64], DMA_Stage entries
        bar_tma_epilog,          # Pointer[Int64], 1 entry (32-arrival)
        gB_tile: cute.Tensor,    # (CTA_N, CTA_K, Tiles_K) — this CTA's B strip
        sB: cute.Tensor,         # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        b_cta_v_map: cute.Layout,
        k_tile_count: cutlass.Int32,
    ):
        DMA_Stage = self.num_ab_stage

        # PDL griddepcontrol.wait: B is the activation, produced by an
        # upstream kernel. Block until that kernel signals completion. A
        # is weights (already resident), so DMA_A doesn't need this.
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        # See dma_a_warp for empty_phase + fused arrive-and-expect-tx logic.
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)

            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    # 1 stage of B SMEM = sizeof(bf16) × CTA_N × CTA_K = 2 KB
                    cute.size_in_bytes(
                        sB.element_type,
                        cute.slice_(sB.layout, (None, None, None, 0)),
                    ),
                )
            cute_ext.tma_load(
                gB_tile[None, None, k_tile],     # (CTA_N, CTA_K) GMEM slice
                sB[None, None, None, stage],     # ((Mma_N,Mma_K),NumMma_N,NumMma_K)
                (bar_full + stage).value,
                cta_v_map=b_cta_v_map,
                tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                update_expect_tx=False,
            )

            if (k_tile % DMA_Stage) == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1

        # 32-thread arrive on bar_tma_epilog: signals epilog "activations issued"
        # (useful as a PDL-prefetch fence even when there's no bias/residual).
        cute.arch.mbarrier_arrive(bar_tma_epilog)

    # ====================================================================
    # MMA WARP — owns the TMEM accumulator and issues every tcgen05.mma.
    # ====================================================================
    @cute.experimental.jit
    def mma_warp(
        self,
        bar_full,                                  # Pointer[Int64], DMA_Stage entries
        bar_empty,                                 # Pointer[Int64], DMA_Stage entries
        bar_mma_epilog,                            # Pointer[Int64], 1 entry
        bar_tmem_alloc,                            # Pointer[Int64], 1 entry, 160-arrival
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,         # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        sB: cute.Tensor,         # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        tmem_base_ptr,           # Pointer[Int32] — SMEM slot for TMEM addr
        acc_layout: cutlass.Constexpr,             # TMEM accumulator layout
        mma_inst_tile_k: cutlass.Constexpr,        # NumMma_K — inner loop count
        k_tile_count: cutlass.Int32,               # Tiles_K — outer loop count
    ):
        DMA_Stage = self.num_ab_stage
        cta_group = tcgen05.CtaGroup.ONE

        # ---- TMEM allocation (manual) ----
        # We allocate TMEM explicitly here rather than via the convenience
        # helper ``cute_ext.allocate(tmem)`` so that we can *hide the TMEM
        # allocation latency*: the alloc must happen here in MMA's prolog
        # (we need the TMEM ptr before the first tcgen05.mma), but it
        # serializes against the next CTA's own alloc on the back-to-back
        # PDL launch path. By manually emitting
        # ``relinquish_tmem_alloc_permit`` right after our alloc completes,
        # the next CTA can start ITS allocation in parallel with this
        # CTA's MMA work rather than stalling at its own prolog waiting
        # for us. cute_ext.allocate(tmem) does not expose this early-
        # relinquish control — worth ~3% on the PDL hot path.
        #
        # TMEM on SM100 has 128 lanes × 512 columns × 4B = 256KB total. We
        # allocate half (256 of 512 cols) so the next CTA can prefetch its
        # alloc on the other half. For CTA_M=64 the accumulator only uses
        # 64 lanes (16 lanes × 4 subpartitions), well within half-of-TMEM.
        num_tmem_cols = 256
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_ptr)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)        # phase 0: 32 of 160
        cute.arch.relinquish_tmem_alloc_permit()         # let next CTA alloc

        # Bind acc_layout to the just-allocated TMEM ptr.
        # acc_view: ((Mma_M, Mma_N), NumMma_M, NumMma_N) = (((16,4),8), 1, 1)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        acc_view = cute.make_tensor(tmem_ptr, acc_layout)

        # MMA atom: each cute_ext.dot call = 1 tcgen05.mma. ACCUMULATE:
        #   False → C  = A*B  (first inst of first k-tile only)
        #   True  → C += A*B
        mma_atom = cute.make_mma_atom(tiled_mma.op)

        # ---- Outer K-loop with try_wait/wait overlap ----
        # The naive `wait → mma...mma → commit` body has the wait spin-loop
        # blocking tcgen05.mma bookkeeping. We run try_wait one step ahead
        # so the blocking wait only runs on a miss — same pattern as
        # cutlass/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp.
        # `old_stage_idx` = THIS iter's stage; `stage_idx` is one ahead
        # (used for the next iter's try_wait). We also manage stage_idx as a
        # manual int counter rather than `k_tile % DMA_Stage` because for
        # non-power-of-2 DMA_Stage the modulo is non-trivial in the hot path.
        #
        # Phase-bit walkthrough (DMA_Stage=2 example), full_phase init = 0:
        #   k_tile  stage  full_phase  → wait for flip
        #     0       0         0           0→1   (1 = slot full)
        #     1       1         0           0→1
        #     2       0         1           1→0   (0 = slot full, slot reused)
        #     3       1         1           1→0
        # full_phase flips whenever stage_idx wraps to 0.
        full_phase    = cutlass.Int32(0)
        stage_idx     = cutlass.Int32(0)
        old_stage_idx = cutlass.Int32(0)

        wait_complete = cute.arch.mbarrier_try_wait(bar_full + stage_idx, full_phase)

        for k_tile in cutlass.range(k_tile_count):
            if ~wait_complete:
                cute.arch.mbarrier_wait(bar_full + stage_idx, full_phase)

            # Advance stage_idx (manual int counter; avoids modulo in hot path).
            old_stage_idx = stage_idx
            stage_idx = stage_idx + 1
            if stage_idx == DMA_Stage:
                full_phase = full_phase ^ 1
                stage_idx  = cutlass.Int32(0)

            # Try-wait one step ahead — overlaps the wait with the MMA below.
            if k_tile < (k_tile_count - 1):
                wait_complete = cute.arch.mbarrier_try_wait(
                    bar_full + stage_idx, full_phase,
                )

            # Inner-K loop: NumMma_K straight-line tcgen05.mma instructions.
            # cute_ext.dot expects rank-3 operands in the MMA fragment profile
            # (MMA_atom, REST_M, REST_K). Slicing `sA[None,None,k_block,stage]`
            # leaves rank 2 ((Mma_M, Mma_K), NumMma_M), so pad explicitly via
            # cute.append_ones.
            for k_block in range(mma_inst_tile_k):
                if k_block == 0:
                    # First inst of the very first k_tile clears TMEM.
                    mma_atom.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                else:
                    mma_atom.set(tcgen05.Field.ACCUMULATE, True)
                a_frag = cute.append_ones(
                    sA[None, None, k_block, old_stage_idx], up_to_rank=3,
                )
                b_frag = cute.append_ones(
                    sB[None, None, k_block, old_stage_idx], up_to_rank=3,
                )
                cute_ext.dot(mma_atom, a_frag, b_frag, acc_view)

            # CRITICAL: tcgen05.commit MUST be inside elect_one(). Unlike
            # the C++ ``cutlass::arch::umma_arrive`` helper (which has an
            # internal elect_one_sync), the DSL's tcgen05.commit has no
            # guard — without this, all 32 lanes commit (32× redundant
            # arrivals on bar_empty). This was a major perf bug encountered
            # during development.
            with cute.arch.elect_one():
                tcgen05.commit(bar_empty + old_stage_idx, None, cta_group)

        # Wake the epilog (1-thread arrive on the 1-arrival barrier).
        with cute.arch.elect_one():
            tcgen05.commit(bar_mma_epilog, None, cta_group)

        # ---- TMEM dealloc: wait for EPILOG's read to complete, then free ----
        # bar_tmem_alloc phase 1 fires after EPILOG's tcgen05.ld is observable
        # (post fence_view_async_tmem_load).
        cute.arch.mbarrier_arrive(bar_tmem_alloc)        # phase 1: 32 of 160
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols)

    # ====================================================================
    # EPILOG WARPS — TMEM → RMEM → bf16 cast → direct STG to GMEM.
    # ====================================================================
    @cute.experimental.jit
    def epilog_warp(
        self,
        bar_tma_epilog,                            # Pointer[Int64], 1 entry (32-arrival)
        bar_mma_epilog,                            # Pointer[Int64], 1 entry
        bar_tmem_alloc,                            # Pointer[Int64], 1 entry, 160-arrival
        tmem_base_ptr,               # Pointer[Int32] — SMEM slot from MMA
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,        # (CTA_M, CTA_N) — this CTA's output tile
        gBias_tile: cute.Tensor,     # (CTA_M, CTA_N) with stride (1,0) — bias broadcast
        epi_tid: cutlass.Int32,      # 0..127 within the 4 EPILOG warps
        c_dtype: cutlass.Constexpr,
        d_layout: cutlass.Constexpr,
    ):
        # Sync with MMA's alloc_tmem (phase 0 of bar_tmem_alloc): 128 arrivals
        # from this warp + 32 from MMA = 160. tmem_base_ptr is uninitialized
        # before the wait clears.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)

        # Bind acc_layout to the TMEM ptr written by alloc_tmem.
        # tCtAcc = ((Mma_M, Mma_N), NumMma_M, NumMma_N) = (((16,4),8), 1, 1)
        # acc_view = (Mma_M, Mma_N) — drop the trivial 1×1 outer modes.
        #
        # Concrete TMEM layout for bf16 M64 N8:
        #   tCtAcc:  tmem_[32b](0x0...) o (((16,4),8),1,1):(((65536,2097152),1),0,0)
        #   TMEM addr = [31:16=dp_lane, 15:0=column]:
        #     stride between dp lanes = 1<<16 = 65536           (Mma_M_per_subp)
        #     stride between subparts = 65536 × 32 = 2097152    (NumSubp=4)
        #     stride between cols     = 1                       (N is contiguous)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        tCtAcc = cute.make_tensor(tmem_ptr, acc_layout)
        acc_view = tCtAcc[((None, None), 0, 0)]

        # ---- t2r tiled-copy + per-thread RMEM layout ----
        # ``sm100_utils.get_tmem_load_op`` picks the best tcgen05.ld atom
        # for the (M, N, K) tile and output dtype. For CTA_M=64, M/N-major
        # output, the chosen atom is SM100_TMEM_LOAD_16dp256b1x — the 16dp
        # variant is required because each MMA subpartition only uses 16
        # of the 32 lanes per subpartition (CTA_M=64 split across 4 subps =
        # 16 lanes each). Other tcgen05.ld atoms would also be functional;
        # 16dp is just optimal here.
        # epi_tile = full CTA tile: N=8 fits in one tcgen05.ld instance
        # (16dp × 8col → 4 regs per thread × 128 threads). Larger N would
        # need a sub-tile loop here.
        epi_tile = (self.cta_m, self.cta_n)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            (self.cta_m, self.cta_n, self.cta_k),
            d_layout, c_dtype, self.acc_dtype, epi_tile, False,
        )
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(copy_atom_t2r, acc_view)

        # flat_divide → (CTA_M, CTA_N, Rest_M=1, Rest_N=1); the rank-4 shape
        # is required because make_t2r_rmem_layout internally does [...,0,0].
        gD_epi = cute.flat_divide(gD_tile, epi_tile)

        # rmem_layout matches one t2r copy's destination. The TMEM-side
        # partition (CpyS) per-subpartition view is shape ((CpyS_N=8,
        # CpyS_M=16), NumCpy_M=1, NumCpy_N=Mma_N/CpyS_N) — i.e. each
        # tcgen05.ld.16dp256bit.x1 instance moves 16dp×8col elements per
        # subpartition. Per the PTX page for tcgen05.ld.16dp256bit, thread
        # 0 of an SM100_TMEM_LOAD_16dp256b1x copy receives 4 registers at
        # output coordinates (0,0), (0,1), (8,0), (8,1) — i.e. a (2,2)
        # per-thread value-tile, hence CpyD = (2,2). The dst rmem layout
        # mirrors this CpyD shape but with stride (1, 2) so the registers
        # are stored contiguously.
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rAcc = cute_ext.allocate(                  # fp32, per-thread
            self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32,
        )
        rD = cute_ext.allocate(                    # bf16, per-thread
            c_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32,
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        # ---- Bias setup (GMEM→RMEM via partition_and_copy, predicate auto) ----
        # gBias_tile has the same (CTA_M, CTA_N) shape as gD_tile but with
        # stride (1, 0) — the bias value depends only on the M coord. Each
        # lane needs to read the same (m, n) coord it writes back to gD so
        # bias and accumulator align register-for-register.
        #
        # partition_and_copy slices the GMEM src using
        # `tiled_copy.layout_src_tv_tiled`. With t2r that's the TMEM (CpyS)
        # side, which is 16dp×8col=128 elements/group — doesn't match rBias's
        # CpyD=4 elements/thread. `make_tiled_copy_D` builds a new TiledCopy
        # whose src/dst TV layouts both equal t2r's *dst* (CpyD) layout, so
        # the GMEM src gets sliced to CpyD per thread — matching rBias. OOB
        # is handled by the cute_ext lowering's auto-predBounds (same story
        # as the STG to mC).
        if cutlass.const_expr(self.has_bias):
            bias_dtype = gBias_tile.element_type
            gBias_epi = cute.flat_divide(gBias_tile, epi_tile)
            tiled_copy_g2r = cute.make_tiled_copy_D(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), bias_dtype),
                tiled_copy_t2r,
            )
            thr_g2r = tiled_copy_g2r.get_slice(epi_tid)
            rBias = cute_ext.allocate(             # bias dtype (e.g. bf16)
                bias_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32,
            )
            rBiasAcc = cute_ext.allocate(          # converted to fp32
                self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32,
            )

        # ---- Wait for data: B loads done + MMA done ----
        cute.arch.mbarrier_wait(bar_tma_epilog, 0)   # all activations issued

        # ---- Bias load: GMEM → RMEM (predicated ldg) and convert to fp32 ----
        # Done before the MMA wait so the load latency hides behind MMA.
        if cutlass.const_expr(self.has_bias):
            cute_ext.partition_and_copy(thr_g2r, gBias_epi[None, None, 0, 0], rBias)
            rBiasAcc.store(rBias.load().to(self.acc_dtype))

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)   # accumulator ready

        # TMEM → RMEM (one tcgen05.ld for the 64×8 tile).
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)

        # tcgen05.ld is async — fence makes the result visible to (a) the
        # rAcc.load() below and (b) MMA's dealloc_tmem after we arrive
        # on bar_tmem_alloc phase 1.
        cute.arch.fence_view_async_tmem_load()

        # Phase 1 of bar_tmem_alloc: 128 from this warp + 32 from MMA = 160.
        # MMA's mbarrier_wait(bar_tmem_alloc, 1) clears and it dealloc's.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        # Add bias in fp32 before the dtype cast.
        if cutlass.const_expr(self.has_bias):
            rAcc.store(rAcc.load() + rBiasAcc.load())

        # fp32 → bf16 cast in RMEM.
        rD.store(rAcc.load().to(c_dtype))

        # RMEM → GMEM: cute_ext.partition_and_copy does two things —
        #   (a) PARTITION: splits non-RMEM operands by thr_copy's thread-value
        #       layout (RMEM operands skipped — already per-thread). Here only
        #       gD_epi (the dst) gets partitioned; rD stays as-is.
        #   (b) INSTRUCTION: picked from (src, dst) memspace pair —
        #         TMEM→RMEM = tcgen05.ld     (atom-driven)
        #         RMEM→GMEM = STG            (simt_auto_vec_copy)
        #         GMEM→SMEM = cp.async       (async_op=True)
        # We REUSE thr_t2r so each lane writes to the same (m, n) GMEM coords
        # where it read from TMEM. A different copy atom would scatter
        # registers to the wrong addresses.
        #
        # OOB STORES are handled automatically by the cute_ext lowering —
        # simt_auto_vec_copy infers predBounds from the destination tensor's
        # MemRef shape (propagated from mC through local_tile / flat_divide).
        # Lanes whose (m, n) coord is past mC's shape simply don't issue an
        # STG. Verified with a non-aligned M=63 test: writes to in-bounds rows
        # 0..62 land correctly, the M=63 row + the rest of the (64,8) tile
        # stay untouched. So the C++ / vanilla-DSL kernels' explicit predicate
        # dance (make_identity_tensor + elem_less + basic_copy_if) isn't
        # needed; the explicit `predicated_tensor_origin` marker isn't either.
        cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])


# =====================================================================
# BMM wrappers — feed (L, M, K) / (L, K, N) / (L, M, N) PyTorch-shaped
# cute tensors and select-permute them into the kernel's (M, K, L) /
# (N, K, L) / (M, N, L) view.
# =====================================================================
@cute.experimental.jit
def _bmm_no_bias(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,            # (L, M, K) from PyTorch
    b: cute.Tensor,            # (L, K, N) from PyTorch (permuted)
    c: cute.Tensor,            # (L, M, N) from PyTorch (permuted)
    stream: cuda.CUstream,
):
    a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0]))
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    # Dummy bias aliasing C — never dereferenced when has_bias=False.
    bias = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[0, 1, 2]))
    gemm_op(a, b, c, bias, stream)


@cute.experimental.jit
def _bmm_bias(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,            # (L, M, K) from PyTorch
    b: cute.Tensor,            # (L, K, N) from PyTorch (permuted)
    c: cute.Tensor,            # (L, M, N) from PyTorch (permuted)
    bias: cute.Tensor,         # (L, M, N):(0,1,0) — M-broadcast via as_strided
    stream: cuda.CUstream,
):
    a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0]))
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    # After [1,2,0] permute: (M, N, L):(1, 0, 0) — matches C++ make_layout_Bias.
    bias = cute.make_tensor(bias.iterator, cute.select(bias.layout, mode=[1, 2, 0]))
    gemm_op(a, b, c, bias, stream)


# =====================================================================
# Compile + cache helpers (FlashInfer-side wrappers around the kernel).
# Caches by *config* only — the kernel uses ``mark_layout_dynamic`` so
# the same compiled binary handles any shape. Representative tensors used
# at compile time pin the dtype, rank, and leading-dim pattern; their
# concrete extents are erased.
# =====================================================================
_TORCH_TO_CUTLASS_DTYPE = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
}

# Per-process cache mapping (dtype, config…) → compiled cute_ext callable.
# We need to construct concrete cute.Tensors once and reuse the resulting
# compiled function across all live calls; a fresh build per call would
# defeat the cache.
_TGV_CUTE_EXT_COMPILE_CACHE: dict = {}


def _make_compile_repr_tensors(dtype: torch.dtype, has_bias: bool):
    """Build representative (1, M, K) / (1, K, N) / (1, M, N) tensors for
    the cute_ext.compile call. Concrete sizes are erased by
    ``mark_layout_dynamic``; only the rank and leading-dim pattern matter
    for the resulting binary.
    """
    # Pick aligned-but-small extents that satisfy the kernel's tile shape.
    M, N, K, L = 64, 8, 128, 1
    A_t = torch.empty((L, M, K), dtype=dtype, device="cuda")
    B_t = torch.empty((L, N, K), dtype=dtype, device="cuda").permute(0, 2, 1)
    C_t = torch.empty((L, N, M), dtype=dtype, device="cuda").permute(0, 2, 1)

    a_ = from_dlpack(A_t, assumed_align=32).mark_layout_dynamic(leading_dim=2)
    b_ = from_dlpack(B_t, assumed_align=32).mark_layout_dynamic(leading_dim=1)
    c_ = from_dlpack(C_t, assumed_align=32).mark_layout_dynamic(leading_dim=1)

    if not has_bias:
        return a_, b_, c_, None

    bias_t = torch.empty((M,), dtype=dtype, device="cuda")
    bias_3d = bias_t.as_strided(size=(L, M, N), stride=(0, 1, 0))
    bias_ = from_dlpack(bias_3d, assumed_align=2).mark_layout_dynamic(leading_dim=1)
    return a_, b_, c_, bias_


def _get_compiled_cute_ext_kernel(
    dtype: torch.dtype,
    cta_m: int,
    cta_n: int,
    cta_k: int,
    num_ab_stage: int,
    use_pdl: bool,
    has_bias: bool,
):
    """Compile (or fetch from cache) a TgvGemmCuteExtKernel for the given
    config. Returns a callable with signature
        compiled(a_, b_, c_, [bias_,] stream)
    accepting cute.Tensors. The same compiled binary handles any
    (M, N, K, L) consistent with the original layout pattern thanks to
    ``mark_layout_dynamic``.
    """
    key = (dtype, cta_m, cta_n, cta_k, num_ab_stage, bool(use_pdl), bool(has_bias))
    cached = _TGV_CUTE_EXT_COMPILE_CACHE.get(key)
    if cached is not None:
        return cached

    if dtype not in _TORCH_TO_CUTLASS_DTYPE:
        raise ValueError(
            f"TGV cute_ext backend supports {list(_TORCH_TO_CUTLASS_DTYPE)}; got {dtype}."
        )

    gemm = TgvGemmCuteExtKernel(
        acc_dtype=cutlass.Float32,
        cta_m=cta_m, cta_n=cta_n, cta_k=cta_k,
        num_ab_stage=num_ab_stage,
        use_pdl=use_pdl,
        has_bias=has_bias,
    )

    a_, b_, c_, bias_ = _make_compile_repr_tensors(dtype, has_bias)
    fake_stream = make_fake_stream()

    if has_bias:
        compiled = cute_ext.compile(_bmm_bias, gemm, a_, b_, c_, bias_, fake_stream)
    else:
        compiled = cute_ext.compile(_bmm_no_bias, gemm, a_, b_, c_, fake_stream)

    _TGV_CUTE_EXT_COMPILE_CACHE[key] = compiled
    return compiled


# =====================================================================
# Tensor adaptation: PyTorch (M, N) row-major -> cute view that matches
# the kernel's M-contiguous output requirement via the same A↔B swap as
# the C++ runner.
# =====================================================================
def _to_cute_swap(
    a_pt: torch.Tensor,        # (..., M, K) row-major (K-contig)
    b_pt: torch.Tensor,        # (..., K, N) col-major (K-contig)
    out_pt: torch.Tensor,      # (..., M, N) row-major (N-contig)
    bias_pt: Optional[torch.Tensor],  # (N,) or None — bias is per-output-feature
):
    """Build the cute.Tensors fed to ``_bmm_no_bias`` / ``_bmm_bias``.

    Internally swaps A↔B so the kernel writes its M-contiguous output into
    the *N*-axis of a row-major PyTorch tensor — same trick as the C++
    runner's ``gemm_fn(b.t(), a.t(), …)``. After swap, the kernel's M =
    PyTorch N and the kernel's N = PyTorch M; bias broadcast over the
    kernel's M axis (= PyTorch N) correctly mirrors the
    per-output-feature semantics.
    """
    # Lift to 3D batched shape with a leading L=1 axis when called from mm.
    if a_pt.dim() == 2:
        a_pt = a_pt.unsqueeze(0)
        b_pt = b_pt.unsqueeze(0)
        out_pt = out_pt.unsqueeze(0)

    # Swap: feed b.transpose(-2,-1) as A_ce (shape (L, N, K)),
    # a.transpose(-2,-1) as B_ce (shape (L, K, M)), and
    # out.transpose(-2,-1) as C_ce (shape (L, N, M), M-contig).
    a_swap = b_pt.transpose(-2, -1)
    b_swap = a_pt.transpose(-2, -1)
    c_swap = out_pt.transpose(-2, -1)

    a_ = from_dlpack(a_swap, assumed_align=32).mark_layout_dynamic(leading_dim=2)
    b_ = from_dlpack(b_swap, assumed_align=32).mark_layout_dynamic(leading_dim=1)
    c_ = from_dlpack(c_swap, assumed_align=32).mark_layout_dynamic(leading_dim=1)

    if bias_pt is None:
        return a_, b_, c_, None

    # bias is per-output-feature in PyTorch terms. After the A↔B swap the
    # kernel's M-axis maps to PyTorch's N-axis, which is exactly what the
    # bias indexes — so this is a direct (0, 1, 0) broadcast.
    L = c_swap.shape[0]
    M_ce = c_swap.shape[1]   # == PyTorch N
    N_ce = c_swap.shape[2]   # == PyTorch M
    bias_3d = bias_pt.as_strided(size=(L, M_ce, N_ce), stride=(0, 1, 0))
    bias_ = from_dlpack(bias_3d, assumed_align=2).mark_layout_dynamic(leading_dim=1)
    return a_, b_, c_, bias_


def _resolve_tactic(tactic: int) -> Tuple[int, int, int]:
    """Return (cta_m, cta_n, num_ab_stage) for the given tactic id."""
    if tactic < 0:
        tactic = _TGV_CUTE_EXT_DEFAULT_TACTIC
    if tactic >= len(_TGV_CUTE_EXT_TACTIC_CONFIGS):
        raise ValueError(
            f"TGV cute_ext tactic {tactic} out of range [0, {len(_TGV_CUTE_EXT_TACTIC_CONFIGS)})."
        )
    return _TGV_CUTE_EXT_TACTIC_CONFIGS[tactic]


def run_tgv_cute_ext(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    pdl: bool,
    tactic: int,
) -> torch.Tensor:
    """Dispatch a single TGV cute_ext GEMM/BMM call.

    Accepts the same (a, b, bias, out, pdl) tuple as the C++ TGV runner
    expects from :func:`bf16_gemm_sm100`. Handles both 2D (mm) and 3D
    (bmm) inputs uniformly through the A↔B swap trick.
    """
    cta_m, cta_n, num_ab_stage = _resolve_tactic(tactic)
    has_bias = bias is not None

    compiled = _get_compiled_cute_ext_kernel(
        dtype=a.dtype,
        cta_m=cta_m,
        cta_n=cta_n,
        cta_k=_TGV_CUTE_EXT_CTA_K,
        num_ab_stage=num_ab_stage,
        use_pdl=bool(pdl),
        has_bias=has_bias,
    )

    a_, b_, c_, bias_ = _to_cute_swap(a, b, out, bias)
    stream = cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream)

    if has_bias:
        compiled(a_, b_, c_, bias_, stream)
    else:
        compiled(a_, b_, c_, stream)
    return out
