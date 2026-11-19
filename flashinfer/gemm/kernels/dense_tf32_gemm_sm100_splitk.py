# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Blackwell/Rubin low-M TF32 GEMM with an in-kernel cluster split-K reduction.

Computes ``out[M, N] = a[M, K] @ b[K, N]`` with FP32 inputs, TF32 tcgen05 MMAs,
FP32 accumulation and FP32 output in one launch (cuBLAS's TF32 path runs a
split-K GEMM plus a separate reduce kernel at low M).

Both operands are TMA-loaded through a TF32 tensor map, so the copy engine
rounds them FP32 -> TF32 (round to nearest even) on the way into shared
memory: the MMA sees exactly the operands of a round-to-nearest TF32 GEMM.

The public problem is swapped so N fills the MMA M dimension and the few rows
of ``a`` sit in the MMA N dimension. In kernel coordinates:
  A:  (N, K, 1)  K-major  -- ``b.T``
  B:  (M, K, 1)  K-major  -- ``a``
  C:  (N, M, 1)  N-major  -- view of the row-major ``out``
N and K are static (one compile per (N, K) and tactic); M is dynamic.

## Grid and clusters
  grid    = (ceil(N / cta_m), ceil(M / cta_n) * split_k, 1),
  cluster = (1, split_k, 1)
  The ``split_k`` CTAs of a cluster share one (cta_m, cta_n) output tile and
  each accumulates an equal, contiguous K slice; the epilogue reduces the
  slices over DSMEM. There is no TMA multicast: each CTA loads its own slice.

## Warp specialization (8 warps, 256 threads/CTA)
  Warps 0-3  EPILOG  TMEM alloc/free; TMEM -> RMEM, split-K reduce-scatter
                     over DSMEM, st.global
  Warp 4     MMA     tcgen05.mma into TMEM; releases each stage
  Warp 5     LOAD_B  TMA-loads ``b`` tiles; with PDL it does not wait for the
                     previous kernel and launches the dependent grid once its
                     loads are issued
  Warp 6     LOAD_A  griddepcontrol.wait (PDL), then TMA-loads ``a`` tiles
  Warp 7     idle

## Split-K reduce-scatter epilogue
  Each rank reduces and stores 1 / split_k of the tile: it receives the other
  ranks' partials of its share by st.async into its sSlots (S - 1 slots, one
  per other rank in rank order, crediting its reduce mbarrier), waits for them
  (acquire at cluster scope), then adds its own partial and the peers' in rank
  order. The order is fixed, so the result is deterministic. Two variants:
  * Fragment-order exchange, when each rank's share of a thread's R TMEM
    registers is a multiple of 4: rank g owns registers [g R/S, (g+1) R/S) of
    every epilogue thread and the others are sent straight from registers as
    st.async.v4; no sStage.
  * Otherwise the partial tile is first staged in sStage (XOR-swizzled), and
    rank g owns the flat segment [g SEG, (g+1) SEG) of it (SEG = cta_m * cta_n
    / split_k, column-major), sent and reduced through 128-bit / 64-bit
    TiledCopies.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import itertools

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch as _torch
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.tensor_utils import LayoutEnum

from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
    _align_up,
    _max_active_clusters,
    _mbarrier_wait_acquire_cluster,
    _smem_capacity_bytes,
    _tcgen05_fence,
)
from flashinfer.gemm.kernels.utils import validate_tf32_gemm_inputs as validate_inputs

#: Kernel-M tiles (public N).
_SUPPORTED_MMA_M = (64, 128)

#: Kernel-N tiles (public M); larger M runs several tiles.
_SUPPORTED_MMA_N = (8, 16, 32, 64)

#: Cluster sizes along K; 16 is the non-portable cluster maximum.
_SUPPORTED_SPLIT_K = (2, 4, 8, 16)

#: K extent of one pipeline stage.
_SUPPORTED_CTA_K = (32, 64, 128)

#: A/B pipeline stage bounds.
_MIN_AB_STAGES = 2
_MAX_AB_STAGES = 24

_FP32_BYTES = 4
_AB_BUFFER_ALIGN_BYTES = 1024  # 128 B swizzle atoms of sA / sB
_MAILBOX_ALIGN_BYTES = 128
_MBARRIER_BYTES = 8

_EPILOG_THREADS = 128
_EPILOG_SYNC_BAR_ID = 1
_TMEM_ALLOC_SYNC_BAR_ID = 2


@dataclasses.dataclass(frozen=True)
class Tf32SplitKTactic:
    """One specialization; mma_m carries public N and mma_n carries public M."""

    mma_m: int
    mma_n: int
    split_k: int
    ab_stages: int
    cta_k: int = 64


def _cluster_capacity(device_index: int, split_k: int) -> int:
    """Co-resident clusters of ``split_k`` CTAs (an estimate where the query fails)."""
    try:
        return _max_active_clusters(device_index, split_k)
    except Exception:
        props = _torch.cuda.get_device_properties(device_index)
        return props.multi_processor_count // split_k


def _direct_exchange(tactic: Tf32SplitKTactic) -> bool:
    """Fragment-order exchange: each rank owns a range of every epilogue
    thread's accumulator registers and sends the others straight from registers
    as st.async.v4, so the range must be a multiple of 4 registers."""
    regs = tactic.mma_m * tactic.mma_n // _EPILOG_THREADS
    return regs % (4 * tactic.split_k) == 0


def _smem_bytes(tactic: Tf32SplitKTactic, ab_stages: int) -> int:
    """Mirror the device allocator's shared-memory layout, in allocation order."""
    # SharedStorage: a full and an empty mbarrier per A/B stage, the full and
    # empty mbarriers of the 1-stage accumulator pipeline, the reduce mbarrier.
    mbarriers = (2 * ab_stages + 2 + 1) * _MBARRIER_BYTES
    tmem_ptr = 4  # TMEM base address slot (Int32)
    # sA and sB are two rings indexed by the same stage (one full mbarrier per
    # stage covers both TMA loads); a stage of each is rows x cta_k FP32.
    row_bytes = tactic.cta_k * _FP32_BYTES * ab_stages  # one row, all stages
    sA_bytes = tactic.mma_m * row_bytes  # the b tile, mma_m rows
    sB_bytes = tactic.mma_n * row_bytes  # the a tile, mma_n rows
    tile = tactic.mma_m * tactic.mma_n
    # This rank's whole partial tile (none with the fragment-order exchange).
    sStage_bytes = 0 if _direct_exchange(tactic) else tile * _FP32_BYTES
    # The other split_k - 1 ranks' partials of this rank's segment.
    sSlots_bytes = (tactic.split_k - 1) * (tile // tactic.split_k) * _FP32_BYTES

    # Each buffer starts at the next aligned address after the previous one.
    cursor = mbarriers + tmem_ptr
    cursor = _align_up(cursor, _AB_BUFFER_ALIGN_BYTES) + sA_bytes
    cursor = _align_up(cursor, _AB_BUFFER_ALIGN_BYTES) + sB_bytes
    if sStage_bytes:
        cursor = _align_up(cursor, _MAILBOX_ALIGN_BYTES) + sStage_bytes
    cursor = _align_up(cursor, _MAILBOX_ALIGN_BYTES) + sSlots_bytes
    return cursor


def _max_ab_stages_for(tactic: Tf32SplitKTactic, smem_capacity: int, k: int) -> int:
    """Deepest ring that fits, never deeper than the K tiles of one rank."""
    k_tiles = k // tactic.cta_k // tactic.split_k
    limit = min(_MAX_AB_STAGES, max(_MIN_AB_STAGES, k_tiles))
    return next(
        (
            stages
            for stages in range(limit, _MIN_AB_STAGES - 1, -1)
            if _smem_bytes(tactic, stages) <= smem_capacity
        ),
        0,
    )


def validate_tactic(
    tactic: Tf32SplitKTactic,
    m: int,
    n: int,
    k: int,
    *,
    smem_capacity: int | None = None,
) -> None:
    """Reject a tactic that cannot serve ``(m, n, k)``.

    ``smem_capacity`` defaults to the current device's per-CTA capacity.
    """
    if smem_capacity is None:
        smem_capacity = _smem_capacity_bytes(_torch.cuda.current_device())
    if tactic.mma_m not in _SUPPORTED_MMA_M:
        raise ValueError(f"unsupported mma_m={tactic.mma_m}")
    if tactic.mma_n not in _SUPPORTED_MMA_N:
        raise ValueError(f"unsupported mma_n={tactic.mma_n}")
    if tactic.mma_m == 128 and tactic.mma_n % 16:
        raise ValueError(f"mma_m=128 needs mma_n % 16 == 0, got {tactic.mma_n}")
    if tactic.split_k not in _SUPPORTED_SPLIT_K:
        raise ValueError(f"unsupported split_k={tactic.split_k}")
    if tactic.cta_k not in _SUPPORTED_CTA_K:
        raise ValueError(f"unsupported cta_k={tactic.cta_k}")
    if not _MIN_AB_STAGES <= tactic.ab_stages <= _MAX_AB_STAGES:
        raise ValueError(
            f"ab_stages must be in [{_MIN_AB_STAGES}, {_MAX_AB_STAGES}], "
            f"got {tactic.ab_stages}"
        )
    if m < 1:
        raise ValueError(f"M must be positive, got {m}")
    # The epilogue stores pairs of FP32 values.
    if n <= 0 or n % 2:
        raise ValueError(f"N must be a positive multiple of 2, got {n}")
    if k <= 0 or k % tactic.cta_k or (k // tactic.cta_k) % tactic.split_k:
        raise ValueError(
            f"K={k} must be a positive multiple of split_k * cta_k = "
            f"{tactic.split_k} * {tactic.cta_k}"
        )
    smem_bytes = _smem_bytes(tactic, tactic.ab_stages)
    if smem_bytes > smem_capacity:
        raise ValueError(
            f"tactic {tactic} needs {smem_bytes} B of shared memory but only "
            f"{smem_capacity} B are available"
        )


#: Grid caps of the default tactic, as fractions of the SM count: clusters of 8
#: or 16 CTAs fit fewer co-resident CTAs than clusters of 2 or 4.
_DEFAULT_CTAS_PER_SM_LARGE_SPLIT = 0.65
_DEFAULT_CTAS_PER_SM_SMALL_SPLIT = 0.85


def default_tactic(
    m: int,
    n: int,
    k: int,
    *,
    mma_m: int | None = None,
    mma_n: int | None = None,
    split_k: int | None = None,
    cta_k: int | None = None,
    ab_stages: int | None = None,
    smem_capacity: int | None = None,
) -> Tf32SplitKTactic:
    """Fill in the fields left unset and validate the tactic.

    Candidates: ``mma_m`` in {64, 128}, ``mma_n`` the smallest tile covering M
    (at most 64) or half of it, ``split_k`` in {16, 8, 4, 2}, ``cta_k`` 64
    (32 for ``mma_m`` = 128, else whichever splits K). The default takes the
    grid with the most CTAs within a cap (0.65 x SMs for split 8/16, 0.85 x SMs
    for split 2/4), then prefers ``mma_m`` = 64 up to M = 256 (128 above), the
    larger split and the larger ``mma_n``; ``ab_stages`` is the deepest ring that fits.
    """
    device = _torch.cuda.current_device()
    if smem_capacity is None:
        smem_capacity = _smem_capacity_bytes(device)
    sm_count = _torch.cuda.get_device_properties(device).multi_processor_count
    cover = next((x for x in _SUPPORTED_MMA_N if x >= m), _SUPPORTED_MMA_N[-1])
    mma_ms = (mma_m,) if mma_m else _SUPPORTED_MMA_M
    mma_ns = (mma_n,) if mma_n else tuple({cover, max(cover // 2, 8)})
    splits = (split_k,) if split_k else _SUPPORTED_SPLIT_K

    best = fallback = None
    for mm, mn, sk in itertools.product(mma_ms, mma_ns, splits):
        if mm == 128 and mn % 16:
            continue
        ck = cta_k or next(
            (
                c
                for c in ((32, 64, 128) if mm == 128 else (64, 32, 128))
                if k % (c * sk) == 0
            ),
            None,
        )
        if ck is None or k % (ck * sk):
            continue
        tactic = Tf32SplitKTactic(mm, mn, sk, _MIN_AB_STAGES, ck)
        if not _max_ab_stages_for(tactic, smem_capacity, k):
            continue
        ctas = -(-n // mm) * -(-m // mn) * sk
        cap = sm_count * (
            _DEFAULT_CTAS_PER_SM_LARGE_SPLIT
            if sk >= 8
            else _DEFAULT_CTAS_PER_SM_SMALL_SPLIT
        )
        # Ties: 64-row tiles up to M = 256, 128-row tiles above (their epilogue
        # exchanges registers directly); then the larger split and mma_n.
        key = (ctas, mm == (64 if m <= 256 else 128), sk, mn)
        if ctas <= cap and (best is None or key > best[0]):
            best = (key, tactic)
        # Nothing fits the cap: the smallest grid.
        if fallback is None or (-ctas, key) > fallback[0]:
            fallback = ((-ctas, key), tactic)
    if best is None and fallback is None:
        raise ValueError(
            f"K={k} must be a positive multiple of split_k * cta_k (at least 64); "
            f"no tactic serves M={m}, N={n}"
        )
    tactic = (best or fallback)[1]
    max_stages = _max_ab_stages_for(tactic, smem_capacity, k)
    tactic = dataclasses.replace(
        tactic,
        ab_stages=min(ab_stages, max_stages) if ab_stages else max_stages,
    )
    validate_tactic(tactic, m, n, k, smem_capacity=smem_capacity)
    return tactic


def autotune_tactics(
    m: int,
    n: int,
    k: int,
    *,
    smem_capacity: int | None = None,
) -> list[Tf32SplitKTactic]:
    """Valid tactics for ``(m, n, k)``, default first, at the deepest ring.

    M tiles: the smallest covering M and half of it; the 128-row tile only for
    N >= 256; grids above two waves of clusters are dropped.
    """
    device = _torch.cuda.current_device()
    if smem_capacity is None:
        smem_capacity = _smem_capacity_bytes(device)
    cover = next((x for x in _SUPPORTED_MMA_N if x >= m), _SUPPORTED_MMA_N[-1])
    mma_ns = [x for x in (cover, cover // 2) if x in _SUPPORTED_MMA_N]
    mma_ms = [x for x in _SUPPORTED_MMA_M if x == 64 or n >= 256]
    tactics = []
    with contextlib.suppress(ValueError):
        tactics.append(default_tactic(m, n, k, smem_capacity=smem_capacity))
    for mma_m in mma_ms:
        for mma_n in mma_ns:
            tiles = -(-n // mma_m) * -(-m // mma_n)
            for cta_k in _SUPPORTED_CTA_K:
                for split_k in _SUPPORTED_SPLIT_K:
                    if tiles > 2 * _cluster_capacity(device, split_k):
                        continue
                    try:
                        tactics.append(
                            default_tactic(
                                m,
                                n,
                                k,
                                mma_m=mma_m,
                                mma_n=mma_n,
                                split_k=split_k,
                                cta_k=cta_k,
                                smem_capacity=smem_capacity,
                            )
                        )
                    except ValueError:
                        continue
    return list(dict.fromkeys(tactics))


class Tf32SplitKGemmKernel:
    """TF32 tcgen05 GEMM, K split over a cluster of ``split_k`` CTAs whose
    partial tiles are summed by a DSMEM reduce-scatter in the epilogue."""

    def __init__(self, tactic: Tf32SplitKTactic, use_pdl: bool, n_tail: bool):
        self.use_pdl = use_pdl
        # N is not a multiple of cta_m: predicate the stores of the last tile.
        self.n_tail = n_tail
        self.acc_dtype = Float32
        self.mma_tiler = (tactic.mma_m, tactic.mma_n, tactic.cta_k)
        self.cta_m, self.cta_n, self.cta_k = self.mma_tiler
        self.num_ab_stage = tactic.ab_stages
        # The stage mbarriers are initialized one per lane of a warp.
        assert self.num_ab_stage <= 32, f"ab_stages={self.num_ab_stage} > 32"
        self.split_k = tactic.split_k
        self.cta_group = tcgen05.CtaGroup.ONE

        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        # ``b`` does not depend on the previous kernel, ``a`` does (with PDL only
        # the ``a`` warp waits).
        self.b_load_warp_id = 5
        self.a_load_warp_id = 6
        # Warp 7 has no role: 8 warps keep the CTA at 256 threads.
        self.threads_per_cta = 32 * 8

        # Each rank owns 1 / split_k of the output tile; the peers' st.async
        # writes credit exactly its sSlots bytes on its reduce mbarrier.
        self.seg = tactic.mma_m * tactic.mma_n // tactic.split_k
        self.expected_transaction_bytes = (tactic.split_k - 1) * self.seg * _FP32_BYTES
        # Fragment-order exchange: rank g owns registers [g C, (g + 1) C) of every
        # epilogue thread (all ranks share the thread -> element map).
        self.direct_exchange = _direct_exchange(tactic)
        self.regs_per_rank = (
            tactic.mma_m * tactic.mma_n // _EPILOG_THREADS // self.split_k
        )

    @cute.jit
    def __call__(
        self,
        w: cute.Tensor,  # (N, K) = b.T, static
        x: cute.Tensor,  # (M, K) = a, M dynamic
        out: cute.Tensor,  # (M, N), M dynamic
        stream: _cuda.CUstream,
    ):
        # --------------------------------------------------------------
        # Kernel views
        # --------------------------------------------------------------
        # A unit batch mode: A (N, K, 1), B (M, K, 1), C (N, M, 1).
        unit = cute.make_layout(1)
        mA = cute.make_tensor(w.iterator, cute.append(w.layout, unit))
        mB = cute.make_tensor(x.iterator, cute.append(x.layout, unit))
        mC = cute.make_tensor(
            out.iterator, cute.append(cute.select(out.layout, mode=[1, 0]), unit)
        )

        # --------------------------------------------------------------
        # TiledMma creation
        # --------------------------------------------------------------
        # TF32 MMA on FP32 operands: Mma_M = cta_m, Mma_N = cta_n, Mma_K = 8.
        tiled_mma = utils.sm100.make_trivial_tiled_mma(
            cutlass.TFloat32,
            cutlass.TFloat32,
            LayoutEnum.from_tensor(mA).mma_major_mode(),
            LayoutEnum.from_tensor(mB).mma_major_mode(),
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        # The cluster runs along K only: the TMA and the pipelines see one CTA.
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )

        # --------------------------------------------------------------
        # Staged SMEM layouts (sA, sB)
        # --------------------------------------------------------------
        a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, cutlass.TFloat32, self.num_ab_stage
        )
        b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, cutlass.TFloat32, self.num_ab_stage
        )
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))

        # --------------------------------------------------------------
        # TMA atoms
        # --------------------------------------------------------------
        tma_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        # internal_type=TFloat32: the TMA rounds FP32 -> TF32 on load.
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            tma_op,
            mA,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            cta_layout_vmnk.shape,
            internal_type=cutlass.TFloat32,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            tma_op,
            mB,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            cta_layout_vmnk.shape,
            internal_type=cutlass.TFloat32,
        )
        self.a_load_bytes = cute.size_in_bytes(cutlass.TFloat32, a_smem_layout)
        self.b_load_bytes = cute.size_in_bytes(cutlass.TFloat32, b_smem_layout)

        # --------------------------------------------------------------
        # TMEM accumulator
        # --------------------------------------------------------------
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))
        self.num_tmem_alloc_cols = cutlass.memory.get_num_tmem_alloc_cols(tCtAcc_fake)

        # --------------------------------------------------------------
        # Launch: grid (N tiles, M tiles x split_k), cluster (1, split_k)
        # --------------------------------------------------------------
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            mC,
            cta_layout_vmnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
        ).launch(
            grid=(
                cute.ceil_div(cute.size(mC, mode=[0]), self.cta_m),
                cute.ceil_div(cute.size(mC, mode=[1]), self.cta_n) * self.split_k,
                1,
            ),
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, self.split_k, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,  # (N, K, 1)
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,  # (M, K, 1)
        mC_mnl: cute.Tensor,  # (N, M, 1)
        cta_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
    ):
        # --------------------------------------------------------------
        # Thread / warp ids
        # --------------------------------------------------------------
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        # Each prologue task below runs on its own warp (a compile-time index);
        # warp 0 initializes the accumulator pipeline's mbarriers in its create().
        init_warp = 1

        # --------------------------------------------------------------
        # TMA descriptor prefetch
        # --------------------------------------------------------------
        if warp_idx == init_warp:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
        init_warp += 1

        # --------------------------------------------------------------
        # Work tile: N tile, M tile, and this rank's K slice
        # --------------------------------------------------------------
        # Public N tile (kernel M, cta_m wide) and public M tile (kernel N, cta_n wide).
        n_tile_idx = bidx
        m_tile_idx = bidy // self.split_k
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        k_tile_count = cute.size(mA_mkl, mode=[1]) // self.cta_k // self.split_k
        k_tile_start = split_rank * k_tile_count

        # --------------------------------------------------------------
        # Shared storage: mbarriers and the TMEM pointer
        # --------------------------------------------------------------
        @cute.struct
        class SharedStorage:
            ab_mbar: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            reduce_mbar: cutlass.Int64
            tmem_holding_buf: Int32

        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # --------------------------------------------------------------
        # Pipelines and TMEM allocator
        # --------------------------------------------------------------
        # full[stage] = 2 producer arrivals (b, a) + their TMA bytes;
        # empty[stage] = 1 MMA tcgen05.commit.
        bar_full = storage.ab_mbar.data_ptr()
        bar_empty = bar_full + self.num_ab_stage
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.epilogue_warp_id)
            ),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(
                barrier_id=_TMEM_ALLOC_SYNC_BAR_ID,
                num_threads=32 * (len(self.epilogue_warp_id) + 1),
            ),
            allocator_warp_id=self.epilogue_warp_id[0],
        )
        mbar_reduce = storage.reduce_mbar.ptr

        # --------------------------------------------------------------
        # mbarrier init
        # --------------------------------------------------------------
        # A/B stage mbarriers: lane i initializes stage i (ab_stages <= 32).
        if warp_idx == init_warp:
            if lane_idx < self.num_ab_stage:
                cute.arch.mbarrier_init(bar_full + lane_idx, 2)
                cute.arch.mbarrier_init(bar_empty + lane_idx, 1)
        init_warp += 1
        # Reduce mbarrier: one local arrival (arrive_and_expect_tx) + the peers' bytes.
        if warp_idx == init_warp:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mbar_reduce, 1)
        init_warp += 1
        # pipeline_init_arrive includes mbarrier_init_fence (fences all the inits
        # above), then arrives on the cluster barrier; the wait below guarantees
        # every peer's mbar_reduce exists before any st.async credits it.
        pipeline_init_arrive(
            cluster_shape_mn=cute.make_layout((1, self.split_k)), is_relaxed=True
        )

        # --------------------------------------------------------------
        # SMEM tensors: sA, sB (stage rings), sStage, sSlots
        # --------------------------------------------------------------
        # (MMA, MMA_M, MMA_K, STAGE) / (MMA, MMA_N, MMA_K, STAGE)
        sA = smem.allocate_tensor(
            element_type=cutlass.TFloat32,
            layout=a_smem_layout_staged.outer,
            byte_alignment=_AB_BUFFER_ALIGN_BYTES,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=cutlass.TFloat32,
            layout=b_smem_layout_staged.outer,
            byte_alignment=_AB_BUFFER_ALIGN_BYTES,
            swizzle=b_smem_layout_staged.inner,
        )
        sStage = None
        if cutlass.const_expr(self.direct_exchange):
            # (4, thread, chunk, split_k - 1): 16 B chunk of each thread
            sSlots = smem.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout(
                    (4, _EPILOG_THREADS, self.regs_per_rank // 4, self.split_k - 1)
                ),
                byte_alignment=_MAILBOX_ALIGN_BYTES,
            )
        else:
            # (cta_m, cta_n): this rank's partial tile, XOR-swizzled layout.
            sStage = smem.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout(
                    (self.cta_m, self.cta_n), stride=(1, self.cta_m)
                ),
                byte_alignment=_MAILBOX_ALIGN_BYTES,
                swizzle=cute.make_swizzle(3, 4, self.cta_m.bit_length() - 3),
            )
            # (seg, split_k - 1): the peers' partials of this rank's segment.
            sSlots = smem.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout((self.seg, self.split_k - 1)),
                byte_alignment=_MAILBOX_ALIGN_BYTES,
            )

        # --------------------------------------------------------------
        # Global tiles, TMA partitions and MMA fragments
        # --------------------------------------------------------------
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))

        # --------------------------------------------------------------
        # Cluster sync: every peer's mbarriers are initialized
        # --------------------------------------------------------------
        pipeline_init_wait(cluster_shape_mn=cute.make_layout((1, self.split_k)))

        # --------------------------------------------------------------
        # Load warps: TMA b -> sA (warp 5), a -> sB (warp 6)
        # --------------------------------------------------------------
        if warp_idx == self.b_load_warp_id:
            self.load(
                tma_atom_a,
                tAgA[(None, n_tile_idx, None, 0)],
                tAsA,
                bar_full,
                bar_empty,
                self.a_load_bytes,
                k_tile_start,
                k_tile_count,
                True,
            )
        elif warp_idx == self.a_load_warp_id:
            self.load(
                tma_atom_b,
                tBgB[(None, m_tile_idx, None, 0)],
                tBsB,
                bar_full,
                bar_empty,
                self.b_load_bytes,
                k_tile_start,
                k_tile_count,
                False,
            )

        # --------------------------------------------------------------
        # MMA warp: tcgen05.mma into TMEM, release each stage
        # --------------------------------------------------------------
        elif warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            _tcgen05_fence(before=False)
            tCtAcc = cute.make_tensor(
                tmem.retrieve_ptr(self.acc_dtype), tCtAcc_fake.layout
            )[(None, None, None, 0)]
            acc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )
            acc_pipeline.producer_acquire(acc_state)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            full_phase = Int32(0)
            for k_tile in cutlass.range(k_tile_count, unroll=1):
                stage = k_tile % self.num_ab_stage
                cute.arch.mbarrier_wait(bar_full + stage, full_phase)
                _tcgen05_fence(before=False)
                for kblk in cutlass.range(cute.size(tCrA, mode=[2]), unroll_full=True):
                    kblk_crd = (None, None, kblk, stage)
                    cute.gemm(tiled_mma, tCtAcc, tCrA[kblk_crd], tCrB[kblk_crd], tCtAcc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                # Release the stage once these MMAs have read it.
                with cute.arch.elect_one():
                    tcgen05.commit(bar_empty + stage, None, self.cta_group)
                if stage == self.num_ab_stage - 1:
                    full_phase = full_phase ^ 1
            acc_pipeline.producer_commit(acc_state)
            acc_state.advance()
            acc_pipeline.producer_tail(acc_state)

        # --------------------------------------------------------------
        # Epilogue warps: TMEM -> registers, split-K reduce-scatter
        # --------------------------------------------------------------
        elif warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            # Order the alloc before the pointer hand-off below.
            _tcgen05_fence(before=True)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)[
                ((None, None), 0, 0, 0)
            ]
            epi_tile = (self.cta_m, self.cta_n)
            copy_atom_t2r = utils.sm100.get_tmem_load_op(
                self.mma_tiler,
                LayoutEnum.from_tensor(mC_mnl),
                Float32,
                self.acc_dtype,
                epi_tile,
                False,
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc)
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc = thr_copy_t2r.partition_S(tAcc)
            # (n, m) coordinate of every fragment register
            tTR_cC = thr_copy_t2r.partition_D(cute.make_identity_tensor(epi_tile))
            tTR_rAcc = cute.make_rmem_tensor(tTR_cC.shape, self.acc_dtype)

            acc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1
            )
            acc_pipeline.consumer_wait(acc_state)
            _tcgen05_fence(before=False)
            cute.copy(tiled_copy_t2r, tTR_tAcc, tTR_rAcc)
            # tcgen05.ld is async: make the registers visible, then free TMEM.
            cute.arch.fence_view_async_tmem_load()
            _tcgen05_fence(before=True)
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_state)
            tmem.relinquish_alloc_permit()
            if cutlass.const_expr(self.direct_exchange):
                self.reduce_direct(
                    tmem,
                    tmem_ptr,
                    tTR_rAcc,
                    tTR_cC,
                    mC_mnl,
                    sSlots,
                    mbar_reduce,
                    tidx,
                    n_tile_idx,
                    m_tile_idx,
                    split_rank,
                )
            else:
                self.reduce_scatter(
                    tmem,
                    tmem_ptr,
                    tTR_rAcc,
                    tTR_cC,
                    mC_mnl,
                    sStage,
                    sSlots,
                    mbar_reduce,
                    tidx,
                    n_tile_idx,
                    m_tile_idx,
                    split_rank,
                )

    @cute.jit
    def load(
        self,
        tma_atom: cute.CopyAtom,
        tXgX: cute.Tensor,  # ((atom_v, rest_v), RestK) of this CTA's tile
        tXsX: cute.Tensor,  # ((atom_v, rest_v), STAGE)
        bar_full: cute.Pointer,
        bar_empty: cute.Pointer,
        load_bytes: cutlass.Constexpr,
        k_tile_start: Int32,
        k_tile_count: cutlass.Constexpr,
        is_b: cutlass.Constexpr,
    ):
        """One TMA producer: this rank's K tiles of one operand into the stages."""
        stages = self.num_ab_stage
        if cutlass.const_expr(self.use_pdl and not is_b):
            # ``a`` comes from the previous kernel.
            cute.arch.griddepcontrol_wait()
        # A fresh mbarrier is in phase 0, so waiting on phase 1 passes at once:
        # the first ``stages`` tiles go out without waiting for the MMA warp.
        empty_phase = Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bar_full + stage, load_bytes)
            cute.copy(
                tma_atom,
                tXgX[(None, k_tile_start + k_tile)],
                tXsX[(None, stage)],
                tma_bar_ptr=bar_full + stage,
            )
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1
        if cutlass.const_expr(self.use_pdl and is_b):
            cute.arch.griddepcontrol_launch_dependents()
        # Keep the CTA alive until the MMA warp has released every stage.
        for tail in cutlass.range(stages, unroll=1):
            stage = (tail + k_tile_count) % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

    @cute.jit
    def reduce_scatter(
        self,
        tmem: cutlass.memory.TmemAllocator,
        tmem_ptr: cute.Pointer,
        tTR_rAcc: cute.Tensor,  # FP32 fragment
        tTR_cC: cute.Tensor,  # (n, m) coordinates of the fragment
        mC_mnl: cute.Tensor,  # (N, M, 1)
        sStage: cute.Tensor,  # (cta_m, cta_n) swizzled: this rank's partial tile
        sSlots: cute.Tensor,  # (seg, split_k - 1): peers' partials of my segment
        mbar_reduce: cute.Pointer,
        epi_tid: Int32,
        n_tile_idx: Int32,
        m_tile_idx: Int32,
        split_rank: Int32,
    ):
        RW = self.cta_m  # outputs per row of the tile
        SEG = self.seg
        S = self.split_k
        # 1. Fragment -> sStage (st.shared.b32).
        for r in cutlass.range_constexpr(cute.size(tTR_rAcc)):
            sStage[tTR_cC[r]] = tTR_rAcc[r]
        pipeline.NamedBarrier(
            barrier_id=_EPILOG_SYNC_BAR_ID, num_threads=_EPILOG_THREADS
        ).arrive_and_wait()
        # Every epilogue warp's tcgen05.ld has completed: free the accumulator.
        _tcgen05_fence(before=False)
        tmem.free(tmem_ptr)

        # 2. Send the other ranks' segments to their DSMEM (st.async.v4).
        if epi_tid == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(
                mbar_reduce, self.expected_transaction_bytes
            )
        thr_m4 = RW // 4
        send_copy = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
            ),
            cute.make_layout((thr_m4, _EPILOG_THREADS // thr_m4), stride=(1, thr_m4)),
            cute.make_layout((4, 1)),
        )
        thr_send = send_copy.get_slice(epi_tid)
        # ((4,1), 1, #CHUNK)
        tSsStage = thr_send.partition_S(sStage)
        tScStage = thr_send.partition_S(
            cute.make_identity_tensor((self.cta_m, self.cta_n))
        )
        tRsend = cute.make_rmem_tensor(tSsStage.shape, Float32)
        for i in cutlass.range_constexpr(cute.size(tSsStage, mode=[2])):
            crd = tScStage[(0, 0), 0, i]
            flat = crd[1] * RW + crd[0]
            owner = flat // SEG
            if owner != split_rank:
                cute.copy(send_copy, tSsStage[None, None, i], tRsend[None, None, i])
                # The owner's S-1 slots hold the other ranks in rank order:
                # sender r < owner uses slot r, sender r > owner slot r - 1.
                slot = split_rank - 1
                if split_rank < owner:
                    slot = split_rank
                cute.arch.store_async_dsmem(
                    sSlots[(None, slot)].iterator + flat % SEG,
                    tuple(tRsend[(v, 0), 0, i].bitcast(Int32) for v in range(4)),
                    mbar_reduce,
                    owner,
                )
        # 3. Reduce results: this rank's (seg_m, seg_n) block of the tile.
        seg_m = min(SEG, RW)
        seg_n = SEG // seg_m
        seg_crd = (split_rank % (RW // seg_m), split_rank // (RW // seg_m))
        thr_m = seg_m // 2
        n_thr = min(_EPILOG_THREADS, SEG // 2)
        tiled_copy = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=64
            ),
            cute.make_layout((thr_m, n_thr // thr_m), stride=(1, thr_m)),
            cute.make_layout((2, 1)),
        )
        thr_copy = tiled_copy.get_slice(epi_tid)
        seg_tile = (seg_m, seg_n)
        # (CPY, 1, ITERS) partitions of the segment views
        tSsOwn = thr_copy.partition_S(cute.local_tile(sStage, seg_tile, seg_crd))
        tRacc = cute.make_rmem_tensor(tSsOwn.shape, Float32)
        tRtmp = cute.make_rmem_tensor(tSsOwn.shape, Float32)
        # The own partial does not depend on the peers: read it before the wait.
        if epi_tid < n_thr:
            cute.copy(tiled_copy, tSsOwn, tRacc)
        rows_left = cute.size(mC_mnl, mode=[1]) - m_tile_idx * self.cta_n
        cols_left = cute.size(mC_mnl, mode=[0]) - n_tile_idx * self.cta_m
        gC = cute.local_tile(
            mC_mnl, (self.cta_m, self.cta_n), (n_tile_idx, m_tile_idx, 0)
        )
        tDgC = thr_copy.partition_D(cute.local_tile(gC, seg_tile, seg_crd))
        # (n, m) coordinates of each pair, for the row (and N tail) guards.
        tDcC = thr_copy.partition_D(
            cute.local_tile(
                cute.make_identity_tensor((self.cta_m, self.cta_n)), seg_tile, seg_crd
            )
        )

        # The peers' st.async complete-tx release at cluster scope.
        _mbarrier_wait_acquire_cluster(mbar_reduce, Int32(0))

        if epi_tid < n_thr:
            for p in cutlass.range_constexpr(S - 1):
                sSlot = cute.make_tensor(
                    sSlots[(None, p)].iterator, cute.make_layout(seg_tile)
                )
                cute.copy(tiled_copy, thr_copy.partition_S(sSlot), tRtmp)
                tRacc.store(tRacc.load() + tRtmp.load())
            tDpC = cute.make_rmem_tensor(
                cute.make_layout(
                    (1, cute.size(tDcC, mode=[1]), cute.size(tDcC, mode=[2])),
                    stride=(0, cute.size(tDcC, mode=[2]), 1),
                ),
                cutlass.Boolean,
            )
            for i in cutlass.range_constexpr(cute.size(tDcC, mode=[1])):
                for j in cutlass.range_constexpr(cute.size(tDcC, mode=[2])):
                    crd = tDcC[0, i, j]
                    if cutlass.const_expr(self.n_tail):
                        tDpC[0, i, j] = cute.elem_less(crd, (cols_left, rows_left))
                    else:
                        tDpC[0, i, j] = crd[1] < rows_left
            cute.copy(tiled_copy, tRacc, tDgC, pred=tDpC)

    @cute.jit
    def reduce_direct(
        self,
        tmem: cutlass.memory.TmemAllocator,
        tmem_ptr: cute.Pointer,
        tTR_rAcc: cute.Tensor,  # FP32 fragment
        tTR_cC: cute.Tensor,  # (n, m) coordinates of the fragment
        mC_mnl: cute.Tensor,  # (N, M, 1)
        sSlots: cute.Tensor,  # (4, thread, chunk, split_k - 1)
        mbar_reduce: cute.Pointer,
        epi_tid: Int32,
        n_tile_idx: Int32,
        m_tile_idx: Int32,
        split_rank: Int32,
    ):
        S = self.split_k
        C = self.regs_per_rank
        pipeline.NamedBarrier(
            barrier_id=_EPILOG_SYNC_BAR_ID, num_threads=_EPILOG_THREADS
        ).arrive_and_wait()
        # Every epilogue warp's tcgen05.ld has completed: free the accumulator.
        _tcgen05_fence(before=False)
        tmem.free(tmem_ptr)

        # 1. Send registers to rank p's DSMEM (st.async.v4).
        if epi_tid == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(
                mbar_reduce, self.expected_transaction_bytes
            )
        for p in cutlass.range_constexpr(S):
            if split_rank != p:
                # Rank p's S-1 slots hold the other ranks in rank order:
                # sender r < p uses slot r, sender r > p slot r - 1.
                slot = split_rank - 1
                if split_rank < p:
                    slot = split_rank
                for c in cutlass.range_constexpr(C // 4):
                    cute.arch.store_async_dsmem(
                        sSlots[(None, epi_tid, c, slot)].iterator,
                        tuple(
                            tTR_rAcc[p * C + 4 * c + v].bitcast(Int32) for v in range(4)
                        ),
                        mbar_reduce,
                        p,
                    )

        # 2. Own registers. Register indices must be static, so the
        #    runtime rank selects one of S static copies (warp-uniform branch).
        # (C/4, 4):(4, 1): row c is one 16 B chunk of this rank's range.
        acc = cute.make_rmem_tensor(
            cute.make_layout((C // 4, 4), stride=(4, 1)), Float32
        )
        for g in cutlass.range_constexpr(S):
            if split_rank == g:
                for i in cutlass.range_constexpr(C):
                    acc[i // 4, i % 4] = tTR_rAcc[g * C + i]
        rows_left = cute.size(mC_mnl, mode=[1]) - m_tile_idx * self.cta_n
        cols_left = cute.size(mC_mnl, mode=[0]) - n_tile_idx * self.cta_m
        gC = cute.local_tile(
            mC_mnl, (self.cta_m, self.cta_n), (n_tile_idx, m_tile_idx, 0)
        )

        # The peers' st.async complete-tx release at cluster scope.
        _mbarrier_wait_acquire_cluster(mbar_reduce, Int32(0))

        # 3. Reduce results.
        r4 = cute.make_rmem_tensor((4,), Float32)
        for p in cutlass.range_constexpr(S - 1):
            for c in cutlass.range_constexpr(C // 4):
                cute.autovec_copy(sSlots[(None, epi_tid, c, p)], r4)
                acc[c, None].store(acc[c, None].load() + r4.load())
        for i in cutlass.range_constexpr(C):
            # Coordinates are arithmetic, so a runtime index is fine here.
            crd = tTR_cC[split_rank * C + i]
            if cutlass.const_expr(self.n_tail):
                if cute.elem_less(crd, (cols_left, rows_left)):
                    gC[crd] = acc[i // 4, i % 4]
            elif crd[1] < rows_left:
                gC[crd] = acc[i // 4, i % 4]


def _cute_operands(x: _torch.Tensor, w: _torch.Tensor, out: _torch.Tensor):
    # w is fully static; x and out have a dynamic row count M.
    return (
        from_dlpack(w, assumed_align=16),
        from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=x.dim_order()
        ),
        from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=out.dim_order()
        ),
    )


@functools.cache
def _get_compiled_kernel(
    device_index: int, tactic: Tf32SplitKTactic, n: int, k: int, use_pdl: bool
):
    # One binary per tactic and (N, K); it serves every M.
    with _torch.cuda.device(device_index):
        x = _torch.empty((tactic.mma_n, k), device="cuda", dtype=_torch.float32)
        w = _torch.empty((n, k), device="cuda", dtype=_torch.float32)
        out = _torch.empty((tactic.mma_n, n), device="cuda", dtype=_torch.float32)
        return cute.compile(
            Tf32SplitKGemmKernel(tactic, use_pdl, n_tail=n % tactic.mma_m != 0),
            *_cute_operands(x, w, out),
            _cuda.CUstream(_torch.cuda.current_stream().cuda_stream),
        )


def run_tf32_splitk_dense(a, b, out, pdl: bool, tactic: Tf32SplitKTactic):
    """Run ``out = a[M, K] @ b[K, N]`` (TF32 MMA, FP32 accumulate) with ``tactic``.

    With ``pdl`` the ``b`` loads may start before the previous kernel finishes.
    """
    m, n, k = validate_inputs(a, b, out)
    validate_tactic(tactic, m, n, k, smem_capacity=_smem_capacity_bytes(a.get_device()))
    with _torch.cuda.device(a.device):
        compiled = _get_compiled_kernel(a.get_device(), tactic, n, k, bool(pdl))
        compiled(
            *_cute_operands(a, b.t(), out),
            _cuda.CUstream(_torch.cuda.current_stream(a.device).cuda_stream),
        )
    return out


__all__ = [
    "Tf32SplitKTactic",
    "autotune_tactics",
    "default_tactic",
    "run_tf32_splitk_dense",
    "validate_inputs",
    "validate_tactic",
]
