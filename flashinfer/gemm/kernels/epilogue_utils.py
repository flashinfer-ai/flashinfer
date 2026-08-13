# Copyright (c) 2025 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
#
# This file contains modified versions of epilogue utility functions from
# NVIDIA CUTLASS DSL, adapted to support optimized output scaling.
#
# The key optimization is applying the output_scale to the accumulator
# BEFORE converting to the output dtype, which:
# 1. Avoids an extra type conversion (BFloat16 -> Float32 promotion)
# 2. Preserves precision by scaling in Float32 before the final conversion

"""
Epilogue Utilities with Optimized Output Scaling
=================================================

Location: flashinfer/gemm/kernels/epilogue_utils.py

Custom epilogue utility functions with optimized output scaling support.

These functions are based on cutlass.utils.gemm.sm100 but modified to accept
an optional output_scale parameter that is applied efficiently before type
conversion.

Key functions:
- epilogue_tma_store_scaled(): Epilogue with TMA store + scaling
- epilogue_scaled(): Epilogue with direct store + scaling
"""

from typing import Optional, Tuple, Union

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Boolean, Constexpr, Int32, const_expr
import cutlass.pipeline as pipeline
from cutlass.utils.static_persistent_tile_scheduler import StaticPersistentTileScheduler
from cutlass.utils.dynamic_persistent_tile_scheduler import (
    ClcDynamicPersistentTileScheduler,
)
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.common import CacheEvictionPriority

# Re-export helper functions from the original module
from cutlass.utils.gemm.sm100 import (
    transform_partitioned_tensor_layout,
    epilogue_tmem_copy_and_partition,
    epilogue_smem_copy_and_partition,
)

__all__ = [
    "epilogue_tma_store_scaled",
    "epilogue_scaled",
    "epilogue_tma_store_with_alpha",
    "epilogue_with_alpha",
]


@cute.jit
def epilogue_tma_store_scaled(
    gemm_kernel,
    epi_tidx: Int32,
    warp_idx: Int32,
    acc_pipeline: pipeline.PipelineAsync,
    tiled_mma: cute.TiledMma,
    tma_atom_c: cute.CopyAtom,
    # Input of epilogue
    tCtAcc_base: cute.Tensor,
    # Staging of epilogue
    sC: cute.Tensor,
    # Output of epilogue
    tCgC_base: cute.Tensor,
    epi_tile: cute.Tile,
    tile_sched: Union[StaticPersistentTileScheduler, ClcDynamicPersistentTileScheduler],
    epilogue_op: Constexpr,
    output_scale: Optional[cutlass.Float32] = None,
    clc_pipeline: Union[pipeline.PipelineClcFetchAsync, None] = None,
    clc_consumer_state: Union[pipeline.PipelineState, None] = None,
) -> None:
    """
    Epilogue function with TMA store and optimized output scaling.

    This is a modified version of cutlass.utils.gemm.sm100.epilogue_tma_store
    that accepts an optional output_scale parameter. When provided, the scale
    is applied to the accumulator BEFORE converting to the output dtype,
    which is more efficient and preserves precision.

    :param gemm_kernel: The kernel instance
    :param epi_tidx: Thread index in epilogue warp groups
    :param warp_idx: Warp index
    :param acc_pipeline: Accumulator pipeline for async operations
    :param tiled_mma: The tiled MMA configuration
    :param tma_atom_c: TMA copy atom for output tensor
    :param tCtAcc_base: Base accumulator tensor in tensor memory
    :param sC: Shared memory tensor for staging
    :param tCgC_base: Global memory output tensor
    :param epi_tile: Epilogue tile configuration
    :param tile_sched: Tile scheduler for persistent scheduling
    :param epilogue_op: Optional elementwise operation to apply
    :param output_scale: Optional scalar to multiply the accumulator by (applied in acc_dtype)
    :param clc_pipeline: Pipeline for dynamic persistent tile scheduling
    :param clc_consumer_state: Consumer state for dynamic persistent tile scheduling
    """
    # Layout transformation for tCgC_base
    # ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, TILE_M, TILE_N, TILE_K)
    # -> ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), TILE_M, TILE_N, TILE_K)
    tCgC = transform_partitioned_tensor_layout(tCgC_base)

    # Layout transformation for tCtAcc_base
    # ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, STAGE)
    # -> ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), STAGE)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)

    tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = epilogue_tmem_copy_and_partition(
        gemm_kernel, epi_tidx, tCtAcc, tCgC, epi_tile, gemm_kernel.use_2cta_instrs
    )

    tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, gemm_kernel.c_dtype)
    tiled_copy_r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(
        gemm_kernel, tiled_copy_t2r, tTR_rC, epi_tidx, sC
    )

    # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
    tCgC_epi = cute.flat_divide(tCgC, epi_tile)
    # ((ATOM_V, REST_V), EPI_M, EPI_N)
    # ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, RestL)
    bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
        tma_atom_c,
        0,
        cute.make_layout(1),
        cute.group_modes(sC, 0, 2),
        cute.group_modes(tCgC_epi, 0, 2),
    )

    acc_consumer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, gemm_kernel.num_acc_stage
    )

    # Threads/warps participating in tma store pipeline
    c_producer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        32 * len(gemm_kernel.epilogue_warp_id),
    )
    c_pipeline = pipeline.PipelineTmaStore.create(
        num_stages=gemm_kernel.num_c_stage, producer_group=c_producer_group
    )

    epilog_sync_barrier = pipeline.NamedBarrier(
        barrier_id=gemm_kernel.epilog_sync_bar_id,
        num_threads=32 * len(gemm_kernel.epilogue_warp_id),
    )

    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        # Get tile coord from tile scheduler
        cur_tile_coord = work_tile.tile_idx
        mma_tile_coord_mnl = (
            cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
            cur_tile_coord[1],
            cur_tile_coord[2],
        )

        #
        # Slice to per mma tile index
        #
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]

        # Set tensor memory buffer for current tile
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
        tTR_tAcc = tTR_tAcc_base[
            (None, None, None, None, None, acc_consumer_state.index)
        ]

        #
        # Wait for accumulator buffer full
        #
        acc_pipeline.consumer_wait(acc_consumer_state)

        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

        #
        # Store accumulator to global memory in subtiles
        #
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
        for subtile_idx in range(subtile_cnt):
            #
            # Load accumulator from tensor memory buffer to register
            #
            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

            #
            # Apply output scale (if provided) in acc_dtype, then convert to C type
            # This is more efficient than converting first, then scaling
            #
            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
            if output_scale is not None:
                # Scale while still in acc_dtype (Float32) for precision
                acc_vec = acc_vec * output_scale
            # Now convert to c_dtype and apply epilogue_op
            acc_vec = epilogue_op(acc_vec.to(gemm_kernel.c_dtype))
            tRS_rC.store(acc_vec)

            #
            # Store C to shared memory
            #
            c_buffer = (num_prev_subtiles + subtile_idx) % gemm_kernel.num_c_stage
            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
            # Fence and barrier to make sure shared memory store is visible to TMA store
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            epilog_sync_barrier.arrive_and_wait()

            #
            # TMA store C to global memory
            #
            if warp_idx == gemm_kernel.epilogue_warp_id[0]:
                cute.copy(
                    tma_atom_c,
                    bSG_sC[(None, c_buffer)],
                    bSG_gC[(None, subtile_idx)],
                )
                # Fence and barrier to make sure shared memory store is visible to TMA store
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()
            epilog_sync_barrier.arrive_and_wait()

        epilog_sync_barrier.arrive_and_wait()

        #
        # Async arrive accumulator buffer empty
        #
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_consumer_state)
        acc_consumer_state.advance()

        #
        # Advance to next tile
        #
        # Check if tile_sched is StaticPersistentTileScheduler or any subclass inheriting from it
        if const_expr(isinstance(tile_sched, StaticPersistentTileScheduler)):
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
        elif const_expr(isinstance(tile_sched, ClcDynamicPersistentTileScheduler)):
            clc_pipeline.consumer_wait(clc_consumer_state)
            work_tile = tile_sched.get_current_work()
            clc_pipeline.consumer_release(clc_consumer_state)
            clc_consumer_state.advance()
        else:
            # Not match
            pass

    # Wait for C store complete
    c_pipeline.producer_tail()


@cute.jit
def epilogue_scaled(
    gemm_kernel,
    epi_tidx: Int32,
    acc_pipeline: pipeline.PipelineAsync,
    tiled_mma: cute.TiledMma,
    tCtAcc_base: cute.Tensor,
    tCgC_base: cute.Tensor,
    epi_tile: cute.Tile,
    tile_sched: Union[StaticPersistentTileScheduler, ClcDynamicPersistentTileScheduler],
    epilogue_op: Constexpr,
    tmem_dealloc_barrier: pipeline.NamedBarrier,
    output_scale: Optional[cutlass.Float32] = None,
    tCcC_base: cute.Tensor = None,
    mC_mnl: cute.Tensor = None,
    clc_pipeline: Union[pipeline.PipelineClcFetchAsync, None] = None,
    clc_consumer_state: Union[pipeline.PipelineState, None] = None,
) -> None:
    """
    Epilogue function that stores accumulator results directly to global memory
    with optimized output scaling.

    This is a modified version of cutlass.utils.gemm.sm100.epilogue that accepts
    an optional output_scale parameter. When provided, the scale is applied to
    the accumulator BEFORE converting to the output dtype, which is more efficient
    and preserves precision.

    :param gemm_kernel: The kernel instance
    :param epi_tidx: Thread index in epilogue warp groups
    :param acc_pipeline: Accumulator pipeline for async operations
    :param tiled_mma: The tiled MMA configuration
    :param tCtAcc_base: Base accumulator tensor in tensor memory
    :param tCgC_base: The global memory tensor C to be copied and partitioned
    :param epi_tile: Epilogue tile configuration
    :param tile_sched: Tile scheduler for persistent scheduling
    :param epilogue_op: Optional elementwise operation to apply
    :param tmem_dealloc_barrier: Barrier for tensor memory deallocation
    :param output_scale: Optional scalar to multiply the accumulator by (applied in acc_dtype)
    :param tCcC_base: Identity/coordinate tensor C
    :param mC_mnl: Global memory tensor C (full tensor for predicate computation)
    :param clc_pipeline: Pipeline for dynamic persistent tile scheduling
    :param clc_consumer_state: Consumer state for dynamic persistent tile scheduling
    """

    # Layout transformation for tCgC_base
    # ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, TILE_M, TILE_N, TILE_K)
    # -> ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), TILE_M, TILE_N, TILE_K)
    tCgC = transform_partitioned_tensor_layout(tCgC_base)

    # Layout transformation for tCtAcc_base
    # ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, STAGE)
    # -> ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), STAGE)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)

    #
    # Partition for epilogue
    #
    (
        tiled_copy_t2r,
        tTR_tAcc_base,
        tTR_rAcc,
    ) = epilogue_tmem_copy_and_partition(
        gemm_kernel, epi_tidx, tCtAcc, tCgC, epi_tile, gemm_kernel.use_2cta_instrs
    )

    gC_epi = cute.flat_divide(tCgC, epi_tile)
    # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
    thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
    tTR_gC_partitioned = thr_copy_t2r.partition_D(gC_epi)
    # (T2R, T2R_M, T2R_N)
    tTR_rC = cute.make_rmem_tensor(
        tTR_gC_partitioned[(None, None, None, 0, 0, 0, 0, 0)].shape, gemm_kernel.c_dtype
    )

    mclD = cute.max_common_layout(
        tTR_rC.layout, tTR_gC_partitioned[(None, None, None, 0, 0, 0, 0, 0)].layout
    )
    num_bits_per_copy = min(
        tTR_gC_partitioned.iterator.alignment * 8,
        cute.size(mclD) * gemm_kernel.c_dtype.width,
        256,
    )

    # Cache policy selection for epilogue store:
    # - Use NO_ALLOCATE since this data is never reused after the store
    simt_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        gemm_kernel.c_dtype,
        num_bits_per_copy=num_bits_per_copy,
        l1c_evict_priority=CacheEvictionPriority.NO_ALLOCATE,
    )
    use_predication = tCcC_base is not None and mC_mnl is not None

    if const_expr(use_predication):
        # Layout transformation for tCcC_base
        # ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, TILE_M, TILE_N, TILE_K)
        # -> ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), TILE_M, TILE_N, TILE_K)
        tCcC = transform_partitioned_tensor_layout(tCcC_base)
        cC_epi = cute.flat_divide(tCcC, epi_tile)
        tTR_cC_partitioned = thr_copy_t2r.partition_D(cC_epi)

    acc_consumer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, gemm_kernel.num_acc_stage
    )

    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        #
        # Pre-advance to next tile
        #
        if const_expr(isinstance(tile_sched, StaticPersistentTileScheduler)):
            tile_sched.advance_to_next_work()
            next_work_tile = tile_sched.get_current_work()

        # Get tile coord from current work tile
        cur_tile_coord = work_tile.tile_idx
        mma_tile_coord_mnl = (
            cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
            cur_tile_coord[1],
            cur_tile_coord[2],
        )

        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
        tTR_gC = tTR_gC_partitioned[
            (
                None,
                None,
                None,
                None,
                None,
                *mma_tile_coord_mnl,
            )
        ]
        if const_expr(use_predication):
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
            tTR_cC = tTR_cC_partitioned[
                (
                    None,
                    None,
                    None,
                    None,
                    None,
                    *mma_tile_coord_mnl,
                )
            ]
            tTR_cC = cute.group_modes(tTR_cC, 3, cute.rank(tTR_cC))

        # Set tensor memory buffer for current tile
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
        tTR_tAcc = tTR_tAcc_base[
            (None, None, None, None, None, acc_consumer_state.index)
        ]

        #
        # Wait for accumulator buffer full
        #
        acc_pipeline.consumer_wait(acc_consumer_state)

        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))
        #
        # Store accumulator to global memory in subtiles
        #
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        for subtile_idx in range(subtile_cnt):
            #
            # Get the destination and coordinate slices for this subtile
            #
            tTR_gC_subtile = tTR_gC[(None, None, None, subtile_idx)]
            #
            # Load accumulator from tensor memory buffer to register
            #
            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
            # Async arrive accumulator buffer empty
            # Release early for perf
            if subtile_idx == subtile_cnt - 1:
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

            #
            # Apply output scale (if provided) in acc_dtype, then convert to C type
            # This is more efficient than converting first, then scaling
            #
            acc_vec = tTR_rAcc.load()
            if output_scale is not None:
                # Scale while still in acc_dtype (Float32) for precision
                acc_vec = acc_vec * output_scale
            # Now convert to c_dtype and apply epilogue_op
            acc_vec = epilogue_op(acc_vec.to(gemm_kernel.c_dtype))
            tTR_rC.store(acc_vec)

            if const_expr(use_predication):
                # compute predicate
                tTR_cC_subtile = tTR_cC[(None, None, None, subtile_idx)]
                pred_C_shape = (1, *tTR_cC_subtile.shape[1:])
                pred_C = cute.make_rmem_tensor(pred_C_shape, Boolean)
                for m_idx in range(tTR_cC_subtile.shape[1]):
                    for n_idx in range(tTR_cC_subtile.shape[2]):
                        vector_first_coord = tTR_cC_subtile[(0, m_idx, n_idx)]
                        pred_C[(0, m_idx, n_idx)] = cute.elem_less(
                            vector_first_coord, mC_mnl.shape
                        )
                # Store C to global memory with predication
                cute.copy(simt_atom, tTR_rC, tTR_gC_subtile, pred=pred_C)
            else:
                # Store C directly to global memory
                cute.copy(simt_atom, tTR_rC, tTR_gC_subtile)

        if const_expr(isinstance(tile_sched, StaticPersistentTileScheduler)):
            work_tile = next_work_tile
        elif const_expr(isinstance(tile_sched, ClcDynamicPersistentTileScheduler)):
            clc_pipeline.consumer_wait(clc_consumer_state)
            work_tile = tile_sched.get_current_work()
            clc_pipeline.consumer_release(clc_consumer_state)
            clc_consumer_state.advance()

    # Synchronize before TMEM dealloc (done by the caller)
    tmem_dealloc_barrier.arrive_and_wait()


@cute.jit
def epilogue_tma_store_with_alpha(
    gemm_kernel,
    epi_tidx: Int32,
    warp_idx: Int32,
    tma_atom_c: cute.CopyAtom,
    # Input of epilogue
    tCtAcc_base: cute.Tensor,
    # Staging of epilogue
    sC: cute.Tensor,
    # Output of epilogue
    tCgC_base: cute.Tensor,
    epi_tile: cute.Tile,
    num_tiles_executed: Int32,
    epilogue_op: Constexpr,
    alpha_value: cutlass.Float32,
    mma_tile_coord_mnl: Tuple[Int32, Int32, Int32],
    acc_consumer_state: pipeline.PipelineState,
    acc_pipeline: pipeline.PipelineAsync,
    c_pipeline: pipeline.PipelineTmaStore,
) -> pipeline.PipelineState:
    """
    Per-tile epilogue with TMA store that applies alpha scaling in Float32 BEFORE
    converting to c_dtype, preventing overflow for narrow types like Float16.

    This is a drop-in replacement for cutlass.utils.gemm.sm100.epilogue_tma_store
    with an additional alpha_value parameter. The alpha is applied to the Float32
    accumulator before the conversion to c_dtype, so large accumulator values (e.g.
    from block-scaled FP4 GEMM) do not overflow Float16.
    """
    tCgC = transform_partitioned_tensor_layout(tCgC_base)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)

    tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = epilogue_tmem_copy_and_partition(
        gemm_kernel, epi_tidx, tCtAcc, tCgC, epi_tile, gemm_kernel.use_2cta_instrs
    )

    tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, gemm_kernel.c_dtype)
    tiled_copy_r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(
        gemm_kernel, tiled_copy_t2r, tTR_rC, epi_tidx, sC
    )

    tCgC_epi = cute.flat_divide(tCgC, epi_tile)
    bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
        tma_atom_c,
        0,
        cute.make_layout(1),
        cute.group_modes(sC, 0, 2),
        cute.group_modes(tCgC_epi, 0, 2),
    )

    epilog_sync_barrier = pipeline.NamedBarrier(
        barrier_id=gemm_kernel.epilog_sync_bar_id,
        num_threads=32 * len(gemm_kernel.epilogue_warp_id),
    )

    bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]

    acc_pipeline.consumer_wait(acc_consumer_state)

    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
    num_prev_subtiles = num_tiles_executed * subtile_cnt
    for subtile_idx in range(subtile_cnt):
        tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

        # Apply alpha in Float32 BEFORE converting to c_dtype to avoid overflow
        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
        acc_vec = epilogue_op((acc_vec * alpha_value).to(gemm_kernel.c_dtype))
        tRS_rC.store(acc_vec)

        c_buffer = (num_prev_subtiles + subtile_idx) % gemm_kernel.num_c_stage
        cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
        cute.arch.fence_proxy("async.shared", space="cta")
        epilog_sync_barrier.arrive_and_wait()

        if warp_idx == gemm_kernel.epilogue_warp_id[0]:
            cute.copy(
                tma_atom_c,
                bSG_sC[(None, c_buffer)],
                bSG_gC[(None, subtile_idx)],
            )
            c_pipeline.producer_commit()
            c_pipeline.producer_acquire()
        epilog_sync_barrier.arrive_and_wait()

    epilog_sync_barrier.arrive_and_wait()

    with cute.arch.elect_one():
        acc_pipeline.consumer_release(acc_consumer_state)
    acc_consumer_state.advance()
    return acc_consumer_state


@cute.jit
def epilogue_with_alpha(
    gemm_kernel,
    epi_tidx: Int32,
    tCtAcc_base: cute.Tensor,
    tCgC_base: cute.Tensor,
    epi_tile: cute.Tile,
    epilogue_op: Constexpr,
    alpha_value: cutlass.Float32,
    mma_tile_coord_mnl: Tuple[Int32, Int32, Int32],
    acc_consumer_state: pipeline.PipelineState,
    acc_pipeline: pipeline.PipelineAsync,
    tCcC_base: cute.Tensor = None,
    mC_mnl: cute.Tensor = None,
    overlapping_accum: Constexpr = False,
) -> pipeline.PipelineState:
    """
    Per-tile epilogue (direct store) that applies alpha scaling in Float32 BEFORE
    converting to c_dtype, preventing overflow for narrow types like Float16.

    This is a drop-in replacement for cutlass.utils.gemm.sm100.epilogue with an
    additional alpha_value parameter. The alpha is applied to the Float32 accumulator
    before the conversion to c_dtype.
    """
    tCgC = transform_partitioned_tensor_layout(tCgC_base)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)

    (
        tiled_copy_t2r,
        tTR_tAcc_base,
        tTR_rAcc,
    ) = epilogue_tmem_copy_and_partition(
        gemm_kernel, epi_tidx, tCtAcc, tCgC, epi_tile, gemm_kernel.use_2cta_instrs
    )

    gC_epi = cute.flat_divide(tCgC, epi_tile)
    thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
    tTR_gC_partitioned = thr_copy_t2r.partition_D(gC_epi)
    tTR_rC = cute.make_rmem_tensor(
        tTR_gC_partitioned[(None, None, None, 0, 0, 0, 0, 0)].shape, gemm_kernel.c_dtype
    )

    mclD = cute.max_common_layout(
        tTR_rC.layout, tTR_gC_partitioned[(None, None, None, 0, 0, 0, 0, 0)].layout
    )
    num_bits_per_copy = min(
        tTR_gC_partitioned.iterator.alignment * 8,
        cute.size(mclD) * gemm_kernel.c_dtype.width,
        256,
    )

    simt_atom = cute.make_copy_atom(
        cute.nvgpu.CopyStgOp(),
        gemm_kernel.c_dtype,
        num_bits_per_copy=num_bits_per_copy,
        l1c_evict_priority=CacheEvictionPriority.NO_ALLOCATE,
    )
    use_predication = tCcC_base is not None and mC_mnl is not None

    if const_expr(use_predication):
        tCcC = transform_partitioned_tensor_layout(tCcC_base)
        cC_epi = cute.flat_divide(tCcC, epi_tile)
        tTR_cC_partitioned = thr_copy_t2r.partition_D(cC_epi)

    tTR_gC = tTR_gC_partitioned[(None, None, None, None, None, *mma_tile_coord_mnl)]

    if const_expr(use_predication):
        tTR_cC = tTR_cC_partitioned[(None, None, None, None, None, *mma_tile_coord_mnl)]
        tTR_cC = cute.group_modes(tTR_cC, 3, cute.rank(tTR_cC))

    if const_expr(overlapping_accum):
        acc_stage_index = acc_consumer_state.phase
        reverse_subtile = acc_stage_index == 0
    else:
        acc_stage_index = acc_consumer_state.index

    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]

    acc_pipeline.consumer_wait(acc_consumer_state)

    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
    tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))

    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
    for subtile_idx in range(subtile_cnt):
        real_subtile_idx = subtile_idx
        if const_expr(overlapping_accum):
            if reverse_subtile:
                real_subtile_idx = subtile_cnt - 1 - subtile_idx

        tTR_gC_subtile = tTR_gC[(None, None, None, real_subtile_idx)]
        tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

        if const_expr(overlapping_accum):
            if subtile_idx == gemm_kernel.iter_acc_early_release_in_epilogue:
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()
        else:
            if subtile_idx == subtile_cnt - 1:
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

        # Apply alpha in Float32 BEFORE converting to c_dtype to avoid overflow
        acc_vec = tTR_rAcc.load()
        acc_vec = epilogue_op((acc_vec * alpha_value).to(gemm_kernel.c_dtype))
        tTR_rC.store(acc_vec)

        if const_expr(use_predication):
            tTR_cC_subtile = tTR_cC[(None, None, None, real_subtile_idx)]
            pred_C_shape = (1, *tTR_cC_subtile.shape[1:])
            pred_C = cute.make_rmem_tensor(pred_C_shape, Boolean)
            for m_idx in range(tTR_cC_subtile.shape[1]):
                for n_idx in range(tTR_cC_subtile.shape[2]):
                    vector_first_coord = tTR_cC_subtile[(0, m_idx, n_idx)]
                    pred_C[(0, m_idx, n_idx)] = cute.elem_less(
                        vector_first_coord, mC_mnl.shape
                    )
            cute.copy(simt_atom, tTR_rC, tTR_gC_subtile, pred=pred_C)
        else:
            cute.copy(simt_atom, tTR_rC, tTR_gC_subtile)

    return acc_consumer_state
