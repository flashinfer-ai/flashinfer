# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The block-scale fp8 GEMM body (class `gemm.e4m3.kblock128`, :func:`flashinfer.gemm_fp8_nt_groupwise` under
``batch_invariant=True``): the dense fp8 blocks with a per-block promotion.

The spec (`spec.reduction_blockscale`) folds K in blocks of ``BLOCK_K`` elements: each block is an ``mma32`` fold
from +0 (its partial ``P_kb``), and the partials are folded in block order from +0 with ``acc = fma.rn(mul.rn(s_a[m, kb],
s_b[n // 128, kb]), P_kb, acc)``. On the dense body a block is exactly one smem stage (128 fp8 elements of K: the lowering's inner
radix of 4 steps), so the kernel is the dense fp8 kernel with three changes, all ours:

* **the mainloop** (:func:`blockscale_mma_loop`) issues the dense ``mma_fold`` once per k-tile instead of once per tile: every
  k-tile's four steps run from a reset accumulator (``ACCUMULATE`` off on the first) into the next TMEM accumulator stage, which is
  committed to the epilogue -- the in-CTA chunk fold's mechanism (`epilogue.chunk_fold_pass`) with one-k-tile chunks and a
  runtime block count; the accumulator ring is as deep as the TMEM allows for the tile (`acc_stages_for`), so
  the MMA warp can run several blocks ahead of the promotion;
* **the scale ring** (:func:`blockscale_producer_loop`): the TMA warp, next to each block's operand stage, stages the block's scales
  in smem -- its 32 lanes ``cp.async`` the CTA tile's 128 A row scales (rows clamped to M - 1: a row past M multiplies a zero-filled
  partial and is never stored) and lane 0 the tile's one B block scale (the tile lies in one 128-column weight block; the block
  index clamped to the tensor) -- into a ring whose stage barrier completes when the copies land, blocks ahead of their use. The
  caller's fp32 tensors are read with their own strides (the ``K`` and ``MN`` layouts are one kernel). Scales read by the promotion
  itself, one global load per block on the epilogue's critical path, bound the kernel by the load's latency (the smem carve-out
  leaves the L1 no room for the gathered lines): section 7 of the design note has the measurement;
  The K-native staging (``native_k``: the API's choice at a few rows in the K layout, `api.native_k_staging`): a ring stage holds
  four consecutive blocks, a row's four scales one 16-byte ``cp.async`` from the caller's tensor as it is (four 4-byte copies
  where a row's address is not 16-byte aligned), the promotion reading each block's scale from the stage's vector in block
  order -- which bytes a stage holds, never the fold; the per-block staging reads the MN-major rows (the API's copy of a
  K-layout tensor), one coalesced 4-byte copy per row and block.
* **the promotion** (:func:`blockscale_epilogue_loop`): eight promotion warps in two warpgroups, each warpgroup owning one
  contiguous column half of the tile (both read the same TMEM lane quadrants at different columns; the group index is a runtime
  value that enters addresses only -- the TMEM columns, the group's C smem stage and store pipeline, the subtile stored -- so the
  fold is one code site; a tile of one epilogue subtile keeps one group), consume one accumulator stage per block: ``scale =
  mul.rn(s_a, s_b)`` once per thread and block from the ring stage (each thread owns one output row -- the 32x32b TMEM atom of the
  128-row CTA tile -- so one A scale and one B scale serve it for the block), ``tcgen05.ld`` of the block partial per epilogue
  subtile of the half, ``sum = fma.rn(scale, P, sum)`` per element -- two elements per ``fma.rn.f32x2``, each lane the same
  correctly rounded fma -- into a **register** running sum (the half's elements of this thread: ``cta_tile_m x tile_n / 256``
  fp32), then each warp fences its reads and arrives on the stage's barrier. After the last block the running sum goes through the
  chain (the single ``cvt.rn``) and the dense store path, one C smem stage and TMA-store pipeline per warpgroup. Half the tile's
  columns per thread lets the compiler issue the half's TMEM loads back to back (a single warpgroup's thread, at the register cap,
  waited for one load's fmas before the next load); twelve warps per CTA: the promotion warpgroups, the MMA and TMA warps and two
  idle warps filling the third warpgroup (``setmaxnreg`` budgets whole warpgroups, per role).

Why registers and not a TMEM running sum (the chunk fold's `chunk_fold_pass`): the promotion runs once per k-tile, so a TMEM sum
would add two TMEM passes (load, store) per block on top of the partial's load; the register sum reads TMEM once per block. The
register budget bounds the CTA tile to 128 x 128 elements (`generator.TILES_N_FP8BS`). Every placement decision here (tile,
cluster, raster, swizzle, the ring depths, the subtile order) moves no arithmetic: the fold's order is the spec's, block by block,
step by step.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.tensor_utils import LayoutEnum
from cutlass.utils.blackwell_helpers import get_smem_store_op, get_tmem_load_op
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

# FlashInfer's copy of the DSL's persistent tile scheduler (the DSL's own module is deprecated and warns per use)
from flashinfer.cute_dsl.attention.fmha.helpers import (
    static_persistent_tile_scheduler as tile_scheduler,
)

from ....batch_invariant.fp_interface.cute import CuteF32
from ..generator import Knobs, MainloopBlock
from ..spec import BLOCK_K, SCALE_BLOCK_N, EpilogueChain
from .epilogue import apply_chain_pre_cvt
from .mainloop import mma_fold
from .operands import (
    ACC_DTYPE,
    BAR_EPILOGUE,
    BAR_TMEM_ALLOC,
    IN_DT,
    OUT_DT,
    StaticPlan,
    TmaSetup,
    build_grid,
)

IN_FMT = "e4m3_bs"
TMEM_COLS = 512  # the accumulator ring's budget: the dense body's 512 TMEM columns
MAX_ACC_STAGES = 8  # the ring's depth is capped: beyond it the hand-off barriers cost more than the latency they hide (empirical)
NUM_PROMO_GROUPS = 2  # the promotion warps: two warpgroups, each owning every other epilogue subtile (a column half) of the tile
NUM_PROMO_WARPS = 4 * NUM_PROMO_GROUPS
PROMO_WARPS = tuple(range(NUM_PROMO_WARPS))
MMA_WARP_BS = NUM_PROMO_WARPS
TMA_WARP_BS = NUM_PROMO_WARPS + 1
NUM_WARPS_BS = (
    -(-(NUM_PROMO_WARPS + 2) // 4) * 4
)  # whole warpgroups (setmaxnreg is a warpgroup instruction); the last warps idle
THREADS_BS = 32 * NUM_WARPS_BS
# register budgets per thread (setmaxnreg): the promotion warps hold the running sum and two blocks' partials, the rest little;
# the sum over the warps stays within the register file
PROMO_REGS = 224
UNIFORM_REGS = 56
BAR_GROUP = (
    BAR_EPILOGUE,
    BAR_EPILOGUE + 2,
)  # one named barrier per promotion warpgroup (the C stage hand-off to its TMA store)
BAR_PROMO_JOIN = (
    BAR_GROUP[1] + 1
)  # both promotion warpgroups once, after their last TMEM loads, before the TMEM free
NUM_B_BLOCKS = 2  # B block scales per ring stage: a tile up to 256 wide spans two 128-column weight scale blocks (one per promotion group)
PROMO_BATCH = 2  # subtiles of a group's half loaded from TMEM before their fmas: the register budget's bound at the 256-wide tile
NUM_SCALE_STAGES = 16  # the scale ring's depth in blocks: blocks whose scales wait in smem ahead of the promotion; deeper than the operand
# ring plus the accumulator ring, so the operand stage is always the producer's binding wait (empirical)
SCALE_GROUP_NATIVE_K = 4  # the K-native staging (a few rows, K-layout scales): blocks per ring stage, a row's four block scales one 16-byte copy


def scale_ring_bytes(knobs: Knobs) -> int:
    """Smem of the scale ring: per stage the CTA tile's A row scales and one B block scale, fp32."""
    return (
        NUM_SCALE_STAGES
        * (knobs.cta_tile_mn[0] + NUM_B_BLOCKS)
        * (ACC_DTYPE.width // 8)
    )  # the same bytes in either staging


def acc_stages_for(knobs: Knobs) -> int:
    """The accumulator ring's depth for the record: one stage per K block in flight between the MMA warp and the promotion, as many
    as the TMEM holds for the tile (`TMEM_COLS // tile_n`, at most `MAX_ACC_STAGES`): 4 at tile N 128, 8 below. A placement: the
    stage a block lands in never enters the fold (Lean `promoteLoop`: the stage index is erased)."""
    return max(2, min(MAX_ACC_STAGES, TMEM_COLS // knobs.tile_n))


@cute.jit
def blockscale_mma_loop(
    lowering,
    num_acc_stage: cutlass.Constexpr,
    k_tile_cnt: cutlass.Int32,
    is_leader_cta,
    tile_sched,
    tiled_mma: cute.TiledMma,
    ab_consumer,
    acc_pipeline,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCtAcc_base: cute.Tensor,
):
    """Per tile, per k-tile (blocks ascending): the dense ``mma_fold`` of ONE k-tile -- the lowering's ``inner`` steps from a reset
    accumulator into the next accumulator stage, committed to the epilogue -- so every block's partial is the spec's ``P_kb``."""
    assert cute.size(tCrA, mode=[2]) == lowering.inner, (
        "one smem stage must hold exactly the lowering's inner steps (one scale block)"
    )
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, num_acc_stage
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        for _kb in range(
            k_tile_cnt
        ):  # blocks ascending, each a fold from +0 into the next stage
            ab_consumer, acc_state = mma_fold(
                lowering,
                1,
                is_leader_cta,
                tiled_mma,
                ab_consumer,
                acc_pipeline,
                acc_state,
                tCrA,
                tCrB,
                tCtAcc_base,
            )
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    acc_pipeline.producer_tail(acc_state)


def _vec4(t: cute.Tensor, coord):
    """A 4-element view of ``t`` from ``coord`` along its contiguous mode (the 16-byte copy or read of four consecutive block scales);
    the callers take this path only where the address is 16-byte aligned, which the pointer then asserts."""
    return cute.make_tensor(
        (t.iterator + cute.crd2idx(coord, t.layout)).align(16), cute.make_layout((4,))
    )


def _elem(t: cute.Tensor, coord):
    """A one-element view of ``t`` at ``coord`` (the copy atoms move tensors)."""
    return cute.make_tensor(
        t.iterator + cute.crd2idx(coord, t.layout), cute.make_layout((1,))
    )


@cute.jit
def blockscale_producer_loop(
    atom_thr_size: cutlass.Constexpr,
    k_tile_cnt: cutlass.Int32,
    tile_sched,
    ab_producer,
    tma_atom_a: cute.CopyAtom,
    tma_atom_b: cute.CopyAtom,
    tAgA: cute.Tensor,
    tBgB: cute.Tensor,
    tAsA: cute.Tensor,
    tBsB: cute.Tensor,
    a_mcast_mask,
    b_mcast_mask,
    scale_pipeline,
    mSFA: cute.Tensor,
    mSFB: cute.Tensor,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    tCcC: cute.Tensor,
    native_k: cutlass.Constexpr,
):
    """The dense TMA producer (`producer.tma_produce_ktiles`: one k-tile per operand stage, blocks ascending) plus the scale ring:
    per block, the warp's 32 lanes ``cp.async`` the CTA tile's A row scales (``mSFA[(row, kb)]``, rows clamped to M - 1) and lane 0
    the tile's B block scale (``mSFB[(n // 128, kb)]``, clamped to the tensor) into the next ring stage, whose barrier completes
    when the copies land. The K-native staging (``native_k``: a few rows, the caller's K-layout scales read as they are): a ring
    stage holds ``SCALE_GROUP_NATIVE_K`` consecutive blocks, filled at the group's first block -- a row's four block scales one
    16-byte copy where the row's address is 16-byte aligned, four 4-byte copies otherwise -- and the short last group block by
    block. Order and staging only: no arithmetic."""
    lane = cute.arch.lane_idx()
    M = mSFA.shape[0]
    NB = mSFB.shape[0]
    rows_per_lane = cute.size(sSFA, mode=[0]) // 32
    atom_g2s = cute.make_copy_atom(
        cpasync.CopyG2SOp(), ACC_DTYPE, num_bits_per_copy=ACC_DTYPE.width
    )
    atom_g2s4 = cute.make_copy_atom(
        cpasync.CopyG2SOp(), ACC_DTYPE, num_bits_per_copy=4 * ACC_DTYPE.width
    )
    SCALE_GROUP = SCALE_GROUP_NATIVE_K
    a_kmajor = mSFA.stride[1] == 1  # the K layout: a row's blocks contiguous
    b_kmajor = mSFB.stride[1] == 1
    k_whole = (
        k_tile_cnt // SCALE_GROUP
    )  # whole groups, then a short last group of k_tail blocks
    k_tail = k_tile_cnt - k_whole * SCALE_GROUP
    scale_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer,
        NUM_SCALE_STAGES // (SCALE_GROUP if native_k else 1),
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        coord = work_tile.tile_idx
        mnl = (coord[0] // atom_thr_size, coord[1], coord[2])
        tAgA_slice = tAgA[
            (None, mnl[0], None, coord[2])
        ]  # ((atom_v, rest_v), RestK): this tile's A rows, all k-tiles
        tBgB_slice = tBgB[(None, coord[1], None, coord[2])]
        c0 = tCcC[(None, None, None, mnl[0], mnl[1], mnl[2])][
            0
        ]  # this CTA tile's first output coordinate (row, column)
        nb = cutlass.min(c0[1] // SCALE_BLOCK_N, NB - 1)
        ab_producer.reset()
        peek_empty = ab_producer.try_acquire()
        if cutlass.const_expr(native_k):
            for g in cutlass.range(
                0, k_whole, 1, unroll=1
            ):  # the whole groups: one ring stage, then four operand stages
                kb0 = g * SCALE_GROUP
                scale_pipeline.producer_acquire(
                    scale_state
                )  # ring stage free (the promotion released it)
                for r in cutlass.range_constexpr(rows_per_lane):
                    i = r * 32 + lane
                    row = cutlass.min(c0[0] + i, M - 1)
                    if a_kmajor and ((row * mSFA.stride[0] + kb0) % 4 == 0):
                        cute.copy(
                            atom_g2s4,
                            _vec4(mSFA, (row, kb0)),
                            _vec4(sSFA, (i, 0, scale_state.index)),
                        )
                    else:
                        for j in cutlass.range_constexpr(SCALE_GROUP):
                            cute.copy(
                                atom_g2s,
                                _elem(mSFA, (row, kb0 + j)),
                                _elem(sSFA, (i, j, scale_state.index)),
                            )
                if (
                    lane < NUM_B_BLOCKS
                ):  # the tile's B block scales, one lane each (clamped to the tensor)
                    nb_l = cutlass.min(nb + lane, NB - 1)
                    if b_kmajor and ((nb_l * mSFB.stride[0] + kb0) % 4 == 0):
                        cute.copy(
                            atom_g2s4,
                            _vec4(mSFB, (nb_l, kb0)),
                            _vec4(sSFB, (lane, 0, scale_state.index)),
                        )
                    else:
                        for j in cutlass.range_constexpr(SCALE_GROUP):
                            cute.copy(
                                atom_g2s,
                                _elem(mSFB, (nb_l, kb0 + j)),
                                _elem(sSFB, (lane, j, scale_state.index)),
                            )
                scale_pipeline.producer_commit(
                    scale_state
                )  # arrives when this lane's copies have landed
                scale_state.advance()
                for _j in cutlass.range_constexpr(
                    SCALE_GROUP
                ):  # the group's blocks: one operand stage each
                    handle = ab_producer.acquire_and_advance(
                        peek_empty
                    )  # operand slot empty; the leader arms the tx barrier
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, handle.count)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_mcast_mask,
                    )
                    peek_empty = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_empty = ab_producer.try_acquire()
            if (
                k_tail > 0
            ):  # the short last group: one stage, k_tail blocks, 4-byte copies
                kb0 = k_whole * SCALE_GROUP
                scale_pipeline.producer_acquire(scale_state)
                for r in cutlass.range_constexpr(rows_per_lane):
                    i = r * 32 + lane
                    row = cutlass.min(c0[0] + i, M - 1)
                    for j in cutlass.range_constexpr(SCALE_GROUP):
                        if j < k_tail:
                            cute.copy(
                                atom_g2s,
                                _elem(mSFA, (row, kb0 + j)),
                                _elem(sSFA, (i, j, scale_state.index)),
                            )
                if lane < NUM_B_BLOCKS:
                    nb_l = cutlass.min(nb + lane, NB - 1)
                    for j in cutlass.range_constexpr(SCALE_GROUP):
                        if j < k_tail:
                            cute.copy(
                                atom_g2s,
                                _elem(mSFB, (nb_l, kb0 + j)),
                                _elem(sSFB, (lane, j, scale_state.index)),
                            )
                scale_pipeline.producer_commit(scale_state)
                for _j in cutlass.range(0, k_tail, 1, unroll=1):
                    handle = ab_producer.acquire_and_advance(peek_empty)
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, handle.count)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_mcast_mask,
                    )
                    peek_empty = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_empty = ab_producer.try_acquire()
            scale_state.advance(
                cutlass.Int32(k_tail > 0)
            )  # the tail's stage, when there is one
        else:
            for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                handle = ab_producer.acquire_and_advance(
                    peek_empty
                )  # operand slot empty; the leader arms the tx barrier
                cute.copy(
                    tma_atom_a,
                    tAgA_slice[(None, handle.count)],
                    tAsA[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                    mcast_mask=a_mcast_mask,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_slice[(None, handle.count)],
                    tBsB[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                    mcast_mask=b_mcast_mask,
                )
                scale_pipeline.producer_acquire(
                    scale_state
                )  # ring stage free (the promotion released it)
                for r in cutlass.range_constexpr(rows_per_lane):
                    i = r * 32 + lane
                    row = cutlass.min(c0[0] + i, M - 1)
                    cute.copy(
                        atom_g2s,
                        _elem(mSFA, (row, handle.count)),
                        _elem(sSFA, (i, scale_state.index)),
                    )
                if (
                    lane < NUM_B_BLOCKS
                ):  # the tile's B block scales, one lane each (clamped to the tensor)
                    nb_l = cutlass.min(nb + lane, NB - 1)
                    cute.copy(
                        atom_g2s,
                        _elem(mSFB, (nb_l, handle.count)),
                        _elem(sSFB, (lane, scale_state.index)),
                    )
                scale_pipeline.producer_commit(
                    scale_state
                )  # arrives when this lane's copies have landed
                scale_state.advance()
                peek_empty = cutlass.Boolean(1)
                if handle.count + 1 < k_tile_cnt:
                    peek_empty = ab_producer.try_acquire()
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    ab_producer.tail()
    scale_pipeline.producer_tail(scale_state)


@cute.jit
def blockscale_epilogue_loop(
    st,
    chain,
    atom_thr_size: cutlass.Constexpr,
    tidx: cutlass.Int32,
    warp_idx: cutlass.Int32,
    tile_sched,
    tma_atom_c: cute.CopyAtom,
    tCtAcc_base: cute.Tensor,
    sC: cute.Tensor,
    tCgC_base: cute.Tensor,
    tCcC_base: cute.Tensor,
    epi_tile: cute.Tile,
    acc_pipeline,
    scale_pipeline,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    k_tile_cnt: cutlass.Int32,
    native_k: cutlass.Constexpr,
):
    """The promotion and the store (module docstring) by two warpgroups, each owning one contiguous column half of the tile (both
    read the same TMEM lane quadrants -- the 32x32b atom -- at different columns; a tile of one epilogue subtile is group 0's
    alone). Per block each thread loads the group's subtiles of the block partial back to back (the register budget of a half
    leaves the compiler free to issue them early), folds them with packed fmas and releases the stage. The group index is a
    runtime value that enters addresses only (TMEM columns, the C smem stage, the subtile stored), never the fold. ``sSFA``
    ``(cta_tile_m, STAGE)`` and ``sSFB`` ``(STAGE,)`` are the scale ring's smem; ``tCcC_base`` the output coordinates partitioned
    like C (each thread finds its row within the CTA tile from them)."""
    g = warp_idx // 4  # this warp's promotion group
    tidx_g = tidx % 128  # the thread within its group
    # ((ATOM_M, MMA_M), (ATOM_N, MMA_N), ...)
    tCgC = transform_partitioned_tensor_layout(tCgC_base)
    tCcC = transform_partitioned_tensor_layout(tCcC_base)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
    # the epilogue subtiles (the TMA store's unit): the 128-thread tiled copy of one subtile, shared by both groups
    copy_atom_t2r = get_tmem_load_op(
        st.cta_tile_shape_mnk,
        st.c_layout,
        st.c_dtype,
        ACC_DTYPE,
        epi_tile,
        st.use_2cta_instrs,
    )
    tAcc_epi = cute.flat_divide(tCtAcc, epi_tile)  # (EPI_M, EPI_N, SUB_M, SUB_N, STAGE)
    tiled_copy_t2r = tcgen05.make_tmem_copy(
        copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
    )
    thr_t2r = tiled_copy_t2r.get_slice(tidx_g)
    tCgC_epi = cute.flat_divide(
        tCgC, epi_tile
    )  # (EPI_M, EPI_N, SUB_M, SUB_N, RestM, RestN, RestL)
    tTR_gC = thr_t2r.partition_D(tCgC_epi)
    tCcC_epi = cute.flat_divide(tCcC, epi_tile)
    tTR_cC = thr_t2r.partition_D(
        tCcC_epi
    )  # the output coordinates in the accumulator's partition
    frag_shape = tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape
    tTR_rP = cute.make_rmem_tensor(
        frag_shape, ACC_DTYPE
    )  # one subtile's values on the way to the store
    tTR_rC = cute.make_rmem_tensor(frag_shape, st.c_dtype)
    V = cute.size(tTR_rP)
    sub_m, sub_n = cute.size(tAcc_epi, mode=[2]), cute.size(tAcc_epi, mode=[3])
    assert sub_m == 1, (
        "the epilogue subtiles run along N only (the CTA tile's 128 rows are one subtile row)"
    )
    subtile_cnt = sub_n
    groups = (
        NUM_PROMO_GROUPS if subtile_cnt >= NUM_PROMO_GROUPS else 1
    )  # a one-subtile tile is group 0's alone
    per_group = subtile_cnt // groups  # subtiles per group: contiguous column halves
    assert per_group * groups == subtile_cnt, (subtile_cnt, groups)
    cols_g = per_group * cute.size(epi_tile[1])  # this group's columns
    assert cols_g % 2 == 0, cols_g  # the promotion packs element pairs
    tTR_tAcc_base = thr_t2r.partition_S(
        tAcc_epi
    )  # (T2R, T2R_M, T2R_N, SUB_M, SUB_N, STAGE)
    # the static shape of the fragment: every element of a fragment, and of every subtile of the tile, lies on ONE output row (the
    # 32x32b TMEM atom of the 128-row CTA tile), so one A scale serves the thread for a block; the columns of a tile lie in one
    # 128-column weight scale block (tile N divides 128), so one B scale does too
    offs0 = tTR_cC[(None, None, None, 0, 0, 0, 0, 0)]
    offs = [offs0.layout(e) for e in range(V)]
    assert all(o[0] == offs[0][0] for o in offs), (
        "the fragment spans more than one output row"
    )
    for s in range(subtile_cnt):
        assert tTR_cC[(None, None, None, 0, s, 0, 0, 0)].layout(0)[0] == offs[0][0], (
            "a subtile changes the output row"
        )
    assert cols_g <= SCALE_BLOCK_N and SCALE_BLOCK_N % cols_g == 0, (
        cols_g
    )  # a group's columns lie in one weight scale block
    assert subtile_cnt * cute.size(epi_tile[1]) <= NUM_B_BLOCKS * SCALE_BLOCK_N
    blk_per_group = (
        cols_g // SCALE_BLOCK_N
    )  # 1 at the 256-wide tile (group g = block g), else 0
    assert per_group % PROMO_BATCH == 0 or per_group < PROMO_BATCH, (
        per_group,
        PROMO_BATCH,
    )
    batch = min(per_group, PROMO_BATCH)
    VW = per_group * V  # this group's elements per thread
    bufs = [
        cute.make_rmem_tensor(frag_shape, ACC_DTYPE) for _ in range(batch)
    ]  # the block partial of a batch of the group's subtiles
    tTR_rSum = cute.make_rmem_tensor(
        (VW,), ACC_DTYPE
    )  # the running sum of this group's half, in registers
    # registers -> smem -> global: each group stores through its own C stage and its own TMA-store pipeline
    copy_atom_r2s = get_smem_store_op(
        st.c_layout, st.c_dtype, ACC_DTYPE, tiled_copy_t2r
    )
    tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
    thr_r2s = tiled_copy_r2s.get_slice(tidx_g)
    tRS_sC = thr_r2s.partition_D(sC)
    tRS_rC = tiled_copy_r2s.retile(tTR_rC)
    bSG_sC, bSG_gC_all = cpasync.tma_partition(
        tma_atom_c,
        0,
        cute.make_layout(1),
        cute.group_modes(sC, 0, 2),
        cute.group_modes(tCgC_epi, 0, 2),
    )
    assert st.num_c_stage >= NUM_PROMO_GROUPS, st.num_c_stage
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, st.num_acc_stage
    )
    SCALE_GROUP = SCALE_GROUP_NATIVE_K
    scale_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer,
        NUM_SCALE_STAGES // (SCALE_GROUP if native_k else 1),
    )
    c_pipeline0 = pipeline.PipelineTmaStore.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
    )
    c_pipeline1 = pipeline.PipelineTmaStore.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
    )
    sync0 = pipeline.NamedBarrier(barrier_id=BAR_GROUP[0], num_threads=128)
    sync1 = pipeline.NamedBarrier(barrier_id=BAR_GROUP[1], num_threads=128)
    gsel = g if groups == 2 else cutlass.Int32(0)
    active = (
        (g == 0) if groups == 1 else (g >= 0)
    )  # a one-subtile tile: group 1 has no columns
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        coord = work_tile.tile_idx
        mnl = (coord[0] // atom_thr_size, coord[1], coord[2])
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
        bSG_gC = bSG_gC_all[(None, None, None, mnl[0], mnl[1], mnl[2])]
        bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
        # this thread's row within the CTA tile: the slot of its A scale in a ring stage
        irow = (
            tTR_cC[(None, None, None, 0, 0, mnl[0], mnl[1], mnl[2])][0][0]
            - tCcC[(None, None, mnl[0], mnl[1], mnl[2])][0][0]
        )
        for i in cutlass.range_constexpr(VW):
            tTR_rSum[i] = CuteF32.const(0.0)  # acc_0 = +0
        if cutlass.const_expr(native_k):
            k_whole = k_tile_cnt // SCALE_GROUP  # whole groups, then a short last group
            k_tail = k_tile_cnt - k_whole * SCALE_GROUP
            for _g in cutlass.range(
                0, k_whole, 1, unroll=1
            ):  # the ring's stages: groups of SCALE_GROUP blocks
                scale_pipeline.consumer_wait(
                    scale_state
                )  # the group's scales are in the ring
                a4 = _vec4(
                    sSFA, (irow, 0, scale_state.index)
                ).load()  # this row's four block scales: one 16-byte read
                b4 = _vec4(sSFB, (gsel * blk_per_group, 0, scale_state.index)).load()
                for jb in cutlass.range_constexpr(
                    SCALE_GROUP
                ):  # blocks ascending: the block fold digit
                    scale = CuteF32.mul(
                        a4[jb], b4[jb]
                    )  # mul.rn(s_a[m, kb], s_b[n // 128, kb]), once
                    acc_pipeline.consumer_wait(acc_state)  # block kb's fold is complete
                    if active:
                        tTR_tAcc = cute.group_modes(
                            tTR_tAcc_base[
                                (None, None, None, None, None, acc_state.index)
                            ],
                            3,
                            5,
                        )
                        for b0 in cutlass.range_constexpr(
                            0, per_group, batch
                        ):  # the group's subtiles in batches: a batch's loads first
                            for j in cutlass.range_constexpr(batch):
                                cute.copy(
                                    tiled_copy_t2r,
                                    tTR_tAcc[
                                        (None, None, None, gsel * per_group + b0 + j)
                                    ],
                                    bufs[j],
                                )
                            for j in cutlass.range_constexpr(batch):
                                k = b0 + j
                                for e in cutlass.range_constexpr(0, V, 2):
                                    # acc = fma(scale, P_kb, acc) for two elements per fma.rn.f32x2: each lane the same correctly rounded fma on
                                    # the same operands (a packing placement, as the attention body's)
                                    r0, r1 = CuteF32.fma2(
                                        (scale, scale),
                                        (bufs[j][e], bufs[j][e + 1]),
                                        (tTR_rSum[k * V + e], tTR_rSum[k * V + e + 1]),
                                    )
                                    tTR_rSum[k * V + e] = r0
                                    tTR_rSum[k * V + e + 1] = r1
                    cute.arch.fence_view_async_tmem_load()  # this warp's reads of the partial are complete
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(
                            acc_state
                        )  # one arrive per warp: the stage is free once all arrived
                    acc_state.advance()
                scale_pipeline.consumer_release(
                    scale_state
                )  # read: the stage may take a later group
                scale_state.advance()
            if k_tail > 0:  # the short last group: its blocks one by one
                scale_pipeline.consumer_wait(scale_state)
            for jb in cutlass.range(0, k_tail, 1, unroll=1):
                scale = CuteF32.mul(
                    sSFA[(irow, jb, scale_state.index)],
                    sSFB[(gsel * blk_per_group, jb, scale_state.index)],
                )  # mul.rn(s_a[m, kb], s_b[n // 128, kb]), once
                acc_pipeline.consumer_wait(acc_state)  # block kb's fold is complete
                if active:
                    tTR_tAcc = cute.group_modes(
                        tTR_tAcc_base[(None, None, None, None, None, acc_state.index)],
                        3,
                        5,
                    )
                    for b0 in cutlass.range_constexpr(
                        0, per_group, batch
                    ):  # the group's subtiles in batches: a batch's loads first
                        for j in cutlass.range_constexpr(batch):
                            cute.copy(
                                tiled_copy_t2r,
                                tTR_tAcc[(None, None, None, gsel * per_group + b0 + j)],
                                bufs[j],
                            )
                        for j in cutlass.range_constexpr(batch):
                            k = b0 + j
                            for e in cutlass.range_constexpr(0, V, 2):
                                # acc = fma(scale, P_kb, acc) for two elements per fma.rn.f32x2: each lane the same correctly rounded fma on
                                # the same operands (a packing placement, as the attention body's)
                                r0, r1 = CuteF32.fma2(
                                    (scale, scale),
                                    (bufs[j][e], bufs[j][e + 1]),
                                    (tTR_rSum[k * V + e], tTR_rSum[k * V + e + 1]),
                                )
                                tTR_rSum[k * V + e] = r0
                                tTR_rSum[k * V + e + 1] = r1
                cute.arch.fence_view_async_tmem_load()  # this warp's reads of the partial are complete
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(
                        acc_state
                    )  # one arrive per warp: the stage is free once all arrived
                acc_state.advance()
            if k_tail > 0:
                scale_pipeline.consumer_release(scale_state)
            scale_state.advance(cutlass.Int32(k_tail > 0))
        else:
            for _kb in range(k_tile_cnt):  # blocks ascending: the block fold digit
                scale_pipeline.consumer_wait(
                    scale_state
                )  # block kb's scales are in the ring
                scale = CuteF32.mul(
                    sSFA[(irow, scale_state.index)],
                    sSFB[(gsel * blk_per_group, scale_state.index)],
                )  # mul.rn(s_a[m, kb], s_b[n // 128, kb]), once
                scale_pipeline.consumer_release(
                    scale_state
                )  # read: the stage may take a later block
                scale_state.advance()
                acc_pipeline.consumer_wait(acc_state)  # block kb's fold is complete
                if active:
                    tTR_tAcc = cute.group_modes(
                        tTR_tAcc_base[(None, None, None, None, None, acc_state.index)],
                        3,
                        5,
                    )
                    for b0 in cutlass.range_constexpr(
                        0, per_group, batch
                    ):  # the group's subtiles in batches: a batch's loads first
                        for j in cutlass.range_constexpr(batch):
                            cute.copy(
                                tiled_copy_t2r,
                                tTR_tAcc[(None, None, None, gsel * per_group + b0 + j)],
                                bufs[j],
                            )
                        for j in cutlass.range_constexpr(batch):
                            k = b0 + j
                            for e in cutlass.range_constexpr(0, V, 2):
                                # acc = fma(scale, P_kb, acc) for two elements per fma.rn.f32x2: each lane the same correctly rounded fma on
                                # the same operands (a packing placement, as the attention body's)
                                r0, r1 = CuteF32.fma2(
                                    (scale, scale),
                                    (bufs[j][e], bufs[j][e + 1]),
                                    (tTR_rSum[k * V + e], tTR_rSum[k * V + e + 1]),
                                )
                                tTR_rSum[k * V + e] = r0
                                tTR_rSum[k * V + e + 1] = r1
                cute.arch.fence_view_async_tmem_load()  # this warp's reads of the partial are complete
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(
                        acc_state
                    )  # one arrive per warp: the stage is free once all arrived
                acc_state.advance()
        # the chain (one cvt.rn) and the store of this group's half of the tile's fold, subtile by subtile
        if active:
            for k in cutlass.range_constexpr(per_group):
                for e in cutlass.range_constexpr(V):
                    tTR_rP[e] = tTR_rSum[k * V + e]
                apply_chain_pre_cvt(
                    chain, tTR_rP, None
                )  # the chain's fp32 ops (none: the scales are in the fold)
                acc_vec = tiled_copy_r2s.retile(tTR_rP).load()
                tRS_rC.store(
                    acc_vec.to(st.c_dtype)
                )  # the chain's cvt.rn <out>, exactly once
                cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, gsel)])
                cute.arch.fence_proxy("async.shared", space="cta")
                if gsel == 0:
                    sync0.arrive_and_wait()
                else:
                    sync1.arrive_and_wait()
                if warp_idx == 4 * gsel:  # the group's leader warp issues the TMA store
                    cute.copy(
                        tma_atom_c,
                        bSG_sC[(None, gsel)],
                        bSG_gC[(None, gsel * per_group + k)],
                    )
                    if gsel == 0:
                        c_pipeline0.producer_commit()
                        c_pipeline0.producer_acquire()
                    else:
                        c_pipeline1.producer_commit()
                        c_pipeline1.producer_acquire()
                if gsel == 0:
                    sync0.arrive_and_wait()
                else:
                    sync1.arrive_and_wait()
    if gsel == 0:
        c_pipeline0.producer_tail()
    else:
        c_pipeline1.producer_tail()


class BlockScaleFp8Gemm:
    """``C[m, n] = cvt( fold_{kb} fma(mul(SFA[m, kb], SFB[n // 128, kb]), P_kb, acc) )`` for the knobs; M, N, K dynamic, K a whole
    number of 128-element blocks. Operands: A ``(M, K, 1)`` and B ``(N, K, 1)`` e4m3 with K contiguous, SFA ``(M, K / 128)`` and
    SFB ``(ceil(N / 128), K / 128)`` fp32 with any strides, C ``(M, N, 1)`` row-major in the output format (the chain is the
    conversion alone)."""

    def __init__(self, knobs: Knobs, out_fmt: str, native_k: bool = False):
        self.knobs = knobs
        self.native_k = native_k  # the K-native scale staging (the API's choice at a few rows in the K layout)
        self.cluster_shape_mn = knobs.cluster_shape_mn
        self.chain = EpilogueChain.for_class(out_fmt, False)
        assert not self.chain.scaled and self.chain.out_fmt == out_fmt, (
            self.chain.describe()
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        assert a.element_type is IN_DT[IN_FMT] and b.element_type is IN_DT[IN_FMT], (
            a.element_type,
            b.element_type,
        )
        assert OUT_DT[self.chain.out_fmt] is c.element_type, (
            self.chain.describe(),
            c.element_type,
        )
        assert sfa.element_type is ACC_DTYPE and sfb.element_type is ACC_DTYPE
        st = StaticPlan(
            self.knobs,
            a.element_type,
            b.element_type,
            c.element_type,
            LayoutEnum.from_tensor(a).mma_major_mode(),
            LayoutEnum.from_tensor(b).mma_major_mode(),
            LayoutEnum.from_tensor(c),
            chunks=1,
            in_fmt=IN_FMT,
            acc_stages=acc_stages_for(self.knobs),
            reserve_smem_bytes=scale_ring_bytes(self.knobs),
        )
        self.st = st
        self.block = MainloopBlock(IN_FMT)
        self.lowering = self.block.lowering()
        assert st.mma_tiler[2] == self.block.k_block == BLOCK_K, (
            st.mma_tiler,
            self.block.k_block,
        )  # one stage = one scale block
        tma = TmaSetup(st, a, b, c)
        self.num_tma_load_bytes = tma.tx_bytes
        params, grid = build_grid(st, c, max_active_clusters)
        self.kernel(
            st.tiled_mma,
            tma.atom_a,
            tma.tensor_a,
            tma.atom_b,
            tma.tensor_b,
            tma.atom_c,
            tma.tensor_c,
            sfa,
            sfb,
            st.cluster_layout_vmnk,
            st.a_smem_layout_staged,
            st.b_smem_layout_staged,
            st.c_smem_layout_staged,
            st.epi_tile,
            params,
        ).launch(
            grid=grid,
            block=[THREADS_BS, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mSFA: cute.Tensor,
        mSFB: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged,
        epi_tile: cute.Tile,
        tile_sched_params: tile_scheduler.PersistentTileSchedulerParams,
    ):
        st = self.st
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == TMA_WARP_BS:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        use_2cta_instrs = atom_thr_size == 2
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % atom_thr_size
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        # --- scaffolding: barriers, pipelines, TMEM allocator (the dense body's)
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, st.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, st.num_acc_stage * 2]
            scale_mbar_ptr: cute.struct.MemRange[cutlass.Int64, NUM_SCALE_STAGES * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=st.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Warp),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            enable_multicast_signaling=True,
            defer_sync=True,
        ).make_participants()
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=st.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(PROMO_WARPS) * (2 if use_2cta_instrs else 1)
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        scale_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.scale_mbar_ptr.data_ptr(),
            num_stages=NUM_SCALE_STAGES
            // (SCALE_GROUP_NATIVE_K if self.native_k else 1),
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 32 * len(PROMO_WARPS)
            ),
            defer_sync=True,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=BAR_TMEM_ALLOC, num_threads=32 * (len(PROMO_WARPS) + 1)
        )
        promo_join = pipeline.NamedBarrier(
            barrier_id=BAR_PROMO_JOIN, num_threads=32 * NUM_PROMO_WARPS
        )
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=PROMO_WARPS[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # --- operand staging: smem rings, multicast masks, global tiles and partitions
        sA = smem.allocate_tensor(
            element_type=st.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=st.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        # the scale ring: a stage holds the CTA tile's A row scales and the tile's B block scale
        tm_cta = st.cta_tile_shape_mnk[0]
        if cutlass.const_expr(self.native_k):
            # a stage holds four blocks; a row's four slots are 16 contiguous bytes (one copy in, one read out)
            G = SCALE_GROUP_NATIVE_K
            sSFA = smem.allocate_tensor(
                element_type=ACC_DTYPE,
                layout=cute.make_layout(
                    (tm_cta, G, NUM_SCALE_STAGES // G), stride=(G, 1, tm_cta * G)
                ),
                byte_alignment=16,
            )
            sSFB = smem.allocate_tensor(
                element_type=ACC_DTYPE,
                layout=cute.make_layout(
                    (NUM_B_BLOCKS, G, NUM_SCALE_STAGES // G),
                    stride=(G, 1, NUM_B_BLOCKS * G),
                ),
                byte_alignment=16,
            )
        else:
            sSFA = smem.allocate_tensor(
                element_type=ACC_DTYPE,
                layout=cute.make_layout((tm_cta, NUM_SCALE_STAGES)),
                byte_alignment=16,
            )
            sSFB = smem.allocate_tensor(
                element_type=ACC_DTYPE,
                layout=cute.make_layout((NUM_B_BLOCKS, NUM_SCALE_STAGES)),
                byte_alignment=16,
            )
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(st.is_a_mcast or st.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(st.mma_tiler, (None, 0, None)), (None, None, None)
        )  # (bM, bK, RestM, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(st.mma_tiler, (0, None, None)), (None, None, None)
        )  # (bN, bK, RestN, RestK, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(st.mma_tiler, (None, None, 0)), (None, None, None)
        )  # (bM, bN, RestM, RestN, RestL)
        # the output coordinates, tiled and partitioned like C: the epilogue reads each thread's row and the tile's column block
        cC_mnl = cute.local_tile(
            cute.make_identity_tensor(mC_mnl.shape),
            cute.slice_(st.mma_tiler, (None, None, 0)),
            (None, None, None),
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])  # = K / BLOCK_K: the block count
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)
        tCcC = thr_mma.partition_C(cC_mnl)
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tCrA = tiled_mma.make_fragment_A(sA)  # (MMA, MMA_M, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)  # (MMA, MMA_N, MMA_K, STAGE)
        acc_shape = tiled_mma.partition_shape_C(st.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, st.num_acc_stage)
        )
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # --- roles
        if warp_idx == TMA_WARP_BS:
            cute.arch.setmaxregister_decrease(UNIFORM_REGS)
            tile_sched = tile_scheduler.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            blockscale_producer_loop(
                atom_thr_size,
                k_tile_cnt,
                tile_sched,
                ab_producer,
                tma_atom_a,
                tma_atom_b,
                tAgA,
                tBgB,
                tAsA,
                tBsB,
                a_full_mcast_mask,
                b_full_mcast_mask,
                scale_pipeline,
                mSFA,
                mSFB,
                sSFA,
                sSFB,
                tCcC,
                self.native_k,
            )
        if warp_idx == MMA_WARP_BS:
            cute.arch.setmaxregister_decrease(UNIFORM_REGS)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tile_sched = tile_scheduler.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            blockscale_mma_loop(
                self.lowering,
                st.num_acc_stage,
                k_tile_cnt,
                is_leader_cta,
                tile_sched,
                tiled_mma,
                ab_consumer,
                acc_pipeline,
                tCrA,
                tCrB,
                tCtAcc_base,
            )
        sC = smem.allocate_tensor(
            element_type=st.c_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=c_smem_layout_staged.inner,
        )
        if warp_idx < NUM_PROMO_WARPS:
            cute.arch.setmaxregister_increase(PROMO_REGS)
            tmem.allocate(st.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tile_sched = tile_scheduler.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            blockscale_epilogue_loop(
                st,
                self.chain,
                atom_thr_size,
                tidx,
                warp_idx,
                tile_sched,
                tma_atom_c,
                tCtAcc_base,
                sC,
                tCgC,
                tCcC,
                epi_tile,
                acc_pipeline,
                scale_pipeline,
                sSFA,
                sSFB,
                k_tile_cnt,
                self.native_k,
            )
            tmem.relinquish_alloc_permit()
            # the allocator warp alone frees the columns (`TmemAllocator.free`): the other warpgroup's last TMEM loads must be
            # complete first; its store barriers are its own, so the groups join here (the dense bodies' one epilogue group is
            # joined by its store barrier already)
            promo_join.arrive_and_wait()
            tmem.free(tmem_ptr)
        if warp_idx > TMA_WARP_BS:
            cute.arch.setmaxregister_decrease(
                UNIFORM_REGS
            )  # the warps padding the last warpgroup: no role
