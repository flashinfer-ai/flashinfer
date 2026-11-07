# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Epilogue block: the spec's elementwise chain after the fold, then the store.

Per epilogue subtile: TMEM -> registers (fp32), the chain's ops before the conversion applied per element by
``EpilogueChain.apply_pre`` (the same function that builds the spec term: ``mul.rn alpha`` for the fp8
classes, ``add.rn bias`` of the element's column for the biased bf16 chain, nothing otherwise), then the chain's
single ``cvt.rn`` as the vector conversion to the output type,
registers -> smem -> TMA store (clipped at the tensor bounds). The subtile order, the C smem buffer rotation and
the barriers are placements. Built from the library's epilogue atoms (``get_tmem_load_op``,
``get_smem_store_op``); the body is ours.

The in-CTA chunk fold (``chunks > 1``: the split classes' ``cta_chunks`` records): the mainloop hands the epilogue one completed accumulator per chunk
(each chunk folded from +0), and ``chunk_fold_pass`` folds it into a TMEM running sum in chunk order
from +0 with ``add.rn`` -- ``sum = add(+0, P_0)``, ``sum = add(sum, P_c)`` -- exactly the ``chunk``
fold digit of ``spec.reduction_chunked`` as the workspace fixup performs it, the partials living in
TMEM instead of an fp32 workspace in HBM. After the last chunk the folded tile sits in the
accumulator stage and goes through the ordinary subtile loop (the chain once, then the store).
TMEM traffic goes through the epilogue's own ``tcgen05.ld`` atom per subtile and its ``tcgen05.st`` twin
(``tmem_store_atom_for``), each warp on its own lanes (the TMEM access restriction), so the fold needs no
inter-warp exchange and works for every tile shape the epilogue already handles.
"""

from __future__ import annotations

from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.utils.blackwell_helpers import get_smem_store_op, get_tmem_load_op
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from ....batch_invariant.fp_interface.cute import CuteF32
from .operands import ACC_DTYPE, BAR_EPILOGUE, EPILOGUE_WARPS


def apply_chain_pre_cvt(chain, frag, alpha):
    """The chain's ops before the conversion, per element, in fp32 through ``CuteF32``; the conversion itself is the
    vector ``.to(out dtype)`` that follows in the caller."""
    if chain.pre:
        for i in range(cute.size(frag)):
            frag[i] = chain.apply_pre(CuteF32, frag[i], alpha)


def tmem_store_atom_for(load_atom: cute.CopyAtom, elem_ty) -> cute.CopyAtom:
    """The ``tcgen05.st`` copy atom of the same shape as a ``tcgen05.ld`` atom (``get_tmem_load_op``'s pick for the epilogue tile):
    the same data-path lanes, bit width and repetition, so a fragment loaded with the one is stored back to the same TMEM
    coordinates with the other (``Ld32x32bOp(x32)`` -> ``St32x32bOp(x32)``, ``Ld16x256bOp(x8)`` -> ``St16x256bOp(x8)``, ...).
    Built from the atom's op (``load_atom.op``: the op class names the shape, ``repeat`` its repetition)."""
    op = load_atom.op
    name = type(op).__name__  # e.g. "Ld32x32bOp"
    assert name.startswith("Ld") and name.endswith("Op"), name
    st_cls = getattr(tcgen05.copy, "St" + name[2:])
    return cute.make_copy_atom(st_cls(op.repeat), elem_ty)


@cute.jit
def bias_per_element(
    chain,
    mBias: cute.Tensor,
    coords: cute.Tensor,
    frag: cute.Tensor,
    alpha,
    N: cutlass.Int32,
):
    """The biased chain's ops before the conversion, element by element: the bias of the element's output column, widened
    exactly; the columns at or beyond ``N`` are left alone (the store drops them and the bias is never read past its end)."""
    for e in cutlass.range_constexpr(cute.size(frag)):
        n = coords[e][1]
        if n < N:
            frag[e] = chain.apply_pre(CuteF32, frag[e], alpha, CuteF32.widen(mBias[n]))


@cute.jit
def apply_chain_biased(
    chain,
    mBias: cute.Tensor,
    coords: cute.Tensor,
    frag: cute.Tensor,
    alpha,
    N: cutlass.Int32,
):
    """The chain's ops before the conversion when the chain has the bias step: per element, the bias of the element's
    output column (``coords``: the output coordinates partitioned like the accumulator), widened exactly, through the
    same ``apply_pre`` as the term. Whether the fragment's elements are consecutive columns of one row is decided at
    trace time from the fragment's coordinate layout (its per-element offsets are static): for the TMEM load shapes
    that give such a run, when the whole run lies inside the bias at a 16-byte boundary the bias slice is read with one
    vector copy, else (a partial tile's last run, an unaligned run) element by element; the load shapes whose fragment
    interleaves rows and columns read element by element."""
    V = cute.size(frag)
    offs = [
        coords.layout(e) for e in range(V)
    ]  # the fragment's (row, column) offsets relative to element 0
    is_run = all(
        o[0] == offs[0][0] and o[1] == offs[0][1] + e for e, o in enumerate(offs)
    )
    if cutlass.const_expr(is_run):
        n0 = coords[0][1]
        if n0 + V <= N and n0 % 8 == 0:
            base = mBias.iterator.toint() + cutlass.Int64(cute.assume(n0, divby=8)) * 2
            gB = cute.make_tensor(
                cute.make_ptr(
                    cutlass.BFloat16, base, cute.AddressSpace.gmem, assumed_align=16
                ),
                cute.make_layout((V,)),
            )
            rB = cute.make_rmem_tensor((V,), cutlass.BFloat16)
            cute.autovec_copy(gB, rB)
            for e in cutlass.range_constexpr(V):
                frag[e] = chain.apply_pre(CuteF32, frag[e], alpha, CuteF32.widen(rB[e]))
        else:
            bias_per_element(chain, mBias, coords, frag, alpha, N)
    else:
        bias_per_element(chain, mBias, coords, frag, alpha, N)


@cute.jit
def chunk_fold_pass(
    c: cutlass.Int32,
    last: cutlass.Constexpr,
    tiled_copy_t2r,
    tiled_copy_r2t,
    tTR_tAcc: cute.Tensor,
    tTR_tSum: cute.Tensor,
    tTR_tAccSt: cute.Tensor,
    tTR_tSumSt: cute.Tensor,
    rA: cute.Tensor,
    rS: cute.Tensor,
    subtile_cnt: cutlass.Constexpr,
):
    """Fold chunk ``c``'s completed partial (the accumulator stage ``tTR_tAcc``, this thread's partition over the epilogue subtiles)
    into the TMEM running sum (``tTR_tSum``), chunk order from +0: ``c == 0``: ``sum = add(+0, P_0)``; later chunks:
    ``sum = add(sum, P_c)``. For the LAST chunk (``last``) the result is written back into the accumulator stage instead
    (``acc = add(sum, P_c)``), where the ordinary subtile loop finds the tile's fold. All arithmetic is ``CuteF32.add``
    (add.rn.f32); the copies move bits only, through the epilogue's own TMEM load atom and its store twin (``tmem_store_atom_for``),
    so every tile shape of the space (incl. the 64-row tiles' 16-lane data paths) uses the partition its epilogue already uses."""
    for s in range(subtile_cnt):  # the epilogue's subtile loop idiom (not unrolled)
        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, s)], rA)
        if cutlass.const_expr(last):
            cute.copy(tiled_copy_t2r, tTR_tSum[(None, None, None, s)], rS)
            for e in cutlass.range_constexpr(cute.size(rA)):
                rA[e] = CuteF32.add(rS[e], rA[e])  # acc = add(sum, P_{C-1})
            cute.copy(tiled_copy_r2t, rA, tTR_tAccSt[(None, None, None, s)])
        else:
            if c == 0:
                for e in cutlass.range_constexpr(cute.size(rA)):
                    rS[e] = CuteF32.add(
                        CuteF32.const(0.0), rA[e]
                    )  # sum = add(+0, P_0): the chunk fold's init
            else:
                cute.copy(tiled_copy_t2r, tTR_tSum[(None, None, None, s)], rS)
                for e in cutlass.range_constexpr(cute.size(rA)):
                    rS[e] = CuteF32.add(rS[e], rA[e])  # sum = add(sum, P_c)
            cute.copy(tiled_copy_r2t, rS, tTR_tSumSt[(None, None, None, s)])
    cute.arch.fence_view_async_tmem_store()  # the sum / folded tile is in TMEM before anyone reads it


@cute.jit
def epilogue_loop(
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
    epi_tile: cute.Tile,
    acc_pipeline,
    alpha,
    chunks: cutlass.Constexpr = 1,
    mBias: Optional[cute.Tensor] = None,
    tCcC_base: Optional[cute.Tensor] = None,
):
    """``chain`` is the specification's epilogue chain (``spec.EpilogueChain``) and ``alpha`` its runtime scalar (None for
    the unscaled chain). ``chunks > 1``: the in-CTA chunk fold (module docstring) -- the mainloop commits one accumulator
    stage per chunk, the first ``chunks - 1`` are folded into the TMEM running sum (``st.sum_stage``) and released, the
    last one receives the fold and is stored through the ordinary subtile loop below (the chain once, after the fold).
    ``mBias`` (the (N,) bias) with ``tCcC_base`` (the output coordinates partitioned like C): the biased chain, the
    bias of each element's column through the same ``apply_pre`` before the conversion -- after the whole fold, chunked
    or not, so the bias enters exactly once."""
    # ((ATOM_M, MMA_M), (ATOM_N, MMA_N), ...)
    tCgC = transform_partitioned_tensor_layout(tCgC_base)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
    # TMEM -> registers: one epilogue subtile per copy
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
    thr_t2r = tiled_copy_t2r.get_slice(tidx)
    tTR_tAcc_base = thr_t2r.partition_S(tAcc_epi)
    tCgC_epi = cute.flat_divide(
        tCgC, epi_tile
    )  # (EPI_M, EPI_N, SUB_M, SUB_N, RestM, RestN, RestL)
    tTR_gC = thr_t2r.partition_D(tCgC_epi)
    tTR_rAcc = cute.make_rmem_tensor(
        tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, ACC_DTYPE
    )
    tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, st.c_dtype)
    # the biased chain: the output coordinates in the accumulator's partition, so element e of a fragment knows its column
    tTR_cC_all = None
    if cutlass.const_expr(mBias is not None):
        tTR_cC_all = thr_t2r.partition_D(
            cute.flat_divide(transform_partitioned_tensor_layout(tCcC_base), epi_tile)
        )
    # the in-CTA chunk fold: the running sum's TMEM partitions (the stage `st.sum_stage`), the store twin of the load atom, a second fragment
    tiled_copy_r2t = tTR_tSum = tTR_tSumSt = tTR_rSum = None
    if cutlass.const_expr(chunks > 1):
        tiled_copy_r2t = tcgen05.make_tmem_copy(
            tmem_store_atom_for(copy_atom_t2r, ACC_DTYPE),
            tAcc_epi[(None, None, 0, 0, 0)],
        )
        thr_r2t = tiled_copy_r2t.get_slice(tidx)
        tTR_tAccSt_base = thr_r2t.partition_D(tAcc_epi)
        tTR_tSum = cute.group_modes(
            tTR_tAcc_base[(None, None, None, None, None, st.sum_stage)], 3, 5
        )
        tTR_tSumSt = cute.group_modes(
            tTR_tAccSt_base[(None, None, None, None, None, st.sum_stage)], 3, 5
        )
        tTR_rSum = cute.make_rmem_tensor(tTR_rAcc.shape, ACC_DTYPE)
    # registers -> smem
    copy_atom_r2s = get_smem_store_op(
        st.c_layout, st.c_dtype, ACC_DTYPE, tiled_copy_t2r
    )
    tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
    thr_r2s = tiled_copy_r2s.get_slice(tidx)
    tRS_sC = thr_r2s.partition_D(sC)
    tRS_rC = tiled_copy_r2s.retile(tTR_rC)
    # smem -> global (TMA store)
    bSG_sC, bSG_gC_all = cpasync.tma_partition(
        tma_atom_c,
        0,
        cute.make_layout(1),
        cute.group_modes(sC, 0, 2),
        cute.group_modes(tCgC_epi, 0, 2),
    )
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, st.num_acc_stage
    )
    c_pipeline = pipeline.PipelineTmaStore.create(
        num_stages=st.num_c_stage,
        producer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 32 * len(EPILOGUE_WARPS)
        ),
    )
    sync = pipeline.NamedBarrier(
        barrier_id=BAR_EPILOGUE, num_threads=32 * len(EPILOGUE_WARPS)
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        coord = work_tile.tile_idx
        mnl = (coord[0] // atom_thr_size, coord[1], coord[2])
        tile_sched.advance_to_next_work()  # pre-advance: the executed count includes this tile
        work_tile = tile_sched.get_current_work()
        num_tiles_executed = tile_sched.num_tiles_executed
        bSG_gC = bSG_gC_all[(None, None, None, *mnl)]
        if cutlass.const_expr(chunks > 1):
            # chunks 0 .. C-2: fold each completed partial into the TMEM running sum (chunk order from +0), release its stage
            for c in cutlass.range(0, chunks - 1, 1, unroll=1):
                acc_pipeline.consumer_wait(acc_state)  # chunk c's fold is complete
                tAcc_c = cute.group_modes(
                    tTR_tAcc_base[(None, None, None, None, None, acc_state.index)], 3, 5
                )
                tAccSt_c = cute.group_modes(
                    tTR_tAccSt_base[(None, None, None, None, None, acc_state.index)],
                    3,
                    5,
                )
                chunk_fold_pass(
                    c,
                    False,
                    tiled_copy_t2r,
                    tiled_copy_r2t,
                    tAcc_c,
                    tTR_tSum,
                    tAccSt_c,
                    tTR_tSumSt,
                    tTR_rAcc,
                    tTR_rSum,
                    cute.size(tAcc_c.shape, mode=[3]),
                )
                cute.arch.fence_view_async_tmem_load()  # the partial has been read
                sync.arrive_and_wait()
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(
                        acc_state
                    )  # TMEM stage free for the next chunk
                acc_state.advance()
        tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_state.index)]
        acc_pipeline.consumer_wait(
            acc_state
        )  # the fold of this tile (its last chunk) is complete
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        if cutlass.const_expr(chunks > 1):
            # the last chunk: acc = add(sum, P_{C-1}) in place; the subtile loop below stores the class's fold
            tAccSt_c = cute.group_modes(
                tTR_tAccSt_base[(None, None, None, None, None, acc_state.index)], 3, 5
            )
            chunk_fold_pass(
                cutlass.Int32(chunks - 1),
                True,
                tiled_copy_t2r,
                tiled_copy_r2t,
                tTR_tAcc,
                tTR_tSum,
                tAccSt_c,
                tTR_tSumSt,
                tTR_rAcc,
                tTR_rSum,
                cute.size(tTR_tAcc.shape, mode=[3]),
            )
        bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        num_prev_subtiles = num_tiles_executed * subtile_cnt
        tTR_cC = None
        if cutlass.const_expr(mBias is not None):
            tTR_cC = cute.group_modes(
                tTR_cC_all[(None, None, None, None, None, *mnl)], 3, 5
            )  # this tile's coordinates per subtile
        for s in range(subtile_cnt):
            cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, s)], tTR_rAcc)
            if cutlass.const_expr(mBias is not None):
                apply_chain_biased(
                    chain,
                    mBias,
                    tTR_cC[(None, None, None, s)],
                    tTR_rAcc,
                    alpha,
                    mBias.shape[0],
                )  # the chain's fp32 ops with the column's bias, once, after the fold
            else:
                apply_chain_pre_cvt(
                    chain, tTR_rAcc, alpha
                )  # the chain's fp32 ops (mul.rn alpha when scaled)
            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
            tRS_rC.store(
                acc_vec.to(st.c_dtype)
            )  # the chain's cvt.rn <out>, exactly once
            c_buffer = (num_prev_subtiles + s) % st.num_c_stage
            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
            cute.arch.fence_proxy("async.shared", space="cta")
            sync.arrive_and_wait()
            if warp_idx == EPILOGUE_WARPS[0]:
                cute.copy(tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, s)])
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()
            sync.arrive_and_wait()
        sync.arrive_and_wait()
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(
                acc_state
            )  # TMEM buffer free for the next fold
        acc_state.advance()
    c_pipeline.producer_tail()
