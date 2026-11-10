# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The mainloop block: the ``kstep`` fold of the spec, issued by its blocked lowering.

The spec's fold has a runtime trip count (K is a runtime argument), so the kernel issues it
through ``canon.BlockedFold`` (``MainloopBlock.lowering()``): blocks of ``inner`` instruction
K-steps (one smem stage), blocks ascending, instructions ascending inside a block, the fold's
first instruction with ``ACCUMULATE`` off (``D_0 = +0``: the digit's init; TMEM is never
cleared) and every later one accumulating: ``D_{t+1} = MMA(D_t, A_t, B_t)``. One ``cute.gemm``
here is one ``tcgen05.mma`` K-step, the opaque digit below which nothing is ours to reorder. The
zero-filled K remainder of the last block is an identity step (Lean ``fold_pad``).

What this file adds to the lowering is the loop idiom over blocks (``for k_tile in range(n)``,
ascending) and the step op; the pipeline hand-offs (wait for the stage, release it, publish the
accumulator) are placements around the fold. The leader CTA of a 2-CTA pair issues for both.

``chunks > 1`` (the in-CTA chunk fold of the split classes, ``Knobs.cta_chunks``): per tile the
fold above is issued once per chunk, chunks ascending, each from a reset accumulator (the fold's
first step has ``ACCUMULATE`` off) into the next accumulator stage, and each chunk's stage is
committed to the epilogue, which folds the chunk partials in order (``blocks/epilogue.py``). The
per-element fold of a chunk is the class's `kstep` digit at that chunk exactly as the workspace
split computes it on another CTA; nothing in the issue order changes.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import tcgen05


@cute.jit
def mma_fold(
    lowering,
    k_tile_cnt: cutlass.Int32,
    is_leader_cta,
    tiled_mma: cute.TiledMma,
    ab_consumer,
    acc_pipeline,
    acc_state,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCtAcc_base: cute.Tensor,
):
    """One ``kstep`` fold (a tile, or one chunk of a tile) into the accumulator stage ``acc_state`` points at, then the
    stage's commit to the epilogue. Returns the participants it advanced (the DSL carries a loop's objects only through
    names the loop body stores)."""
    tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
    ab_consumer.reset()
    peek_full = cutlass.Boolean(1)
    if is_leader_cta:
        peek_full = ab_consumer.try_wait()
    if is_leader_cta:
        acc_pipeline.producer_acquire(
            acc_state
        )  # the epilogue has drained this TMEM buffer
    for k_tile in range(k_tile_cnt):  # blocks ascending (the lowering's outer loop)
        if is_leader_cta:
            handle = ab_consumer.wait_and_advance(peek_full)
            for (
                j
            ) in lowering.inner_steps():  # the lowering's issue order inside a block
                tiled_mma.set(
                    tcgen05.Field.ACCUMULATE, lowering.accumulate(k_tile, j)
                )  # off on the fold's first step only
                crd = (None, None, j, handle.index)
                cute.gemm(
                    tiled_mma, tCtAcc, tCrA[crd], tCrB[crd], tCtAcc
                )  # one fold step
            handle.release()
            peek_full = cutlass.Boolean(1)
            if handle.count + 1 < k_tile_cnt:
                peek_full = ab_consumer.try_wait()
    if is_leader_cta:
        acc_pipeline.producer_commit(
            acc_state
        )  # accumulator complete: hand to the epilogue
    acc_state.advance()
    return ab_consumer, acc_state


@cute.jit
def mma_loop(
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
    chunks: cutlass.Constexpr = 1,
):
    """``chunks``: 1 (one fold per tile), or the class's chunk count for the in-CTA chunk fold (``k_tile_cnt`` is then the
    k-tiles of ONE chunk and the fold is issued per chunk, chunks ascending, each committed to the epilogue)."""
    assert cute.size(tCrA, mode=[2]) == lowering.inner, (
        "one smem stage must hold exactly the lowering's inner steps"
    )
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, num_acc_stage
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        if cutlass.const_expr(chunks == 1):
            ab_consumer, acc_state = mma_fold(
                lowering,
                k_tile_cnt,
                is_leader_cta,
                tiled_mma,
                ab_consumer,
                acc_pipeline,
                acc_state,
                tCrA,
                tCrB,
                tCtAcc_base,
            )
        else:
            for _c in cutlass.range(
                0, chunks, 1, unroll=1
            ):  # chunks ascending, each a fold from +0 into the next stage
                ab_consumer, acc_state = mma_fold(
                    lowering,
                    k_tile_cnt,
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
