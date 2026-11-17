# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""TMA producer block: one warp streams the A and B k-tiles of each owned output tile into the
smem ring, k-tiles ascending, one ring slot per k-tile, multicast across the cluster when the
knobs say so. The order here is the order the MMA block consumes; it carries no arithmetic.

``chunks > 1`` (the in-CTA chunk fold, ``Knobs.cta_chunks``): the operands are the split class's
``(rows, Kc, C)`` views (the batch mode is the K chunk, as for the workspace split), and per tile
the k-tiles of chunk 0, then of chunk 1, ... are streamed, chunks ascending: the order the MMA block
folds them in.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute


@cute.jit
def tma_produce_ktiles(
    k_tile_cnt: cutlass.Int32,
    ab_producer,
    tma_atom_a: cute.CopyAtom,
    tma_atom_b: cute.CopyAtom,
    tAgA_slice: cute.Tensor,
    tBgB_slice: cute.Tensor,
    tAsA: cute.Tensor,
    tBsB: cute.Tensor,
    a_mcast_mask,
    b_mcast_mask,
):
    """The k-tiles of one tile (or of one chunk of a tile), ascending, into the ring. Returns the producer (loop-carried)."""
    ab_producer.reset()
    peek_empty = ab_producer.try_acquire()
    for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
        handle = ab_producer.acquire_and_advance(
            peek_empty
        )  # slot empty; leader arms the tx barrier
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
    return ab_producer


@cute.jit
def tma_producer_loop(
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
    chunks: cutlass.Constexpr = 1,
):
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        coord = work_tile.tile_idx
        if cutlass.const_expr(chunks == 1):
            # ((atom_v, rest_v), RestK): this tile's A rows / B columns, all k-tiles
            tAgA_slice = tAgA[(None, coord[0] // atom_thr_size, None, coord[2])]
            tBgB_slice = tBgB[(None, coord[1], None, coord[2])]
            ab_producer = tma_produce_ktiles(
                k_tile_cnt,
                ab_producer,
                tma_atom_a,
                tma_atom_b,
                tAgA_slice,
                tBgB_slice,
                tAsA,
                tBsB,
                a_mcast_mask,
                b_mcast_mask,
            )
        else:
            for c in cutlass.range(
                0, chunks, 1, unroll=1
            ):  # chunks ascending: the batch mode of the split views
                tAgA_slice = tAgA[(None, coord[0] // atom_thr_size, None, c)]
                tBgB_slice = tBgB[(None, coord[1], None, c)]
                ab_producer = tma_produce_ktiles(
                    k_tile_cnt,
                    ab_producer,
                    tma_atom_a,
                    tma_atom_b,
                    tAgA_slice,
                    tBgB_slice,
                    tAsA,
                    tBsB,
                    a_mcast_mask,
                    b_mcast_mask,
                )
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    ab_producer.tail()
