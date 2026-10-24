# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Split-K reduction + epilogue.

Kernel 1 writes one FP32 ``[M, N]`` partial plane per ``(batch, split)`` into
workspace laid out as ``[batch * split][M][N]``. This kernel reduces those
planes as follows:

* A 128-thread CTA contains four warps; each warp owns one output row segment.
* A warp is divided into 1, 2, or 4 lane groups according to the split count.
  Lanes with the same index inside their groups own the same output vector;
  group ``g`` loads slices ``g, g + groups, g + 2 * groups, ...``.
* Each lane sums its slices in a fixed order. For an uneven slice count, groups
  without a tail slice contribute a zero vector so all lanes can participate
  in the full-warp shuffle.
* A fixed ``shfl.down`` tree merges the group sums. Only group 0 passes the
  complete FP32 result to the generated epilogue for conversion and output.

With PDL enabled, every reducer CTA waits before reading the partial workspace.
The fixed slice assignment and shuffle tree avoid atomics and make the reduction
bit-reproducible for a fixed configuration.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm

SPLITK_REDUCE_THREADS = 128
SPLITK_REDUCE_THREADS_PER_ROW = 32  # one warp per row segment
SPLITK_REDUCE_TILE_M = SPLITK_REDUCE_THREADS // SPLITK_REDUCE_THREADS_PER_ROW
SPLITK_REDUCE_SLICES_PER_THREAD = 4
# Limit shuffle depth and preserve wide contiguous row accesses per slice.
SPLITK_REDUCE_MAX_SLICE_GROUPS = 4


def splitk_reduce_slice_groups(split_k_slices: int) -> int:
    """Choose 1, 2, or 4 lane groups for one warp.

    Double the group count while a lane owns more than four slices, stopping
    at four groups: S <= 4 -> 1, 5 <= S <= 8 -> 2, and S >= 9 -> 4.
    """

    groups = 1
    while groups < SPLITK_REDUCE_MAX_SLICE_GROUPS and split_k_slices > SPLITK_REDUCE_SLICES_PER_THREAD * groups:
        groups *= 2
    return groups


def splitk_reduce_tile_n(split_k_slices: int, splitk_reduce_elems: int) -> int:
    """Return the output columns covered by each CTA row segment."""
    return (SPLITK_REDUCE_THREADS_PER_ROW // splitk_reduce_slice_groups(split_k_slices)) * splitk_reduce_elems


def _splitk_sum(vals):
    """Build a fixed-order sum of one lane's slice vectors at trace time."""
    total = vals[0]
    for v in vals[1:]:
        total = total + v
    return total


def _splitk_group_tree(acc, groups, lanes_per_group):
    """Merge lane-group sums with a fixed warp-shuffle tree.

    Group ``g`` absorbs ``g + d`` for ``d = groups / 2, ..., 1``; group 0
    therefore holds the complete sum that is passed to the epilogue.
    """
    d = groups // 2
    while d >= 1:
        parts = [nvvm.shfl_sync(0xFFFFFFFF, e, d * lanes_per_group, 0x1F, kind=nvvm.Shfl.DOWN) for e in acc.to_elements()]
        acc = acc + cutlass.Vector.from_elements(tuple(parts), cutlass.Float32)
        d //= 2
    return acc


@cute.kernel
def _splitk_reduce_kernel(
    m: cutlass.Int64,
    n: cutlass.Int64,
    batch: cutlass.Int64,
    mSplitK_partials: cute.Tensor,
    taps: tuple,  # the graph's output tensors
    strides: tuple,
    aux: tuple,  # the epilogue's input tensors
    epilogue: cutlass.Constexpr,
    split_k_slices: cutlass.Constexpr,
    splitk_reduce_elems: cutlass.Constexpr,
    use_pdl: cutlass.Constexpr,
) -> None:
    groups = splitk_reduce_slice_groups(split_k_slices)
    lanes_per_group = SPLITK_REDUCE_THREADS_PER_ROW // groups
    tile_n = lanes_per_group * splitk_reduce_elems
    rounds = split_k_slices // groups
    tail = split_k_slices - rounds * groups
    tidx = cute.arch.thread_idx()[0]
    lane = tidx % SPLITK_REDUCE_THREADS_PER_ROW
    group = lane // lanes_per_group
    row = cutlass.Int64(cute.arch.block_idx()[0]) * SPLITK_REDUCE_TILE_M + tidx // SPLITK_REDUCE_THREADS_PER_ROW
    col = cutlass.Int64(cute.arch.block_idx()[1]) * tile_n + (lane % lanes_per_group) * splitk_reduce_elems
    batch_idx = cutlass.Int64(cute.arch.block_idx()[2])
    if cutlass.const_expr(use_pdl):
        nvvm.griddepcontrol("wait")
    elems_per_partial = m * n
    base_ptr = mSplitK_partials.iterator.raw_ptr()
    # Zero accumulator: every lane takes part in the shuffles below.
    acc = cutlass.full((splitk_reduce_elems,), 0.0, cutlass.Float32)
    # `splitk_reduce_elems` divides n, so a thread's group never crosses a row.
    valid = (row < m) & (col < n)
    if valid:
        # partials layout: [batch*split][M][N]; this lane group takes slices
        # group, group + groups, ...
        partials_ptr = base_ptr + (batch_idx * split_k_slices * m + row) * n + col + cutlass.Int64(group) * elems_per_partial
        vals = [(partials_ptr + i * groups * elems_per_partial).load(count=splitk_reduce_elems, alignment=splitk_reduce_elems * 4) for i in range(rounds)]
        if cutlass.const_expr(tail > 0):
            last = acc
            if group < tail:
                last = (partials_ptr + rounds * groups * elems_per_partial).load(count=splitk_reduce_elems, alignment=splitk_reduce_elems * 4)
            vals.append(last)
        acc = _splitk_sum(vals)
    if cutlass.const_expr(groups > 1):
        acc = _splitk_group_tree(acc, groups, lanes_per_group)
        valid = valid & (group == 0)
    if valid:
        epilogue(acc, row, col, batch_idx, m, n, splitk_reduce_elems, taps, strides, aux)


_splitk_reduce_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
