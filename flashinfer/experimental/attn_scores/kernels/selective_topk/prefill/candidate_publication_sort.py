# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Counting sort for unique block IDs in a bounded KV domain."""

import cutlass
import cutlass.cute as cute


@cute.jit
def publication_sort(
    row, tid, ids, bitmap, prefix, words_per_thread: cutlass.Constexpr, last
):
    lane = cute.arch.lane_idx()
    for i in cutlass.range(tid, cute.size(bitmap), 256):
        bitmap[i] = cutlass.Uint32(0)
    cute.arch.barrier()
    for i in cutlass.range(tid, ids.shape[1], 256):
        index = ids[row, i]
        if index >= 0 and index < cute.size(bitmap) * 32:
            cute.arch.atomic_or(
                (bitmap.iterator + index // 32).llvm_ptr,
                cutlass.Uint32(1) << (index % 32),
                sem="relaxed",
                scope="cta",
            )
    cute.arch.barrier()
    if tid == 0 and last >= 0 and last < cute.size(bitmap) * 32:
        present = bitmap[last // 32] & (cutlass.Uint32(1) << (last % 32))
        if present == 0:
            removed = ids[row, 0]
            if removed >= 0 and removed < cute.size(bitmap) * 32:
                bitmap[removed // 32] &= ~(cutlass.Uint32(1) << (removed % 32))
            bitmap[last // 32] |= cutlass.Uint32(1) << (last % 32)
    cute.arch.barrier()
    words = cute.make_rmem_tensor((words_per_thread,), cutlass.Uint32)
    total = cutlass.Int32(0)
    for i in cutlass.range_constexpr(words_per_thread):
        words[i] = bitmap[tid * words_per_thread + i]
        total += cutlass.Int32(cute.arch.popc(words[i]))
    inclusive = total
    for bit in cutlass.range_constexpr(5):
        delta = 1 << bit
        other = cute.arch.shuffle_sync_up(inclusive, delta, mask=-1, mask_and_clamp=0)
        if lane >= delta:
            inclusive += other
    if lane == 31:
        prefix[tid // 32] = inclusive
    cute.arch.barrier()
    position = inclusive - total
    for warp in cutlass.range_constexpr(8):
        if warp < tid // 32:
            position += prefix[warp]
    for i in cutlass.range_constexpr(words_per_thread):
        word = words[i]
        while word != 0:
            lowest = word & (~word + cutlass.Uint32(1))
            bit = cutlass.Int32(cute.arch.popc(lowest - 1))
            ids[row, position] = (tid * words_per_thread + i) * 32 + bit
            position += 1
            word &= word - 1
    cute.arch.barrier()
    total = cutlass.Int32(0)
    for warp in cutlass.range_constexpr(8):
        total += prefix[warp]
    for i in cutlass.range(total + tid, ids.shape[1], 256):
        ids[row, i] = 2147483647
