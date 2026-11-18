# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Exact radix selection for typed block scores and sparse candidates."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from .recovery_selection import select
from .candidate_publication_sort import publication_sort


class SparseSelector:
    def __init__(self, dtype, recovery=False):
        self.dtype = dtype
        self.recovery = recovery

    @cute.jit
    def __call__(self, scores, counts, lengths, blocks, out, stream: cuda.CUstream):
        self.kernel(scores, counts, lengths, blocks, out).launch(
            grid=(out.shape[0], 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, scores, counts, lengths, blocks, out):
        row = cute.arch.block_idx()[0]
        tid = cute.arch.thread_idx()[0]
        smem = cutlass.utils.SmemAllocator()
        hist = smem.allocate_tensor(cutlass.Int32, cute.make_layout(256))
        scalars = smem.allocate_tensor(cutlass.Int32, cute.make_layout(8))
        required = min(512, lengths[row])
        count = counts[row]
        failed = count < required or count > 4096
        execute = failed if cutlass.const_expr(self.recovery) else not failed
        if execute:
            if required > 0:
                extent = lengths[row] if cutlass.const_expr(self.recovery) else count
                kind = cutlass.const_expr("dense" if self.recovery else "linear")
                select(
                    row,
                    tid,
                    scores,
                    cutlass.Int32(0),
                    extent,
                    required,
                    hist,
                    scalars,
                    out,
                    lengths,
                    512,
                    kind,
                    self.dtype,
                    False,
                )
                cute.arch.barrier()
                for i in cutlass.range(tid, 512, 256):
                    local = out[row, i]
                    if local >= 0:
                        out[row, i] = blocks[row, local // 8] * 8 + local % 8
            else:
                for i in cutlass.range(tid, 512, 256):
                    out[row, i] = -1


class Float32BlockSelector:
    """Sample-filter block TopK followed by sorted candidate publication."""

    def __init__(self, columns):
        from flashinfer.topk_varlen.kernels.gvr2_topk_decode import GvrMainKernel

        self.words = (columns + 8191) // 8192
        self.selector = GvrMainKernel(
            256,
            4,
            4,
            256,
            8,
            False,
            varlen=True,
            hint_free=True,
            prefill=True,
            prefill_abs=True,
        )

    @cute.jit
    def __call__(self, scores, ends, out, lengths, starts, stream: cuda.CUstream):
        self.bounds(ends, out, lengths, starts).launch(
            grid=((out.shape[0] + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
        )
        block_starts = cute.make_tensor(
            out.iterator, cute.make_layout(out.shape[0], stride=out.stride[0])
        )
        self.selector.kern(
            scores,
            block_starts,
            out,
            lengths,
            cutlass.Int32(scores.shape[1]),
            cutlass.Int32(scores.shape[1]),
            cutlass.Int32(2048),
            cutlass.Int32(8192),
            cutlass.Int32(1024),
            cutlass.Int32(1),
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
            lengths,
            cutlass.Int32(2816),
            cutlass.Int32(64),
            cutlass.Int32(4096),
            cutlass.Int32(1),
            cutlass.Int32(0),
        ).launch(
            grid=(1, out.shape[0], 1),
            block=(256, 1, 1),
            min_blocks_per_mp=4,
            stream=stream,
        )
        self.finish(ends, out, lengths, starts).launch(
            grid=(out.shape[0], 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def bounds(self, ends, out, lengths, starts):
        row = cute.arch.block_idx()[0] * 256 + cute.arch.thread_idx()[0]
        if row < out.shape[0]:
            first = starts[row] // 8
            out[row, 0] = first
            lengths[row] = max(0, (ends[row] + 7) // 8 - first)

    @cute.kernel
    def finish(self, ends, out, lengths, starts):
        row = cute.arch.block_idx()[0]
        tid = cute.arch.thread_idx()[0]
        smem = cutlass.utils.SmemAllocator()
        bitmap = smem.allocate_tensor(
            cutlass.Uint32, cute.make_layout(256 * self.words)
        )
        prefix = smem.allocate_tensor(cutlass.Int32, cute.make_layout(8))
        required = min(2048, max(0, (ends[row] + 7) // 8 - starts[row] // 8))
        publication_sort(
            row,
            tid,
            out,
            bitmap,
            prefix,
            self.words,
            (ends[row] - 1) // 8 if required > 0 else -1,
        )
        partial = ends[row] % 8
        if tid == 0:
            lengths[row] = required * 8 - (
                8 - partial if partial != 0 and required > 0 else 0
            )
