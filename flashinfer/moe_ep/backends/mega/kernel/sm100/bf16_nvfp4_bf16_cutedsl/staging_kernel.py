# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""BF16 copy plus routing repack and full tail masking in one launch."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute


class Bf16InputStage:
    _threads = 128

    def __init__(self, hidden: int, topk: int) -> None:
        self.hidden = hidden
        self.topk = topk

    @cute.jit
    def __call__(
        self,
        hidden: cute.Tensor,
        ids: cute.Tensor,
        weights: cute.Tensor,
        x: cute.Tensor,
        ids_out: cute.Tensor,
        weights_out: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        self.stage(hidden, ids, weights, x, ids_out, weights_out).launch(
            grid=(x.shape[0], 1, 1), block=(self._threads, 1, 1), stream=stream
        )

    @cute.jit
    def _aligned(self, tensor: cute.Tensor) -> cute.Tensor:
        # Same proven-chunk retagging as DataPreprocess._mark_alignment.
        p = tensor.iterator
        return cute.make_tensor(
            cute.make_ptr(p.dtype, p.toint(), p.memspace, assumed_align=16),
            tensor.layout,
        )

    @cute.kernel
    def stage(
        self,
        hidden: cute.Tensor,
        ids: cute.Tensor,
        weights: cute.Tensor,
        x: cute.Tensor,
        ids_out: cute.Tensor,
        weights_out: cute.Tensor,
    ) -> None:
        row = cute.arch.block_idx()[0]
        tid = cute.arch.thread_idx()[0]
        width: cutlass.Constexpr[int] = self.hidden
        topk: cutlass.Constexpr[int] = self.topk
        if row < hidden.shape[0]:
            # DataPreprocess's eight-BF16 vector-load pattern, with a matching
            # direct global store; no conversion, reduction or shared memory.
            source = cute.zipped_divide(hidden[row, None], (8,))
            target = cute.zipped_divide(x[row, None], (8,))
            atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
            )
            chunk = tid
            while chunk < width // 8:
                values = cute.make_rmem_tensor((8,), cutlass.BFloat16)
                cute.copy(atom, self._aligned(source[(None,), (chunk,)]), values)
                cute.copy(atom, values, self._aligned(target[(None,), (chunk,)]))
                chunk += cutlass.Int32(self._threads)
            # DataPreprocess's one-lane-per-route int64 repack. FP32 weights
            # are copied unchanged, including negative weights and sentinels.
            if tid < topk:
                ids_out[row, tid] = cutlass.Int64(ids[row, tid])
                weights_out[row, tid] = weights[row, tid]
        else:
            if tid < topk:
                ids_out[row, tid] = cutlass.Int64(-1)
            # Match torch staging: x and weights outside the live prefix
            # retain their previous contents; only tail routing IDs change.
