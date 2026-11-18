# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Publish causal and candidate-domain bounds in one launch."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda

from ....selective_prefill_metadata import SelectivePrefillMetadata


class SparseSelectivePrefillMetadata(SelectivePrefillMetadata):
    @cute.jit
    def __call__(
        self,
        weights: cute.Tensor,
        scales: cute.Tensor,
        q_scales: cute.Tensor,
        active: cute.Tensor,
        physical_kv: cute.Tensor,
        row_ends: cute.Tensor,
        starts: cute.Tensor,
        ends: cute.Tensor,
        score_ends: cute.Tensor,
        flags: cute.Tensor,
        error: cute.Tensor,
        candidate_ends: cute.Tensor,
        candidate_lens: cute.Tensor,
        enabled: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.check):
            self.validate(weights, scales, q_scales, active, physical_kv, flags).launch(
                grid=(self.check_ctas, 1, 1), block=(256, 1, 1), stream=stream
            )
        self.publish(
            active,
            row_ends,
            starts,
            ends,
            score_ends,
            flags,
            error,
            candidate_ends,
            candidate_lens,
            enabled,
        ).launch(
            grid=(cute.ceil_div(ends.shape[0], 32), 1, 1),
            block=(32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def publish(
        self,
        active: cute.Tensor,
        row_ends: cute.Tensor,
        starts: cute.Tensor,
        ends: cute.Tensor,
        score_ends: cute.Tensor,
        flags: cute.Tensor,
        error: cute.Tensor,
        candidate_ends: cute.Tensor,
        candidate_lens: cute.Tensor,
        enabled: cutlass.Int32,
    ):
        lane, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        bad = cutlass.Int32(0)
        if cutlass.const_expr(self.check):
            for offset in cutlass.range_constexpr(cute.ceil_div(self.check_ctas, 32)):
                index = offset * 32 + lane
                if index < self.check_ctas:
                    bad = bad | flags[index]
            bad = cute.arch.warp_redux_sync(bad, "max")
        row = block * 32 + lane
        if row < ends.shape[0]:
            live = active[0]
            end = row_ends[min(row, max(live - 1, 0))]
            starts[row] = cutlass.Int32(0) if row < live or bad != 0 else end
            bound = end if bad == 0 else cutlass.Int32(0)
            ends[row] = bound
            start = cutlass.Int32(0) if row < live or bad != 0 else end
            # K3 emits all keys directly for these rows. K1/K2 see an empty
            # scoring interval, while K3/K4 retain the original valid bounds.
            score_ends[row] = bound if bound - start > self.top_k else start
            candidate_end = (
                bound if (bound > start) & (enabled != 0) else cutlass.Int32(0)
            )
            kept = min((candidate_end + 7) // 8, 2048)
            candidate_ends[row] = candidate_end
            candidate_lens[row] = (
                ((kept - 1) * 8 + (candidate_end - 1) % 8 + 1)
                if candidate_end > 0
                else cutlass.Int32(0)
            )

        if block == 0 and lane == 0:
            error[0] = error[0] | (bad * 16)
