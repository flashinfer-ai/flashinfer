# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Per-layer numerical validation and causal bounds for single-request v2."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda


class SelectiveV2Metadata:
    """Reduce live contract violations, then publish fail-closed row bounds.

    Step validation already supplies a contiguous prefix of active rows and
    monotone causal ends. Padding therefore needs the last live end, not a
    prefix scan. A separate launch makes all contract flags visible before any
    row is published; flags are overwritten on every layer and replay.
    """

    def __init__(self, heads, qwen, fast, fp4, check_ctas=256):
        self.heads = heads
        self.qwen = qwen
        self.fast = fast
        self.fp4 = fp4
        self.check = qwen or fast
        self.check_ctas = check_ctas

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
        flags: cute.Tensor,
        error: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.check):
            self.validate(weights, scales, q_scales, active, physical_kv, flags).launch(
                grid=(self.check_ctas, 1, 1), block=(32, 1, 1), stream=stream
            )
        self.publish(active, row_ends, starts, ends, flags, error).launch(
            grid=(cute.ceil_div(ends.shape[0], 32), 1, 1),
            block=(32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def validate(
        self,
        weights: cute.Tensor,
        scales: cute.Tensor,
        q_scales: cute.Tensor,
        active: cute.Tensor,
        physical_kv: cute.Tensor,
        flags: cute.Tensor,
    ):
        lane, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        bad = cutlass.Int32(0)
        index = block * 32 + lane
        live_weights = active[0] * self.heads
        while index < live_weights:
            w = weights[index]
            if cutlass.const_expr(self.qwen):
                good = w == cutlass.Float32(128**-0.5)
            else:
                magnitude = abs(w)
                good = ((magnitude == 0) | (magnitude >= cutlass.Float32(2**-125))) & (
                    magnitude < cutlass.Float32(float("inf"))
                )
            bad = bad | cutlass.Int32(not good)
            if cutlass.const_expr(self.fast and self.fp4):
                bits = cutlass.Uint32(q_scales[index])
                for shift in cutlass.range_constexpr(4):
                    bad = bad | cutlass.Int32(((bits >> (shift * 8)) & 255) > 159)
            index += self.check_ctas * 32
        if cutlass.const_expr((self.qwen and not self.fp4) or (self.fast and self.fp4)):
            index = block * 32 + lane
            while index < physical_kv[1]:
                if cutlass.const_expr(self.fp4):
                    bits = cutlass.Uint32(scales[index])
                    for shift in cutlass.range_constexpr(4):
                        bad = bad | cutlass.Int32(((bits >> (shift * 8)) & 255) > 159)
                else:
                    bad = bad | cutlass.Int32(scales[index] != cutlass.Float32(1))
                index += self.check_ctas * 32
        bad = cute.arch.warp_redux_sync(bad, "max")
        if lane == 0:
            flags[block] = bad

    @cute.kernel
    def publish(
        self,
        active: cute.Tensor,
        row_ends: cute.Tensor,
        starts: cute.Tensor,
        ends: cute.Tensor,
        flags: cute.Tensor,
        error: cute.Tensor,
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
            ends[row] = end if bad == 0 else cutlass.Int32(0)
        if block == 0 and lane == 0:
            error[0] = error[0] | (bad * 16)
