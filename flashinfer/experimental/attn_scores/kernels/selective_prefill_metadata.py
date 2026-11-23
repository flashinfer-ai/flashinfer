# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Per-layer numerical validation and causal bounds for single-request prefill."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.utils.smem_allocator import SmemAllocator


class SelectivePrefillMetadata:
    """Reduce live contract violations, then publish fail-closed row bounds.

    Step validation already supplies a contiguous prefix of active rows and
    monotone causal ends. Padding therefore needs the last live end, not a
    prefix scan. A separate launch makes all contract flags visible before any
    row is published; flags are overwritten on every layer and replay.
    With validation, publish is a programmatic dependent launch: it loads its
    row bound while validate runs and waits for validate's completion before it
    reads any flag or writes a row, which hides the second launch's latency.
    Validate itself must stay an ordinary launch: it then starts only after the
    work that fills active/row_ends has completed, which is what makes
    publish's reads ahead of its wait safe. pdl=False launches both normally.
    """

    def __init__(
        self, heads, qwen, fast, block_scaled, check_ctas=256, top_k=512, pdl=True
    ):
        self.top_k = top_k
        self.heads = heads
        self.qwen = qwen
        self.fast = fast
        self.block_scaled = block_scaled
        self.check = qwen or fast
        self.check_ctas = check_ctas
        # Without validate there is no predecessor in this module to depend on.
        self.pdl = self.check and bool(pdl)

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
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.check):
            # Not a dependent launch (see the class docstring): publish relies on it.
            self.validate(weights, scales, q_scales, active, physical_kv, flags).launch(
                grid=(self.check_ctas, 1, 1),
                block=(256, 1, 1),
                stream=stream,
                use_pdl=False,
            )
        self.publish(active, row_ends, starts, ends, score_ends, flags, error).launch(
            grid=(cute.ceil_div(ends.shape[0], 32), 1, 1),
            block=(32, 1, 1),
            stream=stream,
            use_pdl=self.pdl,
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
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        warp = tid // 32
        block, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()
        bad = cutlass.Int32(0)
        # Eight warps per CTA cover live weights/scales without the long
        # strided loop of a single-warp CTA. Publish one flag per CTA.
        index = block * 256 + tid
        live_weights = active[0] * self.heads
        while index < live_weights:
            w = weights[index]
            if cutlass.const_expr(self.qwen):
                # Compare against the representable input constant. The DSL
                # can fold a BF16 -> FP32 constant cast before rounding it.
                expected = (
                    181 / 2048
                    if cutlass.const_expr(weights.element_type == cutlass.BFloat16)
                    else 128**-0.5
                )
                good = cutlass.Float32(w) == cutlass.Float32(expected)
            else:
                magnitude = abs(w)
                good = ((magnitude == 0) | (magnitude >= cutlass.Float32(2**-125))) & (
                    magnitude < cutlass.Float32(float("inf"))
                )
            bad = bad | cutlass.Int32(not good)
            if cutlass.const_expr(self.fast and self.block_scaled):
                bits = cutlass.Uint32(q_scales[index])
                for shift in cutlass.range_constexpr(4):
                    bad = bad | cutlass.Int32(((bits >> (shift * 8)) & 255) > 159)
            index += self.check_ctas * 256
        if cutlass.const_expr(
            (self.qwen and not self.block_scaled) or (self.fast and self.block_scaled)
        ):
            index = block * 256 + tid
            while index < physical_kv[1]:
                if cutlass.const_expr(self.block_scaled):
                    bits = cutlass.Uint32(scales[index])
                    for shift in cutlass.range_constexpr(4):
                        bad = bad | cutlass.Int32(((bits >> (shift * 8)) & 255) > 159)
                else:
                    bad = bad | cutlass.Int32(scales[index] != cutlass.Float32(1))
                index += self.check_ctas * 256
        bad = cute.arch.warp_redux_sync(bad, "max")
        warp_flags = SmemAllocator().allocate_tensor(
            cutlass.Int32, cute.make_layout(8), byte_alignment=4
        )
        if lane == 0:
            warp_flags[warp] = bad
        cute.arch.barrier()
        if warp == 0:
            bad = cutlass.Int32(0)
            if lane < 8:
                bad = warp_flags[lane]
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
        score_ends: cute.Tensor,
        flags: cute.Tensor,
        error: cute.Tensor,
    ):
        lane, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.pdl):
            # Validate is an ordinary launch, so everything that fills active/row_ends completed before it
            # started and therefore before this kernel. Flags and the row outputs (still read by the
            # previous layer's K4) are touched only after the wait.
            pre_row = block * 32 + lane
            pre_live = active[0]
            pre_end = cutlass.Int32(0)
            if pre_row < ends.shape[0]:
                pre_end = cutlass.Int32(row_ends[min(pre_row, max(pre_live - 1, 0))])
            cute.arch.griddepcontrol_wait()
        bad = cutlass.Int32(0)
        if cutlass.const_expr(self.check):
            for offset in cutlass.range_constexpr(cute.ceil_div(self.check_ctas, 32)):
                index = offset * 32 + lane
                if index < self.check_ctas:
                    bad = bad | flags[index]
            bad = cute.arch.warp_redux_sync(bad, "max")
        row = block * 32 + lane
        if row < ends.shape[0]:
            if cutlass.const_expr(self.pdl):
                live = pre_live
                end = pre_end
            else:
                live = active[0]
                end = row_ends[min(row, max(live - 1, 0))]
            starts[row] = cutlass.Int32(0) if row < live or bad != 0 else end
            bound = end if bad == 0 else cutlass.Int32(0)
            ends[row] = bound
            start = cutlass.Int32(0) if row < live or bad != 0 else end
            # K3 emits all keys directly for these rows. K1/K2 see an empty
            # scoring interval, while K3/K4 retain the original valid bounds.
            score_ends[row] = bound if bound - start > self.top_k else start
        if block == 0 and lane == 0:
            error[0] = error[0] | (bad * 16)
