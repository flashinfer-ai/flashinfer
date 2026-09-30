"""Compact W4A8 FC1 boundary activation and MXFP8 materialization."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64, Uint8, Uint32

from b12x._lib.intrinsics import (
    FLOAT8_E4M3_MAX,
    cvt_f32x4_to_e4m3x4,
    fabs_f32,
    fmax_f32,
    pow2_ceil_ue8m0,
    ue8m0_to_output_scale,
)


class W4A8CompactMicroActivationKernel:
    """Apply routed SwiGLU to BF16 FC1 projections and materialize MXFP8 rows.

    One CTA owns one routed pair and one N128 projection tile.  Its four
    eight-lane subgroups independently quantize the tile's four K32 groups.
    Each routed pair owns one compact materialized row.
    """

    def __init__(
        self,
        n: int,
        num_experts: int,
        *,
        swiglu_limit: float | None,
        fast_math: bool,
    ):
        self.n = int(n)
        self.num_experts = int(num_experts)
        self.has_swiglu_limit = swiglu_limit is not None
        self.swiglu_limit = 0.0 if swiglu_limit is None else float(swiglu_limit)
        self.fast_math = bool(fast_math)
        if self.n < 64 or self.n % 64:
            raise ValueError(
                "compact W4A8 micro activation requires N divisible by 64"
            )
        if self.num_experts < 1:
            raise ValueError(
                "compact W4A8 micro activation requires at least one expert"
            )
        self.n_tiles = (self.n + 127) // 128
        self.n_padded = self.n_tiles * 128

    @cute.jit
    def __call__(
        self,
        projections: cute.Tensor,
        intermediate: cute.Tensor,
        topk_ids: cute.Tensor,
        num_pairs: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(projections, intermediate, topk_ids, num_pairs).launch(
            grid=(num_pairs * Int32(self.n_tiles), 1, 1),
            block=[32, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        projections: cute.Tensor,
        intermediate: cute.Tensor,
        topk_ids: cute.Tensor,
        num_pairs: Int32,
    ) -> None:
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane = Int32(tid)
        pair = Int32(bid) // Int32(self.n_tiles)
        output_tile = Int32(bid) % Int32(self.n_tiles)

        # This predicate is uniform for the CTA.  In particular, invalid routes
        # must not inspect their (intentionally uninitialized) FC1 boundary.
        if pair < num_pairs:
            expert = Int64(topk_ids[pair])
            if expert >= Int64(0) and expert < Int64(self.num_experts):
                col = output_tile * Int32(128) + lane * Int32(4)
                value = cute.make_rmem_tensor((4,), cutlass.Float32)
                value.fill(0.0)
                scale_byte = Uint32(0)
                if col < Int32(self.n):
                    gate = cute.make_rmem_tensor((4,), cutlass.Float32)
                    up = cute.make_rmem_tensor((4,), cutlass.Float32)
                    for i in cutlass.range_constexpr(4):
                        gate[i] = projections[pair, col + Int32(i)].to(cutlass.Float32)
                        up[i] = projections[pair, Int32(self.n) + col + Int32(i)].to(
                            cutlass.Float32
                        )
                        if cutlass.const_expr(self.has_swiglu_limit):
                            limit = cutlass.Float32(self.swiglu_limit)
                            gate[i] = cutlass.min(gate[i], limit)
                            up[i] = cutlass.max(
                                cutlass.min(up[i], limit),
                                -limit,
                            )

                    for i in cutlass.range_constexpr(4):
                        sigmoid = cute.arch.rcp_approx(
                            cutlass.Float32(1.0)
                            + cute.math.exp(-gate[i], fastmath=self.fast_math)
                        )
                        # Preserve the common W4A8 contract: BF16 activation
                        # boundary first; route weights are applied after FC2.
                        value[i] = cutlass.BFloat16(
                            gate[i] * sigmoid * up[i]
                        ).to(cutlass.Float32)

                    amax = fabs_f32(value[0])
                    for i in cutlass.range_constexpr(1, 4):
                        amax = fmax_f32(amax, fabs_f32(value[i]))
                    for shift in cutlass.range_constexpr(3):
                        amax = fmax_f32(
                            amax,
                            cute.arch.shuffle_sync_bfly(amax, offset=1 << shift),
                        )
                    _, scale_byte = pow2_ceil_ue8m0(
                        amax * cutlass.Float32(1.0 / FLOAT8_E4M3_MAX)
                    )
                    inv_scale = ue8m0_to_output_scale(scale_byte)
                    for i in cutlass.range_constexpr(4):
                        value[i] = value[i] * inv_scale

                rows = Int64(projections.shape[0])
                words_per_row = Int64(self.n_padded // 4)
                physical_row = Int64(pair)
                intermediate[
                    physical_row * words_per_row + output_tile * Int32(32) + lane
                ] = cvt_f32x4_to_e4m3x4(
                    value[0],
                    value[1],
                    value[2],
                    value[3],
                )

                # Each N128 scale plane has one packed word per physical row.
                # Tail padding receives zero scale bytes alongside zero values.
                if lane % Int32(8) == Int32(0):
                    intermediate_u8 = cute.recast_tensor(intermediate, Uint8)
                    scale_byte_index = (
                        rows * words_per_row + output_tile * rows + physical_row
                    ) * Int32(4) + lane // Int32(8)
                    intermediate_u8[scale_byte_index] = (scale_byte & Uint32(0xFF)).to(
                        Uint8
                    )


__all__ = ["W4A8CompactMicroActivationKernel"]
