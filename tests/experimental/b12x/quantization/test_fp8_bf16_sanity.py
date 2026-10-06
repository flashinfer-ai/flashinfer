from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import pytest
import torch
from cutlass import Float32, const_expr
from cutlass.cute.runtime import from_dlpack

from b12x._lib.intrinsics import (
    bfloat2_mul,
    bfloat2_to_float2_scaled,
    broadcast_f32_to_bfloat2,
    fp8x4_e4m3_to_bfloat2x2,
)

from b12x.testing.reference.helpers import require_b12x


class TinyFp8Bf16DequantKernel:
    tile_m = 16
    tile_n = 16
    num_threads = 32
    dtype = cutlass.BFloat16

    @cute.jit
    def __call__(
        self,
        mB: cute.Tensor,
        mDescale: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if const_expr(mB.element_type != cutlass.Float8E4M3FN):
            raise TypeError("B must be Float8E4M3FN")
        if const_expr(mDescale.element_type != Float32):
            raise TypeError("descale must be Float32")
        if const_expr(mOut.element_type != cutlass.BFloat16):
            raise TypeError("out must be BFloat16")
        if const_expr(mB.shape != (self.tile_m, self.tile_n)):
            raise ValueError("B must have shape (16, 16)")
        if const_expr(mOut.shape != (self.tile_m, self.tile_n)):
            raise ValueError("out must have shape (16, 16)")
        if const_expr(mDescale.shape != (1,)):
            raise ValueError("descale must have shape (1,)")
        self.kernel(mB, mDescale, mOut).launch(
            grid=(1, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mB: cute.Tensor,
        mDescale: cute.Tensor,
        mOut: cute.Tensor,
    ):
        tidx = cute.arch.thread_idx()[0]
        lane = cute.arch.lane_idx()
        smem = cutlass_utils.SmemAllocator()
        s_layout = cute.make_layout((self.tile_m, self.tile_n))
        sB = smem.allocate_tensor(element_type=self.dtype, layout=s_layout, byte_alignment=128)
        mBu8 = cute.recast_tensor(mB, cutlass.Uint8)
        descale_bf2 = broadcast_f32_to_bfloat2(mDescale[0])
        one = Float32(1.0)
        total_elems = self.tile_m * self.tile_n
        total_vec4 = total_elems // 4

        for idx_iter in cutlass.range_constexpr(cute.ceil_div(total_vec4, cute.arch.WARP_SIZE)):
            vec_idx = lane + idx_iter * cute.arch.WARP_SIZE
            if vec_idx < total_vec4:
                linear_idx = vec_idx * 4
                row = linear_idx // self.tile_n
                col = linear_idx - row * self.tile_n
                packed = (
                    cutlass.Uint32(mBu8[row, col + 0])
                    | (cutlass.Uint32(mBu8[row, col + 1]) << cutlass.Uint32(8))
                    | (cutlass.Uint32(mBu8[row, col + 2]) << cutlass.Uint32(16))
                    | (cutlass.Uint32(mBu8[row, col + 3]) << cutlass.Uint32(24))
                )
                bf2_01, bf2_23 = fp8x4_e4m3_to_bfloat2x2(packed)
                bf2_01 = bfloat2_mul(bf2_01, descale_bf2)
                bf2_23 = bfloat2_mul(bf2_23, descale_bf2)
                value0, value1 = bfloat2_to_float2_scaled(bf2_01, one)
                value2, value3 = bfloat2_to_float2_scaled(bf2_23, one)
                sB[row, col + 0] = value0.to(self.dtype)
                sB[row, col + 1] = value1.to(self.dtype)
                sB[row, col + 2] = value2.to(self.dtype)
                sB[row, col + 3] = value3.to(self.dtype)
        cute.arch.sync_threads()

        for idx_iter in cutlass.range_constexpr(cute.ceil_div(total_elems, self.num_threads)):
            linear_idx = tidx + idx_iter * self.num_threads
            if linear_idx < total_elems:
                row = linear_idx // self.tile_n
                col = linear_idx - row * self.tile_n
                mOut[row, col] = sB[row, col]


def _to_cute_tensor(x: torch.Tensor, dtype) -> cute.Tensor:
    tensor = from_dlpack(x, assumed_align=16)
    tensor.element_type = dtype
    return tensor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tiny_fp8_bf16_dequant_matches_reference() -> None:
    require_b12x()
    torch.manual_seed(0)
    device = torch.device("cuda")

    b_src = torch.randn(16, 16, device=device, dtype=torch.float32) / 4
    descale = torch.tensor([0.5], device=device, dtype=torch.float32)
    b_fp8 = (b_src / descale).clamp(
        min=-torch.finfo(torch.float8_e4m3fn).max,
        max=torch.finfo(torch.float8_e4m3fn).max,
    ).to(torch.float8_e4m3fn)
    out = torch.empty_like(b_src, dtype=torch.bfloat16)

    kernel = TinyFp8Bf16DequantKernel()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        kernel,
        _to_cute_tensor(b_fp8, cutlass.Float8E4M3FN),
        _to_cute_tensor(descale, cutlass.Float32),
        _to_cute_tensor(out, cutlass.BFloat16),
        stream,
    )
    compiled(
        _to_cute_tensor(b_fp8, cutlass.Float8E4M3FN),
        _to_cute_tensor(descale, cutlass.Float32),
        _to_cute_tensor(out, cutlass.BFloat16),
        stream,
    )
    torch.cuda.synchronize()

    ref = (b_fp8.float() * descale[0]).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, atol=4e-3, rtol=0.0)
