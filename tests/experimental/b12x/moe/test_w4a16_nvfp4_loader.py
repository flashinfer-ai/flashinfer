"""Bit-exact register packing for native NVFP4 weight loads."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import pytest
import torch

from b12x.moe._shared.kernels.w4a16.kernel import (
    MoEMicroKernelW4A16SmallMDirect,
    W4A16GemmKernel,
    _pack_modelopt_words,
)


class _PackWords:
    @cute.jit
    def __call__(self, words: cute.Tensor, result: cute.Tensor, stream: cuda.CUstream):
        self.kernel(words, result).launch(grid=(words.shape[0] // 128, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, words: cute.Tensor, result: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        row = block * 128 + tid
        for byte_index in cutlass.range_constexpr(4):
            result[row, byte_index] = _pack_modelopt_words(
                words[row, 0], words[row, 1], words[row, 2], words[row, 3],
                cutlass.Uint32(byte_index),
            )


class _DecodeScaleBytes:
    def __init__(self):
        self.consumer = MoEMicroKernelW4A16SmallMDirect(
            activation="silu", fast_math=True, share_input_across_experts=False,
            share_expert_scales=True, single_token=False,
        )

    @cute.jit
    def __call__(self, codes: cute.Tensor, result: cute.Tensor, stream: cuda.CUstream):
        self.kernel(codes, result).launch(
            grid=(1, 1, 1), block=(128, 1, 1), stream=stream,
        )

    @cute.kernel
    def kernel(self, codes: cute.Tensor, result: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        if tid < 127:
            result[tid] = self.consumer._scale_byte_to_f32(cutlass.Uint32(codes[tid]))


class _DecodeNativeValues:
    def __init__(self, element_dtype):
        self.consumer = W4A16GemmKernel(
            size_m=1, size_n=128, size_k=128, num_experts=1, top_k=1,
            mul_topk_weights=False, tile_n=128, tile_k=128,
            moe_block_size=8, max_m_blocks=1, element_dtype=element_dtype,
            weight_layout="modelopt", scale_format="e4m3_k16",
        )

    @cute.jit
    def __call__(self, words: cute.Tensor, scales: cute.Tensor, result: cute.Tensor, stream: cuda.CUstream):
        self.kernel(words, scales, result).launch(
            grid=((words.shape[0] + 127) // 128, 1, 1), block=(128, 1, 1), stream=stream,
        )

    @cute.kernel
    def kernel(self, words: cute.Tensor, scales: cute.Tensor, result: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        row = block * 128 + tid
        if row < words.shape[0]:
            s0, _ = self.consumer._dequant_scale_x4_to_elem2x2(scales[row])
            fragment = cute.make_rmem_tensor((2, 2), cutlass.Uint32)
            self.consumer._scaled_dequant_b_fragment(fragment, words[row], s0)
            for i in cutlass.range_constexpr(4):
                result[row, i] = fragment[i // 2, i % 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype,element_dtype,compensation", [
    (torch.bfloat16, "bf16", 128.0), (torch.float16, "fp16", 1.0),
])
def test_native_tensor_core_decode_preserves_fp4_and_all_finite_scales(dtype, element_dtype, compensation):
    generator = torch.Generator().manual_seed(18)
    words_cpu = torch.cat((
        torch.arange(256, dtype=torch.int64) * 0x01010101,
        torch.randint(0, 2**32, (256,), generator=generator, dtype=torch.int64),
    )).repeat_interleave(127)
    scale_codes = torch.arange(127, dtype=torch.uint8).repeat(512)
    codes = (words_cpu[:, None] >> (torch.arange(8) * 4)) & 15
    codebook = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
    expected = codebook[codes[:, [0, 4, 1, 5, 2, 6, 3, 7]]] * scale_codes.view(torch.float8_e4m3fn).float()[:, None]
    words = words_cpu.to(device="cuda", dtype=torch.uint32)
    scales = (scale_codes.to(torch.int64) * 0x01010101).to(device="cuda", dtype=torch.uint32)
    result = torch.empty((words.numel(), 4), dtype=torch.uint32, device="cuda")
    args = (from_dlpack(words), from_dlpack(scales), from_dlpack(result), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    launch = cute.compile(_DecodeNativeValues(element_dtype), *args)
    launch(*args)
    torch.testing.assert_close(result.view(dtype).float().cpu() * compensation, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_modelopt_decode_preserves_all_finite_nonnegative_e4m3_scales() -> None:
    codes = torch.arange(127, device="cuda", dtype=torch.uint8)
    expected = codes.view(torch.float8_e4m3fn).float()
    result = torch.empty_like(expected)
    args = (
        from_dlpack(codes), from_dlpack(result),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    launch = cute.compile(_DecodeScaleBytes(), *args)
    launch(*args)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_modelopt_register_pack_preserves_every_fp4_code() -> None:
    generator = torch.Generator().manual_seed(13)
    random_words = torch.randint(0, 2**32, (768, 4), generator=generator, dtype=torch.int64)
    repeated_bytes = (torch.arange(256, dtype=torch.int64) * 0x01010101)[:, None].expand(256, 4)
    words_cpu = torch.cat((repeated_bytes, random_words))
    expected = torch.zeros((1024, 4), dtype=torch.int64)
    for byte_index in range(4):
        for source_index in range(4):
            byte = (words_cpu[:, source_index] >> (8 * byte_index)) & 255
            expected[:, byte_index] |= (byte & 15) << (4 * source_index)
            expected[:, byte_index] |= (byte >> 4) << (16 + 4 * source_index)

    words = words_cpu.to(device="cuda", dtype=torch.uint32)
    result = torch.empty((1024, 4), device="cuda", dtype=torch.uint32)
    args = (from_dlpack(words), from_dlpack(result), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    launch = cute.compile(_PackWords(), *args)
    launch(*args)
    torch.testing.assert_close(result.cpu().to(torch.int64), expected, rtol=0, atol=0)
