"""Native mixed-cache staging must preserve the FP4/FP8 dequantization contract."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.utils import SmemAllocator
import pytest
import torch

from b12x._lib.compiler import compile as compile_kernel
from b12x._lib.intrinsics import shared_ptr_to_u32
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.attention._shared.mla.decode_math import (
    _nvfp4_pair_bfloat2,
    s0_normalize_dsv41_kv_to_fp8,
)
from b12x.attention._shared.mla.io import stage_dsv41_fp8_scales
from b12x.testing.reference.helpers import require_b12x


class StagedMixedPair:
    """Exercise source tags and byte offsets through the production helper."""

    @cute.jit
    def __call__(
        self,
        q: cute.Pointer,
        sf: cute.Pointer,
        tag: cute.Pointer,
        out: cute.Pointer,
        stream,
    ):
        self.kernel(q, sf, tag, out).launch(
            grid=(2048, 1, 1),
            block=(32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self, q: cute.Pointer, sf: cute.Pointer, tag: cute.Pointer, out: cute.Pointer
    ):
        lane = Int32(cute.arch.thread_idx()[0])
        i = Int64(cute.arch.block_idx()[0]) * Int64(32) + lane
        staging = SmemAllocator().allocate_tensor(
            Uint32,
            cute.make_layout(32 * 136),
            16,
        )
        staging[lane * 136] = q[i]
        staging[lane * 136 + 64] = sf[i]
        staging[lane * 136 + 128] = sf[i]
        staging[lane * 136 + 132] = tag[i]
        cute.arch.sync_threads()
        out[i] = _nvfp4_pair_bfloat2(
            shared_ptr_to_u32(staging.iterator),
            lane,
            Int32(0),
            Float32(17.0),
            kv_smem_stride=544,
        )


@pytest.mark.parametrize("source", ["indexed", "swa", "invalid"])
def test_staged_pairs_exhaustive_and_graph_mutation(source):
    """All scale bytes, including UE8M0 zero/subnormals/NaN, match FP64."""
    require_b12x()
    codes = torch.arange(256, device="cuda", dtype=torch.int64).repeat(256)
    scales = torch.arange(256, device="cuda", dtype=torch.int64).repeat_interleave(256)
    if source == "indexed":
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device="cuda",
            dtype=torch.float64,
        )
        values = torch.stack((lut[codes & 15], lut[codes >> 4]), -1)
        factors = scales.to(torch.uint8).view(torch.float8_e4m3fn).double()
        packed = codes
        source_tag = 0
    else:
        values = torch.stack((codes, 255 - codes), -1).to(torch.uint8)
        values = values.view(torch.float8_e4m3fn).double()
        factors = torch.exp2(scales.double() - 127)
        factors[scales == 255] = float("nan")
        packed = codes | ((255 - codes) << 8)
        source_tag = 1 if source == "swa" else 2
    expected = (values * factors[:, None]).to(torch.bfloat16)
    if source == "invalid":
        expected.zero_()
    packed = packed.to(torch.uint32)
    scales = scales.to(torch.uint32)
    tags = torch.full_like(scales, source_tag)
    output = torch.empty_like(expected)
    pointers = [
        make_ptr(Uint32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        for t in (packed, scales, tags, output)
    ]
    fn = compile_kernel(StagedMixedPair(), *pointers, current_cuda_stream())
    fn(*pointers, current_cuda_stream())
    torch.testing.assert_close(output, expected, rtol=0, atol=0, equal_nan=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(*pointers, current_cuda_stream())
    # Invalid source rows must become zero even when their payload contains NaNs.
    tags.fill_(2)
    output.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)


class CanonicalV41Rows:
    def __init__(self, warps):
        self.warps = warps

    @cute.jit
    def __call__(
        self, source: cute.Pointer, tags: cute.Pointer, output: cute.Pointer, stream
    ):
        self.kernel(source, tags, output).launch(
            grid=(1, 1, 1),
            block=(32 * self.warps, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, source: cute.Pointer, tags: cute.Pointer, output: cute.Pointer):
        tid = Int32(cute.arch.thread_idx()[0])
        stage = SmemAllocator().allocate_tensor(
            Uint32,
            cute.make_layout(64 * 156 + 64 * 2),
            16,
        )
        address = shared_ptr_to_u32(stage.iterator)
        source_bytes = cute.make_tensor(
            cute.recast_ptr(source, dtype=cutlass.Uint8), cute.make_layout(64 * 528)
        )
        for i in cutlass.range(tid, 64 * 132, 32 * self.warps):
            row, col = i // 132, i % 132
            stage[row * 156 + col] = source[Int64(i)]
        if tid < 64:
            tag = tags[Int64(tid)]
            stage[tid * 156 + 132] = tag
            ratio_addr = address + tid * Int32(624) + Int32(544)
            scales_addr = address + Int32(64 * 624) + tid * Int32(8)
            if tag == Uint32(1):
                stage_dsv41_fp8_scales(
                    source_bytes, Int64(tid) * Int64(528), ratio_addr, scales_addr,
                    True, swa=True,
                )
            else:
                stage_dsv41_fp8_scales(
                    source_bytes, Int64(tid) * Int64(528), ratio_addr, scales_addr,
                    tag == Uint32(0), swa=False,
                )
        cute.arch.sync_threads()
        s0_normalize_dsv41_kv_to_fp8(
            address,
            address + Int32(544),
            tid // 32,
            tid % 32,
            bi=64,
            kv_smem_stride=624,
            ratio_smem_stride=624,
            math_warps=self.warps,
        )
        cute.arch.sync_threads()
        for i in cutlass.range(tid, 64 * 130, 32 * self.warps):
            row, col = i // 130, i % 130
            value = Uint32(0)
            if col < 128:
                value = stage[row * 156 + col]
            else:
                value = stage[64 * 156 + row * 2 + col - 128]
            output[Int64(i)] = value


@pytest.mark.parametrize("warps", [4, 8])
def test_native_fp8_canonicalization_preserves_metadata_and_inplace_sources(warps):
    """Descending expansion and saved scales match an independent FP8 encoder."""
    require_b12x()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(83143)
    tags = (torch.arange(64, device=device) % 3).to(torch.uint32)
    source = torch.full((64, 528), 0xA5, device=device, dtype=torch.uint8)
    expected = torch.zeros((64, 520), device=device, dtype=torch.uint8)
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=device,
        dtype=torch.float64,
    )
    scale_choices = torch.tensor(
        [0, 0.03125, 0.375, 1, 1.75, 8, 192],
        device=device,
    ).to(torch.float8_e4m3fn)
    for row in range(64):
        if row % 3 == 0:
            codes = torch.randint(16, (512,), generator=generator, device=device)
            scales = scale_choices[
                torch.randint(7, (32,), generator=generator, device=device)
            ]
            source[row, :256] = (codes[::2] | (codes[1::2] << 4)).byte()
            source[row, 256:288] = scales.view(torch.uint8)
            values = levels[codes] * scales.double().repeat_interleave(16)
            bound = scales.double().view(8, 4).amax(1) * (6 / 448)
            exponent = torch.ceil(torch.log2(bound.clamp_min(2.0**-126)))
            encoded_scales = (exponent + 127).byte()
        elif row % 3 == 1:
            values_fp8 = torch.randn((512,), generator=generator, device=device).to(
                torch.float8_e4m3fn
            )
            scales = torch.randint(
                118, 133, (16,), generator=generator, device=device
            ).byte()
            scales[:2] = 0
            source[row, :512] = values_fp8.view(torch.uint8)
            source[row, 512:528] = scales
            values = values_fp8.double() * torch.exp2(
                scales.double() - 127
            ).repeat_interleave(32)
            encoded_scales = scales.view(8, 2).amax(1).clamp_min(1)
            exponent = encoded_scales.double() - 127
        else:
            expected[row, 512:] = 127
            continue
        normalized = values / torch.exp2(exponent).repeat_interleave(64)
        expected[row, :512] = (
            normalized.float().to(torch.float8_e4m3fn).view(torch.uint8)
        )
        expected[row, 512:] = encoded_scales
    output = torch.empty_like(expected)
    pointers = [
        make_ptr(Uint32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        for t in (source, tags, output)
    ]
    fn = compile_kernel(CanonicalV41Rows(warps), *pointers, current_cuda_stream())
    fn(*pointers, current_cuda_stream())
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(*pointers, current_cuda_stream())
    tags.fill_(2)
    output.fill_(255)
    graph.replay()
    expected[:, :512] = 0
    expected[:, 512:] = 127
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
