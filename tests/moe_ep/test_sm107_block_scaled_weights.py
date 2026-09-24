"""Host regressions for SM107 quantization layout and bounded preprocessing."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
    concatenate_block_scaled_weights,
    interleave_gate_up_16,
    pack_f32_to_fp4,
    preprocess_block_scaled_weights,
    quantize_mxfp8_block32,
    quantize_nvfp4_block16,
    to_blocked,
    unpack_fp4_to_f32,
)


@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"])
def test_chunked_preprocess_matches_full_transform(kind):
    # NVFP4 has a partial final 128-row SF atom; MXFP8 pads SF columns.
    hidden, intermediate = (192, 64) if kind == "nvfp4" else (384, 192)
    generator = torch.Generator().manual_seed(41)
    w13 = torch.randn(3, 2 * intermediate, hidden, generator=generator).bfloat16()
    w2 = torch.randn(3, hidden, intermediate, generator=generator).bfloat16()
    transformed = preprocess_block_scaled_weights(
        w13, w2, quant_kind=kind, intermediate_size=intermediate
    )
    dtype = torch.float8_e4m3fn if kind == "mxfp8_e4m3" else torch.float8_e5m2
    physical = (
        interleave_gate_up_16(w13.float(), intermediate_size=intermediate),
        w2.float(),
    )
    for source, (weight, scale) in zip(physical, transformed, strict=True):
        q, sf = (
            quantize_nvfp4_block16(source)
            if kind == "nvfp4"
            else quantize_mxfp8_block32(source, dtype)
        )
        expected_scale = torch.stack([to_blocked(s.view(torch.uint8)) for s in sf])
        assert weight.stride(1) == 1
        assert weight.permute(0, 2, 1).is_contiguous()
        torch.testing.assert_close(
            weight.permute(0, 2, 1).view(torch.uint8), q.view(torch.uint8)
        )
        torch.testing.assert_close(scale.view(torch.uint8), expected_scale)


@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"])
@pytest.mark.parametrize("chunks", [[1], [2, 2], [2, 1]])
def test_concatenated_experts_preserve_k_major_values_and_strides(kind, chunks):
    generator = torch.Generator().manual_seed(13)
    w13 = torch.randn(sum(chunks), 128, 128, generator=generator).bfloat16()
    w2 = torch.randn(sum(chunks), 128, 64, generator=generator).bfloat16()
    expected = preprocess_block_scaled_weights(
        w13, w2, quant_kind=kind, intermediate_size=64
    )
    parts = []
    begin = 0
    for size in chunks:
        parts.append(
            preprocess_block_scaled_weights(
                w13[begin : begin + size],
                w2[begin : begin + size],
                quant_kind=kind,
                intermediate_size=64,
            )
        )
        begin += size
    for leg in (0, 1):
        actual = concatenate_block_scaled_weights([p[leg] for p in parts])
        assert actual[0].stride() == expected[leg][0].stride()
        for got, want in zip(actual, expected[leg], strict=True):
            torch.testing.assert_close(got.view(torch.uint8), want.view(torch.uint8))


def test_nvfp4_preprocessing_bounds_fp32_temporaries():
    class TrackLargestFloatTensor(TorchDispatchMode):
        largest = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            output = func(*args, **(kwargs or {}))
            tensors = output if isinstance(output, (list, tuple)) else (output,)
            for value in tensors:
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    self.largest = max(self.largest, value.numel())
            return output

    hidden, intermediate, rows_per_chunk = 128, 256, 128
    w13 = torch.ones(4, 2 * intermediate, hidden, dtype=torch.bfloat16)
    w2 = torch.ones(4, hidden, intermediate, dtype=torch.bfloat16)
    tracker = TrackLargestFloatTensor()
    with tracker:
        preprocess_block_scaled_weights(
            w13,
            w2,
            quant_kind="nvfp4",
            intermediate_size=intermediate,
            chunk_rows=rows_per_chunk,
        )
    assert tracker.largest <= rows_per_chunk * max(hidden, intermediate) * 8


def test_fp4_rounding_at_every_midpoint():
    magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    midpoints = (magnitudes[:-1] + magnitudes[1:]) / 2
    rounded = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0])
    source = torch.cat((magnitudes, midpoints, torch.tensor([100.0])))
    expected = torch.cat((magnitudes, rounded, torch.tensor([6.0])))
    for sign in (1, -1):
        actual = unpack_fp4_to_f32(pack_f32_to_fp4(source * sign))
        torch.testing.assert_close(actual, expected * sign, rtol=0, atol=0)


@pytest.mark.parametrize("rows, cols", [(64, 6), (128, 8), (192, 6), (256, 12)])
def test_scale_unswizzle_recovers_logical_plane(rows, cols):
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
        from_blocked,
        to_blocked,
    )

    raw = torch.arange(rows * cols).reshape(rows, cols).to(torch.uint8)
    torch.testing.assert_close(from_blocked(to_blocked(raw), rows, cols), raw)
