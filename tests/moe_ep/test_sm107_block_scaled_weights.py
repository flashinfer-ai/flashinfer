"""Host regressions for SM107 quantization layout and bounded preprocessing."""

import math

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
    concatenate_block_scaled_weights,
    from_blocked,
    interleave_gate_up_16,
    pack_f32_to_fp4,
    preprocess_block_scaled_weights,
    preprocess_prequantized_block_scaled_weights,
    quantize_mxfp4_block32,
    quantize_mxfp8_block32,
    quantize_nvfp4_block16,
    to_blocked,
    unpack_fp4_to_f32,
)


@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2", "mxfp4_mxfp8"])
def test_chunked_preprocess_matches_full_transform(kind):
    # NVFP4 has a partial final 128-row SF atom; MXFP8 pads SF columns.
    hidden, intermediate = (192, 64) if kind == "nvfp4" else (384, 192)
    if kind == "mxfp4_mxfp8":
        intermediate = 256
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
            else quantize_mxfp4_block32(source)
            if kind == "mxfp4_mxfp8"
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


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e8m0fnu])
@pytest.mark.parametrize("strided", [False, True])
def test_e8m0_decoder_all_encodings(dtype, strided):
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
        scale_to_f32,
    )

    codes = torch.arange(256, dtype=torch.uint8)
    if strided:
        codes = torch.stack((codes, codes), dim=-1)[:, 0]
    scales = codes.view(dtype)
    expected = torch.tensor(
        [math.ldexp(1.0, exponent) for exponent in range(-127, 128)],
        dtype=torch.float32,
    )
    actual = scale_to_f32(scales)
    # Compare bits to distinguish the smallest scale from zero.
    torch.testing.assert_close(
        actual[:255].view(torch.int32), expected.view(torch.int32), rtol=0, atol=0
    )
    assert torch.isnan(actual[255])


def test_fp4_rounding_at_every_midpoint():
    magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    midpoints = (magnitudes[:-1] + magnitudes[1:]) / 2
    rounded = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0])
    source = torch.cat((magnitudes, midpoints, torch.tensor([100.0])))
    expected = torch.cat((magnitudes, rounded, torch.tensor([6.0])))
    for sign in (1, -1):
        actual = unpack_fp4_to_f32(pack_f32_to_fp4(source * sign))
        torch.testing.assert_close(actual, expected * sign, rtol=0, atol=0)


def test_mxfp4_block_scales_and_packed_values():
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    block = torch.cat((values, -values)).repeat(2)
    source = torch.stack((block, block * 2, block * 0.5, torch.zeros_like(block)))
    packed, scales = quantize_mxfp4_block32(source)
    assert packed.shape == (4, 16)
    assert scales.dtype == torch.float8_e8m0fnu
    assert scales.view(torch.uint8).flatten().tolist() == [127, 128, 126, 1]
    expected = block.expand(3, -1)
    torch.testing.assert_close(unpack_fp4_to_f32(packed[:3]), expected, rtol=0, atol=0)
    assert not packed[3].view(torch.uint8).any()

    # Crossing the largest E2M1 value raises the block scale to the next power.
    source = torch.full((1, 32), 6.0)
    source[0, 0] = torch.nextafter(source[0, 0], torch.tensor(float("inf")))
    _, scales = quantize_mxfp4_block32(source)
    assert scales.view(torch.uint8).item() == 128


@pytest.mark.parametrize("rows, cols", [(64, 6), (128, 8), (192, 6), (256, 12)])
def test_scale_unswizzle_recovers_logical_plane(rows, cols):
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
        from_blocked,
        to_blocked,
    )

    raw = torch.arange(rows * cols).reshape(rows, cols).to(torch.uint8)
    torch.testing.assert_close(from_blocked(to_blocked(raw), rows, cols), raw)


@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2", "mxfp4_mxfp8"])
@pytest.mark.parametrize("storage", ["contiguous", "strided", "unaligned"])
def test_prequantized_ingestion_preserves_bytes(kind, storage):
    from importlib import import_module

    from flashinfer.moe_ep import PrequantizedMoEWeights

    hidden, intermediate, experts = 384, 192, 3
    nvfp4 = kind == "nvfp4"
    fp4 = kind in ("nvfp4", "mxfp4_mxfp8")
    if kind == "mxfp4_mxfp8":
        intermediate = 256
    packing, vec = (2 if fp4 else 1), (16 if nvfp4 else 32)
    dtype = (
        torch.uint8
        if fp4
        else torch.float8_e4m3fn
        if kind == "mxfp8_e4m3"
        else torch.float8_e5m2
    )
    sf_dtype = torch.float8_e4m3fn if nvfp4 else torch.float8_e8m0fnu
    generator = torch.Generator().manual_seed(731)

    def payload(shape, dtype):
        # Arbitrary bytes expose accidental numeric conversion, including
        # float8 NaNs and signed FP4 zero. Only layout changes are permitted.
        if storage == "unaligned":
            count = 1
            for size in shape:
                count *= size
            raw = torch.randint(
                256, (count + 1,), dtype=torch.uint8, generator=generator
            )
            return raw[1:].reshape(shape).view(dtype)
        allocation = torch.randint(
            256, (*shape[:-1], shape[-1] * 2), dtype=torch.uint8, generator=generator
        )
        raw = (
            allocation[..., ::2]
            if storage == "strided"
            else allocation[..., : shape[-1]].contiguous()
        )
        return raw.view(dtype)

    w13 = payload((experts, 2 * intermediate, hidden // packing), dtype)
    w2 = payload((experts, hidden, intermediate // packing), dtype)
    sf1 = payload((experts, 2 * intermediate, hidden // vec), sf_dtype)
    sf2 = payload((experts, hidden, intermediate // vec), sf_dtype)
    backend = "nvfp4_nvfp4_bf16_cutedsl" if nvfp4 else "mxfp8_mxfp8_bf16_cutedsl"
    if kind == "mxfp4_mxfp8":
        backend = "mxfp8_mxfp4_bf16_cutedsl"
    mod = import_module(
        f"flashinfer.moe_ep.backends.mega.kernel.sm107.{backend}.weights"
    )
    result = mod.preprocess_mega_weights(
        PrequantizedMoEWeights(w13, w2, sf1, sf2),
        intermediate_size=intermediate,
        hidden_size=hidden,
        **({} if fp4 else {"kind": kind}),
    )
    for leg, (weight, scale) in enumerate(result):
        source, raw_sf = (w13, sf1) if leg == 0 else (w2, sf2)
        rows, cols = source.shape[1:]
        order = torch.arange(rows)
        if leg == 0:
            order = torch.stack(
                (
                    torch.arange(intermediate).reshape(-1, 16),
                    torch.arange(intermediate, rows).reshape(-1, 16),
                ),
                dim=1,
            ).flatten()
        expected = source.view(torch.uint8)[:, order]
        assert weight.stride(1) == 1
        assert weight.data_ptr() % 16 == 0 and scale.data_ptr() % 16 == 0
        assert weight.permute(0, 2, 1).is_contiguous()
        torch.testing.assert_close(weight.permute(0, 2, 1).view(torch.uint8), expected)
        for expert in range(experts):
            recovered = from_blocked(
                scale[expert].view(torch.uint8), rows, raw_sf.shape[-1]
            )
            torch.testing.assert_close(
                recovered, raw_sf[expert].view(torch.uint8)[order]
            )


@pytest.mark.parametrize(
    "corruption", ["shape", "scale_shape", "data_dtype", "scale_dtype", "device"]
)
def test_prequantized_ingestion_rejects_invalid_contract(corruption):
    tensors = [
        torch.zeros(2, 128, 64, dtype=torch.uint8),
        torch.zeros(2, 128, 32, dtype=torch.uint8),
        torch.zeros(2, 128, 8, dtype=torch.float8_e4m3fn),
        torch.zeros(2, 128, 4, dtype=torch.float8_e4m3fn),
    ]
    if corruption == "shape":
        tensors[0] = tensors[0][:, :-1]
    elif corruption == "scale_shape":
        tensors[2] = tensors[2][:, :, :-1]
    elif corruption == "data_dtype":
        tensors[0] = tensors[0].float()
    elif corruption == "scale_dtype":
        tensors[2] = tensors[2].view(torch.float8_e5m2)
    else:
        tensors[3] = tensors[3].to("meta")
    with pytest.raises(ValueError):
        preprocess_prequantized_block_scaled_weights(
            *tensors, quant_kind="nvfp4", hidden_size=128, intermediate_size=64
        )
