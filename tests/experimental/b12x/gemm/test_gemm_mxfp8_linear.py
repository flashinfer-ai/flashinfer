from __future__ import annotations

import cutlass.cute as cute
import pytest
import torch

from b12x.gemm._shared.block_fp8 import quantize_block_fp8_linear_input_mxfp8
from b12x.gemm.mxfp8_linear._kernel import (
    mxfp8_linear,
    pack_mxfp8_linear_weight,
)
from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch

from b12x.testing.reference.helpers import require_b12x


def require_mxf8_mma() -> None:
    if not hasattr(cute.nvgpu.warp, "MmaMXF8Op"):
        pytest.skip("CUTLASS DSL does not expose cute.nvgpu.warp.MmaMXF8Op")


def _quantize_modelopt_mxfp8_rows(
    source: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = map(int, source.shape)
    if width % 32 != 0:
        raise ValueError(f"width must be divisible by 32, got {width}")
    chunks = width // 32
    blocked = source.to(torch.float32).reshape(rows, chunks, 32)
    max_abs = blocked.abs().amax(dim=-1)
    safe = torch.where(
        max_abs > 0.0,
        max_abs / 448.0,
        torch.ones_like(max_abs),
    )
    scale_exp = torch.ceil(torch.log2(safe)).clamp(-127, 127)
    scale_u8 = (scale_exp + 127).to(torch.uint8)
    scale = scale_u8.view(torch.float8_e8m0fnu).to(torch.float32)
    values = (
        (blocked / scale[..., None])
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .reshape(rows, width)
        .contiguous()
    )
    return values, scale_u8.contiguous()


def _pad_source_for_reference(source: torch.Tensor, padded_width: int) -> torch.Tensor:
    rows, width = map(int, source.shape)
    if width == padded_width:
        return source.contiguous()
    padded = source.new_zeros((rows, padded_width))
    padded[:, :width] = source
    return padded.contiguous()


def _reference_from_packed(source: torch.Tensor, packed_weight) -> torch.Tensor:
    source_padded = _pad_source_for_reference(
        source,
        int(packed_weight.padded_in_features),
    )
    x_q = quantize_block_fp8_linear_input_mxfp8(source_padded)
    x_deq = dequantize_mxfp8_rows_torch(x_q.values, x_q.scale_rows)
    w_deq = dequantize_mxfp8_rows_torch(
        packed_weight.weight.values,
        packed_weight.weight.scale_rows,
    )
    return x_deq @ w_deq.T


def test_mxfp8_linear_matches_quantized_reference_small_n() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260614)

    tokens, in_features, out_features = 7, 128, 32
    source = (
        torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    weight_bf16 = (
        torch.randn((out_features, in_features), device="cuda", dtype=torch.bfloat16)
        / 8
    ).contiguous()
    weight, weight_scale = _quantize_modelopt_mxfp8_rows(weight_bf16)
    packed = pack_mxfp8_linear_weight(weight, weight_scale)

    actual = mxfp8_linear(source, packed)
    expected = _reference_from_packed(source, packed)
    torch.cuda.synchronize()

    assert actual.shape == (tokens, out_features)
    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=0,
        atol=0,
    )


def test_mxfp8_linear_pads_k32_to_dense_tile() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260615)

    tokens, in_features, out_features = 3, 160, 40
    source = (
        torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    weight_bf16 = (
        torch.randn((out_features, in_features), device="cuda", dtype=torch.bfloat16)
        / 8
    ).contiguous()
    weight, weight_scale = _quantize_modelopt_mxfp8_rows(weight_bf16)
    packed = pack_mxfp8_linear_weight(weight, weight_scale)

    assert packed.in_features == in_features
    assert packed.padded_in_features == 256
    assert packed.weight.values.shape == (out_features, 256)
    assert packed.weight.scale_rows.shape == (1, out_features, 8)
    torch.testing.assert_close(
        packed.weight.scale_rows.view(torch.uint8)[0, :, :5],
        weight_scale,
    )
    assert torch.all(packed.weight.scale_rows.view(torch.uint8)[0, :, 5:] == 127)

    actual = mxfp8_linear(source, packed)
    expected = _reference_from_packed(source, packed)
    torch.cuda.synchronize()

    assert actual.shape == (tokens, out_features)
    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=0,
        atol=0,
    )


def test_mxfp8_linear_default_fused_path_captures_with_k_padding() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260616)

    tokens, in_features, out_features = 1, 160, 40
    source = (
        torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    weight_bf16 = (
        torch.randn((out_features, in_features), device="cuda", dtype=torch.bfloat16)
        / 8
    ).contiguous()
    weight, weight_scale = _quantize_modelopt_mxfp8_rows(weight_bf16)
    packed = pack_mxfp8_linear_weight(weight, weight_scale)

    eager = mxfp8_linear(source, packed).clone()
    torch.cuda.synchronize()

    mxfp8_linear(source, packed)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = mxfp8_linear(source, packed)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, eager, rtol=0, atol=0)
