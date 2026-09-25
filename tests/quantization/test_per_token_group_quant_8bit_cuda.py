# SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the native CUDA per-token-group 8-bit quantization backend."""

import pytest
import torch

from flashinfer.quantization import per_token_group_quant_8bit


def _reference(x: torch.Tensor, dst_dtype: torch.dtype, ue8m0: bool = False):
    grouped = x.reshape(-1, 128).float()
    qmax = 127.0 if dst_dtype == torch.int8 else torch.finfo(dst_dtype).max
    qmin = -128.0 if dst_dtype == torch.int8 else torch.finfo(dst_dtype).min
    absmax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = absmax / qmax
    if ue8m0:
        scale = torch.exp2(torch.ceil(torch.log2(scale.abs().clamp(min=1e-10))))
    scaled = (grouped / scale).clamp(qmin, qmax)
    if dst_dtype == torch.int8:
        scaled = scaled.round()
    quantized = scaled.to(dst_dtype).reshape(x.shape)
    scales = scale.reshape(x.shape[:-1] + (x.shape[-1] // 128,)).float()
    return quantized, scales


@pytest.mark.parametrize("shape", [(1, 128), (333, 128), (128, 2048)])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "dst_dtype", [torch.float8_e4m3fn, torch.float8_e5m2, torch.int8]
)
def test_cuda_matches_reference(shape, input_dtype, dst_dtype):
    torch.manual_seed(42)
    x = torch.randn(shape, device="cuda", dtype=input_dtype)

    actual_q, actual_s = per_token_group_quant_8bit(
        x, 128, dst_dtype=dst_dtype, backend="cuda"
    )
    expected_q, expected_s = _reference(x, dst_dtype)

    torch.testing.assert_close(actual_s, expected_s, rtol=0, atol=0)
    torch.testing.assert_close(actual_q.float(), expected_q.float(), rtol=0, atol=0)


@pytest.mark.parametrize("num_tokens", [1, 130])
@pytest.mark.parametrize("tma_aligned", [False, True])
@pytest.mark.parametrize("ue8m0", [False, True])
def test_cuda_column_major_scales(num_tokens, tma_aligned, ue8m0):
    torch.manual_seed(7)
    x = torch.randn((num_tokens, 2048), device="cuda", dtype=torch.bfloat16)

    actual_q, actual_s = per_token_group_quant_8bit(
        x,
        128,
        dst_dtype=torch.float8_e4m3fn,
        column_major_scales=True,
        scale_tma_aligned=tma_aligned,
        scale_ue8m0=ue8m0,
        backend="cuda",
    )
    expected_q, expected_s = _reference(x, torch.float8_e4m3fn, ue8m0)

    expected_stride = ((num_tokens + 3) // 4) * 4 if tma_aligned else num_tokens
    assert actual_s.shape == expected_s.shape
    assert actual_s.stride() == (1, expected_stride)
    torch.testing.assert_close(actual_s, expected_s, rtol=0, atol=0)
    torch.testing.assert_close(actual_q.float(), expected_q.float(), rtol=0, atol=0)


def test_cuda_rejects_unsupported_group_size():
    x = torch.randn((4, 128), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="group_size=128"):
        per_token_group_quant_8bit(x, 64, backend="cuda")


def test_cuda_unaligned_contiguous_input():
    torch.manual_seed(11)
    storage = torch.randn(4 * 128 + 1, device="cuda", dtype=torch.float16)
    x = storage[1:].view(4, 128)
    assert x.is_contiguous()
    assert x.data_ptr() % 16 != 0

    actual_q, actual_s = per_token_group_quant_8bit(
        x, 128, dst_dtype=torch.float8_e4m3fn, backend="cuda"
    )
    expected_q, expected_s = _reference(x, torch.float8_e4m3fn)

    torch.testing.assert_close(actual_s, expected_s, rtol=0, atol=0)
    torch.testing.assert_close(actual_q.float(), expected_q.float(), rtol=0, atol=0)


@pytest.mark.parametrize("column_major", [False, True])
def test_cuda_empty_input(column_major):
    x = torch.empty((0, 128), device="cuda", dtype=torch.bfloat16)
    actual_q, actual_s = per_token_group_quant_8bit(
        x, 128, column_major_scales=column_major, backend="cuda"
    )

    assert actual_q.shape == x.shape
    assert actual_s.shape == (0, 1)
