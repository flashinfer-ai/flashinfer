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
    # Divide by a device tensor: for a CUDA tensor divided by a Python scalar
    # PyTorch multiplies by the reciprocal, which is not the IEEE quotient the
    # kernel (and the cuda.tile backend) produce.
    qmax_t = torch.tensor(qmax, device=x.device, dtype=torch.float32)
    scale = absmax / qmax_t
    if ue8m0:
        scale = torch.exp2(torch.ceil(torch.log2(scale.abs().clamp(min=1e-10))))
    scaled = (grouped / scale).clamp(qmin, qmax)
    if dst_dtype == torch.int8:
        # Round half up, the convention shared with the cuda.tile backend.
        scaled = torch.floor(scaled + 0.5)
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


@pytest.mark.parametrize("ue8m0", [False, True])
def test_cuda_int8_rounds_half_up(ue8m0):
    # absmax 254 gives the scale 254 / 127 = 2 exactly; with UE8M0, absmax 200
    # gives 200 / 127 = 1.57..., rounded up to the power of two 2. Either way
    # the inputs 1, 3, 5 land exactly on the ties 0.5, 1.5, 2.5, where
    # round-half-up (the cuda.tile convention) gives 1, 2, 3 while
    # round-to-nearest-even would give 0, 2, 2. The negative ties -0.5, -1.5,
    # -2.5 go to 0, -1, -2 (nearest-even: 0, -2, -2).
    absmax = 200.0 if ue8m0 else 254.0
    x = torch.zeros((4, 128), device="cuda", dtype=torch.float16)
    x[:, 0] = absmax
    x[:, 1:7] = torch.tensor([1.0, 3.0, 5.0, -1.0, -3.0, -5.0], device="cuda")

    actual_q, actual_s = per_token_group_quant_8bit(
        x,
        128,
        dst_dtype=torch.int8,
        column_major_scales=ue8m0,
        scale_ue8m0=ue8m0,
        backend="cuda",
    )

    expected_q = torch.zeros_like(x, dtype=torch.int8)
    expected_q[:, 0] = 100 if ue8m0 else 127
    expected_q[:, 1:7] = torch.tensor(
        [1, 2, 3, 0, -1, -2], device="cuda", dtype=torch.int8
    )
    torch.testing.assert_close(actual_s, torch.full_like(actual_s, 2.0), rtol=0, atol=0)
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)

    ref_q, ref_s = _reference(x, torch.int8, ue8m0)
    torch.testing.assert_close(actual_s, ref_s, rtol=0, atol=0)
    torch.testing.assert_close(actual_q, ref_q, rtol=0, atol=0)


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
