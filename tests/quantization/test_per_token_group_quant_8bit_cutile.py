# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for per_token_group_quant_8bit_cutile kernel.

Tests correctness against a pure-PyTorch reference implementation,
covering FP8 and INT8 quantization, row-major and column-major scale layouts.
"""

import pytest
import torch

from flashinfer.gemm import is_cuda_tile_available
from flashinfer.utils import get_compute_capability

if not is_cuda_tile_available():
    pytest.skip("cuda.tile not available", allow_module_level=True)

from flashinfer.quantization.kernels.cutile.per_token_group_quant_8bit_cutile import (
    per_token_group_quant_8bit_cutile,
)


# ---------------------------------------------------------------------------
# Reference implementation (pure PyTorch)
# ---------------------------------------------------------------------------


def _ref_per_token_group_quant_8bit(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dst_dtype=torch.float8_e4m3fn,
) -> tuple:
    """Row-major scale reference."""
    x_f = x.float()
    M = x.numel() // group_size
    N = group_size
    x_groups = x_f.reshape(M, N)

    if dst_dtype == torch.int8:
        bit8_max = float(torch.iinfo(dst_dtype).max)
        bit8_min = float(torch.iinfo(dst_dtype).min)
    else:
        bit8_max = torch.finfo(dst_dtype).max
        bit8_min = torch.finfo(dst_dtype).min

    abs_max = x_groups.abs().max(dim=1).values.clamp(min=eps)
    scales = abs_max / bit8_max
    x_q = (x_groups / scales.unsqueeze(1)).clamp(bit8_min, bit8_max).to(dst_dtype)
    return x_q.reshape_as(x), scales.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def require_sm90_or_sm100():
    cc = get_compute_capability(torch.device("cuda"))
    sm = cc[0] * 10 + cc[1]
    if sm < 90:
        pytest.skip(f"per_token_group_quant_8bit_cutile requires sm90+, got sm{sm}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("M", [1, 16, 128, 512])
@pytest.mark.parametrize("N", [64, 128, 256, 512])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("dst_dtype", [torch.float8_e4m3fn, torch.int8])
def test_per_token_group_quant_row_major(M, N, group_size, dst_dtype):
    """Row-major scale layout: compare quantized output and scales vs reference."""
    if N % group_size != 0:
        pytest.skip(f"N={N} not divisible by group_size={group_size}")

    torch.manual_seed(42)
    x = torch.randn(M, N, device="cuda", dtype=torch.float16)

    x_q, x_s = per_token_group_quant_8bit_cutile(
        x, group_size=group_size, dst_dtype=dst_dtype
    )
    x_q_ref, x_s_ref = _ref_per_token_group_quant_8bit(x, group_size, dst_dtype=dst_dtype)

    # Scale check (relative tolerance on fp scales)
    torch.testing.assert_close(x_s, x_s_ref, rtol=1e-3, atol=1e-5)

    # Quantized output: allow ±1 LSB difference (rounding)
    diff = (x_q.float() - x_q_ref.float()).abs().max().item()
    assert diff <= 1.0, f"max quantized diff {diff:.2f} > 1 LSB"

    assert x_q.dtype == dst_dtype
    assert x_s.dtype == torch.float32
    assert x_q.shape == x.shape


@pytest.mark.parametrize("M", [16, 64])
@pytest.mark.parametrize("N", [128, 256])
@pytest.mark.parametrize("group_size", [64, 128])
def test_per_token_group_quant_colmajor(M, N, group_size):
    """Column-major scale layout."""
    if N % group_size != 0:
        pytest.skip(f"N={N} not divisible by group_size={group_size}")

    torch.manual_seed(7)
    x = torch.randn(M, N, device="cuda", dtype=torch.float16)

    x_q_col, x_s_col = per_token_group_quant_8bit_cutile(
        x, group_size=group_size, column_major_scales=True
    )
    x_q_row, x_s_row = per_token_group_quant_8bit_cutile(
        x, group_size=group_size, column_major_scales=False
    )

    # Quantized values should match between layouts
    diff = (x_q_col.float() - x_q_row.float()).abs().max().item()
    assert diff <= 1.0, f"col vs row quant diff {diff:.2f} > 1 LSB"

    # Scale values should also match (transposed)
    num_groups = N // group_size
    # row-major scales: (M, num_groups)
    # col-major scales: (num_groups, M) (transposed)
    x_s_col_t = x_s_col.t() if x_s_col.ndim == 2 else x_s_col
    # Shapes: row-major (M, G), col-major (G, M) before transpose
    # After transpose of col-major → (M, G) — compare directly
    torch.testing.assert_close(x_s_row, x_s_col_t.contiguous(), rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("shape", [(4, 64), (32, 512), (1, 256, 128)])
def test_per_token_group_quant_multidim(shape):
    """Multi-dimensional input tensor."""
    group_size = 64
    if shape[-1] % group_size != 0:
        pytest.skip()

    torch.manual_seed(99)
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16).to(torch.float16)

    x_q, x_s = per_token_group_quant_8bit_cutile(x, group_size=group_size)
    assert x_q.shape == x.shape
    assert x_q.dtype == torch.float8_e4m3fn


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_per_token_group_quant_ue8m0(group_size):
    """UE8M0 (power-of-2) scale mode."""
    M, N = 16, 256
    if N % group_size != 0:
        pytest.skip()

    x = torch.randn(M, N, device="cuda", dtype=torch.float16)
    x_q, x_s = per_token_group_quant_8bit_cutile(
        x, group_size=group_size, column_major_scales=True, scale_ue8m0=True
    )

    # UE8M0 scales must all be powers of 2
    log2_s = torch.log2(x_s.float().abs())
    is_pow2 = (log2_s - log2_s.round()).abs() < 1e-4
    assert is_pow2.all(), "UE8M0 scales must be powers of 2"
