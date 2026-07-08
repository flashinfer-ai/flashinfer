# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for per_token_group_quant_8bit_cutile kernel."""

import importlib.util
import pathlib
import sys

import pytest
import torch

_REPO = pathlib.Path(__file__).resolve().parent.parent.parent

def _load_module(name, rel_path):
    path = _REPO / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

_common = _load_module("cutile_common", "flashinfer/gemm/kernels/cutile/cutile_common.py")
is_cuda_tile_available = _common.is_cuda_tile_available

if not is_cuda_tile_available():
    pytest.skip("cuda.tile not available", allow_module_level=True)

_mod = _load_module(
    "per_token_group_quant_8bit_cutile",
    "flashinfer/quantization/kernels/cutile/per_token_group_quant_8bit_cutile.py",
)
per_token_group_quant_8bit_cutile = _mod.per_token_group_quant_8bit_cutile


@pytest.fixture(autouse=True)
def require_sm90():
    cc = torch.cuda.get_device_capability()
    if cc[0] * 10 + cc[1] < 90:
        pytest.skip(f"requires sm90+, got sm{cc[0]*10+cc[1]}")


def _ref_quant(x, group_size, dst_dtype):
    """Pure-PyTorch reference: per-token-group quantization."""
    orig = x.shape
    K = orig[-1]
    n_groups = K // group_size
    xf = x.reshape(-1, n_groups, group_size).float()
    N = xf.shape[0]
    max_val = 127.0 if dst_dtype == torch.int8 else 448.0
    abs_max = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scales = (abs_max / max_val).squeeze(-1)
    x_scaled = (xf / abs_max * max_val).clamp(-max_val, max_val)
    x_q = x_scaled.round().to(dst_dtype) if dst_dtype == torch.int8 else x_scaled.to(dst_dtype)
    return x_q.reshape(orig), scales.reshape(*orig[:-1], n_groups).to(torch.float32)


@pytest.mark.parametrize("num_tokens,hidden_dim", [(128, 2048), (256, 4096)])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("dst_dtype", [torch.int8, torch.float8_e4m3fn])
def test_basic(num_tokens, hidden_dim, group_size, dst_dtype):
    """Basic correctness: scale accuracy + dequant match."""
    torch.manual_seed(42)
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device="cuda")

    x_q, x_s = per_token_group_quant_8bit_cutile(
        x, group_size=group_size, dst_dtype=dst_dtype, column_major_scales=False
    )
    ref_q, ref_s = _ref_quant(x, group_size, dst_dtype)

    assert x_q.shape == x.shape
    assert x_q.dtype == dst_dtype
    assert x_s.shape == (num_tokens, hidden_dim // group_size)

    torch.testing.assert_close(x_s.float(), ref_s.float(), rtol=1e-3, atol=1e-3)

    n_groups = hidden_dim // group_size
    out_dq = x_q.float().reshape(num_tokens, n_groups, group_size) * x_s.reshape(num_tokens, n_groups, 1)
    ref_dq = ref_q.float().reshape(num_tokens, n_groups, group_size) * ref_s.reshape(num_tokens, n_groups, 1)
    # INT8 quantization has coarser resolution (1/127 of max) than FP8 (1/448),
    # so dequant comparison needs a wider absolute tolerance for INT8.
    dq_atol = 5e-2 if dst_dtype == torch.int8 else 1e-2
    torch.testing.assert_close(out_dq, ref_dq, rtol=2e-1, atol=dq_atol)


@pytest.mark.parametrize("num_tokens,hidden_dim", [(128, 2048), (512, 4096)])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("dst_dtype", [torch.int8, torch.float8_e4m3fn])
def test_column_major_scales(num_tokens, hidden_dim, group_size, dst_dtype):
    """Column-major scale layout."""
    torch.manual_seed(7)
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device="cuda")

    x_q, x_s = per_token_group_quant_8bit_cutile(
        x, group_size=group_size, dst_dtype=dst_dtype, column_major_scales=True
    )

    n_groups = hidden_dim // group_size
    # Kernel always returns shape (num_tokens, n_groups) — "column_major" controls the
    # memory layout/strides (Fortran order vs C order), not the logical shape.
    assert x_s.shape == (num_tokens, n_groups), f"expected ({num_tokens},{n_groups}) got {x_s.shape}"
    assert not x_s.isnan().any()


@pytest.mark.parametrize("num_tokens,hidden_dim,group_size", [(128, 2048, 64), (256, 4096, 128)])
def test_scale_ue8m0(num_tokens, hidden_dim, group_size):
    """UE8M0 scale format (Blackwell only)."""
    cc = torch.cuda.get_device_capability()
    if cc[0] < 10:
        pytest.skip("scale_ue8m0 requires sm100+ (Blackwell)")
    torch.manual_seed(99)
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device="cuda")
    x_q, x_s = per_token_group_quant_8bit_cutile(
        x, group_size=group_size, dst_dtype=torch.float8_e4m3fn,
        column_major_scales=True, scale_ue8m0=True
    )
    assert x_q.shape == x.shape
    assert not x_s.isnan().any()
