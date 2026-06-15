# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for rope_quantize_fp8_cutile kernel.

Tests correctness against a pure-PyTorch reference that applies RoPE
(interleaved layout) and then quantizes to FP8.
"""

import math

import pytest
import torch

from flashinfer.gemm import is_cuda_tile_available
from flashinfer.utils import get_compute_capability

if not is_cuda_tile_available():
    pytest.skip("cuda.tile not available", allow_module_level=True)

from flashinfer.quantization.kernels.cutile.rope_quantize_fp8_cutile import (
    rope_quantize_fp8_cutile,
)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _apply_rope_interleaved_ref(
    x: torch.Tensor,  # [T, H, D]
    cos: torch.Tensor,  # [T, D//2]
    sin: torch.Tensor,  # [T, D//2]
) -> torch.Tensor:
    """Apply interleaved RoPE (is_neox=False) in float32."""
    x_f = x.float()
    T, H, D = x_f.shape
    half = D // 2
    x_3d = x_f.reshape(T, H, half, 2)
    x_even = x_3d[..., 0]  # [T, H, half]
    x_odd = x_3d[..., 1]

    cos_t = cos.unsqueeze(1)  # [T, 1, half]
    sin_t = sin.unsqueeze(1)

    out_even = x_even * cos_t - x_odd * sin_t
    out_odd = x_odd * cos_t + x_even * sin_t
    out = torch.stack([out_even, out_odd], dim=-1).reshape(T, H, D)
    return out


def _ref_rope_quantize_fp8(
    q_rope, k_rope, cos_sin_cache, pos_ids, quant_scale_q=1.0, quant_scale_kv=1.0
):
    """Reference: interleaved RoPE then FP8 quantization."""
    T = q_rope.shape[0]
    rope_dim = q_rope.shape[2]
    half = rope_dim // 2

    pos = pos_ids.cpu()
    cos = cos_sin_cache[pos, :half].to("cuda")   # [T, half]
    sin = cos_sin_cache[pos, half:].to("cuda")   # [T, half]

    q_rot = _apply_rope_interleaved_ref(q_rope, cos, sin)
    k_rot = _apply_rope_interleaved_ref(k_rope.unsqueeze(1), cos, sin).squeeze(1)

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    fp8_min = torch.finfo(torch.float8_e4m3fn).min

    q_out = (q_rot * quant_scale_q).clamp(fp8_min, fp8_max).to(torch.float8_e4m3fn)
    k_out = (k_rot * quant_scale_kv).clamp(fp8_min, fp8_max).to(torch.float8_e4m3fn)
    return q_out, k_out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def require_sm90_or_sm100():
    cc = get_compute_capability(torch.device("cuda"))
    sm = cc[0] * 10 + cc[1]
    if sm < 90:
        pytest.skip(f"rope_quantize_fp8_cutile requires sm90+, got sm{sm}")


def _make_cos_sin_cache(max_seq_len: int, rope_dim: int) -> torch.Tensor:
    """Build a float32 cos+sin cache of shape [max_seq_len, rope_dim]."""
    half = rope_dim // 2
    theta = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, theta)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to("cuda")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_tokens", [1, 8, 32, 128])
@pytest.mark.parametrize("num_qo_heads", [8, 32])
@pytest.mark.parametrize("num_kv_heads", [1, 8])
@pytest.mark.parametrize("rope_dim", [64, 128])
def test_rope_quantize_fp8_basic(num_tokens, num_qo_heads, num_kv_heads, rope_dim):
    """Compare q_rope_out and k_rope_out against PyTorch reference."""
    if num_kv_heads > num_qo_heads:
        pytest.skip("num_kv_heads > num_qo_heads not supported in MHA mode")

    torch.manual_seed(42)
    max_seq_len = 2048

    q_rope = torch.randn(num_tokens, num_qo_heads, rope_dim, device="cuda", dtype=torch.float16)
    # MLA layout: k_rope is [T, rope_dim] (2D)
    k_rope_2d = torch.randn(num_tokens, rope_dim, device="cuda", dtype=torch.float16)
    cos_sin_cache = _make_cos_sin_cache(max_seq_len, rope_dim)
    pos_ids = torch.randint(0, max_seq_len, (num_tokens,), device="cuda", dtype=torch.int32)

    # cuTile kernel (MLA: k_rope is 2D)
    q_out, k_out, q_nope_out, k_nope_out = rope_quantize_fp8_cutile(
        q_rope=q_rope,
        k_rope=k_rope_2d,
        q_nope=None,
        k_nope=None,
        cos_sin_cache=cos_sin_cache,
        pos_ids=pos_ids,
        is_neox=False,
    )

    # Reference
    q_ref, k_ref = _ref_rope_quantize_fp8(q_rope, k_rope_2d, cos_sin_cache, pos_ids)

    assert q_out.dtype == torch.float8_e4m3fn
    assert k_out.dtype == torch.float8_e4m3fn

    # FP8 has limited precision — compare as float32
    # Allow up to 2 ULP (~2 * 2^-3 ≈ 0.25 for e4m3)
    q_diff = (q_out.float() - q_ref.float()).abs().max().item()
    k_diff = (k_out.float() - k_ref.float()).abs().max().item()
    fp8_ulp = torch.finfo(torch.float8_e4m3fn).smallest_subnormal * 4
    assert q_diff <= 0.5, f"q diff {q_diff:.4f} too large (fp8_ulp={fp8_ulp:.6f})"
    assert k_diff <= 0.5, f"k diff {k_diff:.4f} too large"


@pytest.mark.parametrize("num_tokens", [8, 64])
@pytest.mark.parametrize("quant_scale", [0.5, 1.0, 2.0])
def test_rope_quantize_fp8_quant_scale(num_tokens, quant_scale):
    """Verify quantization scale is applied correctly."""
    num_qo_heads, rope_dim = 8, 64
    torch.manual_seed(3)
    max_seq_len = 512

    q_rope = torch.randn(num_tokens, num_qo_heads, rope_dim, device="cuda", dtype=torch.float16)
    k_rope = torch.randn(num_tokens, rope_dim, device="cuda", dtype=torch.float16)
    cos_sin_cache = _make_cos_sin_cache(max_seq_len, rope_dim)
    pos_ids = torch.arange(num_tokens, device="cuda", dtype=torch.int32)

    q_scale1, _, _, _ = rope_quantize_fp8_cutile(
        q_rope, k_rope, None, None, cos_sin_cache, pos_ids, is_neox=False, quant_scale_q=1.0
    )
    q_scale2, _, _, _ = rope_quantize_fp8_cutile(
        q_rope, k_rope, None, None, cos_sin_cache, pos_ids, is_neox=False, quant_scale_q=quant_scale
    )

    # With 2x scale, values should be ~2x larger (saturating at fp8_max)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    not_saturated = q_scale1.float().abs() < fp8_max * 0.9
    if not_saturated.any():
        ratio = (q_scale2.float()[not_saturated].abs() / (q_scale1.float()[not_saturated].abs() + 1e-6))
        assert ratio.mean().item() == pytest.approx(quant_scale, abs=0.5)


@pytest.mark.parametrize("no_rope_dim", [0, 64, 128])
def test_rope_quantize_fp8_with_nope(no_rope_dim):
    """Test with q_nope/k_nope (non-RoPE dimensions)."""
    num_tokens, num_qo_heads, rope_dim = 16, 8, 64
    torch.manual_seed(5)
    max_seq_len = 256

    q_rope = torch.randn(num_tokens, num_qo_heads, rope_dim, device="cuda", dtype=torch.float16)
    k_rope = torch.randn(num_tokens, rope_dim, device="cuda", dtype=torch.float16)
    cos_sin_cache = _make_cos_sin_cache(max_seq_len, rope_dim)
    pos_ids = torch.arange(num_tokens, device="cuda", dtype=torch.int32)

    q_nope = None
    k_nope = None
    if no_rope_dim > 0:
        q_nope = torch.randn(num_tokens, num_qo_heads, no_rope_dim, device="cuda", dtype=torch.float16)
        k_nope = torch.randn(num_tokens, no_rope_dim, device="cuda", dtype=torch.float16)

    q_out, k_out, q_nope_out, k_nope_out = rope_quantize_fp8_cutile(
        q_rope, k_rope, q_nope, k_nope, cos_sin_cache, pos_ids, is_neox=False
    )

    assert q_out.shape == (num_tokens, num_qo_heads, rope_dim)
    assert k_out.shape == (num_tokens, rope_dim)
    if no_rope_dim > 0:
        assert q_nope_out.shape == q_nope.shape
        assert k_nope_out.shape == k_nope.shape
