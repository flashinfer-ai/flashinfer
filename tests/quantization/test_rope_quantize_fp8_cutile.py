# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for rope_quantize_fp8_cutile kernel.
# The kernel is MLA-style only:
#   - q_rope: [num_tokens, num_qo_heads, rope_dim]  (3D, per-head)
#   - k_rope: [num_tokens, rope_dim]                (2D, shared across KV heads)
#   - q_nope: [num_tokens, num_qo_heads, nope_dim]  (3D, per-head)
#   - k_nope: [num_tokens, nope_dim]                (2D, shared latent compressed KV)
#
# Loads kernel directly via importlib to bypass flashinfer/__init__.py.

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
    "rope_quantize_fp8_cutile",
    "flashinfer/quantization/kernels/cutile/rope_quantize_fp8_cutile.py",
)
rope_quantize_fp8_cutile = _mod.rope_quantize_fp8_cutile


@pytest.fixture(autouse=True)
def require_sm90():
    cc = torch.cuda.get_device_capability()
    if cc[0] * 10 + cc[1] < 90:
        pytest.skip(f"requires sm90+, got sm{cc[0]*10+cc[1]}")


def _make_cos_sin_cache(max_seq_len: int, rope_dim: int, base: float = 10000.0) -> torch.Tensor:
    """Build float32 cos_sin_cache [max_seq_len, rope_dim].
    Format: [cos_half | sin_half] — matches interleaved RoPE (is_neox=False).
    """
    half = rope_dim // 2
    theta = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, theta)          # [max_seq_len, half]
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).cuda()  # [max_seq_len, rope_dim]


# ---------------------------------------------------------------------------
# Test 1: Basic smoke test — MLA shapes, no nope
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_tokens,num_qo_heads,rope_dim", [
    (32, 8, 64),
    (64, 16, 128),
    (128, 4, 64),
])
def test_rope_quantize_fp8_basic(num_tokens, num_qo_heads, rope_dim):
    """Smoke test: correct shapes, dtype=float8_e4m3fn, no NaN.
    MLA: k_rope is 2D [num_tokens, rope_dim].
    """
    torch.manual_seed(42)
    dtype = torch.float16

    q_rope = torch.randn(num_tokens, num_qo_heads, rope_dim, device="cuda", dtype=dtype)
    k_rope = torch.randn(num_tokens, rope_dim, device="cuda", dtype=dtype)  # 2D MLA
    cos_sin_cache = _make_cos_sin_cache(max_seq_len=4096, rope_dim=rope_dim)
    pos_ids = torch.arange(num_tokens, device="cuda", dtype=torch.int32)

    q_out, k_out, q_nope_out, k_nope_out = rope_quantize_fp8_cutile(
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=None,
        k_nope=None,
        cos_sin_cache=cos_sin_cache,
        pos_ids=pos_ids,
        is_neox=False,
    )

    assert q_out.shape == (num_tokens, num_qo_heads, rope_dim), f"q shape: {q_out.shape}"
    assert k_out.shape == (num_tokens, rope_dim), f"k shape: {k_out.shape}"
    assert q_out.dtype == torch.float8_e4m3fn, f"q dtype: {q_out.dtype}"
    assert k_out.dtype == torch.float8_e4m3fn, f"k dtype: {k_out.dtype}"
    # No nope → nope outputs are empty tensors
    assert q_nope_out.numel() == 0
    assert k_nope_out.numel() == 0
    assert not q_out.isnan().any(), "NaN in q output"
    assert not k_out.isnan().any(), "NaN in k output"


# ---------------------------------------------------------------------------
# Test 2: Quantization scale variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("quant_scale", [0.5, 1.0, 2.0])
def test_quant_scale(quant_scale):
    """quant_scale_q / quant_scale_kv: accepted without error, output is non-NaN."""
    num_tokens, num_qo_heads, rope_dim = 32, 8, 64
    torch.manual_seed(7)
    dtype = torch.float16

    q_rope = torch.randn(num_tokens, num_qo_heads, rope_dim, device="cuda", dtype=dtype)
    k_rope = torch.randn(num_tokens, rope_dim, device="cuda", dtype=dtype)
    cos_sin_cache = _make_cos_sin_cache(4096, rope_dim)
    pos_ids = torch.arange(num_tokens, device="cuda", dtype=torch.int32)

    q_out, k_out, _, _ = rope_quantize_fp8_cutile(
        q_rope=q_rope, k_rope=k_rope, q_nope=None, k_nope=None,
        cos_sin_cache=cos_sin_cache, pos_ids=pos_ids, is_neox=False,
        quant_scale_q=quant_scale, quant_scale_kv=quant_scale,
    )
    assert q_out.shape == (num_tokens, num_qo_heads, rope_dim)
    assert k_out.shape == (num_tokens, rope_dim)
    assert not q_out.isnan().any()
    assert not k_out.isnan().any()


# ---------------------------------------------------------------------------
# Test 3: With nope tensors (MLA split)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rope_dim,nope_dim", [(64, 32), (128, 64)])
def test_with_nope(rope_dim, nope_dim):
    """MLA-style: q_nope and k_nope provided.
    MLA: k_nope is 2D [num_tokens, nope_dim] (shared compressed latent KV).
    """
    num_tokens, num_qo_heads = 32, 8
    torch.manual_seed(11)
    dtype = torch.float16

    q_rope = torch.randn(num_tokens, num_qo_heads, rope_dim, device="cuda", dtype=dtype)
    k_rope = torch.randn(num_tokens, rope_dim, device="cuda", dtype=dtype)       # 2D MLA
    q_nope = torch.randn(num_tokens, num_qo_heads, nope_dim, device="cuda", dtype=dtype)
    k_nope = torch.randn(num_tokens, nope_dim, device="cuda", dtype=dtype)       # 2D MLA
    cos_sin_cache = _make_cos_sin_cache(4096, rope_dim)
    pos_ids = torch.arange(num_tokens, device="cuda", dtype=torch.int32)

    q_out, k_out, q_nope_out, k_nope_out = rope_quantize_fp8_cutile(
        q_rope=q_rope, k_rope=k_rope, q_nope=q_nope, k_nope=k_nope,
        cos_sin_cache=cos_sin_cache, pos_ids=pos_ids, is_neox=False,
    )

    # MLA output shapes: q is per-head (3D), k is shared (2D)
    assert q_out.shape == (num_tokens, num_qo_heads, rope_dim), f"q_out shape: {q_out.shape}"
    assert k_out.shape == (num_tokens, rope_dim), f"k_out shape: {k_out.shape}"
    assert q_nope_out.shape == (num_tokens, num_qo_heads, nope_dim), f"q_nope_out: {q_nope_out.shape}"
    assert k_nope_out.shape == (num_tokens, nope_dim), f"k_nope_out: {k_nope_out.shape}"

    assert not q_out.isnan().any()
    assert not k_out.isnan().any()
    assert not q_nope_out.isnan().any()
    assert not k_nope_out.isnan().any()


# ---------------------------------------------------------------------------
# Test 4: Non-sequential pos_ids
# ---------------------------------------------------------------------------

def test_noncontiguous_pos_ids():
    """pos_ids can be non-sequential (e.g. prefill with cache)."""
    num_tokens, num_qo_heads, rope_dim = 32, 8, 64
    torch.manual_seed(3)
    dtype = torch.float16

    q_rope = torch.randn(num_tokens, num_qo_heads, rope_dim, device="cuda", dtype=dtype)
    k_rope = torch.randn(num_tokens, rope_dim, device="cuda", dtype=dtype)
    cos_sin_cache = _make_cos_sin_cache(4096, rope_dim)
    # Shuffled positions (e.g. cached + new tokens interleaved)
    pos_ids = torch.randperm(num_tokens, device="cuda", dtype=torch.int32)

    q_out, k_out, _, _ = rope_quantize_fp8_cutile(
        q_rope=q_rope, k_rope=k_rope, q_nope=None, k_nope=None,
        cos_sin_cache=cos_sin_cache, pos_ids=pos_ids, is_neox=False,
    )
    assert not q_out.isnan().any()
    assert not k_out.isnan().any()
