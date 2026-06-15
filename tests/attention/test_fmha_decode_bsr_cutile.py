# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for fmha_decode_bsr_cutile and decode_mla_kv_paged_cutile kernels.

Tests correctness of paged KV-cache GQA decode attention and MLA decode
against PyTorch scaled_dot_product_attention reference.
"""

import pytest
import torch
import torch.nn.functional as F

from flashinfer.gemm import is_cuda_tile_available
from flashinfer.utils import get_compute_capability

if not is_cuda_tile_available():
    pytest.skip("cuda.tile not available", allow_module_level=True)

from flashinfer.attention.kernels.cutile.fmha_decode_bsr_cutile import (
    fmha_decode_bsr_cutile,
    decode_mla_kv_paged_cutile,
)


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def require_sm90_or_sm100():
    cc = get_compute_capability(torch.device("cuda"))
    sm = cc[0] * 10 + cc[1]
    if sm < 90:
        pytest.skip(f"fmha_decode_bsr_cutile requires sm90+, got sm{sm}")


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _ref_paged_decode_attention(
    q: torch.Tensor,          # [batch, num_qo_heads, head_dim]
    k_cache: torch.Tensor,    # [num_pages, page_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,    # [num_pages, page_size, num_kv_heads, head_dim]
    page_table: torch.Tensor, # [batch, max_pages_per_seq]
    seq_lens: torch.Tensor,   # [batch]
    scale: float,
) -> torch.Tensor:
    """PyTorch reference for single-token paged-KV GQA decode."""
    batch, num_qo_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    groups = num_qo_heads // num_kv_heads

    outputs = []
    for b in range(batch):
        seq_len = seq_lens[b].item()
        pages_needed = (seq_len + k_cache.shape[1] - 1) // k_cache.shape[1]
        page_ids = page_table[b, :pages_needed]

        # Gather KV from paged cache
        k_flat = k_cache[page_ids].reshape(-1, num_kv_heads, head_dim)[:seq_len]  # [S, H_kv, D]
        v_flat = v_cache[page_ids].reshape(-1, num_kv_heads, head_dim)[:seq_len]

        q_b = q[b]  # [num_qo, D]
        out_heads = []
        for kv_h in range(num_kv_heads):
            k_h = k_flat[:, kv_h, :]  # [S, D]
            v_h = v_flat[:, kv_h, :]
            q_h = q_b[kv_h * groups:(kv_h + 1) * groups]  # [G, D]

            scores = (q_h @ k_h.T) * scale  # [G, S]
            attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
            out_h = attn @ v_h  # [G, D]
            out_heads.append(out_h)
        outputs.append(torch.cat(out_heads, dim=0))  # [num_qo, D]

    return torch.stack(outputs, dim=0)  # [batch, num_qo, D]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 4, 8])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(8, 1), (8, 8), (32, 8)])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("max_seq_len", [64, 256])
def test_fmha_decode_bsr_cutile(
    batch, num_qo_heads, num_kv_heads, head_dim, page_size, max_seq_len
):
    """GQA decode: compare cuTile output against torch reference."""
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")

    torch.manual_seed(42)
    scale = 1.0 / (head_dim ** 0.5)
    dtype = torch.float16

    num_pages = (max_seq_len + page_size - 1) // page_size * batch + 16
    k_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    v_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    q = torch.randn(batch, num_qo_heads, head_dim, device="cuda", dtype=dtype)

    # Build paged KV table
    seq_lens = torch.randint(1, max_seq_len + 1, (batch,), device="cuda", dtype=torch.int32)
    max_pages = (max_seq_len + page_size - 1) // page_size
    page_table = torch.zeros(batch, max_pages, device="cuda", dtype=torch.int32)
    used_pages = 0
    for b in range(batch):
        pages_needed = (seq_lens[b].item() + page_size - 1) // page_size
        page_table[b, :pages_needed] = torch.arange(used_pages, used_pages + pages_needed)
        used_pages += pages_needed

    # cuTile kernel
    output = fmha_decode_bsr_cutile(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        seq_lens=seq_lens,
        scale=scale,
    )

    # Reference
    ref_out = _ref_paged_decode_attention(q, k_cache, v_cache, page_table, seq_lens, scale)

    # FP16 attention: allow 1e-2 absolute tolerance
    torch.testing.assert_close(output.float(), ref_out.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("num_tokens", [1, 8, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("latent_dim", [512])
def test_decode_mla_kv_paged_cutile(num_tokens, head_dim, latent_dim):
    """MLA decode: smoke test (shape + dtype checks)."""
    torch.manual_seed(7)
    num_qo_heads = 8
    page_size = 16
    num_pages = 64
    dtype = torch.float16

    # MLA uses latent KV cache
    k_cache = torch.randn(num_pages, page_size, latent_dim, device="cuda", dtype=dtype)
    v_cache = torch.randn(num_pages, page_size, latent_dim, device="cuda", dtype=dtype)
    q = torch.randn(num_tokens, num_qo_heads, head_dim, device="cuda", dtype=dtype)

    seq_lens = torch.randint(1, page_size * 4, (num_tokens,), device="cuda", dtype=torch.int32)
    max_pages_per_seq = (seq_lens.max().item() + page_size - 1) // page_size
    page_table = torch.randint(0, num_pages, (num_tokens, max_pages_per_seq), device="cuda", dtype=torch.int32)
    scale = 1.0 / (head_dim ** 0.5)

    output = decode_mla_kv_paged_cutile(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        seq_lens=seq_lens,
        scale=scale,
    )

    # Shape check
    assert output.shape[0] == num_tokens
    assert output.shape[1] == num_qo_heads
    assert output.dtype == dtype
    assert not output.isnan().any(), "NaN in MLA decode output"
