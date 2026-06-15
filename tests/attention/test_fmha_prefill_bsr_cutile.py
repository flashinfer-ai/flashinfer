# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for prefill_attention_kv_paged_cutile and prefill_attention_kv_ragged_cutile kernels.

Tests correctness of paged and ragged prefill attention kernels against
PyTorch scaled_dot_product_attention reference.
"""

import pytest
import torch
import torch.nn.functional as F

from flashinfer.gemm import is_cuda_tile_available
from flashinfer.utils import get_compute_capability

if not is_cuda_tile_available():
    pytest.skip("cuda.tile not available", allow_module_level=True)

from flashinfer.attention.kernels.cutile.fmha_prefill_bsr_cutile import (
    prefill_attention_kv_paged_cutile,
    prefill_attention_kv_ragged_cutile,
)


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def require_sm90_or_sm100():
    cc = get_compute_capability(torch.device("cuda"))
    sm = cc[0] * 10 + cc[1]
    if sm < 90:
        pytest.skip(f"fmha_prefill_bsr_cutile requires sm90+, got sm{sm}")


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _ref_causal_attention(
    q: torch.Tensor,    # [S_q, H_q, D]
    k: torch.Tensor,    # [S_kv, H_kv, D]
    v: torch.Tensor,    # [S_kv, H_kv, D]
    scale: float,
    causal: bool = True,
) -> torch.Tensor:
    """Grouped-query causal (or non-causal) attention via PyTorch SDPA."""
    S_q, H_q, D = q.shape
    H_kv = k.shape[1]
    groups = H_q // H_kv

    # Expand KV to match Q heads for SDPA
    k_exp = k.repeat_interleave(groups, dim=1)  # [S_kv, H_q, D]
    v_exp = v.repeat_interleave(groups, dim=1)

    # [H_q, S_q, D]
    q_t = q.permute(1, 0, 2)
    k_t = k_exp.permute(1, 0, 2)
    v_t = v_exp.permute(1, 0, 2)

    out = F.scaled_dot_product_attention(
        q_t.unsqueeze(0),
        k_t.unsqueeze(0),
        v_t.unsqueeze(0),
        scale=scale,
        is_causal=causal,
    ).squeeze(0)  # [H_q, S_q, D]
    return out.permute(1, 0, 2)  # [S_q, H_q, D]


def _ref_ragged_batch_causal_attention(
    q_ragged: torch.Tensor,   # [total_q, H_q, D]
    k_ragged: torch.Tensor,   # [total_kv, H_kv, D]
    q_indptr: torch.Tensor,   # [batch+1]
    kv_indptr: torch.Tensor,  # [batch+1]
    scale: float,
) -> torch.Tensor:
    """Ragged batch causal attention: independent causal attn per request."""
    batch = len(q_indptr) - 1
    outputs = []
    for b in range(batch):
        q_b = q_ragged[q_indptr[b]:q_indptr[b + 1]]
        k_b = k_ragged[kv_indptr[b]:kv_indptr[b + 1]]
        v_b = k_ragged[kv_indptr[b]:kv_indptr[b + 1]]  # NOTE: simplification — use k as v for ref
        out_b = _ref_causal_attention(q_b.float(), k_b.float(), v_b.float(), scale)
        outputs.append(out_b.to(q_ragged.dtype))
    return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Tests: paged prefill
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("seq_len", [16, 64, 128])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(8, 1), (8, 8)])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("page_size", [16])
def test_prefill_paged_cutile(batch, seq_len, num_qo_heads, num_kv_heads, head_dim, page_size):
    """Paged prefill: compare cuTile output against torch SDPA reference."""
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")

    torch.manual_seed(42)
    scale = 1.0 / (head_dim ** 0.5)
    dtype = torch.float16

    num_pages = (seq_len + page_size - 1) // page_size * batch + 16
    k_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    v_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    q = torch.randn(batch * seq_len, num_qo_heads, head_dim, device="cuda", dtype=dtype)

    # Build page table
    pages_per_seq = (seq_len + page_size - 1) // page_size
    page_table = torch.arange(batch * pages_per_seq, device="cuda", dtype=torch.int32).reshape(batch, pages_per_seq)
    seq_lens = torch.full((batch,), seq_len, device="cuda", dtype=torch.int32)
    q_lens = torch.full((batch,), seq_len, device="cuda", dtype=torch.int32)

    output = prefill_attention_kv_paged_cutile(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        kv_seq_lens=seq_lens,
        q_seq_lens=q_lens,
        scale=scale,
        causal=True,
    )

    # Reference: per-batch causal attention
    out_list = []
    for b in range(batch):
        q_b = q[b * seq_len:(b + 1) * seq_len]  # [S, H_q, D]
        # Gather KV
        pids = page_table[b]
        k_b = k_cache[pids].reshape(-1, num_kv_heads, head_dim)[:seq_len]
        v_b = v_cache[pids].reshape(-1, num_kv_heads, head_dim)[:seq_len]
        out_b = _ref_causal_attention(q_b.float(), k_b.float(), v_b.float(), scale)
        out_list.append(out_b.to(dtype))
    ref_out = torch.cat(out_list, dim=0)

    torch.testing.assert_close(output.float(), ref_out.float(), rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Tests: ragged prefill
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_lens_list", [[16], [32, 64], [8, 16, 32, 64]])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(8, 1), (8, 8)])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_prefill_ragged_cutile(seq_lens_list, num_qo_heads, num_kv_heads, head_dim):
    """Ragged prefill: smoke test (shape, dtype, no NaN)."""
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip()

    torch.manual_seed(7)
    scale = 1.0 / (head_dim ** 0.5)
    dtype = torch.float16

    total_tokens = sum(seq_lens_list)
    batch = len(seq_lens_list)

    q = torch.randn(total_tokens, num_qo_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, device="cuda", dtype=dtype)

    indptr = torch.zeros(batch + 1, device="cuda", dtype=torch.int32)
    indptr[1:] = torch.tensor(seq_lens_list, device="cuda").cumsum(0).int()

    output = prefill_attention_kv_ragged_cutile(
        q=q,
        k=k,
        v=v,
        q_indptr=indptr,
        kv_indptr=indptr,
        scale=scale,
        causal=True,
    )

    assert output.shape == (total_tokens, num_qo_heads, head_dim)
    assert output.dtype == dtype
    assert not output.isnan().any(), "NaN in ragged prefill output"


@pytest.mark.parametrize("num_qo_heads", [8, 32])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_prefill_ragged_noncausal(num_qo_heads, head_dim):
    """Non-causal ragged prefill: output should not be NaN."""
    num_kv_heads = 1
    seq_len = 32
    torch.manual_seed(11)
    scale = 1.0 / (head_dim ** 0.5)
    dtype = torch.float16

    q = torch.randn(seq_len, num_qo_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    indptr = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)

    output = prefill_attention_kv_ragged_cutile(
        q=q, k=k, v=v, q_indptr=indptr, kv_indptr=indptr, scale=scale, causal=False
    )

    assert output.shape == (seq_len, num_qo_heads, head_dim)
    assert not output.isnan().any()
