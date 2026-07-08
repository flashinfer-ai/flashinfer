# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for fmha_decode_bsr_cutile kernel (paged GQA decode)."""

import importlib.util
import math
import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

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
    "fmha_decode_bsr_cutile",
    "flashinfer/attention/kernels/cutile/fmha_decode_bsr_cutile.py",
)
fmha_decode_bsr_cutile = _mod.fmha_decode_bsr_cutile


@pytest.fixture(autouse=True)
def require_sm90():
    cc = torch.cuda.get_device_capability()
    if cc[0] * 10 + cc[1] < 90:
        pytest.skip(f"requires sm90+, got sm{cc[0]*10+cc[1]}")


def _ref_paged_gqa_decode(q, k_cache, v_cache, block_tables, actual_seq_lens, scale):
    """PyTorch reference: single-token paged GQA decode."""
    batch, num_qo_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    page_size = k_cache.shape[1]
    groups = num_qo_heads // num_kv_heads

    outputs = []
    for b in range(batch):
        seq_len = actual_seq_lens[b].item()
        pages_needed = (seq_len + page_size - 1) // page_size
        page_ids = block_tables[b, :pages_needed]

        k_flat = k_cache[page_ids].reshape(-1, num_kv_heads, head_dim)[:seq_len]
        v_flat = v_cache[page_ids].reshape(-1, num_kv_heads, head_dim)[:seq_len]

        q_b = q[b]
        out_heads = []
        for kv_h in range(num_kv_heads):
            k_h = k_flat[:, kv_h, :]
            v_h = v_flat[:, kv_h, :]
            q_h = q_b[kv_h * groups:(kv_h + 1) * groups]
            scores = (q_h @ k_h.T) * scale
            attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
            out_heads.append(attn @ v_h)
        outputs.append(torch.cat(out_heads, dim=0))
    return torch.stack(outputs, dim=0)


@pytest.mark.parametrize("batch", [1, 4, 8])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(8, 1), (8, 8), (32, 8)])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("max_seq_len", [64, 256])
def test_fmha_decode_bsr_cutile(batch, num_qo_heads, num_kv_heads, head_dim, page_size, max_seq_len):
    """GQA paged decode: compare cuTile output against torch reference."""
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")

    torch.manual_seed(42)
    scale = 1.0 / math.sqrt(head_dim)
    dtype = torch.float16

    num_pages = (max_seq_len + page_size - 1) // page_size * batch + 16
    k_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    v_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    q = torch.randn(batch, num_qo_heads, head_dim, device="cuda", dtype=dtype)

    actual_seq_lens = torch.randint(1, max_seq_len + 1, (batch,), device="cuda", dtype=torch.int32)
    max_pages = (max_seq_len + page_size - 1) // page_size
    block_tables = torch.zeros(batch, max_pages, device="cuda", dtype=torch.int32)
    used = 0
    for b in range(batch):
        n = (actual_seq_lens[b].item() + page_size - 1) // page_size
        block_tables[b, :n] = torch.arange(used, used + n)
        used += n

    # k_scale is the softmax attention scale: kernel computes qk_scale = K_SCALE * INV_LOG_2,
    # so passing 1/sqrt(head_dim) makes the kernel compute exp(q·k / sqrt(D)) correctly.
    output = fmha_decode_bsr_cutile(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        actual_seq_lens=actual_seq_lens,
        block_tables=block_tables,
        k_scale=scale,
        v_scale=1.0,
    )

    ref = _ref_paged_gqa_decode(q, k_cache, v_cache, block_tables, actual_seq_lens, scale)

    assert output.shape == (batch, num_qo_heads, head_dim)
    assert output.dtype == dtype
    assert not output.isnan().any(), "NaN in decode output"
    torch.testing.assert_close(output.float(), ref.float(), rtol=1e-2, atol=1e-2)
