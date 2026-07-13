# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for migrated cuTile prefill attention kernels.
# Loads kernels directly via importlib to bypass flashinfer/__init__.py
# (which requires a newer cutlass than the ocean benchmark image).

import importlib.util
import pathlib
import sys

import pytest
import torch

# ---------------------------------------------------------------------------
# Direct kernel loading (bypass flashinfer.__init__.py import chain)
# ---------------------------------------------------------------------------
_REPO = pathlib.Path(__file__).resolve().parent.parent.parent


def _load_module(name, rel_path):
    path = _REPO / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


_common = _load_module(
    "cutile_common",
    "flashinfer/cutile/cutile_common.py",
)
is_cuda_tile_available = _common.is_cuda_tile_available

if not is_cuda_tile_available():
    pytest.skip("cuda.tile not available", allow_module_level=True)

_prefill_mod = _load_module(
    "fmha_prefill_bsr_cutile",
    "flashinfer/attention/kernels/cutile/fmha_prefill_bsr_cutile.py",
)
prefill_attention_kv_paged_cutile = _prefill_mod.prefill_attention_kv_paged_cutile
prefill_attention_kv_ragged_cutile = _prefill_mod.prefill_attention_kv_ragged_cutile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paged_kvcache(batch_seq_lens_kv, page_size, num_kv_heads, head_dim, dtype, device):
    """
    Allocate paged KV cache and build block_tables.
    Returns (k_cache, v_cache, block_tables).
    k_cache/v_cache: [total_pages, page_size, num_kv_heads, head_dim]
    block_tables: [batch, max_pages_per_seq]
    """
    pages_per_seq = [(s + page_size - 1) // page_size for s in batch_seq_lens_kv]
    max_pages_per_seq = max(pages_per_seq)
    total_pages = sum(pages_per_seq)

    k_cache = torch.randn(total_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_cache = torch.randn(total_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=device)

    # block_tables: [batch, max_pages_per_seq], unused slots set to 0
    block_tables = torch.zeros(len(batch_seq_lens_kv), max_pages_per_seq, dtype=torch.int32, device=device)
    page_idx = 0
    for b, n_pages in enumerate(pages_per_seq):
        for p in range(n_pages):
            block_tables[b, p] = page_idx
            page_idx += 1

    return k_cache, v_cache, block_tables


def _make_seq_offsets(seq_lens):
    """Exclusive prefix sum (starting offset for each request in ragged tensor)."""
    offsets = [0]
    for l in seq_lens[:-1]:
        offsets.append(offsets[-1] + l)
    return offsets


# ---------------------------------------------------------------------------
# Test 1: Paged prefill smoke test (causal, GQA)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_lpt", [True, False])
def test_prefill_paged_smoke(use_lpt):
    """
    Smoke test: paged prefill runs and produces valid output shape with no NaN/Inf.
    GQA 4:1 (8 qo heads, 2 kv heads), head_dim=128, page_size=64.
    """
    device = "cuda"
    dtype = torch.bfloat16
    num_qo_heads = 8
    num_kv_heads = 2
    head_dim = 128
    page_size = 64

    batch_seq_lens_q = [128, 128]
    batch_seq_lens_kv = [128, 64]
    num_batch = len(batch_seq_lens_q)
    total_q = sum(batch_seq_lens_q)
    max_seq_len = max(batch_seq_lens_kv)

    # Ragged Q: all sequences concatenated along dim 0
    q = torch.randn(total_q, num_qo_heads, head_dim, dtype=dtype, device=device)

    k_cache, v_cache, block_tables = _make_paged_kvcache(
        batch_seq_lens_kv, page_size, num_kv_heads, head_dim, dtype, device
    )

    actual_seq_lens_q = torch.tensor(batch_seq_lens_q, dtype=torch.int32, device=device)
    actual_seq_lens_kv = torch.tensor(batch_seq_lens_kv, dtype=torch.int32, device=device)
    actual_seq_offset = torch.tensor(
        _make_seq_offsets(batch_seq_lens_q), dtype=torch.int32, device=device
    )

    out, out_lse = prefill_attention_kv_paged_cutile(
        q,
        k_cache,
        v_cache,
        actual_seq_lens_q,
        actual_seq_lens_kv,
        actual_seq_offset,
        block_tables,
        k_scale=1.0,
        v_scale=1.0,
        num_batch=num_batch,
        max_seq_len=max_seq_len,
        is_causal=True,
        use_lpt_scheduler=use_lpt,
    )

    assert out.shape == (total_q, num_qo_heads, head_dim), f"output shape mismatch: {out.shape}"
    assert out_lse.shape == (total_q, num_qo_heads), f"lse shape mismatch: {out_lse.shape}"
    assert not torch.isnan(out).any(), "output contains NaN"
    assert not torch.isinf(out).any(), "output contains Inf"
    assert not torch.isnan(out_lse).any(), "lse contains NaN"


# ---------------------------------------------------------------------------
# Test 2: Paged prefill non-causal
# ---------------------------------------------------------------------------

def test_prefill_paged_noncausal():
    """Non-causal paged prefill: output should differ from causal."""
    device = "cuda"
    dtype = torch.bfloat16
    num_qo_heads = 4
    num_kv_heads = 1
    head_dim = 64
    page_size = 64
    seq_len = 128
    num_batch = 1

    torch.manual_seed(0)
    q = torch.randn(seq_len, num_qo_heads, head_dim, dtype=dtype, device=device)
    k_cache, v_cache, block_tables = _make_paged_kvcache(
        [seq_len], page_size, num_kv_heads, head_dim, dtype, device
    )
    actual_seq_lens_q = torch.tensor([seq_len], dtype=torch.int32, device=device)
    actual_seq_lens_kv = torch.tensor([seq_len], dtype=torch.int32, device=device)
    actual_seq_offset = torch.tensor([0], dtype=torch.int32, device=device)

    out_causal, _ = prefill_attention_kv_paged_cutile(
        q, k_cache, v_cache,
        actual_seq_lens_q, actual_seq_lens_kv, actual_seq_offset,
        block_tables, k_scale=1.0, v_scale=1.0,
        num_batch=num_batch, max_seq_len=seq_len, is_causal=True,
    )
    out_noncausal, _ = prefill_attention_kv_paged_cutile(
        q, k_cache, v_cache,
        actual_seq_lens_q, actual_seq_lens_kv, actual_seq_offset,
        block_tables, k_scale=1.0, v_scale=1.0,
        num_batch=num_batch, max_seq_len=seq_len, is_causal=False,
    )

    assert out_noncausal.shape == (seq_len, num_qo_heads, head_dim)
    assert not torch.isnan(out_noncausal).any()
    # Non-causal sees future tokens → outputs differ
    assert not torch.allclose(out_causal, out_noncausal, atol=1e-2), (
        "causal and non-causal outputs unexpectedly identical"
    )


# ---------------------------------------------------------------------------
# Test 3: Ragged prefill smoke test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_lpt", [True, False])
def test_prefill_ragged_smoke(use_lpt):
    """
    Smoke test: ragged prefill runs with correct output shape and no NaN/Inf.
    MHA (equal qo/kv heads), head_dim=128.
    """
    device = "cuda"
    dtype = torch.bfloat16
    num_qo_heads = 4
    num_kv_heads = 4
    head_dim = 128

    batch_seq_lens_q = [128, 128]
    batch_seq_lens_kv = [128, 128]
    num_batch = 2
    total_q = sum(batch_seq_lens_q)
    total_kv = sum(batch_seq_lens_kv)
    max_seq_len = max(batch_seq_lens_kv)

    q = torch.randn(total_q, num_qo_heads, head_dim, dtype=dtype, device=device)
    # Ragged KV: [total_kv_tokens, num_kv_heads, head_dim]
    k_cache = torch.randn(total_kv, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_cache = torch.randn(total_kv, num_kv_heads, head_dim, dtype=dtype, device=device)

    actual_seq_lens_q = torch.tensor(batch_seq_lens_q, dtype=torch.int32, device=device)
    actual_seq_lens_kv = torch.tensor(batch_seq_lens_kv, dtype=torch.int32, device=device)
    actual_seq_offset = torch.tensor(
        _make_seq_offsets(batch_seq_lens_q), dtype=torch.int32, device=device
    )
    # block_tables accepted but unused by ragged kernel
    block_tables = torch.zeros(num_batch, 1, dtype=torch.int32, device=device)

    out, out_lse = prefill_attention_kv_ragged_cutile(
        q,
        k_cache,
        v_cache,
        actual_seq_lens_q,
        actual_seq_lens_kv,
        actual_seq_offset,
        block_tables,
        k_scale=1.0,
        v_scale=1.0,
        num_batch=num_batch,
        max_seq_len=max_seq_len,
        is_causal=True,
        use_lpt_scheduler=use_lpt,
    )

    assert out.shape == (total_q, num_qo_heads, head_dim), f"output shape mismatch: {out.shape}"
    assert out_lse.shape == (total_q, num_qo_heads), f"lse shape mismatch: {out_lse.shape}"
    assert not torch.isnan(out).any(), "output contains NaN"
    assert not torch.isinf(out).any(), "output contains Inf"


# ---------------------------------------------------------------------------
# Test 4: Ragged prefill GQA
# ---------------------------------------------------------------------------

def test_prefill_ragged_gqa():
    """Ragged prefill with GQA (8 qo heads, 2 kv heads)."""
    device = "cuda"
    dtype = torch.bfloat16
    num_qo_heads = 8
    num_kv_heads = 2
    head_dim = 128
    seq_len = 128
    num_batch = 1

    q = torch.randn(seq_len, num_qo_heads, head_dim, dtype=dtype, device=device)
    k_cache = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_cache = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)

    actual_seq_lens_q = torch.tensor([seq_len], dtype=torch.int32, device=device)
    actual_seq_lens_kv = torch.tensor([seq_len], dtype=torch.int32, device=device)
    actual_seq_offset = torch.tensor([0], dtype=torch.int32, device=device)
    block_tables = torch.zeros(num_batch, 1, dtype=torch.int32, device=device)

    out, out_lse = prefill_attention_kv_ragged_cutile(
        q, k_cache, v_cache,
        actual_seq_lens_q, actual_seq_lens_kv, actual_seq_offset,
        block_tables, k_scale=1.0, v_scale=1.0,
        num_batch=num_batch, max_seq_len=seq_len, is_causal=True,
    )

    assert out.shape == (seq_len, num_qo_heads, head_dim)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


# ---------------------------------------------------------------------------
# Test 5: Paged vs ragged correctness
# ---------------------------------------------------------------------------

def test_prefill_paged_vs_ragged_correctness():
    """
    Paged and ragged prefill should produce numerically equivalent output
    for a single sequence where paged KV stores the same data as ragged KV.
    page_size=64, seq_len=128 → 2 contiguous pages.
    """
    device = "cuda"
    dtype = torch.bfloat16
    num_qo_heads = 4
    num_kv_heads = 1
    head_dim = 64
    page_size = 64
    seq_len = 128
    num_batch = 1

    torch.manual_seed(42)
    q = torch.randn(seq_len, num_qo_heads, head_dim, dtype=dtype, device=device)

    # Ragged KV: [seq_len, num_kv_heads, head_dim]
    k_flat = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_flat = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)

    # Paged KV: reshape ragged → [num_pages, page_size, num_kv_heads, head_dim]
    num_pages = seq_len // page_size  # = 2
    k_cache = k_flat.reshape(num_pages, page_size, num_kv_heads, head_dim).clone()
    v_cache = v_flat.reshape(num_pages, page_size, num_kv_heads, head_dim).clone()
    block_tables_paged = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)

    actual_seq_lens_q = torch.tensor([seq_len], dtype=torch.int32, device=device)
    actual_seq_lens_kv = torch.tensor([seq_len], dtype=torch.int32, device=device)
    actual_seq_offset = torch.tensor([0], dtype=torch.int32, device=device)
    block_tables_ragged = torch.zeros(num_batch, 1, dtype=torch.int32, device=device)

    out_paged, _ = prefill_attention_kv_paged_cutile(
        q, k_cache, v_cache,
        actual_seq_lens_q, actual_seq_lens_kv, actual_seq_offset,
        block_tables_paged, k_scale=1.0, v_scale=1.0,
        num_batch=num_batch, max_seq_len=seq_len, is_causal=True,
    )
    out_ragged, _ = prefill_attention_kv_ragged_cutile(
        q, k_flat, v_flat,
        actual_seq_lens_q, actual_seq_lens_kv, actual_seq_offset,
        block_tables_ragged, k_scale=1.0, v_scale=1.0,
        num_batch=num_batch, max_seq_len=seq_len, is_causal=True,
    )

    assert out_paged.shape == out_ragged.shape
    assert not torch.isnan(out_paged).any()
    assert not torch.isnan(out_ragged).any()

    # Same computation, different cache layout → should match within bf16 tolerance
    max_diff = (out_paged.float() - out_ragged.float()).abs().max().item()
    assert max_diff < 0.05, (
        f"paged vs ragged prefill diverged: max_diff={max_diff:.4f}"
    )
