# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end correctness tests for the SM120 FP8 paged-KV FMHA integration.

Tests ``sm120_fmha_fp8_paged_prefill`` against a float32 PyTorch reference.
Covers uniform and variable-length packed Q, MHA and GQA, causal and non-causal.

Run on an SM120 GPU:
    pytest tests/attention/test_sm120_fmha_paged.py -v
"""

import math

import pytest
import torch

from flashinfer.cute_dsl.availability import is_cute_dsl_experimental_available
from flashinfer.utils import get_compute_capability

if not is_cute_dsl_experimental_available():
    pytest.skip("SM120 CuTe DSL dependencies are unavailable", allow_module_level=True)

_device = torch.device("cuda") if torch.cuda.is_available() else None

pytestmark = pytest.mark.skipif(
    _device is None or get_compute_capability(_device) != (12, 0),
    reason="SM120 FMHA requires SM120 GPU (compute capability 12.0)",
)

from flashinfer.attention.cute_dsl.sm120_fmha import sm120_fmha_fp8_paged_prefill

# ---------------------------------------------------------------------------
# Paged KV construction helpers
# ---------------------------------------------------------------------------


def _make_paged_kv(k_dense, v_dense, page_size):
    """Convert dense K/V to paged format for testing.

    Parameters
    ----------
    k_dense : (B, Skv, Hkv, D)  float8
    v_dense : (B, Skv, Hkv, D)  float8
    page_size : int  tokens per page

    Returns
    -------
    k_pool : (total_k_pages, Hkv, page_size, D)
    v_pool : (total_v_pages, Hkv, page_size, D)
    block_tables : (B, pages_per_seq)  int32
    """
    B, Skv, Hkv, D = k_dense.shape
    assert Skv % page_size == 0, f"Skv={Skv} must be divisible by page_size={page_size}"
    pages_per_seq = Skv // page_size

    # Populate one standard combined HND cache and return its K/V plane views.
    # Their page stride includes both planes, exercising the zero-copy runtime
    # contract used by BatchPrefillWithPagedKVCacheWrapper.
    kv_pool = torch.empty(
        B * pages_per_seq,
        2,
        Hkv,
        page_size,
        D,
        dtype=k_dense.dtype,
        device=k_dense.device,
    )
    k_pages = k_dense.reshape(B, pages_per_seq, page_size, Hkv, D).permute(
        0, 1, 3, 2, 4
    )
    v_pages = v_dense.reshape(B, pages_per_seq, page_size, Hkv, D).permute(
        0, 1, 3, 2, 4
    )
    kv_pool[:, 0].copy_(k_pages.reshape_as(kv_pool[:, 0]))
    kv_pool[:, 1].copy_(v_pages.reshape_as(kv_pool[:, 1]))
    k_pool, v_pool = kv_pool.unbind(dim=1)

    # Separate K/V pools share one physical page-ID table.
    block_tables = torch.arange(
        B * pages_per_seq, dtype=torch.int32, device="cuda"
    ).reshape(B, pages_per_seq)

    return k_pool.cuda(), v_pool.cuda(), block_tables


def _ref_paged_fmha_single(q_b, k_b, v_b, sm_scale, is_causal, kv_len=None):
    """Float32 reference for one batch item.  q_b: (sq, Hq, D), k_b: (skv, Hkv, D).
    Paged causal: bottom-right aligned (q_offset = kv_len - q_len)."""
    sq, Hq, D = q_b.shape
    skv = k_b.shape[0]
    if kv_len is None:
        kv_len = skv
    Hkv = k_b.shape[1]

    q_f = q_b.float().permute(1, 0, 2)  # (Hq, sq, D)
    k_f = k_b.float().permute(1, 0, 2)  # (Hkv, skv, D)
    v_f = v_b.float().permute(1, 0, 2)
    if Hq != Hkv:
        k_f = k_f.repeat_interleave(Hq // Hkv, dim=0)
        v_f = v_f.repeat_interleave(Hq // Hkv, dim=0)

    scores = torch.einsum("hqd,hkd->hqk", q_f, k_f) * sm_scale
    if kv_len < skv:
        scores[:, :, kv_len:] = float("-inf")
    if is_causal:
        # bottom-right aligned: query i attends to key j when j <= i + q_offset
        q_offset = kv_len - sq
        q_idx = (torch.arange(sq, device=q_b.device) + q_offset).view(-1, 1)
        k_idx = torch.arange(kv_len, device=q_b.device).view(1, -1)
        causal_mask = k_idx > q_idx  # (sq, kv_len)
        scores[:, :sq, :kv_len] = scores[:, :sq, :kv_len].masked_fill(
            causal_mask, float("-inf")
        )
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,hkd->hqd", attn, v_f).permute(1, 0, 2)  # (sq, Hq, D)


def _make_fp8(shape, dtype=torch.float8_e4m3fn):
    return torch.randint(-2, 3, shape, dtype=torch.float32, device="cuda").to(dtype)


def _tol():
    return dict(atol=0.2, rtol=0.2)


# ---------------------------------------------------------------------------
# Uniform packed Q + paged KV
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "B,Sq,Skv,Hq,Hkv,D,page_size",
    [
        (1, 128, 128, 4, 4, 32, 64),  # head_dim=32
        (1, 128, 128, 8, 8, 128, 64),  # MHA, 2 pages/seq
        (2, 64, 128, 8, 2, 128, 64),  # GQA 4:1
        (1, 128, 128, 4, 4, 64, 64),  # head_dim=64
        (1, 129, 384, 2, 1, 256, 64),  # D=256 three-stage KV ring
    ],
)
@pytest.mark.parametrize("is_causal", [False, True])
def test_sm120_paged_uniform_q(B, Sq, Skv, Hq, Hkv, D, page_size, is_causal):
    """Uniform Q packed as (B * Sq, Hq, D) + paged KV correctness."""
    sm_scale = 1.0 / math.sqrt(D)
    in_dtype, out_dtype = torch.float8_e4m3fn, torch.float16

    q_dense = _make_fp8((B, Sq, Hq, D), in_dtype)
    q = q_dense.reshape(B * Sq, Hq, D)
    k = _make_fp8((B, Skv, Hkv, D), in_dtype)
    v = _make_fp8((B, Skv, Hkv, D), in_dtype)
    o = torch.empty(B * Sq, Hq, D, device="cuda", dtype=out_dtype)

    k_pool, v_pool, block_tables = _make_paged_kv(k, v, page_size)
    seqlens_kv = torch.full((B,), Skv, dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.arange(B + 1, dtype=torch.int32, device="cuda") * Sq

    sm120_fmha_fp8_paged_prefill(
        q,
        k_pool,
        v_pool,
        o,
        block_tables=block_tables,
        seqlens_kv=seqlens_kv,
        cu_seqlens_q=cu_seqlens_q,
        is_causal=is_causal,
        sm_scale=sm_scale,
        max_seqlen_q=Sq,
    )
    torch.cuda.synchronize()

    ref = (
        torch.stack(
            [
                _ref_paged_fmha_single(q_dense[b], k[b], v[b], sm_scale, is_causal, Skv)
                for b in range(B)
            ]
        )
        .reshape_as(o)
        .to(out_dtype)
    )
    torch.testing.assert_close(o, ref, **_tol())


# ---------------------------------------------------------------------------
# Paged KV with variable KV lengths (seqlens_kv < Skv)
# ---------------------------------------------------------------------------


def test_sm120_paged_variable_kv_lengths():
    """Paged KV with different actual KV lengths per batch item."""
    B, Sq, Skv, Hq, Hkv, D, page_size = 2, 128, 128, 4, 4, 128, 64
    sm_scale = 1.0 / math.sqrt(D)
    in_dtype, out_dtype = torch.float8_e4m3fn, torch.float16

    q_dense = _make_fp8((B, Sq, Hq, D), in_dtype)
    q = q_dense.reshape(B * Sq, Hq, D)
    k = _make_fp8((B, Skv, Hkv, D), in_dtype)
    v = _make_fp8((B, Skv, Hkv, D), in_dtype)
    o = torch.empty(B * Sq, Hq, D, device="cuda", dtype=out_dtype)

    k_pool, v_pool, block_tables = _make_paged_kv(k, v, page_size)
    seqlens_kv = torch.tensor([64, 128], dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.arange(B + 1, dtype=torch.int32, device="cuda") * Sq

    sm120_fmha_fp8_paged_prefill(
        q,
        k_pool,
        v_pool,
        o,
        block_tables=block_tables,
        seqlens_kv=seqlens_kv,
        cu_seqlens_q=cu_seqlens_q,
        is_causal=False,
        sm_scale=sm_scale,
        max_seqlen_q=Sq,
    )
    torch.cuda.synchronize()

    ref = (
        torch.stack(
            [
                _ref_paged_fmha_single(
                    q_dense[b], k[b], v[b], sm_scale, False, int(seqlens_kv[b].item())
                )
                for b in range(B)
            ]
        )
        .reshape_as(o)
        .to(out_dtype)
    )
    torch.testing.assert_close(o, ref, **_tol())


# ---------------------------------------------------------------------------
# Packed-varlen Q + paged KV
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "Hq,Hkv,D,page_size",
    [
        (8, 8, 128, 64),  # MHA
        (8, 2, 128, 64),  # GQA 4:1
    ],
)
@pytest.mark.parametrize("is_causal", [False, True])
def test_sm120_paged_varlen_q(Hq, Hkv, D, page_size, is_causal):
    """Packed-varlen Q (total_q, Hq, D) + paged KV correctness."""
    sm_scale = 1.0 / math.sqrt(D)
    in_dtype, out_dtype = torch.float8_e4m3fn, torch.float16

    # Two requests: Q lengths [64, 96], KV length 128 each
    q_lens = [64, 96]
    kv_len = 128
    B = len(q_lens)
    Skv = kv_len

    total_q = sum(q_lens)
    cu_seqlens_q = torch.tensor(
        [0, q_lens[0], q_lens[0] + q_lens[1]], dtype=torch.int32, device="cuda"
    )
    seqlens_kv = torch.full((B,), kv_len, dtype=torch.int32, device="cuda")

    # Packed Q: (total_q, Hq, D)
    q_packed = _make_fp8((total_q, Hq, D), in_dtype)
    # Dense K/V for paged construction: (B, Skv, Hkv, D)
    k_dense = _make_fp8((B, Skv, Hkv, D), in_dtype)
    v_dense = _make_fp8((B, Skv, Hkv, D), in_dtype)
    o = torch.empty(total_q, Hq, D, device="cuda", dtype=out_dtype)

    k_pool, v_pool, block_tables = _make_paged_kv(k_dense, v_dense, page_size)

    sm120_fmha_fp8_paged_prefill(
        q_packed,
        k_pool,
        v_pool,
        o,
        block_tables=block_tables,
        seqlens_kv=seqlens_kv,
        cu_seqlens_q=cu_seqlens_q,
        is_causal=is_causal,
        sm_scale=sm_scale,
        max_seqlen_q=max(q_lens),
    )
    torch.cuda.synchronize()

    # Reference: per-request bottom-right aligned causal attention
    for b in range(B):
        q_start = int(cu_seqlens_q[b])
        q_end = int(cu_seqlens_q[b + 1])

        q_b = q_packed[q_start:q_end]  # (sq, Hq, D)
        k_b = k_dense[b, :kv_len]  # (kv_len, Hkv, D)
        v_b = v_dense[b, :kv_len]

        ref_b = _ref_paged_fmha_single(q_b, k_b, v_b, sm_scale, is_causal, kv_len).to(
            out_dtype
        )
        torch.testing.assert_close(o[q_start:q_end], ref_b, **_tol())


# ---------------------------------------------------------------------------
# Compile cache
# ---------------------------------------------------------------------------


def test_sm120_paged_compile_cache():
    """Same paged config repeated 3× must reuse cached compilation (hits >= 2)."""
    from flashinfer.cute_dsl.attention.fmha.sm120.compile import (
        compile_sm120_fmha_fp8_paged_kernel,
    )

    B, Sq, Skv, Hq, Hkv, D, page_size = 1, 128, 128, 4, 4, 128, 64
    in_dtype, out_dtype = torch.float8_e4m3fn, torch.float16

    before = compile_sm120_fmha_fp8_paged_kernel.cache_info()
    for _ in range(3):
        q = _make_fp8((B * Sq, Hq, D), in_dtype)
        k = _make_fp8((B, Skv, Hkv, D), in_dtype)
        v = _make_fp8((B, Skv, Hkv, D), in_dtype)
        o = torch.empty(B * Sq, Hq, D, device="cuda", dtype=out_dtype)
        k_pool, v_pool, block_tables = _make_paged_kv(k, v, page_size)
        seqlens_kv = torch.full((B,), Skv, dtype=torch.int32, device="cuda")
        cu_seqlens_q = torch.arange(B + 1, dtype=torch.int32, device="cuda") * Sq
        sm120_fmha_fp8_paged_prefill(
            q,
            k_pool,
            v_pool,
            o,
            block_tables=block_tables,
            seqlens_kv=seqlens_kv,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=Sq,
        )
    after = compile_sm120_fmha_fp8_paged_kernel.cache_info()
    new_calls = (after.hits + after.misses) - (before.hits + before.misses)
    new_misses = after.misses - before.misses
    # 3 calls total; misses must be at most 1 (first call in this test or from earlier)
    assert new_calls == 3, f"expected 3 cache lookups, got {new_calls}"
    assert new_misses <= 1, (
        f"expected at most 1 miss (got {new_misses}), means no caching"
    )
