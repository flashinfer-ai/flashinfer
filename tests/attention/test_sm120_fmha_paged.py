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
        (1, 128, 128, 4, 4, 32, 1),  # page_size=1, head_dim=32
        (1, 128, 128, 4, 4, 64, 1),  # page_size=1, head_dim=64
        (2, 64, 128, 8, 2, 128, 1),  # page_size=1, GQA 4:1, D=128
        (1, 129, 384, 24, 4, 256, 1),  # page_size=1, Qwen GQA 6:1, D=256
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
        (8, 2, 128, 1),  # page_size=1, GQA 4:1
        (24, 4, 256, 1),  # page_size=1, Qwen GQA 6:1, D=256
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


# ---------------------------------------------------------------------------
# Token-granular paged KV (page_size=1): fragmented tables & odd lengths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("D", [32, 64, 128, 256])
@pytest.mark.parametrize("is_causal", [False, True])
def test_sm120_paged_page_size_1_fragmented_odd_lengths(D, is_causal):
    """Shuffled page_size=1 block tables with odd/boundary sequence lengths."""
    Hq, Hkv = (24, 4) if D == 256 else (8, 2)
    q_lens = [1, 17, 65, 130]
    kv_lens = [1, 63, 65, 1025]
    B = len(q_lens)
    sm_scale = 1.0 / math.sqrt(D)
    in_dtype, out_dtype = torch.float8_e4m3fn, torch.bfloat16

    total_q = sum(q_lens)
    total_kv = sum(kv_lens)
    max_kv = max(kv_lens)
    cu_seqlens_q = torch.tensor([0, *q_lens], dtype=torch.int32, device="cuda").cumsum(
        0, dtype=torch.int32
    )
    seqlens_kv = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")

    q_packed = _make_fp8((total_q, Hq, D), in_dtype)
    # Allocate extra unused pages and scatter tokens randomly across the pool.
    num_pool_pages = total_kv + 37
    kv_pool = torch.zeros((num_pool_pages, 2, Hkv, 1, D), dtype=in_dtype, device="cuda")
    k_pool, v_pool = kv_pool.unbind(dim=1)
    perm = torch.randperm(num_pool_pages, device="cuda", dtype=torch.int32)

    block_tables = torch.zeros((B, max_kv), dtype=torch.int32, device="cuda")
    k_seqs, v_seqs = [], []
    cursor = 0
    for b, kv_len in enumerate(kv_lens):
        pages = perm[cursor : cursor + kv_len]
        cursor += kv_len
        block_tables[b, :kv_len] = pages
        k_b = _make_fp8((kv_len, Hkv, D), in_dtype)
        v_b = _make_fp8((kv_len, Hkv, D), in_dtype)
        k_pool[pages, :, 0, :] = k_b
        v_pool[pages, :, 0, :] = v_b
        k_seqs.append(k_b)
        v_seqs.append(v_b)

    o = torch.empty((total_q, Hq, D), device="cuda", dtype=out_dtype)
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

    for b in range(B):
        q_start = int(cu_seqlens_q[b])
        q_end = int(cu_seqlens_q[b + 1])
        ref_b = _ref_paged_fmha_single(
            q_packed[q_start:q_end],
            k_seqs[b],
            v_seqs[b],
            sm_scale,
            is_causal,
            kv_lens[b],
        ).to(out_dtype)
        torch.testing.assert_close(o[q_start:q_end], ref_b, **_tol())


# ---------------------------------------------------------------------------
# Sliding window and attention sinks
# ---------------------------------------------------------------------------


def _ref_window_sinks(q_b, k_b, v_b, sm_scale, is_causal, window_left, sinks):
    """FP32 reference with bottom-right causal/window masks and per-head sinks.

    Returns the output and the log2 LSE.
    """
    sq, Hq, _ = q_b.shape
    skv, Hkv, _ = k_b.shape
    q_f = q_b.float().permute(1, 0, 2)
    k_f = k_b.float().permute(1, 0, 2).repeat_interleave(Hq // Hkv, dim=0)
    v_f = v_b.float().permute(1, 0, 2).repeat_interleave(Hq // Hkv, dim=0)
    scores = torch.einsum("hqd,hkd->hqk", q_f, k_f) * sm_scale
    q_pos = (torch.arange(sq, device=q_b.device) + skv - sq).view(-1, 1)
    k_pos = torch.arange(skv, device=q_b.device).view(1, -1)
    if is_causal:
        scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
    if window_left >= 0:
        scores = scores.masked_fill(k_pos < q_pos - window_left, float("-inf"))
    if sinks is not None:
        scores = torch.cat([scores, sinks[:, None, None].expand(-1, sq, 1)], dim=-1)
    lse = torch.logsumexp(scores, dim=-1).permute(1, 0) * math.log2(math.e)
    attn = torch.softmax(scores, dim=-1)
    if sinks is not None:
        attn = attn[..., :-1]
    return torch.einsum("hqk,hkd->hqd", attn, v_f).permute(1, 0, 2), lse


def _scattered_paged_inputs(q_lens, kv_lens, Hq, Hkv, D, page_size):
    """Packed Q plus a shuffled paged K/V pool and its block table."""
    pages = [-(-kv // page_size) for kv in kv_lens]
    num_pool_pages = sum(pages) + 3
    kv_pool = _make_fp8((num_pool_pages, 2, Hkv, page_size, D))
    k_pool, v_pool = kv_pool.unbind(dim=1)
    perm = torch.randperm(num_pool_pages, device="cuda", dtype=torch.int32)
    block_tables = torch.zeros((len(kv_lens), max(pages)), dtype=torch.int32)
    k_seqs, v_seqs, cursor = [], [], 0
    for b, (kv_len, n) in enumerate(zip(kv_lens, pages, strict=True)):
        ids = perm[cursor : cursor + n]
        cursor += n
        block_tables[b, :n] = ids.cpu()
        k_seqs.append(k_pool[ids].transpose(1, 2).reshape(-1, Hkv, D)[:kv_len])
        v_seqs.append(v_pool[ids].transpose(1, 2).reshape(-1, Hkv, D)[:kv_len])
    q = _make_fp8((sum(q_lens), Hq, D))
    cu_seqlens_q = torch.tensor([0, *q_lens], device="cuda").cumsum(
        0, dtype=torch.int32
    )
    seqlens_kv = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    return (
        q,
        k_pool,
        v_pool,
        block_tables.cuda(),
        cu_seqlens_q,
        seqlens_kv,
        k_seqs,
        v_seqs,
    )


@pytest.mark.parametrize("page_size", [1, 16, 128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize(
    "Hq,Hkv,D,gqa_pack_size,with_sinks",
    [
        (64, 8, 64, 8, True),  # GPT-OSS sliding layers
        (32, 16, 256, 2, False),  # Gemma 4 local layers
        (8, 2, 128, 1, True),  # GQA without Q-head packing
    ],
)
def test_sm120_paged_sliding_window_sinks_and_lse(
    page_size, is_causal, Hq, Hkv, D, gqa_pack_size, with_sinks
):
    q_lens = [1, 17, 130, 300]
    kv_lens = [1, 200, 1025, 300] if is_causal else [1, 200, 1025, 129]
    sm_scale = 1.0 / math.sqrt(D)
    q, k_pool, v_pool, block_tables, cu_q, seqlens_kv, k_seqs, v_seqs = (
        _scattered_paged_inputs(q_lens, kv_lens, Hq, Hkv, D, page_size)
    )
    sinks = torch.randn(Hq, device="cuda") * 2.0 + 1.0 if with_sinks else None
    for window_left in (0, 1, 63, 127, 128, 511, 1023):
        o = torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)
        lse = torch.empty(q.shape[:2], device="cuda", dtype=torch.float32)
        sm120_fmha_fp8_paged_prefill(
            q,
            k_pool,
            v_pool,
            o,
            block_tables=block_tables,
            seqlens_kv=seqlens_kv,
            cu_seqlens_q=cu_q,
            is_causal=is_causal,
            sm_scale=sm_scale,
            max_seqlen_q=max(q_lens),
            lse=lse,
            gqa_pack_size=gqa_pack_size,
            window_left=window_left,
            sinks=sinks,
        )
        torch.cuda.synchronize()
        for b in range(len(q_lens)):
            rows = slice(int(cu_q[b]), int(cu_q[b + 1]))
            ref, ref_lse = _ref_window_sinks(
                q[rows], k_seqs[b], v_seqs[b], sm_scale, is_causal, window_left, sinks
            )
            torch.testing.assert_close(o[rows].float(), ref, **_tol())
            torch.testing.assert_close(lse[rows], ref_lse, **_tol())


@pytest.mark.parametrize("with_sinks", [False, True])
def test_sm120_paged_window_split_kv_merges_to_unsplit(with_sinks):
    """Hand-built split-KV chunks, some holding no key in a row's window, merge to
    the unsplit result; sinks are folded into chunk 0 only."""
    from flashinfer.cascade import merge_states

    Hq, Hkv, D, page_size, window_left = 16, 2, 128, 16, 200
    q_lens, kv_lens = [130, 7], [2000, 1500]
    q, k_pool, v_pool, block_tables, cu_q, seqlens_kv, k_seqs, v_seqs = (
        _scattered_paged_inputs(q_lens, kv_lens, Hq, Hkv, D, page_size)
    )
    sinks = torch.randn(Hq, device="cuda") * 2.0 + 1.0 if with_sinks else None
    # Chunk c of request r covers K/V tiles [begin, end); empty ranges are allowed.
    chunks = [[(12, 14), (9, 10)], [(14, 15), (10, 11)], [(15, 16), (11, 12)]]
    ranges = torch.tensor(
        [list(r) for chunk in chunks for r in chunk], dtype=torch.int32, device="cuda"
    )
    n = len(chunks)
    o_parts = torch.empty((q.shape[0] * n, Hq, D), device="cuda", dtype=torch.bfloat16)
    lse_parts = torch.empty((q.shape[0] * n, Hq), device="cuda", dtype=torch.float32)
    common = dict(
        block_tables=block_tables,
        seqlens_kv=seqlens_kv,
        cu_seqlens_q=cu_q,
        is_causal=True,
        sm_scale=1.0 / math.sqrt(D),
        max_seqlen_q=max(q_lens),
        gqa_pack_size=8,
        window_left=window_left,
        sinks=sinks,
    )
    sm120_fmha_fp8_paged_prefill(
        q, k_pool, v_pool, o_parts, lse=lse_parts, kv_tile_ranges=ranges, **common
    )
    merged, merged_lse = merge_states(
        o_parts.view(q.shape[0], n, Hq, D), lse_parts.view(q.shape[0], n, Hq)
    )
    o = torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(q.shape[:2], device="cuda", dtype=torch.float32)
    sm120_fmha_fp8_paged_prefill(q, k_pool, v_pool, o, lse=lse, **common)
    torch.cuda.synchronize()
    assert torch.isfinite(merged.float()).all()
    # Each chunk quantizes its P to FP8 against its own running max, so split and
    # unsplit outputs differ by FP8 rounding; the LSE is accumulated in FP32.
    torch.testing.assert_close(merged.float(), o.float(), atol=0.1, rtol=0.1)
    torch.testing.assert_close(merged_lse, lse, atol=5e-3, rtol=5e-3)
    for b in range(len(q_lens)):
        rows = slice(int(cu_q[b]), int(cu_q[b + 1]))
        ref, ref_lse = _ref_window_sinks(
            q[rows], k_seqs[b], v_seqs[b], 1.0 / math.sqrt(D), True, window_left, sinks
        )
        torch.testing.assert_close(merged[rows].float(), ref, **_tol())
        torch.testing.assert_close(merged_lse[rows], ref_lse, **_tol())


@pytest.mark.parametrize("D,gqa_pack_size", [(64, 8), (256, 1)])
def test_sm120_paged_window_with_skip_softmax(D, gqa_pack_size):
    """The skip-softmax predicate honours the window: threshold 0 skips nothing
    that matters, so it matches the dense windowed kernel and the reference."""
    Hq, Hkv, page_size, window_left = (64, 8, 16, 127) if D == 64 else (8, 4, 16, 300)
    q_lens, kv_lens = [130, 17], [1025, 600]
    sm_scale = 1.0 / math.sqrt(D)
    q, k_pool, v_pool, block_tables, cu_q, seqlens_kv, k_seqs, v_seqs = (
        _scattered_paged_inputs(q_lens, kv_lens, Hq, Hkv, D, page_size)
    )
    sinks = torch.randn(Hq, device="cuda")
    outs = []
    for threshold in (None, 0.0):
        o = torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)
        sm120_fmha_fp8_paged_prefill(
            q,
            k_pool,
            v_pool,
            o,
            block_tables=block_tables,
            seqlens_kv=seqlens_kv,
            cu_seqlens_q=cu_q,
            is_causal=True,
            sm_scale=sm_scale,
            max_seqlen_q=max(q_lens),
            gqa_pack_size=gqa_pack_size,
            skip_softmax_threshold=threshold,
            window_left=window_left,
            sinks=sinks,
        )
        outs.append(o)
    torch.cuda.synchronize()
    torch.testing.assert_close(outs[1], outs[0], atol=1e-2, rtol=1e-2)
    for b in range(len(q_lens)):
        rows = slice(int(cu_q[b]), int(cu_q[b + 1]))
        ref, _ = _ref_window_sinks(
            q[rows], k_seqs[b], v_seqs[b], sm_scale, True, window_left, sinks
        )
        torch.testing.assert_close(outs[1][rows].float(), ref, **_tol())
