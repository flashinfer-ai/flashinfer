# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the cute-dsl GQA decode integration.

Covers both the standalone CuTe DSL wrappers
(``BatchDecodeCuteDSLWrapper``/``BatchDecodePagedCuteDSLWrapper``) and the
``backend="cute-dsl"`` path of
:class:`flashinfer.BatchDecodeWithPagedKVCacheWrapper`.

Each unique (head_dim, num_qo_heads, num_kv_heads, dtype) combination
triggers a fresh kernel compilation (a few seconds). Test matrices are kept
small so the full suite compiles a handful of kernels.
"""

import math

import pytest
import torch

import flashinfer
from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.utils import is_sm100a_supported


if not is_cute_dsl_available():
    pytest.skip("CuTe DSL not available", allow_module_level=True)


from flashinfer.cute_dsl.attention import (  # noqa: E402
    BatchDecodeCuteDSLWrapper,
    BatchDecodePagedCuteDSLWrapper,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")),
    reason="cute-dsl GQA decode requires Blackwell (SM100a)",
)


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _decode_reference_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    sm_scale: float,
    o_scale: float = 1.0,
):
    """Single-token GQA decode reference using a paged KV cache (NHD layout)."""
    batch_size = kv_indptr.numel() - 1
    page_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]
    num_qo_heads = q.shape[1]
    assert num_qo_heads % num_kv_heads == 0
    group = num_qo_heads // num_kv_heads

    out = torch.empty_like(q, dtype=torch.float32)
    k_f32 = k_cache.float()
    v_f32 = v_cache.float()
    q_f32 = q.float()
    for b in range(batch_size):
        pages_b = kv_indices[kv_indptr[b] : kv_indptr[b + 1]]
        num_pages = pages_b.numel()
        if num_pages == 0:
            out[b] = 0
            continue
        last_len = int(kv_last_page_len[b].item())
        keys = k_f32[pages_b]  # [num_pages, page_size, num_kv_heads, head_dim]
        values = v_f32[pages_b]
        keys = keys.reshape(num_pages * page_size, num_kv_heads, head_dim)
        values = values.reshape(num_pages * page_size, num_kv_heads, head_dim)
        valid_len = (num_pages - 1) * page_size + last_len
        keys = keys[:valid_len]
        values = values[:valid_len]
        # broadcast KV heads to QO heads for GQA
        keys = keys.repeat_interleave(group, dim=1)
        values = values.repeat_interleave(group, dim=1)
        # logits: [num_qo_heads, valid_len]
        logits = torch.einsum("hd,nhd->hn", q_f32[b], keys) * sm_scale
        probs = torch.softmax(logits, dim=-1)
        out[b] = torch.einsum("hn,nhd->hd", probs, values)
    out *= o_scale
    return out.to(q.dtype)


def _decode_reference_paged_non_causal(
    q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale
):
    """Non-causal multi-query GQA paged reference; q is [b, q_len, h_q, d]."""
    batch_size = kv_indptr.numel() - 1
    page_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]
    group = q.shape[2] // num_kv_heads
    out = torch.empty_like(q, dtype=torch.float32)
    for b in range(batch_size):
        pages_b = kv_indices[kv_indptr[b] : kv_indptr[b + 1]]
        valid_len = (pages_b.numel() - 1) * page_size + int(kv_last_page_len[b].item())
        keys = k_cache[pages_b].float().reshape(-1, num_kv_heads, head_dim)[:valid_len]
        values = (
            v_cache[pages_b].float().reshape(-1, num_kv_heads, head_dim)[:valid_len]
        )
        keys = keys.repeat_interleave(group, dim=1)
        values = values.repeat_interleave(group, dim=1)
        logits = torch.einsum("qhd,nhd->hqn", q[b].float(), keys) * sm_scale
        out[b] = torch.einsum("hqn,nhd->qhd", torch.softmax(logits, dim=-1), values)
    return out.to(q.dtype)


def _decode_reference_contiguous(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
):
    """Ragged (contiguous KV) GQA decode reference; q has shape [b, 1, h_q, d]."""
    batch_size, q_len, num_qo_heads, head_dim = q.shape
    assert q_len == 1
    num_kv_heads = k.shape[2]
    group = num_qo_heads // num_kv_heads
    keys = k.repeat_interleave(group, dim=2).float()  # [b, s, h_q, d]
    values = v.repeat_interleave(group, dim=2).float()
    q_f32 = q.float()
    logits = torch.einsum("bqhd,bnhd->bhqn", q_f32, keys) * sm_scale
    probs = torch.softmax(logits, dim=-1)
    out = torch.einsum("bhqn,bnhd->bqhd", probs, values)
    return out.to(q.dtype)


# ---------------------------------------------------------------------------
# Fixtures / common params
# ---------------------------------------------------------------------------


HEAD_DIM = 128
NUM_QO_HEADS = 32
NUM_KV_HEADS = 4
DEVICE = torch.device("cuda")


# ---------------------------------------------------------------------------
# 1. Standalone ragged wrapper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 4, 17])
@pytest.mark.parametrize("kv_len", [128, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cute_dsl_decode_ragged(batch_size, kv_len, dtype):
    torch.manual_seed(0)
    q = torch.randn(batch_size, 1, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype)
    k = torch.randn(
        batch_size, kv_len, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype
    )
    v = torch.randn_like(k)

    wrapper = BatchDecodeCuteDSLWrapper(
        torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=DEVICE),
    )
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    wrapper.plan(
        batch_size=batch_size,
        max_kv_len=kv_len,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        q_data_type=dtype,
        sm_scale=sm_scale,
    )
    out = wrapper.run(q, k, v)
    ref = _decode_reference_contiguous(q, k, v, sm_scale)
    torch.testing.assert_close(out, ref, rtol=5e-3, atol=5e-3)

    # out= path
    out_buf = torch.zeros_like(out)
    ret = wrapper.run(q, k, v, out=out_buf)
    assert ret.data_ptr() == out_buf.data_ptr()
    torch.testing.assert_close(out_buf, ref, rtol=5e-3, atol=5e-3)


# ---------------------------------------------------------------------------
# 2. Standalone paged wrapper
# ---------------------------------------------------------------------------


def _make_paged_kv(
    batch_size: int,
    kv_len: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
):
    pages_per_seq = (kv_len + page_size - 1) // page_size
    total_pages = pages_per_seq * batch_size
    kv = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
        * pages_per_seq
    )
    kv_indices = torch.arange(0, total_pages, device=device, dtype=torch.int32)
    last = (kv_len - 1) % page_size + 1
    kv_last_page_len = torch.full((batch_size,), last, device=device, dtype=torch.int32)
    seq_lens = torch.full((batch_size,), kv_len, device=device, dtype=torch.int32)
    return kv, kv_indptr, kv_indices, kv_last_page_len, seq_lens


def _make_paged_kv_varlen(
    seq_lens: list[int],
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """Create a paged KV cache with per-request sequence lengths."""
    assert all(seq_len > 0 for seq_len in seq_lens)
    pages_per_seq = [(seq_len + page_size - 1) // page_size for seq_len in seq_lens]
    total_pages = sum(pages_per_seq)
    kv = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    indptr = [0]
    for pages in pages_per_seq:
        indptr.append(indptr[-1] + pages)
    kv_indptr = torch.tensor(indptr, device=device, dtype=torch.int32)
    kv_indices = torch.arange(0, total_pages, device=device, dtype=torch.int32)
    kv_last_page_len = torch.tensor(
        [(seq_len - 1) % page_size + 1 for seq_len in seq_lens],
        device=device,
        dtype=torch.int32,
    )
    seq_lens_tensor = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    return kv, kv_indptr, kv_indices, kv_last_page_len, seq_lens_tensor


@pytest.mark.parametrize("batch_size", [1, 4, 17])
@pytest.mark.parametrize("kv_len", [129, 1024])
@pytest.mark.parametrize("page_size", [16, 32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cute_dsl_decode_paged_wrapper(batch_size, kv_len, page_size, dtype):
    torch.manual_seed(0)
    q = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype)
    kv, kv_indptr, kv_indices, kv_last_page_len, seq_lens = _make_paged_kv(
        batch_size, kv_len, page_size, NUM_KV_HEADS, HEAD_DIM, dtype, DEVICE
    )
    k_cache, v_cache = kv.unbind(dim=1)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    wrapper = BatchDecodePagedCuteDSLWrapper(
        torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=DEVICE),
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        seq_lens,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=page_size,
        q_data_type=dtype,
        sm_scale=sm_scale,
    )
    out = wrapper.run(q, k_cache, v_cache)
    ref = _decode_reference_paged(
        q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale
    )
    torch.testing.assert_close(
        out.reshape(batch_size, NUM_QO_HEADS, HEAD_DIM),
        ref,
        rtol=5e-3,
        atol=5e-3,
    )


@pytest.mark.parametrize("batch_size", [1, 4, 17])
@pytest.mark.parametrize("kv_len", [129, 1024])
@pytest.mark.parametrize("page_size", [16, 32])
@pytest.mark.parametrize("q_len_per_req", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cute_dsl_decode_paged_non_causal(
    batch_size, kv_len, page_size, q_len_per_req, dtype
):
    """Non-causal paged decode (regression: seqlen-boundary masking)."""
    torch.manual_seed(0)
    q = torch.randn(
        batch_size, q_len_per_req, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype
    )
    kv, kv_indptr, kv_indices, kv_last_page_len, seq_lens = _make_paged_kv(
        batch_size, kv_len, page_size, NUM_KV_HEADS, HEAD_DIM, dtype, DEVICE
    )
    k_cache, v_cache = kv.unbind(dim=1)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    wrapper = BatchDecodePagedCuteDSLWrapper(
        torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=DEVICE),
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        seq_lens,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=page_size,
        q_data_type=dtype,
        sm_scale=sm_scale,
        q_len_per_req=q_len_per_req,
        is_causal=False,
    )
    out = wrapper.run(
        q.reshape(batch_size * q_len_per_req, NUM_QO_HEADS, HEAD_DIM), k_cache, v_cache
    )
    ref = _decode_reference_paged_non_causal(
        q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale
    )
    torch.testing.assert_close(
        out.reshape(batch_size, q_len_per_req, NUM_QO_HEADS, HEAD_DIM),
        ref,
        rtol=5e-3,
        atol=5e-3,
    )


def test_cute_dsl_decode_paged_non_causal_split_boundary_tile():
    """Non-causal boundary mask on non-zero split and softmax phase."""
    torch.manual_seed(0)
    seq_lens_host = [16, 256, 385]
    batch_size = len(seq_lens_host)
    page_size = 16
    q_len_per_req = 4
    dtype = torch.bfloat16
    q = torch.randn(
        batch_size, q_len_per_req, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype
    )
    kv, kv_indptr, kv_indices, kv_last_page_len, seq_lens = _make_paged_kv_varlen(
        seq_lens_host, page_size, NUM_KV_HEADS, HEAD_DIM, dtype, DEVICE
    )
    k_cache, v_cache = kv.unbind(dim=1)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    wrapper = BatchDecodePagedCuteDSLWrapper(
        torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=DEVICE),
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        seq_lens,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=page_size,
        q_data_type=dtype,
        sm_scale=sm_scale,
        kv_splits=2,
        reduction="kernel",
        q_len_per_req=q_len_per_req,
        is_causal=False,
        max_kv_len=max(seq_lens_host),
    )
    out = wrapper.run(
        q.reshape(batch_size * q_len_per_req, NUM_QO_HEADS, HEAD_DIM), k_cache, v_cache
    )
    ref = _decode_reference_paged_non_causal(
        q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale
    )
    torch.testing.assert_close(
        out.reshape(batch_size, q_len_per_req, NUM_QO_HEADS, HEAD_DIM),
        ref,
        rtol=5e-3,
        atol=5e-3,
    )


# ---------------------------------------------------------------------------
# 3. BatchDecodeWithPagedKVCacheWrapper(backend="cute-dsl")
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 4, 17])
@pytest.mark.parametrize("kv_len", [129, 1024])
@pytest.mark.parametrize("page_size", [16, 32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_batch_decode_wrapper_cute_dsl_backend(batch_size, kv_len, page_size, dtype):
    """Integration test via the standard BatchDecodeWithPagedKVCacheWrapper API."""
    torch.manual_seed(0)
    q = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype)
    kv, kv_indptr, kv_indices, kv_last_page_len, seq_lens = _make_paged_kv(
        batch_size, kv_len, page_size, NUM_KV_HEADS, HEAD_DIM, dtype, DEVICE
    )

    workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    cd_wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", backend="cute-dsl"
    )
    cd_wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    out = cd_wrapper.run(q, kv)

    # Reference via FA2 backend.
    ref_wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=DEVICE),
        kv_layout="NHD",
        backend="fa2",
    )
    ref_wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    ref = ref_wrapper.run(q, kv)
    torch.testing.assert_close(out, ref, rtol=5e-3, atol=5e-3)

    # user-allocated output
    out_buf = torch.zeros_like(out)
    cd_wrapper.run(q, kv, out=out_buf)
    torch.testing.assert_close(out_buf, ref, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("kv_len", [1024])
@pytest.mark.parametrize("page_size", [16])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_batch_decode_wrapper_cute_dsl_hnd(batch_size, kv_len, page_size, dtype):
    """HND layout is presented as a transposed view; output must match NHD."""
    torch.manual_seed(0)
    q = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype)
    pages_per_seq = (kv_len + page_size - 1) // page_size
    total_pages = pages_per_seq * batch_size
    # HND-laid-out combined KV tensor.
    kv_hnd = torch.randn(
        total_pages, 2, NUM_KV_HEADS, page_size, HEAD_DIM, device=DEVICE, dtype=dtype
    )
    kv_nhd = kv_hnd.transpose(-3, -2).contiguous()  # NHD copy for the reference
    kv_indptr = (
        torch.arange(batch_size + 1, device=DEVICE, dtype=torch.int32) * pages_per_seq
    )
    kv_indices = torch.arange(total_pages, device=DEVICE, dtype=torch.int32)
    last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, device=DEVICE, dtype=torch.int32
    )

    workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    cd = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="HND", backend="cute-dsl"
    )
    cd.plan(
        kv_indptr,
        kv_indices,
        last_page_len,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    out_hnd = cd.run(q, kv_hnd)

    # NHD reference via fa2.
    ref_wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=DEVICE),
        kv_layout="NHD",
        backend="fa2",
    )
    ref_wrapper.plan(
        kv_indptr,
        kv_indices,
        last_page_len,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    ref = ref_wrapper.run(q, kv_nhd)
    torch.testing.assert_close(out_hnd, ref, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("kv_len", [1024])
@pytest.mark.parametrize("page_size", [16])
@pytest.mark.parametrize("q_len_per_req", [2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_batch_decode_wrapper_cute_dsl_speculative(
    batch_size, kv_len, page_size, q_len_per_req, dtype
):
    """Speculative decode: q_len_per_req > 1 with bottom-right causal mask."""
    torch.manual_seed(0)
    q = torch.randn(
        batch_size * q_len_per_req,
        NUM_QO_HEADS,
        HEAD_DIM,
        device=DEVICE,
        dtype=dtype,
    )
    kv, kv_indptr, kv_indices, kv_last_page_len, seq_lens = _make_paged_kv(
        batch_size, kv_len, page_size, NUM_KV_HEADS, HEAD_DIM, dtype, DEVICE
    )

    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    cd = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", backend="cute-dsl"
    )
    cd.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
        q_len_per_req=q_len_per_req,
    )

    trt = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", backend="trtllm-gen"
    )
    trt.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
        q_len_per_req=q_len_per_req,
    )

    out = cd.run(q, kv)
    ref = trt.run(q, kv)

    assert out.shape == ref.shape
    torch.testing.assert_close(
        out,
        ref,
        rtol=5e-3,
        atol=5e-3,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_cute_dsl_decode_paged_wrapper_speculative_runtime(dtype):
    """Runtime q_len doesnt match plan time q_len"""
    batch_size, page_size = 4, 16
    kv_len = 1024
    torch.manual_seed(0)
    kv, kv_indptr, kv_indices, kv_last_page_len, seq_lens = _make_paged_kv(
        batch_size, kv_len, page_size, NUM_KV_HEADS, HEAD_DIM, dtype, DEVICE
    )
    k_cache, v_cache = kv.unbind(dim=1)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    cd = BatchDecodePagedCuteDSLWrapper(workspace)
    cd.plan(
        kv_indptr,
        kv_indices,
        seq_lens,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=page_size,
        q_data_type=dtype,
        sm_scale=sm_scale,
        reduction="kernel",
        q_len_per_req=2,
        max_kv_len=kv_len,
    )

    trt = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", backend="trtllm-gen"
    )
    trt.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
        sm_scale=sm_scale,
        q_len_per_req=2,
        seq_lens=seq_lens,
    )

    for q_len_per_req in (1, 2, 4):
        q = torch.randn(
            batch_size * q_len_per_req,
            NUM_QO_HEADS,
            HEAD_DIM,
            device=DEVICE,
            dtype=dtype,
        )
        out = cd.run(q, k_cache, v_cache)
        ref = trt.run(q, kv)

        assert out.shape == ref.shape
        torch.testing.assert_close(
            out,
            ref,
            rtol=5e-3,
            atol=5e-3,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_batch_decode_wrapper_cute_dsl_v_scale(dtype):
    """v_scale is folded into the cute-dsl kernel's o_scale before output store."""
    batch_size, page_size, kv_len = 4, 16, 1024
    torch.manual_seed(0)
    q = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype)
    kv, kv_indptr, kv_indices, kv_last_page_len, seq_lens = _make_paged_kv(
        batch_size, kv_len, page_size, NUM_KV_HEADS, HEAD_DIM, dtype, DEVICE
    )
    workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    cd = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", backend="cute-dsl"
    )
    cd.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    k = 2.5
    out_scaled = cd.run(q, kv, v_scale=k)
    ref_scaled = _decode_reference_paged(
        q,
        kv[:, 0],
        kv[:, 1],
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        1.0 / math.sqrt(HEAD_DIM),
        o_scale=k,
    )
    torch.testing.assert_close(out_scaled, ref_scaled, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_cute_dsl_decode_paged_wrapper_o_scale(dtype):
    """Standalone BatchDecodePagedCuteDSLWrapper.run(o_scale=k) scales before store."""
    batch_size, page_size, kv_len = 4, 16, 1024
    torch.manual_seed(0)
    q = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype)
    kv, kv_indptr, kv_indices, kv_last_page_len, seq_lens = _make_paged_kv(
        batch_size, kv_len, page_size, NUM_KV_HEADS, HEAD_DIM, dtype, DEVICE
    )
    k_cache, v_cache = kv.unbind(dim=1)
    wrapper = BatchDecodePagedCuteDSLWrapper(
        torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=DEVICE),
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        seq_lens,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=page_size,
        q_data_type=dtype,
    )
    k = 0.375
    out_scaled = wrapper.run(q, k_cache, v_cache, o_scale=k)
    ref_scaled = _decode_reference_paged(
        q,
        k_cache,
        v_cache,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        1.0 / math.sqrt(HEAD_DIM),
        o_scale=k,
    )
    torch.testing.assert_close(out_scaled, ref_scaled, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_batch_decode_wrapper_cute_dsl_skip_softmax_per_cta(dtype):
    """Per-batch threshold normalization: with a varying-seqlen batch and a
    sub-threshold scale factor (no actual skipping), the BLASST path's
    output must still match the standard kernel exactly — verifying that
    the kernel-side divide by per-CTA seqlen runs cleanly and that no
    spurious tiles get skipped."""
    page_size = 16
    # Distinct per-batch seqlens to exercise per-CTA normalization
    # (constant-seqlen batches would coincide with the old behavior).
    seqlens = [128, 512, 1024, 2048]
    batch_size = len(seqlens)
    torch.manual_seed(0)
    q = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype)
    pages_per_seq = [(s + page_size - 1) // page_size for s in seqlens]
    kv_indptr = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(pages_per_seq), 0).tolist()),
        device=DEVICE,
        dtype=torch.int32,
    )
    total_pages = int(kv_indptr[-1].item())
    kv = torch.randn(
        total_pages,
        2,
        page_size,
        NUM_KV_HEADS,
        HEAD_DIM,
        device=DEVICE,
        dtype=dtype,
    )
    kv_indices = torch.arange(total_pages, device=DEVICE, dtype=torch.int32)
    kv_last_page_len = torch.tensor(
        [((s - 1) % page_size + 1) for s in seqlens],
        device=DEVICE,
        dtype=torch.int32,
    )

    workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    cd = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", backend="cute-dsl"
    )
    cd.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    out_std = cd.run(q, kv)
    # Per-CTA effective threshold = 1e-6 / seqlen_i, all far below any
    # softmax probability — output must match the standard path.
    out_skip = cd.run(q, kv, skip_softmax_threshold_scale_factor=1e-6)
    torch.testing.assert_close(out_skip, out_std, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_batch_decode_wrapper_cute_dsl_skip_softmax(dtype):
    """skip_softmax_threshold_scale_factor=0 must match the standard path."""
    batch_size, page_size, kv_len = 4, 16, 1024
    torch.manual_seed(0)
    q = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype)
    kv, kv_indptr, kv_indices, kv_last_page_len, seq_lens = _make_paged_kv(
        batch_size, kv_len, page_size, NUM_KV_HEADS, HEAD_DIM, dtype, DEVICE
    )

    workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    cd = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", backend="cute-dsl"
    )
    cd.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    out_std = cd.run(q, kv)
    # threshold = small positive ⇒ BLASST path runs but nothing should be
    # skipped at this magnitude; output should match the standard path.
    out_skip = cd.run(q, kv, skip_softmax_threshold_scale_factor=1e-6)
    torch.testing.assert_close(out_skip, out_std, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_batch_decode_wrapper_cute_dsl_rejects_unsupported(dtype):
    """The cute-dsl decode backend should reject features it doesn't support."""
    batch_size, page_size, kv_len = 4, 16, 256
    kv, kv_indptr, kv_indices, kv_last_page_len, seq_lens = _make_paged_kv(
        batch_size, kv_len, page_size, NUM_KV_HEADS, HEAD_DIM, dtype, DEVICE
    )
    workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", backend="cute-dsl"
    )
    with pytest.raises(NotImplementedError, match="sliding window"):
        wrapper.plan(
            kv_indptr,
            kv_indices,
            seq_lens,
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            page_size,
            q_data_type=dtype,
            kv_data_type=dtype,
            window_left=64,
        )
    with pytest.raises(NotImplementedError, match="logits_soft_cap"):
        wrapper.plan(
            kv_indptr,
            kv_indices,
            seq_lens,
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            page_size,
            q_data_type=dtype,
            kv_data_type=dtype,
            logits_soft_cap=2.0,
        )
    with pytest.raises(NotImplementedError, match="pos_encoding_mode"):
        wrapper.plan(
            kv_indptr,
            kv_indices,
            seq_lens,
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            page_size,
            q_data_type=dtype,
            kv_data_type=dtype,
            pos_encoding_mode="ROPE_LLAMA",
        )

    # Successful plan; verify run-time rejections.
    wrapper.plan(
        kv_indptr,
        kv_indices,
        seq_lens,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    # (return_lse is now supported — covered by dedicated LSE tests below.)


# ---------------------------------------------------------------------------
# 4. LSE return
# ---------------------------------------------------------------------------


def _lse_reference_paged(
    q: torch.Tensor,  # [batch_size, num_qo_heads, head_dim]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """LSE reference in log2 base, shape [batch_size, num_qo_heads]."""
    batch_size = kv_indptr.numel() - 1
    page_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]
    num_qo_heads = q.shape[1]
    group = num_qo_heads // num_kv_heads
    out = torch.empty(batch_size, num_qo_heads, dtype=torch.float32, device=q.device)
    for b in range(batch_size):
        pages_b = kv_indices[kv_indptr[b] : kv_indptr[b + 1]]
        num_pages = pages_b.numel()
        if num_pages == 0:
            out[b] = -float("inf")
            continue
        last = int(kv_last_page_len[b].item())
        valid_len = (num_pages - 1) * page_size + last
        keys = (
            k_cache[pages_b]
            .float()
            .reshape(num_pages * page_size, num_kv_heads, head_dim)[:valid_len]
        )
        keys = keys.repeat_interleave(group, dim=1)
        logits = torch.einsum("hd,nhd->hn", q[b].float(), keys) * sm_scale
        # natural-log logsumexp, converted to log2
        out[b] = torch.logsumexp(logits, dim=-1) * math.log2(math.e)
    return out


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("kv_len", [129, 1024])
@pytest.mark.parametrize("page_size", [16])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_batch_decode_wrapper_cute_dsl_return_lse(batch_size, kv_len, page_size, dtype):
    """LSE values from cute-dsl backend must match the torch reference (log2-base)."""
    torch.manual_seed(0)
    q = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype)
    kv, kv_indptr, kv_indices, kv_last_page_len, seq_lens = _make_paged_kv(
        batch_size, kv_len, page_size, NUM_KV_HEADS, HEAD_DIM, dtype, DEVICE
    )
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    cd = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", backend="cute-dsl"
    )
    cd.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
        sm_scale=sm_scale,
    )
    out, lse = cd.run(q, kv, return_lse=True)

    k_cache, v_cache = kv.unbind(dim=1)
    lse_ref = _lse_reference_paged(
        q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale
    )
    torch.testing.assert_close(lse, lse_ref, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_cute_dsl_decode_paged_wrapper_lse_both_reductions(dtype):
    """All three reduction modes must write the same LSE."""
    batch_size, page_size = 4, 16
    # Pick a kv_len that lands in "atomic" mode for auto by default, then
    # also force "kernel" / "none" reduction in separate plans.
    kv_len = 1024
    torch.manual_seed(0)
    q = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype)
    kv, kv_indptr, kv_indices, kv_last_page_len, seq_lens = _make_paged_kv(
        batch_size, kv_len, page_size, NUM_KV_HEADS, HEAD_DIM, dtype, DEVICE
    )
    k_cache, v_cache = kv.unbind(dim=1)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    lse_ref = _lse_reference_paged(
        q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale
    )

    for reduction in ("kernel", "atomic", "none"):
        wrapper = BatchDecodePagedCuteDSLWrapper(
            torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=DEVICE),
        )
        plan_kwargs = dict(
            num_qo_heads=NUM_QO_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            page_size=page_size,
            q_data_type=dtype,
            sm_scale=sm_scale,
            reduction=reduction,
        )
        if reduction == "none":
            # "none" only supports kv_splits == 1.
            plan_kwargs["kv_splits"] = 1
        wrapper.plan(
            kv_indptr,
            kv_indices,
            seq_lens,
            **plan_kwargs,
        )
        lse_buf = torch.empty(
            batch_size, 1, NUM_QO_HEADS, dtype=torch.float32, device=DEVICE
        )
        wrapper.run(q, k_cache, v_cache, lse=lse_buf)
        # squeeze q_len_per_req=1 axis to compare against (batch, h_q) ref
        torch.testing.assert_close(
            lse_buf.squeeze(1),
            lse_ref,
            rtol=5e-3,
            atol=5e-3,
            msg=f"reduction={reduction}",
        )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("kv_len", [129, 1024])
@pytest.mark.parametrize("page_size", [16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cute_dsl_decode_paged_wrapper_reduction_none(
    batch_size, kv_len, page_size, dtype
):
    """`reduction="none"` (no flash-decoding split-K): direct write to o_bshd,
    no reduction kernel / workspace. Output must match the standard reference."""
    torch.manual_seed(0)
    q = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype)
    kv, kv_indptr, kv_indices, kv_last_page_len, seq_lens = _make_paged_kv(
        batch_size, kv_len, page_size, NUM_KV_HEADS, HEAD_DIM, dtype, DEVICE
    )
    k_cache, v_cache = kv.unbind(dim=1)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    wrapper = BatchDecodePagedCuteDSLWrapper(
        torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=DEVICE),
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        seq_lens,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=page_size,
        q_data_type=dtype,
        sm_scale=sm_scale,
        reduction="none",
        kv_splits=1,
    )
    out = wrapper.run(q, k_cache, v_cache)
    ref = _decode_reference_paged(
        q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale
    )
    torch.testing.assert_close(
        out.reshape(batch_size, NUM_QO_HEADS, HEAD_DIM),
        ref,
        rtol=5e-3,
        atol=5e-3,
    )


def test_cute_dsl_decode_paged_wrapper_reduction_none_rejects_split():
    """`reduction="none"` requires `kv_splits == 1`."""
    batch_size, page_size, kv_len = 4, 16, 1024
    dtype = torch.bfloat16
    kv, kv_indptr, kv_indices, kv_last_page_len, seq_lens = _make_paged_kv(
        batch_size, kv_len, page_size, NUM_KV_HEADS, HEAD_DIM, dtype, DEVICE
    )
    wrapper = BatchDecodePagedCuteDSLWrapper(
        torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=DEVICE),
    )
    with pytest.raises(ValueError, match="kv_splits == 1"):
        wrapper.plan(
            kv_indptr,
            kv_indices,
            seq_lens,
            num_qo_heads=NUM_QO_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            page_size=page_size,
            q_data_type=dtype,
            reduction="none",
            kv_splits=4,
        )


# ---------------------------------------------------------------------------
# 5. float_workspace_buffer reuse / sizing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_cute_dsl_decode_paged_wrapper_reuses_workspace(dtype):
    """Workspace tensors come from `float_workspace_buffer`, not fresh allocs.
    A second run() with different inputs must still produce correct output —
    confirms m_bsh is re-`-inf`-initialized each call (a missed re-init would
    corrupt the second run's softmax)."""
    batch_size, page_size, kv_len = 8, 16, 1024
    torch.manual_seed(0)
    q1 = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype)
    q2 = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype)
    kv, kv_indptr, kv_indices, kv_last_page_len, seq_lens = _make_paged_kv(
        batch_size, kv_len, page_size, NUM_KV_HEADS, HEAD_DIM, dtype, DEVICE
    )
    k_cache, v_cache = kv.unbind(dim=1)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    wrapper = BatchDecodePagedCuteDSLWrapper(
        torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=DEVICE),
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        seq_lens,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=page_size,
        q_data_type=dtype,
        sm_scale=sm_scale,
        reduction="kernel",
    )
    out1 = wrapper.run(q1, k_cache, v_cache)
    out2 = wrapper.run(q2, k_cache, v_cache)

    ref1 = _decode_reference_paged(
        q1, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale
    )
    ref2 = _decode_reference_paged(
        q2, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale
    )
    torch.testing.assert_close(
        out1.reshape(batch_size, NUM_QO_HEADS, HEAD_DIM),
        ref1,
        rtol=5e-3,
        atol=5e-3,
    )
    torch.testing.assert_close(
        out2.reshape(batch_size, NUM_QO_HEADS, HEAD_DIM),
        ref2,
        rtol=5e-3,
        atol=5e-3,
    )


def test_cute_dsl_decode_paged_wrapper_workspace_too_small():
    """plan() with kernel-reduction must reject undersized float_workspace_buffer."""
    batch_size, page_size, kv_len = 4, 16, 2048
    dtype = torch.bfloat16
    kv, kv_indptr, kv_indices, kv_last_page_len, seq_lens = _make_paged_kv(
        batch_size, kv_len, page_size, NUM_KV_HEADS, HEAD_DIM, dtype, DEVICE
    )
    wrapper = BatchDecodePagedCuteDSLWrapper(
        torch.empty(1024, dtype=torch.uint8, device=DEVICE),  # 1 KiB — way too small
    )
    with pytest.raises(RuntimeError, match="exceeds provided workspace"):
        wrapper.plan(
            kv_indptr,
            kv_indices,
            seq_lens,
            num_qo_heads=NUM_QO_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            page_size=page_size,
            q_data_type=dtype,
            reduction="kernel",
            kv_splits=8,
        )


def test_cute_dsl_decode_paged_wrapper_workspace_unused_for_none():
    """`reduction="none"` (and atomic) don't need workspace — a 1-byte buffer
    must be accepted."""
    batch_size, page_size, kv_len = 4, 16, 1024
    dtype = torch.bfloat16
    torch.manual_seed(0)
    q = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype)
    kv, kv_indptr, kv_indices, kv_last_page_len, seq_lens = _make_paged_kv(
        batch_size, kv_len, page_size, NUM_KV_HEADS, HEAD_DIM, dtype, DEVICE
    )
    k_cache, v_cache = kv.unbind(dim=1)
    wrapper = BatchDecodePagedCuteDSLWrapper(
        torch.empty(1, dtype=torch.uint8, device=DEVICE),
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        seq_lens,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=page_size,
        q_data_type=dtype,
        reduction="none",
        kv_splits=1,
    )
    # No exception expected; just verify output runs.
    wrapper.run(q, k_cache, v_cache)
