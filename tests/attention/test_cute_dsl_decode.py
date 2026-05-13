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
    return out.to(q.dtype)


def _decode_reference_paged_speculative(
    q: torch.Tensor,  # [batch_size, q_len_per_req, num_qo_heads, head_dim]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    sm_scale: float,
):
    """Speculative-decode GQA reference: q_len_per_req predicted tokens per
    request with bottom-right causal masking. Predicted token i (0-indexed
    within the prediction window) sits at sequence position
    ``seq - q_len_per_req + i`` and can attend to keys ``0 .. seq - q_len_per_req + i``.
    """
    batch_size, q_len_per_req, num_qo_heads, head_dim = q.shape
    page_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
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
        seq = (num_pages - 1) * page_size + last_len
        keys = k_f32[pages_b].reshape(num_pages * page_size, num_kv_heads, head_dim)[:seq]
        values = v_f32[pages_b].reshape(num_pages * page_size, num_kv_heads, head_dim)[:seq]
        keys = keys.repeat_interleave(group, dim=1)
        values = values.repeat_interleave(group, dim=1)
        # logits: [q_len_per_req, num_qo_heads, seq]
        logits = torch.einsum("qhd,nhd->qhn", q_f32[b], keys) * sm_scale
        # bottom-right causal: row i sees cols [0, seq - q_len_per_req + i]
        mask = torch.zeros(q_len_per_req, seq, dtype=torch.bool, device=q.device)
        for i in range(q_len_per_req):
            allowed = max(0, min(seq, seq - q_len_per_req + i + 1))
            if allowed > 0:
                mask[i, :allowed] = True
        logits.masked_fill_(~mask.unsqueeze(1), float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        out[b] = torch.einsum("qhn,nhd->qhd", probs, values)
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
    q = torch.randn(
        batch_size, 1, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype
    )
    k = torch.randn(
        batch_size, kv_len, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype
    )
    v = torch.randn_like(k)

    wrapper = BatchDecodeCuteDSLWrapper(
        torch.empty(1, dtype=torch.uint8, device=DEVICE),
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
    out_buf = torch.empty_like(out)
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
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * pages_per_seq
    )
    kv_indices = torch.arange(0, total_pages, device=device, dtype=torch.int32)
    last = (kv_len - 1) % page_size + 1
    kv_last_page_len = torch.full(
        (batch_size,), last, device=device, dtype=torch.int32
    )
    return kv, kv_indptr, kv_indices, kv_last_page_len


@pytest.mark.parametrize("batch_size", [1, 4, 17])
@pytest.mark.parametrize("kv_len", [129, 1024])
@pytest.mark.parametrize("page_size", [16, 32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cute_dsl_decode_paged_wrapper(batch_size, kv_len, page_size, dtype):
    torch.manual_seed(0)
    q = torch.randn(
        batch_size, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype
    )
    kv, kv_indptr, kv_indices, kv_last_page_len = _make_paged_kv(
        batch_size, kv_len, page_size, NUM_KV_HEADS, HEAD_DIM, dtype, DEVICE
    )
    k_cache, v_cache = kv.unbind(dim=1)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    wrapper = BatchDecodePagedCuteDSLWrapper(
        torch.empty(1, dtype=torch.uint8, device=DEVICE),
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
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
    q = torch.randn(
        batch_size, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype
    )
    kv, kv_indptr, kv_indices, kv_last_page_len = _make_paged_kv(
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
    out_buf = torch.empty_like(out)
    cd_wrapper.run(q, kv, out=out_buf)
    torch.testing.assert_close(out_buf, ref, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("kv_len", [1024])
@pytest.mark.parametrize("page_size", [16])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_batch_decode_wrapper_cute_dsl_hnd(batch_size, kv_len, page_size, dtype):
    """HND layout is presented as a transposed view; output must match NHD."""
    torch.manual_seed(0)
    q = torch.randn(
        batch_size, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype
    )
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
        kv_indptr, kv_indices, last_page_len,
        NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, page_size,
        q_data_type=dtype, kv_data_type=dtype,
    )
    out_hnd = cd.run(q, kv_hnd)

    # NHD reference via fa2.
    ref_wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=DEVICE),
        kv_layout="NHD", backend="fa2",
    )
    ref_wrapper.plan(
        kv_indptr, kv_indices, last_page_len,
        NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, page_size,
        q_data_type=dtype, kv_data_type=dtype,
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
    kv, kv_indptr, kv_indices, kv_last_page_len = _make_paged_kv(
        batch_size, kv_len, page_size, NUM_KV_HEADS, HEAD_DIM, dtype, DEVICE
    )
    k_cache, v_cache = kv.unbind(dim=1)
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
        q_len_per_req=q_len_per_req,
    )
    out = cd.run(q, kv)

    q_4d = q.view(batch_size, q_len_per_req, NUM_QO_HEADS, HEAD_DIM)
    ref = _decode_reference_paged_speculative(
        q_4d, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale
    )
    torch.testing.assert_close(
        out.view(batch_size, q_len_per_req, NUM_QO_HEADS, HEAD_DIM),
        ref,
        rtol=5e-3,
        atol=5e-3,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_batch_decode_wrapper_cute_dsl_v_scale(dtype):
    """v_scale must be folded into the cute-dsl kernel's o_scale: the output
    of run(v_scale=k) should equal k * run() (within fp tolerance)."""
    batch_size, page_size, kv_len = 4, 16, 1024
    torch.manual_seed(0)
    q = torch.randn(
        batch_size, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype
    )
    kv, kv_indptr, kv_indices, kv_last_page_len = _make_paged_kv(
        batch_size, kv_len, page_size, NUM_KV_HEADS, HEAD_DIM, dtype, DEVICE
    )
    workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    cd = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", backend="cute-dsl"
    )
    cd.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, page_size,
        q_data_type=dtype, kv_data_type=dtype,
    )
    out_unit = cd.run(q, kv)
    k = 2.5
    out_scaled = cd.run(q, kv, v_scale=k)
    torch.testing.assert_close(out_scaled, k * out_unit, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_cute_dsl_decode_paged_wrapper_o_scale(dtype):
    """Standalone BatchDecodePagedCuteDSLWrapper.run(o_scale=k) is equivalent
    to multiplying the unscaled output by k."""
    batch_size, page_size, kv_len = 4, 16, 1024
    torch.manual_seed(0)
    q = torch.randn(
        batch_size, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype
    )
    kv, kv_indptr, kv_indices, kv_last_page_len = _make_paged_kv(
        batch_size, kv_len, page_size, NUM_KV_HEADS, HEAD_DIM, dtype, DEVICE
    )
    k_cache, v_cache = kv.unbind(dim=1)
    wrapper = BatchDecodePagedCuteDSLWrapper(
        torch.empty(1, dtype=torch.uint8, device=DEVICE),
    )
    wrapper.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        num_qo_heads=NUM_QO_HEADS, num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM, page_size=page_size,
        q_data_type=dtype,
    )
    out_unit = wrapper.run(q, k_cache, v_cache)
    k = 0.375
    out_scaled = wrapper.run(q, k_cache, v_cache, o_scale=k)
    torch.testing.assert_close(out_scaled, k * out_unit, rtol=5e-3, atol=5e-3)


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
    q = torch.randn(
        batch_size, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype
    )
    pages_per_seq = [(s + page_size - 1) // page_size for s in seqlens]
    kv_indptr = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(pages_per_seq), 0).tolist()),
        device=DEVICE, dtype=torch.int32,
    )
    total_pages = int(kv_indptr[-1].item())
    kv = torch.randn(
        total_pages, 2, page_size, NUM_KV_HEADS, HEAD_DIM,
        device=DEVICE, dtype=dtype,
    )
    kv_indices = torch.arange(total_pages, device=DEVICE, dtype=torch.int32)
    kv_last_page_len = torch.tensor(
        [((s - 1) % page_size + 1) for s in seqlens],
        device=DEVICE, dtype=torch.int32,
    )

    workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    cd = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", backend="cute-dsl"
    )
    cd.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, page_size,
        q_data_type=dtype, kv_data_type=dtype,
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
    q = torch.randn(
        batch_size, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype
    )
    kv, kv_indptr, kv_indices, kv_last_page_len = _make_paged_kv(
        batch_size, kv_len, page_size, NUM_KV_HEADS, HEAD_DIM, dtype, DEVICE
    )

    workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    cd = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", backend="cute-dsl"
    )
    cd.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, page_size,
        q_data_type=dtype, kv_data_type=dtype,
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
    q = torch.randn(
        batch_size, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype
    )
    kv, kv_indptr, kv_indices, kv_last_page_len = _make_paged_kv(
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
            kv_last_page_len,
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
            kv_last_page_len,
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
            kv_last_page_len,
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
        kv_last_page_len,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    with pytest.raises(NotImplementedError, match="return_lse"):
        wrapper.run(q, kv, return_lse=True)
