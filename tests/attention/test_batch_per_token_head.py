"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Batch prefill/decode FP8 per-token-head KV cache tests.
"""

import pytest
import torch
import flashinfer
from tests.utils_fp8 import get_cos_sim_threshold, to_float8_per_token_head


# ============================================================
# Helpers
# ============================================================


def _cc():
    return torch.cuda.get_device_capability(0)


def _skip_if_sm_below_75():
    if _cc()[0] < 7 or (_cc()[0] == 7 and _cc()[1] < 5):
        pytest.skip("Requires SM75+")


def _skip_if_not_fp16_sm75(dtype: torch.dtype):
    if dtype != torch.float16 and _cc()[0] <= 7:
        pytest.skip(f"{dtype} skipped on SM75")


def check_accuracy(
    o_ref: torch.Tensor,
    o: torch.Tensor,
    kv_dtype: torch.dtype,
    mode,
    label="",
):
    cos_sim = torch.nn.functional.cosine_similarity(
        o_ref.reshape(-1).float(), o.reshape(-1).float(), dim=0
    ).item()
    max_diff = (o_ref - o).abs().max().item()
    prefix = f"[{label}] " if label else ""
    threshold = get_cos_sim_threshold(kv_dtype, mode)
    print(f"{prefix}{kv_dtype} cos_sim={cos_sim:.8f} max_diff={max_diff:.8f}")
    assert cos_sim >= threshold, (
        f"{prefix}cos_sim={cos_sim:.8f} < {threshold} ({kv_dtype})"
    )
    return cos_sim, max_diff


def _alloc_paged_cache(shape, head_dim, dtype, device, layout):
    if layout == "NHD":
        max_pages, page_size, num_kv_heads = shape
        full_shape = (max_pages, page_size, num_kv_heads, head_dim)
        s = (
            page_size * num_kv_heads * (head_dim + 16),
            num_kv_heads * (head_dim + 16),
            head_dim + 16,
            1,
        )
    else:
        max_pages, num_kv_heads, page_size = shape
        full_shape = (max_pages, num_kv_heads, page_size, head_dim)
        s = (
            num_kv_heads * page_size * (head_dim + 16),
            page_size * (head_dim + 16),
            head_dim + 16,
            1,
        )
    total_tokens = max_pages * page_size * num_kv_heads
    buf_size = total_tokens * (head_dim + 16)
    k_buf = torch.empty(buf_size, dtype=torch.uint8, device=device)
    v_buf = torch.empty(buf_size, dtype=torch.uint8, device=device)
    k_cache = torch.as_strided(k_buf, full_shape, s).view(dtype)
    v_cache = torch.as_strided(v_buf, full_shape, s).view(dtype)
    return k_cache, v_cache, k_buf, v_buf


def _write_scales_paged(cache_tensor, buf, scales, head_dim, layout):
    stride = head_dim + 16
    scale_stride_f32 = stride // 4
    scale_offset_f32 = head_dim // 4
    cache_shape = cache_tensor.shape
    if layout == "NHD":
        max_pages, page_size, num_kv_heads, _ = cache_shape
        s = (
            page_size * num_kv_heads * scale_stride_f32,
            num_kv_heads * scale_stride_f32,
            scale_stride_f32,
        )
        scale_view = torch.as_strided(
            buf.view(torch.float32),
            (max_pages, page_size, num_kv_heads),
            s,
            storage_offset=scale_offset_f32,
        )
    else:
        max_pages, num_kv_heads, page_size, _ = cache_shape
        s = (
            num_kv_heads * page_size * scale_stride_f32,
            page_size * scale_stride_f32,
            scale_stride_f32,
        )
        scale_view = torch.as_strided(
            buf.view(torch.float32),
            (max_pages, num_kv_heads, page_size),
            s,
            storage_offset=scale_offset_f32,
        )
    scale_view.copy_(scales.to(torch.float32).reshape(scale_view.shape))


def _build_paged_indices(kv_lens, page_size, device):
    num_pages_per_seq = [(kv + page_size - 1) // page_size for kv in kv_lens]
    total_num_pages = sum(num_pages_per_seq)
    kv_indptr = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.tensor(num_pages_per_seq, dtype=torch.int32, device=device)
            .cumsum(0)
            .to(torch.int32),
        ]
    )
    kv_indices = torch.arange(0, total_num_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor(
        [(kv - 1) % page_size + 1 for kv in kv_lens],
        dtype=torch.int32,
        device=device,
    )
    return kv_indptr, kv_indices, kv_last_page_len, num_pages_per_seq, total_num_pages


def _pad_to_pages(data_list, num_pages_per_seq, page_size):
    padded = []
    for data, n_pages in zip(data_list, num_pages_per_seq, strict=True):
        pad_len = n_pages * page_size - len(data)
        if pad_len > 0:
            data = torch.nn.functional.pad(data, (0, 0, 0, 0, 0, pad_len))
        padded.append(data)
    return padded


def _concat_to_pages(data_list, num_pages_per_seq, page_size, num_kv_heads):
    pages = [
        d.reshape(n, page_size, num_kv_heads, -1)
        for d, n in zip(data_list, num_pages_per_seq, strict=True)
    ]
    return torch.cat(pages, dim=0)


# ============================================================
# Batch Prefill
# ============================================================


def run_batch_prefill_pth(
    qo_lens,
    kv_lens,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    layout,
    dtype,
    kv_dtype,
    backend,
):
    device = "cuda:0"
    batch_size = len(kv_lens)
    total_qo = sum(qo_lens)
    q = torch.randn(total_qo, num_qo_heads, head_dim, dtype=dtype, device=device)

    k_f16_list = [
        0.1
        * torch.randn(kv_lens[i], num_kv_heads, head_dim, dtype=dtype, device=device)
        for i in range(batch_size)
    ]
    v_f16_list = [
        0.1
        * torch.randn(kv_lens[i], num_kv_heads, head_dim, dtype=dtype, device=device)
        for i in range(batch_size)
    ]

    kv_indptr, kv_indices, kv_last_page_len, num_pages_per_seq, total_num_pages = (
        _build_paged_indices(kv_lens, page_size, device)
    )
    qo_indptr = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.tensor(qo_lens, dtype=torch.int32, device=device)
            .cumsum(0)
            .to(torch.int32),
        ]
    )
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)

    # FP16 baseline
    k_f16_paged = _pad_to_pages(k_f16_list, num_pages_per_seq, page_size)
    v_f16_paged = _pad_to_pages(v_f16_list, num_pages_per_seq, page_size)
    k_p_flat = _concat_to_pages(k_f16_paged, num_pages_per_seq, page_size, num_kv_heads)
    v_p_flat = _concat_to_pages(v_f16_paged, num_pages_per_seq, page_size, num_kv_heads)
    if layout == "NHD":
        k_p, v_p = k_p_flat, v_p_flat
    else:
        k_p, v_p = k_p_flat.transpose(1, 2), v_p_flat.transpose(1, 2)
    k_p = k_p.to(dtype)
    v_p = v_p.to(dtype)

    wrapper_f16 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, layout, backend=backend
    )
    wrapper_f16.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o_fp16 = wrapper_f16.run(q, (k_p, v_p))

    # FP8 per-token-head
    k_fp8_paged = []
    v_fp8_paged = []
    k_scales_paged = []
    v_scales_paged = []
    for i in range(batch_size):
        k_fp8, k_s = to_float8_per_token_head(k_f16_paged[i], kv_dtype)
        v_fp8, v_s = to_float8_per_token_head(v_f16_paged[i], kv_dtype)
        k_fp8_paged.append(k_fp8)
        v_fp8_paged.append(v_fp8)
        k_scales_paged.append(k_s)
        v_scales_paged.append(v_s)

    k_fp8_flat = _concat_to_pages(
        k_fp8_paged, num_pages_per_seq, page_size, num_kv_heads
    )
    v_fp8_flat = _concat_to_pages(
        v_fp8_paged, num_pages_per_seq, page_size, num_kv_heads
    )
    k_s_flat = torch.cat(
        [
            s.reshape(n, page_size, num_kv_heads)
            for s, n in zip(k_scales_paged, num_pages_per_seq, strict=True)
        ]
    )
    v_s_flat = torch.cat(
        [
            s.reshape(n, page_size, num_kv_heads)
            for s, n in zip(v_scales_paged, num_pages_per_seq, strict=True)
        ]
    )

    if layout == "NHD":
        cache_shape = (total_num_pages, page_size, num_kv_heads)
        k_fp8_p, v_fp8_p, k_s_p, v_s_p = k_fp8_flat, v_fp8_flat, k_s_flat, v_s_flat
    else:
        cache_shape = (total_num_pages, num_kv_heads, page_size)
        k_fp8_p = k_fp8_flat.transpose(1, 2)
        v_fp8_p = v_fp8_flat.transpose(1, 2)
        k_s_p = k_s_flat.transpose(1, 2)
        v_s_p = v_s_flat.transpose(1, 2)

    k_cache, v_cache, k_buf, v_buf = _alloc_paged_cache(
        cache_shape, head_dim, kv_dtype, device, layout
    )
    k_cache.copy_(k_fp8_p)
    v_cache.copy_(v_fp8_p)
    _write_scales_paged(k_cache, k_buf, k_s_p, head_dim, layout)
    _write_scales_paged(v_cache, v_buf, v_s_p, head_dim, layout)

    wrapper_fp8 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        layout,
        backend=backend,
        use_per_token_head=True,
    )
    wrapper_fp8.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=kv_dtype,
        o_data_type=dtype,
    )
    o_fp8 = wrapper_fp8.run(q, (k_cache, v_cache))

    return check_accuracy(
        o_fp16, o_fp8, kv_dtype, "prefill", label=f"batch prefill {backend}"
    )


@pytest.mark.parametrize("is_gqa", [False, True], ids=["mha", "gqa"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn], ids=["e4m3"])
@pytest.mark.parametrize("layout", ["NHD", "HND"], ids=["NHD", "HND"])
@pytest.mark.parametrize("head_dim", [64, 128, 256], ids=["hd64", "hd128", "hd256"])
@pytest.mark.parametrize("backend", ["fa2"], ids=["fa2"])
def test_batch_prefill_pth_paged(is_gqa, dtype, kv_dtype, layout, head_dim, backend):
    _skip_if_sm_below_75()
    _skip_if_not_fp16_sm75(dtype)
    if _cc()[0] <= 7 and head_dim > 256:
        pytest.skip("head_dim>256 exceeds SM75 smem limit")
    num_qo_heads, num_kv_heads = (4, 2) if is_gqa else (4, 4)
    batch = 3
    run_batch_prefill_pth(
        [8] * batch,
        [32] * batch,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        16,
        layout,
        dtype,
        kv_dtype,
        backend,
    )


@pytest.mark.parametrize("is_gqa", [False, True], ids=["mha", "gqa"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn], ids=["e4m3"])
@pytest.mark.parametrize("layout", ["NHD", "HND"], ids=["NHD", "HND"])
@pytest.mark.parametrize("head_dim", [64, 128, 256], ids=["hd64", "hd128", "hd256"])
@pytest.mark.parametrize("backend", ["fa2"], ids=["fa2"])
def test_batch_prefill_pth_ragged(is_gqa, dtype, kv_dtype, layout, head_dim, backend):
    _skip_if_sm_below_75()
    _skip_if_not_fp16_sm75(dtype)
    num_qo_heads, num_kv_heads = (4, 2) if is_gqa else (4, 4)
    run_batch_prefill_pth(
        [6, 10, 8],
        [16, 48, 32],
        num_qo_heads,
        num_kv_heads,
        head_dim,
        16,
        layout,
        dtype,
        kv_dtype,
        backend,
    )


# ============================================================
# Batch Decode
# ============================================================


def run_batch_decode_pth(
    kv_lens,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    layout,
    dtype,
    kv_dtype,
    backend,
    pos_encoding_mode="NONE",
):
    device = "cuda:0"
    batch_size = len(kv_lens)
    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device)

    k_f16_list = [
        0.1
        * torch.randn(kv_lens[i], num_kv_heads, head_dim, dtype=dtype, device=device)
        for i in range(batch_size)
    ]
    v_f16_list = [
        0.1
        * torch.randn(kv_lens[i], num_kv_heads, head_dim, dtype=dtype, device=device)
        for i in range(batch_size)
    ]

    kv_indptr, kv_indices, kv_last_page_len, num_pages_per_seq, total_num_pages = (
        _build_paged_indices(kv_lens, page_size, device)
    )
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)

    # FP16 baseline
    k_f16_paged = _pad_to_pages(k_f16_list, num_pages_per_seq, page_size)
    v_f16_paged = _pad_to_pages(v_f16_list, num_pages_per_seq, page_size)
    k_p_flat = _concat_to_pages(k_f16_paged, num_pages_per_seq, page_size, num_kv_heads)
    v_p_flat = _concat_to_pages(v_f16_paged, num_pages_per_seq, page_size, num_kv_heads)
    if layout == "NHD":
        k_p, v_p = k_p_flat, v_p_flat
    else:
        k_p, v_p = k_p_flat.transpose(1, 2), v_p_flat.transpose(1, 2)
    k_p = k_p.to(dtype)
    v_p = v_p.to(dtype)

    wrapper_f16 = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, layout, backend=backend
    )
    wrapper_f16.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode=pos_encoding_mode,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o_fp16 = wrapper_f16.run(q, (k_p, v_p))

    # FP8 per-token-head
    k_fp8_paged = []
    k_scales_paged = []
    v_fp8_paged = []
    v_scales_paged = []
    for i in range(batch_size):
        k_fp8_i, k_s_i = to_float8_per_token_head(k_f16_list[i], kv_dtype)
        v_fp8_i, v_s_i = to_float8_per_token_head(v_f16_list[i], kv_dtype)
        n_pages = num_pages_per_seq[i]
        pad_len = n_pages * page_size - kv_lens[i]
        if pad_len > 0:
            k_fp8_i = torch.nn.functional.pad(k_fp8_i, (0, 0, 0, 0, 0, pad_len))
            k_s_i = torch.nn.functional.pad(k_s_i, (0, 0, 0, pad_len))
            v_fp8_i = torch.nn.functional.pad(v_fp8_i, (0, 0, 0, 0, 0, pad_len))
            v_s_i = torch.nn.functional.pad(v_s_i, (0, 0, 0, pad_len))
        k_fp8_paged.append(k_fp8_i)
        k_scales_paged.append(k_s_i)
        v_fp8_paged.append(v_fp8_i)
        v_scales_paged.append(v_s_i)

    k_fp8_flat = _concat_to_pages(
        k_fp8_paged, num_pages_per_seq, page_size, num_kv_heads
    )
    v_fp8_flat = _concat_to_pages(
        v_fp8_paged, num_pages_per_seq, page_size, num_kv_heads
    )
    k_s_flat = torch.cat(
        [
            s.reshape(n, page_size, num_kv_heads)
            for s, n in zip(k_scales_paged, num_pages_per_seq, strict=True)
        ]
    )
    v_s_flat = torch.cat(
        [
            s.reshape(n, page_size, num_kv_heads)
            for s, n in zip(v_scales_paged, num_pages_per_seq, strict=True)
        ]
    )

    if layout == "NHD":
        cache_shape = (total_num_pages, page_size, num_kv_heads)
        k_fp8_p, v_fp8_p, k_s_p, v_s_p = k_fp8_flat, v_fp8_flat, k_s_flat, v_s_flat
    else:
        cache_shape = (total_num_pages, num_kv_heads, page_size)
        k_fp8_p = k_fp8_flat.transpose(1, 2)
        v_fp8_p = v_fp8_flat.transpose(1, 2)
        k_s_p = k_s_flat.transpose(1, 2)
        v_s_p = v_s_flat.transpose(1, 2)

    k_cache, v_cache, k_buf, v_buf = _alloc_paged_cache(
        cache_shape, head_dim, kv_dtype, device, layout
    )
    k_cache.copy_(k_fp8_p)
    v_cache.copy_(v_fp8_p)
    _write_scales_paged(k_cache, k_buf, k_s_p, head_dim, layout)
    _write_scales_paged(v_cache, v_buf, v_s_p, head_dim, layout)

    wrapper_fp8 = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        layout,
        backend=backend,
        use_per_token_head=True,
    )
    wrapper_fp8.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode=pos_encoding_mode,
        q_data_type=dtype,
        kv_data_type=kv_dtype,
        o_data_type=dtype,
    )
    o_fp8 = wrapper_fp8.run(q, (k_cache, v_cache))

    return check_accuracy(
        o_fp16, o_fp8, kv_dtype, "decode", label=f"batch decode {backend}"
    )


@pytest.mark.parametrize("is_gqa", [False, True], ids=["mha", "gqa"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn], ids=["e4m3"])
@pytest.mark.parametrize("layout", ["NHD", "HND"], ids=["NHD", "HND"])
@pytest.mark.parametrize("head_dim", [64, 128, 256], ids=["hd64", "hd128", "hd256"])
@pytest.mark.parametrize("backend", ["fa2"], ids=["fa2"])
def test_batch_decode_pth_paged(is_gqa, dtype, kv_dtype, layout, head_dim, backend):
    _skip_if_sm_below_75()
    _skip_if_not_fp16_sm75(dtype)
    if _cc()[0] <= 7 and head_dim > 256:
        pytest.skip("head_dim>256 exceeds SM75 smem limit")
    num_qo_heads, num_kv_heads = (4, 2) if is_gqa else (4, 4)
    run_batch_decode_pth(
        [32] * 4,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        16,
        layout,
        dtype,
        kv_dtype,
        backend,
    )


@pytest.mark.parametrize("is_gqa", [False, True], ids=["mha", "gqa"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn], ids=["e4m3"])
@pytest.mark.parametrize("layout", ["NHD", "HND"], ids=["NHD", "HND"])
@pytest.mark.parametrize("head_dim", [64, 128, 256], ids=["hd64", "hd128", "hd256"])
@pytest.mark.parametrize("backend", ["fa2"], ids=["fa2"])
def test_batch_decode_pth_ragged(is_gqa, dtype, kv_dtype, layout, head_dim, backend):
    _skip_if_sm_below_75()
    _skip_if_not_fp16_sm75(dtype)
    num_qo_heads, num_kv_heads = (4, 2) if is_gqa else (4, 4)
    run_batch_decode_pth(
        [16, 48, 32, 64],
        num_qo_heads,
        num_kv_heads,
        head_dim,
        16,
        layout,
        dtype,
        kv_dtype,
        backend,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn], ids=["e4m3"])
@pytest.mark.parametrize("backend", ["fa2"], ids=["fa2"])
def test_batch_decode_pth_rope_llama(dtype, kv_dtype, backend):
    _skip_if_sm_below_75()
    _skip_if_not_fp16_sm75(dtype)
    run_batch_decode_pth(
        [32] * 4,
        4,
        2,
        128,
        16,
        "NHD",
        dtype,
        kv_dtype,
        backend,
        pos_encoding_mode="ROPE_LLAMA",
    )


# ============================================================
# BatchPrefillWithRaggedKVCacheWrapper + PTH (continuous mode)
# ============================================================


def run_batch_prefill_ragged_continuous_pth(
    qo_lens,
    kv_lens,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    layout,
    dtype,
    kv_dtype,
    causal,
):
    """Batch prefill with ragged (continuous) KV cache + FP8 per-token-head."""
    device = "cuda:0"
    batch_size = len(kv_lens)
    total_qo = sum(qo_lens)
    q = torch.randn(total_qo, num_qo_heads, head_dim, dtype=dtype, device=device)

    k_f16_list = [
        0.3
        * torch.randn(kv_lens[i], num_kv_heads, head_dim, dtype=dtype, device=device)
        for i in range(batch_size)
    ]
    v_f16_list = [
        0.3
        * torch.randn(kv_lens[i], num_kv_heads, head_dim, dtype=dtype, device=device)
        for i in range(batch_size)
    ]

    k_f16 = torch.cat(k_f16_list, dim=0)
    v_f16 = torch.cat(v_f16_list, dim=0)

    qo_indptr = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.tensor(qo_lens, dtype=torch.int32, device=device)
            .cumsum(0)
            .to(torch.int32),
        ]
    )
    kv_indptr = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.tensor(kv_lens, dtype=torch.int32, device=device)
            .cumsum(0)
            .to(torch.int32),
        ]
    )
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)

    wrapper_f16 = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, layout, backend="fa2"
    )
    wrapper_f16.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o_fp16 = wrapper_f16.run(q, k_f16, v_f16)

    # FP8 per-token-head: build strided continuous cache
    k_fp8_list = []
    v_fp8_list = []
    k_scales_list = []
    v_scales_list = []
    for i in range(batch_size):
        k_fp8, k_s = to_float8_per_token_head(k_f16_list[i], kv_dtype)
        v_fp8, v_s = to_float8_per_token_head(v_f16_list[i], kv_dtype)
        k_fp8_list.append(k_fp8)
        v_fp8_list.append(v_fp8)
        k_scales_list.append(k_s)
        v_scales_list.append(v_s)

    k_fp8 = torch.cat(k_fp8_list, dim=0)
    v_fp8 = torch.cat(v_fp8_list, dim=0)
    k_scales = torch.cat(k_scales_list, dim=0)
    v_scales = torch.cat(v_scales_list, dim=0)

    k_cache = _build_strided_cache(k_fp8, k_scales, head_dim)
    v_cache = _build_strided_cache(v_fp8, v_scales, head_dim)

    wrapper_fp8 = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, layout, backend="fa2", use_per_token_head=True
    )
    wrapper_fp8.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=kv_dtype,
        o_data_type=dtype,
    )
    o_fp8 = wrapper_fp8.run(q, k_cache, v_cache)

    assert not torch.isnan(o_fp8).any(), "PTH output contains NaN"
    return check_accuracy(
        o_fp16, o_fp8, kv_dtype, "prefill", label=f"ragged-continuous prefill {layout}"
    )


def _build_strided_cache(x, scales, head_dim):
    """Build a continuous KV cache tensor with head_dim+16 stride for PTH."""
    kv_len, num_kv_heads = x.shape[0], x.shape[1]
    stride = head_dim + 16
    buf_size = kv_len * num_kv_heads * stride
    buf = torch.zeros(buf_size, dtype=torch.uint8, device=x.device)
    rows = buf.reshape(-1, stride)
    fp8_flat = x.reshape(-1, head_dim).view(torch.uint8)
    rows[:, :head_dim].copy_(fp8_flat)
    scales_f32 = scales.reshape(-1).to(torch.float32)
    scales_bytes = scales_f32.view(torch.uint8)
    rows[:, head_dim : head_dim + 4].copy_(scales_bytes.reshape(-1, 4))
    cache = torch.as_strided(
        buf.view(x.dtype),
        (kv_len, num_kv_heads, head_dim),
        (num_kv_heads * stride, stride, 1),
        storage_offset=0,
    )
    return cache


@pytest.mark.parametrize("is_gqa", [False, True], ids=["mha", "gqa"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn], ids=["e4m3"])
@pytest.mark.parametrize("layout", ["NHD"], ids=["NHD"])
@pytest.mark.parametrize("head_dim", [64, 128], ids=["hd64", "hd128"])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "no-causal"])
def test_batch_prefill_ragged_continuous_pth(
    is_gqa, dtype, kv_dtype, layout, head_dim, causal
):
    """BatchPrefillWithRaggedKVCacheWrapper with FP8 per-token-head."""
    _skip_if_sm_below_75()
    _skip_if_not_fp16_sm75(dtype)
    num_qo_heads, num_kv_heads = (4, 2) if is_gqa else (4, 4)
    run_batch_prefill_ragged_continuous_pth(
        qo_lens=[6, 10, 8],
        kv_lens=[16, 48, 32],
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        layout=layout,
        dtype=dtype,
        kv_dtype=kv_dtype,
        causal=causal,
    )


# ============================================================
# BatchDecode + PTH + non-page-aligned ragged lengths
# ============================================================


def run_batch_decode_non_page_aligned_pth(
    kv_lens,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    layout,
    dtype,
    kv_dtype,
):
    """Batch decode with non-page-aligned KV lengths + PTH."""
    device = "cuda:0"
    batch_size = len(kv_lens)
    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device)

    k_f16_list = [
        0.3
        * torch.randn(kv_lens[i], num_kv_heads, head_dim, dtype=dtype, device=device)
        for i in range(batch_size)
    ]
    v_f16_list = [
        0.3
        * torch.randn(kv_lens[i], num_kv_heads, head_dim, dtype=dtype, device=device)
        for i in range(batch_size)
    ]

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)

    # Single page per sequence, page_size = max(kv_lens)
    page_size = max(kv_lens)
    kv_indptr = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.arange(1, batch_size + 1, dtype=torch.int32, device=device),
        ]
    )
    kv_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    # FP16 baseline (padded)
    k_f16_padded = []
    v_f16_padded = []
    for i in range(batch_size):
        pad_len = page_size - kv_lens[i]
        if pad_len > 0:
            k_f16_padded.append(
                torch.nn.functional.pad(k_f16_list[i], (0, 0, 0, 0, 0, pad_len))
            )
            v_f16_padded.append(
                torch.nn.functional.pad(v_f16_list[i], (0, 0, 0, 0, 0, pad_len))
            )
        else:
            k_f16_padded.append(k_f16_list[i])
            v_f16_padded.append(v_f16_list[i])

    k_f16_paged = torch.stack(k_f16_padded).reshape(
        batch_size, page_size, num_kv_heads, head_dim
    )
    v_f16_paged = torch.stack(v_f16_padded).reshape(
        batch_size, page_size, num_kv_heads, head_dim
    )
    if layout == "HND":
        k_f16_paged = k_f16_paged.transpose(1, 2)
        v_f16_paged = v_f16_paged.transpose(1, 2)

    wrapper_f16 = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, layout, backend="fa2"
    )
    wrapper_f16.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o_fp16 = wrapper_f16.run(q, (k_f16_paged, v_f16_paged))

    # FP8 per-token-head with strided cache
    k_fp8_padded = []
    v_fp8_padded = []
    k_scales_padded = []
    v_scales_padded = []
    for i in range(batch_size):
        k_fp8, k_s = to_float8_per_token_head(k_f16_list[i], kv_dtype)
        v_fp8, v_s = to_float8_per_token_head(v_f16_list[i], kv_dtype)
        pad_len = page_size - kv_lens[i]
        if pad_len > 0:
            k_fp8 = torch.nn.functional.pad(k_fp8, (0, 0, 0, 0, 0, pad_len))
            k_s = torch.nn.functional.pad(k_s, (0, 0, 0, pad_len))
            v_fp8 = torch.nn.functional.pad(v_fp8, (0, 0, 0, 0, 0, pad_len))
            v_s = torch.nn.functional.pad(v_s, (0, 0, 0, pad_len))
        k_fp8_padded.append(k_fp8)
        v_fp8_padded.append(v_fp8)
        k_scales_padded.append(k_s)
        v_scales_padded.append(v_s)

    k_fp8_flat = torch.stack(k_fp8_padded).reshape(
        batch_size, page_size, num_kv_heads, head_dim
    )
    v_fp8_flat = torch.stack(v_fp8_padded).reshape(
        batch_size, page_size, num_kv_heads, head_dim
    )
    k_s_flat = torch.stack(k_scales_padded).reshape(batch_size, page_size, num_kv_heads)
    v_s_flat = torch.stack(v_scales_padded).reshape(batch_size, page_size, num_kv_heads)

    if layout == "HND":
        k_fp8_flat = k_fp8_flat.transpose(1, 2)
        v_fp8_flat = v_fp8_flat.transpose(1, 2)
        k_s_flat = k_s_flat.transpose(1, 2)
        v_s_flat = v_s_flat.transpose(1, 2)

    k_cache, v_cache, k_buf, v_buf = _alloc_cache_decode_strided(
        batch_size, page_size, num_kv_heads, head_dim, kv_dtype, device, layout
    )
    k_cache.copy_(k_fp8_flat)
    v_cache.copy_(v_fp8_flat)
    _write_scales_decode_strided(
        k_cache, k_buf, k_s_flat.to(torch.float32), head_dim, layout
    )
    _write_scales_decode_strided(
        v_cache, v_buf, v_s_flat.to(torch.float32), head_dim, layout
    )

    wrapper_fp8 = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, layout, backend="fa2", use_per_token_head=True
    )
    wrapper_fp8.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=kv_dtype,
        o_data_type=dtype,
    )
    o_fp8 = wrapper_fp8.run(q, (k_cache, v_cache))

    assert not torch.isnan(o_fp8).any(), "PTH output contains NaN"
    return check_accuracy(
        o_fp16, o_fp8, kv_dtype, "decode", label=f"decode non-page-aligned {layout}"
    )


def _alloc_cache_decode_strided(
    batch_size, page_size, num_kv_heads, head_dim, dtype, device, layout
):
    """Allocate strided cache for batch decode with non-page-aligned lengths."""
    stride = head_dim + 16
    if layout == "NHD":
        buf_size = batch_size * page_size * num_kv_heads * stride
        s = (
            page_size * num_kv_heads * stride,
            num_kv_heads * stride,
            stride,
            1,
        )
        shape = (batch_size, page_size, num_kv_heads, head_dim)
    else:
        buf_size = batch_size * num_kv_heads * page_size * stride
        s = (
            num_kv_heads * page_size * stride,
            page_size * stride,
            stride,
            1,
        )
        shape = (batch_size, num_kv_heads, page_size, head_dim)
    k_buf = torch.empty(buf_size, dtype=torch.uint8, device=device)
    v_buf = torch.empty(buf_size, dtype=torch.uint8, device=device)
    k_cache = torch.as_strided(k_buf, shape, s).view(dtype)
    v_cache = torch.as_strided(v_buf, shape, s).view(dtype)
    return k_cache, v_cache, k_buf, v_buf


def _write_scales_decode_strided(cache_tensor, buf, scales, head_dim, layout):
    """Write float32 scales into the strided cache buffer for decode."""
    stride = head_dim + 16
    scale_stride_f32 = stride // 4
    scale_offset_f32 = head_dim // 4
    if layout == "NHD":
        batch_size, page_size, num_kv_heads, _ = cache_tensor.shape
        shape = (batch_size, page_size, num_kv_heads)
        strides = (
            page_size * num_kv_heads * scale_stride_f32,
            num_kv_heads * scale_stride_f32,
            scale_stride_f32,
        )
    else:
        batch_size, num_kv_heads, page_size, _ = cache_tensor.shape
        shape = (batch_size, num_kv_heads, page_size)
        strides = (
            num_kv_heads * page_size * scale_stride_f32,
            page_size * scale_stride_f32,
            scale_stride_f32,
        )
    scale_view = torch.as_strided(
        buf.view(torch.float32), shape, strides, storage_offset=scale_offset_f32
    )
    scale_view.copy_(scales.reshape(shape))


@pytest.mark.parametrize("is_gqa", [False, True], ids=["mha", "gqa"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn], ids=["e4m3"])
@pytest.mark.parametrize("layout", ["NHD", "HND"], ids=["NHD", "HND"])
@pytest.mark.parametrize("head_dim", [64, 128], ids=["hd64", "hd128"])
def test_batch_decode_non_page_aligned_pth(is_gqa, dtype, kv_dtype, layout, head_dim):
    """Batch decode with non-page-aligned KV lengths + PTH."""
    _skip_if_sm_below_75()
    _skip_if_not_fp16_sm75(dtype)
    num_qo_heads, num_kv_heads = (4, 2) if is_gqa else (4, 4)
    run_batch_decode_non_page_aligned_pth(
        kv_lens=[17, 43, 31, 59],
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        layout=layout,
        dtype=dtype,
        kv_dtype=kv_dtype,
    )


# ============================================================
# Backend assertion: PTH + non-fa2 should be rejected
# ============================================================


def test_pth_backend_assertion_prefill_ragged():
    """PTH + non-fa2 backend raises ValueError in ragged prefill plan."""
    _skip_if_sm_below_75()
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, backend="cudnn", use_per_token_head=True
    )
    qo_indptr = torch.tensor([0, 8], dtype=torch.int32, device="cuda:0")
    kv_indptr = torch.tensor([0, 16], dtype=torch.int32, device="cuda:0")
    with pytest.raises(ValueError, match="only supported with the fa2 backend"):
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            num_qo_heads=4,
            num_kv_heads=4,
            head_dim_qk=128,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
        )


def test_pth_backend_assertion_prefill_paged():
    """PTH + non-fa2 backend raises ValueError in paged prefill plan."""
    _skip_if_sm_below_75()
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, backend="cudnn", use_per_token_head=True
    )
    qo_indptr = torch.tensor([0, 8], dtype=torch.int32, device="cuda:0")
    kv_indptr = torch.tensor([0, 2], dtype=torch.int32, device="cuda:0")
    kv_indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda:0")
    kv_last_page_len = torch.tensor([16], dtype=torch.int32, device="cuda:0")
    with pytest.raises(ValueError, match="only supported with the fa2 backend"):
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            4,
            4,
            128,
            16,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
        )


def test_pth_backend_assertion_decode():
    """PTH + non-fa2 backend raises ValueError in batch decode."""
    _skip_if_sm_below_75()
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, backend="fa3", use_tensor_cores=True, use_per_token_head=True
    )
    indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda:0")
    indices = torch.tensor([0], dtype=torch.int32, device="cuda:0")
    last_page_len = torch.tensor([1], dtype=torch.int32, device="cuda:0")
    with pytest.raises(ValueError, match="only supported with the fa2 backend"):
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            num_qo_heads=4,
            num_kv_heads=4,
            head_dim=64,
            page_size=16,
        )


def test_pth_auto_backend_allowed():
    """PTH + auto backend is allowed and forced to fa2."""
    _skip_if_sm_below_75()
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, backend="auto", use_per_token_head=True
    )
    qo_indptr = torch.tensor([0, 16], dtype=torch.int32, device="cuda:0")
    kv_indptr = torch.tensor([0, 64], dtype=torch.int32, device="cuda:0")
    num_qo_heads, num_kv_heads, head_dim = 4, 4, 64
    q = torch.randn(16, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0")
    k = 0.3 * torch.randn(
        64, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    )
    v = 0.3 * torch.randn(
        64, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    )
    k_fp8, k_s = to_float8_per_token_head(k, torch.float8_e4m3fn)
    v_fp8, v_s = to_float8_per_token_head(v, torch.float8_e4m3fn)
    k_cache = _build_strided_cache(k_fp8, k_s, head_dim)
    v_cache = _build_strided_cache(v_fp8, v_s, head_dim)

    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=True,
        q_data_type=torch.float16,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.float16,
    )
    assert wrapper._backend == "fa2", (
        f"Expected backend 'fa2' but got '{wrapper._backend}'"
    )
    o = wrapper.run(q, k_cache, v_cache)
    assert not torch.isnan(o).any(), "Output contains NaN"


# ============================================================
# Smoke tests
# ============================================================

if __name__ == "__main__":
    dtypes = [torch.float16]
    if _cc()[0] > 7:
        dtypes.append(torch.bfloat16)
    for dtype in dtypes:
        kv_dtype = torch.float8_e4m3fn

        test_batch_prefill_pth_paged(
            is_gqa=False,
            dtype=dtype,
            kv_dtype=kv_dtype,
            layout="NHD",
            head_dim=128,
            backend="fa2",
        )
        print(f"batch prefill MHA {dtype}/{kv_dtype} paged smoke passed")

        test_batch_prefill_pth_paged(
            is_gqa=True,
            dtype=dtype,
            kv_dtype=kv_dtype,
            layout="NHD",
            head_dim=64,
            backend="fa2",
        )
        print(f"batch prefill GQA {dtype}/{kv_dtype} paged smoke passed")

        test_batch_prefill_pth_ragged(
            is_gqa=False,
            dtype=dtype,
            kv_dtype=kv_dtype,
            layout="NHD",
            head_dim=128,
            backend="fa2",
        )
        print(f"batch prefill {dtype}/{kv_dtype} ragged smoke passed")

        test_batch_decode_pth_paged(
            is_gqa=False,
            dtype=dtype,
            kv_dtype=kv_dtype,
            layout="NHD",
            head_dim=128,
            backend="fa2",
        )
        print(f"batch decode MHA {dtype}/{kv_dtype} paged smoke passed")

        test_batch_decode_pth_paged(
            is_gqa=True,
            dtype=dtype,
            kv_dtype=kv_dtype,
            layout="NHD",
            head_dim=64,
            backend="fa2",
        )
        print(f"batch decode GQA {dtype}/{kv_dtype} paged smoke passed")

        test_batch_decode_pth_ragged(
            is_gqa=False,
            dtype=dtype,
            kv_dtype=kv_dtype,
            layout="NHD",
            head_dim=64,
            backend="fa2",
        )
        print(f"batch decode {dtype}/{kv_dtype} ragged smoke passed")

        test_batch_prefill_ragged_continuous_pth(
            is_gqa=False,
            dtype=dtype,
            kv_dtype=kv_dtype,
            layout="NHD",
            head_dim=128,
            causal=True,
        )
        print(f"batch prefill {dtype}/{kv_dtype} ragged-continuous smoke passed")

        test_batch_decode_non_page_aligned_pth(
            is_gqa=False,
            dtype=dtype,
            kv_dtype=kv_dtype,
            layout="NHD",
            head_dim=128,
        )
        print(f"batch decode {dtype}/{kv_dtype} non-page-aligned smoke passed")

    test_pth_backend_assertion_prefill_ragged()
    print("pth backend assertion ragged prefill OK")

    test_pth_backend_assertion_prefill_paged()
    print("pth backend assertion paged prefill OK")

    test_pth_backend_assertion_decode()
    print("pth backend assertion decode OK")

    test_pth_auto_backend_allowed()
    print("pth auto backend forced to fa2 OK")

    print("\nAll batch per-token-head smoke tests passed")
