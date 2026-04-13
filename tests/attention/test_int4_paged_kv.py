"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch

import flashinfer
from flashinfer import prefill as flashinfer_prefill


def _allocate_int4_tensor(shape, device="cuda:0"):
    packed_dim = (shape[-1] + 1) // 2
    scale_dim = shape[-1] // 32
    return flashinfer.INT4Tensor(
        torch.zeros(*shape[:-1], packed_dim, dtype=torch.uint8, device=device),
        torch.zeros(*shape[:-1], scale_dim, dtype=torch.float16, device=device),
        original_shape=shape,
    )


def _make_paged_int4_cache(
    num_pages: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    kv_layout: str,
    combined: bool,
):
    if kv_layout == "NHD":
        kv_shape = (num_pages, 2, page_size, num_kv_heads, head_dim)
        k_shape = (num_pages, page_size, num_kv_heads, head_dim)
    else:
        kv_shape = (num_pages, 2, num_kv_heads, page_size, head_dim)
        k_shape = (num_pages, num_kv_heads, page_size, head_dim)
    if combined:
        return _allocate_int4_tensor(kv_shape)
    return _allocate_int4_tensor(k_shape), _allocate_int4_tensor(k_shape)


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("combined", [False, True])
def test_append_paged_kv_cache_int4_matches_quantized_layout(
    kv_layout, head_dim, combined
):
    nnz_kv = 12
    num_kv_heads = 2
    page_size = 4
    device = "cuda:0"

    k_append = torch.randn(
        nnz_kv, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_append = torch.randn(
        nnz_kv, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    kv_append_length = torch.tensor([3, 5, 4], dtype=torch.int32, device=device)
    kv_append_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(kv_append_length, dim=0),
        ]
    )
    num_pages_per_req = torch.tensor([1, 2, 1], dtype=torch.int32, device=device)
    kv_page_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(num_pages_per_req, dim=0),
        ]
    )
    kv_page_indices = torch.arange(4, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor([3, 1, 4], dtype=torch.int32, device=device)

    batch_indices, positions = flashinfer.get_batch_indices_positions(
        kv_append_indptr,
        flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, page_size),
        nnz_kv,
    )

    paged_kv_cache = _make_paged_int4_cache(
        8, page_size, num_kv_heads, head_dim, kv_layout, combined=combined
    )
    flashinfer.append_paged_kv_cache(
        k_append,
        v_append,
        batch_indices,
        positions,
        paged_kv_cache,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_len,
        kv_layout=kv_layout,
    )

    expected_k = flashinfer.int4_quantize(k_append)
    expected_v = flashinfer.int4_quantize(v_append)
    if combined:
        k_cache, v_cache = paged_kv_cache.unbind(dim=1)
    else:
        k_cache, v_cache = paged_kv_cache

    batch_indices_i64 = batch_indices.to(torch.int64)
    positions_i64 = positions.to(torch.int64)
    page_offsets = torch.div(positions_i64, page_size, rounding_mode="floor")
    page_positions = torch.remainder(positions_i64, page_size)
    page_indices = kv_page_indices.to(torch.int64)[
        kv_page_indptr.to(torch.int64)[batch_indices_i64] + page_offsets
    ]

    if kv_layout == "NHD":
        torch.testing.assert_close(
            k_cache.data[page_indices, page_positions], expected_k.data
        )
        torch.testing.assert_close(
            v_cache.data[page_indices, page_positions], expected_v.data
        )
        torch.testing.assert_close(
            k_cache.scale[page_indices, page_positions], expected_k.scale
        )
        torch.testing.assert_close(
            v_cache.scale[page_indices, page_positions], expected_v.scale
        )
        gathered_k = flashinfer.int4_dequantize(k_cache)[page_indices, page_positions]
        gathered_v = flashinfer.int4_dequantize(v_cache)[page_indices, page_positions]
    else:
        torch.testing.assert_close(
            k_cache.data[page_indices, :, page_positions, :], expected_k.data
        )
        torch.testing.assert_close(
            v_cache.data[page_indices, :, page_positions, :], expected_v.data
        )
        torch.testing.assert_close(
            k_cache.scale[page_indices, :, page_positions, :], expected_k.scale
        )
        torch.testing.assert_close(
            v_cache.scale[page_indices, :, page_positions, :], expected_v.scale
        )
        gathered_k = flashinfer.int4_dequantize(k_cache)[page_indices, :, page_positions]
        gathered_v = flashinfer.int4_dequantize(v_cache)[page_indices, :, page_positions]

    torch.testing.assert_close(
        gathered_k,
        flashinfer.int4_dequantize(expected_k),
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        gathered_v,
        flashinfer.int4_dequantize(expected_v),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("use_tensor_cores", [False, True])
def test_single_decode_with_kv_cache_int4(kv_layout, head_dim, use_tensor_cores):
    kv_len = 9
    num_kv_heads = 2
    num_qo_heads = 4
    device = "cuda:0"

    q = torch.randn(num_qo_heads, head_dim, dtype=torch.float16, device=device)
    if kv_layout == "NHD":
        k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device)
        v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device)
    else:
        k = torch.randn(num_kv_heads, kv_len, head_dim, dtype=torch.float16, device=device)
        v = torch.randn(num_kv_heads, kv_len, head_dim, dtype=torch.float16, device=device)

    k_int4 = flashinfer.int4_quantize(k)
    v_int4 = flashinfer.int4_quantize(v)
    out = flashinfer.single_decode_with_kv_cache(
        q,
        k_int4,
        v_int4,
        kv_layout=kv_layout,
        use_tensor_cores=use_tensor_cores,
    )
    out_ref = flashinfer.single_decode_with_kv_cache(
        q,
        flashinfer.int4_dequantize(k_int4),
        flashinfer.int4_dequantize(v_int4),
        kv_layout=kv_layout,
        use_tensor_cores=use_tensor_cores,
    )
    torch.testing.assert_close(out, out_ref, rtol=1e-3, atol=1e-3)


def test_single_decode_with_kv_cache_int4_rejects_scale():
    q = torch.randn(4, 128, dtype=torch.float16, device="cuda:0")
    k = flashinfer.int4_quantize(
        torch.randn(9, 2, 128, dtype=torch.float16, device="cuda:0")
    )
    v = flashinfer.int4_quantize(
        torch.randn(9, 2, 128, dtype=torch.float16, device="cuda:0")
    )

    with pytest.raises(ValueError, match="k_scale and v_scale are not supported"):
        flashinfer.single_decode_with_kv_cache(
            q,
            k,
            v,
            kv_layout="NHD",
            k_scale=0.5,
        )


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_single_prefill_with_kv_cache_int4(kv_layout, head_dim):
    qo_len = 3
    kv_len = 8
    num_kv_heads = 2
    num_qo_heads = 4
    device = "cuda:0"

    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=torch.float16, device=device)
    if kv_layout == "NHD":
        k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device)
        v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device)
    else:
        k = torch.randn(num_kv_heads, kv_len, head_dim, dtype=torch.float16, device=device)
        v = torch.randn(num_kv_heads, kv_len, head_dim, dtype=torch.float16, device=device)

    k_int4 = flashinfer.int4_quantize(k)
    v_int4 = flashinfer.int4_quantize(v)
    out = flashinfer.single_prefill_with_kv_cache(
        q,
        k_int4,
        v_int4,
        kv_layout=kv_layout,
        causal=True,
        backend="fa2",
    )
    out_ref = flashinfer.single_prefill_with_kv_cache(
        q,
        flashinfer.int4_dequantize(k_int4),
        flashinfer.int4_dequantize(v_int4),
        kv_layout=kv_layout,
        causal=True,
        backend="fa2",
    )
    torch.testing.assert_close(out, out_ref, rtol=1e-3, atol=1e-3)


def test_single_prefill_with_kv_cache_int4_rejects_scale():
    q = torch.randn(3, 4, 128, dtype=torch.float16, device="cuda:0")
    k = flashinfer.int4_quantize(
        torch.randn(8, 2, 128, dtype=torch.float16, device="cuda:0")
    )
    v = flashinfer.int4_quantize(
        torch.randn(8, 2, 128, dtype=torch.float16, device="cuda:0")
    )

    with pytest.raises(ValueError, match="scale_k and scale_v are not supported"):
        flashinfer.single_prefill_with_kv_cache(
            q,
            k,
            v,
            kv_layout="NHD",
            scale_k=0.5,
        )


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("use_tensor_cores", [False, True])
@pytest.mark.parametrize("page_size", [4, 8])
def test_batch_decode_with_paged_kv_cache_int4(
    kv_layout, head_dim, use_tensor_cores, page_size
):
    batch_size = 3
    kv_len = 9
    num_kv_heads = 2
    num_qo_heads = 4
    device = "cuda:0"

    q = torch.randn(
        batch_size, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if kv_layout == "NHD":
        kv_data = torch.randn(
            total_num_pages,
            2,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=torch.float16,
            device=device,
        )
    else:
        kv_data = torch.randn(
            total_num_pages,
            2,
            num_kv_heads,
            page_size,
            head_dim,
            dtype=torch.float16,
            device=device,
        )
    kv_data_int4 = flashinfer.int4_quantize(kv_data)
    kv_data_ref = flashinfer.int4_dequantize(kv_data_int4)

    kv_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device=device, dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout,
        use_tensor_cores=use_tensor_cores,
        backend="fa2",
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        data_type="int4",
        q_data_type=torch.float16,
    )
    out = wrapper.run(q, kv_data_int4)

    wrapper_ref = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout,
        use_tensor_cores=use_tensor_cores,
        backend="fa2",
    )
    wrapper_ref.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        data_type=torch.float16,
        q_data_type=torch.float16,
    )
    out_ref = wrapper_ref.run(q, kv_data_ref)

    torch.testing.assert_close(out, out_ref, rtol=1e-3, atol=1e-3)


def test_batch_decode_with_paged_kv_cache_int4_rejects_scale():
    batch_size = 2
    kv_len = 8
    page_size = 4
    num_kv_heads = 2
    num_qo_heads = 4
    head_dim = 128
    device = "cuda:0"

    q = torch.randn(
        batch_size, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = flashinfer.int4_quantize(
        torch.randn(
            total_num_pages,
            2,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=torch.float16,
            device=device,
        )
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device=device, dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        "NHD",
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        data_type="int4",
        q_data_type=torch.float16,
    )
    with pytest.raises(ValueError, match="k_scale and v_scale are not supported"):
        wrapper.run(q, kv_data, k_scale=0.5)


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("page_size", [4, 8])
def test_batch_prefill_with_paged_kv_cache_int4(kv_layout, head_dim, page_size):
    batch_size = 2
    kv_len = 8
    qo_len = 3
    num_kv_heads = 2
    num_qo_heads = 4
    device = "cuda:0"

    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * qo_len
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if kv_layout == "NHD":
        kv_data = torch.randn(
            total_num_pages,
            2,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=torch.float16,
            device=device,
        )
    else:
        kv_data = torch.randn(
            total_num_pages,
            2,
            num_kv_heads,
            page_size,
            head_dim,
            dtype=torch.float16,
            device=device,
        )
    kv_data_int4 = flashinfer.int4_quantize(kv_data)
    kv_data_ref = flashinfer.int4_dequantize(kv_data_int4)

    kv_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device=device, dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout,
        backend="fa2",
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        kv_data_type="int4",
    )
    out = wrapper.run(q, kv_data_int4)

    wrapper_ref = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout,
        backend="fa2",
    )
    wrapper_ref.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    out_ref = wrapper_ref.run(q, kv_data_ref)

    torch.testing.assert_close(out, out_ref, rtol=1e-3, atol=1e-3)


def test_batch_prefill_with_paged_kv_cache_int4_rejects_scale():
    batch_size = 2
    kv_len = 8
    qo_len = 3
    page_size = 4
    num_kv_heads = 2
    num_qo_heads = 4
    head_dim = 128
    device = "cuda:0"

    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * qo_len
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = flashinfer.int4_quantize(
        torch.randn(
            total_num_pages,
            2,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=torch.float16,
            device=device,
        )
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device=device, dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer,
        "NHD",
        backend="auto",
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        kv_data_type="int4",
    )
    with pytest.raises(ValueError, match="k_scale and v_scale are not supported"):
        wrapper.run(q, kv_data, k_scale=0.5)


def test_single_prefill_with_kv_cache_int4_auto_forces_fa2(
    monkeypatch: pytest.MonkeyPatch,
):
    seen = {}

    def fake_get_single_prefill_module(backend, *args, **kwargs):
        seen["backend"] = backend

        class _Module:
            def run(self, q, k, v, tmp, out, lse, *module_args):
                out.zero_()

        return _Module()

    monkeypatch.setattr(
        flashinfer_prefill,
        "get_single_prefill_module",
        fake_get_single_prefill_module,
    )

    q = torch.randn(3, 4, 128, dtype=torch.float16, device="cuda:0")
    k = flashinfer.int4_quantize(
        torch.randn(8, 2, 128, dtype=torch.float16, device="cuda:0")
    )
    v = flashinfer.int4_quantize(
        torch.randn(8, 2, 128, dtype=torch.float16, device="cuda:0")
    )

    flashinfer.single_prefill_with_kv_cache(q, k, v, kv_layout="NHD", backend="auto")
    assert seen["backend"] == "fa2"


def test_batch_wrappers_int4_auto_force_fa2():
    device = "cuda:0"
    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=device)

    decode_wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        "NHD",
        use_tensor_cores=True,
        backend="auto",
    )
    kv_indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
    kv_indices = torch.arange(4, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor([4, 4], dtype=torch.int32, device=device)
    decode_wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        4,
        2,
        128,
        4,
        data_type="int4",
        q_data_type=torch.float16,
    )
    assert decode_wrapper._backend == "fa2"

    prefill_wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer,
        "NHD",
        backend="auto",
    )
    qo_indptr = torch.tensor([0, 3, 6], dtype=torch.int32, device=device)
    prefill_wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        4,
        2,
        128,
        4,
        q_data_type=torch.float16,
        kv_data_type="int4",
    )
    assert prefill_wrapper._backend == "fa2"


def test_int4_paged_kv_cache_cuda_graph_unsupported():
    batch_size = 2
    kv_len = 8
    qo_len = 3
    page_size = 4
    num_kv_heads = 2
    num_qo_heads = 4
    head_dim = 128
    device = "cuda:0"

    kv_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
        * ((kv_len + page_size - 1) // page_size)
    )
    kv_indices = torch.arange(kv_indptr[-1].item(), device=device, dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * qo_len
    )

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=device)

    decode_wrapper = flashinfer.decode.CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        torch.empty(batch_size + 1, dtype=torch.int32, device=device),
        torch.empty(kv_indices.numel(), dtype=torch.int32, device=device),
        torch.empty(batch_size, dtype=torch.int32, device=device),
        "NHD",
    )
    with pytest.raises(NotImplementedError):
        decode_wrapper.plan(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            data_type="int4",
            q_data_type=torch.float16,
        )

    prefill_wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer,
        "NHD",
        use_cuda_graph=True,
        qo_indptr_buf=torch.empty(batch_size + 1, dtype=torch.int32, device=device),
        paged_kv_indptr_buf=torch.empty(
            batch_size + 1, dtype=torch.int32, device=device
        ),
        paged_kv_indices_buf=torch.empty(
            kv_indices.numel(), dtype=torch.int32, device=device
        ),
        paged_kv_last_page_len_buf=torch.empty(
            batch_size, dtype=torch.int32, device=device
        ),
    )
    with pytest.raises(NotImplementedError):
        prefill_wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            q_data_type=torch.float16,
            kv_data_type="int4",
        )
