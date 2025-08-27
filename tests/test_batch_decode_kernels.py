"""
Copyright (c) 2023 by FlashInfer team.

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
from jit_utils import gen_decode_attention_modules, gen_prefill_attention_modules

import flashinfer


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_decode_attention_modules(
            [torch.float16],  # q_dtypes
            [
                torch.float16,
                torch.float8_e4m3fn,
            ],  # kv_dtypes
            [128, 256],  # head_dims
            [0, 1],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [False],  # use_logits_soft_caps
        )
        + gen_prefill_attention_modules(
            [torch.float16],  # q_dtypes
            [
                torch.float16,
                torch.float8_e4m3fn,
            ],  # kv_dtypes
            [128, 256],  # head_dims
            [0, 1],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [False],  # use_logits_soft_caps
            [False],  # use_fp16_qk_reductions
        ),
        verbose=False,
    )
    yield


@pytest.mark.parametrize("batch_size", [12, 17, 128])
@pytest.mark.parametrize("kv_len", [54, 97, 512, 2048, 16384])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
@pytest.mark.parametrize("logits_soft_cap", [0.0])
@pytest.mark.parametrize("return_lse", [True])
@pytest.mark.parametrize("q_dtype", [torch.float16])
@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.float8_e4m3fn])
@pytest.mark.parametrize("contiguous_kv", [True])
def test_batch_decode_with_paged_kv_cache(
    batch_size,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    pos_encoding_mode,
    logits_soft_cap,
    return_lse,
    q_dtype,
    kv_dtype,
    contiguous_kv,
):
    q = torch.randn(batch_size, num_qo_heads, head_dim, device="cuda:0", dtype=q_dtype)
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if kv_layout == "HND":
        kv_shape = [total_num_pages, 2, num_kv_heads, page_size, head_dim]
    else:
        kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v)
        kv_shape = tmp
        kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
        kv_data = kv_data_fp32.to(kv_dtype)
        kv_data = kv_data[:, 1, :, 1, :, 1, :, 1, :]
        kv_data_fp32 = kv_data_fp32[:, 1, :, 1, :, 1, :, 1, :]
        # actual data is stored in non-contiguous memory
        assert (
            kv_data.stride(-4)
            != kv_data.shape[-3] * kv_data.shape[-2] * kv_data.shape[-1]
        )
    else:
        kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
        kv_data = kv_data_fp32.to(kv_dtype)
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda:0"
    )

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        logits_soft_cap=logits_soft_cap,
        pos_encoding_mode=pos_encoding_mode,
        data_type=kv_dtype,
        q_data_type=q_dtype,
    )
    if return_lse:
        o, _ = wrapper.run(q, kv_data, return_lse=True)
    else:
        o = wrapper.run(q, kv_data)

    for i in range(batch_size):
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
        qi = q[i]
        ki = torch.cat(
            [
                kv_data_fp32[kv_indptr[i] : kv_indptr[i + 1] - 1, 0]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data_fp32[kv_indptr[i + 1] - 1, 0, :, : kv_last_page_len[i]]
                    if kv_layout == "HND"
                    else kv_data_fp32[kv_indptr[i + 1] - 1, 0, : kv_last_page_len[i], :]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).to(kv_dtype)
        vi = torch.cat(
            [
                kv_data_fp32[kv_indptr[i] : kv_indptr[i + 1] - 1, 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data_fp32[kv_indptr[i + 1] - 1, 1, :, : kv_last_page_len[i]]
                    if kv_layout == "HND"
                    else kv_data_fp32[kv_indptr[i + 1] - 1, 1, : kv_last_page_len[i], :]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).to(kv_dtype)
        o_ref_i = flashinfer.decode.single_decode_with_kv_cache(
            qi,
            ki,
            vi,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        torch.testing.assert_close(o[i], o_ref_i, rtol=1e-3, atol=1e-3)

    # test user-allocated output
    o_buffer = torch.empty_like(o)
    wrapper.run(q, kv_data, out=o_buffer)
    torch.testing.assert_close(o, o_buffer, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [12, 17, 128])
@pytest.mark.parametrize("kv_len", [54, 97, 512, 2048, 16384])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
@pytest.mark.parametrize("logits_soft_cap", [0.0])
@pytest.mark.parametrize("return_lse", [True])
@pytest.mark.parametrize("q_dtype", [torch.float16])
@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.float8_e4m3fn])
@pytest.mark.parametrize("contiguous_kv", [True])
def test_batch_decode_with_tuple_paged_kv_cache(
    batch_size,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    pos_encoding_mode,
    logits_soft_cap,
    return_lse,
    q_dtype,
    kv_dtype,
    contiguous_kv,
):
    q = torch.randn(batch_size, num_qo_heads, head_dim, device="cuda:0", dtype=q_dtype)
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if kv_layout == "HND":
        kv_shape = [total_num_pages, num_kv_heads, page_size, head_dim]
    else:
        kv_shape = [total_num_pages, page_size, num_kv_heads, head_dim]
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v)
        kv_shape = tmp
        kv_data_fp32 = [
            torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
            for _ in range(2)
        ]
        kv_data = [kv_data_fp32[i].to(kv_dtype) for i in range(2)]
        for i in range(2):
            kv_data_fp32[i] = kv_data_fp32[i][:, 1, :, 1, :, 1, :]
            kv_data[i] = kv_data[i][:, 1, :, 1, :, 1, :]
            # actual data is stored in non-contiguous memory
            assert (
                kv_data[i].stride(-4)
                != kv_data[i].shape[-3] * kv_data[i].shape[-2] * kv_data[i].shape[-1]
            )
    else:
        kv_data_fp32 = [
            torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
            for _ in range(2)
        ]
        kv_data = [kv_data_fp32[i].to(kv_dtype) for i in range(2)]
    kv_data = tuple(kv_data)
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda:0"
    )

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        logits_soft_cap=logits_soft_cap,
        pos_encoding_mode=pos_encoding_mode,
        data_type=kv_dtype,
        q_data_type=q_dtype,
    )
    if return_lse:
        o, _ = wrapper.run(q, kv_data, return_lse=True)
    else:
        o = wrapper.run(q, kv_data)

    k_cache, v_cache = kv_data_fp32
    for i in range(batch_size):
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
        qi = q[i]
        ki = torch.cat(
            [
                k_cache[kv_indptr[i] : kv_indptr[i + 1] - 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    k_cache[kv_indptr[i + 1] - 1, :, : kv_last_page_len[i]]
                    if kv_layout == "HND"
                    else k_cache[kv_indptr[i + 1] - 1, : kv_last_page_len[i], :]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).to(kv_dtype)
        vi = torch.cat(
            [
                v_cache[kv_indptr[i] : kv_indptr[i + 1] - 1]
                .to(torch.float32)  # torch.cat does not support some fp8 types
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    v_cache[kv_indptr[i + 1] - 1, :, : kv_last_page_len[i]]
                    if kv_layout == "HND"
                    else v_cache[kv_indptr[i + 1] - 1, : kv_last_page_len[i], :]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).to(kv_dtype)
        o_ref_i = flashinfer.decode.single_decode_with_kv_cache(
            qi,
            ki,
            vi,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        torch.testing.assert_close(o[i], o_ref_i, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [12, 17, 128])
@pytest.mark.parametrize("kv_len", [54, 2048, 16384])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
@pytest.mark.parametrize("q_dtype", [torch.float16])
@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.float8_e4m3fn])
@pytest.mark.parametrize("contiguous_kv", [True])
def test_cuda_graph_batch_decode_with_paged_kv_cache(
    batch_size,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    pos_encoding_mode,
    q_dtype,
    kv_dtype,
    contiguous_kv,
):
    q = torch.randn(batch_size, num_qo_heads, head_dim, device="cuda:0", dtype=q_dtype)
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if kv_layout == "HND":
        kv_shape = [total_num_pages, 2, num_kv_heads, page_size, head_dim]
    else:
        kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v)
        kv_shape = tmp
        kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
        kv_data = kv_data_fp32.to(kv_dtype)
        kv_data_fp32 = kv_data_fp32[:, 1, :, 1, :, 1, :, 1, :]
        kv_data = kv_data[:, 1, :, 1, :, 1, :, 1, :]
        # actual data is stored in non-contiguous memory
        assert (
            kv_data.stride(-4)
            != kv_data.shape[-3] * kv_data.shape[-2] * kv_data.shape[-1]
        )
    else:
        kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
        kv_data = kv_data_fp32.to(kv_dtype)
    kv_indptr_host_warmup = torch.arange(
        0, batch_size + 1, device="cuda:0", dtype=torch.int32
    )
    kv_indices_host_warmup = torch.arange(
        0, batch_size, device="cuda:0", dtype=torch.int32
    )
    kv_last_page_len_host_warmup = torch.full(
        (batch_size,), page_size, dtype=torch.int32
    )

    # NOTE(Zihao): allocate more space than needed for testing
    kv_indptr_device_buffer = torch.empty(
        batch_size + 1, device="cuda:0", dtype=torch.int32
    )
    kv_indices_device_buffer = torch.empty(
        total_num_pages, device="cuda:0", dtype=torch.int32
    )
    kv_last_page_device_buffer = torch.empty(
        batch_size, device="cuda:0", dtype=torch.int32
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.decode.CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_indptr_device_buffer,
        kv_indices_device_buffer,
        kv_last_page_device_buffer,
        kv_layout,
    )
    wrapper.plan(
        kv_indptr_host_warmup,
        kv_indices_host_warmup,
        kv_last_page_len_host_warmup,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        data_type=kv_dtype,
        pos_encoding_mode=pos_encoding_mode,
        q_data_type=q_dtype,
    )
    # warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            o = wrapper.run(q, kv_data)
    torch.cuda.current_stream().wait_stream(s)

    # capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o = wrapper.run(q, kv_data)

    # replay multiple times
    for i in range(1, min(4, num_pages_per_seq)):
        kv_indptr_host = torch.arange(0, batch_size + 1).int() * i
        kv_indices_host = torch.arange(0, i * batch_size).int()
        kv_last_page_len_host = torch.full((batch_size,), page_size, dtype=torch.int32)

        wrapper.plan(
            kv_indptr_host,
            kv_indices_host,
            kv_last_page_len_host,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            data_type=kv_dtype,
            pos_encoding_mode=pos_encoding_mode,
            q_data_type=q_dtype,
        )
        g.replay()

    # replay again
    kv_indptr_host = torch.arange(0, batch_size + 1).int() * num_pages_per_seq
    kv_indices_host = torch.arange(0, total_num_pages).int()
    kv_last_page_len_host = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )

    wrapper.plan(
        kv_indptr_host,
        kv_indices_host,
        kv_last_page_len_host,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        data_type=kv_dtype,
        pos_encoding_mode=pos_encoding_mode,
        q_data_type=q_dtype,
    )
    g.replay()

    # compute ground truth and compare
    kv_indptr = kv_indptr_host.to(0)
    kv_last_page_len = kv_last_page_len_host.to(0)
    for i in range(batch_size):
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
        qi = q[i]
        ki = torch.cat(
            [
                kv_data_fp32[kv_indptr[i] : kv_indptr[i + 1] - 1, 0]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data_fp32[kv_indptr[i + 1] - 1, 0, :, : kv_last_page_len[i]]
                    if kv_layout == "HND"
                    else kv_data_fp32[kv_indptr[i + 1] - 1, 0, : kv_last_page_len[i], :]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).to(kv_dtype)
        vi = torch.cat(
            [
                kv_data_fp32[kv_indptr[i] : kv_indptr[i + 1] - 1, 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data_fp32[kv_indptr[i + 1] - 1, 1, :, : kv_last_page_len[i]]
                    if kv_layout == "HND"
                    else kv_data_fp32[kv_indptr[i + 1] - 1, 1, : kv_last_page_len[i], :]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).to(kv_dtype)
        o_ref_i = flashinfer.decode.single_decode_with_kv_cache(
            qi, ki, vi, pos_encoding_mode=pos_encoding_mode
        )
        torch.testing.assert_close(o[i], o_ref_i, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_batch_decode_with_paged_kv_cache(
        256,
        54,
        8,
        8,
        8,
        128,
        "NHD",
        "NONE",
        0.0,
        False,
        torch.float16,
        torch.float16,
        True,
    )
    test_batch_decode_with_tuple_paged_kv_cache(
        256,
        54,
        8,
        8,
        8,
        128,
        "NHD",
        "NONE",
        0.0,
        False,
        torch.float16,
        torch.float16,
        True,
    )
    test_batch_decode_with_paged_kv_cache(
        12,
        2048,
        8,
        8,
        8,
        128,
        "NHD",
        "NONE",
        0.0,
        False,
        torch.float16,
        torch.float16,
        True,
    )
    test_batch_decode_with_paged_kv_cache(
        12,
        54,
        1,
        8,
        8,
        128,
        "HND",
        "NONE",
        0.0,
        True,
        torch.float16,
        torch.float8_e5m2,
        True,
    )
    test_cuda_graph_batch_decode_with_paged_kv_cache(
        12, 2048, 8, 8, 8, 128, "NHD", "NONE", torch.float16, torch.float16, True
    )
    test_cuda_graph_batch_decode_with_paged_kv_cache(
        128, 54, 8, 8, 8, 128, "NHD", "NONE", torch.float16, torch.float16, True
    )
    test_batch_decode_with_paged_kv_cache(
        12,
        54,
        1,
        8,
        8,
        128,
        "HND",
        "NONE",
        0.0,
        True,
        torch.float16,
        torch.float8_e5m2,
        True,
    )
    test_cuda_graph_batch_decode_with_paged_kv_cache(
        12, 54, 8, 8, 8, 128, "HND", "NONE", torch.float16, torch.float8_e5m2, True
    )
