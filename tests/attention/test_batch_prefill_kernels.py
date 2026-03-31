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

import numpy
import pytest
import torch
from tests.test_helpers.jit_utils import gen_prefill_attention_modules

import flashinfer
from flashinfer.utils import has_flashinfer_jit_cache


@pytest.fixture(
    autouse=not has_flashinfer_jit_cache(),
    scope="module",
)
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_prefill_attention_modules(
            [torch.float16],  # q_dtypes
            [
                torch.float16,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
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
@pytest.mark.parametrize("kv_len", [54, 97, 512, 2048])
@pytest.mark.parametrize("qo_len", [37, 17, 127, 577])
@pytest.mark.parametrize("page_size", [1, 5, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
@pytest.mark.parametrize("use_cuda_graph", [False, True])
@pytest.mark.parametrize("logits_soft_cap", [0.0])
@pytest.mark.parametrize("return_lse", [True])
@pytest.mark.parametrize("contiguous_kv", [True])
def test_batch_prefill_with_paged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    kv_layout,
    pos_encoding_mode,
    use_cuda_graph,
    logits_soft_cap,
    return_lse,
    contiguous_kv,
):
    if use_cuda_graph:
        pytest.xfail(
            "NOTE(Zihao): temporarily disable cuda graph until we fully fix the workspace buffer overflow issue for prefill + cudagraph"
        )
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")
    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    q_indptr_cpu = torch.arange(0, batch_size + 1).int() * qo_len
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
        kv_data = kv_data_fp32.half()
        kv_data = kv_data[:, 1, :, 1, :, 1, :, 1, :]
        kv_data_fp32 = kv_data_fp32[:, 1, :, 1, :, 1, :, 1, :]
        # actual data is stored in non-contiguous memory
        assert (
            kv_data.stride(-4)
            != kv_data.shape[-3] * kv_data.shape[-2] * kv_data.shape[-1]
        )
    else:
        kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
        kv_data = kv_data_fp32.half()
    kv_indptr_cpu = torch.arange(0, batch_size + 1).int() * num_pages_per_seq
    kv_indices_cpu = torch.arange(0, total_num_pages).int()
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )

    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    if not use_cuda_graph:
        q_indptr_gpu = q_indptr_cpu.to(0)
        kv_indptr_gpu = kv_indptr_cpu.to(0)
        kv_indices_gpu = kv_indices_cpu.to(0)
        kv_last_page_len_gpu = kv_last_page_len_cpu.to(0)
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout
        )
        wrapper.plan(
            q_indptr_gpu,
            kv_indptr_gpu,
            kv_indices_gpu,
            kv_last_page_len_gpu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        if return_lse:
            o, lse = wrapper.run(q, kv_data, return_lse=True)
        else:
            o = wrapper.run(q, kv_data)

        # test with pre-allocated output
        o_buffer = torch.empty_like(o)
        if return_lse:
            lse_buffer = torch.empty_like(lse)
            wrapper.run(q, kv_data, out=o_buffer, lse=lse_buffer, return_lse=True)
        else:
            wrapper.run(q, kv_data, out=o_buffer)
        torch.testing.assert_close(o, o_buffer, rtol=1e-3, atol=1e-3)
    else:
        q_indptr_buffer = torch.empty(
            batch_size + 1, device="cuda:0", dtype=torch.int32
        )
        kv_indptr_buffer = torch.empty(
            batch_size + 1, device="cuda:0", dtype=torch.int32
        )
        kv_indices_buffer = torch.empty(
            total_num_pages, device="cuda:0", dtype=torch.int32
        )
        kv_last_page_len_buffer = torch.empty(
            batch_size, device="cuda:0", dtype=torch.int32
        )
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer,
            kv_layout,
            use_cuda_graph=True,
            qo_indptr_buf=q_indptr_buffer,
            paged_kv_indptr_buf=kv_indptr_buffer,
            paged_kv_indices_buf=kv_indices_buffer,
            paged_kv_last_page_len_buf=kv_last_page_len_buffer,
        )
        q_indptr_warmup = torch.arange(0, batch_size + 1).int() * qo_len
        kv_indptr_warmup = torch.arange(0, batch_size + 1).int()
        kv_indices_warmup = torch.arange(0, batch_size).int()
        kv_last_page_len_warmup = torch.full(
            (batch_size,), page_size, dtype=torch.int32
        )

        wrapper.plan(
            q_indptr_warmup,
            kv_indptr_warmup,
            kv_indices_warmup,
            kv_last_page_len_warmup,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )

        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                if return_lse:
                    o, _ = wrapper.run(q, kv_data, return_lse=True)
                else:
                    o = wrapper.run(q, kv_data)
        torch.cuda.current_stream().wait_stream(s)
        # capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            if return_lse:
                o, _ = wrapper.run(q, kv_data, return_lse=True)
            else:
                o = wrapper.run(q, kv_data)

        wrapper.plan(
            q_indptr_cpu,
            kv_indptr_cpu,
            kv_indices_cpu,
            kv_last_page_len_cpu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )

        g.replay()

    for i in range(batch_size):
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        ki = torch.cat(
            [
                kv_data_fp32[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 0]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 0, :, : kv_last_page_len_cpu[i]
                    ]
                    if kv_layout == "HND"
                    else kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 0, : kv_last_page_len_cpu[i], :
                    ]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).half()
        vi = torch.cat(
            [
                kv_data_fp32[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 1, :, : kv_last_page_len_cpu[i]
                    ]
                    if kv_layout == "HND"
                    else kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 1, : kv_last_page_len_cpu[i], :
                    ]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).half()
        o_ref_i = flashinfer.prefill.single_prefill_with_kv_cache(
            qi,
            ki,
            vi,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        o_i = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        torch.testing.assert_close(o_i, o_ref_i, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [12, 17, 128])
@pytest.mark.parametrize("kv_len", [54, 97, 512, 2048])
@pytest.mark.parametrize("qo_len", [37, 17, 127, 577])
@pytest.mark.parametrize("page_size", [1, 5, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
@pytest.mark.parametrize("use_cuda_graph", [False, True])
@pytest.mark.parametrize("logits_soft_cap", [0.0])
@pytest.mark.parametrize("return_lse", [True])
@pytest.mark.parametrize("contiguous_kv", [True])
def test_batch_prefill_with_tuple_paged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    kv_layout,
    pos_encoding_mode,
    use_cuda_graph,
    logits_soft_cap,
    return_lse,
    contiguous_kv,
):
    if use_cuda_graph:
        pytest.xfail(
            "NOTE(Zihao): temporarily disable cuda graph until we fully fix the workspace buffer overflow issue for prefill + cudagraph"
        )
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")
    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    q_indptr_cpu = torch.arange(0, batch_size + 1).int() * qo_len
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
        kv_data = [kv_data_fp32[i].half() for i in range(2)]
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
        kv_data = [kv_data_fp32[i].half() for i in range(2)]
    kv_data = tuple(kv_data)
    kv_indptr_cpu = torch.arange(0, batch_size + 1).int() * num_pages_per_seq
    kv_indices_cpu = torch.arange(0, total_num_pages).int()
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )

    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    if not use_cuda_graph:
        q_indptr_gpu = q_indptr_cpu.to(0)
        kv_indptr_gpu = kv_indptr_cpu.to(0)
        kv_indices_gpu = kv_indices_cpu.to(0)
        kv_last_page_len_gpu = kv_last_page_len_cpu.to(0)
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout
        )
        wrapper.plan(
            q_indptr_gpu,
            kv_indptr_gpu,
            kv_indices_gpu,
            kv_last_page_len_gpu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        if return_lse:
            o, _ = wrapper.run(q, kv_data, return_lse=True)
        else:
            o = wrapper.run(q, kv_data)
    else:
        q_indptr_buffer = torch.empty(
            batch_size + 1, device="cuda:0", dtype=torch.int32
        )
        kv_indptr_buffer = torch.empty(
            batch_size + 1, device="cuda:0", dtype=torch.int32
        )
        kv_indices_buffer = torch.empty(
            total_num_pages, device="cuda:0", dtype=torch.int32
        )
        kv_last_page_len_buffer = torch.empty(
            batch_size, device="cuda:0", dtype=torch.int32
        )
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer,
            kv_layout,
            use_cuda_graph=True,
            qo_indptr_buf=q_indptr_buffer,
            paged_kv_indptr_buf=kv_indptr_buffer,
            paged_kv_indices_buf=kv_indices_buffer,
            paged_kv_last_page_len_buf=kv_last_page_len_buffer,
        )
        q_indptr_warmup = torch.arange(0, batch_size + 1).int() * qo_len
        kv_indptr_warmup = torch.arange(0, batch_size + 1).int()
        kv_indices_warmup = torch.arange(0, batch_size).int()
        kv_last_page_len_warmup = torch.full(
            (batch_size,), page_size, dtype=torch.int32
        )
        wrapper.plan(
            q_indptr_warmup,
            kv_indptr_warmup,
            kv_indices_warmup,
            kv_last_page_len_warmup,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )

        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                if return_lse:
                    o, _ = wrapper.run(q, kv_data, return_lse=True)
                else:
                    o = wrapper.run(q, kv_data)
        torch.cuda.current_stream().wait_stream(s)
        # capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            if return_lse:
                o, _ = wrapper.run(q, kv_data, return_lse=True)
            else:
                o = wrapper.run(q, kv_data)

        wrapper.plan(
            q_indptr_cpu,
            kv_indptr_cpu,
            kv_indices_cpu,
            kv_last_page_len_cpu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )

        g.replay()

    k_cache, v_cache = kv_data_fp32
    for i in range(batch_size):
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        ki = torch.cat(
            [
                k_cache[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    k_cache[kv_indptr_cpu[i + 1] - 1, :, : kv_last_page_len_cpu[i]]
                    if kv_layout == "HND"
                    else k_cache[kv_indptr_cpu[i + 1] - 1, : kv_last_page_len_cpu[i], :]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).half()
        vi = torch.cat(
            [
                v_cache[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    v_cache[kv_indptr_cpu[i + 1] - 1, :, : kv_last_page_len_cpu[i]]
                    if kv_layout == "HND"
                    else v_cache[kv_indptr_cpu[i + 1] - 1, : kv_last_page_len_cpu[i], :]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).half()
        o_ref_i = flashinfer.prefill.single_prefill_with_kv_cache(
            qi,
            ki,
            vi,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        o_i = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        torch.testing.assert_close(o_i, o_ref_i, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [12, 17, 128])
@pytest.mark.parametrize("kv_len", [54, 97, 512, 2048])
@pytest.mark.parametrize("qo_len", [37, 17, 127, 577])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
@pytest.mark.parametrize("logits_soft_cap", [0.0])
@pytest.mark.parametrize("return_lse", [True])
@pytest.mark.parametrize("contiguous_kv", [True])
def test_batch_prefill_with_paged_kv_cache_custom_mask(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    pos_encoding_mode,
    logits_soft_cap,
    return_lse,
    contiguous_kv,
):
    if qo_len > kv_len:
        pytest.skip("qo_len > kv_len is not supported for custom mask test")
    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    q_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    )
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
        kv_data = torch.randn(*kv_shape, dtype=torch.float16, device="cuda:0")
        kv_data = kv_data[:, 1, :, 1, :, 1, :, 1, :]
        # actual data is stored in non-contiguous memory
        assert (
            kv_data.stride(-4)
            != kv_data.shape[-3] * kv_data.shape[-2] * kv_data.shape[-1]
        )
    else:
        kv_data = torch.randn(*kv_shape, dtype=torch.float16, device="cuda:0")
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda:0"
    )

    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    custom_mask = torch.tril(
        torch.full((batch_size, qo_len, kv_len), True, device="cuda:0"),
        diagonal=(kv_len - qo_len),
    ).reshape(-1)

    # use custom mask
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        custom_mask=custom_mask,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
    )
    if return_lse:
        o_custom, _ = wrapper.run(q, kv_data, return_lse=True)
    else:
        o_custom = wrapper.run(q, kv_data)

    # use causal
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
    )
    if return_lse:
        o_causal, _ = wrapper.run(q, kv_data, return_lse=True)
    else:
        o_causal = wrapper.run(q, kv_data)
    torch.testing.assert_close(o_custom, o_causal, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [12, 17, 128])
@pytest.mark.parametrize("kv_len", [54, 97, 512, 2048])
@pytest.mark.parametrize("qo_len", [37, 17, 127, 577])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
@pytest.mark.parametrize("logits_soft_cap", [0.0])
@pytest.mark.parametrize("return_lse", [True])
def test_batch_prefill_with_ragged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    pos_encoding_mode,
    logits_soft_cap,
    return_lse,
):
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")
    kv_layout = "NHD"
    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    q_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    )

    k = torch.randn(
        batch_size * kv_len,
        num_kv_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    v = torch.randn(
        batch_size * kv_len,
        num_kv_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * kv_len
    )

    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper.plan(
        q_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
    )
    if return_lse:
        o, _ = wrapper.run(q, k, v, return_lse=True)
    else:
        o = wrapper.run(q, k, v)

    for i in range(batch_size):
        o_ref_i = flashinfer.prefill.single_prefill_with_kv_cache(
            q[q_indptr[i] : q_indptr[i + 1]],
            k[kv_indptr[i] : kv_indptr[i + 1]],
            v[kv_indptr[i] : kv_indptr[i + 1]],
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        o_i = o[q_indptr[i] : q_indptr[i + 1]]
        torch.testing.assert_close(o_i, o_ref_i, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [12, 17, 128])
@pytest.mark.parametrize("kv_len", [54, 97, 512, 2048])
@pytest.mark.parametrize("qo_len", [37, 17, 127, 577])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA", "ALIBI"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("return_lse", [True, False])
def test_batch_prefill_with_ragged_kv_cache_custom_mask(
    batch_size,
    kv_len,
    qo_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    pos_encoding_mode,
    logits_soft_cap,
    return_lse,
):
    kv_layout = "NHD"
    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    q_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    )

    k = torch.randn(
        batch_size * kv_len,
        num_kv_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    v = torch.randn(
        batch_size * kv_len,
        num_kv_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * kv_len
    )

    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout
    )

    custom_mask = torch.tril(
        torch.full((batch_size, qo_len, kv_len), True, device="cuda:0"),
        diagonal=(kv_len - qo_len),
    ).reshape(-1)

    # use custom mask
    wrapper.plan(
        q_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        custom_mask=custom_mask,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
    )
    if return_lse:
        o_custom, _ = wrapper.run(q, k, v, return_lse=True)
    else:
        o_custom = wrapper.run(q, k, v)

    # use causal
    wrapper.plan(
        q_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=True,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
    )
    if return_lse:
        o_causal, _ = wrapper.run(q, k, v, return_lse=True)
    else:
        o_causal = wrapper.run(q, k, v)
    torch.testing.assert_close(o_custom, o_causal, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "kv_len, qo_len, prefix_len_ptr, token_pos_in_items_ptr, token_pos_in_items_len, max_item_len_ptr",
    [
        (54, 37, 17, list(range(17)) + list(range(19)) + [0], 100, [18]),
        (97, 81, 16, list(range(80)) + [0], 97, [79]),
    ],
)
@pytest.mark.parametrize("page_size", [1, 5, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["ROPE_LLAMA"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("return_lse", [True, False])
def test_batch_prefill_with_paged_kv_cache_multi_item_scoring(
    batch_size,
    kv_len,
    qo_len,
    prefix_len_ptr,
    token_pos_in_items_ptr,
    token_pos_in_items_len,
    max_item_len_ptr,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    kv_layout,
    pos_encoding_mode,
    logits_soft_cap,
    return_lse,
):
    q = torch.randn(batch_size * qo_len, num_qo_heads, head_dim).to(0).half()
    q_indptr_cpu = torch.arange(0, batch_size + 1).int() * qo_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = (
        torch.randn(total_num_pages, 2, num_kv_heads, page_size, head_dim).to(0).half()
        if kv_layout == "HND"
        else torch.randn(total_num_pages, 2, page_size, num_kv_heads, head_dim)
        .to(0)
        .half()
    )
    kv_indptr_cpu = torch.arange(0, batch_size + 1).int() * num_pages_per_seq
    kv_indices_cpu = torch.arange(0, total_num_pages).int()
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    q_indptr_gpu = q_indptr_cpu.to(0)
    kv_indptr_gpu = kv_indptr_cpu.to(0)
    kv_indices_gpu = kv_indices_cpu.to(0)
    kv_last_page_len_gpu = kv_last_page_len_cpu.to(0)
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper.plan(
        q_indptr_gpu,
        kv_indptr_gpu,
        kv_indices_gpu,
        kv_last_page_len_gpu,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
        prefix_len_ptr=torch.tensor(prefix_len_ptr).to(dtype=torch.uint32).to(0),
        token_pos_in_items_ptr=torch.tensor(token_pos_in_items_ptr)
        .to(dtype=torch.uint16)
        .to(0),
        token_pos_in_items_len=token_pos_in_items_len,
        max_item_len_ptr=torch.tensor(max_item_len_ptr).to(dtype=torch.uint16).to(0),
    )
    if return_lse:
        o, _ = wrapper.run_return_lse(q, kv_data)
    else:
        o = wrapper.run(q, kv_data)

    for i in range(batch_size):
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        ki = torch.cat(
            [
                kv_data[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 0]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data[kv_indptr_cpu[i + 1] - 1, 0, :, : kv_last_page_len_cpu[i]]
                    if kv_layout == "HND"
                    else kv_data[
                        kv_indptr_cpu[i + 1] - 1, 0, : kv_last_page_len_cpu[i], :
                    ]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        )
        vi = torch.cat(
            [
                kv_data[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data[kv_indptr_cpu[i + 1] - 1, 1, :, : kv_last_page_len_cpu[i]]
                    if kv_layout == "HND"
                    else kv_data[
                        kv_indptr_cpu[i + 1] - 1, 1, : kv_last_page_len_cpu[i], :
                    ]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        )

        def create_2D_multi_item_mask_dense(
            is_delimiter, sliding_window_size=-1, prefix_cache_len=None
        ):
            # Function to create custom_mask for multi-item scoring
            #
            # Note, sliding window implementation assumes that candidate_i_size < sliding_window_size < prefix_size
            # Args:
            # is_delimiter: a boolen torch vec to indicate the delimiter position for creating custom attnetion mask in multi-item scoring
            #           currently assume qo len and kv len are the same and 1D (bsz=1) case
            # sliding_window_size: the window size for sliding window attention, -1 means no sliding window attention
            delimiter_idx = is_delimiter.nonzero(as_tuple=True)[0]
            if len(delimiter_idx) == 0:
                return None
            else:
                first_delimiter_pos = delimiter_idx[0]
            seq_len = len(is_delimiter)
            pos = torch.arange(seq_len, device=is_delimiter.device)

            group_ids = torch.cumsum(is_delimiter, 0)
            # Get mask for within-group causal attention
            within_group_causal = (group_ids.unsqueeze(1) == group_ids.unsqueeze(0)) & (
                pos.unsqueeze(0) <= pos.unsqueeze(1)
            )
            # Combine all conditions
            attention_mask = (
                (
                    within_group_causal
                    | (
                        (pos >= first_delimiter_pos).unsqueeze(1)
                        & (pos < first_delimiter_pos).unsqueeze(0)
                    )  # Prefix attention
                )
                & ~is_delimiter.unsqueeze(0)
                & ~is_delimiter.unsqueeze(1)
            )  # No delimiter attention

            if sliding_window_size > 0 and sliding_window_size < len(is_delimiter):
                # Calculate how many positions from right of prefix each token can attend to

                group_size = torch.sum(
                    within_group_causal & ~is_delimiter.unsqueeze(0), dim=1
                )

                # For prefix: after sliding_window_size position, can see window_size tokens
                # For candidate items: can see (sliding_window_size - group_size) tokens from prefix end
                prefix_window = torch.where(
                    pos >= first_delimiter_pos,
                    sliding_window_size - group_size,
                    torch.where(
                        pos < sliding_window_size,
                        first_delimiter_pos,
                        sliding_window_size,
                    ),
                )

                # Starting index of attention window relative to token position for candidate item/group
                prefix_start = first_delimiter_pos - prefix_window.unsqueeze(1)

                attention_mask = attention_mask & (pos >= prefix_start)
            if prefix_cache_len:
                patch = torch.ones(
                    seq_len,
                    prefix_cache_len,
                    device=is_delimiter.device,
                    dtype=torch.bool,
                )
                attention_mask = torch.concat([patch, attention_mask], dim=1)
            return attention_mask.unsqueeze(0).reshape(-1)

        custom_mask = create_2D_multi_item_mask_dense(
            is_delimiter=torch.tensor(token_pos_in_items_ptr).to(0) == 0,
            sliding_window_size=-1,
            prefix_cache_len=prefix_len_ptr,
        )
        o_ref_i = flashinfer.prefill.single_prefill_with_kv_cache(
            qi,
            ki,
            vi,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
            custom_mask=custom_mask,
        )
        o_i_np = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]].cpu().numpy()
        o_ref_i_np = o_ref_i.cpu().numpy()
        numpy.testing.assert_allclose(o_i_np, o_ref_i_np, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "prefix_len, item_lens",
    [
        (17, [3, 2, 4]),
        (16, [40, 40]),
    ],
)
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["ROPE_LLAMA"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
def test_batch_prefill_with_paged_kv_cache_multi_item_scoring_v2(
    batch_size,
    prefix_len,
    item_lens,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    pos_encoding_mode,
    logits_soft_cap,
):
    """Test delimiterless multi-item scoring (V2) against custom mask reference."""
    import torch

    total_items_tokens = sum(item_lens)
    # V2 has no delimiter tokens in the sequence
    qo_len = total_items_tokens  # items region only (prefix is in KV cache)
    kv_len = prefix_len + total_items_tokens

    # Build item_offsets: CSR-style [0, len1, len1+len2, ...]
    offsets = [0]
    for l in item_lens:
        offsets.append(offsets[-1] + l)
    item_offsets_tensor = torch.tensor(offsets, dtype=torch.uint32).to(0)
    item_offsets_len = len(offsets)

    q = torch.randn(batch_size * qo_len, num_qo_heads, head_dim).to(0).half()
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = (
        torch.randn(total_num_pages, 2, page_size, num_kv_heads, head_dim).to(0).half()
    )
    q_indptr_cpu = torch.arange(0, batch_size + 1).int() * qo_len
    kv_indptr_cpu = torch.arange(0, batch_size + 1).int() * num_pages_per_seq
    kv_indices_cpu = torch.arange(0, total_num_pages).int()
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper.plan(
        q_indptr_cpu.to(0),
        kv_indptr_cpu.to(0),
        kv_indices_cpu.to(0),
        kv_last_page_len_cpu.to(0),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
        prefix_len_ptr=torch.tensor([prefix_len], dtype=torch.uint32).to(0),
        item_offsets=item_offsets_tensor,
        item_offsets_len=item_offsets_len,
    )
    o = wrapper.run(q, kv_data)

    # Build reference using custom mask
    # Reconstruct full KV from paged format
    perm_dims_last = [0, 1, 2]
    ki = torch.cat(
        [
            kv_data[kv_indptr_cpu[0] : kv_indptr_cpu[1] - 1, 0]
            .reshape(-1, num_kv_heads, head_dim),
            kv_data[kv_indptr_cpu[1] - 1, 0, : kv_last_page_len_cpu[0], :]
            .reshape(-1, num_kv_heads, head_dim),
        ],
        dim=0,
    )
    vi = torch.cat(
        [
            kv_data[kv_indptr_cpu[0] : kv_indptr_cpu[1] - 1, 1]
            .reshape(-1, num_kv_heads, head_dim),
            kv_data[kv_indptr_cpu[1] - 1, 1, : kv_last_page_len_cpu[0], :]
            .reshape(-1, num_kv_heads, head_dim),
        ],
        dim=0,
    )

    # Build custom mask: [qo_len, kv_len] boolean
    # Each Q token is in the items region (q_idx maps to kv position prefix_len + q_idx)
    custom_mask = torch.zeros(qo_len, kv_len, dtype=torch.bool, device=q.device)
    for q_idx in range(qo_len):
        kv_pos = prefix_len + q_idx  # Q's position in full KV sequence
        # Find which item this Q belongs to
        item_start_rel = 0
        item_end_rel = 0
        for j in range(len(offsets) - 1):
            if offsets[j] <= q_idx < offsets[j + 1]:
                item_start_rel = offsets[j]
                item_end_rel = offsets[j + 1]
                break
        # Q can attend to: all prefix tokens
        custom_mask[q_idx, :prefix_len] = True
        # Q can attend to: same item tokens, causally (kv_pos <= q's kv_pos)
        kv_item_start = prefix_len + item_start_rel
        custom_mask[q_idx, kv_item_start:kv_pos + 1] = True

    custom_mask_flat = custom_mask.unsqueeze(0).reshape(-1)
    o_ref = flashinfer.prefill.single_prefill_with_kv_cache(
        q[:qo_len],
        ki,
        vi,
        causal=True,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
        custom_mask=custom_mask_flat,
    )
    numpy.testing.assert_allclose(
        o[:qo_len].cpu().numpy(), o_ref.cpu().numpy(), rtol=1e-3, atol=1e-3
    )


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "prefix_len, item_lens",
    [
        (17, [3, 2, 4]),
        (16, [40, 40]),
        (32, [16, 8, 16]),
    ],
)
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
def test_mis_v1_v2_output_parity(
    batch_size,
    prefix_len,
    item_lens,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    logits_soft_cap,
):
    """Verify MIS v2 item-token outputs match MIS v1 item-token outputs for equivalent inputs.

    Both modes encode the same logical attention pattern: each item token attends
    to all prefix KV tokens plus same-item KV tokens causally. V1 has delimiter
    tokens interspersed (masked to -inf); V2 has no delimiters.

    Uses pos_encoding_mode=NONE to avoid RoPE position-index differences between V1/V2.
    Shares Q, K, V tensors at item and prefix positions so the only differences are
    the delimiter tokens in V1 (which are masked out and don't affect output).
    """
    import torch

    total_item_tokens = sum(item_lens)
    num_items = len(item_lens)

    # Shared Q for item tokens (used by both V1 and V2)
    q_items = torch.randn(
        batch_size * total_item_tokens, num_qo_heads, head_dim, dtype=torch.float16, device="cuda"
    )

    # Shared K/V for prefix and item tokens
    prefix_k = torch.randn(prefix_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
    prefix_v = torch.randn(prefix_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
    item_k = torch.randn(total_item_tokens, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
    item_v = torch.randn(total_item_tokens, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")

    def pack_flat_kv_to_pages(k_flat, v_flat, kv_len, page_size):
        """Pack flat [kv_len, nh, hd] K and V into paged format [num_pages, 2, page_size, nh, hd]."""
        num_pages = (kv_len + page_size - 1) // page_size
        kv_pages = torch.zeros(num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
        for pos in range(kv_len):
            kv_pages[pos // page_size, 0, pos % page_size] = k_flat[pos]
            kv_pages[pos // page_size, 1, pos % page_size] = v_flat[pos]
        return kv_pages

    # ---- V2 ----
    qo_len_v2 = total_item_tokens
    kv_len_v2 = prefix_len + total_item_tokens

    k_flat_v2 = torch.cat([prefix_k, item_k], dim=0)
    v_flat_v2 = torch.cat([prefix_v, item_v], dim=0)
    kv_data_v2 = pack_flat_kv_to_pages(k_flat_v2, v_flat_v2, kv_len_v2, page_size)

    num_pages_v2 = (kv_len_v2 + page_size - 1) // page_size
    q_indptr_v2 = (torch.arange(batch_size + 1) * qo_len_v2).int().cuda()
    kv_indptr_v2 = (torch.arange(batch_size + 1) * num_pages_v2).int().cuda()
    kv_indices_v2 = torch.arange(num_pages_v2 * batch_size).int().cuda()
    kv_last_page_len_v2 = torch.full(
        (batch_size,), (kv_len_v2 - 1) % page_size + 1, dtype=torch.int32, device="cuda"
    )

    offsets = [0]
    for l in item_lens:
        offsets.append(offsets[-1] + l)
    item_offsets_tensor = torch.tensor(offsets, dtype=torch.uint32).cuda()

    workspace_v2 = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
    wrapper_v2 = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(workspace_v2, kv_layout)
    wrapper_v2.plan(
        q_indptr_v2, kv_indptr_v2, kv_indices_v2, kv_last_page_len_v2,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        causal=True,
        pos_encoding_mode="NONE",
        logits_soft_cap=logits_soft_cap,
        prefix_len_ptr=torch.tensor([prefix_len], dtype=torch.uint32).cuda(),
        item_offsets=item_offsets_tensor,
        item_offsets_len=len(offsets),
    )
    o_v2 = wrapper_v2.run(q_items, kv_data_v2)

    # ---- V1 ----
    # V1: prefix lives ONLY in KV cache (not in Q). Q = [delim + items] per item.
    # kv_len = prefix_len (cache) + qo_len (Q tokens mirrored into KV)
    qo_len_v1 = sum(1 + l for l in item_lens)  # delimiter per item + item tokens
    kv_len_v1 = prefix_len + qo_len_v1

    # Q: [delimiter (random) + shared item tokens] per item — NO prefix tokens in Q
    q_delims = torch.randn(num_items, num_qo_heads, head_dim, dtype=torch.float16, device="cuda")
    q_parts = []
    item_offset = 0
    for i, l in enumerate(item_lens):
        q_parts.append(q_delims[i : i + 1])
        q_parts.append(q_items[item_offset : item_offset + l])
        item_offset += l
    q_v1 = torch.cat(q_parts, dim=0)  # shape [qo_len_v1, ...]

    # KV: [prefix_kv || (delimiter_kv (random) + item_kv) per item]
    # Delimiter KV is random — masked to -inf by V1's mask logic, doesn't affect output.
    k_delims = torch.randn(num_items, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
    v_delims = torch.randn(num_items, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
    k_parts = [prefix_k]
    v_parts = [prefix_v]
    item_offset = 0
    for i, l in enumerate(item_lens):
        k_parts.append(k_delims[i : i + 1])
        v_parts.append(v_delims[i : i + 1])
        k_parts.append(item_k[item_offset : item_offset + l])
        v_parts.append(item_v[item_offset : item_offset + l])
        item_offset += l
    k_flat_v1 = torch.cat(k_parts, dim=0)  # shape [kv_len_v1, ...]
    v_flat_v1 = torch.cat(v_parts, dim=0)
    kv_data_v1 = pack_flat_kv_to_pages(k_flat_v1, v_flat_v1, kv_len_v1, page_size)

    num_pages_v1 = (kv_len_v1 + page_size - 1) // page_size
    q_indptr_v1 = (torch.arange(batch_size + 1) * qo_len_v1).int().cuda()
    kv_indptr_v1 = (torch.arange(batch_size + 1) * num_pages_v1).int().cuda()
    kv_indices_v1 = torch.arange(num_pages_v1 * batch_size).int().cuda()
    kv_last_page_len_v1 = torch.full(
        (batch_size,), (kv_len_v1 - 1) % page_size + 1, dtype=torch.int32, device="cuda"
    )

    # token_pos_in_items: 0=delimiter, 1..l=item tokens (for each item)
    token_pos = []
    for l in item_lens:
        token_pos.append(0)
        for j in range(1, l + 1):
            token_pos.append(j)
    token_pos_tensor = torch.tensor(token_pos, dtype=torch.uint16).cuda()
    max_item_len = max(item_lens)

    workspace_v1 = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
    wrapper_v1 = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(workspace_v1, kv_layout)
    wrapper_v1.plan(
        q_indptr_v1, kv_indptr_v1, kv_indices_v1, kv_last_page_len_v1,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        causal=True,
        pos_encoding_mode="NONE",
        logits_soft_cap=logits_soft_cap,
        prefix_len_ptr=torch.tensor([prefix_len], dtype=torch.uint32).cuda(),
        token_pos_in_items_ptr=token_pos_tensor,
        token_pos_in_items_len=len(token_pos),
        max_item_len_ptr=torch.tensor([max_item_len], dtype=torch.uint16).cuda(),
    )
    o_v1 = wrapper_v1.run(q_v1, kv_data_v1)

    # Extract item-token outputs from V1 (skip delimiter at start of each item)
    v1_item_indices = []
    pos = 0
    for l in item_lens:
        pos += 1  # skip delimiter
        for _ in range(l):
            v1_item_indices.append(pos)
            pos += 1
    v1_item_indices_t = torch.tensor(v1_item_indices, device="cuda")
    o_v1_items = o_v1[v1_item_indices_t]

    torch.testing.assert_close(o_v2, o_v1_items, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_batch_prefill_with_paged_kv_cache(
        12, 54, 37, 1, 4, 4, 128, False, "NHD", "NONE", False, 0.0, True, True
    )
    exit()
    test_batch_prefill_with_paged_kv_cache(
        12, 54, 37, 16, 8, 8, 128, True, "HND", "NONE", True, 0.0, False, True
    )
    test_batch_prefill_with_tuple_paged_kv_cache(
        12, 54, 37, 16, 8, 8, 128, True, "HND", "NONE", True, 0.0, False, True
    )
    test_batch_prefill_with_paged_kv_cache(
        12, 54, 37, 1, 8, 8, 128, True, "HND", "NONE", False, 0.0, False, True
    )
    test_batch_prefill_with_paged_kv_cache_custom_mask(
        1, 137, 137, 1, 8, 8, 128, "HND", "NONE", 0.0, False, True
    )
    test_batch_prefill_with_ragged_kv_cache(
        12, 54, 37, 8, 8, 128, True, "NONE", 0.0, False
    )
    test_batch_prefill_with_ragged_kv_cache_custom_mask(
        1, 137, 137, 8, 8, 128, "NONE", 0.0, False
    )
