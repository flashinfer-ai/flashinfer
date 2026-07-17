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
from tests.test_helpers.test_helpers import assert_close_chunked
from tests.test_helpers.utils_fp4 import create_nvfp4_kv, nvfp4_to_float
from flashinfer.utils import get_compute_capability, has_flashinfer_jit_cache


def head_dim_512_supported() -> bool:
    # 16-bit FA2 head_dim > 256 uses the Ampere+ large-head path.
    return get_compute_capability(torch.device("cuda:0"))[0] >= 8


def skip_if_head_dim_unsupported(head_dim: int):
    if head_dim > 256 and not head_dim_512_supported():
        pytest.skip("16-bit FA2 head_dim > 256 is only supported on SM80 or newer")


def skip_if_nvfp4_asymmetric_unsupported(head_dim_qk: int):
    skip_if_head_dim_unsupported(head_dim_qk)
    if get_compute_capability(torch.device("cuda:0"))[0] < 10:
        pytest.skip(
            "asymmetric NVFP4 KV prefill uses the NVFP4 KV quantization kernel, "
            "which requires SM100 or newer"
        )


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
@pytest.mark.parametrize("head_dim", [64, 128, 256])
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


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
def test_batch_prefill_with_paged_kv_cache_head_dim_512(
    causal,
    pos_encoding_mode,
):
    head_dim = 512
    skip_if_head_dim_unsupported(head_dim)

    batch_size = 2
    kv_len = 97
    qo_len = 17
    page_size = 16
    num_kv_heads = 4
    num_qo_heads = 4
    kv_layout = "NHD"

    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    q_indptr_cpu = torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]
    kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
    kv_data = kv_data_fp32.half()
    kv_indptr_cpu = (
        torch.arange(0, batch_size + 1, dtype=torch.int32) * num_pages_per_seq
    )
    kv_indices_cpu = torch.arange(0, total_num_pages, dtype=torch.int32)
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )

    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
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
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    o, lse = wrapper.run(q, kv_data, return_lse=True)

    o_buffer = torch.empty_like(o)
    lse_buffer = torch.empty_like(lse)
    wrapper.run(q, kv_data, out=o_buffer, lse=lse_buffer, return_lse=True)
    torch.testing.assert_close(o, o_buffer, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lse, lse_buffer, rtol=1e-3, atol=1e-3)

    for i in range(batch_size):
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        ki = torch.cat(
            [
                kv_data_fp32[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 0].reshape(
                    -1, num_kv_heads, head_dim
                ),
                kv_data_fp32[
                    kv_indptr_cpu[i + 1] - 1, 0, : kv_last_page_len_cpu[i], :
                ].reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).half()
        vi = torch.cat(
            [
                kv_data_fp32[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 1].reshape(
                    -1, num_kv_heads, head_dim
                ),
                kv_data_fp32[
                    kv_indptr_cpu[i + 1] - 1, 1, : kv_last_page_len_cpu[i], :
                ].reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).half()
        o_ref_i = flashinfer.prefill.single_prefill_with_kv_cache(
            qi,
            ki,
            vi,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            backend="fa2",
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
@pytest.mark.parametrize("head_dim", [64, 128, 256])
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


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
def test_batch_prefill_with_ragged_kv_cache_head_dim_512(
    causal,
    pos_encoding_mode,
):
    head_dim = 512
    skip_if_head_dim_unsupported(head_dim)

    batch_size = 2
    kv_len = 97
    qo_len = 17
    num_kv_heads = 4
    num_qo_heads = 4
    kv_layout = "NHD"

    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
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
    q_indptr_cpu = torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len
    kv_indptr_cpu = torch.arange(0, batch_size + 1, dtype=torch.int32) * kv_len

    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    wrapper.plan(
        q_indptr_cpu.to(0),
        kv_indptr_cpu.to(0),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    o, lse = wrapper.run(q, k, v, return_lse=True)

    o_buffer = torch.empty_like(o)
    lse_buffer = torch.empty_like(lse)
    wrapper.run(q, k, v, out=o_buffer, lse=lse_buffer, return_lse=True)
    torch.testing.assert_close(o, o_buffer, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lse, lse_buffer, rtol=1e-3, atol=1e-3)

    for i in range(batch_size):
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        ki = k[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1]]
        vi = v[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1]]
        o_ref_i = flashinfer.prefill.single_prefill_with_kv_cache(
            qi,
            ki,
            vi,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            backend="fa2",
        )
        o_i = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
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
    # Free everything but the two outputs before comparing: for the largest
    # parametrizations o_custom/o_causal are ~1.1 GiB each and
    # torch.testing.assert_close allocates several full-size temporaries
    # inside torch.isclose, which OOMs 24 GB CI GPUs (issue #3603).
    del q, k, v, custom_mask, wrapper, workspace_buffer
    assert_close_chunked(o_custom, o_causal, rtol=1e-3, atol=1e-3)


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


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("kv_len", [128, 256])
@pytest.mark.parametrize("qo_len", [64, 128])
@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("num_kv_heads", [1])
@pytest.mark.parametrize("num_qo_heads", [1])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
def test_batch_prefill_with_paged_kv_cache_nvfp4(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    q_dtype,
    pos_encoding_mode="NONE",
):
    """Test BatchPrefillWithPagedKVCacheWrapper with NVFP4 KV cache.

    KV cache layout (NHD):
      kv_cache:    [num_pages, 2, page_size, num_kv_heads, head_dim//2]   uint8 (packed FP4x2)
      kv_cache_sf: [num_pages, 2, page_size, num_kv_heads, head_dim//16]  uint8 (FP8 SFs)

    Reference is computed by dequantizing the packed KV back to q_dtype and running
    single_prefill_with_kv_cache per batch item.
    """
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")

    kv_layout = "NHD"
    torch.manual_seed(42)

    # --- query ---
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=q_dtype
    )
    q_indptr_cpu = torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len

    # --- paged KV metadata ---
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_indptr_cpu = (
        torch.arange(0, batch_size + 1, dtype=torch.int32) * num_pages_per_seq
    )
    kv_indices_cpu = torch.arange(0, total_num_pages, dtype=torch.int32)
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )

    # --- create NVFP4 KV pages directly (NHD: [num_pages, page_size, num_kv_heads, head_dim//2]) ---
    kv_shape = (total_num_pages, page_size, num_kv_heads, head_dim // 2)
    k_packed, k_sf, k_global_scale = create_nvfp4_kv(kv_shape, "cuda:0")
    v_packed, v_sf, v_global_scale = create_nvfp4_kv(kv_shape, "cuda:0")

    # Dequantize for reference attention
    k_dq = nvfp4_to_float(k_packed, k_sf, k_global_scale).to(q_dtype)
    v_dq = nvfp4_to_float(v_packed, v_sf, v_global_scale).to(q_dtype)

    # Pack into combined tensors:
    #   kv_cache:    [num_pages, 2, page_size, num_kv_heads, head_dim//2]
    #   kv_cache_sf: [num_pages, 2, page_size, num_kv_heads, head_dim//16]
    kv_cache = torch.stack([k_packed, v_packed], dim=1)  # [P, 2, S, H, D//2]
    kv_cache_sf = torch.stack([k_sf, v_sf], dim=1)  # [P, 2, S, H, D//16]

    # --- run BatchPrefillWithPagedKVCacheWrapper ---
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    q_indptr_gpu = q_indptr_cpu.to("cuda:0")
    kv_indptr_gpu = kv_indptr_cpu.to("cuda:0")
    kv_indices_gpu = kv_indices_cpu.to("cuda:0")
    kv_last_page_len_gpu = kv_last_page_len_cpu.to("cuda:0")

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
        logits_soft_cap=0.0,
        kv_data_type=torch.uint8,
        q_data_type=q_dtype,
    )
    o = wrapper.run(
        q,
        kv_cache,
        k_scale=k_global_scale.item(),
        v_scale=v_global_scale.item(),
        kv_cache_sf=kv_cache_sf,
    )

    # --- reference: single_prefill_with_kv_cache per batch item using dequantized KV ---
    for i in range(batch_size):
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]

        # Gather full (non-padded) KV for sequence i from pages
        full_pages_k = k_dq[
            kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1
        ]  # [p-1, S, H, D]
        last_page_k = k_dq[
            kv_indptr_cpu[i + 1] - 1, : kv_last_page_len_cpu[i]
        ]  # [l, H, D]
        ki = torch.cat(
            [
                full_pages_k.reshape(-1, num_kv_heads, head_dim),
                last_page_k.reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        )

        full_pages_v = v_dq[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1]
        last_page_v = v_dq[kv_indptr_cpu[i + 1] - 1, : kv_last_page_len_cpu[i]]
        vi = torch.cat(
            [
                full_pages_v.reshape(-1, num_kv_heads, head_dim),
                last_page_v.reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        )

        o_ref_i = flashinfer.prefill.single_prefill_with_kv_cache(
            qi,
            ki,
            vi,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=0.0,
        )
        o_i = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]

        # NVFP4 is 4-bit; use relaxed tolerance
        torch.testing.assert_close(o_i, o_ref_i, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
def test_batch_prefill_with_paged_kv_cache_nvfp4_strided_scale_views(kv_layout):
    """NVFP4 scale tensors may be strided views sharing a packed KV parent."""
    torch.manual_seed(42)
    batch_size = 2
    kv_len = 33
    qo_len = 17
    page_size = 16
    num_kv_heads = 2
    num_qo_heads = 4
    head_dim = 128
    q_dtype = torch.float16

    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=q_dtype,
    )
    q_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda:0"
    )
    custom_mask = torch.tril(
        torch.full((batch_size, qo_len, kv_len), True, device="cuda:0"),
        diagonal=(kv_len - qo_len),
    ).reshape(-1)

    kv_shape = (total_num_pages, page_size, num_kv_heads, head_dim // 2)
    k_packed, k_sf, k_global_scale = create_nvfp4_kv(kv_shape, "cuda:0")
    v_packed, v_sf, v_global_scale = create_nvfp4_kv(kv_shape, "cuda:0")

    if kv_layout == "NHD":
        compact_k = k_packed
        compact_v = v_packed
        compact_k_sf = k_sf
        compact_v_sf = v_sf
        packed_parent = torch.empty(
            (
                total_num_pages,
                2,
                page_size,
                num_kv_heads,
                head_dim // 2 + head_dim // 16,
            ),
            dtype=torch.uint8,
            device="cuda:0",
        )
        packed_parent[:, 0, :, :, : head_dim // 2].copy_(compact_k)
        packed_parent[:, 0, :, :, head_dim // 2 :].copy_(compact_k_sf)
        packed_parent[:, 1, :, :, : head_dim // 2].copy_(compact_v)
        packed_parent[:, 1, :, :, head_dim // 2 :].copy_(compact_v_sf)
        strided_k = packed_parent[:, 0, :, :, : head_dim // 2]
        strided_k_sf = packed_parent[:, 0, :, :, head_dim // 2 :]
        strided_v = packed_parent[:, 1, :, :, : head_dim // 2]
        strided_v_sf = packed_parent[:, 1, :, :, head_dim // 2 :]
    else:
        compact_k = k_packed.permute(0, 2, 1, 3).contiguous()
        compact_v = v_packed.permute(0, 2, 1, 3).contiguous()
        compact_k_sf = k_sf.permute(0, 2, 1, 3).contiguous()
        compact_v_sf = v_sf.permute(0, 2, 1, 3).contiguous()
        packed_parent = torch.empty(
            (
                total_num_pages,
                2,
                num_kv_heads,
                page_size,
                head_dim // 2 + head_dim // 16,
            ),
            dtype=torch.uint8,
            device="cuda:0",
        )
        packed_parent[:, 0, :, :, : head_dim // 2].copy_(compact_k)
        packed_parent[:, 0, :, :, head_dim // 2 :].copy_(compact_k_sf)
        packed_parent[:, 1, :, :, : head_dim // 2].copy_(compact_v)
        packed_parent[:, 1, :, :, head_dim // 2 :].copy_(compact_v_sf)
        strided_k = packed_parent[:, 0, :, :, : head_dim // 2]
        strided_k_sf = packed_parent[:, 0, :, :, head_dim // 2 :]
        strided_v = packed_parent[:, 1, :, :, : head_dim // 2]
        strided_v_sf = packed_parent[:, 1, :, :, head_dim // 2 :]

    assert strided_k.stride(0) == strided_k_sf.stride(0)
    assert strided_v.stride(0) == strided_v_sf.stride(0)

    compact_cache = torch.stack([compact_k, compact_v], dim=1)
    compact_cache_sf = torch.stack([compact_k_sf, compact_v_sf], dim=1)
    strided_cache = (strided_k, strided_v)
    strided_cache_sf = (strided_k_sf, strided_v_sf)

    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
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
        pos_encoding_mode="NONE",
        logits_soft_cap=0.0,
        kv_data_type=torch.uint8,
        q_data_type=q_dtype,
    )
    compact_out = wrapper.run(
        q,
        compact_cache,
        k_scale=k_global_scale.item(),
        v_scale=v_global_scale.item(),
        kv_cache_sf=compact_cache_sf,
    )
    strided_out = wrapper.run(
        q,
        strided_cache,
        k_scale=k_global_scale.item(),
        v_scale=v_global_scale.item(),
        kv_cache_sf=strided_cache_sf,
    )

    assert torch.isfinite(strided_out).all()
    torch.testing.assert_close(strided_out, compact_out, rtol=1e-3, atol=1e-3)

    bad_k_sf_parent = torch.empty(
        *strided_k_sf.shape[:-1],
        strided_k_sf.shape[-1] * 2,
        dtype=torch.uint8,
        device="cuda:0",
    )
    bad_v_sf_parent = torch.empty(
        *strided_v_sf.shape[:-1],
        strided_v_sf.shape[-1] * 2,
        dtype=torch.uint8,
        device="cuda:0",
    )
    bad_k_sf = bad_k_sf_parent[..., ::2]
    bad_v_sf = bad_v_sf_parent[..., ::2]
    assert bad_k_sf.shape == strided_k_sf.shape
    assert bad_v_sf.shape == strided_v_sf.shape
    assert bad_k_sf.stride(-1) == 2
    assert bad_v_sf.stride(-1) == 2
    with pytest.raises(Exception, match="innermost stride must be 1"):
        wrapper.run(
            q,
            strided_cache,
            k_scale=k_global_scale.item(),
            v_scale=v_global_scale.item(),
            kv_cache_sf=(bad_k_sf, bad_v_sf),
        )


@pytest.mark.parametrize("head_dim_qk,head_dim_vo", [(512, 256), (256, 128)])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("num_kv_heads", [2, 8])
@pytest.mark.parametrize("causal", [True])
def test_batch_prefill_with_paged_kv_cache_nvfp4_asymmetric(
    head_dim_qk,
    head_dim_vo,
    page_size,
    num_kv_heads,
    causal,
):
    """Asymmetric (head_dim_qk != head_dim_vo) NVFP4 paged prefill correctness.

    K pages are ``[.., head_dim_qk // 2]`` and V pages ``[.., head_dim_vo // 2]``,
    so the separately allocated K and V pools (and their scale-factor tensors)
    have genuinely different stride families — the layout an asymmetric NVFP4
    KV cache hands the FA2 paged prefill entry point.

    bf16 K/V are quantized with the in-tree NVFP4 KV quantization kernel and
    the FA2 output is checked against a float32 reference attention computed on
    ``nvfp4_kv_dequantize_paged`` output: kernel and reference consume the exact
    same quantized bytes, so the reference is a dequantization oracle rather
    than a requantized approximation.
    """
    skip_if_nvfp4_asymmetric_unsupported(head_dim_qk)

    kv_layout = "NHD"
    torch.manual_seed(42)
    batch_size = 2
    kv_len = 99
    qo_len = 33
    num_qo_heads = 2 * num_kv_heads
    q_dtype = torch.bfloat16

    # --- query ---
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim_qk, device="cuda:0", dtype=q_dtype
    )
    q_indptr_cpu = torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len

    # --- paged KV metadata ---
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_indptr_cpu = (
        torch.arange(0, batch_size + 1, dtype=torch.int32) * num_pages_per_seq
    )
    kv_indices_cpu = torch.arange(0, total_num_pages, dtype=torch.int32)
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )

    # --- bf16 source K/V, quantized via the in-tree NVFP4 KV quantization
    # kernel (the helper tests/utils/test_fp4_kv_quantization.py exercises).
    # It quantizes row-wise over the last dim, so asymmetric K/V widths
    # quantize naturally. ---
    k_bf16 = torch.randn(
        total_num_pages,
        page_size,
        num_kv_heads,
        head_dim_qk,
        device="cuda:0",
        dtype=q_dtype,
    )
    v_bf16 = torch.randn(
        total_num_pages,
        page_size,
        num_kv_heads,
        head_dim_vo,
        device="cuda:0",
        dtype=q_dtype,
    )
    # global_scale=1.0 avoids FP8 E4M3 block-scale underflow (see
    # test_nvfp4_kv_roundtrip).
    k_global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda:0")
    v_global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda:0")
    k_packed, k_sf = flashinfer.nvfp4_kv_quantize(
        k_bf16.reshape(-1, head_dim_qk), k_global_scale
    )
    v_packed, v_sf = flashinfer.nvfp4_kv_quantize(
        v_bf16.reshape(-1, head_dim_vo), v_global_scale
    )
    k_packed = k_packed.reshape(
        total_num_pages, page_size, num_kv_heads, head_dim_qk // 2
    )
    k_sf = k_sf.reshape(total_num_pages, page_size, num_kv_heads, head_dim_qk // 16)
    v_packed = v_packed.reshape(
        total_num_pages, page_size, num_kv_heads, head_dim_vo // 2
    )
    v_sf = v_sf.reshape(total_num_pages, page_size, num_kv_heads, head_dim_vo // 16)

    # The whole point: every consumer reachable from this entry point must
    # support (or explicitly reject) unequal K/V strides.
    assert k_packed.stride() != v_packed.stride()
    assert k_sf.stride() != v_sf.stride()

    # --- run BatchPrefillWithPagedKVCacheWrapper (FA2 NVFP4 paged path) ---
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper.plan(
        q_indptr_cpu.to("cuda:0"),
        kv_indptr_cpu.to("cuda:0"),
        kv_indices_cpu.to("cuda:0"),
        kv_last_page_len_cpu.to("cuda:0"),
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        page_size,
        head_dim_vo=head_dim_vo,
        causal=causal,
        pos_encoding_mode="NONE",
        logits_soft_cap=0.0,
        kv_data_type=torch.uint8,
        q_data_type=q_dtype,
    )
    o = wrapper.run(
        q,
        (k_packed, v_packed),
        k_scale=k_global_scale.item(),
        v_scale=v_global_scale.item(),
        kv_cache_sf=(k_sf, v_sf),
    )
    assert o.shape == (batch_size * qo_len, num_qo_heads, head_dim_vo)
    assert torch.isfinite(o).all()

    # --- dequantization oracle: #3748's paged NVFP4 dequant kernel ---
    block_tables = (
        kv_indices_cpu.to("cuda:0").reshape(batch_size, num_pages_per_seq).contiguous()
    )
    seq_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device="cuda:0")
    k_dq = torch.zeros(
        batch_size, kv_len, num_kv_heads, head_dim_qk, dtype=q_dtype, device="cuda:0"
    )
    v_dq = torch.zeros(
        batch_size, kv_len, num_kv_heads, head_dim_vo, dtype=q_dtype, device="cuda:0"
    )
    flashinfer.nvfp4_kv_dequantize_paged(
        (k_packed, v_packed),
        (k_sf.view(torch.float8_e4m3fn), v_sf.view(torch.float8_e4m3fn)),
        block_tables,
        seq_lens,
        k_global_scale,
        v_global_scale,
        k_dq,
        v_dq,
        kv_layout=kv_layout,
    )

    # --- float32 reference attention on the dequantized K/V ---
    group_size = num_qo_heads // num_kv_heads
    sm_scale = head_dim_qk**-0.5
    for i in range(batch_size):
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]].float()  # [qo, Hq, dqk]
        ki = k_dq[i].float().repeat_interleave(group_size, dim=1)  # [kv, Hq, dqk]
        vi = v_dq[i].float().repeat_interleave(group_size, dim=1)  # [kv, Hq, dvo]

        logits = torch.einsum("qhd,khd->hqk", qi, ki) * sm_scale
        if causal:
            qpos = torch.arange(qo_len, device="cuda:0").unsqueeze(1)
            kpos = torch.arange(kv_len, device="cuda:0").unsqueeze(0)
            allowed = kpos <= qpos + (kv_len - qo_len)
            logits = logits.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        o_ref_i = torch.einsum("hqk,khd->qhd", torch.softmax(logits, dim=-1), vi)
        o_i = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]].float()

        # NVFP4 is 4-bit; use relaxed tolerance
        torch.testing.assert_close(o_i, o_ref_i, rtol=1e-1, atol=1e-1)


_PLAN_INFO_CTA_TILE_Q_IDX = 3  # PrefillPlanInfo::ToVector layout (scheduler.cuh)


@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.float8_e4m3fn])
def test_batch_prefill_paged_cta_tile_q_smem_probe_qk448_vo256(kv_dtype):
    """Pin the FA2DetermineCtaTileQ shared-memory probe at unvalidated head
    dims: neither ``plan()`` nor the JIT path validates head dims, so under
    ``pos_encoding_mode="NONE"`` a config like (qk, vo) = (448, 256) reaches
    the probe's short-q branch today.

    At 2-byte KV the 1x4-layout cost is 16*448*2 + (448+256)*16*4*2 = 104448
    bytes: on 99KB-opt-in parts (SM86/89/120/121) the probe must fire and fall
    back to CTA_TILE_Q=64 -- which keeps the config dispatchable where the
    CTA16 dispatch would exceed the per-block limit -- while on larger-smem
    parts (e.g. SM90) CTA16 is kept. At 1-byte KV the true cost is 59392
    bytes, so the probe selects CTA16 everywhere, pinning that the
    kv_dtype_bytes accuracy changes tile selection at such dims (the previous
    2-byte assumption forced CTA64 on 99KB parts).

    The expected tile is computed from the device's actual per-block opt-in
    limit, so the assertion is exact on every architecture. The 2-byte case
    then runs the kernel against an exact float32 reference. The 1-byte case
    stops at the plan-level assertion: the FA2 1-byte KV producers require
    head_dim to be a multiple of 128 elements (the 128-bit-per-lane load loop
    steps NUM_MMA_D by 8, and the k128B swizzle needs an 8-aligned upcast
    stride), a pre-existing constraint -- so no currently-runnable 1-byte
    config reaches the flipped CTA64->CTA16 region, and the pin locks the
    documented planner behavior for when one does.
    """
    head_dim_qk = 448
    head_dim_vo = 256
    skip_if_head_dim_unsupported(head_dim_qk)
    props = torch.cuda.get_device_properties(0)
    optin = getattr(props, "shared_memory_per_block_optin", None)
    if optin is None:
        pytest.skip("torch does not expose shared_memory_per_block_optin")

    torch.manual_seed(42)
    batch_size = 2
    qo_len = 8  # group_size 1 below -> avg_packed_qo_len = 8 <= 16: probe branch
    kv_len = 65
    page_size = 16
    num_kv_heads = 2
    num_qo_heads = 2

    # Mirror FA2DetermineCtaTileQ's accounting exactly (utils.cuh).
    kv_dtype_bytes = 1 if kv_dtype == torch.float8_e4m3fn else 2
    q_tile_smem = 16 * head_dim_qk * 2
    kv_step_smem_1x4 = (head_dim_qk + head_dim_vo) * 16 * 4 * kv_dtype_bytes
    expected_cta_tile_q = 64 if q_tile_smem + kv_step_smem_1x4 > optin else 16

    q_indptr_cpu = torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_indptr_cpu = (
        torch.arange(0, batch_size + 1, dtype=torch.int32) * num_pages_per_seq
    )
    kv_indices_cpu = torch.arange(0, total_num_pages, dtype=torch.int32)
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", backend="fa2"
    )
    wrapper.plan(
        q_indptr_cpu.to("cuda:0"),
        kv_indptr_cpu.to("cuda:0"),
        kv_indices_cpu.to("cuda:0"),
        kv_last_page_len_cpu.to("cuda:0"),
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        page_size,
        head_dim_vo=head_dim_vo,
        causal=False,
        pos_encoding_mode="NONE",
        q_data_type=torch.float16,
        kv_data_type=kv_dtype,
    )
    assert wrapper._plan_info[_PLAN_INFO_CTA_TILE_Q_IDX] == expected_cta_tile_q

    if kv_dtype != torch.float16:
        # Plan-level pin only; see docstring for why (448, 256) is not
        # runnable at 1-byte KV today.
        return

    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim_qk,
        device="cuda:0",
        dtype=torch.float16,
    )
    k = torch.randn(
        total_num_pages,
        page_size,
        num_kv_heads,
        head_dim_qk,
        device="cuda:0",
        dtype=torch.float16,
    )
    v = torch.randn(
        total_num_pages,
        page_size,
        num_kv_heads,
        head_dim_vo,
        device="cuda:0",
        dtype=torch.float16,
    )
    o = wrapper.run(q, (k, v))
    assert o.shape == (batch_size * qo_len, num_qo_heads, head_dim_vo)

    # Exact float32 reference over the same logical KV.
    sm_scale = head_dim_qk**-0.5
    for i in range(batch_size):
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]].float()
        ki = (
            k[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1]]
            .reshape(-1, num_kv_heads, head_dim_qk)[:kv_len]
            .float()
        )
        vi = (
            v[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1]]
            .reshape(-1, num_kv_heads, head_dim_vo)[:kv_len]
            .float()
        )
        logits = torch.einsum("qhd,khd->hqk", qi, ki) * sm_scale
        o_ref_i = torch.einsum("hqk,khd->qhd", torch.softmax(logits, dim=-1), vi)
        o_i = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]].float()
        torch.testing.assert_close(o_i, o_ref_i, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("kv_len", [128, 256])
@pytest.mark.parametrize("qo_len", [64, 128])
@pytest.mark.parametrize("num_kv_heads", [1])
@pytest.mark.parametrize("num_qo_heads", [1])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
def test_batch_prefill_with_ragged_kv_cache_nvfp4(
    batch_size,
    kv_len,
    qo_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    q_dtype,
    pos_encoding_mode="NONE",
):
    """Test BatchPrefillWithRaggedKVCacheWrapper with NVFP4 KV cache.

    KV cache layout (NHD):
      k/v:    [total_kv_tokens, num_kv_heads, head_dim//2]   uint8 (packed FP4x2)
      k/v_sf: [total_kv_tokens, num_kv_heads, head_dim//16]  uint8 (FP8 SFs)

    Reference is computed by dequantizing the packed KV back to q_dtype and running
    single_prefill_with_kv_cache per batch item.
    """
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")

    kv_layout = "NHD"
    torch.manual_seed(42)

    # --- query ---
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=q_dtype
    )
    q_indptr_cpu = torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len
    q_indptr_gpu = q_indptr_cpu.to("cuda:0")

    # --- ragged KV metadata ---
    kv_indptr_cpu = torch.arange(0, batch_size + 1, dtype=torch.int32) * kv_len
    kv_indptr_gpu = kv_indptr_cpu.to("cuda:0")
    total_kv_tokens = batch_size * kv_len

    # --- create NVFP4 ragged KV (NHD: [total_kv_tokens, num_kv_heads, head_dim//2]) ---
    kv_shape = (total_kv_tokens, num_kv_heads, head_dim // 2)
    k_packed, k_sf, k_global_scale = create_nvfp4_kv(kv_shape, "cuda:0")
    v_packed, v_sf, v_global_scale = create_nvfp4_kv(kv_shape, "cuda:0")

    # Dequantize for reference attention
    k_dq = nvfp4_to_float(k_packed, k_sf, k_global_scale).to(q_dtype)
    v_dq = nvfp4_to_float(v_packed, v_sf, v_global_scale).to(q_dtype)

    # --- run BatchPrefillWithRaggedKVCacheWrapper ---
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper.plan(
        q_indptr_gpu,
        kv_indptr_gpu,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=0.0,
        kv_data_type=torch.uint8,
        q_data_type=q_dtype,
    )
    o = wrapper.run(
        q,
        k_packed,
        v_packed,
        k_scale=k_global_scale.item(),
        v_scale=v_global_scale.item(),
        kv_cache_sf=(k_sf, v_sf),
    )

    # --- reference: single_prefill_with_kv_cache per batch item using dequantized KV ---
    for i in range(batch_size):
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        ki = k_dq[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1]]
        vi = v_dq[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1]]

        o_ref_i = flashinfer.prefill.single_prefill_with_kv_cache(
            qi,
            ki,
            vi,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=0.0,
        )
        o_i = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]

        # NVFP4 is 4-bit; use relaxed tolerance
        torch.testing.assert_close(o_i, o_ref_i, rtol=1e-1, atol=1e-1)


def test_batch_prefill_with_paged_kv_cache_nvfp4_large_head():
    skip_if_head_dim_unsupported(512)
    test_batch_prefill_with_paged_kv_cache_nvfp4(
        batch_size=1,
        kv_len=128,
        qo_len=64,
        page_size=16,
        num_kv_heads=1,
        num_qo_heads=1,
        head_dim=512,
        causal=False,
        q_dtype=torch.float16,
    )


def test_batch_prefill_with_paged_kv_cache_nvfp4_large_head_bf16():
    skip_if_head_dim_unsupported(512)
    test_batch_prefill_with_paged_kv_cache_nvfp4(
        batch_size=1,
        kv_len=128,
        qo_len=64,
        page_size=16,
        num_kv_heads=1,
        num_qo_heads=1,
        head_dim=512,
        causal=False,
        q_dtype=torch.bfloat16,
    )


def test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head():
    skip_if_head_dim_unsupported(512)
    test_batch_prefill_with_paged_kv_cache_nvfp4(
        batch_size=1,
        kv_len=128,
        qo_len=64,
        page_size=16,
        num_kv_heads=1,
        num_qo_heads=1,
        head_dim=512,
        causal=False,
        q_dtype=torch.float16,
        pos_encoding_mode="ROPE_LLAMA",
    )


def test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head_bf16():
    skip_if_head_dim_unsupported(512)
    test_batch_prefill_with_paged_kv_cache_nvfp4(
        batch_size=1,
        kv_len=128,
        qo_len=64,
        page_size=16,
        num_kv_heads=1,
        num_qo_heads=1,
        head_dim=512,
        causal=False,
        q_dtype=torch.bfloat16,
        pos_encoding_mode="ROPE_LLAMA",
    )


def test_batch_prefill_with_ragged_kv_cache_nvfp4_large_head():
    skip_if_head_dim_unsupported(512)
    test_batch_prefill_with_ragged_kv_cache_nvfp4(
        batch_size=1,
        kv_len=128,
        qo_len=64,
        num_kv_heads=1,
        num_qo_heads=1,
        head_dim=512,
        causal=False,
        q_dtype=torch.float16,
    )


def test_batch_prefill_with_ragged_kv_cache_nvfp4_rope_large_head():
    skip_if_head_dim_unsupported(512)
    test_batch_prefill_with_ragged_kv_cache_nvfp4(
        batch_size=1,
        kv_len=128,
        qo_len=64,
        num_kv_heads=1,
        num_qo_heads=1,
        head_dim=512,
        causal=False,
        q_dtype=torch.float16,
        pos_encoding_mode="ROPE_LLAMA",
    )


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
    test_batch_prefill_with_paged_kv_cache_nvfp4(
        4, 128, 64, 64, 1, 1, 128, False, torch.float16
    )
    test_batch_prefill_with_ragged_kv_cache_nvfp4(
        4, 128, 64, 1, 1, 128, False, torch.float16
    )


def test_single_prefill_torch_compile_cuda_graph():
    """Issue #541: single_prefill_with_kv_cache with torch.compile(mode='reduce-overhead')
    should not raise RuntimeError about unaccounted cudagraph pool pointers.

    Runs as a subprocess because torch.library registration must happen at module
    level for torch.compile compatibility -- registering inside a pytest function
    causes dynamo tracing errors.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""\
        import torch
        from flashinfer import single_prefill_with_kv_cache

        torch.library.define(
            "test541p::op", "(Tensor q, Tensor k, Tensor v) -> Tensor")

        @torch.library.impl("test541p::op", "cuda")
        def _impl(q, k, v):
            return single_prefill_with_kv_cache(q, k, v, causal=True)

        @torch.library.register_fake("test541p::op")
        def _fake(q, k, v):
            return torch.empty_like(q)

        compiled_fn = torch.compile(
            lambda q, k, v: torch.ops.test541p.op(q, k, v),
            mode="reduce-overhead", fullgraph=True)

        S, QH, KH, D = 128, 8, 8, 128
        for i in range(3):
            q = torch.randn(S, QH, D, device="cuda", dtype=torch.float16)
            k = torch.randn(S, KH, D, device="cuda", dtype=torch.float16)
            v = torch.randn(S, KH, D, device="cuda", dtype=torch.float16)
            o = compiled_fn(q, k, v)
            assert o.shape == (S, QH, D)
        torch.cuda.synchronize()
        print("PASS")
    """)
    import gc
    import os

    # torch.compile's inductor calls getpass.getuser() for cache dir, which fails
    # in CI containers where the uid has no /etc/passwd entry. Setting USER avoids this.
    env = os.environ.copy()
    env.setdefault("USER", "ci")

    # The parent pytest process has already run thousands of prefill cases in this
    # file. Release its cached blocks before the subprocess initializes
    # torch.compile/cudagraph state on memory-constrained A10G runners.
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert result.returncode == 0 and "PASS" in result.stdout, (
        f"Test failed:\nstdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
    )
