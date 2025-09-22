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
from functools import partial
import flashinfer

MAX_BATCH_SIZE = 128


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
def test_batch_decode_with_paged_kv_cache_with_fast_plan(
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

    global global_override_indptr_cpu
    if global_override_indptr_cpu is None:
        global_override_indptr_cpu = torch.empty(MAX_BATCH_SIZE + 1, device="cpu")

    global_override_indptr_cpu = (
        torch.arange(0, batch_size + 1).int() * num_pages_per_seq
    )

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
    wrapper.plan = partial(flashinfer.fast_decode_plan, wrapper)
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
        non_blocking=True,
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


if __name__ == "__main__":
    test_batch_decode_with_paged_kv_cache_with_fast_plan(
        12,
        54,
        8,
        4,
        4,
        128,
        "NHD",
        "NONE",
        0.0,
        True,
        torch.float16,
        torch.float8_e4m3fn,
        True,
    )
