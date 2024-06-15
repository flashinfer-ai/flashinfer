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

import numpy as np
import pytest
import torch

import flashinfer


@pytest.mark.parametrize("kv_len", [7, 19, 39, 1170, 39275])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128])  # [64, 128, 256])
@pytest.mark.parametrize("kv_layout", ["NHD"])  # ["HND", "NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE"])  # , "ROPE_LLAMA", "ALIBI"])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn])
def test_single_decode_fp8_calibration_scale(
    kv_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    pos_encoding_mode,
    fp8_dtype,
):
    torch.manual_seed(42)
    q = torch.randn(num_qo_heads, head_dim, dtype=torch.float16).to(0)
    k = (
        torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.float16).to(0)
        if kv_layout == "NHD"
        else torch.randn(num_kv_heads, kv_len, head_dim).to(0)
    )
    v = (
        0.1 * torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.float16).to(0)
        if kv_layout == "NHD"
        else 0.1 * torch.randn(num_kv_heads, kv_len, head_dim).to(0)
    )

    o_fp16 = flashinfer.single_decode_with_kv_cache(
        q, k, v, kv_layout=kv_layout, pos_encoding_mode=pos_encoding_mode
    )

    k_scale = k.amax().item() / 256
    v_scale = v.amax().item() / 256
    k_fp8 = (k / k_scale).to(fp8_dtype)
    v_fp8 = (v / v_scale).to(fp8_dtype)

    o_fp8 = flashinfer.single_decode_with_kv_cache(
        q,
        k_fp8,
        v_fp8,
        kv_layout=kv_layout,
        pos_encoding_mode=pos_encoding_mode,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    np.testing.assert_allclose(
        o_fp16.cpu().numpy(), o_fp8.cpu().numpy(), atol=1e-2, rtol=2e-2
    )


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("kv_len", [54, 97])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_batch_decode_with_paged_kv_cache_fp8_calibration_scale(
    batch_size,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    pos_encoding_mode,
    dtype,
):
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=torch.float16).to(0)
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = (
        0.1
        * torch.randn(
            total_num_pages, 2, num_kv_heads, page_size, head_dim, dtype=torch.float16
        ).to(0)
        if kv_layout == "HND"
        else 0.1
        * torch.randn(
            total_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16
        ).to(0)
    )
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * num_pages_per_seq
    kv_indices = torch.arange(0, total_num_pages).to(0).int()
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(0)

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout)
    wrapper.begin_forward(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        "NONE",
        data_type=torch.float16,
        q_data_type=torch.float16,
    )
    o_fp16 = wrapper.forward(q, kv_data, pos_encoding_mode=pos_encoding_mode)
    wrapper.end_forward()

    k_data, v_data = torch.chunk(kv_data, 2, dim=1)
    k_scale = k_data.amax().item() / 256
    v_scale = v_data.amax().item() / 256

    k_fp8 = (k_data / k_scale).to(dtype)
    v_fp8 = (v_data / v_scale).to(dtype)
    kv_data_fp8 = torch.cat([k_fp8, v_fp8], dim=1)

    wrapper.begin_forward(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        "NONE",
        data_type=dtype,
        q_data_type=torch.float16,
    )
    o_fp8 = wrapper.forward(
        q,
        kv_data_fp8.to(dtype),
        pos_encoding_mode=pos_encoding_mode,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    np.testing.assert_allclose(
        o_fp16.cpu().numpy(), o_fp8.cpu().numpy(), atol=1e-2, rtol=2e-1
    )


if __name__ == "__main__":
    test_single_decode_fp8_calibration_scale(
        1170, 4, 32, 128, "NHD", "NONE", torch.float8_e4m3fn
    )
    test_batch_decode_with_paged_kv_cache_fp8_calibration_scale(
        12, 54, 1, 4, 4, 128, "NHD", "NONE", torch.float8_e5m2
    )
