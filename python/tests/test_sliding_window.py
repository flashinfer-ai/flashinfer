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
"""

import numpy
import pytest
import torch
import flashinfer


@pytest.mark.parametrize("seq_len", [1, 3, 19, 99, 199, 1999])
@pytest.mark.parametrize("window_left", [3, 13, 23, 43])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_single_decode_sliding_window(
    seq_len, window_left, num_kv_heads, num_qo_heads, head_dim
):
    q = torch.randn(num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0")
    k = torch.randn(
        seq_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    )
    v = torch.randn(
        seq_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    )

    k_sliced = k[-(window_left + 1) :]
    v_sliced = v[-(window_left + 1) :]

    o_ref = flashinfer.single_decode_with_kv_cache(q, k_sliced, v_sliced)
    o = flashinfer.single_decode_with_kv_cache(q, k, v, window_left=window_left)

    numpy.testing.assert_allclose(o.cpu(), o_ref.cpu(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 3, 13, 32])
@pytest.mark.parametrize("kv_len", [1, 3, 99, 199, 1999])
@pytest.mark.parametrize("window_left", [33, 533])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("page_size", [1, 16])
def test_batch_decode_sliding_window(
    batch_size, kv_len, window_left, num_kv_heads, num_qo_heads, head_dim, page_size
):
    q = torch.randn(
        batch_size, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0"
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    k_data = torch.randn(
        total_num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    v_data = torch.randn(
        total_num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * num_pages_per_seq
    kv_indices = torch.arange(0, total_num_pages).to(0).int()
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(0)

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper.begin_forward(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        "NONE",
    )
    o = wrapper.forward(
        q,
        (k_data, v_data),
        window_left=window_left,
    )

    for i in range(batch_size):
        qi = q[i]
        ki = torch.cat(
            [
                k_data[kv_indptr[i] : kv_indptr[i + 1] - 1].reshape(
                    -1, num_kv_heads, head_dim
                ),
                k_data[kv_indptr[i + 1] - 1, : kv_last_page_len[i], :],
            ],
            dim=0,
        )
        vi = torch.cat(
            [
                v_data[kv_indptr[i] : kv_indptr[i + 1] - 1].reshape(
                    -1, num_kv_heads, head_dim
                ),
                v_data[kv_indptr[i + 1] - 1, : kv_last_page_len[i], :],
            ],
            dim=0,
        )
        o_ref_i = flashinfer.single_decode_with_kv_cache(
            qi,
            ki,
            vi,
            window_left=window_left,
        )
        o_i_np = o[i].cpu().numpy()
        o_ref_i_np = o_ref_i.cpu().numpy()
        numpy.testing.assert_allclose(o_i_np, o_ref_i_np, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("seq_len", [1, 3, 19, 99, 199, 1999])
@pytest.mark.parametrize("window_left", [3, 13, 23, 43])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_single_decode_prefill_sliding_window_match(
    seq_len, window_left, num_kv_heads, num_qo_heads, head_dim
):
    q = torch.randn(1, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0")
    k = torch.randn(
        seq_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    )
    v = torch.randn(
        seq_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    )
    o = flashinfer.single_prefill_with_kv_cache(
        q, k, v, window_left=window_left, causal=True
    )
    o_decoded = flashinfer.single_decode_with_kv_cache(
        q[0], k, v, window_left=window_left
    )
    numpy.testing.assert_allclose(o.cpu()[0], o_decoded.cpu(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("seq_len", [99, 199, 1999])
@pytest.mark.parametrize("window_left", [43, 233])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_single_prefill_sliding_window(
    seq_len, window_left, num_kv_heads, num_qo_heads, head_dim
):
    q = torch.randn(
        seq_len, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0"
    )
    k = torch.randn(
        seq_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    )
    v = torch.randn(
        seq_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    )

    row_idx = torch.arange(seq_len, dtype=torch.int32, device="cuda:0")[:, None]
    col_idx = torch.arange(seq_len, dtype=torch.int32, device="cuda:0")[None, :]
    mask = (row_idx >= col_idx) & (row_idx - window_left <= col_idx)

    o_ref = flashinfer.single_prefill_with_kv_cache(q, k, v, custom_mask=mask)
    o = flashinfer.single_prefill_with_kv_cache(
        q, k, v, window_left=window_left, causal=True
    )
    numpy.testing.assert_allclose(o.cpu(), o_ref.cpu(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("kv_len", [54, 397])
@pytest.mark.parametrize("qo_len", [37, 47])
@pytest.mark.parametrize("window_left", [13, 33])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("page_size", [1, 16])
def test_batch_paged_prefill_sliding_window(
    batch_size,
    kv_len,
    qo_len,
    window_left,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    page_size,
):
    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    q_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    k_data = torch.randn(
        total_num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    v_data = torch.randn(
        total_num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * num_pages_per_seq
    kv_indices = torch.arange(0, total_num_pages).to(0).int()
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(0)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper.begin_forward(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
    )
    o = wrapper.forward(
        q,
        (k_data, v_data),
        window_left=window_left,
    )

    for i in range(batch_size):
        qi = q[q_indptr[i] : q_indptr[i + 1]]
        ki = torch.cat(
            [
                k_data[kv_indptr[i] : kv_indptr[i + 1] - 1].reshape(
                    -1, num_kv_heads, head_dim
                ),
                k_data[kv_indptr[i + 1] - 1, : kv_last_page_len[i], :],
            ],
            dim=0,
        )
        vi = torch.cat(
            [
                v_data[kv_indptr[i] : kv_indptr[i + 1] - 1].reshape(
                    -1, num_kv_heads, head_dim
                ),
                v_data[kv_indptr[i + 1] - 1, : kv_last_page_len[i], :],
            ],
            dim=0,
        )
        o_ref_i = flashinfer.single_prefill_with_kv_cache(
            qi,
            ki,
            vi,
            window_left=window_left,
            causal=True,
        )
        o_i_np = o[q_indptr[i] : q_indptr[i + 1]].cpu().numpy()
        o_ref_i_np = o_ref_i.cpu().numpy()
        numpy.testing.assert_allclose(o_i_np, o_ref_i_np, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("kv_len", [54, 397])
@pytest.mark.parametrize("qo_len", [37, 47])
@pytest.mark.parametrize("window_left", [13, 33])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_batch_ragged_prefill_sliding_window(
    batch_size, kv_len, qo_len, window_left, num_kv_heads, num_qo_heads, head_dim
):
    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    q_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len
    k = torch.randn(
        batch_size * kv_len,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    v = torch.randn(
        batch_size * kv_len,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * kv_len
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper.begin_forward(
        q_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
    )
    o = wrapper.forward(
        q,
        k,
        v,
        window_left=window_left,
    )

    for i in range(batch_size):
        qi = q[q_indptr[i] : q_indptr[i + 1]]
        ki = k[kv_indptr[i] : kv_indptr[i + 1]]
        vi = v[kv_indptr[i] : kv_indptr[i + 1]]
        o_ref_i = flashinfer.single_prefill_with_kv_cache(
            qi,
            ki,
            vi,
            window_left=window_left,
            causal=True,
        )
        o_i_np = o[q_indptr[i] : q_indptr[i + 1]].cpu().numpy()
        o_ref_i_np = o_ref_i.cpu().numpy()
        numpy.testing.assert_allclose(o_i_np, o_ref_i_np, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_single_prefill_sliding_window(13, 20, 1, 4, 128)
    test_batch_paged_prefill_sliding_window(12, 54, 37, 13, 1, 4, 128, 1)
    test_batch_ragged_prefill_sliding_window(12, 54, 37, 13, 1, 4, 128)
