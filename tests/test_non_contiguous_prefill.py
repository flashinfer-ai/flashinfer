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

import pytest
import torch
from jit_utils import jit_prefill_attention_func_args

import flashinfer


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if flashinfer.jit.has_prebuilt_ops:
        yield
    else:
        try:
            flashinfer.jit.parallel_load_modules(
                jit_prefill_attention_func_args(
                    [torch.float16],  # q_dtypes
                    [torch.float16],  # kv_dtypes
                    [64, 128, 256],  # head_dims
                    [0],  # pos_encoding_modes
                    [False],  # use_sliding_windows
                    [False],  # use_logits_soft_caps
                    [False],  # use_fp16_qk_reductions
                )
            )
        except Exception as e:
            # abort the test session if warmup fails
            pytest.exit(str(e))
        finally:
            yield


@pytest.mark.parametrize("seq_len", [1, 7, 127, 999, 3579])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("num_qo_heads", [4, 8, 32])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [True, False])
def test_single_prefill_packed_input(
    seq_len, num_kv_heads, num_qo_heads, head_dim, causal
):
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be a multiple of num_kv_heads")
    qkv_packed = torch.randn(
        seq_len,
        (num_qo_heads + 2 * num_kv_heads) * head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    q = qkv_packed[:, : num_qo_heads * head_dim].reshape(
        seq_len, num_qo_heads, head_dim
    )
    k = qkv_packed[
        :, num_qo_heads * head_dim : (num_qo_heads + num_kv_heads) * head_dim
    ].reshape(seq_len, num_kv_heads, head_dim)
    v = qkv_packed[:, (num_qo_heads + num_kv_heads) * head_dim :].reshape(
        seq_len, num_kv_heads, head_dim
    )

    o_packed = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=causal)
    o_contiguous = flashinfer.single_prefill_with_kv_cache(
        q.contiguous(), k.contiguous(), v.contiguous(), causal=causal
    )

    torch.testing.assert_close(o_packed, o_contiguous, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99])
@pytest.mark.parametrize("seq_len", [1, 7, 127, 257])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [True, False])
def test_batch_ragged_prefill_packed_input(
    batch_size, seq_len, num_kv_heads, num_qo_heads, head_dim, causal
):
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be a multiple of num_kv_heads")
    nnz = batch_size * seq_len
    qkv_packed = torch.randn(
        nnz,
        (num_qo_heads + 2 * num_kv_heads) * head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    q = qkv_packed[:, : num_qo_heads * head_dim].reshape(nnz, num_qo_heads, head_dim)
    k = qkv_packed[
        :, num_qo_heads * head_dim : (num_qo_heads + num_kv_heads) * head_dim
    ].reshape(nnz, num_kv_heads, head_dim)
    v = qkv_packed[:, (num_qo_heads + num_kv_heads) * head_dim :].reshape(
        nnz, num_kv_heads, head_dim
    )
    qo_indptr = torch.tensor(
        [i * seq_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    )
    kv_indptr = qo_indptr

    workspace_buffer = torch.empty(
        (256 * 1024 * 1024,), dtype=torch.uint8, device="cuda:0"
    )
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace_buffer)
    wrapper.plan(
        qo_indptr, kv_indptr, num_qo_heads, num_kv_heads, head_dim, causal=causal
    )
    o_packed = wrapper.run(q, k, v)
    o_contiguous = wrapper.run(q.contiguous(), k.contiguous(), v.contiguous())

    torch.testing.assert_close(o_packed, o_contiguous, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99])
@pytest.mark.parametrize("page_size", [1, 5])
@pytest.mark.parametrize("seq_len", [1, 7, 127, 257])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [True, False])
def test_batch_paged_prefill_packed_input(
    batch_size,
    page_size,
    seq_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
):
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be a multiple of num_kv_heads")

    nnz = batch_size * seq_len
    num_pages_per_req = (seq_len + page_size - 1) // page_size
    num_pages = batch_size * num_pages_per_req
    last_page_len = (seq_len - 1) % page_size + 1
    k_cache = torch.randn(
        size=(num_pages, page_size, num_kv_heads, head_dim),
        dtype=torch.float16,
        device="cuda:0",
    )
    v_cache = torch.randn_like(k_cache)
    paged_kv_cache = (k_cache, v_cache)
    workspace_buffer = torch.empty(
        (256 * 1024 * 1024,), dtype=torch.uint8, device="cuda:0"
    )
    qo_indptr = torch.tensor(
        [i * seq_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    )
    paged_kv_indptr = torch.tensor(
        [i * num_pages_per_req for i in range(batch_size + 1)],
        dtype=torch.int32,
        device="cuda:0",
    )
    paged_kv_indices = torch.tensor(
        list(range(num_pages)), dtype=torch.int32, device="cuda:0"
    )
    paged_kv_last_page_len = torch.tensor(
        [last_page_len for _ in range(batch_size)], dtype=torch.int32, device="cuda:0"
    )
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer)
    wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        causal=causal,
    )

    qkv_packed = torch.randn(
        size=(nnz, (num_qo_heads + 2 * num_kv_heads) * head_dim),
        dtype=torch.float16,
        device="cuda:0",
    )
    qkv_split_idx = (
        num_qo_heads * head_dim,
        num_kv_heads * head_dim,
        num_kv_heads * head_dim,
    )
    q, _, _ = qkv_packed.split(qkv_split_idx, dim=-1)
    # pretend that we have already appended k/v to paged_kv table
    q = q.view(-1, num_qo_heads, head_dim)
    o_packed = wrapper.run(q, paged_kv_cache)
    o_contiguous = wrapper.run(q.contiguous(), paged_kv_cache)
    torch.testing.assert_close(o_packed, o_contiguous, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_single_prefill_packed_input(127, 4, 4, 64, True)
    test_batch_ragged_prefill_packed_input(37, 127, 4, 4, 64, True)
    test_batch_paged_prefill_packed_input(37, 5, 127, 4, 4, 64, True)
