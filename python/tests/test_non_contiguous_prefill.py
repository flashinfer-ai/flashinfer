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

    numpy.testing.assert_allclose(
        o_packed.cpu(), o_contiguous.cpu(), rtol=1e-3, atol=1e-3
    )


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
    wrapper.begin_forward(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
    )
    o_packed = wrapper.forward(q, k, v, causal=causal)
    o_contiguous = wrapper.forward(
        q.contiguous(), k.contiguous(), v.contiguous(), causal=causal
    )

    numpy.testing.assert_allclose(
        o_packed.cpu(), o_contiguous.cpu(), rtol=1e-3, atol=1e-3
    )


if __name__ == "__main__":
    test_single_prefill_packed_input(127, 4, 4, 64, True)
    test_batch_ragged_prefill_packed_input(37, 127, 4, 4, 64, True)
