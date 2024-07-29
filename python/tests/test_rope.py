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

import torch
import numpy as np
import flashinfer
import pytest
from rope_reference import *


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("qkv_len", [1, 4, 19, 204])
@pytest.mark.parametrize("num_qo_heads", [8, 16])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("offset", [0, 15, 99])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_llama_rope_inplace(
    batch_size,
    qkv_len,
    num_qo_heads,
    num_kv_heads,
    offset,
    head_dim,
):
    nnz = batch_size * qkv_len
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
    indptr = torch.tensor(
        [i * qkv_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    )
    offsets = torch.full((batch_size,), offset, dtype=torch.int32, device="cuda:0")

    # reference implementation
    freqs_cis = precompute_freqs_cis(
        head_dim, qkv_len + offset, 10000.0, use_scaled=False
    ).to("cuda:0")
    q_rope_ref, k_rope_ref = apply_rotary_emb(
        q.reshape(batch_size, qkv_len, num_qo_heads, head_dim),
        k.reshape(batch_size, qkv_len, num_kv_heads, head_dim),
        freqs_cis[offset : offset + qkv_len],
    )
    q_rope_ref = q_rope_ref.reshape(nnz, num_qo_heads, head_dim)
    k_rope_ref = k_rope_ref.reshape(nnz, num_kv_heads, head_dim)

    # flashinfer implementation
    flashinfer.apply_rope_inplace(
        q, k, indptr, offsets, interleave=True, rope_theta=1e4
    )

    # compare
    np.testing.assert_allclose(
        q_rope_ref.cpu().numpy(), q.cpu().numpy(), rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        k_rope_ref.cpu().numpy(), k.cpu().numpy(), rtol=1e-3, atol=1e-3
    )


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("qkv_len", [1, 4, 19, 204])
@pytest.mark.parametrize("num_qo_heads", [8, 16])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("offset", [0, 15, 99])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_llama_rope(
    batch_size,
    qkv_len,
    num_qo_heads,
    num_kv_heads,
    offset,
    head_dim,
):
    nnz = batch_size * qkv_len
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
    indptr = torch.tensor(
        [i * qkv_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    )
    offsets = torch.full((batch_size,), offset, dtype=torch.int32, device="cuda:0")

    # reference implementation
    freqs_cis = precompute_freqs_cis(
        head_dim, qkv_len + offset, 10000.0, use_scaled=False
    ).to("cuda:0")
    q_rope_ref, k_rope_ref = apply_rotary_emb(
        q.reshape(batch_size, qkv_len, num_qo_heads, head_dim),
        k.reshape(batch_size, qkv_len, num_kv_heads, head_dim),
        freqs_cis[offset : offset + qkv_len],
    )
    q_rope_ref = q_rope_ref.reshape(nnz, num_qo_heads, head_dim)
    k_rope_ref = k_rope_ref.reshape(nnz, num_kv_heads, head_dim)

    # flashinfer implementation
    q_rope, k_rope = flashinfer.apply_rope(
        q, k, indptr, offsets, interleave=True, rope_theta=1e4
    )

    # compare
    np.testing.assert_allclose(
        q_rope_ref.cpu().numpy(), q_rope.cpu().numpy(), rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        k_rope_ref.cpu().numpy(), k_rope.cpu().numpy(), rtol=1e-3, atol=1e-3
    )


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("qkv_len", [1, 4, 19, 204])
@pytest.mark.parametrize("num_qo_heads", [8, 16])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("offset", [0, 15, 99])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_llama31_rope_inplace(
    batch_size,
    qkv_len,
    num_qo_heads,
    num_kv_heads,
    offset,
    head_dim,
):
    nnz = batch_size * qkv_len
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
    indptr = torch.tensor(
        [i * qkv_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    )
    offsets = torch.full((batch_size,), offset, dtype=torch.int32, device="cuda:0")

    # reference implementation
    freqs_cis = precompute_freqs_cis(
        head_dim, qkv_len + offset, 5e5, use_scaled=True
    ).to("cuda:0")
    q_rope_ref, k_rope_ref = apply_rotary_emb(
        q.reshape(batch_size, qkv_len, num_qo_heads, head_dim),
        k.reshape(batch_size, qkv_len, num_kv_heads, head_dim),
        freqs_cis[offset : offset + qkv_len],
    )
    q_rope_ref = q_rope_ref.reshape(nnz, num_qo_heads, head_dim)
    k_rope_ref = k_rope_ref.reshape(nnz, num_kv_heads, head_dim)

    # flashinfer implementation
    flashinfer.apply_llama31_rope_inplace(
        q, k, indptr, offsets, interleave=True, rope_theta=5e5
    )

    # compare
    np.testing.assert_allclose(
        q_rope_ref.cpu().numpy(), q.cpu().numpy(), rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        k_rope_ref.cpu().numpy(), k.cpu().numpy(), rtol=1e-3, atol=1e-3
    )


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("qkv_len", [1, 4, 19, 204])
@pytest.mark.parametrize("num_qo_heads", [8, 16])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("offset", [0, 15, 99])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_llama31_rope(
    batch_size,
    qkv_len,
    num_qo_heads,
    num_kv_heads,
    offset,
    head_dim,
):
    nnz = batch_size * qkv_len
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
    indptr = torch.tensor(
        [i * qkv_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    )
    offsets = torch.full((batch_size,), offset, dtype=torch.int32, device="cuda:0")

    # reference implementation
    freqs_cis = precompute_freqs_cis(
        head_dim, qkv_len + offset, 5e5, use_scaled=True
    ).to("cuda:0")
    q_rope_ref, k_rope_ref = apply_rotary_emb(
        q.reshape(batch_size, qkv_len, num_qo_heads, head_dim),
        k.reshape(batch_size, qkv_len, num_kv_heads, head_dim),
        freqs_cis[offset : offset + qkv_len],
    )
    q_rope_ref = q_rope_ref.reshape(nnz, num_qo_heads, head_dim)
    k_rope_ref = k_rope_ref.reshape(nnz, num_kv_heads, head_dim)

    # flashinfer implementation
    q_rope, k_rope = flashinfer.apply_llama31_rope(
        q, k, indptr, offsets, interleave=True, rope_theta=5e5
    )

    # compare
    np.testing.assert_allclose(
        q_rope_ref.cpu().numpy(), q_rope.cpu().numpy(), rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        k_rope_ref.cpu().numpy(), k_rope.cpu().numpy(), rtol=1e-3, atol=1e-3
    )


if __name__ == "__main__":
    test_llama_rope_inplace(2, 1, 8, 8, 1, 128)
    test_llama31_rope_inplace(1, 1, 8, 8, 0, 128)
    test_llama_rope(2, 1, 8, 8, 1, 128)
    test_llama31_rope(1, 1, 8, 8, 0, 128)
