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
import numpy as np
import scipy as sp
import flashinfer


def bsr_attention_ref(
    q,
    k,
    v,
    indptr,
    indices,
    mask_data,
):
    M = q.shape[0]
    N = k.shape[0]
    bsr = sp.sparse.bsr_matrix(
        (mask_data.cpu().numpy(), indices.cpu().numpy(), indptr.cpu().numpy()),
        shape=(M, N),
    )
    dense_mask = torch.tensor(bsr.toarray(), dtype=bool, device=q.device)
    o = flashinfer.single_prefill_with_kv_cache(q, k, v, custom_mask=dense_mask)
    return o


@pytest.mark.parametrize("R", [1, 4, 16])
@pytest.mark.parametrize("C", [1, 4, 16])
@pytest.mark.parametrize("M", [64, 128, 256])
@pytest.mark.parametrize("N", [64, 128, 256])
@pytest.mark.parametrize("num_qo_heads", [1, 4, 16])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 16])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("mask_inside_block", [True, False])
def test_block_sparse_attention(
    R, C, M, N, num_qo_heads, num_kv_heads, head_dim, mask_inside_block
):
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")
    rng = np.random.default_rng()
    MB = M // R
    NB = N // C
    S = sp.sparse.random(MB, NB, density=0.25, random_state=rng).tocsr()
    indptr = torch.from_numpy(S.indptr).to(0)
    indices = torch.from_numpy(S.indices).to(0)
    nnz = S.nnz
    if mask_inside_block:
        data_mask = (torch.rand((nnz, R, C)) > 0.5).to(0)
    else:
        data_mask = torch.full((nnz, R, C), True, dtype=bool, device=0)
    q = torch.randn((M, num_qo_heads, head_dim), dtype=torch.float16, device=0)
    k = torch.randn((N, num_kv_heads, head_dim), dtype=torch.float16, device=0)
    v = torch.randn((N, num_kv_heads, head_dim), dtype=torch.float16, device=0)

    o_ref = bsr_attention_ref(q, k, v, indptr, indices, data_mask)
    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=0)
    sparse_attention_wrapper = flashinfer.BlockSparseAttentionWrapper(workspace_buffer)

    sparse_attention_wrapper.begin_forward(
        indptr,
        indices,
        M,
        N,
        R,
        C,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        mask=data_mask if mask_inside_block else None,
    )

    o = sparse_attention_wrapper.forward(q, k, v)
    sparse_attention_wrapper.end_forward()
    np.testing.assert_allclose(o_ref.cpu(), o.cpu(), atol=1e-2, rtol=1e-3)


if __name__ == "__main__":
    test_block_sparse_attention(1, 1, 64, 64, 1, 1, 128, True)
