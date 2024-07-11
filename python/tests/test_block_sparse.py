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
from scipy.sparse import bsr_matrix
import flashinfer

def bsr_attention_ref(
    q, kv,
    indptr,
    indices,
    R, C,
):
    M = q.shape[0]
    N, _, _, H, D = kv.shape[0]
    nnz = indices.shape[0]
    data = np.zeros((nnz, R, C), dtype=np.float32)
    bsr_matrix = bsr_matrix((data, indices, indptr), shape=(M, N * C))
    dense_mask = torch.tensor(bsr_matrix.toarray(), dtype=bool, device=q.device)

    k = kv[:, 0].view(-1, H, D)
    v = kv[:, 1].view(-1, H, D)

    o = flashinfer.single_prefill_with_kv_cache(
        q, k, v, custom_mask=dense_mask
    )
    return o


def test_block_sparse_attention(

):
