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

import flashinfer.page


@pytest.mark.parametrize("batch_size", [1, 7, 19, 128, 517])
@pytest.mark.parametrize("kv_len", [97, 199, 2049, 31791])
@pytest.mark.parametrize("block_size", [1, 3, 7, 16, 64, 79, 128])
@pytest.mark.parametrize("stride_block", [128])
@pytest.mark.parametrize("stride_n", [1])
def test_block_sparse_indices_to_vector_sparse_offsets(
    batch_size, kv_len, block_size, stride_block, stride_n
):
    if batch_size * kv_len > 1048576:
        pytest.skip("skip large test")
    num_blocks_per_row = (kv_len + block_size - 1) // block_size

    block_sparse_indices = torch.arange(
        batch_size * num_blocks_per_row, device="cuda", dtype=torch.int32
    )
    block_sparse_indptr = torch.arange(
        0,
        batch_size * num_blocks_per_row + 1,
        num_blocks_per_row,
        device="cuda",
        dtype=torch.int32,
    )
    vector_sparse_offsets_buf = torch.zeros(
        batch_size * kv_len, device="cuda", dtype=torch.int32
    )
    vector_sparse_indptr = torch.arange(
        0, batch_size * kv_len + 1, kv_len, device="cuda", dtype=torch.int32
    )
    kv_lens = torch.full((batch_size,), kv_len, device="cuda", dtype=torch.int32)

    vector_sparse_offsets = (
        flashinfer.page.block_sparse_indices_to_vector_sparse_offsets(
            block_sparse_indices,
            block_sparse_indptr,
            vector_sparse_offsets_buf,
            vector_sparse_indptr,
            kv_lens,
            stride_block,
            stride_n,
            block_size,
        )
    )

    # Check that the output is correct
    for i in range(batch_size):
        indices_i = block_sparse_indices[
            i * num_blocks_per_row : (i + 1) * num_blocks_per_row
        ].cpu()
        output_i = vector_sparse_offsets[
            vector_sparse_indptr[i] : vector_sparse_indptr[i + 1]
        ].cpu()

        output_ref_i = (
            indices_i[torch.arange(0, kv_len, dtype=torch.int32) // block_size]
            * stride_block
            + (torch.arange(0, kv_len, dtype=torch.int32) % block_size) * stride_n
        )
        torch.testing.assert_close(output_i, output_ref_i)


if __name__ == "__main__":
    pass
