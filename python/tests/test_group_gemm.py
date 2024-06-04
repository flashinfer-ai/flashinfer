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

import flashinfer
import torch
import pytest

@pytest.mark.parametrize("batch_size", [1, 33, 77, 377])
@pytest.mark.parametrize("num_rows_per_batch", [3, 10, 99])
@pytest.mark.parametrize("d_in", [128, 1024, 4096])
@pytest.mark.parametrize("d_out", [128, 1024, 4096])
def test_segment_gemm(
    batch_size,
    num_rows_per_batch,
    d_in,
    d_out,
):
    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0)
    segment_gemm = flashinfer.group_gemm.SegmentGEMMWrapper(workspace_buffer)
    segment_gemm.register_problem(
        batch_size,
        d_in,
        d_out,
        weight_column_major=True,
        seg_lens=torch.full((batch_size,), num_rows_per_batch),
        seg_indptr=None,
        weight_indices=None,
        dtype=torch.float16,
    )


if __name__ == "__main__":
    test_segment_gemm(1, 3, 128, 128)
