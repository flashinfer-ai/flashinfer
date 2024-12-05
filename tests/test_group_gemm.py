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

import flashinfer
from flashinfer.utils import determine_gemm_backend

DTYPES = [torch.float16]
CUDA_DEVICES = ["cuda:0"]


@pytest.mark.parametrize("batch_size", [1, 77, 199])
@pytest.mark.parametrize("num_rows_per_batch", [3, 10, 99])
@pytest.mark.parametrize("d_in", [128, 1024, 4096])
@pytest.mark.parametrize("d_out", [128, 1024, 4096])
@pytest.mark.parametrize("use_weight_indices", [False, True])
@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("backend", ["auto", "sm90", "sm80"])
def test_segment_gemm(
    batch_size,
    num_rows_per_batch,
    d_in,
    d_out,
    use_weight_indices,
    column_major,
    dtype,
    device,
    backend,
):
    if batch_size * num_rows_per_batch > 8192:
        pytest.skip("batch_size * num_rows_per_batch too large for test.")
    latest_supported_backend = determine_gemm_backend(torch.device(device))
    if backend == "sm90" and latest_supported_backend == "sm80":
        pytest.skip("sm90 backend not supported on this device.")
    torch.manual_seed(42)
    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(device)
    segment_gemm = flashinfer.gemm.SegmentGEMMWrapper(workspace_buffer, backend=backend)
    x = torch.randn(batch_size * num_rows_per_batch, d_in, dtype=dtype).to(device)
    if use_weight_indices:
        num_weights = 1024
        if column_major:
            weight = torch.randn(num_weights, d_out, d_in, dtype=dtype).to(device)
        else:
            weight = torch.randn(num_weights, d_in, d_out, dtype=dtype).to(device)
    else:
        if column_major:
            weight = torch.randn(batch_size, d_out, d_in, dtype=dtype).to(device)
        else:
            weight = torch.randn(batch_size, d_in, d_out, dtype=dtype).to(device)
    y = segment_gemm.run(
        x,
        weight,
        batch_size,
        weight_column_major=column_major,
        seg_lens=torch.full((batch_size,), num_rows_per_batch, dtype=torch.int64),
        weight_indices=(
            (torch.arange(0, batch_size) % num_weights).to(device)
            if use_weight_indices
            else None
        ),
    )

    if use_weight_indices:
        for i in range(batch_size):
            torch.testing.assert_close(
                y[i * num_rows_per_batch : (i + 1) * num_rows_per_batch],
                torch.matmul(
                    x[i * num_rows_per_batch : (i + 1) * num_rows_per_batch],
                    (
                        weight[i % num_weights].T
                        if column_major
                        else weight[i % num_weights]
                    ),
                ),
                rtol=1e-3,
                atol=1e-3,
            )
    else:
        torch.testing.assert_close(
            y,
            torch.matmul(
                x.view(batch_size, num_rows_per_batch, d_in),
                weight.transpose(-1, -2) if column_major else weight,
            ).view(batch_size * num_rows_per_batch, d_out),
            rtol=1e-3,
            atol=1e-3,
        )


if __name__ == "__main__":
    test_segment_gemm(199, 17, 128, 1024, False, False, torch.float16, "cuda:0", "auto")
    test_segment_gemm(199, 17, 128, 1024, False, True, torch.float16, "cuda:0", "auto")
    test_segment_gemm(199, 17, 128, 1024, True, False, torch.float16, "cuda:0", "auto")
    test_segment_gemm(199, 17, 128, 1024, True, True, torch.float16, "cuda:0", "auto")
