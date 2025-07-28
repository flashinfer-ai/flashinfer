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

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


def bench_grouped_gemm(
    batch_size, num_tokens_per_group, d_in, d_out, dtype, output_dtype
):
    np.random.seed(42)
    W = torch.randn(batch_size, d_out, d_in, device="cuda:0").to(dtype)
    X = torch.randn(batch_size * num_tokens_per_group, d_in, device="cuda:0").to(dtype)
    Y = torch.empty(
        batch_size * num_tokens_per_group, d_out, dtype=output_dtype, device="cuda:0"
    )

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    segment_gemm = flashinfer.gemm.SegmentGEMMWrapper(workspace_buffer, backend="auto")
    seg_indptr = torch.arange(
        0,
        (batch_size + 1) * num_tokens_per_group,
        num_tokens_per_group,
        dtype=torch.int64,
        device="cuda:0",
    )

    measurements = bench_gpu_time(
        lambda: segment_gemm.run(X, W, batch_size, True, out=Y, seg_indptr=seg_indptr),
        dry_runs=10,
        num_iters=100,
        l2_flush=True,
        l2_flush_size_mb=256,
        l2_flush_device=torch.device("cuda:0"),
    )
    ms = np.median(measurements)
    flops = 2 * batch_size * num_tokens_per_group * d_in * d_out

    print(
        f"Config: batch_size={batch_size}, num_tokens_per_group={num_tokens_per_group}, d_in={d_in}, d_out={d_out}, dtype={dtype}, output_dtype={output_dtype}"
    )
    print(f"FLOPs: {flops / ms * 1e-9:.2f} TFLOPs/s")


if __name__ == "__main__":
    for dtype_in in [torch.float8_e4m3fn, torch.bfloat16]:
        for dtype_out in [torch.bfloat16]:
            for batch_size in [1, 3, 8, 16]:
                for num_tokens_per_group in [32, 64, 128, 256, 512]:
                    for d_in in [4096, 8192]:
                        for d_out in [4096, 8192]:
                            bench_grouped_gemm(
                                batch_size,
                                num_tokens_per_group,
                                d_in,
                                d_out,
                                dtype_in,
                                dtype_out,
                            )
