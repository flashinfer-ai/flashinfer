"""
Copyright (c) 2025 by FlashInfer team.

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
from triton.testing import do_bench

import flashinfer


def bench_groupwise_grouped_gemm_fp8_blackwell(
    batch_size, m, n, k, in_dtype, out_dtype
):
    a = torch.randn(batch_size * m, k, device="cuda:0").to(in_dtype)
    b = torch.randn(batch_size, n, k, device="cuda:0").to(in_dtype)
    out = torch.empty(batch_size * m, n, device="cuda:0", dtype=out_dtype)

    a_scale = torch.randn(
        (k // 128, batch_size * m), dtype=torch.float32, device="cuda:0"
    )
    b_scale = torch.randn(
        (batch_size, k // 128, n // 128), dtype=torch.float32, device="cuda:0"
    )

    segment_offsets = torch.arange(
        0, (batch_size + 1) * m, m, device="cuda:0", dtype=torch.int32
    )

    ms = do_bench(
        lambda: flashinfer.gemm.group_gemm_fp8_nt_groupwise(
            a, b, a_scale, b_scale, segment_offsets, out=out, mma_sm=2
        ),
        warmup=100,
        rep=1000,
    )
    tflops_per_second = 2 * batch_size * m * n * k * 1e-9 / ms
    print(
        f"group_gemm_fp8_nt_groupwise batch_size={batch_size} m={m} n={n} k={k} in_dtype={in_dtype} out_dtype={out_dtype}: {tflops_per_second:.2f} TFLOPs/s"
    )


if __name__ == "__main__":
    for batch_size in [1, 3, 8, 16]:
        for m in [128, 512, 1024, 2048, 4096, 8192]:
            for n in [1024, 2048, 4096, 8192]:
                for k in [1024, 2048, 4096, 8192]:
                    bench_groupwise_grouped_gemm_fp8_blackwell(
                        batch_size, m, n, k, torch.float8_e5m2, torch.bfloat16
                    )
