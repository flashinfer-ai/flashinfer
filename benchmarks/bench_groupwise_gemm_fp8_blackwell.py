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

import pytest
import torch
from triton.testing import do_bench

import flashinfer
from flashinfer.gemm import gemm_fp8_nt_blockscaled, gemm_fp8_nt_groupwise


def bench_groupwise_gemm_fp8_blackwell(m, n, k, in_dtype, out_dtype):
    a = torch.randn((m, k), device="cuda").to(in_dtype)
    b = torch.randn((n, k), device="cuda").to(in_dtype)
    a_scale = torch.rand((k // 128, m), dtype=torch.float32, device="cuda")
    b_scale = torch.rand((k // 128, n // 128), dtype=torch.float32, device="cuda")

    out = torch.empty((m, n), dtype=out_dtype, device="cuda")
    gemm_fp8_nt_groupwise(a, b, a_scale, b_scale, out)

    ms = do_bench(lambda: gemm_fp8_nt_groupwise(a, b, a_scale, b_scale, out))
    tflops_per_second = 2 * m * n * k * 1e-9 / ms
    print(
        f"gemm_fp8_nt_groupwise {m} {n} {k} {in_dtype} {out_dtype}: {tflops_per_second:.2f} TFLOPs/s"
    )


if __name__ == "__main__":
    for m in [1024, 2048, 4096, 8192]:
        for n in [1024, 2048, 4096, 8192]:
            for k in [1024, 2048, 4096, 8192]:
                bench_groupwise_gemm_fp8_blackwell(
                    m, n, k, torch.float8_e5m2, torch.bfloat16
                )
