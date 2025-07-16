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

import torch
from triton.testing import do_bench

import flashinfer
from flashinfer.gemm import (
    batch_deepgemm_fp8_nt_groupwise,
    group_deepgemm_fp8_nt_groupwise,
)
from flashinfer.utils import per_block_cast_to_fp8, per_token_cast_to_fp8


def bench_deepgemm_grouped_fp8_blackwell(batch_size, m, n, k, in_dtype, out_dtype):
    """Benchmark DeepGEMM-based grouped GEMM with FP8 quantization."""

    # Create float32 input tensors
    a_f32 = torch.randn(batch_size * m, k, device="cuda", dtype=torch.float32)
    b_f32 = torch.randn(batch_size, n, k, device="cuda", dtype=torch.float32)

    # Quantize tensor A using per-token quantization
    a_fp8, a_scale = per_token_cast_to_fp8(a_f32)

    # Quantize tensor B using per-block quantization
    b_fp8 = torch.empty_like(b_f32, device="cuda", dtype=torch.float8_e4m3fn)
    b_scale = torch.empty(
        (batch_size, n // 128, k // 128), device="cuda", dtype=torch.float32
    )
    for i in range(batch_size):
        b_fp8[i], b_scale[i] = per_block_cast_to_fp8(b_f32[i])

    # Create group assignment indices
    m_indices = torch.arange(
        batch_size, device="cuda", dtype=torch.int32
    ).repeat_interleave(m)

    # Pre-allocate output tensor
    out = torch.empty(batch_size * m, n, device="cuda", dtype=out_dtype)

    # Benchmark the DeepGEMM function
    ms = do_bench(
        lambda: group_deepgemm_fp8_nt_groupwise(
            a_fp8, b_fp8, a_scale, b_scale, m_indices, out=out, out_dtype=out_dtype
        ),
        warmup=100,
        rep=1000,
    )

    tflops_per_second = 2 * batch_size * m * n * k * 1e-9 / ms
    memory_bandwidth_per_second = (
        sum(
            [
                _.numel() * _.element_size()
                for _ in [a_fp8, b_fp8, a_scale, b_scale, m_indices, out]
            ]
        )
        * 1e-9
        / ms
    )
    print(
        f"group_deepgemm_fp8_nt_groupwise batch_size={batch_size} m={m} n={n} k={k} "
        f"in_dtype={in_dtype} out_dtype={out_dtype}: {tflops_per_second:.2f} TFLOPs/s"
        f"memory_bandwidth: {memory_bandwidth_per_second:.2f} TB/s"
    )

    return tflops_per_second


def bench_deepgemm_batch_fp8_blackwell(batch_size, m, n, k, in_dtype, out_dtype):
    """Benchmark DeepGEMM-based batch GEMM with FP8 quantization."""

    a = torch.rand((batch_size, m, k), device="cuda", dtype=torch.float32)
    b = torch.rand((batch_size, n, k), device="cuda", dtype=torch.float32)
    masked_m = torch.randint(0, m, (batch_size,), device="cuda", dtype=torch.int32)
    a_fp8 = torch.empty_like(a, device="cuda", dtype=torch.float8_e4m3fn)
    a_scale = torch.empty((batch_size, m, k // 128), device="cuda", dtype=torch.float32)
    b_fp8 = torch.empty_like(b, device="cuda", dtype=torch.float8_e4m3fn)
    b_scale = torch.empty(
        (batch_size, n // 128, k // 128), device="cuda", dtype=torch.float32
    )
    for i in range(batch_size):
        a_fp8[i], a_scale[i] = per_token_cast_to_fp8(a[i])
        b_fp8[i], b_scale[i] = per_block_cast_to_fp8(b[i])

    expected_m = min(int(masked_m.float().mean()) + 1, m)

    out = torch.rand((batch_size, m, n), device="cuda", dtype=out_dtype)

    # Benchmark the DeepGEMM function
    ms = do_bench(
        lambda: batch_deepgemm_fp8_nt_groupwise(
            a_fp8,
            b_fp8,
            a_scale,
            b_scale,
            masked_m,
            expected_m,
            out=out,
            out_dtype=out_dtype,
        ),
        warmup=100,
        rep=1000,
    )

    tflops_per_second = 2 * batch_size * m * n * k * 1e-9 / ms
    memory_bandwidth_per_second = (
        sum(
            [
                _.numel() * _.element_size()
                for _ in [a_fp8, b_fp8, a_scale, b_scale, masked_m, out]
            ]
        )
        * 1e-9
        / ms
    )
    print(
        f"group_deepgemm_fp8_nt_groupwise batch_size={batch_size} m={m} n={n} k={k} "
        f"in_dtype={in_dtype} out_dtype={out_dtype}: {tflops_per_second:.2f} TFLOPs/s"
        f"memory_bandwidth: {memory_bandwidth_per_second:.2f} TB/s"
    )

    return tflops_per_second


if __name__ == "__main__":
    print("=== DeepGEMM Grouped FP8 GEMM Benchmark ===\n")

    for batch_size in [1, 4, 8]:
        for m in [128, 256]:
            for n in [4096]:
                for k in [4096]:
                    if m * batch_size <= 16384:  # Limit total problem size
                        bench_deepgemm_grouped_fp8_blackwell(
                            batch_size, m, n, k, torch.float8_e4m3fn, torch.bfloat16
                        )

    for batch_size in [1, 4, 8, 64, 128, 256]:
        for m in [128, 256, 1024, 8192, 16384]:
            for n, k in [(128, 512), (512, 128), (4096, 7168), (7168, 2048)]:
                if m * batch_size <= 16384:  # Limit total problem size
                    bench_deepgemm_batch_fp8_blackwell(
                        batch_size, m, n, k, torch.float8_e4m3fn, torch.bfloat16
                    )
