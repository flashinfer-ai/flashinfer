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

from typing import Dict
from flashinfer.autotuner import autotune
from flashinfer.trtllm_low_latency_gemm import prepare_low_latency_gemm_weights
import numpy as np
import torch

from flashinfer import mm_fp8
from flashinfer.testing.utils import bench_gpu_time

_cache_permute_indices: Dict[torch.Size, torch.Tensor] = {}


def to_float8(
    x: torch.Tensor, dtype=torch.float8_e4m3fn
) -> tuple[torch.Tensor, torch.Tensor]:
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


def bench_mm_fp8(m, n, k, in_dtype, out_dtype):
    torch.manual_seed(123)
    input_tensor = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    input_fp8, input_inv_s = to_float8(input_tensor, dtype=in_dtype)

    # mat2 row  major -> column major
    mat2 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    mat2_fp8, mat2_inv_s = to_float8(mat2, dtype=in_dtype)

    res = torch.zeros([m, n], device="cuda", dtype=out_dtype)
    global_scale = input_inv_s * mat2_inv_s

    # Do row shuffling.
    prepared_weights = prepare_low_latency_gemm_weights(
        mat2_fp8, _cache_permute_indices
    )

    with autotune(True):
        mm_fp8(
            input_fp8,
            prepared_weights,
            global_scale,
            out=res,
        )

    measurements = bench_gpu_time(
        lambda: mm_fp8(
            input_fp8,
            prepared_weights,
            global_scale,
            res,
        ),
        dry_run_time_ms=500,
        repeat_time_ms=2500,
        use_cuda_graph=True,
    )
    ms = np.median(measurements)
    tflops_per_second = 2 * m * n * k * 1e-9 / ms

    bandwidth = (
        (
            input_fp8.numel() * input_fp8.element_size()
            + prepared_weights.numel() * prepared_weights.element_size()
            + res.numel() * res.element_size()
        )
        / ms
        / 1e9
    )

    print(
        f"mm_fp8 m={m} n={n} k={k} in_dtype={in_dtype} out_dtype={out_dtype}: {tflops_per_second:.2f} TFLOPs/s over {ms:.6f} ms, {bandwidth:.2f} TB/s"
    )


if __name__ == "__main__":
    for m in [1, 2, 4, 8, 16, 32, 64]:
        for n in [2560, 5120, 8192]:
            for k in [16384, 32768]:
                bench_mm_fp8(m, n, k, torch.float8_e4m3fn, torch.bfloat16)
