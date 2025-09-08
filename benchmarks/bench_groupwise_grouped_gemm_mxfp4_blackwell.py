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

from itertools import product

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


def bench_groupwise_grouped_gemm_mxfp4_blackwell(
    group_size, m, n, k, in_dtype, out_dtype
):
    torch.random.manual_seed(0)
    assert n % 8 == 0
    assert k % 128 == 0
    tile_size = 32
    alignment_sf = 128
    fp8_info = torch.finfo(in_dtype)
    a = (
        torch.empty(group_size * m, k, dtype=torch.float32, device="cuda:0")
        .uniform_(-fp8_info.max, fp8_info.max)
        .to(in_dtype)
    )
    b = torch.randint(
        0, 256, (group_size, n, k // 2), dtype=torch.uint8, device="cuda:0"
    )
    out = torch.empty(group_size * m, n, dtype=out_dtype, device="cuda:0")

    a_scale = torch.randint(
        0,
        256,
        (
            (group_size * m + (alignment_sf - 1) * group_size)
            // alignment_sf
            * alignment_sf,
            k // tile_size,
        ),
        dtype=torch.uint8,
        device="cuda:0",
    )
    b_scale = torch.randint(
        0,
        256,
        (
            group_size,
            (n + alignment_sf - 1) // alignment_sf * alignment_sf,
            k // tile_size,
        ),
        dtype=torch.uint8,
        device="cuda:0",
    )

    segment_offsets = torch.arange(
        0, (group_size + 1) * m, m, device="cuda:0", dtype=torch.int32
    )

    ms_best = float("inf")
    config_best = None
    mma_sm_list = [1, 2]
    tile_m_list = [128]
    tile_n_list = [64, 128, 192, 256]
    tile_k_list = [128, 256]
    swap_ab_list = [True, False]
    for mma_sm, tile_m, tile_n, tile_k, swap_ab in product(
        mma_sm_list, tile_m_list, tile_n_list, tile_k_list, swap_ab_list
    ):
        measurements = bench_gpu_time(
            lambda: flashinfer.gemm.group_gemm_mxfp4_nt_groupwise(
                a,
                b,
                a_scale,
                b_scale,
                segment_offsets,
                out=out,
                mma_sm=mma_sm,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                swap_ab=swap_ab,
            ),
            dry_run_time_ms=10,
            repeat_time_ms=100,
        )
        ms = np.median(measurements)
        if ms < ms_best:
            ms_best = ms
            config_best = {
                "mma_sm": mma_sm,
                "tile_m": tile_m,
                "tile_n": tile_n,
                "tile_k": tile_k,
                "swap_ab": swap_ab,
            }
    tflops_per_second = 2 * group_size * m * n * k * 1e-9 / ms_best
    print(
        f"group_gemm_mxfp4_nt_groupwise group_size={group_size} m={m} n={n} k={k} in_dtype={in_dtype} out_dtype={out_dtype}: {tflops_per_second:.2f} TFLOPs/s"
    )
    print(f"best config: {config_best}")
    print()


if __name__ == "__main__":
    for group_size in [1, 3, 8, 16]:
        for m in [128, 512, 1024, 2048, 4096, 8192]:
            for n in [1024, 2048, 4096, 8192]:
                for k in [1024, 2048, 4096, 8192]:
                    bench_groupwise_grouped_gemm_mxfp4_blackwell(
                        group_size, m, n, k, torch.float8_e4m3fn, torch.bfloat16
                    )
