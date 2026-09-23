# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the generated SM100a contiguous grouped FP8 GEMM against the CuTe-DSL op.

Usage: python benchmarks/bench_cake_group_gemm_fp8_nt_groupwise_contiguous.py [--json out.json]
"""

import argparse
import json

import torch

from flashinfer.gemm import (
    group_gemm_fp8_nt_groupwise_contiguous,
    prepare_group_gemm_fp8_nt_groupwise_contiguous,
)
from flashinfer.testing import bench_gpu_time

# (label, groups, rows_per_group, N, K): MoE FFN shapes with 256 tokens per expert.
SHAPES = [
    ("tp8_gate_up", 512, 256, 256, 4096),
    ("tp8_down", 512, 256, 4096, 128),
    ("ep8_gate_up", 64, 256, 2048, 4096),
    ("ep8_down", 64, 256, 4096, 1024),
    ("wide_ep32_gate_up", 16, 256, 2048, 4096),
    ("wide_ep32_down", 16, 256, 4096, 1024),
]


def make_inputs(groups, rows_per_group, n, k, device):
    generator = torch.Generator(device=device).manual_seed(4734)
    m = groups * rows_per_group
    a = torch.randn((m, k), generator=generator, device=device).to(torch.float8_e4m3fn)
    b = torch.randn((groups, n, k), generator=generator, device=device).to(
        torch.float8_e4m3fn
    )
    a_scale = torch.pow(
        2.0,
        torch.randint(-8, 1, (m, k // 128), generator=generator, device=device).float(),
    )
    b_scale = torch.pow(
        2.0,
        torch.randint(
            -8, 1, (groups, n // 128, k // 128), generator=generator, device=device
        ).float(),
    )
    m_indices = torch.repeat_interleave(
        torch.arange(groups, dtype=torch.int32, device=device),
        torch.full((groups,), rows_per_group, dtype=torch.int64, device=device),
    )
    return a, b, a_scale.contiguous(), b_scale.contiguous(), m_indices.contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="write per-shape results to this path")
    args = parser.parse_args()
    device = torch.device("cuda")
    rows = []
    for label, groups, rows_per_group, n, k in SHAPES:
        a, b, a_scale, b_scale, m_indices = make_inputs(
            groups, rows_per_group, n, k, device
        )
        out_cake = torch.empty(
            (groups * rows_per_group, n), dtype=torch.bfloat16, device=device
        )
        out_cute = torch.empty_like(out_cake)
        prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous(
            a, b, a_scale, b_scale, m_indices, out=out_cake
        )
        prepared.launch()
        group_gemm_fp8_nt_groupwise_contiguous(
            a, b, a_scale, b_scale, m_indices, out=out_cute
        )
        torch.cuda.synchronize()
        max_abs = float((out_cake.float() - out_cute.float()).abs().max())
        cake_ms = float(
            torch.tensor(bench_gpu_time(prepared.launch, cold_l2_cache=True)).median()
        )
        cute_ms = float(
            torch.tensor(
                bench_gpu_time(
                    lambda: group_gemm_fp8_nt_groupwise_contiguous(
                        a, b, a_scale, b_scale, m_indices, out=out_cute
                    ),
                    cold_l2_cache=True,
                )
            ).median()
        )
        flops = 2.0 * groups * rows_per_group * n * k
        row = dict(
            label=label,
            M=groups * rows_per_group,
            N=n,
            K=k,
            groups=groups,
            route=prepared.route,
            grid=list(prepared.grid),
            cake_ms=cake_ms,
            cute_dsl_ms=cute_ms,
            cake_tflops=flops / cake_ms / 1e9,
            cute_dsl_tflops=flops / cute_ms / 1e9,
            speedup=cute_ms / cake_ms,
            max_abs_diff_vs_cute_dsl=max_abs,
        )
        rows.append(row)
        print(
            f"{label:>18} route={prepared.route:<44} cake {cake_ms * 1e3:8.1f} us "
            f"({row['cake_tflops']:6.0f} TFLOPS)  cute_dsl {cute_ms * 1e3:8.1f} us  "
            f"speedup {row['speedup']:.3f}x  max|diff| {max_abs:.3g}"
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
