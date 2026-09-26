# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the generated SM100a grouped FP8 gate_up GEMM + SwiGLU + FP8 quant programs against the FlashInfer chain.

Chain = CuTe-DSL ``group_gemm_fp8_nt_groupwise_contiguous`` + ``silu_and_mul`` +
``per_token_group_quant_8bit`` (sum of the three kernels' CUPTI times, cold L2).

Usage: python benchmarks/bench_cake_group_gemm_fp8_nt_groupwise_contiguous_silu_quant.py [--json out.json]
"""

import argparse
import json

import torch

from flashinfer.activation import silu_and_mul
from flashinfer.gemm import (
    group_gemm_fp8_nt_groupwise_contiguous,
    prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant,
)
from flashinfer.quantization import per_token_group_quant_8bit
from flashinfer.testing import bench_gpu_time

GROUP_SIZE = 128
EPS = 1e-10

# (label, group_counts, 2H, K): the wide-EP32 gate_up shape (16 experts, 4096 rows) under
# uniform, random 128-aligned and all-odd-block routings.
SHAPES = [
    ("wide_ep32_gate_up_uniform", [256] * 16, 2048, 4096),
    (
        "wide_ep32_gate_up_random_aligned",
        [256, 0, 512, 0, 384, 640, 128, 384, 256, 0, 128, 128, 384, 384, 512, 0],
        2048,
        4096,
    ),
    ("wide_ep32_gate_up_all_odd", [384] * 8 + [128] * 8, 2048, 4096),
    # small-M route (grouped GEMM + generated SwiGLU/group-quant kernel): aligned and partial-tail routings
    ("one_pair", [256], 512, 1024),
    ("odd_blocks_with_empty", [384, 128, 0, 640], 512, 512),
    ("leading_internal_empty_partial_tail", [0, 256, 0, 0, 128, 100], 256, 1024),
]


def make_inputs(group_counts, n2, k, device):
    generator = torch.Generator(device=device).manual_seed(4734)
    groups = len(group_counts)
    m = sum(group_counts)
    a = torch.randn((m, k), generator=generator, device=device).to(torch.float8_e4m3fn)
    b = torch.randn((groups, n2, k), generator=generator, device=device).to(
        torch.float8_e4m3fn
    )
    a_scale = torch.pow(
        2.0,
        torch.randint(-8, 1, (m, k // 128), generator=generator, device=device).float(),
    )
    b_scale = torch.pow(
        2.0,
        torch.randint(
            -8, 1, (groups, n2 // 128, k // 128), generator=generator, device=device
        ).float(),
    )
    m_indices = torch.repeat_interleave(
        torch.arange(groups, dtype=torch.int32, device=device),
        torch.tensor(group_counts, dtype=torch.int64, device=device),
    )
    return a, b, a_scale.contiguous(), b_scale.contiguous(), m_indices.contiguous()


def _median_ms(fn):
    return float(
        torch.tensor(bench_gpu_time(fn, cold_l2_cache=True, enable_cupti=True)).median()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="write per-shape results to this path")
    args = parser.parse_args()
    device = torch.device("cuda")
    rows = []
    for label, group_counts, n2, k in SHAPES:
        a, b, a_scale, b_scale, m_indices = make_inputs(group_counts, n2, k, device)
        m = sum(group_counts)
        prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
            a, b, a_scale, b_scale, m_indices
        )
        fused_q, fused_s = prepared.launch()
        y = torch.empty((m, n2), dtype=torch.bfloat16, device=device)
        group_gemm_fp8_nt_groupwise_contiguous(a, b, a_scale, b_scale, m_indices, out=y)
        act = silu_and_mul(y)
        chain_q, chain_s = per_token_group_quant_8bit(
            act, GROUP_SIZE, EPS, torch.float8_e4m3fn
        )
        torch.cuda.synchronize()
        scale_exact = bool(torch.equal(fused_s, chain_s.reshape(fused_s.shape)))
        q_exact_frac = float(
            (fused_q.view(torch.uint8) == chain_q.view(torch.uint8)).float().mean()
        )
        fused_ms = _median_ms(prepared.launch)
        gemm_ms = _median_ms(
            lambda: group_gemm_fp8_nt_groupwise_contiguous(
                a, b, a_scale, b_scale, m_indices, out=y
            )
        )
        act_ms = _median_ms(lambda: silu_and_mul(y, out=act))
        quant_ms = _median_ms(
            lambda: per_token_group_quant_8bit(
                act, GROUP_SIZE, EPS, torch.float8_e4m3fn
            )
        )
        chain_ms = gemm_ms + act_ms + quant_ms
        row = dict(
            label=label,
            M=m,
            N2=n2,
            K=k,
            groups=len(group_counts),
            route=prepared.route,
            gemm_backend=prepared.gemm_backend,
            prepared_kernels=prepared.num_kernels,
            grid=list(prepared.grid),
            fused_ms=fused_ms,
            chain_ms=chain_ms,
            chain_gemm_ms=gemm_ms,
            chain_silu_and_mul_ms=act_ms,
            chain_quant_ms=quant_ms,
            speedup=chain_ms / fused_ms,
            scales_exact=scale_exact,
            fp8_bitwise_equal_fraction=q_exact_frac,
        )
        rows.append(row)
        print(
            f"{label:36s} {prepared.route}[{prepared.gemm_backend or '-'}] {fused_ms * 1e3:8.2f} us | chain {chain_ms * 1e3:8.2f} us "
            f"(gemm {gemm_ms * 1e3:.2f} + act {act_ms * 1e3:.2f} + quant {quant_ms * 1e3:.2f}) "
            f"| {row['speedup']:.3f}x | scales exact {scale_exact} | fp8 equal {q_exact_frac:.4f}"
        )
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
