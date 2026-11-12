# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the generated contiguous grouped FP8 GEMM programs against the CuTe-DSL op.

Two sections: the FP32-scale family on fully populated groups, and the block-scaled
family (packed UE8M0 scales, compact MoE layout with ``-1`` padding rows) on routed
Qwen3.5-style expert geometries.  The CuTe-DSL op receives the same compact layout
with the padding rows forward-filled, which is what the stock contiguous path
computes; the block-scaled program skips those rows.

Usage: python benchmarks/bench_cake_group_gemm_fp8_nt_groupwise_contiguous.py
           [--json out.json] [--skip-block-scaled]
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


# Block-scaled family: (label, tokens, experts, top_k, N, K) of the Qwen3.5 MoE
# FFNs (35B-A3B TP1 and 397B-A17B TP4) on the compact layout at alignment 128.
BLOCK_SCALED_SHAPES = [
    ("q35b_gate_up_t4096", 4096, 256, 8, 1024, 2048),
    ("q35b_down_t4096", 4096, 256, 8, 2048, 512),
    ("q397b_gate_up_t4096", 4096, 512, 10, 512, 4096),
    ("q397b_down_t4096", 4096, 512, 10, 4096, 256),
    ("q35b_gate_up_t16384", 16384, 256, 8, 1024, 2048),
    ("q35b_down_t256", 256, 256, 8, 2048, 512),
]
BLOCK_SCALED_ALIGNMENT = 128


def routed_group_counts(tokens, experts, top_k, generator, device):
    """Rows per expert of a skewed top-k routing (rank-weighted, deterministic)."""
    weights = 1.0 / torch.arange(1, experts + 1, device=device, dtype=torch.float32)
    weights = weights[torch.randperm(experts, generator=generator, device=device)]
    choice = torch.multinomial(
        weights.expand(tokens, experts), top_k, replacement=False, generator=generator
    )
    return torch.bincount(choice.reshape(-1), minlength=experts).tolist()


def compact_layout(group_counts, alignment, device):
    """``m_indices`` of the compact MoE layout: each expert's rows then ``-1`` padding
    up to ``alignment``, plus the engine's ``G * (alignment - 1)`` tail rounded to 128."""
    pieces = []
    for g, count in enumerate(group_counts):
        padded = -(-count // alignment) * alignment
        if count:
            pieces.append(torch.full((count,), g, dtype=torch.int32))
        if padded > count:
            pieces.append(torch.full((padded - count,), -1, dtype=torch.int32))
    m_indices = torch.cat(pieces)
    total = max(
        int(m_indices.numel()),
        sum(group_counts) + len(group_counts) * (alignment - 1),
    )
    total = -(-total // 128) * 128
    if total > m_indices.numel():
        tail = torch.full((total - m_indices.numel(),), -1, dtype=torch.int32)
        m_indices = torch.cat([m_indices, tail])
    return m_indices.to(device).contiguous()


def pack_ue8m0_mn_major(scale):
    """DeepGEMM's packed activation-scale layout: ``(rows, ceil(k/4))`` int32 words
    with strides ``(1, rows)`` (four UE8M0 exponents per word)."""
    rows, k = scale.shape
    exponents = (scale.contiguous().view(torch.int32) >> 23).to(torch.uint8)
    cols = -(-k // 4)
    padded = torch.zeros((rows, 4 * cols), dtype=torch.uint8, device=scale.device)
    padded[:, :k] = exponents
    packed = torch.empty((cols, rows), dtype=torch.int32, device=scale.device).mT
    packed.copy_(padded.view(torch.int32))
    return packed


def pack_ue8m0_row_repeated(scale):
    """sglang's ``transform_scale_ue8m0`` weight layout: ``(G, N, ceil(k/4))`` int32
    words with strides ``(N*cols, 1, N)``, each 128-row block's word repeated."""
    groups, n_blocks, k = scale.shape
    exponents = (
        (scale.contiguous().view(torch.int32) >> 23)
        .to(torch.uint8)
        .reshape(groups * n_blocks, k)
    )
    cols = -(-k // 4)
    padded = torch.zeros(
        (groups * n_blocks, 4 * cols), dtype=torch.uint8, device=scale.device
    )
    padded[:, :k] = exponents
    words = (
        padded.view(torch.int32)
        .view(groups, n_blocks, cols)
        .repeat_interleave(128, dim=1)
    )
    packed = torch.empty(
        (groups, cols, n_blocks * 128), dtype=torch.int32, device=scale.device
    ).permute(0, 2, 1)
    packed.copy_(words)
    return packed


def make_block_scaled_inputs(tokens, experts, top_k, n, k, device):
    generator = torch.Generator(device=device).manual_seed(9535)
    counts = routed_group_counts(tokens, experts, top_k, generator, device)
    m_indices = compact_layout(counts, BLOCK_SCALED_ALIGNMENT, device)
    m = int(m_indices.numel())
    a = torch.randn((m, k), generator=generator, device=device).to(torch.float8_e4m3fn)
    b = torch.randn((experts, n, k), generator=generator, device=device).to(
        torch.float8_e4m3fn
    )
    a_scale = torch.pow(
        2.0,
        torch.randint(-8, 1, (m, k // 128), generator=generator, device=device).float(),
    )
    b_scale = torch.pow(
        2.0,
        torch.randint(
            -8, 1, (experts, n // 128, k // 128), generator=generator, device=device
        ).float(),
    )
    filled = torch.cummax(m_indices, 0).values.clamp_(min=0)
    return (
        a,
        b,
        a_scale.contiguous(),
        b_scale.contiguous(),
        pack_ue8m0_mn_major(a_scale),
        pack_ue8m0_row_repeated(b_scale),
        m_indices,
        filled,
        sum(counts),
    )


def bench_block_scaled(device):
    rows = []
    for label, tokens, experts, top_k, n, k in BLOCK_SCALED_SHAPES:
        (
            a,
            b,
            a_scale,
            b_scale,
            a_packed,
            b_packed,
            m_indices,
            filled,
            valid_rows,
        ) = make_block_scaled_inputs(tokens, experts, top_k, n, k, device)
        m = int(m_indices.numel())
        out_cake = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
        out_cute = torch.empty_like(out_cake)
        prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous(
            a,
            b,
            a_packed,
            b_packed,
            m_indices,
            out=out_cake,
            alignment=BLOCK_SCALED_ALIGNMENT,
        )
        prepared.launch()
        group_gemm_fp8_nt_groupwise_contiguous(
            a, b, a_scale, b_scale, filled, out=out_cute
        )
        torch.cuda.synchronize()
        valid = m_indices >= 0
        max_abs = float((out_cake[valid].float() - out_cute[valid].float()).abs().max())
        cake_ms = float(
            torch.tensor(
                bench_gpu_time(prepared.launch, cold_l2_cache=True, enable_cupti=True)
            ).median()
        )
        cute_ms = float(
            torch.tensor(
                bench_gpu_time(
                    lambda: group_gemm_fp8_nt_groupwise_contiguous(
                        a, b, a_scale, b_scale, filled, out=out_cute
                    ),
                    cold_l2_cache=True,
                    enable_cupti=True,
                )
            ).median()
        )
        flops = 2.0 * valid_rows * n * k
        row = dict(
            label=label,
            M=m,
            valid_rows=valid_rows,
            N=n,
            K=k,
            groups=experts,
            alignment=BLOCK_SCALED_ALIGNMENT,
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
            f"{label:>20} route={prepared.route:<22} M={m:>6} valid={valid_rows:>6} "
            f"cake {cake_ms * 1e3:8.1f} us ({row['cake_tflops']:6.0f} TFLOPS)  "
            f"cute_dsl {cute_ms * 1e3:8.1f} us  speedup {row['speedup']:.3f}x  "
            f"max|diff| {max_abs:.3g}"
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="write per-shape results to this path")
    parser.add_argument(
        "--skip-block-scaled",
        action="store_true",
        help="only run the FP32-scale family",
    )
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
            torch.tensor(
                bench_gpu_time(prepared.launch, cold_l2_cache=True, enable_cupti=True)
            ).median()
        )
        cute_ms = float(
            torch.tensor(
                bench_gpu_time(
                    lambda: group_gemm_fp8_nt_groupwise_contiguous(
                        a, b, a_scale, b_scale, m_indices, out=out_cute
                    ),
                    cold_l2_cache=True,
                    enable_cupti=True,
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
    if not args.skip_block_scaled:
        rows.extend(bench_block_scaled(device))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
