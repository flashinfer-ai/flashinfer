#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark fused BF16 GEMM+SwiGLU against the unfused composition.

Weight preparation is outside the timed region. Each baseline preserves the
serving boundary and launch sequence: GEMM1 writes BF16 ``[M, 2N]``, then
``silu_and_mul`` reads it and writes BF16 ``[M, N]``. Timing is CUDA-graph
replay with a cold L2, because this kernel targets decode graph replay.

Sweeps ``DEFAULT_SHAPES`` unless ``--n``/``--k`` pin a single shape.

Examples
--------
python benchmarks/bench_mm_bf16_swiglu.py
python benchmarks/bench_mm_bf16_swiglu.py --n 2560 --k 8192 --m 1 8 32
"""

from __future__ import annotations

import argparse
import math
import statistics

import torch
import torch.nn.functional as F
from flashinfer import mm_bf16, mm_bf16_swiglu, prepare_bf16_swiglu_weight, silu_and_mul
from flashinfer.testing import bench_gpu_time

DEFAULT_M = (1, 2, 3, 4, 8, 16, 24, 32)
#: ``(N, K)`` pairs, where ``K`` is the model hidden size and ``N`` the
#: SwiGLU width per tensor-parallel shard. The GLM shared expert this kernel
#: targets, plus the dense-GEMM shapes FlashInfer already benchmarks (see the
#: ``gemm_*`` definitions in ``tests/trace/fi_trace_out``). Together they span
#: a wave-starved tile count, the mma_n widening at mid N, and a shallower K
#: that lowers split-K.
DEFAULT_SHAPES = ((512, 6144), (256, 7168), (1536, 7168), (4096, 4096))
#: ``mm_bf16`` rejects ``pdl=True`` on backends that cannot honour it.
PDL_CAPABLE_BACKENDS = frozenset({"tgv", "tinygemm", "cute-dsl"})


def _fused(a, prepared_weight, out, pdl):
    return mm_bf16_swiglu(a, prepared_weight, out=out, pdl=pdl)


def _mm_bf16_then_swiglu(a, weight_t, gate_up, out, backend, pdl):
    mm_bf16(a, weight_t, out=gate_up, backend=backend, pdl=pdl)
    return silu_and_mul(gate_up, out=out, enable_pdl=True)


def _f_linear_then_swiglu(a, weight, out):
    return silu_and_mul(F.linear(a, weight), out=out, enable_pdl=True)


def _reference(a, weight):
    """Match the unfused BF16 GEMM boundary, then evaluate SwiGLU in FP32."""
    gate, up = F.linear(a, weight).chunk(2, dim=-1)
    return (F.silu(gate.float()) * up.float()).to(torch.bfloat16)


def _median_us(fn, args, repeat_ms):
    samples = bench_gpu_time(
        fn,
        input_args=args,
        dry_run_time_ms=50,
        repeat_time_ms=repeat_ms,
        use_cuda_graph=True,
        num_iters_within_graph=10,
        cold_l2_cache=True,
    )
    return statistics.median(samples) * 1e3


def _check(name, m, reference, actual):
    cosine = F.cosine_similarity(
        reference.float().flatten(), actual.float().flatten(), dim=0
    ).item()
    if cosine <= 0.99:
        raise RuntimeError(f"M={m}: {name} correctness failed, cosine={cosine:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, nargs="+", default=list(DEFAULT_M))
    parser.add_argument("--n", type=int, help="logical SwiGLU width")
    parser.add_argument("--k", type=int, help="hidden size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat-ms", type=int, default=200)
    parser.add_argument("--mm-backend", default="cublaslt", help="mm_bf16 baseline")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    cc = major * 10 + minor
    if not mm_bf16_swiglu.is_compute_capability_supported(cc):
        raise RuntimeError(f"mm_bf16_swiglu does not support SM{cc}")
    if any(m < 1 or m > 128 for m in args.m):
        raise ValueError("every M must be in mm_bf16_swiglu's range [1, 64]")
    shapes = [(args.n, args.k)] if args.n and args.k else list(DEFAULT_SHAPES)
    if any(n <= 0 or n % 64 or k <= 0 or k % 128 for n, k in shapes):
        raise ValueError("N must be divisible by 64 and K by 128")

    mm_pdl = args.mm_backend in PDL_CAPABLE_BACKENDS
    print(
        f"GPU:{torch.cuda.get_device_name()} SM{cc} seed:{args.seed} "
        f"mm_backend:{args.mm_backend} timing:cuda_graph_events L2:cold"
    )
    print(
        "N,K,M,F.linear+act_us,mm_bf16+act_us,fused_pdl0_us,fused_pdl1_us,"
        "speedup_vs_F.linear,speedup_vs_mm_bf16"
    )

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    for n, k in shapes:
        weight = (
            torch.randn(
                (2 * n, k), dtype=torch.bfloat16, device="cuda", generator=generator
            )
            / math.sqrt(k)
        ).to(torch.bfloat16)
        weight_t = weight.T
        prepared = prepare_bf16_swiglu_weight(weight)

        for m in args.m:
            a = torch.randn(
                (m, k), dtype=torch.bfloat16, device="cuda", generator=generator
            )
            gate_up = torch.empty((m, 2 * n), dtype=torch.bfloat16, device="cuda")
            baseline_out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
            fused_out = torch.empty_like(baseline_out)

            reference = _reference(a, weight)
            linear_args = (a, weight, baseline_out)
            mm_args = (a, weight_t, gate_up, baseline_out, args.mm_backend, mm_pdl)
            _check("F.linear+act", m, reference, _f_linear_then_swiglu(*linear_args))
            _check("mm_bf16+act", m, reference, _mm_bf16_then_swiglu(*mm_args))
            for pdl in (False, True):
                actual = _fused(a, prepared, fused_out, pdl)
                _check(f"fused pdl={int(pdl)}", m, reference, actual)

            linear_us = _median_us(_f_linear_then_swiglu, linear_args, args.repeat_ms)
            mm_us = _median_us(_mm_bf16_then_swiglu, mm_args, args.repeat_ms)
            fused_us = [
                _median_us(_fused, (a, prepared, fused_out, pdl), args.repeat_ms)
                for pdl in (False, True)
            ]
            best = min(fused_us)
            print(
                f"{n},{k},{m},{linear_us:.3f},{mm_us:.3f},{fused_us[0]:.3f},{fused_us[1]:.3f},"
                f"{linear_us / best:.4f},{mm_us / best:.4f}"
            )


if __name__ == "__main__":
    main()
