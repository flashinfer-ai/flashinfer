#!/usr/bin/env python3
"""Benchmark MX-FP6 block-scaled dense GEMM (b12x) with optional torch reference check."""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from benchmarks.experimental.b12x.fp6_common import (
    capture_graph_replay,
    check_outputs,
    fmt_us,
    make_l2_flush_fn,
)
from b12x._lib.fp6 import SF_VEC_SIZE_FP6, dequant_mxfp6_torch
from b12x._lib.dense_gemm import _expand_packed_mxfp6_ab, dense_gemm

from b12x.testing.reference.fp6 import _bf16_global_scale, _quantize_bf16_matrix


def bench_one_fp6(
    m: int,
    n: int,
    k: int,
    *,
    warmup: int,
    iters: int,
    check: bool,
    l2_flush,
):
    torch.manual_seed(42)
    a_bf = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.2
    b_bf = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.2
    a_packed, a_sf = _quantize_bf16_matrix(a_bf, fmt="e3m2")
    b_packed, b_sf = _quantize_bf16_matrix(b_bf, fmt="e3m2")
    a_gs = _bf16_global_scale(float(a_bf.abs().max().item()))
    b_gs = _bf16_global_scale(float(b_bf.abs().max().item()))
    alpha = (1.0 / (a_gs[0] * b_gs[0])).view(1)
    out = torch.empty((m, n, 1), device="cuda", dtype=torch.bfloat16)

    # Expand the 3:4-packed operands to byte-containers ONCE at setup, exactly
    # like the production path (dense_fp6_linear keeps weights pre-expanded).
    # Without this, dense_gemm's convenience path re-expands the full weight
    # matrix inside every timed launch, swamping the GEMM (~30x at M=1).
    a_exp = _expand_packed_mxfp6_ab(a_packed.unsqueeze(-1), k)
    b_exp = _expand_packed_mxfp6_ab(b_packed.unsqueeze(-1), k)

    def launch():
        dense_gemm(
            (a_exp, a_sf),
            (b_exp, b_sf),
            alpha=alpha,
            ab_dtype="float6_e3m2fn",
            sf_dtype="float8_e8m0fnu",
            sf_vec_size=SF_VEC_SIZE_FP6,
            c_dtype="bfloat16",
            out=out,
            a_preexpanded=True,
            b_preexpanded=True,
        )

    replay = capture_graph_replay(launch)
    replay()
    from b12x.testing.benchmark import samples_ms

    times = samples_ms(launch, warmup=warmup, iters=iters, l2_flush=l2_flush)

    if check:
        a_f = dequant_mxfp6_torch(
            a_packed, a_sf, num_fp6=k, fmt="e3m2", global_scale=a_gs
        )
        b_f = dequant_mxfp6_torch(
            b_packed, b_sf, num_fp6=k, fmt="e3m2", global_scale=b_gs
        )
        ref = a_f @ b_f.T
        check_outputs(
            out[:, :, 0],
            ref,
            label="torch dequant matmul",
            cosine_threshold=0.95,
            # L2 analog of the deliberately loose 0.95 cosine gate
            # (rel ~= sqrt(2*(1-cos))); still fails hard on any power-of-two
            # scale bug (rel >= 0.5).
            rel_rmse_threshold=0.35,
        )

    return times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--m", type=int, nargs="+", default=[128])
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument(
        "--flush-l2", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    l2_flush = make_l2_flush_fn(enabled=args.flush_l2, bytes_hint=args.l2_flush_bytes)
    print("MX-FP6 dense GEMM (b12x, stream-gated events; graph qualified)")
    for m in args.m:
        times = bench_one_fp6(
            m,
            args.n,
            args.k,
            warmup=args.warmup,
            iters=args.iters,
            check=not args.no_check,
            l2_flush=l2_flush,
        )
        print(f"  M={m} N={args.n} K={args.k}: {fmt_us(times)}")


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
