# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0

"""Benchmark the experimental SM120 CuTe DSL Mega Gate (``prepare_mega_gate``).

Each row times one fused launch (BF16 routing GEMM + sqrt-softplus top-k with an
expert bias, a two-replica physical map and the logical top-k output, as an
expert-parallel serving engine calls it) against the unfused two-kernel path
used on SM12x today: a cuBLAS BF16 GEMM with FP32 logits followed by b12x's
Triton ``route_topk`` (sqrt-softplus, bias, renormalized top-k; it has no
physical map, which only favors the baseline). The cuBLAS GEMM alone is
reported as well. Each path is captured in a CUDA graph, including both stages
of the unfused path. L2 is flushed before each replay, preserving logits in L2
between GEMM and top-k. Timing uses CUPTI under ``--cupti`` (cupti-python >= 13)
and CUDA events otherwise. ``--fixed-routing`` flags every
token in ``fix_routing_mask`` (hash-routed layers): the fused call then takes
the given logical experts of each token instead of ranking them (the baseline
still ranks).

Usage::

    python benchmarks/bench_mega_gate_sm12x.py [--cupti] [--tokens 1 16 128 ...]
        [--hidden 4096 --num-experts 256] [--fixed-routing] [--num-sms 64]
"""

import argparse
import importlib.util
import pathlib
import statistics

import torch

from flashinfer.mega_gate import prepare_mega_gate
from flashinfer.testing import bench_gpu_time

DEFAULT_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)


def make_inputs(num_tokens, hidden, num_experts, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        (num_tokens, hidden), dtype=torch.bfloat16, device=device, generator=gen
    )
    weight = (
        torch.randn((num_experts, hidden), device=device, generator=gen) * hidden**-0.5
    ).to(torch.bfloat16)
    bias = torch.randn((num_experts,), device=device, generator=gen) * 0.1
    mapping = torch.stack(
        (
            torch.arange(num_experts, dtype=torch.int32, device=device),
            torch.arange(num_experts, dtype=torch.int32, device=device) + num_experts,
        ),
        1,
    ).contiguous()
    counts = torch.full((num_experts,), 2, dtype=torch.int32, device=device)
    return x, weight, bias, mapping, counts


def make_gemm(x, weight):
    """Select GEMM once, and label the lower-precision fallback explicitly."""
    try:
        torch.mm(x, weight.t(), out_dtype=torch.float32)
    except (NotImplementedError, RuntimeError, TypeError):
        return lambda: torch.mm(x, weight.t()).float(), "BF16 output + FP32 cast"
    return lambda: torch.mm(x, weight.t(), out_dtype=torch.float32), "FP32 output"


def load_triton_route_topk():
    """b12x's Triton router, loaded by path: it needs torch and triton only, not b12x."""
    import flashinfer

    path = (
        pathlib.Path(flashinfer.__file__).parent
        / "experimental/b12x/moe/_shared/routing.py"
    )
    spec = importlib.util.spec_from_file_location("_b12x_triton_routing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.route_topk


def median_us(fn, use_cupti):
    # Capture explicitly so CUDA-event timing also flushes L2 before each
    # replay; the generic event+graph timer instead needs rotating inputs.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    times = bench_gpu_time(
        graph.replay,
        enable_cupti=use_cupti,
        dry_run_time_ms=25,
        repeat_time_ms=100,
        cold_l2_cache=True,
    )
    return statistics.median(times) * 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=list(DEFAULT_TOKENS))
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cupti", action="store_true")
    parser.add_argument(
        "--pdl",
        action="store_true",
        help="launch with PDL (a standalone timing then also counts the wait for the "
        "previous kernel)",
    )
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument(
        "--fixed-routing",
        action="store_true",
        help="flag every token in fix_routing_mask, with random given experts",
    )
    parser.add_argument(
        "--num-sms",
        type=int,
        default=None,
        help="cap the SMs the fused kernel uses (as when it shares the GPU)",
    )
    args = parser.parse_args()

    props = torch.cuda.get_device_properties(0)
    print(
        f"{props.name} (sm_{props.major}{props.minor}, {props.multi_processor_count} SMs) "
        f"K={args.hidden} E={args.num_experts} top-k={args.num_topk} "
        f"deterministic={args.deterministic} fixed_routing={args.fixed_routing} "
        f"timer={'CUPTI' if args.cupti else 'CUDA events'}; CUDA graph replay, cold L2"
    )
    route_topk = None if args.no_baseline else load_triton_route_topk()
    header = (
        f"{'tokens':>7} {'bt':>3} {'split':>5} {'ctas':>5} {'fused us':>9} {'GB/s':>6} "
        f"{'TFLOPS':>7} {'cublas us':>9} {'+triton us':>10} {'speedup':>8}"
    )
    print(header)
    for num_tokens in args.tokens:
        x, weight, bias, mapping, counts = make_inputs(
            num_tokens, args.hidden, args.num_experts
        )
        logical = torch.empty(
            (num_tokens, args.num_topk), dtype=torch.int64, device="cuda"
        )
        fixed = {}
        if args.fixed_routing:
            gen = torch.Generator(device="cuda").manual_seed(1)
            logical = torch.randint(
                0, args.num_experts, logical.shape, device="cuda", generator=gen
            )
            fixed["fix_routing_mask"] = torch.ones(
                num_tokens, dtype=torch.bool, device="cuda"
            )
        plan = prepare_mega_gate(
            x,
            weight,
            args.num_topk,
            bias=bias,
            to_physical_map=mapping,
            logical_count=counts,
            unmapped_topk_idx=logical,
            routed_scaling_factor=args.scale,
            deterministic=args.deterministic,
            enable_pdl=args.pdl,
            **fixed,
            **({} if args.num_sms is None else {"num_sms": args.num_sms}),
        )
        fused = median_us(plan.run, args.cupti)
        moved = (x.numel() + weight.numel()) * 2
        flops = 2 * num_tokens * args.num_experts * args.hidden
        route = getattr(plan, "route", {})
        row = (
            f"{num_tokens:>7} {route.get('block_tokens', '-'):>3} "
            f"{route.get('num_split_k', '-'):>5} {route.get('launch_ctas', '-'):>5} "
            f"{fused:>9.2f} {moved / fused / 1e3:>6.0f} {flops / fused / 1e6:>7.1f}"
        )
        if not args.no_baseline:
            run_gemm, baseline_label = make_gemm(x, weight)
            print(f"# tokens={num_tokens}: cuBLAS baseline: {baseline_label}")
            gemm = median_us(run_gemm, args.cupti)
            buffers = (
                torch.empty((num_tokens, args.num_topk), device="cuda"),
                torch.empty(
                    (num_tokens, args.num_topk), dtype=torch.int32, device="cuda"
                ),
                torch.empty((num_tokens, args.num_topk), device="cuda"),
            )

            def unfused_router():
                logits = run_gemm()
                return route_topk(
                    logits,
                    *buffers,
                    renormalize=True,
                    score_func="sqrtsoftplus",
                    correction_bias=bias,
                    routed_scaling_factor=args.scale,
                )

            unfused = median_us(unfused_router, args.cupti)
            row += f" {gemm:>9.2f} {unfused:>10.2f} {unfused / fused:>7.2f}x"
        print(row)


if __name__ == "__main__":
    main()
