#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "flashinfer-python"]
# [tool.uv.sources]
# flashinfer-python = { path = "..", editable = true }
# ///
"""Compare TRT-LLM DeepSeek FP8 activation dispatch with its scalar fallback.

Run in an environment with this source checkout installed:
    uv run benchmarks/bench_trtllm_fp8_activation.py --check-only
    uv run benchmarks/bench_trtllm_fp8_activation.py \
        --num-tokens 16384 --intermediate-size 512 --top-k 8 --num-experts 288

The one-byte-offset output selects the original scalar kernel. Timings use
warm-cache CUDA graphs and exclude routing, GEMMs, and buffer allocation.
The staged activation can run on SM120; this does not validate the full MoE path.
"""

import argparse
import statistics

import torch

from flashinfer.fused_moe.core import ActivationType
from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module
from flashinfer.testing import bench_gpu_time


def run_case(module, args, num_tokens, intermediate_size):
    expanded_rows = num_tokens * args.top_k
    rows_per_expert = (expanded_rows + args.num_experts - 1) // args.num_experts
    padded_rows_per_expert = (rows_per_expert + 127) // 128 * 128
    padded_rows = padded_rows_per_expert * args.num_experts
    blocks = intermediate_size // 128
    row = torch.arange(expanded_rows, device="cuda", dtype=torch.int32)
    mapping = row % args.num_experts * padded_rows_per_expert + row // args.num_experts
    mapping[row % 17 == 0] = -1
    used_rows = mapping[mapping >= 0].long()
    active = torch.zeros(padded_rows, device="cuda", dtype=torch.bool)
    active[used_rows] = True
    inputs = (
        torch.randn(padded_rows, intermediate_size * 2, device="cuda")
        .clamp_(-4, 4)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    if used_rows.numel():
        inputs[used_rows[0]] = 0
    input_scales = torch.rand(blocks * 2, padded_rows, device="cuda") * 0.25 + 0.01
    total = torch.tensor([padded_rows], device="cuda", dtype=torch.int32)
    output = torch.full(
        (padded_rows, intermediate_size), 0x55, device="cuda", dtype=torch.uint8
    )
    scalar_storage = torch.full(
        (output.numel() + 1,), 0x55, device="cuda", dtype=torch.uint8
    )
    scalar_output = scalar_storage[1:].view_as(output)
    scales = torch.full((blocks, padded_rows), -1.0, device="cuda")
    scalar_scales = torch.full_like(scales, -1.0)

    def launch(out, out_scales):
        module.trtllm_moe_run_deepseek_fp8_activation(
            inputs,
            input_scales,
            out,
            out_scales,
            mapping,
            total,
            num_tokens,
            args.top_k,
            intermediate_size,
            int(ActivationType.Swiglu),
            args.enable_pdl,
        )

    launch(scalar_output, scalar_scales)
    launch(output, scales)
    assert torch.equal(output, scalar_output), "FP8 output differs from scalar fallback"
    assert torch.equal(scales, scalar_scales), "Scales differ from scalar fallback"
    assert torch.all(output[~active] == 0x55), "Inactive output rows were written"
    assert torch.all(scales[:, ~active] == -1), "Inactive scale rows were written"

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(output, scales)
    output.fill_(0x55)
    scales.fill_(-1)
    graph.replay()
    assert torch.equal(output, scalar_output), "CUDA graph output differs"
    assert torch.equal(scales, scalar_scales), "CUDA graph scales differ"
    label = (
        f"tokens={num_tokens} intermediate={intermediate_size} pdl={args.enable_pdl}"
    )
    if args.check_only:
        print(f"{label}: bitwise parity, inactive rows, and graph replay passed")
        return

    def measure(out, out_scales):
        return (
            statistics.median(
                bench_gpu_time(
                    lambda: launch(out, out_scales),
                    dry_run_iters=10,
                    repeat_iters=50,
                    use_cuda_graph=True,
                    cold_l2_cache=False,
                )
            )
            * 1000
        )

    scalar_us = measure(scalar_output, scalar_scales)
    dispatch_us = measure(output, scales)
    print(
        f"{label}: scalar={scalar_us:.2f} us dispatch={dispatch_us:.2f} us "
        f"speedup={scalar_us / dispatch_us:.2f}x"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tokens", type=int, nargs="+", default=[1023, 1024, 1025])
    parser.add_argument("--intermediate-size", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--enable-pdl", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if min(args.num_tokens) <= 0 or min(args.intermediate_size) <= 0:
        parser.error("Token counts and intermediate sizes must be positive")
    if any(size % 128 for size in args.intermediate_size):
        parser.error("Intermediate sizes must be divisible by 128")
    if not 0 < args.top_k <= args.num_experts:
        parser.error("Require 0 < top-k <= num-experts")
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name()}; CUDA: {torch.version.cuda}")
    module = gen_trtllm_gen_fused_moe_sm100_module().build_and_load()
    for num_tokens in args.num_tokens:
        for intermediate_size in args.intermediate_size:
            run_case(module, args, num_tokens, intermediate_size)


if __name__ == "__main__":
    main()
