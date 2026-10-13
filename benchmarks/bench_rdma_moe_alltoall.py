# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Benchmark SM120 RDMA MoE dispatch and combine with torchrun.

Examples (matching NCCL 2.30.7 headers/runtime and GDAKI are required):
  NCCL_GIN_TYPE=3 torchrun --standalone --nproc-per-node=4 \
      benchmarks/bench_rdma_moe_alltoall.py --dtype nvfp4 --tokens 1 128 4096
  NCCL_GIN_TYPE=3 torchrun --standalone --nproc-per-node=4 \
      benchmarks/bench_rdma_moe_alltoall.py --dtype bf16 \
      --combine-reduction bf16_local_partial_planned \
      --communication-mode coalesced --tokens 1 128 4096

Dispatch takes BF16 inputs; NVFP4 quantization is included in its timing.
Combine uses synthetic BF16 expert outputs; quantization, communication and
reduction are timed, while the initial dispatch and expert GEMM are excluded.
Weights are fixed at 1/top_k. Planned BF16 snapshots routes/weights in dispatch
and rounds weighted leaves, owner partials and final output to BF16; it is not
numerically identical to the default FP32 reduction. Include dispatch cost when
assessing the benefit of its precomputed plan.

CUDA-event Graph replay spans include GPU waiting and inter-rank launch skew,
not just individual kernel durations. Each sample is the slowest rank, and each
captured call is a pair of operations to preserve the buffer's alternating slot
phase. Inputs and output allocations are reused (hot-cache measurement).
Rank zero prints JSON lines; redirect stdout to save results. Correctness and
dynamic Graph coverage belong to tests/comm/rdma_moe_alltoall/.
"""

import argparse
import json
import math
import os
import statistics
from datetime import timedelta

import torch
import torch.distributed as dist

from flashinfer.comm import RdmaMoeAlltoAll, RdmaMoeAlltoAllConfig
from flashinfer.testing import bench_gpu_time


CONFIG_FIELDS = (
    "communication_mode",
    "dispatch_mode",
    "combine_mode",
    "num_sms",
    "num_contexts",
    "dispatch_num_sms",
    "dispatch_num_contexts",
    "combine_num_sms",
    "combine_num_contexts",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("bf16", "nvfp4"), default="nvfp4")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 128, 4096])
    parser.add_argument("--hidden", type=int, choices=(4096, 6144, 7168), default=4096)
    parser.add_argument("--experts", type=int, default=288)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=("dispatch", "combine"),
        default=["dispatch", "combine"],
    )
    parser.add_argument(
        "--combine-reduction",
        choices=("fp32", "bf16_local_partial_planned"),
        default="fp32",
    )
    parser.add_argument(
        "--communication-mode", choices=("auto", "legacy", "coalesced"), default="auto"
    )
    for name in ("dispatch_mode", "combine_mode"):
        parser.add_argument(
            "--" + name.replace("_", "-"), choices=("legacy", "coalesced")
        )
    for name in CONFIG_FIELDS[3:]:
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument("--graph-calls", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if any(t < 1 or t > 4096 for t in args.tokens):
        parser.error("--tokens must be in [1, 4096]")
    if args.experts < 1 or not 1 <= args.topk <= min(8, args.experts):
        parser.error("require positive --experts and 1 <= --topk <= min(8, experts)")
    if args.graph_calls < 2 or args.graph_calls % 2:
        parser.error("--graph-calls must be positive and even")
    if args.warmup < 0 or args.repeats < 1:
        parser.error("require --warmup >= 0 and --repeats >= 1")
    if any(
        getattr(args, name) is not None and getattr(args, name) < 1
        for name in CONFIG_FIELDS[3:]
    ):
        parser.error("SM and context budgets must be positive")
    if args.communication_mode == "auto" and any(
        getattr(args, name) is not None for name in CONFIG_FIELDS[1:]
    ):
        parser.error(
            "explicit launch settings require --communication-mode legacy or coalesced"
        )
    if args.combine_reduction != "fp32" and args.dtype != "bf16":
        parser.error("planned reduction requires --dtype bf16")
    return args


def measure(buffer, x, routes, weights, operation, args):
    planned = args.combine_reduction == "bf16_local_partial_planned"
    dispatch_options = {"topk_weights": weights} if planned else {}
    initial = buffer.dispatch(x, routes, **dispatch_options)
    if operation == "dispatch":
        # Reuse public outputs; dispatch still constructs its handle and plan.
        outputs = (initial.data, initial.scales, initial.counts, *initial.handle[:2])

        def call():
            buffer.dispatch(x, routes, out=outputs, **dispatch_options)

    else:
        handle = initial.handle
        rows = buffer.num_ranks * buffer.num_max_dispatch_tokens_per_rank
        shape = (args.experts // buffer.num_ranks, rows, args.hidden)
        # Only the handle is needed from setup dispatch. Release its payload
        # before allocating expert outputs, especially for large capacities.
        del initial
        expert_x = torch.full(shape, 1.5, dtype=torch.bfloat16, device=x.device)
        options = {} if planned else {"topk_idx": routes, "topk_weights": weights}
        if args.dtype == "nvfp4":
            options["global_scale"] = torch.full(
                (*shape[:2], 1), 256.0, dtype=torch.float32, device=x.device
            )
        output = torch.empty_like(x)

        def call():
            buffer.combine(expert_x, handle, out=output, **options)

    def pair():
        call()
        call()

    # CUPTI kernel sums omit inter-kernel waiting. Use the shared event helper
    # for a complete API span, fixed replay counts and sample-wise rank maximum.
    # Pairing also keeps its three warmup invocations at an even operation count.
    dist.barrier()
    samples = bench_gpu_time(
        pair,
        dry_run_iters=args.warmup,
        repeat_iters=args.repeats,
        use_cuda_graph=True,
        num_iters_within_graph=args.graph_calls // 2,
        enable_cupti=False,
        cold_l2_cache=False,
        aggregate_op=max,
    )
    samples_us = [ms * 1000 / 2 for ms in samples]
    return {
        "operation": operation,
        "p50_us": statistics.median(samples_us),
        "p95_us": sorted(samples_us)[math.ceil(0.95 * len(samples_us)) - 1],
        "max_rank_samples_us": samples_us,
    }


def main():
    args = parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=timedelta(minutes=10))
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        if args.experts % world:
            raise ValueError("--experts must be divisible by the number of ranks")
        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(args.seed + rank)
        config = RdmaMoeAlltoAllConfig(
            communication_dtype=args.dtype,
            combine_reduction=args.combine_reduction,
            **{name: getattr(args, name) for name in CONFIG_FIELDS},
        )
        for tokens in args.tokens:
            x = torch.randn(
                (tokens, args.hidden),
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )
            routes = (
                torch.rand((tokens, args.experts), generator=generator, device=device)
                .topk(args.topk, dim=-1)
                .indices.to(torch.int32)
            )
            weights = torch.full(
                routes.shape, 1 / args.topk, dtype=torch.float32, device=device
            )
            buffer = RdmaMoeAlltoAll(
                dist.group.WORLD, tokens, args.hidden, args.experts, config=config
            )
            try:
                for operation in args.operations:
                    measured = measure(buffer, x, routes, weights, operation, args)
                    if rank == 0:
                        print(
                            json.dumps(
                                {
                                    "dtype": args.dtype,
                                    "tokens_per_rank": tokens,
                                    "hidden": args.hidden,
                                    "experts": args.experts,
                                    "topk": args.topk,
                                    "world_size": world,
                                    "gpu": torch.cuda.get_device_name(device),
                                    "combine_reduction": args.combine_reduction,
                                    "input_dtype": "bf16",
                                    "weights": "uniform_1_over_topk",
                                    "timing": "cuda_graph_replay_api_span_max_rank",
                                    "graph_calls": args.graph_calls,
                                    "warmup": args.warmup,
                                    "repeats": args.repeats,
                                    "seed": args.seed,
                                    "backend_info": buffer.backend_info(),
                                    **measured,
                                }
                            ),
                            flush=True,
                        )
            finally:
                torch.cuda.synchronize(device)
                buffer.close()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
