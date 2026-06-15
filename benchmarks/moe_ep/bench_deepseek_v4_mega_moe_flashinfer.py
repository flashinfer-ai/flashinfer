#!/usr/bin/env python3
"""Minimal multi-rank benchmark for flashinfer MoEEpMegaLayer forward.

Uses ``flashinfer.moe_ep_v2`` MegaMoE APIs end-to-end; the benchmark only
supplies bf16 weights and activation/routing tensors via
``bench_deepseek_v4_mega_moe_common``.

  - Layer: ``MoEEpLayer`` → ``MoEEpMegaLayer``
  - Weights: ``MoEWeightPack`` (preprocessed inside the layer)
  - Forward: ``MoEEpMegaLayer.forward(MoEEpTensors)``

Launch (example, 4 GPUs on one node):
    torchrun --nproc_per_node=4 \\
        benchmarks/bench_deepseek_v4_mega_moe_experts_moe_ep_v2.py

    torchrun --nproc_per_node=4 \\
        benchmarks/bench_deepseek_v4_mega_moe_experts_moe_ep_v2.py \\
        --num-tokens 4096 --num-max-tokens 4096 --warmup 10 --repeat 50

Use the same CLI flags as bench_deepseek_v4_mega_moe_experts.py for comparable runs.

Requires:
    - flashinfer with ``moe_ep_v2`` mega path
    - deep_gemm with ``fp8_fp4_mega_moe``
    - SM100+ GPU per rank
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

import torch
import torch.distributed as dist

from bench_deepseek_v4_mega_moe_common import (
    BenchmarkInputs,
    BenchmarkWeights,
    activation_clamp_from_args,
    bench_forward_ms,
    check_problem,
    make_benchmark_weights,
    make_shared_benchmark_inputs,
    parse_benchmark_args,
    print_benchmark_header,
    print_benchmark_timing,
    require_env_rank,
    require_shared_routing_benchmark,
    routing_mode_from_args,
    MOE_EP_V2_COLD_START_NOTE,
    MOE_EP_V2_TIMING_MODE_NOTE,
    warn_kernel_token_sizing,
)


def _init_distributed(local_rank: int, world_size: int) -> None:
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=int(os.environ["RANK"]),
            device_id=device,
        )


def _build_mega_moe_layer(
    rank: int,
    world_size: int,
    *,
    num_experts: int,
    topk: int,
    hidden: int,
    intermediate: int,
    num_max_tokens: int,
    weights: BenchmarkWeights,
    activation_clamp: float | None,
    fast_math: bool,
):
    from flashinfer.moe_ep_v2 import (
        BootstrapConfig,
        DeepGemmMegaMoeConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpMegaLayer,
        MoEWeightPack,
    )

    mega = MoEEpLayer(
        bootstrap=BootstrapConfig(world_size=world_size, rank=rank),
        fleet_params=FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=num_max_tokens,
            token_hidden_size=hidden,
            weights=MoEWeightPack(w13=weights.w13, w2=weights.w2),
        ),
        backend=MegaConfig(
            kernel=DeepGemmMegaMoeConfig(
                intermediate_size=intermediate,
                top_k=topk,
                activation_clamp=activation_clamp,
                fast_math=fast_math,
            ),
            stage_inputs=True,
            preprocess_weights=True,
        ),
    )
    assert isinstance(mega, MoEEpMegaLayer)
    return mega


def _to_moe_ep_tensors(inputs: BenchmarkInputs):
    from flashinfer.moe_ep_v2 import MoEEpTensors

    return MoEEpTensors(
        hidden_states=inputs.hidden_states,
        topk_ids=inputs.topk_ids,
        topk_weights=inputs.topk_weights,
    )


def main() -> int:
    args = parse_benchmark_args(
        description="Benchmark MoEEpMegaLayer forward (multi-rank EP, moe_ep_v2)."
    )
    require_shared_routing_benchmark(args)
    local_rank, world_size = require_env_rank()

    num_experts = args.num_local_experts * world_size
    check_problem(
        args.hidden,
        args.intermediate,
        args.num_tokens,
        args.num_max_tokens,
        num_experts,
        world_size,
    )
    warn_kernel_token_sizing(
        rank=local_rank,
        num_tokens=args.num_tokens,
        num_max_tokens=args.num_max_tokens,
    )

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required", file=sys.stderr)
        return 1
    capability = torch.cuda.get_device_capability(local_rank)
    if capability[0] != 10:
        print(
            f"ERROR: rank {local_rank} needs SM100 (Blackwell), "
            f"got sm_{capability[0]}{capability[1]}",
            file=sys.stderr,
        )
        return 1

    activation_clamp = activation_clamp_from_args(args)

    _init_distributed(local_rank, world_size)
    rank = dist.get_rank()
    exit_code = 0
    mega = None

    try:
        num_local_experts = args.num_local_experts
        bench_weights = make_benchmark_weights(
            rank,
            num_local_experts=num_local_experts,
            hidden=args.hidden,
            intermediate=args.intermediate,
        )
        mega = _build_mega_moe_layer(
            rank,
            world_size,
            num_experts=num_experts,
            topk=args.topk,
            hidden=args.hidden,
            intermediate=args.intermediate,
            num_max_tokens=args.num_max_tokens,
            weights=bench_weights,
            activation_clamp=activation_clamp,
            fast_math=args.fast_math,
        )

        inputs = make_shared_benchmark_inputs(
            rank,
            args,
            num_experts=num_experts,
        )
        tensors = _to_moe_ep_tensors(inputs)

        def run_once() -> torch.Tensor:
            return mega.forward(tensors)

        timing = bench_forward_ms(
            run_once,
            warmup=args.warmup,
            repeat=args.repeat,
            cold_start=args.cold_start,
        )

        print_benchmark_header(
            title="MoEEpMegaLayer forward benchmark (flashinfer.moe_ep_v2)",
            rank=rank,
            world_size=world_size,
            num_local_experts=num_local_experts,
            args=args,
            num_experts=num_experts,
            activation_clamp=activation_clamp,
            routing_mode=routing_mode_from_args(args),
            extra_lines=("  stage_inputs=True preprocess_weights=True",),
            timing_mode_note=MOE_EP_V2_TIMING_MODE_NOTE,
        )
        print_benchmark_timing(
            timing,
            rank=rank,
            cold_start_note=MOE_EP_V2_COLD_START_NOTE,
        )
    except Exception:
        exit_code = 1
        raise
    finally:
        if mega is not None:
            mega.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
