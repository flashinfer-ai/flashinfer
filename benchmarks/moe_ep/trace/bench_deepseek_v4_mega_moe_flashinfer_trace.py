#!/usr/bin/env python3
"""NVTX-instrumented flashinfer MoEEpMegaLayer forward benchmark for nsys.

Same problem/CLI as ``bench_deepseek_v4_mega_moe_flashinfer.py`` with NVTX
ranges for ``bench_deepseek_v4_mega_moe_nsys.sh`` (--capture-range=nvtx).

Launch (example, 4 GPUs on one node):
    torchrun --nproc_per_node=4 \\
        benchmarks/moe_ep/trace/bench_deepseek_v4_mega_moe_flashinfer_trace.py

Profile steady forwards only:
    bash benchmarks/moe_ep/trace/bench_deepseek_v4_mega_moe_nsys.sh nvtx flashinfer
"""

from __future__ import annotations

import os
import sys

_TRACE_DIR = os.path.dirname(os.path.abspath(__file__))
_BENCH_DIR = os.path.dirname(_TRACE_DIR)
_REPO_ROOT = os.path.abspath(os.path.join(_BENCH_DIR, "..", ".."))
for path in (_REPO_ROOT, _BENCH_DIR, _TRACE_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
import torch.distributed as dist

from bench_deepseek_v4_mega_moe_common import (
    BenchmarkInputs,
    BenchmarkWeights,
    activation_clamp_from_args,
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
from bench_deepseek_v4_mega_moe_trace_common import (
    NVTX_SETUP,
    bench_forward_ms_nvtx,
    nvtx_range,
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
    from flashinfer.moe_ep import (
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
            megakernel=DeepGemmMegaMoeConfig(
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
    from flashinfer.moe_ep import MoEEpTensors

    return MoEEpTensors(
        hidden_states=inputs.hidden_states,
        topk_ids=inputs.topk_ids,
        topk_weights=inputs.topk_weights,
    )


def main() -> int:
    args = parse_benchmark_args(
        description=(
            "NVTX trace benchmark for MoEEpMegaLayer forward "
            "(multi-rank EP, moe_ep)."
        )
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

    mega = None
    exit_code = 0
    rank = local_rank
    num_local_experts = args.num_local_experts

    try:
        with nvtx_range(NVTX_SETUP):
            _init_distributed(local_rank, world_size)
            rank = dist.get_rank()
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

        timing = bench_forward_ms_nvtx(
            run_once,
            warmup=args.warmup,
            repeat=args.repeat,
            cold_start=args.cold_start,
        )

        print_benchmark_header(
            title=(
                "MoEEpMegaLayer forward trace benchmark "
                "(flashinfer.moe_ep, NVTX)"
            ),
            rank=rank,
            world_size=world_size,
            num_local_experts=num_local_experts,
            args=args,
            num_experts=num_experts,
            activation_clamp=activation_clamp,
            routing_mode=routing_mode_from_args(args),
            extra_lines=(
                "  stage_inputs=True preprocess_weights=True",
                "  nvtx_ranges=setup,cold_start,warmup,steady_capture,forward",
            ),
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
