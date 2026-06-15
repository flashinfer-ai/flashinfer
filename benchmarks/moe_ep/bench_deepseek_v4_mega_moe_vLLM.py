#!/usr/bin/env python3
"""Minimal multi-rank benchmark for vLLM DeepseekV4MegaMoEExperts forward.

Uses vLLM's native MegaMoE path end-to-end; the benchmark only supplies bf16
weights and activation/routing tensors via ``bench_deepseek_v4_mega_moe_common``.

  - EP layout: ``get_ep_group()`` (``DeepseekV4MoE._init_mega_moe_experts``)
  - Weights: ``DeepseekV4MegaMoEExperts.weight_loader`` + ``finalize_weights()``
  - Forward: ``experts(...)`` inside ``set_forward_context`` (custom op)

Launch (example, 4 GPUs on one node):
    torchrun --nproc_per_node=4 benchmarks/bench_deepseek_v4_mega_moe_experts.py

    torchrun --nproc_per_node=4 benchmarks/bench_deepseek_v4_mega_moe_experts.py \\
        --num-tokens 4096 --num-max-tokens 4096 --warmup 10 --repeat 50

    # vLLM-only gate or hash routing (not comparable to moe_ep_v2):
    torchrun --nproc_per_node=4 benchmarks/bench_deepseek_v4_mega_moe_experts.py \\
        --no-random-routing

Requires:
    - vLLM v0.20.0 with DeepSeek V4 MegaMoE support
    - deep_gemm with ``fp8_fp4_mega_moe``
    - SM100+ GPU per rank
"""

from __future__ import annotations

import os
import sys
from contextlib import AbstractContextManager
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

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
    routing_mode_from_args,
    VLLM_COLD_START_NOTE,
    VLLM_TIMING_MODE_NOTE,
    warn_kernel_token_sizing,
)

# ``DeepseekV4MoE._init_mega_moe_experts`` uses ``f"{prefix}.experts"``.
EXPERTS_PREFIX = "bench.ffn.experts"
GATE_PREFIX = "bench.ffn.gate"


def _init_vllm_distributed(
    local_rank: int,
    world_size: int,
    *,
    num_max_tokens: int,
) -> tuple[Any, AbstractContextManager[Any]]:
    from vllm.config import (
        CompilationConfig,
        KernelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

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

    parallel_config = ParallelConfig(
        tensor_parallel_size=world_size,
        pipeline_parallel_size=1,
        enable_expert_parallel=True,
        is_moe_model=True,
    )
    compilation_config = CompilationConfig()
    compilation_config.pass_config.fuse_allreduce_rms = False
    vllm_config = VllmConfig(
        parallel_config=parallel_config,
        kernel_config=KernelConfig(moe_backend="deep_gemm_mega_moe"),
        compilation_config=compilation_config,
        scheduler_config=SchedulerConfig.default_factory(
            max_num_batched_tokens=num_max_tokens,
        ),
    )

    config_ctx = set_current_vllm_config(vllm_config)
    config_ctx.__enter__()

    init_distributed_environment(
        world_size=world_size,
        rank=int(os.environ["RANK"]),
        distributed_init_method="env://",
        local_rank=local_rank,
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        pipeline_model_parallel_size=1,
    )
    return vllm_config, config_ctx


def _mega_moe_ep_layout(num_experts: int) -> tuple[int, int, int]:
    """Mirror ``DeepseekV4MoE._init_mega_moe_experts`` EP shard layout."""
    from vllm.distributed import get_ep_group

    ep_group = get_ep_group()
    num_local_experts = num_experts // ep_group.world_size
    experts_start_idx = ep_group.rank_in_group * num_local_experts
    return ep_group.rank_in_group, num_local_experts, experts_start_idx


def _build_mega_moe_experts(
    vllm_config,
    *,
    num_experts: int,
    num_local_experts: int,
    experts_start_idx: int,
    topk: int,
    hidden: int,
    intermediate: int,
    num_max_tokens: int,
):
    """Mirror ``DeepseekV4MoE._init_mega_moe_experts`` module construction."""
    from vllm.model_executor.models.deepseek_v4 import DeepseekV4MegaMoEExperts

    vllm_config.scheduler_config.max_num_batched_tokens = num_max_tokens

    return DeepseekV4MegaMoEExperts(
        vllm_config,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        experts_start_idx=experts_start_idx,
        top_k=topk,
        hidden_size=hidden,
        intermediate_size=intermediate,
        prefix=EXPERTS_PREFIX,
    )


def _float_ue8m0_scale_to_uint8(sf: torch.Tensor) -> torch.Tensor:
    """Checkpoint uint8 UE8M0 bytes (inverse of ``_ue8m0_uint8_to_float``)."""
    return ((sf.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)


def _load_benchmark_weights_into_experts(
    experts,
    weights: BenchmarkWeights,
    *,
    experts_start_idx: int,
    intermediate: int,
) -> None:
    """Drive vLLM's ``weight_loader`` + ``finalize_weights()`` from bf16 fixtures."""
    from vllm.third_party.deep_gemm.utils import per_token_cast_to_fp4

    experts.cuda()
    num_local_experts = weights.w13.shape[0]

    for local_expert_id in range(num_local_experts):
        global_expert_id = experts_start_idx + local_expert_id
        shard_specs = (
            ("w1", weights.w13[local_expert_id, :intermediate]),
            ("w3", weights.w13[local_expert_id, intermediate:]),
            ("w2", weights.w2[local_expert_id]),
        )
        for shard_id, bf16 in shard_specs:
            packed, scale = per_token_cast_to_fp4(
                bf16,
                use_ue8m0=True,
                gran_k=32,
            )
            weight_param = (
                experts.w13_weight if shard_id in ("w1", "w3") else experts.w2_weight
            )
            scale_param = (
                experts.w13_weight_scale
                if shard_id in ("w1", "w3")
                else experts.w2_weight_scale
            )
            weight_name = (
                "experts.w13_weight"
                if shard_id in ("w1", "w3")
                else "experts.w2_weight"
            )
            scale_name = (
                "experts.w13_weight_scale"
                if shard_id in ("w1", "w3")
                else "experts.w2_weight_scale"
            )
            experts.weight_loader(
                weight_param,
                packed.view(torch.uint8),
                weight_name,
                shard_id=shard_id,
                expert_id=global_expert_id,
            )
            experts.weight_loader(
                scale_param,
                _float_ue8m0_scale_to_uint8(scale),
                scale_name,
                shard_id=shard_id,
                expert_id=global_expert_id,
            )

    experts.finalize_weights()


def _build_gate(hidden: int, num_experts: int) -> nn.Module:
    from vllm.model_executor.layers.fused_moe import GateLinear

    gate = GateLinear(
        hidden,
        num_experts,
        out_dtype=torch.float32,
        bias=False,
        prefix=GATE_PREFIX,
    )
    gate.cuda()
    nn.init.normal_(gate.weight, std=0.02)
    return gate


def _make_vllm_e2e_routing_inputs(
    gate: nn.Module,
    rank: int,
    *,
    num_tokens: int,
    hidden: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    hash_moe: bool,
    hash_vocab_size: int,
) -> BenchmarkInputs:
    """Mirror ``DeepseekV4MoE.forward`` routing via vLLM APIs."""
    from bench_deepseek_v4_mega_moe_common import ROUTING_SEED_BASE
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        fused_topk_bias,
    )

    torch.manual_seed(ROUTING_SEED_BASE + rank)
    hidden_states = torch.randn(
        num_tokens, hidden, device="cuda", dtype=torch.bfloat16
    )
    router_logits, _ = gate(hidden_states)

    e_score_correction_bias = None
    hash_indices_table = None
    input_ids = None
    if hash_moe:
        hash_indices_table = torch.randint(
            0,
            num_experts,
            (hash_vocab_size, topk),
            device="cuda",
            dtype=torch.int64,
        )
        input_ids = torch.randint(
            0,
            hash_vocab_size,
            (num_tokens,),
            device="cuda",
            dtype=torch.int64,
        )
    else:
        e_score_correction_bias = torch.zeros(
            num_experts, device="cuda", dtype=torch.float32
        )

    topk_weights, topk_ids = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=router_logits,
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=e_score_correction_bias,
        topk=topk,
        renormalize=renormalize,
        indices_type=torch.int64,
        input_tokens=input_ids,
        hash_indices_table=hash_indices_table,
        routed_scaling_factor=routed_scaling_factor,
    )
    return BenchmarkInputs(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )


def _destroy_symm_buffers() -> None:
    from vllm.model_executor.models.deepseek_v4 import DeepseekV4MegaMoEExperts

    for symm_buffer in list(DeepseekV4MegaMoEExperts._symm_buffer_cache.values()):
        symm_buffer.destroy()
    DeepseekV4MegaMoEExperts._symm_buffer_cache.clear()


def main() -> int:
    args = parse_benchmark_args(
        description="Benchmark DeepseekV4MegaMoEExperts forward (multi-rank EP)."
    )
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

    vllm_config, config_ctx = _init_vllm_distributed(
        local_rank,
        world_size,
        num_max_tokens=args.num_max_tokens,
    )
    rank = dist.get_rank()
    exit_code = 0

    try:
        _, num_local_experts, experts_start_idx = _mega_moe_ep_layout(num_experts)

        experts = _build_mega_moe_experts(
            vllm_config,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            experts_start_idx=experts_start_idx,
            topk=args.topk,
            hidden=args.hidden,
            intermediate=args.intermediate,
            num_max_tokens=args.num_max_tokens,
        )
        bench_weights = make_benchmark_weights(
            rank,
            num_local_experts=num_local_experts,
            hidden=args.hidden,
            intermediate=args.intermediate,
        )
        _load_benchmark_weights_into_experts(
            experts,
            bench_weights,
            experts_start_idx=experts_start_idx,
            intermediate=args.intermediate,
        )

        if args.random_routing:
            inputs = make_shared_benchmark_inputs(
                rank,
                args,
                num_experts=num_experts,
            )
        else:
            gate = _build_gate(args.hidden, num_experts)
            inputs = _make_vllm_e2e_routing_inputs(
                gate,
                rank,
                num_tokens=args.num_tokens,
                hidden=args.hidden,
                num_experts=num_experts,
                topk=args.topk,
                renormalize=args.renormalize,
                routed_scaling_factor=args.routed_scaling_factor,
                hash_moe=args.hash_moe,
                hash_vocab_size=args.hash_vocab_size,
            )
        routing_mode = routing_mode_from_args(args)

        from vllm.forward_context import set_forward_context

        num_tokens = inputs.hidden_states.shape[0]

        def run_once() -> torch.Tensor:
            with set_forward_context(None, vllm_config, num_tokens=num_tokens):
                return experts(
                    inputs.hidden_states,
                    inputs.topk_weights,
                    inputs.topk_ids,
                    activation_clamp=activation_clamp,
                    fast_math=args.fast_math,
                )

        timing = bench_forward_ms(
            run_once,
            warmup=args.warmup,
            repeat=args.repeat,
            cold_start=args.cold_start,
        )

        print_benchmark_header(
            title="DeepseekV4MegaMoEExperts forward benchmark",
            rank=rank,
            world_size=world_size,
            num_local_experts=num_local_experts,
            args=args,
            num_experts=num_experts,
            activation_clamp=activation_clamp,
            routing_mode=routing_mode,
            timing_mode_note=VLLM_TIMING_MODE_NOTE,
        )
        print_benchmark_timing(
            timing,
            rank=rank,
            cold_start_note=VLLM_COLD_START_NOTE,
        )
    except Exception:
        exit_code = 1
        raise
    finally:
        _destroy_symm_buffers()
        config_ctx.__exit__(None, None, None)
        if dist.is_initialized():
            dist.destroy_process_group()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
