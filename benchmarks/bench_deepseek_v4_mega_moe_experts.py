#!/usr/bin/env python3
"""Minimal multi-rank benchmark for vLLM DeepseekV4MegaMoEExperts forward.

Targets vLLM v0.20.0 on Blackwell (SM100) with expert parallelism and the
``deep_gemm.fp8_fp4_mega_moe`` path behind ``DeepseekV4MegaMoEExperts``.

Mirrors the e2e DeepSeek V4 MegaMoE expert path in ``deepseek_v4.py``:
  - ``enable_expert_parallel=True`` + ``kernel_config.moe_backend="deep_gemm_mega_moe"``
  - EP layout from ``get_ep_group()`` (``DeepseekV4MoE._init_mega_moe_experts``)
  - Expert weights loaded per-shard via ``weight_loader`` (w1/w3/w2 + scales),
    then ``finalize_weights()`` as in ``DeepseekV4Model.finalize_mega_moe_weights``
  - Routing inputs from ``GateLinear`` + ``fused_topk_bias(sqrtsoftplus)`` when
    not using ``--random-routing`` (same as ``DeepseekV4MoE.forward``)
  - ``experts.forward()`` through ``deepseek_v4_mega_moe_experts`` inside
    ``set_forward_context`` (populates ``no_compile_layers[prefix]``)

Timing modes:
  - Default (``--warmup N`` then ``--repeat M``): **steady-state** latency. Warmup
    amortizes one-time costs that e2e also pays once per process/layer, then reuses:
    ``DeepseekV4MegaMoEExperts._symm_buffer_cache`` (``get_symm_buffer()``),
    ``_transformed_l{1,2}_weights`` (``finalize_weights()`` no-op), and Triton/JIT
    for ``_stage_deepseek_v4_mega_moe_inputs``. This matches sustained inference
    where the symm buffer lives for the engine lifetime.
  - ``--cold-start``: additionally time the **first** forward before warmup (includes
    symm-buffer allocation and any first-kernel compile).

Launch (example, 4 GPUs on one node):
    torchrun --nproc_per_node=4 benchmarks/bench_deepseek_v4_mega_moe_experts.py

    torchrun --nproc_per_node=4 benchmarks/bench_deepseek_v4_mega_moe_experts.py \\
        --num-tokens 4096 --warmup 10 --repeat 50

Requires:
    - vLLM v0.20.0 with DeepSeek V4 MegaMoE support
    - deep_gemm with ``fp8_fp4_mega_moe``
    - SM100+ GPU per rank
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

# -------------------- default problem size (matches dummy_fp8_fp4_mega_moe.py) ----
DEFAULT_HIDDEN = 4096
DEFAULT_INTERMEDIATE = 2048
DEFAULT_NUM_TOKENS = 8192
DEFAULT_NUM_MAX_TOKENS = 8192
DEFAULT_NUM_LOCAL_EXPERTS = 64
DEFAULT_NUM_TOPK = 6
DEFAULT_ACTIVATION_CLAMP = 10.0
DEFAULT_FAST_MATH = True
DEFAULT_RENORMALIZE = True
DEFAULT_ROUTED_SCALING_FACTOR = 1.0
DEFAULT_HASH_VOCAB_SIZE = 129280

# ``DeepseekV4MoE._init_mega_moe_experts`` uses ``f"{prefix}.experts"``.
EXPERTS_PREFIX = "bench.ffn.experts"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DeepseekV4MegaMoEExperts forward (multi-rank EP)."
    )
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--intermediate", type=int, default=DEFAULT_INTERMEDIATE)
    parser.add_argument("--num-tokens", type=int, default=DEFAULT_NUM_TOKENS)
    parser.add_argument("--num-max-tokens", type=int, default=DEFAULT_NUM_MAX_TOKENS)
    parser.add_argument(
        "--num-local-experts",
        type=int,
        default=DEFAULT_NUM_LOCAL_EXPERTS,
        help="Experts owned by each EP rank.",
    )
    parser.add_argument("--topk", type=int, default=DEFAULT_NUM_TOPK)
    parser.add_argument(
        "--activation-clamp",
        type=float,
        default=DEFAULT_ACTIVATION_CLAMP,
        help="Swiglu clamp (e2e: float(config.swiglu_limit) or None).",
    )
    parser.add_argument(
        "--no-activation-clamp",
        action="store_true",
        help="Pass activation_clamp=None like swiglu_limit=None in e2e.",
    )
    parser.add_argument(
        "--fast-math",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FAST_MATH,
    )
    parser.add_argument(
        "--renormalize",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RENORMALIZE,
        help="norm_topk_prob passed to fused_topk_bias.",
    )
    parser.add_argument(
        "--routed-scaling-factor",
        type=float,
        default=DEFAULT_ROUTED_SCALING_FACTOR,
    )
    parser.add_argument(
        "--random-routing",
        action="store_true",
        help="Use random topk instead of GateLinear+fused_topk_bias.",
    )
    parser.add_argument(
        "--hash-moe",
        action="store_true",
        help="Hash-MoE routing (early layers): tid2eid + input_ids.",
    )
    parser.add_argument(
        "--hash-vocab-size",
        type=int,
        default=DEFAULT_HASH_VOCAB_SIZE,
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--cold-start",
        action="store_true",
        help="Also time the first forward before warmup (symm alloc + cold JIT).",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class BenchTiming:
    """Forward latency breakdown."""

    steady_avg_ms: float
    steady_first_ms: float
    steady_min_ms: float
    steady_max_ms: float
    cold_start_ms: float | None


def _require_env_rank() -> tuple[int, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    return local_rank, world_size


def _check_problem(
    hidden: int,
    intermediate: int,
    num_tokens: int,
    num_max_tokens: int,
    num_experts: int,
    world_size: int,
) -> None:
    if hidden % 128 != 0 or intermediate % 128 != 0:
        raise ValueError("hidden and intermediate must be multiples of 128")
    if num_tokens > num_max_tokens:
        raise ValueError("num_tokens must be <= num_max_tokens")
    if num_experts % world_size != 0:
        raise ValueError("num_experts must be divisible by world_size")


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


def _get_ep_layout(num_experts: int) -> tuple[int, int, int]:
    from vllm.distributed import get_ep_group

    ep_group = get_ep_group()
    ep_rank = ep_group.rank_in_group
    num_local_experts = num_experts // ep_group.world_size
    experts_start_idx = ep_rank * num_local_experts
    return ep_rank, num_local_experts, experts_start_idx


def _float_ue8m0_scale_to_uint8(sf: torch.Tensor) -> torch.Tensor:
    """Match checkpoint uint8 UE8M0 bytes (_ue8m0_uint8_to_float inverse)."""
    return ((sf.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)


def _load_dummy_expert_weights(
    experts,
    rank: int,
    *,
    experts_start_idx: int,
    num_local_experts: int,
    hidden: int,
    intermediate: int,
) -> None:
    """Load w1/w3/w2 shards via weight_loader, then finalize_weights (e2e path)."""
    from vllm.third_party.deep_gemm.utils import per_token_cast_to_fp4

    experts.cuda()
    torch.manual_seed(0 + rank)

    for local_expert_id in range(num_local_experts):
        global_expert_id = experts_start_idx + local_expert_id

        shard_specs = [
            (
                "w1",
                torch.randn(intermediate, hidden, device="cuda", dtype=torch.bfloat16),
                experts.w13_weight,
                experts.w13_weight_scale,
                "experts.w13_weight",
                "experts.w13_weight_scale",
            ),
            (
                "w3",
                torch.randn(intermediate, hidden, device="cuda", dtype=torch.bfloat16),
                experts.w13_weight,
                experts.w13_weight_scale,
                "experts.w13_weight",
                "experts.w13_weight_scale",
            ),
            (
                "w2",
                torch.randn(hidden, intermediate, device="cuda", dtype=torch.bfloat16),
                experts.w2_weight,
                experts.w2_weight_scale,
                "experts.w2_weight",
                "experts.w2_weight_scale",
            ),
        ]

        for (
            shard_id,
            bf16,
            weight_param,
            scale_param,
            weight_name,
            scale_name,
        ) in shard_specs:
            packed, scale = per_token_cast_to_fp4(
                bf16,
                use_ue8m0=True,
                gran_k=32,
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


def _make_random_routing_inputs(
    rank: int,
    *,
    num_tokens: int,
    hidden: int,
    num_experts: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1 + rank)
    hidden_states = torch.randn(
        num_tokens, hidden, device="cuda", dtype=torch.bfloat16
    )
    scores = torch.randn(
        num_tokens, num_experts, device="cuda", dtype=torch.float32
    )
    topk_weights, topk_ids = torch.topk(
        scores, topk, dim=-1, largest=True, sorted=False
    )
    return (
        hidden_states,
        topk_weights.to(torch.float32),
        topk_ids.to(torch.int64),
    )


def _build_gate(hidden: int, num_experts: int) -> nn.Module:
    from vllm.model_executor.layers.fused_moe import GateLinear

    gate = GateLinear(
        hidden,
        num_experts,
        out_dtype=torch.float32,
        bias=False,
        prefix="bench.ffn.gate",
    )
    gate.cuda()
    nn.init.normal_(gate.weight, std=0.02)
    return gate


def _make_e2e_routing_inputs(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror DeepseekV4MoE.forward routing before experts()."""
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        fused_topk_bias,
    )

    torch.manual_seed(1 + rank)
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
    return hidden_states, topk_weights, topk_ids


def _build_experts(
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


def _bench_forward_ms(
    experts,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    vllm_config,
    *,
    activation_clamp: float | None,
    fast_math: bool,
    warmup: int,
    repeat: int,
    cold_start: bool,
) -> BenchTiming:
    from vllm.forward_context import set_forward_context

    num_tokens = hidden_states.shape[0]

    def run_once() -> torch.Tensor:
        with set_forward_context(None, vllm_config, num_tokens=num_tokens):
            return experts(
                hidden_states,
                topk_weights,
                topk_ids,
                activation_clamp=activation_clamp,
                fast_math=fast_math,
            )

    def time_once() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_once()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    cold_start_ms = time_once() if cold_start else None

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    elapsed_ms = [time_once() for _ in range(repeat)]

    return BenchTiming(
        steady_avg_ms=sum(elapsed_ms) / len(elapsed_ms),
        steady_first_ms=elapsed_ms[0],
        steady_min_ms=min(elapsed_ms),
        steady_max_ms=max(elapsed_ms),
        cold_start_ms=cold_start_ms,
    )


def _destroy_symm_buffers() -> None:
    from vllm.model_executor.models.deepseek_v4 import DeepseekV4MegaMoEExperts

    for symm_buffer in list(DeepseekV4MegaMoEExperts._symm_buffer_cache.values()):
        symm_buffer.destroy()
    DeepseekV4MegaMoEExperts._symm_buffer_cache.clear()


def main() -> int:
    args = _parse_args()
    local_rank, world_size = _require_env_rank()

    num_experts = args.num_local_experts * world_size
    _check_problem(
        args.hidden,
        args.intermediate,
        args.num_tokens,
        args.num_max_tokens,
        num_experts,
        world_size,
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

    activation_clamp = (
        None if args.no_activation_clamp else args.activation_clamp
    )

    vllm_config, config_ctx = _init_vllm_distributed(
        local_rank,
        world_size,
        num_max_tokens=args.num_max_tokens,
    )
    rank = dist.get_rank()
    exit_code = 0

    try:
        _, num_local_experts, experts_start_idx = _get_ep_layout(num_experts)

        experts = _build_experts(
            vllm_config,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            experts_start_idx=experts_start_idx,
            topk=args.topk,
            hidden=args.hidden,
            intermediate=args.intermediate,
            num_max_tokens=args.num_max_tokens,
        )
        _load_dummy_expert_weights(
            experts,
            rank,
            experts_start_idx=experts_start_idx,
            num_local_experts=num_local_experts,
            hidden=args.hidden,
            intermediate=args.intermediate,
        )

        if args.random_routing:
            hidden_states, topk_weights, topk_ids = _make_random_routing_inputs(
                rank,
                num_tokens=args.num_tokens,
                hidden=args.hidden,
                num_experts=num_experts,
                topk=args.topk,
            )
            routing_mode = "random"
        else:
            gate = _build_gate(args.hidden, num_experts)
            hidden_states, topk_weights, topk_ids = _make_e2e_routing_inputs(
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
            routing_mode = "hash-moe" if args.hash_moe else "gate+sqrtsoftplus"

        timing = _bench_forward_ms(
            experts,
            hidden_states,
            topk_weights,
            topk_ids,
            vllm_config,
            activation_clamp=activation_clamp,
            fast_math=args.fast_math,
            warmup=args.warmup,
            repeat=args.repeat,
            cold_start=args.cold_start,
        )

        if rank == 0:
            print("DeepseekV4MegaMoEExperts forward benchmark")
            print(f"  world_size={world_size} ep_local_experts={num_local_experts}")
            print(
                f"  hidden={args.hidden} intermediate={args.intermediate} "
                f"num_experts={num_experts} topk={args.topk}"
            )
            print(
                f"  num_tokens={args.num_tokens} num_max_tokens={args.num_max_tokens} "
                f"clamp={activation_clamp} fast_math={args.fast_math}"
            )
            print(
                f"  routing={routing_mode} renormalize={args.renormalize} "
                f"routed_scaling_factor={args.routed_scaling_factor}"
            )
            print(f"  warmup={args.warmup} repeat={args.repeat}")
            print(
                "  timing_mode=steady-state "
                "(symm_buffer + transformed_weights cached; Triton JIT warm)"
            )
            print(
                f"  steady_avg_ms={timing.steady_avg_ms:.3f} "
                f"first={timing.steady_first_ms:.3f} "
                f"min={timing.steady_min_ms:.3f} max={timing.steady_max_ms:.3f}"
            )
            if timing.cold_start_ms is not None:
                print(
                    f"  cold_start_ms={timing.cold_start_ms:.3f} "
                    "(first forward: symm alloc + cold JIT)"
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
