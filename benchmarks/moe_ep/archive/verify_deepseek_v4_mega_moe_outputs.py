#!/usr/bin/env python3
"""Cross-check vLLM vs flashinfer DeepSeek V4 MegaMoE expert forward outputs.

Runs both backend paths in one multi-rank job on shared fixtures from
``bench_deepseek_v4_mega_moe_common`` and asserts the per-rank bf16 outputs
match. Uses the same expert-only forward as the timing benchmarks (shared
random routing fixtures; no gate / shared experts). After outputs match, prints
warm (steady-state) and cold-start latency for each backend.

Launch (example, 4 GPUs on one node):
    torchrun --nproc_per_node=4 \\
        benchmarks/moe_ep/verify_deepseek_v4_mega_moe_outputs.py

    torchrun --nproc_per_node=4 \\
        benchmarks/moe_ep/verify_deepseek_v4_mega_moe_outputs.py \\
        --num-tokens 4096 --num-max-tokens 4096

Requires:
    - vLLM with DeepSeek V4 MegaMoE support
    - flashinfer with ``moe_ep`` mega path
    - deep_gemm with ``fp8_fp4_mega_moe``
    - SM100+ GPU per rank
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from bench_deepseek_v4_mega_moe_common import (
    activation_clamp_from_args,
    bench_forward_ms,
    check_problem,
    make_benchmark_weights,
    make_shared_benchmark_inputs,
    parse_benchmark_args,
    print_benchmark_timing,
    require_env_rank,
    require_shared_routing_benchmark,
    routing_mode_from_args,
    MOE_EP_V2_COLD_START_NOTE,
    VLLM_COLD_START_NOTE,
    warn_kernel_token_sizing,
)
from bench_deepseek_v4_mega_moe_flashinfer import (
    _build_mega_moe_layer,
    _to_moe_ep_tensors,
)
from bench_deepseek_v4_mega_moe_vLLM import (
    EXPERTS_PREFIX,
    _build_mega_moe_experts,
    _destroy_symm_buffers,
    _init_vllm_distributed,
    _load_benchmark_weights_into_experts,
    _mega_moe_ep_layout,
)


def _parse_verify_args():
    atol = 0.0
    rtol = 0.0
    bench_argv = [sys.argv[0]]
    argv = sys.argv[1:]
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--atol":
            idx += 1
            atol = float(argv[idx])
        elif arg == "--rtol":
            idx += 1
            rtol = float(argv[idx])
        elif arg.startswith("--atol="):
            atol = float(arg.split("=", 1)[1])
        elif arg.startswith("--rtol="):
            rtol = float(arg.split("=", 1)[1])
        else:
            bench_argv.append(arg)
        idx += 1

    saved_argv = sys.argv
    try:
        sys.argv = bench_argv
        args = parse_benchmark_args(
            description=(
                "Verify vLLM DeepseekV4MegaMoEExperts and flashinfer MoEEpMegaLayer "
                "produce identical outputs on shared fixtures."
            )
        )
    finally:
        sys.argv = saved_argv

    args.atol = atol
    args.rtol = rtol
    return args


def _unregister_vllm_experts(vllm_config, *, prefix: str = EXPERTS_PREFIX) -> None:
    ctx = vllm_config.compilation_config.static_forward_context
    if ctx.get(prefix) is not None:
        del ctx[prefix]


def _release_vllm_experts(vllm_config, experts) -> None:
    _unregister_vllm_experts(vllm_config, prefix=experts.prefix)
    _destroy_symm_buffers()


def _build_vllm_experts(
    *,
    vllm_config,
    bench_weights,
    num_experts: int,
    num_local_experts: int,
    experts_start_idx: int,
    args,
    intermediate: int,
):
    experts = _build_mega_moe_experts(
        vllm_config,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        experts_start_idx=experts_start_idx,
        topk=args.topk,
        hidden=args.hidden,
        intermediate=intermediate,
        num_max_tokens=args.num_max_tokens,
    )
    _load_benchmark_weights_into_experts(
        experts,
        bench_weights,
        experts_start_idx=experts_start_idx,
        intermediate=intermediate,
    )
    return experts


def _build_flashinfer_layer(
    *,
    rank: int,
    world_size: int,
    bench_weights,
    num_experts: int,
    args,
    activation_clamp: float | None,
):
    return _build_mega_moe_layer(
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


def _require_sm100(local_rank: int) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA is required")
    capability = torch.cuda.get_device_capability(local_rank)
    if capability[0] != 10:
        raise SystemExit(
            f"ERROR: rank {local_rank} needs SM100 (Blackwell), "
            f"got sm_{capability[0]}{capability[1]}"
        )


def _run_vllm_forward(
    *,
    vllm_config,
    experts,
    inputs,
    activation_clamp: float | None,
    fast_math: bool,
) -> torch.Tensor:
    from vllm.forward_context import set_forward_context

    num_tokens = inputs.hidden_states.shape[0]
    with set_forward_context(None, vllm_config, num_tokens=num_tokens):
        return experts(
            inputs.hidden_states,
            inputs.topk_weights,
            inputs.topk_ids,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )


def _compare_outputs(
    y_vllm: torch.Tensor,
    y_flashinfer: torch.Tensor,
    *,
    rank: int,
    atol: float,
    rtol: float,
) -> None:
    if y_vllm.shape != y_flashinfer.shape:
        raise RuntimeError(
            f"rank {rank}: shape mismatch vLLM={tuple(y_vllm.shape)} "
            f"flashinfer={tuple(y_flashinfer.shape)}"
        )
    if y_vllm.dtype != y_flashinfer.dtype:
        raise RuntimeError(
            f"rank {rank}: dtype mismatch vLLM={y_vllm.dtype} "
            f"flashinfer={y_flashinfer.dtype}"
        )

    torch.testing.assert_close(
        y_vllm,
        y_flashinfer,
        atol=atol,
        rtol=rtol,
        msg=lambda _: (
            f"rank {rank}: vLLM vs flashinfer mismatch "
            f"(max_abs={(y_vllm - y_flashinfer).abs().max().item():.6g})"
        ),
    )


def main() -> int:
    args = _parse_verify_args()
    require_shared_routing_benchmark(args)
    local_rank, world_size = require_env_rank()
    _require_sm100(local_rank)

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

    activation_clamp = activation_clamp_from_args(args)
    vllm_config, config_ctx = _init_vllm_distributed(
        local_rank,
        world_size,
        num_max_tokens=args.num_max_tokens,
    )
    rank = dist.get_rank()
    mega = None
    exit_code = 0

    try:
        _, num_local_experts, experts_start_idx = _mega_moe_ep_layout(num_experts)
        bench_weights = make_benchmark_weights(
            rank,
            num_local_experts=num_local_experts,
            hidden=args.hidden,
            intermediate=args.intermediate,
        )
        inputs = make_shared_benchmark_inputs(
            rank,
            args,
            num_experts=num_experts,
        )

        experts = _build_vllm_experts(
            vllm_config=vllm_config,
            bench_weights=bench_weights,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            experts_start_idx=experts_start_idx,
            args=args,
            intermediate=args.intermediate,
        )

        y_vllm = _run_vllm_forward(
            vllm_config=vllm_config,
            experts=experts,
            inputs=inputs,
            activation_clamp=activation_clamp,
            fast_math=args.fast_math,
        )
        torch.cuda.synchronize()
        dist.barrier()

        _release_vllm_experts(vllm_config, experts)
        del experts

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
        y_flashinfer = mega.forward(_to_moe_ep_tensors(inputs))
        torch.cuda.synchronize()
        dist.barrier()

        _compare_outputs(
            y_vllm,
            y_flashinfer,
            rank=rank,
            atol=args.atol,
            rtol=args.rtol,
        )
        dist.barrier()

        if mega is not None:
            mega.destroy()
            mega = None
        _destroy_symm_buffers()

        experts = _build_vllm_experts(
            vllm_config=vllm_config,
            bench_weights=bench_weights,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            experts_start_idx=experts_start_idx,
            args=args,
            intermediate=args.intermediate,
        )

        def run_vllm_once() -> torch.Tensor:
            return _run_vllm_forward(
                vllm_config=vllm_config,
                experts=experts,
                inputs=inputs,
                activation_clamp=activation_clamp,
                fast_math=args.fast_math,
            )

        vllm_timing = bench_forward_ms(
            run_vllm_once,
            warmup=args.warmup,
            repeat=args.repeat,
            cold_start=args.cold_start,
        )
        _release_vllm_experts(vllm_config, experts)
        del experts

        mega = _build_flashinfer_layer(
            rank=rank,
            world_size=world_size,
            bench_weights=bench_weights,
            num_experts=num_experts,
            args=args,
            activation_clamp=activation_clamp,
        )
        tensors = _to_moe_ep_tensors(inputs)

        def run_flashinfer_once() -> torch.Tensor:
            return mega.forward(tensors)

        flashinfer_timing = bench_forward_ms(
            run_flashinfer_once,
            warmup=args.warmup,
            repeat=args.repeat,
            cold_start=args.cold_start,
        )
        dist.barrier()

        if rank == 0:
            print("OK: vLLM and flashinfer MegaMoE expert outputs match")
            print(
                f"  world_size={world_size} num_experts={num_experts} "
                f"num_tokens={args.num_tokens} num_max_tokens={args.num_max_tokens}"
            )
            print(
                f"  routing={routing_mode_from_args(args)} "
                f"clamp={activation_clamp} fast_math={args.fast_math} "
                f"atol={args.atol} rtol={args.rtol}"
            )
            print(
                f"  warmup={args.warmup} repeat={args.repeat} "
                f"cold_start={'on' if args.cold_start else 'off'}"
            )
            print("  vLLM DeepseekV4MegaMoEExperts:")
            print_benchmark_timing(
                vllm_timing,
                rank=rank,
                cold_start_note=VLLM_COLD_START_NOTE,
            )
            print("  flashinfer MoEEpMegaLayer:")
            print_benchmark_timing(
                flashinfer_timing,
                rank=rank,
                cold_start_note=MOE_EP_V2_COLD_START_NOTE,
            )
    except Exception:
        exit_code = 1
        raise
    finally:
        if mega is not None:
            mega.destroy()
        _destroy_symm_buffers()
        config_ctx.__exit__(None, None, None)
        if dist.is_initialized():
            dist.destroy_process_group()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
