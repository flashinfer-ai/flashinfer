#!/usr/bin/env python3
"""Multi-rank MoE-EP-only MegaMoE expert forward benchmark.

Times the expert kernel only (no decoder, gate, or attention) across backends:
  - vLLM ``DeepseekV4MegaMoEExperts`` (deep_gemm_mega_moe)
  - flashinfer ``MoEEpMegaLayer`` + deep_gemm / nvfp4 / mxfp8 megakernels

Default timing uses CUDA graph replay (eager staging + graphed compute) for
steady-state deep_gemm measurements. Use ``--timing cuda_event`` or
``--no-cuda-graph`` for CUDA events with optional cold-start timing.

Launch (example, 4 GPUs on one node):
    cd /lustre/fsw/coreai_libraries_cudnn/mhoqueanik/flashinfer

    torchrun --nproc_per_node=4 benchmarks/moe_ep/bench_moe_ep_mega_moe.py

    torchrun --nproc_per_node=4 benchmarks/moe_ep/bench_moe_ep_mega_moe.py \\
        --backend fi_deep_gemm --num-tokens 4096 --num-max-tokens 4096 \\
        --warmup 10 --repeat 50

    # Compare all backends sequentially:
    torchrun --nproc_per_node=4 benchmarks/moe_ep/bench_moe_ep_mega_moe.py \\
        --backend all --timing cudagraph

    # Prestaged nvfp4/mxfp8 (bf16→quant once outside timed loop):
    torchrun --nproc_per_node=4 benchmarks/moe_ep/bench_moe_ep_mega_moe.py \\
        --backend fi_nvfp4 --bench-scope prestaged --timing cuda_event

    # vLLM native deepgemm baseline with CUDA events + cold-start:
    torchrun --nproc_per_node=4 benchmarks/moe_ep/bench_moe_ep_mega_moe.py \\
        --backend vllm_deepgemm --no-cuda-graph

Requires:
    - vLLM with DeepSeek V4 MegaMoE support (for vllm_deepgemm)
    - flashinfer with ``moe_ep`` mega path
    - deep_gemm with ``fp8_fp4_mega_moe``
    - BUILD_NVEP=1 flashinfer build for fi_nvfp4 / fi_mxfp8
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

from backends import (
    backends_for_cli,
    bench_fi_forward,
    bench_vllm_forward,
    build_fi_mega_layer,
    build_vllm_mega_moe,
    cleanup_vllm_distributed,
    destroy_fi_layer,
    init_distributed,
    init_vllm_distributed,
    make_prestaged_fi_tensors,
    mega_moe_ep_layout,
    release_vllm_experts,
)
from moe_ep_common import (
    BACKEND_LABELS,
    activation_clamp_from_args,
    bench_scope_fallback_note,
    bench_scope_note_for_scope,
    check_problem,
    cold_start_note_for_backend,
    fi_timing_fallback_note,
    make_benchmark_weights,
    make_shared_benchmark_inputs,
    parse_bench_args,
    print_benchmark_header,
    print_benchmark_timing,
    require_env_rank,
    require_shared_routing_benchmark,
    require_sm100,
    resolve_bench_scope,
    resolve_fi_timing_mode,
    resolve_timing_mode,
    routing_mode_from_args,
    timing_mode_note_for_backend,
    warn_kernel_token_sizing,
)


def _needs_vllm(backend_ids: list[str]) -> bool:
    return "vllm_deepgemm" in backend_ids


def _bench_one_backend(
    backend_id: str,
    *,
    vllm_config,
    rank: int,
    world_size: int,
    bench_weights,
    inputs,
    num_experts: int,
    num_local_experts: int,
    experts_start_idx: int,
    args,
    activation_clamp: float | None,
    timing_mode: str,
):
    handle = None
    use_vllm_ep_group = vllm_config is not None
    requested_timing = timing_mode
    effective_timing = (
        resolve_fi_timing_mode(backend_id, timing_mode)
        if backend_id != "vllm_deepgemm"
        else timing_mode
    )
    fallback_note = fi_timing_fallback_note(
        backend_id, requested_timing, effective_timing
    )
    requested_scope = args.bench_scope
    effective_scope = resolve_bench_scope(backend_id, requested_scope)
    scope_fallback_note = bench_scope_fallback_note(
        backend_id, requested_scope, effective_scope
    )
    try:
        if backend_id == "vllm_deepgemm":
            handle = build_vllm_mega_moe(
                vllm_config,
                bench_weights,
                num_experts=num_experts,
                num_local_experts=num_local_experts,
                experts_start_idx=experts_start_idx,
                topk=args.topk,
                hidden=args.hidden,
                intermediate=args.intermediate,
                num_max_tokens=args.num_max_tokens,
            )

            timing = bench_vllm_forward(
                handle,
                inputs,
                activation_clamp=activation_clamp,
                fast_math=args.fast_math,
                timing_mode=timing_mode,
                warmup=args.warmup,
                repeat=args.repeat,
                cold_start=args.cold_start,
                cold_l2_cache=args.cold_l2,
            )
        else:
            quantize_input = effective_scope != "prestaged"
            fi_inputs = inputs
            handle = build_fi_mega_layer(
                rank,
                world_size,
                backend_id=backend_id,
                num_experts=num_experts,
                topk=args.topk,
                hidden=args.hidden,
                intermediate=args.intermediate,
                num_max_tokens=args.num_max_tokens,
                weights=bench_weights,
                activation_clamp=activation_clamp,
                fast_math=args.fast_math,
                use_vllm_ep_group=use_vllm_ep_group,
                quantize_input=quantize_input,
            )
            if effective_scope == "prestaged":
                fi_inputs = make_prestaged_fi_tensors(
                    backend_id,
                    inputs,
                    rank=rank,
                    world_size=world_size,
                    num_experts=num_experts,
                    num_local_experts=num_local_experts,
                    hidden=args.hidden,
                    intermediate=args.intermediate,
                    num_max_tokens=args.num_max_tokens,
                    topk=args.topk,
                    activation_clamp=activation_clamp,
                    use_vllm_ep_group=use_vllm_ep_group,
                )
            timing = bench_fi_forward(
                handle,
                fi_inputs,
                backend_id=backend_id,
                timing_mode=timing_mode,
                warmup=args.warmup,
                repeat=args.repeat,
                cold_start=args.cold_start,
                cold_l2_cache=args.cold_l2,
            )

        if backend_id == "vllm_deepgemm":
            extra_lines = [
                f"  bench_scope={effective_scope}",
                f"  megakernel={BACKEND_LABELS[backend_id]}",
            ]
        else:
            extra_lines = [
                f"  bench_scope={effective_scope}",
                f"  {bench_scope_note_for_scope(effective_scope)}",
                "  preprocess_weights=True",
                f"  megakernel={BACKEND_LABELS[backend_id]}",
            ]
        if fallback_note is not None:
            extra_lines.append(f"  note={fallback_note}")
        if scope_fallback_note is not None:
            extra_lines.append(f"  note={scope_fallback_note}")

        print_benchmark_header(
            title="MoE-EP MegaMoE expert forward benchmark",
            rank=rank,
            world_size=world_size,
            num_local_experts=num_local_experts,
            args=args,
            num_experts=num_experts,
            activation_clamp=activation_clamp,
            routing_mode=routing_mode_from_args(args),
            backend_id=backend_id,
            timing_mode=effective_timing,
            extra_lines=tuple(extra_lines),
            timing_mode_note=timing_mode_note_for_backend(
                backend_id, timing_mode=effective_timing
            ),
        )
        print_benchmark_timing(
            timing,
            rank=rank,
            cold_start_note=cold_start_note_for_backend(backend_id),
        )
        return timing.steady_avg_ms
    finally:
        if handle is not None:
            if backend_id == "vllm_deepgemm":
                release_vllm_experts(vllm_config, handle)
            else:
                destroy_fi_layer(handle)


def main() -> int:
    args = parse_bench_args(
        description="Benchmark MoE-EP-only MegaMoE expert forward (multi-rank EP)."
    )
    require_shared_routing_benchmark(args)
    local_rank, world_size = require_env_rank()
    require_sm100(local_rank)

    backend_ids = backends_for_cli(args.backend)
    timing_mode = resolve_timing_mode(args)

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
    vllm_config = None
    config_ctx = None

    if _needs_vllm(backend_ids):
        vllm_config, config_ctx = init_vllm_distributed(
            local_rank,
            world_size,
            num_max_tokens=args.num_max_tokens,
        )
    else:
        init_distributed(local_rank, world_size)

    rank = dist.get_rank()
    exit_code = 0
    summary: list[tuple[str, float]] = []

    try:
        if _needs_vllm(backend_ids):
            _, num_local_experts, experts_start_idx = mega_moe_ep_layout(num_experts)
        else:
            num_local_experts = args.num_local_experts
            experts_start_idx = rank * num_local_experts

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

        for backend_id in backend_ids:
            dist.barrier()
            steady_avg_ms = _bench_one_backend(
                backend_id,
                vllm_config=vllm_config,
                rank=rank,
                world_size=world_size,
                bench_weights=bench_weights,
                inputs=inputs,
                num_experts=num_experts,
                num_local_experts=num_local_experts,
                experts_start_idx=experts_start_idx,
                args=args,
                activation_clamp=activation_clamp,
                timing_mode=timing_mode,
            )
            summary.append((backend_id, steady_avg_ms))
            dist.barrier()

        if rank == 0 and len(summary) > 1:
            print("Summary (steady_avg_ms, vllm first / quant last):")
            for backend_id, steady_avg_ms in summary:
                print(f"  {backend_id}: {steady_avg_ms:.3f} ms")
    except Exception:
        exit_code = 1
        raise
    finally:
        if config_ctx is not None and vllm_config is not None:
            cleanup_vllm_distributed(vllm_config, config_ctx)
        elif dist.is_initialized():
            dist.destroy_process_group()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
