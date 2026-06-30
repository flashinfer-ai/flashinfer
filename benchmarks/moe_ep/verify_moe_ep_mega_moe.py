#!/usr/bin/env python3
"""Cross-check MoE-EP-only MegaMoE expert forward outputs across backends.

Runs selected backend pairs in one multi-rank job on shared fixtures from
``moe_ep_common`` and asserts per-rank bf16 outputs match within tolerance.

Default: vLLM ``DeepseekV4MegaMoEExperts`` (deep_gemm) vs flashinfer
``MoEEpMegaLayer`` + ``DeepGemmMegaMoeConfig`` (exact, atol=rtol=0).

Launch (example, 4 GPUs on one node):
    cd /lustre/fsw/coreai_libraries_cudnn/mhoqueanik/flashinfer

    torchrun --nproc_per_node=4 benchmarks/moe_ep/verify_moe_ep_mega_moe.py

    torchrun --nproc_per_node=4 benchmarks/moe_ep/verify_moe_ep_mega_moe.py \\
        --reference vllm_deepgemm --candidate fi_deep_gemm \\
        --num-tokens 4096 --num-max-tokens 4096

    # Exact vLLM vs fi_deep_gemm, plus smoke forward for fi_nvfp4 / fi_mxfp8:
    torchrun --nproc_per_node=4 benchmarks/moe_ep/verify_moe_ep_mega_moe.py --all

Requires:
    - vLLM with DeepSeek V4 MegaMoE support (for vllm_deepgemm reference)
    - flashinfer with ``moe_ep`` mega path
    - deep_gemm with ``fp8_fp4_mega_moe`` (deep_gemm backends)
    - BUILD_NVEP=1 flashinfer build for fi_nvfp4 / fi_mxfp8
    - SM100+ GPU per rank
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from backends import (
    bench_fi_forward,
    bench_vllm_forward,
    build_fi_mega_layer,
    build_vllm_mega_moe,
    cleanup_vllm_distributed,
    destroy_fi_layer,
    init_vllm_distributed,
    mega_moe_ep_layout,
    release_vllm_experts,
    run_fi_forward,
    run_vllm_forward,
)
from moe_ep_common import (
    BACKEND_IDS,
    BACKEND_LABELS,
    VERIFY_TIMING_BACKEND_IDS,
    activation_clamp_from_args,
    benchmark_backend_order,
    check_problem,
    flashinfer_backend_ids,
    make_benchmark_weights,
    make_shared_benchmark_inputs,
    print_benchmark_timing,
    require_env_rank,
    require_shared_routing_benchmark,
    require_sm100,
    resolve_timing_mode,
    routing_mode_from_args,
    cold_start_note_for_backend,
    timing_mode_note_for_backend,
    warn_kernel_token_sizing,
)

DEFAULT_EXACT_ATOL = 0.0
DEFAULT_EXACT_RTOL = 0.0

VerifyMode = Literal["exact", "smoke"]

# nvfp4 / mxfp8 are different quant kernels; numeric match vs deep_gemm is not expected.
QUANT_BACKEND_IDS = frozenset({"fi_nvfp4", "fi_mxfp8"})


@dataclass(frozen=True)
class VerificationCase:
    reference: str
    candidate: str
    mode: VerifyMode


def _parse_verify_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify MoE-EP-only MegaMoE expert forwards match across backends."
        )
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="vllm_deepgemm",
        choices=BACKEND_IDS,
        help="Golden backend (default: vLLM native deep_gemm).",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        default="fi_deep_gemm",
        choices=BACKEND_IDS,
        help="Backend under test (ignored when --all).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Exact check vllm_deepgemm vs fi_deep_gemm, then smoke-test "
            "fi_nvfp4 and fi_mxfp8 (forward + finite bf16 output; no numeric "
            "compare vs deep_gemm)."
        ),
    )
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument(
        "--timing",
        type=str,
        default="cudagraph",
        choices=("cudagraph", "cuda_event", "cupti"),
        help="Optional post-check timing for reference and candidate.",
    )
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--cold-l2", action="store_true")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--num-max-tokens", type=int, default=8192)
    parser.add_argument("--num-local-experts", type=int, default=64)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--no-activation-clamp", action="store_true")
    parser.add_argument(
        "--fast-math",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--renormalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--routed-scaling-factor", type=float, default=1.0)
    parser.add_argument(
        "--random-routing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--cold-start",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--time-quant-backends",
        action="store_true",
        help=(
            "Also time fi_nvfp4 / fi_mxfp8 after smoke checks. Off by default "
            "because NVSHMEM init can skew deep_gemm timings in-process."
        ),
    )
    parser.add_argument(
        "--skip-timing",
        action="store_true",
        help="Only run correctness checks; skip post-check latency.",
    )
    return parser.parse_args()


def _tolerance_for_case(case: VerificationCase, args: argparse.Namespace) -> tuple[float, float]:
    if case.mode == "smoke":
        raise ValueError("smoke cases do not use numeric tolerance")
    if args.atol is not None and args.rtol is not None:
        return args.atol, args.rtol
    return DEFAULT_EXACT_ATOL, DEFAULT_EXACT_RTOL


def _verification_cases(args: argparse.Namespace) -> list[VerificationCase]:
    if args.all:
        if args.reference != "vllm_deepgemm":
            raise SystemExit(
                "ERROR: --all requires --reference vllm_deepgemm "
                "(exact deep_gemm cross-check + quant-backend smoke tests)."
            )
        cases = [
            VerificationCase("vllm_deepgemm", "fi_deep_gemm", "exact"),
        ]
        for backend_id in flashinfer_backend_ids():
            if backend_id in QUANT_BACKEND_IDS:
                cases.append(
                    VerificationCase("vllm_deepgemm", backend_id, "smoke")
                )
        return cases

    if args.candidate == args.reference:
        raise SystemExit("ERROR: --candidate must differ from --reference")

    if (
        args.candidate in QUANT_BACKEND_IDS
        and args.reference == "vllm_deepgemm"
        and args.atol is None
        and args.rtol is None
    ):
        raise SystemExit(
            "ERROR: fi_nvfp4 and fi_mxfp8 are not numerically comparable to "
            "vllm_deepgemm (different quant kernels). Use --all for the "
            "recommended check plan, or pick another --reference."
        )

    return [VerificationCase(args.reference, args.candidate, "exact")]


def _compare_outputs(
    y_ref: torch.Tensor,
    y_cand: torch.Tensor,
    *,
    rank: int,
    reference: str,
    candidate: str,
    atol: float,
    rtol: float,
) -> None:
    if y_ref.shape != y_cand.shape:
        raise RuntimeError(
            f"rank {rank}: shape mismatch {reference}={tuple(y_ref.shape)} "
            f"{candidate}={tuple(y_cand.shape)}"
        )
    if y_ref.dtype != y_cand.dtype:
        raise RuntimeError(
            f"rank {rank}: dtype mismatch {reference}={y_ref.dtype} "
            f"{candidate}={y_cand.dtype}"
        )

    torch.testing.assert_close(
        y_ref,
        y_cand,
        atol=atol,
        rtol=rtol,
        msg=lambda _: (
            f"rank {rank}: {reference} vs {candidate} mismatch "
            f"(max_abs={(y_ref - y_cand).abs().max().item():.6g})"
        ),
    )


def _smoke_check_output(
    y: torch.Tensor,
    *,
    rank: int,
    candidate: str,
    num_tokens: int,
    hidden: int,
) -> None:
    expected_shape = (num_tokens, hidden)
    if tuple(y.shape) != expected_shape:
        raise RuntimeError(
            f"rank {rank}: {candidate} smoke check shape mismatch "
            f"expected={expected_shape} got={tuple(y.shape)}"
        )
    if y.dtype != torch.bfloat16:
        raise RuntimeError(
            f"rank {rank}: {candidate} smoke check expected bf16, got {y.dtype}"
        )
    if not torch.isfinite(y).all():
        bad = (~torch.isfinite(y)).sum().item()
        raise RuntimeError(
            f"rank {rank}: {candidate} smoke check found {bad} non-finite values"
        )


def _run_backend_forward(
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
    args: argparse.Namespace,
    activation_clamp: float | None,
    use_vllm_ep_group: bool = False,
) -> tuple[torch.Tensor, object | None]:
    if backend_id == "vllm_deepgemm":
        experts = build_vllm_mega_moe(
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
        out = run_vllm_forward(
            vllm_config=vllm_config,
            experts=experts,
            inputs=inputs,
            activation_clamp=activation_clamp,
            fast_math=args.fast_math,
        )
        return out, experts

    layer = build_fi_mega_layer(
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
    )
    out = run_fi_forward(layer, inputs)
    return out, layer


def _release_backend(
    backend_id: str,
    handle: object | None,
    *,
    vllm_config,
) -> None:
    if handle is None:
        return
    if backend_id == "vllm_deepgemm":
        release_vllm_experts(vllm_config, handle)
    else:
        destroy_fi_layer(handle)


def _backends_in_cases(cases: list[VerificationCase]) -> set[str]:
    out: set[str] = set()
    for case in cases:
        out.add(case.reference)
        out.add(case.candidate)
    return out


def _backends_for_timing(args: argparse.Namespace, cases: list[VerificationCase]) -> list[str]:
    if args.all and not args.time_quant_backends:
        return list(VERIFY_TIMING_BACKEND_IDS)
    backends = _backends_in_cases(cases)
    return benchmark_backend_order(list(backends))


def _time_one_backend(
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
    args: argparse.Namespace,
    activation_clamp: float | None,
    timing_mode: str,
    use_vllm_ep_group: bool,
):
    _, handle = _run_backend_forward(
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
        use_vllm_ep_group=use_vllm_ep_group,
    )

    if backend_id == "vllm_deepgemm":
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
        timing = bench_fi_forward(
            handle,
            inputs,
            backend_id=backend_id,
            timing_mode=timing_mode,
            warmup=args.warmup,
            repeat=args.repeat,
            cold_start=args.cold_start,
            cold_l2_cache=args.cold_l2,
        )
    _release_backend(backend_id, handle, vllm_config=vllm_config)
    return timing


def main() -> int:
    args = _parse_verify_args()
    require_shared_routing_benchmark(args)
    local_rank, world_size = require_env_rank()
    require_sm100(local_rank)

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
    timing_mode = resolve_timing_mode(args)
    rank = dist.get_rank() if dist.is_initialized() else int(os.environ["RANK"])
    cases = _verification_cases(args)
    backends_needed = _backends_in_cases(cases)

    vllm_config = None
    config_ctx = None
    needs_vllm = "vllm_deepgemm" in backends_needed
    if needs_vllm:
        vllm_config, config_ctx = init_vllm_distributed(
            local_rank,
            world_size,
            num_max_tokens=args.num_max_tokens,
        )
    else:
        from backends import init_distributed

        init_distributed(local_rank, world_size)

    exit_code = 0
    results: list[tuple[VerificationCase, str]] = []

    try:
        _, num_local_experts, experts_start_idx = (
            mega_moe_ep_layout(num_experts)
            if needs_vllm
            else (
                rank,
                args.num_local_experts,
                rank * args.num_local_experts,
            )
        )
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

        reference_cache: dict[str, torch.Tensor] = {}
        use_vllm_ep_group = needs_vllm
        exact_cases = [c for c in cases if c.mode == "exact"]
        smoke_cases = [c for c in cases if c.mode == "smoke"]

        for case in exact_cases:
            if case.reference not in reference_cache:
                y_ref, ref_handle = _run_backend_forward(
                    case.reference,
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
                    use_vllm_ep_group=use_vllm_ep_group,
                )
                torch.cuda.synchronize()
                dist.barrier()
                reference_cache[case.reference] = y_ref.detach().clone()
                _release_backend(case.reference, ref_handle, vllm_config=vllm_config)
                dist.barrier()

            y_cand, cand_handle = _run_backend_forward(
                case.candidate,
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
                use_vllm_ep_group=use_vllm_ep_group,
            )
            torch.cuda.synchronize()
            dist.barrier()

            atol, rtol = _tolerance_for_case(case, args)
            _compare_outputs(
                reference_cache[case.reference],
                y_cand,
                rank=rank,
                reference=case.reference,
                candidate=case.candidate,
                atol=atol,
                rtol=rtol,
            )
            results.append((case, f"exact atol={atol} rtol={rtol}"))
            _release_backend(case.candidate, cand_handle, vllm_config=vllm_config)
            dist.barrier()

        if rank == 0:
            print("OK: MoE-EP MegaMoE verification passed")
            print(
                f"  world_size={world_size} num_experts={num_experts} "
                f"num_tokens={args.num_tokens} num_max_tokens={args.num_max_tokens}"
            )
            print(
                f"  routing={routing_mode_from_args(args)} "
                f"clamp={activation_clamp} fast_math={args.fast_math}"
            )
            for case, detail in results:
                print(
                    f"  {case.reference} vs {case.candidate}: "
                    f"{case.mode} {detail} "
                    f"({BACKEND_LABELS[case.candidate]})"
                )

        if not args.skip_timing:
            backend_timings: dict[str, object] = {}
            timing_backends = _backends_for_timing(args, cases)
            for backend_id in timing_backends:
                backend_timings[backend_id] = _time_one_backend(
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
                    use_vllm_ep_group=use_vllm_ep_group,
                )
                dist.barrier()

            if rank == 0:
                print(f"  timing={timing_mode} (deep_gemm pair before quant smoke)")
                for backend_id in timing_backends:
                    timing = backend_timings[backend_id]
                    print(f"  {backend_id}:")
                    print_benchmark_timing(
                        timing,
                        rank=rank,
                        cold_start_note=cold_start_note_for_backend(backend_id),
                    )
                    note = timing_mode_note_for_backend(
                        backend_id, timing_mode=timing_mode
                    )
                    print(f"    ({note})")

        for case in smoke_cases:
            y_cand, cand_handle = _run_backend_forward(
                case.candidate,
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
                use_vllm_ep_group=use_vllm_ep_group,
            )
            torch.cuda.synchronize()
            dist.barrier()

            _smoke_check_output(
                y_cand,
                rank=rank,
                candidate=case.candidate,
                num_tokens=args.num_tokens,
                hidden=args.hidden,
            )
            results.append((case, "smoke (shape/dtype/finite)"))
            _release_backend(case.candidate, cand_handle, vllm_config=vllm_config)
            dist.barrier()

            if rank == 0:
                print(
                    f"  {case.reference} vs {case.candidate}: "
                    f"{case.mode} smoke (shape/dtype/finite) "
                    f"({BACKEND_LABELS[case.candidate]})"
                )

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
