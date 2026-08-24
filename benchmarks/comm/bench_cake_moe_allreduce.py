"""Benchmark the SM100 Cake MoE backend against the TRT-LLM backend.

Launch this file with ``torchrun`` on either two or four B200 GPUs.  Every
timed leg is bracketed by distributed correctness checks and uses a fresh IPC
workspace so protocol state cannot leak between backends or shapes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm.trtllm_ar import get_trtllm_comm_module
from flashinfer.jit import cake_moe_comm
from flashinfer.testing.utils import bench_gpu_time


HIDDEN_SIZE = 7168
ACTIVE_EXPERTS = 8
TOP_K = 8
ATOL = 1e-2
RTOL = 1e-2
MAX_BENCHMARK_LEGS = 512

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def _csv_int(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in _csv(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all integer values must be positive")
    return values


def _require_choices(
    parser: argparse.ArgumentParser,
    label: str,
    values: Sequence[str],
    allowed: set[str],
) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        parser.error(f"{label} contains unsupported values {unknown}; expected {sorted(allowed)}")


def _require_cupti() -> str:
    try:
        from cupti import cupti as _cupti  # noqa: F401

        version = package_version("cupti-python")
    except Exception as exc:
        raise RuntimeError(
            "cupti-python >=13 is required; refusing a fallback timing backend"
        ) from exc
    if int(version.split(".", 1)[0]) < 13:
        raise RuntimeError(
            f"cupti-python >=13 is required, found version {version}"
        )
    return version


def _bounded_rand(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
    scale: float = 0.25,
) -> torch.Tensor:
    values = torch.rand(shape, dtype=torch.float32, device=device, generator=generator)
    return ((values - 0.5) * (2.0 * scale)).to(dtype).contiguous()


def _assert_distributed_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    group: dist.ProcessGroup,
) -> float:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    close = torch.isclose(
        actual_f32,
        expected_f32,
        atol=ATOL,
        rtol=RTOL,
        equal_nan=False,
    ).all()
    failure = (~close).to(torch.int32)
    max_abs = torch.nan_to_num(
        (actual_f32 - expected_f32).abs(), nan=float("inf")
    ).max()
    dist.all_reduce(failure, op=dist.ReduceOp.MAX, group=group)
    dist.all_reduce(max_abs, op=dist.ReduceOp.MAX, group=group)
    if failure.item():
        raise AssertionError(
            f"{label} failed distributed close check: "
            f"max_abs={max_abs.item():.6g}, atol={ATOL}, rtol={RTOL}"
        )
    return float(max_abs.item())


def _rank_order_state_allreduce(
    local: torch.Tensor,
    dtype: torch.dtype,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    locals_by_rank = [
        torch.empty_like(local) for _ in range(dist.get_world_size(group))
    ]
    dist.all_gather(locals_by_rank, local, group=group)
    reduced = locals_by_rank[0]
    for peer_local in locals_by_rank[1:]:
        reduced = (reduced.float() + peer_local.float()).to(dtype)
    return reduced


def _seed(operation: str, world_size: int, rank: int, token_num: int) -> int:
    operation_offset = 0 if operation == "reduction" else 1_000_000
    return 0xCA4E0000 + operation_offset + world_size * 10000 + rank * 100 + token_num


def _make_reduction_case(
    *,
    world_size: int,
    rank: int,
    token_num: int,
    dtype: torch.dtype,
    device: torch.device,
    group: dist.ProcessGroup,
    workspace_tensor: torch.Tensor,
    backend: str,
    launch_with_pdl: bool,
    emit_allreduce: bool,
) -> tuple[Callable[[], None], Callable[[str], float]]:
    generator = torch.Generator(device=device).manual_seed(
        _seed("reduction", world_size, rank, token_num)
    )
    expert_input = _bounded_rand(
        (ACTIVE_EXPERTS, token_num, HIDDEN_SIZE),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    expert_scale = _bounded_rand(
        (ACTIVE_EXPERTS, token_num),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    token_input = _bounded_rand(
        (token_num, HIDDEN_SIZE),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    residual_in = _bounded_rand(
        (token_num, HIDDEN_SIZE),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    rms_gamma = (
        _bounded_rand(
            (HIDDEN_SIZE,),
            dtype=dtype,
            device=device,
            generator=generator,
            scale=0.125,
        )
        + 1
    ).contiguous()
    rms_eps = 1e-5

    local = torch.zeros_like(token_input)
    for expert in range(ACTIVE_EXPERTS):
        contribution = (
            expert_input[expert].float()
            * expert_scale[expert].float().unsqueeze(-1)
        ).to(dtype)
        local = (local.float() + contribution.float()).to(dtype)
    local = (local.float() + token_input.float()).to(dtype)
    allreduce_ref = _rank_order_state_allreduce(local, dtype, group)
    allreduce_ref_f32 = allreduce_ref.float()
    residual_ref = (allreduce_ref_f32 + residual_in.float()).to(dtype)
    residual_ref_f32 = residual_ref.float()
    norm_ref = (
        residual_ref_f32
        * torch.rsqrt(
            residual_ref_f32.square().mean(dim=-1, keepdim=True) + rms_eps
        )
        * rms_gamma.float()
    ).to(dtype)

    moe_allreduce_out = torch.empty_like(residual_in) if emit_allreduce else None
    residual_out = torch.empty_like(residual_in)
    norm_out = torch.empty_like(residual_in)

    def call() -> None:
        comm.trtllm_moe_reduction_allreduce_fusion(
            world_size=world_size,
            world_rank=rank,
            token_num=token_num,
            hidden_dim=HIDDEN_SIZE,
            workspace_ptrs=workspace_tensor,
            launch_with_pdl=launch_with_pdl,
            residual_in=residual_in,
            rms_gamma=rms_gamma,
            rms_eps=rms_eps,
            scale_factor=1.0,
            moe_reduction_device_num_experts=ACTIVE_EXPERTS,
            moe_reduction_scale_input=expert_scale,
            moe_reduction_active_experts_token_input=expert_input,
            moe_reduction_token_input=token_input,
            layout_code=None,
            moe_allreduce_out=moe_allreduce_out,
            residual_out=residual_out,
            norm_out=norm_out,
            quant_out=None,
            scale_out=None,
            backend=backend,
        )

    def validate(stage: str) -> float:
        max_diffs = [
            _assert_distributed_close(
                residual_out,
                residual_ref,
                label=f"{stage}/residual_out",
                group=group,
            ),
            _assert_distributed_close(
                norm_out,
                norm_ref,
                label=f"{stage}/norm_out",
                group=group,
            ),
        ]
        if moe_allreduce_out is not None:
            max_diffs.append(
                _assert_distributed_close(
                    moe_allreduce_out,
                    allreduce_ref_f32,
                    label=f"{stage}/moe_allreduce_out",
                    group=group,
                )
            )
        return max(max_diffs)

    return call, validate


def _make_finalize_case(
    *,
    world_size: int,
    rank: int,
    token_num: int,
    dtype: torch.dtype,
    device: torch.device,
    group: dist.ProcessGroup,
    workspace_tensor: torch.Tensor,
    backend: str,
    launch_with_pdl: bool,
    shared_mode: str,
) -> tuple[Callable[[], None], Callable[[str], float]]:
    generator = torch.Generator(device=device).manual_seed(
        _seed("finalize", world_size, rank, token_num)
    )
    allreduce_in = _bounded_rand(
        (token_num * TOP_K, HIDDEN_SIZE),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    residual_in = _bounded_rand(
        (token_num, HIDDEN_SIZE),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    norm_weight = (
        _bounded_rand(
            (HIDDEN_SIZE,),
            dtype=dtype,
            device=device,
            generator=generator,
            scale=0.125,
        )
        + 1
    ).contiguous()
    expert_scale = _bounded_rand(
        (token_num, TOP_K),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    inverse_indices = torch.arange(
        token_num * TOP_K, dtype=torch.int32, device=device
    ).reshape(token_num, TOP_K)
    shared_expert = _bounded_rand(
        (token_num, HIDDEN_SIZE),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    shared_expert_output = shared_expert if shared_mode == "present" else None
    routed_scaling_factor = 2.5 if shared_mode == "present" else None
    routed = 1.0 if routed_scaling_factor is None else routed_scaling_factor
    eps = 1e-5

    gathered = allreduce_in[inverse_indices]
    local = torch.zeros_like(residual_in)
    for route in range(TOP_K):
        contribution = (
            gathered[:, route].float()
            * expert_scale[:, route].float().unsqueeze(-1)
        ).to(dtype)
        local = (local.float() + contribution.float()).to(dtype)
    local = (local.float() * routed).to(dtype)
    if shared_expert_output is not None:
        local = (local.float() + shared_expert_output.float()).to(dtype)
    finalized_ref = _rank_order_state_allreduce(local, dtype, group)
    residual_ref = (finalized_ref.float() + residual_in.float()).to(dtype)
    residual_ref_f32 = residual_ref.float()
    norm_ref = (
        residual_ref_f32
        * torch.rsqrt(
            residual_ref_f32.square().mean(dim=-1, keepdim=True) + eps
        )
        * norm_weight.float()
    ).to(dtype)

    residual_out = torch.empty_like(residual_in)
    norm_out = torch.empty_like(residual_in)

    def call() -> None:
        comm.trtllm_moe_finalize_allreduce_fusion(
            allreduce_in=allreduce_in,
            residual_in=residual_in,
            norm_weight=norm_weight,
            expanded_idx_to_permuted_idx=inverse_indices,
            norm_out=norm_out,
            residual_out=residual_out,
            quant_out=None,
            scale_out=None,
            workspace_ptrs=workspace_tensor,
            launch_with_pdl=launch_with_pdl,
            world_rank=rank,
            world_size=world_size,
            eps=eps,
            shared_expert_output=shared_expert_output,
            expert_scale_factor=expert_scale,
            routed_scaling_factor=routed_scaling_factor,
            backend=backend,
        )

    def validate(stage: str) -> float:
        return max(
            _assert_distributed_close(
                residual_out,
                residual_ref,
                label=f"{stage}/residual_out",
                group=group,
            ),
            _assert_distributed_close(
                norm_out,
                norm_ref,
                label=f"{stage}/norm_out",
                group=group,
            ),
        )

    return call, validate


def _measure_leg(
    *,
    call: Callable[[], None],
    validate: Callable[[str], float],
    label: str,
    mode: str,
    dry_run_iters: int,
    repeat_iters: int,
    group: dist.ProcessGroup,
) -> tuple[list[float], float, float]:
    dist.barrier(group=group)
    call()
    torch.cuda.synchronize()
    pre_max_abs = validate(f"{label}/pre")
    dist.barrier(group=group)
    samples = bench_gpu_time(
        call,
        enable_cupti=True,
        use_cuda_graph=mode == "graph",
        cold_l2_cache=True,
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
        aggregate_op=max,
    )
    dist.barrier(group=group)
    call()
    torch.cuda.synchronize()
    post_max_abs = validate(f"{label}/post")
    return [float(sample) for sample in samples], pre_max_abs, post_max_abs


def _create_workspace(
    rank: int,
    world_size: int,
    max_token_num: int,
    group: dist.ProcessGroup,
) -> tuple[list[list[int]], torch.Tensor]:
    result = comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
        rank,
        world_size,
        max_token_num,
        HIDDEN_SIZE,
        group=group,
    )
    if len(result) != 2:
        raise RuntimeError("unexpected IPC workspace result")
    return result


def _comparison_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["operation"],
        row["dtype"],
        row["token_num"],
        row["launch_with_pdl"],
        row["mode"],
        row["variant"],
    )


def _summarize_comparisons(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_comparison_key(row)].append(row)

    comparisons = []
    for key, group_rows in grouped.items():
        trtllm = [row["median_ms"] for row in group_rows if row["backend"] == "trtllm"]
        cake = [row["median_ms"] for row in group_rows if row["backend"] == "cake"]
        if not trtllm or not cake:
            continue
        baseline_ms = float(statistics.median(trtllm))
        cake_ms = float(statistics.median(cake))
        comparisons.append(
            {
                "operation": key[0],
                "dtype": key[1],
                "token_num": key[2],
                "launch_with_pdl": key[3],
                "mode": key[4],
                "variant": key[5],
                "trtllm_median_ms": baseline_ms,
                "trtllm_leg_spread_ms": float(max(trtllm) - min(trtllm)),
                "cake_median_ms": cake_ms,
                "speedup": baseline_ms / cake_ms,
            }
        )
    return comparisons


def _parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operations", type=_csv, default=("reduction", "finalize"))
    parser.add_argument("--dtypes", type=_csv, default=("float16", "bfloat16"))
    parser.add_argument("--tokens", type=_csv_int, default=(1, 64, 128, 256, 2048))
    parser.add_argument("--pdl", type=_csv, default=("false", "true"))
    parser.add_argument("--modes", type=_csv, default=("eager", "graph"))
    parser.add_argument(
        "--backends",
        type=_csv,
        default=("trtllm", "cake", "trtllm"),
        help="ordered A/B legs; repeats are retained",
    )
    parser.add_argument(
        "--reduction-outputs",
        type=_csv,
        default=("present", "none"),
    )
    parser.add_argument(
        "--finalize-shared",
        type=_csv,
        default=("present",),
    )
    parser.add_argument("--dry-run-iters", type=int, default=5)
    parser.add_argument("--repeat-iters", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    _require_choices(parser, "--operations", args.operations, {"reduction", "finalize"})
    _require_choices(parser, "--dtypes", args.dtypes, set(_DTYPES))
    _require_choices(parser, "--pdl", args.pdl, {"false", "true"})
    _require_choices(parser, "--modes", args.modes, {"eager", "graph"})
    _require_choices(parser, "--backends", args.backends, {"trtllm", "cake"})
    _require_choices(
        parser, "--reduction-outputs", args.reduction_outputs, {"present", "none"}
    )
    _require_choices(
        parser, "--finalize-shared", args.finalize_shared, {"present", "none"}
    )
    if max(args.tokens) > 2048:
        parser.error("Cake MoE communication supports at most 2048 tokens")
    if args.dry_run_iters <= 0 or args.repeat_iters <= 0:
        parser.error("benchmark iteration counts must be positive")
    for label, values in (
        ("--operations", args.operations),
        ("--dtypes", args.dtypes),
        ("--tokens", args.tokens),
        ("--pdl", args.pdl),
        ("--modes", args.modes),
        ("--reduction-outputs", args.reduction_outputs),
        ("--finalize-shared", args.finalize_shared),
    ):
        if len(values) != len(set(values)):
            parser.error(f"{label} must not contain duplicate values")
    if not {"trtllm", "cake"}.issubset(args.backends):
        parser.error("--backends must include both trtllm and cake")
    if len(args.backends) > 3:
        parser.error("--backends supports at most three ordered A/B/A legs")
    variant_legs = sum(
        len(args.reduction_outputs)
        if operation == "reduction"
        else len(args.finalize_shared)
        for operation in args.operations
    )
    total_legs = (
        variant_legs
        * len(args.dtypes)
        * len(args.tokens)
        * len(args.pdl)
        * len(args.modes)
        * len(args.backends)
    )
    if total_legs > MAX_BENCHMARK_LEGS:
        parser.error(
            f"benchmark matrix expands to {total_legs} legs; "
            f"the safety limit is {MAX_BENCHMARK_LEGS}"
        )
    return parser, args


def main() -> int:
    parser, args = _parse_args()
    if not all(name in os.environ for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        parser.error("launch with torchrun so RANK, LOCAL_RANK, and WORLD_SIZE are set")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    if world_size not in (2, 4):
        parser.error("Cake MoE communication requires torchrun world size 2 or 4")
    if local_world_size != world_size:
        parser.error("Cake MoE communication benchmark requires a single node")
    if local_rank >= torch.cuda.device_count():
        parser.error(f"local rank {local_rank} has no visible CUDA device")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 0):
        parser.error(
            f"Cake MoE communication requires SM100, got {torch.cuda.get_device_capability(device)}"
        )

    started = time.monotonic()
    cupti_version = _require_cupti()
    dist.init_process_group(backend="nccl", init_method="env://")
    group = dist.group.WORLD
    rows: list[dict[str, Any]] = []
    try:
        if "cake" in args.backends:
            cake_moe_comm.load(local_rank)
        if "trtllm" in args.backends:
            get_trtllm_comm_module()
        dist.barrier(group=group)

        source_path, source_bytes = cake_moe_comm._load_source_bundle()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        manifest_path = source_path.parent / "manifest.json"
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

        leg_index = 0
        for operation in args.operations:
            variants = (
                args.reduction_outputs
                if operation == "reduction"
                else args.finalize_shared
            )
            for dtype_name in args.dtypes:
                dtype = _DTYPES[dtype_name]
                for token_num in args.tokens:
                    for pdl_name in args.pdl:
                        launch_with_pdl = pdl_name == "true"
                        for mode in args.modes:
                            for variant in variants:
                                for backend in args.backends:
                                    ipc_handles, workspace_tensor = _create_workspace(
                                        local_rank,
                                        world_size,
                                        max(args.tokens),
                                        group,
                                    )
                                    try:
                                        if operation == "reduction":
                                            call, validate = _make_reduction_case(
                                                world_size=world_size,
                                                rank=rank,
                                                token_num=token_num,
                                                dtype=dtype,
                                                device=device,
                                                group=group,
                                                workspace_tensor=workspace_tensor,
                                                backend=backend,
                                                launch_with_pdl=launch_with_pdl,
                                                emit_allreduce=variant == "present",
                                            )
                                        else:
                                            call, validate = _make_finalize_case(
                                                world_size=world_size,
                                                rank=rank,
                                                token_num=token_num,
                                                dtype=dtype,
                                                device=device,
                                                group=group,
                                                workspace_tensor=workspace_tensor,
                                                backend=backend,
                                                launch_with_pdl=launch_with_pdl,
                                                shared_mode=variant,
                                            )
                                        label = (
                                            f"{operation}/tp{world_size}/{dtype_name}/"
                                            f"tokens{token_num}/pdl{int(launch_with_pdl)}/"
                                            f"{mode}/{variant}/{backend}/leg{leg_index}"
                                        )
                                        samples, pre_max_abs, post_max_abs = _measure_leg(
                                            call=call,
                                            validate=validate,
                                            label=label,
                                            mode=mode,
                                            dry_run_iters=args.dry_run_iters,
                                            repeat_iters=args.repeat_iters,
                                            group=group,
                                        )
                                        if rank == 0:
                                            rows.append(
                                                {
                                                    "leg_index": leg_index,
                                                    "operation": operation,
                                                    "backend": backend,
                                                    "world_size": world_size,
                                                    "dtype": dtype_name,
                                                    "token_num": token_num,
                                                    "launch_with_pdl": launch_with_pdl,
                                                    "mode": mode,
                                                    "variant": variant,
                                                    "samples_ms": samples,
                                                    "median_ms": float(
                                                        statistics.median(samples)
                                                    ),
                                                    "min_ms": float(min(samples)),
                                                    "pre_max_abs": pre_max_abs,
                                                    "post_max_abs": post_max_abs,
                                                }
                                            )
                                    finally:
                                        dist.barrier(group=group)
                                        comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
                                            ipc_handles, group=group
                                        )
                                        dist.barrier(group=group)
                                    leg_index += 1

        dist.barrier(group=group)
        if rank == 0:
            report = {
                "schema_version": 1,
                "world_size": world_size,
                "gpu": torch.cuda.get_device_name(device),
                "cupti_python_version": cupti_version,
                "timing": {
                    "method": "bench_gpu_time",
                    "enable_cupti": True,
                    "cold_l2_cache": True,
                    "aggregate_op": "per-iteration rank max",
                    "dry_run_iters": args.dry_run_iters,
                    "repeat_iters": args.repeat_iters,
                },
                "source": {
                    "path": str(source_path),
                    "sha256": source_sha256,
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": manifest_sha256,
                },
                "rows": rows,
                "comparisons": _summarize_comparisons(rows),
                "physical_runtime_seconds": time.monotonic() - started,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(report, sort_keys=True))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group(group=group)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
