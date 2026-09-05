"""Benchmark Cake indexed MoE finalize against TRT-LLM on TP2, TP4, or TP8.

Launch with ``torchrun --nproc-per-node {2,4,8}``. Every timed leg uses CUPTI
GPU activity timing with cold L2 and is bracketed by distributed correctness
checks. The default TRT-LLM/Cake/TRT-LLM order exposes same-session drift.
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
from flashinfer.comm.trtllm_ar import MAX_COMM_SIZE, get_trtllm_comm_module
from flashinfer.jit import cake_moe_finalize_comm as cake_finalize
from flashinfer.testing.utils import bench_gpu_time


HIDDEN_SIZE = 7168
ATOL = 1e-2
RTOL = 1e-2
MAX_BENCHMARK_LEGS = 384

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return items


def _csv_int(value: str) -> tuple[int, ...]:
    try:
        items = tuple(int(item) for item in _csv(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError("all integers must be positive")
    return items


def _require_choices(
    parser: argparse.ArgumentParser,
    label: str,
    values: Sequence[str],
    allowed: set[str],
) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        parser.error(f"{label} contains {unknown}; expected {sorted(allowed)}")


def _require_cupti() -> tuple[str, Any]:
    try:
        from cupti import cupti as cupti_module

        version = package_version("cupti-python")
    except Exception as exc:
        raise RuntimeError(
            "cupti-python >=13 is required; refusing a fallback timing backend"
        ) from exc
    if int(version.split(".", 1)[0]) < 13:
        raise RuntimeError(f"cupti-python >=13 is required, found {version}")
    return version, cupti_module


def _defer_cupti_finalize(cupti_module: Any) -> Callable[[], Any]:
    """Keep process-global CUPTI state alive until the NCCL watchdog exits."""

    finalize = cupti_module.finalize
    cupti_module.finalize = lambda: None
    return finalize


def _bounded_rand(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    values = torch.rand(shape, dtype=torch.float32, device=device, generator=generator)
    return ((values - 0.5) * 0.125).to(dtype).contiguous()


def _rank_order_sum(local: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    peers = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(peers, local, group=group)
    total = peers[0]
    for peer in peers[1:]:
        total = (total.float() + peer.float()).to(local.dtype)
    return total


def _assert_distributed_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    group: dist.ProcessGroup,
) -> float:
    difference = (actual.float() - expected.float()).abs()
    close = torch.isclose(
        actual.float(), expected.float(), atol=ATOL, rtol=RTOL, equal_nan=False
    ).all()
    failure = (~close).to(torch.int32)
    max_abs = torch.nan_to_num(difference, nan=float("inf")).max()
    dist.all_reduce(failure, op=dist.ReduceOp.MAX, group=group)
    dist.all_reduce(max_abs, op=dist.ReduceOp.MAX, group=group)
    if failure.item():
        raise AssertionError(
            f"{label} failed: max_abs={max_abs.item():.6g}, atol={ATOL}, rtol={RTOL}"
        )
    return float(max_abs.item())


def _make_case(
    *,
    world_size: int,
    rank: int,
    token_num: int,
    top_k: int,
    dtype: torch.dtype,
    device: torch.device,
    group: dist.ProcessGroup,
    workspace_ptrs: torch.Tensor,
    backend: str,
    launch_with_pdl: bool,
    output_profile: str,
    use_shared_expert: bool,
) -> tuple[Callable[[], None], Callable[[str], float]]:
    generator = torch.Generator(device=device).manual_seed(
        0xCA4E0000 + world_size * 10000 + rank * 100 + token_num + top_k
    )
    allreduce_in = _bounded_rand(
        (token_num * top_k, HIDDEN_SIZE),
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
        _bounded_rand((HIDDEN_SIZE,), dtype=dtype, device=device, generator=generator)
        + 1
    ).contiguous()
    expert_scales = _bounded_rand(
        (token_num, top_k), dtype=dtype, device=device, generator=generator
    )
    inverse_indices = torch.arange(
        token_num * top_k, dtype=torch.int32, device=device
    ).reshape(token_num, top_k)
    shared_expert_output = (
        _bounded_rand(
            (token_num, HIDDEN_SIZE),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        if use_shared_expert
        else None
    )
    routed_scaling_factor = 2.5 if use_shared_expert else None
    routed = 1.0 if routed_scaling_factor is None else routed_scaling_factor
    eps = 1e-5

    gathered = allreduce_in[inverse_indices]
    local = torch.zeros_like(residual_in)
    for route in range(top_k):
        contribution = (
            gathered[:, route].float() * expert_scales[:, route].float().unsqueeze(-1)
        ).to(dtype)
        local = (local.float() + contribution.float()).to(dtype)
    local = (local.float() * routed).to(dtype)
    if shared_expert_output is not None:
        local = (local.float() + shared_expert_output.float()).to(dtype)
    reduced = _rank_order_sum(local, group)
    residual_ref = (reduced.float() + residual_in.float()).to(dtype)
    residual_ref_f32 = residual_ref.float()
    norm_ref = (
        residual_ref_f32
        * torch.rsqrt(residual_ref_f32.square().mean(dim=-1, keepdim=True) + eps)
        * norm_weight.float()
    ).to(dtype)

    residual_out = torch.empty_like(residual_in)
    norm_out = torch.empty_like(residual_in)
    quant_out = None
    scale_out = None
    if output_profile == "111":
        quant_out = torch.zeros(residual_in.numel() // 4, dtype=dtype, device=device)
        padded_rows = ((token_num + 127) // 128) * 128
        padded_columns = ((HIDDEN_SIZE // 16 + 3) // 4) * 4
        scale_out = torch.zeros(
            padded_rows * padded_columns, dtype=dtype, device=device
        )

    def call() -> None:
        comm.trtllm_moe_finalize_allreduce_fusion(
            allreduce_in=allreduce_in,
            residual_in=residual_in,
            norm_weight=norm_weight,
            expanded_idx_to_permuted_idx=inverse_indices,
            norm_out=norm_out,
            residual_out=residual_out,
            quant_out=quant_out,
            scale_out=scale_out,
            workspace_ptrs=workspace_ptrs,
            launch_with_pdl=launch_with_pdl,
            world_rank=rank,
            world_size=world_size,
            eps=eps,
            shared_expert_output=shared_expert_output,
            expert_scale_factor=expert_scales,
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


def _measure(
    *,
    call: Callable[[], None],
    validate: Callable[[str], float],
    label: str,
    dry_run_iters: int,
    repeat_iters: int,
    group: dist.ProcessGroup,
) -> tuple[list[list[float]], list[float] | None, float, float]:
    dist.barrier(group=group)
    call()
    torch.cuda.synchronize()
    pre_max_abs = validate(f"{label}/pre")
    dist.barrier(group=group)
    samples = bench_gpu_time(
        call,
        enable_cupti=True,
        use_cuda_graph=False,
        cold_l2_cache=True,
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
        # bench_gpu_time synchronizes its result list when torch.distributed is
        # initialized. Select this caller's rank so the returned list remains
        # rank-local; the receipt aggregation below is deliberately separate.
        aggregate_op=lambda rank_values: rank_values[dist.get_rank(group)],
    )
    local_samples = [float(sample) for sample in samples]
    if len(local_samples) != repeat_iters:
        raise RuntimeError(
            f"CUPTI returned {len(local_samples)} samples, expected {repeat_iters}"
        )
    gathered_samples: list[list[float] | None] = [
        None for _ in range(dist.get_world_size(group))
    ]
    dist.all_gather_object(gathered_samples, local_samples, group=group)
    if any(
        rank_samples is None or len(rank_samples) != repeat_iters
        for rank_samples in gathered_samples
    ):
        lengths = [
            None if rank_samples is None else len(rank_samples)
            for rank_samples in gathered_samples
        ]
        raise RuntimeError(f"per-rank CUPTI sample counts differ: {lengths}")
    per_rank_samples = [
        [float(sample) for sample in rank_samples]
        for rank_samples in gathered_samples
        if rank_samples is not None
    ]
    rank_max_samples = None
    if dist.get_rank(group) == 0:
        rank_max_samples = [
            max(iteration_samples)
            for iteration_samples in zip(*per_rank_samples, strict=True)
        ]
    dist.barrier(group=group)
    call()
    torch.cuda.synchronize()
    post_max_abs = validate(f"{label}/post")
    return per_rank_samples, rank_max_samples, pre_max_abs, post_max_abs


def _create_workspace(
    rank: int, world_size: int, max_token_num: int, group: dist.ProcessGroup
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
        row["dtype"],
        row["token_num"],
        row["top_k"],
        row["launch_with_pdl"],
        row["output_profile"],
        row["shared_expert"],
    )


def _comparisons(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_comparison_key(row)].append(row)
    result = []
    for key, group_rows in grouped.items():
        baseline = [
            row["rank_max_median_ms"]
            for row in group_rows
            if row["backend"] == "trtllm"
        ]
        candidate = [
            row["rank_max_median_ms"] for row in group_rows if row["backend"] == "cake"
        ]
        if not baseline or not candidate:
            continue
        baseline_ms = float(statistics.median(baseline))
        candidate_ms = float(statistics.median(candidate))
        result.append(
            {
                "dtype": key[0],
                "token_num": key[1],
                "top_k": key[2],
                "launch_with_pdl": key[3],
                "output_profile": key[4],
                "shared_expert": key[5],
                "trtllm_median_ms": baseline_ms,
                "trtllm_leg_spread_ms": float(max(baseline) - min(baseline)),
                "cake_median_ms": candidate_ms,
                "speedup": baseline_ms / candidate_ms,
            }
        )
    return result


def _parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtypes", type=_csv, default=("float16", "bfloat16"))
    parser.add_argument("--tokens", type=_csv_int, default=(1, 16, 128, 2048))
    parser.add_argument("--top-k", type=_csv_int, default=(4, 8))
    parser.add_argument("--pdl", type=_csv, default=("false", "true"))
    parser.add_argument("--output-profiles", type=_csv, default=("110", "111"))
    parser.add_argument("--shared-expert", type=_csv, default=("false", "true"))
    parser.add_argument(
        "--backends",
        type=_csv,
        default=("trtllm", "cake", "trtllm"),
        help="ordered paired legs; repeats are retained",
    )
    parser.add_argument("--dry-run-iters", type=int, default=5)
    parser.add_argument("--repeat-iters", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    _require_choices(parser, "--dtypes", args.dtypes, set(_DTYPES))
    _require_choices(parser, "--pdl", args.pdl, {"false", "true"})
    _require_choices(parser, "--output-profiles", args.output_profiles, {"110", "111"})
    _require_choices(parser, "--shared-expert", args.shared_expert, {"false", "true"})
    _require_choices(parser, "--backends", args.backends, {"trtllm", "cake"})
    if any(top_k not in (4, 8) for top_k in args.top_k):
        parser.error("--top-k supports only 4 and 8")
    if args.dry_run_iters <= 0 or args.repeat_iters <= 0:
        parser.error("iteration counts must be positive")
    if not {"trtllm", "cake"}.issubset(args.backends):
        parser.error("--backends must include trtllm and cake")
    if len(args.backends) > 3:
        parser.error("--backends supports at most three ordered legs")
    for label, values in (
        ("--dtypes", args.dtypes),
        ("--tokens", args.tokens),
        ("--top-k", args.top_k),
        ("--pdl", args.pdl),
        ("--output-profiles", args.output_profiles),
        ("--shared-expert", args.shared_expert),
    ):
        if len(values) != len(set(values)):
            parser.error(f"{label} must not contain duplicates")
    total_legs = (
        len(args.dtypes)
        * len(args.tokens)
        * len(args.top_k)
        * len(args.pdl)
        * len(args.output_profiles)
        * len(args.shared_expert)
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
    if world_size not in (2, 4, 8):
        parser.error("world size must be 2, 4, or 8")
    if local_world_size != world_size:
        parser.error("benchmark requires a single node")
    for dtype_name in args.dtypes:
        dtype = _DTYPES[dtype_name]
        for token_num in args.tokens:
            required_lamport_comm_size = (
                token_num * HIDDEN_SIZE * dtype.itemsize * world_size
            )
            if required_lamport_comm_size > MAX_COMM_SIZE:
                parser.error(
                    f"tokens={token_num}, dtype={dtype_name}, and TP{world_size} "
                    f"require {required_lamport_comm_size} Lamport bytes, above "
                    f"MAX_COMM_SIZE={MAX_COMM_SIZE}"
                )
    if local_rank >= torch.cuda.device_count():
        parser.error(f"local rank {local_rank} has no visible CUDA device")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    if capability not in ((10, 0), (10, 3)):
        parser.error(f"benchmark requires SM100 or SM103, got {capability}")
    arch = cake_finalize.target_arch(local_rank)
    started = time.monotonic()
    cupti_version, cupti_module = _require_cupti()
    cupti_finalize = _defer_cupti_finalize(cupti_module)
    rows: list[dict[str, Any]] = []
    group: dist.ProcessGroup | None = None
    try:
        dist.init_process_group(backend="nccl", init_method="env://")
        group = dist.group.WORLD
        if "cake" in args.backends:
            cake_finalize.get_cake_moe_finalize_module_specs()
        if "trtllm" in args.backends:
            get_trtllm_comm_module()
        dist.barrier(group=group)

        leg_index = 0
        for dtype_name in args.dtypes:
            dtype = _DTYPES[dtype_name]
            for token_num in args.tokens:
                for top_k in args.top_k:
                    for pdl_name in args.pdl:
                        launch_with_pdl = pdl_name == "true"
                        for output_profile in args.output_profiles:
                            for shared_name in args.shared_expert:
                                use_shared_expert = shared_name == "true"
                                for backend in args.backends:
                                    handles, workspace_ptrs = _create_workspace(
                                        local_rank,
                                        world_size,
                                        max(args.tokens),
                                        group,
                                    )
                                    try:
                                        call, validate = _make_case(
                                            world_size=world_size,
                                            rank=rank,
                                            token_num=token_num,
                                            top_k=top_k,
                                            dtype=dtype,
                                            device=device,
                                            group=group,
                                            workspace_ptrs=workspace_ptrs,
                                            backend=backend,
                                            launch_with_pdl=launch_with_pdl,
                                            output_profile=output_profile,
                                            use_shared_expert=use_shared_expert,
                                        )
                                        label = (
                                            f"tp{world_size}/{dtype_name}/tokens{token_num}/"
                                            f"topk{top_k}/pdl{int(launch_with_pdl)}/"
                                            f"o{output_profile}/shared{int(use_shared_expert)}/"
                                            f"{backend}/leg{leg_index}"
                                        )
                                        (
                                            per_rank_samples,
                                            rank_max_samples,
                                            pre_max_abs,
                                            post_max_abs,
                                        ) = _measure(
                                            call=call,
                                            validate=validate,
                                            label=label,
                                            dry_run_iters=args.dry_run_iters,
                                            repeat_iters=args.repeat_iters,
                                            group=group,
                                        )
                                        if rank == 0:
                                            if rank_max_samples is None:
                                                raise RuntimeError(
                                                    "rank 0 did not produce rank-max samples"
                                                )
                                            rows.append(
                                                {
                                                    "leg_index": leg_index,
                                                    "backend": backend,
                                                    "world_size": world_size,
                                                    "dtype": dtype_name,
                                                    "token_num": token_num,
                                                    "top_k": top_k,
                                                    "launch_with_pdl": launch_with_pdl,
                                                    "output_profile": output_profile,
                                                    "shared_expert": use_shared_expert,
                                                    "per_rank_samples_ms": (
                                                        per_rank_samples
                                                    ),
                                                    "per_rank_median_ms": [
                                                        float(
                                                            statistics.median(
                                                                rank_samples
                                                            )
                                                        )
                                                        for rank_samples in per_rank_samples
                                                    ],
                                                    "rank_max_samples_ms": (
                                                        rank_max_samples
                                                    ),
                                                    "rank_max_median_ms": float(
                                                        statistics.median(
                                                            rank_max_samples
                                                        )
                                                    ),
                                                    "rank_max_min_ms": float(
                                                        min(rank_max_samples)
                                                    ),
                                                    "pre_max_abs": pre_max_abs,
                                                    "post_max_abs": post_max_abs,
                                                }
                                            )
                                    finally:
                                        dist.barrier(group=group)
                                        comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
                                            handles, group=group
                                        )
                                    leg_index += 1

        dist.barrier(group=group)
        if rank == 0:
            manifest_path = cake_finalize.get_cake_moe_finalize_manifest_path()
            report = {
                "schema_version": 1,
                "world_size": world_size,
                "gpu": torch.cuda.get_device_name(device),
                "compute_capability": list(capability),
                "cake_target_arch": arch,
                "cupti_python_version": cupti_version,
                "timing": {
                    "method": "bench_gpu_time",
                    "enable_cupti": True,
                    "fallback_count": 0,
                    "cold_l2_cache": True,
                    "rank_local_samples": (
                        "bench_gpu_time CUPTI result selected for the caller rank"
                    ),
                    "receipt_aggregation": (
                        "post-measurement all-gather of equal-length rank samples; "
                        "rank 0 computes each iteration's maximum"
                    ),
                    "dry_run_iters": args.dry_run_iters,
                    "repeat_iters": args.repeat_iters,
                },
                "source": {
                    "manifest_sha256": hashlib.sha256(
                        manifest_path.read_bytes()
                    ).hexdigest(),
                },
                "rows": rows,
                "comparisons": _comparisons(rows),
                "physical_runtime_seconds": time.monotonic() - started,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(report, sort_keys=True))
    finally:
        try:
            if dist.is_initialized() and group is not None:
                dist.destroy_process_group(group=group)
        finally:
            cupti_module.finalize = cupti_finalize
    cupti_finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
