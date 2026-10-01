"""Benchmark Cake MoE finalize against TRT-LLM on TP2, TP4, or TP8.

Launch with ``torchrun --nproc-per-node {2,4,8}``. Every timed leg uses CUPTI
GPU activity timing with cold L2 and is bracketed by distributed correctness
checks. The default TRT-LLM/Cake/TRT-LLM leg order exposes same-session drift.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm.trtllm_ar import MAX_COMM_SIZE, get_trtllm_comm_module
from flashinfer.jit import cake_moe_finalize_comm as cake_finalize
from flashinfer.testing.utils import bench_gpu_time

HIDDEN_SIZE = cake_finalize.HIDDEN_DIM
ATOL = 1e-2
RTOL = 1e-2
_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}
_BOOL = {"false": False, "true": True}


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
    close = torch.isclose(actual.float(), expected.float(), atol=ATOL, rtol=RTOL).all()
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

    def rand(shape: tuple[int, ...]) -> torch.Tensor:
        return _bounded_rand(shape, dtype=dtype, device=device, generator=generator)

    allreduce_in = rand((token_num * top_k, HIDDEN_SIZE))
    residual_in = rand((token_num, HIDDEN_SIZE))
    norm_weight = (rand((HIDDEN_SIZE,)) + 1).contiguous()
    expert_scales = rand((token_num, top_k))
    inverse_indices = torch.arange(
        token_num * top_k, dtype=torch.int32, device=device
    ).reshape(token_num, top_k)
    shared_expert_output = rand((token_num, HIDDEN_SIZE)) if use_shared_expert else None
    routed_scaling_factor = 2.5 if use_shared_expert else None
    eps = 1e-5

    # Reference in the kernel's rounding order.
    gathered = allreduce_in[inverse_indices]
    local = torch.zeros_like(residual_in)
    for route in range(top_k):
        contribution = (
            gathered[:, route].float() * expert_scales[:, route].float().unsqueeze(-1)
        ).to(dtype)
        local = (local.float() + contribution.float()).to(dtype)
    local = (local.float() * (routed_scaling_factor or 1.0)).to(dtype)
    if shared_expert_output is not None:
        local = (local.float() + shared_expert_output.float()).to(dtype)
    residual_ref = (_rank_order_sum(local, group).float() + residual_in.float()).to(dtype)
    residual_f32 = residual_ref.float()
    norm_ref = (
        residual_f32
        * torch.rsqrt(residual_f32.square().mean(dim=-1, keepdim=True) + eps)
        * norm_weight.float()
    ).to(dtype)

    residual_out = torch.empty_like(residual_in)
    norm_out = torch.empty_like(residual_in)
    quant_out = scale_out = None
    if output_profile == "111":
        quant_out = torch.zeros(residual_in.numel() // 2, dtype=torch.uint8, device=device)
        padded_rows = ((token_num + 127) // 128) * 128
        padded_columns = ((HIDDEN_SIZE // 16 + 3) // 4) * 4
        scale_out = torch.zeros(
            padded_rows * padded_columns, dtype=torch.float8_e4m3fn, device=device
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
                residual_out, residual_ref, label=f"{stage}/residual_out", group=group
            ),
            _assert_distributed_close(
                norm_out, norm_ref, label=f"{stage}/norm_out", group=group
            ),
        )

    return call, validate


def _measure(
    call: Callable[[], None],
    validate: Callable[[str], float],
    *,
    label: str,
    dry_run_iters: int,
    repeat_iters: int,
    group: dist.ProcessGroup,
) -> dict[str, Any]:
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
        # Keep the returned list rank-local; the rank-max rollup happens below.
        aggregate_op=lambda rank_values: rank_values[dist.get_rank(group)],
    )
    local_samples = [float(sample) for sample in samples]
    gathered: list[list[float] | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, local_samples, group=group)
    per_rank = [list(rank_samples or []) for rank_samples in gathered]
    if any(len(rank_samples) != repeat_iters for rank_samples in per_rank):
        raise RuntimeError(f"per-rank CUPTI sample counts differ: {list(map(len, per_rank))}")
    rank_max = [max(iteration) for iteration in zip(*per_rank, strict=True)]
    dist.barrier(group=group)
    call()
    torch.cuda.synchronize()
    post_max_abs = validate(f"{label}/post")
    return {
        "per_rank_median_ms": [statistics.median(r) for r in per_rank],
        "rank_max_median_ms": statistics.median(rank_max),
        "rank_max_min_ms": min(rank_max),
        "pre_max_abs": pre_max_abs,
        "post_max_abs": post_max_abs,
    }


def _comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("dtype", "token_num", "top_k", "launch_with_pdl", "output_profile", "shared_expert")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, group_rows in grouped.items():
        legs = {
            backend: [r["rank_max_median_ms"] for r in group_rows if r["backend"] == backend]
            for backend in ("trtllm", "cake")
        }
        if not legs["trtllm"] or not legs["cake"]:
            continue
        trtllm_ms = statistics.median(legs["trtllm"])
        cake_ms = statistics.median(legs["cake"])
        result.append(
            {
                **dict(zip(keys, key, strict=True)),
                "trtllm_median_ms": trtllm_ms,
                "trtllm_leg_spread_ms": max(legs["trtllm"]) - min(legs["trtllm"]),
                "cake_median_ms": cake_ms,
                "speedup": trtllm_ms / cake_ms,
            }
        )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtypes", nargs="+", choices=sorted(_DTYPES), default=["float16", "bfloat16"])
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 16, 128, 2048])
    parser.add_argument("--top-k", nargs="+", type=int, choices=[4, 8], default=[4, 8])
    parser.add_argument("--pdl", nargs="+", choices=sorted(_BOOL), default=["false", "true"])
    parser.add_argument("--output-profiles", nargs="+", choices=["110", "111"], default=["110", "111"])
    parser.add_argument("--shared-expert", nargs="+", choices=sorted(_BOOL), default=["false", "true"])
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["trtllm", "cake"],
        default=["trtllm", "cake", "trtllm"],
        help="ordered paired legs; repeats are retained",
    )
    parser.add_argument("--dry-run-iters", type=int, default=5)
    parser.add_argument("--repeat-iters", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not all(name in os.environ for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        parser.error("launch with torchrun so RANK, LOCAL_RANK, and WORLD_SIZE are set")
    if int(os.environ["WORLD_SIZE"]) not in (2, 4, 8):
        parser.error("world size must be 2, 4, or 8")
    if int(os.environ.get("LOCAL_WORLD_SIZE", os.environ["WORLD_SIZE"])) != int(
        os.environ["WORLD_SIZE"]
    ):
        parser.error("benchmark requires a single node")
    return args


def main() -> int:
    args = _parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    arch = cake_finalize.target_arch(local_rank)
    for dtype_name in args.dtypes:
        for token_num in args.tokens:
            lamport_bytes = token_num * HIDDEN_SIZE * _DTYPES[dtype_name].itemsize * world_size
            if lamport_bytes > MAX_COMM_SIZE:
                raise SystemExit(
                    f"tokens={token_num} {dtype_name} TP{world_size} needs {lamport_bytes} "
                    f"Lamport bytes, above MAX_COMM_SIZE={MAX_COMM_SIZE}"
                )

    from cupti import cupti as cupti_module  # CUPTI timing only; no fallback backend

    # Keep process-global CUPTI state alive until the NCCL watchdog exits.
    cupti_finalize, cupti_module.finalize = cupti_module.finalize, lambda: None
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    try:
        dist.init_process_group(backend="nccl", init_method="env://")
        group = dist.group.WORLD
        if "cake" in args.backends:
            cake_finalize.get_cake_moe_finalize_module(arch)
        if "trtllm" in args.backends:
            get_trtllm_comm_module()
        dist.barrier(group=group)
        cases = [
            (dtype_name, token_num, top_k, _BOOL[pdl], profile, _BOOL[shared])
            for dtype_name in args.dtypes
            for token_num in args.tokens
            for top_k in args.top_k
            for pdl in args.pdl
            for profile in args.output_profiles
            for shared in args.shared_expert
        ]
        leg_index = 0
        for dtype_name, token_num, top_k, launch_with_pdl, profile, use_shared in cases:
            for backend in args.backends:
                handles, workspace_ptrs = comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                    local_rank, world_size, max(args.tokens), HIDDEN_SIZE, group=group
                )
                try:
                    call, validate = _make_case(
                        world_size=world_size,
                        rank=rank,
                        token_num=token_num,
                        top_k=top_k,
                        dtype=_DTYPES[dtype_name],
                        device=device,
                        group=group,
                        workspace_ptrs=workspace_ptrs,
                        backend=backend,
                        launch_with_pdl=launch_with_pdl,
                        output_profile=profile,
                        use_shared_expert=use_shared,
                    )
                    label = (
                        f"tp{world_size}/{dtype_name}/tokens{token_num}/topk{top_k}/"
                        f"pdl{int(launch_with_pdl)}/o{profile}/shared{int(use_shared)}/"
                        f"{backend}/leg{leg_index}"
                    )
                    measured = _measure(
                        call,
                        validate,
                        label=label,
                        dry_run_iters=args.dry_run_iters,
                        repeat_iters=args.repeat_iters,
                        group=group,
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
                            "output_profile": profile,
                            "shared_expert": use_shared,
                            **measured,
                        }
                    )
                finally:
                    dist.barrier(group=group)
                    comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(handles, group=group)
                leg_index += 1
        dist.barrier(group=group)
        if rank == 0:
            report = {
                "world_size": world_size,
                "gpu": torch.cuda.get_device_name(device),
                "cake_target_arch": arch,
                "timing": {
                    "method": "bench_gpu_time",
                    "enable_cupti": True,
                    "cold_l2_cache": True,
                    "dry_run_iters": args.dry_run_iters,
                    "repeat_iters": args.repeat_iters,
                    "rollup": "per-iteration maximum over ranks, median over iterations",
                },
                "rows": rows,
                "comparisons": _comparisons(rows),
                "physical_runtime_seconds": time.monotonic() - started,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            for row in report["comparisons"]:
                print(
                    f"{row['dtype']:>8} tokens={row['token_num']:<5} top_k={row['top_k']} "
                    f"pdl={int(row['launch_with_pdl'])} o{row['output_profile']} "
                    f"shared={int(row['shared_expert'])}: trtllm {row['trtllm_median_ms']*1e3:8.2f} us  "
                    f"cake {row['cake_median_ms']*1e3:8.2f} us  x{row['speedup']:.3f}"
                )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        cupti_module.finalize = cupti_finalize
    cupti_finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
