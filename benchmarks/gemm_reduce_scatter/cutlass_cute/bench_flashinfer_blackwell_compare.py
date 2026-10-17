#!/usr/bin/env python3
"""Compare FlashInfer CUTLASS Blackwell GEMM+RS with naive and vLLM fused.

Run under torchrun inside a Slurm allocation.  This benchmark exercises the
FlashInfer library backend, not the standalone CUTLASS example runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

try:
    from cuda.core import Device
except ImportError:  # pragma: no cover - depends on installed CUDA Python.
    from cuda.core.experimental import Device

import nvshmem.core

from flashinfer.comm.gemm_reduce_scatter import (
    BlackwellGemmRSConfig,
    BlackwellGemmRSWorkspace,
    gemm_reduce_scatter,
)


REPO = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO / "benchmarks" / "gemm_reduce_scatter" / "results" / "cutlass_cute"


def _torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _rank0_print(msg: str) -> None:
    if dist.get_rank() == 0:
        print(msg, flush=True)


def _stats_ms(values: list[float]) -> dict[str, float | list[float]]:
    sorted_values = sorted(values)
    count = len(sorted_values)
    return {
        "mean_ms": sum(sorted_values) / count,
        "std_ms": statistics.stdev(sorted_values) if count > 1 else 0.0,
        "min_ms": sorted_values[0],
        "median_ms": sorted_values[count // 2],
        "max_ms": sorted_values[-1],
        "raw_ms": values,
    }


def _time_cuda_events_ms(fn, warmup: int, iterations: int, group) -> dict[str, Any]:
    for _ in range(warmup):
        dist.barrier(group)
        fn()
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iterations):
        dist.barrier(group)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    dist.barrier(group)
    return _stats_ms(times)


def _gather_rank_results(
    local_result: dict[str, Any], group
) -> list[dict[str, Any]] | None:
    gathered: list[dict[str, Any] | None] | None
    if dist.get_rank(group) == 0:
        gathered = [None for _ in range(dist.get_world_size(group))]
    else:
        gathered = None
    dist.gather_object(local_result, gathered, dst=0, group=group)
    return gathered  # type: ignore[return-value]


def _max_rank_mean(gathered: list[dict[str, Any]], backend: str) -> float | None:
    vals = []
    for item in gathered:
        data = item.get(backend, {})
        if data.get("ok") and data.get("mean_ms") is not None:
            vals.append(float(data["mean_ms"]))
    return max(vals) if vals else None


def _all_ok(gathered: list[dict[str, Any]], backend: str) -> bool:
    return all(item.get(backend, {}).get("ok", False) for item in gathered)


def _all_flag(gathered: list[dict[str, Any]], key: str) -> bool:
    return all(bool(item.get(key, False)) for item in gathered)


def _joined_errors(gathered: list[dict[str, Any]]) -> str:
    notes = []
    for idx, item in enumerate(gathered):
        for backend in ("vllm_fused", "flashinfer_cutlass_blackwell"):
            if not item.get(backend, {}).get("ok", False):
                notes.append(
                    f"rank{idx}.{backend}={item.get(backend, {}).get('error')}"
                )
        if not item.get("vllm_correct_vs_naive", False):
            notes.append(f"rank{idx}.vllm_correct={item.get('vllm_correct_error')}")
        if not item.get("flashinfer_correct_vs_naive", False):
            notes.append(
                f"rank{idx}.flashinfer_correct={item.get('flashinfer_correct_error')}"
            )
        if not item.get("flashinfer_stress_ok", True):
            notes.append(
                f"rank{idx}.flashinfer_stress={item.get('flashinfer_stress_error')}"
            )
    return "; ".join(notes)


def torchrun_nvshmem_init() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dev = Device(local_rank)
    dev.set_current()

    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    uid = nvshmem.core.get_unique_id(empty=(rank != 0))
    uid_bytes = uid._data.view(np.uint8).copy()
    uid_tensor = torch.from_numpy(uid_bytes).cuda()
    dist.broadcast(uid_tensor, src=0)
    dist.barrier()
    uid._data[:] = uid_tensor.cpu().numpy().view(uid._data.dtype)

    nvshmem.core.init(
        device=dev,
        uid=uid,
        rank=rank,
        nranks=world_size,
        initializer_method="uid",
    )


def torchrun_nvshmem_finalize() -> None:
    nvshmem.core.finalize()
    dist.destroy_process_group()


def _run_flashinfer_stress(
    *,
    args,
    group,
    dtype: torch.dtype,
    device: torch.device,
    m: int,
    k_local: int,
    n: int,
    workspace: BlackwellGemmRSWorkspace,
) -> None:
    """Stress repeated backend calls on one workspace."""
    if args.stress_loops <= 0:
        return

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    m_local = m // world_size
    pool_size = max(1, args.stress_pointer_pool)
    xs = []
    ws = []
    refs = []
    for slot in range(pool_size):
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed + 1009 * rank + 104729 * slot)
        xs.append(torch.randn((m, k_local), device=device, dtype=dtype, generator=gen))
        ws.append(torch.randn((k_local, n), device=device, dtype=dtype, generator=gen))
        refs.append(torch.empty((m_local, n), device=device, dtype=dtype))

    for loop in range(args.stress_loops):
        slot = loop % pool_size
        dist.barrier(group)
        partial = xs[slot] @ ws[slot]
        dist.reduce_scatter_tensor(refs[slot], partial, group=group)
        got = gemm_reduce_scatter(
            xs[slot],
            ws[slot],
            group,
            workspace=workspace,
            verbose=False,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(got, refs[slot], atol=args.atol, rtol=args.rtol)


def _run_one_shape(args, m: int, k_total: int, group) -> dict[str, Any]:
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    dtype = _torch_dtype(args.dtype)
    k_local = k_total // world_size
    m_local = m // world_size

    torch.manual_seed(args.seed + rank)
    x = symm_mem.empty((m, k_local), device=device, dtype=dtype).normal_()
    w = torch.randn((k_local, args.n), device=device, dtype=dtype)
    out_naive = torch.empty((m_local, args.n), device=device, dtype=dtype)
    workspace = BlackwellGemmRSWorkspace(
        M=m,
        N=args.n,
        K_local=k_local,
        group=group,
        dtype=dtype,
        device=device,
        config=BlackwellGemmRSConfig(
            mma_tiler_mn=(args.mma_m, args.mma_n),
            cluster_shape_mn=(args.cluster_m, args.cluster_n),
            b_layout=args.b_layout,
        ),
    )

    def cublas_naive():
        partial = x @ w
        dist.reduce_scatter_tensor(out_naive, partial, group=group)
        return out_naive

    def vllm_fused():
        return torch.ops.symm_mem.fused_matmul_reduce_scatter(
            x, w, "sum", scatter_dim=0, group_name=group.group_name
        )

    def flashinfer_cutlass():
        return gemm_reduce_scatter(
            x,
            w,
            group,
            workspace=workspace,
            verbose=args.verbose and rank == 0,
        )

    result: dict[str, Any] = {
        "M": m,
        "N": args.n,
        "K_total": k_total,
        "K_local": k_local,
    }
    try:
        try:
            _run_flashinfer_stress(
                args=args,
                group=group,
                dtype=dtype,
                device=device,
                m=m,
                k_local=k_local,
                n=args.n,
                workspace=workspace,
            )
            result["flashinfer_stress_ok"] = True
            result["flashinfer_stress_loops"] = args.stress_loops
            result["flashinfer_stress_pointer_pool"] = args.stress_pointer_pool
        except Exception as exc:
            result["flashinfer_stress_ok"] = False
            result["flashinfer_stress_error"] = repr(exc)
            result["flashinfer_stress_traceback"] = traceback.format_exc()

        try:
            ref = cublas_naive()
            got = vllm_fused()
            torch.cuda.synchronize()
            torch.testing.assert_close(got, ref, atol=args.atol, rtol=args.rtol)
            result["vllm_correct_vs_naive"] = True
        except Exception as exc:
            result["vllm_correct_vs_naive"] = False
            result["vllm_correct_error"] = repr(exc)
            result["vllm_correct_traceback"] = traceback.format_exc()

        try:
            ref = cublas_naive()
            got = flashinfer_cutlass()
            torch.cuda.synchronize()
            torch.testing.assert_close(got, ref, atol=args.atol, rtol=args.rtol)
            result["flashinfer_correct_vs_naive"] = True
        except Exception as exc:
            result["flashinfer_correct_vs_naive"] = False
            result["flashinfer_correct_error"] = repr(exc)
            result["flashinfer_correct_traceback"] = traceback.format_exc()

        for name, fn in (
            ("cublas_naive", cublas_naive),
            ("vllm_fused", vllm_fused),
            ("flashinfer_cutlass_blackwell", flashinfer_cutlass),
        ):
            try:
                result[name] = {
                    "ok": True,
                    **_time_cuda_events_ms(fn, args.warmup, args.iterations, group),
                }
            except Exception as exc:
                result[name] = {
                    "ok": False,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
    finally:
        workspace.destroy()
    return result


def _write_outputs(
    args, rows: list[dict[str, Any]], rank_details: list[dict[str, Any]]
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = (
        RESULTS_DIR
        / f"flashinfer_blackwell_compare_ws{dist.get_world_size()}_{ts}.json"
    )
    csv_path = (
        RESULTS_DIR / f"flashinfer_blackwell_compare_ws{dist.get_world_size()}_{ts}.csv"
    )
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "host": os.uname().nodename,
        "world_size": dist.get_world_size(),
        "torch": torch.__version__,
        "dtype": args.dtype,
        "seed": args.seed,
        "backend": "cutlass_blackwell",
        "b_layout": args.b_layout,
        "note": (
            "FlashInfer CUTLASS Blackwell currently stages W_local.T into "
            "workspace.w_staging each call when b_layout='staged'. "
            "b_layout='nocopy' creates a logical B[N,K,1] view over native "
            "W_local[K,N]. Optional stress mode reuses one workspace across "
            "repeated calls and can rotate input/weight pointers to exercise "
            "compiled-cache invalidation and barrier-flag reuse."
        ),
        "stress_loops": args.stress_loops,
        "stress_pointer_pool": args.stress_pointer_pool,
    }

    with json_path.open("w") as f:
        json.dump(
            {"metadata": metadata, "rows": rows, "rank_details": rank_details},
            f,
            indent=2,
        )

    fields = [
        "M",
        "N",
        "K_total",
        "K_local",
        "world_size",
        "dtype",
        "b_layout",
        "cublas_naive_ms",
        "vllm_fused_ms",
        "flashinfer_cutlass_blackwell_ms",
        "vllm_ok",
        "flashinfer_ok",
        "vllm_correct",
        "flashinfer_correct",
        "flashinfer_stress_ok",
        "flashinfer_stress_loops",
        "flashinfer_stress_pointer_pool",
        "flashinfer_vs_vllm_speedup",
        "flashinfer_vs_naive_speedup",
        "notes",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    print(f"JSON: {json_path}", flush=True)
    print(f"CSV:  {csv_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-values", type=int, nargs="+", default=[2048])
    parser.add_argument("--k-total-values", type=int, nargs="+", default=[8192])
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--mma-m", type=int, default=256)
    parser.add_argument("--mma-n", type=int, default=256)
    parser.add_argument("--cluster-m", type=int, default=2)
    parser.add_argument("--cluster-n", type=int, default=1)
    parser.add_argument("--b-layout", choices=["staged", "nocopy"], default="nocopy")
    parser.add_argument("--atol", type=float, default=1e-1)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument(
        "--stress-loops",
        type=int,
        default=0,
        help=(
            "Run this many extra FlashInfer correctness loops on the same "
            "workspace before timing to stress barrier-flag reuse."
        ),
    )
    parser.add_argument(
        "--stress-pointer-pool",
        type=int,
        default=1,
        help=(
            "Number of distinct X/W tensor pointer pairs to rotate during "
            "stress mode. Use values greater than 1 to exercise cache invalidation."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torchrun_nvshmem_init()
    try:
        group = dist.group.WORLD
        world_size = dist.get_world_size(group)
        if any(k_total % world_size != 0 for k_total in args.k_total_values):
            raise ValueError(f"All K totals must divide world_size={world_size}.")
        if any(m % world_size != 0 for m in args.m_values):
            raise ValueError(f"All M values must divide world_size={world_size}.")

        _rank0_print(
            f"flashinfer_blackwell_compare: ws={world_size} N={args.n} "
            f"M={args.m_values} K_total={args.k_total_values} dtype={args.dtype} "
            f"warmup={args.warmup} iterations={args.iterations}"
        )

        rows: list[dict[str, Any]] = []
        rank_details: list[dict[str, Any]] = []
        for k_total in args.k_total_values:
            for m in args.m_values:
                _rank0_print(f"===== M={m} K_total={k_total} =====")
                local_result = _run_one_shape(args, m, k_total, group)
                gathered = _gather_rank_results(local_result, group)
                if dist.get_rank(group) != 0:
                    continue

                assert gathered is not None
                rank_details.extend(gathered)
                naive_ms = _max_rank_mean(gathered, "cublas_naive")
                fused_ms = _max_rank_mean(gathered, "vllm_fused")
                flashinfer_ms = _max_rank_mean(gathered, "flashinfer_cutlass_blackwell")
                row = {
                    "M": m,
                    "N": args.n,
                    "K_total": k_total,
                    "K_local": k_total // world_size,
                    "world_size": world_size,
                    "dtype": args.dtype,
                    "b_layout": args.b_layout,
                    "cublas_naive_ms": naive_ms,
                    "vllm_fused_ms": fused_ms,
                    "flashinfer_cutlass_blackwell_ms": flashinfer_ms,
                    "vllm_ok": _all_ok(gathered, "vllm_fused"),
                    "flashinfer_ok": _all_ok(gathered, "flashinfer_cutlass_blackwell"),
                    "vllm_correct": all(
                        item.get("vllm_correct_vs_naive", False) for item in gathered
                    ),
                    "flashinfer_correct": all(
                        item.get("flashinfer_correct_vs_naive", False)
                        for item in gathered
                    ),
                    "flashinfer_stress_ok": _all_flag(gathered, "flashinfer_stress_ok"),
                    "flashinfer_stress_loops": args.stress_loops,
                    "flashinfer_stress_pointer_pool": args.stress_pointer_pool,
                    "flashinfer_vs_vllm_speedup": (
                        fused_ms / flashinfer_ms if fused_ms and flashinfer_ms else ""
                    ),
                    "flashinfer_vs_naive_speedup": (
                        naive_ms / flashinfer_ms if naive_ms and flashinfer_ms else ""
                    ),
                    "notes": _joined_errors(gathered),
                }
                rows.append(row)
                _rank0_print(str(row))

        if dist.get_rank(group) == 0:
            _write_outputs(args, rows, rank_details)
    finally:
        torchrun_nvshmem_finalize()


if __name__ == "__main__":
    main()
