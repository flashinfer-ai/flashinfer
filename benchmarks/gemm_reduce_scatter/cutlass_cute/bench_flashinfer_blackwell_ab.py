#!/usr/bin/env python3
"""Same-node A/B comparison for FlashInfer Blackwell B layouts.

Runs all contenders in one torchrun job on the same tensors:

- naive torch.mm + reduce_scatter_tensor
- vLLM fused symm_mem op
- FlashInfer CUTLASS Blackwell with staged B layout
- FlashInfer CUTLASS Blackwell with no-copy B layout
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
    return {
        "mean_ms": sum(values) / len(values),
        "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_ms": sorted_values[0],
        "median_ms": statistics.median(sorted_values),
        "p75_ms": float(np.percentile(sorted_values, 75)),
        "max_ms": sorted_values[-1],
        # Preserve iteration order so rank 0 can form the distributed
        # critical path by taking the maximum rank latency per iteration.
        "raw_ms": values,
    }


def _time_cuda_events_ms(fn, warmup: int, iterations: int, group) -> dict[str, Any]:
    for _ in range(warmup):
        dist.barrier(group)
        fn()
        torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        # Do not overlap a backend call with any previous GPU work.
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


def _critical_path_stats(
    gathered: list[dict[str, Any]], key: str
) -> dict[str, float] | None:
    """Summarize per-iteration completion time of the slowest rank."""
    rank_times = []
    for item in gathered:
        data = item.get(key, {})
        if not data.get("ok") or not data.get("raw_ms"):
            return None
        rank_times.append([float(value) for value in data["raw_ms"]])

    lengths = {len(values) for values in rank_times}
    if len(lengths) != 1:
        raise ValueError(f"Mismatched raw timing counts for {key}: {sorted(lengths)}")

    critical = [max(samples) for samples in zip(*rank_times, strict=True)]
    return {
        "median_ms": float(np.percentile(critical, 50)),
        "p75_ms": float(np.percentile(critical, 75)),
        "max_ms": max(critical),
    }


def _gather(local_result: dict[str, Any], group) -> list[dict[str, Any]] | None:
    gathered = (
        [None for _ in range(dist.get_world_size(group))]
        if dist.get_rank(group) == 0
        else None
    )
    dist.gather_object(local_result, gathered, dst=0, group=group)
    return gathered  # type: ignore[return-value]


def _max_rank_mean(gathered: list[dict[str, Any]], key: str) -> float | None:
    vals = []
    for item in gathered:
        data = item.get(key, {})
        if data.get("ok") and data.get("mean_ms") is not None:
            vals.append(float(data["mean_ms"]))
    return max(vals) if vals else None


def _all_ok(gathered: list[dict[str, Any]], key: str) -> bool:
    return all(item.get(key, {}).get("ok", False) for item in gathered)


def _all_correct(gathered: list[dict[str, Any]], key: str) -> bool:
    return all(item.get(key, False) for item in gathered)


def _joined_errors(gathered: list[dict[str, Any]]) -> str:
    notes = []
    for rank, item in enumerate(gathered):
        for key in ("vllm_fused", "flashinfer_staged", "flashinfer_nocopy"):
            if not item.get(key, {}).get("ok", False):
                notes.append(f"rank{rank}.{key}={item.get(key, {}).get('error')}")
        for key in ("vllm_correct", "staged_correct", "nocopy_correct"):
            if not item.get(key, False):
                notes.append(f"rank{rank}.{key}={item.get(key + '_error')}")
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


def _make_workspace(
    args, m: int, n: int, k_local: int, dtype, device, group, b_layout: str
):
    return BlackwellGemmRSWorkspace(
        M=m,
        N=n,
        K_local=k_local,
        group=group,
        dtype=dtype,
        device=device,
        config=BlackwellGemmRSConfig(
            mma_tiler_mn=(args.mma_m, args.mma_n),
            cluster_shape_mn=(args.cluster_m, args.cluster_n),
            b_layout=b_layout,
        ),
    )


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
    ws_staged = _make_workspace(
        args, m, args.n, k_local, dtype, device, group, "staged"
    )
    ws_nocopy = _make_workspace(
        args, m, args.n, k_local, dtype, device, group, "nocopy"
    )

    def naive():
        partial = x @ w
        dist.reduce_scatter_tensor(out_naive, partial, group=group)
        return out_naive

    def vllm():
        return torch.ops.symm_mem.fused_matmul_reduce_scatter(
            x, w, "sum", scatter_dim=0, group_name=group.group_name
        )

    def staged():
        return gemm_reduce_scatter(
            x,
            w,
            group,
            workspace=ws_staged,
            verbose=args.verbose and rank == 0,
        )

    def nocopy():
        return gemm_reduce_scatter(
            x,
            w,
            group,
            workspace=ws_nocopy,
            verbose=args.verbose and rank == 0,
        )

    result: dict[str, Any] = {
        "M": m,
        "N": args.n,
        "K_total": k_total,
        "K_local": k_local,
    }
    try:
        ref = naive().clone()
        torch.cuda.synchronize()
        for key, fn in (
            ("vllm_correct", vllm),
            ("staged_correct", staged),
            ("nocopy_correct", nocopy),
        ):
            try:
                got = fn()
                torch.cuda.synchronize()
                torch.testing.assert_close(got, ref, atol=args.atol, rtol=args.rtol)
                result[key] = True
            except Exception as exc:
                result[key] = False
                result[f"{key}_error"] = repr(exc)
                result[f"{key}_traceback"] = traceback.format_exc()

        for key, fn in (
            ("cublas_naive", naive),
            ("vllm_fused", vllm),
            ("flashinfer_staged", staged),
            ("flashinfer_nocopy", nocopy),
        ):
            try:
                result[key] = {
                    "ok": True,
                    **_time_cuda_events_ms(fn, args.warmup, args.iterations, group),
                }
            except Exception as exc:
                result[key] = {
                    "ok": False,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
    finally:
        ws_staged.destroy()
        ws_nocopy.destroy()
    return result


def _write_outputs(
    args, rows: list[dict[str, Any]], rank_details: list[dict[str, Any]]
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"flashinfer_blackwell_ab_ws{dist.get_world_size()}_{ts}"
    json_path = RESULTS_DIR / f"{base}.json"
    csv_path = RESULTS_DIR / f"{base}.csv"
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "host": os.uname().nodename,
        "world_size": dist.get_world_size(),
        "torch": torch.__version__,
        "dtype": args.dtype,
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "timer": (
            "CUDA events with a process-group barrier and device synchronize "
            "before every call, plus device synchronize after every call."
        ),
        "summary_statistic": (
            "Per-iteration maximum latency across ranks (distributed critical "
            "path), then median/p75/max across iterations."
        ),
        "note": (
            "Same-allocation same-tensor A/B: staged B, no-copy B, vLLM fused, naive."
        ),
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
        "naive_ms",
        "vllm_ms",
        "staged_ms",
        "nocopy_ms",
        "naive_median_ms",
        "naive_p75_ms",
        "naive_max_ms",
        "vllm_median_ms",
        "vllm_p75_ms",
        "vllm_max_ms",
        "staged_median_ms",
        "staged_p75_ms",
        "staged_max_ms",
        "nocopy_median_ms",
        "nocopy_p75_ms",
        "nocopy_max_ms",
        "vllm_correct",
        "staged_correct",
        "nocopy_correct",
        "staged_vs_vllm",
        "nocopy_vs_vllm",
        "staged_vs_naive",
        "nocopy_vs_naive",
        "nocopy_vs_staged",
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
    parser.add_argument("--atol", type=float, default=1e-1)
    parser.add_argument("--rtol", type=float, default=1e-2)
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
            f"flashinfer_blackwell_ab: ws={world_size} N={args.n} "
            f"M={args.m_values} K_total={args.k_total_values} dtype={args.dtype} "
            f"warmup={args.warmup} iterations={args.iterations}"
        )

        rows = []
        rank_details = []
        for k_total in args.k_total_values:
            for m in args.m_values:
                _rank0_print(f"===== M={m} K_total={k_total} =====")
                local_result = _run_one_shape(args, m, k_total, group)
                gathered = _gather(local_result, group)
                if dist.get_rank(group) != 0:
                    continue

                assert gathered is not None
                rank_details.extend(gathered)
                naive_ms = _max_rank_mean(gathered, "cublas_naive")
                vllm_ms = _max_rank_mean(gathered, "vllm_fused")
                staged_ms = _max_rank_mean(gathered, "flashinfer_staged")
                nocopy_ms = _max_rank_mean(gathered, "flashinfer_nocopy")
                critical_stats = {
                    "naive": _critical_path_stats(gathered, "cublas_naive"),
                    "vllm": _critical_path_stats(gathered, "vllm_fused"),
                    "staged": _critical_path_stats(gathered, "flashinfer_staged"),
                    "nocopy": _critical_path_stats(gathered, "flashinfer_nocopy"),
                }
                row = {
                    "M": m,
                    "N": args.n,
                    "K_total": k_total,
                    "K_local": k_total // world_size,
                    "world_size": world_size,
                    "dtype": args.dtype,
                    "naive_ms": naive_ms,
                    "vllm_ms": vllm_ms,
                    "staged_ms": staged_ms,
                    "nocopy_ms": nocopy_ms,
                    "vllm_correct": _all_correct(gathered, "vllm_correct"),
                    "staged_correct": _all_correct(gathered, "staged_correct"),
                    "nocopy_correct": _all_correct(gathered, "nocopy_correct"),
                    "vllm_ok": _all_ok(gathered, "vllm_fused"),
                    "staged_ok": _all_ok(gathered, "flashinfer_staged"),
                    "nocopy_ok": _all_ok(gathered, "flashinfer_nocopy"),
                    "staged_vs_vllm": vllm_ms / staged_ms
                    if vllm_ms and staged_ms
                    else "",
                    "nocopy_vs_vllm": vllm_ms / nocopy_ms
                    if vllm_ms and nocopy_ms
                    else "",
                    "staged_vs_naive": naive_ms / staged_ms
                    if naive_ms and staged_ms
                    else "",
                    "nocopy_vs_naive": naive_ms / nocopy_ms
                    if naive_ms and nocopy_ms
                    else "",
                    "nocopy_vs_staged": staged_ms / nocopy_ms
                    if staged_ms and nocopy_ms
                    else "",
                    "notes": _joined_errors(gathered),
                }
                for backend, stats in critical_stats.items():
                    for statistic in ("median_ms", "p75_ms", "max_ms"):
                        row[f"{backend}_{statistic}"] = (
                            stats[statistic] if stats is not None else ""
                        )
                rows.append(row)
                _rank0_print(str(row))

        if dist.get_rank(group) == 0:
            _write_outputs(args, rows, rank_details)
    finally:
        torchrun_nvshmem_finalize()


if __name__ == "__main__":
    main()
