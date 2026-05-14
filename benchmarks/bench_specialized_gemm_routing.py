#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.env import (
    FLASHINFER_SPECIALIZED_KERNEL_DISABLE,
    reset_specialized_kernel_env_cache,
)
from flashinfer.gemm.specialized_kernels import (
    is_bmm_fp8_sm121_specialized_problem,
    is_mm_fp4_sm121_specialized_problem,
)
from flashinfer.testing.utils import bench_gpu_time


REPO_ROOT = Path(__file__).resolve().parents[1]
SPECIALIZED_KERNEL_DIR = REPO_ROOT / "flashinfer" / "gemm" / "specialized_kernels"
DEFAULT_WORKLOADS = {
    "mm_fp4": SPECIALIZED_KERNEL_DIR / "mm_fp4_sm121" / "workloads.json",
    "bmm_fp8": SPECIALIZED_KERNEL_DIR / "bmm_fp8_sm121" / "workloads.json",
}
THRESHOLDS = {
    "mm_fp4": 0.97,
    "bmm_fp8": 0.99,
}


@contextmanager
def specialized_routing_disabled(disabled: bool):
    old_values = {
        FLASHINFER_SPECIALIZED_KERNEL_DISABLE: os.environ.get(
            FLASHINFER_SPECIALIZED_KERNEL_DISABLE
        ),
    }
    value = "True" if disabled else "False"
    os.environ[FLASHINFER_SPECIALIZED_KERNEL_DISABLE] = value
    reset_specialized_kernel_env_cache()
    try:
        yield
    finally:
        for name, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value
        reset_specialized_kernel_env_cache()


@contextmanager
def autotune_mode(enabled: bool, cache: Path | None):
    if enabled or cache is not None:
        cache_path = str(cache) if cache is not None else None
        with flashinfer.autotune(enabled, cache=cache_path):
            yield
    else:
        yield


def load_workloads(
    functions: list[str], m_filter: set[int] | None, limit: int | None
) -> list[tuple[str, dict[str, Any]]]:
    cases = []
    for function in functions:
        workloads = json.loads(DEFAULT_WORKLOADS[function].read_text())
        for item in workloads:
            if m_filter is not None and int(item["m"]) not in m_filter:
                continue
            cases.append((function, item))
            if limit is not None and len(cases) >= limit:
                return cases
    return cases


def to_float8(x: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


def cosine(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(lhs.reshape(-1).float(), rhs.reshape(-1).float(), dim=0)
    )


def make_mm_fp4_case(item: dict[str, Any], device: torch.device):
    m, k, n = int(item["m"]), int(item["k"]), int(item["n"])
    a_ref = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    b_ref = torch.randn((n, k), dtype=torch.bfloat16, device=device)
    reference = torch.mm(a_ref, b_ref.T)

    a_global_sf = (448 * 6) / a_ref.float().abs().nan_to_num().max()
    b_global_sf = (448 * 6) / b_ref.float().abs().nan_to_num().max()
    a_fp4, a_sf = flashinfer.nvfp4_quantize(
        a_ref,
        a_global_sf,
        sfLayout=flashinfer.SfLayout.layout_128x4,
        do_shuffle=False,
    )
    b_fp4, b_sf = flashinfer.nvfp4_quantize(
        b_ref,
        b_global_sf,
        sfLayout=flashinfer.SfLayout.layout_128x4,
        do_shuffle=False,
    )
    alpha = 1.0 / (a_global_sf * b_global_sf)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=device)

    def run(out_tensor: torch.Tensor):
        return flashinfer.gemm.mm_fp4(
            a_fp4,
            b_fp4.T,
            a_sf,
            b_sf.T,
            alpha,
            torch.bfloat16,
            out_tensor,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="b12x",
            use_nvfp4=True,
        )

    def is_routed(out_tensor: torch.Tensor):
        return is_mm_fp4_sm121_specialized_problem(
            a_fp4,
            b_fp4.T,
            a_sf,
            b_sf.T,
            alpha,
            torch.bfloat16,
            out_tensor,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="b12x",
            use_nvfp4=True,
        )

    return run, is_routed, out, reference


def make_bmm_fp8_case(item: dict[str, Any], device: torch.device):
    b, m, k, n = int(item["b"]), int(item["m"]), int(item["k"]), int(item["n"])
    inp = torch.randn((b, m, k), dtype=torch.bfloat16, device=device)
    inp_fp8, inp_inv_s = to_float8(inp)
    mat2 = torch.randn((b, n, k), dtype=torch.bfloat16, device=device).transpose(-2, -1)
    mat2_fp8, mat2_inv_s = to_float8(mat2)
    reference = torch.bmm(inp, mat2)
    out = torch.empty((b, m, n), dtype=torch.bfloat16, device=device)

    def run(out_tensor: torch.Tensor):
        return flashinfer.gemm.bmm_fp8(
            inp_fp8,
            mat2_fp8,
            inp_inv_s,
            mat2_inv_s,
            torch.bfloat16,
            out_tensor,
            backend="cublas",
        )

    def is_routed(out_tensor: torch.Tensor):
        return is_bmm_fp8_sm121_specialized_problem(
            inp_fp8,
            mat2_fp8,
            inp_inv_s,
            mat2_inv_s,
            torch.bfloat16,
            out_tensor,
            backend="cublas",
        )

    return run, is_routed, out, reference


CASE_BUILDERS = {
    "mm_fp4": make_mm_fp4_case,
    "bmm_fp8": make_bmm_fp8_case,
}


def benchmark_one(function: str, item: dict[str, Any], args, device: torch.device):
    run, is_routed, out, reference = CASE_BUILDERS[function](item, device)
    disabled_out = torch.empty_like(out)
    enabled_out = torch.empty_like(out)

    def run_and_time(disabled: bool, target: torch.Tensor):
        with specialized_routing_disabled(disabled):
            if args.autotune:
                with autotune_mode(True, args.autotune_cache):
                    run(target)
                    torch.cuda.synchronize(device)

            with autotune_mode(False, args.autotune_cache):
                result = run(target)
                torch.cuda.synchronize(device)
                snapshot = result.detach().clone()
                times = bench_gpu_time(
                    fn=lambda: run(target),
                    dry_run_iters=args.dry_run_iters,
                    repeat_iters=args.num_iters,
                    enable_cupti=not args.no_cupti,
                    use_cuda_graph=not args.no_cuda_graph,
                )
        return snapshot, float(statistics.median(times))

    disabled_snapshot, disabled_ms = run_and_time(True, disabled_out)
    with specialized_routing_disabled(False):
        route_active = is_routed(enabled_out)
    enabled_snapshot, enabled_ms = run_and_time(False, enabled_out)
    threshold = THRESHOLDS[function]
    disabled_cosine = cosine(reference, disabled_snapshot)
    enabled_cosine = cosine(reference, enabled_snapshot)
    routed_cosine = cosine(disabled_snapshot, enabled_snapshot)
    speedup = disabled_ms / enabled_ms

    return {
        "function": function,
        "impl": item["impl"],
        "autotune": args.autotune,
        "autotune_cache": str(args.autotune_cache or ""),
        "b": item.get("b", ""),
        "m": item["m"],
        "n": item["n"],
        "k": item["k"],
        "block_size": item.get("block_size", ""),
        "threshold": threshold,
        "disabled_ms": disabled_ms,
        "enabled_ms": enabled_ms,
        "route_active": route_active,
        "speedup": speedup,
        "improvement_pct": (speedup - 1.0) * 100.0,
        "disabled_cosine": disabled_cosine,
        "enabled_cosine": enabled_cosine,
        "disabled_correct": disabled_cosine > threshold,
        "enabled_correct": enabled_cosine > threshold,
        "disabled_enabled_cosine": routed_cosine,
    }


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--function",
        choices=tuple(DEFAULT_WORKLOADS),
        action="append",
        help="Function to benchmark. Defaults to all specialized workloads.",
    )
    parser.add_argument(
        "--m", type=int, action="append", help="Only run the given M value."
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-iters", type=int, default=30)
    parser.add_argument("--dry-run-iters", type=int, default=5)
    parser.add_argument(
        "--autotune",
        action="store_true",
        help=(
            "Profile FlashInfer autotune before each timing pass. Timed CUDA "
            "graph capture uses the tuned cache without running profiling inside "
            "capture."
        ),
    )
    parser.add_argument(
        "--autotune-cache",
        type=Path,
        default=None,
        help=(
            "Optional autotuner config cache. With --autotune, loads then saves "
            "configs; without --autotune, loads configs without profiling."
        ),
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Disable CUDA graph timing. CUDA graph timing is enabled by default.",
    )
    parser.add_argument(
        "--no-cupti",
        action="store_true",
        help="Disable CUPTI timing. CUPTI timing is enabled by default.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("specialized_gemm_routing_benchmark.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    functions = args.function or list(DEFAULT_WORKLOADS)
    rows = []
    m_filter = set(args.m) if args.m is not None else None
    for function, item in load_workloads(functions, m_filter, args.limit):
        result = benchmark_one(function, item, args, device)
        rows.append(result)
        print(
            f"{function:14s} impl={result['impl']:8s} "
            f"b={result['b']} m={result['m']} n={result['n']} k={result['k']} "
            f"disabled={result['disabled_ms']:.6f} ms "
            f"enabled={result['enabled_ms']:.6f} ms "
            f"speedup={result['speedup']:.3f}x "
            f"route_active={result['route_active']} "
            f"correct={result['enabled_correct']}"
        )

    if not rows:
        raise RuntimeError("No workloads selected.")

    fieldnames = list(rows[0].keys())
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nGeomean speedup, correct enabled cases only:")
    for function in functions:
        for impl in sorted({r["impl"] for r in rows if r["function"] == function}):
            speeds = [
                r["speedup"]
                for r in rows
                if r["function"] == function
                and r["impl"] == impl
                and r["enabled_correct"]
                and r["route_active"]
            ]
            if speeds:
                print(
                    f"{function:14s} {impl:8s} {len(speeds):4d} cases {geomean(speeds):.4f}x"
                )


if __name__ == "__main__":
    main()
