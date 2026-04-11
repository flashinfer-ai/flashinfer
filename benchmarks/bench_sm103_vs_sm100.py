"""Quick SM103 vs SM100 tactic benchmark for FP4 GEMM on Blackwell.

Directly instantiates the CuTe DSL FP4 GEMM kernels and compares
SM100 tactics against SM103-specific 3xFP4 tactics across representative
LLM problem sizes.

Usage:
    python benchmarks/bench_sm103_vs_sm100.py [--sizes small|medium|large|all]
                                               [--out-dtype bfloat16|float16]
                                               [--iters N]

Example:
    python benchmarks/bench_sm103_vs_sm100.py --sizes small --iters 10
"""

import argparse
import csv
from typing import List, Tuple

import numpy as np
import torch

from flashinfer import SfLayout, nvfp4_quantize
from flashinfer.testing.utils import bench_gpu_time


# -- Problem sizes by category ------------------------------------------------
SIZES_SMALL = [
    # Decode-like (small M)
    (1, 4096, 7168),
    (4, 4096, 7168),
    (8, 4096, 7168),
    (16, 4096, 7168),
    (32, 4096, 7168),
    (64, 4096, 7168),
]
SIZES_MEDIUM = [
    # Small-batch prefill
    (128, 4096, 7168),
    (128, 7168, 2048),
    (256, 4096, 7168),
    (256, 14336, 4096),
    (512, 14336, 4096),
]
SIZES_LARGE = [
    # Large prefill / square
    (1024, 4096, 7168),
    (2048, 4096, 7168),
    (4096, 4096, 7168),
    (4096, 4096, 4096),
]


def get_problem_sizes(category: str) -> List[Tuple[int, int, int]]:
    if category == "small":
        return SIZES_SMALL
    if category == "medium":
        return SIZES_MEDIUM
    if category == "large":
        return SIZES_LARGE
    return SIZES_SMALL + SIZES_MEDIUM + SIZES_LARGE


# -- Input preparation --------------------------------------------------------
def prepare_fp4_inputs(m, n, k, device="cuda"):
    """Quantize random tensors to NVF4 format."""
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    b = torch.randn(n, k, device=device, dtype=torch.bfloat16)

    a_gsf = (448 * 6) / a.float().abs().nan_to_num().max()
    b_gsf = (448 * 6) / b.float().abs().nan_to_num().max()

    a_fp4, a_sf = nvfp4_quantize(
        a, a_gsf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    b_fp4, b_sf = nvfp4_quantize(
        b, b_gsf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    alpha = torch.tensor(
        [1.0 / (a_gsf.item() * b_gsf.item())],
        dtype=torch.float32,
        device=device,
    )
    # mm_fp4 API convention: b is (k_packed, n), b_descale is (k_sf, n_sf)
    return a_fp4, b_fp4.T, a_sf, b_sf.T, alpha


# -- Tactic helpers -----------------------------------------------------------
def format_tactic(tactic):
    mma, cluster, swap, prefetch, ktype, tma_store = tactic
    parts = [
        f"tile={mma[0]}x{mma[1]}",
        f"cl={cluster[0]}x{cluster[1]}",
        f"swap={'Y' if swap else 'N'}",
        f"kern={ktype}",
    ]
    if tma_store is not None:
        parts.append(f"tma_st={'Y' if tma_store else 'N'}")
    return " ".join(parts)


def benchmark_one(runner, inputs, tactic, iters):
    """Returns (median_ms, error_string_or_None)."""

    def run_fn():
        runner.forward(inputs, tactic=tactic)

    # Warmup / JIT compile
    try:
        run_fn()
        torch.cuda.synchronize()
    except Exception as e:
        return None, str(e)

    try:
        times = bench_gpu_time(
            run_fn,
            dry_run_iters=max(3, iters // 4),
            repeat_iters=iters,
            enable_cupti=True,
            use_cuda_graph=True,
            cold_l2_cache=True,
            sleep_after_run=True,
        )
        return float(np.median(times)), None
    except Exception:
        try:
            times = bench_gpu_time(
                run_fn,
                dry_run_iters=max(3, iters // 4),
                repeat_iters=iters,
                enable_cupti=False,
                use_cuda_graph=False,
                cold_l2_cache=True,
                sleep_after_run=True,
            )
            return float(np.median(times)), None
        except Exception as e2:
            return None, str(e2)


# -- Main ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SM103 vs SM100 FP4 GEMM benchmark")
    parser.add_argument(
        "--sizes",
        choices=["small", "medium", "large", "all"],
        default="all",
        help="Problem-size category (default: all)",
    )
    parser.add_argument(
        "--out-dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
        help="Output dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--iters", type=int, default=20, help="Benchmark iterations (default: 20)"
    )
    parser.add_argument(
        "--csv", type=str, default=None, help="Output CSV path (optional)"
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    major, minor = torch.cuda.get_device_capability(device)
    sm_version = major * 10 + minor
    gpu_name = torch.cuda.get_device_name(device)

    print(f"GPU: {gpu_name} (SM{sm_version})")
    if sm_version not in (100, 103):
        print(f"WARNING: designed for SM100/SM103, got SM{sm_version}")

    out_dtype = torch.bfloat16 if args.out_dtype == "bfloat16" else torch.float16
    problem_sizes = get_problem_sizes(args.sizes)

    # Create runner (exposes both SM100 and SM103 tactics on SM103 hardware)
    from flashinfer.autotuner import OptimizationProfile
    from flashinfer.gemm.gemm_base import _cute_dsl_gemm_fp4_runner

    runner = _cute_dsl_gemm_fp4_runner(major, minor, True, out_dtype, use_nvfp4=True)

    results = []

    for m, n, k in problem_sizes:
        print(f"\n--- M={m:>5}, N={n:>5}, K={k:>5} ---")

        a_fp4, b_fp4, a_sf, b_sf, alpha = prepare_fp4_inputs(m, n, k)
        out = torch.empty(m, n, device=device, dtype=out_dtype)
        workspace = torch.empty(32 * 1024 * 1024, device=device, dtype=torch.uint8)
        inputs = [a_fp4, b_fp4, a_sf, b_sf, alpha, out_dtype, out, 16, True, workspace]

        all_tactics = runner.get_valid_tactics(inputs, OptimizationProfile([], []))
        sm100_tactics = [t for t in all_tactics if t[4] == "sm100"]
        sm103_tactics = [t for t in all_tactics if t[4] == "sm103"]
        print(f"  Tactics: {len(sm100_tactics)} SM100, {len(sm103_tactics)} SM103")

        best = {"sm100": (float("inf"), None), "sm103": (float("inf"), None)}

        for tag, tactics in [("sm100", sm100_tactics), ("sm103", sm103_tactics)]:
            for tactic in tactics:
                ms, err = benchmark_one(runner, inputs, tactic, args.iters)
                if ms is not None and ms < best[tag][0]:
                    best[tag] = (ms, tactic)

        tflops_factor = 2 * m * n * k / 1e12
        row = {"m": m, "n": n, "k": k}

        for tag in ("sm100", "sm103"):
            ms, tac = best[tag]
            if tac is not None:
                tf = tflops_factor / (ms / 1000)
                row[f"{tag}_ms"] = f"{ms:.4f}"
                row[f"{tag}_tflops"] = f"{tf:.1f}"
                row[f"{tag}_tactic"] = format_tactic(tac)
                print(
                    f"  Best {tag.upper()}: {ms:.4f} ms  ({tf:.1f} TFLOPS)  {format_tactic(tac)}"
                )
            else:
                row[f"{tag}_ms"] = "N/A"
                row[f"{tag}_tflops"] = "N/A"
                row[f"{tag}_tactic"] = "N/A"
                print(f"  Best {tag.upper()}: no valid tactic")

        if best["sm100"][1] and best["sm103"][1]:
            speedup = best["sm100"][0] / best["sm103"][0]
            row["speedup"] = f"{speedup:.2f}x"
            print(f"  SM103/SM100 speedup: {speedup:.2f}x")
        else:
            row["speedup"] = "N/A"

        results.append(row)

    # Final summary table
    print(f"\n{'=' * 130}")
    print(f"Summary: SM103 vs SM100 FP4 GEMM on {gpu_name}")
    print(f"{'=' * 130}")
    fmt = "{:>6} {:>6} {:>6} | {:>10} {:>7} | {:>10} {:>7} | {:>8}"
    print(
        fmt.format("M", "N", "K", "SM100 ms", "TFLOPS", "SM103 ms", "TFLOPS", "Speedup")
    )
    print("-" * 130)
    for r in results:
        print(
            fmt.format(
                r["m"],
                r["n"],
                r["k"],
                r["sm100_ms"],
                r["sm100_tflops"],
                r["sm103_ms"],
                r["sm103_tflops"],
                r["speedup"],
            )
        )

    # Optional CSV output
    csv_path = args.csv or f"bench_sm103_vs_sm100_sm{sm_version}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
