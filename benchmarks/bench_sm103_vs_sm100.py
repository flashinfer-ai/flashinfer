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

import cutlass
import numpy as np
import torch

from flashinfer import SfLayout, nvfp4_quantize
from flashinfer.cute_dsl.utils import torch_to_cutlass_dtype
from flashinfer.gemm.gemm_base import _get_sm100_block_scaled_tactics
from flashinfer.gemm.gemm_mm_fp4_cute_dsl import (
    _compile_block_scaled_gemm,
    _mm_fp4_cache_key,
    _prepare_alpha_for_launch,
)
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100 import (
    Sm100BlockScaledPersistentDenseGemmKernel,
)
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm103 import (
    Sm103BlockScaledPersistentDenseGemmKernel,
)
from flashinfer.gemm.kernels.utils import _SM100_CLUSTER_SHAPE_MN_CANDIDATES
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import get_device_index


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


def get_exhaustive_tactics(m, n, k, out_dtype, device):
    """Enumerate every SM100 and SM103 tactic accepted by can_implement()."""
    sf_vec_size = 16
    ab_dtype = cutlass.Float4E2M1FN
    sf_dtype = cutlass.Float8E4M3FN
    c_dtype = torch_to_cutlass_dtype(out_dtype)

    sm100_tactics = [
        (*tactic, "sm100", None)
        for tactic in _get_sm100_block_scaled_tactics(
            m,
            n,
            k,
            ab_dtype,
            sf_dtype,
            sf_vec_size,
            c_dtype,
            device,
        )
    ]

    sm103_tactics = []
    batch_size = 1
    m_aligned = m % 8 == 0
    n_aligned = n % 8 == 0
    mma_tiler_candidates = [
        (128, 128),
        (256, 128),
        (128, 256),
        (256, 256),
    ]

    for mma_tiler_mn in mma_tiler_candidates:
        for cluster_shape_mn in _SM100_CLUSTER_SHAPE_MN_CANDIDATES:
            for swap_ab in (False, True):
                if not swap_ab and not n_aligned:
                    continue
                if swap_ab and not m_aligned:
                    continue

                if swap_ab:
                    c_major = "m"
                    kernel_m, kernel_n = n, m
                else:
                    c_major = "n"
                    kernel_m, kernel_n = m, n

                for use_tma_store in (True, False):
                    if not Sm103BlockScaledPersistentDenseGemmKernel.can_implement(
                        ab_dtype,
                        sf_dtype,
                        sf_vec_size,
                        c_dtype,
                        mma_tiler_mn,
                        cluster_shape_mn,
                        kernel_m,
                        kernel_n,
                        k,
                        batch_size,
                        "k",
                        "k",
                        c_major,
                        use_tma_store,
                    ):
                        continue

                    sm103_tactics.append(
                        (
                            mma_tiler_mn,
                            cluster_shape_mn,
                            swap_ab,
                            False,
                            "sm103",
                            use_tma_store,
                        )
                    )

    return sm100_tactics, sm103_tactics


_KERNEL_CACHE = {}


def run_tactic(inputs, tactic, out_dtype, enable_pdl=True):
    """Compile and launch one explicitly selected SM100 or SM103 tactic."""
    (a, b, a_descale, b_descale, alpha_tensor, _, out, _, _, _) = inputs
    m = a.shape[0]
    n = b.shape[1]
    real_k = a.shape[1] * 2
    sf_vec_size = 16
    sf_dtype = cutlass.Float8E4M3FN
    batch_size = 1

    (
        mma_tiler_mn,
        cluster_shape_mn,
        swap_ab,
        use_prefetch,
        kernel_type,
        use_tma_store,
    ) = tactic

    if swap_ab:
        kernel_m, kernel_n = n, m
        kernel_a, kernel_b = b.T, a
        kernel_a_sf, kernel_b_sf = b_descale.T, a_descale
    else:
        kernel_m, kernel_n = m, n
        kernel_a, kernel_b = a, b.T
        kernel_a_sf, kernel_b_sf = a_descale, b_descale.T

    sf_m = (kernel_m + 127) // 128
    sf_n = (kernel_n + 127) // 128
    sf_k = (real_k // sf_vec_size + 3) // 4
    cache_key = _mm_fp4_cache_key(sf_vec_size, tactic, enable_pdl, out_dtype)

    if kernel_type == "sm103":
        make_kernel = lambda: Sm103BlockScaledPersistentDenseGemmKernel(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_tma_store,
            enable_pdl,
        )
    else:
        make_kernel = lambda: Sm100BlockScaledPersistentDenseGemmKernel(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )

    compiled_gemm, _ = _compile_block_scaled_gemm(
        _KERNEL_CACHE,
        cache_key,
        make_kernel,
        ab_cutlass_dtype=cutlass.Uint8,
        sf_dtype=sf_dtype,
        c_cutlass_dtype=torch_to_cutlass_dtype(out_dtype),
        ab_assumed_align=32,
        cluster_shape_mn=cluster_shape_mn,
        swap_ab=swap_ab,
        sf_m=sf_m,
        sf_n=sf_n,
        sf_k=sf_k,
        batch_size=batch_size,
        cache_module_name="mm_fp4",
        device_index=get_device_index(a.device),
    )

    alpha_for_launch = _prepare_alpha_for_launch(alpha_tensor, a.device)
    launch_out = out.as_strided(out.shape, (1, out.shape[0])) if swap_ab else out
    compiled_gemm(
        kernel_a,
        kernel_b,
        launch_out,
        sf_m,
        sf_n,
        sf_k,
        kernel_a_sf.data_ptr(),
        kernel_b_sf.data_ptr(),
        alpha_for_launch,
    )
    return out


def benchmark_one(inputs, tactic, out_dtype, iters):
    """Return (median CUPTI CUDA-Graph cold-L2 milliseconds, error)."""

    def run_fn():
        run_tactic(inputs, tactic, out_dtype)

    try:
        run_fn()
        torch.cuda.synchronize()
    except Exception as error:
        return None, str(error)

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
    results = []

    for m, n, k in problem_sizes:
        print(f"\n--- M={m:>5}, N={n:>5}, K={k:>5} ---")

        a_fp4, b_fp4, a_sf, b_sf, alpha = prepare_fp4_inputs(m, n, k)
        out = torch.empty(m, n, device=device, dtype=out_dtype)
        workspace = torch.empty(32 * 1024 * 1024, device=device, dtype=torch.uint8)
        inputs = [a_fp4, b_fp4, a_sf, b_sf, alpha, out_dtype, out, 16, True, workspace]

        sm100_tactics, sm103_tactics = get_exhaustive_tactics(
            m, n, k, out_dtype, device
        )
        print(f"  Tactics: {len(sm100_tactics)} SM100, {len(sm103_tactics)} SM103")

        best = {"sm100": (float("inf"), None), "sm103": (float("inf"), None)}

        for tag, tactics in [
            ("sm100", sm100_tactics),
            ("sm103", sm103_tactics),
        ]:
            for tactic in tactics:
                ms, error = benchmark_one(inputs, tactic, out_dtype, args.iters)
                if error is not None:
                    print(f"  Skipping {format_tactic(tactic)}: {error}")
                elif ms < best[tag][0]:
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
                    f"  Best {tag.upper()}: {ms:.4f} ms  "
                    f"({tf:.1f} TFLOPS)  {format_tactic(tac)}"
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

    print(f"\n{'=' * 130}")
    print(f"Summary: SM103 vs SM100 FP4 GEMM on {gpu_name}")
    print(f"{'=' * 130}")
    fmt = "{:>6} {:>6} {:>6} | {:>10} {:>7} | {:>10} {:>7} | {:>8}"
    print(
        fmt.format("M", "N", "K", "SM100 ms", "TFLOPS", "SM103 ms", "TFLOPS", "Speedup")
    )
    print("-" * 130)
    for row in results:
        print(
            fmt.format(
                row["m"],
                row["n"],
                row["k"],
                row["sm100_ms"],
                row["sm100_tflops"],
                row["sm103_ms"],
                row["sm103_tflops"],
                row["speedup"],
            )
        )

    csv_path = args.csv or f"bench_sm103_vs_sm100_sm{sm_version}.csv"
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
