"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Benchmark: BF16 Dense GEMM Backend Comparison (mm_bf16)

Compares FlashInfer's ``mm_bf16`` cuTile backend (PR #4020) against the other
backends the same op exposes -- CUTLASS (the SOTA baseline), cuBLASLt, cuDNN,
TGV, TinyGEMM -- plus a torch.matmul reference, across an (m x n x k) sweep.
C = A @ B, A: (m, k) row-major bf16, B: (k, n) col-major bf16.

All backends are exposed through the same ``mm_bf16`` entry point; some require
optional build components (e.g. nvidia-cutlass-dsl, JIT sources) and are
reported N/A when those are unavailable in the image.
Backends that do not support a given shape return NaN for that cell.
Emits a per-backend latency table, a speedup summary vs CUTLASS, and a heatmap.

Usage:
    python bench_mm_bf16_backend_comparison.py
    python bench_mm_bf16_backend_comparison.py --providers cutile,cutlass,cublaslt
    python bench_mm_bf16_backend_comparison.py --csv out.csv

Requirements:
    - SM >= 90 for the cutile backend; matplotlib for the heatmap
"""

import argparse
import csv as _csv
import numpy as np
import torch
from typing import Dict, List, Tuple

import flashinfer
from flashinfer.testing.utils import bench_gpu_time

ALL_PROVIDERS = ["cutile", "cutlass", "cublaslt", "cudnn", "tgv", "tinygemm", "torch"]


def get_cc() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _make_inputs(m: int, n: int, k: int):
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).transpose(-2, -1)  # (k, n) col-major
    return a, b


def _run_backend(a, b, provider: str):
    if provider == "torch":
        return torch.matmul(a, b)
    return flashinfer.mm_bf16(a, b, backend=provider)


def provider_available(provider: str, m: int = 256, n: int = 256, k: int = 256) -> bool:
    """A provider is 'available' if it runs on a small reference shape."""
    if provider == "torch":
        return True
    try:
        a, b = _make_inputs(m, n, k)
        _run_backend(a, b, provider)
        torch.cuda.synchronize()
        return True
    except Exception as e:
        print(f"  {provider:>9}: UNAVAILABLE ({type(e).__name__}: {str(e).splitlines()[0][:120]})")
        return False


def verify_correctness(m: int, n: int, k: int, provider: str) -> Tuple[bool, float]:
    """Cosine similarity of provider output vs torch.matmul reference."""
    torch.manual_seed(42)
    a, b = _make_inputs(m, n, k)
    try:
        out = _run_backend(a, b, provider).to(torch.float32).reshape(1, -1)
        ref = torch.matmul(a, b).to(torch.float32).reshape(1, -1)
        cos = torch.nn.functional.cosine_similarity(out, ref).item()
        return cos > 0.99, cos
    except Exception:
        return False, 0.0


def bench_one(m: int, n: int, k: int, provider: str) -> float:
    """Median latency (ms); NaN if the backend does not support the shape."""
    torch.manual_seed(0)
    a, b = _make_inputs(m, n, k)
    try:
        _run_backend(a, b, provider)  # warmup / autotune
        times = bench_gpu_time(
            fn=lambda: _run_backend(a, b, provider),
            enable_cupti=True,
            dry_run_iters=5,
            repeat_iters=30,
            cold_l2_cache=True,
            use_cuda_graph=False,
        )
        return float(np.median(times))
    except Exception:
        return float("nan")


def run_benchmark_sweep(
    shapes: List[Tuple[int, int, int]], providers: List[str]
) -> Dict[str, Dict[Tuple[int, int, int], float]]:
    results: Dict[str, Dict[Tuple[int, int, int], float]] = {p: {} for p in providers}
    print(f"\nBenchmarking mm_bf16 (C = A@B, bf16)  providers={providers}")
    print("=" * (26 + 11 * len(providers)))
    header = f"{'m':>6} {'n':>6} {'k':>6} |"
    for p in providers:
        header += f" {p:>9}"
    print(header)
    print("-" * (26 + 11 * len(providers)))
    for (m, n, k) in shapes:
        row = f"{m:>6} {n:>6} {k:>6} |"
        for p in providers:
            ms = bench_one(m, n, k, p)
            results[p][(m, n, k)] = ms
            row += f" {ms:>9.4f}" if ms == ms else f" {'--':>9}"
        print(row)
    return results


def print_summary_table(
    shapes: List[Tuple[int, int, int]],
    results: Dict[str, Dict[Tuple[int, int, int], float]],
    providers: List[str],
    baseline: str,
):
    if baseline not in results:
        return
    print(f"\n{'=' * 78}")
    print(f"Speedup vs {baseline} (baseline_ms / provider_ms;  >1 = provider faster)")
    print(f"{'=' * 78}")
    header = f"{'m,n,k':>18} |"
    others = [p for p in providers if p != baseline]
    for p in others:
        header += f" {p:>9}"
    print(header)
    print("-" * (20 + 10 * len(others)))
    geo = {p: [] for p in others}
    for (m, n, k) in shapes:
        row = f"{f'{m},{n},{k}':>18} |"
        bt = results[baseline].get((m, n, k), float("nan"))
        for p in others:
            pt = results[p].get((m, n, k), float("nan"))
            if pt == pt and bt == bt and pt > 0:
                r = bt / pt
                geo[p].append(r)
                row += f" {r:>9.2f}"
            else:
                row += f" {'N/A':>9}"
        print(row)
    print("\nGeomean speedup vs " + baseline + ":")
    for p in others:
        if geo[p]:
            print(
                f"  {p:>9}: {np.exp(np.mean(np.log(geo[p]))):.2f}x  "
                f"(min {min(geo[p]):.2f}, max {max(geo[p]):.2f}, "
                f"{sum(1 for r in geo[p] if r > 1)}/{len(geo[p])} shapes faster)"
            )


def write_csv(path: str, results: Dict[str, Dict[Tuple[int, int, int], float]]):
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["provider", "m", "n", "k", "median_ms", "tflops"])
        for provider, d in results.items():
            for (m, n, k), ms in sorted(d.items()):
                tf = (2.0 * m * n * k / (ms * 1e-3) / 1e12) if ms == ms and ms > 0 else float("nan")
                w.writerow([provider, m, n, k, f"{ms:.6f}", f"{tf:.2f}"])
    print(f"\nWrote {path}")


def create_heatmap(
    shapes, results, provider, baseline, output_file
):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not installed, skipping heatmap")
        return
    if provider not in results or baseline not in results:
        return
    labels = [f"{m}x{n}x{k}" for (m, n, k) in shapes]
    vals = []
    for s in shapes:
        bt = results[baseline].get(s, float("nan"))
        pt = results[provider].get(s, float("nan"))
        vals.append(bt / pt if (pt == pt and bt == bt and pt > 0) else float("nan"))
    if all(np.isnan(vals)):
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(shapes)), 2.4))
    arr = np.array(vals).reshape(1, -1)
    norm = mcolors.TwoSlopeNorm(vmin=min(0.5, np.nanmin(arr)), vcenter=1.0, vmax=max(1.5, np.nanmax(arr)))
    im = ax.imshow(arr, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_yticks([0]); ax.set_yticklabels([f"{provider}/{baseline}"])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    for j, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(j, 0, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v < 0.7 or v > 1.5 else "black")
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"mm_bf16: {provider} speedup vs {baseline} (>1 = {provider} faster)")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved heatmap to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark mm_bf16 backends")
    parser.add_argument("--providers", type=str, default=",".join(ALL_PROVIDERS))
    parser.add_argument("--baseline", type=str, default="cutlass")
    parser.add_argument("--output-prefix", type=str, default="mm_bf16_backend_comparison")
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    cc = get_cc()
    print(f"GPU Compute Capability: SM{cc}")
    requested = [p.strip() for p in args.providers.split(",") if p.strip() in ALL_PROVIDERS]
    available = [p for p in requested if provider_available(p)]
    if not available:
        print("No available providers; nothing to benchmark.")
        return
    print(f"Requested: {requested}")
    print(f"Available: {available}  (unavailable are skipped as N/A)")

    # (m, n, k) sweep: square GEMMs + weight-fixed m sweeps (token/batch dim)
    shapes: List[Tuple[int, int, int]] = []
    for mnk in (1024, 2048, 4096, 8192):
        shapes.append((mnk, mnk, mnk))
    for (n, k) in ((4096, 4096), (8192, 8192)):
        for m in (16, 64, 256, 512, 2048):
            shapes.append((m, n, k))

    print("\nCorrectness (cosine-sim vs torch.matmul, at 512x4096x4096):")
    for p in available:
        ok, cos = verify_correctness(512, 4096, 4096, p)
        print(f"  {p:>9}: {'OK ' if ok else 'FAIL'} cos={cos:.4f}")

    results = run_benchmark_sweep(shapes, available)
    baseline = args.baseline if args.baseline in available else available[0]
    print_summary_table(shapes, results, available, baseline)

    if args.csv:
        write_csv(args.csv, results)
    for p in available:
        if p != baseline:
            create_heatmap(shapes, results, p, baseline,
                           f"{args.output_prefix}_{p}_vs_{baseline}.png")

    print("\n" + "=" * 78)
    print("BENCHMARK COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
