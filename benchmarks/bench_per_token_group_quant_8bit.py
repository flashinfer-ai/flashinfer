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

Benchmark: Per-Token-Group 8-bit Quantization Backend Comparison

Compares FlashInfer's ``per_token_group_quant_8bit`` cuTile backend against
external references across a (num_tokens, hidden_dim) sweep:

  - ``cutile`` : FlashInfer ``per_token_group_quant_8bit(backend="cutile")``
  - ``sgl``    : SGLang ``sgl_kernel.sgl_per_token_group_quant_8bit`` (SOTA
                 baseline used by ocean-eval's dashboard for this op)
  - ``triton`` : SGLang Triton ``per_token_group_quant_fp8`` reference
  - ``torch``  : PyTorch-native reference

Providers whose backend is unavailable are skipped (NaN), so ``cutile`` (needs
the cuda.tile toolchain image) and ``sgl``/``triton`` (need an sglang image)
can be measured separately and their result CSVs merged. Emits a per-provider
latency table, a speedup summary vs the SOTA baseline, and a heatmap.

Usage:
    python bench_per_token_group_quant_8bit.py
    python bench_per_token_group_quant_8bit.py --providers cutile,torch
    python bench_per_token_group_quant_8bit.py --dst-dtype int8 --csv out.csv

Requirements:
    - cuda.tile toolchain (cutile provider); sgl_kernel / sglang (sgl, triton)
    - matplotlib for the heatmap
"""

import argparse
import csv as _csv
import numpy as np
import torch
from typing import Callable, Dict, List, Optional, Tuple

from flashinfer.testing.utils import bench_gpu_time

GROUP_SIZE_DEFAULT = 128
EPS = 1e-10


# --------------------------------------------------------------------------- #
# Provider availability
# --------------------------------------------------------------------------- #
def _has_cutile() -> bool:
    try:
        import cuda.tile  # noqa: F401
        from flashinfer.quantization import per_token_group_quant_8bit  # noqa: F401

        return True
    except Exception:
        return False


def _has_sgl() -> bool:
    try:
        from sgl_kernel import sgl_per_token_group_quant_8bit  # noqa: F401

        return True
    except Exception:
        return False


def _has_triton() -> bool:
    try:
        from sglang.srt.layers.quantization.fp8_kernel import (  # noqa: F401
            per_token_group_quant_8bit as _sgl_triton_quant,
        )

        return True
    except Exception:
        return False


ALL_PROVIDERS = ["cutile", "sgl", "triton", "torch"]
_AVAIL = {
    "cutile": _has_cutile,
    "sgl": _has_sgl,
    "triton": _has_triton,
    "torch": lambda: True,
}


# --------------------------------------------------------------------------- #
# Reference + per-provider callables
# --------------------------------------------------------------------------- #
def _torch_reference(
    x: torch.Tensor, group_size: int, dst_dtype: torch.dtype, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    if dst_dtype == torch.int8:
        qmin, qmax = -128.0, 127.0
    else:
        finfo = torch.finfo(dst_dtype)
        qmin, qmax = finfo.min, finfo.max
    x_ = x.reshape(x.numel() // group_size, group_size).to(torch.float32)
    amax = x_.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    x_s = amax / qmax
    x_q = (x_ / x_s).clamp(qmin, qmax)
    x_q = (x_q.round() if dst_dtype == torch.int8 else x_q).to(dst_dtype).reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))
    return x_q, x_s


def _make_execute(
    provider: str,
    x: torch.Tensor,
    group_size: int,
    dst_dtype: torch.dtype,
    eps: float,
) -> Callable[[], Tuple[torch.Tensor, torch.Tensor]]:
    if provider == "cutile":
        from flashinfer.quantization import per_token_group_quant_8bit

        def run():
            return per_token_group_quant_8bit(
                x, group_size, eps=eps, dst_dtype=dst_dtype, backend="cutile"
            )

    elif provider == "sgl":
        from sgl_kernel import sgl_per_token_group_quant_8bit

        if dst_dtype == torch.int8:
            qmin, qmax = -128.0, 127.0
        else:
            finfo = torch.finfo(dst_dtype)
            qmin, qmax = finfo.min, finfo.max
        x_q = torch.empty_like(x, dtype=dst_dtype)
        x_s = torch.empty(
            x.shape[:-1] + (x.shape[-1] // group_size,),
            device=x.device,
            dtype=torch.float32,
        )

        def run():
            sgl_per_token_group_quant_8bit(
                x, x_q, x_s, group_size, eps, qmin, qmax, False, enable_v2=False
            )
            return x_q, x_s

    elif provider == "triton":
        from sglang.srt.layers.quantization.fp8_kernel import (
            per_token_group_quant_8bit as sgl_triton_quant,
        )

        def run():
            return sgl_triton_quant(x, group_size, dst_dtype, eps=eps)

    elif provider == "torch":

        def run():
            return _torch_reference(x, group_size, dst_dtype, eps)

    else:
        raise ValueError(f"Unknown provider: {provider}")

    return run


# --------------------------------------------------------------------------- #
# Correctness + timing
# --------------------------------------------------------------------------- #
def verify_correctness(
    num_tokens: int,
    hidden_dim: int,
    group_size: int,
    dst_dtype: torch.dtype,
    provider: str,
) -> Tuple[bool, float]:
    """Return (ok, cosine_similarity) of provider quant output vs torch reference."""
    torch.manual_seed(42)
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device="cuda")
    try:
        q, _ = _make_execute(provider, x, group_size, dst_dtype, EPS)()
        ref_q, _ = _torch_reference(x, group_size, dst_dtype, EPS)
        a = q.to(torch.float32).reshape(1, -1)
        b = ref_q.to(torch.float32).reshape(1, -1)
        cos = torch.nn.functional.cosine_similarity(a, b).item()
        return cos > 0.99, cos
    except Exception:
        return False, 0.0


def bench_one(
    num_tokens: int,
    hidden_dim: int,
    group_size: int,
    dst_dtype: torch.dtype,
    provider: str,
) -> float:
    """Median latency (ms) for one (shape, provider); NaN if unavailable/failed."""
    if not _AVAIL[provider]():
        return float("nan")
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device="cuda")
    try:
        run = _make_execute(provider, x, group_size, dst_dtype, EPS)
        run()  # warmup / compile
        times = bench_gpu_time(
            fn=run,
            enable_cupti=True,
            dry_run_iters=5,
            repeat_iters=30,
            cold_l2_cache=True,
            use_cuda_graph=False,
        )
        return float(np.median(times))
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def run_benchmark_sweep(
    num_tokens_values: List[int],
    hidden_values: List[int],
    group_size: int,
    dst_dtype: torch.dtype,
    providers: List[str],
) -> Dict[str, Dict[Tuple[int, int], float]]:
    """Return {provider: {(num_tokens, hidden): median_ms}}."""
    results: Dict[str, Dict[Tuple[int, int], float]] = {p: {} for p in providers}
    total = len(num_tokens_values) * len(hidden_values)
    current = 0

    print(f"\nBenchmarking per_token_group_quant_8bit  gs={group_size}  bf16->{dst_dtype}")
    print("=" * (30 + 12 * len(providers)))
    header = f"{'ntok':>6} {'hidden':>7} |"
    for p in providers:
        header += f" {p:>10}"
    print(header)
    print("-" * (30 + 12 * len(providers)))

    for ntok in num_tokens_values:
        for hidden in hidden_values:
            current += 1
            row = f"{ntok:>6} {hidden:>7} |"
            for p in providers:
                ms = bench_one(ntok, hidden, group_size, dst_dtype, p)
                results[p][(ntok, hidden)] = ms
                row += f" {ms:>10.5f}" if ms == ms else f" {'--':>10}"
            print(f"[{current:3d}/{total}] " + row)

    return results


def print_summary_table(
    num_tokens_values: List[int],
    hidden_values: List[int],
    results: Dict[str, Dict[Tuple[int, int], float]],
    providers: List[str],
    baseline: str,
):
    """Speedup of each provider vs the baseline (>1 = provider faster)."""
    if baseline not in results:
        return
    for p in providers:
        if p == baseline:
            continue
        print(f"\n{'=' * 72}")
        print(f"Speedup: {p} vs {baseline} (baseline_ms / {p}_ms;  >1 = {p} faster)")
        print(f"{'=' * 72}")
        header = "ntok\\hidden".ljust(12)
        for h in hidden_values:
            header += f"{h:>10}"
        print(header)
        print("-" * (12 + 10 * len(hidden_values)))
        ratios = []
        for ntok in num_tokens_values:
            row = f"{ntok:<12}"
            for h in hidden_values:
                bt = results[baseline].get((ntok, h), float("nan"))
                pt = results[p].get((ntok, h), float("nan"))
                if pt == pt and bt == bt and pt > 0:
                    r = bt / pt
                    ratios.append(r)
                    row += f"{r:>10.2f}"
                else:
                    row += f"{'N/A':>10}"
            print(row)
        if ratios:
            print(
                f"\n  geomean {np.exp(np.mean(np.log(ratios))):.2f}x  "
                f"min {min(ratios):.2f}x  max {max(ratios):.2f}x  "
                f"({sum(1 for r in ratios if r > 1)}/{len(ratios)} shapes {p} faster)"
            )


def write_csv(path: str, results: Dict[str, Dict[Tuple[int, int], float]]):
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["provider", "num_tokens", "hidden_dim", "median_ms"])
        for provider, d in results.items():
            for (ntok, hidden), ms in sorted(d.items()):
                w.writerow([provider, ntok, hidden, f"{ms:.6f}"])
    print(f"\nWrote {path}")


def create_heatmap(
    num_tokens_values: List[int],
    hidden_values: List[int],
    results: Dict[str, Dict[Tuple[int, int], float]],
    provider: str,
    baseline: str,
    output_file: str,
):
    """Heatmap of `provider` speedup vs `baseline` over (num_tokens x hidden)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not installed, skipping heatmap")
        return
    if provider not in results or baseline not in results:
        return
    mat = np.full((len(num_tokens_values), len(hidden_values)), np.nan)
    for i, ntok in enumerate(num_tokens_values):
        for j, h in enumerate(hidden_values):
            bt = results[baseline].get((ntok, h), float("nan"))
            pt = results[provider].get((ntok, h), float("nan"))
            if pt == pt and bt == bt and pt > 0:
                mat[i, j] = bt / pt
    if np.all(np.isnan(mat)):
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(hidden_values)), max(5, 0.5 * len(num_tokens_values))))
    norm = mcolors.TwoSlopeNorm(
        vmin=min(0.5, np.nanmin(mat)), vcenter=1.0, vmax=max(1.5, np.nanmax(mat))
    )
    im = ax.imshow(mat, cmap="RdYlGn", norm=norm, aspect="auto")
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel(f"{baseline}_ms / {provider}_ms", rotation=-90, va="bottom")
    ax.set_xticks(np.arange(len(hidden_values)))
    ax.set_yticks(np.arange(len(num_tokens_values)))
    ax.set_xticklabels([str(h) for h in hidden_values])
    ax.set_yticklabels([str(n) for n in num_tokens_values])
    for i in range(len(num_tokens_values)):
        for j in range(len(hidden_values)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if mat[i, j] < 0.7 or mat[i, j] > 1.5 else "black")
    ax.set_xlabel("hidden_dim")
    ax.set_ylabel("num_tokens")
    ax.set_title(f"per_token_group_quant_8bit: {provider} speedup vs {baseline}\n(>1.0 = {provider} faster)")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved heatmap to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark per_token_group_quant_8bit backends"
    )
    parser.add_argument("--dst-dtype", choices=["fp8_e4m3", "int8"], default="fp8_e4m3")
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE_DEFAULT)
    parser.add_argument(
        "--providers",
        type=str,
        default=",".join(ALL_PROVIDERS),
        help=f"Comma-separated subset of {ALL_PROVIDERS}",
    )
    parser.add_argument("--baseline", type=str, default="sgl", help="Speedup baseline provider")
    parser.add_argument("--output-prefix", type=str, default="per_token_group_quant_8bit")
    parser.add_argument("--csv", type=str, default=None, help="Write raw medians to CSV")
    args = parser.parse_args()

    dst_dtype = torch.float8_e4m3fn if args.dst_dtype == "fp8_e4m3" else torch.int8
    requested = [p.strip() for p in args.providers.split(",") if p.strip()]
    providers = [p for p in requested if p in ALL_PROVIDERS]
    available = [p for p in providers if _AVAIL[p]()]
    print(f"Requested providers: {providers}")
    print(f"Available here:      {available}  (unavailable are skipped as N/A)")

    num_tokens_values = [1, 16, 64, 128, 256, 384, 512, 768, 1024, 2048, 4096]
    hidden_values = [2048, 4096, 7168]

    # Inline correctness check (once per available provider at a mid shape)
    print("\nCorrectness (cosine-sim of quant vs torch ref, at 512x4096):")
    for p in available:
        ok, cos = verify_correctness(512, 4096, args.group_size, dst_dtype, p)
        print(f"  {p:>8}: {'OK ' if ok else 'FAIL'} cos={cos:.4f}")

    results = run_benchmark_sweep(
        num_tokens_values, hidden_values, args.group_size, dst_dtype, providers
    )
    baseline = args.baseline if args.baseline in providers else providers[0]
    print_summary_table(num_tokens_values, hidden_values, results, providers, baseline)

    if args.csv:
        write_csv(args.csv, results)

    for p in providers:
        if p != baseline:
            create_heatmap(
                num_tokens_values, hidden_values, results, p, baseline,
                f"{args.output_prefix}_{p}_vs_{baseline}_{args.dst_dtype}.png",
            )

    print("\n" + "=" * 72)
    print("BENCHMARK COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
