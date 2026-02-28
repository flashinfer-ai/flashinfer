"""
Benchmark for Top-K operations including:
- top_k: Basic radix-based top-k selection
- top_k_page_table_transform: Fused top-k + page table gather (for sparse attention)
- top_k_ragged_transform: Fused top-k + offset addition (for sparse attention)

Optional comparison with SGLang's sgl_kernel implementation.
"""

import argparse
import os

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


def set_topk_algo(algo: str):
    """Set environment variable to force specific topk algorithm."""
    if algo == "auto":
        os.environ.pop("FLASHINFER_TOPK_ALGO", None)
    else:
        os.environ["FLASHINFER_TOPK_ALGO"] = algo


# Try to import sgl_kernel for comparison
try:
    import sgl_kernel

    HAS_SGL_KERNEL = True
except ImportError:
    HAS_SGL_KERNEL = False


def bench_top_k(
    batch_size: int,
    seq_len: int,
    k: int,
    dtype: torch.dtype = torch.float32,
    compare_sglang: bool = False,
) -> dict:
    """Benchmark basic top_k operation."""
    scores = torch.randn(batch_size, seq_len, device="cuda", dtype=dtype)

    # FlashInfer top_k
    measurements = bench_gpu_time(
        lambda: flashinfer.top_k(scores, k),
        enable_cupti=True,
        dry_run_iters=10,
        repeat_iters=100,
    )
    fi_ms = np.median(measurements)

    result = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "k": k,
        "dtype": str(dtype),
        "flashinfer_us": fi_ms * 1e3,
    }

    # Compare with torch.topk
    measurements = bench_gpu_time(
        lambda: torch.topk(scores, k, dim=-1),
        enable_cupti=True,
        dry_run_iters=10,
        repeat_iters=100,
    )
    torch_ms = np.median(measurements)
    result["torch_us"] = torch_ms * 1e3
    result["speedup_vs_torch"] = torch_ms / fi_ms

    # SGLang comparison (only supports k=2048 and float32)
    if compare_sglang and HAS_SGL_KERNEL and k == 2048 and dtype == torch.float32:
        lengths = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
        measurements = bench_gpu_time(
            lambda: sgl_kernel.fast_topk_v2(scores, lengths, k, row_starts=None),
            enable_cupti=True,
            dry_run_iters=10,
            repeat_iters=100,
        )
        sg_ms = np.median(measurements)
        result["sglang_us"] = sg_ms * 1e3
        result["speedup_vs_sglang"] = sg_ms / fi_ms

    return result


def bench_page_table_transform(
    batch_size: int,
    seq_len: int,
    k: int,
    dtype: torch.dtype = torch.float32,
    compare_sglang: bool = False,
) -> dict:
    """Benchmark fused top_k + page table transform."""
    scores = torch.randn(batch_size, seq_len, device="cuda", dtype=dtype)
    lengths = torch.full((batch_size,), seq_len, device="cuda", dtype=torch.int32)
    src_page_table = (
        torch.arange(seq_len, device="cuda", dtype=torch.int32)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )

    # FlashInfer
    measurements = bench_gpu_time(
        lambda: flashinfer.top_k_page_table_transform(
            scores, src_page_table, lengths, k
        ),
        enable_cupti=True,
        dry_run_iters=10,
        repeat_iters=100,
    )
    fi_ms = np.median(measurements)

    result = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "k": k,
        "dtype": str(dtype),
        "flashinfer_us": fi_ms * 1e3,
    }

    # SGLang comparison (only supports k=2048 and float32)
    if compare_sglang and HAS_SGL_KERNEL and k == 2048 and dtype == torch.float32:
        cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")
        measurements = bench_gpu_time(
            lambda: sgl_kernel.fast_topk_transform_fused(
                scores, lengths, src_page_table, cu_seqlens_q, k
            ),
            enable_cupti=True,
            dry_run_iters=10,
            repeat_iters=100,
        )
        sg_ms = np.median(measurements)
        result["sglang_us"] = sg_ms * 1e3
        result["speedup_vs_sglang"] = sg_ms / fi_ms

    return result


def bench_ragged_transform(
    batch_size: int,
    seq_len: int,
    k: int,
    dtype: torch.dtype = torch.float32,
    compare_sglang: bool = False,
) -> dict:
    """Benchmark fused top_k + ragged index transform."""
    scores = torch.randn(batch_size, seq_len, device="cuda", dtype=dtype)
    lengths = torch.full((batch_size,), seq_len, device="cuda", dtype=torch.int32)
    offsets = torch.arange(
        0, batch_size * seq_len, seq_len, device="cuda", dtype=torch.int32
    )

    # FlashInfer
    measurements = bench_gpu_time(
        lambda: flashinfer.top_k_ragged_transform(scores, offsets, lengths, k),
        enable_cupti=True,
        dry_run_iters=10,
        repeat_iters=100,
    )
    fi_ms = np.median(measurements)

    result = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "k": k,
        "dtype": str(dtype),
        "flashinfer_us": fi_ms * 1e3,
    }

    # SGLang comparison (only supports k=2048 and float32)
    if compare_sglang and HAS_SGL_KERNEL and k == 2048 and dtype == torch.float32:
        measurements = bench_gpu_time(
            lambda: sgl_kernel.fast_topk_transform_ragged_fused(
                scores, lengths, offsets, k
            ),
            enable_cupti=True,
            dry_run_iters=10,
            repeat_iters=100,
        )
        sg_ms = np.median(measurements)
        result["sglang_us"] = sg_ms * 1e3
        result["speedup_vs_sglang"] = sg_ms / fi_ms

    return result


def parse_dtype(dtype_str: str) -> torch.dtype:
    """Parse dtype string to torch.dtype."""
    dtype_map = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_str.lower()]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Benchmark Top-K operations")
    parser.add_argument(
        "--compare-sglang",
        action="store_true",
        help="Compare with SGLang's sgl_kernel (requires sgl_kernel installed)",
    )
    parser.add_argument(
        "--op",
        choices=["all", "top_k", "page_table", "ragged"],
        default="all",
        help="Which operation to benchmark",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        help="Data type: fp32, fp16, or bf16 (default: fp32)",
    )
    parser.add_argument(
        "--compare-algorithms",
        action="store_true",
        help="Compare multi-CTA vs filtered algorithms",
    )
    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)

    if args.compare_sglang and not HAS_SGL_KERNEL:
        print("WARNING: sgl_kernel not found, skipping SGLang comparison")
        args.compare_sglang = False

    # Test configurations
    batch_sizes = [1, 16, 64, 256]
    seq_lens = [4096, 16384, 65536, 131072, 262144, 524288]
    k_values = [256, 512, 1024, 2048, 4096]

    dtype_str = args.dtype.upper()

    # Algorithm comparison mode
    if args.compare_algorithms:
        print("=" * 100)
        print(
            f"Algorithm comparison: Multi-CTA vs Filtered (dtype={dtype_str}, k=2048)"
        )
        print("=" * 100)
        print(
            f"{'batch':>6} {'seq_len':>10} | {'Multi-CTA':>12} {'Filtered':>12} {'Winner':>12}"
        )
        print("-" * 70)

        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                k = 2048
                if k > seq_len:
                    continue
                try:
                    # Benchmark Multi-CTA
                    set_topk_algo("multi_cta")
                    result_mc = bench_page_table_transform(
                        batch_size, seq_len, k, dtype
                    )
                    mc_us = result_mc["flashinfer_us"]

                    # Benchmark Filtered
                    set_topk_algo("filtered")
                    result_f = bench_page_table_transform(batch_size, seq_len, k, dtype)
                    f_us = result_f["flashinfer_us"]

                    # Reset to auto
                    set_topk_algo("auto")

                    winner = "Multi-CTA" if mc_us < f_us else "Filtered"
                    speedup = max(mc_us, f_us) / min(mc_us, f_us)
                    print(
                        f"{batch_size:>6} {seq_len:>10} | "
                        f"{mc_us:>10.2f}us {f_us:>10.2f}us "
                        f"{winner:>8} {speedup:.2f}x"
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"{batch_size:>6} {seq_len:>10} | OOM")
                        torch.cuda.empty_cache()
                    else:
                        raise
        return

    if args.op in ["all", "top_k"]:
        print("=" * 100)
        print(f"top_k: Basic radix-based top-k selection (dtype={dtype_str})")
        if args.compare_sglang:
            print("NOTE: SGLang only supports k=2048 and float32")
        print("=" * 100)

        header = f"{'batch':>6} {'seq_len':>10} {'k':>6} | {'FlashInfer':>12} {'torch.topk':>12} {'Speedup':>10}"
        if args.compare_sglang:
            header += f" {'SGLang':>12} {'Speedup':>10}"
        print(header)
        print("-" * (70 if not args.compare_sglang else 90))

        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                for k in k_values:
                    if k > seq_len:
                        continue
                    try:
                        result = bench_top_k(
                            batch_size,
                            seq_len,
                            k,
                            dtype,
                            compare_sglang=args.compare_sglang,
                        )
                        line = (
                            f"{result['batch_size']:>6} {result['seq_len']:>10} {result['k']:>6} | "
                            f"{result['flashinfer_us']:>10.2f}us {result['torch_us']:>10.2f}us "
                            f"{result['speedup_vs_torch']:>9.2f}x"
                        )
                        if "sglang_us" in result:
                            line += f" {result['sglang_us']:>10.2f}us {result['speedup_vs_sglang']:>9.2f}x"
                        elif args.compare_sglang and k == 2048:
                            line += " (SGLang error)"
                        print(line)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"{batch_size:>6} {seq_len:>10} {k:>6} | OOM")
                            torch.cuda.empty_cache()
                        else:
                            raise

    if args.op in ["all", "page_table"]:
        print("\n" + "=" * 100)
        print(
            f"top_k_page_table_transform: Fused top-k + page table gather (dtype={dtype_str})"
        )
        if args.compare_sglang:
            print("NOTE: SGLang only supports k=2048 and float32")
        print("=" * 100)

        header = f"{'batch':>6} {'seq_len':>10} {'k':>6} | {'FlashInfer':>12}"
        if args.compare_sglang:
            header += f" {'SGLang':>12} {'Speedup':>10}"
        print(header)
        print("-" * (70 if not args.compare_sglang else 90))

        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                for k in k_values:
                    if k > seq_len:
                        continue
                    try:
                        result = bench_page_table_transform(
                            batch_size,
                            seq_len,
                            k,
                            dtype,
                            compare_sglang=args.compare_sglang,
                        )
                        line = (
                            f"{result['batch_size']:>6} {result['seq_len']:>10} {result['k']:>6} | "
                            f"{result['flashinfer_us']:>10.2f}us"
                        )
                        if "sglang_us" in result:
                            line += f" {result['sglang_us']:>10.2f}us {result['speedup_vs_sglang']:>9.2f}x"
                        elif args.compare_sglang and k == 2048:
                            line += " (SGLang error)"
                        print(line)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"{batch_size:>6} {seq_len:>10} {k:>6} | OOM")
                            torch.cuda.empty_cache()
                        else:
                            raise

    if args.op in ["all", "ragged"]:
        print("\n" + "=" * 100)
        print(
            f"top_k_ragged_transform: Fused top-k + ragged index transform (dtype={dtype_str})"
        )
        if args.compare_sglang:
            print("NOTE: SGLang only supports k=2048 and float32")
        print("=" * 100)

        header = f"{'batch':>6} {'seq_len':>10} {'k':>6} | {'FlashInfer':>12}"
        if args.compare_sglang:
            header += f" {'SGLang':>12} {'Speedup':>10}"
        print(header)
        print("-" * (70 if not args.compare_sglang else 90))

        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                for k in k_values:
                    if k > seq_len:
                        continue
                    try:
                        result = bench_ragged_transform(
                            batch_size,
                            seq_len,
                            k,
                            dtype,
                            compare_sglang=args.compare_sglang,
                        )
                        line = (
                            f"{result['batch_size']:>6} {result['seq_len']:>10} {result['k']:>6} | "
                            f"{result['flashinfer_us']:>10.2f}us"
                        )
                        if "sglang_us" in result:
                            line += f" {result['sglang_us']:>10.2f}us {result['speedup_vs_sglang']:>9.2f}x"
                        elif args.compare_sglang and k == 2048:
                            line += " (SGLang error)"
                        print(line)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"{batch_size:>6} {seq_len:>10} {k:>6} | OOM")
                            torch.cuda.empty_cache()
                        else:
                            raise


if __name__ == "__main__":
    main()
