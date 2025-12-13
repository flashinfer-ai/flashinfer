"""
Benchmark for Top-K operations including:
- top_k: Basic radix-based top-k selection
- top_k_page_table_transform: Fused top-k + page table gather (for sparse attention)
- top_k_ragged_transform: Fused top-k + offset addition (for sparse attention)

Optional comparison with SGLang's sgl_kernel implementation.
"""

import argparse

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time

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
        dry_run_time_ms=100,
        repeat_time_ms=1000,
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
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
    torch_ms = np.median(measurements)
    result["torch_us"] = torch_ms * 1e3
    result["speedup_vs_torch"] = torch_ms / fi_ms

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
        # lambda: flashinfer.sampling.top_k_mask_logits(
        #    scores, k
        # ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
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
            dry_run_time_ms=100,
            repeat_time_ms=1000,
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
        dry_run_time_ms=100,
        repeat_time_ms=1000,
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
            dry_run_time_ms=100,
            repeat_time_ms=1000,
        )
        sg_ms = np.median(measurements)
        result["sglang_us"] = sg_ms * 1e3
        result["speedup_vs_sglang"] = sg_ms / fi_ms

    return result


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
    args = parser.parse_args()

    if args.compare_sglang and not HAS_SGL_KERNEL:
        print("WARNING: sgl_kernel not found, skipping SGLang comparison")
        args.compare_sglang = False

    # Test configurations
    batch_sizes = [1, 16, 64, 256]
    seq_lens = [4096, 16384, 65536, 131072, 262144, 524288]
    k_values = [256, 512, 1024, 2048, 4096]

    if args.op in ["all", "top_k"]:
        print("=" * 100)
        print("top_k: Basic radix-based top-k selection")
        print("=" * 100)
        print(
            f"{'batch':>6} {'seq_len':>10} {'k':>6} | {'FlashInfer':>12} {'torch.topk':>12} {'Speedup':>10}"
        )
        print("-" * 70)

        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                for k in k_values:
                    if k > seq_len:
                        continue
                    try:
                        result = bench_top_k(batch_size, seq_len, k)
                        print(
                            f"{result['batch_size']:>6} {result['seq_len']:>10} {result['k']:>6} | "
                            f"{result['flashinfer_us']:>10.2f}us {result['torch_us']:>10.2f}us "
                            f"{result['speedup_vs_torch']:>9.2f}x"
                        )
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"{batch_size:>6} {seq_len:>10} {k:>6} | OOM")
                            torch.cuda.empty_cache()
                        else:
                            raise

    if args.op in ["all", "page_table"]:
        print("\n" + "=" * 100)
        print("top_k_page_table_transform: Fused top-k + page table gather")
        if args.compare_sglang:
            print("NOTE: SGLang only supports k=2048")
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
                            batch_size, seq_len, k, compare_sglang=args.compare_sglang
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
        print("top_k_ragged_transform: Fused top-k + ragged index transform")
        if args.compare_sglang:
            print("NOTE: SGLang only supports k=2048")
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
                            batch_size, seq_len, k, compare_sglang=args.compare_sglang
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

    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)
    print(
        """
FlashInfer Top-K advantages:
- Supports arbitrary k values (SGLang only supports k=2048)
- Supports sequence lengths > 128K (SGLang limited to 131072)
- Faster for long sequences (>= 64K)

SGLang advantages:
- Slightly faster for short sequences (< 32K) with k=2048

Use FlashInfer when:
- You need flexible k values
- You have long sequences (>= 64K)
- You need to support sequences > 128K
"""
    )


if __name__ == "__main__":
    main()
