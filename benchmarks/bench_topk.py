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


# NOTE: based on @AllenLin's benchmarking function here: https://github.com/flashinfer-ai/flashinfer/pull/2814 to compare. 
def bench_dsv3_topk(
    batch_size: int,
    seq_len: int,
    compare_sglang: bool = False,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Benchmark kernel-only top-K for DeepSeek v3.2 shapes (k=2048).

    All buffers pre-allocated outside the timing loop. Measures only the
    GPU kernel execution for both cute-dsl and cuda backends.
    """
    from flashinfer.topk.kernels.top_k_cute_dsl import (
        _get_chunk_config,
        _get_cluster_chunk_config,
        _get_compiled_kernel,
        _compute_max_chunk,
        _get_cached_row_states,
        _get_cached_seq_lens,
        _TORCH_TO_CUTLASS_DTYPE,
        CLUSTER_STATE_SIZE,
        DISTRIBUTED_STATE_SIZE,
    )
    from flashinfer.topk.kernels.single_pass_multi_cta_radix_topk_cluster import (
        _query_max_cluster_size,
    )
    from flashinfer.topk.topk import get_topk_module
    from flashinfer.cute_dsl.utils import get_num_sm
    from flashinfer.utils import _get_cache_buf

    k = 2048
    device = torch.device("cuda")

    logits = torch.empty(batch_size, seq_len, device=device, dtype=dtype)
    logits.uniform_(-1.0, 1.0)

    result = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "k": k,
    }

    # --- cute-dsl kernel-only ---
    cutlass_dtype = _TORCH_TO_CUTLASS_DTYPE[dtype]
    num_sms = get_num_sm(device)
    num_copy_bits = 256
    next_n = 1
    num_rows, num_cols = batch_size, seq_len

    is_fp32 = (dtype == torch.float32)
    if is_fp32 and num_cols < 65536:
        use_multi_cta = False
    else:
        chunk_size, ctas_per_group, _ = _get_chunk_config(
            cutlass_dtype, num_cols,
            num_copy_bits=num_copy_bits, num_rows=num_rows, num_sms=num_sms,
        )
        if ctas_per_group >= 2:
            use_multi_cta = (num_rows * ctas_per_group <= num_sms)
            if is_fp32:
                use_multi_cta = use_multi_cta and num_cols >= 65536
        else:
            use_multi_cta = (not is_fp32 and num_rows <= num_sms)

    if use_multi_cta:
        cluster_config = _get_cluster_chunk_config(
            cutlass_dtype, num_cols,
            num_copy_bits=num_copy_bits, num_rows=num_rows, num_sms=num_sms,
        )
        if cluster_config[0] is not None:
            chunk_size, ctas_per_group, _ = cluster_config
            max_chunk, _ = _compute_max_chunk(cutlass_dtype, num_copy_bits)
            hw_max_cluster = _query_max_cluster_size()
            if num_cols <= max_chunk * hw_max_cluster:
                kernel_variant = "cluster"
                state_size = CLUSTER_STATE_SIZE
            else:
                kernel_variant = "distributed"
                state_size = DISTRIBUTED_STATE_SIZE
                chunk_size, ctas_per_group, _ = _get_chunk_config(
                    cutlass_dtype, num_cols,
                    num_copy_bits=num_copy_bits, num_rows=num_rows, num_sms=num_sms,
                )
        else:
            kernel_variant = "distributed"
            state_size = DISTRIBUTED_STATE_SIZE
            chunk_size, ctas_per_group, _ = _get_chunk_config(
                cutlass_dtype, num_cols,
                num_copy_bits=num_copy_bits, num_rows=num_rows, num_sms=num_sms,
            )
    else:
        kernel_variant = "cluster"
        state_size = CLUSTER_STATE_SIZE
        chunk_size, ctas_per_group, _ = _get_chunk_config(
            cutlass_dtype, num_cols,
            num_copy_bits=num_copy_bits, num_rows=num_rows, num_sms=num_sms,
        )

    compiled_kernel = _get_compiled_kernel(
        kernel_variant, cutlass_dtype, chunk_size, k, next_n,
        num_copy_bits, ctas_per_group, num_sms, False,
    )

    row_states_2d = _get_cached_row_states(
        f"{kernel_variant}_{device}", num_sms, state_size, device
    )
    seq_lens_buf = _get_cached_seq_lens(num_rows, num_cols, device)
    cute_dsl_output = torch.empty(num_rows, k, dtype=torch.int32, device=device)

    measurements = bench_gpu_time(
        lambda: compiled_kernel(
            logits, row_states_2d, seq_lens_buf, cute_dsl_output, None,
        ),
        enable_cupti=True,
        dry_run_iters=10,
        repeat_iters=100,
    )
    result["cute_dsl_us"] = np.median(measurements) * 1e3

    # --- cuda kernel-only ---
    module = get_topk_module()
    cuda_row_states = _get_cache_buf(
        f"radix_topk_row_states_{device}", 1024 * 1024, device, zero_init=True,
    )
    cuda_output_values = torch.empty(num_rows, k, dtype=dtype, device=device)

    measurements = bench_gpu_time(
        lambda: module.radix_topk(logits, k, cuda_row_states, cuda_output_values),
        enable_cupti=True,
        dry_run_iters=10,
        repeat_iters=100,
    )
    cuda_us = np.median(measurements) * 1e3
    result["cuda_us"] = cuda_us
    result["speedup_vs_cuda"] = cuda_us / result["cute_dsl_us"]

    # --- SGLang ---
    if compare_sglang and HAS_SGL_KERNEL:
        seq_lens_sg = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
        measurements = bench_gpu_time(
            lambda: sgl_kernel.fast_topk_v2(logits, seq_lens_sg, k, row_starts=None),
            enable_cupti=True,
            dry_run_iters=10,
            repeat_iters=100,
        )
        sg_us = np.median(measurements) * 1e3
        result["sglang_us"] = sg_us
        result["speedup_vs_sglang"] = sg_us / result["cute_dsl_us"]

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
        choices=["all", "top_k", "page_table", "ragged", "dsv3_topk"],
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

    if args.op in ["all", "dsv3_topk"]:
        dsv3_batch_sizes = [1, 2, 4, 32, 64]
        dsv3_seq_lens = [1024, 4096, 8192, 32768, 40960]
        dsv3_dtypes = [torch.float32, torch.bfloat16]

        print("\n" + "=" * 100)
        print("dsv3_topk: DeepSeek v3.2 sparse-attention top-K (k=2048, multi-CTA configs)")
        print(
            "  cute-dsl:          CuTe-DSL single-pass multi-CTA radix top-K (cluster)"
        )
        print(
            "  cuda:              FlashInfer default CUDA radix top-K"
        )
        if args.compare_sglang:
            print("  SGLang fast_topk_v2: variable-length aware (requires sgl_kernel)")
        print("=" * 100)

        for bench_dtype in dsv3_dtypes:
            dtype_str = "fp32" if bench_dtype == torch.float32 else "bf16"
            print(f"\n  dtype={dtype_str}")
            header = f"  {'batch':>6} {'seq_len':>10} | {'cute-dsl':>12} {'cuda':>12} {'speedup':>10}"
            if args.compare_sglang:
                header += f" {'SGLang':>12} {'speedup':>10}"
            print(header)
            print("  " + "-" * (58 if not args.compare_sglang else 83))

            for batch_size in dsv3_batch_sizes:
                for seq_len in dsv3_seq_lens:
                    try:
                        result = bench_dsv3_topk(
                            batch_size,
                            seq_len,
                            compare_sglang=args.compare_sglang,
                            dtype=bench_dtype,
                        )
                        line = (
                            f"  {result['batch_size']:>6} {result['seq_len']:>10} | "
                            f"{result['cute_dsl_us']:>10.2f}us "
                            f"{result['cuda_us']:>10.2f}us "
                            f"{result['speedup_vs_cuda']:>9.2f}x"
                        )
                        if "sglang_us" in result:
                            line += f" {result['sglang_us']:>10.2f}us {result['speedup_vs_sglang']:>9.2f}x"
                        print(line)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"  {batch_size:>6} {seq_len:>10} | OOM")
                            torch.cuda.empty_cache()
                        else:
                            raise


if __name__ == "__main__":
    main()