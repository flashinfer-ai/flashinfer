"""
Benchmark for Top-K operations including:
- top_k: Basic radix-based top-k selection
- top_k_page_table_transform: Fused top-k + page table gather (for sparse attention)
- top_k_ragged_transform: Fused top-k + offset addition (for sparse attention)

Optional comparison with SGLang's sgl_kernel implementation.
"""

import argparse
import os
from contextlib import contextmanager
from dataclasses import dataclass

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


@contextmanager
def torch_deterministic_algorithms(enabled: bool):
    """Temporarily set PyTorch deterministic algorithm mode."""
    previous = torch.are_deterministic_algorithms_enabled()
    if previous != enabled:
        torch.use_deterministic_algorithms(enabled)
    try:
        yield
    finally:
        if torch.are_deterministic_algorithms_enabled() != previous:
            torch.use_deterministic_algorithms(previous)


def run_torch_topk(scores: torch.Tensor, k: int, deterministic: bool):
    """Run torch.topk with optional deterministic algorithm mode."""
    if deterministic:
        with torch_deterministic_algorithms(True):
            return torch.topk(scores, k, dim=-1)
    return torch.topk(scores, k, dim=-1)


def bench_top_k_from_scores(
    scores: torch.Tensor,
    k: int,
    deterministic: bool = False,
    compare_torch_deterministic: bool = False,
    compare_sglang: bool = False,
) -> dict:
    """Benchmark top-k on a pre-generated score tensor."""
    batch_size, seq_len = scores.shape

    measurements = bench_gpu_time(
        lambda: flashinfer.top_k(
            scores,
            k,
            deterministic=deterministic,
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
        "dtype": str(scores.dtype),
        "flashinfer_us": fi_ms * 1e3,
    }

    measurements = bench_gpu_time(
        lambda: torch.topk(scores, k, dim=-1),
        enable_cupti=True,
        dry_run_iters=10,
        repeat_iters=100,
    )
    torch_ms = np.median(measurements)
    result["torch_us"] = torch_ms * 1e3
    result["speedup_vs_torch"] = torch_ms / fi_ms

    if compare_torch_deterministic:
        measurements = bench_gpu_time(
            lambda: run_torch_topk(scores, k, deterministic=True),
            enable_cupti=True,
            dry_run_iters=10,
            repeat_iters=100,
        )
        torch_det_ms = np.median(measurements)
        result["torch_deterministic_us"] = torch_det_ms * 1e3
        result["speedup_vs_torch_deterministic"] = torch_det_ms / fi_ms

    # SGLang comparison (only supports k=2048 and float32)
    if compare_sglang and HAS_SGL_KERNEL and k == 2048 and scores.dtype == torch.float32:
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


def generate_scores(
    batch_size: int,
    seq_len: int,
    k: int,
    dtype: torch.dtype,
    input_pattern: str,
) -> torch.Tensor:
    """Generate benchmark input scores with controllable tie patterns."""
    if input_pattern == "random":
        return torch.randn(batch_size, seq_len, device="cuda", dtype=dtype)

    if input_pattern == "tie_heavy":
        pattern = (
            torch.arange(seq_len, device="cuda", dtype=torch.float32) % 64
        ) / 64.0
        return pattern.unsqueeze(0).expand(batch_size, -1).contiguous().to(dtype)

    if input_pattern == "pivot_tie":
        # Severe tie at pivot:
        # - majority entries are identical (1.0)
        # - a small tail region is strictly larger (2.0)
        # This creates truncation in == pivot region when k exceeds tail size.
        scores = torch.ones(batch_size, seq_len, device="cuda", dtype=dtype)
        gt_count = max(1, min(k // 4, seq_len // 8))
        scores[:, seq_len - gt_count :] = 2.0
        return scores

    raise ValueError(f"Unsupported input_pattern: {input_pattern}")


def generate_dsa_scores(
    batch_size: int,
    q_len: int,
    seq_len: int,
    dtype: torch.dtype,
    input_pattern: str,
    causal_chunk: bool,
) -> torch.Tensor:
    """Generate DeepSeek DSA-like indexer score workload.

    Source context:
    - DeepSeek-V3.2-Exp config uses index_topk=2048:
      https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/config_671B_v3.2.json
    - Indexer runs topk over index_score last dim:
      https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py

    Returns a 2D tensor with shape (batch_size * q_len, seq_len).
    """
    rows = batch_size * q_len
    if input_pattern == "random":
        scores = torch.randn(rows, seq_len, device="cuda", dtype=dtype)
    elif input_pattern == "dsa_relu":
        # DSA index scores include ReLU; quantization here increases tie pressure.
        base = torch.randn(rows, seq_len, device="cuda", dtype=torch.float32)
        scores = torch.relu(base)
        scores = torch.round(scores * 32.0) / 32.0
        scores = scores.to(dtype)
    else:
        raise ValueError(f"Unsupported dsa input_pattern: {input_pattern}")

    if causal_chunk:
        # Simulate prefill chunk near end of long context:
        # each query row i can only attend [0, start_pos + i].
        start_pos = seq_len - q_len
        lengths = torch.arange(
            start_pos + 1,
            start_pos + q_len + 1,
            device="cuda",
            dtype=torch.int32,
        ).repeat(batch_size)
        col = torch.arange(seq_len, device="cuda", dtype=torch.int32).unsqueeze(0)
        invalid = col >= lengths.unsqueeze(1)
        neg_inf = -torch.inf if dtype == torch.float32 else torch.finfo(dtype).min
        scores = scores.masked_fill(invalid, neg_inf)

    return scores.contiguous()


@dataclass(frozen=True)
class DSATopKCase:
    name: str
    batch_size: int
    q_len: int
    seq_len: int
    causal_chunk: bool


def bench_dsa_top_k(
    batch_size: int,
    q_len: int,
    seq_len: int,
    k: int,
    dtype: torch.dtype = torch.bfloat16,
    input_pattern: str = "dsa_relu",
    deterministic: bool = False,
    compare_torch_deterministic: bool = False,
    compare_sglang: bool = False,
    causal_chunk: bool = False,
) -> dict:
    scores = generate_dsa_scores(
        batch_size=batch_size,
        q_len=q_len,
        seq_len=seq_len,
        dtype=dtype,
        input_pattern=input_pattern,
        causal_chunk=causal_chunk,
    )
    result = bench_top_k_from_scores(
        scores=scores,
        k=k,
        deterministic=deterministic,
        compare_torch_deterministic=compare_torch_deterministic,
        compare_sglang=compare_sglang,
    )
    result["rows"] = batch_size * q_len
    result["q_len"] = q_len
    result["case_type"] = "prefill" if causal_chunk else "decode"
    return result


def bench_top_k(
    batch_size: int,
    seq_len: int,
    k: int,
    dtype: torch.dtype = torch.float32,
    input_pattern: str = "random",
    deterministic: bool = False,
    compare_torch_deterministic: bool = False,
    compare_sglang: bool = False,
) -> dict:
    """Benchmark basic top_k operation."""
    scores = generate_scores(batch_size, seq_len, k, dtype, input_pattern)
    return bench_top_k_from_scores(
        scores=scores,
        k=k,
        deterministic=deterministic,
        compare_torch_deterministic=compare_torch_deterministic,
        compare_sglang=compare_sglang,
    )


def bench_page_table_transform(
    batch_size: int,
    seq_len: int,
    k: int,
    dtype: torch.dtype = torch.float32,
    input_pattern: str = "random",
    deterministic: bool = False,
    compare_sglang: bool = False,
) -> dict:
    """Benchmark fused top_k + page table transform."""
    scores = generate_scores(batch_size, seq_len, k, dtype, input_pattern)
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
            scores,
            src_page_table,
            lengths,
            k,
            deterministic=deterministic,
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
    input_pattern: str = "random",
    deterministic: bool = False,
    compare_sglang: bool = False,
) -> dict:
    """Benchmark fused top_k + ragged index transform."""
    scores = generate_scores(batch_size, seq_len, k, dtype, input_pattern)
    lengths = torch.full((batch_size,), seq_len, device="cuda", dtype=torch.int32)
    offsets = torch.arange(
        0, batch_size * seq_len, seq_len, device="cuda", dtype=torch.int32
    )

    # FlashInfer
    measurements = bench_gpu_time(
        lambda: flashinfer.top_k_ragged_transform(
            scores,
            offsets,
            lengths,
            k,
            deterministic=deterministic,
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
        choices=["all", "top_k", "dsa_topk", "page_table", "ragged"],
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
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode for FlashInfer top-k kernels",
    )
    parser.add_argument(
        "--compare-torch-deterministic",
        action="store_true",
        help="Also benchmark torch.topk under deterministic algorithm mode",
    )
    parser.add_argument(
        "--input-pattern",
        choices=["random", "tie_heavy", "pivot_tie"],
        default="random",
        help="Input score pattern: random | tie_heavy | pivot_tie",
    )
    parser.add_argument(
        "--dsa-input-pattern",
        choices=["random", "dsa_relu"],
        default="dsa_relu",
        help="DSA top-k input pattern: random | dsa_relu",
    )
    parser.add_argument(
        "--dsa-case",
        choices=["all", "decode", "prefill"],
        default="all",
        help="DSA case group: all | decode | prefill",
    )
    parser.add_argument(
        "--dsa-k",
        type=int,
        default=2048,
        help="Top-k for DSA workload (default: 2048, matching DeepSeek DSA config)",
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
        if args.deterministic:
            print(
                "ERROR: --compare-algorithms is only meaningful with non-deterministic mode"
            )
            return
        print("=" * 100)
        print(
            "Algorithm comparison: Multi-CTA vs Filtered "
            f"(dtype={dtype_str}, k=2048, pattern={args.input_pattern})"
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
                        batch_size, seq_len, k, dtype, args.input_pattern
                    )
                    mc_us = result_mc["flashinfer_us"]

                    # Benchmark Filtered
                    set_topk_algo("filtered")
                    result_f = bench_page_table_transform(
                        batch_size, seq_len, k, dtype, args.input_pattern
                    )
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
        print(
            "top_k: Basic radix-based top-k selection "
            f"(dtype={dtype_str}, deterministic={args.deterministic}, "
            f"pattern={args.input_pattern})"
        )
        if args.compare_sglang:
            print("NOTE: SGLang only supports k=2048 and float32")
        if args.compare_torch_deterministic:
            print(
                "NOTE: torch.det means torch.topk with torch.use_deterministic_algorithms(True)"
            )
        print("=" * 100)

        fi_label = "FlashInfer(det)" if args.deterministic else "FlashInfer"
        header = (
            f"{'batch':>6} {'seq_len':>10} {'k':>6} | "
            f"{fi_label:>14} {'torch.topk':>12} {'Speedup':>10}"
        )
        if args.compare_torch_deterministic:
            header += f" {'torch.det':>12} {'Speedup':>10}"
        if args.compare_sglang:
            header += f" {'SGLang':>12} {'Speedup':>10}"
        print(header)
        divider_len = 72
        if args.compare_torch_deterministic:
            divider_len += 24
        if args.compare_sglang:
            divider_len += 24
        print("-" * divider_len)

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
                            input_pattern=args.input_pattern,
                            deterministic=args.deterministic,
                            compare_torch_deterministic=args.compare_torch_deterministic,
                            compare_sglang=args.compare_sglang,
                        )
                        line = (
                            f"{result['batch_size']:>6} {result['seq_len']:>10} {result['k']:>6} | "
                            f"{result['flashinfer_us']:>12.2f}us {result['torch_us']:>10.2f}us "
                            f"{result['speedup_vs_torch']:>9.2f}x"
                        )
                        if "torch_deterministic_us" in result:
                            line += (
                                f" {result['torch_deterministic_us']:>10.2f}us "
                                f"{result['speedup_vs_torch_deterministic']:>9.2f}x"
                            )
                        if "sglang_us" in result:
                            line += (
                                f" {result['sglang_us']:>10.2f}us "
                                f"{result['speedup_vs_sglang']:>9.2f}x"
                            )
                        elif args.compare_sglang and k == 2048:
                            line += " (SGLang error)"
                        print(line)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"{batch_size:>6} {seq_len:>10} {k:>6} | OOM")
                            torch.cuda.empty_cache()
                        else:
                            raise

    if args.op in ["all", "dsa_topk"]:
        print("\n" + "=" * 100)
        print(
            "dsa_topk: DeepSeek DSA-like indexer top-k workload "
            f"(dtype={dtype_str}, deterministic={args.deterministic}, "
            f"dsa_pattern={args.dsa_input_pattern}, k={args.dsa_k})"
        )
        if args.compare_torch_deterministic:
            print(
                "NOTE: torch.det means torch.topk with torch.use_deterministic_algorithms(True)"
            )
        print("=" * 100)

        header = (
            f"{'case':>24} {'rows':>8} {'seq_len':>10} {'k':>6} | "
            f"{'FlashInfer':>12} {'torch.topk':>12} {'Speedup':>10}"
        )
        if args.compare_torch_deterministic:
            header += f" {'torch.det':>12} {'Speedup':>10}"
        print(header)
        print("-" * (86 if not args.compare_torch_deterministic else 110))

        dsa_cases = [
            # DeepSeek Sparse Attention proxy cases:
            # - decode: q_len=1
            # - prefill chunk: q_len>1 with causal availability growth
            # Ref: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf
            DSATopKCase("decode_b1_q1_l128k", 1, 1, 131072, False),
            DSATopKCase("decode_b8_q1_l64k", 8, 1, 65536, False),
            DSATopKCase("decode_b32_q1_l128k", 32, 1, 131072, False),
            DSATopKCase("prefill_b1_q128_l128k", 1, 128, 131072, True),
        ]

        for case in dsa_cases:
            if args.dsa_case == "decode" and case.causal_chunk:
                continue
            if args.dsa_case == "prefill" and not case.causal_chunk:
                continue
            if args.dsa_k > case.seq_len:
                continue
            try:
                result = bench_dsa_top_k(
                    batch_size=case.batch_size,
                    q_len=case.q_len,
                    seq_len=case.seq_len,
                    k=args.dsa_k,
                    dtype=dtype,
                    input_pattern=args.dsa_input_pattern,
                    deterministic=args.deterministic,
                    compare_torch_deterministic=args.compare_torch_deterministic,
                    compare_sglang=False,
                    causal_chunk=case.causal_chunk,
                )
                line = (
                    f"{case.name:>24} {result['rows']:>8} {result['seq_len']:>10} {result['k']:>6} | "
                    f"{result['flashinfer_us']:>10.2f}us {result['torch_us']:>10.2f}us "
                    f"{result['speedup_vs_torch']:>9.2f}x"
                )
                if "torch_deterministic_us" in result:
                    line += (
                        f" {result['torch_deterministic_us']:>10.2f}us "
                        f"{result['speedup_vs_torch_deterministic']:>9.2f}x"
                    )
                print(line)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"{case.name:>24} {case.batch_size * case.q_len:>8} {case.seq_len:>10} {args.dsa_k:>6} | OOM")
                    torch.cuda.empty_cache()
                else:
                    raise

    if args.op in ["all", "page_table"]:
        print("\n" + "=" * 100)
        print(
            "top_k_page_table_transform: Fused top-k + page table gather "
            f"(dtype={dtype_str}, pattern={args.input_pattern})"
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
                            input_pattern=args.input_pattern,
                            deterministic=args.deterministic,
                            compare_sglang=args.compare_sglang,
                        )
                        line = (
                            f"{result['batch_size']:>6} {result['seq_len']:>10} {result['k']:>6} | "
                            f"{result['flashinfer_us']:>10.2f}us"
                        )
                        if "sglang_us" in result:
                            line += (
                                f" {result['sglang_us']:>10.2f}us "
                                f"{result['speedup_vs_sglang']:>9.2f}x"
                            )
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
            "top_k_ragged_transform: Fused top-k + ragged index transform "
            f"(dtype={dtype_str}, pattern={args.input_pattern})"
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
                            input_pattern=args.input_pattern,
                            deterministic=args.deterministic,
                            compare_sglang=args.compare_sglang,
                        )
                        line = (
                            f"{result['batch_size']:>6} {result['seq_len']:>10} {result['k']:>6} | "
                            f"{result['flashinfer_us']:>10.2f}us"
                        )
                        if "sglang_us" in result:
                            line += (
                                f" {result['sglang_us']:>10.2f}us "
                                f"{result['speedup_vs_sglang']:>9.2f}x"
                            )
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
