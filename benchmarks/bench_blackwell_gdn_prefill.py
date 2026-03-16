"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Performance benchmark for GDN (Gated Delta Network) linear attention.

Usage:
    python bench_blackwell_gdn_prefill.py
    python bench_blackwell_gdn_prefill.py --batch-size 8 --seq-len 1024
    python bench_blackwell_gdn_prefill.py --varlen --batch-size 4 --seq-len 2048
    python bench_blackwell_gdn_prefill.py --varlen --cu-seqlens 0 512 1024 2048
    python bench_blackwell_gdn_prefill.py --sweep
"""

import argparse
import sys
from typing import List

import torch
import torch.nn.functional as F
import numpy as np
from flashinfer.testing import bench_gpu_time
from flashinfer.utils import is_sm100a_supported, is_sm110a_supported

# Blackwell GDN prefill
from flashinfer.gdn_prefill import chunk_gated_delta_rule

# FLA baseline
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd as fla_base


def benchmark_gdn_fixlen(
    batch_size: int,
    seq_len: int,
    num_qk_heads: int,
    num_v_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
    use_initial_state: bool = True,
) -> dict:
    """
    Benchmark GDN with fixed-length sequences.

    Returns:
        Dictionary with benchmark results.
    """
    device = "cuda"
    torch.cuda.reset_peak_memory_stats()

    # Create input tensors
    q = torch.randn(
        (batch_size, seq_len, num_qk_heads, head_dim), dtype=dtype, device=device
    )
    k = F.normalize(
        torch.randn(
            batch_size,
            seq_len,
            num_qk_heads,
            head_dim,
            dtype=torch.float32,
            device=device,
        ),
        p=2,
        dim=-1,
    ).to(dtype)
    v = torch.randn(
        (batch_size, seq_len, num_v_heads, head_dim), dtype=dtype, device=device
    )
    g = F.logsigmoid(
        torch.rand(
            1, seq_len * batch_size, num_v_heads, dtype=torch.float32, device=device
        )
    )
    beta = torch.rand(
        1, seq_len * batch_size, num_v_heads, dtype=torch.float32, device=device
    ).sigmoid()

    if use_initial_state:
        h0 = torch.randn(
            (batch_size, num_v_heads, head_dim, head_dim),
            dtype=torch.float32,
            device=device,
        )
    else:
        h0 = None

    output_final_state = True

    o = torch.zeros_like(v)
    state_output = torch.randn(
        (batch_size, num_v_heads, head_dim, head_dim),
        dtype=torch.float32,
        device=device,
    )

    fn_gdn = lambda: chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        None,
        h0,
        output_final_state,
        None,
        False,
        o,
        state_output,
    )

    kernel_times_ms = bench_gpu_time(
        fn_gdn,
        enable_cupti=True,
        dry_run_iters=warmup_iters,
        repeat_iters=benchmark_iters,
    )

    avg_latency_ms = np.average(kernel_times_ms)

    # FLA baseline
    fla_latency_ms = None
    if num_qk_heads == num_v_heads:
        fn_fla = lambda: fla_base(
            q,
            k,
            v,
            g,
            beta,
            None,
            initial_state=h0,
            output_final_state=output_final_state,
        )

        fla_times_ms = bench_gpu_time(
            fn_fla,
            enable_cupti=True,
            dry_run_iters=warmup_iters,
            repeat_iters=benchmark_iters,
        )
        fla_latency_ms = np.average(fla_times_ms)

    result = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_qk_heads": num_qk_heads,
        "num_v_heads": num_v_heads,
        "head_dim": head_dim,
        "dtype": str(dtype).split(".")[-1],
        "gdn_ms": avg_latency_ms,
    }

    if fla_latency_ms is not None:
        result["fla_ms"] = fla_latency_ms
        result["speedup"] = (
            fla_latency_ms / avg_latency_ms if avg_latency_ms > 0 else float("nan")
        )

    return result


def benchmark_gdn_varlen(
    cu_seqlens: List[int],
    num_qk_heads: int,
    num_v_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
) -> dict:
    """
    Benchmark GDN with variable-length sequences.

    Returns:
        Dictionary with benchmark results.
    """
    device = "cuda"
    torch.cuda.reset_peak_memory_stats()

    total_len = cu_seqlens[-1]
    num_seqs = len(cu_seqlens) - 1
    cu_seqlens_tensor = torch.tensor(cu_seqlens, device=device)

    # Create input tensors
    q = torch.randn((1, total_len, num_qk_heads, head_dim), dtype=dtype, device=device)
    k = F.normalize(
        torch.randn(
            1, total_len, num_qk_heads, head_dim, dtype=torch.float32, device=device
        ),
        p=2,
        dim=-1,
    ).to(dtype)
    v = torch.randn((1, total_len, num_v_heads, head_dim), dtype=dtype, device=device)
    g = F.logsigmoid(
        torch.rand(1, total_len, num_v_heads, dtype=torch.float32, device=device)
    )
    beta = torch.rand(
        1, total_len, num_v_heads, dtype=torch.float32, device=device
    ).sigmoid()
    h0 = torch.randn(
        (num_seqs, num_v_heads, head_dim, head_dim), dtype=torch.float32, device=device
    )

    o = torch.zeros_like(v)
    state_output = torch.zeros_like(h0, dtype=torch.float32)

    fn_gdn = lambda: chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        None,
        h0,
        True,
        cu_seqlens_tensor,
        False,
        o,
        state_output,
    )

    kernel_times_ms = bench_gpu_time(
        fn_gdn,
        enable_cupti=True,
        dry_run_iters=warmup_iters,
        repeat_iters=benchmark_iters,
    )

    avg_latency_ms = np.average(kernel_times_ms)

    return {
        "num_seqs": num_seqs,
        "total_len": total_len,
        "avg_seq_len": total_len // num_seqs,
        "num_qk_heads": num_qk_heads,
        "num_v_heads": num_v_heads,
        "head_dim": head_dim,
        "dtype": str(dtype).split(".")[-1],
        "gdn_ms": avg_latency_ms,
    }


def benchmark_sweep(
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Run a sweep of benchmark configurations for both fixlen and varlen."""
    head_dim = 128

    fixlen_configs = [
        # (batch_size, seq_len, num_qk_heads, num_v_heads)
        # Typical use cases:
        (1, 512, 96, 96),
        (1, 1024, 96, 96),
        (1, 4096, 96, 96),
        (1, 8192, 96, 96),
        (9, 512, 32, 32),
        (9, 1024, 32, 32),
        (9, 4096, 32, 32),
        (9, 8192, 32, 32),
        (33, 512, 32, 32),
        (33, 1024, 32, 32),
        (33, 4096, 32, 32),
        (33, 8192, 32, 32),
        # Full SM utilization:
        (1, 512, 148, 148),
        (1, 1024, 148, 148),
        (1, 4096, 148, 148),
        (1, 8192, 148, 148),
    ]

    varlen_configs = [
        # (cu_seqlens, num_qk_heads, num_v_heads) — derived from fixlen_configs
        ([sl * i for i in range(bs + 1)], nqk, nv)
        for bs, sl, nqk, nv in fixlen_configs
    ]

    print("\n" + "#" * 80)
    print(" BENCHMARK SWEEP")
    print("#" * 80)

    fixlen_results = []
    for i, (bs, sl, nqk, nv) in enumerate(fixlen_configs):
        label = f"[{i + 1}/{len(fixlen_configs)}] bs={bs}, sl={sl}, nqk={nqk}, nv={nv}"
        print(f"  Running fixlen  {label} ...", end="", flush=True)
        try:
            result = benchmark_gdn_fixlen(
                batch_size=bs,
                seq_len=sl,
                num_qk_heads=nqk,
                num_v_heads=nv,
                head_dim=head_dim,
                dtype=dtype,
                warmup_iters=warmup_iters,
                benchmark_iters=benchmark_iters,
            )
            fixlen_results.append(result)
            msg = f"  GDN: {result['gdn_ms']:.3f} ms"
            if "fla_ms" in result:
                msg += f"  FLA: {result['fla_ms']:.3f} ms  ({result['speedup']:.2f}x)"
            print(msg)
        except Exception as e:
            print(f"  FAILED: {e}")
            torch.cuda.empty_cache()

    varlen_results = []
    for i, (cu_seqlens, nqk, nv) in enumerate(varlen_configs):
        num_seqs = len(cu_seqlens) - 1
        total_len = cu_seqlens[-1]
        label = f"[{i + 1}/{len(varlen_configs)}] seqs={num_seqs}, total={total_len}, nqk={nqk}, nv={nv}"
        print(f"  Running varlen  {label} ...", end="", flush=True)
        try:
            result = benchmark_gdn_varlen(
                cu_seqlens=cu_seqlens,
                num_qk_heads=nqk,
                num_v_heads=nv,
                head_dim=head_dim,
                dtype=dtype,
                warmup_iters=warmup_iters,
                benchmark_iters=benchmark_iters,
            )
            varlen_results.append(result)
            print(f"  GDN: {result['gdn_ms']:.3f} ms")
        except Exception as e:
            print(f"  FAILED: {e}")
            torch.cuda.empty_cache()

    print_results_table(fixlen_results, "Fixed-Length Sweep Results")
    print_results_table(varlen_results, "Variable-Length Sweep Results")


def print_results_table(results: List[dict], title: str = "Benchmark Results"):
    """Print benchmark results in a formatted table."""
    if not results:
        return

    print(f"\n{'=' * 80}")
    print(f" {title}")
    print(f"{'=' * 80}")

    # Get all keys from first result
    keys = list(results[0].keys())

    # Calculate column widths
    widths = {}
    for key in keys:
        max_len = len(key)
        for r in results:
            val = r[key]
            if isinstance(val, float):
                val_str = f"{val:.4f}" if val < 100 else f"{val:.2f}"
            else:
                val_str = str(val)
            max_len = max(max_len, len(val_str))
        widths[key] = max_len + 2

    # Print header
    header = " | ".join(f"{key:^{widths[key]}}" for key in keys)
    print(header)
    print("-" * len(header))

    # Print rows
    for r in results:
        row_vals = []
        for key in keys:
            val = r[key]
            if isinstance(val, float):
                val_str = f"{val:.4f}" if val < 100 else f"{val:.2f}"
            else:
                val_str = str(val)
            row_vals.append(f"{val_str:^{widths[key]}}")
        print(" | ".join(row_vals))

    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="GDN Performance Benchmark")
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--seq-len", "-t", type=int, default=4096, help="Sequence length"
    )
    parser.add_argument(
        "--num-qk-heads",
        "-nqk",
        type=int,
        default=32,
        help="Number of qk attention heads",
    )
    parser.add_argument(
        "--num-v-heads", "-nv", type=int, default=32, help="Number of v attention heads"
    )
    parser.add_argument(
        "--head-dim", "-d", type=int, default=128, help="Head dimension"
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument(
        "--varlen", action="store_true", help="Run variable-length benchmark"
    )
    parser.add_argument(
        "--cu-seqlens",
        type=int,
        nargs="+",
        default=None,
        help="Cumulative sequence lengths for varlen mode (e.g. --cu-seqlens 0 512 1024 2048)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a sweep of many test cases (fixlen + varlen)",
    )

    args = parser.parse_args()

    device = torch.device("cuda")
    if not (is_sm100a_supported(device) or is_sm110a_supported(device)):
        print("Error: bench_blackwell_gdn_prefill.py requires an SM100A/SM110A GPU.")
        sys.exit(1)

    if args.head_dim != 128:
        print(f"Error: head_dim must be 128, got {args.head_dim}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(" GDN (Gated Delta Network) Linear Attention Benchmark")
    print(" GPU:", torch.cuda.get_device_name(0))
    print("=" * 60)

    if args.sweep:
        benchmark_sweep(
            warmup_iters=args.warmup,
            benchmark_iters=args.iters,
        )
        return

    if args.varlen:
        if args.cu_seqlens is not None:
            cu_seqlens = args.cu_seqlens
            if cu_seqlens[0] != 0:
                print(f"Error: cu_seqlens must start with 0, got {cu_seqlens[0]}")
                sys.exit(1)
            if len(cu_seqlens) < 2:
                print("Error: cu_seqlens must have at least 2 elements (e.g. 0 1024)")
                sys.exit(1)
        else:
            cu_seqlens = [args.seq_len * i for i in range(args.batch_size + 1)]
        print("\nRunning variable-length benchmark:")
        print(f"  cu_seqlens: {cu_seqlens}")
        print(f"  num_qk_heads: {args.num_qk_heads}")
        print(f"  num_v_heads: {args.num_v_heads}")
        print(f"  head_dim: {args.head_dim}")

        result = benchmark_gdn_varlen(
            cu_seqlens=cu_seqlens,
            num_qk_heads=args.num_qk_heads,
            num_v_heads=args.num_v_heads,
            head_dim=args.head_dim,
            warmup_iters=args.warmup,
            benchmark_iters=args.iters,
        )
        print_results_table([result], "Variable-Length Benchmark")
    else:
        # Fixed-length benchmark
        print("\nRunning fixed-length benchmark:")
        print(f"  batch_size: {args.batch_size}")
        print(f"  seq_len: {args.seq_len}")
        print(f"  num_qk_heads: {args.num_qk_heads}")
        print(f"  num_v_heads: {args.num_v_heads}")
        print(f"  head_dim: {args.head_dim}")
        print(f"  warmup: {args.warmup}")
        print(f"  iters: {args.iters}")

        result = benchmark_gdn_fixlen(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_qk_heads=args.num_qk_heads,
            num_v_heads=args.num_v_heads,
            head_dim=args.head_dim,
            warmup_iters=args.warmup,
            benchmark_iters=args.iters,
        )

        print_results_table([result], "Fixed-Length Benchmark")


if __name__ == "__main__":
    main()
