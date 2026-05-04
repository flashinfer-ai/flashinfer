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
"""

"""
Recurrent KDA (Key-Driven Attention) Benchmark

Benchmarks the recurrent KDA kernel with per-K-dimension gating.
KDA differs from GDN by having gate g[B, T, HV, K] instead of a scalar gate.

Usage:
    python benchmarks/bench_recurrent_kda.py --batch-size 1 4 16 64 128 256
    python benchmarks/bench_recurrent_kda.py --head-size 64 --batch-size 1 32 128
    python benchmarks/bench_recurrent_kda.py --seq-len 1 2 3 4 --batch-size 1 32
"""

import argparse
import numpy as np
import torch

from flashinfer.testing import bench_gpu_time

# Import the recurrent KDA kernel
try:
    from flashinfer.kda_kernels import recurrent_kda

    RECURRENT_KDA_AVAILABLE = True
except ImportError:
    RECURRENT_KDA_AVAILABLE = False


# ============================================================================
# FLOPs and Bytes Calculation
# ============================================================================


def recurrent_kda_flops(
    batch_size: int,
    num_q_heads: int,
    _num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    seq_len: int = 1,
) -> int:
    """
    Calculate FLOPs for KDA (Key-Driven Attention) decode.

    8 * K * V FLOPs per token per head:
    1. k @ state (prediction):    2 * K * V
    2. k^T @ v_new (update):      2 * K * V
    3. q @ state (output):        2 * K * V
    4. Per-K gate application:    2 * K * V  (K*V element-wise multiply + K exp() calls)

    Note: K = V = head_size for KDA. State ops are per-HV (value) head.
    """
    total_flops = 8 * seq_len * batch_size * num_v_heads * head_size * head_size
    return total_flops


def recurrent_kda_bytes(
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    seq_len: int = 1,
) -> int:
    """
    Calculate memory bytes for recurrent KDA.

    Includes:
    - Q, K, V tensors: [B, T, H, K] - dtype
    - G tensor (per-K gate): [B, T, HV, K] - dtype (extra vs GDN)
    - Beta: [B, T, HV] - dtype
    - State (read + write): [B, HV, V, K] - bf16 (2 bytes)
    - Output: [B, T, HV, V] - dtype
    """
    elem_size = torch.tensor([], dtype=dtype).element_size()
    state_dtype_bytes = 2  # BF16 state

    # Input tensors: q/k use H (query heads), v uses HV (value heads)
    q_bytes = batch_size * seq_len * num_q_heads * head_size * elem_size
    k_bytes = batch_size * seq_len * num_k_heads * head_size * elem_size
    v_bytes = batch_size * seq_len * num_v_heads * head_size * elem_size

    # Per-K gate: [B, T, HV, K]
    g_bytes = batch_size * seq_len * num_v_heads * head_size * elem_size

    # Beta: [B, T, HV]
    beta_bytes = batch_size * seq_len * num_v_heads * elem_size

    # Output: [B, T, HV, V]
    o_bytes = batch_size * seq_len * num_v_heads * head_size * elem_size

    # State: [B, HV, V, K] read + write
    state_bytes = (
        2 * batch_size * num_v_heads * head_size * head_size * state_dtype_bytes
    )

    total_bytes = (
        q_bytes + k_bytes + v_bytes + g_bytes + beta_bytes + o_bytes + state_bytes
    )
    return total_bytes


# ============================================================================
# Benchmark Function
# ============================================================================


def bench_recurrent_kda(
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    warmup_iters: int = 10,
    bench_iters: int = 100,
):
    """Benchmark recurrent KDA kernel for T=1."""
    if not RECURRENT_KDA_AVAILABLE:
        raise RuntimeError("recurrent KDA kernel is not available")

    assert seq_len == 1, f"recurrent KDA supports T=1 only, got T={seq_len}"

    # Create inputs
    T = seq_len
    q = torch.randn(batch_size, T, num_q_heads, head_size, dtype=dtype, device="cuda")
    k = torch.randn(batch_size, T, num_q_heads, head_size, dtype=dtype, device="cuda")
    v = torch.randn(batch_size, T, num_v_heads, head_size, dtype=dtype, device="cuda")

    # KDA-specific: per-K log-space gate [B, T, HV, K]
    g = torch.randn(batch_size, T, num_v_heads, head_size, dtype=dtype, device="cuda")

    # Beta: [B, T, HV] (pre-sigmoided)
    beta = torch.randn(batch_size, T, num_v_heads, dtype=dtype, device="cuda")

    # Initial state: [B, HV, V, K] (K-last layout, BF16)
    state = torch.randn(
        batch_size,
        num_v_heads,
        head_size,
        head_size,
        dtype=torch.bfloat16,
        device="cuda",
    )

    # Scale factor
    scale = 1.0 / (head_size**0.5)

    # Benchmark with bench_gpu_time (CUPTI for accurate kernel timing)
    kernel_times_ms = bench_gpu_time(
        lambda: recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=state,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        ),
        enable_cupti=True,
        dry_run_iters=warmup_iters,
        repeat_iters=bench_iters,
    )

    # Calculate metrics
    kernel_median_ms = np.median(kernel_times_ms)
    flops = recurrent_kda_flops(
        batch_size, num_q_heads, num_k_heads, num_v_heads, head_size, seq_len
    )
    bytes_accessed = recurrent_kda_bytes(
        batch_size, num_q_heads, num_k_heads, num_v_heads, head_size, dtype, seq_len
    )

    kernel_tflops = flops / kernel_median_ms / 1e9 if kernel_median_ms > 0 else 0
    kernel_tb_per_sec = (
        bytes_accessed / kernel_median_ms / 1e9 if kernel_median_ms > 0 else 0
    )

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "kernel_median_us": kernel_median_ms * 1000,
        "kernel_tflops": kernel_tflops,
        "kernel_tb_per_sec": kernel_tb_per_sec,
    }


# ============================================================================
# Runner
# ============================================================================


def run_recurrent_kda_benchmark(args, dtype):
    """Run recurrent KDA benchmark for T=1."""
    if not RECURRENT_KDA_AVAILABLE:
        print("Error: recurrent KDA kernel is not available.")
        print("Make sure flashinfer.kda_kernels.recurrent_kda is importable.")
        return

    # Filter seq_len to only valid values (T=1 only)
    valid_seq_lens = [t for t in args.seq_len if t == 1]
    if not valid_seq_lens:
        print("Error: --seq-len must include 1 (kernel supports T=1 only)")
        return

    print("\n" + "=" * 100)
    print(f"Recurrent KDA Benchmark (T={valid_seq_lens})")
    print(
        f"Config: q_heads={args.num_q_heads}, k_heads={args.num_k_heads}, "
        f"v_heads={args.num_v_heads}, head_size={args.head_size}, "
        f"dtype={args.dtype}"
    )
    print("=" * 100)
    print()
    print(f"{'batch':>6} {'T':>4} {'time(us)':>10} {'TFLOPS':>10} {'TB/s':>10}")
    print("-" * 100)

    all_results = []
    for batch_size in args.batch_size:
        for seq_len in valid_seq_lens:
            try:
                result = bench_recurrent_kda(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_q_heads=args.num_q_heads,
                    num_k_heads=args.num_k_heads,
                    num_v_heads=args.num_v_heads,
                    head_size=args.head_size,
                    dtype=dtype,
                    warmup_iters=args.warmup,
                    bench_iters=args.iters,
                )
                all_results.append(result)

                print(
                    f"{result['batch_size']:>6} {result['seq_len']:>4} "
                    f"{result['kernel_median_us']:>10.2f} "
                    f"{result['kernel_tflops']:>10.2f} "
                    f"{result['kernel_tb_per_sec']:>10.2f}"
                )
            except Exception as e:
                print(
                    f"{batch_size:>6} {seq_len:>4} {'ERROR':>10} - {type(e).__name__}: {e}"
                )

    print("-" * 100)
    print()

    # Summary by T value
    for t in valid_seq_lens:
        t_results = [r for r in all_results if r["seq_len"] == t]
        if t_results:
            avg_time = np.mean([r["kernel_median_us"] for r in t_results])
            avg_tflops = np.mean([r["kernel_tflops"] for r in t_results])
            print(
                f"T={t}: Average time={avg_time:.2f}us, Average TFLOPS={avg_tflops:.2f}"
            )


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Recurrent KDA Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/bench_recurrent_kda.py --batch-size 1 4 16 64 128 256
  python benchmarks/bench_recurrent_kda.py --head-size 64 --batch-size 1 32 128
  python benchmarks/bench_recurrent_kda.py --seq-len 1 2 3 4 --batch-size 1 32
""",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64, 128, 256],
        help="Batch sizes to benchmark",
    )
    parser.add_argument("--num-q-heads", type=int, default=16)
    parser.add_argument("--num-k-heads", type=int, default=16)
    parser.add_argument("--num-v-heads", type=int, default=32)
    parser.add_argument("--head-size", type=int, default=128, choices=[64, 128])
    parser.add_argument(
        "--dtype", type=str, choices=["float16", "bfloat16"], default="bfloat16"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        nargs="+",
        default=[1],
        help="Sequence length (T=1 only)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    args = parser.parse_args()

    # Resolve dtype
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    run_recurrent_kda_benchmark(args, dtype)


if __name__ == "__main__":
    main()
