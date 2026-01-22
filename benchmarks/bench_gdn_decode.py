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

import argparse
import numpy as np
import torch

from flashinfer.gdn_decode import (
    gated_delta_rule_decode_pretranspose,
    gated_delta_rule_decode,
    gated_delta_rule_mtp,
)
from flashinfer.testing import bench_gpu_time


def gdn_decode_flops(
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    seq_len: int = 1,
) -> int:
    """
    Calculate FLOPs for Gated Delta Rule (GDN).

    Supports both decode (seq_len=1) and MTP (seq_len>1).

    Delta Rule formula (per token):
        g = -exp(A_log) * softplus(a + dt_bias)           # Log-space decay
        beta = sigmoid(b)                                  # Update gate
        state = state * exp(g)                             # State decay
        v_new = v - k @ state                              # Prediction error
        state = state + beta * k^T @ v_new                 # State update
        output = q @ state                                 # Output projection

    Matrix multiplications per token per head:
    1. k @ state: 2 * K * V FLOPs (for each head)
    2. k^T @ v_new (outer product): 2 * K * V FLOPs
    3. q @ state: 2 * K * V FLOPs

    Total per head: 6 * K * V FLOPs
    Note: K = V = head_size for GDN
    """
    num_o_heads = max(num_q_heads, num_v_heads)

    # Per token per head: 6 * d^2 FLOPs (d = head_size)
    # Total: seq_len * batch_size * num_heads * 6 * d^2
    total_flops = 6 * seq_len * batch_size * num_o_heads * head_size * head_size
    return total_flops


def gdn_decode_bytes(
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    seq_len: int = 1,
    disable_state_update: bool = False,
) -> int:
    """
    Calculate memory bytes for GDN.

    Supports both decode (seq_len=1) and MTP (seq_len>1).

    Includes:
    - Q, K, V tensors (input): [B, T, H, K] - dtype
    - State tensor (input/output): [B, HV, K, V] - float32
    - Intermediate states (MTP only): [B, T, HV, K, V] - float32
    - GDN parameters: A_log (float32), a (dtype), dt_bias (dtype), b (dtype)
    - Output tensor: [B, T, HV, V] - dtype

    Note: When disable_state_update=True, state is only read, not written back.
    """
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads
    elem_size = dtype.itemsize

    # Input tensors: [B, T, H, K]
    q_bytes = batch_size * seq_len * num_q_heads * head_size * elem_size
    k_bytes = batch_size * seq_len * num_k_heads * head_size * elem_size
    v_bytes = batch_size * seq_len * num_v_heads * head_size * elem_size

    # Output tensor: [B, T, HV, V]
    o_bytes = batch_size * seq_len * num_o_heads * head_size * elem_size

    # State tensor (float32): [B, HV, K, V]
    # If disable_state_update=True: only read initial state
    # If disable_state_update=False: read initial + write final state
    if disable_state_update:
        # Read only (e.g., MTP verify mode)
        state_bytes = batch_size * num_sab_heads * head_size * head_size * 4
    else:
        # Read + write (e.g., normal decode)
        state_bytes = 2 * batch_size * num_sab_heads * head_size * head_size * 4

    # GDN parameters
    # A_log: [HV] - float32
    A_log_bytes = num_sab_heads * 4
    # a: [B, T, HV] - dtype
    a_bytes = batch_size * seq_len * num_sab_heads * elem_size
    # dt_bias: [HV] - dtype
    dt_bias_bytes = num_sab_heads * elem_size
    # b: [B, T, HV] - dtype
    b_bytes = batch_size * seq_len * num_sab_heads * elem_size

    # Intermediate states (float32): [B, T, HV, K, V] - only for MTP (seq_len > 1)
    # Write all T steps of intermediate states
    intermediate_bytes = 0
    if seq_len > 1:
        intermediate_bytes = (
            batch_size * seq_len * num_sab_heads * head_size * head_size * 4
        )

    total_bytes = (
        q_bytes
        + k_bytes
        + v_bytes
        + o_bytes
        + state_bytes
        + intermediate_bytes
        + A_log_bytes
        + a_bytes
        + dt_bias_bytes
        + b_bytes
    )
    return total_bytes


def bench_gdn_decode(
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    version: str = "pretranspose",
    use_alpha: bool = True,
    use_beta: bool = True,
    use_qk_l2norm: bool = True,
    warmup_iters: int = 10,
    bench_iters: int = 100,
):
    """Benchmark GDN decode kernel using bench_gpu_time with CUPTI.

    Args:
        version: 'pretranspose' or 'nontranspose'
    """
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads

    # Create inputs (T=1 for decode)
    T = 1
    q = torch.randn(batch_size, T, num_q_heads, head_size, dtype=dtype, device="cuda")
    k = torch.randn(batch_size, T, num_k_heads, head_size, dtype=dtype, device="cuda")
    v = torch.randn(batch_size, T, num_v_heads, head_size, dtype=dtype, device="cuda")

    # GDN-specific parameters
    A_log = torch.randn(num_sab_heads, dtype=torch.float32, device="cuda")
    a = (
        torch.randn(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")
        if use_alpha
        else torch.zeros(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")
    )
    dt_bias = torch.randn(num_sab_heads, dtype=dtype, device="cuda")
    b = (
        torch.randn(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")
        if use_beta
        else torch.zeros(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")
    )

    # Initial state - layout depends on version
    # Both versions use [B, HV, head_size, head_size]
    # Pretranspose interprets as [B, HV, V, K] (v-major)
    # Nontranspose interprets as [B, HV, K, V] (k-major)
    state = torch.randn(
        batch_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device="cuda",
    )

    # Pre-allocate output
    output = torch.empty(
        batch_size, T, num_o_heads, head_size, dtype=dtype, device="cuda"
    )

    # Scale factor
    scale = 1.0 / (head_size**0.5)

    # Select API function based on version
    if version == "pretranspose":
        decode_func = gated_delta_rule_decode_pretranspose
    elif version == "nontranspose":
        decode_func = gated_delta_rule_decode
    else:
        raise ValueError(f"Unknown version: {version}")

    # Benchmark with bench_gpu_time (CUPTI for accurate kernel timing)
    kernel_times_ms = bench_gpu_time(
        lambda: decode_func(
            q, k, v, state, A_log, a, dt_bias, b, scale, output, use_qk_l2norm
        ),
        enable_cupti=True,
        dry_run_iters=warmup_iters,
        repeat_iters=bench_iters,
    )

    # Calculate metrics
    kernel_median_ms = np.median(kernel_times_ms)
    flops = gdn_decode_flops(
        batch_size, num_q_heads, num_k_heads, num_v_heads, head_size
    )
    bytes_accessed = gdn_decode_bytes(
        batch_size,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        dtype,
        seq_len=1,
        disable_state_update=False,  # Decode mode: state is read + written
    )

    kernel_tflops = flops / kernel_median_ms / 1e9 if kernel_median_ms > 0 else 0
    kernel_tb_per_sec = (
        bytes_accessed / kernel_median_ms / 1e9 if kernel_median_ms > 0 else 0
    )

    return {
        "batch_size": batch_size,
        "kernel_median_us": kernel_median_ms * 1000,
        "kernel_tflops": kernel_tflops,
        "kernel_tb_per_sec": kernel_tb_per_sec,
    }


def bench_gdn_mtp(
    batch_size: int,
    seq_len: int,  # T > 1
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    use_alpha: bool = True,
    use_beta: bool = True,
    use_qk_l2norm: bool = True,
    cache_intermediate_states: bool = True,
    warmup_iters: int = 10,
    bench_iters: int = 100,
):
    """Benchmark GDN MTP kernel using bench_gpu_time with CUPTI."""
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads

    # Create inputs (T > 1 for MTP)
    T = seq_len
    q = torch.randn(batch_size, T, num_q_heads, head_size, dtype=dtype, device="cuda")
    k = torch.randn(batch_size, T, num_k_heads, head_size, dtype=dtype, device="cuda")
    v = torch.randn(batch_size, T, num_v_heads, head_size, dtype=dtype, device="cuda")

    # GDN-specific parameters
    A_log = torch.randn(num_sab_heads, dtype=torch.float32, device="cuda")
    a = (
        torch.randn(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")
        if use_alpha
        else torch.zeros(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")
    )
    dt_bias = torch.randn(num_sab_heads, dtype=dtype, device="cuda")
    b = (
        torch.randn(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")
        if use_beta
        else torch.zeros(batch_size, T, num_sab_heads, dtype=dtype, device="cuda")
    )

    # Initial state: [pool_size, HV, V, K] (K-last layout for MTP)
    pool_size = batch_size
    initial_state = torch.randn(
        pool_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device="cuda",
    )
    initial_state_indices = torch.arange(batch_size, dtype=torch.int32, device="cuda")

    # Intermediate states buffer (optional)
    if cache_intermediate_states:
        intermediate_states_buffer = torch.zeros(
            pool_size,
            T,
            num_sab_heads,
            head_size,
            head_size,
            dtype=torch.float32,
            device="cuda",
        )
    else:
        intermediate_states_buffer = None

    # Pre-allocate output
    output = torch.empty(
        batch_size, T, num_o_heads, head_size, dtype=dtype, device="cuda"
    )

    # Scale factor
    scale = 1.0 / (head_size**0.5)

    # Benchmark with bench_gpu_time (CUPTI for accurate kernel timing)
    kernel_times_ms = bench_gpu_time(
        lambda: gated_delta_rule_mtp(
            q,
            k,
            v,
            initial_state,
            initial_state_indices,
            A_log,
            a,
            dt_bias,
            b,
            scale,
            output,
            intermediate_states_buffer,
            disable_state_update=True,
            use_qk_l2norm=use_qk_l2norm,
        ),
        enable_cupti=True,
        dry_run_iters=warmup_iters,
        repeat_iters=bench_iters,
    )

    # Calculate metrics
    kernel_median_ms = np.median(kernel_times_ms)
    flops = gdn_decode_flops(
        batch_size, num_q_heads, num_k_heads, num_v_heads, head_size, seq_len
    )
    bytes_accessed = gdn_decode_bytes(
        batch_size,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        dtype,
        seq_len,
        disable_state_update=True,  # MTP mode: state is not written back
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


def main():
    parser = argparse.ArgumentParser(description="Benchmark GDN Decode Kernel")
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32, 64, 128, 256, 512],
        help="Batch sizes to benchmark (number of concurrent decode requests)",
    )
    parser.add_argument("--num-q-heads", type=int, default=16)
    parser.add_argument("--num-k-heads", type=int, default=16)
    parser.add_argument("--num-v-heads", type=int, default=32)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument(
        "--dtype", type=str, choices=["float16", "bfloat16"], default="bfloat16"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["qwen3-next", "custom"],
        default="custom",
        help="Use preset config. qwen3-next: q=k=16, v=32, d=128",
    )
    parser.add_argument(
        "--no-qk-l2norm",
        action="store_true",
        help="Disable Q/K L2 normalization",
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=["pretranspose", "nontranspose", "mtp", "all"],
        default="nontranspose",
        help="Kernel version: pretranspose (V-major state), nontranspose (K-major state), mtp (Multiple Token Processing), or all",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="Sequence lengths for MTP benchmark (T > 1)",
    )
    parser.add_argument(
        "--cache-intermediate-states",
        action="store_true",
        help="Cache intermediate states for MTP benchmark",
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

    # Apply preset configurations
    if args.preset == "qwen3-next":
        # Qwen3-Next-80B-A3B linear attention config (GVA)
        args.num_q_heads = 16
        args.num_k_heads = 16
        args.num_v_heads = 32
        args.head_size = 128

    # Check SM90 support
    device_capability = torch.cuda.get_device_capability()
    if device_capability[0] < 9:
        print(f"Current device capability: {device_capability}")
        print("GDN requires SM90 (Hopper) or later. Exiting...")
        return

    dtype = getattr(torch, args.dtype)
    use_qk_l2norm = not args.no_qk_l2norm

    # Determine which versions to benchmark
    if args.version == "all":
        versions_to_bench = ["pretranspose", "nontranspose", "mtp"]
    else:
        versions_to_bench = [args.version]

    for version in versions_to_bench:
        if version == "mtp":
            # Benchmark MTP version
            print(
                f"\nGDN MTP Benchmark "
                f"(heads: q={args.num_q_heads}, k={args.num_k_heads}, "
                f"v={args.num_v_heads}, d={args.head_size}, dtype={args.dtype}, "
                f"qk_l2norm={'ON' if use_qk_l2norm else 'OFF'})"
            )
            print("-" * 100)
            print(
                f"{'batch':>6} {'seq_len':>8} {'time(us)':>10} {'TFLOPS':>10} {'TB/s':>10}"
            )
            print("-" * 100)

            for batch_size in args.batch_size:
                for seq_len in args.seq_len:
                    result = bench_gdn_mtp(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        num_q_heads=args.num_q_heads,
                        num_k_heads=args.num_k_heads,
                        num_v_heads=args.num_v_heads,
                        head_size=args.head_size,
                        dtype=dtype,
                        use_qk_l2norm=use_qk_l2norm,
                        cache_intermediate_states=args.cache_intermediate_states,
                        warmup_iters=args.warmup,
                        bench_iters=args.iters,
                    )

                    kernel_time_us = result["kernel_median_us"]

                    print(
                        f"{result['batch_size']:>6} {result['seq_len']:>8} {kernel_time_us:>10.2f} "
                        f"{result['kernel_tflops']:>10.2f} {result['kernel_tb_per_sec']:>10.2f}"
                    )

            print("-" * 100)
            continue

        # Benchmark decode versions (pretranspose/nontranspose)
        print(
            f"\nGDN Decode Benchmark - {version.upper()} version "
            f"(heads: q={args.num_q_heads}, k={args.num_k_heads}, "
            f"v={args.num_v_heads}, d={args.head_size}, dtype={args.dtype}, "
            f"qk_l2norm={'ON' if use_qk_l2norm else 'OFF'})"
        )
        print("-" * 90)
        print(
            f"{'batch':>6} {'time(us)':>10} {'TFLOPS':>10} {'TB/s':>10} {'kernel':>15}"
        )
        print("-" * 90)

        for batch_size in args.batch_size:
            result = bench_gdn_decode(
                batch_size=batch_size,
                num_q_heads=args.num_q_heads,
                num_k_heads=args.num_k_heads,
                num_v_heads=args.num_v_heads,
                head_size=args.head_size,
                dtype=dtype,
                version=version,
                use_qk_l2norm=use_qk_l2norm,
                warmup_iters=args.warmup,
                bench_iters=args.iters,
            )

            # Determine which kernel variant was used (based on batch size threshold)
            if version == "pretranspose":
                kernel_variant = "SmallBatch" if batch_size <= 32 else "LargeBatch"
            elif version == "nontranspose":
                kernel_variant = "SmallBatch" if batch_size < 32 else "LargeBatch"

            # Time in microseconds
            kernel_time_us = result["kernel_median_us"]

            print(
                f"{result['batch_size']:>6} {kernel_time_us:>10.2f} "
                f"{result['kernel_tflops']:>10.2f} {result['kernel_tb_per_sec']:>10.2f} "
                f"{kernel_variant:>15}"
            )

        print("-" * 90)


if __name__ == "__main__":
    main()
