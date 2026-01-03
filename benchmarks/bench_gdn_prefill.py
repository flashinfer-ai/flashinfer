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

from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.testing.utils import bench_gpu_time


def gdn_flops(
    total_seq_len: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    num_seqs: int,
) -> int:
    """
    Calculate FLOPs for Gated Delta Rule (GDN) attention.

    Delta Rule formula:
        state_t = alpha_t * state_{t-1} + beta_t * (k_t @ v_t^T)
        output_t = q_t @ state_t

    Matrix multiplications per token per head:
    1. k @ v^T (outer product): 2 * d^2 FLOPs
    2. q @ state: 2 * d^2 FLOPs

    Note: alpha/beta gating are element-wise scalar multiplications,
    not counted in TFLOPS.
    """
    num_o_heads = max(num_q_heads, num_v_heads)

    # k @ v^T (outer product): 2 * d^2 per token per head
    outer_product_flops = 2 * total_seq_len * num_o_heads * head_size * head_size

    # q @ state: 2 * d^2 per token per head
    output_flops = 2 * total_seq_len * num_o_heads * head_size * head_size

    total_flops = outer_product_flops + output_flops
    return total_flops


def gdn_bytes(
    total_seq_len: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    num_seqs: int,
    dtype: torch.dtype,
) -> int:
    """
    Calculate memory bytes for GDN attention.

    Includes:
    - Q, K, V tensors (input)
    - Output tensor
    - State tensor (float32)
    - Alpha, Beta tensors (optional, float32)
    """
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads
    elem_size = dtype.itemsize

    # Input tensors
    q_bytes = total_seq_len * num_q_heads * head_size * elem_size
    k_bytes = total_seq_len * num_k_heads * head_size * elem_size
    v_bytes = total_seq_len * num_v_heads * head_size * elem_size

    # Output tensor
    o_bytes = total_seq_len * num_o_heads * head_size * elem_size

    # State tensor (float32)
    state_bytes = num_seqs * num_sab_heads * head_size * head_size * 4

    # Alpha and Beta (float32)
    alpha_bytes = total_seq_len * num_sab_heads * 4
    beta_bytes = total_seq_len * num_sab_heads * 4

    total_bytes = (
        q_bytes + k_bytes + v_bytes + o_bytes + state_bytes + alpha_bytes + beta_bytes
    )
    return total_bytes


def bench_gdn_prefill(
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    use_alpha: bool = True,
    use_beta: bool = True,
):
    """Benchmark GDN prefill kernel."""
    total_seq_len = batch_size * seq_len
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads

    # Create inputs
    q = torch.randn(total_seq_len, num_q_heads, head_size, dtype=dtype, device="cuda")
    k = torch.randn(total_seq_len, num_k_heads, head_size, dtype=dtype, device="cuda")
    # L2 normalize k for numerical stability
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
    v = torch.randn(total_seq_len, num_v_heads, head_size, dtype=dtype, device="cuda")

    cu_seqlens = torch.arange(
        0, batch_size * seq_len + 1, seq_len, dtype=torch.int64, device="cuda"
    )

    alpha = (
        torch.rand(total_seq_len, num_sab_heads, dtype=torch.float32, device="cuda")
        if use_alpha
        else None
    )
    beta = (
        torch.rand(total_seq_len, num_sab_heads, dtype=torch.float32, device="cuda")
        if use_beta
        else None
    )

    # Pre-allocate outputs
    output = torch.empty(
        total_seq_len, num_o_heads, head_size, dtype=dtype, device="cuda"
    )
    output_state = torch.empty(
        batch_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device="cuda",
    )

    # Warmup
    chunk_gated_delta_rule(
        q, k, v, alpha, beta, None, None, True, cu_seqlens, False, output, output_state
    )
    torch.cuda.synchronize()

    # Benchmark
    times = bench_gpu_time(
        lambda: chunk_gated_delta_rule(
            q,
            k,
            v,
            alpha,
            beta,
            None,
            None,
            True,
            cu_seqlens,
            False,
            output,
            output_state,
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
        enable_cupti=True,
    )

    median_ms = np.median(times)

    # Calculate metrics
    flops = gdn_flops(
        total_seq_len, num_q_heads, num_k_heads, num_v_heads, head_size, batch_size
    )
    bytes_accessed = gdn_bytes(
        total_seq_len,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        batch_size,
        dtype,
    )

    tflops = flops / median_ms / 1e9
    tb_per_sec = bytes_accessed / median_ms / 1e9

    # Get device info for bandwidth calculation
    props = torch.cuda.get_device_properties(0)
    props.total_memory * 2 / 1e12  # Approximate peak bandwidth

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_q_heads": num_q_heads,
        "num_k_heads": num_k_heads,
        "num_v_heads": num_v_heads,
        "head_size": head_size,
        "dtype": str(dtype).replace("torch.", ""),
        "median_ms": median_ms,
        "tflops": tflops,
        "tb_per_sec": tb_per_sec,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark GDN Prefill Kernel")
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--seq-len", type=int, nargs="+", default=[128, 256, 512, 1024])
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

    print(
        f"GDN Prefill Benchmark (heads: q={args.num_q_heads}, k={args.num_k_heads}, v={args.num_v_heads}, d={args.head_size}, dtype={args.dtype})"
    )
    print("-" * 100)
    print(f"{'batch':>6} {'seq_len':>8} {'time(ms)':>10} {'TFLOPS':>10} {'TB/s':>10}")
    print("-" * 100)

    for batch_size in args.batch_size:
        for seq_len in args.seq_len:
            result = bench_gdn_prefill(
                batch_size=batch_size,
                seq_len=seq_len,
                num_q_heads=args.num_q_heads,
                num_k_heads=args.num_k_heads,
                num_v_heads=args.num_v_heads,
                head_size=args.head_size,
                dtype=dtype,
            )
            print(
                f"{result['batch_size']:>6} {result['seq_len']:>8} "
                f"{result['median_ms']:>10.3f} {result['tflops']:>10.2f} "
                f"{result['tb_per_sec']:>10.2f}"
            )

    print("-" * 100)


if __name__ == "__main__":
    main()
