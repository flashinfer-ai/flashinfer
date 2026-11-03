"""
Benchmark: SM120 Sage (QK-INT8/PV-FP8) block-sparse VSA attention.

Benchmark configurations:
  b=8, h=32, head_dim=128
  seqlen: 1024, 2048, 4096, 8192, 16384, 32768, 65536
  density: 10%, 50%, 90%

Usage:
  python benchmarks/bench_vsa_sage_sm120.py
"""

import statistics
import sys

import torch

from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import bsa_attn_sm120_blk64_sage_fwd
from flashinfer.cute_dsl.sparse.bsa_utils.sage_quant_sm120 import (
    quantize_sage_qkv_sm120,
)
from flashinfer.testing import bench_gpu_time


BLOCK = 64
HEAD_DIM = 128


def _build_q2k(batch, heads, num_q_blocks, num_kv_blocks, density, device):
    capacity = num_kv_blocks
    index = torch.zeros(
        batch, heads, num_q_blocks, capacity, dtype=torch.int32, device=device
    )
    nums = torch.zeros(batch, heads, num_q_blocks, dtype=torch.int32, device=device)
    for b in range(batch):
        for h in range(heads):
            for qi in range(num_q_blocks):
                k = max(1, int(round(density * num_kv_blocks)))
                k = min(k, num_kv_blocks)
                chosen = torch.randperm(num_kv_blocks, device=device)[:k].sort().values
                index[b, h, qi, :k] = chosen.to(torch.int32)
                nums[b, h, qi] = k
    return index, nums


def run_benchmark():
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (12, 0):
        print("ERROR: SM120 GPU (compute capability 12.0) required.")
        sys.exit(1)

    torch.manual_seed(42)

    batch = 8
    num_heads = 32
    seqlens = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    densities = [0.10, 0.50, 0.90]

    col_w = [8, 9, 12, 11, 10]
    header = (
        f"{'seqlen':>{col_w[0]}}  {'density':>{col_w[1]}}  "
        f"{'active_blks':>{col_w[2]}}  {'median_ms':>{col_w[3]}}  {'tflops':>{col_w[4]}}"
    )
    sep = "-" * len(header)
    print(f"\nSM120 Sage VSA  b={batch} h={num_heads} head_dim={HEAD_DIM}")
    print(header)
    print(sep)

    for seqlen in seqlens:
        num_blocks = seqlen // BLOCK

        q_bf16 = torch.randn(
            batch, num_heads, seqlen, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        k_bf16 = torch.randn(
            batch, num_heads, seqlen, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        v_bf16 = torch.randn(
            batch, num_heads, seqlen, HEAD_DIM, dtype=torch.bfloat16, device=device
        )

        q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale = quantize_sage_qkv_sm120(
            q_bf16, k_bf16, v_bf16
        )

        for density in densities:
            q2k_index, q2k_nums = _build_q2k(
                batch, num_heads, num_blocks, num_blocks, density, device
            )
            active_blocks = int(q2k_nums.sum().item())

            # warm-up
            bsa_attn_sm120_blk64_sage_fwd(
                q_int8,
                k_int8,
                v_fp8,
                q_scale,
                k_scale,
                v_scale,
                q2k_index,
                block_sparse_num=int(q2k_nums.max().item()),
                q2k_block_nums=q2k_nums,
                backend="cute_dsl",
            )
            torch.cuda.synchronize()

            times = bench_gpu_time(
                lambda: bsa_attn_sm120_blk64_sage_fwd(
                    q_int8,
                    k_int8,
                    v_fp8,
                    q_scale,
                    k_scale,
                    v_scale,
                    q2k_index,
                    block_sparse_num=int(q2k_nums.max().item()),
                    q2k_block_nums=q2k_nums,
                    backend="cute_dsl",
                ),
                repeat_time_ms=500,
            )
            ms = statistics.median(times)

            # FLOPs: active_blocks includes batch*heads; each (q_tile, k_tile) pair
            # does two GEMMs: QK (2*BLOCK*HEAD_DIM*BLOCK) + PV (2*BLOCK*BLOCK*HEAD_DIM)
            # = 4*BLOCK^2*HEAD_DIM per active block.
            flops = 4 * active_blocks * BLOCK * BLOCK * HEAD_DIM
            tflops = flops / (ms * 1e-3) / 1e12
            actual_density = active_blocks / (
                num_blocks * num_blocks * batch * num_heads
            )

            print(
                f"{seqlen:>{col_w[0]}}  {actual_density:>{col_w[1]}.3f}  "
                f"{active_blocks:>{col_w[2]}}  {ms:>{col_w[3]}.3f}  {tflops:>{col_w[4]}.2f}"
            )

        print(sep)


if __name__ == "__main__":
    run_benchmark()
