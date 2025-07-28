"""
Copyright (c) 2024 by FlashInfer team.

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

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


def bench_variable_block_sparse_attention(
    num_qo_heads,
    num_kv_heads,
    head_dim,
    seq_len,
    num_blocks_row,
    num_blocks_col,
    block_density,
):
    if num_qo_heads % num_kv_heads != 0:
        return
    if seq_len // num_blocks_row < 1:
        return
    if seq_len // num_blocks_col < 1:
        return

    # synthesize uniform block sz
    block_row_sz = torch.ones(num_blocks_row, dtype=torch.int32) * (
        seq_len // num_blocks_row
    )
    block_row_sz[-1] = seq_len - (seq_len // num_blocks_row) * (num_blocks_row - 1)
    block_row_sz = block_row_sz.unsqueeze(0).repeat(num_kv_heads, 1)

    block_col_sz = torch.ones(num_blocks_col, dtype=torch.int32) * (
        seq_len // num_blocks_col
    )
    block_col_sz[-1] = seq_len - (seq_len // num_blocks_col) * (num_blocks_col - 1)
    block_col_sz = block_col_sz.unsqueeze(0).repeat(num_kv_heads, 1)

    block_mask_map = (
        torch.rand(num_kv_heads, num_blocks_row, num_blocks_col) < block_density
    )

    q = torch.randn(num_qo_heads, seq_len, head_dim, dtype=torch.half, device="cuda")
    k = torch.randn(num_kv_heads, seq_len, head_dim, dtype=torch.half, device="cuda")
    v = torch.randn(num_kv_heads, seq_len, head_dim, dtype=torch.half, device="cuda")

    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"
    )
    sparse_wrapper_fa2 = flashinfer.sparse.VariableBlockSparseAttentionWrapper(
        float_workspace_buffer, backend="fa2"
    )
    sparse_wrapper_fa3 = flashinfer.sparse.VariableBlockSparseAttentionWrapper(
        float_workspace_buffer, backend="fa3"
    )

    sparse_wrapper_fa2.plan(
        block_mask_map=block_mask_map,
        block_row_sz=block_row_sz,
        block_col_sz=block_col_sz,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_data_type=torch.half,
    )
    sparse_wrapper_fa3.plan(
        block_mask_map=block_mask_map,
        block_row_sz=block_row_sz,
        block_col_sz=block_col_sz,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_data_type=torch.half,
    )

    # Benchmark sparse attention with FA2
    measurements_fa2 = bench_gpu_time(
        lambda: sparse_wrapper_fa2.run(q, k, v),
        dry_runs=100,
        num_iters=1000,
        l2_flush=True,
        l2_flush_size_mb=256,
        l2_flush_device=torch.device("cuda:0"),
    )
    sparse_ms_fa2 = np.mean(measurements_fa2)

    # Benchmark sparse attention with FA3
    measurements_fa3 = bench_gpu_time(
        lambda: sparse_wrapper_fa3.run(q, k, v),
        dry_runs=100,
        num_iters=1000,
        l2_flush=True,
        l2_flush_size_mb=256,
        l2_flush_device=torch.device("cuda:0"),
    )
    sparse_ms_fa3 = np.mean(measurements_fa3)

    q = torch.randn(seq_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda")
    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")
    v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")

    dense_sm80_ms, dense_sm90_ms = (
        np.median(
            bench_gpu_time(
                lambda: flashinfer.single_prefill_with_kv_cache_return_lse(
                    q, k, v, causal=False, backend=backend
                ),
                dry_runs=100,
                num_iters=1000,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=torch.device("cuda:0"),
            )
        )
        for backend in ["fa2", "fa3"]
    )

    def flops(ms):
        return seq_len * seq_len * num_qo_heads * head_dim * 4 / ms / 1e9

    print(
        f"bench_variable_block_sparse_attention (num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}, seq_len={seq_len}, num_blocks_row={num_blocks_row}, num_blocks_col={num_blocks_col}, block_density={block_density}), sparse fa2-template: {flops(sparse_ms_fa2):.3f} TFLOPs/s, sparse fa3-template: {flops(sparse_ms_fa3):.3f} TFLOPs/s, dense fa2-template: {flops(dense_sm80_ms):.3f} TFLOPs/s, dense fa3-template: {flops(dense_sm90_ms):.3f} TFLOPs/s"
    )


if __name__ == "__main__":
    for num_qo_heads in [32]:
        for num_kv_heads in [32]:
            for head_dim in [128]:
                for seq_len in [8192, 16384, 32768]:
                    for num_blocks_row in [20]:
                        for num_blocks_col in [50]:
                            for block_density in [0.1, 0.3, 0.5, 0.7, 0.9]:
                                bench_variable_block_sparse_attention(
                                    num_qo_heads,
                                    num_kv_heads,
                                    head_dim,
                                    seq_len,
                                    num_blocks_row,
                                    num_blocks_col,
                                    block_density,
                                )
