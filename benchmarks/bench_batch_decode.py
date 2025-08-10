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

page_block_size = 16
num_kv_heads = 4
num_qo_heads = 32
head_dim = 128


def bench_batch_decode(
    batch_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_block_size,
    q_dtype,
    kv_dtype,
):
    np.random.seed(42)
    seq_lens = torch.full((batch_size,), seq_len)
    seq_lens_blocks = torch.ceil(seq_lens / page_block_size).int()
    kv_indptr = torch.cat([torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0)
    kv_indptr = kv_indptr.int()
    last_page_len = seq_lens - (seq_lens_blocks - 1) * page_block_size
    last_page_len = last_page_len.int()
    num_blocks = kv_indptr[-1].item()

    q = torch.rand(batch_size, num_qo_heads, head_dim, dtype=q_dtype, device="cuda:0")
    kv_data = torch.randn(
        num_blocks, 2, page_block_size, num_kv_heads, head_dim, device="cuda:0"
    ).to(kv_dtype)
    workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"
    )
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD", use_tensor_cores=True
    )
    wrapper.plan(
        kv_indptr.to(0),
        torch.arange(num_blocks).int().to(0),
        last_page_len.to(0),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_block_size,
        data_type=kv_dtype,
        q_data_type=q_dtype,
    )

    measurements = bench_gpu_time(lambda: wrapper.run(q, kv_data))
    ms = np.median(measurements)

    io = q.numel() * q.element_size() + kv_data.numel() * kv_data.element_size()
    print(
        f"batch_size={batch_size}, seq_len={seq_len}, num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}, page_block_size={page_block_size}, q_dtype={q_dtype}, kv_dtype={kv_dtype}"
    )
    print(f"execution time: {ms}ms")
    print(f"memory bandwidth: {io / ms / 1024 / 1024:.2f} GB/s")


if __name__ == "__main__":
    for q_dtype in [torch.bfloat16]:
        for kv_dtype in [torch.bfloat16, torch.float8_e4m3fn]:
            for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
                for seq_len in [512, 1024, 2048, 4096, 8192, 16384]:
                    bench_batch_decode(
                        batch_size,
                        seq_len,
                        num_qo_heads,
                        num_kv_heads,
                        head_dim,
                        page_block_size,
                        q_dtype,
                        kv_dtype,
                    )
