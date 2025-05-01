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

import torch

import flashinfer
from flashinfer.profiler import export_to_perfetto_trace


def profile_persistent_decode(batch_size, seq_len, profiler_buffer_size):
    device = "cuda"
    num_kv_heads = 4
    num_qo_heads = 28
    head_dim = 128
    page_size = 1

    kv_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)
    kv_lens_pages = torch.ceil(kv_lens.float() / page_size).int()
    q_lens = torch.ones_like(kv_lens)
    q_indptr = torch.cat(
        [torch.tensor([0], dtype=torch.int32), torch.cumsum(q_lens, 0)]
    ).int()
    kv_indptr = torch.cat(
        [torch.tensor([0], dtype=torch.int32), torch.cumsum(kv_lens_pages, 0)]
    ).int()
    kv_indices = torch.arange(kv_indptr[-1], dtype=torch.int32)

    total_q = q_indptr[-1].item()
    total_pages = kv_indptr[-1].item()

    q = torch.randn(
        total_q, num_qo_heads, head_dim, dtype=torch.bfloat16, device=device
    )

    kv_cache = torch.randn(
        total_pages,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )

    wrapper = flashinfer.BatchAttention(kv_layout="NHD")
    wrapper.plan(
        q_indptr.to(device),
        kv_indptr.to(device),
        kv_indices.to(device),
        kv_lens.to(device),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        page_size,
        use_profiler=True,
    )

    profiler_buffer = torch.zeros(
        (profiler_buffer_size,), dtype=torch.uint64, device=device
    )

    wrapper.run(q, kv_cache, profiler_buffer=profiler_buffer)
    profiler_buffer.zero_()

    wrapper.run(q, kv_cache, profiler_buffer=profiler_buffer)

    trace_name = f"persistent-{batch_size}-{seq_len}-{num_qo_heads}.perfetto-trace"
    events = ["runner1", "runner2"]
    export_to_perfetto_trace(profiler_buffer, events, trace_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--profiler-buffer-size", type=int, default=1048576)
    args = parser.parse_args()

    profile_persistent_decode(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        profiler_buffer_size=args.profiler_buffer_size,
    )
