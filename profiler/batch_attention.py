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


def profile_persistent_batch_attention(
    kv_lens,
    qo_lens,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    layout,
    test_dtype,
    causal,
    profiler_buffer_size,
    device="cuda",
):
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32)
    q_lens = torch.tensor(qo_lens, dtype=torch.int32)

    seq_lens_blocks = torch.ceil(seq_lens / page_size).int()

    q_indptr = torch.cat([torch.tensor([0]), torch.cumsum(q_lens, 0)], dim=0).int()
    kv_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0
    ).int()

    num_blocks = kv_indptr[-1].item()

    q = torch.rand(
        q_indptr[-1].item(), num_qo_heads, head_dim, device=device, dtype=test_dtype
    )
    if layout == "NHD":
        kv_data = torch.randn(
            num_blocks,
            2,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=test_dtype,
            device=device,
        )
    elif layout == "HND":
        kv_data = torch.randn(
            num_blocks,
            2,
            num_kv_heads,
            page_size,
            head_dim,
            dtype=test_dtype,
            device=device,
        )

    wrapper = flashinfer.BatchAttention(kv_layout=layout)
    wrapper.plan(
        q_indptr.to(device),
        kv_indptr.to(device),
        torch.arange(num_blocks).int().to(device),
        seq_lens.to(device),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        page_size,
        causal=causal,
        q_data_type=test_dtype,
        kv_data_type=test_dtype,
        use_profiler=True,
    )

    profiler_buffer = torch.zeros(
        (profiler_buffer_size,), dtype=torch.uint64, device=device
    )

    # warmup
    wrapper.run(q, kv_data, profiler_buffer=profiler_buffer)
    profiler_buffer.zero_()

    wrapper.run(q, kv_data, profiler_buffer=profiler_buffer)

    trace_name = f"batch_attention.perfetto-trace"
    events = ["prefill", "decode", "reduction"]
    export_to_perfetto_trace(profiler_buffer, events, trace_name)

    print(f"Profile trace exported to {trace_name}")


def persistent_batch_attention(
    kv_lens,
    qo_lens,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    layout,
    test_dtype,
    causal,
    device="cuda",
):
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32)
    q_lens = torch.tensor(qo_lens, dtype=torch.int32)

    seq_lens_blocks = torch.ceil(seq_lens / page_size).int()

    q_indptr = torch.cat([torch.tensor([0]), torch.cumsum(q_lens, 0)], dim=0).int()
    kv_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0
    ).int()

    num_blocks = kv_indptr[-1].item()

    q = torch.rand(
        q_indptr[-1].item(), num_qo_heads, head_dim, device=device, dtype=test_dtype
    )
    if layout == "NHD":
        kv_data = torch.randn(
            num_blocks,
            2,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=test_dtype,
            device=device,
        )
    elif layout == "HND":
        kv_data = torch.randn(
            num_blocks,
            2,
            num_kv_heads,
            page_size,
            head_dim,
            dtype=test_dtype,
            device=device,
        )

    wrapper = flashinfer.BatchAttention(kv_layout=layout)
    wrapper.plan(
        q_indptr.to(device),
        kv_indptr.to(device),
        torch.arange(num_blocks).int().to(device),
        seq_lens.to(device),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        page_size,
        causal=causal,
        q_data_type=test_dtype,
        kv_data_type=test_dtype,
    )
    wrapper.run(q, kv_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler-buffer-size", type=int, default=1048576)
    parser.add_argument("--use-profiler", action="store_true")
    args = parser.parse_args()

    seq_len_config = [(600, 1)] * 122 + [(10000, 17)] * 8

    kv_lens = [p[0] for p in seq_len_config]
    qo_lens = [p[1] for p in seq_len_config]

    page_size = 1
    num_kv_heads = 4
    num_qo_heads = 28
    head_dim = 128
    layout = "NHD"
    test_dtype = torch.bfloat16
    causal = True

    if args.use_profiler:
        profile_persistent_batch_attention(
            kv_lens=kv_lens,
            qo_lens=qo_lens,
            profiler_buffer_size=args.profiler_buffer_size,
            page_size=page_size,
            num_kv_heads=num_kv_heads,
            num_qo_heads=num_qo_heads,
            head_dim=head_dim,
            layout=layout,
            test_dtype=test_dtype,
            causal=causal,
        )
    else:
        persistent_batch_attention(
            kv_lens=kv_lens,
            qo_lens=qo_lens,
            page_size=page_size,
            num_kv_heads=num_kv_heads,
            num_qo_heads=num_qo_heads,
            head_dim=head_dim,
            layout=layout,
            test_dtype=test_dtype,
            causal=causal,
        )
