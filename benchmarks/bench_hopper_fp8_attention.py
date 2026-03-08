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
from flashinfer.testing.utils import (
    bench_gpu_time,
    attention_tflops_per_sec_with_actual_seq_lens,
)


def per_head_symmetric_quant(x, quant_dtype):
    """Per-head symmetric quantization to FP8."""
    o_min_val, o_max_val = (
        (-448.0, 448.0) if quant_dtype == torch.float8_e4m3fn else (-57344, 57344)
    )
    x_max_val = x.abs().amax(dim=(0, 2)).to(dtype=torch.float32)
    s_out = torch.clamp(x_max_val / o_max_val, min=1e-6)
    s_out_broadcast = s_out.view(1, -1, 1)
    q_x_out = torch.clamp(x / s_out_broadcast, min=o_min_val, max=o_max_val).to(
        dtype=quant_dtype
    )
    return q_x_out, s_out


def bench_fp8_single_prefill(
    seq_len, num_heads, causal, head_dim, dtype=torch.float8_e4m3fn
):
    """Benchmark FP8 single prefill attention."""
    num_qo_heads = num_kv_heads = num_heads

    # Create FP16 tensors first, then quantize
    q_fp16 = torch.randn(
        seq_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda"
    )
    k_fp16 = torch.randn(
        seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
    )
    v_fp16 = torch.randn(
        seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
    )

    # Quantize to FP8
    q_fp8, s_q = per_head_symmetric_quant(q_fp16, dtype)
    k_fp8, s_k = per_head_symmetric_quant(k_fp16, dtype)
    v_fp8, s_v = per_head_symmetric_quant(v_fp16, dtype)

    # FP16 baseline (fa3)
    fp16_ms = np.median(
        bench_gpu_time(
            lambda: flashinfer.single_prefill_with_kv_cache_return_lse(
                q_fp16, k_fp16, v_fp16, causal=causal, backend="fa3"
            ),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
        )
    )

    # FP8 (fa3)
    fp8_ms = np.median(
        bench_gpu_time(
            lambda: flashinfer.single_prefill_with_kv_cache_return_lse(
                q_fp8,
                k_fp8,
                v_fp8,
                causal=causal,
                backend="fa3",
                scale_q=s_q,
                scale_k=s_k,
                scale_v=s_v,
            ),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
        )
    )

    def flops(ms):
        return attention_tflops_per_sec_with_actual_seq_lens(
            torch.tensor([seq_len]),
            torch.tensor([seq_len]),
            head_dim,
            head_dim,
            num_qo_heads,
            causal,
            ms,
        )

    print(
        f"bench_fp8_single_prefill (seq_len={seq_len}, num_heads={num_heads}, causal={causal}, head_dim={head_dim}), "
        f"fp16: {flops(fp16_ms):.3f} TFLOPs/s ({fp16_ms:.3f}ms), "
        f"fp8: {flops(fp8_ms):.3f} TFLOPs/s ({fp8_ms:.3f}ms), "
        f"speedup: {fp16_ms / fp8_ms:.2f}x"
    )


def bench_fp8_batch_ragged_prefill(
    batch_size, num_heads, seq_len, causal, head_dim, dtype=torch.float8_e4m3fn
):
    """Benchmark FP8 batch ragged prefill attention."""
    num_qo_heads = num_kv_heads = num_heads
    total_len = batch_size * seq_len

    # Create FP16 tensors first
    q_fp16 = torch.randn(
        total_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda"
    )
    k_fp16 = torch.randn(
        total_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
    )
    v_fp16 = torch.randn(
        total_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
    )

    # Quantize to FP8
    q_fp8, s_q = per_head_symmetric_quant(q_fp16, dtype)
    k_fp8, s_k = per_head_symmetric_quant(k_fp16, dtype)
    v_fp8, s_v = per_head_symmetric_quant(v_fp16, dtype)

    qo_indptr = torch.arange(
        0, total_len + 1, seq_len, dtype=torch.int32, device="cuda"
    )
    kv_indptr = torch.arange(
        0, total_len + 1, seq_len, dtype=torch.int32, device="cuda"
    )

    # FP16 wrapper
    fp16_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        kv_layout="NHD",
        backend="fa3",
    )
    fp16_wrapper.plan(
        qo_indptr, kv_indptr, num_qo_heads, num_kv_heads, head_dim, causal=causal
    )

    # FP8 wrapper
    fp8_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        kv_layout="NHD",
        backend="fa3",
    )
    fp8_wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=torch.half,
        causal=causal,
    )

    fp16_ms = np.median(
        bench_gpu_time(
            lambda: fp16_wrapper.run(q_fp16, k_fp16, v_fp16),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
        )
    )

    fp8_ms = np.median(
        bench_gpu_time(
            lambda: fp8_wrapper.run(q_fp8, k_fp8, v_fp8, s_q, s_k, s_v),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
        )
    )

    def flops(ms):
        return attention_tflops_per_sec_with_actual_seq_lens(
            torch.full((batch_size,), seq_len),
            torch.full((batch_size,), seq_len),
            head_dim,
            head_dim,
            num_qo_heads,
            causal,
            ms,
        )

    print(
        f"bench_fp8_batch_ragged_prefill (batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, causal={causal}, head_dim={head_dim}), "
        f"fp16: {flops(fp16_ms):.3f} TFLOPs/s ({fp16_ms:.3f}ms), "
        f"fp8: {flops(fp8_ms):.3f} TFLOPs/s ({fp8_ms:.3f}ms), "
        f"speedup: {fp16_ms / fp8_ms:.2f}x"
    )


def bench_fp8_batch_paged_prefill(
    page_size,
    batch_size,
    num_heads,
    seq_len,
    causal,
    head_dim,
    dtype=torch.float8_e4m3fn,
):
    """Benchmark FP8 batch paged prefill attention."""
    num_qo_heads = num_kv_heads = num_heads
    total_qo_len = batch_size * seq_len
    num_pages = batch_size * seq_len // page_size

    # Create FP16 tensors first
    q_fp16 = torch.randn(
        total_qo_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda"
    )
    # Paged KV cache: (num_pages, page_size, num_heads, head_dim)
    k_fp16 = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
    )
    v_fp16 = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
    )

    # Quantize to FP8
    q_fp8, s_q = per_head_symmetric_quant(q_fp16, dtype)
    # For paged KV, reshape to (total_tokens, num_heads, head_dim) for quantization
    k_flat = k_fp16.view(-1, num_kv_heads, head_dim)
    v_flat = v_fp16.view(-1, num_kv_heads, head_dim)
    k_fp8_flat, s_k = per_head_symmetric_quant(k_flat, dtype)
    v_fp8_flat, s_v = per_head_symmetric_quant(v_flat, dtype)
    k_fp8 = k_fp8_flat.view(num_pages, page_size, num_kv_heads, head_dim)
    v_fp8 = v_fp8_flat.view(num_pages, page_size, num_kv_heads, head_dim)

    qo_indptr = torch.arange(
        0, total_qo_len + 1, seq_len, dtype=torch.int32, device="cuda"
    )
    kv_indptr = torch.arange(
        0, num_pages + 1, seq_len // page_size, dtype=torch.int32, device="cuda"
    )
    kv_indices = torch.arange(0, num_pages, dtype=torch.int32, device="cuda")
    last_page_len = torch.ones(batch_size, dtype=torch.int32, device="cuda") * page_size

    # FP16 wrapper
    fp16_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        kv_layout="NHD",
        backend="fa3",
    )
    fp16_wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
    )

    # FP8 wrapper
    fp8_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        kv_layout="NHD",
        backend="fa3",
    )
    fp8_wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=torch.half,
        causal=causal,
    )

    fp16_ms = np.median(
        bench_gpu_time(
            lambda: fp16_wrapper.run(q_fp16, (k_fp16, v_fp16)),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
        )
    )

    fp8_ms = np.median(
        bench_gpu_time(
            lambda: fp8_wrapper.run(q_fp8, (k_fp8, v_fp8), s_q, s_k, s_v),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
        )
    )

    def flops(ms):
        return attention_tflops_per_sec_with_actual_seq_lens(
            torch.full((batch_size,), seq_len),
            torch.full((batch_size,), seq_len),
            head_dim,
            head_dim,
            num_qo_heads,
            causal,
            ms,
        )

    print(
        f"bench_fp8_batch_paged_prefill (page_size={page_size}, batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, causal={causal}, head_dim={head_dim}), "
        f"fp16: {flops(fp16_ms):.3f} TFLOPs/s ({fp16_ms:.3f}ms), "
        f"fp8: {flops(fp8_ms):.3f} TFLOPs/s ({fp8_ms:.3f}ms), "
        f"speedup: {fp16_ms / fp8_ms:.2f}x"
    )


if __name__ == "__main__":
    device_capability = torch.cuda.get_device_capability()
    if device_capability[0] != 9:
        print(f"Current device capability: {device_capability}.")
        print("Current benchmark targets capability (9, 0). Returning...")
        exit()

    # Skip single prefill for now due to compilation issues
    # print("=" * 80)
    # print("FP8 Single Prefill Benchmarks")
    # print("=" * 80)
    # for head_dim in [128, 256]:
    #     for seq_len in [1024, 4096, 8192]:
    #         bench_fp8_single_prefill(seq_len, 32, True, head_dim)

    print()
    print("=" * 80)
    print("FP8 Batch Ragged Prefill Benchmarks")
    print("=" * 80)
    for head_dim in [128, 256]:
        bench_fp8_batch_ragged_prefill(128, 32, 1024, True, head_dim)
        bench_fp8_batch_ragged_prefill(64, 32, 2048, True, head_dim)
        bench_fp8_batch_ragged_prefill(32, 32, 4096, True, head_dim)
        bench_fp8_batch_ragged_prefill(16, 32, 8192, True, head_dim)

    print()
    print("=" * 80)
    print("FP8 Batch Paged Prefill Benchmarks")
    print("=" * 80)
    for head_dim in [128, 256]:
        bench_fp8_batch_paged_prefill(16, 128, 32, 1024, True, head_dim)
        bench_fp8_batch_paged_prefill(16, 64, 32, 2048, True, head_dim)
        bench_fp8_batch_paged_prefill(16, 32, 32, 4096, True, head_dim)
        bench_fp8_batch_paged_prefill(16, 16, 32, 8192, True, head_dim)
