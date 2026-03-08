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
import csv
import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import (
    bench_gpu_time,
    attention_tflops_per_sec_with_actual_seq_lens,
)


def bench_fmha_blackwell(
    batch_size,
    qkv_len,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo,
    causal,
    dtype,
    o_data_type,
):
    # if sizeof(dtype) == 1 like with torch.float8_e4m3fn,
    # create randn from half and then convert to dtype
    init_dtype = torch.half if dtype.itemsize == 1 else dtype
    q = torch.randn(
        batch_size * qkv_len, num_qo_heads, head_dim_qk, dtype=init_dtype, device="cuda"
    ).to(dtype)
    k = torch.randn(
        batch_size * qkv_len, num_kv_heads, head_dim_qk, dtype=init_dtype, device="cuda"
    ).to(dtype)
    v = torch.randn(
        batch_size * qkv_len, num_kv_heads, head_dim_vo, dtype=init_dtype, device="cuda"
    ).to(dtype)

    qo_segment_offsets = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qkv_len
    )
    kv_segment_offsets = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qkv_len
    )
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=dtype, device="cuda"),
        kv_layout="NHD",
        backend="cutlass",
    )
    # For FP8 input, output must be bfloat16
    o_data_type = torch.bfloat16 if dtype.itemsize == 1 else dtype
    wrapper.plan(
        qo_segment_offsets,
        kv_segment_offsets,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo=head_dim_vo,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=o_data_type,
    )
    _o = wrapper.run(q, k, v)
    measurements = bench_gpu_time(
        lambda: wrapper.run(q, k, v),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
    ms = np.median(measurements)

    TFLOPS = attention_tflops_per_sec_with_actual_seq_lens(
        torch.full((batch_size,), qkv_len),
        torch.full((batch_size,), qkv_len),
        head_dim_qk,
        head_dim_vo,
        num_qo_heads,
        causal,
        ms,
    )
    print(
        f"bench_fmha_blackwell (batch_size={batch_size}, qkv_len={qkv_len}, num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads}, head_dim_qk={head_dim_qk}, head_dim_vo={head_dim_vo}, causal={causal}), flops: {TFLOPS:.3f} TFLOPs/s"
    )
    return {
        "config_name": f"Blackwell-{config_name}",
        "batch_size": batch_size,
        "qkv_len": qkv_len,
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim_qk": head_dim_qk,
        "head_dim_vo": head_dim_vo,
        "causal": causal,
        "dtype": dtype,
        "time_ms": ms,
        "tflops": TFLOPS,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark FP8 attention for DeepSeek-R1"
    )
    parser.add_argument(
        "--save-results-to",
        type=str,
        default=None,
        help="Path to save benchmark results as CSV (optional)",
    )
    args = parser.parse_args()

    results = []

    # Define configurations: (batch_size, qkv_len, num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo, config_name)
    # DeepSeek-R1 uses MLA (Multi-head Latent Attention) with 128 heads
    # head_dim_qk=192 (128 nope + 64 rope), head_dim_vo=128
    configs = [
        (16, 512, 128, 128, 192, 128, "DeepSeek-R1"),
        (8, 1024, 128, 128, 192, 128, "DeepSeek-R1"),
        (4, 2048, 128, 128, 192, 128, "DeepSeek-R1"),
        (2, 4096, 128, 128, 192, 128, "DeepSeek-R1"),
        (1, 8192, 128, 128, 192, 128, "DeepSeek-R1"),
    ]

    # Run benchmarks: Causal first, then non-causal
    # For each config: bfloat16 then fp8
    for causal in [True, False]:
        print(f"\n{'=' * 80}")
        print(f"Running {'CAUSAL' if causal else 'NON-CAUSAL'} benchmarks")
        print(f"{'=' * 80}")

        for (
            batch_size,
            qkv_len,
            num_qo_heads,
            num_kv_heads,
            head_dim_qk,
            head_dim_vo,
            config_name,
        ) in configs:
            # Run bfloat16
            print(
                f"\n[{config_name}] BS={batch_size}, SeqLen={qkv_len}, Causal={causal}, BF16"
            )
            result_bf16 = bench_fmha_blackwell(
                batch_size,
                qkv_len,
                num_qo_heads,
                num_kv_heads,
                head_dim_qk,
                head_dim_vo,
                causal,
                torch.bfloat16,
                o_data_type=torch.bfloat16,
            )
            result_bf16["config_name"] = config_name
            results.append(result_bf16)
            print(
                f"  → {result_bf16['tflops']:.2f} TFLOPs/s, {result_bf16['time_ms']:.3f} ms"
            )

            # Run fp8
            print(
                f"[{config_name}] BS={batch_size}, SeqLen={qkv_len}, Causal={causal}, FP8"
            )
            result_fp8 = bench_fmha_blackwell(
                batch_size,
                qkv_len,
                num_qo_heads,
                num_kv_heads,
                head_dim_qk,
                head_dim_vo,
                causal,
                torch.float8_e4m3fn,
                o_data_type=torch.bfloat16,
            )
            result_fp8["config_name"] = config_name
            results.append(result_fp8)
            speedup = result_fp8["tflops"] / result_bf16["tflops"]
            print(
                f"  → {result_fp8['tflops']:.2f} TFLOPs/s, {result_fp8['time_ms']:.3f} ms (speedup: {speedup:.2f}x)"
            )

    # Write results to CSV if requested
    if args.save_results_to:
        fieldnames = [
            "config_name",
            "batch_size",
            "qkv_len",
            "num_qo_heads",
            "num_kv_heads",
            "head_dim_qk",
            "head_dim_vo",
            "causal",
            "dtype",
            "time_ms",
            "tflops",
        ]

        with open(args.save_results_to, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)

        print(f"\n{'=' * 80}")
        print(f"Results saved to: {args.save_results_to}")
        print(f"{'=' * 80}")
