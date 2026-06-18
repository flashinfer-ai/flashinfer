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

import os
import sys
import argparse
import flashinfer
import numpy as np
import torch
from flashinfer.testing.utils import bench_gpu_time_with_cudagraph
from flashinfer.utils import get_gpu_memory_bandwidth

# Add the project root to Python path to import test helpers
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tests.test_helpers.rope_reference import RotaryEmbedding


def benchmark_config(
    config_name,
    num_tokens,
    batch_size=4,
    page_size=16,
    enable_pdl=False,
    single_run=False,
):
    """Benchmark a specific attention configuration with paged KV cache append."""
    input_dtype = torch.bfloat16
    device = "cuda"
    quant_dtype = torch.float8_e4m3fn

    # Configuration-specific parameters
    if config_name == "mla":
        # MLA: DeepSeek-style multi-latent attention
        num_qo_heads, num_kv_heads = 128, 1
        rope_dim, no_rope_dim = 64, 512
    elif config_name == "gqa":
        # GQA: Grouped-query attention (e.g., Llama-style)
        num_qo_heads, num_kv_heads = 32, 8
        rope_dim, no_rope_dim = 64, 64
    elif config_name == "mha":
        # MHA: Standard multi-head attention
        num_qo_heads, num_kv_heads = 32, 32
        rope_dim, no_rope_dim = 64, 64
    else:
        raise ValueError(f"Unknown config: {config_name}")

    head_dim = rope_dim + no_rope_dim

    # Create input tensors
    if config_name == "mla":
        # MLA: 2D K tensors (shared)
        q_rope = torch.randn(
            num_tokens, num_qo_heads, rope_dim, dtype=input_dtype, device=device
        )
        q_nope = torch.randn(
            num_tokens, num_qo_heads, no_rope_dim, dtype=input_dtype, device=device
        )
        k_rope = torch.randn(num_tokens, rope_dim, dtype=input_dtype, device=device)
        k_nope = torch.randn(num_tokens, no_rope_dim, dtype=input_dtype, device=device)
        v = None
    else:
        # GQA/MHA: 3D K/V tensors
        q_rope = torch.randn(
            num_tokens, num_qo_heads, rope_dim, dtype=input_dtype, device=device
        )
        q_nope = torch.randn(
            num_tokens, num_qo_heads, no_rope_dim, dtype=input_dtype, device=device
        )
        k_rope = torch.randn(
            num_tokens, num_kv_heads, rope_dim, dtype=input_dtype, device=device
        )
        k_nope = torch.randn(
            num_tokens, num_kv_heads, no_rope_dim, dtype=input_dtype, device=device
        )
        v = torch.randn(
            num_tokens, num_kv_heads, head_dim, dtype=input_dtype, device=device
        )

    # Create RoPE reference for cos/sin cache (ensure it covers this run)
    max_seq_len = int(num_tokens)
    rope_ref = RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=rope_dim,
        max_position_embeddings=max_seq_len,
        base=10000,
        is_neox_style=False,
        dtype=input_dtype,
        device=device,
    )
    pos_ids = torch.arange(num_tokens, device=device, dtype=torch.int32)

    # Build paged metadata (single request with all tokens)
    kv_append_length = torch.tensor(
        [num_tokens] + [0] * (batch_size - 1), dtype=torch.int32, device=device
    )
    kv_append_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(kv_append_length, dim=0),
        ]
    )
    num_pages_per_req = torch.tensor(
        [(num_tokens + page_size - 1) // page_size] + [0] * (batch_size - 1),
        dtype=torch.int32,
        device=device,
    )
    kv_page_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(num_pages_per_req, dim=0),
        ]
    )
    kv_page_indices = torch.arange(
        kv_page_indptr[-1].item(), dtype=torch.int32, device=device
    )
    kv_last_page_len = torch.tensor(
        [num_tokens % page_size if num_tokens % page_size != 0 else page_size]
        + [0] * (batch_size - 1),
        dtype=torch.int32,
        device=device,
    )

    # Get batch_indices and positions
    seq_lens = flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, page_size)
    batch_indices, positions = flashinfer.get_batch_indices_positions(
        kv_append_indptr, seq_lens, num_tokens
    )

    # Allocate caches
    max_pages = kv_page_indptr[-1].item()

    if config_name == "mla":
        ckv_cache = torch.zeros(
            max_pages, page_size, no_rope_dim, dtype=quant_dtype, device=device
        )
        kpe_cache = torch.zeros(
            max_pages, page_size, rope_dim, dtype=quant_dtype, device=device
        )
        paged_kv_cache = (ckv_cache, kpe_cache)
    else:
        # GQA/MHA: use NHD layout
        k_cache = torch.zeros(
            max_pages,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=quant_dtype,
            device=device,
        )
        v_cache = torch.zeros(
            max_pages,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=quant_dtype,
            device=device,
        )
        paged_kv_cache = (k_cache, v_cache)

    run_idx = 0

    def execute():
        if single_run:
            import torch.cuda.nvtx as nvtx

            nvtx.range_push("rope_append")
        nonlocal run_idx
        run_idx += 1

        flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
            q_rope=q_rope,
            k_rope=k_rope,
            q_nope=q_nope,
            k_nope=k_nope,
            v=v,
            cos_sin_cache=rope_ref.cos_sin_cache,
            pos_ids=pos_ids,
            paged_kv_cache=paged_kv_cache,
            kv_indices=kv_page_indices,
            kv_indptr=kv_page_indptr,
            batch_indices=batch_indices,
            positions=positions,
            page_size=page_size,
            kv_layout="NHD" if config_name != "mla" else "NHD",
            quantize_dtype=quant_dtype,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
            is_neox=False,
            enable_pdl=enable_pdl,
        )
        if single_run:
            # Ensure kernels complete inside the NVTX range for ncu filtering
            torch.cuda.synchronize()
            nvtx.range_pop()

    if single_run:
        execute()
        return None, None, None, None, None
    measurements = bench_gpu_time_with_cudagraph(execute)

    # Calculate I/O bytes
    # Inputs: q_rope, k_rope, q_nope, k_nope, v (if not MLA), cos_sin_cache, pos_ids
    io_bytes = (
        q_rope.numel() * q_rope.element_size()
        + k_rope.numel() * k_rope.element_size()
        + q_nope.numel() * q_nope.element_size()
        + k_nope.numel() * k_nope.element_size()
        + rope_ref.cos_sin_cache.numel() * rope_ref.cos_sin_cache.element_size()
        + pos_ids.numel() * pos_ids.element_size()
    )

    if v is not None:
        io_bytes += v.numel() * v.element_size()

    # Outputs: q_rope_out, q_nope_out (FP8), cache writes (FP8)
    io_bytes += (
        q_rope.numel() * torch.finfo(quant_dtype).bits // 8
        + q_nope.numel() * torch.finfo(quant_dtype).bits // 8
    )

    if config_name == "mla":
        # MLA writes to ckv_cache and kpe_cache
        io_bytes += (
            num_tokens * no_rope_dim * torch.finfo(quant_dtype).bits // 8
            + num_tokens * rope_dim * torch.finfo(quant_dtype).bits // 8
        )
    else:
        # GQA/MHA writes to k_cache and v_cache
        io_bytes += (
            num_tokens * num_kv_heads * head_dim * torch.finfo(quant_dtype).bits // 8
            + num_tokens * num_kv_heads * head_dim * torch.finfo(quant_dtype).bits // 8
        )

    # Calculate statistics
    ms = np.median(measurements)
    min_ms = np.percentile(measurements, 20)
    max_ms = np.percentile(measurements, 80)

    # Calculate bandwidth in GB/s
    bandwidth_gb_s = io_bytes / ms / 1e6

    # Calculate TFLOPs (FP operations)
    # RoPE: 6 FLOPs per dimension pair (2 muls + 1 sub for real, 2 muls + 1 add for imag)
    # For Q: num_tokens * num_qo_heads * (rope_dim/2) pairs * 6 FLOPs
    # For K: depends on architecture
    q_flops = num_tokens * num_qo_heads * (rope_dim / 2) * 6

    if config_name == "mla":
        # MLA: K is 2D (no head dimension)
        k_flops = num_tokens * (rope_dim / 2) * 6
    else:
        # GQA/MHA: K is 3D (has head dimension)
        k_flops = num_tokens * num_kv_heads * (rope_dim / 2) * 6

    total_flops = q_flops + k_flops
    tflops = (
        total_flops / ms / 1e9
    )  # TFLOPs (operations per ms = operations per second / 1e12)

    return ms, min_ms, max_ms, bandwidth_gb_s, tflops


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ncu-single", action="store_true", help="Run a single execute() for ncu"
    )
    parser.add_argument(
        "--config", type=str, default="", help="Config name: mla/gqa/mha"
    )
    parser.add_argument("--num-tokens", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--enable-pdl", type=int, default=0)
    args, unknown = parser.parse_known_args()

    if args.ncu_single:
        # Minimal single-run for ncu profiling
        cfg = args.config or "mla"
        ntok = int(args.num_tokens)
        pgsz = int(args.page_size)
        en_pdl = bool(int(args.enable_pdl))
        # Force a single execution path
        benchmark_config(cfg, ntok, page_size=pgsz, enable_pdl=en_pdl, single_run=True)
        sys.exit(0)

    # Get GPU information (for display only)
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_peak_bandwidth = get_gpu_memory_bandwidth(device)
    print(f"\nDetected GPU: {gpu_name}")
    print(f"Theoretical Peak Memory Bandwidth: {gpu_peak_bandwidth:.2f} GB/s")
    print()

    # Token counts to benchmark
    token_counts = [1, 32, 128, 384, 768, 1024, 2048, 4096, 8192]

    # Helper function to print a table for a specific configuration
    def print_config_table(config_name, config_desc):
        page_size_to_benchmark = 32
        print(f"\n{'=' * 100}")
        print(f"  {config_name.upper()}: {config_desc}")
        print(f"{'=' * 100}")

        print(
            f"{'Tokens':<10} {'Time (ms)':<12} {'BW (GB/s)':<12} {'BW% (Peak)':<14} {'TFLOPs':<12}"
        )
        print("-" * 70)
        for num_tokens in token_counts:
            ms, _, _, bw, tflops = benchmark_config(
                config_name, num_tokens, page_size=page_size_to_benchmark
            )
            bw_pct = (bw / gpu_peak_bandwidth) * 100
            print(
                f"{num_tokens:<10} {ms:<12.5f} {bw:<12.2f} {bw_pct:<14.1f} {tflops:<12.3f}"
            )

    # Print tables for each configuration
    print_config_table("mla", "128 Q heads, 1 K head, 64+512 dims (DeepSeek-style)")
    print_config_table("gqa", "32 Q heads, 8 K heads, 64+64 dims (Llama-style)")
    print_config_table("mha", "32 Q heads, 32 K heads, 64+64 dims (Standard)")

    print("\n" + "=" * 100)
    print("Configuration details:")
    print("  Page size: 32, Batch size: 4")
    print("  Token range: 1 (single decode) → 8192 (large prefill)")
    print(f"  GPU: {gpu_name}")
    print(f"  Theoretical Peak Memory Bandwidth: {gpu_peak_bandwidth:.2f} GB/s")
    print("  BW% calculated as: (achieved_bandwidth / peak_bandwidth) * 100")
    print("=" * 100)
