#!/usr/bin/env python3
"""Benchmark host overhead of cute_dsl_mla_decode.

Measures Python-side overhead by timing many iterations without GPU sync
between them — the GPU queue never drains so we're measuring purely
host-side work (tensor reshaping, TVM-FFI dispatch, etc.).
"""

import time

import torch

from flashinfer.cute_dsl.mla_decode import cute_dsl_mla_decode


def bench_host_overhead(
    batch_size: int = 4,
    seq_len_k: int = 2048,
    page_size: int = 128,
    num_iters: int = 1000,
    warmup_iters: int = 50,
):
    device = torch.device("cuda")
    num_heads = 128
    latent_dim = 512
    rope_dim = 64
    q_len = 1
    D_qk = latent_dim + rope_dim
    softmax_scale = 1.0 / (latent_dim**0.5)
    output_scale = 1.0

    query = torch.randn(batch_size, q_len, num_heads, D_qk, dtype=torch.float16, device=device)
    num_pages_per_batch = (seq_len_k + page_size - 1) // page_size
    total_pages = num_pages_per_batch * batch_size + 10
    kv_cache = torch.randn(total_pages, page_size, D_qk, dtype=torch.float16, device=device)

    block_tables = torch.zeros(batch_size, num_pages_per_batch, dtype=torch.int32, device=device)
    for b in range(batch_size):
        for p in range(num_pages_per_batch):
            block_tables[b, p] = b * num_pages_per_batch + p

    seq_lens = torch.full((batch_size,), seq_len_k, dtype=torch.int32, device=device)
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)

    # Warmup — includes compilation on first call
    print("Warming up...")
    for _ in range(warmup_iters):
        cute_dsl_mla_decode(
            query=query, kv_cache=kv_cache, workspace_buffer=workspace_buffer,
            kv_lora_rank=latent_dim, qk_rope_head_dim=rope_dim,
            block_tables=block_tables, seq_lens=seq_lens, max_seq_len=seq_len_k,
            softmax_scale=softmax_scale, output_scale=output_scale,
        )
    torch.cuda.synchronize()

    # Benchmark: no sync between iterations → measures host overhead only
    print(f"Benchmarking {num_iters} iterations (no inter-iteration sync)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        cute_dsl_mla_decode(
            query=query, kv_cache=kv_cache, workspace_buffer=workspace_buffer,
            kv_lora_rank=latent_dim, qk_rope_head_dim=rope_dim,
            block_tables=block_tables, seq_lens=seq_lens, max_seq_len=seq_len_k,
            softmax_scale=softmax_scale, output_scale=output_scale,
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_us = (t1 - t0) * 1e6
    per_call_us = total_us / num_iters
    print(f"Total: {total_us:.0f} us for {num_iters} calls")
    print(f"Per call: {per_call_us:.1f} us")

    # Also measure with line-level profiling of the key sections
    print("\n--- Profiling individual sections ---")
    profile_sections(query, kv_cache, workspace_buffer, latent_dim, rope_dim,
                     block_tables, seq_lens, seq_len_k, softmax_scale, output_scale,
                     num_iters=num_iters)

    return per_call_us


def profile_sections(query, kv_cache, workspace_buffer, kv_lora_rank, qk_rope_head_dim,
                     block_tables, seq_lens, max_seq_len, softmax_scale, output_scale,
                     num_iters=1000):
    """Profile individual sections of cute_dsl_mla_decode to find hotspots."""
    from flashinfer.cute_dsl.mla_decode import (
        _get_compiled_mla_kernel,
        _get_split_kv_and_workspace_size,
        _LATENT_DIM, _ROPE_DIM, _MMA_QK_TILER_MN, _MAX_ACTIVE_CLUSTERS,
        BlackwellMultiHeadLatentAttentionForwardFP16,
    )
    from flashinfer.cute_dsl.utils import get_num_sm
    from cutlass import Float32, Int32
    import cutlass

    B, q_len, H, D_qk = query.shape
    page_size = kv_cache.shape[1]
    is_fp8 = query.dtype == torch.float8_e4m3fn
    device = query.device

    timings = {}

    def measure(name, fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_iters):
            fn()
        torch.cuda.synchronize()
        elapsed_us = (time.perf_counter() - t0) * 1e6 / num_iters
        timings[name] = elapsed_us

    # 1. Query split + permute
    def query_reshape():
        q_nope = query[..., :kv_lora_rank]
        q_rope = query[..., kv_lora_rank:]
        q_latent_k = q_nope.permute(2, 3, 1, 0)
        q_rope_k = q_rope.permute(2, 3, 1, 0)
        return q_latent_k, q_rope_k
    measure("query_split+permute", query_reshape)

    # 2. KV cache split + permute
    def kv_reshape():
        c_latent_k = kv_cache[:, :, :kv_lora_rank].permute(1, 2, 0)
        c_rope_k = kv_cache[:, :, kv_lora_rank:].permute(1, 2, 0)
        return c_latent_k, c_rope_k
    measure("kv_split+permute", kv_reshape)

    # 3. Page table transpose
    def page_table_transpose():
        return block_tables.t().contiguous().to(torch.int32)
    measure("page_table_transpose", page_table_transpose)

    # 4. split_kv + workspace_size computation (cached)
    max_active_blocks = get_num_sm(device)
    def compute_split():
        return _get_split_kv_and_workspace_size(
            B, q_len, max_seq_len, H, max_active_blocks
        )
    measure("compute_split_kv+workspace(cached)", compute_split)

    # 5. Workspace slice
    split_kv, workspace_size = compute_split()
    def workspace_slice():
        return workspace_buffer[:max(workspace_size, 1)]
    measure("workspace_slice(no .contiguous())", workspace_slice)

    # 6. Output + LSE allocation
    out_dtype = torch.float8_e4m3fn if is_fp8 else torch.float16
    def alloc_output():
        o_k = torch.empty((B, H, q_len, _LATENT_DIM), dtype=out_dtype, device=device).permute(1, 3, 2, 0)
        lse_k = torch.empty((B, q_len, H), dtype=torch.float32, device=device).permute(2, 1, 0)
        return o_k, lse_k
    measure("alloc_output+lse", alloc_output)

    # 7. cache_seqs + block_split_kvs creation
    def create_aux_tensors():
        if seq_lens.dtype == torch.int32 and seq_lens.is_contiguous():
            cache_seqs = seq_lens
        else:
            cache_seqs = seq_lens.to(torch.int32).contiguous()
        block_split_kvs = torch.full((B,), split_kv, dtype=torch.int32, device=device)
        return cache_seqs, block_split_kvs
    measure("create_aux_tensors(optimized)", create_aux_tensors)

    # 8. _get_compiled_mla_kernel (should be cached)
    def get_kernel():
        return _get_compiled_mla_kernel(
            is_fp8=is_fp8, page_size=page_size, num_heads=H, seq_len_q=q_len,
            is_persistent=True, is_var_seq=True, is_var_split_kv=True,
        )
    measure("get_compiled_kernel(cached)", get_kernel)

    # 9. Kernel call only (prepare everything, measure just the call)
    compiled_kernel = get_kernel()
    q_latent_k = query[..., :kv_lora_rank].permute(2, 3, 1, 0)
    q_rope_k = query[..., kv_lora_rank:].permute(2, 3, 1, 0)
    c_latent_k = kv_cache[:, :, :kv_lora_rank].permute(1, 2, 0)
    c_rope_k = kv_cache[:, :, kv_lora_rank:].permute(1, 2, 0)
    page_table_k = block_tables.t().contiguous()
    o_k = torch.empty((B, H, q_len, _LATENT_DIM), dtype=out_dtype, device=device).permute(1, 3, 2, 0)
    lse_k = torch.empty((B, q_len, H), dtype=torch.float32, device=device).permute(2, 1, 0)
    ws = workspace_buffer[:max(workspace_size, 1)]
    cache_seqs = seq_lens.to(torch.int32).contiguous()
    block_split_kvs_t = torch.full((B,), split_kv, dtype=torch.int32, device=device)
    split_kv_scalar = Int32(split_kv)
    softmax_scale_scalar = Float32(softmax_scale)
    output_scale_scalar = Float32(output_scale)

    def kernel_call_pre_cached():
        compiled_kernel(
            q_latent_k, q_rope_k, c_latent_k, c_rope_k, page_table_k,
            o_k, lse_k, ws,
            split_kv_scalar, cache_seqs, block_split_kvs_t,
            softmax_scale_scalar, output_scale_scalar,
        )
    measure("kernel_call(pre-cached scalars)", kernel_call_pre_cached)

    def kernel_call_per_call():
        compiled_kernel(
            q_latent_k, q_rope_k, c_latent_k, c_rope_k, page_table_k,
            o_k, lse_k, ws,
            Int32(split_kv), cache_seqs, block_split_kvs_t,
            Float32(softmax_scale), Float32(output_scale),
        )
    measure("kernel_call(per-call scalars)", kernel_call_per_call)

    # 10. Output reshape
    def output_reshape():
        result = o_k.permute(3, 2, 0, 1).contiguous()
        if q_len == 1:
            result = result.squeeze(1)
        return result
    measure("output_reshape", output_reshape)

    # Print results
    print(f"{'Section':<35} {'us/call':>10}")
    print("-" * 47)
    total = 0.0
    for name, us in timings.items():
        print(f"  {name:<33} {us:>10.1f}")
        total += us
    print("-" * 47)
    print(f"  {'SUM':<33} {total:>10.1f}")


if __name__ == "__main__":
    bench_host_overhead()
