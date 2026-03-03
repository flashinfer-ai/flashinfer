"""FlashInfer Block Extend vs PyTorch Flex Attention 性能对比

对比三种实现:
1. BatchBlockExtendRaggedOffsetWrapper (FlashInfer, ragged KV)
2. BatchBlockExtendPagedOffsetWrapper (FlashInfer, paged KV)
3. torch.nn.attention.flex_attention + create_block_mask (PyTorch native)

Mask 规则 (Block Extend):
  q_global = q_offset + q_idx
  kv_global = kv_offset + kv_idx
  mask[q, k] = (q_global // dllm_block_size) >= (kv_global // dllm_block_size)

Flex Attention KV Cache 格式:
  Q: (B, Hq, L, E)    — dense BHSD
  K: (B, Hkv, S, E)   — dense BHSD
  V: (B, Hkv, S, Ev)  — dense BHSD
  没有 paged KV cache, 只是普通的 dense tensor
"""

import torch
import time
import math
import sys

# ============================================================
# FlashInfer imports
# ============================================================
try:
    from flashinfer import single_prefill_with_kv_cache
    from flashinfer.dllm import (
        BatchBlockExtendPagedOffsetWrapper,
        BatchBlockExtendRaggedOffsetWrapper,
    )
    HAS_FLASHINFER = True
except ImportError as e:
    HAS_FLASHINFER = False
    print(f"[WARN] flashinfer not available: {e}")
    print("       Will skip FlashInfer benchmarks")
except Exception as e:
    HAS_FLASHINFER = False
    print(f"[ERROR] flashinfer import failed with unexpected error: {e}")
    print("        Will skip FlashInfer benchmarks")

# ============================================================
# Flex Attention imports (requires PyTorch >= 2.5)
# ============================================================
try:
    from torch.nn.attention.flex_attention import (
        flex_attention,
        create_block_mask,
    )
    HAS_FLEX_ATTENTION = True
except ImportError:
    HAS_FLEX_ATTENTION = False
    print("[WARN] flex_attention not available (requires PyTorch >= 2.5)")


# ============================================================
# Reference implementation
# ============================================================
def compute_block_extend_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dllm_block_size: int,
    q_offset: int = 0,
    sm_scale: float = None,
) -> torch.Tensor:
    """Reference: single_prefill_with_kv_cache + custom_mask"""
    qo_len = q.shape[0]
    kv_len = k.shape[0]
    head_dim = q.shape[-1]
    device = q.device

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    q_pos = torch.arange(qo_len, device=device) + q_offset
    k_pos = torch.arange(kv_len, device=device)
    q_block = q_pos.unsqueeze(1) // dllm_block_size
    k_block = k_pos.unsqueeze(0) // dllm_block_size
    mask_2d = (q_block >= k_block).to(torch.uint8)

    return single_prefill_with_kv_cache(
        q, k, v, custom_mask=mask_2d, sm_scale=sm_scale,
    )


# ============================================================
# Flex Attention helper: build block_extend mask_mod
# ============================================================
def make_block_extend_mask_mod(dllm_block_size: int, q_offset: int = 0):
    """
    返回 flex_attention 使用的 mask_mod 函数

    mask_mod(b, h, q_idx, kv_idx) -> bool
    True = 允许 attend, False = 屏蔽
    """
    def block_extend_mask(b, h, q_idx, kv_idx):
        q_global = q_idx + q_offset
        q_blk = q_global // dllm_block_size
        kv_blk = kv_idx // dllm_block_size
        return q_blk >= kv_blk
    return block_extend_mask


# ============================================================
# Memory utility
# ============================================================
def get_memory_stats(device=None):
    """Get current GPU memory stats in MB."""
    if device is None:
        device = torch.device("cuda:0")
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
    reserved = torch.cuda.memory_reserved(device) / 1024**2    # MB
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
    return {
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "max_allocated_mb": max_allocated,
    }

def reset_peak_memory(device=None):
    """Reset peak memory stats."""
    if device is None:
        device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)

def measure_memory_fn(fn, warmup_iters=5):
    """Measure peak memory usage of a callable.
    
    Returns:
        dict with keys:
            - peak_allocated_mb: Peak allocated memory during execution
            - peak_reserved_mb: Peak reserved memory
            - baseline_allocated_mb: Memory before execution
    """
    # Warmup
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    # Measure baseline
    baseline = get_memory_stats()
    reset_peak_memory()
    
    # Execute and measure peak
    fn()
    torch.cuda.synchronize()
    
    peak = get_memory_stats()
    
    return {
        "baseline_allocated_mb": baseline["allocated_mb"],
        "peak_allocated_mb": peak["max_allocated_mb"],
        "peak_reserved_mb": peak["reserved_mb"],
        "memory_increase_mb": peak["max_allocated_mb"] - baseline["allocated_mb"],
    }


# ============================================================
# Benchmark utility
# ============================================================
def benchmark_fn(fn, warmup_iters=20, bench_iters=100, label=""):
    """Benchmark a callable, return average time in ms."""
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(bench_iters):
        fn()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / bench_iters * 1000
    return elapsed_ms


def benchmark_with_cuda_graph(fn, warmup_iters=20, bench_iters=100, label=""):
    """Benchmark with CUDA Graph capture, return average time in ms."""
    # warmup
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    # capture
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn()
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()

    # warmup cuda_graph
    for _ in range(warmup_iters):
        graph.replay()
    torch.cuda.synchronize()

    # bench
    start = time.perf_counter()
    for _ in range(bench_iters):
        graph.replay()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / bench_iters * 1000

    del graph
    return elapsed_ms


# ============================================================
# Main benchmark
# ============================================================
def test_flashinfer_vs_flex_attention(
    num_requests: int = 4,
    total_kv_len: int = 2048,
    qo_len: int = 256,
    dllm_block_size: int = 32,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 16,
    dtype: torch.dtype = torch.float16,
    warmup_iters: int = 20,
    bench_iters: int = 100,
    verify: bool = True,
):
    """
    Main benchmark: FlashInfer Block Extend (Ragged + Paged) vs Flex Attention

    场景: 每个 request 的 Q 长度 = qo_len, KV 长度 = total_kv_len
          q_offset = total_kv_len - qo_len  (模拟增量 prefill 的最后一步)
    """
    device = torch.device("cuda:0")
    sm_scale = 1.0 / math.sqrt(head_dim)
    q_offset = total_kv_len - qo_len

    print(f"\n{'='*80}")
    print(f"FlashInfer Block Extend vs PyTorch Flex Attention")
    print(f"{'='*80}")
    print(f"  num_requests     = {num_requests}")
    print(f"  total_kv_len     = {total_kv_len}")
    print(f"  qo_len           = {qo_len}")
    print(f"  q_offset         = {q_offset}")
    print(f"  dllm_block_size  = {dllm_block_size}")
    print(f"  num_heads        = {num_heads}")
    print(f"  num_kv_heads     = {num_kv_heads}")
    print(f"  head_dim         = {head_dim}")
    print(f"  page_size        = {page_size}")
    print(f"  dtype            = {dtype}")
    print()

    results = {}

    # ===========================================================
    # 数据准备
    # ===========================================================
    # Per-request tensors
    all_q = [torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)
             for _ in range(num_requests)]
    all_k = [torch.randn(total_kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
             for _ in range(num_requests)]
    all_v = [torch.randn(total_kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
             for _ in range(num_requests)]

    # Ragged layout: concat along token dim  (NHD format)
    q_ragged = torch.cat(all_q, dim=0)  # [B*qo_len, H, D]
    k_ragged = torch.cat(all_k, dim=0)  # [B*kv_len, Hkv, D]
    v_ragged = torch.cat(all_v, dim=0)

    # Flex Attention layout: BHSD
    # Q: (B, Hq, qo_len, D)   K/V: (B, Hkv, kv_len, D)
    q_bhsd = torch.stack(all_q, dim=0).permute(0, 2, 1, 3).contiguous()
    k_bhsd = torch.stack(all_k, dim=0).permute(0, 2, 1, 3).contiguous()
    v_bhsd = torch.stack(all_v, dim=0).permute(0, 2, 1, 3).contiguous()
    # q_bhsd: [B, Hq, qo_len, D],  k_bhsd: [B, Hkv, kv_len, D]

    # Paged KV cache preparation
    num_pages_per_req = (total_kv_len + page_size - 1) // page_size
    total_pages = num_pages_per_req * num_requests
    # kv_layout="NHD" -> paged_kv_cache: (total_pages, 2, page_size, num_kv_heads, head_dim)
    paged_kv_cache = torch.zeros(
        total_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=dtype, device=device,
    )
    # Fill pages from all_k/all_v
    paged_kv_indices_list = []
    paged_kv_last_page_lens = []
    for req_idx in range(num_requests):
        for page_idx in range(num_pages_per_req):
            global_page = req_idx * num_pages_per_req + page_idx
            start = page_idx * page_size
            end = min(start + page_size, total_kv_len)
            length = end - start
            # all_k shape: (kv_len, num_kv_heads, head_dim) — already NHD
            # page slot:   (page_size, num_kv_heads, head_dim) — NHD
            paged_kv_cache[global_page, 0, :length, :, :] = all_k[req_idx][start:end]
            paged_kv_cache[global_page, 1, :length, :, :] = all_v[req_idx][start:end]
            paged_kv_indices_list.append(global_page)
        last_page_len = total_kv_len - (num_pages_per_req - 1) * page_size
        paged_kv_last_page_lens.append(last_page_len)

    paged_kv_indices = torch.tensor(paged_kv_indices_list, dtype=torch.int32, device=device)
    paged_kv_indptr = torch.tensor(
        [i * num_pages_per_req for i in range(num_requests + 1)],
        dtype=torch.int32, device=device,
    )
    paged_kv_last_page_len = torch.tensor(paged_kv_last_page_lens, dtype=torch.int32, device=device)

    # indptr for ragged
    qo_indptr = torch.tensor(
        [i * qo_len for i in range(num_requests + 1)],
        dtype=torch.int32, device=device,
    )
    kv_indptr = torch.tensor(
        [i * total_kv_len for i in range(num_requests + 1)],
        dtype=torch.int32, device=device,
    )
    q_offsets = torch.full((num_requests,), q_offset, dtype=torch.int32, device=device)

    # ===========================================================
    # 1. Correctness Validation (single request, against reference)
    # ===========================================================
    if verify and HAS_FLASHINFER:
        print(f"{'='*60}")
        print(f"Correctness Validation (request 0)")
        print(f"{'='*60}")

        ref_out = compute_block_extend_reference(
            all_q[0], all_k[0], all_v[0],
            dllm_block_size=dllm_block_size,
            q_offset=q_offset,
            sm_scale=sm_scale,
        )

        tol = 1e-2

        # Ragged Offset
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        single_qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
        single_kv_indptr = torch.tensor([0, total_kv_len], dtype=torch.int32, device=device)
        single_q_offsets = torch.tensor([q_offset], dtype=torch.int32, device=device)

        ragged_wrapper = BatchBlockExtendRaggedOffsetWrapper(
            ws, kv_layout="NHD", dllm_block_size=dllm_block_size,
        )
        ragged_wrapper.plan(
            qo_indptr=single_qo_indptr, kv_indptr=single_kv_indptr,
            num_qo_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
            q_data_type=dtype, sm_scale=sm_scale, q_offsets=single_q_offsets,
        )
        ragged_out = ragged_wrapper.run(all_q[0], all_k[0], all_v[0])
        ragged_diff = (ragged_out - ref_out).abs().max().item()
        ragged_pass = ragged_diff < tol
        print(f"  [Ragged Offset] max_diff={ragged_diff:.6f}  {'PASS' if ragged_pass else 'FAIL'}")

        # Paged Offset
        single_paged_indptr = torch.tensor([0, num_pages_per_req], dtype=torch.int32, device=device)
        single_paged_indices = torch.arange(num_pages_per_req, dtype=torch.int32, device=device)
        single_paged_last = torch.tensor([paged_kv_last_page_lens[0]], dtype=torch.int32, device=device)

        paged_wrapper = BatchBlockExtendPagedOffsetWrapper(
            ws, kv_layout="NHD", dllm_block_size=dllm_block_size,
        )
        paged_wrapper.plan(
            qo_indptr=single_qo_indptr, paged_kv_indptr=single_paged_indptr,
            paged_kv_indices=single_paged_indices, paged_kv_last_page_len=single_paged_last,
            num_qo_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
            page_size=page_size, q_data_type=dtype, sm_scale=sm_scale,
            q_offsets=single_q_offsets,
        )
        paged_out = paged_wrapper.run(all_q[0], paged_kv_cache)
        paged_diff = (paged_out - ref_out).abs().max().item()
        paged_pass = paged_diff < tol
        print(f"  [Paged Offset]  max_diff={paged_diff:.6f}  {'PASS' if paged_pass else 'FAIL'}")

        # Flex Attention
        if HAS_FLEX_ATTENTION:
            mask_mod = make_block_extend_mask_mod(dllm_block_size, q_offset)
            block_mask = create_block_mask(
                mask_mod, B=1, H=1, Q_LEN=qo_len, KV_LEN=total_kv_len, device=device,
            )
            # single request, BHSD format
            q_single = all_q[0].unsqueeze(0).permute(0, 2, 1, 3).contiguous()
            k_single = all_k[0].unsqueeze(0).permute(0, 2, 1, 3).contiguous()
            v_single = all_v[0].unsqueeze(0).permute(0, 2, 1, 3).contiguous()

            flex_out_bhsd = flex_attention(
                q_single, k_single, v_single,
                block_mask=block_mask, scale=sm_scale,
                enable_gqa=(num_heads != num_kv_heads),
            )
            # convert back: (1, H, L, D) -> (L, H, D)
            flex_out = flex_out_bhsd.squeeze(0).permute(1, 0, 2).contiguous()
            flex_diff = (flex_out - ref_out).abs().max().item()
            flex_pass = flex_diff < tol
            print(f"  [Flex Attention] max_diff={flex_diff:.6f}  {'PASS' if flex_pass else 'FAIL'}")

        del ws, ragged_wrapper, paged_wrapper
        torch.cuda.empty_cache()
        print()

    # ===========================================================
    # 2. FlashInfer Ragged Offset Benchmark
    # ===========================================================
    if HAS_FLASHINFER:
        print(f"{'='*60}")
        print(f"[Bench] FlashInfer Ragged Offset (batch={num_requests})")
        print(f"{'='*60}")

        ws_ragged = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        ragged_bench_wrapper = BatchBlockExtendRaggedOffsetWrapper(
            ws_ragged, kv_layout="NHD", dllm_block_size=dllm_block_size,
        )
        ragged_bench_wrapper.plan(
            qo_indptr=qo_indptr, kv_indptr=kv_indptr,
            num_qo_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
            q_data_type=dtype, sm_scale=sm_scale, q_offsets=q_offsets,
        )

        ragged_out_buf = torch.empty(
            num_requests * qo_len, num_heads, head_dim, dtype=dtype, device=device,
        )

        def run_ragged():
            ragged_out_buf.copy_(
                ragged_bench_wrapper.run(q_ragged, k_ragged, v_ragged)
            )

        # no-cuda_graph benchmark
        t_ragged = benchmark_fn(run_ragged, warmup_iters, bench_iters)
        print(f"  No cuda_graph:   {t_ragged:.3f} ms")

        # cuda_graph benchmark
        t_ragged_cuda_graph = benchmark_with_cuda_graph(run_ragged, warmup_iters, bench_iters)
        print(f"  With cuda_graph: {t_ragged_cuda_graph:.3f} ms")

        # Memory measurement
        mem_ragged = measure_memory_fn(run_ragged, warmup_iters=5)
        print(f"  Memory:  Peak={mem_ragged['peak_allocated_mb']:.1f} MB, "
              f"Increase={mem_ragged['memory_increase_mb']:.1f} MB")

        results["ragged_offset"] = {
            "no_cuda_graph_ms": t_ragged, 
            "cuda_graph_ms": t_ragged_cuda_graph,
            "memory": mem_ragged,
        }
        del ws_ragged, ragged_bench_wrapper
        torch.cuda.empty_cache()
        print()

    # ===========================================================
    # 3. FlashInfer Paged Offset Benchmark
    # ===========================================================
    if HAS_FLASHINFER:
        print(f"{'='*60}")
        print(f"[Bench] FlashInfer Paged Offset (batch={num_requests})")
        print(f"{'='*60}")

        ws_paged = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        paged_bench_wrapper = BatchBlockExtendPagedOffsetWrapper(
            ws_paged, kv_layout="NHD", dllm_block_size=dllm_block_size,
        )
        paged_bench_wrapper.plan(
            qo_indptr=qo_indptr, paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            num_qo_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
            page_size=page_size, q_data_type=dtype, sm_scale=sm_scale,
            q_offsets=q_offsets,
        )

        paged_out_buf = torch.empty(
            num_requests * qo_len, num_heads, head_dim, dtype=dtype, device=device,
        )

        def run_paged():
            paged_out_buf.copy_(
                paged_bench_wrapper.run(q_ragged, paged_kv_cache)
            )

        t_paged = benchmark_fn(run_paged, warmup_iters, bench_iters)
        print(f"  No cuda_graph:   {t_paged:.3f} ms")

        t_paged_cuda_graph = benchmark_with_cuda_graph(run_paged, warmup_iters, bench_iters)
        print(f"  With cuda_graph: {t_paged_cuda_graph:.3f} ms")

        # Memory measurement
        mem_paged = measure_memory_fn(run_paged, warmup_iters=5)
        print(f"  Memory:  Peak={mem_paged['peak_allocated_mb']:.1f} MB, "
              f"Increase={mem_paged['memory_increase_mb']:.1f} MB")

        results["paged_offset"] = {
            "no_cuda_graph_ms": t_paged, 
            "cuda_graph_ms": t_paged_cuda_graph,
            "memory": mem_paged,
        }
        del ws_paged, paged_bench_wrapper
        torch.cuda.empty_cache()
        print()

    # ===========================================================
    # 4. Flex Attention Benchmark
    # ===========================================================
    if HAS_FLEX_ATTENTION:
        print(f"{'='*60}")
        print(f"[Bench] PyTorch Flex Attention (batch={num_requests})")
        print(f"{'='*60}")
        print(f"  KV format: dense BHSD  Q({num_requests},{num_heads},{qo_len},{head_dim})")
        print(f"                         K({num_requests},{num_kv_heads},{total_kv_len},{head_dim})")
        print(f"                         V({num_requests},{num_kv_heads},{total_kv_len},{head_dim})")

        # Create block_mask — all requests share the same mask pattern
        mask_mod = make_block_extend_mask_mod(dllm_block_size, q_offset)
        block_mask = create_block_mask(
            mask_mod, B=num_requests, H=1,
            Q_LEN=qo_len, KV_LEN=total_kv_len, device=device,
        )

        use_gqa = (num_heads != num_kv_heads)
        flex_out_buf = torch.empty(
            num_requests, num_heads, qo_len, head_dim, dtype=dtype, device=device,
        )

        # ---------- flex_attention (no compile) ----------
        # Skip no-compile for large sequences (materializes full QxKV score matrix → OOM)
        if qo_len * total_kv_len <= 4096 * 4096:
            def run_flex_no_compile():
                flex_out_buf.copy_(
                    flex_attention(
                        q_bhsd, k_bhsd, v_bhsd,
                        block_mask=block_mask, scale=sm_scale,
                        enable_gqa=use_gqa,
                    )
                )

            t_flex_no_compile = benchmark_fn(
                run_flex_no_compile, warmup_iters, bench_iters,
            )
            print(f"  No compile:                {t_flex_no_compile:.3f} ms")

            # Memory measurement
            mem_flex_no_compile = measure_memory_fn(run_flex_no_compile, warmup_iters=5)
            print(f"  Memory (no compile): Peak={mem_flex_no_compile['peak_allocated_mb']:.1f} MB, "
                  f"Increase={mem_flex_no_compile['memory_increase_mb']:.1f} MB")

            results["flex_no_compile"] = {
                "no_cuda_graph_ms": t_flex_no_compile,
                "memory": mem_flex_no_compile,
            }
        else:
            print(f"  No compile:                SKIPPED (seq too large, would OOM)")

        # ---------- flex_attention (compiled) ----------
        # 每次重新创建 compile 实例 + reset dynamo 缓存，避免跨档位 shape 累积 recompilation
        torch._dynamo.reset()
        _flex_attention_compiled = torch.compile(flex_attention, dynamic=False)

        def run_flex_compiled():
            flex_out_buf.copy_(
                _flex_attention_compiled(
                    q_bhsd, k_bhsd, v_bhsd,
                    block_mask=block_mask, scale=sm_scale,
                    enable_gqa=use_gqa,
                )
            )

        t_flex_compiled = benchmark_fn(
            run_flex_compiled, warmup_iters, bench_iters,
        )
        print(f"  Compiled:            {t_flex_compiled:.3f} ms")
        
        # Memory measurement
        mem_flex = measure_memory_fn(run_flex_compiled, warmup_iters=5)
        print(f"  Memory (compiled):   Peak={mem_flex['peak_allocated_mb']:.1f} MB, "
              f"Increase={mem_flex['memory_increase_mb']:.1f} MB")
        
        results["flex_compiled"] = {
            "no_cuda_graph_ms": t_flex_compiled,
            "memory": mem_flex,
        }

        # ---------- flex_attention (compiled + reduce-overhead / internal cuda_graph) ----------
        torch._dynamo.reset()
        _flex_attention_reduce_overhead = torch.compile(flex_attention, dynamic=False, mode="reduce-overhead")

        def run_flex_reduce_overhead():
            flex_out_buf.copy_(
                _flex_attention_reduce_overhead(
                    q_bhsd, k_bhsd, v_bhsd,
                    block_mask=block_mask, scale=sm_scale,
                    enable_gqa=use_gqa,
                )
            )

        t_flex_reduce = benchmark_fn(
            run_flex_reduce_overhead, warmup_iters, bench_iters,
        )
        print(f"  Compiled (reduce-overhead): {t_flex_reduce:.3f} ms")
        results["flex_reduce_overhead"] = {"no_cuda_graph_ms": t_flex_reduce}

        # ---------- flex_attention (compiled + manual CUDA Graph) ----------
        try:
            t_flex_cuda_graph = benchmark_with_cuda_graph(
                run_flex_compiled, warmup_iters, bench_iters,
            )
            print(f"  Compiled + CUDA Graph:     {t_flex_cuda_graph:.3f} ms")
            results["flex_compiled"]["cuda_graph_ms"] = t_flex_cuda_graph
        except Exception as e:
            print(f"  Compiled + CUDA Graph:     FAILED ({e})")

    # ===========================================================
    # 5. Summary
    # ===========================================================
    print(f"\n{'='*80}")
    print(f"Summary (batch={num_requests}, qo_len={qo_len}, kv_len={total_kv_len}, "
          f"block_size={dllm_block_size})")
    print(f"{'='*80}")
    print(f"  {'Method':<40} | {'No cuda_graph (ms)':>12} | {'With cuda_graph (ms)':>12} | {'Mem Incr (MB)':>14}")
    print(f"  {'-'*40}-+-{'-'*12}-+-{'-'*12}-+-{'-'*14}")

    for key, label in [
        ("ragged_offset", "FlashInfer Ragged Offset"),
        ("paged_offset", "FlashInfer Paged Offset"),
        ("flex_no_compile", "Flex Attention (no compile)"),
        ("flex_compiled", "Flex Attention (compiled)"),
        ("flex_reduce_overhead", "Flex Attention (reduce-overhead)"),
    ]:
        if key in results:
            r = results[key]
            no_cuda_graph = f"{r.get('no_cuda_graph_ms', 0):.3f}" if 'no_cuda_graph_ms' in r else "N/A"
            cuda_graph = f"{r.get('cuda_graph_ms', 0):.3f}" if 'cuda_graph_ms' in r else "N/A"
            mem = r.get('memory', {})
            mem_incr = f"{mem['memory_increase_mb']:.1f}" if mem and 'memory_increase_mb' in mem else "N/A"
            print(f"  {label:<40} | {no_cuda_graph:>12} | {cuda_graph:>12} | {mem_incr:>14}")

    # Speedup vs flex_compiled
    if "flex_compiled" in results and "ragged_offset" in results:
        flex_t = results["flex_compiled"]["no_cuda_graph_ms"]
        ragged_t = results["ragged_offset"]["no_cuda_graph_ms"]
        paged_t = results.get("paged_offset", {}).get("no_cuda_graph_ms", 0)
        print(f"\n  Speedup (vs Flex compiled, no cuda_graph):")
        print(f"    Ragged Offset: {flex_t / ragged_t:.2f}x")
        if paged_t > 0:
            print(f"    Paged Offset:  {flex_t / paged_t:.2f}x")

    if "flex_compiled" in results and "ragged_offset" in results and "cuda_graph_ms" in results["ragged_offset"]:
        ragged_cuda_graph = results["ragged_offset"]["cuda_graph_ms"]
        paged_cuda_graph = results.get("paged_offset", {}).get("cuda_graph_ms", 0)
        flex_t = results["flex_compiled"]["no_cuda_graph_ms"]
        print(f"\n  Speedup (FlashInfer cuda_graph vs Flex compiled):")
        print(f"    Ragged Offset cuda_graph: {flex_t / ragged_cuda_graph:.2f}x")
        if paged_cuda_graph > 0:
            print(f"    Paged Offset cuda_graph:  {flex_t / paged_cuda_graph:.2f}x")

    return results


# ============================================================
# Sweep across different sequence lengths
# ============================================================
def test_sweep_seq_lengths(
    num_requests: int = 4,
    dllm_block_size: int = 32,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 16,
    dtype: torch.dtype = torch.float16,
):
    """
    七档上下文长度测试:
      1K / 2K / 4K / 8K / 16K / 24K / 32K

    每个档位固定 chunk_size=256, 测试最后一个 chunk 的 per-chunk 延迟
    (即 q_offset = kv_len - 256, 模拟增量 prefill 最后一步)
    """
    chunk_size = 256
    configs = [
        # (total_kv_len, qo_len, batch, label)
        (1024,  chunk_size, num_requests, "1K"),
        (2048,  chunk_size, num_requests, "2K"),
        (4096,  chunk_size, num_requests, "4K"),
        (8192,  chunk_size, num_requests, "8K"),
        (16384, chunk_size, num_requests, "16K"),
        (24576, chunk_size, min(num_requests, 2), "24K"),
        (32768, chunk_size, 1, "32K"),
    ]

    all_results = {}
    for total_kv_len, qo_len, batch, desc in configs:
        num_chunks = total_kv_len // qo_len
        q_offset = total_kv_len - qo_len
        tag = f"kv{total_kv_len}_q{qo_len}"
        print(f"\n{'#'*80}")
        print(f"# 档位: {desc}")
        print(f"#   batch={batch}, kv_len={total_kv_len}, chunk_size={qo_len}, "
              f"num_chunks={num_chunks}, q_offset={q_offset}")
        print(f"{'#'*80}")
        r = test_flashinfer_vs_flex_attention(
            num_requests=batch,
            total_kv_len=total_kv_len,
            qo_len=qo_len,
            dllm_block_size=dllm_block_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            dtype=dtype,
            warmup_iters=10,
            bench_iters=50,
            verify=True,
        )
        all_results[tag] = r

    # Final comparison table
    print(f"\n\n{'='*120}")
    print(f"上下文长度对比 (chunk={chunk_size}, block_size={dllm_block_size})")
    print(f"{'='*120}")
    header = (f"  {'档位':<20} | {'KV长度':>8} | {'Batch':>5} | {'Chunks':>6} | "
             f"{'Ragged(ms)':>12} | {'Ragged cuda_graph':>12} | {'Paged(ms)':>12} | "
             f"{'Flex(ms)':>12} | {'加速比':>10}")
    print(header)
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*5}-+-{'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")

    for total_kv_len, qo_len, batch, desc in configs:
        tag = f"kv{total_kv_len}_q{qo_len}"
        r = all_results.get(tag, {})
        num_chunks = total_kv_len // qo_len
        ragged = r.get("ragged_offset", {}).get("no_cuda_graph_ms", 0)
        ragged_cuda_graph = r.get("ragged_offset", {}).get("cuda_graph_ms", 0)
        paged = r.get("paged_offset", {}).get("no_cuda_graph_ms", 0)
        flex = r.get("flex_compiled", {}).get("no_cuda_graph_ms", 0)
        speedup = f"{flex / ragged_cuda_graph:.2f}x" if ragged_cuda_graph > 0 and flex > 0 else "N/A"
        print(f"  {desc:<20} | {total_kv_len:>8} | {batch:>5} | {num_chunks:>6} | "
              f"{ragged:>12.3f} | {ragged_cuda_graph:>12.3f} | {paged:>12.3f} | "
              f"{flex:>12.3f} | {speedup:>10}")

    # 估算全量 prefill 总耗时 (所有 chunk 累加的近似)
    print(f"\n  注: 以上是最后一个 chunk (q_offset 最大) 的 per-chunk 延迟")
    print(f"  实际全量 prefill 中早期 chunk 的 kv_len 更短, 延迟更低")
    print(f"  24K/32K 档位 batch 已降低以避免 OOM")

    return all_results


# ============================================================
# 全量 Prefill 四档上下文长度测试
# ============================================================
def test_full_prefill_four_tiers(
    dllm_block_size: int = 32,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 16,
    dtype: torch.dtype = torch.float16,
):
    """
    全量 Prefill 场景: qo_len = kv_len, q_offset = 0
    一次性处理整个序列, 模拟首次 prefill

    七档: 1K / 2K / 4K / 8K / 16K / 24K / 32K
    长序列使用 batch=1 避免 OOM
    """
    configs = [
        # (seq_len, batch, label)
        (1024,  4, "1K"),
        (2048,  4, "2K"),
        (4096,  4, "4K"),
        (8192,  4, "8K"),
        (16384, 1, "16K"),
        (24576, 1, "24K"),
        (32768, 1, "32K"),
    ]

    all_results = {}
    for seq_len, batch, desc in configs:
        tag = f"seq{seq_len}_b{batch}"
        print(f"\n{'#'*80}")
        print(f"# 全量 Prefill | {desc}")
        print(f"#   batch={batch}, seq_len={seq_len} (qo_len=kv_len={seq_len}, q_offset=0)")
        print(f"{'#'*80}")
        r = test_flashinfer_vs_flex_attention(
            num_requests=batch,
            total_kv_len=seq_len,
            qo_len=seq_len,       # 全量 prefill: Q 和 KV 一样长
            dllm_block_size=dllm_block_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            dtype=dtype,
            warmup_iters=5,
            bench_iters=20,
            verify=(seq_len <= 4096),  # 长序列跳过 verify (custom_mask 太大)
        )
        all_results[tag] = r

    # 汇总表
    print(f"\n\n{'='*110}")
    print(f"全量 Prefill 上下文对比 (block_size={dllm_block_size})")
    print(f"{'='*110}")
    print(f"  {'档位':<18} | {'序列长度':>8} | {'Batch':>5} | "
          f"{'Ragged':>10} | {'Ragged cuda_graph':>10} | {'Paged':>10} | "
          f"{'Flex compiled':>14} | {'加速比':>10}")
    print(f"  {'-'*18}-+-{'-'*8}-+-{'-'*5}-+-"
          f"{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-"
          f"{'-'*14}-+-{'-'*10}")

    for seq_len, batch, desc in configs:
        tag = f"seq{seq_len}_b{batch}"
        r = all_results.get(tag, {})
        ragged = r.get("ragged_offset", {}).get("no_cuda_graph_ms", 0)
        ragged_cuda_graph = r.get("ragged_offset", {}).get("cuda_graph_ms", 0)
        paged = r.get("paged_offset", {}).get("no_cuda_graph_ms", 0)
        flex = r.get("flex_compiled", {}).get("no_cuda_graph_ms", 0)
        speedup = f"{flex / ragged:.2f}x" if ragged > 0 and flex > 0 else "N/A"
        print(f"  {desc:<18} | {seq_len:>8} | {batch:>5} | "
              f"{ragged:>10.3f} | {ragged_cuda_graph:>10.3f} | {paged:>10.3f} | "
              f"{flex:>14.3f} | {speedup:>10}")

    print(f"\n  注: 长/超长上下文使用 batch=1 (避免 OOM)")

    return all_results


# ============================================================
# Block size 对齐效应测试
# ============================================================
def test_block_size_sweep(
    num_requests: int = 4,
    total_kv_len: int = 4096,
    qo_len: int = 512,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 16,
    dtype: torch.dtype = torch.float16,
):
    """
    测试不同 dllm_block_size 对性能的影响

    Flex Attention 的 Triton tile 固定 128x128:
      - block_size=128 时完美对齐: 每个 tile 要么全 FULL 要么全 SKIP
      - block_size<128 时对角线 tile 变 PARTIAL, 需逐元素判断 mask
      - block_size>128 时 tile 粒度比 mask 粒度更细, 同样产生 PARTIAL

    FlashInfer 的 block extend kernel 内部按 dllm_block_size 粒度跳过,
    不受 128 tile 限制。
    """
    block_sizes = [32, 64, 128, 256]
    q_offset = total_kv_len - qo_len

    print(f"\n{'='*100}")
    print(f"Block Size 对齐效应测试")
    print(f"  kv_len={total_kv_len}, qo_len={qo_len}, q_offset={q_offset}")
    print(f"  Flex Attention Triton tile = 128x128 (硬编码)")
    print(f"{'='*100}")

    # 预计算每种 block_size 的 tile 分布
    num_q_tiles = (qo_len + 127) // 128
    num_kv_tiles = (total_kv_len + 127) // 128
    total_tiles = num_q_tiles * num_kv_tiles

    all_results = {}
    for bs in block_sizes:
        # 统计 tile 类型分布
        full, skip, partial = 0, 0, 0
        for qi in range(num_q_tiles):
            for ki in range(num_kv_tiles):
                q_start = qi * 128 + q_offset
                q_end = min(q_start + 128, q_offset + qo_len)
                k_start = ki * 128
                k_end = min(k_start + 128, total_kv_len)
                # 检查 tile 内 mask 是否全 True / 全 False
                q_blk_min = q_start // bs
                q_blk_max = (q_end - 1) // bs
                k_blk_min = k_start // bs
                k_blk_max = (k_end - 1) // bs
                if q_blk_min >= k_blk_max:  # Q 最小 block >= KV 最大 block → 全 True
                    full += 1
                elif q_blk_max < k_blk_min:  # Q 最大 block < KV 最小 block → 全 False
                    skip += 1
                else:
                    partial += 1

        print(f"\n{'#'*80}")
        print(f"# dllm_block_size = {bs}")
        print(f"#   128x128 tile 分布: FULL={full}, SKIP={skip}, PARTIAL={partial} (共{total_tiles})")
        print(f"#   PARTIAL 占比: {partial/total_tiles*100:.1f}%")
        print(f"{'#'*80}")

        r = test_flashinfer_vs_flex_attention(
            num_requests=num_requests,
            total_kv_len=total_kv_len,
            qo_len=qo_len,
            dllm_block_size=bs,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            dtype=dtype,
            warmup_iters=10,
            bench_iters=50,
            verify=(bs <= 128),
        )
        all_results[bs] = r

    # 汇总表
    print(f"\n\n{'='*130}")
    print(f"Block Size 对齐效应汇总 (kv_len={total_kv_len}, qo_len={qo_len}, batch={num_requests})")
    print(f"{'='*130}")
    print(f"  {'block_size':>10} | {'PARTIAL%':>8} | "
          f"{'Ragged(ms)':>10} | {'Ragged CG':>10} | {'Paged(ms)':>10} | "
          f"{'Flex compiled':>14} | {'Ragged Mem':>10} | {'Flex Mem':>10} | "
          f"{'加速比':>10}")
    print(f"  {'-'*10}-+-{'-'*8}-+-"
          f"{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-"
          f"{'-'*14}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for bs in block_sizes:
        r = all_results.get(bs, {})
        ragged = r.get("ragged_offset", {}).get("no_cuda_graph_ms", 0)
        ragged_cg = r.get("ragged_offset", {}).get("cuda_graph_ms", 0)
        paged = r.get("paged_offset", {}).get("no_cuda_graph_ms", 0)
        flex = r.get("flex_compiled", {}).get("no_cuda_graph_ms", 0)
        ragged_mem = r.get("ragged_offset", {}).get("memory", {}).get("memory_increase_mb", 0)
        flex_mem = r.get("flex_compiled", {}).get("memory", {}).get("memory_increase_mb", 0)
        speedup = f"{flex / ragged:.2f}x" if ragged > 0 and flex > 0 else "N/A"

        # 计算 partial 比例
        partial_count = 0
        for qi in range(num_q_tiles):
            for ki in range(num_kv_tiles):
                q_start = qi * 128 + q_offset
                q_end = min(q_start + 128, q_offset + qo_len)
                k_start = ki * 128
                k_end = min(k_start + 128, total_kv_len)
                q_blk_min = q_start // bs
                q_blk_max = (q_end - 1) // bs
                k_blk_min = k_start // bs
                k_blk_max = (k_end - 1) // bs
                if not (q_blk_min >= k_blk_max or q_blk_max < k_blk_min):
                    partial_count += 1
        pct = f"{partial_count/total_tiles*100:.1f}%"

        print(f"  {bs:>10} | {pct:>8} | "
              f"{ragged:>10.3f} | {ragged_cg:>10.3f} | {paged:>10.3f} | "
              f"{flex:>14.3f} | {ragged_mem:>10.1f} | {flex_mem:>10.1f} | {speedup:>10}")

    return all_results


# ============================================================
# 全量 Prefill 总显存+性能全面对比: 扫描上下文长度 × dllm_block_size
# ============================================================
def test_total_memory_comparison(
    num_requests: int = 4,
    qo_len: int = 512,
    dllm_block_size: int = 32,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 16,
    dtype: torch.dtype = torch.float16,
):
    """
    全量 Prefill 全面对比: qo_len = kv_len, q_offset = 0
    扫描 6 档上下文长度 × 4 档 dllm_block_size
    每个组合测量: FlashInfer 和 Flex Attention 的延迟(ms) + 峰值总显存(MB)
    Flex Attention BLOCK_SIZE 保持默认, 仅变化 mask 逻辑粒度
    长序列自动降 batch 避免 OOM
    """
    device = torch.device("cuda:0")
    sm_scale = 1.0 / math.sqrt(head_dim)

    # (seq_len, batch) — 长序列降 batch 避免 OOM
    configs = [
        (2048,  num_requests),
        (4096,  num_requests),
        (8192,  num_requests),
        (16384, 1),
        (24576, 1),
        (32768, 1),
    ]
    dllm_block_sizes = [32, 64, 128, 256]
    warmup_iters, bench_iters = 10, 50

    print(f"\n{'='*140}")
    print(f"全量 Prefill: FlashInfer Ragged vs Flex compiled 全面对比")
    print(f"{'='*140}")
    print(f"  场景: qo_len = kv_len (全量 prefill), q_offset = 0")
    print(f"  heads={num_heads}/{num_kv_heads}, head_dim={head_dim}")
    print(f"  seq_lens: {[c[0] for c in configs]}")
    print(f"  batch:    {[c[1] for c in configs]}")
    print(f"  dllm_block_sizes: {dllm_block_sizes}")
    print(f"  Flex BLOCK_SIZE: 默认 (kernel 自决)")
    print()

    # 探测最小 workspace (最大 seq_len, batch=1)
    min_ws_mb = 256
    if HAS_FLASHINFER:
        max_seq = configs[-1][0]
        probe_batch = configs[-1][1]
        print(f"  探测最小 workspace (seq_len={max_seq}, batch={probe_batch}) ...")
        _q = torch.randn(probe_batch * max_seq, num_heads, head_dim, dtype=dtype, device=device)
        _k = torch.randn(probe_batch * max_seq, num_kv_heads, head_dim, dtype=dtype, device=device)
        _v = torch.randn(probe_batch * max_seq, num_kv_heads, head_dim, dtype=dtype, device=device)
        _qo = torch.tensor([i * max_seq for i in range(probe_batch + 1)], dtype=torch.int32, device=device)
        _kv = torch.tensor([i * max_seq for i in range(probe_batch + 1)], dtype=torch.int32, device=device)
        _qoff = torch.zeros(probe_batch, dtype=torch.int32, device=device)
        for try_mb in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            try:
                _ws = torch.empty(try_mb * 1024 * 1024, dtype=torch.uint8, device=device)
                _w = BatchBlockExtendRaggedOffsetWrapper(_ws, kv_layout="NHD", dllm_block_size=dllm_block_sizes[0])
                _w.plan(qo_indptr=_qo, kv_indptr=_kv,
                        num_qo_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
                        q_data_type=dtype, sm_scale=sm_scale, q_offsets=_qoff)
                _w.run(_q, _k, _v)
                torch.cuda.synchronize(device)
                min_ws_mb = try_mb
                del _ws, _w
                break
            except Exception:
                torch.cuda.empty_cache()
                continue
        del _q, _k, _v, _qo, _kv, _qoff
        torch.cuda.empty_cache()
        print(f"    最小可用 workspace: {min_ws_mb} MB")
    print()

    def _reset():
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.memory_allocated(device) / 1024**2

    # {(seq_len, batch, dllm_bs): {"fi_ms", "fi_peak", "flex_ms", "flex_peak"}}
    all_results = {}

    for seq_len, batch in configs:
        # 全量 prefill: qo_len = kv_len = seq_len, q_offset = 0
        q_offset = 0
        print(f"  --- seq_len={seq_len}, batch={batch} (全量 prefill, q_offset=0) ---")

        for dbs in dllm_block_sizes:
            key = (seq_len, batch, dbs)
            entry = {"batch": batch}

            # ===== FlashInfer Ragged =====
            if HAS_FLASHINFER:
                base = _reset()
                q = torch.randn(batch * seq_len, num_heads, head_dim, dtype=dtype, device=device)
                k = torch.randn(batch * seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
                v = torch.randn(batch * seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
                qo_indptr = torch.tensor([i * seq_len for i in range(batch + 1)], dtype=torch.int32, device=device)
                kv_indptr = torch.tensor([i * seq_len for i in range(batch + 1)], dtype=torch.int32, device=device)
                q_offsets = torch.zeros(batch, dtype=torch.int32, device=device)
                ws = torch.empty(min_ws_mb * 1024 * 1024, dtype=torch.uint8, device=device)

                wrapper = BatchBlockExtendRaggedOffsetWrapper(ws, kv_layout="NHD", dllm_block_size=dbs)
                wrapper.plan(
                    qo_indptr=qo_indptr, kv_indptr=kv_indptr,
                    num_qo_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
                    q_data_type=dtype, sm_scale=sm_scale, q_offsets=q_offsets,
                )

                def _run_fi():
                    wrapper.run(q, k, v)

                for _ in range(warmup_iters):
                    _run_fi()
                torch.cuda.synchronize(device)
                fi_peak = torch.cuda.max_memory_allocated(device) / 1024**2 - base

                # no cuda_graph latency
                t0 = time.perf_counter()
                for _ in range(bench_iters):
                    _run_fi()
                torch.cuda.synchronize(device)
                fi_ms = (time.perf_counter() - t0) / bench_iters * 1000

                # cuda_graph latency
                try:
                    fi_cg_ms = benchmark_with_cuda_graph(_run_fi, warmup_iters, bench_iters)
                    entry["fi_cg_ms"] = fi_cg_ms
                except Exception:
                    pass

                entry["fi_ms"] = fi_ms
                entry["fi_peak"] = fi_peak
                del q, k, v, ws, wrapper, qo_indptr, kv_indptr, q_offsets

            # ===== Flex Attention (compiled, 默认 BLOCK_SIZE) =====
            if HAS_FLEX_ATTENTION:
                base = _reset()
                q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)
                k = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
                v = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)

                mask_mod = make_block_extend_mask_mod(dbs, q_offset)
                block_mask = create_block_mask(
                    mask_mod, B=batch, H=1,
                    Q_LEN=seq_len, KV_LEN=seq_len, device=device,
                )
                use_gqa = (num_heads != num_kv_heads)

                torch._dynamo.reset()
                _compiled = torch.compile(flex_attention, dynamic=False)

                try:
                    for _ in range(warmup_iters):
                        _compiled(q, k, v, block_mask=block_mask, scale=sm_scale, enable_gqa=use_gqa)
                    torch.cuda.synchronize(device)
                    flex_peak = torch.cuda.max_memory_allocated(device) / 1024**2 - base

                    t0 = time.perf_counter()
                    for _ in range(bench_iters):
                        _compiled(q, k, v, block_mask=block_mask, scale=sm_scale, enable_gqa=use_gqa)
                    torch.cuda.synchronize(device)
                    flex_ms = (time.perf_counter() - t0) / bench_iters * 1000

                    entry["flex_ms"] = flex_ms
                    entry["flex_peak"] = flex_peak
                except Exception as e:
                    print(f"    [dllm_bs={dbs}, seq={seq_len}] Flex 失败: {e}")

                del q, k, v, block_mask, _compiled

            all_results[key] = entry

            # 单行进度
            fi_cg = f"CG={entry['fi_cg_ms']:.3f}ms" if "fi_cg_ms" in entry else "CG=N/A"
            fi_s = f"{entry.get('fi_ms', 0):.3f}ms/{fi_cg}/{entry.get('fi_peak', 0):.0f}MB"
            fx_s = f"{entry.get('flex_ms', 0):.3f}ms/{entry.get('flex_peak', 0):.0f}MB" if "flex_ms" in entry else "N/A"
            print(f"    dllm_bs={dbs:<3}  FI={fi_s:<32} Flex={fx_s}")

    # ===== 汇总表 =====
    print(f"\n{'='*170}")
    print(f"全量 Prefill 汇总: FlashInfer Ragged vs Flex compiled")
    print(f"  场景: qo_len = kv_len, q_offset = 0, FI workspace={min_ws_mb}MB, Flex BLOCK_SIZE=默认(kernel自决)")
    print(f"{'='*170}")
    print(f"  {'seq_len':>8} | {'batch':>5} | {'dllm_bs':>7} | {'FI(ms)':>8} | {'FI CG(ms)':>10} | {'FI peak(MB)':>12} | {'Flex(ms)':>9} | {'Flex peak(MB)':>14} | {'加速比':>8} | {'CG加速比':>9} | {'显存节省':>8}")
    print(f"  {'-'*8}-+-{'-'*5}-+-{'-'*7}-+-{'-'*8}-+-{'-'*10}-+-{'-'*12}-+-{'-'*9}-+-{'-'*14}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}")

    fi_wins_perf, fi_wins_cg, fi_wins_mem, total_cmp = 0, 0, 0, 0

    for seq_len, batch in configs:
        for dbs in dllm_block_sizes:
            e = all_results.get((seq_len, batch, dbs), {})
            fi_ms = e.get("fi_ms", 0)
            fi_cg = e.get("fi_cg_ms", 0)
            fi_pk = e.get("fi_peak", 0)
            fx_ms = e.get("flex_ms", 0)
            fx_pk = e.get("flex_peak", 0)

            if fi_ms > 0 and fx_ms > 0:
                ratio = f"{fx_ms / fi_ms:.2f}x"
                cg_ratio = f"{fx_ms / fi_cg:.2f}x" if fi_cg > 0 else "N/A"
                mem_save = f"{(1 - fi_pk / fx_pk) * 100:+.0f}%" if fx_pk > 0 else "N/A"
                total_cmp += 1
                if fi_ms < fx_ms:
                    fi_wins_perf += 1
                if fi_cg > 0 and fi_cg < fx_ms:
                    fi_wins_cg += 1
                if fi_pk < fx_pk:
                    fi_wins_mem += 1
            else:
                ratio = "N/A"
                cg_ratio = "N/A"
                mem_save = "N/A"

            fi_cg_s = f"{fi_cg:>10.3f}" if fi_cg > 0 else f"{'N/A':>10}"
            fx_ms_s = f"{fx_ms:>9.3f}" if fx_ms > 0 else f"{'N/A':>9}"
            fx_pk_s = f"{fx_pk:>14.1f}" if fx_pk > 0 else f"{'N/A':>14}"

            print(f"  {seq_len:>8} | {batch:>5} | {dbs:>7} | {fi_ms:>8.3f} | {fi_cg_s} | {fi_pk:>12.1f} | {fx_ms_s} | {fx_pk_s} | {ratio:>8} | {cg_ratio:>9} | {mem_save:>8}")

    # 统计
    print(f"\n  统计 ({total_cmp} 场有效对比):")
    if total_cmp > 0:
        print(f"    性能 (no CG):  FlashInfer 胜出 {fi_wins_perf}/{total_cmp} 场")
        print(f"    性能 (FI CG):  FlashInfer 胜出 {fi_wins_cg}/{total_cmp} 场")
        print(f"    显存:          FlashInfer 胜出 {fi_wins_mem}/{total_cmp} 场")
    print(f"\n  注: 16K+ 序列使用 batch=1 避免 OOM")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FlashInfer vs Flex Attention Benchmark")
    parser.add_argument("--sweep", action="store_true", help="Run sweep across seq lengths (multi-chunk)")
    parser.add_argument("--single-chunk", action="store_true", help="Single chunk, four context length tiers")
    parser.add_argument("--full-prefill", action="store_true", help="Full prefill (qo=kv), four context length tiers")
    parser.add_argument("--block-size-sweep", action="store_true", help="Sweep dllm_block_size (32/64/128/256) alignment effect")
    parser.add_argument("--memory-compare", action="store_true", help="Total GPU memory comparison from clean state")
    parser.add_argument("--num-requests", type=int, default=4)
    parser.add_argument("--kv-len", type=int, default=4096)
    parser.add_argument("--qo-len", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    if args.sweep:
        test_sweep_seq_lengths(
            num_requests=args.num_requests,
            dllm_block_size=args.block_size,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            page_size=args.page_size,
        )
    elif args.full_prefill:
        test_full_prefill_four_tiers(
            dllm_block_size=args.block_size,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            page_size=args.page_size,
        )
    elif args.block_size_sweep:
        test_block_size_sweep(
            num_requests=args.num_requests,
            total_kv_len=args.kv_len,
            qo_len=args.qo_len,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            page_size=args.page_size,
        )
    elif args.memory_compare:
        test_total_memory_comparison(
            num_requests=args.num_requests,
            qo_len=args.qo_len,
            dllm_block_size=args.block_size,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            page_size=args.page_size,
        )
    else:
        test_flashinfer_vs_flex_attention(
            num_requests=args.num_requests,
            total_kv_len=args.kv_len,
            qo_len=args.qo_len,
            dllm_block_size=args.block_size,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            page_size=args.page_size,
            verify=not args.no_verify,
        )
