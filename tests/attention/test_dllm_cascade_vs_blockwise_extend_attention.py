"""DLLM Block-wise Mask Implementation Comparison

Comparing three implementation approaches:
1. Cascade Attention (SGLang approach): Ragged + Paged + merge_state
   - Ragged: Bidirectional attention within current block (causal=False)
   - Paged: Access all previous cached blocks (causal=False)
   - merge_state: Merge softmax states from both stages

2. Batch Prefill + kBlockExtend + q_offset (FlashInfer optimized approach)
   - Single kernel launch
   - Tile-level skip for invalid computations

3. V2 Serial approach (reference baseline)
   - Each chunk called independently
   - Tile-level skip

Mask rule: mask[q, k] = (q_global // B) >= (k_global // B)
"""

import torch
import time
import math
import flashinfer
from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    merge_state,
    single_prefill_with_kv_cache,
)
from flashinfer.dllm import (
    BatchBlockExtendPagedOffsetWrapper,
    BatchBlockExtendRaggedOffsetWrapper,
    block_extend_attention_with_offset,
)


def compute_block_extend_reference(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    dllm_block_size: int,
    q_offset: int = 0,
    sm_scale: float = None,
) -> torch.Tensor:
    """
    Compute Block Extend Attention reference result using custom_mask
    
    Mask rule: mask[q, k] = ((q_local + q_offset) // B) >= (k // B)
    
    Args:
        q: [qo_len, num_heads, head_dim]
        k: [kv_len, num_kv_heads, head_dim]
        v: [kv_len, num_kv_heads, head_dim]
        dllm_block_size: DLLM block size
        q_offset: Q's global starting position
        sm_scale: softmax scale
    
    Returns:
        output: [qo_len, num_heads, head_dim]
    """
    qo_len = q.shape[0]
    kv_len = k.shape[0]
    head_dim = q.shape[-1]
    device = q.device
    
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    
    # Construct custom_mask
    # q_global = q_local + q_offset
    # mask[q, k] = (q_global // B) >= (k // B)
    q_pos = torch.arange(qo_len, device=device) + q_offset
    k_pos = torch.arange(kv_len, device=device)
    q_block = q_pos.unsqueeze(1) // dllm_block_size  # [qo_len, 1]
    k_block = k_pos.unsqueeze(0) // dllm_block_size  # [1, kv_len]
    mask_2d = (q_block >= k_block).to(torch.uint8)   # [qo_len, kv_len]
    
    return single_prefill_with_kv_cache(
        q, k, v,
        custom_mask=mask_2d,
        sm_scale=sm_scale,
    )

def test_incremental_batchprefill_step_by_step_with_cuda_graph(
    num_requests: int = 4,
    tokens_per_request: int = 256,
    dllm_block_size: int = 32,
    chunk_sizes: list = None,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    verbose: bool = False,
):
    """
    Realistically simulate DLLM incremental Prefill step-by-step execution flow + CUDA Graph
    
    Key points:
      1. Must execute step by step, each chunk step depends on previous step's KV cache
      2. SGLang Cascade forces chunk_size = dllm_block_size
      3. BatchBlockExtend can use larger chunk_size, reducing number of steps
      4. Enable CUDA Graph to reduce CPU overhead and kernel launch latency
    
    CUDA Graph notes:
      - plan() contains CPU-GPU synchronization, cannot execute during capture
      - Must create independent wrapper for each step, complete plan before capture
      - Only capture run() operations
    """
    if chunk_sizes is None:
        chunk_sizes = [32, 64, 128, 256]
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    sm_scale = 1.0 / (head_dim ** 0.5)
    
    # Baseline chunk_size = dllm_block_size
    baseline_chunk_size = dllm_block_size
    baseline_num_chunks = tokens_per_request // baseline_chunk_size
    
    print(f"\n{'='*80}")
    print(f"Step-by-step Execution + CUDA Graph: DLLM Incremental Prefill Performance Comparison")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  num_requests        = {num_requests}")
    print(f"  tokens_per_request  = {tokens_per_request}")
    print(f"  dllm_block_size     = {dllm_block_size}")
    print(f"  chunk_sizes to test = {chunk_sizes}")
    print(f"  num_heads           = {num_heads}")
    print(f"  num_kv_heads        = {num_kv_heads}")
    print(f"  head_dim            = {head_dim}")
    print(f"\nKey features:")
    print(f"  - Step-by-step execution: Each step must wait for previous step to complete")
    print(f"  - SGLang forces chunk_size = dllm_block_size = {dllm_block_size}")
    print(f"  - BatchBlockExtend can use larger chunk_size")
    print(f"  - CUDA Graph: Reduce CPU overhead and kernel launch latency")
    
    # Data preparation, generate complete Q, K, V for each request
    all_qs = [torch.randn(tokens_per_request, num_heads, head_dim, dtype=dtype, device=device) 
              for _ in range(num_requests)]
    all_ks = [torch.randn(tokens_per_request, num_kv_heads, head_dim, dtype=dtype, device=device) 
              for _ in range(num_requests)]
    all_vs = [torch.randn(tokens_per_request, num_kv_heads, head_dim, dtype=dtype, device=device) 
              for _ in range(num_requests)]
    
    # Split each request into chunks 
    def split_chunks(tensor, chunk_size):
        return [tensor[i*chunk_size:(i+1)*chunk_size] for i in range(tensor.shape[0] // chunk_size)]
    
    qs_chunks = [split_chunks(q, baseline_chunk_size) for q in all_qs]
    ks_chunks = [split_chunks(k, baseline_chunk_size) for k in all_ks]
    vs_chunks = [split_chunks(v, baseline_chunk_size) for v in all_vs]
    
    results = {}

    # Correctness verification (sample verification of first request)
    print(f"\n{'='*80}")
    print(f"Correctness verification (sample verification of request_0)")
    print(f"{'='*80}")
    print(f"  Reference implementation: single_prefill_with_kv_cache + custom_mask")
    print(f"  Mask rule: mask[q,k] = ((q + offset) // B) >= (k // B)")
    
    # Sample verification of first request
    req_idx = 0
    k_req = all_ks[req_idx]
    v_req = all_vs[req_idx]
    
    # Cumulative KV buffer
    k_cumul_verify = [k_req[:(i+1)*baseline_chunk_size] for i in range(baseline_num_chunks)]
    v_cumul_verify = [v_req[:(i+1)*baseline_chunk_size] for i in range(baseline_num_chunks)]
    
    # Compute reference results
    ref_outputs = []
    for step_idx in range(baseline_num_chunks):
        q_offset = step_idx * baseline_chunk_size
        ref_out = compute_block_extend_reference(
            qs_chunks[req_idx][step_idx],
            k_cumul_verify[step_idx],
            v_cumul_verify[step_idx],
            dllm_block_size=dllm_block_size,
            q_offset=q_offset,
            sm_scale=sm_scale,
        )
        ref_outputs.append(ref_out)
    
    bbe_outputs = []
    workspace_verify = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    qo_indptr_verify = torch.tensor([0, baseline_chunk_size], dtype=torch.int32, device=device)
    for step_idx in range(baseline_num_chunks):
        kv_len = (step_idx + 1) * baseline_chunk_size
        kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=device)
        q_offset_tensor = torch.tensor([step_idx * baseline_chunk_size], dtype=torch.int32, device=device)
        
        wrapper = BatchBlockExtendRaggedOffsetWrapper(
            workspace_verify, kv_layout="NHD", dllm_block_size=dllm_block_size
        )
        wrapper.plan(
            qo_indptr=qo_indptr_verify,
            kv_indptr=kv_indptr,
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_data_type=dtype,
            sm_scale=sm_scale,
            q_offsets=q_offset_tensor,
        )
        bbe_out = wrapper.run(qs_chunks[req_idx][step_idx], k_cumul_verify[step_idx], v_cumul_verify[step_idx])
        bbe_outputs.append(bbe_out)
    
    # Verification
    tol = 1e-2
    bbe_max_diff = max((bbe_outputs[i] - ref_outputs[i]).abs().max().item() for i in range(baseline_num_chunks))
    bbe_pass = bbe_max_diff < tol
    print(f"\n  [BBE] BatchBlockExtendRaggedOffsetWrapper:")
    print(f"        max_diff = {bbe_max_diff:.6f}, tolerance = {tol}")
    print(f"        {' PASS' if bbe_pass else ' FAIL'}")
    
    if not bbe_pass:
        print(f"\n   Correctness verification failed, but continue with performance test")
    else:
        print(f"\n   BBE correctness verification passed")
    
    del workspace_verify, bbe_outputs, ref_outputs, k_cumul_verify, v_cumul_verify
    torch.cuda.empty_cache()
    
    # Baseline 2: Custom Mask (BatchPrefillWithRaggedKVCacheWrapper + custom_mask) + CUDA Graph
    print(f"\n{'='*80}")
    print(f"[Baseline 2] Custom Mask (BatchPrefill + custom_mask)")
    print(f"{'='*80}")
    print(f"  num_steps: {baseline_num_chunks}")
    print(f"  num_requests: {num_requests}")
    print(f"  Each step: BatchPrefillWithRaggedKVCacheWrapper + custom_mask")
    print(f"  Kernels per step: 1 (batch processes all requests)")
    
    # Pre-allocate Q buffers (concat all requests)
    cm_q_buffers = []
    for step_idx in range(baseline_num_chunks):
        q_list = [qs_chunks[req_idx][step_idx] for req_idx in range(num_requests)]
        cm_q_buffers.append(torch.cat(q_list, dim=0))
    
    # Pre-allocate cumulative KV buffers (concat all requests)
    cm_k_buffers = []
    cm_v_buffers = []
    for step_idx in range(baseline_num_chunks):
        kv_len = (step_idx + 1) * baseline_chunk_size
        k_cumul_list = [all_ks[req_idx][:kv_len] for req_idx in range(num_requests)]
        v_cumul_list = [all_vs[req_idx][:kv_len] for req_idx in range(num_requests)]
        cm_k_buffers.append(torch.cat(k_cumul_list, dim=0))
        cm_v_buffers.append(torch.cat(v_cumul_list, dim=0))
    
    # Construct flattened custom_mask (batch version)
    # custom_mask shape: (sum(q_len[i] * k_len[i] for i in range(batch_size)))
    # Each request's mask is the same, concat num_requests times
    custom_mask_buffers = []
    for step_idx in range(baseline_num_chunks):
        kv_len = (step_idx + 1) * baseline_chunk_size
        q_offset = step_idx * baseline_chunk_size
        # Construct single request's 2D mask: [q_len, kv_len]
        q_pos = torch.arange(baseline_chunk_size, device=device) + q_offset
        k_pos = torch.arange(kv_len, device=device)
        q_block = q_pos.unsqueeze(1) // dllm_block_size
        k_block = k_pos.unsqueeze(0) // dllm_block_size
        mask_2d = (q_block >= k_block)  # [q_len, kv_len], bool
        # Flatten and repeat for batch
        mask_flat = mask_2d.flatten()  # [q_len * kv_len]
        # All requests have the same mask, concat
        batch_mask = mask_flat.repeat(num_requests)  # [num_requests * q_len * kv_len]
        custom_mask_buffers.append(batch_mask)
    
    # qo_indptr and kv_indptr
    cm_qo_indptr = torch.tensor(
        [i * baseline_chunk_size for i in range(num_requests + 1)],
        dtype=torch.int32, device=device
    )
    cm_kv_indptr_list = []
    for step_idx in range(baseline_num_chunks):
        kv_len = (step_idx + 1) * baseline_chunk_size
        cm_kv_indptr_list.append(torch.tensor(
            [i * kv_len for i in range(num_requests + 1)],
            dtype=torch.int32, device=device
        ))
    
    # Create independent wrapper for each step and complete plan
    # Note: custom_mask is only supported in FA2 backend, not FA3
    print(f"  Creating wrappers and completing plan...")
    cm_wrappers = []
    for step_idx in range(baseline_num_chunks):
        wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device), 
            kv_layout="NHD",
            backend="fa2",  # custom_mask only supported in FA2
        )
        wrapper.plan(
            qo_indptr=cm_qo_indptr,
            kv_indptr=cm_kv_indptr_list[step_idx],
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            custom_mask=custom_mask_buffers[step_idx],
            causal=False,
            sm_scale=sm_scale,
        )
        cm_wrappers.append(wrapper)
    
    cm_output = torch.empty(num_requests * baseline_chunk_size, num_heads, head_dim, dtype=dtype, device=device)
    
    def run_custom_mask_pipeline():
        for step_idx in range(baseline_num_chunks):
            cm_output.copy_(cm_wrappers[step_idx].run(
                cm_q_buffers[step_idx], cm_k_buffers[step_idx], cm_v_buffers[step_idx]
            ))

    if verbose:
        print(f"  Step flow preview:")
        for step_id in range(baseline_num_chunks):
            kv_len = (step_id + 1) * baseline_chunk_size
            print(f"    Step {step_id}: Q[{step_id*baseline_chunk_size}:{(step_id+1)*baseline_chunk_size}] attend to KV[0:{kv_len}]")
    
    # Warmup
    print(f"  Warmup...")
    for _ in range(warmup_iters):
        run_custom_mask_pipeline()
    torch.cuda.synchronize()
    
    # CUDA Graph capture
    print(f"  Capturing CUDA Graph...")
    cm_stream = torch.cuda.Stream()
    with torch.cuda.stream(cm_stream):
        run_custom_mask_pipeline()
    cm_stream.synchronize()
    
    cm_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cm_graph, stream=cm_stream):
        run_custom_mask_pipeline()
    
    # Warmup with cuda_graph
    for _ in range(warmup_iters):
        cm_graph.replay()
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"  Benchmark (with cuda_graph)...")
    start = time.perf_counter()
    for _ in range(bench_iters):
        cm_graph.replay()
    torch.cuda.synchronize()
    cm_cuda_graph_time = (time.perf_counter() - start) / bench_iters * 1000
    
    results["custom_mask_baseline"] = {
        "time_cuda_graph_ms": cm_cuda_graph_time,
        "chunk_size": baseline_chunk_size,
        "num_steps": baseline_num_chunks,
        "method": "Custom Mask (Baseline 2)",
    }
    print(f"  => cuda_graph: {cm_cuda_graph_time:.3f} ms ({cm_cuda_graph_time/baseline_num_chunks:.3f} ms/step × {baseline_num_chunks} steps)")
    
    del cm_graph, custom_mask_buffers, cm_wrappers
    torch.cuda.empty_cache()

    # Baseline 1: SGLang Cascade (chunk_size = dllm_block_size) + CUDA Graph
    print(f"\n{'='*80}")
    print(f"[Baseline 1] SGLang Cascade (chunk_size = dllm_block_size = {baseline_chunk_size})")
    print(f"{'='*80}")
    print(f"  num_steps: {baseline_num_chunks}")
    print(f"  Each step: BatchRagged(current chunk) + BatchRagged(prefix) + merge_state")
    print(f"  Kernels per step: 2-3")
    
    # Pre-allocate all buffers (for CUDA Graph)
    q_current_buffers = []
    k_current_buffers = []
    v_current_buffers = []
    for step_idx in range(baseline_num_chunks):
        q_list = [qs_chunks[req_idx][step_idx] for req_idx in range(num_requests)]
        k_list = [ks_chunks[req_idx][step_idx] for req_idx in range(num_requests)]
        v_list = [vs_chunks[req_idx][step_idx] for req_idx in range(num_requests)]
        q_current_buffers.append(torch.cat(q_list, dim=0))
        k_current_buffers.append(torch.cat(k_list, dim=0))
        v_current_buffers.append(torch.cat(v_list, dim=0))
    
    # Pre-allocate prefix KV buffer
    k_prefix_buffers = [None]
    v_prefix_buffers = [None]
    for step_idx in range(1, baseline_num_chunks):
        prefix_len = step_idx * baseline_chunk_size
        k_prefix_list = [all_ks[req_idx][:prefix_len] for req_idx in range(num_requests)]
        v_prefix_list = [all_vs[req_idx][:prefix_len] for req_idx in range(num_requests)]
        k_prefix_buffers.append(torch.cat(k_prefix_list, dim=0))
        v_prefix_buffers.append(torch.cat(v_prefix_list, dim=0))
    
    cascade_output = torch.empty(num_requests * baseline_chunk_size, num_heads, head_dim, dtype=dtype, device=device)
    
    # indptr
    qo_indptr_current = torch.tensor(
        [i * baseline_chunk_size for i in range(num_requests + 1)],
        dtype=torch.int32, device=device
    )
    kv_indptr_prefix_list = [None]
    for step_idx in range(1, baseline_num_chunks):
        prefix_len = step_idx * baseline_chunk_size
        kv_indptr_prefix_list.append(torch.tensor(
            [i * prefix_len for i in range(num_requests + 1)],
            dtype=torch.int32, device=device
        ))
    
    # Create independent wrapper for each step and complete plan (critical!)
    print(f"  Creating wrappers and completing plan...")
    cascade_wrappers_current = []
    cascade_wrappers_prefix = []
    
    for step_idx in range(baseline_num_chunks):
        # Wrapper for current chunk
        wrapper_current = BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device), kv_layout="NHD",backend = "fa3",
        )
        wrapper_current.plan(
            qo_indptr=qo_indptr_current,
            kv_indptr=qo_indptr_current,
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            causal=False,
            sm_scale=sm_scale,
        )
        cascade_wrappers_current.append(wrapper_current)
        
        # Wrapper for prefix (step 0 has no prefix)
        if step_idx == 0:
            cascade_wrappers_prefix.append(None)
        else:
            wrapper_prefix = BatchPrefillWithRaggedKVCacheWrapper(
                torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device), kv_layout="NHD",backend = "fa3",
            )
            wrapper_prefix.plan(
                qo_indptr=qo_indptr_current,
                kv_indptr=kv_indptr_prefix_list[step_idx],
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                causal=False,
                sm_scale=sm_scale,
            )
            cascade_wrappers_prefix.append(wrapper_prefix)

    o1_buffer = torch.empty(num_requests * baseline_chunk_size, num_heads, head_dim, dtype=dtype, device=device)
    s1_buffer = torch.empty(num_requests * baseline_chunk_size, num_heads, dtype=torch.float32, device=device)
    o2_buffer = torch.empty(num_requests * baseline_chunk_size, num_heads, head_dim, dtype=dtype, device=device)
    s2_buffer = torch.empty(num_requests * baseline_chunk_size, num_heads, dtype=torch.float32, device=device)
    
    def run_cascade_pipeline():
        for step_idx in range(baseline_num_chunks):
            q_batch = q_current_buffers[step_idx]
            k_current = k_current_buffers[step_idx]
            v_current = v_current_buffers[step_idx]
            
            if step_idx == 0:
                cascade_output.copy_(cascade_wrappers_current[step_idx].run(q_batch, k_current, v_current))
            else:
                o1, s1 = cascade_wrappers_current[step_idx].run_return_lse(q_batch, k_current, v_current)
                o1_buffer.copy_(o1)
                s1_buffer.copy_(s1)
                
                o2, s2 = cascade_wrappers_prefix[step_idx].run_return_lse(
                    q_batch, k_prefix_buffers[step_idx], v_prefix_buffers[step_idx]
                )
                o2_buffer.copy_(o2)
                s2_buffer.copy_(s2)
                
                o, _ = merge_state(o1_buffer, s1_buffer, o2_buffer, s2_buffer)
                cascade_output.copy_(o)
    
    # Display step flow
    if verbose:
        print(f"  Step flow preview:")
        for step_id in range(baseline_num_chunks):
            prefix_len = step_id * baseline_chunk_size
            if step_id == 0:
                print(f"    Step {step_id}: current_chunk[{baseline_chunk_size}] only (no prefix)")
            else:
                print(f"    Step {step_id}: current_chunk[{baseline_chunk_size}] + prefix[{prefix_len}] + merge")
    
    # Warmup
    print(f"  Warmup...")
    for _ in range(warmup_iters):
        run_cascade_pipeline()
    torch.cuda.synchronize()
    
    # CUDA Graph capture
    print(f"  Capturing CUDA Graph...")
    cascade_stream = torch.cuda.Stream()
    with torch.cuda.stream(cascade_stream):
        run_cascade_pipeline()
    cascade_stream.synchronize()
    
    cascade_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cascade_graph, stream=cascade_stream):
        run_cascade_pipeline()
    
    # Warmup with cuda_graph
    for _ in range(warmup_iters):
        cascade_graph.replay()
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"  Benchmark (with cuda_graph)...")
    start = time.perf_counter()
    for _ in range(bench_iters):
        cascade_graph.replay()
    torch.cuda.synchronize()
    cascade_cuda_graph_time = (time.perf_counter() - start) / bench_iters * 1000
    
    results["cascade_baseline"] = {
        "time_cuda_graph_ms": cascade_cuda_graph_time,
        "chunk_size": baseline_chunk_size,
        "num_steps": baseline_num_chunks,
        "method": "SGLang Cascade (Baseline 1)",
    }
    print(f"  => cuda_graph: {cascade_cuda_graph_time:.3f} ms ({cascade_cuda_graph_time/baseline_num_chunks:.3f} ms/step × {baseline_num_chunks} steps)")
    
    del cascade_wrappers_current, cascade_wrappers_prefix, cascade_graph
    torch.cuda.empty_cache()

    # Comparison: BatchBlockExtend Ragged (different chunk_size) + CUDA Graph

    for test_chunk_size in chunk_sizes:
        if tokens_per_request % test_chunk_size != 0:
            print(f"\n[Skip] chunk_size={test_chunk_size} cannot divide tokens_per_request={tokens_per_request}")
            continue
        
        num_chunks_bbe = tokens_per_request // test_chunk_size
        
        print(f"\n{'-'*60}")
        print(f"[BatchBlockExtend Ragged] chunk_size = {test_chunk_size}")
        print(f"{'-'*60}")
        print(f"  num_steps: {num_chunks_bbe} ({baseline_num_chunks - num_chunks_bbe} steps fewer than Baseline)")
        print(f"  Kernels per step: 1")
        
        # Split each request into chunks
        qs_chunks_bbe = [split_chunks(q, test_chunk_size) for q in all_qs]
        
        # Pre-allocate all buffers
        bbe_q_buffers = []
        for step_idx in range(num_chunks_bbe):
            q_list = [qs_chunks_bbe[req_idx][step_idx] for req_idx in range(num_requests)]
            bbe_q_buffers.append(torch.cat(q_list, dim=0))
        
        bbe_k_buffers = []
        bbe_v_buffers = []
        for step_idx in range(num_chunks_bbe):
            kv_len = (step_idx + 1) * test_chunk_size
            k_cumul_list = [all_ks[req_idx][:kv_len] for req_idx in range(num_requests)]
            v_cumul_list = [all_vs[req_idx][:kv_len] for req_idx in range(num_requests)]
            bbe_k_buffers.append(torch.cat(k_cumul_list, dim=0))
            bbe_v_buffers.append(torch.cat(v_cumul_list, dim=0))
        
        bbe_qo_indptr = torch.tensor(
            [i * test_chunk_size for i in range(num_requests + 1)],
            dtype=torch.int32, device=device
        )
        bbe_kv_indptr_list = []
        bbe_q_offsets_list = []
        for step_idx in range(num_chunks_bbe):
            kv_len = (step_idx + 1) * test_chunk_size
            bbe_kv_indptr_list.append(torch.tensor(
                [i * kv_len for i in range(num_requests + 1)],
                dtype=torch.int32, device=device
            ))
            q_offset = step_idx * test_chunk_size
            """
            bbe_q_offsets_list = [
                # step_idx=0: q_offset = 0 * 64 = 0
                tensor([0, 0, 0, 0], dtype=int32),     # shape: (4,)
                
                # step_idx=1: q_offset = 1 * 64 = 64
                tensor([64, 64, 64, 64], dtype=int32), # shape: (4,)
                
                # step_idx=2: q_offset = 2 * 64 = 128
                tensor([128, 128, 128, 128], dtype=int32), # shape: (4,)
            ]
            """
            bbe_q_offsets_list.append(torch.full((num_requests,), q_offset, dtype=torch.int32, device=device))
        
        bbe_output = torch.empty(num_requests * test_chunk_size, num_heads, head_dim, dtype=dtype, device=device)

        print(f"  Creating wrappers and completing plan...")
        bbe_wrappers = []
        for step_idx in range(num_chunks_bbe):
            wrapper = BatchBlockExtendRaggedOffsetWrapper(
                torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device),
                kv_layout="NHD", dllm_block_size=dllm_block_size
            )
            wrapper.plan(
                qo_indptr=bbe_qo_indptr,
                kv_indptr=bbe_kv_indptr_list[step_idx],
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_data_type=dtype,
                sm_scale=sm_scale,
                q_offsets=bbe_q_offsets_list[step_idx],
            )
            bbe_wrappers.append(wrapper)
        
        def run_bbe_pipeline():
            for step_idx in range(num_chunks_bbe):
                bbe_output.copy_(bbe_wrappers[step_idx].run(
                    bbe_q_buffers[step_idx], bbe_k_buffers[step_idx], bbe_v_buffers[step_idx]
                ))
        
        # Display step flow
        if verbose:
            print(f"  Step flow preview:")
            for step_id in range(num_chunks_bbe):
                kv_len = (step_id + 1) * test_chunk_size
                q_offset = step_id * test_chunk_size
                print(f"    Step {step_id}: Q[{q_offset}:{q_offset+test_chunk_size}] attend to KV[0:{kv_len}] (q_offset={q_offset})")
        
        # Warmup
        print(f"  Warmup...")
        for _ in range(warmup_iters):
            run_bbe_pipeline()
        torch.cuda.synchronize()
        
        
        # CUDA Graph capture
        print(f"  Capturing CUDA Graph...")
        bbe_stream = torch.cuda.Stream()
        with torch.cuda.stream(bbe_stream):
            run_bbe_pipeline()
        bbe_stream.synchronize()
        
        bbe_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(bbe_graph, stream=bbe_stream):
            run_bbe_pipeline()
        
        # Warmup with cuda_graph
        for _ in range(warmup_iters):
            bbe_graph.replay()
        torch.cuda.synchronize()
        
        # Benchmark
        print(f"  Benchmark (with cuda_graph)...")
        start = time.perf_counter()
        for _ in range(bench_iters):
            bbe_graph.replay()
        torch.cuda.synchronize()
        bbe_cuda_graph_time = (time.perf_counter() - start) / bench_iters * 1000
        
        results[f"bbe_chunk{test_chunk_size}"] = {
            "time_cuda_graph_ms": bbe_cuda_graph_time,
            "chunk_size": test_chunk_size,
            "num_steps": num_chunks_bbe,
            "method": "BatchBlockExtend Ragged",
        }
        print(f"  => cuda_graph: {bbe_cuda_graph_time:.3f} ms ({bbe_cuda_graph_time/num_chunks_bbe:.3f} ms/step × {num_chunks_bbe} steps)")
        
        del bbe_wrappers, bbe_graph
        torch.cuda.empty_cache()

    # Custom Mask different chunk_size test (chunk_size = dllm_block_size not required)
    for test_chunk_size in chunk_sizes:
        if tokens_per_request % test_chunk_size != 0:
            print(f"\n[Skip] chunk_size={test_chunk_size} cannot divide tokens_per_request={tokens_per_request}")
            continue
        
        # Skip already tested baseline chunk_size
        if test_chunk_size == baseline_chunk_size:
            continue
        
        num_chunks_cm = tokens_per_request // test_chunk_size
        
        print(f"\n{'-'*60}")
        print(f"[Custom Mask] chunk_size = {test_chunk_size}")
        print(f"{'-'*60}")
        print(f"  num_steps: {num_chunks_cm} ({baseline_num_chunks - num_chunks_cm} steps fewer than Baseline)")
        print(f"  Kernels per step: 1 (batch processes all requests)")
        
        # Split each request into chunks
        qs_chunks_cm = [split_chunks(q, test_chunk_size) for q in all_qs]
        
        # Pre-allocate Q buffers (concat all requests)
        cm_q_buffers_var = []
        for step_idx in range(num_chunks_cm):
            q_list = [qs_chunks_cm[req_idx][step_idx] for req_idx in range(num_requests)]
            cm_q_buffers_var.append(torch.cat(q_list, dim=0))
        
        # Pre-allocate cumulative KV buffers (concat all requests)
        cm_k_buffers_var = []
        cm_v_buffers_var = []
        for step_idx in range(num_chunks_cm):
            kv_len = (step_idx + 1) * test_chunk_size
            k_cumul_list = [all_ks[req_idx][:kv_len] for req_idx in range(num_requests)]
            v_cumul_list = [all_vs[req_idx][:kv_len] for req_idx in range(num_requests)]
            cm_k_buffers_var.append(torch.cat(k_cumul_list, dim=0))
            cm_v_buffers_var.append(torch.cat(v_cumul_list, dim=0))
        
        # Construct flattened custom_mask (batch version)
        # DLLM blockwise mask: mask[q, k] = ((q + q_offset) // B) >= (k // B)
        cm_mask_buffers_var = []
        for step_idx in range(num_chunks_cm):
            kv_len = (step_idx + 1) * test_chunk_size
            q_offset = step_idx * test_chunk_size
            # Construct single request's 2D mask: [q_len, kv_len]
            q_pos = torch.arange(test_chunk_size, device=device) + q_offset
            k_pos = torch.arange(kv_len, device=device)
            q_block = q_pos.unsqueeze(1) // dllm_block_size
            k_block = k_pos.unsqueeze(0) // dllm_block_size
            mask_2d = (q_block >= k_block)  # [q_len, kv_len], bool
            # Flatten and repeat for batch
            mask_flat = mask_2d.flatten()
            batch_mask = mask_flat.repeat(num_requests)
            cm_mask_buffers_var.append(batch_mask)
        
        # qo_indptr and kv_indptr
        cm_qo_indptr_var = torch.tensor(
            [i * test_chunk_size for i in range(num_requests + 1)],
            dtype=torch.int32, device=device
        )
        cm_kv_indptr_list_var = []
        for step_idx in range(num_chunks_cm):
            kv_len = (step_idx + 1) * test_chunk_size
            cm_kv_indptr_list_var.append(torch.tensor(
                [i * kv_len for i in range(num_requests + 1)],
                dtype=torch.int32, device=device
            ))
        
        # Create independent wrapper for each step and complete plan
        print(f"  Creating wrappers and completing plan...")
        cm_wrappers_var = []
        for step_idx in range(num_chunks_cm):
            wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device), 
                kv_layout="NHD",
                backend="fa2",  # custom_mask only supported in FA2
            )
            wrapper.plan(
                qo_indptr=cm_qo_indptr_var,
                kv_indptr=cm_kv_indptr_list_var[step_idx],
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                custom_mask=cm_mask_buffers_var[step_idx],
                causal=False,
                sm_scale=sm_scale,
            )
            cm_wrappers_var.append(wrapper)
        
        cm_output_var = torch.empty(num_requests * test_chunk_size, num_heads, head_dim, dtype=dtype, device=device)
        
        def run_cm_pipeline_var():
            for step_idx in range(num_chunks_cm):
                cm_output_var.copy_(cm_wrappers_var[step_idx].run(
                    cm_q_buffers_var[step_idx], cm_k_buffers_var[step_idx], cm_v_buffers_var[step_idx]
                ))

        if verbose:
            print(f"  Step flow preview:")
            for step_id in range(num_chunks_cm):
                kv_len = (step_id + 1) * test_chunk_size
                q_offset = step_id * test_chunk_size
                print(f"    Step {step_id}: Q[{q_offset}:{q_offset+test_chunk_size}] attend to KV[0:{kv_len}]")
        
        # Warmup
        print(f"  Warmup...")
        for _ in range(warmup_iters):
            run_cm_pipeline_var()
        torch.cuda.synchronize()
        
        # CUDA Graph capture
        print(f"  Capturing CUDA Graph...")
        cm_stream_var = torch.cuda.Stream()
        with torch.cuda.stream(cm_stream_var):
            run_cm_pipeline_var()
        cm_stream_var.synchronize()
        
        cm_graph_var = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cm_graph_var, stream=cm_stream_var):
            run_cm_pipeline_var()
        
        # Warmup with cuda_graph
        for _ in range(warmup_iters):
            cm_graph_var.replay()
        torch.cuda.synchronize()
        
        # Benchmark
        print(f"  Benchmark (with cuda_graph)...")
        start = time.perf_counter()
        for _ in range(bench_iters):
            cm_graph_var.replay()
        torch.cuda.synchronize()
        cm_var_time = (time.perf_counter() - start) / bench_iters * 1000
        
        results[f"cm_chunk{test_chunk_size}"] = {
            "time_cuda_graph_ms": cm_var_time,
            "chunk_size": test_chunk_size,
            "num_steps": num_chunks_cm,
            "method": "Custom Mask",
        }
        print(f"  => cuda_graph: {cm_var_time:.3f} ms ({cm_var_time/num_chunks_cm:.3f} ms/step × {num_chunks_cm} steps)")
        
        del cm_graph_var, cm_mask_buffers_var, cm_wrappers_var
        torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print(f"Results Summary (Step-by-step Execution + CUDA Graph)")
    print(f"{'='*80}")
    
    cm_baseline_time = results["custom_mask_baseline"]["time_cuda_graph_ms"]
    cascade_baseline_time = results["cascade_baseline"]["time_cuda_graph_ms"]
    

    print(f"\nNotes:")
    print(f"  - Baseline 1: SGLang Cascade (BatchPrefill + merge_state)")
    print(f"  - Baseline 2: Custom Mask (BatchPrefill + custom_mask)")
    print(f"  - vs Base1: Speedup relative to SGLang Cascade")
    print(f"  - vs Base2: Speedup relative to Custom Mask")
    
    print(f"\n{'Method':<40} | {'chunk':>6} | {'steps':>6} | {'cuda_graph(ms)':>10} | {'ms/step':>10} | {'vs Base1':>10} | {'vs Base2':>10}")
    print(f"{'-'*40}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    
    # Baseline 1: SGLang Cascade
    r = results["cascade_baseline"]
    print(f"{'[Baseline 1] SGLang Cascade':<40} | {r['chunk_size']:>6} | {r['num_steps']:>6} | {r['time_cuda_graph_ms']:>10.3f} | {r['time_cuda_graph_ms']/r['num_steps']:>10.3f} | {'1.00x':>10} | {cm_baseline_time/cascade_baseline_time:>9.2f}x")
    
    # Baseline 2: Custom Mask
    r = results["custom_mask_baseline"]
    print(f"{'[Baseline 2] Custom Mask':<40} | {r['chunk_size']:>6} | {r['num_steps']:>6} | {r['time_cuda_graph_ms']:>10.3f} | {r['time_cuda_graph_ms']/r['num_steps']:>10.3f} | {cascade_baseline_time/cm_baseline_time:>9.2f}x | {'1.00x':>10}")
    

    cm_keys = [k for k in results.keys() if k.startswith("cm_chunk")]
    cm_keys_sorted = sorted(cm_keys, key=lambda k: results[k]["chunk_size"])
    for key in cm_keys_sorted:
        r = results[key]
        speedup_vs_cascade = cascade_baseline_time / r["time_cuda_graph_ms"]
        speedup_vs_cm = cm_baseline_time / r["time_cuda_graph_ms"]
        print(f"{'Custom Mask':<40} | {r['chunk_size']:>6} | {r['num_steps']:>6} | {r['time_cuda_graph_ms']:>10.3f} | {r['time_cuda_graph_ms']/r['num_steps']:>10.3f} | {speedup_vs_cascade:>9.2f}x | {speedup_vs_cm:>9.2f}x")
    

    bbe_keys = [k for k in results.keys() if k.startswith("bbe_chunk")]
    bbe_keys_sorted = sorted(bbe_keys, key=lambda k: results[k]["chunk_size"])
    for key in bbe_keys_sorted:
        r = results[key]
        speedup_vs_cascade = cascade_baseline_time / r["time_cuda_graph_ms"]
        speedup_vs_cm = cm_baseline_time / r["time_cuda_graph_ms"]
        print(f"{'BatchBlockExtend Ragged':<40} | {r['chunk_size']:>6} | {r['num_steps']:>6} | {r['time_cuda_graph_ms']:>10.3f} | {r['time_cuda_graph_ms']/r['num_steps']:>10.3f} | {speedup_vs_cascade:>9.2f}x | {speedup_vs_cm:>9.2f}x")

    
    return results


def test_incremental_singlereq_prefill_step_by_step_with_cuda_graph(
    tokens_per_request: int = 512,
    dllm_block_size: int = 32,
    chunk_sizes: list = None,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    verbose: bool = False,
):
    """
    Single request incremental Prefill step-by-step execution test + CUDA Graph
    
    Scenario description:
      - Incremental prefill of single request
      - Must execute step by step: chunk1 can only be computed after chunk0 completes (pipeline dependency)
      - SGLang uses BatchPrefill interface even with batch_size=1
    
    Baseline (SGLang 3-stage DLLM Cascade):
      1. BatchPrefillWithRaggedKVCacheWrapper: current chunk (causal=False)
      2. BatchPrefillWithRaggedKVCacheWrapper: prefix KV (causal=False)
      3. merge_state: merge results from both parts
      Each step: 2-3 kernel launches
    
    Comparison:
      1. block_extend_attention_with_offset + CUDA Graph
      2. BatchBlockExtendRaggedOffsetWrapper + CUDA Graph
      Each step: 1 kernel launch
    """
    if chunk_sizes is None:
        chunk_sizes = [32, 64, 128, 256]
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    sm_scale = 1.0 / (head_dim ** 0.5)
    
    # Baseline: chunk_size = dllm_block_size (SGLang constraint)
    baseline_chunk_size = dllm_block_size
    num_chunks = tokens_per_request // baseline_chunk_size
    
    print(f"\n{'='*80}")
    print(f"Single Request Incremental Prefill Step-by-step + CUDA Graph")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  tokens_per_request  = {tokens_per_request}")
    print(f"  dllm_block_size     = {dllm_block_size}")
    print(f"  chunk_sizes to test = {chunk_sizes}")
    print(f"  num_heads           = {num_heads}")
    print(f"  num_kv_heads        = {num_kv_heads}")
    print(f"  head_dim            = {head_dim}")
    print(f"\nScenario description:")
    print(f"  - Single request (batch_size=1), but uses BatchPrefill interface (SGLang approach)")
    print(f"  - Step-by-step execution: chunk1 can only be computed after chunk0 completes (pipeline dependency)")
    print(f"  - num_steps = {num_chunks}")
    
    # Data preparation
    # Single request's complete Q, K, V
    q_full = torch.randn(tokens_per_request, num_heads, head_dim, dtype=dtype, device=device)
    k_full = torch.randn(tokens_per_request, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_full = torch.randn(tokens_per_request, num_kv_heads, head_dim, dtype=dtype, device=device)
    
    # Split into chunks
    def split_chunks(tensor, chunk_size):
        return [tensor[i*chunk_size:(i+1)*chunk_size] for i in range(tensor.shape[0] // chunk_size)]
    
    qs_chunks = split_chunks(q_full, baseline_chunk_size)
    ks_chunks = split_chunks(k_full, baseline_chunk_size)
    vs_chunks = split_chunks(v_full, baseline_chunk_size)
    
    # Cumulative KV buffer
    k_cumul_list = [torch.cat([k_full[:(i+1)*baseline_chunk_size]], dim=0) for i in range(num_chunks)]
    v_cumul_list = [torch.cat([v_full[:(i+1)*baseline_chunk_size]], dim=0) for i in range(num_chunks)]
    
    results = {}

    # Correctness verification
    print(f"\n{'='*80}")
    print(f"Correctness Verification")
    print(f"{'='*80}")
    print(f"  Reference implementation: single_prefill_with_kv_cache + custom_mask")
    print(f"  Mask rule: mask[q,k] = ((q + offset) // B) >= (k // B)")
    
    # Compute reference results (each chunk computed independently)
    ref_outputs = []
    for step_idx in range(num_chunks):
        q_offset = step_idx * baseline_chunk_size
        ref_out = compute_block_extend_reference(
            qs_chunks[step_idx],
            k_cumul_list[step_idx],
            v_cumul_list[step_idx],
            dllm_block_size=dllm_block_size,
            q_offset=q_offset,
            sm_scale=sm_scale,
        )
        ref_outputs.append(ref_out)
    
    # Compute V2 results and verify
    v2_outputs = []
    for step_idx in range(num_chunks):
        q_offset = step_idx * baseline_chunk_size
        v2_out = block_extend_attention_with_offset(
            qs_chunks[step_idx],
            k_cumul_list[step_idx],
            v_cumul_list[step_idx],
            dllm_block_size=dllm_block_size,
            q_offset=q_offset,
            sm_scale=sm_scale,
            backend="fa2",
        )
        v2_outputs.append(v2_out)
    
    # Verify V2
    v2_max_diff = max((v2_outputs[i] - ref_outputs[i]).abs().max().item() for i in range(num_chunks))
    tol = 1e-3
    v2_pass = v2_max_diff < tol
    print(f"\n  [V2] block_extend_attention_with_offset:")
    print(f"       max_diff = {v2_max_diff:.6f}, tolerance = {tol}")
    print(f"       {' PASS' if v2_pass else ' FAIL'}")
    
    # Compute BBE results and verify
    bbe_outputs = []
    workspace_verify = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    qo_indptr_verify = torch.tensor([0, baseline_chunk_size], dtype=torch.int32, device=device)
    for step_idx in range(num_chunks):
        kv_len = (step_idx + 1) * baseline_chunk_size
        kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=device)
        q_offset_tensor = torch.tensor([step_idx * baseline_chunk_size], dtype=torch.int32, device=device)
        
        wrapper = BatchBlockExtendRaggedOffsetWrapper(
            workspace_verify, kv_layout="NHD", dllm_block_size=dllm_block_size
        )
        wrapper.plan(
            qo_indptr=qo_indptr_verify,
            kv_indptr=kv_indptr,
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_data_type=dtype,
            sm_scale=sm_scale,
            q_offsets=q_offset_tensor,
        )
        bbe_out = wrapper.run(qs_chunks[step_idx], k_cumul_list[step_idx], v_cumul_list[step_idx])
        bbe_outputs.append(bbe_out)
    
    # Verify BBE
    bbe_max_diff = max((bbe_outputs[i] - ref_outputs[i]).abs().max().item() for i in range(num_chunks))
    bbe_pass = bbe_max_diff < tol
    print(f"\n  [BBE] BatchBlockExtendRaggedOffsetWrapper:")
    print(f"        max_diff = {bbe_max_diff:.6f}, tolerance = {tol}")
    print(f"        {' PASS' if bbe_pass else ' FAIL'}")
    
    if not (v2_pass and bbe_pass):
        print(f"\n   Correctness verification failed, but continue with performance test")
    else:
        print(f"\n   All methods passed correctness verification")
    
    del workspace_verify, bbe_outputs, v2_outputs, ref_outputs
    torch.cuda.empty_cache()
    

    # Baseline 2: Custom Mask (native single_prefill + custom_mask) + CUDA Graph
    print(f"\n{'='*80}")
    print(f"[Baseline 2] Custom Mask (single_prefill + custom_mask)")
    print(f"{'='*80}")
    print(f"  num_steps: {num_chunks}")
    print(f"  Each step: single_prefill_with_kv_cache + custom_mask")
    print(f"  Kernels per step: 1 (but needs to construct mask tensor)")
    

    custom_mask_buffers = []
    for step_idx in range(num_chunks):
        kv_len = (step_idx + 1) * baseline_chunk_size
        q_offset = step_idx * baseline_chunk_size
        # mask[q, k] = ((q + offset) // B) >= (k // B)
        q_pos = torch.arange(baseline_chunk_size, device=device) + q_offset
        k_pos = torch.arange(kv_len, device=device)
        q_block = q_pos.unsqueeze(1) // dllm_block_size
        k_block = k_pos.unsqueeze(0) // dllm_block_size
        mask_2d = (q_block >= k_block).to(torch.uint8)
        custom_mask_buffers.append(mask_2d)
    
    cm_output = torch.empty(baseline_chunk_size, num_heads, head_dim, dtype=dtype, device=device)
    
    # Note: custom_mask is only supported in FA2 backend, not FA3
    def run_custom_mask_pipeline():
        for step_idx in range(num_chunks):
            cm_output.copy_(single_prefill_with_kv_cache(
                qs_chunks[step_idx],
                k_cumul_list[step_idx],
                v_cumul_list[step_idx],
                custom_mask=custom_mask_buffers[step_idx],
                sm_scale=sm_scale,
                backend="fa2",  # custom_mask only supported in FA2
            ))
    

    if verbose:
        print(f"  Step flow preview:")
        for step_id in range(num_chunks):
            kv_len = (step_id + 1) * baseline_chunk_size
            print(f"    Step {step_id}: Q[{step_id*baseline_chunk_size}:{(step_id+1)*baseline_chunk_size}] attend to KV[0:{kv_len}]")
    
    # Warmup
    print(f"  Warmup...")
    for _ in range(warmup_iters):
        run_custom_mask_pipeline()
    torch.cuda.synchronize()
    
    # CUDA Graph capture
    print(f"  Capturing CUDA Graph...")
    cm_stream = torch.cuda.Stream()
    with torch.cuda.stream(cm_stream):
        run_custom_mask_pipeline()
    cm_stream.synchronize()
    
    cm_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cm_graph, stream=cm_stream):
        run_custom_mask_pipeline()
    
    # Warmup with cuda_graph
    for _ in range(warmup_iters):
        cm_graph.replay()
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"  Benchmark (with cuda_graph)...")
    start = time.perf_counter()
    for _ in range(bench_iters):
        cm_graph.replay()
    torch.cuda.synchronize()
    cm_cuda_graph_time = (time.perf_counter() - start) / bench_iters * 1000
    
    results["custom_mask_baseline"] = {
        "time_cuda_graph_ms": cm_cuda_graph_time,
        "chunk_size": baseline_chunk_size,
        "num_steps": num_chunks,
        "method": "Custom Mask (Baseline 2)",
    }
    print(f"  => cuda_graph: {cm_cuda_graph_time:.3f} ms ({cm_cuda_graph_time/num_chunks:.3f} ms/step × {num_chunks} steps)")
    
    del cm_graph, custom_mask_buffers
    torch.cuda.empty_cache()
    

    # Custom Mask different chunk_size test (chunk_size = dllm_block_size not required)
    for test_chunk_size in chunk_sizes:
        if tokens_per_request % test_chunk_size != 0:
            print(f"\n[Skip] chunk_size={test_chunk_size} cannot divide tokens_per_request={tokens_per_request}")
            continue
        
        # Skip already tested baseline chunk_size
        if test_chunk_size == baseline_chunk_size:
            continue
        
        num_steps_cm = tokens_per_request // test_chunk_size
        
        print(f"\n{'-'*60}")
        print(f"[Custom Mask] chunk_size = {test_chunk_size}")
        print(f"{'-'*60}")
        print(f"  num_steps: {num_steps_cm} ({num_chunks - num_steps_cm} steps fewer than Baseline)")
        print(f"  Kernels per step: 1")
        
        # Split into chunks
        qs_cm = split_chunks(q_full, test_chunk_size)
        
        # Cumulative KV buffer
        k_cumul_cm = [k_full[:(i+1)*test_chunk_size].clone() for i in range(num_steps_cm)]
        v_cumul_cm = [v_full[:(i+1)*test_chunk_size].clone() for i in range(num_steps_cm)]
        
        # Pre-allocate custom_mask buffers (different mask for each step)
        cm_mask_buffers = []
        for step_idx in range(num_steps_cm):
            kv_len = (step_idx + 1) * test_chunk_size
            q_offset = step_idx * test_chunk_size
            # mask[q, k] = ((q + offset) // B) >= (k // B)
            q_pos = torch.arange(test_chunk_size, device=device) + q_offset
            k_pos = torch.arange(kv_len, device=device)
            q_block = q_pos.unsqueeze(1) // dllm_block_size
            k_block = k_pos.unsqueeze(0) // dllm_block_size
            mask_2d = (q_block >= k_block).to(torch.uint8)
            cm_mask_buffers.append(mask_2d)
        
        # Pre-allocate output buffer
        cm_output_var = torch.empty(test_chunk_size, num_heads, head_dim, dtype=dtype, device=device)
        
        def run_cm_pipeline():
            for step_idx in range(num_steps_cm):
                cm_output_var.copy_(single_prefill_with_kv_cache(
                    qs_cm[step_idx], k_cumul_cm[step_idx], v_cumul_cm[step_idx],
                    custom_mask=cm_mask_buffers[step_idx],
                    sm_scale=sm_scale,
                    backend="fa2",  # custom_mask only supported in FA2
                ))
        
        # Display step flow
        if verbose:
            print(f"  Step flow preview:")
            for step_id in range(num_steps_cm):
                kv_len = (step_id + 1) * test_chunk_size
                q_offset = step_id * test_chunk_size
                print(f"    Step {step_id}: Q[{q_offset}:{q_offset+test_chunk_size}] attend to KV[0:{kv_len}]")
        
        # Warmup
        print(f"  Warmup...")
        for _ in range(warmup_iters):
            run_cm_pipeline()
        torch.cuda.synchronize()
        
        # CUDA Graph capture
        print(f"  Capturing CUDA Graph...")
        cm_stream_var = torch.cuda.Stream()
        with torch.cuda.stream(cm_stream_var):
            run_cm_pipeline()
        cm_stream_var.synchronize()
        
        cm_graph_var = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cm_graph_var, stream=cm_stream_var):
            run_cm_pipeline()
        
        # Warmup with cuda_graph
        for _ in range(warmup_iters):
            cm_graph_var.replay()
        torch.cuda.synchronize()
        
        # Benchmark
        print(f"  Benchmark (with cuda_graph)...")
        start = time.perf_counter()
        for _ in range(bench_iters):
            cm_graph_var.replay()
        torch.cuda.synchronize()
        cm_var_time = (time.perf_counter() - start) / bench_iters * 1000
        
        results[f"cm_chunk{test_chunk_size}"] = {
            "time_cuda_graph_ms": cm_var_time,
            "chunk_size": test_chunk_size,
            "num_steps": num_steps_cm,
            "method": "Custom Mask",
        }
        print(f"  => cuda_graph: {cm_var_time:.3f} ms ({cm_var_time/num_steps_cm:.3f} ms/step × {num_steps_cm} steps)")
        
        del cm_graph_var, cm_mask_buffers
        torch.cuda.empty_cache()

    # Baseline 1: SGLang 3-stage DLLM Cascade (BatchPrefill interface, batch_size=1)
    print(f"\n{'='*80}")
    print(f"[Baseline 1] SGLang 3-stage Cascade (BatchPrefill, batch_size=1)")
    print(f"{'='*80}")
    print(f"  num_steps: {num_chunks}")
    print(f"  Each step: BatchPrefill(current chunk) + BatchPrefill(prefix) + merge_state")
    print(f"  Kernels per step: 2-3")
    
    # indptr for batch_size=1
    qo_indptr_chunk = torch.tensor([0, baseline_chunk_size], dtype=torch.int32, device=device)
    
    # workspace size:
    workspace_size = 16 * 1024 * 1024
    
    # Create independent wrapper for each step and complete plan
    print(f"  Creating wrappers and completing plan...")
    cascade_wrappers_current = []
    cascade_wrappers_prefix = []
    kv_indptr_prefix_list = [None]  # step 0 has no prefix
    
    for step_idx in range(num_chunks):
        # Wrapper for current chunk
        wrapper_current = BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(workspace_size, dtype=torch.uint8, device=device), kv_layout="NHD", backend="fa3",
        )
        wrapper_current.plan(
            qo_indptr=qo_indptr_chunk,
            kv_indptr=qo_indptr_chunk,  # current chunk KV length = Q length
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            causal=False,
            sm_scale=sm_scale,
        )
        cascade_wrappers_current.append(wrapper_current)
        
        # Wrapper for prefix (step 0 has no prefix)
        if step_idx == 0:
            cascade_wrappers_prefix.append(None)
        else:
            prefix_len = step_idx * baseline_chunk_size
            kv_indptr_prefix = torch.tensor([0, prefix_len], dtype=torch.int32, device=device)
            kv_indptr_prefix_list.append(kv_indptr_prefix)
            
            wrapper_prefix = BatchPrefillWithRaggedKVCacheWrapper(
                torch.empty(workspace_size, dtype=torch.uint8, device=device), kv_layout="NHD", backend="fa3",
            )
            wrapper_prefix.plan(
                qo_indptr=qo_indptr_chunk,
                kv_indptr=kv_indptr_prefix,
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                causal=False,
                sm_scale=sm_scale,
            )
            cascade_wrappers_prefix.append(wrapper_prefix)
    
    # Pre-allocate buffer
    cascade_output = torch.empty(baseline_chunk_size, num_heads, head_dim, dtype=dtype, device=device)
    o1_buffer = torch.empty(baseline_chunk_size, num_heads, head_dim, dtype=dtype, device=device)
    s1_buffer = torch.empty(baseline_chunk_size, num_heads, dtype=torch.float32, device=device)
    o2_buffer = torch.empty(baseline_chunk_size, num_heads, head_dim, dtype=dtype, device=device)
    s2_buffer = torch.empty(baseline_chunk_size, num_heads, dtype=torch.float32, device=device)
    
    # Pre-allocate prefix KV buffer
    k_prefix_buffers = [None]
    v_prefix_buffers = [None]
    for step_idx in range(1, num_chunks):
        prefix_len = step_idx * baseline_chunk_size
        k_prefix_buffers.append(k_full[:prefix_len].clone())
        v_prefix_buffers.append(v_full[:prefix_len].clone())
    
    def run_cascade_pipeline():
        for step_idx in range(num_chunks):
            q_chunk = qs_chunks[step_idx]
            k_chunk = ks_chunks[step_idx]
            v_chunk = vs_chunks[step_idx]
            
            if step_idx == 0:
                cascade_output.copy_(cascade_wrappers_current[step_idx].run(q_chunk, k_chunk, v_chunk))
            else:
                o1, s1 = cascade_wrappers_current[step_idx].run_return_lse(q_chunk, k_chunk, v_chunk)
                o1_buffer.copy_(o1)
                s1_buffer.copy_(s1)
                
                o2, s2 = cascade_wrappers_prefix[step_idx].run_return_lse(
                    q_chunk, k_prefix_buffers[step_idx], v_prefix_buffers[step_idx]
                )
                o2_buffer.copy_(o2)
                s2_buffer.copy_(s2)
                
                o, _ = merge_state(o1_buffer, s1_buffer, o2_buffer, s2_buffer)
                cascade_output.copy_(o)
    

    if verbose:
        print(f"  Step flow preview:")
        for step_id in range(num_chunks):
            prefix_len = step_id * baseline_chunk_size
            if step_id == 0:
                print(f"    Step {step_id}: current_chunk[{baseline_chunk_size}] only (no prefix)")
            else:
                print(f"    Step {step_id}: current_chunk[{baseline_chunk_size}] + prefix[{prefix_len}] + merge")
    
    # Warmup
    print(f"  Warmup...")
    for _ in range(warmup_iters):
        run_cascade_pipeline()
    torch.cuda.synchronize()
    
    # CUDA Graph capture
    print(f"  Capturing CUDA Graph...")
    cascade_stream = torch.cuda.Stream()
    with torch.cuda.stream(cascade_stream):
        run_cascade_pipeline()
    cascade_stream.synchronize()
    
    cascade_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cascade_graph, stream=cascade_stream):
        run_cascade_pipeline()
    
    # Warmup with cuda_graph
    for _ in range(warmup_iters):
        cascade_graph.replay()
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"  Benchmark (with cuda_graph)...")
    start = time.perf_counter()
    for _ in range(bench_iters):
        cascade_graph.replay()
    torch.cuda.synchronize()
    cascade_cuda_graph_time = (time.perf_counter() - start) / bench_iters * 1000
    
    results["cascade_baseline"] = {
        "time_cuda_graph_ms": cascade_cuda_graph_time,
        "chunk_size": baseline_chunk_size,
        "num_steps": num_chunks,
        "method": "SGLang Cascade (Baseline 1)",
    }
    print(f"  => cuda_graph: {cascade_cuda_graph_time:.3f} ms ({cascade_cuda_graph_time/num_chunks:.3f} ms/step × {num_chunks} steps)")
    
    del cascade_wrappers_current, cascade_wrappers_prefix, cascade_graph
    torch.cuda.empty_cache()

    # Method 1: block_extend_attention_with_offset + CUDA Graph
    for test_chunk_size in chunk_sizes:
        if tokens_per_request % test_chunk_size != 0:
            print(f"\n[Skip] chunk_size={test_chunk_size} cannot divide tokens_per_request={tokens_per_request}")
            continue
        
        num_steps = tokens_per_request // test_chunk_size
        
        print(f"\n{'-'*60}")
        print(f"[V2] block_extend_attention_with_offset (chunk_size={test_chunk_size})")
        print(f"{'-'*60}")
        print(f"  num_steps: {num_steps} ({num_chunks - num_steps} steps fewer than Baseline)")
        print(f"  Kernels per step: 1")
        
        # Split into chunks
        qs_v2 = split_chunks(q_full, test_chunk_size)
        
        # Pre-allocate output buffer
        v2_output = torch.empty(test_chunk_size, num_heads, head_dim, dtype=dtype, device=device)
        
        # Cumulative KV buffer
        k_cumul_v2 = [k_full[:(i+1)*test_chunk_size].clone() for i in range(num_steps)]
        v_cumul_v2 = [v_full[:(i+1)*test_chunk_size].clone() for i in range(num_steps)]
        
        def run_v2_pipeline():
            for step_idx in range(num_steps):
                v2_output.copy_(block_extend_attention_with_offset(
                    qs_v2[step_idx], k_cumul_v2[step_idx], v_cumul_v2[step_idx],
                    dllm_block_size=dllm_block_size,
                    q_offset=step_idx * test_chunk_size,
                    sm_scale=sm_scale,
                    backend="fa2",
                ))
        
        # Display step flow
        if verbose:
            print(f"  Step flow preview:")
            for step_id in range(num_steps):
                kv_len = (step_id + 1) * test_chunk_size
                q_offset = step_id * test_chunk_size
                print(f"    Step {step_id}: Q[{q_offset}:{q_offset+test_chunk_size}] attend to KV[0:{kv_len}] (q_offset={q_offset})")
        
        # Warmup
        print(f"  Warmup...")
        for _ in range(warmup_iters):
            run_v2_pipeline()
        torch.cuda.synchronize()
        
        # CUDA Graph capture
        print(f"  Capturing CUDA Graph...")
        v2_stream = torch.cuda.Stream()
        with torch.cuda.stream(v2_stream):
            run_v2_pipeline()
        v2_stream.synchronize()
        
        v2_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(v2_graph, stream=v2_stream):
            run_v2_pipeline()
        
        # Warmup with cuda_graph
        for _ in range(warmup_iters):
            v2_graph.replay()
        torch.cuda.synchronize()
        
        # Benchmark
        print(f"  Benchmark (with cuda_graph)...")
        start = time.perf_counter()
        for _ in range(bench_iters):
            v2_graph.replay()
        torch.cuda.synchronize()
        v2_cuda_graph_time = (time.perf_counter() - start) / bench_iters * 1000
        
        results[f"v2_chunk{test_chunk_size}"] = {
            "time_cuda_graph_ms": v2_cuda_graph_time,
            "chunk_size": test_chunk_size,
            "num_steps": num_steps,
            "method": "V2 + CUDA Graph",
        }
        print(f"  => cuda_graph: {v2_cuda_graph_time:.3f} ms ({v2_cuda_graph_time/num_steps:.3f} ms/step × {num_steps} steps)")
        
        del v2_graph
        torch.cuda.empty_cache()
    

    # Method 2: BatchBlockExtendRaggedOffsetWrapper + CUDA Graph
    for test_chunk_size in chunk_sizes:
        if tokens_per_request % test_chunk_size != 0:
            continue
        
        num_steps = tokens_per_request // test_chunk_size
        
        print(f"\n{'-'*60}")
        print(f"[BBE] BatchBlockExtendRaggedOffsetWrapper (chunk_size={test_chunk_size})")
        print(f"{'-'*60}")
        print(f"  num_steps: {num_steps} ({num_chunks - num_steps} steps fewer than Baseline)")
        print(f"  Kernels per step: 1")
        
        # Split into chunks
        qs_bbe = split_chunks(q_full, test_chunk_size)
        
        # indptr for batch_size=1
        qo_indptr_bbe = torch.tensor([0, test_chunk_size], dtype=torch.int32, device=device)
        
        # Cumulative KV buffer
        k_cumul_bbe = [k_full[:(i+1)*test_chunk_size].clone() for i in range(num_steps)]
        v_cumul_bbe = [v_full[:(i+1)*test_chunk_size].clone() for i in range(num_steps)]
        
        # Create independent wrapper for each step and complete plan
        print(f"  Creating wrappers and completing plan...")
        
        workspace_size = 128 * 1024 * 1024  # 128MB - BBE uses JIT which requires larger workspace
        bbe_wrappers = []
        for step_idx in range(num_steps):
            kv_len = (step_idx + 1) * test_chunk_size
            kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=device)
            q_offset = torch.tensor([step_idx * test_chunk_size], dtype=torch.int32, device=device)
            
            wrapper = BatchBlockExtendRaggedOffsetWrapper(
                torch.empty(workspace_size, dtype=torch.uint8, device=device),
                kv_layout="NHD", dllm_block_size=dllm_block_size
            )
            wrapper.plan(
                qo_indptr=qo_indptr_bbe,
                kv_indptr=kv_indptr,
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_data_type=dtype,
                sm_scale=sm_scale,
                q_offsets=q_offset,
            )
            bbe_wrappers.append(wrapper)
        
        # Pre-allocate output buffer
        bbe_output = torch.empty(test_chunk_size, num_heads, head_dim, dtype=dtype, device=device)
        
        def run_bbe_pipeline():
            for step_idx in range(num_steps):
                bbe_output.copy_(bbe_wrappers[step_idx].run(
                    qs_bbe[step_idx], k_cumul_bbe[step_idx], v_cumul_bbe[step_idx]
                ))
        
        # Display step flow
        if verbose:
            print(f"  Step flow preview:")
            for step_id in range(num_steps):
                kv_len = (step_id + 1) * test_chunk_size
                q_offset = step_id * test_chunk_size
                print(f"    Step {step_id}: Q[{q_offset}:{q_offset+test_chunk_size}] attend to KV[0:{kv_len}] (q_offset={q_offset})")
        
        # Warmup
        print(f"  Warmup...")
        for _ in range(warmup_iters):
            run_bbe_pipeline()
        torch.cuda.synchronize()
        
        # CUDA Graph capture
        print(f"  Capturing CUDA Graph...")
        bbe_stream = torch.cuda.Stream()
        with torch.cuda.stream(bbe_stream):
            run_bbe_pipeline()
        bbe_stream.synchronize()
        
        bbe_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(bbe_graph, stream=bbe_stream):
            run_bbe_pipeline()
        
        # Warmup with cuda_graph
        for _ in range(warmup_iters):
            bbe_graph.replay()
        torch.cuda.synchronize()
        
        # Benchmark
        print(f"  Benchmark (with cuda_graph)...")
        start = time.perf_counter()
        for _ in range(bench_iters):
            bbe_graph.replay()
        torch.cuda.synchronize()
        bbe_cuda_graph_time = (time.perf_counter() - start) / bench_iters * 1000
        
        results[f"bbe_chunk{test_chunk_size}"] = {
            "time_cuda_graph_ms": bbe_cuda_graph_time,
            "chunk_size": test_chunk_size,
            "num_steps": num_steps,
            "method": "BBE Ragged + CUDA Graph",
        }
        print(f"  => cuda_graph: {bbe_cuda_graph_time:.3f} ms ({bbe_cuda_graph_time/num_steps:.3f} ms/step × {num_steps} steps)")
        
        del bbe_wrappers, bbe_graph
        torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print(f"Results Summary (Single Request Step-by-step + CUDA Graph)")
    print(f"{'='*80}")
    
    cm_baseline_r = results.get("custom_mask_baseline", {})
    cascade_baseline_r = results.get("cascade_baseline", {})
    cascade_skipped = cascade_baseline_r.get("skipped", False)
    
    cm_baseline_time = cm_baseline_r.get("time_cuda_graph_ms", float('nan'))
    cascade_baseline_time = cascade_baseline_r.get("time_cuda_graph_ms", float('nan'))
    
    # Header explanation
    print(f"\nNotes:")
    print(f"  - Baseline 1: SGLang Cascade (BatchPrefill + merge_state)")
    print(f"  - Baseline 2: Custom Mask (single_prefill + custom_mask)")
    print(f"  - vs Base1: Speedup relative to SGLang Cascade")
    print(f"  - vs Base2: Speedup relative to Custom Mask")
    
    print(f"\n{'Method':<45} | {'chunk':>6} | {'steps':>6} | {'cuda_graph(ms)':>10} | {'ms/step':>10} | {'vs Base1':>10} | {'vs Base2':>10}")
    print(f"{'-'*45}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    
    # Baseline 1: SGLang Cascade
    if cascade_skipped:
        print(f"{'[Baseline 1] SGLang Cascade':<45} | {cascade_baseline_r['chunk_size']:>6} | {cascade_baseline_r['num_steps']:>6} | {'SKIPPED':>10} | {'-':>10} | {'-':>10} | {'-':>10}")
    else:
        r = cascade_baseline_r
        print(f"{'[Baseline 1] SGLang Cascade':<45} | {r['chunk_size']:>6} | {r['num_steps']:>6} | {r['time_cuda_graph_ms']:>10.3f} | {r['time_cuda_graph_ms']/r['num_steps']:>10.3f} | {'1.00x':>10} | {cm_baseline_time/cascade_baseline_time:>9.2f}x")
    
    # Baseline 2: Custom Mask
    r = cm_baseline_r
    if cascade_skipped:
        vs_base1_str = "-"
    else:
        vs_base1_str = f"{cascade_baseline_time/cm_baseline_time:>9.2f}x"
    print(f"{'[Baseline 2] Custom Mask':<45} | {r['chunk_size']:>6} | {r['num_steps']:>6} | {r['time_cuda_graph_ms']:>10.3f} | {r['time_cuda_graph_ms']/r['num_steps']:>10.3f} | {vs_base1_str:>10} | {'1.00x':>10}")
    
    # Custom Mask different chunk_size results (sorted by chunk_size ascending)
    cm_keys = sorted([k for k in results.keys() if k.startswith("cm_chunk")],
                     key=lambda k: results[k]["chunk_size"])
    for key in cm_keys:
        r = results[key]
        speedup_vs_cm = cm_baseline_time / r["time_cuda_graph_ms"]
        if cascade_skipped:
            speedup_vs_cascade_str = "-"
        else:
            speedup_vs_cascade = cascade_baseline_time / r["time_cuda_graph_ms"]
            speedup_vs_cascade_str = f"{speedup_vs_cascade:>9.2f}x"
        print(f"{'Custom Mask':<45} | {r['chunk_size']:>6} | {r['num_steps']:>6} | {r['time_cuda_graph_ms']:>10.3f} | {r['time_cuda_graph_ms']/r['num_steps']:>10.3f} | {speedup_vs_cascade_str:>10} | {speedup_vs_cm:>9.2f}x")
    
    # V2 results (sorted by chunk_size ascending)
    v2_keys = sorted([k for k in results.keys() if k.startswith("v2_chunk")], 
                     key=lambda k: results[k]["chunk_size"])
    for key in v2_keys:
        r = results[key]
        speedup_vs_cm = cm_baseline_time / r["time_cuda_graph_ms"]
        if cascade_skipped:
            speedup_vs_cascade_str = "-"
        else:
            speedup_vs_cascade = cascade_baseline_time / r["time_cuda_graph_ms"]
            speedup_vs_cascade_str = f"{speedup_vs_cascade:>9.2f}x"
        print(f"{'V2 block_extend_attention':<45} | {r['chunk_size']:>6} | {r['num_steps']:>6} | {r['time_cuda_graph_ms']:>10.3f} | {r['time_cuda_graph_ms']/r['num_steps']:>10.3f} | {speedup_vs_cascade_str:>10} | {speedup_vs_cm:>9.2f}x")
    
    # BBE results (sorted by chunk_size ascending)
    bbe_keys = sorted([k for k in results.keys() if k.startswith("bbe_chunk")],
                      key=lambda k: results[k]["chunk_size"])
    for key in bbe_keys:
        r = results[key]
        speedup_vs_cm = cm_baseline_time / r["time_cuda_graph_ms"]
        if cascade_skipped:
            speedup_vs_cascade_str = "-"
        else:
            speedup_vs_cascade = cascade_baseline_time / r["time_cuda_graph_ms"]
            speedup_vs_cascade_str = f"{speedup_vs_cascade:>9.2f}x"
        print(f"{'BBE BatchBlockExtendRagged':<45} | {r['chunk_size']:>6} | {r['num_steps']:>6} | {r['time_cuda_graph_ms']:>10.3f} | {r['time_cuda_graph_ms']/r['num_steps']:>10.3f} | {speedup_vs_cascade_str:>10} | {speedup_vs_cm:>9.2f}x")

    
    return results


def test_fa2_fa3_block_extending_vs_causal(
    num_requests: int = 4,
    chunk_sizes: list = None,
    tokens_per_request: int = 2048,
    dllm_block_size: int = 32,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    verbose: bool = False,
):
    """
    FA2 vs FA3 BlockExtend Mask vs FA3 Causal Mask Performance Comparison
    
    Test scenario (Incremental Prefill):
      - Each request's total tokens is fixed (tokens_per_request)
      - Execute incremental prefill in multiple steps by chunk_size
      - Each step accumulates KV, implementing Block Extend Mask
      - Use CUDA Graph to reduce kernel launch overhead
    
    Comparison methods:
      1. FA3 Causal Mask (baseline) - BatchPrefillWithRaggedKVCacheWrapper
      2. FA2 BlockExtend Mask - BatchBlockExtendRaggedOffsetWrapper (backend="fa2")
      3. FA3 BlockExtend Mask - BatchBlockExtendRaggedOffsetWrapper (backend="fa3")
    
    Mask rules:
      - Causal: mask[q, k] = (q + q_offset) >= k
      - BlockExtend: mask[q, k] = ((q + q_offset) // B) >= (k // B)
    """
    if chunk_sizes is None:
        chunk_sizes = [32, 64, 128, 256]
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    sm_scale = 1.0 / (head_dim ** 0.5)
    
    print(f"\n{'='*90}")
    print(f"FA2 vs FA3 BlockExtend vs Causal - Incremental Prefill Performance Comparison")
    print(f"{'='*90}")
    print(f"Configuration:")
    print(f"  num_requests        = {num_requests}")
    print(f"  tokens_per_request  = {tokens_per_request}")
    print(f"  dllm_block_size     = {dllm_block_size}")
    print(f"  chunk_sizes         = {chunk_sizes}")
    print(f"  num_heads           = {num_heads}")
    print(f"  num_kv_heads        = {num_kv_heads}")
    print(f"  head_dim            = {head_dim}")
    print(f"\nScenario description:")
    print(f"  - Fixed total tokens = {tokens_per_request}")
    print(f"  - Execute incremental prefill in multiple steps by chunk_size")
    print(f"  - Use CUDA Graph to reduce overhead")
    
    # Data preparation, generate complete Q, K, V for each request
    all_qs = [torch.randn(tokens_per_request, num_heads, head_dim, dtype=dtype, device=device) 
              for _ in range(num_requests)]
    all_ks = [torch.randn(tokens_per_request, num_kv_heads, head_dim, dtype=dtype, device=device) 
              for _ in range(num_requests)]
    all_vs = [torch.randn(tokens_per_request, num_kv_heads, head_dim, dtype=dtype, device=device) 
              for _ in range(num_requests)]
    
    def split_chunks(tensor, chunk_size):
        return [tensor[i*chunk_size:(i+1)*chunk_size] for i in range(tensor.shape[0] // chunk_size)]
    
    results = {}
    
    for chunk_size in chunk_sizes:
        if tokens_per_request % chunk_size != 0:
            print(f"\n[Skip] chunk_size={chunk_size} cannot divide tokens_per_request={tokens_per_request}")
            continue
        
        num_steps = tokens_per_request // chunk_size
        
        print(f"\n{'-'*90}")
        print(f"chunk_size = {chunk_size}, num_steps = {num_steps}")
        print(f"{'-'*90}")
        
        # Split each request into chunks
        qs_chunks = [split_chunks(q, chunk_size) for q in all_qs]
        
        # Pre-allocate all step buffers
        # Q buffers: Q concat for each step
        q_buffers = []
        for step_idx in range(num_steps):
            q_list = [qs_chunks[req_idx][step_idx] for req_idx in range(num_requests)]
            q_buffers.append(torch.cat(q_list, dim=0))
        
        # KV buffers: cumulative K, V
        k_buffers = []
        v_buffers = []
        for step_idx in range(num_steps):
            kv_len = (step_idx + 1) * chunk_size
            k_cumul_list = [all_ks[req_idx][:kv_len] for req_idx in range(num_requests)]
            v_cumul_list = [all_vs[req_idx][:kv_len] for req_idx in range(num_requests)]
            k_buffers.append(torch.cat(k_cumul_list, dim=0))
            v_buffers.append(torch.cat(v_cumul_list, dim=0))
        
        # indptrs
        qo_indptr = torch.tensor(
            [i * chunk_size for i in range(num_requests + 1)],
            dtype=torch.int32, device=device
        )
        kv_indptr_list = []
        q_offsets_list = []
        for step_idx in range(num_steps):
            kv_len = (step_idx + 1) * chunk_size
            kv_indptr_list.append(torch.tensor(
                [i * kv_len for i in range(num_requests + 1)],
                dtype=torch.int32, device=device
            ))
            q_offset = step_idx * chunk_size
            q_offsets_list.append(torch.full((num_requests,), q_offset, dtype=torch.int32, device=device))
        
        workspace_size = 256 * 1024 * 1024
        output_buffer = torch.empty(num_requests * chunk_size, num_heads, head_dim, dtype=dtype, device=device)

        # [0] Precision verification: FA2 vs FA3 BlockExtend
        print(f"  [Precision verification] Comparing FA2 vs FA3 BlockExtend output...")
        
        # Create temporary wrappers for precision verification
        fa2_verify_wrapper = BatchBlockExtendRaggedOffsetWrapper(
            torch.empty(workspace_size, dtype=torch.uint8, device=device),
            kv_layout="NHD",
            dllm_block_size=dllm_block_size,
            backend="fa2",
        )
        fa3_verify_wrapper = BatchBlockExtendRaggedOffsetWrapper(
            torch.empty(workspace_size, dtype=torch.uint8, device=device),
            kv_layout="NHD",
            dllm_block_size=dllm_block_size,
            backend="fa3",
        )
        
        # Verify precision for each step
        max_diff_all_steps = 0.0
        for step_idx in range(num_steps):
            fa2_verify_wrapper.plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr_list[step_idx],
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_data_type=dtype,
                sm_scale=sm_scale,
                q_offsets=q_offsets_list[step_idx],
            )
            fa3_verify_wrapper.plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr_list[step_idx],
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_data_type=dtype,
                sm_scale=sm_scale,
                q_offsets=q_offsets_list[step_idx],
            )
            
            fa2_out = fa2_verify_wrapper.run(q_buffers[step_idx], k_buffers[step_idx], v_buffers[step_idx])
            fa3_out = fa3_verify_wrapper.run(q_buffers[step_idx], k_buffers[step_idx], v_buffers[step_idx])
            
            step_max_diff = (fa2_out - fa3_out).abs().max().item()
            max_diff_all_steps = max(max_diff_all_steps, step_max_diff)
            
            if verbose:
                print(f"    step {step_idx}: max_diff = {step_max_diff:.6f}")
        
        # fp16 precision ~0.001 difference is normal
        precision_ok = max_diff_all_steps < 0.01
        status = "PASS" if precision_ok else " FAIL"
        print(f"  [Precision verification] FA2 vs FA3 max_diff = {max_diff_all_steps:.6f} {status}")
        
        if not precision_ok:
            print(f"  [Warning] FA2 and FA3 BlockExtend output difference too large, performance data may not be reliable!")
        
        del fa2_verify_wrapper, fa3_verify_wrapper
        torch.cuda.empty_cache()

        # [1] FA3 Causal Mask (baseline)
        print(f"  [FA3 Causal] Creating wrappers...")
        causal_wrappers = []
        for step_idx in range(num_steps):
            wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                torch.empty(workspace_size, dtype=torch.uint8, device=device),
                kv_layout="NHD",
                backend="fa3",
            )
            wrapper.plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr_list[step_idx],
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                causal=True,
                sm_scale=sm_scale,
            )
            causal_wrappers.append(wrapper)
        
        def run_causal_pipeline():
            for step_idx in range(num_steps):
                output_buffer.copy_(causal_wrappers[step_idx].run(
                    q_buffers[step_idx], k_buffers[step_idx], v_buffers[step_idx]
                ))
        
        # Warmup
        for _ in range(warmup_iters):
            run_causal_pipeline()
        torch.cuda.synchronize()
        
        # CUDA Graph capture
        causal_stream = torch.cuda.Stream()
        with torch.cuda.stream(causal_stream):
            run_causal_pipeline()
        causal_stream.synchronize()
        
        causal_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(causal_graph, stream=causal_stream):
            run_causal_pipeline()
        
        # Warmup with cuda_graph
        for _ in range(warmup_iters):
            causal_graph.replay()
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(bench_iters):
            causal_graph.replay()
        torch.cuda.synchronize()
        fa3_causal_time = (time.perf_counter() - start) / bench_iters * 1000
        
        print(f"  [FA3 Causal]       {fa3_causal_time:.3f} ms ({fa3_causal_time/num_steps:.3f} ms/step × {num_steps} steps)")
        
        del causal_wrappers, causal_graph
        torch.cuda.empty_cache()

        # [2] FA2 BlockExtend Mask
        print(f"  [FA2 BlockExp] Creating wrappers...")
        fa2_be_wrappers = []
        for step_idx in range(num_steps):
            wrapper = BatchBlockExtendRaggedOffsetWrapper(
                torch.empty(workspace_size, dtype=torch.uint8, device=device),
                kv_layout="NHD",
                dllm_block_size=dllm_block_size,
                backend="fa2",
            )
            wrapper.plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr_list[step_idx],
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_data_type=dtype,
                sm_scale=sm_scale,
                q_offsets=q_offsets_list[step_idx],
            )
            fa2_be_wrappers.append(wrapper)
        
        def run_fa2_be_pipeline():
            for step_idx in range(num_steps):
                output_buffer.copy_(fa2_be_wrappers[step_idx].run(
                    q_buffers[step_idx], k_buffers[step_idx], v_buffers[step_idx]
                ))
        
        # Warmup
        for _ in range(warmup_iters):
            run_fa2_be_pipeline()
        torch.cuda.synchronize()
        
        # CUDA Graph capture
        fa2_stream = torch.cuda.Stream()
        with torch.cuda.stream(fa2_stream):
            run_fa2_be_pipeline()
        fa2_stream.synchronize()
        
        fa2_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(fa2_graph, stream=fa2_stream):
            run_fa2_be_pipeline()
        
        # Warmup with cuda_graph
        for _ in range(warmup_iters):
            fa2_graph.replay()
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(bench_iters):
            fa2_graph.replay()
        torch.cuda.synchronize()
        fa2_be_time = (time.perf_counter() - start) / bench_iters * 1000
        
        speedup_fa2_vs_causal = fa3_causal_time / fa2_be_time
        print(f"  [FA2 BlockExp]     {fa2_be_time:.3f} ms ({fa2_be_time/num_steps:.3f} ms/step, {speedup_fa2_vs_causal:.2f}x vs Causal)")
        
        del fa2_be_wrappers, fa2_graph
        torch.cuda.empty_cache()
        
        # ================================================================
        # [3] FA3 BlockExtend Mask
        # ================================================================
        print(f"  [FA3 BlockExp] Creating wrappers...")
        fa3_be_wrappers = []
        for step_idx in range(num_steps):
            wrapper = BatchBlockExtendRaggedOffsetWrapper(
                torch.empty(workspace_size, dtype=torch.uint8, device=device),
                kv_layout="NHD",
                dllm_block_size=dllm_block_size,
                backend="fa3",
            )
            wrapper.plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr_list[step_idx],
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_data_type=dtype,
                sm_scale=sm_scale,
                q_offsets=q_offsets_list[step_idx],
            )
            fa3_be_wrappers.append(wrapper)
        
        def run_fa3_be_pipeline():
            for step_idx in range(num_steps):
                output_buffer.copy_(fa3_be_wrappers[step_idx].run(
                    q_buffers[step_idx], k_buffers[step_idx], v_buffers[step_idx]
                ))
        
        # Warmup
        for _ in range(warmup_iters):
            run_fa3_be_pipeline()
        torch.cuda.synchronize()
        
        # CUDA Graph capture
        fa3_stream = torch.cuda.Stream()
        with torch.cuda.stream(fa3_stream):
            run_fa3_be_pipeline()
        fa3_stream.synchronize()
        
        fa3_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(fa3_graph, stream=fa3_stream):
            run_fa3_be_pipeline()
        
        # Warmup with cuda_graph
        for _ in range(warmup_iters):
            fa3_graph.replay()
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(bench_iters):
            fa3_graph.replay()
        torch.cuda.synchronize()
        fa3_be_time = (time.perf_counter() - start) / bench_iters * 1000
        
        speedup_fa3_vs_causal = fa3_causal_time / fa3_be_time
        speedup_fa3_vs_fa2 = fa2_be_time / fa3_be_time
        print(f"  [FA3 BlockExp]     {fa3_be_time:.3f} ms ({fa3_be_time/num_steps:.3f} ms/step, {speedup_fa3_vs_causal:.2f}x vs Causal, {speedup_fa3_vs_fa2:.2f}x vs FA2)")
        
        results[f"chunk{chunk_size}"] = {
            "chunk_size": chunk_size,
            "num_steps": num_steps,
            "fa2_fa3_max_diff": max_diff_all_steps,
            "precision_ok": precision_ok,
            "fa3_causal_ms": fa3_causal_time,
            "fa2_be_ms": fa2_be_time,
            "fa3_be_ms": fa3_be_time,
            "speedup_fa2_vs_causal": speedup_fa2_vs_causal,
            "speedup_fa3_vs_causal": speedup_fa3_vs_causal,
            "speedup_fa3_vs_fa2": speedup_fa3_vs_fa2,
        }
        
        del fa3_be_wrappers, fa3_graph
        torch.cuda.empty_cache()
    

    print(f"\n{'='*90}")
    print(f"Results Summary (num_requests={num_requests}, tokens_per_request={tokens_per_request}, dllm_block_size={dllm_block_size})")
    print(f"{'='*90}")
    
    print(f"\n{'chunk':>8} | {'steps':>5} | {'FA3 Causal':>12} | {'FA2 BlockExp':>12} | {'FA3 BlockExp':>12} | {'FA2/Causal':>10} | {'FA3/Causal':>10} | {'FA3/FA2':>10}")
    print(f"{'-'*8}-+-{'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    
    for key in sorted(results.keys(), key=lambda k: results[k]["chunk_size"]):
        r = results[key]
        print(f"{r['chunk_size']:>8} | {r['num_steps']:>5} | {r['fa3_causal_ms']:>10.3f}ms | {r['fa2_be_ms']:>10.3f}ms | {r['fa3_be_ms']:>10.3f}ms | {r['speedup_fa2_vs_causal']:>9.2f}x | {r['speedup_fa3_vs_causal']:>9.2f}x | {r['speedup_fa3_vs_fa2']:>9.2f}x")
    
    print(f"\nNotes:")
    print(f"  - Scenario: Incremental Prefill, fixed total tokens = {tokens_per_request}, execute in multiple steps by chunk_size")
    print(f"  - FA2/Causal: Speedup of FA2 BlockExtend relative to FA3 Causal (>1 means FA2 BE is faster)")
    print(f"  - FA3/Causal: Speedup of FA3 BlockExtend relative to FA3 Causal (>1 means FA3 BE is faster)")
    print(f"  - FA3/FA2: Speedup of FA3 BlockExtend relative to FA2 BlockExtend (>1 means FA3 is faster)")
    print(f"  - BlockExtend mask has less computation than Causal mask (tile-level skip), should theoretically be faster")
    
    return results


def test_dllm_precision_vs_custom_mask_fa2(
    verbose: bool = True,
    test_dtypes: list = None,
):
    """
    DLLM Component Precision Test: Comparison with native Custom Mask FA2 implementation
    
    Reference implementation (Ground Truth):
      - Single request: single_prefill_with_kv_cache + custom_mask (FA2)
      - Multi-request: BatchPrefillWithRaggedKVCacheWrapper + custom_mask (FA2)
    
    Tested components (three DLLM components imported at lines 31-33):
      1. BatchBlockExtendRaggedOffsetWrapper
      2. BatchBlockExtendPagedOffsetWrapper  
      3. block_extend_attention_with_offset
    
    Test coverage:
      - Data types: fp16, bf16
      - Different dllm_block_size: [16, 32, 64, 128]
      - Different qo_len: [32, 64, 128, 256]
      - Different kv_len: [64, 128, 256, 512, 1024]
      - Different q_offset: [0, 32, 64, 128]
      - Different num_heads / num_kv_heads combinations
      - Different head_dim: [64, 128]
    
    Mask rule: mask[q, k] = ((q_local + q_offset) // B) >= (k // B)
    """
    device = torch.device("cuda:0")
    backends = ["fa2", "fa3"]


    if test_dtypes is None:
        test_dtypes = [torch.float16, torch.bfloat16]
    
    # Precision tolerance for different data types
    dtype_tolerances = {
        torch.float16: 1e-2,
        torch.bfloat16: 2e-2,  # FA3 bf16 tile accumulation order differs from FA2;
                               # max_diff up to ~2 ULP (0.015625) is expected
    }
    
    dtype_names = {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }
    
    # Test parameter combinations
    test_configs = [
        # Basic tests: different dllm_block_size
        {"dllm_block_size": 16, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"dllm_block_size": 32, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"dllm_block_size": 64, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"dllm_block_size": 128, "qo_len": 128, "kv_len": 256, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        
        # Different qo_len
        {"dllm_block_size": 32, "qo_len": 32, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"dllm_block_size": 32, "qo_len": 128, "kv_len": 256, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"dllm_block_size": 32, "qo_len": 256, "kv_len": 512, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        
        # Different kv_len
        {"dllm_block_size": 32, "qo_len": 64, "kv_len": 64, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"dllm_block_size": 32, "qo_len": 64, "kv_len": 256, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"dllm_block_size": 32, "qo_len": 64, "kv_len": 512, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"dllm_block_size": 32, "qo_len": 64, "kv_len": 1024, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        
        # Different q_offset (simulating different steps in incremental prefill)
        {"dllm_block_size": 32, "qo_len": 64, "kv_len": 128, "q_offset": 32, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"dllm_block_size": 32, "qo_len": 64, "kv_len": 192, "q_offset": 64, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"dllm_block_size": 32, "qo_len": 64, "kv_len": 256, "q_offset": 128, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"dllm_block_size": 32, "qo_len": 64, "kv_len": 320, "q_offset": 192, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        
        # Different head configurations (MHA vs GQA vs MQA)
        {"dllm_block_size": 32, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 32, "head_dim": 128},  # MHA
        {"dllm_block_size": 32, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 4, "head_dim": 128},   # GQA-8
        {"dllm_block_size": 32, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 1, "head_dim": 128},   # MQA
        
        # Different head_dim
        {"dllm_block_size": 32, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 64},
        {"dllm_block_size": 32, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 16, "num_kv_heads": 4, "head_dim": 256},
        
        # Boundary condition tests
        {"dllm_block_size": 32, "qo_len": 1, "kv_len": 32, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},     # single query
        {"dllm_block_size": 32, "qo_len": 32, "kv_len": 32, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},    # qo_len == kv_len == block_size
        {"dllm_block_size": 64, "qo_len": 33, "kv_len": 97, "q_offset": 17, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},  # non-aligned boundary
        
        # Long sequence tests
        {"dllm_block_size": 32, "qo_len": 128, "kv_len": 2048, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"dllm_block_size": 64, "qo_len": 256, "kv_len": 4096, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
    ]
    
    # Multi-request test configurations
    multi_req_configs = [
        # Basic multi-request tests
        {"num_requests": 2, "dllm_block_size": 32, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"num_requests": 4, "dllm_block_size": 32, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"num_requests": 8, "dllm_block_size": 32, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        
        # Different dllm_block_size
        {"num_requests": 4, "dllm_block_size": 16, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"num_requests": 4, "dllm_block_size": 64, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        
        # Different q_offset (simulating incremental prefill step)
        {"num_requests": 4, "dllm_block_size": 32, "qo_len": 64, "kv_len": 128, "q_offset": 32, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"num_requests": 4, "dllm_block_size": 32, "qo_len": 64, "kv_len": 192, "q_offset": 64, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"num_requests": 4, "dllm_block_size": 32, "qo_len": 64, "kv_len": 256, "q_offset": 128, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        
        # Different head configurations
        {"num_requests": 4, "dllm_block_size": 32, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 32, "head_dim": 128},  # MHA
        {"num_requests": 4, "dllm_block_size": 32, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 4, "head_dim": 128},   # GQA-8
        {"num_requests": 4, "dllm_block_size": 32, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 1, "head_dim": 128},   # MQA
        
        # Long sequence multi-request tests
        {"num_requests": 4, "dllm_block_size": 32, "qo_len": 128, "kv_len": 1024, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"num_requests": 2, "dllm_block_size": 64, "qo_len": 256, "kv_len": 2048, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
    ]
    
    for dtype in test_dtypes:
        dtype_name = dtype_names[dtype]
        tol = dtype_tolerances[dtype]
        
        print(f"\n{'='*120}")
        print(f"DLLM Component Precision Test: Comparing with Native Custom Mask FA2 [{dtype_name.upper()}]")
        print(f"{'='*120}")
        print(f"Reference implementation: single_prefill_with_kv_cache / BatchPrefillWithRaggedKVCacheWrapper + custom_mask (FA2)")
        print(f"Backends under test: FA2, FA3 (DLLM BlockWise supports both backends)")
        print(f"Data type: {dtype_name}")
        print(f"Mask rule: mask[q, k] = ((q_local + q_offset) // B) >= (k // B)")
        print(f"Precision tolerance: {tol}")

        print(f"\n{'='*100}")
        print(f"[Part 1] Single-request Precision Test (FA2 & FA3 backends) [{dtype_name}]")
        print(f"{'='*100}")
        print(f"Reference implementation: single_prefill_with_kv_cache + custom_mask (FA2)")
        print(f"Objects under test:")
        print(f"  1. BatchBlockExtendRaggedOffsetWrapper (batch_size=1) - FA2 backend")
        print(f"  2. BatchBlockExtendRaggedOffsetWrapper (batch_size=1) - FA3 backend")
        print(f"  3. block_extend_attention_with_offset")
        
        single_req_results = []
        
        for cfg_idx, cfg in enumerate(test_configs):
            dllm_block_size = cfg["dllm_block_size"]
            qo_len = cfg["qo_len"]
            kv_len = cfg["kv_len"]
            q_offset = cfg["q_offset"]
            num_heads = cfg["num_heads"]
            num_kv_heads = cfg["num_kv_heads"]
            head_dim = cfg["head_dim"]
            sm_scale = 1.0 / math.sqrt(head_dim)
            
            # Generate test data
            q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)
            k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
            v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)

            # Reference implementation: single_prefill_with_kv_cache + custom_mask (FA2)
            # Build custom_mask: mask[q, k] = ((q_local + q_offset) // B) >= (k // B)
            q_pos = torch.arange(qo_len, device=device) + q_offset
            k_pos = torch.arange(kv_len, device=device)
            q_block = q_pos.unsqueeze(1) // dllm_block_size  # [qo_len, 1]
            k_block = k_pos.unsqueeze(0) // dllm_block_size  # [1, kv_len]
            mask_2d = (q_block >= k_block).to(torch.uint8)   # [qo_len, kv_len]
            
            ref_output = single_prefill_with_kv_cache(
                q, k, v,
                custom_mask=mask_2d,
                sm_scale=sm_scale,
                backend="fa2",
            )
            
            result = {
                "config_idx": cfg_idx,
                "dllm_block_size": dllm_block_size,
                "qo_len": qo_len,
                "kv_len": kv_len,
                "q_offset": q_offset,
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
            }

            # Object under test 1 & 2: BatchBlockExtendRaggedOffsetWrapper (FA2 and FA3 backends)
            qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
            kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=device)
            q_offset_tensor = torch.tensor([q_offset], dtype=torch.int32, device=device)
            
            for backend in backends:
                workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
                wrapper = BatchBlockExtendRaggedOffsetWrapper(
                    workspace, kv_layout="NHD", dllm_block_size=dllm_block_size, backend=backend
                )
                wrapper.plan(
                    qo_indptr=qo_indptr,
                    kv_indptr=kv_indptr,
                    num_qo_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    q_data_type=dtype,
                    sm_scale=sm_scale,
                    q_offsets=q_offset_tensor,
                )
                bbe_output = wrapper.run(q, k, v)
                
                # Calculate precision differences
                bbe_diff = (bbe_output - ref_output).abs().max().item()
                bbe_mean_diff = (bbe_output - ref_output).abs().mean().item()
                bbe_pass = bbe_diff < tol
                
                result[f"bbe_{backend}_max_diff"] = bbe_diff
                result[f"bbe_{backend}_mean_diff"] = bbe_mean_diff
                result[f"bbe_{backend}_pass"] = bbe_pass
                
                del workspace, wrapper

            # Object under test 3: block_extend_attention_with_offset (backend="fa2")
            v2_output = block_extend_attention_with_offset(
                q, k, v,
                dllm_block_size=dllm_block_size,
                q_offset=q_offset,
                sm_scale=sm_scale,
                backend="fa2",
            )
            
            # Calculate V2 precision differences
            v2_diff = (v2_output - ref_output).abs().max().item()
            v2_mean_diff = (v2_output - ref_output).abs().mean().item()
            v2_pass = v2_diff < tol
            
            result["v2_max_diff"] = v2_diff
            result["v2_mean_diff"] = v2_mean_diff
            result["v2_pass"] = v2_pass
            
            single_req_results.append(result)
            
            if verbose:
                fa2_status = "PASS" if result["bbe_fa2_pass"] else "FAIL"
                fa3_status = "PASS" if result["bbe_fa3_pass"] else "FAIL"
                v2_status = "PASS" if v2_pass else "FAIL"
                print(f"\n  [Test {cfg_idx:02d}] B={dllm_block_size:3d}, qo={qo_len:4d}, kv={kv_len:4d}, "
                      f"q_off={q_offset:3d}, heads={num_heads}/{num_kv_heads}, dim={head_dim}")
                print(f"           BBE-FA2: max_diff={result['bbe_fa2_max_diff']:.6f}, mean_diff={result['bbe_fa2_mean_diff']:.6f} [{fa2_status}]")
                print(f"           BBE-FA3: max_diff={result['bbe_fa3_max_diff']:.6f}, mean_diff={result['bbe_fa3_mean_diff']:.6f} [{fa3_status}]")
                print(f"           V2:      max_diff={v2_diff:.6f}, mean_diff={v2_mean_diff:.6f} [{v2_status}]")
            
            torch.cuda.empty_cache()
        
        # Single-request test summary
        print(f"\n{'-'*100}")
        print(f"[Single-request Precision Test Summary] [{dtype_name}]")
        print(f"{'-'*100}")
        
        total_tests = len(single_req_results)
        
        for backend in backends:
            pass_count = sum(1 for r in single_req_results if r[f"bbe_{backend}_pass"])
            max_diff_all = max(r[f"bbe_{backend}_max_diff"] for r in single_req_results)
            mean_diff_all = sum(r[f"bbe_{backend}_mean_diff"] for r in single_req_results) / total_tests
            print(f"  BatchBlockExtendRaggedOffsetWrapper ({backend.upper()}): {pass_count}/{total_tests} PASS")
            print(f"    max_diff (all tests): {max_diff_all:.6f}")
            print(f"    mean_diff (avg):      {mean_diff_all:.6f}")
        
        v2_pass_count = sum(1 for r in single_req_results if r["v2_pass"])
        v2_max_diff_all = max(r["v2_max_diff"] for r in single_req_results)
        v2_mean_diff_all = sum(r["v2_mean_diff"] for r in single_req_results) / total_tests
        print(f"  block_extend_attention_with_offset: {v2_pass_count}/{total_tests} PASS")
        print(f"    max_diff (all tests): {v2_max_diff_all:.6f}")
        print(f"    mean_diff (avg):      {v2_mean_diff_all:.6f}")
        
        # Failed test details
        for backend in backends:
            failed = [r for r in single_req_results if not r[f"bbe_{backend}_pass"]]
            if failed:
                print(f"\n  [BBE-{backend.upper()} Failed Test Details]")
                for r in failed:
                    print(f"    Test {r['config_idx']:02d}: B={r['dllm_block_size']}, qo={r['qo_len']}, kv={r['kv_len']}, "
                          f"q_off={r['q_offset']}, max_diff={r[f'bbe_{backend}_max_diff']:.6f}")
        
        failed_v2 = [r for r in single_req_results if not r["v2_pass"]]
        if failed_v2:
            print(f"\n  [V2 Failed Test Details]")
            for r in failed_v2:
                print(f"    Test {r['config_idx']:02d}: B={r['dllm_block_size']}, qo={r['qo_len']}, kv={r['kv_len']}, "
                      f"q_off={r['q_offset']}, max_diff={r['v2_max_diff']:.6f}")
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # Part 2: Multi-request Precision Test (FA2 & FA3 backends)
        # ═══════════════════════════════════════════════════════════════════════════════
        print(f"\n{'='*100}")
        print(f"[Part 2] Multi-request Precision Test (FA2 & FA3 backends) [{dtype_name}]")
        print(f"{'='*100}")
        print(f"Reference implementation: BatchPrefillWithRaggedKVCacheWrapper + custom_mask (FA2)")
        print(f"Objects under test:")
        print(f"  1. BatchBlockExtendRaggedOffsetWrapper - FA2 backend")
        print(f"  2. BatchBlockExtendRaggedOffsetWrapper - FA3 backend")
        
        multi_req_results = []
        
        for cfg_idx, cfg in enumerate(multi_req_configs):
            num_requests = cfg["num_requests"]
            dllm_block_size = cfg["dllm_block_size"]
            qo_len = cfg["qo_len"]
            kv_len = cfg["kv_len"]
            q_offset = cfg["q_offset"]
            num_heads = cfg["num_heads"]
            num_kv_heads = cfg["num_kv_heads"]
            head_dim = cfg["head_dim"]
            sm_scale = 1.0 / math.sqrt(head_dim)
            
            q_list = [torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device) for _ in range(num_requests)]
            k_list = [torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device) for _ in range(num_requests)]
            v_list = [torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device) for _ in range(num_requests)]
            
            q_batch = torch.cat(q_list, dim=0)
            k_batch = torch.cat(k_list, dim=0)
            v_batch = torch.cat(v_list, dim=0)
            
            # Build mask
            q_pos = torch.arange(qo_len, device=device) + q_offset
            k_pos = torch.arange(kv_len, device=device)
            q_block = q_pos.unsqueeze(1) // dllm_block_size
            k_block = k_pos.unsqueeze(0) // dllm_block_size
            mask_2d = (q_block >= k_block)
            mask_flat = mask_2d.flatten()
            batch_mask = mask_flat.repeat(num_requests)
            
            qo_indptr = torch.tensor([i * qo_len for i in range(num_requests + 1)], dtype=torch.int32, device=device)
            kv_indptr = torch.tensor([i * kv_len for i in range(num_requests + 1)], dtype=torch.int32, device=device)
            
            # Reference implementation
            ref_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device),
                kv_layout="NHD", backend="fa2",
            )
            ref_wrapper.plan(
                qo_indptr=qo_indptr, kv_indptr=kv_indptr,
                num_qo_heads=num_heads, num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim, q_data_type=dtype,
                custom_mask=batch_mask, causal=False, sm_scale=sm_scale,
            )
            ref_output = ref_wrapper.run(q_batch, k_batch, v_batch)
            
            q_offsets = torch.full((num_requests,), q_offset, dtype=torch.int32, device=device)
            result = {
                "config_idx": cfg_idx, "num_requests": num_requests,
                "dllm_block_size": dllm_block_size, "qo_len": qo_len, "kv_len": kv_len,
                "q_offset": q_offset, "num_heads": num_heads,
                "num_kv_heads": num_kv_heads, "head_dim": head_dim,
            }
            
            for backend in backends:
                bbe_wrapper = BatchBlockExtendRaggedOffsetWrapper(
                    torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device),
                    kv_layout="NHD", dllm_block_size=dllm_block_size, backend=backend
                )
                bbe_wrapper.plan(
                    qo_indptr=qo_indptr, kv_indptr=kv_indptr,
                    num_qo_heads=num_heads, num_kv_heads=num_kv_heads,
                    head_dim=head_dim, q_data_type=dtype,
                    sm_scale=sm_scale, q_offsets=q_offsets,
                )
                bbe_output = bbe_wrapper.run(q_batch, k_batch, v_batch)
                
                bbe_diff = (bbe_output - ref_output).abs().max().item()
                bbe_mean_diff = (bbe_output - ref_output).abs().mean().item()
                bbe_pass = bbe_diff < tol
                
                result[f"bbe_{backend}_max_diff"] = bbe_diff
                result[f"bbe_{backend}_mean_diff"] = bbe_mean_diff
                result[f"bbe_{backend}_pass"] = bbe_pass
                del bbe_wrapper
            
            multi_req_results.append(result)
            
            if verbose:
                fa2_status = "PASS" if result["bbe_fa2_pass"] else "FAIL"
                fa3_status = "PASS" if result["bbe_fa3_pass"] else "FAIL"
                print(f"\n  [Test {cfg_idx:02d}] reqs={num_requests}, B={dllm_block_size:3d}, qo={qo_len:4d}, kv={kv_len:4d}, "
                      f"q_off={q_offset:3d}, heads={num_heads}/{num_kv_heads}, dim={head_dim}")
                print(f"           BBE-FA2: max_diff={result['bbe_fa2_max_diff']:.6f} [{fa2_status}]")
                print(f"           BBE-FA3: max_diff={result['bbe_fa3_max_diff']:.6f} [{fa3_status}]")
            
            del ref_wrapper
            torch.cuda.empty_cache()
        
        # Multi-request test summary
        print(f"\n[Multi-request Precision Test Summary] [{dtype_name}]")
        total_tests = len(multi_req_results)
        for backend in backends:
            pass_count = sum(1 for r in multi_req_results if r[f"bbe_{backend}_pass"])
            max_diff_all = max(r[f"bbe_{backend}_max_diff"] for r in multi_req_results)
            print(f"  BBE ({backend.upper()}): {pass_count}/{total_tests} PASS, max_diff={max_diff_all:.6f}")
        
        for backend in backends:
            failed = [r for r in multi_req_results if not r[f"bbe_{backend}_pass"]]
            if failed:
                print(f"\n  [BBE-{backend.upper()} Failed Details]")
                for r in failed:
                    print(f"    Test {r['config_idx']:02d}: reqs={r['num_requests']}, B={r['dllm_block_size']}, max_diff={r[f'bbe_{backend}_max_diff']:.6f}")
        
        # Part 3: Paged KV Cache Test
        print(f"\n[Part 3] Paged KV Cache Precision Test [{dtype_name}]")
        
        page_size = 16
        paged_configs = [
            {"dllm_block_size": 32, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
            {"dllm_block_size": 32, "qo_len": 64, "kv_len": 128, "q_offset": 32, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
            {"dllm_block_size": 64, "qo_len": 128, "kv_len": 512, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
            {"dllm_block_size": 32, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 4, "head_dim": 128},
            {"dllm_block_size": 32, "qo_len": 128, "kv_len": 1024, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        ]
        
        paged_results = []
        
        for cfg_idx, cfg in enumerate(paged_configs):
            dllm_block_size = cfg["dllm_block_size"]
            qo_len = cfg["qo_len"]
            kv_len = cfg["kv_len"]
            q_offset = cfg["q_offset"]
            num_heads = cfg["num_heads"]
            num_kv_heads = cfg["num_kv_heads"]
            head_dim = cfg["head_dim"]
            sm_scale = 1.0 / math.sqrt(head_dim)
            
            q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)
            num_pages = (kv_len + page_size - 1) // page_size
            kv_data = torch.randn(num_pages, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device=device)
            
            paged_kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
            paged_kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
            last_page_len = kv_len - (num_pages - 1) * page_size if kv_len % page_size != 0 else page_size
            paged_kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=device)
            
            k_continuous = kv_data[:, 0, :, :, :].reshape(-1, num_kv_heads, head_dim)[:kv_len]
            v_continuous = kv_data[:, 1, :, :, :].reshape(-1, num_kv_heads, head_dim)[:kv_len]
            
            # Reference implementation
            q_pos = torch.arange(qo_len, device=device) + q_offset
            k_pos = torch.arange(kv_len, device=device)
            q_block = q_pos.unsqueeze(1) // dllm_block_size
            k_block = k_pos.unsqueeze(0) // dllm_block_size
            mask_2d = (q_block >= k_block).to(torch.uint8)
            
            ref_output = single_prefill_with_kv_cache(
                q, k_continuous, v_continuous,
                custom_mask=mask_2d, sm_scale=sm_scale, backend="fa2",
            )
            
            result = {
                "config_idx": cfg_idx, "dllm_block_size": dllm_block_size,
                "qo_len": qo_len, "kv_len": kv_len, "q_offset": q_offset,
                "num_heads": num_heads, "num_kv_heads": num_kv_heads,
                "head_dim": head_dim, "page_size": page_size,
            }
            
            qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
            q_offsets = torch.tensor([q_offset], dtype=torch.int32, device=device)
            
            for backend in backends:
                paged_wrapper = BatchBlockExtendPagedOffsetWrapper(
                    torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
                    kv_layout="NHD", dllm_block_size=dllm_block_size, backend=backend
                )
                paged_wrapper.plan(
                    qo_indptr=qo_indptr, paged_kv_indptr=paged_kv_indptr,
                    paged_kv_indices=paged_kv_indices, paged_kv_last_page_len=paged_kv_last_page_len,
                    num_qo_heads=num_heads, num_kv_heads=num_kv_heads,
                    head_dim=head_dim, page_size=page_size,
                    q_data_type=dtype, sm_scale=sm_scale, q_offsets=q_offsets,
                )
                paged_output = paged_wrapper.run(q, kv_data)
                
                paged_diff = (paged_output - ref_output).abs().max().item()
                paged_pass = paged_diff < tol
                result[f"paged_{backend}_max_diff"] = paged_diff
                result[f"paged_{backend}_pass"] = paged_pass
                del paged_wrapper
            
            paged_results.append(result)
            
            if verbose:
                fa2_status = "PASS" if result["paged_fa2_pass"] else "FAIL"
                fa3_status = "PASS" if result["paged_fa3_pass"] else "FAIL"
                print(f"  [Test {cfg_idx:02d}] B={dllm_block_size:3d}, qo={qo_len:4d}, kv={kv_len:4d}, "
                      f"q_off={q_offset:3d} - FA2:{fa2_status}, FA3:{fa3_status}")
            
            torch.cuda.empty_cache()
        
        # Paged test summary
        print(f"\n[Paged KV Cache Precision Test Summary] [{dtype_name}]")
        for backend in backends:
            pass_count = sum(1 for r in paged_results if r[f"paged_{backend}_pass"])
            max_diff_all = max(r[f"paged_{backend}_max_diff"] for r in paged_results)
            print(f"  Paged ({backend.upper()}): {pass_count}/{len(paged_results)} PASS, max_diff={max_diff_all:.6f}")
        
        for backend in backends:
            failed = [r for r in paged_results if not r[f"paged_{backend}_pass"]]
            if failed:
                print(f"\n  [Paged-{backend.upper()} Failed Details]")
                for r in failed:
                    print(f"    Test {r['config_idx']:02d}: B={r['dllm_block_size']}, max_diff={r[f'paged_{backend}_max_diff']:.6f}")
    
        # Summary
        print(f"\n{'='*100}")
        print(f"Precision Test Summary [{dtype_name}]")
        print(f"{'='*100}")
        
        print(f"\n  Single-request tests:")
        for backend in backends:
            pass_count = sum(1 for r in single_req_results if r[f"bbe_{backend}_pass"])
            print(f"    BBE ({backend.upper()}): {pass_count}/{len(single_req_results)} PASS")
        v2_pass_count = sum(1 for r in single_req_results if r["v2_pass"])
        print(f"    V2: {v2_pass_count}/{len(single_req_results)} PASS")
        
        print(f"\n  Multi-request tests:")
        for backend in backends:
            pass_count = sum(1 for r in multi_req_results if r[f"bbe_{backend}_pass"])
            print(f"    BBE ({backend.upper()}): {pass_count}/{len(multi_req_results)} PASS")
        
        print(f"\n  Paged tests:")
        for backend in backends:
            pass_count = sum(1 for r in paged_results if r[f"paged_{backend}_pass"])
            print(f"    Paged ({backend.upper()}): {pass_count}/{len(paged_results)} PASS")
        
        # Overall results
        all_single_pass = all(
            r["bbe_fa2_pass"] and r["bbe_fa3_pass"] and r["v2_pass"] 
            for r in single_req_results
        )
        all_multi_pass = all(
            r["bbe_fa2_pass"] and r["bbe_fa3_pass"] 
            for r in multi_req_results
        )
        all_paged_pass = all(
            r["paged_fa2_pass"] and r["paged_fa3_pass"] 
            for r in paged_results
        )
        
        overall_pass = all_single_pass and all_multi_pass and all_paged_pass
        overall_status = "ALL TESTS PASSED" if overall_pass else "SOME TESTS FAILED"
        print(f"\n  Overall results: {overall_status}")
        
        # FA2 vs FA3 comparison
        fa2_single_max = max(r["bbe_fa2_max_diff"] for r in single_req_results)
        fa3_single_max = max(r["bbe_fa3_max_diff"] for r in single_req_results)
        fa2_multi_max = max(r["bbe_fa2_max_diff"] for r in multi_req_results)
        fa3_multi_max = max(r["bbe_fa3_max_diff"] for r in multi_req_results)
        fa2_paged_max = max(r["paged_fa2_max_diff"] for r in paged_results)
        fa3_paged_max = max(r["paged_fa3_max_diff"] for r in paged_results)
        
        print(f"\n  FA2 vs FA3 max_diff:")
        print(f"    Single-request: FA2={fa2_single_max:.6f}, FA3={fa3_single_max:.6f}")
        print(f"    Multi-request: FA2={fa2_multi_max:.6f}, FA3={fa3_multi_max:.6f}")
        print(f"    Paged:  FA2={fa2_paged_max:.6f}, FA3={fa3_paged_max:.6f}")
    
    return {
        "overall_pass": overall_pass,
    }


def test_cascade_interfaces_perf(
    num_requests: int = 4,
    tokens_per_request: int = 512,
    dllm_block_size: int = 32,
    chunk_sizes: list = None,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 16,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    verbose: bool = False,
    backend: str = "fa2",
):
    """
    Compare performance of two Cascade interfaces (Step by Step incremental Prefill scenario)
    
    Interfaces under test:
      1. batch_block_extend_cascade: Uses Block Extend mask
         - Supports chunk_size != dllm_block_size
      2. sglang_style_cascade_attention: Uses Causal mask (SGLang native style)
         - Requires chunk_size == dllm_block_size
    
    Test scenario:
      - Real incremental Prefill: each step depends on previous step's KV Cache
      - Each step: Q attends to (current_chunk KV + prefix KV)
      - Uses Paged KV Cache to store prefix
    
    Key points:
      - When chunk_size == dllm_block_size, both should produce identical results
      - batch_block_extend_cascade can use larger chunk_size
    """
    from flashinfer.dllm import (
        batch_block_extend_cascade,
        sglang_style_cascade_attention,
    )
    
    if chunk_sizes is None:
        chunk_sizes = [dllm_block_size]  # Default: only test chunk_size == dllm_block_size
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    sm_scale = 1.0 / (head_dim ** 0.5)
    
    print(f"\n{'='*90}")
    print(f"Cascade Interface Performance Comparison: Step by Step Incremental Prefill")
    print(f"{'='*90}")
    print(f"Configuration:")
    print(f"  num_requests        = {num_requests}")
    print(f"  tokens_per_request  = {tokens_per_request}")
    print(f"  dllm_block_size     = {dllm_block_size}")
    print(f"  chunk_sizes         = {chunk_sizes}")
    print(f"  num_heads           = {num_heads}")
    print(f"  num_kv_heads        = {num_kv_heads}")
    print(f"  head_dim            = {head_dim}")
    print(f"  page_size           = {page_size}")
    print(f"  backend             = {backend}")
    print(f"\nScenario description:")
    print(f"  - Step by Step incremental Prefill: each step depends on previous step's KV Cache")
    print(f"  - Current Chunk: Ragged KV (contiguous memory)")
    print(f"  - Prefix: Paged KV Cache")
    print(f"  - Uses CUDA Graph to reduce overhead")
    
    # Generate complete Q, K, V for each request
    all_qs = [torch.randn(tokens_per_request, num_heads, head_dim, dtype=dtype, device=device) 
              for _ in range(num_requests)]
    all_ks = [torch.randn(tokens_per_request, num_kv_heads, head_dim, dtype=dtype, device=device) 
              for _ in range(num_requests)]
    all_vs = [torch.randn(tokens_per_request, num_kv_heads, head_dim, dtype=dtype, device=device) 
              for _ in range(num_requests)]
    
    def split_chunks(tensor, chunk_size):
        return [tensor[i*chunk_size:(i+1)*chunk_size] for i in range(tensor.shape[0] // chunk_size)]
    
    results = {}
    
    for chunk_size in chunk_sizes:
        if tokens_per_request % chunk_size != 0:
            print(f"\n[Skip] chunk_size={chunk_size} doesn't evenly divide tokens_per_request={tokens_per_request}")
            continue
        
        num_steps = tokens_per_request // chunk_size
        
        print(f"\n{'-'*90}")
        print(f"chunk_size = {chunk_size}, num_steps = {num_steps}")
        print(f"{'-'*90}")
        
        # Split chunks for each request
        qs_chunks = [split_chunks(q, chunk_size) for q in all_qs]
        ks_chunks = [split_chunks(k, chunk_size) for k in all_ks]
        vs_chunks = [split_chunks(v, chunk_size) for v in all_vs]
        
        # Pre-allocate buffers for all steps
        # Current chunk Q, K, V (concatenate all requests)
        q_current_buffers = []
        k_current_buffers = []
        v_current_buffers = []
        for step_idx in range(num_steps):
            q_list = [qs_chunks[req_idx][step_idx] for req_idx in range(num_requests)]
            k_list = [ks_chunks[req_idx][step_idx] for req_idx in range(num_requests)]
            v_list = [vs_chunks[req_idx][step_idx] for req_idx in range(num_requests)]
            q_current_buffers.append(torch.cat(q_list, dim=0))
            k_current_buffers.append(torch.cat(k_list, dim=0))
            v_current_buffers.append(torch.cat(v_list, dim=0))
        
        # Paged KV Cache setup (prefix storage)
        # Calculate maximum number of pages needed
        max_prefix_len = (num_steps - 1) * chunk_size
        max_pages_per_request = (max_prefix_len + page_size - 1) // page_size if max_prefix_len > 0 else 0
        total_max_pages = num_requests * max_pages_per_request
        
        # Allocate Paged KV Cache
        if total_max_pages > 0:
            paged_kv_cache = torch.randn(
                total_max_pages, 2, page_size, num_kv_heads, head_dim,
                dtype=dtype, device=device
            )
        else:
            paged_kv_cache = None
        
        # Prepare paged kv parameters for each step
        paged_kv_params_list = []
        for step_idx in range(num_steps):
            prefix_len = step_idx * chunk_size
            if prefix_len == 0:
                # First step has no prefix
                paged_kv_params_list.append(None)
            else:
                pages_per_request = (prefix_len + page_size - 1) // page_size
                total_pages = num_requests * pages_per_request
                
                paged_kv_indptr = torch.tensor(
                    [i * pages_per_request for i in range(num_requests + 1)],
                    dtype=torch.int32, device=device
                )
                paged_kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
                last_page_len = prefix_len % page_size if prefix_len % page_size != 0 else page_size
                paged_kv_last_page_len = torch.full(
                    (num_requests,), last_page_len, dtype=torch.int32, device=device
                )
                
                paged_kv_params_list.append({
                    "paged_kv_cache": paged_kv_cache[:total_pages] if paged_kv_cache is not None else None,
                    "paged_kv_indptr": paged_kv_indptr,
                    "paged_kv_indices": paged_kv_indices,
                    "paged_kv_last_page_len": paged_kv_last_page_len,
                })
        
        # indptrs
        qo_indptr = torch.tensor(
            [i * chunk_size for i in range(num_requests + 1)],
            dtype=torch.int32, device=device
        )
        kv_curr_indptr = torch.tensor(
            [i * chunk_size for i in range(num_requests + 1)],
            dtype=torch.int32, device=device
        )
        
        # q_offsets and kv_offsets (required by block_extend_cascade)
        q_offsets_list = []
        for step_idx in range(num_steps):
            prefix_len = step_idx * chunk_size
            q_offsets_list.append(torch.full((num_requests,), prefix_len, dtype=torch.int32, device=device))
        
        # Workspace buffer (shared)
        workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        
        # Output buffer
        output_buffer = torch.empty(num_requests * chunk_size, num_heads, head_dim, dtype=dtype, device=device)

        # Precision verification (when chunk_size == dllm_block_size)
        if chunk_size == dllm_block_size:
            print(f"  [Precision Verification] chunk_size == dllm_block_size, comparing outputs of both interfaces...")
            
            max_diff_all_steps = 0.0
            for step_idx in range(num_steps):
                q_batch = q_current_buffers[step_idx]
                k_current = k_current_buffers[step_idx]
                v_current = v_current_buffers[step_idx]
                paged_params = paged_kv_params_list[step_idx]
                
                # batch_block_extend_cascade (function internally determines has_prefix)
                be_out = batch_block_extend_cascade(
                    q=q_batch,
                    k_current=k_current,
                    v_current=v_current,
                    qo_indptr=qo_indptr,
                    kv_curr_indptr=kv_curr_indptr,
                    paged_kv_cache=paged_params["paged_kv_cache"] if paged_params else None,
                    paged_kv_indptr=paged_params["paged_kv_indptr"] if paged_params else None,
                    paged_kv_indices=paged_params["paged_kv_indices"] if paged_params else None,
                    paged_kv_last_page_len=paged_params["paged_kv_last_page_len"] if paged_params else None,
                    page_size=page_size,
                    dllm_block_size=dllm_block_size,
                    q_offsets=q_offsets_list[step_idx],
                    kv_offsets=q_offsets_list[step_idx],  # Cascade scenario: kv_offset == q_offset == prefix_len
                    workspace_buffer=workspace_buffer,
                    sm_scale=sm_scale,
                    backend=backend,
                )
                
                # sglang_style_cascade_attention (function internally determines has_prefix)
                sg_out = sglang_style_cascade_attention(
                    q=q_batch,
                    k_current=k_current,
                    v_current=v_current,
                    qo_indptr=qo_indptr,
                    kv_curr_indptr=kv_curr_indptr,
                    paged_kv_cache=paged_params["paged_kv_cache"] if paged_params else None,
                    paged_kv_indptr=paged_params["paged_kv_indptr"] if paged_params else None,
                    paged_kv_indices=paged_params["paged_kv_indices"] if paged_params else None,
                    paged_kv_last_page_len=paged_params["paged_kv_last_page_len"] if paged_params else None,
                    page_size=page_size,
                    workspace_buffer=workspace_buffer,
                    sm_scale=sm_scale,
                    backend=backend,
                )
                
                step_diff = (be_out - sg_out).abs().max().item()
                max_diff_all_steps = max(max_diff_all_steps, step_diff)
                
                if verbose:
                    print(f"    step {step_idx}: max_diff = {step_diff:.6f}")
            
            precision_ok = max_diff_all_steps < 0.01
            status = " PASS" if precision_ok else " FAIL"
            print(f"  [Precision Verification] max_diff = {max_diff_all_steps:.6f} {status}")
        

        # [1] batch_block_extend_cascade performance test
        print(f"  [batch_block_extend_cascade] Performance test (without CUDA Graph)...")
        
        # ========== Segmented timing: measure Python overhead vs Kernel overhead ==========
        if verbose:
            import time as time_module
            
            # 1) Measure single complete call
            torch.cuda.synchronize()
            t0 = time_module.perf_counter()
            _ = batch_block_extend_cascade(
                q=q_current_buffers[0],
                k_current=k_current_buffers[0],
                v_current=v_current_buffers[0],
                qo_indptr=qo_indptr,
                kv_curr_indptr=kv_curr_indptr,
                dllm_block_size=dllm_block_size,
                q_offsets=q_offsets_list[0],
                kv_offsets=q_offsets_list[0],
                workspace_buffer=workspace_buffer,
                sm_scale=sm_scale,
                backend=backend,
            )
            torch.cuda.synchronize()
            be_single_call = (time_module.perf_counter() - t0) * 1000
            print(f"    [Segmented Timing] BE single call: {be_single_call:.3f} ms (includes Wrapper creation + plan + run)")
            
            # 2) Measure SG single complete call
            torch.cuda.synchronize()
            t0 = time_module.perf_counter()
            _ = sglang_style_cascade_attention(
                q=q_current_buffers[0],
                k_current=k_current_buffers[0],
                v_current=v_current_buffers[0],
                qo_indptr=qo_indptr,
                kv_curr_indptr=kv_curr_indptr,
                workspace_buffer=workspace_buffer,
                sm_scale=sm_scale,
                backend=backend,
            )
            torch.cuda.synchronize()
            sg_single_call = (time_module.perf_counter() - t0) * 1000
            print(f"    [Segmented Timing] SG single call: {sg_single_call:.3f} ms (includes Wrapper creation + plan + run)")
            print(f"    [Segmented Timing] Single call diff: BE={be_single_call:.3f}ms, SG={sg_single_call:.3f}ms, diff={sg_single_call-be_single_call:.3f}ms")
            
            # 3) Measure pure kernel time with reused Wrapper (excluding Wrapper creation and plan overhead)
            print(f"    [Segmented Timing] Measuring pure run() time with reused Wrapper...")
            
            # BE: Create and plan once
            from flashinfer.dllm.batch_block_extend import (
                BatchBlockExtendRaggedOffsetWrapper,
            )
            be_wrapper = BatchBlockExtendRaggedOffsetWrapper(
                workspace_buffer.clone(),
                kv_layout="NHD",
                dllm_block_size=dllm_block_size,
                backend=backend,
            )
            be_wrapper.plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_curr_indptr,
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_data_type=q_current_buffers[0].dtype,
                sm_scale=sm_scale,
                q_offsets=q_offsets_list[0],
                kv_offsets=q_offsets_list[0],
            )
            
            # SG: Create and plan once
            from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper
            sg_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                workspace_buffer.clone(),
                kv_layout="NHD",
                backend=backend,
            )
            sg_wrapper.plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_curr_indptr,
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                head_dim_vo=head_dim,
                q_data_type=q_current_buffers[0].dtype,
                causal=False,  # Same as BE: use non-causal (fully visible)
            )
            
            # BE: Only measure run() time
            for _ in range(10):  # warmup
                _ = be_wrapper.run(q_current_buffers[0], k_current_buffers[0], v_current_buffers[0])
            torch.cuda.synchronize()
            t0 = time_module.perf_counter()
            for _ in range(100):
                _ = be_wrapper.run(q_current_buffers[0], k_current_buffers[0], v_current_buffers[0])
            torch.cuda.synchronize()
            be_run_only = (time_module.perf_counter() - t0) / 100 * 1000
            
            # SG: Only measure run() time
            for _ in range(10):  # warmup
                _ = sg_wrapper.run(q_current_buffers[0], k_current_buffers[0], v_current_buffers[0])
            torch.cuda.synchronize()
            t0 = time_module.perf_counter()
            for _ in range(100):
                _ = sg_wrapper.run(q_current_buffers[0], k_current_buffers[0], v_current_buffers[0])
            torch.cuda.synchronize()
            sg_run_only = (time_module.perf_counter() - t0) / 100 * 1000
            
            print(f"    [Segmented Timing] BE run() only: {be_run_only:.3f} ms")
            print(f"    [Segmented Timing] SG run() only: {sg_run_only:.3f} ms")
            print(f"    [Segmented Timing] run() diff: BE={be_run_only:.3f}ms, SG={sg_run_only:.3f}ms, diff={sg_run_only-be_run_only:.3f}ms")
            if abs(sg_run_only - be_run_only) < 0.01:
                print(f"    [Conclusion] Kernel-level performance is comparable, diff comes from Python overhead")
            else:
                print(f"    [Conclusion] Kernel-level diff exists, possibly due to different mask_mode implementations")
        
        def run_be_cascade_pipeline():
            for step_idx in range(num_steps):
                q_batch = q_current_buffers[step_idx]
                k_current = k_current_buffers[step_idx]
                v_current = v_current_buffers[step_idx]
                paged_params = paged_kv_params_list[step_idx]
                
                # Function internally determines has_prefix
                output_buffer.copy_(batch_block_extend_cascade(
                    q=q_batch,
                    k_current=k_current,
                    v_current=v_current,
                    qo_indptr=qo_indptr,
                    kv_curr_indptr=kv_curr_indptr,
                    paged_kv_cache=paged_params["paged_kv_cache"] if paged_params else None,
                    paged_kv_indptr=paged_params["paged_kv_indptr"] if paged_params else None,
                    paged_kv_indices=paged_params["paged_kv_indices"] if paged_params else None,
                    paged_kv_last_page_len=paged_params["paged_kv_last_page_len"] if paged_params else None,
                    page_size=page_size,
                    dllm_block_size=dllm_block_size,
                    q_offsets=q_offsets_list[step_idx],
                    kv_offsets=q_offsets_list[step_idx],  # Cascade scenario: kv_offset == q_offset == prefix_len
                    workspace_buffer=workspace_buffer,
                    sm_scale=sm_scale,
                    backend=backend,
                ))
        
        # Warmup
        for _ in range(warmup_iters):
            run_be_cascade_pipeline()
        torch.cuda.synchronize()
        
        # Benchmark (without CUDA Graph)
        start = time.perf_counter()
        for _ in range(bench_iters):
            run_be_cascade_pipeline()
        torch.cuda.synchronize()
        be_time = (time.perf_counter() - start) / bench_iters * 1000
        
        print(f"    => {be_time:.3f} ms ({be_time/num_steps:.3f} ms/step × {num_steps} steps)")
        
        torch.cuda.empty_cache()

        # [2] sglang_style_cascade_attention performance test
        print(f"  [sglang_style_cascade_attention] Performance test (without CUDA Graph)...")
        
        def run_sg_cascade_pipeline():
            for step_idx in range(num_steps):
                q_batch = q_current_buffers[step_idx]
                k_current = k_current_buffers[step_idx]
                v_current = v_current_buffers[step_idx]
                paged_params = paged_kv_params_list[step_idx]
                
                # Function internally determines has_prefix
                output_buffer.copy_(sglang_style_cascade_attention(
                    q=q_batch,
                    k_current=k_current,
                    v_current=v_current,
                    qo_indptr=qo_indptr,
                    kv_curr_indptr=kv_curr_indptr,
                    paged_kv_cache=paged_params["paged_kv_cache"] if paged_params else None,
                    paged_kv_indptr=paged_params["paged_kv_indptr"] if paged_params else None,
                    paged_kv_indices=paged_params["paged_kv_indices"] if paged_params else None,
                    paged_kv_last_page_len=paged_params["paged_kv_last_page_len"] if paged_params else None,
                    page_size=page_size,
                    workspace_buffer=workspace_buffer,
                    sm_scale=sm_scale,
                    backend=backend,
                ))
        
        # Warmup
        for _ in range(warmup_iters):
            run_sg_cascade_pipeline()
        torch.cuda.synchronize()
        
        # Benchmark (without CUDA Graph)
        start = time.perf_counter()
        for _ in range(bench_iters):
            run_sg_cascade_pipeline()
        torch.cuda.synchronize()
        sg_time = (time.perf_counter() - start) / bench_iters * 1000
        
        print(f"    => {sg_time:.3f} ms ({sg_time/num_steps:.3f} ms/step × {num_steps} steps)")
        

        speedup = sg_time / be_time if be_time > 0 else 0
        if speedup > 1:
            print(f"    => batch_block_extend_cascade is faster by {speedup:.2f}x")
        else:
            print(f"    => sglang_style_cascade_attention is faster by {1/speedup:.2f}x")
        
        results[f"chunk{chunk_size}"] = {
            "chunk_size": chunk_size,
            "num_steps": num_steps,
            "be_cascade_ms": be_time,
            "sg_cascade_ms": sg_time,
            "speedup_be_over_sg": speedup,
        }
        
        torch.cuda.empty_cache()
    

    # Results summary
    print(f"\n{'='*90}")
    print(f"Results Summary (num_requests={num_requests}, tokens_per_request={tokens_per_request}, dllm_block_size={dllm_block_size})")
    print(f"{'='*90}")
    
    print(f"\n{'chunk':>8} | {'steps':>6} | {'BE Cascade':>14} | {'SG Cascade':>14} | {'BE/SG':>10}")
    print(f"{'-'*8}-+-{'-'*6}-+-{'-'*14}-+-{'-'*14}-+-{'-'*10}")
    
    for key in sorted(results.keys(), key=lambda k: results[k]["chunk_size"]):
        r = results[key]
        print(f"{r['chunk_size']:>8} | {r['num_steps']:>6} | {r['be_cascade_ms']:>12.3f}ms | {r['sg_cascade_ms']:>12.3f}ms | {r['speedup_be_over_sg']:>9.2f}x")
    
    print(f"\nNotes:")
    print(f"  - BE Cascade: batch_block_extend_cascade (Block Extend mask)")
    print(f"  - SG Cascade: sglang_style_cascade_attention (Causal mask)")
    print(f"  - BE/SG: Speed ratio of batch_block_extend_cascade vs sglang_style")
    print(f"    (>1 means BE is faster, <1 means SG is faster)")
    print(f"  - When chunk_size == dllm_block_size, causal mask = block_extend mask")
    print(f"  - batch_block_extend_cascade supports chunk_size != dllm_block_size")
    
    return results


def test_cascade_interfaces_perf_with_cuda_graph(
    num_requests: int = 4,
    tokens_per_request: int = 512,
    dllm_block_size: int = 32,
    chunk_sizes: list = None,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 16,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    verbose: bool = False,
    backend: str = "fa2",
):
    """
    Compare performance of two Cascade interfaces (with CUDA Graph optimization)
    
    Differences from test_cascade_interfaces_perf:
    ═══════════════════════════════════════════════════════════════════════════════
    - This function: Uses CUDA Graph to capture run() operations, reducing Python/launch overhead
    - Original function: Each call includes Wrapper creation + plan() + run()
    
    CUDA Graph implementation notes:
    ═══════════════════════════════════════════════════════════════════════════════
    1. plan() contains CPU-GPU synchronization, cannot be executed during Graph capture
    2. Pre-create independent Wrappers for each step and complete plan()
    3. CUDA Graph only captures run() operations
    4. Each step has different paged_kv configuration, requires independent Wrapper
    
    Interfaces under test:
      1. batch_block_extend_cascade (via Wrapper.run())
      2. sglang_style_cascade_attention (via Wrapper.run())
    """
    from flashinfer.dllm import (
        BatchBlockExtendRaggedOffsetWrapper,
        BatchBlockExtendPagedOffsetWrapper
    )
    from flashinfer.prefill import (
        BatchPrefillWithRaggedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
    )
    from flashinfer.cascade import merge_state
    
    if chunk_sizes is None:
        chunk_sizes = [dllm_block_size]
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    sm_scale = 1.0 / (head_dim ** 0.5)
    
    print(f"\n{'='*90}")
    print(f"Cascade Interface Performance Comparison: Step by Step Incremental Prefill (CUDA Graph Version)")
    print(f"{'='*90}")
    print(f"Configuration:")
    print(f"  num_requests        = {num_requests}")
    print(f"  tokens_per_request  = {tokens_per_request}")
    print(f"  dllm_block_size     = {dllm_block_size}")
    print(f"  chunk_sizes         = {chunk_sizes}")
    print(f"  num_heads           = {num_heads}")
    print(f"  num_kv_heads        = {num_kv_heads}")
    print(f"  head_dim            = {head_dim}")
    print(f"  page_size           = {page_size}")
    print(f"  backend             = {backend}")
    print(f"\nCUDA Graph Optimization:")
    print(f"  - Pre-create Wrappers for each step and complete plan()")
    print(f"  - CUDA Graph only captures run() operations")
    print(f"  - Reduces Python/launch overhead")
    
    # Generate complete Q, K, V for each request
    all_qs = [torch.randn(tokens_per_request, num_heads, head_dim, dtype=dtype, device=device) 
              for _ in range(num_requests)]
    all_ks = [torch.randn(tokens_per_request, num_kv_heads, head_dim, dtype=dtype, device=device) 
              for _ in range(num_requests)]
    all_vs = [torch.randn(tokens_per_request, num_kv_heads, head_dim, dtype=dtype, device=device) 
              for _ in range(num_requests)]
    
    def split_chunks(tensor, chunk_size):
        return [tensor[i*chunk_size:(i+1)*chunk_size] for i in range(tensor.shape[0] // chunk_size)]
    
    results = {}
    
    for chunk_size in chunk_sizes:
        if tokens_per_request % chunk_size != 0:
            print(f"\n[Skip] chunk_size={chunk_size} doesn't evenly divide tokens_per_request={tokens_per_request}")
            continue
        
        num_steps = tokens_per_request // chunk_size
        
        print(f"\n{'-'*90}")
        print(f"chunk_size = {chunk_size}, num_steps = {num_steps}")
        print(f"{'-'*90}")
        
        # Split chunks for each request
        qs_chunks = [split_chunks(q, chunk_size) for q in all_qs]
        ks_chunks = [split_chunks(k, chunk_size) for k in all_ks]
        vs_chunks = [split_chunks(v, chunk_size) for v in all_vs]
        
        # Pre-allocate buffers for all steps
        q_current_buffers = []
        k_current_buffers = []
        v_current_buffers = []
        for step_idx in range(num_steps):
            q_list = [qs_chunks[req_idx][step_idx] for req_idx in range(num_requests)]
            k_list = [ks_chunks[req_idx][step_idx] for req_idx in range(num_requests)]
            v_list = [vs_chunks[req_idx][step_idx] for req_idx in range(num_requests)]
            q_current_buffers.append(torch.cat(q_list, dim=0))
            k_current_buffers.append(torch.cat(k_list, dim=0))
            v_current_buffers.append(torch.cat(v_list, dim=0))
        
        # Paged KV Cache setup
        max_prefix_len = (num_steps - 1) * chunk_size
        max_pages_per_request = (max_prefix_len + page_size - 1) // page_size if max_prefix_len > 0 else 0
        total_max_pages = num_requests * max_pages_per_request
        
        if total_max_pages > 0:
            paged_kv_cache = torch.randn(
                total_max_pages, 2, page_size, num_kv_heads, head_dim,
                dtype=dtype, device=device
            )
        else:
            paged_kv_cache = None
        
        # Prepare paged kv parameters for each step
        paged_kv_params_list = []
        for step_idx in range(num_steps):
            prefix_len = step_idx * chunk_size
            if prefix_len == 0:
                paged_kv_params_list.append(None)
            else:
                pages_per_request = (prefix_len + page_size - 1) // page_size
                total_pages = num_requests * pages_per_request
                
                paged_kv_indptr = torch.tensor(
                    [i * pages_per_request for i in range(num_requests + 1)],
                    dtype=torch.int32, device=device
                )
                paged_kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
                last_page_len = prefix_len % page_size if prefix_len % page_size != 0 else page_size
                paged_kv_last_page_len = torch.full(
                    (num_requests,), last_page_len, dtype=torch.int32, device=device
                )
                
                paged_kv_params_list.append({
                    "paged_kv_cache": paged_kv_cache[:total_pages] if paged_kv_cache is not None else None,
                    "paged_kv_indptr": paged_kv_indptr,
                    "paged_kv_indices": paged_kv_indices,
                    "paged_kv_last_page_len": paged_kv_last_page_len,
                })
        
        # indptrs
        qo_indptr = torch.tensor(
            [i * chunk_size for i in range(num_requests + 1)],
            dtype=torch.int32, device=device
        )
        kv_curr_indptr = torch.tensor(
            [i * chunk_size for i in range(num_requests + 1)],
            dtype=torch.int32, device=device
        )
        
        # q_offsets
        q_offsets_list = []
        for step_idx in range(num_steps):
            prefix_len = step_idx * chunk_size
            q_offsets_list.append(torch.full((num_requests,), prefix_len, dtype=torch.int32, device=device))
        
        # Output buffers
        output_buffer = torch.empty(num_requests * chunk_size, num_heads, head_dim, dtype=dtype, device=device)
        be_output_buffers = [torch.empty_like(output_buffer) for _ in range(num_steps)]
        sg_output_buffers = [torch.empty_like(output_buffer) for _ in range(num_steps)]
        
        # LSE buffers for merge (only needed for step > 0)
        be_lse_ragged = [torch.empty(num_requests * chunk_size, num_heads, dtype=torch.float32, device=device) for _ in range(num_steps)]
        be_lse_paged = [torch.empty(num_requests * chunk_size, num_heads, dtype=torch.float32, device=device) for _ in range(num_steps)]
        sg_lse_ragged = [torch.empty(num_requests * chunk_size, num_heads, dtype=torch.float32, device=device) for _ in range(num_steps)]
        sg_lse_paged = [torch.empty(num_requests * chunk_size, num_heads, dtype=torch.float32, device=device) for _ in range(num_steps)]

        # Pre-create BE Wrappers (independent for each step)
        print(f"  [Preparation] Pre-creating {num_steps} step Wrappers for BE Cascade...")
        be_ragged_wrappers = []
        be_paged_wrappers = []
        
        for step_idx in range(num_steps):
            prefix_len = step_idx * chunk_size
            
            # Ragged Wrapper (Current Chunk)
            ws_ragged = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
            ragged_wrapper = BatchBlockExtendRaggedOffsetWrapper(
                ws_ragged,
                kv_layout="NHD",
                dllm_block_size=dllm_block_size,
                backend=backend,
            )
            ragged_wrapper.plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_curr_indptr,
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_data_type=dtype,
                sm_scale=sm_scale,
                q_offsets=q_offsets_list[step_idx],
                kv_offsets=q_offsets_list[step_idx],
            )
            be_ragged_wrappers.append(ragged_wrapper)
            
            # Paged Wrapper (Prefix) - only needed for step > 0
            # Cascade scenario: Q's block >= all prefix blocks, so mask is all 1s, use full attention
            if prefix_len > 0:
                paged_params = paged_kv_params_list[step_idx]
                ws_paged = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
                # Use native BatchPrefillWithPagedKVCacheWrapper (causal=False) instead of BlockExtend
                # Because Prefix mask is all 1s, no additional mask computation needed
                paged_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    ws_paged,
                    kv_layout="NHD",
                    backend=backend,
                )
                paged_wrapper.plan(
                    qo_indptr=qo_indptr,
                    paged_kv_indptr=paged_params["paged_kv_indptr"],
                    paged_kv_indices=paged_params["paged_kv_indices"],
                    paged_kv_last_page_len=paged_params["paged_kv_last_page_len"],
                    num_qo_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim_qk=head_dim,
                    head_dim_vo=head_dim,
                    page_size=page_size,
                    q_data_type=dtype,
                    causal=False,  # Prefix is fully visible
                )
                be_paged_wrappers.append(paged_wrapper)
            else:
                be_paged_wrappers.append(None)

        # Pre-create SG Wrappers (independent for each step)
        print(f"  [Preparation] Pre-creating {num_steps} step Wrappers for SG Cascade...")
        sg_ragged_wrappers = []
        sg_paged_wrappers = []
        
        for step_idx in range(num_steps):
            prefix_len = step_idx * chunk_size
            
            # Ragged Wrapper (Current Chunk, causal=True for SGLang style)
            ws_ragged = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
            ragged_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                ws_ragged,
                kv_layout="NHD",
                backend=backend,
            )
            ragged_wrapper.plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_curr_indptr,
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                head_dim_vo=head_dim,
                q_data_type=dtype,
                causal=True,  # SGLang: Current Chunk uses causal=True
            )
            sg_ragged_wrappers.append(ragged_wrapper)
            
            # Paged Wrapper (Prefix, causal=False) - only needed for step > 0
            if prefix_len > 0:
                paged_params = paged_kv_params_list[step_idx]
                ws_paged = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
                paged_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    ws_paged,
                    kv_layout="NHD",
                    backend=backend,
                )
                paged_wrapper.plan(
                    qo_indptr=qo_indptr,
                    paged_kv_indptr=paged_params["paged_kv_indptr"],
                    paged_kv_indices=paged_params["paged_kv_indices"],
                    paged_kv_last_page_len=paged_params["paged_kv_last_page_len"],
                    num_qo_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim_qk=head_dim,
                    head_dim_vo=head_dim,
                    page_size=page_size,
                    q_data_type=dtype,
                    causal=False,
                )
                sg_paged_wrappers.append(paged_wrapper)
            else:
                sg_paged_wrappers.append(None)
        
        torch.cuda.synchronize()

        # BE Cascade performance test (CUDA Graph)
        print(f"  [BE Cascade] Performance test (CUDA Graph)...")
        
        def run_be_cascade_with_wrappers():
            for step_idx in range(num_steps):
                q = q_current_buffers[step_idx]
                k = k_current_buffers[step_idx]
                v = v_current_buffers[step_idx]
                
                if step_idx == 0:
                    # No prefix, only ragged needed
                    be_output_buffers[step_idx].copy_(
                        be_ragged_wrappers[step_idx].run(q, k, v)
                    )
                else:
                    # Has prefix: ragged + paged + merge
                    o1, s1 = be_ragged_wrappers[step_idx].run(q, k, v, return_lse=True)
                    be_lse_ragged[step_idx].copy_(s1)
                    
                    paged_params = paged_kv_params_list[step_idx]
                    o2, s2 = be_paged_wrappers[step_idx].run(q, paged_params["paged_kv_cache"], return_lse=True)
                    be_lse_paged[step_idx].copy_(s2)
                    
                    o, _ = merge_state(o1, s1, o2, s2)
                    be_output_buffers[step_idx].copy_(o)
        
        # Warmup
        for _ in range(warmup_iters):
            run_be_cascade_with_wrappers()
        torch.cuda.synchronize()
        
        # Capture CUDA Graph
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            be_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(be_graph, stream=stream):
                run_be_cascade_with_wrappers()
        torch.cuda.synchronize()
        
        # Benchmark with CUDA Graph
        start = time.perf_counter()
        for _ in range(bench_iters):
            be_graph.replay()
        torch.cuda.synchronize()
        be_cuda_graph_time = (time.perf_counter() - start) / bench_iters * 1000
        
        # Benchmark without CUDA Graph (for comparison)
        start = time.perf_counter()
        for _ in range(bench_iters):
            run_be_cascade_with_wrappers()
        torch.cuda.synchronize()
        be_no_cuda_graph_time = (time.perf_counter() - start) / bench_iters * 1000
        
        print(f"    => CUDA Graph:  {be_cuda_graph_time:.3f} ms ({be_cuda_graph_time/num_steps:.3f} ms/step × {num_steps} steps)")
        print(f"    => No Graph:    {be_no_cuda_graph_time:.3f} ms ({be_no_cuda_graph_time/num_steps:.3f} ms/step × {num_steps} steps)")
        if be_no_cuda_graph_time > 0:
            cuda_graph_speedup = be_no_cuda_graph_time / be_cuda_graph_time
            if cuda_graph_speedup > 1:
                print(f"    => CUDA Graph speedup: {cuda_graph_speedup:.2f}x")
            else:
                print(f"    => CUDA Graph no speedup (probably dominated by kernel time)")

        # SG Cascade performance test (CUDA Graph)
        print(f"  [SG Cascade] Performance test (CUDA Graph)...")
        
        def run_sg_cascade_with_wrappers():
            for step_idx in range(num_steps):
                q = q_current_buffers[step_idx]
                k = k_current_buffers[step_idx]
                v = v_current_buffers[step_idx]
                
                if step_idx == 0:
                    # No prefix, only ragged needed
                    sg_output_buffers[step_idx].copy_(
                        sg_ragged_wrappers[step_idx].run(q, k, v)
                    )
                else:
                    # Has prefix: ragged + paged + merge
                    o1, s1 = sg_ragged_wrappers[step_idx].run(q, k, v, return_lse=True)
                    sg_lse_ragged[step_idx].copy_(s1)
                    
                    paged_params = paged_kv_params_list[step_idx]
                    o2, s2 = sg_paged_wrappers[step_idx].run(q, paged_params["paged_kv_cache"], return_lse=True)
                    sg_lse_paged[step_idx].copy_(s2)
                    
                    o, _ = merge_state(o1, s1, o2, s2)
                    sg_output_buffers[step_idx].copy_(o)
        
        # Warmup
        for _ in range(warmup_iters):
            run_sg_cascade_with_wrappers()
        torch.cuda.synchronize()
        
        # Capture CUDA Graph
        with torch.cuda.stream(stream):
            sg_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(sg_graph, stream=stream):
                run_sg_cascade_with_wrappers()
        torch.cuda.synchronize()
        
        # Benchmark with CUDA Graph
        start = time.perf_counter()
        for _ in range(bench_iters):
            sg_graph.replay()
        torch.cuda.synchronize()
        sg_cuda_graph_time = (time.perf_counter() - start) / bench_iters * 1000
        
        # Benchmark without CUDA Graph
        start = time.perf_counter()
        for _ in range(bench_iters):
            run_sg_cascade_with_wrappers()
        torch.cuda.synchronize()
        sg_no_cuda_graph_time = (time.perf_counter() - start) / bench_iters * 1000
        
        print(f"    => CUDA Graph:  {sg_cuda_graph_time:.3f} ms ({sg_cuda_graph_time/num_steps:.3f} ms/step × {num_steps} steps)")
        print(f"    => No Graph:    {sg_no_cuda_graph_time:.3f} ms ({sg_no_cuda_graph_time/num_steps:.3f} ms/step × {num_steps} steps)")
        if sg_no_cuda_graph_time > 0:
            cuda_graph_speedup = sg_no_cuda_graph_time / sg_cuda_graph_time
            if cuda_graph_speedup > 1:
                print(f"    => CUDA Graph speedup: {cuda_graph_speedup:.2f}x")
            else:
                print(f"    => CUDA Graph no speedup")
        
        # Compare BE vs SG (CUDA Graph)
        if be_cuda_graph_time > 0 and sg_cuda_graph_time > 0:
            speedup = sg_cuda_graph_time / be_cuda_graph_time
            if speedup > 1:
                print(f"  [Comparison] BE Cascade is faster by {speedup:.2f}x (CUDA Graph)")
            else:
                print(f"  [Comparison] SG Cascade is faster by {1/speedup:.2f}x (CUDA Graph)")
        
        results[f"chunk{chunk_size}"] = {
            "chunk_size": chunk_size,
            "num_steps": num_steps,
            "be_cuda_graph_ms": be_cuda_graph_time,
            "be_no_cuda_graph_ms": be_no_cuda_graph_time,
            "sg_cuda_graph_ms": sg_cuda_graph_time,
            "sg_no_cuda_graph_ms": sg_no_cuda_graph_time,
            "speedup_be_over_sg_cuda_graph": sg_cuda_graph_time / be_cuda_graph_time if be_cuda_graph_time > 0 else 0,
        }
        
        # Cleanup
        del be_graph, sg_graph
        torch.cuda.empty_cache()
    
    # ================================================================
    # Results summary
    # ================================================================
    print(f"\n{'='*90}")
    print(f"Results Summary (CUDA Graph Version)")
    print(f"{'='*90}")
    
    print(f"\n{'chunk':>8} | {'steps':>6} | {'BE(cuda_graph)':>10} | {'BE(No)':>10} | {'SG(cuda_graph)':>10} | {'SG(No)':>10} | {'BE/SG':>8}")
    print(f"{'-'*8}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    
    for key in sorted(results.keys(), key=lambda k: results[k]["chunk_size"]):
        r = results[key]
        print(f"{r['chunk_size']:>8} | {r['num_steps']:>6} | {r['be_cuda_graph_ms']:>8.3f}ms | {r['be_no_cuda_graph_ms']:>8.3f}ms | "
              f"{r['sg_cuda_graph_ms']:>8.3f}ms | {r['sg_no_cuda_graph_ms']:>8.3f}ms | {r['speedup_be_over_sg_cuda_graph']:>7.2f}x")
    
    print(f"\nNotes:")
    print(f"  - BE(cuda_graph): batch_block_extend Wrapper.run() with CUDA Graph")
    print(f"  - BE(No): batch_block_extend Wrapper.run() without CUDA Graph")
    print(f"  - SG(cuda_graph): sglang_style Wrapper.run() with CUDA Graph")
    print(f"  - SG(No): sglang_style Wrapper.run() without CUDA Graph")
    print(f"  - BE/SG: Speed ratio of BE vs SG (>1 means BE is faster)")
    print(f"  - CUDA Graph optimization: pre-plan(), only capture run()")
    
    return results


def test_heterogeneous_prefix_batch(
    verbose: bool = True,
    backend: str = "fa2",
):
    """
    Heterogeneous prefix test: different requests have different prefix lengths
    
    Scenario reproduction:
      - Req 0: Already prefilled, has prefix (kv_len=128, q_offset=64)
      - Req 1: New request, no prefix (kv_len=32, q_offset=0)
      - Both requests concatenated together for batch block-extend attention operator
    
    Test purposes:
      1. Verify whether operator supports heterogeneous kv_len input
      2. Check for out-of-bounds memory access issues
      3. Verify precision correctness
    
    Reference implementation: Each request computed independently with custom_mask, then concatenated
    """
    from flashinfer.dllm import BatchBlockExtendRaggedOffsetWrapper
    from flashinfer.prefill import single_prefill_with_kv_cache
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    tol = 1e-2
    
    print(f"\n{'='*100}")
    print(f"Heterogeneous Prefix Test: Different requests have different prefix lengths")
    print(f"{'='*100}")
    print(f"Test backend: {backend}")
    print(f"Precision tolerance: {tol}")
    
    # Test configuration: Heterogeneous prefix scenarios
    # Each config is a list of requests, each request has different qo_len, kv_len, q_offset
    test_configs = [
        # Scenario 1: One with prefix, one without
        {
            "name": "Req0(has_prefix) + Req1(no_prefix)",
            "dllm_block_size": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {"qo_len": 64, "kv_len": 128, "q_offset": 64},  # Req0: step_2, already has 64 tokens prefix
                {"qo_len": 32, "kv_len": 32, "q_offset": 0},    # Req1: step_0, no prefix
            ],
        },
        # Scenario 2: Three requests, different steps
        {
            "name": "Req0(step_3) + Req1(step_0) + Req2(step_1)",
            "dllm_block_size": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {"qo_len": 32, "kv_len": 128, "q_offset": 96},  # Req0: step_3
                {"qo_len": 32, "kv_len": 32, "q_offset": 0},    # Req1: step_0
                {"qo_len": 32, "kv_len": 64, "q_offset": 32},   # Req2: step_1
            ],
        },
        # Scenario 3: Two requests, larger kv_len difference
        {
            "name": "Req0(kv=256) + Req1(kv=32)",
            "dllm_block_size": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {"qo_len": 32, "kv_len": 256, "q_offset": 224}, # Req0: very long prefix
                {"qo_len": 32, "kv_len": 32, "q_offset": 0},    # Req1: no prefix
            ],
        },
        # Scenario 4: Two requests, qo_len also different
        {
            "name": "Req0(qo=64,kv=128) + Req1(qo=32,kv=32)",
            "dllm_block_size": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {"qo_len": 64, "kv_len": 128, "q_offset": 64},  # Req0
                {"qo_len": 32, "kv_len": 32, "q_offset": 0},    # Req1
            ],
        },
        # Scenario 5: Four requests, mixed scenario
        {
            "name": "4_requests_mixed_scenario",
            "dllm_block_size": 64,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {"qo_len": 64, "kv_len": 256, "q_offset": 192}, # Req0: step_3
                {"qo_len": 64, "kv_len": 64, "q_offset": 0},    # Req1: step_0
                {"qo_len": 64, "kv_len": 128, "q_offset": 64},  # Req2: step_1
                {"qo_len": 64, "kv_len": 192, "q_offset": 128}, # Req3: step_2
            ],
        },
        # Scenario 6: Similar to SGLang batch inference - long prompt + short prompt
        {
            "name": "Long_short_prompt_mix(512_vs_32)",
            "dllm_block_size": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {"qo_len": 512, "kv_len": 512, "q_offset": 0},  # Req0: long prompt (e.g., math problem)
                {"qo_len": 32, "kv_len": 32, "q_offset": 0},    # Req1: short prompt (e.g., "Say hello")
            ],
        },
        # Scenario 7: Extreme difference - super long vs super short
        {
            "name": "Extreme_difference(1024_vs_16)",
            "dllm_block_size": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {"qo_len": 1024, "kv_len": 1024, "q_offset": 0}, # Req0: super long prompt
                {"qo_len": 16, "kv_len": 16, "q_offset": 0},     # Req1: super short prompt
            ],
        },
        # Scenario 8: Three requests, increasing lengths
        {
            "name": "Three_requests_increasing_length(64,256,512)",
            "dllm_block_size": 64,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {"qo_len": 64, "kv_len": 64, "q_offset": 0},     # Req0: short
                {"qo_len": 256, "kv_len": 256, "q_offset": 0},   # Req1: medium
                {"qo_len": 512, "kv_len": 512, "q_offset": 0},   # Req2: long
            ],
        },
        # Scenario 9: Mixed prefill stages + different prompt lengths
        {
            "name": "Mixed_prefill_stages+different_prompt_lengths",
            "dllm_block_size": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {"qo_len": 32, "kv_len": 512, "q_offset": 480},  # Req0: long prompt, step_15
                {"qo_len": 32, "kv_len": 32, "q_offset": 0},     # Req1: short prompt, step_0
                {"qo_len": 32, "kv_len": 128, "q_offset": 96},   # Req2: medium prompt, step_3
            ],
        },
    ]
    
    results = []
    all_pass = True
    
    for cfg_idx, cfg in enumerate(test_configs):
        dllm_block_size = cfg["dllm_block_size"]
        num_heads = cfg["num_heads"]
        num_kv_heads = cfg["num_kv_heads"]
        head_dim = cfg["head_dim"]
        requests = cfg["requests"]
        num_requests = len(requests)
        sm_scale = 1.0 / math.sqrt(head_dim)
        
        print(f"\n  [Test {cfg_idx:02d}] {cfg['name']}")
        print(f"           B={dllm_block_size}, heads={num_heads}/{num_kv_heads}, dim={head_dim}")
        
        # Generate data for each request
        qs = []
        ks = []
        vs = []
        for req in requests:
            qo_len = req["qo_len"]
            kv_len = req["kv_len"]
            qs.append(torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device))
            ks.append(torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device))
            vs.append(torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device))
        
        # Reference implementation: compute each request independently
        ref_outputs = []
        for req_idx, req in enumerate(requests):
            qo_len = req["qo_len"]
            kv_len = req["kv_len"]
            q_offset = req["q_offset"]
            
            # Build custom_mask
            q_pos = torch.arange(qo_len, device=device) + q_offset
            k_pos = torch.arange(kv_len, device=device)
            q_block = q_pos.unsqueeze(1) // dllm_block_size
            k_block = k_pos.unsqueeze(0) // dllm_block_size
            mask_2d = (q_block >= k_block).to(torch.uint8)
            
            ref_output = single_prefill_with_kv_cache(
                qs[req_idx], ks[req_idx], vs[req_idx],
                custom_mask=mask_2d,
                sm_scale=sm_scale,
                backend="fa2",
            )
            ref_outputs.append(ref_output)
        
        # Concatenate reference outputs
        ref_output_cat = torch.cat(ref_outputs, dim=0)
        
        # Build batch input
        q_cat = torch.cat(qs, dim=0)
        k_cat = torch.cat(ks, dim=0)
        v_cat = torch.cat(vs, dim=0)
        
        # Build indptr
        qo_lens = [req["qo_len"] for req in requests]
        kv_lens = [req["kv_len"] for req in requests]
        q_offsets_list = [req["q_offset"] for req in requests]
        
        qo_indptr = torch.tensor([0] + list(torch.cumsum(torch.tensor(qo_lens), dim=0).numpy()), dtype=torch.int32, device=device)
        kv_indptr = torch.tensor([0] + list(torch.cumsum(torch.tensor(kv_lens), dim=0).numpy()), dtype=torch.int32, device=device)
        q_offsets = torch.tensor(q_offsets_list, dtype=torch.int32, device=device)
        
        if verbose:
            print(f"           qo_indptr: {qo_indptr.tolist()}")
            print(f"           kv_indptr: {kv_indptr.tolist()}")
            print(f"           q_offsets: {q_offsets.tolist()}")
        
        # Object under test: BatchBlockExtendRaggedOffsetWrapper
        try:
            workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
            wrapper = BatchBlockExtendRaggedOffsetWrapper(
                workspace, kv_layout="NHD", dllm_block_size=dllm_block_size, backend=backend
            )
            wrapper.plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_data_type=dtype,
                sm_scale=sm_scale,
                q_offsets=q_offsets,
            )
            bbe_output = wrapper.run(q_cat, k_cat, v_cat)
            
            # Calculate precision differences
            max_diff = (bbe_output - ref_output_cat).abs().max().item()
            mean_diff = (bbe_output - ref_output_cat).abs().mean().item()
            passed = max_diff < tol
            
            status = "PASS" if passed else "FAIL"
            print(f"           BBE-{backend.upper()}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} [{status}]")
            
            if not passed:
                all_pass = False
                # Print detailed diff for each request
                start_idx = 0
                for req_idx, req in enumerate(requests):
                    qo_len = req["qo_len"]
                    end_idx = start_idx + qo_len
                    req_diff = (bbe_output[start_idx:end_idx] - ref_outputs[req_idx]).abs().max().item()
                    print(f"             Req{req_idx} (qo={qo_len}, kv={req['kv_len']}, q_off={req['q_offset']}): max_diff={req_diff:.6f}")
                    start_idx = end_idx
            
            results.append({
                "config_idx": cfg_idx,
                "name": cfg["name"],
                "max_diff": max_diff,
                "passed": passed,
                "error": None,
            })
            
            del workspace, wrapper
            
        except Exception as e:
            print(f"           BBE-{backend.upper()}: ERROR - {str(e)}")
            results.append({
                "config_idx": cfg_idx,
                "name": cfg["name"],
                "max_diff": float('inf'),
                "passed": False,
                "error": str(e),
            })
            all_pass = False
        
        torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*100}")
    print(f"Heterogeneous Prefix Test Summary")
    print(f"{'='*100}")
    
    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)
    print(f"  Passed: {passed_count}/{total_count}")
    
    if not all_pass:
        print(f"\n  Failed tests:")
        for r in results:
            if not r["passed"]:
                if r["error"]:
                    print(f"    - [{r['config_idx']:02d}] {r['name']}: ERROR - {r['error']}")
                else:
                    print(f"    - [{r['config_idx']:02d}] {r['name']}: max_diff={r['max_diff']:.6f}")
    
    return results


def test_cascade_current_chunk_batch(
    verbose: bool = True,
    backend: str = "fa2",
):
    """
    Test complete three-stage Cascade Attention (simulating SGLang DLLM flow)
    
    Three-stage flow:
      Stage 1 (prefix): BatchBlockExtendRaggedOffsetWrapper computes prefix
        - K/V: [0, prefix_len)
        - kv_offset = 0
      Stage 2 (current chunk): BatchBlockExtendRaggedOffsetWrapper computes current chunk
        - K/V: [prefix_len, prefix_len + chunk_len) (only current chunk)
        - kv_offset = prefix_len
      Stage 3 (merge): merge_state(o1, s1, o2, s2)
    
    Mask rule: mask[q, k] = (q_global // B) >= (k_global // B)
    
    Key points:
      - Stage 2's K/V doesn't start from 0, needs kv_offset
      - Uses blockwise extend mask (not causal mask)
    """
    device = "cuda"
    dtype = torch.bfloat16
    tol = 0.01 if backend == "fa3" else 0.01
    
    print(f"\n{'='*100}")
    print(f"Three-stage Cascade Attention Test (prefix + current_chunk + merge)")
    print(f"{'='*100}")
    print(f"Test backend: {backend}")
    print(f"Precision tolerance: {tol}")
    
    # Test configuration: each request has prefix_len and chunk_len
    test_configs = [
        # Scenario 1: Two requests, one with prefix, one without
        {
            "name": "Req0(has_prefix) + Req1(no_prefix)",
            "dllm_block_size": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {"prefix_len": 64, "chunk_len": 32},   # Req0: step_2, prefix [0,64), chunk [64,96)
                {"prefix_len": 0, "chunk_len": 32},    # Req1: step_0, no prefix, chunk [0,32)
            ],
        },
        # Scenario 2: Three requests, different steps
        {
            "name": "Req0(step_3) + Req1(step_0) + Req2(step_1)",
            "dllm_block_size": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {"prefix_len": 96, "chunk_len": 32},   # Req0: step_3
                {"prefix_len": 0, "chunk_len": 32},    # Req1: step_0
                {"prefix_len": 32, "chunk_len": 32},   # Req2: step_1
            ],
        },
        # Scenario 3: Large prefix
        {
            "name": "Req0(large_prefix=256) + Req1(no_prefix)",
            "dllm_block_size": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {"prefix_len": 256, "chunk_len": 32},  # Req0: step_8
                {"prefix_len": 0, "chunk_len": 32},    # Req1: step_0
            ],
        },
        # Scenario 4: chunk_len != block_size
        {
            "name": "Req0(chunk=64) + Req1(chunk=32)",
            "dllm_block_size": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {"prefix_len": 64, "chunk_len": 64},   # Req0: 2 blocks chunk
                {"prefix_len": 0, "chunk_len": 32},    # Req1: 1 block chunk
            ],
        },
        # Scenario 5: Four requests mixed
        {
            "name": "4_requests_mixed(step_0,1,2,3)",
            "dllm_block_size": 64,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {"prefix_len": 0, "chunk_len": 64},    # Req0: step_0
                {"prefix_len": 64, "chunk_len": 64},   # Req1: step_1
                {"prefix_len": 128, "chunk_len": 64},  # Req2: step_2
                {"prefix_len": 192, "chunk_len": 64},  # Req3: step_3
            ],
        },
    ]
    
    results = []
    all_pass = True
    
    for cfg_idx, cfg in enumerate(test_configs):
        dllm_block_size = cfg["dllm_block_size"]
        num_heads = cfg["num_heads"]
        num_kv_heads = cfg["num_kv_heads"]
        head_dim = cfg["head_dim"]
        requests = cfg["requests"]
        sm_scale = 1.0 / math.sqrt(head_dim)
        
        print(f"\n  [Test {cfg_idx:02d}] {cfg['name']}")
        print(f"           B={dllm_block_size}, heads={num_heads}/{num_kv_heads}, dim={head_dim}")
        
        # Generate data for each request
        # Each request has: Q(chunk_len), K_prefix(prefix_len), V_prefix(prefix_len), K_chunk(chunk_len), V_chunk(chunk_len)
        qs = []           # Q: current chunk's query
        k_prefixes = []   # K_prefix: K of prefix part
        v_prefixes = []   # V_prefix: V of prefix part
        k_chunks = []     # K_chunk: K of current chunk
        v_chunks = []     # V_chunk: V of current chunk
        
        for req in requests:
            prefix_len = req["prefix_len"]
            chunk_len = req["chunk_len"]
            qs.append(torch.randn(chunk_len, num_heads, head_dim, dtype=dtype, device=device))
            if prefix_len > 0:
                k_prefixes.append(torch.randn(prefix_len, num_kv_heads, head_dim, dtype=dtype, device=device))
                v_prefixes.append(torch.randn(prefix_len, num_kv_heads, head_dim, dtype=dtype, device=device))
            else:
                k_prefixes.append(None)
                v_prefixes.append(None)
            k_chunks.append(torch.randn(chunk_len, num_kv_heads, head_dim, dtype=dtype, device=device))
            v_chunks.append(torch.randn(chunk_len, num_kv_heads, head_dim, dtype=dtype, device=device))
        
        # Reference implementation: compute each request independently (full KV + blockwise mask)
        ref_outputs = []
        for req_idx, req in enumerate(requests):
            prefix_len = req["prefix_len"]
            chunk_len = req["chunk_len"]
            total_kv_len = prefix_len + chunk_len
            
            # Concatenate full K/V
            if prefix_len > 0:
                k_full = torch.cat([k_prefixes[req_idx], k_chunks[req_idx]], dim=0)
                v_full = torch.cat([v_prefixes[req_idx], v_chunks[req_idx]], dim=0)
            else:
                k_full = k_chunks[req_idx]
                v_full = v_chunks[req_idx]
            
            # Construct blockwise extend mask
            # Q: [prefix_len, prefix_len + chunk_len)
            # K: [0, prefix_len + chunk_len)
            q_offset = prefix_len
            q_pos = torch.arange(chunk_len, device=device) + q_offset
            k_pos = torch.arange(total_kv_len, device=device)  # K starts from 0
            q_block = q_pos.unsqueeze(1) // dllm_block_size
            k_block = k_pos.unsqueeze(0) // dllm_block_size
            mask_2d = (q_block >= k_block).to(torch.uint8)
            
            ref_output = single_prefill_with_kv_cache(
                qs[req_idx], k_full, v_full,
                custom_mask=mask_2d,
                sm_scale=sm_scale,
                backend="fa2",
            )
            ref_outputs.append(ref_output)
        
        ref_output_cat = torch.cat(ref_outputs, dim=0)
        
        if verbose:
            for req_idx, req in enumerate(requests):
                prefix_len = req["prefix_len"]
                chunk_len = req["chunk_len"]
                print(f"           Req{req_idx}: prefix_len={prefix_len}, chunk_len={chunk_len}, q_offset={prefix_len}, kv_offset={prefix_len}")
        
        # Target under test: three-stage Cascade Attention
        try:
            cascade_outputs = []
            workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
            
            for req_idx, req in enumerate(requests):
                prefix_len = req["prefix_len"]
                chunk_len = req["chunk_len"]
                q = qs[req_idx]
                
                # Stage 1: prefix (if exists)
                if prefix_len > 0:
                    # Construct prefix indptr
                    prefix_qo_indptr = torch.tensor([0, chunk_len], dtype=torch.int32, device=device)
                    prefix_kv_indptr = torch.tensor([0, prefix_len], dtype=torch.int32, device=device)
                    prefix_q_offsets = torch.tensor([prefix_len], dtype=torch.int32, device=device)  # Q's global position
                    prefix_kv_offsets = torch.tensor([0], dtype=torch.int32, device=device)  # prefix K/V starts from 0
                    
                    prefix_wrapper = BatchBlockExtendRaggedOffsetWrapper(
                        workspace, kv_layout="NHD", dllm_block_size=dllm_block_size, backend=backend
                    )
                    prefix_wrapper.plan(
                        qo_indptr=prefix_qo_indptr,
                        kv_indptr=prefix_kv_indptr,
                        num_qo_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        q_data_type=dtype,
                        sm_scale=sm_scale,
                        q_offsets=prefix_q_offsets,
                        kv_offsets=prefix_kv_offsets,
                    )
                    o1, s1 = prefix_wrapper.run(q, k_prefixes[req_idx], v_prefixes[req_idx], return_lse=True)
                    del prefix_wrapper
                else:
                    o1 = None
                    s1 = None
                
                # Stage 2: current chunk
                chunk_qo_indptr = torch.tensor([0, chunk_len], dtype=torch.int32, device=device)
                chunk_kv_indptr = torch.tensor([0, chunk_len], dtype=torch.int32, device=device)
                chunk_q_offsets = torch.tensor([prefix_len], dtype=torch.int32, device=device)   # Q's global position
                chunk_kv_offsets = torch.tensor([prefix_len], dtype=torch.int32, device=device)  # chunk K/V's global position
                
                chunk_wrapper = BatchBlockExtendRaggedOffsetWrapper(
                    workspace, kv_layout="NHD", dllm_block_size=dllm_block_size, backend=backend
                )
                chunk_wrapper.plan(
                    qo_indptr=chunk_qo_indptr,
                    kv_indptr=chunk_kv_indptr,
                    num_qo_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    q_data_type=dtype,
                    sm_scale=sm_scale,
                    q_offsets=chunk_q_offsets,
                    kv_offsets=chunk_kv_offsets,
                )
                o2, s2 = chunk_wrapper.run(q, k_chunks[req_idx], v_chunks[req_idx], return_lse=True)
                del chunk_wrapper
                
                # Stage 3: merge
                if o1 is not None:
                    o_merged, _ = merge_state(o1, s1, o2, s2)
                    cascade_outputs.append(o_merged)
                else:
                    cascade_outputs.append(o2)
            
            cascade_output_cat = torch.cat(cascade_outputs, dim=0)
            
            # Compute precision difference
            max_diff = (cascade_output_cat - ref_output_cat).abs().max().item()
            mean_diff = (cascade_output_cat - ref_output_cat).abs().mean().item()
            passed = max_diff < tol
            
            status = "PASS" if passed else "FAIL"
            print(f"           Cascade-{backend.upper()}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} [{status}]")
            
            if not passed:
                all_pass = False
            
            results.append({
                "config_idx": cfg_idx,
                "name": cfg["name"],
                "max_diff": max_diff,
                "passed": passed,
                "error": None,
            })
            
            del workspace
            
        except Exception as e:
            import traceback
            print(f"           Cascade-{backend.upper()}: ERROR - {str(e)}")
            if verbose:
                traceback.print_exc()
            results.append({
                "config_idx": cfg_idx,
                "name": cfg["name"],
                "max_diff": float('inf'),
                "passed": False,
                "error": str(e),
            })
            all_pass = False
        
        torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*100}")
    print(f"Three-stage Cascade Attention Test Summary")
    print(f"{'='*100}")
    
    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)
    print(f"  Passed: {passed_count}/{total_count}")
    
    if not all_pass:
        print(f"\n  Failed tests:")
        for r in results:
            if not r["passed"]:
                if r["error"]:
                    print(f"    - [{r['config_idx']:02d}] {r['name']}: ERROR - {r['error']}")
                else:
                    print(f"    - [{r['config_idx']:02d}] {r['name']}: max_diff={r['max_diff']:.6f}")
    
    return results


def test_cascade_precision_alignment(
    verbose: bool = True,
):
    """
    Test block_extend_cascade precision alignment (single request version)
    
    Comparison targets:
      1. block_extend_cascade: uses block_extend_attention_with_offset (Current Chunk)
      2. Reference implementation: uses single_prefill_with_kv_cache(causal=True) (Current Chunk)
    
    When chunk_size == dllm_block_size:
      - Causal mask ≡ Block Extend mask
      - The outputs of both implementations should be completely consistent
    
    Test coverage:
      1. Different dllm_block_size: [32, 64, 128]
      2. Different num_steps (with/without prefix)
      3. Different head configurations (MHA, GQA, MQA)
      4. Different head_dim
    """
    from flashinfer.dllm import block_extend_cascade
    from flashinfer.cascade import merge_state_in_place
    from flashinfer.prefill import single_prefill_with_kv_cache
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    tol = 1e-2  # Precision tolerance
    
    print(f"\n{'='*100}")
    print(f"Single Request Cascade Precision Alignment Test: block_extend_cascade vs custom_mask Reference Implementation")
    print(f"{'='*100}")
    print(f"Test condition: chunk_size == dllm_block_size")
    print(f"Note: Block Extend mask != Causal mask (all visible within block, not lower triangular)")
    print(f"Precision tolerance: {tol}")
    
    # Test configurations
    test_configs = [
        # Basic tests: different dllm_block_size
        {"dllm_block_size": 32, "num_steps": 4, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"dllm_block_size": 64, "num_steps": 4, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"dllm_block_size": 128, "num_steps": 2, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        
        # More steps (longer sequences)
        {"dllm_block_size": 32, "num_steps": 8, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"dllm_block_size": 64, "num_steps": 8, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        
        # Different head configurations (MHA, GQA, MQA)
        {"dllm_block_size": 32, "num_steps": 4, "num_heads": 32, "num_kv_heads": 32, "head_dim": 128},  # MHA
        {"dllm_block_size": 32, "num_steps": 4, "num_heads": 32, "num_kv_heads": 4, "head_dim": 128},   # GQA-8
        {"dllm_block_size": 32, "num_steps": 4, "num_heads": 32, "num_kv_heads": 1, "head_dim": 128},   # MQA
        
        # Different head_dim
        {"dllm_block_size": 32, "num_steps": 4, "num_heads": 32, "num_kv_heads": 8, "head_dim": 64},
        {"dllm_block_size": 32, "num_steps": 4, "num_heads": 16, "num_kv_heads": 4, "head_dim": 256},
        
        # Boundary tests: single step (no prefix)
        {"dllm_block_size": 32, "num_steps": 1, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"dllm_block_size": 64, "num_steps": 1, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
    ]
    
    results = []
    
    for cfg_idx, cfg in enumerate(test_configs):
        dllm_block_size = cfg["dllm_block_size"]
        num_steps = cfg["num_steps"]
        num_heads = cfg["num_heads"]
        num_kv_heads = cfg["num_kv_heads"]
        head_dim = cfg["head_dim"]
        
        chunk_size = dllm_block_size  # Key: chunk_size == dllm_block_size
        tokens_per_request = num_steps * chunk_size
        sm_scale = 1.0 / math.sqrt(head_dim)
        
        # Generate test data for full sequence
        q_full = torch.randn(tokens_per_request, num_heads, head_dim, dtype=dtype, device=device)
        k_full = torch.randn(tokens_per_request, num_kv_heads, head_dim, dtype=dtype, device=device)
        v_full = torch.randn(tokens_per_request, num_kv_heads, head_dim, dtype=dtype, device=device)
        
        def split_chunks(tensor, chunk_size):
            return [tensor[i*chunk_size:(i+1)*chunk_size] for i in range(tensor.shape[0] // chunk_size)]
        
        qs_chunks = split_chunks(q_full, chunk_size)
        ks_chunks = split_chunks(k_full, chunk_size)
        vs_chunks = split_chunks(v_full, chunk_size)
        
        max_diff = 0.0
        step_diffs = []
        
        for step_idx in range(num_steps):
            q_current = qs_chunks[step_idx]
            k_current = ks_chunks[step_idx]
            v_current = vs_chunks[step_idx]
            
            # Prefix (KV from all previous chunks)
            if step_idx > 0:
                k_prefix = k_full[:step_idx * chunk_size]
                v_prefix = v_full[:step_idx * chunk_size]
            else:
                k_prefix = None
                v_prefix = None

            # Target under test: block_extend_cascade
            be_out = block_extend_cascade(
                q=q_current,
                k_current=k_current,
                v_current=v_current,
                k_prefix=k_prefix,
                v_prefix=v_prefix,
                dllm_block_size=dllm_block_size,
                sm_scale=sm_scale,
                return_lse=False,
                backend="fa2",
            )

            # Reference: compute Block Extend Attention using custom_mask
            # Note: Block Extend mask != Causal mask
            # Block Extend: mask[q,k] = ((q+offset)//B) >= (k//B)
            # When chunk_size == dllm_block_size, all positions within chunk are visible (not lower triangular)
            prefix_len = step_idx * chunk_size
            if k_prefix is None:
                # No prefix, directly use Block Extend mask
                ref_out = compute_block_extend_reference(
                    q_current, k_current, v_current,
                    dllm_block_size=dllm_block_size,
                    q_offset=prefix_len,
                    sm_scale=sm_scale,
                )
            else:
                # With prefix: Current Chunk (Block Extend) + Prefix (fully visible) + merge
                # Current chunk: q_offset = prefix_len, kv starts from prefix_len
                
                # Construct Block Extend mask for current chunk
                qo_len = q_current.shape[0]
                kv_len = k_current.shape[0]
                q_pos = torch.arange(qo_len, device=q_current.device) + prefix_len
                k_pos = torch.arange(kv_len, device=q_current.device) + prefix_len  # kv also starts from prefix_len
                q_block = q_pos.unsqueeze(1) // dllm_block_size
                k_block = k_pos.unsqueeze(0) // dllm_block_size
                mask_current = (q_block >= k_block).to(torch.uint8)
                
                o1, s1 = single_prefill_with_kv_cache(
                    q_current, k_current, v_current,
                    custom_mask=mask_current,
                    sm_scale=sm_scale,
                    return_lse=True,
                )
                
                # Prefix: q_offset = prefix_len, kv starts from 0 (fully visible)
                o2, s2 = single_prefill_with_kv_cache(
                    q_current, k_prefix, v_prefix,
                    causal=False,
                    sm_scale=sm_scale,
                    return_lse=True,
                )
                merge_state_in_place(o1, s1, o2, s2)
                ref_out = o1
            
            step_diff = (be_out - ref_out).abs().max().item()
            step_diffs.append(step_diff)
            max_diff = max(max_diff, step_diff)
        
        test_pass = max_diff < tol
        
        result = {
            "config_idx": cfg_idx,
            "dllm_block_size": dllm_block_size,
            "num_steps": num_steps,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "max_diff": max_diff,
            "step_diffs": step_diffs,
            "pass": test_pass,
        }
        results.append(result)
        
        if verbose:
            status = "PASS" if test_pass else "FAIL"
            print(f"\n  [Test {cfg_idx:02d}] B={dllm_block_size:3d}, steps={num_steps}, "
                  f"heads={num_heads}/{num_kv_heads}, dim={head_dim}")
            print(f"           max_diff={max_diff:.6f} [{status}]")
            if not test_pass:
                print(f"           step_diffs: {[f'{d:.6f}' for d in step_diffs]}")
        
        torch.cuda.empty_cache()

    print(f"\n{'='*100}")
    print(f"Precision Alignment Test Summary")
    print(f"{'='*100}")
    
    total_tests = len(results)
    pass_count = sum(1 for r in results if r["pass"])
    
    print(f"\n  Total tests: {total_tests}")
    print(f"  Passed:      {pass_count}/{total_tests} PASS")
    print(f"  max_diff (all tests): {max(r['max_diff'] for r in results):.6f}")

    failed = [r for r in results if not r["pass"]]
    if failed:
        print(f"\n  [Failed Test Details]")
        for r in failed:
            print(f"    Test {r['config_idx']:02d}: B={r['dllm_block_size']}, steps={r['num_steps']}, "
                  f"heads={r['num_heads']}/{r['num_kv_heads']}, dim={r['head_dim']}, max_diff={r['max_diff']:.6f}")
    
    overall_pass = all(r["pass"] for r in results)
    overall_status = "ALL TESTS PASSED" if overall_pass else "SOME TESTS FAILED"
    print(f"\n  Overall Result: {overall_status}")
    
    return {
        "results": results,
        "overall_pass": overall_pass,
    }


def test_sglang_vs_block_extend_cascade(
    num_steps: int = 4,
    dllm_block_size: int = 64,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 16,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    verbose: bool = True,
    backend: str = "fa2",
):
    """
    sglang_style_cascade_attention vs block_extend_cascade precision and performance comparison
    
    Key Design:
    ═══════════════════════════════════════════════════════════════════════════════
    Ensure completely identical inputs:
      1. Use the same Q, K_current, V_current and K_prefix, V_prefix data
      2. sglang_style_cascade_attention: K_prefix/V_prefix converted to Paged KV Cache format
      3. block_extend_cascade: K_prefix/V_prefix uses contiguous storage format
    
    Comparison targets:
    ═══════════════════════════════════════════════════════════════════════════════
      - sglang_style_cascade_attention (batch version):
        * Current Chunk: BatchPrefillWithRaggedKVCacheWrapper (causal=False)
        * Prefix: BatchPrefillWithPagedKVCacheWrapper (causal=False)
        * Uses Paged KV Cache to store prefix
    
      - block_extend_cascade (single request version):
        * Current Chunk: block_extend_attention_with_offset (Block Extend mask)
        * Prefix: single_prefill_with_kv_cache (causal=False)
        * Uses contiguous memory to store prefix
    
    Applicable conditions:
    ═══════════════════════════════════════════════════════════════════════════════
      chunk_size == dllm_block_size (when causal mask = block_extend mask)
    """
    from flashinfer.dllm import (
        sglang_style_cascade_attention,
        block_extend_cascade,
    )
    import time as time_module
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    tol = 1e-2  # Precision tolerance
    
    chunk_size = dllm_block_size  # Key: chunk_size == dllm_block_size
    tokens_per_request = num_steps * chunk_size
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    print(f"\n{'='*100}")
    print(f"sglang_style_cascade_attention vs block_extend_cascade Precision and Performance Comparison")
    print(f"{'='*100}")
    print(f"Configuration:")
    print(f"  num_steps           = {num_steps}")
    print(f"  dllm_block_size     = {dllm_block_size}")
    print(f"  chunk_size          = {chunk_size} (= dllm_block_size)")
    print(f"  tokens_per_request  = {tokens_per_request}")
    print(f"  num_heads           = {num_heads}")
    print(f"  num_kv_heads        = {num_kv_heads}")
    print(f"  head_dim            = {head_dim}")
    print(f"  page_size           = {page_size}")
    print(f"  backend             = {backend}")
    print(f"  Precision tolerance = {tol}")
    print(f"\nComparison implementations:")
    print(f"  sglang_style_cascade_attention: Ragged (causal=False) + Paged (causal=False) + merge_state")
    print(f"  block_extend_cascade:        BlockExtend (with offset) + single_prefill (causal=False) + merge_state")

    q_full = torch.randn(tokens_per_request, num_heads, head_dim, dtype=dtype, device=device)
    k_full = torch.randn(tokens_per_request, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_full = torch.randn(tokens_per_request, num_kv_heads, head_dim, dtype=dtype, device=device)
    
    def split_chunks(tensor, chunk_size):
        return [tensor[i*chunk_size:(i+1)*chunk_size] for i in range(tensor.shape[0] // chunk_size)]
    
    qs_chunks = split_chunks(q_full, chunk_size)
    ks_chunks = split_chunks(k_full, chunk_size)
    vs_chunks = split_chunks(v_full, chunk_size)
    
    # Workspace buffer for sglang_style
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    def create_paged_kv_cache_from_prefix(k_prefix, v_prefix, page_size):
        """
        Convert contiguous K/V prefix to Paged KV Cache format
        
        Args:
            k_prefix: [prefix_len, num_kv_heads, head_dim]
            v_prefix: [prefix_len, num_kv_heads, head_dim]
            page_size: Page size
        
        Returns:
            paged_kv_cache: [num_pages, 2, page_size, num_kv_heads, head_dim]
            paged_kv_indptr: [2] - [0, num_pages]
            paged_kv_indices: [num_pages] - [0, 1, ..., num_pages-1]
            paged_kv_last_page_len: [1] - Valid length of last page
        """
        prefix_len = k_prefix.size(0)
        num_kv_heads = k_prefix.size(1)
        head_dim = k_prefix.size(2)
        device = k_prefix.device
        dtype = k_prefix.dtype
        
        # Calculate number of pages needed
        num_pages = (prefix_len + page_size - 1) // page_size
        last_page_len = prefix_len - (num_pages - 1) * page_size if num_pages > 0 else 0
        
        # Create Paged KV Cache: [num_pages, 2, page_size, num_kv_heads, head_dim]
        paged_kv_cache = torch.zeros(
            num_pages, 2, page_size, num_kv_heads, head_dim,
            dtype=dtype, device=device
        )
        
        # Fill data
        for page_idx in range(num_pages):
            start = page_idx * page_size
            end = min(start + page_size, prefix_len)
            actual_len = end - start
            
            # K: paged_kv_cache[page_idx, 0, :actual_len, :, :]
            paged_kv_cache[page_idx, 0, :actual_len, :, :] = k_prefix[start:end]
            # V: paged_kv_cache[page_idx, 1, :actual_len, :, :]
            paged_kv_cache[page_idx, 1, :actual_len, :, :] = v_prefix[start:end]
        
        # indptr, indices, last_page_len
        paged_kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
        paged_kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
        paged_kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=device)
        
        return paged_kv_cache, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len

    print(f"\n{'='*100}")
    print(f"Precision Comparison (Step by Step)")
    print(f"{'='*100}")
    
    max_diff_all_steps = 0.0
    step_diffs = []
    
    for step_idx in range(num_steps):
        q_current = qs_chunks[step_idx]
        k_current = ks_chunks[step_idx]
        v_current = vs_chunks[step_idx]
        
        # Prefix (KV from all previous chunks)
        if step_idx > 0:
            k_prefix = k_full[:step_idx * chunk_size]
            v_prefix = v_full[:step_idx * chunk_size]
        else:
            k_prefix = None
            v_prefix = None

        be_out = block_extend_cascade(
            q=q_current,
            k_current=k_current,
            v_current=v_current,
            k_prefix=k_prefix,
            v_prefix=v_prefix,
            dllm_block_size=dllm_block_size,
            sm_scale=sm_scale,
            return_lse=False,
            backend=backend,
        )

        # Construct batch_size=1 batch parameters
        qo_indptr = torch.tensor([0, chunk_size], dtype=torch.int32, device=device)
        kv_curr_indptr = torch.tensor([0, chunk_size], dtype=torch.int32, device=device)
        
        if k_prefix is not None:
            # Convert contiguous prefix to Paged KV Cache format
            paged_kv_cache, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = \
                create_paged_kv_cache_from_prefix(k_prefix, v_prefix, page_size)
            
            sg_out = sglang_style_cascade_attention(
                q=q_current,
                k_current=k_current,
                v_current=v_current,
                qo_indptr=qo_indptr,
                kv_curr_indptr=kv_curr_indptr,
                paged_kv_cache=paged_kv_cache,
                paged_kv_indptr=paged_kv_indptr,
                paged_kv_indices=paged_kv_indices,
                paged_kv_last_page_len=paged_kv_last_page_len,
                page_size=page_size,
                workspace_buffer=workspace_buffer,
                sm_scale=sm_scale,
                return_lse=False,
                backend=backend,
            )
        else:
            # No prefix (first chunk)
            sg_out = sglang_style_cascade_attention(
                q=q_current,
                k_current=k_current,
                v_current=v_current,
                qo_indptr=qo_indptr,
                kv_curr_indptr=kv_curr_indptr,
                paged_kv_cache=None,
                paged_kv_indptr=None,
                paged_kv_indices=None,
                paged_kv_last_page_len=None,
                page_size=page_size,
                workspace_buffer=workspace_buffer,
                sm_scale=sm_scale,
                return_lse=False,
                backend=backend,
            )
        
        # Compute difference
        step_diff = (be_out - sg_out).abs().max().item()
        step_diffs.append(step_diff)
        max_diff_all_steps = max(max_diff_all_steps, step_diff)
        
        prefix_len = step_idx * chunk_size
        if verbose:
            print(f"  Step {step_idx}: prefix_len={prefix_len:4d}, curr_len={chunk_size}, max_diff={step_diff:.6f}")
    
    precision_ok = max_diff_all_steps < tol
    status = "PASS" if precision_ok else "FAIL"
    print(f"\n  [Precision Summary] max_diff (all steps) = {max_diff_all_steps:.6f} [{status}]")
    print(f"\n{'='*100}")
    print(f"Performance Comparison (measuring last Step: step={num_steps-1}, prefix_len={(num_steps-1)*chunk_size})")
    print(f"{'='*100}")
    
    # Use last step for performance testing (longest prefix)
    test_step = num_steps - 1
    q_current = qs_chunks[test_step]
    k_current = ks_chunks[test_step]
    v_current = vs_chunks[test_step]
    k_prefix = k_full[:test_step * chunk_size]
    v_prefix = v_full[:test_step * chunk_size]
    prefix_len = test_step * chunk_size
    
    # Prepare Paged KV Cache (create only once)
    paged_kv_cache, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = \
        create_paged_kv_cache_from_prefix(k_prefix, v_prefix, page_size)
    qo_indptr = torch.tensor([0, chunk_size], dtype=torch.int32, device=device)
    kv_curr_indptr = torch.tensor([0, chunk_size], dtype=torch.int32, device=device)

    def run_be_cascade():
        return block_extend_cascade(
            q=q_current,
            k_current=k_current,
            v_current=v_current,
            k_prefix=k_prefix,
            v_prefix=v_prefix,
            dllm_block_size=dllm_block_size,
            sm_scale=sm_scale,
            return_lse=False,
            backend=backend,
        )
    
    # Warmup
    for _ in range(warmup_iters):
        run_be_cascade()
    torch.cuda.synchronize()
    
    # Benchmark
    start = time_module.perf_counter()
    for _ in range(bench_iters):
        run_be_cascade()
    torch.cuda.synchronize()
    be_time = (time_module.perf_counter() - start) / bench_iters * 1000

    def run_sg_cascade():
        return sglang_style_cascade_attention(
            q=q_current,
            k_current=k_current,
            v_current=v_current,
            qo_indptr=qo_indptr,
            kv_curr_indptr=kv_curr_indptr,
            paged_kv_cache=paged_kv_cache,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            page_size=page_size,
            workspace_buffer=workspace_buffer,
            sm_scale=sm_scale,
            return_lse=False,
            backend=backend,
        )
    
    # Warmup
    for _ in range(warmup_iters):
        run_sg_cascade()
    torch.cuda.synchronize()
    
    # Benchmark
    start = time_module.perf_counter()
    for _ in range(bench_iters):
        run_sg_cascade()
    torch.cuda.synchronize()
    sg_time = (time_module.perf_counter() - start) / bench_iters * 1000

    print(f"\n  Test parameters: chunk_size={chunk_size}, prefix_len={prefix_len}")
    print(f"  block_extend_cascade:        {be_time:.3f} ms")
    print(f"  sglang_style_cascade_attention: {sg_time:.3f} ms")
    
    if be_time < sg_time:
        speedup = sg_time / be_time
        print(f"  block_extend_cascade faster by {speedup:.2f}x")
    else:
        speedup = be_time / sg_time
        print(f"  sglang_style_cascade_attention faster by {speedup:.2f}x")
    

    return {
        "precision_ok": precision_ok,
        "max_diff": max_diff_all_steps,
        "step_diffs": step_diffs,
        "be_time_ms": be_time,
        "sg_time_ms": sg_time,
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cascade vs Batch Comparison Test")
    parser.add_argument("--batch-prefill", action="store_true", help="Multi Request Batch Prefill comparison (variable chunk_size)")
    parser.add_argument("--step-by-step", action="store_true", help="Real step-by-step execution comparison (simulating pipeline dependencies)")
    parser.add_argument("--step-by-step-cuda_graph", action="store_true", help="Real step-by-step execution comparison + CUDA Graph")
    parser.add_argument("--single-req-cuda_graph", action="store_true", help="Single request step-by-step execution + CUDA Graph (pipeline dependencies)")
    parser.add_argument("--fa2-fa3-be", action="store_true", help="FA2 vs FA3 BlockExtend vs Causal performance comparison")
    parser.add_argument("--cascade-perf", action="store_true", help="Cascade interface performance comparison (batch_block_extend_cascade vs sglang_style)")
    parser.add_argument("--cascade-perf-cuda_graph", action="store_true", help="Cascade interface performance comparison + CUDA Graph (pre-create Wrapper, only capture run)")
    parser.add_argument("--cascade-precision", action="store_true", help="Cascade interface precision alignment test (sglang_style vs block_extend)")
    parser.add_argument("--sglang-vs-be", action="store_true", help="sglang_style_cascade vs block_extend_cascade precision and performance comparison (equal inputs)")
    parser.add_argument("--heterogeneous-prefix", action="store_true", help="Heterogeneous prefix test: different requests have different prefix lengths")
    parser.add_argument("--cascade-chunk", action="store_true", help="Cascade Current Chunk test: K/V only has current block, requires kv_offset")
    parser.add_argument("--tvm-ffi-slice-bug", action="store_true", help="TVM FFI slice tensor bug reproduction test")
    parser.add_argument("--cuda_graph-reuse-bug", action="store_true", help="Test CUDA Graph mode wrapper reuse bug (exposes q_offsets address change issue)")
    parser.add_argument("--precision-test", action="store_true", help="DLLM component precision test (compare with native Custom Mask FA2)")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "all"], 
                        help="Data type for precision test: fp16, bf16, or all (test both)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed Step flow preview information")
    parser.add_argument("--num_chunks", type=int, default=8)
    parser.add_argument("--chunk_len", type=int, default=32, help="Chunk length (prefill seq_len)")
    parser.add_argument("--dllm_block_size", type=int, default=None, help="DLLM block size (default = chunk_len)")
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--total_tokens", type=int, default=256, help="Total tokens (for fair/width mode)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (for --multi mode)")
    parser.add_argument("--num_requests", type=int, default=4, help="Number of requests")
    parser.add_argument("--tokens_per_request", type=int, default=256, help="Tokens per request")
    parser.add_argument("--kv_len", type=int, default=2048, help="Total tokens (for --fa2-fa3-be mode, i.e., tokens_per_request)")
    parser.add_argument("--chunk_sizes", type=str, default="32,64,128,256,512", help="Chunk sizes list, comma separated")
    parser.add_argument("--backend", type=str, default="fa2", choices=["auto", "fa2", "fa3"], help="Backend implementation: auto/fa2/fa3")
    args = parser.parse_args()
    
    dllm_bs = args.dllm_block_size if args.dllm_block_size is not None else args.chunk_len
    
    if args.fa2_fa3_be:
        # FA2 vs FA3 BlockExtend vs Causal performance comparison (incremental Prefill scenario)
        chunk_sizes = [int(x) for x in args.chunk_sizes.split(",")]
        test_fa2_fa3_block_extending_vs_causal(
            num_requests=args.num_requests,
            chunk_sizes=chunk_sizes,
            tokens_per_request=args.kv_len,  # Use kv_len parameter as tokens_per_request
            dllm_block_size=dllm_bs,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            verbose=args.verbose,
        )
    elif args.step_by_step_cuda_graph:
        # Multi-request real step-by-step execution comparison + CUDA Graph
        test_incremental_batchprefill_step_by_step_with_cuda_graph(
            num_requests=args.num_requests,
            tokens_per_request=args.tokens_per_request,
            dllm_block_size=dllm_bs,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            verbose=args.verbose,
        )
    elif args.single_req_cuda_graph:
        # Single request step-by-step execution + CUDA Graph (pipeline dependencies)
        test_incremental_singlereq_prefill_step_by_step_with_cuda_graph(
            tokens_per_request=args.tokens_per_request,
            dllm_block_size=dllm_bs,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            verbose=args.verbose,
        )
    elif args.precision_test:
        if args.dtype == "all":
            print("\n" + "="*120)
            print("Running FP16 precision test...")
            print("="*120)
            test_dllm_precision_vs_custom_mask_fa2(
                verbose=args.verbose,
                test_dtypes=[torch.float16],
            )
            print("\n" + "="*120)
            print("Running BF16 precision test...")
            print("="*120)
            test_dllm_precision_vs_custom_mask_fa2(
                verbose=args.verbose,
                test_dtypes=[torch.bfloat16],
            )
        else:
            if args.dtype == "fp16":
                test_dtypes = [torch.float16]
            else:  # bf16
                test_dtypes = [torch.bfloat16]
            test_dllm_precision_vs_custom_mask_fa2(
                verbose=args.verbose,
                test_dtypes=test_dtypes,
            )
    elif args.cascade_perf:
        # Multi-request end-to-end performance test, baseline is sglang_style_cascade_attention
        chunk_sizes = [int(x) for x in args.chunk_sizes.split(",")]
        test_cascade_interfaces_perf(
            num_requests=args.num_requests,
            tokens_per_request=args.tokens_per_request,
            dllm_block_size=dllm_bs,
            chunk_sizes=chunk_sizes,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            verbose=args.verbose,
            backend=args.backend,
        )
    elif args.cascade_perf_cuda_graph:
        # Multi-request CUDA Graph end-to-end performance test, baseline is sglang_style_cascade_attention CUDA Graph
        chunk_sizes = [int(x) for x in args.chunk_sizes.split(",")]
        test_cascade_interfaces_perf_with_cuda_graph(
            num_requests=args.num_requests,
            tokens_per_request=args.tokens_per_request,
            dllm_block_size=dllm_bs,
            chunk_sizes=chunk_sizes,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            verbose=args.verbose,
            backend=args.backend,
        )
    elif args.cascade_precision:
        # Pure precision test, verify numerical consistency between block_extend_cascade and sglang_style_cascade_attention
        test_cascade_precision_alignment(
            verbose=args.verbose,
        )
    elif args.sglang_vs_be:
        # Single request cascade performance test, baseline is sglang_style_cascade_attention
        num_steps = args.tokens_per_request // dllm_bs
        test_sglang_vs_block_extend_cascade(
            num_steps=num_steps,
            dllm_block_size=dllm_bs,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            verbose=args.verbose,
            backend=args.backend,
        )
    elif args.heterogeneous_prefix:
        # Heterogeneous prefix test: different requests have different prefix lengths
        for be in ["fa2", "fa3"]:
            test_heterogeneous_prefix_batch(
                verbose=args.verbose,
                backend=be,
            )
        
    elif args.cascade_chunk:
        # Cascade Current Chunk test: K/V only has current block, requires kv_offset
        for be in ["fa2", "fa3"]:
            test_cascade_current_chunk_batch(
                verbose=args.verbose,
                backend=be,
            )