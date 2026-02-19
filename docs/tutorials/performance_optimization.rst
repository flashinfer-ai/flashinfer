.. _performance-optimization:

Performance Profiling & Optimization
====================================

This guide provides comprehensive strategies for profiling and optimizing FlashInfer kernels in your LLM inference pipeline. Whether you're deploying on Ampere, Hopper, or Blackwell GPUs, this tutorial will help you maximize performance.

.. contents:: Table of Contents
   :local:
   :depth: 3

Quick Start
-----------

Before diving into profiling, ensure you have the necessary tools installed:

.. code-block:: bash

   # Install CUPTI for performance profiling
   pip install -U cupti-python

   # Enable verbose logging to understand kernel selection
   export FLASHINFER_LOGLEVEL=3
   export FLASHINFER_LOGDEST=flashinfer_debug.log

   # Run a benchmark with profiling enabled
   python benchmarks/flashinfer_benchmark.py --routine batch_decode \
       --batch_size 128 --kv_len 2048 --num_qo_heads 32 --num_kv_heads 8 \
       --head_dim 128 --use_cupti -vv

Understanding FlashInfer Performance
-----------------------------------

FlashInfer automatically selects the best backend for your workload:

Backend Selection
~~~~~~~~~~~~~~~~~

FlashInfer supports multiple backends, each optimized for different scenarios:

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Backend
     - Best For
     - Supported GPUs
     - Key Features
   * - **FlashAttention-2/3**
     - General attention
     - SM75+
     - Memory efficient, fast
   * - **cuDNN**
     - Large batch sizes
     - SM80+
     - Highly optimized, stable
   * - **CUTLASS**
     - Custom GEMM patterns
     - SM75+
     - Flexible, customizable
   * - **TensorRT-LLM**
     - Production deployment
     - SM80+
     - Fused operations, low latency
   * - **CuTe-DSL**
     - Blackwell-specific
     - SM100+
     - Cutting-edge features

The backend is selected based on:

1. **GPU Architecture**: Compute capability determines available backends
2. **Workload Shape**: Batch size, sequence length, head dimensions
3. **Data Type**: FP16, BF16, FP8, FP4 support varies
4. **Feature Requirements**: Paged vs ragged KV-cache, MLA, sparse attention

Profiling Workflow
------------------

Step 1: Baseline Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start by establishing baseline metrics:

.. code-block:: python

   import torch
   import flashinfer
   import time

   # Setup
   batch_size = 128
   qo_len = 1  # Decode: 1 token per request
   kv_len = 2048
   num_qo_heads = 32
   num_kv_heads = 8  # GQA with 4:1 ratio
   head_dim = 128
   device = torch.device("cuda:0")

   # Create test data
   q = torch.randn(batch_size, num_qo_heads, head_dim, 
                   device=device, dtype=torch.float16)
   k = torch.randn(batch_size, kv_len, num_kv_heads, head_dim,
                   device=device, dtype=torch.float16)
   v = torch.randn(batch_size, kv_len, num_kv_heads, head_dim,
                   device=device, dtype=torch.float16)

   # Warmup
   for _ in range(10):
       _ = flashinfer.single_decode_with_kv_cache(q[0], k[0], v[0])
   
   torch.cuda.synchronize()

   # Benchmark
   iterations = 100
   start = time.perf_counter()
   for _ in range(iterations):
       output = flashinfer.single_decode_with_kv_cache(q[0], k[0], v[0])
   torch.cuda.synchronize()
   end = time.perf_counter()

   avg_latency_ms = (end - start) * 1000 / iterations
   print(f"Average latency: {avg_latency_ms:.3f} ms")

Step 2: Detailed Profiling with CUPTI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use CUPTI to get detailed kernel-level metrics:

.. code-block:: bash

   # Profile a specific test case
   python benchmarks/flashinfer_benchmark.py \
       --routine batch_decode \
       --batch_size 128 \
       --kv_len 2048 \
       --num_qo_heads 32 \
       --num_kv_heads 8 \
       --head_dim 128 \
       --use_cupti \
       --generate_repro_command \
       -vv

Key metrics to monitor:

- **Latency**: End-to-end execution time
- **Throughput**: Operations per second (TFLOPS, GB/s)
- **SM Efficiency**: GPU utilization percentage
- **Memory Bandwidth**: Data transfer rates
- **Kernel Launch Overhead**: Time between kernel calls

Step 3: NSight Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

For deep kernel analysis, use NVIDIA Nsight Compute:

.. code-block:: bash

   # Profile with Nsight Compute
   ncu --set full --target-processes all \
       python benchmarks/flashinfer_benchmark.py \
           --routine batch_decode \
           --batch_size 128 \
           --kv_len 2048 \
           --num_qo_heads 32 \
           --num_kv_heads 8 \
           --head_dim 128 \
           --num_iters 1

   # Profile specific kernel
   ncu --kernel-regex ".*flashinfer.*" \
       --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
       python your_script.py

Nsight metrics to examine:

- ``sm__throughput.avg.pct_of_peak_sustained_elapsed``: SM utilization
- ``dram__throughput.avg.pct_of_peak_sustained_elapsed``: Memory bandwidth
- ``l1tex__throughput.avg.pct_of_peak_sustained_elapsed``: L1 cache usage
- ``smsp__sass_thread_inst_executed_op_*.sum``: Instruction breakdown

Architecture-Specific Optimization
----------------------------------

Ampere (SM80/SM86) - A100, A10, RTX 30 Series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Strengths:**
- Strong FP16/BF16 performance with tensor cores
- Large L2 cache (40MB on A100)
- Good memory bandwidth (1.5-2 TB/s on A100)

**Optimization Tips:**

.. code-block:: python

   # Use BF16 for better numerical stability
   q = q.to(torch.bfloat16)
   k = k.to(torch.bfloat16)
   v = v.to(torch.bfloat16)

   # For decode: batch multiple requests together
   from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
   
   wrapper = BatchDecodeWithPagedKVCacheWrapper(
       backend="flashinfer",  # FlashAttention-2 optimized for Ampere
       enable_cuda_graph=True  # Reduce kernel launch overhead
   )

**Avoid:**
- Small batch sizes (< 16) - underutilize SMs
- FP8 operations - not supported on Ampere
- Excessive kernel launches - high overhead

Hopper (SM90) - H100, H200
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Strengths:**
- FP8 tensor cores with transformer engine
- Thread block clusters for better L2 reuse
- Asynchronous memory operations
- TMA (Tensor Memory Accelerator)

**Optimization Tips:**

.. code-block:: python

   # Leverage FP8 for 2x throughput
   from flashinfer import quantization
   
   # Quantize KV cache to FP8
   k_fp8 = quantization.quantize_fp8(k)
   v_fp8 = quantization.quantize_fp8(v)
   
   # Use Hopper-specific attention
   output = flashinfer.single_decode_with_kv_cache(
       q, k_fp8, v_fp8,
       sm_scale=1.0 / (head_dim ** 0.5),
       kv_layout="HND"  # Better for FP8 on Hopper
   )

   # Enable CuDNN backend for large batches
   wrapper = BatchDecodeWithPagedKVCacheWrapper(
       backend="cudnn"  # Optimized for H100
   )

**Best Practices:**
- Use FP8 for memory-bound operations (long context)
- Enable thread block clusters for prefill
- Batch size >= 128 for full SM utilization
- Use paged attention to maximize memory efficiency

Blackwell (SM100/SM120) - B200, RTX 50 Series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Strengths:**
- NVFP4 and MXFP4/8 support
- Enhanced tensor cores
- Improved memory hierarchy
- Deep learning accelerators

**Optimization Tips:**

.. code-block:: python

   # Use FP4 quantization for maximum memory savings
   from flashinfer.fp4_quantization import nvfp4_quantize
   
   # Quantize weights to FP4 (Blackwell only)
   k_fp4, k_scale = nvfp4_quantize(k, scale_layout="per_channel")
   v_fp4, v_scale = nvfp4_quantize(v, scale_layout="per_channel")
   
   # Use CuTe-DSL kernels for Blackwell-specific features
   from flashinfer.cute_dsl import rmsnorm_fp4quant
   
   # Fused normalization + quantization
   output_fp4 = rmsnorm_fp4quant(
       input_tensor,
       weight,
       eps=1e-6
   )

**Blackwell-Specific Features:**
- ``nvfp4_quantize`` for 4-bit KV cache
- ``mxfp8_quantize`` for MX format tensors
- CuTe-DSL fused operations (``add_rmsnorm_fp4quant``)
- Enhanced grouped GEMM for MoE

Common Performance Pitfalls
---------------------------

Memory-Related Issues
~~~~~~~~~~~~~~~~~~~~~

**Problem: Low Memory Bandwidth Utilization**

Symptoms:
- Kernel time >> theoretical minimum
- ``dram__throughput`` < 70% of peak

Solutions:

.. code-block:: python

   # 1. Use proper data layout for your architecture
   wrapper = BatchDecodeWithPagedKVCacheWrapper(
       kv_layout="HND"  # Use HND for FP8/FP4, NHD for FP16
   )

   # 2. Minimize memory allocations in hot path
   # Pre-allocate output buffers
   output = torch.empty(batch_size, num_qo_heads, head_dim,
                        device=device, dtype=dtype)
   flashinfer.single_decode_with_kv_cache(
       q, k, v, out=output  # Reuse buffer
   )

   # 3. Enable paged attention to reduce memory fragmentation
   from flashinfer.page import allocate_kv_cache_with_page_table
   
   kv_cache, kv_indptr, kv_indices, kv_last_page_len = \
       allocate_kv_cache_with_page_table(
           num_layers=32,
           num_pages=10000,
           page_size=16,  # Larger page size reduces overhead
           num_kv_heads=num_kv_heads,
           head_dim=head_dim,
           dtype=torch.float16,
           device=device
       )

**Problem: Excessive Host-Device Transfers**

Symptoms:
- High latency with small tensors
- CPU profiling shows memcpy overhead

Solutions:

.. code-block:: python

   # Keep data on GPU throughout pipeline
   # Bad: repeated CPU<->GPU transfers
   for token_id in token_ids.cpu().numpy():  # DON'T DO THIS
       process_token(token_id)
   
   # Good: batch operations on GPU
   processed = process_tokens_batch(token_ids)  # Keep on GPU

   # Use pinned memory for necessary transfers
   kv_indptr = torch.tensor([0, 512, 1024], 
                            dtype=torch.int32,
                            device='cuda',
                            pin_memory=True)

Compute-Related Issues
~~~~~~~~~~~~~~~~~~~~~~

**Problem: Low SM Utilization**

Symptoms:
- ``sm__throughput`` < 60% of peak
- GPU not fully utilized

Solutions:

.. code-block:: python

   # 1. Increase batch size
   # Bad: batch_size = 1 (0.5% GPU utilization)
   # Good: batch_size >= 64 (80%+ GPU utilization)
   
   # 2. Use split-K for long sequences
   output = flashinfer.single_decode_with_kv_cache(
       q, k, v,
       enable_split_k=True  # Parallelize over sequence length
   )
   
   # 3. Batch multiple operations together
   # Use BatchDecodeWrapper instead of multiple single_decode calls
   wrapper = BatchDecodeWithPagedKVCacheWrapper(...)
   wrapper.begin_forward(...)
   outputs = wrapper.forward(q_batch)  # Process entire batch

**Problem: Kernel Launch Overhead**

Symptoms:
- Total time >> kernel execution time
- Many small kernel launches

Solutions:

.. code-block:: python

   # 1. Enable CUDA graphs to eliminate launch overhead
   wrapper = BatchDecodeWithPagedKVCacheWrapper(
       enable_cuda_graph=True
   )
   
   # 2. Use torch.compile (PyTorch 2.0+)
   compiled_forward = torch.compile(
       wrapper.forward,
       mode="reduce-overhead",
       fullgraph=True
   )
   output = compiled_forward(q)
   
   # 3. Fuse operations where possible
   from flashinfer.norm import fused_add_rmsnorm
   
   # Fused residual + norm instead of separate ops
   output = fused_add_rmsnorm(
       residual, input_tensor, weight, eps=1e-6
   )

Numerical Stability Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem: NaN or Inf in Attention Output**

Symptoms:
- NaN/Inf values in output
- Loss divergence during training
- Incorrect generation results

Debugging:

.. code-block:: python

   # Enable logging to detect NaN/Inf
   import os
   os.environ['FLASHINFER_LOGLEVEL'] = '5'  # Log statistics
   
   # Check inputs
   assert not torch.isnan(q).any(), "NaN in query"
   assert not torch.isnan(k).any(), "NaN in key"
   assert not torch.isnan(v).any(), "NaN in value"
   assert not torch.isinf(q).any(), "Inf in query"
   
   # Use appropriate softmax scale
   sm_scale = 1.0 / (head_dim ** 0.5)  # Standard scaling
   output = flashinfer.single_decode_with_kv_cache(
       q, k, v, sm_scale=sm_scale
   )

Solutions:

.. code-block:: python

   # 1. Use BF16 for better dynamic range
   q = q.to(torch.bfloat16)
   
   # 2. Clip extreme values before attention
   q = torch.clamp(q, min=-10.0, max=10.0)
   
   # 3. Use window attention for very long sequences
   output = flashinfer.single_decode_with_kv_cache(
       q, k, v,
       window_left=4096,  # Limit attention window
   )
   
   # 4. Check for denormals in FP8
   # Ensure proper scaling when using FP8
   from flashinfer.quantization import segment_max
   
   k_max = segment_max(k.abs())
   k_scale = k_max / 448.0  # FP8 E4M3 max value
   k_fp8 = (k / k_scale).to(torch.float8_e4m3fn)

Advanced Optimization Techniques
--------------------------------

Multi-GPU Optimization
~~~~~~~~~~~~~~~~~~~~~~

For multi-GPU inference, FlashInfer provides optimized communication primitives:

.. code-block:: python

   from flashinfer.comm import unified_allreduce
   
   # Efficient tensor-parallel all-reduce
   # Overlaps computation with communication
   output_reduced = unified_allreduce(
       output_local,
       group=tp_group,
       async_op=True  # Non-blocking
   )
   
   # Continue computation while communicating
   hidden_states = mlp_forward(normalized_input)
   
   # Wait for communication to complete
   output_reduced.wait()

MoE Optimization
~~~~~~~~~~~~~~~~

Mixture of Experts requires special handling:

.. code-block:: python

   from flashinfer.fused_moe import cutlass_fused_moe
   
   # Use fused MoE for better performance
   output = cutlass_fused_moe(
       hidden_states,
       gate_logits,
       expert_weights,
       expert_biases,
       top_k=2,
       num_experts=64,
       normalize_routing_weights=True,
       use_fp8=True  # 2x faster on Hopper+
   )
   
   # For expert parallelism across GPUs
   from flashinfer.fused_moe import moe_a2a_dispatch_combine
   
   output = moe_a2a_dispatch_combine(
       hidden_states,
       expert_weights,
       tp_group,
       ep_group,
       use_fp8_quantize=True
   )

Long Context Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

For sequences > 8K tokens:

.. code-block:: python

   # Use cascade attention for shared prefix
   from flashinfer.cascade import MultiLevelCascadeAttentionWrapper
   
   wrapper = MultiLevelCascadeAttentionWrapper()
   
   # Level 0: Shared prefix (e.g., system prompt)
   # Level 1: Request-specific context
   # Level 2: Recent tokens
   wrapper.append_pref ix_kv_cache(
       shared_prefix_k,
       shared_prefix_v,
       level=0
   )
   
   # Much faster than computing attention over full sequence
   output = wrapper.forward(q)
   
   # Or use block-sparse attention
   from flashinfer.sparse import BlockSparseAttentionWrapper
   
   # Define sparsity pattern
   block_sparse_wrapper = BlockSparseAttentionWrapper(
       block_size=64,
       local_blocks=8,   # Attend to 8 nearest blocks
       global_blocks=2,  # Plus 2 global blocks
   )

Memory-Efficient KV Cache Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from flashinfer.page import allocate_kv_cache_with_page_table
   
   # Paged attention reduces memory waste
   kv_cache, kv_indptr, kv_indices, kv_last_page_len = \
       allocate_kv_cache_with_page_table(
           num_layers=40,
           num_pages=20000,
           page_size=16,  # Page size affects both memory and speed
           num_kv_heads=8,
           head_dim=128,
           dtype=torch.float16,
           device=device
       )
   
   # Use smaller dtype for KV cache on Hopper+
   if torch.cuda.get_device_capability()[0] >= 9:  # Hopper
       # FP8 KV cache: 2x memory reduction, minimal accuracy loss
       kv_cache = kv_cache.to(torch.float8_e4m3fn)

Performance Monitoring in Production
------------------------------------

Continuous Performance Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set up monitoring for production deployments:

.. code-block:: python

   import time
   from collections import deque
   from typing import Dict, List
   
   class FlashInferMonitor:
       """Monitor FlashInfer performance in production."""
       
       def __init__(self, window_size: int = 1000):
           self.window_size = window_size
           self.latencies = deque(maxlen=window_size)
           self.throughputs = deque(maxlen=window_size)
           
       def log_request(self, 
                      latency_ms: float,
                      num_tokens: int,
                      batch_size: int):
           """Log a single request's performance."""
           self.latencies.append(latency_ms)
           throughput = (num_tokens * batch_size) / (latency_ms / 1000)
           self.throughputs.append(throughput)
           
       def get_stats(self) -> Dict[str, float]:
           """Get current performance statistics."""
           import numpy as np
           return {
               "p50_latency_ms": np.percentile(self.latencies, 50),
               "p95_latency_ms": np.percentile(self.latencies, 95),
               "p99_latency_ms": np.percentile(self.latencies, 99),
               "mean_throughput": np.mean(self.throughputs),
           }
   
   # Usage in serving loop
   monitor = FlashInferMonitor()
   
   for batch in request_batches:
       start = time.perf_counter()
       outputs = model.generate(batch)
       latency_ms = (time.perf_counter() - start) * 1000
       
       monitor.log_request(
           latency_ms=latency_ms,
           num_tokens=batch.total_tokens,
           batch_size=len(batch)
       )
       
       # Check for performance degradation
       stats = monitor.get_stats()
       if stats["p95_latency_ms"] > latency_slo * 1.2:
           logger.warning(f"Latency SLO violation: {stats}")

Performance Regression Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detect regressions in CI/CD:

.. code-block:: python

   # See tests/performance/ for full implementation
   from flashinfer.testing.performance import BenchmarkRunner
   
   runner = BenchmarkRunner()
   
   # Run standard benchmark suite
   results = runner.run_standard_suite(
       gpu_name="H100",
       precision="fp16"
   )
   
   # Compare against baseline
   baseline = runner.load_baseline("H100_fp16_baseline.json")
   regressions = runner.detect_regressions(results, baseline)
   
   if regressions:
       print(f"⚠️  Performance regressions detected:")
       for reg in regressions:
           print(f"  {reg['test']}: {reg['delta']:.1%} slower")

Troubleshooting Guide
---------------------

Issue: "Module not found" or Import Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clear JIT cache and reinstall
   flashinfer clear-cache
   pip uninstall flashinfer-python flashinfer-cubin flashinfer-jit-cache -y
   pip install flashinfer-python flashinfer-cubin
   
   # Verify installation
   flashinfer show-config

Issue: Slower Than Expected Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 1. Check if correct backend is selected
   import os
   os.environ['FLASHINFER_LOGLEVEL'] = '3'
   
   # Run your code and check logs for backend selection
   
   # 2. Verify GPU is not throttling
   import subprocess
   result = subprocess.run(
       ['nvidia-smi', '--query-gpu=clocks.gr,clocks.mem', 
        '--format=csv,noheader'],
       capture_output=True, text=True
   )
   print(f"GPU clocks: {result.stdout}")
   
   # 3. Check for CPU bottlenecks
   # Profile with py-spy or cProfile
   
   # 4. Ensure CUDA graphs are working
   wrapper = BatchDecodeWithPagedKVCacheWrapper(
       enable_cuda_graph=True
   )
   # Should see "Using CUDA graph" in logs

Issue: OOM (Out of Memory) Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 1. Use paged attention
   from flashinfer.page import allocate_kv_cache_with_page_table
   
   # 2. Reduce batch size or sequence length
   max_batch_size = get_optimal_batch_size(
       gpu_memory_gb=80,  # H100
       sequence_length=2048,
       num_layers=40
   )
   
   # 3. Use FP8/FP4 quantization
   k_fp8 = quantize_fp8(k)  # 2x memory reduction
   
   # 4. Enable memory pooling
   torch.cuda.empty_cache()
   torch.cuda.set_per_process_memory_fraction(0.95)

Best Practices Summary
---------------------

✅ **Do:**

- Use appropriate data types: FP16/BF16 on Ampere, FP8 on Hopper+, FP4 on Blackwell
- Enable CUDA graphs for stable workloads (``enable_cuda_graph=True``)
- Batch requests together for better GPU utilization
- Use paged attention for dynamic workloads
- Profile before optimizing - measure first
- Test across multiple GPU architectures if deploying broadly
- Monitor performance in production with continuous logging

❌ **Don't:**

- Use small batch sizes (< 16) unless latency-critical
- Mix CPU and GPU operations in hot path
- Allocate tensors inside inference loop
- Ignore backend selection - let FlashInfer choose
- Skip warmup iterations when benchmarking
- Forget to synchronize CUDA streams when timing
- Assume performance is portable across GPU generations

Performance Checklist
--------------------

Before deploying to production:

.. code-block:: text

   ☐ Profiled with CUPTI to identify bottlenecks
   ☐ Verified SM utilization > 70% for primary kernels
   ☐ Enabled CUDA graphs where applicable
   ☐ Tested with representative batch sizes and sequence lengths
   ☐ Validated numerical accuracy (check for NaN/Inf)
   ☐ Compared backends (FlashAttention, cuDNN, TensorRT-LLM)
   ☐ Optimized for target GPU architecture
   ☐ Set up performance monitoring and alerting
   ☐ Documented performance characteristics and SLOs
   ☐ Created runbooks for common performance issues

Additional Resources
-------------------

- `FlashInfer Benchmarking Framework <https://github.com/flashinfer-ai/flashinfer/blob/main/benchmarks/README.md>`_
- `NVIDIA Nsight Compute Documentation <https://docs.nvidia.com/nsight-compute/>`_
- `CUDA C++ Best Practices Guide <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>`_
- `FlashInfer Discussion Forum <https://github.com/orgs/flashinfer-ai/discussions>`_
- `FlashInfer Blog <https://flashinfer.ai/>`_ - Latest optimization techniques

Contributing Performance Improvements
-------------------------------------

Found an optimization? Share it with the community:

1. Benchmark your improvement using ``flashinfer_benchmark.py``
2. Document the use case and performance gains
3. Submit a pull request with tests
4. Share results in `GitHub Discussions <https://github.com/orgs/flashinfer-ai/discussions>`_

See :ref:`CONTRIBUTING.md <https://github.com/flashinfer-ai/flashinfer/blob/main/CONTRIBUTING.md>`_ for guidelines.
