# Performance Profiling Guide

This guide provides detailed instructions for profiling FlashInfer kernels and analyzing performance bottlenecks.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Using CUPTI for Profiling](#using-cupti-for-profiling)
3. [NSight Compute Integration](#nsight-compute-integration)
4. [NSight Systems Integration](#nsight-systems-integration)
5. [Interpreting Benchmark Results](#interpreting-benchmark-results)
6. [Comparing Backend Performance](#comparing-backend-performance)
7. [Architecture-Specific Profiling](#architecture-specific-profiling)
8. [Common Profiling Scenarios](#common-profiling-scenarios)
9. [Performance Debugging Workflow](#performance-debugging-workflow)

## Quick Start

### Basic Profiling with CUPTI

The fastest way to profile FlashInfer is using the built-in CUPTI support:

```bash
# Install CUPTI
pip install -U cupti-python

# Run a benchmark with profiling
python benchmarks/flashinfer_benchmark.py \
    --routine batch_decode \
    --batch_size 128 \
    --kv_len 2048 \
    --num_qo_heads 32 \
    --num_kv_heads 8 \
    --head_dim 128 \
    --use_cupti \
    --verbose 2
```

This will output detailed performance metrics including:
- Median latency and standard deviation
- Achieved TFLOPS
- Memory bandwidth utilization
- Kernel execution time breakdown

### Generating Reproduction Commands

Always use `--generate_repro_command` to get the exact command to reproduce results:

```bash
python benchmarks/flashinfer_benchmark.py \
    --routine batch_decode \
    --batch_size 128 \
    --kv_len 2048 \
    --num_qo_heads 32 \
    --num_kv_heads 8 \
    --head_dim 128 \
    --backends flashinfer cudnn cutlass \
    --generate_repro_command \
    -vv
```

## Using CUPTI for Profiling

CUPTI (CUDA Profiling Tools Interface) provides lightweight performance monitoring.

### Installation

```bash
pip install -U cupti-python
```

### Basic Usage

```python
# In your Python code
import os
os.environ['FLASHINFER_LOGLEVEL'] = '3'  # Enable detailed logging

from cupti_python import cupti_profiler
import flashinfer

# Profile attention operation
with cupti_profiler.CUPTIProfiler() as profiler:
    output = flashinfer.single_decode_with_kv_cache(q, k, v)

# Get metrics
metrics = profiler.get_metrics()
print(f"Kernel time: {metrics['kernel_time_ms']} ms")
print(f"Memory transfers: {metrics['memory_transfers_gb']} GB")
```

### Interpreting CUPTI Metrics

Key metrics to monitor:

| Metric | Good Range | Meaning |
|--------|------------|---------|
| **SM Efficiency** | > 70% | GPU compute units utilization |
| **Memory Bandwidth** | > 60% | DRAM bandwidth utilization |
| **Occupancy** | > 50% | Thread block utilization |
| **IPC** | > 1.0 | Instructions per cycle |

**Example output:**

```
[PERF] flashinfer  :: median time 0.285 ms; std 0.005 ms
                    :: achieved tflops 13.18 TFLOPs/sec
                    :: achieved tb_per_sec 0.026 TB/sec
                    :: sm_efficiency 82.3%
                    :: memory_efficiency 65.1%
```

## NSight Compute Integration

NSight Compute provides the most detailed kernel-level analysis.

### Installation

NSight Compute is included with CUDA Toolkit. Verify installation:

```bash
ncu --version
```

### Profiling a Benchmark

```bash
# Full profiling (slow but comprehensive)
ncu --set full \
    --target-processes all \
    --force-overwrite \
    -o profile_report \
    python benchmarks/flashinfer_benchmark.py \
        --routine batch_decode \
        --batch_size 128 \
        --kv_len 2048 \
        --num_qo_heads 32 \
        --num_kv_heads 8 \
        --head_dim 128 \
        --num_iters 1
```

### Profiling Specific Kernels

```bash
# Profile only FlashInfer kernels
ncu --kernel-regex ".*flashinfer.*" \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --target-processes all \
    python your_script.py
```

### Key NSight Compute Metrics

#### Compute Metrics

```bash
# Profile compute efficiency
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
    smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
    smsp__sass_thread_inst_executed_op_ffma_pred_on.sum \
    python your_script.py
```

**Interpretation:**
- `sm__throughput`: Overall SM utilization (target: > 70%)
- `sm__warps_active`: Warp-level parallelism (target: > 60%)
- `*_op_fadd/fmul/ffma`: FP operation counts

#### Memory Metrics

```bash
# Profile memory efficiency
ncu --metrics \
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum \
    python your_script.py
```

**Interpretation:**
- `dram__throughput`: Memory bandwidth usage (target: > 60%)
- `l1tex__throughput`: L1 cache efficiency
- `*_op_ld/st`: Load/store operation counts

### Analyzing NSight Compute Reports

```bash
# Open in GUI
ncu -i profile_report.ncu-rep

# Generate text report
ncu --import profile_report.ncu-rep --page summary > summary.txt
```

## NSight Systems Integration

NSight Systems profiles the entire application, showing CPU-GPU interactions.

### Basic Timeline Profiling

```bash
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=timeline_report \
    --force-overwrite \
    python benchmarks/flashinfer_benchmark.py \
        --routine batch_decode \
        --batch_size 128 \
        --kv_len 2048 \
        --num_qo_heads 32 \
        --num_kv_heads 8 \
        --head_dim 128
```

### Analyzing the Timeline

```bash
# Open in GUI
nsys-ui timeline_report.nsys-rep

# Generate summary
nsys stats timeline_report.nsys-rep
```

**What to look for:**
- **Kernel launch gaps**: Indicates CPU overhead
- **Small kernels**: May benefit from fusion
- **Host-device transfers**: Should be minimized
- **GPU idle time**: Indicates bottlenecks

## Interpreting Benchmark Results

### Understanding Performance Metrics

#### Latency Metrics

```
[PERF] flashinfer :: median time 0.285 ms; std 0.005 ms
```

- **Median time**: Middle value of all measurements (robust to outliers)
- **Std**: Standard deviation (should be < 5% of median)
- **High std**: Indicates inconsistent performance (check for thermal throttling)

#### Throughput Metrics

```
achieved tflops 13.18 TFLOPs/sec
achieved tb_per_sec 0.026 TB/sec
```

- **TFLOPS**: Floating-point operations per second
  - Compare against GPU theoretical peak
  - H100: ~1000 TF (FP16), ~2000 TF (FP8)
  - A100: ~312 TF (FP16)
- **TB/sec**: Memory bandwidth utilization
  - H100 HBM3: 3.35 TB/s theoretical
  - A100 HBM2e: 1.94 TB/s theoretical
  - Target: > 60% of theoretical peak

### Calculating Roofline Metrics

Determine if your kernel is compute-bound or memory-bound:

```python
def analyze_kernel(tflops, bandwidth_tb_per_sec, peak_tflops, peak_bandwidth):
    """Analyze kernel performance profile."""
    compute_efficiency = tflops / peak_tflops
    memory_efficiency = bandwidth_tb_per_sec / peak_bandwidth
    
    if compute_efficiency > 0.7:
        return "Compute-bound (good!)"
    elif memory_efficiency > 0.7:
        return "Memory-bound (good!)"
    else:
        return "Under-utilized (investigate bottlenecks)"

# Example for H100
tflops = 13.18
bandwidth = 0.026  # TB/s
result = analyze_kernel(tflops, bandwidth, 1000, 3.35)
print(result)  # ~1% compute, ~0.8% memory -> Under-utilized
```

## Comparing Backend Performance

FlashInfer supports multiple backends. Compare them systematically:

```bash
python benchmarks/flashinfer_benchmark.py \
    --routine batch_decode \
    --batch_size 128 \
    --kv_len 2048 \
    --num_qo_heads 32 \
    --num_kv_heads 8 \
    --head_dim 128 \
    --backends flashinfer cudnn cutlass trtllm \
    --refcheck \
    -vv
```

### Example Comparison Output

```
[PERF] flashinfer :: median time 0.285 ms; achieved tflops 13.18 TFLOPs/sec
[PERF] cudnn      :: median time 0.298 ms; achieved tflops 12.60 TFLOPs/sec
[PERF] cutlass    :: median time 0.301 ms; achieved tflops 12.48 TFLOPs/sec
[PERF] trtllm     :: median time 0.279 ms; achieved tflops 13.47 TFLOPs/sec
```

**Analysis:**
- TensorRT-LLM is fastest (2% faster than FlashInfer)
- cuDNN is 4.5% slower
- CUTLASS is 5.6% slower

**Backend Selection Guidelines:**

| Backend | Best For | GPU | Trade-offs |
|---------|----------|-----|------------|
| **flashinfer** | General-purpose, flexible | All | Good balance |
| **cudnn** | Large batch decode | SM80+ | Optimized but less flexible |
| **cutlass** | Custom patterns | All | Highly configurable |
| **trtllm** | Production deployment | SM80+ | Best performance, rigid API |

## Architecture-Specific Profiling

### Ampere (A100, A10, RTX 30 Series)

```bash
# Profile with FP16/BF16
python benchmarks/flashinfer_benchmark.py \
    --routine batch_decode \
    --backends flashinfer cudnn \
    --dtype float16 \
    ...

# Check tensor core usage
ncu --metrics \
    smsp__inst_executed_pipe_tensor.sum,\
    smsp__inst_executed_pipe_fp16.sum \
    --kernel-regex ".*flashinfer.*" \
    python your_script.py
```

**Ampere-specific metrics:**
- Tensor core instructions (`pipe_tensor`)
- FP16 vs FP32 instruction mix
- L2 cache hit rate (40MB L2 on A100)

### Hopper (H100, H200)

```bash
# Profile with FP8
python benchmarks/flashinfer_benchmark.py \
    --routine batch_decode \
    --backends flashinfer cudnn \
    --dtype fp8_e4m3 \
    ...

# Check FP8 tensor core usage
ncu --metrics \
    smsp__inst_executed_pipe_tensor_op_hmma.sum,\
    sm__sass_thread_inst_executed_op_fp8.sum \
    --kernel-regex ".*flashinfer.*" \
    python your_script.py
```

**Hopper-specific metrics:**
- FP8 tensor core ops
- TMA (Tensor Memory Accelerator) efficiency
- Thread block cluster utilization

### Blackwell (B200, RTX 50 Series)

```bash
# Profile with FP4 (Blackwell-only)
python benchmarks/flashinfer_benchmark.py \
    --routine rmsnorm_fp4quant \
    --backends cute_dsl \
    ...

# Check FP4 operations
ncu --metrics \
    smsp__inst_executed_pipe_fp4.sum,\
    sm__sass_thread_inst_executed_op_mxfp.sum \
    --kernel-regex ".*flashinfer.*" \
    python your_script.py
```

**Blackwell-specific metrics:**
- FP4/MXFP8 operations
- Enhanced tensor core utilization
- Deep learning accelerator metrics

## Common Profiling Scenarios

### Scenario 1: Decode Attention Profiling

Profile typical decode (single-token generation) workload:

```bash
python benchmarks/flashinfer_benchmark.py \
    --routine batch_decode \
    --batch_size 32 64 128 256 \
    --kv_len 1024 2048 4096 \
    --num_qo_heads 32 \
    --num_kv_heads 8 \
    --head_dim 128 \
    --backends flashinfer cudnn \
    --use_cupti \
    -vv
```

**Expected results:**
- Latency increases linearly with `kv_len`
- Throughput improves with `batch_size` (up to a point)
- cuDNN typically best for large batches on Hopper

### Scenario 2: Prefill Attention Profiling

Profile prefill (context encoding) workload:

```bash
python benchmarks/flashinfer_benchmark.py \
    --routine batch_prefill \
    --batch_size 1 2 4 8 \
    --qo_len 512 1024 2048 \
    --kv_len 512 1024 2048 \
    --num_qo_heads 32 \
    --num_kv_heads 8 \
    --head_dim 128 \
    --backends flashinfer cudnn \
    --use_cupti \
    -vv
```

**Expected results:**
- Latency scales quadratically with sequence length
- FlashAttention generally best for prefill
- Large `qo_len` benefits from split-K optimization

### Scenario 3: MoE Performance Profiling

Profile Mixture of Experts workload:

```bash
python benchmarks/flashinfer_benchmark.py \
    --routine cutlass_fused_moe \
    --batch_size 512 1024 2048 \
    --hidden_size 4096 \
    --num_experts 64 \
    --top_k 2 \
    --use_fp8 \
    --use_cupti \
    -vv
```

**What to check:**
- Expert load balancing (check log output)
- Memory bandwidth (MoE is often memory-bound)
- Benefit of FP8 quantization (should be ~2x faster)

### Scenario 4: Long Context Profiling

Profile long context scenarios (8K+ tokens):

```bash
python benchmarks/flashinfer_benchmark.py \
    --routine batch_decode \
    --batch_size 16 \
    --kv_len 8192 16384 32768 \
    --enable_split_k \
    --num_qo_heads 32 \
    --num_kv_heads 8 \
    --head_dim 128 \
    --backends flashinfer \
    --use_cupti \
    -vv
```

**Optimization strategies:**
- Enable split-K for better parallelism
- Use FP8 KV cache to reduce memory pressure
- Consider cascade attention for shared prefixes

## Performance Debugging Workflow

### Step 1: Establish Baseline

```bash
# Run with multiple iterations for statistical significance
python benchmarks/flashinfer_benchmark.py \
    --routine batch_decode \
    --batch_size 128 \
    --kv_len 2048 \
    --num_qo_heads 32 \
    --num_kv_heads 8 \
    --head_dim 128 \
    --backends flashinfer \
    --num_iters 100 \
    --use_cupti \
    -vv
```

### Step 2: Identify Bottleneck

```bash
# Profile with NSight Compute
ncu --set roofline \
    --kernel-regex ".*flashinfer.*" \
    -o profile_report \
    python benchmarks/flashinfer_benchmark.py \
        --routine batch_decode \
        --batch_size 128 \
        --kv_len 2048 \
        --num_qo_heads 32 \
        --num_kv_heads 8 \
        --head_dim 128 \
        --num_iters 1
```

Analyze the roofline plot:
- **Above the roofline**: Kernel is optimized
- **Below, left side**: Memory-bound (optimize data access)
- **Below, right side**: Compute-bound (optimize arithmetic)

### Step 3: Test Optimizations

```bash
# Try different backends
python benchmarks/flashinfer_benchmark.py \
    --routine batch_decode \
    --backends flashinfer cudnn cutlass trtllm \
    ...

# Try different data types
python benchmarks/flashinfer_benchmark.py \
    --routine batch_decode \
    --dtype float16 bfloat16 fp8_e4m3 \
    ...

# Try different layouts
 python benchmarks/flashinfer_benchmark.py \
    --routine batch_decode \
    --kv_layout NHD HND \
    ...
```

### Step 4: Verify Improvements

```bash
# Compare with baseline
python benchmarks/flashinfer_benchmark.py \
    --routine batch_decode \
    --batch_size 128 \
    --kv_len 2048 \
    --num_qo_heads 32 \
    --num_kv_heads 8 \
    --head_dim 128 \
    --backends flashinfer_optimized flashinfer_baseline \
    --generate_repro_command \
    --use_cupti \
    -vv
```

Calculate speedup:
```python
speedup = baseline_latency / optimized_latency
print(f"Speedup: {speedup:.2f}x")
```

## Performance Checklist

Before concluding profiling:

- [ ] Ran benchmarks with warm-up iterations (at least 5)
- [ ] Used sufficient iterations for statistical significance (30+)
- [ ] Verified GPU is not thermal throttling (`nvidia-smi dmon`)
- [ ] Tested multiple backends to find the best
- [ ] Profiled with NSight Compute for kernel-level insights
- [ ] Checked for kernel launch overhead (use CUDA graphs if high)
- [ ] Verified numerical correctness (use `--refcheck` flag)
- [ ] Documented configuration and results
- [ ] Generated reproduction command (`--generate_repro_command`)

## Troubleshooting Common Issues

### Issue: Inconsistent Performance (High Standard Deviation)

**Symptoms:**
```
[PERF] flashinfer :: median time 0.285 ms; std 0.045 ms
                                                  ^^^^^ > 10% of median
```

**Possible causes:**
1. GPU thermal throttling
2. CPU frequency scaling
3. Background processes

**Solutions:**
```bash
# Check GPU clocks
nvidia-smi -q -d CLOCK

# Set persistent mode
sudo nvidia-smi -pm 1

# Lock GPU clocks (requires sudo)
sudo nvidia-smi -lgc 1410,1410

# Run with higher priority
nice -n -20 python benchmarks/flashinfer_benchmark.py ...
```

### Issue: Lower Than Expected Performance

**Symptoms:**
- Achieved TFLOPS << GPU theoretical peak
- Bandwidth utilization < 50%

**Debug steps:**

1. Check batch size:
```bash
# Try larger batch sizes
python benchmarks/flashinfer_benchmark.py \
    --batch_size 32 64 128 256 \
    ...
```

2. Profile kernel occupancy:
```bash
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    --kernel-regex ".*flashinfer.*" \
    python your_script.py
```

3. Check for memory access patterns:
```bash
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
    --kernel-regex ".*flashinfer.*" \
    python your_script.py
```

### Issue: OOM During Profiling

**Solution:**
```bash
# Profile with smaller batch size
python benchmarks/flashinfer_benchmark.py \
    --batch_size 16 \
    --num_iters 10 \
    ...

# Or use NSight Compute's sampling mode
ncu --sampling-count 1 \
    --sampling-interval auto \
    python your_script.py
```

## Additional Resources

- [FlashInfer Performance Optimization Tutorial](../docs/tutorials/performance_optimization.rst)
- [NVIDIA NSight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [NVIDIA NSight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [CUDA C++ Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [FlashInfer GitHub Discussions](https://github.com/orgs/flashinfer-ai/discussions)

## Contributing Profiling Results

Found interesting performance characteristics? Share with the community:

1. Run benchmarks on your hardware
2. Document configuration and results
3. Create a discussion post or PR with findings
4. Help improve FlashInfer for everyone!
