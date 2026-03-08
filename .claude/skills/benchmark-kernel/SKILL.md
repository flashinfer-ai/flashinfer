---
name: benchmark-kernel
description: Guide for benchmarking FlashInfer kernels with CUPTI timing
---

# Tutorial: Benchmarking FlashInfer Kernels

This tutorial shows you how to accurately benchmark FlashInfer kernels.

## Goal

Measure the performance of FlashInfer kernels:
- Get accurate GPU kernel execution time
- Compare multiple backends (FlashAttention2/3, cuDNN, CUTLASS, TensorRT-LLM)
- Generate reproducible benchmark results
- Save results to CSV for analysis

## Timing Methods

FlashInfer supports two timing methods:

1. **CUPTI (Preferred)**: Hardware-level profiling for most accurate GPU kernel time
   - Measures pure GPU compute time without host-device overhead
   - Requires `cupti-python >= 13.0.0` (CUDA 13+)

2. **CUDA Events (Fallback)**: Standard CUDA event timing
   - Automatically used if CUPTI is not available
   - Good accuracy, slight overhead from host synchronization

**The framework automatically uses CUPTI if available, otherwise falls back to CUDA events.**

## Installation

### Install CUPTI (Recommended)

For the most accurate benchmarking:

```bash
pip install -U cupti-python
```

**Requirements**: CUDA 13+ (CUPTI version 13+)

### Without CUPTI

If you don't install CUPTI, the framework will:
- Print a warning: `CUPTI is not installed. Falling back to CUDA events.`
- Automatically use CUDA events for timing
- Still provide good benchmark results

## Method 1: Using flashinfer_benchmark.py (Recommended)

### Step 1: Choose Your Test Routine

Available routines:
- **Attention**: `BatchDecodeWithPagedKVCacheWrapper`, `BatchPrefillWithPagedKVCacheWrapper`, `BatchPrefillWithRaggedKVCacheWrapper`, `BatchMLAPagedAttentionWrapper`
- **GEMM**: `bmm_fp8`, `gemm_fp8_nt_groupwise`, `group_gemm_fp8_nt_groupwise`, `mm_fp4`
- **MOE**: `trtllm_fp4_block_scale_moe`, `trtllm_fp8_block_scale_moe`, `trtllm_fp8_per_tensor_scale_moe`, `cutlass_fused_moe`

### Step 2: Run a Single Benchmark

Example - Benchmark decode attention:

```bash
# CUPTI will be used automatically if installed
python benchmarks/flashinfer_benchmark.py \
    --routine BatchDecodeWithPagedKVCacheWrapper \
    --backends fa2 fa2_tc cudnn \
    --page_size 16 \
    --batch_size 32 \
    --s_qo 1 \
    --s_kv 2048 \
    --num_qo_heads 32 \
    --num_kv_heads 8 \
    --head_dim_qk 128 \
    --head_dim_vo 128 \
    --q_dtype bfloat16 \
    --kv_dtype bfloat16 \
    --num_iters 30 \
    --dry_run_iters 5 \
    --refcheck \
    -vv
```

Example - Benchmark FP8 GEMM:

```bash
python benchmarks/flashinfer_benchmark.py \
    --routine bmm_fp8 \
    --backends cudnn cublas cutlass \
    --batch_size 256 \
    --m 1 \
    --n 1024 \
    --k 7168 \
    --input_dtype fp8_e4m3 \
    --mat2_dtype fp8_e4m3 \
    --out_dtype bfloat16 \
    --refcheck \
    -vv \
    --generate_repro_command
```

**Timing behavior:**
- ✅ If CUPTI installed: Uses CUPTI (most accurate)
- ⚠️ If CUPTI not installed: Automatically falls back to CUDA events with warning
- 🔧 To force CUDA events: Add `--use_cuda_events` flag

### Step 3: Understand the Output

```
[INFO] FlashInfer version: 0.6.0
[VVERBOSE] gpu_name = 'NVIDIA_H100_PCIe'
[PERF] fa2            :: median time 0.145 ms; std 0.002 ms; achieved tflops 125.3 TFLOPs/sec; achieved tb_per_sec 1.87 TB/sec
[PERF] fa2_tc         :: median time 0.138 ms; std 0.001 ms; achieved tflops 131.5 TFLOPs/sec; achieved tb_per_sec 1.96 TB/sec
[PERF] cudnn          :: median time 0.142 ms; std 0.001 ms; achieved tflops 127.8 TFLOPs/sec; achieved tb_per_sec 1.91 TB/sec
```

**Key metrics:**
- **median time**: Median kernel execution time (lower is better)
- **std**: Standard deviation (lower means more consistent)
- **achieved tflops**: Effective TFLOPS throughput
- **achieved tb_per_sec**: Memory bandwidth utilization

### Step 4: Run Batch Benchmarks

Create a test list file `my_benchmarks.txt`:

```bash
--routine BatchDecodeWithPagedKVCacheWrapper --backends fa2 cudnn --page_size 16 --batch_size 32 --s_kv 2048 --num_qo_heads 32 --num_kv_heads 8 --head_dim_qk 128 --head_dim_vo 128
--routine BatchDecodeWithPagedKVCacheWrapper --backends fa2 cudnn --page_size 16 --batch_size 64 --s_kv 4096 --num_qo_heads 32 --num_kv_heads 8 --head_dim_qk 128 --head_dim_vo 128
--routine bmm_fp8 --backends cudnn cutlass --batch_size 256 --m 1 --n 1024 --k 7168 --input_dtype fp8_e4m3 --mat2_dtype fp8_e4m3 --out_dtype bfloat16
```

Run all tests:

```bash
python benchmarks/flashinfer_benchmark.py \
    --testlist my_benchmarks.txt \
    --output_path results.csv \
    --generate_repro_command \
    --refcheck
```

Results are saved to `results.csv` with all metrics and reproducer commands.

### Step 5: Common Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--num_iters` | Measurement iterations | 30 |
| `--dry_run_iters` | Warmup iterations | 5 |
| `--refcheck` | Verify output correctness | False |
| `--allow_output_mismatch` | Continue on mismatch | False |
| `--use_cuda_events` | Force CUDA events (skip CUPTI) | False |
| `--no_cuda_graph` | Disable CUDA graph | False |
| `-vv` | Very verbose output | - |
| `--generate_repro_command` | Print reproducer command | False |
| `--case_tag` | Tag for CSV output | None |

## Method 2: Using bench_gpu_time() in Python

For custom benchmarking in your own code:

### Step 1: Write Your Benchmark Script

```python
import torch
from flashinfer.testing import bench_gpu_time

# Setup your kernel
def my_kernel_wrapper(q, k, v):
    # Your kernel call here
    return output

# Create test inputs
device = torch.device("cuda")
q = torch.randn(32, 8, 128, dtype=torch.bfloat16, device=device)
k = torch.randn(2048, 8, 128, dtype=torch.bfloat16, device=device)
v = torch.randn(2048, 8, 128, dtype=torch.bfloat16, device=device)

# Benchmark - CUPTI preferred, CUDA events if CUPTI unavailable
median_time, std_time = bench_gpu_time(
    my_kernel_wrapper,
    args=(q, k, v),
    enable_cupti=True,          # Prefer CUPTI, fallback to CUDA events
    num_iters=30,               # Number of iterations
    dry_run_iters=5,            # Warmup iterations
)

print(f"Kernel time: {median_time:.3f} ms ± {std_time:.3f} ms")

# Calculate FLOPS if you know the operation count
flops = ...  # Your FLOP count
tflops = (flops / 1e12) / (median_time / 1000)
print(f"Achieved: {tflops:.2f} TFLOPS/sec")
```

**Note**: If CUPTI is not installed, you'll see a warning and the function will automatically use CUDA events instead.

### Step 2: Run Your Benchmark

```bash
python my_benchmark.py
```

Output with CUPTI:
```
Kernel time: 0.145 ms ± 0.002 ms
Achieved: 125.3 TFLOPS/sec
```

Output without CUPTI (automatic fallback):
```
[WARNING] CUPTI is not installed. Try 'pip install -U cupti-python'. Falling back to CUDA events.
Kernel time: 0.147 ms ± 0.003 ms
Achieved: 124.1 TFLOPS/sec
```

### Step 3: Advanced Options

```python
# Cold L2 cache benchmarking (optional)
median_time, std_time = bench_gpu_time(
    my_kernel,
    args=(x, y),
    enable_cupti=True,          # Will use CUDA events if CUPTI unavailable
    cold_l2_cache=True,         # Flush L2 or rotate buffers automatically
    num_iters=30
)

# Force CUDA events (skip CUPTI even if installed)
median_time, std_time = bench_gpu_time(
    my_kernel,
    args=(x, y),
    enable_cupti=False,         # Explicitly use CUDA events
    num_iters=30
)
```

## Troubleshooting

### CUPTI Warning Message

**Warning**: `CUPTI is not installed. Falling back to CUDA events.`

**What it means**: CUPTI is not available, using CUDA events instead

**Impact**: Less accurate for very fast kernels (5-50 us) due to synchronization overhead, but becomes negligible for longer-running kernels

**Solution (optional)**: Install CUPTI for best accuracy:
```bash
pip install -U cupti-python
```

If installation fails, check:
- CUDA version >= 13
- Compatible `cupti-python` version

**You can still run benchmarks without CUPTI** - the framework handles this automatically.

### Inconsistent Results

**Problem**: Large standard deviation or varying results

**Solutions**:
1. **Increase warmup iterations**:
   ```bash
   --dry_run_iters 10
   ```

2. **Increase measurement iterations**:
   ```bash
   --num_iters 50
   ```

3. **Use cold L2 cache** (in Python):
   ```python
   bench_gpu_time(..., rotate_buffers=True)
   ```

4. **Disable GPU boost** (advanced):
   ```bash
   sudo nvidia-smi -lgc <base_clock>
   ```

### Reference Check Failures

**Error**: `[ERROR] Output mismatch between backends`

**What it means**: Different backends produce different results

**Solutions**:
1. **Allow mismatch and continue**:
   ```bash
   --allow_output_mismatch
   ```

2. **Check numerical tolerance**: Some backends use different precisions (FP32 vs FP16)

3. **Investigate the difference**:
   ```bash
   -vv  # Very verbose mode shows tensor statistics
   ```

### Backend Not Supported

**Error**: `[WARNING] fa3 for routine ... is not supported on compute capability X.X`

**Solution**: Check the backend support matrix in `benchmarks/README.md` or remove that backend from `--backends` list

## Best Practices

1. **Install CUPTI for best accuracy** (but not required):
   ```bash
   pip install -U cupti-python
   ```

2. **Use reference checking** to verify correctness:
   ```bash
   --refcheck
   ```

3. **Use verbose mode** to see input shapes and dtypes:
   ```bash
   -vv
   ```

4. **Generate reproducer commands** for sharing results:
   ```bash
   --generate_repro_command
   ```

5. **Run multiple iterations** for statistical significance:
   ```bash
   --num_iters 30 --dry_run_iters 5
   ```

6. **Save results to CSV** for later analysis:
   ```bash
   --output_path results.csv
   ```

7. **Compare multiple backends** to find the best:
   ```bash
   --backends fa2 fa3 cudnn cutlass
   ```

## Quick Examples

### Decode Attention (H100)
```bash
python benchmarks/flashinfer_benchmark.py \
    --routine BatchDecodeWithPagedKVCacheWrapper \
    --backends fa2 fa2_tc cudnn trtllm-gen \
    --page_size 16 --batch_size 128 --s_kv 8192 \
    --num_qo_heads 64 --num_kv_heads 8 \
    --head_dim_qk 128 --head_dim_vo 128 \
    --refcheck -vv --generate_repro_command
```

### Prefill Attention (Multi-head)
```bash
python benchmarks/flashinfer_benchmark.py \
    --routine BatchPrefillWithRaggedKVCacheWrapper \
    --backends fa2 fa3 cudnn cutlass \
    --batch_size 16 --s_qo 1024 --s_kv 1024 \
    --num_qo_heads 128 --num_kv_heads 128 \
    --head_dim_qk 192 --head_dim_vo 128 \
    --causal --random_actual_seq_len \
    --q_dtype bfloat16 --kv_dtype bfloat16 \
    --refcheck -vv
```

### FP8 GEMM (Batched)
```bash
python benchmarks/flashinfer_benchmark.py \
    --routine bmm_fp8 \
    --backends cudnn cublas cutlass \
    --batch_size 256 --m 1 --n 1024 --k 7168 \
    --input_dtype fp8_e4m3 --mat2_dtype fp8_e4m3 \
    --out_dtype bfloat16 \
    --refcheck -vv
```

### MOE (DeepSeek-style routing)
```bash
python benchmarks/flashinfer_benchmark.py \
    --routine trtllm_fp8_block_scale_moe \
    --backends trtllm \
    --num_tokens 1024 --hidden_size 5120 \
    --intermediate_size 13824 --num_experts 256 \
    --top_k 8 --n_group 8 --topk_group 1 \
    --routing_method deepseek_v3 \
    --routed_scaling_factor 2.5 \
    --use_routing_bias \
    -vv
```

## Summary: CUPTI vs CUDA Events

| Aspect | CUPTI (Preferred) | CUDA Events (Fallback) |
|--------|-------------------|------------------------|
| **Accuracy** | Highest (hardware-level) | Good (slight overhead) |
| **Installation** | `pip install cupti-python` | Built-in with CUDA |
| **Requirements** | CUDA 13+ | Any CUDA version |
| **Fallback** | N/A | Automatic if CUPTI unavailable |
| **When to use** | Always (if available) | When CUPTI can't be installed |

**Recommendation**: Install CUPTI for best results, but benchmarks work fine without it.

## Next Steps

- **Profile kernels** with `nsys` or `ncu` for detailed analysis
- **Debug performance issues** using `FLASHINFER_LOGLEVEL=3`
- **Compare with baselines** using reference implementations
- **Optimize kernels** based on profiling results

## Related Documentation

- See `benchmarks/README.md` for full flag documentation
- See `benchmarks/samples/sample_testlist.txt` for more examples
- See CLAUDE.md "Benchmarking" section for technical details
