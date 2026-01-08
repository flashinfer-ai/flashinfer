---
name: debug-cuda-crash
description: Tutorial for debugging CUDA crashes using API logging
---

# Tutorial: Debugging CUDA Crashes with API Logging

This tutorial shows you how to debug CUDA crashes and errors in FlashInfer using the `@flashinfer_api` logging decorator.

## Goal

When your code crashes with CUDA errors (illegal memory access, out-of-bounds, NaN/Inf), use API logging to:
- Capture input tensors BEFORE the crash occurs
- Understand what data caused the problem
- Track tensor shapes, dtypes, and values through your pipeline
- Detect numerical issues (NaN, Inf, wrong shapes)

## Why Use API Logging?

**Problem**: CUDA errors often crash the program, leaving no debugging information.

**Solution**: FlashInfer's `@flashinfer_api` decorator logs inputs BEFORE execution, so you can see what caused the crash even after the program terminates.

## Step 1: Enable API Logging

### Basic Logging (Function Names Only)

```bash
export FLASHINFER_LOGLEVEL=1        # Log function names
export FLASHINFER_LOGDEST=stdout    # Log to console

python my_script.py
```

Output:
```
[2025-12-18 10:30:45] FlashInfer API Call: batch_decode_with_padded_kv_cache
```

### Detailed Logging (Inputs/Outputs with Metadata)

```bash
export FLASHINFER_LOGLEVEL=3        # Log inputs/outputs with metadata
export FLASHINFER_LOGDEST=debug.log # Save to file

python my_script.py
```

Output in `debug.log`:
```
================================================================================
[2025-12-18 10:30:45] FlashInfer API Logging - System Information
================================================================================
FlashInfer version: 0.6.0
CUDA toolkit version: 12.1
GPU 0: NVIDIA H100 PCIe
  Compute capability: 9.0 (SM90)
PyTorch version: 2.1.0
================================================================================

================================================================================
[2025-12-18 10:30:46] FlashInfer API Call: batch_decode_with_padded_kv_cache
--------------------------------------------------------------------------------
Positional input arguments:
  arg[0]:
    Tensor(
      shape=(32, 8, 128)
      dtype=torch.bfloat16
      device=cuda:0
      requires_grad=False
      is_contiguous=True
    )
Keyword input arguments:
  kv_cache=
    Tensor(
      shape=(1024, 2, 8, 128)
      dtype=torch.bfloat16
      device=cuda:0
      requires_grad=False
      is_contiguous=True
    )
```

### Full Logging (With Tensor Statistics)

```bash
export FLASHINFER_LOGLEVEL=5        # Log with min/max/mean/nan/inf
export FLASHINFER_LOGDEST=debug.log

python my_script.py
```

Additional output:
```
  Tensor(
    shape=(32, 8, 128)
    dtype=torch.bfloat16
    device=cuda:0
    requires_grad=False
    is_contiguous=True
    min=-3.125000
    max=4.250000
    mean=0.015625
    nan_count=0
    inf_count=0
  )
```

## Step 2: Reproduce the Crash

### Example: Shape Mismatch

Your code crashes with:
```
RuntimeError: CUDA error: an illegal memory access was encountered
```

Enable logging and run again:

```bash
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=crash_log.txt

python my_script.py
```

The log shows inputs before the crash:
```
[2025-12-18 10:32:15] FlashInfer API Call: batch_decode_with_padded_kv_cache
Positional input arguments:
  arg[0]:
    Tensor(
      shape=(32, 8, 128)      # Query tensor
      ...
    )
Keyword input arguments:
  kv_cache=
    Tensor(
      shape=(1024, 2, 8, 64)  # ❌ Wrong! Should be (..., 128) not (..., 64)
      ...
    )
```

**Found the bug**: `head_dim` mismatch (64 vs 128)

## Step 3: Common CUDA Errors and How to Debug

### Error 1: Illegal Memory Access

**Error Message**:
```
RuntimeError: CUDA error: an illegal memory access was encountered
```

**Enable logging**:
```bash
export FLASHINFER_LOGLEVEL=3
python my_script.py
```

**What to check in logs**:
- ✅ Tensor shapes match expected dimensions
- ✅ All tensors are on CUDA (not CPU)
- ✅ Tensor strides are reasonable
- ✅ `is_contiguous=True` (if required)

**Common causes**:
- Wrong tensor dimensions
- CPU tensor passed to GPU kernel
- Incorrect stride patterns

### Error 2: NaN or Inf Values

**Error Message**:
```
RuntimeError: Function ... returned nan or inf
```

**Enable statistics logging**:
```bash
export FLASHINFER_LOGLEVEL=5        # Level 5 shows nan_count, inf_count
python my_script.py
```

**What to check in logs**:
```
Tensor(
  ...
  min=-1234567.000000     # ❌ Suspiciously large
  max=9876543.000000      # ❌ Suspiciously large
  mean=nan                # ❌ NaN detected
  nan_count=128           # ❌ 128 NaN values!
  inf_count=0
)
```

**Common causes**:
- Division by zero in previous operation
- Numerical overflow/underflow
- Uninitialized memory

### Error 3: Out of Memory

**Error Message**:
```
RuntimeError: CUDA out of memory
```

**Enable logging**:
```bash
export FLASHINFER_LOGLEVEL=3
python my_script.py
```

**What to check in logs**:
- ✅ Tensor shapes (are they unexpectedly large?)
- ✅ Batch size
- ✅ Sequence length

Example:
```
Tensor(
  shape=(1024, 8192, 128, 128)  # ❌ Way too large! Should be (1024, 128, 128)?
  ...
)
```

### Error 4: Wrong Dtype

**Error Message**:
```
RuntimeError: expected scalar type BFloat16 but found Float16
```

**Enable logging**:
```bash
export FLASHINFER_LOGLEVEL=3
python my_script.py
```

**What to check in logs**:
```
Tensor(
  dtype=torch.float16     # ❌ Should be torch.bfloat16
  ...
)
```

## Step 4: Multi-Process Debugging

When running with multiple GPUs/processes, use `%i` pattern:

```bash
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=debug_rank_%i.txt    # %i = process ID

torchrun --nproc_per_node=4 my_script.py
```

This creates separate logs:
- `debug_rank_12345.txt` (process 12345)
- `debug_rank_12346.txt` (process 12346)
- `debug_rank_12347.txt` (process 12347)
- `debug_rank_12348.txt` (process 12348)

Now you can debug each rank independently.

## Step 5: Advanced Debugging with compute-sanitizer

For harder bugs, combine API logging with CUDA tools:

### Use compute-sanitizer (Memory Checker)

```bash
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=debug.log

compute-sanitizer --tool memcheck python my_script.py
```

Output shows exact memory errors:
```
========= COMPUTE-SANITIZER
========= Invalid __global__ write of size 4 bytes
=========     at 0x1234 in ScaleKernel<float>
=========     by thread (256,0,0) in block (10,0,0)
=========     Address 0x7f1234567890 is out of bounds
```

Check `debug.log` to see what inputs caused this kernel to fail.

### Use cuda-gdb (Debugger)

```bash
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=debug.log

cuda-gdb --args python my_script.py
```

In gdb:
```
(cuda-gdb) run
(cuda-gdb) where     # Show stack trace when it crashes
```

Check `debug.log` for the inputs that led to the crash.

## Step 6: Kernel-Level Debugging with printf()

You can use `printf()` inside CUDA kernels for debugging:

### Basic Usage

```cpp
__global__ void MyKernel(const float* input, float* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Print from one thread to avoid spam
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("n=%d, input[0]=%f\n", n, input[0]);
  }

  if (idx < n) {
    output[idx] = input[idx] * 2.0f;
  }
}
```

**Important**: Flush printf buffer after kernel:
```python
my_kernel(input, output)
torch.cuda.synchronize()  # ← Flushes printf output
```

### ⚠️ Warp-Specialized Kernels: Choosing the Right Print Thread

**Problem**: `threadIdx.x == 0` doesn't work for all warps (warp starting at thread 32 won't have thread 0).

**Solution**: Choose one representative thread per specialization group.

```cpp
__global__ void WarpSpecializedKernel(...) {
  // Define your group's representative thread
  // e.g., first thread of each warp: threadIdx.x % 32 == 0
  // e.g., first thread of each 4-warp group: threadIdx.x % 128 == 0

  if (is_group_representative) {
    printf("Group %d processing\n", group_id);
  }
}
```

**Common mistake** ❌:
```cpp
// ❌ Only warp 0 will print!
if (threadIdx.x == 0) {
  printf("Warp %d processing\n", threadIdx.x / 32);
}
```

### Quick Reference

| Kernel Type | Print Condition | Notes |
|-------------|-----------------|-------|
| Simple kernel | `threadIdx.x == 0` | One thread per block |
| Warp-specialized | One thread per group | Depends on kernel design |

### Other Kernel Debugging Tools

```cpp
// Assert for invariants
assert(value >= 0.0f && "Value must be non-negative");

// Compile-time checks
static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be multiple of warp size");
```

## Environment Variables Reference

| Variable | Values | Description |
|----------|--------|-------------|
| `FLASHINFER_LOGLEVEL` | `0` | No logging (default) |
|  | `1` | Function names only |
|  | `3` | Inputs/outputs with metadata |
|  | `5` | + Tensor statistics (min/max/mean/nan/inf) |
| `FLASHINFER_LOGDEST` | `stdout` | Log to console (default) |
|  | `stderr` | Log to stderr |
|  | `<path>` | Log to file |
|  | `log_%i.txt` | Multi-process: %i = process ID |

## Best Practices

### 1. Always Start with Level 3

```bash
export FLASHINFER_LOGLEVEL=3
```

Level 3 provides tensor metadata (shape, dtype, device) without overwhelming output.

### 2. Use Level 5 for Numerical Issues

```bash
export FLASHINFER_LOGLEVEL=5
```

Only use level 5 when debugging NaN/Inf problems (adds statistics).

### 3. Log to File for Crashes

```bash
export FLASHINFER_LOGDEST=crash_log.txt
```

Console output may be lost when program crashes. File logs persist.

### 4. Compare Before/After

Enable logging and compare:
- Last successful API call (inputs logged, outputs logged) ✅
- First failed API call (inputs logged, no outputs) ❌ ← This is where it crashed!

### 5. Disable Logging in Production

```bash
unset FLASHINFER_LOGLEVEL   # or export FLASHINFER_LOGLEVEL=0
```

Logging has zero overhead when disabled (decorator returns original function).

## Troubleshooting

### No Logs Appearing

**Problem**: Set `FLASHINFER_LOGLEVEL=3` but no logs appear

**Solutions**:
1. **Check if API has the decorator**: Not all FlashInfer APIs have `@flashinfer_api` yet (work in progress)

2. **Verify environment variable**:
   ```bash
   echo $FLASHINFER_LOGLEVEL    # Should print "3"
   ```

3. **Check log destination**:
   ```bash
   echo $FLASHINFER_LOGDEST     # Should print path or "stdout"
   ```

### Too Much Output

**Problem**: Level 5 produces too much output

**Solution**: Use level 3 instead:
```bash
export FLASHINFER_LOGLEVEL=3   # Skip tensor statistics
```

### Statistics Skipped in CUDA Graph

**Warning**: `[statistics skipped: CUDA graph capture in progress]`

**What it means**: Level 5 statistics are automatically skipped during CUDA graph capture (to avoid synchronization)

**This is normal**: The framework protects you from graph capture issues.

## Quick Examples

### Debug Shape Mismatch
```bash
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=stdout
python my_script.py
# Check tensor shapes in output
```

### Debug NaN/Inf
```bash
export FLASHINFER_LOGLEVEL=5         # Show statistics
export FLASHINFER_LOGDEST=debug.log
python my_script.py
# Check nan_count and inf_count in debug.log
```

### Debug Multi-GPU Training
```bash
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=rank_%i.log   # Separate log per rank
torchrun --nproc_per_node=8 train.py
# Check rank_*.log files
```

### Combine with Memory Checker
```bash
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=inputs.log
compute-sanitizer --tool memcheck python my_script.py
# inputs.log shows what data caused the memory error
```

## Example: Full Debug Session

### Your code crashes:
```python
import torch
from flashinfer import batch_decode_with_padded_kv_cache

q = torch.randn(32, 8, 128, dtype=torch.bfloat16, device="cuda")
kv = torch.randn(1024, 2, 8, 64, dtype=torch.bfloat16, device="cuda")  # Wrong dim!

output = batch_decode_with_padded_kv_cache(q, kv)  # ❌ Crashes
```

### Enable logging:
```bash
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=debug.log
python test.py
```

### Check debug.log:
```
[2025-12-18 10:45:23] FlashInfer API Call: batch_decode_with_padded_kv_cache
Positional input arguments:
  arg[0]:
    Tensor(
      shape=(32, 8, 128)
      dtype=torch.bfloat16
      ...
    )
  arg[1]:
    Tensor(
      shape=(1024, 2, 8, 64)    # ❌ Found it! Last dim should be 128
      dtype=torch.bfloat16
      ...
    )
```

### Fix the bug:
```python
kv = torch.randn(1024, 2, 8, 128, dtype=torch.bfloat16, device="cuda")  # ✅ Fixed
```

### Success!
```bash
python test.py
# No crash, outputs logged successfully
```

## Summary

1. **Enable logging** before the crash:
   ```bash
   export FLASHINFER_LOGLEVEL=3
   export FLASHINFER_LOGDEST=debug.log
   ```

2. **Run your code** - inputs are logged BEFORE crash

3. **Check the log** - last API call shows what caused the crash

4. **Fix the issue** based on logged input metadata

5. **Disable logging** when done:
   ```bash
   export FLASHINFER_LOGLEVEL=0
   ```

## Related Documentation

- See CLAUDE.md "API Logging with @flashinfer_api" for technical details
- See `flashinfer/api_logging.py` for implementation
- See CUDA documentation for compute-sanitizer and cuda-gdb
