---
name: add-cuda-kernel
description: Step-by-step tutorial for adding new CUDA kernels to FlashInfer
---

# Tutorial: Adding a New Kernel to FlashInfer

This tutorial walks through adding a simple element-wise scale operation to FlashInfer. We'll implement `scale(x, factor) = x * factor` to demonstrate the complete workflow.

## Goal

Add a new operation that scales each element of a tensor by a scalar factor:

- Input: tensor `x` and scalar `factor`
- Output: `x * factor` (element-wise)
- Support multiple dtypes (FP16, BF16, FP32)

## Step 1: Define CUDA Kernel in `include/`

Create `include/flashinfer/scale.cuh`:

```cpp
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace flashinfer {

/*!
 * \brief Element-wise scale kernel
 * \tparam T Data type (half, __nv_bfloat16, float)
 * \param input Input tensor
 * \param output Output tensor
 * \param factor Scale factor
 * \param n Number of elements
 */
template <typename T>
__global__ void ScaleKernel(const T* input, T* output, T factor, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = input[idx] * factor;
  }
}

/*!
 * \brief Launch scale kernel
 * \tparam T Data type
 * \param input Input pointer
 * \param output Output pointer
 * \param factor Scale factor
 * \param n Number of elements
 * \param stream CUDA stream
 */
template <typename T>
cudaError_t ScaleLauncher(const T* input, T* output, T factor, int n,
                          cudaStream_t stream = nullptr) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;

  ScaleKernel<T><<<blocks, threads, 0, stream>>>(input, output, factor, n);

  return cudaGetLastError();
}

}  // namespace flashinfer
```

**Key points:**

- Framework-agnostic (no Torch headers)
- Uses raw pointers
- Template-based for dtype flexibility
- Only includes what's needed (cuda_runtime, cuda_fp16, cuda_bf16)

## Step 2: Create Launcher in `csrc/`

Create `csrc/scale.cu`:

```cpp
#include "flashinfer/scale.cuh"

using namespace flashinfer;

void scale_launcher(TensorView input, TensorView output,
                    float factor) {
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  TVM_FFI_ICHECK_EQ(input.dtype(), output.dtype());
  int n = input.numel();
  auto stream = get_stream(input.device());

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(input.dtype(), DType, [&] {
    cudaError_t status = ScaleLauncher<DType>(
      input.data_ptr<DType>(),
      output.data_ptr<DType>(),
      static_cast<DType>(factor),
      n,
      stream
    );
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "Failed to run ScaleLauncher: " << cudaGetErrorString(status);
    return true;
  });
}
```

**Key points:**

- Includes TVM FFI utils headers `tvm_ffi_utils.h` (only allowed in `csrc/`)
- Uses `tvm::ffi::TensorView` as input and output tensor types
- Uses macros defined in `tvm_ffi_utils.h` to check the input and output if both on CUDA device, both contiguous, and share the same data type
- Gets CUDA stream by TVM FFI, and prepare all scalar inputs for kernel function
- Dispatches on dtype with macros defined in `tvm_ffi_utils.h`, or adds new one if not covered
- Converts tvm::ffi::TensorView to raw pointers
- Handles the result status of kernel by `TVM_FFI_ICHECK`
- Add descriptive error messages with `<<` operator
- **Use TVM-FFI exceptions**: `TVM_FFI_THROW(ErrorType) << "message"` for custom error checking

**TVM-FFI Error Handling:**

- `TVM_FFI_THROW(ValueError) << "message"` - Throw ValueError with custom message
- `TVM_FFI_THROW(TypeError) << "message"` - Throw TypeError
- Use `<<` to chain multiple values in the error message
- Errors are properly propagated back to Python

**When to use `TVM_FFI_THROW` vs `TVM_FFI_LOG_AND_THROW`:**

- **`TVM_FFI_THROW`**: Use for normal runtime error handling. This is the standard way to report errors that will be caught and propagated to Python.

- **`TVM_FFI_LOG_AND_THROW`**: Use only in cases where:
  1. The function may be called during object construction time (e.g., validation in constructors or setup methods)
  2. The exception may not be caught properly (e.g., during module initialization)
  3. The error condition almost never fails in practice (e.g., internal errors, unsupported dtype combinations in dispatch macros)

  This variant logs the error message before throwing, ensuring visibility even if the exception doesn't propagate correctly.

**Example from fused_moe (see `csrc/trtllm_fused_moe_kernel_launcher.cu`):**
```cpp
// In a setup/validation function that may be called during construction
void check_weights_shape(std::string which_weights) const {
  if (which_weights != "gemm1" && which_weights != "gemm2") {
    // Internal error that should never happen - use LOG_AND_THROW
    TVM_FFI_LOG_AND_THROW(InternalError) << "Internal error: which_weights = " << which_weights;
  }
  // ...
  if (weight_layout is unsupported) {
    // Unsupported config during setup - use LOG_AND_THROW
    TVM_FFI_LOG_AND_THROW(NotImplementedError)
        << "Unsupported weight_layout: " << (int)weight_layout;
  }
}

// In a normal runtime function
void scale_run(TensorView input, TensorView output, double factor) {
  if (!input_tensor.is_cuda()) {
    // Normal validation error - use TVM_FFI_THROW
    TVM_FFI_THROW(ValueError) << "Input must be a CUDA tensor";
  }
}
```

## Step 3: Create TVM-FFI Binding in `csrc/`

Create `csrc/scale_jit_binding.cu`:

```cpp
#include "scale.cu"
#include "tvm_ffi_utils.h"

// Forward declaration
void scale_launcher(TensorView input, TensorView output, float factor);

// Export to TVM-FFI
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, scale_launcher);
```

**Key points:**

- Forward declare the launcher function first
- Export using `TVM_FFI_DLL_EXPORT_TYPED_FUNC(name, function)`

## Step 4: Create JIT Generator (No Jinja for Simple Case)

Create `flashinfer/jit/scale.py`:

```python
import os
import shutil
from pathlib import Path

from . import JitSpec, gen_jit_spec
from . import env as jit_env
from .core import write_if_different


def get_scale_uri(dtype_in: str, dtype_out: str) -> str:
    """Generate unique identifier for scale module."""
    return f"scale_dtype_in_{dtype_in}_dtype_out_{dtype_out}"


def gen_scale_module(dtype_in, dtype_out):
    """
    Generate JIT module for scale operation.

    Note: This is a simple example without Jinja templating.
    The dtype dispatch is handled at runtime in the C++ code.
    """
    # Compute URI
    uri = get_scale_uri(dtype_in, dtype_out)

    # Create generation directory
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    os.makedirs(gen_directory, exist_ok=True)

    # Copy source files (no Jinja needed for this simple case)
    sources = []
    for fname in ["scale.cu", "scale_jit_binding.cu"]:
        src_path = jit_env.FLASHINFER_CSRC_DIR / fname
        dest_path = gen_directory / fname
        shutil.copy(src_path, dest_path)
        sources.append(dest_path)

    # Return JitSpec
    return gen_jit_spec(
        name=uri,
        sources=sources,
        extra_cuda_cflags=[],
    )
```

**Key points:**

- No Jinja template needed for simple operations
- Just copy source files to generation directory
- URI uniquely identifies the module configuration

### (Optional) Specifying Supported CUDA Architectures

FlashInfer uses `CompilationContext` to manage CUDA architecture targets. This is critical because some kernels only work on specific GPU architectures (e.g., Hopper SM90, Blackwell SM100).

#### How CompilationContext Works

**Automatic Detection** (default):
```python
from flashinfer.compilation_context import CompilationContext

ctx = CompilationContext()
# Automatically detects all GPUs in the system
# For SM90+, adds 'a' suffix (e.g., 9.0a for Hopper)
# Result: ctx.TARGET_CUDA_ARCHS = {(9, '0a'), (10, '0a'), ...}
```

**Manual Override** (via environment variable):
```bash
export FLASHINFER_CUDA_ARCH_LIST="8.0 9.0a 10.0a"
# Now only these architectures will be compiled
```

#### Specifying Architectures in Your JIT Module

When creating a JIT module, specify which major SM versions are supported:

```python
from flashinfer.jit.core import gen_jit_spec
from flashinfer.jit import current_compilation_context

def gen_my_hopper_only_module():
    """Example: Kernel works on SM90 and later supported architectures."""
    uri = get_my_uri(...)
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    # ... copy sources ...

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        # Explicitly list supported SM versions - no automatic future compatibility
        supported_major_versions=[9, 10, 11, 12]  # SM90, SM100, SM110, SM120
    )

    return gen_jit_spec(
        name=uri,
        sources=sources,
        extra_cuda_cflags=nvcc_flags,
    )

def gen_my_blackwell_only_module():
    """Example: Kernel only works on SM100 (Blackwell)"""
    uri = get_my_uri(...)
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    # ... copy sources ...

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10]  # SM100 only
    )

    return gen_jit_spec(
        name=uri,
        sources=sources,
        extra_cuda_cflags=nvcc_flags,
    )

def gen_my_universal_module():
    """Example: Kernel works on all architectures"""
    uri = get_my_uri(...)
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    # ... copy sources ...

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=None  # All available architectures
    )

    return gen_jit_spec(
        name=uri,
        sources=sources,
        extra_cuda_cflags=nvcc_flags,
    )
```

**What Happens:**
- ✅ If user's GPU is SM90 and they call a Hopper-only module → Compiles and runs
- ❌ If user's GPU is SM80 and they call a Hopper-only module → `RuntimeError: No supported CUDA architectures found for major versions [9, 10, 11, 12]`

#### Real Examples from FlashInfer

```python
# MLA kernel: Blackwell and newer only
def gen_mla_module() -> JitSpec:
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10, 11]  # SM100, SM110
    )
    return gen_jit_spec(
        name=uri,
        sources=sources,
        extra_cuda_cflags=nvcc_flags,
    )

# Blackwell FMHA: SM120 only
def gen_fmhav2_blackwell_module(...):
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[12]  # SM120 only
    )
    return gen_jit_spec(
        name=uri,
        sources=sources,
        extra_cuda_cflags=nvcc_flags,
    )

# Standard attention: Hopper and later supported architectures
def gen_batch_prefill_module(...):
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[9, 10, 11, 12]  # SM90, SM100, SM110, SM120
    )
    return gen_jit_spec(
        name=uri,
        sources=sources,
        extra_cuda_cflags=nvcc_flags,
    )
```

#### Common Architecture Specifications

| Supported Versions | Architectures | Use Case |
|-------------------|---------------|----------|
| `None` | All available GPUs | Universal kernels (default) |
| `[9, 10, 11, 12]` | SM90, SM100, SM110, SM120 | Hopper, Blackwell |
| `[10, 11, 12]` | SM100, SM110, SM120 | Blackwell only |
| `[12]` | SM120 | Specific architecture only |
| `[8, 9, 10, 11, 12]` | SM80, SM90, SM100, SM110, SM120 | Ampere, Hopper, Blackwell |

#### Testing with Architecture Requirements

When your kernel has architecture requirements, add skip checks in tests (see Step 6 below):

```python
import pytest
import torch
from flashinfer.utils import is_sm90a_supported

def test_hopper_kernel():
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90a is not supported on this GPU")

    # Test code here
    ...
```

## Step 5: Create Python API in `flashinfer/`

Create `flashinfer/scale.py`:

```python
import functools
import torch
from typing import Optional

from .jit.scale import gen_scale_module
from .utils import backend_requirement, supported_compute_capability
from .api_logging import flashinfer_api


@functools.cache
def get_scale_module(dtype_in, dtype_out):
    """Get or compile scale module (cached)."""
    return gen_scale_module(dtype_in, dtype_out).build_and_load()


@supported_compute_capability([80, 86, 89, 90, 100, 103, 110, 120])
def _check_scale_problem_size(input: torch.Tensor, factor: float,
                               out: Optional[torch.Tensor] = None) -> bool:
    """Validate inputs for scale operation."""
    # Validate input
    if not input.is_cuda:
        raise ValueError("Input must be a CUDA tensor")

    # Validate output if provided
    if out is not None:
        if out.shape != input.shape:
            raise ValueError("Output shape mismatch")
        if out.dtype != input.dtype:
            raise ValueError("Output dtype mismatch")
        if not out.is_cuda:
            raise ValueError("Output must be a CUDA tensor")

    return True


@flashinfer_api
@backend_requirement(
    backend_checks={},  # No backend choices for this simple kernel
    common_check=_check_scale_problem_size,
)
def scale(input: torch.Tensor, factor: float,
          out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Element-wise scale operation.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor (CUDA)
    factor : float
        Scale factor
    out : Optional[torch.Tensor]
        Output tensor (if None, allocate new tensor)

    Returns
    -------
    output : torch.Tensor
        Scaled tensor (input * factor)

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> x = torch.randn(1024, dtype=torch.float16, device="cuda")
    >>> y = flashinfer.scale(x, 2.0)
    >>> torch.allclose(y, x * 2.0)
    True
    """
    # Allocate output if needed
    if out is None:
        out = torch.empty_like(input)

    # Get module (compile if first call with this dtype)
    dtype_str = str(input.dtype).replace("torch.", "")
    module = get_scale_module(dtype_str, dtype_str)

    # Call TVM-FFI function (exported as "run")
    module.run(input, out, float(factor))

    return out
```

**Key points:**

- Uses `@functools.cache` to cache compiled modules
- Clean Python API with docstring
- Adding the `@flashinfer_api` decorator enables logging and sets it apart from helper functions
- **Destination passing style**: Output tensor(s) are passed as optional parameters (`out: Optional[torch.Tensor] = None`). This allows users to provide pre-allocated buffers to avoid allocation overhead in performance-critical paths. Only allocate internally when `out` is `None`.
- Validates inputs using `@backend_requirement` decorator

### Using `@backend_requirement` and `@supported_compute_capability` Decorators

FlashInfer provides two decorators for enforcing compute capability and backend requirements:

#### `@supported_compute_capability` Decorator

Marks a function with its supported CUDA compute capabilities:

```python
from flashinfer.utils import supported_compute_capability

@supported_compute_capability([80, 86, 89, 90, 100, 103, 110, 120])
def _my_check_function(input, output):
    """Supports SM80 (Ampere) through SM120 (Blackwell)."""
    # Validation logic here
    return True
```

#### `@backend_requirement` Decorator

Enforces backend and problem size requirements at runtime. There are three usage patterns:

**Pattern 1: Single Backend (No Backend Choices)**

For kernels with only one implementation (like our scale example):

```python
from flashinfer.utils import backend_requirement, supported_compute_capability

@supported_compute_capability([80, 86, 89, 90, 100, 103, 110, 120])
def _check_my_kernel(input, output):
    """Validate inputs. Must return True if valid."""
    if input.shape[-1] > 256:
        raise ValueError("Head dimension must be <= 256")
    return True

@backend_requirement(
    backend_checks={},  # Empty dict = no backend parameter
    common_check=_check_my_kernel,
)
def my_kernel(input, output):
    # Kernel implementation
    pass
```

**Pattern 2: Multiple Backends**

For kernels with multiple implementation backends (e.g., CUTLASS, cuDNN):

```python
@supported_compute_capability([80, 86, 89, 90])
def _cutlass_check(q, k, v, backend):
    """CUTLASS backend: Ampere through Hopper."""
    if q.shape[-1] > 256:
        raise ValueError("CUTLASS: head_dim must be <= 256")
    return True

@supported_compute_capability([75, 80, 86, 89, 90, 100])
def _cudnn_check(q, k, v, backend):
    """cuDNN backend: Turing through Blackwell."""
    return True

@backend_requirement(
    backend_checks={
        "cutlass": _cutlass_check,
        "cudnn": _cudnn_check,
    },
    common_check=None,  # Optional: shared validation for all backends
)
def attention(q, k, v, backend="cutlass"):
    if backend == "cutlass":
        # CUTLASS implementation
        pass
    elif backend == "cudnn":
        # cuDNN implementation
        pass
```

**Pattern 3: Auto Backend Selection**

For kernels that can automatically select the best backend:

```python
def _heuristic_func(suitable_backends, q, k, v, backend):
    """Return backends in order of preference."""
    # Prefer CUTLASS for small head dims, cuDNN for larger
    if q.shape[-1] <= 128:
        preferred = ["cutlass", "cudnn"]
    else:
        preferred = ["cudnn", "cutlass"]
    return [b for b in preferred if b in suitable_backends]

@backend_requirement(
    backend_checks={
        "cutlass": _cutlass_check,
        "cudnn": _cudnn_check,
    },
    common_check=_common_validation,
    heuristic_func=_heuristic_func,  # Required when backend="auto" is used
)
def attention(q, k, v, backend="auto"):
    if backend == "auto":
        # Use the first backend from suitable_auto_backends
        backend = attention.suitable_auto_backends[0]
    # ... rest of implementation
```

#### Features Added by `@backend_requirement`

The decorator adds these methods to the wrapped function:

```python
# Check if a backend is supported (optionally for a specific CC)
scale.is_backend_supported("cutlass")           # True/False
scale.is_backend_supported("cutlass", cc=90)    # True/False for Hopper

# Check if any backend supports this compute capability
scale.is_compute_capability_supported(90)       # True/False

# Check if a backend exists
scale.has_backend("cutlass")                    # True/False

# Check if there are multiple backend choices
scale.has_backend_choices()                     # True/False
```

#### `skip_check` Keyword Argument

The decorator adds a `skip_check` keyword argument to bypass validation for performance-critical code paths:

```python
# Normal call with validation
result = scale(x, 2.0)

# Skip validation for performance (use with caution!)
result = scale(x, 2.0, skip_check=True)
```

#### Check Function Requirements

Check functions must:
1. Accept the same arguments as the decorated function
2. Return `True` if validation passes
3. Raise `ValueError` with descriptive message if validation fails
4. Be decorated with `@supported_compute_capability` to specify supported architectures

## Step 6: Write Tests in `tests/`

Create tests in an appropriate subdirectory (e.g., `tests/elementwise/test_scale.py` or create a new subdir if needed):

```python
import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("size", [128, 1024, 4096])
def test_scale_correctness(dtype, size):
    """Test scale operation correctness."""
    import flashinfer

    # Setup
    x = torch.randn(size, dtype=dtype, device="cuda")
    factor = 3.14

    # Run FlashInfer kernel
    y = flashinfer.scale(x, factor)

    # Reference implementation
    expected = x * factor

    # Compare
    if dtype == torch.float32:
        rtol, atol = 1e-5, 1e-6
    else:
        rtol, atol = 1e-3, 1e-3

    torch.testing.assert_close(y, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scale_inplace(dtype):
    """Test scale with pre-allocated output."""
    import flashinfer

    x = torch.randn(1024, dtype=dtype, device="cuda")
    out = torch.empty_like(x)
    factor = 2.0

    result = flashinfer.scale(x, factor, out=out)

    # Should return the same tensor
    assert result is out

    # Check correctness
    expected = x * factor
    torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)


def test_scale_cpu_error():
    """Test that CPU tensors raise an error."""
    import flashinfer

    x = torch.randn(128, dtype=torch.float32)

    with pytest.raises(ValueError, match="CUDA"):
        flashinfer.scale(x, 2.0)
```

**Key points:**

- Use `pytest.mark.parametrize` for multiple configurations
- Compare against reference implementation
- Set appropriate tolerances for each dtype
- Test error cases

## Step 7: Register in AOT

Register your kernel in AOT so users with `flashinfer-jit-cache` can skip JIT compilation.

Edit `flashinfer/aot.py`, add to the appropriate section:

```python
def gen_scale_modules() -> Iterator[JitSpec]:
    """Generate scale operation modules for AOT compilation."""
    from .jit.scale import gen_scale_module

    # Pre-compile common dtypes
    for dtype in ["float16", "bfloat16", "float32"]:
        yield gen_scale_module(dtype, dtype)


# In the main AOT build loop, add:
# for spec in gen_scale_modules():
#     spec.build()
```

**Key points:**

- Pre-compile common configurations
- Users with `flashinfer-jit-cache` won't need to compile at runtime

## Step 8: Export API

Edit `flashinfer/__init__.py`:

```python
from .scale import scale as scale

# Or in the existing imports section:
# from .scale import scale
```

## Step 9: Run and Test

```bash
# The kernel compiles automatically on first use
pytest tests/test_scale.py -v

# Run with different dtypes
pytest tests/test_scale.py::test_scale_correctness[float16-128] -v
```

## Step 10: Add Benchmark

**All new kernels should have benchmarks.** This helps track performance regressions and allows users to compare against other implementations.

Create a benchmark file in `benchmarks/` (e.g., `benchmarks/bench_scale.py`):

```python
import torch
from flashinfer.testing import bench_gpu_time

def bench_scale():
    """Benchmark scale kernel."""
    import flashinfer

    sizes = [1024, 4096, 16384, 65536, 262144]
    dtypes = [torch.float16, torch.bfloat16]

    print("Scale Kernel Benchmark")
    print("-" * 60)
    print(f"{'Size':>10} {'Dtype':>10} {'Time (us)':>12} {'Std (us)':>10}")
    print("-" * 60)

    for size in sizes:
        for dtype in dtypes:
            x = torch.randn(size, dtype=dtype, device="cuda")

            # Benchmark with CUPTI (auto-fallback to CUDA events)
            median_time, std_time = bench_gpu_time(
                flashinfer.scale,
                args=(x, 2.0),
                enable_cupti=True,
                dry_run_iters=10,
                repeat_iters=100,
            )

            print(f"{size:>10} {str(dtype):>10} {median_time*1e6:>12.2f} {std_time*1e6:>10.2f}")

if __name__ == "__main__":
    bench_scale()
```

**For more complex kernels**, consider:

- Adding comparisons against reference implementations (e.g., PyTorch native, cuBLAS, cuDNN)
- Using the unified benchmarking framework in `benchmarks/flashinfer_benchmark.py` if applicable
- Testing across different problem sizes and configurations

→ **For complete benchmarking guide, see [`.claude/skills/benchmark-kernel/SKILL.md`](../benchmark-kernel/SKILL.md)**

## Summary of Files Created/Modified

```
include/flashinfer/scale.cuh              # NEW: CUDA kernel definition
csrc/scale.cu                              # NEW: PyTorch launcher
csrc/scale_jit_binding.cu                  # NEW: TVM-FFI binding
flashinfer/jit/scale.py                    # NEW: JIT generator
flashinfer/scale.py                        # NEW: Python API
flashinfer/__init__.py                     # MODIFIED: Export API
flashinfer/aot.py                          # MODIFIED: Register AOT
tests/test_scale.py                        # NEW: Unit tests
benchmarks/bench_scale.py                  # NEW: Benchmark script
```
