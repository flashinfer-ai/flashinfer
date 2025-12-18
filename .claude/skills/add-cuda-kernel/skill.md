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
#include <torch/extension.h>
#include "flashinfer/scale.cuh"

using namespace flashinfer;

// Type dispatcher helper
#define DISPATCH_DTYPE(dtype, DType, ...)     \
  if (dtype == torch::kFloat16) {              \
    using DType = half;                        \
    __VA_ARGS__                                \
  } else if (dtype == torch::kBFloat16) {      \
    using DType = __nv_bfloat16;               \
    __VA_ARGS__                                \
  } else if (dtype == torch::kFloat32) {       \
    using DType = float;                       \
    __VA_ARGS__                                \
  }

void scale_launcher(torch::Tensor input, torch::Tensor output,
                    float factor) {
  int n = input.numel();
  auto stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_DTYPE(input.dtype(), DType, {
    ScaleLauncher<DType>(
      input.data_ptr<DType>(),
      output.data_ptr<DType>(),
      static_cast<DType>(factor),
      n,
      stream
    );
  });
}
```

**Key points:**

- Includes Torch headers (only allowed in `csrc/`)
- Converts torch::Tensor to raw pointers
- Dispatches on dtype
- Gets CUDA stream from PyTorch

## Step 3: Create TVM-FFI Binding in `csrc/`

Create `csrc/scale_jit_binding.cu`:

```cpp
#include "scale.cu"
#include "tvm_ffi_utils.h"

// Forward declaration
void scale_run(TensorView input, TensorView output, double factor);

// Implementation
void scale_run(TensorView input, TensorView output, double factor) {
  // Convert TensorView to torch::Tensor
  auto input_tensor = input.get_torch_tensor();
  auto output_tensor = output.get_torch_tensor();

  // Validate inputs using TVM-FFI error handling
  if (!input_tensor.is_cuda()) {
    TVM_FFI_THROW(ValueError) << "Input must be a CUDA tensor";
  }
  if (!output_tensor.is_cuda()) {
    TVM_FFI_THROW(ValueError) << "Output must be a CUDA tensor";
  }
  if (input_tensor.numel() != output_tensor.numel()) {
    TVM_FFI_THROW(ValueError) << "Input and output must have the same number of elements, "
                              << "got input size " << input_tensor.numel()
                              << " and output size " << output_tensor.numel();
  }
  if (input_tensor.dtype() != output_tensor.dtype()) {
    TVM_FFI_THROW(ValueError) << "Input and output must have the same dtype";
  }

  scale_launcher(input_tensor, output_tensor, static_cast<float>(factor));
}

// Export to TVM-FFI
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, scale_run);
```

**Key points:**

- Forward declare the function first
- Implement the function (converts TensorView to torch::Tensor)
- **Use TVM-FFI exceptions**: `TVM_FFI_THROW(ErrorType) << "message"`
- Add descriptive error messages with `<<` operator
- Export using `TVM_FFI_DLL_EXPORT_TYPED_FUNC(name, function)`

**TVM-FFI Error Handling:**

- `TVM_FFI_THROW(ValueError) << "message"` - Throw ValueError with custom message
- `TVM_FFI_THROW(TypeError) << "message"` - Throw TypeError
- Use `<<` to chain multiple values in the error message
- Errors are properly propagated back to Python

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

If your kernel only works on specific GPU architectures (e.g., Hopper SM90+, Blackwell SM100+), specify supported architectures:

```python
from flashinfer.jit.core import gen_jit_spec
from flashinfer.jit import current_compilation_context

def gen_scale_module(dtype_in, dtype_out):
    uri = get_scale_uri(dtype_in, dtype_out)
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    # ... copy sources ...

    # For universal kernels (works on all GPUs)
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=None  # All architectures
    )

    # For Hopper+ only kernels (SM90, SM100, SM110, SM120)
    # nvcc_flags = current_compilation_context.get_nvcc_flags_list(
    #     supported_major_versions=[9, 10, 11, 12]
    # )

    # For Blackwell-only kernels (SM100, SM110)
    # nvcc_flags = current_compilation_context.get_nvcc_flags_list(
    #     supported_major_versions=[10, 11]
    # )

    return gen_jit_spec(
        name=uri,
        sources=sources,
        extra_cuda_cflags=nvcc_flags,  # ← Add architecture flags
    )
```

**Common architecture specifications:**
- `None`: Works on all GPUs (default)
- `[9, 10, 11, 12]`: Hopper and newer (SM90+)
- `[10, 11]`: Blackwell and newer (SM100+)
- `[12]`: Specific architecture only (SM120)

**What happens:**
- ✅ If user's GPU is supported → Kernel compiles and runs
- ❌ If user's GPU is not supported → `RuntimeError: No supported CUDA architectures found`

**Testing tip:** Add architecture skip in tests (see Step 6 below).

## Step 5: Create Python API in `flashinfer/`

Create `flashinfer/scale.py`:

```python
import functools
import torch
from typing import Optional

from .jit.scale import gen_scale_module


@functools.cache
def get_scale_module(dtype_in, dtype_out):
    """Get or compile scale module (cached)."""
    return gen_scale_module(dtype_in, dtype_out).build_and_load()


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
    # Validate input
    assert input.is_cuda, "Input must be a CUDA tensor"

    # Allocate output if needed
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output shape mismatch"
        assert out.dtype == input.dtype, "Output dtype mismatch"
        assert out.is_cuda, "Output must be a CUDA tensor"

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
- Handles output allocation
- Validates inputs

## Step 6: Write Tests in `tests/`

Create `tests/test_scale.py`:

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

    with pytest.raises(AssertionError, match="CUDA"):
        flashinfer.scale(x, 2.0)
```

**Key points:**

- Use `pytest.mark.parametrize` for multiple configurations
- Compare against reference implementation
- Set appropriate tolerances for each dtype
- Test error cases

## Step 7: Register in AOT

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
```
