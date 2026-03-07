---
name: add-cute-dsl-kernel
description: Step-by-step tutorial for adding CuTe DSL kernels to FlashInfer with TVM-FFI AOT caching
---

# Tutorial: Adding a CuTe DSL Kernel to FlashInfer

This tutorial walks through adding a CuTe DSL kernel with `--enable-tvm-ffi` and AOT compilation caching. CuTe DSL kernels are written in Python using NVIDIA's CuTe DSL (`cutlass.cute`) and compiled to GPU code at runtime or ahead of time.

## Prerequisites

- `nvidia-cutlass-dsl` package installed (`pip install nvidia-cutlass-dsl`)
- `apache-tvm-ffi` package installed
- Understanding of CuTe DSL basics: <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html>

## Architecture Overview

```
┌─────────────────────────────┐
│  flashinfer/cute_dsl/       │  Python kernel + compile wrapper
│  my_kernel.py               │
├─────────────────────────────┤
│  flashinfer/jit/cute_dsl.py │  AOT compile/cache/load infrastructure
├─────────────────────────────┤
│  cute.compile() with        │  CuTe DSL compiler (nvidia-cutlass-dsl)
│  --enable-tvm-ffi           │
├─────────────────────────────┤
│  .so file in                │  Cached compiled kernel
│  ~/.cache/flashinfer/       │
│   or flashinfer/data/aot/   │
└─────────────────────────────┘
```

## Step 1: Write the CuTe DSL Kernel

Create `flashinfer/cute_dsl/my_kernel.py`:

```python
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32


class MyKernel:
    """Example CuTe DSL kernel class."""

    def __init__(self, dtype: cutlass.Numeric, hidden_size: int):
        self.dtype = dtype
        self.hidden_size = hidden_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        y: cute.Tensor,
        M: cutlass.Int32,
        stream: cute.cuda.CUstream,
    ):
        # CuTe DSL kernel body
        # Use cute.arch, cute.make_tensor, cute.copy, etc.
        ...
```

**Key rules:**
- The kernel class must have a `@cute.jit` decorated `__call__` method
- Use CuTe DSL types for parameters (`cute.Tensor`, `cutlass.Int32`, `cute.cuda.CUstream`)
- The kernel body runs as DSL code — not normal Python

## Step 2: Write the Compile + Cache Wrapper

In the same file, add a compilation function that uses `compile_and_cache_cute_dsl_kernel`:

```python
import functools
import torch
from flashinfer.jit.cute_dsl import compile_and_cache_cute_dsl_kernel


@functools.cache
def _get_compiled_kernel(hidden_size: int, is_fp16: bool):
    """Compile kernel with TVM-FFI and AOT caching.

    Parameters that affect code generation go in the function signature
    (they become part of the cache key via @functools.cache).
    """
    cutlass_dtype = cutlass.Float16 if is_fp16 else cutlass.BFloat16

    kernel_obj = MyKernel(dtype=cutlass_dtype, hidden_size=hidden_size)

    # Create FAKE tensors for compile-time type inference.
    # Use symbolic sizes for dimensions that vary at runtime.
    sym_m = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype, (sym_m, hidden_size),
        stride_order=(1, 0),  # row-major
        assumed_align=128,
    )
    y_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype, (sym_m, hidden_size),
        stride_order=(1, 0),
        assumed_align=128,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # Unique name for AOT caching — must encode ALL compile-time parameters
    dtype_str = "fp16" if is_fp16 else "bf16"
    aot_func_name = f"my_kernel_{dtype_str}_h{hidden_size}"

    def _do_compile():
        return cute.compile(
            kernel_obj,
            x_fake,
            y_fake,
            Int32(1),       # Dummy scalar (runtime value)
            stream_fake,
            options="--enable-tvm-ffi",
        )

    compiled_kernel = compile_and_cache_cute_dsl_kernel(_do_compile, aot_func_name)

    # Return a closure that accepts torch tensors directly
    def tensor_api(x: torch.Tensor, y: torch.Tensor, M: int):
        compiled_kernel(x, y, Int32(M))

    return tensor_api
```

**Critical patterns:**

1. **`--enable-tvm-ffi`** in `cute.compile()` options — required for AOT export and torch tensor passing
2. **`compile_and_cache_cute_dsl_kernel(compile_fn, func_name)`** — checks cache first, compiles + exports `.so` on miss
3. **`make_fake_compact_tensor`** with correct CUTLASS dtype — NOT `from_dlpack`. Fake tensors provide type info for compilation without needing real data
4. **`make_fake_stream(use_tvm_ffi_env_stream=True)`** — uses the current CUDA stream at runtime
5. **`cute.sym_int()`** for dynamic dimensions — the compiled kernel handles any size
6. **`aot_func_name`** must be unique across ALL kernel variants — encode every compile-time parameter

## Step 3: Write the Public API

```python
from flashinfer.api_logging import flashinfer_api


@flashinfer_api
def my_operation(x: torch.Tensor, y: torch.Tensor) -> None:
    """Public API that users call."""
    M = x.shape[0]
    is_fp16 = x.dtype == torch.float16
    hidden_size = x.shape[1]

    kernel = _get_compiled_kernel(hidden_size, is_fp16)
    kernel(x, y, M)
```

## Step 4: Export in `__init__.py`

Add to `flashinfer/cute_dsl/__init__.py`:

```python
if is_cute_dsl_available():
    from .my_kernel import my_operation
```

## How AOT Caching Works

The `compile_and_cache_cute_dsl_kernel` function:

1. **Checks AOT directory** (`flashinfer/data/aot/cute_dsl/{func_name}.so`) — for pre-compiled packages
2. **Checks JIT cache** (`~/.cache/flashinfer/<version>/<arch>/cached_ops/cute_dsl/{func_name}.so`) — for locally cached compilations
3. **If not found**: calls `_do_compile()`, exports via `compiled_fn.export_to_c()`, links into `.so`, caches it
4. **Loads `.so`** via `cute.runtime.load_module(path, enable_tvm_ffi=True)`

The `.so` persists across runs — subsequent invocations skip compilation entirely.

## Fake Tensors vs from_dlpack

| Approach | Use for | Example |
|----------|---------|---------|
| `make_fake_compact_tensor` | **Compilation** (preferred) | `cute.runtime.make_fake_compact_tensor(cutlass.BFloat16, (sym_m, H))` |
| `from_dlpack` | Legacy (avoid for new code) | `from_dlpack(torch_tensor, assumed_align=16)` |

Fake tensors are preferred because:
- They don't need real GPU memory
- They use symbolic sizes for dynamic shapes
- They specify CUTLASS dtypes directly (e.g., `Float4E2M1FN` for FP4)
- They decouple compilation from runtime tensor creation

## Common Patterns

### FP4 Tensors (stored as uint8 in PyTorch)

```python
# PyTorch stores FP4 as uint8, but the kernel needs Float4E2M1FN
# With fake tensors, specify the correct CUTLASS dtype:
a_fake = cute.runtime.make_fake_compact_tensor(
    cutlass.Float4E2M1FN, (sym_m, sym_k, 1),
    stride_order=(2, 1, 0), assumed_align=16
)
```

### Optional Tensor Arguments

```python
# For kernels with optional tensors, create separate compiled variants:
aot_func_name = f"kernel_{'with_bias' if use_bias else 'no_bias'}"

def _do_compile():
    return cute.compile(kernel, x_fake, bias_fake if use_bias else None, ...)
```

### Kernel with Pointer Arguments (TRT-LLM style)

Some kernels use `cute.Pointer` instead of `cute.Tensor`. Use `make_ptr` for compilation:

```python
from flashinfer.cute_dsl.utils import make_ptr

# For compile-time type hints (null pointer OK)
a_ptr = make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=16)

# At runtime, pass torch tensors — TVM-FFI converts automatically
compiled_kernel(a_tensor, b_tensor, ...)
```

## Reference Examples

| Complexity | File | Pattern |
|-----------|------|---------|
| Simple | `flashinfer/cute_dsl/rmsnorm_fp4quant.py` | Fake tensors, `@functools.cache`, scalar args |
| Moderate | `flashinfer/gemm/gemm_base.py` (MM FP4 section) | Fake tensors, explicit cache dict, multiple variants |
| Complex | `flashinfer/fused_moe/cute_dsl/blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py` | Pointer args, tactic-based caching |

## Testing

```bash
# Run your kernel test
pytest tests/my_test.py -x -v

# Clear CuTe DSL cache to force recompilation
rm -rf ~/.cache/flashinfer/*/cached_ops/cute_dsl/

# Enable verbose logging
export FLASHINFER_LOGGING_LEVEL=DEBUG
```
