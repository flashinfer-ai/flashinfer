---
name: add-cubin-kernel
description: Guide for integrating pre-compiled cubin kernels into FlashInfer via TVM-FFI
---

# Tutorial: Adding a Pre-compiled Cubin Kernel to FlashInfer

This tutorial covers integrating pre-compiled CUDA binary (cubin) kernels into FlashInfer. Cubin kernels are typically provided by NVIDIA (e.g., TRT-LLM attention, GEMM, cuDNN SDPA) as binary blobs that are downloaded at runtime and loaded via TVM-FFI's CubinModule.

## Architecture Overview

```
┌─────────────────────────────────┐
│  Python: flashinfer.get_cubin   │  TVM-FFI registered function
│  (download, cache, SHA256)      │  flashinfer/jit/cubin_loader.py
├─────────────────────────────────┤
│  C++: flashinfer::cubin_loader  │  include/flashinfer/cubin_loader.h
│  getCubinModule / getCubinKernel│
├─────────────────────────────────┤
│  tvm::ffi::CubinModule          │  RAII module loading (CUlibrary)
│  tvm::ffi::CubinKernel          │  Kernel handle (CUkernel)
├─────────────────────────────────┤
│  CubinKernel::Launch()          │  Basic launch (grid/block/stream)
│  CubinKernel::LaunchEx()        │  Extended launch (clusters, PDL)
└─────────────────────────────────┘
```

## How Cubin Loading Works

### Python Side (`flashinfer/jit/cubin_loader.py`)

A TVM-FFI global function `flashinfer.get_cubin` is registered at import time:

```python
@tvm_ffi.register_global_func("flashinfer.get_cubin")
def _tvm_ffi_get_cubin(name: str, sha256: str) -> bytes:
    return get_cubin(name, sha256)
```

This function:
1. Checks the local cache directory (`~/.cache/flashinfer/cubins/` or `flashinfer-cubin` package)
2. Downloads from the NVIDIA artifactory if not cached
3. Verifies SHA256 checksum
4. Returns raw bytes

### C++ Side (`include/flashinfer/cubin_loader.h`)

```cpp
#include <flashinfer/cubin_loader.h>

// Get a CubinKernel — module is cached in CubinModuleCache singleton
auto kernel = flashinfer::cubin_loader::getCubinKernel(
    cubin_path,     // e.g., "artifacts/v1/my_kernel.cubin"
    sha256,         // hex string for verification
    kernel_name,    // function name inside the cubin
    smem_bytes      // optional: sets max dynamic shared memory (>= 48KB)
);
```

## Step-by-Step: Adding a New Cubin Kernel

### Step 1: Register Artifact Metadata

Add the cubin artifact path and checksums in `flashinfer/artifacts.py`:

```python
class ArtifactPath:
    # ... existing entries ...
    MY_KERNELS = "my_kernels/v1"

class CheckSumHash:
    # ... existing entries ...
    MY_KERNELS = b"sha256_hash_1 my_kernel_sm100.cubin\nsha256_hash_2 my_kernel_sm90.cubin\n"
```

### Step 2: Write the C++ Launcher

Create `csrc/my_kernel_launcher.cu`:

```cpp
#include <flashinfer/cubin_loader.h>
#include <tvm/ffi/function.h>
#include "tvm_ffi_utils.h"

namespace flashinfer {

void my_kernel_launcher(
    void* input, void* output, int n,
    const std::string& cubin_path, const std::string& sha256,
    tvm::ffi::cuda_api::StreamHandle stream) {

  // Load kernel via CubinModule (cached, RAII)
  auto kernel = cubin_loader::getCubinKernel(
      cubin_path, sha256, "my_kernel_func", /*smem_bytes=*/0);

  // Prepare args
  void* args[] = {&input, &output, &n};
  tvm::ffi::dim3 grid((n + 255) / 256);
  tvm::ffi::dim3 block(256);

  // Launch
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(
      kernel.Launch(args, grid, block, stream));
}

// Export via TVM-FFI
TVM_FFI_DLL_EXPORT_TYPED_FUNC(my_kernel, [](
    TensorView input, TensorView output) {
  auto stream = static_cast<tvm::ffi::cuda_api::StreamHandle>(
      TVMFFIEnvGetStream(kDLCUDA, input.device().device_id));
  my_kernel_launcher(
      input.data_ptr(), output.data_ptr(), input.size(0),
      "my_kernels/v1/my_kernel.cubin", "sha256...", stream);
});

}  // namespace flashinfer
```

### Step 3: For Kernels Needing Extended Launch (Clusters, PDL)

Use `CubinKernel::LaunchEx` with a `cuda_api::LaunchConfig`:

```cpp
auto kernel = cubin_loader::getCubinKernel(
    cubin_path, sha256, kernel_name, smem_bytes);

// Option A: Use ConstructLaunchConfig helper (simple cluster dim only)
tvm::ffi::cuda_api::LaunchConfig config;
tvm::ffi::cuda_api::LaunchAttrType attr;
tvm::ffi::cuda_api::ConstructLaunchConfig(
    kernel.GetHandle(), stream, smem_bytes,
    grid, block, cluster_dim, config, attr);

TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(
    kernel.LaunchEx(args, config));

// Option B: Build full config manually (multiple attributes)
tvm::ffi::cuda_api::LaunchConfig config;
config.gridDimX = grid_x;
config.gridDimY = grid_y;
config.gridDimZ = 1;
config.blockDimX = block_x;
config.blockDimY = 1;
config.blockDimZ = 1;
config.hStream = stream;
config.sharedMemBytes = smem_bytes;

CUlaunchAttribute attrs[3];
attrs[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
attrs[0].value.clusterDim = {cluster_x, 1, 1};
attrs[1].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
attrs[1].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
attrs[2].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
attrs[2].value.programmaticStreamSerializationAllowed = enable_pdl;
config.attrs = attrs;
config.numAttrs = 3;

TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(
    kernel.LaunchEx(args, config));
```

### Step 4: Write the JIT Module Generator

Create `flashinfer/jit/my_kernel.py`:

```python
from .core import gen_jit_spec
from . import env as jit_env

def gen_my_kernel_module():
    """Generate JIT spec for the cubin kernel launcher."""
    import shutil

    uri = compute_uri("my_kernel", ...)
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri

    # Copy launcher source to gen directory
    sources = []
    for fname in ["my_kernel_launcher.cu"]:
        src = jit_env.FLASHINFER_CSRC_DIR / fname
        dst = gen_directory / fname
        shutil.copy(src, dst)
        sources.append(dst)

    return gen_jit_spec(uri, sources, extra_cuda_cflags=[...])
```

### Step 5: Write Python API

```python
import functools
from .jit.my_kernel import gen_my_kernel_module

@functools.cache
def _get_module():
    mod = gen_my_kernel_module()
    return mod.build_and_load()

def my_operation(x, y):
    op = _get_module().my_kernel
    op(x, y)
```

Note: `setup_cubin_loader()` is no longer needed — the TVM-FFI function registration happens at import time.

## CubinKernel API

From `tvm/ffi/extra/cuda/cubin_launcher.h`:

```cpp
// Load cubin from memory
tvm::ffi::CubinModule module(tvm::ffi::Bytes(data, size));

// Get kernel by name
tvm::ffi::CubinKernel kernel = module.GetKernel("func_name");

// Get kernel with shared memory > 48KB
tvm::ffi::CubinKernel kernel = module.GetKernelWithMaxDynamicSharedMemory("func_name", smem);

// Basic launch
kernel.Launch(args, grid, block, stream, dyn_smem_bytes);

// Extended launch with cluster dimensions, PDL, etc.
kernel.LaunchEx(args, config);

// Get raw handle (only needed for upstream code requiring CUfunction)
cuda_api::KernelHandle handle = kernel.GetHandle();
```

## Unified CUDA API (`tvm::ffi::cuda_api`)

TVM-FFI provides a unified API that works with both CUDA Driver API (< 12.8) and Runtime API (>= 12.8):

| Function | Purpose |
|----------|---------|
| `LaunchKernel(kernel, args, grid, block, stream, smem)` | Basic kernel launch |
| `LaunchKernelEx(kernel, args, config)` | Extended launch with attributes |
| `ConstructLaunchConfig(kernel, stream, smem, grid, block, cluster_dim, config, attr)` | Build launch config |
| `SetKernelMaxDynamicSharedMem(kernel, smem, device)` | Set shared memory attribute |
| `GetKernelSharedMem(kernel, out, device)` | Query static shared memory |

| Type Alias | Driver API | Runtime API |
|-----------|-----------|-------------|
| `KernelHandle` | `CUkernel` | `cudaKernel_t` |
| `LibraryHandle` | `CUlibrary` | `cudaLibrary_t` |
| `LaunchConfig` | `CUlaunchConfig` | `cudaLaunchConfig_t` |
| `StreamHandle` | `CUstream` | `cudaStream_t` |
| `ResultType` | `CUresult` | `cudaError_t` |
| `kSuccess` | `CUDA_SUCCESS` | `cudaSuccess` |

## Legacy Code: When CUfunction is Required

Some vendored upstream code (e.g., TRT-LLM `CudaKernelLauncher.h`) calls `cuFuncSetAttribute` which requires `CUfunction`, not `CUkernel`. In these cases, use the legacy `getCubin` path:

```cpp
// Legacy: get raw cubin bytes for cuModuleLoadData
std::string cubin = flashinfer::getCubin(cubin_path, sha256);
CUmodule hmod;
cuModuleLoadData(&hmod, cubin.data());
CUfunction func;
cuModuleGetFunction(&func, hmod, kernel_name);
```

This is only needed when upstream code you can't modify uses `cuFuncSetAttribute`. For new code, always use `CubinModule`/`CubinKernel`.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `FLASHINFER_CUBINS_REPOSITORY` | Override cubin download URL |
| `FLASHINFER_CUBIN_DIR` | Override local cubin cache directory |
| `FLASHINFER_CUBIN_CHECKSUM_DISABLED` | Skip SHA256 verification |

## Reference Examples

| Pattern | File |
|---------|------|
| CubinModule + LaunchEx | `csrc/cudnn_sdpa_kernel_launcher.cu` |
| CubinModule + upstream cuLaunchKernelEx | `include/flashinfer/trtllm/fmha/fmhaKernels.cuh` |
| Legacy CUmodule (upstream constraint) | `include/flashinfer/trtllm/gemm/trtllmGen_gemm_export/GemmInterface.h` |
| Python cubin download + cache | `flashinfer/jit/cubin_loader.py` |
| Artifact metadata | `flashinfer/artifacts.py` |

Documentation: <https://tvm.apache.org/ffi/guides/cubin_launcher.html>
