---
name: add-cubin-kernel
description: Guide for integrating pre-compiled cubin kernels into FlashInfer via TVM-FFI
---

# Tutorial: Adding a Pre-compiled Cubin Kernel to FlashInfer

This tutorial covers integrating pre-compiled CUDA binary (cubin) kernels into FlashInfer. Cubin kernels are typically provided by NVIDIA (e.g., TRT-LLM attention, GEMM, cuDNN SDPA) as binary blobs that are downloaded at runtime and loaded via the CUDA driver API.

## Architecture Overview

```
┌─────────────────────────────────┐
│  Python: flashinfer.get_cubin   │  TVM-FFI registered function
│  (download, cache, SHA256)      │  flashinfer/jit/cubin_loader.py
├─────────────────────────────────┤
│  C++: flashinfer::cubin_loader  │  include/flashinfer/cubin_loader.h
│  getCubinModule / getCubinKernel│
├─────────────────────────────────┤
│  tvm::ffi::CubinModule          │  RAII module from tvm-ffi
│  tvm::ffi::CubinKernel          │  Kernel handle with launch support
├─────────────────────────────────┤
│  cuLaunchKernelEx               │  CUDA driver API for launch
│  (cluster dims, PDL, etc.)      │  with CUlaunchConfig
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

Two levels of API:

**Preferred: CubinModule (RAII, cached)**
```cpp
#include <flashinfer/cubin_loader.h>

// Get a CubinKernel — module is cached in CubinModuleCache singleton
auto kernel = flashinfer::cubin_loader::getCubinKernel(
    cubin_path,     // e.g., "artifacts/v1/my_kernel.cubin"
    sha256,         // hex string for verification
    kernel_name,    // function name inside the cubin
    smem_bytes      // optional: sets max dynamic shared memory (>= 48KB)
);

// Launch with basic grid/block/stream
kernel.Launch(args, grid, block, stream, dyn_smem_bytes);

// Or get the raw handle for cuLaunchKernelEx
CUfunction func = reinterpret_cast<CUfunction>(kernel.GetHandle());
cuLaunchKernelEx(&launch_config, func, kernel_params, nullptr);
```

**Legacy: Raw bytes (for code using CUmodule/CUfunction directly)**
```cpp
// Get raw cubin bytes — for code that needs cuModuleLoadData
std::string cubin = flashinfer::getCubin(cubin_path, sha256);
CUmodule hmod;
cuModuleLoadData(&hmod, cubin.data());
CUfunction func;
cuModuleGetFunction(&func, hmod, kernel_name);
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
    cudaStream_t stream) {

  // Load kernel via CubinModule (cached, RAII)
  auto kernel = cubin_loader::getCubinKernel(
      cubin_path, sha256, "my_kernel_func", /*smem_bytes=*/0);

  // Prepare args
  void* args[] = {&input, &output, &n};
  tvm::ffi::dim3 grid((n + 255) / 256);
  tvm::ffi::dim3 block(256);

  // Launch
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(
      kernel.Launch(args, grid, block,
                    static_cast<tvm::ffi::cuda_api::StreamHandle>(stream)));
}

// Export via TVM-FFI
TVM_FFI_DLL_EXPORT_TYPED_FUNC(my_kernel, [](
    TensorView input, TensorView output) {
  auto stream = reinterpret_cast<cudaStream_t>(
      TVMFFIEnvGetStream(kDLCUDA, input.device().device_id));
  my_kernel_launcher(
      input.data_ptr(), output.data_ptr(), input.size(0),
      "my_kernels/v1/my_kernel.cubin", "sha256...", stream);
});

}  // namespace flashinfer
```

### Step 3: For Kernels Needing cuLaunchKernelEx (Clusters, PDL)

When kernels need cluster dimensions or other launch attributes:

```cpp
// Get kernel handle
auto cubinKernel = cubin_loader::getCubinKernel(
    cubin_path, sha256, kernel_name, smem_bytes);

// Set up launch config with cluster dimensions
CUlaunchConfig launch_config;
launch_config.blockDimX = threads_per_cta;
launch_config.blockDimY = 1;
launch_config.blockDimZ = 1;
launch_config.gridDimX = num_blocks_x;
launch_config.gridDimY = num_blocks_y;
launch_config.gridDimZ = 1;
launch_config.hStream = stream;
launch_config.sharedMemBytes = smem_bytes;

CUlaunchAttribute attrs[3];
attrs[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
attrs[0].value.clusterDim = {cluster_x, 1, 1};
attrs[1].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
attrs[1].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
attrs[2].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
attrs[2].value.programmaticStreamSerializationAllowed = enable_pdl;
launch_config.attrs = attrs;
launch_config.numAttrs = 3;

// Cast CUkernel to CUfunction for cuLaunchKernelEx
void* kernel_params[] = {&params};
cuLaunchKernelEx(&launch_config,
                 reinterpret_cast<CUfunction>(cubinKernel.GetHandle()),
                 kernel_params, nullptr);
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

## CubinModule vs CUmodule: When to Use Which

| API | Use When | Example Files |
|-----|----------|---------------|
| `getCubinKernel` (CubinModule) | **Default** — new code, simple Launch or cuLaunchKernelEx | `fmhaKernels.cuh`, `cudnn_sdpa_kernel_launcher.cu` |
| `getCubin` (raw bytes + CUmodule) | Upstream code calls `cuFuncSetAttribute` which needs `CUfunction` | `GemmInterface.h` (TRT-LLM vendored code) |

The `CUkernel` handle from `CubinModule` works with:
- `cuLaunchKernelEx` (via `reinterpret_cast<CUfunction>`)
- `cuKernelSetAttribute` (native `CUkernel` API)
- `cuOccupancyMaxActiveClusters`

But does **NOT** work with:
- `cuFuncSetAttribute` (needs real `CUfunction` from `cuModuleGetFunction`)

If upstream vendored code uses `cuFuncSetAttribute`, use the legacy `getCubin` path until the upstream switches to `cuKernelSetAttribute`.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `FLASHINFER_CUBINS_REPOSITORY` | Override cubin download URL |
| `FLASHINFER_CUBIN_DIR` | Override local cubin cache directory |
| `FLASHINFER_CUBIN_CHECKSUM_DISABLED` | Skip SHA256 verification |

## Reference Examples

| Pattern | File |
|---------|------|
| CubinModule + cuLaunchKernelEx | `include/flashinfer/trtllm/fmha/fmhaKernels.cuh` |
| CubinModule + simple launch | `csrc/cudnn_sdpa_kernel_launcher.cu` |
| Legacy CUmodule (upstream constraint) | `include/flashinfer/trtllm/gemm/trtllmGen_gemm_export/GemmInterface.h` |
| Python cubin download + cache | `flashinfer/jit/cubin_loader.py` |
| Artifact metadata | `flashinfer/artifacts.py` |

## TVM-FFI CubinModule API Reference

From `tvm/ffi/extra/cuda/cubin_launcher.h`:

```cpp
// Load cubin from memory
tvm::ffi::CubinModule module(tvm::ffi::Bytes(data, size));

// Get kernel by name
tvm::ffi::CubinKernel kernel = module.GetKernel("func_name");

// Get kernel with shared memory > 48KB
tvm::ffi::CubinKernel kernel = module.GetKernelWithMaxDynamicSharedMemory("func_name", smem);

// Launch
kernel.Launch(args, grid, block, stream, dyn_smem_bytes);

// Get raw handle for cuLaunchKernelEx
cuda_api::KernelHandle handle = kernel.GetHandle();
```

Documentation: <https://tvm.apache.org/ffi/guides/cubin_launcher.html>
