#pragma once

#include <cuda_runtime.h>
#include <tvm/ffi/tvm_ffi.h>

namespace decodem {

// Private to the three host translation units. Never thrown after submission.
struct StreamClusterRefusal {};

inline void check(cudaError_t err, const char* site) {
  TVM_FFI_ICHECK_EQ(err, cudaSuccess) << site << ": " << cudaGetErrorString(err);
}

template <class Kernel>
void admit(const cudaLaunchConfig_t& cfg, Kernel kernel) {
  // Some specializations use __cluster_dims__, HMMA supplies the dimensions
  // in cfg, and ordinary single-CTA kernels require no cluster admission.
  bool dynamic_cluster = false;
  for (unsigned i = 0; i < cfg.numAttrs; ++i) {
    dynamic_cluster |= cfg.attrs[i].id == cudaLaunchAttributeClusterDimension;
  }
  if (!dynamic_cluster) {
    cudaFuncAttributes attr{};
    check(cudaFuncGetAttributes(&attr, kernel), "decodem function attributes");
    if (attr.requiredClusterWidth <= 1 && attr.requiredClusterHeight <= 1 &&
        attr.requiredClusterDepth <= 1)
      return;
  }

  int active = -1;
  cudaError_t err = cudaOccupancyMaxActiveClusters(&active, kernel, &cfg);
  if (err == cudaErrorInvalidClusterSize) {
    // This handled runtime error must not leak into the subsequent stock call.
    cudaError_t pending = cudaGetLastError();
    if (pending != err) check(pending, "decodem pending CUDA error");
    throw StreamClusterRefusal{};
  }
  check(err, "decodem cluster occupancy");
  if (active == 0) throw StreamClusterRefusal{};
  TVM_FFI_ICHECK_GT(active, 0);
}

template <class Kernel, class... Args>
void submit(dim3 grid, dim3 block, size_t bytes, cudaStream_t stream, Kernel kernel, Args... args) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = bytes;
  cfg.stream = stream;
  admit(cfg, kernel);
  check(cudaLaunchKernelEx(&cfg, kernel, args...), "decodem launch");
}

}  // namespace decodem
