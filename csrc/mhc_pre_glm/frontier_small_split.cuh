// TVM-FFI binding tail for the `persistent_split` mHC PRE candidate. The
// framework-independent device code lives in
// include/flashinfer/mhc/mhc_pre_glm/frontier_small_split.cuh; this file keeps the
// TensorView adapter and host-side launch configuration, and is included
// by exactly one TU, csrc/mhc_pre_glm/mhc_pre_glm_sm103.cu.
#pragma once

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cmath>
#include <cstdint>
#include <flashinfer/mhc/mhc_pre_glm/frontier_small_split.cuh>

namespace persistent_split {

namespace ffi = tvm::ffi;

void Run(ffi::TensorView residual, ffi::TensorView fn, ffi::TensorView hc_scale,
         ffi::TensorView hc_base, ffi::TensorView post_mix, ffi::TensorView comb,
         ffi::TensorView layer_input) {
  const int T = static_cast<int>(residual.shape()[0]);
  if (T <= 0) return;
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, residual.device().device_id));

  // Per-device, publish-after-success one-time configuration: the attribute
  // and occupancy results belong to (function, device); the lock is held
  // through the CUDA calls so a concurrent first launch cannot observe a
  // half-configured function, and a device is cached only after success.
  static std::mutex cfg_mu;
  static std::map<int, int> maxclu_by_dev;
  int dev_ = 0;
  cudaGetDevice(&dev_);
  const size_t sm = smem_bytes();
  int maxclu = 0;
  {
    std::lock_guard<std::mutex> cfg_lk(cfg_mu);
    auto cfg_it = maxclu_by_dev.find(dev_);
    if (cfg_it != maxclu_by_dev.end()) {
      maxclu = cfg_it->second;
    } else {
      if (cudaFuncSetAttribute((const void*)mhc_kernel,
                               cudaFuncAttributeNonPortableClusterSizeAllowed, 1) != cudaSuccess ||
          cudaFuncSetAttribute((const void*)mhc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               (int)sm) != cudaSuccess) {
        throw std::runtime_error("mhc_pre attribute opt-in failed (split arm)");
      }
      // The persistent grid must be exactly the number of clusters that fit
      // concurrently: one extra cluster runs as a whole second wave and nearly
      // doubles the kernel.
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = dim3(CLU, 1, 1);
      cfg.blockDim = dim3(NTH, 1, 1);
      cfg.dynamicSmemBytes = sm;
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeClusterDimension;
      attr[0].val.clusterDim.x = CLU;
      attr[0].val.clusterDim.y = 1;
      attr[0].val.clusterDim.z = 1;
      cfg.attrs = attr;
      cfg.numAttrs = 1;
      int n = 0;
      if (cudaOccupancyMaxActiveClusters(&n, (const void*)mhc_kernel, &cfg) != cudaSuccess ||
          n <= 0) {
        n = 8;
      }
      maxclu = n > NCMAX ? NCMAX : n;
      maxclu_by_dev[dev_] = maxclu;
    }
  }  // cfg_lk released before launch

  constexpr int ntiles = 128 / TT;
  const int spare = maxclu - ntiles;
  const int split_pairs = spare > 1 ? 2 : (spare > 0 ? 1 : 0);
  const int NC = ntiles + split_pairs;
  dim3 grid(CLU, NC, 1);
  mhc_kernel<<<grid, NTH, sm, stream>>>(
      static_cast<const __nv_bfloat16*>(residual.data_ptr()),
      static_cast<const float*>(fn.data_ptr()), static_cast<const float*>(hc_scale.data_ptr()),
      static_cast<const float*>(hc_base.data_ptr()), static_cast<float*>(post_mix.data_ptr()),
      static_cast<float*>(comb.data_ptr()), static_cast<__nv_bfloat16*>(layer_input.data_ptr()),
      split_pairs);
}

}  // namespace persistent_split
