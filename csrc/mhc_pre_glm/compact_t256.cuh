// TVM-FFI binding tail for the `persistent_compact_256` mHC PRE candidate. The
// framework-independent device code lives in
// include/flashinfer/mhc/mhc_pre_glm/compact_t256.cuh; this file keeps the
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
#include <flashinfer/mhc/mhc_pre_glm/compact_t256.cuh>

namespace persistent_compact_256 {

namespace ffi = tvm::ffi;

void Run(ffi::TensorView residual, ffi::TensorView fn, ffi::TensorView hc_scale,
         ffi::TensorView hc_base, ffi::TensorView post_mix, ffi::TensorView comb,
         ffi::TensorView layer_input) {
  const int T = static_cast<int>(residual.shape()[0]);
  if (T <= 0) return;
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, residual.device().device_id));

  // Per-device, publish-after-success one-time configuration (see split arm).
  static std::mutex cfg_mu;
  static std::map<int, int> maxclu_by_dev;
  int dev_ = 0;
  cudaGetDevice(&dev_);
  const size_t sm_split = smem_bytes<true, false>();
  const size_t sm_multi = smem_bytes<true, true>();
  const size_t sm_base = smem_bytes<false, false>();
  int maxclu = 0;
  {
    std::lock_guard<std::mutex> cfg_lk(cfg_mu);
    auto cfg_it = maxclu_by_dev.find(dev_);
    if (cfg_it != maxclu_by_dev.end()) {
      maxclu = cfg_it->second;
    } else {
      if (cudaFuncSetAttribute((const void*)mhc_kernel<true, false>,
                               cudaFuncAttributeNonPortableClusterSizeAllowed, 1) != cudaSuccess ||
          cudaFuncSetAttribute((const void*)mhc_kernel<true, false>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               (int)sm_split) != cudaSuccess ||
          cudaFuncSetAttribute((const void*)mhc_kernel<true, true>,
                               cudaFuncAttributeNonPortableClusterSizeAllowed, 1) != cudaSuccess ||
          cudaFuncSetAttribute((const void*)mhc_kernel<true, true>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               (int)sm_multi) != cudaSuccess ||
          cudaFuncSetAttribute((const void*)mhc_kernel<false, false>,
                               cudaFuncAttributeNonPortableClusterSizeAllowed, 1) != cudaSuccess ||
          cudaFuncSetAttribute((const void*)mhc_kernel<false, false>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               (int)sm_base) != cudaSuccess) {
        throw std::runtime_error("mhc_pre attribute opt-in failed (t256 arm)");
      }
      // The persistent grid must be exactly the number of clusters that fit
      // concurrently: one extra cluster runs as a whole second wave and nearly
      // doubles the kernel.
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = dim3(CLU, 1, 1);
      cfg.blockDim = dim3(NTH, 1, 1);
      cfg.dynamicSmemBytes = sm_split;
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeClusterDimension;
      attr[0].val.clusterDim.x = CLU;
      attr[0].val.clusterDim.y = 1;
      attr[0].val.clusterDim.z = 1;
      cfg.attrs = attr;
      cfg.numAttrs = 1;
      int n = 0;
      if (cudaOccupancyMaxActiveClusters(&n, (const void*)mhc_kernel<true, false>, &cfg) !=
              cudaSuccess ||
          n <= 0) {
        n = 8;
      }
      maxclu = n > NCMAX ? NCMAX : n;
      maxclu_by_dev[dev_] = maxclu;
    }
  }  // cfg_lk released before launch

  const int ntiles = (T + TT - 1) / TT;
  // Always split the fn rows across two cluster groups: each CTA then stages
  // 8 or 16 projection rows once instead of all 24, which is 12 MB of weight
  // traffic for the whole launch instead of 25 MB, independent of T.
  const int half = maxclu >> 1;
  const int nc_g = ntiles < half ? ntiles : half;
  const int NC = 2 * (nc_g > 0 ? nc_g : 1);
  const int multi = (nc_g > 0 && ntiles > nc_g) ? 1 : 0;
  dim3 grid(CLU, NC, 1);
  const __nv_bfloat16* res = static_cast<const __nv_bfloat16*>(residual.data_ptr());
  const float* fnp = static_cast<const float*>(fn.data_ptr());
  const float* scp = static_cast<const float*>(hc_scale.data_ptr());
  const float* bap = static_cast<const float*>(hc_base.data_ptr());
  float* pop = static_cast<float*>(post_mix.data_ptr());
  float* cop = static_cast<float*>(comb.data_ptr());
  __nv_bfloat16* lip = static_cast<__nv_bfloat16*>(layer_input.data_ptr());
  if (multi) {
    mhc_kernel<true, true>
        <<<grid, NTH, sm_multi, stream>>>(res, fnp, scp, bap, pop, cop, lip, T, ntiles, NC);
  } else {
    mhc_kernel<true, false>
        <<<grid, NTH, sm_split, stream>>>(res, fnp, scp, bap, pop, cop, lip, T, ntiles, NC);
  }
}

}  // namespace persistent_compact_256
