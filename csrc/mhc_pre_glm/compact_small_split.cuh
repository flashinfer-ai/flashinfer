// TVM-FFI binding tail for the `persistent_compact` mHC PRE candidate. The
// framework-independent device code lives in
// include/flashinfer/mhc/mhc_pre_glm/compact_small_split.cuh; this file keeps the
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
#include <flashinfer/mhc/mhc_pre_glm/compact_small_split.cuh>

namespace persistent_compact {

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
  static std::map<int, std::pair<int, int>> maxclu_by_dev;
  int dev_ = 0;
  cudaGetDevice(&dev_);
  int maxclu = 0;
  int maxclu_base = 0;
  const size_t sm_split = smem_bytes<true>();
  const size_t sm_base = smem_bytes<false>();
  const size_t sm_t2 = smem_bytes<true, 2>();
  const size_t sm_t4 = smem_bytes<true, 4>();
  const size_t sm_t8 = smem_bytes<true, 8>();
  const size_t sm_t2_push = smem_bytes<true, 2, true>();
  const size_t sm_t4_push = smem_bytes<true, 4, true>();
  const size_t sm_t8_push = smem_bytes<true, 8, true>();
  const size_t sm_t16_push = smem_bytes<true, 16, true>();
  {
    std::lock_guard<std::mutex> cfg_lk(cfg_mu);
    auto cfg_it = maxclu_by_dev.find(dev_);
    if (cfg_it != maxclu_by_dev.end()) {
      maxclu = cfg_it->second.first;
      maxclu_base = cfg_it->second.second;
    } else {
      auto set_attr = [](const void* f, cudaFuncAttribute a, int v) {
        if (cudaFuncSetAttribute(f, a, v) != cudaSuccess) {
          throw std::runtime_error("mhc_pre attribute opt-in failed (compact arm)");
        }
      };
      set_attr((const void*)mhc_kernel<true>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      set_attr((const void*)mhc_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
               (int)sm_split);
      set_attr((const void*)mhc_kernel<false>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      set_attr((const void*)mhc_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
               (int)sm_base);
      set_attr((const void*)mhc_kernel<true, 2>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      set_attr((const void*)mhc_kernel<true, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
               (int)sm_t2);
      set_attr((const void*)mhc_kernel<true, 2, 0, true>,
               cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      set_attr((const void*)mhc_kernel<true, 2, 0, true>,
               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm_t2_push);
      set_attr((const void*)mhc_kernel<true, 2, 1, true>,
               cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      set_attr((const void*)mhc_kernel<true, 2, 1, true>,
               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm_t2_push);
      set_attr((const void*)mhc_kernel<true, 2, 2, true>,
               cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      set_attr((const void*)mhc_kernel<true, 2, 2, true>,
               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm_t2_push);
      set_attr((const void*)mhc_kernel<true, 4>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      set_attr((const void*)mhc_kernel<true, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize,
               (int)sm_t4);
      set_attr((const void*)mhc_kernel<true, 4, 0, true>,
               cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      set_attr((const void*)mhc_kernel<true, 4, 0, true>,
               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm_t4_push);
      set_attr((const void*)mhc_kernel<true, 2, 4, true>,
               cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      set_attr((const void*)mhc_kernel<true, 2, 4, true>,
               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm_t2_push);
      set_attr((const void*)mhc_kernel<true, 4, 8, true>,
               cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      set_attr((const void*)mhc_kernel<true, 4, 8, true>,
               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm_t4_push);
      set_attr((const void*)mhc_kernel<true, 4, 4, true>,
               cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      set_attr((const void*)mhc_kernel<true, 4, 4, true>,
               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm_t4_push);
      set_attr((const void*)mhc_kernel<true, 8>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      set_attr((const void*)mhc_kernel<true, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize,
               (int)sm_t8);
      set_attr((const void*)mhc_kernel<true, 8, 0, true>,
               cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      set_attr((const void*)mhc_kernel<true, 8, 0, true>,
               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm_t8_push);
      set_attr((const void*)mhc_kernel<true, 8, 8>, cudaFuncAttributeNonPortableClusterSizeAllowed,
               1);
      set_attr((const void*)mhc_kernel<true, 8, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize,
               (int)sm_t8);
      set_attr((const void*)mhc_kernel<true, 8, 8, true>,
               cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      set_attr((const void*)mhc_kernel<true, 8, 8, true>,
               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm_t8_push);
      set_attr((const void*)mhc_kernel<true, 8, 16>, cudaFuncAttributeNonPortableClusterSizeAllowed,
               1);
      set_attr((const void*)mhc_kernel<true, 8, 16>, cudaFuncAttributeMaxDynamicSharedMemorySize,
               (int)sm_t8);
      set_attr((const void*)mhc_kernel<true, 8, 16, true>,
               cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      set_attr((const void*)mhc_kernel<true, 8, 16, true>,
               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm_t8_push);
      set_attr((const void*)mhc_kernel<true, 16, 32, true>,
               cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      set_attr((const void*)mhc_kernel<true, 16, 32, true>,
               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm_split);
      set_attr((const void*)mhc_kernel<true, 16, 0, true>,
               cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      set_attr((const void*)mhc_kernel<true, 16, 0, true>,
               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm_t16_push);
      set_attr((const void*)mhc_kernel<true, 16, 64>,
               cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      set_attr((const void*)mhc_kernel<true, 16, 64>, cudaFuncAttributeMaxDynamicSharedMemorySize,
               (int)sm_split);
      set_attr((const void*)mhc_kernel<true, 16, 64, true>,
               cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      set_attr((const void*)mhc_kernel<true, 16, 64, true>,
               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm_t16_push);
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
      if (cudaOccupancyMaxActiveClusters(&n, (const void*)mhc_kernel<true>, &cfg) != cudaSuccess ||
          n <= 0) {
        n = 8;
      }
      maxclu = n > NCMAX ? NCMAX : n;
      cudaLaunchConfig_t cfgb = cfg;
      cfgb.dynamicSmemBytes = sm_base;
      int nb = 0;
      if (cudaOccupancyMaxActiveClusters(&nb, (const void*)mhc_kernel<false>, &cfgb) !=
              cudaSuccess ||
          nb <= 0) {
        nb = 8;
      }
      maxclu_base = nb > NCBMAX ? NCBMAX : nb;
      maxclu_by_dev[dev_] = std::make_pair(maxclu, maxclu_base);
    }
  }  // cfg_lk released before launch

  const int ntiles = (T + TT - 1) / TT;
  const int split = (2 * ntiles <= maxclu) ? 1 : 0;
  const int NC = split ? 2 * ntiles : (ntiles < maxclu_base ? ntiles : maxclu_base);
  dim3 grid(CLU, NC, 1);
  const __nv_bfloat16* res = static_cast<const __nv_bfloat16*>(residual.data_ptr());
  const float* fnp = static_cast<const float*>(fn.data_ptr());
  const float* scp = static_cast<const float*>(hc_scale.data_ptr());
  const float* bap = static_cast<const float*>(hc_base.data_ptr());
  float* pop = static_cast<float*>(post_mix.data_ptr());
  float* cop = static_cast<float*>(comb.data_ptr());
  __nv_bfloat16* lip = static_cast<__nv_bfloat16*>(layer_input.data_ptr());
  if (T <= 8) {
    dim3 small_grid(CLU, 2, 1);
    if (T == 1) {
      mhc_kernel<true, 2, 1, true>
          <<<small_grid, NTH, sm_t2_push, stream>>>(res, fnp, scp, bap, pop, cop, lip, T, 1, 2);
    } else if (T == 2) {
      mhc_kernel<true, 2, 2, true>
          <<<small_grid, NTH, sm_t2_push, stream>>>(res, fnp, scp, bap, pop, cop, lip, T, 1, 2);
    } else if (T < 4) {
      mhc_kernel<true, 4, 0, true>
          <<<small_grid, NTH, sm_t4_push, stream>>>(res, fnp, scp, bap, pop, cop, lip, T, 1, 2);
    } else if (T == 4) {
      mhc_kernel<true, 2, 4, true><<<dim3(CLU, 4, 1), NTH, sm_t2_push, stream>>>(
          res, fnp, scp, bap, pop, cop, lip, T, 2, 4);
    } else if (T < 8) {
      mhc_kernel<true, 8, 0, true>
          <<<small_grid, NTH, sm_t8_push, stream>>>(res, fnp, scp, bap, pop, cop, lip, T, 1, 2);
    } else {
      // Two 4-token tiles across four cluster groups instead of one 8-token
      // tile across two: a one-tile cluster's tile is its entire critical
      // path, and the fn bytes a CTA pulls do not depend on tile height, so
      // the shorter tile wins despite twice as many clusters fetching fn.
      mhc_kernel<true, 4, 8, true><<<dim3(CLU, 4, 1), NTH, sm_t4_push, stream>>>(
          res, fnp, scp, bap, pop, cop, lip, T, 2, 4);
    }
    return;
  }
  if (T == 16) {
    mhc_kernel<true, 8, 16, true>
        <<<dim3(CLU, 4, 1), NTH, sm_t8_push, stream>>>(res, fnp, scp, bap, pop, cop, lip, T, 2, 4);
    return;
  }
  if (T == 32) {
    mhc_kernel<true, 16, 32, true>
        <<<dim3(CLU, 4, 1), NTH, sm_split, stream>>>(res, fnp, scp, bap, pop, cop, lip, T, 2, 4);
    return;
  }
  if (T < 32) {
    const int small_ntiles = (T + 15) / 16;
    const int small_nc = 2 * small_ntiles;
    mhc_kernel<true, 16, 0, true><<<dim3(CLU, small_nc, 1), NTH, sm_t16_push, stream>>>(
        res, fnp, scp, bap, pop, cop, lip, T, small_ntiles, small_nc);
    return;
  }
  if (T == 64) {
    mhc_kernel<true, 16, 64, true>
        <<<dim3(CLU, 8, 1), NTH, sm_t16_push, stream>>>(res, fnp, scp, bap, pop, cop, lip, T, 4, 8);
    return;
  }
  if (split) {
    mhc_kernel<true>
        <<<grid, NTH, sm_split, stream>>>(res, fnp, scp, bap, pop, cop, lip, T, ntiles, NC);
  } else {
    mhc_kernel<false>
        <<<grid, NTH, sm_base, stream>>>(res, fnp, scp, bap, pop, cop, lip, T, ntiles, NC);
  }
}

}  // namespace persistent_compact
