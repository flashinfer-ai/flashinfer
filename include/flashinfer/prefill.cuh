#ifndef FLASHINFER_PREFILL_CUH_
#define FLASHINFER_PREFILL_CUH_
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cuda/pipeline>
#include <iostream>
#include <random>

#include "rope.cuh"
#include "state.cuh"
#include "vec_dtypes.cuh"
#include "mma.cuh"

namespace flashinfer {

__global__ void SinglePrefillWithKVCacheKVPartitionKernel() {
  
}

__global__ void SinglePrefillWithKVCacheKernel() {

}

template <typename DTypeIn, typename DTypeOut>
cudaError_t SinglePrefillWithKVCache(DTypeIn *q, DTypeIn *k, DTypeIn *v, DTypeOut *o, float *tmp,
                                     size_t num_heads, size_t q_len, size_t kv_len, size_t head_dim,
                                     RotaryMode rotary_mode = RotaryMode::kNone,
                                     float rope_scale = 1.f, float rope_theta = 1e4,
                                     cudaStream_t stream = nullptr) {
  
  return cudaSuccess;
}

} // namespace flashinfer

#endif // FLASHINFER_PREFILL_CUH_