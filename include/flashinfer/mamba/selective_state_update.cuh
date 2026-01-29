/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_MAMBA_SELECTIVE_STATE_UPDATE_CUH_
#define FLASHINFER_MAMBA_SELECTIVE_STATE_UPDATE_CUH_

// #include <cooperative_groups.h>
// #include <cooperative_groups/memcpy_async.h>
// #include <cuda_runtime_api.h>

// #include <cmath>
#include <cstdint>
// #include <cuda/barrier>

// #include "../utils.cuh"
// #include "../vec_dtypes.cuh"
// #include "conversion.cuh"
// #include "create_tensor_map.cuh"

namespace flashinfer::mamba {

constexpr unsigned warpSize = 32;

struct SelectiveStateUpdateParams {
  uint32_t batch{}, nheads{}, dim{}, dstate{}, ngroups{}, state_cache_size{};
  int32_t pad_slot_id{-1};

  int64_t x_stride_batch{}, dt_stride_batch{}, B_stride_batch{}, C_stride_batch{},
      out_stride_batch{}, z_stride_batch{}, state_stride_batch{};

  void* __restrict__ state{nullptr};  // state_t: (state_cache_size, nheads, dim, dstate)
  void* __restrict__ x{nullptr};      // input_t: (batch, nheads, dim)
  void* __restrict__ dt{
      nullptr};  // weight_t: (batch, nheads) but pretends to be (batch, nheads, dim)
  void* __restrict__ dt_bias{nullptr};  // weight_t (nheads) but pretends to be (nheads, dim)
  void* __restrict__ A{nullptr};  // matrixA_t: (nheads) but pretends to be (nheads, dim, dstate)
  void* __restrict__ B{nullptr};  // input_t: (batch, ngroups, dstate)
  void* __restrict__ C{nullptr};  // input_t: (batch, ngroups, dstate)
  void* __restrict__ D{nullptr};  // weight_t: (nheads) but pretends to be (nheads, dim)
  void* __restrict__ z{nullptr};  // input_t: (batch, nheads, dim)
  void* __restrict__ output{nullptr};               // input_t: (batch, nheads, dim)
  void* __restrict__ state_batch_indices{nullptr};  // state_batch_indices: (batch,)

  bool dt_softplus{false};
  bool update_state{true};
};

__forceinline__ __device__ float softplus(float x) { return __logf(1.f + __expf(x)); }

__device__ __forceinline__ float thresholded_softplus(float dt_value) {
  constexpr float threshold = 20.f;
  return (dt_value <= threshold) ? softplus(dt_value) : dt_value;
}

// Simple packed vector type for loading N elements of type T
template <typename T, int N = sizeof(float4) / sizeof(T)>
struct alignas(N * sizeof(T)) PackedAligned {
  T val[N];
  static constexpr int count = N;
  using dtype = T;
};

template <class load_t>
__device__ __forceinline__ auto make_zeros() -> load_t {
  load_t ret{};
#pragma unroll
  for (int i = 0; i < ret.count; i++)
    ret.val[i] = typename load_t::dtype{};  // default initialization
  return ret;
};

// Computes the vector load size that ensures full warp utilization.
// Avoids cases like: dstate=64, load_t = sizeof(float4)/sizeof(f16), warpsize=32 (32 * 8 > 64)
// in which case a part of the warp would be idle.
template <typename T, int DSTATE>
inline constexpr auto getVectorLoadSizeForFullUtilization() -> unsigned {
  static_assert(sizeof(float4) >= sizeof(T));
  constexpr unsigned maxHardwareLoadSize = sizeof(float4) / sizeof(T);
  constexpr unsigned maxLogicalLoadSize = (unsigned)DSTATE / warpSize;
  return maxHardwareLoadSize < maxLogicalLoadSize ? maxHardwareLoadSize : maxLogicalLoadSize;
}

__device__ __forceinline__ float warpReduceSum(float val) {
  for (int s = warpSize / 2; s > 0; s /= 2) {
    val += __shfl_down_sync(UINT32_MAX, val, s);
  }
  return val;
}

namespace mtp {
// Extended params struct for multi-token prediction (MTP)
struct SelectiveStateMTPParams : public SelectiveStateUpdateParams {
  uint32_t ntokens_mtp{1};
  uint64_t cache_steps{0};

  // MTP-specific strides for the token dimension
  int64_t x_stride_mtp{}, dt_stride_mtp{}, B_stride_mtp{}, C_stride_mtp{}, out_stride_mtp{},
      z_stride_mtp{};
  void* __restrict__ intermediate_states{
      nullptr};  // state_t: (ntokens_mtp, state_cache_size, nheads, dim, dstate)
  void* __restrict__ intermediate_state_indices{nullptr};  // (batch,)
  int64_t intermediate_state_stride_batch{};  // stride for batch dimension of intermediate_states
};
}  // namespace mtp

}  // namespace flashinfer::mamba

#include "selective_state_update_stp.cuh"

#endif  // FLASHINFER_MAMBA_SELECTIVE_STATE_UPDATE_CUH_
