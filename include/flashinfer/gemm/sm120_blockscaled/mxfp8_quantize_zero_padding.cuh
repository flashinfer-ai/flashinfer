/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "math.cuh"

namespace sm120_blockscaled {

__device__ __forceinline__ float exp2f_rcp(uint8_t exp) {
  constexpr uint32_t FP32_EXPONENT_BIAS = 127;
  return (exp == 0) ? 1.0f : exp2f(FP32_EXPONENT_BIAS - static_cast<float>(exp));
}

__device__ __forceinline__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

// Per-token FP8 E4M3 quantization with UE8M0 scale, configurable K-axis granularity.
//
// Each warp processes 512 K elements (32 lanes * 16 elems/lane). Output:
//   GranK=128: 4 UE8M0 / warp packed into 1 int32 along K
//   GranK=32: 16 UE8M0 / warp packed into 4 int32 along K
// SFA m-axis offsets are 4-row aligned via `math::compute_padded_offset`.
template <int GranK, typename InputType, typename OutputType, int WarpsPerBlock = 4>
__global__ void mxfp8_quantize_zero_padding_kernel_sm120(OutputType* __restrict__ fp8_output,
                                                         int32_t* __restrict__ scale_output,
                                                         InputType const* __restrict__ input,
                                                         int32_t const* __restrict__ token_offset,
                                                         int64_t num_experts, int64_t size_k,
                                                         int64_t scale_leading_dim) {
  static_assert(GranK == 32 || GranK == 128, "GranK must be 32 or 128");
  constexpr int kLanesPerGroup = GranK / 16;
  constexpr int kGroupsPerWarp = 32 / kLanesPerGroup;
  constexpr int kInt32PerWarp = kGroupsPerWarp / 4;

  extern __shared__ char shared_memory[];
  int32_t* smem_token_offset = reinterpret_cast<int32_t*>(shared_memory);

  for (int i = threadIdx.x; i <= num_experts; i += blockDim.x) {
    smem_token_offset[i] = token_offset[i];
  }
  __syncthreads();

  const int64_t token_num = smem_token_offset[num_experts];

  int const warp_id = threadIdx.x >> 5;
  int const lane_id = threadIdx.x & 31;

  const int64_t k_block_idx = blockIdx.x;
  const int64_t grid_stride = static_cast<int64_t>(gridDim.y) * WarpsPerBlock;

  for (int64_t token_idx = static_cast<int64_t>(blockIdx.y) * WarpsPerBlock + warp_id;
       token_idx < token_num; token_idx += grid_stride) {
    int64_t expert_idx = 0;
    {
      int left = 0;
      int right = num_experts - 1;
      while (left < right) {
        int mid = (left + right + 1) >> 1;
        if (smem_token_offset[mid] <= token_idx) {
          left = mid;
        } else {
          right = mid - 1;
        }
      }
      expert_idx = left;
    }

    const int64_t local_token_idx = token_idx - smem_token_offset[expert_idx];

    int const k_offset = (k_block_idx * 512 + lane_id * 16);

    auto const cur_input_ptr =
        reinterpret_cast<double4 const*>(input + token_idx * size_k + k_offset);

    constexpr int kLoadNumElems = sizeof(double4) / sizeof(InputType);
    union LoadTrick {
      double4 pack;
      InputType v[kLoadNumElems];
    };
    LoadTrick load_trick;

    load_trick.pack = k_offset < size_k ? cur_input_ptr[0] : double4{};

    InputType max_elem = InputType(0.0f);
#pragma unroll
    for (int i = 0; i < kLoadNumElems; i++) {
      max_elem = __hmax(max_elem, __habs(load_trick.v[i]));
    }

    float amax = float(max_elem);
    if constexpr (kLanesPerGroup >= 8) {
      amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 4, kLanesPerGroup));
    }
    if constexpr (kLanesPerGroup >= 4) {
      amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 2, kLanesPerGroup));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 1, kLanesPerGroup));
    amax = fmaxf(amax, 1e-10f);

    float dequant_scale_raw = amax * reciprocal_approximate_ftz(448.0f);
    __nv_fp8_e8m0 ue8m0_scale;
    ue8m0_scale.__x = __nv_cvt_float_to_e8m0(dequant_scale_raw, __NV_SATFINITE, cudaRoundPosInf);
    float quant_scale = exp2f_rcp(ue8m0_scale.__x);

    constexpr int kStoreNumElems = sizeof(float4) / sizeof(OutputType);
    union StoreTrick {
      float4 pack;
      OutputType v[kStoreNumElems];
    };
    StoreTrick store_trick;
    store_trick.pack = float4{};

#pragma unroll
    for (int i = 0; i < kStoreNumElems; i++) {
      store_trick.v[i] = OutputType(float(load_trick.v[i]) * quant_scale);
    }

    auto cur_output_ptr = reinterpret_cast<float4*>(fp8_output + token_idx * size_k + k_offset);

    if (k_offset < size_k) {
      cur_output_ptr[0] = store_trick.pack;
    }

    uint32_t my_scale = static_cast<uint32_t>(ue8m0_scale.__x);
    if (k_offset >= size_k) my_scale = 0;
    const int64_t scale_padded_offset = math::compute_padded_offset(
        static_cast<int64_t>(smem_token_offset[expert_idx]), expert_idx);

#pragma unroll
    for (int int32_idx = 0; int32_idx < kInt32PerWarp; int32_idx++) {
      int const base_lane = int32_idx * 8;
      uint32_t s0 = __shfl_sync(0xFFFFFFFF, my_scale, base_lane + 0 * kLanesPerGroup);
      uint32_t s1 = __shfl_sync(0xFFFFFFFF, my_scale, base_lane + 1 * kLanesPerGroup);
      uint32_t s2 = __shfl_sync(0xFFFFFFFF, my_scale, base_lane + 2 * kLanesPerGroup);
      uint32_t s3 = __shfl_sync(0xFFFFFFFF, my_scale, base_lane + 3 * kLanesPerGroup);

      if (lane_id == base_lane) {
        uint32_t packed_scale = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);
        int64_t const out_row = k_block_idx * kInt32PerWarp + int32_idx;
        auto cur_scale_ptr = scale_output + out_row * scale_leading_dim + scale_padded_offset;
        *reinterpret_cast<uint32_t*>(&cur_scale_ptr[local_token_idx]) = packed_scale;
      }
    }
  }
}

}  // namespace sm120_blockscaled
