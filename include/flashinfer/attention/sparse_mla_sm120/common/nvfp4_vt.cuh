/*
 * Copyright (c) 2026 by FlashInfer team.
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

#pragma once

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

#include "../model/nvfp4_cache_traits.cuh"
#include "nvfp4_quantization.cuh"

namespace flashinfer::sparse_mla_sm120::nvfp4 {

using DSV4NVFP4Cache = NVFP4CacheTraits<ModelType::DSV4>;
static_assert(DSV4NVFP4Cache::SCALE_GROUP_SIZE == SF_VEC_SIZE);

constexpr int NVFP4_VT_CANDIDATES = 64;
constexpr int NVFP4_VT_PACKED_K_BYTES = NVFP4_VT_CANDIDATES / 2;
constexpr int NVFP4_VT_SCALE_GROUPS = NVFP4_VT_CANDIDATES / SF_VEC_SIZE;
constexpr int NVFP4_VT_DATA_BYTES = DSV4NVFP4Cache::D_NOPE * NVFP4_VT_PACKED_K_BYTES;
constexpr int NVFP4_VT_SCALE_BYTES = DSV4NVFP4Cache::D_NOPE * NVFP4_VT_SCALE_GROUPS;

// Decode two packed E2M1 values with the native SM100+ conversion. The V^T
// preparation consumes two source candidates at a time.
__device__ __forceinline__ float2 e2m1x2_code_to_float2(uint8_t codes) {
  uint32_t fp16x2;
  const uint32_t packed = codes;
  asm volatile(
      "{\n"
      ".reg .b8 fp4_byte;\n"
      "mov.b32 {fp4_byte, _, _, _}, %1;\n"
      "cvt.rn.f16x2.e2m1x2 %0, fp4_byte;\n"
      "}"
      : "=r"(fp16x2)
      : "r"(packed));
  const __half2 h2 = *reinterpret_cast<const __half2*>(&fp16x2);
  return __half22float2(h2);
}

__device__ __forceinline__ uint64_t transpose_e2m1_16x16_stage(uint64_t packed, int lane_in_group,
                                                               int distance, uint64_t low_mask) {
  const uint32_t packed_lo = static_cast<uint32_t>(packed);
  const uint32_t packed_hi = static_cast<uint32_t>(packed >> 32);
  const uint64_t partner =
      static_cast<uint64_t>(__shfl_xor_sync(0xffffffffu, packed_lo, distance, SF_VEC_SIZE)) |
      (static_cast<uint64_t>(__shfl_xor_sync(0xffffffffu, packed_hi, distance, SF_VEC_SIZE)) << 32);
  const int shift = distance * 4;
  if (lane_in_group & distance) {
    return ((partner & ~low_mask) >> shift) | (packed & ~low_mask);
  }
  return (packed & low_mask) | ((partner & low_mask) << shift);
}

// Transpose one 16-candidate x 16-dimension E2M1 tile entirely in registers.
__device__ __forceinline__ uint64_t transpose_e2m1_16x16(uint64_t packed, int lane_in_group) {
  packed = transpose_e2m1_16x16_stage(packed, lane_in_group, 8, 0x00000000ffffffffULL);
  packed = transpose_e2m1_16x16_stage(packed, lane_in_group, 4, 0x0000ffff0000ffffULL);
  packed = transpose_e2m1_16x16_stage(packed, lane_in_group, 2, 0x00ff00ff00ff00ffULL);
  return transpose_e2m1_16x16_stage(packed, lane_in_group, 1, 0x0f0f0f0f0f0f0f0fULL);
}

// Convert one token-major 64-candidate shared-memory tile into the layout
// consumed by block-scaled P x V. Source scales are absorbed and the result is
// requantized without a global V^T workspace.
template <int WORKER_THREADS, int KV_SMEM_STRIDE, int THREAD_BASE = 0>
__device__ __forceinline__ void prepare_nvfp4_vt_from_smem(const uint8_t* __restrict__ kv_fp4,
                                                           const uint8_t* __restrict__ kv_sc,
                                                           uint8_t* __restrict__ vt_data,
                                                           uint8_t* __restrict__ vt_sc) {
  constexpr int NUM_DIM_GROUPS = DSV4NVFP4Cache::D_NOPE / SF_VEC_SIZE;
  constexpr int DIM_GROUPS_PER_ITER = WORKER_THREADS / NVFP4_VT_CANDIDATES;
  static_assert(WORKER_THREADS >= NVFP4_VT_CANDIDATES && WORKER_THREADS % NVFP4_VT_CANDIDATES == 0);

  const int worker_tid = threadIdx.x - THREAD_BASE;
  const int warp = worker_tid / 32;
  const int lane = worker_tid & 31;
  const int lane_in_group = lane & (SF_VEC_SIZE - 1);
  const int half_warp = lane / SF_VEC_SIZE;
  const int warp_cand_pair = warp & 1;
  const int cand_group = warp_cand_pair * 2 + half_warp;
  const int cand = cand_group * SF_VEC_SIZE + lane_in_group;

  for (int dim_group = warp / 2; dim_group < NUM_DIM_GROUPS; dim_group += DIM_GROUPS_PER_ITER) {
    const uint64_t packed = *reinterpret_cast<const uint64_t*>(
        kv_fp4 + (size_t)cand * KV_SMEM_STRIDE + dim_group * FP4_PACKED_PER_GROUP);
    const float source_scale =
        e4m3_byte_to_float(kv_sc[(size_t)cand * DSV4NVFP4Cache::SCALE_BYTES_PER_TOKEN + dim_group]);
    const uint64_t transposed = transpose_e2m1_16x16(packed, lane_in_group);

    float values[SF_VEC_SIZE];
#pragma unroll
    for (int source_pair = 0; source_pair < SF_VEC_SIZE / 2; ++source_pair) {
      const int source_lane0 = source_pair * 2;
      const int source_lane1 = source_lane0 + 1;
      const uint8_t codes = static_cast<uint8_t>(transposed >> (source_pair * 8));
      const float2 decoded = e2m1x2_code_to_float2(codes);
      const float scale0 = __shfl_sync(0xffffffffu, source_scale, source_lane0, SF_VEC_SIZE);
      const float scale1 = __shfl_sync(0xffffffffu, source_scale, source_lane1, SF_VEC_SIZE);
      values[source_lane0] = decoded.x * scale0;
      values[source_lane1] = decoded.y * scale1;
    }

    const int dim = dim_group * SF_VEC_SIZE + lane_in_group;
    uint2 quantized;
    uint8_t quantized_scale;
    quantize_fp32_group16_to_nvfp4_regs(values, quantized, quantized_scale);

    // Adjacent half-warps cover adjacent candidate groups. Merge their 8-byte
    // results into one aligned 16-byte transaction.
    const int peer_lane = lane_in_group + SF_VEC_SIZE;
    const uint32_t peer_lo = __shfl_sync(0xffffffffu, quantized.x, peer_lane);
    const uint32_t peer_hi = __shfl_sync(0xffffffffu, quantized.y, peer_lane);
    const uint32_t peer_scale =
        __shfl_sync(0xffffffffu, static_cast<uint32_t>(quantized_scale), peer_lane);
    if (half_warp == 0) {
      *reinterpret_cast<uint4*>(vt_data + dim * NVFP4_VT_PACKED_K_BYTES + warp_cand_pair * 16) =
          make_uint4(quantized.x, quantized.y, peer_lo, peer_hi);
      *reinterpret_cast<uint16_t*>(vt_sc + dim * NVFP4_VT_SCALE_GROUPS + warp_cand_pair * 2) =
          static_cast<uint16_t>(quantized_scale) | static_cast<uint16_t>(peer_scale << 8);
    }
  }
}

}  // namespace flashinfer::sparse_mla_sm120::nvfp4
