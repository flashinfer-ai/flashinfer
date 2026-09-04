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

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <flashinfer/math.cuh>

#include "../arch/barrier.cuh"
#include "../model/kv_cache_traits.cuh"

namespace flashinfer::sparse_mla_sm120::nvfp4 {

constexpr int SF_VEC_SIZE = 16;
constexpr int FP4_PACKED_PER_GROUP = SF_VEC_SIZE / 2;
constexpr int DSV4_NVFP4_NUM_SCALES = KVCacheTraits<ModelType::DSV4>::D_NOPE / SF_VEC_SIZE;
constexpr int DSV4_NVFP4_Q_PACKED_STRIDE = KVCacheTraits<ModelType::DSV4>::D_NOPE / 2 + 16;
constexpr int DSV4_NVFP4_SCALE_STRIDE = 32;

__device__ __forceinline__ float e4m3_byte_to_float(uint8_t byte) {
  __nv_fp8_e4m3 value;
  value.__x = byte;
  return static_cast<float>(value);
}

__device__ __forceinline__ uint8_t float_to_e4m3_byte(float value) {
  return __nv_fp8_e4m3(value).__x;
}

__device__ __forceinline__ void quantize_fp32_group16_to_nvfp4_regs(const float values[SF_VEC_SIZE],
                                                                    uint2& packed_output,
                                                                    uint8_t& scale_output) {
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < SF_VEC_SIZE; ++i) amax = fmaxf(amax, fabsf(values[i]));

  const uint8_t scale_byte = float_to_e4m3_byte(amax / 6.f);
  scale_output = scale_byte;
  const float scale = e4m3_byte_to_float(scale_byte);
  const float scale_inv = scale == 0.f ? 0.f : 1.f / scale;
  float normalized[SF_VEC_SIZE];
#pragma unroll
  for (int i = 0; i < SF_VEC_SIZE; ++i) normalized[i] = values[i] * scale_inv;

  packed_output = make_uint2(
      math::fp32_vec_to_e2m1(normalized[0], normalized[1], normalized[2], normalized[3],
                             normalized[4], normalized[5], normalized[6], normalized[7]),
      math::fp32_vec_to_e2m1(normalized[8], normalized[9], normalized[10], normalized[11],
                             normalized[12], normalized[13], normalized[14], normalized[15]));
}

__device__ __forceinline__ void quantize_fp32_group16_to_nvfp4(const float values[SF_VEC_SIZE],
                                                               uint8_t* packed_output,
                                                               uint8_t* scale_output) {
  uint2 packed;
  uint8_t scale;
  quantize_fp32_group16_to_nvfp4_regs(values, packed, scale);
  *reinterpret_cast<uint2*>(packed_output) = packed;
  *scale_output = scale;
}

__device__ __forceinline__ void quantize_bf16_group16_to_nvfp4(const bf16* input,
                                                               uint8_t* packed_output,
                                                               uint8_t* scale_output) {
  float values[SF_VEC_SIZE];
#pragma unroll
  for (int i = 0; i < SF_VEC_SIZE / 2; ++i) {
    const __nv_bfloat162 pair = *reinterpret_cast<const __nv_bfloat162*>(input + i * 2);
    const float2 converted = __bfloat1622float2(pair);
    values[i * 2] = converted.x;
    values[i * 2 + 1] = converted.y;
  }
  quantize_fp32_group16_to_nvfp4(values, packed_output, scale_output);
}

template <int MATH_THREADS, int PACKED_STRIDE = DSV4_NVFP4_Q_PACKED_STRIDE,
          int SCALE_STRIDE = DSV4_NVFP4_SCALE_STRIDE>
__device__ __forceinline__ void quantize_q_nvfp4_to_smem(uint8_t* q_nope_fp4,
                                                         uint8_t* q_nope_scales, bf16* q_rope,
                                                         const bf16* q_base, int valid_hpb = HPB) {
  using KV = KVCacheTraits<ModelType::DSV4>;
  constexpr int NUM_GROUPS = HPB * DSV4_NVFP4_NUM_SCALES;

  for (int group = threadIdx.x; group < NUM_GROUPS; group += MATH_THREADS) {
    const int head = group / DSV4_NVFP4_NUM_SCALES;
    const int group_in_head = group % DSV4_NVFP4_NUM_SCALES;
    uint8_t* packed_dst = q_nope_fp4 + head * PACKED_STRIDE + group_in_head * FP4_PACKED_PER_GROUP;
    uint8_t* scale_dst = q_nope_scales + head * SCALE_STRIDE + group_in_head;
    if (head < valid_hpb) {
      const bf16* src = q_base + head * KV::D_QK + group_in_head * SF_VEC_SIZE;
      quantize_bf16_group16_to_nvfp4(src, packed_dst, scale_dst);
    } else {
      *reinterpret_cast<uint2*>(packed_dst) = make_uint2(0, 0);
      *scale_dst = 0;
    }
  }

  for (int i = threadIdx.x; i < HPB * KV::D_ROPE; i += MATH_THREADS) {
    const int head = i / KV::D_ROPE;
    const int dim = i % KV::D_ROPE;
    q_rope[i] =
        (head < valid_hpb) ? q_base[head * KV::D_QK + KV::D_NOPE + dim] : __float2bfloat16(0.f);
  }
  bar_sync_t<2, MATH_THREADS>();
}

}  // namespace flashinfer::sparse_mla_sm120::nvfp4
