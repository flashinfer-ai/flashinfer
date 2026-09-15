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
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <flashinfer/math.cuh>

namespace flashinfer::sparse_mla_sm120::nvfp4 {

constexpr int SF_VEC_SIZE = 16;
constexpr int FP4_PACKED_PER_GROUP = SF_VEC_SIZE / 2;

__device__ __forceinline__ float e4m3_byte_to_float(uint8_t byte) {
  __nv_fp8_e4m3 value;
  value.__x = byte;
  return static_cast<float>(value);
}

__device__ __forceinline__ uint8_t float_to_e4m3_byte(float value) {
  return __nv_fp8_e4m3(value).__x;
}

template <typename T>
__device__ __forceinline__ float nvfp4_input_to_float(T value);

template <>
__device__ __forceinline__ float nvfp4_input_to_float<half>(half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float nvfp4_input_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

// Linear NVFP4 group primitive used by both Q staging and the paged latent-KV
// writer. One E4M3 scale covers 16 values and the E2M1 payload occupies 8 bytes.
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

template <typename T>
__device__ __forceinline__ void quantize_group16_to_nvfp4(const T* input, uint8_t* packed_output,
                                                          uint8_t* scale_output) {
  float values[SF_VEC_SIZE];
#pragma unroll
  for (int i = 0; i < SF_VEC_SIZE; ++i) values[i] = nvfp4_input_to_float(input[i]);
  quantize_fp32_group16_to_nvfp4(values, packed_output, scale_output);
}

// Keep the vectorized conversions used by the attention hot path while the
// scalar template above serves format writers with other 16-bit inputs.
__device__ __forceinline__ void quantize_group16_to_nvfp4(const __nv_bfloat16* input,
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

__device__ __forceinline__ void quantize_group16_to_nvfp4(const half* input, uint8_t* packed_output,
                                                          uint8_t* scale_output) {
  float values[SF_VEC_SIZE];
#pragma unroll
  for (int i = 0; i < SF_VEC_SIZE / 2; ++i) {
    const half2 pair = *reinterpret_cast<const half2*>(input + i * 2);
    const float2 converted = __half22float2(pair);
    values[i * 2] = converted.x;
    values[i * 2 + 1] = converted.y;
  }
  quantize_fp32_group16_to_nvfp4(values, packed_output, scale_output);
}

}  // namespace flashinfer::sparse_mla_sm120::nvfp4
