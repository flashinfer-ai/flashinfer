// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

#include "kv_cache_traits.cuh"
#include "model_type.h"

// Extract or rebuild FP32 exponent bits without rounding or mantissa handling.
// Rebuilding byte 0 yields zero; byte 255 yields positive infinity.

namespace detail {
__host__ __device__ __forceinline__ uint32_t float_as_uint(float v) {
  uint32_t r;
#ifdef __CUDA_ARCH__
  r = __float_as_uint(v);
#else
  memcpy(&r, &v, 4);
#endif
  return r;
}
__host__ __device__ __forceinline__ float uint_as_float(uint32_t v) {
  float r;
#ifdef __CUDA_ARCH__
  r = __uint_as_float(v);
#else
  memcpy(&r, &v, 4);
#endif
  return r;
}
}  // namespace detail

__host__ __device__ __forceinline__ uint8_t fp32_exponent_byte(float v) {
  return static_cast<uint8_t>((detail::float_as_uint(v) >> 23) & 0xFF);
}

__host__ __device__ __forceinline__ float fp32_from_exponent_byte(uint8_t v) {
  uint32_t bits = static_cast<uint32_t>(v) << 23;
  return detail::uint_as_float(bits);
}

// Pack exponent bytes: low byte = first, high byte = second.
__host__ __device__ __forceinline__ uint16_t fp32_exponent_byte_pair(float a, float b) {
  uint8_t ea = fp32_exponent_byte(a);
  uint8_t eb = fp32_exponent_byte(b);
  return static_cast<uint16_t>(ea) | (static_cast<uint16_t>(eb) << 8);
}

// Per-format scale -> UE8M0 conversion for block-scaled MMA. UE8M0 caches
// need no conversion; both FP32 formats reduce to the exponent byte.
template <ScaleFormat F>
struct ScaleConvert {
  static_assert(F == ScaleFormat::POW2_FP32 || F == ScaleFormat::ARBITRARY_FP32,
                "add a ScaleConvert specialization for this scale format");
  __device__ static __forceinline__ uint8_t to_ue8m0(float scale) { return fp32_exponent_byte(scale); }
};

template <>
struct ScaleConvert<ScaleFormat::UE8M0_BYTE> {
  __device__ static __forceinline__ uint8_t to_ue8m0(uint8_t scale) { return scale; }
};

__device__ __forceinline__ uint8_t KVCacheTraits<ModelType::DSV3_2>::scale_to_ue8m0(float scale) {
  return ScaleConvert<Scales::FORMAT>::to_ue8m0(scale);
}

__device__ __forceinline__ uint8_t KVCacheTraits<ModelType::GLM53_NOPE>::scale_to_ue8m0(float scale) {
  return ScaleConvert<Scales::FORMAT>::to_ue8m0(scale);
}

__device__ __forceinline__ uint8_t KVCacheTraits<ModelType::DOTS3_SWA>::scale_to_ue8m0(uint8_t scale) {
  return ScaleConvert<Scales::FORMAT>::to_ue8m0(scale);
}

__device__ __forceinline__ uint8_t KVCacheTraits<ModelType::DSV4>::scale_to_ue8m0(uint8_t scale) {
  return ScaleConvert<Scales::FORMAT>::to_ue8m0(scale);
}

__device__ __forceinline__ uint8_t KVCacheTraits<ModelType::DSV4_1>::scale_to_ue8m0(uint8_t scale) {
  return ScaleConvert<Scales::FORMAT>::to_ue8m0(scale);
}
