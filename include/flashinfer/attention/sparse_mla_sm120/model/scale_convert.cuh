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

#include <cstdint>
#include <cstring>

// FP32 ↔ UE8M0 scale conversion utilities.
//
// UE8M0 = unsigned 8-bit exponent, 0 mantissa bits.
// Represents powers of 2: value = 2^(ue8m0 - 127).
// Range: [2^-127, 2^128].
//
// FlashMLA stores FP32 scales rounded to power-of-2 (via 2^ceil(log2(scale))),
// so FP32→UE8M0 is exact (just extract the exponent byte).

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

__host__ __device__ __forceinline__ uint8_t fp32_to_ue8m0(float v) {
  return static_cast<uint8_t>((detail::float_as_uint(v) >> 23) & 0xFF);
}

__host__ __device__ __forceinline__ float ue8m0_to_fp32(uint8_t v) {
  uint32_t bits = static_cast<uint32_t>(v) << 23;
  return detail::uint_as_float(bits);
}

// Pack 2 FP32 scales into UE8M0x2 (uint16_t): low byte = first, high byte = second
__host__ __device__ __forceinline__ uint16_t fp32x2_to_ue8m0x2(float a, float b) {
  uint8_t ea = fp32_to_ue8m0(a);
  uint8_t eb = fp32_to_ue8m0(b);
  return static_cast<uint16_t>(ea) | (static_cast<uint16_t>(eb) << 8);
}
