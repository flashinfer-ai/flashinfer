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

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#include <cstdio>
#include <cstdlib>

using bf16 = __nv_bfloat16;
using fp8 = __nv_fp8_e4m3;
using bf16_2 = __nv_bfloat162;

constexpr float LOG2E = 1.44269504088896340736f;
constexpr float FP8_MAX = 448.0f;
constexpr float FP8_MIN = -448.0f;
constexpr float FP8_MAX_INV = 1.0f / 448.0f;

__device__ __forceinline__ float to_float(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 to_bf16(float x) { return __float2bfloat16(x); }

__device__ __forceinline__ float clamp_fp8_e4m3_range(float x) {
  return fmaxf(FP8_MIN, fminf(FP8_MAX, x));
}

__device__ __forceinline__ uint8_t quantize_e4m3_byte(float x) {
  __nv_fp8_e4m3 y(clamp_fp8_e4m3_range(x));
  return y.__x;
}

__device__ __forceinline__ float dequantize_e4m3_byte(uint8_t x) {
  __nv_fp8_e4m3 y;
  y.__x = x;
  return static_cast<float>(y);
}

__device__ __forceinline__ uint8_t quantize_e4m3_residual_byte(float x) {
  const float clamped = clamp_fp8_e4m3_range(x);
  const uint8_t high = quantize_e4m3_byte(clamped);
  return quantize_e4m3_byte(clamped - dequantize_e4m3_byte(high));
}

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
  return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) val += __shfl_xor_sync(0xffffffff, val, offset);
  return val;
}

// Reductions over the 8 lanes sharing `lane & 3` — one swapAB softmax row.
__device__ __forceinline__ float warp8_reduce_max(float val) {
#pragma unroll
  for (int offset = 4; offset <= 16; offset <<= 1)
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
  return val;
}

__device__ __forceinline__ float warp8_reduce_sum(float val) {
#pragma unroll
  for (int offset = 4; offset <= 16; offset <<= 1) val += __shfl_xor_sync(0xffffffff, val, offset);
  return val;
}

// 4 × FP32 → one register of packed E4M3.
__device__ __forceinline__ uint32_t cvt_e4m3x4(float a, float b, float c, float d) {
  uint16_t lo, hi;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(b), "f"(a));
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(d), "f"(c));
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

// Residual limb of a packed E4M3 quad: what the first rounding dropped, so the
// two limbs together widen the mantissa. F16x2 unpacks a pair per conversion.
__device__ __forceinline__ uint32_t cvt_e4m3x4_residual(float a, float b, float c, float d,
                                                        uint32_t high) {
  uint32_t ab, cd;
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(ab) : "h"(static_cast<uint16_t>(high)));
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(cd) : "h"(static_cast<uint16_t>(high >> 16)));
  const __half2 h_ab = *reinterpret_cast<const __half2*>(&ab);
  const __half2 h_cd = *reinterpret_cast<const __half2*>(&cd);
  return cvt_e4m3x4(a - __low2float(h_ab), b - __high2float(h_ab), c - __low2float(h_cd),
                    d - __high2float(h_cd));
}

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                                         \
  do {                                                                                           \
    cudaError_t err = (call);                                                                    \
    if (err != cudaSuccess) {                                                                    \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      abort();                                                                                   \
    }                                                                                            \
  } while (0)
#endif
