/*
 * Copyright (c) 2025 by SageAttention team.
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

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <flashinfer/math.cuh>

namespace nvfp4_attention {

CUTLASS_DEVICE void packed_float_to_ue4m3(float const& f0, float const& f1, float const& f2,
                                          float const& f3, uint32_t& out) {
  out = flashinfer::math::fp32_vec_to_e4m3(f0, f1, f2, f3);
}

CUTLASS_DEVICE void packed_float_to_e2m1(float const& f0, float const& f1, float const& f2,
                                         float const& f3, float const& f4, float const& f5,
                                         float const& f6, float const& f7, uint32_t& out) {
  out = flashinfer::math::fp32_vec_to_e2m1(f0, f1, f2, f3, f4, f5, f6, f7);
}

template <int N>
struct FP8E4M3Converter {
  static_assert(N % 4 == 0, "N must be multiple of 4");

  __device__ __forceinline__ static void convert(float const inputs[N], uint32_t outputs[N / 4]) {
#pragma unroll
    for (int i = 0; i < N / 4; ++i) {
      packed_float_to_ue4m3(inputs[i * 4 + 0], inputs[i * 4 + 1], inputs[i * 4 + 2],
                            inputs[i * 4 + 3], outputs[i]);
    }
  }
};

template <int N>
struct FP4E2M1Converter {
  static_assert(N % 8 == 0, "N must be multiple of 8");

  __device__ __forceinline__ static void convert(float const inputs[N], uint32_t outputs[N / 8]) {
#pragma unroll
    for (int i = 0; i < N / 8; ++i) {
      packed_float_to_e2m1(inputs[i * 8 + 0], inputs[i * 8 + 1], inputs[i * 8 + 2],
                           inputs[i * 8 + 3], inputs[i * 8 + 4], inputs[i * 8 + 5],
                           inputs[i * 8 + 6], inputs[i * 8 + 7], outputs[i]);
    }
  }
};

}  // namespace nvfp4_attention
