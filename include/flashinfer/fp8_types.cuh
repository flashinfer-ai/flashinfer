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
#ifndef FLASHINFER_FP8_TYPES_CUH_
#define FLASHINFER_FP8_TYPES_CUH_

#include <cuda_fp8.h>

namespace flashinfer {

/*!
 * \brief Type trait providing the maximum representable value for FP8 types.
 * Used for clamping before FP8 cast to avoid NaN/Inf.
 *
 * - __nv_fp8_e4m3: 4-bit exponent, 3-bit mantissa, max = 448.0
 * - __nv_fp8_e5m2: 5-bit exponent, 2-bit mantissa, max = 57344.0
 */
template <typename T>
struct fp8_clamp_max {
  static_assert(sizeof(T) == 0, "Unsupported FP8 type for fp8_clamp_max");
};

template <>
struct fp8_clamp_max<__nv_fp8_e4m3> {
  static constexpr float value = 448.0f;
};

template <>
struct fp8_clamp_max<__nv_fp8_e5m2> {
  static constexpr float value = 57344.0f;
};

}  // namespace flashinfer

#endif  // FLASHINFER_FP8_TYPES_CUH_
