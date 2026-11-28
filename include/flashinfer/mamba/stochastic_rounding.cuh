// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

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
#ifndef FLASHINFER_MAMBA_STOCHASTIC_ROUNDING_CUH_
#define FLASHINFER_MAMBA_STOCHASTIC_ROUNDING_CUH_

#include "conversion.cuh"

namespace flashinfer::mamba::conversion {

// SR-only random source. The default forwards every Philox word and counter
// unchanged. Shared-seed mode retains uniform marginal bits but intentionally
// correlates rounding across checkpoint elements. Conversion helpers are pure
// functions of their input values and random word under either policy.
template <int PHILOX_ROUNDS>
__device__ __forceinline__ void sr_random4x(int64_t seed, int64_t offset, uint32_t& r0,
                                            uint32_t& r1, uint32_t& r2, uint32_t& r3) {
#if defined(FLASHINFER_MAMBA_SR_SHARED_SEED) && FLASHINFER_MAMBA_SR_SHARED_SEED
  r0 = r1 = r2 = r3 = static_cast<uint32_t>(seed);
#else
  philox_randint4x<PHILOX_ROUNDS>(seed, offset, r0, r1, r2, r3);
#endif
}

// Runtime-uniform policy for the INT8 checkpoint encoder. Only checkpoint
// encoding consumes the policy; each branch preserves the existing random words.
template <int PHILOX_ROUNDS>
__device__ __forceinline__ void sr_random4x_runtime(int64_t seed, int64_t offset, uint32_t& r0,
                                                    uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                                    bool shared_seed) {
  if (shared_seed) {
    r0 = r1 = r2 = r3 = static_cast<uint32_t>(seed);
  } else {
    philox_randint4x<PHILOX_ROUNDS>(seed, offset, r0, r1, r2, r3);
  }
}

}  // namespace flashinfer::mamba::conversion
#endif  // FLASHINFER_MAMBA_STOCHASTIC_ROUNDING_CUH_
