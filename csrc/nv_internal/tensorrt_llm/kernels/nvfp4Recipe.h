/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <cstdint>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/quantization.h"

namespace tensorrt_llm::kernels {

// The enum lives in tensorrt_llm::common; pulled in here the same way
// kernels/quantization.h pulls in flashinfer::QuantizationSFLayout.
using tensorrt_llm::common::NVFP44Over6ErrMode;

// Wire format shared with flashinfer/quantization/nvfp4_quantization_utils.py
// (see issue #5141). Keep the two sides in sync:
//   -1  FROM_ENV  caller supplied nothing -> read the FLASHINFER_NVFP4_4OVER6* env vars
//    0  STANDARD  4over6 off
//   odd enabled; bit0=1, bit1=(e4m3Max == 256), bits2-3=errMode(0=MAE, 1=MSE),
//       bit4=errUseFastMath
inline constexpr int64_t kNVFP44Over6FromEnv = -1;
inline constexpr int64_t kNVFP44Over6Standard = 0;

// Highest bit the decoder below understands. Codes with bits above this are
// rejected rather than silently truncated, so a future producer that widens the
// format fails loudly against an older binary.
inline constexpr int64_t kNVFP44Over6KnownBits = 5;

//! \brief A fully resolved NVFP4 quantization recipe: no env reads below this point.
struct NVFP4RecipeSpec {
  bool use4Over6 = false;
  NVFP44Over6ErrMode errMode = NVFP44Over6ErrMode::MAE;
  bool errUseFastMath = false;
  int e4m3Max = 448;

  // Single source of truth for the runtime global scale, which must agree with
  // the compile-time e4m3Max the quantizer is specialized on.
  float globalScaleInv() const { return 1.f / (static_cast<float>(e4m3Max) * 6.f); }
};

//! \brief Decode a wire code into a recipe. The ONLY decoder of the wire format.
//!
//! \param code One of kNVFP44Over6FromEnv, kNVFP44Over6Standard, or a packed
//!             odd code as documented above.
inline NVFP4RecipeSpec resolveNVFP4Recipe(int64_t code) {
  NVFP4RecipeSpec spec;

  if (code < 0) {
    TLLM_CHECK_WITH_INFO(code == kNVFP44Over6FromEnv,
                         "Unsupported NVFP4 4over6 code %lld: the only negative code is %lld "
                         "(FROM_ENV).",
                         static_cast<long long>(code), static_cast<long long>(kNVFP44Over6FromEnv));
    // Legacy process-wide path, byte-for-byte the behaviour that predates the
    // explicit wire code. The getters deliberately re-read getenv on every call
    // (they are not static-cached), so this stays a per-launch read.
    if (!tensorrt_llm::common::getEnvNVFP4Use4Over6()) {
      return spec;
    }
    spec.use4Over6 = true;
    spec.errMode = tensorrt_llm::common::getEnvNVFP44Over6ErrMode();
    spec.errUseFastMath = tensorrt_llm::common::getEnvNVFP44Over6ErrUseFastMath();
    spec.e4m3Max = tensorrt_llm::common::getEnvNVFP44Over6E4M3Use256() ? 256 : 448;
    return spec;
  }

  TLLM_CHECK_WITH_INFO((code >> kNVFP44Over6KnownBits) == 0,
                       "Unsupported NVFP4 4over6 code %lld: unknown bits set.",
                       static_cast<long long>(code));

  if ((code & 1) == 0) {
    // STANDARD: 4over6 off. Note this also covers kNVFP44Over6Standard.
    return spec;
  }

  spec.use4Over6 = true;
  spec.e4m3Max = ((code >> 1) & 1) != 0 ? 256 : 448;
  // Validated rather than static_cast: the torch op is publicly callable, so an
  // arbitrary int64 can reach here and must not become an out-of-range enum.
  int64_t const errModeBits = (code >> 2) & 0x3;
  TLLM_CHECK_WITH_INFO(errModeBits == 0 || errModeBits == 1,
                       "Unsupported NVFP4 4over6 error mode %lld (expected 0=MAE or 1=MSE).",
                       static_cast<long long>(errModeBits));
  spec.errMode = errModeBits == 0 ? NVFP44Over6ErrMode::MAE : NVFP44Over6ErrMode::MSE;
  spec.errUseFastMath = ((code >> 4) & 1) != 0;
  return spec;
}

}  // namespace tensorrt_llm::kernels
