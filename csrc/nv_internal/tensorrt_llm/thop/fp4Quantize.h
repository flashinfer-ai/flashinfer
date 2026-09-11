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

#include <tvm/ffi/container/tuple.h>

#include <cstdint>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/thop/utils.h"

using tvm::ffi::Optional;
using tvm::ffi::Tuple;

// `nvfp44Over6Code` is the NVFP4 4over6 wire code (see
// tensorrt_llm/kernels/nvfp4Recipe.h and issue #5141). Appended last so existing
// positional construction keeps working. It is deliberately NOT defaulted:
// TVM_FFI_DLL_EXPORT_TYPED_FUNC derives arity from the function type, so a C++
// default would buy nothing on the Python side while letting a C++ caller drop the
// recipe silently. Pass kNVFP44Over6FromEnv to keep the legacy env-driven behaviour.
void fp4_quantize(TensorView self, Optional<TensorView> const& globalScale, TensorView valueE2M1,
                  TensorView scaleFP8SF, int64_t sfVecSize, bool sfUseUE8M0,
                  bool isSfSwizzledLayout, bool isSf8x4Layout, bool isGlobalScaleInversed,
                  bool enable_pdl, int64_t nvfp44Over6Code);

void fp4_batched_quantize(Tensor self, Tensor globalScale, Tensor valueE2M1, Tensor scaleFP8SF,
                          int64_t sfVecSize, bool sfUseUE8M0, int64_t nvfp44Over6Code);

void silu_and_mul_scaled_nvfp4_experts_quantize(Tensor output, Tensor output_scale,
                                                Tensor const input, Tensor const input_global_scale,
                                                Tensor const mask, bool use_silu_and_mul,
                                                int64_t nvfp44Over6Code);

// NOTE: sfLayout lost its `= 2` default because nvfp44Over6Code follows it without
// one. No C++ caller relied on it; Python always passes it explicitly since the FFI
// export ignores C++ defaults.
void nvfp4_quant_and_per_token_scale(TensorView const input, double scale_inv, TensorView output,
                                     TensorView output_scale, TensorView output_per_token_scale,
                                     Optional<TensorView> expanded_idx_to_permuted_idx,
                                     int64_t sfLayout, int64_t nvfp44Over6Code);
