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
using tvm::ffi::Tensor;
using tvm::ffi::Tuple;

void fp4_quantize(Tensor self, Optional<Tensor> const& globalScale, Tensor valueE2M1,
                  Tensor scaleFP8SF, int64_t sfVecSize, bool sfUseUE8M0, bool isSfSwizzledLayout,
                  bool isSf8x4Layout, bool enable_pdl);

void fp4_batched_quantize(Tensor self, Optional<Tensor> const& mask, Tensor globalScale,
                          Tensor valueE2M1, Tensor scaleFP8SF, int64_t sfVecSize, bool sfUseUE8M0);

void silu_and_mul_nvfp4_batched_quantize(Tensor const& self, Tensor const& mask,
                                         Tensor const& globalScale, Tensor valueE2M1,
                                         Tensor scaleFP8SF, int64_t sfVecSize);
