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
#include <ATen/cuda/EmptyTensor.h>

#include <cstdint>

#include "tensorrt_llm/common/cudaUtils.h"

namespace torch_ext {
std::tuple<at::Tensor, at::Tensor> fp4_quantize(at::Tensor const& self, at::Tensor const& globalScale,
    int64_t sfVecSize, bool sfUseUE8M0, bool isSfSwizzledLayout);

// at::Tensor block_scale_interleave(at::Tensor const& unswizzled_sf);

// at::Tensor e2m1_and_ufp8sf_scale_to_float(at::Tensor valueE2M1, at::Tensor scaleFP8SF,
//                                           std::optional<at::Tensor> globalScale, int64_t sfVecSize,
//                                           int64_t sfType, bool isSfSwizzledLayout);
}  // namespace torch_ext
