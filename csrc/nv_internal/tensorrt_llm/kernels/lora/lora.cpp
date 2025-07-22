/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/kernels/lora/lora.h"

#include <algorithm>
#include <utility>

namespace tensorrt_llm::kernels {

int Lora_run(LoraImpl* impl, int64_t numTokens, int64_t numReqs, void const* input,
             int32_t const* loraRanks, void const* const* loraWeightsPtr, int weightIndex,
             void* const* outputs, void* workspace, cudaStream_t stream) {
  return -1;
}

}  // namespace tensorrt_llm::kernels
