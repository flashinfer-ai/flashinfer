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
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer::attention {

// Reset externally observed repair state even when the body is skipped.
static __global__ void SetSelectiveRepairCondition(const int32_t* flags, int32_t* second_flags,
                                                   int32_t* exact_error, int32_t* first_count,
                                                   int32_t* second_count, int64_t rows,
                                                   cudaGraphConditionalHandle handle) {
  bool active = false;
  for (int64_t row = threadIdx.x; row < rows; row += blockDim.x) {
    active |= flags[row] != 0;
    second_flags[row] = 0;
  }
  const int any_active = __syncthreads_or(active);
  if (threadIdx.x == 0) {
    *exact_error = 0;
    if (first_count) *first_count = 0;
    if (second_count) *second_count = 0;
    cudaGraphSetConditional(handle, any_active);
  }
}

}  // namespace flashinfer::attention
