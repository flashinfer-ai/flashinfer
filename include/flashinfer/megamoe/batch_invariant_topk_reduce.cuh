/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer {
namespace cake_megamoe_topk_reduce {

// BI is a separate specialization: every ordered route contributes an RN
// addition to +0, including masked slots. Never read stale masked payloads.
template <typename Id>
__global__ void BatchInvariantReduce(const __nv_bfloat16* __restrict__ partials,
                                     const Id* __restrict__ ids, __nv_bfloat16* __restrict__ out,
                                     int64_t hidden, int64_t top_k, int64_t num_experts) {
  const int64_t pairs_per_token = hidden / 2;
  const int64_t worker = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t token = worker / pairs_per_token;
  const int64_t column = 2 * (worker % pairs_per_token);
  float2 acc = make_float2(0.0f, 0.0f);
  for (int64_t slot = 0; slot < top_k; ++slot) {
    const Id expert = ids[token * top_k + slot];
    float2 value = make_float2(0.0f, 0.0f);
    if (expert >= 0 && expert < num_experts) {
      const auto bits =
          *reinterpret_cast<const uint32_t*>(partials + (token * top_k + slot) * hidden + column);
      value.x = __uint_as_float(bits << 16);
      value.y = __uint_as_float(bits & 0xffff0000u);
    }
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*reinterpret_cast<unsigned long long*>(&acc))
        : "l"(*reinterpret_cast<unsigned long long*>(&acc)),
          "l"(*reinterpret_cast<unsigned long long*>(&value)));
  }
  const __nv_bfloat162 packed = __float22bfloat162_rn(acc);
  *reinterpret_cast<__nv_bfloat162*>(out + token * hidden + column) = packed;
}

}  // namespace cake_megamoe_topk_reduce
}  // namespace flashinfer
