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

#include <cstddef>
#include <cstdint>

// Directly produce the SM120 M16N8K64 NVFP4 B registers from a candidate-major
// packed cache. Each lane loads all eight N dimensions for one candidate as a
// 32-bit word, then warp shuffles perform the 8x8 nibble transpose. Compared
// with scalar per-dimension gathering this reduces shared loads from sixteen
// bytes to two words per lane.
template <int KV_STRIDE>
__device__ __forceinline__ void d2_load_b_nvfp4(uint32_t& b0, uint32_t& b1,
                                                const uint8_t* __restrict__ kv_smem, int entry_base,
                                                int dim, int lane) {
  const int gid = lane >> 2;
  const int tid = lane & 3;
  // dim is the first dimension of an aligned N8 output tile.
  const int byte_column = dim >> 1;
  const int nibble_shift = gid * 4;
  const uint32_t candidate_word0 = *reinterpret_cast<const uint32_t*>(
      kv_smem + (size_t)(entry_base + lane) * KV_STRIDE + byte_column);
  const uint32_t candidate_word1 = *reinterpret_cast<const uint32_t*>(
      kv_smem + (size_t)(entry_base + 32 + lane) * KV_STRIDE + byte_column);

  b0 = 0;
  b1 = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int source_lane = tid * 8 + i;
    const uint32_t v0 =
        (__shfl_sync(0xffffffffu, candidate_word0, source_lane) >> nibble_shift) & 0xFu;
    const uint32_t v1 =
        (__shfl_sync(0xffffffffu, candidate_word1, source_lane) >> nibble_shift) & 0xFu;
    b0 |= v0 << (i * 4);
    b1 |= v1 << (i * 4);
  }
}

// Produce two adjacent N8 operands together. The N16 source for one candidate
// is one aligned uint64 load, so the pair shares address arithmetic and lets
// ptxas issue LDS.64 instead of two independently scheduled LDS.32 operations.
// Shuffle count is unchanged (each output nibble is distinct), but the common
// transpose loop exposes substantially more instruction-level parallelism.
template <int KV_STRIDE>
__device__ __forceinline__ void d2_load_b_nvfp4_n16(uint32_t& b00, uint32_t& b01, uint32_t& b10,
                                                    uint32_t& b11,
                                                    const uint8_t* __restrict__ kv_smem,
                                                    int entry_base, int dim, int lane) {
  const int gid = lane >> 2;
  const int tid = lane & 3;
  const int byte_column = dim >> 1;
  const int nibble_shift = gid * 4;
  const uint2 candidate0 = *reinterpret_cast<const uint2*>(
      kv_smem + (size_t)(entry_base + lane) * KV_STRIDE + byte_column);
  const uint2 candidate1 = *reinterpret_cast<const uint2*>(
      kv_smem + (size_t)(entry_base + 32 + lane) * KV_STRIDE + byte_column);

  b00 = 0;
  b01 = 0;
  b10 = 0;
  b11 = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int source_lane = tid * 8 + i;
    const uint32_t c00 = __shfl_sync(0xffffffffu, candidate0.x, source_lane);
    const uint32_t c01 = __shfl_sync(0xffffffffu, candidate0.y, source_lane);
    const uint32_t c10 = __shfl_sync(0xffffffffu, candidate1.x, source_lane);
    const uint32_t c11 = __shfl_sync(0xffffffffu, candidate1.y, source_lane);
    b00 |= ((c00 >> nibble_shift) & 0xFu) << (i * 4);
    b01 |= ((c10 >> nibble_shift) & 0xFu) << (i * 4);
    b10 |= ((c01 >> nibble_shift) & 0xFu) << (i * 4);
    b11 |= ((c11 >> nibble_shift) & 0xFu) << (i * 4);
  }
}
