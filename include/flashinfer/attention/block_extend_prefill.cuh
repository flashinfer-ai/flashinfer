/*
 * Copyright (c) 2023-2024 by FlashInfer team.
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

#ifndef FLASHINFER_ATTENTION_BLOCK_EXTEND_PREFILL_CUH_
#define FLASHINFER_ATTENTION_BLOCK_EXTEND_PREFILL_CUH_

#include <cuda_runtime.h>
#include "../math.cuh"

namespace flashinfer {

// Block Expanding Mask: mask[q, k] = (q / B) >= (k / B)
// For Q tile [q_start, q_end), visible KV range: [0, ceil(q_end / B) * B)


__device__ __forceinline__ uint64_t block_extend_kv_valid_end(
    uint32_t q_tile_end, uint32_t dllm_block_size, uint64_t q_offset = 0) {
  const uint64_t q_global_last_idx = q_offset + q_tile_end - 1;
  const uint64_t q_block_id = q_global_last_idx / dllm_block_size;
  return (q_block_id + 1) * dllm_block_size;
}

__device__ __forceinline__ uint32_t block_extend_num_iterations(
    uint32_t q_tile_end, uint32_t chunk_start, uint32_t chunk_size,
    uint32_t dllm_block_size, uint32_t CTA_TILE_KV, uint64_t q_offset = 0) {
  const uint64_t kv_valid_end =
      block_extend_kv_valid_end(q_tile_end, dllm_block_size, q_offset);
  const uint64_t effective_chunk_size =
      kv_valid_end <= chunk_start
          ? 0
          : min(static_cast<uint64_t>(chunk_size), kv_valid_end - chunk_start);
  return static_cast<uint32_t>((effective_chunk_size + CTA_TILE_KV - 1) / CTA_TILE_KV);
}

__device__ __forceinline__ uint32_t block_extend_mask_iteration(
    uint32_t q_tile_start, uint32_t chunk_start, uint32_t chunk_size,
    uint32_t dllm_block_size, uint32_t CTA_TILE_KV, uint64_t q_offset = 0,
    uint64_t kv_offset = 0) {
  const uint64_t q_global_start = q_offset + q_tile_start;
  const uint64_t q_first_block_id = q_global_start / dllm_block_size;
  // Consider kv_offset: kv_global < (q_block + 1) * B.
  const uint64_t max_kv_global = (q_first_block_id + 1) * dllm_block_size;
  const uint64_t kv_fully_visible_end =
      kv_offset >= max_kv_global ? 0 : max_kv_global - kv_offset;
  const uint64_t kv_fully_visible_in_chunk =
      kv_fully_visible_end <= chunk_start
          ? 0
          : min(static_cast<uint64_t>(chunk_size), kv_fully_visible_end - chunk_start);
  return static_cast<uint32_t>(kv_fully_visible_in_chunk / CTA_TILE_KV);
}

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_BLOCK_EXTEND_PREFILL_CUH_
