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

#ifndef FLASHINFER_ATTENTION_BLOCK_EXPANDING_PREFILL_CUH_
#define FLASHINFER_ATTENTION_BLOCK_EXPANDING_PREFILL_CUH_

#include <cuda_runtime.h>
#include "../math.cuh"

namespace flashinfer {

// Block Expanding Mask: mask[q, k] = (q / B) >= (k / B)
// For Q tile [q_start, q_end), visible KV range: [0, ceil(q_end / B) * B)

template <uint32_t CTA_TILE_KV>
struct BlockExpandingTileSkipController {
  uint32_t dllm_block_size;
  uint32_t log2_block_size;
  uint32_t kv_len;
  
  __device__ __host__ __forceinline__
  BlockExpandingTileSkipController(uint32_t block_size, uint32_t kv_length)
      : dllm_block_size(block_size), kv_len(kv_length) {
#ifdef __CUDA_ARCH__
    log2_block_size = __popc(block_size - 1);
#else
    log2_block_size = 0;
    while ((1u << log2_block_size) < block_size) ++log2_block_size;
#endif
  }
  
  __device__ __forceinline__
  uint32_t get_kv_valid_end(uint32_t q_tile_end) const {
    uint32_t q_last_idx = q_tile_end - 1;
    uint32_t q_block_id = q_last_idx >> log2_block_size;
    uint32_t valid_end = (q_block_id + 1) << log2_block_size;
    return min(valid_end, kv_len);
  }
  
  __device__ __forceinline__
  uint32_t get_num_iterations(uint32_t q_tile_end, uint32_t chunk_start, 
                              uint32_t chunk_size) const {
    uint32_t kv_valid_end = get_kv_valid_end(q_tile_end);
    uint32_t effective_chunk_size;
    if (kv_valid_end <= chunk_start) {
      effective_chunk_size = 0;
    } else {
      effective_chunk_size = min(chunk_size, kv_valid_end - chunk_start);
    }
    return (effective_chunk_size + CTA_TILE_KV - 1) / CTA_TILE_KV;
  }
  
  __device__ __forceinline__
  uint32_t get_mask_iteration(uint32_t q_tile_start, uint32_t chunk_start,
                               uint32_t chunk_size) const {
    uint32_t q_first_block_id = q_tile_start >> log2_block_size;
    uint32_t kv_fully_visible_end = (q_first_block_id + 1) << log2_block_size;
    uint32_t kv_fully_visible_in_chunk;
    if (kv_fully_visible_end <= chunk_start) {
      kv_fully_visible_in_chunk = 0;
    } else {
      kv_fully_visible_in_chunk = min(chunk_size, kv_fully_visible_end - chunk_start);
    }
    return kv_fully_visible_in_chunk / CTA_TILE_KV;
  }
  
  __device__ __forceinline__
  bool needs_mask(uint32_t iter, uint32_t mask_iteration, 
                  uint32_t num_iterations) const {
    return iter >= mask_iteration && iter < num_iterations;
  }
  
  __device__ __forceinline__
  bool is_tile_fully_visible(uint32_t q_tile_start, uint32_t kv_tile_start,
                              uint32_t kv_tile_end) const {
    uint32_t q_first_block = q_tile_start >> log2_block_size;
    uint32_t k_last_block = (kv_tile_end - 1) >> log2_block_size;
    return q_first_block >= k_last_block;
  }
  
  __device__ __forceinline__
  bool is_tile_fully_masked(uint32_t q_tile_end, uint32_t kv_tile_start) const {
    uint32_t q_last_block = (q_tile_end - 1) >> log2_block_size;
    uint32_t k_first_block = kv_tile_start >> log2_block_size;
    return q_last_block < k_first_block;
  }
};


__device__ __forceinline__ uint32_t block_expanding_kv_valid_end(
    uint32_t q_tile_end, uint32_t dllm_block_size, uint32_t q_offset = 0) {
  uint32_t q_global_last_idx = q_offset + q_tile_end - 1;
  uint32_t q_block_id = q_global_last_idx / dllm_block_size;
  return (q_block_id + 1) * dllm_block_size;
}

__device__ __forceinline__ uint32_t block_expanding_num_iterations(
    uint32_t q_tile_end, uint32_t chunk_start, uint32_t chunk_size,
    uint32_t dllm_block_size, uint32_t CTA_TILE_KV, uint32_t q_offset = 0) {
  uint32_t kv_valid_end = block_expanding_kv_valid_end(q_tile_end, dllm_block_size, q_offset);
  uint32_t effective_chunk_size;
  if (kv_valid_end <= chunk_start) {
    effective_chunk_size = 0;
  } else {
    effective_chunk_size = min(chunk_size, kv_valid_end - chunk_start);
  }
  return (effective_chunk_size + CTA_TILE_KV - 1) / CTA_TILE_KV;
}

__device__ __forceinline__ uint32_t block_expanding_mask_iteration(
    uint32_t q_tile_start, uint32_t chunk_start, uint32_t chunk_size,
    uint32_t dllm_block_size, uint32_t CTA_TILE_KV, uint32_t q_offset = 0, uint32_t kv_offset = 0) {
  uint32_t q_global_start = q_offset + q_tile_start;
  uint32_t q_first_block_id = q_global_start / dllm_block_size;
  // Consider kv_offset: kv_global < (q_block + 1) * B
  // kv_offset + kv_local < (q_block + 1) * B
  // kv_local < (q_block + 1) * B - kv_offset
  int64_t max_kv_global = (int64_t)(q_first_block_id + 1) * dllm_block_size;
  int64_t kv_fully_visible_end_local = max_kv_global - (int64_t)kv_offset;
  uint32_t kv_fully_visible_end = (kv_fully_visible_end_local > 0) ? (uint32_t)kv_fully_visible_end_local : 0;
  uint32_t kv_fully_visible_in_chunk;
  if (kv_fully_visible_end <= chunk_start) {
    kv_fully_visible_in_chunk = 0;
  } else {
    kv_fully_visible_in_chunk = min(chunk_size, kv_fully_visible_end - chunk_start);
  }
  return kv_fully_visible_in_chunk / CTA_TILE_KV;
}

__device__ __forceinline__ bool block_expanding_needs_mask(
    uint32_t iter, uint32_t mask_iteration, uint32_t num_iterations) {
  return iter >= mask_iteration && iter < num_iterations;
}

// MMA tile level check
__device__ __forceinline__ bool block_expanding_tile_fully_visible(
    uint32_t q_start, uint32_t q_end, uint32_t kv_start, uint32_t kv_end,
    uint32_t log2_block_size) {
  uint32_t q_first_block = q_start >> log2_block_size;
  uint32_t k_last_block = (kv_end - 1) >> log2_block_size;
  return q_first_block >= k_last_block;
}

__device__ __forceinline__ bool block_expanding_tile_fully_masked(
    uint32_t q_end, uint32_t kv_start, uint32_t log2_block_size) {
  uint32_t q_last_block = (q_end - 1) >> log2_block_size;
  uint32_t k_first_block = kv_start >> log2_block_size;
  return q_last_block < k_first_block;
}

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_BLOCK_EXPANDING_PREFILL_CUH_
