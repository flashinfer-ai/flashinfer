/*
 * Copyright (c) 2026 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer {

// One warp per (KV head, query token). In particular, speculative queries
// are NOT folded together: their selections and causal lengths are independent.
// Only metadata is reordered; the attention kernel reads the original KV pool.
__global__ void PrepareMSADecodeMetadata(
    const int32_t* topk, const int32_t* block_table, const int32_t* seq_lens, const float* k_scale,
    int32_t* sparse_pages, int32_t* sparse_lens, float* qk_scale_log2, int total_q,
    int num_kv_heads, int query_len, int max_pages, int num_physical_pages, float sm_scale,
    int64_t topk_stride_h, int64_t topk_stride_q, int64_t topk_stride_k, int64_t table_stride_b,
    int64_t table_stride_p, int64_t lengths_stride) {
  const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (row >= total_q * num_kv_heads) return;
  const int token = row % total_q;
  const int head = row / total_q;
  const int batch = token / query_len;
  const int kv_len = max(0, seq_lens[batch * lengths_stride] - query_len + token % query_len + 1);
  const int valid_slots = min(16, (kv_len + 127) / 128);
  // The indexer can return out[:, :total_q, :] from a larger graph buffer.
  // Its head stride is then capacity * 16, not total_q * 16.
  int logical = lane < valid_slots
                    ? topk[head * topk_stride_h + token * topk_stride_q + lane * topk_stride_k]
                    : -1;
  const bool valid_logical = logical >= 0 && logical < max_pages && logical * 128 < kv_len;
  int physical =
      valid_logical ? block_table[batch * table_stride_b + logical * table_stride_p] : -1;
  const bool valid = valid_logical && physical >= 0 && physical < num_physical_pages;
  const int length = valid ? min(128, kv_len - logical * 128) : 0;

  // Rank the 16 entries by logical position, with the partially valid page
  // last. Physical page IDs need not be ordered (prefix sharing is allowed).
  // The lane tie-break makes writes well-defined even on duplicate indices.
  int rank = 0;
  int surviving = 0;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int other_logical = __shfl_sync(0xffffffff, logical, i);
    const int other_length = __shfl_sync(0xffffffff, length, i);
    rank += other_length > 0 && (other_logical < logical || (other_logical == logical && i < lane));
    surviving += other_length;
  }
  // Always overwrite the whole output, including graph replay's inactive tail.
  if (lane < 16) sparse_pages[int64_t(row) * 16 + lane] = 0;
  __syncwarp();
  if (valid) sparse_pages[int64_t(row) * 16 + rank] = physical;
  if (lane == 0) sparse_lens[row] = surviving;
  if (row == 0 && lane == 0) qk_scale_log2[0] = k_scale[0] * sm_scale * 1.4426950408889634f;
}

}  // namespace flashinfer
