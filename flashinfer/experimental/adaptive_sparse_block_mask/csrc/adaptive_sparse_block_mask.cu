/*
 * Copyright (C) 2026 Tencent.
 * Copyright (c) 2026 by FlashInfer team.
 *
 * The adaptive policy is derived from the Stem TPD kernel in Tencent
 * hpc-ops, distributed under the MIT License.  FlashInfer integration and
 * TVM-FFI validation are distributed under the Apache License 2.0.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace adaptive_sparse_mask {

__device__ __forceinline__ int WarpSum(int value) {
#pragma unroll
  for (int offset = 16; offset >= 1; offset /= 2) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ uint16_t Bf16ToOrdered(uint16_t bits) {
  if ((bits & 0x7fff) == 0) {
    return 0x8000;
  }
  return (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits ^ 0x8000);
}

__device__ __forceinline__ bool Bf16IsFinite(uint16_t bits) { return (bits & 0x7f80) != 0x7f80; }

__device__ __forceinline__ int ComputeBudget(int q_row, int64_t key_offset, int64_t prompt_blocks,
                                             float alpha, float medium_rate, int medium_bias,
                                             float large_rate, int large_bias) {
  constexpr int64_t kSmallPromptBlocks = 56;
  constexpr int64_t kMediumPromptBlocks = 160;
  int64_t budget;
  if (prompt_blocks < kSmallPromptBlocks) {
    budget = prompt_blocks;
  } else if (prompt_blocks < kMediumPromptBlocks) {
    const double scaled = static_cast<double>(prompt_blocks) * medium_rate + medium_bias;
    budget = static_cast<int64_t>(min(scaled, static_cast<double>(prompt_blocks)));
  } else {
    const double scaled = static_cast<double>(prompt_blocks) * large_rate + large_bias;
    budget = static_cast<int64_t>(min(scaled, static_cast<double>(prompt_blocks)));
  }
  budget = budget < 1 ? 1 : (budget > prompt_blocks ? prompt_blocks : budget);

  const int64_t query_position = static_cast<int64_t>(q_row) + key_offset;
  const int64_t decay_length = prompt_blocks - budget;
  if (query_position < budget || decay_length <= 1) {
    return static_cast<int>(budget);
  }
  const double t =
      static_cast<double>(query_position - budget) / static_cast<double>(decay_length - 1);
  const int64_t decayed = static_cast<int64_t>(
      floor(static_cast<double>(budget) + t * (static_cast<double>(budget) * alpha - budget)));
  const int64_t clamped = decayed < 1 ? 1 : (decayed > budget ? budget : decayed);
  return static_cast<int>(clamped);
}

template <int kItemsPerThread, int kWarpsPerRow>
__device__ __forceinline__ uint16_t FindThreshold(const uint16_t (&ordered)[kItemsPerThread],
                                                  int budget, int* shared_warp_counts,
                                                  int warp_in_row) {
  const int lane = threadIdx.x & 31;
  uint16_t threshold = 0;
#pragma unroll 1
  for (int bit = 15; bit >= 0; --bit) {
    const uint16_t candidate = threshold | (1u << bit);
    int local_count = 0;
#pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
      local_count += ordered[item] >= candidate;
    }

    int total;
    if constexpr (kWarpsPerRow == 1) {
      total = WarpSum(local_count);
    } else {
      const int warp_count = WarpSum(local_count);
      if (lane == 0) {
        shared_warp_counts[warp_in_row] = warp_count;
      }
      __syncthreads();
      total = 0;
#pragma unroll
      for (int warp = 0; warp < kWarpsPerRow; ++warp) {
        total += shared_warp_counts[warp];
      }
      __syncthreads();
    }
    if (total >= budget) {
      threshold = candidate;
    }
  }
  return threshold;
}

template <int kItemsPerThread, int kWarpsPerRow, int kWarpsPerCta>
__global__ void AdaptiveSparseBlockMaskKernel(const __nv_bfloat16* block_logits, uint8_t* mask,
                                              const int32_t* q_seq_lens, const int32_t* kv_seq_lens,
                                              const int32_t* num_prompt_tokens, float alpha,
                                              int block_size, int initial_blocks, int window_size,
                                              float medium_rate, int medium_bias, float large_rate,
                                              int large_bias, int num_heads, int max_q_blocks,
                                              int max_k_blocks) {
  static_assert(kWarpsPerRow == 1 || kWarpsPerCta == kWarpsPerRow);
  constexpr int kRowsPerCta = kWarpsPerCta / kWarpsPerRow;
  constexpr int kThreadsPerRow = kWarpsPerRow * 32;
  constexpr uint16_t kNegativeInfinityBits = 0xff80;
  constexpr uint16_t kNonFiniteOrdered = 0x007f;

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int row_group = kWarpsPerRow == 1 ? warp : 0;
  const int warp_in_row = kWarpsPerRow == 1 ? 0 : warp;
  const int q_row = blockIdx.x * kRowsPerCta + row_group;
  const int head = blockIdx.y;
  const int request = blockIdx.z;

  const int64_t q_length = q_seq_lens[request] > 0 ? q_seq_lens[request] : 0;
  const int64_t kv_length = kv_seq_lens[request] > 0 ? kv_seq_lens[request] : 0;
  const int64_t prompt_length = num_prompt_tokens[request] > 0 ? num_prompt_tokens[request] : 0;
  const int64_t q_blocks_unclamped = (q_length + block_size - 1) / block_size;
  const int64_t k_blocks_unclamped = (kv_length + block_size - 1) / block_size;
  const int q_blocks =
      static_cast<int>(q_blocks_unclamped < max_q_blocks ? q_blocks_unclamped : max_q_blocks);
  const int k_blocks =
      static_cast<int>(k_blocks_unclamped < max_k_blocks ? k_blocks_unclamped : max_k_blocks);
  const int64_t prompt_blocks_unclamped = (prompt_length + block_size - 1) / block_size;
  const int64_t prompt_blocks = prompt_blocks_unclamped > 1 ? prompt_blocks_unclamped : 1;
  const int64_t token_offset = kv_length > q_length ? kv_length - q_length : 0;
  const int64_t key_offset = (token_offset + block_size - 1) / block_size;
  if (q_blocks == 0 || k_blocks == 0 || q_row >= q_blocks) {
    return;
  }

  const int64_t row_offset = (static_cast<int64_t>(request) * num_heads * max_q_blocks +
                              static_cast<int64_t>(head) * max_q_blocks + q_row) *
                             max_k_blocks;
  const __nv_bfloat16* row = block_logits + row_offset;
  uint8_t* mask_row = mask + row_offset;
  const int linear_thread = warp_in_row * 32 + lane;

  uint16_t ordered[kItemsPerThread];
  int local_finite = 0;
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int column = linear_thread + item * kThreadsPerRow;
    const uint16_t bits =
        column < k_blocks ? __bfloat16_as_ushort(row[column]) : kNegativeInfinityBits;
    const bool finite = Bf16IsFinite(bits);
    ordered[item] = finite ? Bf16ToOrdered(bits) : kNonFiniteOrdered;
    local_finite += finite;
  }

  const int budget = ComputeBudget(q_row, key_offset, prompt_blocks, alpha, medium_rate,
                                   medium_bias, large_rate, large_bias);
  __shared__ int shared_warp_counts[kWarpsPerRow];
  int total_finite;
  if constexpr (kWarpsPerRow == 1) {
    total_finite = WarpSum(local_finite);
  } else {
    const int warp_finite = WarpSum(local_finite);
    if (lane == 0) {
      shared_warp_counts[warp_in_row] = warp_finite;
    }
    __syncthreads();
    total_finite = 0;
#pragma unroll
    for (int warp_index = 0; warp_index < kWarpsPerRow; ++warp_index) {
      total_finite += shared_warp_counts[warp_index];
    }
    __syncthreads();
  }

  uint16_t threshold = kNonFiniteOrdered + 1;
  if (budget < total_finite) {
    threshold = FindThreshold<kItemsPerThread, kWarpsPerRow>(ordered, budget, shared_warp_counts,
                                                             warp_in_row);
  }

  const int64_t diagonal_unclamped = static_cast<int64_t>(q_row) + key_offset;
  const int diagonal =
      static_cast<int>(diagonal_unclamped < k_blocks - 1 ? diagonal_unclamped : k_blocks - 1);
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int column = linear_thread + item * kThreadsPerRow;
    if (column >= k_blocks) {
      continue;
    }
    bool selected = ordered[item] >= threshold;
    selected |= column < initial_blocks;
    selected |= column <= diagonal && column > diagonal - window_size;
    selected |= column == diagonal;
    mask_row[column] = static_cast<uint8_t>(selected);
  }
}

template <int kItemsPerThread, int kWarpsPerRow, int kWarpsPerCta>
void Launch(const __nv_bfloat16* block_logits, uint8_t* mask, const int32_t* q_seq_lens,
            const int32_t* kv_seq_lens, const int32_t* num_prompt_tokens, int batch_size,
            int num_heads, int max_q_blocks, int max_k_blocks, int block_size, float alpha,
            int initial_blocks, int window_size, float medium_rate, int medium_bias,
            float large_rate, int large_bias, cudaStream_t stream) {
  constexpr int kRowsPerCta = kWarpsPerCta / kWarpsPerRow;
  const dim3 grid((max_q_blocks + kRowsPerCta - 1) / kRowsPerCta, num_heads, batch_size);
  AdaptiveSparseBlockMaskKernel<kItemsPerThread, kWarpsPerRow, kWarpsPerCta>
      <<<grid, kWarpsPerCta * 32, 0, stream>>>(block_logits, mask, q_seq_lens, kv_seq_lens,
                                               num_prompt_tokens, alpha, block_size, initial_blocks,
                                               window_size, medium_rate, medium_bias, large_rate,
                                               large_bias, num_heads, max_q_blocks, max_k_blocks);
}

cudaError_t Run(const __nv_bfloat16* block_logits, uint8_t* mask, const int32_t* q_seq_lens,
                const int32_t* kv_seq_lens, const int32_t* num_prompt_tokens, int batch_size,
                int num_heads, int max_q_blocks, int max_k_blocks, int block_size, float alpha,
                int initial_blocks, int window_size, float medium_rate, int medium_bias,
                float large_rate, int large_bias, cudaStream_t stream) {
  if (max_k_blocks <= 1024) {
    Launch<32, 1, 8>(block_logits, mask, q_seq_lens, kv_seq_lens, num_prompt_tokens, batch_size,
                     num_heads, max_q_blocks, max_k_blocks, block_size, alpha, initial_blocks,
                     window_size, medium_rate, medium_bias, large_rate, large_bias, stream);
  } else if (max_k_blocks <= 2048) {
    Launch<32, 2, 2>(block_logits, mask, q_seq_lens, kv_seq_lens, num_prompt_tokens, batch_size,
                     num_heads, max_q_blocks, max_k_blocks, block_size, alpha, initial_blocks,
                     window_size, medium_rate, medium_bias, large_rate, large_bias, stream);
  } else if (max_k_blocks <= 4096) {
    Launch<32, 4, 4>(block_logits, mask, q_seq_lens, kv_seq_lens, num_prompt_tokens, batch_size,
                     num_heads, max_q_blocks, max_k_blocks, block_size, alpha, initial_blocks,
                     window_size, medium_rate, medium_bias, large_rate, large_bias, stream);
  } else if (max_k_blocks <= 8192) {
    Launch<32, 8, 8>(block_logits, mask, q_seq_lens, kv_seq_lens, num_prompt_tokens, batch_size,
                     num_heads, max_q_blocks, max_k_blocks, block_size, alpha, initial_blocks,
                     window_size, medium_rate, medium_bias, large_rate, large_bias, stream);
  } else {
    Launch<128, 8, 8>(block_logits, mask, q_seq_lens, kv_seq_lens, num_prompt_tokens, batch_size,
                      num_heads, max_q_blocks, max_k_blocks, block_size, alpha, initial_blocks,
                      window_size, medium_rate, medium_bias, large_rate, large_bias, stream);
  }
  return cudaGetLastError();
}

}  // namespace adaptive_sparse_mask
}  // namespace flashinfer

void adaptive_sparse_block_mask(TensorView block_logits, TensorView q_seq_lens,
                                TensorView kv_seq_lens, TensorView num_prompt_tokens,
                                TensorView mask, int64_t block_size, double alpha,
                                int64_t initial_blocks, int64_t window_size, double medium_rate,
                                int64_t medium_bias, double large_rate, int64_t large_bias) {
  CHECK_INPUT_AND_TYPE(block_logits, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(q_seq_lens, dl_int32);
  CHECK_INPUT_AND_TYPE(kv_seq_lens, dl_int32);
  CHECK_INPUT_AND_TYPE(num_prompt_tokens, dl_int32);
  CHECK_INPUT_AND_TYPE(mask, dl_bool);
  CHECK_DIM(4, block_logits);
  CHECK_DIM(1, q_seq_lens);
  CHECK_SHAPE(kv_seq_lens, q_seq_lens);
  CHECK_SHAPE(num_prompt_tokens, q_seq_lens);
  CHECK_SHAPE(mask, block_logits);
  CHECK_DEVICE(q_seq_lens, block_logits);
  CHECK_DEVICE(kv_seq_lens, block_logits);
  CHECK_DEVICE(num_prompt_tokens, block_logits);
  CHECK_DEVICE(mask, block_logits);
  TVM_FFI_ICHECK_EQ(block_logits.size(0), q_seq_lens.size(0));
  TVM_FFI_ICHECK_GT(block_size, 0);
  TVM_FFI_ICHECK_LE(block_size, std::numeric_limits<int>::max());
  TVM_FFI_ICHECK_LE(block_logits.size(3), 32768);

  ffi::CUDADeviceGuard guard(block_logits.device().device_id);
  const cudaError_t status = flashinfer::adaptive_sparse_mask::Run(
      static_cast<const __nv_bfloat16*>(block_logits.data_ptr()),
      static_cast<uint8_t*>(mask.data_ptr()), static_cast<const int32_t*>(q_seq_lens.data_ptr()),
      static_cast<const int32_t*>(kv_seq_lens.data_ptr()),
      static_cast<const int32_t*>(num_prompt_tokens.data_ptr()),
      static_cast<int>(block_logits.size(0)), static_cast<int>(block_logits.size(1)),
      static_cast<int>(block_logits.size(2)), static_cast<int>(block_logits.size(3)),
      static_cast<int>(block_size), static_cast<float>(alpha), static_cast<int>(initial_blocks),
      static_cast<int>(window_size), static_cast<float>(medium_rate), static_cast<int>(medium_bias),
      static_cast<float>(large_rate), static_cast<int>(large_bias),
      get_stream(block_logits.device()));
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(adaptive_sparse_block_mask, adaptive_sparse_block_mask);
