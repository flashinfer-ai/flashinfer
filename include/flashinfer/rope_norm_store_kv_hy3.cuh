// Copyright (C) 2026 Tencent.
// SPDX-License-Identifier: MIT
//
// FlashInfer integration of the public HPC-Ops RoPE + optional Q/K RMSNorm
// + paged-KV-store kernel from Tencent/HPC-Ops commit
// 1cd332980ed46bd0172091c1c35d55338fcae47a. The arithmetic and launch
// decomposition preserve the upstream behavior; the SM100 specialization is
// selected only for its measured uniform one-token decode shape.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <flashinfer/fastdiv.cuh>
#include <type_traits>

namespace flashinfer::rope_norm_store_kv_hy3 {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr float kRmsNormEpsilon = 1e-6F;

template <typename T, int N>
struct Vec {
  T data[N];

  __device__ __forceinline__ T& operator[](int index) { return data[index]; }
  __device__ __forceinline__ const T& operator[](int index) const { return data[index]; }
};

template <typename T, int N>
__device__ __forceinline__ Vec<T, N> load_vec(const void* pointer) {
  constexpr int kBytes = sizeof(T) * N;
  static_assert(kBytes == 1 || kBytes == 2 || kBytes == 4 || kBytes == 8 || kBytes == 16);
  Vec<T, N> value;
  if constexpr (kBytes == 1) {
    *reinterpret_cast<uint8_t*>(&value) = *reinterpret_cast<const uint8_t*>(pointer);
  } else if constexpr (kBytes == 2) {
    *reinterpret_cast<uint16_t*>(&value) = *reinterpret_cast<const uint16_t*>(pointer);
  } else if constexpr (kBytes == 4) {
    *reinterpret_cast<uint32_t*>(&value) = *reinterpret_cast<const uint32_t*>(pointer);
  } else if constexpr (kBytes == 8) {
    *reinterpret_cast<uint64_t*>(&value) = *reinterpret_cast<const uint64_t*>(pointer);
  } else {
    *reinterpret_cast<uint4*>(&value) = *reinterpret_cast<const uint4*>(pointer);
  }
  return value;
}

template <typename T, int N>
__device__ __forceinline__ void store_vec(void* pointer, const Vec<T, N>& value) {
  constexpr int kBytes = sizeof(T) * N;
  static_assert(kBytes == 1 || kBytes == 2 || kBytes == 4 || kBytes == 8 || kBytes == 16);
  if constexpr (kBytes == 1) {
    *reinterpret_cast<uint8_t*>(pointer) = *reinterpret_cast<const uint8_t*>(&value);
  } else if constexpr (kBytes == 2) {
    *reinterpret_cast<uint16_t*>(pointer) = *reinterpret_cast<const uint16_t*>(&value);
  } else if constexpr (kBytes == 4) {
    *reinterpret_cast<uint32_t*>(pointer) = *reinterpret_cast<const uint32_t*>(&value);
  } else if constexpr (kBytes == 8) {
    *reinterpret_cast<uint64_t*>(pointer) = *reinterpret_cast<const uint64_t*>(&value);
  } else {
    *reinterpret_cast<uint4*>(pointer) = *reinterpret_cast<const uint4*>(&value);
  }
}

__device__ __forceinline__ float warp_reduce_sum_xor(float value) {
#pragma unroll
  for (int offset = 16; offset >= 1; offset /= 2) {
    value += __shfl_xor_sync(0xffffffffU, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_max_xor(float value) {
#pragma unroll
  for (int offset = 16; offset >= 1; offset /= 2) {
    value = fmaxf(__shfl_xor_sync(0xffffffffU, value, offset), value);
  }
  return value;
}

__device__ __forceinline__ void rotate_neox_pair(float& first, float& second, float cosine,
                                                 float sine) {
  const float rotated_first = first * cosine - second * sine;
  const float rotated_second = second * cosine + first * sine;
  first = rotated_first;
  second = rotated_second;
}

template <int kItemsPerThread, int kHeadDim>
__device__ __forceinline__ void rms_norm_apply(Vec<float, kItemsPerThread>& values,
                                               const float* shared_weight, int lane) {
  float sum_squares = 0.0F;
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    sum_squares += values[item] * values[item];
  }
  sum_squares = warp_reduce_sum_xor(sum_squares);
  const float inverse_rms = rsqrtf(sum_squares / kHeadDim + kRmsNormEpsilon);
  constexpr int kRoundsPerHalf = (kHeadDim / 2 + kWarpSize - 1) / kWarpSize;
#pragma unroll
  for (int round = 0; round < kRoundsPerHalf; ++round) {
    const int index = round * kWarpSize + lane;
    if (index < kHeadDim / 2) {
      values[round * 2] *= inverse_rms * shared_weight[index];
      values[round * 2 + 1] *= inverse_rms * shared_weight[index + kHeadDim / 2];
    }
  }
}

template <int kItemsPerThread>
__device__ __forceinline__ float warp_absolute_max(const Vec<float, kItemsPerThread>& values) {
  float maximum = kRmsNormEpsilon;
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    maximum = fmaxf(maximum, fabsf(values[item]));
  }
  return warp_reduce_max_xor(maximum);
}

template <int kHeadDim, int kItemsPerThread>
__device__ __forceinline__ void load_neox_head(Vec<float, kItemsPerThread>& values,
                                               const __nv_bfloat16* source, int lane) {
  constexpr int kRoundsPerHalf = (kHeadDim / 2 + kWarpSize - 1) / kWarpSize;
  static_assert(kItemsPerThread == kRoundsPerHalf * 2);
#pragma unroll
  for (int round = 0; round < kRoundsPerHalf; ++round) {
    const int index = round * kWarpSize + lane;
    if (index < kHeadDim / 2) {
      values[round * 2] = __bfloat162float(source[index]);
      values[round * 2 + 1] = __bfloat162float(source[index + kHeadDim / 2]);
    }
  }
}

__device__ __forceinline__ void divide_page_position(int position, const uint_fastdiv& page_size,
                                                     int& page_index, int& page_offset) {
  uint32_t quotient;
  uint32_t remainder;
  page_size.divmod(static_cast<uint32_t>(position), quotient, remainder);
  page_index = static_cast<int>(quotient);
  page_offset = static_cast<int>(remainder);
}

template <typename CacheType>
__device__ __forceinline__ CacheType convert_cache_value(float value) {
  if constexpr (std::is_same_v<CacheType, __nv_bfloat16>) {
    return __float2bfloat16(value);
  } else {
    return __nv_fp8_e4m3(value);
  }
}

template <typename CacheType, int kQuantPolicy, int kNumQHeads, int kNumKVHeads, int kQKHeadDim,
          int kVHeadDim, int kNormPolicy, bool kUniformDecodeFastPath = false>
__global__ void RopeNormStoreKVKernel(
    CacheType* output_q, CacheType* key_cache, CacheType* value_cache, CacheType* output_k,
    CacheType* output_v, int32_t* split_k_flag, float* output_q_scale,
    const __nv_bfloat16* packed_qkv, const float* cos_sin, const int32_t* sequence_lengths,
    const int32_t* q_indptr, const int32_t* block_table, const float* q_norm_weight,
    const float* k_norm_weight, const float* k_scale, const float* v_scale,
    const float* q_scale_inverse, float fp8_upper_bound, int max_sequence_length_aligned,
    int64_t key_cache_block_stride, int64_t value_cache_block_stride, int batch_size,
    int max_pages_per_request, int page_size, uint_fastdiv page_size_fastdiv, int num_rows,
    int num_compute_blocks, bool is_prefill) {
  static_assert(kNumQHeads % kNumKVHeads == 0);
  static_assert(kQKHeadDim == 128);
  static_assert(kVHeadDim == 128);
  constexpr bool kFP8 = std::is_same_v<CacheType, __nv_fp8_e4m3>;
  static_assert((kFP8 && (kQuantPolicy == 1 || kQuantPolicy == 2)) || (!kFP8 && kQuantPolicy == 0));

  constexpr int kQHeadsPerKVHead = kNumQHeads / kNumKVHeads;
  constexpr int kQElementsPerRow = kQHeadsPerKVHead * kQKHeadDim;
  constexpr int kKElementsPerRow = kQKHeadDim;
  constexpr int kSharedElementsPerRow = kQElementsPerRow + kKElementsPerRow;
  constexpr int kPackedRowElements =
      kNumQHeads * kQKHeadDim + kNumKVHeads * kQKHeadDim + kNumKVHeads * kVHeadDim;
  constexpr int kRoundsPerHalf = (kQKHeadDim / 2 + kWarpSize - 1) / kWarpSize;
  constexpr int kItemsPerThread = kRoundsPerHalf * 2;

  const int thread_id = threadIdx.x;
  const int warp_id = thread_id / kWarpSize;
  const int lane = thread_id % kWarpSize;
  const int block_x = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int q_head_begin = kv_head * kQHeadsPerKVHead;

  __shared__ float shared_cos_sin[kWarpsPerBlock][kQKHeadDim];
  __shared__ float shared_q_norm_weight[kQKHeadDim];
  __shared__ float shared_k_norm_weight[kQKHeadDim];
  __shared__ int32_t shared_request_id[kWarpsPerBlock];
  __shared__ int32_t shared_position[kWarpsPerBlock];
  extern __shared__ __nv_bfloat16 shared_qk[];

  // The tail of grid.x has one clear CTA per request.  grid.y partitions the
  // cache by KV head, exactly as in the public optimized HPC-Ops kernel.
  if constexpr (!kUniformDecodeFastPath) {
    if (block_x >= num_compute_blocks) {
      const int request_id = block_x - num_compute_blocks;
      if (request_id >= batch_size) {
        return;
      }
      const int last_position = sequence_lengths[request_id] - 1;
      if (last_position < 0) {
        return;
      }
      int logical_page = 0;
      int position_in_page = 0;
      divide_page_position(last_position, page_size_fastdiv, logical_page, position_in_page);
      const int physical_page = block_table[request_id * max_pages_per_request + logical_page];
      const int first_zero_row = position_in_page + 1;
      constexpr int kElementsPerVector = 16 / sizeof(CacheType);
      Vec<CacheType, kElementsPerVector> zero_vector{};
      for (int row = first_zero_row + warp_id; row < page_size; row += kWarpsPerBlock) {
        CacheType* key_row = key_cache +
                             static_cast<int64_t>(physical_page) * key_cache_block_stride +
                             row * (kNumKVHeads * kQKHeadDim) + kv_head * kQKHeadDim;
        CacheType* value_row = value_cache +
                               static_cast<int64_t>(physical_page) * value_cache_block_stride +
                               row * (kNumKVHeads * kVHeadDim) + kv_head * kVHeadDim;
        for (int index = lane * kElementsPerVector; index < kQKHeadDim;
             index += kWarpSize * kElementsPerVector) {
          store_vec(key_row + index, zero_vector);
        }
        for (int index = lane * kElementsPerVector; index < kVHeadDim;
             index += kWarpSize * kElementsPerVector) {
          store_vec(value_row + index, zero_vector);
        }
      }
      return;
    }
  }

  const int row = block_x * kWarpsPerBlock + warp_id;
  if constexpr (!kUniformDecodeFastPath) {
    if (thread_id < kWarpsPerBlock) {
      shared_request_id[thread_id] = -1;
      shared_position[thread_id] = -1;
    }
    __syncthreads();

    // Parallel segment lookup retained from the public H20 implementation.  It
    // also handles repeated q_indptr entries used for CUDA-graph padding.
    constexpr int kSearchThreads = kWarpSize * kWarpsPerBlock;
    const int search_rounds = (batch_size + kSearchThreads - 1) / kSearchThreads;
    for (int search_round = 0; search_round < search_rounds; ++search_round) {
      const int candidate_request = search_round * kSearchThreads + thread_id;
      if (candidate_request < batch_size) {
        const int q_begin = q_indptr[candidate_request];
        const int q_end = q_indptr[candidate_request + 1];
#pragma unroll
        for (int local_warp = 0; local_warp < kWarpsPerBlock; ++local_warp) {
          const int candidate_row = block_x * kWarpsPerBlock + local_warp;
          if (q_begin <= candidate_row && candidate_row < q_end) {
            shared_request_id[local_warp] = candidate_request;
            shared_position[local_warp] = candidate_row + sequence_lengths[candidate_request] -
                                          q_indptr[candidate_request + 1];
          }
        }
      }
    }
  }

  if constexpr (kNormPolicy > 0) {
    constexpr int kFloatElementsPerVector = 16 / sizeof(float);
    constexpr int kWeightVectors = kQKHeadDim / kFloatElementsPerVector;
    if (thread_id < kWeightVectors) {
      const int offset = thread_id * kFloatElementsPerVector;
      store_vec(shared_q_norm_weight + offset,
                load_vec<float, kFloatElementsPerVector>(q_norm_weight + offset));
      store_vec(shared_k_norm_weight + offset,
                load_vec<float, kFloatElementsPerVector>(k_norm_weight + offset));
    }
  }

  // Stage this CTA's GQA group and one K head using 16-byte transactions.
  constexpr int kBF16ElementsPerVector = 16 / sizeof(__nv_bfloat16);
  constexpr int kQVectors = kQElementsPerRow / kBF16ElementsPerVector;
  constexpr int kKVectors = kKElementsPerRow / kBF16ElementsPerVector;
  constexpr int kQLoadRounds = (kQVectors + kWarpSize - 1) / kWarpSize;
  constexpr int kKLoadRounds = (kKVectors + kWarpSize - 1) / kWarpSize;
  if (row < num_rows) {
    const __nv_bfloat16* source_q =
        packed_qkv + static_cast<int64_t>(row) * kPackedRowElements + q_head_begin * kQKHeadDim;
    const __nv_bfloat16* source_k = packed_qkv + static_cast<int64_t>(row) * kPackedRowElements +
                                    kNumQHeads * kQKHeadDim + kv_head * kQKHeadDim;
    __nv_bfloat16* destination_q = shared_qk + warp_id * kSharedElementsPerRow;
    __nv_bfloat16* destination_k = destination_q + kQElementsPerRow;
#pragma unroll
    for (int round = 0; round < kQLoadRounds; ++round) {
      const int offset = (round * kWarpSize + lane) * kBF16ElementsPerVector;
      if (offset < kQElementsPerRow) {
        store_vec(destination_q + offset,
                  load_vec<__nv_bfloat16, kBF16ElementsPerVector>(source_q + offset));
      }
    }
#pragma unroll
    for (int round = 0; round < kKLoadRounds; ++round) {
      const int offset = (round * kWarpSize + lane) * kBF16ElementsPerVector;
      if (offset < kKElementsPerRow) {
        store_vec(destination_k + offset,
                  load_vec<__nv_bfloat16, kBF16ElementsPerVector>(source_k + offset));
      }
    }
  }
  __syncthreads();

  if (row >= num_rows) {
    return;
  }
  // The host only enables the fused-tail specialization after the caller has
  // explicitly guaranteed q_indptr == [0, 1, ..., batch_size].  Compile the
  // general segment search out of that decode-only path.
  const int request_id = kUniformDecodeFastPath ? row : shared_request_id[warp_id];
  const int position =
      kUniformDecodeFastPath ? sequence_lengths[row] - 1 : shared_position[warp_id];
  if (position < 0) {
    return;
  }

  constexpr int kFloatElementsPerVector = 16 / sizeof(float);
  constexpr int kCosSinVectors = kQKHeadDim / kFloatElementsPerVector;
  const float* cos_sin_row = cos_sin + static_cast<int64_t>(position) * kQKHeadDim;
  Vec<float, kRoundsPerHalf> register_cosine;
  Vec<float, kRoundsPerHalf> register_sine;
  if constexpr (kUniformDecodeFastPath) {
#pragma unroll
    for (int round = 0; round < kRoundsPerHalf; ++round) {
      const int index = round * kWarpSize + lane;
      register_cosine[round] = cos_sin_row[index];
      register_sine[round] = cos_sin_row[index + kQKHeadDim / 2];
    }
  } else {
    if (lane < kCosSinVectors) {
      const int offset = lane * kFloatElementsPerVector;
      store_vec(shared_cos_sin[warp_id] + offset,
                load_vec<float, kFloatElementsPerVector>(cos_sin_row + offset));
    }
    __syncwarp();
  }

  int logical_page = 0;
  int position_in_page = 0;
  divide_page_position(position, page_size_fastdiv, logical_page, position_in_page);
  const int physical_page = block_table[request_id * max_pages_per_request + logical_page];
  CacheType* key_cache_row = key_cache +
                             static_cast<int64_t>(physical_page) * key_cache_block_stride +
                             position_in_page * (kNumKVHeads * kQKHeadDim);
  CacheType* value_cache_row = value_cache +
                               static_cast<int64_t>(physical_page) * value_cache_block_stride +
                               position_in_page * (kNumKVHeads * kVHeadDim);
  const __nv_bfloat16* shared_row = shared_qk + warp_id * kSharedElementsPerRow;
  const __nv_bfloat16* global_row = packed_qkv + static_cast<int64_t>(row) * kPackedRowElements;

#pragma unroll
  for (int local_q_head = 0; local_q_head < kQHeadsPerKVHead; ++local_q_head) {
    const int q_head = q_head_begin + local_q_head;
    const __nv_bfloat16* source = shared_row + local_q_head * kQKHeadDim;
    CacheType* destination =
        output_q + static_cast<int64_t>(row) * kNumQHeads * kQKHeadDim + q_head * kQKHeadDim;
    Vec<float, kItemsPerThread> values{};
    load_neox_head<kQKHeadDim>(values, source, lane);
    if constexpr (kNormPolicy == 2) {
      rms_norm_apply<kItemsPerThread, kQKHeadDim>(values, shared_q_norm_weight, lane);
    }
#pragma unroll
    for (int round = 0; round < kRoundsPerHalf; ++round) {
      const int index = round * kWarpSize + lane;
      if (index < kQKHeadDim / 2) {
        const float cosine =
            kUniformDecodeFastPath ? register_cosine[round] : shared_cos_sin[warp_id][index];
        const float sine = kUniformDecodeFastPath ? register_sine[round]
                                                  : shared_cos_sin[warp_id][index + kQKHeadDim / 2];
        rotate_neox_pair(values[round * 2], values[round * 2 + 1], cosine, sine);
      }
    }
    if constexpr (kNormPolicy == 1) {
      rms_norm_apply<kItemsPerThread, kQKHeadDim>(values, shared_q_norm_weight, lane);
    }

    float multiplier = 1.0F;
    if constexpr (kQuantPolicy == 1) {
      const float maximum = warp_absolute_max(values);
      const float scale = maximum / fp8_upper_bound;
      if (lane == 0) {
        if constexpr (kUniformDecodeFastPath) {
          output_q_scale[row * kNumQHeads + q_head] = scale;
        } else if (is_prefill) {
          const int token_in_request = row - q_indptr[request_id];
          output_q_scale[request_id * kNumQHeads * max_sequence_length_aligned +
                         q_head * max_sequence_length_aligned + token_in_request] = scale;
        } else {
          output_q_scale[row * kNumQHeads + q_head] = scale;
        }
      }
      multiplier = __frcp_rn(scale);
    } else if constexpr (kQuantPolicy == 2) {
      multiplier = q_scale_inverse[0];
    }

#pragma unroll
    for (int round = 0; round < kRoundsPerHalf; ++round) {
      const int index = round * kWarpSize + lane;
      if (index < kQKHeadDim / 2) {
        destination[index] = convert_cache_value<CacheType>(values[round * 2] * multiplier);
        destination[index + kQKHeadDim / 2] =
            convert_cache_value<CacheType>(values[round * 2 + 1] * multiplier);
      }
    }
  }

  float key_multiplier = 1.0F;
  if constexpr (kFP8) {
    key_multiplier = __frcp_rn(k_scale[0]);
    if ((kUniformDecodeFastPath || row == q_indptr[request_id]) && lane == 0) {
      split_k_flag[request_id * kNumKVHeads + kv_head] = 0;
    }
  }
  {
    const __nv_bfloat16* source = shared_row + kQElementsPerRow;
    CacheType* destination =
        output_k != nullptr
            ? output_k + static_cast<int64_t>(row) * kNumKVHeads * kQKHeadDim + kv_head * kQKHeadDim
            : key_cache_row + kv_head * kQKHeadDim;
    Vec<float, kItemsPerThread> values{};
    load_neox_head<kQKHeadDim>(values, source, lane);
    if constexpr (kNormPolicy == 2) {
      rms_norm_apply<kItemsPerThread, kQKHeadDim>(values, shared_k_norm_weight, lane);
    }
#pragma unroll
    for (int round = 0; round < kRoundsPerHalf; ++round) {
      const int index = round * kWarpSize + lane;
      if (index < kQKHeadDim / 2) {
        const float cosine =
            kUniformDecodeFastPath ? register_cosine[round] : shared_cos_sin[warp_id][index];
        const float sine = kUniformDecodeFastPath ? register_sine[round]
                                                  : shared_cos_sin[warp_id][index + kQKHeadDim / 2];
        rotate_neox_pair(values[round * 2], values[round * 2 + 1], cosine, sine);
      }
    }
    if constexpr (kNormPolicy == 1) {
      rms_norm_apply<kItemsPerThread, kQKHeadDim>(values, shared_k_norm_weight, lane);
    }
#pragma unroll
    for (int round = 0; round < kRoundsPerHalf; ++round) {
      const int index = round * kWarpSize + lane;
      if (index < kQKHeadDim / 2) {
        destination[index] = convert_cache_value<CacheType>(values[round * 2] * key_multiplier);
        destination[index + kQKHeadDim / 2] =
            convert_cache_value<CacheType>(values[round * 2 + 1] * key_multiplier);
      }
    }
  }

  constexpr int kBF16sPerVector = 16 / sizeof(__nv_bfloat16);
  constexpr int kVLoadVectors = kVHeadDim / kBF16sPerVector;
  constexpr int kVLoadRounds = (kVLoadVectors + kWarpSize - 1) / kWarpSize;
  const __nv_bfloat16* value_source =
      global_row + (kNumQHeads + kNumKVHeads) * kQKHeadDim + kv_head * kVHeadDim;
  CacheType* value_destination =
      output_v != nullptr
          ? output_v + static_cast<int64_t>(row) * kNumKVHeads * kVHeadDim + kv_head * kVHeadDim
          : value_cache_row + kv_head * kVHeadDim;
  if constexpr (!kFP8) {
#pragma unroll
    for (int round = 0; round < kVLoadRounds; ++round) {
      const int offset = (round * kWarpSize + lane) * kBF16sPerVector;
      if (offset < kVHeadDim) {
        store_vec(value_destination + offset,
                  load_vec<__nv_bfloat16, kBF16sPerVector>(value_source + offset));
      }
    }
  } else {
    const float value_multiplier = __frcp_rn(v_scale[0]);
#pragma unroll
    for (int round = 0; round < kVLoadRounds; ++round) {
      const int offset = (round * kWarpSize + lane) * kBF16sPerVector;
      if (offset < kVHeadDim) {
        const auto packed_bf16 =
            load_vec<__nv_bfloat162, kBF16sPerVector / 2>(value_source + offset);
        Vec<float, kBF16sPerVector> values;
#pragma unroll
        for (int pair = 0; pair < kBF16sPerVector / 2; ++pair) {
          const float2 converted = __bfloat1622float2(packed_bf16[pair]);
          values[pair * 2] = converted.x * value_multiplier;
          values[pair * 2 + 1] = converted.y * value_multiplier;
        }
        Vec<__nv_fp8x4_e4m3, kBF16sPerVector / 4> packed_fp8;
#pragma unroll
        for (int pack = 0; pack < kBF16sPerVector / 4; ++pack) {
          packed_fp8[pack] = __nv_fp8x4_e4m3(make_float4(
              values[pack * 4], values[pack * 4 + 1], values[pack * 4 + 2], values[pack * 4 + 3]));
        }
        store_vec(value_destination + offset, packed_fp8);
      }
    }
  }

  // Uniform one-token decode gives every warp exclusive ownership of one
  // request/KV-head pair.  The target-only specialization uses that ownership
  // to clear the remainder of the last cache page without launching the
  // source-faithful batch_size * num_kv_heads clear CTAs.  The reference
  // specialization compiles this block out entirely.
  if constexpr (kUniformDecodeFastPath) {
    constexpr int kElementsPerVector = 16 / sizeof(CacheType);
    Vec<CacheType, kElementsPerVector> zero_vector{};
    if constexpr (sizeof(CacheType) == 1) {
      constexpr int kLanesPerRow = kQKHeadDim / kElementsPerVector;
      constexpr int kRowsPerIteration = kWarpSize / kLanesPerRow;
      static_assert(kLanesPerRow == 8 && kRowsPerIteration == 4);
      static_assert(kQKHeadDim == kVHeadDim);
      const int row_in_iteration = lane / kLanesPerRow;
      const int vector_in_row = lane % kLanesPerRow;
      for (int tail_base = position_in_page + 1; tail_base < page_size;
           tail_base += kRowsPerIteration) {
        const int tail_row = tail_base + row_in_iteration;
        if (tail_row < page_size) {
          CacheType* key_row = key_cache +
                               static_cast<int64_t>(physical_page) * key_cache_block_stride +
                               tail_row * (kNumKVHeads * kQKHeadDim) + kv_head * kQKHeadDim;
          CacheType* value_row = value_cache +
                                 static_cast<int64_t>(physical_page) * value_cache_block_stride +
                                 tail_row * (kNumKVHeads * kVHeadDim) + kv_head * kVHeadDim;
          const int index = vector_in_row * kElementsPerVector;
          store_vec(key_row + index, zero_vector);
          store_vec(value_row + index, zero_vector);
        }
      }
    } else {
      for (int tail_row = position_in_page + 1; tail_row < page_size; ++tail_row) {
        CacheType* key_row = key_cache +
                             static_cast<int64_t>(physical_page) * key_cache_block_stride +
                             tail_row * (kNumKVHeads * kQKHeadDim) + kv_head * kQKHeadDim;
        CacheType* value_row = value_cache +
                               static_cast<int64_t>(physical_page) * value_cache_block_stride +
                               tail_row * (kNumKVHeads * kVHeadDim) + kv_head * kVHeadDim;
        for (int index = lane * kElementsPerVector; index < kQKHeadDim;
             index += kWarpSize * kElementsPerVector) {
          store_vec(key_row + index, zero_vector);
        }
        for (int index = lane * kElementsPerVector; index < kVHeadDim;
             index += kWarpSize * kElementsPerVector) {
          store_vec(value_row + index, zero_vector);
        }
      }
    }
  }
}

template <typename CacheType, int kQuantPolicy, int kNumQHeads, int kNumKVHeads, int kNormPolicy,
          bool kUniformDecodeFastPath = false>
cudaError_t launch_specialized(
    CacheType* output_q, CacheType* key_cache, CacheType* value_cache, CacheType* output_k,
    CacheType* output_v, int32_t* split_k_flag, float* output_q_scale,
    const __nv_bfloat16* packed_qkv, const float* cos_sin, const int32_t* sequence_lengths,
    const int32_t* q_indptr, const int32_t* block_table, const float* q_norm_weight,
    const float* k_norm_weight, const float* k_scale, const float* v_scale,
    const float* q_scale_inverse, float fp8_upper_bound, int max_sequence_length_aligned,
    int64_t key_cache_block_stride, int64_t value_cache_block_stride, int batch_size,
    int max_pages_per_request, int page_size, int num_rows, bool is_prefill, cudaStream_t stream) {
  constexpr int kQKHeadDim = 128;
  constexpr int kVHeadDim = 128;
  constexpr int kQHeadsPerKVHead = kNumQHeads / kNumKVHeads;
  constexpr int kSharedElementsPerRow = (kQHeadsPerKVHead + 1) * kQKHeadDim;
  constexpr int kDynamicSharedBytes =
      kWarpsPerBlock * kSharedElementsPerRow * sizeof(__nv_bfloat16);
  const int num_compute_blocks = (num_rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const dim3 grid(num_compute_blocks + (kUniformDecodeFastPath ? 0 : batch_size), kNumKVHeads);
  const dim3 block(kWarpsPerBlock * kWarpSize);
  RopeNormStoreKVKernel<CacheType, kQuantPolicy, kNumQHeads, kNumKVHeads, kQKHeadDim, kVHeadDim,
                        kNormPolicy, kUniformDecodeFastPath>
      <<<grid, block, kDynamicSharedBytes, stream>>>(
          output_q, key_cache, value_cache, output_k, output_v, split_k_flag, output_q_scale,
          packed_qkv, cos_sin, sequence_lengths, q_indptr, block_table, q_norm_weight,
          k_norm_weight, k_scale, v_scale, q_scale_inverse, fp8_upper_bound,
          max_sequence_length_aligned, key_cache_block_stride, value_cache_block_stride, batch_size,
          max_pages_per_request, page_size, uint_fastdiv(static_cast<uint32_t>(page_size)),
          num_rows, num_compute_blocks, is_prefill);
  return cudaGetLastError();
}

template <typename CacheType, int kQuantPolicy, int kNumQHeads, int kNumKVHeads>
cudaError_t dispatch_uniform_decode_norm(
    CacheType* output_q, CacheType* key_cache, CacheType* value_cache, CacheType* output_k,
    CacheType* output_v, int32_t* split_k_flag, float* output_q_scale,
    const __nv_bfloat16* packed_qkv, const float* cos_sin, const int32_t* sequence_lengths,
    const int32_t* q_indptr, const int32_t* block_table, const float* q_norm_weight,
    const float* k_norm_weight, const float* k_scale, const float* v_scale,
    const float* q_scale_inverse, float fp8_upper_bound, int max_sequence_length_aligned,
    int64_t key_cache_block_stride, int64_t value_cache_block_stride, int batch_size,
    int max_pages_per_request, int page_size, int num_rows, int norm_policy, cudaStream_t stream) {
#define FLASHINFER_ROPE_LAUNCH_FUSED_NORM(NORM)                                                   \
  return launch_specialized<CacheType, kQuantPolicy, kNumQHeads, kNumKVHeads, NORM, true>(        \
      output_q, key_cache, value_cache, output_k, output_v, split_k_flag, output_q_scale,         \
      packed_qkv, cos_sin, sequence_lengths, q_indptr, block_table, q_norm_weight, k_norm_weight, \
      k_scale, v_scale, q_scale_inverse, fp8_upper_bound, max_sequence_length_aligned,            \
      key_cache_block_stride, value_cache_block_stride, batch_size, max_pages_per_request,        \
      page_size, num_rows, false, stream)
  switch (norm_policy) {
    case 0:
      FLASHINFER_ROPE_LAUNCH_FUSED_NORM(0);
    case 1:
      FLASHINFER_ROPE_LAUNCH_FUSED_NORM(1);
    case 2:
      FLASHINFER_ROPE_LAUNCH_FUSED_NORM(2);
    default:
      return cudaErrorInvalidValue;
  }
#undef FLASHINFER_ROPE_LAUNCH_FUSED_NORM
}

template <typename CacheType, int kQuantPolicy, int kNumQHeads, int kNumKVHeads>
cudaError_t dispatch_norm(CacheType* output_q, CacheType* key_cache, CacheType* value_cache,
                          CacheType* output_k, CacheType* output_v, int32_t* split_k_flag,
                          float* output_q_scale, const __nv_bfloat16* packed_qkv,
                          const float* cos_sin, const int32_t* sequence_lengths,
                          const int32_t* q_indptr, const int32_t* block_table,
                          const float* q_norm_weight, const float* k_norm_weight,
                          const float* k_scale, const float* v_scale, const float* q_scale_inverse,
                          float fp8_upper_bound, int max_sequence_length_aligned,
                          int64_t key_cache_block_stride, int64_t value_cache_block_stride,
                          int batch_size, int max_pages_per_request, int page_size, int num_rows,
                          bool is_prefill, int norm_policy, cudaStream_t stream) {
#define FLASHINFER_ROPE_LAUNCH_NORM(NORM)                                                         \
  return launch_specialized<CacheType, kQuantPolicy, kNumQHeads, kNumKVHeads, NORM>(              \
      output_q, key_cache, value_cache, output_k, output_v, split_k_flag, output_q_scale,         \
      packed_qkv, cos_sin, sequence_lengths, q_indptr, block_table, q_norm_weight, k_norm_weight, \
      k_scale, v_scale, q_scale_inverse, fp8_upper_bound, max_sequence_length_aligned,            \
      key_cache_block_stride, value_cache_block_stride, batch_size, max_pages_per_request,        \
      page_size, num_rows, is_prefill, stream)
  switch (norm_policy) {
    case 0:
      FLASHINFER_ROPE_LAUNCH_NORM(0);
    case 1:
      FLASHINFER_ROPE_LAUNCH_NORM(1);
    case 2:
      FLASHINFER_ROPE_LAUNCH_NORM(2);
    default:
      return cudaErrorInvalidValue;
  }
#undef FLASHINFER_ROPE_LAUNCH_NORM
}

template <typename CacheType, int kQuantPolicy>
cudaError_t dispatch_shape(CacheType* output_q, CacheType* key_cache, CacheType* value_cache,
                           CacheType* output_k, CacheType* output_v, int32_t* split_k_flag,
                           float* output_q_scale, const __nv_bfloat16* packed_qkv,
                           const float* cos_sin, const int32_t* sequence_lengths,
                           const int32_t* q_indptr, const int32_t* block_table,
                           const float* q_norm_weight, const float* k_norm_weight,
                           const float* k_scale, const float* v_scale, const float* q_scale_inverse,
                           float fp8_upper_bound, int max_sequence_length_aligned,
                           int64_t key_cache_block_stride, int64_t value_cache_block_stride,
                           int batch_size, int max_pages_per_request, int page_size, int num_rows,
                           int num_q_heads, int num_kv_heads, bool is_prefill, int norm_policy,
                           cudaStream_t stream) {
#define FLASHINFER_ROPE_DISPATCH_SHAPE(Q_HEADS, KV_HEADS)                                         \
  return dispatch_norm<CacheType, kQuantPolicy, Q_HEADS, KV_HEADS>(                               \
      output_q, key_cache, value_cache, output_k, output_v, split_k_flag, output_q_scale,         \
      packed_qkv, cos_sin, sequence_lengths, q_indptr, block_table, q_norm_weight, k_norm_weight, \
      k_scale, v_scale, q_scale_inverse, fp8_upper_bound, max_sequence_length_aligned,            \
      key_cache_block_stride, value_cache_block_stride, batch_size, max_pages_per_request,        \
      page_size, num_rows, is_prefill, norm_policy, stream)
  if (num_q_heads == 8 && num_kv_heads == 1) {
    FLASHINFER_ROPE_DISPATCH_SHAPE(8, 1);
  }
  if (num_q_heads == 64 && num_kv_heads == 8) {
    FLASHINFER_ROPE_DISPATCH_SHAPE(64, 8);
  }
#undef FLASHINFER_ROPE_DISPATCH_SHAPE
  return cudaErrorInvalidValue;
}

template <typename CacheType, int kQuantPolicy>
cudaError_t dispatch_uniform_decode_shape(
    CacheType* output_q, CacheType* key_cache, CacheType* value_cache, CacheType* output_k,
    CacheType* output_v, int32_t* split_k_flag, float* output_q_scale,
    const __nv_bfloat16* packed_qkv, const float* cos_sin, const int32_t* sequence_lengths,
    const int32_t* q_indptr, const int32_t* block_table, const float* q_norm_weight,
    const float* k_norm_weight, const float* k_scale, const float* v_scale,
    const float* q_scale_inverse, float fp8_upper_bound, int max_sequence_length_aligned,
    int64_t key_cache_block_stride, int64_t value_cache_block_stride, int batch_size,
    int max_pages_per_request, int page_size, int num_rows, int num_q_heads, int num_kv_heads,
    int norm_policy, cudaStream_t stream) {
#define FLASHINFER_ROPE_DISPATCH_FUSED_SHAPE(Q_HEADS, KV_HEADS)                                   \
  return dispatch_uniform_decode_norm<CacheType, kQuantPolicy, Q_HEADS, KV_HEADS>(                \
      output_q, key_cache, value_cache, output_k, output_v, split_k_flag, output_q_scale,         \
      packed_qkv, cos_sin, sequence_lengths, q_indptr, block_table, q_norm_weight, k_norm_weight, \
      k_scale, v_scale, q_scale_inverse, fp8_upper_bound, max_sequence_length_aligned,            \
      key_cache_block_stride, value_cache_block_stride, batch_size, max_pages_per_request,        \
      page_size, num_rows, norm_policy, stream)
  if (num_q_heads == 8 && num_kv_heads == 1) {
    FLASHINFER_ROPE_DISPATCH_FUSED_SHAPE(8, 1);
  }
  if (num_q_heads == 64 && num_kv_heads == 8) {
    FLASHINFER_ROPE_DISPATCH_FUSED_SHAPE(64, 8);
  }
#undef FLASHINFER_ROPE_DISPATCH_FUSED_SHAPE
  return cudaErrorInvalidValue;
}

}  // namespace flashinfer::rope_norm_store_kv_hy3
