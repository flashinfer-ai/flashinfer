// Copyright (C) 2026 Tencent.
// SPDX-License-Identifier: MIT
// HY3 sampler derived from Tencent/HPC-Ops commit
// 1cd332980ed46bd0172091c1c35d55338fcae47a and optimized for SM100/B200.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

namespace flashinfer::sampling::hy3 {

#ifdef __CUDACC__
// Call libdevice directly so this kernel retains the same expf implementation
// as the source-faithful build even when FlashInfer's generic JIT defaults to
// --use_fast_math. The dedicated JIT target restores FTZ/division flags too.
extern "C" __device__ float __nv_expf(float);
#endif

constexpr int kVocabSize = 120832;
constexpr int kSoftmaxNone = 0;
constexpr int kSoftmaxBeforeTopK = 1;
constexpr int kSoftmaxAfterTopK = 2;

#ifdef __CUDACC__

template <typename T, int N>
struct Vec {
  T data[N];
  __device__ __forceinline__ T& operator[](int i) { return data[i]; }
  __device__ __forceinline__ const T& operator[](int i) const { return data[i]; }
};

template <typename T, int N>
__device__ __forceinline__ Vec<T, N> load_vec(const T* ptr) {
  constexpr int kBytes = sizeof(T) * N;
  static_assert(kBytes == 4 || kBytes == 8 || kBytes == 16);
  Vec<T, N> out;
  if constexpr (kBytes == 4) {
    *reinterpret_cast<uint32_t*>(&out) = *reinterpret_cast<const uint32_t*>(ptr);
  } else if constexpr (kBytes == 8) {
    *reinterpret_cast<uint64_t*>(&out) = *reinterpret_cast<const uint64_t*>(ptr);
  } else {
    *reinterpret_cast<uint4*>(&out) = *reinterpret_cast<const uint4*>(ptr);
  }
  return out;
}

__device__ __forceinline__ float fast_exp(float x) {
  float out;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(x * 1.4426950408889634f));
  return out;
}

__device__ __forceinline__ float precise_exp(float x) { return __nv_expf(x); }

__device__ __forceinline__ float fast_log(float x) {
  float out;
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(x));
  return out * 0.6931471805599453f;
}

__device__ __forceinline__ float fast_rcp(float x) {
  float out;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(x));
  return out;
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, offset);
  }
  return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset));
  }
  return x;
}

__device__ __forceinline__ bool take_other(float other_score, int other_token, float score,
                                           int token) {
  if (other_score > score) return true;
  if (other_score < score || other_token < 0) return false;
  return token < 0 || other_token < token;
}

__device__ __forceinline__ float gumbel_from_uniform(float u) {
  const float inner = fmaxf(-fast_log(u), 1e-20f);
  return -fast_log(inner);
}

#endif  // __CUDACC__

cudaError_t launch_temperature(int32_t* output, const void* logits, int logits_dtype,
                               int logits_row_stride, const float* temperature,
                               float temperature_value, const float* gumbel_noise,
                               const int64_t* draft_token_ids, int batch_size, int vocab_size,
                               void* workspace, size_t workspace_size, int sm_count, uint64_t seed,
                               uint64_t rng_offset, cudaStream_t stream);

cudaError_t launch_heavy(int32_t* output, const void* logits, int logits_dtype,
                         uint8_t* penalty_mask, const int32_t* slot_id,
                         const float* repetition_penalty, float repetition_penalty_value,
                         const float* temperature, float temperature_value, int softmax_policy,
                         const void* topk, int topk_element_bytes, int topk_value,
                         const float* topp, float topp_value, const float* gumbel_noise,
                         int batch_size, int vocab_size, int penalty_rows, int penalty_row_stride,
                         int logits_row_stride, int max_topk, void* workspace,
                         size_t workspace_size, int sm_count, uint64_t seed, uint64_t rng_offset,
                         cudaStream_t stream);

}  // namespace flashinfer::sampling::hy3

// Kernel implementation retained under the Tencent/HPC-Ops MIT license.

#include <curand_kernel.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <limits>
#include <type_traits>

namespace flashinfer::sampling::hy3 {
namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

template <typename DType, int N>
__device__ __forceinline__ void load_as_float(const DType* ptr, float* out) {
  constexpr uintptr_t kAlignment = sizeof(DType) * N;
  if ((reinterpret_cast<uintptr_t>(ptr) & (kAlignment - 1)) != 0) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if constexpr (std::is_same_v<DType, float>) {
        out[i] = ptr[i];
      } else {
        out[i] = __bfloat162float(ptr[i]);
      }
    }
    return;
  }
  const auto value = load_vec<DType, N>(ptr);
#pragma unroll
  for (int i = 0; i < N; ++i) {
    if constexpr (std::is_same_v<DType, float>) {
      out[i] = value[i];
    } else {
      out[i] = __bfloat162float(value[i]);
    }
  }
}

// --------------------------------------------------------------------------
// Temperature-only path. The geometry and arithmetic intentionally mirror
// HPC-Ops first; SM100 tuning is layered on top only after parity is frozen.
// --------------------------------------------------------------------------

constexpr int kTemperatureThreads = 256;
constexpr int kTemperatureItems = 4;
constexpr int kTemperatureStride = kTemperatureThreads * kTemperatureItems;
constexpr int kTemperatureMinBlocks = 8;

template <typename DType, int VocabSize, bool HasExternalGumbel, bool HasDraftMask>
__global__ __launch_bounds__(kTemperatureThreads, 2) void temperature_kernel(
    int32_t* output, const DType* logits, int row_stride, const float* temperature,
    float temperature_value, const float* gumbel_noise, const int64_t* draft_token_ids,
    float* partial_score, int32_t* partial_token, int32_t* counters, int blocks_per_row,
    int scratch_stride, uint64_t seed, uint64_t rng_offset) {
  constexpr int kWarpCount = kTemperatureThreads / 32;
  const int bid = static_cast<int>(blockIdx.x);
  const int row_id = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid >> 5;
  const int lane = tid & 31;

  const int columns_raw = (VocabSize + blocks_per_row - 1) / blocks_per_row;
  const int columns_per_block =
      (columns_raw + kTemperatureItems - 1) / kTemperatureItems * kTemperatureItems;
  const int column_begin = bid * columns_per_block;
  const int column_end = min(column_begin + columns_per_block, VocabSize);
  const float t = temperature ? temperature[row_id] : temperature_value;
  const bool temperature_active = t > 0.0f;

  curandStatePhilox4_32_10_t rng;
  if constexpr (!HasExternalGumbel) {
    const uint64_t sequence =
        (static_cast<uint64_t>(row_id) * scratch_stride + bid) * kTemperatureThreads + tid;
    curand_init(seed, sequence, rng_offset, &rng);
  }

  const DType* row = logits + static_cast<int64_t>(row_id) * row_stride;
  const float* noise_row =
      HasExternalGumbel ? gumbel_noise + static_cast<int64_t>(row_id) * VocabSize : nullptr;

  __shared__ int32_t shared_mask_token;
  if constexpr (HasDraftMask) {
    if (tid == 0) {
      const int64_t token = draft_token_ids[row_id];
      shared_mask_token = token >= 0 && token < VocabSize ? static_cast<int32_t>(token) : VocabSize;
    }
    __syncthreads();
  }
  const int mask_token = HasDraftMask ? shared_mask_token : VocabSize;

  float best_score = kNegInf;
  int best_token = -1;
  const int base = column_begin + tid * kTemperatureItems;
  const int iterations = (columns_per_block + kTemperatureStride - 1) / kTemperatureStride;
#pragma unroll 1
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const int column_base = base + iteration * kTemperatureStride;
    if (column_base >= column_end) break;
    float values[kTemperatureItems];
    if (column_base + kTemperatureItems <= column_end) {
      load_as_float<DType, kTemperatureItems>(row + column_base, values);
    } else {
#pragma unroll
      for (int i = 0; i < kTemperatureItems; ++i) {
        const int column = column_base + i;
        if (column < column_end) {
          if constexpr (std::is_same_v<DType, float>) {
            values[i] = row[column];
          } else {
            values[i] = __bfloat162float(row[column]);
          }
        } else {
          values[i] = kNegInf;
        }
      }
    }

    float uniforms[kTemperatureItems];
    if constexpr (!HasExternalGumbel) {
      const float4 random = curand_uniform4(&rng);
      uniforms[0] = random.x;
      uniforms[1] = random.y;
      uniforms[2] = random.z;
      uniforms[3] = random.w;
    }
#pragma unroll
    for (int i = 0; i < kTemperatureItems; ++i) {
      const int column = column_base + i;
      if (column >= column_end) continue;
      // Match the heavy path: non-positive per-row values disable scaling.
      // Keep division on the positive path to preserve source-faithful
      // rounding for the normal sampling contract.
      float value = temperature_active ? values[i] / t : values[i];
      if constexpr (HasDraftMask) {
        if (column == mask_token) value = kNegInf;
      }
      const float noise = HasExternalGumbel ? noise_row[column] : gumbel_from_uniform(uniforms[i]);
      const float score = value + noise;
      if (take_other(score, column, best_score, best_token)) {
        best_score = score;
        best_token = column;
      }
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float score = __shfl_xor_sync(0xffffffff, best_score, offset);
    const int token = __shfl_xor_sync(0xffffffff, best_token, offset);
    if (take_other(score, token, best_score, best_token)) {
      best_score = score;
      best_token = token;
    }
  }

  __shared__ float warp_scores[kWarpCount];
  __shared__ int warp_tokens[kWarpCount];
  if (lane == 0) {
    warp_scores[warp_id] = best_score;
    warp_tokens[warp_id] = best_token;
  }
  __syncthreads();

  __shared__ float block_score;
  __shared__ int block_token;
  if (warp_id == 0) {
    float score = lane < kWarpCount ? warp_scores[lane] : kNegInf;
    int token = lane < kWarpCount ? warp_tokens[lane] : -1;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other_score = __shfl_xor_sync(0xffffffff, score, offset);
      const int other_token = __shfl_xor_sync(0xffffffff, token, offset);
      if (take_other(other_score, other_token, score, token)) {
        score = other_score;
        token = other_token;
      }
    }
    if (lane == 0) {
      block_score = score;
      block_token = token;
    }
  }
  __syncthreads();

  __shared__ int previous_counter;
  if (tid == 0) {
    const int scratch_index = row_id * scratch_stride + bid;
    partial_score[scratch_index] = block_score;
    partial_token[scratch_index] = block_token;
    __threadfence();
    previous_counter = atomicAdd(counters + row_id, 1);
  }
  __syncthreads();
  if (previous_counter != blocks_per_row - 1) return;

  __threadfence();
  best_score = tid < blocks_per_row ? partial_score[row_id * scratch_stride + tid] : kNegInf;
  best_token = tid < blocks_per_row ? partial_token[row_id * scratch_stride + tid] : -1;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float score = __shfl_xor_sync(0xffffffff, best_score, offset);
    const int token = __shfl_xor_sync(0xffffffff, best_token, offset);
    if (take_other(score, token, best_score, best_token)) {
      best_score = score;
      best_token = token;
    }
  }
  if (lane == 0) {
    warp_scores[warp_id] = best_score;
    warp_tokens[warp_id] = best_token;
  }
  __syncthreads();
  if (warp_id == 0) {
    float score = lane < kWarpCount ? warp_scores[lane] : kNegInf;
    int token = lane < kWarpCount ? warp_tokens[lane] : -1;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other_score = __shfl_xor_sync(0xffffffff, score, offset);
      const int other_token = __shfl_xor_sync(0xffffffff, token, offset);
      if (take_other(other_score, other_token, score, token)) {
        score = other_score;
        token = other_token;
      }
    }
    if (lane == 0) {
      output[row_id] = token >= 0 ? token : 0;
      counters[row_id] = 0;
    }
  }
}

int temperature_blocks_per_row(int batch_size, int sm_count) {
  const int occupancy_factor = batch_size >= 8 && batch_size <= 64 ? 4 : 1;
  int blocks = (occupancy_factor * sm_count + batch_size - 1) / batch_size;
  blocks = std::max(blocks, kTemperatureMinBlocks);
  return std::min(blocks, sm_count);
}

template <typename DType, bool HasExternalGumbel, bool HasDraftMask>
cudaError_t launch_temperature_typed(int32_t* output, const void* logits, int row_stride,
                                     const float* temperature, float temperature_value,
                                     const float* gumbel_noise, const int64_t* draft_token_ids,
                                     int batch_size, void* workspace, size_t workspace_size,
                                     int sm_count, uint64_t seed, uint64_t rng_offset,
                                     cudaStream_t stream) {
  if (sm_count <= 0 || sm_count > kTemperatureThreads) {
    return cudaErrorInvalidConfiguration;
  }
  const int blocks_per_row = temperature_blocks_per_row(batch_size, sm_count);
  const size_t partial_count = static_cast<size_t>(batch_size) * sm_count;
  const size_t score_bytes = partial_count * sizeof(float);
  const size_t token_bytes = partial_count * sizeof(int32_t);
  const size_t counter_bytes = static_cast<size_t>(batch_size) * sizeof(int32_t);
  if (workspace == nullptr || workspace_size < score_bytes + token_bytes + counter_bytes) {
    return cudaErrorInvalidValue;
  }
  auto* bytes = static_cast<uint8_t*>(workspace);
  auto* partial_score = reinterpret_cast<float*>(bytes);
  auto* partial_token = reinterpret_cast<int32_t*>(bytes + score_bytes);
  auto* counters = reinterpret_cast<int32_t*>(bytes + score_bytes + token_bytes);
  cudaError_t status = cudaMemsetAsync(counters, 0, counter_bytes, stream);
  if (status != cudaSuccess) return status;
  const uint64_t offset = HasExternalGumbel ? 0 : rng_offset;
  temperature_kernel<DType, kVocabSize, HasExternalGumbel, HasDraftMask>
      <<<dim3(blocks_per_row, batch_size), kTemperatureThreads, 0, stream>>>(
          output, static_cast<const DType*>(logits), row_stride, temperature, temperature_value,
          gumbel_noise, draft_token_ids, partial_score, partial_token, counters, blocks_per_row,
          sm_count, seed, offset);
  return cudaGetLastError();
}

template <typename DType>
cudaError_t dispatch_temperature_flags(int32_t* output, const void* logits, int row_stride,
                                       const float* temperature, float temperature_value,
                                       const float* gumbel_noise, const int64_t* draft_token_ids,
                                       int batch_size, void* workspace, size_t workspace_size,
                                       int sm_count, uint64_t seed, uint64_t rng_offset,
                                       cudaStream_t stream) {
  if (gumbel_noise) {
    if (draft_token_ids) {
      return launch_temperature_typed<DType, true, true>(
          output, logits, row_stride, temperature, temperature_value, gumbel_noise, draft_token_ids,
          batch_size, workspace, workspace_size, sm_count, seed, rng_offset, stream);
    }
    return launch_temperature_typed<DType, true, false>(
        output, logits, row_stride, temperature, temperature_value, gumbel_noise, nullptr,
        batch_size, workspace, workspace_size, sm_count, seed, rng_offset, stream);
  }
  if (draft_token_ids) {
    return launch_temperature_typed<DType, false, true>(
        output, logits, row_stride, temperature, temperature_value, nullptr, draft_token_ids,
        batch_size, workspace, workspace_size, sm_count, seed, rng_offset, stream);
  }
  return launch_temperature_typed<DType, false, false>(
      output, logits, row_stride, temperature, temperature_value, nullptr, nullptr, batch_size,
      workspace, workspace_size, sm_count, seed, rng_offset, stream);
}

// --------------------------------------------------------------------------
// Full fused sampler: scan/local-top-k followed by merge/sample/writeback.
// --------------------------------------------------------------------------

constexpr int kHeavySingleBatchThreads = 1024;
constexpr int kHeavyDefaultThreads = 512;
constexpr int kHeavyItems = 4;
constexpr int kHeavyMinBlocks = 8;

template <int HeavyThreads, int MaxTopK>
constexpr int max_heavy_blocks() {
  return (HeavyThreads / MaxTopK) < 32 ? (HeavyThreads / MaxTopK) : 32;
}

template <typename DType, int VocabSize>
__device__ __forceinline__ void load_logits_safe(const DType* row, int column_base, float* values) {
  if (column_base + kHeavyItems <= VocabSize) {
    load_as_float<DType, kHeavyItems>(row + column_base, values);
  } else {
#pragma unroll
    for (int i = 0; i < kHeavyItems; ++i) {
      const int column = column_base + i;
      if (column < VocabSize) {
        if constexpr (std::is_same_v<DType, float>) {
          values[i] = row[column];
        } else {
          values[i] = __bfloat162float(row[column]);
        }
      } else {
        values[i] = kNegInf;
      }
    }
  }
}

__device__ __forceinline__ void apply_penalty_temperature(
    float* values, int column_base, bool penalty_active, float penalty, float inverse_penalty,
    const uint8_t* penalty_row, bool temperature_active, float inverse_temperature,
    int vocab_size) {
  static_assert(8 % kHeavyItems == 0,
                "each aligned heavy-item group must fit in one packed-mask byte");
  if (penalty_active) {
    // column_base is always four-token aligned, so the four
    // penalty bits consumed by this loader live in one packed-mask byte.
    // Load that byte once instead of issuing the same byte load per item.
    const uint8_t penalty_bits =
        column_base < vocab_size
            ? static_cast<uint8_t>(penalty_row[column_base >> 3] >> (column_base & 7))
            : 0;
#pragma unroll
    for (int i = 0; i < kHeavyItems; ++i) {
      const int column = column_base + i;
      if (column < vocab_size) {
        if ((penalty_bits >> i) & 1u) {
          values[i] *= values[i] > 0.0f ? inverse_penalty : penalty;
        }
      }
    }
  }
  if (temperature_active) {
#pragma unroll
    for (int i = 0; i < kHeavyItems; ++i) values[i] *= inverse_temperature;
  }
}

template <int HeavyThreads>
__device__ __forceinline__ float block_max(float value, float* scratch) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  value = warp_max(value);
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  if (warp == 0) {
    float aggregate = lane < HeavyThreads / 32 ? scratch[lane] : kNegInf;
    aggregate = warp_max(aggregate);
    if (lane == 0) scratch[0] = aggregate;
  }
  __syncthreads();
  return scratch[0];
}

template <int HeavyThreads>
__device__ __forceinline__ float block_sum(float value, float* scratch) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  value = warp_sum(value);
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  if (warp == 0) {
    float aggregate = lane < HeavyThreads / 32 ? scratch[lane] : 0.0f;
    aggregate = warp_sum(aggregate);
    if (lane == 0) scratch[0] = aggregate;
  }
  __syncthreads();
  return scratch[0];
}

template <int HeavyThreads, typename DType, int VocabSize, int MaxTopK, int SoftmaxPolicy>
__global__ __launch_bounds__(HeavyThreads) void scan_topk_kernel(
    float* partial_logits, int* partial_tokens, float* partial_max, float* partial_sum,
    const DType* logits, int row_stride, const uint8_t* penalty_mask, const int32_t* slot_id,
    const float* repetition_penalty, float repetition_penalty_value, const float* temperature,
    float temperature_value, int penalty_row_stride, int penalty_rows, int blocks_per_row,
    int scratch_stride) {
  const int row_id = static_cast<int>(blockIdx.y);
  const int block_id = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  constexpr int kKeeperThreads = MaxTopK / kHeavyItems;
  constexpr int kLoaderThreads = HeavyThreads - kKeeperThreads;
  const bool keeper = tid < kKeeperThreads;
  const int loader_id = tid - kKeeperThreads;

  const DType* row = logits + static_cast<int64_t>(row_id) * row_stride;
  const float penalty = repetition_penalty ? repetition_penalty[row_id] : repetition_penalty_value;
  const int slot = slot_id ? slot_id[row_id] : -1;
  const bool slot_valid = static_cast<unsigned>(slot) < static_cast<unsigned>(penalty_rows);
  const bool penalty_active = penalty > 0.0f && penalty_mask && slot_id && slot_valid;
  const float inverse_penalty = penalty_active ? fast_rcp(penalty) : 0.0f;
  const uint8_t* penalty_row = penalty_active ? penalty_mask + slot * penalty_row_stride : nullptr;
  const float temp = temperature ? temperature[row_id] : temperature_value;
  const bool temperature_active = temp > 0.0f;
  const float inverse_temperature = temperature_active ? fast_rcp(temp) : 0.0f;

  using Sort = cub::BlockRadixSort<float, HeavyThreads, kHeavyItems, int>;
  constexpr int kHeavyWarps = HeavyThreads / 32;
  __shared__ union {
    typename Sort::TempStorage sort;
    // Keep max and sum scratch disjoint.  Warps may leave block_max's final
    // barrier at different times, so immediately reusing the same words for
    // block_sum would race warps that have not loaded the maximum yet.
    struct {
      float maximum[kHeavyWarps];
      float sum[kHeavyWarps];
    } reduction;
  } shared;

  float keys[kHeavyItems];
  int values[kHeavyItems];
  if (keeper) {
#pragma unroll
    for (int i = 0; i < kHeavyItems; ++i) {
      keys[i] = kNegInf;
      values[i] = -1;
    }
  }
  float thread_max = kNegInf;
  float thread_sum = 0.0f;
  const int elements_per_iteration = blocks_per_row * kLoaderThreads * kHeavyItems;
  const int iterations = (VocabSize + elements_per_iteration - 1) / elements_per_iteration;

#pragma unroll 1
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (!keeper) {
      const int column_base = iteration * elements_per_iteration +
                              block_id * kLoaderThreads * kHeavyItems + loader_id * kHeavyItems;
      float loaded[kHeavyItems];
      load_logits_safe<DType, VocabSize>(row, column_base, loaded);
      apply_penalty_temperature(loaded, column_base, penalty_active, penalty, inverse_penalty,
                                penalty_row, temperature_active, inverse_temperature, VocabSize);
#pragma unroll
      for (int i = 0; i < kHeavyItems; ++i) {
        const int column = column_base + i;
        if (column < VocabSize) {
          keys[i] = loaded[i];
          values[i] = column;
          if constexpr (SoftmaxPolicy == kSoftmaxBeforeTopK) {
            const float x = loaded[i];
            // An empty loader slice has max=-inf and mass=0.  Avoid
            // exp(-inf - -inf), which would poison the block partial with NaN.
            if (x != kNegInf) {
              if (thread_max == kNegInf) {
                thread_max = x;
                thread_sum = 1.0f;
              } else if (x > thread_max) {
                thread_sum = thread_sum * precise_exp(thread_max - x) + 1.0f;
                thread_max = x;
              } else {
                thread_sum += precise_exp(x - thread_max);
              }
            }
          }
        } else {
          keys[i] = kNegInf;
          values[i] = -1;
        }
      }
    }
    __syncthreads();
    Sort(shared.sort).SortDescending(keys, values);
    __syncthreads();
  }

  if constexpr (SoftmaxPolicy == kSoftmaxBeforeTopK) {
    const float reduced_max = keeper ? kNegInf : thread_max;
    const float maximum = block_max<HeavyThreads>(reduced_max, shared.reduction.maximum);
    const float reduced_sum = keeper ? 0.0f : thread_sum;
    const float corrected =
        maximum == kNegInf ? 0.0f : reduced_sum * precise_exp(reduced_max - maximum);
    const float sum = block_sum<HeavyThreads>(corrected, shared.reduction.sum);
    if (tid == 0) {
      const int index = row_id * scratch_stride + block_id;
      partial_max[index] = maximum;
      partial_sum[index] = sum;
    }
  }

  if (keeper) {
    const int destination =
        row_id * scratch_stride * MaxTopK + block_id * MaxTopK + tid * kHeavyItems;
#pragma unroll
    for (int i = 0; i < kHeavyItems; ++i) {
      partial_logits[destination + i] = keys[i];
      partial_tokens[destination + i] = values[i];
    }
  }
}

template <int HeavyThreads, int MaxTopK, int SoftmaxPolicy, bool HasTopP,
          int MergeBlocks = max_heavy_blocks<HeavyThreads, MaxTopK>()>
__global__ void merge_sample_kernel(int32_t* output, const float* partial_logits,
                                    const int* partial_tokens, const float* partial_max,
                                    const float* partial_sum, const void* topk,
                                    int topk_element_bytes, int topk_value, const float* topp,
                                    float topp_value, const float* gumbel_noise, int vocab_size,
                                    uint8_t* penalty_mask, const int32_t* slot_id,
                                    const float* repetition_penalty, float repetition_penalty_value,
                                    int penalty_row_stride, int penalty_rows, uint64_t seed,
                                    uint64_t rng_offset, int blocks_per_row) {
  constexpr int kMaxBlocks = max_heavy_blocks<HeavyThreads, MaxTopK>();
  constexpr int kThreads = MergeBlocks * MaxTopK;
  static_assert(MergeBlocks <= kMaxBlocks);
  const int row_id = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;

  int effective_k = MaxTopK;
  if (topk) {
    int requested = topk_element_bytes == 4
                        ? reinterpret_cast<const int32_t*>(topk)[row_id]
                        : static_cast<int>(reinterpret_cast<const int64_t*>(topk)[row_id]);
    if (requested > 0) effective_k = min(requested, MaxTopK);
  } else if (topk_value > 0) {
    effective_k = min(topk_value, MaxTopK);
  }

  float global_max = 0.0f;
  float inverse_sum = 0.0f;
  if constexpr (SoftmaxPolicy == kSoftmaxBeforeTopK) {
    __shared__ float shared_global_max;
    __shared__ float shared_inverse_sum;
    if (tid < 32) {
      const float local_max =
          lane < blocks_per_row ? partial_max[row_id * kMaxBlocks + lane] : kNegInf;
      const float maximum = warp_max(local_max);
      const float local_sum =
          lane < blocks_per_row ? partial_sum[row_id * kMaxBlocks + lane] : 0.0f;
      const float corrected = (local_max == kNegInf || maximum == kNegInf)
                                  ? 0.0f
                                  : local_sum * precise_exp(local_max - maximum);
      const float sum = warp_sum(corrected);
      if (lane == 0) {
        shared_global_max = maximum;
        shared_inverse_sum = sum > 0.0f ? fast_rcp(sum) : 0.0f;
      }
    }
    __syncthreads();
    global_max = shared_global_max;
    inverse_sum = shared_inverse_sum;
  }

  using Sort = cub::BlockRadixSort<float, kThreads, 1, int>;
  __shared__ union {
    typename Sort::TempStorage sort;
    struct {
      float logits[kThreads];
      int tokens[kThreads];
    } result;
  } shared;

  const int candidate_block = tid / MaxTopK;
  float key[1];
  int token[1];
  if (candidate_block < blocks_per_row) {
    const int index = row_id * kMaxBlocks * MaxTopK + tid;
    key[0] = partial_logits[index];
    token[0] = partial_tokens[index];
  } else {
    key[0] = kNegInf;
    token[0] = -1;
  }
  if constexpr (SoftmaxPolicy == kSoftmaxBeforeTopK) {
    key[0] = token[0] >= 0 ? precise_exp(key[0] - global_max) * inverse_sum : 0.0f;
  }

  __syncthreads();
  Sort(shared.sort).SortDescending(key, token);
  __syncthreads();
  shared.result.logits[tid] = key[0];
  shared.result.tokens[tid] = token[0];
  __syncthreads();
  if (tid >= 32) return;

  constexpr int kItemsPerLane = MaxTopK / 32;
  float lane_logits[kItemsPerLane];
  float lane_probabilities[kItemsPerLane];
  int lane_tokens[kItemsPerLane];
#pragma unroll
  for (int i = 0; i < kItemsPerLane; ++i) {
    const int index = lane * kItemsPerLane + i;
    lane_logits[i] = shared.result.logits[index];
    lane_tokens[i] = shared.result.tokens[index];
    lane_probabilities[i] = 0.0f;
  }

  if constexpr (SoftmaxPolicy == kSoftmaxAfterTopK) {
    const float maximum = __shfl_sync(0xffffffff, lane_logits[0], 0);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kItemsPerLane; ++i) {
      const int index = lane * kItemsPerLane + i;
      if (index < effective_k && lane_tokens[i] >= 0) {
        lane_probabilities[i] = precise_exp(lane_logits[i] - maximum);
        sum += lane_probabilities[i];
      }
    }
    sum = warp_sum(sum);
    const float inverse = sum > 0.0f ? fast_rcp(sum) : 0.0f;
#pragma unroll
    for (int i = 0; i < kItemsPerLane; ++i) {
      lane_probabilities[i] *= inverse;
    }
  } else if constexpr (SoftmaxPolicy == kSoftmaxBeforeTopK) {
#pragma unroll
    for (int i = 0; i < kItemsPerLane; ++i) {
      const int index = lane * kItemsPerLane + i;
      lane_probabilities[i] = index < effective_k && lane_tokens[i] >= 0 ? lane_logits[i] : 0.0f;
    }
  }

  float lane_prefix = 0.0f;
  if constexpr (HasTopP) {
    const float threshold = topp ? topp[row_id] : topp_value;
    if (threshold > 0.0f) {
      float lane_sum = 0.0f;
#pragma unroll
      for (int i = 0; i < kItemsPerLane; ++i) {
        const int index = lane * kItemsPerLane + i;
        if (index < effective_k) lane_sum += lane_probabilities[i];
      }
      float inclusive = lane_sum;
#pragma unroll
      for (int offset = 1; offset < 32; offset <<= 1) {
        const float other = __shfl_up_sync(0xffffffff, inclusive, offset);
        if (lane >= offset) inclusive += other;
      }
      lane_prefix = inclusive - lane_sum;
    }
  }

  curandStatePhilox4_32_10_t rng;
  float uniforms[kItemsPerLane]{};
  if (!gumbel_noise) {
    const uint64_t sequence = static_cast<uint64_t>(row_id) * 32 + lane;
    curand_init(seed, sequence, rng_offset, &rng);
    const float4 random = curand_uniform4(&rng);
    const float values[4] = {random.x, random.y, random.z, random.w};
#pragma unroll
    for (int i = 0; i < kItemsPerLane; ++i) uniforms[i] = values[i];
  }

  const float top_p_threshold = HasTopP ? (topp ? topp[row_id] : topp_value) : 0.0f;
  float running_probability = lane_prefix;
  float best_score = kNegInf;
  int best_token = -1;
#pragma unroll
  for (int i = 0; i < kItemsPerLane; ++i) {
    const int index = lane * kItemsPerLane + i;
    const int candidate = lane_tokens[i];
    bool keep = index < effective_k && candidate >= 0;
    if constexpr (HasTopP) {
      if (top_p_threshold > 0.0f) {
        keep = keep && (index == 0 || running_probability < top_p_threshold);
      }
    }
    if (keep) {
      float value;
      if constexpr (SoftmaxPolicy == kSoftmaxNone) {
        value = lane_logits[i];
      } else {
        const float probability = lane_probabilities[i];
        value = probability > 0.0f ? fast_log(probability) : kNegInf;
      }
      const float noise = gumbel_noise
                              ? gumbel_noise[static_cast<int64_t>(row_id) * vocab_size + candidate]
                              : gumbel_from_uniform(uniforms[i]);
      const float score = value + noise;
      if (take_other(score, candidate, best_score, best_token)) {
        best_score = score;
        best_token = candidate;
      }
    }
    if constexpr (HasTopP) running_probability += lane_probabilities[i];
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float score = __shfl_xor_sync(0xffffffff, best_score, offset);
    const int candidate = __shfl_xor_sync(0xffffffff, best_token, offset);
    if (take_other(score, candidate, best_score, best_token)) {
      best_score = score;
      best_token = candidate;
    }
  }
  if (lane == 0) {
    const int sampled = best_token >= 0 ? best_token : 0;
    output[row_id] = sampled;
    if (penalty_mask && slot_id) {
      const float penalty =
          repetition_penalty ? repetition_penalty[row_id] : repetition_penalty_value;
      const int slot = slot_id[row_id];
      const bool slot_valid = static_cast<unsigned>(slot) < static_cast<unsigned>(penalty_rows);
      if (penalty > 0.0f && slot_valid) {
        uint8_t* row = penalty_mask + slot * penalty_row_stride;
        uint8_t* byte_pointer = row + (sampled >> 3);
        const uintptr_t byte_address = reinterpret_cast<uintptr_t>(byte_pointer);
        const uintptr_t word_address = byte_address & ~uintptr_t{3};
        const int byte_offset = static_cast<int>(byte_address - word_address);
        const unsigned bit = (1u << (sampled & 7)) << static_cast<unsigned>(byte_offset * 8);
        atomicOr(reinterpret_cast<unsigned*>(word_address), bit);
      }
    }
  }
}

int heavy_blocks_per_row(int batch_size, int sm_count, int maximum) {
  int blocks = sm_count / batch_size;
  blocks = std::max(blocks, kHeavyMinBlocks);
  return std::min(blocks, maximum);
}

template <int HeavyThreads, typename DType, int MaxTopK, int SoftmaxPolicy, bool HasTopP>
cudaError_t launch_heavy_threads(int32_t* output, const void* logits, uint8_t* penalty_mask,
                                 const int32_t* slot_id, const float* repetition_penalty,
                                 float repetition_penalty_value, const float* temperature,
                                 float temperature_value, const void* topk, int topk_element_bytes,
                                 int topk_value, const float* topp, float topp_value,
                                 const float* gumbel_noise, int batch_size, int vocab_size,
                                 int penalty_rows, int penalty_row_stride, int row_stride,
                                 void* workspace, size_t workspace_size, int sm_count,
                                 uint64_t seed, uint64_t rng_offset, cudaStream_t stream) {
  constexpr int kMaxBlocks = max_heavy_blocks<HeavyThreads, MaxTopK>();
  constexpr bool kNeedsSoftmaxPartials = SoftmaxPolicy == kSoftmaxBeforeTopK;
  if (sm_count <= 0) return cudaErrorInvalidConfiguration;
  const int blocks_per_row = heavy_blocks_per_row(batch_size, sm_count, kMaxBlocks);
  const size_t row_blocks = static_cast<size_t>(batch_size) * kMaxBlocks;
  const size_t candidate_bytes = row_blocks * MaxTopK * sizeof(float);
  const size_t partial_bytes = kNeedsSoftmaxPartials ? row_blocks * sizeof(float) : 0;
  const size_t required_bytes = 2 * candidate_bytes + 2 * partial_bytes;
  if (workspace == nullptr || workspace_size < required_bytes) {
    return cudaErrorInvalidValue;
  }
  auto* bytes = static_cast<uint8_t*>(workspace);
  auto* logits_scratch = reinterpret_cast<float*>(bytes);
  auto* token_scratch = reinterpret_cast<int*>(bytes + candidate_bytes);
  auto* max_scratch =
      kNeedsSoftmaxPartials ? reinterpret_cast<float*>(bytes + 2 * candidate_bytes) : nullptr;
  auto* sum_scratch = kNeedsSoftmaxPartials
                          ? reinterpret_cast<float*>(bytes + 2 * candidate_bytes + partial_bytes)
                          : nullptr;

  scan_topk_kernel<HeavyThreads, DType, kVocabSize, MaxTopK, SoftmaxPolicy>
      <<<dim3(blocks_per_row, batch_size), HeavyThreads, 0, stream>>>(
          logits_scratch, token_scratch, max_scratch, sum_scratch,
          static_cast<const DType*>(logits), row_stride, penalty_mask, slot_id, repetition_penalty,
          repetition_penalty_value, temperature, temperature_value, penalty_row_stride,
          penalty_rows, blocks_per_row, kMaxBlocks);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;

  const uint64_t offset = gumbel_noise ? 0 : rng_offset;
  if constexpr (HeavyThreads == kHeavyDefaultThreads && MaxTopK == 32) {
    if (blocks_per_row <= kHeavyMinBlocks) {
      // Do not sort the eight unused partial-block groups.  Keep the scratch
      // stride at kMaxBlocks so this launch optimization does not change the
      // workspace contract shared with the scan kernel.
      constexpr int kCompactMergeBlocks = kHeavyMinBlocks;
      constexpr int kCompactMergeThreads = kCompactMergeBlocks * MaxTopK;
      merge_sample_kernel<HeavyThreads, MaxTopK, SoftmaxPolicy, HasTopP, kCompactMergeBlocks>
          <<<batch_size, kCompactMergeThreads, 0, stream>>>(
              output, logits_scratch, token_scratch, max_scratch, sum_scratch, topk,
              topk_element_bytes, topk_value, topp, topp_value, gumbel_noise, vocab_size,
              penalty_mask, slot_id, repetition_penalty, repetition_penalty_value,
              penalty_row_stride, penalty_rows, seed, offset, blocks_per_row);
      return cudaGetLastError();
    }
  }
  constexpr int kMergeThreads = kMaxBlocks * MaxTopK;
  merge_sample_kernel<HeavyThreads, MaxTopK, SoftmaxPolicy, HasTopP>
      <<<batch_size, kMergeThreads, 0, stream>>>(
          output, logits_scratch, token_scratch, max_scratch, sum_scratch, topk, topk_element_bytes,
          topk_value, topp, topp_value, gumbel_noise, vocab_size, penalty_mask, slot_id,
          repetition_penalty, repetition_penalty_value, penalty_row_stride, penalty_rows, seed,
          offset, blocks_per_row);
  return cudaGetLastError();
}

template <typename DType, int MaxTopK, int SoftmaxPolicy, bool HasTopP>
cudaError_t launch_heavy_typed(int32_t* output, const void* logits, uint8_t* penalty_mask,
                               const int32_t* slot_id, const float* repetition_penalty,
                               float repetition_penalty_value, const float* temperature,
                               float temperature_value, const void* topk, int topk_element_bytes,
                               int topk_value, const float* topp, float topp_value,
                               const float* gumbel_noise, int batch_size, int vocab_size,
                               int penalty_rows, int penalty_row_stride, int row_stride,
                               void* workspace, size_t workspace_size, int sm_count, uint64_t seed,
                               uint64_t rng_offset, cudaStream_t stream) {
  if (batch_size < 8) {
    return launch_heavy_threads<kHeavySingleBatchThreads, DType, MaxTopK, SoftmaxPolicy, HasTopP>(
        output, logits, penalty_mask, slot_id, repetition_penalty, repetition_penalty_value,
        temperature, temperature_value, topk, topk_element_bytes, topk_value, topp, topp_value,
        gumbel_noise, batch_size, vocab_size, penalty_rows, penalty_row_stride, row_stride,
        workspace, workspace_size, sm_count, seed, rng_offset, stream);
  }
  return launch_heavy_threads<kHeavyDefaultThreads, DType, MaxTopK, SoftmaxPolicy, HasTopP>(
      output, logits, penalty_mask, slot_id, repetition_penalty, repetition_penalty_value,
      temperature, temperature_value, topk, topk_element_bytes, topk_value, topp, topp_value,
      gumbel_noise, batch_size, vocab_size, penalty_rows, penalty_row_stride, row_stride, workspace,
      workspace_size, sm_count, seed, rng_offset, stream);
}

template <typename DType, int MaxTopK, int SoftmaxPolicy>
cudaError_t dispatch_heavy_topp(int32_t* output, const void* logits, uint8_t* penalty_mask,
                                const int32_t* slot_id, const float* repetition_penalty,
                                float repetition_penalty_value, const float* temperature,
                                float temperature_value, const void* topk, int topk_element_bytes,
                                int topk_value, const float* topp, float topp_value,
                                const float* gumbel_noise, int batch_size, int vocab_size,
                                int penalty_rows, int penalty_row_stride, int row_stride,
                                void* workspace, size_t workspace_size, int sm_count, uint64_t seed,
                                uint64_t rng_offset, cudaStream_t stream) {
  if (topp || topp_value > 0.0f) {
    return launch_heavy_typed<DType, MaxTopK, SoftmaxPolicy, true>(
        output, logits, penalty_mask, slot_id, repetition_penalty, repetition_penalty_value,
        temperature, temperature_value, topk, topk_element_bytes, topk_value, topp, topp_value,
        gumbel_noise, batch_size, vocab_size, penalty_rows, penalty_row_stride, row_stride,
        workspace, workspace_size, sm_count, seed, rng_offset, stream);
  }
  return launch_heavy_typed<DType, MaxTopK, SoftmaxPolicy, false>(
      output, logits, penalty_mask, slot_id, repetition_penalty, repetition_penalty_value,
      temperature, temperature_value, topk, topk_element_bytes, topk_value, topp, topp_value,
      gumbel_noise, batch_size, vocab_size, penalty_rows, penalty_row_stride, row_stride, workspace,
      workspace_size, sm_count, seed, rng_offset, stream);
}

template <typename DType, int MaxTopK>
cudaError_t dispatch_heavy_policy(
    int32_t* output, const void* logits, uint8_t* penalty_mask, const int32_t* slot_id,
    const float* repetition_penalty, float repetition_penalty_value, const float* temperature,
    float temperature_value, int softmax_policy, const void* topk, int topk_element_bytes,
    int topk_value, const float* topp, float topp_value, const float* gumbel_noise, int batch_size,
    int vocab_size, int penalty_rows, int penalty_row_stride, int row_stride, void* workspace,
    size_t workspace_size, int sm_count, uint64_t seed, uint64_t rng_offset, cudaStream_t stream) {
#define DISPATCH_POLICY(POLICY)                                                               \
  return dispatch_heavy_topp<DType, MaxTopK, POLICY>(                                         \
      output, logits, penalty_mask, slot_id, repetition_penalty, repetition_penalty_value,    \
      temperature, temperature_value, topk, topk_element_bytes, topk_value, topp, topp_value, \
      gumbel_noise, batch_size, vocab_size, penalty_rows, penalty_row_stride, row_stride,     \
      workspace, workspace_size, sm_count, seed, rng_offset, stream)
  switch (softmax_policy) {
    case kSoftmaxNone:
      DISPATCH_POLICY(kSoftmaxNone);
    case kSoftmaxBeforeTopK:
      DISPATCH_POLICY(kSoftmaxBeforeTopK);
    case kSoftmaxAfterTopK:
      DISPATCH_POLICY(kSoftmaxAfterTopK);
    default:
      return cudaErrorInvalidValue;
  }
#undef DISPATCH_POLICY
}

template <typename DType>
cudaError_t dispatch_heavy_topk(int32_t* output, const void* logits, uint8_t* penalty_mask,
                                const int32_t* slot_id, const float* repetition_penalty,
                                float repetition_penalty_value, const float* temperature,
                                float temperature_value, int softmax_policy, const void* topk,
                                int topk_element_bytes, int topk_value, const float* topp,
                                float topp_value, const float* gumbel_noise, int batch_size,
                                int vocab_size, int penalty_rows, int penalty_row_stride,
                                int row_stride, int max_topk, void* workspace,
                                size_t workspace_size, int sm_count, uint64_t seed,
                                uint64_t rng_offset, cudaStream_t stream) {
#define CALL_TOPK(K)                                                                              \
  return dispatch_heavy_policy<DType, K>(                                                         \
      output, logits, penalty_mask, slot_id, repetition_penalty, repetition_penalty_value,        \
      temperature, temperature_value, softmax_policy, topk, topk_element_bytes, topk_value, topp, \
      topp_value, gumbel_noise, batch_size, vocab_size, penalty_rows, penalty_row_stride,         \
      row_stride, workspace, workspace_size, sm_count, seed, rng_offset, stream)
  if (max_topk == 32) CALL_TOPK(32);
  if (max_topk == 64) CALL_TOPK(64);
#undef CALL_TOPK
  return cudaErrorInvalidValue;
}

}  // namespace

cudaError_t launch_temperature(int32_t* output, const void* logits, int logits_dtype,
                               int logits_row_stride, const float* temperature,
                               float temperature_value, const float* gumbel_noise,
                               const int64_t* draft_token_ids, int batch_size, int vocab_size,
                               void* workspace, size_t workspace_size, int sm_count, uint64_t seed,
                               uint64_t rng_offset, cudaStream_t stream) {
  if (vocab_size != kVocabSize || batch_size <= 0) return cudaErrorInvalidValue;
  if (logits_dtype == 0) {
    return dispatch_temperature_flags<float>(
        output, logits, logits_row_stride, temperature, temperature_value, gumbel_noise,
        draft_token_ids, batch_size, workspace, workspace_size, sm_count, seed, rng_offset, stream);
  }
  if (logits_dtype == 1) {
    return dispatch_temperature_flags<__nv_bfloat16>(
        output, logits, logits_row_stride, temperature, temperature_value, gumbel_noise,
        draft_token_ids, batch_size, workspace, workspace_size, sm_count, seed, rng_offset, stream);
  }
  return cudaErrorInvalidValue;
}

cudaError_t launch_heavy(int32_t* output, const void* logits, int logits_dtype,
                         uint8_t* penalty_mask, const int32_t* slot_id,
                         const float* repetition_penalty, float repetition_penalty_value,
                         const float* temperature, float temperature_value, int softmax_policy,
                         const void* topk, int topk_element_bytes, int topk_value,
                         const float* topp, float topp_value, const float* gumbel_noise,
                         int batch_size, int vocab_size, int penalty_rows, int penalty_row_stride,
                         int logits_row_stride, int max_topk, void* workspace,
                         size_t workspace_size, int sm_count, uint64_t seed, uint64_t rng_offset,
                         cudaStream_t stream) {
  if (vocab_size != kVocabSize || batch_size <= 0) return cudaErrorInvalidValue;
  if (logits_dtype == 0) {
    return dispatch_heavy_topk<float>(
        output, logits, penalty_mask, slot_id, repetition_penalty, repetition_penalty_value,
        temperature, temperature_value, softmax_policy, topk, topk_element_bytes, topk_value, topp,
        topp_value, gumbel_noise, batch_size, vocab_size, penalty_rows, penalty_row_stride,
        logits_row_stride, max_topk, workspace, workspace_size, sm_count, seed, rng_offset, stream);
  }
  if (logits_dtype == 1) {
    return dispatch_heavy_topk<__nv_bfloat16>(
        output, logits, penalty_mask, slot_id, repetition_penalty, repetition_penalty_value,
        temperature, temperature_value, softmax_policy, topk, topk_element_bytes, topk_value, topp,
        topp_value, gumbel_noise, batch_size, vocab_size, penalty_rows, penalty_row_stride,
        logits_row_stride, max_topk, workspace, workspace_size, sm_count, seed, rng_offset, stream);
  }
  return cudaErrorInvalidValue;
}

}  // namespace flashinfer::sampling::hy3
