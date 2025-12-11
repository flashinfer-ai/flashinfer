/*
 * Copyright (c) 2024 by FlashInfer team.
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
#ifndef FLASHINFER_SAMPLING_CUH_
#define FLASHINFER_SAMPLING_CUH_

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

#include <cub/cub.cuh>
#include <cuda/functional>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <limits>
#include <numeric>
#include <tuple>

#include "allocator.h"
#include "math.cuh"
#include "utils.cuh"
#include "vec_dtypes.cuh"

// Define reduction operators based on CUDA version
// CUDA 13 (12.9+) deprecated cub::Max/Min in favor of cuda::maximum/minimum
#if CUDA_VERSION >= 12090
using MaxReduceOp = cuda::maximum<>;
using MinReduceOp = cuda::minimum<>;
#else
using MaxReduceOp = cub::Max;
using MinReduceOp = cub::Min;
#endif

namespace flashinfer {

namespace sampling {

using namespace cub;

#define DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, ...) \
  if (deterministic) {                                            \
    constexpr bool DETERMINISTIC = true;                          \
    __VA_ARGS__                                                   \
  } else {                                                        \
    constexpr bool DETERMINISTIC = false;                         \
    __VA_ARGS__                                                   \
  }

#define DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, ...) \
  if (compute_capacity.first >= 8) {                                           \
    constexpr uint32_t BLOCK_THREADS = 1024;                                   \
    __VA_ARGS__                                                                \
  } else {                                                                     \
    constexpr uint32_t BLOCK_THREADS = 1024;                                   \
    __VA_ARGS__                                                                \
  }

#define DISPATCH_SOFTMAX_CACHE_INPUT(cache_input, CACHE_INPUT, ...) \
  if (cache_input) {                                                \
    constexpr bool CACHE_INPUT = true;                              \
    __VA_ARGS__                                                     \
  } else {                                                          \
    constexpr bool CACHE_INPUT = false;                             \
    __VA_ARGS__                                                     \
  }

constexpr BlockScanAlgorithm SCAN_ALGO = BLOCK_SCAN_WARP_SCANS;
constexpr BlockReduceAlgorithm REDUCE_ALGO = BLOCK_REDUCE_WARP_REDUCTIONS;

#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120100)
#define FLASHINFER_CUB_SUBTRACTLEFT_DEFINED
#endif

template <typename T>
struct ValueCount {
  T value;
  int count;

  __device__ ValueCount operator+(const ValueCount& other) const {
    return {value + other.value, count + other.count};
  }
  __device__ ValueCount& operator+=(const ValueCount& other) {
    value += other.value;
    count += other.count;
    return *this;
  }
};

struct BoolDiffOp {
  __device__ __forceinline__ bool operator()(const bool& lhs, const bool& rhs) const {
    return lhs != rhs;
  }
};

struct Float2SoftmaxReduceOp {
  __device__ __forceinline__ float2 operator()(const float2& a, const float2& b) const {
    if (isinf(a.x)) return b;
    if (isinf(b.x)) return a;

    float new_max = max(a.x, b.x);
    float new_denom = a.y * __expf(a.x - new_max) + b.y * __expf(b.x - new_max);
    return make_float2(new_max, new_denom);
  }
};

// ============================================================================
// RadixTopK Type Traits - supports float, half, and bfloat16
// OrderedType: uint32_t for float, uint16_t for half/bf16
// NUM_ROUNDS is computed as: sizeof(OrderedType) * 8 / RADIX_BITS
// ============================================================================
template <typename DType>
struct RadixTopKTraits;

// Specialization for float (32-bit)
template <>
struct RadixTopKTraits<float> {
  using OrderedType = uint32_t;

  // Compute number of rounds based on radix bits (not hardcoded)
  template <uint32_t RADIX_BITS>
  static __host__ __device__ constexpr uint32_t num_rounds() {
    return sizeof(OrderedType) * 8 / RADIX_BITS;
  }

  __device__ __forceinline__ static OrderedType ToOrdered(float val) {
    uint32_t bits = __float_as_uint(val);
    // For descending order: flip all bits if negative, else flip sign bit
    return (bits & 0x80000000) ? ~bits : (bits ^ 0x80000000);
  }

  __device__ __forceinline__ static float FromOrdered(OrderedType ordered) {
    uint32_t bits = (ordered & 0x80000000) ? (ordered ^ 0x80000000) : ~ordered;
    return __uint_as_float(bits);
  }

  __device__ __forceinline__ static float NegInf() {
    return -cuda::std::numeric_limits<float>::infinity();
  }
};

// Specialization for half (16-bit)
template <>
struct RadixTopKTraits<half> {
  using OrderedType = uint16_t;

  template <uint32_t RADIX_BITS>
  static __host__ __device__ constexpr uint32_t num_rounds() {
    return sizeof(OrderedType) * 8 / RADIX_BITS;
  }

  __device__ __forceinline__ static OrderedType ToOrdered(half val) {
    uint16_t bits = __half_as_ushort(val);
    return (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits ^ 0x8000);
  }

  __device__ __forceinline__ static half FromOrdered(OrderedType ordered) {
    uint16_t bits = (ordered & 0x8000) ? static_cast<uint16_t>(ordered ^ 0x8000)
                                       : static_cast<uint16_t>(~ordered);
    return __ushort_as_half(bits);
  }

  __device__ __forceinline__ static half NegInf() {
    return __ushort_as_half(static_cast<uint16_t>(0xFC00));  // -inf in fp16
  }
};

// Specialization for nv_bfloat16 (16-bit)
template <>
struct RadixTopKTraits<nv_bfloat16> {
  using OrderedType = uint16_t;

  template <uint32_t RADIX_BITS>
  static __host__ __device__ constexpr uint32_t num_rounds() {
    return sizeof(OrderedType) * 8 / RADIX_BITS;
  }

  __device__ __forceinline__ static OrderedType ToOrdered(nv_bfloat16 val) {
    uint16_t bits = __bfloat16_as_ushort(val);
    return (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits ^ 0x8000);
  }

  __device__ __forceinline__ static nv_bfloat16 FromOrdered(OrderedType ordered) {
    uint16_t bits = (ordered & 0x8000) ? static_cast<uint16_t>(ordered ^ 0x8000)
                                       : static_cast<uint16_t>(~ordered);
    return __ushort_as_bfloat16(bits);
  }

  __device__ __forceinline__ static nv_bfloat16 NegInf() {
    return __ushort_as_bfloat16(static_cast<uint16_t>(0xFF80));  // -inf in bf16
  }
};

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM>
struct SamplingTempStorage {
  union {
    float deterministic_scan[BLOCK_THREADS / 32];
    typename BlockScan<float, BLOCK_THREADS, SCAN_ALGORITHM>::TempStorage scan;
    typename BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce;
    typename BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce_int;
    typename BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage
        reduce_value_count;
    typename BlockAdjacentDifference<bool, BLOCK_THREADS>::TempStorage adj_diff;
  } block_prim;
  struct {
    int32_t sampled_id;
    int32_t last_valid_id;
    float max_val;
    union {
      float value;
      ValueCount<float> pair;
    } block_aggregate;
  };
};

template <uint32_t BLOCK_THREADS>
struct OnlineSoftmaxTempStorage {
  union {
    typename cub::BlockReduce<float, BLOCK_THREADS>::TempStorage reduce;
    typename cub::BlockReduce<float2, BLOCK_THREADS>::TempStorage reduce_pair;
  } block_prim;

  struct {
    float max_val;
    float denominator;
  } shared_state;
};

struct PartialSoftmaxResult {
  float max_val;
  float denominator;
};

/*!
 * \brief Deterministic inclusive scan implementation, use Belloch scan algorithm.
 * \note This implementation is slower than the cub::BlockScan, but it is deterministic.
 */
template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM>
__device__ __forceinline__ void DeterministicInclusiveSum(
    const float* in_data, float* out_data,
    SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>* temp_storage) {
  float* smem_prefix_sum = temp_storage->block_prim.deterministic_scan;
  float thread_data[VEC_SIZE];
  float thread_sum = 0;
#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) {
    thread_sum += in_data[i];
    thread_data[i] = thread_sum;
  }

  float thread_exclusive_prefix_sum = thread_sum;

#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset *= 2) {
    float tmp = __shfl_up_sync(0xffffffff, thread_exclusive_prefix_sum, offset);
    if ((threadIdx.x + 1) % (offset * 2) == 0) {
      thread_exclusive_prefix_sum += tmp;
    }
  }

  float warp_sum = __shfl_sync(0xffffffff, thread_exclusive_prefix_sum, threadIdx.x | 0xffffffff);
  if (threadIdx.x % 32 == 31) {
    thread_exclusive_prefix_sum = 0;
  }

#pragma unroll
  for (uint32_t offset = 16; offset >= 1; offset /= 2) {
    float tmp = __shfl_xor_sync(0xffffffff, thread_exclusive_prefix_sum, offset);
    if ((threadIdx.x + 1) % (offset * 2) == 0) {
      thread_exclusive_prefix_sum = tmp + thread_exclusive_prefix_sum;
    }
    if ((threadIdx.x + 1) % (offset * 2) == offset) {
      thread_exclusive_prefix_sum = tmp;
    }
  }

  smem_prefix_sum[threadIdx.x / 32] = warp_sum;
  __syncthreads();

  if (threadIdx.x < 32) {
    float warp_exclusive_prefix_sum =
        (threadIdx.x < BLOCK_THREADS / 32) ? smem_prefix_sum[threadIdx.x] : 0;

#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset *= 2) {
      float tmp = __shfl_up_sync(0xffffffff, warp_exclusive_prefix_sum, offset);
      if ((threadIdx.x + 1) % (offset * 2) == 0) {
        warp_exclusive_prefix_sum += tmp;
      }
    }

    if (threadIdx.x % 32 == 31) {
      warp_exclusive_prefix_sum = 0;
    }

#pragma unroll
    for (uint32_t offset = 16; offset >= 1; offset /= 2) {
      float tmp = __shfl_xor_sync(0xffffffff, warp_exclusive_prefix_sum, offset);
      if ((threadIdx.x + 1) % (offset * 2) == 0) {
        warp_exclusive_prefix_sum = tmp + warp_exclusive_prefix_sum;
      }
      if ((threadIdx.x + 1) % (offset * 2) == offset) {
        warp_exclusive_prefix_sum = tmp;
      }
    }
    if (threadIdx.x < BLOCK_THREADS / 32) {
      smem_prefix_sum[threadIdx.x] = warp_exclusive_prefix_sum;
    }
  }
  __syncthreads();

#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) {
    out_data[i] = smem_prefix_sum[threadIdx.x / 32] + thread_exclusive_prefix_sum + thread_data[i];
  }
}

template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS, BlockReduceAlgorithm REDUCE_ALGORITHM,
          typename TempStorage>
__device__ __forceinline__ std::tuple<float, float> GetMinMaxValue(float* in_data, uint32_t row_idx,
                                                                   uint32_t d,
                                                                   TempStorage& temp_storage) {
  const uint32_t tx = threadIdx.x;
  vec_t<float, VEC_SIZE> in_data_vec;
  // Thread-local min/max accumulation (deferred reduction)
  float thread_max = -cuda::std::numeric_limits<float>::infinity();
  float thread_min = cuda::std::numeric_limits<float>::infinity();

  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    in_data_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      in_data_vec.cast_load(in_data + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      thread_max = max(thread_max, static_cast<float>(in_data_vec[j]));
      thread_min = min(thread_min, static_cast<float>(in_data_vec[j]));
    }
  }

  // Single block reduction after loop completes
  float max_val =
      BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
          .Reduce(thread_max, MaxReduceOp{});
  __syncthreads();
  float min_val =
      BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
          .Reduce(thread_min, MinReduceOp{});

  if (tx == 0) {
    temp_storage.max_val = max_val;
    temp_storage.min_val = min_val;
  }
  __syncthreads();
  max_val = temp_storage.max_val;
  min_val = temp_storage.min_val;

  return std::make_tuple(min_val, max_val);
}

template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS, BlockReduceAlgorithm REDUCE_ALGORITHM,
          typename TempStorage>
__device__ __forceinline__ float GetMaxValue(float* in_data, uint32_t row_idx, uint32_t d,
                                             TempStorage& temp_storage) {
  const uint32_t tx = threadIdx.x;
  vec_t<float, VEC_SIZE> in_data_vec;

  // Thread-local max accumulation (deferred reduction)
  float thread_max = 0.0f;
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    in_data_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      in_data_vec.cast_load(in_data + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      thread_max = max(thread_max, static_cast<float>(in_data_vec[j]));
    }
  }

  // Single block reduction after loop completes
  float max_val =
      BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
          .Reduce(thread_max, MaxReduceOp{});
  if (tx == 0) {
    temp_storage.max_val = max_val;
  }
  __syncthreads();
  return temp_storage.max_val;
}

template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, typename DType, bool CACHE_INPUT>
__global__ void OnlineSoftmaxFusedKernel(DType* logits, DType* output, DType* temperature_arr,
                                         DType temperature_val, uint32_t d) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  float temperature = temperature_arr == nullptr ? temperature_val : temperature_arr[bx];
  const float inv_temp = (temperature == 0.f) ? 0.f : 1.f / temperature;

  using TempStorage = OnlineSoftmaxTempStorage<BLOCK_THREADS>;
  extern __shared__ __align__(alignof(TempStorage)) uint8_t smem[];
  auto& temp_storage = reinterpret_cast<TempStorage&>(smem);

  DType* smem_vec_base = nullptr;
  if constexpr (CACHE_INPUT) {
    constexpr size_t vec_alignment = alignof(vec_t<DType, VEC_SIZE>);
    size_t aligned_offset = round_up(sizeof(TempStorage), vec_alignment);
    smem_vec_base = reinterpret_cast<DType*>(smem + aligned_offset);
  }

  vec_t<DType, VEC_SIZE> logits_vec;

  float running_max = -cuda::std::numeric_limits<float>::infinity();
  float running_denominator = 0.0f;
  float threadlocal_running_denominator = 0.0f;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  // Pass 1: Compute running max and denominator
#pragma unroll 2
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    logits_vec.fill(-cuda::std::numeric_limits<DType>::infinity());
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      logits_vec.cast_load(logits + bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);

#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        logits_vec[j] *= inv_temp;
      }

      if constexpr (CACHE_INPUT) {
        logits_vec.store(smem_vec_base + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }
    }

    float thread_max = -cuda::std::numeric_limits<float>::infinity();
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      thread_max = max(thread_max, logits_vec[j]);
    }
    float block_max = cub::BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                          .Reduce(thread_max, MaxReduceOp{});

    if (tx == 0) {
      temp_storage.shared_state.max_val = block_max;
    }
    __syncthreads();
    block_max = temp_storage.shared_state.max_val;
    // if block_max is -inf, then this block contains all -inf values, so we can skip updating
    if (!isinf(block_max)) {
      float threadlocal_sum = 0.0f;
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        threadlocal_sum += __expf(logits_vec[j] - block_max);
      }
      float new_max = max(running_max, block_max);
      threadlocal_running_denominator =
          threadlocal_running_denominator * __expf(running_max - new_max) +
          threadlocal_sum * __expf(block_max - new_max);
      running_max = new_max;
    }
  }

  running_denominator = cub::BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                            .Sum(threadlocal_running_denominator);
  if (tx == 0) {
    temp_storage.shared_state.denominator = running_denominator;
  }
  __syncthreads();
  running_denominator = temp_storage.shared_state.denominator;

  const float final_max = running_max;
  const float inv_denominator = 1.0f / running_denominator;

  // Pass 2: Normalize in place
  vec_t<DType, VEC_SIZE> prob_vec;
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    if constexpr (CACHE_INPUT) {
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        logits_vec.load(smem_vec_base + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }
    } else {
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        logits_vec.cast_load(logits + bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);

#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          logits_vec[j] *= inv_temp;
        }
      }
    }

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      float p = __expf(static_cast<float>(logits_vec[j]) - final_max) * inv_denominator;
      prob_vec[j] = static_cast<DType>(p);
    }

    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      prob_vec.cast_store(output + bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, typename DType>
__global__ void OnlineSoftmaxMapKernel(DType* logits, PartialSoftmaxResult* partial_results,
                                       DType* temperature_arr, float temperature_val, uint32_t d,
                                       uint32_t num_slices) {
  const uint32_t bx = blockIdx.x;
  const uint32_t by = blockIdx.y;  // slice index
  const uint32_t tx = threadIdx.x;
  float temperature = temperature_arr == nullptr ? temperature_val : temperature_arr[bx];
  const float inv_temp = (temperature == 0.f) ? 0.f : 1.f / temperature;

  const uint32_t vec_alignment_elems = alignof(vec_t<DType, VEC_SIZE>) / sizeof(DType);
  const uint32_t slice_stride = round_up(ceil_div(d, num_slices), vec_alignment_elems);
  const uint32_t slice_start = by * slice_stride;
  const uint32_t slice_size = min((by + 1) * slice_stride, d) - slice_start;

  if (slice_start >= d) return;

  using TempStorage = OnlineSoftmaxTempStorage<BLOCK_THREADS>;
  extern __shared__ __align__(alignof(TempStorage)) uint8_t smem[];
  auto& temp_storage = reinterpret_cast<TempStorage&>(smem);

  vec_t<DType, VEC_SIZE> logits_vec;
  float running_max = -cuda::std::numeric_limits<float>::infinity();
  float running_denominator = 0.0f;
  float threadlocal_running_denominator = 0.0f;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

#pragma unroll 2
  for (uint32_t i = 0; i < ceil_div(slice_size, BLOCK_THREADS * VEC_SIZE); ++i) {
    logits_vec.fill(-cuda::std::numeric_limits<DType>::infinity());

    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < slice_size) {
      logits_vec.cast_load(logits + bx * d + slice_start + (i * BLOCK_THREADS + tx) * VEC_SIZE);
    }

    float thread_max = -cuda::std::numeric_limits<float>::infinity();
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      logits_vec[j] *= inv_temp;
      thread_max = max(thread_max, logits_vec[j]);
    }

    float block_max = cub::BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                          .Reduce(thread_max, MaxReduceOp{});

    if (tx == 0) {
      temp_storage.shared_state.max_val = block_max;
    }
    __syncthreads();
    block_max = temp_storage.shared_state.max_val;

    // if block_max is -inf, then this block contains all -inf values, so we can skip updating
    if (!isinf(block_max)) {
      float threadlocal_sum = 0.0f;
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        threadlocal_sum += __expf(logits_vec[j] - block_max);
      }
      float new_max = max(running_max, block_max);
      threadlocal_running_denominator =
          threadlocal_running_denominator * __expf(running_max - new_max) +
          threadlocal_sum * __expf(block_max - new_max);
      running_max = new_max;
    }
  }

  running_denominator = cub::BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                            .Sum(threadlocal_running_denominator);
  if (tx == 0) {
    temp_storage.shared_state.denominator = running_denominator;
  }
  __syncthreads();
  running_denominator = temp_storage.shared_state.denominator;

  if (tx == 0) {
    partial_results[bx * num_slices + by] = {running_max, running_denominator};
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, typename DType>
__global__ void OnlineSoftmaxReduceKernel(DType* logits, DType* output,
                                          PartialSoftmaxResult* partial_results,
                                          DType* temperature_arr, float temperature_val, uint32_t d,
                                          uint32_t num_slices) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  float temperature = temperature_arr == nullptr ? temperature_val : temperature_arr[bx];
  const float inv_temp = (temperature == 0.f) ? 0.f : 1.f / temperature;

  // Reduce slice results
  using TempStorage = OnlineSoftmaxTempStorage<BLOCK_THREADS>;
  extern __shared__ __align__(alignof(TempStorage)) uint8_t smem[];
  auto& temp_storage = reinterpret_cast<TempStorage&>(smem);

  const Float2SoftmaxReduceOp reduce_op;

  float2 thread_aggregate = make_float2(-cuda::std::numeric_limits<float>::infinity(), 0.0f);

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  for (uint32_t i = tx; i < num_slices; i += BLOCK_THREADS) {
    PartialSoftmaxResult partial = partial_results[bx * num_slices + i];
    float2 partial_pair = make_float2(partial.max_val, partial.denominator);
    thread_aggregate = reduce_op(thread_aggregate, partial_pair);
  }

  float2 block_result = cub::BlockReduce<float2, BLOCK_THREADS>(temp_storage.block_prim.reduce_pair)
                            .Reduce(thread_aggregate, reduce_op);

  if (tx == 0) {
    temp_storage.shared_state.max_val = block_result.x;
    temp_storage.shared_state.denominator = block_result.y;
  }
  __syncthreads();

  block_result =
      make_float2(temp_storage.shared_state.max_val, temp_storage.shared_state.denominator);

  const float final_max = temp_storage.shared_state.max_val;
  const float inv_denominator = 1.0f / temp_storage.shared_state.denominator;

  // Apply normalization
  vec_t<DType, VEC_SIZE> logits_vec;
  vec_t<DType, VEC_SIZE> prob_vec;

  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    logits_vec.fill(-cuda::std::numeric_limits<DType>::infinity());

    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      logits_vec.cast_load(logits + bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
    }

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      logits_vec[j] *= inv_temp;
      float p = __expf(static_cast<float>(logits_vec[j]) - final_max) * inv_denominator;
      prob_vec[j] = static_cast<DType>(p);
    }

    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      prob_vec.cast_store(output + bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, bool DETERMINISTIC, typename Predicate>
__device__ __forceinline__ void DeviceSamplingFromProb(
    uint32_t i, uint32_t d, Predicate pred, float u, vec_t<float, VEC_SIZE> prob_vec,
    float& aggregate,
    SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>* temp_storage) {
  const uint32_t tx = threadIdx.x;
  float prob_greater_than_threshold[VEC_SIZE];
  float inclusive_cdf[VEC_SIZE];
  bool greater_than_u[VEC_SIZE], valid[VEC_SIZE];
#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    prob_greater_than_threshold[j] = pred(prob_vec[j]) ? prob_vec[j] : 0;
    valid[j] = pred(prob_vec[j]) && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d;
  }
  float aggregate_local =
      BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce)
          .template Sum<VEC_SIZE>(prob_greater_than_threshold);
  if (tx == 0) {
    temp_storage->block_aggregate.value = aggregate_local;
  }
  __syncthreads();
  aggregate_local = temp_storage->block_aggregate.value;

  if (aggregate + aggregate_local > u) {
    if constexpr (DETERMINISTIC) {
      DeterministicInclusiveSum<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>(
          prob_greater_than_threshold, inclusive_cdf, temp_storage);
    } else {
      BlockScan<float, BLOCK_THREADS, SCAN_ALGORITHM>(temp_storage->block_prim.scan)
          .template InclusiveSum<VEC_SIZE>(prob_greater_than_threshold, inclusive_cdf);

      __syncthreads();
    }

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      greater_than_u[j] = (inclusive_cdf[j] + aggregate > u) && valid[j];
    }

    bool greater_than_u_diff[VEC_SIZE];
#ifdef FLASHINFER_CUB_SUBTRACTLEFT_DEFINED
    BlockAdjacentDifference<bool, BLOCK_THREADS>(temp_storage->block_prim.adj_diff)
        .SubtractLeft<VEC_SIZE>(greater_than_u, greater_than_u_diff, BoolDiffOp());
#else
    BlockAdjacentDifference<bool, BLOCK_THREADS>(temp_storage->block_prim.adj_diff)
        .template FlagHeads<VEC_SIZE>(greater_than_u_diff, greater_than_u, BoolDiffOp(), 0);
#endif
    __syncthreads();

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      if (greater_than_u_diff[j]) {
        atomicMin(&(temp_storage->sampled_id), (i * BLOCK_THREADS + tx) * VEC_SIZE + j);
      }
    }
    __syncthreads();
  }

  // update the last valid index
  int valid_index[VEC_SIZE];
#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    if (valid[j]) {
      valid_index[j] = (i * BLOCK_THREADS + tx) * VEC_SIZE + j;
    } else {
      valid_index[j] = -1;
    }
  }
  int max_valid_index =
      BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce_int)
          .Reduce(valid_index, MaxReduceOp{});
  if (tx == 0 && max_valid_index != -1) {
    temp_storage->last_valid_id = max_valid_index;
  }
  __syncthreads();
  aggregate += aggregate_local;
}

template <typename DType, typename IdType>
struct DataAndIndex {
  DType data;
  IdType index;

  __device__ DataAndIndex operator+(const DataAndIndex& other) const {
    if (data > other.data) {
      return {data, index};
    } else {
      return {other.data, other.index};
    }
  }
  __device__ DataAndIndex& operator+=(const DataAndIndex& other) {
    if (data > other.data) {
      return *this;
    } else {
      data = other.data;
      index = other.index;
      return *this;
    }
  }
};

template <typename DType, uint32_t VEC_SIZE>
__device__ __forceinline__ vec_t<DType, VEC_SIZE> GenerateGumbelNoise(uint64_t philox_seed,
                                                                      uint64_t philox_offset,
                                                                      uint64_t subsequence) {
  curandStatePhilox4_32_10_t state;
  vec_t<float, VEC_SIZE> noise;
  constexpr float kEPSILON = 1e-20f;
  constexpr float kLOG2 = 0.6931471806f;
  auto uniform2gumbel = [](float x) { return -kLOG2 * log2f(-log2f(x + kEPSILON) + kEPSILON); };
// TODO: compare the speed of log2 and log
#pragma unroll
  for (uint32_t i = 0; i + 4 <= VEC_SIZE; i += 4) {
    curand_init(philox_seed, subsequence + i, philox_offset, &state);
    float4 noise_vec = curand_uniform4(&state);
    noise[i] = uniform2gumbel(noise_vec.x);
    noise[i + 1] = uniform2gumbel(noise_vec.y);
    noise[i + 2] = uniform2gumbel(noise_vec.z);
    noise[i + 3] = uniform2gumbel(noise_vec.w);
  }
  if constexpr (VEC_SIZE % 4 != 0) {
    curand_init(philox_seed, subsequence + VEC_SIZE / 4 * 4, philox_offset, &state);
    float4 noise_vec = curand_uniform4(&state);
    if constexpr (VEC_SIZE % 4 == 1) {
      noise[VEC_SIZE - 1] = uniform2gumbel(noise_vec.x);
    } else if constexpr (VEC_SIZE % 4 == 2) {
      noise[VEC_SIZE - 2] = uniform2gumbel(noise_vec.x);
      noise[VEC_SIZE - 1] = uniform2gumbel(noise_vec.y);
    } else if constexpr (VEC_SIZE % 4 == 3) {
      noise[VEC_SIZE - 3] = uniform2gumbel(noise_vec.x);
      noise[VEC_SIZE - 2] = uniform2gumbel(noise_vec.y);
      noise[VEC_SIZE - 1] = uniform2gumbel(noise_vec.z);
    }
  }

  if constexpr (std::is_same_v<DType, float>) {
    return noise;
  } else {
    vec_t<DType, VEC_SIZE> ret;
#pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; ++i) {
      ret[i] = static_cast<DType>(noise[i]);
    }
    return ret;
  }
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType>
__global__ void SamplingFromLogitsKernel(DType* logits, IdType* output, IdType* indices, uint32_t d,
                                         uint64_t philox_seed, uint64_t philox_offset) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  const uint32_t row_idx = indices == nullptr ? bx : indices[bx];
  using SharedMem = typename BlockReduce<DataAndIndex<DType, IdType>, BLOCK_THREADS,
                                         REDUCE_ALGORITHM>::TempStorage;
  extern __shared__ __align__(alignof(SharedMem)) uint8_t smem_sampling_logit[];
  auto& temp_storage = reinterpret_cast<SharedMem&>(smem_sampling_logit);

  vec_t<DType, VEC_SIZE> logits_vec;
  DataAndIndex<DType, IdType> max_data = {-cuda::std::numeric_limits<DType>::infinity(), 0};
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    logits_vec.fill(-cuda::std::numeric_limits<DType>::infinity());
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      logits_vec.cast_load(logits + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
    }

    vec_t<DType, VEC_SIZE> gumbel_noise = GenerateGumbelNoise<DType, VEC_SIZE>(
        philox_seed, philox_offset,
        static_cast<uint64_t>(bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE));
    DataAndIndex<DType, IdType> cur_data[VEC_SIZE];
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      cur_data[j].data = (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d
                             ? logits_vec[j] + gumbel_noise[j]
                             : -cuda::std::numeric_limits<DType>::infinity();
      cur_data[j].index = (i * BLOCK_THREADS + tx) * VEC_SIZE + j;
    }

    max_data +=
        BlockReduce<DataAndIndex<DType, IdType>, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage)
            .template Sum<VEC_SIZE>(cur_data);
  }
  if (tx == 0) {
    output[bx] = max_data.index;
  }
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType>
__global__ void SamplingFromProbKernel(DType* probs, IdType* output, IdType* indices, uint32_t d,
                                       uint64_t philox_seed, uint64_t philox_offset) {
  curandStatePhilox4_32_10_t state;
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  curand_init(philox_seed, bx, philox_offset, &state);
  const uint32_t row_idx = indices == nullptr ? bx : indices[bx];

  extern __shared__ __align__(
      alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(
          smem_sampling);
  temp_storage.sampled_id = d;
  __syncthreads();

  vec_t<float, VEC_SIZE> probs_vec;
  float aggregate(0);
  float u = curand_uniform(&state);

#pragma unroll 2
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    probs_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      probs_vec.cast_load(probs + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
    }

    DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM,
                           DETERMINISTIC>(
        i, d, [](float x) { return x > 0; }, u, probs_vec, aggregate, &temp_storage);
    if (float(aggregate) > u) {
      break;
    }
  }
  int sampled_id = temp_storage.sampled_id;
  if (sampled_id == d) {
    // NOTE(Zihao): this would happen when u is very close to 1
    // and the sum of probabilities is smaller than u
    // In this case, we use the last valid index as the sampled id
    sampled_id = temp_storage.last_valid_id;
  }
  output[bx] = sampled_id;
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType>
__global__ void TopKSamplingFromProbKernel(DType* probs, IdType* output, IdType* indices,
                                           IdType* top_k_arr, uint32_t top_k_val, uint32_t d,
                                           uint64_t philox_seed, uint64_t philox_offset) {
  const uint32_t batch_size = gridDim.x;
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(philox_seed, bx, philox_offset, &state);
  const uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[bx];
  const uint32_t row_idx = indices == nullptr ? bx : indices[bx];

  extern __shared__ __align__(
      alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(
          smem_sampling);

  vec_t<float, VEC_SIZE> probs_vec;
  float aggregate;
  float q = 1;
  double low = 0, high = 1.f;
  int sampled_id;
  int round = 0;
  do {
    round += 1;
    temp_storage.sampled_id = d;
    __syncthreads();
    float u = curand_uniform(&state) * q;
    aggregate = 0;
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0);
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM,
                             DETERMINISTIC>(
          i, d, [&](float x) { return x > low; }, u, probs_vec, aggregate, &temp_storage);
      if (aggregate > u) {
        break;
      }
    }
    __syncthreads();
    sampled_id = temp_storage.sampled_id;
    if (sampled_id == d) {
      // NOTE(Zihao): this would happen when u is very close to 1
      // and the sum of probabilities is smaller than u
      // In this case, we use the last valid index as the sampled id
      sampled_id = temp_storage.last_valid_id;
    }
    double pivot_0 = probs[row_idx * d + sampled_id];
    double pivot_1 = (pivot_0 + high) / 2;

    ValueCount<float> aggregate_gt_pivot_0{0, 0}, aggregate_gt_pivot_1{0, 0};
    ValueCount<float> threadlocal_gt_pivot_0{0, 0}, threadlocal_gt_pivot_1{0, 0};
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0);
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      ValueCount<float> probs_gt_pivot_0[VEC_SIZE], probs_gt_pivot_1[VEC_SIZE];
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        probs_gt_pivot_0[j] = {
            (probs_vec[j] > pivot_0) ? probs_vec[j] : 0,
            (probs_vec[j] > pivot_0 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};
        probs_gt_pivot_1[j] = {
            (probs_vec[j] > pivot_1) ? probs_vec[j] : 0,
            (probs_vec[j] > pivot_1 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};
        threadlocal_gt_pivot_0 += probs_gt_pivot_0[j];
        threadlocal_gt_pivot_1 += probs_gt_pivot_1[j];
      }
    }
    aggregate_gt_pivot_0 += BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>(
                                temp_storage.block_prim.reduce_value_count)
                                .Sum(threadlocal_gt_pivot_0);
    if (tx == 0) {
      temp_storage.block_aggregate.pair = aggregate_gt_pivot_0;
    }
    __syncthreads();
    aggregate_gt_pivot_0 = temp_storage.block_aggregate.pair;

    aggregate_gt_pivot_1 += BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>(
                                temp_storage.block_prim.reduce_value_count)
                                .Sum(threadlocal_gt_pivot_1);
    if (tx == 0) {
      temp_storage.block_aggregate.pair = aggregate_gt_pivot_1;
    }
    __syncthreads();
    aggregate_gt_pivot_1 = temp_storage.block_aggregate.pair;
    if (aggregate_gt_pivot_0.count < k) {
      // case 1: pivot_0 accepted
      break;
    }
    if (aggregate_gt_pivot_1.count < k) {
      // case 2: pivot_0 rejected, pivot_1 accepted
      low = pivot_0;
      high = pivot_1;
      q = aggregate_gt_pivot_0.value;
    } else {
      // case 3: pivot_0 rejected, pivot_1 rejected
      low = pivot_1;
      q = aggregate_gt_pivot_1.value;
    }
  } while (low < high);
  __syncthreads();
  if (tx == 0) {
    output[bx] = sampled_id;
  }
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType>
__global__ void TopPSamplingFromProbKernel(DType* probs, IdType* output, IdType* indices,
                                           float* top_p_arr, float top_p_val, uint32_t d,
                                           uint64_t philox_seed, uint64_t philox_offset) {
  const uint32_t batch_size = gridDim.x;
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(philox_seed, bx, philox_offset, &state);
  const uint32_t row_idx = indices == nullptr ? bx : indices[bx];
  float top_p = (top_p_arr == nullptr) ? top_p_val : top_p_arr[row_idx];

  extern __shared__ __align__(
      alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(
          smem_sampling);

  vec_t<float, VEC_SIZE> probs_vec;
  float aggregate;
  float q = 1;
  double low = 0, high = 1.f;
  int sampled_id;
  do {
    temp_storage.sampled_id = d;
    __syncthreads();
    float u = curand_uniform(&state) * q;
    aggregate = 0;
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0);
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM,
                             DETERMINISTIC>(
          i, d, [&](float x) { return x > low; }, u, probs_vec, aggregate, &temp_storage);
      if (aggregate > u) {
        break;
      }
    }
    __syncthreads();
    sampled_id = temp_storage.sampled_id;
    if (sampled_id == d) {
      // NOTE(Zihao): this would happen when u is very close to 1
      // and the sum of probabilities is smaller than u
      // In this case, we use the last valid index as the sampled id
      sampled_id = temp_storage.last_valid_id;
    }
    double pivot_0 = probs[row_idx * d + sampled_id];
    double pivot_1 = (pivot_0 + high) / 2;

    float aggregate_gt_pivot_0 = 0, aggregate_gt_pivot_1 = 0;
    float threadlocal_aggregate_gt_pivot_0 = 0;
    float threadlocal_aggregate_gt_pivot_1 = 0;
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0);
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      float probs_gt_pivot_0[VEC_SIZE], probs_gt_pivot_1[VEC_SIZE];
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        probs_gt_pivot_0[j] = (probs_vec[j] > pivot_0) ? probs_vec[j] : 0;
        probs_gt_pivot_1[j] = (probs_vec[j] > pivot_1) ? probs_vec[j] : 0;
        threadlocal_aggregate_gt_pivot_0 += probs_gt_pivot_0[j];
        threadlocal_aggregate_gt_pivot_1 += probs_gt_pivot_1[j];
      }
    }
    aggregate_gt_pivot_0 += BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                                .Sum(threadlocal_aggregate_gt_pivot_0);
    if (tx == 0) {
      temp_storage.block_aggregate.value = aggregate_gt_pivot_0;
    }
    __syncthreads();
    aggregate_gt_pivot_0 = temp_storage.block_aggregate.value;

    aggregate_gt_pivot_1 += BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                                .Sum(threadlocal_aggregate_gt_pivot_1);
    if (tx == 0) {
      temp_storage.block_aggregate.value = aggregate_gt_pivot_1;
    }
    __syncthreads();
    aggregate_gt_pivot_1 = temp_storage.block_aggregate.value;

    if (aggregate_gt_pivot_0 < top_p) {
      // case 1: pivot_0 accepted
      break;
    }
    if (aggregate_gt_pivot_1 < top_p) {
      // case 2: pivot_0 rejected, pivot_1 accepted
      low = pivot_0;
      high = pivot_1;
      q = aggregate_gt_pivot_0;
    } else {
      // case 3: pivot_0 rejected, pivot_1 rejected
      low = pivot_1;
      q = aggregate_gt_pivot_1;
    }
  } while (low < high);
  __syncthreads();
  if (tx == 0) {
    output[bx] = sampled_id;
  }
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType>
__global__ void MinPSamplingFromProbKernel(DType* probs, float* min_p_arr, IdType* output,
                                           IdType* indices, float min_p_val, uint32_t d,
                                           uint64_t philox_seed, uint64_t philox_offset) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  float p = (min_p_arr == nullptr) ? min_p_val : min_p_arr[bx];
  curandStatePhilox4_32_10_t state;
  curand_init(philox_seed, bx, philox_offset, &state);
  const uint32_t row_idx = indices == nullptr ? bx : indices[bx];

  extern __shared__ __align__(
      alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(
          smem_sampling);

  float max_val = GetMaxValue<VEC_SIZE, BLOCK_THREADS, REDUCE_ALGORITHM,
                              SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>>(
      probs, row_idx, d, temp_storage);
  float pivot = max_val * p;

  vec_t<float, VEC_SIZE> probs_vec;
  float aggregate_gt_pivot = 0;
  float threadlocal_aggregate_gt_pivot = 0;
#pragma unroll 2
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    probs_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
    }

    float probs_gt_pivot[VEC_SIZE];
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      probs_gt_pivot[j] = (probs_vec[j] >= pivot) ? probs_vec[j] : 0;
      threadlocal_aggregate_gt_pivot += probs_gt_pivot[j];
    }
  }

  aggregate_gt_pivot += BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                            .Sum(threadlocal_aggregate_gt_pivot);
  if (tx == 0) {
    temp_storage.block_aggregate.value = aggregate_gt_pivot;
  }
  __syncthreads();

  float aggregate = 0;
  float q = temp_storage.block_aggregate.value;

  int sampled_id;
  temp_storage.sampled_id = d;
  __syncthreads();
  float u = curand_uniform(&state) * q;
#pragma unroll 2
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    probs_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
    }

    DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM,
                           DETERMINISTIC>(
        i, d, [&](float x) { return x >= pivot; }, u, probs_vec, aggregate, &temp_storage);
    if (aggregate > u) {
      break;
    }
  }
  sampled_id = temp_storage.sampled_id;
  if (sampled_id == d) {
    // NOTE(Zihao): this would happen when u is very close to 1
    // and the sum of probabilities is smaller than u
    // In this case, we use the last valid index as the sampled id
    sampled_id = temp_storage.last_valid_id;
  }
  output[bx] = sampled_id;
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType>
__global__ void TopKTopPSamplingFromProbKernel(DType* probs, IdType* top_k_arr, float* top_p_arr,
                                               IdType* output, IdType* indices, IdType top_k_val,
                                               float top_p_val, uint32_t d, uint64_t philox_seed,
                                               uint64_t philox_offset) {
  const uint32_t batch_size = gridDim.x;
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(philox_seed, bx, philox_offset, &state);
  const uint32_t row_idx = indices == nullptr ? bx : indices[bx];
  const uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[row_idx];
  const float p = top_p_arr == nullptr ? top_p_val : top_p_arr[row_idx];

  extern __shared__ __align__(
      alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(
          smem_sampling);

  vec_t<float, VEC_SIZE> probs_vec;
  float aggregate;
  float q = 1;
  double low = 0, high = 1.f;
  int sampled_id;
  do {
    temp_storage.sampled_id = d;
    __syncthreads();
    float u = curand_uniform(&state) * q;
    aggregate = 0;
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0);
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM,
                             DETERMINISTIC>(
          i, d, [&](float x) { return x > low; }, u, probs_vec, aggregate, &temp_storage);
      if (aggregate > u) {
        break;
      }
    }
    __syncthreads();
    sampled_id = temp_storage.sampled_id;
    if (sampled_id == d) {
      // NOTE(Zihao): this would happen when u is very close to 1
      // and the sum of probabilities is smaller than u
      // In this case, we use the last valid index as the sampled id
      sampled_id = temp_storage.last_valid_id;
    }
    double pivot_0 = probs[row_idx * d + sampled_id];
    double pivot_1 = (pivot_0 + high) / 2;

    ValueCount<float> aggregate_gt_pivot_0{0, 0}, aggregate_gt_pivot_1{0, 0};
    ValueCount<float> threadlocal_aggregate_gt_pivot_0{0, 0};
    ValueCount<float> threadlocal_aggregate_gt_pivot_1{0, 0};
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0);
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      ValueCount<float> probs_gt_pivot_0[VEC_SIZE], probs_gt_pivot_1[VEC_SIZE];
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        probs_gt_pivot_0[j] = {
            (probs_vec[j] > pivot_0) ? probs_vec[j] : 0,
            (probs_vec[j] > pivot_0 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};
        probs_gt_pivot_1[j] = {
            (probs_vec[j] > pivot_1) ? probs_vec[j] : 0,
            (probs_vec[j] > pivot_1 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};
        threadlocal_aggregate_gt_pivot_0 += probs_gt_pivot_0[j];
        threadlocal_aggregate_gt_pivot_1 += probs_gt_pivot_1[j];
      }
    }
    aggregate_gt_pivot_0 +=
        BlockReduce<ValueCount<float>, BLOCK_THREADS>(temp_storage.block_prim.reduce_value_count)
            .Sum(threadlocal_aggregate_gt_pivot_0);
    if (tx == 0) {
      temp_storage.block_aggregate.pair = aggregate_gt_pivot_0;
    }
    __syncthreads();
    aggregate_gt_pivot_0 = temp_storage.block_aggregate.pair;

    aggregate_gt_pivot_1 +=
        BlockReduce<ValueCount<float>, BLOCK_THREADS>(temp_storage.block_prim.reduce_value_count)
            .Sum(threadlocal_aggregate_gt_pivot_1);
    if (tx == 0) {
      temp_storage.block_aggregate.pair = aggregate_gt_pivot_1;
    }
    __syncthreads();
    aggregate_gt_pivot_1 = temp_storage.block_aggregate.pair;
    if (aggregate_gt_pivot_0.count < k && aggregate_gt_pivot_0.value < p) {
      // case 1: pivot_0 accepted
      break;
    }
    if (aggregate_gt_pivot_1.count < k && aggregate_gt_pivot_1.value < p) {
      // case 2: pivot_0 rejected, pivot_1 accepted
      low = pivot_0;
      high = pivot_1;
      q = aggregate_gt_pivot_0.value;
    } else {
      // case 3: pivot_0 rejected, pivot_1 rejected
      low = pivot_1;
      q = aggregate_gt_pivot_1.value;
    }
  } while (low < high);
  __syncthreads();
  if (tx == 0) {
    output[bx] = sampled_id;
  }
}

template <typename DType>
cudaError_t OnlineSoftmax(DType* logits, DType* output, uint32_t batch_size, uint32_t d,
                          DType* temperature_arr, DType temperature_val, void* workspace_buffer,
                          size_t workspace_buffer_size_in_bytes, bool enable_pdl,
                          cudaStream_t stream = 0) {
  constexpr uint32_t SMALL_BATCH_THRESHOLD = 128;
  constexpr uint32_t LARGE_VOCAB_THRESHOLD = 24576;
  constexpr uint32_t DEFAULT_SLICE_SIZE = 8192;

  const uint32_t vec_size = std::gcd(16 / sizeof(DType), d);
  auto compute_capacity = GetCudaComputeCapability();

  DISPATCH_COMPUTE_CAP_NUM_THREADS(
      compute_capacity, BLOCK_THREADS, {DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
        if (batch_size <= SMALL_BATCH_THRESHOLD && d >= LARGE_VOCAB_THRESHOLD) {
          // Path A: Vocab-Splitting Strategy for small-batch & large-vocab
          uint32_t num_slices = ceil_div(d, DEFAULT_SLICE_SIZE);

          const size_t partial_buffer_size = batch_size * num_slices * sizeof(PartialSoftmaxResult);
          if (workspace_buffer_size_in_bytes < partial_buffer_size) {
            return cudaErrorInvalidValue;
          }

          AlignedAllocator allocator(workspace_buffer, workspace_buffer_size_in_bytes);
          auto partial_results = allocator.aligned_alloc<PartialSoftmaxResult>(
              partial_buffer_size, alignof(PartialSoftmaxResult), "softmax_workspace");

          // Phase 1: Map-Reduce across vocab slices
          dim3 phase1_nblks(batch_size, num_slices);
          dim3 phase1_nthrs(BLOCK_THREADS);
          size_t smem_size = sizeof(OnlineSoftmaxTempStorage<BLOCK_THREADS>);

          auto phase1_kernel = OnlineSoftmaxMapKernel<BLOCK_THREADS, VEC_SIZE, DType>;
          void* phase1_args[] = {&logits, &partial_results, &temperature_arr, &temperature_val,
                                 &d,      &num_slices};

          FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
              phase1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

          if (enable_pdl) {
            cudaLaunchAttribute attribute[1];
            attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attribute[0].val.programmaticStreamSerializationAllowed = 1;

            cudaLaunchConfig_t config;
            config.gridDim = phase1_nblks;
            config.blockDim = phase1_nthrs;
            config.dynamicSmemBytes = smem_size;
            config.stream = stream;
            config.attrs = attribute;
            config.numAttrs = 1;

            FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, phase1_kernel, logits, partial_results,
                                                    temperature_arr, temperature_val, d,
                                                    num_slices));
          } else {
            FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)phase1_kernel, phase1_nblks, phase1_nthrs,
                                                  phase1_args, smem_size, stream));
          }

          // Phase 2: Final reduction and apply normalization
          dim3 phase2_nblks(batch_size);
          dim3 phase2_nthrs(BLOCK_THREADS);

          auto phase2_kernel = OnlineSoftmaxReduceKernel<BLOCK_THREADS, VEC_SIZE, DType>;
          void* phase2_args[] = {&logits,          &output, &partial_results, &temperature_arr,
                                 &temperature_val, &d,      &num_slices};

          FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
              phase2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

          if (enable_pdl) {
            cudaLaunchAttribute attribute[1];
            attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attribute[0].val.programmaticStreamSerializationAllowed = 1;

            cudaLaunchConfig_t config;
            config.gridDim = phase2_nblks;
            config.blockDim = phase2_nthrs;
            config.dynamicSmemBytes = smem_size;
            config.stream = stream;
            config.attrs = attribute;
            config.numAttrs = 1;

            FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, phase2_kernel, logits, output,
                                                    partial_results, temperature_arr,
                                                    temperature_val, d, num_slices));
          } else {
            FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)phase2_kernel, phase2_nblks, phase2_nthrs,
                                                  phase2_args, smem_size, stream));
          }
        } else {
          // Path B: Single-Block Strategy
          // Switch input cache
          uint32_t cache_threshold;
          if (batch_size <= 16) {
            cache_threshold = 4096;
          } else if (batch_size <= 32) {
            cache_threshold = 2048;
          } else {
            cache_threshold = 0;
          }
          const bool cache_input = d <= cache_threshold;

          dim3 nblks(batch_size);
          dim3 nthrs(BLOCK_THREADS);
          void* args[] = {&logits, &output, &temperature_arr, &temperature_val, &d};

          const size_t smem_logits_bytes = (round_up(d, VEC_SIZE) + VEC_SIZE) * sizeof(DType);

          uint32_t smem_size = sizeof(OnlineSoftmaxTempStorage<BLOCK_THREADS>) +
                               (cache_input ? smem_logits_bytes : 0);

          DISPATCH_SOFTMAX_CACHE_INPUT(cache_input, CACHE_INPUT, {
            auto kernel = OnlineSoftmaxFusedKernel<BLOCK_THREADS, VEC_SIZE, DType, CACHE_INPUT>;
            FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

            if (enable_pdl) {
              cudaLaunchAttribute attribute[1];
              attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
              attribute[0].val.programmaticStreamSerializationAllowed = 1;

              cudaLaunchConfig_t config;
              config.gridDim = nblks;
              config.blockDim = nthrs;
              config.dynamicSmemBytes = smem_size;
              config.stream = stream;
              config.attrs = attribute;
              config.numAttrs = 1;

              FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, logits, output,
                                                      temperature_arr, temperature_val, d));
            } else {
              FLASHINFER_CUDA_CALL(
                  cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
            }
          });
        }
      })});
  return cudaSuccess;
}

template <typename T, typename IdType>
cudaError_t SamplingFromLogits(T* logits, IdType* output, IdType* indices, uint32_t batch_size,
                               uint32_t d, bool deterministic, uint64_t philox_seed,
                               uint64_t philox_offset, cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
    dim3 nblks(batch_size);
    dim3 nthrs(BLOCK_THREADS);
    void* args[] = {&logits, &output, &indices, &d, &philox_seed, &philox_offset};
    const uint32_t smem_size = sizeof(
        typename BlockReduce<DataAndIndex<T, IdType>, BLOCK_THREADS, REDUCE_ALGO>::TempStorage);

    DISPATCH_ALIGNED_VEC_SIZE(
        vec_size, VEC_SIZE, {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
          auto kernel = SamplingFromLogitsKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE,
                                                 DETERMINISTIC, T, IdType>;
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        })});
    return cudaSuccess;
  });
}

template <typename T, typename IdType>
cudaError_t SamplingFromProb(T* probs, IdType* output, IdType* indices, uint32_t batch_size,
                             uint32_t d, bool deterministic, uint64_t philox_seed,
                             uint64_t philox_offset, cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
    dim3 nblks(batch_size);
    dim3 nthrs(BLOCK_THREADS);
    void* args[] = {&probs, &output, &indices, &d, &philox_seed, &philox_offset};
    const uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);

    DISPATCH_ALIGNED_VEC_SIZE(
        vec_size, VEC_SIZE, {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
          auto kernel = SamplingFromProbKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE,
                                               DETERMINISTIC, T, IdType>;
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        })});
    return cudaSuccess;
  });
}

template <typename T, typename IdType>
cudaError_t TopKSamplingFromProb(T* probs, IdType* output, IdType* indices, T* top_k_arr,
                                 uint32_t batch_size, uint32_t top_k_val, uint32_t d,
                                 bool deterministic, uint64_t philox_seed, uint64_t philox_offset,
                                 cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
    const uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
    dim3 nblks(batch_size);
    dim3 nthrs(BLOCK_THREADS);
    void* args[] = {&probs,     &output, &indices,     &top_k_arr,
                    &top_k_val, &d,      &philox_seed, &philox_offset};

    DISPATCH_ALIGNED_VEC_SIZE(
        vec_size, VEC_SIZE, {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
          auto kernel = TopKSamplingFromProbKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE,
                                                   DETERMINISTIC, T, IdType>;
          FLASHINFER_CUDA_CALL(
              cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        })});
    return cudaSuccess;
  });
}

template <typename T, typename IdType>
cudaError_t TopPSamplingFromProb(T* probs, IdType* output, IdType* indices, T* top_p_arr,
                                 uint32_t batch_size, T top_p_val, uint32_t d, bool deterministic,
                                 uint64_t philox_seed, uint64_t philox_offset,
                                 cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
    const uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
    dim3 nblks(batch_size);
    dim3 nthrs(BLOCK_THREADS);
    void* args[] = {&probs,     &output, &indices,     &top_p_arr,
                    &top_p_val, &d,      &philox_seed, &philox_offset};

    DISPATCH_ALIGNED_VEC_SIZE(
        vec_size, VEC_SIZE, {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
          auto kernel = TopPSamplingFromProbKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE,
                                                   DETERMINISTIC, T, IdType>;
          FLASHINFER_CUDA_CALL(
              cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        })});
    return cudaSuccess;
  });
}

template <typename T, typename IdType>
cudaError_t MinPSamplingFromProb(T* probs, T* min_p_arr, IdType* output, IdType* indices,
                                 uint32_t batch_size, float min_p_val, uint32_t d,
                                 bool deterministic, uint64_t philox_seed, uint64_t philox_offset,
                                 cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
    const uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
    dim3 nblks(batch_size);
    dim3 nthrs(BLOCK_THREADS);
    void* args[] = {&probs,     &min_p_arr, &output,      &indices,
                    &min_p_val, &d,         &philox_seed, &philox_offset};

    DISPATCH_ALIGNED_VEC_SIZE(
        vec_size, VEC_SIZE, {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
          auto kernel = MinPSamplingFromProbKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE,
                                                   DETERMINISTIC, T, IdType>;
          FLASHINFER_CUDA_CALL(
              cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        })});
    return cudaSuccess;
  });
}

template <typename T, typename IdType>
cudaError_t TopKTopPSamplingFromProb(T* probs, IdType* top_k_arr, T* top_p_arr, IdType* output,
                                     IdType* indices, uint32_t batch_size, IdType top_k_val,
                                     T top_p_val, uint32_t d, bool deterministic,
                                     uint64_t philox_seed, uint64_t philox_offset,
                                     cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
    const uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
    dim3 nblks(batch_size);
    dim3 nthrs(BLOCK_THREADS);
    void* args[] = {&probs,     &top_k_arr, &top_p_arr, &output,      &indices,
                    &top_k_val, &top_p_val, &d,         &philox_seed, &philox_offset};

    DISPATCH_ALIGNED_VEC_SIZE(
        vec_size, VEC_SIZE, {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
          auto kernel = TopKTopPSamplingFromProbKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO,
                                                       VEC_SIZE, DETERMINISTIC, T, IdType>;
          FLASHINFER_CUDA_CALL(
              cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        })});
    return cudaSuccess;
  });
}

template <uint32_t BLOCK_THREADS, BlockReduceAlgorithm REDUCE_ALGORITHM>
struct RenormTempStorage {
  union {
    typename BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce;
    typename BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce_int;
    typename BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage
        reduce_value_count;
  } block_prim;
  struct {
    float max_val;
    float min_val;
    float row_sum;
    union {
      struct {
        float values[2];
      };
      struct {
        int counts[2];
      };
      struct {
        ValueCount<float> pairs[2];
      };
    } block_aggregate;
  };
};

template <uint32_t BLOCK_THREADS, BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE,
          typename DType>
__global__ void TopPRenormProbKernel(DType* probs, DType* renormed_prob, float* top_p_arr,
                                     float top_p_val, uint32_t d) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  const uint32_t row_idx = bx;
  float p = top_p_arr == nullptr ? top_p_val : top_p_arr[bx];

  extern __shared__ __align__(alignof(RenormTempStorage<BLOCK_THREADS, REDUCE_ALGO>))
      uint8_t smem_renorm[];
  auto& temp_storage =
      reinterpret_cast<RenormTempStorage<BLOCK_THREADS, REDUCE_ALGO>&>(smem_renorm);
  vec_t<float, VEC_SIZE> probs_vec;

  // Fast-path: when p >= 1.0 (e.g., p == 1.0), perform simple sum and normalization
  if (p >= 1.0f) {
    // Stage A: per-thread float accumulation over assigned lanes (vectorized)
    float thread_sum = 0.0f;
    const uint32_t num_iters = ceil_div(d, BLOCK_THREADS * VEC_SIZE);
    for (uint32_t i = 0; i < num_iters; ++i) {
      probs_vec.fill(0.0f);
      const uint32_t base_idx = (i * BLOCK_THREADS + tx) * VEC_SIZE;
      if (base_idx < d) {
        probs_vec.cast_load(probs + row_idx * d + base_idx);
      }
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        const uint32_t idx = base_idx + j;
        if (idx < d) thread_sum += probs_vec[j];
      }
    }

    // Block reduce (float)
    float row_sum =
        BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
            .Sum(thread_sum);
    // Broadcast via shared
    if (tx == 0) temp_storage.row_sum = row_sum;
    __syncthreads();
    row_sum = temp_storage.row_sum;

    // Guard against zero sum
    const float denom = (row_sum <= 1e-8f) ? 1.0f : row_sum;
    const float normalizer = math::ptx_rcp(denom);

    // Stage B: normalize and store
    for (uint32_t i = 0; i < num_iters; ++i) {
      probs_vec.fill(0.0f);
      const uint32_t base_idx = (i * BLOCK_THREADS + tx) * VEC_SIZE;
      if (base_idx < d) {
        probs_vec.cast_load(probs + row_idx * d + base_idx);
      }
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        const uint32_t idx = base_idx + j;
        float v = probs_vec[j];
        probs_vec[j] = (idx < d) ? (v * normalizer) : 0.0f;
      }
      if (base_idx < d) {
        probs_vec.cast_store(renormed_prob + row_idx * d + base_idx);
      }
    }
    return;  // Exit after fast-path processing
  }

  // Original Top-P renormalization logic
  temp_storage.max_val = 0;
  float max_val = GetMaxValue<VEC_SIZE, BLOCK_THREADS, REDUCE_ALGORITHM,
                              RenormTempStorage<BLOCK_THREADS, REDUCE_ALGORITHM>>(probs, row_idx, d,
                                                                                  temp_storage);

  double low = 0, high = max_val;
  float min_gt_low, max_le_high;
  float sum_low = 1;
  // f(x) = sum(probs[probs > x]), f(x) is non-increasing
  // min_gt_low = min{p \in probs | p > low}, max_le_high = max{p \in probs | p <= high}
  // loop invariant:
  // - f(low) >= p, f(high) < p
  // - f(low) > f(min_gt_low) >= f(max_le_high) == f(high)
  // stopping condition
  // - f(low) >= p, f(min_gt_low) == f(max_le_high) == f(high) < p
  do {
    double pivot_0 = (high + 2 * low) / 3;
    double pivot_1 = (2 * high + low) / 3;

    float aggregate_gt_pivot_0 = 0, aggregate_gt_pivot_1 = 0;
    min_gt_low = high;
    max_le_high = low;
    float threadlocal_aggregate_gt_pivot_0 = 0;
    float threadlocal_aggregate_gt_pivot_1 = 0;
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0);
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.cast_load(probs + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      }

      float probs_gt_pivot_0[VEC_SIZE], probs_gt_pivot_1[VEC_SIZE];
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        probs_gt_pivot_0[j] = (probs_vec[j] > pivot_0) ? probs_vec[j] : 0;
        probs_gt_pivot_1[j] = (probs_vec[j] > pivot_1) ? probs_vec[j] : 0;

        if (probs_vec[j] > low && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d) {
          min_gt_low = min(min_gt_low, probs_vec[j]);
        }
        if (probs_vec[j] <= high && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d) {
          max_le_high = max(max_le_high, probs_vec[j]);
        }
        threadlocal_aggregate_gt_pivot_0 += probs_gt_pivot_0[j];
        threadlocal_aggregate_gt_pivot_1 += probs_gt_pivot_1[j];
      }
    }
    aggregate_gt_pivot_0 =
        BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
            .Sum(threadlocal_aggregate_gt_pivot_0);
    __syncthreads();
    aggregate_gt_pivot_1 =
        BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
            .Sum(threadlocal_aggregate_gt_pivot_1);
    __syncthreads();

    min_gt_low = BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                     .Reduce(min_gt_low, MinReduceOp{});
    __syncthreads();
    max_le_high =
        BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
            .Reduce(max_le_high, MaxReduceOp{});
    if (tx == 0) {
      temp_storage.block_aggregate.values[0] = aggregate_gt_pivot_0;
      temp_storage.block_aggregate.values[1] = aggregate_gt_pivot_1;
      temp_storage.min_val = min_gt_low;
      temp_storage.max_val = max_le_high;
    }
    __syncthreads();
    aggregate_gt_pivot_0 = temp_storage.block_aggregate.values[0];
    aggregate_gt_pivot_1 = temp_storage.block_aggregate.values[1];
    min_gt_low = temp_storage.min_val;
    max_le_high = temp_storage.max_val;

    if (aggregate_gt_pivot_1 >= p) {
      low = pivot_1;
      sum_low = aggregate_gt_pivot_1;
    } else if (aggregate_gt_pivot_0 >= p) {
      low = pivot_0;
      high = min(pivot_1, max_le_high);
      sum_low = aggregate_gt_pivot_0;
    } else {
      high = min(pivot_0, max_le_high);
    }
  } while (min_gt_low != max_le_high);

  float normalizer = math::ptx_rcp(max(sum_low, 1e-8));

  // normalize
#pragma unroll 2
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    probs_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      probs_vec.cast_load(probs + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      probs_vec[j] = (probs_vec[j] > low) ? probs_vec[j] * normalizer : 0;
    }
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      probs_vec.cast_store(renormed_prob + row_idx * d + i * BLOCK_THREADS * VEC_SIZE +
                           tx * VEC_SIZE);
    }
  }
}

template <typename DType>
cudaError_t TopPRenormProb(DType* probs, DType* renormed_prob, float* top_p_arr,
                           uint32_t batch_size, float top_p_val, uint32_t d,
                           cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(DType), d);

  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
    const uint32_t smem_size = sizeof(RenormTempStorage<BLOCK_THREADS, REDUCE_ALGO>);
    dim3 nblks(batch_size);
    dim3 nthrs(BLOCK_THREADS);
    void* args[] = {&probs, &renormed_prob, &top_p_arr, &top_p_val, &d};
    DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
      auto kernel = TopPRenormProbKernel<BLOCK_THREADS, REDUCE_ALGO, VEC_SIZE, DType>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    });
    return cudaSuccess;
  });
}

// ==================== Multi-CTA Top-K Implementation ====================

// Atomic min/max for float using CAS
__device__ __forceinline__ float atomicMinFloat(float* addr, float value) {
  int* addr_as_int = (int*)addr;
  int old = *addr_as_int, assumed;

  do {
    assumed = old;
    old = atomicCAS(addr_as_int, assumed, __float_as_int(fminf(value, __int_as_float(assumed))));
  } while (assumed != old);

  return __int_as_float(old);
}

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
  int* addr_as_int = (int*)addr;
  int old = *addr_as_int, assumed;

  do {
    assumed = old;
    old = atomicCAS(addr_as_int, assumed, __float_as_int(fmaxf(value, __int_as_float(assumed))));
  } while (assumed != old);

  return __int_as_float(old);
}

// Acquire/Release primitives for inter-CTA synchronization
__device__ __forceinline__ int ld_acquire(int* ptr) {
  int state = 0;

#if (__CUDA_ARCH__ >= 700)
  // SM70 and newer use memory consistency qualifiers
  // Acquire pattern using acquire modifier
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
#else
  asm volatile("ld.cg.global.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
#endif

  return state;
}

__device__ __forceinline__ void red_release(int* ptr, int val) {
#if (__CUDA_ARCH__ >= 700)
  // SM70 and newer use memory consistency qualifiers
  // Release pattern using acq_rel fence + relaxed modifier
  // (The fence also releases data that was weakly-written by other threads prior to the last
  // syncthreads)
  asm volatile("fence.acq_rel.gpu;\n");
  asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(ptr), "r"(val));
#else
  __threadfence();
  atomicAdd(ptr, val);
#endif
}

__device__ __forceinline__ void st_release(int* ptr, int val) {
#if (__CUDA_ARCH__ >= 700)
  // SM70 and newer use memory consistency qualifiers
  // Release pattern: fence + release store
  asm volatile("fence.acq_rel.gpu;\n");
  asm volatile("st.release.gpu.global.b32 [%0], %1;\n" : : "l"(ptr), "r"(val));
#else
  __threadfence();
  atomicExch(ptr, val);
#endif
}

// Wait until the value at ptr reaches target_val using acquire semantics
// Only thread 0 spins, then all threads synchronize
__device__ __forceinline__ void wait_ge(int* ptr, int target_val, int thread_idx) {
  if (thread_idx == 0) {
#pragma unroll 1
    while (ld_acquire(ptr) < target_val) {
    }
  }
  __syncthreads();
}

// Global state for multi-CTA reduction (one per row)
template <typename T>
struct RowReductionState {
  // Ping-pong buffers for atomic reduction
  int count_0_buf[2];
  int count_1_buf[2];
  T min_buf[2];
  T max_buf[2];

  // Arrival counter for acquire/release synchronization
  int arrival_counter;
};

template <uint32_t BLOCK_THREADS, BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE,
          typename DType, typename IdType>
__global__ void __launch_bounds__(BLOCK_THREADS) TopKMaskLogitsKernel_MultiCTA(
    DType* logits,         // [batch, vocab_size]
    DType* masked_logits,  // [batch, vocab_size]
    IdType* top_k_arr,     // [batch] or nullptr
    uint32_t top_k_val, uint32_t vocab_size, uint32_t batch_size,
    RowReductionState<float>* row_states,  // [num_groups], always float for atomic ops
    uint32_t chunk_size,                   // elements per CTA (must be multiple of VEC_SIZE)
    uint32_t ctas_per_group)               // CTAs per row
{
  const uint32_t global_cta_id = blockIdx.x;
  const uint32_t group_id = global_cta_id / ctas_per_group;
  const uint32_t cta_in_group = global_cta_id % ctas_per_group;
  const uint32_t tx = threadIdx.x;

  // Shared memory layout: [temp_storage] [padding] [logits data (16-byte aligned)]
  extern __shared__ uint8_t smem[];
  auto* temp_storage = reinterpret_cast<RenormTempStorage<BLOCK_THREADS, REDUCE_ALGORITHM>*>(smem);

  // Align logits to 16 bytes
  size_t temp_storage_size = sizeof(RenormTempStorage<BLOCK_THREADS, REDUCE_ALGORITHM>);
  size_t logits_offset = ((temp_storage_size + 15) / 16) * 16;
  DType* shared_logits = reinterpret_cast<DType*>(smem + logits_offset);

  // Note: arrival_counter and count buffers should be pre-initialized to zero on the host side

  // Persistent iteration counter for double buffering (never resets across rows)
  int persistent_iteration = 0;

  // Calculate total number of iterations for persistent loop
  uint32_t num_groups = gridDim.x / ctas_per_group;
  uint32_t total_iterations = (batch_size + num_groups - 1) / num_groups;

  int barrier_phase = 0;
  // Each group uses its own state (groups process rows sequentially in persistent loop)
  // Note: state uses float internally for precision and atomic operations
  RowReductionState<float>* state = &row_states[group_id];

  // Initialize min/max buffer for this row (first CTA only)
  if (cta_in_group == 0 && tx == 0) {
    state->min_buf[0] = cuda::std::numeric_limits<float>::max();
    state->max_buf[0] = cuda::std::numeric_limits<float>::lowest();
  }

  // First barrier: ensure all CTAs see the initialized min/max values
  if (tx == 0) {
    red_release(&state->arrival_counter, 1);
  }
  int target = (barrier_phase + 1) * ctas_per_group;
  wait_ge(&state->arrival_counter, target, tx);
  barrier_phase++;

  // Persistent loop over rows
  for (uint32_t iter = 0; iter < total_iterations; iter++) {
    uint32_t row_idx = group_id + iter * num_groups;

    if (row_idx >= batch_size) break;  // Early exit if out of bounds

    const uint32_t chunk_start = cta_in_group * chunk_size;
    const uint32_t chunk_end = min(chunk_start + chunk_size, vocab_size);
    const uint32_t actual_chunk_size = chunk_end - chunk_start;

    uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[row_idx];

    // ========== Stage 1: Load to shared memory ==========
    vec_t<DType, VEC_SIZE> logits_vec;
    const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;

    // Vectorized load for aligned portion
#pragma unroll 2
    for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
      logits_vec.cast_load(logits + row_idx * vocab_size + chunk_start + i);
      logits_vec.store(shared_logits + i);
    }

    // Scalar load for tail (only for last CTA if vocab_size not aligned)
    for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
      shared_logits[i] = logits[row_idx * vocab_size + chunk_start + i];
    }
    __syncthreads();

    double pivot = -cuda::std::numeric_limits<float>::infinity();

    if (k < vocab_size) {
      // ========== Stage 2: Initialize - find global min/max ==========
      float local_min = cuda::std::numeric_limits<float>::max();
      float local_max = cuda::std::numeric_limits<float>::lowest();

      // Vectorized min/max for aligned portion
#pragma unroll 2
      for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
        logits_vec.load(shared_logits + i);
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          float val = logits_vec[j];
          local_min = min(local_min, val);
          local_max = max(local_max, val);
        }
      }

      // Scalar min/max for tail
      for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
        float val = shared_logits[i];
        local_min = min(local_min, val);
        local_max = max(local_max, val);
      }

      // Block reduction
      float block_min =
          BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce)
              .Reduce(local_min, MinReduceOp{});
      __syncthreads();

      float block_max =
          BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce)
              .Reduce(local_max, MaxReduceOp{});
      __syncthreads();

      // Atomic reduction to global state
      if (tx == 0) {
        atomicMinFloat(&state->min_buf[0], block_min);
        atomicMaxFloat(&state->max_buf[0], block_max);

        // Signal arrival using release semantics
        red_release(&state->arrival_counter, 1);
      }
      int target = (barrier_phase + 1) * ctas_per_group;
      wait_ge(&state->arrival_counter, target, tx);
      barrier_phase++;

      float global_min = state->min_buf[0];
      float global_max = state->max_buf[0];

      // ========== Stage 3: Binary search ==========
      double low = (global_min == -cuda::std::numeric_limits<float>::infinity())
                       ? cuda::std::numeric_limits<float>::lowest()
                       : global_min - 1;
      double high = global_max;
      float min_gt_low, max_le_high;

      do {
        double pivot_0 = (high + 2 * low) / 3;
        double pivot_1 = (2 * high + low) / 3;

        // Local counting from shared memory
        int local_count_0 = 0, local_count_1 = 0;
        float local_min_gt_low = high, local_max_le_high = low;

        // Vectorized counting for aligned portion
#pragma unroll 2
        for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
          logits_vec.load(shared_logits + i);
#pragma unroll
          for (uint32_t j = 0; j < VEC_SIZE; ++j) {
            float val = logits_vec[j];
            // Branchless counting
            local_count_0 += (val > pivot_0);
            local_count_1 += (val > pivot_1);
            // Update min/max
            if (val > low) local_min_gt_low = min(local_min_gt_low, val);
            if (val <= high) local_max_le_high = max(local_max_le_high, val);
          }
        }

        // Scalar counting for tail
        for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
          float val = shared_logits[i];
          local_count_0 += (val > pivot_0);
          local_count_1 += (val > pivot_1);
          if (val > low) local_min_gt_low = min(local_min_gt_low, val);
          if (val <= high) local_max_le_high = max(local_max_le_high, val);
        }

        // Block reduction
        int block_count_0 =
            BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce_int)
                .Sum(local_count_0);
        __syncthreads();

        int block_count_1 =
            BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce_int)
                .Sum(local_count_1);
        __syncthreads();

        float block_min_gt_low =
            BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce)
                .Reduce(local_min_gt_low, MinReduceOp{});
        __syncthreads();

        float block_max_le_high =
            BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce)
                .Reduce(local_max_le_high, MaxReduceOp{});
        __syncthreads();

        // Ping-pong buffer index (use persistent_iteration for double buffering)
        int buffer_idx = persistent_iteration & 1;

        // Atomic reduction to global state
        if (tx == 0) {
          atomicAdd(&state->count_0_buf[buffer_idx], block_count_0);
          atomicAdd(&state->count_1_buf[buffer_idx], block_count_1);
          atomicMinFloat(&state->min_buf[buffer_idx], block_min_gt_low);
          atomicMaxFloat(&state->max_buf[buffer_idx], block_max_le_high);

          // Signal arrival using release semantics
          red_release(&state->arrival_counter, 1);

          // Last CTA clears next buffer (no need to reset counter anymore)
          if (cta_in_group == ctas_per_group - 1) {
            int next_buf = (persistent_iteration + 1) & 1;
            state->count_0_buf[next_buf] = 0;
            state->count_1_buf[next_buf] = 0;
            state->min_buf[next_buf] = cuda::std::numeric_limits<float>::max();
            state->max_buf[next_buf] = cuda::std::numeric_limits<float>::lowest();
          }
        }
        int target = (barrier_phase + 1) * ctas_per_group;
        wait_ge(&state->arrival_counter, target, tx);
        barrier_phase++;

        // Read results from current buffer
        int aggregate_gt_pivot_0 = state->count_0_buf[buffer_idx];
        int aggregate_gt_pivot_1 = state->count_1_buf[buffer_idx];
        min_gt_low = state->min_buf[buffer_idx];
        max_le_high = state->max_buf[buffer_idx];

        // Update search range
        if (aggregate_gt_pivot_1 >= k) {
          low = pivot_1;
        } else if (aggregate_gt_pivot_0 >= k) {
          low = pivot_0;
          high = min(pivot_1, max_le_high);
        } else {
          high = min(pivot_0, max_le_high);
        }

        persistent_iteration++;

      } while (min_gt_low != max_le_high);

      pivot = low;
    }

    // ========== Stage 4: Masking ==========
    // Vectorized masking for aligned portion
#pragma unroll 2
    for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
      logits_vec.load(shared_logits + i);
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        logits_vec[j] = (logits_vec[j] >= pivot) ? logits_vec[j]
                                                 : -cuda::std::numeric_limits<float>::infinity();
      }
      logits_vec.store(masked_logits + row_idx * vocab_size + chunk_start + i);
    }

    // Scalar masking for tail
    for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
      float val = shared_logits[i];
      masked_logits[row_idx * vocab_size + chunk_start + i] =
          (val >= pivot) ? val : -cuda::std::numeric_limits<float>::infinity();
    }
  }

  // Finalize: reset counter for this group to prepare for next kernel launch
  // All iterations are done, safe to reset now
  if (cta_in_group == 0 && tx == 0) {
    st_release(&row_states[group_id].arrival_counter, 0);
  }
}

template <typename DType, typename IdType>
cudaError_t TopKMaskLogitsMultiCTA(DType* logits, DType* masked_logits, IdType* top_k_arr,
                                   uint32_t batch_size, uint32_t top_k_val, uint32_t vocab_size,
                                   RowReductionState<float>* row_states_buffer,
                                   cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(DType), vocab_size);

  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
    DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
      // Calculate aligned temp storage size
      constexpr size_t temp_storage_size = sizeof(RenormTempStorage<BLOCK_THREADS, REDUCE_ALGO>);
      constexpr size_t temp_storage_aligned = round_up(temp_storage_size, 16UL);

      // Get device properties
      int device;
      FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
      int max_smem_per_block;
      FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&max_smem_per_block,
                                                  cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

      // Calculate max chunk size that fits in shared memory
      // smem layout: [temp_storage_aligned] [chunk_size * sizeof(DType)]
      const size_t available_for_logits = max_smem_per_block - temp_storage_aligned;
      uint32_t max_chunk_elements = available_for_logits / sizeof(DType);

      // Round down to multiple of VEC_SIZE
      max_chunk_elements = round_down(max_chunk_elements, VEC_SIZE);

      // Ensure minimum chunk size for vectorized access
      constexpr uint32_t min_chunk_size = VEC_SIZE * BLOCK_THREADS;
      max_chunk_elements = std::max(max_chunk_elements, min_chunk_size);

      // Calculate how many CTAs needed per row
      uint32_t ctas_per_group = ceil_div(vocab_size, max_chunk_elements);
      uint32_t chunk_size = ceil_div(vocab_size, ctas_per_group);
      // Round up chunk_size to multiple of VEC_SIZE
      chunk_size = round_up(chunk_size, VEC_SIZE);
      // Ensure minimum chunk size
      chunk_size = std::max(chunk_size, min_chunk_size);

      // Get number of SMs
      int num_sms;
      FLASHINFER_CUDA_CALL(
          cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));

      // Calculate grid size (must be multiple of ctas_per_group, up to num_sms)
      uint32_t num_groups = std::min(static_cast<uint32_t>(num_sms) / ctas_per_group, batch_size);
      if (num_groups == 0) {
        // vocab_size too large to fit in shared memory even with one chunk per SM
        return cudaErrorInvalidConfiguration;
      }
      uint32_t total_ctas = num_groups * ctas_per_group;

      // Calculate shared memory size
      const uint32_t smem_size = temp_storage_aligned + chunk_size * sizeof(DType);

      // Launch kernel
      dim3 nblks(total_ctas);
      dim3 nthrs(BLOCK_THREADS);
      void* args[] = {&logits,     &masked_logits,     &top_k_arr,  &top_k_val,     &vocab_size,
                      &batch_size, &row_states_buffer, &chunk_size, &ctas_per_group};

      auto kernel =
          TopKMaskLogitsKernel_MultiCTA<BLOCK_THREADS, REDUCE_ALGO, VEC_SIZE, DType, IdType>;

      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

      // Use regular kernel launch via cudaLaunchKernel API
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));

      return cudaSuccess;
    });
  });
}

// ==================== Multi-CTA Radix Top-K Mask Logits ====================

// Global state for multi-CTA radix reduction (one per group)
struct RadixRowState {
  uint32_t histogram[2][256];  // Double-buffered histograms for ping-pong
  uint32_t remaining_k;        // Remaining k after current round
  uint32_t prefix;             // Accumulated prefix (high bits of k-th element)
  int arrival_counter;         // For inter-CTA synchronization
  int output_counter;          // For collecting top-k indices (RadixTopK)
  float sum_topk;              // For RenormProb: sum of top-k elements
};

// ==================== Common Device Functions for Radix Top-K ====================

/*!
 * \brief Compute suffix sum in shared memory using parallel reduction.
 *
 * After this function, suffix_sum[i] contains the count of elements >= bucket i.
 * This is computed by summing all histogram values from bucket i to 255.
 *
 * \param suffix_sum Shared memory array of size RADIX (256)
 * \param tx Thread index within the block
 */
template <uint32_t BLOCK_THREADS>
__device__ __forceinline__ void RadixSuffixSum(uint32_t* suffix_sum, uint32_t tx) {
  constexpr uint32_t RADIX = 256;
  // Parallel suffix sum: compute count of elements >= each bucket
  for (uint32_t stride = 1; stride < RADIX; stride *= 2) {
    uint32_t val = 0;
    if (tx < RADIX) {
      val = suffix_sum[tx];
      if (tx + stride < RADIX) {
        val += suffix_sum[tx + stride];
      }
    }
    __syncthreads();
    if (tx < RADIX) {
      suffix_sum[tx] = val;
    }
    __syncthreads();
  }
}

/*!
 * \brief Find the threshold bucket that contains the k-th largest element.
 *
 * The threshold bucket satisfies: count_ge >= k && count_gt < k
 * where count_ge = suffix_sum[bucket] and count_gt = suffix_sum[bucket+1].
 *
 * \param suffix_sum Shared memory array containing suffix sums
 * \param remaining_k Number of top-k elements still to find
 * \param found_bucket Output: the found threshold bucket
 * \param found_remaining_k Output: remaining_k minus count of elements > threshold
 * \param tx Thread index within the block
 */
__device__ __forceinline__ void RadixFindThresholdBucket(uint32_t* suffix_sum, uint32_t remaining_k,
                                                         uint32_t* found_bucket,
                                                         uint32_t* found_remaining_k, uint32_t tx) {
  constexpr uint32_t RADIX = 256;
  // Initialize (only thread 0)
  if (tx == 0) {
    *found_bucket = 0;
    *found_remaining_k = remaining_k;
  }
  __syncthreads();

  // All threads in RADIX range check their bucket
  if (tx < RADIX) {
    uint32_t count_ge = suffix_sum[tx];
    uint32_t count_gt = (tx + 1 < RADIX) ? suffix_sum[tx + 1] : 0;
    if (count_ge >= remaining_k && count_gt < remaining_k) {
      *found_bucket = tx;
      *found_remaining_k = remaining_k - count_gt;
    }
  }
  __syncthreads();
}

/*!
 * \brief Build local histogram for one round of radix select.
 *
 * Counts elements in shared_ordered that match the current prefix and bins them
 * by their byte at the current shift position.
 *
 * \tparam OrderedType The ordered integer type (uint16_t or uint32_t)
 * \param shared_ordered Shared memory containing ordered values
 * \param actual_chunk_size Number of elements in this CTA's chunk
 * \param local_histogram Output shared memory histogram
 * \param prefix Current prefix (high bits determined so far)
 * \param shift Bit shift for extracting current byte
 * \param round Current round (0 to NUM_ROUNDS-1)
 * \param tx Thread index
 */
template <uint32_t BLOCK_THREADS, typename OrderedType>
__device__ __forceinline__ void RadixBuildLocalHistogram(const OrderedType* shared_ordered,
                                                         uint32_t actual_chunk_size,
                                                         uint32_t* local_histogram, uint32_t prefix,
                                                         uint32_t shift, uint32_t round,
                                                         uint32_t tx) {
  constexpr uint32_t ORDERED_BITS = sizeof(OrderedType) * 8;
  constexpr uint32_t RADIX_BITS = 8;

  for (uint32_t i = tx; i < actual_chunk_size; i += BLOCK_THREADS) {
    OrderedType ordered = shared_ordered[i];

    // Check if this element matches the prefix (high bits determined so far)
    OrderedType mask =
        (round == 0)
            ? OrderedType(0)
            : static_cast<OrderedType>(~OrderedType(0) << (ORDERED_BITS - round * RADIX_BITS));
    if ((ordered & mask) == static_cast<OrderedType>(prefix)) {
      uint32_t bucket = (ordered >> shift) & 0xFF;
      atomicAdd(&local_histogram[bucket], 1);
    }
  }
}

/*!
 * \brief Perform one round of radix select with optional multi-CTA synchronization.
 *
 * This is the core radix select logic used by all TopK kernels.
 * It builds histogram, aggregates across CTAs (if multi-CTA), computes suffix sum,
 * and finds the threshold bucket.
 *
 * \tparam BLOCK_THREADS Number of threads per block
 * \tparam SINGLE_CTA True if single-CTA mode (no inter-CTA sync needed)
 * \tparam OrderedType The ordered integer type
 *
 * \param shared_ordered Shared memory containing ordered values
 * \param actual_chunk_size Number of elements in this CTA's chunk
 * \param local_histogram Shared memory for local histogram (size RADIX)
 * \param suffix_sum Shared memory for suffix sum computation (size RADIX)
 * \param state Pointer to RadixRowState for multi-CTA sync (nullptr if SINGLE_CTA)
 * \param prefix Current prefix value
 * \param remaining_k Current remaining k value
 * \param round Current round (0 to NUM_ROUNDS-1)
 * \param barrier_phase Reference to barrier phase counter
 * \param ctas_per_group Number of CTAs per group
 * \param tx Thread index
 * \param out_new_prefix Output: updated prefix after this round
 * \param out_new_remaining_k Output: updated remaining_k after this round
 */
template <uint32_t BLOCK_THREADS, bool SINGLE_CTA, typename OrderedType>
__device__ __forceinline__ void RadixSelectOneRound(
    const OrderedType* shared_ordered, uint32_t actual_chunk_size, uint32_t* local_histogram,
    uint32_t* suffix_sum, uint32_t* shared_scalars, RadixRowState* state, uint32_t prefix,
    uint32_t remaining_k, uint32_t round, int& barrier_phase, uint32_t ctas_per_group, uint32_t tx,
    uint32_t* out_new_prefix, uint32_t* out_new_remaining_k) {
  constexpr uint32_t RADIX = 256;
  constexpr uint32_t ORDERED_BITS = sizeof(OrderedType) * 8;
  constexpr uint32_t RADIX_BITS = 8;
  uint32_t shift = ORDERED_BITS - (round + 1) * RADIX_BITS;

  // For multi-CTA: pointers to global histograms
  uint32_t* current_hist = nullptr;
  uint32_t* other_hist = nullptr;
  if constexpr (!SINGLE_CTA) {
    current_hist = state->histogram[round % 2];
    other_hist = state->histogram[(round + 1) % 2];
  }

  // Clear local histogram AND (for multi-CTA) clear the "other" global histogram
  for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
    local_histogram[i] = 0;
    if constexpr (!SINGLE_CTA) {
      other_hist[i] = 0;  // Prepare for next round
    }
  }
  __syncthreads();

  // Build local histogram from shared memory
  RadixBuildLocalHistogram<BLOCK_THREADS, OrderedType>(shared_ordered, actual_chunk_size,
                                                       local_histogram, prefix, shift, round, tx);
  __syncthreads();

  // For multi-CTA: add to global histogram and barrier
  // For single-CTA: local_histogram is already the complete histogram
  if constexpr (!SINGLE_CTA) {
    for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
      if (local_histogram[i] > 0) {
        atomicAdd(&current_hist[i], local_histogram[i]);
      }
    }

    // Barrier: wait for all CTAs to finish histogram accumulation
    if (tx == 0) {
      red_release(&state->arrival_counter, 1);
    }
    int target = (barrier_phase + 1) * ctas_per_group;
    wait_ge(&state->arrival_counter, target, tx);
    barrier_phase++;
    __syncthreads();

    // Load from global histogram to suffix_sum
    for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
      suffix_sum[i] = current_hist[i];
    }
  } else {
    // Single-CTA: copy local histogram directly to suffix_sum
    for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
      suffix_sum[i] = local_histogram[i];
    }
  }
  __syncthreads();

  // Compute suffix sum
  RadixSuffixSum<BLOCK_THREADS>(suffix_sum, tx);

  // Find threshold bucket using shared_scalars for found_bucket and found_remaining_k
  // shared_scalars[0] = found_bucket, shared_scalars[1] = found_remaining_k
  RadixFindThresholdBucket(suffix_sum, remaining_k, &shared_scalars[0], &shared_scalars[1], tx);

  // Output new prefix and remaining_k
  *out_new_prefix = prefix | (shared_scalars[0] << shift);
  *out_new_remaining_k = shared_scalars[1];
}

/*!
 * \brief Find the k-th largest element pivot using radix select.
 *
 * This is the main entry point for the radix select algorithm.
 * It performs NUM_ROUNDS of radix select to find the exact pivot value.
 *
 * \tparam BLOCK_THREADS Number of threads per block
 * \tparam VEC_SIZE Vector size for memory access
 * \tparam SINGLE_CTA True if single-CTA mode
 * \tparam DType Data type (float, half, nv_bfloat16)
 *
 * \param input Input data pointer (for this row)
 * \param shared_ordered Shared memory for ordered values
 * \param local_histogram Shared memory for local histogram
 * \param suffix_sum Shared memory for suffix sum
 * \param shared_scalars Shared memory for temporary scalar values (size >= 2)
 * \param state RadixRowState pointer (nullptr if SINGLE_CTA)
 * \param chunk_start Start index in vocab for this CTA
 * \param actual_chunk_size Number of elements in this chunk
 * \param k Number of top elements to select
 * \param barrier_phase Reference to barrier phase counter
 * \param ctas_per_group Number of CTAs per group
 * \param tx Thread index
 * \return The pivot value (k-th largest element)
 */
template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, bool SINGLE_CTA, typename DType>
__device__ __forceinline__ DType RadixSelectFindPivot(
    const DType* input, typename RadixTopKTraits<DType>::OrderedType* shared_ordered,
    uint32_t* local_histogram, uint32_t* suffix_sum, uint32_t* shared_scalars, RadixRowState* state,
    uint32_t chunk_start, uint32_t actual_chunk_size, uint32_t k, int& barrier_phase,
    uint32_t ctas_per_group, uint32_t tx) {
  using Traits = RadixTopKTraits<DType>;
  using OrderedType = typename Traits::OrderedType;
  constexpr uint32_t RADIX = 256;
  constexpr uint32_t RADIX_BITS = 8;
  constexpr uint32_t NUM_ROUNDS = Traits::template num_rounds<RADIX_BITS>();
  constexpr uint32_t ORDERED_BITS = sizeof(OrderedType) * 8;

  // Stage 1: Load and convert to ordered representation
  const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;
  vec_t<DType, VEC_SIZE> data_vec;

#pragma unroll 2
  for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
    data_vec.cast_load(input + chunk_start + i);
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      shared_ordered[i + j] = Traits::ToOrdered(data_vec[j]);
    }
  }
  // Handle tail
  for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
    shared_ordered[i] = Traits::ToOrdered(input[chunk_start + i]);
  }
  __syncthreads();

  // Initialize prefix and remaining_k
  uint32_t prefix = 0;
  uint32_t remaining_k = k;

  // Clear global histograms (only needed for multi-CTA)
  if constexpr (!SINGLE_CTA) {
    for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
      state->histogram[0][i] = 0;
      state->histogram[1][i] = 0;
    }
  }
  __syncthreads();

  // Initial barrier (skip for single CTA)
  if constexpr (!SINGLE_CTA) {
    if (tx == 0) {
      red_release(&state->arrival_counter, 1);
    }
    int target = (barrier_phase + 1) * ctas_per_group;
    wait_ge(&state->arrival_counter, target, tx);
    barrier_phase++;
    __syncthreads();
  }

  // Stage 2: NUM_ROUNDS of radix select
  for (uint32_t round = 0; round < NUM_ROUNDS; ++round) {
    uint32_t new_prefix, new_remaining_k;
    RadixSelectOneRound<BLOCK_THREADS, SINGLE_CTA, OrderedType>(
        shared_ordered, actual_chunk_size, local_histogram, suffix_sum, shared_scalars, state,
        prefix, remaining_k, round, barrier_phase, ctas_per_group, tx, &new_prefix,
        &new_remaining_k);
    prefix = new_prefix;
    remaining_k = new_remaining_k;
    __syncthreads();
  }

  // Convert final ordered representation back to DType pivot
  return Traits::FromOrdered(static_cast<OrderedType>(prefix));
}

template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, bool SINGLE_CTA, typename DType,
          typename IdType>
__global__ void __launch_bounds__(BLOCK_THREADS) RadixTopKMaskLogitsKernel_MultiCTA(
    DType* logits,         // [batch, vocab_size]
    DType* masked_logits,  // [batch, vocab_size]
    IdType* top_k_arr,     // [batch] or nullptr
    uint32_t top_k_val, uint32_t vocab_size, uint32_t batch_size,
    RadixRowState* row_states,  // [num_groups] (nullptr if SINGLE_CTA)
    uint32_t chunk_size,        // elements per CTA
    uint32_t ctas_per_group)    // CTAs per row (1 if SINGLE_CTA)
{
  // Type traits for FP16/BF16/FP32 support
  using Traits = RadixTopKTraits<DType>;
  using OrderedType = typename Traits::OrderedType;

  constexpr uint32_t RADIX = 256;  // 8-bit radix
  constexpr uint32_t RADIX_BITS = 8;
  constexpr uint32_t NUM_ROUNDS = Traits::template num_rounds<RADIX_BITS>();
  constexpr uint32_t ORDERED_BITS = sizeof(OrderedType) * 8;

  const uint32_t global_cta_id = blockIdx.x;
  const uint32_t group_id = global_cta_id / ctas_per_group;
  const uint32_t cta_in_group = global_cta_id % ctas_per_group;
  const uint32_t tx = threadIdx.x;

  // Shared memory layout: [fixed storage] [ordered values cache]
  extern __shared__ uint8_t smem[];

  // Fixed shared memory (at the beginning)
  constexpr size_t fixed_smem_size =
      sizeof(uint32_t) * (RADIX + RADIX + 4);  // histogram + suffix + 4 scalars
  uint32_t* local_histogram = reinterpret_cast<uint32_t*>(smem);
  uint32_t* suffix_sum = local_histogram + RADIX;
  uint32_t* shared_scalars =
      suffix_sum + RADIX;  // [prefix_cache, remaining_k_cache, found_bucket, found_remaining_k]

  // Align ordered values cache to 16 bytes
  size_t ordered_offset = ((fixed_smem_size + 15) / 16) * 16;
  OrderedType* shared_ordered = reinterpret_cast<OrderedType*>(smem + ordered_offset);

// Aliases for scalar shared variables
#define prefix_cache shared_scalars[0]
#define remaining_k_cache shared_scalars[1]
#define found_bucket shared_scalars[2]
#define found_remaining_k shared_scalars[3]

  // State pointer only used when not SINGLE_CTA
  RadixRowState* state = nullptr;
  if constexpr (!SINGLE_CTA) {
    state = &row_states[group_id];
  }

  // Calculate total number of iterations for persistent loop
  uint32_t num_groups = gridDim.x / ctas_per_group;
  uint32_t total_iterations = (batch_size + num_groups - 1) / num_groups;

  int barrier_phase = 0;

  // Persistent loop over rows
  for (uint32_t iter = 0; iter < total_iterations; iter++) {
    uint32_t row_idx = group_id + iter * num_groups;

    if (row_idx >= batch_size) break;

    const uint32_t chunk_start = cta_in_group * chunk_size;
    const uint32_t chunk_end = min(chunk_start + chunk_size, vocab_size);

    uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[row_idx];

    DType pivot = Traits::NegInf();

    const uint32_t actual_chunk_size = chunk_end - chunk_start;

    if (k >= vocab_size) {
      // k >= vocab_size: no masking needed, just copy
      vec_t<DType, VEC_SIZE> logits_vec_copy;
#pragma unroll 2
      for (uint32_t i = tx * VEC_SIZE; i < actual_chunk_size; i += BLOCK_THREADS * VEC_SIZE) {
        if (i + VEC_SIZE <= actual_chunk_size) {
          logits_vec_copy.cast_load(logits + row_idx * vocab_size + chunk_start + i);
          logits_vec_copy.store(masked_logits + row_idx * vocab_size + chunk_start + i);
        }
      }
      // Handle tail
      for (uint32_t i = (actual_chunk_size / VEC_SIZE) * VEC_SIZE + tx; i < actual_chunk_size;
           i += BLOCK_THREADS) {
        masked_logits[row_idx * vocab_size + chunk_start + i] =
            logits[row_idx * vocab_size + chunk_start + i];
      }
      continue;
    }

    // ========== Stage 1: Load and convert to ordered representation in shared memory ==========
    // This is done ONCE per row, avoiding NUM_ROUNDS global memory reads
    vec_t<DType, VEC_SIZE> logits_vec;
    const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;

#pragma unroll 2
    for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
      logits_vec.cast_load(logits + row_idx * vocab_size + chunk_start + i);
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        // Use type traits for FP16/BF16/FP32 support
        shared_ordered[i + j] = Traits::ToOrdered(logits_vec[j]);
      }
    }
    // Handle tail
    for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
      shared_ordered[i] = Traits::ToOrdered(logits[row_idx * vocab_size + chunk_start + i]);
    }
    __syncthreads();

    // Initialize local caches
    if (tx == 0) {
      prefix_cache = 0;
      remaining_k_cache = k;
    }
    // Clear global histograms (only needed for multi-CTA)
    if constexpr (!SINGLE_CTA) {
      for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
        state->histogram[0][i] = 0;
        state->histogram[1][i] = 0;
      }
    }
    __syncthreads();

    // Barrier to ensure all CTAs have arrived at this iteration (skip for single CTA)
    if constexpr (!SINGLE_CTA) {
      if (tx == 0) {
        red_release(&state->arrival_counter, 1);
      }
      int target = (barrier_phase + 1) * ctas_per_group;
      wait_ge(&state->arrival_counter, target, tx);
      barrier_phase++;
      __syncthreads();
    }

    // ========== Stage 2: NUM_ROUNDS of radix select ==========
    // Using double-buffering: round N uses histogram[N % 2]
    // Round N clears histogram[(N+1) % 2] for next round's use
    for (uint32_t round = 0; round < NUM_ROUNDS; ++round) {
      uint32_t shift = ORDERED_BITS - (round + 1) * RADIX_BITS;
      // Read from local cache (no global memory access needed!)
      uint32_t prefix = prefix_cache;
      uint32_t remaining_k = remaining_k_cache;

      // For multi-CTA: pointers to global histograms
      // For single-CTA: these are not used
      uint32_t* current_hist = nullptr;
      uint32_t* other_hist = nullptr;
      if constexpr (!SINGLE_CTA) {
        current_hist = state->histogram[round % 2];
        other_hist = state->histogram[(round + 1) % 2];
      }

      // Clear local histogram AND (for multi-CTA) clear the "other" global histogram
      for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
        local_histogram[i] = 0;
        if constexpr (!SINGLE_CTA) {
          other_hist[i] = 0;  // Prepare for next round (no barrier needed!)
        }
      }
      __syncthreads();

      // Build local histogram from SHARED MEMORY (no global memory access!)
      for (uint32_t i = tx; i < actual_chunk_size; i += BLOCK_THREADS) {
        OrderedType ordered = shared_ordered[i];

        // Check if this element matches the prefix (high bits determined so far)
        // Use generic mask based on OrderedType bits
        OrderedType mask =
            (round == 0)
                ? OrderedType(0)
                : static_cast<OrderedType>(~OrderedType(0) << (ORDERED_BITS - round * RADIX_BITS));
        if ((ordered & mask) == static_cast<OrderedType>(prefix)) {
          uint32_t bucket = (ordered >> shift) & 0xFF;
          atomicAdd(&local_histogram[bucket], 1);
        }
      }
      __syncthreads();

      // For multi-CTA: add to global histogram and barrier
      // For single-CTA: local_histogram is already the complete histogram
      if constexpr (!SINGLE_CTA) {
        for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
          if (local_histogram[i] > 0) {
            atomicAdd(&current_hist[i], local_histogram[i]);
          }
        }

        // Barrier: wait for all CTAs to finish histogram accumulation
        if (tx == 0) {
          red_release(&state->arrival_counter, 1);
        }
        int target = (barrier_phase + 1) * ctas_per_group;
        wait_ge(&state->arrival_counter, target, tx);
        barrier_phase++;
        __syncthreads();

        // Load from global histogram to suffix_sum
        for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
          suffix_sum[i] = current_hist[i];
        }
      } else {
        // Single-CTA: copy local histogram directly to suffix_sum
        for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
          suffix_sum[i] = local_histogram[i];
        }
      }
      __syncthreads();

      // Parallel suffix sum in shared memory (much faster than global memory!)
      // Compute count of elements >= each bucket value
      for (uint32_t stride = 1; stride < RADIX; stride *= 2) {
        uint32_t val = 0;
        if (tx < RADIX) {
          val = suffix_sum[tx];
          if (tx + stride < RADIX) {
            val += suffix_sum[tx + stride];
          }
        }
        __syncthreads();
        if (tx < RADIX) {
          suffix_sum[tx] = val;
        }
        __syncthreads();
      }

      // ALL CTAs: find threshold bucket (all compute same result)
      // Use shared variable to communicate the found bucket (via macros to shared_scalars[2..3])
      if (tx == 0) {
        found_bucket = 0;
        found_remaining_k = remaining_k;
      }
      __syncthreads();

      if (tx < RADIX) {
        uint32_t count_ge = suffix_sum[tx];
        uint32_t count_gt = (tx + 1 < RADIX) ? suffix_sum[tx + 1] : 0;
        if (count_ge >= remaining_k && count_gt < remaining_k) {
          found_bucket = tx;
          found_remaining_k = remaining_k - count_gt;
        }
      }
      __syncthreads();

      // Update local caches (all CTAs have same values)
      if (tx == 0) {
        prefix_cache = prefix | (found_bucket << shift);
        remaining_k_cache = found_remaining_k;
      }
      __syncthreads();

      // No second barrier needed! Double-buffering allows next round to proceed
      // because it uses a different histogram (other_hist is already cleared)
    }

    // Convert final ordered representation back to DType pivot using type traits
    OrderedType ordered_pivot = static_cast<OrderedType>(prefix_cache);
    pivot = Traits::FromOrdered(ordered_pivot);

    // ========== Stage 3: Final masking pass ==========
    // Reuse logits_vec from Stage 1
    const DType neg_inf = Traits::NegInf();

#pragma unroll 2
    for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
      logits_vec.cast_load(logits + row_idx * vocab_size + chunk_start + i);
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        logits_vec[j] = (logits_vec[j] >= pivot) ? logits_vec[j] : neg_inf;
      }
      logits_vec.store(masked_logits + row_idx * vocab_size + chunk_start + i);
    }

    // Handle tail
    for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
      DType val = logits[row_idx * vocab_size + chunk_start + i];
      masked_logits[row_idx * vocab_size + chunk_start + i] = (val >= pivot) ? val : neg_inf;
    }
  }

  // Reset arrival counter for next kernel launch (only for multi-CTA)
  if constexpr (!SINGLE_CTA) {
    if (cta_in_group == 0 && tx == 0) {
      st_release(&state->arrival_counter, 0);
    }
  }

#undef prefix_cache
#undef remaining_k_cache
#undef found_bucket
#undef found_remaining_k
}

template <typename DType, typename IdType>
cudaError_t RadixTopKMaskLogitsMultiCTA(DType* logits, DType* masked_logits, IdType* top_k_arr,
                                        uint32_t batch_size, uint32_t top_k_val,
                                        uint32_t vocab_size, RadixRowState* row_states_buffer,
                                        cudaStream_t stream = 0) {
  using OrderedType = typename RadixTopKTraits<DType>::OrderedType;
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(DType), vocab_size);

  // Get device properties
  int device;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
  int num_sms;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  int max_smem_per_block;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

  // Fixed shared memory overhead: histogram[256] + suffix_sum[256] + 4 scalars + alignment
  constexpr size_t fixed_smem_size = sizeof(uint32_t) * (256 + 256 + 4);
  constexpr size_t fixed_smem_aligned = ((fixed_smem_size + 15) / 16) * 16;

  // Calculate max chunk size that fits in shared memory
  // smem layout: [fixed_smem_aligned] [chunk_size * sizeof(OrderedType)]
  // For FP32: OrderedType = uint32_t (4 bytes)
  // For FP16/BF16: OrderedType = uint16_t (2 bytes) - can fit 2x more elements!
  const size_t available_for_ordered = max_smem_per_block - fixed_smem_aligned;
  uint32_t max_chunk_elements = available_for_ordered / sizeof(OrderedType);

  // Round down to multiple of vec_size
  max_chunk_elements = (max_chunk_elements / vec_size) * vec_size;

  // Ensure minimum chunk size for vectorized access
  constexpr uint32_t min_chunk_size = 16 * BLOCK_THREADS;
  max_chunk_elements = std::max(max_chunk_elements, min_chunk_size);

  // Calculate how many CTAs needed per row
  uint32_t ctas_per_group = (vocab_size + max_chunk_elements - 1) / max_chunk_elements;
  uint32_t chunk_size = (vocab_size + ctas_per_group - 1) / ctas_per_group;

  // Round up chunk_size to multiple of vec_size
  chunk_size = ((chunk_size + vec_size - 1) / vec_size) * vec_size;

  // Ensure chunk_size doesn't exceed max
  chunk_size = std::min(chunk_size, max_chunk_elements);

  // Shared memory: fixed overhead + ordered values cache (using OrderedType size)
  const uint32_t smem_size = fixed_smem_aligned + chunk_size * sizeof(OrderedType);

  // Dispatch based on whether we need single-CTA or multi-CTA path
  bool single_cta = (ctas_per_group == 1);

  // Calculate number of groups (how many rows to process concurrently)
  uint32_t num_groups = std::min(static_cast<uint32_t>(num_sms) / ctas_per_group, batch_size);
  if (num_groups == 0) num_groups = 1;
  uint32_t total_ctas = num_groups * ctas_per_group;

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    if (single_cta) {
      auto kernel =
          RadixTopKMaskLogitsKernel_MultiCTA<BLOCK_THREADS, VEC_SIZE, true, DType, IdType>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

      dim3 nblks(total_ctas);
      dim3 nthrs(BLOCK_THREADS);
      void* args[] = {&logits,     &masked_logits,     &top_k_arr,  &top_k_val,     &vocab_size,
                      &batch_size, &row_states_buffer, &chunk_size, &ctas_per_group};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    } else {
      auto kernel =
          RadixTopKMaskLogitsKernel_MultiCTA<BLOCK_THREADS, VEC_SIZE, false, DType, IdType>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

      dim3 nblks(total_ctas);
      dim3 nthrs(BLOCK_THREADS);
      void* args[] = {&logits,     &masked_logits,     &top_k_arr,  &top_k_val,     &vocab_size,
                      &batch_size, &row_states_buffer, &chunk_size, &ctas_per_group};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    }
  });

  return cudaSuccess;
}

// ==================== Multi-CTA Radix Top-K Renorm Probs ====================

/*!
 * \brief Multi-CTA Radix Top-K RenormProb kernel with unified single/multi-CTA paths.
 *
 * Finds the k-th largest probability, then normalizes all probs >= pivot to sum to 1,
 * setting all others to 0. Uses the shared RadixSelectFindPivot function.
 */
template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, bool SINGLE_CTA, typename DType,
          typename IdType>
__global__ void __launch_bounds__(BLOCK_THREADS) RadixTopKRenormProbKernel_MultiCTA(
    DType* probs,          // [batch, vocab_size]
    DType* renormed_prob,  // [batch, vocab_size]
    IdType* top_k_arr,     // [batch] or nullptr
    uint32_t top_k_val, uint32_t vocab_size, uint32_t batch_size,
    RadixRowState* row_states,  // [num_groups] (nullptr if SINGLE_CTA)
    uint32_t chunk_size,        // elements per CTA
    uint32_t ctas_per_group)    // CTAs per row (1 if SINGLE_CTA)
{
  using Traits = RadixTopKTraits<DType>;
  using OrderedType = typename Traits::OrderedType;

  constexpr uint32_t RADIX = 256;

  const uint32_t global_cta_id = blockIdx.x;
  const uint32_t group_id = global_cta_id / ctas_per_group;
  const uint32_t cta_in_group = global_cta_id % ctas_per_group;
  const uint32_t tx = threadIdx.x;

  // Shared memory layout: [fixed storage] [ordered values cache]
  extern __shared__ uint8_t smem[];

  // Fixed shared memory (at the beginning)
  // histogram[256] + suffix[256] + scalars[4] + sum_local[1]
  constexpr size_t fixed_smem_size = sizeof(uint32_t) * (RADIX + RADIX + 4) + sizeof(float);
  uint32_t* local_histogram = reinterpret_cast<uint32_t*>(smem);
  uint32_t* suffix_sum = local_histogram + RADIX;
  uint32_t* shared_scalars = suffix_sum + RADIX;
  float* shared_sum = reinterpret_cast<float*>(shared_scalars + 4);

  // Align ordered values cache to 16 bytes
  size_t ordered_offset = ((fixed_smem_size + 15) / 16) * 16;
  OrderedType* shared_ordered = reinterpret_cast<OrderedType*>(smem + ordered_offset);

  // State pointer only used when not SINGLE_CTA
  RadixRowState* state = nullptr;
  if constexpr (!SINGLE_CTA) {
    state = &row_states[group_id];
  }

  // Calculate total number of iterations for persistent loop
  uint32_t num_groups = gridDim.x / ctas_per_group;
  uint32_t total_iterations = (batch_size + num_groups - 1) / num_groups;

  int barrier_phase = 0;

  // Persistent loop over rows
  for (uint32_t iter = 0; iter < total_iterations; iter++) {
    uint32_t row_idx = group_id + iter * num_groups;

    if (row_idx >= batch_size) break;

    const uint32_t chunk_start = cta_in_group * chunk_size;
    const uint32_t chunk_end = min(chunk_start + chunk_size, vocab_size);
    const uint32_t actual_chunk_size = chunk_end - chunk_start;

    uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[row_idx];

    // For RenormProb, pivot is compared with probs (must be non-negative)
    DType pivot = DType(0);
    float normalizer = 1.0f;

    if (k >= vocab_size) {
      // k >= vocab_size: no filtering needed, just compute sum and renormalize
      // Stage 1: Compute sum
      float thread_sum = 0.0f;
      vec_t<DType, VEC_SIZE> data_vec;
      const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;

#pragma unroll 2
      for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
        data_vec.cast_load(probs + row_idx * vocab_size + chunk_start + i);
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          thread_sum += float(data_vec[j]);
        }
      }
      // Handle tail
      for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
        thread_sum += float(probs[row_idx * vocab_size + chunk_start + i]);
      }

      // Block reduction for sum
      typedef cub::BlockReduce<float, BLOCK_THREADS> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
      __syncthreads();

      if constexpr (!SINGLE_CTA) {
        // Multi-CTA: atomic add to global sum
        if (tx == 0) {
          if (cta_in_group == 0) {
            state->sum_topk = 0.0f;  // First CTA initializes
          }
        }
        // Barrier for initialization
        if (tx == 0) {
          red_release(&state->arrival_counter, 1);
        }
        int target = (barrier_phase + 1) * ctas_per_group;
        wait_ge(&state->arrival_counter, target, tx);
        barrier_phase++;
        __syncthreads();

        if (tx == 0 && block_sum > 0) {
          atomicAdd(&state->sum_topk, block_sum);
        }

        // Barrier to ensure all CTAs have contributed
        if (tx == 0) {
          red_release(&state->arrival_counter, 1);
        }
        target = (barrier_phase + 1) * ctas_per_group;
        wait_ge(&state->arrival_counter, target, tx);
        barrier_phase++;
        __syncthreads();

        normalizer = math::ptx_rcp(max(state->sum_topk, 1e-8f));
      } else {
        // Single-CTA: use block_sum directly
        if (tx == 0) {
          *shared_sum = block_sum;
        }
        __syncthreads();
        normalizer = math::ptx_rcp(max(*shared_sum, 1e-8f));
      }

      // Normalize and store
#pragma unroll 2
      for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
        data_vec.cast_load(probs + row_idx * vocab_size + chunk_start + i);
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          data_vec[j] = DType(float(data_vec[j]) * normalizer);
        }
        data_vec.store(renormed_prob + row_idx * vocab_size + chunk_start + i);
      }
      for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
        renormed_prob[row_idx * vocab_size + chunk_start + i] =
            DType(float(probs[row_idx * vocab_size + chunk_start + i]) * normalizer);
      }
      continue;
    }

    // ========== Stage 1: Find pivot using RadixSelectFindPivot ==========
    pivot = RadixSelectFindPivot<BLOCK_THREADS, VEC_SIZE, SINGLE_CTA, DType>(
        probs + row_idx * vocab_size, shared_ordered, local_histogram, suffix_sum, shared_scalars,
        state, chunk_start, actual_chunk_size, k, barrier_phase, ctas_per_group, tx);

    // ========== Stage 2: Compute sum of elements >= pivot ==========
    float thread_sum = 0.0f;
    vec_t<DType, VEC_SIZE> data_vec;
    const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;

#pragma unroll 2
    for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
      data_vec.cast_load(probs + row_idx * vocab_size + chunk_start + i);
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        if (data_vec[j] >= pivot) {
          thread_sum += float(data_vec[j]);
        }
      }
    }
    // Handle tail
    for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
      DType val = probs[row_idx * vocab_size + chunk_start + i];
      if (val >= pivot) {
        thread_sum += float(val);
      }
    }

    // Block reduction for sum
    typedef cub::BlockReduce<float, BLOCK_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    __syncthreads();

    if constexpr (!SINGLE_CTA) {
      // Multi-CTA: atomic add to global sum
      if (tx == 0) {
        if (cta_in_group == 0) {
          state->sum_topk = 0.0f;  // First CTA initializes
        }
      }
      // Barrier for initialization
      if (tx == 0) {
        red_release(&state->arrival_counter, 1);
      }
      int target = (barrier_phase + 1) * ctas_per_group;
      wait_ge(&state->arrival_counter, target, tx);
      barrier_phase++;
      __syncthreads();

      if (tx == 0 && block_sum > 0) {
        atomicAdd(&state->sum_topk, block_sum);
      }

      // Barrier to ensure all CTAs have contributed
      if (tx == 0) {
        red_release(&state->arrival_counter, 1);
      }
      target = (barrier_phase + 1) * ctas_per_group;
      wait_ge(&state->arrival_counter, target, tx);
      barrier_phase++;
      __syncthreads();

      normalizer = math::ptx_rcp(max(state->sum_topk, 1e-8f));
    } else {
      // Single-CTA: use block_sum directly
      if (tx == 0) {
        *shared_sum = block_sum;
      }
      __syncthreads();
      normalizer = math::ptx_rcp(max(*shared_sum, 1e-8f));
    }

    // ========== Stage 3: Normalize elements >= pivot, set others to 0 ==========
#pragma unroll 2
    for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
      data_vec.cast_load(probs + row_idx * vocab_size + chunk_start + i);
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        data_vec[j] = (data_vec[j] >= pivot) ? DType(float(data_vec[j]) * normalizer) : DType(0);
      }
      data_vec.store(renormed_prob + row_idx * vocab_size + chunk_start + i);
    }
    // Handle tail
    for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
      DType val = probs[row_idx * vocab_size + chunk_start + i];
      renormed_prob[row_idx * vocab_size + chunk_start + i] =
          (val >= pivot) ? DType(float(val) * normalizer) : DType(0);
    }
  }

  // Reset arrival counter for next kernel launch (only for multi-CTA)
  if constexpr (!SINGLE_CTA) {
    if (cta_in_group == 0 && tx == 0) {
      st_release(&state->arrival_counter, 0);
    }
  }
}

template <typename DType, typename IdType>
cudaError_t RadixTopKRenormProbMultiCTA(DType* probs, DType* renormed_prob, IdType* top_k_arr,
                                        uint32_t batch_size, uint32_t top_k_val,
                                        uint32_t vocab_size, RadixRowState* row_states_buffer,
                                        cudaStream_t stream = 0) {
  using OrderedType = typename RadixTopKTraits<DType>::OrderedType;
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(DType), vocab_size);

  // Get device properties
  int device;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
  int num_sms;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  int max_smem_per_block;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

  // Fixed shared memory overhead: histogram[256] + suffix_sum[256] + 4 scalars + 1 float +
  // alignment
  constexpr size_t fixed_smem_size = sizeof(uint32_t) * (256 + 256 + 4) + sizeof(float);
  constexpr size_t fixed_smem_aligned = ((fixed_smem_size + 15) / 16) * 16;

  // Calculate max chunk size that fits in shared memory
  const size_t available_for_ordered = max_smem_per_block - fixed_smem_aligned;
  uint32_t max_chunk_elements = available_for_ordered / sizeof(OrderedType);

  // Round down to multiple of vec_size
  max_chunk_elements = (max_chunk_elements / vec_size) * vec_size;

  // Ensure minimum chunk size for vectorized access
  constexpr uint32_t min_chunk_size = 16 * BLOCK_THREADS;
  max_chunk_elements = std::max(max_chunk_elements, min_chunk_size);

  // Calculate how many CTAs needed per row
  uint32_t ctas_per_group = (vocab_size + max_chunk_elements - 1) / max_chunk_elements;
  uint32_t chunk_size = (vocab_size + ctas_per_group - 1) / ctas_per_group;

  // Round up chunk_size to multiple of vec_size
  chunk_size = ((chunk_size + vec_size - 1) / vec_size) * vec_size;

  // Ensure chunk_size doesn't exceed max
  chunk_size = std::min(chunk_size, max_chunk_elements);

  // Shared memory: fixed overhead + ordered values cache
  const uint32_t smem_size = fixed_smem_aligned + chunk_size * sizeof(OrderedType);

  // Dispatch based on whether we need single-CTA or multi-CTA path
  bool single_cta = (ctas_per_group == 1);

  // Calculate number of groups (how many rows to process concurrently)
  uint32_t num_groups = std::min(static_cast<uint32_t>(num_sms) / ctas_per_group, batch_size);
  if (num_groups == 0) num_groups = 1;
  uint32_t total_ctas = num_groups * ctas_per_group;

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    if (single_cta) {
      auto kernel =
          RadixTopKRenormProbKernel_MultiCTA<BLOCK_THREADS, VEC_SIZE, true, DType, IdType>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

      dim3 nblks(total_ctas);
      dim3 nthrs(BLOCK_THREADS);
      void* args[] = {&probs,      &renormed_prob,     &top_k_arr,  &top_k_val,     &vocab_size,
                      &batch_size, &row_states_buffer, &chunk_size, &ctas_per_group};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    } else {
      auto kernel =
          RadixTopKRenormProbKernel_MultiCTA<BLOCK_THREADS, VEC_SIZE, false, DType, IdType>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

      dim3 nblks(total_ctas);
      dim3 nthrs(BLOCK_THREADS);
      void* args[] = {&probs,      &renormed_prob,     &top_k_arr,  &top_k_val,     &vocab_size,
                      &batch_size, &row_states_buffer, &chunk_size, &ctas_per_group};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    }
  });

  return cudaSuccess;
}

// ==================== Multi-CTA Radix Top-K (Returns Indices) ====================

/*!
 * \brief Multi-CTA Radix Top-K kernel that returns indices of top-k elements.
 *
 * Uses cooperative multi-CTA radix select to find the k-th largest element,
 * then collects indices of all elements >= pivot.
 */
template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, bool SINGLE_CTA, typename DType,
          typename IdType>
__global__ void __launch_bounds__(BLOCK_THREADS)
    RadixTopKKernel_MultiCTA(DType* input,            // [batch, vocab_size]
                             IdType* output_indices,  // [batch, top_k]
                             DType* output_values,    // [batch, top_k] or nullptr
                             IdType* top_k_arr,       // [batch] or nullptr
                             uint32_t top_k_val, uint32_t vocab_size, uint32_t batch_size,
                             RadixRowState* row_states,  // [num_groups] (nullptr if SINGLE_CTA)
                             uint32_t chunk_size,        // elements per CTA
                             uint32_t ctas_per_group)    // CTAs per row (1 if SINGLE_CTA)
{
  // Type traits for FP16/BF16/FP32 support
  using Traits = RadixTopKTraits<DType>;
  using OrderedType = typename Traits::OrderedType;

  constexpr uint32_t RADIX = 256;
  constexpr uint32_t RADIX_BITS = 8;
  constexpr uint32_t NUM_ROUNDS = Traits::template num_rounds<RADIX_BITS>();
  constexpr uint32_t ORDERED_BITS = sizeof(OrderedType) * 8;

  const uint32_t global_cta_id = blockIdx.x;
  const uint32_t group_id = global_cta_id / ctas_per_group;
  const uint32_t cta_in_group = global_cta_id % ctas_per_group;
  const uint32_t tx = threadIdx.x;

  // Shared memory layout: [fixed storage] [ordered values cache]
  extern __shared__ uint8_t smem[];

  // Fixed shared memory (at the beginning)
  // When SINGLE_CTA, we need an extra uint32 for output_counter (no global state)
  constexpr size_t num_scalars = SINGLE_CTA ? 5 : 4;
  constexpr size_t fixed_smem_size =
      sizeof(uint32_t) * (RADIX + RADIX + num_scalars);  // histogram + suffix + scalars
  uint32_t* local_histogram = reinterpret_cast<uint32_t*>(smem);
  uint32_t* suffix_sum = local_histogram + RADIX;
  uint32_t* shared_scalars = suffix_sum + RADIX;  // [prefix_cache, remaining_k_cache, found_bucket,
                                                  // found_remaining_k, (output_counter)]

  // Align ordered values cache to 16 bytes
  size_t ordered_offset = ((fixed_smem_size + 15) / 16) * 16;
  OrderedType* shared_ordered = reinterpret_cast<OrderedType*>(smem + ordered_offset);

// Aliases for scalar shared variables
#define prefix_cache shared_scalars[0]
#define remaining_k_cache shared_scalars[1]
#define found_bucket shared_scalars[2]
#define found_remaining_k shared_scalars[3]
#define shared_output_counter shared_scalars[4]  // Only valid when SINGLE_CTA

  // State pointer only used when not SINGLE_CTA
  RadixRowState* state = nullptr;
  if constexpr (!SINGLE_CTA) {
    state = &row_states[group_id];
  }

  // Calculate total number of iterations for persistent loop
  uint32_t num_groups = gridDim.x / ctas_per_group;
  uint32_t total_iterations = (batch_size + num_groups - 1) / num_groups;

  int barrier_phase = 0;

  // Persistent loop over rows
  for (uint32_t iter = 0; iter < total_iterations; iter++) {
    uint32_t row_idx = group_id + iter * num_groups;

    if (row_idx >= batch_size) break;

    const uint32_t chunk_start = cta_in_group * chunk_size;
    const uint32_t chunk_end = min(chunk_start + chunk_size, vocab_size);

    uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[row_idx];

    const uint32_t actual_chunk_size = chunk_end - chunk_start;

    if (k >= vocab_size) {
      // k >= vocab_size: return all indices
      for (uint32_t i = tx; i < actual_chunk_size; i += BLOCK_THREADS) {
        if (chunk_start + i < k) {
          output_indices[row_idx * top_k_val + chunk_start + i] =
              static_cast<IdType>(chunk_start + i);
          if (output_values != nullptr) {
            output_values[row_idx * top_k_val + chunk_start + i] =
                input[row_idx * vocab_size + chunk_start + i];
          }
        }
      }
      continue;
    }

    // ========== Stage 1: Load and convert to ordered representation in shared memory ==========
    vec_t<DType, VEC_SIZE> input_vec;
    const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;

#pragma unroll 2
    for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
      input_vec.cast_load(input + row_idx * vocab_size + chunk_start + i);
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        shared_ordered[i + j] = Traits::ToOrdered(input_vec[j]);
      }
    }
    // Handle tail
    for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
      shared_ordered[i] = Traits::ToOrdered(input[row_idx * vocab_size + chunk_start + i]);
    }
    __syncthreads();

    // Initialize local caches and clear global state
    if (tx == 0) {
      prefix_cache = 0;
      remaining_k_cache = k;
      if constexpr (SINGLE_CTA) {
        shared_output_counter = 0;  // Use shared memory counter for single CTA
      }
    }
    // Clear global histograms (only needed for multi-CTA)
    if constexpr (!SINGLE_CTA) {
      for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
        state->histogram[0][i] = 0;
        state->histogram[1][i] = 0;
      }
    }
    __syncthreads();

    // Barrier to ensure all CTAs have arrived at this iteration (skip for single CTA)
    if constexpr (!SINGLE_CTA) {
      if (tx == 0) {
        red_release(&state->arrival_counter, 1);
      }
      int target = (barrier_phase + 1) * ctas_per_group;
      wait_ge(&state->arrival_counter, target, tx);
      barrier_phase++;
      __syncthreads();

      // CTA 0 clears output counter AFTER barrier
      if (cta_in_group == 0 && tx == 0) {
        st_release(&state->output_counter, 0);
      }
      __syncthreads();
    }

    // ========== Stage 2: NUM_ROUNDS of radix select ==========
    // Using double-buffering: round N uses histogram[N % 2]
    // Round N clears histogram[(N+1) % 2] for next round's use
    for (uint32_t round = 0; round < NUM_ROUNDS; ++round) {
      uint32_t shift = ORDERED_BITS - (round + 1) * RADIX_BITS;
      // Read from local cache (no global memory access needed!)
      uint32_t prefix = prefix_cache;
      uint32_t remaining_k = remaining_k_cache;

      // For multi-CTA: pointers to global histograms
      // For single-CTA: these are not used
      uint32_t* current_hist = nullptr;
      uint32_t* other_hist = nullptr;
      if constexpr (!SINGLE_CTA) {
        current_hist = state->histogram[round % 2];
        other_hist = state->histogram[(round + 1) % 2];
      }

      // Clear local histogram AND (for multi-CTA) clear the "other" global histogram
      for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
        local_histogram[i] = 0;
        if constexpr (!SINGLE_CTA) {
          other_hist[i] = 0;  // Prepare for next round (no barrier needed!)
        }
      }
      __syncthreads();

      // Build local histogram from SHARED MEMORY (no global memory access!)
      for (uint32_t i = tx; i < actual_chunk_size; i += BLOCK_THREADS) {
        OrderedType ordered = shared_ordered[i];

        // Check if this element matches the prefix (high bits determined so far)
        OrderedType mask =
            (round == 0)
                ? OrderedType(0)
                : static_cast<OrderedType>(~OrderedType(0) << (ORDERED_BITS - round * RADIX_BITS));
        if ((ordered & mask) == static_cast<OrderedType>(prefix)) {
          uint32_t bucket = (ordered >> shift) & 0xFF;
          atomicAdd(&local_histogram[bucket], 1);
        }
      }
      __syncthreads();

      // For multi-CTA: add to global histogram and barrier
      // For single-CTA: local_histogram is already the complete histogram
      if constexpr (!SINGLE_CTA) {
        for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
          if (local_histogram[i] > 0) {
            atomicAdd(&current_hist[i], local_histogram[i]);
          }
        }

        // Barrier: wait for all CTAs to finish histogram accumulation
        if (tx == 0) {
          red_release(&state->arrival_counter, 1);
        }
        int target = (barrier_phase + 1) * ctas_per_group;
        wait_ge(&state->arrival_counter, target, tx);
        barrier_phase++;
        __syncthreads();

        // Load from global histogram to suffix_sum
        for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
          suffix_sum[i] = current_hist[i];
        }
      } else {
        // Single-CTA: copy local histogram directly to suffix_sum
        for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
          suffix_sum[i] = local_histogram[i];
        }
      }
      __syncthreads();

      // Parallel suffix sum in shared memory
      for (uint32_t stride = 1; stride < RADIX; stride *= 2) {
        uint32_t val = 0;
        if (tx < RADIX) {
          val = suffix_sum[tx];
          if (tx + stride < RADIX) {
            val += suffix_sum[tx + stride];
          }
        }
        __syncthreads();
        if (tx < RADIX) {
          suffix_sum[tx] = val;
        }
        __syncthreads();
      }

      // ALL CTAs: find threshold bucket (all compute same result)
      if (tx == 0) {
        found_bucket = 0;
        found_remaining_k = remaining_k;
      }
      __syncthreads();

      if (tx < RADIX) {
        uint32_t count_ge = suffix_sum[tx];
        uint32_t count_gt = (tx + 1 < RADIX) ? suffix_sum[tx + 1] : 0;
        if (count_ge >= remaining_k && count_gt < remaining_k) {
          found_bucket = tx;
          found_remaining_k = remaining_k - count_gt;
        }
      }
      __syncthreads();

      // Update local caches (all CTAs have same values)
      if (tx == 0) {
        prefix_cache = prefix | (found_bucket << shift);
        remaining_k_cache = found_remaining_k;
      }
      __syncthreads();
    }

    // Get final ordered pivot from prefix_cache
    OrderedType ordered_pivot = static_cast<OrderedType>(prefix_cache);

    // ========== Stage 3: Collect indices >= pivot ==========
    // Two-pass approach to handle ties correctly:
    // Pass 1: collect all elements strictly > pivot (these must be in top-k)
    // Pass 2: fill remaining slots with elements == pivot
    //
    // Optimization for Pass 1 (> pivot): Use shared memory atomic to count locally,
    // then one global atomic per CTA to get base position, then shared atomic to write.
    // This works because all > pivot elements are guaranteed to be in top-k.
    //
    // For Pass 2 (== pivot): Use global atomic directly since we need cross-CTA
    // coordination to respect the k limit (some == pivot elements may be truncated).

    // Reuse local_histogram[0..1] as counters
#define local_counter local_histogram[0]
#define global_base local_histogram[1]

    // Pass 1: Count elements > pivot locally, then write with one global atomic
    if (tx == 0) {
      local_counter = 0;
    }
    __syncthreads();

    // First pass: count how many elements > pivot in this CTA
    for (uint32_t i = tx; i < actual_chunk_size; i += BLOCK_THREADS) {
      OrderedType ordered_val = shared_ordered[i];
      if (ordered_val > ordered_pivot) {
        atomicAdd(&local_counter, 1);
      }
    }
    __syncthreads();

    // Get base position for this CTA
    uint32_t cta_count_gt = local_counter;
    if (tx == 0 && cta_count_gt > 0) {
      if constexpr (SINGLE_CTA) {
        global_base = atomicAdd(&shared_output_counter, cta_count_gt);
      } else {
        global_base = atomicAdd(&state->output_counter, cta_count_gt);
      }
    }
    __syncthreads();

    // Second pass: write elements > pivot using local shared atomic for position
    if (tx == 0) {
      local_counter = 0;  // Reset for use as write position
    }
    __syncthreads();

    if (cta_count_gt > 0) {
      for (uint32_t i = tx; i < actual_chunk_size; i += BLOCK_THREADS) {
        OrderedType ordered_val = shared_ordered[i];
        if (ordered_val > ordered_pivot) {
          uint32_t local_pos = atomicAdd(&local_counter, 1);
          int pos = global_base + local_pos;
          // No need to check pos < k here since all > pivot elements are in top-k
          output_indices[row_idx * top_k_val + pos] = static_cast<IdType>(chunk_start + i);
          if (output_values != nullptr) {
            output_values[row_idx * top_k_val + pos] = Traits::FromOrdered(ordered_val);
          }
        }
      }
    }

    // Barrier to ensure all > pivot elements are collected first (only for multi-CTA)
    if constexpr (!SINGLE_CTA) {
      if (tx == 0) {
        red_release(&state->arrival_counter, 1);
      }
      int target = (barrier_phase + 1) * ctas_per_group;
      wait_ge(&state->arrival_counter, target, tx);
      barrier_phase++;
    }
    __syncthreads();

    // Pass 2: Write elements == pivot
    for (uint32_t i = tx; i < actual_chunk_size; i += BLOCK_THREADS) {
      OrderedType ordered_val = shared_ordered[i];
      if (ordered_val == ordered_pivot) {
        int pos;
        if constexpr (SINGLE_CTA) {
          pos = atomicAdd(&shared_output_counter, 1);
        } else {
          pos = atomicAdd(&state->output_counter, 1);
        }
        if (pos < static_cast<int>(k)) {
          output_indices[row_idx * top_k_val + pos] = static_cast<IdType>(chunk_start + i);
          if (output_values != nullptr) {
            output_values[row_idx * top_k_val + pos] = Traits::FromOrdered(ordered_val);
          }
        }
      }
    }

#undef local_counter
#undef global_base
    // No barrier needed here - the barrier at the start of next iteration
    // ensures all CTAs complete Stage 3 before output_counter is reset
  }

  // Reset arrival counter for next kernel launch (only for multi-CTA)
  if constexpr (!SINGLE_CTA) {
    if (cta_in_group == 0 && tx == 0) {
      st_release(&state->arrival_counter, 0);
    }
  }

#undef prefix_cache
#undef remaining_k_cache
#undef found_bucket
#undef found_remaining_k
#undef shared_output_counter
}

/*!
 * \brief Launch multi-CTA Radix Top-K kernel (returns indices and optionally values)
 *
 * \param input Input tensor [batch_size, vocab_size]
 * \param output_indices Output indices tensor [batch_size, top_k]
 * \param output_values Output values tensor [batch_size, top_k] or nullptr if not needed
 * \param top_k_arr Per-row top-k values or nullptr for uniform top_k
 * \param batch_size Number of rows
 * \param top_k_val Default top-k value (used when top_k_arr is nullptr)
 * \param vocab_size Number of elements per row
 * \param row_states_buffer Buffer for inter-CTA synchronization
 * \param stream CUDA stream
 */
template <typename DType, typename IdType>
cudaError_t RadixTopKMultiCTA(DType* input, IdType* output_indices, DType* output_values,
                              IdType* top_k_arr, uint32_t batch_size, uint32_t top_k_val,
                              uint32_t vocab_size, RadixRowState* row_states_buffer,
                              cudaStream_t stream = 0) {
  using OrderedType = typename RadixTopKTraits<DType>::OrderedType;
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(DType), vocab_size);

  int device;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
  int num_sms;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  int max_smem_per_block;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

  // Fixed smem: histogram[256] + suffix_sum[256] + scalars
  // Multi-CTA: 4 scalars; Single-CTA: 5 scalars (extra output_counter)
  constexpr size_t fixed_smem_multi = sizeof(uint32_t) * (256 + 256 + 4);
  constexpr size_t fixed_smem_single = sizeof(uint32_t) * (256 + 256 + 5);
  constexpr size_t fixed_smem_multi_aligned = ((fixed_smem_multi + 15) / 16) * 16;
  constexpr size_t fixed_smem_single_aligned = ((fixed_smem_single + 15) / 16) * 16;

  // Use the larger one for initial calculation to be conservative
  const size_t available_for_ordered = max_smem_per_block - fixed_smem_single_aligned;
  uint32_t max_chunk_elements = available_for_ordered / sizeof(OrderedType);
  max_chunk_elements = (max_chunk_elements / vec_size) * vec_size;
  constexpr uint32_t min_chunk_size = 16 * BLOCK_THREADS;
  max_chunk_elements = std::max(max_chunk_elements, min_chunk_size);

  uint32_t ctas_per_group = (vocab_size + max_chunk_elements - 1) / max_chunk_elements;
  uint32_t chunk_size = (vocab_size + ctas_per_group - 1) / ctas_per_group;
  chunk_size = ((chunk_size + vec_size - 1) / vec_size) * vec_size;
  chunk_size = std::min(chunk_size, max_chunk_elements);

  // Determine if we use single-CTA path
  const bool single_cta = (ctas_per_group == 1);

  // Calculate smem_size
  const uint32_t smem_size = fixed_smem_multi_aligned + chunk_size * sizeof(OrderedType);

  // Calculate number of groups (how many rows to process concurrently)
  uint32_t num_groups = std::min(static_cast<uint32_t>(num_sms) / ctas_per_group, batch_size);
  if (num_groups == 0) num_groups = 1;
  uint32_t total_ctas = num_groups * ctas_per_group;

  // Helper macro that sets attribute and launches kernel
#define DISPATCH_VEC_SIZE_LAUNCH(vec_size, VEC_SIZE, SINGLE_CTA)                                  \
  if (vec_size == VEC_SIZE) {                                                                     \
    auto kernel = RadixTopKKernel_MultiCTA<BLOCK_THREADS, VEC_SIZE, SINGLE_CTA, DType, IdType>;   \
    FLASHINFER_CUDA_CALL(                                                                         \
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));    \
    dim3 nblks(total_ctas);                                                                       \
    dim3 nthrs(BLOCK_THREADS);                                                                    \
    void* args[] = {                                                                              \
        &input,      &output_indices, &output_values,     &top_k_arr,  &top_k_val,                \
        &vocab_size, &batch_size,     &row_states_buffer, &chunk_size, &ctas_per_group};          \
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream)); \
  }

  if (single_cta) {
    DISPATCH_VEC_SIZE_LAUNCH(vec_size, 1, true);
    DISPATCH_VEC_SIZE_LAUNCH(vec_size, 2, true);
    DISPATCH_VEC_SIZE_LAUNCH(vec_size, 4, true);
    DISPATCH_VEC_SIZE_LAUNCH(vec_size, 8, true);
  } else {
    DISPATCH_VEC_SIZE_LAUNCH(vec_size, 1, false);
    DISPATCH_VEC_SIZE_LAUNCH(vec_size, 2, false);
    DISPATCH_VEC_SIZE_LAUNCH(vec_size, 4, false);
    DISPATCH_VEC_SIZE_LAUNCH(vec_size, 8, false);
  }

#undef DISPATCH_VEC_SIZE_LAUNCH

  return cudaSuccess;
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType>
__global__ void ChainSpeculativeSampling(DType* draft_probs, IdType* draft_token_ids,
                                         DType* target_probs, IdType* output_token_ids,
                                         IdType* output_accepted_token_num,
                                         IdType* output_emitted_draft_token_num,
                                         uint32_t num_speculative_tokens, uint32_t d,
                                         uint64_t philox_seed, uint64_t philox_offset) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  const uint32_t row_idx = bx;
  curandStatePhilox4_32_10_t curand_state;
  curand_init(philox_seed, bx, philox_offset, &curand_state);

  extern __shared__ __align__(
      alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(
          smem_sampling);

  uint32_t pos = num_speculative_tokens;
  for (uint32_t i = 0; i < num_speculative_tokens; ++i) {
    IdType draft_id = draft_token_ids[row_idx * num_speculative_tokens + i];
    float q = target_probs[(row_idx * (num_speculative_tokens + 1) + i) * d + draft_id],
          p = draft_probs[(row_idx * num_speculative_tokens + i) * d + draft_id];
    float u = curand_uniform(&curand_state);
    if (u * p < q) {
      // accept the draft models output
      output_token_ids[row_idx * (num_speculative_tokens + 1) + i] = draft_id;
    } else {
      pos = i;
      break;
    }
  }

  uint32_t emitted_token_num = pos;
  uint32_t accepted_token_num = pos;
  for (uint32_t i = pos; i < num_speculative_tokens; ++i) {
    int draft_id = draft_token_ids[row_idx * num_speculative_tokens + i];
    float q = target_probs[(row_idx * (num_speculative_tokens + 1) + i) * d + draft_id],
          p = draft_probs[(row_idx * num_speculative_tokens + i) * d + draft_id];
    float u = curand_uniform(&curand_state);
    if (u * p < q) {
      ++accepted_token_num;
    }
  }

  if (tx == 0) {
    output_accepted_token_num[row_idx] += accepted_token_num;
    output_emitted_draft_token_num[row_idx] += emitted_token_num;
  }

  // sample from relu(target_probs - draft_probs)
  float sum_relu_q_minus_p = 0;
  vec_t<float, VEC_SIZE> q_vec, p_vec;
  float relu_q_minus_p[VEC_SIZE];
#pragma unroll 2
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    q_vec.fill(0);
    p_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      q_vec.cast_load(target_probs + (row_idx * (num_speculative_tokens + 1) + pos) * d +
                      i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      if (pos != num_speculative_tokens) {
        // there is no draft_probs for the bonus token
        p_vec.cast_load(draft_probs + (row_idx * num_speculative_tokens + pos) * d +
                        i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      }
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      relu_q_minus_p[j] = max(q_vec[j] - p_vec[j], 0.0f);
    }
    sum_relu_q_minus_p +=
        BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
            .Sum<VEC_SIZE>(relu_q_minus_p);
    __syncthreads();
  }
  if (tx == 0) {
    temp_storage.block_aggregate.value = sum_relu_q_minus_p;
  }
  // init the first rejected token to d
  temp_storage.sampled_id = d;
  __syncthreads();
  sum_relu_q_minus_p = temp_storage.block_aggregate.value;
  float u = curand_uniform(&curand_state) * sum_relu_q_minus_p;

  float aggregate_relu_q_minus_p(0);
#pragma unroll 2
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    q_vec.fill(0);
    p_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      q_vec.cast_load(target_probs + (row_idx * (num_speculative_tokens + 1) + pos) * d +
                      i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      if (pos != num_speculative_tokens) {
        // there is no draft_probs for the bonus token
        p_vec.cast_load(draft_probs + (row_idx * num_speculative_tokens + pos) * d +
                        i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      }
    }

    vec_t<float, VEC_SIZE> relu_q_minus_p_vec;
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      relu_q_minus_p_vec[j] = max(q_vec[j] - p_vec[j], 0.0f);
    }

    DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM,
                           DETERMINISTIC>(
        i, d, [&](float x) { return x > 0; }, u, relu_q_minus_p_vec, aggregate_relu_q_minus_p,
        &temp_storage);
    if (aggregate_relu_q_minus_p > u) {
      break;
    }
  }
  __syncthreads();
  int sampled_id = temp_storage.sampled_id;
  if (sampled_id == d) {
    // NOTE(Zihao): this would happen when u is very close to 1
    // and the sum of probabilities is smaller than u
    // In this case, we use the last valid index as the sampled id
    sampled_id = temp_storage.last_valid_id;
  }
  // set the first rejected token
  output_token_ids[row_idx * (num_speculative_tokens + 1) + pos] = sampled_id;
  // move to the next token
  pos++;

  // pad remaining tokens with -1
  for (; pos < num_speculative_tokens + 1; ++pos) {
    output_token_ids[row_idx * (num_speculative_tokens + 1) + pos] = -1;
  }
}

template <typename DType, typename IdType>
cudaError_t ChainSpeculativeSampling(DType* draft_probs, IdType* draft_token_ids,
                                     DType* target_probs, IdType* output_token_ids,
                                     IdType* output_accepted_token_num,
                                     IdType* output_emitted_draft_token_num, uint32_t batch_size,
                                     uint32_t num_speculative_tokens, uint32_t d,
                                     bool deterministic, uint64_t philox_seed,
                                     uint64_t philox_offset, cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(DType), d);

  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
    const uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
    dim3 nblks(batch_size);
    dim3 nthrs(BLOCK_THREADS);
    void* args[] = {&draft_probs,
                    &draft_token_ids,
                    &target_probs,
                    &output_token_ids,
                    &output_accepted_token_num,
                    &output_emitted_draft_token_num,
                    &num_speculative_tokens,
                    &d,
                    &philox_seed,
                    &philox_offset};
    DISPATCH_ALIGNED_VEC_SIZE(
        vec_size, VEC_SIZE, {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
          auto kernel = ChainSpeculativeSampling<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE,
                                                 DETERMINISTIC, DType, IdType>;
          FLASHINFER_CUDA_CALL(
              cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        })});
    return cudaSuccess;
  });
}

}  // namespace sampling

}  // namespace flashinfer

#endif  // FLASHINFER_SAMPLING_CUH_
