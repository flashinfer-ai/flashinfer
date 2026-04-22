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
#include <cooperative_groups.h>

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

// Helper function to print kernel resource usage for debugging
template <typename KernelFunc>
void PrintKernelResourceUsage(KernelFunc kernel, uint32_t block_threads, uint32_t smem_size,
                               int cluster_size, const char* kernel_name) {
  cudaFuncAttributes attr;
  cudaError_t err = cudaFuncGetAttributes(&attr, kernel);
  if (err != cudaSuccess) {
    printf("[%s] Failed to get kernel attributes: %s\n", kernel_name, cudaGetErrorString(err));
    return;
  }

  int numBlocksPerSM = 0;
  err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, kernel, block_threads, smem_size);
  if (err != cudaSuccess) {
    printf("[%s] Failed to query occupancy: %s\n", kernel_name, cudaGetErrorString(err));
  }

  printf("=== Kernel Resource Usage: %s ===\n", kernel_name);
  printf("  Registers per thread:     %d\n", attr.numRegs);
  printf("  Static shared memory:     %zu bytes\n", attr.sharedSizeBytes);
  printf("  Dynamic shared memory:    %u bytes\n", smem_size);
  printf("  Total shared memory:      %zu bytes\n", attr.sharedSizeBytes + smem_size);
  printf("  Max threads per block:    %d\n", attr.maxThreadsPerBlock);
  printf("  Requested threads/block:  %u\n", block_threads);
  printf("  Cluster size:             %d\n", cluster_size);
  printf("  Max blocks per SM:        %d\n", numBlocksPerSM);
  printf("  Blocks needed per SM for cluster: %d\n", cluster_size);

  // Check if launch is feasible
  if (numBlocksPerSM < cluster_size) {
    printf("  [ERROR] Cannot launch: need %d blocks per SM for cluster, but only %d available!\n",
           cluster_size, numBlocksPerSM);
    printf("  Possible causes:\n");
    printf("    - Registers: %d threads * %d regs = %d total (SM has 65536)\n",
           block_threads, attr.numRegs, block_threads * attr.numRegs);
    printf("    - Shared mem: %zu bytes per block, %d blocks need %zu bytes (SM has ~228KB)\n",
           attr.sharedSizeBytes + smem_size, cluster_size,
           (attr.sharedSizeBytes + smem_size) * cluster_size);
  } else {
    printf("  [OK] Launch should succeed\n");
  }
  printf("==========================================\n");
}

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
    constexpr uint32_t BLOCK_THREADS = 512;                                    \
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
constexpr double PIVOT_CONVERGENCE_THRESHOLD = 1e-7;

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
  float max_val = -cuda::std::numeric_limits<float>::infinity(),
        min_val = cuda::std::numeric_limits<float>::infinity();
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    in_data_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      in_data_vec.cast_load(in_data + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
    }
    float in_data_[VEC_SIZE];
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      in_data_[j] = in_data_vec[j];
    }
    max_val = max(
        max_val, BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                     .Reduce<VEC_SIZE>(in_data_, MaxReduceOp{}));
    __syncthreads();
    min_val = min(
        min_val, BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                     .Reduce<VEC_SIZE>(in_data_, MinReduceOp{}));
    __syncthreads();
  }
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

  float max_val = 0;
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    in_data_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      in_data_vec.cast_load(in_data + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
    }
    float in_data_[VEC_SIZE];
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      in_data_[j] = in_data_vec[j];
    }
    max_val = max(
        max_val, BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                     .template Reduce<VEC_SIZE>(in_data_, MaxReduceOp{}));
    __syncthreads();
  }
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
      float thread_sum = 0.0f;
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        thread_sum += __expf(logits_vec[j] - block_max);
      }

      float block_sum =
          cub::BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce).Sum(thread_sum);
      __syncthreads();

      if (tx == 0) {
        float new_max = max(running_max, block_max);
        running_denominator = running_denominator * __expf(running_max - new_max) +
                              block_sum * __expf(block_max - new_max);
        running_max = new_max;

        temp_storage.shared_state.max_val = running_max;
        temp_storage.shared_state.denominator = running_denominator;
      }
      __syncthreads();
      running_max = temp_storage.shared_state.max_val;
      running_denominator = temp_storage.shared_state.denominator;
    }
  }

  const float final_max = running_max;
  const float inv_denominator = 1.0f / running_denominator;

  __syncthreads();

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
      float thread_sum = 0.0f;
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        thread_sum += __expf(logits_vec[j] - block_max);
      }

      float block_sum =
          cub::BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce).Sum(thread_sum);
      __syncthreads();

      if (tx == 0) {
        float new_max = max(running_max, block_max);
        running_denominator = running_denominator * __expf(running_max - new_max) +
                              block_sum * __expf(block_max - new_max);
        running_max = new_max;

        temp_storage.shared_state.max_val = running_max;
        temp_storage.shared_state.denominator = running_denominator;
      }
      __syncthreads();
      running_max = temp_storage.shared_state.max_val;
      running_denominator = temp_storage.shared_state.denominator;
    }
  }

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
      }

      aggregate_gt_pivot_0 += BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>(
                                  temp_storage.block_prim.reduce_value_count)
                                  .Sum<VEC_SIZE>(probs_gt_pivot_0);
      if (tx == 0) {
        temp_storage.block_aggregate.pair = aggregate_gt_pivot_0;
      }
      __syncthreads();
      aggregate_gt_pivot_0 = temp_storage.block_aggregate.pair;

      aggregate_gt_pivot_1 += BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>(
                                  temp_storage.block_prim.reduce_value_count)
                                  .Sum<VEC_SIZE>(probs_gt_pivot_1);
      if (tx == 0) {
        temp_storage.block_aggregate.pair = aggregate_gt_pivot_1;
      }
      __syncthreads();
      aggregate_gt_pivot_1 = temp_storage.block_aggregate.pair;
    }
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
      }

      aggregate_gt_pivot_0 += BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                                  .template Sum<VEC_SIZE>(probs_gt_pivot_0);
      if (tx == 0) {
        temp_storage.block_aggregate.value = aggregate_gt_pivot_0;
      }
      __syncthreads();
      aggregate_gt_pivot_0 = temp_storage.block_aggregate.value;

      aggregate_gt_pivot_1 += BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                                  .template Sum<VEC_SIZE>(probs_gt_pivot_1);
      if (tx == 0) {
        temp_storage.block_aggregate.value = aggregate_gt_pivot_1;
      }
      __syncthreads();
      aggregate_gt_pivot_1 = temp_storage.block_aggregate.value;
    }
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
    }

    aggregate_gt_pivot += BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                              .Sum<VEC_SIZE>(probs_gt_pivot);
    if (tx == 0) {
      temp_storage.block_aggregate.value = aggregate_gt_pivot;
    }
    __syncthreads();
  }

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

// Helper struct for dynamic shared memory layout in GetTopKTopPFilteredProb
template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, int PIVOTS_PER_BLOCK>
struct GetTopKTopPFilteredProbSmemLayout {
  SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM> temp_storage;
  double smem_low;
  double smem_high;
  float smem_gt_low_count;
  float smem_gt_high_count;
  // Use max(2, PIVOTS_PER_BLOCK) to ensure fused kernel Phase 1 has enough slots
  // Phase 1 always needs 2 slots for partial_0 and partial_1
  ValueCount<float> smem_pivot_aggregates[PIVOTS_PER_BLOCK > 2 ? PIVOTS_PER_BLOCK : 2];
};

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType, int cluster_size=1, int PIVOTS_PER_BLOCK=4>
__device__ void GetTopKTopPFilteredProbDeviceWithSmem(
    DType* probs, DType* filtered_probs, IdType* top_k_arr, float* top_p_arr,
    IdType* indices, IdType top_k_val, float top_p_val, uint32_t d,
    uint64_t philox_seed, uint64_t philox_offset,
    GetTopKTopPFilteredProbSmemLayout<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM, PIVOTS_PER_BLOCK>& smem,
    double low = 0, double high = 1);

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType, int cluster_size=1, int PIVOTS_PER_BLOCK=4>
__device__ void GetTopKTopPFilteredProbDevice(DType* probs, DType* filtered_probs, IdType* top_k_arr, float* top_p_arr,
                                               IdType* indices, IdType top_k_val,
                                               float top_p_val, uint32_t d, uint64_t philox_seed,
                                               uint64_t philox_offset,
                                               double low = 0, double high = 1) {
  // Dynamic shared memory
  extern __shared__ __align__(alignof(GetTopKTopPFilteredProbSmemLayout<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM, PIVOTS_PER_BLOCK>))
      uint8_t smem_raw[];
  auto& smem = *reinterpret_cast<GetTopKTopPFilteredProbSmemLayout<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM, PIVOTS_PER_BLOCK>*>(smem_raw);

  GetTopKTopPFilteredProbDeviceWithSmem<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM,
                                         VEC_SIZE, DETERMINISTIC, DType, IdType,
                                         cluster_size, PIVOTS_PER_BLOCK>(
      probs, filtered_probs, top_k_arr, top_p_arr, indices, top_k_val, top_p_val,
      d, philox_seed, philox_offset, smem, low, high);
}

// Implementation with external shared memory
template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType, int cluster_size, int PIVOTS_PER_BLOCK>
__device__ void GetTopKTopPFilteredProbDeviceWithSmem(
    DType* probs, DType* filtered_probs, IdType* top_k_arr, float* top_p_arr,
    IdType* indices, IdType top_k_val, float top_p_val, uint32_t d,
    uint64_t philox_seed, uint64_t philox_offset,
    GetTopKTopPFilteredProbSmemLayout<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM, PIVOTS_PER_BLOCK>& smem,
    double low, double high) {
  namespace cg = cooperative_groups;
  auto cluster = cg::this_cluster();
  unsigned int clusterBlockRank = cluster.block_rank();

  const uint32_t batch_size = gridDim.x / cluster_size;
  const uint32_t bx = blockIdx.x / cluster_size, tx = threadIdx.x;
  const uint32_t row_idx = indices == nullptr ? bx : indices[bx];
  const uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[row_idx];
  const float p = top_p_arr == nullptr ? top_p_val : top_p_arr[row_idx];

  // Aliases for cleaner code
  auto& temp_storage = smem.temp_storage;
  auto& smem_low = smem.smem_low;
  auto& smem_high = smem.smem_high;
  auto& smem_gt_low_count = smem.smem_gt_low_count;
  auto& smem_gt_high_count = smem.smem_gt_high_count;
  auto& smem_pivot_aggregates = smem.smem_pivot_aggregates;

  vec_t<float, VEC_SIZE> probs_vec;
  double pivot;
  int n_iter = 0;
  int gt_low_count = d, gt_high_count = 0;

  // Optimized path for cluster_size > 1: each block reads 1/cluster_size
  // of vocab and computes PIVOTS_PER_BLOCK pivots, then aggregates via DSM
  if constexpr (cluster_size > 1) {
    // Calculate this block's chunk of the vocab
    // Ensure chunk_size is aligned to VEC_SIZE so that my_start is always VEC_SIZE-aligned
    const uint32_t raw_chunk_size = ceil_div(d, (uint32_t)cluster_size);
    const uint32_t chunk_size = ((raw_chunk_size + VEC_SIZE - 1) / VEC_SIZE) * VEC_SIZE;
    const uint32_t my_start = clusterBlockRank * chunk_size;
    const uint32_t my_end = min(my_start + chunk_size, d);
    const uint32_t my_chunk_elems = (my_start < d) ? (my_end - my_start) : 0;

    do {
      if (gt_low_count - gt_high_count <= 1 || (high - low) < PIVOT_CONVERGENCE_THRESHOLD) {
        break;
      }

      // Compute PIVOTS_PER_BLOCK pivot values
      double step = (high - low) / (PIVOTS_PER_BLOCK + 1);
      double pivots[PIVOTS_PER_BLOCK];
      #pragma unroll
      for (int pv = 0; pv < PIVOTS_PER_BLOCK; ++pv) {
        pivots[pv] = low + (pv + 1) * step;
      }

      // Each thread maintains private accumulators for all pivots
      ValueCount<float> thread_sums[PIVOTS_PER_BLOCK];
      #pragma unroll
      for (int pv = 0; pv < PIVOTS_PER_BLOCK; ++pv) {
        thread_sums[pv] = {0, 0};
      }

      // Phase 1: Load and accumulate to registers (no reduce, no sync)
      #pragma unroll 2
      for (uint32_t i = 0; i < ceil_div(my_chunk_elems, BLOCK_THREADS * VEC_SIZE); ++i) {
        const uint32_t local_offset = (i * BLOCK_THREADS + tx) * VEC_SIZE;
        const uint32_t global_idx = my_start + local_offset;

        probs_vec.fill(0);
        if (global_idx < my_end && local_offset < my_chunk_elems) {
          probs_vec.cast_load(probs + row_idx * d + global_idx);
        }

        // Accumulate to thread-local registers for all pivots
        #pragma unroll
        for (int pv = 0; pv < PIVOTS_PER_BLOCK; ++pv) {
          #pragma unroll
          for (uint32_t j = 0; j < VEC_SIZE; ++j) {
            bool valid = (global_idx + j < my_end) && (local_offset + j < my_chunk_elems);
            if (probs_vec[j] > pivots[pv] && valid) {
              thread_sums[pv].value += probs_vec[j];
              thread_sums[pv].count += 1;
            }
          }
        }
        // No BlockReduce here, no __syncthreads!
      }

      // Phase 2: Single reduce at the end for all pivots
      ValueCount<float> partial_aggregates[PIVOTS_PER_BLOCK];
      #pragma unroll
      for (int pv = 0; pv < PIVOTS_PER_BLOCK; ++pv) {
        partial_aggregates[pv] =
            BlockReduce<ValueCount<float>, BLOCK_THREADS>(temp_storage.block_prim.reduce_value_count)
                .Sum(thread_sums[pv]);
        __syncthreads();
      }

      // Store partial aggregates to shared memory for DSM access
      if (tx == 0) {
        #pragma unroll
        for (int pv = 0; pv < PIVOTS_PER_BLOCK; ++pv) {
          smem_pivot_aggregates[pv] = partial_aggregates[pv];
        }
      }
      __syncthreads();

      // DSM aggregation: block 0 collects all partial results from all blocks
      cluster.sync();

      if (clusterBlockRank == 0) {
        // Aggregate results from all blocks for each pivot
        ValueCount<float> full_aggregates[PIVOTS_PER_BLOCK];
        #pragma unroll
        for (int pv = 0; pv < PIVOTS_PER_BLOCK; ++pv) {
          full_aggregates[pv] = {0, 0};
        }

        for (int blk = 0; blk < cluster_size; ++blk) {
          ValueCount<float>* remote_aggregates =
              cluster.map_shared_rank(smem_pivot_aggregates, blk);
          #pragma unroll
          for (int pv = 0; pv < PIVOTS_PER_BLOCK; ++pv) {
            full_aggregates[pv] += remote_aggregates[pv];
          }
        }

        // Binary search: find the pivot that satisfies top-k/top-p constraint
        double old_low = low;
        #pragma unroll
        for (int pv = 0; pv < PIVOTS_PER_BLOCK; ++pv) {
          pivot = old_low + (pv + 1) * step;
          if (full_aggregates[pv].count < k && full_aggregates[pv].value < p) {
            high = pivot;
            gt_high_count = full_aggregates[pv].count;
            break;
          } else {
            low = pivot;
            gt_low_count = full_aggregates[pv].count;
          }
        }

        if (tx == 0) {
          smem_low = low;
          smem_high = high;
          smem_gt_low_count = gt_low_count;
          smem_gt_high_count = gt_high_count;
        }
        __syncthreads();
      }

      // Broadcast updated [low, high] to all blocks
      cluster.sync();
      if (clusterBlockRank != 0) {
        low = *cluster.map_shared_rank(&smem_low, 0);
        high = *cluster.map_shared_rank(&smem_high, 0);
        gt_low_count = *cluster.map_shared_rank(&smem_gt_low_count, 0);
        gt_high_count = *cluster.map_shared_rank(&smem_gt_high_count, 0);
      }

      ++n_iter;
    } while (low < high);

  } else {
    // Original path for cluster_size == 1: single block reads full vocab
    // Optimized: accumulate to registers first, then reduce once at the end
    do {
      if (gt_low_count-gt_high_count <= 1 || (high-low) < 1e-7) {
        break;
      }
      double step = (high-low) / (cluster_size+1);
      pivot = low + (clusterBlockRank+1) * step;

      // Thread-local accumulator
      ValueCount<float> thread_sum{0, 0};

      // Phase 1: Load and accumulate to registers (no reduce, no sync)
#pragma unroll 2
      for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
        probs_vec.fill(0);
        if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
          probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
        }

        // Accumulate to thread-local register
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          bool valid = (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d;
          if (probs_vec[j] > pivot && valid) {
            thread_sum.value += probs_vec[j];
            thread_sum.count += 1;
          }
        }
        // No BlockReduce here, no __syncthreads!
      }

      // Phase 2: Single reduce at the end
      ValueCount<float> aggregate_gt_pivot =
          BlockReduce<ValueCount<float>, BLOCK_THREADS>(temp_storage.block_prim.reduce_value_count)
              .Sum(thread_sum);
      if (tx == 0) {
        temp_storage.block_aggregate.pair = aggregate_gt_pivot;
      }
      __syncthreads();
      aggregate_gt_pivot = temp_storage.block_aggregate.pair;

      if (aggregate_gt_pivot.count < k && aggregate_gt_pivot.value < p) {
        high = pivot;
        gt_high_count = aggregate_gt_pivot.count;
      } else {
        low = pivot;
        gt_low_count = aggregate_gt_pivot.count;
      }

      ++n_iter;
    } while (low < high);
  }

  if (clusterBlockRank != 0) return;
  __syncthreads();

  // return filtered p
  auto pred = [&](float x) { return x >= high; };
#pragma unroll 2
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    probs_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      probs_vec[j] = pred(probs_vec[j]) ? probs_vec[j] : 0;
    }
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      probs_vec.cast_store(filtered_probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
    }
  }
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
  int n_iter = 0;
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
      }

      aggregate_gt_pivot_0 +=
          BlockReduce<ValueCount<float>, BLOCK_THREADS>(temp_storage.block_prim.reduce_value_count)
              .Sum<VEC_SIZE>(probs_gt_pivot_0);
      if (tx == 0) {
        temp_storage.block_aggregate.pair = aggregate_gt_pivot_0;
      }
      __syncthreads();
      aggregate_gt_pivot_0 = temp_storage.block_aggregate.pair;

      aggregate_gt_pivot_1 +=
          BlockReduce<ValueCount<float>, BLOCK_THREADS>(temp_storage.block_prim.reduce_value_count)
              .Sum<VEC_SIZE>(probs_gt_pivot_1);
      if (tx == 0) {
        temp_storage.block_aggregate.pair = aggregate_gt_pivot_1;
      }
      __syncthreads();
      aggregate_gt_pivot_1 = temp_storage.block_aggregate.pair;
    }
    ++n_iter;
    // if(tx == 0){
    //   printf("n_iter=%d, \n", n_iter);
    // }
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

// Fused TopK-TopP Sampling + Filter Kernel
// Calls GetTopKTopPFilteredProbDevice at the end of sampling
// Supports Cluster (DSM) optimization and loop-external reduce

// Reuse GetTopKTopPFilteredProbSmemLayout for both phases
// Sampling phase uses: smem_low, smem_high, smem_pivot_aggregates[0..1]
// Filter phase uses: all fields

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType, int cluster_size = 1, int PIVOTS_PER_BLOCK = 4>
__global__ void TopKTopPSamplingAndFilterKernel(
    DType* probs, DType* filtered_probs, IdType* top_k_arr, float* top_p_arr,
    IdType* output, IdType* indices, IdType top_k_val, float top_p_val,
    uint32_t d, uint64_t philox_seed, uint64_t philox_offset) {

  namespace cg = cooperative_groups;
  auto cluster = cg::this_cluster();
  unsigned int clusterBlockRank = cluster.block_rank();

  const uint32_t batch_size = gridDim.x / cluster_size;
  const uint32_t bx = blockIdx.x / cluster_size, tx = threadIdx.x;
  const uint32_t row_idx = indices == nullptr ? bx : indices[bx];
  const uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[row_idx];
  const float p = top_p_arr == nullptr ? top_p_val : top_p_arr[row_idx];

  // Dynamic shared memory - reuse GetTopKTopPFilteredProbSmemLayout
  extern __shared__ __align__(
      alignof(GetTopKTopPFilteredProbSmemLayout<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM, PIVOTS_PER_BLOCK>))
      uint8_t smem_raw[];
  auto& smem = *reinterpret_cast<
      GetTopKTopPFilteredProbSmemLayout<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM, PIVOTS_PER_BLOCK>*>(smem_raw);
  auto& temp_storage = smem.temp_storage;

  // Random state (only block 0 uses it)
  curandStatePhilox4_32_10_t state;
  if (clusterBlockRank == 0) {
    curand_init(philox_seed, bx, philox_offset, &state);
  }

  // Calculate this block's chunk of the vocab for cluster mode
  // Ensure chunk_size is aligned to VEC_SIZE so that my_start is always VEC_SIZE-aligned
  const uint32_t raw_chunk_size = ceil_div(d, (uint32_t)cluster_size);
  const uint32_t chunk_size = ((raw_chunk_size + VEC_SIZE - 1) / VEC_SIZE) * VEC_SIZE;
  const uint32_t my_start = clusterBlockRank * chunk_size;
  const uint32_t my_end = min(my_start + chunk_size, d);
  const uint32_t my_chunk_elems = (my_start < d) ? (my_end - my_start) : 0;

  vec_t<float, VEC_SIZE> probs_vec;
  float aggregate;
  float q = 1;
  double low = 0, high = 1.0;
  int sampled_id = d;
  int n_iter = 0;

  // Phase 1: Sampling with Cluster + Loop-external Reduce optimization
  do {
    // Step 1: Only block 0 performs sampling
    if (clusterBlockRank == 0) {
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
        sampled_id = temp_storage.last_valid_id;
      }

      // Compute pivots
      double pivot_0 = probs[row_idx * d + sampled_id];
      double pivot_1 = (pivot_0 + high) / 2;

      // Store to shared memory for cluster broadcast (reuse smem_low/smem_high)
      if (tx == 0) {
        smem.smem_low = pivot_0;
        smem.smem_high = pivot_1;
      }
    }

    // Sync across cluster to broadcast pivots
    cluster.sync();

    // All blocks read pivots from block 0
    double pivot_0, pivot_1;
    if (clusterBlockRank == 0) {
      pivot_0 = smem.smem_low;
      pivot_1 = smem.smem_high;
    } else {
      pivot_0 = *cluster.map_shared_rank(&smem.smem_low, 0);
      pivot_1 = *cluster.map_shared_rank(&smem.smem_high, 0);
    }

    // Step 2: All blocks cooperatively compute statistics (DSM + loop-external reduce)
    // Each block processes its chunk of vocab
    ValueCount<float> thread_sum_0{0, 0}, thread_sum_1{0, 0};

    if constexpr (cluster_size > 1) {
      // Cluster mode: each block processes 1/cluster_size of vocab
#pragma unroll 2
      for (uint32_t i = 0; i < ceil_div(my_chunk_elems, BLOCK_THREADS * VEC_SIZE); ++i) {
        const uint32_t local_offset = (i * BLOCK_THREADS + tx) * VEC_SIZE;
        const uint32_t global_idx = my_start + local_offset;

        probs_vec.fill(0);
        if (global_idx < my_end && local_offset < my_chunk_elems) {
          probs_vec.cast_load(probs + row_idx * d + global_idx);
        }

        // Accumulate to thread-local registers (no reduce, no sync in loop)
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          bool valid = (global_idx + j < my_end) && (local_offset + j < my_chunk_elems);
          if (probs_vec[j] > pivot_0 && valid) {
            thread_sum_0.value += probs_vec[j];
            thread_sum_0.count += 1;
          }
          if (probs_vec[j] > pivot_1 && valid) {
            thread_sum_1.value += probs_vec[j];
            thread_sum_1.count += 1;
          }
        }
      }
    } else {
      // Non-cluster mode: single block processes full vocab
#pragma unroll 2
      for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
        probs_vec.fill(0);
        if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
          probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
        }

        // Accumulate to thread-local registers (no reduce, no sync in loop)
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          bool valid = (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d;
          if (probs_vec[j] > pivot_0 && valid) {
            thread_sum_0.value += probs_vec[j];
            thread_sum_0.count += 1;
          }
          if (probs_vec[j] > pivot_1 && valid) {
            thread_sum_1.value += probs_vec[j];
            thread_sum_1.count += 1;
          }
        }
      }
    }

    // Loop-external reduce: only once after processing all chunks
    ValueCount<float> partial_0 =
        BlockReduce<ValueCount<float>, BLOCK_THREADS>(temp_storage.block_prim.reduce_value_count)
            .Sum(thread_sum_0);
    __syncthreads();

    ValueCount<float> partial_1 =
        BlockReduce<ValueCount<float>, BLOCK_THREADS>(temp_storage.block_prim.reduce_value_count)
            .Sum(thread_sum_1);
    __syncthreads();

    // Store partial results to shared memory for DSM (reuse smem_pivot_aggregates)
    if (tx == 0) {
      smem.smem_pivot_aggregates[0] = partial_0;
      smem.smem_pivot_aggregates[1] = partial_1;
    }
    __syncthreads();

    // Step 3: Block 0 aggregates DSM results and updates interval
    cluster.sync();

    ValueCount<float> aggregate_gt_pivot_0{0, 0}, aggregate_gt_pivot_1{0, 0};
    if (clusterBlockRank == 0) {
      // Aggregate from all blocks
      for (int blk = 0; blk < cluster_size; ++blk) {
        if (blk == 0) {
          aggregate_gt_pivot_0 += smem.smem_pivot_aggregates[0];
          aggregate_gt_pivot_1 += smem.smem_pivot_aggregates[1];
        } else {
          aggregate_gt_pivot_0 += *cluster.map_shared_rank(&smem.smem_pivot_aggregates[0], blk);
          aggregate_gt_pivot_1 += *cluster.map_shared_rank(&smem.smem_pivot_aggregates[1], blk);
        }
      }

      // Update interval based on statistics
      ++n_iter;
      if (aggregate_gt_pivot_0.count < k && aggregate_gt_pivot_0.value < p) {
        // case 1: pivot_0 accepted, break
        if (tx == 0) {
          smem.smem_low = low;
          smem.smem_high = high;
          smem.smem_gt_low_count = -1.0f;  // signal to break (reuse smem_gt_low_count)
        }
      } else if (aggregate_gt_pivot_1.count < k && aggregate_gt_pivot_1.value < p) {
        // case 2: pivot_0 rejected, pivot_1 accepted
        low = pivot_0;
        high = pivot_1;
        q = aggregate_gt_pivot_0.value;
        if (tx == 0) {
          smem.smem_low = low;
          smem.smem_high = high;
          smem.smem_gt_low_count = q;
        }
      } else {
        // case 3: pivot_0 rejected, pivot_1 rejected
        low = pivot_1;
        q = aggregate_gt_pivot_1.value;
        if (tx == 0) {
          smem.smem_low = low;
          smem.smem_high = high;
          smem.smem_gt_low_count = q;
        }
      }
      __syncthreads();
    }

    // Broadcast updated interval to all blocks
    cluster.sync();
    if (clusterBlockRank != 0) {
      low = *cluster.map_shared_rank(&smem.smem_low, 0);
      high = *cluster.map_shared_rank(&smem.smem_high, 0);
      q = *cluster.map_shared_rank(&smem.smem_gt_low_count, 0);
    } else {
      low = smem.smem_low;
      high = smem.smem_high;
      q = smem.smem_gt_low_count;
    }

    // Check break condition
    if (q < 0) {
      break;
    }

  } while (low < high);

  // Get sampled_prob for Phase 2 filtering threshold
  // This ensures sampled token is always in the filtered set
  double sampled_prob = 0;
  if (clusterBlockRank == 0) {
    sampled_prob = probs[row_idx * d + sampled_id];
    // Store to shared memory for broadcast (reuse smem_low)
    if (tx == 0) {
      smem.smem_low = sampled_prob;
    }
  }
  __syncthreads();
  cluster.sync();

  // All blocks read sampled_prob from block 0
  if (clusterBlockRank != 0) {
    sampled_prob = *cluster.map_shared_rank(&smem.smem_low, 0);
  } else {
    sampled_prob = smem.smem_low;
  }

  // Write sampling result (only block 0, thread 0)
  if (clusterBlockRank == 0 && tx == 0) {
    output[row_idx] = sampled_id;
  }

  // Phase 2: Call GetTopKTopPFilteredProbDeviceWithSmem with [low, high]
  // Use min(high, sampled_prob) as threshold to ensure sampled token is included
  // Reuse the same shared memory (layout is compatible)

  // Early exit: if no filtering needed (k >= vocab_size && p >= 1.0),
  // just copy probs to filtered_probs directly
  if (k >= d && p >= 1.0f) {
    if (clusterBlockRank == 0) {
      for (uint32_t i = tx; i < d; i += BLOCK_THREADS) {
        filtered_probs[row_idx * d + i] = probs[row_idx * d + i];
      }
    }
    return;
  }

  double filter_threshold = (high < sampled_prob) ? high : sampled_prob;
  auto& filter_smem = *reinterpret_cast<
      GetTopKTopPFilteredProbSmemLayout<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM, PIVOTS_PER_BLOCK>*>(smem_raw);
  GetTopKTopPFilteredProbDeviceWithSmem<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM,
                                         VEC_SIZE, DETERMINISTIC, DType, IdType,
                                         cluster_size, PIVOTS_PER_BLOCK>(
      probs, filtered_probs, top_k_arr, top_p_arr, indices,
      top_k_val, top_p_val, d, philox_seed, philox_offset,
      filter_smem, low, filter_threshold);
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
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);
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
}

template <typename T, typename IdType>
cudaError_t SamplingFromProb(T* probs, IdType* output, IdType* indices, uint32_t batch_size,
                             uint32_t d, bool deterministic, uint64_t philox_seed,
                             uint64_t philox_offset, cudaStream_t stream = 0) {
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);
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
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

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
}

template <typename T, typename IdType>
cudaError_t MinPSamplingFromProb(T* probs, T* min_p_arr, IdType* output, IdType* indices,
                                 uint32_t batch_size, float min_p_val, uint32_t d,
                                 bool deterministic, uint64_t philox_seed, uint64_t philox_offset,
                                 cudaStream_t stream = 0) {
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

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

// Host function for Fused TopK-TopP Sampling + Filter
template <typename T, typename IdType>
cudaError_t TopKTopPSamplingAndFilter(T* probs, IdType* top_k_arr, T* top_p_arr,
                                       T* filtered_probs, IdType* output, IdType* indices,
                                       uint32_t batch_size, IdType top_k_val, T top_p_val,
                                       uint32_t d, bool deterministic,
                                       uint64_t philox_seed, uint64_t philox_offset,
                                       cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_ALIGNED_VEC_SIZE(
      vec_size, VEC_SIZE, {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {

        // Helper lambda to launch kernel with cluster config
        auto launch_kernel = [&](auto kernel, uint32_t smem_size, int cluster_size,
                                 uint32_t block_threads) -> cudaError_t {
          cudaLaunchAttribute attribute[1];
          attribute[0].id = cudaLaunchAttributeClusterDimension;
          attribute[0].val.clusterDim.x = cluster_size;
          attribute[0].val.clusterDim.y = 1;
          attribute[0].val.clusterDim.z = 1;

          cudaLaunchConfig_t config = {0};
          config.gridDim = batch_size * cluster_size;
          config.blockDim = block_threads;
          config.dynamicSmemBytes = smem_size;
          config.stream = stream;
          config.numAttrs = 1;
          config.attrs = attribute;

          cudaError_t status =
              cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
          if (status != cudaSuccess) return status;
          status = cudaLaunchKernelEx(&config, kernel, probs, filtered_probs, top_k_arr, top_p_arr,
                                      output, indices, top_k_val, top_p_val, d, philox_seed,
                                      philox_offset);
          return status;
        };

        cudaError_t status = cudaSuccess;
        if (batch_size <= 8) {
          // Small batch: maximize cluster parallelism
          constexpr uint32_t BLOCK_THREADS = 1024;
          constexpr int cluster_size = 8;
          constexpr int PIVOTS_PER_BLOCK = 1;
          const uint32_t smem_size =
              sizeof(GetTopKTopPFilteredProbSmemLayout<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, PIVOTS_PER_BLOCK>);
          auto kernel = TopKTopPSamplingAndFilterKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO,
                                                        VEC_SIZE, DETERMINISTIC, T, IdType,
                                                        cluster_size, PIVOTS_PER_BLOCK>;
          status = launch_kernel(kernel, smem_size, cluster_size, BLOCK_THREADS);

        } else if (batch_size <= 32) {
          // Medium batch: reduce cluster size to avoid scheduling pressure
          constexpr uint32_t BLOCK_THREADS = 1024;
          constexpr int cluster_size = 4;
          constexpr int PIVOTS_PER_BLOCK = 1;
          const uint32_t smem_size =
              sizeof(GetTopKTopPFilteredProbSmemLayout<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, PIVOTS_PER_BLOCK>);
          auto kernel = TopKTopPSamplingAndFilterKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO,
                                                        VEC_SIZE, DETERMINISTIC, T, IdType,
                                                        cluster_size, PIVOTS_PER_BLOCK>;
          status = launch_kernel(kernel, smem_size, cluster_size, BLOCK_THREADS);

        } else if (batch_size <= 40) {
          // Medium-large batch: use smaller cluster
          constexpr uint32_t BLOCK_THREADS = 1024;
          constexpr int cluster_size = 2;
          constexpr int PIVOTS_PER_BLOCK = 1;
          const uint32_t smem_size =
              sizeof(GetTopKTopPFilteredProbSmemLayout<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, PIVOTS_PER_BLOCK>);
          auto kernel = TopKTopPSamplingAndFilterKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO,
                                                        VEC_SIZE, DETERMINISTIC, T, IdType,
                                                        cluster_size, PIVOTS_PER_BLOCK>;
          status = launch_kernel(kernel, smem_size, cluster_size, BLOCK_THREADS);

        } else {
          // Large batch: batch parallelism is sufficient, no cluster needed
          constexpr uint32_t BLOCK_THREADS = 1024;
          constexpr int cluster_size = 1;
          constexpr int PIVOTS_PER_BLOCK = 1;
          const uint32_t smem_size =
              sizeof(GetTopKTopPFilteredProbSmemLayout<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, PIVOTS_PER_BLOCK>);
          auto kernel = TopKTopPSamplingAndFilterKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO,
                                                        VEC_SIZE, DETERMINISTIC, T, IdType,
                                                        cluster_size, PIVOTS_PER_BLOCK>;
          status = launch_kernel(kernel, smem_size, cluster_size, BLOCK_THREADS);
        }
        if (status != cudaSuccess) return status;
      })});
  return cudaSuccess;
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
      }

      aggregate_gt_pivot_0 +=
          BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
              .template Sum<VEC_SIZE>(probs_gt_pivot_0);
      __syncthreads();

      aggregate_gt_pivot_1 +=
          BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
              .template Sum<VEC_SIZE>(probs_gt_pivot_1);
      __syncthreads();
    }
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

template <uint32_t BLOCK_THREADS, BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE,
          typename DType, typename IdType>
__global__ void TopKMaskLogitsKernel(DType* logits, DType* masked_logits, IdType* top_k_arr,
                                     uint32_t top_k_val, uint32_t d) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  const uint32_t row_idx = bx;
  uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[bx];
  double pivot = -cuda::std::numeric_limits<float>::infinity();
  vec_t<float, VEC_SIZE> logits_vec;
  if (k < d) {
    extern __shared__ __align__(alignof(RenormTempStorage<BLOCK_THREADS, REDUCE_ALGO>))
        uint8_t smem_renorm[];
    auto& temp_storage =
        reinterpret_cast<RenormTempStorage<BLOCK_THREADS, REDUCE_ALGO>&>(smem_renorm);
    float logits_greater_than_pivot[VEC_SIZE];  // pivot initialized to 0

    auto [min_val, max_val] = GetMinMaxValue<VEC_SIZE, BLOCK_THREADS, REDUCE_ALGORITHM,
                                             RenormTempStorage<BLOCK_THREADS, REDUCE_ALGORITHM>>(
        logits, row_idx, d, temp_storage);

    double low = (min_val == -cuda::std::numeric_limits<float>::infinity())
                     ? cuda::std::numeric_limits<float>::lowest()
                     : min_val - 1,
           high = max_val;
    float min_gt_low, max_le_high;
    // f(x) = len(nonzero(probs > x)), f(x) is non-increasing
    // min_gt_low = min{p \in probs | p > low}, max_le_high = max{p \in probs | p <= high}
    // loop invariant:
    // - f(low) >= k, f(high) < k
    // - f(low) > f(min_gt_low) >= f(max_le_high) == f(high)
    // stopping condition: min_gt_low == max_le_high
    // - f(low) >= k, f(min_gt_low) == f(max_le_high) == f(high) < k
    do {
      double pivot_0 = (high + 2 * low) / 3;
      double pivot_1 = (2 * high + low) / 3;

      int aggregate_gt_pivot_0 = 0, aggregate_gt_pivot_1 = 0;
      min_gt_low = high;
      max_le_high = low;
#pragma unroll 2
      for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
        logits_vec.fill(0);
        if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
          logits_vec.cast_load(logits + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
        }
        int probs_gt_pivot_0_count[VEC_SIZE], probs_gt_pivot_1_count[VEC_SIZE];
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          probs_gt_pivot_0_count[j] =
              logits_vec[j] > pivot_0 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d;
          probs_gt_pivot_1_count[j] =
              logits_vec[j] > pivot_1 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d;

          if (logits_vec[j] > low && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d) {
            min_gt_low = min(min_gt_low, logits_vec[j]);
          }
          if (logits_vec[j] <= high && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d) {
            max_le_high = max(max_le_high, logits_vec[j]);
          }
        }

        aggregate_gt_pivot_0 +=
            BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce_int)
                .Sum<VEC_SIZE>(probs_gt_pivot_0_count);
        __syncthreads();

        aggregate_gt_pivot_1 +=
            BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce_int)
                .Sum<VEC_SIZE>(probs_gt_pivot_1_count);
        __syncthreads();
      }
      min_gt_low =
          BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
              .Reduce(min_gt_low, MinReduceOp{});
      __syncthreads();
      max_le_high =
          BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
              .Reduce(max_le_high, MaxReduceOp{});
      if (tx == 0) {
        temp_storage.block_aggregate.counts[0] = aggregate_gt_pivot_0;
        temp_storage.block_aggregate.counts[1] = aggregate_gt_pivot_1;
        temp_storage.min_val = min_gt_low;
        temp_storage.max_val = max_le_high;
      }
      __syncthreads();
      aggregate_gt_pivot_0 = temp_storage.block_aggregate.counts[0];
      aggregate_gt_pivot_1 = temp_storage.block_aggregate.counts[1];
      min_gt_low = temp_storage.min_val;
      max_le_high = temp_storage.max_val;

      if (aggregate_gt_pivot_1 >= k) {
        low = pivot_1;
      } else if (aggregate_gt_pivot_0 >= k) {
        low = pivot_0;
        high = min(pivot_1, max_le_high);
      } else {
        high = min(pivot_0, max_le_high);
      }
    } while (min_gt_low != max_le_high);
    pivot = low;
  }

  // masking
#pragma unroll 2
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    logits_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      logits_vec.cast_load(logits + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      logits_vec[j] =
          (logits_vec[j] > pivot) ? logits_vec[j] : -cuda::std::numeric_limits<float>::infinity();
    }
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      logits_vec.store(masked_logits + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
    }
  }
}

template <uint32_t BLOCK_THREADS, BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE,
          typename DType, typename IdType>
__global__ void TopKRenormProbKernel(DType* probs, DType* renormed_prob, IdType* top_k_arr,
                                     uint32_t top_k_val, uint32_t d) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  const uint32_t row_idx = bx;
  uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[bx];
  double pivot = -cuda::std::numeric_limits<float>::infinity(), normalizer = 1;
  vec_t<float, VEC_SIZE> probs_vec;
  if (k < d) {
    extern __shared__ __align__(alignof(RenormTempStorage<BLOCK_THREADS, REDUCE_ALGO>))
        uint8_t smem_renorm[];
    auto& temp_storage =
        reinterpret_cast<RenormTempStorage<BLOCK_THREADS, REDUCE_ALGO>&>(smem_renorm);
    temp_storage.max_val = 0;

    float max_val = GetMaxValue<VEC_SIZE, BLOCK_THREADS, REDUCE_ALGORITHM,
                                RenormTempStorage<BLOCK_THREADS, REDUCE_ALGORITHM>>(
        probs, row_idx, d, temp_storage);

    double low = 0, high = max_val;
    float min_gt_low, max_le_high;
    float sum_low = 1;
    // f(x) = len(nonzero(probs > x)), f(x) is non-increasing
    // min_gt_low = min{p \in probs | p > low}, max_le_high = max{p \in probs | p <= high}
    // loop invariant:
    // - f(low) >= k, f(high) < k
    // - f(low) > f(min_gt_low) >= f(max_le_high) == f(high)
    // stopping condition: min_gt_low == max_le_high
    // - f(low) >= k, f(min_gt_low) == f(max_le_high) == f(high) < k
    do {
      double pivot_0 = (high + 2 * low) / 3;
      double pivot_1 = (2 * high + low) / 3;

      ValueCount<float> aggregate_gt_pivot_0{0, 0}, aggregate_gt_pivot_1{0, 0};
      min_gt_low = high;
      max_le_high = low;
#pragma unroll 1
      for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
        probs_vec.fill(0);
        if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
          probs_vec.cast_load(probs + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
        }
        ValueCount<float> probs_gt_pivot_0_pair[VEC_SIZE], probs_gt_pivot_1_pair[VEC_SIZE];
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          probs_gt_pivot_0_pair[j] = {
              (probs_vec[j] > pivot_0) ? probs_vec[j] : 0,
              (probs_vec[j] > pivot_0 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};
          probs_gt_pivot_1_pair[j] = {
              (probs_vec[j] > pivot_1) ? probs_vec[j] : 0,
              (probs_vec[j] > pivot_1 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};

          if (probs_vec[j] > low && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d) {
            min_gt_low = min(min_gt_low, probs_vec[j]);
          }
          if (probs_vec[j] <= high && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d) {
            max_le_high = max(max_le_high, probs_vec[j]);
          }
        }

        aggregate_gt_pivot_0 += BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>(
                                    temp_storage.block_prim.reduce_value_count)
                                    .template Sum<VEC_SIZE>(probs_gt_pivot_0_pair);
        __syncthreads();

        aggregate_gt_pivot_1 += BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>(
                                    temp_storage.block_prim.reduce_value_count)
                                    .template Sum<VEC_SIZE>(probs_gt_pivot_1_pair);
        __syncthreads();
      }
      min_gt_low =
          BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
              .Reduce(min_gt_low, MinReduceOp{});
      __syncthreads();
      max_le_high =
          BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
              .Reduce(max_le_high, MaxReduceOp{});
      if (tx == 0) {
        temp_storage.block_aggregate.pairs[0] = aggregate_gt_pivot_0;
        temp_storage.block_aggregate.pairs[1] = aggregate_gt_pivot_1;
        temp_storage.min_val = min_gt_low;
        temp_storage.max_val = max_le_high;
      }
      __syncthreads();
      aggregate_gt_pivot_0 = temp_storage.block_aggregate.pairs[0];
      aggregate_gt_pivot_1 = temp_storage.block_aggregate.pairs[1];
      min_gt_low = temp_storage.min_val;
      max_le_high = temp_storage.max_val;

      if (aggregate_gt_pivot_1.count >= k) {
        low = pivot_1;
        sum_low = float(aggregate_gt_pivot_1.value);
      } else if (aggregate_gt_pivot_0.count >= k) {
        low = pivot_0;
        high = min(pivot_1, max_le_high);
        sum_low = float(aggregate_gt_pivot_0.value);
      } else {
        high = min(pivot_0, max_le_high);
      }
    } while (min_gt_low != max_le_high);

    normalizer = math::ptx_rcp(max(sum_low, 1e-8));
    pivot = low;
  }

  // normalize
#pragma unroll 2
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    probs_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      probs_vec.cast_load(probs + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      probs_vec[j] = (probs_vec[j] > pivot) ? probs_vec[j] * normalizer : 0;
    }
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      probs_vec.store(renormed_prob + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
    }
  }
}

template <typename DType>
cudaError_t TopPRenormProb(DType* probs, DType* renormed_prob, float* top_p_arr,
                           uint32_t batch_size, float top_p_val, uint32_t d,
                           cudaStream_t stream = 0) {
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(DType), d);

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
}

template <typename DType, typename IdType>
cudaError_t TopKRenormProb(DType* probs, DType* renormed_prob, IdType* top_k_arr,
                           uint32_t batch_size, uint32_t top_k_val, uint32_t d,
                           cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(DType), d);

  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
    const uint32_t smem_size = sizeof(RenormTempStorage<BLOCK_THREADS, REDUCE_ALGO>);
    dim3 nblks(batch_size);
    dim3 nthrs(BLOCK_THREADS);
    void* args[] = {&probs, &renormed_prob, &top_k_arr, &top_k_val, &d};
    DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
      auto kernel = TopKRenormProbKernel<BLOCK_THREADS, REDUCE_ALGO, VEC_SIZE, DType, IdType>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    });
    return cudaSuccess;
  });
}

template <typename DType, typename IdType>
cudaError_t TopKMaskLogits(DType* logits, DType* masked_logits, IdType* top_k_arr,
                           uint32_t batch_size, uint32_t top_k_val, uint32_t d,
                           cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(DType), d);

  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
    const uint32_t smem_size = sizeof(RenormTempStorage<BLOCK_THREADS, REDUCE_ALGO>);
    dim3 nblks(batch_size);
    dim3 nthrs(BLOCK_THREADS);
    void* args[] = {&logits, &masked_logits, &top_k_arr, &top_k_val, &d};
    DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
      auto kernel = TopKMaskLogitsKernel<BLOCK_THREADS, REDUCE_ALGO, VEC_SIZE, DType, IdType>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    });
    return cudaSuccess;
  });
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
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(DType), d);

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
}

}  // namespace sampling

}  // namespace flashinfer

#endif  // FLASHINFER_SAMPLING_CUH_
