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

#include <cstdint>
#include <cub/block/block_adjacent_difference.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <numeric>

#include "utils.cuh"
#include "vec_dtypes.cuh"

namespace flashinfer {

namespace sampling {

using namespace cub;

struct BoolDiffOp {
  __device__ __forceinline__ bool operator()(const bool& lhs, const bool& rhs) const {
    return lhs != rhs;
  }
};

template <typename T>
struct Pair {
  T value;
  int count;

  __device__ Pair operator+(const Pair& other) const {
    return {value + other.value, count + other.count};
  }
  __device__ Pair& operator+=(const Pair& other) {
    value += other.value;
    count += other.count;
    return *this;
  }
};

template <typename T, typename ReduceT, uint32_t BLOCK_THREADS, BlockScanAlgorithm ALGORITHM>
union SamplingTempStorage {
  typename BlockScan<T, BLOCK_THREADS, ALGORITHM>::TempStorage scan;
  typename BlockReduce<T, BLOCK_THREADS>::TempStorage reduce_;
  typename BlockReduce<ReduceT, BLOCK_THREADS>::TempStorage reduce;
  typename BlockAdjacentDifference<bool, BLOCK_THREADS>::TempStorage adj_diff;
  T reduce_result;
  struct {
    int32_t sampled_id;
    ReduceT block_aggregate;
  } data;
};

template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS, BlockScanAlgorithm ALGORITHM, typename T,
          typename ReduceT>
__device__ void DeviceSamplingFromProb(
    uint32_t i, T threshold, T u, vec_t<T, VEC_SIZE> prob_vec, T& aggregate,
    SamplingTempStorage<T, ReduceT, BLOCK_THREADS, ALGORITHM>* temp_storage) {
  T inclusive_cdf[VEC_SIZE];
  bool greater_than_u[VEC_SIZE];
  T aggregate_local = T(0);
  T prob_greater_than_threshold[VEC_SIZE];
#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    prob_greater_than_threshold[j] = (prob_vec[j] > threshold) ? prob_vec[j] : T(0);
  }
  aggregate_local = BlockReduce<T, BLOCK_THREADS>(temp_storage->reduce_)
      .Sum<VEC_SIZE>(prob_greater_than_threshold);
  if (threadIdx.x == 0) {
    temp_storage->reduce_result = aggregate_local;
  }
  __syncthreads();
  aggregate_local = temp_storage->reduce_result;

  if (aggregate > u) {
    BlockScan<T, BLOCK_THREADS, ALGORITHM>(temp_storage->scan)
        .InclusiveSum<VEC_SIZE>(prob_greater_than_threshold, inclusive_cdf);
    __syncthreads();

  #pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      inclusive_cdf[j] = inclusive_cdf[j] + aggregate;
      greater_than_u[j] = inclusive_cdf[j] > u;
    }

    BlockAdjacentDifference<bool, BLOCK_THREADS>(temp_storage->adj_diff)
        .SubtractLeft<VEC_SIZE>(greater_than_u, greater_than_u, BoolDiffOp());
    __syncthreads();

    const uint32_t tx = threadIdx.x;

  #pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      if (greater_than_u[j]) {
        temp_storage->data.sampled_id = (i * BLOCK_THREADS + tx) * VEC_SIZE + j;
      }
    }
    __syncthreads();
  } else {
    aggregate += aggregate_local;
  }
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm ALGORITHM, uint32_t VEC_SIZE, typename DType,
          typename IdType>
__global__ void SamplingFromProbKernel(DType* probs, DType* uniform_samples, IdType* output,
                                       uint32_t d) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;

  extern __shared__ __align__(alignof(SamplingTempStorage<DType, DType, BLOCK_THREADS, ALGORITHM>))
      uint8_t smem[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<DType, DType, BLOCK_THREADS, ALGORITHM>&>(smem);

  vec_t<DType, VEC_SIZE> probs_vec;
  DType aggregate(0);
  float u = uniform_samples[bx];

  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    probs_vec.fill(DType(0));
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      probs_vec.load(probs + bx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
    }

    DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, ALGORITHM, DType>(i, DType(0), u, probs_vec,
                                                                      aggregate, &temp_storage);
    if (aggregate > u) {
      break;
    }
  }
  output[bx] = (aggregate > u) ? temp_storage.data.sampled_id : d - 1;
}

template <uint32_t MAX_TOP_K_ROUNDS, uint32_t BLOCK_THREADS, BlockScanAlgorithm ALGORITHM,
          uint32_t VEC_SIZE, typename DType, typename IdType>
__global__ void TopKSamplingFromProbKernel(DType* probs, DType* uniform_samples, IdType* output,
                                           bool* success, uint32_t k, uint32_t d) {
  const uint32_t batch_size = gridDim.x;
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;

  extern __shared__ __align__(
      alignof(SamplingTempStorage<DType, Pair<DType>, BLOCK_THREADS, ALGORITHM>)) uint8_t smem[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<DType, Pair<DType>, BLOCK_THREADS, ALGORITHM>&>(smem);

  vec_t<DType, VEC_SIZE> probs_vec;
  DType aggregate;
  DType q = DType(0);
  DType pivot = DType(0);
  IdType sampled_id;
  for (uint32_t round = 0; round < MAX_TOP_K_ROUNDS; ++round) {
    DType u = uniform_samples[round * batch_size + bx] * (1 - q);
    aggregate = DType(0);
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(DType(0));
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.load(probs + bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, ALGORITHM, DType>(i, pivot, u, probs_vec,
                                                                        aggregate, &temp_storage);
      if (aggregate > u) {
        break;
      }
    }
    __syncthreads();
    sampled_id = (aggregate > u) ? temp_storage.data.sampled_id : d - 1;
    pivot = probs[bx * d + sampled_id];

    Pair<DType> aggregate_leq_pivot{DType(0), 0};
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(DType(0));
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.load(probs + bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      Pair<DType> probs_leq_pivot[VEC_SIZE];
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        probs_leq_pivot[j] = {
            (probs_vec[j] <= pivot) ? probs_vec[j] : DType(0),
            (probs_vec[j] <= pivot && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};
      }

      aggregate_leq_pivot += BlockReduce<Pair<DType>, BLOCK_THREADS>(temp_storage.reduce)
                                 .Sum<VEC_SIZE>(probs_leq_pivot);
      if (tx == 0) {
        temp_storage.data.block_aggregate = aggregate_leq_pivot;
      }
      __syncthreads();
      if (temp_storage.data.block_aggregate.count + k > d) {
        break;
      }
    }
    q = temp_storage.data.block_aggregate.value;
    if (temp_storage.data.block_aggregate.count + k > d) {
      break;
    }
  }
  __syncthreads();
  if (tx == 0) {
    if (temp_storage.data.block_aggregate.count + k <= d) {
      // failed to sample within MAX_TOP_P_ROUNDS
      success[bx] = false;
    } else {
      output[bx] = sampled_id;
      success[bx] = true;
    }
  }
}

constexpr float eps = 1e-5;

template <uint32_t MAX_TOP_P_ROUNDS, uint32_t BLOCK_THREADS, BlockScanAlgorithm ALGORITHM,
          uint32_t VEC_SIZE, typename DType, typename IdType>
__global__ void TopPSamplingFromProbKernel(DType* probs, DType* uniform_samples, IdType* output,
                                           bool* success, float p, uint32_t d) {
  const uint32_t batch_size = gridDim.x;
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;

  extern __shared__ __align__(alignof(SamplingTempStorage<DType, DType, BLOCK_THREADS, ALGORITHM>))
      uint8_t smem[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<DType, DType, BLOCK_THREADS, ALGORITHM>&>(smem);

  vec_t<DType, VEC_SIZE> probs_vec;
  DType aggregate;
  DType q = DType(0);
  DType pivot = DType(0);
  IdType sampled_id;
  for (uint32_t round = 0; round < MAX_TOP_P_ROUNDS; ++round) {
    DType u = uniform_samples[round * batch_size + bx] * (1 - q);
    aggregate = DType(0);
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(DType(0));
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.load(probs + bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, ALGORITHM, DType>(i, pivot, u, probs_vec,
                                                                        aggregate, &temp_storage);
      if (aggregate > u) {
        break;
      }
    }
    __syncthreads();
    sampled_id = (aggregate > u) ? temp_storage.data.sampled_id : d - 1;
    pivot = probs[bx * d + sampled_id];

    DType aggregate_leq_pivot = DType(0);
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(DType(0));
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.load(probs + bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      DType probs_leq_pivot[VEC_SIZE];
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        probs_leq_pivot[j] = (probs_vec[j] <= pivot) ? probs_vec[j] : DType(0);
      }

      aggregate_leq_pivot +=
          BlockReduce<DType, BLOCK_THREADS>(temp_storage.reduce).Sum<VEC_SIZE>(probs_leq_pivot);
      if (tx == 0) {
        temp_storage.data.block_aggregate = aggregate_leq_pivot;
      }
      __syncthreads();
      if (temp_storage.data.block_aggregate + p > 1 + eps) {
        break;
      }
    }
    q = temp_storage.data.block_aggregate;
    if (q + p > 1 + eps) {
      break;
    }
  }
  __syncthreads();
  if (tx == 0) {
    if (q + p <= 1 + eps) {
      // failed to sample within MAX_TOP_P_ROUNDS
      success[bx] = false;
    } else {
      output[bx] = sampled_id;
      success[bx] = true;
    }
  }
}

template <typename T, typename IdType>
cudaError_t SamplingFromProb(T* probs, T* uniform_samples, IdType* output, uint32_t batch_size,
                             uint32_t d, cudaStream_t stream = 0) {
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);
  dim3 nblks(batch_size);
  dim3 nthrs(BLOCK_THREADS);
  void* args[] = {&probs, &uniform_samples, &output, &d};
  const uint32_t smem_size =
      sizeof(SamplingTempStorage<T, T, BLOCK_THREADS, BLOCK_SCAN_RAKING_MEMOIZE>);

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel =
        SamplingFromProbKernel<BLOCK_THREADS, BLOCK_SCAN_RAKING_MEMOIZE, VEC_SIZE, T, IdType>;
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  });
  return cudaSuccess;
}

template <uint32_t MAX_TOP_K_ROUNDS, typename T, typename IdType>
cudaError_t TopKSamplingFromProb(T* probs, T* uniform_samples, IdType* output, bool* success,
                                 IdType k, uint32_t batch_size, uint32_t d,
                                 cudaStream_t stream = 0) {
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t smem_size =
      sizeof(SamplingTempStorage<T, Pair<T>, BLOCK_THREADS, BLOCK_SCAN_RAKING_MEMOIZE>);
  dim3 nblks(batch_size);
  dim3 nthrs(BLOCK_THREADS);
  void* args[] = {&probs, &uniform_samples, &output, &success, &k, &d};

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel = TopKSamplingFromProbKernel<MAX_TOP_K_ROUNDS, BLOCK_THREADS,
                                             BLOCK_SCAN_RAKING_MEMOIZE, VEC_SIZE, T, IdType>;
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  });
  return cudaSuccess;
}

template <uint32_t MAX_TOP_P_ROUNDS, typename T, typename IdType>
cudaError_t TopPSamplingFromProb(T* probs, T* uniform_samples, IdType* output, bool* success, T p,
                                 uint32_t batch_size, uint32_t d, cudaStream_t stream = 0) {
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t smem_size =
      sizeof(SamplingTempStorage<T, T, BLOCK_THREADS, BLOCK_SCAN_RAKING_MEMOIZE>);
  dim3 nblks(batch_size);
  dim3 nthrs(BLOCK_THREADS);
  void* args[] = {&probs, &uniform_samples, &output, &success, &p, &d};

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel = TopPSamplingFromProbKernel<MAX_TOP_P_ROUNDS, BLOCK_THREADS,
                                             BLOCK_SCAN_RAKING_MEMOIZE, VEC_SIZE, T, IdType>;
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  });
  return cudaSuccess;
}

}  // namespace sampling

}  // namespace flashinfer

#endif  // FLASHINFER_SAMPLING_CUH_
