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
  typename BlockReduce<ReduceT, BLOCK_THREADS>::TempStorage reduce;
  typename BlockAdjacentDifference<bool, BLOCK_THREADS>::TempStorage adj_diff;
  struct {
    int32_t sampled_id;
    ReduceT block_aggregate;
  } data;
};

template <uint32_t vec_size, uint32_t BLOCK_THREADS, BlockScanAlgorithm ALGORITHM, typename T,
          typename ReduceT>
__device__ void DeviceSamplingFromProb(
    uint32_t i, T threshold, T u, T* prob, T& aggregate,
    SamplingTempStorage<T, ReduceT, BLOCK_THREADS, ALGORITHM>* temp_storage) {
  T inclusive_cdf[vec_size];
  bool greater_than_u[vec_size];
  T aggregate_local = T(0);
  T prob_greater_than_threshold[vec_size];
#pragma unroll
  for (uint32_t j = 0; j < vec_size; ++j) {
    prob_greater_than_threshold[j] = (prob[j] > threshold) ? prob[j] : T(0);
  }
  BlockScan<T, BLOCK_THREADS, ALGORITHM>(temp_storage->scan)
      .InclusiveSum<vec_size>(prob_greater_than_threshold, inclusive_cdf, aggregate_local);
  __syncthreads();

#pragma unroll
  for (uint32_t j = 0; j < vec_size; ++j) {
    inclusive_cdf[j] = inclusive_cdf[j] + aggregate;
    greater_than_u[j] = inclusive_cdf[j] > u;
  }
  aggregate += aggregate_local;

  BlockAdjacentDifference<bool, BLOCK_THREADS>(temp_storage->adj_diff)
      .SubtractLeft<vec_size>(greater_than_u, greater_than_u, BoolDiffOp());
  __syncthreads();

  const uint32_t tx = threadIdx.x;

#pragma unroll
  for (uint32_t j = 0; j < vec_size; ++j) {
    if (greater_than_u[j]) {
      temp_storage->data.sampled_id = (i * BLOCK_THREADS + tx) * vec_size + j;
    }
  }
  __syncthreads();
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm ALGORITHM, typename DType, typename IdType>
__global__ void SamplingFromProbKernel(DType* probs, DType* uniform_samples, IdType* output,
                                       uint32_t d) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  constexpr uint32_t vec_size = 16 / sizeof(DType);

  extern __shared__ __align__(alignof(SamplingTempStorage<DType, DType, BLOCK_THREADS, ALGORITHM>))
      uint8_t smem[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<DType, DType, BLOCK_THREADS, ALGORITHM>&>(smem);

  DType probs_local[vec_size];
  DType aggregate(0);
  float u = uniform_samples[bx];

  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * vec_size); ++i) {
    if ((i * BLOCK_THREADS + tx) * vec_size < d) {
      *((int4*)probs_local) = *((int4*)&probs[bx * d + (i * BLOCK_THREADS + tx) * vec_size]);
    } else {
      *((int4*)probs_local) = make_int4(0, 0, 0, 0);
    }

    DeviceSamplingFromProb<vec_size, BLOCK_THREADS, ALGORITHM, DType>(i, DType(0), u, probs_local,
                                                                      aggregate, &temp_storage);
    if (aggregate > u) {
      break;
    }
  }
  output[bx] = (aggregate > u) ? temp_storage.data.sampled_id : d - 1;
}

template <uint32_t MAX_TOP_K_ROUNDS, uint32_t BLOCK_THREADS, BlockScanAlgorithm ALGORITHM,
          typename DType, typename IdType>
__global__ void TopKSamplingFromProbKernel(DType* probs, DType* uniform_samples, IdType* output,
                                           bool* success, uint32_t k, uint32_t d) {
  const uint32_t batch_size = gridDim.x;
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  constexpr uint32_t vec_size = 16 / sizeof(DType);

  extern __shared__ __align__(
      alignof(SamplingTempStorage<DType, Pair<DType>, BLOCK_THREADS, ALGORITHM>)) uint8_t smem[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<DType, Pair<DType>, BLOCK_THREADS, ALGORITHM>&>(smem);

  DType probs_local[vec_size];
  DType aggregate;
  DType q = DType(0);
  DType pivot = DType(0);
  IdType sampled_id;
  for (uint32_t round = 0; round < MAX_TOP_K_ROUNDS; ++round) {
    DType u = uniform_samples[round * batch_size + bx] * (1 - q);
    aggregate = DType(0);
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * vec_size); ++i) {
      if ((i * BLOCK_THREADS + tx) * vec_size < d) {
        *((int4*)probs_local) = *((int4*)&probs[bx * d + (i * BLOCK_THREADS + tx) * vec_size]);
      } else {
        *((int4*)probs_local) = make_int4(0, 0, 0, 0);
      }

      DeviceSamplingFromProb<vec_size, BLOCK_THREADS, ALGORITHM, DType>(i, pivot, u, probs_local,
                                                                        aggregate, &temp_storage);
      if (aggregate > u) {
        break;
      }
    }
    __syncthreads();
    sampled_id = (aggregate > u) ? temp_storage.data.sampled_id : d - 1;
    pivot = probs[bx * d + sampled_id];

    Pair<DType> aggregate_leq_pivot{DType(0), 0};
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * vec_size); ++i) {
      if ((i * BLOCK_THREADS + tx) * vec_size < d) {
        *((int4*)probs_local) = *((int4*)&probs[bx * d + (i * BLOCK_THREADS + tx) * vec_size]);
      } else {
        *((int4*)probs_local) = make_int4(0, 0, 0, 0);
      }

      Pair<DType> probs_leq_pivot[vec_size];
#pragma unroll
      for (uint32_t j = 0; j < vec_size; ++j) {
        probs_leq_pivot[j] = {
            (probs_local[j] <= pivot) ? probs_local[j] : DType(0),
            (probs_local[j] <= pivot && (i * BLOCK_THREADS + tx) * vec_size + j < d)};
      }

      aggregate_leq_pivot += BlockReduce<Pair<DType>, BLOCK_THREADS>(temp_storage.reduce)
                                 .Sum<vec_size>(probs_leq_pivot);
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
          typename DType, typename IdType>
__global__ void TopPSamplingFromProbKernel(DType* probs, DType* uniform_samples, IdType* output,
                                           bool* success, float p, uint32_t d) {
  const uint32_t batch_size = gridDim.x;
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  constexpr uint32_t vec_size = 16 / sizeof(DType);

  extern __shared__ __align__(alignof(SamplingTempStorage<DType, DType, BLOCK_THREADS, ALGORITHM>))
      uint8_t smem[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<DType, DType, BLOCK_THREADS, ALGORITHM>&>(smem);

  DType probs_local[vec_size];
  DType aggregate;
  DType q = DType(0);
  DType pivot = DType(0);
  IdType sampled_id;
  for (uint32_t round = 0; round < MAX_TOP_P_ROUNDS; ++round) {
    DType u = uniform_samples[round * batch_size + bx] * (1 - q);
    aggregate = DType(0);
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * vec_size); ++i) {
      if ((i * BLOCK_THREADS + tx) * vec_size < d) {
        *((int4*)probs_local) = *((int4*)&probs[bx * d + (i * BLOCK_THREADS + tx) * vec_size]);
      } else {
        *((int4*)probs_local) = make_int4(0, 0, 0, 0);
      }

      DeviceSamplingFromProb<vec_size, BLOCK_THREADS, ALGORITHM, DType>(i, pivot, u, probs_local,
                                                                        aggregate, &temp_storage);
      if (aggregate > u) {
        break;
      }
    }
    __syncthreads();
    sampled_id = (aggregate > u) ? temp_storage.data.sampled_id : d - 1;
    pivot = probs[bx * d + sampled_id];

    DType aggregate_leq_pivot = DType(0);
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * vec_size); ++i) {
      if ((i * BLOCK_THREADS + tx) * vec_size < d) {
        *((int4*)probs_local) = *((int4*)&probs[bx * d + (i * BLOCK_THREADS + tx) * vec_size]);
      } else {
        *((int4*)probs_local) = make_int4(0, 0, 0, 0);
      }

      DType probs_leq_pivot[vec_size];
#pragma unroll
      for (uint32_t j = 0; j < vec_size; ++j) {
        probs_leq_pivot[j] = (probs_local[j] <= pivot) ? probs_local[j] : DType(0);
      }

      aggregate_leq_pivot +=
          BlockReduce<DType, BLOCK_THREADS>(temp_storage.reduce).Sum<vec_size>(probs_leq_pivot);
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

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm ALGORITHM, typename T>
__global__ void DebugThreadBlockSMEMPrefixSumKernel(T* probs, T* exclusive_cdf, uint32_t d) {
  using BlockScanT = BlockScan<T, BLOCK_THREADS, ALGORITHM>;
  using TempStorageT = typename BlockScanT::TempStorage;
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  constexpr uint32_t vec_size = 16 / sizeof(T);

  extern __shared__ uint8_t smem[];
  TempStorageT& temp_storage = reinterpret_cast<TempStorageT&>(smem);
  T* p_smem = (T*)(smem + sizeof(TempStorageT));
  T probs_local[vec_size];
  T exclusive_cdf_local[vec_size];
  T aggregate(0);
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * vec_size); ++i) {
    if ((i * BLOCK_THREADS + tx) * vec_size < d) {
      *((int4*)&p_smem[(i * BLOCK_THREADS + tx) * vec_size]) =
          *((int4*)&probs[bx * d + (i * BLOCK_THREADS + tx) * vec_size]);
    } else {
      *((int4*)&p_smem[(i * BLOCK_THREADS + tx) * vec_size]) = make_int4(0, 0, 0, 0);
    }
  }

  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * vec_size); ++i) {
    *((int4*)probs_local) = *((int4*)&p_smem[(i * BLOCK_THREADS + tx) * vec_size]);
    T aggregate_local;
    BlockScanT(temp_storage)
        .ExclusiveSum<vec_size>(probs_local, exclusive_cdf_local, aggregate_local);
    __syncthreads();
#pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      exclusive_cdf_local[j] = exclusive_cdf_local[j] + aggregate;
    }
    aggregate += aggregate_local;
    *((int4*)&p_smem[(i * BLOCK_THREADS + tx) * vec_size]) = *((int4*)exclusive_cdf_local);
  }

  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * vec_size); ++i) {
    if ((i * BLOCK_THREADS + tx) * vec_size < d) {
      *((int4*)&exclusive_cdf[bx * d + (i * BLOCK_THREADS + tx) * vec_size]) =
          *((int4*)&p_smem[(i * BLOCK_THREADS + tx) * vec_size]);
    }
  }
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm ALGORITHM, typename T>
__global__ void DebugThreadBlockPrefixSumKernel(T* probs, T* exclusive_cdf, uint32_t d) {
  using BlockScanT = BlockScan<T, BLOCK_THREADS, ALGORITHM>;
  using TempStorageT = typename BlockScanT::TempStorage;
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;

  extern __shared__ uint8_t smem[];

  auto& temp_storage = reinterpret_cast<TempStorageT&>(smem);
  constexpr uint32_t vec_size = 16 / sizeof(T);
  T probs_local[vec_size];
  T exclusive_cdf_local[vec_size];
  T aggregate(0);
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * vec_size); ++i) {
    if ((i * BLOCK_THREADS + tx) * vec_size < d) {
      *((int4*)probs_local) = *((int4*)&probs[bx * d + (i * BLOCK_THREADS + tx) * vec_size]);
    } else {
      *((int4*)probs_local) = make_int4(0, 0, 0, 0);
    }
    T aggregate_local;
    BlockScanT(temp_storage)
        .ExclusiveSum<vec_size>(probs_local, exclusive_cdf_local, aggregate_local);
    __syncthreads();
#pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      exclusive_cdf_local[j] = exclusive_cdf_local[j] + aggregate;
    }
    aggregate += aggregate_local;
    if ((i * BLOCK_THREADS + tx) * vec_size < d) {
      *((int4*)&exclusive_cdf[bx * d + (i * BLOCK_THREADS + tx) * vec_size]) =
          *((int4*)exclusive_cdf_local);
    }
  }
}

template <typename T, typename IdType>
cudaError_t SamplingFromProb(T* probs, T* uniform_samples, IdType* output, uint32_t batch_size,
                             uint32_t d, cudaStream_t stream = 0) {
  constexpr uint32_t BLOCK_THREADS = 1024;
  constexpr uint32_t vec_size = 16 / sizeof(T);
  if (d % vec_size != 0) {
    return cudaErrorInvalidValue;
  }
  const uint32_t smem_size =
      sizeof(SamplingTempStorage<T, T, BLOCK_THREADS, BLOCK_SCAN_RAKING_MEMOIZE>);
  dim3 nblks(batch_size);
  dim3 nthrs(BLOCK_THREADS);
  void* args[] = {&probs, &uniform_samples, &output, &d};
  auto kernel = SamplingFromProbKernel<BLOCK_THREADS, BLOCK_SCAN_RAKING_MEMOIZE, T, IdType>;
  FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  return cudaSuccess;
}

template <uint32_t MAX_TOP_K_ROUNDS, typename T, typename IdType>
cudaError_t TopKSamplingFromProb(T* probs, T* uniform_samples, IdType* output, bool* success,
                                 IdType k, uint32_t batch_size, uint32_t d,
                                 cudaStream_t stream = 0) {
  constexpr uint32_t BLOCK_THREADS = 1024;
  constexpr uint32_t vec_size = 16 / sizeof(T);
  if (d % vec_size != 0) {
    return cudaErrorInvalidValue;
  }
  auto kernel = TopKSamplingFromProbKernel<MAX_TOP_K_ROUNDS, BLOCK_THREADS,
                                           BLOCK_SCAN_RAKING_MEMOIZE, T, IdType>;
  const uint32_t smem_size =
      sizeof(SamplingTempStorage<T, Pair<T>, BLOCK_THREADS, BLOCK_SCAN_RAKING_MEMOIZE>);
  FLASHINFER_CUDA_CALL(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  dim3 nblks(batch_size);
  dim3 nthrs(BLOCK_THREADS);
  void* args[] = {&probs, &uniform_samples, &output, &success, &k, &d};
  FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  return cudaSuccess;
}

template <uint32_t MAX_TOP_P_ROUNDS, typename T, typename IdType>
cudaError_t TopPSamplingFromProb(T* probs, T* uniform_samples, IdType* output, bool* success, T p,
                                 uint32_t batch_size, uint32_t d, cudaStream_t stream = 0) {
  constexpr uint32_t BLOCK_THREADS = 1024;
  constexpr uint32_t vec_size = 16 / sizeof(T);
  if (d % vec_size != 0) {
    return cudaErrorInvalidValue;
  }
  auto kernel = TopPSamplingFromProbKernel<MAX_TOP_P_ROUNDS, BLOCK_THREADS,
                                           BLOCK_SCAN_RAKING_MEMOIZE, T, IdType>;
  const uint32_t smem_size =
      sizeof(SamplingTempStorage<T, T, BLOCK_THREADS, BLOCK_SCAN_RAKING_MEMOIZE>);
  FLASHINFER_CUDA_CALL(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  dim3 nblks(batch_size);
  dim3 nthrs(BLOCK_THREADS);
  void* args[] = {&probs, &uniform_samples, &output, &success, &p, &d};
  FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  return cudaSuccess;
}

template <typename T>
cudaError_t DebugThreadBlockPrefixSum(T* probs, T* exclusive_cdf, uint32_t batch_size, uint32_t d,
                                      cudaStream_t stream = 0) {
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t smem_size = sizeof(typename cub::BlockScan<T, BLOCK_THREADS>::TempStorage);
  dim3 nblks(batch_size);
  dim3 nthrs(BLOCK_THREADS);
  void* args[] = {&probs, &exclusive_cdf, &d};
  constexpr uint32_t vec_size = 16 / sizeof(T);
  if (d % vec_size != 0) {
    return cudaErrorInvalidValue;
  }
  auto kernel = DebugThreadBlockPrefixSumKernel<BLOCK_THREADS, BLOCK_SCAN_RAKING_MEMOIZE, T>;
  FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  return cudaSuccess;
}

template <typename T>
cudaError_t DebugThreadBlockSMEMPrefixSum(T* probs, T* exclusive_cdf, uint32_t batch_size,
                                          uint32_t d, cudaStream_t stream = 0) {
  constexpr uint32_t BLOCK_THREADS = 1024;
  constexpr uint32_t vec_size = 16 / sizeof(T);
  if (d % vec_size != 0) {
    return cudaErrorInvalidValue;
  }
  const uint32_t smem_size =
      sizeof(typename cub::BlockScan<T, BLOCK_THREADS>::TempStorage) +
      ceil_div(d, BLOCK_THREADS * vec_size) * (BLOCK_THREADS * vec_size) * sizeof(T);
  auto kernel = DebugThreadBlockSMEMPrefixSumKernel<BLOCK_THREADS, BLOCK_SCAN_RAKING_MEMOIZE, T>;
  FLASHINFER_CUDA_CALL(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  dim3 nblks(batch_size);
  dim3 nthrs(BLOCK_THREADS);
  void* args[] = {&probs, &exclusive_cdf, &d};
  if (d % vec_size != 0) {
    return cudaErrorInvalidValue;
  }
  FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  return cudaSuccess;
}

}  // namespace sampling

}  // namespace flashinfer

#endif  // FLASHINFER_SAMPLING_CUH_
