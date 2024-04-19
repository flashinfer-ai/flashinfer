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
#include <cub/block/block_scan.cuh>

#include "gtest/gtest.h"
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

template <typename T, uint32_t BLOCK_THREADS, BlockScanAlgorithm ALGORITHM>
union SamplingTempStorage {
  typename BlockScan<T, BLOCK_THREADS, ALGORITHM>::TempStorage scan;
  typename BlockAdjacentDifference<bool, BLOCK_THREADS>::TempStorage adj_diff;
  int32_t sampled_id;
};

template <uint32_t vec_size, uint32_t BLOCK_THREADS, BlockScanAlgorithm ALGORITHM, typename T>
__device__ void DeviceSamplingFromProb(
    uint32_t i, T u, T* prob, T& aggregate,
    SamplingTempStorage<T, BLOCK_THREADS, ALGORITHM>* temp_storage) {
  T prob_[vec_size];
#pragma unroll
  for (uint32_t j = 0; j < vec_size; ++j) {
    prob_[j] = prob[j];
  }
  T inclusive_cdf[vec_size];
  bool greater_than_p[vec_size];
  T aggregate_local = T(0);
  BlockScan<T, BLOCK_THREADS, ALGORITHM>(temp_storage->scan)
      .InclusiveSum<vec_size>(prob_, inclusive_cdf, aggregate_local);
  __syncthreads();

#pragma unroll
  for (uint32_t j = 0; j < vec_size; ++j) {
    inclusive_cdf[j] = inclusive_cdf[j] + aggregate;
    greater_than_p[j] = inclusive_cdf[j] > u;
  }
  aggregate += aggregate_local;

  BlockAdjacentDifference<bool, BLOCK_THREADS>(temp_storage->adj_diff)
      .SubtractLeft<vec_size>(greater_than_p, greater_than_p, BoolDiffOp());
  __syncthreads();

  const uint32_t tx = threadIdx.x;

#pragma unroll
  for (uint32_t j = 0; j < vec_size; ++j) {
    if (greater_than_p[j]) {
      temp_storage->sampled_id = (i * BLOCK_THREADS + tx) * vec_size + j;
    }
  }
  __syncthreads();
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm ALGORITHM, typename DType, typename IdType>
__global__ void SamplingFromProbKernel(DType* probs, DType* uniform_samples, IdType* output,
                                       uint32_t d) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  constexpr uint32_t vec_size = 16 / sizeof(DType);

  extern __shared__ __align__(alignof(SamplingTempStorage<DType, BLOCK_THREADS, ALGORITHM>))
      uint8_t smem[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<DType, BLOCK_THREADS, ALGORITHM>&>(smem);
  using BlockScanT = BlockScan<DType, BLOCK_THREADS, ALGORITHM>;
  using BlockAdjacentDifferenceT = BlockAdjacentDifference<bool, BLOCK_THREADS>;

  DType probs_local[vec_size];
  DType aggregate(0);
  float u = uniform_samples[bx];

  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * vec_size); ++i) {
    if ((i * BLOCK_THREADS + tx) * vec_size < d) {
      *((int4*)probs_local) = *((int4*)&probs[bx * d + (i * BLOCK_THREADS + tx) * vec_size]);
    } else {
      *((int4*)probs_local) = make_int4(0, 0, 0, 0);
    }

    DeviceSamplingFromProb<vec_size, BLOCK_THREADS, ALGORITHM, DType>(i, u, probs_local, aggregate,
                                                                      &temp_storage);
    if (aggregate > u) {
      break;
    }
  }
  if (aggregate <= u) {
    temp_storage.sampled_id = d - 1;
  }
  __syncthreads();
  output[bx] = temp_storage.sampled_id;
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
  const uint32_t smem_size =
      sizeof(SamplingTempStorage<T, BLOCK_THREADS, BLOCK_SCAN_RAKING_MEMOIZE>);
  dim3 nblks(batch_size);
  dim3 nthrs(BLOCK_THREADS);
  void* args[] = {&probs, &uniform_samples, &output, &d};
  auto kernel = SamplingFromProbKernel<BLOCK_THREADS, BLOCK_SCAN_RAKING_MEMOIZE, T, IdType>;
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
