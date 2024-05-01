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

#include <cub/block/block_adjacent_difference.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <numeric>

#include "utils.cuh"
#include "vec_dtypes.cuh"

namespace flashinfer {

namespace sampling {

using namespace cub;

constexpr BlockScanAlgorithm SCAN_ALGO = BLOCK_SCAN_WARP_SCANS;
constexpr BlockReduceAlgorithm REDUCE_ALGO = BLOCK_REDUCE_WARP_REDUCTIONS;

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

struct BoolDiffOp {
  __device__ __forceinline__ bool operator()(const bool& lhs, const bool& rhs) const {
    return lhs != rhs;
  }
};

template <typename T, uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM>
struct SamplingTempStorage {
  union {
    typename BlockScan<T, BLOCK_THREADS, SCAN_ALGORITHM>::TempStorage scan;
    typename BlockReduce<T, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce;
    typename BlockReduce<Pair<T>, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce_pair;
    typename BlockAdjacentDifference<bool, BLOCK_THREADS>::TempStorage adj_diff;
  } block_prim;
  struct {
    int32_t sampled_id;
    union {
      T value;
      Pair<T> pair;
    } block_aggregate;
  } data;
};

template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, typename T>
__device__ __forceinline__ void DeviceSamplingFromProb(
    uint32_t i, uint32_t d, T threshold, T u, vec_t<T, VEC_SIZE> prob_vec, T& aggregate,
    SamplingTempStorage<T, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>* temp_storage) {
  const uint32_t tx = threadIdx.x;
  T prob_greater_than_threshold[VEC_SIZE];
  T inclusive_cdf[VEC_SIZE];
  bool greater_than_u[VEC_SIZE], valid[VEC_SIZE];
#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    prob_greater_than_threshold[j] = (prob_vec[j] > threshold) ? prob_vec[j] : T(0);
    valid[j] = prob_vec[j] > threshold && (i * BLOCK_THREADS + tx) * VEC_SIZE < d;
  }
  T aggregate_local =
      BlockReduce<T, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce)
          .Sum<VEC_SIZE>(prob_greater_than_threshold);
  if (tx == 0) {
    temp_storage->data.block_aggregate.value = aggregate_local;
  }
  __syncthreads();
  aggregate_local = temp_storage->data.block_aggregate.value;

  if (aggregate + aggregate_local > u) {
    BlockScan<T, BLOCK_THREADS, SCAN_ALGORITHM>(temp_storage->block_prim.scan)
        .InclusiveSum<VEC_SIZE>(prob_greater_than_threshold, inclusive_cdf);
    __syncthreads();

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      greater_than_u[j] = inclusive_cdf[j] + aggregate > u;
    }

    BlockAdjacentDifference<bool, BLOCK_THREADS>(temp_storage->block_prim.adj_diff)
        .SubtractLeft<VEC_SIZE>(greater_than_u, greater_than_u, BoolDiffOp());
    __syncthreads();

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      if (greater_than_u[j] && valid[j]) {
        atomicMin(&(temp_storage->data.sampled_id), (i * BLOCK_THREADS + tx) * VEC_SIZE + j);
      }
    }
    __syncthreads();
  }
  aggregate += aggregate_local;
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, typename DType, typename IdType>
__global__ void SamplingFromProbKernel(DType* probs, DType* uniform_samples, IdType* output,
                                       IdType* row_indices, uint32_t d) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  const uint32_t row_idx = row_indices == nullptr ? bx : row_indices[bx];

  extern __shared__ __align__(alignof(
      SamplingTempStorage<DType, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>)) uint8_t smem[];
  auto& temp_storage = reinterpret_cast<
      SamplingTempStorage<DType, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(smem);
  temp_storage.data.sampled_id = d - 1;
  __syncthreads();

  vec_t<DType, VEC_SIZE> probs_vec;
  DType aggregate(0);
  float u = uniform_samples[bx];

  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    probs_vec.fill(DType(0));
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      probs_vec.load(probs + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
    }

    DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM, DType>(
        i, d, DType(0), u, probs_vec, aggregate, &temp_storage);
    if (float(aggregate) > u) {
      break;
    }
  }
  output[bx] = temp_storage.data.sampled_id;
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, typename DType, typename IdType>
__global__ void TopKSamplingFromProbKernel(DType* probs, DType* uniform_samples, IdType* output,
                                           bool* success, uint32_t k, uint32_t d,
                                           uint32_t max_top_k_rounds) {
  const uint32_t batch_size = gridDim.x;
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;

  extern __shared__ __align__(alignof(
      SamplingTempStorage<DType, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>)) uint8_t smem[];
  auto& temp_storage = reinterpret_cast<
      SamplingTempStorage<DType, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(smem);

  vec_t<DType, VEC_SIZE> probs_vec;
  DType aggregate;
  DType q = DType(0);
  DType pivot = DType(0);
  IdType sampled_id;
  for (uint32_t round = 0; round < max_top_k_rounds; ++round) {
    temp_storage.data.sampled_id = d - 1;
    __syncthreads();
    DType u = uniform_samples[round * batch_size + bx] * (DType(1) - q);
    aggregate = DType(0);
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(DType(0));
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.load(probs + bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM, DType>(
          i, d, pivot, u, probs_vec, aggregate, &temp_storage);
      if (aggregate > u) {
        break;
      }
    }
    __syncthreads();
    sampled_id = temp_storage.data.sampled_id;
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

      aggregate_leq_pivot +=
          BlockReduce<Pair<DType>, BLOCK_THREADS>(temp_storage.block_prim.reduce_pair)
              .Sum<VEC_SIZE>(probs_leq_pivot);
      if (tx == 0) {
        temp_storage.data.block_aggregate.pair = aggregate_leq_pivot;
      }
      __syncthreads();
      if (temp_storage.data.block_aggregate.pair.count + k > d) {
        break;
      }
    }
    q = temp_storage.data.block_aggregate.pair.value;
    if (temp_storage.data.block_aggregate.pair.count + k > d) {
      break;
    }
  }
  __syncthreads();
  if (tx == 0) {
    if (temp_storage.data.block_aggregate.pair.count + k <= d) {
      // failed to sample within MAX_TOP_P_ROUNDS
      if (success != nullptr) {
        success[bx] = false;
      }
    } else {
      output[bx] = sampled_id;
      if (success != nullptr) {
        success[bx] = true;
      }
    }
  }
}

constexpr float eps = 1e-5;

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, typename DType, typename IdType>
__global__ void TopPSamplingFromProbKernel(DType* probs, DType* uniform_samples, IdType* output,
                                           bool* success, IdType* row_indices, float* top_p_arr,
                                           float top_p, uint32_t d, uint32_t max_top_p_rounds) {
  const uint32_t batch_size = gridDim.x;
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;

  if (top_p_arr != nullptr) {
    top_p = top_p_arr[bx];
  }
  const uint32_t row_idx = row_indices == nullptr ? bx : row_indices[bx];

  extern __shared__ __align__(alignof(
      SamplingTempStorage<DType, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>)) uint8_t smem[];
  auto& temp_storage = reinterpret_cast<
      SamplingTempStorage<DType, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(smem);

  vec_t<DType, VEC_SIZE> probs_vec;
  DType aggregate;
  DType q = DType(0);
  DType pivot = DType(0);
  IdType sampled_id;
  for (uint32_t round = 0; round < max_top_p_rounds; ++round) {
    temp_storage.data.sampled_id = d - 1;
    __syncthreads();
    DType u = uniform_samples[round * batch_size + bx] * (DType(1) - q);
    aggregate = DType(0);
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(DType(0));
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM, DType>(
          i, d, pivot, u, probs_vec, aggregate, &temp_storage);
      if (aggregate > u) {
        break;
      }
    }
    __syncthreads();
    sampled_id = temp_storage.data.sampled_id;
    pivot = probs[row_idx * d + sampled_id];

    DType aggregate_leq_pivot = DType(0);
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(DType(0));
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      DType probs_leq_pivot[VEC_SIZE];
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        probs_leq_pivot[j] = (probs_vec[j] <= pivot) ? probs_vec[j] : DType(0);
      }

      aggregate_leq_pivot += BlockReduce<DType, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                                 .Sum<VEC_SIZE>(probs_leq_pivot);
      if (tx == 0) {
        temp_storage.data.block_aggregate.value = aggregate_leq_pivot;
      }
      __syncthreads();
      if (float(temp_storage.data.block_aggregate.value) + top_p > 1 + eps) {
        break;
      }
    }
    q = temp_storage.data.block_aggregate.value;
    if (float(q) + top_p > 1 + eps) {
      break;
    }
  }
  __syncthreads();
  if (tx == 0) {
    if (float(q) + top_p <= 1 + eps) {
      // failed to sample within MAX_TOP_P_ROUNDS
      if (success != nullptr) {
        success[bx] = false;
      }
    } else {
      output[bx] = sampled_id;
      if (success != nullptr) {
        success[bx] = true;
      }
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
  IdType* row_indices_placeholder = nullptr;
  void* args[] = {&probs, &uniform_samples, &output, &row_indices_placeholder, &d};
  const uint32_t smem_size = sizeof(SamplingTempStorage<T, BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel =
        SamplingFromProbKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE, T, IdType>;
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  });
  return cudaSuccess;
}

template <typename T, typename IdType>
cudaError_t ParallelSamplingFromProb(T* probs, T* uniform_samples, IdType* output,
                                     IdType* row_indices, uint32_t batch_size, uint32_t d,
                                     cudaStream_t stream = 0) {
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);
  dim3 nblks(batch_size);
  dim3 nthrs(BLOCK_THREADS);
  void* args[] = {&probs, &uniform_samples, &output, &row_indices, &d};
  const uint32_t smem_size = sizeof(SamplingTempStorage<T, BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel =
        SamplingFromProbKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE, T, IdType>;
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  });
  return cudaSuccess;
}

template <typename T, typename IdType>
cudaError_t TopKSamplingFromProb(T* probs, T* uniform_samples, IdType* output, bool* success,
                                 IdType top_k, uint32_t batch_size, uint32_t d,
                                 uint32_t max_top_k_rounds, cudaStream_t stream = 0) {
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t smem_size = sizeof(SamplingTempStorage<T, BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
  dim3 nblks(batch_size);
  dim3 nthrs(BLOCK_THREADS);
  void* args[] = {&probs, &uniform_samples, &output, &success, &top_k, &d, &max_top_k_rounds};

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel =
        TopKSamplingFromProbKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE, T, IdType>;
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  });
  return cudaSuccess;
}

template <typename T, typename IdType>
cudaError_t TopPSamplingFromProb(T* probs, T* uniform_samples, IdType* output, bool* success,
                                 T top_p, uint32_t batch_size, uint32_t d,
                                 uint32_t max_top_p_rounds, cudaStream_t stream = 0) {
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t smem_size = sizeof(SamplingTempStorage<T, BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
  dim3 nblks(batch_size);
  dim3 nthrs(BLOCK_THREADS);
  IdType* row_indices_placeholder = nullptr;
  T* top_p_arr_placeholder = nullptr;
  void* args[] = {&probs,
                  &uniform_samples,
                  &output,
                  &success,
                  &row_indices_placeholder,
                  &top_p_arr_placeholder,
                  &top_p,
                  &d,
                  &max_top_p_rounds};

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel =
        TopPSamplingFromProbKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE, T, IdType>;
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  });
  return cudaSuccess;
}

template <typename T, typename IdType>
cudaError_t ParallelTopPSamplingFromProb(T* probs, T* uniform_samples, IdType* output,
                                         bool* success, IdType* row_indices, T* top_p_arr,
                                         uint32_t batch_size, uint32_t d, uint32_t max_top_p_rounds,
                                         cudaStream_t stream = 0) {
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t smem_size = sizeof(SamplingTempStorage<T, BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
  dim3 nblks(batch_size);
  dim3 nthrs(BLOCK_THREADS);
  T top_p_placeholder = 0;
  void* args[] = {&probs,     &uniform_samples,   &output, &success,         &row_indices,
                  &top_p_arr, &top_p_placeholder, &d,      &max_top_p_rounds};

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel =
        TopPSamplingFromProbKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE, T, IdType>;
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  });
  return cudaSuccess;
}

}  // namespace sampling

}  // namespace flashinfer

#endif  // FLASHINFER_SAMPLING_CUH_
