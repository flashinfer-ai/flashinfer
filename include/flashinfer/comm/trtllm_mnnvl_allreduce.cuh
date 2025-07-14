/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#include <iostream>

#include "../exception.h"
#include "../logging.h"
namespace flashinfer {
namespace trtllm_mnnvl_allreduce {

template <typename T>
struct AllReduceParams {
  int nranks;
  int rank;
  int buffer_M;
  int num_tokens;
  int token_dim;
  void** buffer_ptrs_dev;
  void* multicast_ptr;
  void* buffer_flags;
  bool wait_for_results;
  bool launch_with_pdl;

  void* input;
  void* output;
  cudaStream_t stream;
};

__device__ bool isNegZero(float v) { return v == 0.f && signbit(v); }

__device__ bool isNegZero(__nv_bfloat16 val) { return isNegZero(__bfloat162float(val)); }

template <typename T>
inline __device__ float toFloat(T val) {
  return val;
}

template <>
inline __device__ float toFloat<__nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}

template <typename T>
inline __device__ T fromFloat(float val) {
  return val;
}

template <>
inline __device__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float val) {
  return __float2bfloat16(val);
}

template <int WORLD_SIZE, typename T>
__global__ void twoshot_allreduce_kernel(T* output_ptr, T* shard_ptr, T** input_ptrs, T* mcast_ptr,
                                         int num_tokens, int buffer_M, int token_dim, int rank,
                                         uint32_t* buffer_flags, bool wait_for_results) {
  int elt = blockIdx.y * blockDim.x + threadIdx.x;

  if (elt >= token_dim) return;
  int token = blockIdx.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  uint32_t* offset_access_ptr = &buffer_flags[3];
  // Buffer size is M * N, and we need two buffers for reduce-scatter and allgather
  uint32_t buffer_size = (buffer_flags[2] << 1);
  uint32_t input_offset = buffer_flags[0] * buffer_size;
  uint32_t clear_offset = buffer_flags[1] * buffer_size;

  if (wait_for_results) {
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(offset_access_ptr, 1);
    }
  }

  if (elt < token_dim) {
    // Scatter token
    int dest_rank = token % WORLD_SIZE;
    int dest_token_offset = token / WORLD_SIZE;
    T val = shard_ptr[token * token_dim + elt];
    if (isNegZero(val)) val = fromFloat<T>(0.f);
    input_ptrs[dest_rank][input_offset + dest_token_offset * token_dim * WORLD_SIZE +
                          rank * token_dim + elt] = val;

    // Reduce and broadcast

    int global_token = token * WORLD_SIZE + rank;
    if (global_token < num_tokens) {
      float accum = 0.f;

      T values[WORLD_SIZE];

      for (int r = 0; r < WORLD_SIZE; r++) {
        input_ptrs[rank][clear_offset + token * token_dim * WORLD_SIZE + r * token_dim + elt] =
            fromFloat<T>(-0.f);
      }

      while (1) {
        bool valid = true;
        for (int r = 0; r < WORLD_SIZE; r++) {
          T volatile* lamport_ptr =
              (T volatile*)&input_ptrs[rank][input_offset + token * token_dim * WORLD_SIZE +
                                             r * token_dim + elt];
          values[r] = *lamport_ptr;
          valid &= !isNegZero(values[r]);
        }
        if (valid) break;
      }
      for (int r = 0; r < WORLD_SIZE; r++) {
        accum += toFloat<T>(values[r]);
      }
      mcast_ptr[input_offset + buffer_M * token_dim + global_token * token_dim + elt] =
          fromFloat<T>(accum);
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif

  input_ptrs[rank][clear_offset + buffer_M * token_dim + token * token_dim + elt] =
      fromFloat<T>(-0.f);

  // Optionally wait for results if the next layer isn't doing the Lamport check
  if (wait_for_results) {
    T volatile* lamport_ptr = (T volatile*)&input_ptrs[rank][input_offset + buffer_M * token_dim +
                                                             token * token_dim + elt];
    T val = *lamport_ptr;
    while (isNegZero(val)) val = *lamport_ptr;

    // Copy if requested
    if (output_ptr) output_ptr[token * token_dim + elt] = val;
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
      // Make sure all blocks have finished reading the offsets, 2-D grid
      while (*reinterpret_cast<uint32_t volatile*>(offset_access_ptr) < gridDim.x * gridDim.y) {
      }
      buffer_flags[0] = (buffer_flags[0] + 1) % 3;
      buffer_flags[1] = (buffer_flags[1] + 1) % 3;
      *(offset_access_ptr) = 0;
    }
  }
}

// Template-based dispatch functions following the same pattern as trtllm_allreduce.cuh
template <typename T, int WORLD_SIZE>
cudaError_t twoshot_allreduce_dispatch(AllReduceParams<T>& params) {
  int const num_threads = 128;
  int const num_blocks = (params.token_dim + num_threads - 1) / num_threads;

  dim3 grid(params.num_tokens, num_blocks);

  FLASHINFER_LOG_DEBUG(
      "[MNNVL TwoShot AllReduce] twoshot allreduce on rank %d, world_size: %d, buffer_M: %d, "
      "num_tokens: %d, token_dim: "
      "%d, wait_for_results: %d, launch_with_pdl: %d",
      params.rank, params.nranks, params.buffer_M, params.num_tokens, params.token_dim,
      params.wait_for_results, params.launch_with_pdl);

  cudaLaunchConfig_t config;
  cudaLaunchAttribute attrs[1];
  config.dynamicSmemBytes = 0;
  config.stream = params.stream;
  config.gridDim = grid;
  config.blockDim = num_threads;
  config.attrs = attrs;
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = params.launch_with_pdl ? 1 : 0;
  config.numAttrs = 1;

  cudaLaunchKernelEx(&config, &twoshot_allreduce_kernel<WORLD_SIZE, T>,
                     reinterpret_cast<T*>(params.output), reinterpret_cast<T*>(params.input),
                     reinterpret_cast<T**>(params.buffer_ptrs_dev),
                     reinterpret_cast<T*>(params.multicast_ptr), params.num_tokens, params.buffer_M,
                     params.token_dim, params.rank,
                     reinterpret_cast<uint32_t*>(params.buffer_flags), params.wait_for_results);

  return cudaSuccess;
}

template <typename T>
cudaError_t twoshot_allreduce_dispatch_world_size(AllReduceParams<T>& params) {
  FLASHINFER_LOG_DEBUG("twoshot_allreduce_dispatch_world_size");
  switch (params.nranks) {
    case 2:
      return twoshot_allreduce_dispatch<T, 2>(params);
    case 4:
      return twoshot_allreduce_dispatch<T, 4>(params);
    case 8:
      return twoshot_allreduce_dispatch<T, 8>(params);
    case 16:
      return twoshot_allreduce_dispatch<T, 16>(params);
    case 32:
      return twoshot_allreduce_dispatch<T, 32>(params);
    case 64:
      return twoshot_allreduce_dispatch<T, 64>(params);
    default:
      FLASHINFER_ERROR("MNNVL AllReduce: unsupported world_size " + std::to_string(params.nranks) +
                       ". Supported sizes: {2, 4, 8, 16, 32, 64}");
      return cudaErrorInvalidValue;
  }
}

}  // namespace trtllm_mnnvl_allreduce
}  // namespace flashinfer
