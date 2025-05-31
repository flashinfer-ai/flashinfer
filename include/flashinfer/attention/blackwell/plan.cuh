/*
 * Copyright (c) 2025 by FlashInfer team.
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
#include <cub/block/block_scan.cuh>

#include "../../utils.cuh"

namespace flashinfer {

__device__ __forceinline__ float cost_function(int qo_len, int kv_len) {
  return 0.6 * float(qo_len) + kv_len;
}

union CostIndex {
  struct {
    int bucket_idx;
    float cost;
  };
  long long packed;
};

__device__ __forceinline__ CostIndex get_min_cost_index(CostIndex* warp_min_cost,
                                                        CostIndex cost_index, int num_buckets) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    CostIndex other;
    other.packed = __shfl_xor_sync(0xffffffff, cost_index.packed, offset);
    if (other.cost < cost_index.cost ||
        (other.cost == cost_index.cost && other.bucket_idx < cost_index.bucket_idx)) {
      cost_index = other;
    }
  }
  if (threadIdx.x % 32 == 0) {
    warp_min_cost[threadIdx.x / 32] = cost_index;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    cost_index = threadIdx.x * 32 < num_buckets
                     ? warp_min_cost[threadIdx.x]
                     : CostIndex{threadIdx.x * 32, cuda::std::numeric_limits<float>::infinity()};
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      CostIndex other;
      other.packed = __shfl_xor_sync(0xffffffff, cost_index.packed, offset);
      if (other.cost < cost_index.cost ||
          (other.cost == cost_index.cost && other.bucket_idx < cost_index.bucket_idx)) {
        cost_index = other;
      }
    }
    if (threadIdx.x == 0) {
      warp_min_cost[0] = cost_index;
    }
  }
  __syncthreads();
  return warp_min_cost[0];
}

__global__ void plan_kernel(int* qo_lens, int* kv_lens, int* work_indptr, int* qo_tile_indices,
                            int* head_indices, int* batch_indices, int qo_tile_size, int batch_size,
                            int num_heads, int num_buckets) {
  __shared__ CostIndex warp_min_cost[32];
  constexpr int MAX_BUCKET_SIZE = 256;
  using BlockScan = cub::BlockScan<int, MAX_BUCKET_SIZE>;
  __shared__ typename BlockScan::TempStorage temp_storage;
  // first round, calculate the work count for each bucket
  CostIndex thread_local_cost_index = {threadIdx.x, 0.f};
  int thread_local_work_counter = 0;
  if (threadIdx.x >= num_buckets) {
    thread_local_cost_index.cost = cuda::std::numeric_limits<float>::infinity();
  }

  for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      for (int qo_tile_idx = 0; qo_tile_idx < ceil_div(qo_lens[batch_idx], qo_tile_size);
           ++qo_tile_idx) {
        auto min_cost_index =
            get_min_cost_index(warp_min_cost, thread_local_cost_index, num_buckets);
        int bucket_idx = min_cost_index.bucket_idx;
        float cost = min_cost_index.cost;
        if (bucket_idx == threadIdx.x) {
          thread_local_cost_index.cost += cost_function(qo_lens[batch_idx], kv_lens[batch_idx]);
          thread_local_work_counter++;
        }
      }
    }
  }
  // compute exclusive prefix sum of
  int thread_local_work_indptr = 0;
  BlockScan(temp_storage).InclusiveSum(thread_local_work_counter, thread_local_work_indptr);
  __syncthreads();
  if (threadIdx.x < num_buckets) {
    work_indptr[threadIdx.x] = thread_local_work_indptr;
  }
  if (threadIdx.x + 1 == num_buckets) {
    work_indptr[num_buckets] = thread_local_work_indptr + thread_local_work_counter;
  }

  // second round, write qo_tile_idx, head_idx, batch_idx to the output
  thread_local_work_counter = 0;
  if (threadIdx.x >= num_buckets) {
    thread_local_cost_index.cost = cuda::std::numeric_limits<float>::infinity();
  }
  for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      for (int qo_tile_idx = 0; qo_tile_idx < ceil_div(qo_lens[batch_idx], qo_tile_size);
           ++qo_tile_idx) {
        auto min_cost_index =
            get_min_cost_index(warp_min_cost, thread_local_cost_index, num_buckets);
        int bucket_idx = min_cost_index.bucket_idx;
        float cost = min_cost_index.cost;
        if (bucket_idx == threadIdx.x) {
          thread_local_cost_index.cost += cost_function(qo_lens[batch_idx], kv_lens[batch_idx]);
          qo_tile_indices[thread_local_work_indptr + thread_local_work_counter] = qo_tile_idx;
          head_indices[thread_local_work_indptr + thread_local_work_counter] = head_idx;
          batch_indices[thread_local_work_indptr + thread_local_work_counter] = batch_idx;
          thread_local_work_counter++;
        }
      }
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

cudaError_t plan_kernel_wrapper(int* qo_lens, int* kv_lens, int* work_indptr, int* qo_tile_indices,
                                int* head_indices, int* batch_indices, int qo_tile_size,
                                int batch_size, int num_heads, int num_buckets, bool enable_pdl,
                                cudaStream_t stream) {
  cudaLaunchConfig_t config;
  config.gridDim = 1;
  config.blockDim = 256;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, plan_kernel, qo_lens, kv_lens, work_indptr,
                                          qo_tile_indices, head_indices, batch_indices,
                                          qo_tile_size, batch_size, num_heads, num_buckets));
  return cudaSuccess;
}

}  // namespace flashinfer
