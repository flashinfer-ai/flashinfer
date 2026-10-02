// Copyright (c) 2026 FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
//
// Archived experimental engine, disabled by default because the 16x16 WMMA
// implementation does not provide sufficient FC1 throughput.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace flashinfer::sm90_push_bf16::fc1_fused {

constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 16;

struct WorkspaceView {
  int64_t* tile_prefix;
  int64_t* total_tasks;
};

inline size_t workspace_size(int num_experts) {
  return (static_cast<size_t>(num_experts) + 2) * sizeof(int64_t);
}

inline WorkspaceView bind_workspace(void* workspace, size_t bytes, int num_experts) {
  if (workspace == nullptr || bytes < workspace_size(num_experts)) return {};
  auto* base = static_cast<int64_t*>(workspace);
  return {base, base + num_experts + 1};
}

__global__ void prepare_fc1_schedule_kernel(int64_t const* offsets, int num_experts,
                                            int row_capacity, int n_tiles, int64_t* tile_prefix,
                                            int64_t* total_tasks) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  tile_prefix[0] = 0;
  int64_t previous = offsets[0];
  if (previous != 0) {
    printf("sm90_push_bf16_fc1_fused: offsets[0] must be zero\n");
    asm volatile("trap;");
  }
  for (int expert = 0; expert < num_experts; ++expert) {
    int64_t const next = offsets[expert + 1];
    if (next < previous || next > row_capacity) {
      printf("sm90_push_bf16_fc1_fused: invalid offsets for expert %d (%lld, %lld)\n", expert,
             static_cast<long long>(previous), static_cast<long long>(next));
      asm volatile("trap;");
    }
    int64_t const row_tiles = (next - previous + kTileM - 1) / kTileM;
    tile_prefix[expert + 1] = tile_prefix[expert] + row_tiles * n_tiles;
    previous = next;
  }
  *total_tasks = tile_prefix[num_experts];
}

__device__ __forceinline__ int find_expert(int64_t task, int64_t const* tile_prefix,
                                           int num_experts) {
  int low = 0;
  int high = num_experts;
  while (low + 1 < high) {
    int const middle = (low + high) >> 1;
    if (tile_prefix[middle] <= task) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return low;
}

__global__ void bf16_fc1_fused_kernel(__nv_bfloat16* output, __nv_bfloat16 const* activation,
                                      __nv_bfloat16 const* weights, int64_t const* offsets,
                                      int64_t const* tile_prefix, int64_t const* total_tasks,
                                      int num_experts, int intermediate_size, int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  int64_t const task = static_cast<int64_t>(blockIdx.x);
  if (task >= *total_tasks) return;

  int const expert = find_expert(task, tile_prefix, num_experts);
  int const n_tiles = intermediate_size / kTileN;
  int64_t const local_task = task - tile_prefix[expert];
  int const m_tile = static_cast<int>(local_task / n_tiles);
  int const n_tile = static_cast<int>(local_task % n_tiles);
  int64_t const expert_begin = offsets[expert];
  int64_t const expert_end = offsets[expert + 1];
  int const remaining_rows = static_cast<int>(expert_end - expert_begin - m_tile * kTileM);
  int const valid_rows = remaining_rows < kTileM ? remaining_rows : kTileM;
  int64_t const global_row = expert_begin + static_cast<int64_t>(m_tile) * kTileM;
  int const global_col = n_tile * kTileN;

  __shared__ __align__(32) __nv_bfloat16 activation_tile[kTileM * kTileK];
  __shared__ __align__(32) float output_tile[kTileM * kTileN];

  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a, kTileM, kTileN, kTileK, __nv_bfloat16, wmma::row_major> a_fragment;
  wmma::fragment<wmma::matrix_b, kTileM, kTileN, kTileK, __nv_bfloat16, wmma::col_major>
      gate_fragment;
  wmma::fragment<wmma::matrix_b, kTileM, kTileN, kTileK, __nv_bfloat16, wmma::col_major>
      up_fragment;
  wmma::fragment<wmma::accumulator, kTileM, kTileN, kTileK, float> gate_accumulator;
  wmma::fragment<wmma::accumulator, kTileM, kTileN, kTileK, float> up_accumulator;
  wmma::fill_fragment(gate_accumulator, 0.0f);
  wmma::fill_fragment(up_accumulator, 0.0f);

  int64_t const expert_weight_base = static_cast<int64_t>(expert) * 2 * intermediate_size * k;
  for (int k_base = 0; k_base < k; k_base += kTileK) {
    for (int element = threadIdx.x; element < kTileM * kTileK; element += blockDim.x) {
      int const row = element / kTileK;
      int const column = element % kTileK;
      activation_tile[element] = row < valid_rows
                                     ? activation[(global_row + row) * k + k_base + column]
                                     : __float2bfloat16_rn(0.0f);
    }
    __syncthreads();

    auto const* gate = weights + expert_weight_base + static_cast<int64_t>(global_col) * k + k_base;
    auto const* up = weights + expert_weight_base +
                     static_cast<int64_t>(intermediate_size + global_col) * k + k_base;
    wmma::load_matrix_sync(a_fragment, activation_tile, kTileK);
    wmma::load_matrix_sync(gate_fragment, gate, k);
    wmma::load_matrix_sync(up_fragment, up, k);
    wmma::mma_sync(gate_accumulator, a_fragment, gate_fragment, gate_accumulator);
    wmma::mma_sync(up_accumulator, a_fragment, up_fragment, up_accumulator);
    __syncthreads();
  }

#pragma unroll
  for (int element = 0; element < gate_accumulator.num_elements; ++element) {
    float const gate = gate_accumulator.x[element];
    gate_accumulator.x[element] = gate / (1.0f + expf(-gate)) * up_accumulator.x[element];
  }
  wmma::store_matrix_sync(output_tile, gate_accumulator, kTileN, wmma::mem_row_major);
  __syncthreads();

  for (int element = threadIdx.x; element < valid_rows * kTileN; element += blockDim.x) {
    int const row = element / kTileN;
    int const column = element % kTileN;
    output[(global_row + row) * intermediate_size + global_col + column] =
        __float2bfloat16_rn(output_tile[element]);
  }
#endif
}

__global__ void bf16_fc1_unfused_epilogue_kernel(__nv_bfloat16* output,
                                                 __nv_bfloat16 const* projected,
                                                 int64_t const* offsets, int num_experts,
                                                 int row_capacity, int intermediate_size) {
  int64_t const active_rows = offsets[num_experts];
  if (active_rows < 0 || active_rows > row_capacity) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      printf("sm90_push_bf16_fc1_fused: active rows exceed the epilogue capacity\n");
    }
    asm volatile("trap;");
    return;
  }
  int64_t const elements = active_rows * intermediate_size;
  for (int64_t element = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       element < elements; element += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t const row = element / intermediate_size;
    int const column = static_cast<int>(element % intermediate_size);
    int64_t const row_base = row * 2 * intermediate_size;
    float const gate = __bfloat162float(projected[row_base + column]);
    float const up = __bfloat162float(projected[row_base + intermediate_size + column]);
    output[element] = __float2bfloat16_rn(gate / (1.0f + expf(-gate)) * up);
  }
}

inline cudaError_t launch_fc1_fused(void* workspace, size_t workspace_bytes, int row_capacity,
                                    int num_experts, int intermediate_size, int k,
                                    __nv_bfloat16 const* activation, __nv_bfloat16 const* weights,
                                    __nv_bfloat16* output, int64_t const* offsets,
                                    cudaStream_t stream) {
  WorkspaceView const view = bind_workspace(workspace, workspace_bytes, num_experts);
  if (view.total_tasks == nullptr) return cudaErrorInvalidValue;
  int const n_tiles = intermediate_size / kTileN;
  prepare_fc1_schedule_kernel<<<1, 1, 0, stream>>>(offsets, num_experts, row_capacity, n_tiles,
                                                   view.tile_prefix, view.total_tasks);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;

  int64_t const max_row_tiles = (static_cast<int64_t>(row_capacity) + kTileM - 1) / kTileM;
  int64_t const max_tasks = (max_row_tiles + num_experts) * n_tiles;
  if (max_tasks == 0) return cudaSuccess;
  if (max_tasks > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    return cudaErrorInvalidConfiguration;
  }
  bf16_fc1_fused_kernel<<<static_cast<unsigned int>(max_tasks), 32, 0, stream>>>(
      output, activation, weights, offsets, view.tile_prefix, view.total_tasks, num_experts,
      intermediate_size, k);
  return cudaGetLastError();
}

inline cudaError_t launch_unfused_epilogue(__nv_bfloat16* output, __nv_bfloat16 const* projected,
                                           int64_t const* offsets, int num_experts,
                                           int row_capacity, int intermediate_size,
                                           cudaStream_t stream) {
  int64_t const capacity_elements = static_cast<int64_t>(row_capacity) * intermediate_size;
  if (capacity_elements == 0) return cudaSuccess;
  constexpr int kThreads = 256;
  int64_t const requested_blocks = (capacity_elements + kThreads - 1) / kThreads;
  int const blocks = static_cast<int>(requested_blocks < 4096 ? requested_blocks : 4096);
  bf16_fc1_unfused_epilogue_kernel<<<blocks, kThreads, 0, stream>>>(
      output, projected, offsets, num_experts, row_capacity, intermediate_size);
  return cudaGetLastError();
}

}  // namespace flashinfer::sm90_push_bf16::fc1_fused
