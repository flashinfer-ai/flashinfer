/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Requires an LP64 CUDA host ABI");
typedef signed int int32_t;
typedef short int int16_t;
struct __align__(128) LoomTensorMap {
  uint64_t opaque[16];
};
struct __align__(64) LoomTensorMap64 {
  uint64_t opaque[16];
};
static_assert(sizeof(LoomTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(LoomTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) {
  uint64_t opaque[16];
} CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(LoomTensorMap) >= alignof(CUtensorMap),
              "LoomTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
  int result;
  asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;" : "=r"(result) : "r"(x));
  return result;
}

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_LOCAL_OFFSETS_OFF 0
#define SMEM_LOCAL_OFFSETS_STAGE_BYTES 4100
#define SMEM_LOCAL_OFFSETS_STRIDE 4100
#define SMEM_SCAN_VALUES_OFF 4100
#define SMEM_SCAN_VALUES_STAGE_BYTES 32
#define SMEM_SCAN_VALUES_STRIDE 32
#define SMEM_TOTAL 4224
#define THREADS 256
#define NUM_WARPS 8
#define MAX_EXPERTS 512
#define MAX_TOP_K 16
#define MAX_BLOCK_M 16
#define PUBLIC_SHARED_SOFTMAX 0

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void kernel_alpha_moe_fused_router_large_tail(
    float* __restrict__ logits, float* __restrict__ topk_weights, int* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids,
    int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts,
    int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M, int E,
    int top_k, int block_m, int has_shared_expert) {
  const int tid = threadIdx.x;
  const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
  uint32_t lane;
  asm("mov.u32 %0, %%laneid;" : "=r"(lane));

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  // Kernel setup ops
  int* local_offsets = reinterpret_cast<int*>(smem_raw + 0);
  const int local_offsets_addr = smem + 0;
  int* scan_values = reinterpret_cast<int*>(smem_raw + 4100);
  const int scan_values_addr = smem + 4100;

  // === Task calls (dependency order) ===
  int global_thread = bid * THREADS + tid;
  if (bid * THREADS < M * top_k || bid * THREADS < E) {
    int first_stored_pair = 0;
    {
      if (global_thread < M * top_k) {
        first_stored_pair = topk_ids[global_thread];
      }
    }
    int scan_padded[MAX_EXPERTS / THREADS];
    int scan_thread_total = 0;
#pragma unroll
    for (int expert_slot_scan_init = 0; expert_slot_scan_init < MAX_EXPERTS / THREADS;
         expert_slot_scan_init++) {
      int expert_scan_init = tid * (MAX_EXPERTS / THREADS) + expert_slot_scan_init;
      int padded_count_init = 0;
      if (expert_scan_init < E) {
        int count_init = expert_counts[expert_scan_init];
        if (block_m == 8) {
          padded_count_init = count_init + 7 & -8;
        } else if (block_m == 16) {
          padded_count_init = count_init + 15 & -16;
        } else {
          padded_count_init = (count_init + block_m - 1) / block_m * block_m;
        }
      }
      scan_padded[expert_slot_scan_init] = padded_count_init;
      scan_thread_total = scan_thread_total + padded_count_init;
    }
    int scan_thread_inclusive = scan_thread_total;
    int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, scan_thread_inclusive, 1, 32);
    int scan_peer = _shfl_up_0;
    if (lane >= 1) {
      scan_thread_inclusive = scan_thread_inclusive + scan_peer;
    }
    int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, scan_thread_inclusive, 2, 32);
    int scan_peer_0 = _shfl_up_1;
    if (lane >= 2) {
      scan_thread_inclusive = scan_thread_inclusive + scan_peer_0;
    }
    int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, scan_thread_inclusive, 4, 32);
    int scan_peer_1 = _shfl_up_2;
    if (lane >= 4) {
      scan_thread_inclusive = scan_thread_inclusive + scan_peer_1;
    }
    int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, scan_thread_inclusive, 8, 32);
    int scan_peer_2 = _shfl_up_3;
    if (lane >= 8) {
      scan_thread_inclusive = scan_thread_inclusive + scan_peer_2;
    }
    int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, scan_thread_inclusive, 16, 32);
    int scan_peer_3 = _shfl_up_4;
    if (lane >= 16) {
      scan_thread_inclusive = scan_thread_inclusive + scan_peer_3;
    }
    if (lane == 31) {
      scan_values[warp] = scan_thread_inclusive;
    }
    __syncthreads();
    if (warp == 0) {
      int scan_warp_total = 0;
      if (lane < (unsigned int)NUM_WARPS) {
        scan_warp_total = scan_values[lane];
      }
      int scan_warp_inclusive = scan_warp_total;
      int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, scan_warp_inclusive, 1, 32);
      int scan_warp_peer = _shfl_up_5;
      if (lane >= 1) {
        scan_warp_inclusive = scan_warp_inclusive + scan_warp_peer;
      }
      int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, scan_warp_inclusive, 2, 32);
      int scan_warp_peer_0 = _shfl_up_6;
      if (lane >= 2) {
        scan_warp_inclusive = scan_warp_inclusive + scan_warp_peer_0;
      }
      int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, scan_warp_inclusive, 4, 32);
      int scan_warp_peer_1 = _shfl_up_7;
      if (lane >= 4) {
        scan_warp_inclusive = scan_warp_inclusive + scan_warp_peer_1;
      }
      if (lane < (unsigned int)NUM_WARPS) {
        scan_values[lane] = scan_warp_inclusive - scan_warp_total;
      }
      if (lane == (unsigned int)(NUM_WARPS - 1)) {
        local_offsets[E] = scan_warp_inclusive;
        if (bid == 0) {
          num_tokens_post_padded[0] = scan_warp_inclusive;
          expert_offsets[E] = scan_warp_inclusive;
        }
      }
    }
    __syncthreads();
    int scan_expert_prefix = scan_values[warp] + scan_thread_inclusive - scan_thread_total;
#pragma unroll
    for (int expert_slot_scan_store = 0; expert_slot_scan_store < MAX_EXPERTS / THREADS;
         expert_slot_scan_store++) {
      int expert_scan_store = tid * (MAX_EXPERTS / THREADS) + expert_slot_scan_store;
      if (expert_scan_store < E) {
        local_offsets[expert_scan_store] = scan_expert_prefix;
        if (bid == 0) {
          expert_offsets[expert_scan_store] = scan_expert_prefix;
          if (M > 512 && M <= 2147483647 / (MAX_EXPERTS * MAX_TOP_K)) {
            expert_scatter_offsets[expert_scan_store] = expert_counts[expert_scan_store];
          }
        }
      }
      scan_expert_prefix = scan_expert_prefix + scan_padded[expert_slot_scan_store];
    }
    __syncthreads();
    int scatter_block_mask = block_m - 1;
    int scatter_block_shift = 0;
    if (block_m == 8) {
      scatter_block_shift = 3;
    } else if (block_m == 16) {
      scatter_block_shift = 4;
    }
    for (int pair = global_thread; pair < M * top_k; pair += num_bids * THREADS) {
      int stored_pair = 0;
      {
        stored_pair = first_stored_pair;
        if (pair != global_thread) {
          stored_pair = topk_ids[pair];
        }
      }
      int pair_expert = stored_pair;
      int local_row = 0;
      if (M > 512 && M <= 2147483647 / (MAX_EXPERTS * MAX_TOP_K)) {
        pair_expert = stored_pair % MAX_EXPERTS;
        local_row = stored_pair / MAX_EXPERTS;
        topk_ids[pair] = pair_expert;
      } else {
        int _atomic_old_0 = atomicAdd(&expert_scatter_offsets[pair_expert], 1);
        local_row = _atomic_old_0;
      }
      int grouped_row = local_offsets[pair_expert] + local_row;
      sorted_token_ids[grouped_row] = pair;
      int scatter_block_remainder = 0;
      if (scatter_block_shift != 0) {
        scatter_block_remainder = local_row & scatter_block_mask;
      } else {
        scatter_block_remainder = local_row % block_m;
      }
      if (scatter_block_remainder == 0) {
        int scatter_block_index = 0;
        if (scatter_block_shift != 0) {
          scatter_block_index = grouped_row >> scatter_block_shift;
        } else {
          scatter_block_index = grouped_row / block_m;
        }
        expert_ids[scatter_block_index] = pair_expert;
      }
    }
    for (int padding_expert = global_thread; padding_expert < E;
         padding_expert += num_bids * THREADS) {
      int count_final = expert_counts[padding_expert];
      int expert_start = local_offsets[padding_expert];
      int padded_count_final = local_offsets[padding_expert + 1] - expert_start;
      int padding_count = padded_count_final - count_final;
#pragma unroll
      for (int padding_slot = 0; padding_slot < MAX_BLOCK_M; padding_slot++) {
        if (padding_count > padding_slot) {
          sorted_token_ids[expert_start + count_final + padding_slot] = M * top_k;
        }
      }
    }
  }
}

}  // extern "C"
