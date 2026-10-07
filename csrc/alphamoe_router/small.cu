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
#define SMEM_SHARED_COUNTS_OFF 0
#define SMEM_SHARED_COUNTS_STAGE_BYTES 4096
#define SMEM_SHARED_COUNTS_STRIDE 4096
#define SMEM_SHARED_OFFSETS_OFF 4096
#define SMEM_SHARED_OFFSETS_STAGE_BYTES 4100
#define SMEM_SHARED_OFFSETS_STRIDE 4100
#define SMEM_SCAN_VALUES_OFF 20488
#define SMEM_SCAN_VALUES_STAGE_BYTES 32
#define SMEM_SCAN_VALUES_STRIDE 32
#define SMEM_SHARED_SCATTER_OFF 8196
#define SMEM_SHARED_SCATTER_STAGE_BYTES 4096
#define SMEM_SHARED_SCATTER_STRIDE 4096
#define SMEM_SHARED_IDS_OFF 12292
#define SMEM_SHARED_IDS_STAGE_BYTES 8192
#define SMEM_SHARED_IDS_STRIDE 8192
#define SMEM_SHARED_TOTAL_OFF 20484
#define SMEM_SHARED_TOTAL_STAGE_BYTES 4
#define SMEM_SHARED_TOTAL_STRIDE 4
#define SMEM_TOTAL 20608
#define THREADS 256
#define NUM_WARPS 8
#define MAX_EXPERTS 512
#define MAX_TOP_K 16
#define MAX_BLOCK_M 16
#define PUBLIC_SHARED_SOFTMAX 1
#define FAST_BLOCK_ALIGN 0
#define SINGLE_CTA 0

#include <cooperative_groups.h>
#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float approx_rcp(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float max_noftz(float a, float b) {
  float c;
  asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
  return c;
}

extern "C" {

__global__ __launch_bounds__(256) void kernel_alpha_moe_fused_router_small(
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
  int* shared_counts = reinterpret_cast<int*>(smem_raw + 0);
  const int shared_counts_addr = smem + 0;
  int* shared_offsets = reinterpret_cast<int*>(smem_raw + 4096);
  const int shared_offsets_addr = smem + 4096;
  int* scan_values = reinterpret_cast<int*>(smem_raw + 20488);
  const int scan_values_addr = smem + 20488;
  int* shared_scatter = reinterpret_cast<int*>(smem_raw + 8196);
  const int shared_scatter_addr = smem + 8196;
  int* shared_ids = reinterpret_cast<int*>(smem_raw + 12292);
  const int shared_ids_addr = smem + 12292;
  int* shared_total = reinterpret_cast<int*>(smem_raw + 20484);
  const int shared_total_addr = smem + 20484;

  // === Task calls (dependency order) ===
  int routed_experts = E - has_shared_expert;
  int routed_top_k = top_k - has_shared_expert;
  for (int token = (unsigned int)(bid * NUM_WARPS) + warp; token < M;
       token += num_bids * NUM_WARPS) {
    unsigned long long row_base = (unsigned long long)token * (unsigned long long)E;
    float row_values[MAX_EXPERTS / 32];
#pragma unroll
    for (int expert_slot_load = 0; expert_slot_load < MAX_EXPERTS / 32; expert_slot_load++) {
      int expert_load = lane + (unsigned int)(expert_slot_load * 32);
      row_values[expert_slot_load] = -LOOM_INF;
      if (expert_load < routed_experts) {
        row_values[expert_slot_load] = logits[row_base + (unsigned long long)expert_load];
      }
    }
    unsigned long long output_base = (unsigned long long)token * (unsigned long long)top_k;
    float selected_logit = -LOOM_INF;
    int selected_expert = MAX_EXPERTS;
#pragma unroll 1
    for (int route = 0; route < routed_top_k; route++) {
      float local_max_0 = -LOOM_INF;
      float local_max_1 = -LOOM_INF;
      float local_max_2 = -LOOM_INF;
      float local_max_3 = -LOOM_INF;
#pragma unroll
      for (int scan_group = 0; scan_group < MAX_EXPERTS / 128; scan_group++) {
        float _fmax_0 = fmaxf(local_max_0, row_values[scan_group * 4]);
        local_max_0 = _fmax_0;
        float _fmax_1 = fmaxf(local_max_1, row_values[scan_group * 4 + 1]);
        local_max_1 = _fmax_1;
        float _fmax_2 = fmaxf(local_max_2, row_values[scan_group * 4 + 2]);
        local_max_2 = _fmax_2;
        float _fmax_3 = fmaxf(local_max_3, row_values[scan_group * 4 + 3]);
        local_max_3 = _fmax_3;
      }
      float _fmax_4 = fmaxf(local_max_0, local_max_1);
      float local_max_low = _fmax_4;
      float _fmax_5 = fmaxf(local_max_2, local_max_3);
      float local_max_high = _fmax_5;
      float _fmax_6 = fmaxf(local_max_low, local_max_high);
      float local_value = _fmax_6;
      unsigned int local_bits = 0;
      local_bits = reinterpret_cast<unsigned int*>(&local_value)[0];
      unsigned int local_key = local_bits ^ 2147483648;
      if ((local_bits & 2147483648) != 0) {
        local_key = local_bits ^ 4294967295;
      }
      unsigned int _warp_redux_u32_0;
      asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;"
                   : "=r"(_warp_redux_u32_0)
                   : "r"(local_key));
      unsigned int best_key = _warp_redux_u32_0;
      unsigned int best_bits = best_key ^ 2147483648;
      if ((best_key & 2147483648) == 0) {
        best_bits = best_key ^ 4294967295;
      }
      float best_value = 0.0f;
      best_value = reinterpret_cast<float*>(&best_bits)[0];
      unsigned int tied_index = MAX_EXPERTS;
      if (best_value != -LOOM_INF) {
#pragma unroll
        for (int expert_slot_finite_tie = 0; expert_slot_finite_tie < MAX_EXPERTS / 32;
             expert_slot_finite_tie++) {
          if (row_values[MAX_EXPERTS / 32 - 1 - expert_slot_finite_tie] == best_value) {
            int expert_finite_tie =
                lane + (unsigned int)((MAX_EXPERTS / 32 - 1 - expert_slot_finite_tie) * 32);
            tied_index = (unsigned int)expert_finite_tie;
          }
        }
      } else {
#pragma unroll
        for (int expert_slot_tie = 0; expert_slot_tie < MAX_EXPERTS / 32; expert_slot_tie++) {
          int expert_tie = lane + (unsigned int)((MAX_EXPERTS / 32 - 1 - expert_slot_tie) * 32);
          if (expert_tie < routed_experts) {
            if (row_values[MAX_EXPERTS / 32 - 1 - expert_slot_tie] == best_value) {
              tied_index = (unsigned int)expert_tie;
            }
          }
        }
      }
      unsigned int _warp_redux_u32_1;
      asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;"
                   : "=r"(_warp_redux_u32_1)
                   : "r"(tied_index));
      int best_index = (int)_warp_redux_u32_1;
      if (lane == (unsigned int)route) {
        selected_logit = best_value;
        selected_expert = best_index;
      }
#pragma unroll
      for (int expert_slot_remove = 0; expert_slot_remove < MAX_EXPERTS / 32;
           expert_slot_remove++) {
        if (lane + (unsigned int)(expert_slot_remove * 32) == (unsigned int)best_index) {
          row_values[expert_slot_remove] = -LOOM_INF;
        }
      }
    }
    if (has_shared_expert != 0) {
      if (lane == (unsigned int)routed_top_k) {
        selected_expert = E - 1;
        selected_logit = logits[row_base + (unsigned long long)selected_expert];
      }
    }
    int softmax_top_k = routed_top_k;
    {
      softmax_top_k = top_k;
    }
    float selected_for_max = -LOOM_INF;
    if ((unsigned int)softmax_top_k > lane) {
      selected_for_max = selected_logit;
    }
    float _warp_reduce_0 = selected_for_max;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      _warp_reduce_0 =
          max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
    float selected_max = _warp_reduce_0;
    float selected_exp = 0.0f;
    if ((unsigned int)softmax_top_k > lane) {
      float _exp2_0 = approx_exp2((selected_logit - selected_max) * 1.4426950408889634f);
      selected_exp = _exp2_0;
    }
    float _warp_reduce_1 = selected_exp;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
    float selected_sum = _warp_reduce_1;
    float _rcp_0 = approx_rcp(selected_sum);
    float selected_sum_rcp = _rcp_0;
    if (lane < (unsigned int)top_k) {
      float weight = selected_logit;
      if ((unsigned int)softmax_top_k > lane) {
        weight = selected_exp * selected_sum_rcp;
      }
      topk_weights[output_base + (unsigned long long)lane] = weight;
      topk_ids[output_base + (unsigned long long)lane] = selected_expert;
    }
  }
  {
    __threadfence();
    cooperative_groups::this_grid().sync();
  }
  if (bid == 0) {
#pragma unroll
    for (int expert_slot_zero = 0; expert_slot_zero < MAX_EXPERTS / THREADS; expert_slot_zero++) {
      int expert_zero = tid + expert_slot_zero * THREADS;
      if (expert_zero < E) {
        shared_counts[expert_zero] = 0;
        shared_scatter[expert_zero] = 0;
      }
    }
    __syncthreads();
    for (int count_pair = tid; count_pair < M * top_k; count_pair += THREADS) {
      int count_expert = topk_ids[count_pair];
      shared_ids[count_pair] = count_expert;
      atomicAdd(&shared_counts[count_expert], 1);
    }
    __syncthreads();
    int scan_padded[MAX_EXPERTS / THREADS];
    int scan_thread_total = 0;
#pragma unroll
    for (int expert_slot_scan_init = 0; expert_slot_scan_init < MAX_EXPERTS / THREADS;
         expert_slot_scan_init++) {
      int expert_scan_init = tid * (MAX_EXPERTS / THREADS) + expert_slot_scan_init;
      int padded_count_init = 0;
      if (expert_scan_init < E) {
        int count_init = shared_counts[expert_scan_init];
        {
          if (block_m == 8) {
            padded_count_init = count_init + 7 & -8;
          } else if (block_m == 16) {
            padded_count_init = count_init + 15 & -16;
          } else {
            padded_count_init = (count_init + block_m - 1) / block_m * block_m;
          }
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
        shared_total[0] = scan_warp_inclusive;
      }
    }
    __syncthreads();
    int scan_expert_prefix = scan_values[warp] + scan_thread_inclusive - scan_thread_total;
#pragma unroll
    for (int expert_slot_scan_store = 0; expert_slot_scan_store < MAX_EXPERTS / THREADS;
         expert_slot_scan_store++) {
      int expert_scan_store = tid * (MAX_EXPERTS / THREADS) + expert_slot_scan_store;
      if (expert_scan_store < E) {
        shared_offsets[expert_scan_store] = scan_expert_prefix;
      }
      scan_expert_prefix = scan_expert_prefix + scan_padded[expert_slot_scan_store];
    }
    if (tid == 0) {
      shared_offsets[E] = shared_total[0];
    }
    __syncthreads();
    for (int pair = tid; pair < M * top_k; pair += THREADS) {
      int pair_expert = shared_ids[pair];
      int _atomic_old_0 = atomicAdd(&shared_scatter[pair_expert], 1);
      int local_row = _atomic_old_0;
      int grouped_row = shared_offsets[pair_expert] + local_row;
      sorted_token_ids[grouped_row] = pair;
      {
        if (block_m == 8) {
          if ((local_row & 7) == 0) {
            expert_ids[grouped_row >> 3] = pair_expert;
          }
        } else if (block_m == 16) {
          if ((local_row & 15) == 0) {
            expert_ids[grouped_row >> 4] = pair_expert;
          }
        } else {
          if (local_row % block_m == 0) {
            expert_ids[grouped_row / block_m] = pair_expert;
          }
        }
      }
    }
    for (int padding_expert = tid; padding_expert < E; padding_expert += THREADS) {
      int count_final = shared_counts[padding_expert];
      int expert_start = shared_offsets[padding_expert];
      int padded_count_final = shared_offsets[padding_expert + 1] - expert_start;
      int padding_count = padded_count_final - count_final;
#pragma unroll
      for (int padding_slot = 0; padding_slot < MAX_BLOCK_M; padding_slot++) {
        if (padding_count > padding_slot) {
          sorted_token_ids[expert_start + count_final + padding_slot] = M * top_k;
        }
      }
    }
    __syncthreads();
    for (int publish_expert = tid; publish_expert < E; publish_expert += THREADS) {
      expert_counts[publish_expert] = shared_counts[publish_expert];
      expert_offsets[publish_expert] = shared_offsets[publish_expert];
      expert_scatter_offsets[publish_expert] = shared_scatter[publish_expert];
    }
    if (tid == 0) {
      expert_offsets[E] = shared_total[0];
      num_tokens_post_padded[0] = shared_total[0];
    }
  }
}

}  // extern "C"
