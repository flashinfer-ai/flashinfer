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

__global__ __launch_bounds__(256) void kernel_alpha_moe_fused_router(
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
  int use_cta_reservations = 0;
  if (MAX_EXPERTS == 512 && PUBLIC_SHARED_SOFTMAX == 0 && E == 512 && has_shared_expert == 0 &&
      M > 512 && M <= 2147483647 / (MAX_EXPERTS * MAX_TOP_K)) {
    use_cta_reservations = 1;
  }
  if (bid == 0) {
#pragma unroll
    for (int expert_slot_zero = 0; expert_slot_zero < MAX_EXPERTS / THREADS; expert_slot_zero++) {
      int expert_zero = tid + expert_slot_zero * THREADS;
      if (expert_zero < E) {
        expert_counts[expert_zero] = 0;
        expert_offsets[expert_zero] = 0;
        expert_scatter_offsets[expert_zero] = 0;
      }
    }
    if (tid == 0) {
      expert_offsets[E] = 0;
      num_tokens_post_padded[0] = 0;
    }
  }
  if (use_cta_reservations != 0) {
#pragma unroll
    for (int expert_slot_local_zero = 0; expert_slot_local_zero < MAX_EXPERTS / THREADS;
         expert_slot_local_zero++) {
      int expert_local_zero = tid + expert_slot_local_zero * THREADS;
      local_offsets[expert_local_zero] = 0;
    }
  }
  cooperative_groups::this_grid().sync();
  int routed_experts = E - has_shared_expert;
  int routed_top_k = top_k - has_shared_expert;
  for (int token = (unsigned int)(bid * NUM_WARPS) + warp; token < M;
       token += num_bids * NUM_WARPS) {
    unsigned long long row_base = (unsigned long long)token * (unsigned long long)E;
    float row_values[MAX_EXPERTS / 32];
    if (MAX_EXPERTS == 512 && PUBLIC_SHARED_SOFTMAX == 0 && E == 512 && has_shared_expert == 0) {
      unsigned long long lane_row_base = row_base + (unsigned long long)lane;
#pragma unroll
      for (int expert_slot_full = 0; expert_slot_full < MAX_EXPERTS / 32; expert_slot_full++) {
        row_values[expert_slot_full] =
            logits[lane_row_base + (unsigned long long)(expert_slot_full * 32)];
      }
    } else {
#pragma unroll
      for (int expert_slot_load = 0; expert_slot_load < MAX_EXPERTS / 32; expert_slot_load++) {
        int expert_load = lane + (unsigned int)(expert_slot_load * 32);
        row_values[expert_slot_load] = -LOOM_INF;
        if (expert_load < routed_experts) {
          row_values[expert_slot_load] = logits[row_base + (unsigned long long)expert_load];
        }
      }
    }
    unsigned long long output_base = (unsigned long long)token * (unsigned long long)top_k;
    float selected_logit = -LOOM_INF;
    int selected_expert = MAX_EXPERTS;
#pragma unroll 1
    for (int route = 0; route < routed_top_k; route++) {
      float local_value = 0.0f;
      {
        float _fmax_0 = fmaxf(row_values[0], row_values[1]);
        float _fmax_1 = fmaxf(_fmax_0, row_values[2]);
        float local_max_t0 = _fmax_1;
        float _fmax_2 = fmaxf(row_values[3], row_values[4]);
        float _fmax_3 = fmaxf(_fmax_2, row_values[5]);
        float local_max_t1 = _fmax_3;
        float _fmax_4 = fmaxf(row_values[6], row_values[7]);
        float _fmax_5 = fmaxf(_fmax_4, row_values[8]);
        float local_max_t2 = _fmax_5;
        float _fmax_6 = fmaxf(row_values[9], row_values[10]);
        float _fmax_7 = fmaxf(_fmax_6, row_values[11]);
        float local_max_t3 = _fmax_7;
        float _fmax_8 = fmaxf(row_values[12], row_values[13]);
        float _fmax_9 = fmaxf(_fmax_8, row_values[14]);
        float local_max_t4 = _fmax_9;
        float _fmax_10 = fmaxf(local_max_t0, local_max_t1);
        float _fmax_11 = fmaxf(_fmax_10, local_max_t2);
        float local_max_u = _fmax_11;
        float _fmax_12 = fmaxf(local_max_t3, local_max_t4);
        float _fmax_13 = fmaxf(_fmax_12, row_values[15]);
        float local_max_v = _fmax_13;
        float _fmax_14 = fmaxf(local_max_u, local_max_v);
        float _fmax_15 = fmaxf(_fmax_14, -LOOM_INF);
        local_value = _fmax_15;
      }
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
      int descending_tie_key = 0;
      {
        unsigned int bounded_tie_masks[4];
#pragma unroll
        for (int bounded_mask_init = 0; bounded_mask_init < 4; bounded_mask_init++) {
          bounded_tie_masks[bounded_mask_init] = 0;
        }
#pragma unroll
        for (int bounded_mask_slot = 0; bounded_mask_slot < MAX_EXPERTS / 128;
             bounded_mask_slot++) {
#pragma unroll
          for (int bounded_mask_group = 0; bounded_mask_group < 4; bounded_mask_group++) {
            if (row_values[bounded_mask_group * (MAX_EXPERTS / 128) + bounded_mask_slot] ==
                best_value) {
              bounded_tie_masks[bounded_mask_group] =
                  bounded_tie_masks[bounded_mask_group] |
                  (unsigned int)(2147483648 >>
                                 bounded_mask_group * (MAX_EXPERTS / 128) + bounded_mask_slot);
            }
          }
        }
        unsigned int bounded_mask_low = bounded_tie_masks[0] | bounded_tie_masks[1];
        unsigned int bounded_mask_high = bounded_tie_masks[2] | bounded_tie_masks[3];
        unsigned int bounded_mask_all = bounded_mask_low | bounded_mask_high;
        int _find_msb_0;
        asm volatile("bfind.u32 %0, %1;" : "=r"(_find_msb_0) : "r"(bounded_mask_all));
        int bounded_high_bit = _find_msb_0;
        descending_tie_key = bounded_high_bit * 32 - (int)lane;
      }
      int best_index = 0;
      {
        int _warp_redux_i32_0;
        asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;"
                     : "=r"(_warp_redux_i32_0)
                     : "r"(descending_tie_key));
        int best_tie_key = _warp_redux_i32_0;
        best_index = 992 - best_tie_key;
      }
      {
        best_index = ((best_index < routed_experts) ? best_index : MAX_EXPERTS);
      }
      if (lane == (unsigned int)route) {
        selected_logit = best_value;
        selected_expert = best_index;
      }
#pragma unroll
      for (int remove_group = 0; remove_group < MAX_EXPERTS / 256; remove_group++) {
        if (best_index >= remove_group * 256 && best_index < (remove_group + 1) * 256) {
          if (best_index < remove_group * 256 + 128) {
            if (best_index < remove_group * 256 + 64) {
              if (best_index < remove_group * 256 + 32) {
                if (lane + (unsigned int)(remove_group * 8 * 32) == (unsigned int)best_index) {
                  row_values[remove_group * 8] = -LOOM_INF;
                }
              } else if (lane + (unsigned int)((remove_group * 8 + 1) * 32) ==
                         (unsigned int)best_index) {
                row_values[remove_group * 8 + 1] = -LOOM_INF;
              }
            } else if (best_index < remove_group * 256 + 96) {
              if (lane + (unsigned int)((remove_group * 8 + 2) * 32) == (unsigned int)best_index) {
                row_values[remove_group * 8 + 2] = -LOOM_INF;
              }
            } else {
              if (lane + (unsigned int)((remove_group * 8 + 3) * 32) == (unsigned int)best_index) {
                row_values[remove_group * 8 + 3] = -LOOM_INF;
              }
            }
          } else if (best_index < remove_group * 256 + 192) {
            if (best_index < remove_group * 256 + 160) {
              if (lane + (unsigned int)((remove_group * 8 + 4) * 32) == (unsigned int)best_index) {
                row_values[remove_group * 8 + 4] = -LOOM_INF;
              }
            } else if (lane + (unsigned int)((remove_group * 8 + 5) * 32) ==
                       (unsigned int)best_index) {
              row_values[remove_group * 8 + 5] = -LOOM_INF;
            }
          } else {
            if (best_index < remove_group * 256 + 224) {
              if (lane + (unsigned int)((remove_group * 8 + 6) * 32) == (unsigned int)best_index) {
                row_values[remove_group * 8 + 6] = -LOOM_INF;
              }
            } else if (lane + (unsigned int)((remove_group * 8 + 7) * 32) ==
                       (unsigned int)best_index) {
              row_values[remove_group * 8 + 7] = -LOOM_INF;
            }
          }
        }
      }
    }
    if (has_shared_expert != 0) {
      if (lane == (unsigned int)routed_top_k) {
        selected_expert = E - 1;
        selected_logit = logits[row_base + (unsigned long long)selected_expert];
      }
    }
    int count_ordinal = 0;
    if (M > 512 && M <= 2147483647 / (MAX_EXPERTS * MAX_TOP_K)) {
      if (lane < (unsigned int)top_k) {
        if (use_cta_reservations != 0) {
          int _atomic_old_0 = atomicAdd(&local_offsets[selected_expert], 1);
          count_ordinal = _atomic_old_0;
        } else {
          int _atomic_old_1 = atomicAdd(&expert_counts[selected_expert], 1);
          count_ordinal = _atomic_old_1;
        }
      }
    }
    int softmax_top_k = routed_top_k;
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
      int stored_expert = selected_expert;
      if (M > 512 && M <= 2147483647 / (MAX_EXPERTS * MAX_TOP_K)) {
        stored_expert = count_ordinal * MAX_EXPERTS + selected_expert;
      } else {
        atomicAdd(&expert_counts[selected_expert], 1);
      }
      topk_ids[output_base + (unsigned long long)lane] = stored_expert;
    }
  }
  if (use_cta_reservations != 0) {
    int first_rebase_token = (unsigned int)(bid * NUM_WARPS) + warp;
    int first_stored_expert = 0;
    if (first_rebase_token < M) {
      if (lane < (unsigned int)top_k) {
        unsigned long long first_rebase_pair =
            (unsigned long long)first_rebase_token * (unsigned long long)top_k +
            (unsigned long long)lane;
        first_stored_expert = topk_ids[first_rebase_pair];
      }
    }
    __syncthreads();
#pragma unroll
    for (int expert_slot_reserve = 0; expert_slot_reserve < MAX_EXPERTS / THREADS;
         expert_slot_reserve++) {
      int expert_reserve = tid + expert_slot_reserve * THREADS;
      int count_reserve = local_offsets[expert_reserve];
      if (count_reserve > 0) {
        int _atomic_old_2 = atomicAdd(&expert_counts[expert_reserve], count_reserve);
        int base_reserve = _atomic_old_2;
        local_offsets[expert_reserve] = base_reserve;
      }
    }
    __syncthreads();
    for (int token_rebase = (unsigned int)(bid * NUM_WARPS) + warp; token_rebase < M;
         token_rebase += num_bids * NUM_WARPS) {
      if (lane < (unsigned int)top_k) {
        unsigned long long pair_rebase =
            (unsigned long long)token_rebase * (unsigned long long)top_k + (unsigned long long)lane;
        int stored_rebase = first_stored_expert;
        if (token_rebase != first_rebase_token) {
          stored_rebase = topk_ids[pair_rebase];
        }
        int expert_rebase = stored_rebase & MAX_EXPERTS - 1;
        topk_ids[pair_rebase] = stored_rebase + local_offsets[expert_rebase] * MAX_EXPERTS;
      }
    }
  }
}

}  // extern "C"
