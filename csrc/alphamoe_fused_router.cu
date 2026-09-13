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

// Generated router entries with the retained TVM-FFI semantic wrapper.
// Host dispatch retains each source entry's cooperative-launch policy.
// Routed-only calls use their exact source specialization; shared calls retain
// selected shared-logit softmax. Only entry identifiers are renamed; generated
// function bodies stay unchanged, with isolated helpers and macros.

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <mutex>
#include <unordered_map>

#include "tvm_ffi_utils.h"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1000 && __CUDA_ARCH__ != 1030
#error "AlphaMoE fused router is supported only on SM100a and SM103a"
#endif

#include <cuda_fp8.h>
// clang-format off

namespace alphamoe_router_large_generated {
typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) LoomTensorMap { uint64_t opaque[16]; };
struct __align__(64) LoomTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(LoomTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(LoomTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else

#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(LoomTensorMap) >= alignof(CUtensorMap), "LoomTensorMap alignment must cover the CUtensorMap CUDA ABI");


__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SCAN_VALUES_OFF 0
#define SMEM_SCAN_VALUES_STAGE_BYTES 4096
#define SMEM_SCAN_VALUES_STRIDE 4096
#define SMEM_TOTAL 4096
#define THREADS 256
#define NUM_WARPS 8
#define MAX_EXPERTS 512
#define MAX_TOP_K 16
#define MAX_BLOCK_M 16
#define PUBLIC_SHARED_SOFTMAX 1


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

__global__ __launch_bounds__(256) void
kernel_alpha_moe_fused_router(float* __restrict__ logits, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M, int E, int top_k, int block_m, int has_shared_expert)
{
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
    int* scan_values = reinterpret_cast<int*>(smem_raw + 0);
    const int scan_values_addr = smem + 0;

    // === Task calls (dependency order) ===
    int global_thread = bid * THREADS + tid;
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
    __threadfence();
    cooperative_groups::this_grid().sync();
    int routed_experts = E - has_shared_expert;
    int routed_top_k = top_k - has_shared_expert;
    for (int token = (unsigned int)(bid * NUM_WARPS) + warp; token < M; token += num_bids * NUM_WARPS) {
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
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(local_key));
            unsigned int best_key = _warp_redux_u32_0;
            unsigned int best_bits = best_key ^ 2147483648;
            if ((best_key & 2147483648) == 0) {
                best_bits = best_key ^ 4294967295;
            }
            float best_value = 0.0f;
            best_value = reinterpret_cast<float*>(&best_bits)[0];
            unsigned int tied_index = MAX_EXPERTS;
            #pragma unroll
            for (int expert_slot_tie = 0; expert_slot_tie < MAX_EXPERTS / 32; expert_slot_tie++) {
                int expert_tie = lane + (unsigned int)((MAX_EXPERTS / 32 - 1 - expert_slot_tie) * 32);
                if (expert_tie < routed_experts) {
                    if (row_values[MAX_EXPERTS / 32 - 1 - expert_slot_tie] == best_value) {
                        tied_index = (unsigned int)expert_tie;
                    }
                }
            }
            unsigned int _warp_redux_u32_1;
            asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_1) : "r"(tied_index));
            int best_index = (int)_warp_redux_u32_1;
            if (lane == (unsigned int)route) {
                selected_logit = best_value;
                selected_expert = best_index;
            }
            #pragma unroll
            for (int expert_slot_remove = 0; expert_slot_remove < MAX_EXPERTS / 32; expert_slot_remove++) {
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
            _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
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
            atomicAdd(&expert_counts[selected_expert], 1);
        }
    }
    __threadfence();
    cooperative_groups::this_grid().sync();
    if (bid == 0) {
        #pragma unroll
        for (int expert_slot_scan_init = 0; expert_slot_scan_init < MAX_EXPERTS / THREADS; expert_slot_scan_init++) {
            int expert_scan_init = tid + expert_slot_scan_init * THREADS;
            int padded_count_init = 0;
            if (expert_scan_init < E) {
                int count_init = expert_counts[expert_scan_init];
                padded_count_init = (count_init + block_m - 1) / block_m * block_m;
            }
            scan_values[expert_scan_init] = padded_count_init;
        }
        __syncthreads();
        int scan_index_up = (tid + 1) * 2 - 1;
        if (scan_index_up < MAX_EXPERTS) {
            scan_values[scan_index_up] = scan_values[scan_index_up] + scan_values[scan_index_up - 1];
        }
        int scan_index_up_0 = (THREADS + tid + 1) * 2 - 1;
        if (scan_index_up_0 < MAX_EXPERTS) {
            scan_values[scan_index_up_0] = scan_values[scan_index_up_0] + scan_values[scan_index_up_0 - 1];
        }
        __syncthreads();
        int scan_index_up_1 = (tid + 1) * 4 - 1;
        if (scan_index_up_1 < MAX_EXPERTS) {
            scan_values[scan_index_up_1] = scan_values[scan_index_up_1] + scan_values[scan_index_up_1 - 2];
        }
        int scan_index_up_2 = (THREADS + tid + 1) * 4 - 1;
        if (scan_index_up_2 < MAX_EXPERTS) {
            scan_values[scan_index_up_2] = scan_values[scan_index_up_2] + scan_values[scan_index_up_2 - 2];
        }
        __syncthreads();
        int scan_index_up_3 = (tid + 1) * 8 - 1;
        if (scan_index_up_3 < MAX_EXPERTS) {
            scan_values[scan_index_up_3] = scan_values[scan_index_up_3] + scan_values[scan_index_up_3 - 4];
        }
        int scan_index_up_4 = (THREADS + tid + 1) * 8 - 1;
        if (scan_index_up_4 < MAX_EXPERTS) {
            scan_values[scan_index_up_4] = scan_values[scan_index_up_4] + scan_values[scan_index_up_4 - 4];
        }
        __syncthreads();
        int scan_index_up_5 = (tid + 1) * 16 - 1;
        if (scan_index_up_5 < MAX_EXPERTS) {
            scan_values[scan_index_up_5] = scan_values[scan_index_up_5] + scan_values[scan_index_up_5 - 8];
        }
        int scan_index_up_6 = (THREADS + tid + 1) * 16 - 1;
        if (scan_index_up_6 < MAX_EXPERTS) {
            scan_values[scan_index_up_6] = scan_values[scan_index_up_6] + scan_values[scan_index_up_6 - 8];
        }
        __syncthreads();
        int scan_index_up_7 = (tid + 1) * 32 - 1;
        if (scan_index_up_7 < MAX_EXPERTS) {
            scan_values[scan_index_up_7] = scan_values[scan_index_up_7] + scan_values[scan_index_up_7 - 16];
        }
        int scan_index_up_8 = (THREADS + tid + 1) * 32 - 1;
        if (scan_index_up_8 < MAX_EXPERTS) {
            scan_values[scan_index_up_8] = scan_values[scan_index_up_8] + scan_values[scan_index_up_8 - 16];
        }
        __syncthreads();
        int scan_index_up_9 = (tid + 1) * 64 - 1;
        if (scan_index_up_9 < MAX_EXPERTS) {
            scan_values[scan_index_up_9] = scan_values[scan_index_up_9] + scan_values[scan_index_up_9 - 32];
        }
        int scan_index_up_10 = (THREADS + tid + 1) * 64 - 1;
        if (scan_index_up_10 < MAX_EXPERTS) {
            scan_values[scan_index_up_10] = scan_values[scan_index_up_10] + scan_values[scan_index_up_10 - 32];
        }
        __syncthreads();
        int scan_index_up_11 = (tid + 1) * 128 - 1;
        if (scan_index_up_11 < MAX_EXPERTS) {
            scan_values[scan_index_up_11] = scan_values[scan_index_up_11] + scan_values[scan_index_up_11 - 64];
        }
        int scan_index_up_12 = (THREADS + tid + 1) * 128 - 1;
        if (scan_index_up_12 < MAX_EXPERTS) {
            scan_values[scan_index_up_12] = scan_values[scan_index_up_12] + scan_values[scan_index_up_12 - 64];
        }
        __syncthreads();
        int scan_index_up_13 = (tid + 1) * 256 - 1;
        if (scan_index_up_13 < MAX_EXPERTS) {
            scan_values[scan_index_up_13] = scan_values[scan_index_up_13] + scan_values[scan_index_up_13 - 128];
        }
        int scan_index_up_14 = (THREADS + tid + 1) * 256 - 1;
        if (scan_index_up_14 < MAX_EXPERTS) {
            scan_values[scan_index_up_14] = scan_values[scan_index_up_14] + scan_values[scan_index_up_14 - 128];
        }
        __syncthreads();
        int scan_index_up_15 = (tid + 1) * 512 - 1;
        if (scan_index_up_15 < MAX_EXPERTS) {
            scan_values[scan_index_up_15] = scan_values[scan_index_up_15] + scan_values[scan_index_up_15 - 256];
        }
        int scan_index_up_16 = (THREADS + tid + 1) * 512 - 1;
        if (scan_index_up_16 < MAX_EXPERTS) {
            scan_values[scan_index_up_16] = scan_values[scan_index_up_16] + scan_values[scan_index_up_16 - 256];
        }
        __syncthreads();
        int scan_index_up_17 = (tid + 1) * 1024 - 1;
        if (scan_index_up_17 < MAX_EXPERTS) {
            scan_values[scan_index_up_17] = scan_values[scan_index_up_17] + scan_values[scan_index_up_17 - 512];
        }
        int scan_index_up_18 = (THREADS + tid + 1) * 1024 - 1;
        if (scan_index_up_18 < MAX_EXPERTS) {
            scan_values[scan_index_up_18] = scan_values[scan_index_up_18] + scan_values[scan_index_up_18 - 512];
        }
        __syncthreads();
        if (tid == 0) {
            int padded_total = scan_values[MAX_EXPERTS - 1];
            num_tokens_post_padded[0] = padded_total;
            scan_values[MAX_EXPERTS - 1] = 0;
        }
        __syncthreads();
        int scan_index_down = (tid + 1) * 1024 - 1;
        if (scan_index_down < MAX_EXPERTS) {
            int scan_left = scan_values[scan_index_down - 512];
            scan_values[scan_index_down - 512] = scan_values[scan_index_down];
            scan_values[scan_index_down] = scan_values[scan_index_down] + scan_left;
        }
        int scan_index_down_19 = (THREADS + tid + 1) * 1024 - 1;
        if (scan_index_down_19 < MAX_EXPERTS) {
            int scan_left_1 = scan_values[scan_index_down_19 - 512];
            scan_values[scan_index_down_19 - 512] = scan_values[scan_index_down_19];
            scan_values[scan_index_down_19] = scan_values[scan_index_down_19] + scan_left_1;
        }
        __syncthreads();
        int scan_index_down_20 = (tid + 1) * 512 - 1;
        if (scan_index_down_20 < MAX_EXPERTS) {
            int scan_left_2 = scan_values[scan_index_down_20 - 256];
            scan_values[scan_index_down_20 - 256] = scan_values[scan_index_down_20];
            scan_values[scan_index_down_20] = scan_values[scan_index_down_20] + scan_left_2;
        }
        int scan_index_down_21 = (THREADS + tid + 1) * 512 - 1;
        if (scan_index_down_21 < MAX_EXPERTS) {
            int scan_left_3 = scan_values[scan_index_down_21 - 256];
            scan_values[scan_index_down_21 - 256] = scan_values[scan_index_down_21];
            scan_values[scan_index_down_21] = scan_values[scan_index_down_21] + scan_left_3;
        }
        __syncthreads();
        int scan_index_down_22 = (tid + 1) * 256 - 1;
        if (scan_index_down_22 < MAX_EXPERTS) {
            int scan_left_4 = scan_values[scan_index_down_22 - 128];
            scan_values[scan_index_down_22 - 128] = scan_values[scan_index_down_22];
            scan_values[scan_index_down_22] = scan_values[scan_index_down_22] + scan_left_4;
        }
        int scan_index_down_23 = (THREADS + tid + 1) * 256 - 1;
        if (scan_index_down_23 < MAX_EXPERTS) {
            int scan_left_5 = scan_values[scan_index_down_23 - 128];
            scan_values[scan_index_down_23 - 128] = scan_values[scan_index_down_23];
            scan_values[scan_index_down_23] = scan_values[scan_index_down_23] + scan_left_5;
        }
        __syncthreads();
        int scan_index_down_24 = (tid + 1) * 128 - 1;
        if (scan_index_down_24 < MAX_EXPERTS) {
            int scan_left_6 = scan_values[scan_index_down_24 - 64];
            scan_values[scan_index_down_24 - 64] = scan_values[scan_index_down_24];
            scan_values[scan_index_down_24] = scan_values[scan_index_down_24] + scan_left_6;
        }
        int scan_index_down_25 = (THREADS + tid + 1) * 128 - 1;
        if (scan_index_down_25 < MAX_EXPERTS) {
            int scan_left_7 = scan_values[scan_index_down_25 - 64];
            scan_values[scan_index_down_25 - 64] = scan_values[scan_index_down_25];
            scan_values[scan_index_down_25] = scan_values[scan_index_down_25] + scan_left_7;
        }
        __syncthreads();
        int scan_index_down_26 = (tid + 1) * 64 - 1;
        if (scan_index_down_26 < MAX_EXPERTS) {
            int scan_left_8 = scan_values[scan_index_down_26 - 32];
            scan_values[scan_index_down_26 - 32] = scan_values[scan_index_down_26];
            scan_values[scan_index_down_26] = scan_values[scan_index_down_26] + scan_left_8;
        }
        int scan_index_down_27 = (THREADS + tid + 1) * 64 - 1;
        if (scan_index_down_27 < MAX_EXPERTS) {
            int scan_left_9 = scan_values[scan_index_down_27 - 32];
            scan_values[scan_index_down_27 - 32] = scan_values[scan_index_down_27];
            scan_values[scan_index_down_27] = scan_values[scan_index_down_27] + scan_left_9;
        }
        __syncthreads();
        int scan_index_down_28 = (tid + 1) * 32 - 1;
        if (scan_index_down_28 < MAX_EXPERTS) {
            int scan_left_10 = scan_values[scan_index_down_28 - 16];
            scan_values[scan_index_down_28 - 16] = scan_values[scan_index_down_28];
            scan_values[scan_index_down_28] = scan_values[scan_index_down_28] + scan_left_10;
        }
        int scan_index_down_29 = (THREADS + tid + 1) * 32 - 1;
        if (scan_index_down_29 < MAX_EXPERTS) {
            int scan_left_11 = scan_values[scan_index_down_29 - 16];
            scan_values[scan_index_down_29 - 16] = scan_values[scan_index_down_29];
            scan_values[scan_index_down_29] = scan_values[scan_index_down_29] + scan_left_11;
        }
        __syncthreads();
        int scan_index_down_30 = (tid + 1) * 16 - 1;
        if (scan_index_down_30 < MAX_EXPERTS) {
            int scan_left_12 = scan_values[scan_index_down_30 - 8];
            scan_values[scan_index_down_30 - 8] = scan_values[scan_index_down_30];
            scan_values[scan_index_down_30] = scan_values[scan_index_down_30] + scan_left_12;
        }
        int scan_index_down_31 = (THREADS + tid + 1) * 16 - 1;
        if (scan_index_down_31 < MAX_EXPERTS) {
            int scan_left_13 = scan_values[scan_index_down_31 - 8];
            scan_values[scan_index_down_31 - 8] = scan_values[scan_index_down_31];
            scan_values[scan_index_down_31] = scan_values[scan_index_down_31] + scan_left_13;
        }
        __syncthreads();
        int scan_index_down_32 = (tid + 1) * 8 - 1;
        if (scan_index_down_32 < MAX_EXPERTS) {
            int scan_left_14 = scan_values[scan_index_down_32 - 4];
            scan_values[scan_index_down_32 - 4] = scan_values[scan_index_down_32];
            scan_values[scan_index_down_32] = scan_values[scan_index_down_32] + scan_left_14;
        }
        int scan_index_down_33 = (THREADS + tid + 1) * 8 - 1;
        if (scan_index_down_33 < MAX_EXPERTS) {
            int scan_left_15 = scan_values[scan_index_down_33 - 4];
            scan_values[scan_index_down_33 - 4] = scan_values[scan_index_down_33];
            scan_values[scan_index_down_33] = scan_values[scan_index_down_33] + scan_left_15;
        }
        __syncthreads();
        int scan_index_down_34 = (tid + 1) * 4 - 1;
        if (scan_index_down_34 < MAX_EXPERTS) {
            int scan_left_16 = scan_values[scan_index_down_34 - 2];
            scan_values[scan_index_down_34 - 2] = scan_values[scan_index_down_34];
            scan_values[scan_index_down_34] = scan_values[scan_index_down_34] + scan_left_16;
        }
        int scan_index_down_35 = (THREADS + tid + 1) * 4 - 1;
        if (scan_index_down_35 < MAX_EXPERTS) {
            int scan_left_17 = scan_values[scan_index_down_35 - 2];
            scan_values[scan_index_down_35 - 2] = scan_values[scan_index_down_35];
            scan_values[scan_index_down_35] = scan_values[scan_index_down_35] + scan_left_17;
        }
        __syncthreads();
        int scan_index_down_36 = (tid + 1) * 2 - 1;
        if (scan_index_down_36 < MAX_EXPERTS) {
            int scan_left_18 = scan_values[scan_index_down_36 - 1];
            scan_values[scan_index_down_36 - 1] = scan_values[scan_index_down_36];
            scan_values[scan_index_down_36] = scan_values[scan_index_down_36] + scan_left_18;
        }
        int scan_index_down_37 = (THREADS + tid + 1) * 2 - 1;
        if (scan_index_down_37 < MAX_EXPERTS) {
            int scan_left_19 = scan_values[scan_index_down_37 - 1];
            scan_values[scan_index_down_37 - 1] = scan_values[scan_index_down_37];
            scan_values[scan_index_down_37] = scan_values[scan_index_down_37] + scan_left_19;
        }
        __syncthreads();
        #pragma unroll
        for (int expert_slot_scan_store = 0; expert_slot_scan_store < MAX_EXPERTS / THREADS; expert_slot_scan_store++) {
            int expert_scan_store = tid + expert_slot_scan_store * THREADS;
            if (expert_scan_store < E) {
                expert_offsets[expert_scan_store] = scan_values[expert_scan_store];
            }
        }
        if (tid == 0) {
            expert_offsets[E] = num_tokens_post_padded[0];
        }
    }
    __threadfence();
    cooperative_groups::this_grid().sync();
    for (int pair = global_thread; pair < M * top_k; pair += num_bids * THREADS) {
        int pair_expert = topk_ids[pair];
        int _atomic_old_0 = atomicAdd(&expert_scatter_offsets[pair_expert], 1);
        int local_row = _atomic_old_0;
        int grouped_row = expert_offsets[pair_expert] + local_row;
        sorted_token_ids[grouped_row] = pair;
        if (local_row % block_m == 0) {
            expert_ids[grouped_row / block_m] = pair_expert;
        }
    }
    for (int padding_expert = global_thread; padding_expert < E; padding_expert += num_bids * THREADS) {
        int count_final = expert_counts[padding_expert];
        int expert_start = expert_offsets[padding_expert];
        int padded_count_final = (count_final + block_m - 1) / block_m * block_m;
        int padding_count = padded_count_final - count_final;
        #pragma unroll
        for (int padding_slot = 0; padding_slot < MAX_BLOCK_M; padding_slot++) {
            if (padding_count > padding_slot) {
                sorted_token_ids[expert_start + count_final + padding_slot] = M * top_k;
            }
        }
    }
}

} // extern "C"


constexpr int kThreads = THREADS;
constexpr int kSmemTotal = SMEM_TOTAL;
constexpr int kWarps = NUM_WARPS;
static_assert(THREADS == NUM_WARPS * 32);
static_assert(MAX_EXPERTS == 512);
#undef LOOM_INF
#undef MAX_BLOCK_M
#undef MAX_EXPERTS
#undef MAX_TOP_K
#undef NUM_MAIN_STAGES
#undef NUM_WARPS
#undef PUBLIC_SHARED_SOFTMAX
#undef SMEM_SCAN_VALUES_OFF
#undef SMEM_SCAN_VALUES_STAGE_BYTES
#undef SMEM_SCAN_VALUES_STRIDE
#undef SMEM_TOTAL
#undef THREADS
}  // namespace alphamoe_router_large_generated

namespace alphamoe_router_small_generated {
typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) LoomTensorMap { uint64_t opaque[16]; };
struct __align__(64) LoomTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(LoomTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(LoomTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else

#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(LoomTensorMap) >= alignof(CUtensorMap), "LoomTensorMap alignment must cover the CUtensorMap CUDA ABI");


__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
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
#define SMEM_SCAN_VALUES_OFF 4096
#define SMEM_SCAN_VALUES_STAGE_BYTES 4096
#define SMEM_SCAN_VALUES_STRIDE 4096
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

__global__ __launch_bounds__(256) void
kernel_alpha_moe_fused_router_small(float* __restrict__ logits, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M, int E, int top_k, int block_m, int has_shared_expert)
{
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
    int* scan_values = reinterpret_cast<int*>(smem_raw + 4096);
    const int scan_values_addr = smem + 4096;
    int* shared_scatter = reinterpret_cast<int*>(smem_raw + 8196);
    const int shared_scatter_addr = smem + 8196;
    int* shared_ids = reinterpret_cast<int*>(smem_raw + 12292);
    const int shared_ids_addr = smem + 12292;
    int* shared_total = reinterpret_cast<int*>(smem_raw + 20484);
    const int shared_total_addr = smem + 20484;

    // === Task calls (dependency order) ===
    int routed_experts = E - has_shared_expert;
    int routed_top_k = top_k - has_shared_expert;
    for (int token = (unsigned int)(bid * NUM_WARPS) + warp; token < M; token += num_bids * NUM_WARPS) {
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
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(local_key));
            unsigned int best_key = _warp_redux_u32_0;
            unsigned int best_bits = best_key ^ 2147483648;
            if ((best_key & 2147483648) == 0) {
                best_bits = best_key ^ 4294967295;
            }
            float best_value = 0.0f;
            best_value = reinterpret_cast<float*>(&best_bits)[0];
            unsigned int tied_index = MAX_EXPERTS;
            #pragma unroll
            for (int expert_slot_tie = 0; expert_slot_tie < MAX_EXPERTS / 32; expert_slot_tie++) {
                int expert_tie = lane + (unsigned int)((MAX_EXPERTS / 32 - 1 - expert_slot_tie) * 32);
                if (expert_tie < routed_experts) {
                    if (row_values[MAX_EXPERTS / 32 - 1 - expert_slot_tie] == best_value) {
                        tied_index = (unsigned int)expert_tie;
                    }
                }
            }
            unsigned int _warp_redux_u32_1;
            asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_1) : "r"(tied_index));
            int best_index = (int)_warp_redux_u32_1;
            if (lane == (unsigned int)route) {
                selected_logit = best_value;
                selected_expert = best_index;
            }
            #pragma unroll
            for (int expert_slot_remove = 0; expert_slot_remove < MAX_EXPERTS / 32; expert_slot_remove++) {
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
            _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
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
        #pragma unroll
        for (int expert_slot_scan_init = 0; expert_slot_scan_init < MAX_EXPERTS / THREADS; expert_slot_scan_init++) {
            int expert_scan_init = tid + expert_slot_scan_init * THREADS;
            int padded_count_init = 0;
            if (expert_scan_init < E) {
                int count_init = shared_counts[expert_scan_init];
                {
                    padded_count_init = (count_init + block_m - 1) / block_m * block_m;
                }
            }
            scan_values[expert_scan_init] = padded_count_init;
        }
        __syncthreads();
        int scan_index_up = (tid + 1) * 2 - 1;
        if (scan_index_up < MAX_EXPERTS) {
            scan_values[scan_index_up] = scan_values[scan_index_up] + scan_values[scan_index_up - 1];
        }
        int scan_index_up_0 = (THREADS + tid + 1) * 2 - 1;
        if (scan_index_up_0 < MAX_EXPERTS) {
            scan_values[scan_index_up_0] = scan_values[scan_index_up_0] + scan_values[scan_index_up_0 - 1];
        }
        __syncthreads();
        int scan_index_up_1 = (tid + 1) * 4 - 1;
        if (scan_index_up_1 < MAX_EXPERTS) {
            scan_values[scan_index_up_1] = scan_values[scan_index_up_1] + scan_values[scan_index_up_1 - 2];
        }
        int scan_index_up_2 = (THREADS + tid + 1) * 4 - 1;
        if (scan_index_up_2 < MAX_EXPERTS) {
            scan_values[scan_index_up_2] = scan_values[scan_index_up_2] + scan_values[scan_index_up_2 - 2];
        }
        __syncthreads();
        int scan_index_up_3 = (tid + 1) * 8 - 1;
        if (scan_index_up_3 < MAX_EXPERTS) {
            scan_values[scan_index_up_3] = scan_values[scan_index_up_3] + scan_values[scan_index_up_3 - 4];
        }
        int scan_index_up_4 = (THREADS + tid + 1) * 8 - 1;
        if (scan_index_up_4 < MAX_EXPERTS) {
            scan_values[scan_index_up_4] = scan_values[scan_index_up_4] + scan_values[scan_index_up_4 - 4];
        }
        __syncthreads();
        int scan_index_up_5 = (tid + 1) * 16 - 1;
        if (scan_index_up_5 < MAX_EXPERTS) {
            scan_values[scan_index_up_5] = scan_values[scan_index_up_5] + scan_values[scan_index_up_5 - 8];
        }
        int scan_index_up_6 = (THREADS + tid + 1) * 16 - 1;
        if (scan_index_up_6 < MAX_EXPERTS) {
            scan_values[scan_index_up_6] = scan_values[scan_index_up_6] + scan_values[scan_index_up_6 - 8];
        }
        __syncthreads();
        int scan_index_up_7 = (tid + 1) * 32 - 1;
        if (scan_index_up_7 < MAX_EXPERTS) {
            scan_values[scan_index_up_7] = scan_values[scan_index_up_7] + scan_values[scan_index_up_7 - 16];
        }
        int scan_index_up_8 = (THREADS + tid + 1) * 32 - 1;
        if (scan_index_up_8 < MAX_EXPERTS) {
            scan_values[scan_index_up_8] = scan_values[scan_index_up_8] + scan_values[scan_index_up_8 - 16];
        }
        __syncthreads();
        int scan_index_up_9 = (tid + 1) * 64 - 1;
        if (scan_index_up_9 < MAX_EXPERTS) {
            scan_values[scan_index_up_9] = scan_values[scan_index_up_9] + scan_values[scan_index_up_9 - 32];
        }
        int scan_index_up_10 = (THREADS + tid + 1) * 64 - 1;
        if (scan_index_up_10 < MAX_EXPERTS) {
            scan_values[scan_index_up_10] = scan_values[scan_index_up_10] + scan_values[scan_index_up_10 - 32];
        }
        __syncthreads();
        int scan_index_up_11 = (tid + 1) * 128 - 1;
        if (scan_index_up_11 < MAX_EXPERTS) {
            scan_values[scan_index_up_11] = scan_values[scan_index_up_11] + scan_values[scan_index_up_11 - 64];
        }
        int scan_index_up_12 = (THREADS + tid + 1) * 128 - 1;
        if (scan_index_up_12 < MAX_EXPERTS) {
            scan_values[scan_index_up_12] = scan_values[scan_index_up_12] + scan_values[scan_index_up_12 - 64];
        }
        __syncthreads();
        int scan_index_up_13 = (tid + 1) * 256 - 1;
        if (scan_index_up_13 < MAX_EXPERTS) {
            scan_values[scan_index_up_13] = scan_values[scan_index_up_13] + scan_values[scan_index_up_13 - 128];
        }
        int scan_index_up_14 = (THREADS + tid + 1) * 256 - 1;
        if (scan_index_up_14 < MAX_EXPERTS) {
            scan_values[scan_index_up_14] = scan_values[scan_index_up_14] + scan_values[scan_index_up_14 - 128];
        }
        __syncthreads();
        int scan_index_up_15 = (tid + 1) * 512 - 1;
        if (scan_index_up_15 < MAX_EXPERTS) {
            scan_values[scan_index_up_15] = scan_values[scan_index_up_15] + scan_values[scan_index_up_15 - 256];
        }
        int scan_index_up_16 = (THREADS + tid + 1) * 512 - 1;
        if (scan_index_up_16 < MAX_EXPERTS) {
            scan_values[scan_index_up_16] = scan_values[scan_index_up_16] + scan_values[scan_index_up_16 - 256];
        }
        __syncthreads();
        int scan_index_up_17 = (tid + 1) * 1024 - 1;
        if (scan_index_up_17 < MAX_EXPERTS) {
            scan_values[scan_index_up_17] = scan_values[scan_index_up_17] + scan_values[scan_index_up_17 - 512];
        }
        int scan_index_up_18 = (THREADS + tid + 1) * 1024 - 1;
        if (scan_index_up_18 < MAX_EXPERTS) {
            scan_values[scan_index_up_18] = scan_values[scan_index_up_18] + scan_values[scan_index_up_18 - 512];
        }
        __syncthreads();
        if (tid == 0) {
            int padded_total = scan_values[MAX_EXPERTS - 1];
            shared_total[0] = padded_total;
            scan_values[MAX_EXPERTS - 1] = 0;
        }
        __syncthreads();
        int scan_index_down = (tid + 1) * 1024 - 1;
        if (scan_index_down < MAX_EXPERTS) {
            int scan_left = scan_values[scan_index_down - 512];
            scan_values[scan_index_down - 512] = scan_values[scan_index_down];
            scan_values[scan_index_down] = scan_values[scan_index_down] + scan_left;
        }
        int scan_index_down_19 = (THREADS + tid + 1) * 1024 - 1;
        if (scan_index_down_19 < MAX_EXPERTS) {
            int scan_left_1 = scan_values[scan_index_down_19 - 512];
            scan_values[scan_index_down_19 - 512] = scan_values[scan_index_down_19];
            scan_values[scan_index_down_19] = scan_values[scan_index_down_19] + scan_left_1;
        }
        __syncthreads();
        int scan_index_down_20 = (tid + 1) * 512 - 1;
        if (scan_index_down_20 < MAX_EXPERTS) {
            int scan_left_2 = scan_values[scan_index_down_20 - 256];
            scan_values[scan_index_down_20 - 256] = scan_values[scan_index_down_20];
            scan_values[scan_index_down_20] = scan_values[scan_index_down_20] + scan_left_2;
        }
        int scan_index_down_21 = (THREADS + tid + 1) * 512 - 1;
        if (scan_index_down_21 < MAX_EXPERTS) {
            int scan_left_3 = scan_values[scan_index_down_21 - 256];
            scan_values[scan_index_down_21 - 256] = scan_values[scan_index_down_21];
            scan_values[scan_index_down_21] = scan_values[scan_index_down_21] + scan_left_3;
        }
        __syncthreads();
        int scan_index_down_22 = (tid + 1) * 256 - 1;
        if (scan_index_down_22 < MAX_EXPERTS) {
            int scan_left_4 = scan_values[scan_index_down_22 - 128];
            scan_values[scan_index_down_22 - 128] = scan_values[scan_index_down_22];
            scan_values[scan_index_down_22] = scan_values[scan_index_down_22] + scan_left_4;
        }
        int scan_index_down_23 = (THREADS + tid + 1) * 256 - 1;
        if (scan_index_down_23 < MAX_EXPERTS) {
            int scan_left_5 = scan_values[scan_index_down_23 - 128];
            scan_values[scan_index_down_23 - 128] = scan_values[scan_index_down_23];
            scan_values[scan_index_down_23] = scan_values[scan_index_down_23] + scan_left_5;
        }
        __syncthreads();
        int scan_index_down_24 = (tid + 1) * 128 - 1;
        if (scan_index_down_24 < MAX_EXPERTS) {
            int scan_left_6 = scan_values[scan_index_down_24 - 64];
            scan_values[scan_index_down_24 - 64] = scan_values[scan_index_down_24];
            scan_values[scan_index_down_24] = scan_values[scan_index_down_24] + scan_left_6;
        }
        int scan_index_down_25 = (THREADS + tid + 1) * 128 - 1;
        if (scan_index_down_25 < MAX_EXPERTS) {
            int scan_left_7 = scan_values[scan_index_down_25 - 64];
            scan_values[scan_index_down_25 - 64] = scan_values[scan_index_down_25];
            scan_values[scan_index_down_25] = scan_values[scan_index_down_25] + scan_left_7;
        }
        __syncthreads();
        int scan_index_down_26 = (tid + 1) * 64 - 1;
        if (scan_index_down_26 < MAX_EXPERTS) {
            int scan_left_8 = scan_values[scan_index_down_26 - 32];
            scan_values[scan_index_down_26 - 32] = scan_values[scan_index_down_26];
            scan_values[scan_index_down_26] = scan_values[scan_index_down_26] + scan_left_8;
        }
        int scan_index_down_27 = (THREADS + tid + 1) * 64 - 1;
        if (scan_index_down_27 < MAX_EXPERTS) {
            int scan_left_9 = scan_values[scan_index_down_27 - 32];
            scan_values[scan_index_down_27 - 32] = scan_values[scan_index_down_27];
            scan_values[scan_index_down_27] = scan_values[scan_index_down_27] + scan_left_9;
        }
        __syncthreads();
        int scan_index_down_28 = (tid + 1) * 32 - 1;
        if (scan_index_down_28 < MAX_EXPERTS) {
            int scan_left_10 = scan_values[scan_index_down_28 - 16];
            scan_values[scan_index_down_28 - 16] = scan_values[scan_index_down_28];
            scan_values[scan_index_down_28] = scan_values[scan_index_down_28] + scan_left_10;
        }
        int scan_index_down_29 = (THREADS + tid + 1) * 32 - 1;
        if (scan_index_down_29 < MAX_EXPERTS) {
            int scan_left_11 = scan_values[scan_index_down_29 - 16];
            scan_values[scan_index_down_29 - 16] = scan_values[scan_index_down_29];
            scan_values[scan_index_down_29] = scan_values[scan_index_down_29] + scan_left_11;
        }
        __syncthreads();
        int scan_index_down_30 = (tid + 1) * 16 - 1;
        if (scan_index_down_30 < MAX_EXPERTS) {
            int scan_left_12 = scan_values[scan_index_down_30 - 8];
            scan_values[scan_index_down_30 - 8] = scan_values[scan_index_down_30];
            scan_values[scan_index_down_30] = scan_values[scan_index_down_30] + scan_left_12;
        }
        int scan_index_down_31 = (THREADS + tid + 1) * 16 - 1;
        if (scan_index_down_31 < MAX_EXPERTS) {
            int scan_left_13 = scan_values[scan_index_down_31 - 8];
            scan_values[scan_index_down_31 - 8] = scan_values[scan_index_down_31];
            scan_values[scan_index_down_31] = scan_values[scan_index_down_31] + scan_left_13;
        }
        __syncthreads();
        int scan_index_down_32 = (tid + 1) * 8 - 1;
        if (scan_index_down_32 < MAX_EXPERTS) {
            int scan_left_14 = scan_values[scan_index_down_32 - 4];
            scan_values[scan_index_down_32 - 4] = scan_values[scan_index_down_32];
            scan_values[scan_index_down_32] = scan_values[scan_index_down_32] + scan_left_14;
        }
        int scan_index_down_33 = (THREADS + tid + 1) * 8 - 1;
        if (scan_index_down_33 < MAX_EXPERTS) {
            int scan_left_15 = scan_values[scan_index_down_33 - 4];
            scan_values[scan_index_down_33 - 4] = scan_values[scan_index_down_33];
            scan_values[scan_index_down_33] = scan_values[scan_index_down_33] + scan_left_15;
        }
        __syncthreads();
        int scan_index_down_34 = (tid + 1) * 4 - 1;
        if (scan_index_down_34 < MAX_EXPERTS) {
            int scan_left_16 = scan_values[scan_index_down_34 - 2];
            scan_values[scan_index_down_34 - 2] = scan_values[scan_index_down_34];
            scan_values[scan_index_down_34] = scan_values[scan_index_down_34] + scan_left_16;
        }
        int scan_index_down_35 = (THREADS + tid + 1) * 4 - 1;
        if (scan_index_down_35 < MAX_EXPERTS) {
            int scan_left_17 = scan_values[scan_index_down_35 - 2];
            scan_values[scan_index_down_35 - 2] = scan_values[scan_index_down_35];
            scan_values[scan_index_down_35] = scan_values[scan_index_down_35] + scan_left_17;
        }
        __syncthreads();
        int scan_index_down_36 = (tid + 1) * 2 - 1;
        if (scan_index_down_36 < MAX_EXPERTS) {
            int scan_left_18 = scan_values[scan_index_down_36 - 1];
            scan_values[scan_index_down_36 - 1] = scan_values[scan_index_down_36];
            scan_values[scan_index_down_36] = scan_values[scan_index_down_36] + scan_left_18;
        }
        int scan_index_down_37 = (THREADS + tid + 1) * 2 - 1;
        if (scan_index_down_37 < MAX_EXPERTS) {
            int scan_left_19 = scan_values[scan_index_down_37 - 1];
            scan_values[scan_index_down_37 - 1] = scan_values[scan_index_down_37];
            scan_values[scan_index_down_37] = scan_values[scan_index_down_37] + scan_left_19;
        }
        __syncthreads();
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
                if (local_row % block_m == 0) {
                    expert_ids[grouped_row / block_m] = pair_expert;
                }
            }
        }
        for (int padding_expert = tid; padding_expert < E; padding_expert += THREADS) {
            int count_final = shared_counts[padding_expert];
            int expert_start = shared_offsets[padding_expert];
            int padded_count_final = 0;
            {
                padded_count_final = (count_final + block_m - 1) / block_m * block_m;
            }
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

} // extern "C"


constexpr int kThreads = THREADS;
constexpr int kSmemTotal = SMEM_TOTAL;
constexpr int kWarps = NUM_WARPS;
static_assert(THREADS == NUM_WARPS * 32);
static_assert(MAX_EXPERTS == 512);
#undef FAST_BLOCK_ALIGN
#undef LOOM_INF
#undef MAX_BLOCK_M
#undef MAX_EXPERTS
#undef MAX_TOP_K
#undef NUM_MAIN_STAGES
#undef NUM_WARPS
#undef PUBLIC_SHARED_SOFTMAX
#undef SINGLE_CTA
#undef SMEM_SCAN_VALUES_OFF
#undef SMEM_SCAN_VALUES_STAGE_BYTES
#undef SMEM_SCAN_VALUES_STRIDE
#undef SMEM_SHARED_COUNTS_OFF
#undef SMEM_SHARED_COUNTS_STAGE_BYTES
#undef SMEM_SHARED_COUNTS_STRIDE
#undef SMEM_SHARED_IDS_OFF
#undef SMEM_SHARED_IDS_STAGE_BYTES
#undef SMEM_SHARED_IDS_STRIDE
#undef SMEM_SHARED_OFFSETS_OFF
#undef SMEM_SHARED_OFFSETS_STAGE_BYTES
#undef SMEM_SHARED_OFFSETS_STRIDE
#undef SMEM_SHARED_SCATTER_OFF
#undef SMEM_SHARED_SCATTER_STAGE_BYTES
#undef SMEM_SHARED_SCATTER_STRIDE
#undef SMEM_SHARED_TOTAL_OFF
#undef SMEM_SHARED_TOTAL_STAGE_BYTES
#undef SMEM_SHARED_TOTAL_STRIDE
#undef SMEM_TOTAL
#undef THREADS
}  // namespace alphamoe_router_small_generated

using alphamoe_router_large_generated::kernel_alpha_moe_fused_router;
using alphamoe_router_small_generated::kernel_alpha_moe_fused_router_small;
constexpr int kGeneratedThreads = alphamoe_router_large_generated::kThreads;
constexpr int kGeneratedSmemTotal = alphamoe_router_large_generated::kSmemTotal;
constexpr int kGeneratedWarps = alphamoe_router_large_generated::kWarps;
constexpr int kSmallGeneratedThreads = alphamoe_router_small_generated::kThreads;
constexpr int kSmallGeneratedSmemTotal = alphamoe_router_small_generated::kSmemTotal;
constexpr int kSmallGeneratedWarps = alphamoe_router_small_generated::kWarps;


namespace alphamoe_router_large_routed_generated {
typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) LoomTensorMap { uint64_t opaque[16]; };
struct __align__(64) LoomTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(LoomTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(LoomTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else

#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(LoomTensorMap) >= alignof(CUtensorMap), "LoomTensorMap alignment must cover the CUtensorMap CUDA ABI");


__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SCAN_VALUES_OFF 0
#define SMEM_SCAN_VALUES_STAGE_BYTES 4096
#define SMEM_SCAN_VALUES_STRIDE 4096
#define SMEM_TOTAL 4096
#define THREADS 256
#define NUM_WARPS 8
#define MAX_EXPERTS 512
#define MAX_TOP_K 16
#define MAX_BLOCK_M 16
#define PUBLIC_SHARED_SOFTMAX 0


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

__global__ __launch_bounds__(256) void
kernel_alpha_moe_fused_router_routed(float* __restrict__ logits, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M, int E, int top_k, int block_m, int has_shared_expert)
{
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
    int* scan_values = reinterpret_cast<int*>(smem_raw + 0);
    const int scan_values_addr = smem + 0;

    // === Task calls (dependency order) ===
    int global_thread = bid * THREADS + tid;
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
    __threadfence();
    cooperative_groups::this_grid().sync();
    int routed_experts = E - has_shared_expert;
    int routed_top_k = top_k - has_shared_expert;
    for (int token = (unsigned int)(bid * NUM_WARPS) + warp; token < M; token += num_bids * NUM_WARPS) {
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
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(local_key));
            unsigned int best_key = _warp_redux_u32_0;
            unsigned int best_bits = best_key ^ 2147483648;
            if ((best_key & 2147483648) == 0) {
                best_bits = best_key ^ 4294967295;
            }
            float best_value = 0.0f;
            best_value = reinterpret_cast<float*>(&best_bits)[0];
            unsigned int tied_index = MAX_EXPERTS;
            #pragma unroll
            for (int expert_slot_tie = 0; expert_slot_tie < MAX_EXPERTS / 32; expert_slot_tie++) {
                int expert_tie = lane + (unsigned int)((MAX_EXPERTS / 32 - 1 - expert_slot_tie) * 32);
                if (expert_tie < routed_experts) {
                    if (row_values[MAX_EXPERTS / 32 - 1 - expert_slot_tie] == best_value) {
                        tied_index = (unsigned int)expert_tie;
                    }
                }
            }
            unsigned int _warp_redux_u32_1;
            asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_1) : "r"(tied_index));
            int best_index = (int)_warp_redux_u32_1;
            if (lane == (unsigned int)route) {
                selected_logit = best_value;
                selected_expert = best_index;
            }
            #pragma unroll
            for (int expert_slot_remove = 0; expert_slot_remove < MAX_EXPERTS / 32; expert_slot_remove++) {
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
        float selected_for_max = -LOOM_INF;
        if ((unsigned int)softmax_top_k > lane) {
            selected_for_max = selected_logit;
        }
        float _warp_reduce_0 = selected_for_max;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
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
            atomicAdd(&expert_counts[selected_expert], 1);
        }
    }
    __threadfence();
    cooperative_groups::this_grid().sync();
    if (bid == 0) {
        #pragma unroll
        for (int expert_slot_scan_init = 0; expert_slot_scan_init < MAX_EXPERTS / THREADS; expert_slot_scan_init++) {
            int expert_scan_init = tid + expert_slot_scan_init * THREADS;
            int padded_count_init = 0;
            if (expert_scan_init < E) {
                int count_init = expert_counts[expert_scan_init];
                padded_count_init = (count_init + block_m - 1) / block_m * block_m;
            }
            scan_values[expert_scan_init] = padded_count_init;
        }
        __syncthreads();
        int scan_index_up = (tid + 1) * 2 - 1;
        if (scan_index_up < MAX_EXPERTS) {
            scan_values[scan_index_up] = scan_values[scan_index_up] + scan_values[scan_index_up - 1];
        }
        int scan_index_up_0 = (THREADS + tid + 1) * 2 - 1;
        if (scan_index_up_0 < MAX_EXPERTS) {
            scan_values[scan_index_up_0] = scan_values[scan_index_up_0] + scan_values[scan_index_up_0 - 1];
        }
        __syncthreads();
        int scan_index_up_1 = (tid + 1) * 4 - 1;
        if (scan_index_up_1 < MAX_EXPERTS) {
            scan_values[scan_index_up_1] = scan_values[scan_index_up_1] + scan_values[scan_index_up_1 - 2];
        }
        int scan_index_up_2 = (THREADS + tid + 1) * 4 - 1;
        if (scan_index_up_2 < MAX_EXPERTS) {
            scan_values[scan_index_up_2] = scan_values[scan_index_up_2] + scan_values[scan_index_up_2 - 2];
        }
        __syncthreads();
        int scan_index_up_3 = (tid + 1) * 8 - 1;
        if (scan_index_up_3 < MAX_EXPERTS) {
            scan_values[scan_index_up_3] = scan_values[scan_index_up_3] + scan_values[scan_index_up_3 - 4];
        }
        int scan_index_up_4 = (THREADS + tid + 1) * 8 - 1;
        if (scan_index_up_4 < MAX_EXPERTS) {
            scan_values[scan_index_up_4] = scan_values[scan_index_up_4] + scan_values[scan_index_up_4 - 4];
        }
        __syncthreads();
        int scan_index_up_5 = (tid + 1) * 16 - 1;
        if (scan_index_up_5 < MAX_EXPERTS) {
            scan_values[scan_index_up_5] = scan_values[scan_index_up_5] + scan_values[scan_index_up_5 - 8];
        }
        int scan_index_up_6 = (THREADS + tid + 1) * 16 - 1;
        if (scan_index_up_6 < MAX_EXPERTS) {
            scan_values[scan_index_up_6] = scan_values[scan_index_up_6] + scan_values[scan_index_up_6 - 8];
        }
        __syncthreads();
        int scan_index_up_7 = (tid + 1) * 32 - 1;
        if (scan_index_up_7 < MAX_EXPERTS) {
            scan_values[scan_index_up_7] = scan_values[scan_index_up_7] + scan_values[scan_index_up_7 - 16];
        }
        int scan_index_up_8 = (THREADS + tid + 1) * 32 - 1;
        if (scan_index_up_8 < MAX_EXPERTS) {
            scan_values[scan_index_up_8] = scan_values[scan_index_up_8] + scan_values[scan_index_up_8 - 16];
        }
        __syncthreads();
        int scan_index_up_9 = (tid + 1) * 64 - 1;
        if (scan_index_up_9 < MAX_EXPERTS) {
            scan_values[scan_index_up_9] = scan_values[scan_index_up_9] + scan_values[scan_index_up_9 - 32];
        }
        int scan_index_up_10 = (THREADS + tid + 1) * 64 - 1;
        if (scan_index_up_10 < MAX_EXPERTS) {
            scan_values[scan_index_up_10] = scan_values[scan_index_up_10] + scan_values[scan_index_up_10 - 32];
        }
        __syncthreads();
        int scan_index_up_11 = (tid + 1) * 128 - 1;
        if (scan_index_up_11 < MAX_EXPERTS) {
            scan_values[scan_index_up_11] = scan_values[scan_index_up_11] + scan_values[scan_index_up_11 - 64];
        }
        int scan_index_up_12 = (THREADS + tid + 1) * 128 - 1;
        if (scan_index_up_12 < MAX_EXPERTS) {
            scan_values[scan_index_up_12] = scan_values[scan_index_up_12] + scan_values[scan_index_up_12 - 64];
        }
        __syncthreads();
        int scan_index_up_13 = (tid + 1) * 256 - 1;
        if (scan_index_up_13 < MAX_EXPERTS) {
            scan_values[scan_index_up_13] = scan_values[scan_index_up_13] + scan_values[scan_index_up_13 - 128];
        }
        int scan_index_up_14 = (THREADS + tid + 1) * 256 - 1;
        if (scan_index_up_14 < MAX_EXPERTS) {
            scan_values[scan_index_up_14] = scan_values[scan_index_up_14] + scan_values[scan_index_up_14 - 128];
        }
        __syncthreads();
        int scan_index_up_15 = (tid + 1) * 512 - 1;
        if (scan_index_up_15 < MAX_EXPERTS) {
            scan_values[scan_index_up_15] = scan_values[scan_index_up_15] + scan_values[scan_index_up_15 - 256];
        }
        int scan_index_up_16 = (THREADS + tid + 1) * 512 - 1;
        if (scan_index_up_16 < MAX_EXPERTS) {
            scan_values[scan_index_up_16] = scan_values[scan_index_up_16] + scan_values[scan_index_up_16 - 256];
        }
        __syncthreads();
        int scan_index_up_17 = (tid + 1) * 1024 - 1;
        if (scan_index_up_17 < MAX_EXPERTS) {
            scan_values[scan_index_up_17] = scan_values[scan_index_up_17] + scan_values[scan_index_up_17 - 512];
        }
        int scan_index_up_18 = (THREADS + tid + 1) * 1024 - 1;
        if (scan_index_up_18 < MAX_EXPERTS) {
            scan_values[scan_index_up_18] = scan_values[scan_index_up_18] + scan_values[scan_index_up_18 - 512];
        }
        __syncthreads();
        if (tid == 0) {
            int padded_total = scan_values[MAX_EXPERTS - 1];
            num_tokens_post_padded[0] = padded_total;
            scan_values[MAX_EXPERTS - 1] = 0;
        }
        __syncthreads();
        int scan_index_down = (tid + 1) * 1024 - 1;
        if (scan_index_down < MAX_EXPERTS) {
            int scan_left = scan_values[scan_index_down - 512];
            scan_values[scan_index_down - 512] = scan_values[scan_index_down];
            scan_values[scan_index_down] = scan_values[scan_index_down] + scan_left;
        }
        int scan_index_down_19 = (THREADS + tid + 1) * 1024 - 1;
        if (scan_index_down_19 < MAX_EXPERTS) {
            int scan_left_1 = scan_values[scan_index_down_19 - 512];
            scan_values[scan_index_down_19 - 512] = scan_values[scan_index_down_19];
            scan_values[scan_index_down_19] = scan_values[scan_index_down_19] + scan_left_1;
        }
        __syncthreads();
        int scan_index_down_20 = (tid + 1) * 512 - 1;
        if (scan_index_down_20 < MAX_EXPERTS) {
            int scan_left_2 = scan_values[scan_index_down_20 - 256];
            scan_values[scan_index_down_20 - 256] = scan_values[scan_index_down_20];
            scan_values[scan_index_down_20] = scan_values[scan_index_down_20] + scan_left_2;
        }
        int scan_index_down_21 = (THREADS + tid + 1) * 512 - 1;
        if (scan_index_down_21 < MAX_EXPERTS) {
            int scan_left_3 = scan_values[scan_index_down_21 - 256];
            scan_values[scan_index_down_21 - 256] = scan_values[scan_index_down_21];
            scan_values[scan_index_down_21] = scan_values[scan_index_down_21] + scan_left_3;
        }
        __syncthreads();
        int scan_index_down_22 = (tid + 1) * 256 - 1;
        if (scan_index_down_22 < MAX_EXPERTS) {
            int scan_left_4 = scan_values[scan_index_down_22 - 128];
            scan_values[scan_index_down_22 - 128] = scan_values[scan_index_down_22];
            scan_values[scan_index_down_22] = scan_values[scan_index_down_22] + scan_left_4;
        }
        int scan_index_down_23 = (THREADS + tid + 1) * 256 - 1;
        if (scan_index_down_23 < MAX_EXPERTS) {
            int scan_left_5 = scan_values[scan_index_down_23 - 128];
            scan_values[scan_index_down_23 - 128] = scan_values[scan_index_down_23];
            scan_values[scan_index_down_23] = scan_values[scan_index_down_23] + scan_left_5;
        }
        __syncthreads();
        int scan_index_down_24 = (tid + 1) * 128 - 1;
        if (scan_index_down_24 < MAX_EXPERTS) {
            int scan_left_6 = scan_values[scan_index_down_24 - 64];
            scan_values[scan_index_down_24 - 64] = scan_values[scan_index_down_24];
            scan_values[scan_index_down_24] = scan_values[scan_index_down_24] + scan_left_6;
        }
        int scan_index_down_25 = (THREADS + tid + 1) * 128 - 1;
        if (scan_index_down_25 < MAX_EXPERTS) {
            int scan_left_7 = scan_values[scan_index_down_25 - 64];
            scan_values[scan_index_down_25 - 64] = scan_values[scan_index_down_25];
            scan_values[scan_index_down_25] = scan_values[scan_index_down_25] + scan_left_7;
        }
        __syncthreads();
        int scan_index_down_26 = (tid + 1) * 64 - 1;
        if (scan_index_down_26 < MAX_EXPERTS) {
            int scan_left_8 = scan_values[scan_index_down_26 - 32];
            scan_values[scan_index_down_26 - 32] = scan_values[scan_index_down_26];
            scan_values[scan_index_down_26] = scan_values[scan_index_down_26] + scan_left_8;
        }
        int scan_index_down_27 = (THREADS + tid + 1) * 64 - 1;
        if (scan_index_down_27 < MAX_EXPERTS) {
            int scan_left_9 = scan_values[scan_index_down_27 - 32];
            scan_values[scan_index_down_27 - 32] = scan_values[scan_index_down_27];
            scan_values[scan_index_down_27] = scan_values[scan_index_down_27] + scan_left_9;
        }
        __syncthreads();
        int scan_index_down_28 = (tid + 1) * 32 - 1;
        if (scan_index_down_28 < MAX_EXPERTS) {
            int scan_left_10 = scan_values[scan_index_down_28 - 16];
            scan_values[scan_index_down_28 - 16] = scan_values[scan_index_down_28];
            scan_values[scan_index_down_28] = scan_values[scan_index_down_28] + scan_left_10;
        }
        int scan_index_down_29 = (THREADS + tid + 1) * 32 - 1;
        if (scan_index_down_29 < MAX_EXPERTS) {
            int scan_left_11 = scan_values[scan_index_down_29 - 16];
            scan_values[scan_index_down_29 - 16] = scan_values[scan_index_down_29];
            scan_values[scan_index_down_29] = scan_values[scan_index_down_29] + scan_left_11;
        }
        __syncthreads();
        int scan_index_down_30 = (tid + 1) * 16 - 1;
        if (scan_index_down_30 < MAX_EXPERTS) {
            int scan_left_12 = scan_values[scan_index_down_30 - 8];
            scan_values[scan_index_down_30 - 8] = scan_values[scan_index_down_30];
            scan_values[scan_index_down_30] = scan_values[scan_index_down_30] + scan_left_12;
        }
        int scan_index_down_31 = (THREADS + tid + 1) * 16 - 1;
        if (scan_index_down_31 < MAX_EXPERTS) {
            int scan_left_13 = scan_values[scan_index_down_31 - 8];
            scan_values[scan_index_down_31 - 8] = scan_values[scan_index_down_31];
            scan_values[scan_index_down_31] = scan_values[scan_index_down_31] + scan_left_13;
        }
        __syncthreads();
        int scan_index_down_32 = (tid + 1) * 8 - 1;
        if (scan_index_down_32 < MAX_EXPERTS) {
            int scan_left_14 = scan_values[scan_index_down_32 - 4];
            scan_values[scan_index_down_32 - 4] = scan_values[scan_index_down_32];
            scan_values[scan_index_down_32] = scan_values[scan_index_down_32] + scan_left_14;
        }
        int scan_index_down_33 = (THREADS + tid + 1) * 8 - 1;
        if (scan_index_down_33 < MAX_EXPERTS) {
            int scan_left_15 = scan_values[scan_index_down_33 - 4];
            scan_values[scan_index_down_33 - 4] = scan_values[scan_index_down_33];
            scan_values[scan_index_down_33] = scan_values[scan_index_down_33] + scan_left_15;
        }
        __syncthreads();
        int scan_index_down_34 = (tid + 1) * 4 - 1;
        if (scan_index_down_34 < MAX_EXPERTS) {
            int scan_left_16 = scan_values[scan_index_down_34 - 2];
            scan_values[scan_index_down_34 - 2] = scan_values[scan_index_down_34];
            scan_values[scan_index_down_34] = scan_values[scan_index_down_34] + scan_left_16;
        }
        int scan_index_down_35 = (THREADS + tid + 1) * 4 - 1;
        if (scan_index_down_35 < MAX_EXPERTS) {
            int scan_left_17 = scan_values[scan_index_down_35 - 2];
            scan_values[scan_index_down_35 - 2] = scan_values[scan_index_down_35];
            scan_values[scan_index_down_35] = scan_values[scan_index_down_35] + scan_left_17;
        }
        __syncthreads();
        int scan_index_down_36 = (tid + 1) * 2 - 1;
        if (scan_index_down_36 < MAX_EXPERTS) {
            int scan_left_18 = scan_values[scan_index_down_36 - 1];
            scan_values[scan_index_down_36 - 1] = scan_values[scan_index_down_36];
            scan_values[scan_index_down_36] = scan_values[scan_index_down_36] + scan_left_18;
        }
        int scan_index_down_37 = (THREADS + tid + 1) * 2 - 1;
        if (scan_index_down_37 < MAX_EXPERTS) {
            int scan_left_19 = scan_values[scan_index_down_37 - 1];
            scan_values[scan_index_down_37 - 1] = scan_values[scan_index_down_37];
            scan_values[scan_index_down_37] = scan_values[scan_index_down_37] + scan_left_19;
        }
        __syncthreads();
        #pragma unroll
        for (int expert_slot_scan_store = 0; expert_slot_scan_store < MAX_EXPERTS / THREADS; expert_slot_scan_store++) {
            int expert_scan_store = tid + expert_slot_scan_store * THREADS;
            if (expert_scan_store < E) {
                expert_offsets[expert_scan_store] = scan_values[expert_scan_store];
            }
        }
        if (tid == 0) {
            expert_offsets[E] = num_tokens_post_padded[0];
        }
    }
    __threadfence();
    cooperative_groups::this_grid().sync();
    for (int pair = global_thread; pair < M * top_k; pair += num_bids * THREADS) {
        int pair_expert = topk_ids[pair];
        int _atomic_old_0 = atomicAdd(&expert_scatter_offsets[pair_expert], 1);
        int local_row = _atomic_old_0;
        int grouped_row = expert_offsets[pair_expert] + local_row;
        sorted_token_ids[grouped_row] = pair;
        if (local_row % block_m == 0) {
            expert_ids[grouped_row / block_m] = pair_expert;
        }
    }
    for (int padding_expert = global_thread; padding_expert < E; padding_expert += num_bids * THREADS) {
        int count_final = expert_counts[padding_expert];
        int expert_start = expert_offsets[padding_expert];
        int padded_count_final = (count_final + block_m - 1) / block_m * block_m;
        int padding_count = padded_count_final - count_final;
        #pragma unroll
        for (int padding_slot = 0; padding_slot < MAX_BLOCK_M; padding_slot++) {
            if (padding_count > padding_slot) {
                sorted_token_ids[expert_start + count_final + padding_slot] = M * top_k;
            }
        }
    }
}

} // extern "C"


constexpr int kThreads = THREADS;
constexpr int kSmemTotal = SMEM_TOTAL;
constexpr int kWarps = NUM_WARPS;
static_assert(THREADS == NUM_WARPS * 32);
static_assert(MAX_EXPERTS == 512);
#undef LOOM_INF
#undef MAX_BLOCK_M
#undef MAX_EXPERTS
#undef MAX_TOP_K
#undef NUM_MAIN_STAGES
#undef NUM_WARPS
#undef PUBLIC_SHARED_SOFTMAX
#undef SMEM_SCAN_VALUES_OFF
#undef SMEM_SCAN_VALUES_STAGE_BYTES
#undef SMEM_SCAN_VALUES_STRIDE
#undef SMEM_TOTAL
#undef THREADS
}  // namespace alphamoe_router_large_routed_generated

namespace alphamoe_router_small_routed_generated {
typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) LoomTensorMap { uint64_t opaque[16]; };
struct __align__(64) LoomTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(LoomTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(LoomTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else

#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(LoomTensorMap) >= alignof(CUtensorMap), "LoomTensorMap alignment must cover the CUtensorMap CUDA ABI");


__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
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
#define SMEM_SCAN_VALUES_OFF 4096
#define SMEM_SCAN_VALUES_STAGE_BYTES 4096
#define SMEM_SCAN_VALUES_STRIDE 4096
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
#define PUBLIC_SHARED_SOFTMAX 0
#define FAST_BLOCK_ALIGN 0
#define SINGLE_CTA 0


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

__global__ __launch_bounds__(256) void
kernel_alpha_moe_fused_router_small_routed(float* __restrict__ logits, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M, int E, int top_k, int block_m, int has_shared_expert)
{
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
    int* scan_values = reinterpret_cast<int*>(smem_raw + 4096);
    const int scan_values_addr = smem + 4096;
    int* shared_scatter = reinterpret_cast<int*>(smem_raw + 8196);
    const int shared_scatter_addr = smem + 8196;
    int* shared_ids = reinterpret_cast<int*>(smem_raw + 12292);
    const int shared_ids_addr = smem + 12292;
    int* shared_total = reinterpret_cast<int*>(smem_raw + 20484);
    const int shared_total_addr = smem + 20484;

    // === Task calls (dependency order) ===
    int routed_experts = E - has_shared_expert;
    int routed_top_k = top_k - has_shared_expert;
    for (int token = (unsigned int)(bid * NUM_WARPS) + warp; token < M; token += num_bids * NUM_WARPS) {
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
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(local_key));
            unsigned int best_key = _warp_redux_u32_0;
            unsigned int best_bits = best_key ^ 2147483648;
            if ((best_key & 2147483648) == 0) {
                best_bits = best_key ^ 4294967295;
            }
            float best_value = 0.0f;
            best_value = reinterpret_cast<float*>(&best_bits)[0];
            unsigned int tied_index = MAX_EXPERTS;
            #pragma unroll
            for (int expert_slot_tie = 0; expert_slot_tie < MAX_EXPERTS / 32; expert_slot_tie++) {
                int expert_tie = lane + (unsigned int)((MAX_EXPERTS / 32 - 1 - expert_slot_tie) * 32);
                if (expert_tie < routed_experts) {
                    if (row_values[MAX_EXPERTS / 32 - 1 - expert_slot_tie] == best_value) {
                        tied_index = (unsigned int)expert_tie;
                    }
                }
            }
            unsigned int _warp_redux_u32_1;
            asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_1) : "r"(tied_index));
            int best_index = (int)_warp_redux_u32_1;
            if (lane == (unsigned int)route) {
                selected_logit = best_value;
                selected_expert = best_index;
            }
            #pragma unroll
            for (int expert_slot_remove = 0; expert_slot_remove < MAX_EXPERTS / 32; expert_slot_remove++) {
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
        float selected_for_max = -LOOM_INF;
        if ((unsigned int)softmax_top_k > lane) {
            selected_for_max = selected_logit;
        }
        float _warp_reduce_0 = selected_for_max;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
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
        #pragma unroll
        for (int expert_slot_scan_init = 0; expert_slot_scan_init < MAX_EXPERTS / THREADS; expert_slot_scan_init++) {
            int expert_scan_init = tid + expert_slot_scan_init * THREADS;
            int padded_count_init = 0;
            if (expert_scan_init < E) {
                int count_init = shared_counts[expert_scan_init];
                {
                    padded_count_init = (count_init + block_m - 1) / block_m * block_m;
                }
            }
            scan_values[expert_scan_init] = padded_count_init;
        }
        __syncthreads();
        int scan_index_up = (tid + 1) * 2 - 1;
        if (scan_index_up < MAX_EXPERTS) {
            scan_values[scan_index_up] = scan_values[scan_index_up] + scan_values[scan_index_up - 1];
        }
        int scan_index_up_0 = (THREADS + tid + 1) * 2 - 1;
        if (scan_index_up_0 < MAX_EXPERTS) {
            scan_values[scan_index_up_0] = scan_values[scan_index_up_0] + scan_values[scan_index_up_0 - 1];
        }
        __syncthreads();
        int scan_index_up_1 = (tid + 1) * 4 - 1;
        if (scan_index_up_1 < MAX_EXPERTS) {
            scan_values[scan_index_up_1] = scan_values[scan_index_up_1] + scan_values[scan_index_up_1 - 2];
        }
        int scan_index_up_2 = (THREADS + tid + 1) * 4 - 1;
        if (scan_index_up_2 < MAX_EXPERTS) {
            scan_values[scan_index_up_2] = scan_values[scan_index_up_2] + scan_values[scan_index_up_2 - 2];
        }
        __syncthreads();
        int scan_index_up_3 = (tid + 1) * 8 - 1;
        if (scan_index_up_3 < MAX_EXPERTS) {
            scan_values[scan_index_up_3] = scan_values[scan_index_up_3] + scan_values[scan_index_up_3 - 4];
        }
        int scan_index_up_4 = (THREADS + tid + 1) * 8 - 1;
        if (scan_index_up_4 < MAX_EXPERTS) {
            scan_values[scan_index_up_4] = scan_values[scan_index_up_4] + scan_values[scan_index_up_4 - 4];
        }
        __syncthreads();
        int scan_index_up_5 = (tid + 1) * 16 - 1;
        if (scan_index_up_5 < MAX_EXPERTS) {
            scan_values[scan_index_up_5] = scan_values[scan_index_up_5] + scan_values[scan_index_up_5 - 8];
        }
        int scan_index_up_6 = (THREADS + tid + 1) * 16 - 1;
        if (scan_index_up_6 < MAX_EXPERTS) {
            scan_values[scan_index_up_6] = scan_values[scan_index_up_6] + scan_values[scan_index_up_6 - 8];
        }
        __syncthreads();
        int scan_index_up_7 = (tid + 1) * 32 - 1;
        if (scan_index_up_7 < MAX_EXPERTS) {
            scan_values[scan_index_up_7] = scan_values[scan_index_up_7] + scan_values[scan_index_up_7 - 16];
        }
        int scan_index_up_8 = (THREADS + tid + 1) * 32 - 1;
        if (scan_index_up_8 < MAX_EXPERTS) {
            scan_values[scan_index_up_8] = scan_values[scan_index_up_8] + scan_values[scan_index_up_8 - 16];
        }
        __syncthreads();
        int scan_index_up_9 = (tid + 1) * 64 - 1;
        if (scan_index_up_9 < MAX_EXPERTS) {
            scan_values[scan_index_up_9] = scan_values[scan_index_up_9] + scan_values[scan_index_up_9 - 32];
        }
        int scan_index_up_10 = (THREADS + tid + 1) * 64 - 1;
        if (scan_index_up_10 < MAX_EXPERTS) {
            scan_values[scan_index_up_10] = scan_values[scan_index_up_10] + scan_values[scan_index_up_10 - 32];
        }
        __syncthreads();
        int scan_index_up_11 = (tid + 1) * 128 - 1;
        if (scan_index_up_11 < MAX_EXPERTS) {
            scan_values[scan_index_up_11] = scan_values[scan_index_up_11] + scan_values[scan_index_up_11 - 64];
        }
        int scan_index_up_12 = (THREADS + tid + 1) * 128 - 1;
        if (scan_index_up_12 < MAX_EXPERTS) {
            scan_values[scan_index_up_12] = scan_values[scan_index_up_12] + scan_values[scan_index_up_12 - 64];
        }
        __syncthreads();
        int scan_index_up_13 = (tid + 1) * 256 - 1;
        if (scan_index_up_13 < MAX_EXPERTS) {
            scan_values[scan_index_up_13] = scan_values[scan_index_up_13] + scan_values[scan_index_up_13 - 128];
        }
        int scan_index_up_14 = (THREADS + tid + 1) * 256 - 1;
        if (scan_index_up_14 < MAX_EXPERTS) {
            scan_values[scan_index_up_14] = scan_values[scan_index_up_14] + scan_values[scan_index_up_14 - 128];
        }
        __syncthreads();
        int scan_index_up_15 = (tid + 1) * 512 - 1;
        if (scan_index_up_15 < MAX_EXPERTS) {
            scan_values[scan_index_up_15] = scan_values[scan_index_up_15] + scan_values[scan_index_up_15 - 256];
        }
        int scan_index_up_16 = (THREADS + tid + 1) * 512 - 1;
        if (scan_index_up_16 < MAX_EXPERTS) {
            scan_values[scan_index_up_16] = scan_values[scan_index_up_16] + scan_values[scan_index_up_16 - 256];
        }
        __syncthreads();
        int scan_index_up_17 = (tid + 1) * 1024 - 1;
        if (scan_index_up_17 < MAX_EXPERTS) {
            scan_values[scan_index_up_17] = scan_values[scan_index_up_17] + scan_values[scan_index_up_17 - 512];
        }
        int scan_index_up_18 = (THREADS + tid + 1) * 1024 - 1;
        if (scan_index_up_18 < MAX_EXPERTS) {
            scan_values[scan_index_up_18] = scan_values[scan_index_up_18] + scan_values[scan_index_up_18 - 512];
        }
        __syncthreads();
        if (tid == 0) {
            int padded_total = scan_values[MAX_EXPERTS - 1];
            shared_total[0] = padded_total;
            scan_values[MAX_EXPERTS - 1] = 0;
        }
        __syncthreads();
        int scan_index_down = (tid + 1) * 1024 - 1;
        if (scan_index_down < MAX_EXPERTS) {
            int scan_left = scan_values[scan_index_down - 512];
            scan_values[scan_index_down - 512] = scan_values[scan_index_down];
            scan_values[scan_index_down] = scan_values[scan_index_down] + scan_left;
        }
        int scan_index_down_19 = (THREADS + tid + 1) * 1024 - 1;
        if (scan_index_down_19 < MAX_EXPERTS) {
            int scan_left_1 = scan_values[scan_index_down_19 - 512];
            scan_values[scan_index_down_19 - 512] = scan_values[scan_index_down_19];
            scan_values[scan_index_down_19] = scan_values[scan_index_down_19] + scan_left_1;
        }
        __syncthreads();
        int scan_index_down_20 = (tid + 1) * 512 - 1;
        if (scan_index_down_20 < MAX_EXPERTS) {
            int scan_left_2 = scan_values[scan_index_down_20 - 256];
            scan_values[scan_index_down_20 - 256] = scan_values[scan_index_down_20];
            scan_values[scan_index_down_20] = scan_values[scan_index_down_20] + scan_left_2;
        }
        int scan_index_down_21 = (THREADS + tid + 1) * 512 - 1;
        if (scan_index_down_21 < MAX_EXPERTS) {
            int scan_left_3 = scan_values[scan_index_down_21 - 256];
            scan_values[scan_index_down_21 - 256] = scan_values[scan_index_down_21];
            scan_values[scan_index_down_21] = scan_values[scan_index_down_21] + scan_left_3;
        }
        __syncthreads();
        int scan_index_down_22 = (tid + 1) * 256 - 1;
        if (scan_index_down_22 < MAX_EXPERTS) {
            int scan_left_4 = scan_values[scan_index_down_22 - 128];
            scan_values[scan_index_down_22 - 128] = scan_values[scan_index_down_22];
            scan_values[scan_index_down_22] = scan_values[scan_index_down_22] + scan_left_4;
        }
        int scan_index_down_23 = (THREADS + tid + 1) * 256 - 1;
        if (scan_index_down_23 < MAX_EXPERTS) {
            int scan_left_5 = scan_values[scan_index_down_23 - 128];
            scan_values[scan_index_down_23 - 128] = scan_values[scan_index_down_23];
            scan_values[scan_index_down_23] = scan_values[scan_index_down_23] + scan_left_5;
        }
        __syncthreads();
        int scan_index_down_24 = (tid + 1) * 128 - 1;
        if (scan_index_down_24 < MAX_EXPERTS) {
            int scan_left_6 = scan_values[scan_index_down_24 - 64];
            scan_values[scan_index_down_24 - 64] = scan_values[scan_index_down_24];
            scan_values[scan_index_down_24] = scan_values[scan_index_down_24] + scan_left_6;
        }
        int scan_index_down_25 = (THREADS + tid + 1) * 128 - 1;
        if (scan_index_down_25 < MAX_EXPERTS) {
            int scan_left_7 = scan_values[scan_index_down_25 - 64];
            scan_values[scan_index_down_25 - 64] = scan_values[scan_index_down_25];
            scan_values[scan_index_down_25] = scan_values[scan_index_down_25] + scan_left_7;
        }
        __syncthreads();
        int scan_index_down_26 = (tid + 1) * 64 - 1;
        if (scan_index_down_26 < MAX_EXPERTS) {
            int scan_left_8 = scan_values[scan_index_down_26 - 32];
            scan_values[scan_index_down_26 - 32] = scan_values[scan_index_down_26];
            scan_values[scan_index_down_26] = scan_values[scan_index_down_26] + scan_left_8;
        }
        int scan_index_down_27 = (THREADS + tid + 1) * 64 - 1;
        if (scan_index_down_27 < MAX_EXPERTS) {
            int scan_left_9 = scan_values[scan_index_down_27 - 32];
            scan_values[scan_index_down_27 - 32] = scan_values[scan_index_down_27];
            scan_values[scan_index_down_27] = scan_values[scan_index_down_27] + scan_left_9;
        }
        __syncthreads();
        int scan_index_down_28 = (tid + 1) * 32 - 1;
        if (scan_index_down_28 < MAX_EXPERTS) {
            int scan_left_10 = scan_values[scan_index_down_28 - 16];
            scan_values[scan_index_down_28 - 16] = scan_values[scan_index_down_28];
            scan_values[scan_index_down_28] = scan_values[scan_index_down_28] + scan_left_10;
        }
        int scan_index_down_29 = (THREADS + tid + 1) * 32 - 1;
        if (scan_index_down_29 < MAX_EXPERTS) {
            int scan_left_11 = scan_values[scan_index_down_29 - 16];
            scan_values[scan_index_down_29 - 16] = scan_values[scan_index_down_29];
            scan_values[scan_index_down_29] = scan_values[scan_index_down_29] + scan_left_11;
        }
        __syncthreads();
        int scan_index_down_30 = (tid + 1) * 16 - 1;
        if (scan_index_down_30 < MAX_EXPERTS) {
            int scan_left_12 = scan_values[scan_index_down_30 - 8];
            scan_values[scan_index_down_30 - 8] = scan_values[scan_index_down_30];
            scan_values[scan_index_down_30] = scan_values[scan_index_down_30] + scan_left_12;
        }
        int scan_index_down_31 = (THREADS + tid + 1) * 16 - 1;
        if (scan_index_down_31 < MAX_EXPERTS) {
            int scan_left_13 = scan_values[scan_index_down_31 - 8];
            scan_values[scan_index_down_31 - 8] = scan_values[scan_index_down_31];
            scan_values[scan_index_down_31] = scan_values[scan_index_down_31] + scan_left_13;
        }
        __syncthreads();
        int scan_index_down_32 = (tid + 1) * 8 - 1;
        if (scan_index_down_32 < MAX_EXPERTS) {
            int scan_left_14 = scan_values[scan_index_down_32 - 4];
            scan_values[scan_index_down_32 - 4] = scan_values[scan_index_down_32];
            scan_values[scan_index_down_32] = scan_values[scan_index_down_32] + scan_left_14;
        }
        int scan_index_down_33 = (THREADS + tid + 1) * 8 - 1;
        if (scan_index_down_33 < MAX_EXPERTS) {
            int scan_left_15 = scan_values[scan_index_down_33 - 4];
            scan_values[scan_index_down_33 - 4] = scan_values[scan_index_down_33];
            scan_values[scan_index_down_33] = scan_values[scan_index_down_33] + scan_left_15;
        }
        __syncthreads();
        int scan_index_down_34 = (tid + 1) * 4 - 1;
        if (scan_index_down_34 < MAX_EXPERTS) {
            int scan_left_16 = scan_values[scan_index_down_34 - 2];
            scan_values[scan_index_down_34 - 2] = scan_values[scan_index_down_34];
            scan_values[scan_index_down_34] = scan_values[scan_index_down_34] + scan_left_16;
        }
        int scan_index_down_35 = (THREADS + tid + 1) * 4 - 1;
        if (scan_index_down_35 < MAX_EXPERTS) {
            int scan_left_17 = scan_values[scan_index_down_35 - 2];
            scan_values[scan_index_down_35 - 2] = scan_values[scan_index_down_35];
            scan_values[scan_index_down_35] = scan_values[scan_index_down_35] + scan_left_17;
        }
        __syncthreads();
        int scan_index_down_36 = (tid + 1) * 2 - 1;
        if (scan_index_down_36 < MAX_EXPERTS) {
            int scan_left_18 = scan_values[scan_index_down_36 - 1];
            scan_values[scan_index_down_36 - 1] = scan_values[scan_index_down_36];
            scan_values[scan_index_down_36] = scan_values[scan_index_down_36] + scan_left_18;
        }
        int scan_index_down_37 = (THREADS + tid + 1) * 2 - 1;
        if (scan_index_down_37 < MAX_EXPERTS) {
            int scan_left_19 = scan_values[scan_index_down_37 - 1];
            scan_values[scan_index_down_37 - 1] = scan_values[scan_index_down_37];
            scan_values[scan_index_down_37] = scan_values[scan_index_down_37] + scan_left_19;
        }
        __syncthreads();
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
                if (local_row % block_m == 0) {
                    expert_ids[grouped_row / block_m] = pair_expert;
                }
            }
        }
        for (int padding_expert = tid; padding_expert < E; padding_expert += THREADS) {
            int count_final = shared_counts[padding_expert];
            int expert_start = shared_offsets[padding_expert];
            int padded_count_final = 0;
            {
                padded_count_final = (count_final + block_m - 1) / block_m * block_m;
            }
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

} // extern "C"


constexpr int kThreads = THREADS;
constexpr int kSmemTotal = SMEM_TOTAL;
constexpr int kWarps = NUM_WARPS;
static_assert(THREADS == NUM_WARPS * 32);
static_assert(MAX_EXPERTS == 512);
#undef FAST_BLOCK_ALIGN
#undef LOOM_INF
#undef MAX_BLOCK_M
#undef MAX_EXPERTS
#undef MAX_TOP_K
#undef NUM_MAIN_STAGES
#undef NUM_WARPS
#undef PUBLIC_SHARED_SOFTMAX
#undef SINGLE_CTA
#undef SMEM_SCAN_VALUES_OFF
#undef SMEM_SCAN_VALUES_STAGE_BYTES
#undef SMEM_SCAN_VALUES_STRIDE
#undef SMEM_SHARED_COUNTS_OFF
#undef SMEM_SHARED_COUNTS_STAGE_BYTES
#undef SMEM_SHARED_COUNTS_STRIDE
#undef SMEM_SHARED_IDS_OFF
#undef SMEM_SHARED_IDS_STAGE_BYTES
#undef SMEM_SHARED_IDS_STRIDE
#undef SMEM_SHARED_OFFSETS_OFF
#undef SMEM_SHARED_OFFSETS_STAGE_BYTES
#undef SMEM_SHARED_OFFSETS_STRIDE
#undef SMEM_SHARED_SCATTER_OFF
#undef SMEM_SHARED_SCATTER_STAGE_BYTES
#undef SMEM_SHARED_SCATTER_STRIDE
#undef SMEM_SHARED_TOTAL_OFF
#undef SMEM_SHARED_TOTAL_STAGE_BYTES
#undef SMEM_SHARED_TOTAL_STRIDE
#undef SMEM_TOTAL
#undef THREADS
}  // namespace alphamoe_router_small_routed_generated

using alphamoe_router_large_routed_generated::kernel_alpha_moe_fused_router_routed;
static_assert(alphamoe_router_large_routed_generated::kThreads == alphamoe_router_large_generated::kThreads);
static_assert(alphamoe_router_large_routed_generated::kSmemTotal == alphamoe_router_large_generated::kSmemTotal);
static_assert(alphamoe_router_large_routed_generated::kWarps == alphamoe_router_large_generated::kWarps);
using alphamoe_router_small_routed_generated::kernel_alpha_moe_fused_router_small_routed;
static_assert(alphamoe_router_small_routed_generated::kThreads == alphamoe_router_small_generated::kThreads);
static_assert(alphamoe_router_small_routed_generated::kSmemTotal == alphamoe_router_small_generated::kSmemTotal);
static_assert(alphamoe_router_small_routed_generated::kWarps == alphamoe_router_small_generated::kWarps);


namespace alphamoe_router_tiny_generated {
typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) LoomTensorMap { uint64_t opaque[16]; };
struct __align__(64) LoomTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(LoomTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(LoomTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else

#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(LoomTensorMap) >= alignof(CUtensorMap), "LoomTensorMap alignment must cover the CUtensorMap CUDA ABI");


__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
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
#define SMEM_SCAN_VALUES_OFF 4096
#define SMEM_SCAN_VALUES_STAGE_BYTES 4096
#define SMEM_SCAN_VALUES_STRIDE 4096
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
#define FAST_BLOCK_ALIGN 1
#define SINGLE_CTA 1


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

__global__ __launch_bounds__(256) void
kernel_alpha_moe_fused_router_tiny(float* __restrict__ logits, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M, int E, int top_k, int block_m, int has_shared_expert)
{
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
    int* scan_values = reinterpret_cast<int*>(smem_raw + 4096);
    const int scan_values_addr = smem + 4096;
    int* shared_scatter = reinterpret_cast<int*>(smem_raw + 8196);
    const int shared_scatter_addr = smem + 8196;
    int* shared_ids = reinterpret_cast<int*>(smem_raw + 12292);
    const int shared_ids_addr = smem + 12292;
    int* shared_total = reinterpret_cast<int*>(smem_raw + 20484);
    const int shared_total_addr = smem + 20484;

    // === Task calls (dependency order) ===
    int routed_experts = E - has_shared_expert;
    int routed_top_k = top_k - has_shared_expert;
    for (int token = (unsigned int)(bid * NUM_WARPS) + warp; token < M; token += num_bids * NUM_WARPS) {
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
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(local_key));
            unsigned int best_key = _warp_redux_u32_0;
            unsigned int best_bits = best_key ^ 2147483648;
            if ((best_key & 2147483648) == 0) {
                best_bits = best_key ^ 4294967295;
            }
            float best_value = 0.0f;
            best_value = reinterpret_cast<float*>(&best_bits)[0];
            unsigned int tied_index = MAX_EXPERTS;
            #pragma unroll
            for (int expert_slot_tie = 0; expert_slot_tie < MAX_EXPERTS / 32; expert_slot_tie++) {
                int expert_tie = lane + (unsigned int)((MAX_EXPERTS / 32 - 1 - expert_slot_tie) * 32);
                if (expert_tie < routed_experts) {
                    if (row_values[MAX_EXPERTS / 32 - 1 - expert_slot_tie] == best_value) {
                        tied_index = (unsigned int)expert_tie;
                    }
                }
            }
            unsigned int _warp_redux_u32_1;
            asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_1) : "r"(tied_index));
            int best_index = (int)_warp_redux_u32_1;
            if (lane == (unsigned int)route) {
                selected_logit = best_value;
                selected_expert = best_index;
            }
            #pragma unroll
            for (int expert_slot_remove = 0; expert_slot_remove < MAX_EXPERTS / 32; expert_slot_remove++) {
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
            _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
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
        __syncthreads();
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
        #pragma unroll
        for (int expert_slot_scan_init = 0; expert_slot_scan_init < MAX_EXPERTS / THREADS; expert_slot_scan_init++) {
            int expert_scan_init = tid + expert_slot_scan_init * THREADS;
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
            scan_values[expert_scan_init] = padded_count_init;
        }
        __syncthreads();
        int scan_index_up = (tid + 1) * 2 - 1;
        if (scan_index_up < MAX_EXPERTS) {
            scan_values[scan_index_up] = scan_values[scan_index_up] + scan_values[scan_index_up - 1];
        }
        int scan_index_up_0 = (THREADS + tid + 1) * 2 - 1;
        if (scan_index_up_0 < MAX_EXPERTS) {
            scan_values[scan_index_up_0] = scan_values[scan_index_up_0] + scan_values[scan_index_up_0 - 1];
        }
        __syncthreads();
        int scan_index_up_1 = (tid + 1) * 4 - 1;
        if (scan_index_up_1 < MAX_EXPERTS) {
            scan_values[scan_index_up_1] = scan_values[scan_index_up_1] + scan_values[scan_index_up_1 - 2];
        }
        int scan_index_up_2 = (THREADS + tid + 1) * 4 - 1;
        if (scan_index_up_2 < MAX_EXPERTS) {
            scan_values[scan_index_up_2] = scan_values[scan_index_up_2] + scan_values[scan_index_up_2 - 2];
        }
        __syncthreads();
        int scan_index_up_3 = (tid + 1) * 8 - 1;
        if (scan_index_up_3 < MAX_EXPERTS) {
            scan_values[scan_index_up_3] = scan_values[scan_index_up_3] + scan_values[scan_index_up_3 - 4];
        }
        int scan_index_up_4 = (THREADS + tid + 1) * 8 - 1;
        if (scan_index_up_4 < MAX_EXPERTS) {
            scan_values[scan_index_up_4] = scan_values[scan_index_up_4] + scan_values[scan_index_up_4 - 4];
        }
        __syncthreads();
        int scan_index_up_5 = (tid + 1) * 16 - 1;
        if (scan_index_up_5 < MAX_EXPERTS) {
            scan_values[scan_index_up_5] = scan_values[scan_index_up_5] + scan_values[scan_index_up_5 - 8];
        }
        int scan_index_up_6 = (THREADS + tid + 1) * 16 - 1;
        if (scan_index_up_6 < MAX_EXPERTS) {
            scan_values[scan_index_up_6] = scan_values[scan_index_up_6] + scan_values[scan_index_up_6 - 8];
        }
        __syncthreads();
        int scan_index_up_7 = (tid + 1) * 32 - 1;
        if (scan_index_up_7 < MAX_EXPERTS) {
            scan_values[scan_index_up_7] = scan_values[scan_index_up_7] + scan_values[scan_index_up_7 - 16];
        }
        int scan_index_up_8 = (THREADS + tid + 1) * 32 - 1;
        if (scan_index_up_8 < MAX_EXPERTS) {
            scan_values[scan_index_up_8] = scan_values[scan_index_up_8] + scan_values[scan_index_up_8 - 16];
        }
        __syncthreads();
        int scan_index_up_9 = (tid + 1) * 64 - 1;
        if (scan_index_up_9 < MAX_EXPERTS) {
            scan_values[scan_index_up_9] = scan_values[scan_index_up_9] + scan_values[scan_index_up_9 - 32];
        }
        int scan_index_up_10 = (THREADS + tid + 1) * 64 - 1;
        if (scan_index_up_10 < MAX_EXPERTS) {
            scan_values[scan_index_up_10] = scan_values[scan_index_up_10] + scan_values[scan_index_up_10 - 32];
        }
        __syncthreads();
        int scan_index_up_11 = (tid + 1) * 128 - 1;
        if (scan_index_up_11 < MAX_EXPERTS) {
            scan_values[scan_index_up_11] = scan_values[scan_index_up_11] + scan_values[scan_index_up_11 - 64];
        }
        int scan_index_up_12 = (THREADS + tid + 1) * 128 - 1;
        if (scan_index_up_12 < MAX_EXPERTS) {
            scan_values[scan_index_up_12] = scan_values[scan_index_up_12] + scan_values[scan_index_up_12 - 64];
        }
        __syncthreads();
        int scan_index_up_13 = (tid + 1) * 256 - 1;
        if (scan_index_up_13 < MAX_EXPERTS) {
            scan_values[scan_index_up_13] = scan_values[scan_index_up_13] + scan_values[scan_index_up_13 - 128];
        }
        int scan_index_up_14 = (THREADS + tid + 1) * 256 - 1;
        if (scan_index_up_14 < MAX_EXPERTS) {
            scan_values[scan_index_up_14] = scan_values[scan_index_up_14] + scan_values[scan_index_up_14 - 128];
        }
        __syncthreads();
        int scan_index_up_15 = (tid + 1) * 512 - 1;
        if (scan_index_up_15 < MAX_EXPERTS) {
            scan_values[scan_index_up_15] = scan_values[scan_index_up_15] + scan_values[scan_index_up_15 - 256];
        }
        int scan_index_up_16 = (THREADS + tid + 1) * 512 - 1;
        if (scan_index_up_16 < MAX_EXPERTS) {
            scan_values[scan_index_up_16] = scan_values[scan_index_up_16] + scan_values[scan_index_up_16 - 256];
        }
        __syncthreads();
        int scan_index_up_17 = (tid + 1) * 1024 - 1;
        if (scan_index_up_17 < MAX_EXPERTS) {
            scan_values[scan_index_up_17] = scan_values[scan_index_up_17] + scan_values[scan_index_up_17 - 512];
        }
        int scan_index_up_18 = (THREADS + tid + 1) * 1024 - 1;
        if (scan_index_up_18 < MAX_EXPERTS) {
            scan_values[scan_index_up_18] = scan_values[scan_index_up_18] + scan_values[scan_index_up_18 - 512];
        }
        __syncthreads();
        if (tid == 0) {
            int padded_total = scan_values[MAX_EXPERTS - 1];
            shared_total[0] = padded_total;
            scan_values[MAX_EXPERTS - 1] = 0;
        }
        __syncthreads();
        int scan_index_down = (tid + 1) * 1024 - 1;
        if (scan_index_down < MAX_EXPERTS) {
            int scan_left = scan_values[scan_index_down - 512];
            scan_values[scan_index_down - 512] = scan_values[scan_index_down];
            scan_values[scan_index_down] = scan_values[scan_index_down] + scan_left;
        }
        int scan_index_down_19 = (THREADS + tid + 1) * 1024 - 1;
        if (scan_index_down_19 < MAX_EXPERTS) {
            int scan_left_1 = scan_values[scan_index_down_19 - 512];
            scan_values[scan_index_down_19 - 512] = scan_values[scan_index_down_19];
            scan_values[scan_index_down_19] = scan_values[scan_index_down_19] + scan_left_1;
        }
        __syncthreads();
        int scan_index_down_20 = (tid + 1) * 512 - 1;
        if (scan_index_down_20 < MAX_EXPERTS) {
            int scan_left_2 = scan_values[scan_index_down_20 - 256];
            scan_values[scan_index_down_20 - 256] = scan_values[scan_index_down_20];
            scan_values[scan_index_down_20] = scan_values[scan_index_down_20] + scan_left_2;
        }
        int scan_index_down_21 = (THREADS + tid + 1) * 512 - 1;
        if (scan_index_down_21 < MAX_EXPERTS) {
            int scan_left_3 = scan_values[scan_index_down_21 - 256];
            scan_values[scan_index_down_21 - 256] = scan_values[scan_index_down_21];
            scan_values[scan_index_down_21] = scan_values[scan_index_down_21] + scan_left_3;
        }
        __syncthreads();
        int scan_index_down_22 = (tid + 1) * 256 - 1;
        if (scan_index_down_22 < MAX_EXPERTS) {
            int scan_left_4 = scan_values[scan_index_down_22 - 128];
            scan_values[scan_index_down_22 - 128] = scan_values[scan_index_down_22];
            scan_values[scan_index_down_22] = scan_values[scan_index_down_22] + scan_left_4;
        }
        int scan_index_down_23 = (THREADS + tid + 1) * 256 - 1;
        if (scan_index_down_23 < MAX_EXPERTS) {
            int scan_left_5 = scan_values[scan_index_down_23 - 128];
            scan_values[scan_index_down_23 - 128] = scan_values[scan_index_down_23];
            scan_values[scan_index_down_23] = scan_values[scan_index_down_23] + scan_left_5;
        }
        __syncthreads();
        int scan_index_down_24 = (tid + 1) * 128 - 1;
        if (scan_index_down_24 < MAX_EXPERTS) {
            int scan_left_6 = scan_values[scan_index_down_24 - 64];
            scan_values[scan_index_down_24 - 64] = scan_values[scan_index_down_24];
            scan_values[scan_index_down_24] = scan_values[scan_index_down_24] + scan_left_6;
        }
        int scan_index_down_25 = (THREADS + tid + 1) * 128 - 1;
        if (scan_index_down_25 < MAX_EXPERTS) {
            int scan_left_7 = scan_values[scan_index_down_25 - 64];
            scan_values[scan_index_down_25 - 64] = scan_values[scan_index_down_25];
            scan_values[scan_index_down_25] = scan_values[scan_index_down_25] + scan_left_7;
        }
        __syncthreads();
        int scan_index_down_26 = (tid + 1) * 64 - 1;
        if (scan_index_down_26 < MAX_EXPERTS) {
            int scan_left_8 = scan_values[scan_index_down_26 - 32];
            scan_values[scan_index_down_26 - 32] = scan_values[scan_index_down_26];
            scan_values[scan_index_down_26] = scan_values[scan_index_down_26] + scan_left_8;
        }
        int scan_index_down_27 = (THREADS + tid + 1) * 64 - 1;
        if (scan_index_down_27 < MAX_EXPERTS) {
            int scan_left_9 = scan_values[scan_index_down_27 - 32];
            scan_values[scan_index_down_27 - 32] = scan_values[scan_index_down_27];
            scan_values[scan_index_down_27] = scan_values[scan_index_down_27] + scan_left_9;
        }
        __syncthreads();
        int scan_index_down_28 = (tid + 1) * 32 - 1;
        if (scan_index_down_28 < MAX_EXPERTS) {
            int scan_left_10 = scan_values[scan_index_down_28 - 16];
            scan_values[scan_index_down_28 - 16] = scan_values[scan_index_down_28];
            scan_values[scan_index_down_28] = scan_values[scan_index_down_28] + scan_left_10;
        }
        int scan_index_down_29 = (THREADS + tid + 1) * 32 - 1;
        if (scan_index_down_29 < MAX_EXPERTS) {
            int scan_left_11 = scan_values[scan_index_down_29 - 16];
            scan_values[scan_index_down_29 - 16] = scan_values[scan_index_down_29];
            scan_values[scan_index_down_29] = scan_values[scan_index_down_29] + scan_left_11;
        }
        __syncthreads();
        int scan_index_down_30 = (tid + 1) * 16 - 1;
        if (scan_index_down_30 < MAX_EXPERTS) {
            int scan_left_12 = scan_values[scan_index_down_30 - 8];
            scan_values[scan_index_down_30 - 8] = scan_values[scan_index_down_30];
            scan_values[scan_index_down_30] = scan_values[scan_index_down_30] + scan_left_12;
        }
        int scan_index_down_31 = (THREADS + tid + 1) * 16 - 1;
        if (scan_index_down_31 < MAX_EXPERTS) {
            int scan_left_13 = scan_values[scan_index_down_31 - 8];
            scan_values[scan_index_down_31 - 8] = scan_values[scan_index_down_31];
            scan_values[scan_index_down_31] = scan_values[scan_index_down_31] + scan_left_13;
        }
        __syncthreads();
        int scan_index_down_32 = (tid + 1) * 8 - 1;
        if (scan_index_down_32 < MAX_EXPERTS) {
            int scan_left_14 = scan_values[scan_index_down_32 - 4];
            scan_values[scan_index_down_32 - 4] = scan_values[scan_index_down_32];
            scan_values[scan_index_down_32] = scan_values[scan_index_down_32] + scan_left_14;
        }
        int scan_index_down_33 = (THREADS + tid + 1) * 8 - 1;
        if (scan_index_down_33 < MAX_EXPERTS) {
            int scan_left_15 = scan_values[scan_index_down_33 - 4];
            scan_values[scan_index_down_33 - 4] = scan_values[scan_index_down_33];
            scan_values[scan_index_down_33] = scan_values[scan_index_down_33] + scan_left_15;
        }
        __syncthreads();
        int scan_index_down_34 = (tid + 1) * 4 - 1;
        if (scan_index_down_34 < MAX_EXPERTS) {
            int scan_left_16 = scan_values[scan_index_down_34 - 2];
            scan_values[scan_index_down_34 - 2] = scan_values[scan_index_down_34];
            scan_values[scan_index_down_34] = scan_values[scan_index_down_34] + scan_left_16;
        }
        int scan_index_down_35 = (THREADS + tid + 1) * 4 - 1;
        if (scan_index_down_35 < MAX_EXPERTS) {
            int scan_left_17 = scan_values[scan_index_down_35 - 2];
            scan_values[scan_index_down_35 - 2] = scan_values[scan_index_down_35];
            scan_values[scan_index_down_35] = scan_values[scan_index_down_35] + scan_left_17;
        }
        __syncthreads();
        int scan_index_down_36 = (tid + 1) * 2 - 1;
        if (scan_index_down_36 < MAX_EXPERTS) {
            int scan_left_18 = scan_values[scan_index_down_36 - 1];
            scan_values[scan_index_down_36 - 1] = scan_values[scan_index_down_36];
            scan_values[scan_index_down_36] = scan_values[scan_index_down_36] + scan_left_18;
        }
        int scan_index_down_37 = (THREADS + tid + 1) * 2 - 1;
        if (scan_index_down_37 < MAX_EXPERTS) {
            int scan_left_19 = scan_values[scan_index_down_37 - 1];
            scan_values[scan_index_down_37 - 1] = scan_values[scan_index_down_37];
            scan_values[scan_index_down_37] = scan_values[scan_index_down_37] + scan_left_19;
        }
        __syncthreads();
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
            int padded_count_final = 0;
            {
                if (block_m == 8) {
                    padded_count_final = count_final + 7 & -8;
                } else if (block_m == 16) {
                    padded_count_final = count_final + 15 & -16;
                } else {
                    padded_count_final = (count_final + block_m - 1) / block_m * block_m;
                }
            }
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

} // extern "C"


constexpr int kThreads = THREADS;
constexpr int kSmemTotal = SMEM_TOTAL;
constexpr int kWarps = NUM_WARPS;
static_assert(THREADS == NUM_WARPS * 32);
static_assert(MAX_EXPERTS == 512);
#undef FAST_BLOCK_ALIGN
#undef LOOM_INF
#undef MAX_BLOCK_M
#undef MAX_EXPERTS
#undef MAX_TOP_K
#undef NUM_MAIN_STAGES
#undef NUM_WARPS
#undef PUBLIC_SHARED_SOFTMAX
#undef SINGLE_CTA
#undef SMEM_SCAN_VALUES_OFF
#undef SMEM_SCAN_VALUES_STAGE_BYTES
#undef SMEM_SCAN_VALUES_STRIDE
#undef SMEM_SHARED_COUNTS_OFF
#undef SMEM_SHARED_COUNTS_STAGE_BYTES
#undef SMEM_SHARED_COUNTS_STRIDE
#undef SMEM_SHARED_IDS_OFF
#undef SMEM_SHARED_IDS_STAGE_BYTES
#undef SMEM_SHARED_IDS_STRIDE
#undef SMEM_SHARED_OFFSETS_OFF
#undef SMEM_SHARED_OFFSETS_STAGE_BYTES
#undef SMEM_SHARED_OFFSETS_STRIDE
#undef SMEM_SHARED_SCATTER_OFF
#undef SMEM_SHARED_SCATTER_STAGE_BYTES
#undef SMEM_SHARED_SCATTER_STRIDE
#undef SMEM_SHARED_TOTAL_OFF
#undef SMEM_SHARED_TOTAL_STAGE_BYTES
#undef SMEM_SHARED_TOTAL_STRIDE
#undef SMEM_TOTAL
#undef THREADS
}  // namespace alphamoe_router_tiny_generated

using alphamoe_router_tiny_generated::kernel_alpha_moe_fused_router_tiny;

static_assert(alphamoe_router_tiny_generated::kThreads == alphamoe_router_small_generated::kThreads);

static_assert(alphamoe_router_tiny_generated::kSmemTotal == alphamoe_router_small_generated::kSmemTotal);

static_assert(alphamoe_router_tiny_generated::kWarps == alphamoe_router_small_generated::kWarps);

namespace alphamoe_router_tiny_routed_generated {
typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) LoomTensorMap { uint64_t opaque[16]; };
struct __align__(64) LoomTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(LoomTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(LoomTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else

#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(LoomTensorMap) >= alignof(CUtensorMap), "LoomTensorMap alignment must cover the CUtensorMap CUDA ABI");


__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
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
#define SMEM_SCAN_VALUES_OFF 4096
#define SMEM_SCAN_VALUES_STAGE_BYTES 4096
#define SMEM_SCAN_VALUES_STRIDE 4096
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
#define PUBLIC_SHARED_SOFTMAX 0
#define FAST_BLOCK_ALIGN 1
#define SINGLE_CTA 1


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

__global__ __launch_bounds__(256) void
kernel_alpha_moe_fused_router_tiny_routed(float* __restrict__ logits, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M, int E, int top_k, int block_m, int has_shared_expert)
{
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
    int* scan_values = reinterpret_cast<int*>(smem_raw + 4096);
    const int scan_values_addr = smem + 4096;
    int* shared_scatter = reinterpret_cast<int*>(smem_raw + 8196);
    const int shared_scatter_addr = smem + 8196;
    int* shared_ids = reinterpret_cast<int*>(smem_raw + 12292);
    const int shared_ids_addr = smem + 12292;
    int* shared_total = reinterpret_cast<int*>(smem_raw + 20484);
    const int shared_total_addr = smem + 20484;

    // === Task calls (dependency order) ===
    int routed_experts = E - has_shared_expert;
    int routed_top_k = top_k - has_shared_expert;
    for (int token = (unsigned int)(bid * NUM_WARPS) + warp; token < M; token += num_bids * NUM_WARPS) {
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
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(local_key));
            unsigned int best_key = _warp_redux_u32_0;
            unsigned int best_bits = best_key ^ 2147483648;
            if ((best_key & 2147483648) == 0) {
                best_bits = best_key ^ 4294967295;
            }
            float best_value = 0.0f;
            best_value = reinterpret_cast<float*>(&best_bits)[0];
            unsigned int tied_index = MAX_EXPERTS;
            #pragma unroll
            for (int expert_slot_tie = 0; expert_slot_tie < MAX_EXPERTS / 32; expert_slot_tie++) {
                int expert_tie = lane + (unsigned int)((MAX_EXPERTS / 32 - 1 - expert_slot_tie) * 32);
                if (expert_tie < routed_experts) {
                    if (row_values[MAX_EXPERTS / 32 - 1 - expert_slot_tie] == best_value) {
                        tied_index = (unsigned int)expert_tie;
                    }
                }
            }
            unsigned int _warp_redux_u32_1;
            asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_1) : "r"(tied_index));
            int best_index = (int)_warp_redux_u32_1;
            if (lane == (unsigned int)route) {
                selected_logit = best_value;
                selected_expert = best_index;
            }
            #pragma unroll
            for (int expert_slot_remove = 0; expert_slot_remove < MAX_EXPERTS / 32; expert_slot_remove++) {
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
        float selected_for_max = -LOOM_INF;
        if ((unsigned int)softmax_top_k > lane) {
            selected_for_max = selected_logit;
        }
        float _warp_reduce_0 = selected_for_max;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
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
        __syncthreads();
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
        #pragma unroll
        for (int expert_slot_scan_init = 0; expert_slot_scan_init < MAX_EXPERTS / THREADS; expert_slot_scan_init++) {
            int expert_scan_init = tid + expert_slot_scan_init * THREADS;
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
            scan_values[expert_scan_init] = padded_count_init;
        }
        __syncthreads();
        int scan_index_up = (tid + 1) * 2 - 1;
        if (scan_index_up < MAX_EXPERTS) {
            scan_values[scan_index_up] = scan_values[scan_index_up] + scan_values[scan_index_up - 1];
        }
        int scan_index_up_0 = (THREADS + tid + 1) * 2 - 1;
        if (scan_index_up_0 < MAX_EXPERTS) {
            scan_values[scan_index_up_0] = scan_values[scan_index_up_0] + scan_values[scan_index_up_0 - 1];
        }
        __syncthreads();
        int scan_index_up_1 = (tid + 1) * 4 - 1;
        if (scan_index_up_1 < MAX_EXPERTS) {
            scan_values[scan_index_up_1] = scan_values[scan_index_up_1] + scan_values[scan_index_up_1 - 2];
        }
        int scan_index_up_2 = (THREADS + tid + 1) * 4 - 1;
        if (scan_index_up_2 < MAX_EXPERTS) {
            scan_values[scan_index_up_2] = scan_values[scan_index_up_2] + scan_values[scan_index_up_2 - 2];
        }
        __syncthreads();
        int scan_index_up_3 = (tid + 1) * 8 - 1;
        if (scan_index_up_3 < MAX_EXPERTS) {
            scan_values[scan_index_up_3] = scan_values[scan_index_up_3] + scan_values[scan_index_up_3 - 4];
        }
        int scan_index_up_4 = (THREADS + tid + 1) * 8 - 1;
        if (scan_index_up_4 < MAX_EXPERTS) {
            scan_values[scan_index_up_4] = scan_values[scan_index_up_4] + scan_values[scan_index_up_4 - 4];
        }
        __syncthreads();
        int scan_index_up_5 = (tid + 1) * 16 - 1;
        if (scan_index_up_5 < MAX_EXPERTS) {
            scan_values[scan_index_up_5] = scan_values[scan_index_up_5] + scan_values[scan_index_up_5 - 8];
        }
        int scan_index_up_6 = (THREADS + tid + 1) * 16 - 1;
        if (scan_index_up_6 < MAX_EXPERTS) {
            scan_values[scan_index_up_6] = scan_values[scan_index_up_6] + scan_values[scan_index_up_6 - 8];
        }
        __syncthreads();
        int scan_index_up_7 = (tid + 1) * 32 - 1;
        if (scan_index_up_7 < MAX_EXPERTS) {
            scan_values[scan_index_up_7] = scan_values[scan_index_up_7] + scan_values[scan_index_up_7 - 16];
        }
        int scan_index_up_8 = (THREADS + tid + 1) * 32 - 1;
        if (scan_index_up_8 < MAX_EXPERTS) {
            scan_values[scan_index_up_8] = scan_values[scan_index_up_8] + scan_values[scan_index_up_8 - 16];
        }
        __syncthreads();
        int scan_index_up_9 = (tid + 1) * 64 - 1;
        if (scan_index_up_9 < MAX_EXPERTS) {
            scan_values[scan_index_up_9] = scan_values[scan_index_up_9] + scan_values[scan_index_up_9 - 32];
        }
        int scan_index_up_10 = (THREADS + tid + 1) * 64 - 1;
        if (scan_index_up_10 < MAX_EXPERTS) {
            scan_values[scan_index_up_10] = scan_values[scan_index_up_10] + scan_values[scan_index_up_10 - 32];
        }
        __syncthreads();
        int scan_index_up_11 = (tid + 1) * 128 - 1;
        if (scan_index_up_11 < MAX_EXPERTS) {
            scan_values[scan_index_up_11] = scan_values[scan_index_up_11] + scan_values[scan_index_up_11 - 64];
        }
        int scan_index_up_12 = (THREADS + tid + 1) * 128 - 1;
        if (scan_index_up_12 < MAX_EXPERTS) {
            scan_values[scan_index_up_12] = scan_values[scan_index_up_12] + scan_values[scan_index_up_12 - 64];
        }
        __syncthreads();
        int scan_index_up_13 = (tid + 1) * 256 - 1;
        if (scan_index_up_13 < MAX_EXPERTS) {
            scan_values[scan_index_up_13] = scan_values[scan_index_up_13] + scan_values[scan_index_up_13 - 128];
        }
        int scan_index_up_14 = (THREADS + tid + 1) * 256 - 1;
        if (scan_index_up_14 < MAX_EXPERTS) {
            scan_values[scan_index_up_14] = scan_values[scan_index_up_14] + scan_values[scan_index_up_14 - 128];
        }
        __syncthreads();
        int scan_index_up_15 = (tid + 1) * 512 - 1;
        if (scan_index_up_15 < MAX_EXPERTS) {
            scan_values[scan_index_up_15] = scan_values[scan_index_up_15] + scan_values[scan_index_up_15 - 256];
        }
        int scan_index_up_16 = (THREADS + tid + 1) * 512 - 1;
        if (scan_index_up_16 < MAX_EXPERTS) {
            scan_values[scan_index_up_16] = scan_values[scan_index_up_16] + scan_values[scan_index_up_16 - 256];
        }
        __syncthreads();
        int scan_index_up_17 = (tid + 1) * 1024 - 1;
        if (scan_index_up_17 < MAX_EXPERTS) {
            scan_values[scan_index_up_17] = scan_values[scan_index_up_17] + scan_values[scan_index_up_17 - 512];
        }
        int scan_index_up_18 = (THREADS + tid + 1) * 1024 - 1;
        if (scan_index_up_18 < MAX_EXPERTS) {
            scan_values[scan_index_up_18] = scan_values[scan_index_up_18] + scan_values[scan_index_up_18 - 512];
        }
        __syncthreads();
        if (tid == 0) {
            int padded_total = scan_values[MAX_EXPERTS - 1];
            shared_total[0] = padded_total;
            scan_values[MAX_EXPERTS - 1] = 0;
        }
        __syncthreads();
        int scan_index_down = (tid + 1) * 1024 - 1;
        if (scan_index_down < MAX_EXPERTS) {
            int scan_left = scan_values[scan_index_down - 512];
            scan_values[scan_index_down - 512] = scan_values[scan_index_down];
            scan_values[scan_index_down] = scan_values[scan_index_down] + scan_left;
        }
        int scan_index_down_19 = (THREADS + tid + 1) * 1024 - 1;
        if (scan_index_down_19 < MAX_EXPERTS) {
            int scan_left_1 = scan_values[scan_index_down_19 - 512];
            scan_values[scan_index_down_19 - 512] = scan_values[scan_index_down_19];
            scan_values[scan_index_down_19] = scan_values[scan_index_down_19] + scan_left_1;
        }
        __syncthreads();
        int scan_index_down_20 = (tid + 1) * 512 - 1;
        if (scan_index_down_20 < MAX_EXPERTS) {
            int scan_left_2 = scan_values[scan_index_down_20 - 256];
            scan_values[scan_index_down_20 - 256] = scan_values[scan_index_down_20];
            scan_values[scan_index_down_20] = scan_values[scan_index_down_20] + scan_left_2;
        }
        int scan_index_down_21 = (THREADS + tid + 1) * 512 - 1;
        if (scan_index_down_21 < MAX_EXPERTS) {
            int scan_left_3 = scan_values[scan_index_down_21 - 256];
            scan_values[scan_index_down_21 - 256] = scan_values[scan_index_down_21];
            scan_values[scan_index_down_21] = scan_values[scan_index_down_21] + scan_left_3;
        }
        __syncthreads();
        int scan_index_down_22 = (tid + 1) * 256 - 1;
        if (scan_index_down_22 < MAX_EXPERTS) {
            int scan_left_4 = scan_values[scan_index_down_22 - 128];
            scan_values[scan_index_down_22 - 128] = scan_values[scan_index_down_22];
            scan_values[scan_index_down_22] = scan_values[scan_index_down_22] + scan_left_4;
        }
        int scan_index_down_23 = (THREADS + tid + 1) * 256 - 1;
        if (scan_index_down_23 < MAX_EXPERTS) {
            int scan_left_5 = scan_values[scan_index_down_23 - 128];
            scan_values[scan_index_down_23 - 128] = scan_values[scan_index_down_23];
            scan_values[scan_index_down_23] = scan_values[scan_index_down_23] + scan_left_5;
        }
        __syncthreads();
        int scan_index_down_24 = (tid + 1) * 128 - 1;
        if (scan_index_down_24 < MAX_EXPERTS) {
            int scan_left_6 = scan_values[scan_index_down_24 - 64];
            scan_values[scan_index_down_24 - 64] = scan_values[scan_index_down_24];
            scan_values[scan_index_down_24] = scan_values[scan_index_down_24] + scan_left_6;
        }
        int scan_index_down_25 = (THREADS + tid + 1) * 128 - 1;
        if (scan_index_down_25 < MAX_EXPERTS) {
            int scan_left_7 = scan_values[scan_index_down_25 - 64];
            scan_values[scan_index_down_25 - 64] = scan_values[scan_index_down_25];
            scan_values[scan_index_down_25] = scan_values[scan_index_down_25] + scan_left_7;
        }
        __syncthreads();
        int scan_index_down_26 = (tid + 1) * 64 - 1;
        if (scan_index_down_26 < MAX_EXPERTS) {
            int scan_left_8 = scan_values[scan_index_down_26 - 32];
            scan_values[scan_index_down_26 - 32] = scan_values[scan_index_down_26];
            scan_values[scan_index_down_26] = scan_values[scan_index_down_26] + scan_left_8;
        }
        int scan_index_down_27 = (THREADS + tid + 1) * 64 - 1;
        if (scan_index_down_27 < MAX_EXPERTS) {
            int scan_left_9 = scan_values[scan_index_down_27 - 32];
            scan_values[scan_index_down_27 - 32] = scan_values[scan_index_down_27];
            scan_values[scan_index_down_27] = scan_values[scan_index_down_27] + scan_left_9;
        }
        __syncthreads();
        int scan_index_down_28 = (tid + 1) * 32 - 1;
        if (scan_index_down_28 < MAX_EXPERTS) {
            int scan_left_10 = scan_values[scan_index_down_28 - 16];
            scan_values[scan_index_down_28 - 16] = scan_values[scan_index_down_28];
            scan_values[scan_index_down_28] = scan_values[scan_index_down_28] + scan_left_10;
        }
        int scan_index_down_29 = (THREADS + tid + 1) * 32 - 1;
        if (scan_index_down_29 < MAX_EXPERTS) {
            int scan_left_11 = scan_values[scan_index_down_29 - 16];
            scan_values[scan_index_down_29 - 16] = scan_values[scan_index_down_29];
            scan_values[scan_index_down_29] = scan_values[scan_index_down_29] + scan_left_11;
        }
        __syncthreads();
        int scan_index_down_30 = (tid + 1) * 16 - 1;
        if (scan_index_down_30 < MAX_EXPERTS) {
            int scan_left_12 = scan_values[scan_index_down_30 - 8];
            scan_values[scan_index_down_30 - 8] = scan_values[scan_index_down_30];
            scan_values[scan_index_down_30] = scan_values[scan_index_down_30] + scan_left_12;
        }
        int scan_index_down_31 = (THREADS + tid + 1) * 16 - 1;
        if (scan_index_down_31 < MAX_EXPERTS) {
            int scan_left_13 = scan_values[scan_index_down_31 - 8];
            scan_values[scan_index_down_31 - 8] = scan_values[scan_index_down_31];
            scan_values[scan_index_down_31] = scan_values[scan_index_down_31] + scan_left_13;
        }
        __syncthreads();
        int scan_index_down_32 = (tid + 1) * 8 - 1;
        if (scan_index_down_32 < MAX_EXPERTS) {
            int scan_left_14 = scan_values[scan_index_down_32 - 4];
            scan_values[scan_index_down_32 - 4] = scan_values[scan_index_down_32];
            scan_values[scan_index_down_32] = scan_values[scan_index_down_32] + scan_left_14;
        }
        int scan_index_down_33 = (THREADS + tid + 1) * 8 - 1;
        if (scan_index_down_33 < MAX_EXPERTS) {
            int scan_left_15 = scan_values[scan_index_down_33 - 4];
            scan_values[scan_index_down_33 - 4] = scan_values[scan_index_down_33];
            scan_values[scan_index_down_33] = scan_values[scan_index_down_33] + scan_left_15;
        }
        __syncthreads();
        int scan_index_down_34 = (tid + 1) * 4 - 1;
        if (scan_index_down_34 < MAX_EXPERTS) {
            int scan_left_16 = scan_values[scan_index_down_34 - 2];
            scan_values[scan_index_down_34 - 2] = scan_values[scan_index_down_34];
            scan_values[scan_index_down_34] = scan_values[scan_index_down_34] + scan_left_16;
        }
        int scan_index_down_35 = (THREADS + tid + 1) * 4 - 1;
        if (scan_index_down_35 < MAX_EXPERTS) {
            int scan_left_17 = scan_values[scan_index_down_35 - 2];
            scan_values[scan_index_down_35 - 2] = scan_values[scan_index_down_35];
            scan_values[scan_index_down_35] = scan_values[scan_index_down_35] + scan_left_17;
        }
        __syncthreads();
        int scan_index_down_36 = (tid + 1) * 2 - 1;
        if (scan_index_down_36 < MAX_EXPERTS) {
            int scan_left_18 = scan_values[scan_index_down_36 - 1];
            scan_values[scan_index_down_36 - 1] = scan_values[scan_index_down_36];
            scan_values[scan_index_down_36] = scan_values[scan_index_down_36] + scan_left_18;
        }
        int scan_index_down_37 = (THREADS + tid + 1) * 2 - 1;
        if (scan_index_down_37 < MAX_EXPERTS) {
            int scan_left_19 = scan_values[scan_index_down_37 - 1];
            scan_values[scan_index_down_37 - 1] = scan_values[scan_index_down_37];
            scan_values[scan_index_down_37] = scan_values[scan_index_down_37] + scan_left_19;
        }
        __syncthreads();
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
            int padded_count_final = 0;
            {
                if (block_m == 8) {
                    padded_count_final = count_final + 7 & -8;
                } else if (block_m == 16) {
                    padded_count_final = count_final + 15 & -16;
                } else {
                    padded_count_final = (count_final + block_m - 1) / block_m * block_m;
                }
            }
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

} // extern "C"


constexpr int kThreads = THREADS;
constexpr int kSmemTotal = SMEM_TOTAL;
constexpr int kWarps = NUM_WARPS;
static_assert(THREADS == NUM_WARPS * 32);
static_assert(MAX_EXPERTS == 512);
#undef FAST_BLOCK_ALIGN
#undef LOOM_INF
#undef MAX_BLOCK_M
#undef MAX_EXPERTS
#undef MAX_TOP_K
#undef NUM_MAIN_STAGES
#undef NUM_WARPS
#undef PUBLIC_SHARED_SOFTMAX
#undef SINGLE_CTA
#undef SMEM_SCAN_VALUES_OFF
#undef SMEM_SCAN_VALUES_STAGE_BYTES
#undef SMEM_SCAN_VALUES_STRIDE
#undef SMEM_SHARED_COUNTS_OFF
#undef SMEM_SHARED_COUNTS_STAGE_BYTES
#undef SMEM_SHARED_COUNTS_STRIDE
#undef SMEM_SHARED_IDS_OFF
#undef SMEM_SHARED_IDS_STAGE_BYTES
#undef SMEM_SHARED_IDS_STRIDE
#undef SMEM_SHARED_OFFSETS_OFF
#undef SMEM_SHARED_OFFSETS_STAGE_BYTES
#undef SMEM_SHARED_OFFSETS_STRIDE
#undef SMEM_SHARED_SCATTER_OFF
#undef SMEM_SHARED_SCATTER_STAGE_BYTES
#undef SMEM_SHARED_SCATTER_STRIDE
#undef SMEM_SHARED_TOTAL_OFF
#undef SMEM_SHARED_TOTAL_STAGE_BYTES
#undef SMEM_SHARED_TOTAL_STRIDE
#undef SMEM_TOTAL
#undef THREADS
}  // namespace alphamoe_router_tiny_routed_generated

using alphamoe_router_tiny_routed_generated::kernel_alpha_moe_fused_router_tiny_routed;

static_assert(alphamoe_router_tiny_routed_generated::kThreads == alphamoe_router_small_generated::kThreads);

static_assert(alphamoe_router_tiny_routed_generated::kSmemTotal == alphamoe_router_small_generated::kSmemTotal);

static_assert(alphamoe_router_tiny_routed_generated::kWarps == alphamoe_router_small_generated::kWarps);

// clang-format on

namespace flashinfer {
namespace alphamoe_fused_router {

constexpr int64_t kMaxExperts = 512;
constexpr int64_t kMaxTopK = 16;
constexpr int64_t kMaxBlockM = 16;
constexpr int64_t kThreads = kGeneratedThreads;
constexpr int64_t kDynamicSmemBytes = kGeneratedSmemTotal;
constexpr int64_t kSmallThreads = kSmallGeneratedThreads;
constexpr int64_t kSmallDynamicSmemBytes = kSmallGeneratedSmemTotal;
constexpr int64_t kSmallMLimit = 128;

static_assert(kGeneratedThreads == kThreads);
static_assert(kGeneratedSmemTotal == kDynamicSmemBytes);

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

struct RouterLaunchConfig {
  int sm_count;
  int active_blocks_per_sm;
  int small_active_blocks_per_sm;
};

inline RouterLaunchConfig GetRouterLaunchConfig(int32_t device_id) {
  // Run selects the tensor device before resolving its runtime launch configuration.
  static std::mutex mutex;
  static std::unordered_map<int32_t, RouterLaunchConfig> cache;
  std::lock_guard<std::mutex> lock(mutex);
  const auto cached = cache.find(device_id);
  if (cached != cache.end()) return cached->second;

  int major = 0, minor = 0, sm_count = 0, cooperative_launch = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(compute capability major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(compute capability minor)");
  TVM_FFI_ICHECK(major == 10 && (minor == 0 || minor == 3))
      << "AlphaMoE fused router requires compute capability 10.0 or 10.3, got " << major << "."
      << minor;
  CheckCuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id),
            "cudaDeviceGetAttribute(multiprocessor count)");
  CheckCuda(cudaDeviceGetAttribute(&cooperative_launch, cudaDevAttrCooperativeLaunch, device_id),
            "cudaDeviceGetAttribute(cooperative launch)");
  TVM_FFI_ICHECK(cooperative_launch != 0)
      << "AlphaMoE fused router requires cooperative-launch support";
  CheckCuda(cudaFuncSetAttribute(kernel_alpha_moe_fused_router,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(kDynamicSmemBytes)),
            "cudaFuncSetAttribute(AlphaMoE router dynamic smem)");
  CheckCuda(cudaFuncSetAttribute(kernel_alpha_moe_fused_router_small,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(kSmallDynamicSmemBytes)),
            "cudaFuncSetAttribute(AlphaMoE small router dynamic smem)");
  int active_blocks_per_sm = 0;
  CheckCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks_per_sm, kernel_alpha_moe_fused_router, static_cast<int>(kThreads),
                static_cast<size_t>(kDynamicSmemBytes)),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor(AlphaMoE router)");
  TVM_FFI_ICHECK(active_blocks_per_sm > 0)
      << "AlphaMoE fused router has zero cooperative occupancy";
  int small_active_blocks_per_sm = 0;
  CheckCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &small_active_blocks_per_sm, kernel_alpha_moe_fused_router_small,
                static_cast<int>(kSmallThreads), static_cast<size_t>(kSmallDynamicSmemBytes)),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor(AlphaMoE small router)");
  TVM_FFI_ICHECK(small_active_blocks_per_sm > 0)
      << "AlphaMoE small fused router has zero cooperative occupancy";
  CheckCuda(cudaFuncSetAttribute(kernel_alpha_moe_fused_router_routed,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(kDynamicSmemBytes)),
            "cudaFuncSetAttribute(AlphaMoE large_routed router dynamic smem)");
  int routed_active_blocks_per_sm = 0;
  CheckCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &routed_active_blocks_per_sm, kernel_alpha_moe_fused_router_routed,
                static_cast<int>(kThreads), static_cast<size_t>(kDynamicSmemBytes)),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor(AlphaMoE large_routed router)");
  TVM_FFI_ICHECK(routed_active_blocks_per_sm > 0)
      << "AlphaMoE large_routed router has zero cooperative occupancy";
  CheckCuda(cudaFuncSetAttribute(kernel_alpha_moe_fused_router_small_routed,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(kSmallDynamicSmemBytes)),
            "cudaFuncSetAttribute(AlphaMoE small_routed router dynamic smem)");
  int small_routed_active_blocks_per_sm = 0;
  CheckCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &small_routed_active_blocks_per_sm, kernel_alpha_moe_fused_router_small_routed,
                static_cast<int>(kSmallThreads), static_cast<size_t>(kSmallDynamicSmemBytes)),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor(AlphaMoE small_routed router)");
  TVM_FFI_ICHECK(small_routed_active_blocks_per_sm > 0)
      << "AlphaMoE small_routed router has zero cooperative occupancy";
  CheckCuda(cudaFuncSetAttribute(kernel_alpha_moe_fused_router_tiny,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(kSmallDynamicSmemBytes)),
            "cudaFuncSetAttribute(AlphaMoE tiny router dynamic smem)");
  int tiny_active_blocks_per_sm = 0;
  CheckCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &tiny_active_blocks_per_sm, kernel_alpha_moe_fused_router_tiny,
                static_cast<int>(kSmallThreads), static_cast<size_t>(kSmallDynamicSmemBytes)),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor(AlphaMoE tiny router)");
  TVM_FFI_ICHECK(tiny_active_blocks_per_sm > 0)
      << "AlphaMoE tiny router has zero cooperative occupancy";
  CheckCuda(cudaFuncSetAttribute(kernel_alpha_moe_fused_router_tiny_routed,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(kSmallDynamicSmemBytes)),
            "cudaFuncSetAttribute(AlphaMoE tiny_routed router dynamic smem)");
  int tiny_routed_active_blocks_per_sm = 0;
  CheckCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &tiny_routed_active_blocks_per_sm, kernel_alpha_moe_fused_router_tiny_routed,
                static_cast<int>(kSmallThreads), static_cast<size_t>(kSmallDynamicSmemBytes)),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor(AlphaMoE tiny_routed router)");
  TVM_FFI_ICHECK(tiny_routed_active_blocks_per_sm > 0)
      << "AlphaMoE tiny_routed router has zero cooperative occupancy";
  const RouterLaunchConfig config{
      sm_count, std::min(active_blocks_per_sm, routed_active_blocks_per_sm),
      std::min({small_active_blocks_per_sm, small_routed_active_blocks_per_sm,
                tiny_active_blocks_per_sm, tiny_routed_active_blocks_per_sm})};
  cache.emplace(device_id, config);
  return config;
}

inline void CheckTensor(const TensorView& tensor, const char* name, DLDataType dtype, int64_t ndim,
                        int32_t device_id) {
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA) << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK(tensor.device().device_id == device_id)
      << name << " must be on CUDA device " << device_id << ", got " << tensor.device().device_id;
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
  TVM_FFI_ICHECK(tensor.dtype() == dtype) << name << " has an unsupported dtype";
  TVM_FFI_ICHECK(tensor.ndim() == ndim)
      << name << " must be rank " << ndim << ", got rank " << tensor.ndim();
}

struct TensorRange {
  uintptr_t begin;
  uintptr_t end;
};

inline TensorRange GetTensorRange(const TensorView& tensor, const char* name) {
  const DLDataType dtype = tensor.dtype();
  const uint64_t bits = static_cast<uint64_t>(dtype.bits) * dtype.lanes;
  TVM_FFI_ICHECK(bits > 0 && bits % 8 == 0) << name << " must have a byte-addressable dtype";
  const uint64_t bytes_per_element = bits / 8;
  TVM_FFI_ICHECK(static_cast<uint64_t>(tensor.numel()) <=
                 std::numeric_limits<uint64_t>::max() / bytes_per_element)
      << name << " byte count overflows uint64_t";
  const uint64_t bytes = static_cast<uint64_t>(tensor.numel()) * bytes_per_element;
  const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  TVM_FFI_ICHECK(bytes <= std::numeric_limits<uintptr_t>::max() - begin)
      << name << " byte range overflows uintptr_t";
  return {begin, begin + static_cast<uintptr_t>(bytes)};
}

inline void CheckNoAlias(const TensorView& lhs, const char* lhs_name, const TensorView& rhs,
                         const char* rhs_name) {
  const TensorRange lhs_range = GetTensorRange(lhs, lhs_name);
  const TensorRange rhs_range = GetTensorRange(rhs, rhs_name);
  TVM_FFI_ICHECK(!(lhs_range.begin < rhs_range.end && rhs_range.begin < lhs_range.end))
      << lhs_name << " must not overlap " << rhs_name
      << ": the frozen kernel uses __restrict__ pointers";
}

inline int64_t MaxRouteBlocks(int64_t m, int64_t top_k, int64_t num_experts, int64_t block_m) {
  const int64_t pairs = m * top_k;
  const int64_t nonempty = std::min(num_experts, pairs);
  return nonempty + (pairs - nonempty) / block_m;
}

void Run(TensorView logits, TensorView topk_weights, TensorView topk_ids,
         TensorView sorted_token_ids, TensorView expert_ids, TensorView num_tokens_post_padded,
         TensorView expert_counts, TensorView expert_offsets, TensorView expert_scatter_offsets,
         int64_t top_k, int64_t block_m, bool has_shared_expert) {
  TVM_FFI_ICHECK(logits.device().device_type == kDLCUDA) << "logits must be a CUDA tensor";
  const int32_t device_id = logits.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);

  CheckTensor(logits, "logits", dl_float32, 2, device_id);
  CheckTensor(topk_weights, "topk_weights", dl_float32, 2, device_id);
  CheckTensor(topk_ids, "topk_ids", dl_int32, 2, device_id);
  CheckTensor(sorted_token_ids, "sorted_token_ids", dl_int32, 1, device_id);
  CheckTensor(expert_ids, "expert_ids", dl_int32, 1, device_id);
  CheckTensor(num_tokens_post_padded, "num_tokens_post_padded", dl_int32, 1, device_id);
  CheckTensor(expert_counts, "expert_counts", dl_int32, 1, device_id);
  CheckTensor(expert_offsets, "expert_offsets", dl_int32, 1, device_id);
  CheckTensor(expert_scatter_offsets, "expert_scatter_offsets", dl_int32, 1, device_id);

  const int64_t m = logits.size(0);
  const int64_t num_experts = logits.size(1);
  TVM_FFI_ICHECK(m > 0) << "logits must contain at least one token";
  TVM_FFI_ICHECK(num_experts >= 1 && num_experts <= kMaxExperts)
      << "num_experts must be in [1, " << kMaxExperts << "], got " << num_experts;
  TVM_FFI_ICHECK(top_k >= 1 && top_k <= std::min(num_experts, kMaxTopK))
      << "top_k must be in [1, min(num_experts, " << kMaxTopK << ")], got " << top_k;
  TVM_FFI_ICHECK(block_m >= 1 && block_m <= kMaxBlockM)
      << "block_m must be in [1, " << kMaxBlockM << "], got " << block_m;
  TVM_FFI_ICHECK(!has_shared_expert || top_k >= 2) << "a forced shared expert requires top_k >= 2";
  TVM_FFI_ICHECK(m <= std::numeric_limits<int>::max()) << "num_tokens must fit in int32";
  TVM_FFI_ICHECK(m * top_k <= std::numeric_limits<int>::max())
      << "num_tokens * top_k must fit in int32";

  const int64_t max_route_blocks = MaxRouteBlocks(m, top_k, num_experts, block_m);
  TVM_FFI_ICHECK(max_route_blocks <= std::numeric_limits<int>::max() / block_m)
      << "maximum padded route count exceeds int32";
  const int64_t max_padded_pairs = max_route_blocks * block_m;

  TVM_FFI_ICHECK(topk_weights.size(0) == m && topk_weights.size(1) == top_k)
      << "topk_weights must have shape (" << m << ", " << top_k << ")";
  TVM_FFI_ICHECK(topk_ids.size(0) == m && topk_ids.size(1) == top_k)
      << "topk_ids must have shape (" << m << ", " << top_k << ")";
  TVM_FFI_ICHECK(sorted_token_ids.numel() >= max_padded_pairs)
      << "sorted_token_ids capacity must be at least " << max_padded_pairs;
  TVM_FFI_ICHECK(expert_ids.numel() >= max_route_blocks)
      << "expert_ids capacity must be at least " << max_route_blocks;
  TVM_FFI_ICHECK(num_tokens_post_padded.numel() == 1)
      << "num_tokens_post_padded must contain exactly one int32 element";
  TVM_FFI_ICHECK(expert_counts.numel() == num_experts)
      << "expert_counts must have shape (" << num_experts << ",)";
  TVM_FFI_ICHECK(expert_offsets.numel() == num_experts + 1)
      << "expert_offsets must have shape (" << num_experts + 1 << ",)";
  TVM_FFI_ICHECK(expert_scatter_offsets.numel() == num_experts)
      << "expert_scatter_offsets must have shape (" << num_experts << ",)";

  const std::array<const TensorView*, 9> tensors = {
      &logits,           &topk_weights,   &topk_ids,
      &sorted_token_ids, &expert_ids,     &num_tokens_post_padded,
      &expert_counts,    &expert_offsets, &expert_scatter_offsets,
  };
  const std::array<const char*, 9> names = {
      "logits",           "topk_weights",   "topk_ids",
      "sorted_token_ids", "expert_ids",     "num_tokens_post_padded",
      "expert_counts",    "expert_offsets", "expert_scatter_offsets",
  };
  for (size_t i = 0; i < tensors.size(); ++i) {
    for (size_t j = i + 1; j < tensors.size(); ++j) {
      CheckNoAlias(*tensors[i], names[i], *tensors[j], names[j]);
    }
  }

  const RouterLaunchConfig config = GetRouterLaunchConfig(device_id);
  const bool use_small = m <= kSmallMLimit;
  const int64_t grid_x =
      use_small
          ? std::max<int64_t>(
                1, std::min<int64_t>((m + kSmallGeneratedWarps - 1) / kSmallGeneratedWarps,
                                     config.sm_count))
          : std::max<int64_t>(
                1, std::min<int64_t>(
                       ((m <= 512) ? (std::min<int64_t>((m + 7) / 8, config.sm_count))
                                   : ((((m + 3) / 4 + config.sm_count - 1) / config.sm_count) *
                                      config.sm_count)),
                       static_cast<int64_t>(config.sm_count) *
                           std::min(2, config.active_blocks_per_sm)));
  const int64_t cooperative_capacity =
      static_cast<int64_t>(use_small ? config.small_active_blocks_per_sm
                                     : config.active_blocks_per_sm) *
      config.sm_count;
  TVM_FFI_ICHECK(grid_x >= 1 && grid_x <= cooperative_capacity)
      << "AlphaMoE fused router grid " << grid_x << " exceeds cooperative residency capacity "
      << cooperative_capacity;

  float* logits_ptr = static_cast<float*>(logits.data_ptr());
  float* topk_weights_ptr = static_cast<float*>(topk_weights.data_ptr());
  int* topk_ids_ptr = static_cast<int*>(topk_ids.data_ptr());
  int* sorted_token_ids_ptr = static_cast<int*>(sorted_token_ids.data_ptr());
  int* expert_ids_ptr = static_cast<int*>(expert_ids.data_ptr());
  int* num_tokens_post_padded_ptr = static_cast<int*>(num_tokens_post_padded.data_ptr());
  int* expert_counts_ptr = static_cast<int*>(expert_counts.data_ptr());
  int* expert_offsets_ptr = static_cast<int*>(expert_offsets.data_ptr());
  int* expert_scatter_offsets_ptr = static_cast<int*>(expert_scatter_offsets.data_ptr());
  int m_arg = static_cast<int>(m);
  int num_experts_arg = static_cast<int>(num_experts);
  int top_k_arg = static_cast<int>(top_k);
  int block_m_arg = static_cast<int>(block_m);
  int has_shared_expert_arg = static_cast<int>(has_shared_expert);
  void* arguments[] = {
      &logits_ptr,
      &topk_weights_ptr,
      &topk_ids_ptr,
      &sorted_token_ids_ptr,
      &expert_ids_ptr,
      &num_tokens_post_padded_ptr,
      &expert_counts_ptr,
      &expert_offsets_ptr,
      &expert_scatter_offsets_ptr,
      &m_arg,
      &num_experts_arg,
      &top_k_arg,
      &block_m_arg,
      &has_shared_expert_arg,
  };

  const cudaStream_t stream = get_stream(logits.device());
  if (m <= 8) {
    CheckCuda(cudaLaunchKernel(reinterpret_cast<const void*>(
                                   has_shared_expert ? kernel_alpha_moe_fused_router_tiny
                                                     : kernel_alpha_moe_fused_router_tiny_routed),
                               dim3(1, 1, 1), dim3(static_cast<unsigned int>(kSmallThreads), 1, 1),
                               arguments, static_cast<size_t>(kSmallDynamicSmemBytes), stream),
              "cudaLaunchKernel(AlphaMoE single-CTA fused router)");
  } else if (use_small) {
    CheckCuda(cudaLaunchCooperativeKernel(
                  reinterpret_cast<const void*>(has_shared_expert
                                                    ? kernel_alpha_moe_fused_router_small
                                                    : kernel_alpha_moe_fused_router_small_routed),
                  dim3(static_cast<unsigned int>(grid_x), 1, 1),
                  dim3(static_cast<unsigned int>(kSmallThreads), 1, 1), arguments,
                  static_cast<size_t>(kSmallDynamicSmemBytes), stream),
              "cudaLaunchCooperativeKernel(AlphaMoE small fused router)");
  } else {
    CheckCuda(
        cudaLaunchCooperativeKernel(
            reinterpret_cast<const void*>(has_shared_expert ? kernel_alpha_moe_fused_router
                                                            : kernel_alpha_moe_fused_router_routed),
            dim3(static_cast<unsigned int>(grid_x), 1, 1),
            dim3(static_cast<unsigned int>(kThreads), 1, 1), arguments,
            static_cast<size_t>(kDynamicSmemBytes), stream),
        "cudaLaunchCooperativeKernel(AlphaMoE fused router)");
  }
}

}  // namespace alphamoe_fused_router
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_router_op, flashinfer::alphamoe_fused_router::Run);
