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
    int* local_offsets = reinterpret_cast<int*>(smem_raw + 0);
    const int local_offsets_addr = smem + 0;
    int* scan_values = reinterpret_cast<int*>(smem_raw + 4100);
    const int scan_values_addr = smem + 4100;

    // === Task calls (dependency order) ===
    int global_thread = bid * THREADS + tid;
    int use_cta_reservations = 0;
    if (MAX_EXPERTS == 512 && PUBLIC_SHARED_SOFTMAX == 0 && E == 512 && has_shared_expert == 0 && M > 512 && M <= 2147483647 / (MAX_EXPERTS * MAX_TOP_K)) {
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
        for (int expert_slot_local_zero = 0; expert_slot_local_zero < MAX_EXPERTS / THREADS; expert_slot_local_zero++) {
            int expert_local_zero = tid + expert_slot_local_zero * THREADS;
            local_offsets[expert_local_zero] = 0;
        }
    }
    cooperative_groups::this_grid().sync();
    int routed_experts = E - has_shared_expert;
    int routed_top_k = top_k - has_shared_expert;
    for (int token = (unsigned int)(bid * NUM_WARPS) + warp; token < M; token += num_bids * NUM_WARPS) {
        unsigned long long row_base = (unsigned long long)token * (unsigned long long)E;
        float row_values[MAX_EXPERTS / 32];
        if (MAX_EXPERTS == 512 && PUBLIC_SHARED_SOFTMAX == 0 && E == 512 && has_shared_expert == 0) {
            unsigned long long lane_row_base = row_base + (unsigned long long)lane;
            #pragma unroll
            for (int expert_slot_full = 0; expert_slot_full < MAX_EXPERTS / 32; expert_slot_full++) {
                row_values[expert_slot_full] = logits[lane_row_base + (unsigned long long)(expert_slot_full * 32)];
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
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(local_key));
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
                for (int bounded_mask_slot = 0; bounded_mask_slot < MAX_EXPERTS / 128; bounded_mask_slot++) {
                    #pragma unroll
                    for (int bounded_mask_group = 0; bounded_mask_group < 4; bounded_mask_group++) {
                        if (row_values[bounded_mask_group * (MAX_EXPERTS / 128) + bounded_mask_slot] == best_value) {
                            bounded_tie_masks[bounded_mask_group] = bounded_tie_masks[bounded_mask_group] | (unsigned int)(2147483648 >> bounded_mask_group * (MAX_EXPERTS / 128) + bounded_mask_slot);
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
                asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_i32_0) : "r"(descending_tie_key));
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
                            } else if (lane + (unsigned int)((remove_group * 8 + 1) * 32) == (unsigned int)best_index) {
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
                        } else if (lane + (unsigned int)((remove_group * 8 + 5) * 32) == (unsigned int)best_index) {
                            row_values[remove_group * 8 + 5] = -LOOM_INF;
                        }
                    } else {
                        if (best_index < remove_group * 256 + 224) {
                            if (lane + (unsigned int)((remove_group * 8 + 6) * 32) == (unsigned int)best_index) {
                                row_values[remove_group * 8 + 6] = -LOOM_INF;
                            }
                        } else if (lane + (unsigned int)((remove_group * 8 + 7) * 32) == (unsigned int)best_index) {
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
                unsigned long long first_rebase_pair = (unsigned long long)first_rebase_token * (unsigned long long)top_k + (unsigned long long)lane;
                first_stored_expert = topk_ids[first_rebase_pair];
            }
        }
        __syncthreads();
        #pragma unroll
        for (int expert_slot_reserve = 0; expert_slot_reserve < MAX_EXPERTS / THREADS; expert_slot_reserve++) {
            int expert_reserve = tid + expert_slot_reserve * THREADS;
            int count_reserve = local_offsets[expert_reserve];
            if (count_reserve > 0) {
                int _atomic_old_2 = atomicAdd(&expert_counts[expert_reserve], count_reserve);
                int base_reserve = _atomic_old_2;
                local_offsets[expert_reserve] = base_reserve;
            }
        }
        __syncthreads();
        for (int token_rebase = (unsigned int)(bid * NUM_WARPS) + warp; token_rebase < M; token_rebase += num_bids * NUM_WARPS) {
            if (lane < (unsigned int)top_k) {
                unsigned long long pair_rebase = (unsigned long long)token_rebase * (unsigned long long)top_k + (unsigned long long)lane;
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
#undef SMEM_LOCAL_OFFSETS_OFF
#undef SMEM_LOCAL_OFFSETS_STAGE_BYTES
#undef SMEM_LOCAL_OFFSETS_STRIDE
#undef SMEM_SCAN_VALUES_OFF
#undef SMEM_SCAN_VALUES_STAGE_BYTES
#undef SMEM_SCAN_VALUES_STRIDE
#undef SMEM_TOTAL
#undef THREADS
}  // namespace alphamoe_router_large_generated

using alphamoe_router_large_generated::kernel_alpha_moe_fused_router;

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
    int* local_offsets = reinterpret_cast<int*>(smem_raw + 0);
    const int local_offsets_addr = smem + 0;
    int* scan_values = reinterpret_cast<int*>(smem_raw + 4100);
    const int scan_values_addr = smem + 4100;

    // === Task calls (dependency order) ===
    int global_thread = bid * THREADS + tid;
    int use_cta_reservations = 0;
    if (MAX_EXPERTS == 512 && PUBLIC_SHARED_SOFTMAX == 0 && E == 512 && has_shared_expert == 0 && M > 512 && M <= 2147483647 / (MAX_EXPERTS * MAX_TOP_K)) {
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
        for (int expert_slot_local_zero = 0; expert_slot_local_zero < MAX_EXPERTS / THREADS; expert_slot_local_zero++) {
            int expert_local_zero = tid + expert_slot_local_zero * THREADS;
            local_offsets[expert_local_zero] = 0;
        }
    }
    cooperative_groups::this_grid().sync();
    int routed_experts = E - has_shared_expert;
    int routed_top_k = top_k - has_shared_expert;
    for (int token = (unsigned int)(bid * NUM_WARPS) + warp; token < M; token += num_bids * NUM_WARPS) {
        unsigned long long row_base = (unsigned long long)token * (unsigned long long)E;
        float row_values[MAX_EXPERTS / 32];
        if (MAX_EXPERTS == 512 && PUBLIC_SHARED_SOFTMAX == 0 && E == 512 && has_shared_expert == 0) {
            unsigned long long lane_row_base = row_base + (unsigned long long)lane;
            #pragma unroll
            for (int expert_slot_full = 0; expert_slot_full < MAX_EXPERTS / 32; expert_slot_full++) {
                row_values[expert_slot_full] = logits[lane_row_base + (unsigned long long)(expert_slot_full * 32)];
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
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(local_key));
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
                for (int bounded_mask_slot = 0; bounded_mask_slot < MAX_EXPERTS / 128; bounded_mask_slot++) {
                    #pragma unroll
                    for (int bounded_mask_group = 0; bounded_mask_group < 4; bounded_mask_group++) {
                        if (row_values[bounded_mask_group * (MAX_EXPERTS / 128) + bounded_mask_slot] == best_value) {
                            bounded_tie_masks[bounded_mask_group] = bounded_tie_masks[bounded_mask_group] | (unsigned int)(2147483648 >> bounded_mask_group * (MAX_EXPERTS / 128) + bounded_mask_slot);
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
                asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_i32_0) : "r"(descending_tie_key));
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
                            } else if (lane + (unsigned int)((remove_group * 8 + 1) * 32) == (unsigned int)best_index) {
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
                        } else if (lane + (unsigned int)((remove_group * 8 + 5) * 32) == (unsigned int)best_index) {
                            row_values[remove_group * 8 + 5] = -LOOM_INF;
                        }
                    } else {
                        if (best_index < remove_group * 256 + 224) {
                            if (lane + (unsigned int)((remove_group * 8 + 6) * 32) == (unsigned int)best_index) {
                                row_values[remove_group * 8 + 6] = -LOOM_INF;
                            }
                        } else if (lane + (unsigned int)((remove_group * 8 + 7) * 32) == (unsigned int)best_index) {
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
                unsigned long long first_rebase_pair = (unsigned long long)first_rebase_token * (unsigned long long)top_k + (unsigned long long)lane;
                first_stored_expert = topk_ids[first_rebase_pair];
            }
        }
        __syncthreads();
        #pragma unroll
        for (int expert_slot_reserve = 0; expert_slot_reserve < MAX_EXPERTS / THREADS; expert_slot_reserve++) {
            int expert_reserve = tid + expert_slot_reserve * THREADS;
            int count_reserve = local_offsets[expert_reserve];
            if (count_reserve > 0) {
                int _atomic_old_2 = atomicAdd(&expert_counts[expert_reserve], count_reserve);
                int base_reserve = _atomic_old_2;
                local_offsets[expert_reserve] = base_reserve;
            }
        }
        __syncthreads();
        for (int token_rebase = (unsigned int)(bid * NUM_WARPS) + warp; token_rebase < M; token_rebase += num_bids * NUM_WARPS) {
            if (lane < (unsigned int)top_k) {
                unsigned long long pair_rebase = (unsigned long long)token_rebase * (unsigned long long)top_k + (unsigned long long)lane;
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
#undef SMEM_LOCAL_OFFSETS_OFF
#undef SMEM_LOCAL_OFFSETS_STAGE_BYTES
#undef SMEM_LOCAL_OFFSETS_STRIDE
#undef SMEM_SCAN_VALUES_OFF
#undef SMEM_SCAN_VALUES_STAGE_BYTES
#undef SMEM_SCAN_VALUES_STRIDE
#undef SMEM_TOTAL
#undef THREADS
}  // namespace alphamoe_router_large_routed_generated

using alphamoe_router_large_routed_generated::kernel_alpha_moe_fused_router_routed;

namespace alphamoe_router_medium_generated {
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
#define SMEM_LOCAL_OFFSETS_OFF 0
#define SMEM_LOCAL_OFFSETS_STAGE_BYTES 4100
#define SMEM_LOCAL_OFFSETS_STRIDE 4100
#define SMEM_SCAN_VALUES_OFF 4100
#define SMEM_SCAN_VALUES_STAGE_BYTES 16
#define SMEM_SCAN_VALUES_STRIDE 16
#define SMEM_PRIVATE_COUNTS_OFF 2052
#define SMEM_PRIVATE_COUNTS_STAGE_BYTES 2048
#define SMEM_PRIVATE_COUNTS_STRIDE 2048
#define SMEM_TOTAL 4224
#define THREADS 128
#define NUM_WARPS 4
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

__global__ __launch_bounds__(128) void
kernel_alpha_moe_fused_router_medium(float* __restrict__ logits, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M, int E, int top_k, int block_m, int has_shared_expert)
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
    int* local_offsets = reinterpret_cast<int*>(smem_raw + 0);
    const int local_offsets_addr = smem + 0;
    int* scan_values = reinterpret_cast<int*>(smem_raw + 4100);
    const int scan_values_addr = smem + 4100;
    int* private_counts = reinterpret_cast<int*>(smem_raw + 2052);
    const int private_counts_addr = smem + 2052;

    // === Task calls (dependency order) ===
    int global_thread = bid * THREADS + tid;
    int private_raw_hist = 0;
    if (MAX_EXPERTS == 512 && PUBLIC_SHARED_SOFTMAX == 0 && E == 512 && has_shared_expert == 0 && M <= 512 && M <= num_bids * NUM_WARPS) {
        private_raw_hist = 1;
    }
    if (bid == 0) {
        #pragma unroll
        for (int expert_slot_zero = 0; expert_slot_zero < MAX_EXPERTS / THREADS; expert_slot_zero++) {
            int expert_zero = tid + expert_slot_zero * THREADS;
            if (expert_zero < E) {
                if (private_raw_hist == 0) {
                    expert_counts[expert_zero] = 0;
                }
                expert_offsets[expert_zero] = 0;
                expert_scatter_offsets[expert_zero] = 0;
            }
        }
        if (tid == 0) {
            expert_offsets[E] = 0;
            num_tokens_post_padded[0] = 0;
        }
    }
    float row_values[MAX_EXPERTS / 32];
    int preload_single_row = 0;
    if (MAX_EXPERTS == 512 && PUBLIC_SHARED_SOFTMAX == 0 && E == 512 && has_shared_expert == 0 && M <= num_bids * NUM_WARPS) {
        preload_single_row = 1;
        int preload_token = (unsigned int)(bid * NUM_WARPS) + warp;
        if (preload_token < M) {
            unsigned long long preload_row_base = (unsigned long long)preload_token * (unsigned long long)E;
            unsigned long long preload_lane_base = preload_row_base + (unsigned long long)lane;
            #pragma unroll
            for (int preload_slot = 0; preload_slot < MAX_EXPERTS / 32; preload_slot++) {
                row_values[preload_slot] = logits[preload_lane_base + (unsigned long long)(preload_slot * 32)];
            }
        }
    }
    if (private_raw_hist == 0) {
        cooperative_groups::this_grid().sync();
    }
    int routed_experts = E - has_shared_expert;
    int routed_top_k = top_k - has_shared_expert;
    for (int token = (unsigned int)(bid * NUM_WARPS) + warp; token < M; token += num_bids * NUM_WARPS) {
        unsigned long long row_base = 0;
        if (preload_single_row == 0) {
            row_base = (unsigned long long)token * (unsigned long long)E;
            if (MAX_EXPERTS == 512 && PUBLIC_SHARED_SOFTMAX == 0 && E == 512 && has_shared_expert == 0) {
                unsigned long long lane_row_base = row_base + (unsigned long long)lane;
                #pragma unroll
                for (int expert_slot_full = 0; expert_slot_full < MAX_EXPERTS / 32; expert_slot_full++) {
                    row_values[expert_slot_full] = logits[lane_row_base + (unsigned long long)(expert_slot_full * 32)];
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
            if (best_value != -LOOM_INF) {
                #pragma unroll
                for (int expert_slot_finite_tie = 0; expert_slot_finite_tie < MAX_EXPERTS / 32; expert_slot_finite_tie++) {
                    if (row_values[MAX_EXPERTS / 32 - 1 - expert_slot_finite_tie] == best_value) {
                        int expert_finite_tie = lane + (unsigned int)((MAX_EXPERTS / 32 - 1 - expert_slot_finite_tie) * 32);
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
            int stored_expert = selected_expert;
            if (M > 512 && M <= 2147483647 / (MAX_EXPERTS * MAX_TOP_K)) {
                int _atomic_old_0 = atomicAdd(&expert_counts[selected_expert], 1);
                int counted_row = _atomic_old_0;
                stored_expert = counted_row * MAX_EXPERTS + selected_expert;
            } else if (private_raw_hist == 0) {
                atomicAdd(&expert_counts[selected_expert], 1);
            }
            topk_ids[output_base + (unsigned long long)lane] = stored_expert;
        }
    }
    cooperative_groups::this_grid().sync();
    if (bid * THREADS < M * top_k || bid * THREADS < E) {
        int scan_counts[MAX_EXPERTS / THREADS];
        if (private_raw_hist != 0) {
            #pragma unroll
            for (int private_zero_slot = 0; private_zero_slot < 512 / THREADS; private_zero_slot++) {
                private_counts[tid + private_zero_slot * THREADS] = 0;
            }
            __syncthreads();
            for (int count_pair = tid; count_pair < M * top_k; count_pair += THREADS) {
                int count_expert = topk_ids[count_pair];
                atomicAdd(&private_counts[count_expert], 1);
            }
            __syncthreads();
            #pragma unroll
            for (int private_load_slot = 0; private_load_slot < MAX_EXPERTS / THREADS; private_load_slot++) {
                int private_expert = tid * (MAX_EXPERTS / THREADS) + private_load_slot;
                int private_count = private_counts[private_expert];
                scan_counts[private_load_slot] = private_count;
                if (bid == 0) {
                    expert_counts[private_expert] = private_count;
                }
            }
        } else if (MAX_EXPERTS == 512 && E == 512 && ((unsigned long long)expert_counts & 15) == 0) {
            {
                int4 _iv4 = *reinterpret_cast<const int4*>(expert_counts + tid * 4);
                scan_counts[0 + 0] = _iv4.x;
                scan_counts[0 + 1] = _iv4.y;
                scan_counts[0 + 2] = _iv4.z;
                scan_counts[0 + 3] = _iv4.w;
            }
        } else {
            #pragma unroll
            for (int expert_slot_count_load = 0; expert_slot_count_load < MAX_EXPERTS / THREADS; expert_slot_count_load++) {
                int expert_count_load = tid * (MAX_EXPERTS / THREADS) + expert_slot_count_load;
                int count_loaded = 0;
                if (expert_count_load < E) {
                    count_loaded = expert_counts[expert_count_load];
                }
                scan_counts[expert_slot_count_load] = count_loaded;
            }
        }
        int scan_padded[MAX_EXPERTS / THREADS];
        int scan_thread_total = 0;
        #pragma unroll
        for (int expert_slot_scan_init = 0; expert_slot_scan_init < MAX_EXPERTS / THREADS; expert_slot_scan_init++) {
            int expert_scan_init = tid * (MAX_EXPERTS / THREADS) + expert_slot_scan_init;
            int padded_count_init = 0;
            if (expert_scan_init < E) {
                int count_init = scan_counts[expert_slot_scan_init];
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
        if (MAX_EXPERTS == 512 && E == 512) {
            #pragma unroll
            for (int expert_slot_full_prefix = 0; expert_slot_full_prefix < MAX_EXPERTS / THREADS; expert_slot_full_prefix++) {
                int expert_full_prefix = tid * (MAX_EXPERTS / THREADS) + expert_slot_full_prefix;
                local_offsets[expert_full_prefix] = scan_expert_prefix;
                if (bid == 0) {
                    expert_offsets[expert_full_prefix] = scan_expert_prefix;
                    if (M > 512 && M <= 2147483647 / (MAX_EXPERTS * MAX_TOP_K)) {
                        expert_scatter_offsets[expert_full_prefix] = expert_counts[expert_full_prefix];
                    }
                }
                scan_expert_prefix = scan_expert_prefix + scan_padded[expert_slot_full_prefix];
            }
        } else {
            #pragma unroll
            for (int expert_slot_scan_store = 0; expert_slot_scan_store < MAX_EXPERTS / THREADS; expert_slot_scan_store++) {
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
        }
        __syncthreads();
        int scatter_block_mask = block_m - 1;
        int scatter_block_shift = 0;
        if (block_m == 8) {
            scatter_block_shift = 3;
        } else if (block_m == 16) {
            scatter_block_shift = 4;
        }
        if (M * top_k <= num_bids * THREADS) {
            if (global_thread < M * top_k) {
                int pair = global_thread;
                int stored_pair = topk_ids[pair];
                int pair_expert = stored_pair;
                int local_row = 0;
                if (M > 512 && M <= 2147483647 / (MAX_EXPERTS * MAX_TOP_K)) {
                    pair_expert = stored_pair % MAX_EXPERTS;
                    local_row = stored_pair / MAX_EXPERTS;
                    topk_ids[pair] = pair_expert;
                } else {
                    int _atomic_old_1 = atomicAdd(&expert_scatter_offsets[pair_expert], 1);
                    local_row = _atomic_old_1;
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
        } else {
            for (int pair_1 = global_thread; pair_1 < M * top_k; pair_1 += num_bids * THREADS) {
                int stored_pair_1 = topk_ids[pair_1];
                int pair_expert_1 = stored_pair_1;
                int local_row_1 = 0;
                if (M > 512 && M <= 2147483647 / (MAX_EXPERTS * MAX_TOP_K)) {
                    pair_expert_1 = stored_pair_1 % MAX_EXPERTS;
                    local_row_1 = stored_pair_1 / MAX_EXPERTS;
                    topk_ids[pair_1] = pair_expert_1;
                } else {
                    int _atomic_old_2 = atomicAdd(&expert_scatter_offsets[pair_expert_1], 1);
                    local_row_1 = _atomic_old_2;
                }
                int grouped_row_1 = local_offsets[pair_expert_1] + local_row_1;
                sorted_token_ids[grouped_row_1] = pair_1;
                int scatter_block_remainder_1 = 0;
                if (scatter_block_shift != 0) {
                    scatter_block_remainder_1 = local_row_1 & scatter_block_mask;
                } else {
                    scatter_block_remainder_1 = local_row_1 % block_m;
                }
                if (scatter_block_remainder_1 == 0) {
                    int scatter_block_index_1 = 0;
                    if (scatter_block_shift != 0) {
                        scatter_block_index_1 = grouped_row_1 >> scatter_block_shift;
                    } else {
                        scatter_block_index_1 = grouped_row_1 / block_m;
                    }
                    expert_ids[scatter_block_index_1] = pair_expert_1;
                }
            }
        }
        for (int padding_expert = global_thread; padding_expert < E; padding_expert += num_bids * THREADS) {
            int count_final = 0;
            if (private_raw_hist != 0) {
                count_final = private_counts[padding_expert];
            } else {
                count_final = expert_counts[padding_expert];
            }
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
#undef SMEM_LOCAL_OFFSETS_OFF
#undef SMEM_LOCAL_OFFSETS_STAGE_BYTES
#undef SMEM_LOCAL_OFFSETS_STRIDE
#undef SMEM_PRIVATE_COUNTS_OFF
#undef SMEM_PRIVATE_COUNTS_STAGE_BYTES
#undef SMEM_PRIVATE_COUNTS_STRIDE
#undef SMEM_SCAN_VALUES_OFF
#undef SMEM_SCAN_VALUES_STAGE_BYTES
#undef SMEM_SCAN_VALUES_STRIDE
#undef SMEM_TOTAL
#undef THREADS
}  // namespace alphamoe_router_medium_generated

using alphamoe_router_medium_generated::kernel_alpha_moe_fused_router_medium;

namespace alphamoe_router_medium_routed_generated {
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
#define SMEM_LOCAL_OFFSETS_OFF 0
#define SMEM_LOCAL_OFFSETS_STAGE_BYTES 4100
#define SMEM_LOCAL_OFFSETS_STRIDE 4100
#define SMEM_SCAN_VALUES_OFF 4100
#define SMEM_SCAN_VALUES_STAGE_BYTES 16
#define SMEM_SCAN_VALUES_STRIDE 16
#define SMEM_PRIVATE_COUNTS_OFF 2052
#define SMEM_PRIVATE_COUNTS_STAGE_BYTES 2048
#define SMEM_PRIVATE_COUNTS_STRIDE 2048
#define SMEM_TOTAL 4224
#define THREADS 128
#define NUM_WARPS 4
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

__global__ __launch_bounds__(128) void
kernel_alpha_moe_fused_router_medium_routed(float* __restrict__ logits, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M, int E, int top_k, int block_m, int has_shared_expert)
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
    int* local_offsets = reinterpret_cast<int*>(smem_raw + 0);
    const int local_offsets_addr = smem + 0;
    int* scan_values = reinterpret_cast<int*>(smem_raw + 4100);
    const int scan_values_addr = smem + 4100;
    int* private_counts = reinterpret_cast<int*>(smem_raw + 2052);
    const int private_counts_addr = smem + 2052;

    // === Task calls (dependency order) ===
    int global_thread = bid * THREADS + tid;
    int private_raw_hist = 0;
    if (MAX_EXPERTS == 512 && PUBLIC_SHARED_SOFTMAX == 0 && E == 512 && has_shared_expert == 0 && M <= 512 && M <= num_bids * NUM_WARPS) {
        private_raw_hist = 1;
    }
    if (bid == 0) {
        #pragma unroll
        for (int expert_slot_zero = 0; expert_slot_zero < MAX_EXPERTS / THREADS; expert_slot_zero++) {
            int expert_zero = tid + expert_slot_zero * THREADS;
            if (expert_zero < E) {
                if (private_raw_hist == 0) {
                    expert_counts[expert_zero] = 0;
                }
                expert_offsets[expert_zero] = 0;
                expert_scatter_offsets[expert_zero] = 0;
            }
        }
        if (tid == 0) {
            expert_offsets[E] = 0;
            num_tokens_post_padded[0] = 0;
        }
    }
    float row_values[MAX_EXPERTS / 32];
    int preload_single_row = 0;
    if (MAX_EXPERTS == 512 && PUBLIC_SHARED_SOFTMAX == 0 && E == 512 && has_shared_expert == 0 && M <= num_bids * NUM_WARPS) {
        preload_single_row = 1;
        int preload_token = (unsigned int)(bid * NUM_WARPS) + warp;
        if (preload_token < M) {
            unsigned long long preload_row_base = (unsigned long long)preload_token * (unsigned long long)E;
            unsigned long long preload_lane_base = preload_row_base + (unsigned long long)lane;
            #pragma unroll
            for (int preload_slot = 0; preload_slot < MAX_EXPERTS / 32; preload_slot++) {
                row_values[preload_slot] = logits[preload_lane_base + (unsigned long long)(preload_slot * 32)];
            }
        }
    }
    if (private_raw_hist == 0) {
        cooperative_groups::this_grid().sync();
    }
    int routed_experts = E - has_shared_expert;
    int routed_top_k = top_k - has_shared_expert;
    for (int token = (unsigned int)(bid * NUM_WARPS) + warp; token < M; token += num_bids * NUM_WARPS) {
        unsigned long long row_base = 0;
        if (preload_single_row == 0) {
            row_base = (unsigned long long)token * (unsigned long long)E;
            if (MAX_EXPERTS == 512 && PUBLIC_SHARED_SOFTMAX == 0 && E == 512 && has_shared_expert == 0) {
                unsigned long long lane_row_base = row_base + (unsigned long long)lane;
                #pragma unroll
                for (int expert_slot_full = 0; expert_slot_full < MAX_EXPERTS / 32; expert_slot_full++) {
                    row_values[expert_slot_full] = logits[lane_row_base + (unsigned long long)(expert_slot_full * 32)];
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
            if (best_value != -LOOM_INF) {
                #pragma unroll
                for (int expert_slot_finite_tie = 0; expert_slot_finite_tie < MAX_EXPERTS / 32; expert_slot_finite_tie++) {
                    if (row_values[MAX_EXPERTS / 32 - 1 - expert_slot_finite_tie] == best_value) {
                        int expert_finite_tie = lane + (unsigned int)((MAX_EXPERTS / 32 - 1 - expert_slot_finite_tie) * 32);
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
            int stored_expert = selected_expert;
            if (M > 512 && M <= 2147483647 / (MAX_EXPERTS * MAX_TOP_K)) {
                int _atomic_old_0 = atomicAdd(&expert_counts[selected_expert], 1);
                int counted_row = _atomic_old_0;
                stored_expert = counted_row * MAX_EXPERTS + selected_expert;
            } else if (private_raw_hist == 0) {
                atomicAdd(&expert_counts[selected_expert], 1);
            }
            topk_ids[output_base + (unsigned long long)lane] = stored_expert;
        }
    }
    cooperative_groups::this_grid().sync();
    if (bid * THREADS < M * top_k || bid * THREADS < E) {
        int scan_counts[MAX_EXPERTS / THREADS];
        if (private_raw_hist != 0) {
            #pragma unroll
            for (int private_zero_slot = 0; private_zero_slot < 512 / THREADS; private_zero_slot++) {
                private_counts[tid + private_zero_slot * THREADS] = 0;
            }
            __syncthreads();
            for (int count_pair = tid; count_pair < M * top_k; count_pair += THREADS) {
                int count_expert = topk_ids[count_pair];
                atomicAdd(&private_counts[count_expert], 1);
            }
            __syncthreads();
            #pragma unroll
            for (int private_load_slot = 0; private_load_slot < MAX_EXPERTS / THREADS; private_load_slot++) {
                int private_expert = tid * (MAX_EXPERTS / THREADS) + private_load_slot;
                int private_count = private_counts[private_expert];
                scan_counts[private_load_slot] = private_count;
                if (bid == 0) {
                    expert_counts[private_expert] = private_count;
                }
            }
        } else if (MAX_EXPERTS == 512 && E == 512 && ((unsigned long long)expert_counts & 15) == 0) {
            {
                int4 _iv4 = *reinterpret_cast<const int4*>(expert_counts + tid * 4);
                scan_counts[0 + 0] = _iv4.x;
                scan_counts[0 + 1] = _iv4.y;
                scan_counts[0 + 2] = _iv4.z;
                scan_counts[0 + 3] = _iv4.w;
            }
        } else {
            #pragma unroll
            for (int expert_slot_count_load = 0; expert_slot_count_load < MAX_EXPERTS / THREADS; expert_slot_count_load++) {
                int expert_count_load = tid * (MAX_EXPERTS / THREADS) + expert_slot_count_load;
                int count_loaded = 0;
                if (expert_count_load < E) {
                    count_loaded = expert_counts[expert_count_load];
                }
                scan_counts[expert_slot_count_load] = count_loaded;
            }
        }
        int scan_padded[MAX_EXPERTS / THREADS];
        int scan_thread_total = 0;
        #pragma unroll
        for (int expert_slot_scan_init = 0; expert_slot_scan_init < MAX_EXPERTS / THREADS; expert_slot_scan_init++) {
            int expert_scan_init = tid * (MAX_EXPERTS / THREADS) + expert_slot_scan_init;
            int padded_count_init = 0;
            if (expert_scan_init < E) {
                int count_init = scan_counts[expert_slot_scan_init];
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
        if (MAX_EXPERTS == 512 && E == 512) {
            #pragma unroll
            for (int expert_slot_full_prefix = 0; expert_slot_full_prefix < MAX_EXPERTS / THREADS; expert_slot_full_prefix++) {
                int expert_full_prefix = tid * (MAX_EXPERTS / THREADS) + expert_slot_full_prefix;
                local_offsets[expert_full_prefix] = scan_expert_prefix;
                if (bid == 0) {
                    expert_offsets[expert_full_prefix] = scan_expert_prefix;
                    if (M > 512 && M <= 2147483647 / (MAX_EXPERTS * MAX_TOP_K)) {
                        expert_scatter_offsets[expert_full_prefix] = expert_counts[expert_full_prefix];
                    }
                }
                scan_expert_prefix = scan_expert_prefix + scan_padded[expert_slot_full_prefix];
            }
        } else {
            #pragma unroll
            for (int expert_slot_scan_store = 0; expert_slot_scan_store < MAX_EXPERTS / THREADS; expert_slot_scan_store++) {
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
        }
        __syncthreads();
        int scatter_block_mask = block_m - 1;
        int scatter_block_shift = 0;
        if (block_m == 8) {
            scatter_block_shift = 3;
        } else if (block_m == 16) {
            scatter_block_shift = 4;
        }
        if (M * top_k <= num_bids * THREADS) {
            if (global_thread < M * top_k) {
                int pair = global_thread;
                int stored_pair = topk_ids[pair];
                int pair_expert = stored_pair;
                int local_row = 0;
                if (M > 512 && M <= 2147483647 / (MAX_EXPERTS * MAX_TOP_K)) {
                    pair_expert = stored_pair % MAX_EXPERTS;
                    local_row = stored_pair / MAX_EXPERTS;
                    topk_ids[pair] = pair_expert;
                } else {
                    int _atomic_old_1 = atomicAdd(&expert_scatter_offsets[pair_expert], 1);
                    local_row = _atomic_old_1;
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
        } else {
            for (int pair_1 = global_thread; pair_1 < M * top_k; pair_1 += num_bids * THREADS) {
                int stored_pair_1 = topk_ids[pair_1];
                int pair_expert_1 = stored_pair_1;
                int local_row_1 = 0;
                if (M > 512 && M <= 2147483647 / (MAX_EXPERTS * MAX_TOP_K)) {
                    pair_expert_1 = stored_pair_1 % MAX_EXPERTS;
                    local_row_1 = stored_pair_1 / MAX_EXPERTS;
                    topk_ids[pair_1] = pair_expert_1;
                } else {
                    int _atomic_old_2 = atomicAdd(&expert_scatter_offsets[pair_expert_1], 1);
                    local_row_1 = _atomic_old_2;
                }
                int grouped_row_1 = local_offsets[pair_expert_1] + local_row_1;
                sorted_token_ids[grouped_row_1] = pair_1;
                int scatter_block_remainder_1 = 0;
                if (scatter_block_shift != 0) {
                    scatter_block_remainder_1 = local_row_1 & scatter_block_mask;
                } else {
                    scatter_block_remainder_1 = local_row_1 % block_m;
                }
                if (scatter_block_remainder_1 == 0) {
                    int scatter_block_index_1 = 0;
                    if (scatter_block_shift != 0) {
                        scatter_block_index_1 = grouped_row_1 >> scatter_block_shift;
                    } else {
                        scatter_block_index_1 = grouped_row_1 / block_m;
                    }
                    expert_ids[scatter_block_index_1] = pair_expert_1;
                }
            }
        }
        for (int padding_expert = global_thread; padding_expert < E; padding_expert += num_bids * THREADS) {
            int count_final = 0;
            if (private_raw_hist != 0) {
                count_final = private_counts[padding_expert];
            } else {
                count_final = expert_counts[padding_expert];
            }
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
#undef SMEM_LOCAL_OFFSETS_OFF
#undef SMEM_LOCAL_OFFSETS_STAGE_BYTES
#undef SMEM_LOCAL_OFFSETS_STRIDE
#undef SMEM_PRIVATE_COUNTS_OFF
#undef SMEM_PRIVATE_COUNTS_STAGE_BYTES
#undef SMEM_PRIVATE_COUNTS_STRIDE
#undef SMEM_SCAN_VALUES_OFF
#undef SMEM_SCAN_VALUES_STAGE_BYTES
#undef SMEM_SCAN_VALUES_STRIDE
#undef SMEM_TOTAL
#undef THREADS
}  // namespace alphamoe_router_medium_routed_generated

using alphamoe_router_medium_routed_generated::kernel_alpha_moe_fused_router_medium_routed;

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
            if (best_value != -LOOM_INF) {
                #pragma unroll
                for (int expert_slot_finite_tie = 0; expert_slot_finite_tie < MAX_EXPERTS / 32; expert_slot_finite_tie++) {
                    if (row_values[MAX_EXPERTS / 32 - 1 - expert_slot_finite_tie] == best_value) {
                        int expert_finite_tie = lane + (unsigned int)((MAX_EXPERTS / 32 - 1 - expert_slot_finite_tie) * 32);
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
        int scan_padded[MAX_EXPERTS / THREADS];
        int scan_thread_total = 0;
        #pragma unroll
        for (int expert_slot_scan_init = 0; expert_slot_scan_init < MAX_EXPERTS / THREADS; expert_slot_scan_init++) {
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
        for (int expert_slot_scan_store = 0; expert_slot_scan_store < MAX_EXPERTS / THREADS; expert_slot_scan_store++) {
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

using alphamoe_router_small_generated::kernel_alpha_moe_fused_router_small;

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
            if (best_value != -LOOM_INF) {
                #pragma unroll
                for (int expert_slot_finite_tie = 0; expert_slot_finite_tie < MAX_EXPERTS / 32; expert_slot_finite_tie++) {
                    if (row_values[MAX_EXPERTS / 32 - 1 - expert_slot_finite_tie] == best_value) {
                        int expert_finite_tie = lane + (unsigned int)((MAX_EXPERTS / 32 - 1 - expert_slot_finite_tie) * 32);
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
        int scan_padded[MAX_EXPERTS / THREADS];
        int scan_thread_total = 0;
        #pragma unroll
        for (int expert_slot_scan_init = 0; expert_slot_scan_init < MAX_EXPERTS / THREADS; expert_slot_scan_init++) {
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
        for (int expert_slot_scan_store = 0; expert_slot_scan_store < MAX_EXPERTS / THREADS; expert_slot_scan_store++) {
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

using alphamoe_router_small_routed_generated::kernel_alpha_moe_fused_router_small_routed;

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
            if (best_value != -LOOM_INF) {
                #pragma unroll
                for (int expert_slot_finite_tie = 0; expert_slot_finite_tie < MAX_EXPERTS / 32; expert_slot_finite_tie++) {
                    if (row_values[MAX_EXPERTS / 32 - 1 - expert_slot_finite_tie] == best_value) {
                        int expert_finite_tie = lane + (unsigned int)((MAX_EXPERTS / 32 - 1 - expert_slot_finite_tie) * 32);
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
        int scan_padded[MAX_EXPERTS / THREADS];
        int scan_thread_total = 0;
        #pragma unroll
        for (int expert_slot_scan_init = 0; expert_slot_scan_init < MAX_EXPERTS / THREADS; expert_slot_scan_init++) {
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
        for (int expert_slot_scan_store = 0; expert_slot_scan_store < MAX_EXPERTS / THREADS; expert_slot_scan_store++) {
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
            if (best_value != -LOOM_INF) {
                #pragma unroll
                for (int expert_slot_finite_tie = 0; expert_slot_finite_tie < MAX_EXPERTS / 32; expert_slot_finite_tie++) {
                    if (row_values[MAX_EXPERTS / 32 - 1 - expert_slot_finite_tie] == best_value) {
                        int expert_finite_tie = lane + (unsigned int)((MAX_EXPERTS / 32 - 1 - expert_slot_finite_tie) * 32);
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
        int scan_padded[MAX_EXPERTS / THREADS];
        int scan_thread_total = 0;
        #pragma unroll
        for (int expert_slot_scan_init = 0; expert_slot_scan_init < MAX_EXPERTS / THREADS; expert_slot_scan_init++) {
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
        for (int expert_slot_scan_store = 0; expert_slot_scan_store < MAX_EXPERTS / THREADS; expert_slot_scan_store++) {
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

namespace alphamoe_router_large_tail_generated {
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
extern "C" {

__global__ __launch_bounds__(256) void
kernel_alpha_moe_fused_router_large_tail(float* __restrict__ logits, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M, int E, int top_k, int block_m, int has_shared_expert)
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
        for (int expert_slot_scan_init = 0; expert_slot_scan_init < MAX_EXPERTS / THREADS; expert_slot_scan_init++) {
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
        for (int expert_slot_scan_store = 0; expert_slot_scan_store < MAX_EXPERTS / THREADS; expert_slot_scan_store++) {
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
        for (int padding_expert = global_thread; padding_expert < E; padding_expert += num_bids * THREADS) {
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
#undef SMEM_LOCAL_OFFSETS_OFF
#undef SMEM_LOCAL_OFFSETS_STAGE_BYTES
#undef SMEM_LOCAL_OFFSETS_STRIDE
#undef SMEM_SCAN_VALUES_OFF
#undef SMEM_SCAN_VALUES_STAGE_BYTES
#undef SMEM_SCAN_VALUES_STRIDE
#undef SMEM_TOTAL
#undef THREADS
}  // namespace alphamoe_router_large_tail_generated

using alphamoe_router_large_tail_generated::kernel_alpha_moe_fused_router_large_tail;

// clang-format on

namespace flashinfer {
namespace alphamoe_fused_router {

constexpr int64_t kMaxExperts = 512;
constexpr int64_t kMaxTopK = 16;
constexpr int64_t kMaxBlockM = 16;
inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

struct RouterLaunchConfig {
  int sm_count;
  int active_blocks_per_sm;
  int medium_active_blocks_per_sm;
  int small_active_blocks_per_sm;
};

inline RouterLaunchConfig GetRouterLaunchConfig(int32_t device_id) {
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
                                 alphamoe_router_large_generated::kSmemTotal),
            "cudaFuncSetAttribute(AlphaMoE large dynamic smem)");
  int large_active = 0;
  CheckCuda(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&large_active, kernel_alpha_moe_fused_router,
                                                    alphamoe_router_large_generated::kThreads,
                                                    alphamoe_router_large_generated::kSmemTotal),
      "cudaOccupancyMaxActiveBlocksPerMultiprocessor(AlphaMoE large)");
  TVM_FFI_ICHECK(large_active > 0) << "AlphaMoE large has zero occupancy";
  CheckCuda(cudaFuncSetAttribute(kernel_alpha_moe_fused_router_routed,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 alphamoe_router_large_routed_generated::kSmemTotal),
            "cudaFuncSetAttribute(AlphaMoE large_routed dynamic smem)");
  int large_routed_active = 0;
  CheckCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &large_routed_active, kernel_alpha_moe_fused_router_routed,
                alphamoe_router_large_routed_generated::kThreads,
                alphamoe_router_large_routed_generated::kSmemTotal),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor(AlphaMoE large_routed)");
  TVM_FFI_ICHECK(large_routed_active > 0) << "AlphaMoE large_routed has zero occupancy";
  CheckCuda(cudaFuncSetAttribute(kernel_alpha_moe_fused_router_medium,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 alphamoe_router_medium_generated::kSmemTotal),
            "cudaFuncSetAttribute(AlphaMoE medium dynamic smem)");
  int medium_active = 0;
  CheckCuda(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &medium_active, kernel_alpha_moe_fused_router_medium,
          alphamoe_router_medium_generated::kThreads, alphamoe_router_medium_generated::kSmemTotal),
      "cudaOccupancyMaxActiveBlocksPerMultiprocessor(AlphaMoE medium)");
  TVM_FFI_ICHECK(medium_active > 0) << "AlphaMoE medium has zero occupancy";
  CheckCuda(cudaFuncSetAttribute(kernel_alpha_moe_fused_router_medium_routed,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 alphamoe_router_medium_routed_generated::kSmemTotal),
            "cudaFuncSetAttribute(AlphaMoE medium_routed dynamic smem)");
  int medium_routed_active = 0;
  CheckCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &medium_routed_active, kernel_alpha_moe_fused_router_medium_routed,
                alphamoe_router_medium_routed_generated::kThreads,
                alphamoe_router_medium_routed_generated::kSmemTotal),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor(AlphaMoE medium_routed)");
  TVM_FFI_ICHECK(medium_routed_active > 0) << "AlphaMoE medium_routed has zero occupancy";
  CheckCuda(cudaFuncSetAttribute(kernel_alpha_moe_fused_router_small,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 alphamoe_router_small_generated::kSmemTotal),
            "cudaFuncSetAttribute(AlphaMoE small dynamic smem)");
  int small_active = 0;
  CheckCuda(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &small_active, kernel_alpha_moe_fused_router_small,
          alphamoe_router_small_generated::kThreads, alphamoe_router_small_generated::kSmemTotal),
      "cudaOccupancyMaxActiveBlocksPerMultiprocessor(AlphaMoE small)");
  TVM_FFI_ICHECK(small_active > 0) << "AlphaMoE small has zero occupancy";
  CheckCuda(cudaFuncSetAttribute(kernel_alpha_moe_fused_router_small_routed,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 alphamoe_router_small_routed_generated::kSmemTotal),
            "cudaFuncSetAttribute(AlphaMoE small_routed dynamic smem)");
  int small_routed_active = 0;
  CheckCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &small_routed_active, kernel_alpha_moe_fused_router_small_routed,
                alphamoe_router_small_routed_generated::kThreads,
                alphamoe_router_small_routed_generated::kSmemTotal),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor(AlphaMoE small_routed)");
  TVM_FFI_ICHECK(small_routed_active > 0) << "AlphaMoE small_routed has zero occupancy";
  CheckCuda(cudaFuncSetAttribute(kernel_alpha_moe_fused_router_tiny,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 alphamoe_router_tiny_generated::kSmemTotal),
            "cudaFuncSetAttribute(AlphaMoE tiny dynamic smem)");
  int tiny_active = 0;
  CheckCuda(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &tiny_active, kernel_alpha_moe_fused_router_tiny,
          alphamoe_router_tiny_generated::kThreads, alphamoe_router_tiny_generated::kSmemTotal),
      "cudaOccupancyMaxActiveBlocksPerMultiprocessor(AlphaMoE tiny)");
  TVM_FFI_ICHECK(tiny_active > 0) << "AlphaMoE tiny has zero occupancy";
  CheckCuda(cudaFuncSetAttribute(kernel_alpha_moe_fused_router_tiny_routed,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 alphamoe_router_tiny_routed_generated::kSmemTotal),
            "cudaFuncSetAttribute(AlphaMoE tiny_routed dynamic smem)");
  int tiny_routed_active = 0;
  CheckCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &tiny_routed_active, kernel_alpha_moe_fused_router_tiny_routed,
                alphamoe_router_tiny_routed_generated::kThreads,
                alphamoe_router_tiny_routed_generated::kSmemTotal),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor(AlphaMoE tiny_routed)");
  TVM_FFI_ICHECK(tiny_routed_active > 0) << "AlphaMoE tiny_routed has zero occupancy";
  CheckCuda(cudaFuncSetAttribute(kernel_alpha_moe_fused_router_large_tail,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 alphamoe_router_large_tail_generated::kSmemTotal),
            "cudaFuncSetAttribute(AlphaMoE large_tail dynamic smem)");
  int large_tail_active = 0;
  CheckCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &large_tail_active, kernel_alpha_moe_fused_router_large_tail,
                alphamoe_router_large_tail_generated::kThreads,
                alphamoe_router_large_tail_generated::kSmemTotal),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor(AlphaMoE large_tail)");
  TVM_FFI_ICHECK(large_tail_active > 0) << "AlphaMoE large_tail has zero occupancy";
  const RouterLaunchConfig config{
      sm_count, std::min(large_active, large_routed_active),
      std::min(medium_active, medium_routed_active),
      std::min({small_active, small_routed_active, tiny_active, tiny_routed_active})};
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
  const bool use_tiny = m <= 8;
  const bool use_small = m <= 128;
  const bool use_medium = m <= 512;
  int64_t grid_x =
      use_tiny ? 1 : std::max<int64_t>(1, std::min<int64_t>((m + 7) / 8, config.sm_count));
  int active_blocks = config.small_active_blocks_per_sm;
  if (!use_small && use_medium) {
    active_blocks = config.medium_active_blocks_per_sm;
    grid_x = std::max<int64_t>(
        1, std::min<int64_t>((m + 3) / 4, static_cast<int64_t>(config.sm_count) * active_blocks));
  } else if (!use_medium) {
    active_blocks = config.active_blocks_per_sm;
    const int64_t full_wave_grid =
        (((m + 3) / 4 + config.sm_count - 1) / config.sm_count) * config.sm_count;
    grid_x = std::max<int64_t>(
        1, std::min<int64_t>(full_wave_grid,
                             static_cast<int64_t>(config.sm_count) * std::min(5, active_blocks)));
  }
  const int64_t cooperative_capacity = static_cast<int64_t>(active_blocks) * config.sm_count;
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
  if (use_tiny) {
    CheckCuda(cudaLaunchKernel(reinterpret_cast<const void*>(
                                   has_shared_expert ? kernel_alpha_moe_fused_router_tiny
                                                     : kernel_alpha_moe_fused_router_tiny_routed),
                               dim3(static_cast<unsigned int>(grid_x), 1, 1),
                               dim3(alphamoe_router_tiny_generated::kThreads, 1, 1), arguments,
                               alphamoe_router_tiny_generated::kSmemTotal, stream),
              "cudaLaunchKernel(AlphaMoE tiny)");
  } else if (use_small) {
    CheckCuda(cudaLaunchCooperativeKernel(
                  reinterpret_cast<const void*>(has_shared_expert
                                                    ? kernel_alpha_moe_fused_router_small
                                                    : kernel_alpha_moe_fused_router_small_routed),
                  dim3(static_cast<unsigned int>(grid_x), 1, 1),
                  dim3(alphamoe_router_small_generated::kThreads, 1, 1), arguments,
                  alphamoe_router_small_generated::kSmemTotal, stream),
              "cudaLaunchCooperativeKernel(AlphaMoE small)");
  } else if (use_medium) {
    CheckCuda(cudaLaunchCooperativeKernel(
                  reinterpret_cast<const void*>(has_shared_expert
                                                    ? kernel_alpha_moe_fused_router_medium
                                                    : kernel_alpha_moe_fused_router_medium_routed),
                  dim3(static_cast<unsigned int>(grid_x), 1, 1),
                  dim3(alphamoe_router_medium_generated::kThreads, 1, 1), arguments,
                  alphamoe_router_medium_generated::kSmemTotal, stream),
              "cudaLaunchCooperativeKernel(AlphaMoE medium)");
  } else {
    CheckCuda(
        cudaLaunchCooperativeKernel(
            reinterpret_cast<const void*>(has_shared_expert ? kernel_alpha_moe_fused_router
                                                            : kernel_alpha_moe_fused_router_routed),
            dim3(static_cast<unsigned int>(grid_x), 1, 1),
            dim3(alphamoe_router_large_generated::kThreads, 1, 1), arguments,
            alphamoe_router_large_generated::kSmemTotal, stream),
        "cudaLaunchCooperativeKernel(AlphaMoE large)");
    CheckCuda(
        cudaLaunchKernel(reinterpret_cast<const void*>(kernel_alpha_moe_fused_router_large_tail),
                         dim3(static_cast<unsigned int>(grid_x), 1, 1),
                         dim3(alphamoe_router_large_tail_generated::kThreads, 1, 1), arguments,
                         alphamoe_router_large_tail_generated::kSmemTotal, stream),
        "cudaLaunchKernel(AlphaMoE large tail)");
  }
}

}  // namespace alphamoe_fused_router
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_router_op, flashinfer::alphamoe_fused_router::Run);
