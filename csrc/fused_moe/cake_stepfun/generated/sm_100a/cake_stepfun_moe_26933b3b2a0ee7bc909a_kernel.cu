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

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_EXPERT_COUNT_OFF 0
#define SMEM_SMEM_EXPERT_COUNT_STAGE_BYTES 512
#define SMEM_SMEM_EXPERT_COUNT_STRIDE 512
#define SMEM_SMEM_EXPERT_OFFSET_OFF 512
#define SMEM_SMEM_EXPERT_OFFSET_STAGE_BYTES 512
#define SMEM_SMEM_EXPERT_OFFSET_STRIDE 512
#define SMEM_WARP_TOTALS_OFF 1024
#define SMEM_WARP_TOTALS_STAGE_BYTES 128
#define SMEM_WARP_TOTALS_STRIDE 128
#define SMEM_TOTAL 1152
#define THREADS 128

#include <math_constants.h>
#include <cooperative_groups.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_stepfun_moe_26933b3b2a0ee7bc909a(float* __restrict__ scores, __nv_bfloat16* __restrict__ topk_weights, int* __restrict__ topk_packed, int* __restrict__ expert_counts, int* __restrict__ permuted_idx_size, int* __restrict__ expanded_idx_to_permuted_idx, int* __restrict__ permuted_idx_to_token_idx, int* __restrict__ cta_idx_xy_to_batch_idx, int* __restrict__ cta_idx_xy_to_mn_limit, int* __restrict__ num_non_exiting_ctas, int num_tokens, int num_experts, int top_k, int padding_log2, int tile_tokens_dim, int local_experts_start_idx, int local_experts_stride_log2, int num_local_experts)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    unsigned int* smem_expert_count = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int smem_expert_count_addr = smem + 0;
    int* smem_expert_offset = reinterpret_cast<int*>(smem_raw + 512);
    const int smem_expert_offset_addr = smem + 512;
    int* warp_totals = reinterpret_cast<int*>(smem_raw + 1024);
    const int warp_totals_addr = smem + 1024;

    // === Task calls (dependency order) ===
    int tid_0 = (int)tid;
    int bid_1 = (int)bid;
    int nbids = (int)num_bids;
    int neg_four[4];
    neg_four[0] = -1;
    neg_four[1] = -1;
    neg_four[2] = -1;
    neg_four[3] = -1;
    smem_expert_count[tid_0] = 0;
    __syncthreads();
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int grid_tid = 128 * bid_1 + tid_0;
    int grid_threads = nbids * 128;
    int expanded_size = num_tokens * top_k;
    int exp_idx[64];
    int exp_off[64];
    int done = 0;
    int expanded = 0;
    if (done == 0) {
        if (expanded_size >= 4 * grid_threads) {
            expanded = grid_tid;
            int idx_ii = 0;
            unsigned int word0 = 0;
            unsigned int word1 = 0;
            unsigned int remote = 0;
            int src_rank = 0;
            int src_slot = 0;
            int packed_word = 0;
            packed_word = topk_packed[expanded];
            idx_ii = packed_word >> 16;
            word0 = (unsigned int)packed_word & 65535;
            exp_idx[0] = idx_ii;
            int local_idx = idx_ii - local_experts_start_idx;
            int extent = num_local_experts << local_experts_stride_log2;
            int stride_mask = (1 << local_experts_stride_log2) - 1;
            int is_local = ((local_idx >= 0 && local_idx < extent && (local_idx & stride_mask) == 0) ? 1 : 0);
            int is_local_ii = is_local;
            exp_off[0] = 0;
            if (is_local_ii != 0) {
                uint32_t _shared_atomic_old_0;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_0) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[0] = (int)_shared_atomic_old_0;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0 << 16);
            expanded = grid_tid + grid_threads;
            int idx_ii_0 = 0;
            unsigned int word0_1 = 0;
            unsigned int word1_2 = 0;
            unsigned int remote_3 = 0;
            int src_rank_4 = 0;
            int src_slot_5 = 0;
            int packed_word_6 = 0;
            packed_word_6 = topk_packed[expanded];
            idx_ii_0 = packed_word_6 >> 16;
            word0_1 = (unsigned int)packed_word_6 & 65535;
            exp_idx[1] = idx_ii_0;
            int local_idx_7 = idx_ii_0 - local_experts_start_idx;
            int extent_8 = num_local_experts << local_experts_stride_log2;
            int stride_mask_9 = (1 << local_experts_stride_log2) - 1;
            int is_local_10 = ((local_idx_7 >= 0 && local_idx_7 < extent_8 && (local_idx_7 & stride_mask_9) == 0) ? 1 : 0);
            int is_local_ii_11 = is_local_10;
            exp_off[1] = 0;
            if (is_local_ii_11 != 0) {
                uint32_t _shared_atomic_old_1;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_1) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_0)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[1] = (int)_shared_atomic_old_1;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_1 << 16);
            expanded = grid_tid + 2 * grid_threads;
            int idx_ii_12 = 0;
            unsigned int word0_13 = 0;
            unsigned int word1_14 = 0;
            unsigned int remote_15 = 0;
            int src_rank_16 = 0;
            int src_slot_17 = 0;
            int packed_word_18 = 0;
            packed_word_18 = topk_packed[expanded];
            idx_ii_12 = packed_word_18 >> 16;
            word0_13 = (unsigned int)packed_word_18 & 65535;
            exp_idx[2] = idx_ii_12;
            int local_idx_19 = idx_ii_12 - local_experts_start_idx;
            int extent_20 = num_local_experts << local_experts_stride_log2;
            int stride_mask_21 = (1 << local_experts_stride_log2) - 1;
            int is_local_22 = ((local_idx_19 >= 0 && local_idx_19 < extent_20 && (local_idx_19 & stride_mask_21) == 0) ? 1 : 0);
            int is_local_ii_23 = is_local_22;
            exp_off[2] = 0;
            if (is_local_ii_23 != 0) {
                uint32_t _shared_atomic_old_2;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_2) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_12)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[2] = (int)_shared_atomic_old_2;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_13 << 16);
            expanded = grid_tid + 3 * grid_threads;
            int idx_ii_24 = 0;
            unsigned int word0_25 = 0;
            unsigned int word1_26 = 0;
            unsigned int remote_27 = 0;
            int src_rank_28 = 0;
            int src_slot_29 = 0;
            int packed_word_30 = 0;
            packed_word_30 = topk_packed[expanded];
            idx_ii_24 = packed_word_30 >> 16;
            word0_25 = (unsigned int)packed_word_30 & 65535;
            exp_idx[3] = idx_ii_24;
            int local_idx_31 = idx_ii_24 - local_experts_start_idx;
            int extent_32 = num_local_experts << local_experts_stride_log2;
            int stride_mask_33 = (1 << local_experts_stride_log2) - 1;
            int is_local_34 = ((local_idx_31 >= 0 && local_idx_31 < extent_32 && (local_idx_31 & stride_mask_33) == 0) ? 1 : 0);
            int is_local_ii_35 = is_local_34;
            exp_off[3] = 0;
            if (is_local_ii_35 != 0) {
                uint32_t _shared_atomic_old_3;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_3) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_24)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[3] = (int)_shared_atomic_old_3;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_25 << 16);
        } else {
            if (done == 0) {
                expanded = grid_tid;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_1 = 0;
                    unsigned int word0_2 = 0;
                    unsigned int word1_1 = 0;
                    unsigned int remote_1 = 0;
                    int src_rank_1 = 0;
                    int src_slot_1 = 0;
                    int packed_word_1 = 0;
                    packed_word_1 = topk_packed[expanded];
                    idx_ii_1 = packed_word_1 >> 16;
                    word0_2 = (unsigned int)packed_word_1 & 65535;
                    exp_idx[0] = idx_ii_1;
                    int local_idx_1 = idx_ii_1 - local_experts_start_idx;
                    int extent_1 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_1 = (1 << local_experts_stride_log2) - 1;
                    int is_local_1 = ((local_idx_1 >= 0 && local_idx_1 < extent_1 && (local_idx_1 & stride_mask_1) == 0) ? 1 : 0);
                    int is_local_ii_1 = is_local_1;
                    exp_off[0] = 0;
                    if (is_local_ii_1 != 0) {
                        uint32_t _shared_atomic_old_4;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_4) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_1)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[0] = (int)_shared_atomic_old_4;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_2 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_2 = 0;
                    unsigned int word0_3 = 0;
                    unsigned int word1_3 = 0;
                    unsigned int remote_2 = 0;
                    int src_rank_2 = 0;
                    int src_slot_2 = 0;
                    int packed_word_2 = 0;
                    packed_word_2 = topk_packed[expanded];
                    idx_ii_2 = packed_word_2 >> 16;
                    word0_3 = (unsigned int)packed_word_2 & 65535;
                    exp_idx[1] = idx_ii_2;
                    int local_idx_2 = idx_ii_2 - local_experts_start_idx;
                    int extent_2 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_2 = (1 << local_experts_stride_log2) - 1;
                    int is_local_2 = ((local_idx_2 >= 0 && local_idx_2 < extent_2 && (local_idx_2 & stride_mask_2) == 0) ? 1 : 0);
                    int is_local_ii_2 = is_local_2;
                    exp_off[1] = 0;
                    if (is_local_ii_2 != 0) {
                        uint32_t _shared_atomic_old_5;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_5) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_2)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[1] = (int)_shared_atomic_old_5;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_3 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 2 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_3 = 0;
                    unsigned int word0_4 = 0;
                    unsigned int word1_4 = 0;
                    unsigned int remote_4 = 0;
                    int src_rank_3 = 0;
                    int src_slot_3 = 0;
                    int packed_word_3 = 0;
                    packed_word_3 = topk_packed[expanded];
                    idx_ii_3 = packed_word_3 >> 16;
                    word0_4 = (unsigned int)packed_word_3 & 65535;
                    exp_idx[2] = idx_ii_3;
                    int local_idx_3 = idx_ii_3 - local_experts_start_idx;
                    int extent_3 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_3 = (1 << local_experts_stride_log2) - 1;
                    int is_local_3 = ((local_idx_3 >= 0 && local_idx_3 < extent_3 && (local_idx_3 & stride_mask_3) == 0) ? 1 : 0);
                    int is_local_ii_3 = is_local_3;
                    exp_off[2] = 0;
                    if (is_local_ii_3 != 0) {
                        uint32_t _shared_atomic_old_6;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_6) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_3)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[2] = (int)_shared_atomic_old_6;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_4 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 3 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_4 = 0;
                    unsigned int word0_5 = 0;
                    unsigned int word1_5 = 0;
                    unsigned int remote_5 = 0;
                    int src_rank_5 = 0;
                    int src_slot_4 = 0;
                    int packed_word_4 = 0;
                    packed_word_4 = topk_packed[expanded];
                    idx_ii_4 = packed_word_4 >> 16;
                    word0_5 = (unsigned int)packed_word_4 & 65535;
                    exp_idx[3] = idx_ii_4;
                    int local_idx_4 = idx_ii_4 - local_experts_start_idx;
                    int extent_4 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_4 = (1 << local_experts_stride_log2) - 1;
                    int is_local_4 = ((local_idx_4 >= 0 && local_idx_4 < extent_4 && (local_idx_4 & stride_mask_4) == 0) ? 1 : 0);
                    int is_local_ii_4 = is_local_4;
                    exp_off[3] = 0;
                    if (is_local_ii_4 != 0) {
                        uint32_t _shared_atomic_old_7;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_7) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_4)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[3] = (int)_shared_atomic_old_7;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_5 << 16);
                }
            }
        }
    }
    if (done == 0) {
        if (expanded_size >= 8 * grid_threads) {
            expanded = grid_tid + 4 * grid_threads;
            int idx_ii_5 = 0;
            unsigned int word0_6 = 0;
            unsigned int word1_6 = 0;
            unsigned int remote_6 = 0;
            int src_rank_6 = 0;
            int src_slot_6 = 0;
            int packed_word_5 = 0;
            packed_word_5 = topk_packed[expanded];
            idx_ii_5 = packed_word_5 >> 16;
            word0_6 = (unsigned int)packed_word_5 & 65535;
            exp_idx[4] = idx_ii_5;
            int local_idx_5 = idx_ii_5 - local_experts_start_idx;
            int extent_5 = num_local_experts << local_experts_stride_log2;
            int stride_mask_5 = (1 << local_experts_stride_log2) - 1;
            int is_local_5 = ((local_idx_5 >= 0 && local_idx_5 < extent_5 && (local_idx_5 & stride_mask_5) == 0) ? 1 : 0);
            int is_local_ii_5 = is_local_5;
            exp_off[4] = 0;
            if (is_local_ii_5 != 0) {
                uint32_t _shared_atomic_old_8;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_8) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_5)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[4] = (int)_shared_atomic_old_8;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_6 << 16);
            expanded = grid_tid + 5 * grid_threads;
            int idx_ii_0_1 = 0;
            unsigned int word0_1_1 = 0;
            unsigned int word1_2_1 = 0;
            unsigned int remote_3_1 = 0;
            int src_rank_4_1 = 0;
            int src_slot_5_1 = 0;
            int packed_word_6_1 = 0;
            packed_word_6_1 = topk_packed[expanded];
            idx_ii_0_1 = packed_word_6_1 >> 16;
            word0_1_1 = (unsigned int)packed_word_6_1 & 65535;
            exp_idx[5] = idx_ii_0_1;
            int local_idx_7_1 = idx_ii_0_1 - local_experts_start_idx;
            int extent_8_1 = num_local_experts << local_experts_stride_log2;
            int stride_mask_9_1 = (1 << local_experts_stride_log2) - 1;
            int is_local_10_1 = ((local_idx_7_1 >= 0 && local_idx_7_1 < extent_8_1 && (local_idx_7_1 & stride_mask_9_1) == 0) ? 1 : 0);
            int is_local_ii_11_1 = is_local_10_1;
            exp_off[5] = 0;
            if (is_local_ii_11_1 != 0) {
                uint32_t _shared_atomic_old_9;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_9) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_0_1)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[5] = (int)_shared_atomic_old_9;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_1_1 << 16);
            expanded = grid_tid + 6 * grid_threads;
            int idx_ii_12_1 = 0;
            unsigned int word0_13_1 = 0;
            unsigned int word1_14_1 = 0;
            unsigned int remote_15_1 = 0;
            int src_rank_16_1 = 0;
            int src_slot_17_1 = 0;
            int packed_word_18_1 = 0;
            packed_word_18_1 = topk_packed[expanded];
            idx_ii_12_1 = packed_word_18_1 >> 16;
            word0_13_1 = (unsigned int)packed_word_18_1 & 65535;
            exp_idx[6] = idx_ii_12_1;
            int local_idx_19_1 = idx_ii_12_1 - local_experts_start_idx;
            int extent_20_1 = num_local_experts << local_experts_stride_log2;
            int stride_mask_21_1 = (1 << local_experts_stride_log2) - 1;
            int is_local_22_1 = ((local_idx_19_1 >= 0 && local_idx_19_1 < extent_20_1 && (local_idx_19_1 & stride_mask_21_1) == 0) ? 1 : 0);
            int is_local_ii_23_1 = is_local_22_1;
            exp_off[6] = 0;
            if (is_local_ii_23_1 != 0) {
                uint32_t _shared_atomic_old_10;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_10) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_12_1)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[6] = (int)_shared_atomic_old_10;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_13_1 << 16);
            expanded = grid_tid + 7 * grid_threads;
            int idx_ii_24_1 = 0;
            unsigned int word0_25_1 = 0;
            unsigned int word1_26_1 = 0;
            unsigned int remote_27_1 = 0;
            int src_rank_28_1 = 0;
            int src_slot_29_1 = 0;
            int packed_word_30_1 = 0;
            packed_word_30_1 = topk_packed[expanded];
            idx_ii_24_1 = packed_word_30_1 >> 16;
            word0_25_1 = (unsigned int)packed_word_30_1 & 65535;
            exp_idx[7] = idx_ii_24_1;
            int local_idx_31_1 = idx_ii_24_1 - local_experts_start_idx;
            int extent_32_1 = num_local_experts << local_experts_stride_log2;
            int stride_mask_33_1 = (1 << local_experts_stride_log2) - 1;
            int is_local_34_1 = ((local_idx_31_1 >= 0 && local_idx_31_1 < extent_32_1 && (local_idx_31_1 & stride_mask_33_1) == 0) ? 1 : 0);
            int is_local_ii_35_1 = is_local_34_1;
            exp_off[7] = 0;
            if (is_local_ii_35_1 != 0) {
                uint32_t _shared_atomic_old_11;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_11) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_24_1)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[7] = (int)_shared_atomic_old_11;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_25_1 << 16);
        } else {
            if (done == 0) {
                expanded = grid_tid + 4 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_6 = 0;
                    unsigned int word0_7 = 0;
                    unsigned int word1_7 = 0;
                    unsigned int remote_7 = 0;
                    int src_rank_7 = 0;
                    int src_slot_7 = 0;
                    int packed_word_7 = 0;
                    packed_word_7 = topk_packed[expanded];
                    idx_ii_6 = packed_word_7 >> 16;
                    word0_7 = (unsigned int)packed_word_7 & 65535;
                    exp_idx[4] = idx_ii_6;
                    int local_idx_6 = idx_ii_6 - local_experts_start_idx;
                    int extent_6 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_6 = (1 << local_experts_stride_log2) - 1;
                    int is_local_6 = ((local_idx_6 >= 0 && local_idx_6 < extent_6 && (local_idx_6 & stride_mask_6) == 0) ? 1 : 0);
                    int is_local_ii_6 = is_local_6;
                    exp_off[4] = 0;
                    if (is_local_ii_6 != 0) {
                        uint32_t _shared_atomic_old_12;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_12) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_6)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[4] = (int)_shared_atomic_old_12;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_7 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 5 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_7 = 0;
                    unsigned int word0_8 = 0;
                    unsigned int word1_8 = 0;
                    unsigned int remote_8 = 0;
                    int src_rank_8 = 0;
                    int src_slot_8 = 0;
                    int packed_word_8 = 0;
                    packed_word_8 = topk_packed[expanded];
                    idx_ii_7 = packed_word_8 >> 16;
                    word0_8 = (unsigned int)packed_word_8 & 65535;
                    exp_idx[5] = idx_ii_7;
                    int local_idx_8 = idx_ii_7 - local_experts_start_idx;
                    int extent_7 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_7 = (1 << local_experts_stride_log2) - 1;
                    int is_local_7 = ((local_idx_8 >= 0 && local_idx_8 < extent_7 && (local_idx_8 & stride_mask_7) == 0) ? 1 : 0);
                    int is_local_ii_7 = is_local_7;
                    exp_off[5] = 0;
                    if (is_local_ii_7 != 0) {
                        uint32_t _shared_atomic_old_13;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_13) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_7)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[5] = (int)_shared_atomic_old_13;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_8 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 6 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_8 = 0;
                    unsigned int word0_9 = 0;
                    unsigned int word1_9 = 0;
                    unsigned int remote_9 = 0;
                    int src_rank_9 = 0;
                    int src_slot_9 = 0;
                    int packed_word_9 = 0;
                    packed_word_9 = topk_packed[expanded];
                    idx_ii_8 = packed_word_9 >> 16;
                    word0_9 = (unsigned int)packed_word_9 & 65535;
                    exp_idx[6] = idx_ii_8;
                    int local_idx_9 = idx_ii_8 - local_experts_start_idx;
                    int extent_9 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_8 = (1 << local_experts_stride_log2) - 1;
                    int is_local_8 = ((local_idx_9 >= 0 && local_idx_9 < extent_9 && (local_idx_9 & stride_mask_8) == 0) ? 1 : 0);
                    int is_local_ii_8 = is_local_8;
                    exp_off[6] = 0;
                    if (is_local_ii_8 != 0) {
                        uint32_t _shared_atomic_old_14;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_14) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_8)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[6] = (int)_shared_atomic_old_14;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_9 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 7 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_9 = 0;
                    unsigned int word0_10 = 0;
                    unsigned int word1_10 = 0;
                    unsigned int remote_10 = 0;
                    int src_rank_10 = 0;
                    int src_slot_10 = 0;
                    int packed_word_10 = 0;
                    packed_word_10 = topk_packed[expanded];
                    idx_ii_9 = packed_word_10 >> 16;
                    word0_10 = (unsigned int)packed_word_10 & 65535;
                    exp_idx[7] = idx_ii_9;
                    int local_idx_10 = idx_ii_9 - local_experts_start_idx;
                    int extent_10 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_10 = (1 << local_experts_stride_log2) - 1;
                    int is_local_9 = ((local_idx_10 >= 0 && local_idx_10 < extent_10 && (local_idx_10 & stride_mask_10) == 0) ? 1 : 0);
                    int is_local_ii_9 = is_local_9;
                    exp_off[7] = 0;
                    if (is_local_ii_9 != 0) {
                        uint32_t _shared_atomic_old_15;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_15) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_9)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[7] = (int)_shared_atomic_old_15;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_10 << 16);
                }
            }
        }
    }
    if (done == 0) {
        if (expanded_size >= 12 * grid_threads) {
            expanded = grid_tid + 8 * grid_threads;
            int idx_ii_10 = 0;
            unsigned int word0_11 = 0;
            unsigned int word1_11 = 0;
            unsigned int remote_11 = 0;
            int src_rank_11 = 0;
            int src_slot_11 = 0;
            int packed_word_11 = 0;
            packed_word_11 = topk_packed[expanded];
            idx_ii_10 = packed_word_11 >> 16;
            word0_11 = (unsigned int)packed_word_11 & 65535;
            exp_idx[8] = idx_ii_10;
            int local_idx_11 = idx_ii_10 - local_experts_start_idx;
            int extent_11 = num_local_experts << local_experts_stride_log2;
            int stride_mask_11 = (1 << local_experts_stride_log2) - 1;
            int is_local_11 = ((local_idx_11 >= 0 && local_idx_11 < extent_11 && (local_idx_11 & stride_mask_11) == 0) ? 1 : 0);
            int is_local_ii_10 = is_local_11;
            exp_off[8] = 0;
            if (is_local_ii_10 != 0) {
                uint32_t _shared_atomic_old_16;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_16) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_10)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[8] = (int)_shared_atomic_old_16;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_11 << 16);
            expanded = grid_tid + 9 * grid_threads;
            int idx_ii_0_2 = 0;
            unsigned int word0_1_2 = 0;
            unsigned int word1_2_2 = 0;
            unsigned int remote_3_2 = 0;
            int src_rank_4_2 = 0;
            int src_slot_5_2 = 0;
            int packed_word_6_2 = 0;
            packed_word_6_2 = topk_packed[expanded];
            idx_ii_0_2 = packed_word_6_2 >> 16;
            word0_1_2 = (unsigned int)packed_word_6_2 & 65535;
            exp_idx[9] = idx_ii_0_2;
            int local_idx_7_2 = idx_ii_0_2 - local_experts_start_idx;
            int extent_8_2 = num_local_experts << local_experts_stride_log2;
            int stride_mask_9_2 = (1 << local_experts_stride_log2) - 1;
            int is_local_10_2 = ((local_idx_7_2 >= 0 && local_idx_7_2 < extent_8_2 && (local_idx_7_2 & stride_mask_9_2) == 0) ? 1 : 0);
            int is_local_ii_11_2 = is_local_10_2;
            exp_off[9] = 0;
            if (is_local_ii_11_2 != 0) {
                uint32_t _shared_atomic_old_17;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_17) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_0_2)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[9] = (int)_shared_atomic_old_17;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_1_2 << 16);
            expanded = grid_tid + 10 * grid_threads;
            int idx_ii_12_2 = 0;
            unsigned int word0_13_2 = 0;
            unsigned int word1_14_2 = 0;
            unsigned int remote_15_2 = 0;
            int src_rank_16_2 = 0;
            int src_slot_17_2 = 0;
            int packed_word_18_2 = 0;
            packed_word_18_2 = topk_packed[expanded];
            idx_ii_12_2 = packed_word_18_2 >> 16;
            word0_13_2 = (unsigned int)packed_word_18_2 & 65535;
            exp_idx[10] = idx_ii_12_2;
            int local_idx_19_2 = idx_ii_12_2 - local_experts_start_idx;
            int extent_20_2 = num_local_experts << local_experts_stride_log2;
            int stride_mask_21_2 = (1 << local_experts_stride_log2) - 1;
            int is_local_22_2 = ((local_idx_19_2 >= 0 && local_idx_19_2 < extent_20_2 && (local_idx_19_2 & stride_mask_21_2) == 0) ? 1 : 0);
            int is_local_ii_23_2 = is_local_22_2;
            exp_off[10] = 0;
            if (is_local_ii_23_2 != 0) {
                uint32_t _shared_atomic_old_18;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_18) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_12_2)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[10] = (int)_shared_atomic_old_18;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_13_2 << 16);
            expanded = grid_tid + 11 * grid_threads;
            int idx_ii_24_2 = 0;
            unsigned int word0_25_2 = 0;
            unsigned int word1_26_2 = 0;
            unsigned int remote_27_2 = 0;
            int src_rank_28_2 = 0;
            int src_slot_29_2 = 0;
            int packed_word_30_2 = 0;
            packed_word_30_2 = topk_packed[expanded];
            idx_ii_24_2 = packed_word_30_2 >> 16;
            word0_25_2 = (unsigned int)packed_word_30_2 & 65535;
            exp_idx[11] = idx_ii_24_2;
            int local_idx_31_2 = idx_ii_24_2 - local_experts_start_idx;
            int extent_32_2 = num_local_experts << local_experts_stride_log2;
            int stride_mask_33_2 = (1 << local_experts_stride_log2) - 1;
            int is_local_34_2 = ((local_idx_31_2 >= 0 && local_idx_31_2 < extent_32_2 && (local_idx_31_2 & stride_mask_33_2) == 0) ? 1 : 0);
            int is_local_ii_35_2 = is_local_34_2;
            exp_off[11] = 0;
            if (is_local_ii_35_2 != 0) {
                uint32_t _shared_atomic_old_19;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_19) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_24_2)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[11] = (int)_shared_atomic_old_19;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_25_2 << 16);
        } else {
            if (done == 0) {
                expanded = grid_tid + 8 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_11 = 0;
                    unsigned int word0_12 = 0;
                    unsigned int word1_12 = 0;
                    unsigned int remote_12 = 0;
                    int src_rank_12 = 0;
                    int src_slot_12 = 0;
                    int packed_word_12 = 0;
                    packed_word_12 = topk_packed[expanded];
                    idx_ii_11 = packed_word_12 >> 16;
                    word0_12 = (unsigned int)packed_word_12 & 65535;
                    exp_idx[8] = idx_ii_11;
                    int local_idx_12 = idx_ii_11 - local_experts_start_idx;
                    int extent_12 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_12 = (1 << local_experts_stride_log2) - 1;
                    int is_local_12 = ((local_idx_12 >= 0 && local_idx_12 < extent_12 && (local_idx_12 & stride_mask_12) == 0) ? 1 : 0);
                    int is_local_ii_12 = is_local_12;
                    exp_off[8] = 0;
                    if (is_local_ii_12 != 0) {
                        uint32_t _shared_atomic_old_20;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_20) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_11)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[8] = (int)_shared_atomic_old_20;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_12 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 9 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_13 = 0;
                    unsigned int word0_14 = 0;
                    unsigned int word1_13 = 0;
                    unsigned int remote_13 = 0;
                    int src_rank_13 = 0;
                    int src_slot_13 = 0;
                    int packed_word_13 = 0;
                    packed_word_13 = topk_packed[expanded];
                    idx_ii_13 = packed_word_13 >> 16;
                    word0_14 = (unsigned int)packed_word_13 & 65535;
                    exp_idx[9] = idx_ii_13;
                    int local_idx_13 = idx_ii_13 - local_experts_start_idx;
                    int extent_13 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_13 = (1 << local_experts_stride_log2) - 1;
                    int is_local_13 = ((local_idx_13 >= 0 && local_idx_13 < extent_13 && (local_idx_13 & stride_mask_13) == 0) ? 1 : 0);
                    int is_local_ii_13 = is_local_13;
                    exp_off[9] = 0;
                    if (is_local_ii_13 != 0) {
                        uint32_t _shared_atomic_old_21;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_21) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_13)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[9] = (int)_shared_atomic_old_21;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_14 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 10 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_14 = 0;
                    unsigned int word0_15 = 0;
                    unsigned int word1_15 = 0;
                    unsigned int remote_14 = 0;
                    int src_rank_14 = 0;
                    int src_slot_14 = 0;
                    int packed_word_14 = 0;
                    packed_word_14 = topk_packed[expanded];
                    idx_ii_14 = packed_word_14 >> 16;
                    word0_15 = (unsigned int)packed_word_14 & 65535;
                    exp_idx[10] = idx_ii_14;
                    int local_idx_14 = idx_ii_14 - local_experts_start_idx;
                    int extent_14 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_14 = (1 << local_experts_stride_log2) - 1;
                    int is_local_14 = ((local_idx_14 >= 0 && local_idx_14 < extent_14 && (local_idx_14 & stride_mask_14) == 0) ? 1 : 0);
                    int is_local_ii_14 = is_local_14;
                    exp_off[10] = 0;
                    if (is_local_ii_14 != 0) {
                        uint32_t _shared_atomic_old_22;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_22) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_14)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[10] = (int)_shared_atomic_old_22;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_15 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 11 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_15 = 0;
                    unsigned int word0_16 = 0;
                    unsigned int word1_16 = 0;
                    unsigned int remote_16 = 0;
                    int src_rank_15 = 0;
                    int src_slot_15 = 0;
                    int packed_word_15 = 0;
                    packed_word_15 = topk_packed[expanded];
                    idx_ii_15 = packed_word_15 >> 16;
                    word0_16 = (unsigned int)packed_word_15 & 65535;
                    exp_idx[11] = idx_ii_15;
                    int local_idx_15 = idx_ii_15 - local_experts_start_idx;
                    int extent_15 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_15 = (1 << local_experts_stride_log2) - 1;
                    int is_local_15 = ((local_idx_15 >= 0 && local_idx_15 < extent_15 && (local_idx_15 & stride_mask_15) == 0) ? 1 : 0);
                    int is_local_ii_15 = is_local_15;
                    exp_off[11] = 0;
                    if (is_local_ii_15 != 0) {
                        uint32_t _shared_atomic_old_23;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_23) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_15)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[11] = (int)_shared_atomic_old_23;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_16 << 16);
                }
            }
        }
    }
    if (done == 0) {
        if (expanded_size >= 16 * grid_threads) {
            expanded = grid_tid + 12 * grid_threads;
            int idx_ii_16 = 0;
            unsigned int word0_17 = 0;
            unsigned int word1_17 = 0;
            unsigned int remote_17 = 0;
            int src_rank_17 = 0;
            int src_slot_16 = 0;
            int packed_word_16 = 0;
            packed_word_16 = topk_packed[expanded];
            idx_ii_16 = packed_word_16 >> 16;
            word0_17 = (unsigned int)packed_word_16 & 65535;
            exp_idx[12] = idx_ii_16;
            int local_idx_16 = idx_ii_16 - local_experts_start_idx;
            int extent_16 = num_local_experts << local_experts_stride_log2;
            int stride_mask_16 = (1 << local_experts_stride_log2) - 1;
            int is_local_16 = ((local_idx_16 >= 0 && local_idx_16 < extent_16 && (local_idx_16 & stride_mask_16) == 0) ? 1 : 0);
            int is_local_ii_16 = is_local_16;
            exp_off[12] = 0;
            if (is_local_ii_16 != 0) {
                uint32_t _shared_atomic_old_24;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_24) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_16)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[12] = (int)_shared_atomic_old_24;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_17 << 16);
            expanded = grid_tid + 13 * grid_threads;
            int idx_ii_0_3 = 0;
            unsigned int word0_1_3 = 0;
            unsigned int word1_2_3 = 0;
            unsigned int remote_3_3 = 0;
            int src_rank_4_3 = 0;
            int src_slot_5_3 = 0;
            int packed_word_6_3 = 0;
            packed_word_6_3 = topk_packed[expanded];
            idx_ii_0_3 = packed_word_6_3 >> 16;
            word0_1_3 = (unsigned int)packed_word_6_3 & 65535;
            exp_idx[13] = idx_ii_0_3;
            int local_idx_7_3 = idx_ii_0_3 - local_experts_start_idx;
            int extent_8_3 = num_local_experts << local_experts_stride_log2;
            int stride_mask_9_3 = (1 << local_experts_stride_log2) - 1;
            int is_local_10_3 = ((local_idx_7_3 >= 0 && local_idx_7_3 < extent_8_3 && (local_idx_7_3 & stride_mask_9_3) == 0) ? 1 : 0);
            int is_local_ii_11_3 = is_local_10_3;
            exp_off[13] = 0;
            if (is_local_ii_11_3 != 0) {
                uint32_t _shared_atomic_old_25;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_25) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_0_3)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[13] = (int)_shared_atomic_old_25;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_1_3 << 16);
            expanded = grid_tid + 14 * grid_threads;
            int idx_ii_12_3 = 0;
            unsigned int word0_13_3 = 0;
            unsigned int word1_14_3 = 0;
            unsigned int remote_15_3 = 0;
            int src_rank_16_3 = 0;
            int src_slot_17_3 = 0;
            int packed_word_18_3 = 0;
            packed_word_18_3 = topk_packed[expanded];
            idx_ii_12_3 = packed_word_18_3 >> 16;
            word0_13_3 = (unsigned int)packed_word_18_3 & 65535;
            exp_idx[14] = idx_ii_12_3;
            int local_idx_19_3 = idx_ii_12_3 - local_experts_start_idx;
            int extent_20_3 = num_local_experts << local_experts_stride_log2;
            int stride_mask_21_3 = (1 << local_experts_stride_log2) - 1;
            int is_local_22_3 = ((local_idx_19_3 >= 0 && local_idx_19_3 < extent_20_3 && (local_idx_19_3 & stride_mask_21_3) == 0) ? 1 : 0);
            int is_local_ii_23_3 = is_local_22_3;
            exp_off[14] = 0;
            if (is_local_ii_23_3 != 0) {
                uint32_t _shared_atomic_old_26;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_26) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_12_3)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[14] = (int)_shared_atomic_old_26;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_13_3 << 16);
            expanded = grid_tid + 15 * grid_threads;
            int idx_ii_24_3 = 0;
            unsigned int word0_25_3 = 0;
            unsigned int word1_26_3 = 0;
            unsigned int remote_27_3 = 0;
            int src_rank_28_3 = 0;
            int src_slot_29_3 = 0;
            int packed_word_30_3 = 0;
            packed_word_30_3 = topk_packed[expanded];
            idx_ii_24_3 = packed_word_30_3 >> 16;
            word0_25_3 = (unsigned int)packed_word_30_3 & 65535;
            exp_idx[15] = idx_ii_24_3;
            int local_idx_31_3 = idx_ii_24_3 - local_experts_start_idx;
            int extent_32_3 = num_local_experts << local_experts_stride_log2;
            int stride_mask_33_3 = (1 << local_experts_stride_log2) - 1;
            int is_local_34_3 = ((local_idx_31_3 >= 0 && local_idx_31_3 < extent_32_3 && (local_idx_31_3 & stride_mask_33_3) == 0) ? 1 : 0);
            int is_local_ii_35_3 = is_local_34_3;
            exp_off[15] = 0;
            if (is_local_ii_35_3 != 0) {
                uint32_t _shared_atomic_old_27;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_27) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_24_3)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[15] = (int)_shared_atomic_old_27;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_25_3 << 16);
        } else {
            if (done == 0) {
                expanded = grid_tid + 12 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_17 = 0;
                    unsigned int word0_18 = 0;
                    unsigned int word1_18 = 0;
                    unsigned int remote_18 = 0;
                    int src_rank_18 = 0;
                    int src_slot_18 = 0;
                    int packed_word_17 = 0;
                    packed_word_17 = topk_packed[expanded];
                    idx_ii_17 = packed_word_17 >> 16;
                    word0_18 = (unsigned int)packed_word_17 & 65535;
                    exp_idx[12] = idx_ii_17;
                    int local_idx_17 = idx_ii_17 - local_experts_start_idx;
                    int extent_17 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_17 = (1 << local_experts_stride_log2) - 1;
                    int is_local_17 = ((local_idx_17 >= 0 && local_idx_17 < extent_17 && (local_idx_17 & stride_mask_17) == 0) ? 1 : 0);
                    int is_local_ii_17 = is_local_17;
                    exp_off[12] = 0;
                    if (is_local_ii_17 != 0) {
                        uint32_t _shared_atomic_old_28;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_28) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_17)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[12] = (int)_shared_atomic_old_28;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_18 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 13 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_18 = 0;
                    unsigned int word0_19 = 0;
                    unsigned int word1_19 = 0;
                    unsigned int remote_19 = 0;
                    int src_rank_19 = 0;
                    int src_slot_19 = 0;
                    int packed_word_19 = 0;
                    packed_word_19 = topk_packed[expanded];
                    idx_ii_18 = packed_word_19 >> 16;
                    word0_19 = (unsigned int)packed_word_19 & 65535;
                    exp_idx[13] = idx_ii_18;
                    int local_idx_18 = idx_ii_18 - local_experts_start_idx;
                    int extent_18 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_18 = (1 << local_experts_stride_log2) - 1;
                    int is_local_18 = ((local_idx_18 >= 0 && local_idx_18 < extent_18 && (local_idx_18 & stride_mask_18) == 0) ? 1 : 0);
                    int is_local_ii_18 = is_local_18;
                    exp_off[13] = 0;
                    if (is_local_ii_18 != 0) {
                        uint32_t _shared_atomic_old_29;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_29) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_18)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[13] = (int)_shared_atomic_old_29;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_19 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 14 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_19 = 0;
                    unsigned int word0_20 = 0;
                    unsigned int word1_20 = 0;
                    unsigned int remote_20 = 0;
                    int src_rank_20 = 0;
                    int src_slot_20 = 0;
                    int packed_word_20 = 0;
                    packed_word_20 = topk_packed[expanded];
                    idx_ii_19 = packed_word_20 >> 16;
                    word0_20 = (unsigned int)packed_word_20 & 65535;
                    exp_idx[14] = idx_ii_19;
                    int local_idx_20 = idx_ii_19 - local_experts_start_idx;
                    int extent_19 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_19 = (1 << local_experts_stride_log2) - 1;
                    int is_local_19 = ((local_idx_20 >= 0 && local_idx_20 < extent_19 && (local_idx_20 & stride_mask_19) == 0) ? 1 : 0);
                    int is_local_ii_19 = is_local_19;
                    exp_off[14] = 0;
                    if (is_local_ii_19 != 0) {
                        uint32_t _shared_atomic_old_30;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_30) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_19)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[14] = (int)_shared_atomic_old_30;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_20 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 15 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_20 = 0;
                    unsigned int word0_21 = 0;
                    unsigned int word1_21 = 0;
                    unsigned int remote_21 = 0;
                    int src_rank_21 = 0;
                    int src_slot_21 = 0;
                    int packed_word_21 = 0;
                    packed_word_21 = topk_packed[expanded];
                    idx_ii_20 = packed_word_21 >> 16;
                    word0_21 = (unsigned int)packed_word_21 & 65535;
                    exp_idx[15] = idx_ii_20;
                    int local_idx_21 = idx_ii_20 - local_experts_start_idx;
                    int extent_21 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_20 = (1 << local_experts_stride_log2) - 1;
                    int is_local_20 = ((local_idx_21 >= 0 && local_idx_21 < extent_21 && (local_idx_21 & stride_mask_20) == 0) ? 1 : 0);
                    int is_local_ii_20 = is_local_20;
                    exp_off[15] = 0;
                    if (is_local_ii_20 != 0) {
                        uint32_t _shared_atomic_old_31;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_31) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_20)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[15] = (int)_shared_atomic_old_31;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_21 << 16);
                }
            }
        }
    }
    if (done == 0) {
        if (expanded_size >= 20 * grid_threads) {
            expanded = grid_tid + 16 * grid_threads;
            int idx_ii_21 = 0;
            unsigned int word0_22 = 0;
            unsigned int word1_22 = 0;
            unsigned int remote_22 = 0;
            int src_rank_22 = 0;
            int src_slot_22 = 0;
            int packed_word_22 = 0;
            packed_word_22 = topk_packed[expanded];
            idx_ii_21 = packed_word_22 >> 16;
            word0_22 = (unsigned int)packed_word_22 & 65535;
            exp_idx[16] = idx_ii_21;
            int local_idx_22 = idx_ii_21 - local_experts_start_idx;
            int extent_22 = num_local_experts << local_experts_stride_log2;
            int stride_mask_22 = (1 << local_experts_stride_log2) - 1;
            int is_local_21 = ((local_idx_22 >= 0 && local_idx_22 < extent_22 && (local_idx_22 & stride_mask_22) == 0) ? 1 : 0);
            int is_local_ii_21 = is_local_21;
            exp_off[16] = 0;
            if (is_local_ii_21 != 0) {
                uint32_t _shared_atomic_old_32;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_32) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_21)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[16] = (int)_shared_atomic_old_32;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_22 << 16);
            expanded = grid_tid + 17 * grid_threads;
            int idx_ii_0_4 = 0;
            unsigned int word0_1_4 = 0;
            unsigned int word1_2_4 = 0;
            unsigned int remote_3_4 = 0;
            int src_rank_4_4 = 0;
            int src_slot_5_4 = 0;
            int packed_word_6_4 = 0;
            packed_word_6_4 = topk_packed[expanded];
            idx_ii_0_4 = packed_word_6_4 >> 16;
            word0_1_4 = (unsigned int)packed_word_6_4 & 65535;
            exp_idx[17] = idx_ii_0_4;
            int local_idx_7_4 = idx_ii_0_4 - local_experts_start_idx;
            int extent_8_4 = num_local_experts << local_experts_stride_log2;
            int stride_mask_9_4 = (1 << local_experts_stride_log2) - 1;
            int is_local_10_4 = ((local_idx_7_4 >= 0 && local_idx_7_4 < extent_8_4 && (local_idx_7_4 & stride_mask_9_4) == 0) ? 1 : 0);
            int is_local_ii_11_4 = is_local_10_4;
            exp_off[17] = 0;
            if (is_local_ii_11_4 != 0) {
                uint32_t _shared_atomic_old_33;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_33) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_0_4)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[17] = (int)_shared_atomic_old_33;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_1_4 << 16);
            expanded = grid_tid + 18 * grid_threads;
            int idx_ii_12_4 = 0;
            unsigned int word0_13_4 = 0;
            unsigned int word1_14_4 = 0;
            unsigned int remote_15_4 = 0;
            int src_rank_16_4 = 0;
            int src_slot_17_4 = 0;
            int packed_word_18_4 = 0;
            packed_word_18_4 = topk_packed[expanded];
            idx_ii_12_4 = packed_word_18_4 >> 16;
            word0_13_4 = (unsigned int)packed_word_18_4 & 65535;
            exp_idx[18] = idx_ii_12_4;
            int local_idx_19_4 = idx_ii_12_4 - local_experts_start_idx;
            int extent_20_4 = num_local_experts << local_experts_stride_log2;
            int stride_mask_21_4 = (1 << local_experts_stride_log2) - 1;
            int is_local_22_4 = ((local_idx_19_4 >= 0 && local_idx_19_4 < extent_20_4 && (local_idx_19_4 & stride_mask_21_4) == 0) ? 1 : 0);
            int is_local_ii_23_4 = is_local_22_4;
            exp_off[18] = 0;
            if (is_local_ii_23_4 != 0) {
                uint32_t _shared_atomic_old_34;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_34) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_12_4)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[18] = (int)_shared_atomic_old_34;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_13_4 << 16);
            expanded = grid_tid + 19 * grid_threads;
            int idx_ii_24_4 = 0;
            unsigned int word0_25_4 = 0;
            unsigned int word1_26_4 = 0;
            unsigned int remote_27_4 = 0;
            int src_rank_28_4 = 0;
            int src_slot_29_4 = 0;
            int packed_word_30_4 = 0;
            packed_word_30_4 = topk_packed[expanded];
            idx_ii_24_4 = packed_word_30_4 >> 16;
            word0_25_4 = (unsigned int)packed_word_30_4 & 65535;
            exp_idx[19] = idx_ii_24_4;
            int local_idx_31_4 = idx_ii_24_4 - local_experts_start_idx;
            int extent_32_4 = num_local_experts << local_experts_stride_log2;
            int stride_mask_33_4 = (1 << local_experts_stride_log2) - 1;
            int is_local_34_4 = ((local_idx_31_4 >= 0 && local_idx_31_4 < extent_32_4 && (local_idx_31_4 & stride_mask_33_4) == 0) ? 1 : 0);
            int is_local_ii_35_4 = is_local_34_4;
            exp_off[19] = 0;
            if (is_local_ii_35_4 != 0) {
                uint32_t _shared_atomic_old_35;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_35) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_24_4)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[19] = (int)_shared_atomic_old_35;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_25_4 << 16);
        } else {
            if (done == 0) {
                expanded = grid_tid + 16 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_22 = 0;
                    unsigned int word0_23 = 0;
                    unsigned int word1_23 = 0;
                    unsigned int remote_23 = 0;
                    int src_rank_23 = 0;
                    int src_slot_23 = 0;
                    int packed_word_23 = 0;
                    packed_word_23 = topk_packed[expanded];
                    idx_ii_22 = packed_word_23 >> 16;
                    word0_23 = (unsigned int)packed_word_23 & 65535;
                    exp_idx[16] = idx_ii_22;
                    int local_idx_23 = idx_ii_22 - local_experts_start_idx;
                    int extent_23 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_23 = (1 << local_experts_stride_log2) - 1;
                    int is_local_23 = ((local_idx_23 >= 0 && local_idx_23 < extent_23 && (local_idx_23 & stride_mask_23) == 0) ? 1 : 0);
                    int is_local_ii_22 = is_local_23;
                    exp_off[16] = 0;
                    if (is_local_ii_22 != 0) {
                        uint32_t _shared_atomic_old_36;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_36) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_22)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[16] = (int)_shared_atomic_old_36;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_23 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 17 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_23 = 0;
                    unsigned int word0_24 = 0;
                    unsigned int word1_24 = 0;
                    unsigned int remote_24 = 0;
                    int src_rank_24 = 0;
                    int src_slot_24 = 0;
                    int packed_word_24 = 0;
                    packed_word_24 = topk_packed[expanded];
                    idx_ii_23 = packed_word_24 >> 16;
                    word0_24 = (unsigned int)packed_word_24 & 65535;
                    exp_idx[17] = idx_ii_23;
                    int local_idx_24 = idx_ii_23 - local_experts_start_idx;
                    int extent_24 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_24 = (1 << local_experts_stride_log2) - 1;
                    int is_local_24 = ((local_idx_24 >= 0 && local_idx_24 < extent_24 && (local_idx_24 & stride_mask_24) == 0) ? 1 : 0);
                    int is_local_ii_24 = is_local_24;
                    exp_off[17] = 0;
                    if (is_local_ii_24 != 0) {
                        uint32_t _shared_atomic_old_37;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_37) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_23)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[17] = (int)_shared_atomic_old_37;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_24 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 18 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_25 = 0;
                    unsigned int word0_26 = 0;
                    unsigned int word1_25 = 0;
                    unsigned int remote_25 = 0;
                    int src_rank_25 = 0;
                    int src_slot_25 = 0;
                    int packed_word_25 = 0;
                    packed_word_25 = topk_packed[expanded];
                    idx_ii_25 = packed_word_25 >> 16;
                    word0_26 = (unsigned int)packed_word_25 & 65535;
                    exp_idx[18] = idx_ii_25;
                    int local_idx_25 = idx_ii_25 - local_experts_start_idx;
                    int extent_25 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_25 = (1 << local_experts_stride_log2) - 1;
                    int is_local_25 = ((local_idx_25 >= 0 && local_idx_25 < extent_25 && (local_idx_25 & stride_mask_25) == 0) ? 1 : 0);
                    int is_local_ii_25 = is_local_25;
                    exp_off[18] = 0;
                    if (is_local_ii_25 != 0) {
                        uint32_t _shared_atomic_old_38;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_38) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_25)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[18] = (int)_shared_atomic_old_38;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_26 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 19 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_26 = 0;
                    unsigned int word0_27 = 0;
                    unsigned int word1_27 = 0;
                    unsigned int remote_26 = 0;
                    int src_rank_26 = 0;
                    int src_slot_26 = 0;
                    int packed_word_26 = 0;
                    packed_word_26 = topk_packed[expanded];
                    idx_ii_26 = packed_word_26 >> 16;
                    word0_27 = (unsigned int)packed_word_26 & 65535;
                    exp_idx[19] = idx_ii_26;
                    int local_idx_26 = idx_ii_26 - local_experts_start_idx;
                    int extent_26 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_26 = (1 << local_experts_stride_log2) - 1;
                    int is_local_26 = ((local_idx_26 >= 0 && local_idx_26 < extent_26 && (local_idx_26 & stride_mask_26) == 0) ? 1 : 0);
                    int is_local_ii_26 = is_local_26;
                    exp_off[19] = 0;
                    if (is_local_ii_26 != 0) {
                        uint32_t _shared_atomic_old_39;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_39) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_26)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[19] = (int)_shared_atomic_old_39;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_27 << 16);
                }
            }
        }
    }
    if (done == 0) {
        if (expanded_size >= 24 * grid_threads) {
            expanded = grid_tid + 20 * grid_threads;
            int idx_ii_27 = 0;
            unsigned int word0_28 = 0;
            unsigned int word1_28 = 0;
            unsigned int remote_28 = 0;
            int src_rank_27 = 0;
            int src_slot_27 = 0;
            int packed_word_27 = 0;
            packed_word_27 = topk_packed[expanded];
            idx_ii_27 = packed_word_27 >> 16;
            word0_28 = (unsigned int)packed_word_27 & 65535;
            exp_idx[20] = idx_ii_27;
            int local_idx_27 = idx_ii_27 - local_experts_start_idx;
            int extent_27 = num_local_experts << local_experts_stride_log2;
            int stride_mask_27 = (1 << local_experts_stride_log2) - 1;
            int is_local_27 = ((local_idx_27 >= 0 && local_idx_27 < extent_27 && (local_idx_27 & stride_mask_27) == 0) ? 1 : 0);
            int is_local_ii_27 = is_local_27;
            exp_off[20] = 0;
            if (is_local_ii_27 != 0) {
                uint32_t _shared_atomic_old_40;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_40) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_27)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[20] = (int)_shared_atomic_old_40;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_28 << 16);
            expanded = grid_tid + 21 * grid_threads;
            int idx_ii_0_5 = 0;
            unsigned int word0_1_5 = 0;
            unsigned int word1_2_5 = 0;
            unsigned int remote_3_5 = 0;
            int src_rank_4_5 = 0;
            int src_slot_5_5 = 0;
            int packed_word_6_5 = 0;
            packed_word_6_5 = topk_packed[expanded];
            idx_ii_0_5 = packed_word_6_5 >> 16;
            word0_1_5 = (unsigned int)packed_word_6_5 & 65535;
            exp_idx[21] = idx_ii_0_5;
            int local_idx_7_5 = idx_ii_0_5 - local_experts_start_idx;
            int extent_8_5 = num_local_experts << local_experts_stride_log2;
            int stride_mask_9_5 = (1 << local_experts_stride_log2) - 1;
            int is_local_10_5 = ((local_idx_7_5 >= 0 && local_idx_7_5 < extent_8_5 && (local_idx_7_5 & stride_mask_9_5) == 0) ? 1 : 0);
            int is_local_ii_11_5 = is_local_10_5;
            exp_off[21] = 0;
            if (is_local_ii_11_5 != 0) {
                uint32_t _shared_atomic_old_41;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_41) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_0_5)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[21] = (int)_shared_atomic_old_41;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_1_5 << 16);
            expanded = grid_tid + 22 * grid_threads;
            int idx_ii_12_5 = 0;
            unsigned int word0_13_5 = 0;
            unsigned int word1_14_5 = 0;
            unsigned int remote_15_5 = 0;
            int src_rank_16_5 = 0;
            int src_slot_17_5 = 0;
            int packed_word_18_5 = 0;
            packed_word_18_5 = topk_packed[expanded];
            idx_ii_12_5 = packed_word_18_5 >> 16;
            word0_13_5 = (unsigned int)packed_word_18_5 & 65535;
            exp_idx[22] = idx_ii_12_5;
            int local_idx_19_5 = idx_ii_12_5 - local_experts_start_idx;
            int extent_20_5 = num_local_experts << local_experts_stride_log2;
            int stride_mask_21_5 = (1 << local_experts_stride_log2) - 1;
            int is_local_22_5 = ((local_idx_19_5 >= 0 && local_idx_19_5 < extent_20_5 && (local_idx_19_5 & stride_mask_21_5) == 0) ? 1 : 0);
            int is_local_ii_23_5 = is_local_22_5;
            exp_off[22] = 0;
            if (is_local_ii_23_5 != 0) {
                uint32_t _shared_atomic_old_42;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_42) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_12_5)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[22] = (int)_shared_atomic_old_42;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_13_5 << 16);
            expanded = grid_tid + 23 * grid_threads;
            int idx_ii_24_5 = 0;
            unsigned int word0_25_5 = 0;
            unsigned int word1_26_5 = 0;
            unsigned int remote_27_5 = 0;
            int src_rank_28_5 = 0;
            int src_slot_29_5 = 0;
            int packed_word_30_5 = 0;
            packed_word_30_5 = topk_packed[expanded];
            idx_ii_24_5 = packed_word_30_5 >> 16;
            word0_25_5 = (unsigned int)packed_word_30_5 & 65535;
            exp_idx[23] = idx_ii_24_5;
            int local_idx_31_5 = idx_ii_24_5 - local_experts_start_idx;
            int extent_32_5 = num_local_experts << local_experts_stride_log2;
            int stride_mask_33_5 = (1 << local_experts_stride_log2) - 1;
            int is_local_34_5 = ((local_idx_31_5 >= 0 && local_idx_31_5 < extent_32_5 && (local_idx_31_5 & stride_mask_33_5) == 0) ? 1 : 0);
            int is_local_ii_35_5 = is_local_34_5;
            exp_off[23] = 0;
            if (is_local_ii_35_5 != 0) {
                uint32_t _shared_atomic_old_43;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_43) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_24_5)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[23] = (int)_shared_atomic_old_43;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_25_5 << 16);
        } else {
            if (done == 0) {
                expanded = grid_tid + 20 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_28 = 0;
                    unsigned int word0_29 = 0;
                    unsigned int word1_29 = 0;
                    unsigned int remote_29 = 0;
                    int src_rank_29 = 0;
                    int src_slot_28 = 0;
                    int packed_word_28 = 0;
                    packed_word_28 = topk_packed[expanded];
                    idx_ii_28 = packed_word_28 >> 16;
                    word0_29 = (unsigned int)packed_word_28 & 65535;
                    exp_idx[20] = idx_ii_28;
                    int local_idx_28 = idx_ii_28 - local_experts_start_idx;
                    int extent_28 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_28 = (1 << local_experts_stride_log2) - 1;
                    int is_local_28 = ((local_idx_28 >= 0 && local_idx_28 < extent_28 && (local_idx_28 & stride_mask_28) == 0) ? 1 : 0);
                    int is_local_ii_28 = is_local_28;
                    exp_off[20] = 0;
                    if (is_local_ii_28 != 0) {
                        uint32_t _shared_atomic_old_44;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_44) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_28)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[20] = (int)_shared_atomic_old_44;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_29 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 21 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_29 = 0;
                    unsigned int word0_30 = 0;
                    unsigned int word1_30 = 0;
                    unsigned int remote_30 = 0;
                    int src_rank_30 = 0;
                    int src_slot_30 = 0;
                    int packed_word_29 = 0;
                    packed_word_29 = topk_packed[expanded];
                    idx_ii_29 = packed_word_29 >> 16;
                    word0_30 = (unsigned int)packed_word_29 & 65535;
                    exp_idx[21] = idx_ii_29;
                    int local_idx_29 = idx_ii_29 - local_experts_start_idx;
                    int extent_29 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_29 = (1 << local_experts_stride_log2) - 1;
                    int is_local_29 = ((local_idx_29 >= 0 && local_idx_29 < extent_29 && (local_idx_29 & stride_mask_29) == 0) ? 1 : 0);
                    int is_local_ii_29 = is_local_29;
                    exp_off[21] = 0;
                    if (is_local_ii_29 != 0) {
                        uint32_t _shared_atomic_old_45;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_45) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_29)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[21] = (int)_shared_atomic_old_45;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_30 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 22 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_30 = 0;
                    unsigned int word0_31 = 0;
                    unsigned int word1_31 = 0;
                    unsigned int remote_31 = 0;
                    int src_rank_31 = 0;
                    int src_slot_31 = 0;
                    int packed_word_31 = 0;
                    packed_word_31 = topk_packed[expanded];
                    idx_ii_30 = packed_word_31 >> 16;
                    word0_31 = (unsigned int)packed_word_31 & 65535;
                    exp_idx[22] = idx_ii_30;
                    int local_idx_30 = idx_ii_30 - local_experts_start_idx;
                    int extent_30 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_30 = (1 << local_experts_stride_log2) - 1;
                    int is_local_30 = ((local_idx_30 >= 0 && local_idx_30 < extent_30 && (local_idx_30 & stride_mask_30) == 0) ? 1 : 0);
                    int is_local_ii_30 = is_local_30;
                    exp_off[22] = 0;
                    if (is_local_ii_30 != 0) {
                        uint32_t _shared_atomic_old_46;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_46) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_30)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[22] = (int)_shared_atomic_old_46;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_31 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 23 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_31 = 0;
                    unsigned int word0_32 = 0;
                    unsigned int word1_32 = 0;
                    unsigned int remote_32 = 0;
                    int src_rank_32 = 0;
                    int src_slot_32 = 0;
                    int packed_word_32 = 0;
                    packed_word_32 = topk_packed[expanded];
                    idx_ii_31 = packed_word_32 >> 16;
                    word0_32 = (unsigned int)packed_word_32 & 65535;
                    exp_idx[23] = idx_ii_31;
                    int local_idx_32 = idx_ii_31 - local_experts_start_idx;
                    int extent_31 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_31 = (1 << local_experts_stride_log2) - 1;
                    int is_local_31 = ((local_idx_32 >= 0 && local_idx_32 < extent_31 && (local_idx_32 & stride_mask_31) == 0) ? 1 : 0);
                    int is_local_ii_31 = is_local_31;
                    exp_off[23] = 0;
                    if (is_local_ii_31 != 0) {
                        uint32_t _shared_atomic_old_47;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_47) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_31)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[23] = (int)_shared_atomic_old_47;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_32 << 16);
                }
            }
        }
    }
    if (done == 0) {
        if (expanded_size >= 28 * grid_threads) {
            expanded = grid_tid + 24 * grid_threads;
            int idx_ii_32 = 0;
            unsigned int word0_33 = 0;
            unsigned int word1_33 = 0;
            unsigned int remote_33 = 0;
            int src_rank_33 = 0;
            int src_slot_33 = 0;
            int packed_word_33 = 0;
            packed_word_33 = topk_packed[expanded];
            idx_ii_32 = packed_word_33 >> 16;
            word0_33 = (unsigned int)packed_word_33 & 65535;
            exp_idx[24] = idx_ii_32;
            int local_idx_33 = idx_ii_32 - local_experts_start_idx;
            int extent_33 = num_local_experts << local_experts_stride_log2;
            int stride_mask_32 = (1 << local_experts_stride_log2) - 1;
            int is_local_32 = ((local_idx_33 >= 0 && local_idx_33 < extent_33 && (local_idx_33 & stride_mask_32) == 0) ? 1 : 0);
            int is_local_ii_32 = is_local_32;
            exp_off[24] = 0;
            if (is_local_ii_32 != 0) {
                uint32_t _shared_atomic_old_48;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_48) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_32)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[24] = (int)_shared_atomic_old_48;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_33 << 16);
            expanded = grid_tid + 25 * grid_threads;
            int idx_ii_0_6 = 0;
            unsigned int word0_1_6 = 0;
            unsigned int word1_2_6 = 0;
            unsigned int remote_3_6 = 0;
            int src_rank_4_6 = 0;
            int src_slot_5_6 = 0;
            int packed_word_6_6 = 0;
            packed_word_6_6 = topk_packed[expanded];
            idx_ii_0_6 = packed_word_6_6 >> 16;
            word0_1_6 = (unsigned int)packed_word_6_6 & 65535;
            exp_idx[25] = idx_ii_0_6;
            int local_idx_7_6 = idx_ii_0_6 - local_experts_start_idx;
            int extent_8_6 = num_local_experts << local_experts_stride_log2;
            int stride_mask_9_6 = (1 << local_experts_stride_log2) - 1;
            int is_local_10_6 = ((local_idx_7_6 >= 0 && local_idx_7_6 < extent_8_6 && (local_idx_7_6 & stride_mask_9_6) == 0) ? 1 : 0);
            int is_local_ii_11_6 = is_local_10_6;
            exp_off[25] = 0;
            if (is_local_ii_11_6 != 0) {
                uint32_t _shared_atomic_old_49;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_49) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_0_6)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[25] = (int)_shared_atomic_old_49;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_1_6 << 16);
            expanded = grid_tid + 26 * grid_threads;
            int idx_ii_12_6 = 0;
            unsigned int word0_13_6 = 0;
            unsigned int word1_14_6 = 0;
            unsigned int remote_15_6 = 0;
            int src_rank_16_6 = 0;
            int src_slot_17_6 = 0;
            int packed_word_18_6 = 0;
            packed_word_18_6 = topk_packed[expanded];
            idx_ii_12_6 = packed_word_18_6 >> 16;
            word0_13_6 = (unsigned int)packed_word_18_6 & 65535;
            exp_idx[26] = idx_ii_12_6;
            int local_idx_19_6 = idx_ii_12_6 - local_experts_start_idx;
            int extent_20_6 = num_local_experts << local_experts_stride_log2;
            int stride_mask_21_6 = (1 << local_experts_stride_log2) - 1;
            int is_local_22_6 = ((local_idx_19_6 >= 0 && local_idx_19_6 < extent_20_6 && (local_idx_19_6 & stride_mask_21_6) == 0) ? 1 : 0);
            int is_local_ii_23_6 = is_local_22_6;
            exp_off[26] = 0;
            if (is_local_ii_23_6 != 0) {
                uint32_t _shared_atomic_old_50;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_50) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_12_6)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[26] = (int)_shared_atomic_old_50;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_13_6 << 16);
            expanded = grid_tid + 27 * grid_threads;
            int idx_ii_24_6 = 0;
            unsigned int word0_25_6 = 0;
            unsigned int word1_26_6 = 0;
            unsigned int remote_27_6 = 0;
            int src_rank_28_6 = 0;
            int src_slot_29_6 = 0;
            int packed_word_30_6 = 0;
            packed_word_30_6 = topk_packed[expanded];
            idx_ii_24_6 = packed_word_30_6 >> 16;
            word0_25_6 = (unsigned int)packed_word_30_6 & 65535;
            exp_idx[27] = idx_ii_24_6;
            int local_idx_31_6 = idx_ii_24_6 - local_experts_start_idx;
            int extent_32_6 = num_local_experts << local_experts_stride_log2;
            int stride_mask_33_6 = (1 << local_experts_stride_log2) - 1;
            int is_local_34_6 = ((local_idx_31_6 >= 0 && local_idx_31_6 < extent_32_6 && (local_idx_31_6 & stride_mask_33_6) == 0) ? 1 : 0);
            int is_local_ii_35_6 = is_local_34_6;
            exp_off[27] = 0;
            if (is_local_ii_35_6 != 0) {
                uint32_t _shared_atomic_old_51;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_51) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_24_6)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[27] = (int)_shared_atomic_old_51;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_25_6 << 16);
        } else {
            if (done == 0) {
                expanded = grid_tid + 24 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_33 = 0;
                    unsigned int word0_34 = 0;
                    unsigned int word1_34 = 0;
                    unsigned int remote_34 = 0;
                    int src_rank_34 = 0;
                    int src_slot_34 = 0;
                    int packed_word_34 = 0;
                    packed_word_34 = topk_packed[expanded];
                    idx_ii_33 = packed_word_34 >> 16;
                    word0_34 = (unsigned int)packed_word_34 & 65535;
                    exp_idx[24] = idx_ii_33;
                    int local_idx_34 = idx_ii_33 - local_experts_start_idx;
                    int extent_34 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_34 = (1 << local_experts_stride_log2) - 1;
                    int is_local_33 = ((local_idx_34 >= 0 && local_idx_34 < extent_34 && (local_idx_34 & stride_mask_34) == 0) ? 1 : 0);
                    int is_local_ii_33 = is_local_33;
                    exp_off[24] = 0;
                    if (is_local_ii_33 != 0) {
                        uint32_t _shared_atomic_old_52;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_52) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_33)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[24] = (int)_shared_atomic_old_52;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_34 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 25 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_34 = 0;
                    unsigned int word0_35 = 0;
                    unsigned int word1_35 = 0;
                    unsigned int remote_35 = 0;
                    int src_rank_35 = 0;
                    int src_slot_35 = 0;
                    int packed_word_35 = 0;
                    packed_word_35 = topk_packed[expanded];
                    idx_ii_34 = packed_word_35 >> 16;
                    word0_35 = (unsigned int)packed_word_35 & 65535;
                    exp_idx[25] = idx_ii_34;
                    int local_idx_35 = idx_ii_34 - local_experts_start_idx;
                    int extent_35 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_35 = (1 << local_experts_stride_log2) - 1;
                    int is_local_35 = ((local_idx_35 >= 0 && local_idx_35 < extent_35 && (local_idx_35 & stride_mask_35) == 0) ? 1 : 0);
                    int is_local_ii_34 = is_local_35;
                    exp_off[25] = 0;
                    if (is_local_ii_34 != 0) {
                        uint32_t _shared_atomic_old_53;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_53) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_34)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[25] = (int)_shared_atomic_old_53;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_35 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 26 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_35 = 0;
                    unsigned int word0_36 = 0;
                    unsigned int word1_36 = 0;
                    unsigned int remote_36 = 0;
                    int src_rank_36 = 0;
                    int src_slot_36 = 0;
                    int packed_word_36 = 0;
                    packed_word_36 = topk_packed[expanded];
                    idx_ii_35 = packed_word_36 >> 16;
                    word0_36 = (unsigned int)packed_word_36 & 65535;
                    exp_idx[26] = idx_ii_35;
                    int local_idx_36 = idx_ii_35 - local_experts_start_idx;
                    int extent_36 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_36 = (1 << local_experts_stride_log2) - 1;
                    int is_local_36 = ((local_idx_36 >= 0 && local_idx_36 < extent_36 && (local_idx_36 & stride_mask_36) == 0) ? 1 : 0);
                    int is_local_ii_36 = is_local_36;
                    exp_off[26] = 0;
                    if (is_local_ii_36 != 0) {
                        uint32_t _shared_atomic_old_54;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_54) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_35)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[26] = (int)_shared_atomic_old_54;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_36 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 27 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_36 = 0;
                    unsigned int word0_37 = 0;
                    unsigned int word1_37 = 0;
                    unsigned int remote_37 = 0;
                    int src_rank_37 = 0;
                    int src_slot_37 = 0;
                    int packed_word_37 = 0;
                    packed_word_37 = topk_packed[expanded];
                    idx_ii_36 = packed_word_37 >> 16;
                    word0_37 = (unsigned int)packed_word_37 & 65535;
                    exp_idx[27] = idx_ii_36;
                    int local_idx_37 = idx_ii_36 - local_experts_start_idx;
                    int extent_37 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_37 = (1 << local_experts_stride_log2) - 1;
                    int is_local_37 = ((local_idx_37 >= 0 && local_idx_37 < extent_37 && (local_idx_37 & stride_mask_37) == 0) ? 1 : 0);
                    int is_local_ii_37 = is_local_37;
                    exp_off[27] = 0;
                    if (is_local_ii_37 != 0) {
                        uint32_t _shared_atomic_old_55;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_55) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_36)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[27] = (int)_shared_atomic_old_55;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_37 << 16);
                }
            }
        }
    }
    if (done == 0) {
        if (expanded_size >= 32 * grid_threads) {
            expanded = grid_tid + 28 * grid_threads;
            int idx_ii_37 = 0;
            unsigned int word0_38 = 0;
            unsigned int word1_38 = 0;
            unsigned int remote_38 = 0;
            int src_rank_38 = 0;
            int src_slot_38 = 0;
            int packed_word_38 = 0;
            packed_word_38 = topk_packed[expanded];
            idx_ii_37 = packed_word_38 >> 16;
            word0_38 = (unsigned int)packed_word_38 & 65535;
            exp_idx[28] = idx_ii_37;
            int local_idx_38 = idx_ii_37 - local_experts_start_idx;
            int extent_38 = num_local_experts << local_experts_stride_log2;
            int stride_mask_38 = (1 << local_experts_stride_log2) - 1;
            int is_local_38 = ((local_idx_38 >= 0 && local_idx_38 < extent_38 && (local_idx_38 & stride_mask_38) == 0) ? 1 : 0);
            int is_local_ii_38 = is_local_38;
            exp_off[28] = 0;
            if (is_local_ii_38 != 0) {
                uint32_t _shared_atomic_old_56;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_56) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_37)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[28] = (int)_shared_atomic_old_56;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_38 << 16);
            expanded = grid_tid + 29 * grid_threads;
            int idx_ii_0_7 = 0;
            unsigned int word0_1_7 = 0;
            unsigned int word1_2_7 = 0;
            unsigned int remote_3_7 = 0;
            int src_rank_4_7 = 0;
            int src_slot_5_7 = 0;
            int packed_word_6_7 = 0;
            packed_word_6_7 = topk_packed[expanded];
            idx_ii_0_7 = packed_word_6_7 >> 16;
            word0_1_7 = (unsigned int)packed_word_6_7 & 65535;
            exp_idx[29] = idx_ii_0_7;
            int local_idx_7_7 = idx_ii_0_7 - local_experts_start_idx;
            int extent_8_7 = num_local_experts << local_experts_stride_log2;
            int stride_mask_9_7 = (1 << local_experts_stride_log2) - 1;
            int is_local_10_7 = ((local_idx_7_7 >= 0 && local_idx_7_7 < extent_8_7 && (local_idx_7_7 & stride_mask_9_7) == 0) ? 1 : 0);
            int is_local_ii_11_7 = is_local_10_7;
            exp_off[29] = 0;
            if (is_local_ii_11_7 != 0) {
                uint32_t _shared_atomic_old_57;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_57) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_0_7)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[29] = (int)_shared_atomic_old_57;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_1_7 << 16);
            expanded = grid_tid + 30 * grid_threads;
            int idx_ii_12_7 = 0;
            unsigned int word0_13_7 = 0;
            unsigned int word1_14_7 = 0;
            unsigned int remote_15_7 = 0;
            int src_rank_16_7 = 0;
            int src_slot_17_7 = 0;
            int packed_word_18_7 = 0;
            packed_word_18_7 = topk_packed[expanded];
            idx_ii_12_7 = packed_word_18_7 >> 16;
            word0_13_7 = (unsigned int)packed_word_18_7 & 65535;
            exp_idx[30] = idx_ii_12_7;
            int local_idx_19_7 = idx_ii_12_7 - local_experts_start_idx;
            int extent_20_7 = num_local_experts << local_experts_stride_log2;
            int stride_mask_21_7 = (1 << local_experts_stride_log2) - 1;
            int is_local_22_7 = ((local_idx_19_7 >= 0 && local_idx_19_7 < extent_20_7 && (local_idx_19_7 & stride_mask_21_7) == 0) ? 1 : 0);
            int is_local_ii_23_7 = is_local_22_7;
            exp_off[30] = 0;
            if (is_local_ii_23_7 != 0) {
                uint32_t _shared_atomic_old_58;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_58) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_12_7)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[30] = (int)_shared_atomic_old_58;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_13_7 << 16);
            expanded = grid_tid + 31 * grid_threads;
            int idx_ii_24_7 = 0;
            unsigned int word0_25_7 = 0;
            unsigned int word1_26_7 = 0;
            unsigned int remote_27_7 = 0;
            int src_rank_28_7 = 0;
            int src_slot_29_7 = 0;
            int packed_word_30_7 = 0;
            packed_word_30_7 = topk_packed[expanded];
            idx_ii_24_7 = packed_word_30_7 >> 16;
            word0_25_7 = (unsigned int)packed_word_30_7 & 65535;
            exp_idx[31] = idx_ii_24_7;
            int local_idx_31_7 = idx_ii_24_7 - local_experts_start_idx;
            int extent_32_7 = num_local_experts << local_experts_stride_log2;
            int stride_mask_33_7 = (1 << local_experts_stride_log2) - 1;
            int is_local_34_7 = ((local_idx_31_7 >= 0 && local_idx_31_7 < extent_32_7 && (local_idx_31_7 & stride_mask_33_7) == 0) ? 1 : 0);
            int is_local_ii_35_7 = is_local_34_7;
            exp_off[31] = 0;
            if (is_local_ii_35_7 != 0) {
                uint32_t _shared_atomic_old_59;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_59) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_24_7)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[31] = (int)_shared_atomic_old_59;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_25_7 << 16);
        } else {
            if (done == 0) {
                expanded = grid_tid + 28 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_38 = 0;
                    unsigned int word0_39 = 0;
                    unsigned int word1_39 = 0;
                    unsigned int remote_39 = 0;
                    int src_rank_39 = 0;
                    int src_slot_39 = 0;
                    int packed_word_39 = 0;
                    packed_word_39 = topk_packed[expanded];
                    idx_ii_38 = packed_word_39 >> 16;
                    word0_39 = (unsigned int)packed_word_39 & 65535;
                    exp_idx[28] = idx_ii_38;
                    int local_idx_39 = idx_ii_38 - local_experts_start_idx;
                    int extent_39 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_39 = (1 << local_experts_stride_log2) - 1;
                    int is_local_39 = ((local_idx_39 >= 0 && local_idx_39 < extent_39 && (local_idx_39 & stride_mask_39) == 0) ? 1 : 0);
                    int is_local_ii_39 = is_local_39;
                    exp_off[28] = 0;
                    if (is_local_ii_39 != 0) {
                        uint32_t _shared_atomic_old_60;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_60) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_38)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[28] = (int)_shared_atomic_old_60;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_39 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 29 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_39 = 0;
                    unsigned int word0_40 = 0;
                    unsigned int word1_40 = 0;
                    unsigned int remote_40 = 0;
                    int src_rank_40 = 0;
                    int src_slot_40 = 0;
                    int packed_word_40 = 0;
                    packed_word_40 = topk_packed[expanded];
                    idx_ii_39 = packed_word_40 >> 16;
                    word0_40 = (unsigned int)packed_word_40 & 65535;
                    exp_idx[29] = idx_ii_39;
                    int local_idx_40 = idx_ii_39 - local_experts_start_idx;
                    int extent_40 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_40 = (1 << local_experts_stride_log2) - 1;
                    int is_local_40 = ((local_idx_40 >= 0 && local_idx_40 < extent_40 && (local_idx_40 & stride_mask_40) == 0) ? 1 : 0);
                    int is_local_ii_40 = is_local_40;
                    exp_off[29] = 0;
                    if (is_local_ii_40 != 0) {
                        uint32_t _shared_atomic_old_61;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_61) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_39)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[29] = (int)_shared_atomic_old_61;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_40 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 30 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_40 = 0;
                    unsigned int word0_41 = 0;
                    unsigned int word1_41 = 0;
                    unsigned int remote_41 = 0;
                    int src_rank_41 = 0;
                    int src_slot_41 = 0;
                    int packed_word_41 = 0;
                    packed_word_41 = topk_packed[expanded];
                    idx_ii_40 = packed_word_41 >> 16;
                    word0_41 = (unsigned int)packed_word_41 & 65535;
                    exp_idx[30] = idx_ii_40;
                    int local_idx_41 = idx_ii_40 - local_experts_start_idx;
                    int extent_41 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_41 = (1 << local_experts_stride_log2) - 1;
                    int is_local_41 = ((local_idx_41 >= 0 && local_idx_41 < extent_41 && (local_idx_41 & stride_mask_41) == 0) ? 1 : 0);
                    int is_local_ii_41 = is_local_41;
                    exp_off[30] = 0;
                    if (is_local_ii_41 != 0) {
                        uint32_t _shared_atomic_old_62;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_62) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_40)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[30] = (int)_shared_atomic_old_62;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_41 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 31 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_41 = 0;
                    unsigned int word0_42 = 0;
                    unsigned int word1_42 = 0;
                    unsigned int remote_42 = 0;
                    int src_rank_42 = 0;
                    int src_slot_42 = 0;
                    int packed_word_42 = 0;
                    packed_word_42 = topk_packed[expanded];
                    idx_ii_41 = packed_word_42 >> 16;
                    word0_42 = (unsigned int)packed_word_42 & 65535;
                    exp_idx[31] = idx_ii_41;
                    int local_idx_42 = idx_ii_41 - local_experts_start_idx;
                    int extent_42 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_42 = (1 << local_experts_stride_log2) - 1;
                    int is_local_42 = ((local_idx_42 >= 0 && local_idx_42 < extent_42 && (local_idx_42 & stride_mask_42) == 0) ? 1 : 0);
                    int is_local_ii_42 = is_local_42;
                    exp_off[31] = 0;
                    if (is_local_ii_42 != 0) {
                        uint32_t _shared_atomic_old_63;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_63) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_41)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[31] = (int)_shared_atomic_old_63;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_42 << 16);
                }
            }
        }
    }
    if (done == 0) {
        if (expanded_size >= 36 * grid_threads) {
            expanded = grid_tid + 32 * grid_threads;
            int idx_ii_42 = 0;
            unsigned int word0_43 = 0;
            unsigned int word1_43 = 0;
            unsigned int remote_43 = 0;
            int src_rank_43 = 0;
            int src_slot_43 = 0;
            int packed_word_43 = 0;
            packed_word_43 = topk_packed[expanded];
            idx_ii_42 = packed_word_43 >> 16;
            word0_43 = (unsigned int)packed_word_43 & 65535;
            exp_idx[32] = idx_ii_42;
            int local_idx_43 = idx_ii_42 - local_experts_start_idx;
            int extent_43 = num_local_experts << local_experts_stride_log2;
            int stride_mask_43 = (1 << local_experts_stride_log2) - 1;
            int is_local_43 = ((local_idx_43 >= 0 && local_idx_43 < extent_43 && (local_idx_43 & stride_mask_43) == 0) ? 1 : 0);
            int is_local_ii_43 = is_local_43;
            exp_off[32] = 0;
            if (is_local_ii_43 != 0) {
                uint32_t _shared_atomic_old_64;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_64) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_42)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[32] = (int)_shared_atomic_old_64;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_43 << 16);
            expanded = grid_tid + 33 * grid_threads;
            int idx_ii_0_8 = 0;
            unsigned int word0_1_8 = 0;
            unsigned int word1_2_8 = 0;
            unsigned int remote_3_8 = 0;
            int src_rank_4_8 = 0;
            int src_slot_5_8 = 0;
            int packed_word_6_8 = 0;
            packed_word_6_8 = topk_packed[expanded];
            idx_ii_0_8 = packed_word_6_8 >> 16;
            word0_1_8 = (unsigned int)packed_word_6_8 & 65535;
            exp_idx[33] = idx_ii_0_8;
            int local_idx_7_8 = idx_ii_0_8 - local_experts_start_idx;
            int extent_8_8 = num_local_experts << local_experts_stride_log2;
            int stride_mask_9_8 = (1 << local_experts_stride_log2) - 1;
            int is_local_10_8 = ((local_idx_7_8 >= 0 && local_idx_7_8 < extent_8_8 && (local_idx_7_8 & stride_mask_9_8) == 0) ? 1 : 0);
            int is_local_ii_11_8 = is_local_10_8;
            exp_off[33] = 0;
            if (is_local_ii_11_8 != 0) {
                uint32_t _shared_atomic_old_65;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_65) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_0_8)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[33] = (int)_shared_atomic_old_65;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_1_8 << 16);
            expanded = grid_tid + 34 * grid_threads;
            int idx_ii_12_8 = 0;
            unsigned int word0_13_8 = 0;
            unsigned int word1_14_8 = 0;
            unsigned int remote_15_8 = 0;
            int src_rank_16_8 = 0;
            int src_slot_17_8 = 0;
            int packed_word_18_8 = 0;
            packed_word_18_8 = topk_packed[expanded];
            idx_ii_12_8 = packed_word_18_8 >> 16;
            word0_13_8 = (unsigned int)packed_word_18_8 & 65535;
            exp_idx[34] = idx_ii_12_8;
            int local_idx_19_8 = idx_ii_12_8 - local_experts_start_idx;
            int extent_20_8 = num_local_experts << local_experts_stride_log2;
            int stride_mask_21_8 = (1 << local_experts_stride_log2) - 1;
            int is_local_22_8 = ((local_idx_19_8 >= 0 && local_idx_19_8 < extent_20_8 && (local_idx_19_8 & stride_mask_21_8) == 0) ? 1 : 0);
            int is_local_ii_23_8 = is_local_22_8;
            exp_off[34] = 0;
            if (is_local_ii_23_8 != 0) {
                uint32_t _shared_atomic_old_66;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_66) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_12_8)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[34] = (int)_shared_atomic_old_66;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_13_8 << 16);
            expanded = grid_tid + 35 * grid_threads;
            int idx_ii_24_8 = 0;
            unsigned int word0_25_8 = 0;
            unsigned int word1_26_8 = 0;
            unsigned int remote_27_8 = 0;
            int src_rank_28_8 = 0;
            int src_slot_29_8 = 0;
            int packed_word_30_8 = 0;
            packed_word_30_8 = topk_packed[expanded];
            idx_ii_24_8 = packed_word_30_8 >> 16;
            word0_25_8 = (unsigned int)packed_word_30_8 & 65535;
            exp_idx[35] = idx_ii_24_8;
            int local_idx_31_8 = idx_ii_24_8 - local_experts_start_idx;
            int extent_32_8 = num_local_experts << local_experts_stride_log2;
            int stride_mask_33_8 = (1 << local_experts_stride_log2) - 1;
            int is_local_34_8 = ((local_idx_31_8 >= 0 && local_idx_31_8 < extent_32_8 && (local_idx_31_8 & stride_mask_33_8) == 0) ? 1 : 0);
            int is_local_ii_35_8 = is_local_34_8;
            exp_off[35] = 0;
            if (is_local_ii_35_8 != 0) {
                uint32_t _shared_atomic_old_67;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_67) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_24_8)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[35] = (int)_shared_atomic_old_67;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_25_8 << 16);
        } else {
            if (done == 0) {
                expanded = grid_tid + 32 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_43 = 0;
                    unsigned int word0_44 = 0;
                    unsigned int word1_44 = 0;
                    unsigned int remote_44 = 0;
                    int src_rank_44 = 0;
                    int src_slot_44 = 0;
                    int packed_word_44 = 0;
                    packed_word_44 = topk_packed[expanded];
                    idx_ii_43 = packed_word_44 >> 16;
                    word0_44 = (unsigned int)packed_word_44 & 65535;
                    exp_idx[32] = idx_ii_43;
                    int local_idx_44 = idx_ii_43 - local_experts_start_idx;
                    int extent_44 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_44 = (1 << local_experts_stride_log2) - 1;
                    int is_local_44 = ((local_idx_44 >= 0 && local_idx_44 < extent_44 && (local_idx_44 & stride_mask_44) == 0) ? 1 : 0);
                    int is_local_ii_44 = is_local_44;
                    exp_off[32] = 0;
                    if (is_local_ii_44 != 0) {
                        uint32_t _shared_atomic_old_68;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_68) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_43)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[32] = (int)_shared_atomic_old_68;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_44 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 33 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_44 = 0;
                    unsigned int word0_45 = 0;
                    unsigned int word1_45 = 0;
                    unsigned int remote_45 = 0;
                    int src_rank_45 = 0;
                    int src_slot_45 = 0;
                    int packed_word_45 = 0;
                    packed_word_45 = topk_packed[expanded];
                    idx_ii_44 = packed_word_45 >> 16;
                    word0_45 = (unsigned int)packed_word_45 & 65535;
                    exp_idx[33] = idx_ii_44;
                    int local_idx_45 = idx_ii_44 - local_experts_start_idx;
                    int extent_45 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_45 = (1 << local_experts_stride_log2) - 1;
                    int is_local_45 = ((local_idx_45 >= 0 && local_idx_45 < extent_45 && (local_idx_45 & stride_mask_45) == 0) ? 1 : 0);
                    int is_local_ii_45 = is_local_45;
                    exp_off[33] = 0;
                    if (is_local_ii_45 != 0) {
                        uint32_t _shared_atomic_old_69;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_69) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_44)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[33] = (int)_shared_atomic_old_69;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_45 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 34 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_45 = 0;
                    unsigned int word0_46 = 0;
                    unsigned int word1_46 = 0;
                    unsigned int remote_46 = 0;
                    int src_rank_46 = 0;
                    int src_slot_46 = 0;
                    int packed_word_46 = 0;
                    packed_word_46 = topk_packed[expanded];
                    idx_ii_45 = packed_word_46 >> 16;
                    word0_46 = (unsigned int)packed_word_46 & 65535;
                    exp_idx[34] = idx_ii_45;
                    int local_idx_46 = idx_ii_45 - local_experts_start_idx;
                    int extent_46 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_46 = (1 << local_experts_stride_log2) - 1;
                    int is_local_46 = ((local_idx_46 >= 0 && local_idx_46 < extent_46 && (local_idx_46 & stride_mask_46) == 0) ? 1 : 0);
                    int is_local_ii_46 = is_local_46;
                    exp_off[34] = 0;
                    if (is_local_ii_46 != 0) {
                        uint32_t _shared_atomic_old_70;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_70) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_45)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[34] = (int)_shared_atomic_old_70;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_46 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 35 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_46 = 0;
                    unsigned int word0_47 = 0;
                    unsigned int word1_47 = 0;
                    unsigned int remote_47 = 0;
                    int src_rank_47 = 0;
                    int src_slot_47 = 0;
                    int packed_word_47 = 0;
                    packed_word_47 = topk_packed[expanded];
                    idx_ii_46 = packed_word_47 >> 16;
                    word0_47 = (unsigned int)packed_word_47 & 65535;
                    exp_idx[35] = idx_ii_46;
                    int local_idx_47 = idx_ii_46 - local_experts_start_idx;
                    int extent_47 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_47 = (1 << local_experts_stride_log2) - 1;
                    int is_local_47 = ((local_idx_47 >= 0 && local_idx_47 < extent_47 && (local_idx_47 & stride_mask_47) == 0) ? 1 : 0);
                    int is_local_ii_47 = is_local_47;
                    exp_off[35] = 0;
                    if (is_local_ii_47 != 0) {
                        uint32_t _shared_atomic_old_71;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_71) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_46)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[35] = (int)_shared_atomic_old_71;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_47 << 16);
                }
            }
        }
    }
    if (done == 0) {
        if (expanded_size >= 40 * grid_threads) {
            expanded = grid_tid + 36 * grid_threads;
            int idx_ii_47 = 0;
            unsigned int word0_48 = 0;
            unsigned int word1_48 = 0;
            unsigned int remote_48 = 0;
            int src_rank_48 = 0;
            int src_slot_48 = 0;
            int packed_word_48 = 0;
            packed_word_48 = topk_packed[expanded];
            idx_ii_47 = packed_word_48 >> 16;
            word0_48 = (unsigned int)packed_word_48 & 65535;
            exp_idx[36] = idx_ii_47;
            int local_idx_48 = idx_ii_47 - local_experts_start_idx;
            int extent_48 = num_local_experts << local_experts_stride_log2;
            int stride_mask_48 = (1 << local_experts_stride_log2) - 1;
            int is_local_48 = ((local_idx_48 >= 0 && local_idx_48 < extent_48 && (local_idx_48 & stride_mask_48) == 0) ? 1 : 0);
            int is_local_ii_48 = is_local_48;
            exp_off[36] = 0;
            if (is_local_ii_48 != 0) {
                uint32_t _shared_atomic_old_72;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_72) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_47)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[36] = (int)_shared_atomic_old_72;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_48 << 16);
            expanded = grid_tid + 37 * grid_threads;
            int idx_ii_0_9 = 0;
            unsigned int word0_1_9 = 0;
            unsigned int word1_2_9 = 0;
            unsigned int remote_3_9 = 0;
            int src_rank_4_9 = 0;
            int src_slot_5_9 = 0;
            int packed_word_6_9 = 0;
            packed_word_6_9 = topk_packed[expanded];
            idx_ii_0_9 = packed_word_6_9 >> 16;
            word0_1_9 = (unsigned int)packed_word_6_9 & 65535;
            exp_idx[37] = idx_ii_0_9;
            int local_idx_7_9 = idx_ii_0_9 - local_experts_start_idx;
            int extent_8_9 = num_local_experts << local_experts_stride_log2;
            int stride_mask_9_9 = (1 << local_experts_stride_log2) - 1;
            int is_local_10_9 = ((local_idx_7_9 >= 0 && local_idx_7_9 < extent_8_9 && (local_idx_7_9 & stride_mask_9_9) == 0) ? 1 : 0);
            int is_local_ii_11_9 = is_local_10_9;
            exp_off[37] = 0;
            if (is_local_ii_11_9 != 0) {
                uint32_t _shared_atomic_old_73;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_73) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_0_9)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[37] = (int)_shared_atomic_old_73;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_1_9 << 16);
            expanded = grid_tid + 38 * grid_threads;
            int idx_ii_12_9 = 0;
            unsigned int word0_13_9 = 0;
            unsigned int word1_14_9 = 0;
            unsigned int remote_15_9 = 0;
            int src_rank_16_9 = 0;
            int src_slot_17_9 = 0;
            int packed_word_18_9 = 0;
            packed_word_18_9 = topk_packed[expanded];
            idx_ii_12_9 = packed_word_18_9 >> 16;
            word0_13_9 = (unsigned int)packed_word_18_9 & 65535;
            exp_idx[38] = idx_ii_12_9;
            int local_idx_19_9 = idx_ii_12_9 - local_experts_start_idx;
            int extent_20_9 = num_local_experts << local_experts_stride_log2;
            int stride_mask_21_9 = (1 << local_experts_stride_log2) - 1;
            int is_local_22_9 = ((local_idx_19_9 >= 0 && local_idx_19_9 < extent_20_9 && (local_idx_19_9 & stride_mask_21_9) == 0) ? 1 : 0);
            int is_local_ii_23_9 = is_local_22_9;
            exp_off[38] = 0;
            if (is_local_ii_23_9 != 0) {
                uint32_t _shared_atomic_old_74;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_74) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_12_9)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[38] = (int)_shared_atomic_old_74;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_13_9 << 16);
            expanded = grid_tid + 39 * grid_threads;
            int idx_ii_24_9 = 0;
            unsigned int word0_25_9 = 0;
            unsigned int word1_26_9 = 0;
            unsigned int remote_27_9 = 0;
            int src_rank_28_9 = 0;
            int src_slot_29_9 = 0;
            int packed_word_30_9 = 0;
            packed_word_30_9 = topk_packed[expanded];
            idx_ii_24_9 = packed_word_30_9 >> 16;
            word0_25_9 = (unsigned int)packed_word_30_9 & 65535;
            exp_idx[39] = idx_ii_24_9;
            int local_idx_31_9 = idx_ii_24_9 - local_experts_start_idx;
            int extent_32_9 = num_local_experts << local_experts_stride_log2;
            int stride_mask_33_9 = (1 << local_experts_stride_log2) - 1;
            int is_local_34_9 = ((local_idx_31_9 >= 0 && local_idx_31_9 < extent_32_9 && (local_idx_31_9 & stride_mask_33_9) == 0) ? 1 : 0);
            int is_local_ii_35_9 = is_local_34_9;
            exp_off[39] = 0;
            if (is_local_ii_35_9 != 0) {
                uint32_t _shared_atomic_old_75;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_75) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_24_9)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[39] = (int)_shared_atomic_old_75;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_25_9 << 16);
        } else {
            if (done == 0) {
                expanded = grid_tid + 36 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_48 = 0;
                    unsigned int word0_49 = 0;
                    unsigned int word1_49 = 0;
                    unsigned int remote_49 = 0;
                    int src_rank_49 = 0;
                    int src_slot_49 = 0;
                    int packed_word_49 = 0;
                    packed_word_49 = topk_packed[expanded];
                    idx_ii_48 = packed_word_49 >> 16;
                    word0_49 = (unsigned int)packed_word_49 & 65535;
                    exp_idx[36] = idx_ii_48;
                    int local_idx_49 = idx_ii_48 - local_experts_start_idx;
                    int extent_49 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_49 = (1 << local_experts_stride_log2) - 1;
                    int is_local_49 = ((local_idx_49 >= 0 && local_idx_49 < extent_49 && (local_idx_49 & stride_mask_49) == 0) ? 1 : 0);
                    int is_local_ii_49 = is_local_49;
                    exp_off[36] = 0;
                    if (is_local_ii_49 != 0) {
                        uint32_t _shared_atomic_old_76;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_76) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_48)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[36] = (int)_shared_atomic_old_76;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_49 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 37 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_49 = 0;
                    unsigned int word0_50 = 0;
                    unsigned int word1_50 = 0;
                    unsigned int remote_50 = 0;
                    int src_rank_50 = 0;
                    int src_slot_50 = 0;
                    int packed_word_50 = 0;
                    packed_word_50 = topk_packed[expanded];
                    idx_ii_49 = packed_word_50 >> 16;
                    word0_50 = (unsigned int)packed_word_50 & 65535;
                    exp_idx[37] = idx_ii_49;
                    int local_idx_50 = idx_ii_49 - local_experts_start_idx;
                    int extent_50 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_50 = (1 << local_experts_stride_log2) - 1;
                    int is_local_50 = ((local_idx_50 >= 0 && local_idx_50 < extent_50 && (local_idx_50 & stride_mask_50) == 0) ? 1 : 0);
                    int is_local_ii_50 = is_local_50;
                    exp_off[37] = 0;
                    if (is_local_ii_50 != 0) {
                        uint32_t _shared_atomic_old_77;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_77) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_49)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[37] = (int)_shared_atomic_old_77;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_50 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 38 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_50 = 0;
                    unsigned int word0_51 = 0;
                    unsigned int word1_51 = 0;
                    unsigned int remote_51 = 0;
                    int src_rank_51 = 0;
                    int src_slot_51 = 0;
                    int packed_word_51 = 0;
                    packed_word_51 = topk_packed[expanded];
                    idx_ii_50 = packed_word_51 >> 16;
                    word0_51 = (unsigned int)packed_word_51 & 65535;
                    exp_idx[38] = idx_ii_50;
                    int local_idx_51 = idx_ii_50 - local_experts_start_idx;
                    int extent_51 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_51 = (1 << local_experts_stride_log2) - 1;
                    int is_local_51 = ((local_idx_51 >= 0 && local_idx_51 < extent_51 && (local_idx_51 & stride_mask_51) == 0) ? 1 : 0);
                    int is_local_ii_51 = is_local_51;
                    exp_off[38] = 0;
                    if (is_local_ii_51 != 0) {
                        uint32_t _shared_atomic_old_78;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_78) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_50)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[38] = (int)_shared_atomic_old_78;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_51 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 39 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_51 = 0;
                    unsigned int word0_52 = 0;
                    unsigned int word1_52 = 0;
                    unsigned int remote_52 = 0;
                    int src_rank_52 = 0;
                    int src_slot_52 = 0;
                    int packed_word_52 = 0;
                    packed_word_52 = topk_packed[expanded];
                    idx_ii_51 = packed_word_52 >> 16;
                    word0_52 = (unsigned int)packed_word_52 & 65535;
                    exp_idx[39] = idx_ii_51;
                    int local_idx_52 = idx_ii_51 - local_experts_start_idx;
                    int extent_52 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_52 = (1 << local_experts_stride_log2) - 1;
                    int is_local_52 = ((local_idx_52 >= 0 && local_idx_52 < extent_52 && (local_idx_52 & stride_mask_52) == 0) ? 1 : 0);
                    int is_local_ii_52 = is_local_52;
                    exp_off[39] = 0;
                    if (is_local_ii_52 != 0) {
                        uint32_t _shared_atomic_old_79;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_79) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_51)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[39] = (int)_shared_atomic_old_79;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_52 << 16);
                }
            }
        }
    }
    if (done == 0) {
        if (expanded_size >= 44 * grid_threads) {
            expanded = grid_tid + 40 * grid_threads;
            int idx_ii_52 = 0;
            unsigned int word0_53 = 0;
            unsigned int word1_53 = 0;
            unsigned int remote_53 = 0;
            int src_rank_53 = 0;
            int src_slot_53 = 0;
            int packed_word_53 = 0;
            packed_word_53 = topk_packed[expanded];
            idx_ii_52 = packed_word_53 >> 16;
            word0_53 = (unsigned int)packed_word_53 & 65535;
            exp_idx[40] = idx_ii_52;
            int local_idx_53 = idx_ii_52 - local_experts_start_idx;
            int extent_53 = num_local_experts << local_experts_stride_log2;
            int stride_mask_53 = (1 << local_experts_stride_log2) - 1;
            int is_local_53 = ((local_idx_53 >= 0 && local_idx_53 < extent_53 && (local_idx_53 & stride_mask_53) == 0) ? 1 : 0);
            int is_local_ii_53 = is_local_53;
            exp_off[40] = 0;
            if (is_local_ii_53 != 0) {
                uint32_t _shared_atomic_old_80;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_80) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_52)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[40] = (int)_shared_atomic_old_80;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_53 << 16);
            expanded = grid_tid + 41 * grid_threads;
            int idx_ii_0_10 = 0;
            unsigned int word0_1_10 = 0;
            unsigned int word1_2_10 = 0;
            unsigned int remote_3_10 = 0;
            int src_rank_4_10 = 0;
            int src_slot_5_10 = 0;
            int packed_word_6_10 = 0;
            packed_word_6_10 = topk_packed[expanded];
            idx_ii_0_10 = packed_word_6_10 >> 16;
            word0_1_10 = (unsigned int)packed_word_6_10 & 65535;
            exp_idx[41] = idx_ii_0_10;
            int local_idx_7_10 = idx_ii_0_10 - local_experts_start_idx;
            int extent_8_10 = num_local_experts << local_experts_stride_log2;
            int stride_mask_9_10 = (1 << local_experts_stride_log2) - 1;
            int is_local_10_10 = ((local_idx_7_10 >= 0 && local_idx_7_10 < extent_8_10 && (local_idx_7_10 & stride_mask_9_10) == 0) ? 1 : 0);
            int is_local_ii_11_10 = is_local_10_10;
            exp_off[41] = 0;
            if (is_local_ii_11_10 != 0) {
                uint32_t _shared_atomic_old_81;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_81) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_0_10)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[41] = (int)_shared_atomic_old_81;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_1_10 << 16);
            expanded = grid_tid + 42 * grid_threads;
            int idx_ii_12_10 = 0;
            unsigned int word0_13_10 = 0;
            unsigned int word1_14_10 = 0;
            unsigned int remote_15_10 = 0;
            int src_rank_16_10 = 0;
            int src_slot_17_10 = 0;
            int packed_word_18_10 = 0;
            packed_word_18_10 = topk_packed[expanded];
            idx_ii_12_10 = packed_word_18_10 >> 16;
            word0_13_10 = (unsigned int)packed_word_18_10 & 65535;
            exp_idx[42] = idx_ii_12_10;
            int local_idx_19_10 = idx_ii_12_10 - local_experts_start_idx;
            int extent_20_10 = num_local_experts << local_experts_stride_log2;
            int stride_mask_21_10 = (1 << local_experts_stride_log2) - 1;
            int is_local_22_10 = ((local_idx_19_10 >= 0 && local_idx_19_10 < extent_20_10 && (local_idx_19_10 & stride_mask_21_10) == 0) ? 1 : 0);
            int is_local_ii_23_10 = is_local_22_10;
            exp_off[42] = 0;
            if (is_local_ii_23_10 != 0) {
                uint32_t _shared_atomic_old_82;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_82) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_12_10)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[42] = (int)_shared_atomic_old_82;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_13_10 << 16);
            expanded = grid_tid + 43 * grid_threads;
            int idx_ii_24_10 = 0;
            unsigned int word0_25_10 = 0;
            unsigned int word1_26_10 = 0;
            unsigned int remote_27_10 = 0;
            int src_rank_28_10 = 0;
            int src_slot_29_10 = 0;
            int packed_word_30_10 = 0;
            packed_word_30_10 = topk_packed[expanded];
            idx_ii_24_10 = packed_word_30_10 >> 16;
            word0_25_10 = (unsigned int)packed_word_30_10 & 65535;
            exp_idx[43] = idx_ii_24_10;
            int local_idx_31_10 = idx_ii_24_10 - local_experts_start_idx;
            int extent_32_10 = num_local_experts << local_experts_stride_log2;
            int stride_mask_33_10 = (1 << local_experts_stride_log2) - 1;
            int is_local_34_10 = ((local_idx_31_10 >= 0 && local_idx_31_10 < extent_32_10 && (local_idx_31_10 & stride_mask_33_10) == 0) ? 1 : 0);
            int is_local_ii_35_10 = is_local_34_10;
            exp_off[43] = 0;
            if (is_local_ii_35_10 != 0) {
                uint32_t _shared_atomic_old_83;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_83) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_24_10)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[43] = (int)_shared_atomic_old_83;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_25_10 << 16);
        } else {
            if (done == 0) {
                expanded = grid_tid + 40 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_53 = 0;
                    unsigned int word0_54 = 0;
                    unsigned int word1_54 = 0;
                    unsigned int remote_54 = 0;
                    int src_rank_54 = 0;
                    int src_slot_54 = 0;
                    int packed_word_54 = 0;
                    packed_word_54 = topk_packed[expanded];
                    idx_ii_53 = packed_word_54 >> 16;
                    word0_54 = (unsigned int)packed_word_54 & 65535;
                    exp_idx[40] = idx_ii_53;
                    int local_idx_54 = idx_ii_53 - local_experts_start_idx;
                    int extent_54 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_54 = (1 << local_experts_stride_log2) - 1;
                    int is_local_54 = ((local_idx_54 >= 0 && local_idx_54 < extent_54 && (local_idx_54 & stride_mask_54) == 0) ? 1 : 0);
                    int is_local_ii_54 = is_local_54;
                    exp_off[40] = 0;
                    if (is_local_ii_54 != 0) {
                        uint32_t _shared_atomic_old_84;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_84) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_53)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[40] = (int)_shared_atomic_old_84;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_54 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 41 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_54 = 0;
                    unsigned int word0_55 = 0;
                    unsigned int word1_55 = 0;
                    unsigned int remote_55 = 0;
                    int src_rank_55 = 0;
                    int src_slot_55 = 0;
                    int packed_word_55 = 0;
                    packed_word_55 = topk_packed[expanded];
                    idx_ii_54 = packed_word_55 >> 16;
                    word0_55 = (unsigned int)packed_word_55 & 65535;
                    exp_idx[41] = idx_ii_54;
                    int local_idx_55 = idx_ii_54 - local_experts_start_idx;
                    int extent_55 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_55 = (1 << local_experts_stride_log2) - 1;
                    int is_local_55 = ((local_idx_55 >= 0 && local_idx_55 < extent_55 && (local_idx_55 & stride_mask_55) == 0) ? 1 : 0);
                    int is_local_ii_55 = is_local_55;
                    exp_off[41] = 0;
                    if (is_local_ii_55 != 0) {
                        uint32_t _shared_atomic_old_85;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_85) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_54)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[41] = (int)_shared_atomic_old_85;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_55 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 42 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_55 = 0;
                    unsigned int word0_56 = 0;
                    unsigned int word1_56 = 0;
                    unsigned int remote_56 = 0;
                    int src_rank_56 = 0;
                    int src_slot_56 = 0;
                    int packed_word_56 = 0;
                    packed_word_56 = topk_packed[expanded];
                    idx_ii_55 = packed_word_56 >> 16;
                    word0_56 = (unsigned int)packed_word_56 & 65535;
                    exp_idx[42] = idx_ii_55;
                    int local_idx_56 = idx_ii_55 - local_experts_start_idx;
                    int extent_56 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_56 = (1 << local_experts_stride_log2) - 1;
                    int is_local_56 = ((local_idx_56 >= 0 && local_idx_56 < extent_56 && (local_idx_56 & stride_mask_56) == 0) ? 1 : 0);
                    int is_local_ii_56 = is_local_56;
                    exp_off[42] = 0;
                    if (is_local_ii_56 != 0) {
                        uint32_t _shared_atomic_old_86;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_86) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_55)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[42] = (int)_shared_atomic_old_86;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_56 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 43 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_56 = 0;
                    unsigned int word0_57 = 0;
                    unsigned int word1_57 = 0;
                    unsigned int remote_57 = 0;
                    int src_rank_57 = 0;
                    int src_slot_57 = 0;
                    int packed_word_57 = 0;
                    packed_word_57 = topk_packed[expanded];
                    idx_ii_56 = packed_word_57 >> 16;
                    word0_57 = (unsigned int)packed_word_57 & 65535;
                    exp_idx[43] = idx_ii_56;
                    int local_idx_57 = idx_ii_56 - local_experts_start_idx;
                    int extent_57 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_57 = (1 << local_experts_stride_log2) - 1;
                    int is_local_57 = ((local_idx_57 >= 0 && local_idx_57 < extent_57 && (local_idx_57 & stride_mask_57) == 0) ? 1 : 0);
                    int is_local_ii_57 = is_local_57;
                    exp_off[43] = 0;
                    if (is_local_ii_57 != 0) {
                        uint32_t _shared_atomic_old_87;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_87) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_56)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[43] = (int)_shared_atomic_old_87;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_57 << 16);
                }
            }
        }
    }
    if (done == 0) {
        if (expanded_size >= 48 * grid_threads) {
            expanded = grid_tid + 44 * grid_threads;
            int idx_ii_57 = 0;
            unsigned int word0_58 = 0;
            unsigned int word1_58 = 0;
            unsigned int remote_58 = 0;
            int src_rank_58 = 0;
            int src_slot_58 = 0;
            int packed_word_58 = 0;
            packed_word_58 = topk_packed[expanded];
            idx_ii_57 = packed_word_58 >> 16;
            word0_58 = (unsigned int)packed_word_58 & 65535;
            exp_idx[44] = idx_ii_57;
            int local_idx_58 = idx_ii_57 - local_experts_start_idx;
            int extent_58 = num_local_experts << local_experts_stride_log2;
            int stride_mask_58 = (1 << local_experts_stride_log2) - 1;
            int is_local_58 = ((local_idx_58 >= 0 && local_idx_58 < extent_58 && (local_idx_58 & stride_mask_58) == 0) ? 1 : 0);
            int is_local_ii_58 = is_local_58;
            exp_off[44] = 0;
            if (is_local_ii_58 != 0) {
                uint32_t _shared_atomic_old_88;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_88) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_57)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[44] = (int)_shared_atomic_old_88;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_58 << 16);
            expanded = grid_tid + 45 * grid_threads;
            int idx_ii_0_11 = 0;
            unsigned int word0_1_11 = 0;
            unsigned int word1_2_11 = 0;
            unsigned int remote_3_11 = 0;
            int src_rank_4_11 = 0;
            int src_slot_5_11 = 0;
            int packed_word_6_11 = 0;
            packed_word_6_11 = topk_packed[expanded];
            idx_ii_0_11 = packed_word_6_11 >> 16;
            word0_1_11 = (unsigned int)packed_word_6_11 & 65535;
            exp_idx[45] = idx_ii_0_11;
            int local_idx_7_11 = idx_ii_0_11 - local_experts_start_idx;
            int extent_8_11 = num_local_experts << local_experts_stride_log2;
            int stride_mask_9_11 = (1 << local_experts_stride_log2) - 1;
            int is_local_10_11 = ((local_idx_7_11 >= 0 && local_idx_7_11 < extent_8_11 && (local_idx_7_11 & stride_mask_9_11) == 0) ? 1 : 0);
            int is_local_ii_11_11 = is_local_10_11;
            exp_off[45] = 0;
            if (is_local_ii_11_11 != 0) {
                uint32_t _shared_atomic_old_89;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_89) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_0_11)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[45] = (int)_shared_atomic_old_89;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_1_11 << 16);
            expanded = grid_tid + 46 * grid_threads;
            int idx_ii_12_11 = 0;
            unsigned int word0_13_11 = 0;
            unsigned int word1_14_11 = 0;
            unsigned int remote_15_11 = 0;
            int src_rank_16_11 = 0;
            int src_slot_17_11 = 0;
            int packed_word_18_11 = 0;
            packed_word_18_11 = topk_packed[expanded];
            idx_ii_12_11 = packed_word_18_11 >> 16;
            word0_13_11 = (unsigned int)packed_word_18_11 & 65535;
            exp_idx[46] = idx_ii_12_11;
            int local_idx_19_11 = idx_ii_12_11 - local_experts_start_idx;
            int extent_20_11 = num_local_experts << local_experts_stride_log2;
            int stride_mask_21_11 = (1 << local_experts_stride_log2) - 1;
            int is_local_22_11 = ((local_idx_19_11 >= 0 && local_idx_19_11 < extent_20_11 && (local_idx_19_11 & stride_mask_21_11) == 0) ? 1 : 0);
            int is_local_ii_23_11 = is_local_22_11;
            exp_off[46] = 0;
            if (is_local_ii_23_11 != 0) {
                uint32_t _shared_atomic_old_90;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_90) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_12_11)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[46] = (int)_shared_atomic_old_90;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_13_11 << 16);
            expanded = grid_tid + 47 * grid_threads;
            int idx_ii_24_11 = 0;
            unsigned int word0_25_11 = 0;
            unsigned int word1_26_11 = 0;
            unsigned int remote_27_11 = 0;
            int src_rank_28_11 = 0;
            int src_slot_29_11 = 0;
            int packed_word_30_11 = 0;
            packed_word_30_11 = topk_packed[expanded];
            idx_ii_24_11 = packed_word_30_11 >> 16;
            word0_25_11 = (unsigned int)packed_word_30_11 & 65535;
            exp_idx[47] = idx_ii_24_11;
            int local_idx_31_11 = idx_ii_24_11 - local_experts_start_idx;
            int extent_32_11 = num_local_experts << local_experts_stride_log2;
            int stride_mask_33_11 = (1 << local_experts_stride_log2) - 1;
            int is_local_34_11 = ((local_idx_31_11 >= 0 && local_idx_31_11 < extent_32_11 && (local_idx_31_11 & stride_mask_33_11) == 0) ? 1 : 0);
            int is_local_ii_35_11 = is_local_34_11;
            exp_off[47] = 0;
            if (is_local_ii_35_11 != 0) {
                uint32_t _shared_atomic_old_91;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_91) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_24_11)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[47] = (int)_shared_atomic_old_91;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_25_11 << 16);
        } else {
            if (done == 0) {
                expanded = grid_tid + 44 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_58 = 0;
                    unsigned int word0_59 = 0;
                    unsigned int word1_59 = 0;
                    unsigned int remote_59 = 0;
                    int src_rank_59 = 0;
                    int src_slot_59 = 0;
                    int packed_word_59 = 0;
                    packed_word_59 = topk_packed[expanded];
                    idx_ii_58 = packed_word_59 >> 16;
                    word0_59 = (unsigned int)packed_word_59 & 65535;
                    exp_idx[44] = idx_ii_58;
                    int local_idx_59 = idx_ii_58 - local_experts_start_idx;
                    int extent_59 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_59 = (1 << local_experts_stride_log2) - 1;
                    int is_local_59 = ((local_idx_59 >= 0 && local_idx_59 < extent_59 && (local_idx_59 & stride_mask_59) == 0) ? 1 : 0);
                    int is_local_ii_59 = is_local_59;
                    exp_off[44] = 0;
                    if (is_local_ii_59 != 0) {
                        uint32_t _shared_atomic_old_92;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_92) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_58)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[44] = (int)_shared_atomic_old_92;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_59 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 45 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_59 = 0;
                    unsigned int word0_60 = 0;
                    unsigned int word1_60 = 0;
                    unsigned int remote_60 = 0;
                    int src_rank_60 = 0;
                    int src_slot_60 = 0;
                    int packed_word_60 = 0;
                    packed_word_60 = topk_packed[expanded];
                    idx_ii_59 = packed_word_60 >> 16;
                    word0_60 = (unsigned int)packed_word_60 & 65535;
                    exp_idx[45] = idx_ii_59;
                    int local_idx_60 = idx_ii_59 - local_experts_start_idx;
                    int extent_60 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_60 = (1 << local_experts_stride_log2) - 1;
                    int is_local_60 = ((local_idx_60 >= 0 && local_idx_60 < extent_60 && (local_idx_60 & stride_mask_60) == 0) ? 1 : 0);
                    int is_local_ii_60 = is_local_60;
                    exp_off[45] = 0;
                    if (is_local_ii_60 != 0) {
                        uint32_t _shared_atomic_old_93;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_93) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_59)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[45] = (int)_shared_atomic_old_93;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_60 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 46 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_60 = 0;
                    unsigned int word0_61 = 0;
                    unsigned int word1_61 = 0;
                    unsigned int remote_61 = 0;
                    int src_rank_61 = 0;
                    int src_slot_61 = 0;
                    int packed_word_61 = 0;
                    packed_word_61 = topk_packed[expanded];
                    idx_ii_60 = packed_word_61 >> 16;
                    word0_61 = (unsigned int)packed_word_61 & 65535;
                    exp_idx[46] = idx_ii_60;
                    int local_idx_61 = idx_ii_60 - local_experts_start_idx;
                    int extent_61 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_61 = (1 << local_experts_stride_log2) - 1;
                    int is_local_61 = ((local_idx_61 >= 0 && local_idx_61 < extent_61 && (local_idx_61 & stride_mask_61) == 0) ? 1 : 0);
                    int is_local_ii_61 = is_local_61;
                    exp_off[46] = 0;
                    if (is_local_ii_61 != 0) {
                        uint32_t _shared_atomic_old_94;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_94) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_60)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[46] = (int)_shared_atomic_old_94;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_61 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 47 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_61 = 0;
                    unsigned int word0_62 = 0;
                    unsigned int word1_62 = 0;
                    unsigned int remote_62 = 0;
                    int src_rank_62 = 0;
                    int src_slot_62 = 0;
                    int packed_word_62 = 0;
                    packed_word_62 = topk_packed[expanded];
                    idx_ii_61 = packed_word_62 >> 16;
                    word0_62 = (unsigned int)packed_word_62 & 65535;
                    exp_idx[47] = idx_ii_61;
                    int local_idx_62 = idx_ii_61 - local_experts_start_idx;
                    int extent_62 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_62 = (1 << local_experts_stride_log2) - 1;
                    int is_local_62 = ((local_idx_62 >= 0 && local_idx_62 < extent_62 && (local_idx_62 & stride_mask_62) == 0) ? 1 : 0);
                    int is_local_ii_62 = is_local_62;
                    exp_off[47] = 0;
                    if (is_local_ii_62 != 0) {
                        uint32_t _shared_atomic_old_95;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_95) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_61)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[47] = (int)_shared_atomic_old_95;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_62 << 16);
                }
            }
        }
    }
    if (done == 0) {
        if (expanded_size >= 52 * grid_threads) {
            expanded = grid_tid + 48 * grid_threads;
            int idx_ii_62 = 0;
            unsigned int word0_63 = 0;
            unsigned int word1_63 = 0;
            unsigned int remote_63 = 0;
            int src_rank_63 = 0;
            int src_slot_63 = 0;
            int packed_word_63 = 0;
            packed_word_63 = topk_packed[expanded];
            idx_ii_62 = packed_word_63 >> 16;
            word0_63 = (unsigned int)packed_word_63 & 65535;
            exp_idx[48] = idx_ii_62;
            int local_idx_63 = idx_ii_62 - local_experts_start_idx;
            int extent_63 = num_local_experts << local_experts_stride_log2;
            int stride_mask_63 = (1 << local_experts_stride_log2) - 1;
            int is_local_63 = ((local_idx_63 >= 0 && local_idx_63 < extent_63 && (local_idx_63 & stride_mask_63) == 0) ? 1 : 0);
            int is_local_ii_63 = is_local_63;
            exp_off[48] = 0;
            if (is_local_ii_63 != 0) {
                uint32_t _shared_atomic_old_96;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_96) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_62)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[48] = (int)_shared_atomic_old_96;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_63 << 16);
            expanded = grid_tid + 49 * grid_threads;
            int idx_ii_0_12 = 0;
            unsigned int word0_1_12 = 0;
            unsigned int word1_2_12 = 0;
            unsigned int remote_3_12 = 0;
            int src_rank_4_12 = 0;
            int src_slot_5_12 = 0;
            int packed_word_6_12 = 0;
            packed_word_6_12 = topk_packed[expanded];
            idx_ii_0_12 = packed_word_6_12 >> 16;
            word0_1_12 = (unsigned int)packed_word_6_12 & 65535;
            exp_idx[49] = idx_ii_0_12;
            int local_idx_7_12 = idx_ii_0_12 - local_experts_start_idx;
            int extent_8_12 = num_local_experts << local_experts_stride_log2;
            int stride_mask_9_12 = (1 << local_experts_stride_log2) - 1;
            int is_local_10_12 = ((local_idx_7_12 >= 0 && local_idx_7_12 < extent_8_12 && (local_idx_7_12 & stride_mask_9_12) == 0) ? 1 : 0);
            int is_local_ii_11_12 = is_local_10_12;
            exp_off[49] = 0;
            if (is_local_ii_11_12 != 0) {
                uint32_t _shared_atomic_old_97;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_97) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_0_12)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[49] = (int)_shared_atomic_old_97;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_1_12 << 16);
            expanded = grid_tid + 50 * grid_threads;
            int idx_ii_12_12 = 0;
            unsigned int word0_13_12 = 0;
            unsigned int word1_14_12 = 0;
            unsigned int remote_15_12 = 0;
            int src_rank_16_12 = 0;
            int src_slot_17_12 = 0;
            int packed_word_18_12 = 0;
            packed_word_18_12 = topk_packed[expanded];
            idx_ii_12_12 = packed_word_18_12 >> 16;
            word0_13_12 = (unsigned int)packed_word_18_12 & 65535;
            exp_idx[50] = idx_ii_12_12;
            int local_idx_19_12 = idx_ii_12_12 - local_experts_start_idx;
            int extent_20_12 = num_local_experts << local_experts_stride_log2;
            int stride_mask_21_12 = (1 << local_experts_stride_log2) - 1;
            int is_local_22_12 = ((local_idx_19_12 >= 0 && local_idx_19_12 < extent_20_12 && (local_idx_19_12 & stride_mask_21_12) == 0) ? 1 : 0);
            int is_local_ii_23_12 = is_local_22_12;
            exp_off[50] = 0;
            if (is_local_ii_23_12 != 0) {
                uint32_t _shared_atomic_old_98;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_98) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_12_12)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[50] = (int)_shared_atomic_old_98;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_13_12 << 16);
            expanded = grid_tid + 51 * grid_threads;
            int idx_ii_24_12 = 0;
            unsigned int word0_25_12 = 0;
            unsigned int word1_26_12 = 0;
            unsigned int remote_27_12 = 0;
            int src_rank_28_12 = 0;
            int src_slot_29_12 = 0;
            int packed_word_30_12 = 0;
            packed_word_30_12 = topk_packed[expanded];
            idx_ii_24_12 = packed_word_30_12 >> 16;
            word0_25_12 = (unsigned int)packed_word_30_12 & 65535;
            exp_idx[51] = idx_ii_24_12;
            int local_idx_31_12 = idx_ii_24_12 - local_experts_start_idx;
            int extent_32_12 = num_local_experts << local_experts_stride_log2;
            int stride_mask_33_12 = (1 << local_experts_stride_log2) - 1;
            int is_local_34_12 = ((local_idx_31_12 >= 0 && local_idx_31_12 < extent_32_12 && (local_idx_31_12 & stride_mask_33_12) == 0) ? 1 : 0);
            int is_local_ii_35_12 = is_local_34_12;
            exp_off[51] = 0;
            if (is_local_ii_35_12 != 0) {
                uint32_t _shared_atomic_old_99;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_99) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_24_12)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[51] = (int)_shared_atomic_old_99;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_25_12 << 16);
        } else {
            if (done == 0) {
                expanded = grid_tid + 48 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_63 = 0;
                    unsigned int word0_64 = 0;
                    unsigned int word1_64 = 0;
                    unsigned int remote_64 = 0;
                    int src_rank_64 = 0;
                    int src_slot_64 = 0;
                    int packed_word_64 = 0;
                    packed_word_64 = topk_packed[expanded];
                    idx_ii_63 = packed_word_64 >> 16;
                    word0_64 = (unsigned int)packed_word_64 & 65535;
                    exp_idx[48] = idx_ii_63;
                    int local_idx_64 = idx_ii_63 - local_experts_start_idx;
                    int extent_64 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_64 = (1 << local_experts_stride_log2) - 1;
                    int is_local_64 = ((local_idx_64 >= 0 && local_idx_64 < extent_64 && (local_idx_64 & stride_mask_64) == 0) ? 1 : 0);
                    int is_local_ii_64 = is_local_64;
                    exp_off[48] = 0;
                    if (is_local_ii_64 != 0) {
                        uint32_t _shared_atomic_old_100;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_100) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_63)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[48] = (int)_shared_atomic_old_100;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_64 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 49 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_64 = 0;
                    unsigned int word0_65 = 0;
                    unsigned int word1_65 = 0;
                    unsigned int remote_65 = 0;
                    int src_rank_65 = 0;
                    int src_slot_65 = 0;
                    int packed_word_65 = 0;
                    packed_word_65 = topk_packed[expanded];
                    idx_ii_64 = packed_word_65 >> 16;
                    word0_65 = (unsigned int)packed_word_65 & 65535;
                    exp_idx[49] = idx_ii_64;
                    int local_idx_65 = idx_ii_64 - local_experts_start_idx;
                    int extent_65 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_65 = (1 << local_experts_stride_log2) - 1;
                    int is_local_65 = ((local_idx_65 >= 0 && local_idx_65 < extent_65 && (local_idx_65 & stride_mask_65) == 0) ? 1 : 0);
                    int is_local_ii_65 = is_local_65;
                    exp_off[49] = 0;
                    if (is_local_ii_65 != 0) {
                        uint32_t _shared_atomic_old_101;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_101) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_64)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[49] = (int)_shared_atomic_old_101;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_65 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 50 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_65 = 0;
                    unsigned int word0_66 = 0;
                    unsigned int word1_66 = 0;
                    unsigned int remote_66 = 0;
                    int src_rank_66 = 0;
                    int src_slot_66 = 0;
                    int packed_word_66 = 0;
                    packed_word_66 = topk_packed[expanded];
                    idx_ii_65 = packed_word_66 >> 16;
                    word0_66 = (unsigned int)packed_word_66 & 65535;
                    exp_idx[50] = idx_ii_65;
                    int local_idx_66 = idx_ii_65 - local_experts_start_idx;
                    int extent_66 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_66 = (1 << local_experts_stride_log2) - 1;
                    int is_local_66 = ((local_idx_66 >= 0 && local_idx_66 < extent_66 && (local_idx_66 & stride_mask_66) == 0) ? 1 : 0);
                    int is_local_ii_66 = is_local_66;
                    exp_off[50] = 0;
                    if (is_local_ii_66 != 0) {
                        uint32_t _shared_atomic_old_102;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_102) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_65)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[50] = (int)_shared_atomic_old_102;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_66 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 51 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_66 = 0;
                    unsigned int word0_67 = 0;
                    unsigned int word1_67 = 0;
                    unsigned int remote_67 = 0;
                    int src_rank_67 = 0;
                    int src_slot_67 = 0;
                    int packed_word_67 = 0;
                    packed_word_67 = topk_packed[expanded];
                    idx_ii_66 = packed_word_67 >> 16;
                    word0_67 = (unsigned int)packed_word_67 & 65535;
                    exp_idx[51] = idx_ii_66;
                    int local_idx_67 = idx_ii_66 - local_experts_start_idx;
                    int extent_67 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_67 = (1 << local_experts_stride_log2) - 1;
                    int is_local_67 = ((local_idx_67 >= 0 && local_idx_67 < extent_67 && (local_idx_67 & stride_mask_67) == 0) ? 1 : 0);
                    int is_local_ii_67 = is_local_67;
                    exp_off[51] = 0;
                    if (is_local_ii_67 != 0) {
                        uint32_t _shared_atomic_old_103;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_103) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_66)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[51] = (int)_shared_atomic_old_103;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_67 << 16);
                }
            }
        }
    }
    if (done == 0) {
        if (expanded_size >= 56 * grid_threads) {
            expanded = grid_tid + 52 * grid_threads;
            int idx_ii_67 = 0;
            unsigned int word0_68 = 0;
            unsigned int word1_68 = 0;
            unsigned int remote_68 = 0;
            int src_rank_68 = 0;
            int src_slot_68 = 0;
            int packed_word_68 = 0;
            packed_word_68 = topk_packed[expanded];
            idx_ii_67 = packed_word_68 >> 16;
            word0_68 = (unsigned int)packed_word_68 & 65535;
            exp_idx[52] = idx_ii_67;
            int local_idx_68 = idx_ii_67 - local_experts_start_idx;
            int extent_68 = num_local_experts << local_experts_stride_log2;
            int stride_mask_68 = (1 << local_experts_stride_log2) - 1;
            int is_local_68 = ((local_idx_68 >= 0 && local_idx_68 < extent_68 && (local_idx_68 & stride_mask_68) == 0) ? 1 : 0);
            int is_local_ii_68 = is_local_68;
            exp_off[52] = 0;
            if (is_local_ii_68 != 0) {
                uint32_t _shared_atomic_old_104;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_104) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_67)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[52] = (int)_shared_atomic_old_104;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_68 << 16);
            expanded = grid_tid + 53 * grid_threads;
            int idx_ii_0_13 = 0;
            unsigned int word0_1_13 = 0;
            unsigned int word1_2_13 = 0;
            unsigned int remote_3_13 = 0;
            int src_rank_4_13 = 0;
            int src_slot_5_13 = 0;
            int packed_word_6_13 = 0;
            packed_word_6_13 = topk_packed[expanded];
            idx_ii_0_13 = packed_word_6_13 >> 16;
            word0_1_13 = (unsigned int)packed_word_6_13 & 65535;
            exp_idx[53] = idx_ii_0_13;
            int local_idx_7_13 = idx_ii_0_13 - local_experts_start_idx;
            int extent_8_13 = num_local_experts << local_experts_stride_log2;
            int stride_mask_9_13 = (1 << local_experts_stride_log2) - 1;
            int is_local_10_13 = ((local_idx_7_13 >= 0 && local_idx_7_13 < extent_8_13 && (local_idx_7_13 & stride_mask_9_13) == 0) ? 1 : 0);
            int is_local_ii_11_13 = is_local_10_13;
            exp_off[53] = 0;
            if (is_local_ii_11_13 != 0) {
                uint32_t _shared_atomic_old_105;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_105) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_0_13)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[53] = (int)_shared_atomic_old_105;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_1_13 << 16);
            expanded = grid_tid + 54 * grid_threads;
            int idx_ii_12_13 = 0;
            unsigned int word0_13_13 = 0;
            unsigned int word1_14_13 = 0;
            unsigned int remote_15_13 = 0;
            int src_rank_16_13 = 0;
            int src_slot_17_13 = 0;
            int packed_word_18_13 = 0;
            packed_word_18_13 = topk_packed[expanded];
            idx_ii_12_13 = packed_word_18_13 >> 16;
            word0_13_13 = (unsigned int)packed_word_18_13 & 65535;
            exp_idx[54] = idx_ii_12_13;
            int local_idx_19_13 = idx_ii_12_13 - local_experts_start_idx;
            int extent_20_13 = num_local_experts << local_experts_stride_log2;
            int stride_mask_21_13 = (1 << local_experts_stride_log2) - 1;
            int is_local_22_13 = ((local_idx_19_13 >= 0 && local_idx_19_13 < extent_20_13 && (local_idx_19_13 & stride_mask_21_13) == 0) ? 1 : 0);
            int is_local_ii_23_13 = is_local_22_13;
            exp_off[54] = 0;
            if (is_local_ii_23_13 != 0) {
                uint32_t _shared_atomic_old_106;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_106) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_12_13)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[54] = (int)_shared_atomic_old_106;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_13_13 << 16);
            expanded = grid_tid + 55 * grid_threads;
            int idx_ii_24_13 = 0;
            unsigned int word0_25_13 = 0;
            unsigned int word1_26_13 = 0;
            unsigned int remote_27_13 = 0;
            int src_rank_28_13 = 0;
            int src_slot_29_13 = 0;
            int packed_word_30_13 = 0;
            packed_word_30_13 = topk_packed[expanded];
            idx_ii_24_13 = packed_word_30_13 >> 16;
            word0_25_13 = (unsigned int)packed_word_30_13 & 65535;
            exp_idx[55] = idx_ii_24_13;
            int local_idx_31_13 = idx_ii_24_13 - local_experts_start_idx;
            int extent_32_13 = num_local_experts << local_experts_stride_log2;
            int stride_mask_33_13 = (1 << local_experts_stride_log2) - 1;
            int is_local_34_13 = ((local_idx_31_13 >= 0 && local_idx_31_13 < extent_32_13 && (local_idx_31_13 & stride_mask_33_13) == 0) ? 1 : 0);
            int is_local_ii_35_13 = is_local_34_13;
            exp_off[55] = 0;
            if (is_local_ii_35_13 != 0) {
                uint32_t _shared_atomic_old_107;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_107) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_24_13)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[55] = (int)_shared_atomic_old_107;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_25_13 << 16);
        } else {
            if (done == 0) {
                expanded = grid_tid + 52 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_68 = 0;
                    unsigned int word0_69 = 0;
                    unsigned int word1_69 = 0;
                    unsigned int remote_69 = 0;
                    int src_rank_69 = 0;
                    int src_slot_69 = 0;
                    int packed_word_69 = 0;
                    packed_word_69 = topk_packed[expanded];
                    idx_ii_68 = packed_word_69 >> 16;
                    word0_69 = (unsigned int)packed_word_69 & 65535;
                    exp_idx[52] = idx_ii_68;
                    int local_idx_69 = idx_ii_68 - local_experts_start_idx;
                    int extent_69 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_69 = (1 << local_experts_stride_log2) - 1;
                    int is_local_69 = ((local_idx_69 >= 0 && local_idx_69 < extent_69 && (local_idx_69 & stride_mask_69) == 0) ? 1 : 0);
                    int is_local_ii_69 = is_local_69;
                    exp_off[52] = 0;
                    if (is_local_ii_69 != 0) {
                        uint32_t _shared_atomic_old_108;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_108) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_68)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[52] = (int)_shared_atomic_old_108;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_69 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 53 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_69 = 0;
                    unsigned int word0_70 = 0;
                    unsigned int word1_70 = 0;
                    unsigned int remote_70 = 0;
                    int src_rank_70 = 0;
                    int src_slot_70 = 0;
                    int packed_word_70 = 0;
                    packed_word_70 = topk_packed[expanded];
                    idx_ii_69 = packed_word_70 >> 16;
                    word0_70 = (unsigned int)packed_word_70 & 65535;
                    exp_idx[53] = idx_ii_69;
                    int local_idx_70 = idx_ii_69 - local_experts_start_idx;
                    int extent_70 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_70 = (1 << local_experts_stride_log2) - 1;
                    int is_local_70 = ((local_idx_70 >= 0 && local_idx_70 < extent_70 && (local_idx_70 & stride_mask_70) == 0) ? 1 : 0);
                    int is_local_ii_70 = is_local_70;
                    exp_off[53] = 0;
                    if (is_local_ii_70 != 0) {
                        uint32_t _shared_atomic_old_109;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_109) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_69)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[53] = (int)_shared_atomic_old_109;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_70 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 54 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_70 = 0;
                    unsigned int word0_71 = 0;
                    unsigned int word1_71 = 0;
                    unsigned int remote_71 = 0;
                    int src_rank_71 = 0;
                    int src_slot_71 = 0;
                    int packed_word_71 = 0;
                    packed_word_71 = topk_packed[expanded];
                    idx_ii_70 = packed_word_71 >> 16;
                    word0_71 = (unsigned int)packed_word_71 & 65535;
                    exp_idx[54] = idx_ii_70;
                    int local_idx_71 = idx_ii_70 - local_experts_start_idx;
                    int extent_71 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_71 = (1 << local_experts_stride_log2) - 1;
                    int is_local_71 = ((local_idx_71 >= 0 && local_idx_71 < extent_71 && (local_idx_71 & stride_mask_71) == 0) ? 1 : 0);
                    int is_local_ii_71 = is_local_71;
                    exp_off[54] = 0;
                    if (is_local_ii_71 != 0) {
                        uint32_t _shared_atomic_old_110;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_110) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_70)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[54] = (int)_shared_atomic_old_110;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_71 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 55 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_71 = 0;
                    unsigned int word0_72 = 0;
                    unsigned int word1_72 = 0;
                    unsigned int remote_72 = 0;
                    int src_rank_72 = 0;
                    int src_slot_72 = 0;
                    int packed_word_72 = 0;
                    packed_word_72 = topk_packed[expanded];
                    idx_ii_71 = packed_word_72 >> 16;
                    word0_72 = (unsigned int)packed_word_72 & 65535;
                    exp_idx[55] = idx_ii_71;
                    int local_idx_72 = idx_ii_71 - local_experts_start_idx;
                    int extent_72 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_72 = (1 << local_experts_stride_log2) - 1;
                    int is_local_72 = ((local_idx_72 >= 0 && local_idx_72 < extent_72 && (local_idx_72 & stride_mask_72) == 0) ? 1 : 0);
                    int is_local_ii_72 = is_local_72;
                    exp_off[55] = 0;
                    if (is_local_ii_72 != 0) {
                        uint32_t _shared_atomic_old_111;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_111) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_71)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[55] = (int)_shared_atomic_old_111;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_72 << 16);
                }
            }
        }
    }
    if (done == 0) {
        if (expanded_size >= 60 * grid_threads) {
            expanded = grid_tid + 56 * grid_threads;
            int idx_ii_72 = 0;
            unsigned int word0_73 = 0;
            unsigned int word1_73 = 0;
            unsigned int remote_73 = 0;
            int src_rank_73 = 0;
            int src_slot_73 = 0;
            int packed_word_73 = 0;
            packed_word_73 = topk_packed[expanded];
            idx_ii_72 = packed_word_73 >> 16;
            word0_73 = (unsigned int)packed_word_73 & 65535;
            exp_idx[56] = idx_ii_72;
            int local_idx_73 = idx_ii_72 - local_experts_start_idx;
            int extent_73 = num_local_experts << local_experts_stride_log2;
            int stride_mask_73 = (1 << local_experts_stride_log2) - 1;
            int is_local_73 = ((local_idx_73 >= 0 && local_idx_73 < extent_73 && (local_idx_73 & stride_mask_73) == 0) ? 1 : 0);
            int is_local_ii_73 = is_local_73;
            exp_off[56] = 0;
            if (is_local_ii_73 != 0) {
                uint32_t _shared_atomic_old_112;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_112) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_72)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[56] = (int)_shared_atomic_old_112;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_73 << 16);
            expanded = grid_tid + 57 * grid_threads;
            int idx_ii_0_14 = 0;
            unsigned int word0_1_14 = 0;
            unsigned int word1_2_14 = 0;
            unsigned int remote_3_14 = 0;
            int src_rank_4_14 = 0;
            int src_slot_5_14 = 0;
            int packed_word_6_14 = 0;
            packed_word_6_14 = topk_packed[expanded];
            idx_ii_0_14 = packed_word_6_14 >> 16;
            word0_1_14 = (unsigned int)packed_word_6_14 & 65535;
            exp_idx[57] = idx_ii_0_14;
            int local_idx_7_14 = idx_ii_0_14 - local_experts_start_idx;
            int extent_8_14 = num_local_experts << local_experts_stride_log2;
            int stride_mask_9_14 = (1 << local_experts_stride_log2) - 1;
            int is_local_10_14 = ((local_idx_7_14 >= 0 && local_idx_7_14 < extent_8_14 && (local_idx_7_14 & stride_mask_9_14) == 0) ? 1 : 0);
            int is_local_ii_11_14 = is_local_10_14;
            exp_off[57] = 0;
            if (is_local_ii_11_14 != 0) {
                uint32_t _shared_atomic_old_113;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_113) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_0_14)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[57] = (int)_shared_atomic_old_113;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_1_14 << 16);
            expanded = grid_tid + 58 * grid_threads;
            int idx_ii_12_14 = 0;
            unsigned int word0_13_14 = 0;
            unsigned int word1_14_14 = 0;
            unsigned int remote_15_14 = 0;
            int src_rank_16_14 = 0;
            int src_slot_17_14 = 0;
            int packed_word_18_14 = 0;
            packed_word_18_14 = topk_packed[expanded];
            idx_ii_12_14 = packed_word_18_14 >> 16;
            word0_13_14 = (unsigned int)packed_word_18_14 & 65535;
            exp_idx[58] = idx_ii_12_14;
            int local_idx_19_14 = idx_ii_12_14 - local_experts_start_idx;
            int extent_20_14 = num_local_experts << local_experts_stride_log2;
            int stride_mask_21_14 = (1 << local_experts_stride_log2) - 1;
            int is_local_22_14 = ((local_idx_19_14 >= 0 && local_idx_19_14 < extent_20_14 && (local_idx_19_14 & stride_mask_21_14) == 0) ? 1 : 0);
            int is_local_ii_23_14 = is_local_22_14;
            exp_off[58] = 0;
            if (is_local_ii_23_14 != 0) {
                uint32_t _shared_atomic_old_114;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_114) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_12_14)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[58] = (int)_shared_atomic_old_114;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_13_14 << 16);
            expanded = grid_tid + 59 * grid_threads;
            int idx_ii_24_14 = 0;
            unsigned int word0_25_14 = 0;
            unsigned int word1_26_14 = 0;
            unsigned int remote_27_14 = 0;
            int src_rank_28_14 = 0;
            int src_slot_29_14 = 0;
            int packed_word_30_14 = 0;
            packed_word_30_14 = topk_packed[expanded];
            idx_ii_24_14 = packed_word_30_14 >> 16;
            word0_25_14 = (unsigned int)packed_word_30_14 & 65535;
            exp_idx[59] = idx_ii_24_14;
            int local_idx_31_14 = idx_ii_24_14 - local_experts_start_idx;
            int extent_32_14 = num_local_experts << local_experts_stride_log2;
            int stride_mask_33_14 = (1 << local_experts_stride_log2) - 1;
            int is_local_34_14 = ((local_idx_31_14 >= 0 && local_idx_31_14 < extent_32_14 && (local_idx_31_14 & stride_mask_33_14) == 0) ? 1 : 0);
            int is_local_ii_35_14 = is_local_34_14;
            exp_off[59] = 0;
            if (is_local_ii_35_14 != 0) {
                uint32_t _shared_atomic_old_115;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_115) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_24_14)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[59] = (int)_shared_atomic_old_115;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_25_14 << 16);
        } else {
            if (done == 0) {
                expanded = grid_tid + 56 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_73 = 0;
                    unsigned int word0_74 = 0;
                    unsigned int word1_74 = 0;
                    unsigned int remote_74 = 0;
                    int src_rank_74 = 0;
                    int src_slot_74 = 0;
                    int packed_word_74 = 0;
                    packed_word_74 = topk_packed[expanded];
                    idx_ii_73 = packed_word_74 >> 16;
                    word0_74 = (unsigned int)packed_word_74 & 65535;
                    exp_idx[56] = idx_ii_73;
                    int local_idx_74 = idx_ii_73 - local_experts_start_idx;
                    int extent_74 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_74 = (1 << local_experts_stride_log2) - 1;
                    int is_local_74 = ((local_idx_74 >= 0 && local_idx_74 < extent_74 && (local_idx_74 & stride_mask_74) == 0) ? 1 : 0);
                    int is_local_ii_74 = is_local_74;
                    exp_off[56] = 0;
                    if (is_local_ii_74 != 0) {
                        uint32_t _shared_atomic_old_116;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_116) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_73)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[56] = (int)_shared_atomic_old_116;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_74 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 57 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_74 = 0;
                    unsigned int word0_75 = 0;
                    unsigned int word1_75 = 0;
                    unsigned int remote_75 = 0;
                    int src_rank_75 = 0;
                    int src_slot_75 = 0;
                    int packed_word_75 = 0;
                    packed_word_75 = topk_packed[expanded];
                    idx_ii_74 = packed_word_75 >> 16;
                    word0_75 = (unsigned int)packed_word_75 & 65535;
                    exp_idx[57] = idx_ii_74;
                    int local_idx_75 = idx_ii_74 - local_experts_start_idx;
                    int extent_75 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_75 = (1 << local_experts_stride_log2) - 1;
                    int is_local_75 = ((local_idx_75 >= 0 && local_idx_75 < extent_75 && (local_idx_75 & stride_mask_75) == 0) ? 1 : 0);
                    int is_local_ii_75 = is_local_75;
                    exp_off[57] = 0;
                    if (is_local_ii_75 != 0) {
                        uint32_t _shared_atomic_old_117;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_117) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_74)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[57] = (int)_shared_atomic_old_117;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_75 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 58 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_75 = 0;
                    unsigned int word0_76 = 0;
                    unsigned int word1_76 = 0;
                    unsigned int remote_76 = 0;
                    int src_rank_76 = 0;
                    int src_slot_76 = 0;
                    int packed_word_76 = 0;
                    packed_word_76 = topk_packed[expanded];
                    idx_ii_75 = packed_word_76 >> 16;
                    word0_76 = (unsigned int)packed_word_76 & 65535;
                    exp_idx[58] = idx_ii_75;
                    int local_idx_76 = idx_ii_75 - local_experts_start_idx;
                    int extent_76 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_76 = (1 << local_experts_stride_log2) - 1;
                    int is_local_76 = ((local_idx_76 >= 0 && local_idx_76 < extent_76 && (local_idx_76 & stride_mask_76) == 0) ? 1 : 0);
                    int is_local_ii_76 = is_local_76;
                    exp_off[58] = 0;
                    if (is_local_ii_76 != 0) {
                        uint32_t _shared_atomic_old_118;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_118) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_75)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[58] = (int)_shared_atomic_old_118;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_76 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 59 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_76 = 0;
                    unsigned int word0_77 = 0;
                    unsigned int word1_77 = 0;
                    unsigned int remote_77 = 0;
                    int src_rank_77 = 0;
                    int src_slot_77 = 0;
                    int packed_word_77 = 0;
                    packed_word_77 = topk_packed[expanded];
                    idx_ii_76 = packed_word_77 >> 16;
                    word0_77 = (unsigned int)packed_word_77 & 65535;
                    exp_idx[59] = idx_ii_76;
                    int local_idx_77 = idx_ii_76 - local_experts_start_idx;
                    int extent_77 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_77 = (1 << local_experts_stride_log2) - 1;
                    int is_local_77 = ((local_idx_77 >= 0 && local_idx_77 < extent_77 && (local_idx_77 & stride_mask_77) == 0) ? 1 : 0);
                    int is_local_ii_77 = is_local_77;
                    exp_off[59] = 0;
                    if (is_local_ii_77 != 0) {
                        uint32_t _shared_atomic_old_119;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_119) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_76)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[59] = (int)_shared_atomic_old_119;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_77 << 16);
                }
            }
        }
    }
    if (done == 0) {
        if (expanded_size >= 64 * grid_threads) {
            expanded = grid_tid + 60 * grid_threads;
            int idx_ii_77 = 0;
            unsigned int word0_78 = 0;
            unsigned int word1_78 = 0;
            unsigned int remote_78 = 0;
            int src_rank_78 = 0;
            int src_slot_78 = 0;
            int packed_word_78 = 0;
            packed_word_78 = topk_packed[expanded];
            idx_ii_77 = packed_word_78 >> 16;
            word0_78 = (unsigned int)packed_word_78 & 65535;
            exp_idx[60] = idx_ii_77;
            int local_idx_78 = idx_ii_77 - local_experts_start_idx;
            int extent_78 = num_local_experts << local_experts_stride_log2;
            int stride_mask_78 = (1 << local_experts_stride_log2) - 1;
            int is_local_78 = ((local_idx_78 >= 0 && local_idx_78 < extent_78 && (local_idx_78 & stride_mask_78) == 0) ? 1 : 0);
            int is_local_ii_78 = is_local_78;
            exp_off[60] = 0;
            if (is_local_ii_78 != 0) {
                uint32_t _shared_atomic_old_120;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_120) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_77)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[60] = (int)_shared_atomic_old_120;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_78 << 16);
            expanded = grid_tid + 61 * grid_threads;
            int idx_ii_0_15 = 0;
            unsigned int word0_1_15 = 0;
            unsigned int word1_2_15 = 0;
            unsigned int remote_3_15 = 0;
            int src_rank_4_15 = 0;
            int src_slot_5_15 = 0;
            int packed_word_6_15 = 0;
            packed_word_6_15 = topk_packed[expanded];
            idx_ii_0_15 = packed_word_6_15 >> 16;
            word0_1_15 = (unsigned int)packed_word_6_15 & 65535;
            exp_idx[61] = idx_ii_0_15;
            int local_idx_7_15 = idx_ii_0_15 - local_experts_start_idx;
            int extent_8_15 = num_local_experts << local_experts_stride_log2;
            int stride_mask_9_15 = (1 << local_experts_stride_log2) - 1;
            int is_local_10_15 = ((local_idx_7_15 >= 0 && local_idx_7_15 < extent_8_15 && (local_idx_7_15 & stride_mask_9_15) == 0) ? 1 : 0);
            int is_local_ii_11_15 = is_local_10_15;
            exp_off[61] = 0;
            if (is_local_ii_11_15 != 0) {
                uint32_t _shared_atomic_old_121;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_121) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_0_15)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[61] = (int)_shared_atomic_old_121;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_1_15 << 16);
            expanded = grid_tid + 62 * grid_threads;
            int idx_ii_12_15 = 0;
            unsigned int word0_13_15 = 0;
            unsigned int word1_14_15 = 0;
            unsigned int remote_15_15 = 0;
            int src_rank_16_15 = 0;
            int src_slot_17_15 = 0;
            int packed_word_18_15 = 0;
            packed_word_18_15 = topk_packed[expanded];
            idx_ii_12_15 = packed_word_18_15 >> 16;
            word0_13_15 = (unsigned int)packed_word_18_15 & 65535;
            exp_idx[62] = idx_ii_12_15;
            int local_idx_19_15 = idx_ii_12_15 - local_experts_start_idx;
            int extent_20_15 = num_local_experts << local_experts_stride_log2;
            int stride_mask_21_15 = (1 << local_experts_stride_log2) - 1;
            int is_local_22_15 = ((local_idx_19_15 >= 0 && local_idx_19_15 < extent_20_15 && (local_idx_19_15 & stride_mask_21_15) == 0) ? 1 : 0);
            int is_local_ii_23_15 = is_local_22_15;
            exp_off[62] = 0;
            if (is_local_ii_23_15 != 0) {
                uint32_t _shared_atomic_old_122;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_122) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_12_15)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[62] = (int)_shared_atomic_old_122;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_13_15 << 16);
            expanded = grid_tid + 63 * grid_threads;
            int idx_ii_24_15 = 0;
            unsigned int word0_25_15 = 0;
            unsigned int word1_26_15 = 0;
            unsigned int remote_27_15 = 0;
            int src_rank_28_15 = 0;
            int src_slot_29_15 = 0;
            int packed_word_30_15 = 0;
            packed_word_30_15 = topk_packed[expanded];
            idx_ii_24_15 = packed_word_30_15 >> 16;
            word0_25_15 = (unsigned int)packed_word_30_15 & 65535;
            exp_idx[63] = idx_ii_24_15;
            int local_idx_31_15 = idx_ii_24_15 - local_experts_start_idx;
            int extent_32_15 = num_local_experts << local_experts_stride_log2;
            int stride_mask_33_15 = (1 << local_experts_stride_log2) - 1;
            int is_local_34_15 = ((local_idx_31_15 >= 0 && local_idx_31_15 < extent_32_15 && (local_idx_31_15 & stride_mask_33_15) == 0) ? 1 : 0);
            int is_local_ii_35_15 = is_local_34_15;
            exp_off[63] = 0;
            if (is_local_ii_35_15 != 0) {
                uint32_t _shared_atomic_old_123;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_123) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_24_15)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[63] = (int)_shared_atomic_old_123;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_25_15 << 16);
        } else {
            if (done == 0) {
                expanded = grid_tid + 60 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_78 = 0;
                    unsigned int word0_79 = 0;
                    unsigned int word1_79 = 0;
                    unsigned int remote_79 = 0;
                    int src_rank_79 = 0;
                    int src_slot_79 = 0;
                    int packed_word_79 = 0;
                    packed_word_79 = topk_packed[expanded];
                    idx_ii_78 = packed_word_79 >> 16;
                    word0_79 = (unsigned int)packed_word_79 & 65535;
                    exp_idx[60] = idx_ii_78;
                    int local_idx_79 = idx_ii_78 - local_experts_start_idx;
                    int extent_79 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_79 = (1 << local_experts_stride_log2) - 1;
                    int is_local_79 = ((local_idx_79 >= 0 && local_idx_79 < extent_79 && (local_idx_79 & stride_mask_79) == 0) ? 1 : 0);
                    int is_local_ii_79 = is_local_79;
                    exp_off[60] = 0;
                    if (is_local_ii_79 != 0) {
                        uint32_t _shared_atomic_old_124;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_124) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_78)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[60] = (int)_shared_atomic_old_124;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_79 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 61 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_79 = 0;
                    unsigned int word0_80 = 0;
                    unsigned int word1_80 = 0;
                    unsigned int remote_80 = 0;
                    int src_rank_80 = 0;
                    int src_slot_80 = 0;
                    int packed_word_80 = 0;
                    packed_word_80 = topk_packed[expanded];
                    idx_ii_79 = packed_word_80 >> 16;
                    word0_80 = (unsigned int)packed_word_80 & 65535;
                    exp_idx[61] = idx_ii_79;
                    int local_idx_80 = idx_ii_79 - local_experts_start_idx;
                    int extent_80 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_80 = (1 << local_experts_stride_log2) - 1;
                    int is_local_80 = ((local_idx_80 >= 0 && local_idx_80 < extent_80 && (local_idx_80 & stride_mask_80) == 0) ? 1 : 0);
                    int is_local_ii_80 = is_local_80;
                    exp_off[61] = 0;
                    if (is_local_ii_80 != 0) {
                        uint32_t _shared_atomic_old_125;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_125) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_79)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[61] = (int)_shared_atomic_old_125;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_80 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 62 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_80 = 0;
                    unsigned int word0_81 = 0;
                    unsigned int word1_81 = 0;
                    unsigned int remote_81 = 0;
                    int src_rank_81 = 0;
                    int src_slot_81 = 0;
                    int packed_word_81 = 0;
                    packed_word_81 = topk_packed[expanded];
                    idx_ii_80 = packed_word_81 >> 16;
                    word0_81 = (unsigned int)packed_word_81 & 65535;
                    exp_idx[62] = idx_ii_80;
                    int local_idx_81 = idx_ii_80 - local_experts_start_idx;
                    int extent_81 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_81 = (1 << local_experts_stride_log2) - 1;
                    int is_local_81 = ((local_idx_81 >= 0 && local_idx_81 < extent_81 && (local_idx_81 & stride_mask_81) == 0) ? 1 : 0);
                    int is_local_ii_81 = is_local_81;
                    exp_off[62] = 0;
                    if (is_local_ii_81 != 0) {
                        uint32_t _shared_atomic_old_126;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_126) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_80)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[62] = (int)_shared_atomic_old_126;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_81 << 16);
                }
            }
            if (done == 0) {
                expanded = grid_tid + 63 * grid_threads;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_81 = 0;
                    unsigned int word0_82 = 0;
                    unsigned int word1_82 = 0;
                    unsigned int remote_82 = 0;
                    int src_rank_82 = 0;
                    int src_slot_82 = 0;
                    int packed_word_82 = 0;
                    packed_word_82 = topk_packed[expanded];
                    idx_ii_81 = packed_word_82 >> 16;
                    word0_82 = (unsigned int)packed_word_82 & 65535;
                    exp_idx[63] = idx_ii_81;
                    int local_idx_82 = idx_ii_81 - local_experts_start_idx;
                    int extent_82 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_82 = (1 << local_experts_stride_log2) - 1;
                    int is_local_82 = ((local_idx_82 >= 0 && local_idx_82 < extent_82 && (local_idx_82 & stride_mask_82) == 0) ? 1 : 0);
                    int is_local_ii_82 = is_local_82;
                    exp_off[63] = 0;
                    if (is_local_ii_82 != 0) {
                        uint32_t _shared_atomic_old_127;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_127) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_81)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[63] = (int)_shared_atomic_old_127;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_82 << 16);
                }
            }
        }
    }
    __syncthreads();
    int local_expert_count = (int)smem_expert_count[tid_0];
    int block_expert_offset = 0;
    if (tid_0 < num_experts) {
        int _atomic_old_0 = atomicAdd(&expert_counts[tid_0], local_expert_count);
        block_expert_offset = _atomic_old_0;
    }
    cooperative_groups::this_grid().sync();
    int count = 0;
    if (tid_0 < num_experts) {
        count = expert_counts[tid_0];
    }
    int num_cta = (count + tile_tokens_dim - 1) / tile_tokens_dim;
    if (padding_log2 > 0) {
        num_cta = count + (1 << padding_log2) - 1 >> padding_log2;
    }
    int num_cta_2 = num_cta;
    int inclusive = num_cta_2;
    int lane_s = (int)lane;
    int peer = 0;
    int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, inclusive, 1, 32);
    peer = _shfl_up_0;
    if (lane_s >= 1) {
        inclusive = inclusive + peer;
    }
    int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, inclusive, 2, 32);
    peer = _shfl_up_1;
    if (lane_s >= 2) {
        inclusive = inclusive + peer;
    }
    int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, inclusive, 4, 32);
    peer = _shfl_up_2;
    if (lane_s >= 4) {
        inclusive = inclusive + peer;
    }
    int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, inclusive, 8, 32);
    peer = _shfl_up_3;
    if (lane_s >= 8) {
        inclusive = inclusive + peer;
    }
    int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, inclusive, 16, 32);
    peer = _shfl_up_4;
    if (lane_s >= 16) {
        inclusive = inclusive + peer;
    }
    if (lane_s == 31) {
        warp_totals[warp] = inclusive;
    }
    __syncthreads();
    int warp_prefix = 0;
    int block_total = 0;
    int wt = 0;
    wt = warp_totals[0];
    if (warp > 0) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[1];
    if (warp > 1) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[2];
    if (warp > 2) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[3];
    if (warp > 3) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    int exclusive = inclusive - num_cta_2 + warp_prefix;
    __syncthreads();
    int local_idx_83 = tid_0 - local_experts_start_idx >> local_experts_stride_log2;
    int mn_limit1 = 0;
    int mn_limit2 = 0;
    int mn_limit = 0;
    for (int cta = bid_1; cta < num_cta_2; cta += nbids) {
        cta_idx_xy_to_batch_idx[exclusive + cta] = local_idx_83;
        int scaled = (exclusive + cta + 1) * tile_tokens_dim;
        if (padding_log2 > 0) {
            scaled = exclusive + cta + 1 << padding_log2;
        }
        mn_limit1 = scaled;
        int scaled_0 = exclusive * tile_tokens_dim;
        if (padding_log2 > 0) {
            scaled_0 = exclusive << padding_log2;
        }
        mn_limit2 = scaled_0 + count;
        int _min_0 = ((mn_limit1) < (mn_limit2) ? (mn_limit1) : (mn_limit2));
        mn_limit = _min_0;
        cta_idx_xy_to_mn_limit[exclusive + cta] = mn_limit;
        int _min_1 = ((mn_limit1) < (mn_limit + (-mn_limit & 3)) ? (mn_limit1) : (mn_limit + (-mn_limit & 3)));
        int head_end = _min_1;
        for (int head = mn_limit; head < head_end; head++) {
            permuted_idx_to_token_idx[head] = -1;
        }
        int _max_0 = ((mn_limit1 - head_end) > (0) ? (mn_limit1 - head_end) : (0));
        int body_len = _max_0;
        int tail_begin = head_end + (body_len >> 2 << 2);
        for (int body = head_end; body < tail_begin; body += 4) {
            {
                int4 _iv4 = make_int4(neg_four[0 + 0], neg_four[0 + 1], neg_four[0 + 2], neg_four[0 + 3]);
                *reinterpret_cast<int4*>(permuted_idx_to_token_idx + body) = _iv4;
            }
        }
        for (int tail = tail_begin; tail < mn_limit1; tail++) {
            permuted_idx_to_token_idx[tail] = -1;
        }
    }
    int scaled_1 = exclusive * tile_tokens_dim;
    if (padding_log2 > 0) {
        scaled_1 = exclusive << padding_log2;
    }
    int expert_rows = scaled_1;
    if (bid_1 == 0) {
        if (warp == 3) {
            if (elect_sync()) {
                int scaled_0_1 = block_total * tile_tokens_dim;
                if (padding_log2 > 0) {
                    scaled_0_1 = block_total << padding_log2;
                }
                int padded_rows = scaled_0_1;
                permuted_idx_size[0] = padded_rows;
                num_non_exiting_ctas[0] = block_total;
            }
        }
    }
    smem_expert_offset[tid_0] = expert_rows + block_expert_offset;
    __syncthreads();
    int done_3 = 0;
    int expanded_j = 0;
    int idx_j = 0;
    int is_local_j = 0;
    int token_j = 0;
    int permuted_j = -1;
    int neg_one = -1;
    int slot_base = 0;
    if (done_3 == 0) {
        expanded_j = grid_tid;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[0];
            int local_idx_0 = idx_j - local_experts_start_idx;
            int extent_83 = num_local_experts << local_experts_stride_log2;
            int stride_mask_83 = (1 << local_experts_stride_log2) - 1;
            int is_local_83 = ((local_idx_0 >= 0 && local_idx_0 < extent_83 && (local_idx_0 & stride_mask_83) == 0) ? 1 : 0);
            is_local_j = is_local_83;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[0] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[1];
            int local_idx_0_1 = idx_j - local_experts_start_idx;
            int extent_84 = num_local_experts << local_experts_stride_log2;
            int stride_mask_84 = (1 << local_experts_stride_log2) - 1;
            int is_local_84 = ((local_idx_0_1 >= 0 && local_idx_0_1 < extent_84 && (local_idx_0_1 & stride_mask_84) == 0) ? 1 : 0);
            is_local_j = is_local_84;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[1] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 2 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[2];
            int local_idx_0_2 = idx_j - local_experts_start_idx;
            int extent_85 = num_local_experts << local_experts_stride_log2;
            int stride_mask_85 = (1 << local_experts_stride_log2) - 1;
            int is_local_85 = ((local_idx_0_2 >= 0 && local_idx_0_2 < extent_85 && (local_idx_0_2 & stride_mask_85) == 0) ? 1 : 0);
            is_local_j = is_local_85;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[2] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 3 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[3];
            int local_idx_0_3 = idx_j - local_experts_start_idx;
            int extent_86 = num_local_experts << local_experts_stride_log2;
            int stride_mask_86 = (1 << local_experts_stride_log2) - 1;
            int is_local_86 = ((local_idx_0_3 >= 0 && local_idx_0_3 < extent_86 && (local_idx_0_3 & stride_mask_86) == 0) ? 1 : 0);
            is_local_j = is_local_86;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[3] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 4 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[4];
            int local_idx_0_4 = idx_j - local_experts_start_idx;
            int extent_87 = num_local_experts << local_experts_stride_log2;
            int stride_mask_87 = (1 << local_experts_stride_log2) - 1;
            int is_local_87 = ((local_idx_0_4 >= 0 && local_idx_0_4 < extent_87 && (local_idx_0_4 & stride_mask_87) == 0) ? 1 : 0);
            is_local_j = is_local_87;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[4] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 5 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[5];
            int local_idx_0_5 = idx_j - local_experts_start_idx;
            int extent_88 = num_local_experts << local_experts_stride_log2;
            int stride_mask_88 = (1 << local_experts_stride_log2) - 1;
            int is_local_88 = ((local_idx_0_5 >= 0 && local_idx_0_5 < extent_88 && (local_idx_0_5 & stride_mask_88) == 0) ? 1 : 0);
            is_local_j = is_local_88;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[5] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 6 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[6];
            int local_idx_0_6 = idx_j - local_experts_start_idx;
            int extent_89 = num_local_experts << local_experts_stride_log2;
            int stride_mask_89 = (1 << local_experts_stride_log2) - 1;
            int is_local_89 = ((local_idx_0_6 >= 0 && local_idx_0_6 < extent_89 && (local_idx_0_6 & stride_mask_89) == 0) ? 1 : 0);
            is_local_j = is_local_89;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[6] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 7 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[7];
            int local_idx_0_7 = idx_j - local_experts_start_idx;
            int extent_90 = num_local_experts << local_experts_stride_log2;
            int stride_mask_90 = (1 << local_experts_stride_log2) - 1;
            int is_local_90 = ((local_idx_0_7 >= 0 && local_idx_0_7 < extent_90 && (local_idx_0_7 & stride_mask_90) == 0) ? 1 : 0);
            is_local_j = is_local_90;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[7] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 8 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[8];
            int local_idx_0_8 = idx_j - local_experts_start_idx;
            int extent_91 = num_local_experts << local_experts_stride_log2;
            int stride_mask_91 = (1 << local_experts_stride_log2) - 1;
            int is_local_91 = ((local_idx_0_8 >= 0 && local_idx_0_8 < extent_91 && (local_idx_0_8 & stride_mask_91) == 0) ? 1 : 0);
            is_local_j = is_local_91;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[8] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 9 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[9];
            int local_idx_0_9 = idx_j - local_experts_start_idx;
            int extent_92 = num_local_experts << local_experts_stride_log2;
            int stride_mask_92 = (1 << local_experts_stride_log2) - 1;
            int is_local_92 = ((local_idx_0_9 >= 0 && local_idx_0_9 < extent_92 && (local_idx_0_9 & stride_mask_92) == 0) ? 1 : 0);
            is_local_j = is_local_92;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[9] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 10 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[10];
            int local_idx_0_10 = idx_j - local_experts_start_idx;
            int extent_93 = num_local_experts << local_experts_stride_log2;
            int stride_mask_93 = (1 << local_experts_stride_log2) - 1;
            int is_local_93 = ((local_idx_0_10 >= 0 && local_idx_0_10 < extent_93 && (local_idx_0_10 & stride_mask_93) == 0) ? 1 : 0);
            is_local_j = is_local_93;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[10] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 11 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[11];
            int local_idx_0_11 = idx_j - local_experts_start_idx;
            int extent_94 = num_local_experts << local_experts_stride_log2;
            int stride_mask_94 = (1 << local_experts_stride_log2) - 1;
            int is_local_94 = ((local_idx_0_11 >= 0 && local_idx_0_11 < extent_94 && (local_idx_0_11 & stride_mask_94) == 0) ? 1 : 0);
            is_local_j = is_local_94;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[11] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 12 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[12];
            int local_idx_0_12 = idx_j - local_experts_start_idx;
            int extent_95 = num_local_experts << local_experts_stride_log2;
            int stride_mask_95 = (1 << local_experts_stride_log2) - 1;
            int is_local_95 = ((local_idx_0_12 >= 0 && local_idx_0_12 < extent_95 && (local_idx_0_12 & stride_mask_95) == 0) ? 1 : 0);
            is_local_j = is_local_95;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[12] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 13 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[13];
            int local_idx_0_13 = idx_j - local_experts_start_idx;
            int extent_96 = num_local_experts << local_experts_stride_log2;
            int stride_mask_96 = (1 << local_experts_stride_log2) - 1;
            int is_local_96 = ((local_idx_0_13 >= 0 && local_idx_0_13 < extent_96 && (local_idx_0_13 & stride_mask_96) == 0) ? 1 : 0);
            is_local_j = is_local_96;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[13] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 14 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[14];
            int local_idx_0_14 = idx_j - local_experts_start_idx;
            int extent_97 = num_local_experts << local_experts_stride_log2;
            int stride_mask_97 = (1 << local_experts_stride_log2) - 1;
            int is_local_97 = ((local_idx_0_14 >= 0 && local_idx_0_14 < extent_97 && (local_idx_0_14 & stride_mask_97) == 0) ? 1 : 0);
            is_local_j = is_local_97;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[14] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 15 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[15];
            int local_idx_0_15 = idx_j - local_experts_start_idx;
            int extent_98 = num_local_experts << local_experts_stride_log2;
            int stride_mask_98 = (1 << local_experts_stride_log2) - 1;
            int is_local_98 = ((local_idx_0_15 >= 0 && local_idx_0_15 < extent_98 && (local_idx_0_15 & stride_mask_98) == 0) ? 1 : 0);
            is_local_j = is_local_98;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[15] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 16 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[16];
            int local_idx_0_16 = idx_j - local_experts_start_idx;
            int extent_99 = num_local_experts << local_experts_stride_log2;
            int stride_mask_99 = (1 << local_experts_stride_log2) - 1;
            int is_local_99 = ((local_idx_0_16 >= 0 && local_idx_0_16 < extent_99 && (local_idx_0_16 & stride_mask_99) == 0) ? 1 : 0);
            is_local_j = is_local_99;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[16] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 17 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[17];
            int local_idx_0_17 = idx_j - local_experts_start_idx;
            int extent_100 = num_local_experts << local_experts_stride_log2;
            int stride_mask_100 = (1 << local_experts_stride_log2) - 1;
            int is_local_100 = ((local_idx_0_17 >= 0 && local_idx_0_17 < extent_100 && (local_idx_0_17 & stride_mask_100) == 0) ? 1 : 0);
            is_local_j = is_local_100;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[17] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 18 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[18];
            int local_idx_0_18 = idx_j - local_experts_start_idx;
            int extent_101 = num_local_experts << local_experts_stride_log2;
            int stride_mask_101 = (1 << local_experts_stride_log2) - 1;
            int is_local_101 = ((local_idx_0_18 >= 0 && local_idx_0_18 < extent_101 && (local_idx_0_18 & stride_mask_101) == 0) ? 1 : 0);
            is_local_j = is_local_101;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[18] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 19 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[19];
            int local_idx_0_19 = idx_j - local_experts_start_idx;
            int extent_102 = num_local_experts << local_experts_stride_log2;
            int stride_mask_102 = (1 << local_experts_stride_log2) - 1;
            int is_local_102 = ((local_idx_0_19 >= 0 && local_idx_0_19 < extent_102 && (local_idx_0_19 & stride_mask_102) == 0) ? 1 : 0);
            is_local_j = is_local_102;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[19] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 20 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[20];
            int local_idx_0_20 = idx_j - local_experts_start_idx;
            int extent_103 = num_local_experts << local_experts_stride_log2;
            int stride_mask_103 = (1 << local_experts_stride_log2) - 1;
            int is_local_103 = ((local_idx_0_20 >= 0 && local_idx_0_20 < extent_103 && (local_idx_0_20 & stride_mask_103) == 0) ? 1 : 0);
            is_local_j = is_local_103;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[20] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 21 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[21];
            int local_idx_0_21 = idx_j - local_experts_start_idx;
            int extent_104 = num_local_experts << local_experts_stride_log2;
            int stride_mask_104 = (1 << local_experts_stride_log2) - 1;
            int is_local_104 = ((local_idx_0_21 >= 0 && local_idx_0_21 < extent_104 && (local_idx_0_21 & stride_mask_104) == 0) ? 1 : 0);
            is_local_j = is_local_104;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[21] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 22 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[22];
            int local_idx_0_22 = idx_j - local_experts_start_idx;
            int extent_105 = num_local_experts << local_experts_stride_log2;
            int stride_mask_105 = (1 << local_experts_stride_log2) - 1;
            int is_local_105 = ((local_idx_0_22 >= 0 && local_idx_0_22 < extent_105 && (local_idx_0_22 & stride_mask_105) == 0) ? 1 : 0);
            is_local_j = is_local_105;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[22] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 23 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[23];
            int local_idx_0_23 = idx_j - local_experts_start_idx;
            int extent_106 = num_local_experts << local_experts_stride_log2;
            int stride_mask_106 = (1 << local_experts_stride_log2) - 1;
            int is_local_106 = ((local_idx_0_23 >= 0 && local_idx_0_23 < extent_106 && (local_idx_0_23 & stride_mask_106) == 0) ? 1 : 0);
            is_local_j = is_local_106;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[23] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 24 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[24];
            int local_idx_0_24 = idx_j - local_experts_start_idx;
            int extent_107 = num_local_experts << local_experts_stride_log2;
            int stride_mask_107 = (1 << local_experts_stride_log2) - 1;
            int is_local_107 = ((local_idx_0_24 >= 0 && local_idx_0_24 < extent_107 && (local_idx_0_24 & stride_mask_107) == 0) ? 1 : 0);
            is_local_j = is_local_107;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[24] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 25 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[25];
            int local_idx_0_25 = idx_j - local_experts_start_idx;
            int extent_108 = num_local_experts << local_experts_stride_log2;
            int stride_mask_108 = (1 << local_experts_stride_log2) - 1;
            int is_local_108 = ((local_idx_0_25 >= 0 && local_idx_0_25 < extent_108 && (local_idx_0_25 & stride_mask_108) == 0) ? 1 : 0);
            is_local_j = is_local_108;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[25] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 26 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[26];
            int local_idx_0_26 = idx_j - local_experts_start_idx;
            int extent_109 = num_local_experts << local_experts_stride_log2;
            int stride_mask_109 = (1 << local_experts_stride_log2) - 1;
            int is_local_109 = ((local_idx_0_26 >= 0 && local_idx_0_26 < extent_109 && (local_idx_0_26 & stride_mask_109) == 0) ? 1 : 0);
            is_local_j = is_local_109;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[26] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 27 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[27];
            int local_idx_0_27 = idx_j - local_experts_start_idx;
            int extent_110 = num_local_experts << local_experts_stride_log2;
            int stride_mask_110 = (1 << local_experts_stride_log2) - 1;
            int is_local_110 = ((local_idx_0_27 >= 0 && local_idx_0_27 < extent_110 && (local_idx_0_27 & stride_mask_110) == 0) ? 1 : 0);
            is_local_j = is_local_110;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[27] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 28 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[28];
            int local_idx_0_28 = idx_j - local_experts_start_idx;
            int extent_111 = num_local_experts << local_experts_stride_log2;
            int stride_mask_111 = (1 << local_experts_stride_log2) - 1;
            int is_local_111 = ((local_idx_0_28 >= 0 && local_idx_0_28 < extent_111 && (local_idx_0_28 & stride_mask_111) == 0) ? 1 : 0);
            is_local_j = is_local_111;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[28] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 29 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[29];
            int local_idx_0_29 = idx_j - local_experts_start_idx;
            int extent_112 = num_local_experts << local_experts_stride_log2;
            int stride_mask_112 = (1 << local_experts_stride_log2) - 1;
            int is_local_112 = ((local_idx_0_29 >= 0 && local_idx_0_29 < extent_112 && (local_idx_0_29 & stride_mask_112) == 0) ? 1 : 0);
            is_local_j = is_local_112;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[29] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 30 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[30];
            int local_idx_0_30 = idx_j - local_experts_start_idx;
            int extent_113 = num_local_experts << local_experts_stride_log2;
            int stride_mask_113 = (1 << local_experts_stride_log2) - 1;
            int is_local_113 = ((local_idx_0_30 >= 0 && local_idx_0_30 < extent_113 && (local_idx_0_30 & stride_mask_113) == 0) ? 1 : 0);
            is_local_j = is_local_113;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[30] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 31 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[31];
            int local_idx_0_31 = idx_j - local_experts_start_idx;
            int extent_114 = num_local_experts << local_experts_stride_log2;
            int stride_mask_114 = (1 << local_experts_stride_log2) - 1;
            int is_local_114 = ((local_idx_0_31 >= 0 && local_idx_0_31 < extent_114 && (local_idx_0_31 & stride_mask_114) == 0) ? 1 : 0);
            is_local_j = is_local_114;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[31] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 32 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[32];
            int local_idx_0_32 = idx_j - local_experts_start_idx;
            int extent_115 = num_local_experts << local_experts_stride_log2;
            int stride_mask_115 = (1 << local_experts_stride_log2) - 1;
            int is_local_115 = ((local_idx_0_32 >= 0 && local_idx_0_32 < extent_115 && (local_idx_0_32 & stride_mask_115) == 0) ? 1 : 0);
            is_local_j = is_local_115;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[32] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 33 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[33];
            int local_idx_0_33 = idx_j - local_experts_start_idx;
            int extent_116 = num_local_experts << local_experts_stride_log2;
            int stride_mask_116 = (1 << local_experts_stride_log2) - 1;
            int is_local_116 = ((local_idx_0_33 >= 0 && local_idx_0_33 < extent_116 && (local_idx_0_33 & stride_mask_116) == 0) ? 1 : 0);
            is_local_j = is_local_116;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[33] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 34 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[34];
            int local_idx_0_34 = idx_j - local_experts_start_idx;
            int extent_117 = num_local_experts << local_experts_stride_log2;
            int stride_mask_117 = (1 << local_experts_stride_log2) - 1;
            int is_local_117 = ((local_idx_0_34 >= 0 && local_idx_0_34 < extent_117 && (local_idx_0_34 & stride_mask_117) == 0) ? 1 : 0);
            is_local_j = is_local_117;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[34] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 35 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[35];
            int local_idx_0_35 = idx_j - local_experts_start_idx;
            int extent_118 = num_local_experts << local_experts_stride_log2;
            int stride_mask_118 = (1 << local_experts_stride_log2) - 1;
            int is_local_118 = ((local_idx_0_35 >= 0 && local_idx_0_35 < extent_118 && (local_idx_0_35 & stride_mask_118) == 0) ? 1 : 0);
            is_local_j = is_local_118;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[35] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 36 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[36];
            int local_idx_0_36 = idx_j - local_experts_start_idx;
            int extent_119 = num_local_experts << local_experts_stride_log2;
            int stride_mask_119 = (1 << local_experts_stride_log2) - 1;
            int is_local_119 = ((local_idx_0_36 >= 0 && local_idx_0_36 < extent_119 && (local_idx_0_36 & stride_mask_119) == 0) ? 1 : 0);
            is_local_j = is_local_119;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[36] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 37 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[37];
            int local_idx_0_37 = idx_j - local_experts_start_idx;
            int extent_120 = num_local_experts << local_experts_stride_log2;
            int stride_mask_120 = (1 << local_experts_stride_log2) - 1;
            int is_local_120 = ((local_idx_0_37 >= 0 && local_idx_0_37 < extent_120 && (local_idx_0_37 & stride_mask_120) == 0) ? 1 : 0);
            is_local_j = is_local_120;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[37] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 38 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[38];
            int local_idx_0_38 = idx_j - local_experts_start_idx;
            int extent_121 = num_local_experts << local_experts_stride_log2;
            int stride_mask_121 = (1 << local_experts_stride_log2) - 1;
            int is_local_121 = ((local_idx_0_38 >= 0 && local_idx_0_38 < extent_121 && (local_idx_0_38 & stride_mask_121) == 0) ? 1 : 0);
            is_local_j = is_local_121;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[38] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 39 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[39];
            int local_idx_0_39 = idx_j - local_experts_start_idx;
            int extent_122 = num_local_experts << local_experts_stride_log2;
            int stride_mask_122 = (1 << local_experts_stride_log2) - 1;
            int is_local_122 = ((local_idx_0_39 >= 0 && local_idx_0_39 < extent_122 && (local_idx_0_39 & stride_mask_122) == 0) ? 1 : 0);
            is_local_j = is_local_122;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[39] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 40 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[40];
            int local_idx_0_40 = idx_j - local_experts_start_idx;
            int extent_123 = num_local_experts << local_experts_stride_log2;
            int stride_mask_123 = (1 << local_experts_stride_log2) - 1;
            int is_local_123 = ((local_idx_0_40 >= 0 && local_idx_0_40 < extent_123 && (local_idx_0_40 & stride_mask_123) == 0) ? 1 : 0);
            is_local_j = is_local_123;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[40] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 41 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[41];
            int local_idx_0_41 = idx_j - local_experts_start_idx;
            int extent_124 = num_local_experts << local_experts_stride_log2;
            int stride_mask_124 = (1 << local_experts_stride_log2) - 1;
            int is_local_124 = ((local_idx_0_41 >= 0 && local_idx_0_41 < extent_124 && (local_idx_0_41 & stride_mask_124) == 0) ? 1 : 0);
            is_local_j = is_local_124;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[41] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 42 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[42];
            int local_idx_0_42 = idx_j - local_experts_start_idx;
            int extent_125 = num_local_experts << local_experts_stride_log2;
            int stride_mask_125 = (1 << local_experts_stride_log2) - 1;
            int is_local_125 = ((local_idx_0_42 >= 0 && local_idx_0_42 < extent_125 && (local_idx_0_42 & stride_mask_125) == 0) ? 1 : 0);
            is_local_j = is_local_125;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[42] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 43 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[43];
            int local_idx_0_43 = idx_j - local_experts_start_idx;
            int extent_126 = num_local_experts << local_experts_stride_log2;
            int stride_mask_126 = (1 << local_experts_stride_log2) - 1;
            int is_local_126 = ((local_idx_0_43 >= 0 && local_idx_0_43 < extent_126 && (local_idx_0_43 & stride_mask_126) == 0) ? 1 : 0);
            is_local_j = is_local_126;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[43] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 44 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[44];
            int local_idx_0_44 = idx_j - local_experts_start_idx;
            int extent_127 = num_local_experts << local_experts_stride_log2;
            int stride_mask_127 = (1 << local_experts_stride_log2) - 1;
            int is_local_127 = ((local_idx_0_44 >= 0 && local_idx_0_44 < extent_127 && (local_idx_0_44 & stride_mask_127) == 0) ? 1 : 0);
            is_local_j = is_local_127;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[44] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 45 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[45];
            int local_idx_0_45 = idx_j - local_experts_start_idx;
            int extent_128 = num_local_experts << local_experts_stride_log2;
            int stride_mask_128 = (1 << local_experts_stride_log2) - 1;
            int is_local_128 = ((local_idx_0_45 >= 0 && local_idx_0_45 < extent_128 && (local_idx_0_45 & stride_mask_128) == 0) ? 1 : 0);
            is_local_j = is_local_128;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[45] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 46 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[46];
            int local_idx_0_46 = idx_j - local_experts_start_idx;
            int extent_129 = num_local_experts << local_experts_stride_log2;
            int stride_mask_129 = (1 << local_experts_stride_log2) - 1;
            int is_local_129 = ((local_idx_0_46 >= 0 && local_idx_0_46 < extent_129 && (local_idx_0_46 & stride_mask_129) == 0) ? 1 : 0);
            is_local_j = is_local_129;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[46] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 47 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[47];
            int local_idx_0_47 = idx_j - local_experts_start_idx;
            int extent_130 = num_local_experts << local_experts_stride_log2;
            int stride_mask_130 = (1 << local_experts_stride_log2) - 1;
            int is_local_130 = ((local_idx_0_47 >= 0 && local_idx_0_47 < extent_130 && (local_idx_0_47 & stride_mask_130) == 0) ? 1 : 0);
            is_local_j = is_local_130;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[47] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 48 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[48];
            int local_idx_0_48 = idx_j - local_experts_start_idx;
            int extent_131 = num_local_experts << local_experts_stride_log2;
            int stride_mask_131 = (1 << local_experts_stride_log2) - 1;
            int is_local_131 = ((local_idx_0_48 >= 0 && local_idx_0_48 < extent_131 && (local_idx_0_48 & stride_mask_131) == 0) ? 1 : 0);
            is_local_j = is_local_131;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[48] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 49 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[49];
            int local_idx_0_49 = idx_j - local_experts_start_idx;
            int extent_132 = num_local_experts << local_experts_stride_log2;
            int stride_mask_132 = (1 << local_experts_stride_log2) - 1;
            int is_local_132 = ((local_idx_0_49 >= 0 && local_idx_0_49 < extent_132 && (local_idx_0_49 & stride_mask_132) == 0) ? 1 : 0);
            is_local_j = is_local_132;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[49] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 50 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[50];
            int local_idx_0_50 = idx_j - local_experts_start_idx;
            int extent_133 = num_local_experts << local_experts_stride_log2;
            int stride_mask_133 = (1 << local_experts_stride_log2) - 1;
            int is_local_133 = ((local_idx_0_50 >= 0 && local_idx_0_50 < extent_133 && (local_idx_0_50 & stride_mask_133) == 0) ? 1 : 0);
            is_local_j = is_local_133;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[50] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 51 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[51];
            int local_idx_0_51 = idx_j - local_experts_start_idx;
            int extent_134 = num_local_experts << local_experts_stride_log2;
            int stride_mask_134 = (1 << local_experts_stride_log2) - 1;
            int is_local_134 = ((local_idx_0_51 >= 0 && local_idx_0_51 < extent_134 && (local_idx_0_51 & stride_mask_134) == 0) ? 1 : 0);
            is_local_j = is_local_134;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[51] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 52 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[52];
            int local_idx_0_52 = idx_j - local_experts_start_idx;
            int extent_135 = num_local_experts << local_experts_stride_log2;
            int stride_mask_135 = (1 << local_experts_stride_log2) - 1;
            int is_local_135 = ((local_idx_0_52 >= 0 && local_idx_0_52 < extent_135 && (local_idx_0_52 & stride_mask_135) == 0) ? 1 : 0);
            is_local_j = is_local_135;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[52] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 53 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[53];
            int local_idx_0_53 = idx_j - local_experts_start_idx;
            int extent_136 = num_local_experts << local_experts_stride_log2;
            int stride_mask_136 = (1 << local_experts_stride_log2) - 1;
            int is_local_136 = ((local_idx_0_53 >= 0 && local_idx_0_53 < extent_136 && (local_idx_0_53 & stride_mask_136) == 0) ? 1 : 0);
            is_local_j = is_local_136;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[53] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 54 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[54];
            int local_idx_0_54 = idx_j - local_experts_start_idx;
            int extent_137 = num_local_experts << local_experts_stride_log2;
            int stride_mask_137 = (1 << local_experts_stride_log2) - 1;
            int is_local_137 = ((local_idx_0_54 >= 0 && local_idx_0_54 < extent_137 && (local_idx_0_54 & stride_mask_137) == 0) ? 1 : 0);
            is_local_j = is_local_137;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[54] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 55 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[55];
            int local_idx_0_55 = idx_j - local_experts_start_idx;
            int extent_138 = num_local_experts << local_experts_stride_log2;
            int stride_mask_138 = (1 << local_experts_stride_log2) - 1;
            int is_local_138 = ((local_idx_0_55 >= 0 && local_idx_0_55 < extent_138 && (local_idx_0_55 & stride_mask_138) == 0) ? 1 : 0);
            is_local_j = is_local_138;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[55] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 56 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[56];
            int local_idx_0_56 = idx_j - local_experts_start_idx;
            int extent_139 = num_local_experts << local_experts_stride_log2;
            int stride_mask_139 = (1 << local_experts_stride_log2) - 1;
            int is_local_139 = ((local_idx_0_56 >= 0 && local_idx_0_56 < extent_139 && (local_idx_0_56 & stride_mask_139) == 0) ? 1 : 0);
            is_local_j = is_local_139;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[56] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 57 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[57];
            int local_idx_0_57 = idx_j - local_experts_start_idx;
            int extent_140 = num_local_experts << local_experts_stride_log2;
            int stride_mask_140 = (1 << local_experts_stride_log2) - 1;
            int is_local_140 = ((local_idx_0_57 >= 0 && local_idx_0_57 < extent_140 && (local_idx_0_57 & stride_mask_140) == 0) ? 1 : 0);
            is_local_j = is_local_140;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[57] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 58 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[58];
            int local_idx_0_58 = idx_j - local_experts_start_idx;
            int extent_141 = num_local_experts << local_experts_stride_log2;
            int stride_mask_141 = (1 << local_experts_stride_log2) - 1;
            int is_local_141 = ((local_idx_0_58 >= 0 && local_idx_0_58 < extent_141 && (local_idx_0_58 & stride_mask_141) == 0) ? 1 : 0);
            is_local_j = is_local_141;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[58] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 59 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[59];
            int local_idx_0_59 = idx_j - local_experts_start_idx;
            int extent_142 = num_local_experts << local_experts_stride_log2;
            int stride_mask_142 = (1 << local_experts_stride_log2) - 1;
            int is_local_142 = ((local_idx_0_59 >= 0 && local_idx_0_59 < extent_142 && (local_idx_0_59 & stride_mask_142) == 0) ? 1 : 0);
            is_local_j = is_local_142;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[59] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 60 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[60];
            int local_idx_0_60 = idx_j - local_experts_start_idx;
            int extent_143 = num_local_experts << local_experts_stride_log2;
            int stride_mask_143 = (1 << local_experts_stride_log2) - 1;
            int is_local_143 = ((local_idx_0_60 >= 0 && local_idx_0_60 < extent_143 && (local_idx_0_60 & stride_mask_143) == 0) ? 1 : 0);
            is_local_j = is_local_143;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[60] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 61 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[61];
            int local_idx_0_61 = idx_j - local_experts_start_idx;
            int extent_144 = num_local_experts << local_experts_stride_log2;
            int stride_mask_144 = (1 << local_experts_stride_log2) - 1;
            int is_local_144 = ((local_idx_0_61 >= 0 && local_idx_0_61 < extent_144 && (local_idx_0_61 & stride_mask_144) == 0) ? 1 : 0);
            is_local_j = is_local_144;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[61] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 62 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[62];
            int local_idx_0_62 = idx_j - local_experts_start_idx;
            int extent_145 = num_local_experts << local_experts_stride_log2;
            int stride_mask_145 = (1 << local_experts_stride_log2) - 1;
            int is_local_145 = ((local_idx_0_62 >= 0 && local_idx_0_62 < extent_145 && (local_idx_0_62 & stride_mask_145) == 0) ? 1 : 0);
            is_local_j = is_local_145;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[62] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_3 == 0) {
        expanded_j = grid_tid + 63 * grid_threads;
        if (expanded_j >= expanded_size) {
            done_3 = 1;
        } else {
            idx_j = exp_idx[63];
            int local_idx_0_63 = idx_j - local_experts_start_idx;
            int extent_146 = num_local_experts << local_experts_stride_log2;
            int stride_mask_146 = (1 << local_experts_stride_log2) - 1;
            int is_local_146 = ((local_idx_0_63 >= 0 && local_idx_0_63 < extent_146 && (local_idx_0_63 & stride_mask_146) == 0) ? 1 : 0);
            is_local_j = is_local_146;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[63] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
