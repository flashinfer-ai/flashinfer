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
#define SMEM_SMEM_KIDX_OFF 0
#define SMEM_SMEM_KIDX_STAGE_BYTES 640
#define SMEM_SMEM_KIDX_STRIDE 640
#define SMEM_SMEM_OFFSET16_OFF 640
#define SMEM_SMEM_OFFSET16_STAGE_BYTES 1280
#define SMEM_SMEM_OFFSET16_STRIDE 1280
#define SMEM_WARP_TOTALS1_OFF 1920
#define SMEM_WARP_TOTALS1_STAGE_BYTES 128
#define SMEM_WARP_TOTALS1_STRIDE 128
#define SMEM_WARP_TOTALS2_OFF 2048
#define SMEM_WARP_TOTALS2_STAGE_BYTES 128
#define SMEM_WARP_TOTALS2_STRIDE 128
#define SMEM_TOTAL 2176
#define THREADS 160

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(160) void
kernel_cake_stepfun_moe_5eed5f20e054f523d9b4(int* __restrict__ topk_ids, __nv_bfloat16* __restrict__ topk_weights, int* __restrict__ topk_packed, int* __restrict__ expert_counts, int* __restrict__ permuted_idx_size, int* __restrict__ expanded_idx_to_permuted_idx, int* __restrict__ permuted_idx_to_token_idx, int* __restrict__ cta_idx_xy_to_batch_idx, int* __restrict__ cta_idx_xy_to_mn_limit, int* __restrict__ num_non_exiting_ctas, int num_tokens, int num_experts, int top_k, int padding_log2, int tile_tokens_dim, int local_experts_start_idx, int local_experts_stride_log2, int num_local_experts)
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
    int8_t* smem_kidx = reinterpret_cast<int8_t*>(smem_raw + 0);
    const int smem_kidx_addr = smem + 0;
    uint16_t* smem_offset16 = reinterpret_cast<uint16_t*>(smem_raw + 640);
    const int smem_offset16_addr = smem + 640;
    int* warp_totals1 = reinterpret_cast<int*>(smem_raw + 1920);
    const int warp_totals1_addr = smem + 1920;
    int* warp_totals2 = reinterpret_cast<int*>(smem_raw + 2048);
    const int warp_totals2_addr = smem + 2048;

    // === Task calls (dependency order) ===
    int tid_0 = (int)tid;
    int lane_1 = (int)lane;
    int warp_2 = (int)warp;
    int neg_four[4];
    neg_four[0] = -1;
    neg_four[1] = -1;
    neg_four[2] = -1;
    neg_four[3] = -1;
    for (int slot_init = tid_0; slot_init < num_tokens * 128; slot_init += 160) {
        smem_kidx[slot_init] = (int8_t)-1;
    }
    __syncthreads();
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int expanded_p = 0;
    int expert_p = 0;
    for (int token = warp_2; token < num_tokens; token += 5) {
        if (lane_1 < top_k) {
            expanded_p = token * top_k + lane_1;
            expanded_idx_to_permuted_idx[expanded_p] = -1;
            expert_p = topk_ids[expanded_p];
            if (expert_p > -1 && expert_p < num_experts) {
                smem_kidx[token * 128 + expert_p] = (int8_t)lane_1;
            }
        }
    }
    __syncthreads();
    int expert = tid_0;
    int local_idx = expert - local_experts_start_idx;
    int extent = num_local_experts << local_experts_stride_log2;
    int stride_mask = (1 << local_experts_stride_log2) - 1;
    int is_local = ((local_idx >= 0 && local_idx < extent && (local_idx & stride_mask) == 0) ? 1 : 0);
    int is_local_3 = is_local;
    int acc_count = 0;
    int slot_e = 0;
    int kidx_e = 0;
    if (tid_0 < 128) {
        if (is_local_3 != 0) {
            slot_e = expert;
            for (int j = 0; j < num_tokens; j++) {
                kidx_e = (int)smem_kidx[slot_e];
                if (kidx_e >= 0) {
                    smem_offset16[slot_e] = (uint16_t)acc_count;
                    acc_count = acc_count + 1;
                }
                slot_e = slot_e + 128;
            }
        }
    }
    int num_cta = 0;
    int padded_count = 0;
    if (tid_0 < 128) {
        int num_cta_0 = (acc_count + tile_tokens_dim - 1) / tile_tokens_dim;
        if (padding_log2 > 0) {
            num_cta_0 = acc_count + (1 << padding_log2) - 1 >> padding_log2;
        }
        num_cta = num_cta_0;
        int scaled = num_cta * tile_tokens_dim;
        if (padding_log2 > 0) {
            scaled = num_cta << padding_log2;
        }
        padded_count = scaled;
    }
    int inclusive = num_cta;
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
        warp_totals1[warp] = inclusive;
    }
    __syncthreads();
    int warp_prefix = 0;
    int block_total = 0;
    int wt = 0;
    wt = warp_totals1[0];
    if (warp > 0) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals1[1];
    if (warp > 1) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals1[2];
    if (warp > 2) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals1[3];
    if (warp > 3) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals1[4];
    if (warp > 4) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    int exclusive = inclusive - num_cta + warp_prefix;
    __syncthreads();
    int inclusive_4 = padded_count;
    int lane_s_5 = (int)lane;
    int peer_6 = 0;
    int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, inclusive_4, 1, 32);
    peer_6 = _shfl_up_5;
    if (lane_s_5 >= 1) {
        inclusive_4 = inclusive_4 + peer_6;
    }
    int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, inclusive_4, 2, 32);
    peer_6 = _shfl_up_6;
    if (lane_s_5 >= 2) {
        inclusive_4 = inclusive_4 + peer_6;
    }
    int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, inclusive_4, 4, 32);
    peer_6 = _shfl_up_7;
    if (lane_s_5 >= 4) {
        inclusive_4 = inclusive_4 + peer_6;
    }
    int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, inclusive_4, 8, 32);
    peer_6 = _shfl_up_8;
    if (lane_s_5 >= 8) {
        inclusive_4 = inclusive_4 + peer_6;
    }
    int _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, inclusive_4, 16, 32);
    peer_6 = _shfl_up_9;
    if (lane_s_5 >= 16) {
        inclusive_4 = inclusive_4 + peer_6;
    }
    if (lane_s_5 == 31) {
        warp_totals2[warp] = inclusive_4;
    }
    __syncthreads();
    int warp_prefix_7 = 0;
    int block_total_8 = 0;
    int wt_9 = 0;
    wt_9 = warp_totals2[0];
    if (warp > 0) {
        warp_prefix_7 = warp_prefix_7 + wt_9;
    }
    block_total_8 = block_total_8 + wt_9;
    wt_9 = warp_totals2[1];
    if (warp > 1) {
        warp_prefix_7 = warp_prefix_7 + wt_9;
    }
    block_total_8 = block_total_8 + wt_9;
    wt_9 = warp_totals2[2];
    if (warp > 2) {
        warp_prefix_7 = warp_prefix_7 + wt_9;
    }
    block_total_8 = block_total_8 + wt_9;
    wt_9 = warp_totals2[3];
    if (warp > 3) {
        warp_prefix_7 = warp_prefix_7 + wt_9;
    }
    block_total_8 = block_total_8 + wt_9;
    wt_9 = warp_totals2[4];
    if (warp > 4) {
        warp_prefix_7 = warp_prefix_7 + wt_9;
    }
    block_total_8 = block_total_8 + wt_9;
    int exclusive_10 = inclusive_4 - padded_count + warp_prefix_7;
    __syncthreads();
    if (tid_0 < 128) {
        if (is_local_3 != 0) {
            int local_idx_0 = expert - local_experts_start_idx >> local_experts_stride_log2;
            int mn_limit1 = 0;
            int mn_limit2 = 0;
            int mn_limit = 0;
            for (int cta = 0; cta < num_cta; cta++) {
                cta_idx_xy_to_batch_idx[exclusive + cta] = local_idx_0;
                int scaled_1 = (exclusive + cta + 1) * tile_tokens_dim;
                if (padding_log2 > 0) {
                    scaled_1 = exclusive + cta + 1 << padding_log2;
                }
                mn_limit1 = scaled_1;
                int scaled_0 = exclusive * tile_tokens_dim;
                if (padding_log2 > 0) {
                    scaled_0 = exclusive << padding_log2;
                }
                mn_limit2 = scaled_0 + acc_count;
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
        }
    }
    if (tid_0 == 0) {
        int scaled_2 = block_total * tile_tokens_dim;
        if (padding_log2 > 0) {
            scaled_2 = block_total << padding_log2;
        }
        int padded_rows = scaled_2;
        permuted_idx_size[0] = padded_rows;
        num_non_exiting_ctas[0] = block_total;
    }
    if (tid_0 < 128) {
        int slot_p = 0;
        int k_of_slot = 0;
        int expanded_q = 0;
        int permuted_q = -1;
        for (int token_p = 0; token_p < num_tokens; token_p++) {
            slot_p = token_p * 128 + expert;
            k_of_slot = (int)smem_kidx[slot_p];
            if (k_of_slot >= 0) {
                expanded_q = token_p * top_k + k_of_slot;
                permuted_q = -1;
                if (is_local_3 != 0) {
                    permuted_q = exclusive_10 + (int)smem_offset16[slot_p];
                }
                expanded_idx_to_permuted_idx[expanded_q] = permuted_q;
                if (is_local_3 != 0) {
                    permuted_idx_to_token_idx[permuted_q] = token_p;
                }
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
