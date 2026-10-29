/*
 * Copyright (c) 2026 by FlashInfer team.
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
#define SMEM_WARP_HIST_OFF 0
#define SMEM_WARP_HIST_STAGE_BYTES 39424
#define SMEM_WARP_HIST_STRIDE 39424
#define SMEM_HIST_F32_OFF 0
#define SMEM_HIST_F32_STAGE_BYTES 39424
#define SMEM_HIST_F32_STRIDE 39424
#define SMEM_CAND_COMP_OFF 0
#define SMEM_CAND_COMP_STAGE_BYTES 39424
#define SMEM_CAND_COMP_STRIDE 39424
#define SMEM_WINNER_IDS_OFF 28672
#define SMEM_WINNER_IDS_STAGE_BYTES 448
#define SMEM_WINNER_IDS_STRIDE 448
#define SMEM_WINNER_SCORES_OFF 29120
#define SMEM_WINNER_SCORES_STAGE_BYTES 448
#define SMEM_WINNER_SCORES_STRIDE 448
#define SMEM_BIAS_SMEM_OFF 29568
#define SMEM_BIAS_SMEM_STAGE_BYTES 3584
#define SMEM_BIAS_SMEM_STRIDE 3584
#define SMEM_OWN_REC_OFF 38528
#define SMEM_OWN_REC_STAGE_BYTES 896
#define SMEM_OWN_REC_STRIDE 896
#define SMEM_PHASE2_WARP_SUMS_OFF 0
#define SMEM_PHASE2_WARP_SUMS_STAGE_BYTES 28
#define SMEM_PHASE2_WARP_SUMS_STRIDE 28
#define SMEM_LOCAL_OFFSETS_OFF 8192
#define SMEM_LOCAL_OFFSETS_STAGE_BYTES 3584
#define SMEM_LOCAL_OFFSETS_STRIDE 3584
#define SMEM_BITMAP_A_OFF 1024
#define SMEM_BITMAP_A_STAGE_BYTES 1024
#define SMEM_BITMAP_A_STRIDE 1024
#define SMEM_BITMAP_B_OFF 2048
#define SMEM_BITMAP_B_STAGE_BYTES 1024
#define SMEM_BITMAP_B_STRIDE 1024
#define SMEM_TOTALS_A_OFF 3072
#define SMEM_TOTALS_A_STAGE_BYTES 32
#define SMEM_TOTALS_A_STRIDE 32
#define SMEM_TOTALS_B_OFF 3104
#define SMEM_TOTALS_B_STAGE_BYTES 32
#define SMEM_TOTALS_B_STRIDE 32
#define SMEM_ROUTE_A_OFF 12288
#define SMEM_ROUTE_A_STAGE_BYTES 8192
#define SMEM_ROUTE_A_STRIDE 8192
#define SMEM_ROUTE_B_OFF 20480
#define SMEM_ROUTE_B_STAGE_BYTES 8192
#define SMEM_ROUTE_B_STRIDE 8192
#define SMEM_WRANK_A_OFF 4096
#define SMEM_WRANK_A_STAGE_BYTES 1024
#define SMEM_WRANK_A_STRIDE 1024
#define SMEM_WRANK_B_OFF 5120
#define SMEM_WRANK_B_STAGE_BYTES 1024
#define SMEM_WRANK_B_STRIDE 1024
#define SMEM_TOTAL 39424
#define THREADS 224
#ifndef BLOCK_M
#error "BLOCK_M is a downstream specialization of this program; define it on the compile line"
#endif
#define USE_WARP_PACKED_PHASE3 1
#define NUM_EXPERTS 896
#define TOP_K 16
#define ITEMS_PER_THREAD 4
#define NUM_WARPS 7
#define RADIX_CLEAR_THREADS 128
#define RADIX_SCAN_THREADS 32
#define RADIX_BINS_PER_LANE 8
#define MAX_BLOCK_M 16
#define BLOCK_MASK (BLOCK_M - 1)
#define BLOCK_SHIFT (3 + (BLOCK_M >> 4))
#define SCATTER_ROW_GROUPS (THREADS >> 4)
#define PADDING_GROUPS (THREADS >> BLOCK_SHIFT)
#define LAUNCH_MIN_BLOCKS 4

#include <math_constants.h>
#include <cooperative_groups.h>

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


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(224, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_kimi_k3_fused_router_fe39f52a822c7aa2a709(float* __restrict__ logits, float* __restrict__ bias, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M)
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
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    int* warp_hist = reinterpret_cast<int*>(smem_raw + 0);
    const int warp_hist_addr = smem + 0;
    float* hist_f32 = reinterpret_cast<float*>(smem_raw + 0);
    const int hist_f32_addr = smem + 0;
    unsigned long long* cand_comp = reinterpret_cast<unsigned long long*>(smem_raw + 0);
    const int cand_comp_addr = smem + 0;
    int* winner_ids = reinterpret_cast<int*>(smem_raw + 28672);
    const int winner_ids_addr = smem + 28672;
    float* winner_scores = reinterpret_cast<float*>(smem_raw + 29120);
    const int winner_scores_addr = smem + 29120;
    float* bias_smem = reinterpret_cast<float*>(smem_raw + 29568);
    const int bias_smem_addr = smem + 29568;
    int* own_rec = reinterpret_cast<int*>(smem_raw + 38528);
    const int own_rec_addr = smem + 38528;
    int* phase2_warp_sums = reinterpret_cast<int*>(smem_raw + 0);
    const int phase2_warp_sums_addr = smem + 0;
    int* local_offsets = reinterpret_cast<int*>(smem_raw + 8192);
    const int local_offsets_addr = smem + 8192;
    unsigned int* bitmap_a = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int bitmap_a_addr = smem + 1024;
    unsigned int* bitmap_b = reinterpret_cast<unsigned int*>(smem_raw + 2048);
    const int bitmap_b_addr = smem + 2048;
    int* totals_a = reinterpret_cast<int*>(smem_raw + 3072);
    const int totals_a_addr = smem + 3072;
    int* totals_b = reinterpret_cast<int*>(smem_raw + 3104);
    const int totals_b_addr = smem + 3104;
    uint8_t* route_a = reinterpret_cast<uint8_t*>(smem_raw + 12288);
    const int route_a_addr = smem + 12288;
    uint8_t* route_b = reinterpret_cast<uint8_t*>(smem_raw + 20480);
    const int route_b_addr = smem + 20480;
    int* wrank_a = reinterpret_cast<int*>(smem_raw + 4096);
    const int wrank_a_addr = smem + 4096;
    int* wrank_b = reinterpret_cast<int*>(smem_raw + 5120);
    const int wrank_b_addr = smem + 5120;

    // === Task calls (declaration order) ===
    int global_thread = bid * THREADS + tid;
    int grid_threads = num_bids * THREADS;
    #pragma unroll 1
    for (int expert_zero = global_thread; expert_zero < NUM_EXPERTS; expert_zero += grid_threads) {
        expert_counts[expert_zero] = 0;
        if (M * TOP_K >= NUM_EXPERTS * 8) {
            sorted_token_ids[expert_zero * 8] = 0;
        }
    }
    if (global_thread == 0) {
        expert_offsets[0] = -1;
        expert_offsets[NUM_EXPERTS] = 0;
    }
    int expert_base = tid * ITEMS_PER_THREAD;
    int stash_base = warp * TOP_K;
    unsigned int join_token = 0;
    int ae_now = 1;
    if (M <= num_bids * NUM_WARPS) {
        ae_now = 0;
    }
    if (ae_now != 0) {
        unsigned int _grid_arrive_0 = cooperative_groups::this_grid().barrier_arrive();
        join_token = _grid_arrive_0;
        if (warp != 0) {
            int m_base = tid - 32;
            float m_vals[5];
            #pragma unroll
            for (int m_slot = 0; m_slot < 5; m_slot++) {
                m_vals[m_slot] = 0.0f;
                if (m_base + m_slot * (THREADS - 32) < NUM_EXPERTS) {
                    m_vals[m_slot] = bias[m_base + m_slot * (THREADS - 32)];
                }
            }
            #pragma unroll
            for (int m_st = 0; m_st < 5; m_st++) {
                if (m_base + m_st * (THREADS - 32) < NUM_EXPERTS) {
                    bias_smem[m_base + m_st * (THREADS - 32)] = m_vals[m_st];
                }
            }
        }
        if (lane < TOP_K) {
            winner_ids[stash_base + lane] = 0;
            winner_scores[stash_base + lane] = 0.0f;
        }
        __syncthreads();
    }
    if (ae_now == 0) {
        float _vec_load_0[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(bias + expert_base);
            _vec_load_0[0 + 0] = _v4.x;
            _vec_load_0[0 + 1] = _v4.y;
            _vec_load_0[0 + 2] = _v4.z;
            _vec_load_0[0 + 3] = _v4.w;
        }
        #pragma unroll
        for (int f_slot = 0; f_slot < 4; f_slot++) {
            bias_smem[expert_base + f_slot] = _vec_load_0[f_slot];
        }
        if (lane < TOP_K) {
            winner_ids[stash_base + lane] = 0;
            winner_scores[stash_base + lane] = 0.0f;
        }
        __syncthreads();
        unsigned int _grid_arrive_1 = cooperative_groups::this_grid().barrier_arrive();
        join_token = _grid_arrive_1;
    }
    int hist_base = warp * 896;
    int cand_base = 6272 + warp * 128;
    int lane_hist_base = hist_base + lane * 9;
    int scan_bin_base = lane * 8;
    unsigned int one_u32 = 1;
    unsigned int lanemask_lt = (one_u32 << (unsigned int)lane) - 1;
    int token_stride = num_bids * NUM_WARPS;
    int cta_first = bid * NUM_WARPS;
    int n_it = 0;
    if (cta_first < M) {
        n_it = (M - cta_first + token_stride - 1) / token_stride;
    }
    int row_token = cta_first + warp;
    int it = 0;
    while (it < n_it) {
        int token = row_token;
        if (token < M) {
            unsigned long long row_base = (unsigned long long)token * (unsigned long long)NUM_EXPERTS;
            unsigned int keys[28];
            float v_val[28];
            #pragma unroll
            for (int item_vl = 0; item_vl < 28; item_vl++) {
                v_val[item_vl] = logits[row_base + (unsigned long long)(item_vl * 32 + lane)];
            }
            float v_min = v_val[0];
            #pragma unroll
            for (int item_vm = 1; item_vm < 28; item_vm++) {
                v_min = fminf(v_min, v_val[item_vm]);
            }
            unsigned int v_ok = 0;
            if (v_min >= -86.0f) {
                v_ok = 1;
            }
            unsigned int v_all;
            asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(v_all) : "r"(v_ok));
            if (v_all != 0) {
                #pragma unroll
                for (int item_vf = 0; item_vf < 28; item_vf++) {
                    float v_ef = approx_exp2(v_val[item_vf] * -1.442695f);
                    float _rcp_1 = approx_rcp(1.0f + v_ef);
                    v_val[item_vf] = _rcp_1;
                }
            } else {
                #pragma unroll
                for (int item_vs = 0; item_vs < 28; item_vs++) {
                    float v_es = approx_exp2(v_val[item_vs] * -1.442695f);
                    float _fdiv_2;
                    asm volatile("div.full.f32 %0, %1, %2;" : "=f"(_fdiv_2) : "f"(1.0f), "f"(1.0f + v_es));
                    v_val[item_vs] = _fdiv_2;
                }
            }
            #pragma unroll
            for (int item = 0; item < 28; item++) {
                int col = item * 32 + lane;
                float score = v_val[item];
                hist_f32[hist_base + col] = score;
                float bias_col = bias_smem[col];
                float ranking = score + bias_col;
                unsigned int bits = __as_u32(ranking);
                unsigned int ordered = bits ^ ((unsigned int)((int)bits >> 31) | 2147483648u);
                keys[item] = ordered;
            }
            int use_radix = 1;
            unsigned int p_lane_max = keys[0];
            #pragma unroll
            for (int item_pmax = 1; item_pmax < 28; item_pmax++) {
                unsigned int _max_3 = ((p_lane_max) > (keys[item_pmax]) ? (p_lane_max) : (keys[item_pmax]));
                p_lane_max = _max_3;
            }
            unsigned int p_sort = p_lane_max;
            unsigned int p_peer_1_0 = __shfl_xor_sync(0xFFFFFFFF, p_sort, 1);
            int p_take_max_2_0 = (lane >> 1 ^ lane) & 1;
            if (p_take_max_2_0 == 0) {
                unsigned int _min_4 = ((p_sort) < (p_peer_1_0) ? (p_sort) : (p_peer_1_0));
                p_sort = _min_4;
            } else {
                unsigned int _max_5 = ((p_sort) > (p_peer_1_0) ? (p_sort) : (p_peer_1_0));
                p_sort = _max_5;
            }
            unsigned int p_peer_1_1 = __shfl_xor_sync(0xFFFFFFFF, p_sort, 2);
            int p_take_max_2_1 = (lane >> 2 ^ lane >> 1) & 1;
            if (p_take_max_2_1 == 0) {
                unsigned int _min_6 = ((p_sort) < (p_peer_1_1) ? (p_sort) : (p_peer_1_1));
                p_sort = _min_6;
            } else {
                unsigned int _max_7 = ((p_sort) > (p_peer_1_1) ? (p_sort) : (p_peer_1_1));
                p_sort = _max_7;
            }
            unsigned int p_peer_1_2 = __shfl_xor_sync(0xFFFFFFFF, p_sort, 1);
            int p_take_max_2_2 = (lane >> 2 ^ lane) & 1;
            if (p_take_max_2_2 == 0) {
                unsigned int _min_8 = ((p_sort) < (p_peer_1_2) ? (p_sort) : (p_peer_1_2));
                p_sort = _min_8;
            } else {
                unsigned int _max_9 = ((p_sort) > (p_peer_1_2) ? (p_sort) : (p_peer_1_2));
                p_sort = _max_9;
            }
            unsigned int p_peer_1_3 = __shfl_xor_sync(0xFFFFFFFF, p_sort, 4);
            int p_take_max_2_3 = (lane >> 3 ^ lane >> 2) & 1;
            if (p_take_max_2_3 == 0) {
                unsigned int _min_10 = ((p_sort) < (p_peer_1_3) ? (p_sort) : (p_peer_1_3));
                p_sort = _min_10;
            } else {
                unsigned int _max_11 = ((p_sort) > (p_peer_1_3) ? (p_sort) : (p_peer_1_3));
                p_sort = _max_11;
            }
            unsigned int p_peer_1_4 = __shfl_xor_sync(0xFFFFFFFF, p_sort, 2);
            int p_take_max_2_4 = (lane >> 3 ^ lane >> 1) & 1;
            if (p_take_max_2_4 == 0) {
                unsigned int _min_12 = ((p_sort) < (p_peer_1_4) ? (p_sort) : (p_peer_1_4));
                p_sort = _min_12;
            } else {
                unsigned int _max_13 = ((p_sort) > (p_peer_1_4) ? (p_sort) : (p_peer_1_4));
                p_sort = _max_13;
            }
            unsigned int p_peer_1_5 = __shfl_xor_sync(0xFFFFFFFF, p_sort, 1);
            int p_take_max_2_5 = (lane >> 3 ^ lane) & 1;
            if (p_take_max_2_5 == 0) {
                unsigned int _min_14 = ((p_sort) < (p_peer_1_5) ? (p_sort) : (p_peer_1_5));
                p_sort = _min_14;
            } else {
                unsigned int _max_15 = ((p_sort) > (p_peer_1_5) ? (p_sort) : (p_peer_1_5));
                p_sort = _max_15;
            }
            unsigned int p_peer_1_6 = __shfl_xor_sync(0xFFFFFFFF, p_sort, 8);
            int p_take_max_2_6 = (lane >> 4 ^ lane >> 3) & 1;
            if (p_take_max_2_6 == 0) {
                unsigned int _min_16 = ((p_sort) < (p_peer_1_6) ? (p_sort) : (p_peer_1_6));
                p_sort = _min_16;
            } else {
                unsigned int _max_17 = ((p_sort) > (p_peer_1_6) ? (p_sort) : (p_peer_1_6));
                p_sort = _max_17;
            }
            unsigned int p_peer_1_7 = __shfl_xor_sync(0xFFFFFFFF, p_sort, 4);
            int p_take_max_2_7 = (lane >> 4 ^ lane >> 2) & 1;
            if (p_take_max_2_7 == 0) {
                unsigned int _min_18 = ((p_sort) < (p_peer_1_7) ? (p_sort) : (p_peer_1_7));
                p_sort = _min_18;
            } else {
                unsigned int _max_19 = ((p_sort) > (p_peer_1_7) ? (p_sort) : (p_peer_1_7));
                p_sort = _max_19;
            }
            unsigned int p_peer_1_8 = __shfl_xor_sync(0xFFFFFFFF, p_sort, 2);
            int p_take_max_2_8 = (lane >> 4 ^ lane >> 1) & 1;
            if (p_take_max_2_8 == 0) {
                unsigned int _min_20 = ((p_sort) < (p_peer_1_8) ? (p_sort) : (p_peer_1_8));
                p_sort = _min_20;
            } else {
                unsigned int _max_21 = ((p_sort) > (p_peer_1_8) ? (p_sort) : (p_peer_1_8));
                p_sort = _max_21;
            }
            unsigned int p_peer_1_9 = __shfl_xor_sync(0xFFFFFFFF, p_sort, 1);
            int p_take_max_2_9 = (lane >> 4 ^ lane) & 1;
            if (p_take_max_2_9 == 0) {
                unsigned int _min_22 = ((p_sort) < (p_peer_1_9) ? (p_sort) : (p_peer_1_9));
                p_sort = _min_22;
            } else {
                unsigned int _max_23 = ((p_sort) > (p_peer_1_9) ? (p_sort) : (p_peer_1_9));
                p_sort = _max_23;
            }
            unsigned int p_peer_1_10 = __shfl_xor_sync(0xFFFFFFFF, p_sort, 16);
            int p_take_max_2_10 = (lane >> 5 ^ lane >> 4) & 1;
            if (p_take_max_2_10 == 0) {
                unsigned int _min_24 = ((p_sort) < (p_peer_1_10) ? (p_sort) : (p_peer_1_10));
                p_sort = _min_24;
            } else {
                unsigned int _max_25 = ((p_sort) > (p_peer_1_10) ? (p_sort) : (p_peer_1_10));
                p_sort = _max_25;
            }
            unsigned int p_upper = 4294967295;
            if (lane >= 16) {
                p_upper = p_sort;
            }
            unsigned int p_threshold;
            asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(p_threshold) : "r"(p_upper));
            unsigned int p_mask = 0;
            #pragma unroll
            for (int item_pc = 0; item_pc < 28; item_pc++) {
                if (keys[item_pc] >= p_threshold) {
                    p_mask |= one_u32 << (unsigned int)item_pc;
                }
            }
            unsigned int p_count = (unsigned int)__popc(p_mask);
            unsigned int _warp_redux_26;
            asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_26) : "r"(p_count));
            int p_total = (int)_warp_redux_26;
            if (p_total <= 32) {
                use_radix = 0;
                unsigned int p_incl = p_count;
                unsigned int p_up_4_0 = __shfl_up_sync(0xFFFFFFFF, p_incl, 1, 32);
                if (lane >= 1) {
                    p_incl += p_up_4_0;
                }
                unsigned int p_up_4_1 = __shfl_up_sync(0xFFFFFFFF, p_incl, 2, 32);
                if (lane >= 2) {
                    p_incl += p_up_4_1;
                }
                unsigned int p_up_4_2 = __shfl_up_sync(0xFFFFFFFF, p_incl, 4, 32);
                if (lane >= 4) {
                    p_incl += p_up_4_2;
                }
                unsigned int p_up_4_3 = __shfl_up_sync(0xFFFFFFFF, p_incl, 8, 32);
                if (lane >= 8) {
                    p_incl += p_up_4_3;
                }
                unsigned int p_up_4_4 = __shfl_up_sync(0xFFFFFFFF, p_incl, 16, 32);
                if (lane >= 16) {
                    p_incl += p_up_4_4;
                }
                int p_slot = (int)(p_incl - p_count);
                unsigned int p_walk = p_mask;
                while (p_walk != 0) {
                    int _find_first_set_27 = __ffs((int)p_walk);
                    int p_item = _find_first_set_27 - 1;
                    warp_hist[cand_base + p_slot] = p_item * 32 + lane;
                    p_walk = p_walk & p_walk - one_u32;
                    p_slot += 1;
                }
                __syncwarp();
                int p_id = 0;
                float p_score = 0.0f;
                unsigned long long p_comp = 0;
                int comp_base = cand_base + 64 >> 1;
                if (p_total > lane) {
                    p_id = warp_hist[cand_base + lane];
                    p_score = hist_f32[hist_base + p_id];
                    float p_ranking = p_score + bias_smem[p_id];
                    unsigned int p_bits = __as_u32(p_ranking);
                    unsigned int p_ordered = p_bits ^ ((unsigned int)((int)p_bits >> 31) | 2147483648u);
                    p_comp = (unsigned long long)p_ordered << 32 | (unsigned long long)(1023 - p_id);
                }
                cand_comp[comp_base + lane] = p_comp;
                __syncwarp();
                int p_ra = 0;
                int p_rb = 0;
                int p_rc = 0;
                int p_rd = 0;
                unsigned int p_peers[16];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[0])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(0) + 3]))
                    : "r"(cand_comp_addr + (unsigned int)((comp_base + 0 + 0) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[4])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(4) + 3]))
                    : "r"(cand_comp_addr + (unsigned int)((comp_base + 0 + 2) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[8])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(8) + 3]))
                    : "r"(cand_comp_addr + (unsigned int)((comp_base + 0 + 4) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[12])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(12) + 3]))
                    : "r"(cand_comp_addr + (unsigned int)((comp_base + 0 + 6) * 8)));
                p_ra += ((p_comp < ((unsigned long long)p_peers[1] << 32 | (unsigned long long)p_peers[0])) ? 1 : 0);
                p_rb += ((p_comp < ((unsigned long long)p_peers[3] << 32 | (unsigned long long)p_peers[2])) ? 1 : 0);
                p_rc += ((p_comp < ((unsigned long long)p_peers[5] << 32 | (unsigned long long)p_peers[4])) ? 1 : 0);
                p_rd += ((p_comp < ((unsigned long long)p_peers[7] << 32 | (unsigned long long)p_peers[6])) ? 1 : 0);
                p_ra += ((p_comp < ((unsigned long long)p_peers[9] << 32 | (unsigned long long)p_peers[8])) ? 1 : 0);
                p_rb += ((p_comp < ((unsigned long long)p_peers[11] << 32 | (unsigned long long)p_peers[10])) ? 1 : 0);
                p_rc += ((p_comp < ((unsigned long long)p_peers[13] << 32 | (unsigned long long)p_peers[12])) ? 1 : 0);
                p_rd += ((p_comp < ((unsigned long long)p_peers[15] << 32 | (unsigned long long)p_peers[14])) ? 1 : 0);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[0])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(0) + 3]))
                    : "r"(cand_comp_addr + (unsigned int)((comp_base + 8 + 0) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[4])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(4) + 3]))
                    : "r"(cand_comp_addr + (unsigned int)((comp_base + 8 + 2) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[8])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(8) + 3]))
                    : "r"(cand_comp_addr + (unsigned int)((comp_base + 8 + 4) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[12])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(12) + 3]))
                    : "r"(cand_comp_addr + (unsigned int)((comp_base + 8 + 6) * 8)));
                p_ra += ((p_comp < ((unsigned long long)p_peers[1] << 32 | (unsigned long long)p_peers[0])) ? 1 : 0);
                p_rb += ((p_comp < ((unsigned long long)p_peers[3] << 32 | (unsigned long long)p_peers[2])) ? 1 : 0);
                p_rc += ((p_comp < ((unsigned long long)p_peers[5] << 32 | (unsigned long long)p_peers[4])) ? 1 : 0);
                p_rd += ((p_comp < ((unsigned long long)p_peers[7] << 32 | (unsigned long long)p_peers[6])) ? 1 : 0);
                p_ra += ((p_comp < ((unsigned long long)p_peers[9] << 32 | (unsigned long long)p_peers[8])) ? 1 : 0);
                p_rb += ((p_comp < ((unsigned long long)p_peers[11] << 32 | (unsigned long long)p_peers[10])) ? 1 : 0);
                p_rc += ((p_comp < ((unsigned long long)p_peers[13] << 32 | (unsigned long long)p_peers[12])) ? 1 : 0);
                p_rd += ((p_comp < ((unsigned long long)p_peers[15] << 32 | (unsigned long long)p_peers[14])) ? 1 : 0);
                if (p_total > 16) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[0])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(0) + 3]))
                        : "r"(cand_comp_addr + (unsigned int)((comp_base + 16 + 0) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[4])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(4) + 3]))
                        : "r"(cand_comp_addr + (unsigned int)((comp_base + 16 + 2) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[8])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(8) + 3]))
                        : "r"(cand_comp_addr + (unsigned int)((comp_base + 16 + 4) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[12])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(12) + 3]))
                        : "r"(cand_comp_addr + (unsigned int)((comp_base + 16 + 6) * 8)));
                    p_ra += ((p_comp < ((unsigned long long)p_peers[1] << 32 | (unsigned long long)p_peers[0])) ? 1 : 0);
                    p_rb += ((p_comp < ((unsigned long long)p_peers[3] << 32 | (unsigned long long)p_peers[2])) ? 1 : 0);
                    p_rc += ((p_comp < ((unsigned long long)p_peers[5] << 32 | (unsigned long long)p_peers[4])) ? 1 : 0);
                    p_rd += ((p_comp < ((unsigned long long)p_peers[7] << 32 | (unsigned long long)p_peers[6])) ? 1 : 0);
                    p_ra += ((p_comp < ((unsigned long long)p_peers[9] << 32 | (unsigned long long)p_peers[8])) ? 1 : 0);
                    p_rb += ((p_comp < ((unsigned long long)p_peers[11] << 32 | (unsigned long long)p_peers[10])) ? 1 : 0);
                    p_rc += ((p_comp < ((unsigned long long)p_peers[13] << 32 | (unsigned long long)p_peers[12])) ? 1 : 0);
                    p_rd += ((p_comp < ((unsigned long long)p_peers[15] << 32 | (unsigned long long)p_peers[14])) ? 1 : 0);
                }
                if (p_total > 24) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[0])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(0) + 3]))
                        : "r"(cand_comp_addr + (unsigned int)((comp_base + 24 + 0) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[4])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(4) + 3]))
                        : "r"(cand_comp_addr + (unsigned int)((comp_base + 24 + 2) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[8])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(8) + 3]))
                        : "r"(cand_comp_addr + (unsigned int)((comp_base + 24 + 4) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[12])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(12) + 3]))
                        : "r"(cand_comp_addr + (unsigned int)((comp_base + 24 + 6) * 8)));
                    p_ra += ((p_comp < ((unsigned long long)p_peers[1] << 32 | (unsigned long long)p_peers[0])) ? 1 : 0);
                    p_rb += ((p_comp < ((unsigned long long)p_peers[3] << 32 | (unsigned long long)p_peers[2])) ? 1 : 0);
                    p_rc += ((p_comp < ((unsigned long long)p_peers[5] << 32 | (unsigned long long)p_peers[4])) ? 1 : 0);
                    p_rd += ((p_comp < ((unsigned long long)p_peers[7] << 32 | (unsigned long long)p_peers[6])) ? 1 : 0);
                    p_ra += ((p_comp < ((unsigned long long)p_peers[9] << 32 | (unsigned long long)p_peers[8])) ? 1 : 0);
                    p_rb += ((p_comp < ((unsigned long long)p_peers[11] << 32 | (unsigned long long)p_peers[10])) ? 1 : 0);
                    p_rc += ((p_comp < ((unsigned long long)p_peers[13] << 32 | (unsigned long long)p_peers[12])) ? 1 : 0);
                    p_rd += ((p_comp < ((unsigned long long)p_peers[15] << 32 | (unsigned long long)p_peers[14])) ? 1 : 0);
                }
                int p_rank = p_ra + p_rb + (p_rc + p_rd);
                if (p_total > lane) {
                    if (p_rank < TOP_K) {
                        warp_hist[cand_base + 32 + p_rank] = p_id;
                        hist_f32[cand_base + 48 + p_rank] = p_score;
                    }
                }
                __syncwarp();
                if (lane < TOP_K) {
                    int p_win_id = warp_hist[cand_base + 32 + lane];
                    float p_win_score = hist_f32[cand_base + 48 + lane];
                    int p_sa = 0;
                    int p_sb = 0;
                    int p_sc = 0;
                    int p_sd = 0;
                    unsigned int p_wins[16];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&p_wins[0])), "=r"(*reinterpret_cast<uint32_t*>(&p_wins[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_wins[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_wins[(0) + 3]))
                        : "r"(warp_hist_addr + (unsigned int)((cand_base + 32 + 0) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&p_wins[4])), "=r"(*reinterpret_cast<uint32_t*>(&p_wins[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_wins[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_wins[(4) + 3]))
                        : "r"(warp_hist_addr + (unsigned int)((cand_base + 32 + 4) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&p_wins[8])), "=r"(*reinterpret_cast<uint32_t*>(&p_wins[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_wins[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_wins[(8) + 3]))
                        : "r"(warp_hist_addr + (unsigned int)((cand_base + 32 + 8) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&p_wins[12])), "=r"(*reinterpret_cast<uint32_t*>(&p_wins[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_wins[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_wins[(12) + 3]))
                        : "r"(warp_hist_addr + (unsigned int)((cand_base + 32 + 12) * 4)));
                    p_sa += ((p_win_id > (int)p_wins[0]) ? 1 : 0);
                    p_sb += ((p_win_id > (int)p_wins[1]) ? 1 : 0);
                    p_sc += ((p_win_id > (int)p_wins[2]) ? 1 : 0);
                    p_sd += ((p_win_id > (int)p_wins[3]) ? 1 : 0);
                    p_sa += ((p_win_id > (int)p_wins[4]) ? 1 : 0);
                    p_sb += ((p_win_id > (int)p_wins[5]) ? 1 : 0);
                    p_sc += ((p_win_id > (int)p_wins[6]) ? 1 : 0);
                    p_sd += ((p_win_id > (int)p_wins[7]) ? 1 : 0);
                    p_sa += ((p_win_id > (int)p_wins[8]) ? 1 : 0);
                    p_sb += ((p_win_id > (int)p_wins[9]) ? 1 : 0);
                    p_sc += ((p_win_id > (int)p_wins[10]) ? 1 : 0);
                    p_sd += ((p_win_id > (int)p_wins[11]) ? 1 : 0);
                    p_sa += ((p_win_id > (int)p_wins[12]) ? 1 : 0);
                    p_sb += ((p_win_id > (int)p_wins[13]) ? 1 : 0);
                    p_sc += ((p_win_id > (int)p_wins[14]) ? 1 : 0);
                    p_sd += ((p_win_id > (int)p_wins[15]) ? 1 : 0);
                    int p_win_slot = p_sa + p_sb + (p_sc + p_sd);
                    winner_ids[stash_base + p_win_slot] = p_win_id;
                    winner_scores[stash_base + p_win_slot] = p_win_score;
                }
                __syncwarp();
            }
            if (p_total > 32) {
                if (p_total <= 64) {
                    use_radix = 0;
                    int q_base = 8288 + warp * 192;
                    unsigned int q_incl = p_count;
                    unsigned int q_up_4_0 = __shfl_up_sync(0xFFFFFFFF, q_incl, 1, 32);
                    if (lane >= 1) {
                        q_incl += q_up_4_0;
                    }
                    unsigned int q_up_4_1 = __shfl_up_sync(0xFFFFFFFF, q_incl, 2, 32);
                    if (lane >= 2) {
                        q_incl += q_up_4_1;
                    }
                    unsigned int q_up_4_2 = __shfl_up_sync(0xFFFFFFFF, q_incl, 4, 32);
                    if (lane >= 4) {
                        q_incl += q_up_4_2;
                    }
                    unsigned int q_up_4_3 = __shfl_up_sync(0xFFFFFFFF, q_incl, 8, 32);
                    if (lane >= 8) {
                        q_incl += q_up_4_3;
                    }
                    unsigned int q_up_4_4 = __shfl_up_sync(0xFFFFFFFF, q_incl, 16, 32);
                    if (lane >= 16) {
                        q_incl += q_up_4_4;
                    }
                    int q_slot = (int)(q_incl - p_count);
                    unsigned int q_walk = p_mask;
                    while (q_walk != 0) {
                        int _find_first_set_28 = __ffs((int)q_walk);
                        int q_item = _find_first_set_28 - 1;
                        warp_hist[q_base + q_slot] = q_item * 32 + lane;
                        q_walk = q_walk & q_walk - one_u32;
                        q_slot += 1;
                    }
                    __syncwarp();
                    int q_comp_base = q_base + 64 >> 1;
                    int q_hi = lane + 32;
                    int q_id_lo = warp_hist[q_base + lane];
                    int q_id_hi = 0;
                    float q_score_lo = 0.0f;
                    float q_score_hi = 0.0f;
                    unsigned long long q_comp_lo = 0;
                    unsigned long long q_comp_hi = 0;
                    q_score_lo = hist_f32[hist_base + q_id_lo];
                    q_comp_lo = (unsigned long long)(__as_u32(q_score_lo + bias_smem[q_id_lo]) ^ ((unsigned int)((int)__as_u32(q_score_lo + bias_smem[q_id_lo]) >> 31) | 2147483648u)) << 32 | (unsigned long long)(1023 - q_id_lo);
                    cand_comp[q_comp_base + lane] = q_comp_lo;
                    if (q_hi < p_total) {
                        q_id_hi = warp_hist[q_base + q_hi];
                        q_score_hi = hist_f32[hist_base + q_id_hi];
                        q_comp_hi = (unsigned long long)(__as_u32(q_score_hi + bias_smem[q_id_hi]) ^ ((unsigned int)((int)__as_u32(q_score_hi + bias_smem[q_id_hi]) >> 31) | 2147483648u)) << 32 | (unsigned long long)(1023 - q_id_hi);
                    }
                    cand_comp[q_comp_base + q_hi] = q_comp_hi;
                    __syncwarp();
                    int q_lo_a = 0;
                    int q_lo_b = 0;
                    int q_hi_a = 0;
                    int q_hi_b = 0;
                    unsigned int q_peers[16];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 3]))
                        : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 0 + 0) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 3]))
                        : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 0 + 2) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 3]))
                        : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 0 + 4) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 3]))
                        : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 0 + 6) * 8)));
                    q_lo_a += ((((unsigned long long)q_peers[1] << 32 | (unsigned long long)q_peers[0]) > q_comp_lo) ? 1 : 0);
                    q_lo_b += ((((unsigned long long)q_peers[3] << 32 | (unsigned long long)q_peers[2]) > q_comp_lo) ? 1 : 0);
                    q_hi_a += ((((unsigned long long)q_peers[1] << 32 | (unsigned long long)q_peers[0]) > q_comp_hi) ? 1 : 0);
                    q_hi_b += ((((unsigned long long)q_peers[3] << 32 | (unsigned long long)q_peers[2]) > q_comp_hi) ? 1 : 0);
                    q_lo_a += ((((unsigned long long)q_peers[5] << 32 | (unsigned long long)q_peers[4]) > q_comp_lo) ? 1 : 0);
                    q_lo_b += ((((unsigned long long)q_peers[7] << 32 | (unsigned long long)q_peers[6]) > q_comp_lo) ? 1 : 0);
                    q_hi_a += ((((unsigned long long)q_peers[5] << 32 | (unsigned long long)q_peers[4]) > q_comp_hi) ? 1 : 0);
                    q_hi_b += ((((unsigned long long)q_peers[7] << 32 | (unsigned long long)q_peers[6]) > q_comp_hi) ? 1 : 0);
                    q_lo_a += ((((unsigned long long)q_peers[9] << 32 | (unsigned long long)q_peers[8]) > q_comp_lo) ? 1 : 0);
                    q_lo_b += ((((unsigned long long)q_peers[11] << 32 | (unsigned long long)q_peers[10]) > q_comp_lo) ? 1 : 0);
                    q_hi_a += ((((unsigned long long)q_peers[9] << 32 | (unsigned long long)q_peers[8]) > q_comp_hi) ? 1 : 0);
                    q_hi_b += ((((unsigned long long)q_peers[11] << 32 | (unsigned long long)q_peers[10]) > q_comp_hi) ? 1 : 0);
                    q_lo_a += ((((unsigned long long)q_peers[13] << 32 | (unsigned long long)q_peers[12]) > q_comp_lo) ? 1 : 0);
                    q_lo_b += ((((unsigned long long)q_peers[15] << 32 | (unsigned long long)q_peers[14]) > q_comp_lo) ? 1 : 0);
                    q_hi_a += ((((unsigned long long)q_peers[13] << 32 | (unsigned long long)q_peers[12]) > q_comp_hi) ? 1 : 0);
                    q_hi_b += ((((unsigned long long)q_peers[15] << 32 | (unsigned long long)q_peers[14]) > q_comp_hi) ? 1 : 0);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 3]))
                        : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 8 + 0) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 3]))
                        : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 8 + 2) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 3]))
                        : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 8 + 4) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 3]))
                        : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 8 + 6) * 8)));
                    q_lo_a += ((((unsigned long long)q_peers[1] << 32 | (unsigned long long)q_peers[0]) > q_comp_lo) ? 1 : 0);
                    q_lo_b += ((((unsigned long long)q_peers[3] << 32 | (unsigned long long)q_peers[2]) > q_comp_lo) ? 1 : 0);
                    q_hi_a += ((((unsigned long long)q_peers[1] << 32 | (unsigned long long)q_peers[0]) > q_comp_hi) ? 1 : 0);
                    q_hi_b += ((((unsigned long long)q_peers[3] << 32 | (unsigned long long)q_peers[2]) > q_comp_hi) ? 1 : 0);
                    q_lo_a += ((((unsigned long long)q_peers[5] << 32 | (unsigned long long)q_peers[4]) > q_comp_lo) ? 1 : 0);
                    q_lo_b += ((((unsigned long long)q_peers[7] << 32 | (unsigned long long)q_peers[6]) > q_comp_lo) ? 1 : 0);
                    q_hi_a += ((((unsigned long long)q_peers[5] << 32 | (unsigned long long)q_peers[4]) > q_comp_hi) ? 1 : 0);
                    q_hi_b += ((((unsigned long long)q_peers[7] << 32 | (unsigned long long)q_peers[6]) > q_comp_hi) ? 1 : 0);
                    q_lo_a += ((((unsigned long long)q_peers[9] << 32 | (unsigned long long)q_peers[8]) > q_comp_lo) ? 1 : 0);
                    q_lo_b += ((((unsigned long long)q_peers[11] << 32 | (unsigned long long)q_peers[10]) > q_comp_lo) ? 1 : 0);
                    q_hi_a += ((((unsigned long long)q_peers[9] << 32 | (unsigned long long)q_peers[8]) > q_comp_hi) ? 1 : 0);
                    q_hi_b += ((((unsigned long long)q_peers[11] << 32 | (unsigned long long)q_peers[10]) > q_comp_hi) ? 1 : 0);
                    q_lo_a += ((((unsigned long long)q_peers[13] << 32 | (unsigned long long)q_peers[12]) > q_comp_lo) ? 1 : 0);
                    q_lo_b += ((((unsigned long long)q_peers[15] << 32 | (unsigned long long)q_peers[14]) > q_comp_lo) ? 1 : 0);
                    q_hi_a += ((((unsigned long long)q_peers[13] << 32 | (unsigned long long)q_peers[12]) > q_comp_hi) ? 1 : 0);
                    q_hi_b += ((((unsigned long long)q_peers[15] << 32 | (unsigned long long)q_peers[14]) > q_comp_hi) ? 1 : 0);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 3]))
                        : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 16 + 0) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 3]))
                        : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 16 + 2) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 3]))
                        : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 16 + 4) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 3]))
                        : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 16 + 6) * 8)));
                    q_lo_a += ((((unsigned long long)q_peers[1] << 32 | (unsigned long long)q_peers[0]) > q_comp_lo) ? 1 : 0);
                    q_lo_b += ((((unsigned long long)q_peers[3] << 32 | (unsigned long long)q_peers[2]) > q_comp_lo) ? 1 : 0);
                    q_hi_a += ((((unsigned long long)q_peers[1] << 32 | (unsigned long long)q_peers[0]) > q_comp_hi) ? 1 : 0);
                    q_hi_b += ((((unsigned long long)q_peers[3] << 32 | (unsigned long long)q_peers[2]) > q_comp_hi) ? 1 : 0);
                    q_lo_a += ((((unsigned long long)q_peers[5] << 32 | (unsigned long long)q_peers[4]) > q_comp_lo) ? 1 : 0);
                    q_lo_b += ((((unsigned long long)q_peers[7] << 32 | (unsigned long long)q_peers[6]) > q_comp_lo) ? 1 : 0);
                    q_hi_a += ((((unsigned long long)q_peers[5] << 32 | (unsigned long long)q_peers[4]) > q_comp_hi) ? 1 : 0);
                    q_hi_b += ((((unsigned long long)q_peers[7] << 32 | (unsigned long long)q_peers[6]) > q_comp_hi) ? 1 : 0);
                    q_lo_a += ((((unsigned long long)q_peers[9] << 32 | (unsigned long long)q_peers[8]) > q_comp_lo) ? 1 : 0);
                    q_lo_b += ((((unsigned long long)q_peers[11] << 32 | (unsigned long long)q_peers[10]) > q_comp_lo) ? 1 : 0);
                    q_hi_a += ((((unsigned long long)q_peers[9] << 32 | (unsigned long long)q_peers[8]) > q_comp_hi) ? 1 : 0);
                    q_hi_b += ((((unsigned long long)q_peers[11] << 32 | (unsigned long long)q_peers[10]) > q_comp_hi) ? 1 : 0);
                    q_lo_a += ((((unsigned long long)q_peers[13] << 32 | (unsigned long long)q_peers[12]) > q_comp_lo) ? 1 : 0);
                    q_lo_b += ((((unsigned long long)q_peers[15] << 32 | (unsigned long long)q_peers[14]) > q_comp_lo) ? 1 : 0);
                    q_hi_a += ((((unsigned long long)q_peers[13] << 32 | (unsigned long long)q_peers[12]) > q_comp_hi) ? 1 : 0);
                    q_hi_b += ((((unsigned long long)q_peers[15] << 32 | (unsigned long long)q_peers[14]) > q_comp_hi) ? 1 : 0);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 3]))
                        : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 24 + 0) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 3]))
                        : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 24 + 2) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 3]))
                        : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 24 + 4) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 3]))
                        : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 24 + 6) * 8)));
                    q_lo_a += ((((unsigned long long)q_peers[1] << 32 | (unsigned long long)q_peers[0]) > q_comp_lo) ? 1 : 0);
                    q_lo_b += ((((unsigned long long)q_peers[3] << 32 | (unsigned long long)q_peers[2]) > q_comp_lo) ? 1 : 0);
                    q_hi_a += ((((unsigned long long)q_peers[1] << 32 | (unsigned long long)q_peers[0]) > q_comp_hi) ? 1 : 0);
                    q_hi_b += ((((unsigned long long)q_peers[3] << 32 | (unsigned long long)q_peers[2]) > q_comp_hi) ? 1 : 0);
                    q_lo_a += ((((unsigned long long)q_peers[5] << 32 | (unsigned long long)q_peers[4]) > q_comp_lo) ? 1 : 0);
                    q_lo_b += ((((unsigned long long)q_peers[7] << 32 | (unsigned long long)q_peers[6]) > q_comp_lo) ? 1 : 0);
                    q_hi_a += ((((unsigned long long)q_peers[5] << 32 | (unsigned long long)q_peers[4]) > q_comp_hi) ? 1 : 0);
                    q_hi_b += ((((unsigned long long)q_peers[7] << 32 | (unsigned long long)q_peers[6]) > q_comp_hi) ? 1 : 0);
                    q_lo_a += ((((unsigned long long)q_peers[9] << 32 | (unsigned long long)q_peers[8]) > q_comp_lo) ? 1 : 0);
                    q_lo_b += ((((unsigned long long)q_peers[11] << 32 | (unsigned long long)q_peers[10]) > q_comp_lo) ? 1 : 0);
                    q_hi_a += ((((unsigned long long)q_peers[9] << 32 | (unsigned long long)q_peers[8]) > q_comp_hi) ? 1 : 0);
                    q_hi_b += ((((unsigned long long)q_peers[11] << 32 | (unsigned long long)q_peers[10]) > q_comp_hi) ? 1 : 0);
                    q_lo_a += ((((unsigned long long)q_peers[13] << 32 | (unsigned long long)q_peers[12]) > q_comp_lo) ? 1 : 0);
                    q_lo_b += ((((unsigned long long)q_peers[15] << 32 | (unsigned long long)q_peers[14]) > q_comp_lo) ? 1 : 0);
                    q_hi_a += ((((unsigned long long)q_peers[13] << 32 | (unsigned long long)q_peers[12]) > q_comp_hi) ? 1 : 0);
                    q_hi_b += ((((unsigned long long)q_peers[15] << 32 | (unsigned long long)q_peers[14]) > q_comp_hi) ? 1 : 0);
                    if (p_total > 32) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 3]))
                            : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 32 + 0) * 8)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 3]))
                            : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 32 + 2) * 8)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 3]))
                            : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 32 + 4) * 8)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 3]))
                            : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 32 + 6) * 8)));
                        q_lo_a += ((((unsigned long long)q_peers[1] << 32 | (unsigned long long)q_peers[0]) > q_comp_lo) ? 1 : 0);
                        q_lo_b += ((((unsigned long long)q_peers[3] << 32 | (unsigned long long)q_peers[2]) > q_comp_lo) ? 1 : 0);
                        q_hi_a += ((((unsigned long long)q_peers[1] << 32 | (unsigned long long)q_peers[0]) > q_comp_hi) ? 1 : 0);
                        q_hi_b += ((((unsigned long long)q_peers[3] << 32 | (unsigned long long)q_peers[2]) > q_comp_hi) ? 1 : 0);
                        q_lo_a += ((((unsigned long long)q_peers[5] << 32 | (unsigned long long)q_peers[4]) > q_comp_lo) ? 1 : 0);
                        q_lo_b += ((((unsigned long long)q_peers[7] << 32 | (unsigned long long)q_peers[6]) > q_comp_lo) ? 1 : 0);
                        q_hi_a += ((((unsigned long long)q_peers[5] << 32 | (unsigned long long)q_peers[4]) > q_comp_hi) ? 1 : 0);
                        q_hi_b += ((((unsigned long long)q_peers[7] << 32 | (unsigned long long)q_peers[6]) > q_comp_hi) ? 1 : 0);
                        q_lo_a += ((((unsigned long long)q_peers[9] << 32 | (unsigned long long)q_peers[8]) > q_comp_lo) ? 1 : 0);
                        q_lo_b += ((((unsigned long long)q_peers[11] << 32 | (unsigned long long)q_peers[10]) > q_comp_lo) ? 1 : 0);
                        q_hi_a += ((((unsigned long long)q_peers[9] << 32 | (unsigned long long)q_peers[8]) > q_comp_hi) ? 1 : 0);
                        q_hi_b += ((((unsigned long long)q_peers[11] << 32 | (unsigned long long)q_peers[10]) > q_comp_hi) ? 1 : 0);
                        q_lo_a += ((((unsigned long long)q_peers[13] << 32 | (unsigned long long)q_peers[12]) > q_comp_lo) ? 1 : 0);
                        q_lo_b += ((((unsigned long long)q_peers[15] << 32 | (unsigned long long)q_peers[14]) > q_comp_lo) ? 1 : 0);
                        q_hi_a += ((((unsigned long long)q_peers[13] << 32 | (unsigned long long)q_peers[12]) > q_comp_hi) ? 1 : 0);
                        q_hi_b += ((((unsigned long long)q_peers[15] << 32 | (unsigned long long)q_peers[14]) > q_comp_hi) ? 1 : 0);
                    }
                    if (p_total > 40) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 3]))
                            : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 40 + 0) * 8)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 3]))
                            : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 40 + 2) * 8)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 3]))
                            : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 40 + 4) * 8)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 3]))
                            : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 40 + 6) * 8)));
                        q_lo_a += ((((unsigned long long)q_peers[1] << 32 | (unsigned long long)q_peers[0]) > q_comp_lo) ? 1 : 0);
                        q_lo_b += ((((unsigned long long)q_peers[3] << 32 | (unsigned long long)q_peers[2]) > q_comp_lo) ? 1 : 0);
                        q_hi_a += ((((unsigned long long)q_peers[1] << 32 | (unsigned long long)q_peers[0]) > q_comp_hi) ? 1 : 0);
                        q_hi_b += ((((unsigned long long)q_peers[3] << 32 | (unsigned long long)q_peers[2]) > q_comp_hi) ? 1 : 0);
                        q_lo_a += ((((unsigned long long)q_peers[5] << 32 | (unsigned long long)q_peers[4]) > q_comp_lo) ? 1 : 0);
                        q_lo_b += ((((unsigned long long)q_peers[7] << 32 | (unsigned long long)q_peers[6]) > q_comp_lo) ? 1 : 0);
                        q_hi_a += ((((unsigned long long)q_peers[5] << 32 | (unsigned long long)q_peers[4]) > q_comp_hi) ? 1 : 0);
                        q_hi_b += ((((unsigned long long)q_peers[7] << 32 | (unsigned long long)q_peers[6]) > q_comp_hi) ? 1 : 0);
                        q_lo_a += ((((unsigned long long)q_peers[9] << 32 | (unsigned long long)q_peers[8]) > q_comp_lo) ? 1 : 0);
                        q_lo_b += ((((unsigned long long)q_peers[11] << 32 | (unsigned long long)q_peers[10]) > q_comp_lo) ? 1 : 0);
                        q_hi_a += ((((unsigned long long)q_peers[9] << 32 | (unsigned long long)q_peers[8]) > q_comp_hi) ? 1 : 0);
                        q_hi_b += ((((unsigned long long)q_peers[11] << 32 | (unsigned long long)q_peers[10]) > q_comp_hi) ? 1 : 0);
                        q_lo_a += ((((unsigned long long)q_peers[13] << 32 | (unsigned long long)q_peers[12]) > q_comp_lo) ? 1 : 0);
                        q_lo_b += ((((unsigned long long)q_peers[15] << 32 | (unsigned long long)q_peers[14]) > q_comp_lo) ? 1 : 0);
                        q_hi_a += ((((unsigned long long)q_peers[13] << 32 | (unsigned long long)q_peers[12]) > q_comp_hi) ? 1 : 0);
                        q_hi_b += ((((unsigned long long)q_peers[15] << 32 | (unsigned long long)q_peers[14]) > q_comp_hi) ? 1 : 0);
                    }
                    if (p_total > 48) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 3]))
                            : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 48 + 0) * 8)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 3]))
                            : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 48 + 2) * 8)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 3]))
                            : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 48 + 4) * 8)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 3]))
                            : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 48 + 6) * 8)));
                        q_lo_a += ((((unsigned long long)q_peers[1] << 32 | (unsigned long long)q_peers[0]) > q_comp_lo) ? 1 : 0);
                        q_lo_b += ((((unsigned long long)q_peers[3] << 32 | (unsigned long long)q_peers[2]) > q_comp_lo) ? 1 : 0);
                        q_hi_a += ((((unsigned long long)q_peers[1] << 32 | (unsigned long long)q_peers[0]) > q_comp_hi) ? 1 : 0);
                        q_hi_b += ((((unsigned long long)q_peers[3] << 32 | (unsigned long long)q_peers[2]) > q_comp_hi) ? 1 : 0);
                        q_lo_a += ((((unsigned long long)q_peers[5] << 32 | (unsigned long long)q_peers[4]) > q_comp_lo) ? 1 : 0);
                        q_lo_b += ((((unsigned long long)q_peers[7] << 32 | (unsigned long long)q_peers[6]) > q_comp_lo) ? 1 : 0);
                        q_hi_a += ((((unsigned long long)q_peers[5] << 32 | (unsigned long long)q_peers[4]) > q_comp_hi) ? 1 : 0);
                        q_hi_b += ((((unsigned long long)q_peers[7] << 32 | (unsigned long long)q_peers[6]) > q_comp_hi) ? 1 : 0);
                        q_lo_a += ((((unsigned long long)q_peers[9] << 32 | (unsigned long long)q_peers[8]) > q_comp_lo) ? 1 : 0);
                        q_lo_b += ((((unsigned long long)q_peers[11] << 32 | (unsigned long long)q_peers[10]) > q_comp_lo) ? 1 : 0);
                        q_hi_a += ((((unsigned long long)q_peers[9] << 32 | (unsigned long long)q_peers[8]) > q_comp_hi) ? 1 : 0);
                        q_hi_b += ((((unsigned long long)q_peers[11] << 32 | (unsigned long long)q_peers[10]) > q_comp_hi) ? 1 : 0);
                        q_lo_a += ((((unsigned long long)q_peers[13] << 32 | (unsigned long long)q_peers[12]) > q_comp_lo) ? 1 : 0);
                        q_lo_b += ((((unsigned long long)q_peers[15] << 32 | (unsigned long long)q_peers[14]) > q_comp_lo) ? 1 : 0);
                        q_hi_a += ((((unsigned long long)q_peers[13] << 32 | (unsigned long long)q_peers[12]) > q_comp_hi) ? 1 : 0);
                        q_hi_b += ((((unsigned long long)q_peers[15] << 32 | (unsigned long long)q_peers[14]) > q_comp_hi) ? 1 : 0);
                    }
                    if (p_total > 56) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 3]))
                            : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 56 + 0) * 8)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 3]))
                            : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 56 + 2) * 8)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 3]))
                            : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 56 + 4) * 8)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 3]))
                            : "r"(cand_comp_addr + (unsigned int)((q_comp_base + 56 + 6) * 8)));
                        q_lo_a += ((((unsigned long long)q_peers[1] << 32 | (unsigned long long)q_peers[0]) > q_comp_lo) ? 1 : 0);
                        q_lo_b += ((((unsigned long long)q_peers[3] << 32 | (unsigned long long)q_peers[2]) > q_comp_lo) ? 1 : 0);
                        q_hi_a += ((((unsigned long long)q_peers[1] << 32 | (unsigned long long)q_peers[0]) > q_comp_hi) ? 1 : 0);
                        q_hi_b += ((((unsigned long long)q_peers[3] << 32 | (unsigned long long)q_peers[2]) > q_comp_hi) ? 1 : 0);
                        q_lo_a += ((((unsigned long long)q_peers[5] << 32 | (unsigned long long)q_peers[4]) > q_comp_lo) ? 1 : 0);
                        q_lo_b += ((((unsigned long long)q_peers[7] << 32 | (unsigned long long)q_peers[6]) > q_comp_lo) ? 1 : 0);
                        q_hi_a += ((((unsigned long long)q_peers[5] << 32 | (unsigned long long)q_peers[4]) > q_comp_hi) ? 1 : 0);
                        q_hi_b += ((((unsigned long long)q_peers[7] << 32 | (unsigned long long)q_peers[6]) > q_comp_hi) ? 1 : 0);
                        q_lo_a += ((((unsigned long long)q_peers[9] << 32 | (unsigned long long)q_peers[8]) > q_comp_lo) ? 1 : 0);
                        q_lo_b += ((((unsigned long long)q_peers[11] << 32 | (unsigned long long)q_peers[10]) > q_comp_lo) ? 1 : 0);
                        q_hi_a += ((((unsigned long long)q_peers[9] << 32 | (unsigned long long)q_peers[8]) > q_comp_hi) ? 1 : 0);
                        q_hi_b += ((((unsigned long long)q_peers[11] << 32 | (unsigned long long)q_peers[10]) > q_comp_hi) ? 1 : 0);
                        q_lo_a += ((((unsigned long long)q_peers[13] << 32 | (unsigned long long)q_peers[12]) > q_comp_lo) ? 1 : 0);
                        q_lo_b += ((((unsigned long long)q_peers[15] << 32 | (unsigned long long)q_peers[14]) > q_comp_lo) ? 1 : 0);
                        q_hi_a += ((((unsigned long long)q_peers[13] << 32 | (unsigned long long)q_peers[12]) > q_comp_hi) ? 1 : 0);
                        q_hi_b += ((((unsigned long long)q_peers[15] << 32 | (unsigned long long)q_peers[14]) > q_comp_hi) ? 1 : 0);
                    }
                    int q_rank_lo = q_lo_a + q_lo_b;
                    int q_rank_hi = q_hi_a + q_hi_b;
                    if (q_rank_lo < TOP_K) {
                        warp_hist[cand_base + 32 + q_rank_lo] = q_id_lo;
                        hist_f32[cand_base + 48 + q_rank_lo] = q_score_lo;
                    }
                    if (q_hi < p_total) {
                        if (q_rank_hi < TOP_K) {
                            warp_hist[cand_base + 32 + q_rank_hi] = q_id_hi;
                            hist_f32[cand_base + 48 + q_rank_hi] = q_score_hi;
                        }
                    }
                    __syncwarp();
                    if (lane < TOP_K) {
                        int q_win_id = warp_hist[cand_base + 32 + lane];
                        float q_win_score = hist_f32[cand_base + 48 + lane];
                        int q_sa = 0;
                        int q_sb = 0;
                        int q_sc = 0;
                        int q_sd = 0;
                        unsigned int q_wins[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_wins[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_wins[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_wins[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_wins[(0) + 3]))
                            : "r"(warp_hist_addr + (unsigned int)((cand_base + 32 + 0) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_wins[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_wins[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_wins[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_wins[(4) + 3]))
                            : "r"(warp_hist_addr + (unsigned int)((cand_base + 32 + 4) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_wins[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_wins[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_wins[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_wins[(8) + 3]))
                            : "r"(warp_hist_addr + (unsigned int)((cand_base + 32 + 8) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_wins[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_wins[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_wins[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_wins[(12) + 3]))
                            : "r"(warp_hist_addr + (unsigned int)((cand_base + 32 + 12) * 4)));
                        q_sa += ((q_win_id > (int)q_wins[0]) ? 1 : 0);
                        q_sb += ((q_win_id > (int)q_wins[1]) ? 1 : 0);
                        q_sc += ((q_win_id > (int)q_wins[2]) ? 1 : 0);
                        q_sd += ((q_win_id > (int)q_wins[3]) ? 1 : 0);
                        q_sa += ((q_win_id > (int)q_wins[4]) ? 1 : 0);
                        q_sb += ((q_win_id > (int)q_wins[5]) ? 1 : 0);
                        q_sc += ((q_win_id > (int)q_wins[6]) ? 1 : 0);
                        q_sd += ((q_win_id > (int)q_wins[7]) ? 1 : 0);
                        q_sa += ((q_win_id > (int)q_wins[8]) ? 1 : 0);
                        q_sb += ((q_win_id > (int)q_wins[9]) ? 1 : 0);
                        q_sc += ((q_win_id > (int)q_wins[10]) ? 1 : 0);
                        q_sd += ((q_win_id > (int)q_wins[11]) ? 1 : 0);
                        q_sa += ((q_win_id > (int)q_wins[12]) ? 1 : 0);
                        q_sb += ((q_win_id > (int)q_wins[13]) ? 1 : 0);
                        q_sc += ((q_win_id > (int)q_wins[14]) ? 1 : 0);
                        q_sd += ((q_win_id > (int)q_wins[15]) ? 1 : 0);
                        int q_win_slot = q_sa + q_sb + (q_sc + q_sd);
                        winner_ids[stash_base + q_win_slot] = q_win_id;
                        winner_scores[stash_base + q_win_slot] = q_win_score;
                    }
                    __syncwarp();
                }
            }
            if (use_radix != 0) {
                int remaining = TOP_K;
                int active_total = NUM_EXPERTS;
                unsigned int threshold = 0;
                unsigned int examined_mask = 0;
                int take_all_equals = 0;
                unsigned int lane_top = 0;
                #pragma unroll
                for (int item_top = 0; item_top < 28; item_top++) {
                    unsigned int top_digit = keys[item_top] >> 21;
                    if (top_digit > lane_top) {
                        lane_top = top_digit;
                    }
                }
                unsigned int warp_top;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(warp_top) : "r"(lane_top));
                unsigned int lane_top_count = 0;
                #pragma unroll
                for (int item_cnt = 0; item_cnt < 28; item_cnt++) {
                    if (keys[item_cnt] >> 21 == warp_top) {
                        lane_top_count += 1;
                    }
                }
                unsigned int warp_top_count;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(warp_top_count) : "r"(lane_top_count));
                int shift = 13;
                int spec_round = 0;
                if (warp_top_count >= (unsigned int)TOP_K) {
                    threshold = warp_top << 21;
                    examined_mask = (unsigned int)2047 << 21;
                    active_total = (int)warp_top_count;
                    if (remaining == active_total) {
                        take_all_equals = 1;
                    }
                } else {
                    threshold = warp_top >> 3 << 24;
                    examined_mask = (unsigned int)255 << 24;
                    shift = 16;
                    spec_round = 1;
                }
                if (take_all_equals == 0) {
                    #pragma unroll 1
                    for (int radix_round = 0; radix_round < 5; radix_round++) {
                        __syncwarp();
                        #pragma unroll
                        for (int clear_slot = 0; clear_slot < 8; clear_slot++) {
                            warp_hist[lane_hist_base + clear_slot] = 0;
                        }
                        __syncwarp();
                        #pragma unroll
                        for (int item_hist = 0; item_hist < 28; item_hist++) {
                            if ((keys[item_hist] & examined_mask) == threshold) {
                                int digit = (int)(keys[item_hist] >> (unsigned int)shift & 255);
                                atomicAdd(&warp_hist[hist_base + digit + (digit >> 3)], 1);
                            }
                        }
                        __syncwarp();
                        int scan_counts[8];
                        int lane_count = 0;
                        #pragma unroll
                        for (int scan_offset = 0; scan_offset < 8; scan_offset++) {
                            int scan_count = warp_hist[lane_hist_base + scan_offset];
                            scan_counts[scan_offset] = scan_count;
                            lane_count += scan_count;
                        }
                        int inclusive = lane_count;
                        int scan_peer_4_0 = __shfl_up_sync(0xFFFFFFFF, inclusive, 1, 32);
                        if (lane >= 1) {
                            inclusive += scan_peer_4_0;
                        }
                        int scan_peer_4_1 = __shfl_up_sync(0xFFFFFFFF, inclusive, 2, 32);
                        if (lane >= 2) {
                            inclusive += scan_peer_4_1;
                        }
                        int scan_peer_4_2 = __shfl_up_sync(0xFFFFFFFF, inclusive, 4, 32);
                        if (lane >= 4) {
                            inclusive += scan_peer_4_2;
                        }
                        int scan_peer_4_3 = __shfl_up_sync(0xFFFFFFFF, inclusive, 8, 32);
                        if (lane >= 8) {
                            inclusive += scan_peer_4_3;
                        }
                        int scan_peer_4_4 = __shfl_up_sync(0xFFFFFFFF, inclusive, 16, 32);
                        if (lane >= 16) {
                            inclusive += scan_peer_4_4;
                        }
                        int consume_round = 1;
                        if (spec_round != 0) {
                            spec_round = 0;
                            int _shfl_sync_29 = __shfl_sync(0xFFFFFFFF, inclusive, 31);
                            active_total = _shfl_sync_29;
                            if (active_total < TOP_K) {
                                threshold = (unsigned int)0;
                                examined_mask = (unsigned int)0;
                                active_total = NUM_EXPERTS;
                                shift = 24;
                                consume_round = 0;
                            }
                        }
                        if (consume_round != 0) {
                            int need = active_total - remaining;
                            int lane_excl = inclusive - lane_count;
                            int run_prefix = lane_excl;
                            int bin_below = lane_excl;
                            int bin_top = inclusive;
                            int bins_before = 0;
                            #pragma unroll
                            for (int scan_offset_1 = 0; scan_offset_1 < 8; scan_offset_1++) {
                                int bin_count = scan_counts[scan_offset_1];
                                run_prefix += bin_count;
                                if (run_prefix <= need) {
                                    bin_below = run_prefix;
                                    bins_before += 1;
                                } else {
                                    int _min_30 = ((bin_top) < (run_prefix) ? (bin_top) : (run_prefix));
                                    bin_top = _min_30;
                                }
                            }
                            unsigned int warp_match_record = 0;
                            if (lane_excl <= need && need < inclusive) {
                                int chosen_scan_bin = scan_bin_base + bins_before;
                                int bin_above = active_total - bin_top;
                                int chosen_count = bin_top - bin_below;
                                warp_match_record = (unsigned int)(chosen_scan_bin & 255) | (unsigned int)(bin_above & 1023) << 8 | (unsigned int)(chosen_count & 1023) << 18;
                            }
                            unsigned int match_record;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(match_record) : "r"(warp_match_record));
                            int chosen_bin = (int)(match_record & 255);
                            remaining -= (int)(match_record >> 8 & 1023);
                            active_total = (int)(match_record >> 18 & 1023);
                            threshold |= (unsigned int)chosen_bin << (unsigned int)shift;
                            examined_mask |= (unsigned int)255 << (unsigned int)shift;
                            if (remaining == active_total) {
                                take_all_equals = 1;
                                break;
                            }
                            if (shift == 0) {
                                break;
                            }
                            shift -= 8;
                            if (shift < 0) {
                                shift = 0;
                            }
                        }
                    }
                }
                unsigned int selected_mask = 0;
                if (take_all_equals != 0) {
                    #pragma unroll
                    for (int item_fast = 0; item_fast < 28; item_fast++) {
                        if (threshold <= (keys[item_fast] & examined_mask)) {
                            selected_mask |= one_u32 << (unsigned int)item_fast;
                        }
                    }
                } else {
                    int tie_prior = 0;
                    #pragma unroll
                    for (int item_tie = 0; item_tie < 28; item_tie++) {
                        int tie_active = 0;
                        if ((keys[item_tie] & examined_mask) == threshold) {
                            tie_active = 1;
                        }
                        unsigned int tie_ballot = __ballot_sync(0xFFFFFFFF, tie_active != 0);
                        int tie_rank = tie_prior + (int)(unsigned int)__popc(tie_ballot & lanemask_lt);
                        if (threshold < (keys[item_tie] & examined_mask)) {
                            selected_mask |= one_u32 << (unsigned int)item_tie;
                        }
                        if (tie_active != 0) {
                            if (tie_rank < remaining) {
                                selected_mask |= one_u32 << (unsigned int)item_tie;
                            }
                        }
                        tie_prior += (int)(unsigned int)__popc(tie_ballot);
                    }
                }
                int compact_prior = 0;
                #pragma unroll
                for (int item_compact = 0; item_compact < 28; item_compact++) {
                    int chosen = (int)(selected_mask >> (unsigned int)item_compact & 1);
                    unsigned int compact_ballot = __ballot_sync(0xFFFFFFFF, chosen != 0);
                    if (chosen != 0) {
                        int slot = compact_prior + (int)(unsigned int)__popc(compact_ballot & lanemask_lt);
                        int winner_col = item_compact * 32 + lane;
                        winner_ids[stash_base + slot] = winner_col;
                    }
                    compact_prior += (int)(unsigned int)__popc(compact_ballot);
                }
                __syncwarp();
                if (lane < TOP_K) {
                    int score_col = winner_ids[stash_base + lane];
                    float score_logit = reinterpret_cast<volatile float*>(logits)[row_base + (unsigned long long)score_col];
                    float _exp_32 = __expf(-score_logit);
                    float _fdiv_31;
                    asm volatile("div.full.f32 %0, %1, %2;" : "=f"(_fdiv_31) : "f"(1.0f), "f"(1.0f + _exp_32));
                    winner_scores[stash_base + lane] = _fdiv_31;
                }
                __syncwarp();
            }
        }
        if (it == 0) {
            cooperative_groups::this_grid().barrier_wait(static_cast<cooperative_groups::grid_group::arrival_token>(join_token));
            __syncthreads();
        }
        if (token < M) {
            if (lane < TOP_K) {
                int selected_id = winner_ids[stash_base + lane];
                unsigned long long output_index = (unsigned long long)token * (unsigned long long)TOP_K + (unsigned long long)lane;
                float selected_sum = 0.0f;
                #pragma unroll
                for (int route_sum = 0; route_sum < TOP_K; route_sum++) {
                    selected_sum += winner_scores[stash_base + route_sum];
                }
                float selected_norm = 1.0f;
                if (selected_sum > 0.0f) {
                    selected_norm = selected_sum;
                }
                float _fdiv_33;
                asm volatile("div.full.f32 %0, %1, %2;" : "=f"(_fdiv_33) : "f"(winner_scores[stash_base + lane]), "f"(selected_norm));
                topk_weights[output_index] = _fdiv_33;
                int local_rank = 0;
                if (M * TOP_K >= NUM_EXPERTS * 8) {
                    int _atomic_fetch_add_34 = atomicAdd(&sorted_token_ids[selected_id * 8], 1);
                    local_rank = _atomic_fetch_add_34;
                } else {
                    int _atomic_fetch_add_35 = atomicAdd(&expert_counts[selected_id], 1);
                    local_rank = _atomic_fetch_add_35;
                }
                unsigned int packed_rank = (unsigned int)local_rank;
                unsigned int packed_expert = (unsigned int)selected_id;
                unsigned int packed_topk = packed_rank << 10 | packed_expert;
                if (M <= num_bids * (NUM_WARPS * 2)) {
                    topk_ids[output_index] = selected_id;
                    own_rec[(it * NUM_WARPS + warp) * TOP_K + lane] = (int)packed_topk;
                } else {
                    topk_ids[output_index] = (int)packed_topk;
                }
            }
            __syncwarp();
        }
        row_token += token_stride;
        it += 1;
    }
    if (n_it == 0) {
        cooperative_groups::this_grid().barrier_wait(static_cast<cooperative_groups::grid_group::arrival_token>(join_token));
        __syncthreads();
    }
    __syncthreads();
    if (tid == 0) {
        unsigned int cj_old;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
            : "=r"(cj_old) : "l"(&reinterpret_cast<unsigned int*>(expert_offsets)[NUM_EXPERTS]), "r"(static_cast<uint32_t>(1)) : "memory");
        warp_hist[8] = (int)cj_old;
    }
    __syncthreads();
    int cj_last = 0;
    if (warp_hist[8] == num_bids - 1) {
        cj_last = 1;
    }
    if (cj_last != 0) {
        if (M * TOP_K >= NUM_EXPERTS * 8) {
            int cj_counts[4];
            #pragma unroll
            for (int cj_slot = 0; cj_slot < ITEMS_PER_THREAD; cj_slot++) {
                cj_counts[cj_slot] = sorted_token_ids[(tid + cj_slot * THREADS) * 8];
            }
            #pragma unroll
            for (int cj_store = 0; cj_store < ITEMS_PER_THREAD; cj_store++) {
                expert_counts[tid + cj_store * THREADS] = cj_counts[cj_store];
            }
        }
        __syncthreads();
        if (tid == 0) {
            unsigned int _atomic_old_0;
            asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                : "=r"(_atomic_old_0) : "l"(&reinterpret_cast<unsigned int*>(expert_offsets)[0]), "r"(static_cast<uint32_t>(1)) : "memory");
        }
    } else {
        if (tid == 0) {
            {
                unsigned int* _global_wait_zero_p_36 = (reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(expert_offsets)) + (0));
                unsigned int _global_wait_zero_v_36;
                do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_global_wait_zero_v_36) : "l"(_global_wait_zero_p_36) : "memory");
                } while (_global_wait_zero_v_36 != 0u);
            }
        }
        __syncthreads();
    }
    int expert_scan_base = warp * 32 * ITEMS_PER_THREAD + lane * ITEMS_PER_THREAD;
    int lane_prefixes[4];
    int lane_total = 0;
    #pragma unroll
    for (int expert_slot_scan = 0; expert_slot_scan < ITEMS_PER_THREAD; expert_slot_scan++) {
        int expert_scan = expert_scan_base + expert_slot_scan;
        int count_scan = expert_counts[expert_scan];
        if (bid == 0) {
            expert_scatter_offsets[expert_scan] = count_scan;
        }
        lane_prefixes[expert_slot_scan] = lane_total;
        int padded_count_scan = count_scan + BLOCK_MASK & ~BLOCK_MASK;
        lane_total += padded_count_scan;
    }
    int warp_inclusive = lane_total;
    int phase2_scan_peer_4_0 = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, 1, 32);
    if (lane >= 1) {
        warp_inclusive += phase2_scan_peer_4_0;
    }
    int phase2_scan_peer_4_1 = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, 2, 32);
    if (lane >= 2) {
        warp_inclusive += phase2_scan_peer_4_1;
    }
    int phase2_scan_peer_4_2 = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, 4, 32);
    if (lane >= 4) {
        warp_inclusive += phase2_scan_peer_4_2;
    }
    int phase2_scan_peer_4_3 = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, 8, 32);
    if (lane >= 8) {
        warp_inclusive += phase2_scan_peer_4_3;
    }
    int phase2_scan_peer_4_4 = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, 16, 32);
    if (lane >= 16) {
        warp_inclusive += phase2_scan_peer_4_4;
    }
    if (lane == 31) {
        phase2_warp_sums[warp] = warp_inclusive;
    }
    __syncthreads();
    int phase2_warp_prefix_lane0 = 0;
    if (lane == 0) {
        #pragma unroll
        for (int prior_scan_warp = 0; prior_scan_warp < NUM_WARPS; prior_scan_warp++) {
            if (prior_scan_warp < warp) {
                phase2_warp_prefix_lane0 += phase2_warp_sums[prior_scan_warp];
            }
        }
    }
    int phase2_warp_prefix = __shfl_sync(0xFFFFFFFF, phase2_warp_prefix_lane0, 0);
    int lane_prefix = phase2_warp_prefix + warp_inclusive - lane_total;
    #pragma unroll
    for (int expert_slot_store = 0; expert_slot_store < ITEMS_PER_THREAD; expert_slot_store++) {
        int expert_store = expert_scan_base + expert_slot_store;
        int offset_store = lane_prefix + lane_prefixes[expert_slot_store];
        local_offsets[expert_store] = offset_store;
        if (bid == 0) {
            expert_offsets[expert_store] = offset_store;
        }
    }
    if (bid == 0) {
        if (tid == THREADS - 1) {
            int padded_total = phase2_warp_prefix + warp_inclusive;
            expert_offsets[NUM_EXPERTS] = padded_total;
            num_tokens_post_padded[0] = padded_total;
        }
    }
    __syncthreads();
    {
        int scatter_global = 1;
        if (M <= num_bids * (NUM_WARPS * 2)) {
            scatter_global = 0;
            int own_slot = tid >> 4;
            int own_iter = 0;
            if (own_slot >= NUM_WARPS) {
                own_iter = 1;
            }
            int own_token = cta_first + own_slot - own_iter * NUM_WARPS + own_iter * token_stride;
            if (own_token < M) {
                unsigned int own_route = (unsigned int)own_rec[tid];
                int own_expert = (int)(own_route & 1023);
                int own_row = (int)(own_route >> 10);
                int own_grouped = local_offsets[own_expert] + own_row;
                sorted_token_ids[own_grouped] = own_token * TOP_K + (tid & 15);
                if ((own_row & BLOCK_MASK) == 0) {
                    expert_ids[own_grouped >> BLOCK_SHIFT] = own_expert;
                }
            }
        }
        if (scatter_global != 0) {
            int scatter_row_group = tid >> 4;
            int scatter_route = tid & 15;
            #pragma unroll 1
            for (int scatter_token = bid * SCATTER_ROW_GROUPS + scatter_row_group; scatter_token < M; scatter_token += num_bids * SCATTER_ROW_GROUPS) {
                int pair = scatter_token * TOP_K + scatter_route;
                unsigned int packed_route = (unsigned int)topk_ids[pair];
                int pair_expert = (int)(packed_route & 1023);
                int local_row = (int)(packed_route >> 10);
                topk_ids[pair] = pair_expert;
                int grouped_row = local_offsets[pair_expert] + local_row;
                sorted_token_ids[grouped_row] = pair;
                if ((local_row & BLOCK_MASK) == 0) {
                    expert_ids[grouped_row >> BLOCK_SHIFT] = pair_expert;
                }
            }
        }
        int padding_group = tid >> BLOCK_SHIFT;
        int padding_lane = tid & BLOCK_MASK;
        #pragma unroll 1
        for (int padding_expert = bid * PADDING_GROUPS + padding_group; padding_expert < NUM_EXPERTS; padding_expert += num_bids * PADDING_GROUPS) {
            int padding_expert_count = expert_counts[padding_expert];
            int padded_expert_count = padding_expert_count + BLOCK_MASK & ~BLOCK_MASK;
            int padding_count = padded_expert_count - padding_expert_count;
            if (padding_lane < padding_count) {
                int padding_start = local_offsets[padding_expert] + padding_expert_count;
                sorted_token_ids[padding_start + padding_lane] = M * TOP_K;
            }
        }
    }
    if (M > 1) {
        int bitmap_words = M + 31 >> 5;
        int cnt_a = 0;
        int cnt_b = 0;
        int st_a = 0;
        int st_b = 0;
        if (bid < 896) {
            cnt_a = expert_counts[bid];
            st_a = local_offsets[bid];
        }
        if (bid + num_bids < 896) {
            cnt_b = expert_counts[bid + num_bids];
            st_b = local_offsets[bid + num_bids];
        }
        int ws_cta = bid;
        int ws_ctas = num_bids;
        if (num_bids * 2 > 896) {
            if (num_bids < 896) {
                ws_cta = bid - (896 - num_bids);
                ws_ctas = num_bids * 2 - 896;
            }
        }
        int ws_first = 896;
        if (ws_cta >= 0) {
            ws_first = ws_cta * 7 + warp;
        }
        if (num_bids * 2 >= 896) {
            ws_first = 896;
        }
        int ws_cnt = 0;
        if (ws_first < 896) {
            ws_cnt = expert_counts[ws_first];
        }
        #pragma unroll 1
        for (int clear_word = tid; clear_word < bitmap_words; clear_word += 224) {
            bitmap_a[clear_word] = (unsigned int)0;
            bitmap_b[clear_word] = (unsigned int)0;
        }
        cooperative_groups::this_grid().sync();
        if (bid >= 896) {
            int ws_count = ws_cnt;
            #pragma unroll 1
            for (int warp_sort_expert = ws_first; warp_sort_expert < 896; warp_sort_expert += ws_ctas * 7) {
                int warp_sort_count = ws_count;
                if (warp_sort_count > 1) {
                    if (warp_sort_count <= 32) {
                        int warp_sort_start = local_offsets[warp_sort_expert];
                        int warp_sort_value = M * 16;
                        if (warp_sort_count > lane) {
                            warp_sort_value = sorted_token_ids[warp_sort_start + lane];
                        }
                        int warp_sort_peer_1_0 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 1);
                        int warp_sort_take_max_2_0 = (lane >> 1 ^ lane) & 1;
                        if (warp_sort_take_max_2_0 == 0) {
                            if (warp_sort_peer_1_0 < warp_sort_value) {
                                warp_sort_value = warp_sort_peer_1_0;
                            }
                        } else if (warp_sort_peer_1_0 > warp_sort_value) {
                            warp_sort_value = warp_sort_peer_1_0;
                        }
                        int warp_sort_peer_1_1 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 2);
                        int warp_sort_take_max_2_1 = (lane >> 2 ^ lane >> 1) & 1;
                        if (warp_sort_take_max_2_1 == 0) {
                            if (warp_sort_peer_1_1 < warp_sort_value) {
                                warp_sort_value = warp_sort_peer_1_1;
                            }
                        } else if (warp_sort_peer_1_1 > warp_sort_value) {
                            warp_sort_value = warp_sort_peer_1_1;
                        }
                        int warp_sort_peer_1_2 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 1);
                        int warp_sort_take_max_2_2 = (lane >> 2 ^ lane) & 1;
                        if (warp_sort_take_max_2_2 == 0) {
                            if (warp_sort_peer_1_2 < warp_sort_value) {
                                warp_sort_value = warp_sort_peer_1_2;
                            }
                        } else if (warp_sort_peer_1_2 > warp_sort_value) {
                            warp_sort_value = warp_sort_peer_1_2;
                        }
                        int warp_sort_peer_1_3 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 4);
                        int warp_sort_take_max_2_3 = (lane >> 3 ^ lane >> 2) & 1;
                        if (warp_sort_take_max_2_3 == 0) {
                            if (warp_sort_peer_1_3 < warp_sort_value) {
                                warp_sort_value = warp_sort_peer_1_3;
                            }
                        } else if (warp_sort_peer_1_3 > warp_sort_value) {
                            warp_sort_value = warp_sort_peer_1_3;
                        }
                        int warp_sort_peer_1_4 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 2);
                        int warp_sort_take_max_2_4 = (lane >> 3 ^ lane >> 1) & 1;
                        if (warp_sort_take_max_2_4 == 0) {
                            if (warp_sort_peer_1_4 < warp_sort_value) {
                                warp_sort_value = warp_sort_peer_1_4;
                            }
                        } else if (warp_sort_peer_1_4 > warp_sort_value) {
                            warp_sort_value = warp_sort_peer_1_4;
                        }
                        int warp_sort_peer_1_5 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 1);
                        int warp_sort_take_max_2_5 = (lane >> 3 ^ lane) & 1;
                        if (warp_sort_take_max_2_5 == 0) {
                            if (warp_sort_peer_1_5 < warp_sort_value) {
                                warp_sort_value = warp_sort_peer_1_5;
                            }
                        } else if (warp_sort_peer_1_5 > warp_sort_value) {
                            warp_sort_value = warp_sort_peer_1_5;
                        }
                        int warp_sort_peer_1_6 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 8);
                        int warp_sort_take_max_2_6 = (lane >> 4 ^ lane >> 3) & 1;
                        if (warp_sort_take_max_2_6 == 0) {
                            if (warp_sort_peer_1_6 < warp_sort_value) {
                                warp_sort_value = warp_sort_peer_1_6;
                            }
                        } else if (warp_sort_peer_1_6 > warp_sort_value) {
                            warp_sort_value = warp_sort_peer_1_6;
                        }
                        int warp_sort_peer_1_7 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 4);
                        int warp_sort_take_max_2_7 = (lane >> 4 ^ lane >> 2) & 1;
                        if (warp_sort_take_max_2_7 == 0) {
                            if (warp_sort_peer_1_7 < warp_sort_value) {
                                warp_sort_value = warp_sort_peer_1_7;
                            }
                        } else if (warp_sort_peer_1_7 > warp_sort_value) {
                            warp_sort_value = warp_sort_peer_1_7;
                        }
                        int warp_sort_peer_1_8 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 2);
                        int warp_sort_take_max_2_8 = (lane >> 4 ^ lane >> 1) & 1;
                        if (warp_sort_take_max_2_8 == 0) {
                            if (warp_sort_peer_1_8 < warp_sort_value) {
                                warp_sort_value = warp_sort_peer_1_8;
                            }
                        } else if (warp_sort_peer_1_8 > warp_sort_value) {
                            warp_sort_value = warp_sort_peer_1_8;
                        }
                        int warp_sort_peer_1_9 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 1);
                        int warp_sort_take_max_2_9 = (lane >> 4 ^ lane) & 1;
                        if (warp_sort_take_max_2_9 == 0) {
                            if (warp_sort_peer_1_9 < warp_sort_value) {
                                warp_sort_value = warp_sort_peer_1_9;
                            }
                        } else if (warp_sort_peer_1_9 > warp_sort_value) {
                            warp_sort_value = warp_sort_peer_1_9;
                        }
                        int warp_sort_peer_1_10 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 16);
                        int warp_sort_take_max_2_10 = (lane >> 5 ^ lane >> 4) & 1;
                        if (warp_sort_take_max_2_10 == 0) {
                            if (warp_sort_peer_1_10 < warp_sort_value) {
                                warp_sort_value = warp_sort_peer_1_10;
                            }
                        } else if (warp_sort_peer_1_10 > warp_sort_value) {
                            warp_sort_value = warp_sort_peer_1_10;
                        }
                        int warp_sort_peer_1_11 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 8);
                        int warp_sort_take_max_2_11 = (lane >> 5 ^ lane >> 3) & 1;
                        if (warp_sort_take_max_2_11 == 0) {
                            if (warp_sort_peer_1_11 < warp_sort_value) {
                                warp_sort_value = warp_sort_peer_1_11;
                            }
                        } else if (warp_sort_peer_1_11 > warp_sort_value) {
                            warp_sort_value = warp_sort_peer_1_11;
                        }
                        int warp_sort_peer_1_12 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 4);
                        int warp_sort_take_max_2_12 = (lane >> 5 ^ lane >> 2) & 1;
                        if (warp_sort_take_max_2_12 == 0) {
                            if (warp_sort_peer_1_12 < warp_sort_value) {
                                warp_sort_value = warp_sort_peer_1_12;
                            }
                        } else if (warp_sort_peer_1_12 > warp_sort_value) {
                            warp_sort_value = warp_sort_peer_1_12;
                        }
                        int warp_sort_peer_1_13 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 2);
                        int warp_sort_take_max_2_13 = (lane >> 5 ^ lane >> 1) & 1;
                        if (warp_sort_take_max_2_13 == 0) {
                            if (warp_sort_peer_1_13 < warp_sort_value) {
                                warp_sort_value = warp_sort_peer_1_13;
                            }
                        } else if (warp_sort_peer_1_13 > warp_sort_value) {
                            warp_sort_value = warp_sort_peer_1_13;
                        }
                        int warp_sort_peer_1_14 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 1);
                        int warp_sort_take_max_2_14 = (lane >> 5 ^ lane) & 1;
                        if (warp_sort_take_max_2_14 == 0) {
                            if (warp_sort_peer_1_14 < warp_sort_value) {
                                warp_sort_value = warp_sort_peer_1_14;
                            }
                        } else if (warp_sort_peer_1_14 > warp_sort_value) {
                            warp_sort_value = warp_sort_peer_1_14;
                        }
                        if (warp_sort_count > lane) {
                            sorted_token_ids[warp_sort_start + lane] = warp_sort_value;
                        }
                    }
                }
                int ws_next = warp_sort_expert + ws_ctas * 7;
                if (ws_next < 896) {
                    ws_count = expert_counts[ws_next];
                }
            }
        }
        #pragma unroll 1
        for (int sort_expert_a = bid; sort_expert_a < 896; sort_expert_a += num_bids * 2) {
            int eff_a = cnt_a;
            int eff_b = cnt_b;
            if (num_bids * 2 < 896) {
                if (eff_a <= 32) {
                    eff_a = 0;
                }
                if (eff_b <= 32) {
                    eff_b = 0;
                }
            } else {
                if (eff_a <= 1) {
                    eff_a = 0;
                }
                if (eff_b <= 1) {
                    eff_b = 0;
                }
            }
            int r_fast = 0;
            int r_wide = 0;
            int keep_a[8];
            int keep_b[8];
            int o_max = ((eff_a) > (eff_b) ? (eff_a) : (eff_b));
            if (o_max > 0) {
                if (o_max <= 1792) {
                    r_fast = 1;
                }
            }
            if (o_max > 448) {
                r_wide = 1;
            }
            #pragma unroll
            for (int o_init = 0; o_init < 8; o_init++) {
                keep_a[o_init] = 0;
                keep_b[o_init] = 0;
            }
            if (r_fast != 0) {
                #pragma unroll
                for (int o_lo = 0; o_lo < 2; o_lo++) {
                    if (eff_a > tid + o_lo * 224) {
                        keep_a[o_lo] = sorted_token_ids[st_a + tid + o_lo * 224];
                    }
                    if (eff_b > tid + o_lo * 224) {
                        keep_b[o_lo] = sorted_token_ids[st_b + tid + o_lo * 224];
                    }
                }
                if (r_wide != 0) {
                    #pragma unroll
                    for (int o_hi = 2; o_hi < 8; o_hi++) {
                        if (eff_a > tid + o_hi * 224) {
                            keep_a[o_hi] = sorted_token_ids[st_a + tid + o_hi * 224];
                        }
                        if (eff_b > tid + o_hi * 224) {
                            keep_b[o_hi] = sorted_token_ids[st_b + tid + o_hi * 224];
                        }
                    }
                }
            }
            if (sort_expert_a == bid) {
                int ws_count_1 = ws_cnt;
                #pragma unroll 1
                for (int warp_sort_expert_1 = ws_first; warp_sort_expert_1 < 896; warp_sort_expert_1 += ws_ctas * 7) {
                    int warp_sort_count_1 = ws_count_1;
                    if (warp_sort_count_1 > 1) {
                        if (warp_sort_count_1 <= 32) {
                            int warp_sort_start_1 = local_offsets[warp_sort_expert_1];
                            int warp_sort_value_1 = M * 16;
                            if (warp_sort_count_1 > lane) {
                                warp_sort_value_1 = sorted_token_ids[warp_sort_start_1 + lane];
                            }
                            int warp_sort_peer_1_15 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value_1, 1);
                            int warp_sort_take_max_2_15 = (lane >> 1 ^ lane) & 1;
                            if (warp_sort_take_max_2_15 == 0) {
                                if (warp_sort_peer_1_15 < warp_sort_value_1) {
                                    warp_sort_value_1 = warp_sort_peer_1_15;
                                }
                            } else if (warp_sort_peer_1_15 > warp_sort_value_1) {
                                warp_sort_value_1 = warp_sort_peer_1_15;
                            }
                            int warp_sort_peer_1_16 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value_1, 2);
                            int warp_sort_take_max_2_16 = (lane >> 2 ^ lane >> 1) & 1;
                            if (warp_sort_take_max_2_16 == 0) {
                                if (warp_sort_peer_1_16 < warp_sort_value_1) {
                                    warp_sort_value_1 = warp_sort_peer_1_16;
                                }
                            } else if (warp_sort_peer_1_16 > warp_sort_value_1) {
                                warp_sort_value_1 = warp_sort_peer_1_16;
                            }
                            int warp_sort_peer_1_17 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value_1, 1);
                            int warp_sort_take_max_2_17 = (lane >> 2 ^ lane) & 1;
                            if (warp_sort_take_max_2_17 == 0) {
                                if (warp_sort_peer_1_17 < warp_sort_value_1) {
                                    warp_sort_value_1 = warp_sort_peer_1_17;
                                }
                            } else if (warp_sort_peer_1_17 > warp_sort_value_1) {
                                warp_sort_value_1 = warp_sort_peer_1_17;
                            }
                            int warp_sort_peer_1_18 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value_1, 4);
                            int warp_sort_take_max_2_18 = (lane >> 3 ^ lane >> 2) & 1;
                            if (warp_sort_take_max_2_18 == 0) {
                                if (warp_sort_peer_1_18 < warp_sort_value_1) {
                                    warp_sort_value_1 = warp_sort_peer_1_18;
                                }
                            } else if (warp_sort_peer_1_18 > warp_sort_value_1) {
                                warp_sort_value_1 = warp_sort_peer_1_18;
                            }
                            int warp_sort_peer_1_19 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value_1, 2);
                            int warp_sort_take_max_2_19 = (lane >> 3 ^ lane >> 1) & 1;
                            if (warp_sort_take_max_2_19 == 0) {
                                if (warp_sort_peer_1_19 < warp_sort_value_1) {
                                    warp_sort_value_1 = warp_sort_peer_1_19;
                                }
                            } else if (warp_sort_peer_1_19 > warp_sort_value_1) {
                                warp_sort_value_1 = warp_sort_peer_1_19;
                            }
                            int warp_sort_peer_1_20 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value_1, 1);
                            int warp_sort_take_max_2_20 = (lane >> 3 ^ lane) & 1;
                            if (warp_sort_take_max_2_20 == 0) {
                                if (warp_sort_peer_1_20 < warp_sort_value_1) {
                                    warp_sort_value_1 = warp_sort_peer_1_20;
                                }
                            } else if (warp_sort_peer_1_20 > warp_sort_value_1) {
                                warp_sort_value_1 = warp_sort_peer_1_20;
                            }
                            int warp_sort_peer_1_21 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value_1, 8);
                            int warp_sort_take_max_2_21 = (lane >> 4 ^ lane >> 3) & 1;
                            if (warp_sort_take_max_2_21 == 0) {
                                if (warp_sort_peer_1_21 < warp_sort_value_1) {
                                    warp_sort_value_1 = warp_sort_peer_1_21;
                                }
                            } else if (warp_sort_peer_1_21 > warp_sort_value_1) {
                                warp_sort_value_1 = warp_sort_peer_1_21;
                            }
                            int warp_sort_peer_1_22 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value_1, 4);
                            int warp_sort_take_max_2_22 = (lane >> 4 ^ lane >> 2) & 1;
                            if (warp_sort_take_max_2_22 == 0) {
                                if (warp_sort_peer_1_22 < warp_sort_value_1) {
                                    warp_sort_value_1 = warp_sort_peer_1_22;
                                }
                            } else if (warp_sort_peer_1_22 > warp_sort_value_1) {
                                warp_sort_value_1 = warp_sort_peer_1_22;
                            }
                            int warp_sort_peer_1_23 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value_1, 2);
                            int warp_sort_take_max_2_23 = (lane >> 4 ^ lane >> 1) & 1;
                            if (warp_sort_take_max_2_23 == 0) {
                                if (warp_sort_peer_1_23 < warp_sort_value_1) {
                                    warp_sort_value_1 = warp_sort_peer_1_23;
                                }
                            } else if (warp_sort_peer_1_23 > warp_sort_value_1) {
                                warp_sort_value_1 = warp_sort_peer_1_23;
                            }
                            int warp_sort_peer_1_24 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value_1, 1);
                            int warp_sort_take_max_2_24 = (lane >> 4 ^ lane) & 1;
                            if (warp_sort_take_max_2_24 == 0) {
                                if (warp_sort_peer_1_24 < warp_sort_value_1) {
                                    warp_sort_value_1 = warp_sort_peer_1_24;
                                }
                            } else if (warp_sort_peer_1_24 > warp_sort_value_1) {
                                warp_sort_value_1 = warp_sort_peer_1_24;
                            }
                            int warp_sort_peer_1_25 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value_1, 16);
                            int warp_sort_take_max_2_25 = (lane >> 5 ^ lane >> 4) & 1;
                            if (warp_sort_take_max_2_25 == 0) {
                                if (warp_sort_peer_1_25 < warp_sort_value_1) {
                                    warp_sort_value_1 = warp_sort_peer_1_25;
                                }
                            } else if (warp_sort_peer_1_25 > warp_sort_value_1) {
                                warp_sort_value_1 = warp_sort_peer_1_25;
                            }
                            int warp_sort_peer_1_26 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value_1, 8);
                            int warp_sort_take_max_2_26 = (lane >> 5 ^ lane >> 3) & 1;
                            if (warp_sort_take_max_2_26 == 0) {
                                if (warp_sort_peer_1_26 < warp_sort_value_1) {
                                    warp_sort_value_1 = warp_sort_peer_1_26;
                                }
                            } else if (warp_sort_peer_1_26 > warp_sort_value_1) {
                                warp_sort_value_1 = warp_sort_peer_1_26;
                            }
                            int warp_sort_peer_1_27 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value_1, 4);
                            int warp_sort_take_max_2_27 = (lane >> 5 ^ lane >> 2) & 1;
                            if (warp_sort_take_max_2_27 == 0) {
                                if (warp_sort_peer_1_27 < warp_sort_value_1) {
                                    warp_sort_value_1 = warp_sort_peer_1_27;
                                }
                            } else if (warp_sort_peer_1_27 > warp_sort_value_1) {
                                warp_sort_value_1 = warp_sort_peer_1_27;
                            }
                            int warp_sort_peer_1_28 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value_1, 2);
                            int warp_sort_take_max_2_28 = (lane >> 5 ^ lane >> 1) & 1;
                            if (warp_sort_take_max_2_28 == 0) {
                                if (warp_sort_peer_1_28 < warp_sort_value_1) {
                                    warp_sort_value_1 = warp_sort_peer_1_28;
                                }
                            } else if (warp_sort_peer_1_28 > warp_sort_value_1) {
                                warp_sort_value_1 = warp_sort_peer_1_28;
                            }
                            int warp_sort_peer_1_29 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value_1, 1);
                            int warp_sort_take_max_2_29 = (lane >> 5 ^ lane) & 1;
                            if (warp_sort_take_max_2_29 == 0) {
                                if (warp_sort_peer_1_29 < warp_sort_value_1) {
                                    warp_sort_value_1 = warp_sort_peer_1_29;
                                }
                            } else if (warp_sort_peer_1_29 > warp_sort_value_1) {
                                warp_sort_value_1 = warp_sort_peer_1_29;
                            }
                            if (warp_sort_count_1 > lane) {
                                sorted_token_ids[warp_sort_start_1 + lane] = warp_sort_value_1;
                            }
                        }
                    }
                    int ws_next_1 = warp_sort_expert_1 + ws_ctas * 7;
                    if (ws_next_1 < 896) {
                        ws_count_1 = expert_counts[ws_next_1];
                    }
                }
            }
            if (eff_a + eff_b > 0) {
                int eff_max = ((eff_a) > (eff_b) ? (eff_a) : (eff_b));
                if (r_fast != 0) {
                    #pragma unroll
                    for (int i_lo = 0; i_lo < 2; i_lo++) {
                        if (eff_a > tid + i_lo * 224) {
                            int token_1 = keep_a[i_lo] >> 4;
                            atomicAdd(&bitmap_a[token_1 >> 5], (unsigned int)1 << (unsigned int)(token_1 & 31));
                        }
                        if (eff_b > tid + i_lo * 224) {
                            int token_2 = keep_b[i_lo] >> 4;
                            atomicAdd(&bitmap_b[token_2 >> 5], (unsigned int)1 << (unsigned int)(token_2 & 31));
                        }
                    }
                    if (r_wide != 0) {
                        #pragma unroll
                        for (int i_hi = 2; i_hi < 8; i_hi++) {
                            if (eff_a > tid + i_hi * 224) {
                                int token_3 = keep_a[i_hi] >> 4;
                                atomicAdd(&bitmap_a[token_3 >> 5], (unsigned int)1 << (unsigned int)(token_3 & 31));
                            }
                            if (eff_b > tid + i_hi * 224) {
                                int token_4 = keep_b[i_hi] >> 4;
                                atomicAdd(&bitmap_b[token_4 >> 5], (unsigned int)1 << (unsigned int)(token_4 & 31));
                            }
                        }
                    }
                }
                if (r_fast == 0) {
                    #pragma unroll 1
                    for (int input_v = tid; input_v < eff_max; input_v += 1792) {
                        int vec_a[8];
                        int vec_b[8];
                        #pragma unroll
                        for (int v_load = 0; v_load < 8; v_load++) {
                            vec_a[v_load] = 0;
                            vec_b[v_load] = 0;
                            if (eff_a > input_v + v_load * 224) {
                                vec_a[v_load] = sorted_token_ids[st_a + input_v + v_load * 224];
                            }
                            if (eff_b > input_v + v_load * 224) {
                                vec_b[v_load] = sorted_token_ids[st_b + input_v + v_load * 224];
                            }
                        }
                        #pragma unroll
                        for (int v_ins = 0; v_ins < 8; v_ins++) {
                            if (eff_a > input_v + v_ins * 224) {
                                int token_5 = vec_a[v_ins] >> 4;
                                atomicAdd(&bitmap_a[token_5 >> 5], (unsigned int)1 << (unsigned int)(token_5 & 31));
                                route_a[token_5] = (uint8_t)(vec_a[v_ins] & 15);
                            }
                            if (eff_b > input_v + v_ins * 224) {
                                int token_6 = vec_b[v_ins] >> 4;
                                atomicAdd(&bitmap_b[token_6 >> 5], (unsigned int)1 << (unsigned int)(token_6 & 31));
                                route_b[token_6] = (uint8_t)(vec_b[v_ins] & 15);
                            }
                        }
                    }
                }
                __syncthreads();
                int word0 = tid * 2;
                unsigned int wa0 = 0;
                unsigned int wa1 = 0;
                unsigned int wb0 = 0;
                unsigned int wb1 = 0;
                if (word0 < bitmap_words) {
                    wa0 = bitmap_a[word0];
                    wb0 = bitmap_b[word0];
                }
                if (bitmap_words > word0 + 1) {
                    wa1 = bitmap_a[word0 + 1];
                    wb1 = bitmap_b[word0 + 1];
                }
                int ca0 = (int)(unsigned int)__popc(wa0);
                int cb0 = (int)(unsigned int)__popc(wb0);
                int ca = ca0 + (int)(unsigned int)__popc(wa1);
                int cb = cb0 + (int)(unsigned int)__popc(wb1);
                int inc_a = ca;
                int inc_b = cb;
                int up_a_1_0 = __shfl_up_sync(0xFFFFFFFF, inc_a, 1, 32);
                int up_b_2_0 = __shfl_up_sync(0xFFFFFFFF, inc_b, 1, 32);
                if (lane >= 1) {
                    inc_a += up_a_1_0;
                    inc_b += up_b_2_0;
                }
                int up_a_1_1 = __shfl_up_sync(0xFFFFFFFF, inc_a, 2, 32);
                int up_b_2_1 = __shfl_up_sync(0xFFFFFFFF, inc_b, 2, 32);
                if (lane >= 2) {
                    inc_a += up_a_1_1;
                    inc_b += up_b_2_1;
                }
                int up_a_1_2 = __shfl_up_sync(0xFFFFFFFF, inc_a, 4, 32);
                int up_b_2_2 = __shfl_up_sync(0xFFFFFFFF, inc_b, 4, 32);
                if (lane >= 4) {
                    inc_a += up_a_1_2;
                    inc_b += up_b_2_2;
                }
                int up_a_1_3 = __shfl_up_sync(0xFFFFFFFF, inc_a, 8, 32);
                int up_b_2_3 = __shfl_up_sync(0xFFFFFFFF, inc_b, 8, 32);
                if (lane >= 8) {
                    inc_a += up_a_1_3;
                    inc_b += up_b_2_3;
                }
                int up_a_1_4 = __shfl_up_sync(0xFFFFFFFF, inc_a, 16, 32);
                int up_b_2_4 = __shfl_up_sync(0xFFFFFFFF, inc_b, 16, 32);
                if (lane >= 16) {
                    inc_a += up_a_1_4;
                    inc_b += up_b_2_4;
                }
                if (lane == 31) {
                    totals_a[warp] = inc_a;
                    totals_b[warp] = inc_b;
                }
                __syncthreads();
                int rank_a = st_a + inc_a - ca;
                int rank_b = st_b + inc_b - cb;
                #pragma unroll
                for (int prior_warp = 0; prior_warp < 7; prior_warp++) {
                    if (prior_warp < warp) {
                        rank_a += totals_a[prior_warp];
                        rank_b += totals_b[prior_warp];
                    }
                }
                if (r_fast != 0) {
                    if (word0 < bitmap_words) {
                        wrank_a[word0] = rank_a;
                        wrank_b[word0] = rank_b;
                    }
                    if (bitmap_words > word0 + 1) {
                        wrank_a[word0 + 1] = rank_a + ca0;
                        wrank_b[word0 + 1] = rank_b + cb0;
                    }
                    __syncthreads();
                    #pragma unroll
                    for (int e_lo = 0; e_lo < 2; e_lo++) {
                        if (eff_a > tid + e_lo * 224) {
                            int token_7 = keep_a[e_lo] >> 4;
                            int word = token_7 >> 5;
                            unsigned int below = bitmap_a[word] & ((unsigned int)1 << (unsigned int)(token_7 & 31)) - 1;
                            sorted_token_ids[wrank_a[word] + (int)(unsigned int)__popc(below)] = keep_a[e_lo];
                        }
                        if (eff_b > tid + e_lo * 224) {
                            int token_8 = keep_b[e_lo] >> 4;
                            int word_1 = token_8 >> 5;
                            unsigned int below_1 = bitmap_b[word_1] & ((unsigned int)1 << (unsigned int)(token_8 & 31)) - 1;
                            sorted_token_ids[wrank_b[word_1] + (int)(unsigned int)__popc(below_1)] = keep_b[e_lo];
                        }
                    }
                    if (r_wide != 0) {
                        #pragma unroll
                        for (int e_hi = 2; e_hi < 8; e_hi++) {
                            if (eff_a > tid + e_hi * 224) {
                                int token_9 = keep_a[e_hi] >> 4;
                                int word_2 = token_9 >> 5;
                                unsigned int below_2 = bitmap_a[word_2] & ((unsigned int)1 << (unsigned int)(token_9 & 31)) - 1;
                                sorted_token_ids[wrank_a[word_2] + (int)(unsigned int)__popc(below_2)] = keep_a[e_hi];
                            }
                            if (eff_b > tid + e_hi * 224) {
                                int token_10 = keep_b[e_hi] >> 4;
                                int word_3 = token_10 >> 5;
                                unsigned int below_3 = bitmap_b[word_3] & ((unsigned int)1 << (unsigned int)(token_10 & 31)) - 1;
                                sorted_token_ids[wrank_b[word_3] + (int)(unsigned int)__popc(below_3)] = keep_b[e_hi];
                            }
                        }
                    }
                }
                if (r_fast == 0) {
                    if (eff_a > 0) {
                        unsigned int walk = wa0;
                        int rank = rank_a;
                        while (walk != 0) {
                            int _find_first_set_37 = __ffs((int)walk);
                            int bit = _find_first_set_37 - 1;
                            int token_11 = (word0 << 5) + bit;
                            sorted_token_ids[rank] = token_11 * 16 + (int)route_a[token_11];
                            walk = walk & walk - 1;
                            rank += 1;
                        }
                        unsigned int walk_0 = wa1;
                        int rank_0 = rank_a + ca0;
                        while (walk_0 != 0) {
                            int _find_first_set_38 = __ffs((int)walk_0);
                            int bit_1 = _find_first_set_38 - 1;
                            int token_12 = (word0 + 1 << 5) + bit_1;
                            sorted_token_ids[rank_0] = token_12 * 16 + (int)route_a[token_12];
                            walk_0 = walk_0 & walk_0 - 1;
                            rank_0 += 1;
                        }
                    }
                    if (eff_b > 0) {
                        unsigned int walk_1 = wb0;
                        int rank_1 = rank_b;
                        while (walk_1 != 0) {
                            int _find_first_set_39 = __ffs((int)walk_1);
                            int bit_2 = _find_first_set_39 - 1;
                            int token_13 = (word0 << 5) + bit_2;
                            sorted_token_ids[rank_1] = token_13 * 16 + (int)route_b[token_13];
                            walk_1 = walk_1 & walk_1 - 1;
                            rank_1 += 1;
                        }
                        unsigned int walk_1_1 = wb1;
                        int rank_1_1 = rank_b + cb0;
                        while (walk_1_1 != 0) {
                            int _find_first_set_40 = __ffs((int)walk_1_1);
                            int bit_3 = _find_first_set_40 - 1;
                            int token_14 = (word0 + 1 << 5) + bit_3;
                            sorted_token_ids[rank_1_1] = token_14 * 16 + (int)route_b[token_14];
                            walk_1_1 = walk_1_1 & walk_1_1 - 1;
                            rank_1_1 += 1;
                        }
                    }
                }
                if (sort_expert_a + num_bids * 2 < 896) {
                    __syncthreads();
                    #pragma unroll 1
                    for (int clear_last = tid; clear_last < bitmap_words; clear_last += 224) {
                        bitmap_a[clear_last] = (unsigned int)0;
                        bitmap_b[clear_last] = (unsigned int)0;
                    }
                    __syncthreads();
                }
            }
            int next_a = sort_expert_a + num_bids * 2;
            cnt_a = 0;
            cnt_b = 0;
            if (next_a < 896) {
                cnt_a = expert_counts[next_a];
                st_a = local_offsets[next_a];
            }
            if (next_a + num_bids < 896) {
                cnt_b = expert_counts[next_a + num_bids];
                st_b = local_offsets[next_a + num_bids];
            }
        }
    }
}

} // extern "C"
