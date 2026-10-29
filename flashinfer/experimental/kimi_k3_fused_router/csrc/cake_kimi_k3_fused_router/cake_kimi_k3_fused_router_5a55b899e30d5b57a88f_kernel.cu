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
#define SMEM_BOUND_VALUES_OFF 49408
#define SMEM_BOUND_VALUES_STAGE_BYTES 128
#define SMEM_BOUND_VALUES_STRIDE 128
#define SMEM_CAND_COUNTS_OFF 49536
#define SMEM_CAND_COUNTS_STAGE_BYTES 32
#define SMEM_CAND_COUNTS_STRIDE 32
#define SMEM_CAND_KEYS_OFF 49568
#define SMEM_CAND_KEYS_STAGE_BYTES 896
#define SMEM_CAND_KEYS_STRIDE 896
#define SMEM_CAND_IDS_OFF 50464
#define SMEM_CAND_IDS_STAGE_BYTES 896
#define SMEM_CAND_IDS_STRIDE 896
#define SMEM_CAND_SCORES_OFF 51360
#define SMEM_CAND_SCORES_STAGE_BYTES 896
#define SMEM_CAND_SCORES_STRIDE 896
#define SMEM_RANK_KEYS_OFF 52256
#define SMEM_RANK_KEYS_STAGE_BYTES 128
#define SMEM_RANK_KEYS_STRIDE 128
#define SMEM_HISTOGRAM_OFF 0
#define SMEM_HISTOGRAM_STAGE_BYTES 1024
#define SMEM_HISTOGRAM_STRIDE 1024
#define SMEM_TIE_WARP_SUMS_OFF 1040
#define SMEM_TIE_WARP_SUMS_STAGE_BYTES 28
#define SMEM_TIE_WARP_SUMS_STRIDE 28
#define SMEM_SELECTED_WARP_SUMS_OFF 1072
#define SMEM_SELECTED_WARP_SUMS_STAGE_BYTES 28
#define SMEM_SELECTED_WARP_SUMS_STRIDE 28
#define SMEM_RADIX_MATCH_OFF 1248
#define SMEM_RADIX_MATCH_STAGE_BYTES 8
#define SMEM_RADIX_MATCH_STRIDE 8
#define SMEM_WINNER_IDS_OFF 1120
#define SMEM_WINNER_IDS_STAGE_BYTES 64
#define SMEM_WINNER_IDS_STRIDE 64
#define SMEM_WINNER_SCORES_OFF 1184
#define SMEM_WINNER_SCORES_STAGE_BYTES 64
#define SMEM_WINNER_SCORES_STRIDE 64
#define SMEM_PHASE2_WARP_SUMS_OFF 7168
#define SMEM_PHASE2_WARP_SUMS_STAGE_BYTES 28
#define SMEM_PHASE2_WARP_SUMS_STRIDE 28
#define SMEM_PLAN_COUNTS_OFF 0
#define SMEM_PLAN_COUNTS_STAGE_BYTES 3584
#define SMEM_PLAN_COUNTS_STRIDE 3584
#define SMEM_PLAN_TOTALS_OFF 45824
#define SMEM_PLAN_TOTALS_STAGE_BYTES 3584
#define SMEM_PLAN_TOTALS_STRIDE 3584
#define SMEM_PLAN_OFFSETS_OFF 3584
#define SMEM_PLAN_OFFSETS_STAGE_BYTES 3584
#define SMEM_PLAN_OFFSETS_STRIDE 3584
#define SMEM_OWN_BITMAP_OFF 8192
#define SMEM_OWN_BITMAP_STAGE_BYTES 1792
#define SMEM_OWN_BITMAP_STRIDE 1792
#define SMEM_OWN_ROUTES_OFF 9984
#define SMEM_OWN_ROUTES_STAGE_BYTES 7168
#define SMEM_OWN_ROUTES_STRIDE 7168
#define SMEM_STREAM_RING_OFF 17152
#define SMEM_STREAM_RING_STAGE_BYTES 28672
#define SMEM_STREAM_RING_STRIDE 28672
#define SMEM_RANK_BITS_OFF 49408
#define SMEM_RANK_BITS_STAGE_BYTES 1904
#define SMEM_RANK_BITS_STRIDE 1904
#define SMEM_RANK_BASE_OFF 51328
#define SMEM_RANK_BASE_STAGE_BYTES 112
#define SMEM_RANK_BASE_STRIDE 112
#define SMEM_RANK_HITS_OFF 8192
#define SMEM_RANK_HITS_STAGE_BYTES 8960
#define SMEM_RANK_HITS_STRIDE 8960
#define SMEM_RANK_COUNT_OFF 51456
#define SMEM_RANK_COUNT_STAGE_BYTES 4
#define SMEM_RANK_COUNT_STRIDE 4
#define SMEM_RANK_WORDS_OFF 17152
#define SMEM_RANK_WORDS_STAGE_BYTES 3808
#define SMEM_RANK_WORDS_STRIDE 3808
#define SMEM_W_HIST_OFF 0
#define SMEM_W_HIST_STAGE_BYTES 38528
#define SMEM_W_HIST_STRIDE 38528
#define SMEM_W_F32_OFF 0
#define SMEM_W_F32_STAGE_BYTES 38528
#define SMEM_W_F32_STRIDE 38528
#define SMEM_W_COMP_OFF 0
#define SMEM_W_COMP_STAGE_BYTES 38528
#define SMEM_W_COMP_STRIDE 38528
#define SMEM_W_WIN_IDS_OFF 28672
#define SMEM_W_WIN_IDS_STAGE_BYTES 448
#define SMEM_W_WIN_IDS_STRIDE 448
#define SMEM_W_WIN_SCORES_OFF 29120
#define SMEM_W_WIN_SCORES_STAGE_BYTES 448
#define SMEM_W_WIN_SCORES_STRIDE 448
#define SMEM_W_BIAS_OFF 29568
#define SMEM_W_BIAS_STAGE_BYTES 3584
#define SMEM_W_BIAS_STRIDE 3584
#define SMEM_TOTAL 52480
#define THREADS 224
#ifndef BLOCK_M
#error "BLOCK_M is a downstream specialization of this program; define it on the compile line"
#endif
#define NUM_EXPERTS 896
#define TOP_K 16
#define ITEMS_PER_THREAD 4
#define NUM_WARPS 7
#define RADIX_CLEAR_THREADS 128
#define RADIX_SCAN_THREADS 32
#define RADIX_BINS_PER_LANE 8
#define MAX_BLOCK_M 16
#define OWNER_CTAS 128
#define CLUSTER 4
#define CLUSTER_EXPERTS 28
#define GATHER_LOADS 16
#define GATHER_PASS_PAIRS 3584
#define BITMAP_STRIDE 64
#define ROUTE_STRIDE 256
#define WORD_PASSES 2
#define STAGE_PAIRS 3584
#define STAGE_BYTES 14336
#define COPIES_PER_THREAD 4
#define BLOCK_MASK (BLOCK_M - 1)
#define BLOCK_SHIFT (3 + (BLOCK_M >> 4))
#define SLICE_EXPERTS (NUM_EXPERTS / CLUSTER)
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

__global__ __launch_bounds__(224, LAUNCH_MIN_BLOCKS) __cluster_dims__(4,1,1) void
kernel_cake_kimi_k3_fused_router_5a55b899e30d5b57a88f(float* __restrict__ logits, float* __restrict__ bias, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M)
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
    const unsigned int clusters_x = gridDim.x / 4;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 4;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    unsigned int* bound_values = reinterpret_cast<unsigned int*>(smem_raw + 49408);
    const int bound_values_addr = smem + 49408;
    int* cand_counts = reinterpret_cast<int*>(smem_raw + 49536);
    const int cand_counts_addr = smem + 49536;
    unsigned int* cand_keys = reinterpret_cast<unsigned int*>(smem_raw + 49568);
    const int cand_keys_addr = smem + 49568;
    int* cand_ids = reinterpret_cast<int*>(smem_raw + 50464);
    const int cand_ids_addr = smem + 50464;
    float* cand_scores = reinterpret_cast<float*>(smem_raw + 51360);
    const int cand_scores_addr = smem + 51360;
    unsigned int* rank_keys = reinterpret_cast<unsigned int*>(smem_raw + 52256);
    const int rank_keys_addr = smem + 52256;
    int* histogram = reinterpret_cast<int*>(smem_raw + 0);
    const int histogram_addr = smem + 0;
    int* tie_warp_sums = reinterpret_cast<int*>(smem_raw + 1040);
    const int tie_warp_sums_addr = smem + 1040;
    int* selected_warp_sums = reinterpret_cast<int*>(smem_raw + 1072);
    const int selected_warp_sums_addr = smem + 1072;
    int* radix_match = reinterpret_cast<int*>(smem_raw + 1248);
    const int radix_match_addr = smem + 1248;
    int* winner_ids = reinterpret_cast<int*>(smem_raw + 1120);
    const int winner_ids_addr = smem + 1120;
    float* winner_scores = reinterpret_cast<float*>(smem_raw + 1184);
    const int winner_scores_addr = smem + 1184;
    int* phase2_warp_sums = reinterpret_cast<int*>(smem_raw + 7168);
    const int phase2_warp_sums_addr = smem + 7168;
    int* plan_counts = reinterpret_cast<int*>(smem_raw + 0);
    const int plan_counts_addr = smem + 0;
    int* plan_totals = reinterpret_cast<int*>(smem_raw + 45824);
    const int plan_totals_addr = smem + 45824;
    int* plan_offsets = reinterpret_cast<int*>(smem_raw + 3584);
    const int plan_offsets_addr = smem + 3584;
    unsigned int* own_bitmap = reinterpret_cast<unsigned int*>(smem_raw + 8192);
    const int own_bitmap_addr = smem + 8192;
    unsigned int* own_routes = reinterpret_cast<unsigned int*>(smem_raw + 9984);
    const int own_routes_addr = smem + 9984;
    int* stream_ring = reinterpret_cast<int*>(smem_raw + 17152);
    const int stream_ring_addr = smem + 17152;
    unsigned int* rank_bits = reinterpret_cast<unsigned int*>(smem_raw + 49408);
    const int rank_bits_addr = smem + 49408;
    int* rank_base = reinterpret_cast<int*>(smem_raw + 51328);
    const int rank_base_addr = smem + 51328;
    unsigned int* rank_hits = reinterpret_cast<unsigned int*>(smem_raw + 8192);
    const int rank_hits_addr = smem + 8192;
    int* rank_count = reinterpret_cast<int*>(smem_raw + 51456);
    const int rank_count_addr = smem + 51456;
    unsigned int* rank_words = reinterpret_cast<unsigned int*>(smem_raw + 17152);
    const int rank_words_addr = smem + 17152;
    int* w_hist = reinterpret_cast<int*>(smem_raw + 0);
    const int w_hist_addr = smem + 0;
    float* w_f32 = reinterpret_cast<float*>(smem_raw + 0);
    const int w_f32_addr = smem + 0;
    unsigned long long* w_comp = reinterpret_cast<unsigned long long*>(smem_raw + 0);
    const int w_comp_addr = smem + 0;
    int* w_win_ids = reinterpret_cast<int*>(smem_raw + 28672);
    const int w_win_ids_addr = smem + 28672;
    float* w_win_scores = reinterpret_cast<float*>(smem_raw + 29120);
    const int w_win_scores_addr = smem + 29120;
    float* w_bias = reinterpret_cast<float*>(smem_raw + 29568);
    const int w_bias_addr = smem + 29568;

    // === Task calls (declaration order) ===
    int global_thread = bid * THREADS + tid;
    int grid_threads = num_bids * THREADS;
    int expert_base = tid * ITEMS_PER_THREAD;
    int stash_base = warp * TOP_K;
    float _vec_load_0[4];
    {
        float4 _v4 = *reinterpret_cast<const float4*>(bias + expert_base);
        _vec_load_0[0 + 0] = _v4.x;
        _vec_load_0[0 + 1] = _v4.y;
        _vec_load_0[0 + 2] = _v4.z;
        _vec_load_0[0 + 3] = _v4.w;
    }
    int token_stride = num_bids * NUM_WARPS;
    int token = bid + num_bids * warp;
    #pragma unroll
    for (int w_bias_slot = 0; w_bias_slot < 4; w_bias_slot++) {
        w_bias[expert_base + w_bias_slot] = _vec_load_0[w_bias_slot];
    }
    if (lane < TOP_K) {
        w_win_ids[stash_base + lane] = 0;
        w_win_scores[stash_base + lane] = 0.0f;
    }
    __syncthreads();
    int hist_base = warp * 896;
    int cand_base = 6272 + warp * 128;
    int lane_hist_base = hist_base + lane * 9;
    int scan_bin_base = lane * 8;
    unsigned int one_u32 = 1;
    unsigned int lanemask_lt = (one_u32 << (unsigned int)lane) - 1;
    while (token < M) {
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
            w_f32[hist_base + col] = score;
            float bias_col = w_bias[col];
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
                w_hist[cand_base + p_slot] = p_item * 32 + lane;
                p_walk = p_walk & p_walk - one_u32;
                p_slot += 1;
            }
            __syncwarp();
            int p_id = 0;
            float p_score = 0.0f;
            unsigned long long p_comp = 0;
            int comp_base = cand_base + 64 >> 1;
            if (p_total > lane) {
                p_id = w_hist[cand_base + lane];
                p_score = w_f32[hist_base + p_id];
                float p_ranking = p_score + w_bias[p_id];
                unsigned int p_bits = __as_u32(p_ranking);
                unsigned int p_ordered = p_bits ^ ((unsigned int)((int)p_bits >> 31) | 2147483648u);
                p_comp = (unsigned long long)p_ordered << 32 | (unsigned long long)(1023 - p_id);
            }
            w_comp[comp_base + lane] = p_comp;
            __syncwarp();
            int p_ra = 0;
            int p_rb = 0;
            int p_rc = 0;
            int p_rd = 0;
            unsigned int p_peers[16];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[0])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(0) + 3]))
                : "r"(w_comp_addr + (unsigned int)((comp_base + 0 + 0) * 8)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[4])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(4) + 3]))
                : "r"(w_comp_addr + (unsigned int)((comp_base + 0 + 2) * 8)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[8])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(8) + 3]))
                : "r"(w_comp_addr + (unsigned int)((comp_base + 0 + 4) * 8)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[12])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(12) + 3]))
                : "r"(w_comp_addr + (unsigned int)((comp_base + 0 + 6) * 8)));
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
                : "r"(w_comp_addr + (unsigned int)((comp_base + 8 + 0) * 8)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[4])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(4) + 3]))
                : "r"(w_comp_addr + (unsigned int)((comp_base + 8 + 2) * 8)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[8])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(8) + 3]))
                : "r"(w_comp_addr + (unsigned int)((comp_base + 8 + 4) * 8)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[12])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(12) + 3]))
                : "r"(w_comp_addr + (unsigned int)((comp_base + 8 + 6) * 8)));
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
                    : "r"(w_comp_addr + (unsigned int)((comp_base + 16 + 0) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[4])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(4) + 3]))
                    : "r"(w_comp_addr + (unsigned int)((comp_base + 16 + 2) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[8])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(8) + 3]))
                    : "r"(w_comp_addr + (unsigned int)((comp_base + 16 + 4) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[12])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(12) + 3]))
                    : "r"(w_comp_addr + (unsigned int)((comp_base + 16 + 6) * 8)));
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
                    : "r"(w_comp_addr + (unsigned int)((comp_base + 24 + 0) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[4])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(4) + 3]))
                    : "r"(w_comp_addr + (unsigned int)((comp_base + 24 + 2) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[8])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(8) + 3]))
                    : "r"(w_comp_addr + (unsigned int)((comp_base + 24 + 4) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&p_peers[12])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_peers[(12) + 3]))
                    : "r"(w_comp_addr + (unsigned int)((comp_base + 24 + 6) * 8)));
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
                    w_hist[cand_base + 32 + p_rank] = p_id;
                    w_f32[cand_base + 48 + p_rank] = p_score;
                }
            }
            __syncwarp();
            if (lane < TOP_K) {
                int p_win_id = w_hist[cand_base + 32 + lane];
                float p_win_score = w_f32[cand_base + 48 + lane];
                int p_sa = 0;
                int p_sb = 0;
                int p_sc = 0;
                int p_sd = 0;
                unsigned int p_wins[16];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&p_wins[0])), "=r"(*reinterpret_cast<uint32_t*>(&p_wins[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_wins[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_wins[(0) + 3]))
                    : "r"(w_hist_addr + (unsigned int)((cand_base + 32 + 0) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&p_wins[4])), "=r"(*reinterpret_cast<uint32_t*>(&p_wins[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_wins[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_wins[(4) + 3]))
                    : "r"(w_hist_addr + (unsigned int)((cand_base + 32 + 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&p_wins[8])), "=r"(*reinterpret_cast<uint32_t*>(&p_wins[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_wins[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_wins[(8) + 3]))
                    : "r"(w_hist_addr + (unsigned int)((cand_base + 32 + 8) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&p_wins[12])), "=r"(*reinterpret_cast<uint32_t*>(&p_wins[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p_wins[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p_wins[(12) + 3]))
                    : "r"(w_hist_addr + (unsigned int)((cand_base + 32 + 12) * 4)));
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
                w_win_ids[stash_base + p_win_slot] = p_win_id;
                w_win_scores[stash_base + p_win_slot] = p_win_score;
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
                    w_hist[q_base + q_slot] = q_item * 32 + lane;
                    q_walk = q_walk & q_walk - one_u32;
                    q_slot += 1;
                }
                __syncwarp();
                int q_comp_base = q_base + 64 >> 1;
                int q_hi = lane + 32;
                int q_id_lo = w_hist[q_base + lane];
                int q_id_hi = 0;
                float q_score_lo = 0.0f;
                float q_score_hi = 0.0f;
                unsigned long long q_comp_lo = 0;
                unsigned long long q_comp_hi = 0;
                q_score_lo = w_f32[hist_base + q_id_lo];
                q_comp_lo = (unsigned long long)(__as_u32(q_score_lo + w_bias[q_id_lo]) ^ ((unsigned int)((int)__as_u32(q_score_lo + w_bias[q_id_lo]) >> 31) | 2147483648u)) << 32 | (unsigned long long)(1023 - q_id_lo);
                w_comp[q_comp_base + lane] = q_comp_lo;
                if (q_hi < p_total) {
                    q_id_hi = w_hist[q_base + q_hi];
                    q_score_hi = w_f32[hist_base + q_id_hi];
                    q_comp_hi = (unsigned long long)(__as_u32(q_score_hi + w_bias[q_id_hi]) ^ ((unsigned int)((int)__as_u32(q_score_hi + w_bias[q_id_hi]) >> 31) | 2147483648u)) << 32 | (unsigned long long)(1023 - q_id_hi);
                }
                w_comp[q_comp_base + q_hi] = q_comp_hi;
                __syncwarp();
                int q_lo_a = 0;
                int q_lo_b = 0;
                int q_hi_a = 0;
                int q_hi_b = 0;
                unsigned int q_peers[16];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(0) + 3]))
                    : "r"(w_comp_addr + (unsigned int)((q_comp_base + 0 + 0) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 3]))
                    : "r"(w_comp_addr + (unsigned int)((q_comp_base + 0 + 2) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 3]))
                    : "r"(w_comp_addr + (unsigned int)((q_comp_base + 0 + 4) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 3]))
                    : "r"(w_comp_addr + (unsigned int)((q_comp_base + 0 + 6) * 8)));
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
                    : "r"(w_comp_addr + (unsigned int)((q_comp_base + 8 + 0) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 3]))
                    : "r"(w_comp_addr + (unsigned int)((q_comp_base + 8 + 2) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 3]))
                    : "r"(w_comp_addr + (unsigned int)((q_comp_base + 8 + 4) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 3]))
                    : "r"(w_comp_addr + (unsigned int)((q_comp_base + 8 + 6) * 8)));
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
                    : "r"(w_comp_addr + (unsigned int)((q_comp_base + 16 + 0) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 3]))
                    : "r"(w_comp_addr + (unsigned int)((q_comp_base + 16 + 2) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 3]))
                    : "r"(w_comp_addr + (unsigned int)((q_comp_base + 16 + 4) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 3]))
                    : "r"(w_comp_addr + (unsigned int)((q_comp_base + 16 + 6) * 8)));
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
                    : "r"(w_comp_addr + (unsigned int)((q_comp_base + 24 + 0) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 3]))
                    : "r"(w_comp_addr + (unsigned int)((q_comp_base + 24 + 2) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 3]))
                    : "r"(w_comp_addr + (unsigned int)((q_comp_base + 24 + 4) * 8)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 3]))
                    : "r"(w_comp_addr + (unsigned int)((q_comp_base + 24 + 6) * 8)));
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
                        : "r"(w_comp_addr + (unsigned int)((q_comp_base + 32 + 0) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 3]))
                        : "r"(w_comp_addr + (unsigned int)((q_comp_base + 32 + 2) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 3]))
                        : "r"(w_comp_addr + (unsigned int)((q_comp_base + 32 + 4) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 3]))
                        : "r"(w_comp_addr + (unsigned int)((q_comp_base + 32 + 6) * 8)));
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
                        : "r"(w_comp_addr + (unsigned int)((q_comp_base + 40 + 0) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 3]))
                        : "r"(w_comp_addr + (unsigned int)((q_comp_base + 40 + 2) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 3]))
                        : "r"(w_comp_addr + (unsigned int)((q_comp_base + 40 + 4) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 3]))
                        : "r"(w_comp_addr + (unsigned int)((q_comp_base + 40 + 6) * 8)));
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
                        : "r"(w_comp_addr + (unsigned int)((q_comp_base + 48 + 0) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 3]))
                        : "r"(w_comp_addr + (unsigned int)((q_comp_base + 48 + 2) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 3]))
                        : "r"(w_comp_addr + (unsigned int)((q_comp_base + 48 + 4) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 3]))
                        : "r"(w_comp_addr + (unsigned int)((q_comp_base + 48 + 6) * 8)));
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
                        : "r"(w_comp_addr + (unsigned int)((q_comp_base + 56 + 0) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(4) + 3]))
                        : "r"(w_comp_addr + (unsigned int)((q_comp_base + 56 + 2) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(8) + 3]))
                        : "r"(w_comp_addr + (unsigned int)((q_comp_base + 56 + 4) * 8)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_peers[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_peers[(12) + 3]))
                        : "r"(w_comp_addr + (unsigned int)((q_comp_base + 56 + 6) * 8)));
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
                    w_hist[cand_base + 32 + q_rank_lo] = q_id_lo;
                    w_f32[cand_base + 48 + q_rank_lo] = q_score_lo;
                }
                if (q_hi < p_total) {
                    if (q_rank_hi < TOP_K) {
                        w_hist[cand_base + 32 + q_rank_hi] = q_id_hi;
                        w_f32[cand_base + 48 + q_rank_hi] = q_score_hi;
                    }
                }
                __syncwarp();
                if (lane < TOP_K) {
                    int q_win_id = w_hist[cand_base + 32 + lane];
                    float q_win_score = w_f32[cand_base + 48 + lane];
                    int q_sa = 0;
                    int q_sb = 0;
                    int q_sc = 0;
                    int q_sd = 0;
                    unsigned int q_wins[16];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_wins[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_wins[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_wins[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_wins[(0) + 3]))
                        : "r"(w_hist_addr + (unsigned int)((cand_base + 32 + 0) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_wins[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_wins[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_wins[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_wins[(4) + 3]))
                        : "r"(w_hist_addr + (unsigned int)((cand_base + 32 + 4) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_wins[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_wins[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_wins[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_wins[(8) + 3]))
                        : "r"(w_hist_addr + (unsigned int)((cand_base + 32 + 8) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&q_wins[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_wins[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_wins[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_wins[(12) + 3]))
                        : "r"(w_hist_addr + (unsigned int)((cand_base + 32 + 12) * 4)));
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
                    w_win_ids[stash_base + q_win_slot] = q_win_id;
                    w_win_scores[stash_base + q_win_slot] = q_win_score;
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
                        w_hist[lane_hist_base + clear_slot] = 0;
                    }
                    __syncwarp();
                    #pragma unroll
                    for (int item_hist = 0; item_hist < 28; item_hist++) {
                        if ((keys[item_hist] & examined_mask) == threshold) {
                            int digit = (int)(keys[item_hist] >> (unsigned int)shift & 255);
                            atomicAdd(&w_hist[hist_base + digit + (digit >> 3)], 1);
                        }
                    }
                    __syncwarp();
                    int scan_counts[8];
                    int lane_count = 0;
                    #pragma unroll
                    for (int scan_offset = 0; scan_offset < 8; scan_offset++) {
                        int scan_count = w_hist[lane_hist_base + scan_offset];
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
                    w_win_ids[stash_base + slot] = winner_col;
                }
                compact_prior += (int)(unsigned int)__popc(compact_ballot);
            }
            __syncwarp();
            if (lane < TOP_K) {
                int score_col = w_win_ids[stash_base + lane];
                float score_logit = reinterpret_cast<volatile float*>(logits)[row_base + (unsigned long long)score_col];
                float _exp_32 = __expf(-score_logit);
                float _fdiv_31;
                asm volatile("div.full.f32 %0, %1, %2;" : "=f"(_fdiv_31) : "f"(1.0f), "f"(1.0f + _exp_32));
                w_win_scores[stash_base + lane] = _fdiv_31;
            }
            __syncwarp();
        }
        if (lane < TOP_K) {
            float selected_sum = 0.0f;
            #pragma unroll
            for (int route_sum = 0; route_sum < TOP_K; route_sum++) {
                selected_sum += w_win_scores[stash_base + route_sum];
            }
            float selected_norm = 1.0f;
            if (selected_sum > 0.0f) {
                selected_norm = selected_sum;
            }
            unsigned long long output_index = (unsigned long long)token * (unsigned long long)TOP_K + (unsigned long long)lane;
            float _fdiv_33;
            asm volatile("div.full.f32 %0, %1, %2;" : "=f"(_fdiv_33) : "f"(w_win_scores[stash_base + lane]), "f"(selected_norm));
            topk_weights[output_index] = _fdiv_33;
            topk_ids[output_index] = w_win_ids[stash_base + lane];
        }
        __syncwarp();
        token += token_stride;
    }
    __syncthreads();
    unsigned int _grid_arrive_0 = cooperative_groups::this_grid().barrier_arrive();
    unsigned int join_token = _grid_arrive_0;
    int own_expert_base = bid * NUM_WARPS;
    if (bid < OWNER_CTAS) {
        #pragma unroll 1
        for (int count_clear = tid; count_clear < NUM_EXPERTS; count_clear += THREADS) {
            plan_counts[count_clear] = 0;
        }
        #pragma unroll 1
        for (int rank_clear = tid; rank_clear < CLUSTER_EXPERTS * 17; rank_clear += THREADS) {
            rank_bits[rank_clear] = (unsigned int)0;
        }
        if (tid == 0) {
            rank_count[0] = 0;
        }
        __syncthreads();
    }
    if (bid < OWNER_CTAS) {
        cooperative_groups::this_grid().barrier_wait(static_cast<cooperative_groups::grid_group::arrival_token>(join_token));
        int total_pairs = M * TOP_K;
        int bitmap_words = M + 31 >> 5;
        int slice_pairs = (total_pairs + CLUSTER - 1) / CLUSTER + 3 & -4;
        int slice_begin = cta_rank * slice_pairs;
        int slice_end = slice_begin + slice_pairs;
        if (slice_end > total_pairs) {
            slice_end = total_pairs;
        }
        int cluster_expert_base = (bid - cta_rank) * NUM_WARPS;
        #pragma unroll
        for (int prologue_copy = 0; prologue_copy < COPIES_PER_THREAD; prologue_copy++) {
            int prologue_chunk = prologue_copy * THREADS + tid;
            int prologue_pair = slice_begin + prologue_chunk * 4;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(stream_ring_addr + (unsigned int)(prologue_chunk * 16)), "l"(topk_ids + prologue_pair), "r"((prologue_pair < slice_end) ? 16 : 0));
        }
        asm volatile("cp.async.commit_group;");
        int first_token = slice_begin >> 4;
        int stream_stage = 0;
        #pragma unroll 1
        for (int gather_base = slice_begin; gather_base < slice_end; gather_base += STAGE_PAIRS) {
            int next_base = gather_base + STAGE_PAIRS;
            if (next_base < slice_end) {
                int next_stage_bytes = (1 - stream_stage) * STAGE_BYTES;
                #pragma unroll
                for (int next_copy = 0; next_copy < COPIES_PER_THREAD; next_copy++) {
                    int next_chunk = next_copy * THREADS + tid;
                    int next_pair = next_base + next_chunk * 4;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"(stream_ring_addr + (unsigned int)(next_stage_bytes + next_chunk * 16)), "l"(topk_ids + next_pair), "r"((next_pair < slice_end) ? 16 : 0));
                }
                asm volatile("cp.async.commit_group;");
                asm volatile("cp.async.wait_group 1;");
            } else {
                asm volatile("cp.async.wait_group 0;");
            }
            int own_stage_bytes = stream_stage * STAGE_BYTES;
            unsigned int own_ids[16];
            #pragma unroll
            for (int own_copy = 0; own_copy < COPIES_PER_THREAD; own_copy++) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&own_ids[own_copy * 4])), "=r"(*reinterpret_cast<uint32_t*>(&own_ids[(own_copy * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own_ids[(own_copy * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own_ids[(own_copy * 4) + 3]))
                    : "r"(stream_ring_addr + (unsigned int)(own_stage_bytes + (own_copy * THREADS + tid) * 16)));
            }
            unsigned int hit_mask = 0;
            #pragma unroll
            for (int own_copy_1 = 0; own_copy_1 < COPIES_PER_THREAD; own_copy_1++) {
                int copy_pair = gather_base + (own_copy_1 * THREADS + tid) * 4;
                if (copy_pair < slice_end) {
                    #pragma unroll
                    for (int copy_lane = 0; copy_lane < 4; copy_lane++) {
                        int copy_id = (int)own_ids[own_copy_1 * 4 + copy_lane];
                        atomicAdd(&plan_counts[copy_id], 1);
                        int copy_slot = copy_id - cluster_expert_base;
                        if (copy_slot >= 0 && copy_slot < CLUSTER_EXPERTS) {
                            hit_mask |= (unsigned int)1 << (unsigned int)(own_copy_1 * 4 + copy_lane);
                        }
                    }
                }
            }
            int hit_count = (int)(unsigned int)__popc(hit_mask);
            int stage_before = 0;
            int stage_total = 0;
            unsigned int stage_ballot_4_0 = __ballot_sync(0xFFFFFFFF, (hit_count >> 0 & 1) != 0);
            stage_before += (int)(unsigned int)__popc(stage_ballot_4_0 & lanemask_lt) << 0;
            stage_total += (int)(unsigned int)__popc(stage_ballot_4_0) << 0;
            unsigned int stage_ballot_4_1 = __ballot_sync(0xFFFFFFFF, (hit_count >> 1 & 1) != 0);
            stage_before += (int)(unsigned int)__popc(stage_ballot_4_1 & lanemask_lt) << 1;
            stage_total += (int)(unsigned int)__popc(stage_ballot_4_1) << 1;
            unsigned int stage_ballot_4_2 = __ballot_sync(0xFFFFFFFF, (hit_count >> 2 & 1) != 0);
            stage_before += (int)(unsigned int)__popc(stage_ballot_4_2 & lanemask_lt) << 2;
            stage_total += (int)(unsigned int)__popc(stage_ballot_4_2) << 2;
            unsigned int stage_ballot_4_3 = __ballot_sync(0xFFFFFFFF, (hit_count >> 3 & 1) != 0);
            stage_before += (int)(unsigned int)__popc(stage_ballot_4_3 & lanemask_lt) << 3;
            stage_total += (int)(unsigned int)__popc(stage_ballot_4_3) << 3;
            unsigned int stage_ballot_4_4 = __ballot_sync(0xFFFFFFFF, (hit_count >> 4 & 1) != 0);
            stage_before += (int)(unsigned int)__popc(stage_ballot_4_4 & lanemask_lt) << 4;
            stage_total += (int)(unsigned int)__popc(stage_ballot_4_4) << 4;
            int stage_warp_base = 0;
            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %3, 0;\n\t@p atom.shared.add.s32 %0, [%1], %2;\n}"
                : "+r"(stage_warp_base) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&rank_count[0]))), "r"(static_cast<int>(stage_total)), "r"(static_cast<uint32_t>(lane == 0)) : "memory");
            int _shfl_sync_34 = __shfl_sync(0xFFFFFFFF, stage_warp_base, 0);
            int stage_base = _shfl_sync_34 + stage_before;
            #pragma unroll 1
            for (int hit = 0; hit < hit_count; hit++) {
                int _find_first_set_35 = __ffs(hit_mask);
                int hit_bit = _find_first_set_35 - 1;
                int hit_slot = (hit_bit >> 2) * THREADS * 4 + tid * 4 + (hit_bit & 3);
                int hit_pair = gather_base + hit_slot;
                int hit_id = stream_ring[stream_stage * STAGE_PAIRS + hit_slot];
                int hit_cluster_slot = hit_id - cluster_expert_base;
                int hit_local = (hit_pair >> 4) - first_token;
                atomicAdd(&rank_bits[hit_cluster_slot * 17 + (hit_local >> 5)], (unsigned int)1 << (unsigned int)(hit_local & 31));
                int hit_index = stage_base + hit;
                if (hit_index < 2240) {
                    rank_hits[hit_index] = (unsigned int)(hit_pair << 5 | hit_cluster_slot);
                }
                hit_mask = hit_mask & hit_mask - (unsigned int)1;
            }
            stream_stage = 1 - stream_stage;
        }
        asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
        asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        unsigned int peer_counts_base[4];
        #pragma unroll
        for (int count_peer_map = 0; count_peer_map < CLUSTER; count_peer_map++) {
            uint32_t _cluster_mapa_36;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_36) : "r"(plan_counts_addr), "r"(count_peer_map));
            peer_counts_base[count_peer_map] = _cluster_mapa_36;
        }
        int expert_vec_base = warp * 32 * ITEMS_PER_THREAD + lane * ITEMS_PER_THREAD;
        unsigned int peer_count_words[16];
        #pragma unroll
        for (int zero_word = 0; zero_word < 16; zero_word++) {
            peer_count_words[zero_word] = (unsigned int)0;
        }
        if (expert_vec_base < cluster_expert_base + CLUSTER_EXPERTS) {
            #pragma unroll
            for (int count_peer_vec = 0; count_peer_vec < CLUSTER; count_peer_vec++) {
                asm volatile("ld.shared::cluster.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&peer_count_words[count_peer_vec * ITEMS_PER_THREAD])), "=r"(*reinterpret_cast<uint32_t*>(&peer_count_words[(count_peer_vec * ITEMS_PER_THREAD) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&peer_count_words[(count_peer_vec * ITEMS_PER_THREAD) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&peer_count_words[(count_peer_vec * ITEMS_PER_THREAD) + 3]))
                    : "r"(peer_counts_base[count_peer_vec] + (unsigned int)(expert_vec_base * 4)));
            }
        }
        int expert_scan_base = warp * 32 * ITEMS_PER_THREAD + lane * ITEMS_PER_THREAD;
        int lane_prefixes[4];
        int lane_total = 0;
        #pragma unroll
        for (int expert_slot_scan = 0; expert_slot_scan < ITEMS_PER_THREAD; expert_slot_scan++) {
            int expert_scan = expert_scan_base + expert_slot_scan;
            int count_scan = 0;
            #pragma unroll
            for (int count_peer = 0; count_peer < CLUSTER; count_peer++) {
                count_scan += (int)peer_count_words[count_peer * ITEMS_PER_THREAD + expert_slot_scan];
            }
            plan_totals[expert_scan] = count_scan;
            lane_prefixes[expert_slot_scan] = lane_total;
            int padded_count_scan = count_scan + BLOCK_MASK & ~BLOCK_MASK;
            lane_total += padded_count_scan;
            int base_slot = expert_scan - cluster_expert_base;
            if (base_slot >= 0 && base_slot < CLUSTER_EXPERTS) {
                int count_before = 0;
                #pragma unroll
                for (int before_peer = 0; before_peer < CLUSTER; before_peer++) {
                    if (before_peer < cta_rank) {
                        count_before += (int)peer_count_words[before_peer * ITEMS_PER_THREAD + expert_slot_scan];
                    }
                }
                rank_base[base_slot] = count_before;
            }
        }
        #pragma unroll
        for (int word_round = 0; word_round < 4; word_round++) {
            int word_slot = word_round * NUM_WARPS + warp;
            unsigned int slot_bits = 0;
            if (lane < 17) {
                slot_bits = rank_bits[word_slot * 17 + lane];
            }
            int slot_count = (int)(unsigned int)__popc(slot_bits);
            int slot_before = 0;
            unsigned int slot_ballot_5_0 = __ballot_sync(0xFFFFFFFF, (slot_count >> 0 & 1) != 0);
            slot_before += (int)(unsigned int)__popc(slot_ballot_5_0 & lanemask_lt) << 0;
            unsigned int slot_ballot_5_1 = __ballot_sync(0xFFFFFFFF, (slot_count >> 1 & 1) != 0);
            slot_before += (int)(unsigned int)__popc(slot_ballot_5_1 & lanemask_lt) << 1;
            unsigned int slot_ballot_5_2 = __ballot_sync(0xFFFFFFFF, (slot_count >> 2 & 1) != 0);
            slot_before += (int)(unsigned int)__popc(slot_ballot_5_2 & lanemask_lt) << 2;
            unsigned int slot_ballot_5_3 = __ballot_sync(0xFFFFFFFF, (slot_count >> 3 & 1) != 0);
            slot_before += (int)(unsigned int)__popc(slot_ballot_5_3 & lanemask_lt) << 3;
            unsigned int slot_ballot_5_4 = __ballot_sync(0xFFFFFFFF, (slot_count >> 4 & 1) != 0);
            slot_before += (int)(unsigned int)__popc(slot_ballot_5_4 & lanemask_lt) << 4;
            unsigned int slot_ballot_5_5 = __ballot_sync(0xFFFFFFFF, (slot_count >> 5 & 1) != 0);
            slot_before += (int)(unsigned int)__popc(slot_ballot_5_5 & lanemask_lt) << 5;
            if (lane < 17) {
                rank_words[(word_slot * 17 + lane) * 2] = slot_bits;
                rank_words[(word_slot * 17 + lane) * 2 + 1] = (unsigned int)slot_before;
            }
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
        asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
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
            plan_offsets[expert_store] = offset_store;
        }
        if (bid == OWNER_CTAS - CLUSTER) {
            if (tid == THREADS - 1) {
                int padded_total = phase2_warp_prefix + warp_inclusive;
                expert_offsets[NUM_EXPERTS] = padded_total;
                num_tokens_post_padded[0] = padded_total;
            }
        }
        __syncthreads();
        if (bid >= OWNER_CTAS - CLUSTER) {
            int slice_global = cta_rank * SLICE_EXPERTS + tid;
            int slice_count = plan_totals[slice_global];
            expert_counts[slice_global] = slice_count;
            expert_scatter_offsets[slice_global] = slice_count;
            expert_offsets[slice_global] = plan_offsets[slice_global];
        }
        int own_expert = own_expert_base + warp;
        int own_count = plan_totals[own_expert];
        int own_base = plan_offsets[own_expert];
        int own_padded = own_count + BLOCK_MASK & ~BLOCK_MASK;
        int total_hits = rank_count[0];
        if (total_hits > 2240) {
            #pragma unroll 1
            for (int rescan_base = slice_begin; rescan_base < slice_end; rescan_base += STAGE_PAIRS) {
                #pragma unroll
                for (int rescan_copy = 0; rescan_copy < COPIES_PER_THREAD; rescan_copy++) {
                    int rescan_chunk = rescan_base + (rescan_copy * THREADS + tid) * 4;
                    if (rescan_chunk < slice_end) {
                        #pragma unroll
                        for (int rescan_lane = 0; rescan_lane < 4; rescan_lane++) {
                            int rescan_pair = rescan_chunk + rescan_lane;
                            int rescan_slot = topk_ids[rescan_pair] - cluster_expert_base;
                            if (rescan_slot >= 0 && rescan_slot < CLUSTER_EXPERTS) {
                                int local_token = (rescan_pair >> 4) - first_token;
                                unsigned int word_pair[2];
                                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&word_pair[0])), "=r"(*reinterpret_cast<uint32_t*>(&word_pair[(0) + 1]))
                                    : "r"(rank_words_addr + (unsigned int)((rescan_slot * 17 + (local_token >> 5)) * 8)));
                                unsigned int below = word_pair[0] & ((unsigned int)1 << (unsigned int)(local_token & 31)) - (unsigned int)1;
                                int rank = plan_offsets[cluster_expert_base + rescan_slot] + rank_base[rescan_slot] + (int)word_pair[1] + (int)(unsigned int)__popc(below);
                                sorted_token_ids[rank] = rescan_pair;
                            }
                        }
                    }
                }
            }
        } else {
            #pragma unroll 1
            for (int hit_entry_index = tid; hit_entry_index < total_hits; hit_entry_index += THREADS) {
                unsigned int hit_entry = rank_hits[hit_entry_index];
                int local_token_1 = ((int)(hit_entry >> 5) >> 4) - first_token;
                unsigned int word_pair_1[2];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&word_pair_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&word_pair_1[(0) + 1]))
                    : "r"(rank_words_addr + (unsigned int)(((int)(hit_entry & 31) * 17 + (local_token_1 >> 5)) * 8)));
                unsigned int below_1 = word_pair_1[0] & ((unsigned int)1 << (unsigned int)(local_token_1 & 31)) - (unsigned int)1;
                int rank_1 = plan_offsets[cluster_expert_base + (int)(hit_entry & 31)] + rank_base[(int)(hit_entry & 31)] + (int)word_pair_1[1] + (int)(unsigned int)__popc(below_1);
                sorted_token_ids[rank_1] = (int)(hit_entry >> 5);
            }
        }
        int own_bitmap_base = warp * BITMAP_STRIDE;
        int own_route_base = warp * ROUTE_STRIDE;
        int emitted = 0;
        #pragma unroll 1
        for (int pad_slot = own_count + lane; pad_slot < own_padded; pad_slot += 32) {
            sorted_token_ids[own_base + pad_slot] = total_pairs;
        }
        #pragma unroll 1
        for (int block_slot = lane; block_slot < own_padded >> BLOCK_SHIFT; block_slot += 32) {
            expert_ids[(own_base >> BLOCK_SHIFT) + block_slot] = own_expert;
        }
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
}

} // extern "C"
