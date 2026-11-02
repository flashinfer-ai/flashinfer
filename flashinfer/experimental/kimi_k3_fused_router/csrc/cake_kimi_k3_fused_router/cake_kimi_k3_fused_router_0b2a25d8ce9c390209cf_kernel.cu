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
#define SMEM_BOUND_VALUES_OFF 1280
#define SMEM_BOUND_VALUES_STAGE_BYTES 128
#define SMEM_BOUND_VALUES_STRIDE 128
#define SMEM_CAND_COUNTS_OFF 1408
#define SMEM_CAND_COUNTS_STAGE_BYTES 32
#define SMEM_CAND_COUNTS_STRIDE 32
#define SMEM_CAND_KEYS_OFF 1440
#define SMEM_CAND_KEYS_STAGE_BYTES 896
#define SMEM_CAND_KEYS_STRIDE 896
#define SMEM_CAND_IDS_OFF 2336
#define SMEM_CAND_IDS_STAGE_BYTES 896
#define SMEM_CAND_IDS_STRIDE 896
#define SMEM_CAND_SCORES_OFF 3232
#define SMEM_CAND_SCORES_STAGE_BYTES 896
#define SMEM_CAND_SCORES_STRIDE 896
#define SMEM_RANK_KEYS_OFF 4128
#define SMEM_RANK_KEYS_STAGE_BYTES 128
#define SMEM_RANK_KEYS_STRIDE 128
#define SMEM_TOTAL 4352
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
#define BLOCK_MASK (BLOCK_M - 1)
#define BLOCK_SHIFT (3 + (BLOCK_M >> 4))

#include <math_constants.h>

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

__global__ __launch_bounds__(224) void
kernel_cake_kimi_k3_fused_router_0b2a25d8ce9c390209cf(float* __restrict__ logits, float* __restrict__ bias, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M)
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
    const unsigned int clusters_x = gridDim.x / 1;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 1;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    const int cta_rank = 0;

    // Kernel setup ops
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
    unsigned int* bound_values = reinterpret_cast<unsigned int*>(smem_raw + 1280);
    const int bound_values_addr = smem + 1280;
    int* cand_counts = reinterpret_cast<int*>(smem_raw + 1408);
    const int cand_counts_addr = smem + 1408;
    unsigned int* cand_keys = reinterpret_cast<unsigned int*>(smem_raw + 1440);
    const int cand_keys_addr = smem + 1440;
    int* cand_ids = reinterpret_cast<int*>(smem_raw + 2336);
    const int cand_ids_addr = smem + 2336;
    float* cand_scores = reinterpret_cast<float*>(smem_raw + 3232);
    const int cand_scores_addr = smem + 3232;
    unsigned int* rank_keys = reinterpret_cast<unsigned int*>(smem_raw + 4128);
    const int rank_keys_addr = smem + 4128;

    // === Task calls (declaration order) ===
    if (warp == 0) {
        if (elect_sync()) {
            #pragma unroll
            for (int winner_init = 0; winner_init < TOP_K; winner_init++) {
                winner_ids[winner_init] = 0;
                winner_scores[winner_init] = 0.0f;
            }
        }
    }
    if (tid < 4) {
        bound_values[28 + tid] = (unsigned int)0;
    }
    unsigned int one_u32 = (unsigned int)1;
    unsigned int lanemask_lt = (one_u32 << (unsigned int)lane) - 1;
    int expert_base = tid * ITEMS_PER_THREAD;
    float _vec_load_0[4];
    {
        float4 _v4 = *reinterpret_cast<const float4*>(bias + expert_base);
        _vec_load_0[0 + 0] = _v4.x;
        _vec_load_0[0 + 1] = _v4.y;
        _vec_load_0[0 + 2] = _v4.z;
        _vec_load_0[0 + 3] = _v4.w;
    }
    int token = bid;
    unsigned long long row_base = (unsigned long long)token * (unsigned long long)NUM_EXPERTS;
    float _vec_load_1[4];
    {
        float4 _v4 = *reinterpret_cast<const float4*>(logits + row_base + (unsigned long long)expert_base);
        _vec_load_1[0 + 0] = _v4.x;
        _vec_load_1[0 + 1] = _v4.y;
        _vec_load_1[0 + 2] = _v4.z;
        _vec_load_1[0 + 3] = _v4.w;
    }
    #pragma unroll 1
    for (int const_slot = tid; const_slot < TOP_K * BLOCK_M; const_slot += THREADS) {
        int const_value = M * TOP_K;
        if ((const_slot & BLOCK_MASK) == 0) {
            const_value = const_slot >> BLOCK_SHIFT;
        }
        sorted_token_ids[const_slot] = const_value;
    }
    int pz_zero[4];
    pz_zero[0] = 0;
    pz_zero[1] = 0;
    pz_zero[2] = 0;
    pz_zero[3] = 0;
    {
        int4 _iv4 = make_int4(pz_zero[0 + 0], pz_zero[0 + 1], pz_zero[0 + 2], pz_zero[0 + 3]);
        *reinterpret_cast<int4*>(expert_counts + expert_base) = _iv4;
    }
    {
        int4 _iv4 = make_int4(pz_zero[0 + 0], pz_zero[0 + 1], pz_zero[0 + 2], pz_zero[0 + 3]);
        *reinterpret_cast<int4*>(expert_scatter_offsets + expert_base) = _iv4;
    }
    if (tid == THREADS - 1) {
        expert_offsets[NUM_EXPERTS] = TOP_K * BLOCK_M;
        num_tokens_post_padded[0] = TOP_K * BLOCK_M;
    }
    float unbiased[4];
    unsigned int keys[4];
    int active[4];
    #pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; item++) {
        float _exp_2 = __expf(-_vec_load_1[item]);
        float score;
        asm volatile("div.full.f32 %0, %1, %2;" : "=f"(score) : "f"(1.0f), "f"(1.0f + _exp_2));
        float ranking = score + _vec_load_0[item];
        if (ranking == 0.0f) {
            ranking = 0.0f;
        }
        unsigned int bits = __as_u32(ranking);
        unsigned int ordered = bits | 2147483648u;
        if (bits >= 2147483648u) {
            ordered = bits ^ 4294967295u;
        }
        unbiased[item] = score;
        keys[item] = ordered;
        active[item] = 1;
    }
    unsigned int lane_bound = keys[0];
    #pragma unroll
    for (int item_max = 1; item_max < 4; item_max++) {
        if (keys[item_max] > lane_bound) {
            lane_bound = keys[item_max];
        }
    }
    unsigned int published = 0;
    unsigned int warp_top0_1_0;
    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(warp_top0_1_0) : "r"(lane_bound));
    if (lane == 0) {
        published = warp_top0_1_0;
    }
    if (lane_bound == warp_top0_1_0) {
        lane_bound = (unsigned int)0;
    }
    unsigned int warp_top0_1_1;
    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(warp_top0_1_1) : "r"(lane_bound));
    if (lane == 1) {
        published = warp_top0_1_1;
    }
    if (lane_bound == warp_top0_1_1) {
        lane_bound = (unsigned int)0;
    }
    unsigned int warp_top0_1_2;
    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(warp_top0_1_2) : "r"(lane_bound));
    if (lane == 2) {
        published = warp_top0_1_2;
    }
    if (lane_bound == warp_top0_1_2) {
        lane_bound = (unsigned int)0;
    }
    unsigned int warp_top0_1_3;
    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(warp_top0_1_3) : "r"(lane_bound));
    if (lane == 3) {
        published = warp_top0_1_3;
    }
    if (lane_bound == warp_top0_1_3) {
        lane_bound = (unsigned int)0;
    }
    if (lane < 4) {
        bound_values[warp * 4 + lane] = published;
    }
    __syncthreads();
    unsigned int bound_mine = bound_values[lane];
    int bound_part0 = ((bound_mine <= bound_values[0]) ? 1 : 0) + ((bound_mine <= bound_values[1]) ? 1 : 0) + ((bound_mine <= bound_values[2]) ? 1 : 0) + ((bound_mine <= bound_values[3]) ? 1 : 0);
    int bound_part1 = ((bound_mine <= bound_values[4]) ? 1 : 0) + ((bound_mine <= bound_values[5]) ? 1 : 0) + ((bound_mine <= bound_values[6]) ? 1 : 0) + ((bound_mine <= bound_values[7]) ? 1 : 0);
    int bound_part2 = ((bound_mine <= bound_values[8]) ? 1 : 0) + ((bound_mine <= bound_values[9]) ? 1 : 0) + ((bound_mine <= bound_values[10]) ? 1 : 0) + ((bound_mine <= bound_values[11]) ? 1 : 0);
    int bound_part3 = ((bound_mine <= bound_values[12]) ? 1 : 0) + ((bound_mine <= bound_values[13]) ? 1 : 0) + ((bound_mine <= bound_values[14]) ? 1 : 0) + ((bound_mine <= bound_values[15]) ? 1 : 0);
    int bound_part4 = ((bound_mine <= bound_values[16]) ? 1 : 0) + ((bound_mine <= bound_values[17]) ? 1 : 0) + ((bound_mine <= bound_values[18]) ? 1 : 0) + ((bound_mine <= bound_values[19]) ? 1 : 0);
    int bound_part5 = ((bound_mine <= bound_values[20]) ? 1 : 0) + ((bound_mine <= bound_values[21]) ? 1 : 0) + ((bound_mine <= bound_values[22]) ? 1 : 0) + ((bound_mine <= bound_values[23]) ? 1 : 0);
    int bound_part6 = ((bound_mine <= bound_values[24]) ? 1 : 0) + ((bound_mine <= bound_values[25]) ? 1 : 0) + ((bound_mine <= bound_values[26]) ? 1 : 0) + ((bound_mine <= bound_values[27]) ? 1 : 0);
    int bound_at_least = bound_part0 + bound_part1 + (bound_part2 + bound_part3) + (bound_part4 + bound_part5 + bound_part6);
    unsigned int cw_pre = (unsigned int)0;
    if (warp == 0) {
        unsigned int cw_word = bound_mine & 4294967232u | (unsigned int)(bound_at_least + 32);
        cw_pre = cw_word;
    }
    if (bound_at_least < 16) {
        bound_mine = (unsigned int)0;
    }
    unsigned int cand_floor;
    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(cand_floor) : "r"(bound_mine));
    int cand_below = 0;
    int cand_warp = 0;
    #pragma unroll
    for (int item_vote = 0; item_vote < 4; item_vote++) {
        unsigned int cand_ballot = __ballot_sync(0xFFFFFFFF, keys[item_vote] >= cand_floor);
        cand_below += (int)(unsigned int)__popc(cand_ballot & lanemask_lt);
        cand_warp += (int)(unsigned int)__popc(cand_ballot);
    }
    int cand_slot = cand_below;
    if (keys[0] >= cand_floor && cand_slot < 32) {
        int cand_index0 = warp * 32 + cand_slot;
        cand_keys[cand_index0] = keys[0];
        cand_ids[cand_index0] = expert_base;
        cand_scores[cand_index0] = unbiased[0];
    }
    cand_slot += ((keys[0] >= cand_floor) ? 1 : 0);
    if (keys[1] >= cand_floor && cand_slot < 32) {
        int cand_index_0 = warp * 32 + cand_slot;
        cand_keys[cand_index_0] = keys[1];
        cand_ids[cand_index_0] = expert_base + 1;
        cand_scores[cand_index_0] = unbiased[1];
    }
    cand_slot += ((keys[1] >= cand_floor) ? 1 : 0);
    if (keys[2] >= cand_floor && cand_slot < 32) {
        int cand_index_1 = warp * 32 + cand_slot;
        cand_keys[cand_index_1] = keys[2];
        cand_ids[cand_index_1] = expert_base + 2;
        cand_scores[cand_index_1] = unbiased[2];
    }
    cand_slot += ((keys[2] >= cand_floor) ? 1 : 0);
    if (keys[3] >= cand_floor && cand_slot < 32) {
        int cand_index_2 = warp * 32 + cand_slot;
        cand_keys[cand_index_2] = keys[3];
        cand_ids[cand_index_2] = expert_base + 3;
        cand_scores[cand_index_2] = unbiased[3];
    }
    if (lane == 0) {
        cand_counts[warp] = cand_warp;
    }
    __syncthreads();
    int cand_total = cand_counts[0] + cand_counts[1] + (cand_counts[2] + cand_counts[3]) + (cand_counts[4] + cand_counts[5] + cand_counts[6]);
    if (cand_total > 32) {
        int remaining = TOP_K;
        int active_total = NUM_EXPERTS;
        unsigned int threshold = 0;
        unsigned int examined_mask = 0;
        int take_all_equals = 0;
        #pragma unroll
        for (int radix_round = 0; radix_round < 4; radix_round++) {
            int shift = 24 - radix_round * 8;
            int match_base = radix_round & 1;
            if (tid < RADIX_CLEAR_THREADS) {
                histogram[tid] = 0;
                histogram[tid + RADIX_CLEAR_THREADS] = 0;
            }
            __syncthreads();
            #pragma unroll
            for (int item_hist = 0; item_hist < ITEMS_PER_THREAD; item_hist++) {
                if (active[item_hist] != 0) {
                    int digit = (int)(keys[item_hist] >> (unsigned int)shift & 255);
                    atomicAdd(&histogram[digit], 1);
                }
            }
            __syncthreads();
            int scan_counts[8];
            int lane_count = 0;
            int inclusive = 0;
            unsigned int warp_match_record = 0;
            if (tid < RADIX_SCAN_THREADS) {
                int scan_bin_base = tid * RADIX_BINS_PER_LANE;
                #pragma unroll
                for (int scan_offset = 0; scan_offset < RADIX_BINS_PER_LANE; scan_offset++) {
                    int scan_count = histogram[scan_bin_base + scan_offset];
                    scan_counts[scan_offset] = scan_count;
                    lane_count += scan_count;
                }
                inclusive = lane_count;
                int scan_peer = __shfl_up_sync(0xFFFFFFFF, inclusive, 1, 32);
                if (lane >= 1) {
                    inclusive += scan_peer;
                }
                int scan_peer_0 = __shfl_up_sync(0xFFFFFFFF, inclusive, 2, 32);
                if (lane >= 2) {
                    inclusive += scan_peer_0;
                }
                int scan_peer_1 = __shfl_up_sync(0xFFFFFFFF, inclusive, 4, 32);
                if (lane >= 4) {
                    inclusive += scan_peer_1;
                }
                int scan_peer_2 = __shfl_up_sync(0xFFFFFFFF, inclusive, 8, 32);
                if (lane >= 8) {
                    inclusive += scan_peer_2;
                }
                int scan_peer_3 = __shfl_up_sync(0xFFFFFFFF, inclusive, 16, 32);
                if (lane >= 16) {
                    inclusive += scan_peer_3;
                }
                int bin_inclusive = inclusive - lane_count;
                #pragma unroll
                for (int scan_offset_1 = 0; scan_offset_1 < RADIX_BINS_PER_LANE; scan_offset_1++) {
                    int bin_count = scan_counts[scan_offset_1];
                    bin_inclusive += bin_count;
                    int bin_above = active_total - bin_inclusive;
                    if (bin_above < remaining && remaining <= bin_above + bin_count) {
                        int chosen_scan_bin = scan_bin_base + scan_offset_1;
                        warp_match_record = (unsigned int)(chosen_scan_bin & 255) | (unsigned int)(bin_above & 1023) << 8 | (unsigned int)(bin_count & 1023) << 18;
                    }
                }
                unsigned int _warp_redux_3;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_3) : "r"(warp_match_record));
                warp_match_record = _warp_redux_3;
                if (tid == 0) {
                    radix_match[match_base] = (int)warp_match_record;
                }
            }
            __syncthreads();
            int match_record = radix_match[match_base];
            int chosen_bin = match_record & 255;
            remaining -= match_record >> 8 & 1023;
            active_total = match_record >> 18 & 1023;
            threshold |= (unsigned int)chosen_bin << (unsigned int)shift;
            examined_mask |= (unsigned int)255 << (unsigned int)shift;
            #pragma unroll
            for (int item_filter = 0; item_filter < ITEMS_PER_THREAD; item_filter++) {
                int item_digit = (int)(keys[item_filter] >> (unsigned int)shift & 255);
                if (item_digit != chosen_bin) {
                    active[item_filter] = 0;
                }
            }
            if (remaining == active_total) {
                take_all_equals = 1;
                break;
            }
        }
        int selected[4];
        int selected_count = 0;
        if (take_all_equals != 0) {
            #pragma unroll
            for (int item_fast = 0; item_fast < ITEMS_PER_THREAD; item_fast++) {
                int choose_fast = 0;
                if (active[item_fast] != 0) {
                    choose_fast = 1;
                }
                if (threshold < (keys[item_fast] & examined_mask)) {
                    choose_fast = 1;
                }
                selected[item_fast] = choose_fast;
                selected_count += choose_fast;
            }
        } else {
            int tie_count = 0;
            #pragma unroll
            for (int item_tie_count = 0; item_tie_count < ITEMS_PER_THREAD; item_tie_count++) {
                tie_count += active[item_tie_count];
            }
            int tie_inclusive = tie_count;
            int tie_peer = __shfl_up_sync(0xFFFFFFFF, tie_inclusive, 1, 32);
            if (lane >= 1) {
                tie_inclusive += tie_peer;
            }
            int tie_peer_0 = __shfl_up_sync(0xFFFFFFFF, tie_inclusive, 2, 32);
            if (lane >= 2) {
                tie_inclusive += tie_peer_0;
            }
            int tie_peer_1 = __shfl_up_sync(0xFFFFFFFF, tie_inclusive, 4, 32);
            if (lane >= 4) {
                tie_inclusive += tie_peer_1;
            }
            int tie_peer_2 = __shfl_up_sync(0xFFFFFFFF, tie_inclusive, 8, 32);
            if (lane >= 8) {
                tie_inclusive += tie_peer_2;
            }
            int tie_peer_3 = __shfl_up_sync(0xFFFFFFFF, tie_inclusive, 16, 32);
            if (lane >= 16) {
                tie_inclusive += tie_peer_3;
            }
            if (lane == 31) {
                tie_warp_sums[warp] = tie_inclusive;
            }
            __syncthreads();
            int tie_prefix_lane0 = 0;
            if (lane == 0) {
                #pragma unroll
                for (int tie_warp = 0; tie_warp < NUM_WARPS; tie_warp++) {
                    if (tie_warp < warp) {
                        tie_prefix_lane0 += tie_warp_sums[tie_warp];
                    }
                }
            }
            int tie_prefix = __shfl_sync(0xFFFFFFFF, tie_prefix_lane0, 0);
            int tie_rank = tie_prefix + tie_inclusive - tie_count;
            #pragma unroll
            for (int item_select = 0; item_select < ITEMS_PER_THREAD; item_select++) {
                int choose_tie = 0;
                if (threshold < (keys[item_select] & examined_mask)) {
                    choose_tie = 1;
                }
                if (active[item_select] != 0) {
                    if (tie_rank < remaining) {
                        choose_tie = 1;
                    }
                    tie_rank += 1;
                }
                selected[item_select] = choose_tie;
                selected_count += choose_tie;
            }
        }
        int selected_inclusive = selected_count;
        int selected_peer = __shfl_up_sync(0xFFFFFFFF, selected_inclusive, 1, 32);
        if (lane >= 1) {
            selected_inclusive += selected_peer;
        }
        int selected_peer_0 = __shfl_up_sync(0xFFFFFFFF, selected_inclusive, 2, 32);
        if (lane >= 2) {
            selected_inclusive += selected_peer_0;
        }
        int selected_peer_1 = __shfl_up_sync(0xFFFFFFFF, selected_inclusive, 4, 32);
        if (lane >= 4) {
            selected_inclusive += selected_peer_1;
        }
        int selected_peer_2 = __shfl_up_sync(0xFFFFFFFF, selected_inclusive, 8, 32);
        if (lane >= 8) {
            selected_inclusive += selected_peer_2;
        }
        int selected_peer_3 = __shfl_up_sync(0xFFFFFFFF, selected_inclusive, 16, 32);
        if (lane >= 16) {
            selected_inclusive += selected_peer_3;
        }
        if (lane == 31) {
            selected_warp_sums[warp] = selected_inclusive;
        }
        __syncthreads();
        int selected_prefix_lane0 = 0;
        if (lane == 0) {
            #pragma unroll
            for (int selected_warp = 0; selected_warp < NUM_WARPS; selected_warp++) {
                if (selected_warp < warp) {
                    selected_prefix_lane0 += selected_warp_sums[selected_warp];
                }
            }
        }
        int selected_prefix = __shfl_sync(0xFFFFFFFF, selected_prefix_lane0, 0);
        int selected_rank = selected_prefix + selected_inclusive - selected_count;
        #pragma unroll
        for (int item_compact = 0; item_compact < ITEMS_PER_THREAD; item_compact++) {
            if (selected[item_compact] != 0) {
                if (selected_rank >= 0 && selected_rank < TOP_K) {
                    winner_ids[selected_rank] = expert_base + item_compact;
                    winner_scores[selected_rank] = unbiased[item_compact];
                }
            }
            selected_rank += selected[item_compact];
        }
        __syncthreads();
    } else if (warp == 0) {
        int cand_region = 0;
        int cand_offset = lane;
        int cand_prefix = 0;
        if (cand_prefix <= lane) {
            cand_region = 0;
            cand_offset = lane - cand_prefix;
        }
        cand_prefix += cand_counts[0];
        if (cand_prefix <= lane) {
            cand_region = 1;
            cand_offset = lane - cand_prefix;
        }
        cand_prefix += cand_counts[1];
        if (cand_prefix <= lane) {
            cand_region = 2;
            cand_offset = lane - cand_prefix;
        }
        cand_prefix += cand_counts[2];
        if (cand_prefix <= lane) {
            cand_region = 3;
            cand_offset = lane - cand_prefix;
        }
        cand_prefix += cand_counts[3];
        if (cand_prefix <= lane) {
            cand_region = 4;
            cand_offset = lane - cand_prefix;
        }
        cand_prefix += cand_counts[4];
        if (cand_prefix <= lane) {
            cand_region = 5;
            cand_offset = lane - cand_prefix;
        }
        cand_prefix += cand_counts[5];
        if (cand_prefix <= lane) {
            cand_region = 6;
            cand_offset = lane - cand_prefix;
        }
        cand_prefix += cand_counts[6];
        unsigned int my_key = 0;
        int my_id = 0;
        float my_score = 0.0f;
        if (cand_total > lane) {
            int my_slot = cand_region * 32 + cand_offset;
            my_key = cand_keys[my_slot];
            my_id = cand_ids[my_slot];
            my_score = cand_scores[my_slot];
        }
        unsigned int cw_f = cw_pre & 63;
        unsigned int cw_pv = (unsigned int)127;
        if (cw_f >= (unsigned int)48) {
            cw_pv = cw_f;
        }
        unsigned int cw_p32;
        asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(cw_p32) : "r"(cw_pv));
        unsigned int cw_tv = (unsigned int)0;
        if (cw_f + (unsigned int)cand_total >= cw_p32 + (unsigned int)17) {
            cw_tv = cw_pre | 63;
        }
        unsigned int cw_theta;
        asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(cw_theta) : "r"(cw_tv));
        unsigned int cw_valid = __ballot_sync(0xFFFFFFFF, cand_total > lane);
        unsigned int _warp_vote_4 = __ballot_sync(0xFFFFFFFF, my_key > cw_theta);
        unsigned int cw_c = cw_valid & _warp_vote_4;
        unsigned int cw_alive = cw_valid & (cw_c ^ 4294967295u);
        int cw_need = 16 - (int)(unsigned int)__popc(cw_c);
        if (cw_need > 0) {
            if (cw_need > 0) {
                unsigned int cw_mine_0 = cw_alive >> (unsigned int)lane & 1;
                unsigned int cw_ak_0 = (unsigned int)0;
                if (cw_mine_0 != (unsigned int)0) {
                    cw_ak_0 = my_key;
                }
                unsigned int cw_m_0;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(cw_m_0) : "r"(cw_ak_0));
                unsigned int _warp_vote_5 = __ballot_sync(0xFFFFFFFF, my_key == cw_m_0);
                unsigned int cw_eq_0 = cw_alive & _warp_vote_5;
                unsigned int cw_pick_0 = cw_eq_0 & (cw_eq_0 ^ 4294967295u) + (unsigned int)1;
                cw_c = cw_c | cw_pick_0;
                cw_alive = cw_alive & (cw_pick_0 ^ 4294967295u);
            }
            if (cw_need > 1) {
                unsigned int cw_mine_1 = cw_alive >> (unsigned int)lane & 1;
                unsigned int cw_ak_1 = (unsigned int)0;
                if (cw_mine_1 != (unsigned int)0) {
                    cw_ak_1 = my_key;
                }
                unsigned int cw_m_1;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(cw_m_1) : "r"(cw_ak_1));
                unsigned int _warp_vote_6 = __ballot_sync(0xFFFFFFFF, my_key == cw_m_1);
                unsigned int cw_eq_1 = cw_alive & _warp_vote_6;
                unsigned int cw_pick_1 = cw_eq_1 & (cw_eq_1 ^ 4294967295u) + (unsigned int)1;
                cw_c = cw_c | cw_pick_1;
                cw_alive = cw_alive & (cw_pick_1 ^ 4294967295u);
            }
            if (cw_need > 2) {
                unsigned int cw_mine_2 = cw_alive >> (unsigned int)lane & 1;
                unsigned int cw_ak_2 = (unsigned int)0;
                if (cw_mine_2 != (unsigned int)0) {
                    cw_ak_2 = my_key;
                }
                unsigned int cw_m_2;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(cw_m_2) : "r"(cw_ak_2));
                unsigned int _warp_vote_7 = __ballot_sync(0xFFFFFFFF, my_key == cw_m_2);
                unsigned int cw_eq_2 = cw_alive & _warp_vote_7;
                unsigned int cw_pick_2 = cw_eq_2 & (cw_eq_2 ^ 4294967295u) + (unsigned int)1;
                cw_c = cw_c | cw_pick_2;
                cw_alive = cw_alive & (cw_pick_2 ^ 4294967295u);
            }
            if (cw_need > 3) {
                unsigned int cw_mine_3 = cw_alive >> (unsigned int)lane & 1;
                unsigned int cw_ak_3 = (unsigned int)0;
                if (cw_mine_3 != (unsigned int)0) {
                    cw_ak_3 = my_key;
                }
                unsigned int cw_m_3;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(cw_m_3) : "r"(cw_ak_3));
                unsigned int _warp_vote_8 = __ballot_sync(0xFFFFFFFF, my_key == cw_m_3);
                unsigned int cw_eq_3 = cw_alive & _warp_vote_8;
                unsigned int cw_pick_3 = cw_eq_3 & (cw_eq_3 ^ 4294967295u) + (unsigned int)1;
                cw_c = cw_c | cw_pick_3;
                cw_alive = cw_alive & (cw_pick_3 ^ 4294967295u);
            }
            if (cw_need > 4) {
                unsigned int cw_mine_4 = cw_alive >> (unsigned int)lane & 1;
                unsigned int cw_ak_4 = (unsigned int)0;
                if (cw_mine_4 != (unsigned int)0) {
                    cw_ak_4 = my_key;
                }
                unsigned int cw_m_4;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(cw_m_4) : "r"(cw_ak_4));
                unsigned int _warp_vote_9 = __ballot_sync(0xFFFFFFFF, my_key == cw_m_4);
                unsigned int cw_eq_4 = cw_alive & _warp_vote_9;
                unsigned int cw_pick_4 = cw_eq_4 & (cw_eq_4 ^ 4294967295u) + (unsigned int)1;
                cw_c = cw_c | cw_pick_4;
                cw_alive = cw_alive & (cw_pick_4 ^ 4294967295u);
            }
            if (cw_need > 5) {
                unsigned int cw_mine_5 = cw_alive >> (unsigned int)lane & 1;
                unsigned int cw_ak_5 = (unsigned int)0;
                if (cw_mine_5 != (unsigned int)0) {
                    cw_ak_5 = my_key;
                }
                unsigned int cw_m_5;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(cw_m_5) : "r"(cw_ak_5));
                unsigned int _warp_vote_10 = __ballot_sync(0xFFFFFFFF, my_key == cw_m_5);
                unsigned int cw_eq_5 = cw_alive & _warp_vote_10;
                unsigned int cw_pick_5 = cw_eq_5 & (cw_eq_5 ^ 4294967295u) + (unsigned int)1;
                cw_c = cw_c | cw_pick_5;
                cw_alive = cw_alive & (cw_pick_5 ^ 4294967295u);
            }
            if (cw_need > 6) {
                unsigned int cw_mine_6 = cw_alive >> (unsigned int)lane & 1;
                unsigned int cw_ak_6 = (unsigned int)0;
                if (cw_mine_6 != (unsigned int)0) {
                    cw_ak_6 = my_key;
                }
                unsigned int cw_m_6;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(cw_m_6) : "r"(cw_ak_6));
                unsigned int _warp_vote_11 = __ballot_sync(0xFFFFFFFF, my_key == cw_m_6);
                unsigned int cw_eq_6 = cw_alive & _warp_vote_11;
                unsigned int cw_pick_6 = cw_eq_6 & (cw_eq_6 ^ 4294967295u) + (unsigned int)1;
                cw_c = cw_c | cw_pick_6;
                cw_alive = cw_alive & (cw_pick_6 ^ 4294967295u);
            }
            if (cw_need > 7) {
                unsigned int cw_mine_7 = cw_alive >> (unsigned int)lane & 1;
                unsigned int cw_ak_7 = (unsigned int)0;
                if (cw_mine_7 != (unsigned int)0) {
                    cw_ak_7 = my_key;
                }
                unsigned int cw_m_7;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(cw_m_7) : "r"(cw_ak_7));
                unsigned int _warp_vote_12 = __ballot_sync(0xFFFFFFFF, my_key == cw_m_7);
                unsigned int cw_eq_7 = cw_alive & _warp_vote_12;
                unsigned int cw_pick_7 = cw_eq_7 & (cw_eq_7 ^ 4294967295u) + (unsigned int)1;
                cw_c = cw_c | cw_pick_7;
                cw_alive = cw_alive & (cw_pick_7 ^ 4294967295u);
            }
            if (cw_need > 8) {
                unsigned int cw_mine_8 = cw_alive >> (unsigned int)lane & 1;
                unsigned int cw_ak_8 = (unsigned int)0;
                if (cw_mine_8 != (unsigned int)0) {
                    cw_ak_8 = my_key;
                }
                unsigned int cw_m_8;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(cw_m_8) : "r"(cw_ak_8));
                unsigned int _warp_vote_13 = __ballot_sync(0xFFFFFFFF, my_key == cw_m_8);
                unsigned int cw_eq_8 = cw_alive & _warp_vote_13;
                unsigned int cw_pick_8 = cw_eq_8 & (cw_eq_8 ^ 4294967295u) + (unsigned int)1;
                cw_c = cw_c | cw_pick_8;
                cw_alive = cw_alive & (cw_pick_8 ^ 4294967295u);
            }
            if (cw_need > 9) {
                unsigned int cw_mine_9 = cw_alive >> (unsigned int)lane & 1;
                unsigned int cw_ak_9 = (unsigned int)0;
                if (cw_mine_9 != (unsigned int)0) {
                    cw_ak_9 = my_key;
                }
                unsigned int cw_m_9;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(cw_m_9) : "r"(cw_ak_9));
                unsigned int _warp_vote_14 = __ballot_sync(0xFFFFFFFF, my_key == cw_m_9);
                unsigned int cw_eq_9 = cw_alive & _warp_vote_14;
                unsigned int cw_pick_9 = cw_eq_9 & (cw_eq_9 ^ 4294967295u) + (unsigned int)1;
                cw_c = cw_c | cw_pick_9;
                cw_alive = cw_alive & (cw_pick_9 ^ 4294967295u);
            }
            if (cw_need > 10) {
                unsigned int cw_mine_10 = cw_alive >> (unsigned int)lane & 1;
                unsigned int cw_ak_10 = (unsigned int)0;
                if (cw_mine_10 != (unsigned int)0) {
                    cw_ak_10 = my_key;
                }
                unsigned int cw_m_10;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(cw_m_10) : "r"(cw_ak_10));
                unsigned int _warp_vote_15 = __ballot_sync(0xFFFFFFFF, my_key == cw_m_10);
                unsigned int cw_eq_10 = cw_alive & _warp_vote_15;
                unsigned int cw_pick_10 = cw_eq_10 & (cw_eq_10 ^ 4294967295u) + (unsigned int)1;
                cw_c = cw_c | cw_pick_10;
                cw_alive = cw_alive & (cw_pick_10 ^ 4294967295u);
            }
            if (cw_need > 11) {
                unsigned int cw_mine_11 = cw_alive >> (unsigned int)lane & 1;
                unsigned int cw_ak_11 = (unsigned int)0;
                if (cw_mine_11 != (unsigned int)0) {
                    cw_ak_11 = my_key;
                }
                unsigned int cw_m_11;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(cw_m_11) : "r"(cw_ak_11));
                unsigned int _warp_vote_16 = __ballot_sync(0xFFFFFFFF, my_key == cw_m_11);
                unsigned int cw_eq_11 = cw_alive & _warp_vote_16;
                unsigned int cw_pick_11 = cw_eq_11 & (cw_eq_11 ^ 4294967295u) + (unsigned int)1;
                cw_c = cw_c | cw_pick_11;
                cw_alive = cw_alive & (cw_pick_11 ^ 4294967295u);
            }
            if (cw_need > 12) {
                unsigned int cw_mine_12 = cw_alive >> (unsigned int)lane & 1;
                unsigned int cw_ak_12 = (unsigned int)0;
                if (cw_mine_12 != (unsigned int)0) {
                    cw_ak_12 = my_key;
                }
                unsigned int cw_m_12;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(cw_m_12) : "r"(cw_ak_12));
                unsigned int _warp_vote_17 = __ballot_sync(0xFFFFFFFF, my_key == cw_m_12);
                unsigned int cw_eq_12 = cw_alive & _warp_vote_17;
                unsigned int cw_pick_12 = cw_eq_12 & (cw_eq_12 ^ 4294967295u) + (unsigned int)1;
                cw_c = cw_c | cw_pick_12;
                cw_alive = cw_alive & (cw_pick_12 ^ 4294967295u);
            }
            if (cw_need > 13) {
                unsigned int cw_mine_13 = cw_alive >> (unsigned int)lane & 1;
                unsigned int cw_ak_13 = (unsigned int)0;
                if (cw_mine_13 != (unsigned int)0) {
                    cw_ak_13 = my_key;
                }
                unsigned int cw_m_13;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(cw_m_13) : "r"(cw_ak_13));
                unsigned int _warp_vote_18 = __ballot_sync(0xFFFFFFFF, my_key == cw_m_13);
                unsigned int cw_eq_13 = cw_alive & _warp_vote_18;
                unsigned int cw_pick_13 = cw_eq_13 & (cw_eq_13 ^ 4294967295u) + (unsigned int)1;
                cw_c = cw_c | cw_pick_13;
                cw_alive = cw_alive & (cw_pick_13 ^ 4294967295u);
            }
            if (cw_need > 14) {
                unsigned int cw_mine_14 = cw_alive >> (unsigned int)lane & 1;
                unsigned int cw_ak_14 = (unsigned int)0;
                if (cw_mine_14 != (unsigned int)0) {
                    cw_ak_14 = my_key;
                }
                unsigned int cw_m_14;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(cw_m_14) : "r"(cw_ak_14));
                unsigned int _warp_vote_19 = __ballot_sync(0xFFFFFFFF, my_key == cw_m_14);
                unsigned int cw_eq_14 = cw_alive & _warp_vote_19;
                unsigned int cw_pick_14 = cw_eq_14 & (cw_eq_14 ^ 4294967295u) + (unsigned int)1;
                cw_c = cw_c | cw_pick_14;
                cw_alive = cw_alive & (cw_pick_14 ^ 4294967295u);
            }
            if (cw_need > 15) {
                unsigned int cw_mine_15 = cw_alive >> (unsigned int)lane & 1;
                unsigned int cw_ak_15 = (unsigned int)0;
                if (cw_mine_15 != (unsigned int)0) {
                    cw_ak_15 = my_key;
                }
                unsigned int cw_m_15;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(cw_m_15) : "r"(cw_ak_15));
                unsigned int _warp_vote_20 = __ballot_sync(0xFFFFFFFF, my_key == cw_m_15);
                unsigned int cw_eq_15 = cw_alive & _warp_vote_20;
                unsigned int cw_pick_15 = cw_eq_15 & (cw_eq_15 ^ 4294967295u) + (unsigned int)1;
                cw_c = cw_c | cw_pick_15;
                cw_alive = cw_alive & (cw_pick_15 ^ 4294967295u);
            }
        }
        unsigned int win_ballot = cw_c;
        if ((win_ballot >> (unsigned int)lane & (unsigned int)1) != (unsigned int)0) {
            int win_slot = (int)(unsigned int)__popc(win_ballot & lanemask_lt);
            if (win_slot < 16) {
                winner_ids[win_slot] = my_id;
                winner_scores[win_slot] = my_score;
            }
        }
        __syncwarp();
    }
    if (tid < TOP_K) {
        float selected_sum = 0.0f;
        #pragma unroll
        for (int route_sum = 0; route_sum < TOP_K; route_sum++) {
            selected_sum += winner_scores[route_sum];
        }
        float selected_norm = 1.0f;
        if (selected_sum > 0.0f) {
            selected_norm = selected_sum;
        }
        int selected_id = winner_ids[tid];
        unsigned long long output_index = (unsigned long long)token * (unsigned long long)TOP_K + (unsigned long long)tid;
        float _fdiv_21;
        asm volatile("div.full.f32 %0, %1, %2;" : "=f"(_fdiv_21) : "f"(winner_scores[tid]), "f"(selected_norm));
        topk_weights[output_index] = _fdiv_21;
        topk_ids[output_index] = selected_id;
        expert_ids[tid] = selected_id;
        expert_counts[selected_id] = 1;
        expert_scatter_offsets[selected_id] = 1;
    }
    __syncthreads();
    int plan_winner = winner_ids[lane & TOP_K - 1];
    int plan_rel = plan_winner - warp * (32 * ITEMS_PER_THREAD);
    unsigned int plan_below_warp = __ballot_sync(0xFFFFFFFF, plan_rel < 0);
    unsigned int plan_bit = (unsigned int)0;
    if (plan_rel >= 0 && plan_rel < 32 * ITEMS_PER_THREAD) {
        plan_bit = (unsigned int)1 << (unsigned int)(plan_rel >> 2);
    }
    int plan_item = plan_rel & ITEMS_PER_THREAD - 1;
    unsigned int plan_mask0;
    asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(plan_mask0) : "r"(((plan_item == 0) ? plan_bit : (unsigned int)0)));
    unsigned int plan_mask1;
    asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(plan_mask1) : "r"(((plan_item == 1) ? plan_bit : (unsigned int)0)));
    unsigned int plan_mask2;
    asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(plan_mask2) : "r"(((plan_item == 2) ? plan_bit : (unsigned int)0)));
    unsigned int plan_mask3;
    asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(plan_mask3) : "r"(((plan_item == 3) ? plan_bit : (unsigned int)0)));
    int plan_running = (int)(unsigned int)__popc(plan_below_warp & 65535) + ((int)(unsigned int)__popc(plan_mask0 & lanemask_lt) + (int)(unsigned int)__popc(plan_mask1 & lanemask_lt)) + ((int)(unsigned int)__popc(plan_mask2 & lanemask_lt) + (int)(unsigned int)__popc(plan_mask3 & lanemask_lt));
    int plan_cnt[4];
    int plan_off[4];
    int plan_flag0_1_0 = (int)(plan_mask0 >> (unsigned int)lane & 1);
    plan_cnt[0] = plan_flag0_1_0;
    plan_off[0] = plan_running << BLOCK_SHIFT;
    plan_running += plan_flag0_1_0;
    int plan_flag0_1_1 = (int)(plan_mask1 >> (unsigned int)lane & 1);
    plan_cnt[1] = plan_flag0_1_1;
    plan_off[1] = plan_running << BLOCK_SHIFT;
    plan_running += plan_flag0_1_1;
    int plan_flag0_1_2 = (int)(plan_mask2 >> (unsigned int)lane & 1);
    plan_cnt[2] = plan_flag0_1_2;
    plan_off[2] = plan_running << BLOCK_SHIFT;
    plan_running += plan_flag0_1_2;
    int plan_flag0_1_3 = (int)(plan_mask3 >> (unsigned int)lane & 1);
    plan_cnt[3] = plan_flag0_1_3;
    plan_off[3] = plan_running << BLOCK_SHIFT;
    plan_running += plan_flag0_1_3;
    {
        int4 _iv4 = make_int4(plan_off[0 + 0], plan_off[0 + 1], plan_off[0 + 2], plan_off[0 + 3]);
        *reinterpret_cast<int4*>(expert_offsets + expert_base) = _iv4;
    }
}

} // extern "C"
