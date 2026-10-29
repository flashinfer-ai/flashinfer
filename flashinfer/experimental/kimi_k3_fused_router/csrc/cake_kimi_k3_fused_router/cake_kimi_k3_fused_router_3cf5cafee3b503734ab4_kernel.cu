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
#define SMEM_BOUND_VALUES_OFF 17152
#define SMEM_BOUND_VALUES_STAGE_BYTES 128
#define SMEM_BOUND_VALUES_STRIDE 128
#define SMEM_CAND_COUNTS_OFF 17280
#define SMEM_CAND_COUNTS_STAGE_BYTES 32
#define SMEM_CAND_COUNTS_STRIDE 32
#define SMEM_CAND_KEYS_OFF 17312
#define SMEM_CAND_KEYS_STAGE_BYTES 896
#define SMEM_CAND_KEYS_STRIDE 896
#define SMEM_CAND_IDS_OFF 18208
#define SMEM_CAND_IDS_STAGE_BYTES 896
#define SMEM_CAND_IDS_STRIDE 896
#define SMEM_CAND_SCORES_OFF 19104
#define SMEM_CAND_SCORES_STAGE_BYTES 896
#define SMEM_CAND_SCORES_STRIDE 896
#define SMEM_RANK_KEYS_OFF 20000
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
#define SMEM_PLAN_OFFSETS_OFF 3584
#define SMEM_PLAN_OFFSETS_STAGE_BYTES 3584
#define SMEM_PLAN_OFFSETS_STRIDE 3584
#define SMEM_OWN_BITMAP_OFF 8192
#define SMEM_OWN_BITMAP_STAGE_BYTES 1792
#define SMEM_OWN_BITMAP_STRIDE 1792
#define SMEM_OWN_ROUTES_OFF 9984
#define SMEM_OWN_ROUTES_STAGE_BYTES 7168
#define SMEM_OWN_ROUTES_STRIDE 7168
#define SMEM_TOTAL 20224
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
#define GATHER_LOADS 5
#define GATHER_PASS_PAIRS 4480
#define OWNER_STRIDE 1
#define BITMAP_STRIDE 64
#define ROUTE_STRIDE 256
#define WORD_PASSES 2
#define BLOCK_MASK (BLOCK_M - 1)
#define BLOCK_SHIFT (3 + (BLOCK_M >> 4))
#define LAUNCH_MIN_BLOCKS 3

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

__global__ __launch_bounds__(224, LAUNCH_MIN_BLOCKS) void
kernel_cake_kimi_k3_fused_router_3cf5cafee3b503734ab4(float* __restrict__ logits, float* __restrict__ bias, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M)
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
    unsigned int* bound_values = reinterpret_cast<unsigned int*>(smem_raw + 17152);
    const int bound_values_addr = smem + 17152;
    int* cand_counts = reinterpret_cast<int*>(smem_raw + 17280);
    const int cand_counts_addr = smem + 17280;
    unsigned int* cand_keys = reinterpret_cast<unsigned int*>(smem_raw + 17312);
    const int cand_keys_addr = smem + 17312;
    int* cand_ids = reinterpret_cast<int*>(smem_raw + 18208);
    const int cand_ids_addr = smem + 18208;
    float* cand_scores = reinterpret_cast<float*>(smem_raw + 19104);
    const int cand_scores_addr = smem + 19104;
    unsigned int* rank_keys = reinterpret_cast<unsigned int*>(smem_raw + 20000);
    const int rank_keys_addr = smem + 20000;
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
    int* plan_offsets = reinterpret_cast<int*>(smem_raw + 3584);
    const int plan_offsets_addr = smem + 3584;
    unsigned int* own_bitmap = reinterpret_cast<unsigned int*>(smem_raw + 8192);
    const int own_bitmap_addr = smem + 8192;
    unsigned int* own_routes = reinterpret_cast<unsigned int*>(smem_raw + 9984);
    const int own_routes_addr = smem + 9984;

    // === Task calls (declaration order) ===
    int global_thread = bid * THREADS + tid;
    int grid_threads = num_bids * THREADS;
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
    #pragma unroll 1
    for (int token = bid; token < M; token += num_bids) {
        unsigned long long row_base = (unsigned long long)token * (unsigned long long)NUM_EXPERTS;
        float _vec_load_1[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(logits + row_base + (unsigned long long)expert_base);
            _vec_load_1[0 + 0] = _v4.x;
            _vec_load_1[0 + 1] = _v4.y;
            _vec_load_1[0 + 2] = _v4.z;
            _vec_load_1[0 + 3] = _v4.w;
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
        #pragma unroll
        for (int item_cand = 0; item_cand < 4; item_cand++) {
            if (keys[item_cand] >= cand_floor) {
                if (cand_slot < 32) {
                    int cand_index = warp * 32 + cand_slot;
                    cand_keys[cand_index] = keys[item_cand];
                    cand_ids[cand_index] = expert_base + item_cand;
                    cand_scores[cand_index] = unbiased[item_cand];
                }
                cand_slot += 1;
            }
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
            rank_keys[lane] = my_key;
            __syncwarp();
            int rank_part0 = 0;
            int rank_part1 = 0;
            int rank_part2 = 0;
            int rank_part3 = 0;
            rank_part0 += ((my_key < rank_keys[0]) ? 1 : 0);
            rank_part1 += ((my_key < rank_keys[1]) ? 1 : 0);
            rank_part2 += ((my_key < rank_keys[2]) ? 1 : 0);
            rank_part3 += ((my_key < rank_keys[3]) ? 1 : 0);
            rank_part0 += ((my_key < rank_keys[4]) ? 1 : 0);
            rank_part1 += ((my_key < rank_keys[5]) ? 1 : 0);
            rank_part2 += ((my_key < rank_keys[6]) ? 1 : 0);
            rank_part3 += ((my_key < rank_keys[7]) ? 1 : 0);
            rank_part0 += ((my_key < rank_keys[8]) ? 1 : 0);
            rank_part1 += ((my_key < rank_keys[9]) ? 1 : 0);
            rank_part2 += ((my_key < rank_keys[10]) ? 1 : 0);
            rank_part3 += ((my_key < rank_keys[11]) ? 1 : 0);
            rank_part0 += ((my_key < rank_keys[12]) ? 1 : 0);
            rank_part1 += ((my_key < rank_keys[13]) ? 1 : 0);
            rank_part2 += ((my_key < rank_keys[14]) ? 1 : 0);
            rank_part3 += ((my_key < rank_keys[15]) ? 1 : 0);
            if (cand_total > 16) {
                rank_part0 += ((my_key < rank_keys[16]) ? 1 : 0);
                rank_part1 += ((my_key < rank_keys[17]) ? 1 : 0);
                rank_part2 += ((my_key < rank_keys[18]) ? 1 : 0);
                rank_part3 += ((my_key < rank_keys[19]) ? 1 : 0);
                rank_part0 += ((my_key < rank_keys[20]) ? 1 : 0);
                rank_part1 += ((my_key < rank_keys[21]) ? 1 : 0);
                rank_part2 += ((my_key < rank_keys[22]) ? 1 : 0);
                rank_part3 += ((my_key < rank_keys[23]) ? 1 : 0);
            }
            if (cand_total > 24) {
                rank_part0 += ((my_key < rank_keys[24]) ? 1 : 0);
                rank_part1 += ((my_key < rank_keys[25]) ? 1 : 0);
                rank_part2 += ((my_key < rank_keys[26]) ? 1 : 0);
                rank_part3 += ((my_key < rank_keys[27]) ? 1 : 0);
                rank_part0 += ((my_key < rank_keys[28]) ? 1 : 0);
                rank_part1 += ((my_key < rank_keys[29]) ? 1 : 0);
                rank_part2 += ((my_key < rank_keys[30]) ? 1 : 0);
                rank_part3 += ((my_key < rank_keys[31]) ? 1 : 0);
            }
            int my_rank = rank_part0 + rank_part1 + (rank_part2 + rank_part3);
            unsigned int rank_valid = 0;
            if (cand_total > lane) {
                rank_valid = (unsigned int)my_rank;
            }
            unsigned int rank_sum;
            asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(rank_sum) : "r"(rank_valid));
            if ((int)rank_sum != cand_total * (cand_total - 1) >> 1) {
                if (rank_keys[0] == my_key) {
                    if (lane > 0) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[1] == my_key) {
                    if (lane > 1) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[2] == my_key) {
                    if (lane > 2) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[3] == my_key) {
                    if (lane > 3) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[4] == my_key) {
                    if (lane > 4) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[5] == my_key) {
                    if (lane > 5) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[6] == my_key) {
                    if (lane > 6) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[7] == my_key) {
                    if (lane > 7) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[8] == my_key) {
                    if (lane > 8) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[9] == my_key) {
                    if (lane > 9) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[10] == my_key) {
                    if (lane > 10) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[11] == my_key) {
                    if (lane > 11) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[12] == my_key) {
                    if (lane > 12) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[13] == my_key) {
                    if (lane > 13) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[14] == my_key) {
                    if (lane > 14) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[15] == my_key) {
                    if (lane > 15) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[16] == my_key) {
                    if (lane > 16) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[17] == my_key) {
                    if (lane > 17) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[18] == my_key) {
                    if (lane > 18) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[19] == my_key) {
                    if (lane > 19) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[20] == my_key) {
                    if (lane > 20) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[21] == my_key) {
                    if (lane > 21) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[22] == my_key) {
                    if (lane > 22) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[23] == my_key) {
                    if (lane > 23) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[24] == my_key) {
                    if (lane > 24) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[25] == my_key) {
                    if (lane > 25) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[26] == my_key) {
                    if (lane > 26) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[27] == my_key) {
                    if (lane > 27) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[28] == my_key) {
                    if (lane > 28) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[29] == my_key) {
                    if (lane > 29) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[30] == my_key) {
                    if (lane > 30) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[31] == my_key) {
                    if (lane > 31) {
                        my_rank += 1;
                    }
                }
            }
            unsigned int win_ballot = __ballot_sync(0xFFFFFFFF, my_rank < 16);
            if (my_rank < 16) {
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
            float _fdiv_4;
            asm volatile("div.full.f32 %0, %1, %2;" : "=f"(_fdiv_4) : "f"(winner_scores[tid]), "f"(selected_norm));
            topk_weights[output_index] = _fdiv_4;
            topk_ids[output_index] = selected_id;
        }
        __syncthreads();
    }
    unsigned int _grid_arrive_0 = cooperative_groups::this_grid().barrier_arrive();
    unsigned int join_token = _grid_arrive_0;
    int owner_index = bid / OWNER_STRIDE;
    int own_expert_base = owner_index * NUM_WARPS;
    if (bid % OWNER_STRIDE == 0 && owner_index < OWNER_CTAS) {
        #pragma unroll 1
        for (int count_clear = tid; count_clear < NUM_EXPERTS; count_clear += THREADS) {
            plan_counts[count_clear] = 0;
        }
        #pragma unroll 1
        for (int bitmap_clear = tid; bitmap_clear < NUM_WARPS * BITMAP_STRIDE; bitmap_clear += THREADS) {
            own_bitmap[bitmap_clear] = (unsigned int)0;
        }
        #pragma unroll 1
        for (int route_clear = tid; route_clear < NUM_WARPS * ROUTE_STRIDE; route_clear += THREADS) {
            own_routes[route_clear] = (unsigned int)0;
        }
    }
    if (bid % OWNER_STRIDE == 0 && owner_index < OWNER_CTAS) {
        cooperative_groups::this_grid().barrier_wait(static_cast<cooperative_groups::grid_group::arrival_token>(join_token));
        int total_pairs = M * TOP_K;
        int bitmap_words = M + 31 >> 5;
        #pragma unroll 1
        for (int gather_base = 0; gather_base < total_pairs; gather_base += GATHER_PASS_PAIRS) {
            int gathered[20];
            #pragma unroll
            for (int gather_slot = 0; gather_slot < GATHER_LOADS; gather_slot++) {
                #pragma unroll
                for (int gather_lane = 0; gather_lane < 4; gather_lane++) {
                    gathered[gather_slot * 4 + gather_lane] = -1;
                }
                int gather_pair = gather_base + (gather_slot * THREADS + tid) * 4;
                if (gather_pair < total_pairs) {
                    {
                        const int4* _ivptr_5 = reinterpret_cast<const int4*>(topk_ids + gather_pair);
                        int4 _ivld_5;
                        _ivld_5 = *_ivptr_5;
                        gathered[gather_slot * 4 + 0] = _ivld_5.x;
                        gathered[gather_slot * 4 + 1] = _ivld_5.y;
                        gathered[gather_slot * 4 + 2] = _ivld_5.z;
                        gathered[gather_slot * 4 + 3] = _ivld_5.w;
                    }
                }
            }
            unsigned int hit_mask = 0;
            unsigned int hit_slots_lo = 0;
            unsigned int hit_slots_hi = 0;
            #pragma unroll
            for (int count_slot = 0; count_slot < GATHER_LOADS; count_slot++) {
                int count_pair = gather_base + (count_slot * THREADS + tid) * 4;
                if (count_pair < total_pairs) {
                    #pragma unroll
                    for (int count_lane = 0; count_lane < 4; count_lane++) {
                        int count_id = gathered[count_slot * 4 + count_lane];
                        atomicAdd(&plan_counts[count_id], 1);
                        int count_own = count_id - own_expert_base;
                        if (count_own >= 0 && count_own < NUM_WARPS) {
                            hit_mask |= (unsigned int)1 << (unsigned int)(count_slot * 4 + count_lane);
                            if (count_slot * 4 + count_lane < 10) {
                                hit_slots_lo |= (unsigned int)count_own << (unsigned int)((count_slot * 4 + count_lane) * 3);
                            } else {
                                hit_slots_hi |= (unsigned int)count_own << (unsigned int)((count_slot * 4 + count_lane - 10) * 3);
                            }
                        }
                    }
                }
            }
            int hit_count = (int)(unsigned int)__popc(hit_mask);
            #pragma unroll 1
            for (int hit = 0; hit < hit_count; hit++) {
                int _find_first_set_6 = __ffs(hit_mask);
                int hit_bit = _find_first_set_6 - 1;
                int hit_pair = gather_base + ((hit_bit >> 2) * THREADS + tid) * 4 + (hit_bit & 3);
                unsigned int hit_slots = ((hit_bit >= 10) ? hit_slots_hi : hit_slots_lo);
                int hit_shift = ((hit_bit >= 10) ? hit_bit - 10 : hit_bit) * 3;
                int hit_slot = (int)(hit_slots >> (unsigned int)hit_shift & (unsigned int)7);
                int hit_token = hit_pair >> 4;
                int hit_route = hit_pair & 15;
                atomicAdd(&own_bitmap[hit_slot * BITMAP_STRIDE + (hit_token >> 5)], (unsigned int)1 << (unsigned int)(hit_token & 31));
                atomicAdd(&own_routes[hit_slot * ROUTE_STRIDE + (hit_token >> 3)], (unsigned int)hit_route << (unsigned int)((hit_token & 7) * 4));
                hit_mask = hit_mask & hit_mask - (unsigned int)1;
            }
        }
        __syncthreads();
        int group_total = 0;
        group_total += plan_counts[tid * 4 + 0] + BLOCK_MASK & ~BLOCK_MASK;
        group_total += plan_counts[tid * 4 + 1] + BLOCK_MASK & ~BLOCK_MASK;
        group_total += plan_counts[tid * 4 + 2] + BLOCK_MASK & ~BLOCK_MASK;
        group_total += plan_counts[tid * 4 + 3] + BLOCK_MASK & ~BLOCK_MASK;
        plan_offsets[tid] = group_total;
        __syncthreads();
        int first_group = own_expert_base >> 2;
        int warp_expert = own_expert_base + warp;
        int below_part = 0;
        int all_part = 0;
        all_part += plan_offsets[lane + 0];
        if (lane + 0 < first_group) {
            below_part += plan_offsets[lane + 0];
        }
        all_part += plan_offsets[lane + 32];
        if (lane + 32 < first_group) {
            below_part += plan_offsets[lane + 32];
        }
        all_part += plan_offsets[lane + 64];
        if (lane + 64 < first_group) {
            below_part += plan_offsets[lane + 64];
        }
        all_part += plan_offsets[lane + 96];
        if (lane + 96 < first_group) {
            below_part += plan_offsets[lane + 96];
        }
        all_part += plan_offsets[lane + 128];
        if (lane + 128 < first_group) {
            below_part += plan_offsets[lane + 128];
        }
        all_part += plan_offsets[lane + 160];
        if (lane + 160 < first_group) {
            below_part += plan_offsets[lane + 160];
        }
        all_part += plan_offsets[lane + 192];
        if (lane + 192 < first_group) {
            below_part += plan_offsets[lane + 192];
        }
        if ((first_group << 2) + lane < warp_expert) {
            below_part += plan_counts[(first_group << 2) + lane] + BLOCK_MASK & ~BLOCK_MASK;
        }
        unsigned int _warp_redux_7;
        asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_7) : "r"((unsigned int)below_part));
        int warp_offset = (int)_warp_redux_7;
        if (bid == 0) {
            if (warp == 0) {
                unsigned int _warp_redux_8;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_8) : "r"((unsigned int)all_part));
                int padded_total = (int)_warp_redux_8;
                if (lane == 0) {
                    expert_offsets[NUM_EXPERTS] = padded_total;
                    num_tokens_post_padded[0] = padded_total;
                }
            }
        }
        if (lane == 0) {
            int publish_count = plan_counts[warp_expert];
            expert_counts[warp_expert] = publish_count;
            expert_scatter_offsets[warp_expert] = publish_count;
            expert_offsets[warp_expert] = warp_offset;
        }
        int own_expert = own_expert_base + warp;
        int own_count = plan_counts[own_expert];
        int own_base = warp_offset;
        int own_padded = own_count + BLOCK_MASK & ~BLOCK_MASK;
        int own_bitmap_base = warp * BITMAP_STRIDE;
        int own_route_base = warp * ROUTE_STRIDE;
        int emitted = 0;
        if (own_count > 0) {
            #pragma unroll 1
            for (int byte_pass = 0; byte_pass < bitmap_words + 7 >> 3; byte_pass++) {
                int byte_word = (byte_pass << 3) + (lane >> 2);
                unsigned int pass_words[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&pass_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&pass_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pass_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pass_words[(0) + 3]))
                    : "r"(own_bitmap_addr + (unsigned int)((own_bitmap_base + (byte_pass << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&pass_words[4])), "=r"(*reinterpret_cast<uint32_t*>(&pass_words[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pass_words[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pass_words[(4) + 3]))
                    : "r"(own_bitmap_addr + (unsigned int)((own_bitmap_base + (byte_pass << 3) + 4) * 4)));
                unsigned int byte_full = 0;
                unsigned int byte_routes = 0;
                if (byte_word < bitmap_words) {
                    byte_full = own_bitmap[own_bitmap_base + byte_word];
                    byte_routes = own_routes[own_route_base + (byte_pass << 5) + lane];
                }
                unsigned int byte_shift = (unsigned int)((lane & 3) * 8);
                unsigned int byte_bits = byte_full >> byte_shift & (unsigned int)255;
                int byte_count = (int)(unsigned int)__popc(byte_bits);
                unsigned int byte_below = ((unsigned int)1 << byte_shift) - (unsigned int)1;
                int byte_rank = own_base + emitted + (int)(unsigned int)__popc(byte_full & byte_below);
                int pass_total = 0;
                int word_count_0 = (int)(unsigned int)__popc(pass_words[0]);
                pass_total += word_count_0;
                if (0 < lane >> 2) {
                    byte_rank += word_count_0;
                }
                int word_count_1 = (int)(unsigned int)__popc(pass_words[1]);
                pass_total += word_count_1;
                if (1 < lane >> 2) {
                    byte_rank += word_count_1;
                }
                int word_count_2 = (int)(unsigned int)__popc(pass_words[2]);
                pass_total += word_count_2;
                if (2 < lane >> 2) {
                    byte_rank += word_count_2;
                }
                int word_count_3 = (int)(unsigned int)__popc(pass_words[3]);
                pass_total += word_count_3;
                if (3 < lane >> 2) {
                    byte_rank += word_count_3;
                }
                int word_count_4 = (int)(unsigned int)__popc(pass_words[4]);
                pass_total += word_count_4;
                if (4 < lane >> 2) {
                    byte_rank += word_count_4;
                }
                int word_count_5 = (int)(unsigned int)__popc(pass_words[5]);
                pass_total += word_count_5;
                if (5 < lane >> 2) {
                    byte_rank += word_count_5;
                }
                int word_count_6 = (int)(unsigned int)__popc(pass_words[6]);
                pass_total += word_count_6;
                if (6 < lane >> 2) {
                    byte_rank += word_count_6;
                }
                int word_count_7 = (int)(unsigned int)__popc(pass_words[7]);
                pass_total += word_count_7;
                if (7 < lane >> 2) {
                    byte_rank += word_count_7;
                }
                int byte_token_base = (byte_pass << 8) + (lane << 3);
                #pragma unroll 1
                for (int byte_hit = 0; byte_hit < byte_count; byte_hit++) {
                    int _find_first_set_9 = __ffs(byte_bits);
                    int byte_offset = _find_first_set_9 - 1;
                    unsigned int byte_route = byte_routes >> (unsigned int)(byte_offset * 4) & (unsigned int)15;
                    int byte_pair = (byte_token_base + byte_offset) * TOP_K + (int)byte_route;
                    sorted_token_ids[byte_rank + byte_hit] = byte_pair;
                    byte_bits = byte_bits & byte_bits - (unsigned int)1;
                }
                emitted += pass_total;
            }
        }
        #pragma unroll 1
        for (int pad_slot = own_count + lane; pad_slot < own_padded; pad_slot += 32) {
            sorted_token_ids[own_base + pad_slot] = total_pairs;
        }
        #pragma unroll 1
        for (int block_slot = lane; block_slot < own_padded >> BLOCK_SHIFT; block_slot += 32) {
            expert_ids[(own_base >> BLOCK_SHIFT) + block_slot] = own_expert;
        }
    }
}

} // extern "C"
