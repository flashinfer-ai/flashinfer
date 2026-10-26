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
#define SMEM_BOUND_VALUES_OFF 9312
#define SMEM_BOUND_VALUES_STAGE_BYTES 128
#define SMEM_BOUND_VALUES_STRIDE 128
#define SMEM_CAND_COUNTS_OFF 9440
#define SMEM_CAND_COUNTS_STAGE_BYTES 32
#define SMEM_CAND_COUNTS_STRIDE 32
#define SMEM_CAND_KEYS_OFF 9472
#define SMEM_CAND_KEYS_STAGE_BYTES 896
#define SMEM_CAND_KEYS_STRIDE 896
#define SMEM_CAND_IDS_OFF 10368
#define SMEM_CAND_IDS_STAGE_BYTES 896
#define SMEM_CAND_IDS_STRIDE 896
#define SMEM_CAND_SCORES_OFF 11264
#define SMEM_CAND_SCORES_STAGE_BYTES 896
#define SMEM_CAND_SCORES_STRIDE 896
#define SMEM_RANK_KEYS_OFF 12160
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
#define SMEM_OWN_BITMAP_STAGE_BYTES 224
#define SMEM_OWN_BITMAP_STRIDE 224
#define SMEM_OWN_ROUTES_OFF 8416
#define SMEM_OWN_ROUTES_STAGE_BYTES 896
#define SMEM_OWN_ROUTES_STRIDE 896
#define SMEM_TOTAL 12288
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
#define BITMAP_STRIDE 8
#define ROUTE_STRIDE 32
#define WORD_PASSES 1
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
kernel_cake_kimi_k3_fused_router_49a2f49970db28d2d0b3(float* __restrict__ logits, float* __restrict__ bias, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M)
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
    unsigned int* bound_values = reinterpret_cast<unsigned int*>(smem_raw + 9312);
    const int bound_values_addr = smem + 9312;
    int* cand_counts = reinterpret_cast<int*>(smem_raw + 9440);
    const int cand_counts_addr = smem + 9440;
    unsigned int* cand_keys = reinterpret_cast<unsigned int*>(smem_raw + 9472);
    const int cand_keys_addr = smem + 9472;
    int* cand_ids = reinterpret_cast<int*>(smem_raw + 10368);
    const int cand_ids_addr = smem + 10368;
    float* cand_scores = reinterpret_cast<float*>(smem_raw + 11264);
    const int cand_scores_addr = smem + 11264;
    unsigned int* rank_keys = reinterpret_cast<unsigned int*>(smem_raw + 12160);
    const int rank_keys_addr = smem + 12160;
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
    unsigned int* own_routes = reinterpret_cast<unsigned int*>(smem_raw + 8416);
    const int own_routes_addr = smem + 8416;

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
        unsigned int warp_top0;
        asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(warp_top0) : "r"(lane_bound));
        if (lane == 0) {
            published = warp_top0;
        }
        if (lane_bound == warp_top0) {
            lane_bound = (unsigned int)0;
        }
        unsigned int warp_top1;
        asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(warp_top1) : "r"(lane_bound));
        if (lane == 1) {
            published = warp_top1;
        }
        if (lane_bound == warp_top1) {
            lane_bound = (unsigned int)0;
        }
        unsigned int warp_top2;
        asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(warp_top2) : "r"(lane_bound));
        if (lane == 2) {
            published = warp_top2;
        }
        if (lane_bound == warp_top2) {
            lane_bound = (unsigned int)0;
        }
        unsigned int warp_top3;
        asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(warp_top3) : "r"(lane_bound));
        if (lane == 3) {
            published = warp_top3;
        }
        if (lane_bound == warp_top3) {
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
                int hit_id = topk_ids[hit_pair];
                int hit_slot = hit_id - own_expert_base;
                int hit_token = hit_pair >> 4;
                int hit_route = hit_pair & 15;
                atomicAdd(&own_bitmap[hit_slot * BITMAP_STRIDE + (hit_token >> 5)], (unsigned int)1 << (unsigned int)(hit_token & 31));
                atomicAdd(&own_routes[hit_slot * ROUTE_STRIDE + (hit_token >> 3)], (unsigned int)hit_route << (unsigned int)((hit_token & 7) * 4));
                hit_mask = hit_mask & hit_mask - (unsigned int)1;
            }
        }
        __syncthreads();
        int expert_scan_base = warp * 32 * ITEMS_PER_THREAD + lane * ITEMS_PER_THREAD;
        int lane_prefixes[4];
        int lane_total = 0;
        #pragma unroll
        for (int expert_slot_scan = 0; expert_slot_scan < ITEMS_PER_THREAD; expert_slot_scan++) {
            int expert_scan = expert_scan_base + expert_slot_scan;
            int count_scan = plan_counts[expert_scan];
            lane_prefixes[expert_slot_scan] = lane_total;
            int padded_count_scan = count_scan + BLOCK_MASK & ~BLOCK_MASK;
            lane_total += padded_count_scan;
        }
        int warp_inclusive = lane_total;
        int phase2_scan_peer = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, 1, 32);
        if (lane >= 1) {
            warp_inclusive += phase2_scan_peer;
        }
        int phase2_scan_peer_0 = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, 2, 32);
        if (lane >= 2) {
            warp_inclusive += phase2_scan_peer_0;
        }
        int phase2_scan_peer_1 = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, 4, 32);
        if (lane >= 4) {
            warp_inclusive += phase2_scan_peer_1;
        }
        int phase2_scan_peer_2 = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, 8, 32);
        if (lane >= 8) {
            warp_inclusive += phase2_scan_peer_2;
        }
        int phase2_scan_peer_3 = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, 16, 32);
        if (lane >= 16) {
            warp_inclusive += phase2_scan_peer_3;
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
            plan_offsets[expert_store] = offset_store;
        }
        if (bid == 0) {
            if (tid == THREADS - 1) {
                int padded_total = phase2_warp_prefix + warp_inclusive;
                expert_offsets[NUM_EXPERTS] = padded_total;
                num_tokens_post_padded[0] = padded_total;
            }
        }
        __syncthreads();
        if (tid < NUM_WARPS) {
            int publish_expert = own_expert_base + tid;
            int publish_count = plan_counts[publish_expert];
            int publish_offset = plan_offsets[publish_expert];
            expert_counts[publish_expert] = publish_count;
            expert_scatter_offsets[publish_expert] = publish_count;
            expert_offsets[publish_expert] = publish_offset;
        }
        int own_expert = own_expert_base + warp;
        int own_count = plan_counts[own_expert];
        int own_base = plan_offsets[own_expert];
        int own_padded = own_count + BLOCK_MASK & ~BLOCK_MASK;
        int own_bitmap_base = warp * BITMAP_STRIDE;
        int own_route_base = warp * ROUTE_STRIDE;
        int emitted = 0;
        if (own_count > 0) {
            #pragma unroll
            for (int word_pass = 0; word_pass < WORD_PASSES; word_pass++) {
                if (bitmap_words > word_pass * 32) {
                    int pass_word = word_pass * 32 + lane;
                    unsigned int pass_bits = 0;
                    unsigned int pass_routes[4];
                    if (pass_word < bitmap_words) {
                        pass_bits = own_bitmap[own_bitmap_base + pass_word];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pass_routes[0])), "=r"(*reinterpret_cast<uint32_t*>(&pass_routes[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pass_routes[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pass_routes[(0) + 3]))
                            : "r"(own_routes_addr + (unsigned int)((own_route_base + pass_word * 4) * 4)));
                    }
                    int pass_count = (int)(unsigned int)__popc(pass_bits);
                    int pass_inclusive = pass_count;
                    int pass_peer = __shfl_up_sync(0xFFFFFFFF, pass_inclusive, 1, 32);
                    if (lane >= 1) {
                        pass_inclusive += pass_peer;
                    }
                    int pass_peer_0 = __shfl_up_sync(0xFFFFFFFF, pass_inclusive, 2, 32);
                    if (lane >= 2) {
                        pass_inclusive += pass_peer_0;
                    }
                    int pass_peer_1 = __shfl_up_sync(0xFFFFFFFF, pass_inclusive, 4, 32);
                    if (lane >= 4) {
                        pass_inclusive += pass_peer_1;
                    }
                    int pass_peer_2 = __shfl_up_sync(0xFFFFFFFF, pass_inclusive, 8, 32);
                    if (lane >= 8) {
                        pass_inclusive += pass_peer_2;
                    }
                    int pass_peer_3 = __shfl_up_sync(0xFFFFFFFF, pass_inclusive, 16, 32);
                    if (lane >= 16) {
                        pass_inclusive += pass_peer_3;
                    }
                    int lane_rank = own_base + emitted + pass_inclusive - pass_count;
                    #pragma unroll 1
                    for (int emit_bit = 0; emit_bit < pass_count; emit_bit++) {
                        int _find_first_set_7 = __ffs(pass_bits);
                        int emit_offset = _find_first_set_7 - 1;
                        int emit_token = (pass_word << 5) + emit_offset;
                        unsigned int route_word = (((emit_offset & 16) != 0) ? (((emit_offset & 8) != 0) ? pass_routes[3] : pass_routes[2]) : (((emit_offset & 8) != 0) ? pass_routes[1] : pass_routes[0]));
                        unsigned int emit_route = route_word >> (unsigned int)((emit_token & 7) * 4) & (unsigned int)15;
                        sorted_token_ids[lane_rank + emit_bit] = emit_token * TOP_K + (int)emit_route;
                        pass_bits = pass_bits & pass_bits - (unsigned int)1;
                    }
                    int _shfl_sync_8 = __shfl_sync(0xFFFFFFFF, pass_inclusive, 31);
                    emitted += _shfl_sync_8;
                }
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
