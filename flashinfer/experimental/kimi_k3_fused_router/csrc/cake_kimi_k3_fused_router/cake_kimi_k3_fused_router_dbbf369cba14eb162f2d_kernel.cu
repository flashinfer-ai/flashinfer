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
#define SMEM_BOUND_VALUES_OFF 40960
#define SMEM_BOUND_VALUES_STAGE_BYTES 128
#define SMEM_BOUND_VALUES_STRIDE 128
#define SMEM_CAND_COUNTS_OFF 41088
#define SMEM_CAND_COUNTS_STAGE_BYTES 32
#define SMEM_CAND_COUNTS_STRIDE 32
#define SMEM_CAND_KEYS_OFF 41120
#define SMEM_CAND_KEYS_STAGE_BYTES 896
#define SMEM_CAND_KEYS_STRIDE 896
#define SMEM_CAND_IDS_OFF 42016
#define SMEM_CAND_IDS_STAGE_BYTES 896
#define SMEM_CAND_IDS_STRIDE 896
#define SMEM_CAND_SCORES_OFF 42912
#define SMEM_CAND_SCORES_STAGE_BYTES 896
#define SMEM_CAND_SCORES_STRIDE 896
#define SMEM_RANK_KEYS_OFF 43808
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
#define SMEM_PLAN_IDS_OFF 8192
#define SMEM_PLAN_IDS_STAGE_BYTES 32768
#define SMEM_PLAN_IDS_STRIDE 32768
#define SMEM_PLAN_BITMAP_OFF 16384
#define SMEM_PLAN_BITMAP_STAGE_BYTES 14336
#define SMEM_PLAN_BITMAP_STRIDE 14336
#define SMEM_TOTAL 44032
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
#define GATHER_LOADS 10
#define GATHER_PASS_PAIRS 2240
#define EMIT_SUBCHUNKS 4
#define EMIT_CHUNK_PAIRS 128
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
kernel_cake_kimi_k3_fused_router_dbbf369cba14eb162f2d(float* __restrict__ logits, float* __restrict__ bias, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M)
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
    unsigned int* bound_values = reinterpret_cast<unsigned int*>(smem_raw + 40960);
    const int bound_values_addr = smem + 40960;
    int* cand_counts = reinterpret_cast<int*>(smem_raw + 41088);
    const int cand_counts_addr = smem + 41088;
    unsigned int* cand_keys = reinterpret_cast<unsigned int*>(smem_raw + 41120);
    const int cand_keys_addr = smem + 41120;
    int* cand_ids = reinterpret_cast<int*>(smem_raw + 42016);
    const int cand_ids_addr = smem + 42016;
    float* cand_scores = reinterpret_cast<float*>(smem_raw + 42912);
    const int cand_scores_addr = smem + 42912;
    unsigned int* rank_keys = reinterpret_cast<unsigned int*>(smem_raw + 43808);
    const int rank_keys_addr = smem + 43808;
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
    int* plan_ids = reinterpret_cast<int*>(smem_raw + 8192);
    const int plan_ids_addr = smem + 8192;
    unsigned int* plan_bitmap = reinterpret_cast<unsigned int*>(smem_raw + 16384);
    const int plan_bitmap_addr = smem + 16384;

    // === Task calls (declaration order) ===
    int global_thread = bid * THREADS + tid;
    int grid_threads = num_bids * THREADS;
    int expert_base = tid * ITEMS_PER_THREAD;
    float _vec_load_0[4];
    {
        float4 _v4 = *reinterpret_cast<const float4*>(bias + expert_base);
        _vec_load_0[0 + 0] = _v4.x;
        _vec_load_0[0 + 1] = _v4.y;
        _vec_load_0[0 + 2] = _v4.z;
        _vec_load_0[0 + 3] = _v4.w;
    }
    float row_logits[4];
    if (bid < M) {
        {
            float4 _v4 = *reinterpret_cast<const float4*>(logits + (unsigned long long)bid * (unsigned long long)NUM_EXPERTS + (unsigned long long)expert_base);
            row_logits[0 + 0] = _v4.x;
            row_logits[0 + 1] = _v4.y;
            row_logits[0 + 2] = _v4.z;
            row_logits[0 + 3] = _v4.w;
        }
    } else {
        #pragma unroll
        for (int first_zero = 0; first_zero < ITEMS_PER_THREAD; first_zero++) {
            row_logits[first_zero] = 0.0f;
        }
    }
    if (warp == 0) {
        if (elect_sync()) {
            #pragma unroll
            for (int winner_init = 0; winner_init < TOP_K; winner_init++) {
                winner_ids[winner_init] = 0;
                winner_scores[winner_init] = 0.0f;
            }
        }
    }
    int bitmap_words = M + 31 >> 5;
    unsigned int bitmap_zero = (unsigned int)0;
    #pragma unroll
    for (int bitmap_clear_pass = 0; bitmap_clear_pass < 4; bitmap_clear_pass++) {
        unsigned int bitmap_clear_addr = plan_bitmap_addr + (unsigned int)((bitmap_clear_pass * THREADS + tid) * 16);
        if (bitmap_words > bitmap_clear_pass) {
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(bitmap_clear_addr), "r"(bitmap_zero), "r"(bitmap_zero), "r"(bitmap_zero), "r"(bitmap_zero) : "memory");
        }
    }
    if (tid < 4) {
        bound_values[28 + tid] = (unsigned int)0;
    }
    unsigned int one_u32 = (unsigned int)1;
    unsigned int lanemask_lt = (one_u32 << (unsigned int)lane) - 1;
    #pragma unroll 1
    for (int token = bid; token < M; token += num_bids) {
        float unbiased[4];
        unsigned int keys[4];
        int active[4];
        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; item++) {
            float _exp_2 = __expf(-row_logits[item]);
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
        int next_token = token + num_bids;
        if (next_token < M) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(logits + (unsigned long long)next_token * (unsigned long long)NUM_EXPERTS + (unsigned long long)expert_base);
                row_logits[0 + 0] = _v4.x;
                row_logits[0 + 1] = _v4.y;
                row_logits[0 + 2] = _v4.z;
                row_logits[0 + 3] = _v4.w;
            }
        } else {
            #pragma unroll
            for (int next_zero = 0; next_zero < ITEMS_PER_THREAD; next_zero++) {
                row_logits[next_zero] = 0.0f;
            }
        }
    }
    cooperative_groups::this_grid().sync();
    int owner_ctas = num_bids;
    if (owner_ctas > OWNER_CTAS) {
        owner_ctas = OWNER_CTAS;
    }
    if (owner_ctas > bid) {
        int total_pairs = M * TOP_K;
        #pragma unroll 1
        for (int gather_base = 0; gather_base < total_pairs; gather_base += GATHER_PASS_PAIRS) {
            int gathered[16];
            #pragma unroll
            for (int gather_slot = 0; gather_slot < GATHER_LOADS; gather_slot++) {
                gathered[gather_slot] = -1;
                int gather_pair = gather_base + gather_slot * THREADS + tid;
                if (gather_pair < total_pairs) {
                    gathered[gather_slot] = topk_ids[gather_pair];
                }
            }
            #pragma unroll
            for (int store_slot = 0; store_slot < 5; store_slot++) {
                int store_id = gathered[store_slot];
                int store_pair = gather_base + store_slot * THREADS + tid;
                int store_token = store_pair >> 4;
                unsigned int store_bit = (unsigned int)0;
                int store_index = lane;
                if (store_id >= 0) {
                    plan_ids[store_pair] = store_id;
                    store_bit = (unsigned int)1 << (unsigned int)(store_token & 31);
                    store_index = (store_token >> 5) * NUM_EXPERTS + store_id;
                }
                atomicAdd(&plan_bitmap[store_index], store_bit);
            }
            if (gather_base + 5 * THREADS < total_pairs) {
                #pragma unroll
                for (int late_slot = 5; late_slot < GATHER_LOADS; late_slot++) {
                    int late_id = gathered[late_slot];
                    int late_pair = gather_base + late_slot * THREADS + tid;
                    int late_token = late_pair >> 4;
                    unsigned int late_bit = (unsigned int)0;
                    int late_index = lane;
                    if (late_id >= 0) {
                        plan_ids[late_pair] = late_id;
                        late_bit = (unsigned int)1 << (unsigned int)(late_token & 31);
                        late_index = (late_token >> 5) * NUM_EXPERTS + late_id;
                    }
                    atomicAdd(&plan_bitmap[late_index], late_bit);
                }
            }
        }
        __syncthreads();
        int expert_scan_base = warp * 32 * ITEMS_PER_THREAD + lane * ITEMS_PER_THREAD;
        int lane_prefixes[4];
        int lane_total = 0;
        int word_counts[4];
        #pragma unroll
        for (int word_count_init = 0; word_count_init < ITEMS_PER_THREAD; word_count_init++) {
            word_counts[word_count_init] = 0;
        }
        #pragma unroll
        for (int word_scan = 0; word_scan < 4; word_scan++) {
            if (bitmap_words > word_scan) {
                #pragma unroll
                for (int word_item = 0; word_item < ITEMS_PER_THREAD; word_item++) {
                    unsigned int word_bits = plan_bitmap[word_scan * NUM_EXPERTS + expert_scan_base + word_item];
                    word_counts[word_item] = word_counts[word_item] + (int)(unsigned int)__popc(word_bits);
                }
            }
        }
        #pragma unroll
        for (int expert_slot_scan = 0; expert_slot_scan < ITEMS_PER_THREAD; expert_slot_scan++) {
            int expert_scan = expert_scan_base + expert_slot_scan;
            int count_scan = 0;
            count_scan = word_counts[expert_slot_scan];
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
        int warp_lane_prefix = warp_inclusive - lane_total;
        #pragma unroll
        for (int expert_slot_store = 0; expert_slot_store < ITEMS_PER_THREAD; expert_slot_store++) {
            int expert_store = expert_scan_base + expert_slot_store;
            plan_offsets[expert_store] = warp_lane_prefix + lane_prefixes[expert_slot_store];
        }
        __syncthreads();
        int phase2_warp_prefix = 0;
        #pragma unroll
        for (int prior_scan_warp = 0; prior_scan_warp < NUM_WARPS; prior_scan_warp++) {
            if (prior_scan_warp < warp) {
                phase2_warp_prefix += phase2_warp_sums[prior_scan_warp];
            }
        }
        if (bid == 0) {
            if (tid == THREADS - 1) {
                int owner_total = phase2_warp_prefix + warp_inclusive;
                expert_offsets[NUM_EXPERTS] = owner_total;
                num_tokens_post_padded[0] = owner_total;
            }
        }
        #pragma unroll 1
        for (int emit_token = bid; emit_token < M; emit_token += num_bids) {
            if (tid < TOP_K) {
                int emit_pair = emit_token * TOP_K + tid;
                int emit_expert = plan_ids[emit_pair];
                int emit_base = plan_offsets[emit_expert];
                int emit_scan_warp = emit_expert >> 7;
                #pragma unroll
                for (int emit_prior_warp = 0; emit_prior_warp < NUM_WARPS; emit_prior_warp++) {
                    if (emit_scan_warp > emit_prior_warp) {
                        emit_base += phase2_warp_sums[emit_prior_warp];
                    }
                }
                int emit_word = emit_token >> 5;
                unsigned int emit_below = ((unsigned int)1 << (unsigned int)(emit_token & 31)) - 1;
                int emit_rank = 0;
                #pragma unroll
                for (int rank_word = 0; rank_word < 4; rank_word++) {
                    if (emit_word >= rank_word) {
                        unsigned int rank_bits = plan_bitmap[rank_word * NUM_EXPERTS + emit_expert];
                        if (emit_word > rank_word) {
                            emit_rank += (int)(unsigned int)__popc(rank_bits);
                        }
                        if (emit_word == rank_word) {
                            emit_rank += (int)(unsigned int)__popc(rank_bits & emit_below);
                        }
                    }
                }
                sorted_token_ids[emit_base + emit_rank] = emit_pair;
            }
        }
        int pair_sentinel = total_pairs;
        #pragma unroll 1
        for (int own_expert = bid * NUM_WARPS + warp; own_expert < NUM_EXPERTS; own_expert += owner_ctas * NUM_WARPS) {
            int own_count = 0;
            #pragma unroll
            for (int own_word = 0; own_word < 4; own_word++) {
                if (bitmap_words > own_word) {
                    unsigned int own_bits = plan_bitmap[own_word * NUM_EXPERTS + own_expert];
                    own_count += (int)(unsigned int)__popc(own_bits);
                }
            }
            int own_base = plan_offsets[own_expert];
            int own_scan_warp = own_expert >> 7;
            #pragma unroll
            for (int own_prior_warp = 0; own_prior_warp < NUM_WARPS; own_prior_warp++) {
                if (own_scan_warp > own_prior_warp) {
                    own_base += phase2_warp_sums[own_prior_warp];
                }
            }
            int own_padded = own_count + BLOCK_MASK & ~BLOCK_MASK;
            if (lane == 0) {
                expert_counts[own_expert] = own_count;
                expert_scatter_offsets[own_expert] = own_count;
                expert_offsets[own_expert] = own_base;
            }
            #pragma unroll 1
            for (int pad_slot = own_count + lane; pad_slot < own_padded; pad_slot += 32) {
                sorted_token_ids[own_base + pad_slot] = pair_sentinel;
            }
            #pragma unroll 1
            for (int block_slot = lane; block_slot < own_padded >> BLOCK_SHIFT; block_slot += 32) {
                expert_ids[(own_base >> BLOCK_SHIFT) + block_slot] = own_expert;
            }
        }
    }
}

} // extern "C"
