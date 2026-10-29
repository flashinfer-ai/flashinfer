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
#define SMEM_FLOOR_SLOTS_OFF 54784
#define SMEM_FLOOR_SLOTS_STAGE_BYTES 32
#define SMEM_FLOOR_SLOTS_STRIDE 32
#define SMEM_BOUND_VALUES_OFF 22528
#define SMEM_BOUND_VALUES_STAGE_BYTES 128
#define SMEM_BOUND_VALUES_STRIDE 128
#define SMEM_CAND_COUNTS_OFF 22656
#define SMEM_CAND_COUNTS_STAGE_BYTES 32
#define SMEM_CAND_COUNTS_STRIDE 32
#define SMEM_CAND_KEYS_OFF 22688
#define SMEM_CAND_KEYS_STAGE_BYTES 896
#define SMEM_CAND_KEYS_STRIDE 896
#define SMEM_CAND_IDS_OFF 23584
#define SMEM_CAND_IDS_STAGE_BYTES 896
#define SMEM_CAND_IDS_STRIDE 896
#define SMEM_CAND_SCORES_OFF 24480
#define SMEM_CAND_SCORES_STAGE_BYTES 896
#define SMEM_CAND_SCORES_STRIDE 896
#define SMEM_RANK_KEYS_OFF 25376
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
#define SMEM_PLAN_TOTALS_OFF 51200
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
#define SMEM_SMEM_V18_OFF 8192
#define SMEM_SMEM_V18_STAGE_BYTES 14336
#define SMEM_SMEM_V18_STRIDE 14336
#define SMEM_STREAM_RING_OFF 22528
#define SMEM_STREAM_RING_STAGE_BYTES 28672
#define SMEM_STREAM_RING_STRIDE 28672
#define SMEM_RANK_BASE_OFF 16512
#define SMEM_RANK_BASE_STAGE_BYTES 112
#define SMEM_RANK_BASE_STRIDE 112
#define SMEM_TOTAL 54912
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

__global__ __launch_bounds__(224, LAUNCH_MIN_BLOCKS) __cluster_dims__(4,1,1) void
kernel_cake_kimi_k3_fused_router_931b47f83cb304759ea0(float* __restrict__ logits, float* __restrict__ bias, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M)
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
    unsigned int* floor_slots = reinterpret_cast<unsigned int*>(smem_raw + 54784);
    const int floor_slots_addr = smem + 54784;
    unsigned int* bound_values = reinterpret_cast<unsigned int*>(smem_raw + 22528);
    const int bound_values_addr = smem + 22528;
    int* cand_counts = reinterpret_cast<int*>(smem_raw + 22656);
    const int cand_counts_addr = smem + 22656;
    unsigned int* cand_keys = reinterpret_cast<unsigned int*>(smem_raw + 22688);
    const int cand_keys_addr = smem + 22688;
    int* cand_ids = reinterpret_cast<int*>(smem_raw + 23584);
    const int cand_ids_addr = smem + 23584;
    float* cand_scores = reinterpret_cast<float*>(smem_raw + 24480);
    const int cand_scores_addr = smem + 24480;
    unsigned int* rank_keys = reinterpret_cast<unsigned int*>(smem_raw + 25376);
    const int rank_keys_addr = smem + 25376;
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
    int* plan_totals = reinterpret_cast<int*>(smem_raw + 51200);
    const int plan_totals_addr = smem + 51200;
    int* plan_offsets = reinterpret_cast<int*>(smem_raw + 3584);
    const int plan_offsets_addr = smem + 3584;
    unsigned int* own_bitmap = reinterpret_cast<unsigned int*>(smem_raw + 8192);
    const int own_bitmap_addr = smem + 8192;
    unsigned int* own_routes = reinterpret_cast<unsigned int*>(smem_raw + 9984);
    const int own_routes_addr = smem + 9984;
    unsigned int* smem_v18 = reinterpret_cast<unsigned int*>(smem_raw + 8192);
    const int smem_v18_addr = smem + 8192;
    int* stream_ring = reinterpret_cast<int*>(smem_raw + 22528);
    const int stream_ring_addr = smem + 22528;
    int* rank_base = reinterpret_cast<int*>(smem_raw + 16512);
    const int rank_base_addr = smem + 16512;

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
        unsigned int own_top[4];
        unsigned int warp_top0_1_0;
        asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(warp_top0_1_0) : "r"(lane_bound));
        own_top[0] = warp_top0_1_0;
        if (lane == 0) {
            published = warp_top0_1_0;
        }
        if (lane_bound == warp_top0_1_0) {
            lane_bound = (unsigned int)0;
        }
        unsigned int warp_top0_1_1;
        asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(warp_top0_1_1) : "r"(lane_bound));
        own_top[1] = warp_top0_1_1;
        if (lane == 1) {
            published = warp_top0_1_1;
        }
        if (lane_bound == warp_top0_1_1) {
            lane_bound = (unsigned int)0;
        }
        unsigned int warp_top0_1_2;
        asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(warp_top0_1_2) : "r"(lane_bound));
        own_top[2] = warp_top0_1_2;
        if (lane == 2) {
            published = warp_top0_1_2;
        }
        if (lane_bound == warp_top0_1_2) {
            lane_bound = (unsigned int)0;
        }
        unsigned int warp_top0_1_3;
        asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(warp_top0_1_3) : "r"(lane_bound));
        own_top[3] = warp_top0_1_3;
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
        unsigned int own_b0 = __ballot_sync(0xFFFFFFFF, bound_mine >= own_top[0]);
        unsigned int own_c0 = (unsigned int)0;
        if ((int)(unsigned int)__popc(own_b0) >= 16) {
            own_c0 = own_top[0];
        }
        unsigned int own_b1 = __ballot_sync(0xFFFFFFFF, bound_mine >= own_top[1]);
        unsigned int own_c1 = (unsigned int)0;
        if ((int)(unsigned int)__popc(own_b1) >= 16) {
            own_c1 = own_top[1];
        }
        unsigned int own_b2 = __ballot_sync(0xFFFFFFFF, bound_mine >= own_top[2]);
        unsigned int own_c2 = (unsigned int)0;
        if ((int)(unsigned int)__popc(own_b2) >= 16) {
            own_c2 = own_top[2];
        }
        unsigned int own_b3 = __ballot_sync(0xFFFFFFFF, bound_mine >= own_top[3]);
        unsigned int own_c3 = (unsigned int)0;
        if ((int)(unsigned int)__popc(own_b3) >= 16) {
            own_c3 = own_top[3];
        }
        if (own_c1 > own_c0) {
            own_c0 = own_c1;
        }
        if (own_c3 > own_c2) {
            own_c2 = own_c3;
        }
        if (own_c2 > own_c0) {
            own_c0 = own_c2;
        }
        if (lane == 0) {
            floor_slots[warp] = own_c0;
        }
        __syncthreads();
        unsigned int floor_seen = (unsigned int)0;
        if (lane < 7) {
            floor_seen = floor_slots[lane];
        }
        unsigned int cand_floor;
        asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(cand_floor) : "r"(floor_seen));
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
    int own_expert_base = bid * NUM_WARPS;
    if (bid < OWNER_CTAS) {
        #pragma unroll 1
        for (int count_clear = tid; count_clear < NUM_EXPERTS; count_clear += THREADS) {
            plan_counts[count_clear] = 0;
        }
        #pragma unroll 1
        for (int map_clear = tid; map_clear < CLUSTER_EXPERTS * 288 / 16; map_clear += THREADS) {
            unsigned int map_zero = 0;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v18_addr + (unsigned int)(map_clear * 16)), "r"(map_zero), "r"(map_zero), "r"(map_zero), "r"(map_zero) : "memory");
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
        unsigned int byte_dump = smem_v18_addr + (unsigned int)(CLUSTER_EXPERTS * 288 + tid);
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
            #pragma unroll
            for (int own_copy_1 = 0; own_copy_1 < COPIES_PER_THREAD; own_copy_1++) {
                int copy_pair = gather_base + (own_copy_1 * THREADS + tid) * 4;
                if (copy_pair < slice_end) {
                    #pragma unroll
                    for (int copy_lane = 0; copy_lane < 4; copy_lane++) {
                        int copy_id = (int)own_ids[own_copy_1 * 4 + copy_lane];
                        atomicAdd(&plan_counts[copy_id], 1);
                    }
                }
            }
            #pragma unroll
            for (int byte_copy = 0; byte_copy < COPIES_PER_THREAD; byte_copy++) {
                int byte_pair = gather_base + (byte_copy * THREADS + tid) * 4;
                int byte_lt = (byte_pair >> 4) - first_token;
                #pragma unroll
                for (int byte_lane = 0; byte_lane < 4; byte_lane++) {
                    int byte_slot = (int)own_ids[byte_copy * 4 + byte_lane] - cluster_expert_base;
                    unsigned int byte_map = smem_v18_addr + (unsigned int)(byte_slot * 288 + (byte_lt >> 5) * 32 + (byte_lt & 7) * 4 + (byte_lt >> 3 & 3));
                    unsigned int byte_addr = ((byte_pair < slice_end && (unsigned int)byte_slot < 28 && (unsigned int)byte_lt < 288) ? byte_map : byte_dump);
                    {
                        uint32_t _byte_5 = static_cast<uint32_t>((unsigned int)((byte_pair & 15) + byte_lane | 16)) & 0xFFu;
                        uint32_t _addr_5 = static_cast<uint32_t>(byte_addr);
                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_5), "r"(_byte_5));
                    }
                }
            }
            stream_stage = 1 - stream_stage;
        }
        asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
        __syncthreads();
        asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
        int lb_slot = (lane >> 3) * NUM_WARPS + warp;
        int lb_q = lane & 7;
        int lb_lt = lb_q * 32;
        if (M <= 512) {
            lb_lt = lb_q * 16;
        }
        unsigned int lb_words[8];
        #pragma unroll
        for (int lb_half = 0; lb_half < 2; lb_half++) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lb_words[lb_half * 4])), "=r"(*reinterpret_cast<uint32_t*>(&lb_words[(lb_half * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lb_words[(lb_half * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lb_words[(lb_half * 4) + 3]))
                : "r"(smem_v18_addr + (unsigned int)(lb_slot * 288 + (lb_lt >> 5) * 32 + lb_half * 16)));
        }
        unsigned int lb_marks = 0;
        #pragma unroll
        for (int lb_mark_word = 0; lb_mark_word < 8; lb_mark_word++) {
            lb_marks |= (lb_words[lb_mark_word] >> 4 & (unsigned int)16843009) << (unsigned int)lb_mark_word;
        }
        if (M <= 512) {
            lb_marks = lb_marks >> (unsigned int)(lb_lt & 16) & (unsigned int)65535;
            #pragma unroll
            for (int lb_shift_word = 0; lb_shift_word < 8; lb_shift_word++) {
                lb_words[lb_shift_word] = lb_words[lb_shift_word] >> (unsigned int)(lb_lt & 16);
            }
        }
        int lb_count = (int)(unsigned int)__popc(lb_marks);
        int lb_inclusive = lb_count;
        int lb_peer_0 = __shfl_up_sync(0xFFFFFFFF, lb_inclusive, 1, 32);
        if (lb_q >= 1) {
            lb_inclusive += lb_peer_0;
        }
        int lb_peer_1 = __shfl_up_sync(0xFFFFFFFF, lb_inclusive, 2, 32);
        if (lb_q >= 2) {
            lb_inclusive += lb_peer_1;
        }
        int lb_peer_2 = __shfl_up_sync(0xFFFFFFFF, lb_inclusive, 4, 32);
        if (lb_q >= 4) {
            lb_inclusive += lb_peer_2;
        }
        int lb_total = __shfl_sync(0xFFFFFFFF, lb_inclusive, lane | 7);
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        unsigned int peer_counts_base[4];
        #pragma unroll
        for (int count_peer_map = 0; count_peer_map < CLUSTER; count_peer_map++) {
            uint32_t _cluster_mapa_6;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_6) : "r"(plan_counts_addr), "r"(count_peer_map));
            peer_counts_base[count_peer_map] = _cluster_mapa_6;
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
        int lb_window_tokens = 256;
        if (M <= 512) {
            lb_window_tokens = 128;
        }
        int lb_windows = ((slice_end - 1 >> 4) - first_token + lb_window_tokens) / lb_window_tokens;
        int lb_base = plan_offsets[cluster_expert_base + lb_slot] + rank_base[lb_slot];
        #pragma unroll 1
        for (int lb_window = 0; lb_window < lb_windows; lb_window++) {
            if (lb_window > 0) {
                lb_lt = lb_lt + lb_window_tokens;
                #pragma unroll
                for (int lb_zero = 0; lb_zero < 8; lb_zero++) {
                    lb_words[lb_zero] = (unsigned int)0;
                }
                if (lb_lt < 288) {
                    #pragma unroll
                    for (int lb_half_w = 0; lb_half_w < 2; lb_half_w++) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lb_words[lb_half_w * 4])), "=r"(*reinterpret_cast<uint32_t*>(&lb_words[(lb_half_w * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lb_words[(lb_half_w * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lb_words[(lb_half_w * 4) + 3]))
                            : "r"(smem_v18_addr + (unsigned int)(lb_slot * 288 + (lb_lt >> 5) * 32 + lb_half_w * 16)));
                    }
                }
                lb_marks = (unsigned int)0;
                #pragma unroll
                for (int lb_mark_word_w = 0; lb_mark_word_w < 8; lb_mark_word_w++) {
                    lb_marks |= (lb_words[lb_mark_word_w] >> 4 & (unsigned int)16843009) << (unsigned int)lb_mark_word_w;
                }
                if (M <= 512) {
                    lb_marks = lb_marks >> (unsigned int)(lb_lt & 16) & (unsigned int)65535;
                    #pragma unroll
                    for (int lb_shift_word_w = 0; lb_shift_word_w < 8; lb_shift_word_w++) {
                        lb_words[lb_shift_word_w] = lb_words[lb_shift_word_w] >> (unsigned int)(lb_lt & 16);
                    }
                }
                lb_count = (int)(unsigned int)__popc(lb_marks);
                lb_inclusive = lb_count;
                int lb_peer_w_0 = __shfl_up_sync(0xFFFFFFFF, lb_inclusive, 1, 32);
                if (lb_q >= 1) {
                    lb_inclusive += lb_peer_w_0;
                }
                int lb_peer_w_1 = __shfl_up_sync(0xFFFFFFFF, lb_inclusive, 2, 32);
                if (lb_q >= 2) {
                    lb_inclusive += lb_peer_w_1;
                }
                int lb_peer_w_2 = __shfl_up_sync(0xFFFFFFFF, lb_inclusive, 4, 32);
                if (lb_q >= 4) {
                    lb_inclusive += lb_peer_w_2;
                }
                int _shfl_sync_7 = __shfl_sync(0xFFFFFFFF, lb_inclusive, lane | 7);
                lb_total = _shfl_sync_7;
            }
            int lb_rank = lb_base + lb_inclusive - lb_count;
            int lb_token = first_token + lb_lt;
            #pragma unroll 1
            for (int lb_bit = 0; lb_bit < lb_count; lb_bit++) {
                int _find_first_set_8 = __ffs(lb_marks);
                int lb_offset = _find_first_set_8 - 1;
                unsigned int lb_word = (((lb_offset & 4) != 0) ? (((lb_offset & 2) != 0) ? (((lb_offset & 1) != 0) ? lb_words[7] : lb_words[6]) : (((lb_offset & 1) != 0) ? lb_words[5] : lb_words[4])) : (((lb_offset & 2) != 0) ? (((lb_offset & 1) != 0) ? lb_words[3] : lb_words[2]) : (((lb_offset & 1) != 0) ? lb_words[1] : lb_words[0])));
                unsigned int lb_route = lb_word >> (unsigned int)((lb_offset >> 3) * 8) & (unsigned int)15;
                sorted_token_ids[lb_rank + lb_bit] = (lb_token + lb_offset) * TOP_K + (int)lb_route;
                lb_marks = lb_marks & lb_marks - (unsigned int)1;
            }
            lb_base += lb_total;
        }
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
