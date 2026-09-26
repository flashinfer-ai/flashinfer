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
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
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
#define SMEM_WARP_HIST_STAGE_BYTES 8064
#define SMEM_WARP_HIST_STRIDE 8064
#define SMEM_WINNER_IDS_OFF 12288
#define SMEM_WINNER_IDS_STAGE_BYTES 448
#define SMEM_WINNER_IDS_STRIDE 448
#define SMEM_WINNER_SCORES_OFF 12736
#define SMEM_WINNER_SCORES_STAGE_BYTES 448
#define SMEM_WINNER_SCORES_STRIDE 448
#define SMEM_BIAS_SMEM_OFF 13312
#define SMEM_BIAS_SMEM_STAGE_BYTES 3584
#define SMEM_BIAS_SMEM_STRIDE 3584
#define SMEM_PHASE2_WARP_SUMS_OFF 0
#define SMEM_PHASE2_WARP_SUMS_STAGE_BYTES 28
#define SMEM_PHASE2_WARP_SUMS_STRIDE 28
#define SMEM_LOCAL_OFFSETS_OFF 8192
#define SMEM_LOCAL_OFFSETS_STAGE_BYTES 3584
#define SMEM_LOCAL_OFFSETS_STRIDE 3584
#define SMEM_BITMAP_A_OFF 0
#define SMEM_BITMAP_A_STAGE_BYTES 1024
#define SMEM_BITMAP_A_STRIDE 1024
#define SMEM_ROUTES_A_OFF 1024
#define SMEM_ROUTES_A_STAGE_BYTES 4096
#define SMEM_ROUTES_A_STRIDE 4096
#define SMEM_WORD_PREFIX_A_OFF 5120
#define SMEM_WORD_PREFIX_A_STAGE_BYTES 1024
#define SMEM_WORD_PREFIX_A_STRIDE 1024
#define SMEM_CHUNK_PREFIX_A_OFF 6144
#define SMEM_CHUNK_PREFIX_A_STAGE_BYTES 32
#define SMEM_CHUNK_PREFIX_A_STRIDE 32
#define SMEM_BITMAP_B_OFF 6176
#define SMEM_BITMAP_B_STAGE_BYTES 1024
#define SMEM_BITMAP_B_STRIDE 1024
#define SMEM_ROUTES_B_OFF 7200
#define SMEM_ROUTES_B_STAGE_BYTES 4096
#define SMEM_ROUTES_B_STRIDE 4096
#define SMEM_WORD_PREFIX_B_OFF 11296
#define SMEM_WORD_PREFIX_B_STAGE_BYTES 1024
#define SMEM_WORD_PREFIX_B_STRIDE 1024
#define SMEM_CHUNK_PREFIX_B_OFF 12320
#define SMEM_CHUNK_PREFIX_B_STAGE_BYTES 32
#define SMEM_CHUNK_PREFIX_B_STRIDE 32
#define SMEM_TOTAL 32768
#define THREADS 224
#define BLOCK_M 16
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

#include <math_constants.h>
#include <cooperative_groups.h>


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

__global__ __launch_bounds__(224, 4) void
kernel_cake_kimi_k3_fused_router_eba0da520bfae74745f3(float* __restrict__ logits, float* __restrict__ bias, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    int* warp_hist = reinterpret_cast<int*>(smem_raw + 0);
    const int warp_hist_addr = smem + 0;
    int* winner_ids = reinterpret_cast<int*>(smem_raw + 12288);
    const int winner_ids_addr = smem + 12288;
    float* winner_scores = reinterpret_cast<float*>(smem_raw + 12736);
    const int winner_scores_addr = smem + 12736;
    float* bias_smem = reinterpret_cast<float*>(smem_raw + 13312);
    const int bias_smem_addr = smem + 13312;
    int* phase2_warp_sums = reinterpret_cast<int*>(smem_raw + 0);
    const int phase2_warp_sums_addr = smem + 0;
    int* local_offsets = reinterpret_cast<int*>(smem_raw + 8192);
    const int local_offsets_addr = smem + 8192;
    unsigned int* bitmap_a = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int bitmap_a_addr = smem + 0;
    unsigned int* routes_a = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int routes_a_addr = smem + 1024;
    int* word_prefix_a = reinterpret_cast<int*>(smem_raw + 5120);
    const int word_prefix_a_addr = smem + 5120;
    int* chunk_prefix_a = reinterpret_cast<int*>(smem_raw + 6144);
    const int chunk_prefix_a_addr = smem + 6144;
    unsigned int* bitmap_b = reinterpret_cast<unsigned int*>(smem_raw + 6176);
    const int bitmap_b_addr = smem + 6176;
    unsigned int* routes_b = reinterpret_cast<unsigned int*>(smem_raw + 7200);
    const int routes_b_addr = smem + 7200;
    int* word_prefix_b = reinterpret_cast<int*>(smem_raw + 11296);
    const int word_prefix_b_addr = smem + 11296;
    int* chunk_prefix_b = reinterpret_cast<int*>(smem_raw + 12320);
    const int chunk_prefix_b_addr = smem + 12320;

    // === Task calls (dependency order) ===
    int global_thread = bid * THREADS + tid;
    int grid_threads = num_bids * THREADS;
    #pragma unroll 1
    for (int expert_zero = global_thread; expert_zero < NUM_EXPERTS; expert_zero += grid_threads) {
        expert_counts[expert_zero] = 0;
    }
    __threadfence();
    cooperative_groups::this_grid().sync();
    int expert_base = tid * ITEMS_PER_THREAD;
    float _vec_load_0[4];
    {
        float4 _v4 = *reinterpret_cast<const float4*>(bias + expert_base);
        _vec_load_0[0 + 0] = _v4.x;
        _vec_load_0[0 + 1] = _v4.y;
        _vec_load_0[0 + 2] = _v4.z;
        _vec_load_0[0 + 3] = _v4.w;
    }
    #pragma unroll
    for (int bias_slot = 0; bias_slot < 4; bias_slot++) {
        bias_smem[expert_base + bias_slot] = _vec_load_0[bias_slot];
    }
    int stash_base = warp * TOP_K;
    if (lane < TOP_K) {
        winner_ids[stash_base + lane] = 0;
        winner_scores[stash_base + lane] = 0.0f;
    }
    __syncthreads();
    int hist_base = warp * 288;
    int lane_hist_base = hist_base + lane * 9;
    int scan_bin_base = lane * 8;
    unsigned int one_u32 = 1;
    unsigned int lanemask_lt = (one_u32 << (unsigned int)lane) - 1;
    #pragma unroll 1
    for (int token = bid * NUM_WARPS + warp; token < M; token += num_bids * NUM_WARPS) {
        unsigned long long row_base = (unsigned long long)token * (unsigned long long)NUM_EXPERTS;
        unsigned int keys[28];
        #pragma unroll
        for (int item = 0; item < 28; item++) {
            int col = item * 32 + lane;
            float logit = logits[row_base + (unsigned long long)col];
            float _expf_0 = __expf(-logit);
            float _fdiv_full_0;
            asm volatile("div.full.f32 %0, %1, %2;" : "=f"(_fdiv_full_0) : "f"(1.0f), "f"(1.0f + _expf_0));
            float score = _fdiv_full_0;
            float bias_col = bias_smem[col];
            float ranking = score + bias_col;
            if (ranking == 0.0f) {
                ranking = 0.0f;
            }
            unsigned int bits = __as_u32(ranking);
            unsigned int ordered = bits | 2147483648;
            if (bits >= 2147483648) {
                ordered = bits ^ 4294967295;
            }
            keys[item] = ordered;
        }
        int remaining = TOP_K;
        int active_total = NUM_EXPERTS;
        unsigned int threshold = 0;
        unsigned int examined_mask = 0;
        int take_all_equals = 0;
        #pragma unroll
        for (int radix_round = 0; radix_round < 4; radix_round++) {
            int shift = 24 - radix_round * 8;
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
            int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, inclusive, 1, 32);
            int scan_peer = _shfl_up_0;
            if (lane >= 1) {
                inclusive += scan_peer;
            }
            int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, inclusive, 2, 32);
            int scan_peer_0 = _shfl_up_1;
            if (lane >= 2) {
                inclusive += scan_peer_0;
            }
            int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, inclusive, 4, 32);
            int scan_peer_1 = _shfl_up_2;
            if (lane >= 4) {
                inclusive += scan_peer_1;
            }
            int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, inclusive, 8, 32);
            int scan_peer_2 = _shfl_up_3;
            if (lane >= 8) {
                inclusive += scan_peer_2;
            }
            int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, inclusive, 16, 32);
            int scan_peer_3 = _shfl_up_4;
            if (lane >= 16) {
                inclusive += scan_peer_3;
            }
            int bin_inclusive = inclusive - lane_count;
            unsigned int warp_match_record = 0;
            #pragma unroll
            for (int scan_offset_1 = 0; scan_offset_1 < 8; scan_offset_1++) {
                int bin_count = scan_counts[scan_offset_1];
                bin_inclusive += bin_count;
                int bin_above = active_total - bin_inclusive;
                if (bin_above < remaining && remaining <= bin_above + bin_count) {
                    int chosen_scan_bin = scan_bin_base + scan_offset_1;
                    warp_match_record = (unsigned int)(chosen_scan_bin & 255) | (unsigned int)(bin_above & 1023) << 8 | (unsigned int)(bin_count & 1023) << 18;
                }
            }
            unsigned int _warp_redux_u32_0;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(warp_match_record));
            unsigned int match_record = _warp_redux_u32_0;
            int chosen_bin = (int)(match_record & 255);
            remaining -= (int)(match_record >> 8 & 1023);
            active_total = (int)(match_record >> 18 & 1023);
            threshold |= (unsigned int)chosen_bin << (unsigned int)shift;
            examined_mask |= (unsigned int)255 << (unsigned int)shift;
            if (remaining == active_total) {
                take_all_equals = 1;
                break;
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
                unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, tie_active != 0);
                unsigned int tie_ballot = _vote_0;
                int _popc_0 = __popc(tie_ballot & lanemask_lt);
                int tie_rank = tie_prior + _popc_0;
                if (threshold < (keys[item_tie] & examined_mask)) {
                    selected_mask |= one_u32 << (unsigned int)item_tie;
                }
                if (tie_active != 0) {
                    if (tie_rank < remaining) {
                        selected_mask |= one_u32 << (unsigned int)item_tie;
                    }
                }
                int _popc_1 = __popc(tie_ballot);
                tie_prior += _popc_1;
            }
        }
        float logits_r[28];
        #pragma unroll
        for (int item_reload = 0; item_reload < 28; item_reload++) {
            int reload_col = item_reload * 32 + lane;
            logits_r[item_reload] = reinterpret_cast<volatile float*>(logits)[row_base + (unsigned long long)reload_col];
        }
        int compact_prior = 0;
        #pragma unroll
        for (int item_compact = 0; item_compact < 28; item_compact++) {
            int chosen = (int)(selected_mask >> (unsigned int)item_compact & 1);
            unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, chosen != 0);
            unsigned int compact_ballot = _vote_1;
            if (chosen != 0) {
                int _popc_2 = __popc(compact_ballot & lanemask_lt);
                int slot = compact_prior + _popc_2;
                int winner_col = item_compact * 32 + lane;
                float _expf_1 = __expf(-logits_r[item_compact]);
                float _fdiv_full_1;
                asm volatile("div.full.f32 %0, %1, %2;" : "=f"(_fdiv_full_1) : "f"(1.0f), "f"(1.0f + _expf_1));
                float winner_score = _fdiv_full_1;
                winner_ids[stash_base + slot] = winner_col;
                winner_scores[stash_base + slot] = winner_score;
            }
            int _popc_3 = __popc(compact_ballot);
            compact_prior += _popc_3;
        }
        __syncwarp();
        if (lane < TOP_K) {
            float selected_sum = 0.0f;
            #pragma unroll
            for (int route_sum = 0; route_sum < TOP_K; route_sum++) {
                selected_sum += winner_scores[stash_base + route_sum];
            }
            float selected_norm = 1.0f;
            if (selected_sum > 0.0f) {
                selected_norm = selected_sum;
            }
            int selected_id = winner_ids[stash_base + lane];
            unsigned long long output_index = (unsigned long long)token * (unsigned long long)TOP_K + (unsigned long long)lane;
            float _fdiv_full_2;
            asm volatile("div.full.f32 %0, %1, %2;" : "=f"(_fdiv_full_2) : "f"(winner_scores[stash_base + lane]), "f"(selected_norm));
            topk_weights[output_index] = _fdiv_full_2;
            int _atomic_old_0 = atomicAdd(&expert_counts[selected_id], 1);
            int local_rank = _atomic_old_0;
            unsigned int packed_rank = (unsigned int)local_rank;
            unsigned int packed_expert = (unsigned int)selected_id;
            unsigned int packed_topk = packed_rank << 10 | packed_expert;
            topk_ids[output_index] = (int)packed_topk;
        }
        __syncwarp();
    }
    __threadfence();
    cooperative_groups::this_grid().sync();
    int expert_scan_base = warp * 32 * ITEMS_PER_THREAD + lane * ITEMS_PER_THREAD;
    int lane_prefixes[ITEMS_PER_THREAD];
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
    int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, 1, 32);
    int phase2_scan_peer = _shfl_up_5;
    if (lane >= 1) {
        warp_inclusive += phase2_scan_peer;
    }
    int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, 2, 32);
    int phase2_scan_peer_0 = _shfl_up_6;
    if (lane >= 2) {
        warp_inclusive += phase2_scan_peer_0;
    }
    int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, 4, 32);
    int phase2_scan_peer_1 = _shfl_up_7;
    if (lane >= 4) {
        warp_inclusive += phase2_scan_peer_1;
    }
    int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, 8, 32);
    int phase2_scan_peer_2 = _shfl_up_8;
    if (lane >= 8) {
        warp_inclusive += phase2_scan_peer_2;
    }
    int _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, 16, 32);
    int phase2_scan_peer_3 = _shfl_up_9;
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
    int _shfl_0 = __shfl_sync(0xFFFFFFFF, phase2_warp_prefix_lane0, 0);
    int phase2_warp_prefix = _shfl_0;
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
        __threadfence();
        cooperative_groups::this_grid().sync();
        #pragma unroll 1
        for (int warp_sort_expert = bid * 7 + warp; warp_sort_expert < 896; warp_sort_expert += num_bids * 7) {
            int warp_sort_count = expert_counts[warp_sort_expert];
            if (warp_sort_count > 1) {
                if (warp_sort_count <= 32) {
                    int warp_sort_start = expert_offsets[warp_sort_expert];
                    int warp_sort_value = M * 16;
                    if (warp_sort_count > lane) {
                        warp_sort_value = sorted_token_ids[warp_sort_start + lane];
                    }
                    int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 1);
                    int warp_sort_peer = _shfl_xor_0;
                    int warp_sort_take_max = (lane >> 1 ^ lane) & 1;
                    if (warp_sort_take_max == 0) {
                        if (warp_sort_peer < warp_sort_value) {
                            warp_sort_value = warp_sort_peer;
                        }
                    } else if (warp_sort_peer > warp_sort_value) {
                        warp_sort_value = warp_sort_peer;
                    }
                    int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 2);
                    int warp_sort_peer_0 = _shfl_xor_1;
                    int warp_sort_take_max_1 = (lane >> 2 ^ lane >> 1) & 1;
                    if (warp_sort_take_max_1 == 0) {
                        if (warp_sort_peer_0 < warp_sort_value) {
                            warp_sort_value = warp_sort_peer_0;
                        }
                    } else if (warp_sort_peer_0 > warp_sort_value) {
                        warp_sort_value = warp_sort_peer_0;
                    }
                    int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 1);
                    int warp_sort_peer_2 = _shfl_xor_2;
                    int warp_sort_take_max_3 = (lane >> 2 ^ lane) & 1;
                    if (warp_sort_take_max_3 == 0) {
                        if (warp_sort_peer_2 < warp_sort_value) {
                            warp_sort_value = warp_sort_peer_2;
                        }
                    } else if (warp_sort_peer_2 > warp_sort_value) {
                        warp_sort_value = warp_sort_peer_2;
                    }
                    int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 4);
                    int warp_sort_peer_4 = _shfl_xor_3;
                    int warp_sort_take_max_5 = (lane >> 3 ^ lane >> 2) & 1;
                    if (warp_sort_take_max_5 == 0) {
                        if (warp_sort_peer_4 < warp_sort_value) {
                            warp_sort_value = warp_sort_peer_4;
                        }
                    } else if (warp_sort_peer_4 > warp_sort_value) {
                        warp_sort_value = warp_sort_peer_4;
                    }
                    int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 2);
                    int warp_sort_peer_6 = _shfl_xor_4;
                    int warp_sort_take_max_7 = (lane >> 3 ^ lane >> 1) & 1;
                    if (warp_sort_take_max_7 == 0) {
                        if (warp_sort_peer_6 < warp_sort_value) {
                            warp_sort_value = warp_sort_peer_6;
                        }
                    } else if (warp_sort_peer_6 > warp_sort_value) {
                        warp_sort_value = warp_sort_peer_6;
                    }
                    int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 1);
                    int warp_sort_peer_8 = _shfl_xor_5;
                    int warp_sort_take_max_9 = (lane >> 3 ^ lane) & 1;
                    if (warp_sort_take_max_9 == 0) {
                        if (warp_sort_peer_8 < warp_sort_value) {
                            warp_sort_value = warp_sort_peer_8;
                        }
                    } else if (warp_sort_peer_8 > warp_sort_value) {
                        warp_sort_value = warp_sort_peer_8;
                    }
                    int _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 8);
                    int warp_sort_peer_10 = _shfl_xor_6;
                    int warp_sort_take_max_11 = (lane >> 4 ^ lane >> 3) & 1;
                    if (warp_sort_take_max_11 == 0) {
                        if (warp_sort_peer_10 < warp_sort_value) {
                            warp_sort_value = warp_sort_peer_10;
                        }
                    } else if (warp_sort_peer_10 > warp_sort_value) {
                        warp_sort_value = warp_sort_peer_10;
                    }
                    int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 4);
                    int warp_sort_peer_12 = _shfl_xor_7;
                    int warp_sort_take_max_13 = (lane >> 4 ^ lane >> 2) & 1;
                    if (warp_sort_take_max_13 == 0) {
                        if (warp_sort_peer_12 < warp_sort_value) {
                            warp_sort_value = warp_sort_peer_12;
                        }
                    } else if (warp_sort_peer_12 > warp_sort_value) {
                        warp_sort_value = warp_sort_peer_12;
                    }
                    int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 2);
                    int warp_sort_peer_14 = _shfl_xor_8;
                    int warp_sort_take_max_15 = (lane >> 4 ^ lane >> 1) & 1;
                    if (warp_sort_take_max_15 == 0) {
                        if (warp_sort_peer_14 < warp_sort_value) {
                            warp_sort_value = warp_sort_peer_14;
                        }
                    } else if (warp_sort_peer_14 > warp_sort_value) {
                        warp_sort_value = warp_sort_peer_14;
                    }
                    int _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 1);
                    int warp_sort_peer_16 = _shfl_xor_9;
                    int warp_sort_take_max_17 = (lane >> 4 ^ lane) & 1;
                    if (warp_sort_take_max_17 == 0) {
                        if (warp_sort_peer_16 < warp_sort_value) {
                            warp_sort_value = warp_sort_peer_16;
                        }
                    } else if (warp_sort_peer_16 > warp_sort_value) {
                        warp_sort_value = warp_sort_peer_16;
                    }
                    int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 16);
                    int warp_sort_peer_18 = _shfl_xor_10;
                    int warp_sort_take_max_19 = (lane >> 5 ^ lane >> 4) & 1;
                    if (warp_sort_take_max_19 == 0) {
                        if (warp_sort_peer_18 < warp_sort_value) {
                            warp_sort_value = warp_sort_peer_18;
                        }
                    } else if (warp_sort_peer_18 > warp_sort_value) {
                        warp_sort_value = warp_sort_peer_18;
                    }
                    int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 8);
                    int warp_sort_peer_20 = _shfl_xor_11;
                    int warp_sort_take_max_21 = (lane >> 5 ^ lane >> 3) & 1;
                    if (warp_sort_take_max_21 == 0) {
                        if (warp_sort_peer_20 < warp_sort_value) {
                            warp_sort_value = warp_sort_peer_20;
                        }
                    } else if (warp_sort_peer_20 > warp_sort_value) {
                        warp_sort_value = warp_sort_peer_20;
                    }
                    int _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 4);
                    int warp_sort_peer_22 = _shfl_xor_12;
                    int warp_sort_take_max_23 = (lane >> 5 ^ lane >> 2) & 1;
                    if (warp_sort_take_max_23 == 0) {
                        if (warp_sort_peer_22 < warp_sort_value) {
                            warp_sort_value = warp_sort_peer_22;
                        }
                    } else if (warp_sort_peer_22 > warp_sort_value) {
                        warp_sort_value = warp_sort_peer_22;
                    }
                    int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 2);
                    int warp_sort_peer_24 = _shfl_xor_13;
                    int warp_sort_take_max_25 = (lane >> 5 ^ lane >> 1) & 1;
                    if (warp_sort_take_max_25 == 0) {
                        if (warp_sort_peer_24 < warp_sort_value) {
                            warp_sort_value = warp_sort_peer_24;
                        }
                    } else if (warp_sort_peer_24 > warp_sort_value) {
                        warp_sort_value = warp_sort_peer_24;
                    }
                    int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, warp_sort_value, 1);
                    int warp_sort_peer_26 = _shfl_xor_14;
                    int warp_sort_take_max_27 = (lane >> 5 ^ lane) & 1;
                    if (warp_sort_take_max_27 == 0) {
                        if (warp_sort_peer_26 < warp_sort_value) {
                            warp_sort_value = warp_sort_peer_26;
                        }
                    } else if (warp_sort_peer_26 > warp_sort_value) {
                        warp_sort_value = warp_sort_peer_26;
                    }
                    if (warp_sort_count > lane) {
                        sorted_token_ids[warp_sort_start + lane] = warp_sort_value;
                    }
                }
            }
        }
        #pragma unroll 1
        for (int sort_expert_a = bid; sort_expert_a < 896; sort_expert_a += num_bids * 2) {
            int sort_expert_b = sort_expert_a + num_bids;
            int sort_count_a = expert_counts[sort_expert_a];
            int sort_count_b = 0;
            if (sort_expert_b < 896) {
                sort_count_b = expert_counts[sort_expert_b];
            }
            if (sort_count_a <= 32) {
                sort_count_a = 0;
            }
            if (sort_count_b <= 32) {
                sort_count_b = 0;
            }
            if (sort_count_a + sort_count_b > 0) {
                int sort_start_a = 0;
                int sort_start_b = 0;
                if (sort_count_a > 0) {
                    sort_start_a = expert_offsets[sort_expert_a];
                }
                if (sort_count_b > 0) {
                    sort_start_b = expert_offsets[sort_expert_b];
                }
                int bitmap_words = M + 31 >> 5;
                int route_words = M + 7 >> 3;
                #pragma unroll 1
                for (int clear_word = tid; clear_word < bitmap_words; clear_word += 224) {
                    bitmap_a[clear_word] = (unsigned int)0;
                    bitmap_b[clear_word] = (unsigned int)0;
                }
                #pragma unroll 1
                for (int clear_route = tid; clear_route < route_words; clear_route += 224) {
                    routes_a[clear_route] = (unsigned int)0;
                    routes_b[clear_route] = (unsigned int)0;
                }
                __syncthreads();
                int sort_count_max = sort_count_a;
                if (sort_count_b > sort_count_max) {
                    sort_count_max = sort_count_b;
                }
                #pragma unroll 1
                for (int input_slot = tid; input_slot < sort_count_max; input_slot += 224) {
                    int pair_a = 0;
                    int pair_b = 0;
                    if (sort_count_a > input_slot) {
                        pair_a = sorted_token_ids[sort_start_a + input_slot];
                    }
                    if (sort_count_b > input_slot) {
                        pair_b = sorted_token_ids[sort_start_b + input_slot];
                    }
                    if (sort_count_a > input_slot) {
                        int pair_token_a = pair_a >> 4;
                        int pair_route_a = pair_a & 15;
                        atomicAdd(&bitmap_a[pair_token_a >> 5], (unsigned int)1 << (unsigned int)(pair_token_a & 31));
                        atomicAdd(&routes_a[pair_token_a >> 3], (unsigned int)pair_route_a << (unsigned int)((pair_token_a & 7) * 4));
                    }
                    if (sort_count_b > input_slot) {
                        int pair_token_b = pair_b >> 4;
                        int pair_route_b = pair_b & 15;
                        atomicAdd(&bitmap_b[pair_token_b >> 5], (unsigned int)1 << (unsigned int)(pair_token_b & 31));
                        atomicAdd(&routes_b[pair_token_b >> 3], (unsigned int)pair_route_b << (unsigned int)((pair_token_b & 7) * 4));
                    }
                }
                __syncthreads();
                int first_count_a = 0;
                int first_count_b = 0;
                if (bitmap_words > tid) {
                    int _popc_4 = __popc(bitmap_a[tid]);
                    first_count_a = _popc_4;
                    int _popc_5 = __popc(bitmap_b[tid]);
                    first_count_b = _popc_5;
                }
                int first_inclusive_a = first_count_a;
                int first_inclusive_b = first_count_b;
                int _shfl_up_10 = __shfl_up_sync(0xFFFFFFFF, first_inclusive_a, 1, 32);
                int first_peer_a = _shfl_up_10;
                int _shfl_up_11 = __shfl_up_sync(0xFFFFFFFF, first_inclusive_b, 1, 32);
                int first_peer_b = _shfl_up_11;
                if (lane >= 1) {
                    first_inclusive_a += first_peer_a;
                    first_inclusive_b += first_peer_b;
                }
                int _shfl_up_12 = __shfl_up_sync(0xFFFFFFFF, first_inclusive_a, 2, 32);
                int first_peer_a_0 = _shfl_up_12;
                int _shfl_up_13 = __shfl_up_sync(0xFFFFFFFF, first_inclusive_b, 2, 32);
                int first_peer_b_1 = _shfl_up_13;
                if (lane >= 2) {
                    first_inclusive_a += first_peer_a_0;
                    first_inclusive_b += first_peer_b_1;
                }
                int _shfl_up_14 = __shfl_up_sync(0xFFFFFFFF, first_inclusive_a, 4, 32);
                int first_peer_a_2 = _shfl_up_14;
                int _shfl_up_15 = __shfl_up_sync(0xFFFFFFFF, first_inclusive_b, 4, 32);
                int first_peer_b_3 = _shfl_up_15;
                if (lane >= 4) {
                    first_inclusive_a += first_peer_a_2;
                    first_inclusive_b += first_peer_b_3;
                }
                int _shfl_up_16 = __shfl_up_sync(0xFFFFFFFF, first_inclusive_a, 8, 32);
                int first_peer_a_4 = _shfl_up_16;
                int _shfl_up_17 = __shfl_up_sync(0xFFFFFFFF, first_inclusive_b, 8, 32);
                int first_peer_b_5 = _shfl_up_17;
                if (lane >= 8) {
                    first_inclusive_a += first_peer_a_4;
                    first_inclusive_b += first_peer_b_5;
                }
                int _shfl_up_18 = __shfl_up_sync(0xFFFFFFFF, first_inclusive_a, 16, 32);
                int first_peer_a_6 = _shfl_up_18;
                int _shfl_up_19 = __shfl_up_sync(0xFFFFFFFF, first_inclusive_b, 16, 32);
                int first_peer_b_7 = _shfl_up_19;
                if (lane >= 16) {
                    first_inclusive_a += first_peer_a_6;
                    first_inclusive_b += first_peer_b_7;
                }
                if (lane == 31) {
                    chunk_prefix_a[warp] = first_inclusive_a;
                    chunk_prefix_b[warp] = first_inclusive_b;
                }
                int last_count_a = 0;
                int last_count_b = 0;
                int last_inclusive_a = 0;
                int last_inclusive_b = 0;
                if (warp == 0) {
                    if (bitmap_words > tid + 224) {
                        int _popc_6 = __popc(bitmap_a[tid + 224]);
                        last_count_a = _popc_6;
                        int _popc_7 = __popc(bitmap_b[tid + 224]);
                        last_count_b = _popc_7;
                    }
                    last_inclusive_a = last_count_a;
                    last_inclusive_b = last_count_b;
                    int _shfl_up_20 = __shfl_up_sync(0xFFFFFFFF, last_inclusive_a, 1, 32);
                    int last_peer_a = _shfl_up_20;
                    int _shfl_up_21 = __shfl_up_sync(0xFFFFFFFF, last_inclusive_b, 1, 32);
                    int last_peer_b = _shfl_up_21;
                    if (lane >= 1) {
                        last_inclusive_a += last_peer_a;
                        last_inclusive_b += last_peer_b;
                    }
                    int _shfl_up_22 = __shfl_up_sync(0xFFFFFFFF, last_inclusive_a, 2, 32);
                    int last_peer_a_0 = _shfl_up_22;
                    int _shfl_up_23 = __shfl_up_sync(0xFFFFFFFF, last_inclusive_b, 2, 32);
                    int last_peer_b_1 = _shfl_up_23;
                    if (lane >= 2) {
                        last_inclusive_a += last_peer_a_0;
                        last_inclusive_b += last_peer_b_1;
                    }
                    int _shfl_up_24 = __shfl_up_sync(0xFFFFFFFF, last_inclusive_a, 4, 32);
                    int last_peer_a_2 = _shfl_up_24;
                    int _shfl_up_25 = __shfl_up_sync(0xFFFFFFFF, last_inclusive_b, 4, 32);
                    int last_peer_b_3 = _shfl_up_25;
                    if (lane >= 4) {
                        last_inclusive_a += last_peer_a_2;
                        last_inclusive_b += last_peer_b_3;
                    }
                    int _shfl_up_26 = __shfl_up_sync(0xFFFFFFFF, last_inclusive_a, 8, 32);
                    int last_peer_a_4 = _shfl_up_26;
                    int _shfl_up_27 = __shfl_up_sync(0xFFFFFFFF, last_inclusive_b, 8, 32);
                    int last_peer_b_5 = _shfl_up_27;
                    if (lane >= 8) {
                        last_inclusive_a += last_peer_a_4;
                        last_inclusive_b += last_peer_b_5;
                    }
                    int _shfl_up_28 = __shfl_up_sync(0xFFFFFFFF, last_inclusive_a, 16, 32);
                    int last_peer_a_6 = _shfl_up_28;
                    int _shfl_up_29 = __shfl_up_sync(0xFFFFFFFF, last_inclusive_b, 16, 32);
                    int last_peer_b_7 = _shfl_up_29;
                    if (lane >= 16) {
                        last_inclusive_a += last_peer_a_6;
                        last_inclusive_b += last_peer_b_7;
                    }
                    if (lane == 31) {
                        chunk_prefix_a[7] = last_inclusive_a;
                        chunk_prefix_b[7] = last_inclusive_b;
                    }
                }
                __syncthreads();
                if (warp == 0) {
                    int chunk_count_a = 0;
                    int chunk_count_b = 0;
                    if (lane < 8) {
                        chunk_count_a = chunk_prefix_a[lane];
                        chunk_count_b = chunk_prefix_b[lane];
                    }
                    int chunk_inclusive_a = chunk_count_a;
                    int chunk_inclusive_b = chunk_count_b;
                    int _shfl_up_30 = __shfl_up_sync(0xFFFFFFFF, chunk_inclusive_a, 1, 32);
                    int chunk_peer_a = _shfl_up_30;
                    int _shfl_up_31 = __shfl_up_sync(0xFFFFFFFF, chunk_inclusive_b, 1, 32);
                    int chunk_peer_b = _shfl_up_31;
                    if (lane >= 1) {
                        chunk_inclusive_a += chunk_peer_a;
                        chunk_inclusive_b += chunk_peer_b;
                    }
                    int _shfl_up_32 = __shfl_up_sync(0xFFFFFFFF, chunk_inclusive_a, 2, 32);
                    int chunk_peer_a_0 = _shfl_up_32;
                    int _shfl_up_33 = __shfl_up_sync(0xFFFFFFFF, chunk_inclusive_b, 2, 32);
                    int chunk_peer_b_1 = _shfl_up_33;
                    if (lane >= 2) {
                        chunk_inclusive_a += chunk_peer_a_0;
                        chunk_inclusive_b += chunk_peer_b_1;
                    }
                    int _shfl_up_34 = __shfl_up_sync(0xFFFFFFFF, chunk_inclusive_a, 4, 32);
                    int chunk_peer_a_2 = _shfl_up_34;
                    int _shfl_up_35 = __shfl_up_sync(0xFFFFFFFF, chunk_inclusive_b, 4, 32);
                    int chunk_peer_b_3 = _shfl_up_35;
                    if (lane >= 4) {
                        chunk_inclusive_a += chunk_peer_a_2;
                        chunk_inclusive_b += chunk_peer_b_3;
                    }
                    int _shfl_up_36 = __shfl_up_sync(0xFFFFFFFF, chunk_inclusive_a, 8, 32);
                    int chunk_peer_a_4 = _shfl_up_36;
                    int _shfl_up_37 = __shfl_up_sync(0xFFFFFFFF, chunk_inclusive_b, 8, 32);
                    int chunk_peer_b_5 = _shfl_up_37;
                    if (lane >= 8) {
                        chunk_inclusive_a += chunk_peer_a_4;
                        chunk_inclusive_b += chunk_peer_b_5;
                    }
                    int _shfl_up_38 = __shfl_up_sync(0xFFFFFFFF, chunk_inclusive_a, 16, 32);
                    int chunk_peer_a_6 = _shfl_up_38;
                    int _shfl_up_39 = __shfl_up_sync(0xFFFFFFFF, chunk_inclusive_b, 16, 32);
                    int chunk_peer_b_7 = _shfl_up_39;
                    if (lane >= 16) {
                        chunk_inclusive_a += chunk_peer_a_6;
                        chunk_inclusive_b += chunk_peer_b_7;
                    }
                    if (lane < 8) {
                        chunk_prefix_a[lane] = chunk_inclusive_a - chunk_count_a;
                        chunk_prefix_b[lane] = chunk_inclusive_b - chunk_count_b;
                    }
                }
                __syncthreads();
                if (bitmap_words > tid) {
                    word_prefix_a[tid] = chunk_prefix_a[warp] + first_inclusive_a - first_count_a;
                    word_prefix_b[tid] = chunk_prefix_b[warp] + first_inclusive_b - first_count_b;
                }
                if (bitmap_words > tid + 224) {
                    word_prefix_a[tid + 224] = chunk_prefix_a[7] + last_inclusive_a - last_count_a;
                    word_prefix_b[tid + 224] = chunk_prefix_b[7] + last_inclusive_b - last_count_b;
                }
                __syncthreads();
                #pragma unroll 1
                for (int output_word = warp; output_word < bitmap_words; output_word += 7) {
                    unsigned int word_bits_a = bitmap_a[output_word];
                    unsigned int word_bits_b = bitmap_b[output_word];
                    unsigned int lane_bit = (unsigned int)1 << (unsigned int)lane;
                    unsigned int lower_bits = lane_bit - (unsigned int)1;
                    int output_token = (output_word << 5) + lane;
                    if ((word_bits_a & lane_bit) != 0) {
                        int _popc_8 = __popc(word_bits_a & lower_bits);
                        int output_rank_a = word_prefix_a[output_word] + _popc_8;
                        unsigned int route_word_a = routes_a[output_token >> 3];
                        unsigned int output_route_a = route_word_a >> (unsigned int)((output_token & 7) * 4) & (unsigned int)15;
                        sorted_token_ids[sort_start_a + output_rank_a] = output_token * 16 + (int)output_route_a;
                    }
                    if ((word_bits_b & lane_bit) != 0) {
                        int _popc_9 = __popc(word_bits_b & lower_bits);
                        int output_rank_b = word_prefix_b[output_word] + _popc_9;
                        unsigned int route_word_b = routes_b[output_token >> 3];
                        unsigned int output_route_b = route_word_b >> (unsigned int)((output_token & 7) * 4) & (unsigned int)15;
                        sorted_token_ids[sort_start_b + output_rank_b] = output_token * 16 + (int)output_route_b;
                    }
                }
                __syncthreads();
            }
        }
    }
}

} // extern "C"
