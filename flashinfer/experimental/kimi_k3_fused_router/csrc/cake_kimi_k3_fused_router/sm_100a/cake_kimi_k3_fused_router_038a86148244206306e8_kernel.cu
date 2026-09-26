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
#define SMEM_ROW_BITS_OFF 1280
#define SMEM_ROW_BITS_STAGE_BYTES 3584
#define SMEM_ROW_BITS_STRIDE 3584
#define SMEM_ROUTE_SLOTS_OFF 4864
#define SMEM_ROUTE_SLOTS_STAGE_BYTES 3584
#define SMEM_ROUTE_SLOTS_STRIDE 3584
#define SMEM_OFFSETS_OFF 12032
#define SMEM_OFFSETS_STAGE_BYTES 3588
#define SMEM_OFFSETS_STRIDE 3588
#define SMEM_SELECTED_IDS_OFF 15680
#define SMEM_SELECTED_IDS_STAGE_BYTES 256
#define SMEM_SELECTED_IDS_STRIDE 256
#define SMEM_WARP_TOTALS_OFF 16768
#define SMEM_WARP_TOTALS_STAGE_BYTES 28
#define SMEM_WARP_TOTALS_STRIDE 28
#define SMEM_TOTAL 16896
#define THREADS 224
#define BLOCK_M 8
#define NUM_EXPERTS 896
#define TOP_K 16
#define ITEMS_PER_THREAD 4
#define NUM_WARPS 7
#define RADIX_CLEAR_THREADS 128
#define RADIX_SCAN_THREADS 32
#define RADIX_BINS_PER_LANE 8
#define MAX_BLOCK_M 16
#define ROWS 4
#define ROUTE_WORDS 1
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


__device__ __forceinline__ uint32_t smem_addr(const void* ptr) {
    uint32_t addr;
    asm("{\n\t"
        ".reg .u64 u64addr;\n\t"
        "cvta.to.shared.u64 u64addr, %1;\n\t"
        "cvt.u32.u64 %0, u64addr;\n\t"
        "}\n" : "=r"(addr) : "l"(ptr));
    return addr;
}


__device__ __forceinline__ uint32_t mapa_to_rank(uint32_t local_addr, uint32_t rank) {
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(remote) : "r"(local_addr), "r"(rank));
    return remote;
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

__global__ __launch_bounds__(224) __cluster_dims__(4,1,1) void
kernel_cake_kimi_k3_fused_router_038a86148244206306e8(float* __restrict__ logits, float* __restrict__ bias, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 4;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 4;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

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
    unsigned int* row_bits = reinterpret_cast<unsigned int*>(smem_raw + 1280);
    const int row_bits_addr = smem + 1280;
    unsigned int* route_slots = reinterpret_cast<unsigned int*>(smem_raw + 4864);
    const int route_slots_addr = smem + 4864;
    int* offsets = reinterpret_cast<int*>(smem_raw + 12032);
    const int offsets_addr = smem + 12032;
    int* selected_ids = reinterpret_cast<int*>(smem_raw + 15680);
    const int selected_ids_addr = smem + 15680;
    int* warp_totals = reinterpret_cast<int*>(smem_raw + 16768);
    const int warp_totals_addr = smem + 16768;

    // === Task calls (dependency order) ===
    #pragma unroll 1
    for (int clear_word = tid; clear_word < NUM_EXPERTS + NUM_EXPERTS * ROUTE_WORDS; clear_word += THREADS) {
        if (clear_word < NUM_EXPERTS) {
            row_bits[clear_word] = (unsigned int)0;
        } else {
            route_slots[clear_word - NUM_EXPERTS] = (unsigned int)0;
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
    __syncthreads();
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
        float unbiased[ITEMS_PER_THREAD];
        unsigned int keys[ITEMS_PER_THREAD];
        int active[ITEMS_PER_THREAD];
        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; item++) {
            float _expf_0 = __expf(-_vec_load_1[item]);
            float _fdiv_full_0;
            asm volatile("div.full.f32 %0, %1, %2;" : "=f"(_fdiv_full_0) : "f"(1.0f), "f"(1.0f + _expf_0));
            float score = _fdiv_full_0;
            float ranking = score + _vec_load_0[item];
            if (ranking == 0.0f) {
                ranking = 0.0f;
            }
            unsigned int bits = __as_u32(ranking);
            unsigned int ordered = bits | 2147483648;
            if (bits >= 2147483648) {
                ordered = bits ^ 4294967295;
            }
            unbiased[item] = score;
            keys[item] = ordered;
            active[item] = 1;
        }
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
            int scan_counts[RADIX_BINS_PER_LANE];
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
                unsigned int _warp_redux_u32_0;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(warp_match_record));
                warp_match_record = _warp_redux_u32_0;
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
        int selected[ITEMS_PER_THREAD];
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
            int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, tie_inclusive, 1, 32);
            int tie_peer = _shfl_up_5;
            if (lane >= 1) {
                tie_inclusive += tie_peer;
            }
            int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, tie_inclusive, 2, 32);
            int tie_peer_0 = _shfl_up_6;
            if (lane >= 2) {
                tie_inclusive += tie_peer_0;
            }
            int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, tie_inclusive, 4, 32);
            int tie_peer_1 = _shfl_up_7;
            if (lane >= 4) {
                tie_inclusive += tie_peer_1;
            }
            int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, tie_inclusive, 8, 32);
            int tie_peer_2 = _shfl_up_8;
            if (lane >= 8) {
                tie_inclusive += tie_peer_2;
            }
            int _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, tie_inclusive, 16, 32);
            int tie_peer_3 = _shfl_up_9;
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
            int _shfl_0 = __shfl_sync(0xFFFFFFFF, tie_prefix_lane0, 0);
            int tie_prefix = _shfl_0;
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
        int _shfl_up_10 = __shfl_up_sync(0xFFFFFFFF, selected_inclusive, 1, 32);
        int selected_peer = _shfl_up_10;
        if (lane >= 1) {
            selected_inclusive += selected_peer;
        }
        int _shfl_up_11 = __shfl_up_sync(0xFFFFFFFF, selected_inclusive, 2, 32);
        int selected_peer_0 = _shfl_up_11;
        if (lane >= 2) {
            selected_inclusive += selected_peer_0;
        }
        int _shfl_up_12 = __shfl_up_sync(0xFFFFFFFF, selected_inclusive, 4, 32);
        int selected_peer_1 = _shfl_up_12;
        if (lane >= 4) {
            selected_inclusive += selected_peer_1;
        }
        int _shfl_up_13 = __shfl_up_sync(0xFFFFFFFF, selected_inclusive, 8, 32);
        int selected_peer_2 = _shfl_up_13;
        if (lane >= 8) {
            selected_inclusive += selected_peer_2;
        }
        int _shfl_up_14 = __shfl_up_sync(0xFFFFFFFF, selected_inclusive, 16, 32);
        int selected_peer_3 = _shfl_up_14;
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
        int _shfl_1 = __shfl_sync(0xFFFFFFFF, selected_prefix_lane0, 0);
        int selected_prefix = _shfl_1;
        int selected_rank = selected_prefix + selected_inclusive - selected_count;
        #pragma unroll
        for (int item_compact = 0; item_compact < ITEMS_PER_THREAD; item_compact++) {
            if (selected[item_compact] != 0) {
                winner_ids[selected_rank] = expert_base + item_compact;
                winner_scores[selected_rank] = unbiased[item_compact];
            }
            selected_rank += selected[item_compact];
        }
        __syncthreads();
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
            float _fdiv_full_1;
            asm volatile("div.full.f32 %0, %1, %2;" : "=f"(_fdiv_full_1) : "f"(winner_scores[tid]), "f"(selected_norm));
            topk_weights[output_index] = _fdiv_full_1;
            topk_ids[output_index] = selected_id;
            selected_ids[token * TOP_K + tid] = selected_id;
        }
        __syncthreads();
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    #pragma unroll 1
    for (int pair_slot = tid; pair_slot < ROWS * TOP_K; pair_slot += THREADS) {
        int owner_rank = pair_slot >> 4;
        int gathered_id = selected_ids[pair_slot];
        if (owner_rank != cta_rank) {
            uint32_t _mapa_0;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_0) : "r"(histogram_addr), "r"(owner_rank));
            unsigned int peer_pool = _mapa_0;
            int _cluster_ld_0;
            asm volatile(
                "ld.shared::cluster.s32 %0, [%1];"
                : "=r"(_cluster_ld_0) : "r"(peer_pool + (unsigned int)(15680 + pair_slot * 4)) : "memory");
            gathered_id = _cluster_ld_0;
            selected_ids[pair_slot] = gathered_id;
        }
        int gathered_row = pair_slot >> 4;
        atomicAdd(&row_bits[gathered_id], (unsigned int)1 << (unsigned int)gathered_row);
        atomicAdd(&route_slots[gathered_id * ROUTE_WORDS + (gathered_row >> 3)], (unsigned int)(pair_slot & 15) << (unsigned int)((gathered_row & 7) * 4));
    }
    __syncthreads();
    int expert_scan_base = warp * 32 * ITEMS_PER_THREAD + lane * ITEMS_PER_THREAD;
    int lane_prefixes[ITEMS_PER_THREAD];
    int lane_total = 0;
    #pragma unroll
    for (int count_slot = 0; count_slot < ITEMS_PER_THREAD; count_slot++) {
        int count_expert = expert_scan_base + count_slot;
        int _popc_0 = __popc(row_bits[count_expert]);
        int count_value = _popc_0;
        if (cta_rank == 0) {
            expert_counts[count_expert] = count_value;
            expert_scatter_offsets[count_expert] = count_value;
        }
        lane_prefixes[count_slot] = lane_total;
        lane_total += count_value + BLOCK_MASK & ~BLOCK_MASK;
    }
    int warp_inclusive = lane_total;
    int _shfl_up_15 = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, 1, 32);
    int scan_peer_4 = _shfl_up_15;
    if (lane >= 1) {
        warp_inclusive += scan_peer_4;
    }
    int _shfl_up_16 = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, 2, 32);
    int scan_peer_0_1 = _shfl_up_16;
    if (lane >= 2) {
        warp_inclusive += scan_peer_0_1;
    }
    int _shfl_up_17 = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, 4, 32);
    int scan_peer_1_1 = _shfl_up_17;
    if (lane >= 4) {
        warp_inclusive += scan_peer_1_1;
    }
    int _shfl_up_18 = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, 8, 32);
    int scan_peer_2_1 = _shfl_up_18;
    if (lane >= 8) {
        warp_inclusive += scan_peer_2_1;
    }
    int _shfl_up_19 = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, 16, 32);
    int scan_peer_3_1 = _shfl_up_19;
    if (lane >= 16) {
        warp_inclusive += scan_peer_3_1;
    }
    if (lane == 31) {
        warp_totals[warp] = warp_inclusive;
    }
    __syncthreads();
    int warp_prefix_lane0 = 0;
    if (lane == 0) {
        #pragma unroll
        for (int prior_warp = 0; prior_warp < NUM_WARPS; prior_warp++) {
            if (prior_warp < warp) {
                warp_prefix_lane0 += warp_totals[prior_warp];
            }
        }
    }
    int _shfl_2 = __shfl_sync(0xFFFFFFFF, warp_prefix_lane0, 0);
    int warp_prefix = _shfl_2;
    int lane_prefix = warp_prefix + warp_inclusive - lane_total;
    #pragma unroll
    for (int offset_slot = 0; offset_slot < ITEMS_PER_THREAD; offset_slot++) {
        int offset_expert = expert_scan_base + offset_slot;
        int offset_value = lane_prefix + lane_prefixes[offset_slot];
        offsets[offset_expert] = offset_value;
        if (cta_rank == 0) {
            expert_offsets[offset_expert] = offset_value;
        }
    }
    if (tid == THREADS - 1) {
        int padded_total = warp_prefix + warp_inclusive;
        offsets[NUM_EXPERTS] = padded_total;
        if (cta_rank == 0) {
            expert_offsets[NUM_EXPERTS] = padded_total;
            num_tokens_post_padded[0] = padded_total;
        }
    }
    __syncthreads();
    int total_pairs = M * TOP_K;
    #pragma unroll 1
    for (int local_pair = warp; local_pair < TOP_K; local_pair += NUM_WARPS) {
        int owner_pair = cta_rank * TOP_K + local_pair;
        int owner_expert = selected_ids[owner_pair];
        unsigned int owner_bits = row_bits[owner_expert];
        int owner_row = cta_rank;
        unsigned int owner_lower = ((unsigned int)1 << (unsigned int)owner_row) - 1;
        if ((owner_bits & owner_lower) == 0) {
            int _popc_1 = __popc(owner_bits);
            int owner_count = _popc_1;
            int owner_start = offsets[owner_expert];
            if (lane < ROWS) {
                unsigned int emit_bit = (unsigned int)1 << (unsigned int)lane;
                if ((owner_bits & emit_bit) != 0) {
                    int _popc_2 = __popc(owner_bits & emit_bit - 1);
                    int emit_rank = _popc_2;
                    unsigned int packed_slots = route_slots[owner_expert * ROUTE_WORDS + (lane >> 3)];
                    int emit_slot = (int)(packed_slots >> (unsigned int)((lane & 7) * 4) & 15);
                    int grouped = owner_start + emit_rank;
                    sorted_token_ids[grouped] = lane * TOP_K + emit_slot;
                    if ((emit_rank & BLOCK_MASK) == 0) {
                        expert_ids[grouped >> BLOCK_SHIFT] = owner_expert;
                    }
                }
            }
            int padded_count = owner_count + BLOCK_MASK & ~BLOCK_MASK;
            if (lane < padded_count - owner_count) {
                sorted_token_ids[owner_start + owner_count + lane] = total_pairs;
            }
        }
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

} // extern "C"
