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
#define SMEM_HISTOGRAM_OFF 1024
#define SMEM_HISTOGRAM_STAGE_BYTES 2048
#define SMEM_HISTOGRAM_STRIDE 2048
#define SMEM_TIE_WARP_SUMS_OFF 3072
#define SMEM_TIE_WARP_SUMS_STAGE_BYTES 64
#define SMEM_TIE_WARP_SUMS_STRIDE 64
#define SMEM_SELECTED_WARP_SUMS_OFF 3136
#define SMEM_SELECTED_WARP_SUMS_STAGE_BYTES 64
#define SMEM_SELECTED_WARP_SUMS_STRIDE 64
#define SMEM_WINNER_IDS_OFF 3200
#define SMEM_WINNER_IDS_STAGE_BYTES 128
#define SMEM_WINNER_IDS_STRIDE 128
#define SMEM_WINNER_SCORES_OFF 3328
#define SMEM_WINNER_SCORES_STAGE_BYTES 128
#define SMEM_WINNER_SCORES_STRIDE 128
#define SMEM_RADIX_MATCH_OFF 3456
#define SMEM_RADIX_MATCH_STAGE_BYTES 16
#define SMEM_RADIX_MATCH_STRIDE 16
#define SMEM_BOUND_VALUES_OFF 3472
#define SMEM_BOUND_VALUES_STAGE_BYTES 256
#define SMEM_BOUND_VALUES_STRIDE 256
#define SMEM_CAND_COUNTS_OFF 3728
#define SMEM_CAND_COUNTS_STAGE_BYTES 64
#define SMEM_CAND_COUNTS_STRIDE 64
#define SMEM_CAND_KEYS_OFF 3792
#define SMEM_CAND_KEYS_STAGE_BYTES 1792
#define SMEM_CAND_KEYS_STRIDE 1792
#define SMEM_CAND_IDS_OFF 5584
#define SMEM_CAND_IDS_STAGE_BYTES 1792
#define SMEM_CAND_IDS_STRIDE 1792
#define SMEM_CAND_SCORES_OFF 7376
#define SMEM_CAND_SCORES_STAGE_BYTES 1792
#define SMEM_CAND_SCORES_STRIDE 1792
#define SMEM_RANK_KEYS_OFF 9168
#define SMEM_RANK_KEYS_STAGE_BYTES 256
#define SMEM_RANK_KEYS_STRIDE 256
#define SMEM_ROW_BITS_OFF 9424
#define SMEM_ROW_BITS_STAGE_BYTES 3584
#define SMEM_ROW_BITS_STRIDE 3584
#define SMEM_LEVEL_MASKS_OFF 13008
#define SMEM_LEVEL_MASKS_STAGE_BYTES 448
#define SMEM_LEVEL_MASKS_STRIDE 448
#define SMEM_SELECTED_IDS_OFF 13456
#define SMEM_SELECTED_IDS_STAGE_BYTES 2048
#define SMEM_SELECTED_IDS_STRIDE 2048
#define SMEM_TOTAL 15616
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
#define BLOCK_MASK (BLOCK_M - 1)
#define BLOCK_SHIFT (3 + (BLOCK_M >> 4))
#define MASK_LEVELS (32 >> BLOCK_SHIFT)
#define LAUNCH_MIN_BLOCKS 1

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


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}


__device__ __forceinline__ void mbarrier_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}



// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
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

__global__ __launch_bounds__(448, LAUNCH_MIN_BLOCKS) __cluster_dims__(16,1,1) void
kernel_cake_kimi_k3_fused_router_6d1a3cbbec0fdd51dbf5(float* __restrict__ logits, float* __restrict__ bias, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M)
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

    const int mbar_base = smem;
    #define ids_full_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 16;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 16;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    int* histogram = reinterpret_cast<int*>(smem_raw + 1024);
    const int histogram_addr = smem + 1024;
    int* tie_warp_sums = reinterpret_cast<int*>(smem_raw + 3072);
    const int tie_warp_sums_addr = smem + 3072;
    int* selected_warp_sums = reinterpret_cast<int*>(smem_raw + 3136);
    const int selected_warp_sums_addr = smem + 3136;
    int* winner_ids = reinterpret_cast<int*>(smem_raw + 3200);
    const int winner_ids_addr = smem + 3200;
    float* winner_scores = reinterpret_cast<float*>(smem_raw + 3328);
    const int winner_scores_addr = smem + 3328;
    int* radix_match = reinterpret_cast<int*>(smem_raw + 3456);
    const int radix_match_addr = smem + 3456;
    unsigned int* bound_values = reinterpret_cast<unsigned int*>(smem_raw + 3472);
    const int bound_values_addr = smem + 3472;
    int* cand_counts = reinterpret_cast<int*>(smem_raw + 3728);
    const int cand_counts_addr = smem + 3728;
    unsigned int* cand_keys = reinterpret_cast<unsigned int*>(smem_raw + 3792);
    const int cand_keys_addr = smem + 3792;
    int* cand_ids = reinterpret_cast<int*>(smem_raw + 5584);
    const int cand_ids_addr = smem + 5584;
    float* cand_scores = reinterpret_cast<float*>(smem_raw + 7376);
    const int cand_scores_addr = smem + 7376;
    unsigned int* rank_keys = reinterpret_cast<unsigned int*>(smem_raw + 9168);
    const int rank_keys_addr = smem + 9168;
    unsigned int* row_bits = reinterpret_cast<unsigned int*>(smem_raw + 9424);
    const int row_bits_addr = smem + 9424;
    unsigned int* level_masks = reinterpret_cast<unsigned int*>(smem_raw + 13008);
    const int level_masks_addr = smem + 13008;
    int* selected_ids = reinterpret_cast<int*>(smem_raw + 13456);
    const int selected_ids_addr = smem + 13456;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 1 barriers)
    // Mbarriers at smem_raw[0..8)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // ids_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 0, 2048);
        }
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    // === Task calls (declaration order) ===
    int group = warp + (32 - NUM_WARPS) >> 5;
    int group_tid = tid - group * THREADS;
    int group_warp = warp - group * NUM_WARPS;
    int group_row = cta_rank + group * 16;
    int expert_base = group_tid * ITEMS_PER_THREAD;
    float _vec_load_0[4];
    {
        unsigned _v4_0_0;
        unsigned _v4_0_1;
        unsigned _v4_0_2;
        unsigned _v4_0_3;
        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
            : "=r"(_v4_0_0), "=r"(_v4_0_1), "=r"(_v4_0_2), "=r"(_v4_0_3) : "l"((const void*)(bias + (expert_base))) : "memory");
        _vec_load_0[0 + 0] = __uint_as_float(_v4_0_0);
        _vec_load_0[0 + 1] = __uint_as_float(_v4_0_1);
        _vec_load_0[0 + 2] = __uint_as_float(_v4_0_2);
        _vec_load_0[0 + 3] = __uint_as_float(_v4_0_3);
    }
    int load_row = group_row;
    if (load_row >= M) {
        load_row = M - 1;
    }
    float _vec_load_1[4];
    {
        unsigned _v4_1_0;
        unsigned _v4_1_1;
        unsigned _v4_1_2;
        unsigned _v4_1_3;
        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
            : "=r"(_v4_1_0), "=r"(_v4_1_1), "=r"(_v4_1_2), "=r"(_v4_1_3) : "l"((const void*)(logits + ((unsigned long long)load_row * (unsigned long long)NUM_EXPERTS + (unsigned long long)expert_base))) : "memory");
        _vec_load_1[0 + 0] = __uint_as_float(_v4_1_0);
        _vec_load_1[0 + 1] = __uint_as_float(_v4_1_1);
        _vec_load_1[0 + 2] = __uint_as_float(_v4_1_2);
        _vec_load_1[0 + 3] = __uint_as_float(_v4_1_3);
    }
    if (warp == 0) {
        if (elect_sync()) {
            mbarrier_arrive(ids_full_addr);
        }
    }
    #pragma unroll
    for (int clear_slot = 0; clear_slot < 2; clear_slot++) {
        row_bits[clear_slot * 448 + tid] = (unsigned int)0;
    }
    if (tid < 112) {
        level_masks[tid] = (unsigned int)0;
    }
    int hist_base = group * 256;
    int sums_base = group * 8;
    int winner_base = group * TOP_K;
    int match_slot = group * 2;
    int bound_base = group * 32;
    int cand_base = group * 224;
    int rank_base = group * 32;
    if (group_tid == 0) {
        #pragma unroll
        for (int winner_init = 0; winner_init < TOP_K; winner_init++) {
            winner_ids[winner_base + winner_init] = 0;
            winner_scores[winner_base + winner_init] = 0.0f;
        }
    }
    if (group_tid < 4) {
        bound_values[bound_base + 28 + group_tid] = (unsigned int)0;
    }
    unsigned int one_u32 = (unsigned int)1;
    unsigned int lanemask_lt = (one_u32 << (unsigned int)lane) - 1;
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (group_row < M) {
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
            bound_values[bound_base + group_warp * 4 + lane] = published;
        }
        asm volatile("barrier.sync %0, 224;" :: "r"(group + 1) : "memory");
        unsigned int bound_mine = bound_values[bound_base + lane];
        int bound_part0 = ((bound_mine <= bound_values[bound_base]) ? 1 : 0) + ((bound_mine <= bound_values[bound_base + 1]) ? 1 : 0) + ((bound_mine <= bound_values[bound_base + 2]) ? 1 : 0) + ((bound_mine <= bound_values[bound_base + 3]) ? 1 : 0);
        int bound_part1 = ((bound_mine <= bound_values[bound_base + 4]) ? 1 : 0) + ((bound_mine <= bound_values[bound_base + 4 + 1]) ? 1 : 0) + ((bound_mine <= bound_values[bound_base + 4 + 2]) ? 1 : 0) + ((bound_mine <= bound_values[bound_base + 4 + 3]) ? 1 : 0);
        int bound_part2 = ((bound_mine <= bound_values[bound_base + 8]) ? 1 : 0) + ((bound_mine <= bound_values[bound_base + 8 + 1]) ? 1 : 0) + ((bound_mine <= bound_values[bound_base + 8 + 2]) ? 1 : 0) + ((bound_mine <= bound_values[bound_base + 8 + 3]) ? 1 : 0);
        int bound_part3 = ((bound_mine <= bound_values[bound_base + 12]) ? 1 : 0) + ((bound_mine <= bound_values[bound_base + 12 + 1]) ? 1 : 0) + ((bound_mine <= bound_values[bound_base + 12 + 2]) ? 1 : 0) + ((bound_mine <= bound_values[bound_base + 12 + 3]) ? 1 : 0);
        int bound_part4 = ((bound_mine <= bound_values[bound_base + 16]) ? 1 : 0) + ((bound_mine <= bound_values[bound_base + 16 + 1]) ? 1 : 0) + ((bound_mine <= bound_values[bound_base + 16 + 2]) ? 1 : 0) + ((bound_mine <= bound_values[bound_base + 16 + 3]) ? 1 : 0);
        int bound_part5 = ((bound_mine <= bound_values[bound_base + 20]) ? 1 : 0) + ((bound_mine <= bound_values[bound_base + 20 + 1]) ? 1 : 0) + ((bound_mine <= bound_values[bound_base + 20 + 2]) ? 1 : 0) + ((bound_mine <= bound_values[bound_base + 20 + 3]) ? 1 : 0);
        int bound_part6 = ((bound_mine <= bound_values[bound_base + 24]) ? 1 : 0) + ((bound_mine <= bound_values[bound_base + 24 + 1]) ? 1 : 0) + ((bound_mine <= bound_values[bound_base + 24 + 2]) ? 1 : 0) + ((bound_mine <= bound_values[bound_base + 24 + 3]) ? 1 : 0);
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
        int cand_region = cand_base + group_warp * 32;
        if (keys[0] >= cand_floor && cand_slot < 32) {
            int cand_index0 = cand_region + cand_slot;
            cand_keys[cand_index0] = keys[0];
            cand_ids[cand_index0] = expert_base;
            cand_scores[cand_index0] = unbiased[0];
        }
        cand_slot += ((keys[0] >= cand_floor) ? 1 : 0);
        if (keys[1] >= cand_floor && cand_slot < 32) {
            int cand_index_0 = cand_region + cand_slot;
            cand_keys[cand_index_0] = keys[1];
            cand_ids[cand_index_0] = expert_base + 1;
            cand_scores[cand_index_0] = unbiased[1];
        }
        cand_slot += ((keys[1] >= cand_floor) ? 1 : 0);
        if (keys[2] >= cand_floor && cand_slot < 32) {
            int cand_index_1 = cand_region + cand_slot;
            cand_keys[cand_index_1] = keys[2];
            cand_ids[cand_index_1] = expert_base + 2;
            cand_scores[cand_index_1] = unbiased[2];
        }
        cand_slot += ((keys[2] >= cand_floor) ? 1 : 0);
        if (keys[3] >= cand_floor && cand_slot < 32) {
            int cand_index_2 = cand_region + cand_slot;
            cand_keys[cand_index_2] = keys[3];
            cand_ids[cand_index_2] = expert_base + 3;
            cand_scores[cand_index_2] = unbiased[3];
        }
        if (lane == 0) {
            cand_counts[sums_base + group_warp] = cand_warp;
        }
        asm volatile("barrier.sync %0, 224;" :: "r"(group + 1) : "memory");
        int cand_total = cand_counts[sums_base] + cand_counts[sums_base + 1] + (cand_counts[sums_base + 2] + cand_counts[sums_base + 3]) + (cand_counts[sums_base + 4] + cand_counts[sums_base + 5] + cand_counts[sums_base + 6]);
        if (cand_total > 32) {
            int remaining = TOP_K;
            int active_total = NUM_EXPERTS;
            unsigned int threshold = 0;
            #pragma unroll
            for (int radix_round = 0; radix_round < 4; radix_round++) {
                int shift = 24 - radix_round * 8;
                int match_base = match_slot + (radix_round & 1);
                if (group_tid < RADIX_CLEAR_THREADS) {
                    histogram[hist_base + group_tid] = 0;
                    histogram[hist_base + group_tid + RADIX_CLEAR_THREADS] = 0;
                }
                asm volatile("barrier.sync %0, 224;" :: "r"(group + 1) : "memory");
                #pragma unroll
                for (int item_hist = 0; item_hist < ITEMS_PER_THREAD; item_hist++) {
                    if (active[item_hist] != 0) {
                        int digit = (int)(keys[item_hist] >> (unsigned int)shift & 255);
                        atomicAdd(&histogram[hist_base + digit], 1);
                    }
                }
                asm volatile("barrier.sync %0, 224;" :: "r"(group + 1) : "memory");
                int scan_counts[8];
                int lane_count = 0;
                int inclusive = 0;
                unsigned int warp_match_record = 0;
                if (group_tid < RADIX_SCAN_THREADS) {
                    int scan_bin_base = group_tid * RADIX_BINS_PER_LANE;
                    #pragma unroll
                    for (int scan_offset = 0; scan_offset < RADIX_BINS_PER_LANE; scan_offset++) {
                        int scan_count = histogram[hist_base + scan_bin_base + scan_offset];
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
                    if (group_tid == 0) {
                        radix_match[match_base] = (int)warp_match_record;
                    }
                }
                asm volatile("barrier.sync %0, 224;" :: "r"(group + 1) : "memory");
                int match_record = radix_match[match_base];
                int chosen_bin = match_record & 255;
                remaining -= match_record >> 8 & 1023;
                active_total = match_record >> 18 & 1023;
                threshold |= (unsigned int)chosen_bin << (unsigned int)shift;
                #pragma unroll
                for (int item_filter = 0; item_filter < ITEMS_PER_THREAD; item_filter++) {
                    int item_digit = (int)(keys[item_filter] >> (unsigned int)shift & 255);
                    if (item_digit != chosen_bin) {
                        active[item_filter] = 0;
                    }
                }
            }
            int selected[4];
            int selected_count = 0;
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
                tie_warp_sums[sums_base + group_warp] = tie_inclusive;
            }
            asm volatile("barrier.sync %0, 224;" :: "r"(group + 1) : "memory");
            int tie_prefix_lane0 = 0;
            if (lane == 0) {
                #pragma unroll
                for (int tie_warp = 0; tie_warp < NUM_WARPS; tie_warp++) {
                    if (group_warp > tie_warp) {
                        tie_prefix_lane0 += tie_warp_sums[sums_base + tie_warp];
                    }
                }
            }
            int tie_prefix = __shfl_sync(0xFFFFFFFF, tie_prefix_lane0, 0);
            int tie_rank = tie_prefix + tie_inclusive - tie_count;
            #pragma unroll
            for (int item_select = 0; item_select < ITEMS_PER_THREAD; item_select++) {
                int choose_tie = 0;
                if (keys[item_select] > threshold) {
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
            int selected_inclusive = selected_count;
            int selected_peer = __shfl_up_sync(0xFFFFFFFF, selected_inclusive, 1, 32);
            if (lane >= 1) {
                selected_inclusive += selected_peer;
            }
            int selected_peer_4 = __shfl_up_sync(0xFFFFFFFF, selected_inclusive, 2, 32);
            if (lane >= 2) {
                selected_inclusive += selected_peer_4;
            }
            int selected_peer_5 = __shfl_up_sync(0xFFFFFFFF, selected_inclusive, 4, 32);
            if (lane >= 4) {
                selected_inclusive += selected_peer_5;
            }
            int selected_peer_6 = __shfl_up_sync(0xFFFFFFFF, selected_inclusive, 8, 32);
            if (lane >= 8) {
                selected_inclusive += selected_peer_6;
            }
            int selected_peer_7 = __shfl_up_sync(0xFFFFFFFF, selected_inclusive, 16, 32);
            if (lane >= 16) {
                selected_inclusive += selected_peer_7;
            }
            if (lane == 31) {
                selected_warp_sums[sums_base + group_warp] = selected_inclusive;
            }
            asm volatile("barrier.sync %0, 224;" :: "r"(group + 1) : "memory");
            int selected_prefix_lane0 = 0;
            if (lane == 0) {
                #pragma unroll
                for (int selected_warp = 0; selected_warp < NUM_WARPS; selected_warp++) {
                    if (group_warp > selected_warp) {
                        selected_prefix_lane0 += selected_warp_sums[sums_base + selected_warp];
                    }
                }
            }
            int selected_prefix = __shfl_sync(0xFFFFFFFF, selected_prefix_lane0, 0);
            int selected_rank = selected_prefix + selected_inclusive - selected_count;
            #pragma unroll
            for (int item_compact = 0; item_compact < ITEMS_PER_THREAD; item_compact++) {
                if (selected[item_compact] != 0) {
                    winner_ids[winner_base + selected_rank] = expert_base + item_compact;
                    winner_scores[winner_base + selected_rank] = unbiased[item_compact];
                }
                selected_rank += selected[item_compact];
            }
            asm volatile("barrier.sync %0, 224;" :: "r"(group + 1) : "memory");
        } else if (group_warp == 0) {
            int cand_region_0 = 0;
            int cand_offset = lane;
            int cand_prefix = 0;
            if (cand_prefix <= lane) {
                cand_region_0 = 0;
                cand_offset = lane - cand_prefix;
            }
            cand_prefix += cand_counts[sums_base];
            if (cand_prefix <= lane) {
                cand_region_0 = 1;
                cand_offset = lane - cand_prefix;
            }
            cand_prefix += cand_counts[sums_base + 1];
            if (cand_prefix <= lane) {
                cand_region_0 = 2;
                cand_offset = lane - cand_prefix;
            }
            cand_prefix += cand_counts[sums_base + 2];
            if (cand_prefix <= lane) {
                cand_region_0 = 3;
                cand_offset = lane - cand_prefix;
            }
            cand_prefix += cand_counts[sums_base + 3];
            if (cand_prefix <= lane) {
                cand_region_0 = 4;
                cand_offset = lane - cand_prefix;
            }
            cand_prefix += cand_counts[sums_base + 4];
            if (cand_prefix <= lane) {
                cand_region_0 = 5;
                cand_offset = lane - cand_prefix;
            }
            cand_prefix += cand_counts[sums_base + 5];
            if (cand_prefix <= lane) {
                cand_region_0 = 6;
                cand_offset = lane - cand_prefix;
            }
            cand_prefix += cand_counts[sums_base + 6];
            unsigned int my_key = 0;
            int my_id = 0;
            float my_score = 0.0f;
            if (cand_total > lane) {
                int my_slot = cand_base + cand_region_0 * 32 + cand_offset;
                my_key = cand_keys[my_slot];
                my_id = cand_ids[my_slot];
                my_score = cand_scores[my_slot];
            }
            rank_keys[rank_base + lane] = my_key;
            __syncwarp();
            int rank_part0 = 0;
            int rank_part1 = 0;
            int rank_part2 = 0;
            int rank_part3 = 0;
            rank_part0 += ((my_key < rank_keys[rank_base]) ? 1 : 0);
            rank_part1 += ((my_key < rank_keys[rank_base + 1]) ? 1 : 0);
            rank_part2 += ((my_key < rank_keys[rank_base + 2]) ? 1 : 0);
            rank_part3 += ((my_key < rank_keys[rank_base + 3]) ? 1 : 0);
            rank_part0 += ((my_key < rank_keys[rank_base + 4]) ? 1 : 0);
            rank_part1 += ((my_key < rank_keys[rank_base + 5]) ? 1 : 0);
            rank_part2 += ((my_key < rank_keys[rank_base + 6]) ? 1 : 0);
            rank_part3 += ((my_key < rank_keys[rank_base + 7]) ? 1 : 0);
            rank_part0 += ((my_key < rank_keys[rank_base + 8]) ? 1 : 0);
            rank_part1 += ((my_key < rank_keys[rank_base + 9]) ? 1 : 0);
            rank_part2 += ((my_key < rank_keys[rank_base + 10]) ? 1 : 0);
            rank_part3 += ((my_key < rank_keys[rank_base + 11]) ? 1 : 0);
            rank_part0 += ((my_key < rank_keys[rank_base + 12]) ? 1 : 0);
            rank_part1 += ((my_key < rank_keys[rank_base + 13]) ? 1 : 0);
            rank_part2 += ((my_key < rank_keys[rank_base + 14]) ? 1 : 0);
            rank_part3 += ((my_key < rank_keys[rank_base + 15]) ? 1 : 0);
            if (cand_total > 16) {
                rank_part0 += ((my_key < rank_keys[rank_base + 16]) ? 1 : 0);
                rank_part1 += ((my_key < rank_keys[rank_base + 17]) ? 1 : 0);
                rank_part2 += ((my_key < rank_keys[rank_base + 18]) ? 1 : 0);
                rank_part3 += ((my_key < rank_keys[rank_base + 19]) ? 1 : 0);
                rank_part0 += ((my_key < rank_keys[rank_base + 20]) ? 1 : 0);
                rank_part1 += ((my_key < rank_keys[rank_base + 21]) ? 1 : 0);
                rank_part2 += ((my_key < rank_keys[rank_base + 22]) ? 1 : 0);
                rank_part3 += ((my_key < rank_keys[rank_base + 23]) ? 1 : 0);
            }
            if (cand_total > 24) {
                rank_part0 += ((my_key < rank_keys[rank_base + 24]) ? 1 : 0);
                rank_part1 += ((my_key < rank_keys[rank_base + 25]) ? 1 : 0);
                rank_part2 += ((my_key < rank_keys[rank_base + 26]) ? 1 : 0);
                rank_part3 += ((my_key < rank_keys[rank_base + 27]) ? 1 : 0);
                rank_part0 += ((my_key < rank_keys[rank_base + 28]) ? 1 : 0);
                rank_part1 += ((my_key < rank_keys[rank_base + 29]) ? 1 : 0);
                rank_part2 += ((my_key < rank_keys[rank_base + 30]) ? 1 : 0);
                rank_part3 += ((my_key < rank_keys[rank_base + 31]) ? 1 : 0);
            }
            int my_rank = rank_part0 + rank_part1 + (rank_part2 + rank_part3);
            unsigned int rank_valid = 0;
            if (cand_total > lane) {
                rank_valid = (unsigned int)my_rank;
            }
            unsigned int rank_sum;
            asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(rank_sum) : "r"(rank_valid));
            if ((int)rank_sum != cand_total * (cand_total - 1) >> 1) {
                if (rank_keys[rank_base] == my_key) {
                    if (lane > 0) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 1] == my_key) {
                    if (lane > 1) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 2] == my_key) {
                    if (lane > 2) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 3] == my_key) {
                    if (lane > 3) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 4] == my_key) {
                    if (lane > 4) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 5] == my_key) {
                    if (lane > 5) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 6] == my_key) {
                    if (lane > 6) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 7] == my_key) {
                    if (lane > 7) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 8] == my_key) {
                    if (lane > 8) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 9] == my_key) {
                    if (lane > 9) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 10] == my_key) {
                    if (lane > 10) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 11] == my_key) {
                    if (lane > 11) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 12] == my_key) {
                    if (lane > 12) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 13] == my_key) {
                    if (lane > 13) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 14] == my_key) {
                    if (lane > 14) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 15] == my_key) {
                    if (lane > 15) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 16] == my_key) {
                    if (lane > 16) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 17] == my_key) {
                    if (lane > 17) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 18] == my_key) {
                    if (lane > 18) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 19] == my_key) {
                    if (lane > 19) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 20] == my_key) {
                    if (lane > 20) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 21] == my_key) {
                    if (lane > 21) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 22] == my_key) {
                    if (lane > 22) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 23] == my_key) {
                    if (lane > 23) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 24] == my_key) {
                    if (lane > 24) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 25] == my_key) {
                    if (lane > 25) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 26] == my_key) {
                    if (lane > 26) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 27] == my_key) {
                    if (lane > 27) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 28] == my_key) {
                    if (lane > 28) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 29] == my_key) {
                    if (lane > 29) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 30] == my_key) {
                    if (lane > 30) {
                        my_rank += 1;
                    }
                }
                if (rank_keys[rank_base + 31] == my_key) {
                    if (lane > 31) {
                        my_rank += 1;
                    }
                }
            }
            unsigned int win_ballot = __ballot_sync(0xFFFFFFFF, my_rank < 16);
            if (my_rank < 16) {
                int win_slot = (int)(unsigned int)__popc(win_ballot & lanemask_lt);
                if (win_slot < 16) {
                    winner_ids[winner_base + win_slot] = my_id;
                    winner_scores[winner_base + win_slot] = my_score;
                }
            }
            __syncwarp();
        }
        if (group_tid < TOP_K) {
            int early_id = winner_ids[winner_base + group_tid];
            int early_pair = group_row * TOP_K + group_tid;
            uint32_t _cluster_mapa_4;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_4) : "r"(selected_ids_addr + (unsigned int)(early_pair * 4)), "r"(0));
            uint32_t _mbar_peer_5;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_5) : "r"(ids_full_addr), "r"(0));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                :: "r"(_cluster_mapa_4), "r"(early_id), "r"(_mbar_peer_5) : "memory");
            uint32_t _cluster_mapa_6;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_6) : "r"(selected_ids_addr + (unsigned int)(early_pair * 4)), "r"(1));
            uint32_t _mbar_peer_7;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_7) : "r"(ids_full_addr), "r"(1));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                :: "r"(_cluster_mapa_6), "r"(early_id), "r"(_mbar_peer_7) : "memory");
            uint32_t _cluster_mapa_8;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_8) : "r"(selected_ids_addr + (unsigned int)(early_pair * 4)), "r"(2));
            uint32_t _mbar_peer_9;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_9) : "r"(ids_full_addr), "r"(2));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                :: "r"(_cluster_mapa_8), "r"(early_id), "r"(_mbar_peer_9) : "memory");
            uint32_t _cluster_mapa_10;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_10) : "r"(selected_ids_addr + (unsigned int)(early_pair * 4)), "r"(3));
            uint32_t _mbar_peer_11;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_11) : "r"(ids_full_addr), "r"(3));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                :: "r"(_cluster_mapa_10), "r"(early_id), "r"(_mbar_peer_11) : "memory");
            uint32_t _cluster_mapa_12;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_12) : "r"(selected_ids_addr + (unsigned int)(early_pair * 4)), "r"(4));
            uint32_t _mbar_peer_13;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_13) : "r"(ids_full_addr), "r"(4));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                :: "r"(_cluster_mapa_12), "r"(early_id), "r"(_mbar_peer_13) : "memory");
            uint32_t _cluster_mapa_14;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_14) : "r"(selected_ids_addr + (unsigned int)(early_pair * 4)), "r"(5));
            uint32_t _mbar_peer_15;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_15) : "r"(ids_full_addr), "r"(5));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                :: "r"(_cluster_mapa_14), "r"(early_id), "r"(_mbar_peer_15) : "memory");
            uint32_t _cluster_mapa_16;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_16) : "r"(selected_ids_addr + (unsigned int)(early_pair * 4)), "r"(6));
            uint32_t _mbar_peer_17;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_17) : "r"(ids_full_addr), "r"(6));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                :: "r"(_cluster_mapa_16), "r"(early_id), "r"(_mbar_peer_17) : "memory");
            uint32_t _cluster_mapa_18;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_18) : "r"(selected_ids_addr + (unsigned int)(early_pair * 4)), "r"(7));
            uint32_t _mbar_peer_19;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_19) : "r"(ids_full_addr), "r"(7));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                :: "r"(_cluster_mapa_18), "r"(early_id), "r"(_mbar_peer_19) : "memory");
            uint32_t _cluster_mapa_20;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_20) : "r"(selected_ids_addr + (unsigned int)(early_pair * 4)), "r"(8));
            uint32_t _mbar_peer_21;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_21) : "r"(ids_full_addr), "r"(8));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                :: "r"(_cluster_mapa_20), "r"(early_id), "r"(_mbar_peer_21) : "memory");
            uint32_t _cluster_mapa_22;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_22) : "r"(selected_ids_addr + (unsigned int)(early_pair * 4)), "r"(9));
            uint32_t _mbar_peer_23;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_23) : "r"(ids_full_addr), "r"(9));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                :: "r"(_cluster_mapa_22), "r"(early_id), "r"(_mbar_peer_23) : "memory");
            uint32_t _cluster_mapa_24;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_24) : "r"(selected_ids_addr + (unsigned int)(early_pair * 4)), "r"(10));
            uint32_t _mbar_peer_25;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_25) : "r"(ids_full_addr), "r"(10));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                :: "r"(_cluster_mapa_24), "r"(early_id), "r"(_mbar_peer_25) : "memory");
            uint32_t _cluster_mapa_26;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_26) : "r"(selected_ids_addr + (unsigned int)(early_pair * 4)), "r"(11));
            uint32_t _mbar_peer_27;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_27) : "r"(ids_full_addr), "r"(11));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                :: "r"(_cluster_mapa_26), "r"(early_id), "r"(_mbar_peer_27) : "memory");
            uint32_t _cluster_mapa_28;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_28) : "r"(selected_ids_addr + (unsigned int)(early_pair * 4)), "r"(12));
            uint32_t _mbar_peer_29;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_29) : "r"(ids_full_addr), "r"(12));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                :: "r"(_cluster_mapa_28), "r"(early_id), "r"(_mbar_peer_29) : "memory");
            uint32_t _cluster_mapa_30;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_30) : "r"(selected_ids_addr + (unsigned int)(early_pair * 4)), "r"(13));
            uint32_t _mbar_peer_31;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_31) : "r"(ids_full_addr), "r"(13));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                :: "r"(_cluster_mapa_30), "r"(early_id), "r"(_mbar_peer_31) : "memory");
            uint32_t _cluster_mapa_32;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_32) : "r"(selected_ids_addr + (unsigned int)(early_pair * 4)), "r"(14));
            uint32_t _mbar_peer_33;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_33) : "r"(ids_full_addr), "r"(14));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                :: "r"(_cluster_mapa_32), "r"(early_id), "r"(_mbar_peer_33) : "memory");
            uint32_t _cluster_mapa_34;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_34) : "r"(selected_ids_addr + (unsigned int)(early_pair * 4)), "r"(15));
            uint32_t _mbar_peer_35;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_35) : "r"(ids_full_addr), "r"(15));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                :: "r"(_cluster_mapa_34), "r"(early_id), "r"(_mbar_peer_35) : "memory");
            float selected_sum = 0.0f;
            #pragma unroll
            for (int route_sum = 0; route_sum < TOP_K; route_sum++) {
                selected_sum += winner_scores[winner_base + route_sum];
            }
            float selected_norm = 1.0f;
            if (selected_sum > 0.0f) {
                selected_norm = selected_sum;
            }
            unsigned long long output_index = (unsigned long long)group_row * (unsigned long long)TOP_K + (unsigned long long)group_tid;
            float _fdiv_36;
            asm volatile("div.full.f32 %0, %1, %2;" : "=f"(_fdiv_36) : "f"(winner_scores[winner_base + group_tid]), "f"(selected_norm));
            topk_weights[output_index] = _fdiv_36;
            topk_ids[output_index] = early_id;
        }
    } else if (group_tid < TOP_K) {
        int filler_pair = group_row * TOP_K + group_tid;
        uint32_t _cluster_mapa_37;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_37) : "r"(selected_ids_addr + (unsigned int)(filler_pair * 4)), "r"(0));
        uint32_t _mbar_peer_38;
        asm(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_mbar_peer_38) : "r"(ids_full_addr), "r"(0));
        asm volatile(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
            :: "r"(_cluster_mapa_37), "r"(filler_pair), "r"(_mbar_peer_38) : "memory");
        uint32_t _cluster_mapa_39;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_39) : "r"(selected_ids_addr + (unsigned int)(filler_pair * 4)), "r"(1));
        uint32_t _mbar_peer_40;
        asm(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_mbar_peer_40) : "r"(ids_full_addr), "r"(1));
        asm volatile(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
            :: "r"(_cluster_mapa_39), "r"(filler_pair), "r"(_mbar_peer_40) : "memory");
        uint32_t _cluster_mapa_41;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_41) : "r"(selected_ids_addr + (unsigned int)(filler_pair * 4)), "r"(2));
        uint32_t _mbar_peer_42;
        asm(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_mbar_peer_42) : "r"(ids_full_addr), "r"(2));
        asm volatile(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
            :: "r"(_cluster_mapa_41), "r"(filler_pair), "r"(_mbar_peer_42) : "memory");
        uint32_t _cluster_mapa_43;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_43) : "r"(selected_ids_addr + (unsigned int)(filler_pair * 4)), "r"(3));
        uint32_t _mbar_peer_44;
        asm(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_mbar_peer_44) : "r"(ids_full_addr), "r"(3));
        asm volatile(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
            :: "r"(_cluster_mapa_43), "r"(filler_pair), "r"(_mbar_peer_44) : "memory");
        uint32_t _cluster_mapa_45;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_45) : "r"(selected_ids_addr + (unsigned int)(filler_pair * 4)), "r"(4));
        uint32_t _mbar_peer_46;
        asm(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_mbar_peer_46) : "r"(ids_full_addr), "r"(4));
        asm volatile(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
            :: "r"(_cluster_mapa_45), "r"(filler_pair), "r"(_mbar_peer_46) : "memory");
        uint32_t _cluster_mapa_47;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_47) : "r"(selected_ids_addr + (unsigned int)(filler_pair * 4)), "r"(5));
        uint32_t _mbar_peer_48;
        asm(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_mbar_peer_48) : "r"(ids_full_addr), "r"(5));
        asm volatile(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
            :: "r"(_cluster_mapa_47), "r"(filler_pair), "r"(_mbar_peer_48) : "memory");
        uint32_t _cluster_mapa_49;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_49) : "r"(selected_ids_addr + (unsigned int)(filler_pair * 4)), "r"(6));
        uint32_t _mbar_peer_50;
        asm(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_mbar_peer_50) : "r"(ids_full_addr), "r"(6));
        asm volatile(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
            :: "r"(_cluster_mapa_49), "r"(filler_pair), "r"(_mbar_peer_50) : "memory");
        uint32_t _cluster_mapa_51;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_51) : "r"(selected_ids_addr + (unsigned int)(filler_pair * 4)), "r"(7));
        uint32_t _mbar_peer_52;
        asm(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_mbar_peer_52) : "r"(ids_full_addr), "r"(7));
        asm volatile(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
            :: "r"(_cluster_mapa_51), "r"(filler_pair), "r"(_mbar_peer_52) : "memory");
        uint32_t _cluster_mapa_53;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_53) : "r"(selected_ids_addr + (unsigned int)(filler_pair * 4)), "r"(8));
        uint32_t _mbar_peer_54;
        asm(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_mbar_peer_54) : "r"(ids_full_addr), "r"(8));
        asm volatile(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
            :: "r"(_cluster_mapa_53), "r"(filler_pair), "r"(_mbar_peer_54) : "memory");
        uint32_t _cluster_mapa_55;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_55) : "r"(selected_ids_addr + (unsigned int)(filler_pair * 4)), "r"(9));
        uint32_t _mbar_peer_56;
        asm(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_mbar_peer_56) : "r"(ids_full_addr), "r"(9));
        asm volatile(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
            :: "r"(_cluster_mapa_55), "r"(filler_pair), "r"(_mbar_peer_56) : "memory");
        uint32_t _cluster_mapa_57;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_57) : "r"(selected_ids_addr + (unsigned int)(filler_pair * 4)), "r"(10));
        uint32_t _mbar_peer_58;
        asm(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_mbar_peer_58) : "r"(ids_full_addr), "r"(10));
        asm volatile(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
            :: "r"(_cluster_mapa_57), "r"(filler_pair), "r"(_mbar_peer_58) : "memory");
        uint32_t _cluster_mapa_59;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_59) : "r"(selected_ids_addr + (unsigned int)(filler_pair * 4)), "r"(11));
        uint32_t _mbar_peer_60;
        asm(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_mbar_peer_60) : "r"(ids_full_addr), "r"(11));
        asm volatile(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
            :: "r"(_cluster_mapa_59), "r"(filler_pair), "r"(_mbar_peer_60) : "memory");
        uint32_t _cluster_mapa_61;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_61) : "r"(selected_ids_addr + (unsigned int)(filler_pair * 4)), "r"(12));
        uint32_t _mbar_peer_62;
        asm(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_mbar_peer_62) : "r"(ids_full_addr), "r"(12));
        asm volatile(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
            :: "r"(_cluster_mapa_61), "r"(filler_pair), "r"(_mbar_peer_62) : "memory");
        uint32_t _cluster_mapa_63;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_63) : "r"(selected_ids_addr + (unsigned int)(filler_pair * 4)), "r"(13));
        uint32_t _mbar_peer_64;
        asm(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_mbar_peer_64) : "r"(ids_full_addr), "r"(13));
        asm volatile(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
            :: "r"(_cluster_mapa_63), "r"(filler_pair), "r"(_mbar_peer_64) : "memory");
        uint32_t _cluster_mapa_65;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_65) : "r"(selected_ids_addr + (unsigned int)(filler_pair * 4)), "r"(14));
        uint32_t _mbar_peer_66;
        asm(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_mbar_peer_66) : "r"(ids_full_addr), "r"(14));
        asm volatile(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
            :: "r"(_cluster_mapa_65), "r"(filler_pair), "r"(_mbar_peer_66) : "memory");
        uint32_t _cluster_mapa_67;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_67) : "r"(selected_ids_addr + (unsigned int)(filler_pair * 4)), "r"(15));
        uint32_t _mbar_peer_68;
        asm(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_mbar_peer_68) : "r"(ids_full_addr), "r"(15));
        asm volatile(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
            :: "r"(_cluster_mapa_67), "r"(filler_pair), "r"(_mbar_peer_68) : "memory");
    }
    __syncthreads();
    mbarrier_wait(ids_full_addr, 0);
    int gathered[2];
    int pair_bound = M * TOP_K;
    #pragma unroll
    for (int gather_slot = 0; gather_slot < 2; gather_slot++) {
        int gather_pair = gather_slot * 448 + tid;
        gathered[gather_slot] = 0;
        if (gather_pair < pair_bound) {
            gathered[gather_slot] = selected_ids[gather_pair];
        }
    }
    #pragma unroll
    for (int bit_slot = 0; bit_slot < 1; bit_slot++) {
        int bit_pair = bit_slot * 448 + tid;
        unsigned int bit_value = (unsigned int)0;
        int bit_index = lane;
        if (bit_pair < pair_bound) {
            bit_value = (unsigned int)1 << (unsigned int)(bit_pair >> 4);
            bit_index = gathered[bit_slot];
        }
        unsigned int bit_old = atomicAdd(&row_bits[bit_index], bit_value);
        int bit_level = (int)(unsigned int)__popc(bit_old);
        unsigned int level_value = (unsigned int)0;
        int level_index = lane;
        if (bit_pair < pair_bound && (bit_level & BLOCK_MASK) == 0) {
            level_value = (unsigned int)1 << (unsigned int)(bit_index & 31);
            level_index = (bit_level >> BLOCK_SHIFT) * 28 + (bit_index >> 5);
        }
        atomicAdd(&level_masks[level_index], level_value);
    }
    if (448 < pair_bound) {
        #pragma unroll
        for (int tail_slot = 1; tail_slot < 2; tail_slot++) {
            int tail_pair = tail_slot * 448 + tid;
            unsigned int tail_value = (unsigned int)0;
            int tail_index = lane;
            if (tail_pair < pair_bound) {
                tail_value = (unsigned int)1 << (unsigned int)(tail_pair >> 4);
                tail_index = gathered[tail_slot];
            }
            unsigned int tail_old = atomicAdd(&row_bits[tail_index], tail_value);
            int tail_level = (int)(unsigned int)__popc(tail_old);
            unsigned int tail_level_value = (unsigned int)0;
            int tail_level_index = lane;
            if (tail_pair < pair_bound && (tail_level & BLOCK_MASK) == 0) {
                tail_level_value = (unsigned int)1 << (unsigned int)(tail_index & 31);
                tail_level_index = (tail_level >> BLOCK_SHIFT) * 28 + (tail_index >> 5);
            }
            atomicAdd(&level_masks[tail_level_index], tail_level_value);
        }
    }
    __syncthreads();
    if (warp < 4) {
        int emit_local = tid & 31;
        int emit_slot = emit_local & TOP_K - 1;
        int emit_sub = tid >> 5;
        int emit_row = cta_rank + (emit_local >> 4) * 16;
        int emit_src_row = emit_row;
        if (emit_row >= M) {
            emit_src_row = cta_rank;
        }
        int emit_expert = selected_ids[emit_src_row * TOP_K + emit_slot];
        unsigned int emit_bits = row_bits[emit_expert];
        unsigned int emit_below = ((unsigned int)1 << (unsigned int)emit_src_row) - (unsigned int)1;
        int emit_rank = (int)(unsigned int)__popc(emit_bits & emit_below);
        int emit_count = (int)(unsigned int)__popc(emit_bits);
        unsigned int level_words[4];
        int mask_lane_blocks = 0;
        #pragma unroll
        for (int level_load = 0; level_load < MASK_LEVELS; level_load++) {
            level_words[level_load] = (unsigned int)0;
            if (lane < 28) {
                level_words[level_load] = level_masks[level_load * 28 + lane];
            }
            mask_lane_blocks += (int)(unsigned int)__popc(level_words[level_load]);
        }
        int mask_inclusive = mask_lane_blocks;
        int mask_peer_4_0 = __shfl_up_sync(0xFFFFFFFF, mask_inclusive, 1, 32);
        if (lane >= 1) {
            mask_inclusive += mask_peer_4_0;
        }
        int mask_peer_4_1 = __shfl_up_sync(0xFFFFFFFF, mask_inclusive, 2, 32);
        if (lane >= 2) {
            mask_inclusive += mask_peer_4_1;
        }
        int mask_peer_4_2 = __shfl_up_sync(0xFFFFFFFF, mask_inclusive, 4, 32);
        if (lane >= 4) {
            mask_inclusive += mask_peer_4_2;
        }
        int mask_peer_4_3 = __shfl_up_sync(0xFFFFFFFF, mask_inclusive, 8, 32);
        if (lane >= 8) {
            mask_inclusive += mask_peer_4_3;
        }
        int mask_peer_4_4 = __shfl_up_sync(0xFFFFFFFF, mask_inclusive, 16, 32);
        if (lane >= 16) {
            mask_inclusive += mask_peer_4_4;
        }
        int mask_exclusive = mask_inclusive - mask_lane_blocks;
        int mask_total = __shfl_sync(0xFFFFFFFF, mask_inclusive, 31);
        int emit_word = emit_expert >> 5;
        unsigned int emit_low = ((unsigned int)1 << (unsigned int)(emit_expert & 31)) - (unsigned int)1;
        int emit_blocks = __shfl_sync(0xFFFFFFFF, mask_exclusive, emit_word);
        #pragma unroll
        for (int level_emit = 0; level_emit < MASK_LEVELS; level_emit++) {
            unsigned int _shfl_sync_69 = __shfl_sync(0xFFFFFFFF, level_words[level_emit], emit_word);
            emit_blocks += (int)(unsigned int)__popc(_shfl_sync_69 & emit_low);
        }
        int emit_start = emit_blocks << BLOCK_SHIFT;
        if (emit_row < M) {
            if (emit_sub == 0) {
                int emit_grouped = emit_start + emit_rank;
                sorted_token_ids[emit_grouped] = emit_row * TOP_K + emit_slot;
                if ((emit_rank & BLOCK_MASK) == 0) {
                    expert_ids[emit_grouped >> BLOCK_SHIFT] = emit_expert;
                }
            }
            if (emit_rank == 0) {
                int emit_padded = emit_count + BLOCK_MASK & ~BLOCK_MASK;
                #pragma unroll 1
                for (int emit_pad = emit_count + emit_sub; emit_pad < emit_padded; emit_pad += 4) {
                    sorted_token_ids[emit_start + emit_pad] = M * TOP_K;
                }
            }
        }
        int slice_expert = cta_rank * 56;
        if (tid < 56) {
            slice_expert += tid;
        }
        int slice_word = slice_expert >> 5;
        unsigned int slice_low = ((unsigned int)1 << (unsigned int)(slice_expert & 31)) - (unsigned int)1;
        int slice_blocks = __shfl_sync(0xFFFFFFFF, mask_exclusive, slice_word);
        #pragma unroll
        for (int level_slice = 0; level_slice < MASK_LEVELS; level_slice++) {
            unsigned int _shfl_sync_70 = __shfl_sync(0xFFFFFFFF, level_words[level_slice], slice_word);
            slice_blocks += (int)(unsigned int)__popc(_shfl_sync_70 & slice_low);
        }
        if (tid < 56) {
            int slice_count = (int)(unsigned int)__popc(row_bits[slice_expert]);
            expert_counts[slice_expert] = slice_count;
            expert_scatter_offsets[slice_expert] = slice_count;
            expert_offsets[slice_expert] = slice_blocks << BLOCK_SHIFT;
        }
        if (cta_rank == 0) {
            if (tid == 127) {
                expert_offsets[NUM_EXPERTS] = mask_total << BLOCK_SHIFT;
                num_tokens_post_padded[0] = mask_total << BLOCK_SHIFT;
            }
        }
    }
}

} // extern "C"
