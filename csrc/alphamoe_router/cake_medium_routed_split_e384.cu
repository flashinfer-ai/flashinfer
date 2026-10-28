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
// clang-format off
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
#define SMEM_STAGE_OFF 4224
#define SMEM_STAGE_STAGE_BYTES 2048
#define SMEM_STAGE_STRIDE 2048
#define SMEM_LOCAL_OFFSETS_OFF 0
#define SMEM_LOCAL_OFFSETS_STAGE_BYTES 4100
#define SMEM_LOCAL_OFFSETS_STRIDE 4100
#define SMEM_SCAN_VALUES_OFF 4100
#define SMEM_SCAN_VALUES_STAGE_BYTES 16
#define SMEM_SCAN_VALUES_STRIDE 16
#define SMEM_PRIVATE_COUNTS_OFF 2052
#define SMEM_PRIVATE_COUNTS_STAGE_BYTES 2048
#define SMEM_PRIVATE_COUNTS_STRIDE 2048
#define SMEM_TOTAL 6272
#define THREADS 128
#define NUM_WARPS 4
#define MAX_EXPERTS 512
#define MAX_TOP_K 16
#define MAX_BLOCK_M 16
#define PUBLIC_SHARED_SOFTMAX 0

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

extern "C" {

__global__ __launch_bounds__(128) void
kernel_alpha_moe_fused_router_medium_e512_threshold_split_start_rs12(float* __restrict__ logits, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M, int E, int top_k, int block_m, int has_shared_expert)
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
    unsigned int* stage = reinterpret_cast<unsigned int*>(smem_raw + 4224);
    const int stage_addr = smem + 4224;
    int* local_offsets = reinterpret_cast<int*>(smem_raw + 0);
    const int local_offsets_addr = smem + 0;
    int* scan_values = reinterpret_cast<int*>(smem_raw + 4100);
    const int scan_values_addr = smem + 4100;
    int* private_counts = reinterpret_cast<int*>(smem_raw + 2052);
    const int private_counts_addr = smem + 2052;

    // === Task calls (declaration order) ===
    int global_thread = bid * THREADS + tid;
    int private_raw_hist = 0;
    if (E <= 512 && has_shared_expert == 0 && M <= 512 && M <= num_bids * NUM_WARPS) {
        private_raw_hist = 1;
    }
    if (bid == 0) {
        for (int zero_cell = tid; zero_cell < E + 1; zero_cell += THREADS) {
            expert_offsets[zero_cell] = 0;
            if (zero_cell < E) {
                if (private_raw_hist == 0) {
                    expert_counts[zero_cell] = 0;
                }
                expert_scatter_offsets[zero_cell] = 0;
            }
        }
        if (tid == 0) {
            num_tokens_post_padded[0] = 0;
        }
    }
    float row_values[12];
    int preload_single_row = 0;
    if (M <= num_bids * NUM_WARPS) {
        preload_single_row = 1;
        int preload_token = bid * NUM_WARPS + warp;
        if (preload_token < M) {
            unsigned long long preload_row_base = (unsigned long long)preload_token * (unsigned long long)E;
            if (E == 384) {
                unsigned long long preload_lane_base = preload_row_base + (unsigned long long)lane;
                #pragma unroll
                for (int preload_slot = 0; preload_slot < 12; preload_slot++) {
                    row_values[preload_slot] = logits[preload_lane_base + (unsigned long long)(preload_slot * 32)];
                }
            } else {
                #pragma unroll
                for (int preload_masked_slot = 0; preload_masked_slot < 12; preload_masked_slot++) {
                    int preload_expert = lane + preload_masked_slot * 32;
                    row_values[preload_masked_slot] = -CUDART_INF_F;
                    if (preload_expert < E) {
                        row_values[preload_masked_slot] = logits[preload_row_base + (unsigned long long)preload_expert];
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int table_clear = 0; table_clear < 4; table_clear++) {
        private_counts[tid * 4 + table_clear] = 0;
    }
    __syncthreads();
    if (private_raw_hist == 0) {
        cooperative_groups::this_grid().sync();
    }
    int routed_experts = E - has_shared_expert;
    int routed_top_k = top_k - has_shared_expert;
    for (int token = bid * NUM_WARPS + warp; token < M; token += num_bids * NUM_WARPS) {
        unsigned long long row_base = 0;
        if (preload_single_row == 0) {
            row_base = (unsigned long long)token * (unsigned long long)E;
            if (MAX_EXPERTS == 512 && PUBLIC_SHARED_SOFTMAX == 0 && E == 384 && has_shared_expert == 0) {
                unsigned long long lane_row_base = row_base + (unsigned long long)lane;
                #pragma unroll
                for (int expert_slot_full = 0; expert_slot_full < 12; expert_slot_full++) {
                    row_values[expert_slot_full] = logits[lane_row_base + (unsigned long long)(expert_slot_full * 32)];
                }
            } else {
                #pragma unroll
                for (int expert_slot_load = 0; expert_slot_load < 12; expert_slot_load++) {
                    int expert_load = lane + expert_slot_load * 32;
                    row_values[expert_slot_load] = -CUDART_INF_F;
                    if (expert_load < routed_experts) {
                        row_values[expert_slot_load] = logits[row_base + (unsigned long long)expert_load];
                    }
                }
            }
        }
        unsigned int nan_bits = 0;
        float _max_5;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_5) : "f"(row_values[0]), "f"(row_values[1]));
        float _max_4;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_4) : "f"(_max_5), "f"(row_values[2]));
        float _max_7;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_7) : "f"(row_values[3]), "f"(row_values[4]));
        float _max_6;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_6) : "f"(_max_7), "f"(row_values[5]));
        float _max_3;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_3) : "f"(_max_4), "f"(_max_6));
        float _max_9;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_9) : "f"(row_values[6]), "f"(row_values[7]));
        float _max_8;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_8) : "f"(_max_9), "f"(row_values[8]));
        float _max_2;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_2) : "f"(_max_3), "f"(_max_8));
        float _max_11;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_11) : "f"(row_values[9]), "f"(row_values[10]));
        float _max_10;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_10) : "f"(_max_11), "f"(row_values[11]));
        float _max_1;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_1) : "f"(_max_2), "f"(_max_10));
        float _max_0;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_0) : "f"(_max_1), "f"(-CUDART_INF_F));
        float _bit_cast_src_12 = _max_0;
        nan_bits = reinterpret_cast<unsigned int*>(&_bit_cast_src_12)[0];
        int _warp_vote_13 = __any_sync(0xFFFFFFFF, (nan_bits & 2147483647) > 2139095040);
        if (_warp_vote_13 != 0) {
            #pragma unroll
            for (int nan_slot = 0; nan_slot < 12; nan_slot++) {
                row_values[nan_slot] = fminf(row_values[nan_slot], CUDART_INF_F);
            }
        }
        unsigned long long output_base = (unsigned long long)token * (unsigned long long)top_k;
        float selected_logit = -CUDART_INF_F;
        int selected_expert = MAX_EXPERTS;
        unsigned int lanes_below = (1 << (unsigned int)lane) - 1;
        int route_by_route = 1;
        float lane_max_0 = fmaxf(row_values[0], row_values[1]);
        float lane_max_1 = fmaxf(row_values[2], row_values[3]);
        float lane_max_2 = fmaxf(row_values[4], row_values[5]);
        float lane_max_3 = fmaxf(row_values[6], row_values[7]);
        #pragma unroll
        for (int lane_group = 1; lane_group < 1; lane_group++) {
            lane_max_0 = fmaxf(lane_max_0, fmaxf(row_values[lane_group * 8], row_values[lane_group * 8 + 1]));
            lane_max_1 = fmaxf(lane_max_1, fmaxf(row_values[lane_group * 8 + 2], row_values[lane_group * 8 + 3]));
            lane_max_2 = fmaxf(lane_max_2, fmaxf(row_values[lane_group * 8 + 4], row_values[lane_group * 8 + 5]));
            lane_max_3 = fmaxf(lane_max_3, fmaxf(row_values[lane_group * 8 + 6], row_values[lane_group * 8 + 7]));
        }
        lane_max_0 = fmaxf(lane_max_0, row_values[8]);
        lane_max_1 = fmaxf(lane_max_1, row_values[9]);
        lane_max_2 = fmaxf(lane_max_2, row_values[10]);
        lane_max_3 = fmaxf(lane_max_3, row_values[11]);
        unsigned int lane_bits = 0;
        float _bit_cast_src_14 = fmaxf(fmaxf(lane_max_0, lane_max_1), fmaxf(lane_max_2, lane_max_3));
        lane_bits = reinterpret_cast<unsigned int*>(&_bit_cast_src_14)[0];
        if ((lane_bits & 2139095040) == 0) {
            lane_bits = 0;
        }
        unsigned int lane_key = lane_bits ^ 2147483648u;
        if ((lane_bits & 2147483648u) != 0) {
            lane_key = lane_bits ^ 4294967295u;
        }
        int stage_base = warp * 128;
        __syncwarp();
        stage[stage_base + 96 + lane] = lane_key;
        __syncwarp();
        int lanes_above = 0;
        unsigned int lane_keys[16];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[0])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[(0) + 3]))
            : "r"(stage_addr + (unsigned int)((stage_base + 96) * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[4])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[(4) + 3]))
            : "r"(stage_addr + (unsigned int)((stage_base + 96 + 4) * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[8])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[(8) + 3]))
            : "r"(stage_addr + (unsigned int)((stage_base + 96 + 8) * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[12])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[(12) + 3]))
            : "r"(stage_addr + (unsigned int)((stage_base + 96 + 12) * 4)));
        if (lane_keys[0] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[1] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[2] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[3] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[4] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[5] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[6] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[7] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[8] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[9] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[10] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[11] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[12] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[13] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[14] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[15] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        unsigned int lane_keys_0[16];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[(0) + 3]))
            : "r"(stage_addr + (unsigned int)((stage_base + 96 + 16) * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[(4) + 3]))
            : "r"(stage_addr + (unsigned int)((stage_base + 96 + 20) * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[(8) + 3]))
            : "r"(stage_addr + (unsigned int)((stage_base + 96 + 24) * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[(12) + 3]))
            : "r"(stage_addr + (unsigned int)((stage_base + 96 + 28) * 4)));
        if (lane_keys_0[0] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[1] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[2] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[3] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[4] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[5] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[6] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[7] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[8] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[9] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[10] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[11] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[12] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[13] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[14] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[15] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        unsigned int bound_vote = 4294967295;
        if (lanes_above < routed_top_k) {
            bound_vote = lane_key;
        }
        unsigned int bound_key;
        asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(bound_key) : "r"(bound_vote));
        unsigned int bound_bits = bound_key ^ 2147483648u;
        if ((bound_key & 2147483648u) == 0) {
            bound_bits = bound_key ^ 4294967295u;
        }
        float bound_value = 0.0f;
        bound_value = reinterpret_cast<float*>(&bound_bits)[0];
        unsigned int slot_lanes[12];
        int candidates = 0;
        unsigned int _warp_vote_15 = __ballot_sync(0xFFFFFFFF, row_values[0] >= bound_value);
        slot_lanes[0] = _warp_vote_15;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[0]);
        unsigned int _warp_vote_16 = __ballot_sync(0xFFFFFFFF, row_values[1] >= bound_value);
        slot_lanes[1] = _warp_vote_16;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[1]);
        unsigned int _warp_vote_17 = __ballot_sync(0xFFFFFFFF, row_values[2] >= bound_value);
        slot_lanes[2] = _warp_vote_17;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[2]);
        unsigned int _warp_vote_18 = __ballot_sync(0xFFFFFFFF, row_values[3] >= bound_value);
        slot_lanes[3] = _warp_vote_18;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[3]);
        unsigned int _warp_vote_19 = __ballot_sync(0xFFFFFFFF, row_values[4] >= bound_value);
        slot_lanes[4] = _warp_vote_19;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[4]);
        unsigned int _warp_vote_20 = __ballot_sync(0xFFFFFFFF, row_values[5] >= bound_value);
        slot_lanes[5] = _warp_vote_20;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[5]);
        unsigned int _warp_vote_21 = __ballot_sync(0xFFFFFFFF, row_values[6] >= bound_value);
        slot_lanes[6] = _warp_vote_21;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[6]);
        unsigned int _warp_vote_22 = __ballot_sync(0xFFFFFFFF, row_values[7] >= bound_value);
        slot_lanes[7] = _warp_vote_22;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[7]);
        unsigned int _warp_vote_23 = __ballot_sync(0xFFFFFFFF, row_values[8] >= bound_value);
        slot_lanes[8] = _warp_vote_23;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[8]);
        unsigned int _warp_vote_24 = __ballot_sync(0xFFFFFFFF, row_values[9] >= bound_value);
        slot_lanes[9] = _warp_vote_24;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[9]);
        unsigned int _warp_vote_25 = __ballot_sync(0xFFFFFFFF, row_values[10] >= bound_value);
        slot_lanes[10] = _warp_vote_25;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[10]);
        unsigned int _warp_vote_26 = __ballot_sync(0xFFFFFFFF, row_values[11] >= bound_value);
        slot_lanes[11] = _warp_vote_26;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[11]);
        if (candidates <= 32 && bound_key != 2147483648u && bound_key > 8388607) {
            route_by_route = 0;
            int candidate_position = 0;
            int stage_rows[12];
            unsigned int stage_word = 0;
            stage_rows[0] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[0] & lanes_below);
            if ((slot_lanes[0] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_27 = row_values[0];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_27)[0];
                stage[stage_rows[0]] = stage_word;
                stage[stage_rows[0] + 32] = (unsigned int)lane;
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[0]);
            stage_rows[1] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[1] & lanes_below);
            if ((slot_lanes[1] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_28 = row_values[1];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_28)[0];
                stage[stage_rows[1]] = stage_word;
                stage[stage_rows[1] + 32] = (unsigned int)(lane + 32);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[1]);
            stage_rows[2] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[2] & lanes_below);
            if ((slot_lanes[2] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_29 = row_values[2];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_29)[0];
                stage[stage_rows[2]] = stage_word;
                stage[stage_rows[2] + 32] = (unsigned int)(lane + 64);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[2]);
            stage_rows[3] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[3] & lanes_below);
            if ((slot_lanes[3] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_30 = row_values[3];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_30)[0];
                stage[stage_rows[3]] = stage_word;
                stage[stage_rows[3] + 32] = (unsigned int)(lane + 96);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[3]);
            stage_rows[4] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[4] & lanes_below);
            if ((slot_lanes[4] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_31 = row_values[4];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_31)[0];
                stage[stage_rows[4]] = stage_word;
                stage[stage_rows[4] + 32] = (unsigned int)(lane + 128);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[4]);
            stage_rows[5] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[5] & lanes_below);
            if ((slot_lanes[5] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_32 = row_values[5];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_32)[0];
                stage[stage_rows[5]] = stage_word;
                stage[stage_rows[5] + 32] = (unsigned int)(lane + 160);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[5]);
            stage_rows[6] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[6] & lanes_below);
            if ((slot_lanes[6] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_33 = row_values[6];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_33)[0];
                stage[stage_rows[6]] = stage_word;
                stage[stage_rows[6] + 32] = (unsigned int)(lane + 192);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[6]);
            stage_rows[7] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[7] & lanes_below);
            if ((slot_lanes[7] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_34 = row_values[7];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_34)[0];
                stage[stage_rows[7]] = stage_word;
                stage[stage_rows[7] + 32] = (unsigned int)(lane + 224);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[7]);
            stage_rows[8] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[8] & lanes_below);
            if ((slot_lanes[8] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_35 = row_values[8];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_35)[0];
                stage[stage_rows[8]] = stage_word;
                stage[stage_rows[8] + 32] = (unsigned int)(lane + 256);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[8]);
            stage_rows[9] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[9] & lanes_below);
            if ((slot_lanes[9] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_36 = row_values[9];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_36)[0];
                stage[stage_rows[9]] = stage_word;
                stage[stage_rows[9] + 32] = (unsigned int)(lane + 288);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[9]);
            stage_rows[10] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[10] & lanes_below);
            if ((slot_lanes[10] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_37 = row_values[10];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_37)[0];
                stage[stage_rows[10]] = stage_word;
                stage[stage_rows[10] + 32] = (unsigned int)(lane + 320);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[10]);
            stage_rows[11] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[11] & lanes_below);
            if ((slot_lanes[11] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_38 = row_values[11];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_38)[0];
                stage[stage_rows[11]] = stage_word;
                stage[stage_rows[11] + 32] = (unsigned int)(lane + 352);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[11]);
            __syncwarp();
            unsigned int candidate_bits = 0;
            unsigned int candidate_id = 0;
            unsigned int candidate_key = 0;
            if (candidates > lane) {
                candidate_bits = stage[stage_base + lane];
                candidate_id = stage[stage_base + 32 + lane];
                unsigned int candidate_flat = candidate_bits;
                if ((candidate_bits & 2139095040) == 0) {
                    candidate_flat = 0;
                }
                candidate_key = candidate_flat ^ 2147483648u;
                if ((candidate_flat & 2147483648u) != 0) {
                    candidate_key = candidate_flat ^ 4294967295u;
                }
            }
            stage[stage_base + 96 + lane] = candidate_key;
            __syncwarp();
            int candidate_rank = 0;
            unsigned int candidate_keys[16];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[0])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[(0) + 3]))
                : "r"(stage_addr + (unsigned int)((stage_base + 96) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[4])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[(4) + 3]))
                : "r"(stage_addr + (unsigned int)((stage_base + 96 + 4) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[8])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[(8) + 3]))
                : "r"(stage_addr + (unsigned int)((stage_base + 96 + 8) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[12])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[(12) + 3]))
                : "r"(stage_addr + (unsigned int)((stage_base + 96 + 12) * 4)));
            if (candidate_keys[0] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[1] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[2] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[3] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[4] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[5] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[6] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[7] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[8] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[9] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[10] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[11] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[12] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[13] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[14] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[15] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidates > 16) {
                unsigned int candidate_keys_0[16];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[(0) + 3]))
                    : "r"(stage_addr + (unsigned int)((stage_base + 96 + 16) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[(4) + 3]))
                    : "r"(stage_addr + (unsigned int)((stage_base + 96 + 20) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[(8) + 3]))
                    : "r"(stage_addr + (unsigned int)((stage_base + 96 + 24) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[(12) + 3]))
                    : "r"(stage_addr + (unsigned int)((stage_base + 96 + 28) * 4)));
                if (candidate_keys_0[0] > candidate_key) {
                    candidate_rank = candidate_rank + 1;
                }
                if (candidate_keys_0[1] > candidate_key) {
                    candidate_rank = candidate_rank + 1;
                }
                if (candidate_keys_0[2] > candidate_key) {
                    candidate_rank = candidate_rank + 1;
                }
                if (candidate_keys_0[3] > candidate_key) {
                    candidate_rank = candidate_rank + 1;
                }
                if (candidate_keys_0[4] > candidate_key) {
                    candidate_rank = candidate_rank + 1;
                }
                if (candidate_keys_0[5] > candidate_key) {
                    candidate_rank = candidate_rank + 1;
                }
                if (candidate_keys_0[6] > candidate_key) {
                    candidate_rank = candidate_rank + 1;
                }
                if (candidate_keys_0[7] > candidate_key) {
                    candidate_rank = candidate_rank + 1;
                }
                if (candidate_keys_0[8] > candidate_key) {
                    candidate_rank = candidate_rank + 1;
                }
                if (candidate_keys_0[9] > candidate_key) {
                    candidate_rank = candidate_rank + 1;
                }
                if (candidate_keys_0[10] > candidate_key) {
                    candidate_rank = candidate_rank + 1;
                }
                if (candidate_keys_0[11] > candidate_key) {
                    candidate_rank = candidate_rank + 1;
                }
                if (candidate_keys_0[12] > candidate_key) {
                    candidate_rank = candidate_rank + 1;
                }
                if (candidate_keys_0[13] > candidate_key) {
                    candidate_rank = candidate_rank + 1;
                }
                if (candidate_keys_0[14] > candidate_key) {
                    candidate_rank = candidate_rank + 1;
                }
                if (candidate_keys_0[15] > candidate_key) {
                    candidate_rank = candidate_rank + 1;
                }
            }
            unsigned int tied_lanes = __ballot_sync(0xFFFFFFFF, candidates > lane);
            int rank_bit_set = candidate_rank & 1;
            unsigned int rank_bit_lanes = __ballot_sync(0xFFFFFFFF, rank_bit_set != 0);
            if (rank_bit_set == 0) {
                rank_bit_lanes = rank_bit_lanes ^ 4294967295u;
            }
            tied_lanes = tied_lanes & rank_bit_lanes;
            int rank_bit_set_0 = candidate_rank >> 1 & 1;
            unsigned int rank_bit_lanes_1 = __ballot_sync(0xFFFFFFFF, rank_bit_set_0 != 0);
            if (rank_bit_set_0 == 0) {
                rank_bit_lanes_1 = rank_bit_lanes_1 ^ 4294967295u;
            }
            tied_lanes = tied_lanes & rank_bit_lanes_1;
            int rank_bit_set_2 = candidate_rank >> 2 & 1;
            unsigned int rank_bit_lanes_3 = __ballot_sync(0xFFFFFFFF, rank_bit_set_2 != 0);
            if (rank_bit_set_2 == 0) {
                rank_bit_lanes_3 = rank_bit_lanes_3 ^ 4294967295u;
            }
            tied_lanes = tied_lanes & rank_bit_lanes_3;
            int rank_bit_set_4 = candidate_rank >> 3 & 1;
            unsigned int rank_bit_lanes_5 = __ballot_sync(0xFFFFFFFF, rank_bit_set_4 != 0);
            if (rank_bit_set_4 == 0) {
                rank_bit_lanes_5 = rank_bit_lanes_5 ^ 4294967295u;
            }
            tied_lanes = tied_lanes & rank_bit_lanes_5;
            int rank_bit_set_6 = candidate_rank >> 4 & 1;
            unsigned int rank_bit_lanes_7 = __ballot_sync(0xFFFFFFFF, rank_bit_set_6 != 0);
            if (rank_bit_set_6 == 0) {
                rank_bit_lanes_7 = rank_bit_lanes_7 ^ 4294967295u;
            }
            tied_lanes = tied_lanes & rank_bit_lanes_7;
            candidate_rank = candidate_rank + (int)(unsigned int)__popc(tied_lanes & lanes_below);
            if (candidates > lane && candidate_rank < routed_top_k) {
                stage[stage_base + 64 + candidate_rank] = candidate_bits;
                stage[stage_base + 80 + candidate_rank] = candidate_id;
            }
            __syncwarp();
            if (routed_top_k > lane) {
                unsigned int _bit_cast_src_39 = stage[stage_base + 64 + lane];
                selected_logit = reinterpret_cast<float*>(&_bit_cast_src_39)[0];
                selected_expert = (int)stage[stage_base + 80 + lane];
            }
        }
        #pragma unroll 1
        for (int route = 0; route < routed_top_k * route_by_route; route++) {
            float local_max_0 = -CUDART_INF_F;
            float local_max_1 = -CUDART_INF_F;
            float local_max_2 = -CUDART_INF_F;
            float local_max_3 = -CUDART_INF_F;
            #pragma unroll
            for (int scan_group = 0; scan_group < 3; scan_group++) {
                local_max_0 = fmaxf(local_max_0, row_values[scan_group * 4]);
                local_max_1 = fmaxf(local_max_1, row_values[scan_group * 4 + 1]);
                local_max_2 = fmaxf(local_max_2, row_values[scan_group * 4 + 2]);
                local_max_3 = fmaxf(local_max_3, row_values[scan_group * 4 + 3]);
            }
            float local_max_low = fmaxf(local_max_0, local_max_1);
            float local_max_high = fmaxf(local_max_2, local_max_3);
            float local_value = fmaxf(local_max_low, local_max_high);
            unsigned int local_bits = 0;
            local_bits = reinterpret_cast<unsigned int*>(&local_value)[0];
            unsigned int local_key = local_bits ^ 2147483648u;
            if ((local_bits & 2147483648u) != 0) {
                local_key = local_bits ^ 4294967295u;
            }
            unsigned int best_key;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(best_key) : "r"(local_key));
            unsigned int best_bits = best_key ^ 2147483648u;
            if ((best_key & 2147483648u) == 0) {
                best_bits = best_key ^ 4294967295u;
            }
            float best_value = 0.0f;
            best_value = reinterpret_cast<float*>(&best_bits)[0];
            unsigned int tied_index = MAX_EXPERTS;
            if (best_value != -CUDART_INF_F) {
                #pragma unroll
                for (int expert_slot_finite_tie = 0; expert_slot_finite_tie < 12; expert_slot_finite_tie++) {
                    if (row_values[11 - expert_slot_finite_tie] == best_value) {
                        int expert_finite_tie = lane + (11 - expert_slot_finite_tie) * 32;
                        tied_index = (unsigned int)expert_finite_tie;
                    }
                }
            } else {
                #pragma unroll
                for (int expert_slot_tie = 0; expert_slot_tie < 12; expert_slot_tie++) {
                    int expert_tie = lane + (11 - expert_slot_tie) * 32;
                    if (expert_tie < routed_experts) {
                        if (row_values[11 - expert_slot_tie] == best_value) {
                            tied_index = (unsigned int)expert_tie;
                        }
                    }
                }
            }
            unsigned int _warp_redux_40;
            asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_40) : "r"(tied_index));
            int best_index = (int)_warp_redux_40;
            if (lane == route) {
                selected_logit = best_value;
                selected_expert = best_index;
            }
            #pragma unroll
            for (int expert_slot_remove = 0; expert_slot_remove < 12; expert_slot_remove++) {
                if (lane + expert_slot_remove * 32 == best_index) {
                    row_values[expert_slot_remove] = -CUDART_INF_F;
                }
            }
        }
        if (has_shared_expert != 0) {
            if (lane == routed_top_k) {
                selected_expert = E - 1;
                selected_logit = logits[row_base + (unsigned long long)selected_expert];
            }
        }
        int softmax_top_k = routed_top_k;
        float selected_for_max = -CUDART_INF_F;
        if (softmax_top_k > lane) {
            selected_for_max = selected_logit;
        }
        float selected_max;
        asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(selected_max) : "f"(selected_for_max));
        float selected_exp = 0.0f;
        if (softmax_top_k > lane) {
            float _exp2_41 = approx_exp2((selected_logit - selected_max) * 1.442695f);
            selected_exp = _exp2_41;
        }
        float selected_sum = selected_exp;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            selected_sum += __shfl_xor_sync(0xFFFFFFFF, selected_sum, offset);
        float selected_sum_rcp = approx_rcp(selected_sum);
        if (lane < top_k) {
            float weight = selected_logit;
            if (softmax_top_k > lane) {
                weight = selected_exp * selected_sum_rcp;
            }
            topk_weights[output_base + (unsigned long long)lane] = weight;
            if (private_raw_hist == 0) {
                atomicAdd(&private_counts[selected_expert & 511], 1);
            }
            topk_ids[output_base + (unsigned long long)lane] = selected_expert;
        }
    }
    if (private_raw_hist == 0) {
        __syncthreads();
        #pragma unroll
        for (int table_add = 0; table_add < 4; table_add++) {
            int table_count = private_counts[tid * 4 + table_add];
            if (table_count > 0) {
                atomicAdd(&expert_counts[tid * 4 + table_add], table_count);
            }
        }
    }
    cooperative_groups::this_grid().sync();
    if (bid * THREADS < M * top_k || bid * THREADS < E) {
        int scan_counts[4];
        if (private_raw_hist != 0) {
            if (((unsigned long long)topk_ids & 15) == 0) {
                if (((M * top_k >> 2) + THREADS - 1) / THREADS > 0 && ((M * top_k >> 2) + THREADS - 1) / THREADS <= 4) {
                    for (int raw_batch = 0; raw_batch < M * top_k >> 2 << 2; raw_batch += THREADS * 4 * 4) {
                        int batch_ids[16];
                        if (raw_batch + tid * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_42 = reinterpret_cast<const int4*>(topk_ids + raw_batch + tid * 4);
                                int4 _ivld_42;
                                _ivld_42 = *_ivptr_42;
                                batch_ids[0 + 0] = _ivld_42.x;
                                batch_ids[0 + 1] = _ivld_42.y;
                                batch_ids[0 + 2] = _ivld_42.z;
                                batch_ids[0 + 3] = _ivld_42.w;
                            }
                        }
                        if (raw_batch + (THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_43 = reinterpret_cast<const int4*>(topk_ids + raw_batch + (THREADS + tid) * 4);
                                int4 _ivld_43;
                                _ivld_43 = *_ivptr_43;
                                batch_ids[4 + 0] = _ivld_43.x;
                                batch_ids[4 + 1] = _ivld_43.y;
                                batch_ids[4 + 2] = _ivld_43.z;
                                batch_ids[4 + 3] = _ivld_43.w;
                            }
                        }
                        if (raw_batch + (2 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_44 = reinterpret_cast<const int4*>(topk_ids + raw_batch + (2 * THREADS + tid) * 4);
                                int4 _ivld_44;
                                _ivld_44 = *_ivptr_44;
                                batch_ids[8 + 0] = _ivld_44.x;
                                batch_ids[8 + 1] = _ivld_44.y;
                                batch_ids[8 + 2] = _ivld_44.z;
                                batch_ids[8 + 3] = _ivld_44.w;
                            }
                        }
                        if (raw_batch + (3 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_45 = reinterpret_cast<const int4*>(topk_ids + raw_batch + (3 * THREADS + tid) * 4);
                                int4 _ivld_45;
                                _ivld_45 = *_ivptr_45;
                                batch_ids[12 + 0] = _ivld_45.x;
                                batch_ids[12 + 1] = _ivld_45.y;
                                batch_ids[12 + 2] = _ivld_45.z;
                                batch_ids[12 + 3] = _ivld_45.w;
                            }
                        }
                        if (raw_batch + tid * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids[0] & 511], 1);
                            atomicAdd(&private_counts[batch_ids[1] & 511], 1);
                            atomicAdd(&private_counts[batch_ids[2] & 511], 1);
                            atomicAdd(&private_counts[batch_ids[3] & 511], 1);
                        }
                        if (raw_batch + (THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids[4] & 511], 1);
                            atomicAdd(&private_counts[batch_ids[5] & 511], 1);
                            atomicAdd(&private_counts[batch_ids[6] & 511], 1);
                            atomicAdd(&private_counts[batch_ids[7] & 511], 1);
                        }
                        if (raw_batch + (2 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids[8] & 511], 1);
                            atomicAdd(&private_counts[batch_ids[9] & 511], 1);
                            atomicAdd(&private_counts[batch_ids[10] & 511], 1);
                            atomicAdd(&private_counts[batch_ids[11] & 511], 1);
                        }
                        if (raw_batch + (3 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids[12] & 511], 1);
                            atomicAdd(&private_counts[batch_ids[13] & 511], 1);
                            atomicAdd(&private_counts[batch_ids[14] & 511], 1);
                            atomicAdd(&private_counts[batch_ids[15] & 511], 1);
                        }
                    }
                }
                if (((M * top_k >> 2) + THREADS - 1) / THREADS > 4 && ((M * top_k >> 2) + THREADS - 1) / THREADS <= 8) {
                    for (int raw_batch_1 = 0; raw_batch_1 < M * top_k >> 2 << 2; raw_batch_1 += THREADS * 4 * 8) {
                        int batch_ids_1[32];
                        if (raw_batch_1 + tid * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_46 = reinterpret_cast<const int4*>(topk_ids + raw_batch_1 + tid * 4);
                                int4 _ivld_46;
                                _ivld_46 = *_ivptr_46;
                                batch_ids_1[0 + 0] = _ivld_46.x;
                                batch_ids_1[0 + 1] = _ivld_46.y;
                                batch_ids_1[0 + 2] = _ivld_46.z;
                                batch_ids_1[0 + 3] = _ivld_46.w;
                            }
                        }
                        if (raw_batch_1 + (THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_47 = reinterpret_cast<const int4*>(topk_ids + raw_batch_1 + (THREADS + tid) * 4);
                                int4 _ivld_47;
                                _ivld_47 = *_ivptr_47;
                                batch_ids_1[4 + 0] = _ivld_47.x;
                                batch_ids_1[4 + 1] = _ivld_47.y;
                                batch_ids_1[4 + 2] = _ivld_47.z;
                                batch_ids_1[4 + 3] = _ivld_47.w;
                            }
                        }
                        if (raw_batch_1 + (2 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_48 = reinterpret_cast<const int4*>(topk_ids + raw_batch_1 + (2 * THREADS + tid) * 4);
                                int4 _ivld_48;
                                _ivld_48 = *_ivptr_48;
                                batch_ids_1[8 + 0] = _ivld_48.x;
                                batch_ids_1[8 + 1] = _ivld_48.y;
                                batch_ids_1[8 + 2] = _ivld_48.z;
                                batch_ids_1[8 + 3] = _ivld_48.w;
                            }
                        }
                        if (raw_batch_1 + (3 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_49 = reinterpret_cast<const int4*>(topk_ids + raw_batch_1 + (3 * THREADS + tid) * 4);
                                int4 _ivld_49;
                                _ivld_49 = *_ivptr_49;
                                batch_ids_1[12 + 0] = _ivld_49.x;
                                batch_ids_1[12 + 1] = _ivld_49.y;
                                batch_ids_1[12 + 2] = _ivld_49.z;
                                batch_ids_1[12 + 3] = _ivld_49.w;
                            }
                        }
                        if (raw_batch_1 + (4 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_50 = reinterpret_cast<const int4*>(topk_ids + raw_batch_1 + (4 * THREADS + tid) * 4);
                                int4 _ivld_50;
                                _ivld_50 = *_ivptr_50;
                                batch_ids_1[16 + 0] = _ivld_50.x;
                                batch_ids_1[16 + 1] = _ivld_50.y;
                                batch_ids_1[16 + 2] = _ivld_50.z;
                                batch_ids_1[16 + 3] = _ivld_50.w;
                            }
                        }
                        if (raw_batch_1 + (5 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_51 = reinterpret_cast<const int4*>(topk_ids + raw_batch_1 + (5 * THREADS + tid) * 4);
                                int4 _ivld_51;
                                _ivld_51 = *_ivptr_51;
                                batch_ids_1[20 + 0] = _ivld_51.x;
                                batch_ids_1[20 + 1] = _ivld_51.y;
                                batch_ids_1[20 + 2] = _ivld_51.z;
                                batch_ids_1[20 + 3] = _ivld_51.w;
                            }
                        }
                        if (raw_batch_1 + (6 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_52 = reinterpret_cast<const int4*>(topk_ids + raw_batch_1 + (6 * THREADS + tid) * 4);
                                int4 _ivld_52;
                                _ivld_52 = *_ivptr_52;
                                batch_ids_1[24 + 0] = _ivld_52.x;
                                batch_ids_1[24 + 1] = _ivld_52.y;
                                batch_ids_1[24 + 2] = _ivld_52.z;
                                batch_ids_1[24 + 3] = _ivld_52.w;
                            }
                        }
                        if (raw_batch_1 + (7 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_53 = reinterpret_cast<const int4*>(topk_ids + raw_batch_1 + (7 * THREADS + tid) * 4);
                                int4 _ivld_53;
                                _ivld_53 = *_ivptr_53;
                                batch_ids_1[28 + 0] = _ivld_53.x;
                                batch_ids_1[28 + 1] = _ivld_53.y;
                                batch_ids_1[28 + 2] = _ivld_53.z;
                                batch_ids_1[28 + 3] = _ivld_53.w;
                            }
                        }
                        if (raw_batch_1 + tid * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids_1[0] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_1[1] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_1[2] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_1[3] & 511], 1);
                        }
                        if (raw_batch_1 + (THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids_1[4] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_1[5] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_1[6] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_1[7] & 511], 1);
                        }
                        if (raw_batch_1 + (2 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids_1[8] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_1[9] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_1[10] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_1[11] & 511], 1);
                        }
                        if (raw_batch_1 + (3 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids_1[12] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_1[13] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_1[14] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_1[15] & 511], 1);
                        }
                        if (raw_batch_1 + (4 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids_1[16] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_1[17] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_1[18] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_1[19] & 511], 1);
                        }
                        if (raw_batch_1 + (5 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids_1[20] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_1[21] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_1[22] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_1[23] & 511], 1);
                        }
                        if (raw_batch_1 + (6 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids_1[24] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_1[25] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_1[26] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_1[27] & 511], 1);
                        }
                        if (raw_batch_1 + (7 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids_1[28] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_1[29] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_1[30] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_1[31] & 511], 1);
                        }
                    }
                }
                if (((M * top_k >> 2) + THREADS - 1) / THREADS > 8 && ((M * top_k >> 2) + THREADS - 1) / THREADS <= 16) {
                    for (int raw_batch_2 = 0; raw_batch_2 < M * top_k >> 2 << 2; raw_batch_2 += THREADS * 4 * 16) {
                        int batch_ids_2[64];
                        if (raw_batch_2 + tid * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_54 = reinterpret_cast<const int4*>(topk_ids + raw_batch_2 + tid * 4);
                                int4 _ivld_54;
                                _ivld_54 = *_ivptr_54;
                                batch_ids_2[0 + 0] = _ivld_54.x;
                                batch_ids_2[0 + 1] = _ivld_54.y;
                                batch_ids_2[0 + 2] = _ivld_54.z;
                                batch_ids_2[0 + 3] = _ivld_54.w;
                            }
                        }
                        if (raw_batch_2 + (THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_55 = reinterpret_cast<const int4*>(topk_ids + raw_batch_2 + (THREADS + tid) * 4);
                                int4 _ivld_55;
                                _ivld_55 = *_ivptr_55;
                                batch_ids_2[4 + 0] = _ivld_55.x;
                                batch_ids_2[4 + 1] = _ivld_55.y;
                                batch_ids_2[4 + 2] = _ivld_55.z;
                                batch_ids_2[4 + 3] = _ivld_55.w;
                            }
                        }
                        if (raw_batch_2 + (2 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_56 = reinterpret_cast<const int4*>(topk_ids + raw_batch_2 + (2 * THREADS + tid) * 4);
                                int4 _ivld_56;
                                _ivld_56 = *_ivptr_56;
                                batch_ids_2[8 + 0] = _ivld_56.x;
                                batch_ids_2[8 + 1] = _ivld_56.y;
                                batch_ids_2[8 + 2] = _ivld_56.z;
                                batch_ids_2[8 + 3] = _ivld_56.w;
                            }
                        }
                        if (raw_batch_2 + (3 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_57 = reinterpret_cast<const int4*>(topk_ids + raw_batch_2 + (3 * THREADS + tid) * 4);
                                int4 _ivld_57;
                                _ivld_57 = *_ivptr_57;
                                batch_ids_2[12 + 0] = _ivld_57.x;
                                batch_ids_2[12 + 1] = _ivld_57.y;
                                batch_ids_2[12 + 2] = _ivld_57.z;
                                batch_ids_2[12 + 3] = _ivld_57.w;
                            }
                        }
                        if (raw_batch_2 + (4 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_58 = reinterpret_cast<const int4*>(topk_ids + raw_batch_2 + (4 * THREADS + tid) * 4);
                                int4 _ivld_58;
                                _ivld_58 = *_ivptr_58;
                                batch_ids_2[16 + 0] = _ivld_58.x;
                                batch_ids_2[16 + 1] = _ivld_58.y;
                                batch_ids_2[16 + 2] = _ivld_58.z;
                                batch_ids_2[16 + 3] = _ivld_58.w;
                            }
                        }
                        if (raw_batch_2 + (5 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_59 = reinterpret_cast<const int4*>(topk_ids + raw_batch_2 + (5 * THREADS + tid) * 4);
                                int4 _ivld_59;
                                _ivld_59 = *_ivptr_59;
                                batch_ids_2[20 + 0] = _ivld_59.x;
                                batch_ids_2[20 + 1] = _ivld_59.y;
                                batch_ids_2[20 + 2] = _ivld_59.z;
                                batch_ids_2[20 + 3] = _ivld_59.w;
                            }
                        }
                        if (raw_batch_2 + (6 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_60 = reinterpret_cast<const int4*>(topk_ids + raw_batch_2 + (6 * THREADS + tid) * 4);
                                int4 _ivld_60;
                                _ivld_60 = *_ivptr_60;
                                batch_ids_2[24 + 0] = _ivld_60.x;
                                batch_ids_2[24 + 1] = _ivld_60.y;
                                batch_ids_2[24 + 2] = _ivld_60.z;
                                batch_ids_2[24 + 3] = _ivld_60.w;
                            }
                        }
                        if (raw_batch_2 + (7 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_61 = reinterpret_cast<const int4*>(topk_ids + raw_batch_2 + (7 * THREADS + tid) * 4);
                                int4 _ivld_61;
                                _ivld_61 = *_ivptr_61;
                                batch_ids_2[28 + 0] = _ivld_61.x;
                                batch_ids_2[28 + 1] = _ivld_61.y;
                                batch_ids_2[28 + 2] = _ivld_61.z;
                                batch_ids_2[28 + 3] = _ivld_61.w;
                            }
                        }
                        if (raw_batch_2 + (8 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_62 = reinterpret_cast<const int4*>(topk_ids + raw_batch_2 + (8 * THREADS + tid) * 4);
                                int4 _ivld_62;
                                _ivld_62 = *_ivptr_62;
                                batch_ids_2[32 + 0] = _ivld_62.x;
                                batch_ids_2[32 + 1] = _ivld_62.y;
                                batch_ids_2[32 + 2] = _ivld_62.z;
                                batch_ids_2[32 + 3] = _ivld_62.w;
                            }
                        }
                        if (raw_batch_2 + (9 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_63 = reinterpret_cast<const int4*>(topk_ids + raw_batch_2 + (9 * THREADS + tid) * 4);
                                int4 _ivld_63;
                                _ivld_63 = *_ivptr_63;
                                batch_ids_2[36 + 0] = _ivld_63.x;
                                batch_ids_2[36 + 1] = _ivld_63.y;
                                batch_ids_2[36 + 2] = _ivld_63.z;
                                batch_ids_2[36 + 3] = _ivld_63.w;
                            }
                        }
                        if (raw_batch_2 + (10 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_64 = reinterpret_cast<const int4*>(topk_ids + raw_batch_2 + (10 * THREADS + tid) * 4);
                                int4 _ivld_64;
                                _ivld_64 = *_ivptr_64;
                                batch_ids_2[40 + 0] = _ivld_64.x;
                                batch_ids_2[40 + 1] = _ivld_64.y;
                                batch_ids_2[40 + 2] = _ivld_64.z;
                                batch_ids_2[40 + 3] = _ivld_64.w;
                            }
                        }
                        if (raw_batch_2 + (11 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_65 = reinterpret_cast<const int4*>(topk_ids + raw_batch_2 + (11 * THREADS + tid) * 4);
                                int4 _ivld_65;
                                _ivld_65 = *_ivptr_65;
                                batch_ids_2[44 + 0] = _ivld_65.x;
                                batch_ids_2[44 + 1] = _ivld_65.y;
                                batch_ids_2[44 + 2] = _ivld_65.z;
                                batch_ids_2[44 + 3] = _ivld_65.w;
                            }
                        }
                        if (raw_batch_2 + (12 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_66 = reinterpret_cast<const int4*>(topk_ids + raw_batch_2 + (12 * THREADS + tid) * 4);
                                int4 _ivld_66;
                                _ivld_66 = *_ivptr_66;
                                batch_ids_2[48 + 0] = _ivld_66.x;
                                batch_ids_2[48 + 1] = _ivld_66.y;
                                batch_ids_2[48 + 2] = _ivld_66.z;
                                batch_ids_2[48 + 3] = _ivld_66.w;
                            }
                        }
                        if (raw_batch_2 + (13 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_67 = reinterpret_cast<const int4*>(topk_ids + raw_batch_2 + (13 * THREADS + tid) * 4);
                                int4 _ivld_67;
                                _ivld_67 = *_ivptr_67;
                                batch_ids_2[52 + 0] = _ivld_67.x;
                                batch_ids_2[52 + 1] = _ivld_67.y;
                                batch_ids_2[52 + 2] = _ivld_67.z;
                                batch_ids_2[52 + 3] = _ivld_67.w;
                            }
                        }
                        if (raw_batch_2 + (14 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_68 = reinterpret_cast<const int4*>(topk_ids + raw_batch_2 + (14 * THREADS + tid) * 4);
                                int4 _ivld_68;
                                _ivld_68 = *_ivptr_68;
                                batch_ids_2[56 + 0] = _ivld_68.x;
                                batch_ids_2[56 + 1] = _ivld_68.y;
                                batch_ids_2[56 + 2] = _ivld_68.z;
                                batch_ids_2[56 + 3] = _ivld_68.w;
                            }
                        }
                        if (raw_batch_2 + (15 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            {
                                const int4* _ivptr_69 = reinterpret_cast<const int4*>(topk_ids + raw_batch_2 + (15 * THREADS + tid) * 4);
                                int4 _ivld_69;
                                _ivld_69 = *_ivptr_69;
                                batch_ids_2[60 + 0] = _ivld_69.x;
                                batch_ids_2[60 + 1] = _ivld_69.y;
                                batch_ids_2[60 + 2] = _ivld_69.z;
                                batch_ids_2[60 + 3] = _ivld_69.w;
                            }
                        }
                        if (raw_batch_2 + tid * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids_2[0] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[1] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[2] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[3] & 511], 1);
                        }
                        if (raw_batch_2 + (THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids_2[4] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[5] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[6] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[7] & 511], 1);
                        }
                        if (raw_batch_2 + (2 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids_2[8] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[9] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[10] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[11] & 511], 1);
                        }
                        if (raw_batch_2 + (3 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids_2[12] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[13] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[14] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[15] & 511], 1);
                        }
                        if (raw_batch_2 + (4 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids_2[16] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[17] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[18] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[19] & 511], 1);
                        }
                        if (raw_batch_2 + (5 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids_2[20] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[21] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[22] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[23] & 511], 1);
                        }
                        if (raw_batch_2 + (6 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids_2[24] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[25] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[26] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[27] & 511], 1);
                        }
                        if (raw_batch_2 + (7 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids_2[28] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[29] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[30] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[31] & 511], 1);
                        }
                        if (raw_batch_2 + (8 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids_2[32] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[33] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[34] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[35] & 511], 1);
                        }
                        if (raw_batch_2 + (9 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids_2[36] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[37] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[38] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[39] & 511], 1);
                        }
                        if (raw_batch_2 + (10 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids_2[40] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[41] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[42] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[43] & 511], 1);
                        }
                        if (raw_batch_2 + (11 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids_2[44] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[45] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[46] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[47] & 511], 1);
                        }
                        if (raw_batch_2 + (12 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids_2[48] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[49] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[50] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[51] & 511], 1);
                        }
                        if (raw_batch_2 + (13 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids_2[52] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[53] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[54] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[55] & 511], 1);
                        }
                        if (raw_batch_2 + (14 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids_2[56] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[57] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[58] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[59] & 511], 1);
                        }
                        if (raw_batch_2 + (15 * THREADS + tid) * 4 < M * top_k >> 2 << 2) {
                            atomicAdd(&private_counts[batch_ids_2[60] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[61] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[62] & 511], 1);
                            atomicAdd(&private_counts[batch_ids_2[63] & 511], 1);
                        }
                    }
                }
                if (tid < (M * top_k & 3)) {
                    int rest_expert = topk_ids[(M * top_k >> 2 << 2) + tid] & 511;
                    atomicAdd(&private_counts[rest_expert], 1);
                }
            } else {
                for (int count_unaligned = tid; count_unaligned < M * top_k; count_unaligned += THREADS) {
                    int unaligned_expert = topk_ids[count_unaligned] & 511;
                    atomicAdd(&private_counts[unaligned_expert], 1);
                }
            }
            __syncthreads();
            #pragma unroll
            for (int private_load_slot = 0; private_load_slot < MAX_EXPERTS / THREADS; private_load_slot++) {
                int private_expert = tid * (MAX_EXPERTS / THREADS) + private_load_slot;
                int private_count = private_counts[private_expert];
                scan_counts[private_load_slot] = private_count;
                if (bid == 0) {
                    if (private_expert < E) {
                        expert_counts[private_expert] = private_count;
                    }
                }
            }
        } else {
            if (MAX_EXPERTS == 512 && E == 512 && ((unsigned long long)expert_counts & 15) == 0) {
                {
                    const int4* _ivptr_70 = reinterpret_cast<const int4*>(expert_counts + tid * 4);
                    int4 _ivld_70;
                    _ivld_70 = *_ivptr_70;
                    scan_counts[0 + 0] = _ivld_70.x;
                    scan_counts[0 + 1] = _ivld_70.y;
                    scan_counts[0 + 2] = _ivld_70.z;
                    scan_counts[0 + 3] = _ivld_70.w;
                }
            } else {
                #pragma unroll
                for (int expert_slot_count_load = 0; expert_slot_count_load < MAX_EXPERTS / THREADS; expert_slot_count_load++) {
                    int expert_count_load = tid * (MAX_EXPERTS / THREADS) + expert_slot_count_load;
                    int count_loaded = 0;
                    if (expert_count_load < E) {
                        count_loaded = expert_counts[expert_count_load];
                    }
                    scan_counts[expert_slot_count_load] = count_loaded;
                }
            }
            #pragma unroll
            for (int table_keep = 0; table_keep < MAX_EXPERTS / THREADS; table_keep++) {
                private_counts[tid * (MAX_EXPERTS / THREADS) + table_keep] = scan_counts[table_keep];
            }
        }
        int scan_padded[4];
        int scan_thread_total = 0;
        #pragma unroll
        for (int expert_slot_scan_init = 0; expert_slot_scan_init < MAX_EXPERTS / THREADS; expert_slot_scan_init++) {
            int expert_scan_init = tid * (MAX_EXPERTS / THREADS) + expert_slot_scan_init;
            int padded_count_init = 0;
            if (expert_scan_init < E) {
                int count_init = scan_counts[expert_slot_scan_init];
                if (block_m == 8) {
                    padded_count_init = count_init + 7 & -8;
                } else if (block_m == 16) {
                    padded_count_init = count_init + 15 & -16;
                } else {
                    padded_count_init = (count_init + block_m - 1) / block_m * block_m;
                }
            }
            scan_padded[expert_slot_scan_init] = padded_count_init;
            scan_thread_total = scan_thread_total + padded_count_init;
        }
        // CtaScanSum: threads=128, warps=4
        int _cta_scan_sum_prefix_0;
        int _cta_scan_sum_total_0;
        {
            const uint32_t _cta_scan_sum_prefix_0_own = (uint32_t)(scan_thread_total);
            uint32_t _cta_scan_sum_prefix_0_ws = _cta_scan_sum_prefix_0_own;
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _cta_scan_sum_prefix_0_ws, 1, 32); if (lane >= 1) { _cta_scan_sum_prefix_0_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _cta_scan_sum_prefix_0_ws, 2, 32); if (lane >= 2) { _cta_scan_sum_prefix_0_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _cta_scan_sum_prefix_0_ws, 4, 32); if (lane >= 4) { _cta_scan_sum_prefix_0_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _cta_scan_sum_prefix_0_ws, 8, 32); if (lane >= 8) { _cta_scan_sum_prefix_0_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _cta_scan_sum_prefix_0_ws, 16, 32); if (lane >= 16) { _cta_scan_sum_prefix_0_ws += peer; } }
            if (lane == 31) { scan_values[warp] = (int)_cta_scan_sum_prefix_0_ws; }
            __syncthreads();
            const uint32_t _cta_scan_sum_prefix_0_wt = (lane < 4) ? (uint32_t)scan_values[lane] : 0;
            uint32_t _cta_scan_sum_prefix_0_wsum = _cta_scan_sum_prefix_0_wt;
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _cta_scan_sum_prefix_0_wsum, 1, 32); if (lane >= 1) { _cta_scan_sum_prefix_0_wsum += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _cta_scan_sum_prefix_0_wsum, 2, 32); if (lane >= 2) { _cta_scan_sum_prefix_0_wsum += peer; } }
            _cta_scan_sum_total_0 = (int)__shfl_sync(0xFFFFFFFF, _cta_scan_sum_prefix_0_wsum, 3);
            const uint32_t _cta_scan_sum_prefix_0_before = __shfl_sync(0xFFFFFFFF, _cta_scan_sum_prefix_0_wsum - _cta_scan_sum_prefix_0_wt, warp);
            _cta_scan_sum_prefix_0 = (int)(_cta_scan_sum_prefix_0_ws - _cta_scan_sum_prefix_0_own + _cta_scan_sum_prefix_0_before);
        }
        if (tid == 0) {
            local_offsets[E] = _cta_scan_sum_total_0;
        }
        int scan_expert_prefix = _cta_scan_sum_prefix_0;
        if (MAX_EXPERTS == 512 && E == 512) {
            #pragma unroll
            for (int expert_slot_full_prefix = 0; expert_slot_full_prefix < MAX_EXPERTS / THREADS; expert_slot_full_prefix++) {
                int expert_full_prefix = tid * (MAX_EXPERTS / THREADS) + expert_slot_full_prefix;
                local_offsets[expert_full_prefix] = scan_expert_prefix;
                if (bid == 0) {
                    expert_offsets[expert_full_prefix] = scan_expert_prefix;
                }
                scan_expert_prefix = scan_expert_prefix + scan_padded[expert_slot_full_prefix];
            }
        } else {
            #pragma unroll
            for (int expert_slot_scan_store = 0; expert_slot_scan_store < MAX_EXPERTS / THREADS; expert_slot_scan_store++) {
                int expert_scan_store = tid * (MAX_EXPERTS / THREADS) + expert_slot_scan_store;
                if (expert_scan_store < E) {
                    local_offsets[expert_scan_store] = scan_expert_prefix;
                    if (bid == 0) {
                        expert_offsets[expert_scan_store] = scan_expert_prefix;
                    }
                }
                scan_expert_prefix = scan_expert_prefix + scan_padded[expert_slot_scan_store];
            }
        }
        __syncthreads();
        if (tid == 0 && bid == 0) {
            num_tokens_post_padded[0] = _cta_scan_sum_total_0;
            expert_offsets[E] = _cta_scan_sum_total_0;
        }
        int scatter_block_mask = block_m - 1;
        int scatter_block_shift = 0;
        if (block_m == 8) {
            scatter_block_shift = 3;
        } else if (block_m == 16) {
            scatter_block_shift = 4;
        }
        if (M * top_k <= num_bids * THREADS) {
            if (global_thread < M * top_k) {
                int pair = global_thread;
                int stored_pair = topk_ids[pair];
                int pair_expert = stored_pair;
                int local_row = 0;
                int _atomic_fetch_add_71 = atomicAdd(&expert_scatter_offsets[pair_expert], 1);
                local_row = _atomic_fetch_add_71;
                int grouped_row = local_offsets[pair_expert & MAX_EXPERTS - 1] + local_row;
                sorted_token_ids[grouped_row] = pair;
                int scatter_block_remainder = 0;
                if (scatter_block_shift != 0) {
                    scatter_block_remainder = local_row & scatter_block_mask;
                } else {
                    scatter_block_remainder = local_row % block_m;
                }
                if (scatter_block_remainder == 0) {
                    int scatter_block_index = 0;
                    if (scatter_block_shift != 0) {
                        scatter_block_index = grouped_row >> scatter_block_shift;
                    } else {
                        scatter_block_index = grouped_row / block_m;
                    }
                    expert_ids[scatter_block_index] = pair_expert;
                }
            }
        } else {
            for (int pair_1 = global_thread; pair_1 < M * top_k; pair_1 += num_bids * THREADS) {
                int stored_pair_1 = topk_ids[pair_1];
                int pair_expert_1 = stored_pair_1;
                int local_row_1 = 0;
                int _atomic_fetch_add_72 = atomicAdd(&expert_scatter_offsets[pair_expert_1], 1);
                local_row_1 = _atomic_fetch_add_72;
                int grouped_row_1 = local_offsets[pair_expert_1 & MAX_EXPERTS - 1] + local_row_1;
                sorted_token_ids[grouped_row_1] = pair_1;
                int scatter_block_remainder_1 = 0;
                if (scatter_block_shift != 0) {
                    scatter_block_remainder_1 = local_row_1 & scatter_block_mask;
                } else {
                    scatter_block_remainder_1 = local_row_1 % block_m;
                }
                if (scatter_block_remainder_1 == 0) {
                    int scatter_block_index_1 = 0;
                    if (scatter_block_shift != 0) {
                        scatter_block_index_1 = grouped_row_1 >> scatter_block_shift;
                    } else {
                        scatter_block_index_1 = grouped_row_1 / block_m;
                    }
                    expert_ids[scatter_block_index_1] = pair_expert_1;
                }
            }
        }
        for (int padding_expert = global_thread; padding_expert < E; padding_expert += num_bids * THREADS) {
            int count_final = 0;
            count_final = private_counts[padding_expert];
            int expert_start = local_offsets[padding_expert];
            int padded_count_final = local_offsets[padding_expert + 1] - expert_start;
            int padding_count = padded_count_final - count_final;
            #pragma unroll
            for (int padding_slot = 0; padding_slot < MAX_BLOCK_M; padding_slot++) {
                if (padding_count > padding_slot) {
                    sorted_token_ids[expert_start + count_final + padding_slot] = M * top_k;
                }
            }
        }
    }
}

} // extern "C"
