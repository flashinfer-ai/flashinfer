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
#define SMEM_STAGE_STAGE_BYTES 16384
#define SMEM_STAGE_STRIDE 16384
#define SMEM_LOCAL_OFFSETS_OFF 0
#define SMEM_LOCAL_OFFSETS_STAGE_BYTES 4100
#define SMEM_LOCAL_OFFSETS_STRIDE 4100
#define SMEM_SCAN_VALUES_OFF 4100
#define SMEM_SCAN_VALUES_STAGE_BYTES 32
#define SMEM_SCAN_VALUES_STRIDE 32
#define SMEM_TOTAL 20608
#define THREADS 1024
#define NUM_WARPS 32
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

__global__ __launch_bounds__(1024) void
kernel_alpha_moe_fused_router_e512_threshold_cta_per_sm(float* __restrict__ logits, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M, int E, int top_k, int block_m, int has_shared_expert)
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

    // === Task calls (declaration order) ===
    int global_thread = bid * THREADS + tid;
    int use_cta_reservations = 0;
    if (MAX_EXPERTS == 512 && PUBLIC_SHARED_SOFTMAX == 0 && M > 512 && M <= 2147483647 / (MAX_EXPERTS * MAX_TOP_K)) {
        use_cta_reservations = 1;
    }
    if (bid == 0) {
        for (int zero_cell = tid; zero_cell < E + 1; zero_cell += THREADS) {
            expert_offsets[zero_cell] = 0;
            if (zero_cell < E) {
                expert_counts[zero_cell] = 0;
                expert_scatter_offsets[zero_cell] = 0;
            }
        }
        if (tid == 0) {
            num_tokens_post_padded[0] = 0;
        }
    }
    if (tid < MAX_EXPERTS) {
        local_offsets[tid] = 0;
    }
    __syncthreads();
    unsigned int _grid_arrive_0 = cooperative_groups::this_grid().barrier_arrive();
    unsigned int grid_token = _grid_arrive_0;
    int routed_experts = E - has_shared_expert;
    int routed_top_k = top_k - has_shared_expert;
    int im1_ctas = num_bids;
    int im1_cta = bid;
    int im1_grid_warps = im1_ctas * NUM_WARPS;
    int im1_full_passes = M / im1_grid_warps;
    int im1_partial_tokens = M - im1_full_passes * im1_grid_warps;
    int im1_partial_share = im1_partial_tokens / im1_ctas;
    int im1_partial_extra = im1_partial_tokens - im1_partial_share * im1_ctas;
    int im1_partial_run = im1_partial_share;
    if (im1_cta < im1_partial_extra) {
        im1_partial_run = im1_partial_share + 1;
    }
    int _min_0 = ((im1_cta) < (im1_partial_extra) ? (im1_cta) : (im1_partial_extra));
    int im1_partial_start = im1_full_passes * im1_grid_warps + im1_cta * im1_partial_share + _min_0;
    int im1_warp_passes = im1_full_passes;
    if (im1_partial_run > warp) {
        im1_warp_passes = im1_full_passes + 1;
    }
    int im1_full_base = im1_cta * NUM_WARPS + warp;
    int im1_partial_token = im1_partial_start + warp;
    for (int token = 0; token < im1_warp_passes; token++) {
        int im1_token = ((im1_full_passes > token) ? im1_full_base + token * im1_grid_warps : im1_partial_token);
        unsigned long long row_base = (unsigned long long)im1_token * (unsigned long long)E;
        float row_values[16];
        if (MAX_EXPERTS == 512 && PUBLIC_SHARED_SOFTMAX == 0 && E == 512 && has_shared_expert == 0) {
            unsigned long long lane_row_base = row_base + (unsigned long long)lane;
            #pragma unroll
            for (int expert_slot_full = 0; expert_slot_full < MAX_EXPERTS / 32; expert_slot_full++) {
                row_values[expert_slot_full] = logits[lane_row_base + (unsigned long long)(expert_slot_full * 32)];
            }
        } else {
            #pragma unroll
            for (int expert_slot_load = 0; expert_slot_load < MAX_EXPERTS / 32; expert_slot_load++) {
                int expert_load = lane + expert_slot_load * 32;
                row_values[expert_slot_load] = -CUDART_INF_F;
                if (expert_load < routed_experts) {
                    row_values[expert_slot_load] = logits[row_base + (unsigned long long)expert_load];
                }
            }
        }
        unsigned long long output_base = (unsigned long long)im1_token * (unsigned long long)top_k;
        unsigned int nan_bits = 0;
        float _max_6;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_6) : "f"(row_values[0]), "f"(row_values[1]));
        float _max_5;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_5) : "f"(_max_6), "f"(row_values[2]));
        float _max_8;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_8) : "f"(row_values[3]), "f"(row_values[4]));
        float _max_7;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_7) : "f"(_max_8), "f"(row_values[5]));
        float _max_4;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_4) : "f"(_max_5), "f"(_max_7));
        float _max_10;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_10) : "f"(row_values[6]), "f"(row_values[7]));
        float _max_9;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_9) : "f"(_max_10), "f"(row_values[8]));
        float _max_3;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_3) : "f"(_max_4), "f"(_max_9));
        float _max_14;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_14) : "f"(row_values[9]), "f"(row_values[10]));
        float _max_13;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_13) : "f"(_max_14), "f"(row_values[11]));
        float _max_16;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_16) : "f"(row_values[12]), "f"(row_values[13]));
        float _max_15;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_15) : "f"(_max_16), "f"(row_values[14]));
        float _max_12;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_12) : "f"(_max_13), "f"(_max_15));
        float _max_11;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_11) : "f"(_max_12), "f"(row_values[15]));
        float _max_2;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_2) : "f"(_max_3), "f"(_max_11));
        float _max_1;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_1) : "f"(_max_2), "f"(-CUDART_INF_F));
        float _bit_cast_src_17 = _max_1;
        nan_bits = reinterpret_cast<unsigned int*>(&_bit_cast_src_17)[0];
        int _warp_vote_18 = __any_sync(0xFFFFFFFF, (nan_bits & 2147483647) > 2139095040);
        if (_warp_vote_18 != 0) {
            #pragma unroll
            for (int nan_slot = 0; nan_slot < MAX_EXPERTS / 32; nan_slot++) {
                row_values[nan_slot] = fminf(row_values[nan_slot], CUDART_INF_F);
            }
        }
        float selected_logit = -CUDART_INF_F;
        int selected_expert = MAX_EXPERTS;
        unsigned int lanes_below = (1 << (unsigned int)lane) - 1;
        int route_by_route = 1;
        float lane_max_0 = fmaxf(row_values[0], row_values[1]);
        float lane_max_1 = fmaxf(row_values[2], row_values[3]);
        float lane_max_2 = fmaxf(row_values[4], row_values[5]);
        float lane_max_3 = fmaxf(row_values[6], row_values[7]);
        #pragma unroll
        for (int lane_group = 1; lane_group < MAX_EXPERTS / 256; lane_group++) {
            lane_max_0 = fmaxf(lane_max_0, fmaxf(row_values[lane_group * 8], row_values[lane_group * 8 + 1]));
            lane_max_1 = fmaxf(lane_max_1, fmaxf(row_values[lane_group * 8 + 2], row_values[lane_group * 8 + 3]));
            lane_max_2 = fmaxf(lane_max_2, fmaxf(row_values[lane_group * 8 + 4], row_values[lane_group * 8 + 5]));
            lane_max_3 = fmaxf(lane_max_3, fmaxf(row_values[lane_group * 8 + 6], row_values[lane_group * 8 + 7]));
        }
        unsigned int lane_bits = 0;
        float _bit_cast_src_19 = fmaxf(fmaxf(lane_max_0, lane_max_1), fmaxf(lane_max_2, lane_max_3));
        lane_bits = reinterpret_cast<unsigned int*>(&_bit_cast_src_19)[0];
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
        unsigned int slot_lanes[16];
        int candidates = 0;
        unsigned int _warp_vote_20 = __ballot_sync(0xFFFFFFFF, row_values[0] >= bound_value);
        slot_lanes[0] = _warp_vote_20;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[0]);
        unsigned int _warp_vote_21 = __ballot_sync(0xFFFFFFFF, row_values[1] >= bound_value);
        slot_lanes[1] = _warp_vote_21;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[1]);
        unsigned int _warp_vote_22 = __ballot_sync(0xFFFFFFFF, row_values[2] >= bound_value);
        slot_lanes[2] = _warp_vote_22;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[2]);
        unsigned int _warp_vote_23 = __ballot_sync(0xFFFFFFFF, row_values[3] >= bound_value);
        slot_lanes[3] = _warp_vote_23;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[3]);
        unsigned int _warp_vote_24 = __ballot_sync(0xFFFFFFFF, row_values[4] >= bound_value);
        slot_lanes[4] = _warp_vote_24;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[4]);
        unsigned int _warp_vote_25 = __ballot_sync(0xFFFFFFFF, row_values[5] >= bound_value);
        slot_lanes[5] = _warp_vote_25;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[5]);
        unsigned int _warp_vote_26 = __ballot_sync(0xFFFFFFFF, row_values[6] >= bound_value);
        slot_lanes[6] = _warp_vote_26;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[6]);
        unsigned int _warp_vote_27 = __ballot_sync(0xFFFFFFFF, row_values[7] >= bound_value);
        slot_lanes[7] = _warp_vote_27;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[7]);
        unsigned int _warp_vote_28 = __ballot_sync(0xFFFFFFFF, row_values[8] >= bound_value);
        slot_lanes[8] = _warp_vote_28;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[8]);
        unsigned int _warp_vote_29 = __ballot_sync(0xFFFFFFFF, row_values[9] >= bound_value);
        slot_lanes[9] = _warp_vote_29;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[9]);
        unsigned int _warp_vote_30 = __ballot_sync(0xFFFFFFFF, row_values[10] >= bound_value);
        slot_lanes[10] = _warp_vote_30;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[10]);
        unsigned int _warp_vote_31 = __ballot_sync(0xFFFFFFFF, row_values[11] >= bound_value);
        slot_lanes[11] = _warp_vote_31;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[11]);
        unsigned int _warp_vote_32 = __ballot_sync(0xFFFFFFFF, row_values[12] >= bound_value);
        slot_lanes[12] = _warp_vote_32;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[12]);
        unsigned int _warp_vote_33 = __ballot_sync(0xFFFFFFFF, row_values[13] >= bound_value);
        slot_lanes[13] = _warp_vote_33;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[13]);
        unsigned int _warp_vote_34 = __ballot_sync(0xFFFFFFFF, row_values[14] >= bound_value);
        slot_lanes[14] = _warp_vote_34;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[14]);
        unsigned int _warp_vote_35 = __ballot_sync(0xFFFFFFFF, row_values[15] >= bound_value);
        slot_lanes[15] = _warp_vote_35;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[15]);
        if (candidates <= 32 && bound_key != 2147483648u && bound_key > 8388607) {
            route_by_route = 0;
            int candidate_position = 0;
            int stage_rows[16];
            unsigned int stage_word = 0;
            stage_rows[0] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[0] & lanes_below);
            if ((slot_lanes[0] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_36 = row_values[0];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_36)[0];
                stage[stage_rows[0]] = stage_word;
                stage[stage_rows[0] + 32] = (unsigned int)lane;
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[0]);
            stage_rows[1] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[1] & lanes_below);
            if ((slot_lanes[1] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_37 = row_values[1];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_37)[0];
                stage[stage_rows[1]] = stage_word;
                stage[stage_rows[1] + 32] = (unsigned int)(lane + 32);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[1]);
            stage_rows[2] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[2] & lanes_below);
            if ((slot_lanes[2] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_38 = row_values[2];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_38)[0];
                stage[stage_rows[2]] = stage_word;
                stage[stage_rows[2] + 32] = (unsigned int)(lane + 64);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[2]);
            stage_rows[3] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[3] & lanes_below);
            if ((slot_lanes[3] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_39 = row_values[3];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_39)[0];
                stage[stage_rows[3]] = stage_word;
                stage[stage_rows[3] + 32] = (unsigned int)(lane + 96);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[3]);
            stage_rows[4] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[4] & lanes_below);
            if ((slot_lanes[4] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_40 = row_values[4];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_40)[0];
                stage[stage_rows[4]] = stage_word;
                stage[stage_rows[4] + 32] = (unsigned int)(lane + 128);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[4]);
            stage_rows[5] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[5] & lanes_below);
            if ((slot_lanes[5] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_41 = row_values[5];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_41)[0];
                stage[stage_rows[5]] = stage_word;
                stage[stage_rows[5] + 32] = (unsigned int)(lane + 160);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[5]);
            stage_rows[6] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[6] & lanes_below);
            if ((slot_lanes[6] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_42 = row_values[6];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_42)[0];
                stage[stage_rows[6]] = stage_word;
                stage[stage_rows[6] + 32] = (unsigned int)(lane + 192);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[6]);
            stage_rows[7] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[7] & lanes_below);
            if ((slot_lanes[7] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_43 = row_values[7];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_43)[0];
                stage[stage_rows[7]] = stage_word;
                stage[stage_rows[7] + 32] = (unsigned int)(lane + 224);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[7]);
            stage_rows[8] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[8] & lanes_below);
            if ((slot_lanes[8] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_44 = row_values[8];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_44)[0];
                stage[stage_rows[8]] = stage_word;
                stage[stage_rows[8] + 32] = (unsigned int)(lane + 256);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[8]);
            stage_rows[9] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[9] & lanes_below);
            if ((slot_lanes[9] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_45 = row_values[9];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_45)[0];
                stage[stage_rows[9]] = stage_word;
                stage[stage_rows[9] + 32] = (unsigned int)(lane + 288);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[9]);
            stage_rows[10] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[10] & lanes_below);
            if ((slot_lanes[10] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_46 = row_values[10];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_46)[0];
                stage[stage_rows[10]] = stage_word;
                stage[stage_rows[10] + 32] = (unsigned int)(lane + 320);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[10]);
            stage_rows[11] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[11] & lanes_below);
            if ((slot_lanes[11] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_47 = row_values[11];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_47)[0];
                stage[stage_rows[11]] = stage_word;
                stage[stage_rows[11] + 32] = (unsigned int)(lane + 352);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[11]);
            stage_rows[12] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[12] & lanes_below);
            if ((slot_lanes[12] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_48 = row_values[12];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_48)[0];
                stage[stage_rows[12]] = stage_word;
                stage[stage_rows[12] + 32] = (unsigned int)(lane + 384);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[12]);
            stage_rows[13] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[13] & lanes_below);
            if ((slot_lanes[13] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_49 = row_values[13];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_49)[0];
                stage[stage_rows[13]] = stage_word;
                stage[stage_rows[13] + 32] = (unsigned int)(lane + 416);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[13]);
            stage_rows[14] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[14] & lanes_below);
            if ((slot_lanes[14] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_50 = row_values[14];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_50)[0];
                stage[stage_rows[14]] = stage_word;
                stage[stage_rows[14] + 32] = (unsigned int)(lane + 448);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[14]);
            stage_rows[15] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[15] & lanes_below);
            if ((slot_lanes[15] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_51 = row_values[15];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_51)[0];
                stage[stage_rows[15]] = stage_word;
                stage[stage_rows[15] + 32] = (unsigned int)(lane + 480);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[15]);
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
                unsigned int _bit_cast_src_52 = stage[stage_base + 64 + lane];
                selected_logit = reinterpret_cast<float*>(&_bit_cast_src_52)[0];
                selected_expert = (int)stage[stage_base + 80 + lane];
            }
        }
        #pragma unroll 1
        for (int route = 0; route < routed_top_k * route_by_route; route++) {
            float local_value = 0.0f;
            {
                float local_max_t0 = fmaxf(fmaxf(row_values[0], row_values[1]), row_values[2]);
                float local_max_t1 = fmaxf(fmaxf(row_values[3], row_values[4]), row_values[5]);
                float local_max_t2 = fmaxf(fmaxf(row_values[6], row_values[7]), row_values[8]);
                float local_max_t3 = fmaxf(fmaxf(row_values[9], row_values[10]), row_values[11]);
                float local_max_t4 = fmaxf(fmaxf(row_values[12], row_values[13]), row_values[14]);
                float local_max_u = fmaxf(fmaxf(local_max_t0, local_max_t1), local_max_t2);
                float local_max_v = fmaxf(fmaxf(local_max_t3, local_max_t4), row_values[15]);
                local_value = fmaxf(fmaxf(local_max_u, local_max_v), -CUDART_INF_F);
            }
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
            int descending_tie_key = 0;
            {
                unsigned int bounded_tie_masks[4];
                #pragma unroll
                for (int bounded_mask_init = 0; bounded_mask_init < 4; bounded_mask_init++) {
                    bounded_tie_masks[bounded_mask_init] = 0;
                }
                #pragma unroll
                for (int bounded_mask_slot = 0; bounded_mask_slot < MAX_EXPERTS / 128; bounded_mask_slot++) {
                    #pragma unroll
                    for (int bounded_mask_group = 0; bounded_mask_group < 4; bounded_mask_group++) {
                        if (row_values[bounded_mask_group * (MAX_EXPERTS / 128) + bounded_mask_slot] == best_value) {
                            bounded_tie_masks[bounded_mask_group] = bounded_tie_masks[bounded_mask_group] | (unsigned int)(2147483648 >> bounded_mask_group * (MAX_EXPERTS / 128) + bounded_mask_slot);
                        }
                    }
                }
                unsigned int bounded_mask_low = bounded_tie_masks[0] | bounded_tie_masks[1];
                unsigned int bounded_mask_high = bounded_tie_masks[2] | bounded_tie_masks[3];
                unsigned int bounded_mask_all = bounded_mask_low | bounded_mask_high;
                int bounded_high_bit;
                asm volatile("bfind.u32 %0, %1;" : "=r"(bounded_high_bit) : "r"(bounded_mask_all));
                descending_tie_key = bounded_high_bit * 32 - (int)lane;
            }
            int best_index = 0;
            {
                int best_tie_key;
                asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(best_tie_key) : "r"(descending_tie_key));
                best_index = 992 - best_tie_key;
            }
            {
                best_index = ((best_index < routed_experts) ? best_index : MAX_EXPERTS);
            }
            if (lane == route) {
                selected_logit = best_value;
                selected_expert = best_index;
            }
            #pragma unroll
            for (int remove_group = 0; remove_group < MAX_EXPERTS / 256; remove_group++) {
                if (best_index >= remove_group * 256 && best_index < (remove_group + 1) * 256) {
                    if (best_index < remove_group * 256 + 128) {
                        if (best_index < remove_group * 256 + 64) {
                            if (best_index < remove_group * 256 + 32) {
                                if (lane + remove_group * 8 * 32 == best_index) {
                                    row_values[remove_group * 8] = -CUDART_INF_F;
                                }
                            } else if (lane + (remove_group * 8 + 1) * 32 == best_index) {
                                row_values[remove_group * 8 + 1] = -CUDART_INF_F;
                            }
                        } else if (best_index < remove_group * 256 + 96) {
                            if (lane + (remove_group * 8 + 2) * 32 == best_index) {
                                row_values[remove_group * 8 + 2] = -CUDART_INF_F;
                            }
                        } else {
                            if (lane + (remove_group * 8 + 3) * 32 == best_index) {
                                row_values[remove_group * 8 + 3] = -CUDART_INF_F;
                            }
                        }
                    } else if (best_index < remove_group * 256 + 192) {
                        if (best_index < remove_group * 256 + 160) {
                            if (lane + (remove_group * 8 + 4) * 32 == best_index) {
                                row_values[remove_group * 8 + 4] = -CUDART_INF_F;
                            }
                        } else if (lane + (remove_group * 8 + 5) * 32 == best_index) {
                            row_values[remove_group * 8 + 5] = -CUDART_INF_F;
                        }
                    } else {
                        if (best_index < remove_group * 256 + 224) {
                            if (lane + (remove_group * 8 + 6) * 32 == best_index) {
                                row_values[remove_group * 8 + 6] = -CUDART_INF_F;
                            }
                        } else if (lane + (remove_group * 8 + 7) * 32 == best_index) {
                            row_values[remove_group * 8 + 7] = -CUDART_INF_F;
                        }
                    }
                }
            }
        }
        if (has_shared_expert != 0) {
            if (lane == routed_top_k) {
                selected_expert = E - 1;
                selected_logit = logits[row_base + (unsigned long long)selected_expert];
            }
        }
        int count_ordinal = 0;
        if (lane < top_k) {
            if (use_cta_reservations != 0) {
                int _atomic_fetch_add_53 = atomicAdd(&local_offsets[selected_expert & MAX_EXPERTS - 1], 1);
                count_ordinal = _atomic_fetch_add_53;
            } else {
                atomicAdd(&local_offsets[selected_expert & MAX_EXPERTS - 1], 1);
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
            float _exp2_54 = approx_exp2((selected_logit - selected_max) * 1.442695f);
            selected_exp = _exp2_54;
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
            int stored_expert = selected_expert;
            if (use_cta_reservations != 0) {
                stored_expert = count_ordinal * MAX_EXPERTS + selected_expert;
            }
            topk_ids[output_base + (unsigned long long)lane] = stored_expert;
        }
    }
    int first_stored_expert = 0;
    if (use_cta_reservations != 0) {
        if (im1_warp_passes > 0 && lane < top_k) {
            int first_rebase_token = ((im1_full_passes > 0) ? im1_full_base : im1_partial_token);
            unsigned long long first_rebase_pair = (unsigned long long)first_rebase_token * (unsigned long long)top_k + (unsigned long long)lane;
            first_stored_expert = topk_ids[first_rebase_pair];
        }
    }
    cooperative_groups::this_grid().barrier_wait(static_cast<cooperative_groups::grid_group::arrival_token>(grid_token));
    __syncthreads();
    if (tid < MAX_EXPERTS) {
        int count_reserve = local_offsets[tid];
        if (count_reserve > 0) {
            if (use_cta_reservations != 0) {
                int base_reserve = atomicAdd(&expert_counts[tid], count_reserve);
                local_offsets[tid] = base_reserve;
            } else {
                atomicAdd(&expert_counts[tid], count_reserve);
            }
        }
    }
    if (use_cta_reservations != 0) {
        __syncthreads();
        for (int rebase_pass = 0; rebase_pass < im1_warp_passes; rebase_pass++) {
            if (lane < top_k) {
                int token_rebase = ((im1_full_passes > rebase_pass) ? im1_full_base + rebase_pass * im1_grid_warps : im1_partial_token);
                unsigned long long pair_rebase = (unsigned long long)token_rebase * (unsigned long long)top_k + (unsigned long long)lane;
                int stored_rebase = first_stored_expert;
                if (rebase_pass != 0) {
                    stored_rebase = topk_ids[pair_rebase];
                }
                int expert_rebase = stored_rebase & MAX_EXPERTS - 1;
                topk_ids[pair_rebase] = stored_rebase + local_offsets[expert_rebase] * MAX_EXPERTS;
            }
        }
    }
}

} // extern "C"
