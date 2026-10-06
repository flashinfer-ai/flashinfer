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
#define THREADS 128

#include <math_constants.h>

__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
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

__global__ __launch_bounds__(128) void
kernel_cake_stepfun_moe_ef50826a59e50d172cbf(__nv_bfloat16* __restrict__ scores, __nv_bfloat16* __restrict__ topk_weights, int* __restrict__ topk_packed, int* __restrict__ expert_counts, int* __restrict__ permuted_idx_size, int* __restrict__ expanded_idx_to_permuted_idx, int* __restrict__ permuted_idx_to_token_idx, int* __restrict__ cta_idx_xy_to_batch_idx, int* __restrict__ cta_idx_xy_to_mn_limit, int* __restrict__ num_non_exiting_ctas, int num_tokens, int num_experts, int top_k, int padding_log2, int tile_tokens_dim, int local_experts_start_idx, int local_experts_stride_log2, int num_local_experts)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    int tid_0 = (int)tid;
    int lane_1 = (int)lane;
    int warp_2 = (int)warp;
    int bid_3 = (int)bid;
    int nbids = (int)num_bids;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int global_tid = bid_3 * 128 + tid_0;
    int global_tstride = nbids * 128;
    for (int zero_i = global_tid; zero_i < 2 * num_experts; zero_i += global_tstride) {
        expert_counts[zero_i] = 0;
    }
    int global_warp = bid_3 * 4 + warp_2;
    int global_wstride = nbids * 4;
    unsigned int packed_word = 0;
    for (int token = global_warp; token < num_tokens; token += global_wstride) {
        int lane_0 = (int)lane;
        unsigned int out_bits[8];
        int out_idx[8];
        int expert_cand = 0;
        unsigned int raw_bits = 0;
        unsigned int tw_mask = 0;
        int update = 0;
        unsigned int keys32[4];
        unsigned int hi16 = 32768;
        unsigned int all16 = 65535;
        expert_cand = lane_0;
        raw_bits = 65408;
        if (expert_cand < num_experts) {
            raw_bits = __as_u32((float)scores[token * num_experts + expert_cand]) >> 16;
        }
        tw_mask = (((raw_bits & hi16) != 0) ? all16 : hi16);
        keys32[0] = (raw_bits ^ tw_mask) << 16 | (unsigned int)(65535 - expert_cand) & 65535;
        expert_cand = 32 + lane_0;
        raw_bits = 65408;
        if (expert_cand < num_experts) {
            raw_bits = __as_u32((float)scores[token * num_experts + expert_cand]) >> 16;
        }
        tw_mask = (((raw_bits & hi16) != 0) ? all16 : hi16);
        keys32[1] = (raw_bits ^ tw_mask) << 16 | (unsigned int)(65535 - expert_cand) & 65535;
        expert_cand = 64 + lane_0;
        raw_bits = 65408;
        if (expert_cand < num_experts) {
            raw_bits = __as_u32((float)scores[token * num_experts + expert_cand]) >> 16;
        }
        tw_mask = (((raw_bits & hi16) != 0) ? all16 : hi16);
        keys32[2] = (raw_bits ^ tw_mask) << 16 | (unsigned int)(65535 - expert_cand) & 65535;
        expert_cand = 96 + lane_0;
        raw_bits = 65408;
        if (expert_cand < num_experts) {
            raw_bits = __as_u32((float)scores[token * num_experts + expert_cand]) >> 16;
        }
        tw_mask = (((raw_bits & hi16) != 0) ? all16 : hi16);
        keys32[3] = (raw_bits ^ tw_mask) << 16 | (unsigned int)(65535 - expert_cand) & 65535;
        unsigned int pair_min32 = 0;
        unsigned int pair_max32 = 0;
        unsigned int _min_0 = ((keys32[0]) < (keys32[2]) ? (keys32[0]) : (keys32[2]));
        pair_min32 = _min_0;
        unsigned int _max_0 = ((keys32[0]) > (keys32[2]) ? (keys32[0]) : (keys32[2]));
        pair_max32 = _max_0;
        keys32[0] = pair_max32;
        keys32[2] = pair_min32;
        unsigned int _min_1 = ((keys32[1]) < (keys32[3]) ? (keys32[1]) : (keys32[3]));
        pair_min32 = _min_1;
        unsigned int _max_1 = ((keys32[1]) > (keys32[3]) ? (keys32[1]) : (keys32[3]));
        pair_max32 = _max_1;
        keys32[1] = pair_max32;
        keys32[3] = pair_min32;
        unsigned int _min_2 = ((keys32[0]) < (keys32[1]) ? (keys32[0]) : (keys32[1]));
        pair_min32 = _min_2;
        unsigned int _max_2 = ((keys32[0]) > (keys32[1]) ? (keys32[0]) : (keys32[1]));
        pair_max32 = _max_2;
        keys32[0] = pair_max32;
        keys32[1] = pair_min32;
        unsigned int _min_3 = ((keys32[2]) < (keys32[3]) ? (keys32[2]) : (keys32[3]));
        pair_min32 = _min_3;
        unsigned int _max_3 = ((keys32[2]) > (keys32[3]) ? (keys32[2]) : (keys32[3]));
        pair_max32 = _max_3;
        keys32[2] = pair_max32;
        keys32[3] = pair_min32;
        unsigned int _min_4 = ((keys32[1]) < (keys32[2]) ? (keys32[1]) : (keys32[2]));
        pair_min32 = _min_4;
        unsigned int _max_4 = ((keys32[1]) > (keys32[2]) ? (keys32[1]) : (keys32[2]));
        pair_max32 = _max_4;
        keys32[1] = pair_max32;
        keys32[2] = pair_min32;
        unsigned int refill32 = 8323072 | (unsigned int)(65535 - (96 + lane_0)) & 65535;
        unsigned int packed_max32 = 0;
        unsigned int tw16 = 0;
        if (top_k > 0) {
            unsigned int _warp_redux_u32_0;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(keys32[0]));
            packed_max32 = _warp_redux_u32_0;
            out_idx[0] = 65535 - (int)(packed_max32 & 65535);
            tw16 = packed_max32 >> 16;
            tw_mask = (((tw16 & hi16) != 0) ? hi16 : all16);
            out_bits[0] = tw16 ^ tw_mask;
        }
        if (top_k > 1) {
            {
                update = ((packed_max32 == keys32[0]) ? 1 : 0);
                keys32[0] = ((update != 0) ? keys32[1] : keys32[0]);
                keys32[1] = ((update != 0) ? keys32[2] : keys32[1]);
                keys32[2] = ((update != 0) ? keys32[3] : keys32[2]);
                keys32[3] = ((update != 0) ? refill32 : keys32[3]);
            }
            unsigned int _warp_redux_u32_1;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_1) : "r"(keys32[0]));
            packed_max32 = _warp_redux_u32_1;
            out_idx[1] = 65535 - (int)(packed_max32 & 65535);
            tw16 = packed_max32 >> 16;
            tw_mask = (((tw16 & hi16) != 0) ? hi16 : all16);
            out_bits[1] = tw16 ^ tw_mask;
        }
        if (top_k > 2) {
            {
                update = ((packed_max32 == keys32[0]) ? 1 : 0);
                keys32[0] = ((update != 0) ? keys32[1] : keys32[0]);
                keys32[1] = ((update != 0) ? keys32[2] : keys32[1]);
                keys32[2] = ((update != 0) ? keys32[3] : keys32[2]);
                keys32[3] = ((update != 0) ? refill32 : keys32[3]);
            }
            unsigned int _warp_redux_u32_2;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_2) : "r"(keys32[0]));
            packed_max32 = _warp_redux_u32_2;
            out_idx[2] = 65535 - (int)(packed_max32 & 65535);
            tw16 = packed_max32 >> 16;
            tw_mask = (((tw16 & hi16) != 0) ? hi16 : all16);
            out_bits[2] = tw16 ^ tw_mask;
        }
        if (top_k > 3) {
            {
                update = ((packed_max32 == keys32[0]) ? 1 : 0);
                keys32[0] = ((update != 0) ? keys32[1] : keys32[0]);
                keys32[1] = ((update != 0) ? keys32[2] : keys32[1]);
                keys32[2] = ((update != 0) ? keys32[3] : keys32[2]);
                keys32[3] = ((update != 0) ? refill32 : keys32[3]);
            }
            unsigned int _warp_redux_u32_3;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_3) : "r"(keys32[0]));
            packed_max32 = _warp_redux_u32_3;
            out_idx[3] = 65535 - (int)(packed_max32 & 65535);
            tw16 = packed_max32 >> 16;
            tw_mask = (((tw16 & hi16) != 0) ? hi16 : all16);
            out_bits[3] = tw16 ^ tw_mask;
        }
        if (top_k > 4) {
            {
                update = ((packed_max32 == keys32[0]) ? 1 : 0);
                keys32[0] = ((update != 0) ? keys32[1] : keys32[0]);
                keys32[1] = ((update != 0) ? keys32[2] : keys32[1]);
                keys32[2] = ((update != 0) ? keys32[3] : keys32[2]);
                keys32[3] = ((update != 0) ? refill32 : keys32[3]);
            }
            unsigned int _warp_redux_u32_4;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_4) : "r"(keys32[0]));
            packed_max32 = _warp_redux_u32_4;
            out_idx[4] = 65535 - (int)(packed_max32 & 65535);
            tw16 = packed_max32 >> 16;
            tw_mask = (((tw16 & hi16) != 0) ? hi16 : all16);
            out_bits[4] = tw16 ^ tw_mask;
        }
        if (top_k > 5) {
            {
                update = ((packed_max32 == keys32[0]) ? 1 : 0);
                keys32[0] = ((update != 0) ? keys32[1] : keys32[0]);
                keys32[1] = ((update != 0) ? keys32[2] : keys32[1]);
                keys32[2] = ((update != 0) ? keys32[3] : keys32[2]);
                keys32[3] = ((update != 0) ? refill32 : keys32[3]);
            }
            unsigned int _warp_redux_u32_5;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_5) : "r"(keys32[0]));
            packed_max32 = _warp_redux_u32_5;
            out_idx[5] = 65535 - (int)(packed_max32 & 65535);
            tw16 = packed_max32 >> 16;
            tw_mask = (((tw16 & hi16) != 0) ? hi16 : all16);
            out_bits[5] = tw16 ^ tw_mask;
        }
        if (top_k > 6) {
            {
                update = ((packed_max32 == keys32[0]) ? 1 : 0);
                keys32[0] = ((update != 0) ? keys32[1] : keys32[0]);
                keys32[1] = ((update != 0) ? keys32[2] : keys32[1]);
                keys32[2] = ((update != 0) ? keys32[3] : keys32[2]);
                keys32[3] = ((update != 0) ? refill32 : keys32[3]);
            }
            unsigned int _warp_redux_u32_6;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_6) : "r"(keys32[0]));
            packed_max32 = _warp_redux_u32_6;
            out_idx[6] = 65535 - (int)(packed_max32 & 65535);
            tw16 = packed_max32 >> 16;
            tw_mask = (((tw16 & hi16) != 0) ? hi16 : all16);
            out_bits[6] = tw16 ^ tw_mask;
        }
        if (top_k > 7) {
            {
                update = ((packed_max32 == keys32[0]) ? 1 : 0);
                keys32[0] = ((update != 0) ? keys32[1] : keys32[0]);
                keys32[1] = ((update != 0) ? keys32[2] : keys32[1]);
                keys32[2] = ((update != 0) ? keys32[3] : keys32[2]);
                keys32[3] = ((update != 0) ? refill32 : keys32[3]);
            }
            unsigned int _warp_redux_u32_7;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_7) : "r"(keys32[0]));
            packed_max32 = _warp_redux_u32_7;
            out_idx[7] = 65535 - (int)(packed_max32 & 65535);
            tw16 = packed_max32 >> 16;
            tw_mask = (((tw16 & hi16) != 0) ? hi16 : all16);
            out_bits[7] = tw16 ^ tw_mask;
        }
        unsigned int lane_bits = 65408;
        int lane_idx = -1;
        if (lane_0 < top_k) {
            lane_bits = out_bits[lane_0];
            lane_idx = out_idx[lane_0];
        }
        float score_f = -CAKE_INF;
        score_f = __uint_as_float(lane_bits << 16);
        float max_score = -CAKE_INF;
        if (lane_0 < top_k) {
            max_score = ((score_f >= max_score) ? score_f : max_score);
        }
        float _warp_reduce_0 = max_score;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
        max_score = _warp_reduce_0;
        float sum_score = 0.0f;
        float new_score = 0.0f;
        if (lane_0 < top_k) {
            new_score = score_f - max_score;
            float _expf_0 = __expf(new_score);
            new_score = _expf_0;
            sum_score = sum_score + new_score;
        }
        float _warp_reduce_1 = sum_score;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
        sum_score = _warp_reduce_1;
        float quotient = 0.0f;
        unsigned int softmax_bits = 0;
        if (lane_0 < top_k) {
            quotient = new_score / sum_score;
            __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(quotient, 0.0f));
            softmax_bits = __as_u32(_bf16x2_0) & 65535;
        }
        if (lane_1 < top_k) {
            packed_word = (unsigned int)lane_idx << 16 | softmax_bits & 65535;
            topk_packed[token * top_k + lane_1] = (int)packed_word;
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
