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
#define SMEM_SMEM_OFFSET8_OFF 0
#define SMEM_SMEM_OFFSET8_STAGE_BYTES 512
#define SMEM_SMEM_OFFSET8_STRIDE 512
#define SMEM_SMEM_KIDX_OFF 512
#define SMEM_SMEM_KIDX_STAGE_BYTES 512
#define SMEM_SMEM_KIDX_STRIDE 512
#define SMEM_WARP_TOTALS_OFF 1024
#define SMEM_WARP_TOTALS_STAGE_BYTES 128
#define SMEM_WARP_TOTALS_STRIDE 128
#define SMEM_TOTAL 1152
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
kernel_cake_stepfun_moe_0700c90d445cc01006c5(float* __restrict__ scores, __nv_bfloat16* __restrict__ topk_weights, int* __restrict__ topk_packed, int* __restrict__ expert_counts, int* __restrict__ permuted_idx_size, int* __restrict__ expanded_idx_to_permuted_idx, int* __restrict__ permuted_idx_to_token_idx, int* __restrict__ cta_idx_xy_to_batch_idx, int* __restrict__ cta_idx_xy_to_mn_limit, int* __restrict__ num_non_exiting_ctas, int num_tokens, int num_experts, int top_k, int padding_log2, int tile_tokens_dim, int local_experts_start_idx, int local_experts_stride_log2, int num_local_experts)
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
    int8_t* smem_offset8 = reinterpret_cast<int8_t*>(smem_raw + 0);
    const int smem_offset8_addr = smem + 0;
    int8_t* smem_kidx = reinterpret_cast<int8_t*>(smem_raw + 512);
    const int smem_kidx_addr = smem + 512;
    int* warp_totals = reinterpret_cast<int*>(smem_raw + 1024);
    const int warp_totals_addr = smem + 1024;

    // === Task calls (dependency order) ===
    int tid_0 = (int)tid;
    int lane_1 = (int)lane;
    int warp_2 = (int)warp;
    int neg_four[4];
    neg_four[0] = -1;
    neg_four[1] = -1;
    neg_four[2] = -1;
    neg_four[3] = -1;
    for (int slot_init = tid_0; slot_init < 512; slot_init += 128) {
        smem_offset8[slot_init] = (int8_t)-1;
        smem_kidx[slot_init] = (int8_t)-1;
    }
    __syncthreads();
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int expanded_p = 0;
    int expert_p = 0;
    if (warp_2 < num_tokens) {
        int lane_0 = (int)lane;
        unsigned int out_bits[8];
        int out_idx[8];
        int expert_cand = 0;
        unsigned int raw_bits = 0;
        unsigned int tw_mask = 0;
        int update = 0;
        unsigned long long keys[4];
        unsigned int sign32 = 1;
        sign32 = sign32 << 31;
        unsigned int all32 = 0;
        all32 = ~all32;
        expert_cand = lane_0;
        raw_bits = __as_u32(-CAKE_INF);
        if (expert_cand < num_experts) {
            raw_bits = __as_u32(scores[warp_2 * num_experts + expert_cand]);
        }
        tw_mask = (((raw_bits & sign32) != 0) ? all32 : sign32);
        keys[0] = (unsigned long long)(raw_bits ^ tw_mask) << 32 | (unsigned long long)((unsigned int)(65535 - expert_cand) & 65535);
        expert_cand = 32 + lane_0;
        raw_bits = __as_u32(-CAKE_INF);
        if (expert_cand < num_experts) {
            raw_bits = __as_u32(scores[warp_2 * num_experts + expert_cand]);
        }
        tw_mask = (((raw_bits & sign32) != 0) ? all32 : sign32);
        keys[1] = (unsigned long long)(raw_bits ^ tw_mask) << 32 | (unsigned long long)((unsigned int)(65535 - expert_cand) & 65535);
        expert_cand = 64 + lane_0;
        raw_bits = __as_u32(-CAKE_INF);
        if (expert_cand < num_experts) {
            raw_bits = __as_u32(scores[warp_2 * num_experts + expert_cand]);
        }
        tw_mask = (((raw_bits & sign32) != 0) ? all32 : sign32);
        keys[2] = (unsigned long long)(raw_bits ^ tw_mask) << 32 | (unsigned long long)((unsigned int)(65535 - expert_cand) & 65535);
        expert_cand = 96 + lane_0;
        raw_bits = __as_u32(-CAKE_INF);
        if (expert_cand < num_experts) {
            raw_bits = __as_u32(scores[warp_2 * num_experts + expert_cand]);
        }
        tw_mask = (((raw_bits & sign32) != 0) ? all32 : sign32);
        keys[3] = (unsigned long long)(raw_bits ^ tw_mask) << 32 | (unsigned long long)((unsigned int)(65535 - expert_cand) & 65535);
        unsigned long long zero_u64 = (unsigned long long)0;
        unsigned long long pair_min64 = zero_u64;
        unsigned long long pair_max64 = zero_u64;
        unsigned long long _min_0 = ((keys[0]) < (keys[2]) ? (keys[0]) : (keys[2]));
        pair_min64 = _min_0;
        unsigned long long _max_0 = ((keys[0]) > (keys[2]) ? (keys[0]) : (keys[2]));
        pair_max64 = _max_0;
        keys[0] = pair_max64;
        keys[2] = pair_min64;
        unsigned long long _min_1 = ((keys[1]) < (keys[3]) ? (keys[1]) : (keys[3]));
        pair_min64 = _min_1;
        unsigned long long _max_1 = ((keys[1]) > (keys[3]) ? (keys[1]) : (keys[3]));
        pair_max64 = _max_1;
        keys[1] = pair_max64;
        keys[3] = pair_min64;
        unsigned long long _min_2 = ((keys[0]) < (keys[1]) ? (keys[0]) : (keys[1]));
        pair_min64 = _min_2;
        unsigned long long _max_2 = ((keys[0]) > (keys[1]) ? (keys[0]) : (keys[1]));
        pair_max64 = _max_2;
        keys[0] = pair_max64;
        keys[1] = pair_min64;
        unsigned long long _min_3 = ((keys[2]) < (keys[3]) ? (keys[2]) : (keys[3]));
        pair_min64 = _min_3;
        unsigned long long _max_3 = ((keys[2]) > (keys[3]) ? (keys[2]) : (keys[3]));
        pair_max64 = _max_3;
        keys[2] = pair_max64;
        keys[3] = pair_min64;
        unsigned long long _min_4 = ((keys[1]) < (keys[2]) ? (keys[1]) : (keys[2]));
        pair_min64 = _min_4;
        unsigned long long _max_4 = ((keys[1]) > (keys[2]) ? (keys[1]) : (keys[2]));
        pair_max64 = _max_4;
        keys[1] = pair_max64;
        keys[2] = pair_min64;
        unsigned long long refill64 = (unsigned long long)8388607 << 32 | (unsigned long long)((unsigned int)(65535 - (96 + lane_0)) & 65535);
        unsigned long long packed_max64 = zero_u64;
        unsigned int hi0 = 0;
        unsigned int lo0 = 0;
        unsigned int max_hi = 0;
        unsigned int lo_contrib = 0;
        unsigned int max_lo = 0;
        unsigned int zero_u32 = 0;
        if (top_k > 0) {
            hi0 = (unsigned int)(keys[0] >> 32);
            lo0 = (unsigned int)keys[0];
            unsigned int _warp_redux_u32_0;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(hi0));
            max_hi = _warp_redux_u32_0;
            lo_contrib = ((hi0 == max_hi) ? lo0 : zero_u32);
            unsigned int _warp_redux_u32_1;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_1) : "r"(lo_contrib));
            max_lo = _warp_redux_u32_1;
            packed_max64 = (unsigned long long)max_hi << 32 | (unsigned long long)max_lo;
            out_idx[0] = 65535 - (int)(max_lo & 65535);
            tw_mask = (((max_hi & sign32) != 0) ? sign32 : all32);
            out_bits[0] = max_hi ^ tw_mask;
        }
        if (top_k > 1) {
            {
                update = ((packed_max64 == keys[0]) ? 1 : 0);
                keys[0] = ((update != 0) ? keys[1] : keys[0]);
                keys[1] = ((update != 0) ? keys[2] : keys[1]);
                keys[2] = ((update != 0) ? keys[3] : keys[2]);
                keys[3] = ((update != 0) ? refill64 : keys[3]);
            }
            hi0 = (unsigned int)(keys[0] >> 32);
            lo0 = (unsigned int)keys[0];
            unsigned int _warp_redux_u32_2;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_2) : "r"(hi0));
            max_hi = _warp_redux_u32_2;
            lo_contrib = ((hi0 == max_hi) ? lo0 : zero_u32);
            unsigned int _warp_redux_u32_3;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_3) : "r"(lo_contrib));
            max_lo = _warp_redux_u32_3;
            packed_max64 = (unsigned long long)max_hi << 32 | (unsigned long long)max_lo;
            out_idx[1] = 65535 - (int)(max_lo & 65535);
            tw_mask = (((max_hi & sign32) != 0) ? sign32 : all32);
            out_bits[1] = max_hi ^ tw_mask;
        }
        if (top_k > 2) {
            {
                update = ((packed_max64 == keys[0]) ? 1 : 0);
                keys[0] = ((update != 0) ? keys[1] : keys[0]);
                keys[1] = ((update != 0) ? keys[2] : keys[1]);
                keys[2] = ((update != 0) ? keys[3] : keys[2]);
                keys[3] = ((update != 0) ? refill64 : keys[3]);
            }
            hi0 = (unsigned int)(keys[0] >> 32);
            lo0 = (unsigned int)keys[0];
            unsigned int _warp_redux_u32_4;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_4) : "r"(hi0));
            max_hi = _warp_redux_u32_4;
            lo_contrib = ((hi0 == max_hi) ? lo0 : zero_u32);
            unsigned int _warp_redux_u32_5;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_5) : "r"(lo_contrib));
            max_lo = _warp_redux_u32_5;
            packed_max64 = (unsigned long long)max_hi << 32 | (unsigned long long)max_lo;
            out_idx[2] = 65535 - (int)(max_lo & 65535);
            tw_mask = (((max_hi & sign32) != 0) ? sign32 : all32);
            out_bits[2] = max_hi ^ tw_mask;
        }
        if (top_k > 3) {
            {
                update = ((packed_max64 == keys[0]) ? 1 : 0);
                keys[0] = ((update != 0) ? keys[1] : keys[0]);
                keys[1] = ((update != 0) ? keys[2] : keys[1]);
                keys[2] = ((update != 0) ? keys[3] : keys[2]);
                keys[3] = ((update != 0) ? refill64 : keys[3]);
            }
            hi0 = (unsigned int)(keys[0] >> 32);
            lo0 = (unsigned int)keys[0];
            unsigned int _warp_redux_u32_6;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_6) : "r"(hi0));
            max_hi = _warp_redux_u32_6;
            lo_contrib = ((hi0 == max_hi) ? lo0 : zero_u32);
            unsigned int _warp_redux_u32_7;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_7) : "r"(lo_contrib));
            max_lo = _warp_redux_u32_7;
            packed_max64 = (unsigned long long)max_hi << 32 | (unsigned long long)max_lo;
            out_idx[3] = 65535 - (int)(max_lo & 65535);
            tw_mask = (((max_hi & sign32) != 0) ? sign32 : all32);
            out_bits[3] = max_hi ^ tw_mask;
        }
        if (top_k > 4) {
            {
                update = ((packed_max64 == keys[0]) ? 1 : 0);
                keys[0] = ((update != 0) ? keys[1] : keys[0]);
                keys[1] = ((update != 0) ? keys[2] : keys[1]);
                keys[2] = ((update != 0) ? keys[3] : keys[2]);
                keys[3] = ((update != 0) ? refill64 : keys[3]);
            }
            hi0 = (unsigned int)(keys[0] >> 32);
            lo0 = (unsigned int)keys[0];
            unsigned int _warp_redux_u32_8;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_8) : "r"(hi0));
            max_hi = _warp_redux_u32_8;
            lo_contrib = ((hi0 == max_hi) ? lo0 : zero_u32);
            unsigned int _warp_redux_u32_9;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_9) : "r"(lo_contrib));
            max_lo = _warp_redux_u32_9;
            packed_max64 = (unsigned long long)max_hi << 32 | (unsigned long long)max_lo;
            out_idx[4] = 65535 - (int)(max_lo & 65535);
            tw_mask = (((max_hi & sign32) != 0) ? sign32 : all32);
            out_bits[4] = max_hi ^ tw_mask;
        }
        if (top_k > 5) {
            {
                update = ((packed_max64 == keys[0]) ? 1 : 0);
                keys[0] = ((update != 0) ? keys[1] : keys[0]);
                keys[1] = ((update != 0) ? keys[2] : keys[1]);
                keys[2] = ((update != 0) ? keys[3] : keys[2]);
                keys[3] = ((update != 0) ? refill64 : keys[3]);
            }
            hi0 = (unsigned int)(keys[0] >> 32);
            lo0 = (unsigned int)keys[0];
            unsigned int _warp_redux_u32_10;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_10) : "r"(hi0));
            max_hi = _warp_redux_u32_10;
            lo_contrib = ((hi0 == max_hi) ? lo0 : zero_u32);
            unsigned int _warp_redux_u32_11;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_11) : "r"(lo_contrib));
            max_lo = _warp_redux_u32_11;
            packed_max64 = (unsigned long long)max_hi << 32 | (unsigned long long)max_lo;
            out_idx[5] = 65535 - (int)(max_lo & 65535);
            tw_mask = (((max_hi & sign32) != 0) ? sign32 : all32);
            out_bits[5] = max_hi ^ tw_mask;
        }
        if (top_k > 6) {
            {
                update = ((packed_max64 == keys[0]) ? 1 : 0);
                keys[0] = ((update != 0) ? keys[1] : keys[0]);
                keys[1] = ((update != 0) ? keys[2] : keys[1]);
                keys[2] = ((update != 0) ? keys[3] : keys[2]);
                keys[3] = ((update != 0) ? refill64 : keys[3]);
            }
            hi0 = (unsigned int)(keys[0] >> 32);
            lo0 = (unsigned int)keys[0];
            unsigned int _warp_redux_u32_12;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_12) : "r"(hi0));
            max_hi = _warp_redux_u32_12;
            lo_contrib = ((hi0 == max_hi) ? lo0 : zero_u32);
            unsigned int _warp_redux_u32_13;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_13) : "r"(lo_contrib));
            max_lo = _warp_redux_u32_13;
            packed_max64 = (unsigned long long)max_hi << 32 | (unsigned long long)max_lo;
            out_idx[6] = 65535 - (int)(max_lo & 65535);
            tw_mask = (((max_hi & sign32) != 0) ? sign32 : all32);
            out_bits[6] = max_hi ^ tw_mask;
        }
        if (top_k > 7) {
            {
                update = ((packed_max64 == keys[0]) ? 1 : 0);
                keys[0] = ((update != 0) ? keys[1] : keys[0]);
                keys[1] = ((update != 0) ? keys[2] : keys[1]);
                keys[2] = ((update != 0) ? keys[3] : keys[2]);
                keys[3] = ((update != 0) ? refill64 : keys[3]);
            }
            hi0 = (unsigned int)(keys[0] >> 32);
            lo0 = (unsigned int)keys[0];
            unsigned int _warp_redux_u32_14;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_14) : "r"(hi0));
            max_hi = _warp_redux_u32_14;
            lo_contrib = ((hi0 == max_hi) ? lo0 : zero_u32);
            unsigned int _warp_redux_u32_15;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_15) : "r"(lo_contrib));
            max_lo = _warp_redux_u32_15;
            packed_max64 = (unsigned long long)max_hi << 32 | (unsigned long long)max_lo;
            out_idx[7] = 65535 - (int)(max_lo & 65535);
            tw_mask = (((max_hi & sign32) != 0) ? sign32 : all32);
            out_bits[7] = max_hi ^ tw_mask;
        }
        unsigned int lane_bits = 65408;
        lane_bits = __as_u32(-CAKE_INF);
        int lane_idx = -1;
        if (lane_0 < top_k) {
            lane_bits = out_bits[lane_0];
            lane_idx = out_idx[lane_0];
        }
        float score_f = -CAKE_INF;
        score_f = __uint_as_float(lane_bits);
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
            softmax_bits = __as_u32(quotient);
        }
        if (lane_1 < top_k) {
            smem_kidx[warp_2 * 128 + lane_idx] = (int8_t)lane_1;
            topk_weights[warp_2 * top_k + lane_1] = (__nv_bfloat16)__uint_as_float(softmax_bits);
        }
    }
    __syncthreads();
    int expert = tid_0;
    int local_idx = expert - local_experts_start_idx;
    int extent = num_local_experts << local_experts_stride_log2;
    int stride_mask = (1 << local_experts_stride_log2) - 1;
    int is_local = ((local_idx >= 0 && local_idx < extent && (local_idx & stride_mask) == 0) ? 1 : 0);
    int is_local_3 = is_local;
    int acc_count = 0;
    int slot_j = 0;
    int kidx_j = 0;
    if (is_local_3 != 0) {
        slot_j = expert;
        kidx_j = (int)smem_kidx[slot_j];
        if (kidx_j >= 0) {
            smem_offset8[slot_j] = (int8_t)acc_count;
            acc_count = acc_count + 1;
        }
        slot_j = 128 + expert;
        kidx_j = (int)smem_kidx[slot_j];
        if (kidx_j >= 0) {
            smem_offset8[slot_j] = (int8_t)acc_count;
            acc_count = acc_count + 1;
        }
        slot_j = 256 + expert;
        kidx_j = (int)smem_kidx[slot_j];
        if (kidx_j >= 0) {
            smem_offset8[slot_j] = (int8_t)acc_count;
            acc_count = acc_count + 1;
        }
        slot_j = 384 + expert;
        kidx_j = (int)smem_kidx[slot_j];
        if (kidx_j >= 0) {
            smem_offset8[slot_j] = (int8_t)acc_count;
            acc_count = acc_count + 1;
        }
    }
    __syncthreads();
    int num_cta = (acc_count + tile_tokens_dim - 1) / tile_tokens_dim;
    if (padding_log2 > 0) {
        num_cta = acc_count + (1 << padding_log2) - 1 >> padding_log2;
    }
    int num_cta_4 = num_cta;
    int inclusive = num_cta_4;
    int lane_s = (int)lane;
    int peer = 0;
    int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, inclusive, 1, 32);
    peer = _shfl_up_0;
    if (lane_s >= 1) {
        inclusive = inclusive + peer;
    }
    int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, inclusive, 2, 32);
    peer = _shfl_up_1;
    if (lane_s >= 2) {
        inclusive = inclusive + peer;
    }
    int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, inclusive, 4, 32);
    peer = _shfl_up_2;
    if (lane_s >= 4) {
        inclusive = inclusive + peer;
    }
    int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, inclusive, 8, 32);
    peer = _shfl_up_3;
    if (lane_s >= 8) {
        inclusive = inclusive + peer;
    }
    int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, inclusive, 16, 32);
    peer = _shfl_up_4;
    if (lane_s >= 16) {
        inclusive = inclusive + peer;
    }
    if (lane_s == 31) {
        warp_totals[warp] = inclusive;
    }
    __syncthreads();
    int warp_prefix = 0;
    int block_total = 0;
    int wt = 0;
    wt = warp_totals[0];
    if (warp > 0) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[1];
    if (warp > 1) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[2];
    if (warp > 2) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[3];
    if (warp > 3) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    int exclusive = inclusive - num_cta_4 + warp_prefix;
    __syncthreads();
    int scaled = num_cta_4 * tile_tokens_dim;
    if (padding_log2 > 0) {
        scaled = num_cta_4 << padding_log2;
    }
    int padded_count = scaled;
    int inclusive_5 = padded_count;
    int lane_s_6 = (int)lane;
    int peer_7 = 0;
    int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, inclusive_5, 1, 32);
    peer_7 = _shfl_up_5;
    if (lane_s_6 >= 1) {
        inclusive_5 = inclusive_5 + peer_7;
    }
    int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, inclusive_5, 2, 32);
    peer_7 = _shfl_up_6;
    if (lane_s_6 >= 2) {
        inclusive_5 = inclusive_5 + peer_7;
    }
    int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, inclusive_5, 4, 32);
    peer_7 = _shfl_up_7;
    if (lane_s_6 >= 4) {
        inclusive_5 = inclusive_5 + peer_7;
    }
    int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, inclusive_5, 8, 32);
    peer_7 = _shfl_up_8;
    if (lane_s_6 >= 8) {
        inclusive_5 = inclusive_5 + peer_7;
    }
    int _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, inclusive_5, 16, 32);
    peer_7 = _shfl_up_9;
    if (lane_s_6 >= 16) {
        inclusive_5 = inclusive_5 + peer_7;
    }
    if (lane_s_6 == 31) {
        warp_totals[warp] = inclusive_5;
    }
    __syncthreads();
    int warp_prefix_8 = 0;
    int block_total_9 = 0;
    int wt_10 = 0;
    wt_10 = warp_totals[0];
    if (warp > 0) {
        warp_prefix_8 = warp_prefix_8 + wt_10;
    }
    block_total_9 = block_total_9 + wt_10;
    wt_10 = warp_totals[1];
    if (warp > 1) {
        warp_prefix_8 = warp_prefix_8 + wt_10;
    }
    block_total_9 = block_total_9 + wt_10;
    wt_10 = warp_totals[2];
    if (warp > 2) {
        warp_prefix_8 = warp_prefix_8 + wt_10;
    }
    block_total_9 = block_total_9 + wt_10;
    wt_10 = warp_totals[3];
    if (warp > 3) {
        warp_prefix_8 = warp_prefix_8 + wt_10;
    }
    block_total_9 = block_total_9 + wt_10;
    int exclusive_11 = inclusive_5 - padded_count + warp_prefix_8;
    __syncthreads();
    if (is_local_3 != 0) {
        int local_idx_0 = expert - local_experts_start_idx >> local_experts_stride_log2;
        int mn_limit1 = 0;
        int mn_limit2 = 0;
        int mn_limit = 0;
        for (int cta = 0; cta < num_cta_4; cta++) {
            cta_idx_xy_to_batch_idx[exclusive + cta] = local_idx_0;
            int scaled_0 = (exclusive + cta + 1) * tile_tokens_dim;
            if (padding_log2 > 0) {
                scaled_0 = exclusive + cta + 1 << padding_log2;
            }
            mn_limit1 = scaled_0;
            int scaled_1 = exclusive * tile_tokens_dim;
            if (padding_log2 > 0) {
                scaled_1 = exclusive << padding_log2;
            }
            mn_limit2 = scaled_1 + acc_count;
            int _min_5 = ((mn_limit1) < (mn_limit2) ? (mn_limit1) : (mn_limit2));
            mn_limit = _min_5;
            cta_idx_xy_to_mn_limit[exclusive + cta] = mn_limit;
            int _min_6 = ((mn_limit1) < (mn_limit + (-mn_limit & 3)) ? (mn_limit1) : (mn_limit + (-mn_limit & 3)));
            int head_end = _min_6;
            for (int head = mn_limit; head < head_end; head++) {
                permuted_idx_to_token_idx[head] = -1;
            }
            int _max_5 = ((mn_limit1 - head_end) > (0) ? (mn_limit1 - head_end) : (0));
            int body_len = _max_5;
            int tail_begin = head_end + (body_len >> 2 << 2);
            for (int body = head_end; body < tail_begin; body += 4) {
                {
                    int4 _iv4 = make_int4(neg_four[0 + 0], neg_four[0 + 1], neg_four[0 + 2], neg_four[0 + 3]);
                    *reinterpret_cast<int4*>(permuted_idx_to_token_idx + body) = _iv4;
                }
            }
            for (int tail = tail_begin; tail < mn_limit1; tail++) {
                permuted_idx_to_token_idx[tail] = -1;
            }
        }
    }
    if (tid_0 == 0) {
        int scaled_0_1 = block_total * tile_tokens_dim;
        if (padding_log2 > 0) {
            scaled_0_1 = block_total << padding_log2;
        }
        int padded_rows = scaled_0_1;
        permuted_idx_size[0] = padded_rows;
        num_non_exiting_ctas[0] = block_total;
    }
    int slot_p = 0;
    int k_of_slot = 0;
    int expanded_q = 0;
    int permuted_q = -1;
    for (int token_p = 0; token_p < num_tokens; token_p++) {
        slot_p = token_p * 128 + expert;
        k_of_slot = (int)smem_kidx[slot_p];
        if (k_of_slot >= 0) {
            expanded_q = token_p * top_k + k_of_slot;
            permuted_q = -1;
            if (is_local_3 != 0) {
                permuted_q = exclusive_11 + (int)smem_offset8[slot_p];
            }
            expanded_idx_to_permuted_idx[expanded_q] = permuted_q;
            if (is_local_3 != 0) {
                permuted_idx_to_token_idx[permuted_q] = token_p;
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
