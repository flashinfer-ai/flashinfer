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
#define SMEM_SMEM_PACKED_OFF 0
#define SMEM_SMEM_PACKED_STAGE_BYTES 1024
#define SMEM_SMEM_PACKED_STRIDE 1024
#define SMEM_SMEM_EXPERT_COUNT_OFF 1024
#define SMEM_SMEM_EXPERT_COUNT_STAGE_BYTES 512
#define SMEM_SMEM_EXPERT_COUNT_STRIDE 512
#define SMEM_SMEM_EXPERT_OFFSET_OFF 1536
#define SMEM_SMEM_EXPERT_OFFSET_STAGE_BYTES 512
#define SMEM_SMEM_EXPERT_OFFSET_STRIDE 512
#define SMEM_WARP_TOTALS_OFF 2048
#define SMEM_WARP_TOTALS_STAGE_BYTES 128
#define SMEM_WARP_TOTALS_STRIDE 128
#define SMEM_TOTAL 2176
#define THREADS 1024

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

__global__ __launch_bounds__(1024) __cluster_dims__(8,1,1) void
kernel_cake_stepfun_moe_a02c424cd4f9b2a2288d(__nv_bfloat16* __restrict__ scores, __nv_bfloat16* __restrict__ topk_weights, int* __restrict__ topk_packed, int* __restrict__ expert_counts, int* __restrict__ permuted_idx_size, int* __restrict__ expanded_idx_to_permuted_idx, int* __restrict__ permuted_idx_to_token_idx, int* __restrict__ cta_idx_xy_to_batch_idx, int* __restrict__ cta_idx_xy_to_mn_limit, int* __restrict__ num_non_exiting_ctas, int num_tokens, int num_experts, int top_k, int padding_log2, int tile_tokens_dim, int local_experts_start_idx, int local_experts_stride_log2, int num_local_experts)
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
    const unsigned int clusters_x = gridDim.x / 8;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 8;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    unsigned int* smem_packed = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int smem_packed_addr = smem + 0;
    unsigned int* smem_expert_count = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int smem_expert_count_addr = smem + 1024;
    int* smem_expert_offset = reinterpret_cast<int*>(smem_raw + 1536);
    const int smem_expert_offset_addr = smem + 1536;
    int* warp_totals = reinterpret_cast<int*>(smem_raw + 2048);
    const int warp_totals_addr = smem + 2048;

    // === Task calls (dependency order) ===
    int rank = (int)cta_rank;
    int tid_0 = (int)tid;
    int lane_1 = (int)lane;
    int warp_2 = (int)warp;
    int neg_four[4];
    neg_four[0] = -1;
    neg_four[1] = -1;
    neg_four[2] = -1;
    neg_four[3] = -1;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int warp_token = rank * 32 + warp_2;
    if (warp_token < num_tokens) {
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
            raw_bits = __as_u32((float)scores[warp_token * num_experts + expert_cand]) >> 16;
        }
        tw_mask = (((raw_bits & hi16) != 0) ? all16 : hi16);
        keys32[0] = (raw_bits ^ tw_mask) << 16 | (unsigned int)(65535 - expert_cand) & 65535;
        expert_cand = 32 + lane_0;
        raw_bits = 65408;
        if (expert_cand < num_experts) {
            raw_bits = __as_u32((float)scores[warp_token * num_experts + expert_cand]) >> 16;
        }
        tw_mask = (((raw_bits & hi16) != 0) ? all16 : hi16);
        keys32[1] = (raw_bits ^ tw_mask) << 16 | (unsigned int)(65535 - expert_cand) & 65535;
        expert_cand = 64 + lane_0;
        raw_bits = 65408;
        if (expert_cand < num_experts) {
            raw_bits = __as_u32((float)scores[warp_token * num_experts + expert_cand]) >> 16;
        }
        tw_mask = (((raw_bits & hi16) != 0) ? all16 : hi16);
        keys32[2] = (raw_bits ^ tw_mask) << 16 | (unsigned int)(65535 - expert_cand) & 65535;
        expert_cand = 96 + lane_0;
        raw_bits = 65408;
        if (expert_cand < num_experts) {
            raw_bits = __as_u32((float)scores[warp_token * num_experts + expert_cand]) >> 16;
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
            int slot = warp_2 * top_k + lane_1;
            smem_packed[slot] = (unsigned int)lane_idx << 16 | softmax_bits;
        }
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (tid_0 < num_experts) {
        smem_expert_count[tid_0] = 0;
    }
    __syncthreads();
    int cluster_tid = 1024 * rank + tid_0;
    int expanded_size = num_tokens * top_k;
    int exp_idx[8];
    int exp_off[8];
    int done = 0;
    int expanded = 0;
    if (done == 0) {
        if (expanded_size >= 32768) {
            expanded = cluster_tid;
            int idx_ii = 0;
            unsigned int word0 = 0;
            unsigned int word1 = 0;
            unsigned int remote = 0;
            int src_rank = 0;
            int src_slot = 0;
            int packed_word = 0;
            src_rank = expanded / (32 * top_k);
            src_slot = expanded % (32 * top_k);
            uint32_t _mapa_0;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_0) : "r"(smem_packed_addr), "r"(src_rank));
            remote = _mapa_0;
            unsigned int _cluster_ld_0;
            asm volatile(
                "ld.shared::cluster.u32 %0, [%1];"
                : "=r"(_cluster_ld_0) : "r"(remote + (unsigned int)(src_slot * 4)) : "memory");
            word0 = _cluster_ld_0;
            idx_ii = (int)word0 >> 16;
            word0 = word0 & 65535;
            exp_idx[0] = idx_ii;
            int local_idx = idx_ii - local_experts_start_idx;
            int extent = num_local_experts << local_experts_stride_log2;
            int stride_mask = (1 << local_experts_stride_log2) - 1;
            int is_local = ((local_idx >= 0 && local_idx < extent && (local_idx & stride_mask) == 0) ? 1 : 0);
            int is_local_ii = is_local;
            exp_off[0] = 0;
            if (is_local_ii != 0) {
                uint32_t _shared_atomic_old_0;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_0) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[0] = (int)_shared_atomic_old_0;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0 << 16);
            expanded = cluster_tid + 8192;
            int idx_ii_0 = 0;
            unsigned int word0_1 = 0;
            unsigned int word1_2 = 0;
            unsigned int remote_3 = 0;
            int src_rank_4 = 0;
            int src_slot_5 = 0;
            int packed_word_6 = 0;
            src_rank_4 = expanded / (32 * top_k);
            src_slot_5 = expanded % (32 * top_k);
            uint32_t _mapa_1;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_1) : "r"(smem_packed_addr), "r"(src_rank_4));
            remote_3 = _mapa_1;
            unsigned int _cluster_ld_1;
            asm volatile(
                "ld.shared::cluster.u32 %0, [%1];"
                : "=r"(_cluster_ld_1) : "r"(remote_3 + (unsigned int)(src_slot_5 * 4)) : "memory");
            word0_1 = _cluster_ld_1;
            idx_ii_0 = (int)word0_1 >> 16;
            word0_1 = word0_1 & 65535;
            exp_idx[1] = idx_ii_0;
            int local_idx_7 = idx_ii_0 - local_experts_start_idx;
            int extent_8 = num_local_experts << local_experts_stride_log2;
            int stride_mask_9 = (1 << local_experts_stride_log2) - 1;
            int is_local_10 = ((local_idx_7 >= 0 && local_idx_7 < extent_8 && (local_idx_7 & stride_mask_9) == 0) ? 1 : 0);
            int is_local_ii_11 = is_local_10;
            exp_off[1] = 0;
            if (is_local_ii_11 != 0) {
                uint32_t _shared_atomic_old_1;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_1) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_0)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[1] = (int)_shared_atomic_old_1;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_1 << 16);
            expanded = cluster_tid + 16384;
            int idx_ii_12 = 0;
            unsigned int word0_13 = 0;
            unsigned int word1_14 = 0;
            unsigned int remote_15 = 0;
            int src_rank_16 = 0;
            int src_slot_17 = 0;
            int packed_word_18 = 0;
            src_rank_16 = expanded / (32 * top_k);
            src_slot_17 = expanded % (32 * top_k);
            uint32_t _mapa_2;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_2) : "r"(smem_packed_addr), "r"(src_rank_16));
            remote_15 = _mapa_2;
            unsigned int _cluster_ld_2;
            asm volatile(
                "ld.shared::cluster.u32 %0, [%1];"
                : "=r"(_cluster_ld_2) : "r"(remote_15 + (unsigned int)(src_slot_17 * 4)) : "memory");
            word0_13 = _cluster_ld_2;
            idx_ii_12 = (int)word0_13 >> 16;
            word0_13 = word0_13 & 65535;
            exp_idx[2] = idx_ii_12;
            int local_idx_19 = idx_ii_12 - local_experts_start_idx;
            int extent_20 = num_local_experts << local_experts_stride_log2;
            int stride_mask_21 = (1 << local_experts_stride_log2) - 1;
            int is_local_22 = ((local_idx_19 >= 0 && local_idx_19 < extent_20 && (local_idx_19 & stride_mask_21) == 0) ? 1 : 0);
            int is_local_ii_23 = is_local_22;
            exp_off[2] = 0;
            if (is_local_ii_23 != 0) {
                uint32_t _shared_atomic_old_2;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_2) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_12)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[2] = (int)_shared_atomic_old_2;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_13 << 16);
            expanded = cluster_tid + 24576;
            int idx_ii_24 = 0;
            unsigned int word0_25 = 0;
            unsigned int word1_26 = 0;
            unsigned int remote_27 = 0;
            int src_rank_28 = 0;
            int src_slot_29 = 0;
            int packed_word_30 = 0;
            src_rank_28 = expanded / (32 * top_k);
            src_slot_29 = expanded % (32 * top_k);
            uint32_t _mapa_3;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_3) : "r"(smem_packed_addr), "r"(src_rank_28));
            remote_27 = _mapa_3;
            unsigned int _cluster_ld_3;
            asm volatile(
                "ld.shared::cluster.u32 %0, [%1];"
                : "=r"(_cluster_ld_3) : "r"(remote_27 + (unsigned int)(src_slot_29 * 4)) : "memory");
            word0_25 = _cluster_ld_3;
            idx_ii_24 = (int)word0_25 >> 16;
            word0_25 = word0_25 & 65535;
            exp_idx[3] = idx_ii_24;
            int local_idx_31 = idx_ii_24 - local_experts_start_idx;
            int extent_32 = num_local_experts << local_experts_stride_log2;
            int stride_mask_33 = (1 << local_experts_stride_log2) - 1;
            int is_local_34 = ((local_idx_31 >= 0 && local_idx_31 < extent_32 && (local_idx_31 & stride_mask_33) == 0) ? 1 : 0);
            int is_local_ii_35 = is_local_34;
            exp_off[3] = 0;
            if (is_local_ii_35 != 0) {
                uint32_t _shared_atomic_old_3;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_3) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_24)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[3] = (int)_shared_atomic_old_3;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_25 << 16);
        } else {
            if (done == 0) {
                expanded = cluster_tid;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_1 = 0;
                    unsigned int word0_2 = 0;
                    unsigned int word1_1 = 0;
                    unsigned int remote_1 = 0;
                    int src_rank_1 = 0;
                    int src_slot_1 = 0;
                    int packed_word_1 = 0;
                    src_rank_1 = expanded / (32 * top_k);
                    src_slot_1 = expanded % (32 * top_k);
                    uint32_t _mapa_4;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mapa_4) : "r"(smem_packed_addr), "r"(src_rank_1));
                    remote_1 = _mapa_4;
                    unsigned int _cluster_ld_4;
                    asm volatile(
                        "ld.shared::cluster.u32 %0, [%1];"
                        : "=r"(_cluster_ld_4) : "r"(remote_1 + (unsigned int)(src_slot_1 * 4)) : "memory");
                    word0_2 = _cluster_ld_4;
                    idx_ii_1 = (int)word0_2 >> 16;
                    word0_2 = word0_2 & 65535;
                    exp_idx[0] = idx_ii_1;
                    int local_idx_1 = idx_ii_1 - local_experts_start_idx;
                    int extent_1 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_1 = (1 << local_experts_stride_log2) - 1;
                    int is_local_1 = ((local_idx_1 >= 0 && local_idx_1 < extent_1 && (local_idx_1 & stride_mask_1) == 0) ? 1 : 0);
                    int is_local_ii_1 = is_local_1;
                    exp_off[0] = 0;
                    if (is_local_ii_1 != 0) {
                        uint32_t _shared_atomic_old_4;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_4) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_1)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[0] = (int)_shared_atomic_old_4;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_2 << 16);
                }
            }
            if (done == 0) {
                expanded = cluster_tid + 8192;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_2 = 0;
                    unsigned int word0_3 = 0;
                    unsigned int word1_3 = 0;
                    unsigned int remote_2 = 0;
                    int src_rank_2 = 0;
                    int src_slot_2 = 0;
                    int packed_word_2 = 0;
                    src_rank_2 = expanded / (32 * top_k);
                    src_slot_2 = expanded % (32 * top_k);
                    uint32_t _mapa_5;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mapa_5) : "r"(smem_packed_addr), "r"(src_rank_2));
                    remote_2 = _mapa_5;
                    unsigned int _cluster_ld_5;
                    asm volatile(
                        "ld.shared::cluster.u32 %0, [%1];"
                        : "=r"(_cluster_ld_5) : "r"(remote_2 + (unsigned int)(src_slot_2 * 4)) : "memory");
                    word0_3 = _cluster_ld_5;
                    idx_ii_2 = (int)word0_3 >> 16;
                    word0_3 = word0_3 & 65535;
                    exp_idx[1] = idx_ii_2;
                    int local_idx_2 = idx_ii_2 - local_experts_start_idx;
                    int extent_2 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_2 = (1 << local_experts_stride_log2) - 1;
                    int is_local_2 = ((local_idx_2 >= 0 && local_idx_2 < extent_2 && (local_idx_2 & stride_mask_2) == 0) ? 1 : 0);
                    int is_local_ii_2 = is_local_2;
                    exp_off[1] = 0;
                    if (is_local_ii_2 != 0) {
                        uint32_t _shared_atomic_old_5;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_5) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_2)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[1] = (int)_shared_atomic_old_5;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_3 << 16);
                }
            }
            if (done == 0) {
                expanded = cluster_tid + 16384;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_3 = 0;
                    unsigned int word0_4 = 0;
                    unsigned int word1_4 = 0;
                    unsigned int remote_4 = 0;
                    int src_rank_3 = 0;
                    int src_slot_3 = 0;
                    int packed_word_3 = 0;
                    src_rank_3 = expanded / (32 * top_k);
                    src_slot_3 = expanded % (32 * top_k);
                    uint32_t _mapa_6;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mapa_6) : "r"(smem_packed_addr), "r"(src_rank_3));
                    remote_4 = _mapa_6;
                    unsigned int _cluster_ld_6;
                    asm volatile(
                        "ld.shared::cluster.u32 %0, [%1];"
                        : "=r"(_cluster_ld_6) : "r"(remote_4 + (unsigned int)(src_slot_3 * 4)) : "memory");
                    word0_4 = _cluster_ld_6;
                    idx_ii_3 = (int)word0_4 >> 16;
                    word0_4 = word0_4 & 65535;
                    exp_idx[2] = idx_ii_3;
                    int local_idx_3 = idx_ii_3 - local_experts_start_idx;
                    int extent_3 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_3 = (1 << local_experts_stride_log2) - 1;
                    int is_local_3 = ((local_idx_3 >= 0 && local_idx_3 < extent_3 && (local_idx_3 & stride_mask_3) == 0) ? 1 : 0);
                    int is_local_ii_3 = is_local_3;
                    exp_off[2] = 0;
                    if (is_local_ii_3 != 0) {
                        uint32_t _shared_atomic_old_6;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_6) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_3)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[2] = (int)_shared_atomic_old_6;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_4 << 16);
                }
            }
            if (done == 0) {
                expanded = cluster_tid + 24576;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_4 = 0;
                    unsigned int word0_5 = 0;
                    unsigned int word1_5 = 0;
                    unsigned int remote_5 = 0;
                    int src_rank_5 = 0;
                    int src_slot_4 = 0;
                    int packed_word_4 = 0;
                    src_rank_5 = expanded / (32 * top_k);
                    src_slot_4 = expanded % (32 * top_k);
                    uint32_t _mapa_7;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mapa_7) : "r"(smem_packed_addr), "r"(src_rank_5));
                    remote_5 = _mapa_7;
                    unsigned int _cluster_ld_7;
                    asm volatile(
                        "ld.shared::cluster.u32 %0, [%1];"
                        : "=r"(_cluster_ld_7) : "r"(remote_5 + (unsigned int)(src_slot_4 * 4)) : "memory");
                    word0_5 = _cluster_ld_7;
                    idx_ii_4 = (int)word0_5 >> 16;
                    word0_5 = word0_5 & 65535;
                    exp_idx[3] = idx_ii_4;
                    int local_idx_4 = idx_ii_4 - local_experts_start_idx;
                    int extent_4 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_4 = (1 << local_experts_stride_log2) - 1;
                    int is_local_4 = ((local_idx_4 >= 0 && local_idx_4 < extent_4 && (local_idx_4 & stride_mask_4) == 0) ? 1 : 0);
                    int is_local_ii_4 = is_local_4;
                    exp_off[3] = 0;
                    if (is_local_ii_4 != 0) {
                        uint32_t _shared_atomic_old_7;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_7) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_4)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[3] = (int)_shared_atomic_old_7;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_5 << 16);
                }
            }
        }
    }
    if (done == 0) {
        if (expanded_size >= 65536) {
            expanded = cluster_tid + 32768;
            int idx_ii_5 = 0;
            unsigned int word0_6 = 0;
            unsigned int word1_6 = 0;
            unsigned int remote_6 = 0;
            int src_rank_6 = 0;
            int src_slot_6 = 0;
            int packed_word_5 = 0;
            src_rank_6 = expanded / (32 * top_k);
            src_slot_6 = expanded % (32 * top_k);
            uint32_t _mapa_8;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_8) : "r"(smem_packed_addr), "r"(src_rank_6));
            remote_6 = _mapa_8;
            unsigned int _cluster_ld_8;
            asm volatile(
                "ld.shared::cluster.u32 %0, [%1];"
                : "=r"(_cluster_ld_8) : "r"(remote_6 + (unsigned int)(src_slot_6 * 4)) : "memory");
            word0_6 = _cluster_ld_8;
            idx_ii_5 = (int)word0_6 >> 16;
            word0_6 = word0_6 & 65535;
            exp_idx[4] = idx_ii_5;
            int local_idx_5 = idx_ii_5 - local_experts_start_idx;
            int extent_5 = num_local_experts << local_experts_stride_log2;
            int stride_mask_5 = (1 << local_experts_stride_log2) - 1;
            int is_local_5 = ((local_idx_5 >= 0 && local_idx_5 < extent_5 && (local_idx_5 & stride_mask_5) == 0) ? 1 : 0);
            int is_local_ii_5 = is_local_5;
            exp_off[4] = 0;
            if (is_local_ii_5 != 0) {
                uint32_t _shared_atomic_old_8;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_8) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_5)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[4] = (int)_shared_atomic_old_8;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_6 << 16);
            expanded = cluster_tid + 40960;
            int idx_ii_0_1 = 0;
            unsigned int word0_1_1 = 0;
            unsigned int word1_2_1 = 0;
            unsigned int remote_3_1 = 0;
            int src_rank_4_1 = 0;
            int src_slot_5_1 = 0;
            int packed_word_6_1 = 0;
            src_rank_4_1 = expanded / (32 * top_k);
            src_slot_5_1 = expanded % (32 * top_k);
            uint32_t _mapa_9;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_9) : "r"(smem_packed_addr), "r"(src_rank_4_1));
            remote_3_1 = _mapa_9;
            unsigned int _cluster_ld_9;
            asm volatile(
                "ld.shared::cluster.u32 %0, [%1];"
                : "=r"(_cluster_ld_9) : "r"(remote_3_1 + (unsigned int)(src_slot_5_1 * 4)) : "memory");
            word0_1_1 = _cluster_ld_9;
            idx_ii_0_1 = (int)word0_1_1 >> 16;
            word0_1_1 = word0_1_1 & 65535;
            exp_idx[5] = idx_ii_0_1;
            int local_idx_7_1 = idx_ii_0_1 - local_experts_start_idx;
            int extent_8_1 = num_local_experts << local_experts_stride_log2;
            int stride_mask_9_1 = (1 << local_experts_stride_log2) - 1;
            int is_local_10_1 = ((local_idx_7_1 >= 0 && local_idx_7_1 < extent_8_1 && (local_idx_7_1 & stride_mask_9_1) == 0) ? 1 : 0);
            int is_local_ii_11_1 = is_local_10_1;
            exp_off[5] = 0;
            if (is_local_ii_11_1 != 0) {
                uint32_t _shared_atomic_old_9;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_9) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_0_1)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[5] = (int)_shared_atomic_old_9;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_1_1 << 16);
            expanded = cluster_tid + 49152;
            int idx_ii_12_1 = 0;
            unsigned int word0_13_1 = 0;
            unsigned int word1_14_1 = 0;
            unsigned int remote_15_1 = 0;
            int src_rank_16_1 = 0;
            int src_slot_17_1 = 0;
            int packed_word_18_1 = 0;
            src_rank_16_1 = expanded / (32 * top_k);
            src_slot_17_1 = expanded % (32 * top_k);
            uint32_t _mapa_10;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_10) : "r"(smem_packed_addr), "r"(src_rank_16_1));
            remote_15_1 = _mapa_10;
            unsigned int _cluster_ld_10;
            asm volatile(
                "ld.shared::cluster.u32 %0, [%1];"
                : "=r"(_cluster_ld_10) : "r"(remote_15_1 + (unsigned int)(src_slot_17_1 * 4)) : "memory");
            word0_13_1 = _cluster_ld_10;
            idx_ii_12_1 = (int)word0_13_1 >> 16;
            word0_13_1 = word0_13_1 & 65535;
            exp_idx[6] = idx_ii_12_1;
            int local_idx_19_1 = idx_ii_12_1 - local_experts_start_idx;
            int extent_20_1 = num_local_experts << local_experts_stride_log2;
            int stride_mask_21_1 = (1 << local_experts_stride_log2) - 1;
            int is_local_22_1 = ((local_idx_19_1 >= 0 && local_idx_19_1 < extent_20_1 && (local_idx_19_1 & stride_mask_21_1) == 0) ? 1 : 0);
            int is_local_ii_23_1 = is_local_22_1;
            exp_off[6] = 0;
            if (is_local_ii_23_1 != 0) {
                uint32_t _shared_atomic_old_10;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_10) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_12_1)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[6] = (int)_shared_atomic_old_10;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_13_1 << 16);
            expanded = cluster_tid + 57344;
            int idx_ii_24_1 = 0;
            unsigned int word0_25_1 = 0;
            unsigned int word1_26_1 = 0;
            unsigned int remote_27_1 = 0;
            int src_rank_28_1 = 0;
            int src_slot_29_1 = 0;
            int packed_word_30_1 = 0;
            src_rank_28_1 = expanded / (32 * top_k);
            src_slot_29_1 = expanded % (32 * top_k);
            uint32_t _mapa_11;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_11) : "r"(smem_packed_addr), "r"(src_rank_28_1));
            remote_27_1 = _mapa_11;
            unsigned int _cluster_ld_11;
            asm volatile(
                "ld.shared::cluster.u32 %0, [%1];"
                : "=r"(_cluster_ld_11) : "r"(remote_27_1 + (unsigned int)(src_slot_29_1 * 4)) : "memory");
            word0_25_1 = _cluster_ld_11;
            idx_ii_24_1 = (int)word0_25_1 >> 16;
            word0_25_1 = word0_25_1 & 65535;
            exp_idx[7] = idx_ii_24_1;
            int local_idx_31_1 = idx_ii_24_1 - local_experts_start_idx;
            int extent_32_1 = num_local_experts << local_experts_stride_log2;
            int stride_mask_33_1 = (1 << local_experts_stride_log2) - 1;
            int is_local_34_1 = ((local_idx_31_1 >= 0 && local_idx_31_1 < extent_32_1 && (local_idx_31_1 & stride_mask_33_1) == 0) ? 1 : 0);
            int is_local_ii_35_1 = is_local_34_1;
            exp_off[7] = 0;
            if (is_local_ii_35_1 != 0) {
                uint32_t _shared_atomic_old_11;
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_11) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_24_1)))), "r"(static_cast<uint32_t>(1)) : "memory");
                exp_off[7] = (int)_shared_atomic_old_11;
            }
            topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_25_1 << 16);
        } else {
            if (done == 0) {
                expanded = cluster_tid + 32768;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_6 = 0;
                    unsigned int word0_7 = 0;
                    unsigned int word1_7 = 0;
                    unsigned int remote_7 = 0;
                    int src_rank_7 = 0;
                    int src_slot_7 = 0;
                    int packed_word_7 = 0;
                    src_rank_7 = expanded / (32 * top_k);
                    src_slot_7 = expanded % (32 * top_k);
                    uint32_t _mapa_12;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mapa_12) : "r"(smem_packed_addr), "r"(src_rank_7));
                    remote_7 = _mapa_12;
                    unsigned int _cluster_ld_12;
                    asm volatile(
                        "ld.shared::cluster.u32 %0, [%1];"
                        : "=r"(_cluster_ld_12) : "r"(remote_7 + (unsigned int)(src_slot_7 * 4)) : "memory");
                    word0_7 = _cluster_ld_12;
                    idx_ii_6 = (int)word0_7 >> 16;
                    word0_7 = word0_7 & 65535;
                    exp_idx[4] = idx_ii_6;
                    int local_idx_6 = idx_ii_6 - local_experts_start_idx;
                    int extent_6 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_6 = (1 << local_experts_stride_log2) - 1;
                    int is_local_6 = ((local_idx_6 >= 0 && local_idx_6 < extent_6 && (local_idx_6 & stride_mask_6) == 0) ? 1 : 0);
                    int is_local_ii_6 = is_local_6;
                    exp_off[4] = 0;
                    if (is_local_ii_6 != 0) {
                        uint32_t _shared_atomic_old_12;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_12) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_6)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[4] = (int)_shared_atomic_old_12;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_7 << 16);
                }
            }
            if (done == 0) {
                expanded = cluster_tid + 40960;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_7 = 0;
                    unsigned int word0_8 = 0;
                    unsigned int word1_8 = 0;
                    unsigned int remote_8 = 0;
                    int src_rank_8 = 0;
                    int src_slot_8 = 0;
                    int packed_word_8 = 0;
                    src_rank_8 = expanded / (32 * top_k);
                    src_slot_8 = expanded % (32 * top_k);
                    uint32_t _mapa_13;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mapa_13) : "r"(smem_packed_addr), "r"(src_rank_8));
                    remote_8 = _mapa_13;
                    unsigned int _cluster_ld_13;
                    asm volatile(
                        "ld.shared::cluster.u32 %0, [%1];"
                        : "=r"(_cluster_ld_13) : "r"(remote_8 + (unsigned int)(src_slot_8 * 4)) : "memory");
                    word0_8 = _cluster_ld_13;
                    idx_ii_7 = (int)word0_8 >> 16;
                    word0_8 = word0_8 & 65535;
                    exp_idx[5] = idx_ii_7;
                    int local_idx_8 = idx_ii_7 - local_experts_start_idx;
                    int extent_7 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_7 = (1 << local_experts_stride_log2) - 1;
                    int is_local_7 = ((local_idx_8 >= 0 && local_idx_8 < extent_7 && (local_idx_8 & stride_mask_7) == 0) ? 1 : 0);
                    int is_local_ii_7 = is_local_7;
                    exp_off[5] = 0;
                    if (is_local_ii_7 != 0) {
                        uint32_t _shared_atomic_old_13;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_13) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_7)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[5] = (int)_shared_atomic_old_13;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_8 << 16);
                }
            }
            if (done == 0) {
                expanded = cluster_tid + 49152;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_8 = 0;
                    unsigned int word0_9 = 0;
                    unsigned int word1_9 = 0;
                    unsigned int remote_9 = 0;
                    int src_rank_9 = 0;
                    int src_slot_9 = 0;
                    int packed_word_9 = 0;
                    src_rank_9 = expanded / (32 * top_k);
                    src_slot_9 = expanded % (32 * top_k);
                    uint32_t _mapa_14;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mapa_14) : "r"(smem_packed_addr), "r"(src_rank_9));
                    remote_9 = _mapa_14;
                    unsigned int _cluster_ld_14;
                    asm volatile(
                        "ld.shared::cluster.u32 %0, [%1];"
                        : "=r"(_cluster_ld_14) : "r"(remote_9 + (unsigned int)(src_slot_9 * 4)) : "memory");
                    word0_9 = _cluster_ld_14;
                    idx_ii_8 = (int)word0_9 >> 16;
                    word0_9 = word0_9 & 65535;
                    exp_idx[6] = idx_ii_8;
                    int local_idx_9 = idx_ii_8 - local_experts_start_idx;
                    int extent_9 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_8 = (1 << local_experts_stride_log2) - 1;
                    int is_local_8 = ((local_idx_9 >= 0 && local_idx_9 < extent_9 && (local_idx_9 & stride_mask_8) == 0) ? 1 : 0);
                    int is_local_ii_8 = is_local_8;
                    exp_off[6] = 0;
                    if (is_local_ii_8 != 0) {
                        uint32_t _shared_atomic_old_14;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_14) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_8)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[6] = (int)_shared_atomic_old_14;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_9 << 16);
                }
            }
            if (done == 0) {
                expanded = cluster_tid + 57344;
                if (expanded >= expanded_size) {
                    done = 1;
                } else {
                    int idx_ii_9 = 0;
                    unsigned int word0_10 = 0;
                    unsigned int word1_10 = 0;
                    unsigned int remote_10 = 0;
                    int src_rank_10 = 0;
                    int src_slot_10 = 0;
                    int packed_word_10 = 0;
                    src_rank_10 = expanded / (32 * top_k);
                    src_slot_10 = expanded % (32 * top_k);
                    uint32_t _mapa_15;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mapa_15) : "r"(smem_packed_addr), "r"(src_rank_10));
                    remote_10 = _mapa_15;
                    unsigned int _cluster_ld_15;
                    asm volatile(
                        "ld.shared::cluster.u32 %0, [%1];"
                        : "=r"(_cluster_ld_15) : "r"(remote_10 + (unsigned int)(src_slot_10 * 4)) : "memory");
                    word0_10 = _cluster_ld_15;
                    idx_ii_9 = (int)word0_10 >> 16;
                    word0_10 = word0_10 & 65535;
                    exp_idx[7] = idx_ii_9;
                    int local_idx_10 = idx_ii_9 - local_experts_start_idx;
                    int extent_10 = num_local_experts << local_experts_stride_log2;
                    int stride_mask_10 = (1 << local_experts_stride_log2) - 1;
                    int is_local_9 = ((local_idx_10 >= 0 && local_idx_10 < extent_10 && (local_idx_10 & stride_mask_10) == 0) ? 1 : 0);
                    int is_local_ii_9 = is_local_9;
                    exp_off[7] = 0;
                    if (is_local_ii_9 != 0) {
                        uint32_t _shared_atomic_old_15;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_15) : "r"(static_cast<uint32_t>((smem_expert_count_addr + 4 * (idx_ii_9)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        exp_off[7] = (int)_shared_atomic_old_15;
                    }
                    topk_weights[expanded] = (__nv_bfloat16)__uint_as_float(word0_10 << 16);
                }
            }
        }
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    int expert = tid_0;
    int count = 0;
    int block_expert_offset = 0;
    int peer_count = 0;
    unsigned int peer_addr = 0;
    if (expert < num_experts) {
        peer_count = 0;
        if (num_tokens > 0) {
            uint32_t _mapa_16;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_16) : "r"(smem_expert_count_addr), "r"(0));
            peer_addr = _mapa_16;
            int _cluster_ld_16;
            asm volatile(
                "ld.shared::cluster.s32 %0, [%1];"
                : "=r"(_cluster_ld_16) : "r"(peer_addr + (unsigned int)(expert * 4)) : "memory");
            peer_count = _cluster_ld_16;
        }
        if (rank == 0) {
            block_expert_offset = count;
        }
        count = count + peer_count;
        peer_count = 0;
        if (num_tokens > 32) {
            uint32_t _mapa_17;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_17) : "r"(smem_expert_count_addr), "r"(1));
            peer_addr = _mapa_17;
            int _cluster_ld_17;
            asm volatile(
                "ld.shared::cluster.s32 %0, [%1];"
                : "=r"(_cluster_ld_17) : "r"(peer_addr + (unsigned int)(expert * 4)) : "memory");
            peer_count = _cluster_ld_17;
        }
        if (rank == 1) {
            block_expert_offset = count;
        }
        count = count + peer_count;
        peer_count = 0;
        if (num_tokens > 64) {
            uint32_t _mapa_18;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_18) : "r"(smem_expert_count_addr), "r"(2));
            peer_addr = _mapa_18;
            int _cluster_ld_18;
            asm volatile(
                "ld.shared::cluster.s32 %0, [%1];"
                : "=r"(_cluster_ld_18) : "r"(peer_addr + (unsigned int)(expert * 4)) : "memory");
            peer_count = _cluster_ld_18;
        }
        if (rank == 2) {
            block_expert_offset = count;
        }
        count = count + peer_count;
        peer_count = 0;
        if (num_tokens > 96) {
            uint32_t _mapa_19;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_19) : "r"(smem_expert_count_addr), "r"(3));
            peer_addr = _mapa_19;
            int _cluster_ld_19;
            asm volatile(
                "ld.shared::cluster.s32 %0, [%1];"
                : "=r"(_cluster_ld_19) : "r"(peer_addr + (unsigned int)(expert * 4)) : "memory");
            peer_count = _cluster_ld_19;
        }
        if (rank == 3) {
            block_expert_offset = count;
        }
        count = count + peer_count;
        peer_count = 0;
        if (num_tokens > 128) {
            uint32_t _mapa_20;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_20) : "r"(smem_expert_count_addr), "r"(4));
            peer_addr = _mapa_20;
            int _cluster_ld_20;
            asm volatile(
                "ld.shared::cluster.s32 %0, [%1];"
                : "=r"(_cluster_ld_20) : "r"(peer_addr + (unsigned int)(expert * 4)) : "memory");
            peer_count = _cluster_ld_20;
        }
        if (rank == 4) {
            block_expert_offset = count;
        }
        count = count + peer_count;
        peer_count = 0;
        if (num_tokens > 160) {
            uint32_t _mapa_21;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_21) : "r"(smem_expert_count_addr), "r"(5));
            peer_addr = _mapa_21;
            int _cluster_ld_21;
            asm volatile(
                "ld.shared::cluster.s32 %0, [%1];"
                : "=r"(_cluster_ld_21) : "r"(peer_addr + (unsigned int)(expert * 4)) : "memory");
            peer_count = _cluster_ld_21;
        }
        if (rank == 5) {
            block_expert_offset = count;
        }
        count = count + peer_count;
        peer_count = 0;
        if (num_tokens > 192) {
            uint32_t _mapa_22;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_22) : "r"(smem_expert_count_addr), "r"(6));
            peer_addr = _mapa_22;
            int _cluster_ld_22;
            asm volatile(
                "ld.shared::cluster.s32 %0, [%1];"
                : "=r"(_cluster_ld_22) : "r"(peer_addr + (unsigned int)(expert * 4)) : "memory");
            peer_count = _cluster_ld_22;
        }
        if (rank == 6) {
            block_expert_offset = count;
        }
        count = count + peer_count;
        peer_count = 0;
        if (num_tokens > 224) {
            uint32_t _mapa_23;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_23) : "r"(smem_expert_count_addr), "r"(7));
            peer_addr = _mapa_23;
            int _cluster_ld_23;
            asm volatile(
                "ld.shared::cluster.s32 %0, [%1];"
                : "=r"(_cluster_ld_23) : "r"(peer_addr + (unsigned int)(expert * 4)) : "memory");
            peer_count = _cluster_ld_23;
        }
        if (rank == 7) {
            block_expert_offset = count;
        }
        count = count + peer_count;
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    int num_cta = (count + tile_tokens_dim - 1) / tile_tokens_dim;
    if (padding_log2 > 0) {
        num_cta = count + (1 << padding_log2) - 1 >> padding_log2;
    }
    int num_cta_3 = num_cta;
    int inclusive = num_cta_3;
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
    wt = warp_totals[4];
    if (warp > 4) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[5];
    if (warp > 5) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[6];
    if (warp > 6) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[7];
    if (warp > 7) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[8];
    if (warp > 8) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[9];
    if (warp > 9) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[10];
    if (warp > 10) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[11];
    if (warp > 11) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[12];
    if (warp > 12) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[13];
    if (warp > 13) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[14];
    if (warp > 14) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[15];
    if (warp > 15) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[16];
    if (warp > 16) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[17];
    if (warp > 17) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[18];
    if (warp > 18) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[19];
    if (warp > 19) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[20];
    if (warp > 20) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[21];
    if (warp > 21) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[22];
    if (warp > 22) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[23];
    if (warp > 23) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[24];
    if (warp > 24) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[25];
    if (warp > 25) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[26];
    if (warp > 26) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[27];
    if (warp > 27) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[28];
    if (warp > 28) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[29];
    if (warp > 29) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[30];
    if (warp > 30) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    wt = warp_totals[31];
    if (warp > 31) {
        warp_prefix = warp_prefix + wt;
    }
    block_total = block_total + wt;
    int exclusive = inclusive - num_cta_3 + warp_prefix;
    __syncthreads();
    if (expert < num_experts) {
        int local_idx_11 = expert - local_experts_start_idx >> local_experts_stride_log2;
        int mn_limit1 = 0;
        int mn_limit2 = 0;
        int mn_limit = 0;
        for (int cta = rank; cta < num_cta_3; cta += 8) {
            cta_idx_xy_to_batch_idx[exclusive + cta] = local_idx_11;
            int scaled = (exclusive + cta + 1) * tile_tokens_dim;
            if (padding_log2 > 0) {
                scaled = exclusive + cta + 1 << padding_log2;
            }
            mn_limit1 = scaled;
            int scaled_0 = exclusive * tile_tokens_dim;
            if (padding_log2 > 0) {
                scaled_0 = exclusive << padding_log2;
            }
            mn_limit2 = scaled_0 + count;
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
        int scaled_1 = exclusive * tile_tokens_dim;
        if (padding_log2 > 0) {
            scaled_1 = exclusive << padding_log2;
        }
        int expert_rows = scaled_1;
        smem_expert_offset[expert] = expert_rows + block_expert_offset;
    }
    if (rank == 0) {
        if (warp == 31) {
            if (elect_sync()) {
                int scaled_2 = block_total * tile_tokens_dim;
                if (padding_log2 > 0) {
                    scaled_2 = block_total << padding_log2;
                }
                int padded_rows = scaled_2;
                permuted_idx_size[0] = padded_rows;
                num_non_exiting_ctas[0] = block_total;
            }
        }
    }
    __syncthreads();
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    int done_4 = 0;
    int expanded_j = 0;
    int idx_j = 0;
    int is_local_j = 0;
    int token_j = 0;
    int permuted_j = -1;
    int neg_one = -1;
    int slot_base = 0;
    if (done_4 == 0) {
        expanded_j = cluster_tid;
        if (expanded_j >= expanded_size) {
            done_4 = 1;
        } else {
            idx_j = exp_idx[0];
            int local_idx_12 = idx_j - local_experts_start_idx;
            int extent_11 = num_local_experts << local_experts_stride_log2;
            int stride_mask_11 = (1 << local_experts_stride_log2) - 1;
            int is_local_11 = ((local_idx_12 >= 0 && local_idx_12 < extent_11 && (local_idx_12 & stride_mask_11) == 0) ? 1 : 0);
            is_local_j = is_local_11;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[0] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_4 == 0) {
        expanded_j = cluster_tid + 8192;
        if (expanded_j >= expanded_size) {
            done_4 = 1;
        } else {
            idx_j = exp_idx[1];
            int local_idx_13 = idx_j - local_experts_start_idx;
            int extent_12 = num_local_experts << local_experts_stride_log2;
            int stride_mask_12 = (1 << local_experts_stride_log2) - 1;
            int is_local_12 = ((local_idx_13 >= 0 && local_idx_13 < extent_12 && (local_idx_13 & stride_mask_12) == 0) ? 1 : 0);
            is_local_j = is_local_12;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[1] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_4 == 0) {
        expanded_j = cluster_tid + 16384;
        if (expanded_j >= expanded_size) {
            done_4 = 1;
        } else {
            idx_j = exp_idx[2];
            int local_idx_14 = idx_j - local_experts_start_idx;
            int extent_13 = num_local_experts << local_experts_stride_log2;
            int stride_mask_13 = (1 << local_experts_stride_log2) - 1;
            int is_local_13 = ((local_idx_14 >= 0 && local_idx_14 < extent_13 && (local_idx_14 & stride_mask_13) == 0) ? 1 : 0);
            is_local_j = is_local_13;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[2] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_4 == 0) {
        expanded_j = cluster_tid + 24576;
        if (expanded_j >= expanded_size) {
            done_4 = 1;
        } else {
            idx_j = exp_idx[3];
            int local_idx_15 = idx_j - local_experts_start_idx;
            int extent_14 = num_local_experts << local_experts_stride_log2;
            int stride_mask_14 = (1 << local_experts_stride_log2) - 1;
            int is_local_14 = ((local_idx_15 >= 0 && local_idx_15 < extent_14 && (local_idx_15 & stride_mask_14) == 0) ? 1 : 0);
            is_local_j = is_local_14;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[3] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_4 == 0) {
        expanded_j = cluster_tid + 32768;
        if (expanded_j >= expanded_size) {
            done_4 = 1;
        } else {
            idx_j = exp_idx[4];
            int local_idx_16 = idx_j - local_experts_start_idx;
            int extent_15 = num_local_experts << local_experts_stride_log2;
            int stride_mask_15 = (1 << local_experts_stride_log2) - 1;
            int is_local_15 = ((local_idx_16 >= 0 && local_idx_16 < extent_15 && (local_idx_16 & stride_mask_15) == 0) ? 1 : 0);
            is_local_j = is_local_15;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[4] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_4 == 0) {
        expanded_j = cluster_tid + 40960;
        if (expanded_j >= expanded_size) {
            done_4 = 1;
        } else {
            idx_j = exp_idx[5];
            int local_idx_17 = idx_j - local_experts_start_idx;
            int extent_16 = num_local_experts << local_experts_stride_log2;
            int stride_mask_16 = (1 << local_experts_stride_log2) - 1;
            int is_local_16 = ((local_idx_17 >= 0 && local_idx_17 < extent_16 && (local_idx_17 & stride_mask_16) == 0) ? 1 : 0);
            is_local_j = is_local_16;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[5] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_4 == 0) {
        expanded_j = cluster_tid + 49152;
        if (expanded_j >= expanded_size) {
            done_4 = 1;
        } else {
            idx_j = exp_idx[6];
            int local_idx_18 = idx_j - local_experts_start_idx;
            int extent_17 = num_local_experts << local_experts_stride_log2;
            int stride_mask_17 = (1 << local_experts_stride_log2) - 1;
            int is_local_17 = ((local_idx_18 >= 0 && local_idx_18 < extent_17 && (local_idx_18 & stride_mask_17) == 0) ? 1 : 0);
            is_local_j = is_local_17;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[6] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    if (done_4 == 0) {
        expanded_j = cluster_tid + 57344;
        if (expanded_j >= expanded_size) {
            done_4 = 1;
        } else {
            idx_j = exp_idx[7];
            int local_idx_20 = idx_j - local_experts_start_idx;
            int extent_18 = num_local_experts << local_experts_stride_log2;
            int stride_mask_18 = (1 << local_experts_stride_log2) - 1;
            int is_local_18 = ((local_idx_20 >= 0 && local_idx_20 < extent_18 && (local_idx_20 & stride_mask_18) == 0) ? 1 : 0);
            is_local_j = is_local_18;
            token_j = expanded_j / top_k;
            slot_base = smem_expert_offset[idx_j];
            permuted_j = ((is_local_j != 0) ? slot_base + exp_off[7] : neg_one);
            expanded_idx_to_permuted_idx[expanded_j] = permuted_j;
            if (is_local_j != 0) {
                permuted_idx_to_token_idx[permuted_j] = token_j;
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
