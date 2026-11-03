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
#define SMEM_SUM_SQ_PARTIALS_OFF 0
#define SMEM_SUM_SQ_PARTIALS_STAGE_BYTES 32
#define SMEM_SUM_SQ_PARTIALS_STRIDE 32
#define SMEM_DOT_PARTIALS_OFF 32
#define SMEM_DOT_PARTIALS_STAGE_BYTES 32
#define SMEM_DOT_PARTIALS_STRIDE 32
#define SMEM_SOURCE_SCORES_OFF 64
#define SMEM_SOURCE_SCORES_STAGE_BYTES 36
#define SMEM_SOURCE_SCORES_STRIDE 36
#define SMEM_SOURCE_PROBABILITIES_OFF 112
#define SMEM_SOURCE_PROBABILITIES_STAGE_BYTES 36
#define SMEM_SOURCE_PROBABILITIES_STRIDE 36
#define SMEM_OUTPUT_NORM_SCALAR_OFF 160
#define SMEM_OUTPUT_NORM_SCALAR_STAGE_BYTES 4
#define SMEM_OUTPUT_NORM_SCALAR_STRIDE 4
#define SMEM_TOTAL 256
#define THREADS 256

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


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_kimi_k3_attn_res_65fea76173756111a940(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long out_stride, unsigned long long prefix_stride, unsigned long long delta_stride, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int block_write_idx)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* sum_sq_partials = reinterpret_cast<float*>(smem_raw + 0);
    const int sum_sq_partials_addr = smem + 0;
    float* dot_partials = reinterpret_cast<float*>(smem_raw + 32);
    const int dot_partials_addr = smem + 32;
    float* source_scores = reinterpret_cast<float*>(smem_raw + 64);
    const int source_scores_addr = smem + 64;
    float* source_probabilities = reinterpret_cast<float*>(smem_raw + 112);
    const int source_probabilities_addr = smem + 112;
    float* output_norm_scalar = reinterpret_cast<float*>(smem_raw + 160);
    const int output_norm_scalar_addr = smem + 160;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned long long row = (unsigned long long)bid;
    unsigned long long out_base = row * out_stride;
    unsigned long long prefix_base = row * prefix_stride;
    unsigned long long delta_base = row * delta_stride;
    unsigned long long block_row_base = row * blocks_m_stride;
    #pragma unroll
    for (int item = 0; item < 28; item++) {
        unsigned long long elem = (unsigned long long)tid + (unsigned long long)(item * 256);
        float updated = (float)prefix[prefix_base + elem];
        updated = (float)(__nv_bfloat16)(updated + (float)delta[delta_base + elem]);
        prefix[prefix_base + elem] = (__nv_bfloat16)updated;
    }
    float mixed[28];
    #pragma unroll
    for (int item_1 = 0; item_1 < 28; item_1++) {
        mixed[item_1] = 0.0f;
    }
    float q_cache[28];
    #pragma unroll
    for (int item_2 = 0; item_2 < 28; item_2++) {
        unsigned long long elem_1 = (unsigned long long)tid + (unsigned long long)(item_2 * 256);
        q_cache[item_2] = (float)norm_weight[elem_1] * (float)qk_weight[elem_1];
    }
    #pragma unroll
    for (int source = 0; source < 9; source++) {
        float local_sum_sq = 0.0f;
        float local_dot = 0.0f;
        #pragma unroll
        for (int item_3 = 0; item_3 < 28; item_3++) {
            unsigned long long elem_2 = (unsigned long long)tid + (unsigned long long)(item_3 * 256);
            float value = 0.0f;
            if (source == 8) {
                value = (float)prefix[prefix_base + elem_2];
            } else {
                unsigned long long source_offset = block_row_base + (unsigned long long)source * blocks_k_stride + elem_2;
                value = (float)blocks[source_offset];
            }
            local_sum_sq = __fmaf_rn(value, value, local_sum_sq);
            local_dot = __fmaf_rn(value, q_cache[item_3], local_dot);
        }
        float _warp_reduce_0 = local_sum_sq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        local_sum_sq = _warp_reduce_0;
        float _warp_reduce_1 = local_dot;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
        local_dot = _warp_reduce_1;
        if (lane == 0) {
            sum_sq_partials[warp] = local_sum_sq;
            dot_partials[warp] = local_dot;
        }
        __syncthreads();
        float block_sum_sq = ((lane < 8) ? sum_sq_partials[lane] : 0.0f);
        float block_dot = ((lane < 8) ? dot_partials[lane] : 0.0f);
        float _warp_reduce_2 = block_sum_sq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
        block_sum_sq = _warp_reduce_2;
        float _warp_reduce_3 = block_dot;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
        block_dot = _warp_reduce_3;
        if (warp == 0) {
            if (elect_sync()) {
                float inv_rms = rsqrtf(block_sum_sq / 7168.0f + eps);
                source_scores[source] = block_dot * inv_rms;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        float row_max = source_scores[0];
        #pragma unroll
        for (int source_1 = 1; source_1 < 9; source_1++) {
            row_max = fmaxf(row_max, source_scores[source_1]);
        }
        float denominator = 0.0f;
        #pragma unroll
        for (int source_2 = 0; source_2 < 9; source_2++) {
            float probability = expf(source_scores[source_2] - row_max);
            source_probabilities[source_2] = probability;
            denominator += probability;
        }
        float denominator_rcp = approx_rcp(denominator);
        #pragma unroll
        for (int source_3 = 0; source_3 < 9; source_3++) {
            source_probabilities[source_3] = source_probabilities[source_3] * denominator_rcp;
        }
    }
    __syncthreads();
    #pragma unroll
    for (int source_4 = 0; source_4 < 9; source_4++) {
        float probability_1 = source_probabilities[source_4];
        #pragma unroll
        for (int item_4 = 0; item_4 < 28; item_4++) {
            unsigned long long elem_3 = (unsigned long long)tid + (unsigned long long)(item_4 * 256);
            float value_1 = 0.0f;
            if (source_4 == 8) {
                value_1 = (float)prefix[prefix_base + elem_3];
            } else {
                unsigned long long source_offset_1 = block_row_base + (unsigned long long)source_4 * blocks_k_stride + elem_3;
                value_1 = (float)blocks[source_offset_1];
            }
            mixed[item_4] = __fmaf_rn(probability_1, value_1, mixed[item_4]);
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    #pragma unroll
    for (int item_5 = 0; item_5 < 28; item_5++) {
        unsigned long long elem_4 = (unsigned long long)tid + (unsigned long long)(item_5 * 256);
        out[out_base + elem_4] = (__nv_bfloat16)mixed[item_5];
    }
}

} // extern "C"
