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
#define NUM_BLOCKS 0
#define HAS_DELTA 0
#define WRITE_BLOCK 1
#define APPLY_OUTPUT_NORM 1

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



extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_kimi_k3_attn_res_fc09e868f74b8c6fdf0e(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long out_stride, unsigned long long prefix_stride, unsigned long long delta_stride, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int block_write_idx)
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

    // === Task calls (dependency order) ===
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
        {
            unsigned long long write_offset = block_row_base + (unsigned long long)block_write_idx * blocks_k_stride + elem;
            __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(updated);
            blocks[write_offset] = _cvt_bf16_2;
        }
    }
    float mixed[28];
    #pragma unroll
    for (int item_1 = 0; item_1 < 28; item_1++) {
        mixed[item_1] = 0.0f;
    }
    {
        #pragma unroll
        for (int item_2 = 0; item_2 < 28; item_2++) {
            unsigned long long elem_1 = (unsigned long long)tid + (unsigned long long)(item_2 * 256);
            mixed[item_2] = (float)prefix[prefix_base + elem_1];
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    {
        float local_output_sum_sq = 0.0f;
        #pragma unroll
        for (int item_3 = 0; item_3 < 28; item_3++) {
            local_output_sum_sq += mixed[item_3] * mixed[item_3];
        }
        float _warp_reduce_4 = local_output_sum_sq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_4 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_4, offset);
        local_output_sum_sq = _warp_reduce_4;
        if (lane == 0) {
            sum_sq_partials[warp] = local_output_sum_sq;
        }
        __syncthreads();
        float output_sum_sq = ((lane < 8) ? sum_sq_partials[lane] : 0.0f);
        float _warp_reduce_5 = output_sum_sq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_5 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_5, offset);
        output_sum_sq = _warp_reduce_5;
        if (warp == 0) {
            if (elect_sync()) {
                float _rsqrt_1 = rsqrtf(output_sum_sq / 7168.0f + output_norm_eps);
                output_norm_scalar[0] = _rsqrt_1;
            }
        }
        __syncthreads();
        float output_inv_rms = output_norm_scalar[0];
        #pragma unroll
        for (int item_4 = 0; item_4 < 28; item_4++) {
            unsigned long long elem_2 = (unsigned long long)tid + (unsigned long long)(item_4 * 256);
            float value = mixed[item_4] * output_inv_rms * (float)output_norm_weight[elem_2];
            __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(value);
            out[out_base + elem_2] = _cvt_bf16_3;
        }
    }
}

} // extern "C"
