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
#define SMEM_WARP_AGGREGATES_OFF 0
#define SMEM_WARP_AGGREGATES_STAGE_BYTES 16
#define SMEM_WARP_AGGREGATES_STRIDE 16
#define SMEM_ROW_AMAX_SMEM_OFF 16
#define SMEM_ROW_AMAX_SMEM_STAGE_BYTES 4
#define SMEM_ROW_AMAX_SMEM_STRIDE 4
#define SMEM_TOTAL 0
#define THREADS 128

#include <math_constants.h>

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_stepfun_moe_97ebed750eec2cdf32ba(const __nv_bfloat16* input, const int* expanded_idx_to_permuted_idx, uint8_t* out_fp4, uint8_t* out_sf, float* per_token_scale, float global_scale_inv, int m, int n)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    __shared__ __align__(16) unsigned char smem_static_raw[32];

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* warp_aggregates = reinterpret_cast<float*>(smem_static_raw + 0);
    const int warp_aggregates_addr = (int)(unsigned long long)__cvta_generic_to_shared(warp_aggregates);
    float* row_amax_smem = reinterpret_cast<float*>(smem_static_raw + 16);
    const int row_amax_smem_addr = (int)(unsigned long long)__cvta_generic_to_shared(row_amax_smem);

    // === Task calls (dependency order) ===
    int row_idx = bid;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (row_idx < m) {
        row_idx = expanded_idx_to_permuted_idx[row_idx];
        if (row_idx >= 0) {
            unsigned int num_vecs_per_row = (unsigned int)((n + 16 - 1) / 16);
            unsigned int tid_u32 = (unsigned int)tid;
            float local_amax = 0.0f;
            for (unsigned int vec_idx = tid_u32; vec_idx < num_vecs_per_row; vec_idx += 128) {
                long long vec_offset = (long long)row_idx * (long long)num_vecs_per_row + (long long)vec_idx;
                unsigned int words[8];
                {
                    asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(words[0 + 0]), "=r"(words[0 + 1]), "=r"(words[0 + 2]), "=r"(words[0 + 3]), "=r"(words[0 + 4]), "=r"(words[0 + 5]), "=r"(words[0 + 6]), "=r"(words[0 + 7]) : "l"((const void*)((const char*)(input + vec_offset * 16) + 0)) : "memory");
                }
                uint32_t _bf16x2_abs_0;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_0) : "r"(words[0]));
                unsigned int pair_max = _bf16x2_abs_0;
                #pragma unroll
                for (int w = 1; w < 8; w++) {
                    uint32_t _bf16x2_abs_1;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_1) : "r"(words[w]));
                    uint32_t _bf16x2_max_0;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_0) : "r"(pair_max), "r"(_bf16x2_abs_1));
                    pair_max = _bf16x2_max_0;
                }
                uint16_t _bf16_max_0;
                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_0) : "h"((uint16_t)(pair_max & 65535)), "h"((uint16_t)(pair_max >> 16)));
                float _cvt_f32_bf16_0;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_0) : "h"((uint16_t)(_bf16_max_0)));
                float _fmax_0 = fmaxf(local_amax, _cvt_f32_bf16_0);
                local_amax = _fmax_0;
            }
            float warp_val = local_amax;
            float _shfl_down_0 = __shfl_down_sync(0xFFFFFFFF, warp_val, 1, 32);
            float _fmax_1 = fmaxf(warp_val, _shfl_down_0);
            warp_val = _fmax_1;
            float _shfl_down_1 = __shfl_down_sync(0xFFFFFFFF, warp_val, 2, 32);
            float _fmax_2 = fmaxf(warp_val, _shfl_down_1);
            warp_val = _fmax_2;
            float _shfl_down_2 = __shfl_down_sync(0xFFFFFFFF, warp_val, 4, 32);
            float _fmax_3 = fmaxf(warp_val, _shfl_down_2);
            warp_val = _fmax_3;
            float _shfl_down_3 = __shfl_down_sync(0xFFFFFFFF, warp_val, 8, 32);
            float _fmax_4 = fmaxf(warp_val, _shfl_down_3);
            warp_val = _fmax_4;
            float _shfl_down_4 = __shfl_down_sync(0xFFFFFFFF, warp_val, 16, 32);
            float _fmax_5 = fmaxf(warp_val, _shfl_down_4);
            warp_val = _fmax_5;
            int lane_0 = lane;
            int warp_1 = warp;
            if (lane_0 == 0) {
                warp_aggregates[warp_1] = warp_val;
            }
            __syncthreads();
            float row_amax = warp_val;
            if (tid == 0) {
                #pragma unroll
                for (int w_1 = 1; w_1 < 4; w_1++) {
                    float _fmax_6 = fmaxf(row_amax, warp_aggregates[w_1]);
                    row_amax = _fmax_6;
                }
                row_amax_smem[0] = row_amax;
            }
            __syncthreads();
            row_amax = row_amax_smem[0];
            float per_token = row_amax * global_scale_inv;
            if (tid == 0) {
                *(reinterpret_cast<float*>(per_token_scale + row_idx) + (0)) = per_token;
            }
            __syncthreads();
            per_token = per_token_scale[row_idx];
            float global_encode_scale = 0.0f;
            if (per_token != 0.0f) {
                float _rcp_0 = approx_rcp(per_token);
                global_encode_scale = _rcp_0;
            }
            for (unsigned int vec_idx2 = tid_u32; vec_idx2 < num_vecs_per_row; vec_idx2 += 128) {
                unsigned int vec_offset2 = (unsigned int)row_idx * num_vecs_per_row + vec_idx2;
                unsigned int words2[8];
                {
                    asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(words2[0 + 0]), "=r"(words2[0 + 1]), "=r"(words2[0 + 2]), "=r"(words2[0 + 3]), "=r"(words2[0 + 4]), "=r"(words2[0 + 5]), "=r"(words2[0 + 6]), "=r"(words2[0 + 7]) : "l"((const void*)((const char*)(input + (unsigned long long)vec_offset2 * 16) + 0)) : "memory");
                }
                uint32_t _bf16x2_abs_2;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_2) : "r"(words2[0]));
                unsigned int vec_pair_max = _bf16x2_abs_2;
                #pragma unroll
                for (int w2 = 1; w2 < 8; w2++) {
                    uint32_t _bf16x2_abs_3;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_3) : "r"(words2[w2]));
                    uint32_t _bf16x2_max_1;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_1) : "r"(vec_pair_max), "r"(_bf16x2_abs_3));
                    vec_pair_max = _bf16x2_max_1;
                }
                uint16_t _bf16_max_1;
                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_1) : "h"((uint16_t)(vec_pair_max & 65535)), "h"((uint16_t)(vec_pair_max >> 16)));
                float _cvt_f32_bf16_1;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_1) : "h"((uint16_t)(_bf16_max_1)));
                float vec_max = _cvt_f32_bf16_1;
                float _rcp_1 = approx_rcp(6.0f);
                float sf_value = global_encode_scale * (vec_max * _rcp_1);
                float _fp8_rt_0;
                uint16_t _e4m3x2_2;
                uint32_t _f16x2_2;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_2) : "f"(0.0f), "f"(sf_value));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2) : "h"(_e4m3x2_2));
                uint16_t _fp8_h0_2 = (uint16_t)(_f16x2_2 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_2));
                float sf_decoded = _fp8_rt_0;
                float output_scale = 0.0f;
                if (vec_max != 0.0f) {
                    float _rcp_2 = approx_rcp(global_encode_scale);
                    float _rcp_3 = approx_rcp(sf_decoded * _rcp_2);
                    output_scale = _rcp_3;
                }
                unsigned int packed[2];
                float values[8];
                #pragma unroll
                for (int w3 = 0; w3 < 4; w3++) {
                    unsigned int word = words2[w3];
                    values[2 * w3] = __uint_as_float(word << 16) * output_scale;
                    values[2 * w3 + 1] = __uint_as_float(word & 4294901760u) * output_scale;
                }
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(values[0]), "f"(values[1]), "f"(values[2]), "f"(values[3]), "f"(values[4]), "f"(values[5]), "f"(values[6]), "f"(values[7]));
                #pragma unroll
                for (int w3_1 = 0; w3_1 < 4; w3_1++) {
                    unsigned int word_1 = words2[4 + w3_1];
                    values[2 * w3_1] = __uint_as_float(word_1 << 16) * output_scale;
                    values[2 * w3_1 + 1] = __uint_as_float(word_1 & 4294901760u) * output_scale;
                }
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[1]) : "f"(values[0]), "f"(values[1]), "f"(values[2]), "f"(values[3]), "f"(values[4]), "f"(values[5]), "f"(values[6]), "f"(values[7]));
                asm volatile("st.global.v2.b32 [%0], {%1, %2};" :: "l"(out_fp4 + ((unsigned long long)vec_offset2 * 8) + (0)), "r"(packed[(0) + 0]), "r"(packed[(0) + 1]) : "memory");
                long long row64 = (long long)row_idx;
                long long vec64 = (long long)vec_idx2;
                long long nv64 = (long long)num_vecs_per_row;
                long long inner_k = vec64 % 4;
                long long k_tile = vec64 / 4;
                long long num_k_tiles = (nv64 + 3) / 4;
                long long sf_offset = 0;
                long long inner_m8 = row64 % 8;
                long long m_tile8 = row64 / 8;
                sf_offset = m_tile8 * (num_k_tiles * 32) + k_tile * 32 + inner_m8 * 4 + inner_k;
                {
                    unsigned short _sf_pair;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(sf_value));
                    *(reinterpret_cast<unsigned char*>(out_sf + sf_offset) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
}

} // extern "C"
