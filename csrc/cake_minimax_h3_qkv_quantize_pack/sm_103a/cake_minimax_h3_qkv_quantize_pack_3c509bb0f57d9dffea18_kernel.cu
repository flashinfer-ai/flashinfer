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
#define THREADS 256
#define P 1
#define HEADS_PER_DESTINATION 56

#include <math_constants.h>

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

extern "C" {

__global__ __launch_bounds__(256, 5) void
kernel_cake_minimax_h3_qkv_quantize_pack_3c509bb0f57d9dffea18(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, float* __restrict__ out_global_scale, uint8_t* __restrict__ out_q, uint8_t* __restrict__ out_sf, int M, int token_stride, int head_stride, int ROWS_PER_DESTINATION, int SCALE_STRIDE)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int row_slot = lane / 8;
    int lane8 = lane % 8;
    int rows_per_token = HEADS_PER_DESTINATION * 3;
    int warps_per_destination = (M + 16 - 1) / 16 * (HEADS_PER_DESTINATION * 3);
    int destination = blockIdx.y;
    int warp_in_destination = bid * 8 + warp;
    if (bid == num_bids - 1) {
        int padding_rows = (128 - ROWS_PER_DESTINATION % 128) % 128;
        #pragma unroll
        for (int pad_pass = 0; pad_pass < 4; pad_pass++) {
            int pad_idx = tid + pad_pass * 256;
            if (pad_idx < padding_rows * 8) {
                int pad_row = ROWS_PER_DESTINATION + pad_idx / 8;
                int pad_col = pad_idx % 8;
                unsigned int pad_sr = (unsigned int)pad_row;
                unsigned int pad_bits = (pad_sr >> 7) * 1024 | (pad_sr & 31) << 4 | (pad_sr >> 5 & 3) << 2;
                unsigned long long pad_offset = (unsigned long long)destination * (unsigned long long)SCALE_STRIDE + (unsigned long long)pad_bits + (unsigned long long)(pad_col / 4 * 512 + pad_col % 4);
                *(reinterpret_cast<unsigned char*>(out_sf + pad_offset) + (0)) = (unsigned char)((unsigned int)0);
            }
        }
    } else if (warp_in_destination < warps_per_destination) {
        int token_group = warp_in_destination / rows_per_token;
        int head_kind = warp_in_destination % rows_per_token;
        int local_head = head_kind / 3;
        int kind = head_kind % 3;
        int head = destination * HEADS_PER_DESTINATION + local_head;
        int group_token = token_group * 16;
        int dim = lane8 * 16;
        unsigned long long head_offset = (unsigned long long)(head * head_stride);
        int row_base = destination * ROWS_PER_DESTINATION + head_kind;
        int lane_q_off = lane8 * 8;
        unsigned long long scale_lane_base = (unsigned long long)destination * (unsigned long long)SCALE_STRIDE + (unsigned long long)(lane8 / 4 * 512) + (unsigned long long)(lane8 % 4);
        float global_scale = out_global_scale[0];
        float _rcp_0 = approx_rcp(global_scale);
        float global_scale_rcp = _rcp_0;
        float _rcp_1 = approx_rcp(6.0f);
        float rcp_six = _rcp_1;
        unsigned int words[32];
        float quant_values[16];
        unsigned int packed[2];
        #pragma unroll
        for (int q0 = 0; q0 < 4; q0 += 4) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int load_row = group_token + (q0 + i) * 4 + row_slot;
                int load_token = ((load_row < M) ? load_row : M - 1);
                unsigned long long source = (unsigned long long)load_token * (unsigned long long)token_stride + head_offset + (unsigned long long)dim;
                {
                    asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(words[i * 8 + 0]), "=r"(words[i * 8 + 1]), "=r"(words[i * 8 + 2]), "=r"(words[i * 8 + 3]), "=r"(words[i * 8 + 4]), "=r"(words[i * 8 + 5]), "=r"(words[i * 8 + 6]), "=r"(words[i * 8 + 7]) : "l"((const void*)((const char*)(((kind == 0) ? q : ((kind == 1) ? k : v)) + source) + 0)) : "memory");
                }
            }
            #pragma unroll
            for (int i_1 = 0; i_1 < 4; i_1++) {
                int token = group_token + (q0 + i_1) * 4 + row_slot;
                if (token < M) {
                    uint32_t _bf16x2_abs_0;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_0) : "r"(words[i_1 * 8]));
                    unsigned int pair_max = _bf16x2_abs_0;
                    #pragma unroll
                    for (int j = 1; j < 8; j++) {
                        uint32_t _bf16x2_abs_1;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_1) : "r"(words[i_1 * 8 + j]));
                        uint32_t _bf16x2_max_0;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_0) : "r"(pair_max), "r"(_bf16x2_abs_1));
                        pair_max = _bf16x2_max_0;
                    }
                    uint16_t _bf16_max_0;
                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_0) : "h"((uint16_t)(pair_max & 65535)), "h"((uint16_t)(pair_max >> 16)));
                    float _cvt_f32_bf16_0;
                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_0) : "h"((uint16_t)(_bf16_max_0)));
                    float amax = _cvt_f32_bf16_0;
                    float sf_value = global_scale * (amax * rcp_six);
                    float _fp8_rt_0;
                    uint16_t _e4m3x2_1;
                    uint32_t _f16x2_1;
                    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(sf_value));
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                    uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
                    float sf_rounded = _fp8_rt_0;
                    float _rcp_2 = approx_rcp(sf_rounded * global_scale_rcp);
                    float _min_0 = fminf(_rcp_2, 3.4028234663852886e+38f);
                    float output_scale = _min_0;
                    float2 _f2_0 = make_float2(output_scale, output_scale);
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 8; j_1++) {
                        float _cvt_f32_bf16_1;
                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_1) : "h"((uint16_t)(words[i_1 * 8 + j_1] & 65535)));
                        float _cvt_f32_bf16_2;
                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_2) : "h"((uint16_t)(words[i_1 * 8 + j_1] >> 16)));
                        float2 _f2_1 = make_float2(_cvt_f32_bf16_1, _cvt_f32_bf16_2);
                        float2 _mul_f32x2_0;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&_f2_1), "l"(*(const unsigned long long*)&_f2_0));
                        quant_values[2 * j_1] = _mul_f32x2_0.x;
                        quant_values[2 * j_1 + 1] = _mul_f32x2_0.y;
                    }
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[1]) : "f"(quant_values[8]), "f"(quant_values[9]), "f"(quant_values[10]), "f"(quant_values[11]), "f"(quant_values[12]), "f"(quant_values[13]), "f"(quant_values[14]), "f"(quant_values[15]));
                    int row_in_destination = token * rows_per_token + head_kind;
                    int output_row = row_base + token * rows_per_token;
                    unsigned long long q_offset = (unsigned long long)output_row * 64 + (unsigned long long)lane_q_off;
                    *(reinterpret_cast<int*>(out_q + q_offset) + (0)) = packed[0];
                    *(reinterpret_cast<int*>(out_q + (q_offset + 4)) + (0)) = packed[1];
                    unsigned int scale_swizzle = (unsigned int)row_in_destination >> 7 << 10 | (((unsigned int)row_in_destination & 31) << 4 | ((unsigned int)row_in_destination >> 5 & 3) << 2);
                    unsigned long long scale_offset = scale_lane_base + (unsigned long long)scale_swizzle;
                    {
                        unsigned short _sf_pair;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(sf_value));
                        *(reinterpret_cast<unsigned char*>(out_sf + scale_offset) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                    }
                }
            }
        }
    }
}

} // extern "C"
