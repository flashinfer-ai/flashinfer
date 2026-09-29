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
#define P 8
#define HEADS_PER_DESTINATION 7

#include <math_constants.h>

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__global__ __launch_bounds__(256, 5) void
kernel_cake_minimax_h3_qkv_quantize_pack_3e6cf43ecb7b415ff17c(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, uint8_t* __restrict__ out_q, uint8_t* __restrict__ out_sf, int M, int token_stride, int head_stride, int ROWS_PER_DESTINATION, int SCALE_STRIDE)
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
        for (int pad_pass = 0; pad_pass < 2; pad_pass++) {
            int pad_idx = tid + pad_pass * 256;
            if (pad_idx < padding_rows * 4) {
                int pad_row = ROWS_PER_DESTINATION + pad_idx / 4;
                int pad_col = pad_idx % 4;
                unsigned int pad_sr = (unsigned int)pad_row;
                unsigned int pad_bits = (pad_sr >> 7) * 512 | (pad_sr & 31) << 4 | (pad_sr >> 5 & 3) << 2;
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
        int lane_q_off = lane8 * 16;
        int block_col = lane8 / 2;
        unsigned long long scale_lane_base = (unsigned long long)destination * (unsigned long long)SCALE_STRIDE + (unsigned long long)block_col;
        float _rcp_0 = approx_rcp(448.0f);
        float rcp_448 = _rcp_0;
        unsigned int words[32];
        float quant_values[16];
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
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, amax, 1);
                float _max_0 = max_noftz(amax, _shfl_xor_0);
                amax = _max_0;
                if (token < M) {
                    float block_scale = amax * rcp_448;
                    unsigned int scale_bits = __as_u32(block_scale);
                    unsigned int exponent = scale_bits >> 23 & 255;
                    unsigned int mantissa = scale_bits & 8388607;
                    unsigned int has_mantissa = ((mantissa != 0) ? 1 : 0);
                    unsigned int normal = ((exponent != 0) ? 1 : 0);
                    unsigned int large_subnormal = ((mantissa > 4194304) ? 1 : 0);
                    unsigned int _min_0 = ((exponent + (has_mantissa & (normal | large_subnormal))) < (254) ? (exponent + (has_mantissa & (normal | large_subnormal))) : (254));
                    unsigned int scale_code = _min_0;
                    unsigned int inverse_nonzero_bits = 254 - scale_code << 23;
                    unsigned int zero_bits = 0;
                    unsigned int inverse_bits = ((scale_code == 0) ? zero_bits : inverse_nonzero_bits);
                    float inverse = 0.0f;
                    inverse = reinterpret_cast<float*>(&inverse_bits)[0];
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 8; j_1++) {
                        float _cvt_f32_bf16_1;
                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_1) : "h"((uint16_t)(words[i_1 * 8 + j_1] & 65535)));
                        quant_values[2 * j_1] = _cvt_f32_bf16_1 * inverse;
                        float _cvt_f32_bf16_2;
                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_2) : "h"((uint16_t)(words[i_1 * 8 + j_1] >> 16)));
                        quant_values[2 * j_1 + 1] = _cvt_f32_bf16_2 * inverse;
                    }
                    int row_in_destination = token * rows_per_token + head_kind;
                    int output_row = row_base + token * rows_per_token;
                    unsigned long long q_offset = (unsigned long long)output_row * 128 + (unsigned long long)lane_q_off;
                    {
                        unsigned int _fp8_pk[4];
                        asm("{\n\t"
                            ".reg .b16 _lo, _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}\n"
                            : "=r"(_fp8_pk[0]) : "f"(quant_values[0 + 0]), "f"(quant_values[0 + 1]), "f"(quant_values[0 + 2]), "f"(quant_values[0 + 3]));
                        asm("{\n\t"
                            ".reg .b16 _lo, _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}\n"
                            : "=r"(_fp8_pk[1]) : "f"(quant_values[0 + 4]), "f"(quant_values[0 + 5]), "f"(quant_values[0 + 6]), "f"(quant_values[0 + 7]));
                        asm("{\n\t"
                            ".reg .b16 _lo, _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}\n"
                            : "=r"(_fp8_pk[2]) : "f"(quant_values[0 + 8]), "f"(quant_values[0 + 9]), "f"(quant_values[0 + 10]), "f"(quant_values[0 + 11]));
                        asm("{\n\t"
                            ".reg .b16 _lo, _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}\n"
                            : "=r"(_fp8_pk[3]) : "f"(quant_values[0 + 12]), "f"(quant_values[0 + 13]), "f"(quant_values[0 + 14]), "f"(quant_values[0 + 15]));
                        *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(out_q + q_offset) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                    }
                    if (lane8 % 2 == 0) {
                        unsigned int scale_swizzle = (unsigned int)row_in_destination >> 7 << 9 | (((unsigned int)row_in_destination & 31) << 4 | ((unsigned int)row_in_destination >> 5 & 3) << 2);
                        unsigned long long scale_offset = scale_lane_base + (unsigned long long)scale_swizzle;
                        *(reinterpret_cast<unsigned char*>(out_sf + scale_offset) + (0)) = (unsigned char)(scale_code);
                    }
                }
            }
        }
    }
}

} // extern "C"
