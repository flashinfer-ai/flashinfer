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
#define P 4
#define HEADS_PER_DESTINATION 14

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


__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max_noftz(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}


__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}


__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}


__device__ __forceinline__ void row_max_x32_accum(const float* sv, float2& acc) {
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (j % 2 == 0)
            acc.x = max_noftz(acc.x, max_noftz(sv[j*2], sv[j*2+1]));
        else
            acc.y = max_noftz(acc.y, max_noftz(sv[j*2], sv[j*2+1]));
    }
}

extern "C" {

__global__ __launch_bounds__(256, 4) void
kernel_cake_minimax_h3_nvfp4_pre_attention_43607f08690670a8d7f2(__nv_bfloat16* __restrict__ qkv_bf16, __nv_bfloat16* __restrict__ q_norm_weight, __nv_bfloat16* __restrict__ k_norm_weight, __nv_bfloat16* __restrict__ rope_cos_sin, float* __restrict__ out_global_scale, uint8_t* __restrict__ out_q, uint8_t* __restrict__ out_sf, __nv_bfloat16* __restrict__ debug_q_bf16, __nv_bfloat16* __restrict__ debug_k_bf16, int write_debug, float eps, int M, int ROWS_PER_DESTINATION, int SCALE_STRIDE)
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
    int token_quads = (M + 4 - 1) / 4;
    int warps_per_destination = token_quads * rows_per_token;
    int destination = blockIdx.y;
    int warp_in_destination = bid * 8 + warp;
    if (warp_in_destination < warps_per_destination) {
        int token_quad = warp_in_destination / rows_per_token;
        int head_kind = warp_in_destination % rows_per_token;
        int local_head = head_kind / 3;
        int kind = head_kind % 3;
        int head = destination * HEADS_PER_DESTINATION + local_head;
        int token = token_quad * 4 + row_slot;
        int load_token = ((token < M) ? token : M - 1);
        int row_in_destination = token * rows_per_token + head_kind;
        int output_row = destination * ROWS_PER_DESTINATION + row_in_destination;
        int dim = lane8 * 16;
        unsigned long long source_base = (((unsigned long long)load_token * 56 + (unsigned long long)head) * 3 + (unsigned long long)kind) * 128;
        float _vec_load_0[16];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(qkv_bf16 + (source_base + (unsigned long long)dim) + 0);
            uint4 _vld_0[2];
            #pragma unroll
            for (int _blk = 0; _blk < 2; _blk++) {
                _vld_0[_blk] = _vptr_0[_blk];
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_0[_pair]));
                }
            }
        }
        float output_values[16];
        float absolute[16];
        float quant_values[16];
        unsigned int packed[2];
        if (kind < 2) {
            int rope_col = ((dim < 48) ? dim : dim - 48);
            rope_col = ((dim < 96) ? rope_col : 0);
            float _vec_load_1[16];
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(rope_cos_sin + ((unsigned long long)load_token * 96 + (unsigned long long)rope_col) + 0);
                uint4 _vld_1[2];
                #pragma unroll
                for (int _blk = 0; _blk < 2; _blk++) {
                    _vld_1[_blk] = _vptr_1[_blk];
                    uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_1[_pair]));
                    }
                }
            }
            float _vec_load_2[16];
            {
                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(rope_cos_sin + ((unsigned long long)load_token * 96 + 48 + rope_col) + 0);
                uint4 _vld_2[2];
                #pragma unroll
                for (int _blk = 0; _blk < 2; _blk++) {
                    _vld_2[_blk] = _vptr_2[_blk];
                    uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_2[_pair]));
                    }
                }
            }
            float _vec_load_3[16];
            {
                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(((kind == 0) ? q_norm_weight + dim : k_norm_weight + dim) + 0);
                uint4 _vld_3[2];
                #pragma unroll
                for (int _blk = 0; _blk < 2; _blk++) {
                    _vld_3[_blk] = _vptr_3[_blk];
                    uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_3[_pair]));
                    }
                }
            }
            float sum_lo = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float _fma_0 = __fmaf_rn(_vec_load_0[j], _vec_load_0[j], sum_lo);
                sum_lo = _fma_0;
            }
            float sum_hi = 0.0f;
            #pragma unroll
            for (int j_1 = 8; j_1 < 16; j_1++) {
                float _fma_1 = __fmaf_rn(_vec_load_0[j_1], _vec_load_0[j_1], sum_hi);
                sum_hi = _fma_1;
            }
            float sum_sq = sum_lo + sum_hi;
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 1);
            sum_sq += _shfl_xor_0;
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 2);
            sum_sq += _shfl_xor_1;
            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 4);
            sum_sq += _shfl_xor_2;
            float _fdiv_rn_0 = __fdiv_rn(sum_sq, 128.0f);
            float mean_sq = _fdiv_rn_0;
            float _rsqrt_0 = rsqrtf(mean_sq + eps);
            float rstd = _rsqrt_0;
            float normalized[16];
            #pragma unroll
            for (int j_2 = 0; j_2 < 16; j_2++) {
                __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(_vec_load_0[j_2] * rstd * _vec_load_3[j_2]);
                float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                normalized[j_2] = _cvt_f32_0;
            }
            int partner_lane = ((lane8 < 3) ? lane + 3 : ((lane8 < 6) ? lane - 3 : lane));
            #pragma unroll
            for (int j_3 = 0; j_3 < 16; j_3++) {
                float _shfl_0;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_0) : "f"(normalized[j_3]), "r"(partner_lane));
                float partner = _shfl_0;
                float cos_term = normalized[j_3];
                float _mul_0 = cos_term * _vec_load_1[j_3];
                cos_term = _mul_0;
                float sin_term = partner;
                float _mul_1 = sin_term * _vec_load_2[j_3];
                sin_term = _mul_1;
                float rotated = normalized[j_3];
                if (dim < 48) {
                    rotated = cos_term - sin_term;
                } else if (dim < 96) {
                    rotated = cos_term + sin_term;
                }
                __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(rotated);
                float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                output_values[j_3] = _cvt_f32_1;
            }
            if (write_debug != 0) {
                if (token < M) {
                    unsigned long long debug_base = ((unsigned long long)token * 56 + (unsigned long long)head) * 128 + (unsigned long long)dim;
                    if (kind == 0) {
                        {
                            __nv_bfloat162 _pk[8];
                            _pk[0] = __floats2bfloat162_rn(output_values[0 + 0], output_values[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(output_values[0 + 2], output_values[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(output_values[0 + 4], output_values[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(output_values[0 + 6], output_values[0 + 7]);
                            _pk[4] = __floats2bfloat162_rn(output_values[0 + 8], output_values[0 + 9]);
                            _pk[5] = __floats2bfloat162_rn(output_values[0 + 10], output_values[0 + 11]);
                            _pk[6] = __floats2bfloat162_rn(output_values[0 + 12], output_values[0 + 13]);
                            _pk[7] = __floats2bfloat162_rn(output_values[0 + 14], output_values[0 + 15]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(debug_q_bf16 + debug_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(debug_q_bf16 + debug_base))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                        }
                    } else {
                        {
                            __nv_bfloat162 _pk[8];
                            _pk[0] = __floats2bfloat162_rn(output_values[0 + 0], output_values[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(output_values[0 + 2], output_values[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(output_values[0 + 4], output_values[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(output_values[0 + 6], output_values[0 + 7]);
                            _pk[4] = __floats2bfloat162_rn(output_values[0 + 8], output_values[0 + 9]);
                            _pk[5] = __floats2bfloat162_rn(output_values[0 + 10], output_values[0 + 11]);
                            _pk[6] = __floats2bfloat162_rn(output_values[0 + 12], output_values[0 + 13]);
                            _pk[7] = __floats2bfloat162_rn(output_values[0 + 14], output_values[0 + 15]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(debug_k_bf16 + debug_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(debug_k_bf16 + debug_base))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                        }
                    }
                }
            }
            #pragma unroll
            for (int j_4 = 0; j_4 < 16; j_4++) {
                absolute[j_4] = output_values[j_4];
            }
            float _fabs_0 = fabsf(absolute[0]);
            absolute[0] = _fabs_0;
            float _fabs_1 = fabsf(absolute[1]);
            absolute[1] = _fabs_1;
            float _fabs_2 = fabsf(absolute[2]);
            absolute[2] = _fabs_2;
            float _fabs_3 = fabsf(absolute[3]);
            absolute[3] = _fabs_3;
            float _fabs_4 = fabsf(absolute[4]);
            absolute[4] = _fabs_4;
            float _fabs_5 = fabsf(absolute[5]);
            absolute[5] = _fabs_5;
            float _fabs_6 = fabsf(absolute[6]);
            absolute[6] = _fabs_6;
            float _fabs_7 = fabsf(absolute[7]);
            absolute[7] = _fabs_7;
            float _fabs_8 = fabsf(absolute[8]);
            absolute[8] = _fabs_8;
            float _fabs_9 = fabsf(absolute[9]);
            absolute[9] = _fabs_9;
            float _fabs_10 = fabsf(absolute[10]);
            absolute[10] = _fabs_10;
            float _fabs_11 = fabsf(absolute[11]);
            absolute[11] = _fabs_11;
            float _fabs_12 = fabsf(absolute[12]);
            absolute[12] = _fabs_12;
            float _fabs_13 = fabsf(absolute[13]);
            absolute[13] = _fabs_13;
            float _fabs_14 = fabsf(absolute[14]);
            absolute[14] = _fabs_14;
            float _fabs_15 = fabsf(absolute[15]);
            absolute[15] = _fabs_15;
            float absolute_max = absolute[0];
            #pragma unroll
            for (int _lr = 1; _lr < 16; _lr++) {
                absolute_max = max_noftz(absolute_max, absolute[_lr]);
            }
            float amax = absolute_max;
            float global_scale = out_global_scale[0];
            float _rcp_0 = approx_rcp(6.0f);
            float sf_value = global_scale * (amax * _rcp_0);
            float _fp8_rt_0;
            uint16_t _e4m3x2_4;
            uint32_t _f16x2_4;
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_4) : "f"(0.0f), "f"(sf_value));
            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_4) : "h"(_e4m3x2_4));
            uint16_t _fp8_h0_4 = (uint16_t)(_f16x2_4 & 0xFFFFu);
            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_4));
            float sf_rounded = _fp8_rt_0;
            float _rcp_1 = approx_rcp(global_scale);
            float _rcp_2 = approx_rcp(sf_rounded * _rcp_1);
            float _min_0 = fminf(_rcp_2, 3.4028234663852886e+38f);
            float output_scale = _min_0;
            #pragma unroll
            for (int j_5 = 0; j_5 < 16; j_5++) {
                quant_values[j_5] = output_values[j_5] * output_scale;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[1]) : "f"(quant_values[8]), "f"(quant_values[9]), "f"(quant_values[10]), "f"(quant_values[11]), "f"(quant_values[12]), "f"(quant_values[13]), "f"(quant_values[14]), "f"(quant_values[15]));
            if (token < M) {
                *(reinterpret_cast<int*>(out_q + ((unsigned long long)output_row * 64 + (unsigned long long)(lane8 * 8))) + (0)) = packed[0];
                *(reinterpret_cast<int*>(out_q + ((unsigned long long)output_row * 64 + (unsigned long long)(lane8 * 8) + 4)) + (0)) = packed[1];
                unsigned long long scale_offset = (unsigned long long)destination * (unsigned long long)SCALE_STRIDE + (unsigned long long)(row_in_destination / 128) * 1024 + (unsigned long long)(lane8 / 4 * 512) + (unsigned long long)(row_in_destination % 32 * 16) + (unsigned long long)(row_in_destination % 128 / 32 * 4) + (unsigned long long)(lane8 % 4);
                {
                    unsigned short _sf_pair;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(sf_value));
                    *(reinterpret_cast<unsigned char*>(out_sf + scale_offset) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                }
            }
        } else {
            #pragma unroll
            for (int j_6 = 0; j_6 < 16; j_6++) {
                absolute[j_6] = _vec_load_0[j_6];
            }
            float _fabs_16 = fabsf(absolute[0]);
            absolute[0] = _fabs_16;
            float _fabs_17 = fabsf(absolute[1]);
            absolute[1] = _fabs_17;
            float _fabs_18 = fabsf(absolute[2]);
            absolute[2] = _fabs_18;
            float _fabs_19 = fabsf(absolute[3]);
            absolute[3] = _fabs_19;
            float _fabs_20 = fabsf(absolute[4]);
            absolute[4] = _fabs_20;
            float _fabs_21 = fabsf(absolute[5]);
            absolute[5] = _fabs_21;
            float _fabs_22 = fabsf(absolute[6]);
            absolute[6] = _fabs_22;
            float _fabs_23 = fabsf(absolute[7]);
            absolute[7] = _fabs_23;
            float _fabs_24 = fabsf(absolute[8]);
            absolute[8] = _fabs_24;
            float _fabs_25 = fabsf(absolute[9]);
            absolute[9] = _fabs_25;
            float _fabs_26 = fabsf(absolute[10]);
            absolute[10] = _fabs_26;
            float _fabs_27 = fabsf(absolute[11]);
            absolute[11] = _fabs_27;
            float _fabs_28 = fabsf(absolute[12]);
            absolute[12] = _fabs_28;
            float _fabs_29 = fabsf(absolute[13]);
            absolute[13] = _fabs_29;
            float _fabs_30 = fabsf(absolute[14]);
            absolute[14] = _fabs_30;
            float _fabs_31 = fabsf(absolute[15]);
            absolute[15] = _fabs_31;
            float absolute_max_1 = absolute[0];
            #pragma unroll
            for (int _lr = 1; _lr < 16; _lr++) {
                absolute_max_1 = max_noftz(absolute_max_1, absolute[_lr]);
            }
            float amax_1 = absolute_max_1;
            float global_scale_1 = out_global_scale[0];
            float _rcp_3 = approx_rcp(6.0f);
            float sf_value_1 = global_scale_1 * (amax_1 * _rcp_3);
            float _fp8_rt_1;
            uint16_t _e4m3x2_5;
            uint32_t _f16x2_5;
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_5) : "f"(0.0f), "f"(sf_value_1));
            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_5) : "h"(_e4m3x2_5));
            uint16_t _fp8_h0_5 = (uint16_t)(_f16x2_5 & 0xFFFFu);
            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_1) : "h"(_fp8_h0_5));
            float sf_rounded_1 = _fp8_rt_1;
            float _rcp_4 = approx_rcp(global_scale_1);
            float _rcp_5 = approx_rcp(sf_rounded_1 * _rcp_4);
            float _min_1 = fminf(_rcp_5, 3.4028234663852886e+38f);
            float output_scale_1 = _min_1;
            #pragma unroll
            for (int j_7 = 0; j_7 < 16; j_7++) {
                quant_values[j_7] = _vec_load_0[j_7] * output_scale_1;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[1]) : "f"(quant_values[8]), "f"(quant_values[9]), "f"(quant_values[10]), "f"(quant_values[11]), "f"(quant_values[12]), "f"(quant_values[13]), "f"(quant_values[14]), "f"(quant_values[15]));
            if (token < M) {
                *(reinterpret_cast<int*>(out_q + ((unsigned long long)output_row * 64 + (unsigned long long)(lane8 * 8))) + (0)) = packed[0];
                *(reinterpret_cast<int*>(out_q + ((unsigned long long)output_row * 64 + (unsigned long long)(lane8 * 8) + 4)) + (0)) = packed[1];
                unsigned long long scale_offset_1 = (unsigned long long)destination * (unsigned long long)SCALE_STRIDE + (unsigned long long)(row_in_destination / 128) * 1024 + (unsigned long long)(lane8 / 4 * 512) + (unsigned long long)(row_in_destination % 32 * 16) + (unsigned long long)(row_in_destination % 128 / 32 * 4) + (unsigned long long)(lane8 % 4);
                {
                    unsigned short _sf_pair;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(sf_value_1));
                    *(reinterpret_cast<unsigned char*>(out_sf + scale_offset_1) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                }
            }
        }
    }
}

} // extern "C"
