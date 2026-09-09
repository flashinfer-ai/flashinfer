/*
 * Copyright (c) 2023 by FlashInfer team.
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
static_assert(alignof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128-byte aligned");
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
#define M 9664
#define P 4
#define HEADS_PER_DESTINATION 14
#define ROWS_PER_DESTINATION 405888
#define SCALE_STRIDE 1623552

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

__global__ __launch_bounds__(256) void
kernel_cake_minimax_h3_mxfp8_pre_attention_sm103a_748ddc49ba1eb152240d(__nv_bfloat16* __restrict__ qkv_bf16, __nv_bfloat16* __restrict__ q_norm_weight, __nv_bfloat16* __restrict__ k_norm_weight, __nv_bfloat16* __restrict__ rope_cos_sin, uint8_t* __restrict__ out_q, uint8_t* __restrict__ out_sf, __nv_bfloat16* __restrict__ debug_q_bf16, __nv_bfloat16* __restrict__ debug_k_bf16, int write_debug, float eps)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int output_row = (unsigned int)(bid * 8) + warp;
    int total_rows = P * ROWS_PER_DESTINATION;
    if (output_row < total_rows) {
        int destination = output_row / ROWS_PER_DESTINATION;
        int row_in_destination = output_row % ROWS_PER_DESTINATION;
        int rows_per_token = HEADS_PER_DESTINATION * 3;
        int token = row_in_destination / rows_per_token;
        int head_kind = row_in_destination % rows_per_token;
        int local_head = head_kind / 3;
        int kind = head_kind % 3;
        int head = destination * HEADS_PER_DESTINATION + local_head;
        int dim = lane * 4;
        unsigned long long source_base = (((unsigned long long)token * 56 + (unsigned long long)head) * 3 + (unsigned long long)kind) * 128;
        float _vec_load_0[4];
        {
            uint2 _vld_0;
            _vld_0 = *reinterpret_cast<const uint2*>(qkv_bf16 + (source_base + (unsigned long long)dim) + 0);
            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_0[0 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _pair * 2])[1])
                    : "r"(_vpairs_0[_pair]));
            }
        }
        float output_values[4];
        if (kind < 2) {
            float sum_sq = 0.0f;
            if (lane < 16) {
                float _vec_load_1[8];
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(qkv_bf16 + (source_base + (unsigned long long)(lane * 8)) + 0);
                    uint4 _vld_1[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
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
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    float _fma_0 = __fmaf_rn(_vec_load_1[j], _vec_load_1[j], sum_sq);
                    sum_sq = _fma_0;
                }
            }
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 16);
            sum_sq += _shfl_xor_0;
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 8);
            sum_sq += _shfl_xor_1;
            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 4);
            sum_sq += _shfl_xor_2;
            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 2);
            sum_sq += _shfl_xor_3;
            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 1);
            sum_sq += _shfl_xor_4;
            float _rsqrt_0;
            asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(_rsqrt_0) : "f"(sum_sq / 128.0f + eps));
            float rstd = _rsqrt_0;
            float _vec_load_2[4];
            {
                uint2 _vld_2;
                _vld_2 = *reinterpret_cast<const uint2*>(((kind == 0) ? q_norm_weight + dim : k_norm_weight + dim) + 0);
                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_2[0 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _pair * 2])[1])
                        : "r"(_vpairs_2[_pair]));
                }
            }
            float normalized[4];
            #pragma unroll
            for (int j_1 = 0; j_1 < 4; j_1++) {
                __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(_vec_load_0[j_1] * rstd * _vec_load_2[j_1]);
                float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                normalized[j_1] = _cvt_f32_0;
            }
            int partner_lane = ((lane < 12) ? lane + 12 : ((lane < 24) ? lane - 12 : lane));
            int rope_col = ((dim < 48) ? dim : dim - 48);
            rope_col = ((dim < 96) ? rope_col : 0);
            float _vec_load_3[4];
            {
                uint2 _vld_3;
                _vld_3 = *reinterpret_cast<const uint2*>(rope_cos_sin + ((unsigned long long)token * 96 + (unsigned long long)rope_col) + 0);
                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_3[0 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _pair * 2])[1])
                        : "r"(_vpairs_3[_pair]));
                }
            }
            float _vec_load_4[4];
            {
                uint2 _vld_4;
                _vld_4 = *reinterpret_cast<const uint2*>(rope_cos_sin + ((unsigned long long)token * 96 + 48 + rope_col) + 0);
                uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_4[0 + _pair * 2])[0]), "=f"((&_vec_load_4[0 + _pair * 2])[1])
                        : "r"(_vpairs_4[_pair]));
                }
            }
            #pragma unroll
            for (int j_2 = 0; j_2 < 4; j_2++) {
                float _shfl_0 = __shfl_sync(0xFFFFFFFF, normalized[j_2], partner_lane);
                float partner = _shfl_0;
                float cos_term = normalized[j_2];
                float _mul_0;
                asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(cos_term), "f"(_vec_load_3[j_2]));
                cos_term = _mul_0;
                float sin_term = partner;
                float _mul_1;
                asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(sin_term), "f"(_vec_load_4[j_2]));
                sin_term = _mul_1;
                float rotated = normalized[j_2];
                if (dim < 48) {
                    rotated = cos_term - sin_term;
                } else if (dim < 96) {
                    rotated = cos_term + sin_term;
                }
                __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(rotated);
                float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                output_values[j_2] = _cvt_f32_1;
            }
            if (write_debug != 0) {
                unsigned long long debug_base = ((unsigned long long)token * 56 + (unsigned long long)head) * 128 + (unsigned long long)dim;
                if (kind == 0) {
                    {
                        uint2 _pk2;
                        __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                        _pk[0] = __floats2bfloat162_rn(output_values[0 + 0], output_values[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(output_values[0 + 2], output_values[0 + 3]);
                        *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(debug_q_bf16 + debug_base))[0]) = _pk2;
                    }
                } else {
                    {
                        uint2 _pk2;
                        __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                        _pk[0] = __floats2bfloat162_rn(output_values[0 + 0], output_values[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(output_values[0 + 2], output_values[0 + 3]);
                        *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(debug_k_bf16 + debug_base))[0]) = _pk2;
                    }
                }
            }
        } else {
            #pragma unroll
            for (int j_3 = 0; j_3 < 4; j_3++) {
                output_values[j_3] = _vec_load_0[j_3];
            }
        }
        float absolute[4];
        #pragma unroll
        for (int j_4 = 0; j_4 < 4; j_4++) {
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
        float absolute_max = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 4; _lr++) {
            absolute_max = max_noftz(absolute_max, absolute[_lr]);
        }
        float amax = absolute_max;
        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, amax, 1);
        float _max_0 = max_noftz(amax, _shfl_xor_5);
        amax = _max_0;
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, amax, 2);
        float _max_1 = max_noftz(amax, _shfl_xor_6);
        amax = _max_1;
        float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, amax, 4);
        float _max_2 = max_noftz(amax, _shfl_xor_7);
        amax = _max_2;
        float _rcp_0 = approx_rcp(448.0f);
        float block_scale = amax * _rcp_0;
        int scale_bits;
        scale_bits = reinterpret_cast<int*>(&block_scale)[0];
        int exponent = scale_bits >> 23 & 255;
        int mantissa = scale_bits & 8388607;
        int round_up = ((mantissa != 0 && (exponent != 0 || mantissa > 4194304)) ? 1 : 0);
        int scale_code = exponent + round_up;
        scale_code = ((scale_code < 255) ? scale_code : 254);
        scale_code = ((block_scale > 0.0f) ? scale_code : 0);
        int scale_value_bits = scale_code << 23;
        float scale_value;
        scale_value = reinterpret_cast<float*>(&scale_value_bits)[0];
        float inverse_scale;
        inverse_scale = 0.0f;
        if (scale_value != 0.0f) {
            float _rcp_1 = approx_rcp(scale_value);
            inverse_scale = _rcp_1;
        }
        float quant_values[4];
        #pragma unroll
        for (int j_5 = 0; j_5 < 4; j_5++) {
            quant_values[j_5] = output_values[j_5] * inverse_scale;
        }
        unsigned int packed[1];
        {
            uint32_t _packed;
            asm volatile("{\n\t"
                ".reg .b16 _lo;\n\t"
                ".reg .b16 _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}"
                : "=r"(_packed) : "f"(quant_values[0]), "f"(quant_values[1]),
                                   "f"(quant_values[2]), "f"(quant_values[3]));
            packed[0] = _packed;
        }
        unsigned long long output_base = (unsigned long long)output_row * 128 + (unsigned long long)dim;
        *(reinterpret_cast<int*>(out_q + output_base) + (0)) = packed[0];
        int block_col = lane / 8;
        if (lane % 8 == 0) {
            unsigned long long scale_offset = (unsigned long long)destination * (unsigned long long)SCALE_STRIDE + (unsigned long long)(row_in_destination / 128 * 512) + (unsigned long long)(row_in_destination % 32 * 16) + (unsigned long long)(row_in_destination % 128 / 32 * 4) + (unsigned long long)block_col;
            *(reinterpret_cast<unsigned char*>(out_sf + scale_offset) + (0)) = (unsigned char)(scale_code);
        }
    }
}

} // extern "C"
