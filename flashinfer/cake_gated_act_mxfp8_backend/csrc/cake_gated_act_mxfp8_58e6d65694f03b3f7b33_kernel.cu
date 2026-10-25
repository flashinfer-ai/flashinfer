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
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 8

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


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


__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)


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

__device__ __forceinline__ __nv_bfloat162 __as_bf16x2(unsigned int v) {
    __nv_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(v);
    raw.y = static_cast<unsigned short>(v >> 16);
    return __nv_bfloat162(raw);
}

__device__ __forceinline__ float2 cake_bf16x2_to_f32x2(__nv_bfloat162 v) {
    const unsigned int w = *reinterpret_cast<const unsigned int*>(&v);
    return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}

__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_gated_act_mxfp8_58e6d65694f03b3f7b33(__nv_bfloat16* __restrict__ gated_input, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int half = tid & 1;
    int blk = tid >> 1 & 3;
    int row = tid >> 3;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int grow = by * 32 + row;
    int col = bx * 128 + blk * 32 + half * 16;
    int gate_index = grow * (2 * K) + col;
    unsigned int _vec_load_0[8];
    {
        uint32_t _iv_0_0;
        uint32_t _iv_0_1;
        uint32_t _iv_0_2;
        uint32_t _iv_0_3;
        uint32_t _iv_0_4;
        uint32_t _iv_0_5;
        uint32_t _iv_0_6;
        uint32_t _iv_0_7;
        asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_iv_0_0), "=r"(_iv_0_1), "=r"(_iv_0_2), "=r"(_iv_0_3), "=r"(_iv_0_4), "=r"(_iv_0_5), "=r"(_iv_0_6), "=r"(_iv_0_7) : "l"((const void*)(gated_input + gate_index + (0))) : "memory");
        _vec_load_0[0 + 0] = (uint32_t)_iv_0_0;
        _vec_load_0[0 + 1] = (uint32_t)_iv_0_1;
        _vec_load_0[0 + 2] = (uint32_t)_iv_0_2;
        _vec_load_0[0 + 3] = (uint32_t)_iv_0_3;
        _vec_load_0[0 + 4] = (uint32_t)_iv_0_4;
        _vec_load_0[0 + 5] = (uint32_t)_iv_0_5;
        _vec_load_0[0 + 6] = (uint32_t)_iv_0_6;
        _vec_load_0[0 + 7] = (uint32_t)_iv_0_7;
    }
    unsigned int _vec_load_1[8];
    {
        uint32_t _iv_1_0;
        uint32_t _iv_1_1;
        uint32_t _iv_1_2;
        uint32_t _iv_1_3;
        uint32_t _iv_1_4;
        uint32_t _iv_1_5;
        uint32_t _iv_1_6;
        uint32_t _iv_1_7;
        asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_iv_1_0), "=r"(_iv_1_1), "=r"(_iv_1_2), "=r"(_iv_1_3), "=r"(_iv_1_4), "=r"(_iv_1_5), "=r"(_iv_1_6), "=r"(_iv_1_7) : "l"((const void*)(gated_input + (gate_index + K) + (0))) : "memory");
        _vec_load_1[0 + 0] = (uint32_t)_iv_1_0;
        _vec_load_1[0 + 1] = (uint32_t)_iv_1_1;
        _vec_load_1[0 + 2] = (uint32_t)_iv_1_2;
        _vec_load_1[0 + 3] = (uint32_t)_iv_1_3;
        _vec_load_1[0 + 4] = (uint32_t)_iv_1_4;
        _vec_load_1[0 + 5] = (uint32_t)_iv_1_5;
        _vec_load_1[0 + 6] = (uint32_t)_iv_1_6;
        _vec_load_1[0 + 7] = (uint32_t)_iv_1_7;
    }
    unsigned int packed[8];
    __nv_bfloat162 lowest = __as_bf16x2(_vec_load_0[0]);
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[7]));
    float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(lowest);
    float _min_3;
    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_3) : "f"(_cvt_f32_0.x), "f"(_cvt_f32_0.y));
    int _warp_vote_2 = __all_sync(0xFFFFFFFF, _min_3 > -87.0f);
    if (_warp_vote_2 != 0) {
        float2 _cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 _cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float e0_0 = approx_exp2((-_cvt_f32_1.x) * 1.442695f);
        float e1_0 = approx_exp2((-_cvt_f32_1.y) * 1.442695f);
        float d0_0 = 1.0f + e0_0;
        float d1_0 = 1.0f + e1_0;
        float r0_0 = approx_rcp(d0_0);
        float r1_0 = approx_rcp(d1_0);
        float _fma_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(-d0_0), "f"(r0_0), "f"(1.0f));
        float _fma_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(r0_0), "f"(_fma_5), "f"(r0_0));
        float _fma_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(-d1_0), "f"(r1_0), "f"(1.0f));
        float _fma_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(r1_0), "f"(_fma_7), "f"(r1_0));
        float2 _f32x2_mul_a_8 = mul_f32x2_noftz(_cvt_f32_1, make_float2(_fma_4, _fma_6));
        float2 _mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_8), "l"(*(const unsigned long long*)&_cvt_f32_2));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(_mul_0.x, _mul_0.y)));
        float2 _cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 _cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float e0_1 = approx_exp2((-_cvt_f32_3.x) * 1.442695f);
        float e1_1 = approx_exp2((-_cvt_f32_3.y) * 1.442695f);
        float d0_1 = 1.0f + e0_1;
        float d1_1 = 1.0f + e1_1;
        float r0_1 = approx_rcp(d0_1);
        float r1_1 = approx_rcp(d1_1);
        float _fma_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(-d0_1), "f"(r0_1), "f"(1.0f));
        float _fma_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(r0_1), "f"(_fma_10), "f"(r0_1));
        float _fma_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(-d1_1), "f"(r1_1), "f"(1.0f));
        float _fma_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(r1_1), "f"(_fma_12), "f"(r1_1));
        float2 _f32x2_mul_a_13 = mul_f32x2_noftz(_cvt_f32_3, make_float2(_fma_9, _fma_11));
        float2 _mul_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_13), "l"(*(const unsigned long long*)&_cvt_f32_4));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(_mul_1.x, _mul_1.y)));
        float2 _cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 _cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float e0_2 = approx_exp2((-_cvt_f32_5.x) * 1.442695f);
        float e1_2 = approx_exp2((-_cvt_f32_5.y) * 1.442695f);
        float d0_2 = 1.0f + e0_2;
        float d1_2 = 1.0f + e1_2;
        float r0_2 = approx_rcp(d0_2);
        float r1_2 = approx_rcp(d1_2);
        float _fma_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(-d0_2), "f"(r0_2), "f"(1.0f));
        float _fma_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(r0_2), "f"(_fma_15), "f"(r0_2));
        float _fma_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(-d1_2), "f"(r1_2), "f"(1.0f));
        float _fma_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(r1_2), "f"(_fma_17), "f"(r1_2));
        float2 _f32x2_mul_a_18 = mul_f32x2_noftz(_cvt_f32_5, make_float2(_fma_14, _fma_16));
        float2 _mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&_cvt_f32_6));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(_mul_2.x, _mul_2.y)));
        float2 _cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 _cvt_f32_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float e0_3 = approx_exp2((-_cvt_f32_7.x) * 1.442695f);
        float e1_3 = approx_exp2((-_cvt_f32_7.y) * 1.442695f);
        float d0_3 = 1.0f + e0_3;
        float d1_3 = 1.0f + e1_3;
        float r0_3 = approx_rcp(d0_3);
        float r1_3 = approx_rcp(d1_3);
        float _fma_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(-d0_3), "f"(r0_3), "f"(1.0f));
        float _fma_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(r0_3), "f"(_fma_20), "f"(r0_3));
        float _fma_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(-d1_3), "f"(r1_3), "f"(1.0f));
        float _fma_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(r1_3), "f"(_fma_22), "f"(r1_3));
        float2 _f32x2_mul_a_23 = mul_f32x2_noftz(_cvt_f32_7, make_float2(_fma_19, _fma_21));
        float2 _mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&_cvt_f32_8));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(_mul_3.x, _mul_3.y)));
        float2 _cvt_f32_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 _cvt_f32_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float e0_4 = approx_exp2((-_cvt_f32_9.x) * 1.442695f);
        float e1_4 = approx_exp2((-_cvt_f32_9.y) * 1.442695f);
        float d0_4 = 1.0f + e0_4;
        float d1_4 = 1.0f + e1_4;
        float r0_4 = approx_rcp(d0_4);
        float r1_4 = approx_rcp(d1_4);
        float _fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(-d0_4), "f"(r0_4), "f"(1.0f));
        float _fma_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(r0_4), "f"(_fma_25), "f"(r0_4));
        float _fma_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(-d1_4), "f"(r1_4), "f"(1.0f));
        float _fma_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(r1_4), "f"(_fma_27), "f"(r1_4));
        float2 _f32x2_mul_a_28 = mul_f32x2_noftz(_cvt_f32_9, make_float2(_fma_24, _fma_26));
        float2 _mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&_cvt_f32_10));
        packed[4] = __as_u32(__float22bfloat162_rn(make_float2(_mul_4.x, _mul_4.y)));
        float2 _cvt_f32_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 _cvt_f32_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float e0_5 = approx_exp2((-_cvt_f32_11.x) * 1.442695f);
        float e1_5 = approx_exp2((-_cvt_f32_11.y) * 1.442695f);
        float d0_5 = 1.0f + e0_5;
        float d1_5 = 1.0f + e1_5;
        float r0_5 = approx_rcp(d0_5);
        float r1_5 = approx_rcp(d1_5);
        float _fma_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(-d0_5), "f"(r0_5), "f"(1.0f));
        float _fma_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(r0_5), "f"(_fma_30), "f"(r0_5));
        float _fma_32;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(-d1_5), "f"(r1_5), "f"(1.0f));
        float _fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(r1_5), "f"(_fma_32), "f"(r1_5));
        float2 _f32x2_mul_a_33 = mul_f32x2_noftz(_cvt_f32_11, make_float2(_fma_29, _fma_31));
        float2 _mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&_cvt_f32_12));
        packed[5] = __as_u32(__float22bfloat162_rn(make_float2(_mul_5.x, _mul_5.y)));
        float2 _cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 _cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float e0_6 = approx_exp2((-_cvt_f32_13.x) * 1.442695f);
        float e1_6 = approx_exp2((-_cvt_f32_13.y) * 1.442695f);
        float d0_6 = 1.0f + e0_6;
        float d1_6 = 1.0f + e1_6;
        float r0_6 = approx_rcp(d0_6);
        float r1_6 = approx_rcp(d1_6);
        float _fma_35;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(-d0_6), "f"(r0_6), "f"(1.0f));
        float _fma_34;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(r0_6), "f"(_fma_35), "f"(r0_6));
        float _fma_37;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(-d1_6), "f"(r1_6), "f"(1.0f));
        float _fma_36;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(r1_6), "f"(_fma_37), "f"(r1_6));
        float2 _f32x2_mul_a_38 = mul_f32x2_noftz(_cvt_f32_13, make_float2(_fma_34, _fma_36));
        float2 _mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_38), "l"(*(const unsigned long long*)&_cvt_f32_14));
        packed[6] = __as_u32(__float22bfloat162_rn(make_float2(_mul_6.x, _mul_6.y)));
        float2 _cvt_f32_15 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 _cvt_f32_16 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float e0_7 = approx_exp2((-_cvt_f32_15.x) * 1.442695f);
        float e1_7 = approx_exp2((-_cvt_f32_15.y) * 1.442695f);
        float d0_7 = 1.0f + e0_7;
        float d1_7 = 1.0f + e1_7;
        float r0_7 = approx_rcp(d0_7);
        float r1_7 = approx_rcp(d1_7);
        float _fma_40;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(-d0_7), "f"(r0_7), "f"(1.0f));
        float _fma_39;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(r0_7), "f"(_fma_40), "f"(r0_7));
        float _fma_42;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(-d1_7), "f"(r1_7), "f"(1.0f));
        float _fma_41;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(r1_7), "f"(_fma_42), "f"(r1_7));
        float2 _f32x2_mul_a_43 = mul_f32x2_noftz(_cvt_f32_15, make_float2(_fma_39, _fma_41));
        float2 _mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_43), "l"(*(const unsigned long long*)&_cvt_f32_16));
        packed[7] = __as_u32(__float22bfloat162_rn(make_float2(_mul_7.x, _mul_7.y)));
    } else {
        int band = 0;
        float2 _cvt_f32_17 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 _cvt_f32_18 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float e0_8 = approx_exp2((-_cvt_f32_17.x) * 1.442695f);
        float e1_8 = approx_exp2((-_cvt_f32_17.y) * 1.442695f);
        float d0_8 = 1.0f + e0_8;
        float d1_8 = 1.0f + e1_8;
        float r0_8 = approx_rcp(d0_8);
        float r1_8 = approx_rcp(d1_8);
        float _fma_44;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(-d0_8), "f"(r0_8), "f"(1.0f));
        float _fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(r0_8), "f"(_fma_44), "f"(r0_8));
        float _fma_45;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(-d1_8), "f"(r1_8), "f"(1.0f));
        float _fma_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(r1_8), "f"(_fma_45), "f"(r1_8));
        float t0_8 = _fma_0;
        float t1_8 = _fma_1;
        band = band | ((fminf(t0_8, t1_8) == 0.0f) ? 1 : 0);
        float2 _f32x2_mul_a_46 = mul_f32x2_noftz(_cvt_f32_17, make_float2(max_noftz(t0_8, 0.0f), max_noftz(t1_8, 0.0f)));
        float2 _mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_46), "l"(*(const unsigned long long*)&_cvt_f32_18));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(_mul_8.x, _mul_8.y)));
        float2 _cvt_f32_19 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 _cvt_f32_20 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float e0_9 = approx_exp2((-_cvt_f32_19.x) * 1.442695f);
        float e1_9 = approx_exp2((-_cvt_f32_19.y) * 1.442695f);
        float d0_9 = 1.0f + e0_9;
        float d1_9 = 1.0f + e1_9;
        float r0_9 = approx_rcp(d0_9);
        float r1_9 = approx_rcp(d1_9);
        float _fma_47;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(-d0_9), "f"(r0_9), "f"(1.0f));
        float _fma_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(r0_9), "f"(_fma_47), "f"(r0_9));
        float _fma_48;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(-d1_9), "f"(r1_9), "f"(1.0f));
        float _fma_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(r1_9), "f"(_fma_48), "f"(r1_9));
        float t0_9 = _fma_2;
        float t1_9 = _fma_3;
        band = band | ((fminf(t0_9, t1_9) == 0.0f) ? 2 : 0);
        float2 _f32x2_mul_a_49 = mul_f32x2_noftz(_cvt_f32_19, make_float2(max_noftz(t0_9, 0.0f), max_noftz(t1_9, 0.0f)));
        float2 _mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_49), "l"(*(const unsigned long long*)&_cvt_f32_20));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(_mul_9.x, _mul_9.y)));
        float2 _cvt_f32_21 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 _cvt_f32_22 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float e0_10 = approx_exp2((-_cvt_f32_21.x) * 1.442695f);
        float e1_10 = approx_exp2((-_cvt_f32_21.y) * 1.442695f);
        float d0_10 = 1.0f + e0_10;
        float d1_10 = 1.0f + e1_10;
        float r0_10 = approx_rcp(d0_10);
        float r1_10 = approx_rcp(d1_10);
        float _fma_50;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(-d0_10), "f"(r0_10), "f"(1.0f));
        float _fma_4_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_4_1) : "f"(r0_10), "f"(_fma_50), "f"(r0_10));
        float _fma_51;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(-d1_10), "f"(r1_10), "f"(1.0f));
        float _fma_5_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5_1) : "f"(r1_10), "f"(_fma_51), "f"(r1_10));
        float t0_10 = _fma_4_1;
        float t1_10 = _fma_5_1;
        band = band | ((fminf(t0_10, t1_10) == 0.0f) ? 4 : 0);
        float2 _f32x2_mul_a_52 = mul_f32x2_noftz(_cvt_f32_21, make_float2(max_noftz(t0_10, 0.0f), max_noftz(t1_10, 0.0f)));
        float2 _mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_52), "l"(*(const unsigned long long*)&_cvt_f32_22));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(_mul_10.x, _mul_10.y)));
        float2 _cvt_f32_23 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 _cvt_f32_24 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float e0_11 = approx_exp2((-_cvt_f32_23.x) * 1.442695f);
        float e1_11 = approx_exp2((-_cvt_f32_23.y) * 1.442695f);
        float d0_11 = 1.0f + e0_11;
        float d1_11 = 1.0f + e1_11;
        float r0_11 = approx_rcp(d0_11);
        float r1_11 = approx_rcp(d1_11);
        float _fma_53;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(-d0_11), "f"(r0_11), "f"(1.0f));
        float _fma_6_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6_1) : "f"(r0_11), "f"(_fma_53), "f"(r0_11));
        float _fma_54;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(-d1_11), "f"(r1_11), "f"(1.0f));
        float _fma_7_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7_1) : "f"(r1_11), "f"(_fma_54), "f"(r1_11));
        float t0_11 = _fma_6_1;
        float t1_11 = _fma_7_1;
        band = band | ((fminf(t0_11, t1_11) == 0.0f) ? 8 : 0);
        float2 _f32x2_mul_a_55 = mul_f32x2_noftz(_cvt_f32_23, make_float2(max_noftz(t0_11, 0.0f), max_noftz(t1_11, 0.0f)));
        float2 _mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_55), "l"(*(const unsigned long long*)&_cvt_f32_24));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(_mul_11.x, _mul_11.y)));
        float2 _cvt_f32_25 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 _cvt_f32_26 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float e0_12 = approx_exp2((-_cvt_f32_25.x) * 1.442695f);
        float e1_12 = approx_exp2((-_cvt_f32_25.y) * 1.442695f);
        float d0_12 = 1.0f + e0_12;
        float d1_12 = 1.0f + e1_12;
        float r0_12 = approx_rcp(d0_12);
        float r1_12 = approx_rcp(d1_12);
        float _fma_56;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(-d0_12), "f"(r0_12), "f"(1.0f));
        float _fma_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(r0_12), "f"(_fma_56), "f"(r0_12));
        float _fma_57;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(-d1_12), "f"(r1_12), "f"(1.0f));
        float _fma_9_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9_1) : "f"(r1_12), "f"(_fma_57), "f"(r1_12));
        float t0_12 = _fma_8;
        float t1_12 = _fma_9_1;
        band = band | ((fminf(t0_12, t1_12) == 0.0f) ? 16 : 0);
        float2 _f32x2_mul_a_58 = mul_f32x2_noftz(_cvt_f32_25, make_float2(max_noftz(t0_12, 0.0f), max_noftz(t1_12, 0.0f)));
        float2 _mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_58), "l"(*(const unsigned long long*)&_cvt_f32_26));
        packed[4] = __as_u32(__float22bfloat162_rn(make_float2(_mul_12.x, _mul_12.y)));
        float2 _cvt_f32_27 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 _cvt_f32_28 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float e0_13 = approx_exp2((-_cvt_f32_27.x) * 1.442695f);
        float e1_13 = approx_exp2((-_cvt_f32_27.y) * 1.442695f);
        float d0_13 = 1.0f + e0_13;
        float d1_13 = 1.0f + e1_13;
        float r0_13 = approx_rcp(d0_13);
        float r1_13 = approx_rcp(d1_13);
        float _fma_59;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(-d0_13), "f"(r0_13), "f"(1.0f));
        float _fma_10_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10_1) : "f"(r0_13), "f"(_fma_59), "f"(r0_13));
        float _fma_60;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(-d1_13), "f"(r1_13), "f"(1.0f));
        float _fma_11_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11_1) : "f"(r1_13), "f"(_fma_60), "f"(r1_13));
        float t0_13 = _fma_10_1;
        float t1_13 = _fma_11_1;
        band = band | ((fminf(t0_13, t1_13) == 0.0f) ? 32 : 0);
        float2 _f32x2_mul_a_61 = mul_f32x2_noftz(_cvt_f32_27, make_float2(max_noftz(t0_13, 0.0f), max_noftz(t1_13, 0.0f)));
        float2 _mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_61), "l"(*(const unsigned long long*)&_cvt_f32_28));
        packed[5] = __as_u32(__float22bfloat162_rn(make_float2(_mul_13.x, _mul_13.y)));
        float2 _cvt_f32_29 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 _cvt_f32_30 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float e0_14 = approx_exp2((-_cvt_f32_29.x) * 1.442695f);
        float e1_14 = approx_exp2((-_cvt_f32_29.y) * 1.442695f);
        float d0_14 = 1.0f + e0_14;
        float d1_14 = 1.0f + e1_14;
        float r0_14 = approx_rcp(d0_14);
        float r1_14 = approx_rcp(d1_14);
        float _fma_62;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(-d0_14), "f"(r0_14), "f"(1.0f));
        float _fma_12_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12_1) : "f"(r0_14), "f"(_fma_62), "f"(r0_14));
        float _fma_63;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(-d1_14), "f"(r1_14), "f"(1.0f));
        float _fma_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(r1_14), "f"(_fma_63), "f"(r1_14));
        float t0_14 = _fma_12_1;
        float t1_14 = _fma_13;
        band = band | ((fminf(t0_14, t1_14) == 0.0f) ? 64 : 0);
        float2 _f32x2_mul_a_64 = mul_f32x2_noftz(_cvt_f32_29, make_float2(max_noftz(t0_14, 0.0f), max_noftz(t1_14, 0.0f)));
        float2 _mul_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_64), "l"(*(const unsigned long long*)&_cvt_f32_30));
        packed[6] = __as_u32(__float22bfloat162_rn(make_float2(_mul_14.x, _mul_14.y)));
        float2 _cvt_f32_31 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 _cvt_f32_32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float e0_15 = approx_exp2((-_cvt_f32_31.x) * 1.442695f);
        float e1_15 = approx_exp2((-_cvt_f32_31.y) * 1.442695f);
        float d0_15 = 1.0f + e0_15;
        float d1_15 = 1.0f + e1_15;
        float r0_15 = approx_rcp(d0_15);
        float r1_15 = approx_rcp(d1_15);
        float _fma_65;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(-d0_15), "f"(r0_15), "f"(1.0f));
        float _fma_14_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14_1) : "f"(r0_15), "f"(_fma_65), "f"(r0_15));
        float _fma_66;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(-d1_15), "f"(r1_15), "f"(1.0f));
        float _fma_15_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15_1) : "f"(r1_15), "f"(_fma_66), "f"(r1_15));
        float t0_15 = _fma_14_1;
        float t1_15 = _fma_15_1;
        band = band | ((fminf(t0_15, t1_15) == 0.0f) ? 128 : 0);
        float2 _f32x2_mul_a_67 = mul_f32x2_noftz(_cvt_f32_31, make_float2(max_noftz(t0_15, 0.0f), max_noftz(t1_15, 0.0f)));
        float2 _mul_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_67), "l"(*(const unsigned long long*)&_cvt_f32_32));
        packed[7] = __as_u32(__float22bfloat162_rn(make_float2(_mul_15.x, _mul_15.y)));
        if (band != 0) {
            #pragma unroll 1
            for (int i = 0; i < 8; i++) {
                if ((band >> i & 1) != 0) {
                    unsigned int _vec_load_2[1];
                    {
                        _vec_load_2[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index + 2 * i) + 0);
                    }
                    unsigned int _vec_load_3[1];
                    {
                        _vec_load_3[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index + K + 2 * i) + 0);
                    }
                    float2 _cvt_f32_33 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
                    float2 _cvt_f32_34 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
                    float e0_16 = approx_exp2((-_cvt_f32_33.x) * 1.442695f);
                    float e1_16 = approx_exp2((-_cvt_f32_33.y) * 1.442695f);
                    float d0_16 = 1.0f + e0_16;
                    float d1_16 = 1.0f + e1_16;
                    float r0_16 = approx_rcp(d0_16);
                    float r1_16 = approx_rcp(d1_16);
                    float _fma_68;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(-d0_16), "f"(r0_16), "f"(1.0f));
                    float _fma_16_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16_1) : "f"(r0_16), "f"(_fma_68), "f"(r0_16));
                    float _fma_69;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(-d1_16), "f"(r1_16), "f"(1.0f));
                    float _fma_17_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17_1) : "f"(r1_16), "f"(_fma_69), "f"(r1_16));
                    float t0_16 = _fma_16_1;
                    float t1_16 = _fma_17_1;
                    float d4_16a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_16a) : "f"(d0_16), "f"(0.25f));
                    float r4_16a = approx_rcp(d4_16a);
                    float _fma_71;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(-d4_16a), "f"(r4_16a), "f"(1.0f));
                    float _fma_70;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(r4_16a), "f"(_fma_71), "f"(r4_16a));
                    float big_16a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_16a) : "f"(mul_f32_rn_noftz(_fma_70, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_72;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_72) : "f"(big_16a), "f"(8.388608e+06f));
                    float whole_16a;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_16a) : "f"(_add_72), "f"(8.388608e+06f));
                    float above_16a;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_16a) : "f"(-d4_16a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_16a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_74;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_74) : "f"(big_16a), "f"(whole_16a));
                    float _abs_73;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_73) : "f"(_sub_74));
                    float _add_75;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_75) : "f"(big_16a), "f"(0.5f));
                    float _sub_76;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_76) : "f"(big_16a), "f"(0.5f));
                    float rounded_16a = ((_abs_73 == 0.5f) ? ((above_16a > 0.0f) ? _add_75 : _sub_76) : whole_16a);
                    float u0_16 = ((d0_16 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_16a, 7.888609e-31f), 1.7763568e-15f) : t0_16);
                    float d4_16b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_16b) : "f"(d1_16), "f"(0.25f));
                    float r4_16b = approx_rcp(d4_16b);
                    float _fma_78;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_78) : "f"(-d4_16b), "f"(r4_16b), "f"(1.0f));
                    float _fma_77;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_77) : "f"(r4_16b), "f"(_fma_78), "f"(r4_16b));
                    float big_16b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_16b) : "f"(mul_f32_rn_noftz(_fma_77, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_79;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(big_16b), "f"(8.388608e+06f));
                    float whole_16b;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_16b) : "f"(_add_79), "f"(8.388608e+06f));
                    float above_16b;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_16b) : "f"(-d4_16b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_16b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_81;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_81) : "f"(big_16b), "f"(whole_16b));
                    float _abs_80;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_80) : "f"(_sub_81));
                    float _add_82;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_82) : "f"(big_16b), "f"(0.5f));
                    float _sub_83;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_83) : "f"(big_16b), "f"(0.5f));
                    float rounded_16b = ((_abs_80 == 0.5f) ? ((above_16b > 0.0f) ? _add_82 : _sub_83) : whole_16b);
                    float u1_16 = ((d1_16 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_16b, 7.888609e-31f), 1.7763568e-15f) : t1_16);
                    float2 _f32x2_mul_a_84 = mul_f32x2_noftz(_cvt_f32_33, make_float2(max_noftz(u0_16, 0.0f), max_noftz(u1_16, 0.0f)));
                    float2 _mul_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_84), "l"(*(const unsigned long long*)&_cvt_f32_34));
                    unsigned int pair = __as_u32(__float22bfloat162_rn(make_float2(_mul_16.x, _mul_16.y)));
                    packed[0] = ((i == 0) ? pair : packed[0]);
                    packed[1] = ((i == 1) ? pair : packed[1]);
                    packed[2] = ((i == 2) ? pair : packed[2]);
                    packed[3] = ((i == 3) ? pair : packed[3]);
                    packed[4] = ((i == 4) ? pair : packed[4]);
                    packed[5] = ((i == 5) ? pair : packed[5]);
                    packed[6] = ((i == 6) ? pair : packed[6]);
                    packed[7] = ((i == 7) ? pair : packed[7]);
                }
            }
        }
    }
    __nv_bfloat162 amax = __as_bf16x2(0);
    __nv_bfloat162 _hxa_86 = amax;
    __nv_bfloat162 _hxb_86 = __as_bf16x2(packed[0]);
    unsigned int _hxr_86;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_86) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_86)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_86)));
    __nv_bfloat162 _max_85 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_86);
    amax = _max_85;
    __nv_bfloat162 _hxa_88 = amax;
    __nv_bfloat162 _hxb_88 = __as_bf16x2(packed[1]);
    unsigned int _hxr_88;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_88) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_88)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_88)));
    __nv_bfloat162 _max_87 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_88);
    amax = _max_87;
    __nv_bfloat162 _hxa_90 = amax;
    __nv_bfloat162 _hxb_90 = __as_bf16x2(packed[2]);
    unsigned int _hxr_90;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_90) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_90)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_90)));
    __nv_bfloat162 _max_89 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_90);
    amax = _max_89;
    __nv_bfloat162 _hxa_92 = amax;
    __nv_bfloat162 _hxb_92 = __as_bf16x2(packed[3]);
    unsigned int _hxr_92;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_92) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_92)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_92)));
    __nv_bfloat162 _max_91 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_92);
    amax = _max_91;
    __nv_bfloat162 _hxa_94 = amax;
    __nv_bfloat162 _hxb_94 = __as_bf16x2(packed[4]);
    unsigned int _hxr_94;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_94) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_94)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_94)));
    __nv_bfloat162 _max_93 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_94);
    amax = _max_93;
    __nv_bfloat162 _hxa_96 = amax;
    __nv_bfloat162 _hxb_96 = __as_bf16x2(packed[5]);
    unsigned int _hxr_96;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_96) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_96)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_96)));
    __nv_bfloat162 _max_95 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_96);
    amax = _max_95;
    __nv_bfloat162 _hxa_98 = amax;
    __nv_bfloat162 _hxb_98 = __as_bf16x2(packed[6]);
    unsigned int _hxr_98;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_98) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_98)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_98)));
    __nv_bfloat162 _max_97 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_98);
    amax = _max_97;
    __nv_bfloat162 _hxa_100 = amax;
    __nv_bfloat162 _hxb_100 = __as_bf16x2(packed[7]);
    unsigned int _hxr_100;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_100) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_100)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_100)));
    __nv_bfloat162 _max_99 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_100);
    amax = _max_99;
    unsigned int mag = __as_u32(amax) & 2147450879;
    unsigned int peer = __shfl_xor_sync(0xFFFFFFFF, mag, 1);
    __nv_bfloat162 both = __hmax2_nan(__as_bf16x2(mag), __as_bf16x2(peer));
    unsigned int folded = __as_u32(__hmax2_nan(both, __as_bf16x2(__as_u32(both) >> 16)));
    unsigned int amax_bits = (folded & 65535) << 16;
    int scale_i32 = (int)(amax_bits + 2031616 >> 23) - 8;
    if (scale_i32 < 0) {
        scale_i32 = 0;
    }
    if ((amax_bits & 2139095040) == 2139095040) {
        scale_i32 = 255;
    }
    unsigned int scale = (unsigned int)scale_i32;
    unsigned int inv = 254 - scale << 7;
    if (scale == 255) {
        inv = 32704;
    }
    unsigned int inv2 = inv | inv << 16;
    int scale_col = bx * 4 + blk;
    int row_scale_index = ((grow >> 7) * (K >> 7) + (scale_col >> 2)) * 512 + (grow & 31) * 16 + (grow >> 5 & 3) * 4 + (scale_col & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + row_scale_index) + (0)) = (unsigned char)(scale);
    }
    unsigned int row_words[4];
    uint32_t _bf16x2_mul_0;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(packed[0]), "r"(inv2));
    uint16_t _e4m3x2_bf16x2_0;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_0) : "r"(_bf16x2_mul_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_0) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_1;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_1) : "r"(packed[1]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_102;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_102) : "r"(_bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_102) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_101;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_101) : "h"(_e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_102));
    row_words[0] = _mov_b32_pack_u16x2_101;
    uint32_t _bf16x2_mul_2;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_2) : "r"(packed[2]), "r"(inv2));
    uint16_t _e4m3x2_bf16x2_1;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_1) : "r"(_bf16x2_mul_2));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_2);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_1) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_3;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_3) : "r"(packed[3]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_104;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_104) : "r"(_bf16x2_mul_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_104) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_103;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_103) : "h"(_e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_104));
    row_words[1] = _mov_b32_pack_u16x2_103;
    uint32_t _bf16x2_mul_4;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_4) : "r"(packed[4]), "r"(inv2));
    uint16_t _e4m3x2_bf16x2_2;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_2) : "r"(_bf16x2_mul_4));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_4);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_2) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_5;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_5) : "r"(packed[5]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_106;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_106) : "r"(_bf16x2_mul_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_106) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_105;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_105) : "h"(_e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_106));
    row_words[2] = _mov_b32_pack_u16x2_105;
    uint32_t _bf16x2_mul_6;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_6) : "r"(packed[6]), "r"(inv2));
    uint16_t _e4m3x2_bf16x2_3;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_3) : "r"(_bf16x2_mul_6));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_6);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_3) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_7;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_7) : "r"(packed[7]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_108;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_108) : "r"(_bf16x2_mul_7));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_7);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_108) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_107;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_107) : "h"(_e4m3x2_bf16x2_3), "h"(_cvt_satfinite_e4m3x2_bf16x2_108));
    row_words[3] = _mov_b32_pack_u16x2_107;
    reinterpret_cast<int4*>(row_output + (grow * K + col))[0] = reinterpret_cast<int4*>(row_words + 0)[0];
}

} // extern "C"
