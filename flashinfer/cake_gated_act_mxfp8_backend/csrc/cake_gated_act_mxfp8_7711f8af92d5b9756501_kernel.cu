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
#define LAUNCH_MIN_BLOCKS 4

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
kernel_cake_gated_act_mxfp8_7711f8af92d5b9756501(__nv_bfloat16* __restrict__ gated_input, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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
        asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
            : "=r"(_iv_0_0), "=r"(_iv_0_1), "=r"(_iv_0_2), "=r"(_iv_0_3), "=r"(_iv_0_4), "=r"(_iv_0_5), "=r"(_iv_0_6), "=r"(_iv_0_7) : "l"((const void*)(gated_input + gate_index + (0))), "l"(0x12F0000000000000ULL) : "memory");
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
        asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
            : "=r"(_iv_1_0), "=r"(_iv_1_1), "=r"(_iv_1_2), "=r"(_iv_1_3), "=r"(_iv_1_4), "=r"(_iv_1_5), "=r"(_iv_1_6), "=r"(_iv_1_7) : "l"((const void*)(gated_input + (gate_index + K) + (0))), "l"(0x12F0000000000000ULL) : "memory");
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
        float2 cvt_f32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float e0 = approx_exp2((-cvt_f32.x) * 1.442695f);
        float e1 = approx_exp2((-cvt_f32.y) * 1.442695f);
        float d0 = 1.0f + e0;
        float d1 = 1.0f + e1;
        float r0 = approx_rcp(d0);
        float r1 = approx_rcp(d1);
        float _fma_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(-d0), "f"(r0), "f"(1.0f));
        float _fma_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(r0), "f"(_fma_5), "f"(r0));
        float _fma_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(-d1), "f"(r1), "f"(1.0f));
        float _fma_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(r1), "f"(_fma_7), "f"(r1));
        float2 _f32x2_mul_a_8 = mul_f32x2_noftz(cvt_f32, make_float2(_fma_4, _fma_6));
        float2 mul;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul) : "l"(*(const unsigned long long*)&_f32x2_mul_a_8), "l"(*(const unsigned long long*)&cvt_f32_1));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(mul.x, mul.y)));
        float2 cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 cvt_f32_1_0 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float e0_0 = approx_exp2((-cvt_f32_0.x) * 1.442695f);
        float e1_0 = approx_exp2((-cvt_f32_0.y) * 1.442695f);
        float d0_0 = 1.0f + e0_0;
        float d1_0 = 1.0f + e1_0;
        float r0_0 = approx_rcp(d0_0);
        float r1_0 = approx_rcp(d1_0);
        float _fma_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(-d0_0), "f"(r0_0), "f"(1.0f));
        float _fma_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(r0_0), "f"(_fma_10), "f"(r0_0));
        float _fma_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(-d1_0), "f"(r1_0), "f"(1.0f));
        float _fma_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(r1_0), "f"(_fma_12), "f"(r1_0));
        float2 _f32x2_mul_a_13 = mul_f32x2_noftz(cvt_f32_0, make_float2(_fma_9, _fma_11));
        float2 mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_13), "l"(*(const unsigned long long*)&cvt_f32_1_0));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_0.x, mul_0.y)));
        float2 cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 cvt_f32_1_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float e0_1 = approx_exp2((-cvt_f32_2.x) * 1.442695f);
        float e1_1 = approx_exp2((-cvt_f32_2.y) * 1.442695f);
        float d0_1 = 1.0f + e0_1;
        float d1_1 = 1.0f + e1_1;
        float r0_1 = approx_rcp(d0_1);
        float r1_1 = approx_rcp(d1_1);
        float _fma_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(-d0_1), "f"(r0_1), "f"(1.0f));
        float _fma_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(r0_1), "f"(_fma_15), "f"(r0_1));
        float _fma_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(-d1_1), "f"(r1_1), "f"(1.0f));
        float _fma_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(r1_1), "f"(_fma_17), "f"(r1_1));
        float2 _f32x2_mul_a_18 = mul_f32x2_noftz(cvt_f32_2, make_float2(_fma_14, _fma_16));
        float2 mul_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&cvt_f32_1_1));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_1.x, mul_1.y)));
        float2 cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 cvt_f32_1_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float e0_2 = approx_exp2((-cvt_f32_3.x) * 1.442695f);
        float e1_2 = approx_exp2((-cvt_f32_3.y) * 1.442695f);
        float d0_2 = 1.0f + e0_2;
        float d1_2 = 1.0f + e1_2;
        float r0_2 = approx_rcp(d0_2);
        float r1_2 = approx_rcp(d1_2);
        float _fma_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(-d0_2), "f"(r0_2), "f"(1.0f));
        float _fma_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(r0_2), "f"(_fma_20), "f"(r0_2));
        float _fma_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(-d1_2), "f"(r1_2), "f"(1.0f));
        float _fma_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(r1_2), "f"(_fma_22), "f"(r1_2));
        float2 _f32x2_mul_a_23 = mul_f32x2_noftz(cvt_f32_3, make_float2(_fma_19, _fma_21));
        float2 mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&cvt_f32_1_2));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_2.x, mul_2.y)));
        float2 cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 cvt_f32_1_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float e0_3 = approx_exp2((-cvt_f32_4.x) * 1.442695f);
        float e1_3 = approx_exp2((-cvt_f32_4.y) * 1.442695f);
        float d0_3 = 1.0f + e0_3;
        float d1_3 = 1.0f + e1_3;
        float r0_3 = approx_rcp(d0_3);
        float r1_3 = approx_rcp(d1_3);
        float _fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(-d0_3), "f"(r0_3), "f"(1.0f));
        float _fma_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(r0_3), "f"(_fma_25), "f"(r0_3));
        float _fma_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(-d1_3), "f"(r1_3), "f"(1.0f));
        float _fma_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(r1_3), "f"(_fma_27), "f"(r1_3));
        float2 _f32x2_mul_a_28 = mul_f32x2_noftz(cvt_f32_4, make_float2(_fma_24, _fma_26));
        float2 mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&cvt_f32_1_3));
        packed[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_3.x, mul_3.y)));
        float2 cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 cvt_f32_1_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float e0_4 = approx_exp2((-cvt_f32_5.x) * 1.442695f);
        float e1_4 = approx_exp2((-cvt_f32_5.y) * 1.442695f);
        float d0_4 = 1.0f + e0_4;
        float d1_4 = 1.0f + e1_4;
        float r0_4 = approx_rcp(d0_4);
        float r1_4 = approx_rcp(d1_4);
        float _fma_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(-d0_4), "f"(r0_4), "f"(1.0f));
        float _fma_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(r0_4), "f"(_fma_30), "f"(r0_4));
        float _fma_32;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(-d1_4), "f"(r1_4), "f"(1.0f));
        float _fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(r1_4), "f"(_fma_32), "f"(r1_4));
        float2 _f32x2_mul_a_33 = mul_f32x2_noftz(cvt_f32_5, make_float2(_fma_29, _fma_31));
        float2 mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&cvt_f32_1_4));
        packed[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_4.x, mul_4.y)));
        float2 cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 cvt_f32_1_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float e0_5 = approx_exp2((-cvt_f32_6.x) * 1.442695f);
        float e1_5 = approx_exp2((-cvt_f32_6.y) * 1.442695f);
        float d0_5 = 1.0f + e0_5;
        float d1_5 = 1.0f + e1_5;
        float r0_5 = approx_rcp(d0_5);
        float r1_5 = approx_rcp(d1_5);
        float _fma_35;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(-d0_5), "f"(r0_5), "f"(1.0f));
        float _fma_34;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(r0_5), "f"(_fma_35), "f"(r0_5));
        float _fma_37;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(-d1_5), "f"(r1_5), "f"(1.0f));
        float _fma_36;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(r1_5), "f"(_fma_37), "f"(r1_5));
        float2 _f32x2_mul_a_38 = mul_f32x2_noftz(cvt_f32_6, make_float2(_fma_34, _fma_36));
        float2 mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_38), "l"(*(const unsigned long long*)&cvt_f32_1_5));
        packed[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_5.x, mul_5.y)));
        float2 cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 cvt_f32_1_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float e0_6 = approx_exp2((-cvt_f32_7.x) * 1.442695f);
        float e1_6 = approx_exp2((-cvt_f32_7.y) * 1.442695f);
        float d0_6 = 1.0f + e0_6;
        float d1_6 = 1.0f + e1_6;
        float r0_6 = approx_rcp(d0_6);
        float r1_6 = approx_rcp(d1_6);
        float _fma_40;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(-d0_6), "f"(r0_6), "f"(1.0f));
        float _fma_39;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(r0_6), "f"(_fma_40), "f"(r0_6));
        float _fma_42;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(-d1_6), "f"(r1_6), "f"(1.0f));
        float _fma_41;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(r1_6), "f"(_fma_42), "f"(r1_6));
        float2 _f32x2_mul_a_43 = mul_f32x2_noftz(cvt_f32_7, make_float2(_fma_39, _fma_41));
        float2 mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_43), "l"(*(const unsigned long long*)&cvt_f32_1_6));
        packed[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_6.x, mul_6.y)));
    } else {
        int band = 0;
        float2 cvt_f32_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 cvt_f32_1_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float e0_7 = approx_exp2((-cvt_f32_8.x) * 1.442695f);
        float e1_7 = approx_exp2((-cvt_f32_8.y) * 1.442695f);
        float d0_7 = 1.0f + e0_7;
        float d1_7 = 1.0f + e1_7;
        float r0_7 = approx_rcp(d0_7);
        float r1_7 = approx_rcp(d1_7);
        float _fma_44;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(-d0_7), "f"(r0_7), "f"(1.0f));
        float fma;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma) : "f"(r0_7), "f"(_fma_44), "f"(r0_7));
        float _fma_45;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(-d1_7), "f"(r1_7), "f"(1.0f));
        float fma_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1) : "f"(r1_7), "f"(_fma_45), "f"(r1_7));
        float t0 = fma;
        float t1 = fma_1;
        int _warp_vote_46 = __any_sync(0xFFFFFFFF, fminf(t0, t1) == 0.0f);
        if (_warp_vote_46 != 0) {
            float d4_8a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a) : "f"(d0_7), "f"(0.25f));
            float r4_8a = approx_rcp(d4_8a);
            float _fma_48;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(-d4_8a), "f"(r4_8a), "f"(1.0f));
            float _fma_47;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(r4_8a), "f"(_fma_48), "f"(r4_8a));
            float big_8a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a) : "f"(mul_f32_rn_noftz(_fma_47, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_49;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(big_8a), "f"(8.388608e+06f));
            float whole_8a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a) : "f"(_add_49), "f"(8.388608e+06f));
            float above_8a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a) : "f"(-d4_8a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_51;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_51) : "f"(big_8a), "f"(whole_8a));
            float _abs_50;
            asm("abs.f32 %0, %1;" : "=f"(_abs_50) : "f"(_sub_51));
            float _add_52;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(big_8a), "f"(0.5f));
            float _sub_53;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_53) : "f"(big_8a), "f"(0.5f));
            float rounded_8a = ((_abs_50 == 0.5f) ? ((above_8a > 0.0f) ? _add_52 : _sub_53) : whole_8a);
            t0 = ((d0_7 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a, 7.888609e-31f), 1.7763568e-15f) : t0);
            float d4_8a_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_0) : "f"(d1_7), "f"(0.25f));
            float r4_8a_0 = approx_rcp(d4_8a_0);
            float _fma_55;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(-d4_8a_0), "f"(r4_8a_0), "f"(1.0f));
            float _fma_54;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(r4_8a_0), "f"(_fma_55), "f"(r4_8a_0));
            float big_8a_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_0) : "f"(mul_f32_rn_noftz(_fma_54, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_56;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_56) : "f"(big_8a_0), "f"(8.388608e+06f));
            float whole_8a_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_0) : "f"(_add_56), "f"(8.388608e+06f));
            float above_8a_0;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_0) : "f"(-d4_8a_0), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_0, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_58;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_58) : "f"(big_8a_0), "f"(whole_8a_0));
            float _abs_57;
            asm("abs.f32 %0, %1;" : "=f"(_abs_57) : "f"(_sub_58));
            float _add_59;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(big_8a_0), "f"(0.5f));
            float _sub_60;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_60) : "f"(big_8a_0), "f"(0.5f));
            float rounded_8a_0 = ((_abs_57 == 0.5f) ? ((above_8a_0 > 0.0f) ? _add_59 : _sub_60) : whole_8a_0);
            t1 = ((d1_7 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_0, 7.888609e-31f), 1.7763568e-15f) : t1);
        }
        float2 _f32x2_mul_a_61 = mul_f32x2_noftz(cvt_f32_8, make_float2(max_noftz(t0, 0.0f), max_noftz(t1, 0.0f)));
        float2 mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_61), "l"(*(const unsigned long long*)&cvt_f32_1_7));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_7.x, mul_7.y)));
        float2 cvt_f32_8_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 cvt_f32_1_7_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float e0_7_1 = approx_exp2((-cvt_f32_8_1.x) * 1.442695f);
        float e1_7_1 = approx_exp2((-cvt_f32_8_1.y) * 1.442695f);
        float d0_7_1 = 1.0f + e0_7_1;
        float d1_7_1 = 1.0f + e1_7_1;
        float r0_7_1 = approx_rcp(d0_7_1);
        float r1_7_1 = approx_rcp(d1_7_1);
        float _fma_62;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(-d0_7_1), "f"(r0_7_1), "f"(1.0f));
        float fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_0) : "f"(r0_7_1), "f"(_fma_62), "f"(r0_7_1));
        float _fma_63;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(-d1_7_1), "f"(r1_7_1), "f"(1.0f));
        float fma_1_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_0) : "f"(r1_7_1), "f"(_fma_63), "f"(r1_7_1));
        float t0_0 = fma_0;
        float t1_0 = fma_1_0;
        int _warp_vote_64 = __any_sync(0xFFFFFFFF, fminf(t0_0, t1_0) == 0.0f);
        if (_warp_vote_64 != 0) {
            float d4_8a_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_1) : "f"(d0_7_1), "f"(0.25f));
            float r4_8a_1 = approx_rcp(d4_8a_1);
            float _fma_66;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(-d4_8a_1), "f"(r4_8a_1), "f"(1.0f));
            float _fma_65;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(r4_8a_1), "f"(_fma_66), "f"(r4_8a_1));
            float big_8a_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_1) : "f"(mul_f32_rn_noftz(_fma_65, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_67;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_67) : "f"(big_8a_1), "f"(8.388608e+06f));
            float whole_8a_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_1) : "f"(_add_67), "f"(8.388608e+06f));
            float above_8a_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_1) : "f"(-d4_8a_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_69;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_69) : "f"(big_8a_1), "f"(whole_8a_1));
            float _abs_68;
            asm("abs.f32 %0, %1;" : "=f"(_abs_68) : "f"(_sub_69));
            float _add_70;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_70) : "f"(big_8a_1), "f"(0.5f));
            float _sub_71;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_71) : "f"(big_8a_1), "f"(0.5f));
            float rounded_8a_1 = ((_abs_68 == 0.5f) ? ((above_8a_1 > 0.0f) ? _add_70 : _sub_71) : whole_8a_1);
            t0_0 = ((d0_7_1 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_1, 7.888609e-31f), 1.7763568e-15f) : t0_0);
            float d4_8a_1_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_1_1) : "f"(d1_7_1), "f"(0.25f));
            float r4_8a_1_1 = approx_rcp(d4_8a_1_1);
            float _fma_73;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(-d4_8a_1_1), "f"(r4_8a_1_1), "f"(1.0f));
            float _fma_72;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(r4_8a_1_1), "f"(_fma_73), "f"(r4_8a_1_1));
            float big_8a_1_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_1_1) : "f"(mul_f32_rn_noftz(_fma_72, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_74;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_74) : "f"(big_8a_1_1), "f"(8.388608e+06f));
            float whole_8a_1_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_1_1) : "f"(_add_74), "f"(8.388608e+06f));
            float above_8a_1_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_1_1) : "f"(-d4_8a_1_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_1_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_76;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_76) : "f"(big_8a_1_1), "f"(whole_8a_1_1));
            float _abs_75;
            asm("abs.f32 %0, %1;" : "=f"(_abs_75) : "f"(_sub_76));
            float _add_77;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(big_8a_1_1), "f"(0.5f));
            float _sub_78;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_78) : "f"(big_8a_1_1), "f"(0.5f));
            float rounded_8a_1_1 = ((_abs_75 == 0.5f) ? ((above_8a_1_1 > 0.0f) ? _add_77 : _sub_78) : whole_8a_1_1);
            t1_0 = ((d1_7_1 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_1_1, 7.888609e-31f), 1.7763568e-15f) : t1_0);
        }
        float2 _f32x2_mul_a_79 = mul_f32x2_noftz(cvt_f32_8_1, make_float2(max_noftz(t0_0, 0.0f), max_noftz(t1_0, 0.0f)));
        float2 mul_7_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_7_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_79), "l"(*(const unsigned long long*)&cvt_f32_1_7_1));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_7_1.x, mul_7_1.y)));
        float2 cvt_f32_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 cvt_f32_1_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float e0_8 = approx_exp2((-cvt_f32_9.x) * 1.442695f);
        float e1_8 = approx_exp2((-cvt_f32_9.y) * 1.442695f);
        float d0_8 = 1.0f + e0_8;
        float d1_8 = 1.0f + e1_8;
        float r0_8 = approx_rcp(d0_8);
        float r1_8 = approx_rcp(d1_8);
        float _fma_80;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(-d0_8), "f"(r0_8), "f"(1.0f));
        float fma_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_2) : "f"(r0_8), "f"(_fma_80), "f"(r0_8));
        float _fma_81;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(-d1_8), "f"(r1_8), "f"(1.0f));
        float fma_1_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_1) : "f"(r1_8), "f"(_fma_81), "f"(r1_8));
        float t0_1 = fma_2;
        float t1_1 = fma_1_1;
        int _warp_vote_82 = __any_sync(0xFFFFFFFF, fminf(t0_1, t1_1) == 0.0f);
        if (_warp_vote_82 != 0) {
            float d4_8a_2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_2) : "f"(d0_8), "f"(0.25f));
            float r4_8a_2 = approx_rcp(d4_8a_2);
            float _fma_84;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(-d4_8a_2), "f"(r4_8a_2), "f"(1.0f));
            float _fma_83;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(r4_8a_2), "f"(_fma_84), "f"(r4_8a_2));
            float big_8a_2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_2) : "f"(mul_f32_rn_noftz(_fma_83, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_85;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_85) : "f"(big_8a_2), "f"(8.388608e+06f));
            float whole_8a_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_2) : "f"(_add_85), "f"(8.388608e+06f));
            float above_8a_2;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_2) : "f"(-d4_8a_2), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_2, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_87;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_87) : "f"(big_8a_2), "f"(whole_8a_2));
            float _abs_86;
            asm("abs.f32 %0, %1;" : "=f"(_abs_86) : "f"(_sub_87));
            float _add_88;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_88) : "f"(big_8a_2), "f"(0.5f));
            float _sub_89;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_89) : "f"(big_8a_2), "f"(0.5f));
            float rounded_8a_2 = ((_abs_86 == 0.5f) ? ((above_8a_2 > 0.0f) ? _add_88 : _sub_89) : whole_8a_2);
            t0_1 = ((d0_8 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_2, 7.888609e-31f), 1.7763568e-15f) : t0_1);
            float d4_8a_2_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_2_1) : "f"(d1_8), "f"(0.25f));
            float r4_8a_2_1 = approx_rcp(d4_8a_2_1);
            float _fma_91;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(-d4_8a_2_1), "f"(r4_8a_2_1), "f"(1.0f));
            float _fma_90;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(r4_8a_2_1), "f"(_fma_91), "f"(r4_8a_2_1));
            float big_8a_2_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_2_1) : "f"(mul_f32_rn_noftz(_fma_90, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_92;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(big_8a_2_1), "f"(8.388608e+06f));
            float whole_8a_2_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_2_1) : "f"(_add_92), "f"(8.388608e+06f));
            float above_8a_2_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_2_1) : "f"(-d4_8a_2_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_2_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_94;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_94) : "f"(big_8a_2_1), "f"(whole_8a_2_1));
            float _abs_93;
            asm("abs.f32 %0, %1;" : "=f"(_abs_93) : "f"(_sub_94));
            float _add_95;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_95) : "f"(big_8a_2_1), "f"(0.5f));
            float _sub_96;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_96) : "f"(big_8a_2_1), "f"(0.5f));
            float rounded_8a_2_1 = ((_abs_93 == 0.5f) ? ((above_8a_2_1 > 0.0f) ? _add_95 : _sub_96) : whole_8a_2_1);
            t1_1 = ((d1_8 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_2_1, 7.888609e-31f), 1.7763568e-15f) : t1_1);
        }
        float2 _f32x2_mul_a_97 = mul_f32x2_noftz(cvt_f32_9, make_float2(max_noftz(t0_1, 0.0f), max_noftz(t1_1, 0.0f)));
        float2 mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_97), "l"(*(const unsigned long long*)&cvt_f32_1_8));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_8.x, mul_8.y)));
        float2 cvt_f32_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 cvt_f32_1_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float e0_9 = approx_exp2((-cvt_f32_10.x) * 1.442695f);
        float e1_9 = approx_exp2((-cvt_f32_10.y) * 1.442695f);
        float d0_9 = 1.0f + e0_9;
        float d1_9 = 1.0f + e1_9;
        float r0_9 = approx_rcp(d0_9);
        float r1_9 = approx_rcp(d1_9);
        float _fma_98;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(-d0_9), "f"(r0_9), "f"(1.0f));
        float fma_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_3) : "f"(r0_9), "f"(_fma_98), "f"(r0_9));
        float _fma_99;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(-d1_9), "f"(r1_9), "f"(1.0f));
        float fma_1_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_2) : "f"(r1_9), "f"(_fma_99), "f"(r1_9));
        float t0_2 = fma_3;
        float t1_2 = fma_1_2;
        int _warp_vote_100 = __any_sync(0xFFFFFFFF, fminf(t0_2, t1_2) == 0.0f);
        if (_warp_vote_100 != 0) {
            float d4_8a_3;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_3) : "f"(d0_9), "f"(0.25f));
            float r4_8a_3 = approx_rcp(d4_8a_3);
            float _fma_102;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_102) : "f"(-d4_8a_3), "f"(r4_8a_3), "f"(1.0f));
            float _fma_101;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(r4_8a_3), "f"(_fma_102), "f"(r4_8a_3));
            float big_8a_3;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_3) : "f"(mul_f32_rn_noftz(_fma_101, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_103;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_103) : "f"(big_8a_3), "f"(8.388608e+06f));
            float whole_8a_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_3) : "f"(_add_103), "f"(8.388608e+06f));
            float above_8a_3;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_3) : "f"(-d4_8a_3), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_3, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_105;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_105) : "f"(big_8a_3), "f"(whole_8a_3));
            float _abs_104;
            asm("abs.f32 %0, %1;" : "=f"(_abs_104) : "f"(_sub_105));
            float _add_106;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_106) : "f"(big_8a_3), "f"(0.5f));
            float _sub_107;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_107) : "f"(big_8a_3), "f"(0.5f));
            float rounded_8a_3 = ((_abs_104 == 0.5f) ? ((above_8a_3 > 0.0f) ? _add_106 : _sub_107) : whole_8a_3);
            t0_2 = ((d0_9 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_3, 7.888609e-31f), 1.7763568e-15f) : t0_2);
            float d4_8a_3_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_3_1) : "f"(d1_9), "f"(0.25f));
            float r4_8a_3_1 = approx_rcp(d4_8a_3_1);
            float _fma_109;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_109) : "f"(-d4_8a_3_1), "f"(r4_8a_3_1), "f"(1.0f));
            float _fma_108;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_108) : "f"(r4_8a_3_1), "f"(_fma_109), "f"(r4_8a_3_1));
            float big_8a_3_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_3_1) : "f"(mul_f32_rn_noftz(_fma_108, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_110;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_110) : "f"(big_8a_3_1), "f"(8.388608e+06f));
            float whole_8a_3_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_3_1) : "f"(_add_110), "f"(8.388608e+06f));
            float above_8a_3_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_3_1) : "f"(-d4_8a_3_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_3_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_112;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_112) : "f"(big_8a_3_1), "f"(whole_8a_3_1));
            float _abs_111;
            asm("abs.f32 %0, %1;" : "=f"(_abs_111) : "f"(_sub_112));
            float _add_113;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_113) : "f"(big_8a_3_1), "f"(0.5f));
            float _sub_114;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_114) : "f"(big_8a_3_1), "f"(0.5f));
            float rounded_8a_3_1 = ((_abs_111 == 0.5f) ? ((above_8a_3_1 > 0.0f) ? _add_113 : _sub_114) : whole_8a_3_1);
            t1_2 = ((d1_9 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_3_1, 7.888609e-31f), 1.7763568e-15f) : t1_2);
        }
        float2 _f32x2_mul_a_115 = mul_f32x2_noftz(cvt_f32_10, make_float2(max_noftz(t0_2, 0.0f), max_noftz(t1_2, 0.0f)));
        float2 mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_115), "l"(*(const unsigned long long*)&cvt_f32_1_9));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_9.x, mul_9.y)));
        float2 cvt_f32_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 cvt_f32_1_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float e0_10 = approx_exp2((-cvt_f32_11.x) * 1.442695f);
        float e1_10 = approx_exp2((-cvt_f32_11.y) * 1.442695f);
        float d0_10 = 1.0f + e0_10;
        float d1_10 = 1.0f + e1_10;
        float r0_10 = approx_rcp(d0_10);
        float r1_10 = approx_rcp(d1_10);
        float _fma_116;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_116) : "f"(-d0_10), "f"(r0_10), "f"(1.0f));
        float fma_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_4) : "f"(r0_10), "f"(_fma_116), "f"(r0_10));
        float _fma_117;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_117) : "f"(-d1_10), "f"(r1_10), "f"(1.0f));
        float fma_1_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_3) : "f"(r1_10), "f"(_fma_117), "f"(r1_10));
        float t0_3 = fma_4;
        float t1_3 = fma_1_3;
        int _warp_vote_118 = __any_sync(0xFFFFFFFF, fminf(t0_3, t1_3) == 0.0f);
        if (_warp_vote_118 != 0) {
            float d4_8a_4;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_4) : "f"(d0_10), "f"(0.25f));
            float r4_8a_4 = approx_rcp(d4_8a_4);
            float _fma_120;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_120) : "f"(-d4_8a_4), "f"(r4_8a_4), "f"(1.0f));
            float _fma_119;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_119) : "f"(r4_8a_4), "f"(_fma_120), "f"(r4_8a_4));
            float big_8a_4;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_4) : "f"(mul_f32_rn_noftz(_fma_119, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_121;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_121) : "f"(big_8a_4), "f"(8.388608e+06f));
            float whole_8a_4;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_4) : "f"(_add_121), "f"(8.388608e+06f));
            float above_8a_4;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_4) : "f"(-d4_8a_4), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_4, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_123;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_123) : "f"(big_8a_4), "f"(whole_8a_4));
            float _abs_122;
            asm("abs.f32 %0, %1;" : "=f"(_abs_122) : "f"(_sub_123));
            float _add_124;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_124) : "f"(big_8a_4), "f"(0.5f));
            float _sub_125;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_125) : "f"(big_8a_4), "f"(0.5f));
            float rounded_8a_4 = ((_abs_122 == 0.5f) ? ((above_8a_4 > 0.0f) ? _add_124 : _sub_125) : whole_8a_4);
            t0_3 = ((d0_10 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_4, 7.888609e-31f), 1.7763568e-15f) : t0_3);
            float d4_8a_4_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_4_1) : "f"(d1_10), "f"(0.25f));
            float r4_8a_4_1 = approx_rcp(d4_8a_4_1);
            float _fma_127;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_127) : "f"(-d4_8a_4_1), "f"(r4_8a_4_1), "f"(1.0f));
            float _fma_126;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_126) : "f"(r4_8a_4_1), "f"(_fma_127), "f"(r4_8a_4_1));
            float big_8a_4_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_4_1) : "f"(mul_f32_rn_noftz(_fma_126, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_128;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_128) : "f"(big_8a_4_1), "f"(8.388608e+06f));
            float whole_8a_4_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_4_1) : "f"(_add_128), "f"(8.388608e+06f));
            float above_8a_4_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_4_1) : "f"(-d4_8a_4_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_4_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_130;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_130) : "f"(big_8a_4_1), "f"(whole_8a_4_1));
            float _abs_129;
            asm("abs.f32 %0, %1;" : "=f"(_abs_129) : "f"(_sub_130));
            float _add_131;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_131) : "f"(big_8a_4_1), "f"(0.5f));
            float _sub_132;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_132) : "f"(big_8a_4_1), "f"(0.5f));
            float rounded_8a_4_1 = ((_abs_129 == 0.5f) ? ((above_8a_4_1 > 0.0f) ? _add_131 : _sub_132) : whole_8a_4_1);
            t1_3 = ((d1_10 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_4_1, 7.888609e-31f), 1.7763568e-15f) : t1_3);
        }
        float2 _f32x2_mul_a_133 = mul_f32x2_noftz(cvt_f32_11, make_float2(max_noftz(t0_3, 0.0f), max_noftz(t1_3, 0.0f)));
        float2 mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_133), "l"(*(const unsigned long long*)&cvt_f32_1_10));
        packed[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_10.x, mul_10.y)));
        float2 cvt_f32_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 cvt_f32_1_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float e0_11 = approx_exp2((-cvt_f32_12.x) * 1.442695f);
        float e1_11 = approx_exp2((-cvt_f32_12.y) * 1.442695f);
        float d0_11 = 1.0f + e0_11;
        float d1_11 = 1.0f + e1_11;
        float r0_11 = approx_rcp(d0_11);
        float r1_11 = approx_rcp(d1_11);
        float _fma_134;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_134) : "f"(-d0_11), "f"(r0_11), "f"(1.0f));
        float fma_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_5) : "f"(r0_11), "f"(_fma_134), "f"(r0_11));
        float _fma_135;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_135) : "f"(-d1_11), "f"(r1_11), "f"(1.0f));
        float fma_1_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_4) : "f"(r1_11), "f"(_fma_135), "f"(r1_11));
        float t0_4 = fma_5;
        float t1_4 = fma_1_4;
        int _warp_vote_136 = __any_sync(0xFFFFFFFF, fminf(t0_4, t1_4) == 0.0f);
        if (_warp_vote_136 != 0) {
            float d4_8a_5;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_5) : "f"(d0_11), "f"(0.25f));
            float r4_8a_5 = approx_rcp(d4_8a_5);
            float _fma_138;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_138) : "f"(-d4_8a_5), "f"(r4_8a_5), "f"(1.0f));
            float _fma_137;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_137) : "f"(r4_8a_5), "f"(_fma_138), "f"(r4_8a_5));
            float big_8a_5;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_5) : "f"(mul_f32_rn_noftz(_fma_137, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_139;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_139) : "f"(big_8a_5), "f"(8.388608e+06f));
            float whole_8a_5;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_5) : "f"(_add_139), "f"(8.388608e+06f));
            float above_8a_5;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_5) : "f"(-d4_8a_5), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_5, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_141;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_141) : "f"(big_8a_5), "f"(whole_8a_5));
            float _abs_140;
            asm("abs.f32 %0, %1;" : "=f"(_abs_140) : "f"(_sub_141));
            float _add_142;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_142) : "f"(big_8a_5), "f"(0.5f));
            float _sub_143;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_143) : "f"(big_8a_5), "f"(0.5f));
            float rounded_8a_5 = ((_abs_140 == 0.5f) ? ((above_8a_5 > 0.0f) ? _add_142 : _sub_143) : whole_8a_5);
            t0_4 = ((d0_11 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_5, 7.888609e-31f), 1.7763568e-15f) : t0_4);
            float d4_8a_5_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_5_1) : "f"(d1_11), "f"(0.25f));
            float r4_8a_5_1 = approx_rcp(d4_8a_5_1);
            float _fma_145;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_145) : "f"(-d4_8a_5_1), "f"(r4_8a_5_1), "f"(1.0f));
            float _fma_144;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_144) : "f"(r4_8a_5_1), "f"(_fma_145), "f"(r4_8a_5_1));
            float big_8a_5_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_5_1) : "f"(mul_f32_rn_noftz(_fma_144, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_146;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_146) : "f"(big_8a_5_1), "f"(8.388608e+06f));
            float whole_8a_5_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_5_1) : "f"(_add_146), "f"(8.388608e+06f));
            float above_8a_5_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_5_1) : "f"(-d4_8a_5_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_5_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_148;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_148) : "f"(big_8a_5_1), "f"(whole_8a_5_1));
            float _abs_147;
            asm("abs.f32 %0, %1;" : "=f"(_abs_147) : "f"(_sub_148));
            float _add_149;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_149) : "f"(big_8a_5_1), "f"(0.5f));
            float _sub_150;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_150) : "f"(big_8a_5_1), "f"(0.5f));
            float rounded_8a_5_1 = ((_abs_147 == 0.5f) ? ((above_8a_5_1 > 0.0f) ? _add_149 : _sub_150) : whole_8a_5_1);
            t1_4 = ((d1_11 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_5_1, 7.888609e-31f), 1.7763568e-15f) : t1_4);
        }
        float2 _f32x2_mul_a_151 = mul_f32x2_noftz(cvt_f32_12, make_float2(max_noftz(t0_4, 0.0f), max_noftz(t1_4, 0.0f)));
        float2 mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_151), "l"(*(const unsigned long long*)&cvt_f32_1_11));
        packed[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_11.x, mul_11.y)));
        float2 cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 cvt_f32_1_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float e0_12 = approx_exp2((-cvt_f32_13.x) * 1.442695f);
        float e1_12 = approx_exp2((-cvt_f32_13.y) * 1.442695f);
        float d0_12 = 1.0f + e0_12;
        float d1_12 = 1.0f + e1_12;
        float r0_12 = approx_rcp(d0_12);
        float r1_12 = approx_rcp(d1_12);
        float _fma_152;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_152) : "f"(-d0_12), "f"(r0_12), "f"(1.0f));
        float fma_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_6) : "f"(r0_12), "f"(_fma_152), "f"(r0_12));
        float _fma_153;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_153) : "f"(-d1_12), "f"(r1_12), "f"(1.0f));
        float fma_1_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_5) : "f"(r1_12), "f"(_fma_153), "f"(r1_12));
        float t0_5 = fma_6;
        float t1_5 = fma_1_5;
        int _warp_vote_154 = __any_sync(0xFFFFFFFF, fminf(t0_5, t1_5) == 0.0f);
        if (_warp_vote_154 != 0) {
            float d4_8a_6;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_6) : "f"(d0_12), "f"(0.25f));
            float r4_8a_6 = approx_rcp(d4_8a_6);
            float _fma_156;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_156) : "f"(-d4_8a_6), "f"(r4_8a_6), "f"(1.0f));
            float _fma_155;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_155) : "f"(r4_8a_6), "f"(_fma_156), "f"(r4_8a_6));
            float big_8a_6;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_6) : "f"(mul_f32_rn_noftz(_fma_155, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_157;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_157) : "f"(big_8a_6), "f"(8.388608e+06f));
            float whole_8a_6;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_6) : "f"(_add_157), "f"(8.388608e+06f));
            float above_8a_6;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_6) : "f"(-d4_8a_6), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_6, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_159;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_159) : "f"(big_8a_6), "f"(whole_8a_6));
            float _abs_158;
            asm("abs.f32 %0, %1;" : "=f"(_abs_158) : "f"(_sub_159));
            float _add_160;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_160) : "f"(big_8a_6), "f"(0.5f));
            float _sub_161;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_161) : "f"(big_8a_6), "f"(0.5f));
            float rounded_8a_6 = ((_abs_158 == 0.5f) ? ((above_8a_6 > 0.0f) ? _add_160 : _sub_161) : whole_8a_6);
            t0_5 = ((d0_12 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_6, 7.888609e-31f), 1.7763568e-15f) : t0_5);
            float d4_8a_6_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_6_1) : "f"(d1_12), "f"(0.25f));
            float r4_8a_6_1 = approx_rcp(d4_8a_6_1);
            float _fma_163;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_163) : "f"(-d4_8a_6_1), "f"(r4_8a_6_1), "f"(1.0f));
            float _fma_162;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_162) : "f"(r4_8a_6_1), "f"(_fma_163), "f"(r4_8a_6_1));
            float big_8a_6_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_6_1) : "f"(mul_f32_rn_noftz(_fma_162, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_164;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_164) : "f"(big_8a_6_1), "f"(8.388608e+06f));
            float whole_8a_6_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_6_1) : "f"(_add_164), "f"(8.388608e+06f));
            float above_8a_6_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_6_1) : "f"(-d4_8a_6_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_6_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_166;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_166) : "f"(big_8a_6_1), "f"(whole_8a_6_1));
            float _abs_165;
            asm("abs.f32 %0, %1;" : "=f"(_abs_165) : "f"(_sub_166));
            float _add_167;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_167) : "f"(big_8a_6_1), "f"(0.5f));
            float _sub_168;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_168) : "f"(big_8a_6_1), "f"(0.5f));
            float rounded_8a_6_1 = ((_abs_165 == 0.5f) ? ((above_8a_6_1 > 0.0f) ? _add_167 : _sub_168) : whole_8a_6_1);
            t1_5 = ((d1_12 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_6_1, 7.888609e-31f), 1.7763568e-15f) : t1_5);
        }
        float2 _f32x2_mul_a_169 = mul_f32x2_noftz(cvt_f32_13, make_float2(max_noftz(t0_5, 0.0f), max_noftz(t1_5, 0.0f)));
        float2 mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_169), "l"(*(const unsigned long long*)&cvt_f32_1_12));
        packed[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_12.x, mul_12.y)));
        float2 cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 cvt_f32_1_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float e0_13 = approx_exp2((-cvt_f32_14.x) * 1.442695f);
        float e1_13 = approx_exp2((-cvt_f32_14.y) * 1.442695f);
        float d0_13 = 1.0f + e0_13;
        float d1_13 = 1.0f + e1_13;
        float r0_13 = approx_rcp(d0_13);
        float r1_13 = approx_rcp(d1_13);
        float _fma_170;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_170) : "f"(-d0_13), "f"(r0_13), "f"(1.0f));
        float fma_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_7) : "f"(r0_13), "f"(_fma_170), "f"(r0_13));
        float _fma_171;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_171) : "f"(-d1_13), "f"(r1_13), "f"(1.0f));
        float fma_1_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_6) : "f"(r1_13), "f"(_fma_171), "f"(r1_13));
        float t0_6 = fma_7;
        float t1_6 = fma_1_6;
        int _warp_vote_172 = __any_sync(0xFFFFFFFF, fminf(t0_6, t1_6) == 0.0f);
        if (_warp_vote_172 != 0) {
            float d4_8a_7;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_7) : "f"(d0_13), "f"(0.25f));
            float r4_8a_7 = approx_rcp(d4_8a_7);
            float _fma_174;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_174) : "f"(-d4_8a_7), "f"(r4_8a_7), "f"(1.0f));
            float _fma_173;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_173) : "f"(r4_8a_7), "f"(_fma_174), "f"(r4_8a_7));
            float big_8a_7;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_7) : "f"(mul_f32_rn_noftz(_fma_173, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_175;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_175) : "f"(big_8a_7), "f"(8.388608e+06f));
            float whole_8a_7;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_7) : "f"(_add_175), "f"(8.388608e+06f));
            float above_8a_7;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_7) : "f"(-d4_8a_7), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_7, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_177;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_177) : "f"(big_8a_7), "f"(whole_8a_7));
            float _abs_176;
            asm("abs.f32 %0, %1;" : "=f"(_abs_176) : "f"(_sub_177));
            float _add_178;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_178) : "f"(big_8a_7), "f"(0.5f));
            float _sub_179;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_179) : "f"(big_8a_7), "f"(0.5f));
            float rounded_8a_7 = ((_abs_176 == 0.5f) ? ((above_8a_7 > 0.0f) ? _add_178 : _sub_179) : whole_8a_7);
            t0_6 = ((d0_13 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_7, 7.888609e-31f), 1.7763568e-15f) : t0_6);
            float d4_8a_7_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_7_1) : "f"(d1_13), "f"(0.25f));
            float r4_8a_7_1 = approx_rcp(d4_8a_7_1);
            float _fma_181;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_181) : "f"(-d4_8a_7_1), "f"(r4_8a_7_1), "f"(1.0f));
            float _fma_180;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_180) : "f"(r4_8a_7_1), "f"(_fma_181), "f"(r4_8a_7_1));
            float big_8a_7_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_7_1) : "f"(mul_f32_rn_noftz(_fma_180, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_182;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_182) : "f"(big_8a_7_1), "f"(8.388608e+06f));
            float whole_8a_7_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_7_1) : "f"(_add_182), "f"(8.388608e+06f));
            float above_8a_7_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_7_1) : "f"(-d4_8a_7_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_7_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_184;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_184) : "f"(big_8a_7_1), "f"(whole_8a_7_1));
            float _abs_183;
            asm("abs.f32 %0, %1;" : "=f"(_abs_183) : "f"(_sub_184));
            float _add_185;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_185) : "f"(big_8a_7_1), "f"(0.5f));
            float _sub_186;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_186) : "f"(big_8a_7_1), "f"(0.5f));
            float rounded_8a_7_1 = ((_abs_183 == 0.5f) ? ((above_8a_7_1 > 0.0f) ? _add_185 : _sub_186) : whole_8a_7_1);
            t1_6 = ((d1_13 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_7_1, 7.888609e-31f), 1.7763568e-15f) : t1_6);
        }
        float2 _f32x2_mul_a_187 = mul_f32x2_noftz(cvt_f32_14, make_float2(max_noftz(t0_6, 0.0f), max_noftz(t1_6, 0.0f)));
        float2 mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_187), "l"(*(const unsigned long long*)&cvt_f32_1_13));
        packed[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_13.x, mul_13.y)));
    }
    __nv_bfloat162 amax = __as_bf16x2(0);
    __nv_bfloat162 _hxa_189 = amax;
    __nv_bfloat162 _hxb_189 = __as_bf16x2(packed[0]);
    unsigned int _hxr_189;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_189) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_189)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_189)));
    __nv_bfloat162 _max_188 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_189);
    amax = _max_188;
    __nv_bfloat162 _hxa_191 = amax;
    __nv_bfloat162 _hxb_191 = __as_bf16x2(packed[1]);
    unsigned int _hxr_191;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_191) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_191)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_191)));
    __nv_bfloat162 _max_190 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_191);
    amax = _max_190;
    __nv_bfloat162 _hxa_193 = amax;
    __nv_bfloat162 _hxb_193 = __as_bf16x2(packed[2]);
    unsigned int _hxr_193;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_193) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_193)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_193)));
    __nv_bfloat162 _max_192 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_193);
    amax = _max_192;
    __nv_bfloat162 _hxa_195 = amax;
    __nv_bfloat162 _hxb_195 = __as_bf16x2(packed[3]);
    unsigned int _hxr_195;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_195) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_195)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_195)));
    __nv_bfloat162 _max_194 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_195);
    amax = _max_194;
    __nv_bfloat162 _hxa_197 = amax;
    __nv_bfloat162 _hxb_197 = __as_bf16x2(packed[4]);
    unsigned int _hxr_197;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_197) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_197)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_197)));
    __nv_bfloat162 _max_196 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_197);
    amax = _max_196;
    __nv_bfloat162 _hxa_199 = amax;
    __nv_bfloat162 _hxb_199 = __as_bf16x2(packed[5]);
    unsigned int _hxr_199;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_199) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_199)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_199)));
    __nv_bfloat162 _max_198 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_199);
    amax = _max_198;
    __nv_bfloat162 _hxa_201 = amax;
    __nv_bfloat162 _hxb_201 = __as_bf16x2(packed[6]);
    unsigned int _hxr_201;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_201) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_201)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_201)));
    __nv_bfloat162 _max_200 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_201);
    amax = _max_200;
    __nv_bfloat162 _hxa_203 = amax;
    __nv_bfloat162 _hxb_203 = __as_bf16x2(packed[7]);
    unsigned int _hxr_203;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_203) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_203)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_203)));
    __nv_bfloat162 _max_202 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_203);
    amax = _max_202;
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
    uint32_t bf16x2_mul;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul) : "r"(packed[0]), "r"(inv2));
    uint16_t e4m3x2_bf16x2;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2) : "r"(bf16x2_mul));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1) : "r"(packed[1]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_205;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_205) : "r"(bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_205) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_204;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_204) : "h"(e4m3x2_bf16x2), "h"(_cvt_satfinite_e4m3x2_bf16x2_205));
    row_words[0] = _mov_b32_pack_u16x2_204;
    uint32_t bf16x2_mul_0;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_0) : "r"(packed[2]), "r"(inv2));
    uint16_t e4m3x2_bf16x2_0;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_0) : "r"(bf16x2_mul_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_0) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_0;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_0) : "r"(packed[3]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_207;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_207) : "r"(bf16x2_mul_1_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_207) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_206;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_206) : "h"(e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_207));
    row_words[1] = _mov_b32_pack_u16x2_206;
    uint32_t bf16x2_mul_2;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2) : "r"(packed[4]), "r"(inv2));
    uint16_t e4m3x2_bf16x2_1;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_1) : "r"(bf16x2_mul_2));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_2);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_1) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_1;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_1) : "r"(packed[5]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_209;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_209) : "r"(bf16x2_mul_1_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_209) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_208;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_208) : "h"(e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_209));
    row_words[2] = _mov_b32_pack_u16x2_208;
    uint32_t bf16x2_mul_3;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3) : "r"(packed[6]), "r"(inv2));
    uint16_t e4m3x2_bf16x2_2;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_2) : "r"(bf16x2_mul_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_2) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_2;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_2) : "r"(packed[7]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_211;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_211) : "r"(bf16x2_mul_1_2));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_2);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_211) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_210;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_210) : "h"(e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_211));
    row_words[3] = _mov_b32_pack_u16x2_210;
    reinterpret_cast<int4*>(row_output + (grow * K + col))[0] = reinterpret_cast<int4*>(row_words + 0)[0];
}

} // extern "C"
