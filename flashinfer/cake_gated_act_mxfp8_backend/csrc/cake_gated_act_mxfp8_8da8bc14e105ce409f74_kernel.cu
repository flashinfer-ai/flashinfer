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
#define THREADS 128

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


__device__ __forceinline__ float2 sub_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("sub.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
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

__global__ __launch_bounds__(128) void
kernel_cake_gated_act_mxfp8_8da8bc14e105ce409f74(__nv_bfloat16* __restrict__ gated_input, __nv_bfloat16* __restrict__ grad, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int quarter = tid & 3;
    int row = tid >> 2;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int grow = by * 32 + row;
    int col = bx * 32 + quarter * 8;
    int gate_index = grow * (2 * K) + col;
    unsigned int _vec_load_0[4];
    {
        uint32_t _uv4_0_0;
        uint32_t _uv4_0_1;
        uint32_t _uv4_0_2;
        uint32_t _uv4_0_3;
        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
            : "=r"(_uv4_0_0), "=r"(_uv4_0_1), "=r"(_uv4_0_2), "=r"(_uv4_0_3) : "l"((const void*)(gated_input + gate_index + (0))), "l"(0x12F0000000000000ULL) : "memory");
        _vec_load_0[0 + 0] = _uv4_0_0;
        _vec_load_0[0 + 1] = _uv4_0_1;
        _vec_load_0[0 + 2] = _uv4_0_2;
        _vec_load_0[0 + 3] = _uv4_0_3;
    }
    unsigned int _vec_load_1[4];
    {
        uint32_t _uv4_1_0;
        uint32_t _uv4_1_1;
        uint32_t _uv4_1_2;
        uint32_t _uv4_1_3;
        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
            : "=r"(_uv4_1_0), "=r"(_uv4_1_1), "=r"(_uv4_1_2), "=r"(_uv4_1_3) : "l"((const void*)(gated_input + (gate_index + K) + (0))), "l"(0x12F0000000000000ULL) : "memory");
        _vec_load_1[0 + 0] = _uv4_1_0;
        _vec_load_1[0 + 1] = _uv4_1_1;
        _vec_load_1[0 + 2] = _uv4_1_2;
        _vec_load_1[0 + 3] = _uv4_1_3;
    }
    unsigned int _vec_load_2[4];
    {
        uint32_t _uv4_2_0;
        uint32_t _uv4_2_1;
        uint32_t _uv4_2_2;
        uint32_t _uv4_2_3;
        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
            : "=r"(_uv4_2_0), "=r"(_uv4_2_1), "=r"(_uv4_2_2), "=r"(_uv4_2_3) : "l"((const void*)(grad + (grow * K + col) + (0))), "l"(0x12F0000000000000ULL) : "memory");
        _vec_load_2[0 + 0] = _uv4_2_0;
        _vec_load_2[0 + 1] = _uv4_2_1;
        _vec_load_2[0 + 2] = _uv4_2_2;
        _vec_load_2[0 + 3] = _uv4_2_3;
    }
    unsigned int dgate[4];
    unsigned int dup[4];
    __nv_bfloat162 lowest = __as_bf16x2(_vec_load_0[0]);
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[3]));
    float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(lowest);
    float _min_4;
    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_4) : "f"(_cvt_f32_0.x), "f"(_cvt_f32_0.y));
    int _warp_vote_3 = __all_sync(0xFFFFFFFF, _min_4 > -87.0f);
    if (_warp_vote_3 != 0) {
        float2 _cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 _cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 _cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float e0_1 = approx_exp2((-_cvt_f32_1.x) * 1.442695f);
        float e1_1 = approx_exp2((-_cvt_f32_1.y) * 1.442695f);
        float d0_1 = 1.0f + e0_1;
        float d1_1 = 1.0f + e1_1;
        float r0_1 = approx_rcp(d0_1);
        float r1_1 = approx_rcp(d1_1);
        float _fma_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(-d0_1), "f"(r0_1), "f"(1.0f));
        float _fma_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(r0_1), "f"(_fma_6), "f"(r0_1));
        float _fma_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(-d1_1), "f"(r1_1), "f"(1.0f));
        float _fma_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(r1_1), "f"(_fma_8), "f"(r1_1));
        float2 s_0 = make_float2(_fma_5, _fma_7);
        float2 act_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_0) : "l"(*(const unsigned long long*)&_cvt_f32_1), "l"(*(const unsigned long long*)&s_0));
        float2 _f32x2_mul_a_9 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_0), s_0), _cvt_f32_3);
        float2 _mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_9), "l"(*(const unsigned long long*)&_cvt_f32_2));
        float2 _mul_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_1) : "l"(*(const unsigned long long*)&act_0), "l"(*(const unsigned long long*)&_cvt_f32_3));
        __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(_mul_0.x, _mul_0.y));
        __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(_mul_1.x, _mul_1.y));
        dgate[0] = __as_u32(_bf16x2_0);
        dup[0] = __as_u32(_bf16x2_1);
        float2 _cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 _cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 _cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float e0_3 = approx_exp2((-_cvt_f32_4.x) * 1.442695f);
        float e1_3 = approx_exp2((-_cvt_f32_4.y) * 1.442695f);
        float d0_3 = 1.0f + e0_3;
        float d1_3 = 1.0f + e1_3;
        float r0_3 = approx_rcp(d0_3);
        float r1_3 = approx_rcp(d1_3);
        float _fma_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(-d0_3), "f"(r0_3), "f"(1.0f));
        float _fma_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(r0_3), "f"(_fma_11), "f"(r0_3));
        float _fma_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(-d1_3), "f"(r1_3), "f"(1.0f));
        float _fma_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(r1_3), "f"(_fma_13), "f"(r1_3));
        float2 s_2 = make_float2(_fma_10, _fma_12);
        float2 act_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_2) : "l"(*(const unsigned long long*)&_cvt_f32_4), "l"(*(const unsigned long long*)&s_2));
        float2 _f32x2_mul_a_14 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_2), s_2), _cvt_f32_6);
        float2 _mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&_cvt_f32_5));
        float2 _mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_3) : "l"(*(const unsigned long long*)&act_2), "l"(*(const unsigned long long*)&_cvt_f32_6));
        __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(_mul_2.x, _mul_2.y));
        __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(_mul_3.x, _mul_3.y));
        dgate[1] = __as_u32(_bf16x2_2);
        dup[1] = __as_u32(_bf16x2_3);
        float2 _cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 _cvt_f32_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 _cvt_f32_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float e0_5 = approx_exp2((-_cvt_f32_7.x) * 1.442695f);
        float e1_5 = approx_exp2((-_cvt_f32_7.y) * 1.442695f);
        float d0_5 = 1.0f + e0_5;
        float d1_5 = 1.0f + e1_5;
        float r0_5 = approx_rcp(d0_5);
        float r1_5 = approx_rcp(d1_5);
        float _fma_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(-d0_5), "f"(r0_5), "f"(1.0f));
        float _fma_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(r0_5), "f"(_fma_16), "f"(r0_5));
        float _fma_18;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(-d1_5), "f"(r1_5), "f"(1.0f));
        float _fma_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(r1_5), "f"(_fma_18), "f"(r1_5));
        float2 s_4 = make_float2(_fma_15, _fma_17);
        float2 act_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_4) : "l"(*(const unsigned long long*)&_cvt_f32_7), "l"(*(const unsigned long long*)&s_4));
        float2 _f32x2_mul_a_19 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_4), s_4), _cvt_f32_9);
        float2 _mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&_cvt_f32_8));
        float2 _mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_5) : "l"(*(const unsigned long long*)&act_4), "l"(*(const unsigned long long*)&_cvt_f32_9));
        __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(_mul_4.x, _mul_4.y));
        __nv_bfloat162 _bf16x2_5 = __float22bfloat162_rn(make_float2(_mul_5.x, _mul_5.y));
        dgate[2] = __as_u32(_bf16x2_4);
        dup[2] = __as_u32(_bf16x2_5);
        float2 _cvt_f32_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 _cvt_f32_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 _cvt_f32_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float e0_7 = approx_exp2((-_cvt_f32_10.x) * 1.442695f);
        float e1_7 = approx_exp2((-_cvt_f32_10.y) * 1.442695f);
        float d0_7 = 1.0f + e0_7;
        float d1_7 = 1.0f + e1_7;
        float r0_7 = approx_rcp(d0_7);
        float r1_7 = approx_rcp(d1_7);
        float _fma_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(-d0_7), "f"(r0_7), "f"(1.0f));
        float _fma_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(r0_7), "f"(_fma_21), "f"(r0_7));
        float _fma_23;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(-d1_7), "f"(r1_7), "f"(1.0f));
        float _fma_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(r1_7), "f"(_fma_23), "f"(r1_7));
        float2 s_6 = make_float2(_fma_20, _fma_22);
        float2 act_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_6) : "l"(*(const unsigned long long*)&_cvt_f32_10), "l"(*(const unsigned long long*)&s_6));
        float2 _f32x2_mul_a_24 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_6), s_6), _cvt_f32_12);
        float2 _mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&_cvt_f32_11));
        float2 _mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_7) : "l"(*(const unsigned long long*)&act_6), "l"(*(const unsigned long long*)&_cvt_f32_12));
        __nv_bfloat162 _bf16x2_6 = __float22bfloat162_rn(make_float2(_mul_6.x, _mul_6.y));
        __nv_bfloat162 _bf16x2_7 = __float22bfloat162_rn(make_float2(_mul_7.x, _mul_7.y));
        dgate[3] = __as_u32(_bf16x2_6);
        dup[3] = __as_u32(_bf16x2_7);
    } else {
        int band = 0;
        float2 _cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 _cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 _cvt_f32_15 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float e0_9 = approx_exp2((-_cvt_f32_13.x) * 1.442695f);
        float e1_9 = approx_exp2((-_cvt_f32_13.y) * 1.442695f);
        float d0_9 = 1.0f + e0_9;
        float d1_9 = 1.0f + e1_9;
        float r0_9 = approx_rcp(d0_9);
        float r1_9 = approx_rcp(d1_9);
        float _fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(-d0_9), "f"(r0_9), "f"(1.0f));
        float _fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(r0_9), "f"(_fma_25), "f"(r0_9));
        float _fma_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(-d1_9), "f"(r1_9), "f"(1.0f));
        float _fma_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(r1_9), "f"(_fma_26), "f"(r1_9));
        float t0_9 = _fma_0;
        float t1_9 = _fma_1;
        band = band | ((fminf(t0_9, t1_9) == 0.0f) ? 1 : 0);
        float2 s_8 = make_float2(max_noftz(t0_9, 0.0f), max_noftz(t1_9, 0.0f));
        float2 act_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_8) : "l"(*(const unsigned long long*)&_cvt_f32_13), "l"(*(const unsigned long long*)&s_8));
        float2 _f32x2_mul_a_27 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_8, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_8), s_8), _cvt_f32_15);
        float2 _mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_27), "l"(*(const unsigned long long*)&_cvt_f32_14));
        float2 _mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_9) : "l"(*(const unsigned long long*)&act_8), "l"(*(const unsigned long long*)&_cvt_f32_15));
        __nv_bfloat162 _bf16x2_8 = __float22bfloat162_rn(make_float2(_mul_8.x, _mul_8.y));
        __nv_bfloat162 _bf16x2_9 = __float22bfloat162_rn(make_float2(_mul_9.x, _mul_9.y));
        dgate[0] = __as_u32(_bf16x2_8);
        dup[0] = __as_u32(_bf16x2_9);
        float2 _cvt_f32_16 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 _cvt_f32_17 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 _cvt_f32_18 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float e0_11 = approx_exp2((-_cvt_f32_16.x) * 1.442695f);
        float e1_11 = approx_exp2((-_cvt_f32_16.y) * 1.442695f);
        float d0_11 = 1.0f + e0_11;
        float d1_11 = 1.0f + e1_11;
        float r0_11 = approx_rcp(d0_11);
        float r1_11 = approx_rcp(d1_11);
        float _fma_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(-d0_11), "f"(r0_11), "f"(1.0f));
        float _fma_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(r0_11), "f"(_fma_28), "f"(r0_11));
        float _fma_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(-d1_11), "f"(r1_11), "f"(1.0f));
        float _fma_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(r1_11), "f"(_fma_29), "f"(r1_11));
        float t0_11 = _fma_2;
        float t1_11 = _fma_3;
        band = band | ((fminf(t0_11, t1_11) == 0.0f) ? 2 : 0);
        float2 s_10 = make_float2(max_noftz(t0_11, 0.0f), max_noftz(t1_11, 0.0f));
        float2 act_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_10) : "l"(*(const unsigned long long*)&_cvt_f32_16), "l"(*(const unsigned long long*)&s_10));
        float2 _f32x2_mul_a_30 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_10, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_10), s_10), _cvt_f32_18);
        float2 _mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&_cvt_f32_17));
        float2 _mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_11) : "l"(*(const unsigned long long*)&act_10), "l"(*(const unsigned long long*)&_cvt_f32_18));
        __nv_bfloat162 _bf16x2_10 = __float22bfloat162_rn(make_float2(_mul_10.x, _mul_10.y));
        __nv_bfloat162 _bf16x2_11 = __float22bfloat162_rn(make_float2(_mul_11.x, _mul_11.y));
        dgate[1] = __as_u32(_bf16x2_10);
        dup[1] = __as_u32(_bf16x2_11);
        float2 _cvt_f32_19 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 _cvt_f32_20 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 _cvt_f32_21 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float e0_13 = approx_exp2((-_cvt_f32_19.x) * 1.442695f);
        float e1_13 = approx_exp2((-_cvt_f32_19.y) * 1.442695f);
        float d0_13 = 1.0f + e0_13;
        float d1_13 = 1.0f + e1_13;
        float r0_13 = approx_rcp(d0_13);
        float r1_13 = approx_rcp(d1_13);
        float _fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(-d0_13), "f"(r0_13), "f"(1.0f));
        float _fma_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(r0_13), "f"(_fma_31), "f"(r0_13));
        float _fma_32;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(-d1_13), "f"(r1_13), "f"(1.0f));
        float _fma_5_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5_1) : "f"(r1_13), "f"(_fma_32), "f"(r1_13));
        float t0_13 = _fma_4;
        float t1_13 = _fma_5_1;
        band = band | ((fminf(t0_13, t1_13) == 0.0f) ? 4 : 0);
        float2 s_12 = make_float2(max_noftz(t0_13, 0.0f), max_noftz(t1_13, 0.0f));
        float2 act_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_12) : "l"(*(const unsigned long long*)&_cvt_f32_19), "l"(*(const unsigned long long*)&s_12));
        float2 _f32x2_mul_a_33 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_12, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_12), s_12), _cvt_f32_21);
        float2 _mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&_cvt_f32_20));
        float2 _mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_13) : "l"(*(const unsigned long long*)&act_12), "l"(*(const unsigned long long*)&_cvt_f32_21));
        __nv_bfloat162 _bf16x2_12 = __float22bfloat162_rn(make_float2(_mul_12.x, _mul_12.y));
        __nv_bfloat162 _bf16x2_13 = __float22bfloat162_rn(make_float2(_mul_13.x, _mul_13.y));
        dgate[2] = __as_u32(_bf16x2_12);
        dup[2] = __as_u32(_bf16x2_13);
        float2 _cvt_f32_22 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 _cvt_f32_23 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 _cvt_f32_24 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float e0_15 = approx_exp2((-_cvt_f32_22.x) * 1.442695f);
        float e1_15 = approx_exp2((-_cvt_f32_22.y) * 1.442695f);
        float d0_15 = 1.0f + e0_15;
        float d1_15 = 1.0f + e1_15;
        float r0_15 = approx_rcp(d0_15);
        float r1_15 = approx_rcp(d1_15);
        float _fma_34;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(-d0_15), "f"(r0_15), "f"(1.0f));
        float _fma_6_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6_1) : "f"(r0_15), "f"(_fma_34), "f"(r0_15));
        float _fma_35;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(-d1_15), "f"(r1_15), "f"(1.0f));
        float _fma_7_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7_1) : "f"(r1_15), "f"(_fma_35), "f"(r1_15));
        float t0_15 = _fma_6_1;
        float t1_15 = _fma_7_1;
        band = band | ((fminf(t0_15, t1_15) == 0.0f) ? 8 : 0);
        float2 s_14 = make_float2(max_noftz(t0_15, 0.0f), max_noftz(t1_15, 0.0f));
        float2 act_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_14) : "l"(*(const unsigned long long*)&_cvt_f32_22), "l"(*(const unsigned long long*)&s_14));
        float2 _f32x2_mul_a_36 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_14, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_14), s_14), _cvt_f32_24);
        float2 _mul_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_36), "l"(*(const unsigned long long*)&_cvt_f32_23));
        float2 _mul_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_15) : "l"(*(const unsigned long long*)&act_14), "l"(*(const unsigned long long*)&_cvt_f32_24));
        __nv_bfloat162 _bf16x2_14 = __float22bfloat162_rn(make_float2(_mul_14.x, _mul_14.y));
        __nv_bfloat162 _bf16x2_15 = __float22bfloat162_rn(make_float2(_mul_15.x, _mul_15.y));
        dgate[3] = __as_u32(_bf16x2_14);
        dup[3] = __as_u32(_bf16x2_15);
        if (band != 0) {
            #pragma unroll 1
            for (int i = 0; i < 4; i++) {
                if ((band >> i & 1) != 0) {
                    unsigned int _vec_load_3[1];
                    {
                        _vec_load_3[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index + 2 * i) + 0);
                    }
                    unsigned int _vec_load_4[1];
                    {
                        _vec_load_4[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index + K + 2 * i) + 0);
                    }
                    unsigned int _vec_load_5[1];
                    {
                        _vec_load_5[0] = *reinterpret_cast<const unsigned int*>(grad + (grow * K + col + 2 * i) + 0);
                    }
                    float2 _cvt_f32_25 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
                    float2 _cvt_f32_26 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[0]));
                    float2 _cvt_f32_27 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[0]));
                    float e0_17 = approx_exp2((-_cvt_f32_25.x) * 1.442695f);
                    float e1_17 = approx_exp2((-_cvt_f32_25.y) * 1.442695f);
                    float d0_17 = 1.0f + e0_17;
                    float d1_17 = 1.0f + e1_17;
                    float r0_17 = approx_rcp(d0_17);
                    float r1_17 = approx_rcp(d1_17);
                    float _fma_37;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(-d0_17), "f"(r0_17), "f"(1.0f));
                    float _fma_8_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8_1) : "f"(r0_17), "f"(_fma_37), "f"(r0_17));
                    float _fma_38;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(-d1_17), "f"(r1_17), "f"(1.0f));
                    float _fma_9;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(r1_17), "f"(_fma_38), "f"(r1_17));
                    float t0_17 = _fma_8_1;
                    float t1_17 = _fma_9;
                    float d4_17a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_17a) : "f"(d0_17), "f"(0.25f));
                    float r4_17a = approx_rcp(d4_17a);
                    float _fma_40;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(-d4_17a), "f"(r4_17a), "f"(1.0f));
                    float _fma_39;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(r4_17a), "f"(_fma_40), "f"(r4_17a));
                    float big_17a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_17a) : "f"(mul_f32_rn_noftz(_fma_39, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_41;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(big_17a), "f"(8.388608e+06f));
                    float whole_17a;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_17a) : "f"(_add_41), "f"(8.388608e+06f));
                    float above_17a;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_17a) : "f"(-d4_17a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_17a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_43;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43) : "f"(big_17a), "f"(whole_17a));
                    float _abs_42;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_42) : "f"(_sub_43));
                    float _add_44;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(big_17a), "f"(0.5f));
                    float _sub_45;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_45) : "f"(big_17a), "f"(0.5f));
                    float rounded_17a = ((_abs_42 == 0.5f) ? ((above_17a > 0.0f) ? _add_44 : _sub_45) : whole_17a);
                    float u0_17 = ((d0_17 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_17a, 7.888609e-31f), 1.7763568e-15f) : t0_17);
                    float d4_17b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_17b) : "f"(d1_17), "f"(0.25f));
                    float r4_17b = approx_rcp(d4_17b);
                    float _fma_47;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(-d4_17b), "f"(r4_17b), "f"(1.0f));
                    float _fma_46;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(r4_17b), "f"(_fma_47), "f"(r4_17b));
                    float big_17b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_17b) : "f"(mul_f32_rn_noftz(_fma_46, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_48;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(big_17b), "f"(8.388608e+06f));
                    float whole_17b;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_17b) : "f"(_add_48), "f"(8.388608e+06f));
                    float above_17b;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_17b) : "f"(-d4_17b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_17b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_50;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_50) : "f"(big_17b), "f"(whole_17b));
                    float _abs_49;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_49) : "f"(_sub_50));
                    float _add_51;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(big_17b), "f"(0.5f));
                    float _sub_52;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_52) : "f"(big_17b), "f"(0.5f));
                    float rounded_17b = ((_abs_49 == 0.5f) ? ((above_17b > 0.0f) ? _add_51 : _sub_52) : whole_17b);
                    float u1_17 = ((d1_17 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_17b, 7.888609e-31f), 1.7763568e-15f) : t1_17);
                    float2 s_16 = make_float2(max_noftz(u0_17, 0.0f), max_noftz(u1_17, 0.0f));
                    float2 act_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_16) : "l"(*(const unsigned long long*)&_cvt_f32_25), "l"(*(const unsigned long long*)&s_16));
                    float2 _f32x2_mul_a_53 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_16, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_16), s_16), _cvt_f32_27);
                    float2 _mul_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_53), "l"(*(const unsigned long long*)&_cvt_f32_26));
                    float2 _mul_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_17) : "l"(*(const unsigned long long*)&act_16), "l"(*(const unsigned long long*)&_cvt_f32_27));
                    __nv_bfloat162 _bf16x2_16 = __float22bfloat162_rn(make_float2(_mul_16.x, _mul_16.y));
                    __nv_bfloat162 _bf16x2_17 = __float22bfloat162_rn(make_float2(_mul_17.x, _mul_17.y));
                    unsigned int dgate_new = __as_u32(_bf16x2_16);
                    unsigned int dup_new = __as_u32(_bf16x2_17);
                    dgate[0] = ((i == 0) ? dgate_new : dgate[0]);
                    dup[0] = ((i == 0) ? dup_new : dup[0]);
                    dgate[1] = ((i == 1) ? dgate_new : dgate[1]);
                    dup[1] = ((i == 1) ? dup_new : dup[1]);
                    dgate[2] = ((i == 2) ? dgate_new : dgate[2]);
                    dup[2] = ((i == 2) ? dup_new : dup[2]);
                    dgate[3] = ((i == 3) ? dgate_new : dgate[3]);
                    dup[3] = ((i == 3) ? dup_new : dup[3]);
                }
            }
        }
    }
    int scale_cols = K >> 6;
    int out_index = grow * (2 * K) + col;
    __nv_bfloat162 amax = __as_bf16x2(0);
    __nv_bfloat162 _hxa_55 = amax;
    __nv_bfloat162 _hxb_55 = __as_bf16x2(dgate[0]);
    unsigned int _hxr_55;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_55) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_55)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_55)));
    __nv_bfloat162 _max_54 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_55);
    amax = _max_54;
    __nv_bfloat162 _hxa_57 = amax;
    __nv_bfloat162 _hxb_57 = __as_bf16x2(dgate[1]);
    unsigned int _hxr_57;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_57) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_57)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_57)));
    __nv_bfloat162 _max_56 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_57);
    amax = _max_56;
    __nv_bfloat162 _hxa_59 = amax;
    __nv_bfloat162 _hxb_59 = __as_bf16x2(dgate[2]);
    unsigned int _hxr_59;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_59) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_59)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_59)));
    __nv_bfloat162 _max_58 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_59);
    amax = _max_58;
    __nv_bfloat162 _hxa_61 = amax;
    __nv_bfloat162 _hxb_61 = __as_bf16x2(dgate[3]);
    unsigned int _hxr_61;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_61) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_61)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_61)));
    __nv_bfloat162 _max_60 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_61);
    amax = _max_60;
    unsigned int mag = __as_u32(amax) & 2147450879;
    unsigned int peer1 = __shfl_xor_sync(0xFFFFFFFF, mag, 1);
    __nv_bfloat162 pair_max = __hmax2_nan(__as_bf16x2(mag), __as_bf16x2(peer1));
    unsigned int peer2 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(pair_max), 2);
    __nv_bfloat162 quad_max = __hmax2_nan(pair_max, __as_bf16x2(peer2));
    unsigned int folded = __as_u32(__hmax2_nan(quad_max, __as_bf16x2(__as_u32(quad_max) >> 16)));
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
    int scale_index = ((grow >> 7) * scale_cols + (bx >> 2)) * 512 + (grow & 31) * 16 + (grow >> 5 & 3) * 4 + (bx & 3);
    if (quarter == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + scale_index) + (0)) = (unsigned char)(scale);
    }
    unsigned int words[2];
    uint32_t _bf16x2_mul_0;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(dgate[0]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_1) : "r"(dgate[1]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_63;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_63) : "r"(_bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_63) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_62;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_62) : "h"(_e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_63));
    words[0] = _mov_b32_pack_u16x2_62;
    uint32_t _bf16x2_mul_2;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_2) : "r"(dgate[2]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_3) : "r"(dgate[3]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_65;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_65) : "r"(_bf16x2_mul_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_65) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_64;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_64) : "h"(_e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_65));
    words[1] = _mov_b32_pack_u16x2_64;
    reinterpret_cast<int2*>(row_output + out_index)[0] = reinterpret_cast<int2*>(words + 0)[0];
    __nv_bfloat162 amax_0 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_67 = amax_0;
    __nv_bfloat162 _hxb_67 = __as_bf16x2(dup[0]);
    unsigned int _hxr_67;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_67) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_67)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_67)));
    __nv_bfloat162 _max_66 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_67);
    amax_0 = _max_66;
    __nv_bfloat162 _hxa_69 = amax_0;
    __nv_bfloat162 _hxb_69 = __as_bf16x2(dup[1]);
    unsigned int _hxr_69;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_69) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_69)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_69)));
    __nv_bfloat162 _max_68 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_69);
    amax_0 = _max_68;
    __nv_bfloat162 _hxa_71 = amax_0;
    __nv_bfloat162 _hxb_71 = __as_bf16x2(dup[2]);
    unsigned int _hxr_71;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_71) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_71)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_71)));
    __nv_bfloat162 _max_70 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_71);
    amax_0 = _max_70;
    __nv_bfloat162 _hxa_73 = amax_0;
    __nv_bfloat162 _hxb_73 = __as_bf16x2(dup[3]);
    unsigned int _hxr_73;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_73) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_73)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_73)));
    __nv_bfloat162 _max_72 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_73);
    amax_0 = _max_72;
    unsigned int mag_1 = __as_u32(amax_0) & 2147450879;
    unsigned int peer1_2 = __shfl_xor_sync(0xFFFFFFFF, mag_1, 1);
    __nv_bfloat162 pair_max_3 = __hmax2_nan(__as_bf16x2(mag_1), __as_bf16x2(peer1_2));
    unsigned int peer2_4 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(pair_max_3), 2);
    __nv_bfloat162 quad_max_5 = __hmax2_nan(pair_max_3, __as_bf16x2(peer2_4));
    unsigned int folded_6 = __as_u32(__hmax2_nan(quad_max_5, __as_bf16x2(__as_u32(quad_max_5) >> 16)));
    unsigned int amax_bits_7 = (folded_6 & 65535) << 16;
    int scale_i32_8 = (int)(amax_bits_7 + 2031616 >> 23) - 8;
    if (scale_i32_8 < 0) {
        scale_i32_8 = 0;
    }
    if ((amax_bits_7 & 2139095040) == 2139095040) {
        scale_i32_8 = 255;
    }
    unsigned int scale_9 = (unsigned int)scale_i32_8;
    unsigned int inv_10 = 254 - scale_9 << 7;
    if (scale_9 == 255) {
        inv_10 = 32704;
    }
    unsigned int inv2_11 = inv_10 | inv_10 << 16;
    int scale_index_12 = ((grow >> 7) * scale_cols + (bx + (K >> 5) >> 2)) * 512 + (grow & 31) * 16 + (grow >> 5 & 3) * 4 + (bx + (K >> 5) & 3);
    if (quarter == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + scale_index_12) + (0)) = (unsigned char)(scale_9);
    }
    unsigned int words_13[2];
    uint32_t _bf16x2_mul_4;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_4) : "r"(dup[0]), "r"(inv2_11));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_5) : "r"(dup[1]), "r"(inv2_11));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_75;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_75) : "r"(_bf16x2_mul_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_75) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_74;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_74) : "h"(_e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_75));
    words_13[0] = _mov_b32_pack_u16x2_74;
    uint32_t _bf16x2_mul_6;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_6) : "r"(dup[2]), "r"(inv2_11));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_7) : "r"(dup[3]), "r"(inv2_11));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_77;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_77) : "r"(_bf16x2_mul_7));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_7);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_77) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_76;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_76) : "h"(_e4m3x2_bf16x2_3), "h"(_cvt_satfinite_e4m3x2_bf16x2_77));
    words_13[1] = _mov_b32_pack_u16x2_76;
    reinterpret_cast<int2*>(row_output + (out_index + K))[0] = reinterpret_cast<int2*>(words_13 + 0)[0];
}

} // extern "C"
