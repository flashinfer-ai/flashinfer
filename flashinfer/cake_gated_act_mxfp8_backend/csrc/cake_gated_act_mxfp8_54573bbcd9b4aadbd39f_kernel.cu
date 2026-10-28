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
#define LAUNCH_MIN_BLOCKS 12

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

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_gated_act_mxfp8_54573bbcd9b4aadbd39f(__nv_bfloat16* __restrict__ gated_input, __nv_bfloat16* __restrict__ grad, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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
        float2 cvt_f32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float e0 = approx_exp2((-cvt_f32.x) * 1.442695f);
        float e1 = approx_exp2((-cvt_f32.y) * 1.442695f);
        float d0 = 1.0f + e0;
        float d1 = 1.0f + e1;
        float r0 = approx_rcp(d0);
        float r1 = approx_rcp(d1);
        float _fma_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(-d0), "f"(r0), "f"(1.0f));
        float _fma_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(r0), "f"(_fma_6), "f"(r0));
        float _fma_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(-d1), "f"(r1), "f"(1.0f));
        float _fma_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(r1), "f"(_fma_8), "f"(r1));
        float2 s = make_float2(_fma_5, _fma_7);
        float2 act;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act) : "l"(*(const unsigned long long*)&cvt_f32), "l"(*(const unsigned long long*)&s));
        float2 _f32x2_mul_a_9 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s), s), cvt_f32_2);
        float2 mul;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul) : "l"(*(const unsigned long long*)&_f32x2_mul_a_9), "l"(*(const unsigned long long*)&cvt_f32_1));
        float2 mul_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1) : "l"(*(const unsigned long long*)&act), "l"(*(const unsigned long long*)&cvt_f32_2));
        __nv_bfloat162 bf16x2 = __float22bfloat162_rn(make_float2(mul.x, mul.y));
        __nv_bfloat162 bf16x2_1 = __float22bfloat162_rn(make_float2(mul_1.x, mul_1.y));
        dgate[0] = __as_u32(bf16x2);
        dup[0] = __as_u32(bf16x2_1);
        float2 cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 cvt_f32_1_0 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 cvt_f32_2_0 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float e0_0 = approx_exp2((-cvt_f32_0.x) * 1.442695f);
        float e1_0 = approx_exp2((-cvt_f32_0.y) * 1.442695f);
        float d0_0 = 1.0f + e0_0;
        float d1_0 = 1.0f + e1_0;
        float r0_0 = approx_rcp(d0_0);
        float r1_0 = approx_rcp(d1_0);
        float _fma_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(-d0_0), "f"(r0_0), "f"(1.0f));
        float _fma_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(r0_0), "f"(_fma_11), "f"(r0_0));
        float _fma_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(-d1_0), "f"(r1_0), "f"(1.0f));
        float _fma_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(r1_0), "f"(_fma_13), "f"(r1_0));
        float2 s_0 = make_float2(_fma_10, _fma_12);
        float2 act_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_0) : "l"(*(const unsigned long long*)&cvt_f32_0), "l"(*(const unsigned long long*)&s_0));
        float2 _f32x2_mul_a_14 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_0), s_0), cvt_f32_2_0);
        float2 mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&cvt_f32_1_0));
        float2 mul_1_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_0) : "l"(*(const unsigned long long*)&act_0), "l"(*(const unsigned long long*)&cvt_f32_2_0));
        __nv_bfloat162 bf16x2_0 = __float22bfloat162_rn(make_float2(mul_0.x, mul_0.y));
        __nv_bfloat162 bf16x2_1_0 = __float22bfloat162_rn(make_float2(mul_1_0.x, mul_1_0.y));
        dgate[1] = __as_u32(bf16x2_0);
        dup[1] = __as_u32(bf16x2_1_0);
        float2 cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 cvt_f32_1_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 cvt_f32_2_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float e0_1 = approx_exp2((-cvt_f32_3.x) * 1.442695f);
        float e1_1 = approx_exp2((-cvt_f32_3.y) * 1.442695f);
        float d0_1 = 1.0f + e0_1;
        float d1_1 = 1.0f + e1_1;
        float r0_1 = approx_rcp(d0_1);
        float r1_1 = approx_rcp(d1_1);
        float _fma_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(-d0_1), "f"(r0_1), "f"(1.0f));
        float _fma_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(r0_1), "f"(_fma_16), "f"(r0_1));
        float _fma_18;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(-d1_1), "f"(r1_1), "f"(1.0f));
        float _fma_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(r1_1), "f"(_fma_18), "f"(r1_1));
        float2 s_1 = make_float2(_fma_15, _fma_17);
        float2 act_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_1) : "l"(*(const unsigned long long*)&cvt_f32_3), "l"(*(const unsigned long long*)&s_1));
        float2 _f32x2_mul_a_19 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1), s_1), cvt_f32_2_1);
        float2 mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&cvt_f32_1_1));
        float2 mul_1_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_1) : "l"(*(const unsigned long long*)&act_1), "l"(*(const unsigned long long*)&cvt_f32_2_1));
        __nv_bfloat162 bf16x2_2 = __float22bfloat162_rn(make_float2(mul_2.x, mul_2.y));
        __nv_bfloat162 bf16x2_1_1 = __float22bfloat162_rn(make_float2(mul_1_1.x, mul_1_1.y));
        dgate[2] = __as_u32(bf16x2_2);
        dup[2] = __as_u32(bf16x2_1_1);
        float2 cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 cvt_f32_1_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 cvt_f32_2_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float e0_2 = approx_exp2((-cvt_f32_4.x) * 1.442695f);
        float e1_2 = approx_exp2((-cvt_f32_4.y) * 1.442695f);
        float d0_2 = 1.0f + e0_2;
        float d1_2 = 1.0f + e1_2;
        float r0_2 = approx_rcp(d0_2);
        float r1_2 = approx_rcp(d1_2);
        float _fma_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(-d0_2), "f"(r0_2), "f"(1.0f));
        float _fma_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(r0_2), "f"(_fma_21), "f"(r0_2));
        float _fma_23;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(-d1_2), "f"(r1_2), "f"(1.0f));
        float _fma_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(r1_2), "f"(_fma_23), "f"(r1_2));
        float2 s_2 = make_float2(_fma_20, _fma_22);
        float2 act_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_2) : "l"(*(const unsigned long long*)&cvt_f32_4), "l"(*(const unsigned long long*)&s_2));
        float2 _f32x2_mul_a_24 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_2), s_2), cvt_f32_2_2);
        float2 mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&cvt_f32_1_2));
        float2 mul_1_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_2) : "l"(*(const unsigned long long*)&act_2), "l"(*(const unsigned long long*)&cvt_f32_2_2));
        __nv_bfloat162 bf16x2_3 = __float22bfloat162_rn(make_float2(mul_3.x, mul_3.y));
        __nv_bfloat162 bf16x2_1_2 = __float22bfloat162_rn(make_float2(mul_1_2.x, mul_1_2.y));
        dgate[3] = __as_u32(bf16x2_3);
        dup[3] = __as_u32(bf16x2_1_2);
    } else {
        int band = 0;
        float2 cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 cvt_f32_1_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 cvt_f32_2_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float e0_3 = approx_exp2((-cvt_f32_5.x) * 1.442695f);
        float e1_3 = approx_exp2((-cvt_f32_5.y) * 1.442695f);
        float d0_3 = 1.0f + e0_3;
        float d1_3 = 1.0f + e1_3;
        float r0_3 = approx_rcp(d0_3);
        float r1_3 = approx_rcp(d1_3);
        float _fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(-d0_3), "f"(r0_3), "f"(1.0f));
        float fma;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma) : "f"(r0_3), "f"(_fma_25), "f"(r0_3));
        float _fma_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(-d1_3), "f"(r1_3), "f"(1.0f));
        float fma_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1) : "f"(r1_3), "f"(_fma_26), "f"(r1_3));
        float t0 = fma;
        float t1 = fma_1;
        int _warp_vote_27 = __any_sync(0xFFFFFFFF, fminf(t0, t1) == 0.0f);
        if (_warp_vote_27 != 0) {
            float d4_9a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_9a) : "f"(d0_3), "f"(0.25f));
            float r4_9a = approx_rcp(d4_9a);
            float _fma_29;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(-d4_9a), "f"(r4_9a), "f"(1.0f));
            float _fma_28;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(r4_9a), "f"(_fma_29), "f"(r4_9a));
            float big_9a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_9a) : "f"(mul_f32_rn_noftz(_fma_28, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_30;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(big_9a), "f"(8.388608e+06f));
            float whole_9a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_9a) : "f"(_add_30), "f"(8.388608e+06f));
            float above_9a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_9a) : "f"(-d4_9a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_9a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_32;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32) : "f"(big_9a), "f"(whole_9a));
            float _abs_31;
            asm("abs.f32 %0, %1;" : "=f"(_abs_31) : "f"(_sub_32));
            float _add_33;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(big_9a), "f"(0.5f));
            float _sub_34;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34) : "f"(big_9a), "f"(0.5f));
            float rounded_9a = ((_abs_31 == 0.5f) ? ((above_9a > 0.0f) ? _add_33 : _sub_34) : whole_9a);
            t0 = ((d0_3 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_9a, 7.888609e-31f), 1.7763568e-15f) : t0);
            float d4_9a_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_9a_0) : "f"(d1_3), "f"(0.25f));
            float r4_9a_0 = approx_rcp(d4_9a_0);
            float _fma_36;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(-d4_9a_0), "f"(r4_9a_0), "f"(1.0f));
            float _fma_35;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(r4_9a_0), "f"(_fma_36), "f"(r4_9a_0));
            float big_9a_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_9a_0) : "f"(mul_f32_rn_noftz(_fma_35, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_37;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(big_9a_0), "f"(8.388608e+06f));
            float whole_9a_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_9a_0) : "f"(_add_37), "f"(8.388608e+06f));
            float above_9a_0;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_9a_0) : "f"(-d4_9a_0), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_9a_0, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_39;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_39) : "f"(big_9a_0), "f"(whole_9a_0));
            float _abs_38;
            asm("abs.f32 %0, %1;" : "=f"(_abs_38) : "f"(_sub_39));
            float _add_40;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(big_9a_0), "f"(0.5f));
            float _sub_41;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41) : "f"(big_9a_0), "f"(0.5f));
            float rounded_9a_0 = ((_abs_38 == 0.5f) ? ((above_9a_0 > 0.0f) ? _add_40 : _sub_41) : whole_9a_0);
            t1 = ((d1_3 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_9a_0, 7.888609e-31f), 1.7763568e-15f) : t1);
        }
        float2 s_3 = make_float2(max_noftz(t0, 0.0f), max_noftz(t1, 0.0f));
        float2 act_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_3) : "l"(*(const unsigned long long*)&cvt_f32_5), "l"(*(const unsigned long long*)&s_3));
        float2 _f32x2_mul_a_42 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_3), s_3), cvt_f32_2_3);
        float2 mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&cvt_f32_1_3));
        float2 mul_1_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_3) : "l"(*(const unsigned long long*)&act_3), "l"(*(const unsigned long long*)&cvt_f32_2_3));
        __nv_bfloat162 bf16x2_4 = __float22bfloat162_rn(make_float2(mul_4.x, mul_4.y));
        __nv_bfloat162 bf16x2_1_3 = __float22bfloat162_rn(make_float2(mul_1_3.x, mul_1_3.y));
        dgate[0] = __as_u32(bf16x2_4);
        dup[0] = __as_u32(bf16x2_1_3);
        float2 cvt_f32_5_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 cvt_f32_1_3_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 cvt_f32_2_3_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float e0_3_1 = approx_exp2((-cvt_f32_5_1.x) * 1.442695f);
        float e1_3_1 = approx_exp2((-cvt_f32_5_1.y) * 1.442695f);
        float d0_3_1 = 1.0f + e0_3_1;
        float d1_3_1 = 1.0f + e1_3_1;
        float r0_3_1 = approx_rcp(d0_3_1);
        float r1_3_1 = approx_rcp(d1_3_1);
        float _fma_43;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(-d0_3_1), "f"(r0_3_1), "f"(1.0f));
        float fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_0) : "f"(r0_3_1), "f"(_fma_43), "f"(r0_3_1));
        float _fma_44;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(-d1_3_1), "f"(r1_3_1), "f"(1.0f));
        float fma_1_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_0) : "f"(r1_3_1), "f"(_fma_44), "f"(r1_3_1));
        float t0_0 = fma_0;
        float t1_0 = fma_1_0;
        int _warp_vote_45 = __any_sync(0xFFFFFFFF, fminf(t0_0, t1_0) == 0.0f);
        if (_warp_vote_45 != 0) {
            float d4_9a_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_9a_1) : "f"(d0_3_1), "f"(0.25f));
            float r4_9a_1 = approx_rcp(d4_9a_1);
            float _fma_47;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(-d4_9a_1), "f"(r4_9a_1), "f"(1.0f));
            float _fma_46;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(r4_9a_1), "f"(_fma_47), "f"(r4_9a_1));
            float big_9a_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_9a_1) : "f"(mul_f32_rn_noftz(_fma_46, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_48;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(big_9a_1), "f"(8.388608e+06f));
            float whole_9a_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_9a_1) : "f"(_add_48), "f"(8.388608e+06f));
            float above_9a_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_9a_1) : "f"(-d4_9a_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_9a_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_50;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_50) : "f"(big_9a_1), "f"(whole_9a_1));
            float _abs_49;
            asm("abs.f32 %0, %1;" : "=f"(_abs_49) : "f"(_sub_50));
            float _add_51;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(big_9a_1), "f"(0.5f));
            float _sub_52;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_52) : "f"(big_9a_1), "f"(0.5f));
            float rounded_9a_1 = ((_abs_49 == 0.5f) ? ((above_9a_1 > 0.0f) ? _add_51 : _sub_52) : whole_9a_1);
            t0_0 = ((d0_3_1 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_9a_1, 7.888609e-31f), 1.7763568e-15f) : t0_0);
            float d4_9a_1_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_9a_1_1) : "f"(d1_3_1), "f"(0.25f));
            float r4_9a_1_1 = approx_rcp(d4_9a_1_1);
            float _fma_54;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(-d4_9a_1_1), "f"(r4_9a_1_1), "f"(1.0f));
            float _fma_53;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(r4_9a_1_1), "f"(_fma_54), "f"(r4_9a_1_1));
            float big_9a_1_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_9a_1_1) : "f"(mul_f32_rn_noftz(_fma_53, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_55;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_55) : "f"(big_9a_1_1), "f"(8.388608e+06f));
            float whole_9a_1_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_9a_1_1) : "f"(_add_55), "f"(8.388608e+06f));
            float above_9a_1_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_9a_1_1) : "f"(-d4_9a_1_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_9a_1_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_57;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_57) : "f"(big_9a_1_1), "f"(whole_9a_1_1));
            float _abs_56;
            asm("abs.f32 %0, %1;" : "=f"(_abs_56) : "f"(_sub_57));
            float _add_58;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_58) : "f"(big_9a_1_1), "f"(0.5f));
            float _sub_59;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_59) : "f"(big_9a_1_1), "f"(0.5f));
            float rounded_9a_1_1 = ((_abs_56 == 0.5f) ? ((above_9a_1_1 > 0.0f) ? _add_58 : _sub_59) : whole_9a_1_1);
            t1_0 = ((d1_3_1 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_9a_1_1, 7.888609e-31f), 1.7763568e-15f) : t1_0);
        }
        float2 s_3_1 = make_float2(max_noftz(t0_0, 0.0f), max_noftz(t1_0, 0.0f));
        float2 act_3_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_3_1) : "l"(*(const unsigned long long*)&cvt_f32_5_1), "l"(*(const unsigned long long*)&s_3_1));
        float2 _f32x2_mul_a_60 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_3_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_3_1), s_3_1), cvt_f32_2_3_1);
        float2 mul_4_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&cvt_f32_1_3_1));
        float2 mul_1_3_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_3_1) : "l"(*(const unsigned long long*)&act_3_1), "l"(*(const unsigned long long*)&cvt_f32_2_3_1));
        __nv_bfloat162 bf16x2_4_1 = __float22bfloat162_rn(make_float2(mul_4_1.x, mul_4_1.y));
        __nv_bfloat162 bf16x2_1_3_1 = __float22bfloat162_rn(make_float2(mul_1_3_1.x, mul_1_3_1.y));
        dgate[1] = __as_u32(bf16x2_4_1);
        dup[1] = __as_u32(bf16x2_1_3_1);
        float2 cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 cvt_f32_1_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 cvt_f32_2_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float e0_4 = approx_exp2((-cvt_f32_6.x) * 1.442695f);
        float e1_4 = approx_exp2((-cvt_f32_6.y) * 1.442695f);
        float d0_4 = 1.0f + e0_4;
        float d1_4 = 1.0f + e1_4;
        float r0_4 = approx_rcp(d0_4);
        float r1_4 = approx_rcp(d1_4);
        float _fma_61;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(-d0_4), "f"(r0_4), "f"(1.0f));
        float fma_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_2) : "f"(r0_4), "f"(_fma_61), "f"(r0_4));
        float _fma_62;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(-d1_4), "f"(r1_4), "f"(1.0f));
        float fma_1_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_1) : "f"(r1_4), "f"(_fma_62), "f"(r1_4));
        float t0_1 = fma_2;
        float t1_1 = fma_1_1;
        int _warp_vote_63 = __any_sync(0xFFFFFFFF, fminf(t0_1, t1_1) == 0.0f);
        if (_warp_vote_63 != 0) {
            float d4_9a_2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_9a_2) : "f"(d0_4), "f"(0.25f));
            float r4_9a_2 = approx_rcp(d4_9a_2);
            float _fma_65;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(-d4_9a_2), "f"(r4_9a_2), "f"(1.0f));
            float _fma_64;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(r4_9a_2), "f"(_fma_65), "f"(r4_9a_2));
            float big_9a_2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_9a_2) : "f"(mul_f32_rn_noftz(_fma_64, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_66;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_66) : "f"(big_9a_2), "f"(8.388608e+06f));
            float whole_9a_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_9a_2) : "f"(_add_66), "f"(8.388608e+06f));
            float above_9a_2;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_9a_2) : "f"(-d4_9a_2), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_9a_2, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_68;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_68) : "f"(big_9a_2), "f"(whole_9a_2));
            float _abs_67;
            asm("abs.f32 %0, %1;" : "=f"(_abs_67) : "f"(_sub_68));
            float _add_69;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(big_9a_2), "f"(0.5f));
            float _sub_70;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_70) : "f"(big_9a_2), "f"(0.5f));
            float rounded_9a_2 = ((_abs_67 == 0.5f) ? ((above_9a_2 > 0.0f) ? _add_69 : _sub_70) : whole_9a_2);
            t0_1 = ((d0_4 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_9a_2, 7.888609e-31f), 1.7763568e-15f) : t0_1);
            float d4_9a_2_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_9a_2_1) : "f"(d1_4), "f"(0.25f));
            float r4_9a_2_1 = approx_rcp(d4_9a_2_1);
            float _fma_72;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(-d4_9a_2_1), "f"(r4_9a_2_1), "f"(1.0f));
            float _fma_71;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(r4_9a_2_1), "f"(_fma_72), "f"(r4_9a_2_1));
            float big_9a_2_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_9a_2_1) : "f"(mul_f32_rn_noftz(_fma_71, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_73;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(big_9a_2_1), "f"(8.388608e+06f));
            float whole_9a_2_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_9a_2_1) : "f"(_add_73), "f"(8.388608e+06f));
            float above_9a_2_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_9a_2_1) : "f"(-d4_9a_2_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_9a_2_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_75;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_75) : "f"(big_9a_2_1), "f"(whole_9a_2_1));
            float _abs_74;
            asm("abs.f32 %0, %1;" : "=f"(_abs_74) : "f"(_sub_75));
            float _add_76;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(big_9a_2_1), "f"(0.5f));
            float _sub_77;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_77) : "f"(big_9a_2_1), "f"(0.5f));
            float rounded_9a_2_1 = ((_abs_74 == 0.5f) ? ((above_9a_2_1 > 0.0f) ? _add_76 : _sub_77) : whole_9a_2_1);
            t1_1 = ((d1_4 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_9a_2_1, 7.888609e-31f), 1.7763568e-15f) : t1_1);
        }
        float2 s_4 = make_float2(max_noftz(t0_1, 0.0f), max_noftz(t1_1, 0.0f));
        float2 act_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_4) : "l"(*(const unsigned long long*)&cvt_f32_6), "l"(*(const unsigned long long*)&s_4));
        float2 _f32x2_mul_a_78 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_4), s_4), cvt_f32_2_4);
        float2 mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_78), "l"(*(const unsigned long long*)&cvt_f32_1_4));
        float2 mul_1_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_4) : "l"(*(const unsigned long long*)&act_4), "l"(*(const unsigned long long*)&cvt_f32_2_4));
        __nv_bfloat162 bf16x2_5 = __float22bfloat162_rn(make_float2(mul_5.x, mul_5.y));
        __nv_bfloat162 bf16x2_1_4 = __float22bfloat162_rn(make_float2(mul_1_4.x, mul_1_4.y));
        dgate[2] = __as_u32(bf16x2_5);
        dup[2] = __as_u32(bf16x2_1_4);
        float2 cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 cvt_f32_1_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 cvt_f32_2_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float e0_5 = approx_exp2((-cvt_f32_7.x) * 1.442695f);
        float e1_5 = approx_exp2((-cvt_f32_7.y) * 1.442695f);
        float d0_5 = 1.0f + e0_5;
        float d1_5 = 1.0f + e1_5;
        float r0_5 = approx_rcp(d0_5);
        float r1_5 = approx_rcp(d1_5);
        float _fma_79;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(-d0_5), "f"(r0_5), "f"(1.0f));
        float fma_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_3) : "f"(r0_5), "f"(_fma_79), "f"(r0_5));
        float _fma_80;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(-d1_5), "f"(r1_5), "f"(1.0f));
        float fma_1_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_2) : "f"(r1_5), "f"(_fma_80), "f"(r1_5));
        float t0_2 = fma_3;
        float t1_2 = fma_1_2;
        int _warp_vote_81 = __any_sync(0xFFFFFFFF, fminf(t0_2, t1_2) == 0.0f);
        if (_warp_vote_81 != 0) {
            float d4_9a_3;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_9a_3) : "f"(d0_5), "f"(0.25f));
            float r4_9a_3 = approx_rcp(d4_9a_3);
            float _fma_83;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(-d4_9a_3), "f"(r4_9a_3), "f"(1.0f));
            float _fma_82;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(r4_9a_3), "f"(_fma_83), "f"(r4_9a_3));
            float big_9a_3;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_9a_3) : "f"(mul_f32_rn_noftz(_fma_82, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_84;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_84) : "f"(big_9a_3), "f"(8.388608e+06f));
            float whole_9a_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_9a_3) : "f"(_add_84), "f"(8.388608e+06f));
            float above_9a_3;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_9a_3) : "f"(-d4_9a_3), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_9a_3, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_86;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_86) : "f"(big_9a_3), "f"(whole_9a_3));
            float _abs_85;
            asm("abs.f32 %0, %1;" : "=f"(_abs_85) : "f"(_sub_86));
            float _add_87;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(big_9a_3), "f"(0.5f));
            float _sub_88;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_88) : "f"(big_9a_3), "f"(0.5f));
            float rounded_9a_3 = ((_abs_85 == 0.5f) ? ((above_9a_3 > 0.0f) ? _add_87 : _sub_88) : whole_9a_3);
            t0_2 = ((d0_5 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_9a_3, 7.888609e-31f), 1.7763568e-15f) : t0_2);
            float d4_9a_3_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_9a_3_1) : "f"(d1_5), "f"(0.25f));
            float r4_9a_3_1 = approx_rcp(d4_9a_3_1);
            float _fma_90;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(-d4_9a_3_1), "f"(r4_9a_3_1), "f"(1.0f));
            float _fma_89;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(r4_9a_3_1), "f"(_fma_90), "f"(r4_9a_3_1));
            float big_9a_3_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_9a_3_1) : "f"(mul_f32_rn_noftz(_fma_89, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_91;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_91) : "f"(big_9a_3_1), "f"(8.388608e+06f));
            float whole_9a_3_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_9a_3_1) : "f"(_add_91), "f"(8.388608e+06f));
            float above_9a_3_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_9a_3_1) : "f"(-d4_9a_3_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_9a_3_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_93;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_93) : "f"(big_9a_3_1), "f"(whole_9a_3_1));
            float _abs_92;
            asm("abs.f32 %0, %1;" : "=f"(_abs_92) : "f"(_sub_93));
            float _add_94;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_94) : "f"(big_9a_3_1), "f"(0.5f));
            float _sub_95;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_95) : "f"(big_9a_3_1), "f"(0.5f));
            float rounded_9a_3_1 = ((_abs_92 == 0.5f) ? ((above_9a_3_1 > 0.0f) ? _add_94 : _sub_95) : whole_9a_3_1);
            t1_2 = ((d1_5 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_9a_3_1, 7.888609e-31f), 1.7763568e-15f) : t1_2);
        }
        float2 s_5 = make_float2(max_noftz(t0_2, 0.0f), max_noftz(t1_2, 0.0f));
        float2 act_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_5) : "l"(*(const unsigned long long*)&cvt_f32_7), "l"(*(const unsigned long long*)&s_5));
        float2 _f32x2_mul_a_96 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_5), s_5), cvt_f32_2_5);
        float2 mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_96), "l"(*(const unsigned long long*)&cvt_f32_1_5));
        float2 mul_1_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_5) : "l"(*(const unsigned long long*)&act_5), "l"(*(const unsigned long long*)&cvt_f32_2_5));
        __nv_bfloat162 bf16x2_6 = __float22bfloat162_rn(make_float2(mul_6.x, mul_6.y));
        __nv_bfloat162 bf16x2_1_5 = __float22bfloat162_rn(make_float2(mul_1_5.x, mul_1_5.y));
        dgate[3] = __as_u32(bf16x2_6);
        dup[3] = __as_u32(bf16x2_1_5);
    }
    int scale_cols = K >> 6;
    int out_index = grow * (2 * K) + col;
    __nv_bfloat162 amax = __as_bf16x2(0);
    __nv_bfloat162 _hxa_98 = amax;
    __nv_bfloat162 _hxb_98 = __as_bf16x2(dgate[0]);
    unsigned int _hxr_98;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_98) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_98)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_98)));
    __nv_bfloat162 _max_97 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_98);
    amax = _max_97;
    __nv_bfloat162 _hxa_100 = amax;
    __nv_bfloat162 _hxb_100 = __as_bf16x2(dgate[1]);
    unsigned int _hxr_100;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_100) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_100)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_100)));
    __nv_bfloat162 _max_99 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_100);
    amax = _max_99;
    __nv_bfloat162 _hxa_102 = amax;
    __nv_bfloat162 _hxb_102 = __as_bf16x2(dgate[2]);
    unsigned int _hxr_102;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_102) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_102)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_102)));
    __nv_bfloat162 _max_101 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_102);
    amax = _max_101;
    __nv_bfloat162 _hxa_104 = amax;
    __nv_bfloat162 _hxb_104 = __as_bf16x2(dgate[3]);
    unsigned int _hxr_104;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_104) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_104)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_104)));
    __nv_bfloat162 _max_103 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_104);
    amax = _max_103;
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
    uint32_t bf16x2_mul;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul) : "r"(dgate[0]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1) : "r"(dgate[1]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_106;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_106) : "r"(bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_106) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_105;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_105) : "h"(e4m3x2_bf16x2), "h"(_cvt_satfinite_e4m3x2_bf16x2_106));
    words[0] = _mov_b32_pack_u16x2_105;
    uint32_t bf16x2_mul_0;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_0) : "r"(dgate[2]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_0) : "r"(dgate[3]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_108;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_108) : "r"(bf16x2_mul_1_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_108) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_107;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_107) : "h"(e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_108));
    words[1] = _mov_b32_pack_u16x2_107;
    reinterpret_cast<int2*>(row_output + out_index)[0] = reinterpret_cast<int2*>(words + 0)[0];
    __nv_bfloat162 amax_0 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_110 = amax_0;
    __nv_bfloat162 _hxb_110 = __as_bf16x2(dup[0]);
    unsigned int _hxr_110;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_110) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_110)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_110)));
    __nv_bfloat162 _max_109 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_110);
    amax_0 = _max_109;
    __nv_bfloat162 _hxa_112 = amax_0;
    __nv_bfloat162 _hxb_112 = __as_bf16x2(dup[1]);
    unsigned int _hxr_112;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_112) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_112)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_112)));
    __nv_bfloat162 _max_111 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_112);
    amax_0 = _max_111;
    __nv_bfloat162 _hxa_114 = amax_0;
    __nv_bfloat162 _hxb_114 = __as_bf16x2(dup[2]);
    unsigned int _hxr_114;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_114) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_114)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_114)));
    __nv_bfloat162 _max_113 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_114);
    amax_0 = _max_113;
    __nv_bfloat162 _hxa_116 = amax_0;
    __nv_bfloat162 _hxb_116 = __as_bf16x2(dup[3]);
    unsigned int _hxr_116;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_116) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_116)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_116)));
    __nv_bfloat162 _max_115 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_116);
    amax_0 = _max_115;
    unsigned int mag_0 = __as_u32(amax_0) & 2147450879;
    unsigned int peer1_0 = __shfl_xor_sync(0xFFFFFFFF, mag_0, 1);
    __nv_bfloat162 pair_max_0 = __hmax2_nan(__as_bf16x2(mag_0), __as_bf16x2(peer1_0));
    unsigned int peer2_0 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(pair_max_0), 2);
    __nv_bfloat162 quad_max_0 = __hmax2_nan(pair_max_0, __as_bf16x2(peer2_0));
    unsigned int folded_0 = __as_u32(__hmax2_nan(quad_max_0, __as_bf16x2(__as_u32(quad_max_0) >> 16)));
    unsigned int amax_bits_0 = (folded_0 & 65535) << 16;
    int scale_i32_0 = (int)(amax_bits_0 + 2031616 >> 23) - 8;
    if (scale_i32_0 < 0) {
        scale_i32_0 = 0;
    }
    if ((amax_bits_0 & 2139095040) == 2139095040) {
        scale_i32_0 = 255;
    }
    unsigned int scale_0 = (unsigned int)scale_i32_0;
    unsigned int inv_0 = 254 - scale_0 << 7;
    if (scale_0 == 255) {
        inv_0 = 32704;
    }
    unsigned int inv2_0 = inv_0 | inv_0 << 16;
    int scale_index_0 = ((grow >> 7) * scale_cols + (bx + (K >> 5) >> 2)) * 512 + (grow & 31) * 16 + (grow >> 5 & 3) * 4 + (bx + (K >> 5) & 3);
    if (quarter == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + scale_index_0) + (0)) = (unsigned char)(scale_0);
    }
    unsigned int words_0[2];
    uint32_t bf16x2_mul_2;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2) : "r"(dup[0]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_1) : "r"(dup[1]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_118;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_118) : "r"(bf16x2_mul_1_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_118) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_117;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_117) : "h"(e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_118));
    words_0[0] = _mov_b32_pack_u16x2_117;
    uint32_t bf16x2_mul_3;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3) : "r"(dup[2]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_2) : "r"(dup[3]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_120;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_120) : "r"(bf16x2_mul_1_2));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_2);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_120) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_119;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_119) : "h"(e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_120));
    words_0[1] = _mov_b32_pack_u16x2_119;
    reinterpret_cast<int2*>(row_output + (out_index + K))[0] = reinterpret_cast<int2*>(words_0 + 0)[0];
}

} // extern "C"
