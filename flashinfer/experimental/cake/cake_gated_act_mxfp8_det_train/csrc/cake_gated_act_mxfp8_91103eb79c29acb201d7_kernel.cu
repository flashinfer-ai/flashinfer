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
kernel_cake_gated_act_mxfp8_91103eb79c29acb201d7(__nv_bfloat16* __restrict__ gated_input, __nv_bfloat16* __restrict__ grad, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int half = tid & 1;
    int blk = tid >> 1 & 1;
    int row = tid >> 2;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int grow = by * 32 + row;
    int col = bx * 64 + blk * 32 + half * 16;
    int gate_index = grow * (2 * K) + col;
    unsigned int vec_load[8];
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
        vec_load[0 + 0] = (uint32_t)_iv_0_0;
        vec_load[0 + 1] = (uint32_t)_iv_0_1;
        vec_load[0 + 2] = (uint32_t)_iv_0_2;
        vec_load[0 + 3] = (uint32_t)_iv_0_3;
        vec_load[0 + 4] = (uint32_t)_iv_0_4;
        vec_load[0 + 5] = (uint32_t)_iv_0_5;
        vec_load[0 + 6] = (uint32_t)_iv_0_6;
        vec_load[0 + 7] = (uint32_t)_iv_0_7;
    }
    unsigned int vec_load_1[8];
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
        vec_load_1[0 + 0] = (uint32_t)_iv_1_0;
        vec_load_1[0 + 1] = (uint32_t)_iv_1_1;
        vec_load_1[0 + 2] = (uint32_t)_iv_1_2;
        vec_load_1[0 + 3] = (uint32_t)_iv_1_3;
        vec_load_1[0 + 4] = (uint32_t)_iv_1_4;
        vec_load_1[0 + 5] = (uint32_t)_iv_1_5;
        vec_load_1[0 + 6] = (uint32_t)_iv_1_6;
        vec_load_1[0 + 7] = (uint32_t)_iv_1_7;
    }
    unsigned int vec_load_2[8];
    {
        uint32_t _iv_2_0;
        uint32_t _iv_2_1;
        uint32_t _iv_2_2;
        uint32_t _iv_2_3;
        uint32_t _iv_2_4;
        uint32_t _iv_2_5;
        uint32_t _iv_2_6;
        uint32_t _iv_2_7;
        asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_iv_2_0), "=r"(_iv_2_1), "=r"(_iv_2_2), "=r"(_iv_2_3), "=r"(_iv_2_4), "=r"(_iv_2_5), "=r"(_iv_2_6), "=r"(_iv_2_7) : "l"((const void*)(grad + (grow * K + col) + (0))) : "memory");
        vec_load_2[0 + 0] = (uint32_t)_iv_2_0;
        vec_load_2[0 + 1] = (uint32_t)_iv_2_1;
        vec_load_2[0 + 2] = (uint32_t)_iv_2_2;
        vec_load_2[0 + 3] = (uint32_t)_iv_2_3;
        vec_load_2[0 + 4] = (uint32_t)_iv_2_4;
        vec_load_2[0 + 5] = (uint32_t)_iv_2_5;
        vec_load_2[0 + 6] = (uint32_t)_iv_2_6;
        vec_load_2[0 + 7] = (uint32_t)_iv_2_7;
    }
    unsigned int dgate[8];
    unsigned int dup[8];
    __nv_bfloat162 lowest = __as_bf16x2(vec_load[0]);
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[7]));
    float2 lowest_f32 = cake_bf16x2_to_f32x2(lowest);
    float _min_4;
    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_4) : "f"(lowest_f32.x), "f"(lowest_f32.y));
    int _warp_vote_3 = __all_sync(0xFFFFFFFF, _min_4 > -87.0f);
    if (_warp_vote_3 != 0) {
        float2 x = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[0]));
        float2 u = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[0]));
        float2 g = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[0]));
        float e0 = approx_exp2(-x.x * 1.442695f);
        float e1 = approx_exp2(-x.y * 1.442695f);
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
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act) : "l"(*(const unsigned long long*)&x), "l"(*(const unsigned long long*)&s));
        float2 _f32x2_mul_a_9 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s), s), g);
        float2 mul;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul) : "l"(*(const unsigned long long*)&_f32x2_mul_a_9), "l"(*(const unsigned long long*)&u));
        float2 mul_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1) : "l"(*(const unsigned long long*)&act), "l"(*(const unsigned long long*)&g));
        __nv_bfloat162 bf16x2_0 = __float22bfloat162_rn(make_float2(mul.x, mul.y));
        __nv_bfloat162 bf16x2_1_0 = __float22bfloat162_rn(make_float2(mul_1.x, mul_1.y));
        dgate[0] = __as_u32(bf16x2_0);
        dup[0] = __as_u32(bf16x2_1_0);
        float2 x_0 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[1]));
        float2 u_0 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[1]));
        float2 g_0 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[1]));
        float e0_0 = approx_exp2(-x_0.x * 1.442695f);
        float e1_0 = approx_exp2(-x_0.y * 1.442695f);
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
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_0) : "l"(*(const unsigned long long*)&x_0), "l"(*(const unsigned long long*)&s_0));
        float2 _f32x2_mul_a_14 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_0), s_0), g_0);
        float2 mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&u_0));
        float2 mul_1_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_0) : "l"(*(const unsigned long long*)&act_0), "l"(*(const unsigned long long*)&g_0));
        __nv_bfloat162 bf16x2_2 = __float22bfloat162_rn(make_float2(mul_0.x, mul_0.y));
        __nv_bfloat162 bf16x2_1_1 = __float22bfloat162_rn(make_float2(mul_1_0.x, mul_1_0.y));
        dgate[1] = __as_u32(bf16x2_2);
        dup[1] = __as_u32(bf16x2_1_1);
        float2 x_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[2]));
        float2 u_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[2]));
        float2 g_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[2]));
        float e0_1 = approx_exp2(-x_1.x * 1.442695f);
        float e1_1 = approx_exp2(-x_1.y * 1.442695f);
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
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_1) : "l"(*(const unsigned long long*)&x_1), "l"(*(const unsigned long long*)&s_1));
        float2 _f32x2_mul_a_19 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1), s_1), g_1);
        float2 mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&u_1));
        float2 mul_1_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_1) : "l"(*(const unsigned long long*)&act_1), "l"(*(const unsigned long long*)&g_1));
        __nv_bfloat162 bf16x2_3 = __float22bfloat162_rn(make_float2(mul_2.x, mul_2.y));
        __nv_bfloat162 bf16x2_1_2 = __float22bfloat162_rn(make_float2(mul_1_1.x, mul_1_1.y));
        dgate[2] = __as_u32(bf16x2_3);
        dup[2] = __as_u32(bf16x2_1_2);
        float2 x_2 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[3]));
        float2 u_2 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[3]));
        float2 g_2 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[3]));
        float e0_2 = approx_exp2(-x_2.x * 1.442695f);
        float e1_2 = approx_exp2(-x_2.y * 1.442695f);
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
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_2) : "l"(*(const unsigned long long*)&x_2), "l"(*(const unsigned long long*)&s_2));
        float2 _f32x2_mul_a_24 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_2), s_2), g_2);
        float2 mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&u_2));
        float2 mul_1_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_2) : "l"(*(const unsigned long long*)&act_2), "l"(*(const unsigned long long*)&g_2));
        __nv_bfloat162 bf16x2_4 = __float22bfloat162_rn(make_float2(mul_3.x, mul_3.y));
        __nv_bfloat162 bf16x2_1_3 = __float22bfloat162_rn(make_float2(mul_1_2.x, mul_1_2.y));
        dgate[3] = __as_u32(bf16x2_4);
        dup[3] = __as_u32(bf16x2_1_3);
        float2 x_3 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[4]));
        float2 u_3 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[4]));
        float2 g_3 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[4]));
        float e0_3 = approx_exp2(-x_3.x * 1.442695f);
        float e1_3 = approx_exp2(-x_3.y * 1.442695f);
        float d0_3 = 1.0f + e0_3;
        float d1_3 = 1.0f + e1_3;
        float r0_3 = approx_rcp(d0_3);
        float r1_3 = approx_rcp(d1_3);
        float _fma_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(-d0_3), "f"(r0_3), "f"(1.0f));
        float _fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(r0_3), "f"(_fma_26), "f"(r0_3));
        float _fma_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(-d1_3), "f"(r1_3), "f"(1.0f));
        float _fma_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(r1_3), "f"(_fma_28), "f"(r1_3));
        float2 s_3 = make_float2(_fma_25, _fma_27);
        float2 act_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_3) : "l"(*(const unsigned long long*)&x_3), "l"(*(const unsigned long long*)&s_3));
        float2 _f32x2_mul_a_29 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_3), s_3), g_3);
        float2 mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&u_3));
        float2 mul_1_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_3) : "l"(*(const unsigned long long*)&act_3), "l"(*(const unsigned long long*)&g_3));
        __nv_bfloat162 bf16x2_5 = __float22bfloat162_rn(make_float2(mul_4.x, mul_4.y));
        __nv_bfloat162 bf16x2_1_4 = __float22bfloat162_rn(make_float2(mul_1_3.x, mul_1_3.y));
        dgate[4] = __as_u32(bf16x2_5);
        dup[4] = __as_u32(bf16x2_1_4);
        float2 x_4 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[5]));
        float2 u_4 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[5]));
        float2 g_4 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[5]));
        float e0_4 = approx_exp2(-x_4.x * 1.442695f);
        float e1_4 = approx_exp2(-x_4.y * 1.442695f);
        float d0_4 = 1.0f + e0_4;
        float d1_4 = 1.0f + e1_4;
        float r0_4 = approx_rcp(d0_4);
        float r1_4 = approx_rcp(d1_4);
        float _fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(-d0_4), "f"(r0_4), "f"(1.0f));
        float _fma_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(r0_4), "f"(_fma_31), "f"(r0_4));
        float _fma_33;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(-d1_4), "f"(r1_4), "f"(1.0f));
        float _fma_32;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(r1_4), "f"(_fma_33), "f"(r1_4));
        float2 s_4 = make_float2(_fma_30, _fma_32);
        float2 act_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_4) : "l"(*(const unsigned long long*)&x_4), "l"(*(const unsigned long long*)&s_4));
        float2 _f32x2_mul_a_34 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_4), s_4), g_4);
        float2 mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_34), "l"(*(const unsigned long long*)&u_4));
        float2 mul_1_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_4) : "l"(*(const unsigned long long*)&act_4), "l"(*(const unsigned long long*)&g_4));
        __nv_bfloat162 bf16x2_6 = __float22bfloat162_rn(make_float2(mul_5.x, mul_5.y));
        __nv_bfloat162 bf16x2_1_5 = __float22bfloat162_rn(make_float2(mul_1_4.x, mul_1_4.y));
        dgate[5] = __as_u32(bf16x2_6);
        dup[5] = __as_u32(bf16x2_1_5);
        float2 x_5 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[6]));
        float2 u_5 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[6]));
        float2 g_5 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[6]));
        float e0_5 = approx_exp2(-x_5.x * 1.442695f);
        float e1_5 = approx_exp2(-x_5.y * 1.442695f);
        float d0_5 = 1.0f + e0_5;
        float d1_5 = 1.0f + e1_5;
        float r0_5 = approx_rcp(d0_5);
        float r1_5 = approx_rcp(d1_5);
        float _fma_36;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(-d0_5), "f"(r0_5), "f"(1.0f));
        float _fma_35;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(r0_5), "f"(_fma_36), "f"(r0_5));
        float _fma_38;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(-d1_5), "f"(r1_5), "f"(1.0f));
        float _fma_37;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(r1_5), "f"(_fma_38), "f"(r1_5));
        float2 s_5 = make_float2(_fma_35, _fma_37);
        float2 act_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_5) : "l"(*(const unsigned long long*)&x_5), "l"(*(const unsigned long long*)&s_5));
        float2 _f32x2_mul_a_39 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_5), s_5), g_5);
        float2 mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_39), "l"(*(const unsigned long long*)&u_5));
        float2 mul_1_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_5) : "l"(*(const unsigned long long*)&act_5), "l"(*(const unsigned long long*)&g_5));
        __nv_bfloat162 bf16x2_7 = __float22bfloat162_rn(make_float2(mul_6.x, mul_6.y));
        __nv_bfloat162 bf16x2_1_6 = __float22bfloat162_rn(make_float2(mul_1_5.x, mul_1_5.y));
        dgate[6] = __as_u32(bf16x2_7);
        dup[6] = __as_u32(bf16x2_1_6);
        float2 x_6 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[7]));
        float2 u_6 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[7]));
        float2 g_6 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[7]));
        float e0_6 = approx_exp2(-x_6.x * 1.442695f);
        float e1_6 = approx_exp2(-x_6.y * 1.442695f);
        float d0_6 = 1.0f + e0_6;
        float d1_6 = 1.0f + e1_6;
        float r0_6 = approx_rcp(d0_6);
        float r1_6 = approx_rcp(d1_6);
        float _fma_41;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(-d0_6), "f"(r0_6), "f"(1.0f));
        float _fma_40;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(r0_6), "f"(_fma_41), "f"(r0_6));
        float _fma_43;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(-d1_6), "f"(r1_6), "f"(1.0f));
        float _fma_42;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(r1_6), "f"(_fma_43), "f"(r1_6));
        float2 s_6 = make_float2(_fma_40, _fma_42);
        float2 act_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_6) : "l"(*(const unsigned long long*)&x_6), "l"(*(const unsigned long long*)&s_6));
        float2 _f32x2_mul_a_44 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_6), s_6), g_6);
        float2 mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&u_6));
        float2 mul_1_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_6) : "l"(*(const unsigned long long*)&act_6), "l"(*(const unsigned long long*)&g_6));
        __nv_bfloat162 bf16x2_8 = __float22bfloat162_rn(make_float2(mul_7.x, mul_7.y));
        __nv_bfloat162 bf16x2_1_7 = __float22bfloat162_rn(make_float2(mul_1_6.x, mul_1_6.y));
        dgate[7] = __as_u32(bf16x2_8);
        dup[7] = __as_u32(bf16x2_1_7);
    } else {
        int band = 0;
        float2 x_7 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[0]));
        float2 u_7 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[0]));
        float2 g_7 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[0]));
        float e0_7 = approx_exp2(-x_7.x * 1.442695f);
        float e1_7 = approx_exp2(-x_7.y * 1.442695f);
        float d0_7 = 1.0f + e0_7;
        float d1_7 = 1.0f + e1_7;
        float r0_7 = approx_rcp(d0_7);
        float r1_7 = approx_rcp(d1_7);
        float _fma_45;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(-d0_7), "f"(r0_7), "f"(1.0f));
        float fma;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma) : "f"(r0_7), "f"(_fma_45), "f"(r0_7));
        float _fma_46;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(-d1_7), "f"(r1_7), "f"(1.0f));
        float fma_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1) : "f"(r1_7), "f"(_fma_46), "f"(r1_7));
        float t0 = fma;
        float t1 = fma_1;
        int _warp_vote_47 = __any_sync(0xFFFFFFFF, fminf(t0, t1) == 0.0f);
        if (_warp_vote_47 != 0) {
            float d4_1a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a) : "f"(d0_7), "f"(0.25f));
            float r4_1a = approx_rcp(d4_1a);
            float _fma_49;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(-d4_1a), "f"(r4_1a), "f"(1.0f));
            float _fma_48;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(r4_1a), "f"(_fma_49), "f"(r4_1a));
            float big_1a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a) : "f"(mul_f32_rn_noftz(_fma_48, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_50;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_50) : "f"(big_1a), "f"(8.388608e+06f));
            float whole_1a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a) : "f"(_add_50), "f"(8.388608e+06f));
            float above_1a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a) : "f"(-d4_1a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_52;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_52) : "f"(big_1a), "f"(whole_1a));
            float _abs_51;
            asm("abs.f32 %0, %1;" : "=f"(_abs_51) : "f"(_sub_52));
            float _add_53;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53) : "f"(big_1a), "f"(0.5f));
            float _sub_54;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_54) : "f"(big_1a), "f"(0.5f));
            float rounded_1a = ((_abs_51 == 0.5f) ? ((above_1a > 0.0f) ? _add_53 : _sub_54) : whole_1a);
            t0 = ((d0_7 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a, 7.888609e-31f), 1.7763568e-15f) : t0);
            float d4_1a_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_0) : "f"(d1_7), "f"(0.25f));
            float r4_1a_0 = approx_rcp(d4_1a_0);
            float _fma_56;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(-d4_1a_0), "f"(r4_1a_0), "f"(1.0f));
            float _fma_55;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(r4_1a_0), "f"(_fma_56), "f"(r4_1a_0));
            float big_1a_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_0) : "f"(mul_f32_rn_noftz(_fma_55, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_57;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(big_1a_0), "f"(8.388608e+06f));
            float whole_1a_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_0) : "f"(_add_57), "f"(8.388608e+06f));
            float above_1a_0;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_0) : "f"(-d4_1a_0), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_0, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_59;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_59) : "f"(big_1a_0), "f"(whole_1a_0));
            float _abs_58;
            asm("abs.f32 %0, %1;" : "=f"(_abs_58) : "f"(_sub_59));
            float _add_60;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_60) : "f"(big_1a_0), "f"(0.5f));
            float _sub_61;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_61) : "f"(big_1a_0), "f"(0.5f));
            float rounded_1a_0 = ((_abs_58 == 0.5f) ? ((above_1a_0 > 0.0f) ? _add_60 : _sub_61) : whole_1a_0);
            t1 = ((d1_7 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_0, 7.888609e-31f), 1.7763568e-15f) : t1);
        }
        float2 s_7 = make_float2(max_noftz(t0, 0.0f), max_noftz(t1, 0.0f));
        float2 act_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_7) : "l"(*(const unsigned long long*)&x_7), "l"(*(const unsigned long long*)&s_7));
        float2 _f32x2_mul_a_62 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_7, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_7), s_7), g_7);
        float2 mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&u_7));
        float2 mul_1_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_7) : "l"(*(const unsigned long long*)&act_7), "l"(*(const unsigned long long*)&g_7));
        __nv_bfloat162 bf16x2_9 = __float22bfloat162_rn(make_float2(mul_8.x, mul_8.y));
        __nv_bfloat162 bf16x2_1_8 = __float22bfloat162_rn(make_float2(mul_1_7.x, mul_1_7.y));
        dgate[0] = __as_u32(bf16x2_9);
        dup[0] = __as_u32(bf16x2_1_8);
        float2 x_7_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[1]));
        float2 u_7_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[1]));
        float2 g_7_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[1]));
        float e0_7_1 = approx_exp2(-x_7_1.x * 1.442695f);
        float e1_7_1 = approx_exp2(-x_7_1.y * 1.442695f);
        float d0_7_1 = 1.0f + e0_7_1;
        float d1_7_1 = 1.0f + e1_7_1;
        float r0_7_1 = approx_rcp(d0_7_1);
        float r1_7_1 = approx_rcp(d1_7_1);
        float _fma_63;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(-d0_7_1), "f"(r0_7_1), "f"(1.0f));
        float fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_0) : "f"(r0_7_1), "f"(_fma_63), "f"(r0_7_1));
        float _fma_64;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(-d1_7_1), "f"(r1_7_1), "f"(1.0f));
        float fma_1_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_0) : "f"(r1_7_1), "f"(_fma_64), "f"(r1_7_1));
        float t0_0 = fma_0;
        float t1_0 = fma_1_0;
        int _warp_vote_65 = __any_sync(0xFFFFFFFF, fminf(t0_0, t1_0) == 0.0f);
        if (_warp_vote_65 != 0) {
            float d4_1a_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_1) : "f"(d0_7_1), "f"(0.25f));
            float r4_1a_1 = approx_rcp(d4_1a_1);
            float _fma_67;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(-d4_1a_1), "f"(r4_1a_1), "f"(1.0f));
            float _fma_66;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(r4_1a_1), "f"(_fma_67), "f"(r4_1a_1));
            float big_1a_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_1) : "f"(mul_f32_rn_noftz(_fma_66, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_68;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(big_1a_1), "f"(8.388608e+06f));
            float whole_1a_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_1) : "f"(_add_68), "f"(8.388608e+06f));
            float above_1a_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_1) : "f"(-d4_1a_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_70;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_70) : "f"(big_1a_1), "f"(whole_1a_1));
            float _abs_69;
            asm("abs.f32 %0, %1;" : "=f"(_abs_69) : "f"(_sub_70));
            float _add_71;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_71) : "f"(big_1a_1), "f"(0.5f));
            float _sub_72;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_72) : "f"(big_1a_1), "f"(0.5f));
            float rounded_1a_1 = ((_abs_69 == 0.5f) ? ((above_1a_1 > 0.0f) ? _add_71 : _sub_72) : whole_1a_1);
            t0_0 = ((d0_7_1 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_1, 7.888609e-31f), 1.7763568e-15f) : t0_0);
            float d4_1a_1_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_1_1) : "f"(d1_7_1), "f"(0.25f));
            float r4_1a_1_1 = approx_rcp(d4_1a_1_1);
            float _fma_74;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(-d4_1a_1_1), "f"(r4_1a_1_1), "f"(1.0f));
            float _fma_73;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(r4_1a_1_1), "f"(_fma_74), "f"(r4_1a_1_1));
            float big_1a_1_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_1_1) : "f"(mul_f32_rn_noftz(_fma_73, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_75;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_75) : "f"(big_1a_1_1), "f"(8.388608e+06f));
            float whole_1a_1_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_1_1) : "f"(_add_75), "f"(8.388608e+06f));
            float above_1a_1_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_1_1) : "f"(-d4_1a_1_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_1_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_77;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_77) : "f"(big_1a_1_1), "f"(whole_1a_1_1));
            float _abs_76;
            asm("abs.f32 %0, %1;" : "=f"(_abs_76) : "f"(_sub_77));
            float _add_78;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_78) : "f"(big_1a_1_1), "f"(0.5f));
            float _sub_79;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_79) : "f"(big_1a_1_1), "f"(0.5f));
            float rounded_1a_1_1 = ((_abs_76 == 0.5f) ? ((above_1a_1_1 > 0.0f) ? _add_78 : _sub_79) : whole_1a_1_1);
            t1_0 = ((d1_7_1 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_1_1, 7.888609e-31f), 1.7763568e-15f) : t1_0);
        }
        float2 s_7_1 = make_float2(max_noftz(t0_0, 0.0f), max_noftz(t1_0, 0.0f));
        float2 act_7_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_7_1) : "l"(*(const unsigned long long*)&x_7_1), "l"(*(const unsigned long long*)&s_7_1));
        float2 _f32x2_mul_a_80 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_7_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_7_1), s_7_1), g_7_1);
        float2 mul_8_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_8_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_80), "l"(*(const unsigned long long*)&u_7_1));
        float2 mul_1_7_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_7_1) : "l"(*(const unsigned long long*)&act_7_1), "l"(*(const unsigned long long*)&g_7_1));
        __nv_bfloat162 bf16x2_10 = __float22bfloat162_rn(make_float2(mul_8_1.x, mul_8_1.y));
        __nv_bfloat162 bf16x2_1_9 = __float22bfloat162_rn(make_float2(mul_1_7_1.x, mul_1_7_1.y));
        dgate[1] = __as_u32(bf16x2_10);
        dup[1] = __as_u32(bf16x2_1_9);
        float2 x_8 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[2]));
        float2 u_8 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[2]));
        float2 g_8 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[2]));
        float e0_8 = approx_exp2(-x_8.x * 1.442695f);
        float e1_8 = approx_exp2(-x_8.y * 1.442695f);
        float d0_8 = 1.0f + e0_8;
        float d1_8 = 1.0f + e1_8;
        float r0_8 = approx_rcp(d0_8);
        float r1_8 = approx_rcp(d1_8);
        float _fma_81;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(-d0_8), "f"(r0_8), "f"(1.0f));
        float fma_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_2) : "f"(r0_8), "f"(_fma_81), "f"(r0_8));
        float _fma_82;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(-d1_8), "f"(r1_8), "f"(1.0f));
        float fma_1_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_1) : "f"(r1_8), "f"(_fma_82), "f"(r1_8));
        float t0_1 = fma_2;
        float t1_1 = fma_1_1;
        int _warp_vote_83 = __any_sync(0xFFFFFFFF, fminf(t0_1, t1_1) == 0.0f);
        if (_warp_vote_83 != 0) {
            float d4_1a_2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_2) : "f"(d0_8), "f"(0.25f));
            float r4_1a_2 = approx_rcp(d4_1a_2);
            float _fma_85;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(-d4_1a_2), "f"(r4_1a_2), "f"(1.0f));
            float _fma_84;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(r4_1a_2), "f"(_fma_85), "f"(r4_1a_2));
            float big_1a_2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_2) : "f"(mul_f32_rn_noftz(_fma_84, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_86;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_86) : "f"(big_1a_2), "f"(8.388608e+06f));
            float whole_1a_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_2) : "f"(_add_86), "f"(8.388608e+06f));
            float above_1a_2;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_2) : "f"(-d4_1a_2), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_2, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_88;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_88) : "f"(big_1a_2), "f"(whole_1a_2));
            float _abs_87;
            asm("abs.f32 %0, %1;" : "=f"(_abs_87) : "f"(_sub_88));
            float _add_89;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_89) : "f"(big_1a_2), "f"(0.5f));
            float _sub_90;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_90) : "f"(big_1a_2), "f"(0.5f));
            float rounded_1a_2 = ((_abs_87 == 0.5f) ? ((above_1a_2 > 0.0f) ? _add_89 : _sub_90) : whole_1a_2);
            t0_1 = ((d0_8 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_2, 7.888609e-31f), 1.7763568e-15f) : t0_1);
            float d4_1a_2_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_2_1) : "f"(d1_8), "f"(0.25f));
            float r4_1a_2_1 = approx_rcp(d4_1a_2_1);
            float _fma_92;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(-d4_1a_2_1), "f"(r4_1a_2_1), "f"(1.0f));
            float _fma_91;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(r4_1a_2_1), "f"(_fma_92), "f"(r4_1a_2_1));
            float big_1a_2_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_2_1) : "f"(mul_f32_rn_noftz(_fma_91, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_93;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(big_1a_2_1), "f"(8.388608e+06f));
            float whole_1a_2_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_2_1) : "f"(_add_93), "f"(8.388608e+06f));
            float above_1a_2_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_2_1) : "f"(-d4_1a_2_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_2_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_95;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_95) : "f"(big_1a_2_1), "f"(whole_1a_2_1));
            float _abs_94;
            asm("abs.f32 %0, %1;" : "=f"(_abs_94) : "f"(_sub_95));
            float _add_96;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_96) : "f"(big_1a_2_1), "f"(0.5f));
            float _sub_97;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_97) : "f"(big_1a_2_1), "f"(0.5f));
            float rounded_1a_2_1 = ((_abs_94 == 0.5f) ? ((above_1a_2_1 > 0.0f) ? _add_96 : _sub_97) : whole_1a_2_1);
            t1_1 = ((d1_8 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_2_1, 7.888609e-31f), 1.7763568e-15f) : t1_1);
        }
        float2 s_8 = make_float2(max_noftz(t0_1, 0.0f), max_noftz(t1_1, 0.0f));
        float2 act_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_8) : "l"(*(const unsigned long long*)&x_8), "l"(*(const unsigned long long*)&s_8));
        float2 _f32x2_mul_a_98 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_8, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_8), s_8), g_8);
        float2 mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_98), "l"(*(const unsigned long long*)&u_8));
        float2 mul_1_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_8) : "l"(*(const unsigned long long*)&act_8), "l"(*(const unsigned long long*)&g_8));
        __nv_bfloat162 bf16x2_11 = __float22bfloat162_rn(make_float2(mul_9.x, mul_9.y));
        __nv_bfloat162 bf16x2_1_10 = __float22bfloat162_rn(make_float2(mul_1_8.x, mul_1_8.y));
        dgate[2] = __as_u32(bf16x2_11);
        dup[2] = __as_u32(bf16x2_1_10);
        float2 x_9 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[3]));
        float2 u_9 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[3]));
        float2 g_9 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[3]));
        float e0_9 = approx_exp2(-x_9.x * 1.442695f);
        float e1_9 = approx_exp2(-x_9.y * 1.442695f);
        float d0_9 = 1.0f + e0_9;
        float d1_9 = 1.0f + e1_9;
        float r0_9 = approx_rcp(d0_9);
        float r1_9 = approx_rcp(d1_9);
        float _fma_99;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(-d0_9), "f"(r0_9), "f"(1.0f));
        float fma_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_3) : "f"(r0_9), "f"(_fma_99), "f"(r0_9));
        float _fma_100;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_100) : "f"(-d1_9), "f"(r1_9), "f"(1.0f));
        float fma_1_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_2) : "f"(r1_9), "f"(_fma_100), "f"(r1_9));
        float t0_2 = fma_3;
        float t1_2 = fma_1_2;
        int _warp_vote_101 = __any_sync(0xFFFFFFFF, fminf(t0_2, t1_2) == 0.0f);
        if (_warp_vote_101 != 0) {
            float d4_1a_3;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_3) : "f"(d0_9), "f"(0.25f));
            float r4_1a_3 = approx_rcp(d4_1a_3);
            float _fma_103;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_103) : "f"(-d4_1a_3), "f"(r4_1a_3), "f"(1.0f));
            float _fma_102;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_102) : "f"(r4_1a_3), "f"(_fma_103), "f"(r4_1a_3));
            float big_1a_3;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_3) : "f"(mul_f32_rn_noftz(_fma_102, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_104;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_104) : "f"(big_1a_3), "f"(8.388608e+06f));
            float whole_1a_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_3) : "f"(_add_104), "f"(8.388608e+06f));
            float above_1a_3;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_3) : "f"(-d4_1a_3), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_3, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_106;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_106) : "f"(big_1a_3), "f"(whole_1a_3));
            float _abs_105;
            asm("abs.f32 %0, %1;" : "=f"(_abs_105) : "f"(_sub_106));
            float _add_107;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_107) : "f"(big_1a_3), "f"(0.5f));
            float _sub_108;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_108) : "f"(big_1a_3), "f"(0.5f));
            float rounded_1a_3 = ((_abs_105 == 0.5f) ? ((above_1a_3 > 0.0f) ? _add_107 : _sub_108) : whole_1a_3);
            t0_2 = ((d0_9 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_3, 7.888609e-31f), 1.7763568e-15f) : t0_2);
            float d4_1a_3_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_3_1) : "f"(d1_9), "f"(0.25f));
            float r4_1a_3_1 = approx_rcp(d4_1a_3_1);
            float _fma_110;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_110) : "f"(-d4_1a_3_1), "f"(r4_1a_3_1), "f"(1.0f));
            float _fma_109;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_109) : "f"(r4_1a_3_1), "f"(_fma_110), "f"(r4_1a_3_1));
            float big_1a_3_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_3_1) : "f"(mul_f32_rn_noftz(_fma_109, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_111;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_111) : "f"(big_1a_3_1), "f"(8.388608e+06f));
            float whole_1a_3_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_3_1) : "f"(_add_111), "f"(8.388608e+06f));
            float above_1a_3_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_3_1) : "f"(-d4_1a_3_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_3_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_113;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_113) : "f"(big_1a_3_1), "f"(whole_1a_3_1));
            float _abs_112;
            asm("abs.f32 %0, %1;" : "=f"(_abs_112) : "f"(_sub_113));
            float _add_114;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_114) : "f"(big_1a_3_1), "f"(0.5f));
            float _sub_115;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_115) : "f"(big_1a_3_1), "f"(0.5f));
            float rounded_1a_3_1 = ((_abs_112 == 0.5f) ? ((above_1a_3_1 > 0.0f) ? _add_114 : _sub_115) : whole_1a_3_1);
            t1_2 = ((d1_9 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_3_1, 7.888609e-31f), 1.7763568e-15f) : t1_2);
        }
        float2 s_9 = make_float2(max_noftz(t0_2, 0.0f), max_noftz(t1_2, 0.0f));
        float2 act_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_9) : "l"(*(const unsigned long long*)&x_9), "l"(*(const unsigned long long*)&s_9));
        float2 _f32x2_mul_a_116 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_9, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_9), s_9), g_9);
        float2 mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_116), "l"(*(const unsigned long long*)&u_9));
        float2 mul_1_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_9) : "l"(*(const unsigned long long*)&act_9), "l"(*(const unsigned long long*)&g_9));
        __nv_bfloat162 bf16x2_12 = __float22bfloat162_rn(make_float2(mul_10.x, mul_10.y));
        __nv_bfloat162 bf16x2_1_11 = __float22bfloat162_rn(make_float2(mul_1_9.x, mul_1_9.y));
        dgate[3] = __as_u32(bf16x2_12);
        dup[3] = __as_u32(bf16x2_1_11);
        float2 x_10 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[4]));
        float2 u_10 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[4]));
        float2 g_10 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[4]));
        float e0_10 = approx_exp2(-x_10.x * 1.442695f);
        float e1_10 = approx_exp2(-x_10.y * 1.442695f);
        float d0_10 = 1.0f + e0_10;
        float d1_10 = 1.0f + e1_10;
        float r0_10 = approx_rcp(d0_10);
        float r1_10 = approx_rcp(d1_10);
        float _fma_117;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_117) : "f"(-d0_10), "f"(r0_10), "f"(1.0f));
        float fma_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_4) : "f"(r0_10), "f"(_fma_117), "f"(r0_10));
        float _fma_118;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_118) : "f"(-d1_10), "f"(r1_10), "f"(1.0f));
        float fma_1_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_3) : "f"(r1_10), "f"(_fma_118), "f"(r1_10));
        float t0_3 = fma_4;
        float t1_3 = fma_1_3;
        int _warp_vote_119 = __any_sync(0xFFFFFFFF, fminf(t0_3, t1_3) == 0.0f);
        if (_warp_vote_119 != 0) {
            float d4_1a_4;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_4) : "f"(d0_10), "f"(0.25f));
            float r4_1a_4 = approx_rcp(d4_1a_4);
            float _fma_121;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_121) : "f"(-d4_1a_4), "f"(r4_1a_4), "f"(1.0f));
            float _fma_120;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_120) : "f"(r4_1a_4), "f"(_fma_121), "f"(r4_1a_4));
            float big_1a_4;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_4) : "f"(mul_f32_rn_noftz(_fma_120, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_122;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_122) : "f"(big_1a_4), "f"(8.388608e+06f));
            float whole_1a_4;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_4) : "f"(_add_122), "f"(8.388608e+06f));
            float above_1a_4;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_4) : "f"(-d4_1a_4), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_4, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_124;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_124) : "f"(big_1a_4), "f"(whole_1a_4));
            float _abs_123;
            asm("abs.f32 %0, %1;" : "=f"(_abs_123) : "f"(_sub_124));
            float _add_125;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_125) : "f"(big_1a_4), "f"(0.5f));
            float _sub_126;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_126) : "f"(big_1a_4), "f"(0.5f));
            float rounded_1a_4 = ((_abs_123 == 0.5f) ? ((above_1a_4 > 0.0f) ? _add_125 : _sub_126) : whole_1a_4);
            t0_3 = ((d0_10 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_4, 7.888609e-31f), 1.7763568e-15f) : t0_3);
            float d4_1a_4_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_4_1) : "f"(d1_10), "f"(0.25f));
            float r4_1a_4_1 = approx_rcp(d4_1a_4_1);
            float _fma_128;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_128) : "f"(-d4_1a_4_1), "f"(r4_1a_4_1), "f"(1.0f));
            float _fma_127;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_127) : "f"(r4_1a_4_1), "f"(_fma_128), "f"(r4_1a_4_1));
            float big_1a_4_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_4_1) : "f"(mul_f32_rn_noftz(_fma_127, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_129;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_129) : "f"(big_1a_4_1), "f"(8.388608e+06f));
            float whole_1a_4_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_4_1) : "f"(_add_129), "f"(8.388608e+06f));
            float above_1a_4_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_4_1) : "f"(-d4_1a_4_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_4_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_131;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_131) : "f"(big_1a_4_1), "f"(whole_1a_4_1));
            float _abs_130;
            asm("abs.f32 %0, %1;" : "=f"(_abs_130) : "f"(_sub_131));
            float _add_132;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_132) : "f"(big_1a_4_1), "f"(0.5f));
            float _sub_133;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_133) : "f"(big_1a_4_1), "f"(0.5f));
            float rounded_1a_4_1 = ((_abs_130 == 0.5f) ? ((above_1a_4_1 > 0.0f) ? _add_132 : _sub_133) : whole_1a_4_1);
            t1_3 = ((d1_10 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_4_1, 7.888609e-31f), 1.7763568e-15f) : t1_3);
        }
        float2 s_10 = make_float2(max_noftz(t0_3, 0.0f), max_noftz(t1_3, 0.0f));
        float2 act_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_10) : "l"(*(const unsigned long long*)&x_10), "l"(*(const unsigned long long*)&s_10));
        float2 _f32x2_mul_a_134 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_10, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_10), s_10), g_10);
        float2 mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_134), "l"(*(const unsigned long long*)&u_10));
        float2 mul_1_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_10) : "l"(*(const unsigned long long*)&act_10), "l"(*(const unsigned long long*)&g_10));
        __nv_bfloat162 bf16x2_13 = __float22bfloat162_rn(make_float2(mul_11.x, mul_11.y));
        __nv_bfloat162 bf16x2_1_12 = __float22bfloat162_rn(make_float2(mul_1_10.x, mul_1_10.y));
        dgate[4] = __as_u32(bf16x2_13);
        dup[4] = __as_u32(bf16x2_1_12);
        float2 x_11 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[5]));
        float2 u_11 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[5]));
        float2 g_11 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[5]));
        float e0_11 = approx_exp2(-x_11.x * 1.442695f);
        float e1_11 = approx_exp2(-x_11.y * 1.442695f);
        float d0_11 = 1.0f + e0_11;
        float d1_11 = 1.0f + e1_11;
        float r0_11 = approx_rcp(d0_11);
        float r1_11 = approx_rcp(d1_11);
        float _fma_135;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_135) : "f"(-d0_11), "f"(r0_11), "f"(1.0f));
        float fma_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_5) : "f"(r0_11), "f"(_fma_135), "f"(r0_11));
        float _fma_136;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_136) : "f"(-d1_11), "f"(r1_11), "f"(1.0f));
        float fma_1_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_4) : "f"(r1_11), "f"(_fma_136), "f"(r1_11));
        float t0_4 = fma_5;
        float t1_4 = fma_1_4;
        int _warp_vote_137 = __any_sync(0xFFFFFFFF, fminf(t0_4, t1_4) == 0.0f);
        if (_warp_vote_137 != 0) {
            float d4_1a_5;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_5) : "f"(d0_11), "f"(0.25f));
            float r4_1a_5 = approx_rcp(d4_1a_5);
            float _fma_139;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_139) : "f"(-d4_1a_5), "f"(r4_1a_5), "f"(1.0f));
            float _fma_138;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_138) : "f"(r4_1a_5), "f"(_fma_139), "f"(r4_1a_5));
            float big_1a_5;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_5) : "f"(mul_f32_rn_noftz(_fma_138, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_140;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_140) : "f"(big_1a_5), "f"(8.388608e+06f));
            float whole_1a_5;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_5) : "f"(_add_140), "f"(8.388608e+06f));
            float above_1a_5;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_5) : "f"(-d4_1a_5), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_5, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_142;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_142) : "f"(big_1a_5), "f"(whole_1a_5));
            float _abs_141;
            asm("abs.f32 %0, %1;" : "=f"(_abs_141) : "f"(_sub_142));
            float _add_143;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_143) : "f"(big_1a_5), "f"(0.5f));
            float _sub_144;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_144) : "f"(big_1a_5), "f"(0.5f));
            float rounded_1a_5 = ((_abs_141 == 0.5f) ? ((above_1a_5 > 0.0f) ? _add_143 : _sub_144) : whole_1a_5);
            t0_4 = ((d0_11 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_5, 7.888609e-31f), 1.7763568e-15f) : t0_4);
            float d4_1a_5_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_5_1) : "f"(d1_11), "f"(0.25f));
            float r4_1a_5_1 = approx_rcp(d4_1a_5_1);
            float _fma_146;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_146) : "f"(-d4_1a_5_1), "f"(r4_1a_5_1), "f"(1.0f));
            float _fma_145;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_145) : "f"(r4_1a_5_1), "f"(_fma_146), "f"(r4_1a_5_1));
            float big_1a_5_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_5_1) : "f"(mul_f32_rn_noftz(_fma_145, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_147;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_147) : "f"(big_1a_5_1), "f"(8.388608e+06f));
            float whole_1a_5_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_5_1) : "f"(_add_147), "f"(8.388608e+06f));
            float above_1a_5_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_5_1) : "f"(-d4_1a_5_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_5_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_149;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_149) : "f"(big_1a_5_1), "f"(whole_1a_5_1));
            float _abs_148;
            asm("abs.f32 %0, %1;" : "=f"(_abs_148) : "f"(_sub_149));
            float _add_150;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_150) : "f"(big_1a_5_1), "f"(0.5f));
            float _sub_151;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_151) : "f"(big_1a_5_1), "f"(0.5f));
            float rounded_1a_5_1 = ((_abs_148 == 0.5f) ? ((above_1a_5_1 > 0.0f) ? _add_150 : _sub_151) : whole_1a_5_1);
            t1_4 = ((d1_11 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_5_1, 7.888609e-31f), 1.7763568e-15f) : t1_4);
        }
        float2 s_11 = make_float2(max_noftz(t0_4, 0.0f), max_noftz(t1_4, 0.0f));
        float2 act_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_11) : "l"(*(const unsigned long long*)&x_11), "l"(*(const unsigned long long*)&s_11));
        float2 _f32x2_mul_a_152 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_11, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_11), s_11), g_11);
        float2 mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_152), "l"(*(const unsigned long long*)&u_11));
        float2 mul_1_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_11) : "l"(*(const unsigned long long*)&act_11), "l"(*(const unsigned long long*)&g_11));
        __nv_bfloat162 bf16x2_14 = __float22bfloat162_rn(make_float2(mul_12.x, mul_12.y));
        __nv_bfloat162 bf16x2_1_13 = __float22bfloat162_rn(make_float2(mul_1_11.x, mul_1_11.y));
        dgate[5] = __as_u32(bf16x2_14);
        dup[5] = __as_u32(bf16x2_1_13);
        float2 x_12 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[6]));
        float2 u_12 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[6]));
        float2 g_12 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[6]));
        float e0_12 = approx_exp2(-x_12.x * 1.442695f);
        float e1_12 = approx_exp2(-x_12.y * 1.442695f);
        float d0_12 = 1.0f + e0_12;
        float d1_12 = 1.0f + e1_12;
        float r0_12 = approx_rcp(d0_12);
        float r1_12 = approx_rcp(d1_12);
        float _fma_153;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_153) : "f"(-d0_12), "f"(r0_12), "f"(1.0f));
        float fma_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_6) : "f"(r0_12), "f"(_fma_153), "f"(r0_12));
        float _fma_154;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_154) : "f"(-d1_12), "f"(r1_12), "f"(1.0f));
        float fma_1_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_5) : "f"(r1_12), "f"(_fma_154), "f"(r1_12));
        float t0_5 = fma_6;
        float t1_5 = fma_1_5;
        int _warp_vote_155 = __any_sync(0xFFFFFFFF, fminf(t0_5, t1_5) == 0.0f);
        if (_warp_vote_155 != 0) {
            float d4_1a_6;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_6) : "f"(d0_12), "f"(0.25f));
            float r4_1a_6 = approx_rcp(d4_1a_6);
            float _fma_157;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_157) : "f"(-d4_1a_6), "f"(r4_1a_6), "f"(1.0f));
            float _fma_156;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_156) : "f"(r4_1a_6), "f"(_fma_157), "f"(r4_1a_6));
            float big_1a_6;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_6) : "f"(mul_f32_rn_noftz(_fma_156, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_158;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_158) : "f"(big_1a_6), "f"(8.388608e+06f));
            float whole_1a_6;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_6) : "f"(_add_158), "f"(8.388608e+06f));
            float above_1a_6;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_6) : "f"(-d4_1a_6), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_6, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_160;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_160) : "f"(big_1a_6), "f"(whole_1a_6));
            float _abs_159;
            asm("abs.f32 %0, %1;" : "=f"(_abs_159) : "f"(_sub_160));
            float _add_161;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_161) : "f"(big_1a_6), "f"(0.5f));
            float _sub_162;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_162) : "f"(big_1a_6), "f"(0.5f));
            float rounded_1a_6 = ((_abs_159 == 0.5f) ? ((above_1a_6 > 0.0f) ? _add_161 : _sub_162) : whole_1a_6);
            t0_5 = ((d0_12 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_6, 7.888609e-31f), 1.7763568e-15f) : t0_5);
            float d4_1a_6_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_6_1) : "f"(d1_12), "f"(0.25f));
            float r4_1a_6_1 = approx_rcp(d4_1a_6_1);
            float _fma_164;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_164) : "f"(-d4_1a_6_1), "f"(r4_1a_6_1), "f"(1.0f));
            float _fma_163;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_163) : "f"(r4_1a_6_1), "f"(_fma_164), "f"(r4_1a_6_1));
            float big_1a_6_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_6_1) : "f"(mul_f32_rn_noftz(_fma_163, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_165;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_165) : "f"(big_1a_6_1), "f"(8.388608e+06f));
            float whole_1a_6_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_6_1) : "f"(_add_165), "f"(8.388608e+06f));
            float above_1a_6_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_6_1) : "f"(-d4_1a_6_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_6_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_167;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_167) : "f"(big_1a_6_1), "f"(whole_1a_6_1));
            float _abs_166;
            asm("abs.f32 %0, %1;" : "=f"(_abs_166) : "f"(_sub_167));
            float _add_168;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_168) : "f"(big_1a_6_1), "f"(0.5f));
            float _sub_169;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_169) : "f"(big_1a_6_1), "f"(0.5f));
            float rounded_1a_6_1 = ((_abs_166 == 0.5f) ? ((above_1a_6_1 > 0.0f) ? _add_168 : _sub_169) : whole_1a_6_1);
            t1_5 = ((d1_12 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_6_1, 7.888609e-31f), 1.7763568e-15f) : t1_5);
        }
        float2 s_12 = make_float2(max_noftz(t0_5, 0.0f), max_noftz(t1_5, 0.0f));
        float2 act_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_12) : "l"(*(const unsigned long long*)&x_12), "l"(*(const unsigned long long*)&s_12));
        float2 _f32x2_mul_a_170 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_12, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_12), s_12), g_12);
        float2 mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_170), "l"(*(const unsigned long long*)&u_12));
        float2 mul_1_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_12) : "l"(*(const unsigned long long*)&act_12), "l"(*(const unsigned long long*)&g_12));
        __nv_bfloat162 bf16x2_15 = __float22bfloat162_rn(make_float2(mul_13.x, mul_13.y));
        __nv_bfloat162 bf16x2_1_14 = __float22bfloat162_rn(make_float2(mul_1_12.x, mul_1_12.y));
        dgate[6] = __as_u32(bf16x2_15);
        dup[6] = __as_u32(bf16x2_1_14);
        float2 x_13 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[7]));
        float2 u_13 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[7]));
        float2 g_13 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[7]));
        float e0_13 = approx_exp2(-x_13.x * 1.442695f);
        float e1_13 = approx_exp2(-x_13.y * 1.442695f);
        float d0_13 = 1.0f + e0_13;
        float d1_13 = 1.0f + e1_13;
        float r0_13 = approx_rcp(d0_13);
        float r1_13 = approx_rcp(d1_13);
        float _fma_171;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_171) : "f"(-d0_13), "f"(r0_13), "f"(1.0f));
        float fma_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_7) : "f"(r0_13), "f"(_fma_171), "f"(r0_13));
        float _fma_172;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_172) : "f"(-d1_13), "f"(r1_13), "f"(1.0f));
        float fma_1_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_6) : "f"(r1_13), "f"(_fma_172), "f"(r1_13));
        float t0_6 = fma_7;
        float t1_6 = fma_1_6;
        int _warp_vote_173 = __any_sync(0xFFFFFFFF, fminf(t0_6, t1_6) == 0.0f);
        if (_warp_vote_173 != 0) {
            float d4_1a_7;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_7) : "f"(d0_13), "f"(0.25f));
            float r4_1a_7 = approx_rcp(d4_1a_7);
            float _fma_175;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_175) : "f"(-d4_1a_7), "f"(r4_1a_7), "f"(1.0f));
            float _fma_174;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_174) : "f"(r4_1a_7), "f"(_fma_175), "f"(r4_1a_7));
            float big_1a_7;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_7) : "f"(mul_f32_rn_noftz(_fma_174, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_176;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_176) : "f"(big_1a_7), "f"(8.388608e+06f));
            float whole_1a_7;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_7) : "f"(_add_176), "f"(8.388608e+06f));
            float above_1a_7;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_7) : "f"(-d4_1a_7), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_7, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_178;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_178) : "f"(big_1a_7), "f"(whole_1a_7));
            float _abs_177;
            asm("abs.f32 %0, %1;" : "=f"(_abs_177) : "f"(_sub_178));
            float _add_179;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_179) : "f"(big_1a_7), "f"(0.5f));
            float _sub_180;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_180) : "f"(big_1a_7), "f"(0.5f));
            float rounded_1a_7 = ((_abs_177 == 0.5f) ? ((above_1a_7 > 0.0f) ? _add_179 : _sub_180) : whole_1a_7);
            t0_6 = ((d0_13 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_7, 7.888609e-31f), 1.7763568e-15f) : t0_6);
            float d4_1a_7_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_7_1) : "f"(d1_13), "f"(0.25f));
            float r4_1a_7_1 = approx_rcp(d4_1a_7_1);
            float _fma_182;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_182) : "f"(-d4_1a_7_1), "f"(r4_1a_7_1), "f"(1.0f));
            float _fma_181;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_181) : "f"(r4_1a_7_1), "f"(_fma_182), "f"(r4_1a_7_1));
            float big_1a_7_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_7_1) : "f"(mul_f32_rn_noftz(_fma_181, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_183;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_183) : "f"(big_1a_7_1), "f"(8.388608e+06f));
            float whole_1a_7_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_7_1) : "f"(_add_183), "f"(8.388608e+06f));
            float above_1a_7_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_7_1) : "f"(-d4_1a_7_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_7_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_185;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_185) : "f"(big_1a_7_1), "f"(whole_1a_7_1));
            float _abs_184;
            asm("abs.f32 %0, %1;" : "=f"(_abs_184) : "f"(_sub_185));
            float _add_186;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_186) : "f"(big_1a_7_1), "f"(0.5f));
            float _sub_187;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_187) : "f"(big_1a_7_1), "f"(0.5f));
            float rounded_1a_7_1 = ((_abs_184 == 0.5f) ? ((above_1a_7_1 > 0.0f) ? _add_186 : _sub_187) : whole_1a_7_1);
            t1_6 = ((d1_13 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_7_1, 7.888609e-31f), 1.7763568e-15f) : t1_6);
        }
        float2 s_13 = make_float2(max_noftz(t0_6, 0.0f), max_noftz(t1_6, 0.0f));
        float2 act_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_13) : "l"(*(const unsigned long long*)&x_13), "l"(*(const unsigned long long*)&s_13));
        float2 _f32x2_mul_a_188 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_13, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_13), s_13), g_13);
        float2 mul_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_188), "l"(*(const unsigned long long*)&u_13));
        float2 mul_1_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_13) : "l"(*(const unsigned long long*)&act_13), "l"(*(const unsigned long long*)&g_13));
        __nv_bfloat162 bf16x2_16 = __float22bfloat162_rn(make_float2(mul_14.x, mul_14.y));
        __nv_bfloat162 bf16x2_1_15 = __float22bfloat162_rn(make_float2(mul_1_13.x, mul_1_13.y));
        dgate[7] = __as_u32(bf16x2_16);
        dup[7] = __as_u32(bf16x2_1_15);
    }
    int scale_cols = K >> 6;
    int scale_col = bx * 2 + blk;
    int out_index = grow * (2 * K) + col;
    __nv_bfloat162 amax = __as_bf16x2(0);
    __nv_bfloat162 _hxa_190 = amax;
    __nv_bfloat162 _hxb_190 = __as_bf16x2(dgate[0]);
    unsigned int _hxr_190;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_190) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_190)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_190)));
    __nv_bfloat162 _max_189 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_190);
    amax = _max_189;
    __nv_bfloat162 _hxa_192 = amax;
    __nv_bfloat162 _hxb_192 = __as_bf16x2(dgate[1]);
    unsigned int _hxr_192;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_192) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_192)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_192)));
    __nv_bfloat162 _max_191 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_192);
    amax = _max_191;
    __nv_bfloat162 _hxa_194 = amax;
    __nv_bfloat162 _hxb_194 = __as_bf16x2(dgate[2]);
    unsigned int _hxr_194;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_194) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_194)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_194)));
    __nv_bfloat162 _max_193 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_194);
    amax = _max_193;
    __nv_bfloat162 _hxa_196 = amax;
    __nv_bfloat162 _hxb_196 = __as_bf16x2(dgate[3]);
    unsigned int _hxr_196;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_196) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_196)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_196)));
    __nv_bfloat162 _max_195 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_196);
    amax = _max_195;
    __nv_bfloat162 _hxa_198 = amax;
    __nv_bfloat162 _hxb_198 = __as_bf16x2(dgate[4]);
    unsigned int _hxr_198;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_198) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_198)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_198)));
    __nv_bfloat162 _max_197 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_198);
    amax = _max_197;
    __nv_bfloat162 _hxa_200 = amax;
    __nv_bfloat162 _hxb_200 = __as_bf16x2(dgate[5]);
    unsigned int _hxr_200;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_200) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_200)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_200)));
    __nv_bfloat162 _max_199 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_200);
    amax = _max_199;
    __nv_bfloat162 _hxa_202 = amax;
    __nv_bfloat162 _hxb_202 = __as_bf16x2(dgate[6]);
    unsigned int _hxr_202;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_202) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_202)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_202)));
    __nv_bfloat162 _max_201 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_202);
    amax = _max_201;
    __nv_bfloat162 _hxa_204 = amax;
    __nv_bfloat162 _hxb_204 = __as_bf16x2(dgate[7]);
    unsigned int _hxr_204;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_204) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_204)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_204)));
    __nv_bfloat162 _max_203 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_204);
    amax = _max_203;
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
    int scale_index = ((grow >> 7) * scale_cols + (scale_col >> 2)) * 512 + (grow & 31) * 16 + (grow >> 5 & 3) * 4 + (scale_col & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + scale_index) + (0)) = (unsigned char)(scale);
    }
    unsigned int words[4];
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_206;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_206) : "r"(bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_206) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_205;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_205) : "h"(e4m3x2_bf16x2), "h"(_cvt_satfinite_e4m3x2_bf16x2_206));
    words[0] = _mov_b32_pack_u16x2_205;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_208;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_208) : "r"(bf16x2_mul_1_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_208) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_207;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_207) : "h"(e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_208));
    words[1] = _mov_b32_pack_u16x2_207;
    uint32_t bf16x2_mul_4;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4) : "r"(dgate[4]), "r"(inv2));
    uint16_t e4m3x2_bf16x2_4;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_4) : "r"(bf16x2_mul_4));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_4);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_4) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_1;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_1) : "r"(dgate[5]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_210;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_210) : "r"(bf16x2_mul_1_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_210) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_209;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_209) : "h"(e4m3x2_bf16x2_4), "h"(_cvt_satfinite_e4m3x2_bf16x2_210));
    words[2] = _mov_b32_pack_u16x2_209;
    uint32_t bf16x2_mul_5;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5) : "r"(dgate[6]), "r"(inv2));
    uint16_t e4m3x2_bf16x2_5;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_5) : "r"(bf16x2_mul_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_5) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_2;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_2) : "r"(dgate[7]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_212;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_212) : "r"(bf16x2_mul_1_2));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_2);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_212) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_211;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_211) : "h"(e4m3x2_bf16x2_5), "h"(_cvt_satfinite_e4m3x2_bf16x2_212));
    words[3] = _mov_b32_pack_u16x2_211;
    reinterpret_cast<int4*>(row_output + out_index)[0] = reinterpret_cast<int4*>(words + 0)[0];
    __nv_bfloat162 amax_0 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_214 = amax_0;
    __nv_bfloat162 _hxb_214 = __as_bf16x2(dup[0]);
    unsigned int _hxr_214;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_214) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_214)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_214)));
    __nv_bfloat162 _max_213 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_214);
    amax_0 = _max_213;
    __nv_bfloat162 _hxa_216 = amax_0;
    __nv_bfloat162 _hxb_216 = __as_bf16x2(dup[1]);
    unsigned int _hxr_216;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_216) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_216)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_216)));
    __nv_bfloat162 _max_215 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_216);
    amax_0 = _max_215;
    __nv_bfloat162 _hxa_218 = amax_0;
    __nv_bfloat162 _hxb_218 = __as_bf16x2(dup[2]);
    unsigned int _hxr_218;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_218) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_218)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_218)));
    __nv_bfloat162 _max_217 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_218);
    amax_0 = _max_217;
    __nv_bfloat162 _hxa_220 = amax_0;
    __nv_bfloat162 _hxb_220 = __as_bf16x2(dup[3]);
    unsigned int _hxr_220;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_220) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_220)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_220)));
    __nv_bfloat162 _max_219 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_220);
    amax_0 = _max_219;
    __nv_bfloat162 _hxa_222 = amax_0;
    __nv_bfloat162 _hxb_222 = __as_bf16x2(dup[4]);
    unsigned int _hxr_222;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_222) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_222)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_222)));
    __nv_bfloat162 _max_221 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_222);
    amax_0 = _max_221;
    __nv_bfloat162 _hxa_224 = amax_0;
    __nv_bfloat162 _hxb_224 = __as_bf16x2(dup[5]);
    unsigned int _hxr_224;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_224) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_224)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_224)));
    __nv_bfloat162 _max_223 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_224);
    amax_0 = _max_223;
    __nv_bfloat162 _hxa_226 = amax_0;
    __nv_bfloat162 _hxb_226 = __as_bf16x2(dup[6]);
    unsigned int _hxr_226;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_226) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_226)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_226)));
    __nv_bfloat162 _max_225 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_226);
    amax_0 = _max_225;
    __nv_bfloat162 _hxa_228 = amax_0;
    __nv_bfloat162 _hxb_228 = __as_bf16x2(dup[7]);
    unsigned int _hxr_228;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_228) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_228)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_228)));
    __nv_bfloat162 _max_227 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_228);
    amax_0 = _max_227;
    unsigned int mag_0 = __as_u32(amax_0) & 2147450879;
    unsigned int peer_0 = __shfl_xor_sync(0xFFFFFFFF, mag_0, 1);
    __nv_bfloat162 both_0 = __hmax2_nan(__as_bf16x2(mag_0), __as_bf16x2(peer_0));
    unsigned int folded_0 = __as_u32(__hmax2_nan(both_0, __as_bf16x2(__as_u32(both_0) >> 16)));
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
    int scale_index_0 = ((grow >> 7) * scale_cols + (scale_col + (K >> 5) >> 2)) * 512 + (grow & 31) * 16 + (grow >> 5 & 3) * 4 + (scale_col + (K >> 5) & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + scale_index_0) + (0)) = (unsigned char)(scale_0);
    }
    unsigned int words_0[4];
    uint32_t bf16x2_mul_6;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6) : "r"(dup[0]), "r"(inv2_0));
    uint16_t e4m3x2_bf16x2_6;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_6) : "r"(bf16x2_mul_6));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_6);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_6) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_3;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_3) : "r"(dup[1]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_230;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_230) : "r"(bf16x2_mul_1_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_230) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_229;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_229) : "h"(e4m3x2_bf16x2_6), "h"(_cvt_satfinite_e4m3x2_bf16x2_230));
    words_0[0] = _mov_b32_pack_u16x2_229;
    uint32_t bf16x2_mul_7;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7) : "r"(dup[2]), "r"(inv2_0));
    uint16_t e4m3x2_bf16x2_7;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_7) : "r"(bf16x2_mul_7));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_7);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_7) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_4;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_4) : "r"(dup[3]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_232;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_232) : "r"(bf16x2_mul_1_4));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_4);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_232) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_231;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_231) : "h"(e4m3x2_bf16x2_7), "h"(_cvt_satfinite_e4m3x2_bf16x2_232));
    words_0[1] = _mov_b32_pack_u16x2_231;
    uint32_t bf16x2_mul_8;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_8) : "r"(dup[4]), "r"(inv2_0));
    uint16_t e4m3x2_bf16x2_8;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_8) : "r"(bf16x2_mul_8));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_8);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_8) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_5;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_5) : "r"(dup[5]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_234;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_234) : "r"(bf16x2_mul_1_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_234) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_233;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_233) : "h"(e4m3x2_bf16x2_8), "h"(_cvt_satfinite_e4m3x2_bf16x2_234));
    words_0[2] = _mov_b32_pack_u16x2_233;
    uint32_t bf16x2_mul_9;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_9) : "r"(dup[6]), "r"(inv2_0));
    uint16_t e4m3x2_bf16x2_9;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_9) : "r"(bf16x2_mul_9));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_9);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_9) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_6;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_6) : "r"(dup[7]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_236;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_236) : "r"(bf16x2_mul_1_6));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_6);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_236) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_235;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_235) : "h"(e4m3x2_bf16x2_9), "h"(_cvt_satfinite_e4m3x2_bf16x2_236));
    words_0[3] = _mov_b32_pack_u16x2_235;
    reinterpret_cast<int4*>(row_output + (out_index + K))[0] = reinterpret_cast<int4*>(words_0 + 0)[0];
}

} // extern "C"
