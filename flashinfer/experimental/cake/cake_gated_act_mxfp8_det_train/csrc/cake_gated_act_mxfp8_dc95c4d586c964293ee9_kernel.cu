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
kernel_cake_gated_act_mxfp8_dc95c4d586c964293ee9(__nv_bfloat16* __restrict__ gated_input, __nv_bfloat16* __restrict__ grad, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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
    int grow = by * 64 + row;
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
    int grow_b = grow + 32;
    int gate_index_0 = grow_b * (2 * K) + col;
    unsigned int vec_load_0[8];
    {
        uint32_t _iv_3_0;
        uint32_t _iv_3_1;
        uint32_t _iv_3_2;
        uint32_t _iv_3_3;
        uint32_t _iv_3_4;
        uint32_t _iv_3_5;
        uint32_t _iv_3_6;
        uint32_t _iv_3_7;
        asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_iv_3_0), "=r"(_iv_3_1), "=r"(_iv_3_2), "=r"(_iv_3_3), "=r"(_iv_3_4), "=r"(_iv_3_5), "=r"(_iv_3_6), "=r"(_iv_3_7) : "l"((const void*)(gated_input + gate_index_0 + (0))) : "memory");
        vec_load_0[0 + 0] = (uint32_t)_iv_3_0;
        vec_load_0[0 + 1] = (uint32_t)_iv_3_1;
        vec_load_0[0 + 2] = (uint32_t)_iv_3_2;
        vec_load_0[0 + 3] = (uint32_t)_iv_3_3;
        vec_load_0[0 + 4] = (uint32_t)_iv_3_4;
        vec_load_0[0 + 5] = (uint32_t)_iv_3_5;
        vec_load_0[0 + 6] = (uint32_t)_iv_3_6;
        vec_load_0[0 + 7] = (uint32_t)_iv_3_7;
    }
    unsigned int vec_load_1_0[8];
    {
        uint32_t _iv_4_0;
        uint32_t _iv_4_1;
        uint32_t _iv_4_2;
        uint32_t _iv_4_3;
        uint32_t _iv_4_4;
        uint32_t _iv_4_5;
        uint32_t _iv_4_6;
        uint32_t _iv_4_7;
        asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_iv_4_0), "=r"(_iv_4_1), "=r"(_iv_4_2), "=r"(_iv_4_3), "=r"(_iv_4_4), "=r"(_iv_4_5), "=r"(_iv_4_6), "=r"(_iv_4_7) : "l"((const void*)(gated_input + (gate_index_0 + K) + (0))) : "memory");
        vec_load_1_0[0 + 0] = (uint32_t)_iv_4_0;
        vec_load_1_0[0 + 1] = (uint32_t)_iv_4_1;
        vec_load_1_0[0 + 2] = (uint32_t)_iv_4_2;
        vec_load_1_0[0 + 3] = (uint32_t)_iv_4_3;
        vec_load_1_0[0 + 4] = (uint32_t)_iv_4_4;
        vec_load_1_0[0 + 5] = (uint32_t)_iv_4_5;
        vec_load_1_0[0 + 6] = (uint32_t)_iv_4_6;
        vec_load_1_0[0 + 7] = (uint32_t)_iv_4_7;
    }
    unsigned int vec_load_2_0[8];
    {
        uint32_t _iv_5_0;
        uint32_t _iv_5_1;
        uint32_t _iv_5_2;
        uint32_t _iv_5_3;
        uint32_t _iv_5_4;
        uint32_t _iv_5_5;
        uint32_t _iv_5_6;
        uint32_t _iv_5_7;
        asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_iv_5_0), "=r"(_iv_5_1), "=r"(_iv_5_2), "=r"(_iv_5_3), "=r"(_iv_5_4), "=r"(_iv_5_5), "=r"(_iv_5_6), "=r"(_iv_5_7) : "l"((const void*)(grad + (grow_b * K + col) + (0))) : "memory");
        vec_load_2_0[0 + 0] = (uint32_t)_iv_5_0;
        vec_load_2_0[0 + 1] = (uint32_t)_iv_5_1;
        vec_load_2_0[0 + 2] = (uint32_t)_iv_5_2;
        vec_load_2_0[0 + 3] = (uint32_t)_iv_5_3;
        vec_load_2_0[0 + 4] = (uint32_t)_iv_5_4;
        vec_load_2_0[0 + 5] = (uint32_t)_iv_5_5;
        vec_load_2_0[0 + 6] = (uint32_t)_iv_5_6;
        vec_load_2_0[0 + 7] = (uint32_t)_iv_5_7;
    }
    unsigned int dgate[8];
    unsigned int dup[8];
    unsigned int dgate_b[8];
    unsigned int dup_b[8];
    __nv_bfloat162 lowest = __as_bf16x2(vec_load[0]);
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[7]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_0[0]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_0[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_0[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_0[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_0[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_0[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_0[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_0[7]));
    float2 lowest_f32 = cake_bf16x2_to_f32x2(lowest);
    float _min_7;
    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_7) : "f"(lowest_f32.x), "f"(lowest_f32.y));
    int _warp_vote_6 = __all_sync(0xFFFFFFFF, _min_7 > -87.0f);
    if (_warp_vote_6 != 0) {
        float2 x = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[0]));
        float2 u = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[0]));
        float2 g = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[0]));
        float e0 = approx_exp2(-x.x * 1.442695f);
        float e1 = approx_exp2(-x.y * 1.442695f);
        float d0 = 1.0f + e0;
        float d1 = 1.0f + e1;
        float r0 = approx_rcp(d0);
        float r1 = approx_rcp(d1);
        float _fma_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(-d0), "f"(r0), "f"(1.0f));
        float _fma_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(r0), "f"(_fma_9), "f"(r0));
        float _fma_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(-d1), "f"(r1), "f"(1.0f));
        float _fma_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(r1), "f"(_fma_11), "f"(r1));
        float2 s = make_float2(_fma_8, _fma_10);
        float2 act;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act) : "l"(*(const unsigned long long*)&x), "l"(*(const unsigned long long*)&s));
        float2 _f32x2_mul_a_12 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s), s), g);
        float2 mul;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul) : "l"(*(const unsigned long long*)&_f32x2_mul_a_12), "l"(*(const unsigned long long*)&u));
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
        float _fma_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(-d0_0), "f"(r0_0), "f"(1.0f));
        float _fma_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(r0_0), "f"(_fma_14), "f"(r0_0));
        float _fma_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(-d1_0), "f"(r1_0), "f"(1.0f));
        float _fma_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(r1_0), "f"(_fma_16), "f"(r1_0));
        float2 s_0 = make_float2(_fma_13, _fma_15);
        float2 act_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_0) : "l"(*(const unsigned long long*)&x_0), "l"(*(const unsigned long long*)&s_0));
        float2 _f32x2_mul_a_17 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_0), s_0), g_0);
        float2 mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&u_0));
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
        float _fma_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(-d0_1), "f"(r0_1), "f"(1.0f));
        float _fma_18;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(r0_1), "f"(_fma_19), "f"(r0_1));
        float _fma_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(-d1_1), "f"(r1_1), "f"(1.0f));
        float _fma_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(r1_1), "f"(_fma_21), "f"(r1_1));
        float2 s_1 = make_float2(_fma_18, _fma_20);
        float2 act_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_1) : "l"(*(const unsigned long long*)&x_1), "l"(*(const unsigned long long*)&s_1));
        float2 _f32x2_mul_a_22 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1), s_1), g_1);
        float2 mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_22), "l"(*(const unsigned long long*)&u_1));
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
        float _fma_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(-d0_2), "f"(r0_2), "f"(1.0f));
        float _fma_23;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(r0_2), "f"(_fma_24), "f"(r0_2));
        float _fma_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(-d1_2), "f"(r1_2), "f"(1.0f));
        float _fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(r1_2), "f"(_fma_26), "f"(r1_2));
        float2 s_2 = make_float2(_fma_23, _fma_25);
        float2 act_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_2) : "l"(*(const unsigned long long*)&x_2), "l"(*(const unsigned long long*)&s_2));
        float2 _f32x2_mul_a_27 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_2), s_2), g_2);
        float2 mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_27), "l"(*(const unsigned long long*)&u_2));
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
        float _fma_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(-d0_3), "f"(r0_3), "f"(1.0f));
        float _fma_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(r0_3), "f"(_fma_29), "f"(r0_3));
        float _fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(-d1_3), "f"(r1_3), "f"(1.0f));
        float _fma_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(r1_3), "f"(_fma_31), "f"(r1_3));
        float2 s_3 = make_float2(_fma_28, _fma_30);
        float2 act_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_3) : "l"(*(const unsigned long long*)&x_3), "l"(*(const unsigned long long*)&s_3));
        float2 _f32x2_mul_a_32 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_3), s_3), g_3);
        float2 mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_32), "l"(*(const unsigned long long*)&u_3));
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
        float _fma_34;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(-d0_4), "f"(r0_4), "f"(1.0f));
        float _fma_33;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(r0_4), "f"(_fma_34), "f"(r0_4));
        float _fma_36;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(-d1_4), "f"(r1_4), "f"(1.0f));
        float _fma_35;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(r1_4), "f"(_fma_36), "f"(r1_4));
        float2 s_4 = make_float2(_fma_33, _fma_35);
        float2 act_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_4) : "l"(*(const unsigned long long*)&x_4), "l"(*(const unsigned long long*)&s_4));
        float2 _f32x2_mul_a_37 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_4), s_4), g_4);
        float2 mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_37), "l"(*(const unsigned long long*)&u_4));
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
        float _fma_39;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(-d0_5), "f"(r0_5), "f"(1.0f));
        float _fma_38;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(r0_5), "f"(_fma_39), "f"(r0_5));
        float _fma_41;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(-d1_5), "f"(r1_5), "f"(1.0f));
        float _fma_40;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(r1_5), "f"(_fma_41), "f"(r1_5));
        float2 s_5 = make_float2(_fma_38, _fma_40);
        float2 act_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_5) : "l"(*(const unsigned long long*)&x_5), "l"(*(const unsigned long long*)&s_5));
        float2 _f32x2_mul_a_42 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_5), s_5), g_5);
        float2 mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&u_5));
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
        float _fma_44;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(-d0_6), "f"(r0_6), "f"(1.0f));
        float _fma_43;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(r0_6), "f"(_fma_44), "f"(r0_6));
        float _fma_46;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(-d1_6), "f"(r1_6), "f"(1.0f));
        float _fma_45;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(r1_6), "f"(_fma_46), "f"(r1_6));
        float2 s_6 = make_float2(_fma_43, _fma_45);
        float2 act_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_6) : "l"(*(const unsigned long long*)&x_6), "l"(*(const unsigned long long*)&s_6));
        float2 _f32x2_mul_a_47 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_6), s_6), g_6);
        float2 mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_47), "l"(*(const unsigned long long*)&u_6));
        float2 mul_1_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_6) : "l"(*(const unsigned long long*)&act_6), "l"(*(const unsigned long long*)&g_6));
        __nv_bfloat162 bf16x2_8 = __float22bfloat162_rn(make_float2(mul_7.x, mul_7.y));
        __nv_bfloat162 bf16x2_1_7 = __float22bfloat162_rn(make_float2(mul_1_6.x, mul_1_6.y));
        dgate[7] = __as_u32(bf16x2_8);
        dup[7] = __as_u32(bf16x2_1_7);
        float2 x_7 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[0]));
        float2 u_7 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[0]));
        float2 g_7 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[0]));
        float e0_7 = approx_exp2(-x_7.x * 1.442695f);
        float e1_7 = approx_exp2(-x_7.y * 1.442695f);
        float d0_7 = 1.0f + e0_7;
        float d1_7 = 1.0f + e1_7;
        float r0_7 = approx_rcp(d0_7);
        float r1_7 = approx_rcp(d1_7);
        float _fma_49;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(-d0_7), "f"(r0_7), "f"(1.0f));
        float _fma_48;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(r0_7), "f"(_fma_49), "f"(r0_7));
        float _fma_51;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(-d1_7), "f"(r1_7), "f"(1.0f));
        float _fma_50;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(r1_7), "f"(_fma_51), "f"(r1_7));
        float2 s_7 = make_float2(_fma_48, _fma_50);
        float2 act_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_7) : "l"(*(const unsigned long long*)&x_7), "l"(*(const unsigned long long*)&s_7));
        float2 _f32x2_mul_a_52 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_7, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_7), s_7), g_7);
        float2 mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_52), "l"(*(const unsigned long long*)&u_7));
        float2 mul_1_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_7) : "l"(*(const unsigned long long*)&act_7), "l"(*(const unsigned long long*)&g_7));
        __nv_bfloat162 bf16x2_9 = __float22bfloat162_rn(make_float2(mul_8.x, mul_8.y));
        __nv_bfloat162 bf16x2_1_8 = __float22bfloat162_rn(make_float2(mul_1_7.x, mul_1_7.y));
        dgate_b[0] = __as_u32(bf16x2_9);
        dup_b[0] = __as_u32(bf16x2_1_8);
        float2 x_8 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[1]));
        float2 u_8 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[1]));
        float2 g_8 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[1]));
        float e0_8 = approx_exp2(-x_8.x * 1.442695f);
        float e1_8 = approx_exp2(-x_8.y * 1.442695f);
        float d0_8 = 1.0f + e0_8;
        float d1_8 = 1.0f + e1_8;
        float r0_8 = approx_rcp(d0_8);
        float r1_8 = approx_rcp(d1_8);
        float _fma_54;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(-d0_8), "f"(r0_8), "f"(1.0f));
        float _fma_53;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(r0_8), "f"(_fma_54), "f"(r0_8));
        float _fma_56;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(-d1_8), "f"(r1_8), "f"(1.0f));
        float _fma_55;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(r1_8), "f"(_fma_56), "f"(r1_8));
        float2 s_8 = make_float2(_fma_53, _fma_55);
        float2 act_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_8) : "l"(*(const unsigned long long*)&x_8), "l"(*(const unsigned long long*)&s_8));
        float2 _f32x2_mul_a_57 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_8, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_8), s_8), g_8);
        float2 mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_57), "l"(*(const unsigned long long*)&u_8));
        float2 mul_1_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_8) : "l"(*(const unsigned long long*)&act_8), "l"(*(const unsigned long long*)&g_8));
        __nv_bfloat162 bf16x2_10 = __float22bfloat162_rn(make_float2(mul_9.x, mul_9.y));
        __nv_bfloat162 bf16x2_1_9 = __float22bfloat162_rn(make_float2(mul_1_8.x, mul_1_8.y));
        dgate_b[1] = __as_u32(bf16x2_10);
        dup_b[1] = __as_u32(bf16x2_1_9);
        float2 x_9 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[2]));
        float2 u_9 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[2]));
        float2 g_9 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[2]));
        float e0_9 = approx_exp2(-x_9.x * 1.442695f);
        float e1_9 = approx_exp2(-x_9.y * 1.442695f);
        float d0_9 = 1.0f + e0_9;
        float d1_9 = 1.0f + e1_9;
        float r0_9 = approx_rcp(d0_9);
        float r1_9 = approx_rcp(d1_9);
        float _fma_59;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(-d0_9), "f"(r0_9), "f"(1.0f));
        float _fma_58;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(r0_9), "f"(_fma_59), "f"(r0_9));
        float _fma_61;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(-d1_9), "f"(r1_9), "f"(1.0f));
        float _fma_60;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(r1_9), "f"(_fma_61), "f"(r1_9));
        float2 s_9 = make_float2(_fma_58, _fma_60);
        float2 act_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_9) : "l"(*(const unsigned long long*)&x_9), "l"(*(const unsigned long long*)&s_9));
        float2 _f32x2_mul_a_62 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_9, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_9), s_9), g_9);
        float2 mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&u_9));
        float2 mul_1_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_9) : "l"(*(const unsigned long long*)&act_9), "l"(*(const unsigned long long*)&g_9));
        __nv_bfloat162 bf16x2_11 = __float22bfloat162_rn(make_float2(mul_10.x, mul_10.y));
        __nv_bfloat162 bf16x2_1_10 = __float22bfloat162_rn(make_float2(mul_1_9.x, mul_1_9.y));
        dgate_b[2] = __as_u32(bf16x2_11);
        dup_b[2] = __as_u32(bf16x2_1_10);
        float2 x_10 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[3]));
        float2 u_10 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[3]));
        float2 g_10 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[3]));
        float e0_10 = approx_exp2(-x_10.x * 1.442695f);
        float e1_10 = approx_exp2(-x_10.y * 1.442695f);
        float d0_10 = 1.0f + e0_10;
        float d1_10 = 1.0f + e1_10;
        float r0_10 = approx_rcp(d0_10);
        float r1_10 = approx_rcp(d1_10);
        float _fma_64;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(-d0_10), "f"(r0_10), "f"(1.0f));
        float _fma_63;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(r0_10), "f"(_fma_64), "f"(r0_10));
        float _fma_66;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(-d1_10), "f"(r1_10), "f"(1.0f));
        float _fma_65;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(r1_10), "f"(_fma_66), "f"(r1_10));
        float2 s_10 = make_float2(_fma_63, _fma_65);
        float2 act_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_10) : "l"(*(const unsigned long long*)&x_10), "l"(*(const unsigned long long*)&s_10));
        float2 _f32x2_mul_a_67 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_10, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_10), s_10), g_10);
        float2 mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_67), "l"(*(const unsigned long long*)&u_10));
        float2 mul_1_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_10) : "l"(*(const unsigned long long*)&act_10), "l"(*(const unsigned long long*)&g_10));
        __nv_bfloat162 bf16x2_12 = __float22bfloat162_rn(make_float2(mul_11.x, mul_11.y));
        __nv_bfloat162 bf16x2_1_11 = __float22bfloat162_rn(make_float2(mul_1_10.x, mul_1_10.y));
        dgate_b[3] = __as_u32(bf16x2_12);
        dup_b[3] = __as_u32(bf16x2_1_11);
        float2 x_11 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[4]));
        float2 u_11 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[4]));
        float2 g_11 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[4]));
        float e0_11 = approx_exp2(-x_11.x * 1.442695f);
        float e1_11 = approx_exp2(-x_11.y * 1.442695f);
        float d0_11 = 1.0f + e0_11;
        float d1_11 = 1.0f + e1_11;
        float r0_11 = approx_rcp(d0_11);
        float r1_11 = approx_rcp(d1_11);
        float _fma_69;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(-d0_11), "f"(r0_11), "f"(1.0f));
        float _fma_68;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(r0_11), "f"(_fma_69), "f"(r0_11));
        float _fma_71;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(-d1_11), "f"(r1_11), "f"(1.0f));
        float _fma_70;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(r1_11), "f"(_fma_71), "f"(r1_11));
        float2 s_11 = make_float2(_fma_68, _fma_70);
        float2 act_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_11) : "l"(*(const unsigned long long*)&x_11), "l"(*(const unsigned long long*)&s_11));
        float2 _f32x2_mul_a_72 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_11, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_11), s_11), g_11);
        float2 mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_72), "l"(*(const unsigned long long*)&u_11));
        float2 mul_1_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_11) : "l"(*(const unsigned long long*)&act_11), "l"(*(const unsigned long long*)&g_11));
        __nv_bfloat162 bf16x2_13 = __float22bfloat162_rn(make_float2(mul_12.x, mul_12.y));
        __nv_bfloat162 bf16x2_1_12 = __float22bfloat162_rn(make_float2(mul_1_11.x, mul_1_11.y));
        dgate_b[4] = __as_u32(bf16x2_13);
        dup_b[4] = __as_u32(bf16x2_1_12);
        float2 x_12 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[5]));
        float2 u_12 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[5]));
        float2 g_12 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[5]));
        float e0_12 = approx_exp2(-x_12.x * 1.442695f);
        float e1_12 = approx_exp2(-x_12.y * 1.442695f);
        float d0_12 = 1.0f + e0_12;
        float d1_12 = 1.0f + e1_12;
        float r0_12 = approx_rcp(d0_12);
        float r1_12 = approx_rcp(d1_12);
        float _fma_74;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(-d0_12), "f"(r0_12), "f"(1.0f));
        float _fma_73;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(r0_12), "f"(_fma_74), "f"(r0_12));
        float _fma_76;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(-d1_12), "f"(r1_12), "f"(1.0f));
        float _fma_75;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(r1_12), "f"(_fma_76), "f"(r1_12));
        float2 s_12 = make_float2(_fma_73, _fma_75);
        float2 act_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_12) : "l"(*(const unsigned long long*)&x_12), "l"(*(const unsigned long long*)&s_12));
        float2 _f32x2_mul_a_77 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_12, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_12), s_12), g_12);
        float2 mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_77), "l"(*(const unsigned long long*)&u_12));
        float2 mul_1_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_12) : "l"(*(const unsigned long long*)&act_12), "l"(*(const unsigned long long*)&g_12));
        __nv_bfloat162 bf16x2_14 = __float22bfloat162_rn(make_float2(mul_13.x, mul_13.y));
        __nv_bfloat162 bf16x2_1_13 = __float22bfloat162_rn(make_float2(mul_1_12.x, mul_1_12.y));
        dgate_b[5] = __as_u32(bf16x2_14);
        dup_b[5] = __as_u32(bf16x2_1_13);
        float2 x_13 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[6]));
        float2 u_13 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[6]));
        float2 g_13 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[6]));
        float e0_13 = approx_exp2(-x_13.x * 1.442695f);
        float e1_13 = approx_exp2(-x_13.y * 1.442695f);
        float d0_13 = 1.0f + e0_13;
        float d1_13 = 1.0f + e1_13;
        float r0_13 = approx_rcp(d0_13);
        float r1_13 = approx_rcp(d1_13);
        float _fma_79;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(-d0_13), "f"(r0_13), "f"(1.0f));
        float _fma_78;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_78) : "f"(r0_13), "f"(_fma_79), "f"(r0_13));
        float _fma_81;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(-d1_13), "f"(r1_13), "f"(1.0f));
        float _fma_80;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(r1_13), "f"(_fma_81), "f"(r1_13));
        float2 s_13 = make_float2(_fma_78, _fma_80);
        float2 act_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_13) : "l"(*(const unsigned long long*)&x_13), "l"(*(const unsigned long long*)&s_13));
        float2 _f32x2_mul_a_82 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_13, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_13), s_13), g_13);
        float2 mul_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_82), "l"(*(const unsigned long long*)&u_13));
        float2 mul_1_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_13) : "l"(*(const unsigned long long*)&act_13), "l"(*(const unsigned long long*)&g_13));
        __nv_bfloat162 bf16x2_15 = __float22bfloat162_rn(make_float2(mul_14.x, mul_14.y));
        __nv_bfloat162 bf16x2_1_14 = __float22bfloat162_rn(make_float2(mul_1_13.x, mul_1_13.y));
        dgate_b[6] = __as_u32(bf16x2_15);
        dup_b[6] = __as_u32(bf16x2_1_14);
        float2 x_14 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[7]));
        float2 u_14 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[7]));
        float2 g_14 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[7]));
        float e0_14 = approx_exp2(-x_14.x * 1.442695f);
        float e1_14 = approx_exp2(-x_14.y * 1.442695f);
        float d0_14 = 1.0f + e0_14;
        float d1_14 = 1.0f + e1_14;
        float r0_14 = approx_rcp(d0_14);
        float r1_14 = approx_rcp(d1_14);
        float _fma_84;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(-d0_14), "f"(r0_14), "f"(1.0f));
        float _fma_83;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(r0_14), "f"(_fma_84), "f"(r0_14));
        float _fma_86;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(-d1_14), "f"(r1_14), "f"(1.0f));
        float _fma_85;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(r1_14), "f"(_fma_86), "f"(r1_14));
        float2 s_14 = make_float2(_fma_83, _fma_85);
        float2 act_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_14) : "l"(*(const unsigned long long*)&x_14), "l"(*(const unsigned long long*)&s_14));
        float2 _f32x2_mul_a_87 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_14, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_14), s_14), g_14);
        float2 mul_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_87), "l"(*(const unsigned long long*)&u_14));
        float2 mul_1_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_14) : "l"(*(const unsigned long long*)&act_14), "l"(*(const unsigned long long*)&g_14));
        __nv_bfloat162 bf16x2_16 = __float22bfloat162_rn(make_float2(mul_15.x, mul_15.y));
        __nv_bfloat162 bf16x2_1_15 = __float22bfloat162_rn(make_float2(mul_1_14.x, mul_1_14.y));
        dgate_b[7] = __as_u32(bf16x2_16);
        dup_b[7] = __as_u32(bf16x2_1_15);
    } else {
        int band = 0;
        float2 x_15 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[0]));
        float2 u_15 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[0]));
        float2 g_15 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[0]));
        float e0_15 = approx_exp2(-x_15.x * 1.442695f);
        float e1_15 = approx_exp2(-x_15.y * 1.442695f);
        float d0_15 = 1.0f + e0_15;
        float d1_15 = 1.0f + e1_15;
        float r0_15 = approx_rcp(d0_15);
        float r1_15 = approx_rcp(d1_15);
        float _fma_88;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_88) : "f"(-d0_15), "f"(r0_15), "f"(1.0f));
        float fma;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma) : "f"(r0_15), "f"(_fma_88), "f"(r0_15));
        float _fma_89;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(-d1_15), "f"(r1_15), "f"(1.0f));
        float fma_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1) : "f"(r1_15), "f"(_fma_89), "f"(r1_15));
        float t0 = fma;
        float t1 = fma_1;
        int _warp_vote_90 = __any_sync(0xFFFFFFFF, fminf(t0, t1) == 0.0f);
        if (_warp_vote_90 != 0) {
            float d4_1a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a) : "f"(d0_15), "f"(0.25f));
            float r4_1a = approx_rcp(d4_1a);
            float _fma_92;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(-d4_1a), "f"(r4_1a), "f"(1.0f));
            float _fma_91;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(r4_1a), "f"(_fma_92), "f"(r4_1a));
            float big_1a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a) : "f"(mul_f32_rn_noftz(_fma_91, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_93;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(big_1a), "f"(8.388608e+06f));
            float whole_1a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a) : "f"(_add_93), "f"(8.388608e+06f));
            float above_1a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a) : "f"(-d4_1a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_95;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_95) : "f"(big_1a), "f"(whole_1a));
            float _abs_94;
            asm("abs.f32 %0, %1;" : "=f"(_abs_94) : "f"(_sub_95));
            float _add_96;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_96) : "f"(big_1a), "f"(0.5f));
            float _sub_97;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_97) : "f"(big_1a), "f"(0.5f));
            float rounded_1a = ((_abs_94 == 0.5f) ? ((above_1a > 0.0f) ? _add_96 : _sub_97) : whole_1a);
            t0 = ((d0_15 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a, 7.888609e-31f), 1.7763568e-15f) : t0);
            float d4_1a_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_0) : "f"(d1_15), "f"(0.25f));
            float r4_1a_0 = approx_rcp(d4_1a_0);
            float _fma_99;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(-d4_1a_0), "f"(r4_1a_0), "f"(1.0f));
            float _fma_98;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(r4_1a_0), "f"(_fma_99), "f"(r4_1a_0));
            float big_1a_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_0) : "f"(mul_f32_rn_noftz(_fma_98, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_100;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_100) : "f"(big_1a_0), "f"(8.388608e+06f));
            float whole_1a_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_0) : "f"(_add_100), "f"(8.388608e+06f));
            float above_1a_0;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_0) : "f"(-d4_1a_0), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_0, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_102;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_102) : "f"(big_1a_0), "f"(whole_1a_0));
            float _abs_101;
            asm("abs.f32 %0, %1;" : "=f"(_abs_101) : "f"(_sub_102));
            float _add_103;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_103) : "f"(big_1a_0), "f"(0.5f));
            float _sub_104;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_104) : "f"(big_1a_0), "f"(0.5f));
            float rounded_1a_0 = ((_abs_101 == 0.5f) ? ((above_1a_0 > 0.0f) ? _add_103 : _sub_104) : whole_1a_0);
            t1 = ((d1_15 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_0, 7.888609e-31f), 1.7763568e-15f) : t1);
        }
        float2 s_15 = make_float2(max_noftz(t0, 0.0f), max_noftz(t1, 0.0f));
        float2 act_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_15) : "l"(*(const unsigned long long*)&x_15), "l"(*(const unsigned long long*)&s_15));
        float2 _f32x2_mul_a_105 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_15, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_15), s_15), g_15);
        float2 mul_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_105), "l"(*(const unsigned long long*)&u_15));
        float2 mul_1_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_15) : "l"(*(const unsigned long long*)&act_15), "l"(*(const unsigned long long*)&g_15));
        __nv_bfloat162 bf16x2_17 = __float22bfloat162_rn(make_float2(mul_16.x, mul_16.y));
        __nv_bfloat162 bf16x2_1_16 = __float22bfloat162_rn(make_float2(mul_1_15.x, mul_1_15.y));
        dgate[0] = __as_u32(bf16x2_17);
        dup[0] = __as_u32(bf16x2_1_16);
        float2 x_15_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[1]));
        float2 u_15_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[1]));
        float2 g_15_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[1]));
        float e0_15_1 = approx_exp2(-x_15_1.x * 1.442695f);
        float e1_15_1 = approx_exp2(-x_15_1.y * 1.442695f);
        float d0_15_1 = 1.0f + e0_15_1;
        float d1_15_1 = 1.0f + e1_15_1;
        float r0_15_1 = approx_rcp(d0_15_1);
        float r1_15_1 = approx_rcp(d1_15_1);
        float _fma_106;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_106) : "f"(-d0_15_1), "f"(r0_15_1), "f"(1.0f));
        float fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_0) : "f"(r0_15_1), "f"(_fma_106), "f"(r0_15_1));
        float _fma_107;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(-d1_15_1), "f"(r1_15_1), "f"(1.0f));
        float fma_1_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_0) : "f"(r1_15_1), "f"(_fma_107), "f"(r1_15_1));
        float t0_0 = fma_0;
        float t1_0 = fma_1_0;
        int _warp_vote_108 = __any_sync(0xFFFFFFFF, fminf(t0_0, t1_0) == 0.0f);
        if (_warp_vote_108 != 0) {
            float d4_1a_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_1) : "f"(d0_15_1), "f"(0.25f));
            float r4_1a_1 = approx_rcp(d4_1a_1);
            float _fma_110;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_110) : "f"(-d4_1a_1), "f"(r4_1a_1), "f"(1.0f));
            float _fma_109;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_109) : "f"(r4_1a_1), "f"(_fma_110), "f"(r4_1a_1));
            float big_1a_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_1) : "f"(mul_f32_rn_noftz(_fma_109, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_111;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_111) : "f"(big_1a_1), "f"(8.388608e+06f));
            float whole_1a_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_1) : "f"(_add_111), "f"(8.388608e+06f));
            float above_1a_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_1) : "f"(-d4_1a_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_113;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_113) : "f"(big_1a_1), "f"(whole_1a_1));
            float _abs_112;
            asm("abs.f32 %0, %1;" : "=f"(_abs_112) : "f"(_sub_113));
            float _add_114;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_114) : "f"(big_1a_1), "f"(0.5f));
            float _sub_115;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_115) : "f"(big_1a_1), "f"(0.5f));
            float rounded_1a_1 = ((_abs_112 == 0.5f) ? ((above_1a_1 > 0.0f) ? _add_114 : _sub_115) : whole_1a_1);
            t0_0 = ((d0_15_1 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_1, 7.888609e-31f), 1.7763568e-15f) : t0_0);
            float d4_1a_1_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_1_1) : "f"(d1_15_1), "f"(0.25f));
            float r4_1a_1_1 = approx_rcp(d4_1a_1_1);
            float _fma_117;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_117) : "f"(-d4_1a_1_1), "f"(r4_1a_1_1), "f"(1.0f));
            float _fma_116;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_116) : "f"(r4_1a_1_1), "f"(_fma_117), "f"(r4_1a_1_1));
            float big_1a_1_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_1_1) : "f"(mul_f32_rn_noftz(_fma_116, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_118;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_118) : "f"(big_1a_1_1), "f"(8.388608e+06f));
            float whole_1a_1_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_1_1) : "f"(_add_118), "f"(8.388608e+06f));
            float above_1a_1_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_1_1) : "f"(-d4_1a_1_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_1_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_120;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_120) : "f"(big_1a_1_1), "f"(whole_1a_1_1));
            float _abs_119;
            asm("abs.f32 %0, %1;" : "=f"(_abs_119) : "f"(_sub_120));
            float _add_121;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_121) : "f"(big_1a_1_1), "f"(0.5f));
            float _sub_122;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_122) : "f"(big_1a_1_1), "f"(0.5f));
            float rounded_1a_1_1 = ((_abs_119 == 0.5f) ? ((above_1a_1_1 > 0.0f) ? _add_121 : _sub_122) : whole_1a_1_1);
            t1_0 = ((d1_15_1 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_1_1, 7.888609e-31f), 1.7763568e-15f) : t1_0);
        }
        float2 s_15_1 = make_float2(max_noftz(t0_0, 0.0f), max_noftz(t1_0, 0.0f));
        float2 act_15_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_15_1) : "l"(*(const unsigned long long*)&x_15_1), "l"(*(const unsigned long long*)&s_15_1));
        float2 _f32x2_mul_a_123 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_15_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_15_1), s_15_1), g_15_1);
        float2 mul_16_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_16_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_123), "l"(*(const unsigned long long*)&u_15_1));
        float2 mul_1_15_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_15_1) : "l"(*(const unsigned long long*)&act_15_1), "l"(*(const unsigned long long*)&g_15_1));
        __nv_bfloat162 bf16x2_18 = __float22bfloat162_rn(make_float2(mul_16_1.x, mul_16_1.y));
        __nv_bfloat162 bf16x2_1_17 = __float22bfloat162_rn(make_float2(mul_1_15_1.x, mul_1_15_1.y));
        dgate[1] = __as_u32(bf16x2_18);
        dup[1] = __as_u32(bf16x2_1_17);
        float2 x_16 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[2]));
        float2 u_16 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[2]));
        float2 g_16 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[2]));
        float e0_16 = approx_exp2(-x_16.x * 1.442695f);
        float e1_16 = approx_exp2(-x_16.y * 1.442695f);
        float d0_16 = 1.0f + e0_16;
        float d1_16 = 1.0f + e1_16;
        float r0_16 = approx_rcp(d0_16);
        float r1_16 = approx_rcp(d1_16);
        float _fma_124;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_124) : "f"(-d0_16), "f"(r0_16), "f"(1.0f));
        float fma_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_2) : "f"(r0_16), "f"(_fma_124), "f"(r0_16));
        float _fma_125;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(-d1_16), "f"(r1_16), "f"(1.0f));
        float fma_1_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_1) : "f"(r1_16), "f"(_fma_125), "f"(r1_16));
        float t0_1 = fma_2;
        float t1_1 = fma_1_1;
        int _warp_vote_126 = __any_sync(0xFFFFFFFF, fminf(t0_1, t1_1) == 0.0f);
        if (_warp_vote_126 != 0) {
            float d4_1a_2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_2) : "f"(d0_16), "f"(0.25f));
            float r4_1a_2 = approx_rcp(d4_1a_2);
            float _fma_128;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_128) : "f"(-d4_1a_2), "f"(r4_1a_2), "f"(1.0f));
            float _fma_127;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_127) : "f"(r4_1a_2), "f"(_fma_128), "f"(r4_1a_2));
            float big_1a_2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_2) : "f"(mul_f32_rn_noftz(_fma_127, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_129;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_129) : "f"(big_1a_2), "f"(8.388608e+06f));
            float whole_1a_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_2) : "f"(_add_129), "f"(8.388608e+06f));
            float above_1a_2;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_2) : "f"(-d4_1a_2), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_2, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_131;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_131) : "f"(big_1a_2), "f"(whole_1a_2));
            float _abs_130;
            asm("abs.f32 %0, %1;" : "=f"(_abs_130) : "f"(_sub_131));
            float _add_132;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_132) : "f"(big_1a_2), "f"(0.5f));
            float _sub_133;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_133) : "f"(big_1a_2), "f"(0.5f));
            float rounded_1a_2 = ((_abs_130 == 0.5f) ? ((above_1a_2 > 0.0f) ? _add_132 : _sub_133) : whole_1a_2);
            t0_1 = ((d0_16 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_2, 7.888609e-31f), 1.7763568e-15f) : t0_1);
            float d4_1a_2_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_2_1) : "f"(d1_16), "f"(0.25f));
            float r4_1a_2_1 = approx_rcp(d4_1a_2_1);
            float _fma_135;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_135) : "f"(-d4_1a_2_1), "f"(r4_1a_2_1), "f"(1.0f));
            float _fma_134;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_134) : "f"(r4_1a_2_1), "f"(_fma_135), "f"(r4_1a_2_1));
            float big_1a_2_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_2_1) : "f"(mul_f32_rn_noftz(_fma_134, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_136;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_136) : "f"(big_1a_2_1), "f"(8.388608e+06f));
            float whole_1a_2_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_2_1) : "f"(_add_136), "f"(8.388608e+06f));
            float above_1a_2_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_2_1) : "f"(-d4_1a_2_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_2_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_138;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_138) : "f"(big_1a_2_1), "f"(whole_1a_2_1));
            float _abs_137;
            asm("abs.f32 %0, %1;" : "=f"(_abs_137) : "f"(_sub_138));
            float _add_139;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_139) : "f"(big_1a_2_1), "f"(0.5f));
            float _sub_140;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_140) : "f"(big_1a_2_1), "f"(0.5f));
            float rounded_1a_2_1 = ((_abs_137 == 0.5f) ? ((above_1a_2_1 > 0.0f) ? _add_139 : _sub_140) : whole_1a_2_1);
            t1_1 = ((d1_16 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_2_1, 7.888609e-31f), 1.7763568e-15f) : t1_1);
        }
        float2 s_16 = make_float2(max_noftz(t0_1, 0.0f), max_noftz(t1_1, 0.0f));
        float2 act_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_16) : "l"(*(const unsigned long long*)&x_16), "l"(*(const unsigned long long*)&s_16));
        float2 _f32x2_mul_a_141 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_16, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_16), s_16), g_16);
        float2 mul_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_141), "l"(*(const unsigned long long*)&u_16));
        float2 mul_1_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_16) : "l"(*(const unsigned long long*)&act_16), "l"(*(const unsigned long long*)&g_16));
        __nv_bfloat162 bf16x2_19 = __float22bfloat162_rn(make_float2(mul_17.x, mul_17.y));
        __nv_bfloat162 bf16x2_1_18 = __float22bfloat162_rn(make_float2(mul_1_16.x, mul_1_16.y));
        dgate[2] = __as_u32(bf16x2_19);
        dup[2] = __as_u32(bf16x2_1_18);
        float2 x_17 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[3]));
        float2 u_17 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[3]));
        float2 g_17 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[3]));
        float e0_17 = approx_exp2(-x_17.x * 1.442695f);
        float e1_17 = approx_exp2(-x_17.y * 1.442695f);
        float d0_17 = 1.0f + e0_17;
        float d1_17 = 1.0f + e1_17;
        float r0_17 = approx_rcp(d0_17);
        float r1_17 = approx_rcp(d1_17);
        float _fma_142;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_142) : "f"(-d0_17), "f"(r0_17), "f"(1.0f));
        float fma_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_3) : "f"(r0_17), "f"(_fma_142), "f"(r0_17));
        float _fma_143;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_143) : "f"(-d1_17), "f"(r1_17), "f"(1.0f));
        float fma_1_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_2) : "f"(r1_17), "f"(_fma_143), "f"(r1_17));
        float t0_2 = fma_3;
        float t1_2 = fma_1_2;
        int _warp_vote_144 = __any_sync(0xFFFFFFFF, fminf(t0_2, t1_2) == 0.0f);
        if (_warp_vote_144 != 0) {
            float d4_1a_3;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_3) : "f"(d0_17), "f"(0.25f));
            float r4_1a_3 = approx_rcp(d4_1a_3);
            float _fma_146;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_146) : "f"(-d4_1a_3), "f"(r4_1a_3), "f"(1.0f));
            float _fma_145;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_145) : "f"(r4_1a_3), "f"(_fma_146), "f"(r4_1a_3));
            float big_1a_3;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_3) : "f"(mul_f32_rn_noftz(_fma_145, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_147;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_147) : "f"(big_1a_3), "f"(8.388608e+06f));
            float whole_1a_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_3) : "f"(_add_147), "f"(8.388608e+06f));
            float above_1a_3;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_3) : "f"(-d4_1a_3), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_3, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_149;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_149) : "f"(big_1a_3), "f"(whole_1a_3));
            float _abs_148;
            asm("abs.f32 %0, %1;" : "=f"(_abs_148) : "f"(_sub_149));
            float _add_150;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_150) : "f"(big_1a_3), "f"(0.5f));
            float _sub_151;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_151) : "f"(big_1a_3), "f"(0.5f));
            float rounded_1a_3 = ((_abs_148 == 0.5f) ? ((above_1a_3 > 0.0f) ? _add_150 : _sub_151) : whole_1a_3);
            t0_2 = ((d0_17 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_3, 7.888609e-31f), 1.7763568e-15f) : t0_2);
            float d4_1a_3_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_3_1) : "f"(d1_17), "f"(0.25f));
            float r4_1a_3_1 = approx_rcp(d4_1a_3_1);
            float _fma_153;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_153) : "f"(-d4_1a_3_1), "f"(r4_1a_3_1), "f"(1.0f));
            float _fma_152;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_152) : "f"(r4_1a_3_1), "f"(_fma_153), "f"(r4_1a_3_1));
            float big_1a_3_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_3_1) : "f"(mul_f32_rn_noftz(_fma_152, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_154;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_154) : "f"(big_1a_3_1), "f"(8.388608e+06f));
            float whole_1a_3_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_3_1) : "f"(_add_154), "f"(8.388608e+06f));
            float above_1a_3_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_3_1) : "f"(-d4_1a_3_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_3_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_156;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_156) : "f"(big_1a_3_1), "f"(whole_1a_3_1));
            float _abs_155;
            asm("abs.f32 %0, %1;" : "=f"(_abs_155) : "f"(_sub_156));
            float _add_157;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_157) : "f"(big_1a_3_1), "f"(0.5f));
            float _sub_158;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_158) : "f"(big_1a_3_1), "f"(0.5f));
            float rounded_1a_3_1 = ((_abs_155 == 0.5f) ? ((above_1a_3_1 > 0.0f) ? _add_157 : _sub_158) : whole_1a_3_1);
            t1_2 = ((d1_17 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_3_1, 7.888609e-31f), 1.7763568e-15f) : t1_2);
        }
        float2 s_17 = make_float2(max_noftz(t0_2, 0.0f), max_noftz(t1_2, 0.0f));
        float2 act_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_17) : "l"(*(const unsigned long long*)&x_17), "l"(*(const unsigned long long*)&s_17));
        float2 _f32x2_mul_a_159 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_17, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_17), s_17), g_17);
        float2 mul_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_159), "l"(*(const unsigned long long*)&u_17));
        float2 mul_1_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_17) : "l"(*(const unsigned long long*)&act_17), "l"(*(const unsigned long long*)&g_17));
        __nv_bfloat162 bf16x2_20 = __float22bfloat162_rn(make_float2(mul_18.x, mul_18.y));
        __nv_bfloat162 bf16x2_1_19 = __float22bfloat162_rn(make_float2(mul_1_17.x, mul_1_17.y));
        dgate[3] = __as_u32(bf16x2_20);
        dup[3] = __as_u32(bf16x2_1_19);
        float2 x_18 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[4]));
        float2 u_18 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[4]));
        float2 g_18 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[4]));
        float e0_18 = approx_exp2(-x_18.x * 1.442695f);
        float e1_18 = approx_exp2(-x_18.y * 1.442695f);
        float d0_18 = 1.0f + e0_18;
        float d1_18 = 1.0f + e1_18;
        float r0_18 = approx_rcp(d0_18);
        float r1_18 = approx_rcp(d1_18);
        float _fma_160;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_160) : "f"(-d0_18), "f"(r0_18), "f"(1.0f));
        float fma_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_4) : "f"(r0_18), "f"(_fma_160), "f"(r0_18));
        float _fma_161;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_161) : "f"(-d1_18), "f"(r1_18), "f"(1.0f));
        float fma_1_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_3) : "f"(r1_18), "f"(_fma_161), "f"(r1_18));
        float t0_3 = fma_4;
        float t1_3 = fma_1_3;
        int _warp_vote_162 = __any_sync(0xFFFFFFFF, fminf(t0_3, t1_3) == 0.0f);
        if (_warp_vote_162 != 0) {
            float d4_1a_4;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_4) : "f"(d0_18), "f"(0.25f));
            float r4_1a_4 = approx_rcp(d4_1a_4);
            float _fma_164;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_164) : "f"(-d4_1a_4), "f"(r4_1a_4), "f"(1.0f));
            float _fma_163;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_163) : "f"(r4_1a_4), "f"(_fma_164), "f"(r4_1a_4));
            float big_1a_4;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_4) : "f"(mul_f32_rn_noftz(_fma_163, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_165;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_165) : "f"(big_1a_4), "f"(8.388608e+06f));
            float whole_1a_4;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_4) : "f"(_add_165), "f"(8.388608e+06f));
            float above_1a_4;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_4) : "f"(-d4_1a_4), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_4, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_167;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_167) : "f"(big_1a_4), "f"(whole_1a_4));
            float _abs_166;
            asm("abs.f32 %0, %1;" : "=f"(_abs_166) : "f"(_sub_167));
            float _add_168;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_168) : "f"(big_1a_4), "f"(0.5f));
            float _sub_169;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_169) : "f"(big_1a_4), "f"(0.5f));
            float rounded_1a_4 = ((_abs_166 == 0.5f) ? ((above_1a_4 > 0.0f) ? _add_168 : _sub_169) : whole_1a_4);
            t0_3 = ((d0_18 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_4, 7.888609e-31f), 1.7763568e-15f) : t0_3);
            float d4_1a_4_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_4_1) : "f"(d1_18), "f"(0.25f));
            float r4_1a_4_1 = approx_rcp(d4_1a_4_1);
            float _fma_171;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_171) : "f"(-d4_1a_4_1), "f"(r4_1a_4_1), "f"(1.0f));
            float _fma_170;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_170) : "f"(r4_1a_4_1), "f"(_fma_171), "f"(r4_1a_4_1));
            float big_1a_4_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_4_1) : "f"(mul_f32_rn_noftz(_fma_170, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_172;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_172) : "f"(big_1a_4_1), "f"(8.388608e+06f));
            float whole_1a_4_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_4_1) : "f"(_add_172), "f"(8.388608e+06f));
            float above_1a_4_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_4_1) : "f"(-d4_1a_4_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_4_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_174;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_174) : "f"(big_1a_4_1), "f"(whole_1a_4_1));
            float _abs_173;
            asm("abs.f32 %0, %1;" : "=f"(_abs_173) : "f"(_sub_174));
            float _add_175;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_175) : "f"(big_1a_4_1), "f"(0.5f));
            float _sub_176;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_176) : "f"(big_1a_4_1), "f"(0.5f));
            float rounded_1a_4_1 = ((_abs_173 == 0.5f) ? ((above_1a_4_1 > 0.0f) ? _add_175 : _sub_176) : whole_1a_4_1);
            t1_3 = ((d1_18 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_4_1, 7.888609e-31f), 1.7763568e-15f) : t1_3);
        }
        float2 s_18 = make_float2(max_noftz(t0_3, 0.0f), max_noftz(t1_3, 0.0f));
        float2 act_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_18) : "l"(*(const unsigned long long*)&x_18), "l"(*(const unsigned long long*)&s_18));
        float2 _f32x2_mul_a_177 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_18, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_18), s_18), g_18);
        float2 mul_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_177), "l"(*(const unsigned long long*)&u_18));
        float2 mul_1_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_18) : "l"(*(const unsigned long long*)&act_18), "l"(*(const unsigned long long*)&g_18));
        __nv_bfloat162 bf16x2_21 = __float22bfloat162_rn(make_float2(mul_19.x, mul_19.y));
        __nv_bfloat162 bf16x2_1_20 = __float22bfloat162_rn(make_float2(mul_1_18.x, mul_1_18.y));
        dgate[4] = __as_u32(bf16x2_21);
        dup[4] = __as_u32(bf16x2_1_20);
        float2 x_19 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[5]));
        float2 u_19 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[5]));
        float2 g_19 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[5]));
        float e0_19 = approx_exp2(-x_19.x * 1.442695f);
        float e1_19 = approx_exp2(-x_19.y * 1.442695f);
        float d0_19 = 1.0f + e0_19;
        float d1_19 = 1.0f + e1_19;
        float r0_19 = approx_rcp(d0_19);
        float r1_19 = approx_rcp(d1_19);
        float _fma_178;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_178) : "f"(-d0_19), "f"(r0_19), "f"(1.0f));
        float fma_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_5) : "f"(r0_19), "f"(_fma_178), "f"(r0_19));
        float _fma_179;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_179) : "f"(-d1_19), "f"(r1_19), "f"(1.0f));
        float fma_1_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_4) : "f"(r1_19), "f"(_fma_179), "f"(r1_19));
        float t0_4 = fma_5;
        float t1_4 = fma_1_4;
        int _warp_vote_180 = __any_sync(0xFFFFFFFF, fminf(t0_4, t1_4) == 0.0f);
        if (_warp_vote_180 != 0) {
            float d4_1a_5;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_5) : "f"(d0_19), "f"(0.25f));
            float r4_1a_5 = approx_rcp(d4_1a_5);
            float _fma_182;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_182) : "f"(-d4_1a_5), "f"(r4_1a_5), "f"(1.0f));
            float _fma_181;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_181) : "f"(r4_1a_5), "f"(_fma_182), "f"(r4_1a_5));
            float big_1a_5;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_5) : "f"(mul_f32_rn_noftz(_fma_181, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_183;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_183) : "f"(big_1a_5), "f"(8.388608e+06f));
            float whole_1a_5;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_5) : "f"(_add_183), "f"(8.388608e+06f));
            float above_1a_5;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_5) : "f"(-d4_1a_5), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_5, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_185;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_185) : "f"(big_1a_5), "f"(whole_1a_5));
            float _abs_184;
            asm("abs.f32 %0, %1;" : "=f"(_abs_184) : "f"(_sub_185));
            float _add_186;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_186) : "f"(big_1a_5), "f"(0.5f));
            float _sub_187;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_187) : "f"(big_1a_5), "f"(0.5f));
            float rounded_1a_5 = ((_abs_184 == 0.5f) ? ((above_1a_5 > 0.0f) ? _add_186 : _sub_187) : whole_1a_5);
            t0_4 = ((d0_19 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_5, 7.888609e-31f), 1.7763568e-15f) : t0_4);
            float d4_1a_5_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_5_1) : "f"(d1_19), "f"(0.25f));
            float r4_1a_5_1 = approx_rcp(d4_1a_5_1);
            float _fma_189;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_189) : "f"(-d4_1a_5_1), "f"(r4_1a_5_1), "f"(1.0f));
            float _fma_188;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_188) : "f"(r4_1a_5_1), "f"(_fma_189), "f"(r4_1a_5_1));
            float big_1a_5_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_5_1) : "f"(mul_f32_rn_noftz(_fma_188, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_190;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_190) : "f"(big_1a_5_1), "f"(8.388608e+06f));
            float whole_1a_5_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_5_1) : "f"(_add_190), "f"(8.388608e+06f));
            float above_1a_5_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_5_1) : "f"(-d4_1a_5_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_5_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_192;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_192) : "f"(big_1a_5_1), "f"(whole_1a_5_1));
            float _abs_191;
            asm("abs.f32 %0, %1;" : "=f"(_abs_191) : "f"(_sub_192));
            float _add_193;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_193) : "f"(big_1a_5_1), "f"(0.5f));
            float _sub_194;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_194) : "f"(big_1a_5_1), "f"(0.5f));
            float rounded_1a_5_1 = ((_abs_191 == 0.5f) ? ((above_1a_5_1 > 0.0f) ? _add_193 : _sub_194) : whole_1a_5_1);
            t1_4 = ((d1_19 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_5_1, 7.888609e-31f), 1.7763568e-15f) : t1_4);
        }
        float2 s_19 = make_float2(max_noftz(t0_4, 0.0f), max_noftz(t1_4, 0.0f));
        float2 act_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_19) : "l"(*(const unsigned long long*)&x_19), "l"(*(const unsigned long long*)&s_19));
        float2 _f32x2_mul_a_195 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_19, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_19), s_19), g_19);
        float2 mul_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_195), "l"(*(const unsigned long long*)&u_19));
        float2 mul_1_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_19) : "l"(*(const unsigned long long*)&act_19), "l"(*(const unsigned long long*)&g_19));
        __nv_bfloat162 bf16x2_22 = __float22bfloat162_rn(make_float2(mul_20.x, mul_20.y));
        __nv_bfloat162 bf16x2_1_21 = __float22bfloat162_rn(make_float2(mul_1_19.x, mul_1_19.y));
        dgate[5] = __as_u32(bf16x2_22);
        dup[5] = __as_u32(bf16x2_1_21);
        float2 x_20 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[6]));
        float2 u_20 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[6]));
        float2 g_20 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[6]));
        float e0_20 = approx_exp2(-x_20.x * 1.442695f);
        float e1_20 = approx_exp2(-x_20.y * 1.442695f);
        float d0_20 = 1.0f + e0_20;
        float d1_20 = 1.0f + e1_20;
        float r0_20 = approx_rcp(d0_20);
        float r1_20 = approx_rcp(d1_20);
        float _fma_196;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_196) : "f"(-d0_20), "f"(r0_20), "f"(1.0f));
        float fma_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_6) : "f"(r0_20), "f"(_fma_196), "f"(r0_20));
        float _fma_197;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_197) : "f"(-d1_20), "f"(r1_20), "f"(1.0f));
        float fma_1_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_5) : "f"(r1_20), "f"(_fma_197), "f"(r1_20));
        float t0_5 = fma_6;
        float t1_5 = fma_1_5;
        int _warp_vote_198 = __any_sync(0xFFFFFFFF, fminf(t0_5, t1_5) == 0.0f);
        if (_warp_vote_198 != 0) {
            float d4_1a_6;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_6) : "f"(d0_20), "f"(0.25f));
            float r4_1a_6 = approx_rcp(d4_1a_6);
            float _fma_200;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_200) : "f"(-d4_1a_6), "f"(r4_1a_6), "f"(1.0f));
            float _fma_199;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_199) : "f"(r4_1a_6), "f"(_fma_200), "f"(r4_1a_6));
            float big_1a_6;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_6) : "f"(mul_f32_rn_noftz(_fma_199, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_201;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_201) : "f"(big_1a_6), "f"(8.388608e+06f));
            float whole_1a_6;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_6) : "f"(_add_201), "f"(8.388608e+06f));
            float above_1a_6;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_6) : "f"(-d4_1a_6), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_6, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_203;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_203) : "f"(big_1a_6), "f"(whole_1a_6));
            float _abs_202;
            asm("abs.f32 %0, %1;" : "=f"(_abs_202) : "f"(_sub_203));
            float _add_204;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_204) : "f"(big_1a_6), "f"(0.5f));
            float _sub_205;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_205) : "f"(big_1a_6), "f"(0.5f));
            float rounded_1a_6 = ((_abs_202 == 0.5f) ? ((above_1a_6 > 0.0f) ? _add_204 : _sub_205) : whole_1a_6);
            t0_5 = ((d0_20 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_6, 7.888609e-31f), 1.7763568e-15f) : t0_5);
            float d4_1a_6_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_6_1) : "f"(d1_20), "f"(0.25f));
            float r4_1a_6_1 = approx_rcp(d4_1a_6_1);
            float _fma_207;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_207) : "f"(-d4_1a_6_1), "f"(r4_1a_6_1), "f"(1.0f));
            float _fma_206;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_206) : "f"(r4_1a_6_1), "f"(_fma_207), "f"(r4_1a_6_1));
            float big_1a_6_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_6_1) : "f"(mul_f32_rn_noftz(_fma_206, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_208;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_208) : "f"(big_1a_6_1), "f"(8.388608e+06f));
            float whole_1a_6_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_6_1) : "f"(_add_208), "f"(8.388608e+06f));
            float above_1a_6_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_6_1) : "f"(-d4_1a_6_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_6_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_210;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_210) : "f"(big_1a_6_1), "f"(whole_1a_6_1));
            float _abs_209;
            asm("abs.f32 %0, %1;" : "=f"(_abs_209) : "f"(_sub_210));
            float _add_211;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_211) : "f"(big_1a_6_1), "f"(0.5f));
            float _sub_212;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_212) : "f"(big_1a_6_1), "f"(0.5f));
            float rounded_1a_6_1 = ((_abs_209 == 0.5f) ? ((above_1a_6_1 > 0.0f) ? _add_211 : _sub_212) : whole_1a_6_1);
            t1_5 = ((d1_20 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_6_1, 7.888609e-31f), 1.7763568e-15f) : t1_5);
        }
        float2 s_20 = make_float2(max_noftz(t0_5, 0.0f), max_noftz(t1_5, 0.0f));
        float2 act_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_20) : "l"(*(const unsigned long long*)&x_20), "l"(*(const unsigned long long*)&s_20));
        float2 _f32x2_mul_a_213 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_20, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_20), s_20), g_20);
        float2 mul_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_213), "l"(*(const unsigned long long*)&u_20));
        float2 mul_1_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_20) : "l"(*(const unsigned long long*)&act_20), "l"(*(const unsigned long long*)&g_20));
        __nv_bfloat162 bf16x2_23 = __float22bfloat162_rn(make_float2(mul_21.x, mul_21.y));
        __nv_bfloat162 bf16x2_1_22 = __float22bfloat162_rn(make_float2(mul_1_20.x, mul_1_20.y));
        dgate[6] = __as_u32(bf16x2_23);
        dup[6] = __as_u32(bf16x2_1_22);
        float2 x_21 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[7]));
        float2 u_21 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[7]));
        float2 g_21 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[7]));
        float e0_21 = approx_exp2(-x_21.x * 1.442695f);
        float e1_21 = approx_exp2(-x_21.y * 1.442695f);
        float d0_21 = 1.0f + e0_21;
        float d1_21 = 1.0f + e1_21;
        float r0_21 = approx_rcp(d0_21);
        float r1_21 = approx_rcp(d1_21);
        float _fma_214;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_214) : "f"(-d0_21), "f"(r0_21), "f"(1.0f));
        float fma_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_7) : "f"(r0_21), "f"(_fma_214), "f"(r0_21));
        float _fma_215;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_215) : "f"(-d1_21), "f"(r1_21), "f"(1.0f));
        float fma_1_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_6) : "f"(r1_21), "f"(_fma_215), "f"(r1_21));
        float t0_6 = fma_7;
        float t1_6 = fma_1_6;
        int _warp_vote_216 = __any_sync(0xFFFFFFFF, fminf(t0_6, t1_6) == 0.0f);
        if (_warp_vote_216 != 0) {
            float d4_1a_7;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_7) : "f"(d0_21), "f"(0.25f));
            float r4_1a_7 = approx_rcp(d4_1a_7);
            float _fma_218;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_218) : "f"(-d4_1a_7), "f"(r4_1a_7), "f"(1.0f));
            float _fma_217;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_217) : "f"(r4_1a_7), "f"(_fma_218), "f"(r4_1a_7));
            float big_1a_7;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_7) : "f"(mul_f32_rn_noftz(_fma_217, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_219;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_219) : "f"(big_1a_7), "f"(8.388608e+06f));
            float whole_1a_7;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_7) : "f"(_add_219), "f"(8.388608e+06f));
            float above_1a_7;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_7) : "f"(-d4_1a_7), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_7, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_221;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_221) : "f"(big_1a_7), "f"(whole_1a_7));
            float _abs_220;
            asm("abs.f32 %0, %1;" : "=f"(_abs_220) : "f"(_sub_221));
            float _add_222;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_222) : "f"(big_1a_7), "f"(0.5f));
            float _sub_223;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_223) : "f"(big_1a_7), "f"(0.5f));
            float rounded_1a_7 = ((_abs_220 == 0.5f) ? ((above_1a_7 > 0.0f) ? _add_222 : _sub_223) : whole_1a_7);
            t0_6 = ((d0_21 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_7, 7.888609e-31f), 1.7763568e-15f) : t0_6);
            float d4_1a_7_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_7_1) : "f"(d1_21), "f"(0.25f));
            float r4_1a_7_1 = approx_rcp(d4_1a_7_1);
            float _fma_225;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_225) : "f"(-d4_1a_7_1), "f"(r4_1a_7_1), "f"(1.0f));
            float _fma_224;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_224) : "f"(r4_1a_7_1), "f"(_fma_225), "f"(r4_1a_7_1));
            float big_1a_7_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_7_1) : "f"(mul_f32_rn_noftz(_fma_224, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_226;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_226) : "f"(big_1a_7_1), "f"(8.388608e+06f));
            float whole_1a_7_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_7_1) : "f"(_add_226), "f"(8.388608e+06f));
            float above_1a_7_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_7_1) : "f"(-d4_1a_7_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_7_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_228;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_228) : "f"(big_1a_7_1), "f"(whole_1a_7_1));
            float _abs_227;
            asm("abs.f32 %0, %1;" : "=f"(_abs_227) : "f"(_sub_228));
            float _add_229;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_229) : "f"(big_1a_7_1), "f"(0.5f));
            float _sub_230;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_230) : "f"(big_1a_7_1), "f"(0.5f));
            float rounded_1a_7_1 = ((_abs_227 == 0.5f) ? ((above_1a_7_1 > 0.0f) ? _add_229 : _sub_230) : whole_1a_7_1);
            t1_6 = ((d1_21 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_7_1, 7.888609e-31f), 1.7763568e-15f) : t1_6);
        }
        float2 s_21 = make_float2(max_noftz(t0_6, 0.0f), max_noftz(t1_6, 0.0f));
        float2 act_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_21) : "l"(*(const unsigned long long*)&x_21), "l"(*(const unsigned long long*)&s_21));
        float2 _f32x2_mul_a_231 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_21, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_21), s_21), g_21);
        float2 mul_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_231), "l"(*(const unsigned long long*)&u_21));
        float2 mul_1_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_21) : "l"(*(const unsigned long long*)&act_21), "l"(*(const unsigned long long*)&g_21));
        __nv_bfloat162 bf16x2_24 = __float22bfloat162_rn(make_float2(mul_22.x, mul_22.y));
        __nv_bfloat162 bf16x2_1_23 = __float22bfloat162_rn(make_float2(mul_1_21.x, mul_1_21.y));
        dgate[7] = __as_u32(bf16x2_24);
        dup[7] = __as_u32(bf16x2_1_23);
        int band_b = 0;
        float2 x_22 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[0]));
        float2 u_22 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[0]));
        float2 g_22 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[0]));
        float e0_22 = approx_exp2(-x_22.x * 1.442695f);
        float e1_22 = approx_exp2(-x_22.y * 1.442695f);
        float d0_22 = 1.0f + e0_22;
        float d1_22 = 1.0f + e1_22;
        float r0_22 = approx_rcp(d0_22);
        float r1_22 = approx_rcp(d1_22);
        float _fma_232;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_232) : "f"(-d0_22), "f"(r0_22), "f"(1.0f));
        float fma_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_8) : "f"(r0_22), "f"(_fma_232), "f"(r0_22));
        float _fma_233;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_233) : "f"(-d1_22), "f"(r1_22), "f"(1.0f));
        float fma_1_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_7) : "f"(r1_22), "f"(_fma_233), "f"(r1_22));
        float t0_7 = fma_8;
        float t1_7 = fma_1_7;
        int _warp_vote_234 = __any_sync(0xFFFFFFFF, fminf(t0_7, t1_7) == 0.0f);
        if (_warp_vote_234 != 0) {
            float d4_1a_8;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_8) : "f"(d0_22), "f"(0.25f));
            float r4_1a_8 = approx_rcp(d4_1a_8);
            float _fma_236;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_236) : "f"(-d4_1a_8), "f"(r4_1a_8), "f"(1.0f));
            float _fma_235;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_235) : "f"(r4_1a_8), "f"(_fma_236), "f"(r4_1a_8));
            float big_1a_8;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_8) : "f"(mul_f32_rn_noftz(_fma_235, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_237;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_237) : "f"(big_1a_8), "f"(8.388608e+06f));
            float whole_1a_8;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_8) : "f"(_add_237), "f"(8.388608e+06f));
            float above_1a_8;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_8) : "f"(-d4_1a_8), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_8, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_239;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_239) : "f"(big_1a_8), "f"(whole_1a_8));
            float _abs_238;
            asm("abs.f32 %0, %1;" : "=f"(_abs_238) : "f"(_sub_239));
            float _add_240;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_240) : "f"(big_1a_8), "f"(0.5f));
            float _sub_241;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_241) : "f"(big_1a_8), "f"(0.5f));
            float rounded_1a_8 = ((_abs_238 == 0.5f) ? ((above_1a_8 > 0.0f) ? _add_240 : _sub_241) : whole_1a_8);
            t0_7 = ((d0_22 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_8, 7.888609e-31f), 1.7763568e-15f) : t0_7);
            float d4_1a_8_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_8_1) : "f"(d1_22), "f"(0.25f));
            float r4_1a_8_1 = approx_rcp(d4_1a_8_1);
            float _fma_243;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_243) : "f"(-d4_1a_8_1), "f"(r4_1a_8_1), "f"(1.0f));
            float _fma_242;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_242) : "f"(r4_1a_8_1), "f"(_fma_243), "f"(r4_1a_8_1));
            float big_1a_8_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_8_1) : "f"(mul_f32_rn_noftz(_fma_242, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_244;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_244) : "f"(big_1a_8_1), "f"(8.388608e+06f));
            float whole_1a_8_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_8_1) : "f"(_add_244), "f"(8.388608e+06f));
            float above_1a_8_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_8_1) : "f"(-d4_1a_8_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_8_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_246;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_246) : "f"(big_1a_8_1), "f"(whole_1a_8_1));
            float _abs_245;
            asm("abs.f32 %0, %1;" : "=f"(_abs_245) : "f"(_sub_246));
            float _add_247;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_247) : "f"(big_1a_8_1), "f"(0.5f));
            float _sub_248;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_248) : "f"(big_1a_8_1), "f"(0.5f));
            float rounded_1a_8_1 = ((_abs_245 == 0.5f) ? ((above_1a_8_1 > 0.0f) ? _add_247 : _sub_248) : whole_1a_8_1);
            t1_7 = ((d1_22 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_8_1, 7.888609e-31f), 1.7763568e-15f) : t1_7);
        }
        float2 s_22 = make_float2(max_noftz(t0_7, 0.0f), max_noftz(t1_7, 0.0f));
        float2 act_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_22) : "l"(*(const unsigned long long*)&x_22), "l"(*(const unsigned long long*)&s_22));
        float2 _f32x2_mul_a_249 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_22, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_22), s_22), g_22);
        float2 mul_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_249), "l"(*(const unsigned long long*)&u_22));
        float2 mul_1_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_22) : "l"(*(const unsigned long long*)&act_22), "l"(*(const unsigned long long*)&g_22));
        __nv_bfloat162 bf16x2_25 = __float22bfloat162_rn(make_float2(mul_23.x, mul_23.y));
        __nv_bfloat162 bf16x2_1_24 = __float22bfloat162_rn(make_float2(mul_1_22.x, mul_1_22.y));
        dgate_b[0] = __as_u32(bf16x2_25);
        dup_b[0] = __as_u32(bf16x2_1_24);
        float2 x_23 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[1]));
        float2 u_23 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[1]));
        float2 g_23 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[1]));
        float e0_23 = approx_exp2(-x_23.x * 1.442695f);
        float e1_23 = approx_exp2(-x_23.y * 1.442695f);
        float d0_23 = 1.0f + e0_23;
        float d1_23 = 1.0f + e1_23;
        float r0_23 = approx_rcp(d0_23);
        float r1_23 = approx_rcp(d1_23);
        float _fma_250;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_250) : "f"(-d0_23), "f"(r0_23), "f"(1.0f));
        float fma_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_9) : "f"(r0_23), "f"(_fma_250), "f"(r0_23));
        float _fma_251;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_251) : "f"(-d1_23), "f"(r1_23), "f"(1.0f));
        float fma_1_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_8) : "f"(r1_23), "f"(_fma_251), "f"(r1_23));
        float t0_8 = fma_9;
        float t1_8 = fma_1_8;
        int _warp_vote_252 = __any_sync(0xFFFFFFFF, fminf(t0_8, t1_8) == 0.0f);
        if (_warp_vote_252 != 0) {
            float d4_1a_9;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_9) : "f"(d0_23), "f"(0.25f));
            float r4_1a_9 = approx_rcp(d4_1a_9);
            float _fma_254;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_254) : "f"(-d4_1a_9), "f"(r4_1a_9), "f"(1.0f));
            float _fma_253;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_253) : "f"(r4_1a_9), "f"(_fma_254), "f"(r4_1a_9));
            float big_1a_9;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_9) : "f"(mul_f32_rn_noftz(_fma_253, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_255;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_255) : "f"(big_1a_9), "f"(8.388608e+06f));
            float whole_1a_9;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_9) : "f"(_add_255), "f"(8.388608e+06f));
            float above_1a_9;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_9) : "f"(-d4_1a_9), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_9, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_257;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_257) : "f"(big_1a_9), "f"(whole_1a_9));
            float _abs_256;
            asm("abs.f32 %0, %1;" : "=f"(_abs_256) : "f"(_sub_257));
            float _add_258;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_258) : "f"(big_1a_9), "f"(0.5f));
            float _sub_259;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_259) : "f"(big_1a_9), "f"(0.5f));
            float rounded_1a_9 = ((_abs_256 == 0.5f) ? ((above_1a_9 > 0.0f) ? _add_258 : _sub_259) : whole_1a_9);
            t0_8 = ((d0_23 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_9, 7.888609e-31f), 1.7763568e-15f) : t0_8);
            float d4_1a_9_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_9_1) : "f"(d1_23), "f"(0.25f));
            float r4_1a_9_1 = approx_rcp(d4_1a_9_1);
            float _fma_261;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_261) : "f"(-d4_1a_9_1), "f"(r4_1a_9_1), "f"(1.0f));
            float _fma_260;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_260) : "f"(r4_1a_9_1), "f"(_fma_261), "f"(r4_1a_9_1));
            float big_1a_9_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_9_1) : "f"(mul_f32_rn_noftz(_fma_260, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_262;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_262) : "f"(big_1a_9_1), "f"(8.388608e+06f));
            float whole_1a_9_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_9_1) : "f"(_add_262), "f"(8.388608e+06f));
            float above_1a_9_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_9_1) : "f"(-d4_1a_9_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_9_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_264;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_264) : "f"(big_1a_9_1), "f"(whole_1a_9_1));
            float _abs_263;
            asm("abs.f32 %0, %1;" : "=f"(_abs_263) : "f"(_sub_264));
            float _add_265;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_265) : "f"(big_1a_9_1), "f"(0.5f));
            float _sub_266;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_266) : "f"(big_1a_9_1), "f"(0.5f));
            float rounded_1a_9_1 = ((_abs_263 == 0.5f) ? ((above_1a_9_1 > 0.0f) ? _add_265 : _sub_266) : whole_1a_9_1);
            t1_8 = ((d1_23 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_9_1, 7.888609e-31f), 1.7763568e-15f) : t1_8);
        }
        float2 s_23 = make_float2(max_noftz(t0_8, 0.0f), max_noftz(t1_8, 0.0f));
        float2 act_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_23) : "l"(*(const unsigned long long*)&x_23), "l"(*(const unsigned long long*)&s_23));
        float2 _f32x2_mul_a_267 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_23, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_23), s_23), g_23);
        float2 mul_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_267), "l"(*(const unsigned long long*)&u_23));
        float2 mul_1_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_23) : "l"(*(const unsigned long long*)&act_23), "l"(*(const unsigned long long*)&g_23));
        __nv_bfloat162 bf16x2_26 = __float22bfloat162_rn(make_float2(mul_24.x, mul_24.y));
        __nv_bfloat162 bf16x2_1_25 = __float22bfloat162_rn(make_float2(mul_1_23.x, mul_1_23.y));
        dgate_b[1] = __as_u32(bf16x2_26);
        dup_b[1] = __as_u32(bf16x2_1_25);
        float2 x_24 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[2]));
        float2 u_24 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[2]));
        float2 g_24 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[2]));
        float e0_24 = approx_exp2(-x_24.x * 1.442695f);
        float e1_24 = approx_exp2(-x_24.y * 1.442695f);
        float d0_24 = 1.0f + e0_24;
        float d1_24 = 1.0f + e1_24;
        float r0_24 = approx_rcp(d0_24);
        float r1_24 = approx_rcp(d1_24);
        float _fma_268;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_268) : "f"(-d0_24), "f"(r0_24), "f"(1.0f));
        float fma_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_10) : "f"(r0_24), "f"(_fma_268), "f"(r0_24));
        float _fma_269;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_269) : "f"(-d1_24), "f"(r1_24), "f"(1.0f));
        float fma_1_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_9) : "f"(r1_24), "f"(_fma_269), "f"(r1_24));
        float t0_9 = fma_10;
        float t1_9 = fma_1_9;
        int _warp_vote_270 = __any_sync(0xFFFFFFFF, fminf(t0_9, t1_9) == 0.0f);
        if (_warp_vote_270 != 0) {
            float d4_1a_10;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_10) : "f"(d0_24), "f"(0.25f));
            float r4_1a_10 = approx_rcp(d4_1a_10);
            float _fma_272;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_272) : "f"(-d4_1a_10), "f"(r4_1a_10), "f"(1.0f));
            float _fma_271;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_271) : "f"(r4_1a_10), "f"(_fma_272), "f"(r4_1a_10));
            float big_1a_10;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_10) : "f"(mul_f32_rn_noftz(_fma_271, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_273;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_273) : "f"(big_1a_10), "f"(8.388608e+06f));
            float whole_1a_10;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_10) : "f"(_add_273), "f"(8.388608e+06f));
            float above_1a_10;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_10) : "f"(-d4_1a_10), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_10, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_275;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_275) : "f"(big_1a_10), "f"(whole_1a_10));
            float _abs_274;
            asm("abs.f32 %0, %1;" : "=f"(_abs_274) : "f"(_sub_275));
            float _add_276;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_276) : "f"(big_1a_10), "f"(0.5f));
            float _sub_277;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_277) : "f"(big_1a_10), "f"(0.5f));
            float rounded_1a_10 = ((_abs_274 == 0.5f) ? ((above_1a_10 > 0.0f) ? _add_276 : _sub_277) : whole_1a_10);
            t0_9 = ((d0_24 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_10, 7.888609e-31f), 1.7763568e-15f) : t0_9);
            float d4_1a_10_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_10_1) : "f"(d1_24), "f"(0.25f));
            float r4_1a_10_1 = approx_rcp(d4_1a_10_1);
            float _fma_279;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_279) : "f"(-d4_1a_10_1), "f"(r4_1a_10_1), "f"(1.0f));
            float _fma_278;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_278) : "f"(r4_1a_10_1), "f"(_fma_279), "f"(r4_1a_10_1));
            float big_1a_10_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_10_1) : "f"(mul_f32_rn_noftz(_fma_278, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_280;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_280) : "f"(big_1a_10_1), "f"(8.388608e+06f));
            float whole_1a_10_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_10_1) : "f"(_add_280), "f"(8.388608e+06f));
            float above_1a_10_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_10_1) : "f"(-d4_1a_10_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_10_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_282;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_282) : "f"(big_1a_10_1), "f"(whole_1a_10_1));
            float _abs_281;
            asm("abs.f32 %0, %1;" : "=f"(_abs_281) : "f"(_sub_282));
            float _add_283;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_283) : "f"(big_1a_10_1), "f"(0.5f));
            float _sub_284;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_284) : "f"(big_1a_10_1), "f"(0.5f));
            float rounded_1a_10_1 = ((_abs_281 == 0.5f) ? ((above_1a_10_1 > 0.0f) ? _add_283 : _sub_284) : whole_1a_10_1);
            t1_9 = ((d1_24 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_10_1, 7.888609e-31f), 1.7763568e-15f) : t1_9);
        }
        float2 s_24 = make_float2(max_noftz(t0_9, 0.0f), max_noftz(t1_9, 0.0f));
        float2 act_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_24) : "l"(*(const unsigned long long*)&x_24), "l"(*(const unsigned long long*)&s_24));
        float2 _f32x2_mul_a_285 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_24, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_24), s_24), g_24);
        float2 mul_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_285), "l"(*(const unsigned long long*)&u_24));
        float2 mul_1_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_24) : "l"(*(const unsigned long long*)&act_24), "l"(*(const unsigned long long*)&g_24));
        __nv_bfloat162 bf16x2_27 = __float22bfloat162_rn(make_float2(mul_25.x, mul_25.y));
        __nv_bfloat162 bf16x2_1_26 = __float22bfloat162_rn(make_float2(mul_1_24.x, mul_1_24.y));
        dgate_b[2] = __as_u32(bf16x2_27);
        dup_b[2] = __as_u32(bf16x2_1_26);
        float2 x_25 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[3]));
        float2 u_25 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[3]));
        float2 g_25 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[3]));
        float e0_25 = approx_exp2(-x_25.x * 1.442695f);
        float e1_25 = approx_exp2(-x_25.y * 1.442695f);
        float d0_25 = 1.0f + e0_25;
        float d1_25 = 1.0f + e1_25;
        float r0_25 = approx_rcp(d0_25);
        float r1_25 = approx_rcp(d1_25);
        float _fma_286;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_286) : "f"(-d0_25), "f"(r0_25), "f"(1.0f));
        float fma_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_11) : "f"(r0_25), "f"(_fma_286), "f"(r0_25));
        float _fma_287;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_287) : "f"(-d1_25), "f"(r1_25), "f"(1.0f));
        float fma_1_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_10) : "f"(r1_25), "f"(_fma_287), "f"(r1_25));
        float t0_10 = fma_11;
        float t1_10 = fma_1_10;
        int _warp_vote_288 = __any_sync(0xFFFFFFFF, fminf(t0_10, t1_10) == 0.0f);
        if (_warp_vote_288 != 0) {
            float d4_1a_11;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_11) : "f"(d0_25), "f"(0.25f));
            float r4_1a_11 = approx_rcp(d4_1a_11);
            float _fma_290;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_290) : "f"(-d4_1a_11), "f"(r4_1a_11), "f"(1.0f));
            float _fma_289;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_289) : "f"(r4_1a_11), "f"(_fma_290), "f"(r4_1a_11));
            float big_1a_11;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_11) : "f"(mul_f32_rn_noftz(_fma_289, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_291;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_291) : "f"(big_1a_11), "f"(8.388608e+06f));
            float whole_1a_11;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_11) : "f"(_add_291), "f"(8.388608e+06f));
            float above_1a_11;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_11) : "f"(-d4_1a_11), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_11, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_293;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_293) : "f"(big_1a_11), "f"(whole_1a_11));
            float _abs_292;
            asm("abs.f32 %0, %1;" : "=f"(_abs_292) : "f"(_sub_293));
            float _add_294;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_294) : "f"(big_1a_11), "f"(0.5f));
            float _sub_295;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_295) : "f"(big_1a_11), "f"(0.5f));
            float rounded_1a_11 = ((_abs_292 == 0.5f) ? ((above_1a_11 > 0.0f) ? _add_294 : _sub_295) : whole_1a_11);
            t0_10 = ((d0_25 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_11, 7.888609e-31f), 1.7763568e-15f) : t0_10);
            float d4_1a_11_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_11_1) : "f"(d1_25), "f"(0.25f));
            float r4_1a_11_1 = approx_rcp(d4_1a_11_1);
            float _fma_297;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_297) : "f"(-d4_1a_11_1), "f"(r4_1a_11_1), "f"(1.0f));
            float _fma_296;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_296) : "f"(r4_1a_11_1), "f"(_fma_297), "f"(r4_1a_11_1));
            float big_1a_11_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_11_1) : "f"(mul_f32_rn_noftz(_fma_296, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_298;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_298) : "f"(big_1a_11_1), "f"(8.388608e+06f));
            float whole_1a_11_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_11_1) : "f"(_add_298), "f"(8.388608e+06f));
            float above_1a_11_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_11_1) : "f"(-d4_1a_11_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_11_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_300;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_300) : "f"(big_1a_11_1), "f"(whole_1a_11_1));
            float _abs_299;
            asm("abs.f32 %0, %1;" : "=f"(_abs_299) : "f"(_sub_300));
            float _add_301;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_301) : "f"(big_1a_11_1), "f"(0.5f));
            float _sub_302;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_302) : "f"(big_1a_11_1), "f"(0.5f));
            float rounded_1a_11_1 = ((_abs_299 == 0.5f) ? ((above_1a_11_1 > 0.0f) ? _add_301 : _sub_302) : whole_1a_11_1);
            t1_10 = ((d1_25 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_11_1, 7.888609e-31f), 1.7763568e-15f) : t1_10);
        }
        float2 s_25 = make_float2(max_noftz(t0_10, 0.0f), max_noftz(t1_10, 0.0f));
        float2 act_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_25) : "l"(*(const unsigned long long*)&x_25), "l"(*(const unsigned long long*)&s_25));
        float2 _f32x2_mul_a_303 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_25, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_25), s_25), g_25);
        float2 mul_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_303), "l"(*(const unsigned long long*)&u_25));
        float2 mul_1_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_25) : "l"(*(const unsigned long long*)&act_25), "l"(*(const unsigned long long*)&g_25));
        __nv_bfloat162 bf16x2_28 = __float22bfloat162_rn(make_float2(mul_26.x, mul_26.y));
        __nv_bfloat162 bf16x2_1_27 = __float22bfloat162_rn(make_float2(mul_1_25.x, mul_1_25.y));
        dgate_b[3] = __as_u32(bf16x2_28);
        dup_b[3] = __as_u32(bf16x2_1_27);
        float2 x_26 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[4]));
        float2 u_26 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[4]));
        float2 g_26 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[4]));
        float e0_26 = approx_exp2(-x_26.x * 1.442695f);
        float e1_26 = approx_exp2(-x_26.y * 1.442695f);
        float d0_26 = 1.0f + e0_26;
        float d1_26 = 1.0f + e1_26;
        float r0_26 = approx_rcp(d0_26);
        float r1_26 = approx_rcp(d1_26);
        float _fma_304;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_304) : "f"(-d0_26), "f"(r0_26), "f"(1.0f));
        float fma_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_12) : "f"(r0_26), "f"(_fma_304), "f"(r0_26));
        float _fma_305;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_305) : "f"(-d1_26), "f"(r1_26), "f"(1.0f));
        float fma_1_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_11) : "f"(r1_26), "f"(_fma_305), "f"(r1_26));
        float t0_11 = fma_12;
        float t1_11 = fma_1_11;
        int _warp_vote_306 = __any_sync(0xFFFFFFFF, fminf(t0_11, t1_11) == 0.0f);
        if (_warp_vote_306 != 0) {
            float d4_1a_12;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_12) : "f"(d0_26), "f"(0.25f));
            float r4_1a_12 = approx_rcp(d4_1a_12);
            float _fma_308;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_308) : "f"(-d4_1a_12), "f"(r4_1a_12), "f"(1.0f));
            float _fma_307;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_307) : "f"(r4_1a_12), "f"(_fma_308), "f"(r4_1a_12));
            float big_1a_12;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_12) : "f"(mul_f32_rn_noftz(_fma_307, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_309;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_309) : "f"(big_1a_12), "f"(8.388608e+06f));
            float whole_1a_12;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_12) : "f"(_add_309), "f"(8.388608e+06f));
            float above_1a_12;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_12) : "f"(-d4_1a_12), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_12, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_311;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_311) : "f"(big_1a_12), "f"(whole_1a_12));
            float _abs_310;
            asm("abs.f32 %0, %1;" : "=f"(_abs_310) : "f"(_sub_311));
            float _add_312;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_312) : "f"(big_1a_12), "f"(0.5f));
            float _sub_313;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_313) : "f"(big_1a_12), "f"(0.5f));
            float rounded_1a_12 = ((_abs_310 == 0.5f) ? ((above_1a_12 > 0.0f) ? _add_312 : _sub_313) : whole_1a_12);
            t0_11 = ((d0_26 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_12, 7.888609e-31f), 1.7763568e-15f) : t0_11);
            float d4_1a_12_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_12_1) : "f"(d1_26), "f"(0.25f));
            float r4_1a_12_1 = approx_rcp(d4_1a_12_1);
            float _fma_315;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_315) : "f"(-d4_1a_12_1), "f"(r4_1a_12_1), "f"(1.0f));
            float _fma_314;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_314) : "f"(r4_1a_12_1), "f"(_fma_315), "f"(r4_1a_12_1));
            float big_1a_12_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_12_1) : "f"(mul_f32_rn_noftz(_fma_314, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_316;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_316) : "f"(big_1a_12_1), "f"(8.388608e+06f));
            float whole_1a_12_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_12_1) : "f"(_add_316), "f"(8.388608e+06f));
            float above_1a_12_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_12_1) : "f"(-d4_1a_12_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_12_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_318;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_318) : "f"(big_1a_12_1), "f"(whole_1a_12_1));
            float _abs_317;
            asm("abs.f32 %0, %1;" : "=f"(_abs_317) : "f"(_sub_318));
            float _add_319;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_319) : "f"(big_1a_12_1), "f"(0.5f));
            float _sub_320;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_320) : "f"(big_1a_12_1), "f"(0.5f));
            float rounded_1a_12_1 = ((_abs_317 == 0.5f) ? ((above_1a_12_1 > 0.0f) ? _add_319 : _sub_320) : whole_1a_12_1);
            t1_11 = ((d1_26 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_12_1, 7.888609e-31f), 1.7763568e-15f) : t1_11);
        }
        float2 s_26 = make_float2(max_noftz(t0_11, 0.0f), max_noftz(t1_11, 0.0f));
        float2 act_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_26) : "l"(*(const unsigned long long*)&x_26), "l"(*(const unsigned long long*)&s_26));
        float2 _f32x2_mul_a_321 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_26, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_26), s_26), g_26);
        float2 mul_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_321), "l"(*(const unsigned long long*)&u_26));
        float2 mul_1_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_26) : "l"(*(const unsigned long long*)&act_26), "l"(*(const unsigned long long*)&g_26));
        __nv_bfloat162 bf16x2_29 = __float22bfloat162_rn(make_float2(mul_27.x, mul_27.y));
        __nv_bfloat162 bf16x2_1_28 = __float22bfloat162_rn(make_float2(mul_1_26.x, mul_1_26.y));
        dgate_b[4] = __as_u32(bf16x2_29);
        dup_b[4] = __as_u32(bf16x2_1_28);
        float2 x_27 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[5]));
        float2 u_27 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[5]));
        float2 g_27 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[5]));
        float e0_27 = approx_exp2(-x_27.x * 1.442695f);
        float e1_27 = approx_exp2(-x_27.y * 1.442695f);
        float d0_27 = 1.0f + e0_27;
        float d1_27 = 1.0f + e1_27;
        float r0_27 = approx_rcp(d0_27);
        float r1_27 = approx_rcp(d1_27);
        float _fma_322;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_322) : "f"(-d0_27), "f"(r0_27), "f"(1.0f));
        float fma_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_13) : "f"(r0_27), "f"(_fma_322), "f"(r0_27));
        float _fma_323;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_323) : "f"(-d1_27), "f"(r1_27), "f"(1.0f));
        float fma_1_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_12) : "f"(r1_27), "f"(_fma_323), "f"(r1_27));
        float t0_12 = fma_13;
        float t1_12 = fma_1_12;
        int _warp_vote_324 = __any_sync(0xFFFFFFFF, fminf(t0_12, t1_12) == 0.0f);
        if (_warp_vote_324 != 0) {
            float d4_1a_13;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_13) : "f"(d0_27), "f"(0.25f));
            float r4_1a_13 = approx_rcp(d4_1a_13);
            float _fma_326;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_326) : "f"(-d4_1a_13), "f"(r4_1a_13), "f"(1.0f));
            float _fma_325;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_325) : "f"(r4_1a_13), "f"(_fma_326), "f"(r4_1a_13));
            float big_1a_13;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_13) : "f"(mul_f32_rn_noftz(_fma_325, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_327;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_327) : "f"(big_1a_13), "f"(8.388608e+06f));
            float whole_1a_13;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_13) : "f"(_add_327), "f"(8.388608e+06f));
            float above_1a_13;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_13) : "f"(-d4_1a_13), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_13, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_329;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_329) : "f"(big_1a_13), "f"(whole_1a_13));
            float _abs_328;
            asm("abs.f32 %0, %1;" : "=f"(_abs_328) : "f"(_sub_329));
            float _add_330;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_330) : "f"(big_1a_13), "f"(0.5f));
            float _sub_331;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_331) : "f"(big_1a_13), "f"(0.5f));
            float rounded_1a_13 = ((_abs_328 == 0.5f) ? ((above_1a_13 > 0.0f) ? _add_330 : _sub_331) : whole_1a_13);
            t0_12 = ((d0_27 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_13, 7.888609e-31f), 1.7763568e-15f) : t0_12);
            float d4_1a_13_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_13_1) : "f"(d1_27), "f"(0.25f));
            float r4_1a_13_1 = approx_rcp(d4_1a_13_1);
            float _fma_333;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_333) : "f"(-d4_1a_13_1), "f"(r4_1a_13_1), "f"(1.0f));
            float _fma_332;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_332) : "f"(r4_1a_13_1), "f"(_fma_333), "f"(r4_1a_13_1));
            float big_1a_13_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_13_1) : "f"(mul_f32_rn_noftz(_fma_332, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_334;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_334) : "f"(big_1a_13_1), "f"(8.388608e+06f));
            float whole_1a_13_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_13_1) : "f"(_add_334), "f"(8.388608e+06f));
            float above_1a_13_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_13_1) : "f"(-d4_1a_13_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_13_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_336;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_336) : "f"(big_1a_13_1), "f"(whole_1a_13_1));
            float _abs_335;
            asm("abs.f32 %0, %1;" : "=f"(_abs_335) : "f"(_sub_336));
            float _add_337;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_337) : "f"(big_1a_13_1), "f"(0.5f));
            float _sub_338;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_338) : "f"(big_1a_13_1), "f"(0.5f));
            float rounded_1a_13_1 = ((_abs_335 == 0.5f) ? ((above_1a_13_1 > 0.0f) ? _add_337 : _sub_338) : whole_1a_13_1);
            t1_12 = ((d1_27 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_13_1, 7.888609e-31f), 1.7763568e-15f) : t1_12);
        }
        float2 s_27 = make_float2(max_noftz(t0_12, 0.0f), max_noftz(t1_12, 0.0f));
        float2 act_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_27) : "l"(*(const unsigned long long*)&x_27), "l"(*(const unsigned long long*)&s_27));
        float2 _f32x2_mul_a_339 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_27, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_27), s_27), g_27);
        float2 mul_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_339), "l"(*(const unsigned long long*)&u_27));
        float2 mul_1_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_27) : "l"(*(const unsigned long long*)&act_27), "l"(*(const unsigned long long*)&g_27));
        __nv_bfloat162 bf16x2_30 = __float22bfloat162_rn(make_float2(mul_28.x, mul_28.y));
        __nv_bfloat162 bf16x2_1_29 = __float22bfloat162_rn(make_float2(mul_1_27.x, mul_1_27.y));
        dgate_b[5] = __as_u32(bf16x2_30);
        dup_b[5] = __as_u32(bf16x2_1_29);
        float2 x_28 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[6]));
        float2 u_28 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[6]));
        float2 g_28 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[6]));
        float e0_28 = approx_exp2(-x_28.x * 1.442695f);
        float e1_28 = approx_exp2(-x_28.y * 1.442695f);
        float d0_28 = 1.0f + e0_28;
        float d1_28 = 1.0f + e1_28;
        float r0_28 = approx_rcp(d0_28);
        float r1_28 = approx_rcp(d1_28);
        float _fma_340;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_340) : "f"(-d0_28), "f"(r0_28), "f"(1.0f));
        float fma_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_14) : "f"(r0_28), "f"(_fma_340), "f"(r0_28));
        float _fma_341;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_341) : "f"(-d1_28), "f"(r1_28), "f"(1.0f));
        float fma_1_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_13) : "f"(r1_28), "f"(_fma_341), "f"(r1_28));
        float t0_13 = fma_14;
        float t1_13 = fma_1_13;
        int _warp_vote_342 = __any_sync(0xFFFFFFFF, fminf(t0_13, t1_13) == 0.0f);
        if (_warp_vote_342 != 0) {
            float d4_1a_14;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_14) : "f"(d0_28), "f"(0.25f));
            float r4_1a_14 = approx_rcp(d4_1a_14);
            float _fma_344;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_344) : "f"(-d4_1a_14), "f"(r4_1a_14), "f"(1.0f));
            float _fma_343;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_343) : "f"(r4_1a_14), "f"(_fma_344), "f"(r4_1a_14));
            float big_1a_14;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_14) : "f"(mul_f32_rn_noftz(_fma_343, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_345;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_345) : "f"(big_1a_14), "f"(8.388608e+06f));
            float whole_1a_14;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_14) : "f"(_add_345), "f"(8.388608e+06f));
            float above_1a_14;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_14) : "f"(-d4_1a_14), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_14, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_347;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_347) : "f"(big_1a_14), "f"(whole_1a_14));
            float _abs_346;
            asm("abs.f32 %0, %1;" : "=f"(_abs_346) : "f"(_sub_347));
            float _add_348;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_348) : "f"(big_1a_14), "f"(0.5f));
            float _sub_349;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_349) : "f"(big_1a_14), "f"(0.5f));
            float rounded_1a_14 = ((_abs_346 == 0.5f) ? ((above_1a_14 > 0.0f) ? _add_348 : _sub_349) : whole_1a_14);
            t0_13 = ((d0_28 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_14, 7.888609e-31f), 1.7763568e-15f) : t0_13);
            float d4_1a_14_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_14_1) : "f"(d1_28), "f"(0.25f));
            float r4_1a_14_1 = approx_rcp(d4_1a_14_1);
            float _fma_351;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_351) : "f"(-d4_1a_14_1), "f"(r4_1a_14_1), "f"(1.0f));
            float _fma_350;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_350) : "f"(r4_1a_14_1), "f"(_fma_351), "f"(r4_1a_14_1));
            float big_1a_14_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_14_1) : "f"(mul_f32_rn_noftz(_fma_350, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_352;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_352) : "f"(big_1a_14_1), "f"(8.388608e+06f));
            float whole_1a_14_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_14_1) : "f"(_add_352), "f"(8.388608e+06f));
            float above_1a_14_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_14_1) : "f"(-d4_1a_14_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_14_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_354;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_354) : "f"(big_1a_14_1), "f"(whole_1a_14_1));
            float _abs_353;
            asm("abs.f32 %0, %1;" : "=f"(_abs_353) : "f"(_sub_354));
            float _add_355;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_355) : "f"(big_1a_14_1), "f"(0.5f));
            float _sub_356;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_356) : "f"(big_1a_14_1), "f"(0.5f));
            float rounded_1a_14_1 = ((_abs_353 == 0.5f) ? ((above_1a_14_1 > 0.0f) ? _add_355 : _sub_356) : whole_1a_14_1);
            t1_13 = ((d1_28 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_14_1, 7.888609e-31f), 1.7763568e-15f) : t1_13);
        }
        float2 s_28 = make_float2(max_noftz(t0_13, 0.0f), max_noftz(t1_13, 0.0f));
        float2 act_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_28) : "l"(*(const unsigned long long*)&x_28), "l"(*(const unsigned long long*)&s_28));
        float2 _f32x2_mul_a_357 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_28, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_28), s_28), g_28);
        float2 mul_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_357), "l"(*(const unsigned long long*)&u_28));
        float2 mul_1_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_28) : "l"(*(const unsigned long long*)&act_28), "l"(*(const unsigned long long*)&g_28));
        __nv_bfloat162 bf16x2_31 = __float22bfloat162_rn(make_float2(mul_29.x, mul_29.y));
        __nv_bfloat162 bf16x2_1_30 = __float22bfloat162_rn(make_float2(mul_1_28.x, mul_1_28.y));
        dgate_b[6] = __as_u32(bf16x2_31);
        dup_b[6] = __as_u32(bf16x2_1_30);
        float2 x_29 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[7]));
        float2 u_29 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[7]));
        float2 g_29 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[7]));
        float e0_29 = approx_exp2(-x_29.x * 1.442695f);
        float e1_29 = approx_exp2(-x_29.y * 1.442695f);
        float d0_29 = 1.0f + e0_29;
        float d1_29 = 1.0f + e1_29;
        float r0_29 = approx_rcp(d0_29);
        float r1_29 = approx_rcp(d1_29);
        float _fma_358;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_358) : "f"(-d0_29), "f"(r0_29), "f"(1.0f));
        float fma_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_15) : "f"(r0_29), "f"(_fma_358), "f"(r0_29));
        float _fma_359;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_359) : "f"(-d1_29), "f"(r1_29), "f"(1.0f));
        float fma_1_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_14) : "f"(r1_29), "f"(_fma_359), "f"(r1_29));
        float t0_14 = fma_15;
        float t1_14 = fma_1_14;
        int _warp_vote_360 = __any_sync(0xFFFFFFFF, fminf(t0_14, t1_14) == 0.0f);
        if (_warp_vote_360 != 0) {
            float d4_1a_15;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_15) : "f"(d0_29), "f"(0.25f));
            float r4_1a_15 = approx_rcp(d4_1a_15);
            float _fma_362;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_362) : "f"(-d4_1a_15), "f"(r4_1a_15), "f"(1.0f));
            float _fma_361;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_361) : "f"(r4_1a_15), "f"(_fma_362), "f"(r4_1a_15));
            float big_1a_15;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_15) : "f"(mul_f32_rn_noftz(_fma_361, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_363;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_363) : "f"(big_1a_15), "f"(8.388608e+06f));
            float whole_1a_15;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_15) : "f"(_add_363), "f"(8.388608e+06f));
            float above_1a_15;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_15) : "f"(-d4_1a_15), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_15, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_365;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_365) : "f"(big_1a_15), "f"(whole_1a_15));
            float _abs_364;
            asm("abs.f32 %0, %1;" : "=f"(_abs_364) : "f"(_sub_365));
            float _add_366;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_366) : "f"(big_1a_15), "f"(0.5f));
            float _sub_367;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_367) : "f"(big_1a_15), "f"(0.5f));
            float rounded_1a_15 = ((_abs_364 == 0.5f) ? ((above_1a_15 > 0.0f) ? _add_366 : _sub_367) : whole_1a_15);
            t0_14 = ((d0_29 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_15, 7.888609e-31f), 1.7763568e-15f) : t0_14);
            float d4_1a_15_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_15_1) : "f"(d1_29), "f"(0.25f));
            float r4_1a_15_1 = approx_rcp(d4_1a_15_1);
            float _fma_369;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_369) : "f"(-d4_1a_15_1), "f"(r4_1a_15_1), "f"(1.0f));
            float _fma_368;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_368) : "f"(r4_1a_15_1), "f"(_fma_369), "f"(r4_1a_15_1));
            float big_1a_15_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_15_1) : "f"(mul_f32_rn_noftz(_fma_368, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_370;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_370) : "f"(big_1a_15_1), "f"(8.388608e+06f));
            float whole_1a_15_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_15_1) : "f"(_add_370), "f"(8.388608e+06f));
            float above_1a_15_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_15_1) : "f"(-d4_1a_15_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_15_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_372;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_372) : "f"(big_1a_15_1), "f"(whole_1a_15_1));
            float _abs_371;
            asm("abs.f32 %0, %1;" : "=f"(_abs_371) : "f"(_sub_372));
            float _add_373;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_373) : "f"(big_1a_15_1), "f"(0.5f));
            float _sub_374;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_374) : "f"(big_1a_15_1), "f"(0.5f));
            float rounded_1a_15_1 = ((_abs_371 == 0.5f) ? ((above_1a_15_1 > 0.0f) ? _add_373 : _sub_374) : whole_1a_15_1);
            t1_14 = ((d1_29 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_15_1, 7.888609e-31f), 1.7763568e-15f) : t1_14);
        }
        float2 s_29 = make_float2(max_noftz(t0_14, 0.0f), max_noftz(t1_14, 0.0f));
        float2 act_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_29) : "l"(*(const unsigned long long*)&x_29), "l"(*(const unsigned long long*)&s_29));
        float2 _f32x2_mul_a_375 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_29, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_29), s_29), g_29);
        float2 mul_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_375), "l"(*(const unsigned long long*)&u_29));
        float2 mul_1_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_29) : "l"(*(const unsigned long long*)&act_29), "l"(*(const unsigned long long*)&g_29));
        __nv_bfloat162 bf16x2_32 = __float22bfloat162_rn(make_float2(mul_30.x, mul_30.y));
        __nv_bfloat162 bf16x2_1_31 = __float22bfloat162_rn(make_float2(mul_1_29.x, mul_1_29.y));
        dgate_b[7] = __as_u32(bf16x2_32);
        dup_b[7] = __as_u32(bf16x2_1_31);
    }
    int scale_cols = K >> 6;
    int scale_col = bx * 2 + blk;
    int out_index = grow * (2 * K) + col;
    __nv_bfloat162 amax = __as_bf16x2(0);
    __nv_bfloat162 _hxa_377 = amax;
    __nv_bfloat162 _hxb_377 = __as_bf16x2(dgate[0]);
    unsigned int _hxr_377;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_377) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_377)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_377)));
    __nv_bfloat162 _max_376 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_377);
    amax = _max_376;
    __nv_bfloat162 _hxa_379 = amax;
    __nv_bfloat162 _hxb_379 = __as_bf16x2(dgate[1]);
    unsigned int _hxr_379;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_379) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_379)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_379)));
    __nv_bfloat162 _max_378 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_379);
    amax = _max_378;
    __nv_bfloat162 _hxa_381 = amax;
    __nv_bfloat162 _hxb_381 = __as_bf16x2(dgate[2]);
    unsigned int _hxr_381;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_381) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_381)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_381)));
    __nv_bfloat162 _max_380 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_381);
    amax = _max_380;
    __nv_bfloat162 _hxa_383 = amax;
    __nv_bfloat162 _hxb_383 = __as_bf16x2(dgate[3]);
    unsigned int _hxr_383;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_383) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_383)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_383)));
    __nv_bfloat162 _max_382 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_383);
    amax = _max_382;
    __nv_bfloat162 _hxa_385 = amax;
    __nv_bfloat162 _hxb_385 = __as_bf16x2(dgate[4]);
    unsigned int _hxr_385;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_385) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_385)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_385)));
    __nv_bfloat162 _max_384 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_385);
    amax = _max_384;
    __nv_bfloat162 _hxa_387 = amax;
    __nv_bfloat162 _hxb_387 = __as_bf16x2(dgate[5]);
    unsigned int _hxr_387;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_387) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_387)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_387)));
    __nv_bfloat162 _max_386 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_387);
    amax = _max_386;
    __nv_bfloat162 _hxa_389 = amax;
    __nv_bfloat162 _hxb_389 = __as_bf16x2(dgate[6]);
    unsigned int _hxr_389;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_389) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_389)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_389)));
    __nv_bfloat162 _max_388 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_389);
    amax = _max_388;
    __nv_bfloat162 _hxa_391 = amax;
    __nv_bfloat162 _hxb_391 = __as_bf16x2(dgate[7]);
    unsigned int _hxr_391;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_391) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_391)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_391)));
    __nv_bfloat162 _max_390 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_391);
    amax = _max_390;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_393;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_393) : "r"(bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_393) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_392;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_392) : "h"(e4m3x2_bf16x2), "h"(_cvt_satfinite_e4m3x2_bf16x2_393));
    words[0] = _mov_b32_pack_u16x2_392;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_395;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_395) : "r"(bf16x2_mul_1_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_395) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_394;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_394) : "h"(e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_395));
    words[1] = _mov_b32_pack_u16x2_394;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_397;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_397) : "r"(bf16x2_mul_1_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_397) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_396;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_396) : "h"(e4m3x2_bf16x2_4), "h"(_cvt_satfinite_e4m3x2_bf16x2_397));
    words[2] = _mov_b32_pack_u16x2_396;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_399;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_399) : "r"(bf16x2_mul_1_2));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_2);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_399) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_398;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_398) : "h"(e4m3x2_bf16x2_5), "h"(_cvt_satfinite_e4m3x2_bf16x2_399));
    words[3] = _mov_b32_pack_u16x2_398;
    reinterpret_cast<int4*>(row_output + out_index)[0] = reinterpret_cast<int4*>(words + 0)[0];
    __nv_bfloat162 amax_0 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_401 = amax_0;
    __nv_bfloat162 _hxb_401 = __as_bf16x2(dup[0]);
    unsigned int _hxr_401;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_401) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_401)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_401)));
    __nv_bfloat162 _max_400 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_401);
    amax_0 = _max_400;
    __nv_bfloat162 _hxa_403 = amax_0;
    __nv_bfloat162 _hxb_403 = __as_bf16x2(dup[1]);
    unsigned int _hxr_403;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_403) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_403)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_403)));
    __nv_bfloat162 _max_402 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_403);
    amax_0 = _max_402;
    __nv_bfloat162 _hxa_405 = amax_0;
    __nv_bfloat162 _hxb_405 = __as_bf16x2(dup[2]);
    unsigned int _hxr_405;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_405) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_405)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_405)));
    __nv_bfloat162 _max_404 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_405);
    amax_0 = _max_404;
    __nv_bfloat162 _hxa_407 = amax_0;
    __nv_bfloat162 _hxb_407 = __as_bf16x2(dup[3]);
    unsigned int _hxr_407;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_407) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_407)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_407)));
    __nv_bfloat162 _max_406 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_407);
    amax_0 = _max_406;
    __nv_bfloat162 _hxa_409 = amax_0;
    __nv_bfloat162 _hxb_409 = __as_bf16x2(dup[4]);
    unsigned int _hxr_409;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_409) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_409)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_409)));
    __nv_bfloat162 _max_408 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_409);
    amax_0 = _max_408;
    __nv_bfloat162 _hxa_411 = amax_0;
    __nv_bfloat162 _hxb_411 = __as_bf16x2(dup[5]);
    unsigned int _hxr_411;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_411) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_411)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_411)));
    __nv_bfloat162 _max_410 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_411);
    amax_0 = _max_410;
    __nv_bfloat162 _hxa_413 = amax_0;
    __nv_bfloat162 _hxb_413 = __as_bf16x2(dup[6]);
    unsigned int _hxr_413;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_413) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_413)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_413)));
    __nv_bfloat162 _max_412 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_413);
    amax_0 = _max_412;
    __nv_bfloat162 _hxa_415 = amax_0;
    __nv_bfloat162 _hxb_415 = __as_bf16x2(dup[7]);
    unsigned int _hxr_415;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_415) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_415)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_415)));
    __nv_bfloat162 _max_414 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_415);
    amax_0 = _max_414;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_417;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_417) : "r"(bf16x2_mul_1_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_417) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_416;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_416) : "h"(e4m3x2_bf16x2_6), "h"(_cvt_satfinite_e4m3x2_bf16x2_417));
    words_0[0] = _mov_b32_pack_u16x2_416;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_419;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_419) : "r"(bf16x2_mul_1_4));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_4);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_419) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_418;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_418) : "h"(e4m3x2_bf16x2_7), "h"(_cvt_satfinite_e4m3x2_bf16x2_419));
    words_0[1] = _mov_b32_pack_u16x2_418;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_421;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_421) : "r"(bf16x2_mul_1_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_421) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_420;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_420) : "h"(e4m3x2_bf16x2_8), "h"(_cvt_satfinite_e4m3x2_bf16x2_421));
    words_0[2] = _mov_b32_pack_u16x2_420;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_423;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_423) : "r"(bf16x2_mul_1_6));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_6);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_423) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_422;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_422) : "h"(e4m3x2_bf16x2_9), "h"(_cvt_satfinite_e4m3x2_bf16x2_423));
    words_0[3] = _mov_b32_pack_u16x2_422;
    reinterpret_cast<int4*>(row_output + (out_index + K))[0] = reinterpret_cast<int4*>(words_0 + 0)[0];
    int out_index_hi = grow_b * (2 * K) + col;
    __nv_bfloat162 amax_1 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_425 = amax_1;
    __nv_bfloat162 _hxb_425 = __as_bf16x2(dgate_b[0]);
    unsigned int _hxr_425;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_425) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_425)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_425)));
    __nv_bfloat162 _max_424 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_425);
    amax_1 = _max_424;
    __nv_bfloat162 _hxa_427 = amax_1;
    __nv_bfloat162 _hxb_427 = __as_bf16x2(dgate_b[1]);
    unsigned int _hxr_427;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_427) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_427)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_427)));
    __nv_bfloat162 _max_426 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_427);
    amax_1 = _max_426;
    __nv_bfloat162 _hxa_429 = amax_1;
    __nv_bfloat162 _hxb_429 = __as_bf16x2(dgate_b[2]);
    unsigned int _hxr_429;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_429) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_429)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_429)));
    __nv_bfloat162 _max_428 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_429);
    amax_1 = _max_428;
    __nv_bfloat162 _hxa_431 = amax_1;
    __nv_bfloat162 _hxb_431 = __as_bf16x2(dgate_b[3]);
    unsigned int _hxr_431;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_431) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_431)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_431)));
    __nv_bfloat162 _max_430 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_431);
    amax_1 = _max_430;
    __nv_bfloat162 _hxa_433 = amax_1;
    __nv_bfloat162 _hxb_433 = __as_bf16x2(dgate_b[4]);
    unsigned int _hxr_433;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_433) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_433)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_433)));
    __nv_bfloat162 _max_432 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_433);
    amax_1 = _max_432;
    __nv_bfloat162 _hxa_435 = amax_1;
    __nv_bfloat162 _hxb_435 = __as_bf16x2(dgate_b[5]);
    unsigned int _hxr_435;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_435) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_435)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_435)));
    __nv_bfloat162 _max_434 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_435);
    amax_1 = _max_434;
    __nv_bfloat162 _hxa_437 = amax_1;
    __nv_bfloat162 _hxb_437 = __as_bf16x2(dgate_b[6]);
    unsigned int _hxr_437;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_437) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_437)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_437)));
    __nv_bfloat162 _max_436 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_437);
    amax_1 = _max_436;
    __nv_bfloat162 _hxa_439 = amax_1;
    __nv_bfloat162 _hxb_439 = __as_bf16x2(dgate_b[7]);
    unsigned int _hxr_439;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_439) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_439)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_439)));
    __nv_bfloat162 _max_438 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_439);
    amax_1 = _max_438;
    unsigned int mag_1 = __as_u32(amax_1) & 2147450879;
    unsigned int peer_1 = __shfl_xor_sync(0xFFFFFFFF, mag_1, 1);
    __nv_bfloat162 both_1 = __hmax2_nan(__as_bf16x2(mag_1), __as_bf16x2(peer_1));
    unsigned int folded_1 = __as_u32(__hmax2_nan(both_1, __as_bf16x2(__as_u32(both_1) >> 16)));
    unsigned int amax_bits_1 = (folded_1 & 65535) << 16;
    int scale_i32_1 = (int)(amax_bits_1 + 2031616 >> 23) - 8;
    if (scale_i32_1 < 0) {
        scale_i32_1 = 0;
    }
    if ((amax_bits_1 & 2139095040) == 2139095040) {
        scale_i32_1 = 255;
    }
    unsigned int scale_3 = (unsigned int)scale_i32_1;
    unsigned int inv_1 = 254 - scale_3 << 7;
    if (scale_3 == 255) {
        inv_1 = 32704;
    }
    unsigned int inv2_2 = inv_1 | inv_1 << 16;
    int scale_index_1 = ((grow_b >> 7) * scale_cols + (scale_col >> 2)) * 512 + (grow_b & 31) * 16 + (grow_b >> 5 & 3) * 4 + (scale_col & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + scale_index_1) + (0)) = (unsigned char)(scale_3);
    }
    unsigned int words_1[4];
    uint32_t bf16x2_mul_10;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_10) : "r"(dgate_b[0]), "r"(inv2_2));
    uint16_t e4m3x2_bf16x2_10;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_10) : "r"(bf16x2_mul_10));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_10);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_10) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_7;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_7) : "r"(dgate_b[1]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_441;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_441) : "r"(bf16x2_mul_1_7));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_7);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_441) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_440;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_440) : "h"(e4m3x2_bf16x2_10), "h"(_cvt_satfinite_e4m3x2_bf16x2_441));
    words_1[0] = _mov_b32_pack_u16x2_440;
    uint32_t bf16x2_mul_11;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_11) : "r"(dgate_b[2]), "r"(inv2_2));
    uint16_t e4m3x2_bf16x2_11;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_11) : "r"(bf16x2_mul_11));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_11);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_11) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_8;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_8) : "r"(dgate_b[3]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_443;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_443) : "r"(bf16x2_mul_1_8));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_8);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_443) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_442;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_442) : "h"(e4m3x2_bf16x2_11), "h"(_cvt_satfinite_e4m3x2_bf16x2_443));
    words_1[1] = _mov_b32_pack_u16x2_442;
    uint32_t bf16x2_mul_12;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_12) : "r"(dgate_b[4]), "r"(inv2_2));
    uint16_t e4m3x2_bf16x2_12;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_12) : "r"(bf16x2_mul_12));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_12);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_12) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_9;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_9) : "r"(dgate_b[5]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_445;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_445) : "r"(bf16x2_mul_1_9));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_9);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_445) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_444;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_444) : "h"(e4m3x2_bf16x2_12), "h"(_cvt_satfinite_e4m3x2_bf16x2_445));
    words_1[2] = _mov_b32_pack_u16x2_444;
    uint32_t bf16x2_mul_13;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_13) : "r"(dgate_b[6]), "r"(inv2_2));
    uint16_t e4m3x2_bf16x2_13;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_13) : "r"(bf16x2_mul_13));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_13);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_13) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_10;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_10) : "r"(dgate_b[7]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_447;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_447) : "r"(bf16x2_mul_1_10));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_10);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_447) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_446;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_446) : "h"(e4m3x2_bf16x2_13), "h"(_cvt_satfinite_e4m3x2_bf16x2_447));
    words_1[3] = _mov_b32_pack_u16x2_446;
    reinterpret_cast<int4*>(row_output + out_index_hi)[0] = reinterpret_cast<int4*>(words_1 + 0)[0];
    __nv_bfloat162 amax_2 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_449 = amax_2;
    __nv_bfloat162 _hxb_449 = __as_bf16x2(dup_b[0]);
    unsigned int _hxr_449;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_449) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_449)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_449)));
    __nv_bfloat162 _max_448 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_449);
    amax_2 = _max_448;
    __nv_bfloat162 _hxa_451 = amax_2;
    __nv_bfloat162 _hxb_451 = __as_bf16x2(dup_b[1]);
    unsigned int _hxr_451;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_451) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_451)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_451)));
    __nv_bfloat162 _max_450 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_451);
    amax_2 = _max_450;
    __nv_bfloat162 _hxa_453 = amax_2;
    __nv_bfloat162 _hxb_453 = __as_bf16x2(dup_b[2]);
    unsigned int _hxr_453;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_453) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_453)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_453)));
    __nv_bfloat162 _max_452 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_453);
    amax_2 = _max_452;
    __nv_bfloat162 _hxa_455 = amax_2;
    __nv_bfloat162 _hxb_455 = __as_bf16x2(dup_b[3]);
    unsigned int _hxr_455;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_455) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_455)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_455)));
    __nv_bfloat162 _max_454 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_455);
    amax_2 = _max_454;
    __nv_bfloat162 _hxa_457 = amax_2;
    __nv_bfloat162 _hxb_457 = __as_bf16x2(dup_b[4]);
    unsigned int _hxr_457;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_457) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_457)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_457)));
    __nv_bfloat162 _max_456 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_457);
    amax_2 = _max_456;
    __nv_bfloat162 _hxa_459 = amax_2;
    __nv_bfloat162 _hxb_459 = __as_bf16x2(dup_b[5]);
    unsigned int _hxr_459;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_459) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_459)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_459)));
    __nv_bfloat162 _max_458 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_459);
    amax_2 = _max_458;
    __nv_bfloat162 _hxa_461 = amax_2;
    __nv_bfloat162 _hxb_461 = __as_bf16x2(dup_b[6]);
    unsigned int _hxr_461;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_461) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_461)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_461)));
    __nv_bfloat162 _max_460 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_461);
    amax_2 = _max_460;
    __nv_bfloat162 _hxa_463 = amax_2;
    __nv_bfloat162 _hxb_463 = __as_bf16x2(dup_b[7]);
    unsigned int _hxr_463;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_463) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_463)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_463)));
    __nv_bfloat162 _max_462 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_463);
    amax_2 = _max_462;
    unsigned int mag_2 = __as_u32(amax_2) & 2147450879;
    unsigned int peer_2 = __shfl_xor_sync(0xFFFFFFFF, mag_2, 1);
    __nv_bfloat162 both_2 = __hmax2_nan(__as_bf16x2(mag_2), __as_bf16x2(peer_2));
    unsigned int folded_2 = __as_u32(__hmax2_nan(both_2, __as_bf16x2(__as_u32(both_2) >> 16)));
    unsigned int amax_bits_2 = (folded_2 & 65535) << 16;
    int scale_i32_2 = (int)(amax_bits_2 + 2031616 >> 23) - 8;
    if (scale_i32_2 < 0) {
        scale_i32_2 = 0;
    }
    if ((amax_bits_2 & 2139095040) == 2139095040) {
        scale_i32_2 = 255;
    }
    unsigned int scale_4 = (unsigned int)scale_i32_2;
    unsigned int inv_2 = 254 - scale_4 << 7;
    if (scale_4 == 255) {
        inv_2 = 32704;
    }
    unsigned int inv2_3 = inv_2 | inv_2 << 16;
    int scale_index_2 = ((grow_b >> 7) * scale_cols + (scale_col + (K >> 5) >> 2)) * 512 + (grow_b & 31) * 16 + (grow_b >> 5 & 3) * 4 + (scale_col + (K >> 5) & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + scale_index_2) + (0)) = (unsigned char)(scale_4);
    }
    unsigned int words_2[4];
    uint32_t bf16x2_mul_14;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_14) : "r"(dup_b[0]), "r"(inv2_3));
    uint16_t e4m3x2_bf16x2_14;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_14) : "r"(bf16x2_mul_14));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_14);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_14) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_11;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_11) : "r"(dup_b[1]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_465;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_465) : "r"(bf16x2_mul_1_11));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_11);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_465) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_464;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_464) : "h"(e4m3x2_bf16x2_14), "h"(_cvt_satfinite_e4m3x2_bf16x2_465));
    words_2[0] = _mov_b32_pack_u16x2_464;
    uint32_t bf16x2_mul_15;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_15) : "r"(dup_b[2]), "r"(inv2_3));
    uint16_t e4m3x2_bf16x2_15;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_15) : "r"(bf16x2_mul_15));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_15);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_15) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_12;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_12) : "r"(dup_b[3]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_467;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_467) : "r"(bf16x2_mul_1_12));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_12);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_467) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_466;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_466) : "h"(e4m3x2_bf16x2_15), "h"(_cvt_satfinite_e4m3x2_bf16x2_467));
    words_2[1] = _mov_b32_pack_u16x2_466;
    uint32_t bf16x2_mul_16;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_16) : "r"(dup_b[4]), "r"(inv2_3));
    uint16_t e4m3x2_bf16x2_16;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_16) : "r"(bf16x2_mul_16));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_16);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_16) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_13;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_13) : "r"(dup_b[5]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_469;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_469) : "r"(bf16x2_mul_1_13));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_13);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_469) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_468;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_468) : "h"(e4m3x2_bf16x2_16), "h"(_cvt_satfinite_e4m3x2_bf16x2_469));
    words_2[2] = _mov_b32_pack_u16x2_468;
    uint32_t bf16x2_mul_17;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_17) : "r"(dup_b[6]), "r"(inv2_3));
    uint16_t e4m3x2_bf16x2_17;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_17) : "r"(bf16x2_mul_17));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_17);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_17) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_14;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_14) : "r"(dup_b[7]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_471;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_471) : "r"(bf16x2_mul_1_14));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_14);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_471) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_470;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_470) : "h"(e4m3x2_bf16x2_17), "h"(_cvt_satfinite_e4m3x2_bf16x2_471));
    words_2[3] = _mov_b32_pack_u16x2_470;
    reinterpret_cast<int4*>(row_output + (out_index_hi + K))[0] = reinterpret_cast<int4*>(words_2 + 0)[0];
}

} // extern "C"
