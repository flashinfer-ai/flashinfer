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

#define LAUNCH_MIN_BLOCKS 9

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
kernel_cake_gated_act_mxfp8_1f02c63ba6b7d1e0a3af(__nv_bfloat16* __restrict__ gated_input, __nv_bfloat16* __restrict__ grad, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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
        asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
            : "=r"(_iv_0_0), "=r"(_iv_0_1), "=r"(_iv_0_2), "=r"(_iv_0_3), "=r"(_iv_0_4), "=r"(_iv_0_5), "=r"(_iv_0_6), "=r"(_iv_0_7) : "l"((const void*)(gated_input + gate_index + (0))), "l"(0x12F0000000000000ULL) : "memory");
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
        asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
            : "=r"(_iv_1_0), "=r"(_iv_1_1), "=r"(_iv_1_2), "=r"(_iv_1_3), "=r"(_iv_1_4), "=r"(_iv_1_5), "=r"(_iv_1_6), "=r"(_iv_1_7) : "l"((const void*)(gated_input + (gate_index + K) + (0))), "l"(0x12F0000000000000ULL) : "memory");
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
        asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
            : "=r"(_iv_2_0), "=r"(_iv_2_1), "=r"(_iv_2_2), "=r"(_iv_2_3), "=r"(_iv_2_4), "=r"(_iv_2_5), "=r"(_iv_2_6), "=r"(_iv_2_7) : "l"((const void*)(grad + (grow * K + col) + (0))), "l"(0x12F0000000000000ULL) : "memory");
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
        float2 _f32x2_mul_b_5 = make_float2(-1.442695f, -1.442695f);
        float2 t;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t) : "l"(*(const unsigned long long*)&x), "l"(*(const unsigned long long*)&_f32x2_mul_b_5));
        float e0 = approx_exp2(t.x);
        float e1 = approx_exp2(t.y);
        float2 nd = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0, e1));
        float d0 = -nd.x;
        float d1 = -nd.y;
        float r0 = approx_rcp(d0);
        float r1 = approx_rcp(d1);
        float2 r = make_float2(r0, r1);
        float2 s_0 = fma_f32x2_rn_noftz(r, fma_f32x2_rn_noftz(nd, r, make_float2(1.0f, 1.0f)), r);
        float2 s = s_0;
        float2 act;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act) : "l"(*(const unsigned long long*)&x), "l"(*(const unsigned long long*)&s));
        float2 _f32x2_mul_a_6 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s), s), g);
        float2 mul;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul) : "l"(*(const unsigned long long*)&_f32x2_mul_a_6), "l"(*(const unsigned long long*)&u));
        float2 mul_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1) : "l"(*(const unsigned long long*)&act), "l"(*(const unsigned long long*)&g));
        __nv_bfloat162 bf16x2_0 = __float22bfloat162_rn(make_float2(mul.x, mul.y));
        __nv_bfloat162 bf16x2_1_0 = __float22bfloat162_rn(make_float2(mul_1.x, mul_1.y));
        dgate[0] = __as_u32(bf16x2_0);
        dup[0] = __as_u32(bf16x2_1_0);
        float2 x_0 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[1]));
        float2 u_0 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[1]));
        float2 g_0 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[1]));
        float2 _f32x2_mul_b_7 = make_float2(-1.442695f, -1.442695f);
        float2 t_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_0) : "l"(*(const unsigned long long*)&x_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_7));
        float e0_0 = approx_exp2(t_0.x);
        float e1_0 = approx_exp2(t_0.y);
        float2 nd_0 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_0, e1_0));
        float d0_0 = -nd_0.x;
        float d1_0 = -nd_0.y;
        float r0_0 = approx_rcp(d0_0);
        float r1_0 = approx_rcp(d1_0);
        float2 r_0 = make_float2(r0_0, r1_0);
        float2 s_2 = fma_f32x2_rn_noftz(r_0, fma_f32x2_rn_noftz(nd_0, r_0, make_float2(1.0f, 1.0f)), r_0);
        float2 s_1 = s_2;
        float2 act_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_0) : "l"(*(const unsigned long long*)&x_0), "l"(*(const unsigned long long*)&s_1));
        float2 _f32x2_mul_a_8 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1), s_1), g_0);
        float2 mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_8), "l"(*(const unsigned long long*)&u_0));
        float2 mul_1_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_0) : "l"(*(const unsigned long long*)&act_0), "l"(*(const unsigned long long*)&g_0));
        __nv_bfloat162 bf16x2_2 = __float22bfloat162_rn(make_float2(mul_0.x, mul_0.y));
        __nv_bfloat162 bf16x2_1_1 = __float22bfloat162_rn(make_float2(mul_1_0.x, mul_1_0.y));
        dgate[1] = __as_u32(bf16x2_2);
        dup[1] = __as_u32(bf16x2_1_1);
        float2 x_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[2]));
        float2 u_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[2]));
        float2 g_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[2]));
        float2 _f32x2_mul_b_9 = make_float2(-1.442695f, -1.442695f);
        float2 t_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_1) : "l"(*(const unsigned long long*)&x_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_9));
        float e0_1 = approx_exp2(t_1.x);
        float e1_1 = approx_exp2(t_1.y);
        float2 nd_1 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_1, e1_1));
        float d0_1 = -nd_1.x;
        float d1_1 = -nd_1.y;
        float r0_1 = approx_rcp(d0_1);
        float r1_1 = approx_rcp(d1_1);
        float2 r_1 = make_float2(r0_1, r1_1);
        float2 s_4 = fma_f32x2_rn_noftz(r_1, fma_f32x2_rn_noftz(nd_1, r_1, make_float2(1.0f, 1.0f)), r_1);
        float2 s_3 = s_4;
        float2 act_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_1) : "l"(*(const unsigned long long*)&x_1), "l"(*(const unsigned long long*)&s_3));
        float2 _f32x2_mul_a_10 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_3), s_3), g_1);
        float2 mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_10), "l"(*(const unsigned long long*)&u_1));
        float2 mul_1_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_1) : "l"(*(const unsigned long long*)&act_1), "l"(*(const unsigned long long*)&g_1));
        __nv_bfloat162 bf16x2_3 = __float22bfloat162_rn(make_float2(mul_2.x, mul_2.y));
        __nv_bfloat162 bf16x2_1_2 = __float22bfloat162_rn(make_float2(mul_1_1.x, mul_1_1.y));
        dgate[2] = __as_u32(bf16x2_3);
        dup[2] = __as_u32(bf16x2_1_2);
        float2 x_2 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[3]));
        float2 u_2 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[3]));
        float2 g_2 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[3]));
        float2 _f32x2_mul_b_11 = make_float2(-1.442695f, -1.442695f);
        float2 t_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_2) : "l"(*(const unsigned long long*)&x_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_11));
        float e0_2 = approx_exp2(t_2.x);
        float e1_2 = approx_exp2(t_2.y);
        float2 nd_2 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_2, e1_2));
        float d0_2 = -nd_2.x;
        float d1_2 = -nd_2.y;
        float r0_2 = approx_rcp(d0_2);
        float r1_2 = approx_rcp(d1_2);
        float2 r_2 = make_float2(r0_2, r1_2);
        float2 s_6 = fma_f32x2_rn_noftz(r_2, fma_f32x2_rn_noftz(nd_2, r_2, make_float2(1.0f, 1.0f)), r_2);
        float2 s_5 = s_6;
        float2 act_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_2) : "l"(*(const unsigned long long*)&x_2), "l"(*(const unsigned long long*)&s_5));
        float2 _f32x2_mul_a_12 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_5), s_5), g_2);
        float2 mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_12), "l"(*(const unsigned long long*)&u_2));
        float2 mul_1_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_2) : "l"(*(const unsigned long long*)&act_2), "l"(*(const unsigned long long*)&g_2));
        __nv_bfloat162 bf16x2_4 = __float22bfloat162_rn(make_float2(mul_3.x, mul_3.y));
        __nv_bfloat162 bf16x2_1_3 = __float22bfloat162_rn(make_float2(mul_1_2.x, mul_1_2.y));
        dgate[3] = __as_u32(bf16x2_4);
        dup[3] = __as_u32(bf16x2_1_3);
        float2 x_3 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[4]));
        float2 u_3 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[4]));
        float2 g_3 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[4]));
        float2 _f32x2_mul_b_13 = make_float2(-1.442695f, -1.442695f);
        float2 t_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_3) : "l"(*(const unsigned long long*)&x_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_13));
        float e0_3 = approx_exp2(t_3.x);
        float e1_3 = approx_exp2(t_3.y);
        float2 nd_3 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_3, e1_3));
        float d0_3 = -nd_3.x;
        float d1_3 = -nd_3.y;
        float r0_3 = approx_rcp(d0_3);
        float r1_3 = approx_rcp(d1_3);
        float2 r_3 = make_float2(r0_3, r1_3);
        float2 s_8 = fma_f32x2_rn_noftz(r_3, fma_f32x2_rn_noftz(nd_3, r_3, make_float2(1.0f, 1.0f)), r_3);
        float2 s_7 = s_8;
        float2 act_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_3) : "l"(*(const unsigned long long*)&x_3), "l"(*(const unsigned long long*)&s_7));
        float2 _f32x2_mul_a_14 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_7), s_7), g_3);
        float2 mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&u_3));
        float2 mul_1_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_3) : "l"(*(const unsigned long long*)&act_3), "l"(*(const unsigned long long*)&g_3));
        __nv_bfloat162 bf16x2_5 = __float22bfloat162_rn(make_float2(mul_4.x, mul_4.y));
        __nv_bfloat162 bf16x2_1_4 = __float22bfloat162_rn(make_float2(mul_1_3.x, mul_1_3.y));
        dgate[4] = __as_u32(bf16x2_5);
        dup[4] = __as_u32(bf16x2_1_4);
        float2 x_4 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[5]));
        float2 u_4 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[5]));
        float2 g_4 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[5]));
        float2 _f32x2_mul_b_15 = make_float2(-1.442695f, -1.442695f);
        float2 t_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_4) : "l"(*(const unsigned long long*)&x_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_15));
        float e0_4 = approx_exp2(t_4.x);
        float e1_4 = approx_exp2(t_4.y);
        float2 nd_4 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_4, e1_4));
        float d0_4 = -nd_4.x;
        float d1_4 = -nd_4.y;
        float r0_4 = approx_rcp(d0_4);
        float r1_4 = approx_rcp(d1_4);
        float2 r_4 = make_float2(r0_4, r1_4);
        float2 s_10 = fma_f32x2_rn_noftz(r_4, fma_f32x2_rn_noftz(nd_4, r_4, make_float2(1.0f, 1.0f)), r_4);
        float2 s_9 = s_10;
        float2 act_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_4) : "l"(*(const unsigned long long*)&x_4), "l"(*(const unsigned long long*)&s_9));
        float2 _f32x2_mul_a_16 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_9), s_9), g_4);
        float2 mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_16), "l"(*(const unsigned long long*)&u_4));
        float2 mul_1_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_4) : "l"(*(const unsigned long long*)&act_4), "l"(*(const unsigned long long*)&g_4));
        __nv_bfloat162 bf16x2_6 = __float22bfloat162_rn(make_float2(mul_5.x, mul_5.y));
        __nv_bfloat162 bf16x2_1_5 = __float22bfloat162_rn(make_float2(mul_1_4.x, mul_1_4.y));
        dgate[5] = __as_u32(bf16x2_6);
        dup[5] = __as_u32(bf16x2_1_5);
        float2 x_5 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[6]));
        float2 u_5 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[6]));
        float2 g_5 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[6]));
        float2 _f32x2_mul_b_17 = make_float2(-1.442695f, -1.442695f);
        float2 t_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_5) : "l"(*(const unsigned long long*)&x_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_17));
        float e0_5 = approx_exp2(t_5.x);
        float e1_5 = approx_exp2(t_5.y);
        float2 nd_5 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_5, e1_5));
        float d0_5 = -nd_5.x;
        float d1_5 = -nd_5.y;
        float r0_5 = approx_rcp(d0_5);
        float r1_5 = approx_rcp(d1_5);
        float2 r_5 = make_float2(r0_5, r1_5);
        float2 s_12 = fma_f32x2_rn_noftz(r_5, fma_f32x2_rn_noftz(nd_5, r_5, make_float2(1.0f, 1.0f)), r_5);
        float2 s_11 = s_12;
        float2 act_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_5) : "l"(*(const unsigned long long*)&x_5), "l"(*(const unsigned long long*)&s_11));
        float2 _f32x2_mul_a_18 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_11), s_11), g_5);
        float2 mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&u_5));
        float2 mul_1_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_5) : "l"(*(const unsigned long long*)&act_5), "l"(*(const unsigned long long*)&g_5));
        __nv_bfloat162 bf16x2_7 = __float22bfloat162_rn(make_float2(mul_6.x, mul_6.y));
        __nv_bfloat162 bf16x2_1_6 = __float22bfloat162_rn(make_float2(mul_1_5.x, mul_1_5.y));
        dgate[6] = __as_u32(bf16x2_7);
        dup[6] = __as_u32(bf16x2_1_6);
        float2 x_6 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[7]));
        float2 u_6 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[7]));
        float2 g_6 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[7]));
        float2 _f32x2_mul_b_19 = make_float2(-1.442695f, -1.442695f);
        float2 t_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_6) : "l"(*(const unsigned long long*)&x_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_19));
        float e0_6 = approx_exp2(t_6.x);
        float e1_6 = approx_exp2(t_6.y);
        float2 nd_6 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_6, e1_6));
        float d0_6 = -nd_6.x;
        float d1_6 = -nd_6.y;
        float r0_6 = approx_rcp(d0_6);
        float r1_6 = approx_rcp(d1_6);
        float2 r_6 = make_float2(r0_6, r1_6);
        float2 s_14 = fma_f32x2_rn_noftz(r_6, fma_f32x2_rn_noftz(nd_6, r_6, make_float2(1.0f, 1.0f)), r_6);
        float2 s_13 = s_14;
        float2 act_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_6) : "l"(*(const unsigned long long*)&x_6), "l"(*(const unsigned long long*)&s_13));
        float2 _f32x2_mul_a_20 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_13), s_13), g_6);
        float2 mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_20), "l"(*(const unsigned long long*)&u_6));
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
        float2 _f32x2_mul_b_21 = make_float2(-1.442695f, -1.442695f);
        float2 t_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_7) : "l"(*(const unsigned long long*)&x_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_21));
        float e0_7 = approx_exp2(t_7.x);
        float e1_7 = approx_exp2(t_7.y);
        float2 nd_7 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_7, e1_7));
        float d0_7 = -nd_7.x;
        float d1_7 = -nd_7.y;
        float r0_7 = approx_rcp(d0_7);
        float r1_7 = approx_rcp(d1_7);
        float2 r_7 = make_float2(r0_7, r1_7);
        float2 s_15 = fma_f32x2_rn_noftz(r_7, fma_f32x2_rn_noftz(nd_7, r_7, make_float2(1.0f, 1.0f)), r_7);
        float t0 = s_15.x;
        float t1 = s_15.y;
        int _warp_vote_22 = __any_sync(0xFFFFFFFF, fminf(t0, t1) == 0.0f);
        if (_warp_vote_22 != 0) {
            float d4_1a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a) : "f"(d0_7), "f"(0.25f));
            float r4_1a = approx_rcp(d4_1a);
            float _fma_24;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(-d4_1a), "f"(r4_1a), "f"(1.0f));
            float _fma_23;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(r4_1a), "f"(_fma_24), "f"(r4_1a));
            float big_1a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a) : "f"(mul_f32_rn_noftz(_fma_23, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_25;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(big_1a), "f"(8.388608e+06f));
            float whole_1a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a) : "f"(_add_25), "f"(8.388608e+06f));
            float above_1a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a) : "f"(-d4_1a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_27;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27) : "f"(big_1a), "f"(whole_1a));
            float _abs_26;
            asm("abs.f32 %0, %1;" : "=f"(_abs_26) : "f"(_sub_27));
            float _add_28;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(big_1a), "f"(0.5f));
            float _sub_29;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_29) : "f"(big_1a), "f"(0.5f));
            float rounded_1a = ((_abs_26 == 0.5f) ? ((above_1a > 0.0f) ? _add_28 : _sub_29) : whole_1a);
            t0 = ((d0_7 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a, 7.888609e-31f), 1.7763568e-15f) : t0);
            float d4_1a_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_0) : "f"(d1_7), "f"(0.25f));
            float r4_1a_0 = approx_rcp(d4_1a_0);
            float _fma_31;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(-d4_1a_0), "f"(r4_1a_0), "f"(1.0f));
            float _fma_30;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(r4_1a_0), "f"(_fma_31), "f"(r4_1a_0));
            float big_1a_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_0) : "f"(mul_f32_rn_noftz(_fma_30, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_32;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(big_1a_0), "f"(8.388608e+06f));
            float whole_1a_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_0) : "f"(_add_32), "f"(8.388608e+06f));
            float above_1a_0;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_0) : "f"(-d4_1a_0), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_0, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_34;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34) : "f"(big_1a_0), "f"(whole_1a_0));
            float _abs_33;
            asm("abs.f32 %0, %1;" : "=f"(_abs_33) : "f"(_sub_34));
            float _add_35;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35) : "f"(big_1a_0), "f"(0.5f));
            float _sub_36;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_36) : "f"(big_1a_0), "f"(0.5f));
            float rounded_1a_0 = ((_abs_33 == 0.5f) ? ((above_1a_0 > 0.0f) ? _add_35 : _sub_36) : whole_1a_0);
            t1 = ((d1_7 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_0, 7.888609e-31f), 1.7763568e-15f) : t1);
        }
        float2 s_15_1 = make_float2(max_noftz(t0, 0.0f), max_noftz(t1, 0.0f));
        float2 act_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_7) : "l"(*(const unsigned long long*)&x_7), "l"(*(const unsigned long long*)&s_15_1));
        float2 _f32x2_mul_a_37 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_7, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_15_1), s_15_1), g_7);
        float2 mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_37), "l"(*(const unsigned long long*)&u_7));
        float2 mul_1_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_7) : "l"(*(const unsigned long long*)&act_7), "l"(*(const unsigned long long*)&g_7));
        __nv_bfloat162 bf16x2_9 = __float22bfloat162_rn(make_float2(mul_8.x, mul_8.y));
        __nv_bfloat162 bf16x2_1_8 = __float22bfloat162_rn(make_float2(mul_1_7.x, mul_1_7.y));
        dgate[0] = __as_u32(bf16x2_9);
        dup[0] = __as_u32(bf16x2_1_8);
        float2 x_7_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[1]));
        float2 u_7_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[1]));
        float2 g_7_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[1]));
        float2 _f32x2_mul_b_38 = make_float2(-1.442695f, -1.442695f);
        float2 t_7_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_7_1) : "l"(*(const unsigned long long*)&x_7_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_38));
        float e0_7_1 = approx_exp2(t_7_1.x);
        float e1_7_1 = approx_exp2(t_7_1.y);
        float2 nd_7_1 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_7_1, e1_7_1));
        float d0_7_1 = -nd_7_1.x;
        float d1_7_1 = -nd_7_1.y;
        float r0_7_1 = approx_rcp(d0_7_1);
        float r1_7_1 = approx_rcp(d1_7_1);
        float2 r_7_1 = make_float2(r0_7_1, r1_7_1);
        float2 s_16 = fma_f32x2_rn_noftz(r_7_1, fma_f32x2_rn_noftz(nd_7_1, r_7_1, make_float2(1.0f, 1.0f)), r_7_1);
        float t0_0 = s_16.x;
        float t1_0 = s_16.y;
        int _warp_vote_39 = __any_sync(0xFFFFFFFF, fminf(t0_0, t1_0) == 0.0f);
        if (_warp_vote_39 != 0) {
            float d4_1a_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_1) : "f"(d0_7_1), "f"(0.25f));
            float r4_1a_1 = approx_rcp(d4_1a_1);
            float _fma_41;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(-d4_1a_1), "f"(r4_1a_1), "f"(1.0f));
            float _fma_40;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(r4_1a_1), "f"(_fma_41), "f"(r4_1a_1));
            float big_1a_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_1) : "f"(mul_f32_rn_noftz(_fma_40, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_42;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(big_1a_1), "f"(8.388608e+06f));
            float whole_1a_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_1) : "f"(_add_42), "f"(8.388608e+06f));
            float above_1a_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_1) : "f"(-d4_1a_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_44;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44) : "f"(big_1a_1), "f"(whole_1a_1));
            float _abs_43;
            asm("abs.f32 %0, %1;" : "=f"(_abs_43) : "f"(_sub_44));
            float _add_45;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(big_1a_1), "f"(0.5f));
            float _sub_46;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_46) : "f"(big_1a_1), "f"(0.5f));
            float rounded_1a_1 = ((_abs_43 == 0.5f) ? ((above_1a_1 > 0.0f) ? _add_45 : _sub_46) : whole_1a_1);
            t0_0 = ((d0_7_1 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_1, 7.888609e-31f), 1.7763568e-15f) : t0_0);
            float d4_1a_1_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_1_1) : "f"(d1_7_1), "f"(0.25f));
            float r4_1a_1_1 = approx_rcp(d4_1a_1_1);
            float _fma_48;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(-d4_1a_1_1), "f"(r4_1a_1_1), "f"(1.0f));
            float _fma_47;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(r4_1a_1_1), "f"(_fma_48), "f"(r4_1a_1_1));
            float big_1a_1_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_1_1) : "f"(mul_f32_rn_noftz(_fma_47, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_49;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(big_1a_1_1), "f"(8.388608e+06f));
            float whole_1a_1_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_1_1) : "f"(_add_49), "f"(8.388608e+06f));
            float above_1a_1_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_1_1) : "f"(-d4_1a_1_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_1_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_51;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_51) : "f"(big_1a_1_1), "f"(whole_1a_1_1));
            float _abs_50;
            asm("abs.f32 %0, %1;" : "=f"(_abs_50) : "f"(_sub_51));
            float _add_52;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(big_1a_1_1), "f"(0.5f));
            float _sub_53;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_53) : "f"(big_1a_1_1), "f"(0.5f));
            float rounded_1a_1_1 = ((_abs_50 == 0.5f) ? ((above_1a_1_1 > 0.0f) ? _add_52 : _sub_53) : whole_1a_1_1);
            t1_0 = ((d1_7_1 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_1_1, 7.888609e-31f), 1.7763568e-15f) : t1_0);
        }
        float2 s_17 = make_float2(max_noftz(t0_0, 0.0f), max_noftz(t1_0, 0.0f));
        float2 act_7_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_7_1) : "l"(*(const unsigned long long*)&x_7_1), "l"(*(const unsigned long long*)&s_17));
        float2 _f32x2_mul_a_54 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_7_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_17), s_17), g_7_1);
        float2 mul_8_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_8_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_54), "l"(*(const unsigned long long*)&u_7_1));
        float2 mul_1_7_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_7_1) : "l"(*(const unsigned long long*)&act_7_1), "l"(*(const unsigned long long*)&g_7_1));
        __nv_bfloat162 bf16x2_10 = __float22bfloat162_rn(make_float2(mul_8_1.x, mul_8_1.y));
        __nv_bfloat162 bf16x2_1_9 = __float22bfloat162_rn(make_float2(mul_1_7_1.x, mul_1_7_1.y));
        dgate[1] = __as_u32(bf16x2_10);
        dup[1] = __as_u32(bf16x2_1_9);
        float2 x_8 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[2]));
        float2 u_8 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[2]));
        float2 g_8 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[2]));
        float2 _f32x2_mul_b_55 = make_float2(-1.442695f, -1.442695f);
        float2 t_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_8) : "l"(*(const unsigned long long*)&x_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_55));
        float e0_8 = approx_exp2(t_8.x);
        float e1_8 = approx_exp2(t_8.y);
        float2 nd_8 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_8, e1_8));
        float d0_8 = -nd_8.x;
        float d1_8 = -nd_8.y;
        float r0_8 = approx_rcp(d0_8);
        float r1_8 = approx_rcp(d1_8);
        float2 r_8 = make_float2(r0_8, r1_8);
        float2 s_18 = fma_f32x2_rn_noftz(r_8, fma_f32x2_rn_noftz(nd_8, r_8, make_float2(1.0f, 1.0f)), r_8);
        float t0_1 = s_18.x;
        float t1_1 = s_18.y;
        int _warp_vote_56 = __any_sync(0xFFFFFFFF, fminf(t0_1, t1_1) == 0.0f);
        if (_warp_vote_56 != 0) {
            float d4_1a_2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_2) : "f"(d0_8), "f"(0.25f));
            float r4_1a_2 = approx_rcp(d4_1a_2);
            float _fma_58;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(-d4_1a_2), "f"(r4_1a_2), "f"(1.0f));
            float _fma_57;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(r4_1a_2), "f"(_fma_58), "f"(r4_1a_2));
            float big_1a_2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_2) : "f"(mul_f32_rn_noftz(_fma_57, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_59;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(big_1a_2), "f"(8.388608e+06f));
            float whole_1a_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_2) : "f"(_add_59), "f"(8.388608e+06f));
            float above_1a_2;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_2) : "f"(-d4_1a_2), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_2, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_61;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_61) : "f"(big_1a_2), "f"(whole_1a_2));
            float _abs_60;
            asm("abs.f32 %0, %1;" : "=f"(_abs_60) : "f"(_sub_61));
            float _add_62;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_62) : "f"(big_1a_2), "f"(0.5f));
            float _sub_63;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_63) : "f"(big_1a_2), "f"(0.5f));
            float rounded_1a_2 = ((_abs_60 == 0.5f) ? ((above_1a_2 > 0.0f) ? _add_62 : _sub_63) : whole_1a_2);
            t0_1 = ((d0_8 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_2, 7.888609e-31f), 1.7763568e-15f) : t0_1);
            float d4_1a_2_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_2_1) : "f"(d1_8), "f"(0.25f));
            float r4_1a_2_1 = approx_rcp(d4_1a_2_1);
            float _fma_65;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(-d4_1a_2_1), "f"(r4_1a_2_1), "f"(1.0f));
            float _fma_64;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(r4_1a_2_1), "f"(_fma_65), "f"(r4_1a_2_1));
            float big_1a_2_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_2_1) : "f"(mul_f32_rn_noftz(_fma_64, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_66;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_66) : "f"(big_1a_2_1), "f"(8.388608e+06f));
            float whole_1a_2_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_2_1) : "f"(_add_66), "f"(8.388608e+06f));
            float above_1a_2_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_2_1) : "f"(-d4_1a_2_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_2_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_68;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_68) : "f"(big_1a_2_1), "f"(whole_1a_2_1));
            float _abs_67;
            asm("abs.f32 %0, %1;" : "=f"(_abs_67) : "f"(_sub_68));
            float _add_69;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(big_1a_2_1), "f"(0.5f));
            float _sub_70;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_70) : "f"(big_1a_2_1), "f"(0.5f));
            float rounded_1a_2_1 = ((_abs_67 == 0.5f) ? ((above_1a_2_1 > 0.0f) ? _add_69 : _sub_70) : whole_1a_2_1);
            t1_1 = ((d1_8 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_2_1, 7.888609e-31f), 1.7763568e-15f) : t1_1);
        }
        float2 s_19 = make_float2(max_noftz(t0_1, 0.0f), max_noftz(t1_1, 0.0f));
        float2 act_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_8) : "l"(*(const unsigned long long*)&x_8), "l"(*(const unsigned long long*)&s_19));
        float2 _f32x2_mul_a_71 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_8, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_19), s_19), g_8);
        float2 mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_71), "l"(*(const unsigned long long*)&u_8));
        float2 mul_1_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_8) : "l"(*(const unsigned long long*)&act_8), "l"(*(const unsigned long long*)&g_8));
        __nv_bfloat162 bf16x2_11 = __float22bfloat162_rn(make_float2(mul_9.x, mul_9.y));
        __nv_bfloat162 bf16x2_1_10 = __float22bfloat162_rn(make_float2(mul_1_8.x, mul_1_8.y));
        dgate[2] = __as_u32(bf16x2_11);
        dup[2] = __as_u32(bf16x2_1_10);
        float2 x_9 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[3]));
        float2 u_9 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[3]));
        float2 g_9 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[3]));
        float2 _f32x2_mul_b_72 = make_float2(-1.442695f, -1.442695f);
        float2 t_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_9) : "l"(*(const unsigned long long*)&x_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_72));
        float e0_9 = approx_exp2(t_9.x);
        float e1_9 = approx_exp2(t_9.y);
        float2 nd_9 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_9, e1_9));
        float d0_9 = -nd_9.x;
        float d1_9 = -nd_9.y;
        float r0_9 = approx_rcp(d0_9);
        float r1_9 = approx_rcp(d1_9);
        float2 r_9 = make_float2(r0_9, r1_9);
        float2 s_20 = fma_f32x2_rn_noftz(r_9, fma_f32x2_rn_noftz(nd_9, r_9, make_float2(1.0f, 1.0f)), r_9);
        float t0_2 = s_20.x;
        float t1_2 = s_20.y;
        int _warp_vote_73 = __any_sync(0xFFFFFFFF, fminf(t0_2, t1_2) == 0.0f);
        if (_warp_vote_73 != 0) {
            float d4_1a_3;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_3) : "f"(d0_9), "f"(0.25f));
            float r4_1a_3 = approx_rcp(d4_1a_3);
            float _fma_75;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(-d4_1a_3), "f"(r4_1a_3), "f"(1.0f));
            float _fma_74;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(r4_1a_3), "f"(_fma_75), "f"(r4_1a_3));
            float big_1a_3;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_3) : "f"(mul_f32_rn_noftz(_fma_74, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_76;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(big_1a_3), "f"(8.388608e+06f));
            float whole_1a_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_3) : "f"(_add_76), "f"(8.388608e+06f));
            float above_1a_3;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_3) : "f"(-d4_1a_3), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_3, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_78;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_78) : "f"(big_1a_3), "f"(whole_1a_3));
            float _abs_77;
            asm("abs.f32 %0, %1;" : "=f"(_abs_77) : "f"(_sub_78));
            float _add_79;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(big_1a_3), "f"(0.5f));
            float _sub_80;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_80) : "f"(big_1a_3), "f"(0.5f));
            float rounded_1a_3 = ((_abs_77 == 0.5f) ? ((above_1a_3 > 0.0f) ? _add_79 : _sub_80) : whole_1a_3);
            t0_2 = ((d0_9 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_3, 7.888609e-31f), 1.7763568e-15f) : t0_2);
            float d4_1a_3_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_3_1) : "f"(d1_9), "f"(0.25f));
            float r4_1a_3_1 = approx_rcp(d4_1a_3_1);
            float _fma_82;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(-d4_1a_3_1), "f"(r4_1a_3_1), "f"(1.0f));
            float _fma_81;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(r4_1a_3_1), "f"(_fma_82), "f"(r4_1a_3_1));
            float big_1a_3_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_3_1) : "f"(mul_f32_rn_noftz(_fma_81, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_83;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(big_1a_3_1), "f"(8.388608e+06f));
            float whole_1a_3_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_3_1) : "f"(_add_83), "f"(8.388608e+06f));
            float above_1a_3_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_3_1) : "f"(-d4_1a_3_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_3_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_85;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_85) : "f"(big_1a_3_1), "f"(whole_1a_3_1));
            float _abs_84;
            asm("abs.f32 %0, %1;" : "=f"(_abs_84) : "f"(_sub_85));
            float _add_86;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_86) : "f"(big_1a_3_1), "f"(0.5f));
            float _sub_87;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_87) : "f"(big_1a_3_1), "f"(0.5f));
            float rounded_1a_3_1 = ((_abs_84 == 0.5f) ? ((above_1a_3_1 > 0.0f) ? _add_86 : _sub_87) : whole_1a_3_1);
            t1_2 = ((d1_9 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_3_1, 7.888609e-31f), 1.7763568e-15f) : t1_2);
        }
        float2 s_21 = make_float2(max_noftz(t0_2, 0.0f), max_noftz(t1_2, 0.0f));
        float2 act_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_9) : "l"(*(const unsigned long long*)&x_9), "l"(*(const unsigned long long*)&s_21));
        float2 _f32x2_mul_a_88 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_9, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_21), s_21), g_9);
        float2 mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_88), "l"(*(const unsigned long long*)&u_9));
        float2 mul_1_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_9) : "l"(*(const unsigned long long*)&act_9), "l"(*(const unsigned long long*)&g_9));
        __nv_bfloat162 bf16x2_12 = __float22bfloat162_rn(make_float2(mul_10.x, mul_10.y));
        __nv_bfloat162 bf16x2_1_11 = __float22bfloat162_rn(make_float2(mul_1_9.x, mul_1_9.y));
        dgate[3] = __as_u32(bf16x2_12);
        dup[3] = __as_u32(bf16x2_1_11);
        float2 x_10 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[4]));
        float2 u_10 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[4]));
        float2 g_10 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[4]));
        float2 _f32x2_mul_b_89 = make_float2(-1.442695f, -1.442695f);
        float2 t_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_10) : "l"(*(const unsigned long long*)&x_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_89));
        float e0_10 = approx_exp2(t_10.x);
        float e1_10 = approx_exp2(t_10.y);
        float2 nd_10 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_10, e1_10));
        float d0_10 = -nd_10.x;
        float d1_10 = -nd_10.y;
        float r0_10 = approx_rcp(d0_10);
        float r1_10 = approx_rcp(d1_10);
        float2 r_10 = make_float2(r0_10, r1_10);
        float2 s_22 = fma_f32x2_rn_noftz(r_10, fma_f32x2_rn_noftz(nd_10, r_10, make_float2(1.0f, 1.0f)), r_10);
        float t0_3 = s_22.x;
        float t1_3 = s_22.y;
        int _warp_vote_90 = __any_sync(0xFFFFFFFF, fminf(t0_3, t1_3) == 0.0f);
        if (_warp_vote_90 != 0) {
            float d4_1a_4;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_4) : "f"(d0_10), "f"(0.25f));
            float r4_1a_4 = approx_rcp(d4_1a_4);
            float _fma_92;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(-d4_1a_4), "f"(r4_1a_4), "f"(1.0f));
            float _fma_91;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(r4_1a_4), "f"(_fma_92), "f"(r4_1a_4));
            float big_1a_4;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_4) : "f"(mul_f32_rn_noftz(_fma_91, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_93;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(big_1a_4), "f"(8.388608e+06f));
            float whole_1a_4;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_4) : "f"(_add_93), "f"(8.388608e+06f));
            float above_1a_4;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_4) : "f"(-d4_1a_4), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_4, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_95;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_95) : "f"(big_1a_4), "f"(whole_1a_4));
            float _abs_94;
            asm("abs.f32 %0, %1;" : "=f"(_abs_94) : "f"(_sub_95));
            float _add_96;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_96) : "f"(big_1a_4), "f"(0.5f));
            float _sub_97;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_97) : "f"(big_1a_4), "f"(0.5f));
            float rounded_1a_4 = ((_abs_94 == 0.5f) ? ((above_1a_4 > 0.0f) ? _add_96 : _sub_97) : whole_1a_4);
            t0_3 = ((d0_10 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_4, 7.888609e-31f), 1.7763568e-15f) : t0_3);
            float d4_1a_4_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_4_1) : "f"(d1_10), "f"(0.25f));
            float r4_1a_4_1 = approx_rcp(d4_1a_4_1);
            float _fma_99;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(-d4_1a_4_1), "f"(r4_1a_4_1), "f"(1.0f));
            float _fma_98;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(r4_1a_4_1), "f"(_fma_99), "f"(r4_1a_4_1));
            float big_1a_4_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_4_1) : "f"(mul_f32_rn_noftz(_fma_98, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_100;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_100) : "f"(big_1a_4_1), "f"(8.388608e+06f));
            float whole_1a_4_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_4_1) : "f"(_add_100), "f"(8.388608e+06f));
            float above_1a_4_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_4_1) : "f"(-d4_1a_4_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_4_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_102;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_102) : "f"(big_1a_4_1), "f"(whole_1a_4_1));
            float _abs_101;
            asm("abs.f32 %0, %1;" : "=f"(_abs_101) : "f"(_sub_102));
            float _add_103;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_103) : "f"(big_1a_4_1), "f"(0.5f));
            float _sub_104;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_104) : "f"(big_1a_4_1), "f"(0.5f));
            float rounded_1a_4_1 = ((_abs_101 == 0.5f) ? ((above_1a_4_1 > 0.0f) ? _add_103 : _sub_104) : whole_1a_4_1);
            t1_3 = ((d1_10 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_4_1, 7.888609e-31f), 1.7763568e-15f) : t1_3);
        }
        float2 s_23 = make_float2(max_noftz(t0_3, 0.0f), max_noftz(t1_3, 0.0f));
        float2 act_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_10) : "l"(*(const unsigned long long*)&x_10), "l"(*(const unsigned long long*)&s_23));
        float2 _f32x2_mul_a_105 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_10, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_23), s_23), g_10);
        float2 mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_105), "l"(*(const unsigned long long*)&u_10));
        float2 mul_1_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_10) : "l"(*(const unsigned long long*)&act_10), "l"(*(const unsigned long long*)&g_10));
        __nv_bfloat162 bf16x2_13 = __float22bfloat162_rn(make_float2(mul_11.x, mul_11.y));
        __nv_bfloat162 bf16x2_1_12 = __float22bfloat162_rn(make_float2(mul_1_10.x, mul_1_10.y));
        dgate[4] = __as_u32(bf16x2_13);
        dup[4] = __as_u32(bf16x2_1_12);
        float2 x_11 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[5]));
        float2 u_11 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[5]));
        float2 g_11 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[5]));
        float2 _f32x2_mul_b_106 = make_float2(-1.442695f, -1.442695f);
        float2 t_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_11) : "l"(*(const unsigned long long*)&x_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_106));
        float e0_11 = approx_exp2(t_11.x);
        float e1_11 = approx_exp2(t_11.y);
        float2 nd_11 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_11, e1_11));
        float d0_11 = -nd_11.x;
        float d1_11 = -nd_11.y;
        float r0_11 = approx_rcp(d0_11);
        float r1_11 = approx_rcp(d1_11);
        float2 r_11 = make_float2(r0_11, r1_11);
        float2 s_24 = fma_f32x2_rn_noftz(r_11, fma_f32x2_rn_noftz(nd_11, r_11, make_float2(1.0f, 1.0f)), r_11);
        float t0_4 = s_24.x;
        float t1_4 = s_24.y;
        int _warp_vote_107 = __any_sync(0xFFFFFFFF, fminf(t0_4, t1_4) == 0.0f);
        if (_warp_vote_107 != 0) {
            float d4_1a_5;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_5) : "f"(d0_11), "f"(0.25f));
            float r4_1a_5 = approx_rcp(d4_1a_5);
            float _fma_109;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_109) : "f"(-d4_1a_5), "f"(r4_1a_5), "f"(1.0f));
            float _fma_108;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_108) : "f"(r4_1a_5), "f"(_fma_109), "f"(r4_1a_5));
            float big_1a_5;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_5) : "f"(mul_f32_rn_noftz(_fma_108, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_110;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_110) : "f"(big_1a_5), "f"(8.388608e+06f));
            float whole_1a_5;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_5) : "f"(_add_110), "f"(8.388608e+06f));
            float above_1a_5;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_5) : "f"(-d4_1a_5), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_5, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_112;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_112) : "f"(big_1a_5), "f"(whole_1a_5));
            float _abs_111;
            asm("abs.f32 %0, %1;" : "=f"(_abs_111) : "f"(_sub_112));
            float _add_113;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_113) : "f"(big_1a_5), "f"(0.5f));
            float _sub_114;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_114) : "f"(big_1a_5), "f"(0.5f));
            float rounded_1a_5 = ((_abs_111 == 0.5f) ? ((above_1a_5 > 0.0f) ? _add_113 : _sub_114) : whole_1a_5);
            t0_4 = ((d0_11 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_5, 7.888609e-31f), 1.7763568e-15f) : t0_4);
            float d4_1a_5_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_5_1) : "f"(d1_11), "f"(0.25f));
            float r4_1a_5_1 = approx_rcp(d4_1a_5_1);
            float _fma_116;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_116) : "f"(-d4_1a_5_1), "f"(r4_1a_5_1), "f"(1.0f));
            float _fma_115;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_115) : "f"(r4_1a_5_1), "f"(_fma_116), "f"(r4_1a_5_1));
            float big_1a_5_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_5_1) : "f"(mul_f32_rn_noftz(_fma_115, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_117;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_117) : "f"(big_1a_5_1), "f"(8.388608e+06f));
            float whole_1a_5_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_5_1) : "f"(_add_117), "f"(8.388608e+06f));
            float above_1a_5_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_5_1) : "f"(-d4_1a_5_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_5_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_119;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_119) : "f"(big_1a_5_1), "f"(whole_1a_5_1));
            float _abs_118;
            asm("abs.f32 %0, %1;" : "=f"(_abs_118) : "f"(_sub_119));
            float _add_120;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_120) : "f"(big_1a_5_1), "f"(0.5f));
            float _sub_121;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_121) : "f"(big_1a_5_1), "f"(0.5f));
            float rounded_1a_5_1 = ((_abs_118 == 0.5f) ? ((above_1a_5_1 > 0.0f) ? _add_120 : _sub_121) : whole_1a_5_1);
            t1_4 = ((d1_11 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_5_1, 7.888609e-31f), 1.7763568e-15f) : t1_4);
        }
        float2 s_25 = make_float2(max_noftz(t0_4, 0.0f), max_noftz(t1_4, 0.0f));
        float2 act_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_11) : "l"(*(const unsigned long long*)&x_11), "l"(*(const unsigned long long*)&s_25));
        float2 _f32x2_mul_a_122 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_11, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_25), s_25), g_11);
        float2 mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_122), "l"(*(const unsigned long long*)&u_11));
        float2 mul_1_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_11) : "l"(*(const unsigned long long*)&act_11), "l"(*(const unsigned long long*)&g_11));
        __nv_bfloat162 bf16x2_14 = __float22bfloat162_rn(make_float2(mul_12.x, mul_12.y));
        __nv_bfloat162 bf16x2_1_13 = __float22bfloat162_rn(make_float2(mul_1_11.x, mul_1_11.y));
        dgate[5] = __as_u32(bf16x2_14);
        dup[5] = __as_u32(bf16x2_1_13);
        float2 x_12 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[6]));
        float2 u_12 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[6]));
        float2 g_12 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[6]));
        float2 _f32x2_mul_b_123 = make_float2(-1.442695f, -1.442695f);
        float2 t_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_12) : "l"(*(const unsigned long long*)&x_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_123));
        float e0_12 = approx_exp2(t_12.x);
        float e1_12 = approx_exp2(t_12.y);
        float2 nd_12 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_12, e1_12));
        float d0_12 = -nd_12.x;
        float d1_12 = -nd_12.y;
        float r0_12 = approx_rcp(d0_12);
        float r1_12 = approx_rcp(d1_12);
        float2 r_12 = make_float2(r0_12, r1_12);
        float2 s_26 = fma_f32x2_rn_noftz(r_12, fma_f32x2_rn_noftz(nd_12, r_12, make_float2(1.0f, 1.0f)), r_12);
        float t0_5 = s_26.x;
        float t1_5 = s_26.y;
        int _warp_vote_124 = __any_sync(0xFFFFFFFF, fminf(t0_5, t1_5) == 0.0f);
        if (_warp_vote_124 != 0) {
            float d4_1a_6;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_6) : "f"(d0_12), "f"(0.25f));
            float r4_1a_6 = approx_rcp(d4_1a_6);
            float _fma_126;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_126) : "f"(-d4_1a_6), "f"(r4_1a_6), "f"(1.0f));
            float _fma_125;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(r4_1a_6), "f"(_fma_126), "f"(r4_1a_6));
            float big_1a_6;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_6) : "f"(mul_f32_rn_noftz(_fma_125, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_127;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_127) : "f"(big_1a_6), "f"(8.388608e+06f));
            float whole_1a_6;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_6) : "f"(_add_127), "f"(8.388608e+06f));
            float above_1a_6;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_6) : "f"(-d4_1a_6), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_6, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_129;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_129) : "f"(big_1a_6), "f"(whole_1a_6));
            float _abs_128;
            asm("abs.f32 %0, %1;" : "=f"(_abs_128) : "f"(_sub_129));
            float _add_130;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_130) : "f"(big_1a_6), "f"(0.5f));
            float _sub_131;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_131) : "f"(big_1a_6), "f"(0.5f));
            float rounded_1a_6 = ((_abs_128 == 0.5f) ? ((above_1a_6 > 0.0f) ? _add_130 : _sub_131) : whole_1a_6);
            t0_5 = ((d0_12 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_6, 7.888609e-31f), 1.7763568e-15f) : t0_5);
            float d4_1a_6_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_6_1) : "f"(d1_12), "f"(0.25f));
            float r4_1a_6_1 = approx_rcp(d4_1a_6_1);
            float _fma_133;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_133) : "f"(-d4_1a_6_1), "f"(r4_1a_6_1), "f"(1.0f));
            float _fma_132;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_132) : "f"(r4_1a_6_1), "f"(_fma_133), "f"(r4_1a_6_1));
            float big_1a_6_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_6_1) : "f"(mul_f32_rn_noftz(_fma_132, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_134;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_134) : "f"(big_1a_6_1), "f"(8.388608e+06f));
            float whole_1a_6_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_6_1) : "f"(_add_134), "f"(8.388608e+06f));
            float above_1a_6_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_6_1) : "f"(-d4_1a_6_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_6_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_136;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_136) : "f"(big_1a_6_1), "f"(whole_1a_6_1));
            float _abs_135;
            asm("abs.f32 %0, %1;" : "=f"(_abs_135) : "f"(_sub_136));
            float _add_137;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_137) : "f"(big_1a_6_1), "f"(0.5f));
            float _sub_138;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_138) : "f"(big_1a_6_1), "f"(0.5f));
            float rounded_1a_6_1 = ((_abs_135 == 0.5f) ? ((above_1a_6_1 > 0.0f) ? _add_137 : _sub_138) : whole_1a_6_1);
            t1_5 = ((d1_12 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_6_1, 7.888609e-31f), 1.7763568e-15f) : t1_5);
        }
        float2 s_27 = make_float2(max_noftz(t0_5, 0.0f), max_noftz(t1_5, 0.0f));
        float2 act_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_12) : "l"(*(const unsigned long long*)&x_12), "l"(*(const unsigned long long*)&s_27));
        float2 _f32x2_mul_a_139 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_12, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_27), s_27), g_12);
        float2 mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_139), "l"(*(const unsigned long long*)&u_12));
        float2 mul_1_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_12) : "l"(*(const unsigned long long*)&act_12), "l"(*(const unsigned long long*)&g_12));
        __nv_bfloat162 bf16x2_15 = __float22bfloat162_rn(make_float2(mul_13.x, mul_13.y));
        __nv_bfloat162 bf16x2_1_14 = __float22bfloat162_rn(make_float2(mul_1_12.x, mul_1_12.y));
        dgate[6] = __as_u32(bf16x2_15);
        dup[6] = __as_u32(bf16x2_1_14);
        float2 x_13 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[7]));
        float2 u_13 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[7]));
        float2 g_13 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[7]));
        float2 _f32x2_mul_b_140 = make_float2(-1.442695f, -1.442695f);
        float2 t_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_13) : "l"(*(const unsigned long long*)&x_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_140));
        float e0_13 = approx_exp2(t_13.x);
        float e1_13 = approx_exp2(t_13.y);
        float2 nd_13 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_13, e1_13));
        float d0_13 = -nd_13.x;
        float d1_13 = -nd_13.y;
        float r0_13 = approx_rcp(d0_13);
        float r1_13 = approx_rcp(d1_13);
        float2 r_13 = make_float2(r0_13, r1_13);
        float2 s_28 = fma_f32x2_rn_noftz(r_13, fma_f32x2_rn_noftz(nd_13, r_13, make_float2(1.0f, 1.0f)), r_13);
        float t0_6 = s_28.x;
        float t1_6 = s_28.y;
        int _warp_vote_141 = __any_sync(0xFFFFFFFF, fminf(t0_6, t1_6) == 0.0f);
        if (_warp_vote_141 != 0) {
            float d4_1a_7;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_7) : "f"(d0_13), "f"(0.25f));
            float r4_1a_7 = approx_rcp(d4_1a_7);
            float _fma_143;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_143) : "f"(-d4_1a_7), "f"(r4_1a_7), "f"(1.0f));
            float _fma_142;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_142) : "f"(r4_1a_7), "f"(_fma_143), "f"(r4_1a_7));
            float big_1a_7;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_7) : "f"(mul_f32_rn_noftz(_fma_142, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_144;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_144) : "f"(big_1a_7), "f"(8.388608e+06f));
            float whole_1a_7;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_7) : "f"(_add_144), "f"(8.388608e+06f));
            float above_1a_7;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_7) : "f"(-d4_1a_7), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_7, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_146;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_146) : "f"(big_1a_7), "f"(whole_1a_7));
            float _abs_145;
            asm("abs.f32 %0, %1;" : "=f"(_abs_145) : "f"(_sub_146));
            float _add_147;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_147) : "f"(big_1a_7), "f"(0.5f));
            float _sub_148;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_148) : "f"(big_1a_7), "f"(0.5f));
            float rounded_1a_7 = ((_abs_145 == 0.5f) ? ((above_1a_7 > 0.0f) ? _add_147 : _sub_148) : whole_1a_7);
            t0_6 = ((d0_13 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_7, 7.888609e-31f), 1.7763568e-15f) : t0_6);
            float d4_1a_7_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_7_1) : "f"(d1_13), "f"(0.25f));
            float r4_1a_7_1 = approx_rcp(d4_1a_7_1);
            float _fma_150;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_150) : "f"(-d4_1a_7_1), "f"(r4_1a_7_1), "f"(1.0f));
            float _fma_149;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_149) : "f"(r4_1a_7_1), "f"(_fma_150), "f"(r4_1a_7_1));
            float big_1a_7_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_7_1) : "f"(mul_f32_rn_noftz(_fma_149, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_151;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_151) : "f"(big_1a_7_1), "f"(8.388608e+06f));
            float whole_1a_7_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_7_1) : "f"(_add_151), "f"(8.388608e+06f));
            float above_1a_7_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_7_1) : "f"(-d4_1a_7_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_7_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_153;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_153) : "f"(big_1a_7_1), "f"(whole_1a_7_1));
            float _abs_152;
            asm("abs.f32 %0, %1;" : "=f"(_abs_152) : "f"(_sub_153));
            float _add_154;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_154) : "f"(big_1a_7_1), "f"(0.5f));
            float _sub_155;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_155) : "f"(big_1a_7_1), "f"(0.5f));
            float rounded_1a_7_1 = ((_abs_152 == 0.5f) ? ((above_1a_7_1 > 0.0f) ? _add_154 : _sub_155) : whole_1a_7_1);
            t1_6 = ((d1_13 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_7_1, 7.888609e-31f), 1.7763568e-15f) : t1_6);
        }
        float2 s_29 = make_float2(max_noftz(t0_6, 0.0f), max_noftz(t1_6, 0.0f));
        float2 act_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_13) : "l"(*(const unsigned long long*)&x_13), "l"(*(const unsigned long long*)&s_29));
        float2 _f32x2_mul_a_156 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_13, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_29), s_29), g_13);
        float2 mul_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_156), "l"(*(const unsigned long long*)&u_13));
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
    __nv_bfloat162 _hxa_158 = amax;
    __nv_bfloat162 _hxb_158 = __as_bf16x2(dgate[0]);
    unsigned int _hxr_158;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_158) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_158)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_158)));
    __nv_bfloat162 _max_157 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_158);
    amax = _max_157;
    __nv_bfloat162 _hxa_160 = amax;
    __nv_bfloat162 _hxb_160 = __as_bf16x2(dgate[1]);
    unsigned int _hxr_160;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_160) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_160)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_160)));
    __nv_bfloat162 _max_159 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_160);
    amax = _max_159;
    __nv_bfloat162 _hxa_162 = amax;
    __nv_bfloat162 _hxb_162 = __as_bf16x2(dgate[2]);
    unsigned int _hxr_162;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_162) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_162)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_162)));
    __nv_bfloat162 _max_161 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_162);
    amax = _max_161;
    __nv_bfloat162 _hxa_164 = amax;
    __nv_bfloat162 _hxb_164 = __as_bf16x2(dgate[3]);
    unsigned int _hxr_164;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_164) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_164)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_164)));
    __nv_bfloat162 _max_163 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_164);
    amax = _max_163;
    __nv_bfloat162 _hxa_166 = amax;
    __nv_bfloat162 _hxb_166 = __as_bf16x2(dgate[4]);
    unsigned int _hxr_166;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_166) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_166)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_166)));
    __nv_bfloat162 _max_165 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_166);
    amax = _max_165;
    __nv_bfloat162 _hxa_168 = amax;
    __nv_bfloat162 _hxb_168 = __as_bf16x2(dgate[5]);
    unsigned int _hxr_168;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_168) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_168)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_168)));
    __nv_bfloat162 _max_167 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_168);
    amax = _max_167;
    __nv_bfloat162 _hxa_170 = amax;
    __nv_bfloat162 _hxb_170 = __as_bf16x2(dgate[6]);
    unsigned int _hxr_170;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_170) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_170)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_170)));
    __nv_bfloat162 _max_169 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_170);
    amax = _max_169;
    __nv_bfloat162 _hxa_172 = amax;
    __nv_bfloat162 _hxb_172 = __as_bf16x2(dgate[7]);
    unsigned int _hxr_172;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_172) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_172)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_172)));
    __nv_bfloat162 _max_171 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_172);
    amax = _max_171;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_174;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_174) : "r"(bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_174) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_173;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_173) : "h"(e4m3x2_bf16x2), "h"(_cvt_satfinite_e4m3x2_bf16x2_174));
    words[0] = _mov_b32_pack_u16x2_173;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_176;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_176) : "r"(bf16x2_mul_1_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_176) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_175;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_175) : "h"(e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_176));
    words[1] = _mov_b32_pack_u16x2_175;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_178;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_178) : "r"(bf16x2_mul_1_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_178) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_177;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_177) : "h"(e4m3x2_bf16x2_4), "h"(_cvt_satfinite_e4m3x2_bf16x2_178));
    words[2] = _mov_b32_pack_u16x2_177;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_180;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_180) : "r"(bf16x2_mul_1_2));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_2);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_180) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_179;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_179) : "h"(e4m3x2_bf16x2_5), "h"(_cvt_satfinite_e4m3x2_bf16x2_180));
    words[3] = _mov_b32_pack_u16x2_179;
    reinterpret_cast<int4*>(row_output + out_index)[0] = reinterpret_cast<int4*>(words + 0)[0];
    __nv_bfloat162 amax_0 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_182 = amax_0;
    __nv_bfloat162 _hxb_182 = __as_bf16x2(dup[0]);
    unsigned int _hxr_182;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_182) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_182)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_182)));
    __nv_bfloat162 _max_181 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_182);
    amax_0 = _max_181;
    __nv_bfloat162 _hxa_184 = amax_0;
    __nv_bfloat162 _hxb_184 = __as_bf16x2(dup[1]);
    unsigned int _hxr_184;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_184) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_184)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_184)));
    __nv_bfloat162 _max_183 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_184);
    amax_0 = _max_183;
    __nv_bfloat162 _hxa_186 = amax_0;
    __nv_bfloat162 _hxb_186 = __as_bf16x2(dup[2]);
    unsigned int _hxr_186;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_186) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_186)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_186)));
    __nv_bfloat162 _max_185 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_186);
    amax_0 = _max_185;
    __nv_bfloat162 _hxa_188 = amax_0;
    __nv_bfloat162 _hxb_188 = __as_bf16x2(dup[3]);
    unsigned int _hxr_188;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_188) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_188)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_188)));
    __nv_bfloat162 _max_187 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_188);
    amax_0 = _max_187;
    __nv_bfloat162 _hxa_190 = amax_0;
    __nv_bfloat162 _hxb_190 = __as_bf16x2(dup[4]);
    unsigned int _hxr_190;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_190) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_190)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_190)));
    __nv_bfloat162 _max_189 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_190);
    amax_0 = _max_189;
    __nv_bfloat162 _hxa_192 = amax_0;
    __nv_bfloat162 _hxb_192 = __as_bf16x2(dup[5]);
    unsigned int _hxr_192;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_192) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_192)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_192)));
    __nv_bfloat162 _max_191 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_192);
    amax_0 = _max_191;
    __nv_bfloat162 _hxa_194 = amax_0;
    __nv_bfloat162 _hxb_194 = __as_bf16x2(dup[6]);
    unsigned int _hxr_194;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_194) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_194)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_194)));
    __nv_bfloat162 _max_193 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_194);
    amax_0 = _max_193;
    __nv_bfloat162 _hxa_196 = amax_0;
    __nv_bfloat162 _hxb_196 = __as_bf16x2(dup[7]);
    unsigned int _hxr_196;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_196) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_196)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_196)));
    __nv_bfloat162 _max_195 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_196);
    amax_0 = _max_195;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_198;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_198) : "r"(bf16x2_mul_1_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_198) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_197;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_197) : "h"(e4m3x2_bf16x2_6), "h"(_cvt_satfinite_e4m3x2_bf16x2_198));
    words_0[0] = _mov_b32_pack_u16x2_197;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_200;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_200) : "r"(bf16x2_mul_1_4));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_4);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_200) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_199;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_199) : "h"(e4m3x2_bf16x2_7), "h"(_cvt_satfinite_e4m3x2_bf16x2_200));
    words_0[1] = _mov_b32_pack_u16x2_199;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_202;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_202) : "r"(bf16x2_mul_1_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_202) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_201;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_201) : "h"(e4m3x2_bf16x2_8), "h"(_cvt_satfinite_e4m3x2_bf16x2_202));
    words_0[2] = _mov_b32_pack_u16x2_201;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_204;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_204) : "r"(bf16x2_mul_1_6));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_6);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_204) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_203;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_203) : "h"(e4m3x2_bf16x2_9), "h"(_cvt_satfinite_e4m3x2_bf16x2_204));
    words_0[3] = _mov_b32_pack_u16x2_203;
    reinterpret_cast<int4*>(row_output + (out_index + K))[0] = reinterpret_cast<int4*>(words_0 + 0)[0];
}

} // extern "C"
