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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_gated_act_mxfp8_7ed7c74a790b9b536276(__nv_bfloat16* __restrict__ gated_input, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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
        float2 _f32x2_mul_b_4 = make_float2(-1.442695f, -1.442695f);
        float2 t;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t) : "l"(*(const unsigned long long*)&cvt_f32), "l"(*(const unsigned long long*)&_f32x2_mul_b_4));
        float e0 = approx_exp2(t.x);
        float e1 = approx_exp2(t.y);
        float2 nd = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0, e1));
        float d0 = -nd.x;
        float d1 = -nd.y;
        float r0 = approx_rcp(d0);
        float r1 = approx_rcp(d1);
        float2 r = make_float2(r0, r1);
        float2 s = fma_f32x2_rn_noftz(r, fma_f32x2_rn_noftz(nd, r, make_float2(1.0f, 1.0f)), r);
        float2 _f32x2_mul_a_5 = mul_f32x2_noftz(cvt_f32, s);
        float2 mul;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul) : "l"(*(const unsigned long long*)&_f32x2_mul_a_5), "l"(*(const unsigned long long*)&cvt_f32_1));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(mul.x, mul.y)));
        float2 cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 cvt_f32_1_0 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 _f32x2_mul_b_6 = make_float2(-1.442695f, -1.442695f);
        float2 t_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_0) : "l"(*(const unsigned long long*)&cvt_f32_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_6));
        float e0_0 = approx_exp2(t_0.x);
        float e1_0 = approx_exp2(t_0.y);
        float2 nd_0 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_0, e1_0));
        float d0_0 = -nd_0.x;
        float d1_0 = -nd_0.y;
        float r0_0 = approx_rcp(d0_0);
        float r1_0 = approx_rcp(d1_0);
        float2 r_0 = make_float2(r0_0, r1_0);
        float2 s_0 = fma_f32x2_rn_noftz(r_0, fma_f32x2_rn_noftz(nd_0, r_0, make_float2(1.0f, 1.0f)), r_0);
        float2 _f32x2_mul_a_7 = mul_f32x2_noftz(cvt_f32_0, s_0);
        float2 mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_7), "l"(*(const unsigned long long*)&cvt_f32_1_0));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_0.x, mul_0.y)));
        float2 cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 cvt_f32_1_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 _f32x2_mul_b_8 = make_float2(-1.442695f, -1.442695f);
        float2 t_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_1) : "l"(*(const unsigned long long*)&cvt_f32_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_8));
        float e0_1 = approx_exp2(t_1.x);
        float e1_1 = approx_exp2(t_1.y);
        float2 nd_1 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_1, e1_1));
        float d0_1 = -nd_1.x;
        float d1_1 = -nd_1.y;
        float r0_1 = approx_rcp(d0_1);
        float r1_1 = approx_rcp(d1_1);
        float2 r_1 = make_float2(r0_1, r1_1);
        float2 s_1 = fma_f32x2_rn_noftz(r_1, fma_f32x2_rn_noftz(nd_1, r_1, make_float2(1.0f, 1.0f)), r_1);
        float2 _f32x2_mul_a_9 = mul_f32x2_noftz(cvt_f32_2, s_1);
        float2 mul_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_9), "l"(*(const unsigned long long*)&cvt_f32_1_1));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_1.x, mul_1.y)));
        float2 cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 cvt_f32_1_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 _f32x2_mul_b_10 = make_float2(-1.442695f, -1.442695f);
        float2 t_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_2) : "l"(*(const unsigned long long*)&cvt_f32_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_10));
        float e0_2 = approx_exp2(t_2.x);
        float e1_2 = approx_exp2(t_2.y);
        float2 nd_2 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_2, e1_2));
        float d0_2 = -nd_2.x;
        float d1_2 = -nd_2.y;
        float r0_2 = approx_rcp(d0_2);
        float r1_2 = approx_rcp(d1_2);
        float2 r_2 = make_float2(r0_2, r1_2);
        float2 s_2 = fma_f32x2_rn_noftz(r_2, fma_f32x2_rn_noftz(nd_2, r_2, make_float2(1.0f, 1.0f)), r_2);
        float2 _f32x2_mul_a_11 = mul_f32x2_noftz(cvt_f32_3, s_2);
        float2 mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_11), "l"(*(const unsigned long long*)&cvt_f32_1_2));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_2.x, mul_2.y)));
        float2 cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 cvt_f32_1_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float2 _f32x2_mul_b_12 = make_float2(-1.442695f, -1.442695f);
        float2 t_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_3) : "l"(*(const unsigned long long*)&cvt_f32_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_12));
        float e0_3 = approx_exp2(t_3.x);
        float e1_3 = approx_exp2(t_3.y);
        float2 nd_3 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_3, e1_3));
        float d0_3 = -nd_3.x;
        float d1_3 = -nd_3.y;
        float r0_3 = approx_rcp(d0_3);
        float r1_3 = approx_rcp(d1_3);
        float2 r_3 = make_float2(r0_3, r1_3);
        float2 s_3 = fma_f32x2_rn_noftz(r_3, fma_f32x2_rn_noftz(nd_3, r_3, make_float2(1.0f, 1.0f)), r_3);
        float2 _f32x2_mul_a_13 = mul_f32x2_noftz(cvt_f32_4, s_3);
        float2 mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_13), "l"(*(const unsigned long long*)&cvt_f32_1_3));
        packed[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_3.x, mul_3.y)));
        float2 cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 cvt_f32_1_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float2 _f32x2_mul_b_14 = make_float2(-1.442695f, -1.442695f);
        float2 t_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_4) : "l"(*(const unsigned long long*)&cvt_f32_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_14));
        float e0_4 = approx_exp2(t_4.x);
        float e1_4 = approx_exp2(t_4.y);
        float2 nd_4 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_4, e1_4));
        float d0_4 = -nd_4.x;
        float d1_4 = -nd_4.y;
        float r0_4 = approx_rcp(d0_4);
        float r1_4 = approx_rcp(d1_4);
        float2 r_4 = make_float2(r0_4, r1_4);
        float2 s_4 = fma_f32x2_rn_noftz(r_4, fma_f32x2_rn_noftz(nd_4, r_4, make_float2(1.0f, 1.0f)), r_4);
        float2 _f32x2_mul_a_15 = mul_f32x2_noftz(cvt_f32_5, s_4);
        float2 mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&cvt_f32_1_4));
        packed[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_4.x, mul_4.y)));
        float2 cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 cvt_f32_1_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float2 _f32x2_mul_b_16 = make_float2(-1.442695f, -1.442695f);
        float2 t_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_5) : "l"(*(const unsigned long long*)&cvt_f32_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_16));
        float e0_5 = approx_exp2(t_5.x);
        float e1_5 = approx_exp2(t_5.y);
        float2 nd_5 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_5, e1_5));
        float d0_5 = -nd_5.x;
        float d1_5 = -nd_5.y;
        float r0_5 = approx_rcp(d0_5);
        float r1_5 = approx_rcp(d1_5);
        float2 r_5 = make_float2(r0_5, r1_5);
        float2 s_5 = fma_f32x2_rn_noftz(r_5, fma_f32x2_rn_noftz(nd_5, r_5, make_float2(1.0f, 1.0f)), r_5);
        float2 _f32x2_mul_a_17 = mul_f32x2_noftz(cvt_f32_6, s_5);
        float2 mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&cvt_f32_1_5));
        packed[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_5.x, mul_5.y)));
        float2 cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 cvt_f32_1_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float2 _f32x2_mul_b_18 = make_float2(-1.442695f, -1.442695f);
        float2 t_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_6) : "l"(*(const unsigned long long*)&cvt_f32_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_18));
        float e0_6 = approx_exp2(t_6.x);
        float e1_6 = approx_exp2(t_6.y);
        float2 nd_6 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_6, e1_6));
        float d0_6 = -nd_6.x;
        float d1_6 = -nd_6.y;
        float r0_6 = approx_rcp(d0_6);
        float r1_6 = approx_rcp(d1_6);
        float2 r_6 = make_float2(r0_6, r1_6);
        float2 s_6 = fma_f32x2_rn_noftz(r_6, fma_f32x2_rn_noftz(nd_6, r_6, make_float2(1.0f, 1.0f)), r_6);
        float2 _f32x2_mul_a_19 = mul_f32x2_noftz(cvt_f32_7, s_6);
        float2 mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&cvt_f32_1_6));
        packed[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_6.x, mul_6.y)));
    } else {
        int band = 0;
        float2 cvt_f32_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 cvt_f32_1_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 _f32x2_mul_b_20 = make_float2(-1.442695f, -1.442695f);
        float2 t_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_7) : "l"(*(const unsigned long long*)&cvt_f32_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
        float e0_7 = approx_exp2(t_7.x);
        float e1_7 = approx_exp2(t_7.y);
        float2 nd_7 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_7, e1_7));
        float d0_7 = -nd_7.x;
        float d1_7 = -nd_7.y;
        float r0_7 = approx_rcp(d0_7);
        float r1_7 = approx_rcp(d1_7);
        float2 r_7 = make_float2(r0_7, r1_7);
        float2 s_7 = fma_f32x2_rn_noftz(r_7, fma_f32x2_rn_noftz(nd_7, r_7, make_float2(1.0f, 1.0f)), r_7);
        float t0 = s_7.x;
        float t1 = s_7.y;
        int _warp_vote_21 = __any_sync(0xFFFFFFFF, fminf(t0, t1) == 0.0f);
        if (_warp_vote_21 != 0) {
            float d4_8a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a) : "f"(d0_7), "f"(0.25f));
            float r4_8a = approx_rcp(d4_8a);
            float _fma_23;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(-d4_8a), "f"(r4_8a), "f"(1.0f));
            float _fma_22;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(r4_8a), "f"(_fma_23), "f"(r4_8a));
            float big_8a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a) : "f"(mul_f32_rn_noftz(_fma_22, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_24;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(big_8a), "f"(8.388608e+06f));
            float whole_8a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a) : "f"(_add_24), "f"(8.388608e+06f));
            float above_8a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a) : "f"(-d4_8a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_26;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26) : "f"(big_8a), "f"(whole_8a));
            float _abs_25;
            asm("abs.f32 %0, %1;" : "=f"(_abs_25) : "f"(_sub_26));
            float _add_27;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(big_8a), "f"(0.5f));
            float _sub_28;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_28) : "f"(big_8a), "f"(0.5f));
            float rounded_8a = ((_abs_25 == 0.5f) ? ((above_8a > 0.0f) ? _add_27 : _sub_28) : whole_8a);
            t0 = ((d0_7 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a, 7.888609e-31f), 1.7763568e-15f) : t0);
            float d4_8a_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_0) : "f"(d1_7), "f"(0.25f));
            float r4_8a_0 = approx_rcp(d4_8a_0);
            float _fma_30;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(-d4_8a_0), "f"(r4_8a_0), "f"(1.0f));
            float _fma_29;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(r4_8a_0), "f"(_fma_30), "f"(r4_8a_0));
            float big_8a_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_0) : "f"(mul_f32_rn_noftz(_fma_29, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_31;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(big_8a_0), "f"(8.388608e+06f));
            float whole_8a_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_0) : "f"(_add_31), "f"(8.388608e+06f));
            float above_8a_0;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_0) : "f"(-d4_8a_0), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_0, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_33;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_33) : "f"(big_8a_0), "f"(whole_8a_0));
            float _abs_32;
            asm("abs.f32 %0, %1;" : "=f"(_abs_32) : "f"(_sub_33));
            float _add_34;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(big_8a_0), "f"(0.5f));
            float _sub_35;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_35) : "f"(big_8a_0), "f"(0.5f));
            float rounded_8a_0 = ((_abs_32 == 0.5f) ? ((above_8a_0 > 0.0f) ? _add_34 : _sub_35) : whole_8a_0);
            t1 = ((d1_7 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_0, 7.888609e-31f), 1.7763568e-15f) : t1);
        }
        float2 _f32x2_mul_a_36 = mul_f32x2_noftz(cvt_f32_8, make_float2(max_noftz(t0, 0.0f), max_noftz(t1, 0.0f)));
        float2 mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_36), "l"(*(const unsigned long long*)&cvt_f32_1_7));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_7.x, mul_7.y)));
        float2 cvt_f32_8_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 cvt_f32_1_7_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 _f32x2_mul_b_37 = make_float2(-1.442695f, -1.442695f);
        float2 t_7_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_7_1) : "l"(*(const unsigned long long*)&cvt_f32_8_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_37));
        float e0_7_1 = approx_exp2(t_7_1.x);
        float e1_7_1 = approx_exp2(t_7_1.y);
        float2 nd_7_1 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_7_1, e1_7_1));
        float d0_7_1 = -nd_7_1.x;
        float d1_7_1 = -nd_7_1.y;
        float r0_7_1 = approx_rcp(d0_7_1);
        float r1_7_1 = approx_rcp(d1_7_1);
        float2 r_7_1 = make_float2(r0_7_1, r1_7_1);
        float2 s_7_1 = fma_f32x2_rn_noftz(r_7_1, fma_f32x2_rn_noftz(nd_7_1, r_7_1, make_float2(1.0f, 1.0f)), r_7_1);
        float t0_0 = s_7_1.x;
        float t1_0 = s_7_1.y;
        int _warp_vote_38 = __any_sync(0xFFFFFFFF, fminf(t0_0, t1_0) == 0.0f);
        if (_warp_vote_38 != 0) {
            float d4_8a_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_1) : "f"(d0_7_1), "f"(0.25f));
            float r4_8a_1 = approx_rcp(d4_8a_1);
            float _fma_40;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(-d4_8a_1), "f"(r4_8a_1), "f"(1.0f));
            float _fma_39;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(r4_8a_1), "f"(_fma_40), "f"(r4_8a_1));
            float big_8a_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_1) : "f"(mul_f32_rn_noftz(_fma_39, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_41;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(big_8a_1), "f"(8.388608e+06f));
            float whole_8a_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_1) : "f"(_add_41), "f"(8.388608e+06f));
            float above_8a_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_1) : "f"(-d4_8a_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_43;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43) : "f"(big_8a_1), "f"(whole_8a_1));
            float _abs_42;
            asm("abs.f32 %0, %1;" : "=f"(_abs_42) : "f"(_sub_43));
            float _add_44;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(big_8a_1), "f"(0.5f));
            float _sub_45;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_45) : "f"(big_8a_1), "f"(0.5f));
            float rounded_8a_1 = ((_abs_42 == 0.5f) ? ((above_8a_1 > 0.0f) ? _add_44 : _sub_45) : whole_8a_1);
            t0_0 = ((d0_7_1 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_1, 7.888609e-31f), 1.7763568e-15f) : t0_0);
            float d4_8a_1_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_1_1) : "f"(d1_7_1), "f"(0.25f));
            float r4_8a_1_1 = approx_rcp(d4_8a_1_1);
            float _fma_47;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(-d4_8a_1_1), "f"(r4_8a_1_1), "f"(1.0f));
            float _fma_46;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(r4_8a_1_1), "f"(_fma_47), "f"(r4_8a_1_1));
            float big_8a_1_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_1_1) : "f"(mul_f32_rn_noftz(_fma_46, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_48;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(big_8a_1_1), "f"(8.388608e+06f));
            float whole_8a_1_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_1_1) : "f"(_add_48), "f"(8.388608e+06f));
            float above_8a_1_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_1_1) : "f"(-d4_8a_1_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_1_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_50;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_50) : "f"(big_8a_1_1), "f"(whole_8a_1_1));
            float _abs_49;
            asm("abs.f32 %0, %1;" : "=f"(_abs_49) : "f"(_sub_50));
            float _add_51;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(big_8a_1_1), "f"(0.5f));
            float _sub_52;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_52) : "f"(big_8a_1_1), "f"(0.5f));
            float rounded_8a_1_1 = ((_abs_49 == 0.5f) ? ((above_8a_1_1 > 0.0f) ? _add_51 : _sub_52) : whole_8a_1_1);
            t1_0 = ((d1_7_1 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_1_1, 7.888609e-31f), 1.7763568e-15f) : t1_0);
        }
        float2 _f32x2_mul_a_53 = mul_f32x2_noftz(cvt_f32_8_1, make_float2(max_noftz(t0_0, 0.0f), max_noftz(t1_0, 0.0f)));
        float2 mul_7_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_7_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_53), "l"(*(const unsigned long long*)&cvt_f32_1_7_1));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_7_1.x, mul_7_1.y)));
        float2 cvt_f32_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 cvt_f32_1_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 _f32x2_mul_b_54 = make_float2(-1.442695f, -1.442695f);
        float2 t_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_8) : "l"(*(const unsigned long long*)&cvt_f32_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_54));
        float e0_8 = approx_exp2(t_8.x);
        float e1_8 = approx_exp2(t_8.y);
        float2 nd_8 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_8, e1_8));
        float d0_8 = -nd_8.x;
        float d1_8 = -nd_8.y;
        float r0_8 = approx_rcp(d0_8);
        float r1_8 = approx_rcp(d1_8);
        float2 r_8 = make_float2(r0_8, r1_8);
        float2 s_8 = fma_f32x2_rn_noftz(r_8, fma_f32x2_rn_noftz(nd_8, r_8, make_float2(1.0f, 1.0f)), r_8);
        float t0_1 = s_8.x;
        float t1_1 = s_8.y;
        int _warp_vote_55 = __any_sync(0xFFFFFFFF, fminf(t0_1, t1_1) == 0.0f);
        if (_warp_vote_55 != 0) {
            float d4_8a_2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_2) : "f"(d0_8), "f"(0.25f));
            float r4_8a_2 = approx_rcp(d4_8a_2);
            float _fma_57;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(-d4_8a_2), "f"(r4_8a_2), "f"(1.0f));
            float _fma_56;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(r4_8a_2), "f"(_fma_57), "f"(r4_8a_2));
            float big_8a_2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_2) : "f"(mul_f32_rn_noftz(_fma_56, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_58;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_58) : "f"(big_8a_2), "f"(8.388608e+06f));
            float whole_8a_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_2) : "f"(_add_58), "f"(8.388608e+06f));
            float above_8a_2;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_2) : "f"(-d4_8a_2), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_2, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_60;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_60) : "f"(big_8a_2), "f"(whole_8a_2));
            float _abs_59;
            asm("abs.f32 %0, %1;" : "=f"(_abs_59) : "f"(_sub_60));
            float _add_61;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_61) : "f"(big_8a_2), "f"(0.5f));
            float _sub_62;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_62) : "f"(big_8a_2), "f"(0.5f));
            float rounded_8a_2 = ((_abs_59 == 0.5f) ? ((above_8a_2 > 0.0f) ? _add_61 : _sub_62) : whole_8a_2);
            t0_1 = ((d0_8 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_2, 7.888609e-31f), 1.7763568e-15f) : t0_1);
            float d4_8a_2_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_2_1) : "f"(d1_8), "f"(0.25f));
            float r4_8a_2_1 = approx_rcp(d4_8a_2_1);
            float _fma_64;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(-d4_8a_2_1), "f"(r4_8a_2_1), "f"(1.0f));
            float _fma_63;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(r4_8a_2_1), "f"(_fma_64), "f"(r4_8a_2_1));
            float big_8a_2_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_2_1) : "f"(mul_f32_rn_noftz(_fma_63, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_65;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_65) : "f"(big_8a_2_1), "f"(8.388608e+06f));
            float whole_8a_2_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_2_1) : "f"(_add_65), "f"(8.388608e+06f));
            float above_8a_2_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_2_1) : "f"(-d4_8a_2_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_2_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_67;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_67) : "f"(big_8a_2_1), "f"(whole_8a_2_1));
            float _abs_66;
            asm("abs.f32 %0, %1;" : "=f"(_abs_66) : "f"(_sub_67));
            float _add_68;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(big_8a_2_1), "f"(0.5f));
            float _sub_69;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_69) : "f"(big_8a_2_1), "f"(0.5f));
            float rounded_8a_2_1 = ((_abs_66 == 0.5f) ? ((above_8a_2_1 > 0.0f) ? _add_68 : _sub_69) : whole_8a_2_1);
            t1_1 = ((d1_8 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_2_1, 7.888609e-31f), 1.7763568e-15f) : t1_1);
        }
        float2 _f32x2_mul_a_70 = mul_f32x2_noftz(cvt_f32_9, make_float2(max_noftz(t0_1, 0.0f), max_noftz(t1_1, 0.0f)));
        float2 mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&cvt_f32_1_8));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_8.x, mul_8.y)));
        float2 cvt_f32_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 cvt_f32_1_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 _f32x2_mul_b_71 = make_float2(-1.442695f, -1.442695f);
        float2 t_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_9) : "l"(*(const unsigned long long*)&cvt_f32_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_71));
        float e0_9 = approx_exp2(t_9.x);
        float e1_9 = approx_exp2(t_9.y);
        float2 nd_9 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_9, e1_9));
        float d0_9 = -nd_9.x;
        float d1_9 = -nd_9.y;
        float r0_9 = approx_rcp(d0_9);
        float r1_9 = approx_rcp(d1_9);
        float2 r_9 = make_float2(r0_9, r1_9);
        float2 s_9 = fma_f32x2_rn_noftz(r_9, fma_f32x2_rn_noftz(nd_9, r_9, make_float2(1.0f, 1.0f)), r_9);
        float t0_2 = s_9.x;
        float t1_2 = s_9.y;
        int _warp_vote_72 = __any_sync(0xFFFFFFFF, fminf(t0_2, t1_2) == 0.0f);
        if (_warp_vote_72 != 0) {
            float d4_8a_3;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_3) : "f"(d0_9), "f"(0.25f));
            float r4_8a_3 = approx_rcp(d4_8a_3);
            float _fma_74;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(-d4_8a_3), "f"(r4_8a_3), "f"(1.0f));
            float _fma_73;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(r4_8a_3), "f"(_fma_74), "f"(r4_8a_3));
            float big_8a_3;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_3) : "f"(mul_f32_rn_noftz(_fma_73, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_75;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_75) : "f"(big_8a_3), "f"(8.388608e+06f));
            float whole_8a_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_3) : "f"(_add_75), "f"(8.388608e+06f));
            float above_8a_3;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_3) : "f"(-d4_8a_3), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_3, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_77;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_77) : "f"(big_8a_3), "f"(whole_8a_3));
            float _abs_76;
            asm("abs.f32 %0, %1;" : "=f"(_abs_76) : "f"(_sub_77));
            float _add_78;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_78) : "f"(big_8a_3), "f"(0.5f));
            float _sub_79;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_79) : "f"(big_8a_3), "f"(0.5f));
            float rounded_8a_3 = ((_abs_76 == 0.5f) ? ((above_8a_3 > 0.0f) ? _add_78 : _sub_79) : whole_8a_3);
            t0_2 = ((d0_9 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_3, 7.888609e-31f), 1.7763568e-15f) : t0_2);
            float d4_8a_3_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_3_1) : "f"(d1_9), "f"(0.25f));
            float r4_8a_3_1 = approx_rcp(d4_8a_3_1);
            float _fma_81;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(-d4_8a_3_1), "f"(r4_8a_3_1), "f"(1.0f));
            float _fma_80;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(r4_8a_3_1), "f"(_fma_81), "f"(r4_8a_3_1));
            float big_8a_3_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_3_1) : "f"(mul_f32_rn_noftz(_fma_80, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_82;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_82) : "f"(big_8a_3_1), "f"(8.388608e+06f));
            float whole_8a_3_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_3_1) : "f"(_add_82), "f"(8.388608e+06f));
            float above_8a_3_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_3_1) : "f"(-d4_8a_3_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_3_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_84;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_84) : "f"(big_8a_3_1), "f"(whole_8a_3_1));
            float _abs_83;
            asm("abs.f32 %0, %1;" : "=f"(_abs_83) : "f"(_sub_84));
            float _add_85;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_85) : "f"(big_8a_3_1), "f"(0.5f));
            float _sub_86;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_86) : "f"(big_8a_3_1), "f"(0.5f));
            float rounded_8a_3_1 = ((_abs_83 == 0.5f) ? ((above_8a_3_1 > 0.0f) ? _add_85 : _sub_86) : whole_8a_3_1);
            t1_2 = ((d1_9 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_3_1, 7.888609e-31f), 1.7763568e-15f) : t1_2);
        }
        float2 _f32x2_mul_a_87 = mul_f32x2_noftz(cvt_f32_10, make_float2(max_noftz(t0_2, 0.0f), max_noftz(t1_2, 0.0f)));
        float2 mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_87), "l"(*(const unsigned long long*)&cvt_f32_1_9));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_9.x, mul_9.y)));
        float2 cvt_f32_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 cvt_f32_1_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float2 _f32x2_mul_b_88 = make_float2(-1.442695f, -1.442695f);
        float2 t_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_10) : "l"(*(const unsigned long long*)&cvt_f32_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_88));
        float e0_10 = approx_exp2(t_10.x);
        float e1_10 = approx_exp2(t_10.y);
        float2 nd_10 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_10, e1_10));
        float d0_10 = -nd_10.x;
        float d1_10 = -nd_10.y;
        float r0_10 = approx_rcp(d0_10);
        float r1_10 = approx_rcp(d1_10);
        float2 r_10 = make_float2(r0_10, r1_10);
        float2 s_10 = fma_f32x2_rn_noftz(r_10, fma_f32x2_rn_noftz(nd_10, r_10, make_float2(1.0f, 1.0f)), r_10);
        float t0_3 = s_10.x;
        float t1_3 = s_10.y;
        int _warp_vote_89 = __any_sync(0xFFFFFFFF, fminf(t0_3, t1_3) == 0.0f);
        if (_warp_vote_89 != 0) {
            float d4_8a_4;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_4) : "f"(d0_10), "f"(0.25f));
            float r4_8a_4 = approx_rcp(d4_8a_4);
            float _fma_91;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(-d4_8a_4), "f"(r4_8a_4), "f"(1.0f));
            float _fma_90;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(r4_8a_4), "f"(_fma_91), "f"(r4_8a_4));
            float big_8a_4;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_4) : "f"(mul_f32_rn_noftz(_fma_90, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_92;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(big_8a_4), "f"(8.388608e+06f));
            float whole_8a_4;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_4) : "f"(_add_92), "f"(8.388608e+06f));
            float above_8a_4;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_4) : "f"(-d4_8a_4), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_4, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_94;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_94) : "f"(big_8a_4), "f"(whole_8a_4));
            float _abs_93;
            asm("abs.f32 %0, %1;" : "=f"(_abs_93) : "f"(_sub_94));
            float _add_95;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_95) : "f"(big_8a_4), "f"(0.5f));
            float _sub_96;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_96) : "f"(big_8a_4), "f"(0.5f));
            float rounded_8a_4 = ((_abs_93 == 0.5f) ? ((above_8a_4 > 0.0f) ? _add_95 : _sub_96) : whole_8a_4);
            t0_3 = ((d0_10 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_4, 7.888609e-31f), 1.7763568e-15f) : t0_3);
            float d4_8a_4_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_4_1) : "f"(d1_10), "f"(0.25f));
            float r4_8a_4_1 = approx_rcp(d4_8a_4_1);
            float _fma_98;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(-d4_8a_4_1), "f"(r4_8a_4_1), "f"(1.0f));
            float _fma_97;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(r4_8a_4_1), "f"(_fma_98), "f"(r4_8a_4_1));
            float big_8a_4_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_4_1) : "f"(mul_f32_rn_noftz(_fma_97, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_99;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_99) : "f"(big_8a_4_1), "f"(8.388608e+06f));
            float whole_8a_4_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_4_1) : "f"(_add_99), "f"(8.388608e+06f));
            float above_8a_4_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_4_1) : "f"(-d4_8a_4_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_4_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_101;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_101) : "f"(big_8a_4_1), "f"(whole_8a_4_1));
            float _abs_100;
            asm("abs.f32 %0, %1;" : "=f"(_abs_100) : "f"(_sub_101));
            float _add_102;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_102) : "f"(big_8a_4_1), "f"(0.5f));
            float _sub_103;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_103) : "f"(big_8a_4_1), "f"(0.5f));
            float rounded_8a_4_1 = ((_abs_100 == 0.5f) ? ((above_8a_4_1 > 0.0f) ? _add_102 : _sub_103) : whole_8a_4_1);
            t1_3 = ((d1_10 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_4_1, 7.888609e-31f), 1.7763568e-15f) : t1_3);
        }
        float2 _f32x2_mul_a_104 = mul_f32x2_noftz(cvt_f32_11, make_float2(max_noftz(t0_3, 0.0f), max_noftz(t1_3, 0.0f)));
        float2 mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_104), "l"(*(const unsigned long long*)&cvt_f32_1_10));
        packed[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_10.x, mul_10.y)));
        float2 cvt_f32_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 cvt_f32_1_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float2 _f32x2_mul_b_105 = make_float2(-1.442695f, -1.442695f);
        float2 t_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_11) : "l"(*(const unsigned long long*)&cvt_f32_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_105));
        float e0_11 = approx_exp2(t_11.x);
        float e1_11 = approx_exp2(t_11.y);
        float2 nd_11 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_11, e1_11));
        float d0_11 = -nd_11.x;
        float d1_11 = -nd_11.y;
        float r0_11 = approx_rcp(d0_11);
        float r1_11 = approx_rcp(d1_11);
        float2 r_11 = make_float2(r0_11, r1_11);
        float2 s_11 = fma_f32x2_rn_noftz(r_11, fma_f32x2_rn_noftz(nd_11, r_11, make_float2(1.0f, 1.0f)), r_11);
        float t0_4 = s_11.x;
        float t1_4 = s_11.y;
        int _warp_vote_106 = __any_sync(0xFFFFFFFF, fminf(t0_4, t1_4) == 0.0f);
        if (_warp_vote_106 != 0) {
            float d4_8a_5;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_5) : "f"(d0_11), "f"(0.25f));
            float r4_8a_5 = approx_rcp(d4_8a_5);
            float _fma_108;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_108) : "f"(-d4_8a_5), "f"(r4_8a_5), "f"(1.0f));
            float _fma_107;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(r4_8a_5), "f"(_fma_108), "f"(r4_8a_5));
            float big_8a_5;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_5) : "f"(mul_f32_rn_noftz(_fma_107, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_109;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_109) : "f"(big_8a_5), "f"(8.388608e+06f));
            float whole_8a_5;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_5) : "f"(_add_109), "f"(8.388608e+06f));
            float above_8a_5;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_5) : "f"(-d4_8a_5), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_5, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_111;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_111) : "f"(big_8a_5), "f"(whole_8a_5));
            float _abs_110;
            asm("abs.f32 %0, %1;" : "=f"(_abs_110) : "f"(_sub_111));
            float _add_112;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_112) : "f"(big_8a_5), "f"(0.5f));
            float _sub_113;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_113) : "f"(big_8a_5), "f"(0.5f));
            float rounded_8a_5 = ((_abs_110 == 0.5f) ? ((above_8a_5 > 0.0f) ? _add_112 : _sub_113) : whole_8a_5);
            t0_4 = ((d0_11 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_5, 7.888609e-31f), 1.7763568e-15f) : t0_4);
            float d4_8a_5_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_5_1) : "f"(d1_11), "f"(0.25f));
            float r4_8a_5_1 = approx_rcp(d4_8a_5_1);
            float _fma_115;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_115) : "f"(-d4_8a_5_1), "f"(r4_8a_5_1), "f"(1.0f));
            float _fma_114;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_114) : "f"(r4_8a_5_1), "f"(_fma_115), "f"(r4_8a_5_1));
            float big_8a_5_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_5_1) : "f"(mul_f32_rn_noftz(_fma_114, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_116;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_116) : "f"(big_8a_5_1), "f"(8.388608e+06f));
            float whole_8a_5_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_5_1) : "f"(_add_116), "f"(8.388608e+06f));
            float above_8a_5_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_5_1) : "f"(-d4_8a_5_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_5_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_118;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_118) : "f"(big_8a_5_1), "f"(whole_8a_5_1));
            float _abs_117;
            asm("abs.f32 %0, %1;" : "=f"(_abs_117) : "f"(_sub_118));
            float _add_119;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_119) : "f"(big_8a_5_1), "f"(0.5f));
            float _sub_120;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_120) : "f"(big_8a_5_1), "f"(0.5f));
            float rounded_8a_5_1 = ((_abs_117 == 0.5f) ? ((above_8a_5_1 > 0.0f) ? _add_119 : _sub_120) : whole_8a_5_1);
            t1_4 = ((d1_11 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_5_1, 7.888609e-31f), 1.7763568e-15f) : t1_4);
        }
        float2 _f32x2_mul_a_121 = mul_f32x2_noftz(cvt_f32_12, make_float2(max_noftz(t0_4, 0.0f), max_noftz(t1_4, 0.0f)));
        float2 mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_121), "l"(*(const unsigned long long*)&cvt_f32_1_11));
        packed[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_11.x, mul_11.y)));
        float2 cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 cvt_f32_1_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float2 _f32x2_mul_b_122 = make_float2(-1.442695f, -1.442695f);
        float2 t_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_12) : "l"(*(const unsigned long long*)&cvt_f32_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_122));
        float e0_12 = approx_exp2(t_12.x);
        float e1_12 = approx_exp2(t_12.y);
        float2 nd_12 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_12, e1_12));
        float d0_12 = -nd_12.x;
        float d1_12 = -nd_12.y;
        float r0_12 = approx_rcp(d0_12);
        float r1_12 = approx_rcp(d1_12);
        float2 r_12 = make_float2(r0_12, r1_12);
        float2 s_12 = fma_f32x2_rn_noftz(r_12, fma_f32x2_rn_noftz(nd_12, r_12, make_float2(1.0f, 1.0f)), r_12);
        float t0_5 = s_12.x;
        float t1_5 = s_12.y;
        int _warp_vote_123 = __any_sync(0xFFFFFFFF, fminf(t0_5, t1_5) == 0.0f);
        if (_warp_vote_123 != 0) {
            float d4_8a_6;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_6) : "f"(d0_12), "f"(0.25f));
            float r4_8a_6 = approx_rcp(d4_8a_6);
            float _fma_125;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(-d4_8a_6), "f"(r4_8a_6), "f"(1.0f));
            float _fma_124;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_124) : "f"(r4_8a_6), "f"(_fma_125), "f"(r4_8a_6));
            float big_8a_6;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_6) : "f"(mul_f32_rn_noftz(_fma_124, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_126;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_126) : "f"(big_8a_6), "f"(8.388608e+06f));
            float whole_8a_6;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_6) : "f"(_add_126), "f"(8.388608e+06f));
            float above_8a_6;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_6) : "f"(-d4_8a_6), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_6, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_128;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_128) : "f"(big_8a_6), "f"(whole_8a_6));
            float _abs_127;
            asm("abs.f32 %0, %1;" : "=f"(_abs_127) : "f"(_sub_128));
            float _add_129;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_129) : "f"(big_8a_6), "f"(0.5f));
            float _sub_130;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_130) : "f"(big_8a_6), "f"(0.5f));
            float rounded_8a_6 = ((_abs_127 == 0.5f) ? ((above_8a_6 > 0.0f) ? _add_129 : _sub_130) : whole_8a_6);
            t0_5 = ((d0_12 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_6, 7.888609e-31f), 1.7763568e-15f) : t0_5);
            float d4_8a_6_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_6_1) : "f"(d1_12), "f"(0.25f));
            float r4_8a_6_1 = approx_rcp(d4_8a_6_1);
            float _fma_132;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_132) : "f"(-d4_8a_6_1), "f"(r4_8a_6_1), "f"(1.0f));
            float _fma_131;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_131) : "f"(r4_8a_6_1), "f"(_fma_132), "f"(r4_8a_6_1));
            float big_8a_6_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_6_1) : "f"(mul_f32_rn_noftz(_fma_131, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_133;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_133) : "f"(big_8a_6_1), "f"(8.388608e+06f));
            float whole_8a_6_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_6_1) : "f"(_add_133), "f"(8.388608e+06f));
            float above_8a_6_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_6_1) : "f"(-d4_8a_6_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_6_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_135;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_135) : "f"(big_8a_6_1), "f"(whole_8a_6_1));
            float _abs_134;
            asm("abs.f32 %0, %1;" : "=f"(_abs_134) : "f"(_sub_135));
            float _add_136;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_136) : "f"(big_8a_6_1), "f"(0.5f));
            float _sub_137;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_137) : "f"(big_8a_6_1), "f"(0.5f));
            float rounded_8a_6_1 = ((_abs_134 == 0.5f) ? ((above_8a_6_1 > 0.0f) ? _add_136 : _sub_137) : whole_8a_6_1);
            t1_5 = ((d1_12 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_6_1, 7.888609e-31f), 1.7763568e-15f) : t1_5);
        }
        float2 _f32x2_mul_a_138 = mul_f32x2_noftz(cvt_f32_13, make_float2(max_noftz(t0_5, 0.0f), max_noftz(t1_5, 0.0f)));
        float2 mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_138), "l"(*(const unsigned long long*)&cvt_f32_1_12));
        packed[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_12.x, mul_12.y)));
        float2 cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 cvt_f32_1_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float2 _f32x2_mul_b_139 = make_float2(-1.442695f, -1.442695f);
        float2 t_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_13) : "l"(*(const unsigned long long*)&cvt_f32_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_139));
        float e0_13 = approx_exp2(t_13.x);
        float e1_13 = approx_exp2(t_13.y);
        float2 nd_13 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_13, e1_13));
        float d0_13 = -nd_13.x;
        float d1_13 = -nd_13.y;
        float r0_13 = approx_rcp(d0_13);
        float r1_13 = approx_rcp(d1_13);
        float2 r_13 = make_float2(r0_13, r1_13);
        float2 s_13 = fma_f32x2_rn_noftz(r_13, fma_f32x2_rn_noftz(nd_13, r_13, make_float2(1.0f, 1.0f)), r_13);
        float t0_6 = s_13.x;
        float t1_6 = s_13.y;
        int _warp_vote_140 = __any_sync(0xFFFFFFFF, fminf(t0_6, t1_6) == 0.0f);
        if (_warp_vote_140 != 0) {
            float d4_8a_7;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_7) : "f"(d0_13), "f"(0.25f));
            float r4_8a_7 = approx_rcp(d4_8a_7);
            float _fma_142;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_142) : "f"(-d4_8a_7), "f"(r4_8a_7), "f"(1.0f));
            float _fma_141;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_141) : "f"(r4_8a_7), "f"(_fma_142), "f"(r4_8a_7));
            float big_8a_7;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_7) : "f"(mul_f32_rn_noftz(_fma_141, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_143;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_143) : "f"(big_8a_7), "f"(8.388608e+06f));
            float whole_8a_7;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_7) : "f"(_add_143), "f"(8.388608e+06f));
            float above_8a_7;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_7) : "f"(-d4_8a_7), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_7, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_145;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_145) : "f"(big_8a_7), "f"(whole_8a_7));
            float _abs_144;
            asm("abs.f32 %0, %1;" : "=f"(_abs_144) : "f"(_sub_145));
            float _add_146;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_146) : "f"(big_8a_7), "f"(0.5f));
            float _sub_147;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_147) : "f"(big_8a_7), "f"(0.5f));
            float rounded_8a_7 = ((_abs_144 == 0.5f) ? ((above_8a_7 > 0.0f) ? _add_146 : _sub_147) : whole_8a_7);
            t0_6 = ((d0_13 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_7, 7.888609e-31f), 1.7763568e-15f) : t0_6);
            float d4_8a_7_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_7_1) : "f"(d1_13), "f"(0.25f));
            float r4_8a_7_1 = approx_rcp(d4_8a_7_1);
            float _fma_149;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_149) : "f"(-d4_8a_7_1), "f"(r4_8a_7_1), "f"(1.0f));
            float _fma_148;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_148) : "f"(r4_8a_7_1), "f"(_fma_149), "f"(r4_8a_7_1));
            float big_8a_7_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_7_1) : "f"(mul_f32_rn_noftz(_fma_148, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_150;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_150) : "f"(big_8a_7_1), "f"(8.388608e+06f));
            float whole_8a_7_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_7_1) : "f"(_add_150), "f"(8.388608e+06f));
            float above_8a_7_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_7_1) : "f"(-d4_8a_7_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_7_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_152;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_152) : "f"(big_8a_7_1), "f"(whole_8a_7_1));
            float _abs_151;
            asm("abs.f32 %0, %1;" : "=f"(_abs_151) : "f"(_sub_152));
            float _add_153;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_153) : "f"(big_8a_7_1), "f"(0.5f));
            float _sub_154;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_154) : "f"(big_8a_7_1), "f"(0.5f));
            float rounded_8a_7_1 = ((_abs_151 == 0.5f) ? ((above_8a_7_1 > 0.0f) ? _add_153 : _sub_154) : whole_8a_7_1);
            t1_6 = ((d1_13 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_7_1, 7.888609e-31f), 1.7763568e-15f) : t1_6);
        }
        float2 _f32x2_mul_a_155 = mul_f32x2_noftz(cvt_f32_14, make_float2(max_noftz(t0_6, 0.0f), max_noftz(t1_6, 0.0f)));
        float2 mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_155), "l"(*(const unsigned long long*)&cvt_f32_1_13));
        packed[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_13.x, mul_13.y)));
    }
    __nv_bfloat162 amax = __as_bf16x2(0);
    __nv_bfloat162 _hxa_157 = amax;
    __nv_bfloat162 _hxb_157 = __as_bf16x2(packed[0]);
    unsigned int _hxr_157;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_157) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_157)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_157)));
    __nv_bfloat162 _max_156 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_157);
    amax = _max_156;
    __nv_bfloat162 _hxa_159 = amax;
    __nv_bfloat162 _hxb_159 = __as_bf16x2(packed[1]);
    unsigned int _hxr_159;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_159) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_159)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_159)));
    __nv_bfloat162 _max_158 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_159);
    amax = _max_158;
    __nv_bfloat162 _hxa_161 = amax;
    __nv_bfloat162 _hxb_161 = __as_bf16x2(packed[2]);
    unsigned int _hxr_161;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_161) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_161)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_161)));
    __nv_bfloat162 _max_160 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_161);
    amax = _max_160;
    __nv_bfloat162 _hxa_163 = amax;
    __nv_bfloat162 _hxb_163 = __as_bf16x2(packed[3]);
    unsigned int _hxr_163;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_163) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_163)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_163)));
    __nv_bfloat162 _max_162 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_163);
    amax = _max_162;
    __nv_bfloat162 _hxa_165 = amax;
    __nv_bfloat162 _hxb_165 = __as_bf16x2(packed[4]);
    unsigned int _hxr_165;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_165) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_165)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_165)));
    __nv_bfloat162 _max_164 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_165);
    amax = _max_164;
    __nv_bfloat162 _hxa_167 = amax;
    __nv_bfloat162 _hxb_167 = __as_bf16x2(packed[5]);
    unsigned int _hxr_167;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_167) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_167)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_167)));
    __nv_bfloat162 _max_166 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_167);
    amax = _max_166;
    __nv_bfloat162 _hxa_169 = amax;
    __nv_bfloat162 _hxb_169 = __as_bf16x2(packed[6]);
    unsigned int _hxr_169;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_169) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_169)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_169)));
    __nv_bfloat162 _max_168 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_169);
    amax = _max_168;
    __nv_bfloat162 _hxa_171 = amax;
    __nv_bfloat162 _hxb_171 = __as_bf16x2(packed[7]);
    unsigned int _hxr_171;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_171) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_171)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_171)));
    __nv_bfloat162 _max_170 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_171);
    amax = _max_170;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_173;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_173) : "r"(bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_173) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_172;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_172) : "h"(e4m3x2_bf16x2), "h"(_cvt_satfinite_e4m3x2_bf16x2_173));
    row_words[0] = _mov_b32_pack_u16x2_172;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_175;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_175) : "r"(bf16x2_mul_1_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_175) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_174;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_174) : "h"(e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_175));
    row_words[1] = _mov_b32_pack_u16x2_174;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_177;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_177) : "r"(bf16x2_mul_1_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_177) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_176;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_176) : "h"(e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_177));
    row_words[2] = _mov_b32_pack_u16x2_176;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_179;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_179) : "r"(bf16x2_mul_1_2));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_2);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_179) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_178;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_178) : "h"(e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_179));
    row_words[3] = _mov_b32_pack_u16x2_178;
    reinterpret_cast<int4*>(row_output + (grow * K + col))[0] = reinterpret_cast<int4*>(row_words + 0)[0];
}

} // extern "C"
