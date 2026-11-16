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

#define LAUNCH_MIN_BLOCKS 6

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

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_gated_act_mxfp8_40b0da5244b08b1db510(__nv_bfloat16* __restrict__ gated_input, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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
    unsigned int words[4];
    {
        uint32_t _uv4_0_0;
        uint32_t _uv4_0_1;
        uint32_t _uv4_0_2;
        uint32_t _uv4_0_3;
        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
            : "=r"(_uv4_0_0), "=r"(_uv4_0_1), "=r"(_uv4_0_2), "=r"(_uv4_0_3) : "l"((const void*)(gated_input + gate_index + (0))), "l"(0x12F0000000000000ULL) : "memory");
        words[0 + 0] = _uv4_0_0;
        words[0 + 1] = _uv4_0_1;
        words[0 + 2] = _uv4_0_2;
        words[0 + 3] = _uv4_0_3;
    }
    unsigned int words_0[4];
    {
        uint32_t _uv4_1_0;
        uint32_t _uv4_1_1;
        uint32_t _uv4_1_2;
        uint32_t _uv4_1_3;
        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
            : "=r"(_uv4_1_0), "=r"(_uv4_1_1), "=r"(_uv4_1_2), "=r"(_uv4_1_3) : "l"((const void*)(gated_input + (gate_index + K) + (0))), "l"(0x12F0000000000000ULL) : "memory");
        words_0[0 + 0] = _uv4_1_0;
        words_0[0 + 1] = _uv4_1_1;
        words_0[0 + 2] = _uv4_1_2;
        words_0[0 + 3] = _uv4_1_3;
    }
    unsigned int packed[4];
    __nv_bfloat162 lowest = __as_bf16x2(words[0]);
    lowest = __hmin2_nan(lowest, __as_bf16x2(words[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(words[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(words[3]));
    float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(lowest);
    float _min_3;
    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_3) : "f"(_cvt_f32_0.x), "f"(_cvt_f32_0.y));
    int _warp_vote_2 = __all_sync(0xFFFFFFFF, _min_3 > -87.0f);
    if (_warp_vote_2 != 0) {
        float2 cvt_f32 = cake_bf16x2_to_f32x2(__as_bf16x2(words[0]));
        float2 cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[0]));
        float e0 = approx_exp2(-cvt_f32.x * 1.442695f);
        float e1 = approx_exp2(-cvt_f32.y * 1.442695f);
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
        float2 cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(words[1]));
        float2 cvt_f32_1_0 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[1]));
        float e0_0 = approx_exp2(-cvt_f32_0.x * 1.442695f);
        float e1_0 = approx_exp2(-cvt_f32_0.y * 1.442695f);
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
        float2 cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(words[2]));
        float2 cvt_f32_1_1 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[2]));
        float e0_1 = approx_exp2(-cvt_f32_2.x * 1.442695f);
        float e1_1 = approx_exp2(-cvt_f32_2.y * 1.442695f);
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
        float2 cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(words[3]));
        float2 cvt_f32_1_2 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[3]));
        float e0_2 = approx_exp2(-cvt_f32_3.x * 1.442695f);
        float e1_2 = approx_exp2(-cvt_f32_3.y * 1.442695f);
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
    } else {
        int band = 0;
        float2 cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(words[0]));
        float2 cvt_f32_1_3 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[0]));
        float e0_3 = approx_exp2(-cvt_f32_4.x * 1.442695f);
        float e1_3 = approx_exp2(-cvt_f32_4.y * 1.442695f);
        float d0_3 = 1.0f + e0_3;
        float d1_3 = 1.0f + e1_3;
        float dn;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn) : "f"(e0_3), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r = approx_rcp(dn);
        float _fma_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(-dn), "f"(r), "f"(1.0f));
        float q;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q) : "f"(r), "f"(_fma_24), "f"(r));
        float s0;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0) : "f"(q), "f"(1.1754944e-38f));
        float err;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err) : "f"(s0), "f"(-8.507059e+37f), "f"(q));
        float y2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2) : "f"(-dn), "f"(q), "f"(1.0f));
        float nudge = __uint_as_float(__as_u32(y2) & 2147483648u | 729808896);
        float _add_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(err), "f"(nudge));
        float u0;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0) : "f"(_add_25), "f"(1.1754944e-38f), "f"(s0));
        float dn_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_0) : "f"(e1_3), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_0 = approx_rcp(dn_0);
        float _fma_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(-dn_0), "f"(r_0), "f"(1.0f));
        float q_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_0) : "f"(r_0), "f"(_fma_26), "f"(r_0));
        float s0_0;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_0) : "f"(q_0), "f"(1.1754944e-38f));
        float err_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_0) : "f"(s0_0), "f"(-8.507059e+37f), "f"(q_0));
        float y2_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_0) : "f"(-dn_0), "f"(q_0), "f"(1.0f));
        float nudge_0 = __uint_as_float(__as_u32(y2_0) & 2147483648u | 729808896);
        float _add_27;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(err_0), "f"(nudge_0));
        float u1;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1) : "f"(_add_27), "f"(1.1754944e-38f), "f"(s0_0));
        float2 _f32x2_mul_a_28 = mul_f32x2_noftz(cvt_f32_4, make_float2(u0, u1));
        float2 mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&cvt_f32_1_3));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_3.x, mul_3.y)));
        float2 cvt_f32_4_1 = cake_bf16x2_to_f32x2(__as_bf16x2(words[1]));
        float2 cvt_f32_1_3_1 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[1]));
        float e0_3_1 = approx_exp2(-cvt_f32_4_1.x * 1.442695f);
        float e1_3_1 = approx_exp2(-cvt_f32_4_1.y * 1.442695f);
        float d0_3_1 = 1.0f + e0_3_1;
        float d1_3_1 = 1.0f + e1_3_1;
        float dn_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_1) : "f"(e0_3_1), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_1 = approx_rcp(dn_1);
        float _fma_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(-dn_1), "f"(r_1), "f"(1.0f));
        float q_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_1) : "f"(r_1), "f"(_fma_29), "f"(r_1));
        float s0_1;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_1) : "f"(q_1), "f"(1.1754944e-38f));
        float err_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_1) : "f"(s0_1), "f"(-8.507059e+37f), "f"(q_1));
        float y2_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_1) : "f"(-dn_1), "f"(q_1), "f"(1.0f));
        float nudge_1 = __uint_as_float(__as_u32(y2_1) & 2147483648u | 729808896);
        float _add_30;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(err_1), "f"(nudge_1));
        float u0_0;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0_0) : "f"(_add_30), "f"(1.1754944e-38f), "f"(s0_1));
        float dn_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_2) : "f"(e1_3_1), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_2 = approx_rcp(dn_2);
        float _fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(-dn_2), "f"(r_2), "f"(1.0f));
        float q_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_2) : "f"(r_2), "f"(_fma_31), "f"(r_2));
        float s0_2;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_2) : "f"(q_2), "f"(1.1754944e-38f));
        float err_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_2) : "f"(s0_2), "f"(-8.507059e+37f), "f"(q_2));
        float y2_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_2) : "f"(-dn_2), "f"(q_2), "f"(1.0f));
        float nudge_2 = __uint_as_float(__as_u32(y2_2) & 2147483648u | 729808896);
        float _add_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(err_2), "f"(nudge_2));
        float u1_0;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1_0) : "f"(_add_32), "f"(1.1754944e-38f), "f"(s0_2));
        float2 _f32x2_mul_a_33 = mul_f32x2_noftz(cvt_f32_4_1, make_float2(u0_0, u1_0));
        float2 mul_3_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&cvt_f32_1_3_1));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_3_1.x, mul_3_1.y)));
        float2 cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(words[2]));
        float2 cvt_f32_1_4 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[2]));
        float e0_4 = approx_exp2(-cvt_f32_5.x * 1.442695f);
        float e1_4 = approx_exp2(-cvt_f32_5.y * 1.442695f);
        float d0_4 = 1.0f + e0_4;
        float d1_4 = 1.0f + e1_4;
        float dn_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_3) : "f"(e0_4), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_3 = approx_rcp(dn_3);
        float _fma_34;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(-dn_3), "f"(r_3), "f"(1.0f));
        float q_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_3) : "f"(r_3), "f"(_fma_34), "f"(r_3));
        float s0_3;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_3) : "f"(q_3), "f"(1.1754944e-38f));
        float err_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_3) : "f"(s0_3), "f"(-8.507059e+37f), "f"(q_3));
        float y2_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_3) : "f"(-dn_3), "f"(q_3), "f"(1.0f));
        float nudge_3 = __uint_as_float(__as_u32(y2_3) & 2147483648u | 729808896);
        float _add_35;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35) : "f"(err_3), "f"(nudge_3));
        float u0_1;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0_1) : "f"(_add_35), "f"(1.1754944e-38f), "f"(s0_3));
        float dn_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_4) : "f"(e1_4), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_4 = approx_rcp(dn_4);
        float _fma_36;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(-dn_4), "f"(r_4), "f"(1.0f));
        float q_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_4) : "f"(r_4), "f"(_fma_36), "f"(r_4));
        float s0_4;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_4) : "f"(q_4), "f"(1.1754944e-38f));
        float err_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_4) : "f"(s0_4), "f"(-8.507059e+37f), "f"(q_4));
        float y2_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_4) : "f"(-dn_4), "f"(q_4), "f"(1.0f));
        float nudge_4 = __uint_as_float(__as_u32(y2_4) & 2147483648u | 729808896);
        float _add_37;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(err_4), "f"(nudge_4));
        float u1_1;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1_1) : "f"(_add_37), "f"(1.1754944e-38f), "f"(s0_4));
        float2 _f32x2_mul_a_38 = mul_f32x2_noftz(cvt_f32_5, make_float2(u0_1, u1_1));
        float2 mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_38), "l"(*(const unsigned long long*)&cvt_f32_1_4));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_4.x, mul_4.y)));
        float2 cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(words[3]));
        float2 cvt_f32_1_5 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[3]));
        float e0_5 = approx_exp2(-cvt_f32_6.x * 1.442695f);
        float e1_5 = approx_exp2(-cvt_f32_6.y * 1.442695f);
        float d0_5 = 1.0f + e0_5;
        float d1_5 = 1.0f + e1_5;
        float dn_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_5) : "f"(e0_5), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_5 = approx_rcp(dn_5);
        float _fma_39;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(-dn_5), "f"(r_5), "f"(1.0f));
        float q_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_5) : "f"(r_5), "f"(_fma_39), "f"(r_5));
        float s0_5;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_5) : "f"(q_5), "f"(1.1754944e-38f));
        float err_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_5) : "f"(s0_5), "f"(-8.507059e+37f), "f"(q_5));
        float y2_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_5) : "f"(-dn_5), "f"(q_5), "f"(1.0f));
        float nudge_5 = __uint_as_float(__as_u32(y2_5) & 2147483648u | 729808896);
        float _add_40;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(err_5), "f"(nudge_5));
        float u0_2;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0_2) : "f"(_add_40), "f"(1.1754944e-38f), "f"(s0_5));
        float dn_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_6) : "f"(e1_5), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_6 = approx_rcp(dn_6);
        float _fma_41;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(-dn_6), "f"(r_6), "f"(1.0f));
        float q_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_6) : "f"(r_6), "f"(_fma_41), "f"(r_6));
        float s0_6;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_6) : "f"(q_6), "f"(1.1754944e-38f));
        float err_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_6) : "f"(s0_6), "f"(-8.507059e+37f), "f"(q_6));
        float y2_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_6) : "f"(-dn_6), "f"(q_6), "f"(1.0f));
        float nudge_6 = __uint_as_float(__as_u32(y2_6) & 2147483648u | 729808896);
        float _add_42;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(err_6), "f"(nudge_6));
        float u1_2;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1_2) : "f"(_add_42), "f"(1.1754944e-38f), "f"(s0_6));
        float2 _f32x2_mul_a_43 = mul_f32x2_noftz(cvt_f32_6, make_float2(u0_2, u1_2));
        float2 mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_43), "l"(*(const unsigned long long*)&cvt_f32_1_5));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_5.x, mul_5.y)));
    }
    __nv_bfloat162 amax = __as_bf16x2(0);
    __nv_bfloat162 _hxa_45 = amax;
    __nv_bfloat162 _hxb_45 = __as_bf16x2(packed[0]);
    unsigned int _hxr_45;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_45) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_45)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_45)));
    __nv_bfloat162 _max_44 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_45);
    amax = _max_44;
    __nv_bfloat162 _hxa_47 = amax;
    __nv_bfloat162 _hxb_47 = __as_bf16x2(packed[1]);
    unsigned int _hxr_47;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_47) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_47)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_47)));
    __nv_bfloat162 _max_46 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_47);
    amax = _max_46;
    __nv_bfloat162 _hxa_49 = amax;
    __nv_bfloat162 _hxb_49 = __as_bf16x2(packed[2]);
    unsigned int _hxr_49;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_49) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_49)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_49)));
    __nv_bfloat162 _max_48 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_49);
    amax = _max_48;
    __nv_bfloat162 _hxa_51 = amax;
    __nv_bfloat162 _hxb_51 = __as_bf16x2(packed[3]);
    unsigned int _hxr_51;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_51) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_51)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_51)));
    __nv_bfloat162 _max_50 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_51);
    amax = _max_50;
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
    int row_scale_index = ((grow >> 7) * (K >> 7) + (bx >> 2)) * 512 + (grow & 31) * 16 + (grow >> 5 & 3) * 4 + (bx & 3);
    if (quarter == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + row_scale_index) + (0)) = (unsigned char)(scale);
    }
    unsigned int row_words[2];
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_53;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_53) : "r"(bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_53) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_52;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_52) : "h"(e4m3x2_bf16x2), "h"(_cvt_satfinite_e4m3x2_bf16x2_53));
    row_words[0] = _mov_b32_pack_u16x2_52;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_55;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_55) : "r"(bf16x2_mul_1_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_55) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_54;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_54) : "h"(e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_55));
    row_words[1] = _mov_b32_pack_u16x2_54;
    reinterpret_cast<int2*>(row_output + (grow * K + col))[0] = reinterpret_cast<int2*>(row_words + 0)[0];
}

} // extern "C"
