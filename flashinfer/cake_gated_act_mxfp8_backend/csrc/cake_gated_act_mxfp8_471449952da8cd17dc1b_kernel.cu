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
kernel_cake_gated_act_mxfp8_471449952da8cd17dc1b(__nv_bfloat16* __restrict__ gated_input, __nv_bfloat16* __restrict__ grad, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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
    unsigned int _vec_load_2[8];
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
        _vec_load_2[0 + 0] = (uint32_t)_iv_2_0;
        _vec_load_2[0 + 1] = (uint32_t)_iv_2_1;
        _vec_load_2[0 + 2] = (uint32_t)_iv_2_2;
        _vec_load_2[0 + 3] = (uint32_t)_iv_2_3;
        _vec_load_2[0 + 4] = (uint32_t)_iv_2_4;
        _vec_load_2[0 + 5] = (uint32_t)_iv_2_5;
        _vec_load_2[0 + 6] = (uint32_t)_iv_2_6;
        _vec_load_2[0 + 7] = (uint32_t)_iv_2_7;
    }
    int grow_hi = grow + 32;
    int gate_index_hi = grow_hi * (2 * K) + col;
    unsigned int _vec_load_3[8];
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
            : "=r"(_iv_3_0), "=r"(_iv_3_1), "=r"(_iv_3_2), "=r"(_iv_3_3), "=r"(_iv_3_4), "=r"(_iv_3_5), "=r"(_iv_3_6), "=r"(_iv_3_7) : "l"((const void*)(gated_input + gate_index_hi + (0))) : "memory");
        _vec_load_3[0 + 0] = (uint32_t)_iv_3_0;
        _vec_load_3[0 + 1] = (uint32_t)_iv_3_1;
        _vec_load_3[0 + 2] = (uint32_t)_iv_3_2;
        _vec_load_3[0 + 3] = (uint32_t)_iv_3_3;
        _vec_load_3[0 + 4] = (uint32_t)_iv_3_4;
        _vec_load_3[0 + 5] = (uint32_t)_iv_3_5;
        _vec_load_3[0 + 6] = (uint32_t)_iv_3_6;
        _vec_load_3[0 + 7] = (uint32_t)_iv_3_7;
    }
    unsigned int _vec_load_4[8];
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
            : "=r"(_iv_4_0), "=r"(_iv_4_1), "=r"(_iv_4_2), "=r"(_iv_4_3), "=r"(_iv_4_4), "=r"(_iv_4_5), "=r"(_iv_4_6), "=r"(_iv_4_7) : "l"((const void*)(gated_input + (gate_index_hi + K) + (0))) : "memory");
        _vec_load_4[0 + 0] = (uint32_t)_iv_4_0;
        _vec_load_4[0 + 1] = (uint32_t)_iv_4_1;
        _vec_load_4[0 + 2] = (uint32_t)_iv_4_2;
        _vec_load_4[0 + 3] = (uint32_t)_iv_4_3;
        _vec_load_4[0 + 4] = (uint32_t)_iv_4_4;
        _vec_load_4[0 + 5] = (uint32_t)_iv_4_5;
        _vec_load_4[0 + 6] = (uint32_t)_iv_4_6;
        _vec_load_4[0 + 7] = (uint32_t)_iv_4_7;
    }
    unsigned int _vec_load_5[8];
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
            : "=r"(_iv_5_0), "=r"(_iv_5_1), "=r"(_iv_5_2), "=r"(_iv_5_3), "=r"(_iv_5_4), "=r"(_iv_5_5), "=r"(_iv_5_6), "=r"(_iv_5_7) : "l"((const void*)(grad + (grow_hi * K + col) + (0))) : "memory");
        _vec_load_5[0 + 0] = (uint32_t)_iv_5_0;
        _vec_load_5[0 + 1] = (uint32_t)_iv_5_1;
        _vec_load_5[0 + 2] = (uint32_t)_iv_5_2;
        _vec_load_5[0 + 3] = (uint32_t)_iv_5_3;
        _vec_load_5[0 + 4] = (uint32_t)_iv_5_4;
        _vec_load_5[0 + 5] = (uint32_t)_iv_5_5;
        _vec_load_5[0 + 6] = (uint32_t)_iv_5_6;
        _vec_load_5[0 + 7] = (uint32_t)_iv_5_7;
    }
    unsigned int dgate[8];
    unsigned int dup[8];
    unsigned int dgate_hi[8];
    unsigned int dup_hi[8];
    __nv_bfloat162 lowest = __as_bf16x2(_vec_load_0[0]);
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[7]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_3[0]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_3[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_3[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_3[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_3[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_3[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_3[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_3[7]));
    float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(lowest);
    float _min_7;
    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_7) : "f"(_cvt_f32_0.x), "f"(_cvt_f32_0.y));
    int _warp_vote_6 = __all_sync(0xFFFFFFFF, _min_7 > -87.0f);
    if (_warp_vote_6 != 0) {
        float2 cvt_f32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float e0 = approx_exp2((-cvt_f32.x) * 1.442695f);
        float e1 = approx_exp2((-cvt_f32.y) * 1.442695f);
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
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act) : "l"(*(const unsigned long long*)&cvt_f32), "l"(*(const unsigned long long*)&s));
        float2 _f32x2_mul_a_12 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s), s), cvt_f32_2);
        float2 mul;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul) : "l"(*(const unsigned long long*)&_f32x2_mul_a_12), "l"(*(const unsigned long long*)&cvt_f32_1));
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
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_0) : "l"(*(const unsigned long long*)&cvt_f32_0), "l"(*(const unsigned long long*)&s_0));
        float2 _f32x2_mul_a_17 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_0), s_0), cvt_f32_2_0);
        float2 mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&cvt_f32_1_0));
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
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_1) : "l"(*(const unsigned long long*)&cvt_f32_3), "l"(*(const unsigned long long*)&s_1));
        float2 _f32x2_mul_a_22 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1), s_1), cvt_f32_2_1);
        float2 mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_22), "l"(*(const unsigned long long*)&cvt_f32_1_1));
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
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_2) : "l"(*(const unsigned long long*)&cvt_f32_4), "l"(*(const unsigned long long*)&s_2));
        float2 _f32x2_mul_a_27 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_2), s_2), cvt_f32_2_2);
        float2 mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_27), "l"(*(const unsigned long long*)&cvt_f32_1_2));
        float2 mul_1_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_2) : "l"(*(const unsigned long long*)&act_2), "l"(*(const unsigned long long*)&cvt_f32_2_2));
        __nv_bfloat162 bf16x2_3 = __float22bfloat162_rn(make_float2(mul_3.x, mul_3.y));
        __nv_bfloat162 bf16x2_1_2 = __float22bfloat162_rn(make_float2(mul_1_2.x, mul_1_2.y));
        dgate[3] = __as_u32(bf16x2_3);
        dup[3] = __as_u32(bf16x2_1_2);
        float2 cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 cvt_f32_1_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float2 cvt_f32_2_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[4]));
        float e0_3 = approx_exp2((-cvt_f32_5.x) * 1.442695f);
        float e1_3 = approx_exp2((-cvt_f32_5.y) * 1.442695f);
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
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_3) : "l"(*(const unsigned long long*)&cvt_f32_5), "l"(*(const unsigned long long*)&s_3));
        float2 _f32x2_mul_a_32 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_3), s_3), cvt_f32_2_3);
        float2 mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_32), "l"(*(const unsigned long long*)&cvt_f32_1_3));
        float2 mul_1_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_3) : "l"(*(const unsigned long long*)&act_3), "l"(*(const unsigned long long*)&cvt_f32_2_3));
        __nv_bfloat162 bf16x2_4 = __float22bfloat162_rn(make_float2(mul_4.x, mul_4.y));
        __nv_bfloat162 bf16x2_1_3 = __float22bfloat162_rn(make_float2(mul_1_3.x, mul_1_3.y));
        dgate[4] = __as_u32(bf16x2_4);
        dup[4] = __as_u32(bf16x2_1_3);
        float2 cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 cvt_f32_1_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float2 cvt_f32_2_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[5]));
        float e0_4 = approx_exp2((-cvt_f32_6.x) * 1.442695f);
        float e1_4 = approx_exp2((-cvt_f32_6.y) * 1.442695f);
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
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_4) : "l"(*(const unsigned long long*)&cvt_f32_6), "l"(*(const unsigned long long*)&s_4));
        float2 _f32x2_mul_a_37 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_4), s_4), cvt_f32_2_4);
        float2 mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_37), "l"(*(const unsigned long long*)&cvt_f32_1_4));
        float2 mul_1_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_4) : "l"(*(const unsigned long long*)&act_4), "l"(*(const unsigned long long*)&cvt_f32_2_4));
        __nv_bfloat162 bf16x2_5 = __float22bfloat162_rn(make_float2(mul_5.x, mul_5.y));
        __nv_bfloat162 bf16x2_1_4 = __float22bfloat162_rn(make_float2(mul_1_4.x, mul_1_4.y));
        dgate[5] = __as_u32(bf16x2_5);
        dup[5] = __as_u32(bf16x2_1_4);
        float2 cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 cvt_f32_1_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float2 cvt_f32_2_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[6]));
        float e0_5 = approx_exp2((-cvt_f32_7.x) * 1.442695f);
        float e1_5 = approx_exp2((-cvt_f32_7.y) * 1.442695f);
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
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_5) : "l"(*(const unsigned long long*)&cvt_f32_7), "l"(*(const unsigned long long*)&s_5));
        float2 _f32x2_mul_a_42 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_5), s_5), cvt_f32_2_5);
        float2 mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&cvt_f32_1_5));
        float2 mul_1_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_5) : "l"(*(const unsigned long long*)&act_5), "l"(*(const unsigned long long*)&cvt_f32_2_5));
        __nv_bfloat162 bf16x2_6 = __float22bfloat162_rn(make_float2(mul_6.x, mul_6.y));
        __nv_bfloat162 bf16x2_1_5 = __float22bfloat162_rn(make_float2(mul_1_5.x, mul_1_5.y));
        dgate[6] = __as_u32(bf16x2_6);
        dup[6] = __as_u32(bf16x2_1_5);
        float2 cvt_f32_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 cvt_f32_1_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float2 cvt_f32_2_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[7]));
        float e0_6 = approx_exp2((-cvt_f32_8.x) * 1.442695f);
        float e1_6 = approx_exp2((-cvt_f32_8.y) * 1.442695f);
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
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_6) : "l"(*(const unsigned long long*)&cvt_f32_8), "l"(*(const unsigned long long*)&s_6));
        float2 _f32x2_mul_a_47 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_6), s_6), cvt_f32_2_6);
        float2 mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_47), "l"(*(const unsigned long long*)&cvt_f32_1_6));
        float2 mul_1_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_6) : "l"(*(const unsigned long long*)&act_6), "l"(*(const unsigned long long*)&cvt_f32_2_6));
        __nv_bfloat162 bf16x2_7 = __float22bfloat162_rn(make_float2(mul_7.x, mul_7.y));
        __nv_bfloat162 bf16x2_1_6 = __float22bfloat162_rn(make_float2(mul_1_6.x, mul_1_6.y));
        dgate[7] = __as_u32(bf16x2_7);
        dup[7] = __as_u32(bf16x2_1_6);
        float2 cvt_f32_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
        float2 cvt_f32_1_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[0]));
        float2 cvt_f32_2_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[0]));
        float e0_7 = approx_exp2((-cvt_f32_9.x) * 1.442695f);
        float e1_7 = approx_exp2((-cvt_f32_9.y) * 1.442695f);
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
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_7) : "l"(*(const unsigned long long*)&cvt_f32_9), "l"(*(const unsigned long long*)&s_7));
        float2 _f32x2_mul_a_52 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_7, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_7), s_7), cvt_f32_2_7);
        float2 mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_52), "l"(*(const unsigned long long*)&cvt_f32_1_7));
        float2 mul_1_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_7) : "l"(*(const unsigned long long*)&act_7), "l"(*(const unsigned long long*)&cvt_f32_2_7));
        __nv_bfloat162 bf16x2_8 = __float22bfloat162_rn(make_float2(mul_8.x, mul_8.y));
        __nv_bfloat162 bf16x2_1_7 = __float22bfloat162_rn(make_float2(mul_1_7.x, mul_1_7.y));
        dgate_hi[0] = __as_u32(bf16x2_8);
        dup_hi[0] = __as_u32(bf16x2_1_7);
        float2 cvt_f32_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[1]));
        float2 cvt_f32_1_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[1]));
        float2 cvt_f32_2_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[1]));
        float e0_8 = approx_exp2((-cvt_f32_10.x) * 1.442695f);
        float e1_8 = approx_exp2((-cvt_f32_10.y) * 1.442695f);
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
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_8) : "l"(*(const unsigned long long*)&cvt_f32_10), "l"(*(const unsigned long long*)&s_8));
        float2 _f32x2_mul_a_57 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_8, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_8), s_8), cvt_f32_2_8);
        float2 mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_57), "l"(*(const unsigned long long*)&cvt_f32_1_8));
        float2 mul_1_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_8) : "l"(*(const unsigned long long*)&act_8), "l"(*(const unsigned long long*)&cvt_f32_2_8));
        __nv_bfloat162 bf16x2_9 = __float22bfloat162_rn(make_float2(mul_9.x, mul_9.y));
        __nv_bfloat162 bf16x2_1_8 = __float22bfloat162_rn(make_float2(mul_1_8.x, mul_1_8.y));
        dgate_hi[1] = __as_u32(bf16x2_9);
        dup_hi[1] = __as_u32(bf16x2_1_8);
        float2 cvt_f32_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[2]));
        float2 cvt_f32_1_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[2]));
        float2 cvt_f32_2_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[2]));
        float e0_9 = approx_exp2((-cvt_f32_11.x) * 1.442695f);
        float e1_9 = approx_exp2((-cvt_f32_11.y) * 1.442695f);
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
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_9) : "l"(*(const unsigned long long*)&cvt_f32_11), "l"(*(const unsigned long long*)&s_9));
        float2 _f32x2_mul_a_62 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_9, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_9), s_9), cvt_f32_2_9);
        float2 mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&cvt_f32_1_9));
        float2 mul_1_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_9) : "l"(*(const unsigned long long*)&act_9), "l"(*(const unsigned long long*)&cvt_f32_2_9));
        __nv_bfloat162 bf16x2_10 = __float22bfloat162_rn(make_float2(mul_10.x, mul_10.y));
        __nv_bfloat162 bf16x2_1_9 = __float22bfloat162_rn(make_float2(mul_1_9.x, mul_1_9.y));
        dgate_hi[2] = __as_u32(bf16x2_10);
        dup_hi[2] = __as_u32(bf16x2_1_9);
        float2 cvt_f32_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[3]));
        float2 cvt_f32_1_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[3]));
        float2 cvt_f32_2_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[3]));
        float e0_10 = approx_exp2((-cvt_f32_12.x) * 1.442695f);
        float e1_10 = approx_exp2((-cvt_f32_12.y) * 1.442695f);
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
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_10) : "l"(*(const unsigned long long*)&cvt_f32_12), "l"(*(const unsigned long long*)&s_10));
        float2 _f32x2_mul_a_67 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_10, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_10), s_10), cvt_f32_2_10);
        float2 mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_67), "l"(*(const unsigned long long*)&cvt_f32_1_10));
        float2 mul_1_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_10) : "l"(*(const unsigned long long*)&act_10), "l"(*(const unsigned long long*)&cvt_f32_2_10));
        __nv_bfloat162 bf16x2_11 = __float22bfloat162_rn(make_float2(mul_11.x, mul_11.y));
        __nv_bfloat162 bf16x2_1_10 = __float22bfloat162_rn(make_float2(mul_1_10.x, mul_1_10.y));
        dgate_hi[3] = __as_u32(bf16x2_11);
        dup_hi[3] = __as_u32(bf16x2_1_10);
        float2 cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[4]));
        float2 cvt_f32_1_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[4]));
        float2 cvt_f32_2_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[4]));
        float e0_11 = approx_exp2((-cvt_f32_13.x) * 1.442695f);
        float e1_11 = approx_exp2((-cvt_f32_13.y) * 1.442695f);
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
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_11) : "l"(*(const unsigned long long*)&cvt_f32_13), "l"(*(const unsigned long long*)&s_11));
        float2 _f32x2_mul_a_72 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_11, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_11), s_11), cvt_f32_2_11);
        float2 mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_72), "l"(*(const unsigned long long*)&cvt_f32_1_11));
        float2 mul_1_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_11) : "l"(*(const unsigned long long*)&act_11), "l"(*(const unsigned long long*)&cvt_f32_2_11));
        __nv_bfloat162 bf16x2_12 = __float22bfloat162_rn(make_float2(mul_12.x, mul_12.y));
        __nv_bfloat162 bf16x2_1_11 = __float22bfloat162_rn(make_float2(mul_1_11.x, mul_1_11.y));
        dgate_hi[4] = __as_u32(bf16x2_12);
        dup_hi[4] = __as_u32(bf16x2_1_11);
        float2 cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[5]));
        float2 cvt_f32_1_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[5]));
        float2 cvt_f32_2_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[5]));
        float e0_12 = approx_exp2((-cvt_f32_14.x) * 1.442695f);
        float e1_12 = approx_exp2((-cvt_f32_14.y) * 1.442695f);
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
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_12) : "l"(*(const unsigned long long*)&cvt_f32_14), "l"(*(const unsigned long long*)&s_12));
        float2 _f32x2_mul_a_77 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_12, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_12), s_12), cvt_f32_2_12);
        float2 mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_77), "l"(*(const unsigned long long*)&cvt_f32_1_12));
        float2 mul_1_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_12) : "l"(*(const unsigned long long*)&act_12), "l"(*(const unsigned long long*)&cvt_f32_2_12));
        __nv_bfloat162 bf16x2_13 = __float22bfloat162_rn(make_float2(mul_13.x, mul_13.y));
        __nv_bfloat162 bf16x2_1_12 = __float22bfloat162_rn(make_float2(mul_1_12.x, mul_1_12.y));
        dgate_hi[5] = __as_u32(bf16x2_13);
        dup_hi[5] = __as_u32(bf16x2_1_12);
        float2 cvt_f32_15 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[6]));
        float2 cvt_f32_1_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[6]));
        float2 cvt_f32_2_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[6]));
        float e0_13 = approx_exp2((-cvt_f32_15.x) * 1.442695f);
        float e1_13 = approx_exp2((-cvt_f32_15.y) * 1.442695f);
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
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_13) : "l"(*(const unsigned long long*)&cvt_f32_15), "l"(*(const unsigned long long*)&s_13));
        float2 _f32x2_mul_a_82 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_13, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_13), s_13), cvt_f32_2_13);
        float2 mul_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_82), "l"(*(const unsigned long long*)&cvt_f32_1_13));
        float2 mul_1_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_13) : "l"(*(const unsigned long long*)&act_13), "l"(*(const unsigned long long*)&cvt_f32_2_13));
        __nv_bfloat162 bf16x2_14 = __float22bfloat162_rn(make_float2(mul_14.x, mul_14.y));
        __nv_bfloat162 bf16x2_1_13 = __float22bfloat162_rn(make_float2(mul_1_13.x, mul_1_13.y));
        dgate_hi[6] = __as_u32(bf16x2_14);
        dup_hi[6] = __as_u32(bf16x2_1_13);
        float2 cvt_f32_16 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[7]));
        float2 cvt_f32_1_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[7]));
        float2 cvt_f32_2_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[7]));
        float e0_14 = approx_exp2((-cvt_f32_16.x) * 1.442695f);
        float e1_14 = approx_exp2((-cvt_f32_16.y) * 1.442695f);
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
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_14) : "l"(*(const unsigned long long*)&cvt_f32_16), "l"(*(const unsigned long long*)&s_14));
        float2 _f32x2_mul_a_87 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_14, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_14), s_14), cvt_f32_2_14);
        float2 mul_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_87), "l"(*(const unsigned long long*)&cvt_f32_1_14));
        float2 mul_1_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_14) : "l"(*(const unsigned long long*)&act_14), "l"(*(const unsigned long long*)&cvt_f32_2_14));
        __nv_bfloat162 bf16x2_15 = __float22bfloat162_rn(make_float2(mul_15.x, mul_15.y));
        __nv_bfloat162 bf16x2_1_14 = __float22bfloat162_rn(make_float2(mul_1_14.x, mul_1_14.y));
        dgate_hi[7] = __as_u32(bf16x2_15);
        dup_hi[7] = __as_u32(bf16x2_1_14);
    } else {
        int band = 0;
        float2 cvt_f32_17 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 cvt_f32_1_15 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 cvt_f32_2_15 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float e0_15 = approx_exp2((-cvt_f32_17.x) * 1.442695f);
        float e1_15 = approx_exp2((-cvt_f32_17.y) * 1.442695f);
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
        band = band | ((fminf(t0, t1) == 0.0f) ? 1 : 0);
        float2 s_15 = make_float2(max_noftz(t0, 0.0f), max_noftz(t1, 0.0f));
        float2 act_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_15) : "l"(*(const unsigned long long*)&cvt_f32_17), "l"(*(const unsigned long long*)&s_15));
        float2 _f32x2_mul_a_90 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_15, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_15), s_15), cvt_f32_2_15);
        float2 mul_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_90), "l"(*(const unsigned long long*)&cvt_f32_1_15));
        float2 mul_1_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_15) : "l"(*(const unsigned long long*)&act_15), "l"(*(const unsigned long long*)&cvt_f32_2_15));
        __nv_bfloat162 bf16x2_16 = __float22bfloat162_rn(make_float2(mul_16.x, mul_16.y));
        __nv_bfloat162 bf16x2_1_15 = __float22bfloat162_rn(make_float2(mul_1_15.x, mul_1_15.y));
        dgate[0] = __as_u32(bf16x2_16);
        dup[0] = __as_u32(bf16x2_1_15);
        float2 cvt_f32_17_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 cvt_f32_1_15_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 cvt_f32_2_15_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float e0_15_1 = approx_exp2((-cvt_f32_17_1.x) * 1.442695f);
        float e1_15_1 = approx_exp2((-cvt_f32_17_1.y) * 1.442695f);
        float d0_15_1 = 1.0f + e0_15_1;
        float d1_15_1 = 1.0f + e1_15_1;
        float r0_15_1 = approx_rcp(d0_15_1);
        float r1_15_1 = approx_rcp(d1_15_1);
        float _fma_91;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(-d0_15_1), "f"(r0_15_1), "f"(1.0f));
        float fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_0) : "f"(r0_15_1), "f"(_fma_91), "f"(r0_15_1));
        float _fma_92;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(-d1_15_1), "f"(r1_15_1), "f"(1.0f));
        float fma_1_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_0) : "f"(r1_15_1), "f"(_fma_92), "f"(r1_15_1));
        float t0_0 = fma_0;
        float t1_0 = fma_1_0;
        band = band | ((fminf(t0_0, t1_0) == 0.0f) ? 2 : 0);
        float2 s_15_1 = make_float2(max_noftz(t0_0, 0.0f), max_noftz(t1_0, 0.0f));
        float2 act_15_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_15_1) : "l"(*(const unsigned long long*)&cvt_f32_17_1), "l"(*(const unsigned long long*)&s_15_1));
        float2 _f32x2_mul_a_93 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_15_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_15_1), s_15_1), cvt_f32_2_15_1);
        float2 mul_16_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_16_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_93), "l"(*(const unsigned long long*)&cvt_f32_1_15_1));
        float2 mul_1_15_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_15_1) : "l"(*(const unsigned long long*)&act_15_1), "l"(*(const unsigned long long*)&cvt_f32_2_15_1));
        __nv_bfloat162 bf16x2_16_1 = __float22bfloat162_rn(make_float2(mul_16_1.x, mul_16_1.y));
        __nv_bfloat162 bf16x2_1_15_1 = __float22bfloat162_rn(make_float2(mul_1_15_1.x, mul_1_15_1.y));
        dgate[1] = __as_u32(bf16x2_16_1);
        dup[1] = __as_u32(bf16x2_1_15_1);
        float2 cvt_f32_18 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 cvt_f32_1_16 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 cvt_f32_2_16 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float e0_16 = approx_exp2((-cvt_f32_18.x) * 1.442695f);
        float e1_16 = approx_exp2((-cvt_f32_18.y) * 1.442695f);
        float d0_16 = 1.0f + e0_16;
        float d1_16 = 1.0f + e1_16;
        float r0_16 = approx_rcp(d0_16);
        float r1_16 = approx_rcp(d1_16);
        float _fma_94;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_94) : "f"(-d0_16), "f"(r0_16), "f"(1.0f));
        float fma_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_2) : "f"(r0_16), "f"(_fma_94), "f"(r0_16));
        float _fma_95;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(-d1_16), "f"(r1_16), "f"(1.0f));
        float fma_1_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_1) : "f"(r1_16), "f"(_fma_95), "f"(r1_16));
        float t0_1 = fma_2;
        float t1_1 = fma_1_1;
        band = band | ((fminf(t0_1, t1_1) == 0.0f) ? 4 : 0);
        float2 s_16 = make_float2(max_noftz(t0_1, 0.0f), max_noftz(t1_1, 0.0f));
        float2 act_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_16) : "l"(*(const unsigned long long*)&cvt_f32_18), "l"(*(const unsigned long long*)&s_16));
        float2 _f32x2_mul_a_96 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_16, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_16), s_16), cvt_f32_2_16);
        float2 mul_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_96), "l"(*(const unsigned long long*)&cvt_f32_1_16));
        float2 mul_1_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_16) : "l"(*(const unsigned long long*)&act_16), "l"(*(const unsigned long long*)&cvt_f32_2_16));
        __nv_bfloat162 bf16x2_17 = __float22bfloat162_rn(make_float2(mul_17.x, mul_17.y));
        __nv_bfloat162 bf16x2_1_16 = __float22bfloat162_rn(make_float2(mul_1_16.x, mul_1_16.y));
        dgate[2] = __as_u32(bf16x2_17);
        dup[2] = __as_u32(bf16x2_1_16);
        float2 cvt_f32_19 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 cvt_f32_1_17 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 cvt_f32_2_17 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float e0_17 = approx_exp2((-cvt_f32_19.x) * 1.442695f);
        float e1_17 = approx_exp2((-cvt_f32_19.y) * 1.442695f);
        float d0_17 = 1.0f + e0_17;
        float d1_17 = 1.0f + e1_17;
        float r0_17 = approx_rcp(d0_17);
        float r1_17 = approx_rcp(d1_17);
        float _fma_97;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(-d0_17), "f"(r0_17), "f"(1.0f));
        float fma_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_3) : "f"(r0_17), "f"(_fma_97), "f"(r0_17));
        float _fma_98;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(-d1_17), "f"(r1_17), "f"(1.0f));
        float fma_1_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_2) : "f"(r1_17), "f"(_fma_98), "f"(r1_17));
        float t0_2 = fma_3;
        float t1_2 = fma_1_2;
        band = band | ((fminf(t0_2, t1_2) == 0.0f) ? 8 : 0);
        float2 s_17 = make_float2(max_noftz(t0_2, 0.0f), max_noftz(t1_2, 0.0f));
        float2 act_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_17) : "l"(*(const unsigned long long*)&cvt_f32_19), "l"(*(const unsigned long long*)&s_17));
        float2 _f32x2_mul_a_99 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_17, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_17), s_17), cvt_f32_2_17);
        float2 mul_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_99), "l"(*(const unsigned long long*)&cvt_f32_1_17));
        float2 mul_1_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_17) : "l"(*(const unsigned long long*)&act_17), "l"(*(const unsigned long long*)&cvt_f32_2_17));
        __nv_bfloat162 bf16x2_18 = __float22bfloat162_rn(make_float2(mul_18.x, mul_18.y));
        __nv_bfloat162 bf16x2_1_17 = __float22bfloat162_rn(make_float2(mul_1_17.x, mul_1_17.y));
        dgate[3] = __as_u32(bf16x2_18);
        dup[3] = __as_u32(bf16x2_1_17);
        float2 cvt_f32_20 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 cvt_f32_1_18 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float2 cvt_f32_2_18 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[4]));
        float e0_18 = approx_exp2((-cvt_f32_20.x) * 1.442695f);
        float e1_18 = approx_exp2((-cvt_f32_20.y) * 1.442695f);
        float d0_18 = 1.0f + e0_18;
        float d1_18 = 1.0f + e1_18;
        float r0_18 = approx_rcp(d0_18);
        float r1_18 = approx_rcp(d1_18);
        float _fma_100;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_100) : "f"(-d0_18), "f"(r0_18), "f"(1.0f));
        float fma_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_4) : "f"(r0_18), "f"(_fma_100), "f"(r0_18));
        float _fma_101;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(-d1_18), "f"(r1_18), "f"(1.0f));
        float fma_1_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_3) : "f"(r1_18), "f"(_fma_101), "f"(r1_18));
        float t0_3 = fma_4;
        float t1_3 = fma_1_3;
        band = band | ((fminf(t0_3, t1_3) == 0.0f) ? 16 : 0);
        float2 s_18 = make_float2(max_noftz(t0_3, 0.0f), max_noftz(t1_3, 0.0f));
        float2 act_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_18) : "l"(*(const unsigned long long*)&cvt_f32_20), "l"(*(const unsigned long long*)&s_18));
        float2 _f32x2_mul_a_102 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_18, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_18), s_18), cvt_f32_2_18);
        float2 mul_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_102), "l"(*(const unsigned long long*)&cvt_f32_1_18));
        float2 mul_1_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_18) : "l"(*(const unsigned long long*)&act_18), "l"(*(const unsigned long long*)&cvt_f32_2_18));
        __nv_bfloat162 bf16x2_19 = __float22bfloat162_rn(make_float2(mul_19.x, mul_19.y));
        __nv_bfloat162 bf16x2_1_18 = __float22bfloat162_rn(make_float2(mul_1_18.x, mul_1_18.y));
        dgate[4] = __as_u32(bf16x2_19);
        dup[4] = __as_u32(bf16x2_1_18);
        float2 cvt_f32_21 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 cvt_f32_1_19 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float2 cvt_f32_2_19 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[5]));
        float e0_19 = approx_exp2((-cvt_f32_21.x) * 1.442695f);
        float e1_19 = approx_exp2((-cvt_f32_21.y) * 1.442695f);
        float d0_19 = 1.0f + e0_19;
        float d1_19 = 1.0f + e1_19;
        float r0_19 = approx_rcp(d0_19);
        float r1_19 = approx_rcp(d1_19);
        float _fma_103;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_103) : "f"(-d0_19), "f"(r0_19), "f"(1.0f));
        float fma_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_5) : "f"(r0_19), "f"(_fma_103), "f"(r0_19));
        float _fma_104;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_104) : "f"(-d1_19), "f"(r1_19), "f"(1.0f));
        float fma_1_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_4) : "f"(r1_19), "f"(_fma_104), "f"(r1_19));
        float t0_4 = fma_5;
        float t1_4 = fma_1_4;
        band = band | ((fminf(t0_4, t1_4) == 0.0f) ? 32 : 0);
        float2 s_19 = make_float2(max_noftz(t0_4, 0.0f), max_noftz(t1_4, 0.0f));
        float2 act_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_19) : "l"(*(const unsigned long long*)&cvt_f32_21), "l"(*(const unsigned long long*)&s_19));
        float2 _f32x2_mul_a_105 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_19, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_19), s_19), cvt_f32_2_19);
        float2 mul_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_105), "l"(*(const unsigned long long*)&cvt_f32_1_19));
        float2 mul_1_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_19) : "l"(*(const unsigned long long*)&act_19), "l"(*(const unsigned long long*)&cvt_f32_2_19));
        __nv_bfloat162 bf16x2_20 = __float22bfloat162_rn(make_float2(mul_20.x, mul_20.y));
        __nv_bfloat162 bf16x2_1_19 = __float22bfloat162_rn(make_float2(mul_1_19.x, mul_1_19.y));
        dgate[5] = __as_u32(bf16x2_20);
        dup[5] = __as_u32(bf16x2_1_19);
        float2 cvt_f32_22 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 cvt_f32_1_20 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float2 cvt_f32_2_20 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[6]));
        float e0_20 = approx_exp2((-cvt_f32_22.x) * 1.442695f);
        float e1_20 = approx_exp2((-cvt_f32_22.y) * 1.442695f);
        float d0_20 = 1.0f + e0_20;
        float d1_20 = 1.0f + e1_20;
        float r0_20 = approx_rcp(d0_20);
        float r1_20 = approx_rcp(d1_20);
        float _fma_106;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_106) : "f"(-d0_20), "f"(r0_20), "f"(1.0f));
        float fma_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_6) : "f"(r0_20), "f"(_fma_106), "f"(r0_20));
        float _fma_107;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(-d1_20), "f"(r1_20), "f"(1.0f));
        float fma_1_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_5) : "f"(r1_20), "f"(_fma_107), "f"(r1_20));
        float t0_5 = fma_6;
        float t1_5 = fma_1_5;
        band = band | ((fminf(t0_5, t1_5) == 0.0f) ? 64 : 0);
        float2 s_20 = make_float2(max_noftz(t0_5, 0.0f), max_noftz(t1_5, 0.0f));
        float2 act_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_20) : "l"(*(const unsigned long long*)&cvt_f32_22), "l"(*(const unsigned long long*)&s_20));
        float2 _f32x2_mul_a_108 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_20, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_20), s_20), cvt_f32_2_20);
        float2 mul_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_108), "l"(*(const unsigned long long*)&cvt_f32_1_20));
        float2 mul_1_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_20) : "l"(*(const unsigned long long*)&act_20), "l"(*(const unsigned long long*)&cvt_f32_2_20));
        __nv_bfloat162 bf16x2_21 = __float22bfloat162_rn(make_float2(mul_21.x, mul_21.y));
        __nv_bfloat162 bf16x2_1_20 = __float22bfloat162_rn(make_float2(mul_1_20.x, mul_1_20.y));
        dgate[6] = __as_u32(bf16x2_21);
        dup[6] = __as_u32(bf16x2_1_20);
        float2 cvt_f32_23 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 cvt_f32_1_21 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float2 cvt_f32_2_21 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[7]));
        float e0_21 = approx_exp2((-cvt_f32_23.x) * 1.442695f);
        float e1_21 = approx_exp2((-cvt_f32_23.y) * 1.442695f);
        float d0_21 = 1.0f + e0_21;
        float d1_21 = 1.0f + e1_21;
        float r0_21 = approx_rcp(d0_21);
        float r1_21 = approx_rcp(d1_21);
        float _fma_109;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_109) : "f"(-d0_21), "f"(r0_21), "f"(1.0f));
        float fma_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_7) : "f"(r0_21), "f"(_fma_109), "f"(r0_21));
        float _fma_110;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_110) : "f"(-d1_21), "f"(r1_21), "f"(1.0f));
        float fma_1_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_6) : "f"(r1_21), "f"(_fma_110), "f"(r1_21));
        float t0_6 = fma_7;
        float t1_6 = fma_1_6;
        band = band | ((fminf(t0_6, t1_6) == 0.0f) ? 128 : 0);
        float2 s_21 = make_float2(max_noftz(t0_6, 0.0f), max_noftz(t1_6, 0.0f));
        float2 act_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_21) : "l"(*(const unsigned long long*)&cvt_f32_23), "l"(*(const unsigned long long*)&s_21));
        float2 _f32x2_mul_a_111 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_21, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_21), s_21), cvt_f32_2_21);
        float2 mul_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_111), "l"(*(const unsigned long long*)&cvt_f32_1_21));
        float2 mul_1_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_21) : "l"(*(const unsigned long long*)&act_21), "l"(*(const unsigned long long*)&cvt_f32_2_21));
        __nv_bfloat162 bf16x2_22 = __float22bfloat162_rn(make_float2(mul_22.x, mul_22.y));
        __nv_bfloat162 bf16x2_1_21 = __float22bfloat162_rn(make_float2(mul_1_21.x, mul_1_21.y));
        dgate[7] = __as_u32(bf16x2_22);
        dup[7] = __as_u32(bf16x2_1_21);
        int band_hi = 0;
        float2 cvt_f32_24 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
        float2 cvt_f32_1_22 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[0]));
        float2 cvt_f32_2_22 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[0]));
        float e0_22 = approx_exp2((-cvt_f32_24.x) * 1.442695f);
        float e1_22 = approx_exp2((-cvt_f32_24.y) * 1.442695f);
        float d0_22 = 1.0f + e0_22;
        float d1_22 = 1.0f + e1_22;
        float r0_22 = approx_rcp(d0_22);
        float r1_22 = approx_rcp(d1_22);
        float _fma_112;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_112) : "f"(-d0_22), "f"(r0_22), "f"(1.0f));
        float fma_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_8) : "f"(r0_22), "f"(_fma_112), "f"(r0_22));
        float _fma_113;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_113) : "f"(-d1_22), "f"(r1_22), "f"(1.0f));
        float fma_1_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_7) : "f"(r1_22), "f"(_fma_113), "f"(r1_22));
        float t0_7 = fma_8;
        float t1_7 = fma_1_7;
        band_hi = band_hi | ((fminf(t0_7, t1_7) == 0.0f) ? 1 : 0);
        float2 s_22 = make_float2(max_noftz(t0_7, 0.0f), max_noftz(t1_7, 0.0f));
        float2 act_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_22) : "l"(*(const unsigned long long*)&cvt_f32_24), "l"(*(const unsigned long long*)&s_22));
        float2 _f32x2_mul_a_114 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_22, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_22), s_22), cvt_f32_2_22);
        float2 mul_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_114), "l"(*(const unsigned long long*)&cvt_f32_1_22));
        float2 mul_1_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_22) : "l"(*(const unsigned long long*)&act_22), "l"(*(const unsigned long long*)&cvt_f32_2_22));
        __nv_bfloat162 bf16x2_23 = __float22bfloat162_rn(make_float2(mul_23.x, mul_23.y));
        __nv_bfloat162 bf16x2_1_22 = __float22bfloat162_rn(make_float2(mul_1_22.x, mul_1_22.y));
        dgate_hi[0] = __as_u32(bf16x2_23);
        dup_hi[0] = __as_u32(bf16x2_1_22);
        float2 cvt_f32_25 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[1]));
        float2 cvt_f32_1_23 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[1]));
        float2 cvt_f32_2_23 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[1]));
        float e0_23 = approx_exp2((-cvt_f32_25.x) * 1.442695f);
        float e1_23 = approx_exp2((-cvt_f32_25.y) * 1.442695f);
        float d0_23 = 1.0f + e0_23;
        float d1_23 = 1.0f + e1_23;
        float r0_23 = approx_rcp(d0_23);
        float r1_23 = approx_rcp(d1_23);
        float _fma_115;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_115) : "f"(-d0_23), "f"(r0_23), "f"(1.0f));
        float fma_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_9) : "f"(r0_23), "f"(_fma_115), "f"(r0_23));
        float _fma_116;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_116) : "f"(-d1_23), "f"(r1_23), "f"(1.0f));
        float fma_1_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_8) : "f"(r1_23), "f"(_fma_116), "f"(r1_23));
        float t0_8 = fma_9;
        float t1_8 = fma_1_8;
        band_hi = band_hi | ((fminf(t0_8, t1_8) == 0.0f) ? 2 : 0);
        float2 s_23 = make_float2(max_noftz(t0_8, 0.0f), max_noftz(t1_8, 0.0f));
        float2 act_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_23) : "l"(*(const unsigned long long*)&cvt_f32_25), "l"(*(const unsigned long long*)&s_23));
        float2 _f32x2_mul_a_117 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_23, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_23), s_23), cvt_f32_2_23);
        float2 mul_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_117), "l"(*(const unsigned long long*)&cvt_f32_1_23));
        float2 mul_1_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_23) : "l"(*(const unsigned long long*)&act_23), "l"(*(const unsigned long long*)&cvt_f32_2_23));
        __nv_bfloat162 bf16x2_24 = __float22bfloat162_rn(make_float2(mul_24.x, mul_24.y));
        __nv_bfloat162 bf16x2_1_23 = __float22bfloat162_rn(make_float2(mul_1_23.x, mul_1_23.y));
        dgate_hi[1] = __as_u32(bf16x2_24);
        dup_hi[1] = __as_u32(bf16x2_1_23);
        float2 cvt_f32_26 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[2]));
        float2 cvt_f32_1_24 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[2]));
        float2 cvt_f32_2_24 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[2]));
        float e0_24 = approx_exp2((-cvt_f32_26.x) * 1.442695f);
        float e1_24 = approx_exp2((-cvt_f32_26.y) * 1.442695f);
        float d0_24 = 1.0f + e0_24;
        float d1_24 = 1.0f + e1_24;
        float r0_24 = approx_rcp(d0_24);
        float r1_24 = approx_rcp(d1_24);
        float _fma_118;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_118) : "f"(-d0_24), "f"(r0_24), "f"(1.0f));
        float fma_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_10) : "f"(r0_24), "f"(_fma_118), "f"(r0_24));
        float _fma_119;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_119) : "f"(-d1_24), "f"(r1_24), "f"(1.0f));
        float fma_1_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_9) : "f"(r1_24), "f"(_fma_119), "f"(r1_24));
        float t0_9 = fma_10;
        float t1_9 = fma_1_9;
        band_hi = band_hi | ((fminf(t0_9, t1_9) == 0.0f) ? 4 : 0);
        float2 s_24 = make_float2(max_noftz(t0_9, 0.0f), max_noftz(t1_9, 0.0f));
        float2 act_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_24) : "l"(*(const unsigned long long*)&cvt_f32_26), "l"(*(const unsigned long long*)&s_24));
        float2 _f32x2_mul_a_120 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_24, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_24), s_24), cvt_f32_2_24);
        float2 mul_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_120), "l"(*(const unsigned long long*)&cvt_f32_1_24));
        float2 mul_1_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_24) : "l"(*(const unsigned long long*)&act_24), "l"(*(const unsigned long long*)&cvt_f32_2_24));
        __nv_bfloat162 bf16x2_25 = __float22bfloat162_rn(make_float2(mul_25.x, mul_25.y));
        __nv_bfloat162 bf16x2_1_24 = __float22bfloat162_rn(make_float2(mul_1_24.x, mul_1_24.y));
        dgate_hi[2] = __as_u32(bf16x2_25);
        dup_hi[2] = __as_u32(bf16x2_1_24);
        float2 cvt_f32_27 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[3]));
        float2 cvt_f32_1_25 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[3]));
        float2 cvt_f32_2_25 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[3]));
        float e0_25 = approx_exp2((-cvt_f32_27.x) * 1.442695f);
        float e1_25 = approx_exp2((-cvt_f32_27.y) * 1.442695f);
        float d0_25 = 1.0f + e0_25;
        float d1_25 = 1.0f + e1_25;
        float r0_25 = approx_rcp(d0_25);
        float r1_25 = approx_rcp(d1_25);
        float _fma_121;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_121) : "f"(-d0_25), "f"(r0_25), "f"(1.0f));
        float fma_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_11) : "f"(r0_25), "f"(_fma_121), "f"(r0_25));
        float _fma_122;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_122) : "f"(-d1_25), "f"(r1_25), "f"(1.0f));
        float fma_1_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_10) : "f"(r1_25), "f"(_fma_122), "f"(r1_25));
        float t0_10 = fma_11;
        float t1_10 = fma_1_10;
        band_hi = band_hi | ((fminf(t0_10, t1_10) == 0.0f) ? 8 : 0);
        float2 s_25 = make_float2(max_noftz(t0_10, 0.0f), max_noftz(t1_10, 0.0f));
        float2 act_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_25) : "l"(*(const unsigned long long*)&cvt_f32_27), "l"(*(const unsigned long long*)&s_25));
        float2 _f32x2_mul_a_123 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_25, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_25), s_25), cvt_f32_2_25);
        float2 mul_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_123), "l"(*(const unsigned long long*)&cvt_f32_1_25));
        float2 mul_1_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_25) : "l"(*(const unsigned long long*)&act_25), "l"(*(const unsigned long long*)&cvt_f32_2_25));
        __nv_bfloat162 bf16x2_26 = __float22bfloat162_rn(make_float2(mul_26.x, mul_26.y));
        __nv_bfloat162 bf16x2_1_25 = __float22bfloat162_rn(make_float2(mul_1_25.x, mul_1_25.y));
        dgate_hi[3] = __as_u32(bf16x2_26);
        dup_hi[3] = __as_u32(bf16x2_1_25);
        float2 cvt_f32_28 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[4]));
        float2 cvt_f32_1_26 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[4]));
        float2 cvt_f32_2_26 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[4]));
        float e0_26 = approx_exp2((-cvt_f32_28.x) * 1.442695f);
        float e1_26 = approx_exp2((-cvt_f32_28.y) * 1.442695f);
        float d0_26 = 1.0f + e0_26;
        float d1_26 = 1.0f + e1_26;
        float r0_26 = approx_rcp(d0_26);
        float r1_26 = approx_rcp(d1_26);
        float _fma_124;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_124) : "f"(-d0_26), "f"(r0_26), "f"(1.0f));
        float fma_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_12) : "f"(r0_26), "f"(_fma_124), "f"(r0_26));
        float _fma_125;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(-d1_26), "f"(r1_26), "f"(1.0f));
        float fma_1_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_11) : "f"(r1_26), "f"(_fma_125), "f"(r1_26));
        float t0_11 = fma_12;
        float t1_11 = fma_1_11;
        band_hi = band_hi | ((fminf(t0_11, t1_11) == 0.0f) ? 16 : 0);
        float2 s_26 = make_float2(max_noftz(t0_11, 0.0f), max_noftz(t1_11, 0.0f));
        float2 act_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_26) : "l"(*(const unsigned long long*)&cvt_f32_28), "l"(*(const unsigned long long*)&s_26));
        float2 _f32x2_mul_a_126 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_26, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_26), s_26), cvt_f32_2_26);
        float2 mul_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_126), "l"(*(const unsigned long long*)&cvt_f32_1_26));
        float2 mul_1_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_26) : "l"(*(const unsigned long long*)&act_26), "l"(*(const unsigned long long*)&cvt_f32_2_26));
        __nv_bfloat162 bf16x2_27 = __float22bfloat162_rn(make_float2(mul_27.x, mul_27.y));
        __nv_bfloat162 bf16x2_1_26 = __float22bfloat162_rn(make_float2(mul_1_26.x, mul_1_26.y));
        dgate_hi[4] = __as_u32(bf16x2_27);
        dup_hi[4] = __as_u32(bf16x2_1_26);
        float2 cvt_f32_29 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[5]));
        float2 cvt_f32_1_27 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[5]));
        float2 cvt_f32_2_27 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[5]));
        float e0_27 = approx_exp2((-cvt_f32_29.x) * 1.442695f);
        float e1_27 = approx_exp2((-cvt_f32_29.y) * 1.442695f);
        float d0_27 = 1.0f + e0_27;
        float d1_27 = 1.0f + e1_27;
        float r0_27 = approx_rcp(d0_27);
        float r1_27 = approx_rcp(d1_27);
        float _fma_127;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_127) : "f"(-d0_27), "f"(r0_27), "f"(1.0f));
        float fma_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_13) : "f"(r0_27), "f"(_fma_127), "f"(r0_27));
        float _fma_128;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_128) : "f"(-d1_27), "f"(r1_27), "f"(1.0f));
        float fma_1_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_12) : "f"(r1_27), "f"(_fma_128), "f"(r1_27));
        float t0_12 = fma_13;
        float t1_12 = fma_1_12;
        band_hi = band_hi | ((fminf(t0_12, t1_12) == 0.0f) ? 32 : 0);
        float2 s_27 = make_float2(max_noftz(t0_12, 0.0f), max_noftz(t1_12, 0.0f));
        float2 act_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_27) : "l"(*(const unsigned long long*)&cvt_f32_29), "l"(*(const unsigned long long*)&s_27));
        float2 _f32x2_mul_a_129 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_27, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_27), s_27), cvt_f32_2_27);
        float2 mul_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_129), "l"(*(const unsigned long long*)&cvt_f32_1_27));
        float2 mul_1_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_27) : "l"(*(const unsigned long long*)&act_27), "l"(*(const unsigned long long*)&cvt_f32_2_27));
        __nv_bfloat162 bf16x2_28 = __float22bfloat162_rn(make_float2(mul_28.x, mul_28.y));
        __nv_bfloat162 bf16x2_1_27 = __float22bfloat162_rn(make_float2(mul_1_27.x, mul_1_27.y));
        dgate_hi[5] = __as_u32(bf16x2_28);
        dup_hi[5] = __as_u32(bf16x2_1_27);
        float2 cvt_f32_30 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[6]));
        float2 cvt_f32_1_28 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[6]));
        float2 cvt_f32_2_28 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[6]));
        float e0_28 = approx_exp2((-cvt_f32_30.x) * 1.442695f);
        float e1_28 = approx_exp2((-cvt_f32_30.y) * 1.442695f);
        float d0_28 = 1.0f + e0_28;
        float d1_28 = 1.0f + e1_28;
        float r0_28 = approx_rcp(d0_28);
        float r1_28 = approx_rcp(d1_28);
        float _fma_130;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_130) : "f"(-d0_28), "f"(r0_28), "f"(1.0f));
        float fma_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_14) : "f"(r0_28), "f"(_fma_130), "f"(r0_28));
        float _fma_131;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_131) : "f"(-d1_28), "f"(r1_28), "f"(1.0f));
        float fma_1_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_13) : "f"(r1_28), "f"(_fma_131), "f"(r1_28));
        float t0_13 = fma_14;
        float t1_13 = fma_1_13;
        band_hi = band_hi | ((fminf(t0_13, t1_13) == 0.0f) ? 64 : 0);
        float2 s_28 = make_float2(max_noftz(t0_13, 0.0f), max_noftz(t1_13, 0.0f));
        float2 act_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_28) : "l"(*(const unsigned long long*)&cvt_f32_30), "l"(*(const unsigned long long*)&s_28));
        float2 _f32x2_mul_a_132 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_28, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_28), s_28), cvt_f32_2_28);
        float2 mul_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_132), "l"(*(const unsigned long long*)&cvt_f32_1_28));
        float2 mul_1_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_28) : "l"(*(const unsigned long long*)&act_28), "l"(*(const unsigned long long*)&cvt_f32_2_28));
        __nv_bfloat162 bf16x2_29 = __float22bfloat162_rn(make_float2(mul_29.x, mul_29.y));
        __nv_bfloat162 bf16x2_1_28 = __float22bfloat162_rn(make_float2(mul_1_28.x, mul_1_28.y));
        dgate_hi[6] = __as_u32(bf16x2_29);
        dup_hi[6] = __as_u32(bf16x2_1_28);
        float2 cvt_f32_31 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[7]));
        float2 cvt_f32_1_29 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[7]));
        float2 cvt_f32_2_29 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[7]));
        float e0_29 = approx_exp2((-cvt_f32_31.x) * 1.442695f);
        float e1_29 = approx_exp2((-cvt_f32_31.y) * 1.442695f);
        float d0_29 = 1.0f + e0_29;
        float d1_29 = 1.0f + e1_29;
        float r0_29 = approx_rcp(d0_29);
        float r1_29 = approx_rcp(d1_29);
        float _fma_133;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_133) : "f"(-d0_29), "f"(r0_29), "f"(1.0f));
        float fma_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_15) : "f"(r0_29), "f"(_fma_133), "f"(r0_29));
        float _fma_134;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_134) : "f"(-d1_29), "f"(r1_29), "f"(1.0f));
        float fma_1_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_14) : "f"(r1_29), "f"(_fma_134), "f"(r1_29));
        float t0_14 = fma_15;
        float t1_14 = fma_1_14;
        band_hi = band_hi | ((fminf(t0_14, t1_14) == 0.0f) ? 128 : 0);
        float2 s_29 = make_float2(max_noftz(t0_14, 0.0f), max_noftz(t1_14, 0.0f));
        float2 act_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_29) : "l"(*(const unsigned long long*)&cvt_f32_31), "l"(*(const unsigned long long*)&s_29));
        float2 _f32x2_mul_a_135 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_29, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_29), s_29), cvt_f32_2_29);
        float2 mul_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_135), "l"(*(const unsigned long long*)&cvt_f32_1_29));
        float2 mul_1_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_29) : "l"(*(const unsigned long long*)&act_29), "l"(*(const unsigned long long*)&cvt_f32_2_29));
        __nv_bfloat162 bf16x2_30 = __float22bfloat162_rn(make_float2(mul_30.x, mul_30.y));
        __nv_bfloat162 bf16x2_1_29 = __float22bfloat162_rn(make_float2(mul_1_29.x, mul_1_29.y));
        dgate_hi[7] = __as_u32(bf16x2_30);
        dup_hi[7] = __as_u32(bf16x2_1_29);
        if (band != 0) {
            unsigned int _popc_0 = (unsigned int)__popc(band);
            #pragma unroll 1
            for (int _j = 0; _j < _popc_0; _j++) {
                int _find_first_set_136 = __ffs(band);
                int j = _find_first_set_136 - 1;
                band = band & band - 1;
                unsigned int _vec_load_6[1];
                {
                    _vec_load_6[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index + 2 * j) + 0);
                }
                unsigned int _vec_load_7[1];
                {
                    _vec_load_7[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index + K + 2 * j) + 0);
                }
                unsigned int _vec_load_8[1];
                {
                    _vec_load_8[0] = *reinterpret_cast<const unsigned int*>(grad + (grow * K + col + 2 * j) + 0);
                }
                float2 cvt_f32_32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[0]));
                float2 cvt_f32_1_30 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[0]));
                float2 cvt_f32_2_30 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_8[0]));
                float e0_30 = approx_exp2((-cvt_f32_32.x) * 1.442695f);
                float e1_30 = approx_exp2((-cvt_f32_32.y) * 1.442695f);
                float d0_30 = 1.0f + e0_30;
                float d1_30 = 1.0f + e1_30;
                float r0_30 = approx_rcp(d0_30);
                float r1_30 = approx_rcp(d1_30);
                float _fma_137;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_137) : "f"(-d0_30), "f"(r0_30), "f"(1.0f));
                float fma_16;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_16) : "f"(r0_30), "f"(_fma_137), "f"(r0_30));
                float _fma_138;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_138) : "f"(-d1_30), "f"(r1_30), "f"(1.0f));
                float fma_1_15;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_15) : "f"(r1_30), "f"(_fma_138), "f"(r1_30));
                float t0_15 = fma_16;
                float t1_15 = fma_1_15;
                float d4_1a;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a) : "f"(d0_30), "f"(0.25f));
                float r4_1a = approx_rcp(d4_1a);
                float _fma_140;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_140) : "f"(-d4_1a), "f"(r4_1a), "f"(1.0f));
                float _fma_139;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_139) : "f"(r4_1a), "f"(_fma_140), "f"(r4_1a));
                float big_1a;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a) : "f"(mul_f32_rn_noftz(_fma_139, 1.2676506e+30f)), "f"(1.4073749e+14f));
                float _add_141;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_141) : "f"(big_1a), "f"(8.388608e+06f));
                float whole_1a;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a) : "f"(_add_141), "f"(8.388608e+06f));
                float above_1a;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a) : "f"(-d4_1a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                float _sub_143;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_143) : "f"(big_1a), "f"(whole_1a));
                float _abs_142;
                asm("abs.f32 %0, %1;" : "=f"(_abs_142) : "f"(_sub_143));
                float _add_144;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_144) : "f"(big_1a), "f"(0.5f));
                float _sub_145;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_145) : "f"(big_1a), "f"(0.5f));
                float rounded_1a = ((_abs_142 == 0.5f) ? ((above_1a > 0.0f) ? _add_144 : _sub_145) : whole_1a);
                float u0 = ((d0_30 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a, 7.888609e-31f), 1.7763568e-15f) : t0_15);
                float d4_1a_0;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_0) : "f"(d1_30), "f"(0.25f));
                float r4_1a_0 = approx_rcp(d4_1a_0);
                float _fma_147;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_147) : "f"(-d4_1a_0), "f"(r4_1a_0), "f"(1.0f));
                float _fma_146;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_146) : "f"(r4_1a_0), "f"(_fma_147), "f"(r4_1a_0));
                float big_1a_0;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_0) : "f"(mul_f32_rn_noftz(_fma_146, 1.2676506e+30f)), "f"(1.4073749e+14f));
                float _add_148;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_148) : "f"(big_1a_0), "f"(8.388608e+06f));
                float whole_1a_0;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_0) : "f"(_add_148), "f"(8.388608e+06f));
                float above_1a_0;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_0) : "f"(-d4_1a_0), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_0, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                float _sub_150;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_150) : "f"(big_1a_0), "f"(whole_1a_0));
                float _abs_149;
                asm("abs.f32 %0, %1;" : "=f"(_abs_149) : "f"(_sub_150));
                float _add_151;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_151) : "f"(big_1a_0), "f"(0.5f));
                float _sub_152;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_152) : "f"(big_1a_0), "f"(0.5f));
                float rounded_1a_0 = ((_abs_149 == 0.5f) ? ((above_1a_0 > 0.0f) ? _add_151 : _sub_152) : whole_1a_0);
                float u1 = ((d1_30 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_0, 7.888609e-31f), 1.7763568e-15f) : t1_15);
                float2 s_30 = make_float2(max_noftz(u0, 0.0f), max_noftz(u1, 0.0f));
                float2 act_30;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_30) : "l"(*(const unsigned long long*)&cvt_f32_32), "l"(*(const unsigned long long*)&s_30));
                float2 _f32x2_mul_a_153 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_30, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_30), s_30), cvt_f32_2_30);
                float2 mul_31;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_153), "l"(*(const unsigned long long*)&cvt_f32_1_30));
                float2 mul_1_30;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_30) : "l"(*(const unsigned long long*)&act_30), "l"(*(const unsigned long long*)&cvt_f32_2_30));
                __nv_bfloat162 bf16x2_31 = __float22bfloat162_rn(make_float2(mul_31.x, mul_31.y));
                __nv_bfloat162 bf16x2_1_30 = __float22bfloat162_rn(make_float2(mul_1_30.x, mul_1_30.y));
                unsigned int dgate_new = __as_u32(bf16x2_31);
                unsigned int dup_new = __as_u32(bf16x2_1_30);
                dgate[0] = ((j == 0) ? dgate_new : dgate[0]);
                dup[0] = ((j == 0) ? dup_new : dup[0]);
                dgate[1] = ((j == 1) ? dgate_new : dgate[1]);
                dup[1] = ((j == 1) ? dup_new : dup[1]);
                dgate[2] = ((j == 2) ? dgate_new : dgate[2]);
                dup[2] = ((j == 2) ? dup_new : dup[2]);
                dgate[3] = ((j == 3) ? dgate_new : dgate[3]);
                dup[3] = ((j == 3) ? dup_new : dup[3]);
                dgate[4] = ((j == 4) ? dgate_new : dgate[4]);
                dup[4] = ((j == 4) ? dup_new : dup[4]);
                dgate[5] = ((j == 5) ? dgate_new : dgate[5]);
                dup[5] = ((j == 5) ? dup_new : dup[5]);
                dgate[6] = ((j == 6) ? dgate_new : dgate[6]);
                dup[6] = ((j == 6) ? dup_new : dup[6]);
                dgate[7] = ((j == 7) ? dgate_new : dgate[7]);
                dup[7] = ((j == 7) ? dup_new : dup[7]);
            }
        }
        if (band_hi != 0) {
            unsigned int _popc_1 = (unsigned int)__popc(band_hi);
            #pragma unroll 1
            for (int _j_hi = 0; _j_hi < _popc_1; _j_hi++) {
                int _find_first_set_154 = __ffs(band_hi);
                int j_hi = _find_first_set_154 - 1;
                band_hi = band_hi & band_hi - 1;
                unsigned int _vec_load_9[1];
                {
                    _vec_load_9[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_hi + 2 * j_hi) + 0);
                }
                unsigned int _vec_load_10[1];
                {
                    _vec_load_10[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_hi + K + 2 * j_hi) + 0);
                }
                unsigned int _vec_load_11[1];
                {
                    _vec_load_11[0] = *reinterpret_cast<const unsigned int*>(grad + (grow_hi * K + col + 2 * j_hi) + 0);
                }
                float2 cvt_f32_33 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_9[0]));
                float2 cvt_f32_1_31 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_10[0]));
                float2 cvt_f32_2_31 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_11[0]));
                float e0_31 = approx_exp2((-cvt_f32_33.x) * 1.442695f);
                float e1_31 = approx_exp2((-cvt_f32_33.y) * 1.442695f);
                float d0_31 = 1.0f + e0_31;
                float d1_31 = 1.0f + e1_31;
                float r0_31 = approx_rcp(d0_31);
                float r1_31 = approx_rcp(d1_31);
                float _fma_155;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_155) : "f"(-d0_31), "f"(r0_31), "f"(1.0f));
                float fma_17;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_17) : "f"(r0_31), "f"(_fma_155), "f"(r0_31));
                float _fma_156;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_156) : "f"(-d1_31), "f"(r1_31), "f"(1.0f));
                float fma_1_16;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_16) : "f"(r1_31), "f"(_fma_156), "f"(r1_31));
                float t0_16 = fma_17;
                float t1_16 = fma_1_16;
                float d4_1a_1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_1) : "f"(d0_31), "f"(0.25f));
                float r4_1a_1 = approx_rcp(d4_1a_1);
                float _fma_158;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_158) : "f"(-d4_1a_1), "f"(r4_1a_1), "f"(1.0f));
                float _fma_157;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_157) : "f"(r4_1a_1), "f"(_fma_158), "f"(r4_1a_1));
                float big_1a_1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_1) : "f"(mul_f32_rn_noftz(_fma_157, 1.2676506e+30f)), "f"(1.4073749e+14f));
                float _add_159;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_159) : "f"(big_1a_1), "f"(8.388608e+06f));
                float whole_1a_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_1) : "f"(_add_159), "f"(8.388608e+06f));
                float above_1a_1;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_1) : "f"(-d4_1a_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                float _sub_161;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_161) : "f"(big_1a_1), "f"(whole_1a_1));
                float _abs_160;
                asm("abs.f32 %0, %1;" : "=f"(_abs_160) : "f"(_sub_161));
                float _add_162;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_162) : "f"(big_1a_1), "f"(0.5f));
                float _sub_163;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_163) : "f"(big_1a_1), "f"(0.5f));
                float rounded_1a_1 = ((_abs_160 == 0.5f) ? ((above_1a_1 > 0.0f) ? _add_162 : _sub_163) : whole_1a_1);
                float u0_1 = ((d0_31 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_1, 7.888609e-31f), 1.7763568e-15f) : t0_16);
                float d4_1a_1_1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_1_1) : "f"(d1_31), "f"(0.25f));
                float r4_1a_1_1 = approx_rcp(d4_1a_1_1);
                float _fma_165;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_165) : "f"(-d4_1a_1_1), "f"(r4_1a_1_1), "f"(1.0f));
                float _fma_164;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_164) : "f"(r4_1a_1_1), "f"(_fma_165), "f"(r4_1a_1_1));
                float big_1a_1_1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_1_1) : "f"(mul_f32_rn_noftz(_fma_164, 1.2676506e+30f)), "f"(1.4073749e+14f));
                float _add_166;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_166) : "f"(big_1a_1_1), "f"(8.388608e+06f));
                float whole_1a_1_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_1_1) : "f"(_add_166), "f"(8.388608e+06f));
                float above_1a_1_1;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_1_1) : "f"(-d4_1a_1_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_1_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                float _sub_168;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_168) : "f"(big_1a_1_1), "f"(whole_1a_1_1));
                float _abs_167;
                asm("abs.f32 %0, %1;" : "=f"(_abs_167) : "f"(_sub_168));
                float _add_169;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_169) : "f"(big_1a_1_1), "f"(0.5f));
                float _sub_170;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_170) : "f"(big_1a_1_1), "f"(0.5f));
                float rounded_1a_1_1 = ((_abs_167 == 0.5f) ? ((above_1a_1_1 > 0.0f) ? _add_169 : _sub_170) : whole_1a_1_1);
                float u1_1 = ((d1_31 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_1_1, 7.888609e-31f), 1.7763568e-15f) : t1_16);
                float2 s_31 = make_float2(max_noftz(u0_1, 0.0f), max_noftz(u1_1, 0.0f));
                float2 act_31;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_31) : "l"(*(const unsigned long long*)&cvt_f32_33), "l"(*(const unsigned long long*)&s_31));
                float2 _f32x2_mul_a_171 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_31, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_31), s_31), cvt_f32_2_31);
                float2 mul_32;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_32) : "l"(*(const unsigned long long*)&_f32x2_mul_a_171), "l"(*(const unsigned long long*)&cvt_f32_1_31));
                float2 mul_1_31;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_31) : "l"(*(const unsigned long long*)&act_31), "l"(*(const unsigned long long*)&cvt_f32_2_31));
                __nv_bfloat162 bf16x2_32 = __float22bfloat162_rn(make_float2(mul_32.x, mul_32.y));
                __nv_bfloat162 bf16x2_1_31 = __float22bfloat162_rn(make_float2(mul_1_31.x, mul_1_31.y));
                unsigned int dgate_new_hi = __as_u32(bf16x2_32);
                unsigned int dup_new_hi = __as_u32(bf16x2_1_31);
                dgate_hi[0] = ((j_hi == 0) ? dgate_new_hi : dgate_hi[0]);
                dup_hi[0] = ((j_hi == 0) ? dup_new_hi : dup_hi[0]);
                dgate_hi[1] = ((j_hi == 1) ? dgate_new_hi : dgate_hi[1]);
                dup_hi[1] = ((j_hi == 1) ? dup_new_hi : dup_hi[1]);
                dgate_hi[2] = ((j_hi == 2) ? dgate_new_hi : dgate_hi[2]);
                dup_hi[2] = ((j_hi == 2) ? dup_new_hi : dup_hi[2]);
                dgate_hi[3] = ((j_hi == 3) ? dgate_new_hi : dgate_hi[3]);
                dup_hi[3] = ((j_hi == 3) ? dup_new_hi : dup_hi[3]);
                dgate_hi[4] = ((j_hi == 4) ? dgate_new_hi : dgate_hi[4]);
                dup_hi[4] = ((j_hi == 4) ? dup_new_hi : dup_hi[4]);
                dgate_hi[5] = ((j_hi == 5) ? dgate_new_hi : dgate_hi[5]);
                dup_hi[5] = ((j_hi == 5) ? dup_new_hi : dup_hi[5]);
                dgate_hi[6] = ((j_hi == 6) ? dgate_new_hi : dgate_hi[6]);
                dup_hi[6] = ((j_hi == 6) ? dup_new_hi : dup_hi[6]);
                dgate_hi[7] = ((j_hi == 7) ? dgate_new_hi : dgate_hi[7]);
                dup_hi[7] = ((j_hi == 7) ? dup_new_hi : dup_hi[7]);
            }
        }
    }
    int scale_cols = K >> 6;
    int scale_col = bx * 2 + blk;
    int out_index = grow * (2 * K) + col;
    __nv_bfloat162 amax = __as_bf16x2(0);
    __nv_bfloat162 _hxa_173 = amax;
    __nv_bfloat162 _hxb_173 = __as_bf16x2(dgate[0]);
    unsigned int _hxr_173;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_173) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_173)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_173)));
    __nv_bfloat162 _max_172 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_173);
    amax = _max_172;
    __nv_bfloat162 _hxa_175 = amax;
    __nv_bfloat162 _hxb_175 = __as_bf16x2(dgate[1]);
    unsigned int _hxr_175;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_175) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_175)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_175)));
    __nv_bfloat162 _max_174 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_175);
    amax = _max_174;
    __nv_bfloat162 _hxa_177 = amax;
    __nv_bfloat162 _hxb_177 = __as_bf16x2(dgate[2]);
    unsigned int _hxr_177;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_177) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_177)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_177)));
    __nv_bfloat162 _max_176 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_177);
    amax = _max_176;
    __nv_bfloat162 _hxa_179 = amax;
    __nv_bfloat162 _hxb_179 = __as_bf16x2(dgate[3]);
    unsigned int _hxr_179;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_179) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_179)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_179)));
    __nv_bfloat162 _max_178 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_179);
    amax = _max_178;
    __nv_bfloat162 _hxa_181 = amax;
    __nv_bfloat162 _hxb_181 = __as_bf16x2(dgate[4]);
    unsigned int _hxr_181;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_181) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_181)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_181)));
    __nv_bfloat162 _max_180 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_181);
    amax = _max_180;
    __nv_bfloat162 _hxa_183 = amax;
    __nv_bfloat162 _hxb_183 = __as_bf16x2(dgate[5]);
    unsigned int _hxr_183;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_183) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_183)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_183)));
    __nv_bfloat162 _max_182 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_183);
    amax = _max_182;
    __nv_bfloat162 _hxa_185 = amax;
    __nv_bfloat162 _hxb_185 = __as_bf16x2(dgate[6]);
    unsigned int _hxr_185;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_185) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_185)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_185)));
    __nv_bfloat162 _max_184 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_185);
    amax = _max_184;
    __nv_bfloat162 _hxa_187 = amax;
    __nv_bfloat162 _hxb_187 = __as_bf16x2(dgate[7]);
    unsigned int _hxr_187;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_187) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_187)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_187)));
    __nv_bfloat162 _max_186 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_187);
    amax = _max_186;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_189;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_189) : "r"(bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_189) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_188;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_188) : "h"(e4m3x2_bf16x2), "h"(_cvt_satfinite_e4m3x2_bf16x2_189));
    words[0] = _mov_b32_pack_u16x2_188;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_191;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_191) : "r"(bf16x2_mul_1_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_191) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_190;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_190) : "h"(e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_191));
    words[1] = _mov_b32_pack_u16x2_190;
    uint32_t bf16x2_mul_2;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2) : "r"(dgate[4]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_1) : "r"(dgate[5]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_193;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_193) : "r"(bf16x2_mul_1_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_193) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_192;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_192) : "h"(e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_193));
    words[2] = _mov_b32_pack_u16x2_192;
    uint32_t bf16x2_mul_3;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3) : "r"(dgate[6]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_2) : "r"(dgate[7]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_195;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_195) : "r"(bf16x2_mul_1_2));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_2);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_195) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_194;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_194) : "h"(e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_195));
    words[3] = _mov_b32_pack_u16x2_194;
    reinterpret_cast<int4*>(row_output + out_index)[0] = reinterpret_cast<int4*>(words + 0)[0];
    __nv_bfloat162 amax_0 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_197 = amax_0;
    __nv_bfloat162 _hxb_197 = __as_bf16x2(dup[0]);
    unsigned int _hxr_197;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_197) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_197)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_197)));
    __nv_bfloat162 _max_196 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_197);
    amax_0 = _max_196;
    __nv_bfloat162 _hxa_199 = amax_0;
    __nv_bfloat162 _hxb_199 = __as_bf16x2(dup[1]);
    unsigned int _hxr_199;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_199) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_199)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_199)));
    __nv_bfloat162 _max_198 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_199);
    amax_0 = _max_198;
    __nv_bfloat162 _hxa_201 = amax_0;
    __nv_bfloat162 _hxb_201 = __as_bf16x2(dup[2]);
    unsigned int _hxr_201;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_201) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_201)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_201)));
    __nv_bfloat162 _max_200 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_201);
    amax_0 = _max_200;
    __nv_bfloat162 _hxa_203 = amax_0;
    __nv_bfloat162 _hxb_203 = __as_bf16x2(dup[3]);
    unsigned int _hxr_203;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_203) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_203)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_203)));
    __nv_bfloat162 _max_202 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_203);
    amax_0 = _max_202;
    __nv_bfloat162 _hxa_205 = amax_0;
    __nv_bfloat162 _hxb_205 = __as_bf16x2(dup[4]);
    unsigned int _hxr_205;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_205) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_205)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_205)));
    __nv_bfloat162 _max_204 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_205);
    amax_0 = _max_204;
    __nv_bfloat162 _hxa_207 = amax_0;
    __nv_bfloat162 _hxb_207 = __as_bf16x2(dup[5]);
    unsigned int _hxr_207;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_207) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_207)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_207)));
    __nv_bfloat162 _max_206 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_207);
    amax_0 = _max_206;
    __nv_bfloat162 _hxa_209 = amax_0;
    __nv_bfloat162 _hxb_209 = __as_bf16x2(dup[6]);
    unsigned int _hxr_209;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_209) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_209)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_209)));
    __nv_bfloat162 _max_208 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_209);
    amax_0 = _max_208;
    __nv_bfloat162 _hxa_211 = amax_0;
    __nv_bfloat162 _hxb_211 = __as_bf16x2(dup[7]);
    unsigned int _hxr_211;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_211) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_211)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_211)));
    __nv_bfloat162 _max_210 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_211);
    amax_0 = _max_210;
    unsigned int mag_1 = __as_u32(amax_0) & 2147450879;
    unsigned int peer_2 = __shfl_xor_sync(0xFFFFFFFF, mag_1, 1);
    __nv_bfloat162 both_3 = __hmax2_nan(__as_bf16x2(mag_1), __as_bf16x2(peer_2));
    unsigned int folded_4 = __as_u32(__hmax2_nan(both_3, __as_bf16x2(__as_u32(both_3) >> 16)));
    unsigned int amax_bits_0 = (folded_4 & 65535) << 16;
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
    int scale_index_10 = ((grow >> 7) * scale_cols + (scale_col + (K >> 5) >> 2)) * 512 + (grow & 31) * 16 + (grow >> 5 & 3) * 4 + (scale_col + (K >> 5) & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + scale_index_10) + (0)) = (unsigned char)(scale_0);
    }
    unsigned int words_11[4];
    uint32_t bf16x2_mul_4;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4) : "r"(dup[0]), "r"(inv2_0));
    uint16_t e4m3x2_bf16x2_3;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_3) : "r"(bf16x2_mul_4));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_4);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_3) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_3;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_3) : "r"(dup[1]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_213;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_213) : "r"(bf16x2_mul_1_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_213) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_212;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_212) : "h"(e4m3x2_bf16x2_3), "h"(_cvt_satfinite_e4m3x2_bf16x2_213));
    words_11[0] = _mov_b32_pack_u16x2_212;
    uint32_t bf16x2_mul_5;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5) : "r"(dup[2]), "r"(inv2_0));
    uint16_t e4m3x2_bf16x2_4;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_4) : "r"(bf16x2_mul_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_4) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_4;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_4) : "r"(dup[3]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_215;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_215) : "r"(bf16x2_mul_1_4));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_4);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_215) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_214;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_214) : "h"(e4m3x2_bf16x2_4), "h"(_cvt_satfinite_e4m3x2_bf16x2_215));
    words_11[1] = _mov_b32_pack_u16x2_214;
    uint32_t bf16x2_mul_6;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6) : "r"(dup[4]), "r"(inv2_0));
    uint16_t e4m3x2_bf16x2_5;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_5) : "r"(bf16x2_mul_6));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_6);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_5) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_5;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_5) : "r"(dup[5]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_217;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_217) : "r"(bf16x2_mul_1_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_217) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_216;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_216) : "h"(e4m3x2_bf16x2_5), "h"(_cvt_satfinite_e4m3x2_bf16x2_217));
    words_11[2] = _mov_b32_pack_u16x2_216;
    uint32_t bf16x2_mul_7;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7) : "r"(dup[6]), "r"(inv2_0));
    uint16_t e4m3x2_bf16x2_6;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_6) : "r"(bf16x2_mul_7));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_7);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_6) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_6;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_6) : "r"(dup[7]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_219;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_219) : "r"(bf16x2_mul_1_6));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_6);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_219) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_218;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_218) : "h"(e4m3x2_bf16x2_6), "h"(_cvt_satfinite_e4m3x2_bf16x2_219));
    words_11[3] = _mov_b32_pack_u16x2_218;
    reinterpret_cast<int4*>(row_output + (out_index + K))[0] = reinterpret_cast<int4*>(words_11 + 0)[0];
    int out_index_hi = grow_hi * (2 * K) + col;
    __nv_bfloat162 amax_12 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_221 = amax_12;
    __nv_bfloat162 _hxb_221 = __as_bf16x2(dgate_hi[0]);
    unsigned int _hxr_221;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_221) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_221)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_221)));
    __nv_bfloat162 _max_220 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_221);
    amax_12 = _max_220;
    __nv_bfloat162 _hxa_223 = amax_12;
    __nv_bfloat162 _hxb_223 = __as_bf16x2(dgate_hi[1]);
    unsigned int _hxr_223;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_223) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_223)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_223)));
    __nv_bfloat162 _max_222 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_223);
    amax_12 = _max_222;
    __nv_bfloat162 _hxa_225 = amax_12;
    __nv_bfloat162 _hxb_225 = __as_bf16x2(dgate_hi[2]);
    unsigned int _hxr_225;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_225) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_225)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_225)));
    __nv_bfloat162 _max_224 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_225);
    amax_12 = _max_224;
    __nv_bfloat162 _hxa_227 = amax_12;
    __nv_bfloat162 _hxb_227 = __as_bf16x2(dgate_hi[3]);
    unsigned int _hxr_227;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_227) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_227)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_227)));
    __nv_bfloat162 _max_226 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_227);
    amax_12 = _max_226;
    __nv_bfloat162 _hxa_229 = amax_12;
    __nv_bfloat162 _hxb_229 = __as_bf16x2(dgate_hi[4]);
    unsigned int _hxr_229;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_229) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_229)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_229)));
    __nv_bfloat162 _max_228 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_229);
    amax_12 = _max_228;
    __nv_bfloat162 _hxa_231 = amax_12;
    __nv_bfloat162 _hxb_231 = __as_bf16x2(dgate_hi[5]);
    unsigned int _hxr_231;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_231) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_231)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_231)));
    __nv_bfloat162 _max_230 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_231);
    amax_12 = _max_230;
    __nv_bfloat162 _hxa_233 = amax_12;
    __nv_bfloat162 _hxb_233 = __as_bf16x2(dgate_hi[6]);
    unsigned int _hxr_233;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_233) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_233)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_233)));
    __nv_bfloat162 _max_232 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_233);
    amax_12 = _max_232;
    __nv_bfloat162 _hxa_235 = amax_12;
    __nv_bfloat162 _hxb_235 = __as_bf16x2(dgate_hi[7]);
    unsigned int _hxr_235;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_235) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_235)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_235)));
    __nv_bfloat162 _max_234 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_235);
    amax_12 = _max_234;
    unsigned int mag_13 = __as_u32(amax_12) & 2147450879;
    unsigned int peer_14 = __shfl_xor_sync(0xFFFFFFFF, mag_13, 1);
    __nv_bfloat162 both_15 = __hmax2_nan(__as_bf16x2(mag_13), __as_bf16x2(peer_14));
    unsigned int folded_16 = __as_u32(__hmax2_nan(both_15, __as_bf16x2(__as_u32(both_15) >> 16)));
    unsigned int amax_bits_1 = (folded_16 & 65535) << 16;
    int scale_i32_1 = (int)(amax_bits_1 + 2031616 >> 23) - 8;
    if (scale_i32_1 < 0) {
        scale_i32_1 = 0;
    }
    if ((amax_bits_1 & 2139095040) == 2139095040) {
        scale_i32_1 = 255;
    }
    unsigned int scale_1 = (unsigned int)scale_i32_1;
    unsigned int inv_1 = 254 - scale_1 << 7;
    if (scale_1 == 255) {
        inv_1 = 32704;
    }
    unsigned int inv2_1 = inv_1 | inv_1 << 16;
    int scale_index_22 = ((grow_hi >> 7) * scale_cols + (scale_col >> 2)) * 512 + (grow_hi & 31) * 16 + (grow_hi >> 5 & 3) * 4 + (scale_col & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + scale_index_22) + (0)) = (unsigned char)(scale_1);
    }
    unsigned int words_23[4];
    uint32_t bf16x2_mul_8;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_8) : "r"(dgate_hi[0]), "r"(inv2_1));
    uint16_t e4m3x2_bf16x2_7;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_7) : "r"(bf16x2_mul_8));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_8);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_7) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_7;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_7) : "r"(dgate_hi[1]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_237;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_237) : "r"(bf16x2_mul_1_7));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_7);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_237) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_236;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_236) : "h"(e4m3x2_bf16x2_7), "h"(_cvt_satfinite_e4m3x2_bf16x2_237));
    words_23[0] = _mov_b32_pack_u16x2_236;
    uint32_t bf16x2_mul_9;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_9) : "r"(dgate_hi[2]), "r"(inv2_1));
    uint16_t e4m3x2_bf16x2_8;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_8) : "r"(bf16x2_mul_9));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_9);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_8) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_8;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_8) : "r"(dgate_hi[3]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_239;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_239) : "r"(bf16x2_mul_1_8));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_8);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_239) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_238;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_238) : "h"(e4m3x2_bf16x2_8), "h"(_cvt_satfinite_e4m3x2_bf16x2_239));
    words_23[1] = _mov_b32_pack_u16x2_238;
    uint32_t bf16x2_mul_10;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_10) : "r"(dgate_hi[4]), "r"(inv2_1));
    uint16_t e4m3x2_bf16x2_9;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_9) : "r"(bf16x2_mul_10));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_10);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_9) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_9;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_9) : "r"(dgate_hi[5]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_241;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_241) : "r"(bf16x2_mul_1_9));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_9);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_241) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_240;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_240) : "h"(e4m3x2_bf16x2_9), "h"(_cvt_satfinite_e4m3x2_bf16x2_241));
    words_23[2] = _mov_b32_pack_u16x2_240;
    uint32_t bf16x2_mul_11;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_11) : "r"(dgate_hi[6]), "r"(inv2_1));
    uint16_t e4m3x2_bf16x2_10;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_10) : "r"(bf16x2_mul_11));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_11);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_10) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_10;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_10) : "r"(dgate_hi[7]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_243;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_243) : "r"(bf16x2_mul_1_10));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_10);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_243) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_242;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_242) : "h"(e4m3x2_bf16x2_10), "h"(_cvt_satfinite_e4m3x2_bf16x2_243));
    words_23[3] = _mov_b32_pack_u16x2_242;
    reinterpret_cast<int4*>(row_output + out_index_hi)[0] = reinterpret_cast<int4*>(words_23 + 0)[0];
    __nv_bfloat162 amax_24 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_245 = amax_24;
    __nv_bfloat162 _hxb_245 = __as_bf16x2(dup_hi[0]);
    unsigned int _hxr_245;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_245) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_245)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_245)));
    __nv_bfloat162 _max_244 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_245);
    amax_24 = _max_244;
    __nv_bfloat162 _hxa_247 = amax_24;
    __nv_bfloat162 _hxb_247 = __as_bf16x2(dup_hi[1]);
    unsigned int _hxr_247;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_247) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_247)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_247)));
    __nv_bfloat162 _max_246 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_247);
    amax_24 = _max_246;
    __nv_bfloat162 _hxa_249 = amax_24;
    __nv_bfloat162 _hxb_249 = __as_bf16x2(dup_hi[2]);
    unsigned int _hxr_249;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_249) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_249)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_249)));
    __nv_bfloat162 _max_248 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_249);
    amax_24 = _max_248;
    __nv_bfloat162 _hxa_251 = amax_24;
    __nv_bfloat162 _hxb_251 = __as_bf16x2(dup_hi[3]);
    unsigned int _hxr_251;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_251) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_251)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_251)));
    __nv_bfloat162 _max_250 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_251);
    amax_24 = _max_250;
    __nv_bfloat162 _hxa_253 = amax_24;
    __nv_bfloat162 _hxb_253 = __as_bf16x2(dup_hi[4]);
    unsigned int _hxr_253;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_253) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_253)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_253)));
    __nv_bfloat162 _max_252 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_253);
    amax_24 = _max_252;
    __nv_bfloat162 _hxa_255 = amax_24;
    __nv_bfloat162 _hxb_255 = __as_bf16x2(dup_hi[5]);
    unsigned int _hxr_255;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_255) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_255)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_255)));
    __nv_bfloat162 _max_254 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_255);
    amax_24 = _max_254;
    __nv_bfloat162 _hxa_257 = amax_24;
    __nv_bfloat162 _hxb_257 = __as_bf16x2(dup_hi[6]);
    unsigned int _hxr_257;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_257) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_257)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_257)));
    __nv_bfloat162 _max_256 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_257);
    amax_24 = _max_256;
    __nv_bfloat162 _hxa_259 = amax_24;
    __nv_bfloat162 _hxb_259 = __as_bf16x2(dup_hi[7]);
    unsigned int _hxr_259;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_259) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_259)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_259)));
    __nv_bfloat162 _max_258 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_259);
    amax_24 = _max_258;
    unsigned int mag_25 = __as_u32(amax_24) & 2147450879;
    unsigned int peer_26 = __shfl_xor_sync(0xFFFFFFFF, mag_25, 1);
    __nv_bfloat162 both_27 = __hmax2_nan(__as_bf16x2(mag_25), __as_bf16x2(peer_26));
    unsigned int folded_28 = __as_u32(__hmax2_nan(both_27, __as_bf16x2(__as_u32(both_27) >> 16)));
    unsigned int amax_bits_2 = (folded_28 & 65535) << 16;
    int scale_i32_2 = (int)(amax_bits_2 + 2031616 >> 23) - 8;
    if (scale_i32_2 < 0) {
        scale_i32_2 = 0;
    }
    if ((amax_bits_2 & 2139095040) == 2139095040) {
        scale_i32_2 = 255;
    }
    unsigned int scale_2 = (unsigned int)scale_i32_2;
    unsigned int inv_2 = 254 - scale_2 << 7;
    if (scale_2 == 255) {
        inv_2 = 32704;
    }
    unsigned int inv2_2 = inv_2 | inv_2 << 16;
    int scale_index_34 = ((grow_hi >> 7) * scale_cols + (scale_col + (K >> 5) >> 2)) * 512 + (grow_hi & 31) * 16 + (grow_hi >> 5 & 3) * 4 + (scale_col + (K >> 5) & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + scale_index_34) + (0)) = (unsigned char)(scale_2);
    }
    unsigned int words_35[4];
    uint32_t bf16x2_mul_12;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_12) : "r"(dup_hi[0]), "r"(inv2_2));
    uint16_t e4m3x2_bf16x2_11;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_11) : "r"(bf16x2_mul_12));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_12);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_11) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_11;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_11) : "r"(dup_hi[1]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_261;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_261) : "r"(bf16x2_mul_1_11));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_11);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_261) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_260;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_260) : "h"(e4m3x2_bf16x2_11), "h"(_cvt_satfinite_e4m3x2_bf16x2_261));
    words_35[0] = _mov_b32_pack_u16x2_260;
    uint32_t bf16x2_mul_13;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_13) : "r"(dup_hi[2]), "r"(inv2_2));
    uint16_t e4m3x2_bf16x2_12;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_12) : "r"(bf16x2_mul_13));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_13);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_12) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_12;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_12) : "r"(dup_hi[3]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_263;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_263) : "r"(bf16x2_mul_1_12));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_12);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_263) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_262;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_262) : "h"(e4m3x2_bf16x2_12), "h"(_cvt_satfinite_e4m3x2_bf16x2_263));
    words_35[1] = _mov_b32_pack_u16x2_262;
    uint32_t bf16x2_mul_14;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_14) : "r"(dup_hi[4]), "r"(inv2_2));
    uint16_t e4m3x2_bf16x2_13;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_13) : "r"(bf16x2_mul_14));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_14);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_13) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_13;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_13) : "r"(dup_hi[5]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_265;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_265) : "r"(bf16x2_mul_1_13));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_13);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_265) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_264;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_264) : "h"(e4m3x2_bf16x2_13), "h"(_cvt_satfinite_e4m3x2_bf16x2_265));
    words_35[2] = _mov_b32_pack_u16x2_264;
    uint32_t bf16x2_mul_15;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_15) : "r"(dup_hi[6]), "r"(inv2_2));
    uint16_t e4m3x2_bf16x2_14;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_14) : "r"(bf16x2_mul_15));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_15);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_14) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_14;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_14) : "r"(dup_hi[7]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_267;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_267) : "r"(bf16x2_mul_1_14));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_14);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_267) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_266;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_266) : "h"(e4m3x2_bf16x2_14), "h"(_cvt_satfinite_e4m3x2_bf16x2_267));
    words_35[3] = _mov_b32_pack_u16x2_266;
    reinterpret_cast<int4*>(row_output + (out_index_hi + K))[0] = reinterpret_cast<int4*>(words_35 + 0)[0];
}

} // extern "C"
