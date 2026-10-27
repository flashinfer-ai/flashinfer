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
        float2 _cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 _cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 _cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float e0_1 = approx_exp2((-_cvt_f32_1.x) * 1.442695f);
        float e1_1 = approx_exp2((-_cvt_f32_1.y) * 1.442695f);
        float d0_1 = 1.0f + e0_1;
        float d1_1 = 1.0f + e1_1;
        float r0_1 = approx_rcp(d0_1);
        float r1_1 = approx_rcp(d1_1);
        float _fma_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(-d0_1), "f"(r0_1), "f"(1.0f));
        float _fma_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(r0_1), "f"(_fma_9), "f"(r0_1));
        float _fma_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(-d1_1), "f"(r1_1), "f"(1.0f));
        float _fma_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(r1_1), "f"(_fma_11), "f"(r1_1));
        float2 s_0 = make_float2(_fma_8, _fma_10);
        float2 act_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_0) : "l"(*(const unsigned long long*)&_cvt_f32_1), "l"(*(const unsigned long long*)&s_0));
        float2 _f32x2_mul_a_12 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_0), s_0), _cvt_f32_3);
        float2 _mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_12), "l"(*(const unsigned long long*)&_cvt_f32_2));
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
        float _fma_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(-d0_3), "f"(r0_3), "f"(1.0f));
        float _fma_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(r0_3), "f"(_fma_14), "f"(r0_3));
        float _fma_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(-d1_3), "f"(r1_3), "f"(1.0f));
        float _fma_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(r1_3), "f"(_fma_16), "f"(r1_3));
        float2 s_2 = make_float2(_fma_13, _fma_15);
        float2 act_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_2) : "l"(*(const unsigned long long*)&_cvt_f32_4), "l"(*(const unsigned long long*)&s_2));
        float2 _f32x2_mul_a_17 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_2), s_2), _cvt_f32_6);
        float2 _mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&_cvt_f32_5));
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
        float _fma_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(-d0_5), "f"(r0_5), "f"(1.0f));
        float _fma_18;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(r0_5), "f"(_fma_19), "f"(r0_5));
        float _fma_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(-d1_5), "f"(r1_5), "f"(1.0f));
        float _fma_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(r1_5), "f"(_fma_21), "f"(r1_5));
        float2 s_4 = make_float2(_fma_18, _fma_20);
        float2 act_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_4) : "l"(*(const unsigned long long*)&_cvt_f32_7), "l"(*(const unsigned long long*)&s_4));
        float2 _f32x2_mul_a_22 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_4), s_4), _cvt_f32_9);
        float2 _mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_22), "l"(*(const unsigned long long*)&_cvt_f32_8));
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
        float _fma_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(-d0_7), "f"(r0_7), "f"(1.0f));
        float _fma_23;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(r0_7), "f"(_fma_24), "f"(r0_7));
        float _fma_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(-d1_7), "f"(r1_7), "f"(1.0f));
        float _fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(r1_7), "f"(_fma_26), "f"(r1_7));
        float2 s_6 = make_float2(_fma_23, _fma_25);
        float2 act_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_6) : "l"(*(const unsigned long long*)&_cvt_f32_10), "l"(*(const unsigned long long*)&s_6));
        float2 _f32x2_mul_a_27 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_6), s_6), _cvt_f32_12);
        float2 _mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_27), "l"(*(const unsigned long long*)&_cvt_f32_11));
        float2 _mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_7) : "l"(*(const unsigned long long*)&act_6), "l"(*(const unsigned long long*)&_cvt_f32_12));
        __nv_bfloat162 _bf16x2_6 = __float22bfloat162_rn(make_float2(_mul_6.x, _mul_6.y));
        __nv_bfloat162 _bf16x2_7 = __float22bfloat162_rn(make_float2(_mul_7.x, _mul_7.y));
        dgate[3] = __as_u32(_bf16x2_6);
        dup[3] = __as_u32(_bf16x2_7);
        float2 _cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 _cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float2 _cvt_f32_15 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[4]));
        float e0_9 = approx_exp2((-_cvt_f32_13.x) * 1.442695f);
        float e1_9 = approx_exp2((-_cvt_f32_13.y) * 1.442695f);
        float d0_9 = 1.0f + e0_9;
        float d1_9 = 1.0f + e1_9;
        float r0_9 = approx_rcp(d0_9);
        float r1_9 = approx_rcp(d1_9);
        float _fma_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(-d0_9), "f"(r0_9), "f"(1.0f));
        float _fma_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(r0_9), "f"(_fma_29), "f"(r0_9));
        float _fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(-d1_9), "f"(r1_9), "f"(1.0f));
        float _fma_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(r1_9), "f"(_fma_31), "f"(r1_9));
        float2 s_8 = make_float2(_fma_28, _fma_30);
        float2 act_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_8) : "l"(*(const unsigned long long*)&_cvt_f32_13), "l"(*(const unsigned long long*)&s_8));
        float2 _f32x2_mul_a_32 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_8, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_8), s_8), _cvt_f32_15);
        float2 _mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_32), "l"(*(const unsigned long long*)&_cvt_f32_14));
        float2 _mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_9) : "l"(*(const unsigned long long*)&act_8), "l"(*(const unsigned long long*)&_cvt_f32_15));
        __nv_bfloat162 _bf16x2_8 = __float22bfloat162_rn(make_float2(_mul_8.x, _mul_8.y));
        __nv_bfloat162 _bf16x2_9 = __float22bfloat162_rn(make_float2(_mul_9.x, _mul_9.y));
        dgate[4] = __as_u32(_bf16x2_8);
        dup[4] = __as_u32(_bf16x2_9);
        float2 _cvt_f32_16 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 _cvt_f32_17 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float2 _cvt_f32_18 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[5]));
        float e0_11 = approx_exp2((-_cvt_f32_16.x) * 1.442695f);
        float e1_11 = approx_exp2((-_cvt_f32_16.y) * 1.442695f);
        float d0_11 = 1.0f + e0_11;
        float d1_11 = 1.0f + e1_11;
        float r0_11 = approx_rcp(d0_11);
        float r1_11 = approx_rcp(d1_11);
        float _fma_34;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(-d0_11), "f"(r0_11), "f"(1.0f));
        float _fma_33;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(r0_11), "f"(_fma_34), "f"(r0_11));
        float _fma_36;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(-d1_11), "f"(r1_11), "f"(1.0f));
        float _fma_35;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(r1_11), "f"(_fma_36), "f"(r1_11));
        float2 s_10 = make_float2(_fma_33, _fma_35);
        float2 act_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_10) : "l"(*(const unsigned long long*)&_cvt_f32_16), "l"(*(const unsigned long long*)&s_10));
        float2 _f32x2_mul_a_37 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_10, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_10), s_10), _cvt_f32_18);
        float2 _mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_37), "l"(*(const unsigned long long*)&_cvt_f32_17));
        float2 _mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_11) : "l"(*(const unsigned long long*)&act_10), "l"(*(const unsigned long long*)&_cvt_f32_18));
        __nv_bfloat162 _bf16x2_10 = __float22bfloat162_rn(make_float2(_mul_10.x, _mul_10.y));
        __nv_bfloat162 _bf16x2_11 = __float22bfloat162_rn(make_float2(_mul_11.x, _mul_11.y));
        dgate[5] = __as_u32(_bf16x2_10);
        dup[5] = __as_u32(_bf16x2_11);
        float2 _cvt_f32_19 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 _cvt_f32_20 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float2 _cvt_f32_21 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[6]));
        float e0_13 = approx_exp2((-_cvt_f32_19.x) * 1.442695f);
        float e1_13 = approx_exp2((-_cvt_f32_19.y) * 1.442695f);
        float d0_13 = 1.0f + e0_13;
        float d1_13 = 1.0f + e1_13;
        float r0_13 = approx_rcp(d0_13);
        float r1_13 = approx_rcp(d1_13);
        float _fma_39;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(-d0_13), "f"(r0_13), "f"(1.0f));
        float _fma_38;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(r0_13), "f"(_fma_39), "f"(r0_13));
        float _fma_41;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(-d1_13), "f"(r1_13), "f"(1.0f));
        float _fma_40;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(r1_13), "f"(_fma_41), "f"(r1_13));
        float2 s_12 = make_float2(_fma_38, _fma_40);
        float2 act_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_12) : "l"(*(const unsigned long long*)&_cvt_f32_19), "l"(*(const unsigned long long*)&s_12));
        float2 _f32x2_mul_a_42 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_12, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_12), s_12), _cvt_f32_21);
        float2 _mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&_cvt_f32_20));
        float2 _mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_13) : "l"(*(const unsigned long long*)&act_12), "l"(*(const unsigned long long*)&_cvt_f32_21));
        __nv_bfloat162 _bf16x2_12 = __float22bfloat162_rn(make_float2(_mul_12.x, _mul_12.y));
        __nv_bfloat162 _bf16x2_13 = __float22bfloat162_rn(make_float2(_mul_13.x, _mul_13.y));
        dgate[6] = __as_u32(_bf16x2_12);
        dup[6] = __as_u32(_bf16x2_13);
        float2 _cvt_f32_22 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 _cvt_f32_23 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float2 _cvt_f32_24 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[7]));
        float e0_15 = approx_exp2((-_cvt_f32_22.x) * 1.442695f);
        float e1_15 = approx_exp2((-_cvt_f32_22.y) * 1.442695f);
        float d0_15 = 1.0f + e0_15;
        float d1_15 = 1.0f + e1_15;
        float r0_15 = approx_rcp(d0_15);
        float r1_15 = approx_rcp(d1_15);
        float _fma_44;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(-d0_15), "f"(r0_15), "f"(1.0f));
        float _fma_43;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(r0_15), "f"(_fma_44), "f"(r0_15));
        float _fma_46;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(-d1_15), "f"(r1_15), "f"(1.0f));
        float _fma_45;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(r1_15), "f"(_fma_46), "f"(r1_15));
        float2 s_14 = make_float2(_fma_43, _fma_45);
        float2 act_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_14) : "l"(*(const unsigned long long*)&_cvt_f32_22), "l"(*(const unsigned long long*)&s_14));
        float2 _f32x2_mul_a_47 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_14, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_14), s_14), _cvt_f32_24);
        float2 _mul_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_47), "l"(*(const unsigned long long*)&_cvt_f32_23));
        float2 _mul_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_15) : "l"(*(const unsigned long long*)&act_14), "l"(*(const unsigned long long*)&_cvt_f32_24));
        __nv_bfloat162 _bf16x2_14 = __float22bfloat162_rn(make_float2(_mul_14.x, _mul_14.y));
        __nv_bfloat162 _bf16x2_15 = __float22bfloat162_rn(make_float2(_mul_15.x, _mul_15.y));
        dgate[7] = __as_u32(_bf16x2_14);
        dup[7] = __as_u32(_bf16x2_15);
        float2 _cvt_f32_25 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
        float2 _cvt_f32_26 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[0]));
        float2 _cvt_f32_27 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[0]));
        float e0_17 = approx_exp2((-_cvt_f32_25.x) * 1.442695f);
        float e1_17 = approx_exp2((-_cvt_f32_25.y) * 1.442695f);
        float d0_17 = 1.0f + e0_17;
        float d1_17 = 1.0f + e1_17;
        float r0_17 = approx_rcp(d0_17);
        float r1_17 = approx_rcp(d1_17);
        float _fma_49;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(-d0_17), "f"(r0_17), "f"(1.0f));
        float _fma_48;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(r0_17), "f"(_fma_49), "f"(r0_17));
        float _fma_51;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(-d1_17), "f"(r1_17), "f"(1.0f));
        float _fma_50;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(r1_17), "f"(_fma_51), "f"(r1_17));
        float2 s_16 = make_float2(_fma_48, _fma_50);
        float2 act_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_16) : "l"(*(const unsigned long long*)&_cvt_f32_25), "l"(*(const unsigned long long*)&s_16));
        float2 _f32x2_mul_a_52 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_16, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_16), s_16), _cvt_f32_27);
        float2 _mul_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_52), "l"(*(const unsigned long long*)&_cvt_f32_26));
        float2 _mul_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_17) : "l"(*(const unsigned long long*)&act_16), "l"(*(const unsigned long long*)&_cvt_f32_27));
        __nv_bfloat162 _bf16x2_16 = __float22bfloat162_rn(make_float2(_mul_16.x, _mul_16.y));
        __nv_bfloat162 _bf16x2_17 = __float22bfloat162_rn(make_float2(_mul_17.x, _mul_17.y));
        dgate_hi[0] = __as_u32(_bf16x2_16);
        dup_hi[0] = __as_u32(_bf16x2_17);
        float2 _cvt_f32_28 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[1]));
        float2 _cvt_f32_29 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[1]));
        float2 _cvt_f32_30 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[1]));
        float e0_19 = approx_exp2((-_cvt_f32_28.x) * 1.442695f);
        float e1_19 = approx_exp2((-_cvt_f32_28.y) * 1.442695f);
        float d0_19 = 1.0f + e0_19;
        float d1_19 = 1.0f + e1_19;
        float r0_19 = approx_rcp(d0_19);
        float r1_19 = approx_rcp(d1_19);
        float _fma_54;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(-d0_19), "f"(r0_19), "f"(1.0f));
        float _fma_53;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(r0_19), "f"(_fma_54), "f"(r0_19));
        float _fma_56;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(-d1_19), "f"(r1_19), "f"(1.0f));
        float _fma_55;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(r1_19), "f"(_fma_56), "f"(r1_19));
        float2 s_18 = make_float2(_fma_53, _fma_55);
        float2 act_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_18) : "l"(*(const unsigned long long*)&_cvt_f32_28), "l"(*(const unsigned long long*)&s_18));
        float2 _f32x2_mul_a_57 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_18, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_18), s_18), _cvt_f32_30);
        float2 _mul_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_57), "l"(*(const unsigned long long*)&_cvt_f32_29));
        float2 _mul_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_19) : "l"(*(const unsigned long long*)&act_18), "l"(*(const unsigned long long*)&_cvt_f32_30));
        __nv_bfloat162 _bf16x2_18 = __float22bfloat162_rn(make_float2(_mul_18.x, _mul_18.y));
        __nv_bfloat162 _bf16x2_19 = __float22bfloat162_rn(make_float2(_mul_19.x, _mul_19.y));
        dgate_hi[1] = __as_u32(_bf16x2_18);
        dup_hi[1] = __as_u32(_bf16x2_19);
        float2 _cvt_f32_31 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[2]));
        float2 _cvt_f32_32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[2]));
        float2 _cvt_f32_33 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[2]));
        float e0_21 = approx_exp2((-_cvt_f32_31.x) * 1.442695f);
        float e1_21 = approx_exp2((-_cvt_f32_31.y) * 1.442695f);
        float d0_21 = 1.0f + e0_21;
        float d1_21 = 1.0f + e1_21;
        float r0_21 = approx_rcp(d0_21);
        float r1_21 = approx_rcp(d1_21);
        float _fma_59;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(-d0_21), "f"(r0_21), "f"(1.0f));
        float _fma_58;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(r0_21), "f"(_fma_59), "f"(r0_21));
        float _fma_61;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(-d1_21), "f"(r1_21), "f"(1.0f));
        float _fma_60;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(r1_21), "f"(_fma_61), "f"(r1_21));
        float2 s_20 = make_float2(_fma_58, _fma_60);
        float2 act_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_20) : "l"(*(const unsigned long long*)&_cvt_f32_31), "l"(*(const unsigned long long*)&s_20));
        float2 _f32x2_mul_a_62 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_20, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_20), s_20), _cvt_f32_33);
        float2 _mul_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&_cvt_f32_32));
        float2 _mul_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_21) : "l"(*(const unsigned long long*)&act_20), "l"(*(const unsigned long long*)&_cvt_f32_33));
        __nv_bfloat162 _bf16x2_20 = __float22bfloat162_rn(make_float2(_mul_20.x, _mul_20.y));
        __nv_bfloat162 _bf16x2_21 = __float22bfloat162_rn(make_float2(_mul_21.x, _mul_21.y));
        dgate_hi[2] = __as_u32(_bf16x2_20);
        dup_hi[2] = __as_u32(_bf16x2_21);
        float2 _cvt_f32_34 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[3]));
        float2 _cvt_f32_35 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[3]));
        float2 _cvt_f32_36 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[3]));
        float e0_23 = approx_exp2((-_cvt_f32_34.x) * 1.442695f);
        float e1_23 = approx_exp2((-_cvt_f32_34.y) * 1.442695f);
        float d0_23 = 1.0f + e0_23;
        float d1_23 = 1.0f + e1_23;
        float r0_23 = approx_rcp(d0_23);
        float r1_23 = approx_rcp(d1_23);
        float _fma_64;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(-d0_23), "f"(r0_23), "f"(1.0f));
        float _fma_63;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(r0_23), "f"(_fma_64), "f"(r0_23));
        float _fma_66;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(-d1_23), "f"(r1_23), "f"(1.0f));
        float _fma_65;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(r1_23), "f"(_fma_66), "f"(r1_23));
        float2 s_22 = make_float2(_fma_63, _fma_65);
        float2 act_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_22) : "l"(*(const unsigned long long*)&_cvt_f32_34), "l"(*(const unsigned long long*)&s_22));
        float2 _f32x2_mul_a_67 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_22, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_22), s_22), _cvt_f32_36);
        float2 _mul_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_67), "l"(*(const unsigned long long*)&_cvt_f32_35));
        float2 _mul_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_23) : "l"(*(const unsigned long long*)&act_22), "l"(*(const unsigned long long*)&_cvt_f32_36));
        __nv_bfloat162 _bf16x2_22 = __float22bfloat162_rn(make_float2(_mul_22.x, _mul_22.y));
        __nv_bfloat162 _bf16x2_23 = __float22bfloat162_rn(make_float2(_mul_23.x, _mul_23.y));
        dgate_hi[3] = __as_u32(_bf16x2_22);
        dup_hi[3] = __as_u32(_bf16x2_23);
        float2 _cvt_f32_37 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[4]));
        float2 _cvt_f32_38 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[4]));
        float2 _cvt_f32_39 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[4]));
        float e0_25 = approx_exp2((-_cvt_f32_37.x) * 1.442695f);
        float e1_25 = approx_exp2((-_cvt_f32_37.y) * 1.442695f);
        float d0_25 = 1.0f + e0_25;
        float d1_25 = 1.0f + e1_25;
        float r0_25 = approx_rcp(d0_25);
        float r1_25 = approx_rcp(d1_25);
        float _fma_69;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(-d0_25), "f"(r0_25), "f"(1.0f));
        float _fma_68;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(r0_25), "f"(_fma_69), "f"(r0_25));
        float _fma_71;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(-d1_25), "f"(r1_25), "f"(1.0f));
        float _fma_70;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(r1_25), "f"(_fma_71), "f"(r1_25));
        float2 s_24 = make_float2(_fma_68, _fma_70);
        float2 act_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_24) : "l"(*(const unsigned long long*)&_cvt_f32_37), "l"(*(const unsigned long long*)&s_24));
        float2 _f32x2_mul_a_72 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_24, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_24), s_24), _cvt_f32_39);
        float2 _mul_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_72), "l"(*(const unsigned long long*)&_cvt_f32_38));
        float2 _mul_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_25) : "l"(*(const unsigned long long*)&act_24), "l"(*(const unsigned long long*)&_cvt_f32_39));
        __nv_bfloat162 _bf16x2_24 = __float22bfloat162_rn(make_float2(_mul_24.x, _mul_24.y));
        __nv_bfloat162 _bf16x2_25 = __float22bfloat162_rn(make_float2(_mul_25.x, _mul_25.y));
        dgate_hi[4] = __as_u32(_bf16x2_24);
        dup_hi[4] = __as_u32(_bf16x2_25);
        float2 _cvt_f32_40 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[5]));
        float2 _cvt_f32_41 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[5]));
        float2 _cvt_f32_42 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[5]));
        float e0_27 = approx_exp2((-_cvt_f32_40.x) * 1.442695f);
        float e1_27 = approx_exp2((-_cvt_f32_40.y) * 1.442695f);
        float d0_27 = 1.0f + e0_27;
        float d1_27 = 1.0f + e1_27;
        float r0_27 = approx_rcp(d0_27);
        float r1_27 = approx_rcp(d1_27);
        float _fma_74;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(-d0_27), "f"(r0_27), "f"(1.0f));
        float _fma_73;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(r0_27), "f"(_fma_74), "f"(r0_27));
        float _fma_76;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(-d1_27), "f"(r1_27), "f"(1.0f));
        float _fma_75;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(r1_27), "f"(_fma_76), "f"(r1_27));
        float2 s_26 = make_float2(_fma_73, _fma_75);
        float2 act_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_26) : "l"(*(const unsigned long long*)&_cvt_f32_40), "l"(*(const unsigned long long*)&s_26));
        float2 _f32x2_mul_a_77 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_26, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_26), s_26), _cvt_f32_42);
        float2 _mul_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_77), "l"(*(const unsigned long long*)&_cvt_f32_41));
        float2 _mul_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_27) : "l"(*(const unsigned long long*)&act_26), "l"(*(const unsigned long long*)&_cvt_f32_42));
        __nv_bfloat162 _bf16x2_26 = __float22bfloat162_rn(make_float2(_mul_26.x, _mul_26.y));
        __nv_bfloat162 _bf16x2_27 = __float22bfloat162_rn(make_float2(_mul_27.x, _mul_27.y));
        dgate_hi[5] = __as_u32(_bf16x2_26);
        dup_hi[5] = __as_u32(_bf16x2_27);
        float2 _cvt_f32_43 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[6]));
        float2 _cvt_f32_44 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[6]));
        float2 _cvt_f32_45 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[6]));
        float e0_29 = approx_exp2((-_cvt_f32_43.x) * 1.442695f);
        float e1_29 = approx_exp2((-_cvt_f32_43.y) * 1.442695f);
        float d0_29 = 1.0f + e0_29;
        float d1_29 = 1.0f + e1_29;
        float r0_29 = approx_rcp(d0_29);
        float r1_29 = approx_rcp(d1_29);
        float _fma_79;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(-d0_29), "f"(r0_29), "f"(1.0f));
        float _fma_78;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_78) : "f"(r0_29), "f"(_fma_79), "f"(r0_29));
        float _fma_81;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(-d1_29), "f"(r1_29), "f"(1.0f));
        float _fma_80;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(r1_29), "f"(_fma_81), "f"(r1_29));
        float2 s_28 = make_float2(_fma_78, _fma_80);
        float2 act_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_28) : "l"(*(const unsigned long long*)&_cvt_f32_43), "l"(*(const unsigned long long*)&s_28));
        float2 _f32x2_mul_a_82 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_28, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_28), s_28), _cvt_f32_45);
        float2 _mul_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_82), "l"(*(const unsigned long long*)&_cvt_f32_44));
        float2 _mul_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_29) : "l"(*(const unsigned long long*)&act_28), "l"(*(const unsigned long long*)&_cvt_f32_45));
        __nv_bfloat162 _bf16x2_28 = __float22bfloat162_rn(make_float2(_mul_28.x, _mul_28.y));
        __nv_bfloat162 _bf16x2_29 = __float22bfloat162_rn(make_float2(_mul_29.x, _mul_29.y));
        dgate_hi[6] = __as_u32(_bf16x2_28);
        dup_hi[6] = __as_u32(_bf16x2_29);
        float2 _cvt_f32_46 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[7]));
        float2 _cvt_f32_47 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[7]));
        float2 _cvt_f32_48 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[7]));
        float e0_31 = approx_exp2((-_cvt_f32_46.x) * 1.442695f);
        float e1_31 = approx_exp2((-_cvt_f32_46.y) * 1.442695f);
        float d0_31 = 1.0f + e0_31;
        float d1_31 = 1.0f + e1_31;
        float r0_31 = approx_rcp(d0_31);
        float r1_31 = approx_rcp(d1_31);
        float _fma_84;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(-d0_31), "f"(r0_31), "f"(1.0f));
        float _fma_83;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(r0_31), "f"(_fma_84), "f"(r0_31));
        float _fma_86;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(-d1_31), "f"(r1_31), "f"(1.0f));
        float _fma_85;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(r1_31), "f"(_fma_86), "f"(r1_31));
        float2 s_30 = make_float2(_fma_83, _fma_85);
        float2 act_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_30) : "l"(*(const unsigned long long*)&_cvt_f32_46), "l"(*(const unsigned long long*)&s_30));
        float2 _f32x2_mul_a_87 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_30, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_30), s_30), _cvt_f32_48);
        float2 _mul_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_87), "l"(*(const unsigned long long*)&_cvt_f32_47));
        float2 _mul_31;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_31) : "l"(*(const unsigned long long*)&act_30), "l"(*(const unsigned long long*)&_cvt_f32_48));
        __nv_bfloat162 _bf16x2_30 = __float22bfloat162_rn(make_float2(_mul_30.x, _mul_30.y));
        __nv_bfloat162 _bf16x2_31 = __float22bfloat162_rn(make_float2(_mul_31.x, _mul_31.y));
        dgate_hi[7] = __as_u32(_bf16x2_30);
        dup_hi[7] = __as_u32(_bf16x2_31);
    } else {
        int band = 0;
        float2 _cvt_f32_49 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 _cvt_f32_50 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 _cvt_f32_51 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float e0_33 = approx_exp2((-_cvt_f32_49.x) * 1.442695f);
        float e1_33 = approx_exp2((-_cvt_f32_49.y) * 1.442695f);
        float d0_33 = 1.0f + e0_33;
        float d1_33 = 1.0f + e1_33;
        float r0_33 = approx_rcp(d0_33);
        float r1_33 = approx_rcp(d1_33);
        float _fma_88;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_88) : "f"(-d0_33), "f"(r0_33), "f"(1.0f));
        float _fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(r0_33), "f"(_fma_88), "f"(r0_33));
        float _fma_89;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(-d1_33), "f"(r1_33), "f"(1.0f));
        float _fma_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(r1_33), "f"(_fma_89), "f"(r1_33));
        float t0_33 = _fma_0;
        float t1_33 = _fma_1;
        band = band | ((fminf(t0_33, t1_33) == 0.0f) ? 1 : 0);
        float2 s_32 = make_float2(max_noftz(t0_33, 0.0f), max_noftz(t1_33, 0.0f));
        float2 act_32;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_32) : "l"(*(const unsigned long long*)&_cvt_f32_49), "l"(*(const unsigned long long*)&s_32));
        float2 _f32x2_mul_a_90 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_32, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_32), s_32), _cvt_f32_51);
        float2 _mul_32;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_32) : "l"(*(const unsigned long long*)&_f32x2_mul_a_90), "l"(*(const unsigned long long*)&_cvt_f32_50));
        float2 _mul_33;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_33) : "l"(*(const unsigned long long*)&act_32), "l"(*(const unsigned long long*)&_cvt_f32_51));
        __nv_bfloat162 _bf16x2_32 = __float22bfloat162_rn(make_float2(_mul_32.x, _mul_32.y));
        __nv_bfloat162 _bf16x2_33 = __float22bfloat162_rn(make_float2(_mul_33.x, _mul_33.y));
        dgate[0] = __as_u32(_bf16x2_32);
        dup[0] = __as_u32(_bf16x2_33);
        float2 _cvt_f32_52 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 _cvt_f32_53 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 _cvt_f32_54 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float e0_35 = approx_exp2((-_cvt_f32_52.x) * 1.442695f);
        float e1_35 = approx_exp2((-_cvt_f32_52.y) * 1.442695f);
        float d0_35 = 1.0f + e0_35;
        float d1_35 = 1.0f + e1_35;
        float r0_35 = approx_rcp(d0_35);
        float r1_35 = approx_rcp(d1_35);
        float _fma_91;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(-d0_35), "f"(r0_35), "f"(1.0f));
        float _fma_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(r0_35), "f"(_fma_91), "f"(r0_35));
        float _fma_92;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(-d1_35), "f"(r1_35), "f"(1.0f));
        float _fma_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(r1_35), "f"(_fma_92), "f"(r1_35));
        float t0_35 = _fma_2;
        float t1_35 = _fma_3;
        band = band | ((fminf(t0_35, t1_35) == 0.0f) ? 2 : 0);
        float2 s_34 = make_float2(max_noftz(t0_35, 0.0f), max_noftz(t1_35, 0.0f));
        float2 act_34;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_34) : "l"(*(const unsigned long long*)&_cvt_f32_52), "l"(*(const unsigned long long*)&s_34));
        float2 _f32x2_mul_a_93 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_34, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_34), s_34), _cvt_f32_54);
        float2 _mul_34;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_34) : "l"(*(const unsigned long long*)&_f32x2_mul_a_93), "l"(*(const unsigned long long*)&_cvt_f32_53));
        float2 _mul_35;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_35) : "l"(*(const unsigned long long*)&act_34), "l"(*(const unsigned long long*)&_cvt_f32_54));
        __nv_bfloat162 _bf16x2_34 = __float22bfloat162_rn(make_float2(_mul_34.x, _mul_34.y));
        __nv_bfloat162 _bf16x2_35 = __float22bfloat162_rn(make_float2(_mul_35.x, _mul_35.y));
        dgate[1] = __as_u32(_bf16x2_34);
        dup[1] = __as_u32(_bf16x2_35);
        float2 _cvt_f32_55 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 _cvt_f32_56 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 _cvt_f32_57 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float e0_37 = approx_exp2((-_cvt_f32_55.x) * 1.442695f);
        float e1_37 = approx_exp2((-_cvt_f32_55.y) * 1.442695f);
        float d0_37 = 1.0f + e0_37;
        float d1_37 = 1.0f + e1_37;
        float r0_37 = approx_rcp(d0_37);
        float r1_37 = approx_rcp(d1_37);
        float _fma_94;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_94) : "f"(-d0_37), "f"(r0_37), "f"(1.0f));
        float _fma_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(r0_37), "f"(_fma_94), "f"(r0_37));
        float _fma_95;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(-d1_37), "f"(r1_37), "f"(1.0f));
        float _fma_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(r1_37), "f"(_fma_95), "f"(r1_37));
        float t0_37 = _fma_4;
        float t1_37 = _fma_5;
        band = band | ((fminf(t0_37, t1_37) == 0.0f) ? 4 : 0);
        float2 s_36 = make_float2(max_noftz(t0_37, 0.0f), max_noftz(t1_37, 0.0f));
        float2 act_36;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_36) : "l"(*(const unsigned long long*)&_cvt_f32_55), "l"(*(const unsigned long long*)&s_36));
        float2 _f32x2_mul_a_96 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_36, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_36), s_36), _cvt_f32_57);
        float2 _mul_36;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_36) : "l"(*(const unsigned long long*)&_f32x2_mul_a_96), "l"(*(const unsigned long long*)&_cvt_f32_56));
        float2 _mul_37;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_37) : "l"(*(const unsigned long long*)&act_36), "l"(*(const unsigned long long*)&_cvt_f32_57));
        __nv_bfloat162 _bf16x2_36 = __float22bfloat162_rn(make_float2(_mul_36.x, _mul_36.y));
        __nv_bfloat162 _bf16x2_37 = __float22bfloat162_rn(make_float2(_mul_37.x, _mul_37.y));
        dgate[2] = __as_u32(_bf16x2_36);
        dup[2] = __as_u32(_bf16x2_37);
        float2 _cvt_f32_58 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 _cvt_f32_59 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 _cvt_f32_60 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float e0_39 = approx_exp2((-_cvt_f32_58.x) * 1.442695f);
        float e1_39 = approx_exp2((-_cvt_f32_58.y) * 1.442695f);
        float d0_39 = 1.0f + e0_39;
        float d1_39 = 1.0f + e1_39;
        float r0_39 = approx_rcp(d0_39);
        float r1_39 = approx_rcp(d1_39);
        float _fma_97;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(-d0_39), "f"(r0_39), "f"(1.0f));
        float _fma_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(r0_39), "f"(_fma_97), "f"(r0_39));
        float _fma_98;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(-d1_39), "f"(r1_39), "f"(1.0f));
        float _fma_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(r1_39), "f"(_fma_98), "f"(r1_39));
        float t0_39 = _fma_6;
        float t1_39 = _fma_7;
        band = band | ((fminf(t0_39, t1_39) == 0.0f) ? 8 : 0);
        float2 s_38 = make_float2(max_noftz(t0_39, 0.0f), max_noftz(t1_39, 0.0f));
        float2 act_38;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_38) : "l"(*(const unsigned long long*)&_cvt_f32_58), "l"(*(const unsigned long long*)&s_38));
        float2 _f32x2_mul_a_99 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_38, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_38), s_38), _cvt_f32_60);
        float2 _mul_38;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_38) : "l"(*(const unsigned long long*)&_f32x2_mul_a_99), "l"(*(const unsigned long long*)&_cvt_f32_59));
        float2 _mul_39;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_39) : "l"(*(const unsigned long long*)&act_38), "l"(*(const unsigned long long*)&_cvt_f32_60));
        __nv_bfloat162 _bf16x2_38 = __float22bfloat162_rn(make_float2(_mul_38.x, _mul_38.y));
        __nv_bfloat162 _bf16x2_39 = __float22bfloat162_rn(make_float2(_mul_39.x, _mul_39.y));
        dgate[3] = __as_u32(_bf16x2_38);
        dup[3] = __as_u32(_bf16x2_39);
        float2 _cvt_f32_61 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 _cvt_f32_62 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float2 _cvt_f32_63 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[4]));
        float e0_41 = approx_exp2((-_cvt_f32_61.x) * 1.442695f);
        float e1_41 = approx_exp2((-_cvt_f32_61.y) * 1.442695f);
        float d0_41 = 1.0f + e0_41;
        float d1_41 = 1.0f + e1_41;
        float r0_41 = approx_rcp(d0_41);
        float r1_41 = approx_rcp(d1_41);
        float _fma_100;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_100) : "f"(-d0_41), "f"(r0_41), "f"(1.0f));
        float _fma_8_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8_1) : "f"(r0_41), "f"(_fma_100), "f"(r0_41));
        float _fma_101;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(-d1_41), "f"(r1_41), "f"(1.0f));
        float _fma_9_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9_1) : "f"(r1_41), "f"(_fma_101), "f"(r1_41));
        float t0_41 = _fma_8_1;
        float t1_41 = _fma_9_1;
        band = band | ((fminf(t0_41, t1_41) == 0.0f) ? 16 : 0);
        float2 s_40 = make_float2(max_noftz(t0_41, 0.0f), max_noftz(t1_41, 0.0f));
        float2 act_40;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_40) : "l"(*(const unsigned long long*)&_cvt_f32_61), "l"(*(const unsigned long long*)&s_40));
        float2 _f32x2_mul_a_102 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_40, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_40), s_40), _cvt_f32_63);
        float2 _mul_40;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_40) : "l"(*(const unsigned long long*)&_f32x2_mul_a_102), "l"(*(const unsigned long long*)&_cvt_f32_62));
        float2 _mul_41;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_41) : "l"(*(const unsigned long long*)&act_40), "l"(*(const unsigned long long*)&_cvt_f32_63));
        __nv_bfloat162 _bf16x2_40 = __float22bfloat162_rn(make_float2(_mul_40.x, _mul_40.y));
        __nv_bfloat162 _bf16x2_41 = __float22bfloat162_rn(make_float2(_mul_41.x, _mul_41.y));
        dgate[4] = __as_u32(_bf16x2_40);
        dup[4] = __as_u32(_bf16x2_41);
        float2 _cvt_f32_64 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 _cvt_f32_65 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float2 _cvt_f32_66 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[5]));
        float e0_43 = approx_exp2((-_cvt_f32_64.x) * 1.442695f);
        float e1_43 = approx_exp2((-_cvt_f32_64.y) * 1.442695f);
        float d0_43 = 1.0f + e0_43;
        float d1_43 = 1.0f + e1_43;
        float r0_43 = approx_rcp(d0_43);
        float r1_43 = approx_rcp(d1_43);
        float _fma_103;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_103) : "f"(-d0_43), "f"(r0_43), "f"(1.0f));
        float _fma_10_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10_1) : "f"(r0_43), "f"(_fma_103), "f"(r0_43));
        float _fma_104;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_104) : "f"(-d1_43), "f"(r1_43), "f"(1.0f));
        float _fma_11_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11_1) : "f"(r1_43), "f"(_fma_104), "f"(r1_43));
        float t0_43 = _fma_10_1;
        float t1_43 = _fma_11_1;
        band = band | ((fminf(t0_43, t1_43) == 0.0f) ? 32 : 0);
        float2 s_42 = make_float2(max_noftz(t0_43, 0.0f), max_noftz(t1_43, 0.0f));
        float2 act_42;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_42) : "l"(*(const unsigned long long*)&_cvt_f32_64), "l"(*(const unsigned long long*)&s_42));
        float2 _f32x2_mul_a_105 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_42, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_42), s_42), _cvt_f32_66);
        float2 _mul_42;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_42) : "l"(*(const unsigned long long*)&_f32x2_mul_a_105), "l"(*(const unsigned long long*)&_cvt_f32_65));
        float2 _mul_43;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_43) : "l"(*(const unsigned long long*)&act_42), "l"(*(const unsigned long long*)&_cvt_f32_66));
        __nv_bfloat162 _bf16x2_42 = __float22bfloat162_rn(make_float2(_mul_42.x, _mul_42.y));
        __nv_bfloat162 _bf16x2_43 = __float22bfloat162_rn(make_float2(_mul_43.x, _mul_43.y));
        dgate[5] = __as_u32(_bf16x2_42);
        dup[5] = __as_u32(_bf16x2_43);
        float2 _cvt_f32_67 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 _cvt_f32_68 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float2 _cvt_f32_69 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[6]));
        float e0_45 = approx_exp2((-_cvt_f32_67.x) * 1.442695f);
        float e1_45 = approx_exp2((-_cvt_f32_67.y) * 1.442695f);
        float d0_45 = 1.0f + e0_45;
        float d1_45 = 1.0f + e1_45;
        float r0_45 = approx_rcp(d0_45);
        float r1_45 = approx_rcp(d1_45);
        float _fma_106;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_106) : "f"(-d0_45), "f"(r0_45), "f"(1.0f));
        float _fma_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(r0_45), "f"(_fma_106), "f"(r0_45));
        float _fma_107;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(-d1_45), "f"(r1_45), "f"(1.0f));
        float _fma_13_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13_1) : "f"(r1_45), "f"(_fma_107), "f"(r1_45));
        float t0_45 = _fma_12;
        float t1_45 = _fma_13_1;
        band = band | ((fminf(t0_45, t1_45) == 0.0f) ? 64 : 0);
        float2 s_44 = make_float2(max_noftz(t0_45, 0.0f), max_noftz(t1_45, 0.0f));
        float2 act_44;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_44) : "l"(*(const unsigned long long*)&_cvt_f32_67), "l"(*(const unsigned long long*)&s_44));
        float2 _f32x2_mul_a_108 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_44, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_44), s_44), _cvt_f32_69);
        float2 _mul_44;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_44) : "l"(*(const unsigned long long*)&_f32x2_mul_a_108), "l"(*(const unsigned long long*)&_cvt_f32_68));
        float2 _mul_45;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_45) : "l"(*(const unsigned long long*)&act_44), "l"(*(const unsigned long long*)&_cvt_f32_69));
        __nv_bfloat162 _bf16x2_44 = __float22bfloat162_rn(make_float2(_mul_44.x, _mul_44.y));
        __nv_bfloat162 _bf16x2_45 = __float22bfloat162_rn(make_float2(_mul_45.x, _mul_45.y));
        dgate[6] = __as_u32(_bf16x2_44);
        dup[6] = __as_u32(_bf16x2_45);
        float2 _cvt_f32_70 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 _cvt_f32_71 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float2 _cvt_f32_72 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[7]));
        float e0_47 = approx_exp2((-_cvt_f32_70.x) * 1.442695f);
        float e1_47 = approx_exp2((-_cvt_f32_70.y) * 1.442695f);
        float d0_47 = 1.0f + e0_47;
        float d1_47 = 1.0f + e1_47;
        float r0_47 = approx_rcp(d0_47);
        float r1_47 = approx_rcp(d1_47);
        float _fma_109;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_109) : "f"(-d0_47), "f"(r0_47), "f"(1.0f));
        float _fma_14_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14_1) : "f"(r0_47), "f"(_fma_109), "f"(r0_47));
        float _fma_110;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_110) : "f"(-d1_47), "f"(r1_47), "f"(1.0f));
        float _fma_15_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15_1) : "f"(r1_47), "f"(_fma_110), "f"(r1_47));
        float t0_47 = _fma_14_1;
        float t1_47 = _fma_15_1;
        band = band | ((fminf(t0_47, t1_47) == 0.0f) ? 128 : 0);
        float2 s_46 = make_float2(max_noftz(t0_47, 0.0f), max_noftz(t1_47, 0.0f));
        float2 act_46;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_46) : "l"(*(const unsigned long long*)&_cvt_f32_70), "l"(*(const unsigned long long*)&s_46));
        float2 _f32x2_mul_a_111 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_46, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_46), s_46), _cvt_f32_72);
        float2 _mul_46;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_46) : "l"(*(const unsigned long long*)&_f32x2_mul_a_111), "l"(*(const unsigned long long*)&_cvt_f32_71));
        float2 _mul_47;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_47) : "l"(*(const unsigned long long*)&act_46), "l"(*(const unsigned long long*)&_cvt_f32_72));
        __nv_bfloat162 _bf16x2_46 = __float22bfloat162_rn(make_float2(_mul_46.x, _mul_46.y));
        __nv_bfloat162 _bf16x2_47 = __float22bfloat162_rn(make_float2(_mul_47.x, _mul_47.y));
        dgate[7] = __as_u32(_bf16x2_46);
        dup[7] = __as_u32(_bf16x2_47);
        int band_hi = 0;
        float2 _cvt_f32_73 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
        float2 _cvt_f32_74 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[0]));
        float2 _cvt_f32_75 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[0]));
        float e0_49 = approx_exp2((-_cvt_f32_73.x) * 1.442695f);
        float e1_49 = approx_exp2((-_cvt_f32_73.y) * 1.442695f);
        float d0_49 = 1.0f + e0_49;
        float d1_49 = 1.0f + e1_49;
        float r0_49 = approx_rcp(d0_49);
        float r1_49 = approx_rcp(d1_49);
        float _fma_112;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_112) : "f"(-d0_49), "f"(r0_49), "f"(1.0f));
        float _fma_16_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16_1) : "f"(r0_49), "f"(_fma_112), "f"(r0_49));
        float _fma_113;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_113) : "f"(-d1_49), "f"(r1_49), "f"(1.0f));
        float _fma_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(r1_49), "f"(_fma_113), "f"(r1_49));
        float t0_49 = _fma_16_1;
        float t1_49 = _fma_17;
        band_hi = band_hi | ((fminf(t0_49, t1_49) == 0.0f) ? 1 : 0);
        float2 s_48 = make_float2(max_noftz(t0_49, 0.0f), max_noftz(t1_49, 0.0f));
        float2 act_48;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_48) : "l"(*(const unsigned long long*)&_cvt_f32_73), "l"(*(const unsigned long long*)&s_48));
        float2 _f32x2_mul_a_114 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_48, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_48), s_48), _cvt_f32_75);
        float2 _mul_48;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_48) : "l"(*(const unsigned long long*)&_f32x2_mul_a_114), "l"(*(const unsigned long long*)&_cvt_f32_74));
        float2 _mul_49;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_49) : "l"(*(const unsigned long long*)&act_48), "l"(*(const unsigned long long*)&_cvt_f32_75));
        __nv_bfloat162 _bf16x2_48 = __float22bfloat162_rn(make_float2(_mul_48.x, _mul_48.y));
        __nv_bfloat162 _bf16x2_49 = __float22bfloat162_rn(make_float2(_mul_49.x, _mul_49.y));
        dgate_hi[0] = __as_u32(_bf16x2_48);
        dup_hi[0] = __as_u32(_bf16x2_49);
        float2 _cvt_f32_76 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[1]));
        float2 _cvt_f32_77 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[1]));
        float2 _cvt_f32_78 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[1]));
        float e0_51 = approx_exp2((-_cvt_f32_76.x) * 1.442695f);
        float e1_51 = approx_exp2((-_cvt_f32_76.y) * 1.442695f);
        float d0_51 = 1.0f + e0_51;
        float d1_51 = 1.0f + e1_51;
        float r0_51 = approx_rcp(d0_51);
        float r1_51 = approx_rcp(d1_51);
        float _fma_115;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_115) : "f"(-d0_51), "f"(r0_51), "f"(1.0f));
        float _fma_18_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18_1) : "f"(r0_51), "f"(_fma_115), "f"(r0_51));
        float _fma_116;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_116) : "f"(-d1_51), "f"(r1_51), "f"(1.0f));
        float _fma_19_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19_1) : "f"(r1_51), "f"(_fma_116), "f"(r1_51));
        float t0_51 = _fma_18_1;
        float t1_51 = _fma_19_1;
        band_hi = band_hi | ((fminf(t0_51, t1_51) == 0.0f) ? 2 : 0);
        float2 s_50 = make_float2(max_noftz(t0_51, 0.0f), max_noftz(t1_51, 0.0f));
        float2 act_50;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_50) : "l"(*(const unsigned long long*)&_cvt_f32_76), "l"(*(const unsigned long long*)&s_50));
        float2 _f32x2_mul_a_117 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_50, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_50), s_50), _cvt_f32_78);
        float2 _mul_50;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_50) : "l"(*(const unsigned long long*)&_f32x2_mul_a_117), "l"(*(const unsigned long long*)&_cvt_f32_77));
        float2 _mul_51;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_51) : "l"(*(const unsigned long long*)&act_50), "l"(*(const unsigned long long*)&_cvt_f32_78));
        __nv_bfloat162 _bf16x2_50 = __float22bfloat162_rn(make_float2(_mul_50.x, _mul_50.y));
        __nv_bfloat162 _bf16x2_51 = __float22bfloat162_rn(make_float2(_mul_51.x, _mul_51.y));
        dgate_hi[1] = __as_u32(_bf16x2_50);
        dup_hi[1] = __as_u32(_bf16x2_51);
        float2 _cvt_f32_79 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[2]));
        float2 _cvt_f32_80 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[2]));
        float2 _cvt_f32_81 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[2]));
        float e0_53 = approx_exp2((-_cvt_f32_79.x) * 1.442695f);
        float e1_53 = approx_exp2((-_cvt_f32_79.y) * 1.442695f);
        float d0_53 = 1.0f + e0_53;
        float d1_53 = 1.0f + e1_53;
        float r0_53 = approx_rcp(d0_53);
        float r1_53 = approx_rcp(d1_53);
        float _fma_118;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_118) : "f"(-d0_53), "f"(r0_53), "f"(1.0f));
        float _fma_20_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20_1) : "f"(r0_53), "f"(_fma_118), "f"(r0_53));
        float _fma_119;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_119) : "f"(-d1_53), "f"(r1_53), "f"(1.0f));
        float _fma_21_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21_1) : "f"(r1_53), "f"(_fma_119), "f"(r1_53));
        float t0_53 = _fma_20_1;
        float t1_53 = _fma_21_1;
        band_hi = band_hi | ((fminf(t0_53, t1_53) == 0.0f) ? 4 : 0);
        float2 s_52 = make_float2(max_noftz(t0_53, 0.0f), max_noftz(t1_53, 0.0f));
        float2 act_52;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_52) : "l"(*(const unsigned long long*)&_cvt_f32_79), "l"(*(const unsigned long long*)&s_52));
        float2 _f32x2_mul_a_120 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_52, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_52), s_52), _cvt_f32_81);
        float2 _mul_52;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_52) : "l"(*(const unsigned long long*)&_f32x2_mul_a_120), "l"(*(const unsigned long long*)&_cvt_f32_80));
        float2 _mul_53;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_53) : "l"(*(const unsigned long long*)&act_52), "l"(*(const unsigned long long*)&_cvt_f32_81));
        __nv_bfloat162 _bf16x2_52 = __float22bfloat162_rn(make_float2(_mul_52.x, _mul_52.y));
        __nv_bfloat162 _bf16x2_53 = __float22bfloat162_rn(make_float2(_mul_53.x, _mul_53.y));
        dgate_hi[2] = __as_u32(_bf16x2_52);
        dup_hi[2] = __as_u32(_bf16x2_53);
        float2 _cvt_f32_82 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[3]));
        float2 _cvt_f32_83 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[3]));
        float2 _cvt_f32_84 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[3]));
        float e0_55 = approx_exp2((-_cvt_f32_82.x) * 1.442695f);
        float e1_55 = approx_exp2((-_cvt_f32_82.y) * 1.442695f);
        float d0_55 = 1.0f + e0_55;
        float d1_55 = 1.0f + e1_55;
        float r0_55 = approx_rcp(d0_55);
        float r1_55 = approx_rcp(d1_55);
        float _fma_121;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_121) : "f"(-d0_55), "f"(r0_55), "f"(1.0f));
        float _fma_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(r0_55), "f"(_fma_121), "f"(r0_55));
        float _fma_122;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_122) : "f"(-d1_55), "f"(r1_55), "f"(1.0f));
        float _fma_23_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23_1) : "f"(r1_55), "f"(_fma_122), "f"(r1_55));
        float t0_55 = _fma_22;
        float t1_55 = _fma_23_1;
        band_hi = band_hi | ((fminf(t0_55, t1_55) == 0.0f) ? 8 : 0);
        float2 s_54 = make_float2(max_noftz(t0_55, 0.0f), max_noftz(t1_55, 0.0f));
        float2 act_54;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_54) : "l"(*(const unsigned long long*)&_cvt_f32_82), "l"(*(const unsigned long long*)&s_54));
        float2 _f32x2_mul_a_123 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_54, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_54), s_54), _cvt_f32_84);
        float2 _mul_54;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_54) : "l"(*(const unsigned long long*)&_f32x2_mul_a_123), "l"(*(const unsigned long long*)&_cvt_f32_83));
        float2 _mul_55;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_55) : "l"(*(const unsigned long long*)&act_54), "l"(*(const unsigned long long*)&_cvt_f32_84));
        __nv_bfloat162 _bf16x2_54 = __float22bfloat162_rn(make_float2(_mul_54.x, _mul_54.y));
        __nv_bfloat162 _bf16x2_55 = __float22bfloat162_rn(make_float2(_mul_55.x, _mul_55.y));
        dgate_hi[3] = __as_u32(_bf16x2_54);
        dup_hi[3] = __as_u32(_bf16x2_55);
        float2 _cvt_f32_85 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[4]));
        float2 _cvt_f32_86 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[4]));
        float2 _cvt_f32_87 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[4]));
        float e0_57 = approx_exp2((-_cvt_f32_85.x) * 1.442695f);
        float e1_57 = approx_exp2((-_cvt_f32_85.y) * 1.442695f);
        float d0_57 = 1.0f + e0_57;
        float d1_57 = 1.0f + e1_57;
        float r0_57 = approx_rcp(d0_57);
        float r1_57 = approx_rcp(d1_57);
        float _fma_124;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_124) : "f"(-d0_57), "f"(r0_57), "f"(1.0f));
        float _fma_24_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24_1) : "f"(r0_57), "f"(_fma_124), "f"(r0_57));
        float _fma_125;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(-d1_57), "f"(r1_57), "f"(1.0f));
        float _fma_25_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25_1) : "f"(r1_57), "f"(_fma_125), "f"(r1_57));
        float t0_57 = _fma_24_1;
        float t1_57 = _fma_25_1;
        band_hi = band_hi | ((fminf(t0_57, t1_57) == 0.0f) ? 16 : 0);
        float2 s_56 = make_float2(max_noftz(t0_57, 0.0f), max_noftz(t1_57, 0.0f));
        float2 act_56;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_56) : "l"(*(const unsigned long long*)&_cvt_f32_85), "l"(*(const unsigned long long*)&s_56));
        float2 _f32x2_mul_a_126 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_56, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_56), s_56), _cvt_f32_87);
        float2 _mul_56;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_56) : "l"(*(const unsigned long long*)&_f32x2_mul_a_126), "l"(*(const unsigned long long*)&_cvt_f32_86));
        float2 _mul_57;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_57) : "l"(*(const unsigned long long*)&act_56), "l"(*(const unsigned long long*)&_cvt_f32_87));
        __nv_bfloat162 _bf16x2_56 = __float22bfloat162_rn(make_float2(_mul_56.x, _mul_56.y));
        __nv_bfloat162 _bf16x2_57 = __float22bfloat162_rn(make_float2(_mul_57.x, _mul_57.y));
        dgate_hi[4] = __as_u32(_bf16x2_56);
        dup_hi[4] = __as_u32(_bf16x2_57);
        float2 _cvt_f32_88 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[5]));
        float2 _cvt_f32_89 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[5]));
        float2 _cvt_f32_90 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[5]));
        float e0_59 = approx_exp2((-_cvt_f32_88.x) * 1.442695f);
        float e1_59 = approx_exp2((-_cvt_f32_88.y) * 1.442695f);
        float d0_59 = 1.0f + e0_59;
        float d1_59 = 1.0f + e1_59;
        float r0_59 = approx_rcp(d0_59);
        float r1_59 = approx_rcp(d1_59);
        float _fma_127;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_127) : "f"(-d0_59), "f"(r0_59), "f"(1.0f));
        float _fma_26_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26_1) : "f"(r0_59), "f"(_fma_127), "f"(r0_59));
        float _fma_128;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_128) : "f"(-d1_59), "f"(r1_59), "f"(1.0f));
        float _fma_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(r1_59), "f"(_fma_128), "f"(r1_59));
        float t0_59 = _fma_26_1;
        float t1_59 = _fma_27;
        band_hi = band_hi | ((fminf(t0_59, t1_59) == 0.0f) ? 32 : 0);
        float2 s_58 = make_float2(max_noftz(t0_59, 0.0f), max_noftz(t1_59, 0.0f));
        float2 act_58;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_58) : "l"(*(const unsigned long long*)&_cvt_f32_88), "l"(*(const unsigned long long*)&s_58));
        float2 _f32x2_mul_a_129 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_58, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_58), s_58), _cvt_f32_90);
        float2 _mul_58;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_58) : "l"(*(const unsigned long long*)&_f32x2_mul_a_129), "l"(*(const unsigned long long*)&_cvt_f32_89));
        float2 _mul_59;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_59) : "l"(*(const unsigned long long*)&act_58), "l"(*(const unsigned long long*)&_cvt_f32_90));
        __nv_bfloat162 _bf16x2_58 = __float22bfloat162_rn(make_float2(_mul_58.x, _mul_58.y));
        __nv_bfloat162 _bf16x2_59 = __float22bfloat162_rn(make_float2(_mul_59.x, _mul_59.y));
        dgate_hi[5] = __as_u32(_bf16x2_58);
        dup_hi[5] = __as_u32(_bf16x2_59);
        float2 _cvt_f32_91 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[6]));
        float2 _cvt_f32_92 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[6]));
        float2 _cvt_f32_93 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[6]));
        float e0_61 = approx_exp2((-_cvt_f32_91.x) * 1.442695f);
        float e1_61 = approx_exp2((-_cvt_f32_91.y) * 1.442695f);
        float d0_61 = 1.0f + e0_61;
        float d1_61 = 1.0f + e1_61;
        float r0_61 = approx_rcp(d0_61);
        float r1_61 = approx_rcp(d1_61);
        float _fma_130;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_130) : "f"(-d0_61), "f"(r0_61), "f"(1.0f));
        float _fma_28_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28_1) : "f"(r0_61), "f"(_fma_130), "f"(r0_61));
        float _fma_131;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_131) : "f"(-d1_61), "f"(r1_61), "f"(1.0f));
        float _fma_29_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29_1) : "f"(r1_61), "f"(_fma_131), "f"(r1_61));
        float t0_61 = _fma_28_1;
        float t1_61 = _fma_29_1;
        band_hi = band_hi | ((fminf(t0_61, t1_61) == 0.0f) ? 64 : 0);
        float2 s_60 = make_float2(max_noftz(t0_61, 0.0f), max_noftz(t1_61, 0.0f));
        float2 act_60;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_60) : "l"(*(const unsigned long long*)&_cvt_f32_91), "l"(*(const unsigned long long*)&s_60));
        float2 _f32x2_mul_a_132 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_60, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_60), s_60), _cvt_f32_93);
        float2 _mul_60;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_60) : "l"(*(const unsigned long long*)&_f32x2_mul_a_132), "l"(*(const unsigned long long*)&_cvt_f32_92));
        float2 _mul_61;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_61) : "l"(*(const unsigned long long*)&act_60), "l"(*(const unsigned long long*)&_cvt_f32_93));
        __nv_bfloat162 _bf16x2_60 = __float22bfloat162_rn(make_float2(_mul_60.x, _mul_60.y));
        __nv_bfloat162 _bf16x2_61 = __float22bfloat162_rn(make_float2(_mul_61.x, _mul_61.y));
        dgate_hi[6] = __as_u32(_bf16x2_60);
        dup_hi[6] = __as_u32(_bf16x2_61);
        float2 _cvt_f32_94 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[7]));
        float2 _cvt_f32_95 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[7]));
        float2 _cvt_f32_96 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[7]));
        float e0_63 = approx_exp2((-_cvt_f32_94.x) * 1.442695f);
        float e1_63 = approx_exp2((-_cvt_f32_94.y) * 1.442695f);
        float d0_63 = 1.0f + e0_63;
        float d1_63 = 1.0f + e1_63;
        float r0_63 = approx_rcp(d0_63);
        float r1_63 = approx_rcp(d1_63);
        float _fma_133;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_133) : "f"(-d0_63), "f"(r0_63), "f"(1.0f));
        float _fma_30_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30_1) : "f"(r0_63), "f"(_fma_133), "f"(r0_63));
        float _fma_134;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_134) : "f"(-d1_63), "f"(r1_63), "f"(1.0f));
        float _fma_31_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31_1) : "f"(r1_63), "f"(_fma_134), "f"(r1_63));
        float t0_63 = _fma_30_1;
        float t1_63 = _fma_31_1;
        band_hi = band_hi | ((fminf(t0_63, t1_63) == 0.0f) ? 128 : 0);
        float2 s_62 = make_float2(max_noftz(t0_63, 0.0f), max_noftz(t1_63, 0.0f));
        float2 act_62;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_62) : "l"(*(const unsigned long long*)&_cvt_f32_94), "l"(*(const unsigned long long*)&s_62));
        float2 _f32x2_mul_a_135 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_62, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_62), s_62), _cvt_f32_96);
        float2 _mul_62;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_62) : "l"(*(const unsigned long long*)&_f32x2_mul_a_135), "l"(*(const unsigned long long*)&_cvt_f32_95));
        float2 _mul_63;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_63) : "l"(*(const unsigned long long*)&act_62), "l"(*(const unsigned long long*)&_cvt_f32_96));
        __nv_bfloat162 _bf16x2_62 = __float22bfloat162_rn(make_float2(_mul_62.x, _mul_62.y));
        __nv_bfloat162 _bf16x2_63 = __float22bfloat162_rn(make_float2(_mul_63.x, _mul_63.y));
        dgate_hi[7] = __as_u32(_bf16x2_62);
        dup_hi[7] = __as_u32(_bf16x2_63);
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
                float2 _cvt_f32_97 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[0]));
                float2 _cvt_f32_98 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[0]));
                float2 _cvt_f32_99 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_8[0]));
                float e0_65 = approx_exp2((-_cvt_f32_97.x) * 1.442695f);
                float e1_65 = approx_exp2((-_cvt_f32_97.y) * 1.442695f);
                float d0_65 = 1.0f + e0_65;
                float d1_65 = 1.0f + e1_65;
                float r0_65 = approx_rcp(d0_65);
                float r1_65 = approx_rcp(d1_65);
                float _fma_137;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_137) : "f"(-d0_65), "f"(r0_65), "f"(1.0f));
                float _fma_32;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(r0_65), "f"(_fma_137), "f"(r0_65));
                float _fma_138;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_138) : "f"(-d1_65), "f"(r1_65), "f"(1.0f));
                float _fma_33_1;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_33_1) : "f"(r1_65), "f"(_fma_138), "f"(r1_65));
                float t0_65 = _fma_32;
                float t1_65 = _fma_33_1;
                float d4_65a;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_65a) : "f"(d0_65), "f"(0.25f));
                float r4_65a = approx_rcp(d4_65a);
                float _fma_140;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_140) : "f"(-d4_65a), "f"(r4_65a), "f"(1.0f));
                float _fma_139;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_139) : "f"(r4_65a), "f"(_fma_140), "f"(r4_65a));
                float big_65a;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_65a) : "f"(mul_f32_rn_noftz(_fma_139, 1.2676506e+30f)), "f"(1.4073749e+14f));
                float _add_141;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_141) : "f"(big_65a), "f"(8.388608e+06f));
                float whole_65a;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_65a) : "f"(_add_141), "f"(8.388608e+06f));
                float above_65a;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_65a) : "f"(-d4_65a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_65a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                float _sub_143;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_143) : "f"(big_65a), "f"(whole_65a));
                float _abs_142;
                asm("abs.f32 %0, %1;" : "=f"(_abs_142) : "f"(_sub_143));
                float _add_144;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_144) : "f"(big_65a), "f"(0.5f));
                float _sub_145;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_145) : "f"(big_65a), "f"(0.5f));
                float rounded_65a = ((_abs_142 == 0.5f) ? ((above_65a > 0.0f) ? _add_144 : _sub_145) : whole_65a);
                float u0_65 = ((d0_65 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_65a, 7.888609e-31f), 1.7763568e-15f) : t0_65);
                float d4_65b;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_65b) : "f"(d1_65), "f"(0.25f));
                float r4_65b = approx_rcp(d4_65b);
                float _fma_147;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_147) : "f"(-d4_65b), "f"(r4_65b), "f"(1.0f));
                float _fma_146;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_146) : "f"(r4_65b), "f"(_fma_147), "f"(r4_65b));
                float big_65b;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_65b) : "f"(mul_f32_rn_noftz(_fma_146, 1.2676506e+30f)), "f"(1.4073749e+14f));
                float _add_148;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_148) : "f"(big_65b), "f"(8.388608e+06f));
                float whole_65b;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_65b) : "f"(_add_148), "f"(8.388608e+06f));
                float above_65b;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_65b) : "f"(-d4_65b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_65b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                float _sub_150;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_150) : "f"(big_65b), "f"(whole_65b));
                float _abs_149;
                asm("abs.f32 %0, %1;" : "=f"(_abs_149) : "f"(_sub_150));
                float _add_151;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_151) : "f"(big_65b), "f"(0.5f));
                float _sub_152;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_152) : "f"(big_65b), "f"(0.5f));
                float rounded_65b = ((_abs_149 == 0.5f) ? ((above_65b > 0.0f) ? _add_151 : _sub_152) : whole_65b);
                float u1_65 = ((d1_65 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_65b, 7.888609e-31f), 1.7763568e-15f) : t1_65);
                float2 s_64 = make_float2(max_noftz(u0_65, 0.0f), max_noftz(u1_65, 0.0f));
                float2 act_64;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_64) : "l"(*(const unsigned long long*)&_cvt_f32_97), "l"(*(const unsigned long long*)&s_64));
                float2 _f32x2_mul_a_153 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_64, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_64), s_64), _cvt_f32_99);
                float2 _mul_64;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_64) : "l"(*(const unsigned long long*)&_f32x2_mul_a_153), "l"(*(const unsigned long long*)&_cvt_f32_98));
                float2 _mul_65;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_65) : "l"(*(const unsigned long long*)&act_64), "l"(*(const unsigned long long*)&_cvt_f32_99));
                __nv_bfloat162 _bf16x2_64 = __float22bfloat162_rn(make_float2(_mul_64.x, _mul_64.y));
                __nv_bfloat162 _bf16x2_65 = __float22bfloat162_rn(make_float2(_mul_65.x, _mul_65.y));
                unsigned int dgate_new = __as_u32(_bf16x2_64);
                unsigned int dup_new = __as_u32(_bf16x2_65);
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
                float2 _cvt_f32_100 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_9[0]));
                float2 _cvt_f32_101 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_10[0]));
                float2 _cvt_f32_102 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_11[0]));
                float e0_67 = approx_exp2((-_cvt_f32_100.x) * 1.442695f);
                float e1_67 = approx_exp2((-_cvt_f32_100.y) * 1.442695f);
                float d0_67 = 1.0f + e0_67;
                float d1_67 = 1.0f + e1_67;
                float r0_67 = approx_rcp(d0_67);
                float r1_67 = approx_rcp(d1_67);
                float _fma_155;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_155) : "f"(-d0_67), "f"(r0_67), "f"(1.0f));
                float _fma_34_1;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34_1) : "f"(r0_67), "f"(_fma_155), "f"(r0_67));
                float _fma_156;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_156) : "f"(-d1_67), "f"(r1_67), "f"(1.0f));
                float _fma_35_1;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35_1) : "f"(r1_67), "f"(_fma_156), "f"(r1_67));
                float t0_67 = _fma_34_1;
                float t1_67 = _fma_35_1;
                float d4_67a;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_67a) : "f"(d0_67), "f"(0.25f));
                float r4_67a = approx_rcp(d4_67a);
                float _fma_158;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_158) : "f"(-d4_67a), "f"(r4_67a), "f"(1.0f));
                float _fma_157;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_157) : "f"(r4_67a), "f"(_fma_158), "f"(r4_67a));
                float big_67a;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_67a) : "f"(mul_f32_rn_noftz(_fma_157, 1.2676506e+30f)), "f"(1.4073749e+14f));
                float _add_159;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_159) : "f"(big_67a), "f"(8.388608e+06f));
                float whole_67a;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_67a) : "f"(_add_159), "f"(8.388608e+06f));
                float above_67a;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_67a) : "f"(-d4_67a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_67a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                float _sub_161;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_161) : "f"(big_67a), "f"(whole_67a));
                float _abs_160;
                asm("abs.f32 %0, %1;" : "=f"(_abs_160) : "f"(_sub_161));
                float _add_162;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_162) : "f"(big_67a), "f"(0.5f));
                float _sub_163;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_163) : "f"(big_67a), "f"(0.5f));
                float rounded_67a = ((_abs_160 == 0.5f) ? ((above_67a > 0.0f) ? _add_162 : _sub_163) : whole_67a);
                float u0_67 = ((d0_67 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_67a, 7.888609e-31f), 1.7763568e-15f) : t0_67);
                float d4_67b;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_67b) : "f"(d1_67), "f"(0.25f));
                float r4_67b = approx_rcp(d4_67b);
                float _fma_165;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_165) : "f"(-d4_67b), "f"(r4_67b), "f"(1.0f));
                float _fma_164;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_164) : "f"(r4_67b), "f"(_fma_165), "f"(r4_67b));
                float big_67b;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_67b) : "f"(mul_f32_rn_noftz(_fma_164, 1.2676506e+30f)), "f"(1.4073749e+14f));
                float _add_166;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_166) : "f"(big_67b), "f"(8.388608e+06f));
                float whole_67b;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_67b) : "f"(_add_166), "f"(8.388608e+06f));
                float above_67b;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_67b) : "f"(-d4_67b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_67b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                float _sub_168;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_168) : "f"(big_67b), "f"(whole_67b));
                float _abs_167;
                asm("abs.f32 %0, %1;" : "=f"(_abs_167) : "f"(_sub_168));
                float _add_169;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_169) : "f"(big_67b), "f"(0.5f));
                float _sub_170;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_170) : "f"(big_67b), "f"(0.5f));
                float rounded_67b = ((_abs_167 == 0.5f) ? ((above_67b > 0.0f) ? _add_169 : _sub_170) : whole_67b);
                float u1_67 = ((d1_67 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_67b, 7.888609e-31f), 1.7763568e-15f) : t1_67);
                float2 s_66 = make_float2(max_noftz(u0_67, 0.0f), max_noftz(u1_67, 0.0f));
                float2 act_66;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_66) : "l"(*(const unsigned long long*)&_cvt_f32_100), "l"(*(const unsigned long long*)&s_66));
                float2 _f32x2_mul_a_171 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_66, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_66), s_66), _cvt_f32_102);
                float2 _mul_66;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_66) : "l"(*(const unsigned long long*)&_f32x2_mul_a_171), "l"(*(const unsigned long long*)&_cvt_f32_101));
                float2 _mul_67;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_67) : "l"(*(const unsigned long long*)&act_66), "l"(*(const unsigned long long*)&_cvt_f32_102));
                __nv_bfloat162 _bf16x2_66 = __float22bfloat162_rn(make_float2(_mul_66.x, _mul_66.y));
                __nv_bfloat162 _bf16x2_67 = __float22bfloat162_rn(make_float2(_mul_67.x, _mul_67.y));
                unsigned int dgate_new_hi = __as_u32(_bf16x2_66);
                unsigned int dup_new_hi = __as_u32(_bf16x2_67);
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_189;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_189) : "r"(_bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_189) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_188;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_188) : "h"(_e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_189));
    words[0] = _mov_b32_pack_u16x2_188;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_191;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_191) : "r"(_bf16x2_mul_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_191) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_190;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_190) : "h"(_e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_191));
    words[1] = _mov_b32_pack_u16x2_190;
    uint32_t _bf16x2_mul_4;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_4) : "r"(dgate[4]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_5) : "r"(dgate[5]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_193;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_193) : "r"(_bf16x2_mul_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_193) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_192;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_192) : "h"(_e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_193));
    words[2] = _mov_b32_pack_u16x2_192;
    uint32_t _bf16x2_mul_6;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_6) : "r"(dgate[6]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_7) : "r"(dgate[7]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_195;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_195) : "r"(_bf16x2_mul_7));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_7);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_195) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_194;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_194) : "h"(_e4m3x2_bf16x2_3), "h"(_cvt_satfinite_e4m3x2_bf16x2_195));
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
    unsigned int amax_bits_5 = (folded_4 & 65535) << 16;
    int scale_i32_6 = (int)(amax_bits_5 + 2031616 >> 23) - 8;
    if (scale_i32_6 < 0) {
        scale_i32_6 = 0;
    }
    if ((amax_bits_5 & 2139095040) == 2139095040) {
        scale_i32_6 = 255;
    }
    unsigned int scale_7 = (unsigned int)scale_i32_6;
    unsigned int inv_8 = 254 - scale_7 << 7;
    if (scale_7 == 255) {
        inv_8 = 32704;
    }
    unsigned int inv2_9 = inv_8 | inv_8 << 16;
    int scale_index_10 = ((grow >> 7) * scale_cols + (scale_col + (K >> 5) >> 2)) * 512 + (grow & 31) * 16 + (grow >> 5 & 3) * 4 + (scale_col + (K >> 5) & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + scale_index_10) + (0)) = (unsigned char)(scale_7);
    }
    unsigned int words_11[4];
    uint32_t _bf16x2_mul_8;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_8) : "r"(dup[0]), "r"(inv2_9));
    uint16_t _e4m3x2_bf16x2_4;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_4) : "r"(_bf16x2_mul_8));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_8);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_4) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_9;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_9) : "r"(dup[1]), "r"(inv2_9));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_213;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_213) : "r"(_bf16x2_mul_9));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_9);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_213) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_212;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_212) : "h"(_e4m3x2_bf16x2_4), "h"(_cvt_satfinite_e4m3x2_bf16x2_213));
    words_11[0] = _mov_b32_pack_u16x2_212;
    uint32_t _bf16x2_mul_10;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_10) : "r"(dup[2]), "r"(inv2_9));
    uint16_t _e4m3x2_bf16x2_5;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_5) : "r"(_bf16x2_mul_10));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_10);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_5) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_11;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_11) : "r"(dup[3]), "r"(inv2_9));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_215;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_215) : "r"(_bf16x2_mul_11));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_11);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_215) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_214;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_214) : "h"(_e4m3x2_bf16x2_5), "h"(_cvt_satfinite_e4m3x2_bf16x2_215));
    words_11[1] = _mov_b32_pack_u16x2_214;
    uint32_t _bf16x2_mul_12;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_12) : "r"(dup[4]), "r"(inv2_9));
    uint16_t _e4m3x2_bf16x2_6;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_6) : "r"(_bf16x2_mul_12));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_12);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_6) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_13;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_13) : "r"(dup[5]), "r"(inv2_9));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_217;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_217) : "r"(_bf16x2_mul_13));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_13);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_217) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_216;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_216) : "h"(_e4m3x2_bf16x2_6), "h"(_cvt_satfinite_e4m3x2_bf16x2_217));
    words_11[2] = _mov_b32_pack_u16x2_216;
    uint32_t _bf16x2_mul_14;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_14) : "r"(dup[6]), "r"(inv2_9));
    uint16_t _e4m3x2_bf16x2_7;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_7) : "r"(_bf16x2_mul_14));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_14);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_7) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_15;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_15) : "r"(dup[7]), "r"(inv2_9));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_219;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_219) : "r"(_bf16x2_mul_15));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_15);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_219) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_218;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_218) : "h"(_e4m3x2_bf16x2_7), "h"(_cvt_satfinite_e4m3x2_bf16x2_219));
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
    unsigned int amax_bits_17 = (folded_16 & 65535) << 16;
    int scale_i32_18 = (int)(amax_bits_17 + 2031616 >> 23) - 8;
    if (scale_i32_18 < 0) {
        scale_i32_18 = 0;
    }
    if ((amax_bits_17 & 2139095040) == 2139095040) {
        scale_i32_18 = 255;
    }
    unsigned int scale_19 = (unsigned int)scale_i32_18;
    unsigned int inv_20 = 254 - scale_19 << 7;
    if (scale_19 == 255) {
        inv_20 = 32704;
    }
    unsigned int inv2_21 = inv_20 | inv_20 << 16;
    int scale_index_22 = ((grow_hi >> 7) * scale_cols + (scale_col >> 2)) * 512 + (grow_hi & 31) * 16 + (grow_hi >> 5 & 3) * 4 + (scale_col & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + scale_index_22) + (0)) = (unsigned char)(scale_19);
    }
    unsigned int words_23[4];
    uint32_t _bf16x2_mul_16;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_16) : "r"(dgate_hi[0]), "r"(inv2_21));
    uint16_t _e4m3x2_bf16x2_8;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_8) : "r"(_bf16x2_mul_16));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_16);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_8) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_17;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_17) : "r"(dgate_hi[1]), "r"(inv2_21));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_237;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_237) : "r"(_bf16x2_mul_17));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_17);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_237) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_236;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_236) : "h"(_e4m3x2_bf16x2_8), "h"(_cvt_satfinite_e4m3x2_bf16x2_237));
    words_23[0] = _mov_b32_pack_u16x2_236;
    uint32_t _bf16x2_mul_18;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_18) : "r"(dgate_hi[2]), "r"(inv2_21));
    uint16_t _e4m3x2_bf16x2_9;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_9) : "r"(_bf16x2_mul_18));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_18);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_9) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_19;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_19) : "r"(dgate_hi[3]), "r"(inv2_21));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_239;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_239) : "r"(_bf16x2_mul_19));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_19);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_239) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_238;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_238) : "h"(_e4m3x2_bf16x2_9), "h"(_cvt_satfinite_e4m3x2_bf16x2_239));
    words_23[1] = _mov_b32_pack_u16x2_238;
    uint32_t _bf16x2_mul_20;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_20) : "r"(dgate_hi[4]), "r"(inv2_21));
    uint16_t _e4m3x2_bf16x2_10;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_10) : "r"(_bf16x2_mul_20));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_20);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_10) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_21;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_21) : "r"(dgate_hi[5]), "r"(inv2_21));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_241;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_241) : "r"(_bf16x2_mul_21));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_21);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_241) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_240;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_240) : "h"(_e4m3x2_bf16x2_10), "h"(_cvt_satfinite_e4m3x2_bf16x2_241));
    words_23[2] = _mov_b32_pack_u16x2_240;
    uint32_t _bf16x2_mul_22;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_22) : "r"(dgate_hi[6]), "r"(inv2_21));
    uint16_t _e4m3x2_bf16x2_11;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_11) : "r"(_bf16x2_mul_22));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_22);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_11) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_23;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_23) : "r"(dgate_hi[7]), "r"(inv2_21));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_243;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_243) : "r"(_bf16x2_mul_23));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_23);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_243) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_242;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_242) : "h"(_e4m3x2_bf16x2_11), "h"(_cvt_satfinite_e4m3x2_bf16x2_243));
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
    unsigned int amax_bits_29 = (folded_28 & 65535) << 16;
    int scale_i32_30 = (int)(amax_bits_29 + 2031616 >> 23) - 8;
    if (scale_i32_30 < 0) {
        scale_i32_30 = 0;
    }
    if ((amax_bits_29 & 2139095040) == 2139095040) {
        scale_i32_30 = 255;
    }
    unsigned int scale_31 = (unsigned int)scale_i32_30;
    unsigned int inv_32 = 254 - scale_31 << 7;
    if (scale_31 == 255) {
        inv_32 = 32704;
    }
    unsigned int inv2_33 = inv_32 | inv_32 << 16;
    int scale_index_34 = ((grow_hi >> 7) * scale_cols + (scale_col + (K >> 5) >> 2)) * 512 + (grow_hi & 31) * 16 + (grow_hi >> 5 & 3) * 4 + (scale_col + (K >> 5) & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + scale_index_34) + (0)) = (unsigned char)(scale_31);
    }
    unsigned int words_35[4];
    uint32_t _bf16x2_mul_24;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_24) : "r"(dup_hi[0]), "r"(inv2_33));
    uint16_t _e4m3x2_bf16x2_12;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_12) : "r"(_bf16x2_mul_24));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_24);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_12) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_25;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_25) : "r"(dup_hi[1]), "r"(inv2_33));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_261;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_261) : "r"(_bf16x2_mul_25));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_25);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_261) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_260;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_260) : "h"(_e4m3x2_bf16x2_12), "h"(_cvt_satfinite_e4m3x2_bf16x2_261));
    words_35[0] = _mov_b32_pack_u16x2_260;
    uint32_t _bf16x2_mul_26;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_26) : "r"(dup_hi[2]), "r"(inv2_33));
    uint16_t _e4m3x2_bf16x2_13;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_13) : "r"(_bf16x2_mul_26));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_26);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_13) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_27;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_27) : "r"(dup_hi[3]), "r"(inv2_33));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_263;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_263) : "r"(_bf16x2_mul_27));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_27);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_263) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_262;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_262) : "h"(_e4m3x2_bf16x2_13), "h"(_cvt_satfinite_e4m3x2_bf16x2_263));
    words_35[1] = _mov_b32_pack_u16x2_262;
    uint32_t _bf16x2_mul_28;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_28) : "r"(dup_hi[4]), "r"(inv2_33));
    uint16_t _e4m3x2_bf16x2_14;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_14) : "r"(_bf16x2_mul_28));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_28);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_14) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_29;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_29) : "r"(dup_hi[5]), "r"(inv2_33));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_265;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_265) : "r"(_bf16x2_mul_29));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_29);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_265) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_264;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_264) : "h"(_e4m3x2_bf16x2_14), "h"(_cvt_satfinite_e4m3x2_bf16x2_265));
    words_35[2] = _mov_b32_pack_u16x2_264;
    uint32_t _bf16x2_mul_30;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_30) : "r"(dup_hi[6]), "r"(inv2_33));
    uint16_t _e4m3x2_bf16x2_15;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_15) : "r"(_bf16x2_mul_30));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_30);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_15) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_31;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_31) : "r"(dup_hi[7]), "r"(inv2_33));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_267;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_267) : "r"(_bf16x2_mul_31));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_31);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_267) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_266;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_266) : "h"(_e4m3x2_bf16x2_15), "h"(_cvt_satfinite_e4m3x2_bf16x2_267));
    words_35[3] = _mov_b32_pack_u16x2_266;
    reinterpret_cast<int4*>(row_output + (out_index_hi + K))[0] = reinterpret_cast<int4*>(words_35 + 0)[0];
}

} // extern "C"
