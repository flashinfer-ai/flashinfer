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

__global__ __launch_bounds__(128) void
kernel_cake_gated_act_mxfp8_a097ced99ffe2f27b973(__nv_bfloat16* __restrict__ gated_input, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int quarter = tid & 15;
    int row = tid >> 4;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int grow = by * 8 + row;
    int col = bx * 32 + quarter * 2;
    int gate_index = grow * (2 * K) + col;
    unsigned int words[1];
    {
        uint32_t _scalar_bits_0;
        asm volatile("ld.global.L2::cache_hint.b32 %0, [%1], %2;"
            : "=r"(_scalar_bits_0) : "l"((const void*)(gated_input + gate_index + (0))), "l"(0x12F0000000000000ULL) : "memory");
        words[0] = (unsigned int)_scalar_bits_0;
    }
    unsigned int words_0[1];
    {
        uint32_t _scalar_bits_1;
        asm volatile("ld.global.L2::cache_hint.b32 %0, [%1], %2;"
            : "=r"(_scalar_bits_1) : "l"((const void*)(gated_input + (gate_index + K) + (0))), "l"(0x12F0000000000000ULL) : "memory");
        words_0[0] = (unsigned int)_scalar_bits_1;
    }
    unsigned int packed[1];
    __nv_bfloat162 lowest = __as_bf16x2(words[0]);
    float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(lowest);
    float2 cvt_f32 = cake_bf16x2_to_f32x2(__as_bf16x2(words[0]));
    float2 cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[0]));
    float e0 = approx_exp2(-cvt_f32.x * 1.442695f);
    float e1 = approx_exp2(-cvt_f32.y * 1.442695f);
    float d0 = 1.0f + e0;
    float d1 = 1.0f + e1;
    float dn;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn) : "f"(e0), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
    float r = approx_rcp(dn);
    float _fma_2;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(-dn), "f"(r), "f"(1.0f));
    float q;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q) : "f"(r), "f"(_fma_2), "f"(r));
    float s0;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0) : "f"(q), "f"(1.1754944e-38f));
    float err;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err) : "f"(s0), "f"(-8.507059e+37f), "f"(q));
    float y2;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2) : "f"(-dn), "f"(q), "f"(1.0f));
    float nudge = __uint_as_float(__as_u32(y2) & 2147483648u | 729808896);
    float _add_3;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(err), "f"(nudge));
    float u0;
    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0) : "f"(_add_3), "f"(1.1754944e-38f), "f"(s0));
    float dn_0;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_0) : "f"(e1), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
    float r_0 = approx_rcp(dn_0);
    float _fma_4;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(-dn_0), "f"(r_0), "f"(1.0f));
    float q_0;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_0) : "f"(r_0), "f"(_fma_4), "f"(r_0));
    float s0_0;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_0) : "f"(q_0), "f"(1.1754944e-38f));
    float err_0;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_0) : "f"(s0_0), "f"(-8.507059e+37f), "f"(q_0));
    float y2_0;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_0) : "f"(-dn_0), "f"(q_0), "f"(1.0f));
    float nudge_0 = __uint_as_float(__as_u32(y2_0) & 2147483648u | 729808896);
    float _add_5;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(err_0), "f"(nudge_0));
    float u1;
    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1) : "f"(_add_5), "f"(1.1754944e-38f), "f"(s0_0));
    float2 _f32x2_mul_a_6 = mul_f32x2_noftz(cvt_f32, make_float2(u0, u1));
    float2 mul;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul) : "l"(*(const unsigned long long*)&_f32x2_mul_a_6), "l"(*(const unsigned long long*)&cvt_f32_1));
    packed[0] = __as_u32(__float22bfloat162_rn(make_float2(mul.x, mul.y)));
    __nv_bfloat162 amax = __as_bf16x2(0);
    __nv_bfloat162 _hxa_8 = amax;
    __nv_bfloat162 _hxb_8 = __as_bf16x2(packed[0]);
    unsigned int _hxr_8;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_8) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_8)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_8)));
    __nv_bfloat162 _max_7 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_8);
    amax = _max_7;
    unsigned int mag = __as_u32(amax) & 2147450879;
    unsigned int m = ((mag & 65535) > (mag >> 16) ? (mag & 65535) : (mag >> 16));
    int wblk = (tid & 31) >> 4;
    unsigned int folded = 0;
    unsigned int rb_0;
    asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(rb_0) : "r"(((wblk == 0) ? m : 0)));
    folded = ((wblk == 0) ? rb_0 : folded);
    unsigned int rb_1;
    asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(rb_1) : "r"(((wblk == 1) ? m : 0)));
    folded = ((wblk == 1) ? rb_1 : folded);
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
    uint32_t scaled_pair;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(scaled_pair) : "r"(packed[0]), "r"(inv2));
    uint16_t half_word;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(half_word) : "r"(scaled_pair));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(scaled_pair);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(half_word) : "f"(_hi), "f"(_lo));
    }
    #endif
    *reinterpret_cast<unsigned short*>(row_output + (grow * K + col)) = (unsigned short)(half_word);
}

} // extern "C"
