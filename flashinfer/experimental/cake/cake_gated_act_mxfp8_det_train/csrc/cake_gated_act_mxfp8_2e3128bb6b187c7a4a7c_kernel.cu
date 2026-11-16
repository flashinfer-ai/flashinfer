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

__global__ __launch_bounds__(256) void
kernel_cake_gated_act_mxfp8_2e3128bb6b187c7a4a7c(__nv_bfloat16* __restrict__ gated_input, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    uint8_t* tile = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int tile_addr = smem + 0;

    // === Task calls (declaration order) ===
    int half = tid & 1;
    int blk = tid >> 1 & 3;
    int row = tid >> 3;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int grow = by * 64 + row;
    int col = bx * 128 + blk * 32 + half * 16;
    int gate_index = grow * (2 * K) + col;
    unsigned int words[8];
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
        words[0 + 0] = (uint32_t)_iv_0_0;
        words[0 + 1] = (uint32_t)_iv_0_1;
        words[0 + 2] = (uint32_t)_iv_0_2;
        words[0 + 3] = (uint32_t)_iv_0_3;
        words[0 + 4] = (uint32_t)_iv_0_4;
        words[0 + 5] = (uint32_t)_iv_0_5;
        words[0 + 6] = (uint32_t)_iv_0_6;
        words[0 + 7] = (uint32_t)_iv_0_7;
    }
    unsigned int words_0[8];
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
        words_0[0 + 0] = (uint32_t)_iv_1_0;
        words_0[0 + 1] = (uint32_t)_iv_1_1;
        words_0[0 + 2] = (uint32_t)_iv_1_2;
        words_0[0 + 3] = (uint32_t)_iv_1_3;
        words_0[0 + 4] = (uint32_t)_iv_1_4;
        words_0[0 + 5] = (uint32_t)_iv_1_5;
        words_0[0 + 6] = (uint32_t)_iv_1_6;
        words_0[0 + 7] = (uint32_t)_iv_1_7;
    }
    int grow_hi = grow + 32;
    int gate_index_hi = grow_hi * (2 * K) + col;
    unsigned int words_1[8];
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
            : "=r"(_iv_2_0), "=r"(_iv_2_1), "=r"(_iv_2_2), "=r"(_iv_2_3), "=r"(_iv_2_4), "=r"(_iv_2_5), "=r"(_iv_2_6), "=r"(_iv_2_7) : "l"((const void*)(gated_input + gate_index_hi + (0))), "l"(0x12F0000000000000ULL) : "memory");
        words_1[0 + 0] = (uint32_t)_iv_2_0;
        words_1[0 + 1] = (uint32_t)_iv_2_1;
        words_1[0 + 2] = (uint32_t)_iv_2_2;
        words_1[0 + 3] = (uint32_t)_iv_2_3;
        words_1[0 + 4] = (uint32_t)_iv_2_4;
        words_1[0 + 5] = (uint32_t)_iv_2_5;
        words_1[0 + 6] = (uint32_t)_iv_2_6;
        words_1[0 + 7] = (uint32_t)_iv_2_7;
    }
    unsigned int words_2[8];
    {
        uint32_t _iv_3_0;
        uint32_t _iv_3_1;
        uint32_t _iv_3_2;
        uint32_t _iv_3_3;
        uint32_t _iv_3_4;
        uint32_t _iv_3_5;
        uint32_t _iv_3_6;
        uint32_t _iv_3_7;
        asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
            : "=r"(_iv_3_0), "=r"(_iv_3_1), "=r"(_iv_3_2), "=r"(_iv_3_3), "=r"(_iv_3_4), "=r"(_iv_3_5), "=r"(_iv_3_6), "=r"(_iv_3_7) : "l"((const void*)(gated_input + (gate_index_hi + K) + (0))), "l"(0x12F0000000000000ULL) : "memory");
        words_2[0 + 0] = (uint32_t)_iv_3_0;
        words_2[0 + 1] = (uint32_t)_iv_3_1;
        words_2[0 + 2] = (uint32_t)_iv_3_2;
        words_2[0 + 3] = (uint32_t)_iv_3_3;
        words_2[0 + 4] = (uint32_t)_iv_3_4;
        words_2[0 + 5] = (uint32_t)_iv_3_5;
        words_2[0 + 6] = (uint32_t)_iv_3_6;
        words_2[0 + 7] = (uint32_t)_iv_3_7;
    }
    unsigned int packed[8];
    unsigned int packed_hi[8];
    __nv_bfloat162 lowest = __as_bf16x2(words[0]);
    lowest = __hmin2_nan(lowest, __as_bf16x2(words[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(words[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(words[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(words[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(words[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(words[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(words[7]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(words_1[0]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(words_1[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(words_1[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(words_1[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(words_1[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(words_1[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(words_1[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(words_1[7]));
    float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(lowest);
    float _min_5;
    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_5) : "f"(_cvt_f32_0.x), "f"(_cvt_f32_0.y));
    int _warp_vote_4 = __all_sync(0xFFFFFFFF, _min_5 > -87.0f);
    if (_warp_vote_4 != 0) {
        float2 cvt_f32 = cake_bf16x2_to_f32x2(__as_bf16x2(words[0]));
        float2 cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[0]));
        float2 _f32x2_mul_b_6 = make_float2(-1.442695f, -1.442695f);
        float2 t;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t) : "l"(*(const unsigned long long*)&cvt_f32), "l"(*(const unsigned long long*)&_f32x2_mul_b_6));
        float e0 = approx_exp2(t.x);
        float e1 = approx_exp2(t.y);
        float2 nd = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0, e1));
        float d0 = -nd.x;
        float d1 = -nd.y;
        float r0 = approx_rcp(d0);
        float r1 = approx_rcp(d1);
        float2 r = make_float2(r0, r1);
        float2 s = fma_f32x2_rn_noftz(r, fma_f32x2_rn_noftz(nd, r, make_float2(1.0f, 1.0f)), r);
        float2 _f32x2_mul_a_7 = mul_f32x2_noftz(cvt_f32, s);
        float2 mul;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul) : "l"(*(const unsigned long long*)&_f32x2_mul_a_7), "l"(*(const unsigned long long*)&cvt_f32_1));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(mul.x, mul.y)));
        float2 cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(words[1]));
        float2 cvt_f32_1_0 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[1]));
        float2 _f32x2_mul_b_8 = make_float2(-1.442695f, -1.442695f);
        float2 t_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_0) : "l"(*(const unsigned long long*)&cvt_f32_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_8));
        float e0_0 = approx_exp2(t_0.x);
        float e1_0 = approx_exp2(t_0.y);
        float2 nd_0 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_0, e1_0));
        float d0_0 = -nd_0.x;
        float d1_0 = -nd_0.y;
        float r0_0 = approx_rcp(d0_0);
        float r1_0 = approx_rcp(d1_0);
        float2 r_0 = make_float2(r0_0, r1_0);
        float2 s_0 = fma_f32x2_rn_noftz(r_0, fma_f32x2_rn_noftz(nd_0, r_0, make_float2(1.0f, 1.0f)), r_0);
        float2 _f32x2_mul_a_9 = mul_f32x2_noftz(cvt_f32_0, s_0);
        float2 mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_9), "l"(*(const unsigned long long*)&cvt_f32_1_0));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_0.x, mul_0.y)));
        float2 cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(words[2]));
        float2 cvt_f32_1_1 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[2]));
        float2 _f32x2_mul_b_10 = make_float2(-1.442695f, -1.442695f);
        float2 t_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_1) : "l"(*(const unsigned long long*)&cvt_f32_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_10));
        float e0_1 = approx_exp2(t_1.x);
        float e1_1 = approx_exp2(t_1.y);
        float2 nd_1 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_1, e1_1));
        float d0_1 = -nd_1.x;
        float d1_1 = -nd_1.y;
        float r0_1 = approx_rcp(d0_1);
        float r1_1 = approx_rcp(d1_1);
        float2 r_1 = make_float2(r0_1, r1_1);
        float2 s_1 = fma_f32x2_rn_noftz(r_1, fma_f32x2_rn_noftz(nd_1, r_1, make_float2(1.0f, 1.0f)), r_1);
        float2 _f32x2_mul_a_11 = mul_f32x2_noftz(cvt_f32_2, s_1);
        float2 mul_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_11), "l"(*(const unsigned long long*)&cvt_f32_1_1));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_1.x, mul_1.y)));
        float2 cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(words[3]));
        float2 cvt_f32_1_2 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[3]));
        float2 _f32x2_mul_b_12 = make_float2(-1.442695f, -1.442695f);
        float2 t_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_2) : "l"(*(const unsigned long long*)&cvt_f32_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_12));
        float e0_2 = approx_exp2(t_2.x);
        float e1_2 = approx_exp2(t_2.y);
        float2 nd_2 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_2, e1_2));
        float d0_2 = -nd_2.x;
        float d1_2 = -nd_2.y;
        float r0_2 = approx_rcp(d0_2);
        float r1_2 = approx_rcp(d1_2);
        float2 r_2 = make_float2(r0_2, r1_2);
        float2 s_2 = fma_f32x2_rn_noftz(r_2, fma_f32x2_rn_noftz(nd_2, r_2, make_float2(1.0f, 1.0f)), r_2);
        float2 _f32x2_mul_a_13 = mul_f32x2_noftz(cvt_f32_3, s_2);
        float2 mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_13), "l"(*(const unsigned long long*)&cvt_f32_1_2));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_2.x, mul_2.y)));
        float2 cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(words[4]));
        float2 cvt_f32_1_3 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[4]));
        float2 _f32x2_mul_b_14 = make_float2(-1.442695f, -1.442695f);
        float2 t_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_3) : "l"(*(const unsigned long long*)&cvt_f32_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_14));
        float e0_3 = approx_exp2(t_3.x);
        float e1_3 = approx_exp2(t_3.y);
        float2 nd_3 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_3, e1_3));
        float d0_3 = -nd_3.x;
        float d1_3 = -nd_3.y;
        float r0_3 = approx_rcp(d0_3);
        float r1_3 = approx_rcp(d1_3);
        float2 r_3 = make_float2(r0_3, r1_3);
        float2 s_3 = fma_f32x2_rn_noftz(r_3, fma_f32x2_rn_noftz(nd_3, r_3, make_float2(1.0f, 1.0f)), r_3);
        float2 _f32x2_mul_a_15 = mul_f32x2_noftz(cvt_f32_4, s_3);
        float2 mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&cvt_f32_1_3));
        packed[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_3.x, mul_3.y)));
        float2 cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(words[5]));
        float2 cvt_f32_1_4 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[5]));
        float2 _f32x2_mul_b_16 = make_float2(-1.442695f, -1.442695f);
        float2 t_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_4) : "l"(*(const unsigned long long*)&cvt_f32_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_16));
        float e0_4 = approx_exp2(t_4.x);
        float e1_4 = approx_exp2(t_4.y);
        float2 nd_4 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_4, e1_4));
        float d0_4 = -nd_4.x;
        float d1_4 = -nd_4.y;
        float r0_4 = approx_rcp(d0_4);
        float r1_4 = approx_rcp(d1_4);
        float2 r_4 = make_float2(r0_4, r1_4);
        float2 s_4 = fma_f32x2_rn_noftz(r_4, fma_f32x2_rn_noftz(nd_4, r_4, make_float2(1.0f, 1.0f)), r_4);
        float2 _f32x2_mul_a_17 = mul_f32x2_noftz(cvt_f32_5, s_4);
        float2 mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&cvt_f32_1_4));
        packed[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_4.x, mul_4.y)));
        float2 cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(words[6]));
        float2 cvt_f32_1_5 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[6]));
        float2 _f32x2_mul_b_18 = make_float2(-1.442695f, -1.442695f);
        float2 t_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_5) : "l"(*(const unsigned long long*)&cvt_f32_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_18));
        float e0_5 = approx_exp2(t_5.x);
        float e1_5 = approx_exp2(t_5.y);
        float2 nd_5 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_5, e1_5));
        float d0_5 = -nd_5.x;
        float d1_5 = -nd_5.y;
        float r0_5 = approx_rcp(d0_5);
        float r1_5 = approx_rcp(d1_5);
        float2 r_5 = make_float2(r0_5, r1_5);
        float2 s_5 = fma_f32x2_rn_noftz(r_5, fma_f32x2_rn_noftz(nd_5, r_5, make_float2(1.0f, 1.0f)), r_5);
        float2 _f32x2_mul_a_19 = mul_f32x2_noftz(cvt_f32_6, s_5);
        float2 mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&cvt_f32_1_5));
        packed[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_5.x, mul_5.y)));
        float2 cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(words[7]));
        float2 cvt_f32_1_6 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[7]));
        float2 _f32x2_mul_b_20 = make_float2(-1.442695f, -1.442695f);
        float2 t_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_6) : "l"(*(const unsigned long long*)&cvt_f32_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
        float e0_6 = approx_exp2(t_6.x);
        float e1_6 = approx_exp2(t_6.y);
        float2 nd_6 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_6, e1_6));
        float d0_6 = -nd_6.x;
        float d1_6 = -nd_6.y;
        float r0_6 = approx_rcp(d0_6);
        float r1_6 = approx_rcp(d1_6);
        float2 r_6 = make_float2(r0_6, r1_6);
        float2 s_6 = fma_f32x2_rn_noftz(r_6, fma_f32x2_rn_noftz(nd_6, r_6, make_float2(1.0f, 1.0f)), r_6);
        float2 _f32x2_mul_a_21 = mul_f32x2_noftz(cvt_f32_7, s_6);
        float2 mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&cvt_f32_1_6));
        packed[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_6.x, mul_6.y)));
        float2 cvt_f32_8 = cake_bf16x2_to_f32x2(__as_bf16x2(words_1[0]));
        float2 cvt_f32_1_7 = cake_bf16x2_to_f32x2(__as_bf16x2(words_2[0]));
        float2 _f32x2_mul_b_22 = make_float2(-1.442695f, -1.442695f);
        float2 t_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_7) : "l"(*(const unsigned long long*)&cvt_f32_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_22));
        float e0_7 = approx_exp2(t_7.x);
        float e1_7 = approx_exp2(t_7.y);
        float2 nd_7 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_7, e1_7));
        float d0_7 = -nd_7.x;
        float d1_7 = -nd_7.y;
        float r0_7 = approx_rcp(d0_7);
        float r1_7 = approx_rcp(d1_7);
        float2 r_7 = make_float2(r0_7, r1_7);
        float2 s_7 = fma_f32x2_rn_noftz(r_7, fma_f32x2_rn_noftz(nd_7, r_7, make_float2(1.0f, 1.0f)), r_7);
        float2 _f32x2_mul_a_23 = mul_f32x2_noftz(cvt_f32_8, s_7);
        float2 mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&cvt_f32_1_7));
        packed_hi[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_7.x, mul_7.y)));
        float2 cvt_f32_9 = cake_bf16x2_to_f32x2(__as_bf16x2(words_1[1]));
        float2 cvt_f32_1_8 = cake_bf16x2_to_f32x2(__as_bf16x2(words_2[1]));
        float2 _f32x2_mul_b_24 = make_float2(-1.442695f, -1.442695f);
        float2 t_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_8) : "l"(*(const unsigned long long*)&cvt_f32_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_24));
        float e0_8 = approx_exp2(t_8.x);
        float e1_8 = approx_exp2(t_8.y);
        float2 nd_8 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_8, e1_8));
        float d0_8 = -nd_8.x;
        float d1_8 = -nd_8.y;
        float r0_8 = approx_rcp(d0_8);
        float r1_8 = approx_rcp(d1_8);
        float2 r_8 = make_float2(r0_8, r1_8);
        float2 s_8 = fma_f32x2_rn_noftz(r_8, fma_f32x2_rn_noftz(nd_8, r_8, make_float2(1.0f, 1.0f)), r_8);
        float2 _f32x2_mul_a_25 = mul_f32x2_noftz(cvt_f32_9, s_8);
        float2 mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_25), "l"(*(const unsigned long long*)&cvt_f32_1_8));
        packed_hi[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_8.x, mul_8.y)));
        float2 cvt_f32_10 = cake_bf16x2_to_f32x2(__as_bf16x2(words_1[2]));
        float2 cvt_f32_1_9 = cake_bf16x2_to_f32x2(__as_bf16x2(words_2[2]));
        float2 _f32x2_mul_b_26 = make_float2(-1.442695f, -1.442695f);
        float2 t_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_9) : "l"(*(const unsigned long long*)&cvt_f32_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_26));
        float e0_9 = approx_exp2(t_9.x);
        float e1_9 = approx_exp2(t_9.y);
        float2 nd_9 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_9, e1_9));
        float d0_9 = -nd_9.x;
        float d1_9 = -nd_9.y;
        float r0_9 = approx_rcp(d0_9);
        float r1_9 = approx_rcp(d1_9);
        float2 r_9 = make_float2(r0_9, r1_9);
        float2 s_9 = fma_f32x2_rn_noftz(r_9, fma_f32x2_rn_noftz(nd_9, r_9, make_float2(1.0f, 1.0f)), r_9);
        float2 _f32x2_mul_a_27 = mul_f32x2_noftz(cvt_f32_10, s_9);
        float2 mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_27), "l"(*(const unsigned long long*)&cvt_f32_1_9));
        packed_hi[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_9.x, mul_9.y)));
        float2 cvt_f32_11 = cake_bf16x2_to_f32x2(__as_bf16x2(words_1[3]));
        float2 cvt_f32_1_10 = cake_bf16x2_to_f32x2(__as_bf16x2(words_2[3]));
        float2 _f32x2_mul_b_28 = make_float2(-1.442695f, -1.442695f);
        float2 t_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_10) : "l"(*(const unsigned long long*)&cvt_f32_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_28));
        float e0_10 = approx_exp2(t_10.x);
        float e1_10 = approx_exp2(t_10.y);
        float2 nd_10 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_10, e1_10));
        float d0_10 = -nd_10.x;
        float d1_10 = -nd_10.y;
        float r0_10 = approx_rcp(d0_10);
        float r1_10 = approx_rcp(d1_10);
        float2 r_10 = make_float2(r0_10, r1_10);
        float2 s_10 = fma_f32x2_rn_noftz(r_10, fma_f32x2_rn_noftz(nd_10, r_10, make_float2(1.0f, 1.0f)), r_10);
        float2 _f32x2_mul_a_29 = mul_f32x2_noftz(cvt_f32_11, s_10);
        float2 mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&cvt_f32_1_10));
        packed_hi[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_10.x, mul_10.y)));
        float2 cvt_f32_12 = cake_bf16x2_to_f32x2(__as_bf16x2(words_1[4]));
        float2 cvt_f32_1_11 = cake_bf16x2_to_f32x2(__as_bf16x2(words_2[4]));
        float2 _f32x2_mul_b_30 = make_float2(-1.442695f, -1.442695f);
        float2 t_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_11) : "l"(*(const unsigned long long*)&cvt_f32_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_30));
        float e0_11 = approx_exp2(t_11.x);
        float e1_11 = approx_exp2(t_11.y);
        float2 nd_11 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_11, e1_11));
        float d0_11 = -nd_11.x;
        float d1_11 = -nd_11.y;
        float r0_11 = approx_rcp(d0_11);
        float r1_11 = approx_rcp(d1_11);
        float2 r_11 = make_float2(r0_11, r1_11);
        float2 s_11 = fma_f32x2_rn_noftz(r_11, fma_f32x2_rn_noftz(nd_11, r_11, make_float2(1.0f, 1.0f)), r_11);
        float2 _f32x2_mul_a_31 = mul_f32x2_noftz(cvt_f32_12, s_11);
        float2 mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_31), "l"(*(const unsigned long long*)&cvt_f32_1_11));
        packed_hi[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_11.x, mul_11.y)));
        float2 cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(words_1[5]));
        float2 cvt_f32_1_12 = cake_bf16x2_to_f32x2(__as_bf16x2(words_2[5]));
        float2 _f32x2_mul_b_32 = make_float2(-1.442695f, -1.442695f);
        float2 t_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_12) : "l"(*(const unsigned long long*)&cvt_f32_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_32));
        float e0_12 = approx_exp2(t_12.x);
        float e1_12 = approx_exp2(t_12.y);
        float2 nd_12 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_12, e1_12));
        float d0_12 = -nd_12.x;
        float d1_12 = -nd_12.y;
        float r0_12 = approx_rcp(d0_12);
        float r1_12 = approx_rcp(d1_12);
        float2 r_12 = make_float2(r0_12, r1_12);
        float2 s_12 = fma_f32x2_rn_noftz(r_12, fma_f32x2_rn_noftz(nd_12, r_12, make_float2(1.0f, 1.0f)), r_12);
        float2 _f32x2_mul_a_33 = mul_f32x2_noftz(cvt_f32_13, s_12);
        float2 mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&cvt_f32_1_12));
        packed_hi[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_12.x, mul_12.y)));
        float2 cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(words_1[6]));
        float2 cvt_f32_1_13 = cake_bf16x2_to_f32x2(__as_bf16x2(words_2[6]));
        float2 _f32x2_mul_b_34 = make_float2(-1.442695f, -1.442695f);
        float2 t_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_13) : "l"(*(const unsigned long long*)&cvt_f32_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_34));
        float e0_13 = approx_exp2(t_13.x);
        float e1_13 = approx_exp2(t_13.y);
        float2 nd_13 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_13, e1_13));
        float d0_13 = -nd_13.x;
        float d1_13 = -nd_13.y;
        float r0_13 = approx_rcp(d0_13);
        float r1_13 = approx_rcp(d1_13);
        float2 r_13 = make_float2(r0_13, r1_13);
        float2 s_13 = fma_f32x2_rn_noftz(r_13, fma_f32x2_rn_noftz(nd_13, r_13, make_float2(1.0f, 1.0f)), r_13);
        float2 _f32x2_mul_a_35 = mul_f32x2_noftz(cvt_f32_14, s_13);
        float2 mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&cvt_f32_1_13));
        packed_hi[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_13.x, mul_13.y)));
        float2 cvt_f32_15 = cake_bf16x2_to_f32x2(__as_bf16x2(words_1[7]));
        float2 cvt_f32_1_14 = cake_bf16x2_to_f32x2(__as_bf16x2(words_2[7]));
        float2 _f32x2_mul_b_36 = make_float2(-1.442695f, -1.442695f);
        float2 t_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_14) : "l"(*(const unsigned long long*)&cvt_f32_15), "l"(*(const unsigned long long*)&_f32x2_mul_b_36));
        float e0_14 = approx_exp2(t_14.x);
        float e1_14 = approx_exp2(t_14.y);
        float2 nd_14 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_14, e1_14));
        float d0_14 = -nd_14.x;
        float d1_14 = -nd_14.y;
        float r0_14 = approx_rcp(d0_14);
        float r1_14 = approx_rcp(d1_14);
        float2 r_14 = make_float2(r0_14, r1_14);
        float2 s_14 = fma_f32x2_rn_noftz(r_14, fma_f32x2_rn_noftz(nd_14, r_14, make_float2(1.0f, 1.0f)), r_14);
        float2 _f32x2_mul_a_37 = mul_f32x2_noftz(cvt_f32_15, s_14);
        float2 mul_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_37), "l"(*(const unsigned long long*)&cvt_f32_1_14));
        packed_hi[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_14.x, mul_14.y)));
    } else {
        int band = 0;
        float2 cvt_f32_16 = cake_bf16x2_to_f32x2(__as_bf16x2(words[0]));
        float2 cvt_f32_1_15 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[0]));
        float2 _f32x2_mul_b_38 = make_float2(-1.442695f, -1.442695f);
        float2 t_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_15) : "l"(*(const unsigned long long*)&cvt_f32_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_38));
        float e0_15 = approx_exp2(t_15.x);
        float e1_15 = approx_exp2(t_15.y);
        float2 nd_15 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_15, e1_15));
        float d0_15 = -nd_15.x;
        float d1_15 = -nd_15.y;
        float r0_15 = approx_rcp(d0_15);
        float r1_15 = approx_rcp(d1_15);
        float2 r_15 = make_float2(r0_15, r1_15);
        float2 s_15 = fma_f32x2_rn_noftz(r_15, fma_f32x2_rn_noftz(nd_15, r_15, make_float2(1.0f, 1.0f)), r_15);
        float t0 = s_15.x;
        float t1 = s_15.y;
        int _warp_vote_39 = __any_sync(0xFFFFFFFF, fminf(t0, t1) == 0.0f);
        if (_warp_vote_39 != 0) {
            float d4_8a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a) : "f"(d0_15), "f"(0.25f));
            float r4_8a = approx_rcp(d4_8a);
            float _fma_41;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(-d4_8a), "f"(r4_8a), "f"(1.0f));
            float _fma_40;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(r4_8a), "f"(_fma_41), "f"(r4_8a));
            float big_8a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a) : "f"(mul_f32_rn_noftz(_fma_40, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_42;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(big_8a), "f"(8.388608e+06f));
            float whole_8a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a) : "f"(_add_42), "f"(8.388608e+06f));
            float above_8a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a) : "f"(-d4_8a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_44;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44) : "f"(big_8a), "f"(whole_8a));
            float _abs_43;
            asm("abs.f32 %0, %1;" : "=f"(_abs_43) : "f"(_sub_44));
            float _add_45;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(big_8a), "f"(0.5f));
            float _sub_46;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_46) : "f"(big_8a), "f"(0.5f));
            float rounded_8a = ((_abs_43 == 0.5f) ? ((above_8a > 0.0f) ? _add_45 : _sub_46) : whole_8a);
            t0 = ((d0_15 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a, 7.888609e-31f), 1.7763568e-15f) : t0);
            float d4_8a_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_0) : "f"(d1_15), "f"(0.25f));
            float r4_8a_0 = approx_rcp(d4_8a_0);
            float _fma_48;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(-d4_8a_0), "f"(r4_8a_0), "f"(1.0f));
            float _fma_47;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(r4_8a_0), "f"(_fma_48), "f"(r4_8a_0));
            float big_8a_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_0) : "f"(mul_f32_rn_noftz(_fma_47, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_49;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(big_8a_0), "f"(8.388608e+06f));
            float whole_8a_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_0) : "f"(_add_49), "f"(8.388608e+06f));
            float above_8a_0;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_0) : "f"(-d4_8a_0), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_0, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_51;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_51) : "f"(big_8a_0), "f"(whole_8a_0));
            float _abs_50;
            asm("abs.f32 %0, %1;" : "=f"(_abs_50) : "f"(_sub_51));
            float _add_52;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(big_8a_0), "f"(0.5f));
            float _sub_53;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_53) : "f"(big_8a_0), "f"(0.5f));
            float rounded_8a_0 = ((_abs_50 == 0.5f) ? ((above_8a_0 > 0.0f) ? _add_52 : _sub_53) : whole_8a_0);
            t1 = ((d1_15 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_0, 7.888609e-31f), 1.7763568e-15f) : t1);
        }
        float2 _f32x2_mul_a_54 = mul_f32x2_noftz(cvt_f32_16, make_float2(max_noftz(t0, 0.0f), max_noftz(t1, 0.0f)));
        float2 mul_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_54), "l"(*(const unsigned long long*)&cvt_f32_1_15));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_15.x, mul_15.y)));
        float2 cvt_f32_16_1 = cake_bf16x2_to_f32x2(__as_bf16x2(words[1]));
        float2 cvt_f32_1_15_1 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[1]));
        float2 _f32x2_mul_b_55 = make_float2(-1.442695f, -1.442695f);
        float2 t_15_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_15_1) : "l"(*(const unsigned long long*)&cvt_f32_16_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_55));
        float e0_15_1 = approx_exp2(t_15_1.x);
        float e1_15_1 = approx_exp2(t_15_1.y);
        float2 nd_15_1 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_15_1, e1_15_1));
        float d0_15_1 = -nd_15_1.x;
        float d1_15_1 = -nd_15_1.y;
        float r0_15_1 = approx_rcp(d0_15_1);
        float r1_15_1 = approx_rcp(d1_15_1);
        float2 r_15_1 = make_float2(r0_15_1, r1_15_1);
        float2 s_15_1 = fma_f32x2_rn_noftz(r_15_1, fma_f32x2_rn_noftz(nd_15_1, r_15_1, make_float2(1.0f, 1.0f)), r_15_1);
        float t0_0 = s_15_1.x;
        float t1_0 = s_15_1.y;
        int _warp_vote_56 = __any_sync(0xFFFFFFFF, fminf(t0_0, t1_0) == 0.0f);
        if (_warp_vote_56 != 0) {
            float d4_8a_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_1) : "f"(d0_15_1), "f"(0.25f));
            float r4_8a_1 = approx_rcp(d4_8a_1);
            float _fma_58;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(-d4_8a_1), "f"(r4_8a_1), "f"(1.0f));
            float _fma_57;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(r4_8a_1), "f"(_fma_58), "f"(r4_8a_1));
            float big_8a_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_1) : "f"(mul_f32_rn_noftz(_fma_57, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_59;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(big_8a_1), "f"(8.388608e+06f));
            float whole_8a_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_1) : "f"(_add_59), "f"(8.388608e+06f));
            float above_8a_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_1) : "f"(-d4_8a_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_61;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_61) : "f"(big_8a_1), "f"(whole_8a_1));
            float _abs_60;
            asm("abs.f32 %0, %1;" : "=f"(_abs_60) : "f"(_sub_61));
            float _add_62;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_62) : "f"(big_8a_1), "f"(0.5f));
            float _sub_63;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_63) : "f"(big_8a_1), "f"(0.5f));
            float rounded_8a_1 = ((_abs_60 == 0.5f) ? ((above_8a_1 > 0.0f) ? _add_62 : _sub_63) : whole_8a_1);
            t0_0 = ((d0_15_1 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_1, 7.888609e-31f), 1.7763568e-15f) : t0_0);
            float d4_8a_1_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_1_1) : "f"(d1_15_1), "f"(0.25f));
            float r4_8a_1_1 = approx_rcp(d4_8a_1_1);
            float _fma_65;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(-d4_8a_1_1), "f"(r4_8a_1_1), "f"(1.0f));
            float _fma_64;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(r4_8a_1_1), "f"(_fma_65), "f"(r4_8a_1_1));
            float big_8a_1_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_1_1) : "f"(mul_f32_rn_noftz(_fma_64, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_66;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_66) : "f"(big_8a_1_1), "f"(8.388608e+06f));
            float whole_8a_1_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_1_1) : "f"(_add_66), "f"(8.388608e+06f));
            float above_8a_1_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_1_1) : "f"(-d4_8a_1_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_1_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_68;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_68) : "f"(big_8a_1_1), "f"(whole_8a_1_1));
            float _abs_67;
            asm("abs.f32 %0, %1;" : "=f"(_abs_67) : "f"(_sub_68));
            float _add_69;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(big_8a_1_1), "f"(0.5f));
            float _sub_70;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_70) : "f"(big_8a_1_1), "f"(0.5f));
            float rounded_8a_1_1 = ((_abs_67 == 0.5f) ? ((above_8a_1_1 > 0.0f) ? _add_69 : _sub_70) : whole_8a_1_1);
            t1_0 = ((d1_15_1 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_1_1, 7.888609e-31f), 1.7763568e-15f) : t1_0);
        }
        float2 _f32x2_mul_a_71 = mul_f32x2_noftz(cvt_f32_16_1, make_float2(max_noftz(t0_0, 0.0f), max_noftz(t1_0, 0.0f)));
        float2 mul_15_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_15_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_71), "l"(*(const unsigned long long*)&cvt_f32_1_15_1));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_15_1.x, mul_15_1.y)));
        float2 cvt_f32_17 = cake_bf16x2_to_f32x2(__as_bf16x2(words[2]));
        float2 cvt_f32_1_16 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[2]));
        float2 _f32x2_mul_b_72 = make_float2(-1.442695f, -1.442695f);
        float2 t_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_16) : "l"(*(const unsigned long long*)&cvt_f32_17), "l"(*(const unsigned long long*)&_f32x2_mul_b_72));
        float e0_16 = approx_exp2(t_16.x);
        float e1_16 = approx_exp2(t_16.y);
        float2 nd_16 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_16, e1_16));
        float d0_16 = -nd_16.x;
        float d1_16 = -nd_16.y;
        float r0_16 = approx_rcp(d0_16);
        float r1_16 = approx_rcp(d1_16);
        float2 r_16 = make_float2(r0_16, r1_16);
        float2 s_16 = fma_f32x2_rn_noftz(r_16, fma_f32x2_rn_noftz(nd_16, r_16, make_float2(1.0f, 1.0f)), r_16);
        float t0_1 = s_16.x;
        float t1_1 = s_16.y;
        int _warp_vote_73 = __any_sync(0xFFFFFFFF, fminf(t0_1, t1_1) == 0.0f);
        if (_warp_vote_73 != 0) {
            float d4_8a_2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_2) : "f"(d0_16), "f"(0.25f));
            float r4_8a_2 = approx_rcp(d4_8a_2);
            float _fma_75;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(-d4_8a_2), "f"(r4_8a_2), "f"(1.0f));
            float _fma_74;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(r4_8a_2), "f"(_fma_75), "f"(r4_8a_2));
            float big_8a_2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_2) : "f"(mul_f32_rn_noftz(_fma_74, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_76;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(big_8a_2), "f"(8.388608e+06f));
            float whole_8a_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_2) : "f"(_add_76), "f"(8.388608e+06f));
            float above_8a_2;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_2) : "f"(-d4_8a_2), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_2, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_78;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_78) : "f"(big_8a_2), "f"(whole_8a_2));
            float _abs_77;
            asm("abs.f32 %0, %1;" : "=f"(_abs_77) : "f"(_sub_78));
            float _add_79;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(big_8a_2), "f"(0.5f));
            float _sub_80;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_80) : "f"(big_8a_2), "f"(0.5f));
            float rounded_8a_2 = ((_abs_77 == 0.5f) ? ((above_8a_2 > 0.0f) ? _add_79 : _sub_80) : whole_8a_2);
            t0_1 = ((d0_16 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_2, 7.888609e-31f), 1.7763568e-15f) : t0_1);
            float d4_8a_2_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_2_1) : "f"(d1_16), "f"(0.25f));
            float r4_8a_2_1 = approx_rcp(d4_8a_2_1);
            float _fma_82;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(-d4_8a_2_1), "f"(r4_8a_2_1), "f"(1.0f));
            float _fma_81;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(r4_8a_2_1), "f"(_fma_82), "f"(r4_8a_2_1));
            float big_8a_2_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_2_1) : "f"(mul_f32_rn_noftz(_fma_81, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_83;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(big_8a_2_1), "f"(8.388608e+06f));
            float whole_8a_2_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_2_1) : "f"(_add_83), "f"(8.388608e+06f));
            float above_8a_2_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_2_1) : "f"(-d4_8a_2_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_2_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_85;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_85) : "f"(big_8a_2_1), "f"(whole_8a_2_1));
            float _abs_84;
            asm("abs.f32 %0, %1;" : "=f"(_abs_84) : "f"(_sub_85));
            float _add_86;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_86) : "f"(big_8a_2_1), "f"(0.5f));
            float _sub_87;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_87) : "f"(big_8a_2_1), "f"(0.5f));
            float rounded_8a_2_1 = ((_abs_84 == 0.5f) ? ((above_8a_2_1 > 0.0f) ? _add_86 : _sub_87) : whole_8a_2_1);
            t1_1 = ((d1_16 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_2_1, 7.888609e-31f), 1.7763568e-15f) : t1_1);
        }
        float2 _f32x2_mul_a_88 = mul_f32x2_noftz(cvt_f32_17, make_float2(max_noftz(t0_1, 0.0f), max_noftz(t1_1, 0.0f)));
        float2 mul_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_88), "l"(*(const unsigned long long*)&cvt_f32_1_16));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_16.x, mul_16.y)));
        float2 cvt_f32_18 = cake_bf16x2_to_f32x2(__as_bf16x2(words[3]));
        float2 cvt_f32_1_17 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[3]));
        float2 _f32x2_mul_b_89 = make_float2(-1.442695f, -1.442695f);
        float2 t_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_17) : "l"(*(const unsigned long long*)&cvt_f32_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_89));
        float e0_17 = approx_exp2(t_17.x);
        float e1_17 = approx_exp2(t_17.y);
        float2 nd_17 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_17, e1_17));
        float d0_17 = -nd_17.x;
        float d1_17 = -nd_17.y;
        float r0_17 = approx_rcp(d0_17);
        float r1_17 = approx_rcp(d1_17);
        float2 r_17 = make_float2(r0_17, r1_17);
        float2 s_17 = fma_f32x2_rn_noftz(r_17, fma_f32x2_rn_noftz(nd_17, r_17, make_float2(1.0f, 1.0f)), r_17);
        float t0_2 = s_17.x;
        float t1_2 = s_17.y;
        int _warp_vote_90 = __any_sync(0xFFFFFFFF, fminf(t0_2, t1_2) == 0.0f);
        if (_warp_vote_90 != 0) {
            float d4_8a_3;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_3) : "f"(d0_17), "f"(0.25f));
            float r4_8a_3 = approx_rcp(d4_8a_3);
            float _fma_92;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(-d4_8a_3), "f"(r4_8a_3), "f"(1.0f));
            float _fma_91;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(r4_8a_3), "f"(_fma_92), "f"(r4_8a_3));
            float big_8a_3;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_3) : "f"(mul_f32_rn_noftz(_fma_91, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_93;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(big_8a_3), "f"(8.388608e+06f));
            float whole_8a_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_3) : "f"(_add_93), "f"(8.388608e+06f));
            float above_8a_3;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_3) : "f"(-d4_8a_3), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_3, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_95;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_95) : "f"(big_8a_3), "f"(whole_8a_3));
            float _abs_94;
            asm("abs.f32 %0, %1;" : "=f"(_abs_94) : "f"(_sub_95));
            float _add_96;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_96) : "f"(big_8a_3), "f"(0.5f));
            float _sub_97;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_97) : "f"(big_8a_3), "f"(0.5f));
            float rounded_8a_3 = ((_abs_94 == 0.5f) ? ((above_8a_3 > 0.0f) ? _add_96 : _sub_97) : whole_8a_3);
            t0_2 = ((d0_17 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_3, 7.888609e-31f), 1.7763568e-15f) : t0_2);
            float d4_8a_3_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_3_1) : "f"(d1_17), "f"(0.25f));
            float r4_8a_3_1 = approx_rcp(d4_8a_3_1);
            float _fma_99;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(-d4_8a_3_1), "f"(r4_8a_3_1), "f"(1.0f));
            float _fma_98;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(r4_8a_3_1), "f"(_fma_99), "f"(r4_8a_3_1));
            float big_8a_3_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_3_1) : "f"(mul_f32_rn_noftz(_fma_98, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_100;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_100) : "f"(big_8a_3_1), "f"(8.388608e+06f));
            float whole_8a_3_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_3_1) : "f"(_add_100), "f"(8.388608e+06f));
            float above_8a_3_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_3_1) : "f"(-d4_8a_3_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_3_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_102;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_102) : "f"(big_8a_3_1), "f"(whole_8a_3_1));
            float _abs_101;
            asm("abs.f32 %0, %1;" : "=f"(_abs_101) : "f"(_sub_102));
            float _add_103;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_103) : "f"(big_8a_3_1), "f"(0.5f));
            float _sub_104;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_104) : "f"(big_8a_3_1), "f"(0.5f));
            float rounded_8a_3_1 = ((_abs_101 == 0.5f) ? ((above_8a_3_1 > 0.0f) ? _add_103 : _sub_104) : whole_8a_3_1);
            t1_2 = ((d1_17 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_3_1, 7.888609e-31f), 1.7763568e-15f) : t1_2);
        }
        float2 _f32x2_mul_a_105 = mul_f32x2_noftz(cvt_f32_18, make_float2(max_noftz(t0_2, 0.0f), max_noftz(t1_2, 0.0f)));
        float2 mul_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_105), "l"(*(const unsigned long long*)&cvt_f32_1_17));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_17.x, mul_17.y)));
        float2 cvt_f32_19 = cake_bf16x2_to_f32x2(__as_bf16x2(words[4]));
        float2 cvt_f32_1_18 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[4]));
        float2 _f32x2_mul_b_106 = make_float2(-1.442695f, -1.442695f);
        float2 t_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_18) : "l"(*(const unsigned long long*)&cvt_f32_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_106));
        float e0_18 = approx_exp2(t_18.x);
        float e1_18 = approx_exp2(t_18.y);
        float2 nd_18 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_18, e1_18));
        float d0_18 = -nd_18.x;
        float d1_18 = -nd_18.y;
        float r0_18 = approx_rcp(d0_18);
        float r1_18 = approx_rcp(d1_18);
        float2 r_18 = make_float2(r0_18, r1_18);
        float2 s_18 = fma_f32x2_rn_noftz(r_18, fma_f32x2_rn_noftz(nd_18, r_18, make_float2(1.0f, 1.0f)), r_18);
        float t0_3 = s_18.x;
        float t1_3 = s_18.y;
        int _warp_vote_107 = __any_sync(0xFFFFFFFF, fminf(t0_3, t1_3) == 0.0f);
        if (_warp_vote_107 != 0) {
            float d4_8a_4;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_4) : "f"(d0_18), "f"(0.25f));
            float r4_8a_4 = approx_rcp(d4_8a_4);
            float _fma_109;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_109) : "f"(-d4_8a_4), "f"(r4_8a_4), "f"(1.0f));
            float _fma_108;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_108) : "f"(r4_8a_4), "f"(_fma_109), "f"(r4_8a_4));
            float big_8a_4;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_4) : "f"(mul_f32_rn_noftz(_fma_108, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_110;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_110) : "f"(big_8a_4), "f"(8.388608e+06f));
            float whole_8a_4;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_4) : "f"(_add_110), "f"(8.388608e+06f));
            float above_8a_4;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_4) : "f"(-d4_8a_4), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_4, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_112;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_112) : "f"(big_8a_4), "f"(whole_8a_4));
            float _abs_111;
            asm("abs.f32 %0, %1;" : "=f"(_abs_111) : "f"(_sub_112));
            float _add_113;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_113) : "f"(big_8a_4), "f"(0.5f));
            float _sub_114;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_114) : "f"(big_8a_4), "f"(0.5f));
            float rounded_8a_4 = ((_abs_111 == 0.5f) ? ((above_8a_4 > 0.0f) ? _add_113 : _sub_114) : whole_8a_4);
            t0_3 = ((d0_18 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_4, 7.888609e-31f), 1.7763568e-15f) : t0_3);
            float d4_8a_4_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_4_1) : "f"(d1_18), "f"(0.25f));
            float r4_8a_4_1 = approx_rcp(d4_8a_4_1);
            float _fma_116;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_116) : "f"(-d4_8a_4_1), "f"(r4_8a_4_1), "f"(1.0f));
            float _fma_115;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_115) : "f"(r4_8a_4_1), "f"(_fma_116), "f"(r4_8a_4_1));
            float big_8a_4_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_4_1) : "f"(mul_f32_rn_noftz(_fma_115, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_117;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_117) : "f"(big_8a_4_1), "f"(8.388608e+06f));
            float whole_8a_4_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_4_1) : "f"(_add_117), "f"(8.388608e+06f));
            float above_8a_4_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_4_1) : "f"(-d4_8a_4_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_4_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_119;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_119) : "f"(big_8a_4_1), "f"(whole_8a_4_1));
            float _abs_118;
            asm("abs.f32 %0, %1;" : "=f"(_abs_118) : "f"(_sub_119));
            float _add_120;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_120) : "f"(big_8a_4_1), "f"(0.5f));
            float _sub_121;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_121) : "f"(big_8a_4_1), "f"(0.5f));
            float rounded_8a_4_1 = ((_abs_118 == 0.5f) ? ((above_8a_4_1 > 0.0f) ? _add_120 : _sub_121) : whole_8a_4_1);
            t1_3 = ((d1_18 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_4_1, 7.888609e-31f), 1.7763568e-15f) : t1_3);
        }
        float2 _f32x2_mul_a_122 = mul_f32x2_noftz(cvt_f32_19, make_float2(max_noftz(t0_3, 0.0f), max_noftz(t1_3, 0.0f)));
        float2 mul_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_122), "l"(*(const unsigned long long*)&cvt_f32_1_18));
        packed[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_18.x, mul_18.y)));
        float2 cvt_f32_20 = cake_bf16x2_to_f32x2(__as_bf16x2(words[5]));
        float2 cvt_f32_1_19 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[5]));
        float2 _f32x2_mul_b_123 = make_float2(-1.442695f, -1.442695f);
        float2 t_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_19) : "l"(*(const unsigned long long*)&cvt_f32_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_123));
        float e0_19 = approx_exp2(t_19.x);
        float e1_19 = approx_exp2(t_19.y);
        float2 nd_19 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_19, e1_19));
        float d0_19 = -nd_19.x;
        float d1_19 = -nd_19.y;
        float r0_19 = approx_rcp(d0_19);
        float r1_19 = approx_rcp(d1_19);
        float2 r_19 = make_float2(r0_19, r1_19);
        float2 s_19 = fma_f32x2_rn_noftz(r_19, fma_f32x2_rn_noftz(nd_19, r_19, make_float2(1.0f, 1.0f)), r_19);
        float t0_4 = s_19.x;
        float t1_4 = s_19.y;
        int _warp_vote_124 = __any_sync(0xFFFFFFFF, fminf(t0_4, t1_4) == 0.0f);
        if (_warp_vote_124 != 0) {
            float d4_8a_5;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_5) : "f"(d0_19), "f"(0.25f));
            float r4_8a_5 = approx_rcp(d4_8a_5);
            float _fma_126;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_126) : "f"(-d4_8a_5), "f"(r4_8a_5), "f"(1.0f));
            float _fma_125;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(r4_8a_5), "f"(_fma_126), "f"(r4_8a_5));
            float big_8a_5;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_5) : "f"(mul_f32_rn_noftz(_fma_125, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_127;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_127) : "f"(big_8a_5), "f"(8.388608e+06f));
            float whole_8a_5;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_5) : "f"(_add_127), "f"(8.388608e+06f));
            float above_8a_5;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_5) : "f"(-d4_8a_5), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_5, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_129;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_129) : "f"(big_8a_5), "f"(whole_8a_5));
            float _abs_128;
            asm("abs.f32 %0, %1;" : "=f"(_abs_128) : "f"(_sub_129));
            float _add_130;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_130) : "f"(big_8a_5), "f"(0.5f));
            float _sub_131;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_131) : "f"(big_8a_5), "f"(0.5f));
            float rounded_8a_5 = ((_abs_128 == 0.5f) ? ((above_8a_5 > 0.0f) ? _add_130 : _sub_131) : whole_8a_5);
            t0_4 = ((d0_19 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_5, 7.888609e-31f), 1.7763568e-15f) : t0_4);
            float d4_8a_5_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_5_1) : "f"(d1_19), "f"(0.25f));
            float r4_8a_5_1 = approx_rcp(d4_8a_5_1);
            float _fma_133;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_133) : "f"(-d4_8a_5_1), "f"(r4_8a_5_1), "f"(1.0f));
            float _fma_132;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_132) : "f"(r4_8a_5_1), "f"(_fma_133), "f"(r4_8a_5_1));
            float big_8a_5_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_5_1) : "f"(mul_f32_rn_noftz(_fma_132, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_134;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_134) : "f"(big_8a_5_1), "f"(8.388608e+06f));
            float whole_8a_5_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_5_1) : "f"(_add_134), "f"(8.388608e+06f));
            float above_8a_5_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_5_1) : "f"(-d4_8a_5_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_5_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_136;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_136) : "f"(big_8a_5_1), "f"(whole_8a_5_1));
            float _abs_135;
            asm("abs.f32 %0, %1;" : "=f"(_abs_135) : "f"(_sub_136));
            float _add_137;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_137) : "f"(big_8a_5_1), "f"(0.5f));
            float _sub_138;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_138) : "f"(big_8a_5_1), "f"(0.5f));
            float rounded_8a_5_1 = ((_abs_135 == 0.5f) ? ((above_8a_5_1 > 0.0f) ? _add_137 : _sub_138) : whole_8a_5_1);
            t1_4 = ((d1_19 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_5_1, 7.888609e-31f), 1.7763568e-15f) : t1_4);
        }
        float2 _f32x2_mul_a_139 = mul_f32x2_noftz(cvt_f32_20, make_float2(max_noftz(t0_4, 0.0f), max_noftz(t1_4, 0.0f)));
        float2 mul_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_139), "l"(*(const unsigned long long*)&cvt_f32_1_19));
        packed[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_19.x, mul_19.y)));
        float2 cvt_f32_21 = cake_bf16x2_to_f32x2(__as_bf16x2(words[6]));
        float2 cvt_f32_1_20 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[6]));
        float2 _f32x2_mul_b_140 = make_float2(-1.442695f, -1.442695f);
        float2 t_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_20) : "l"(*(const unsigned long long*)&cvt_f32_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_140));
        float e0_20 = approx_exp2(t_20.x);
        float e1_20 = approx_exp2(t_20.y);
        float2 nd_20 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_20, e1_20));
        float d0_20 = -nd_20.x;
        float d1_20 = -nd_20.y;
        float r0_20 = approx_rcp(d0_20);
        float r1_20 = approx_rcp(d1_20);
        float2 r_20 = make_float2(r0_20, r1_20);
        float2 s_20 = fma_f32x2_rn_noftz(r_20, fma_f32x2_rn_noftz(nd_20, r_20, make_float2(1.0f, 1.0f)), r_20);
        float t0_5 = s_20.x;
        float t1_5 = s_20.y;
        int _warp_vote_141 = __any_sync(0xFFFFFFFF, fminf(t0_5, t1_5) == 0.0f);
        if (_warp_vote_141 != 0) {
            float d4_8a_6;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_6) : "f"(d0_20), "f"(0.25f));
            float r4_8a_6 = approx_rcp(d4_8a_6);
            float _fma_143;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_143) : "f"(-d4_8a_6), "f"(r4_8a_6), "f"(1.0f));
            float _fma_142;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_142) : "f"(r4_8a_6), "f"(_fma_143), "f"(r4_8a_6));
            float big_8a_6;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_6) : "f"(mul_f32_rn_noftz(_fma_142, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_144;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_144) : "f"(big_8a_6), "f"(8.388608e+06f));
            float whole_8a_6;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_6) : "f"(_add_144), "f"(8.388608e+06f));
            float above_8a_6;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_6) : "f"(-d4_8a_6), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_6, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_146;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_146) : "f"(big_8a_6), "f"(whole_8a_6));
            float _abs_145;
            asm("abs.f32 %0, %1;" : "=f"(_abs_145) : "f"(_sub_146));
            float _add_147;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_147) : "f"(big_8a_6), "f"(0.5f));
            float _sub_148;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_148) : "f"(big_8a_6), "f"(0.5f));
            float rounded_8a_6 = ((_abs_145 == 0.5f) ? ((above_8a_6 > 0.0f) ? _add_147 : _sub_148) : whole_8a_6);
            t0_5 = ((d0_20 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_6, 7.888609e-31f), 1.7763568e-15f) : t0_5);
            float d4_8a_6_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_6_1) : "f"(d1_20), "f"(0.25f));
            float r4_8a_6_1 = approx_rcp(d4_8a_6_1);
            float _fma_150;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_150) : "f"(-d4_8a_6_1), "f"(r4_8a_6_1), "f"(1.0f));
            float _fma_149;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_149) : "f"(r4_8a_6_1), "f"(_fma_150), "f"(r4_8a_6_1));
            float big_8a_6_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_6_1) : "f"(mul_f32_rn_noftz(_fma_149, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_151;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_151) : "f"(big_8a_6_1), "f"(8.388608e+06f));
            float whole_8a_6_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_6_1) : "f"(_add_151), "f"(8.388608e+06f));
            float above_8a_6_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_6_1) : "f"(-d4_8a_6_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_6_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_153;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_153) : "f"(big_8a_6_1), "f"(whole_8a_6_1));
            float _abs_152;
            asm("abs.f32 %0, %1;" : "=f"(_abs_152) : "f"(_sub_153));
            float _add_154;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_154) : "f"(big_8a_6_1), "f"(0.5f));
            float _sub_155;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_155) : "f"(big_8a_6_1), "f"(0.5f));
            float rounded_8a_6_1 = ((_abs_152 == 0.5f) ? ((above_8a_6_1 > 0.0f) ? _add_154 : _sub_155) : whole_8a_6_1);
            t1_5 = ((d1_20 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_6_1, 7.888609e-31f), 1.7763568e-15f) : t1_5);
        }
        float2 _f32x2_mul_a_156 = mul_f32x2_noftz(cvt_f32_21, make_float2(max_noftz(t0_5, 0.0f), max_noftz(t1_5, 0.0f)));
        float2 mul_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_156), "l"(*(const unsigned long long*)&cvt_f32_1_20));
        packed[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_20.x, mul_20.y)));
        float2 cvt_f32_22 = cake_bf16x2_to_f32x2(__as_bf16x2(words[7]));
        float2 cvt_f32_1_21 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[7]));
        float2 _f32x2_mul_b_157 = make_float2(-1.442695f, -1.442695f);
        float2 t_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_21) : "l"(*(const unsigned long long*)&cvt_f32_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_157));
        float e0_21 = approx_exp2(t_21.x);
        float e1_21 = approx_exp2(t_21.y);
        float2 nd_21 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_21, e1_21));
        float d0_21 = -nd_21.x;
        float d1_21 = -nd_21.y;
        float r0_21 = approx_rcp(d0_21);
        float r1_21 = approx_rcp(d1_21);
        float2 r_21 = make_float2(r0_21, r1_21);
        float2 s_21 = fma_f32x2_rn_noftz(r_21, fma_f32x2_rn_noftz(nd_21, r_21, make_float2(1.0f, 1.0f)), r_21);
        float t0_6 = s_21.x;
        float t1_6 = s_21.y;
        int _warp_vote_158 = __any_sync(0xFFFFFFFF, fminf(t0_6, t1_6) == 0.0f);
        if (_warp_vote_158 != 0) {
            float d4_8a_7;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_7) : "f"(d0_21), "f"(0.25f));
            float r4_8a_7 = approx_rcp(d4_8a_7);
            float _fma_160;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_160) : "f"(-d4_8a_7), "f"(r4_8a_7), "f"(1.0f));
            float _fma_159;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_159) : "f"(r4_8a_7), "f"(_fma_160), "f"(r4_8a_7));
            float big_8a_7;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_7) : "f"(mul_f32_rn_noftz(_fma_159, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_161;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_161) : "f"(big_8a_7), "f"(8.388608e+06f));
            float whole_8a_7;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_7) : "f"(_add_161), "f"(8.388608e+06f));
            float above_8a_7;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_7) : "f"(-d4_8a_7), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_7, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_163;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_163) : "f"(big_8a_7), "f"(whole_8a_7));
            float _abs_162;
            asm("abs.f32 %0, %1;" : "=f"(_abs_162) : "f"(_sub_163));
            float _add_164;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_164) : "f"(big_8a_7), "f"(0.5f));
            float _sub_165;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_165) : "f"(big_8a_7), "f"(0.5f));
            float rounded_8a_7 = ((_abs_162 == 0.5f) ? ((above_8a_7 > 0.0f) ? _add_164 : _sub_165) : whole_8a_7);
            t0_6 = ((d0_21 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_7, 7.888609e-31f), 1.7763568e-15f) : t0_6);
            float d4_8a_7_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_7_1) : "f"(d1_21), "f"(0.25f));
            float r4_8a_7_1 = approx_rcp(d4_8a_7_1);
            float _fma_167;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_167) : "f"(-d4_8a_7_1), "f"(r4_8a_7_1), "f"(1.0f));
            float _fma_166;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_166) : "f"(r4_8a_7_1), "f"(_fma_167), "f"(r4_8a_7_1));
            float big_8a_7_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_7_1) : "f"(mul_f32_rn_noftz(_fma_166, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_168;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_168) : "f"(big_8a_7_1), "f"(8.388608e+06f));
            float whole_8a_7_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_7_1) : "f"(_add_168), "f"(8.388608e+06f));
            float above_8a_7_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_7_1) : "f"(-d4_8a_7_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_7_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_170;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_170) : "f"(big_8a_7_1), "f"(whole_8a_7_1));
            float _abs_169;
            asm("abs.f32 %0, %1;" : "=f"(_abs_169) : "f"(_sub_170));
            float _add_171;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_171) : "f"(big_8a_7_1), "f"(0.5f));
            float _sub_172;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_172) : "f"(big_8a_7_1), "f"(0.5f));
            float rounded_8a_7_1 = ((_abs_169 == 0.5f) ? ((above_8a_7_1 > 0.0f) ? _add_171 : _sub_172) : whole_8a_7_1);
            t1_6 = ((d1_21 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_7_1, 7.888609e-31f), 1.7763568e-15f) : t1_6);
        }
        float2 _f32x2_mul_a_173 = mul_f32x2_noftz(cvt_f32_22, make_float2(max_noftz(t0_6, 0.0f), max_noftz(t1_6, 0.0f)));
        float2 mul_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_173), "l"(*(const unsigned long long*)&cvt_f32_1_21));
        packed[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_21.x, mul_21.y)));
        int band_hi = 0;
        float2 cvt_f32_23 = cake_bf16x2_to_f32x2(__as_bf16x2(words_1[0]));
        float2 cvt_f32_1_22 = cake_bf16x2_to_f32x2(__as_bf16x2(words_2[0]));
        float2 _f32x2_mul_b_174 = make_float2(-1.442695f, -1.442695f);
        float2 t_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_22) : "l"(*(const unsigned long long*)&cvt_f32_23), "l"(*(const unsigned long long*)&_f32x2_mul_b_174));
        float e0_22 = approx_exp2(t_22.x);
        float e1_22 = approx_exp2(t_22.y);
        float2 nd_22 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_22, e1_22));
        float d0_22 = -nd_22.x;
        float d1_22 = -nd_22.y;
        float r0_22 = approx_rcp(d0_22);
        float r1_22 = approx_rcp(d1_22);
        float2 r_22 = make_float2(r0_22, r1_22);
        float2 s_22 = fma_f32x2_rn_noftz(r_22, fma_f32x2_rn_noftz(nd_22, r_22, make_float2(1.0f, 1.0f)), r_22);
        float t0_7 = s_22.x;
        float t1_7 = s_22.y;
        int _warp_vote_175 = __any_sync(0xFFFFFFFF, fminf(t0_7, t1_7) == 0.0f);
        if (_warp_vote_175 != 0) {
            float d4_8a_8;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_8) : "f"(d0_22), "f"(0.25f));
            float r4_8a_8 = approx_rcp(d4_8a_8);
            float _fma_177;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_177) : "f"(-d4_8a_8), "f"(r4_8a_8), "f"(1.0f));
            float _fma_176;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_176) : "f"(r4_8a_8), "f"(_fma_177), "f"(r4_8a_8));
            float big_8a_8;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_8) : "f"(mul_f32_rn_noftz(_fma_176, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_178;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_178) : "f"(big_8a_8), "f"(8.388608e+06f));
            float whole_8a_8;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_8) : "f"(_add_178), "f"(8.388608e+06f));
            float above_8a_8;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_8) : "f"(-d4_8a_8), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_8, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_180;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_180) : "f"(big_8a_8), "f"(whole_8a_8));
            float _abs_179;
            asm("abs.f32 %0, %1;" : "=f"(_abs_179) : "f"(_sub_180));
            float _add_181;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_181) : "f"(big_8a_8), "f"(0.5f));
            float _sub_182;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_182) : "f"(big_8a_8), "f"(0.5f));
            float rounded_8a_8 = ((_abs_179 == 0.5f) ? ((above_8a_8 > 0.0f) ? _add_181 : _sub_182) : whole_8a_8);
            t0_7 = ((d0_22 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_8, 7.888609e-31f), 1.7763568e-15f) : t0_7);
            float d4_8a_8_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_8_1) : "f"(d1_22), "f"(0.25f));
            float r4_8a_8_1 = approx_rcp(d4_8a_8_1);
            float _fma_184;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_184) : "f"(-d4_8a_8_1), "f"(r4_8a_8_1), "f"(1.0f));
            float _fma_183;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_183) : "f"(r4_8a_8_1), "f"(_fma_184), "f"(r4_8a_8_1));
            float big_8a_8_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_8_1) : "f"(mul_f32_rn_noftz(_fma_183, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_185;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_185) : "f"(big_8a_8_1), "f"(8.388608e+06f));
            float whole_8a_8_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_8_1) : "f"(_add_185), "f"(8.388608e+06f));
            float above_8a_8_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_8_1) : "f"(-d4_8a_8_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_8_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_187;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_187) : "f"(big_8a_8_1), "f"(whole_8a_8_1));
            float _abs_186;
            asm("abs.f32 %0, %1;" : "=f"(_abs_186) : "f"(_sub_187));
            float _add_188;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_188) : "f"(big_8a_8_1), "f"(0.5f));
            float _sub_189;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_189) : "f"(big_8a_8_1), "f"(0.5f));
            float rounded_8a_8_1 = ((_abs_186 == 0.5f) ? ((above_8a_8_1 > 0.0f) ? _add_188 : _sub_189) : whole_8a_8_1);
            t1_7 = ((d1_22 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_8_1, 7.888609e-31f), 1.7763568e-15f) : t1_7);
        }
        float2 _f32x2_mul_a_190 = mul_f32x2_noftz(cvt_f32_23, make_float2(max_noftz(t0_7, 0.0f), max_noftz(t1_7, 0.0f)));
        float2 mul_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_190), "l"(*(const unsigned long long*)&cvt_f32_1_22));
        packed_hi[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_22.x, mul_22.y)));
        float2 cvt_f32_24 = cake_bf16x2_to_f32x2(__as_bf16x2(words_1[1]));
        float2 cvt_f32_1_23 = cake_bf16x2_to_f32x2(__as_bf16x2(words_2[1]));
        float2 _f32x2_mul_b_191 = make_float2(-1.442695f, -1.442695f);
        float2 t_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_23) : "l"(*(const unsigned long long*)&cvt_f32_24), "l"(*(const unsigned long long*)&_f32x2_mul_b_191));
        float e0_23 = approx_exp2(t_23.x);
        float e1_23 = approx_exp2(t_23.y);
        float2 nd_23 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_23, e1_23));
        float d0_23 = -nd_23.x;
        float d1_23 = -nd_23.y;
        float r0_23 = approx_rcp(d0_23);
        float r1_23 = approx_rcp(d1_23);
        float2 r_23 = make_float2(r0_23, r1_23);
        float2 s_23 = fma_f32x2_rn_noftz(r_23, fma_f32x2_rn_noftz(nd_23, r_23, make_float2(1.0f, 1.0f)), r_23);
        float t0_8 = s_23.x;
        float t1_8 = s_23.y;
        int _warp_vote_192 = __any_sync(0xFFFFFFFF, fminf(t0_8, t1_8) == 0.0f);
        if (_warp_vote_192 != 0) {
            float d4_8a_9;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_9) : "f"(d0_23), "f"(0.25f));
            float r4_8a_9 = approx_rcp(d4_8a_9);
            float _fma_194;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_194) : "f"(-d4_8a_9), "f"(r4_8a_9), "f"(1.0f));
            float _fma_193;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_193) : "f"(r4_8a_9), "f"(_fma_194), "f"(r4_8a_9));
            float big_8a_9;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_9) : "f"(mul_f32_rn_noftz(_fma_193, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_195;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_195) : "f"(big_8a_9), "f"(8.388608e+06f));
            float whole_8a_9;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_9) : "f"(_add_195), "f"(8.388608e+06f));
            float above_8a_9;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_9) : "f"(-d4_8a_9), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_9, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_197;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_197) : "f"(big_8a_9), "f"(whole_8a_9));
            float _abs_196;
            asm("abs.f32 %0, %1;" : "=f"(_abs_196) : "f"(_sub_197));
            float _add_198;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_198) : "f"(big_8a_9), "f"(0.5f));
            float _sub_199;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_199) : "f"(big_8a_9), "f"(0.5f));
            float rounded_8a_9 = ((_abs_196 == 0.5f) ? ((above_8a_9 > 0.0f) ? _add_198 : _sub_199) : whole_8a_9);
            t0_8 = ((d0_23 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_9, 7.888609e-31f), 1.7763568e-15f) : t0_8);
            float d4_8a_9_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_9_1) : "f"(d1_23), "f"(0.25f));
            float r4_8a_9_1 = approx_rcp(d4_8a_9_1);
            float _fma_201;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_201) : "f"(-d4_8a_9_1), "f"(r4_8a_9_1), "f"(1.0f));
            float _fma_200;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_200) : "f"(r4_8a_9_1), "f"(_fma_201), "f"(r4_8a_9_1));
            float big_8a_9_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_9_1) : "f"(mul_f32_rn_noftz(_fma_200, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_202;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_202) : "f"(big_8a_9_1), "f"(8.388608e+06f));
            float whole_8a_9_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_9_1) : "f"(_add_202), "f"(8.388608e+06f));
            float above_8a_9_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_9_1) : "f"(-d4_8a_9_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_9_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_204;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_204) : "f"(big_8a_9_1), "f"(whole_8a_9_1));
            float _abs_203;
            asm("abs.f32 %0, %1;" : "=f"(_abs_203) : "f"(_sub_204));
            float _add_205;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_205) : "f"(big_8a_9_1), "f"(0.5f));
            float _sub_206;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_206) : "f"(big_8a_9_1), "f"(0.5f));
            float rounded_8a_9_1 = ((_abs_203 == 0.5f) ? ((above_8a_9_1 > 0.0f) ? _add_205 : _sub_206) : whole_8a_9_1);
            t1_8 = ((d1_23 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_9_1, 7.888609e-31f), 1.7763568e-15f) : t1_8);
        }
        float2 _f32x2_mul_a_207 = mul_f32x2_noftz(cvt_f32_24, make_float2(max_noftz(t0_8, 0.0f), max_noftz(t1_8, 0.0f)));
        float2 mul_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_207), "l"(*(const unsigned long long*)&cvt_f32_1_23));
        packed_hi[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_23.x, mul_23.y)));
        float2 cvt_f32_25 = cake_bf16x2_to_f32x2(__as_bf16x2(words_1[2]));
        float2 cvt_f32_1_24 = cake_bf16x2_to_f32x2(__as_bf16x2(words_2[2]));
        float2 _f32x2_mul_b_208 = make_float2(-1.442695f, -1.442695f);
        float2 t_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_24) : "l"(*(const unsigned long long*)&cvt_f32_25), "l"(*(const unsigned long long*)&_f32x2_mul_b_208));
        float e0_24 = approx_exp2(t_24.x);
        float e1_24 = approx_exp2(t_24.y);
        float2 nd_24 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_24, e1_24));
        float d0_24 = -nd_24.x;
        float d1_24 = -nd_24.y;
        float r0_24 = approx_rcp(d0_24);
        float r1_24 = approx_rcp(d1_24);
        float2 r_24 = make_float2(r0_24, r1_24);
        float2 s_24 = fma_f32x2_rn_noftz(r_24, fma_f32x2_rn_noftz(nd_24, r_24, make_float2(1.0f, 1.0f)), r_24);
        float t0_9 = s_24.x;
        float t1_9 = s_24.y;
        int _warp_vote_209 = __any_sync(0xFFFFFFFF, fminf(t0_9, t1_9) == 0.0f);
        if (_warp_vote_209 != 0) {
            float d4_8a_10;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_10) : "f"(d0_24), "f"(0.25f));
            float r4_8a_10 = approx_rcp(d4_8a_10);
            float _fma_211;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_211) : "f"(-d4_8a_10), "f"(r4_8a_10), "f"(1.0f));
            float _fma_210;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_210) : "f"(r4_8a_10), "f"(_fma_211), "f"(r4_8a_10));
            float big_8a_10;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_10) : "f"(mul_f32_rn_noftz(_fma_210, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_212;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_212) : "f"(big_8a_10), "f"(8.388608e+06f));
            float whole_8a_10;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_10) : "f"(_add_212), "f"(8.388608e+06f));
            float above_8a_10;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_10) : "f"(-d4_8a_10), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_10, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_214;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_214) : "f"(big_8a_10), "f"(whole_8a_10));
            float _abs_213;
            asm("abs.f32 %0, %1;" : "=f"(_abs_213) : "f"(_sub_214));
            float _add_215;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_215) : "f"(big_8a_10), "f"(0.5f));
            float _sub_216;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_216) : "f"(big_8a_10), "f"(0.5f));
            float rounded_8a_10 = ((_abs_213 == 0.5f) ? ((above_8a_10 > 0.0f) ? _add_215 : _sub_216) : whole_8a_10);
            t0_9 = ((d0_24 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_10, 7.888609e-31f), 1.7763568e-15f) : t0_9);
            float d4_8a_10_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_10_1) : "f"(d1_24), "f"(0.25f));
            float r4_8a_10_1 = approx_rcp(d4_8a_10_1);
            float _fma_218;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_218) : "f"(-d4_8a_10_1), "f"(r4_8a_10_1), "f"(1.0f));
            float _fma_217;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_217) : "f"(r4_8a_10_1), "f"(_fma_218), "f"(r4_8a_10_1));
            float big_8a_10_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_10_1) : "f"(mul_f32_rn_noftz(_fma_217, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_219;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_219) : "f"(big_8a_10_1), "f"(8.388608e+06f));
            float whole_8a_10_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_10_1) : "f"(_add_219), "f"(8.388608e+06f));
            float above_8a_10_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_10_1) : "f"(-d4_8a_10_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_10_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_221;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_221) : "f"(big_8a_10_1), "f"(whole_8a_10_1));
            float _abs_220;
            asm("abs.f32 %0, %1;" : "=f"(_abs_220) : "f"(_sub_221));
            float _add_222;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_222) : "f"(big_8a_10_1), "f"(0.5f));
            float _sub_223;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_223) : "f"(big_8a_10_1), "f"(0.5f));
            float rounded_8a_10_1 = ((_abs_220 == 0.5f) ? ((above_8a_10_1 > 0.0f) ? _add_222 : _sub_223) : whole_8a_10_1);
            t1_9 = ((d1_24 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_10_1, 7.888609e-31f), 1.7763568e-15f) : t1_9);
        }
        float2 _f32x2_mul_a_224 = mul_f32x2_noftz(cvt_f32_25, make_float2(max_noftz(t0_9, 0.0f), max_noftz(t1_9, 0.0f)));
        float2 mul_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_224), "l"(*(const unsigned long long*)&cvt_f32_1_24));
        packed_hi[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_24.x, mul_24.y)));
        float2 cvt_f32_26 = cake_bf16x2_to_f32x2(__as_bf16x2(words_1[3]));
        float2 cvt_f32_1_25 = cake_bf16x2_to_f32x2(__as_bf16x2(words_2[3]));
        float2 _f32x2_mul_b_225 = make_float2(-1.442695f, -1.442695f);
        float2 t_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_25) : "l"(*(const unsigned long long*)&cvt_f32_26), "l"(*(const unsigned long long*)&_f32x2_mul_b_225));
        float e0_25 = approx_exp2(t_25.x);
        float e1_25 = approx_exp2(t_25.y);
        float2 nd_25 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_25, e1_25));
        float d0_25 = -nd_25.x;
        float d1_25 = -nd_25.y;
        float r0_25 = approx_rcp(d0_25);
        float r1_25 = approx_rcp(d1_25);
        float2 r_25 = make_float2(r0_25, r1_25);
        float2 s_25 = fma_f32x2_rn_noftz(r_25, fma_f32x2_rn_noftz(nd_25, r_25, make_float2(1.0f, 1.0f)), r_25);
        float t0_10 = s_25.x;
        float t1_10 = s_25.y;
        int _warp_vote_226 = __any_sync(0xFFFFFFFF, fminf(t0_10, t1_10) == 0.0f);
        if (_warp_vote_226 != 0) {
            float d4_8a_11;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_11) : "f"(d0_25), "f"(0.25f));
            float r4_8a_11 = approx_rcp(d4_8a_11);
            float _fma_228;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_228) : "f"(-d4_8a_11), "f"(r4_8a_11), "f"(1.0f));
            float _fma_227;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_227) : "f"(r4_8a_11), "f"(_fma_228), "f"(r4_8a_11));
            float big_8a_11;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_11) : "f"(mul_f32_rn_noftz(_fma_227, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_229;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_229) : "f"(big_8a_11), "f"(8.388608e+06f));
            float whole_8a_11;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_11) : "f"(_add_229), "f"(8.388608e+06f));
            float above_8a_11;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_11) : "f"(-d4_8a_11), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_11, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_231;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_231) : "f"(big_8a_11), "f"(whole_8a_11));
            float _abs_230;
            asm("abs.f32 %0, %1;" : "=f"(_abs_230) : "f"(_sub_231));
            float _add_232;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_232) : "f"(big_8a_11), "f"(0.5f));
            float _sub_233;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_233) : "f"(big_8a_11), "f"(0.5f));
            float rounded_8a_11 = ((_abs_230 == 0.5f) ? ((above_8a_11 > 0.0f) ? _add_232 : _sub_233) : whole_8a_11);
            t0_10 = ((d0_25 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_11, 7.888609e-31f), 1.7763568e-15f) : t0_10);
            float d4_8a_11_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_11_1) : "f"(d1_25), "f"(0.25f));
            float r4_8a_11_1 = approx_rcp(d4_8a_11_1);
            float _fma_235;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_235) : "f"(-d4_8a_11_1), "f"(r4_8a_11_1), "f"(1.0f));
            float _fma_234;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_234) : "f"(r4_8a_11_1), "f"(_fma_235), "f"(r4_8a_11_1));
            float big_8a_11_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_11_1) : "f"(mul_f32_rn_noftz(_fma_234, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_236;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_236) : "f"(big_8a_11_1), "f"(8.388608e+06f));
            float whole_8a_11_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_11_1) : "f"(_add_236), "f"(8.388608e+06f));
            float above_8a_11_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_11_1) : "f"(-d4_8a_11_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_11_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_238;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_238) : "f"(big_8a_11_1), "f"(whole_8a_11_1));
            float _abs_237;
            asm("abs.f32 %0, %1;" : "=f"(_abs_237) : "f"(_sub_238));
            float _add_239;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_239) : "f"(big_8a_11_1), "f"(0.5f));
            float _sub_240;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_240) : "f"(big_8a_11_1), "f"(0.5f));
            float rounded_8a_11_1 = ((_abs_237 == 0.5f) ? ((above_8a_11_1 > 0.0f) ? _add_239 : _sub_240) : whole_8a_11_1);
            t1_10 = ((d1_25 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_11_1, 7.888609e-31f), 1.7763568e-15f) : t1_10);
        }
        float2 _f32x2_mul_a_241 = mul_f32x2_noftz(cvt_f32_26, make_float2(max_noftz(t0_10, 0.0f), max_noftz(t1_10, 0.0f)));
        float2 mul_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_241), "l"(*(const unsigned long long*)&cvt_f32_1_25));
        packed_hi[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_25.x, mul_25.y)));
        float2 cvt_f32_27 = cake_bf16x2_to_f32x2(__as_bf16x2(words_1[4]));
        float2 cvt_f32_1_26 = cake_bf16x2_to_f32x2(__as_bf16x2(words_2[4]));
        float2 _f32x2_mul_b_242 = make_float2(-1.442695f, -1.442695f);
        float2 t_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_26) : "l"(*(const unsigned long long*)&cvt_f32_27), "l"(*(const unsigned long long*)&_f32x2_mul_b_242));
        float e0_26 = approx_exp2(t_26.x);
        float e1_26 = approx_exp2(t_26.y);
        float2 nd_26 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_26, e1_26));
        float d0_26 = -nd_26.x;
        float d1_26 = -nd_26.y;
        float r0_26 = approx_rcp(d0_26);
        float r1_26 = approx_rcp(d1_26);
        float2 r_26 = make_float2(r0_26, r1_26);
        float2 s_26 = fma_f32x2_rn_noftz(r_26, fma_f32x2_rn_noftz(nd_26, r_26, make_float2(1.0f, 1.0f)), r_26);
        float t0_11 = s_26.x;
        float t1_11 = s_26.y;
        int _warp_vote_243 = __any_sync(0xFFFFFFFF, fminf(t0_11, t1_11) == 0.0f);
        if (_warp_vote_243 != 0) {
            float d4_8a_12;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_12) : "f"(d0_26), "f"(0.25f));
            float r4_8a_12 = approx_rcp(d4_8a_12);
            float _fma_245;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_245) : "f"(-d4_8a_12), "f"(r4_8a_12), "f"(1.0f));
            float _fma_244;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_244) : "f"(r4_8a_12), "f"(_fma_245), "f"(r4_8a_12));
            float big_8a_12;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_12) : "f"(mul_f32_rn_noftz(_fma_244, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_246;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_246) : "f"(big_8a_12), "f"(8.388608e+06f));
            float whole_8a_12;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_12) : "f"(_add_246), "f"(8.388608e+06f));
            float above_8a_12;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_12) : "f"(-d4_8a_12), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_12, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_248;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_248) : "f"(big_8a_12), "f"(whole_8a_12));
            float _abs_247;
            asm("abs.f32 %0, %1;" : "=f"(_abs_247) : "f"(_sub_248));
            float _add_249;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_249) : "f"(big_8a_12), "f"(0.5f));
            float _sub_250;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_250) : "f"(big_8a_12), "f"(0.5f));
            float rounded_8a_12 = ((_abs_247 == 0.5f) ? ((above_8a_12 > 0.0f) ? _add_249 : _sub_250) : whole_8a_12);
            t0_11 = ((d0_26 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_12, 7.888609e-31f), 1.7763568e-15f) : t0_11);
            float d4_8a_12_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_12_1) : "f"(d1_26), "f"(0.25f));
            float r4_8a_12_1 = approx_rcp(d4_8a_12_1);
            float _fma_252;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_252) : "f"(-d4_8a_12_1), "f"(r4_8a_12_1), "f"(1.0f));
            float _fma_251;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_251) : "f"(r4_8a_12_1), "f"(_fma_252), "f"(r4_8a_12_1));
            float big_8a_12_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_12_1) : "f"(mul_f32_rn_noftz(_fma_251, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_253;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_253) : "f"(big_8a_12_1), "f"(8.388608e+06f));
            float whole_8a_12_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_12_1) : "f"(_add_253), "f"(8.388608e+06f));
            float above_8a_12_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_12_1) : "f"(-d4_8a_12_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_12_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_255;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_255) : "f"(big_8a_12_1), "f"(whole_8a_12_1));
            float _abs_254;
            asm("abs.f32 %0, %1;" : "=f"(_abs_254) : "f"(_sub_255));
            float _add_256;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_256) : "f"(big_8a_12_1), "f"(0.5f));
            float _sub_257;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_257) : "f"(big_8a_12_1), "f"(0.5f));
            float rounded_8a_12_1 = ((_abs_254 == 0.5f) ? ((above_8a_12_1 > 0.0f) ? _add_256 : _sub_257) : whole_8a_12_1);
            t1_11 = ((d1_26 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_12_1, 7.888609e-31f), 1.7763568e-15f) : t1_11);
        }
        float2 _f32x2_mul_a_258 = mul_f32x2_noftz(cvt_f32_27, make_float2(max_noftz(t0_11, 0.0f), max_noftz(t1_11, 0.0f)));
        float2 mul_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_258), "l"(*(const unsigned long long*)&cvt_f32_1_26));
        packed_hi[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_26.x, mul_26.y)));
        float2 cvt_f32_28 = cake_bf16x2_to_f32x2(__as_bf16x2(words_1[5]));
        float2 cvt_f32_1_27 = cake_bf16x2_to_f32x2(__as_bf16x2(words_2[5]));
        float2 _f32x2_mul_b_259 = make_float2(-1.442695f, -1.442695f);
        float2 t_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_27) : "l"(*(const unsigned long long*)&cvt_f32_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_259));
        float e0_27 = approx_exp2(t_27.x);
        float e1_27 = approx_exp2(t_27.y);
        float2 nd_27 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_27, e1_27));
        float d0_27 = -nd_27.x;
        float d1_27 = -nd_27.y;
        float r0_27 = approx_rcp(d0_27);
        float r1_27 = approx_rcp(d1_27);
        float2 r_27 = make_float2(r0_27, r1_27);
        float2 s_27 = fma_f32x2_rn_noftz(r_27, fma_f32x2_rn_noftz(nd_27, r_27, make_float2(1.0f, 1.0f)), r_27);
        float t0_12 = s_27.x;
        float t1_12 = s_27.y;
        int _warp_vote_260 = __any_sync(0xFFFFFFFF, fminf(t0_12, t1_12) == 0.0f);
        if (_warp_vote_260 != 0) {
            float d4_8a_13;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_13) : "f"(d0_27), "f"(0.25f));
            float r4_8a_13 = approx_rcp(d4_8a_13);
            float _fma_262;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_262) : "f"(-d4_8a_13), "f"(r4_8a_13), "f"(1.0f));
            float _fma_261;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_261) : "f"(r4_8a_13), "f"(_fma_262), "f"(r4_8a_13));
            float big_8a_13;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_13) : "f"(mul_f32_rn_noftz(_fma_261, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_263;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_263) : "f"(big_8a_13), "f"(8.388608e+06f));
            float whole_8a_13;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_13) : "f"(_add_263), "f"(8.388608e+06f));
            float above_8a_13;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_13) : "f"(-d4_8a_13), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_13, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_265;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_265) : "f"(big_8a_13), "f"(whole_8a_13));
            float _abs_264;
            asm("abs.f32 %0, %1;" : "=f"(_abs_264) : "f"(_sub_265));
            float _add_266;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_266) : "f"(big_8a_13), "f"(0.5f));
            float _sub_267;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_267) : "f"(big_8a_13), "f"(0.5f));
            float rounded_8a_13 = ((_abs_264 == 0.5f) ? ((above_8a_13 > 0.0f) ? _add_266 : _sub_267) : whole_8a_13);
            t0_12 = ((d0_27 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_13, 7.888609e-31f), 1.7763568e-15f) : t0_12);
            float d4_8a_13_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_13_1) : "f"(d1_27), "f"(0.25f));
            float r4_8a_13_1 = approx_rcp(d4_8a_13_1);
            float _fma_269;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_269) : "f"(-d4_8a_13_1), "f"(r4_8a_13_1), "f"(1.0f));
            float _fma_268;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_268) : "f"(r4_8a_13_1), "f"(_fma_269), "f"(r4_8a_13_1));
            float big_8a_13_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_13_1) : "f"(mul_f32_rn_noftz(_fma_268, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_270;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_270) : "f"(big_8a_13_1), "f"(8.388608e+06f));
            float whole_8a_13_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_13_1) : "f"(_add_270), "f"(8.388608e+06f));
            float above_8a_13_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_13_1) : "f"(-d4_8a_13_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_13_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_272;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_272) : "f"(big_8a_13_1), "f"(whole_8a_13_1));
            float _abs_271;
            asm("abs.f32 %0, %1;" : "=f"(_abs_271) : "f"(_sub_272));
            float _add_273;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_273) : "f"(big_8a_13_1), "f"(0.5f));
            float _sub_274;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_274) : "f"(big_8a_13_1), "f"(0.5f));
            float rounded_8a_13_1 = ((_abs_271 == 0.5f) ? ((above_8a_13_1 > 0.0f) ? _add_273 : _sub_274) : whole_8a_13_1);
            t1_12 = ((d1_27 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_13_1, 7.888609e-31f), 1.7763568e-15f) : t1_12);
        }
        float2 _f32x2_mul_a_275 = mul_f32x2_noftz(cvt_f32_28, make_float2(max_noftz(t0_12, 0.0f), max_noftz(t1_12, 0.0f)));
        float2 mul_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_275), "l"(*(const unsigned long long*)&cvt_f32_1_27));
        packed_hi[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_27.x, mul_27.y)));
        float2 cvt_f32_29 = cake_bf16x2_to_f32x2(__as_bf16x2(words_1[6]));
        float2 cvt_f32_1_28 = cake_bf16x2_to_f32x2(__as_bf16x2(words_2[6]));
        float2 _f32x2_mul_b_276 = make_float2(-1.442695f, -1.442695f);
        float2 t_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_28) : "l"(*(const unsigned long long*)&cvt_f32_29), "l"(*(const unsigned long long*)&_f32x2_mul_b_276));
        float e0_28 = approx_exp2(t_28.x);
        float e1_28 = approx_exp2(t_28.y);
        float2 nd_28 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_28, e1_28));
        float d0_28 = -nd_28.x;
        float d1_28 = -nd_28.y;
        float r0_28 = approx_rcp(d0_28);
        float r1_28 = approx_rcp(d1_28);
        float2 r_28 = make_float2(r0_28, r1_28);
        float2 s_28 = fma_f32x2_rn_noftz(r_28, fma_f32x2_rn_noftz(nd_28, r_28, make_float2(1.0f, 1.0f)), r_28);
        float t0_13 = s_28.x;
        float t1_13 = s_28.y;
        int _warp_vote_277 = __any_sync(0xFFFFFFFF, fminf(t0_13, t1_13) == 0.0f);
        if (_warp_vote_277 != 0) {
            float d4_8a_14;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_14) : "f"(d0_28), "f"(0.25f));
            float r4_8a_14 = approx_rcp(d4_8a_14);
            float _fma_279;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_279) : "f"(-d4_8a_14), "f"(r4_8a_14), "f"(1.0f));
            float _fma_278;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_278) : "f"(r4_8a_14), "f"(_fma_279), "f"(r4_8a_14));
            float big_8a_14;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_14) : "f"(mul_f32_rn_noftz(_fma_278, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_280;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_280) : "f"(big_8a_14), "f"(8.388608e+06f));
            float whole_8a_14;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_14) : "f"(_add_280), "f"(8.388608e+06f));
            float above_8a_14;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_14) : "f"(-d4_8a_14), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_14, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_282;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_282) : "f"(big_8a_14), "f"(whole_8a_14));
            float _abs_281;
            asm("abs.f32 %0, %1;" : "=f"(_abs_281) : "f"(_sub_282));
            float _add_283;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_283) : "f"(big_8a_14), "f"(0.5f));
            float _sub_284;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_284) : "f"(big_8a_14), "f"(0.5f));
            float rounded_8a_14 = ((_abs_281 == 0.5f) ? ((above_8a_14 > 0.0f) ? _add_283 : _sub_284) : whole_8a_14);
            t0_13 = ((d0_28 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_14, 7.888609e-31f), 1.7763568e-15f) : t0_13);
            float d4_8a_14_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_14_1) : "f"(d1_28), "f"(0.25f));
            float r4_8a_14_1 = approx_rcp(d4_8a_14_1);
            float _fma_286;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_286) : "f"(-d4_8a_14_1), "f"(r4_8a_14_1), "f"(1.0f));
            float _fma_285;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_285) : "f"(r4_8a_14_1), "f"(_fma_286), "f"(r4_8a_14_1));
            float big_8a_14_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_14_1) : "f"(mul_f32_rn_noftz(_fma_285, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_287;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_287) : "f"(big_8a_14_1), "f"(8.388608e+06f));
            float whole_8a_14_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_14_1) : "f"(_add_287), "f"(8.388608e+06f));
            float above_8a_14_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_14_1) : "f"(-d4_8a_14_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_14_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_289;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_289) : "f"(big_8a_14_1), "f"(whole_8a_14_1));
            float _abs_288;
            asm("abs.f32 %0, %1;" : "=f"(_abs_288) : "f"(_sub_289));
            float _add_290;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_290) : "f"(big_8a_14_1), "f"(0.5f));
            float _sub_291;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_291) : "f"(big_8a_14_1), "f"(0.5f));
            float rounded_8a_14_1 = ((_abs_288 == 0.5f) ? ((above_8a_14_1 > 0.0f) ? _add_290 : _sub_291) : whole_8a_14_1);
            t1_13 = ((d1_28 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_14_1, 7.888609e-31f), 1.7763568e-15f) : t1_13);
        }
        float2 _f32x2_mul_a_292 = mul_f32x2_noftz(cvt_f32_29, make_float2(max_noftz(t0_13, 0.0f), max_noftz(t1_13, 0.0f)));
        float2 mul_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_292), "l"(*(const unsigned long long*)&cvt_f32_1_28));
        packed_hi[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_28.x, mul_28.y)));
        float2 cvt_f32_30 = cake_bf16x2_to_f32x2(__as_bf16x2(words_1[7]));
        float2 cvt_f32_1_29 = cake_bf16x2_to_f32x2(__as_bf16x2(words_2[7]));
        float2 _f32x2_mul_b_293 = make_float2(-1.442695f, -1.442695f);
        float2 t_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_29) : "l"(*(const unsigned long long*)&cvt_f32_30), "l"(*(const unsigned long long*)&_f32x2_mul_b_293));
        float e0_29 = approx_exp2(t_29.x);
        float e1_29 = approx_exp2(t_29.y);
        float2 nd_29 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_29, e1_29));
        float d0_29 = -nd_29.x;
        float d1_29 = -nd_29.y;
        float r0_29 = approx_rcp(d0_29);
        float r1_29 = approx_rcp(d1_29);
        float2 r_29 = make_float2(r0_29, r1_29);
        float2 s_29 = fma_f32x2_rn_noftz(r_29, fma_f32x2_rn_noftz(nd_29, r_29, make_float2(1.0f, 1.0f)), r_29);
        float t0_14 = s_29.x;
        float t1_14 = s_29.y;
        int _warp_vote_294 = __any_sync(0xFFFFFFFF, fminf(t0_14, t1_14) == 0.0f);
        if (_warp_vote_294 != 0) {
            float d4_8a_15;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_15) : "f"(d0_29), "f"(0.25f));
            float r4_8a_15 = approx_rcp(d4_8a_15);
            float _fma_296;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_296) : "f"(-d4_8a_15), "f"(r4_8a_15), "f"(1.0f));
            float _fma_295;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_295) : "f"(r4_8a_15), "f"(_fma_296), "f"(r4_8a_15));
            float big_8a_15;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_15) : "f"(mul_f32_rn_noftz(_fma_295, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_297;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_297) : "f"(big_8a_15), "f"(8.388608e+06f));
            float whole_8a_15;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_15) : "f"(_add_297), "f"(8.388608e+06f));
            float above_8a_15;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_15) : "f"(-d4_8a_15), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_15, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_299;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_299) : "f"(big_8a_15), "f"(whole_8a_15));
            float _abs_298;
            asm("abs.f32 %0, %1;" : "=f"(_abs_298) : "f"(_sub_299));
            float _add_300;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_300) : "f"(big_8a_15), "f"(0.5f));
            float _sub_301;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_301) : "f"(big_8a_15), "f"(0.5f));
            float rounded_8a_15 = ((_abs_298 == 0.5f) ? ((above_8a_15 > 0.0f) ? _add_300 : _sub_301) : whole_8a_15);
            t0_14 = ((d0_29 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_15, 7.888609e-31f), 1.7763568e-15f) : t0_14);
            float d4_8a_15_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_15_1) : "f"(d1_29), "f"(0.25f));
            float r4_8a_15_1 = approx_rcp(d4_8a_15_1);
            float _fma_303;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_303) : "f"(-d4_8a_15_1), "f"(r4_8a_15_1), "f"(1.0f));
            float _fma_302;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_302) : "f"(r4_8a_15_1), "f"(_fma_303), "f"(r4_8a_15_1));
            float big_8a_15_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_15_1) : "f"(mul_f32_rn_noftz(_fma_302, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_304;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_304) : "f"(big_8a_15_1), "f"(8.388608e+06f));
            float whole_8a_15_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_15_1) : "f"(_add_304), "f"(8.388608e+06f));
            float above_8a_15_1;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_15_1) : "f"(-d4_8a_15_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_15_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_306;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_306) : "f"(big_8a_15_1), "f"(whole_8a_15_1));
            float _abs_305;
            asm("abs.f32 %0, %1;" : "=f"(_abs_305) : "f"(_sub_306));
            float _add_307;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_307) : "f"(big_8a_15_1), "f"(0.5f));
            float _sub_308;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_308) : "f"(big_8a_15_1), "f"(0.5f));
            float rounded_8a_15_1 = ((_abs_305 == 0.5f) ? ((above_8a_15_1 > 0.0f) ? _add_307 : _sub_308) : whole_8a_15_1);
            t1_14 = ((d1_29 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_15_1, 7.888609e-31f), 1.7763568e-15f) : t1_14);
        }
        float2 _f32x2_mul_a_309 = mul_f32x2_noftz(cvt_f32_30, make_float2(max_noftz(t0_14, 0.0f), max_noftz(t1_14, 0.0f)));
        float2 mul_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_309), "l"(*(const unsigned long long*)&cvt_f32_1_29));
        packed_hi[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_29.x, mul_29.y)));
    }
    unsigned int swizzle = (unsigned int)(row >> 3 & 6 | row & 1);
    unsigned int row_base = tile_addr + (unsigned int)(row * 256);
    unsigned int chunk = (unsigned int)((tid & 7) * 2);
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + ((chunk ^ swizzle) << 4)), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + ((chunk + 1 ^ swizzle) << 4)), "r"(packed[4]), "r"(packed[5]), "r"(packed[6]), "r"(packed[7]) : "memory");
    unsigned int swizzle_hi = swizzle ^ 4;
    unsigned int row_base_hi = row_base + 8192;
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base_hi + ((chunk ^ swizzle_hi) << 4)), "r"(packed_hi[0]), "r"(packed_hi[1]), "r"(packed_hi[2]), "r"(packed_hi[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base_hi + ((chunk + 1 ^ swizzle_hi) << 4)), "r"(packed_hi[4]), "r"(packed_hi[5]), "r"(packed_hi[6]), "r"(packed_hi[7]) : "memory");
    __syncthreads();
    int lane_1 = tid & 31;
    int warp_1 = tid >> 5;
    int k8 = lane_1 & 7;
    int srow = (k8 >> 1) * 16 + (lane_1 >> 3) * 2 + (k8 & 1);
    unsigned int sswizzle = (unsigned int)(srow >> 3 & 6 | srow & 1);
    unsigned int cols[16];
    unsigned int strip_chunk0_1_0 = (unsigned int)(warp_1 * 2);
    unsigned int strip_src0_1_0 = tile_addr + (unsigned int)(srow * 256) + ((strip_chunk0_1_0 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[0]), "=r"(cols[1]), "=r"(cols[2]), "=r"(cols[3])
        : "r"(strip_src0_1_0)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[4]), "=r"(cols[5]), "=r"(cols[6]), "=r"(cols[7])
        : "r"(strip_src0_1_0 + 2048)
        : "memory");
    unsigned int strip_chunk0_1_1 = (unsigned int)(warp_1 * 2 + 1);
    unsigned int strip_src0_1_1 = tile_addr + (unsigned int)(srow * 256) + ((strip_chunk0_1_1 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[8]), "=r"(cols[9]), "=r"(cols[10]), "=r"(cols[11])
        : "r"(strip_src0_1_1)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[12]), "=r"(cols[13]), "=r"(cols[14]), "=r"(cols[15])
        : "r"(strip_src0_1_1 + 2048)
        : "memory");
    __nv_bfloat162 amax_a = __as_bf16x2(0);
    __nv_bfloat162 amax_b = __as_bf16x2(0);
    __nv_bfloat162 _hxa_311 = amax_a;
    __nv_bfloat162 _hxb_311 = __as_bf16x2(cols[0]);
    unsigned int _hxr_311;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_311) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_311)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_311)));
    __nv_bfloat162 _max_310 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_311);
    amax_a = _max_310;
    __nv_bfloat162 _hxa_313 = amax_b;
    __nv_bfloat162 _hxb_313 = __as_bf16x2(cols[8]);
    unsigned int _hxr_313;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_313) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_313)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_313)));
    __nv_bfloat162 _max_312 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_313);
    amax_b = _max_312;
    __nv_bfloat162 _hxa_315 = amax_a;
    __nv_bfloat162 _hxb_315 = __as_bf16x2(cols[1]);
    unsigned int _hxr_315;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_315) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_315)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_315)));
    __nv_bfloat162 _max_314 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_315);
    amax_a = _max_314;
    __nv_bfloat162 _hxa_317 = amax_b;
    __nv_bfloat162 _hxb_317 = __as_bf16x2(cols[9]);
    unsigned int _hxr_317;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_317) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_317)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_317)));
    __nv_bfloat162 _max_316 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_317);
    amax_b = _max_316;
    __nv_bfloat162 _hxa_319 = amax_a;
    __nv_bfloat162 _hxb_319 = __as_bf16x2(cols[2]);
    unsigned int _hxr_319;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_319) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_319)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_319)));
    __nv_bfloat162 _max_318 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_319);
    amax_a = _max_318;
    __nv_bfloat162 _hxa_321 = amax_b;
    __nv_bfloat162 _hxb_321 = __as_bf16x2(cols[10]);
    unsigned int _hxr_321;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_321) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_321)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_321)));
    __nv_bfloat162 _max_320 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_321);
    amax_b = _max_320;
    __nv_bfloat162 _hxa_323 = amax_a;
    __nv_bfloat162 _hxb_323 = __as_bf16x2(cols[3]);
    unsigned int _hxr_323;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_323) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_323)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_323)));
    __nv_bfloat162 _max_322 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_323);
    amax_a = _max_322;
    __nv_bfloat162 _hxa_325 = amax_b;
    __nv_bfloat162 _hxb_325 = __as_bf16x2(cols[11]);
    unsigned int _hxr_325;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_325) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_325)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_325)));
    __nv_bfloat162 _max_324 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_325);
    amax_b = _max_324;
    __nv_bfloat162 _hxa_327 = amax_a;
    __nv_bfloat162 _hxb_327 = __as_bf16x2(cols[4]);
    unsigned int _hxr_327;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_327) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_327)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_327)));
    __nv_bfloat162 _max_326 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_327);
    amax_a = _max_326;
    __nv_bfloat162 _hxa_329 = amax_b;
    __nv_bfloat162 _hxb_329 = __as_bf16x2(cols[12]);
    unsigned int _hxr_329;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_329) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_329)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_329)));
    __nv_bfloat162 _max_328 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_329);
    amax_b = _max_328;
    __nv_bfloat162 _hxa_331 = amax_a;
    __nv_bfloat162 _hxb_331 = __as_bf16x2(cols[5]);
    unsigned int _hxr_331;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_331) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_331)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_331)));
    __nv_bfloat162 _max_330 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_331);
    amax_a = _max_330;
    __nv_bfloat162 _hxa_333 = amax_b;
    __nv_bfloat162 _hxb_333 = __as_bf16x2(cols[13]);
    unsigned int _hxr_333;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_333) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_333)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_333)));
    __nv_bfloat162 _max_332 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_333);
    amax_b = _max_332;
    __nv_bfloat162 _hxa_335 = amax_a;
    __nv_bfloat162 _hxb_335 = __as_bf16x2(cols[6]);
    unsigned int _hxr_335;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_335) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_335)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_335)));
    __nv_bfloat162 _max_334 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_335);
    amax_a = _max_334;
    __nv_bfloat162 _hxa_337 = amax_b;
    __nv_bfloat162 _hxb_337 = __as_bf16x2(cols[14]);
    unsigned int _hxr_337;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_337) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_337)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_337)));
    __nv_bfloat162 _max_336 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_337);
    amax_b = _max_336;
    __nv_bfloat162 _hxa_339 = amax_a;
    __nv_bfloat162 _hxb_339 = __as_bf16x2(cols[7]);
    unsigned int _hxr_339;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_339) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_339)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_339)));
    __nv_bfloat162 _max_338 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_339);
    amax_a = _max_338;
    __nv_bfloat162 _hxa_341 = amax_b;
    __nv_bfloat162 _hxb_341 = __as_bf16x2(cols[15]);
    unsigned int _hxr_341;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_341) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_341)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_341)));
    __nv_bfloat162 _max_340 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_341);
    amax_b = _max_340;
    unsigned int mag_a = __as_u32(amax_a) & 2147450879;
    unsigned int mag_b = __as_u32(amax_b) & 2147450879;
    unsigned int lows_ab = mag_a & 65535 | mag_b << 16;
    unsigned int highs_ab = mag_a >> 16 | mag_b & 4294901760u;
    __nv_bfloat162 col_max = __hmax2_nan(__as_bf16x2(lows_ab), __as_bf16x2(highs_ab));
    unsigned int peer1 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(col_max), 1);
    col_max = __hmax2_nan(col_max, __as_bf16x2(peer1));
    unsigned int col_max_bits = __as_u32(col_max);
    unsigned int amax_bits = (col_max_bits & 65535) << 16;
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
    unsigned int amax_bits_0 = (col_max_bits >> 16 & 65535) << 16;
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
    int gcol_a = bx * 128 + warp_1 * 16 + (lane_1 >> 2);
    int gcol_b = gcol_a + 8;
    int out_row = by * 64 + (lane_1 & 3) * 16;
    unsigned int words_a[4];
    unsigned int words_b[4];
    uint32_t bf16x2_mul;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul) : "r"(cols[0]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1) : "r"(cols[1]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_343;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_343) : "r"(bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_343) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_342;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_342) : "h"(e4m3x2_bf16x2), "h"(_cvt_satfinite_e4m3x2_bf16x2_343));
    words_a[0] = _mov_b32_pack_u16x2_342;
    uint32_t bf16x2_mul_0;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_0) : "r"(cols[8]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_0) : "r"(cols[9]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_345;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_345) : "r"(bf16x2_mul_1_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_345) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_344;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_344) : "h"(e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_345));
    words_b[0] = _mov_b32_pack_u16x2_344;
    uint32_t bf16x2_mul_2;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2) : "r"(cols[2]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_1) : "r"(cols[3]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_347;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_347) : "r"(bf16x2_mul_1_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_347) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_346;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_346) : "h"(e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_347));
    words_a[1] = _mov_b32_pack_u16x2_346;
    uint32_t bf16x2_mul_3;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3) : "r"(cols[10]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_2) : "r"(cols[11]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_349;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_349) : "r"(bf16x2_mul_1_2));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_2);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_349) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_348;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_348) : "h"(e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_349));
    words_b[1] = _mov_b32_pack_u16x2_348;
    uint32_t bf16x2_mul_4;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4) : "r"(cols[4]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_3) : "r"(cols[5]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_351;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_351) : "r"(bf16x2_mul_1_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_351) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_350;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_350) : "h"(e4m3x2_bf16x2_3), "h"(_cvt_satfinite_e4m3x2_bf16x2_351));
    words_a[2] = _mov_b32_pack_u16x2_350;
    uint32_t bf16x2_mul_5;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5) : "r"(cols[12]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_4) : "r"(cols[13]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_353;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_353) : "r"(bf16x2_mul_1_4));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_4);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_353) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_352;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_352) : "h"(e4m3x2_bf16x2_4), "h"(_cvt_satfinite_e4m3x2_bf16x2_353));
    words_b[2] = _mov_b32_pack_u16x2_352;
    uint32_t bf16x2_mul_6;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6) : "r"(cols[6]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_5) : "r"(cols[7]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_355;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_355) : "r"(bf16x2_mul_1_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_355) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_354;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_354) : "h"(e4m3x2_bf16x2_5), "h"(_cvt_satfinite_e4m3x2_bf16x2_355));
    words_a[3] = _mov_b32_pack_u16x2_354;
    uint32_t bf16x2_mul_7;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7) : "r"(cols[14]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_6) : "r"(cols[15]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_357;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_357) : "r"(bf16x2_mul_1_6));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_6);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_357) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_356;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_356) : "h"(e4m3x2_bf16x2_6), "h"(_cvt_satfinite_e4m3x2_bf16x2_357));
    words_b[3] = _mov_b32_pack_u16x2_356;
    reinterpret_cast<int4*>(col_output + (gcol_a * M + out_row))[0] = reinterpret_cast<int4*>(words_a + 0)[0];
    reinterpret_cast<int4*>(col_output + (gcol_b * M + out_row))[0] = reinterpret_cast<int4*>(words_b + 0)[0];
    if ((lane_1 & 1) == 0) {
        int row_tiles = M >> 7;
        int rblk = by * 2 + ((lane_1 & 3) >> 1);
        int col_scale_a = ((gcol_a >> 7) * row_tiles + (rblk >> 2)) * 512 + (gcol_a & 31) * 16 + (gcol_a >> 5 & 3) * 4 + (rblk & 3);
        int col_scale_b = ((gcol_b >> 7) * row_tiles + (rblk >> 2)) * 512 + (gcol_b & 31) * 16 + (gcol_b >> 5 & 3) * 4 + (rblk & 3);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_a) + (0)) = (unsigned char)(scale);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b) + (0)) = (unsigned char)(scale_0);
    }
}

} // extern "C"
