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
kernel_cake_gated_act_mxfp8_d1729b9c66e203234b04(__nv_bfloat16* __restrict__ gated_input, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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
        float2 cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(words[1]));
        float2 cvt_f32_1_0 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[1]));
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
        float2 cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(words[2]));
        float2 cvt_f32_1_1 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[2]));
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
        float2 cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(words[3]));
        float2 cvt_f32_1_2 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[3]));
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
    } else {
        int band = 0;
        float2 cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(words[0]));
        float2 cvt_f32_1_3 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[0]));
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
        float t0 = s_3.x;
        float t1 = s_3.y;
        band = band | ((fminf(t0, t1) == 0.0f) ? 1 : 0);
        float2 _f32x2_mul_a_13 = mul_f32x2_noftz(cvt_f32_4, make_float2(max_noftz(t0, 0.0f), max_noftz(t1, 0.0f)));
        float2 mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_13), "l"(*(const unsigned long long*)&cvt_f32_1_3));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_3.x, mul_3.y)));
        float2 cvt_f32_4_1 = cake_bf16x2_to_f32x2(__as_bf16x2(words[1]));
        float2 cvt_f32_1_3_1 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[1]));
        float2 _f32x2_mul_b_14 = make_float2(-1.442695f, -1.442695f);
        float2 t_3_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_3_1) : "l"(*(const unsigned long long*)&cvt_f32_4_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_14));
        float e0_3_1 = approx_exp2(t_3_1.x);
        float e1_3_1 = approx_exp2(t_3_1.y);
        float2 nd_3_1 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_3_1, e1_3_1));
        float d0_3_1 = -nd_3_1.x;
        float d1_3_1 = -nd_3_1.y;
        float r0_3_1 = approx_rcp(d0_3_1);
        float r1_3_1 = approx_rcp(d1_3_1);
        float2 r_3_1 = make_float2(r0_3_1, r1_3_1);
        float2 s_3_1 = fma_f32x2_rn_noftz(r_3_1, fma_f32x2_rn_noftz(nd_3_1, r_3_1, make_float2(1.0f, 1.0f)), r_3_1);
        float t0_0 = s_3_1.x;
        float t1_0 = s_3_1.y;
        band = band | ((fminf(t0_0, t1_0) == 0.0f) ? 2 : 0);
        float2 _f32x2_mul_a_15 = mul_f32x2_noftz(cvt_f32_4_1, make_float2(max_noftz(t0_0, 0.0f), max_noftz(t1_0, 0.0f)));
        float2 mul_3_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&cvt_f32_1_3_1));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_3_1.x, mul_3_1.y)));
        float2 cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(words[2]));
        float2 cvt_f32_1_4 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[2]));
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
        float t0_1 = s_4.x;
        float t1_1 = s_4.y;
        band = band | ((fminf(t0_1, t1_1) == 0.0f) ? 4 : 0);
        float2 _f32x2_mul_a_17 = mul_f32x2_noftz(cvt_f32_5, make_float2(max_noftz(t0_1, 0.0f), max_noftz(t1_1, 0.0f)));
        float2 mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&cvt_f32_1_4));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_4.x, mul_4.y)));
        float2 cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(words[3]));
        float2 cvt_f32_1_5 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[3]));
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
        float t0_2 = s_5.x;
        float t1_2 = s_5.y;
        band = band | ((fminf(t0_2, t1_2) == 0.0f) ? 8 : 0);
        float2 _f32x2_mul_a_19 = mul_f32x2_noftz(cvt_f32_6, make_float2(max_noftz(t0_2, 0.0f), max_noftz(t1_2, 0.0f)));
        float2 mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&cvt_f32_1_5));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_5.x, mul_5.y)));
        if (band != 0) {
            unsigned int popc = (unsigned int)__popc(band);
            #pragma unroll 1
            for (int _j = 0; _j < popc; _j++) {
                int _find_first_set_20 = __ffs(band);
                int j = _find_first_set_20 - 1;
                band = band & band - 1;
                unsigned int vec_load[1];
                {
                    vec_load[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index + 2 * j) + 0);
                }
                unsigned int vec_load_1[1];
                {
                    vec_load_1[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index + K + 2 * j) + 0);
                }
                float2 cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[0]));
                float2 cvt_f32_1_6 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[0]));
                float2 _f32x2_mul_b_21 = make_float2(-1.442695f, -1.442695f);
                float2 t_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_6) : "l"(*(const unsigned long long*)&cvt_f32_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_21));
                float e0_6 = approx_exp2(t_6.x);
                float e1_6 = approx_exp2(t_6.y);
                float2 nd_6 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_6, e1_6));
                float d0_6 = -nd_6.x;
                float d1_6 = -nd_6.y;
                float r0_6 = approx_rcp(d0_6);
                float r1_6 = approx_rcp(d1_6);
                float2 r_6 = make_float2(r0_6, r1_6);
                float2 s_6 = fma_f32x2_rn_noftz(r_6, fma_f32x2_rn_noftz(nd_6, r_6, make_float2(1.0f, 1.0f)), r_6);
                float t0_3 = s_6.x;
                float t1_3 = s_6.y;
                float d4_8a;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a) : "f"(d0_6), "f"(0.25f));
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
                float u0 = ((d0_6 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a, 7.888609e-31f), 1.7763568e-15f) : t0_3);
                float d4_8a_0;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_0) : "f"(d1_6), "f"(0.25f));
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
                float u1 = ((d1_6 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_0, 7.888609e-31f), 1.7763568e-15f) : t1_3);
                float2 _f32x2_mul_a_36 = mul_f32x2_noftz(cvt_f32_7, make_float2(max_noftz(u0, 0.0f), max_noftz(u1, 0.0f)));
                float2 mul_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_36), "l"(*(const unsigned long long*)&cvt_f32_1_6));
                unsigned int pair = __as_u32(__float22bfloat162_rn(make_float2(mul_6.x, mul_6.y)));
                packed[0] = ((j == 0) ? pair : packed[0]);
                packed[1] = ((j == 1) ? pair : packed[1]);
                packed[2] = ((j == 2) ? pair : packed[2]);
                packed[3] = ((j == 3) ? pair : packed[3]);
            }
        }
    }
    unsigned int nswizzle = (unsigned int)(row >> 3 & 3);
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(tile_addr + (unsigned int)(row * 64) + (((unsigned int)quarter ^ nswizzle) << 4)), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
    __nv_bfloat162 amax = __as_bf16x2(0);
    __nv_bfloat162 _hxa_38 = amax;
    __nv_bfloat162 _hxb_38 = __as_bf16x2(packed[0]);
    unsigned int _hxr_38;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_38) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_38)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_38)));
    __nv_bfloat162 _max_37 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_38);
    amax = _max_37;
    __nv_bfloat162 _hxa_40 = amax;
    __nv_bfloat162 _hxb_40 = __as_bf16x2(packed[1]);
    unsigned int _hxr_40;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_40) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_40)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_40)));
    __nv_bfloat162 _max_39 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_40);
    amax = _max_39;
    __nv_bfloat162 _hxa_42 = amax;
    __nv_bfloat162 _hxb_42 = __as_bf16x2(packed[2]);
    unsigned int _hxr_42;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_42) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_42)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_42)));
    __nv_bfloat162 _max_41 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_42);
    amax = _max_41;
    __nv_bfloat162 _hxa_44 = amax;
    __nv_bfloat162 _hxb_44 = __as_bf16x2(packed[3]);
    unsigned int _hxr_44;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_44) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_44)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_44)));
    __nv_bfloat162 _max_43 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_44);
    amax = _max_43;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_46;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_46) : "r"(bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_46) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_45;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_45) : "h"(e4m3x2_bf16x2), "h"(_cvt_satfinite_e4m3x2_bf16x2_46));
    row_words[0] = _mov_b32_pack_u16x2_45;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_48;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_48) : "r"(bf16x2_mul_1_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_48) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_47;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_47) : "h"(e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_48));
    row_words[1] = _mov_b32_pack_u16x2_47;
    reinterpret_cast<int2*>(row_output + (grow * K + col))[0] = reinterpret_cast<int2*>(row_words + 0)[0];
    __syncthreads();
    int lane_1 = tid & 31;
    int warp_1 = tid >> 5;
    int k8 = lane_1 & 7;
    int srow = (k8 >> 1) * 8 + (lane_1 >> 3) * 2 + (k8 & 1);
    unsigned int sswizzle = (unsigned int)(srow >> 3 & 3);
    unsigned int cols[4];
    unsigned int src = tile_addr + (unsigned int)(srow * 64) + (((unsigned int)warp_1 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[0]), "=r"(cols[1]), "=r"(cols[2]), "=r"(cols[3])
        : "r"(src)
        : "memory");
    __nv_bfloat162 amax_c = __as_bf16x2(0);
    __nv_bfloat162 _hxa_50 = amax_c;
    __nv_bfloat162 _hxb_50 = __as_bf16x2(cols[0]);
    unsigned int _hxr_50;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_50) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_50)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_50)));
    __nv_bfloat162 _max_49 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_50);
    amax_c = _max_49;
    __nv_bfloat162 _hxa_52 = amax_c;
    __nv_bfloat162 _hxb_52 = __as_bf16x2(cols[1]);
    unsigned int _hxr_52;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_52) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_52)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_52)));
    __nv_bfloat162 _max_51 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_52);
    amax_c = _max_51;
    __nv_bfloat162 _hxa_54 = amax_c;
    __nv_bfloat162 _hxb_54 = __as_bf16x2(cols[2]);
    unsigned int _hxr_54;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_54) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_54)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_54)));
    __nv_bfloat162 _max_53 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_54);
    amax_c = _max_53;
    __nv_bfloat162 _hxa_56 = amax_c;
    __nv_bfloat162 _hxb_56 = __as_bf16x2(cols[3]);
    unsigned int _hxr_56;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_56) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_56)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_56)));
    __nv_bfloat162 _max_55 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_56);
    amax_c = _max_55;
    unsigned int mag_c = __as_u32(amax_c) & 2147450879;
    __nv_bfloat162 col_max = __hmax2_nan(__as_bf16x2(mag_c), __as_bf16x2(mag_c >> 16));
    unsigned int cpeer1 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(col_max), 1);
    col_max = __hmax2_nan(col_max, __as_bf16x2(cpeer1));
    unsigned int cpeer2 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(col_max), 2);
    col_max = __hmax2_nan(col_max, __as_bf16x2(cpeer2));
    unsigned int amax_bits_0 = (__as_u32(col_max) & 65535) << 16;
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
    int gcol = bx * 32 + warp_1 * 8 + (lane_1 >> 2);
    int out_row = by * 32 + (lane_1 & 3) * 8;
    unsigned int col_words[2];
    uint32_t bf16x2_mul_2;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2) : "r"(cols[0]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_1) : "r"(cols[1]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_58;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_58) : "r"(bf16x2_mul_1_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_58) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_57;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_57) : "h"(e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_58));
    col_words[0] = _mov_b32_pack_u16x2_57;
    uint32_t bf16x2_mul_3;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3) : "r"(cols[2]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_2) : "r"(cols[3]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_60;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_60) : "r"(bf16x2_mul_1_2));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_2);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_60) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_59;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_59) : "h"(e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_60));
    col_words[1] = _mov_b32_pack_u16x2_59;
    reinterpret_cast<int2*>(col_output + (gcol * M + out_row))[0] = reinterpret_cast<int2*>(col_words + 0)[0];
    if ((lane_1 & 3) == 0) {
        int col_scale_index = ((gcol >> 7) * (M >> 7) + (by >> 2)) * 512 + (gcol & 31) * 16 + (gcol >> 5 & 3) * 4 + (by & 3);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_index) + (0)) = (unsigned char)(scale_0);
    }
}

} // extern "C"
