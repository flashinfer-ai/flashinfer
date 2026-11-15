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

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_gated_act_mxfp8_6ffdb2bd1b07f3dbf991(__nv_bfloat16* __restrict__ gated_input, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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
        float dn;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn) : "f"(e0_3), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_3 = approx_rcp(dn);
        float _fma_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(-dn), "f"(r_3), "f"(1.0f));
        float q;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q) : "f"(r_3), "f"(_fma_13), "f"(r_3));
        float s0;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0) : "f"(q), "f"(1.1754944e-38f));
        float err;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err) : "f"(s0), "f"(-8.507059e+37f), "f"(q));
        float y2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2) : "f"(-dn), "f"(q), "f"(1.0f));
        float nudge = __uint_as_float(__as_u32(y2) & 2147483648u | 729808896);
        float _add_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(err), "f"(nudge));
        float u0;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0) : "f"(_add_14), "f"(1.1754944e-38f), "f"(s0));
        float dn_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_0) : "f"(e1_3), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_3_1 = approx_rcp(dn_0);
        float _fma_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(-dn_0), "f"(r_3_1), "f"(1.0f));
        float q_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_0) : "f"(r_3_1), "f"(_fma_15), "f"(r_3_1));
        float s0_0;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_0) : "f"(q_0), "f"(1.1754944e-38f));
        float err_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_0) : "f"(s0_0), "f"(-8.507059e+37f), "f"(q_0));
        float y2_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_0) : "f"(-dn_0), "f"(q_0), "f"(1.0f));
        float nudge_0 = __uint_as_float(__as_u32(y2_0) & 2147483648u | 729808896);
        float _add_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(err_0), "f"(nudge_0));
        float u1;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1) : "f"(_add_16), "f"(1.1754944e-38f), "f"(s0_0));
        float2 _f32x2_mul_a_17 = mul_f32x2_noftz(cvt_f32_4, make_float2(u0, u1));
        float2 mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&cvt_f32_1_3));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_3.x, mul_3.y)));
        float2 cvt_f32_4_1 = cake_bf16x2_to_f32x2(__as_bf16x2(words[1]));
        float2 cvt_f32_1_3_1 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[1]));
        float2 _f32x2_mul_b_18 = make_float2(-1.442695f, -1.442695f);
        float2 t_3_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_3_1) : "l"(*(const unsigned long long*)&cvt_f32_4_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_18));
        float e0_3_1 = approx_exp2(t_3_1.x);
        float e1_3_1 = approx_exp2(t_3_1.y);
        float2 nd_3_1 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_3_1, e1_3_1));
        float d0_3_1 = -nd_3_1.x;
        float d1_3_1 = -nd_3_1.y;
        float dn_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_1) : "f"(e0_3_1), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_4 = approx_rcp(dn_1);
        float _fma_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(-dn_1), "f"(r_4), "f"(1.0f));
        float q_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_1) : "f"(r_4), "f"(_fma_19), "f"(r_4));
        float s0_1;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_1) : "f"(q_1), "f"(1.1754944e-38f));
        float err_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_1) : "f"(s0_1), "f"(-8.507059e+37f), "f"(q_1));
        float y2_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_1) : "f"(-dn_1), "f"(q_1), "f"(1.0f));
        float nudge_1 = __uint_as_float(__as_u32(y2_1) & 2147483648u | 729808896);
        float _add_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(err_1), "f"(nudge_1));
        float u0_0;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0_0) : "f"(_add_20), "f"(1.1754944e-38f), "f"(s0_1));
        float dn_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_2) : "f"(e1_3_1), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_5 = approx_rcp(dn_2);
        float _fma_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(-dn_2), "f"(r_5), "f"(1.0f));
        float q_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_2) : "f"(r_5), "f"(_fma_21), "f"(r_5));
        float s0_2;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_2) : "f"(q_2), "f"(1.1754944e-38f));
        float err_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_2) : "f"(s0_2), "f"(-8.507059e+37f), "f"(q_2));
        float y2_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_2) : "f"(-dn_2), "f"(q_2), "f"(1.0f));
        float nudge_2 = __uint_as_float(__as_u32(y2_2) & 2147483648u | 729808896);
        float _add_22;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(err_2), "f"(nudge_2));
        float u1_0;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1_0) : "f"(_add_22), "f"(1.1754944e-38f), "f"(s0_2));
        float2 _f32x2_mul_a_23 = mul_f32x2_noftz(cvt_f32_4_1, make_float2(u0_0, u1_0));
        float2 mul_3_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&cvt_f32_1_3_1));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_3_1.x, mul_3_1.y)));
        float2 cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(words[2]));
        float2 cvt_f32_1_4 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[2]));
        float2 _f32x2_mul_b_24 = make_float2(-1.442695f, -1.442695f);
        float2 t_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_4) : "l"(*(const unsigned long long*)&cvt_f32_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_24));
        float e0_4 = approx_exp2(t_4.x);
        float e1_4 = approx_exp2(t_4.y);
        float2 nd_4 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_4, e1_4));
        float d0_4 = -nd_4.x;
        float d1_4 = -nd_4.y;
        float dn_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_3) : "f"(e0_4), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_6 = approx_rcp(dn_3);
        float _fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(-dn_3), "f"(r_6), "f"(1.0f));
        float q_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_3) : "f"(r_6), "f"(_fma_25), "f"(r_6));
        float s0_3;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_3) : "f"(q_3), "f"(1.1754944e-38f));
        float err_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_3) : "f"(s0_3), "f"(-8.507059e+37f), "f"(q_3));
        float y2_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_3) : "f"(-dn_3), "f"(q_3), "f"(1.0f));
        float nudge_3 = __uint_as_float(__as_u32(y2_3) & 2147483648u | 729808896);
        float _add_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(err_3), "f"(nudge_3));
        float u0_1;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0_1) : "f"(_add_26), "f"(1.1754944e-38f), "f"(s0_3));
        float dn_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_4) : "f"(e1_4), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_7 = approx_rcp(dn_4);
        float _fma_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(-dn_4), "f"(r_7), "f"(1.0f));
        float q_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_4) : "f"(r_7), "f"(_fma_27), "f"(r_7));
        float s0_4;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_4) : "f"(q_4), "f"(1.1754944e-38f));
        float err_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_4) : "f"(s0_4), "f"(-8.507059e+37f), "f"(q_4));
        float y2_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_4) : "f"(-dn_4), "f"(q_4), "f"(1.0f));
        float nudge_4 = __uint_as_float(__as_u32(y2_4) & 2147483648u | 729808896);
        float _add_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(err_4), "f"(nudge_4));
        float u1_1;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1_1) : "f"(_add_28), "f"(1.1754944e-38f), "f"(s0_4));
        float2 _f32x2_mul_a_29 = mul_f32x2_noftz(cvt_f32_5, make_float2(u0_1, u1_1));
        float2 mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&cvt_f32_1_4));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_4.x, mul_4.y)));
        float2 cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(words[3]));
        float2 cvt_f32_1_5 = cake_bf16x2_to_f32x2(__as_bf16x2(words_0[3]));
        float2 _f32x2_mul_b_30 = make_float2(-1.442695f, -1.442695f);
        float2 t_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_5) : "l"(*(const unsigned long long*)&cvt_f32_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_30));
        float e0_5 = approx_exp2(t_5.x);
        float e1_5 = approx_exp2(t_5.y);
        float2 nd_5 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_5, e1_5));
        float d0_5 = -nd_5.x;
        float d1_5 = -nd_5.y;
        float dn_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_5) : "f"(e0_5), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_8 = approx_rcp(dn_5);
        float _fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(-dn_5), "f"(r_8), "f"(1.0f));
        float q_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_5) : "f"(r_8), "f"(_fma_31), "f"(r_8));
        float s0_5;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_5) : "f"(q_5), "f"(1.1754944e-38f));
        float err_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_5) : "f"(s0_5), "f"(-8.507059e+37f), "f"(q_5));
        float y2_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_5) : "f"(-dn_5), "f"(q_5), "f"(1.0f));
        float nudge_5 = __uint_as_float(__as_u32(y2_5) & 2147483648u | 729808896);
        float _add_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(err_5), "f"(nudge_5));
        float u0_2;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0_2) : "f"(_add_32), "f"(1.1754944e-38f), "f"(s0_5));
        float dn_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_6) : "f"(e1_5), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_9 = approx_rcp(dn_6);
        float _fma_33;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(-dn_6), "f"(r_9), "f"(1.0f));
        float q_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_6) : "f"(r_9), "f"(_fma_33), "f"(r_9));
        float s0_6;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_6) : "f"(q_6), "f"(1.1754944e-38f));
        float err_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_6) : "f"(s0_6), "f"(-8.507059e+37f), "f"(q_6));
        float y2_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_6) : "f"(-dn_6), "f"(q_6), "f"(1.0f));
        float nudge_6 = __uint_as_float(__as_u32(y2_6) & 2147483648u | 729808896);
        float _add_34;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(err_6), "f"(nudge_6));
        float u1_2;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1_2) : "f"(_add_34), "f"(1.1754944e-38f), "f"(s0_6));
        float2 _f32x2_mul_a_35 = mul_f32x2_noftz(cvt_f32_6, make_float2(u0_2, u1_2));
        float2 mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&cvt_f32_1_5));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_5.x, mul_5.y)));
    }
    unsigned int nswizzle = (unsigned int)(row >> 3 & 3);
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(tile_addr + (unsigned int)(row * 64) + (((unsigned int)quarter ^ nswizzle) << 4)), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
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
    __nv_bfloat162 _hxa_37 = amax_c;
    __nv_bfloat162 _hxb_37 = __as_bf16x2(cols[0]);
    unsigned int _hxr_37;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_37) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_37)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_37)));
    __nv_bfloat162 _max_36 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_37);
    amax_c = _max_36;
    __nv_bfloat162 _hxa_39 = amax_c;
    __nv_bfloat162 _hxb_39 = __as_bf16x2(cols[1]);
    unsigned int _hxr_39;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_39) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_39)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_39)));
    __nv_bfloat162 _max_38 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_39);
    amax_c = _max_38;
    __nv_bfloat162 _hxa_41 = amax_c;
    __nv_bfloat162 _hxb_41 = __as_bf16x2(cols[2]);
    unsigned int _hxr_41;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_41) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_41)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_41)));
    __nv_bfloat162 _max_40 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_41);
    amax_c = _max_40;
    __nv_bfloat162 _hxa_43 = amax_c;
    __nv_bfloat162 _hxb_43 = __as_bf16x2(cols[3]);
    unsigned int _hxr_43;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_43) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_43)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_43)));
    __nv_bfloat162 _max_42 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_43);
    amax_c = _max_42;
    unsigned int mag_c = __as_u32(amax_c) & 2147450879;
    __nv_bfloat162 col_max = __hmax2_nan(__as_bf16x2(mag_c), __as_bf16x2(mag_c >> 16));
    unsigned int cpeer1 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(col_max), 1);
    col_max = __hmax2_nan(col_max, __as_bf16x2(cpeer1));
    unsigned int cpeer2 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(col_max), 2);
    col_max = __hmax2_nan(col_max, __as_bf16x2(cpeer2));
    unsigned int amax_bits = (__as_u32(col_max) & 65535) << 16;
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
    int gcol = bx * 32 + warp_1 * 8 + (lane_1 >> 2);
    int out_row = by * 32 + (lane_1 & 3) * 8;
    unsigned int col_words[2];
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_45;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_45) : "r"(bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_45) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_44;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_44) : "h"(e4m3x2_bf16x2), "h"(_cvt_satfinite_e4m3x2_bf16x2_45));
    col_words[0] = _mov_b32_pack_u16x2_44;
    uint32_t bf16x2_mul_0;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_0) : "r"(cols[2]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_0) : "r"(cols[3]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_47;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_47) : "r"(bf16x2_mul_1_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_47) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_46;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_46) : "h"(e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_47));
    col_words[1] = _mov_b32_pack_u16x2_46;
    reinterpret_cast<int2*>(col_output + (gcol * M + out_row))[0] = reinterpret_cast<int2*>(col_words + 0)[0];
    if ((lane_1 & 3) == 0) {
        int col_scale_index = ((gcol >> 7) * (M >> 7) + (by >> 2)) * 512 + (gcol & 31) * 16 + (gcol >> 5 & 3) * 4 + (by & 3);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_index) + (0)) = (unsigned char)(scale);
    }
}

} // extern "C"
