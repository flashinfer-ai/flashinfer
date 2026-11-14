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

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_gated_act_mxfp8_2ea78301fb3ef01a1b41(__nv_bfloat16* __restrict__ gated_input, __nv_bfloat16* __restrict__ grad, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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
    uint8_t* tiles = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int tiles_addr = smem + 0;

    // === Task calls (declaration order) ===
    int quarter = tid & 3;
    int row = tid >> 2;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int grow = by * 32 + row;
    int col = bx * 32 + quarter * 8;
    int gate_index = grow * (2 * K) + col;
    unsigned int vec_load[4];
    {
        uint32_t _uv4_0_0;
        uint32_t _uv4_0_1;
        uint32_t _uv4_0_2;
        uint32_t _uv4_0_3;
        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
            : "=r"(_uv4_0_0), "=r"(_uv4_0_1), "=r"(_uv4_0_2), "=r"(_uv4_0_3) : "l"((const void*)(gated_input + gate_index + (0))), "l"(0x12F0000000000000ULL) : "memory");
        vec_load[0 + 0] = _uv4_0_0;
        vec_load[0 + 1] = _uv4_0_1;
        vec_load[0 + 2] = _uv4_0_2;
        vec_load[0 + 3] = _uv4_0_3;
    }
    unsigned int vec_load_1[4];
    {
        uint32_t _uv4_1_0;
        uint32_t _uv4_1_1;
        uint32_t _uv4_1_2;
        uint32_t _uv4_1_3;
        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
            : "=r"(_uv4_1_0), "=r"(_uv4_1_1), "=r"(_uv4_1_2), "=r"(_uv4_1_3) : "l"((const void*)(gated_input + (gate_index + K) + (0))), "l"(0x12F0000000000000ULL) : "memory");
        vec_load_1[0 + 0] = _uv4_1_0;
        vec_load_1[0 + 1] = _uv4_1_1;
        vec_load_1[0 + 2] = _uv4_1_2;
        vec_load_1[0 + 3] = _uv4_1_3;
    }
    unsigned int vec_load_2[4];
    {
        uint32_t _uv4_2_0;
        uint32_t _uv4_2_1;
        uint32_t _uv4_2_2;
        uint32_t _uv4_2_3;
        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
            : "=r"(_uv4_2_0), "=r"(_uv4_2_1), "=r"(_uv4_2_2), "=r"(_uv4_2_3) : "l"((const void*)(grad + (grow * K + col) + (0))), "l"(0x12F0000000000000ULL) : "memory");
        vec_load_2[0 + 0] = _uv4_2_0;
        vec_load_2[0 + 1] = _uv4_2_1;
        vec_load_2[0 + 2] = _uv4_2_2;
        vec_load_2[0 + 3] = _uv4_2_3;
    }
    unsigned int dgate[4];
    unsigned int dup[4];
    __nv_bfloat162 lowest = __as_bf16x2(vec_load[0]);
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[3]));
    float2 lowest_f32 = cake_bf16x2_to_f32x2(lowest);
    float _min_4;
    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_4) : "f"(lowest_f32.x), "f"(lowest_f32.y));
    int _warp_vote_3 = __all_sync(0xFFFFFFFF, _min_4 > -87.0f);
    if (_warp_vote_3 != 0) {
        float2 x = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[0]));
        float2 u = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[0]));
        float2 g = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[0]));
        float e0 = approx_exp2(-x.x * 1.442695f);
        float e1 = approx_exp2(-x.y * 1.442695f);
        float d0 = 1.0f + e0;
        float d1 = 1.0f + e1;
        float r0 = approx_rcp(d0);
        float r1 = approx_rcp(d1);
        float _fma_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(-d0), "f"(r0), "f"(1.0f));
        float _fma_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(r0), "f"(_fma_6), "f"(r0));
        float _fma_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(-d1), "f"(r1), "f"(1.0f));
        float _fma_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(r1), "f"(_fma_8), "f"(r1));
        float2 s = make_float2(_fma_5, _fma_7);
        float2 act;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act) : "l"(*(const unsigned long long*)&x), "l"(*(const unsigned long long*)&s));
        float2 _f32x2_mul_a_9 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s), s), g);
        float2 mul;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul) : "l"(*(const unsigned long long*)&_f32x2_mul_a_9), "l"(*(const unsigned long long*)&u));
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
        float _fma_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(-d0_0), "f"(r0_0), "f"(1.0f));
        float _fma_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(r0_0), "f"(_fma_11), "f"(r0_0));
        float _fma_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(-d1_0), "f"(r1_0), "f"(1.0f));
        float _fma_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(r1_0), "f"(_fma_13), "f"(r1_0));
        float2 s_0 = make_float2(_fma_10, _fma_12);
        float2 act_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_0) : "l"(*(const unsigned long long*)&x_0), "l"(*(const unsigned long long*)&s_0));
        float2 _f32x2_mul_a_14 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_0), s_0), g_0);
        float2 mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&u_0));
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
        float _fma_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(-d0_1), "f"(r0_1), "f"(1.0f));
        float _fma_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(r0_1), "f"(_fma_16), "f"(r0_1));
        float _fma_18;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(-d1_1), "f"(r1_1), "f"(1.0f));
        float _fma_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(r1_1), "f"(_fma_18), "f"(r1_1));
        float2 s_1 = make_float2(_fma_15, _fma_17);
        float2 act_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_1) : "l"(*(const unsigned long long*)&x_1), "l"(*(const unsigned long long*)&s_1));
        float2 _f32x2_mul_a_19 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1), s_1), g_1);
        float2 mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&u_1));
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
        float _fma_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(-d0_2), "f"(r0_2), "f"(1.0f));
        float _fma_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(r0_2), "f"(_fma_21), "f"(r0_2));
        float _fma_23;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(-d1_2), "f"(r1_2), "f"(1.0f));
        float _fma_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(r1_2), "f"(_fma_23), "f"(r1_2));
        float2 s_2 = make_float2(_fma_20, _fma_22);
        float2 act_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_2) : "l"(*(const unsigned long long*)&x_2), "l"(*(const unsigned long long*)&s_2));
        float2 _f32x2_mul_a_24 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_2), s_2), g_2);
        float2 mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&u_2));
        float2 mul_1_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_2) : "l"(*(const unsigned long long*)&act_2), "l"(*(const unsigned long long*)&g_2));
        __nv_bfloat162 bf16x2_4 = __float22bfloat162_rn(make_float2(mul_3.x, mul_3.y));
        __nv_bfloat162 bf16x2_1_3 = __float22bfloat162_rn(make_float2(mul_1_2.x, mul_1_2.y));
        dgate[3] = __as_u32(bf16x2_4);
        dup[3] = __as_u32(bf16x2_1_3);
    } else {
        int band = 0;
        float2 x_3 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[0]));
        float2 u_3 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[0]));
        float2 g_3 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[0]));
        float e0_3 = approx_exp2(-x_3.x * 1.442695f);
        float e1_3 = approx_exp2(-x_3.y * 1.442695f);
        float d0_3 = 1.0f + e0_3;
        float d1_3 = 1.0f + e1_3;
        float dn;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn) : "f"(e0_3), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r = approx_rcp(dn);
        float _fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(-dn), "f"(r), "f"(1.0f));
        float q;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q) : "f"(r), "f"(_fma_25), "f"(r));
        float s0;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0) : "f"(q), "f"(1.1754944e-38f));
        float err;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err) : "f"(s0), "f"(-8.507059e+37f), "f"(q));
        float y2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2) : "f"(-dn), "f"(q), "f"(1.0f));
        float nudge = __uint_as_float(__as_u32(y2) & 2147483648u | 729808896);
        float _add_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(err), "f"(nudge));
        float u0;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0) : "f"(_add_26), "f"(1.1754944e-38f), "f"(s0));
        float dn_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_0) : "f"(e1_3), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_0 = approx_rcp(dn_0);
        float _fma_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(-dn_0), "f"(r_0), "f"(1.0f));
        float q_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_0) : "f"(r_0), "f"(_fma_27), "f"(r_0));
        float s0_0;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_0) : "f"(q_0), "f"(1.1754944e-38f));
        float err_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_0) : "f"(s0_0), "f"(-8.507059e+37f), "f"(q_0));
        float y2_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_0) : "f"(-dn_0), "f"(q_0), "f"(1.0f));
        float nudge_0 = __uint_as_float(__as_u32(y2_0) & 2147483648u | 729808896);
        float _add_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(err_0), "f"(nudge_0));
        float u1;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1) : "f"(_add_28), "f"(1.1754944e-38f), "f"(s0_0));
        float2 s_3 = make_float2(u0, u1);
        float2 act_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_3) : "l"(*(const unsigned long long*)&x_3), "l"(*(const unsigned long long*)&s_3));
        float2 _f32x2_mul_a_29 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_3), s_3), g_3);
        float2 mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&u_3));
        float2 mul_1_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_3) : "l"(*(const unsigned long long*)&act_3), "l"(*(const unsigned long long*)&g_3));
        __nv_bfloat162 bf16x2_5 = __float22bfloat162_rn(make_float2(mul_4.x, mul_4.y));
        __nv_bfloat162 bf16x2_1_4 = __float22bfloat162_rn(make_float2(mul_1_3.x, mul_1_3.y));
        dgate[0] = __as_u32(bf16x2_5);
        dup[0] = __as_u32(bf16x2_1_4);
        float2 x_3_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[1]));
        float2 u_3_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[1]));
        float2 g_3_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[1]));
        float e0_3_1 = approx_exp2(-x_3_1.x * 1.442695f);
        float e1_3_1 = approx_exp2(-x_3_1.y * 1.442695f);
        float d0_3_1 = 1.0f + e0_3_1;
        float d1_3_1 = 1.0f + e1_3_1;
        float dn_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_1) : "f"(e0_3_1), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_1 = approx_rcp(dn_1);
        float _fma_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(-dn_1), "f"(r_1), "f"(1.0f));
        float q_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_1) : "f"(r_1), "f"(_fma_30), "f"(r_1));
        float s0_1;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_1) : "f"(q_1), "f"(1.1754944e-38f));
        float err_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_1) : "f"(s0_1), "f"(-8.507059e+37f), "f"(q_1));
        float y2_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_1) : "f"(-dn_1), "f"(q_1), "f"(1.0f));
        float nudge_1 = __uint_as_float(__as_u32(y2_1) & 2147483648u | 729808896);
        float _add_31;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(err_1), "f"(nudge_1));
        float u0_0;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0_0) : "f"(_add_31), "f"(1.1754944e-38f), "f"(s0_1));
        float dn_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_2) : "f"(e1_3_1), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_2 = approx_rcp(dn_2);
        float _fma_32;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(-dn_2), "f"(r_2), "f"(1.0f));
        float q_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_2) : "f"(r_2), "f"(_fma_32), "f"(r_2));
        float s0_2;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_2) : "f"(q_2), "f"(1.1754944e-38f));
        float err_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_2) : "f"(s0_2), "f"(-8.507059e+37f), "f"(q_2));
        float y2_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_2) : "f"(-dn_2), "f"(q_2), "f"(1.0f));
        float nudge_2 = __uint_as_float(__as_u32(y2_2) & 2147483648u | 729808896);
        float _add_33;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(err_2), "f"(nudge_2));
        float u1_0;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1_0) : "f"(_add_33), "f"(1.1754944e-38f), "f"(s0_2));
        float2 s_3_1 = make_float2(u0_0, u1_0);
        float2 act_3_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_3_1) : "l"(*(const unsigned long long*)&x_3_1), "l"(*(const unsigned long long*)&s_3_1));
        float2 _f32x2_mul_a_34 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_3_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_3_1), s_3_1), g_3_1);
        float2 mul_4_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_34), "l"(*(const unsigned long long*)&u_3_1));
        float2 mul_1_3_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_3_1) : "l"(*(const unsigned long long*)&act_3_1), "l"(*(const unsigned long long*)&g_3_1));
        __nv_bfloat162 bf16x2_6 = __float22bfloat162_rn(make_float2(mul_4_1.x, mul_4_1.y));
        __nv_bfloat162 bf16x2_1_5 = __float22bfloat162_rn(make_float2(mul_1_3_1.x, mul_1_3_1.y));
        dgate[1] = __as_u32(bf16x2_6);
        dup[1] = __as_u32(bf16x2_1_5);
        float2 x_4 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[2]));
        float2 u_4 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[2]));
        float2 g_4 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[2]));
        float e0_4 = approx_exp2(-x_4.x * 1.442695f);
        float e1_4 = approx_exp2(-x_4.y * 1.442695f);
        float d0_4 = 1.0f + e0_4;
        float d1_4 = 1.0f + e1_4;
        float dn_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_3) : "f"(e0_4), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_3 = approx_rcp(dn_3);
        float _fma_35;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(-dn_3), "f"(r_3), "f"(1.0f));
        float q_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_3) : "f"(r_3), "f"(_fma_35), "f"(r_3));
        float s0_3;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_3) : "f"(q_3), "f"(1.1754944e-38f));
        float err_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_3) : "f"(s0_3), "f"(-8.507059e+37f), "f"(q_3));
        float y2_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_3) : "f"(-dn_3), "f"(q_3), "f"(1.0f));
        float nudge_3 = __uint_as_float(__as_u32(y2_3) & 2147483648u | 729808896);
        float _add_36;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(err_3), "f"(nudge_3));
        float u0_1;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0_1) : "f"(_add_36), "f"(1.1754944e-38f), "f"(s0_3));
        float dn_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_4) : "f"(e1_4), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_4 = approx_rcp(dn_4);
        float _fma_37;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(-dn_4), "f"(r_4), "f"(1.0f));
        float q_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_4) : "f"(r_4), "f"(_fma_37), "f"(r_4));
        float s0_4;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_4) : "f"(q_4), "f"(1.1754944e-38f));
        float err_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_4) : "f"(s0_4), "f"(-8.507059e+37f), "f"(q_4));
        float y2_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_4) : "f"(-dn_4), "f"(q_4), "f"(1.0f));
        float nudge_4 = __uint_as_float(__as_u32(y2_4) & 2147483648u | 729808896);
        float _add_38;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(err_4), "f"(nudge_4));
        float u1_1;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1_1) : "f"(_add_38), "f"(1.1754944e-38f), "f"(s0_4));
        float2 s_4 = make_float2(u0_1, u1_1);
        float2 act_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_4) : "l"(*(const unsigned long long*)&x_4), "l"(*(const unsigned long long*)&s_4));
        float2 _f32x2_mul_a_39 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_4), s_4), g_4);
        float2 mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_39), "l"(*(const unsigned long long*)&u_4));
        float2 mul_1_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_4) : "l"(*(const unsigned long long*)&act_4), "l"(*(const unsigned long long*)&g_4));
        __nv_bfloat162 bf16x2_7 = __float22bfloat162_rn(make_float2(mul_5.x, mul_5.y));
        __nv_bfloat162 bf16x2_1_6 = __float22bfloat162_rn(make_float2(mul_1_4.x, mul_1_4.y));
        dgate[2] = __as_u32(bf16x2_7);
        dup[2] = __as_u32(bf16x2_1_6);
        float2 x_5 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[3]));
        float2 u_5 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[3]));
        float2 g_5 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[3]));
        float e0_5 = approx_exp2(-x_5.x * 1.442695f);
        float e1_5 = approx_exp2(-x_5.y * 1.442695f);
        float d0_5 = 1.0f + e0_5;
        float d1_5 = 1.0f + e1_5;
        float dn_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_5) : "f"(e0_5), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_5 = approx_rcp(dn_5);
        float _fma_40;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(-dn_5), "f"(r_5), "f"(1.0f));
        float q_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_5) : "f"(r_5), "f"(_fma_40), "f"(r_5));
        float s0_5;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_5) : "f"(q_5), "f"(1.1754944e-38f));
        float err_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_5) : "f"(s0_5), "f"(-8.507059e+37f), "f"(q_5));
        float y2_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_5) : "f"(-dn_5), "f"(q_5), "f"(1.0f));
        float nudge_5 = __uint_as_float(__as_u32(y2_5) & 2147483648u | 729808896);
        float _add_41;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(err_5), "f"(nudge_5));
        float u0_2;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0_2) : "f"(_add_41), "f"(1.1754944e-38f), "f"(s0_5));
        float dn_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_6) : "f"(e1_5), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_6 = approx_rcp(dn_6);
        float _fma_42;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(-dn_6), "f"(r_6), "f"(1.0f));
        float q_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_6) : "f"(r_6), "f"(_fma_42), "f"(r_6));
        float s0_6;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_6) : "f"(q_6), "f"(1.1754944e-38f));
        float err_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_6) : "f"(s0_6), "f"(-8.507059e+37f), "f"(q_6));
        float y2_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_6) : "f"(-dn_6), "f"(q_6), "f"(1.0f));
        float nudge_6 = __uint_as_float(__as_u32(y2_6) & 2147483648u | 729808896);
        float _add_43;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_43) : "f"(err_6), "f"(nudge_6));
        float u1_2;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1_2) : "f"(_add_43), "f"(1.1754944e-38f), "f"(s0_6));
        float2 s_5 = make_float2(u0_2, u1_2);
        float2 act_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_5) : "l"(*(const unsigned long long*)&x_5), "l"(*(const unsigned long long*)&s_5));
        float2 _f32x2_mul_a_44 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_5), s_5), g_5);
        float2 mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&u_5));
        float2 mul_1_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_5) : "l"(*(const unsigned long long*)&act_5), "l"(*(const unsigned long long*)&g_5));
        __nv_bfloat162 bf16x2_8 = __float22bfloat162_rn(make_float2(mul_6.x, mul_6.y));
        __nv_bfloat162 bf16x2_1_7 = __float22bfloat162_rn(make_float2(mul_1_5.x, mul_1_5.y));
        dgate[3] = __as_u32(bf16x2_8);
        dup[3] = __as_u32(bf16x2_1_7);
    }
    unsigned int chunk_addr = tiles_addr + (unsigned int)(row * 64) + (((unsigned int)quarter ^ (unsigned int)(row >> 3 & 3)) << 4);
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(chunk_addr), "r"(dgate[0]), "r"(dgate[1]), "r"(dgate[2]), "r"(dgate[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(chunk_addr + 2048), "r"(dup[0]), "r"(dup[1]), "r"(dup[2]), "r"(dup[3]) : "memory");
    int scale_cols = K >> 6;
    int out_index = grow * (2 * K) + col;
    __nv_bfloat162 amax = __as_bf16x2(0);
    __nv_bfloat162 _hxa_46 = amax;
    __nv_bfloat162 _hxb_46 = __as_bf16x2(dgate[0]);
    unsigned int _hxr_46;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_46) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_46)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_46)));
    __nv_bfloat162 _max_45 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_46);
    amax = _max_45;
    __nv_bfloat162 _hxa_48 = amax;
    __nv_bfloat162 _hxb_48 = __as_bf16x2(dgate[1]);
    unsigned int _hxr_48;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_48) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_48)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_48)));
    __nv_bfloat162 _max_47 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_48);
    amax = _max_47;
    __nv_bfloat162 _hxa_50 = amax;
    __nv_bfloat162 _hxb_50 = __as_bf16x2(dgate[2]);
    unsigned int _hxr_50;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_50) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_50)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_50)));
    __nv_bfloat162 _max_49 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_50);
    amax = _max_49;
    __nv_bfloat162 _hxa_52 = amax;
    __nv_bfloat162 _hxb_52 = __as_bf16x2(dgate[3]);
    unsigned int _hxr_52;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_52) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_52)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_52)));
    __nv_bfloat162 _max_51 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_52);
    amax = _max_51;
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
    int scale_index = ((grow >> 7) * scale_cols + (bx >> 2)) * 512 + (grow & 31) * 16 + (grow >> 5 & 3) * 4 + (bx & 3);
    if (quarter == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + scale_index) + (0)) = (unsigned char)(scale);
    }
    unsigned int words[2];
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_54;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_54) : "r"(bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_54) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_53;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_53) : "h"(e4m3x2_bf16x2), "h"(_cvt_satfinite_e4m3x2_bf16x2_54));
    words[0] = _mov_b32_pack_u16x2_53;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_56;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_56) : "r"(bf16x2_mul_1_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_56) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_55;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_55) : "h"(e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_56));
    words[1] = _mov_b32_pack_u16x2_55;
    reinterpret_cast<int2*>(row_output + out_index)[0] = reinterpret_cast<int2*>(words + 0)[0];
    __nv_bfloat162 amax_0 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_58 = amax_0;
    __nv_bfloat162 _hxb_58 = __as_bf16x2(dup[0]);
    unsigned int _hxr_58;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_58) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_58)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_58)));
    __nv_bfloat162 _max_57 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_58);
    amax_0 = _max_57;
    __nv_bfloat162 _hxa_60 = amax_0;
    __nv_bfloat162 _hxb_60 = __as_bf16x2(dup[1]);
    unsigned int _hxr_60;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_60) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_60)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_60)));
    __nv_bfloat162 _max_59 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_60);
    amax_0 = _max_59;
    __nv_bfloat162 _hxa_62 = amax_0;
    __nv_bfloat162 _hxb_62 = __as_bf16x2(dup[2]);
    unsigned int _hxr_62;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_62) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_62)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_62)));
    __nv_bfloat162 _max_61 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_62);
    amax_0 = _max_61;
    __nv_bfloat162 _hxa_64 = amax_0;
    __nv_bfloat162 _hxb_64 = __as_bf16x2(dup[3]);
    unsigned int _hxr_64;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_64) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_64)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_64)));
    __nv_bfloat162 _max_63 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_64);
    amax_0 = _max_63;
    unsigned int mag_0 = __as_u32(amax_0) & 2147450879;
    unsigned int peer1_0 = __shfl_xor_sync(0xFFFFFFFF, mag_0, 1);
    __nv_bfloat162 pair_max_0 = __hmax2_nan(__as_bf16x2(mag_0), __as_bf16x2(peer1_0));
    unsigned int peer2_0 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(pair_max_0), 2);
    __nv_bfloat162 quad_max_0 = __hmax2_nan(pair_max_0, __as_bf16x2(peer2_0));
    unsigned int folded_0 = __as_u32(__hmax2_nan(quad_max_0, __as_bf16x2(__as_u32(quad_max_0) >> 16)));
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
    int scale_index_0 = ((grow >> 7) * scale_cols + (bx + (K >> 5) >> 2)) * 512 + (grow & 31) * 16 + (grow >> 5 & 3) * 4 + (bx + (K >> 5) & 3);
    if (quarter == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + scale_index_0) + (0)) = (unsigned char)(scale_0);
    }
    unsigned int words_0[2];
    uint32_t bf16x2_mul_4;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4) : "r"(dup[0]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_1) : "r"(dup[1]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_66;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_66) : "r"(bf16x2_mul_1_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_66) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_65;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_65) : "h"(e4m3x2_bf16x2_4), "h"(_cvt_satfinite_e4m3x2_bf16x2_66));
    words_0[0] = _mov_b32_pack_u16x2_65;
    uint32_t bf16x2_mul_5;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5) : "r"(dup[2]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_2) : "r"(dup[3]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_68;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_68) : "r"(bf16x2_mul_1_2));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_2);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_68) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_67;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_67) : "h"(e4m3x2_bf16x2_5), "h"(_cvt_satfinite_e4m3x2_bf16x2_68));
    words_0[1] = _mov_b32_pack_u16x2_67;
    reinterpret_cast<int2*>(row_output + (out_index + K))[0] = reinterpret_cast<int2*>(words_0 + 0)[0];
    __syncthreads();
    int lane_1 = tid & 31;
    int warp_1 = tid >> 5;
    int k8 = lane_1 & 7;
    int srow = (k8 >> 1) * 8 + (lane_1 >> 3) * 2 + (k8 & 1);
    unsigned int sswizzle = (unsigned int)(srow >> 3 & 3);
    unsigned int cols_1[4];
    unsigned int src = tiles_addr + (unsigned int)(srow * 64) + (((unsigned int)warp_1 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_1[0]), "=r"(cols_1[1]), "=r"(cols_1[2]), "=r"(cols_1[3])
        : "r"(src)
        : "memory");
    __nv_bfloat162 amax_1 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_70 = amax_1;
    __nv_bfloat162 _hxb_70 = __as_bf16x2(cols_1[0]);
    unsigned int _hxr_70;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_70) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_70)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_70)));
    __nv_bfloat162 _max_69 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_70);
    amax_1 = _max_69;
    __nv_bfloat162 _hxa_72 = amax_1;
    __nv_bfloat162 _hxb_72 = __as_bf16x2(cols_1[1]);
    unsigned int _hxr_72;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_72) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_72)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_72)));
    __nv_bfloat162 _max_71 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_72);
    amax_1 = _max_71;
    __nv_bfloat162 _hxa_74 = amax_1;
    __nv_bfloat162 _hxb_74 = __as_bf16x2(cols_1[2]);
    unsigned int _hxr_74;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_74) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_74)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_74)));
    __nv_bfloat162 _max_73 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_74);
    amax_1 = _max_73;
    __nv_bfloat162 _hxa_76 = amax_1;
    __nv_bfloat162 _hxb_76 = __as_bf16x2(cols_1[3]);
    unsigned int _hxr_76;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_76) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_76)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_76)));
    __nv_bfloat162 _max_75 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_76);
    amax_1 = _max_75;
    unsigned int mag_1 = __as_u32(amax_1) & 2147450879;
    __nv_bfloat162 col_max = __hmax2_nan(__as_bf16x2(mag_1), __as_bf16x2(mag_1 >> 16));
    unsigned int peer1_1 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(col_max), 1);
    col_max = __hmax2_nan(col_max, __as_bf16x2(peer1_1));
    unsigned int peer2_1 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(col_max), 2);
    col_max = __hmax2_nan(col_max, __as_bf16x2(peer2_1));
    unsigned int amax_bits_1 = (__as_u32(col_max) & 65535) << 16;
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
    int gcol = bx * 32 + warp_1 * 8 + (lane_1 >> 2);
    int out_row = by * 32 + (lane_1 & 3) * 8;
    unsigned int words_1[2];
    uint32_t bf16x2_mul_6;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6) : "r"(cols_1[0]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_3) : "r"(cols_1[1]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_78;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_78) : "r"(bf16x2_mul_1_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_78) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_77;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_77) : "h"(e4m3x2_bf16x2_6), "h"(_cvt_satfinite_e4m3x2_bf16x2_78));
    words_1[0] = _mov_b32_pack_u16x2_77;
    uint32_t bf16x2_mul_7;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7) : "r"(cols_1[2]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_4) : "r"(cols_1[3]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_80;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_80) : "r"(bf16x2_mul_1_4));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_4);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_80) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_79;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_79) : "h"(e4m3x2_bf16x2_7), "h"(_cvt_satfinite_e4m3x2_bf16x2_80));
    words_1[1] = _mov_b32_pack_u16x2_79;
    reinterpret_cast<int2*>(col_output + (gcol * M + out_row))[0] = reinterpret_cast<int2*>(words_1 + 0)[0];
    if ((lane_1 & 3) == 0) {
        int scale_index_1 = ((gcol >> 7) * (M >> 7) + (by >> 2)) * 512 + (gcol & 31) * 16 + (gcol >> 5 & 3) * 4 + (by & 3);
        *(reinterpret_cast<unsigned char*>(col_scales + scale_index_1) + (0)) = (unsigned char)(scale_3);
    }
    int lane_0 = tid & 31;
    int warp_0 = tid >> 5;
    int k8_0 = lane_0 & 7;
    int srow_0 = (k8_0 >> 1) * 8 + (lane_0 >> 3) * 2 + (k8_0 & 1);
    unsigned int sswizzle_0 = (unsigned int)(srow_0 >> 3 & 3);
    unsigned int cols_1_0[4];
    unsigned int src_0 = tiles_addr + 2048 + (unsigned int)(srow_0 * 64) + (((unsigned int)warp_0 ^ sswizzle_0) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_1_0[0]), "=r"(cols_1_0[1]), "=r"(cols_1_0[2]), "=r"(cols_1_0[3])
        : "r"(src_0)
        : "memory");
    __nv_bfloat162 amax_2 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_82 = amax_2;
    __nv_bfloat162 _hxb_82 = __as_bf16x2(cols_1_0[0]);
    unsigned int _hxr_82;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_82) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_82)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_82)));
    __nv_bfloat162 _max_81 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_82);
    amax_2 = _max_81;
    __nv_bfloat162 _hxa_84 = amax_2;
    __nv_bfloat162 _hxb_84 = __as_bf16x2(cols_1_0[1]);
    unsigned int _hxr_84;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_84) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_84)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_84)));
    __nv_bfloat162 _max_83 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_84);
    amax_2 = _max_83;
    __nv_bfloat162 _hxa_86 = amax_2;
    __nv_bfloat162 _hxb_86 = __as_bf16x2(cols_1_0[2]);
    unsigned int _hxr_86;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_86) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_86)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_86)));
    __nv_bfloat162 _max_85 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_86);
    amax_2 = _max_85;
    __nv_bfloat162 _hxa_88 = amax_2;
    __nv_bfloat162 _hxb_88 = __as_bf16x2(cols_1_0[3]);
    unsigned int _hxr_88;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_88) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_88)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_88)));
    __nv_bfloat162 _max_87 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_88);
    amax_2 = _max_87;
    unsigned int mag_2 = __as_u32(amax_2) & 2147450879;
    __nv_bfloat162 col_max_0 = __hmax2_nan(__as_bf16x2(mag_2), __as_bf16x2(mag_2 >> 16));
    unsigned int peer1_2 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(col_max_0), 1);
    col_max_0 = __hmax2_nan(col_max_0, __as_bf16x2(peer1_2));
    unsigned int peer2_2 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(col_max_0), 2);
    col_max_0 = __hmax2_nan(col_max_0, __as_bf16x2(peer2_2));
    unsigned int amax_bits_2 = (__as_u32(col_max_0) & 65535) << 16;
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
    int gcol_0 = bx * 32 + K + warp_0 * 8 + (lane_0 >> 2);
    int out_row_0 = by * 32 + (lane_0 & 3) * 8;
    unsigned int words_2[2];
    uint32_t bf16x2_mul_8;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_8) : "r"(cols_1_0[0]), "r"(inv2_3));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_5) : "r"(cols_1_0[1]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_90;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_90) : "r"(bf16x2_mul_1_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_90) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_89;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_89) : "h"(e4m3x2_bf16x2_8), "h"(_cvt_satfinite_e4m3x2_bf16x2_90));
    words_2[0] = _mov_b32_pack_u16x2_89;
    uint32_t bf16x2_mul_9;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_9) : "r"(cols_1_0[2]), "r"(inv2_3));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_6) : "r"(cols_1_0[3]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_92;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_92) : "r"(bf16x2_mul_1_6));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_6);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_92) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_91;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_91) : "h"(e4m3x2_bf16x2_9), "h"(_cvt_satfinite_e4m3x2_bf16x2_92));
    words_2[1] = _mov_b32_pack_u16x2_91;
    reinterpret_cast<int2*>(col_output + (gcol_0 * M + out_row_0))[0] = reinterpret_cast<int2*>(words_2 + 0)[0];
    if ((lane_0 & 3) == 0) {
        int scale_index_2 = ((gcol_0 >> 7) * (M >> 7) + (by >> 2)) * 512 + (gcol_0 & 31) * 16 + (gcol_0 >> 5 & 3) * 4 + (by & 3);
        *(reinterpret_cast<unsigned char*>(col_scales + scale_index_2) + (0)) = (unsigned char)(scale_4);
    }
}

} // extern "C"
