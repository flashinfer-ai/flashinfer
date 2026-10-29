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
#define SMEM_TILES_OFF 0
#define SMEM_TILES_STAGE_BYTES 8192
#define SMEM_TILES_STRIDE 8192
#define SMEM_TOTAL 8192
#define THREADS 128

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
kernel_cake_gated_act_mxfp8_6d1956b2f454b08e6d34(__nv_bfloat16* __restrict__ gated_input, __nv_bfloat16* __restrict__ grad, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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

    // Kernel setup ops
    uint8_t* tiles = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int tiles_addr = smem + 0;

    // === Task calls (declaration order) ===
    int half = tid & 1;
    int blk = tid >> 1 & 1;
    int row = tid >> 2;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int grow = by * 32 + row;
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
        asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
            : "=r"(_iv_2_0), "=r"(_iv_2_1), "=r"(_iv_2_2), "=r"(_iv_2_3), "=r"(_iv_2_4), "=r"(_iv_2_5), "=r"(_iv_2_6), "=r"(_iv_2_7) : "l"((const void*)(grad + (grow * K + col) + (0))), "l"(0x12F0000000000000ULL) : "memory");
        _vec_load_2[0 + 0] = (uint32_t)_iv_2_0;
        _vec_load_2[0 + 1] = (uint32_t)_iv_2_1;
        _vec_load_2[0 + 2] = (uint32_t)_iv_2_2;
        _vec_load_2[0 + 3] = (uint32_t)_iv_2_3;
        _vec_load_2[0 + 4] = (uint32_t)_iv_2_4;
        _vec_load_2[0 + 5] = (uint32_t)_iv_2_5;
        _vec_load_2[0 + 6] = (uint32_t)_iv_2_6;
        _vec_load_2[0 + 7] = (uint32_t)_iv_2_7;
    }
    unsigned int dgate[8];
    unsigned int dup[8];
    __nv_bfloat162 lowest = __as_bf16x2(_vec_load_0[0]);
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[7]));
    float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(lowest);
    float _min_4;
    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_4) : "f"(_cvt_f32_0.x), "f"(_cvt_f32_0.y));
    int _warp_vote_3 = __all_sync(0xFFFFFFFF, _min_4 > -87.0f);
    if (_warp_vote_3 != 0) {
        float2 cvt_f32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float e0 = approx_exp2((-cvt_f32.x) * 1.442695f);
        float e1 = approx_exp2((-cvt_f32.y) * 1.442695f);
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
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act) : "l"(*(const unsigned long long*)&cvt_f32), "l"(*(const unsigned long long*)&s));
        float2 _f32x2_mul_a_9 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s), s), cvt_f32_2);
        float2 mul;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul) : "l"(*(const unsigned long long*)&_f32x2_mul_a_9), "l"(*(const unsigned long long*)&cvt_f32_1));
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
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_0) : "l"(*(const unsigned long long*)&cvt_f32_0), "l"(*(const unsigned long long*)&s_0));
        float2 _f32x2_mul_a_14 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_0), s_0), cvt_f32_2_0);
        float2 mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&cvt_f32_1_0));
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
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_1) : "l"(*(const unsigned long long*)&cvt_f32_3), "l"(*(const unsigned long long*)&s_1));
        float2 _f32x2_mul_a_19 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1), s_1), cvt_f32_2_1);
        float2 mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&cvt_f32_1_1));
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
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_2) : "l"(*(const unsigned long long*)&cvt_f32_4), "l"(*(const unsigned long long*)&s_2));
        float2 _f32x2_mul_a_24 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_2), s_2), cvt_f32_2_2);
        float2 mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&cvt_f32_1_2));
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
        float _fma_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(-d0_3), "f"(r0_3), "f"(1.0f));
        float _fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(r0_3), "f"(_fma_26), "f"(r0_3));
        float _fma_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(-d1_3), "f"(r1_3), "f"(1.0f));
        float _fma_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(r1_3), "f"(_fma_28), "f"(r1_3));
        float2 s_3 = make_float2(_fma_25, _fma_27);
        float2 act_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_3) : "l"(*(const unsigned long long*)&cvt_f32_5), "l"(*(const unsigned long long*)&s_3));
        float2 _f32x2_mul_a_29 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_3), s_3), cvt_f32_2_3);
        float2 mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&cvt_f32_1_3));
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
        float _fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(-d0_4), "f"(r0_4), "f"(1.0f));
        float _fma_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(r0_4), "f"(_fma_31), "f"(r0_4));
        float _fma_33;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(-d1_4), "f"(r1_4), "f"(1.0f));
        float _fma_32;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(r1_4), "f"(_fma_33), "f"(r1_4));
        float2 s_4 = make_float2(_fma_30, _fma_32);
        float2 act_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_4) : "l"(*(const unsigned long long*)&cvt_f32_6), "l"(*(const unsigned long long*)&s_4));
        float2 _f32x2_mul_a_34 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_4), s_4), cvt_f32_2_4);
        float2 mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_34), "l"(*(const unsigned long long*)&cvt_f32_1_4));
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
        float _fma_36;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(-d0_5), "f"(r0_5), "f"(1.0f));
        float _fma_35;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(r0_5), "f"(_fma_36), "f"(r0_5));
        float _fma_38;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(-d1_5), "f"(r1_5), "f"(1.0f));
        float _fma_37;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(r1_5), "f"(_fma_38), "f"(r1_5));
        float2 s_5 = make_float2(_fma_35, _fma_37);
        float2 act_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_5) : "l"(*(const unsigned long long*)&cvt_f32_7), "l"(*(const unsigned long long*)&s_5));
        float2 _f32x2_mul_a_39 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_5), s_5), cvt_f32_2_5);
        float2 mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_39), "l"(*(const unsigned long long*)&cvt_f32_1_5));
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
        float _fma_41;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(-d0_6), "f"(r0_6), "f"(1.0f));
        float _fma_40;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(r0_6), "f"(_fma_41), "f"(r0_6));
        float _fma_43;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(-d1_6), "f"(r1_6), "f"(1.0f));
        float _fma_42;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(r1_6), "f"(_fma_43), "f"(r1_6));
        float2 s_6 = make_float2(_fma_40, _fma_42);
        float2 act_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_6) : "l"(*(const unsigned long long*)&cvt_f32_8), "l"(*(const unsigned long long*)&s_6));
        float2 _f32x2_mul_a_44 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_6), s_6), cvt_f32_2_6);
        float2 mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&cvt_f32_1_6));
        float2 mul_1_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_6) : "l"(*(const unsigned long long*)&act_6), "l"(*(const unsigned long long*)&cvt_f32_2_6));
        __nv_bfloat162 bf16x2_7 = __float22bfloat162_rn(make_float2(mul_7.x, mul_7.y));
        __nv_bfloat162 bf16x2_1_6 = __float22bfloat162_rn(make_float2(mul_1_6.x, mul_1_6.y));
        dgate[7] = __as_u32(bf16x2_7);
        dup[7] = __as_u32(bf16x2_1_6);
    } else {
        int band = 0;
        float2 cvt_f32_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 cvt_f32_1_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 cvt_f32_2_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float e0_7 = approx_exp2((-cvt_f32_9.x) * 1.442695f);
        float e1_7 = approx_exp2((-cvt_f32_9.y) * 1.442695f);
        float d0_7 = 1.0f + e0_7;
        float d1_7 = 1.0f + e1_7;
        float r0_7 = approx_rcp(d0_7);
        float r1_7 = approx_rcp(d1_7);
        float _fma_45;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(-d0_7), "f"(r0_7), "f"(1.0f));
        float fma;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma) : "f"(r0_7), "f"(_fma_45), "f"(r0_7));
        float _fma_46;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(-d1_7), "f"(r1_7), "f"(1.0f));
        float fma_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1) : "f"(r1_7), "f"(_fma_46), "f"(r1_7));
        float t0 = fma;
        float t1 = fma_1;
        band = band | ((fminf(t0, t1) == 0.0f) ? 1 : 0);
        float2 s_7 = make_float2(max_noftz(t0, 0.0f), max_noftz(t1, 0.0f));
        float2 act_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_7) : "l"(*(const unsigned long long*)&cvt_f32_9), "l"(*(const unsigned long long*)&s_7));
        float2 _f32x2_mul_a_47 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_7, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_7), s_7), cvt_f32_2_7);
        float2 mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_47), "l"(*(const unsigned long long*)&cvt_f32_1_7));
        float2 mul_1_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_7) : "l"(*(const unsigned long long*)&act_7), "l"(*(const unsigned long long*)&cvt_f32_2_7));
        __nv_bfloat162 bf16x2_8 = __float22bfloat162_rn(make_float2(mul_8.x, mul_8.y));
        __nv_bfloat162 bf16x2_1_7 = __float22bfloat162_rn(make_float2(mul_1_7.x, mul_1_7.y));
        dgate[0] = __as_u32(bf16x2_8);
        dup[0] = __as_u32(bf16x2_1_7);
        float2 cvt_f32_9_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 cvt_f32_1_7_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 cvt_f32_2_7_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float e0_7_1 = approx_exp2((-cvt_f32_9_1.x) * 1.442695f);
        float e1_7_1 = approx_exp2((-cvt_f32_9_1.y) * 1.442695f);
        float d0_7_1 = 1.0f + e0_7_1;
        float d1_7_1 = 1.0f + e1_7_1;
        float r0_7_1 = approx_rcp(d0_7_1);
        float r1_7_1 = approx_rcp(d1_7_1);
        float _fma_48;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(-d0_7_1), "f"(r0_7_1), "f"(1.0f));
        float fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_0) : "f"(r0_7_1), "f"(_fma_48), "f"(r0_7_1));
        float _fma_49;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(-d1_7_1), "f"(r1_7_1), "f"(1.0f));
        float fma_1_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_0) : "f"(r1_7_1), "f"(_fma_49), "f"(r1_7_1));
        float t0_0 = fma_0;
        float t1_0 = fma_1_0;
        band = band | ((fminf(t0_0, t1_0) == 0.0f) ? 2 : 0);
        float2 s_7_1 = make_float2(max_noftz(t0_0, 0.0f), max_noftz(t1_0, 0.0f));
        float2 act_7_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_7_1) : "l"(*(const unsigned long long*)&cvt_f32_9_1), "l"(*(const unsigned long long*)&s_7_1));
        float2 _f32x2_mul_a_50 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_7_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_7_1), s_7_1), cvt_f32_2_7_1);
        float2 mul_8_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_8_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_50), "l"(*(const unsigned long long*)&cvt_f32_1_7_1));
        float2 mul_1_7_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_7_1) : "l"(*(const unsigned long long*)&act_7_1), "l"(*(const unsigned long long*)&cvt_f32_2_7_1));
        __nv_bfloat162 bf16x2_8_1 = __float22bfloat162_rn(make_float2(mul_8_1.x, mul_8_1.y));
        __nv_bfloat162 bf16x2_1_7_1 = __float22bfloat162_rn(make_float2(mul_1_7_1.x, mul_1_7_1.y));
        dgate[1] = __as_u32(bf16x2_8_1);
        dup[1] = __as_u32(bf16x2_1_7_1);
        float2 cvt_f32_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 cvt_f32_1_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 cvt_f32_2_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float e0_8 = approx_exp2((-cvt_f32_10.x) * 1.442695f);
        float e1_8 = approx_exp2((-cvt_f32_10.y) * 1.442695f);
        float d0_8 = 1.0f + e0_8;
        float d1_8 = 1.0f + e1_8;
        float r0_8 = approx_rcp(d0_8);
        float r1_8 = approx_rcp(d1_8);
        float _fma_51;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(-d0_8), "f"(r0_8), "f"(1.0f));
        float fma_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_2) : "f"(r0_8), "f"(_fma_51), "f"(r0_8));
        float _fma_52;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(-d1_8), "f"(r1_8), "f"(1.0f));
        float fma_1_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_1) : "f"(r1_8), "f"(_fma_52), "f"(r1_8));
        float t0_1 = fma_2;
        float t1_1 = fma_1_1;
        band = band | ((fminf(t0_1, t1_1) == 0.0f) ? 4 : 0);
        float2 s_8 = make_float2(max_noftz(t0_1, 0.0f), max_noftz(t1_1, 0.0f));
        float2 act_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_8) : "l"(*(const unsigned long long*)&cvt_f32_10), "l"(*(const unsigned long long*)&s_8));
        float2 _f32x2_mul_a_53 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_8, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_8), s_8), cvt_f32_2_8);
        float2 mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_53), "l"(*(const unsigned long long*)&cvt_f32_1_8));
        float2 mul_1_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_8) : "l"(*(const unsigned long long*)&act_8), "l"(*(const unsigned long long*)&cvt_f32_2_8));
        __nv_bfloat162 bf16x2_9 = __float22bfloat162_rn(make_float2(mul_9.x, mul_9.y));
        __nv_bfloat162 bf16x2_1_8 = __float22bfloat162_rn(make_float2(mul_1_8.x, mul_1_8.y));
        dgate[2] = __as_u32(bf16x2_9);
        dup[2] = __as_u32(bf16x2_1_8);
        float2 cvt_f32_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 cvt_f32_1_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 cvt_f32_2_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float e0_9 = approx_exp2((-cvt_f32_11.x) * 1.442695f);
        float e1_9 = approx_exp2((-cvt_f32_11.y) * 1.442695f);
        float d0_9 = 1.0f + e0_9;
        float d1_9 = 1.0f + e1_9;
        float r0_9 = approx_rcp(d0_9);
        float r1_9 = approx_rcp(d1_9);
        float _fma_54;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(-d0_9), "f"(r0_9), "f"(1.0f));
        float fma_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_3) : "f"(r0_9), "f"(_fma_54), "f"(r0_9));
        float _fma_55;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(-d1_9), "f"(r1_9), "f"(1.0f));
        float fma_1_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_2) : "f"(r1_9), "f"(_fma_55), "f"(r1_9));
        float t0_2 = fma_3;
        float t1_2 = fma_1_2;
        band = band | ((fminf(t0_2, t1_2) == 0.0f) ? 8 : 0);
        float2 s_9 = make_float2(max_noftz(t0_2, 0.0f), max_noftz(t1_2, 0.0f));
        float2 act_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_9) : "l"(*(const unsigned long long*)&cvt_f32_11), "l"(*(const unsigned long long*)&s_9));
        float2 _f32x2_mul_a_56 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_9, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_9), s_9), cvt_f32_2_9);
        float2 mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_56), "l"(*(const unsigned long long*)&cvt_f32_1_9));
        float2 mul_1_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_9) : "l"(*(const unsigned long long*)&act_9), "l"(*(const unsigned long long*)&cvt_f32_2_9));
        __nv_bfloat162 bf16x2_10 = __float22bfloat162_rn(make_float2(mul_10.x, mul_10.y));
        __nv_bfloat162 bf16x2_1_9 = __float22bfloat162_rn(make_float2(mul_1_9.x, mul_1_9.y));
        dgate[3] = __as_u32(bf16x2_10);
        dup[3] = __as_u32(bf16x2_1_9);
        float2 cvt_f32_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 cvt_f32_1_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float2 cvt_f32_2_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[4]));
        float e0_10 = approx_exp2((-cvt_f32_12.x) * 1.442695f);
        float e1_10 = approx_exp2((-cvt_f32_12.y) * 1.442695f);
        float d0_10 = 1.0f + e0_10;
        float d1_10 = 1.0f + e1_10;
        float r0_10 = approx_rcp(d0_10);
        float r1_10 = approx_rcp(d1_10);
        float _fma_57;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(-d0_10), "f"(r0_10), "f"(1.0f));
        float fma_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_4) : "f"(r0_10), "f"(_fma_57), "f"(r0_10));
        float _fma_58;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(-d1_10), "f"(r1_10), "f"(1.0f));
        float fma_1_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_3) : "f"(r1_10), "f"(_fma_58), "f"(r1_10));
        float t0_3 = fma_4;
        float t1_3 = fma_1_3;
        band = band | ((fminf(t0_3, t1_3) == 0.0f) ? 16 : 0);
        float2 s_10 = make_float2(max_noftz(t0_3, 0.0f), max_noftz(t1_3, 0.0f));
        float2 act_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_10) : "l"(*(const unsigned long long*)&cvt_f32_12), "l"(*(const unsigned long long*)&s_10));
        float2 _f32x2_mul_a_59 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_10, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_10), s_10), cvt_f32_2_10);
        float2 mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_59), "l"(*(const unsigned long long*)&cvt_f32_1_10));
        float2 mul_1_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_10) : "l"(*(const unsigned long long*)&act_10), "l"(*(const unsigned long long*)&cvt_f32_2_10));
        __nv_bfloat162 bf16x2_11 = __float22bfloat162_rn(make_float2(mul_11.x, mul_11.y));
        __nv_bfloat162 bf16x2_1_10 = __float22bfloat162_rn(make_float2(mul_1_10.x, mul_1_10.y));
        dgate[4] = __as_u32(bf16x2_11);
        dup[4] = __as_u32(bf16x2_1_10);
        float2 cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 cvt_f32_1_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float2 cvt_f32_2_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[5]));
        float e0_11 = approx_exp2((-cvt_f32_13.x) * 1.442695f);
        float e1_11 = approx_exp2((-cvt_f32_13.y) * 1.442695f);
        float d0_11 = 1.0f + e0_11;
        float d1_11 = 1.0f + e1_11;
        float r0_11 = approx_rcp(d0_11);
        float r1_11 = approx_rcp(d1_11);
        float _fma_60;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(-d0_11), "f"(r0_11), "f"(1.0f));
        float fma_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_5) : "f"(r0_11), "f"(_fma_60), "f"(r0_11));
        float _fma_61;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(-d1_11), "f"(r1_11), "f"(1.0f));
        float fma_1_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_4) : "f"(r1_11), "f"(_fma_61), "f"(r1_11));
        float t0_4 = fma_5;
        float t1_4 = fma_1_4;
        band = band | ((fminf(t0_4, t1_4) == 0.0f) ? 32 : 0);
        float2 s_11 = make_float2(max_noftz(t0_4, 0.0f), max_noftz(t1_4, 0.0f));
        float2 act_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_11) : "l"(*(const unsigned long long*)&cvt_f32_13), "l"(*(const unsigned long long*)&s_11));
        float2 _f32x2_mul_a_62 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_11, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_11), s_11), cvt_f32_2_11);
        float2 mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&cvt_f32_1_11));
        float2 mul_1_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_11) : "l"(*(const unsigned long long*)&act_11), "l"(*(const unsigned long long*)&cvt_f32_2_11));
        __nv_bfloat162 bf16x2_12 = __float22bfloat162_rn(make_float2(mul_12.x, mul_12.y));
        __nv_bfloat162 bf16x2_1_11 = __float22bfloat162_rn(make_float2(mul_1_11.x, mul_1_11.y));
        dgate[5] = __as_u32(bf16x2_12);
        dup[5] = __as_u32(bf16x2_1_11);
        float2 cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 cvt_f32_1_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float2 cvt_f32_2_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[6]));
        float e0_12 = approx_exp2((-cvt_f32_14.x) * 1.442695f);
        float e1_12 = approx_exp2((-cvt_f32_14.y) * 1.442695f);
        float d0_12 = 1.0f + e0_12;
        float d1_12 = 1.0f + e1_12;
        float r0_12 = approx_rcp(d0_12);
        float r1_12 = approx_rcp(d1_12);
        float _fma_63;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(-d0_12), "f"(r0_12), "f"(1.0f));
        float fma_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_6) : "f"(r0_12), "f"(_fma_63), "f"(r0_12));
        float _fma_64;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(-d1_12), "f"(r1_12), "f"(1.0f));
        float fma_1_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_5) : "f"(r1_12), "f"(_fma_64), "f"(r1_12));
        float t0_5 = fma_6;
        float t1_5 = fma_1_5;
        band = band | ((fminf(t0_5, t1_5) == 0.0f) ? 64 : 0);
        float2 s_12 = make_float2(max_noftz(t0_5, 0.0f), max_noftz(t1_5, 0.0f));
        float2 act_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_12) : "l"(*(const unsigned long long*)&cvt_f32_14), "l"(*(const unsigned long long*)&s_12));
        float2 _f32x2_mul_a_65 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_12, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_12), s_12), cvt_f32_2_12);
        float2 mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_65), "l"(*(const unsigned long long*)&cvt_f32_1_12));
        float2 mul_1_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_12) : "l"(*(const unsigned long long*)&act_12), "l"(*(const unsigned long long*)&cvt_f32_2_12));
        __nv_bfloat162 bf16x2_13 = __float22bfloat162_rn(make_float2(mul_13.x, mul_13.y));
        __nv_bfloat162 bf16x2_1_12 = __float22bfloat162_rn(make_float2(mul_1_12.x, mul_1_12.y));
        dgate[6] = __as_u32(bf16x2_13);
        dup[6] = __as_u32(bf16x2_1_12);
        float2 cvt_f32_15 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 cvt_f32_1_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float2 cvt_f32_2_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[7]));
        float e0_13 = approx_exp2((-cvt_f32_15.x) * 1.442695f);
        float e1_13 = approx_exp2((-cvt_f32_15.y) * 1.442695f);
        float d0_13 = 1.0f + e0_13;
        float d1_13 = 1.0f + e1_13;
        float r0_13 = approx_rcp(d0_13);
        float r1_13 = approx_rcp(d1_13);
        float _fma_66;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(-d0_13), "f"(r0_13), "f"(1.0f));
        float fma_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_7) : "f"(r0_13), "f"(_fma_66), "f"(r0_13));
        float _fma_67;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(-d1_13), "f"(r1_13), "f"(1.0f));
        float fma_1_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_6) : "f"(r1_13), "f"(_fma_67), "f"(r1_13));
        float t0_6 = fma_7;
        float t1_6 = fma_1_6;
        band = band | ((fminf(t0_6, t1_6) == 0.0f) ? 128 : 0);
        float2 s_13 = make_float2(max_noftz(t0_6, 0.0f), max_noftz(t1_6, 0.0f));
        float2 act_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_13) : "l"(*(const unsigned long long*)&cvt_f32_15), "l"(*(const unsigned long long*)&s_13));
        float2 _f32x2_mul_a_68 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_13, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_13), s_13), cvt_f32_2_13);
        float2 mul_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&cvt_f32_1_13));
        float2 mul_1_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_13) : "l"(*(const unsigned long long*)&act_13), "l"(*(const unsigned long long*)&cvt_f32_2_13));
        __nv_bfloat162 bf16x2_14 = __float22bfloat162_rn(make_float2(mul_14.x, mul_14.y));
        __nv_bfloat162 bf16x2_1_13 = __float22bfloat162_rn(make_float2(mul_1_13.x, mul_1_13.y));
        dgate[7] = __as_u32(bf16x2_14);
        dup[7] = __as_u32(bf16x2_1_13);
        if (band != 0) {
            #pragma unroll 1
            for (int i = 0; i < 8; i++) {
                if ((band >> i & 1) != 0) {
                    unsigned int _vec_load_3[1];
                    {
                        _vec_load_3[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index + 2 * i) + 0);
                    }
                    unsigned int _vec_load_4[1];
                    {
                        _vec_load_4[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index + K + 2 * i) + 0);
                    }
                    unsigned int _vec_load_5[1];
                    {
                        _vec_load_5[0] = *reinterpret_cast<const unsigned int*>(grad + (grow * K + col + 2 * i) + 0);
                    }
                    float2 cvt_f32_16 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
                    float2 cvt_f32_1_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[0]));
                    float2 cvt_f32_2_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[0]));
                    float e0_14 = approx_exp2((-cvt_f32_16.x) * 1.442695f);
                    float e1_14 = approx_exp2((-cvt_f32_16.y) * 1.442695f);
                    float d0_14 = 1.0f + e0_14;
                    float d1_14 = 1.0f + e1_14;
                    float r0_14 = approx_rcp(d0_14);
                    float r1_14 = approx_rcp(d1_14);
                    float _fma_69;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(-d0_14), "f"(r0_14), "f"(1.0f));
                    float fma_8;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_8) : "f"(r0_14), "f"(_fma_69), "f"(r0_14));
                    float _fma_70;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(-d1_14), "f"(r1_14), "f"(1.0f));
                    float fma_1_7;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_7) : "f"(r1_14), "f"(_fma_70), "f"(r1_14));
                    float t0_7 = fma_8;
                    float t1_7 = fma_1_7;
                    float d4_1a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a) : "f"(d0_14), "f"(0.25f));
                    float r4_1a = approx_rcp(d4_1a);
                    float _fma_72;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(-d4_1a), "f"(r4_1a), "f"(1.0f));
                    float _fma_71;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(r4_1a), "f"(_fma_72), "f"(r4_1a));
                    float big_1a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a) : "f"(mul_f32_rn_noftz(_fma_71, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_73;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(big_1a), "f"(8.388608e+06f));
                    float whole_1a;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a) : "f"(_add_73), "f"(8.388608e+06f));
                    float above_1a;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a) : "f"(-d4_1a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_75;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_75) : "f"(big_1a), "f"(whole_1a));
                    float _abs_74;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_74) : "f"(_sub_75));
                    float _add_76;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(big_1a), "f"(0.5f));
                    float _sub_77;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_77) : "f"(big_1a), "f"(0.5f));
                    float rounded_1a = ((_abs_74 == 0.5f) ? ((above_1a > 0.0f) ? _add_76 : _sub_77) : whole_1a);
                    float u0 = ((d0_14 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a, 7.888609e-31f), 1.7763568e-15f) : t0_7);
                    float d4_1a_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_0) : "f"(d1_14), "f"(0.25f));
                    float r4_1a_0 = approx_rcp(d4_1a_0);
                    float _fma_79;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(-d4_1a_0), "f"(r4_1a_0), "f"(1.0f));
                    float _fma_78;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_78) : "f"(r4_1a_0), "f"(_fma_79), "f"(r4_1a_0));
                    float big_1a_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_0) : "f"(mul_f32_rn_noftz(_fma_78, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_80;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_80) : "f"(big_1a_0), "f"(8.388608e+06f));
                    float whole_1a_0;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_0) : "f"(_add_80), "f"(8.388608e+06f));
                    float above_1a_0;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_0) : "f"(-d4_1a_0), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_0, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_82;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_82) : "f"(big_1a_0), "f"(whole_1a_0));
                    float _abs_81;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_81) : "f"(_sub_82));
                    float _add_83;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(big_1a_0), "f"(0.5f));
                    float _sub_84;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_84) : "f"(big_1a_0), "f"(0.5f));
                    float rounded_1a_0 = ((_abs_81 == 0.5f) ? ((above_1a_0 > 0.0f) ? _add_83 : _sub_84) : whole_1a_0);
                    float u1 = ((d1_14 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_0, 7.888609e-31f), 1.7763568e-15f) : t1_7);
                    float2 s_14 = make_float2(max_noftz(u0, 0.0f), max_noftz(u1, 0.0f));
                    float2 act_14;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_14) : "l"(*(const unsigned long long*)&cvt_f32_16), "l"(*(const unsigned long long*)&s_14));
                    float2 _f32x2_mul_a_85 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_14, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_14), s_14), cvt_f32_2_14);
                    float2 mul_15;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_85), "l"(*(const unsigned long long*)&cvt_f32_1_14));
                    float2 mul_1_14;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_14) : "l"(*(const unsigned long long*)&act_14), "l"(*(const unsigned long long*)&cvt_f32_2_14));
                    __nv_bfloat162 bf16x2_15 = __float22bfloat162_rn(make_float2(mul_15.x, mul_15.y));
                    __nv_bfloat162 bf16x2_1_14 = __float22bfloat162_rn(make_float2(mul_1_14.x, mul_1_14.y));
                    unsigned int dgate_new = __as_u32(bf16x2_15);
                    unsigned int dup_new = __as_u32(bf16x2_1_14);
                    dgate[0] = ((i == 0) ? dgate_new : dgate[0]);
                    dup[0] = ((i == 0) ? dup_new : dup[0]);
                    dgate[1] = ((i == 1) ? dgate_new : dgate[1]);
                    dup[1] = ((i == 1) ? dup_new : dup[1]);
                    dgate[2] = ((i == 2) ? dgate_new : dgate[2]);
                    dup[2] = ((i == 2) ? dup_new : dup[2]);
                    dgate[3] = ((i == 3) ? dgate_new : dgate[3]);
                    dup[3] = ((i == 3) ? dup_new : dup[3]);
                    dgate[4] = ((i == 4) ? dgate_new : dgate[4]);
                    dup[4] = ((i == 4) ? dup_new : dup[4]);
                    dgate[5] = ((i == 5) ? dgate_new : dgate[5]);
                    dup[5] = ((i == 5) ? dup_new : dup[5]);
                    dgate[6] = ((i == 6) ? dgate_new : dgate[6]);
                    dup[6] = ((i == 6) ? dup_new : dup[6]);
                    dgate[7] = ((i == 7) ? dgate_new : dgate[7]);
                    dup[7] = ((i == 7) ? dup_new : dup[7]);
                }
            }
        }
    }
    unsigned int swizzle = (unsigned int)(row >> 2 & 6 | row & 1);
    unsigned int row_base = tiles_addr + (unsigned int)(row * 128);
    unsigned int chunk = (unsigned int)((tid & 3) * 2);
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + ((chunk ^ swizzle) << 4)), "r"(dgate[0]), "r"(dgate[1]), "r"(dgate[2]), "r"(dgate[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + ((chunk + 1 ^ swizzle) << 4)), "r"(dgate[4]), "r"(dgate[5]), "r"(dgate[6]), "r"(dgate[7]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + (4096 + ((chunk ^ swizzle) << 4))), "r"(dup[0]), "r"(dup[1]), "r"(dup[2]), "r"(dup[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + (4096 + ((chunk + 1 ^ swizzle) << 4))), "r"(dup[4]), "r"(dup[5]), "r"(dup[6]), "r"(dup[7]) : "memory");
    int scale_cols = K >> 6;
    int scale_col = bx * 2 + blk;
    int out_index = grow * (2 * K) + col;
    __nv_bfloat162 amax = __as_bf16x2(0);
    __nv_bfloat162 _hxa_87 = amax;
    __nv_bfloat162 _hxb_87 = __as_bf16x2(dgate[0]);
    unsigned int _hxr_87;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_87) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_87)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_87)));
    __nv_bfloat162 _max_86 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_87);
    amax = _max_86;
    __nv_bfloat162 _hxa_89 = amax;
    __nv_bfloat162 _hxb_89 = __as_bf16x2(dgate[1]);
    unsigned int _hxr_89;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_89) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_89)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_89)));
    __nv_bfloat162 _max_88 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_89);
    amax = _max_88;
    __nv_bfloat162 _hxa_91 = amax;
    __nv_bfloat162 _hxb_91 = __as_bf16x2(dgate[2]);
    unsigned int _hxr_91;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_91) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_91)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_91)));
    __nv_bfloat162 _max_90 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_91);
    amax = _max_90;
    __nv_bfloat162 _hxa_93 = amax;
    __nv_bfloat162 _hxb_93 = __as_bf16x2(dgate[3]);
    unsigned int _hxr_93;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_93) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_93)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_93)));
    __nv_bfloat162 _max_92 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_93);
    amax = _max_92;
    __nv_bfloat162 _hxa_95 = amax;
    __nv_bfloat162 _hxb_95 = __as_bf16x2(dgate[4]);
    unsigned int _hxr_95;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_95) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_95)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_95)));
    __nv_bfloat162 _max_94 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_95);
    amax = _max_94;
    __nv_bfloat162 _hxa_97 = amax;
    __nv_bfloat162 _hxb_97 = __as_bf16x2(dgate[5]);
    unsigned int _hxr_97;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_97) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_97)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_97)));
    __nv_bfloat162 _max_96 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_97);
    amax = _max_96;
    __nv_bfloat162 _hxa_99 = amax;
    __nv_bfloat162 _hxb_99 = __as_bf16x2(dgate[6]);
    unsigned int _hxr_99;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_99) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_99)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_99)));
    __nv_bfloat162 _max_98 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_99);
    amax = _max_98;
    __nv_bfloat162 _hxa_101 = amax;
    __nv_bfloat162 _hxb_101 = __as_bf16x2(dgate[7]);
    unsigned int _hxr_101;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_101) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_101)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_101)));
    __nv_bfloat162 _max_100 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_101);
    amax = _max_100;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_103;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_103) : "r"(bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_103) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_102;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_102) : "h"(e4m3x2_bf16x2), "h"(_cvt_satfinite_e4m3x2_bf16x2_103));
    words[0] = _mov_b32_pack_u16x2_102;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_105;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_105) : "r"(bf16x2_mul_1_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_105) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_104;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_104) : "h"(e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_105));
    words[1] = _mov_b32_pack_u16x2_104;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_107;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_107) : "r"(bf16x2_mul_1_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_107) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_106;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_106) : "h"(e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_107));
    words[2] = _mov_b32_pack_u16x2_106;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_109;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_109) : "r"(bf16x2_mul_1_2));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_2);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_109) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_108;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_108) : "h"(e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_109));
    words[3] = _mov_b32_pack_u16x2_108;
    reinterpret_cast<int4*>(row_output + out_index)[0] = reinterpret_cast<int4*>(words + 0)[0];
    __nv_bfloat162 amax_0 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_111 = amax_0;
    __nv_bfloat162 _hxb_111 = __as_bf16x2(dup[0]);
    unsigned int _hxr_111;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_111) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_111)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_111)));
    __nv_bfloat162 _max_110 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_111);
    amax_0 = _max_110;
    __nv_bfloat162 _hxa_113 = amax_0;
    __nv_bfloat162 _hxb_113 = __as_bf16x2(dup[1]);
    unsigned int _hxr_113;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_113) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_113)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_113)));
    __nv_bfloat162 _max_112 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_113);
    amax_0 = _max_112;
    __nv_bfloat162 _hxa_115 = amax_0;
    __nv_bfloat162 _hxb_115 = __as_bf16x2(dup[2]);
    unsigned int _hxr_115;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_115) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_115)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_115)));
    __nv_bfloat162 _max_114 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_115);
    amax_0 = _max_114;
    __nv_bfloat162 _hxa_117 = amax_0;
    __nv_bfloat162 _hxb_117 = __as_bf16x2(dup[3]);
    unsigned int _hxr_117;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_117) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_117)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_117)));
    __nv_bfloat162 _max_116 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_117);
    amax_0 = _max_116;
    __nv_bfloat162 _hxa_119 = amax_0;
    __nv_bfloat162 _hxb_119 = __as_bf16x2(dup[4]);
    unsigned int _hxr_119;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_119) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_119)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_119)));
    __nv_bfloat162 _max_118 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_119);
    amax_0 = _max_118;
    __nv_bfloat162 _hxa_121 = amax_0;
    __nv_bfloat162 _hxb_121 = __as_bf16x2(dup[5]);
    unsigned int _hxr_121;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_121) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_121)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_121)));
    __nv_bfloat162 _max_120 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_121);
    amax_0 = _max_120;
    __nv_bfloat162 _hxa_123 = amax_0;
    __nv_bfloat162 _hxb_123 = __as_bf16x2(dup[6]);
    unsigned int _hxr_123;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_123) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_123)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_123)));
    __nv_bfloat162 _max_122 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_123);
    amax_0 = _max_122;
    __nv_bfloat162 _hxa_125 = amax_0;
    __nv_bfloat162 _hxb_125 = __as_bf16x2(dup[7]);
    unsigned int _hxr_125;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_125) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_125)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_125)));
    __nv_bfloat162 _max_124 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_125);
    amax_0 = _max_124;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_127;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_127) : "r"(bf16x2_mul_1_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_127) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_126;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_126) : "h"(e4m3x2_bf16x2_3), "h"(_cvt_satfinite_e4m3x2_bf16x2_127));
    words_11[0] = _mov_b32_pack_u16x2_126;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_129;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_129) : "r"(bf16x2_mul_1_4));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_4);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_129) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_128;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_128) : "h"(e4m3x2_bf16x2_4), "h"(_cvt_satfinite_e4m3x2_bf16x2_129));
    words_11[1] = _mov_b32_pack_u16x2_128;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_131;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_131) : "r"(bf16x2_mul_1_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_131) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_130;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_130) : "h"(e4m3x2_bf16x2_5), "h"(_cvt_satfinite_e4m3x2_bf16x2_131));
    words_11[2] = _mov_b32_pack_u16x2_130;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_133;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_133) : "r"(bf16x2_mul_1_6));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_6);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_133) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_132;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_132) : "h"(e4m3x2_bf16x2_6), "h"(_cvt_satfinite_e4m3x2_bf16x2_133));
    words_11[3] = _mov_b32_pack_u16x2_132;
    reinterpret_cast<int4*>(row_output + (out_index + K))[0] = reinterpret_cast<int4*>(words_11 + 0)[0];
    __syncthreads();
    int lane_1 = tid & 31;
    int warp_1 = tid >> 5;
    int k8 = lane_1 & 7;
    int srow = (k8 >> 1) * 8 + (lane_1 >> 3) * 2 + (k8 & 1);
    unsigned int sswizzle = (unsigned int)(srow >> 2 & 6 | srow & 1);
    unsigned int cols_1[8];
    unsigned int strip_chunk0_1_0 = (unsigned int)(warp_1 * 2);
    unsigned int strip_src0_1_0 = tiles_addr + (unsigned int)(srow * 128) + ((strip_chunk0_1_0 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_1[0]), "=r"(cols_1[1]), "=r"(cols_1[2]), "=r"(cols_1[3])
        : "r"(strip_src0_1_0)
        : "memory");
    unsigned int strip_chunk0_1_1 = (unsigned int)(warp_1 * 2 + 1);
    unsigned int strip_src0_1_1 = tiles_addr + (unsigned int)(srow * 128) + ((strip_chunk0_1_1 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_1[4]), "=r"(cols_1[5]), "=r"(cols_1[6]), "=r"(cols_1[7])
        : "r"(strip_src0_1_1)
        : "memory");
    __nv_bfloat162 amax_a = __as_bf16x2(0);
    __nv_bfloat162 amax_b = __as_bf16x2(0);
    __nv_bfloat162 _hxa_135 = amax_a;
    __nv_bfloat162 _hxb_135 = __as_bf16x2(cols_1[0]);
    unsigned int _hxr_135;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_135) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_135)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_135)));
    __nv_bfloat162 _max_134 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_135);
    amax_a = _max_134;
    __nv_bfloat162 _hxa_137 = amax_b;
    __nv_bfloat162 _hxb_137 = __as_bf16x2(cols_1[4]);
    unsigned int _hxr_137;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_137) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_137)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_137)));
    __nv_bfloat162 _max_136 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_137);
    amax_b = _max_136;
    __nv_bfloat162 _hxa_139 = amax_a;
    __nv_bfloat162 _hxb_139 = __as_bf16x2(cols_1[1]);
    unsigned int _hxr_139;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_139) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_139)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_139)));
    __nv_bfloat162 _max_138 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_139);
    amax_a = _max_138;
    __nv_bfloat162 _hxa_141 = amax_b;
    __nv_bfloat162 _hxb_141 = __as_bf16x2(cols_1[5]);
    unsigned int _hxr_141;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_141) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_141)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_141)));
    __nv_bfloat162 _max_140 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_141);
    amax_b = _max_140;
    __nv_bfloat162 _hxa_143 = amax_a;
    __nv_bfloat162 _hxb_143 = __as_bf16x2(cols_1[2]);
    unsigned int _hxr_143;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_143) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_143)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_143)));
    __nv_bfloat162 _max_142 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_143);
    amax_a = _max_142;
    __nv_bfloat162 _hxa_145 = amax_b;
    __nv_bfloat162 _hxb_145 = __as_bf16x2(cols_1[6]);
    unsigned int _hxr_145;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_145) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_145)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_145)));
    __nv_bfloat162 _max_144 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_145);
    amax_b = _max_144;
    __nv_bfloat162 _hxa_147 = amax_a;
    __nv_bfloat162 _hxb_147 = __as_bf16x2(cols_1[3]);
    unsigned int _hxr_147;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_147) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_147)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_147)));
    __nv_bfloat162 _max_146 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_147);
    amax_a = _max_146;
    __nv_bfloat162 _hxa_149 = amax_b;
    __nv_bfloat162 _hxb_149 = __as_bf16x2(cols_1[7]);
    unsigned int _hxr_149;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_149) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_149)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_149)));
    __nv_bfloat162 _max_148 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_149);
    amax_b = _max_148;
    unsigned int mag_a = __as_u32(amax_a) & 2147450879;
    unsigned int mag_b = __as_u32(amax_b) & 2147450879;
    unsigned int lows_ab = mag_a & 65535 | mag_b << 16;
    unsigned int highs_ab = mag_a >> 16 | mag_b & 4294901760u;
    __nv_bfloat162 col_max = __hmax2_nan(__as_bf16x2(lows_ab), __as_bf16x2(highs_ab));
    unsigned int peer1 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(col_max), 1);
    col_max = __hmax2_nan(col_max, __as_bf16x2(peer1));
    unsigned int peer2 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(col_max), 2);
    col_max = __hmax2_nan(col_max, __as_bf16x2(peer2));
    unsigned int col_max_bits = __as_u32(col_max);
    unsigned int amax_bits_1 = (col_max_bits & 65535) << 16;
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
    unsigned int amax_bits_2 = (col_max_bits >> 16 & 65535) << 16;
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
    int gcol_a = bx * 64 + warp_1 * 16 + (lane_1 >> 2);
    int gcol_b = gcol_a + 8;
    int out_row = by * 32 + (lane_1 & 3) * 8;
    unsigned int words_a[2];
    unsigned int words_b[2];
    uint32_t bf16x2_mul_8;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_8) : "r"(cols_1[0]), "r"(inv2_1));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_7) : "r"(cols_1[1]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_151;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_151) : "r"(bf16x2_mul_1_7));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_7);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_151) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_150;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_150) : "h"(e4m3x2_bf16x2_7), "h"(_cvt_satfinite_e4m3x2_bf16x2_151));
    words_a[0] = _mov_b32_pack_u16x2_150;
    uint32_t bf16x2_mul_9;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_9) : "r"(cols_1[2]), "r"(inv2_1));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_8) : "r"(cols_1[3]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_153;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_153) : "r"(bf16x2_mul_1_8));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_8);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_153) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_152;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_152) : "h"(e4m3x2_bf16x2_8), "h"(_cvt_satfinite_e4m3x2_bf16x2_153));
    words_a[1] = _mov_b32_pack_u16x2_152;
    uint32_t bf16x2_mul_10;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_10) : "r"(cols_1[4]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_9) : "r"(cols_1[5]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_155;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_155) : "r"(bf16x2_mul_1_9));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_9);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_155) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_154;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_154) : "h"(e4m3x2_bf16x2_9), "h"(_cvt_satfinite_e4m3x2_bf16x2_155));
    words_b[0] = _mov_b32_pack_u16x2_154;
    uint32_t bf16x2_mul_11;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_11) : "r"(cols_1[6]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_10) : "r"(cols_1[7]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_157;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_157) : "r"(bf16x2_mul_1_10));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_10);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_157) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_156;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_156) : "h"(e4m3x2_bf16x2_10), "h"(_cvt_satfinite_e4m3x2_bf16x2_157));
    words_b[1] = _mov_b32_pack_u16x2_156;
    reinterpret_cast<int2*>(col_output + (gcol_a * M + out_row))[0] = reinterpret_cast<int2*>(words_a + 0)[0];
    reinterpret_cast<int2*>(col_output + (gcol_b * M + out_row))[0] = reinterpret_cast<int2*>(words_b + 0)[0];
    if ((lane_1 & 3) == 0) {
        int row_tiles = M >> 7;
        int col_scale_a = ((gcol_a >> 7) * row_tiles + (by >> 2)) * 512 + (gcol_a & 31) * 16 + (gcol_a >> 5 & 3) * 4 + (by & 3);
        int col_scale_b = ((gcol_b >> 7) * row_tiles + (by >> 2)) * 512 + (gcol_b & 31) * 16 + (gcol_b >> 5 & 3) * 4 + (by & 3);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_a) + (0)) = (unsigned char)(scale_1);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b) + (0)) = (unsigned char)(scale_2);
    }
    int lane_0 = tid & 31;
    int warp_0 = tid >> 5;
    int k8_0 = lane_0 & 7;
    int srow_0 = (k8_0 >> 1) * 8 + (lane_0 >> 3) * 2 + (k8_0 & 1);
    unsigned int sswizzle_0 = (unsigned int)(srow_0 >> 2 & 6 | srow_0 & 1);
    unsigned int cols_1_0[8];
    unsigned int strip_chunk0_1_2 = (unsigned int)(warp_0 * 2);
    unsigned int strip_src0_1_2 = tiles_addr + 4096 + (unsigned int)(srow_0 * 128) + ((strip_chunk0_1_2 ^ sswizzle_0) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_1_0[0]), "=r"(cols_1_0[1]), "=r"(cols_1_0[2]), "=r"(cols_1_0[3])
        : "r"(strip_src0_1_2)
        : "memory");
    unsigned int strip_chunk0_1_3 = (unsigned int)(warp_0 * 2 + 1);
    unsigned int strip_src0_1_3 = tiles_addr + 4096 + (unsigned int)(srow_0 * 128) + ((strip_chunk0_1_3 ^ sswizzle_0) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_1_0[4]), "=r"(cols_1_0[5]), "=r"(cols_1_0[6]), "=r"(cols_1_0[7])
        : "r"(strip_src0_1_3)
        : "memory");
    __nv_bfloat162 amax_a_0 = __as_bf16x2(0);
    __nv_bfloat162 amax_b_0 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_159 = amax_a_0;
    __nv_bfloat162 _hxb_159 = __as_bf16x2(cols_1_0[0]);
    unsigned int _hxr_159;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_159) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_159)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_159)));
    __nv_bfloat162 _max_158 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_159);
    amax_a_0 = _max_158;
    __nv_bfloat162 _hxa_161 = amax_b_0;
    __nv_bfloat162 _hxb_161 = __as_bf16x2(cols_1_0[4]);
    unsigned int _hxr_161;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_161) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_161)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_161)));
    __nv_bfloat162 _max_160 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_161);
    amax_b_0 = _max_160;
    __nv_bfloat162 _hxa_163 = amax_a_0;
    __nv_bfloat162 _hxb_163 = __as_bf16x2(cols_1_0[1]);
    unsigned int _hxr_163;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_163) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_163)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_163)));
    __nv_bfloat162 _max_162 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_163);
    amax_a_0 = _max_162;
    __nv_bfloat162 _hxa_165 = amax_b_0;
    __nv_bfloat162 _hxb_165 = __as_bf16x2(cols_1_0[5]);
    unsigned int _hxr_165;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_165) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_165)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_165)));
    __nv_bfloat162 _max_164 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_165);
    amax_b_0 = _max_164;
    __nv_bfloat162 _hxa_167 = amax_a_0;
    __nv_bfloat162 _hxb_167 = __as_bf16x2(cols_1_0[2]);
    unsigned int _hxr_167;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_167) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_167)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_167)));
    __nv_bfloat162 _max_166 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_167);
    amax_a_0 = _max_166;
    __nv_bfloat162 _hxa_169 = amax_b_0;
    __nv_bfloat162 _hxb_169 = __as_bf16x2(cols_1_0[6]);
    unsigned int _hxr_169;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_169) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_169)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_169)));
    __nv_bfloat162 _max_168 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_169);
    amax_b_0 = _max_168;
    __nv_bfloat162 _hxa_171 = amax_a_0;
    __nv_bfloat162 _hxb_171 = __as_bf16x2(cols_1_0[3]);
    unsigned int _hxr_171;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_171) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_171)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_171)));
    __nv_bfloat162 _max_170 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_171);
    amax_a_0 = _max_170;
    __nv_bfloat162 _hxa_173 = amax_b_0;
    __nv_bfloat162 _hxb_173 = __as_bf16x2(cols_1_0[7]);
    unsigned int _hxr_173;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_173) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_173)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_173)));
    __nv_bfloat162 _max_172 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_173);
    amax_b_0 = _max_172;
    unsigned int mag_a_0 = __as_u32(amax_a_0) & 2147450879;
    unsigned int mag_b_0 = __as_u32(amax_b_0) & 2147450879;
    unsigned int lows_ab_0 = mag_a_0 & 65535 | mag_b_0 << 16;
    unsigned int highs_ab_0 = mag_a_0 >> 16 | mag_b_0 & 4294901760u;
    __nv_bfloat162 col_max_0 = __hmax2_nan(__as_bf16x2(lows_ab_0), __as_bf16x2(highs_ab_0));
    unsigned int peer1_0 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(col_max_0), 1);
    col_max_0 = __hmax2_nan(col_max_0, __as_bf16x2(peer1_0));
    unsigned int peer2_0 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(col_max_0), 2);
    col_max_0 = __hmax2_nan(col_max_0, __as_bf16x2(peer2_0));
    unsigned int col_max_bits_0 = __as_u32(col_max_0);
    unsigned int amax_bits_3 = (col_max_bits_0 & 65535) << 16;
    int scale_i32_3 = (int)(amax_bits_3 + 2031616 >> 23) - 8;
    if (scale_i32_3 < 0) {
        scale_i32_3 = 0;
    }
    if ((amax_bits_3 & 2139095040) == 2139095040) {
        scale_i32_3 = 255;
    }
    unsigned int scale_3 = (unsigned int)scale_i32_3;
    unsigned int inv_3 = 254 - scale_3 << 7;
    if (scale_3 == 255) {
        inv_3 = 32704;
    }
    unsigned int inv2_3 = inv_3 | inv_3 << 16;
    unsigned int amax_bits_4 = (col_max_bits_0 >> 16 & 65535) << 16;
    int scale_i32_4 = (int)(amax_bits_4 + 2031616 >> 23) - 8;
    if (scale_i32_4 < 0) {
        scale_i32_4 = 0;
    }
    if ((amax_bits_4 & 2139095040) == 2139095040) {
        scale_i32_4 = 255;
    }
    unsigned int scale_4 = (unsigned int)scale_i32_4;
    unsigned int inv_4 = 254 - scale_4 << 7;
    if (scale_4 == 255) {
        inv_4 = 32704;
    }
    unsigned int inv2_4 = inv_4 | inv_4 << 16;
    int gcol_a_0 = bx * 64 + K + warp_0 * 16 + (lane_0 >> 2);
    int gcol_b_0 = gcol_a_0 + 8;
    int out_row_0 = by * 32 + (lane_0 & 3) * 8;
    unsigned int words_a_0[2];
    unsigned int words_b_0[2];
    uint32_t bf16x2_mul_12;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_12) : "r"(cols_1_0[0]), "r"(inv2_3));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_11) : "r"(cols_1_0[1]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_175;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_175) : "r"(bf16x2_mul_1_11));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_11);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_175) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_174;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_174) : "h"(e4m3x2_bf16x2_11), "h"(_cvt_satfinite_e4m3x2_bf16x2_175));
    words_a_0[0] = _mov_b32_pack_u16x2_174;
    uint32_t bf16x2_mul_13;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_13) : "r"(cols_1_0[2]), "r"(inv2_3));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_12) : "r"(cols_1_0[3]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_177;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_177) : "r"(bf16x2_mul_1_12));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_12);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_177) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_176;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_176) : "h"(e4m3x2_bf16x2_12), "h"(_cvt_satfinite_e4m3x2_bf16x2_177));
    words_a_0[1] = _mov_b32_pack_u16x2_176;
    uint32_t bf16x2_mul_14;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_14) : "r"(cols_1_0[4]), "r"(inv2_4));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_13) : "r"(cols_1_0[5]), "r"(inv2_4));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_179;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_179) : "r"(bf16x2_mul_1_13));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_13);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_179) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_178;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_178) : "h"(e4m3x2_bf16x2_13), "h"(_cvt_satfinite_e4m3x2_bf16x2_179));
    words_b_0[0] = _mov_b32_pack_u16x2_178;
    uint32_t bf16x2_mul_15;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_15) : "r"(cols_1_0[6]), "r"(inv2_4));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_14) : "r"(cols_1_0[7]), "r"(inv2_4));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_181;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_181) : "r"(bf16x2_mul_1_14));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_14);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_181) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_180;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_180) : "h"(e4m3x2_bf16x2_14), "h"(_cvt_satfinite_e4m3x2_bf16x2_181));
    words_b_0[1] = _mov_b32_pack_u16x2_180;
    reinterpret_cast<int2*>(col_output + (gcol_a_0 * M + out_row_0))[0] = reinterpret_cast<int2*>(words_a_0 + 0)[0];
    reinterpret_cast<int2*>(col_output + (gcol_b_0 * M + out_row_0))[0] = reinterpret_cast<int2*>(words_b_0 + 0)[0];
    if ((lane_0 & 3) == 0) {
        int row_tiles_1 = M >> 7;
        int col_scale_a_1 = ((gcol_a_0 >> 7) * row_tiles_1 + (by >> 2)) * 512 + (gcol_a_0 & 31) * 16 + (gcol_a_0 >> 5 & 3) * 4 + (by & 3);
        int col_scale_b_1 = ((gcol_b_0 >> 7) * row_tiles_1 + (by >> 2)) * 512 + (gcol_b_0 & 31) * 16 + (gcol_b_0 >> 5 & 3) * 4 + (by & 3);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_a_1) + (0)) = (unsigned char)(scale_3);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b_1) + (0)) = (unsigned char)(scale_4);
    }
}

} // extern "C"
