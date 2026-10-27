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
#define LAUNCH_MIN_BLOCKS 10

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
kernel_cake_gated_act_mxfp8_b25fe627c7656aebe218(__nv_bfloat16* __restrict__ gated_input, __nv_bfloat16* __restrict__ grad, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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
        float2 _cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 _cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 _cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float e0_1 = approx_exp2((-_cvt_f32_1.x) * 1.442695f);
        float e1_1 = approx_exp2((-_cvt_f32_1.y) * 1.442695f);
        float d0_1 = 1.0f + e0_1;
        float d1_1 = 1.0f + e1_1;
        float r0_1 = approx_rcp(d0_1);
        float r1_1 = approx_rcp(d1_1);
        float _fma_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(-d0_1), "f"(r0_1), "f"(1.0f));
        float _fma_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(r0_1), "f"(_fma_6), "f"(r0_1));
        float _fma_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(-d1_1), "f"(r1_1), "f"(1.0f));
        float _fma_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(r1_1), "f"(_fma_8), "f"(r1_1));
        float2 s_0 = make_float2(_fma_5, _fma_7);
        float2 act_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_0) : "l"(*(const unsigned long long*)&_cvt_f32_1), "l"(*(const unsigned long long*)&s_0));
        float2 _f32x2_mul_a_9 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_0), s_0), _cvt_f32_3);
        float2 _mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_9), "l"(*(const unsigned long long*)&_cvt_f32_2));
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
        float _fma_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(-d0_3), "f"(r0_3), "f"(1.0f));
        float _fma_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(r0_3), "f"(_fma_11), "f"(r0_3));
        float _fma_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(-d1_3), "f"(r1_3), "f"(1.0f));
        float _fma_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(r1_3), "f"(_fma_13), "f"(r1_3));
        float2 s_2 = make_float2(_fma_10, _fma_12);
        float2 act_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_2) : "l"(*(const unsigned long long*)&_cvt_f32_4), "l"(*(const unsigned long long*)&s_2));
        float2 _f32x2_mul_a_14 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_2), s_2), _cvt_f32_6);
        float2 _mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&_cvt_f32_5));
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
        float _fma_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(-d0_5), "f"(r0_5), "f"(1.0f));
        float _fma_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(r0_5), "f"(_fma_16), "f"(r0_5));
        float _fma_18;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(-d1_5), "f"(r1_5), "f"(1.0f));
        float _fma_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(r1_5), "f"(_fma_18), "f"(r1_5));
        float2 s_4 = make_float2(_fma_15, _fma_17);
        float2 act_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_4) : "l"(*(const unsigned long long*)&_cvt_f32_7), "l"(*(const unsigned long long*)&s_4));
        float2 _f32x2_mul_a_19 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_4), s_4), _cvt_f32_9);
        float2 _mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&_cvt_f32_8));
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
        float _fma_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(-d0_7), "f"(r0_7), "f"(1.0f));
        float _fma_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(r0_7), "f"(_fma_21), "f"(r0_7));
        float _fma_23;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(-d1_7), "f"(r1_7), "f"(1.0f));
        float _fma_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(r1_7), "f"(_fma_23), "f"(r1_7));
        float2 s_6 = make_float2(_fma_20, _fma_22);
        float2 act_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_6) : "l"(*(const unsigned long long*)&_cvt_f32_10), "l"(*(const unsigned long long*)&s_6));
        float2 _f32x2_mul_a_24 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_6), s_6), _cvt_f32_12);
        float2 _mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&_cvt_f32_11));
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
        float _fma_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(-d0_9), "f"(r0_9), "f"(1.0f));
        float _fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(r0_9), "f"(_fma_26), "f"(r0_9));
        float _fma_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(-d1_9), "f"(r1_9), "f"(1.0f));
        float _fma_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(r1_9), "f"(_fma_28), "f"(r1_9));
        float2 s_8 = make_float2(_fma_25, _fma_27);
        float2 act_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_8) : "l"(*(const unsigned long long*)&_cvt_f32_13), "l"(*(const unsigned long long*)&s_8));
        float2 _f32x2_mul_a_29 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_8, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_8), s_8), _cvt_f32_15);
        float2 _mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&_cvt_f32_14));
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
        float _fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(-d0_11), "f"(r0_11), "f"(1.0f));
        float _fma_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(r0_11), "f"(_fma_31), "f"(r0_11));
        float _fma_33;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(-d1_11), "f"(r1_11), "f"(1.0f));
        float _fma_32;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(r1_11), "f"(_fma_33), "f"(r1_11));
        float2 s_10 = make_float2(_fma_30, _fma_32);
        float2 act_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_10) : "l"(*(const unsigned long long*)&_cvt_f32_16), "l"(*(const unsigned long long*)&s_10));
        float2 _f32x2_mul_a_34 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_10, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_10), s_10), _cvt_f32_18);
        float2 _mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_34), "l"(*(const unsigned long long*)&_cvt_f32_17));
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
        float _fma_36;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(-d0_13), "f"(r0_13), "f"(1.0f));
        float _fma_35;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(r0_13), "f"(_fma_36), "f"(r0_13));
        float _fma_38;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(-d1_13), "f"(r1_13), "f"(1.0f));
        float _fma_37;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(r1_13), "f"(_fma_38), "f"(r1_13));
        float2 s_12 = make_float2(_fma_35, _fma_37);
        float2 act_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_12) : "l"(*(const unsigned long long*)&_cvt_f32_19), "l"(*(const unsigned long long*)&s_12));
        float2 _f32x2_mul_a_39 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_12, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_12), s_12), _cvt_f32_21);
        float2 _mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_39), "l"(*(const unsigned long long*)&_cvt_f32_20));
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
        float _fma_41;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(-d0_15), "f"(r0_15), "f"(1.0f));
        float _fma_40;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(r0_15), "f"(_fma_41), "f"(r0_15));
        float _fma_43;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(-d1_15), "f"(r1_15), "f"(1.0f));
        float _fma_42;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(r1_15), "f"(_fma_43), "f"(r1_15));
        float2 s_14 = make_float2(_fma_40, _fma_42);
        float2 act_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_14) : "l"(*(const unsigned long long*)&_cvt_f32_22), "l"(*(const unsigned long long*)&s_14));
        float2 _f32x2_mul_a_44 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_14, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_14), s_14), _cvt_f32_24);
        float2 _mul_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&_cvt_f32_23));
        float2 _mul_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_15) : "l"(*(const unsigned long long*)&act_14), "l"(*(const unsigned long long*)&_cvt_f32_24));
        __nv_bfloat162 _bf16x2_14 = __float22bfloat162_rn(make_float2(_mul_14.x, _mul_14.y));
        __nv_bfloat162 _bf16x2_15 = __float22bfloat162_rn(make_float2(_mul_15.x, _mul_15.y));
        dgate[7] = __as_u32(_bf16x2_14);
        dup[7] = __as_u32(_bf16x2_15);
    } else {
        int band = 0;
        float2 _cvt_f32_25 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 _cvt_f32_26 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 _cvt_f32_27 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float e0_17 = approx_exp2((-_cvt_f32_25.x) * 1.442695f);
        float e1_17 = approx_exp2((-_cvt_f32_25.y) * 1.442695f);
        float d0_17 = 1.0f + e0_17;
        float d1_17 = 1.0f + e1_17;
        float r0_17 = approx_rcp(d0_17);
        float r1_17 = approx_rcp(d1_17);
        float _fma_45;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(-d0_17), "f"(r0_17), "f"(1.0f));
        float _fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(r0_17), "f"(_fma_45), "f"(r0_17));
        float _fma_46;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(-d1_17), "f"(r1_17), "f"(1.0f));
        float _fma_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(r1_17), "f"(_fma_46), "f"(r1_17));
        float t0_17 = _fma_0;
        float t1_17 = _fma_1;
        band = band | ((fminf(t0_17, t1_17) == 0.0f) ? 1 : 0);
        float2 s_16 = make_float2(max_noftz(t0_17, 0.0f), max_noftz(t1_17, 0.0f));
        float2 act_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_16) : "l"(*(const unsigned long long*)&_cvt_f32_25), "l"(*(const unsigned long long*)&s_16));
        float2 _f32x2_mul_a_47 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_16, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_16), s_16), _cvt_f32_27);
        float2 _mul_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_47), "l"(*(const unsigned long long*)&_cvt_f32_26));
        float2 _mul_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_17) : "l"(*(const unsigned long long*)&act_16), "l"(*(const unsigned long long*)&_cvt_f32_27));
        __nv_bfloat162 _bf16x2_16 = __float22bfloat162_rn(make_float2(_mul_16.x, _mul_16.y));
        __nv_bfloat162 _bf16x2_17 = __float22bfloat162_rn(make_float2(_mul_17.x, _mul_17.y));
        dgate[0] = __as_u32(_bf16x2_16);
        dup[0] = __as_u32(_bf16x2_17);
        float2 _cvt_f32_28 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 _cvt_f32_29 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 _cvt_f32_30 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float e0_19 = approx_exp2((-_cvt_f32_28.x) * 1.442695f);
        float e1_19 = approx_exp2((-_cvt_f32_28.y) * 1.442695f);
        float d0_19 = 1.0f + e0_19;
        float d1_19 = 1.0f + e1_19;
        float r0_19 = approx_rcp(d0_19);
        float r1_19 = approx_rcp(d1_19);
        float _fma_48;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(-d0_19), "f"(r0_19), "f"(1.0f));
        float _fma_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(r0_19), "f"(_fma_48), "f"(r0_19));
        float _fma_49;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(-d1_19), "f"(r1_19), "f"(1.0f));
        float _fma_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(r1_19), "f"(_fma_49), "f"(r1_19));
        float t0_19 = _fma_2;
        float t1_19 = _fma_3;
        band = band | ((fminf(t0_19, t1_19) == 0.0f) ? 2 : 0);
        float2 s_18 = make_float2(max_noftz(t0_19, 0.0f), max_noftz(t1_19, 0.0f));
        float2 act_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_18) : "l"(*(const unsigned long long*)&_cvt_f32_28), "l"(*(const unsigned long long*)&s_18));
        float2 _f32x2_mul_a_50 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_18, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_18), s_18), _cvt_f32_30);
        float2 _mul_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_50), "l"(*(const unsigned long long*)&_cvt_f32_29));
        float2 _mul_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_19) : "l"(*(const unsigned long long*)&act_18), "l"(*(const unsigned long long*)&_cvt_f32_30));
        __nv_bfloat162 _bf16x2_18 = __float22bfloat162_rn(make_float2(_mul_18.x, _mul_18.y));
        __nv_bfloat162 _bf16x2_19 = __float22bfloat162_rn(make_float2(_mul_19.x, _mul_19.y));
        dgate[1] = __as_u32(_bf16x2_18);
        dup[1] = __as_u32(_bf16x2_19);
        float2 _cvt_f32_31 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 _cvt_f32_32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 _cvt_f32_33 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float e0_21 = approx_exp2((-_cvt_f32_31.x) * 1.442695f);
        float e1_21 = approx_exp2((-_cvt_f32_31.y) * 1.442695f);
        float d0_21 = 1.0f + e0_21;
        float d1_21 = 1.0f + e1_21;
        float r0_21 = approx_rcp(d0_21);
        float r1_21 = approx_rcp(d1_21);
        float _fma_51;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(-d0_21), "f"(r0_21), "f"(1.0f));
        float _fma_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(r0_21), "f"(_fma_51), "f"(r0_21));
        float _fma_52;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(-d1_21), "f"(r1_21), "f"(1.0f));
        float _fma_5_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5_1) : "f"(r1_21), "f"(_fma_52), "f"(r1_21));
        float t0_21 = _fma_4;
        float t1_21 = _fma_5_1;
        band = band | ((fminf(t0_21, t1_21) == 0.0f) ? 4 : 0);
        float2 s_20 = make_float2(max_noftz(t0_21, 0.0f), max_noftz(t1_21, 0.0f));
        float2 act_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_20) : "l"(*(const unsigned long long*)&_cvt_f32_31), "l"(*(const unsigned long long*)&s_20));
        float2 _f32x2_mul_a_53 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_20, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_20), s_20), _cvt_f32_33);
        float2 _mul_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_53), "l"(*(const unsigned long long*)&_cvt_f32_32));
        float2 _mul_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_21) : "l"(*(const unsigned long long*)&act_20), "l"(*(const unsigned long long*)&_cvt_f32_33));
        __nv_bfloat162 _bf16x2_20 = __float22bfloat162_rn(make_float2(_mul_20.x, _mul_20.y));
        __nv_bfloat162 _bf16x2_21 = __float22bfloat162_rn(make_float2(_mul_21.x, _mul_21.y));
        dgate[2] = __as_u32(_bf16x2_20);
        dup[2] = __as_u32(_bf16x2_21);
        float2 _cvt_f32_34 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 _cvt_f32_35 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 _cvt_f32_36 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float e0_23 = approx_exp2((-_cvt_f32_34.x) * 1.442695f);
        float e1_23 = approx_exp2((-_cvt_f32_34.y) * 1.442695f);
        float d0_23 = 1.0f + e0_23;
        float d1_23 = 1.0f + e1_23;
        float r0_23 = approx_rcp(d0_23);
        float r1_23 = approx_rcp(d1_23);
        float _fma_54;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(-d0_23), "f"(r0_23), "f"(1.0f));
        float _fma_6_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6_1) : "f"(r0_23), "f"(_fma_54), "f"(r0_23));
        float _fma_55;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(-d1_23), "f"(r1_23), "f"(1.0f));
        float _fma_7_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7_1) : "f"(r1_23), "f"(_fma_55), "f"(r1_23));
        float t0_23 = _fma_6_1;
        float t1_23 = _fma_7_1;
        band = band | ((fminf(t0_23, t1_23) == 0.0f) ? 8 : 0);
        float2 s_22 = make_float2(max_noftz(t0_23, 0.0f), max_noftz(t1_23, 0.0f));
        float2 act_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_22) : "l"(*(const unsigned long long*)&_cvt_f32_34), "l"(*(const unsigned long long*)&s_22));
        float2 _f32x2_mul_a_56 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_22, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_22), s_22), _cvt_f32_36);
        float2 _mul_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_56), "l"(*(const unsigned long long*)&_cvt_f32_35));
        float2 _mul_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_23) : "l"(*(const unsigned long long*)&act_22), "l"(*(const unsigned long long*)&_cvt_f32_36));
        __nv_bfloat162 _bf16x2_22 = __float22bfloat162_rn(make_float2(_mul_22.x, _mul_22.y));
        __nv_bfloat162 _bf16x2_23 = __float22bfloat162_rn(make_float2(_mul_23.x, _mul_23.y));
        dgate[3] = __as_u32(_bf16x2_22);
        dup[3] = __as_u32(_bf16x2_23);
        float2 _cvt_f32_37 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 _cvt_f32_38 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float2 _cvt_f32_39 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[4]));
        float e0_25 = approx_exp2((-_cvt_f32_37.x) * 1.442695f);
        float e1_25 = approx_exp2((-_cvt_f32_37.y) * 1.442695f);
        float d0_25 = 1.0f + e0_25;
        float d1_25 = 1.0f + e1_25;
        float r0_25 = approx_rcp(d0_25);
        float r1_25 = approx_rcp(d1_25);
        float _fma_57;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(-d0_25), "f"(r0_25), "f"(1.0f));
        float _fma_8_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8_1) : "f"(r0_25), "f"(_fma_57), "f"(r0_25));
        float _fma_58;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(-d1_25), "f"(r1_25), "f"(1.0f));
        float _fma_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(r1_25), "f"(_fma_58), "f"(r1_25));
        float t0_25 = _fma_8_1;
        float t1_25 = _fma_9;
        band = band | ((fminf(t0_25, t1_25) == 0.0f) ? 16 : 0);
        float2 s_24 = make_float2(max_noftz(t0_25, 0.0f), max_noftz(t1_25, 0.0f));
        float2 act_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_24) : "l"(*(const unsigned long long*)&_cvt_f32_37), "l"(*(const unsigned long long*)&s_24));
        float2 _f32x2_mul_a_59 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_24, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_24), s_24), _cvt_f32_39);
        float2 _mul_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_59), "l"(*(const unsigned long long*)&_cvt_f32_38));
        float2 _mul_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_25) : "l"(*(const unsigned long long*)&act_24), "l"(*(const unsigned long long*)&_cvt_f32_39));
        __nv_bfloat162 _bf16x2_24 = __float22bfloat162_rn(make_float2(_mul_24.x, _mul_24.y));
        __nv_bfloat162 _bf16x2_25 = __float22bfloat162_rn(make_float2(_mul_25.x, _mul_25.y));
        dgate[4] = __as_u32(_bf16x2_24);
        dup[4] = __as_u32(_bf16x2_25);
        float2 _cvt_f32_40 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 _cvt_f32_41 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float2 _cvt_f32_42 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[5]));
        float e0_27 = approx_exp2((-_cvt_f32_40.x) * 1.442695f);
        float e1_27 = approx_exp2((-_cvt_f32_40.y) * 1.442695f);
        float d0_27 = 1.0f + e0_27;
        float d1_27 = 1.0f + e1_27;
        float r0_27 = approx_rcp(d0_27);
        float r1_27 = approx_rcp(d1_27);
        float _fma_60;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(-d0_27), "f"(r0_27), "f"(1.0f));
        float _fma_10_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10_1) : "f"(r0_27), "f"(_fma_60), "f"(r0_27));
        float _fma_61;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(-d1_27), "f"(r1_27), "f"(1.0f));
        float _fma_11_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11_1) : "f"(r1_27), "f"(_fma_61), "f"(r1_27));
        float t0_27 = _fma_10_1;
        float t1_27 = _fma_11_1;
        band = band | ((fminf(t0_27, t1_27) == 0.0f) ? 32 : 0);
        float2 s_26 = make_float2(max_noftz(t0_27, 0.0f), max_noftz(t1_27, 0.0f));
        float2 act_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_26) : "l"(*(const unsigned long long*)&_cvt_f32_40), "l"(*(const unsigned long long*)&s_26));
        float2 _f32x2_mul_a_62 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_26, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_26), s_26), _cvt_f32_42);
        float2 _mul_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&_cvt_f32_41));
        float2 _mul_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_27) : "l"(*(const unsigned long long*)&act_26), "l"(*(const unsigned long long*)&_cvt_f32_42));
        __nv_bfloat162 _bf16x2_26 = __float22bfloat162_rn(make_float2(_mul_26.x, _mul_26.y));
        __nv_bfloat162 _bf16x2_27 = __float22bfloat162_rn(make_float2(_mul_27.x, _mul_27.y));
        dgate[5] = __as_u32(_bf16x2_26);
        dup[5] = __as_u32(_bf16x2_27);
        float2 _cvt_f32_43 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 _cvt_f32_44 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float2 _cvt_f32_45 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[6]));
        float e0_29 = approx_exp2((-_cvt_f32_43.x) * 1.442695f);
        float e1_29 = approx_exp2((-_cvt_f32_43.y) * 1.442695f);
        float d0_29 = 1.0f + e0_29;
        float d1_29 = 1.0f + e1_29;
        float r0_29 = approx_rcp(d0_29);
        float r1_29 = approx_rcp(d1_29);
        float _fma_63;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(-d0_29), "f"(r0_29), "f"(1.0f));
        float _fma_12_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12_1) : "f"(r0_29), "f"(_fma_63), "f"(r0_29));
        float _fma_64;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(-d1_29), "f"(r1_29), "f"(1.0f));
        float _fma_13_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13_1) : "f"(r1_29), "f"(_fma_64), "f"(r1_29));
        float t0_29 = _fma_12_1;
        float t1_29 = _fma_13_1;
        band = band | ((fminf(t0_29, t1_29) == 0.0f) ? 64 : 0);
        float2 s_28 = make_float2(max_noftz(t0_29, 0.0f), max_noftz(t1_29, 0.0f));
        float2 act_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_28) : "l"(*(const unsigned long long*)&_cvt_f32_43), "l"(*(const unsigned long long*)&s_28));
        float2 _f32x2_mul_a_65 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_28, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_28), s_28), _cvt_f32_45);
        float2 _mul_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_65), "l"(*(const unsigned long long*)&_cvt_f32_44));
        float2 _mul_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_29) : "l"(*(const unsigned long long*)&act_28), "l"(*(const unsigned long long*)&_cvt_f32_45));
        __nv_bfloat162 _bf16x2_28 = __float22bfloat162_rn(make_float2(_mul_28.x, _mul_28.y));
        __nv_bfloat162 _bf16x2_29 = __float22bfloat162_rn(make_float2(_mul_29.x, _mul_29.y));
        dgate[6] = __as_u32(_bf16x2_28);
        dup[6] = __as_u32(_bf16x2_29);
        float2 _cvt_f32_46 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 _cvt_f32_47 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float2 _cvt_f32_48 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[7]));
        float e0_31 = approx_exp2((-_cvt_f32_46.x) * 1.442695f);
        float e1_31 = approx_exp2((-_cvt_f32_46.y) * 1.442695f);
        float d0_31 = 1.0f + e0_31;
        float d1_31 = 1.0f + e1_31;
        float r0_31 = approx_rcp(d0_31);
        float r1_31 = approx_rcp(d1_31);
        float _fma_66;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(-d0_31), "f"(r0_31), "f"(1.0f));
        float _fma_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(r0_31), "f"(_fma_66), "f"(r0_31));
        float _fma_67;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(-d1_31), "f"(r1_31), "f"(1.0f));
        float _fma_15_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15_1) : "f"(r1_31), "f"(_fma_67), "f"(r1_31));
        float t0_31 = _fma_14;
        float t1_31 = _fma_15_1;
        band = band | ((fminf(t0_31, t1_31) == 0.0f) ? 128 : 0);
        float2 s_30 = make_float2(max_noftz(t0_31, 0.0f), max_noftz(t1_31, 0.0f));
        float2 act_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_30) : "l"(*(const unsigned long long*)&_cvt_f32_46), "l"(*(const unsigned long long*)&s_30));
        float2 _f32x2_mul_a_68 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_30, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_30), s_30), _cvt_f32_48);
        float2 _mul_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&_cvt_f32_47));
        float2 _mul_31;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_31) : "l"(*(const unsigned long long*)&act_30), "l"(*(const unsigned long long*)&_cvt_f32_48));
        __nv_bfloat162 _bf16x2_30 = __float22bfloat162_rn(make_float2(_mul_30.x, _mul_30.y));
        __nv_bfloat162 _bf16x2_31 = __float22bfloat162_rn(make_float2(_mul_31.x, _mul_31.y));
        dgate[7] = __as_u32(_bf16x2_30);
        dup[7] = __as_u32(_bf16x2_31);
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
                    float2 _cvt_f32_49 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
                    float2 _cvt_f32_50 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[0]));
                    float2 _cvt_f32_51 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[0]));
                    float e0_33 = approx_exp2((-_cvt_f32_49.x) * 1.442695f);
                    float e1_33 = approx_exp2((-_cvt_f32_49.y) * 1.442695f);
                    float d0_33 = 1.0f + e0_33;
                    float d1_33 = 1.0f + e1_33;
                    float r0_33 = approx_rcp(d0_33);
                    float r1_33 = approx_rcp(d1_33);
                    float _fma_69;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(-d0_33), "f"(r0_33), "f"(1.0f));
                    float _fma_16_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16_1) : "f"(r0_33), "f"(_fma_69), "f"(r0_33));
                    float _fma_70;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(-d1_33), "f"(r1_33), "f"(1.0f));
                    float _fma_17_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17_1) : "f"(r1_33), "f"(_fma_70), "f"(r1_33));
                    float t0_33 = _fma_16_1;
                    float t1_33 = _fma_17_1;
                    float d4_33a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_33a) : "f"(d0_33), "f"(0.25f));
                    float r4_33a = approx_rcp(d4_33a);
                    float _fma_72;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(-d4_33a), "f"(r4_33a), "f"(1.0f));
                    float _fma_71;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(r4_33a), "f"(_fma_72), "f"(r4_33a));
                    float big_33a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_33a) : "f"(mul_f32_rn_noftz(_fma_71, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_73;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(big_33a), "f"(8.388608e+06f));
                    float whole_33a;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_33a) : "f"(_add_73), "f"(8.388608e+06f));
                    float above_33a;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_33a) : "f"(-d4_33a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_33a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_75;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_75) : "f"(big_33a), "f"(whole_33a));
                    float _abs_74;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_74) : "f"(_sub_75));
                    float _add_76;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(big_33a), "f"(0.5f));
                    float _sub_77;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_77) : "f"(big_33a), "f"(0.5f));
                    float rounded_33a = ((_abs_74 == 0.5f) ? ((above_33a > 0.0f) ? _add_76 : _sub_77) : whole_33a);
                    float u0_33 = ((d0_33 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_33a, 7.888609e-31f), 1.7763568e-15f) : t0_33);
                    float d4_33b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_33b) : "f"(d1_33), "f"(0.25f));
                    float r4_33b = approx_rcp(d4_33b);
                    float _fma_79;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(-d4_33b), "f"(r4_33b), "f"(1.0f));
                    float _fma_78;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_78) : "f"(r4_33b), "f"(_fma_79), "f"(r4_33b));
                    float big_33b;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_33b) : "f"(mul_f32_rn_noftz(_fma_78, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_80;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_80) : "f"(big_33b), "f"(8.388608e+06f));
                    float whole_33b;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_33b) : "f"(_add_80), "f"(8.388608e+06f));
                    float above_33b;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_33b) : "f"(-d4_33b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_33b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_82;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_82) : "f"(big_33b), "f"(whole_33b));
                    float _abs_81;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_81) : "f"(_sub_82));
                    float _add_83;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(big_33b), "f"(0.5f));
                    float _sub_84;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_84) : "f"(big_33b), "f"(0.5f));
                    float rounded_33b = ((_abs_81 == 0.5f) ? ((above_33b > 0.0f) ? _add_83 : _sub_84) : whole_33b);
                    float u1_33 = ((d1_33 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_33b, 7.888609e-31f), 1.7763568e-15f) : t1_33);
                    float2 s_32 = make_float2(max_noftz(u0_33, 0.0f), max_noftz(u1_33, 0.0f));
                    float2 act_32;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_32) : "l"(*(const unsigned long long*)&_cvt_f32_49), "l"(*(const unsigned long long*)&s_32));
                    float2 _f32x2_mul_a_85 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_32, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_32), s_32), _cvt_f32_51);
                    float2 _mul_32;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_32) : "l"(*(const unsigned long long*)&_f32x2_mul_a_85), "l"(*(const unsigned long long*)&_cvt_f32_50));
                    float2 _mul_33;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_33) : "l"(*(const unsigned long long*)&act_32), "l"(*(const unsigned long long*)&_cvt_f32_51));
                    __nv_bfloat162 _bf16x2_32 = __float22bfloat162_rn(make_float2(_mul_32.x, _mul_32.y));
                    __nv_bfloat162 _bf16x2_33 = __float22bfloat162_rn(make_float2(_mul_33.x, _mul_33.y));
                    unsigned int dgate_new = __as_u32(_bf16x2_32);
                    unsigned int dup_new = __as_u32(_bf16x2_33);
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_103;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_103) : "r"(_bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_103) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_102;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_102) : "h"(_e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_103));
    words[0] = _mov_b32_pack_u16x2_102;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_105;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_105) : "r"(_bf16x2_mul_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_105) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_104;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_104) : "h"(_e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_105));
    words[1] = _mov_b32_pack_u16x2_104;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_107;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_107) : "r"(_bf16x2_mul_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_107) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_106;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_106) : "h"(_e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_107));
    words[2] = _mov_b32_pack_u16x2_106;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_109;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_109) : "r"(_bf16x2_mul_7));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_7);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_109) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_108;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_108) : "h"(_e4m3x2_bf16x2_3), "h"(_cvt_satfinite_e4m3x2_bf16x2_109));
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_127;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_127) : "r"(_bf16x2_mul_9));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_9);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_127) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_126;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_126) : "h"(_e4m3x2_bf16x2_4), "h"(_cvt_satfinite_e4m3x2_bf16x2_127));
    words_11[0] = _mov_b32_pack_u16x2_126;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_129;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_129) : "r"(_bf16x2_mul_11));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_11);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_129) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_128;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_128) : "h"(_e4m3x2_bf16x2_5), "h"(_cvt_satfinite_e4m3x2_bf16x2_129));
    words_11[1] = _mov_b32_pack_u16x2_128;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_131;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_131) : "r"(_bf16x2_mul_13));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_13);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_131) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_130;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_130) : "h"(_e4m3x2_bf16x2_6), "h"(_cvt_satfinite_e4m3x2_bf16x2_131));
    words_11[2] = _mov_b32_pack_u16x2_130;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_133;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_133) : "r"(_bf16x2_mul_15));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_15);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_133) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_132;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_132) : "h"(_e4m3x2_bf16x2_7), "h"(_cvt_satfinite_e4m3x2_bf16x2_133));
    words_11[3] = _mov_b32_pack_u16x2_132;
    reinterpret_cast<int4*>(row_output + (out_index + K))[0] = reinterpret_cast<int4*>(words_11 + 0)[0];
    __syncthreads();
    int lane_1 = tid & 31;
    int warp_1 = tid >> 5;
    int k8 = lane_1 & 7;
    int srow = (k8 >> 1) * 8 + (lane_1 >> 3) * 2 + (k8 & 1);
    unsigned int sswizzle = (unsigned int)(srow >> 2 & 6 | srow & 1);
    unsigned int cols[8];
    unsigned int strip_chunk0 = (unsigned int)(warp_1 * 2);
    unsigned int strip_src0 = tiles_addr + (unsigned int)(srow * 128) + ((strip_chunk0 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[0]), "=r"(cols[1]), "=r"(cols[2]), "=r"(cols[3])
        : "r"(strip_src0)
        : "memory");
    unsigned int strip_chunk1 = (unsigned int)(warp_1 * 2 + 1);
    unsigned int strip_src1 = tiles_addr + (unsigned int)(srow * 128) + ((strip_chunk1 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[4]), "=r"(cols[5]), "=r"(cols[6]), "=r"(cols[7])
        : "r"(strip_src1)
        : "memory");
    __nv_bfloat162 amax_a = __as_bf16x2(0);
    __nv_bfloat162 amax_b = __as_bf16x2(0);
    __nv_bfloat162 _hxa_135 = amax_a;
    __nv_bfloat162 _hxb_135 = __as_bf16x2(cols[0]);
    unsigned int _hxr_135;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_135) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_135)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_135)));
    __nv_bfloat162 _max_134 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_135);
    amax_a = _max_134;
    __nv_bfloat162 _hxa_137 = amax_b;
    __nv_bfloat162 _hxb_137 = __as_bf16x2(cols[4]);
    unsigned int _hxr_137;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_137) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_137)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_137)));
    __nv_bfloat162 _max_136 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_137);
    amax_b = _max_136;
    __nv_bfloat162 _hxa_139 = amax_a;
    __nv_bfloat162 _hxb_139 = __as_bf16x2(cols[1]);
    unsigned int _hxr_139;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_139) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_139)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_139)));
    __nv_bfloat162 _max_138 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_139);
    amax_a = _max_138;
    __nv_bfloat162 _hxa_141 = amax_b;
    __nv_bfloat162 _hxb_141 = __as_bf16x2(cols[5]);
    unsigned int _hxr_141;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_141) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_141)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_141)));
    __nv_bfloat162 _max_140 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_141);
    amax_b = _max_140;
    __nv_bfloat162 _hxa_143 = amax_a;
    __nv_bfloat162 _hxb_143 = __as_bf16x2(cols[2]);
    unsigned int _hxr_143;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_143) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_143)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_143)));
    __nv_bfloat162 _max_142 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_143);
    amax_a = _max_142;
    __nv_bfloat162 _hxa_145 = amax_b;
    __nv_bfloat162 _hxb_145 = __as_bf16x2(cols[6]);
    unsigned int _hxr_145;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_145) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_145)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_145)));
    __nv_bfloat162 _max_144 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_145);
    amax_b = _max_144;
    __nv_bfloat162 _hxa_147 = amax_a;
    __nv_bfloat162 _hxb_147 = __as_bf16x2(cols[3]);
    unsigned int _hxr_147;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_147) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_147)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_147)));
    __nv_bfloat162 _max_146 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_147);
    amax_a = _max_146;
    __nv_bfloat162 _hxa_149 = amax_b;
    __nv_bfloat162 _hxb_149 = __as_bf16x2(cols[7]);
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
    unsigned int amax_bits_12 = (col_max_bits & 65535) << 16;
    int scale_i32_13 = (int)(amax_bits_12 + 2031616 >> 23) - 8;
    if (scale_i32_13 < 0) {
        scale_i32_13 = 0;
    }
    if ((amax_bits_12 & 2139095040) == 2139095040) {
        scale_i32_13 = 255;
    }
    unsigned int scale_14 = (unsigned int)scale_i32_13;
    unsigned int inv_15 = 254 - scale_14 << 7;
    if (scale_14 == 255) {
        inv_15 = 32704;
    }
    unsigned int inv2_16 = inv_15 | inv_15 << 16;
    unsigned int amax_bits_17 = (col_max_bits >> 16 & 65535) << 16;
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
    int gcol_a = bx * 64 + warp_1 * 16 + (lane_1 >> 2);
    int gcol_b = gcol_a + 8;
    int out_row = by * 32 + (lane_1 & 3) * 8;
    unsigned int words_a[2];
    unsigned int words_b[2];
    uint32_t _bf16x2_mul_16;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_16) : "r"(cols[0]), "r"(inv2_16));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_17) : "r"(cols[1]), "r"(inv2_16));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_151;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_151) : "r"(_bf16x2_mul_17));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_17);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_151) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_150;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_150) : "h"(_e4m3x2_bf16x2_8), "h"(_cvt_satfinite_e4m3x2_bf16x2_151));
    words_a[0] = _mov_b32_pack_u16x2_150;
    uint32_t _bf16x2_mul_18;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_18) : "r"(cols[2]), "r"(inv2_16));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_19) : "r"(cols[3]), "r"(inv2_16));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_153;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_153) : "r"(_bf16x2_mul_19));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_19);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_153) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_152;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_152) : "h"(_e4m3x2_bf16x2_9), "h"(_cvt_satfinite_e4m3x2_bf16x2_153));
    words_a[1] = _mov_b32_pack_u16x2_152;
    uint32_t _bf16x2_mul_20;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_20) : "r"(cols[4]), "r"(inv2_21));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_21) : "r"(cols[5]), "r"(inv2_21));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_155;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_155) : "r"(_bf16x2_mul_21));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_21);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_155) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_154;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_154) : "h"(_e4m3x2_bf16x2_10), "h"(_cvt_satfinite_e4m3x2_bf16x2_155));
    words_b[0] = _mov_b32_pack_u16x2_154;
    uint32_t _bf16x2_mul_22;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_22) : "r"(cols[6]), "r"(inv2_21));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_23) : "r"(cols[7]), "r"(inv2_21));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_157;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_157) : "r"(_bf16x2_mul_23));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_23);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_157) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_156;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_156) : "h"(_e4m3x2_bf16x2_11), "h"(_cvt_satfinite_e4m3x2_bf16x2_157));
    words_b[1] = _mov_b32_pack_u16x2_156;
    reinterpret_cast<int2*>(col_output + (gcol_a * M + out_row))[0] = reinterpret_cast<int2*>(words_a + 0)[0];
    reinterpret_cast<int2*>(col_output + (gcol_b * M + out_row))[0] = reinterpret_cast<int2*>(words_b + 0)[0];
    if ((lane_1 & 3) == 0) {
        int row_tiles = M >> 7;
        int col_scale_a = ((gcol_a >> 7) * row_tiles + (by >> 2)) * 512 + (gcol_a & 31) * 16 + (gcol_a >> 5 & 3) * 4 + (by & 3);
        int col_scale_b = ((gcol_b >> 7) * row_tiles + (by >> 2)) * 512 + (gcol_b & 31) * 16 + (gcol_b >> 5 & 3) * 4 + (by & 3);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_a) + (0)) = (unsigned char)(scale_14);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b) + (0)) = (unsigned char)(scale_19);
    }
    int lane_22 = tid & 31;
    int warp_23 = tid >> 5;
    int k8_24 = lane_22 & 7;
    int srow_25 = (k8_24 >> 1) * 8 + (lane_22 >> 3) * 2 + (k8_24 & 1);
    unsigned int sswizzle_26 = (unsigned int)(srow_25 >> 2 & 6 | srow_25 & 1);
    unsigned int cols_27[8];
    unsigned int strip_chunk0_28 = (unsigned int)(warp_23 * 2);
    unsigned int strip_src0_29 = tiles_addr + 4096 + (unsigned int)(srow_25 * 128) + ((strip_chunk0_28 ^ sswizzle_26) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_27[0]), "=r"(cols_27[1]), "=r"(cols_27[2]), "=r"(cols_27[3])
        : "r"(strip_src0_29)
        : "memory");
    unsigned int strip_chunk1_30 = (unsigned int)(warp_23 * 2 + 1);
    unsigned int strip_src1_31 = tiles_addr + 4096 + (unsigned int)(srow_25 * 128) + ((strip_chunk1_30 ^ sswizzle_26) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_27[4]), "=r"(cols_27[5]), "=r"(cols_27[6]), "=r"(cols_27[7])
        : "r"(strip_src1_31)
        : "memory");
    __nv_bfloat162 amax_a_32 = __as_bf16x2(0);
    __nv_bfloat162 amax_b_33 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_159 = amax_a_32;
    __nv_bfloat162 _hxb_159 = __as_bf16x2(cols_27[0]);
    unsigned int _hxr_159;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_159) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_159)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_159)));
    __nv_bfloat162 _max_158 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_159);
    amax_a_32 = _max_158;
    __nv_bfloat162 _hxa_161 = amax_b_33;
    __nv_bfloat162 _hxb_161 = __as_bf16x2(cols_27[4]);
    unsigned int _hxr_161;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_161) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_161)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_161)));
    __nv_bfloat162 _max_160 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_161);
    amax_b_33 = _max_160;
    __nv_bfloat162 _hxa_163 = amax_a_32;
    __nv_bfloat162 _hxb_163 = __as_bf16x2(cols_27[1]);
    unsigned int _hxr_163;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_163) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_163)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_163)));
    __nv_bfloat162 _max_162 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_163);
    amax_a_32 = _max_162;
    __nv_bfloat162 _hxa_165 = amax_b_33;
    __nv_bfloat162 _hxb_165 = __as_bf16x2(cols_27[5]);
    unsigned int _hxr_165;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_165) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_165)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_165)));
    __nv_bfloat162 _max_164 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_165);
    amax_b_33 = _max_164;
    __nv_bfloat162 _hxa_167 = amax_a_32;
    __nv_bfloat162 _hxb_167 = __as_bf16x2(cols_27[2]);
    unsigned int _hxr_167;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_167) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_167)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_167)));
    __nv_bfloat162 _max_166 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_167);
    amax_a_32 = _max_166;
    __nv_bfloat162 _hxa_169 = amax_b_33;
    __nv_bfloat162 _hxb_169 = __as_bf16x2(cols_27[6]);
    unsigned int _hxr_169;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_169) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_169)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_169)));
    __nv_bfloat162 _max_168 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_169);
    amax_b_33 = _max_168;
    __nv_bfloat162 _hxa_171 = amax_a_32;
    __nv_bfloat162 _hxb_171 = __as_bf16x2(cols_27[3]);
    unsigned int _hxr_171;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_171) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_171)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_171)));
    __nv_bfloat162 _max_170 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_171);
    amax_a_32 = _max_170;
    __nv_bfloat162 _hxa_173 = amax_b_33;
    __nv_bfloat162 _hxb_173 = __as_bf16x2(cols_27[7]);
    unsigned int _hxr_173;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_173) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_173)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_173)));
    __nv_bfloat162 _max_172 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_173);
    amax_b_33 = _max_172;
    unsigned int mag_a_34 = __as_u32(amax_a_32) & 2147450879;
    unsigned int mag_b_35 = __as_u32(amax_b_33) & 2147450879;
    unsigned int lows_ab_36 = mag_a_34 & 65535 | mag_b_35 << 16;
    unsigned int highs_ab_37 = mag_a_34 >> 16 | mag_b_35 & 4294901760u;
    __nv_bfloat162 col_max_38 = __hmax2_nan(__as_bf16x2(lows_ab_36), __as_bf16x2(highs_ab_37));
    unsigned int peer1_39 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(col_max_38), 1);
    col_max_38 = __hmax2_nan(col_max_38, __as_bf16x2(peer1_39));
    unsigned int peer2_40 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(col_max_38), 2);
    col_max_38 = __hmax2_nan(col_max_38, __as_bf16x2(peer2_40));
    unsigned int col_max_bits_41 = __as_u32(col_max_38);
    unsigned int amax_bits_42 = (col_max_bits_41 & 65535) << 16;
    int scale_i32_43 = (int)(amax_bits_42 + 2031616 >> 23) - 8;
    if (scale_i32_43 < 0) {
        scale_i32_43 = 0;
    }
    if ((amax_bits_42 & 2139095040) == 2139095040) {
        scale_i32_43 = 255;
    }
    unsigned int scale_44 = (unsigned int)scale_i32_43;
    unsigned int inv_45 = 254 - scale_44 << 7;
    if (scale_44 == 255) {
        inv_45 = 32704;
    }
    unsigned int inv2_46 = inv_45 | inv_45 << 16;
    unsigned int amax_bits_47 = (col_max_bits_41 >> 16 & 65535) << 16;
    int scale_i32_48 = (int)(amax_bits_47 + 2031616 >> 23) - 8;
    if (scale_i32_48 < 0) {
        scale_i32_48 = 0;
    }
    if ((amax_bits_47 & 2139095040) == 2139095040) {
        scale_i32_48 = 255;
    }
    unsigned int scale_49 = (unsigned int)scale_i32_48;
    unsigned int inv_50 = 254 - scale_49 << 7;
    if (scale_49 == 255) {
        inv_50 = 32704;
    }
    unsigned int inv2_51 = inv_50 | inv_50 << 16;
    int gcol_a_52 = bx * 64 + K + warp_23 * 16 + (lane_22 >> 2);
    int gcol_b_53 = gcol_a_52 + 8;
    int out_row_54 = by * 32 + (lane_22 & 3) * 8;
    unsigned int words_a_55[2];
    unsigned int words_b_56[2];
    uint32_t _bf16x2_mul_24;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_24) : "r"(cols_27[0]), "r"(inv2_46));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_25) : "r"(cols_27[1]), "r"(inv2_46));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_175;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_175) : "r"(_bf16x2_mul_25));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_25);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_175) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_174;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_174) : "h"(_e4m3x2_bf16x2_12), "h"(_cvt_satfinite_e4m3x2_bf16x2_175));
    words_a_55[0] = _mov_b32_pack_u16x2_174;
    uint32_t _bf16x2_mul_26;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_26) : "r"(cols_27[2]), "r"(inv2_46));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_27) : "r"(cols_27[3]), "r"(inv2_46));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_177;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_177) : "r"(_bf16x2_mul_27));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_27);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_177) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_176;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_176) : "h"(_e4m3x2_bf16x2_13), "h"(_cvt_satfinite_e4m3x2_bf16x2_177));
    words_a_55[1] = _mov_b32_pack_u16x2_176;
    uint32_t _bf16x2_mul_28;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_28) : "r"(cols_27[4]), "r"(inv2_51));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_29) : "r"(cols_27[5]), "r"(inv2_51));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_179;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_179) : "r"(_bf16x2_mul_29));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_29);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_179) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_178;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_178) : "h"(_e4m3x2_bf16x2_14), "h"(_cvt_satfinite_e4m3x2_bf16x2_179));
    words_b_56[0] = _mov_b32_pack_u16x2_178;
    uint32_t _bf16x2_mul_30;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_30) : "r"(cols_27[6]), "r"(inv2_51));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_31) : "r"(cols_27[7]), "r"(inv2_51));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_181;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_181) : "r"(_bf16x2_mul_31));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_31);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_181) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_180;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_180) : "h"(_e4m3x2_bf16x2_15), "h"(_cvt_satfinite_e4m3x2_bf16x2_181));
    words_b_56[1] = _mov_b32_pack_u16x2_180;
    reinterpret_cast<int2*>(col_output + (gcol_a_52 * M + out_row_54))[0] = reinterpret_cast<int2*>(words_a_55 + 0)[0];
    reinterpret_cast<int2*>(col_output + (gcol_b_53 * M + out_row_54))[0] = reinterpret_cast<int2*>(words_b_56 + 0)[0];
    if ((lane_22 & 3) == 0) {
        int row_tiles_1 = M >> 7;
        int col_scale_a_1 = ((gcol_a_52 >> 7) * row_tiles_1 + (by >> 2)) * 512 + (gcol_a_52 & 31) * 16 + (gcol_a_52 >> 5 & 3) * 4 + (by & 3);
        int col_scale_b_1 = ((gcol_b_53 >> 7) * row_tiles_1 + (by >> 2)) * 512 + (gcol_b_53 & 31) * 16 + (gcol_b_53 >> 5 & 3) * 4 + (by & 3);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_a_1) + (0)) = (unsigned char)(scale_44);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b_1) + (0)) = (unsigned char)(scale_49);
    }
}

} // extern "C"
