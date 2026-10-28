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
#define SMEM_TILES_STAGE_BYTES 4096
#define SMEM_TILES_STRIDE 4096
#define SMEM_TOTAL 4096
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 9

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
kernel_cake_gated_act_mxfp8_c6ab778cd390eb44c75a(__nv_bfloat16* __restrict__ gated_input, __nv_bfloat16* __restrict__ grad, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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
    int quarter = tid & 3;
    int row = tid >> 2;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int grow = by * 32 + row;
    int col = bx * 32 + quarter * 8;
    int gate_index = grow * (2 * K) + col;
    unsigned int _vec_load_0[4];
    {
        uint32_t _uv4_0_0;
        uint32_t _uv4_0_1;
        uint32_t _uv4_0_2;
        uint32_t _uv4_0_3;
        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
            : "=r"(_uv4_0_0), "=r"(_uv4_0_1), "=r"(_uv4_0_2), "=r"(_uv4_0_3) : "l"((const void*)(gated_input + gate_index + (0))), "l"(0x12F0000000000000ULL) : "memory");
        _vec_load_0[0 + 0] = _uv4_0_0;
        _vec_load_0[0 + 1] = _uv4_0_1;
        _vec_load_0[0 + 2] = _uv4_0_2;
        _vec_load_0[0 + 3] = _uv4_0_3;
    }
    unsigned int _vec_load_1[4];
    {
        uint32_t _uv4_1_0;
        uint32_t _uv4_1_1;
        uint32_t _uv4_1_2;
        uint32_t _uv4_1_3;
        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
            : "=r"(_uv4_1_0), "=r"(_uv4_1_1), "=r"(_uv4_1_2), "=r"(_uv4_1_3) : "l"((const void*)(gated_input + (gate_index + K) + (0))), "l"(0x12F0000000000000ULL) : "memory");
        _vec_load_1[0 + 0] = _uv4_1_0;
        _vec_load_1[0 + 1] = _uv4_1_1;
        _vec_load_1[0 + 2] = _uv4_1_2;
        _vec_load_1[0 + 3] = _uv4_1_3;
    }
    unsigned int _vec_load_2[4];
    {
        uint32_t _uv4_2_0;
        uint32_t _uv4_2_1;
        uint32_t _uv4_2_2;
        uint32_t _uv4_2_3;
        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
            : "=r"(_uv4_2_0), "=r"(_uv4_2_1), "=r"(_uv4_2_2), "=r"(_uv4_2_3) : "l"((const void*)(grad + (grow * K + col) + (0))), "l"(0x12F0000000000000ULL) : "memory");
        _vec_load_2[0 + 0] = _uv4_2_0;
        _vec_load_2[0 + 1] = _uv4_2_1;
        _vec_load_2[0 + 2] = _uv4_2_2;
        _vec_load_2[0 + 3] = _uv4_2_3;
    }
    unsigned int dgate[4];
    unsigned int dup[4];
    __nv_bfloat162 lowest = __as_bf16x2(_vec_load_0[0]);
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[3]));
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
    } else {
        int band = 0;
        float2 _cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 _cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 _cvt_f32_15 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float e0_9 = approx_exp2((-_cvt_f32_13.x) * 1.442695f);
        float e1_9 = approx_exp2((-_cvt_f32_13.y) * 1.442695f);
        float d0_9 = 1.0f + e0_9;
        float d1_9 = 1.0f + e1_9;
        float r0_9 = approx_rcp(d0_9);
        float r1_9 = approx_rcp(d1_9);
        float _fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(-d0_9), "f"(r0_9), "f"(1.0f));
        float _fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(r0_9), "f"(_fma_25), "f"(r0_9));
        float _fma_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(-d1_9), "f"(r1_9), "f"(1.0f));
        float _fma_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(r1_9), "f"(_fma_26), "f"(r1_9));
        float t0_9 = _fma_0;
        float t1_9 = _fma_1;
        int _warp_vote_27 = __any_sync(0xFFFFFFFF, fminf(t0_9, t1_9) == 0.0f);
        if (_warp_vote_27 != 0) {
            float d4_9a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_9a) : "f"(d0_9), "f"(0.25f));
            float r4_9a = approx_rcp(d4_9a);
            float _fma_29;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(-d4_9a), "f"(r4_9a), "f"(1.0f));
            float _fma_28;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(r4_9a), "f"(_fma_29), "f"(r4_9a));
            float big_9a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_9a) : "f"(mul_f32_rn_noftz(_fma_28, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_30;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(big_9a), "f"(8.388608e+06f));
            float whole_9a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_9a) : "f"(_add_30), "f"(8.388608e+06f));
            float above_9a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_9a) : "f"(-d4_9a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_9a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_32;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32) : "f"(big_9a), "f"(whole_9a));
            float _abs_31;
            asm("abs.f32 %0, %1;" : "=f"(_abs_31) : "f"(_sub_32));
            float _add_33;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(big_9a), "f"(0.5f));
            float _sub_34;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34) : "f"(big_9a), "f"(0.5f));
            float rounded_9a = ((_abs_31 == 0.5f) ? ((above_9a > 0.0f) ? _add_33 : _sub_34) : whole_9a);
            t0_9 = ((d0_9 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_9a, 7.888609e-31f), 1.7763568e-15f) : t0_9);
            float d4_9b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_9b) : "f"(d1_9), "f"(0.25f));
            float r4_9b = approx_rcp(d4_9b);
            float _fma_36;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(-d4_9b), "f"(r4_9b), "f"(1.0f));
            float _fma_35;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(r4_9b), "f"(_fma_36), "f"(r4_9b));
            float big_9b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_9b) : "f"(mul_f32_rn_noftz(_fma_35, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_37;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(big_9b), "f"(8.388608e+06f));
            float whole_9b;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_9b) : "f"(_add_37), "f"(8.388608e+06f));
            float above_9b;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_9b) : "f"(-d4_9b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_9b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_39;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_39) : "f"(big_9b), "f"(whole_9b));
            float _abs_38;
            asm("abs.f32 %0, %1;" : "=f"(_abs_38) : "f"(_sub_39));
            float _add_40;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(big_9b), "f"(0.5f));
            float _sub_41;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41) : "f"(big_9b), "f"(0.5f));
            float rounded_9b = ((_abs_38 == 0.5f) ? ((above_9b > 0.0f) ? _add_40 : _sub_41) : whole_9b);
            t1_9 = ((d1_9 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_9b, 7.888609e-31f), 1.7763568e-15f) : t1_9);
        }
        float2 s_8 = make_float2(max_noftz(t0_9, 0.0f), max_noftz(t1_9, 0.0f));
        float2 act_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_8) : "l"(*(const unsigned long long*)&_cvt_f32_13), "l"(*(const unsigned long long*)&s_8));
        float2 _f32x2_mul_a_42 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_8, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_8), s_8), _cvt_f32_15);
        float2 _mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&_cvt_f32_14));
        float2 _mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_9) : "l"(*(const unsigned long long*)&act_8), "l"(*(const unsigned long long*)&_cvt_f32_15));
        __nv_bfloat162 _bf16x2_8 = __float22bfloat162_rn(make_float2(_mul_8.x, _mul_8.y));
        __nv_bfloat162 _bf16x2_9 = __float22bfloat162_rn(make_float2(_mul_9.x, _mul_9.y));
        dgate[0] = __as_u32(_bf16x2_8);
        dup[0] = __as_u32(_bf16x2_9);
        float2 _cvt_f32_16 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 _cvt_f32_17 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 _cvt_f32_18 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float e0_11 = approx_exp2((-_cvt_f32_16.x) * 1.442695f);
        float e1_11 = approx_exp2((-_cvt_f32_16.y) * 1.442695f);
        float d0_11 = 1.0f + e0_11;
        float d1_11 = 1.0f + e1_11;
        float r0_11 = approx_rcp(d0_11);
        float r1_11 = approx_rcp(d1_11);
        float _fma_43;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(-d0_11), "f"(r0_11), "f"(1.0f));
        float _fma_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(r0_11), "f"(_fma_43), "f"(r0_11));
        float _fma_44;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(-d1_11), "f"(r1_11), "f"(1.0f));
        float _fma_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(r1_11), "f"(_fma_44), "f"(r1_11));
        float t0_11 = _fma_2;
        float t1_11 = _fma_3;
        int _warp_vote_45 = __any_sync(0xFFFFFFFF, fminf(t0_11, t1_11) == 0.0f);
        if (_warp_vote_45 != 0) {
            float d4_11a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_11a) : "f"(d0_11), "f"(0.25f));
            float r4_11a = approx_rcp(d4_11a);
            float _fma_47;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(-d4_11a), "f"(r4_11a), "f"(1.0f));
            float _fma_46;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(r4_11a), "f"(_fma_47), "f"(r4_11a));
            float big_11a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_11a) : "f"(mul_f32_rn_noftz(_fma_46, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_48;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(big_11a), "f"(8.388608e+06f));
            float whole_11a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_11a) : "f"(_add_48), "f"(8.388608e+06f));
            float above_11a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_11a) : "f"(-d4_11a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_11a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_50;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_50) : "f"(big_11a), "f"(whole_11a));
            float _abs_49;
            asm("abs.f32 %0, %1;" : "=f"(_abs_49) : "f"(_sub_50));
            float _add_51;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(big_11a), "f"(0.5f));
            float _sub_52;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_52) : "f"(big_11a), "f"(0.5f));
            float rounded_11a = ((_abs_49 == 0.5f) ? ((above_11a > 0.0f) ? _add_51 : _sub_52) : whole_11a);
            t0_11 = ((d0_11 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_11a, 7.888609e-31f), 1.7763568e-15f) : t0_11);
            float d4_11b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_11b) : "f"(d1_11), "f"(0.25f));
            float r4_11b = approx_rcp(d4_11b);
            float _fma_54;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(-d4_11b), "f"(r4_11b), "f"(1.0f));
            float _fma_53;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(r4_11b), "f"(_fma_54), "f"(r4_11b));
            float big_11b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_11b) : "f"(mul_f32_rn_noftz(_fma_53, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_55;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_55) : "f"(big_11b), "f"(8.388608e+06f));
            float whole_11b;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_11b) : "f"(_add_55), "f"(8.388608e+06f));
            float above_11b;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_11b) : "f"(-d4_11b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_11b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_57;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_57) : "f"(big_11b), "f"(whole_11b));
            float _abs_56;
            asm("abs.f32 %0, %1;" : "=f"(_abs_56) : "f"(_sub_57));
            float _add_58;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_58) : "f"(big_11b), "f"(0.5f));
            float _sub_59;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_59) : "f"(big_11b), "f"(0.5f));
            float rounded_11b = ((_abs_56 == 0.5f) ? ((above_11b > 0.0f) ? _add_58 : _sub_59) : whole_11b);
            t1_11 = ((d1_11 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_11b, 7.888609e-31f), 1.7763568e-15f) : t1_11);
        }
        float2 s_10 = make_float2(max_noftz(t0_11, 0.0f), max_noftz(t1_11, 0.0f));
        float2 act_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_10) : "l"(*(const unsigned long long*)&_cvt_f32_16), "l"(*(const unsigned long long*)&s_10));
        float2 _f32x2_mul_a_60 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_10, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_10), s_10), _cvt_f32_18);
        float2 _mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&_cvt_f32_17));
        float2 _mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_11) : "l"(*(const unsigned long long*)&act_10), "l"(*(const unsigned long long*)&_cvt_f32_18));
        __nv_bfloat162 _bf16x2_10 = __float22bfloat162_rn(make_float2(_mul_10.x, _mul_10.y));
        __nv_bfloat162 _bf16x2_11 = __float22bfloat162_rn(make_float2(_mul_11.x, _mul_11.y));
        dgate[1] = __as_u32(_bf16x2_10);
        dup[1] = __as_u32(_bf16x2_11);
        float2 _cvt_f32_19 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 _cvt_f32_20 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 _cvt_f32_21 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float e0_13 = approx_exp2((-_cvt_f32_19.x) * 1.442695f);
        float e1_13 = approx_exp2((-_cvt_f32_19.y) * 1.442695f);
        float d0_13 = 1.0f + e0_13;
        float d1_13 = 1.0f + e1_13;
        float r0_13 = approx_rcp(d0_13);
        float r1_13 = approx_rcp(d1_13);
        float _fma_61;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(-d0_13), "f"(r0_13), "f"(1.0f));
        float _fma_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(r0_13), "f"(_fma_61), "f"(r0_13));
        float _fma_62;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(-d1_13), "f"(r1_13), "f"(1.0f));
        float _fma_5_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5_1) : "f"(r1_13), "f"(_fma_62), "f"(r1_13));
        float t0_13 = _fma_4;
        float t1_13 = _fma_5_1;
        int _warp_vote_63 = __any_sync(0xFFFFFFFF, fminf(t0_13, t1_13) == 0.0f);
        if (_warp_vote_63 != 0) {
            float d4_13a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_13a) : "f"(d0_13), "f"(0.25f));
            float r4_13a = approx_rcp(d4_13a);
            float _fma_65;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(-d4_13a), "f"(r4_13a), "f"(1.0f));
            float _fma_64;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(r4_13a), "f"(_fma_65), "f"(r4_13a));
            float big_13a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_13a) : "f"(mul_f32_rn_noftz(_fma_64, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_66;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_66) : "f"(big_13a), "f"(8.388608e+06f));
            float whole_13a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_13a) : "f"(_add_66), "f"(8.388608e+06f));
            float above_13a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_13a) : "f"(-d4_13a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_13a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_68;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_68) : "f"(big_13a), "f"(whole_13a));
            float _abs_67;
            asm("abs.f32 %0, %1;" : "=f"(_abs_67) : "f"(_sub_68));
            float _add_69;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(big_13a), "f"(0.5f));
            float _sub_70;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_70) : "f"(big_13a), "f"(0.5f));
            float rounded_13a = ((_abs_67 == 0.5f) ? ((above_13a > 0.0f) ? _add_69 : _sub_70) : whole_13a);
            t0_13 = ((d0_13 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_13a, 7.888609e-31f), 1.7763568e-15f) : t0_13);
            float d4_13b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_13b) : "f"(d1_13), "f"(0.25f));
            float r4_13b = approx_rcp(d4_13b);
            float _fma_72;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(-d4_13b), "f"(r4_13b), "f"(1.0f));
            float _fma_71;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(r4_13b), "f"(_fma_72), "f"(r4_13b));
            float big_13b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_13b) : "f"(mul_f32_rn_noftz(_fma_71, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_73;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(big_13b), "f"(8.388608e+06f));
            float whole_13b;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_13b) : "f"(_add_73), "f"(8.388608e+06f));
            float above_13b;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_13b) : "f"(-d4_13b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_13b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_75;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_75) : "f"(big_13b), "f"(whole_13b));
            float _abs_74;
            asm("abs.f32 %0, %1;" : "=f"(_abs_74) : "f"(_sub_75));
            float _add_76;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(big_13b), "f"(0.5f));
            float _sub_77;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_77) : "f"(big_13b), "f"(0.5f));
            float rounded_13b = ((_abs_74 == 0.5f) ? ((above_13b > 0.0f) ? _add_76 : _sub_77) : whole_13b);
            t1_13 = ((d1_13 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_13b, 7.888609e-31f), 1.7763568e-15f) : t1_13);
        }
        float2 s_12 = make_float2(max_noftz(t0_13, 0.0f), max_noftz(t1_13, 0.0f));
        float2 act_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_12) : "l"(*(const unsigned long long*)&_cvt_f32_19), "l"(*(const unsigned long long*)&s_12));
        float2 _f32x2_mul_a_78 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_12, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_12), s_12), _cvt_f32_21);
        float2 _mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_78), "l"(*(const unsigned long long*)&_cvt_f32_20));
        float2 _mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_13) : "l"(*(const unsigned long long*)&act_12), "l"(*(const unsigned long long*)&_cvt_f32_21));
        __nv_bfloat162 _bf16x2_12 = __float22bfloat162_rn(make_float2(_mul_12.x, _mul_12.y));
        __nv_bfloat162 _bf16x2_13 = __float22bfloat162_rn(make_float2(_mul_13.x, _mul_13.y));
        dgate[2] = __as_u32(_bf16x2_12);
        dup[2] = __as_u32(_bf16x2_13);
        float2 _cvt_f32_22 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 _cvt_f32_23 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 _cvt_f32_24 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float e0_15 = approx_exp2((-_cvt_f32_22.x) * 1.442695f);
        float e1_15 = approx_exp2((-_cvt_f32_22.y) * 1.442695f);
        float d0_15 = 1.0f + e0_15;
        float d1_15 = 1.0f + e1_15;
        float r0_15 = approx_rcp(d0_15);
        float r1_15 = approx_rcp(d1_15);
        float _fma_79;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(-d0_15), "f"(r0_15), "f"(1.0f));
        float _fma_6_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6_1) : "f"(r0_15), "f"(_fma_79), "f"(r0_15));
        float _fma_80;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(-d1_15), "f"(r1_15), "f"(1.0f));
        float _fma_7_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7_1) : "f"(r1_15), "f"(_fma_80), "f"(r1_15));
        float t0_15 = _fma_6_1;
        float t1_15 = _fma_7_1;
        int _warp_vote_81 = __any_sync(0xFFFFFFFF, fminf(t0_15, t1_15) == 0.0f);
        if (_warp_vote_81 != 0) {
            float d4_15a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_15a) : "f"(d0_15), "f"(0.25f));
            float r4_15a = approx_rcp(d4_15a);
            float _fma_83;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(-d4_15a), "f"(r4_15a), "f"(1.0f));
            float _fma_82;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(r4_15a), "f"(_fma_83), "f"(r4_15a));
            float big_15a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_15a) : "f"(mul_f32_rn_noftz(_fma_82, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_84;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_84) : "f"(big_15a), "f"(8.388608e+06f));
            float whole_15a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_15a) : "f"(_add_84), "f"(8.388608e+06f));
            float above_15a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_15a) : "f"(-d4_15a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_15a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_86;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_86) : "f"(big_15a), "f"(whole_15a));
            float _abs_85;
            asm("abs.f32 %0, %1;" : "=f"(_abs_85) : "f"(_sub_86));
            float _add_87;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(big_15a), "f"(0.5f));
            float _sub_88;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_88) : "f"(big_15a), "f"(0.5f));
            float rounded_15a = ((_abs_85 == 0.5f) ? ((above_15a > 0.0f) ? _add_87 : _sub_88) : whole_15a);
            t0_15 = ((d0_15 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_15a, 7.888609e-31f), 1.7763568e-15f) : t0_15);
            float d4_15b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_15b) : "f"(d1_15), "f"(0.25f));
            float r4_15b = approx_rcp(d4_15b);
            float _fma_90;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(-d4_15b), "f"(r4_15b), "f"(1.0f));
            float _fma_89;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(r4_15b), "f"(_fma_90), "f"(r4_15b));
            float big_15b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_15b) : "f"(mul_f32_rn_noftz(_fma_89, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_91;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_91) : "f"(big_15b), "f"(8.388608e+06f));
            float whole_15b;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_15b) : "f"(_add_91), "f"(8.388608e+06f));
            float above_15b;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_15b) : "f"(-d4_15b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_15b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_93;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_93) : "f"(big_15b), "f"(whole_15b));
            float _abs_92;
            asm("abs.f32 %0, %1;" : "=f"(_abs_92) : "f"(_sub_93));
            float _add_94;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_94) : "f"(big_15b), "f"(0.5f));
            float _sub_95;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_95) : "f"(big_15b), "f"(0.5f));
            float rounded_15b = ((_abs_92 == 0.5f) ? ((above_15b > 0.0f) ? _add_94 : _sub_95) : whole_15b);
            t1_15 = ((d1_15 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_15b, 7.888609e-31f), 1.7763568e-15f) : t1_15);
        }
        float2 s_14 = make_float2(max_noftz(t0_15, 0.0f), max_noftz(t1_15, 0.0f));
        float2 act_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_14) : "l"(*(const unsigned long long*)&_cvt_f32_22), "l"(*(const unsigned long long*)&s_14));
        float2 _f32x2_mul_a_96 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_14, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_14), s_14), _cvt_f32_24);
        float2 _mul_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_96), "l"(*(const unsigned long long*)&_cvt_f32_23));
        float2 _mul_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_15) : "l"(*(const unsigned long long*)&act_14), "l"(*(const unsigned long long*)&_cvt_f32_24));
        __nv_bfloat162 _bf16x2_14 = __float22bfloat162_rn(make_float2(_mul_14.x, _mul_14.y));
        __nv_bfloat162 _bf16x2_15 = __float22bfloat162_rn(make_float2(_mul_15.x, _mul_15.y));
        dgate[3] = __as_u32(_bf16x2_14);
        dup[3] = __as_u32(_bf16x2_15);
    }
    unsigned int chunk_addr = tiles_addr + (unsigned int)(row * 64) + (((unsigned int)quarter ^ (unsigned int)(row >> 3 & 3)) << 4);
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(chunk_addr), "r"(dgate[0]), "r"(dgate[1]), "r"(dgate[2]), "r"(dgate[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(chunk_addr + 2048), "r"(dup[0]), "r"(dup[1]), "r"(dup[2]), "r"(dup[3]) : "memory");
    __syncthreads();
    int lane_1 = tid & 31;
    int warp_1 = tid >> 5;
    int k8 = lane_1 & 7;
    int srow = (k8 >> 1) * 8 + (lane_1 >> 3) * 2 + (k8 & 1);
    unsigned int sswizzle = (unsigned int)(srow >> 3 & 3);
    unsigned int cols[4];
    unsigned int src = tiles_addr + (unsigned int)(srow * 64) + (((unsigned int)warp_1 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[0]), "=r"(cols[1]), "=r"(cols[2]), "=r"(cols[3])
        : "r"(src)
        : "memory");
    __nv_bfloat162 amax = __as_bf16x2(0);
    __nv_bfloat162 _hxa_98 = amax;
    __nv_bfloat162 _hxb_98 = __as_bf16x2(cols[0]);
    unsigned int _hxr_98;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_98) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_98)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_98)));
    __nv_bfloat162 _max_97 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_98);
    amax = _max_97;
    __nv_bfloat162 _hxa_100 = amax;
    __nv_bfloat162 _hxb_100 = __as_bf16x2(cols[1]);
    unsigned int _hxr_100;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_100) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_100)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_100)));
    __nv_bfloat162 _max_99 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_100);
    amax = _max_99;
    __nv_bfloat162 _hxa_102 = amax;
    __nv_bfloat162 _hxb_102 = __as_bf16x2(cols[2]);
    unsigned int _hxr_102;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_102) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_102)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_102)));
    __nv_bfloat162 _max_101 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_102);
    amax = _max_101;
    __nv_bfloat162 _hxa_104 = amax;
    __nv_bfloat162 _hxb_104 = __as_bf16x2(cols[3]);
    unsigned int _hxr_104;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_104) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_104)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_104)));
    __nv_bfloat162 _max_103 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_104);
    amax = _max_103;
    unsigned int mag = __as_u32(amax) & 2147450879;
    __nv_bfloat162 col_max = __hmax2_nan(__as_bf16x2(mag), __as_bf16x2(mag >> 16));
    unsigned int peer1 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(col_max), 1);
    col_max = __hmax2_nan(col_max, __as_bf16x2(peer1));
    unsigned int peer2 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(col_max), 2);
    col_max = __hmax2_nan(col_max, __as_bf16x2(peer2));
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
    unsigned int words[2];
    uint32_t _bf16x2_mul_0;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(cols[0]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_1) : "r"(cols[1]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_106;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_106) : "r"(_bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_106) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_105;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_105) : "h"(_e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_106));
    words[0] = _mov_b32_pack_u16x2_105;
    uint32_t _bf16x2_mul_2;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_2) : "r"(cols[2]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_3) : "r"(cols[3]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_108;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_108) : "r"(_bf16x2_mul_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_108) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_107;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_107) : "h"(_e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_108));
    words[1] = _mov_b32_pack_u16x2_107;
    reinterpret_cast<int2*>(col_output + (gcol * M + out_row))[0] = reinterpret_cast<int2*>(words + 0)[0];
    if ((lane_1 & 3) == 0) {
        int scale_index = ((gcol >> 7) * (M >> 7) + (by >> 2)) * 512 + (gcol & 31) * 16 + (gcol >> 5 & 3) * 4 + (by & 3);
        *(reinterpret_cast<unsigned char*>(col_scales + scale_index) + (0)) = (unsigned char)(scale);
    }
    int lane_0 = tid & 31;
    int warp_1_1 = tid >> 5;
    int k8_2 = lane_0 & 7;
    int srow_3 = (k8_2 >> 1) * 8 + (lane_0 >> 3) * 2 + (k8_2 & 1);
    unsigned int sswizzle_4 = (unsigned int)(srow_3 >> 3 & 3);
    unsigned int cols_5[4];
    unsigned int src_6 = tiles_addr + 2048 + (unsigned int)(srow_3 * 64) + (((unsigned int)warp_1_1 ^ sswizzle_4) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_5[0]), "=r"(cols_5[1]), "=r"(cols_5[2]), "=r"(cols_5[3])
        : "r"(src_6)
        : "memory");
    __nv_bfloat162 amax_7 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_110 = amax_7;
    __nv_bfloat162 _hxb_110 = __as_bf16x2(cols_5[0]);
    unsigned int _hxr_110;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_110) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_110)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_110)));
    __nv_bfloat162 _max_109 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_110);
    amax_7 = _max_109;
    __nv_bfloat162 _hxa_112 = amax_7;
    __nv_bfloat162 _hxb_112 = __as_bf16x2(cols_5[1]);
    unsigned int _hxr_112;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_112) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_112)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_112)));
    __nv_bfloat162 _max_111 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_112);
    amax_7 = _max_111;
    __nv_bfloat162 _hxa_114 = amax_7;
    __nv_bfloat162 _hxb_114 = __as_bf16x2(cols_5[2]);
    unsigned int _hxr_114;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_114) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_114)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_114)));
    __nv_bfloat162 _max_113 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_114);
    amax_7 = _max_113;
    __nv_bfloat162 _hxa_116 = amax_7;
    __nv_bfloat162 _hxb_116 = __as_bf16x2(cols_5[3]);
    unsigned int _hxr_116;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_116) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_116)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_116)));
    __nv_bfloat162 _max_115 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_116);
    amax_7 = _max_115;
    unsigned int mag_8 = __as_u32(amax_7) & 2147450879;
    __nv_bfloat162 col_max_9 = __hmax2_nan(__as_bf16x2(mag_8), __as_bf16x2(mag_8 >> 16));
    unsigned int peer1_10 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(col_max_9), 1);
    col_max_9 = __hmax2_nan(col_max_9, __as_bf16x2(peer1_10));
    unsigned int peer2_11 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(col_max_9), 2);
    col_max_9 = __hmax2_nan(col_max_9, __as_bf16x2(peer2_11));
    unsigned int amax_bits_12 = (__as_u32(col_max_9) & 65535) << 16;
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
    int gcol_17 = bx * 32 + K + warp_1_1 * 8 + (lane_0 >> 2);
    int out_row_18 = by * 32 + (lane_0 & 3) * 8;
    unsigned int words_19[2];
    uint32_t _bf16x2_mul_4;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_4) : "r"(cols_5[0]), "r"(inv2_16));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_5) : "r"(cols_5[1]), "r"(inv2_16));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_118;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_118) : "r"(_bf16x2_mul_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_118) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_117;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_117) : "h"(_e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_118));
    words_19[0] = _mov_b32_pack_u16x2_117;
    uint32_t _bf16x2_mul_6;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_6) : "r"(cols_5[2]), "r"(inv2_16));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_7) : "r"(cols_5[3]), "r"(inv2_16));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_120;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_120) : "r"(_bf16x2_mul_7));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_7);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_120) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_119;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_119) : "h"(_e4m3x2_bf16x2_3), "h"(_cvt_satfinite_e4m3x2_bf16x2_120));
    words_19[1] = _mov_b32_pack_u16x2_119;
    reinterpret_cast<int2*>(col_output + (gcol_17 * M + out_row_18))[0] = reinterpret_cast<int2*>(words_19 + 0)[0];
    if ((lane_0 & 3) == 0) {
        int scale_index_1 = ((gcol_17 >> 7) * (M >> 7) + (by >> 2)) * 512 + (gcol_17 & 31) * 16 + (gcol_17 >> 5 & 3) * 4 + (by & 3);
        *(reinterpret_cast<unsigned char*>(col_scales + scale_index_1) + (0)) = (unsigned char)(scale_14);
    }
}

} // extern "C"
