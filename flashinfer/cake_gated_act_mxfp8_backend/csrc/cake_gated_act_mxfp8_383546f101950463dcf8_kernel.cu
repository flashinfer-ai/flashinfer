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
#define LAUNCH_MIN_BLOCKS 12

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
kernel_cake_gated_act_mxfp8_383546f101950463dcf8(__nv_bfloat16* __restrict__ gated_input, __nv_bfloat16* __restrict__ grad, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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
    } else {
        int band = 0;
        float2 cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 cvt_f32_1_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 cvt_f32_2_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float e0_3 = approx_exp2((-cvt_f32_5.x) * 1.442695f);
        float e1_3 = approx_exp2((-cvt_f32_5.y) * 1.442695f);
        float d0_3 = 1.0f + e0_3;
        float d1_3 = 1.0f + e1_3;
        float r0_3 = approx_rcp(d0_3);
        float r1_3 = approx_rcp(d1_3);
        float _fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(-d0_3), "f"(r0_3), "f"(1.0f));
        float fma;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma) : "f"(r0_3), "f"(_fma_25), "f"(r0_3));
        float _fma_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(-d1_3), "f"(r1_3), "f"(1.0f));
        float fma_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1) : "f"(r1_3), "f"(_fma_26), "f"(r1_3));
        float t0 = fma;
        float t1 = fma_1;
        band = band | ((fminf(t0, t1) == 0.0f) ? 1 : 0);
        float2 s_3 = make_float2(max_noftz(t0, 0.0f), max_noftz(t1, 0.0f));
        float2 act_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_3) : "l"(*(const unsigned long long*)&cvt_f32_5), "l"(*(const unsigned long long*)&s_3));
        float2 _f32x2_mul_a_27 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_3), s_3), cvt_f32_2_3);
        float2 mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_27), "l"(*(const unsigned long long*)&cvt_f32_1_3));
        float2 mul_1_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_3) : "l"(*(const unsigned long long*)&act_3), "l"(*(const unsigned long long*)&cvt_f32_2_3));
        __nv_bfloat162 bf16x2_4 = __float22bfloat162_rn(make_float2(mul_4.x, mul_4.y));
        __nv_bfloat162 bf16x2_1_3 = __float22bfloat162_rn(make_float2(mul_1_3.x, mul_1_3.y));
        dgate[0] = __as_u32(bf16x2_4);
        dup[0] = __as_u32(bf16x2_1_3);
        float2 cvt_f32_5_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 cvt_f32_1_3_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 cvt_f32_2_3_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float e0_3_1 = approx_exp2((-cvt_f32_5_1.x) * 1.442695f);
        float e1_3_1 = approx_exp2((-cvt_f32_5_1.y) * 1.442695f);
        float d0_3_1 = 1.0f + e0_3_1;
        float d1_3_1 = 1.0f + e1_3_1;
        float r0_3_1 = approx_rcp(d0_3_1);
        float r1_3_1 = approx_rcp(d1_3_1);
        float _fma_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(-d0_3_1), "f"(r0_3_1), "f"(1.0f));
        float fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_0) : "f"(r0_3_1), "f"(_fma_28), "f"(r0_3_1));
        float _fma_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(-d1_3_1), "f"(r1_3_1), "f"(1.0f));
        float fma_1_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_0) : "f"(r1_3_1), "f"(_fma_29), "f"(r1_3_1));
        float t0_0 = fma_0;
        float t1_0 = fma_1_0;
        band = band | ((fminf(t0_0, t1_0) == 0.0f) ? 2 : 0);
        float2 s_3_1 = make_float2(max_noftz(t0_0, 0.0f), max_noftz(t1_0, 0.0f));
        float2 act_3_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_3_1) : "l"(*(const unsigned long long*)&cvt_f32_5_1), "l"(*(const unsigned long long*)&s_3_1));
        float2 _f32x2_mul_a_30 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_3_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_3_1), s_3_1), cvt_f32_2_3_1);
        float2 mul_4_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&cvt_f32_1_3_1));
        float2 mul_1_3_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_3_1) : "l"(*(const unsigned long long*)&act_3_1), "l"(*(const unsigned long long*)&cvt_f32_2_3_1));
        __nv_bfloat162 bf16x2_4_1 = __float22bfloat162_rn(make_float2(mul_4_1.x, mul_4_1.y));
        __nv_bfloat162 bf16x2_1_3_1 = __float22bfloat162_rn(make_float2(mul_1_3_1.x, mul_1_3_1.y));
        dgate[1] = __as_u32(bf16x2_4_1);
        dup[1] = __as_u32(bf16x2_1_3_1);
        float2 cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 cvt_f32_1_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 cvt_f32_2_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float e0_4 = approx_exp2((-cvt_f32_6.x) * 1.442695f);
        float e1_4 = approx_exp2((-cvt_f32_6.y) * 1.442695f);
        float d0_4 = 1.0f + e0_4;
        float d1_4 = 1.0f + e1_4;
        float r0_4 = approx_rcp(d0_4);
        float r1_4 = approx_rcp(d1_4);
        float _fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(-d0_4), "f"(r0_4), "f"(1.0f));
        float fma_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_2) : "f"(r0_4), "f"(_fma_31), "f"(r0_4));
        float _fma_32;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(-d1_4), "f"(r1_4), "f"(1.0f));
        float fma_1_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_1) : "f"(r1_4), "f"(_fma_32), "f"(r1_4));
        float t0_1 = fma_2;
        float t1_1 = fma_1_1;
        band = band | ((fminf(t0_1, t1_1) == 0.0f) ? 4 : 0);
        float2 s_4 = make_float2(max_noftz(t0_1, 0.0f), max_noftz(t1_1, 0.0f));
        float2 act_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_4) : "l"(*(const unsigned long long*)&cvt_f32_6), "l"(*(const unsigned long long*)&s_4));
        float2 _f32x2_mul_a_33 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_4), s_4), cvt_f32_2_4);
        float2 mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&cvt_f32_1_4));
        float2 mul_1_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_4) : "l"(*(const unsigned long long*)&act_4), "l"(*(const unsigned long long*)&cvt_f32_2_4));
        __nv_bfloat162 bf16x2_5 = __float22bfloat162_rn(make_float2(mul_5.x, mul_5.y));
        __nv_bfloat162 bf16x2_1_4 = __float22bfloat162_rn(make_float2(mul_1_4.x, mul_1_4.y));
        dgate[2] = __as_u32(bf16x2_5);
        dup[2] = __as_u32(bf16x2_1_4);
        float2 cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 cvt_f32_1_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 cvt_f32_2_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float e0_5 = approx_exp2((-cvt_f32_7.x) * 1.442695f);
        float e1_5 = approx_exp2((-cvt_f32_7.y) * 1.442695f);
        float d0_5 = 1.0f + e0_5;
        float d1_5 = 1.0f + e1_5;
        float r0_5 = approx_rcp(d0_5);
        float r1_5 = approx_rcp(d1_5);
        float _fma_34;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(-d0_5), "f"(r0_5), "f"(1.0f));
        float fma_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_3) : "f"(r0_5), "f"(_fma_34), "f"(r0_5));
        float _fma_35;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(-d1_5), "f"(r1_5), "f"(1.0f));
        float fma_1_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_2) : "f"(r1_5), "f"(_fma_35), "f"(r1_5));
        float t0_2 = fma_3;
        float t1_2 = fma_1_2;
        band = band | ((fminf(t0_2, t1_2) == 0.0f) ? 8 : 0);
        float2 s_5 = make_float2(max_noftz(t0_2, 0.0f), max_noftz(t1_2, 0.0f));
        float2 act_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_5) : "l"(*(const unsigned long long*)&cvt_f32_7), "l"(*(const unsigned long long*)&s_5));
        float2 _f32x2_mul_a_36 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_5), s_5), cvt_f32_2_5);
        float2 mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_36), "l"(*(const unsigned long long*)&cvt_f32_1_5));
        float2 mul_1_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_5) : "l"(*(const unsigned long long*)&act_5), "l"(*(const unsigned long long*)&cvt_f32_2_5));
        __nv_bfloat162 bf16x2_6 = __float22bfloat162_rn(make_float2(mul_6.x, mul_6.y));
        __nv_bfloat162 bf16x2_1_5 = __float22bfloat162_rn(make_float2(mul_1_5.x, mul_1_5.y));
        dgate[3] = __as_u32(bf16x2_6);
        dup[3] = __as_u32(bf16x2_1_5);
        if (band != 0) {
            unsigned int _popc_0 = (unsigned int)__popc(band);
            #pragma unroll 1
            for (int _j = 0; _j < _popc_0; _j++) {
                int _find_first_set_37 = __ffs(band);
                int j = _find_first_set_37 - 1;
                band = band & band - 1;
                unsigned int _vec_load_3[1];
                {
                    _vec_load_3[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index + 2 * j) + 0);
                }
                unsigned int _vec_load_4[1];
                {
                    _vec_load_4[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index + K + 2 * j) + 0);
                }
                unsigned int _vec_load_5[1];
                {
                    _vec_load_5[0] = *reinterpret_cast<const unsigned int*>(grad + (grow * K + col + 2 * j) + 0);
                }
                float2 cvt_f32_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
                float2 cvt_f32_1_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[0]));
                float2 cvt_f32_2_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[0]));
                float e0_6 = approx_exp2((-cvt_f32_8.x) * 1.442695f);
                float e1_6 = approx_exp2((-cvt_f32_8.y) * 1.442695f);
                float d0_6 = 1.0f + e0_6;
                float d1_6 = 1.0f + e1_6;
                float r0_6 = approx_rcp(d0_6);
                float r1_6 = approx_rcp(d1_6);
                float _fma_38;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(-d0_6), "f"(r0_6), "f"(1.0f));
                float fma_4;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_4) : "f"(r0_6), "f"(_fma_38), "f"(r0_6));
                float _fma_39;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(-d1_6), "f"(r1_6), "f"(1.0f));
                float fma_1_3;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_3) : "f"(r1_6), "f"(_fma_39), "f"(r1_6));
                float t0_3 = fma_4;
                float t1_3 = fma_1_3;
                float d4_1a;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a) : "f"(d0_6), "f"(0.25f));
                float r4_1a = approx_rcp(d4_1a);
                float _fma_41;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(-d4_1a), "f"(r4_1a), "f"(1.0f));
                float _fma_40;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(r4_1a), "f"(_fma_41), "f"(r4_1a));
                float big_1a;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a) : "f"(mul_f32_rn_noftz(_fma_40, 1.2676506e+30f)), "f"(1.4073749e+14f));
                float _add_42;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(big_1a), "f"(8.388608e+06f));
                float whole_1a;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a) : "f"(_add_42), "f"(8.388608e+06f));
                float above_1a;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a) : "f"(-d4_1a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                float _sub_44;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44) : "f"(big_1a), "f"(whole_1a));
                float _abs_43;
                asm("abs.f32 %0, %1;" : "=f"(_abs_43) : "f"(_sub_44));
                float _add_45;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(big_1a), "f"(0.5f));
                float _sub_46;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_46) : "f"(big_1a), "f"(0.5f));
                float rounded_1a = ((_abs_43 == 0.5f) ? ((above_1a > 0.0f) ? _add_45 : _sub_46) : whole_1a);
                float u0 = ((d0_6 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a, 7.888609e-31f), 1.7763568e-15f) : t0_3);
                float d4_1a_0;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_0) : "f"(d1_6), "f"(0.25f));
                float r4_1a_0 = approx_rcp(d4_1a_0);
                float _fma_48;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(-d4_1a_0), "f"(r4_1a_0), "f"(1.0f));
                float _fma_47;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(r4_1a_0), "f"(_fma_48), "f"(r4_1a_0));
                float big_1a_0;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_0) : "f"(mul_f32_rn_noftz(_fma_47, 1.2676506e+30f)), "f"(1.4073749e+14f));
                float _add_49;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(big_1a_0), "f"(8.388608e+06f));
                float whole_1a_0;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_0) : "f"(_add_49), "f"(8.388608e+06f));
                float above_1a_0;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_0) : "f"(-d4_1a_0), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_0, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                float _sub_51;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_51) : "f"(big_1a_0), "f"(whole_1a_0));
                float _abs_50;
                asm("abs.f32 %0, %1;" : "=f"(_abs_50) : "f"(_sub_51));
                float _add_52;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(big_1a_0), "f"(0.5f));
                float _sub_53;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_53) : "f"(big_1a_0), "f"(0.5f));
                float rounded_1a_0 = ((_abs_50 == 0.5f) ? ((above_1a_0 > 0.0f) ? _add_52 : _sub_53) : whole_1a_0);
                float u1 = ((d1_6 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_0, 7.888609e-31f), 1.7763568e-15f) : t1_3);
                float2 s_6 = make_float2(max_noftz(u0, 0.0f), max_noftz(u1, 0.0f));
                float2 act_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_6) : "l"(*(const unsigned long long*)&cvt_f32_8), "l"(*(const unsigned long long*)&s_6));
                float2 _f32x2_mul_a_54 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_6), s_6), cvt_f32_2_6);
                float2 mul_7;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_54), "l"(*(const unsigned long long*)&cvt_f32_1_6));
                float2 mul_1_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_6) : "l"(*(const unsigned long long*)&act_6), "l"(*(const unsigned long long*)&cvt_f32_2_6));
                __nv_bfloat162 bf16x2_7 = __float22bfloat162_rn(make_float2(mul_7.x, mul_7.y));
                __nv_bfloat162 bf16x2_1_6 = __float22bfloat162_rn(make_float2(mul_1_6.x, mul_1_6.y));
                unsigned int dgate_new = __as_u32(bf16x2_7);
                unsigned int dup_new = __as_u32(bf16x2_1_6);
                dgate[0] = ((j == 0) ? dgate_new : dgate[0]);
                dup[0] = ((j == 0) ? dup_new : dup[0]);
                dgate[1] = ((j == 1) ? dgate_new : dgate[1]);
                dup[1] = ((j == 1) ? dup_new : dup[1]);
                dgate[2] = ((j == 2) ? dgate_new : dgate[2]);
                dup[2] = ((j == 2) ? dup_new : dup[2]);
                dgate[3] = ((j == 3) ? dgate_new : dgate[3]);
                dup[3] = ((j == 3) ? dup_new : dup[3]);
            }
        }
    }
    unsigned int chunk_addr = tiles_addr + (unsigned int)(row * 64) + (((unsigned int)quarter ^ (unsigned int)(row >> 3 & 3)) << 4);
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(chunk_addr), "r"(dgate[0]), "r"(dgate[1]), "r"(dgate[2]), "r"(dgate[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(chunk_addr + 2048), "r"(dup[0]), "r"(dup[1]), "r"(dup[2]), "r"(dup[3]) : "memory");
    int scale_cols = K >> 6;
    int out_index = grow * (2 * K) + col;
    __nv_bfloat162 amax = __as_bf16x2(0);
    __nv_bfloat162 _hxa_56 = amax;
    __nv_bfloat162 _hxb_56 = __as_bf16x2(dgate[0]);
    unsigned int _hxr_56;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_56) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_56)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_56)));
    __nv_bfloat162 _max_55 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_56);
    amax = _max_55;
    __nv_bfloat162 _hxa_58 = amax;
    __nv_bfloat162 _hxb_58 = __as_bf16x2(dgate[1]);
    unsigned int _hxr_58;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_58) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_58)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_58)));
    __nv_bfloat162 _max_57 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_58);
    amax = _max_57;
    __nv_bfloat162 _hxa_60 = amax;
    __nv_bfloat162 _hxb_60 = __as_bf16x2(dgate[2]);
    unsigned int _hxr_60;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_60) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_60)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_60)));
    __nv_bfloat162 _max_59 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_60);
    amax = _max_59;
    __nv_bfloat162 _hxa_62 = amax;
    __nv_bfloat162 _hxb_62 = __as_bf16x2(dgate[3]);
    unsigned int _hxr_62;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_62) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_62)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_62)));
    __nv_bfloat162 _max_61 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_62);
    amax = _max_61;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_64;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_64) : "r"(bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_64) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_63;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_63) : "h"(e4m3x2_bf16x2), "h"(_cvt_satfinite_e4m3x2_bf16x2_64));
    words[0] = _mov_b32_pack_u16x2_63;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_66;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_66) : "r"(bf16x2_mul_1_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_66) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_65;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_65) : "h"(e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_66));
    words[1] = _mov_b32_pack_u16x2_65;
    reinterpret_cast<int2*>(row_output + out_index)[0] = reinterpret_cast<int2*>(words + 0)[0];
    __nv_bfloat162 amax_0 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_68 = amax_0;
    __nv_bfloat162 _hxb_68 = __as_bf16x2(dup[0]);
    unsigned int _hxr_68;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_68) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_68)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_68)));
    __nv_bfloat162 _max_67 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_68);
    amax_0 = _max_67;
    __nv_bfloat162 _hxa_70 = amax_0;
    __nv_bfloat162 _hxb_70 = __as_bf16x2(dup[1]);
    unsigned int _hxr_70;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_70) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_70)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_70)));
    __nv_bfloat162 _max_69 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_70);
    amax_0 = _max_69;
    __nv_bfloat162 _hxa_72 = amax_0;
    __nv_bfloat162 _hxb_72 = __as_bf16x2(dup[2]);
    unsigned int _hxr_72;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_72) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_72)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_72)));
    __nv_bfloat162 _max_71 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_72);
    amax_0 = _max_71;
    __nv_bfloat162 _hxa_74 = amax_0;
    __nv_bfloat162 _hxb_74 = __as_bf16x2(dup[3]);
    unsigned int _hxr_74;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_74) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_74)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_74)));
    __nv_bfloat162 _max_73 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_74);
    amax_0 = _max_73;
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
    uint32_t bf16x2_mul_2;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2) : "r"(dup[0]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_1) : "r"(dup[1]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_76;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_76) : "r"(bf16x2_mul_1_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_76) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_75;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_75) : "h"(e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_76));
    words_0[0] = _mov_b32_pack_u16x2_75;
    uint32_t bf16x2_mul_3;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3) : "r"(dup[2]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_2) : "r"(dup[3]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_78;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_78) : "r"(bf16x2_mul_1_2));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_2);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_78) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_77;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_77) : "h"(e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_78));
    words_0[1] = _mov_b32_pack_u16x2_77;
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
    __nv_bfloat162 _hxa_80 = amax_1;
    __nv_bfloat162 _hxb_80 = __as_bf16x2(cols_1[0]);
    unsigned int _hxr_80;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_80) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_80)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_80)));
    __nv_bfloat162 _max_79 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_80);
    amax_1 = _max_79;
    __nv_bfloat162 _hxa_82 = amax_1;
    __nv_bfloat162 _hxb_82 = __as_bf16x2(cols_1[1]);
    unsigned int _hxr_82;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_82) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_82)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_82)));
    __nv_bfloat162 _max_81 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_82);
    amax_1 = _max_81;
    __nv_bfloat162 _hxa_84 = amax_1;
    __nv_bfloat162 _hxb_84 = __as_bf16x2(cols_1[2]);
    unsigned int _hxr_84;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_84) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_84)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_84)));
    __nv_bfloat162 _max_83 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_84);
    amax_1 = _max_83;
    __nv_bfloat162 _hxa_86 = amax_1;
    __nv_bfloat162 _hxb_86 = __as_bf16x2(cols_1[3]);
    unsigned int _hxr_86;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_86) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_86)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_86)));
    __nv_bfloat162 _max_85 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_86);
    amax_1 = _max_85;
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
    unsigned int scale_1 = (unsigned int)scale_i32_1;
    unsigned int inv_1 = 254 - scale_1 << 7;
    if (scale_1 == 255) {
        inv_1 = 32704;
    }
    unsigned int inv2_1 = inv_1 | inv_1 << 16;
    int gcol = bx * 32 + warp_1 * 8 + (lane_1 >> 2);
    int out_row = by * 32 + (lane_1 & 3) * 8;
    unsigned int words_1[2];
    uint32_t bf16x2_mul_4;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4) : "r"(cols_1[0]), "r"(inv2_1));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_3) : "r"(cols_1[1]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_88;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_88) : "r"(bf16x2_mul_1_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_88) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_87;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_87) : "h"(e4m3x2_bf16x2_3), "h"(_cvt_satfinite_e4m3x2_bf16x2_88));
    words_1[0] = _mov_b32_pack_u16x2_87;
    uint32_t bf16x2_mul_5;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5) : "r"(cols_1[2]), "r"(inv2_1));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_4) : "r"(cols_1[3]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_90;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_90) : "r"(bf16x2_mul_1_4));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_4);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_90) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_89;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_89) : "h"(e4m3x2_bf16x2_4), "h"(_cvt_satfinite_e4m3x2_bf16x2_90));
    words_1[1] = _mov_b32_pack_u16x2_89;
    reinterpret_cast<int2*>(col_output + (gcol * M + out_row))[0] = reinterpret_cast<int2*>(words_1 + 0)[0];
    if ((lane_1 & 3) == 0) {
        int scale_index_1 = ((gcol >> 7) * (M >> 7) + (by >> 2)) * 512 + (gcol & 31) * 16 + (gcol >> 5 & 3) * 4 + (by & 3);
        *(reinterpret_cast<unsigned char*>(col_scales + scale_index_1) + (0)) = (unsigned char)(scale_1);
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
    __nv_bfloat162 _hxa_92 = amax_2;
    __nv_bfloat162 _hxb_92 = __as_bf16x2(cols_1_0[0]);
    unsigned int _hxr_92;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_92) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_92)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_92)));
    __nv_bfloat162 _max_91 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_92);
    amax_2 = _max_91;
    __nv_bfloat162 _hxa_94 = amax_2;
    __nv_bfloat162 _hxb_94 = __as_bf16x2(cols_1_0[1]);
    unsigned int _hxr_94;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_94) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_94)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_94)));
    __nv_bfloat162 _max_93 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_94);
    amax_2 = _max_93;
    __nv_bfloat162 _hxa_96 = amax_2;
    __nv_bfloat162 _hxb_96 = __as_bf16x2(cols_1_0[2]);
    unsigned int _hxr_96;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_96) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_96)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_96)));
    __nv_bfloat162 _max_95 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_96);
    amax_2 = _max_95;
    __nv_bfloat162 _hxa_98 = amax_2;
    __nv_bfloat162 _hxb_98 = __as_bf16x2(cols_1_0[3]);
    unsigned int _hxr_98;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_98) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_98)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_98)));
    __nv_bfloat162 _max_97 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_98);
    amax_2 = _max_97;
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
    unsigned int scale_2 = (unsigned int)scale_i32_2;
    unsigned int inv_2 = 254 - scale_2 << 7;
    if (scale_2 == 255) {
        inv_2 = 32704;
    }
    unsigned int inv2_2 = inv_2 | inv_2 << 16;
    int gcol_0 = bx * 32 + K + warp_0 * 8 + (lane_0 >> 2);
    int out_row_0 = by * 32 + (lane_0 & 3) * 8;
    unsigned int words_2[2];
    uint32_t bf16x2_mul_6;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6) : "r"(cols_1_0[0]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_5) : "r"(cols_1_0[1]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_100;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_100) : "r"(bf16x2_mul_1_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_100) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_99;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_99) : "h"(e4m3x2_bf16x2_5), "h"(_cvt_satfinite_e4m3x2_bf16x2_100));
    words_2[0] = _mov_b32_pack_u16x2_99;
    uint32_t bf16x2_mul_7;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7) : "r"(cols_1_0[2]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_6) : "r"(cols_1_0[3]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_102;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_102) : "r"(bf16x2_mul_1_6));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_6);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_102) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_101;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_101) : "h"(e4m3x2_bf16x2_6), "h"(_cvt_satfinite_e4m3x2_bf16x2_102));
    words_2[1] = _mov_b32_pack_u16x2_101;
    reinterpret_cast<int2*>(col_output + (gcol_0 * M + out_row_0))[0] = reinterpret_cast<int2*>(words_2 + 0)[0];
    if ((lane_0 & 3) == 0) {
        int scale_index_2 = ((gcol_0 >> 7) * (M >> 7) + (by >> 2)) * 512 + (gcol_0 & 31) * 16 + (gcol_0 >> 5 & 3) * 4 + (by & 3);
        *(reinterpret_cast<unsigned char*>(col_scales + scale_index_2) + (0)) = (unsigned char)(scale_2);
    }
}

} // extern "C"
