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
#define SMEM_TILE_OFF 0
#define SMEM_TILE_STAGE_BYTES 16384
#define SMEM_TILE_STRIDE 16384
#define SMEM_TOTAL 16384
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 5

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

__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_gated_act_mxfp8_0b9dfb9aa4b837a6b7a4(__nv_bfloat16* __restrict__ gated_input, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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
    int grow_hi = grow + 32;
    int gate_index_hi = grow_hi * (2 * K) + col;
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
            : "=r"(_iv_2_0), "=r"(_iv_2_1), "=r"(_iv_2_2), "=r"(_iv_2_3), "=r"(_iv_2_4), "=r"(_iv_2_5), "=r"(_iv_2_6), "=r"(_iv_2_7) : "l"((const void*)(gated_input + gate_index_hi + (0))) : "memory");
        _vec_load_2[0 + 0] = (uint32_t)_iv_2_0;
        _vec_load_2[0 + 1] = (uint32_t)_iv_2_1;
        _vec_load_2[0 + 2] = (uint32_t)_iv_2_2;
        _vec_load_2[0 + 3] = (uint32_t)_iv_2_3;
        _vec_load_2[0 + 4] = (uint32_t)_iv_2_4;
        _vec_load_2[0 + 5] = (uint32_t)_iv_2_5;
        _vec_load_2[0 + 6] = (uint32_t)_iv_2_6;
        _vec_load_2[0 + 7] = (uint32_t)_iv_2_7;
    }
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
            : "=r"(_iv_3_0), "=r"(_iv_3_1), "=r"(_iv_3_2), "=r"(_iv_3_3), "=r"(_iv_3_4), "=r"(_iv_3_5), "=r"(_iv_3_6), "=r"(_iv_3_7) : "l"((const void*)(gated_input + (gate_index_hi + K) + (0))) : "memory");
        _vec_load_3[0 + 0] = (uint32_t)_iv_3_0;
        _vec_load_3[0 + 1] = (uint32_t)_iv_3_1;
        _vec_load_3[0 + 2] = (uint32_t)_iv_3_2;
        _vec_load_3[0 + 3] = (uint32_t)_iv_3_3;
        _vec_load_3[0 + 4] = (uint32_t)_iv_3_4;
        _vec_load_3[0 + 5] = (uint32_t)_iv_3_5;
        _vec_load_3[0 + 6] = (uint32_t)_iv_3_6;
        _vec_load_3[0 + 7] = (uint32_t)_iv_3_7;
    }
    unsigned int packed[8];
    unsigned int packed_hi[8];
    __nv_bfloat162 lowest = __as_bf16x2(_vec_load_0[0]);
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[7]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_2[0]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_2[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_2[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_2[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_2[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_2[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_2[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_2[7]));
    float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(lowest);
    float _min_5;
    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_5) : "f"(_cvt_f32_0.x), "f"(_cvt_f32_0.y));
    int _warp_vote_4 = __all_sync(0xFFFFFFFF, _min_5 > -87.0f);
    if (_warp_vote_4 != 0) {
        float2 _cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 _cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float e0_0 = approx_exp2((-_cvt_f32_1.x) * 1.442695f);
        float e1_0 = approx_exp2((-_cvt_f32_1.y) * 1.442695f);
        float d0_0 = 1.0f + e0_0;
        float d1_0 = 1.0f + e1_0;
        float r0_0 = approx_rcp(d0_0);
        float r1_0 = approx_rcp(d1_0);
        float _fma_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(-d0_0), "f"(r0_0), "f"(1.0f));
        float _fma_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(r0_0), "f"(_fma_7), "f"(r0_0));
        float _fma_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(-d1_0), "f"(r1_0), "f"(1.0f));
        float _fma_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(r1_0), "f"(_fma_9), "f"(r1_0));
        float2 _f32x2_mul_a_10 = mul_f32x2_noftz(_cvt_f32_1, make_float2(_fma_6, _fma_8));
        float2 _mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_10), "l"(*(const unsigned long long*)&_cvt_f32_2));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(_mul_0.x, _mul_0.y)));
        float2 _cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 _cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float e0_1 = approx_exp2((-_cvt_f32_3.x) * 1.442695f);
        float e1_1 = approx_exp2((-_cvt_f32_3.y) * 1.442695f);
        float d0_1 = 1.0f + e0_1;
        float d1_1 = 1.0f + e1_1;
        float r0_1 = approx_rcp(d0_1);
        float r1_1 = approx_rcp(d1_1);
        float _fma_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(-d0_1), "f"(r0_1), "f"(1.0f));
        float _fma_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(r0_1), "f"(_fma_12), "f"(r0_1));
        float _fma_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(-d1_1), "f"(r1_1), "f"(1.0f));
        float _fma_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(r1_1), "f"(_fma_14), "f"(r1_1));
        float2 _f32x2_mul_a_15 = mul_f32x2_noftz(_cvt_f32_3, make_float2(_fma_11, _fma_13));
        float2 _mul_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&_cvt_f32_4));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(_mul_1.x, _mul_1.y)));
        float2 _cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 _cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float e0_2 = approx_exp2((-_cvt_f32_5.x) * 1.442695f);
        float e1_2 = approx_exp2((-_cvt_f32_5.y) * 1.442695f);
        float d0_2 = 1.0f + e0_2;
        float d1_2 = 1.0f + e1_2;
        float r0_2 = approx_rcp(d0_2);
        float r1_2 = approx_rcp(d1_2);
        float _fma_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(-d0_2), "f"(r0_2), "f"(1.0f));
        float _fma_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(r0_2), "f"(_fma_17), "f"(r0_2));
        float _fma_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(-d1_2), "f"(r1_2), "f"(1.0f));
        float _fma_18;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(r1_2), "f"(_fma_19), "f"(r1_2));
        float2 _f32x2_mul_a_20 = mul_f32x2_noftz(_cvt_f32_5, make_float2(_fma_16, _fma_18));
        float2 _mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_20), "l"(*(const unsigned long long*)&_cvt_f32_6));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(_mul_2.x, _mul_2.y)));
        float2 _cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 _cvt_f32_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float e0_3 = approx_exp2((-_cvt_f32_7.x) * 1.442695f);
        float e1_3 = approx_exp2((-_cvt_f32_7.y) * 1.442695f);
        float d0_3 = 1.0f + e0_3;
        float d1_3 = 1.0f + e1_3;
        float r0_3 = approx_rcp(d0_3);
        float r1_3 = approx_rcp(d1_3);
        float _fma_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(-d0_3), "f"(r0_3), "f"(1.0f));
        float _fma_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(r0_3), "f"(_fma_22), "f"(r0_3));
        float _fma_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(-d1_3), "f"(r1_3), "f"(1.0f));
        float _fma_23;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(r1_3), "f"(_fma_24), "f"(r1_3));
        float2 _f32x2_mul_a_25 = mul_f32x2_noftz(_cvt_f32_7, make_float2(_fma_21, _fma_23));
        float2 _mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_25), "l"(*(const unsigned long long*)&_cvt_f32_8));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(_mul_3.x, _mul_3.y)));
        float2 _cvt_f32_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 _cvt_f32_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float e0_4 = approx_exp2((-_cvt_f32_9.x) * 1.442695f);
        float e1_4 = approx_exp2((-_cvt_f32_9.y) * 1.442695f);
        float d0_4 = 1.0f + e0_4;
        float d1_4 = 1.0f + e1_4;
        float r0_4 = approx_rcp(d0_4);
        float r1_4 = approx_rcp(d1_4);
        float _fma_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(-d0_4), "f"(r0_4), "f"(1.0f));
        float _fma_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(r0_4), "f"(_fma_27), "f"(r0_4));
        float _fma_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(-d1_4), "f"(r1_4), "f"(1.0f));
        float _fma_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(r1_4), "f"(_fma_29), "f"(r1_4));
        float2 _f32x2_mul_a_30 = mul_f32x2_noftz(_cvt_f32_9, make_float2(_fma_26, _fma_28));
        float2 _mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&_cvt_f32_10));
        packed[4] = __as_u32(__float22bfloat162_rn(make_float2(_mul_4.x, _mul_4.y)));
        float2 _cvt_f32_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 _cvt_f32_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float e0_5 = approx_exp2((-_cvt_f32_11.x) * 1.442695f);
        float e1_5 = approx_exp2((-_cvt_f32_11.y) * 1.442695f);
        float d0_5 = 1.0f + e0_5;
        float d1_5 = 1.0f + e1_5;
        float r0_5 = approx_rcp(d0_5);
        float r1_5 = approx_rcp(d1_5);
        float _fma_32;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(-d0_5), "f"(r0_5), "f"(1.0f));
        float _fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(r0_5), "f"(_fma_32), "f"(r0_5));
        float _fma_34;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(-d1_5), "f"(r1_5), "f"(1.0f));
        float _fma_33;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(r1_5), "f"(_fma_34), "f"(r1_5));
        float2 _f32x2_mul_a_35 = mul_f32x2_noftz(_cvt_f32_11, make_float2(_fma_31, _fma_33));
        float2 _mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&_cvt_f32_12));
        packed[5] = __as_u32(__float22bfloat162_rn(make_float2(_mul_5.x, _mul_5.y)));
        float2 _cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 _cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float e0_6 = approx_exp2((-_cvt_f32_13.x) * 1.442695f);
        float e1_6 = approx_exp2((-_cvt_f32_13.y) * 1.442695f);
        float d0_6 = 1.0f + e0_6;
        float d1_6 = 1.0f + e1_6;
        float r0_6 = approx_rcp(d0_6);
        float r1_6 = approx_rcp(d1_6);
        float _fma_37;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(-d0_6), "f"(r0_6), "f"(1.0f));
        float _fma_36;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(r0_6), "f"(_fma_37), "f"(r0_6));
        float _fma_39;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(-d1_6), "f"(r1_6), "f"(1.0f));
        float _fma_38;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(r1_6), "f"(_fma_39), "f"(r1_6));
        float2 _f32x2_mul_a_40 = mul_f32x2_noftz(_cvt_f32_13, make_float2(_fma_36, _fma_38));
        float2 _mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_40), "l"(*(const unsigned long long*)&_cvt_f32_14));
        packed[6] = __as_u32(__float22bfloat162_rn(make_float2(_mul_6.x, _mul_6.y)));
        float2 _cvt_f32_15 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 _cvt_f32_16 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float e0_7 = approx_exp2((-_cvt_f32_15.x) * 1.442695f);
        float e1_7 = approx_exp2((-_cvt_f32_15.y) * 1.442695f);
        float d0_7 = 1.0f + e0_7;
        float d1_7 = 1.0f + e1_7;
        float r0_7 = approx_rcp(d0_7);
        float r1_7 = approx_rcp(d1_7);
        float _fma_42;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(-d0_7), "f"(r0_7), "f"(1.0f));
        float _fma_41;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(r0_7), "f"(_fma_42), "f"(r0_7));
        float _fma_44;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(-d1_7), "f"(r1_7), "f"(1.0f));
        float _fma_43;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(r1_7), "f"(_fma_44), "f"(r1_7));
        float2 _f32x2_mul_a_45 = mul_f32x2_noftz(_cvt_f32_15, make_float2(_fma_41, _fma_43));
        float2 _mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_45), "l"(*(const unsigned long long*)&_cvt_f32_16));
        packed[7] = __as_u32(__float22bfloat162_rn(make_float2(_mul_7.x, _mul_7.y)));
        float2 _cvt_f32_17 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float2 _cvt_f32_18 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
        float e0_8 = approx_exp2((-_cvt_f32_17.x) * 1.442695f);
        float e1_8 = approx_exp2((-_cvt_f32_17.y) * 1.442695f);
        float d0_8 = 1.0f + e0_8;
        float d1_8 = 1.0f + e1_8;
        float r0_8 = approx_rcp(d0_8);
        float r1_8 = approx_rcp(d1_8);
        float _fma_47;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(-d0_8), "f"(r0_8), "f"(1.0f));
        float _fma_46;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(r0_8), "f"(_fma_47), "f"(r0_8));
        float _fma_49;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(-d1_8), "f"(r1_8), "f"(1.0f));
        float _fma_48;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(r1_8), "f"(_fma_49), "f"(r1_8));
        float2 _f32x2_mul_a_50 = mul_f32x2_noftz(_cvt_f32_17, make_float2(_fma_46, _fma_48));
        float2 _mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_50), "l"(*(const unsigned long long*)&_cvt_f32_18));
        packed_hi[0] = __as_u32(__float22bfloat162_rn(make_float2(_mul_8.x, _mul_8.y)));
        float2 _cvt_f32_19 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float2 _cvt_f32_20 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[1]));
        float e0_9 = approx_exp2((-_cvt_f32_19.x) * 1.442695f);
        float e1_9 = approx_exp2((-_cvt_f32_19.y) * 1.442695f);
        float d0_9 = 1.0f + e0_9;
        float d1_9 = 1.0f + e1_9;
        float r0_9 = approx_rcp(d0_9);
        float r1_9 = approx_rcp(d1_9);
        float _fma_52;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(-d0_9), "f"(r0_9), "f"(1.0f));
        float _fma_51;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(r0_9), "f"(_fma_52), "f"(r0_9));
        float _fma_54;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(-d1_9), "f"(r1_9), "f"(1.0f));
        float _fma_53;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(r1_9), "f"(_fma_54), "f"(r1_9));
        float2 _f32x2_mul_a_55 = mul_f32x2_noftz(_cvt_f32_19, make_float2(_fma_51, _fma_53));
        float2 _mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_55), "l"(*(const unsigned long long*)&_cvt_f32_20));
        packed_hi[1] = __as_u32(__float22bfloat162_rn(make_float2(_mul_9.x, _mul_9.y)));
        float2 _cvt_f32_21 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float2 _cvt_f32_22 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[2]));
        float e0_10 = approx_exp2((-_cvt_f32_21.x) * 1.442695f);
        float e1_10 = approx_exp2((-_cvt_f32_21.y) * 1.442695f);
        float d0_10 = 1.0f + e0_10;
        float d1_10 = 1.0f + e1_10;
        float r0_10 = approx_rcp(d0_10);
        float r1_10 = approx_rcp(d1_10);
        float _fma_57;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(-d0_10), "f"(r0_10), "f"(1.0f));
        float _fma_56;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(r0_10), "f"(_fma_57), "f"(r0_10));
        float _fma_59;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(-d1_10), "f"(r1_10), "f"(1.0f));
        float _fma_58;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(r1_10), "f"(_fma_59), "f"(r1_10));
        float2 _f32x2_mul_a_60 = mul_f32x2_noftz(_cvt_f32_21, make_float2(_fma_56, _fma_58));
        float2 _mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&_cvt_f32_22));
        packed_hi[2] = __as_u32(__float22bfloat162_rn(make_float2(_mul_10.x, _mul_10.y)));
        float2 _cvt_f32_23 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float2 _cvt_f32_24 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[3]));
        float e0_11 = approx_exp2((-_cvt_f32_23.x) * 1.442695f);
        float e1_11 = approx_exp2((-_cvt_f32_23.y) * 1.442695f);
        float d0_11 = 1.0f + e0_11;
        float d1_11 = 1.0f + e1_11;
        float r0_11 = approx_rcp(d0_11);
        float r1_11 = approx_rcp(d1_11);
        float _fma_62;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(-d0_11), "f"(r0_11), "f"(1.0f));
        float _fma_61;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(r0_11), "f"(_fma_62), "f"(r0_11));
        float _fma_64;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(-d1_11), "f"(r1_11), "f"(1.0f));
        float _fma_63;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(r1_11), "f"(_fma_64), "f"(r1_11));
        float2 _f32x2_mul_a_65 = mul_f32x2_noftz(_cvt_f32_23, make_float2(_fma_61, _fma_63));
        float2 _mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_65), "l"(*(const unsigned long long*)&_cvt_f32_24));
        packed_hi[3] = __as_u32(__float22bfloat162_rn(make_float2(_mul_11.x, _mul_11.y)));
        float2 _cvt_f32_25 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[4]));
        float2 _cvt_f32_26 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[4]));
        float e0_12 = approx_exp2((-_cvt_f32_25.x) * 1.442695f);
        float e1_12 = approx_exp2((-_cvt_f32_25.y) * 1.442695f);
        float d0_12 = 1.0f + e0_12;
        float d1_12 = 1.0f + e1_12;
        float r0_12 = approx_rcp(d0_12);
        float r1_12 = approx_rcp(d1_12);
        float _fma_67;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(-d0_12), "f"(r0_12), "f"(1.0f));
        float _fma_66;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(r0_12), "f"(_fma_67), "f"(r0_12));
        float _fma_69;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(-d1_12), "f"(r1_12), "f"(1.0f));
        float _fma_68;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(r1_12), "f"(_fma_69), "f"(r1_12));
        float2 _f32x2_mul_a_70 = mul_f32x2_noftz(_cvt_f32_25, make_float2(_fma_66, _fma_68));
        float2 _mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&_cvt_f32_26));
        packed_hi[4] = __as_u32(__float22bfloat162_rn(make_float2(_mul_12.x, _mul_12.y)));
        float2 _cvt_f32_27 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[5]));
        float2 _cvt_f32_28 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[5]));
        float e0_13 = approx_exp2((-_cvt_f32_27.x) * 1.442695f);
        float e1_13 = approx_exp2((-_cvt_f32_27.y) * 1.442695f);
        float d0_13 = 1.0f + e0_13;
        float d1_13 = 1.0f + e1_13;
        float r0_13 = approx_rcp(d0_13);
        float r1_13 = approx_rcp(d1_13);
        float _fma_72;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(-d0_13), "f"(r0_13), "f"(1.0f));
        float _fma_71;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(r0_13), "f"(_fma_72), "f"(r0_13));
        float _fma_74;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(-d1_13), "f"(r1_13), "f"(1.0f));
        float _fma_73;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(r1_13), "f"(_fma_74), "f"(r1_13));
        float2 _f32x2_mul_a_75 = mul_f32x2_noftz(_cvt_f32_27, make_float2(_fma_71, _fma_73));
        float2 _mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_75), "l"(*(const unsigned long long*)&_cvt_f32_28));
        packed_hi[5] = __as_u32(__float22bfloat162_rn(make_float2(_mul_13.x, _mul_13.y)));
        float2 _cvt_f32_29 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[6]));
        float2 _cvt_f32_30 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[6]));
        float e0_14 = approx_exp2((-_cvt_f32_29.x) * 1.442695f);
        float e1_14 = approx_exp2((-_cvt_f32_29.y) * 1.442695f);
        float d0_14 = 1.0f + e0_14;
        float d1_14 = 1.0f + e1_14;
        float r0_14 = approx_rcp(d0_14);
        float r1_14 = approx_rcp(d1_14);
        float _fma_77;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_77) : "f"(-d0_14), "f"(r0_14), "f"(1.0f));
        float _fma_76;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(r0_14), "f"(_fma_77), "f"(r0_14));
        float _fma_79;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(-d1_14), "f"(r1_14), "f"(1.0f));
        float _fma_78;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_78) : "f"(r1_14), "f"(_fma_79), "f"(r1_14));
        float2 _f32x2_mul_a_80 = mul_f32x2_noftz(_cvt_f32_29, make_float2(_fma_76, _fma_78));
        float2 _mul_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_80), "l"(*(const unsigned long long*)&_cvt_f32_30));
        packed_hi[6] = __as_u32(__float22bfloat162_rn(make_float2(_mul_14.x, _mul_14.y)));
        float2 _cvt_f32_31 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[7]));
        float2 _cvt_f32_32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[7]));
        float e0_15 = approx_exp2((-_cvt_f32_31.x) * 1.442695f);
        float e1_15 = approx_exp2((-_cvt_f32_31.y) * 1.442695f);
        float d0_15 = 1.0f + e0_15;
        float d1_15 = 1.0f + e1_15;
        float r0_15 = approx_rcp(d0_15);
        float r1_15 = approx_rcp(d1_15);
        float _fma_82;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(-d0_15), "f"(r0_15), "f"(1.0f));
        float _fma_81;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(r0_15), "f"(_fma_82), "f"(r0_15));
        float _fma_84;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(-d1_15), "f"(r1_15), "f"(1.0f));
        float _fma_83;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(r1_15), "f"(_fma_84), "f"(r1_15));
        float2 _f32x2_mul_a_85 = mul_f32x2_noftz(_cvt_f32_31, make_float2(_fma_81, _fma_83));
        float2 _mul_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_85), "l"(*(const unsigned long long*)&_cvt_f32_32));
        packed_hi[7] = __as_u32(__float22bfloat162_rn(make_float2(_mul_15.x, _mul_15.y)));
    } else {
        int band = 0;
        float2 _cvt_f32_33 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 _cvt_f32_34 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float e0_16 = approx_exp2((-_cvt_f32_33.x) * 1.442695f);
        float e1_16 = approx_exp2((-_cvt_f32_33.y) * 1.442695f);
        float d0_16 = 1.0f + e0_16;
        float d1_16 = 1.0f + e1_16;
        float r0_16 = approx_rcp(d0_16);
        float r1_16 = approx_rcp(d1_16);
        float _fma_86;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(-d0_16), "f"(r0_16), "f"(1.0f));
        float _fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(r0_16), "f"(_fma_86), "f"(r0_16));
        float _fma_87;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(-d1_16), "f"(r1_16), "f"(1.0f));
        float _fma_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(r1_16), "f"(_fma_87), "f"(r1_16));
        float t0_16 = _fma_0;
        float t1_16 = _fma_1;
        int _warp_vote_88 = __any_sync(0xFFFFFFFF, fminf(t0_16, t1_16) == 0.0f);
        if (_warp_vote_88 != 0) {
            float d4_16a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_16a) : "f"(d0_16), "f"(0.25f));
            float r4_16a = approx_rcp(d4_16a);
            float _fma_90;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(-d4_16a), "f"(r4_16a), "f"(1.0f));
            float _fma_89;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(r4_16a), "f"(_fma_90), "f"(r4_16a));
            float big_16a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_16a) : "f"(mul_f32_rn_noftz(_fma_89, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_91;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_91) : "f"(big_16a), "f"(8.388608e+06f));
            float whole_16a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_16a) : "f"(_add_91), "f"(8.388608e+06f));
            float above_16a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_16a) : "f"(-d4_16a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_16a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_93;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_93) : "f"(big_16a), "f"(whole_16a));
            float _abs_92;
            asm("abs.f32 %0, %1;" : "=f"(_abs_92) : "f"(_sub_93));
            float _add_94;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_94) : "f"(big_16a), "f"(0.5f));
            float _sub_95;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_95) : "f"(big_16a), "f"(0.5f));
            float rounded_16a = ((_abs_92 == 0.5f) ? ((above_16a > 0.0f) ? _add_94 : _sub_95) : whole_16a);
            t0_16 = ((d0_16 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_16a, 7.888609e-31f), 1.7763568e-15f) : t0_16);
            float d4_16b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_16b) : "f"(d1_16), "f"(0.25f));
            float r4_16b = approx_rcp(d4_16b);
            float _fma_97;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(-d4_16b), "f"(r4_16b), "f"(1.0f));
            float _fma_96;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(r4_16b), "f"(_fma_97), "f"(r4_16b));
            float big_16b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_16b) : "f"(mul_f32_rn_noftz(_fma_96, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_98;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_98) : "f"(big_16b), "f"(8.388608e+06f));
            float whole_16b;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_16b) : "f"(_add_98), "f"(8.388608e+06f));
            float above_16b;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_16b) : "f"(-d4_16b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_16b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_100;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_100) : "f"(big_16b), "f"(whole_16b));
            float _abs_99;
            asm("abs.f32 %0, %1;" : "=f"(_abs_99) : "f"(_sub_100));
            float _add_101;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_101) : "f"(big_16b), "f"(0.5f));
            float _sub_102;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_102) : "f"(big_16b), "f"(0.5f));
            float rounded_16b = ((_abs_99 == 0.5f) ? ((above_16b > 0.0f) ? _add_101 : _sub_102) : whole_16b);
            t1_16 = ((d1_16 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_16b, 7.888609e-31f), 1.7763568e-15f) : t1_16);
        }
        float2 _f32x2_mul_a_103 = mul_f32x2_noftz(_cvt_f32_33, make_float2(max_noftz(t0_16, 0.0f), max_noftz(t1_16, 0.0f)));
        float2 _mul_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_103), "l"(*(const unsigned long long*)&_cvt_f32_34));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(_mul_16.x, _mul_16.y)));
        float2 _cvt_f32_35 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 _cvt_f32_36 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float e0_17 = approx_exp2((-_cvt_f32_35.x) * 1.442695f);
        float e1_17 = approx_exp2((-_cvt_f32_35.y) * 1.442695f);
        float d0_17 = 1.0f + e0_17;
        float d1_17 = 1.0f + e1_17;
        float r0_17 = approx_rcp(d0_17);
        float r1_17 = approx_rcp(d1_17);
        float _fma_104;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_104) : "f"(-d0_17), "f"(r0_17), "f"(1.0f));
        float _fma_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(r0_17), "f"(_fma_104), "f"(r0_17));
        float _fma_105;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_105) : "f"(-d1_17), "f"(r1_17), "f"(1.0f));
        float _fma_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(r1_17), "f"(_fma_105), "f"(r1_17));
        float t0_17 = _fma_2;
        float t1_17 = _fma_3;
        int _warp_vote_106 = __any_sync(0xFFFFFFFF, fminf(t0_17, t1_17) == 0.0f);
        if (_warp_vote_106 != 0) {
            float d4_17a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_17a) : "f"(d0_17), "f"(0.25f));
            float r4_17a = approx_rcp(d4_17a);
            float _fma_108;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_108) : "f"(-d4_17a), "f"(r4_17a), "f"(1.0f));
            float _fma_107;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(r4_17a), "f"(_fma_108), "f"(r4_17a));
            float big_17a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_17a) : "f"(mul_f32_rn_noftz(_fma_107, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_109;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_109) : "f"(big_17a), "f"(8.388608e+06f));
            float whole_17a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_17a) : "f"(_add_109), "f"(8.388608e+06f));
            float above_17a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_17a) : "f"(-d4_17a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_17a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_111;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_111) : "f"(big_17a), "f"(whole_17a));
            float _abs_110;
            asm("abs.f32 %0, %1;" : "=f"(_abs_110) : "f"(_sub_111));
            float _add_112;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_112) : "f"(big_17a), "f"(0.5f));
            float _sub_113;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_113) : "f"(big_17a), "f"(0.5f));
            float rounded_17a = ((_abs_110 == 0.5f) ? ((above_17a > 0.0f) ? _add_112 : _sub_113) : whole_17a);
            t0_17 = ((d0_17 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_17a, 7.888609e-31f), 1.7763568e-15f) : t0_17);
            float d4_17b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_17b) : "f"(d1_17), "f"(0.25f));
            float r4_17b = approx_rcp(d4_17b);
            float _fma_115;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_115) : "f"(-d4_17b), "f"(r4_17b), "f"(1.0f));
            float _fma_114;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_114) : "f"(r4_17b), "f"(_fma_115), "f"(r4_17b));
            float big_17b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_17b) : "f"(mul_f32_rn_noftz(_fma_114, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_116;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_116) : "f"(big_17b), "f"(8.388608e+06f));
            float whole_17b;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_17b) : "f"(_add_116), "f"(8.388608e+06f));
            float above_17b;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_17b) : "f"(-d4_17b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_17b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_118;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_118) : "f"(big_17b), "f"(whole_17b));
            float _abs_117;
            asm("abs.f32 %0, %1;" : "=f"(_abs_117) : "f"(_sub_118));
            float _add_119;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_119) : "f"(big_17b), "f"(0.5f));
            float _sub_120;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_120) : "f"(big_17b), "f"(0.5f));
            float rounded_17b = ((_abs_117 == 0.5f) ? ((above_17b > 0.0f) ? _add_119 : _sub_120) : whole_17b);
            t1_17 = ((d1_17 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_17b, 7.888609e-31f), 1.7763568e-15f) : t1_17);
        }
        float2 _f32x2_mul_a_121 = mul_f32x2_noftz(_cvt_f32_35, make_float2(max_noftz(t0_17, 0.0f), max_noftz(t1_17, 0.0f)));
        float2 _mul_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_121), "l"(*(const unsigned long long*)&_cvt_f32_36));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(_mul_17.x, _mul_17.y)));
        float2 _cvt_f32_37 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 _cvt_f32_38 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float e0_18 = approx_exp2((-_cvt_f32_37.x) * 1.442695f);
        float e1_18 = approx_exp2((-_cvt_f32_37.y) * 1.442695f);
        float d0_18 = 1.0f + e0_18;
        float d1_18 = 1.0f + e1_18;
        float r0_18 = approx_rcp(d0_18);
        float r1_18 = approx_rcp(d1_18);
        float _fma_122;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_122) : "f"(-d0_18), "f"(r0_18), "f"(1.0f));
        float _fma_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(r0_18), "f"(_fma_122), "f"(r0_18));
        float _fma_123;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_123) : "f"(-d1_18), "f"(r1_18), "f"(1.0f));
        float _fma_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(r1_18), "f"(_fma_123), "f"(r1_18));
        float t0_18 = _fma_4;
        float t1_18 = _fma_5;
        int _warp_vote_124 = __any_sync(0xFFFFFFFF, fminf(t0_18, t1_18) == 0.0f);
        if (_warp_vote_124 != 0) {
            float d4_18a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_18a) : "f"(d0_18), "f"(0.25f));
            float r4_18a = approx_rcp(d4_18a);
            float _fma_126;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_126) : "f"(-d4_18a), "f"(r4_18a), "f"(1.0f));
            float _fma_125;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(r4_18a), "f"(_fma_126), "f"(r4_18a));
            float big_18a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_18a) : "f"(mul_f32_rn_noftz(_fma_125, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_127;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_127) : "f"(big_18a), "f"(8.388608e+06f));
            float whole_18a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_18a) : "f"(_add_127), "f"(8.388608e+06f));
            float above_18a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_18a) : "f"(-d4_18a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_18a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_129;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_129) : "f"(big_18a), "f"(whole_18a));
            float _abs_128;
            asm("abs.f32 %0, %1;" : "=f"(_abs_128) : "f"(_sub_129));
            float _add_130;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_130) : "f"(big_18a), "f"(0.5f));
            float _sub_131;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_131) : "f"(big_18a), "f"(0.5f));
            float rounded_18a = ((_abs_128 == 0.5f) ? ((above_18a > 0.0f) ? _add_130 : _sub_131) : whole_18a);
            t0_18 = ((d0_18 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_18a, 7.888609e-31f), 1.7763568e-15f) : t0_18);
            float d4_18b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_18b) : "f"(d1_18), "f"(0.25f));
            float r4_18b = approx_rcp(d4_18b);
            float _fma_133;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_133) : "f"(-d4_18b), "f"(r4_18b), "f"(1.0f));
            float _fma_132;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_132) : "f"(r4_18b), "f"(_fma_133), "f"(r4_18b));
            float big_18b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_18b) : "f"(mul_f32_rn_noftz(_fma_132, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_134;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_134) : "f"(big_18b), "f"(8.388608e+06f));
            float whole_18b;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_18b) : "f"(_add_134), "f"(8.388608e+06f));
            float above_18b;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_18b) : "f"(-d4_18b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_18b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_136;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_136) : "f"(big_18b), "f"(whole_18b));
            float _abs_135;
            asm("abs.f32 %0, %1;" : "=f"(_abs_135) : "f"(_sub_136));
            float _add_137;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_137) : "f"(big_18b), "f"(0.5f));
            float _sub_138;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_138) : "f"(big_18b), "f"(0.5f));
            float rounded_18b = ((_abs_135 == 0.5f) ? ((above_18b > 0.0f) ? _add_137 : _sub_138) : whole_18b);
            t1_18 = ((d1_18 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_18b, 7.888609e-31f), 1.7763568e-15f) : t1_18);
        }
        float2 _f32x2_mul_a_139 = mul_f32x2_noftz(_cvt_f32_37, make_float2(max_noftz(t0_18, 0.0f), max_noftz(t1_18, 0.0f)));
        float2 _mul_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_139), "l"(*(const unsigned long long*)&_cvt_f32_38));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(_mul_18.x, _mul_18.y)));
        float2 _cvt_f32_39 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 _cvt_f32_40 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float e0_19 = approx_exp2((-_cvt_f32_39.x) * 1.442695f);
        float e1_19 = approx_exp2((-_cvt_f32_39.y) * 1.442695f);
        float d0_19 = 1.0f + e0_19;
        float d1_19 = 1.0f + e1_19;
        float r0_19 = approx_rcp(d0_19);
        float r1_19 = approx_rcp(d1_19);
        float _fma_140;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_140) : "f"(-d0_19), "f"(r0_19), "f"(1.0f));
        float _fma_6_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6_1) : "f"(r0_19), "f"(_fma_140), "f"(r0_19));
        float _fma_141;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_141) : "f"(-d1_19), "f"(r1_19), "f"(1.0f));
        float _fma_7_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7_1) : "f"(r1_19), "f"(_fma_141), "f"(r1_19));
        float t0_19 = _fma_6_1;
        float t1_19 = _fma_7_1;
        int _warp_vote_142 = __any_sync(0xFFFFFFFF, fminf(t0_19, t1_19) == 0.0f);
        if (_warp_vote_142 != 0) {
            float d4_19a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_19a) : "f"(d0_19), "f"(0.25f));
            float r4_19a = approx_rcp(d4_19a);
            float _fma_144;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_144) : "f"(-d4_19a), "f"(r4_19a), "f"(1.0f));
            float _fma_143;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_143) : "f"(r4_19a), "f"(_fma_144), "f"(r4_19a));
            float big_19a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_19a) : "f"(mul_f32_rn_noftz(_fma_143, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_145;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_145) : "f"(big_19a), "f"(8.388608e+06f));
            float whole_19a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_19a) : "f"(_add_145), "f"(8.388608e+06f));
            float above_19a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_19a) : "f"(-d4_19a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_19a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_147;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_147) : "f"(big_19a), "f"(whole_19a));
            float _abs_146;
            asm("abs.f32 %0, %1;" : "=f"(_abs_146) : "f"(_sub_147));
            float _add_148;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_148) : "f"(big_19a), "f"(0.5f));
            float _sub_149;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_149) : "f"(big_19a), "f"(0.5f));
            float rounded_19a = ((_abs_146 == 0.5f) ? ((above_19a > 0.0f) ? _add_148 : _sub_149) : whole_19a);
            t0_19 = ((d0_19 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_19a, 7.888609e-31f), 1.7763568e-15f) : t0_19);
            float d4_19b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_19b) : "f"(d1_19), "f"(0.25f));
            float r4_19b = approx_rcp(d4_19b);
            float _fma_151;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_151) : "f"(-d4_19b), "f"(r4_19b), "f"(1.0f));
            float _fma_150;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_150) : "f"(r4_19b), "f"(_fma_151), "f"(r4_19b));
            float big_19b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_19b) : "f"(mul_f32_rn_noftz(_fma_150, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_152;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_152) : "f"(big_19b), "f"(8.388608e+06f));
            float whole_19b;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_19b) : "f"(_add_152), "f"(8.388608e+06f));
            float above_19b;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_19b) : "f"(-d4_19b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_19b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_154;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_154) : "f"(big_19b), "f"(whole_19b));
            float _abs_153;
            asm("abs.f32 %0, %1;" : "=f"(_abs_153) : "f"(_sub_154));
            float _add_155;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_155) : "f"(big_19b), "f"(0.5f));
            float _sub_156;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_156) : "f"(big_19b), "f"(0.5f));
            float rounded_19b = ((_abs_153 == 0.5f) ? ((above_19b > 0.0f) ? _add_155 : _sub_156) : whole_19b);
            t1_19 = ((d1_19 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_19b, 7.888609e-31f), 1.7763568e-15f) : t1_19);
        }
        float2 _f32x2_mul_a_157 = mul_f32x2_noftz(_cvt_f32_39, make_float2(max_noftz(t0_19, 0.0f), max_noftz(t1_19, 0.0f)));
        float2 _mul_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_157), "l"(*(const unsigned long long*)&_cvt_f32_40));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(_mul_19.x, _mul_19.y)));
        float2 _cvt_f32_41 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 _cvt_f32_42 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float e0_20 = approx_exp2((-_cvt_f32_41.x) * 1.442695f);
        float e1_20 = approx_exp2((-_cvt_f32_41.y) * 1.442695f);
        float d0_20 = 1.0f + e0_20;
        float d1_20 = 1.0f + e1_20;
        float r0_20 = approx_rcp(d0_20);
        float r1_20 = approx_rcp(d1_20);
        float _fma_158;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_158) : "f"(-d0_20), "f"(r0_20), "f"(1.0f));
        float _fma_8_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8_1) : "f"(r0_20), "f"(_fma_158), "f"(r0_20));
        float _fma_159;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_159) : "f"(-d1_20), "f"(r1_20), "f"(1.0f));
        float _fma_9_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9_1) : "f"(r1_20), "f"(_fma_159), "f"(r1_20));
        float t0_20 = _fma_8_1;
        float t1_20 = _fma_9_1;
        int _warp_vote_160 = __any_sync(0xFFFFFFFF, fminf(t0_20, t1_20) == 0.0f);
        if (_warp_vote_160 != 0) {
            float d4_20a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_20a) : "f"(d0_20), "f"(0.25f));
            float r4_20a = approx_rcp(d4_20a);
            float _fma_162;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_162) : "f"(-d4_20a), "f"(r4_20a), "f"(1.0f));
            float _fma_161;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_161) : "f"(r4_20a), "f"(_fma_162), "f"(r4_20a));
            float big_20a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_20a) : "f"(mul_f32_rn_noftz(_fma_161, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_163;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_163) : "f"(big_20a), "f"(8.388608e+06f));
            float whole_20a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_20a) : "f"(_add_163), "f"(8.388608e+06f));
            float above_20a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_20a) : "f"(-d4_20a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_20a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_165;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_165) : "f"(big_20a), "f"(whole_20a));
            float _abs_164;
            asm("abs.f32 %0, %1;" : "=f"(_abs_164) : "f"(_sub_165));
            float _add_166;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_166) : "f"(big_20a), "f"(0.5f));
            float _sub_167;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_167) : "f"(big_20a), "f"(0.5f));
            float rounded_20a = ((_abs_164 == 0.5f) ? ((above_20a > 0.0f) ? _add_166 : _sub_167) : whole_20a);
            t0_20 = ((d0_20 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_20a, 7.888609e-31f), 1.7763568e-15f) : t0_20);
            float d4_20b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_20b) : "f"(d1_20), "f"(0.25f));
            float r4_20b = approx_rcp(d4_20b);
            float _fma_169;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_169) : "f"(-d4_20b), "f"(r4_20b), "f"(1.0f));
            float _fma_168;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_168) : "f"(r4_20b), "f"(_fma_169), "f"(r4_20b));
            float big_20b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_20b) : "f"(mul_f32_rn_noftz(_fma_168, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_170;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_170) : "f"(big_20b), "f"(8.388608e+06f));
            float whole_20b;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_20b) : "f"(_add_170), "f"(8.388608e+06f));
            float above_20b;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_20b) : "f"(-d4_20b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_20b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_172;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_172) : "f"(big_20b), "f"(whole_20b));
            float _abs_171;
            asm("abs.f32 %0, %1;" : "=f"(_abs_171) : "f"(_sub_172));
            float _add_173;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_173) : "f"(big_20b), "f"(0.5f));
            float _sub_174;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_174) : "f"(big_20b), "f"(0.5f));
            float rounded_20b = ((_abs_171 == 0.5f) ? ((above_20b > 0.0f) ? _add_173 : _sub_174) : whole_20b);
            t1_20 = ((d1_20 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_20b, 7.888609e-31f), 1.7763568e-15f) : t1_20);
        }
        float2 _f32x2_mul_a_175 = mul_f32x2_noftz(_cvt_f32_41, make_float2(max_noftz(t0_20, 0.0f), max_noftz(t1_20, 0.0f)));
        float2 _mul_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_175), "l"(*(const unsigned long long*)&_cvt_f32_42));
        packed[4] = __as_u32(__float22bfloat162_rn(make_float2(_mul_20.x, _mul_20.y)));
        float2 _cvt_f32_43 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 _cvt_f32_44 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float e0_21 = approx_exp2((-_cvt_f32_43.x) * 1.442695f);
        float e1_21 = approx_exp2((-_cvt_f32_43.y) * 1.442695f);
        float d0_21 = 1.0f + e0_21;
        float d1_21 = 1.0f + e1_21;
        float r0_21 = approx_rcp(d0_21);
        float r1_21 = approx_rcp(d1_21);
        float _fma_176;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_176) : "f"(-d0_21), "f"(r0_21), "f"(1.0f));
        float _fma_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(r0_21), "f"(_fma_176), "f"(r0_21));
        float _fma_177;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_177) : "f"(-d1_21), "f"(r1_21), "f"(1.0f));
        float _fma_11_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11_1) : "f"(r1_21), "f"(_fma_177), "f"(r1_21));
        float t0_21 = _fma_10;
        float t1_21 = _fma_11_1;
        int _warp_vote_178 = __any_sync(0xFFFFFFFF, fminf(t0_21, t1_21) == 0.0f);
        if (_warp_vote_178 != 0) {
            float d4_21a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_21a) : "f"(d0_21), "f"(0.25f));
            float r4_21a = approx_rcp(d4_21a);
            float _fma_180;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_180) : "f"(-d4_21a), "f"(r4_21a), "f"(1.0f));
            float _fma_179;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_179) : "f"(r4_21a), "f"(_fma_180), "f"(r4_21a));
            float big_21a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_21a) : "f"(mul_f32_rn_noftz(_fma_179, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_181;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_181) : "f"(big_21a), "f"(8.388608e+06f));
            float whole_21a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_21a) : "f"(_add_181), "f"(8.388608e+06f));
            float above_21a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_21a) : "f"(-d4_21a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_21a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_183;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_183) : "f"(big_21a), "f"(whole_21a));
            float _abs_182;
            asm("abs.f32 %0, %1;" : "=f"(_abs_182) : "f"(_sub_183));
            float _add_184;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_184) : "f"(big_21a), "f"(0.5f));
            float _sub_185;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_185) : "f"(big_21a), "f"(0.5f));
            float rounded_21a = ((_abs_182 == 0.5f) ? ((above_21a > 0.0f) ? _add_184 : _sub_185) : whole_21a);
            t0_21 = ((d0_21 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_21a, 7.888609e-31f), 1.7763568e-15f) : t0_21);
            float d4_21b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_21b) : "f"(d1_21), "f"(0.25f));
            float r4_21b = approx_rcp(d4_21b);
            float _fma_187;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_187) : "f"(-d4_21b), "f"(r4_21b), "f"(1.0f));
            float _fma_186;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_186) : "f"(r4_21b), "f"(_fma_187), "f"(r4_21b));
            float big_21b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_21b) : "f"(mul_f32_rn_noftz(_fma_186, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_188;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_188) : "f"(big_21b), "f"(8.388608e+06f));
            float whole_21b;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_21b) : "f"(_add_188), "f"(8.388608e+06f));
            float above_21b;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_21b) : "f"(-d4_21b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_21b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_190;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_190) : "f"(big_21b), "f"(whole_21b));
            float _abs_189;
            asm("abs.f32 %0, %1;" : "=f"(_abs_189) : "f"(_sub_190));
            float _add_191;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_191) : "f"(big_21b), "f"(0.5f));
            float _sub_192;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_192) : "f"(big_21b), "f"(0.5f));
            float rounded_21b = ((_abs_189 == 0.5f) ? ((above_21b > 0.0f) ? _add_191 : _sub_192) : whole_21b);
            t1_21 = ((d1_21 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_21b, 7.888609e-31f), 1.7763568e-15f) : t1_21);
        }
        float2 _f32x2_mul_a_193 = mul_f32x2_noftz(_cvt_f32_43, make_float2(max_noftz(t0_21, 0.0f), max_noftz(t1_21, 0.0f)));
        float2 _mul_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_193), "l"(*(const unsigned long long*)&_cvt_f32_44));
        packed[5] = __as_u32(__float22bfloat162_rn(make_float2(_mul_21.x, _mul_21.y)));
        float2 _cvt_f32_45 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 _cvt_f32_46 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float e0_22 = approx_exp2((-_cvt_f32_45.x) * 1.442695f);
        float e1_22 = approx_exp2((-_cvt_f32_45.y) * 1.442695f);
        float d0_22 = 1.0f + e0_22;
        float d1_22 = 1.0f + e1_22;
        float r0_22 = approx_rcp(d0_22);
        float r1_22 = approx_rcp(d1_22);
        float _fma_194;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_194) : "f"(-d0_22), "f"(r0_22), "f"(1.0f));
        float _fma_12_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12_1) : "f"(r0_22), "f"(_fma_194), "f"(r0_22));
        float _fma_195;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_195) : "f"(-d1_22), "f"(r1_22), "f"(1.0f));
        float _fma_13_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13_1) : "f"(r1_22), "f"(_fma_195), "f"(r1_22));
        float t0_22 = _fma_12_1;
        float t1_22 = _fma_13_1;
        int _warp_vote_196 = __any_sync(0xFFFFFFFF, fminf(t0_22, t1_22) == 0.0f);
        if (_warp_vote_196 != 0) {
            float d4_22a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_22a) : "f"(d0_22), "f"(0.25f));
            float r4_22a = approx_rcp(d4_22a);
            float _fma_198;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_198) : "f"(-d4_22a), "f"(r4_22a), "f"(1.0f));
            float _fma_197;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_197) : "f"(r4_22a), "f"(_fma_198), "f"(r4_22a));
            float big_22a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_22a) : "f"(mul_f32_rn_noftz(_fma_197, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_199;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_199) : "f"(big_22a), "f"(8.388608e+06f));
            float whole_22a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_22a) : "f"(_add_199), "f"(8.388608e+06f));
            float above_22a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_22a) : "f"(-d4_22a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_22a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_201;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_201) : "f"(big_22a), "f"(whole_22a));
            float _abs_200;
            asm("abs.f32 %0, %1;" : "=f"(_abs_200) : "f"(_sub_201));
            float _add_202;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_202) : "f"(big_22a), "f"(0.5f));
            float _sub_203;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_203) : "f"(big_22a), "f"(0.5f));
            float rounded_22a = ((_abs_200 == 0.5f) ? ((above_22a > 0.0f) ? _add_202 : _sub_203) : whole_22a);
            t0_22 = ((d0_22 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_22a, 7.888609e-31f), 1.7763568e-15f) : t0_22);
            float d4_22b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_22b) : "f"(d1_22), "f"(0.25f));
            float r4_22b = approx_rcp(d4_22b);
            float _fma_205;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_205) : "f"(-d4_22b), "f"(r4_22b), "f"(1.0f));
            float _fma_204;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_204) : "f"(r4_22b), "f"(_fma_205), "f"(r4_22b));
            float big_22b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_22b) : "f"(mul_f32_rn_noftz(_fma_204, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_206;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_206) : "f"(big_22b), "f"(8.388608e+06f));
            float whole_22b;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_22b) : "f"(_add_206), "f"(8.388608e+06f));
            float above_22b;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_22b) : "f"(-d4_22b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_22b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_208;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_208) : "f"(big_22b), "f"(whole_22b));
            float _abs_207;
            asm("abs.f32 %0, %1;" : "=f"(_abs_207) : "f"(_sub_208));
            float _add_209;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_209) : "f"(big_22b), "f"(0.5f));
            float _sub_210;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_210) : "f"(big_22b), "f"(0.5f));
            float rounded_22b = ((_abs_207 == 0.5f) ? ((above_22b > 0.0f) ? _add_209 : _sub_210) : whole_22b);
            t1_22 = ((d1_22 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_22b, 7.888609e-31f), 1.7763568e-15f) : t1_22);
        }
        float2 _f32x2_mul_a_211 = mul_f32x2_noftz(_cvt_f32_45, make_float2(max_noftz(t0_22, 0.0f), max_noftz(t1_22, 0.0f)));
        float2 _mul_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_211), "l"(*(const unsigned long long*)&_cvt_f32_46));
        packed[6] = __as_u32(__float22bfloat162_rn(make_float2(_mul_22.x, _mul_22.y)));
        float2 _cvt_f32_47 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 _cvt_f32_48 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float e0_23 = approx_exp2((-_cvt_f32_47.x) * 1.442695f);
        float e1_23 = approx_exp2((-_cvt_f32_47.y) * 1.442695f);
        float d0_23 = 1.0f + e0_23;
        float d1_23 = 1.0f + e1_23;
        float r0_23 = approx_rcp(d0_23);
        float r1_23 = approx_rcp(d1_23);
        float _fma_212;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_212) : "f"(-d0_23), "f"(r0_23), "f"(1.0f));
        float _fma_14_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14_1) : "f"(r0_23), "f"(_fma_212), "f"(r0_23));
        float _fma_213;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_213) : "f"(-d1_23), "f"(r1_23), "f"(1.0f));
        float _fma_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(r1_23), "f"(_fma_213), "f"(r1_23));
        float t0_23 = _fma_14_1;
        float t1_23 = _fma_15;
        int _warp_vote_214 = __any_sync(0xFFFFFFFF, fminf(t0_23, t1_23) == 0.0f);
        if (_warp_vote_214 != 0) {
            float d4_23a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_23a) : "f"(d0_23), "f"(0.25f));
            float r4_23a = approx_rcp(d4_23a);
            float _fma_216;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_216) : "f"(-d4_23a), "f"(r4_23a), "f"(1.0f));
            float _fma_215;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_215) : "f"(r4_23a), "f"(_fma_216), "f"(r4_23a));
            float big_23a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_23a) : "f"(mul_f32_rn_noftz(_fma_215, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_217;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_217) : "f"(big_23a), "f"(8.388608e+06f));
            float whole_23a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_23a) : "f"(_add_217), "f"(8.388608e+06f));
            float above_23a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_23a) : "f"(-d4_23a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_23a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_219;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_219) : "f"(big_23a), "f"(whole_23a));
            float _abs_218;
            asm("abs.f32 %0, %1;" : "=f"(_abs_218) : "f"(_sub_219));
            float _add_220;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_220) : "f"(big_23a), "f"(0.5f));
            float _sub_221;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_221) : "f"(big_23a), "f"(0.5f));
            float rounded_23a = ((_abs_218 == 0.5f) ? ((above_23a > 0.0f) ? _add_220 : _sub_221) : whole_23a);
            t0_23 = ((d0_23 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_23a, 7.888609e-31f), 1.7763568e-15f) : t0_23);
            float d4_23b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_23b) : "f"(d1_23), "f"(0.25f));
            float r4_23b = approx_rcp(d4_23b);
            float _fma_223;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_223) : "f"(-d4_23b), "f"(r4_23b), "f"(1.0f));
            float _fma_222;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_222) : "f"(r4_23b), "f"(_fma_223), "f"(r4_23b));
            float big_23b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_23b) : "f"(mul_f32_rn_noftz(_fma_222, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_224;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_224) : "f"(big_23b), "f"(8.388608e+06f));
            float whole_23b;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_23b) : "f"(_add_224), "f"(8.388608e+06f));
            float above_23b;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_23b) : "f"(-d4_23b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_23b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_226;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_226) : "f"(big_23b), "f"(whole_23b));
            float _abs_225;
            asm("abs.f32 %0, %1;" : "=f"(_abs_225) : "f"(_sub_226));
            float _add_227;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_227) : "f"(big_23b), "f"(0.5f));
            float _sub_228;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_228) : "f"(big_23b), "f"(0.5f));
            float rounded_23b = ((_abs_225 == 0.5f) ? ((above_23b > 0.0f) ? _add_227 : _sub_228) : whole_23b);
            t1_23 = ((d1_23 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_23b, 7.888609e-31f), 1.7763568e-15f) : t1_23);
        }
        float2 _f32x2_mul_a_229 = mul_f32x2_noftz(_cvt_f32_47, make_float2(max_noftz(t0_23, 0.0f), max_noftz(t1_23, 0.0f)));
        float2 _mul_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_229), "l"(*(const unsigned long long*)&_cvt_f32_48));
        packed[7] = __as_u32(__float22bfloat162_rn(make_float2(_mul_23.x, _mul_23.y)));
        int band_hi = 0;
        float2 _cvt_f32_49 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float2 _cvt_f32_50 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
        float e0_24 = approx_exp2((-_cvt_f32_49.x) * 1.442695f);
        float e1_24 = approx_exp2((-_cvt_f32_49.y) * 1.442695f);
        float d0_24 = 1.0f + e0_24;
        float d1_24 = 1.0f + e1_24;
        float r0_24 = approx_rcp(d0_24);
        float r1_24 = approx_rcp(d1_24);
        float _fma_230;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_230) : "f"(-d0_24), "f"(r0_24), "f"(1.0f));
        float _fma_16_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16_1) : "f"(r0_24), "f"(_fma_230), "f"(r0_24));
        float _fma_231;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_231) : "f"(-d1_24), "f"(r1_24), "f"(1.0f));
        float _fma_17_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17_1) : "f"(r1_24), "f"(_fma_231), "f"(r1_24));
        float t0_24 = _fma_16_1;
        float t1_24 = _fma_17_1;
        int _warp_vote_232 = __any_sync(0xFFFFFFFF, fminf(t0_24, t1_24) == 0.0f);
        if (_warp_vote_232 != 0) {
            float d4_24a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_24a) : "f"(d0_24), "f"(0.25f));
            float r4_24a = approx_rcp(d4_24a);
            float _fma_234;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_234) : "f"(-d4_24a), "f"(r4_24a), "f"(1.0f));
            float _fma_233;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_233) : "f"(r4_24a), "f"(_fma_234), "f"(r4_24a));
            float big_24a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_24a) : "f"(mul_f32_rn_noftz(_fma_233, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_235;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_235) : "f"(big_24a), "f"(8.388608e+06f));
            float whole_24a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_24a) : "f"(_add_235), "f"(8.388608e+06f));
            float above_24a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_24a) : "f"(-d4_24a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_24a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_237;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_237) : "f"(big_24a), "f"(whole_24a));
            float _abs_236;
            asm("abs.f32 %0, %1;" : "=f"(_abs_236) : "f"(_sub_237));
            float _add_238;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_238) : "f"(big_24a), "f"(0.5f));
            float _sub_239;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_239) : "f"(big_24a), "f"(0.5f));
            float rounded_24a = ((_abs_236 == 0.5f) ? ((above_24a > 0.0f) ? _add_238 : _sub_239) : whole_24a);
            t0_24 = ((d0_24 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_24a, 7.888609e-31f), 1.7763568e-15f) : t0_24);
            float d4_24b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_24b) : "f"(d1_24), "f"(0.25f));
            float r4_24b = approx_rcp(d4_24b);
            float _fma_241;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_241) : "f"(-d4_24b), "f"(r4_24b), "f"(1.0f));
            float _fma_240;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_240) : "f"(r4_24b), "f"(_fma_241), "f"(r4_24b));
            float big_24b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_24b) : "f"(mul_f32_rn_noftz(_fma_240, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_242;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_242) : "f"(big_24b), "f"(8.388608e+06f));
            float whole_24b;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_24b) : "f"(_add_242), "f"(8.388608e+06f));
            float above_24b;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_24b) : "f"(-d4_24b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_24b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_244;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_244) : "f"(big_24b), "f"(whole_24b));
            float _abs_243;
            asm("abs.f32 %0, %1;" : "=f"(_abs_243) : "f"(_sub_244));
            float _add_245;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_245) : "f"(big_24b), "f"(0.5f));
            float _sub_246;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_246) : "f"(big_24b), "f"(0.5f));
            float rounded_24b = ((_abs_243 == 0.5f) ? ((above_24b > 0.0f) ? _add_245 : _sub_246) : whole_24b);
            t1_24 = ((d1_24 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_24b, 7.888609e-31f), 1.7763568e-15f) : t1_24);
        }
        float2 _f32x2_mul_a_247 = mul_f32x2_noftz(_cvt_f32_49, make_float2(max_noftz(t0_24, 0.0f), max_noftz(t1_24, 0.0f)));
        float2 _mul_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_247), "l"(*(const unsigned long long*)&_cvt_f32_50));
        packed_hi[0] = __as_u32(__float22bfloat162_rn(make_float2(_mul_24.x, _mul_24.y)));
        float2 _cvt_f32_51 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float2 _cvt_f32_52 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[1]));
        float e0_25 = approx_exp2((-_cvt_f32_51.x) * 1.442695f);
        float e1_25 = approx_exp2((-_cvt_f32_51.y) * 1.442695f);
        float d0_25 = 1.0f + e0_25;
        float d1_25 = 1.0f + e1_25;
        float r0_25 = approx_rcp(d0_25);
        float r1_25 = approx_rcp(d1_25);
        float _fma_248;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_248) : "f"(-d0_25), "f"(r0_25), "f"(1.0f));
        float _fma_18_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18_1) : "f"(r0_25), "f"(_fma_248), "f"(r0_25));
        float _fma_249;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_249) : "f"(-d1_25), "f"(r1_25), "f"(1.0f));
        float _fma_19_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19_1) : "f"(r1_25), "f"(_fma_249), "f"(r1_25));
        float t0_25 = _fma_18_1;
        float t1_25 = _fma_19_1;
        int _warp_vote_250 = __any_sync(0xFFFFFFFF, fminf(t0_25, t1_25) == 0.0f);
        if (_warp_vote_250 != 0) {
            float d4_25a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_25a) : "f"(d0_25), "f"(0.25f));
            float r4_25a = approx_rcp(d4_25a);
            float _fma_252;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_252) : "f"(-d4_25a), "f"(r4_25a), "f"(1.0f));
            float _fma_251;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_251) : "f"(r4_25a), "f"(_fma_252), "f"(r4_25a));
            float big_25a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_25a) : "f"(mul_f32_rn_noftz(_fma_251, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_253;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_253) : "f"(big_25a), "f"(8.388608e+06f));
            float whole_25a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_25a) : "f"(_add_253), "f"(8.388608e+06f));
            float above_25a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_25a) : "f"(-d4_25a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_25a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_255;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_255) : "f"(big_25a), "f"(whole_25a));
            float _abs_254;
            asm("abs.f32 %0, %1;" : "=f"(_abs_254) : "f"(_sub_255));
            float _add_256;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_256) : "f"(big_25a), "f"(0.5f));
            float _sub_257;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_257) : "f"(big_25a), "f"(0.5f));
            float rounded_25a = ((_abs_254 == 0.5f) ? ((above_25a > 0.0f) ? _add_256 : _sub_257) : whole_25a);
            t0_25 = ((d0_25 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_25a, 7.888609e-31f), 1.7763568e-15f) : t0_25);
            float d4_25b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_25b) : "f"(d1_25), "f"(0.25f));
            float r4_25b = approx_rcp(d4_25b);
            float _fma_259;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_259) : "f"(-d4_25b), "f"(r4_25b), "f"(1.0f));
            float _fma_258;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_258) : "f"(r4_25b), "f"(_fma_259), "f"(r4_25b));
            float big_25b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_25b) : "f"(mul_f32_rn_noftz(_fma_258, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_260;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_260) : "f"(big_25b), "f"(8.388608e+06f));
            float whole_25b;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_25b) : "f"(_add_260), "f"(8.388608e+06f));
            float above_25b;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_25b) : "f"(-d4_25b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_25b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_262;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_262) : "f"(big_25b), "f"(whole_25b));
            float _abs_261;
            asm("abs.f32 %0, %1;" : "=f"(_abs_261) : "f"(_sub_262));
            float _add_263;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_263) : "f"(big_25b), "f"(0.5f));
            float _sub_264;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_264) : "f"(big_25b), "f"(0.5f));
            float rounded_25b = ((_abs_261 == 0.5f) ? ((above_25b > 0.0f) ? _add_263 : _sub_264) : whole_25b);
            t1_25 = ((d1_25 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_25b, 7.888609e-31f), 1.7763568e-15f) : t1_25);
        }
        float2 _f32x2_mul_a_265 = mul_f32x2_noftz(_cvt_f32_51, make_float2(max_noftz(t0_25, 0.0f), max_noftz(t1_25, 0.0f)));
        float2 _mul_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_265), "l"(*(const unsigned long long*)&_cvt_f32_52));
        packed_hi[1] = __as_u32(__float22bfloat162_rn(make_float2(_mul_25.x, _mul_25.y)));
        float2 _cvt_f32_53 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float2 _cvt_f32_54 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[2]));
        float e0_26 = approx_exp2((-_cvt_f32_53.x) * 1.442695f);
        float e1_26 = approx_exp2((-_cvt_f32_53.y) * 1.442695f);
        float d0_26 = 1.0f + e0_26;
        float d1_26 = 1.0f + e1_26;
        float r0_26 = approx_rcp(d0_26);
        float r1_26 = approx_rcp(d1_26);
        float _fma_266;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_266) : "f"(-d0_26), "f"(r0_26), "f"(1.0f));
        float _fma_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(r0_26), "f"(_fma_266), "f"(r0_26));
        float _fma_267;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_267) : "f"(-d1_26), "f"(r1_26), "f"(1.0f));
        float _fma_21_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21_1) : "f"(r1_26), "f"(_fma_267), "f"(r1_26));
        float t0_26 = _fma_20;
        float t1_26 = _fma_21_1;
        int _warp_vote_268 = __any_sync(0xFFFFFFFF, fminf(t0_26, t1_26) == 0.0f);
        if (_warp_vote_268 != 0) {
            float d4_26a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_26a) : "f"(d0_26), "f"(0.25f));
            float r4_26a = approx_rcp(d4_26a);
            float _fma_270;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_270) : "f"(-d4_26a), "f"(r4_26a), "f"(1.0f));
            float _fma_269;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_269) : "f"(r4_26a), "f"(_fma_270), "f"(r4_26a));
            float big_26a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_26a) : "f"(mul_f32_rn_noftz(_fma_269, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_271;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_271) : "f"(big_26a), "f"(8.388608e+06f));
            float whole_26a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_26a) : "f"(_add_271), "f"(8.388608e+06f));
            float above_26a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_26a) : "f"(-d4_26a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_26a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_273;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_273) : "f"(big_26a), "f"(whole_26a));
            float _abs_272;
            asm("abs.f32 %0, %1;" : "=f"(_abs_272) : "f"(_sub_273));
            float _add_274;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_274) : "f"(big_26a), "f"(0.5f));
            float _sub_275;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_275) : "f"(big_26a), "f"(0.5f));
            float rounded_26a = ((_abs_272 == 0.5f) ? ((above_26a > 0.0f) ? _add_274 : _sub_275) : whole_26a);
            t0_26 = ((d0_26 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_26a, 7.888609e-31f), 1.7763568e-15f) : t0_26);
            float d4_26b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_26b) : "f"(d1_26), "f"(0.25f));
            float r4_26b = approx_rcp(d4_26b);
            float _fma_277;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_277) : "f"(-d4_26b), "f"(r4_26b), "f"(1.0f));
            float _fma_276;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_276) : "f"(r4_26b), "f"(_fma_277), "f"(r4_26b));
            float big_26b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_26b) : "f"(mul_f32_rn_noftz(_fma_276, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_278;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_278) : "f"(big_26b), "f"(8.388608e+06f));
            float whole_26b;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_26b) : "f"(_add_278), "f"(8.388608e+06f));
            float above_26b;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_26b) : "f"(-d4_26b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_26b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_280;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_280) : "f"(big_26b), "f"(whole_26b));
            float _abs_279;
            asm("abs.f32 %0, %1;" : "=f"(_abs_279) : "f"(_sub_280));
            float _add_281;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_281) : "f"(big_26b), "f"(0.5f));
            float _sub_282;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_282) : "f"(big_26b), "f"(0.5f));
            float rounded_26b = ((_abs_279 == 0.5f) ? ((above_26b > 0.0f) ? _add_281 : _sub_282) : whole_26b);
            t1_26 = ((d1_26 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_26b, 7.888609e-31f), 1.7763568e-15f) : t1_26);
        }
        float2 _f32x2_mul_a_283 = mul_f32x2_noftz(_cvt_f32_53, make_float2(max_noftz(t0_26, 0.0f), max_noftz(t1_26, 0.0f)));
        float2 _mul_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_283), "l"(*(const unsigned long long*)&_cvt_f32_54));
        packed_hi[2] = __as_u32(__float22bfloat162_rn(make_float2(_mul_26.x, _mul_26.y)));
        float2 _cvt_f32_55 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float2 _cvt_f32_56 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[3]));
        float e0_27 = approx_exp2((-_cvt_f32_55.x) * 1.442695f);
        float e1_27 = approx_exp2((-_cvt_f32_55.y) * 1.442695f);
        float d0_27 = 1.0f + e0_27;
        float d1_27 = 1.0f + e1_27;
        float r0_27 = approx_rcp(d0_27);
        float r1_27 = approx_rcp(d1_27);
        float _fma_284;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_284) : "f"(-d0_27), "f"(r0_27), "f"(1.0f));
        float _fma_22_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22_1) : "f"(r0_27), "f"(_fma_284), "f"(r0_27));
        float _fma_285;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_285) : "f"(-d1_27), "f"(r1_27), "f"(1.0f));
        float _fma_23_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23_1) : "f"(r1_27), "f"(_fma_285), "f"(r1_27));
        float t0_27 = _fma_22_1;
        float t1_27 = _fma_23_1;
        int _warp_vote_286 = __any_sync(0xFFFFFFFF, fminf(t0_27, t1_27) == 0.0f);
        if (_warp_vote_286 != 0) {
            float d4_27a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_27a) : "f"(d0_27), "f"(0.25f));
            float r4_27a = approx_rcp(d4_27a);
            float _fma_288;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_288) : "f"(-d4_27a), "f"(r4_27a), "f"(1.0f));
            float _fma_287;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_287) : "f"(r4_27a), "f"(_fma_288), "f"(r4_27a));
            float big_27a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_27a) : "f"(mul_f32_rn_noftz(_fma_287, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_289;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_289) : "f"(big_27a), "f"(8.388608e+06f));
            float whole_27a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_27a) : "f"(_add_289), "f"(8.388608e+06f));
            float above_27a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_27a) : "f"(-d4_27a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_27a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_291;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_291) : "f"(big_27a), "f"(whole_27a));
            float _abs_290;
            asm("abs.f32 %0, %1;" : "=f"(_abs_290) : "f"(_sub_291));
            float _add_292;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_292) : "f"(big_27a), "f"(0.5f));
            float _sub_293;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_293) : "f"(big_27a), "f"(0.5f));
            float rounded_27a = ((_abs_290 == 0.5f) ? ((above_27a > 0.0f) ? _add_292 : _sub_293) : whole_27a);
            t0_27 = ((d0_27 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_27a, 7.888609e-31f), 1.7763568e-15f) : t0_27);
            float d4_27b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_27b) : "f"(d1_27), "f"(0.25f));
            float r4_27b = approx_rcp(d4_27b);
            float _fma_295;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_295) : "f"(-d4_27b), "f"(r4_27b), "f"(1.0f));
            float _fma_294;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_294) : "f"(r4_27b), "f"(_fma_295), "f"(r4_27b));
            float big_27b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_27b) : "f"(mul_f32_rn_noftz(_fma_294, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_296;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_296) : "f"(big_27b), "f"(8.388608e+06f));
            float whole_27b;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_27b) : "f"(_add_296), "f"(8.388608e+06f));
            float above_27b;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_27b) : "f"(-d4_27b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_27b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_298;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_298) : "f"(big_27b), "f"(whole_27b));
            float _abs_297;
            asm("abs.f32 %0, %1;" : "=f"(_abs_297) : "f"(_sub_298));
            float _add_299;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_299) : "f"(big_27b), "f"(0.5f));
            float _sub_300;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_300) : "f"(big_27b), "f"(0.5f));
            float rounded_27b = ((_abs_297 == 0.5f) ? ((above_27b > 0.0f) ? _add_299 : _sub_300) : whole_27b);
            t1_27 = ((d1_27 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_27b, 7.888609e-31f), 1.7763568e-15f) : t1_27);
        }
        float2 _f32x2_mul_a_301 = mul_f32x2_noftz(_cvt_f32_55, make_float2(max_noftz(t0_27, 0.0f), max_noftz(t1_27, 0.0f)));
        float2 _mul_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_301), "l"(*(const unsigned long long*)&_cvt_f32_56));
        packed_hi[3] = __as_u32(__float22bfloat162_rn(make_float2(_mul_27.x, _mul_27.y)));
        float2 _cvt_f32_57 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[4]));
        float2 _cvt_f32_58 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[4]));
        float e0_28 = approx_exp2((-_cvt_f32_57.x) * 1.442695f);
        float e1_28 = approx_exp2((-_cvt_f32_57.y) * 1.442695f);
        float d0_28 = 1.0f + e0_28;
        float d1_28 = 1.0f + e1_28;
        float r0_28 = approx_rcp(d0_28);
        float r1_28 = approx_rcp(d1_28);
        float _fma_302;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_302) : "f"(-d0_28), "f"(r0_28), "f"(1.0f));
        float _fma_24_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24_1) : "f"(r0_28), "f"(_fma_302), "f"(r0_28));
        float _fma_303;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_303) : "f"(-d1_28), "f"(r1_28), "f"(1.0f));
        float _fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(r1_28), "f"(_fma_303), "f"(r1_28));
        float t0_28 = _fma_24_1;
        float t1_28 = _fma_25;
        int _warp_vote_304 = __any_sync(0xFFFFFFFF, fminf(t0_28, t1_28) == 0.0f);
        if (_warp_vote_304 != 0) {
            float d4_28a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_28a) : "f"(d0_28), "f"(0.25f));
            float r4_28a = approx_rcp(d4_28a);
            float _fma_306;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_306) : "f"(-d4_28a), "f"(r4_28a), "f"(1.0f));
            float _fma_305;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_305) : "f"(r4_28a), "f"(_fma_306), "f"(r4_28a));
            float big_28a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_28a) : "f"(mul_f32_rn_noftz(_fma_305, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_307;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_307) : "f"(big_28a), "f"(8.388608e+06f));
            float whole_28a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_28a) : "f"(_add_307), "f"(8.388608e+06f));
            float above_28a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_28a) : "f"(-d4_28a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_28a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_309;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_309) : "f"(big_28a), "f"(whole_28a));
            float _abs_308;
            asm("abs.f32 %0, %1;" : "=f"(_abs_308) : "f"(_sub_309));
            float _add_310;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_310) : "f"(big_28a), "f"(0.5f));
            float _sub_311;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_311) : "f"(big_28a), "f"(0.5f));
            float rounded_28a = ((_abs_308 == 0.5f) ? ((above_28a > 0.0f) ? _add_310 : _sub_311) : whole_28a);
            t0_28 = ((d0_28 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_28a, 7.888609e-31f), 1.7763568e-15f) : t0_28);
            float d4_28b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_28b) : "f"(d1_28), "f"(0.25f));
            float r4_28b = approx_rcp(d4_28b);
            float _fma_313;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_313) : "f"(-d4_28b), "f"(r4_28b), "f"(1.0f));
            float _fma_312;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_312) : "f"(r4_28b), "f"(_fma_313), "f"(r4_28b));
            float big_28b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_28b) : "f"(mul_f32_rn_noftz(_fma_312, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_314;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_314) : "f"(big_28b), "f"(8.388608e+06f));
            float whole_28b;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_28b) : "f"(_add_314), "f"(8.388608e+06f));
            float above_28b;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_28b) : "f"(-d4_28b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_28b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_316;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_316) : "f"(big_28b), "f"(whole_28b));
            float _abs_315;
            asm("abs.f32 %0, %1;" : "=f"(_abs_315) : "f"(_sub_316));
            float _add_317;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_317) : "f"(big_28b), "f"(0.5f));
            float _sub_318;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_318) : "f"(big_28b), "f"(0.5f));
            float rounded_28b = ((_abs_315 == 0.5f) ? ((above_28b > 0.0f) ? _add_317 : _sub_318) : whole_28b);
            t1_28 = ((d1_28 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_28b, 7.888609e-31f), 1.7763568e-15f) : t1_28);
        }
        float2 _f32x2_mul_a_319 = mul_f32x2_noftz(_cvt_f32_57, make_float2(max_noftz(t0_28, 0.0f), max_noftz(t1_28, 0.0f)));
        float2 _mul_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_319), "l"(*(const unsigned long long*)&_cvt_f32_58));
        packed_hi[4] = __as_u32(__float22bfloat162_rn(make_float2(_mul_28.x, _mul_28.y)));
        float2 _cvt_f32_59 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[5]));
        float2 _cvt_f32_60 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[5]));
        float e0_29 = approx_exp2((-_cvt_f32_59.x) * 1.442695f);
        float e1_29 = approx_exp2((-_cvt_f32_59.y) * 1.442695f);
        float d0_29 = 1.0f + e0_29;
        float d1_29 = 1.0f + e1_29;
        float r0_29 = approx_rcp(d0_29);
        float r1_29 = approx_rcp(d1_29);
        float _fma_320;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_320) : "f"(-d0_29), "f"(r0_29), "f"(1.0f));
        float _fma_26_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26_1) : "f"(r0_29), "f"(_fma_320), "f"(r0_29));
        float _fma_321;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_321) : "f"(-d1_29), "f"(r1_29), "f"(1.0f));
        float _fma_27_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27_1) : "f"(r1_29), "f"(_fma_321), "f"(r1_29));
        float t0_29 = _fma_26_1;
        float t1_29 = _fma_27_1;
        int _warp_vote_322 = __any_sync(0xFFFFFFFF, fminf(t0_29, t1_29) == 0.0f);
        if (_warp_vote_322 != 0) {
            float d4_29a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_29a) : "f"(d0_29), "f"(0.25f));
            float r4_29a = approx_rcp(d4_29a);
            float _fma_324;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_324) : "f"(-d4_29a), "f"(r4_29a), "f"(1.0f));
            float _fma_323;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_323) : "f"(r4_29a), "f"(_fma_324), "f"(r4_29a));
            float big_29a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_29a) : "f"(mul_f32_rn_noftz(_fma_323, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_325;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_325) : "f"(big_29a), "f"(8.388608e+06f));
            float whole_29a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_29a) : "f"(_add_325), "f"(8.388608e+06f));
            float above_29a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_29a) : "f"(-d4_29a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_29a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_327;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_327) : "f"(big_29a), "f"(whole_29a));
            float _abs_326;
            asm("abs.f32 %0, %1;" : "=f"(_abs_326) : "f"(_sub_327));
            float _add_328;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_328) : "f"(big_29a), "f"(0.5f));
            float _sub_329;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_329) : "f"(big_29a), "f"(0.5f));
            float rounded_29a = ((_abs_326 == 0.5f) ? ((above_29a > 0.0f) ? _add_328 : _sub_329) : whole_29a);
            t0_29 = ((d0_29 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_29a, 7.888609e-31f), 1.7763568e-15f) : t0_29);
            float d4_29b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_29b) : "f"(d1_29), "f"(0.25f));
            float r4_29b = approx_rcp(d4_29b);
            float _fma_331;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_331) : "f"(-d4_29b), "f"(r4_29b), "f"(1.0f));
            float _fma_330;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_330) : "f"(r4_29b), "f"(_fma_331), "f"(r4_29b));
            float big_29b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_29b) : "f"(mul_f32_rn_noftz(_fma_330, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_332;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_332) : "f"(big_29b), "f"(8.388608e+06f));
            float whole_29b;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_29b) : "f"(_add_332), "f"(8.388608e+06f));
            float above_29b;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_29b) : "f"(-d4_29b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_29b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_334;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_334) : "f"(big_29b), "f"(whole_29b));
            float _abs_333;
            asm("abs.f32 %0, %1;" : "=f"(_abs_333) : "f"(_sub_334));
            float _add_335;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_335) : "f"(big_29b), "f"(0.5f));
            float _sub_336;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_336) : "f"(big_29b), "f"(0.5f));
            float rounded_29b = ((_abs_333 == 0.5f) ? ((above_29b > 0.0f) ? _add_335 : _sub_336) : whole_29b);
            t1_29 = ((d1_29 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_29b, 7.888609e-31f), 1.7763568e-15f) : t1_29);
        }
        float2 _f32x2_mul_a_337 = mul_f32x2_noftz(_cvt_f32_59, make_float2(max_noftz(t0_29, 0.0f), max_noftz(t1_29, 0.0f)));
        float2 _mul_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_337), "l"(*(const unsigned long long*)&_cvt_f32_60));
        packed_hi[5] = __as_u32(__float22bfloat162_rn(make_float2(_mul_29.x, _mul_29.y)));
        float2 _cvt_f32_61 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[6]));
        float2 _cvt_f32_62 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[6]));
        float e0_30 = approx_exp2((-_cvt_f32_61.x) * 1.442695f);
        float e1_30 = approx_exp2((-_cvt_f32_61.y) * 1.442695f);
        float d0_30 = 1.0f + e0_30;
        float d1_30 = 1.0f + e1_30;
        float r0_30 = approx_rcp(d0_30);
        float r1_30 = approx_rcp(d1_30);
        float _fma_338;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_338) : "f"(-d0_30), "f"(r0_30), "f"(1.0f));
        float _fma_28_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28_1) : "f"(r0_30), "f"(_fma_338), "f"(r0_30));
        float _fma_339;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_339) : "f"(-d1_30), "f"(r1_30), "f"(1.0f));
        float _fma_29_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29_1) : "f"(r1_30), "f"(_fma_339), "f"(r1_30));
        float t0_30 = _fma_28_1;
        float t1_30 = _fma_29_1;
        int _warp_vote_340 = __any_sync(0xFFFFFFFF, fminf(t0_30, t1_30) == 0.0f);
        if (_warp_vote_340 != 0) {
            float d4_30a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_30a) : "f"(d0_30), "f"(0.25f));
            float r4_30a = approx_rcp(d4_30a);
            float _fma_342;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_342) : "f"(-d4_30a), "f"(r4_30a), "f"(1.0f));
            float _fma_341;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_341) : "f"(r4_30a), "f"(_fma_342), "f"(r4_30a));
            float big_30a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_30a) : "f"(mul_f32_rn_noftz(_fma_341, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_343;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_343) : "f"(big_30a), "f"(8.388608e+06f));
            float whole_30a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_30a) : "f"(_add_343), "f"(8.388608e+06f));
            float above_30a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_30a) : "f"(-d4_30a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_30a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_345;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_345) : "f"(big_30a), "f"(whole_30a));
            float _abs_344;
            asm("abs.f32 %0, %1;" : "=f"(_abs_344) : "f"(_sub_345));
            float _add_346;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_346) : "f"(big_30a), "f"(0.5f));
            float _sub_347;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_347) : "f"(big_30a), "f"(0.5f));
            float rounded_30a = ((_abs_344 == 0.5f) ? ((above_30a > 0.0f) ? _add_346 : _sub_347) : whole_30a);
            t0_30 = ((d0_30 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_30a, 7.888609e-31f), 1.7763568e-15f) : t0_30);
            float d4_30b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_30b) : "f"(d1_30), "f"(0.25f));
            float r4_30b = approx_rcp(d4_30b);
            float _fma_349;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_349) : "f"(-d4_30b), "f"(r4_30b), "f"(1.0f));
            float _fma_348;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_348) : "f"(r4_30b), "f"(_fma_349), "f"(r4_30b));
            float big_30b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_30b) : "f"(mul_f32_rn_noftz(_fma_348, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_350;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_350) : "f"(big_30b), "f"(8.388608e+06f));
            float whole_30b;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_30b) : "f"(_add_350), "f"(8.388608e+06f));
            float above_30b;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_30b) : "f"(-d4_30b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_30b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_352;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_352) : "f"(big_30b), "f"(whole_30b));
            float _abs_351;
            asm("abs.f32 %0, %1;" : "=f"(_abs_351) : "f"(_sub_352));
            float _add_353;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_353) : "f"(big_30b), "f"(0.5f));
            float _sub_354;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_354) : "f"(big_30b), "f"(0.5f));
            float rounded_30b = ((_abs_351 == 0.5f) ? ((above_30b > 0.0f) ? _add_353 : _sub_354) : whole_30b);
            t1_30 = ((d1_30 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_30b, 7.888609e-31f), 1.7763568e-15f) : t1_30);
        }
        float2 _f32x2_mul_a_355 = mul_f32x2_noftz(_cvt_f32_61, make_float2(max_noftz(t0_30, 0.0f), max_noftz(t1_30, 0.0f)));
        float2 _mul_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_355), "l"(*(const unsigned long long*)&_cvt_f32_62));
        packed_hi[6] = __as_u32(__float22bfloat162_rn(make_float2(_mul_30.x, _mul_30.y)));
        float2 _cvt_f32_63 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[7]));
        float2 _cvt_f32_64 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[7]));
        float e0_31 = approx_exp2((-_cvt_f32_63.x) * 1.442695f);
        float e1_31 = approx_exp2((-_cvt_f32_63.y) * 1.442695f);
        float d0_31 = 1.0f + e0_31;
        float d1_31 = 1.0f + e1_31;
        float r0_31 = approx_rcp(d0_31);
        float r1_31 = approx_rcp(d1_31);
        float _fma_356;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_356) : "f"(-d0_31), "f"(r0_31), "f"(1.0f));
        float _fma_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(r0_31), "f"(_fma_356), "f"(r0_31));
        float _fma_357;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_357) : "f"(-d1_31), "f"(r1_31), "f"(1.0f));
        float _fma_31_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31_1) : "f"(r1_31), "f"(_fma_357), "f"(r1_31));
        float t0_31 = _fma_30;
        float t1_31 = _fma_31_1;
        int _warp_vote_358 = __any_sync(0xFFFFFFFF, fminf(t0_31, t1_31) == 0.0f);
        if (_warp_vote_358 != 0) {
            float d4_31a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_31a) : "f"(d0_31), "f"(0.25f));
            float r4_31a = approx_rcp(d4_31a);
            float _fma_360;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_360) : "f"(-d4_31a), "f"(r4_31a), "f"(1.0f));
            float _fma_359;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_359) : "f"(r4_31a), "f"(_fma_360), "f"(r4_31a));
            float big_31a;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_31a) : "f"(mul_f32_rn_noftz(_fma_359, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_361;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_361) : "f"(big_31a), "f"(8.388608e+06f));
            float whole_31a;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_31a) : "f"(_add_361), "f"(8.388608e+06f));
            float above_31a;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_31a) : "f"(-d4_31a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_31a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_363;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_363) : "f"(big_31a), "f"(whole_31a));
            float _abs_362;
            asm("abs.f32 %0, %1;" : "=f"(_abs_362) : "f"(_sub_363));
            float _add_364;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_364) : "f"(big_31a), "f"(0.5f));
            float _sub_365;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_365) : "f"(big_31a), "f"(0.5f));
            float rounded_31a = ((_abs_362 == 0.5f) ? ((above_31a > 0.0f) ? _add_364 : _sub_365) : whole_31a);
            t0_31 = ((d0_31 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_31a, 7.888609e-31f), 1.7763568e-15f) : t0_31);
            float d4_31b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_31b) : "f"(d1_31), "f"(0.25f));
            float r4_31b = approx_rcp(d4_31b);
            float _fma_367;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_367) : "f"(-d4_31b), "f"(r4_31b), "f"(1.0f));
            float _fma_366;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_366) : "f"(r4_31b), "f"(_fma_367), "f"(r4_31b));
            float big_31b;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_31b) : "f"(mul_f32_rn_noftz(_fma_366, 1.2676506e+30f)), "f"(1.4073749e+14f));
            float _add_368;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_368) : "f"(big_31b), "f"(8.388608e+06f));
            float whole_31b;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_31b) : "f"(_add_368), "f"(8.388608e+06f));
            float above_31b;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_31b) : "f"(-d4_31b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_31b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
            float _sub_370;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_370) : "f"(big_31b), "f"(whole_31b));
            float _abs_369;
            asm("abs.f32 %0, %1;" : "=f"(_abs_369) : "f"(_sub_370));
            float _add_371;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_371) : "f"(big_31b), "f"(0.5f));
            float _sub_372;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_372) : "f"(big_31b), "f"(0.5f));
            float rounded_31b = ((_abs_369 == 0.5f) ? ((above_31b > 0.0f) ? _add_371 : _sub_372) : whole_31b);
            t1_31 = ((d1_31 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_31b, 7.888609e-31f), 1.7763568e-15f) : t1_31);
        }
        float2 _f32x2_mul_a_373 = mul_f32x2_noftz(_cvt_f32_63, make_float2(max_noftz(t0_31, 0.0f), max_noftz(t1_31, 0.0f)));
        float2 _mul_31;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_373), "l"(*(const unsigned long long*)&_cvt_f32_64));
        packed_hi[7] = __as_u32(__float22bfloat162_rn(make_float2(_mul_31.x, _mul_31.y)));
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
    unsigned int strip_chunk0 = (unsigned int)(warp_1 * 2);
    unsigned int strip_src0 = tile_addr + (unsigned int)(srow * 256) + ((strip_chunk0 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[0]), "=r"(cols[1]), "=r"(cols[2]), "=r"(cols[3])
        : "r"(strip_src0)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[4]), "=r"(cols[5]), "=r"(cols[6]), "=r"(cols[7])
        : "r"(strip_src0 + 2048)
        : "memory");
    unsigned int strip_chunk1 = (unsigned int)(warp_1 * 2 + 1);
    unsigned int strip_src1 = tile_addr + (unsigned int)(srow * 256) + ((strip_chunk1 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[8]), "=r"(cols[9]), "=r"(cols[10]), "=r"(cols[11])
        : "r"(strip_src1)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[12]), "=r"(cols[13]), "=r"(cols[14]), "=r"(cols[15])
        : "r"(strip_src1 + 2048)
        : "memory");
    __nv_bfloat162 amax_a = __as_bf16x2(0);
    __nv_bfloat162 amax_b = __as_bf16x2(0);
    __nv_bfloat162 _hxa_375 = amax_a;
    __nv_bfloat162 _hxb_375 = __as_bf16x2(cols[0]);
    unsigned int _hxr_375;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_375) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_375)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_375)));
    __nv_bfloat162 _max_374 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_375);
    amax_a = _max_374;
    __nv_bfloat162 _hxa_377 = amax_b;
    __nv_bfloat162 _hxb_377 = __as_bf16x2(cols[8]);
    unsigned int _hxr_377;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_377) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_377)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_377)));
    __nv_bfloat162 _max_376 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_377);
    amax_b = _max_376;
    __nv_bfloat162 _hxa_379 = amax_a;
    __nv_bfloat162 _hxb_379 = __as_bf16x2(cols[1]);
    unsigned int _hxr_379;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_379) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_379)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_379)));
    __nv_bfloat162 _max_378 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_379);
    amax_a = _max_378;
    __nv_bfloat162 _hxa_381 = amax_b;
    __nv_bfloat162 _hxb_381 = __as_bf16x2(cols[9]);
    unsigned int _hxr_381;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_381) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_381)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_381)));
    __nv_bfloat162 _max_380 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_381);
    amax_b = _max_380;
    __nv_bfloat162 _hxa_383 = amax_a;
    __nv_bfloat162 _hxb_383 = __as_bf16x2(cols[2]);
    unsigned int _hxr_383;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_383) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_383)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_383)));
    __nv_bfloat162 _max_382 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_383);
    amax_a = _max_382;
    __nv_bfloat162 _hxa_385 = amax_b;
    __nv_bfloat162 _hxb_385 = __as_bf16x2(cols[10]);
    unsigned int _hxr_385;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_385) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_385)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_385)));
    __nv_bfloat162 _max_384 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_385);
    amax_b = _max_384;
    __nv_bfloat162 _hxa_387 = amax_a;
    __nv_bfloat162 _hxb_387 = __as_bf16x2(cols[3]);
    unsigned int _hxr_387;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_387) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_387)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_387)));
    __nv_bfloat162 _max_386 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_387);
    amax_a = _max_386;
    __nv_bfloat162 _hxa_389 = amax_b;
    __nv_bfloat162 _hxb_389 = __as_bf16x2(cols[11]);
    unsigned int _hxr_389;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_389) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_389)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_389)));
    __nv_bfloat162 _max_388 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_389);
    amax_b = _max_388;
    __nv_bfloat162 _hxa_391 = amax_a;
    __nv_bfloat162 _hxb_391 = __as_bf16x2(cols[4]);
    unsigned int _hxr_391;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_391) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_391)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_391)));
    __nv_bfloat162 _max_390 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_391);
    amax_a = _max_390;
    __nv_bfloat162 _hxa_393 = amax_b;
    __nv_bfloat162 _hxb_393 = __as_bf16x2(cols[12]);
    unsigned int _hxr_393;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_393) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_393)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_393)));
    __nv_bfloat162 _max_392 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_393);
    amax_b = _max_392;
    __nv_bfloat162 _hxa_395 = amax_a;
    __nv_bfloat162 _hxb_395 = __as_bf16x2(cols[5]);
    unsigned int _hxr_395;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_395) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_395)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_395)));
    __nv_bfloat162 _max_394 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_395);
    amax_a = _max_394;
    __nv_bfloat162 _hxa_397 = amax_b;
    __nv_bfloat162 _hxb_397 = __as_bf16x2(cols[13]);
    unsigned int _hxr_397;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_397) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_397)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_397)));
    __nv_bfloat162 _max_396 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_397);
    amax_b = _max_396;
    __nv_bfloat162 _hxa_399 = amax_a;
    __nv_bfloat162 _hxb_399 = __as_bf16x2(cols[6]);
    unsigned int _hxr_399;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_399) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_399)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_399)));
    __nv_bfloat162 _max_398 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_399);
    amax_a = _max_398;
    __nv_bfloat162 _hxa_401 = amax_b;
    __nv_bfloat162 _hxb_401 = __as_bf16x2(cols[14]);
    unsigned int _hxr_401;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_401) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_401)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_401)));
    __nv_bfloat162 _max_400 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_401);
    amax_b = _max_400;
    __nv_bfloat162 _hxa_403 = amax_a;
    __nv_bfloat162 _hxb_403 = __as_bf16x2(cols[7]);
    unsigned int _hxr_403;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_403) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_403)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_403)));
    __nv_bfloat162 _max_402 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_403);
    amax_a = _max_402;
    __nv_bfloat162 _hxa_405 = amax_b;
    __nv_bfloat162 _hxb_405 = __as_bf16x2(cols[15]);
    unsigned int _hxr_405;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_405) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_405)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_405)));
    __nv_bfloat162 _max_404 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_405);
    amax_b = _max_404;
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
    int scale_i32_1 = (int)(amax_bits_0 + 2031616 >> 23) - 8;
    if (scale_i32_1 < 0) {
        scale_i32_1 = 0;
    }
    if ((amax_bits_0 & 2139095040) == 2139095040) {
        scale_i32_1 = 255;
    }
    unsigned int scale_2 = (unsigned int)scale_i32_1;
    unsigned int inv_3 = 254 - scale_2 << 7;
    if (scale_2 == 255) {
        inv_3 = 32704;
    }
    unsigned int inv2_4 = inv_3 | inv_3 << 16;
    int gcol_a = bx * 128 + warp_1 * 16 + (lane_1 >> 2);
    int gcol_b = gcol_a + 8;
    int out_row = by * 64 + (lane_1 & 3) * 16;
    unsigned int words_a[4];
    unsigned int words_b[4];
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_407;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_407) : "r"(_bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_407) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_406;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_406) : "h"(_e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_407));
    words_a[0] = _mov_b32_pack_u16x2_406;
    uint32_t _bf16x2_mul_2;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_2) : "r"(cols[8]), "r"(inv2_4));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_3) : "r"(cols[9]), "r"(inv2_4));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_409;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_409) : "r"(_bf16x2_mul_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_409) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_408;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_408) : "h"(_e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_409));
    words_b[0] = _mov_b32_pack_u16x2_408;
    uint32_t _bf16x2_mul_4;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_4) : "r"(cols[2]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_5) : "r"(cols[3]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_411;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_411) : "r"(_bf16x2_mul_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_411) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_410;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_410) : "h"(_e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_411));
    words_a[1] = _mov_b32_pack_u16x2_410;
    uint32_t _bf16x2_mul_6;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_6) : "r"(cols[10]), "r"(inv2_4));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_7) : "r"(cols[11]), "r"(inv2_4));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_413;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_413) : "r"(_bf16x2_mul_7));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_7);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_413) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_412;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_412) : "h"(_e4m3x2_bf16x2_3), "h"(_cvt_satfinite_e4m3x2_bf16x2_413));
    words_b[1] = _mov_b32_pack_u16x2_412;
    uint32_t _bf16x2_mul_8;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_8) : "r"(cols[4]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_9) : "r"(cols[5]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_415;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_415) : "r"(_bf16x2_mul_9));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_9);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_415) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_414;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_414) : "h"(_e4m3x2_bf16x2_4), "h"(_cvt_satfinite_e4m3x2_bf16x2_415));
    words_a[2] = _mov_b32_pack_u16x2_414;
    uint32_t _bf16x2_mul_10;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_10) : "r"(cols[12]), "r"(inv2_4));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_11) : "r"(cols[13]), "r"(inv2_4));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_417;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_417) : "r"(_bf16x2_mul_11));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_11);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_417) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_416;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_416) : "h"(_e4m3x2_bf16x2_5), "h"(_cvt_satfinite_e4m3x2_bf16x2_417));
    words_b[2] = _mov_b32_pack_u16x2_416;
    uint32_t _bf16x2_mul_12;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_12) : "r"(cols[6]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_13) : "r"(cols[7]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_419;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_419) : "r"(_bf16x2_mul_13));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_13);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_419) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_418;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_418) : "h"(_e4m3x2_bf16x2_6), "h"(_cvt_satfinite_e4m3x2_bf16x2_419));
    words_a[3] = _mov_b32_pack_u16x2_418;
    uint32_t _bf16x2_mul_14;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_14) : "r"(cols[14]), "r"(inv2_4));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_15) : "r"(cols[15]), "r"(inv2_4));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_421;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_421) : "r"(_bf16x2_mul_15));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_15);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_421) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_420;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_420) : "h"(_e4m3x2_bf16x2_7), "h"(_cvt_satfinite_e4m3x2_bf16x2_421));
    words_b[3] = _mov_b32_pack_u16x2_420;
    reinterpret_cast<int4*>(col_output + (gcol_a * M + out_row))[0] = reinterpret_cast<int4*>(words_a + 0)[0];
    reinterpret_cast<int4*>(col_output + (gcol_b * M + out_row))[0] = reinterpret_cast<int4*>(words_b + 0)[0];
    if ((lane_1 & 1) == 0) {
        int row_tiles = M >> 7;
        int rblk = by * 2 + ((lane_1 & 3) >> 1);
        int col_scale_a = ((gcol_a >> 7) * row_tiles + (rblk >> 2)) * 512 + (gcol_a & 31) * 16 + (gcol_a >> 5 & 3) * 4 + (rblk & 3);
        int col_scale_b = ((gcol_b >> 7) * row_tiles + (rblk >> 2)) * 512 + (gcol_b & 31) * 16 + (gcol_b >> 5 & 3) * 4 + (rblk & 3);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_a) + (0)) = (unsigned char)(scale);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b) + (0)) = (unsigned char)(scale_2);
    }
}

} // extern "C"
