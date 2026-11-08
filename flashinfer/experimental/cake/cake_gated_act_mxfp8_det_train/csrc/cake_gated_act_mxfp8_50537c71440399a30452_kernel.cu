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

#define LAUNCH_MIN_BLOCKS 4

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

__device__ __noinline__ unsigned long long band_pair(__nv_bfloat16* src, __nv_bfloat16* dsrc, int index, int grad_index, int k)
{
    unsigned int _vec_load_0[1];
    {
        _vec_load_0[0] = *reinterpret_cast<const unsigned int*>(src + index + 0);
    }
    unsigned int _vec_load_1[1];
    {
        _vec_load_1[0] = *reinterpret_cast<const unsigned int*>(src + (index + k) + 0);
    }
    unsigned int _vec_load_2[1];
    {
        _vec_load_2[0] = *reinterpret_cast<const unsigned int*>(dsrc + grad_index + 0);
    }
    float2 x = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
    float2 u = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
    float2 g = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
    float e0 = approx_exp2(-x.x * 1.442695f);
    float e1 = approx_exp2(-x.y * 1.442695f);
    float d0 = 1.0f + e0;
    float d1 = 1.0f + e1;
    float r0 = approx_rcp(d0);
    float r1 = approx_rcp(d1);
    float _fma_0;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(-d0), "f"(r0), "f"(1.0f));
    float fma;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma) : "f"(r0), "f"(_fma_0), "f"(r0));
    float _fma_1;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(-d1), "f"(r1), "f"(1.0f));
    float fma_1;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1) : "f"(r1), "f"(_fma_1), "f"(r1));
    float t0 = fma;
    float t1 = fma_1;
    float d4_1a;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a) : "f"(d0), "f"(0.25f));
    float r4_1a = approx_rcp(d4_1a);
    float _fma_3;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(-d4_1a), "f"(r4_1a), "f"(1.0f));
    float _fma_2;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(r4_1a), "f"(_fma_3), "f"(r4_1a));
    float big_1a;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a) : "f"(mul_f32_rn_noftz(_fma_2, 1.2676506e+30f)), "f"(1.4073749e+14f));
    float _add_4;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(big_1a), "f"(8.388608e+06f));
    float whole_1a;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a) : "f"(_add_4), "f"(8.388608e+06f));
    float above_1a;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a) : "f"(-d4_1a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
    float _sub_6;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6) : "f"(big_1a), "f"(whole_1a));
    float _abs_5;
    asm("abs.f32 %0, %1;" : "=f"(_abs_5) : "f"(_sub_6));
    float _add_7;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(big_1a), "f"(0.5f));
    float _sub_8;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_8) : "f"(big_1a), "f"(0.5f));
    float rounded_1a = ((_abs_5 == 0.5f) ? ((above_1a > 0.0f) ? _add_7 : _sub_8) : whole_1a);
    float u0 = ((d0 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a, 7.888609e-31f), 1.7763568e-15f) : t0);
    float d4_1a_0;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_0) : "f"(d1), "f"(0.25f));
    float r4_1a_0 = approx_rcp(d4_1a_0);
    float _fma_10;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(-d4_1a_0), "f"(r4_1a_0), "f"(1.0f));
    float _fma_9;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(r4_1a_0), "f"(_fma_10), "f"(r4_1a_0));
    float big_1a_0;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_0) : "f"(mul_f32_rn_noftz(_fma_9, 1.2676506e+30f)), "f"(1.4073749e+14f));
    float _add_11;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11) : "f"(big_1a_0), "f"(8.388608e+06f));
    float whole_1a_0;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_0) : "f"(_add_11), "f"(8.388608e+06f));
    float above_1a_0;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_0) : "f"(-d4_1a_0), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_0, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
    float _sub_13;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_13) : "f"(big_1a_0), "f"(whole_1a_0));
    float _abs_12;
    asm("abs.f32 %0, %1;" : "=f"(_abs_12) : "f"(_sub_13));
    float _add_14;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(big_1a_0), "f"(0.5f));
    float _sub_15;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15) : "f"(big_1a_0), "f"(0.5f));
    float rounded_1a_0 = ((_abs_12 == 0.5f) ? ((above_1a_0 > 0.0f) ? _add_14 : _sub_15) : whole_1a_0);
    float u1 = ((d1 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_0, 7.888609e-31f), 1.7763568e-15f) : t1);
    float2 s = make_float2(max_noftz(u0, 0.0f), max_noftz(u1, 0.0f));
    float2 act;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act) : "l"(*(const unsigned long long*)&x), "l"(*(const unsigned long long*)&s));
    float2 _f32x2_mul_a_16 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s), s), g);
    float2 mul;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul) : "l"(*(const unsigned long long*)&_f32x2_mul_a_16), "l"(*(const unsigned long long*)&u));
    float2 mul_1;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1) : "l"(*(const unsigned long long*)&act), "l"(*(const unsigned long long*)&g));
    unsigned long long band_pair_out = (unsigned long long)__as_u32(__float22bfloat162_rn(make_float2(mul.x, mul.y))) | (unsigned long long)__as_u32(__float22bfloat162_rn(make_float2(mul_1.x, mul_1.y))) << 32ull;
    return band_pair_out;
}

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_gated_act_mxfp8_50537c71440399a30452(__nv_bfloat16* __restrict__ gated_input, __nv_bfloat16* __restrict__ grad, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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
    int half = tid & 1;
    int blk = tid >> 1 & 1;
    int row = tid >> 2;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int grow = by * 128 + row;
    int col = bx * 64 + blk * 32 + half * 16;
    int gate_index = grow * (2 * K) + col;
    unsigned int vec_load[8];
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
        vec_load[0 + 0] = (uint32_t)_iv_0_0;
        vec_load[0 + 1] = (uint32_t)_iv_0_1;
        vec_load[0 + 2] = (uint32_t)_iv_0_2;
        vec_load[0 + 3] = (uint32_t)_iv_0_3;
        vec_load[0 + 4] = (uint32_t)_iv_0_4;
        vec_load[0 + 5] = (uint32_t)_iv_0_5;
        vec_load[0 + 6] = (uint32_t)_iv_0_6;
        vec_load[0 + 7] = (uint32_t)_iv_0_7;
    }
    unsigned int vec_load_1[8];
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
        vec_load_1[0 + 0] = (uint32_t)_iv_1_0;
        vec_load_1[0 + 1] = (uint32_t)_iv_1_1;
        vec_load_1[0 + 2] = (uint32_t)_iv_1_2;
        vec_load_1[0 + 3] = (uint32_t)_iv_1_3;
        vec_load_1[0 + 4] = (uint32_t)_iv_1_4;
        vec_load_1[0 + 5] = (uint32_t)_iv_1_5;
        vec_load_1[0 + 6] = (uint32_t)_iv_1_6;
        vec_load_1[0 + 7] = (uint32_t)_iv_1_7;
    }
    unsigned int vec_load_2[8];
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
            : "=r"(_iv_2_0), "=r"(_iv_2_1), "=r"(_iv_2_2), "=r"(_iv_2_3), "=r"(_iv_2_4), "=r"(_iv_2_5), "=r"(_iv_2_6), "=r"(_iv_2_7) : "l"((const void*)(grad + (grow * K + col) + (0))) : "memory");
        vec_load_2[0 + 0] = (uint32_t)_iv_2_0;
        vec_load_2[0 + 1] = (uint32_t)_iv_2_1;
        vec_load_2[0 + 2] = (uint32_t)_iv_2_2;
        vec_load_2[0 + 3] = (uint32_t)_iv_2_3;
        vec_load_2[0 + 4] = (uint32_t)_iv_2_4;
        vec_load_2[0 + 5] = (uint32_t)_iv_2_5;
        vec_load_2[0 + 6] = (uint32_t)_iv_2_6;
        vec_load_2[0 + 7] = (uint32_t)_iv_2_7;
    }
    int grow_b = grow + 32;
    int grow_c = grow + 64;
    int grow_d = grow + 96;
    int gate_index_0 = grow_b * (2 * K) + col;
    unsigned int vec_load_0[8];
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
            : "=r"(_iv_3_0), "=r"(_iv_3_1), "=r"(_iv_3_2), "=r"(_iv_3_3), "=r"(_iv_3_4), "=r"(_iv_3_5), "=r"(_iv_3_6), "=r"(_iv_3_7) : "l"((const void*)(gated_input + gate_index_0 + (0))) : "memory");
        vec_load_0[0 + 0] = (uint32_t)_iv_3_0;
        vec_load_0[0 + 1] = (uint32_t)_iv_3_1;
        vec_load_0[0 + 2] = (uint32_t)_iv_3_2;
        vec_load_0[0 + 3] = (uint32_t)_iv_3_3;
        vec_load_0[0 + 4] = (uint32_t)_iv_3_4;
        vec_load_0[0 + 5] = (uint32_t)_iv_3_5;
        vec_load_0[0 + 6] = (uint32_t)_iv_3_6;
        vec_load_0[0 + 7] = (uint32_t)_iv_3_7;
    }
    unsigned int vec_load_1_0[8];
    {
        uint32_t _iv_4_0;
        uint32_t _iv_4_1;
        uint32_t _iv_4_2;
        uint32_t _iv_4_3;
        uint32_t _iv_4_4;
        uint32_t _iv_4_5;
        uint32_t _iv_4_6;
        uint32_t _iv_4_7;
        asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_iv_4_0), "=r"(_iv_4_1), "=r"(_iv_4_2), "=r"(_iv_4_3), "=r"(_iv_4_4), "=r"(_iv_4_5), "=r"(_iv_4_6), "=r"(_iv_4_7) : "l"((const void*)(gated_input + (gate_index_0 + K) + (0))) : "memory");
        vec_load_1_0[0 + 0] = (uint32_t)_iv_4_0;
        vec_load_1_0[0 + 1] = (uint32_t)_iv_4_1;
        vec_load_1_0[0 + 2] = (uint32_t)_iv_4_2;
        vec_load_1_0[0 + 3] = (uint32_t)_iv_4_3;
        vec_load_1_0[0 + 4] = (uint32_t)_iv_4_4;
        vec_load_1_0[0 + 5] = (uint32_t)_iv_4_5;
        vec_load_1_0[0 + 6] = (uint32_t)_iv_4_6;
        vec_load_1_0[0 + 7] = (uint32_t)_iv_4_7;
    }
    unsigned int vec_load_2_0[8];
    {
        uint32_t _iv_5_0;
        uint32_t _iv_5_1;
        uint32_t _iv_5_2;
        uint32_t _iv_5_3;
        uint32_t _iv_5_4;
        uint32_t _iv_5_5;
        uint32_t _iv_5_6;
        uint32_t _iv_5_7;
        asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_iv_5_0), "=r"(_iv_5_1), "=r"(_iv_5_2), "=r"(_iv_5_3), "=r"(_iv_5_4), "=r"(_iv_5_5), "=r"(_iv_5_6), "=r"(_iv_5_7) : "l"((const void*)(grad + (grow_b * K + col) + (0))) : "memory");
        vec_load_2_0[0 + 0] = (uint32_t)_iv_5_0;
        vec_load_2_0[0 + 1] = (uint32_t)_iv_5_1;
        vec_load_2_0[0 + 2] = (uint32_t)_iv_5_2;
        vec_load_2_0[0 + 3] = (uint32_t)_iv_5_3;
        vec_load_2_0[0 + 4] = (uint32_t)_iv_5_4;
        vec_load_2_0[0 + 5] = (uint32_t)_iv_5_5;
        vec_load_2_0[0 + 6] = (uint32_t)_iv_5_6;
        vec_load_2_0[0 + 7] = (uint32_t)_iv_5_7;
    }
    int gate_index_1 = grow_c * (2 * K) + col;
    unsigned int vec_load_3[8];
    {
        uint32_t _iv_6_0;
        uint32_t _iv_6_1;
        uint32_t _iv_6_2;
        uint32_t _iv_6_3;
        uint32_t _iv_6_4;
        uint32_t _iv_6_5;
        uint32_t _iv_6_6;
        uint32_t _iv_6_7;
        asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_iv_6_0), "=r"(_iv_6_1), "=r"(_iv_6_2), "=r"(_iv_6_3), "=r"(_iv_6_4), "=r"(_iv_6_5), "=r"(_iv_6_6), "=r"(_iv_6_7) : "l"((const void*)(gated_input + gate_index_1 + (0))) : "memory");
        vec_load_3[0 + 0] = (uint32_t)_iv_6_0;
        vec_load_3[0 + 1] = (uint32_t)_iv_6_1;
        vec_load_3[0 + 2] = (uint32_t)_iv_6_2;
        vec_load_3[0 + 3] = (uint32_t)_iv_6_3;
        vec_load_3[0 + 4] = (uint32_t)_iv_6_4;
        vec_load_3[0 + 5] = (uint32_t)_iv_6_5;
        vec_load_3[0 + 6] = (uint32_t)_iv_6_6;
        vec_load_3[0 + 7] = (uint32_t)_iv_6_7;
    }
    unsigned int vec_load_1_1[8];
    {
        uint32_t _iv_7_0;
        uint32_t _iv_7_1;
        uint32_t _iv_7_2;
        uint32_t _iv_7_3;
        uint32_t _iv_7_4;
        uint32_t _iv_7_5;
        uint32_t _iv_7_6;
        uint32_t _iv_7_7;
        asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_iv_7_0), "=r"(_iv_7_1), "=r"(_iv_7_2), "=r"(_iv_7_3), "=r"(_iv_7_4), "=r"(_iv_7_5), "=r"(_iv_7_6), "=r"(_iv_7_7) : "l"((const void*)(gated_input + (gate_index_1 + K) + (0))) : "memory");
        vec_load_1_1[0 + 0] = (uint32_t)_iv_7_0;
        vec_load_1_1[0 + 1] = (uint32_t)_iv_7_1;
        vec_load_1_1[0 + 2] = (uint32_t)_iv_7_2;
        vec_load_1_1[0 + 3] = (uint32_t)_iv_7_3;
        vec_load_1_1[0 + 4] = (uint32_t)_iv_7_4;
        vec_load_1_1[0 + 5] = (uint32_t)_iv_7_5;
        vec_load_1_1[0 + 6] = (uint32_t)_iv_7_6;
        vec_load_1_1[0 + 7] = (uint32_t)_iv_7_7;
    }
    unsigned int vec_load_2_1[8];
    {
        uint32_t _iv_8_0;
        uint32_t _iv_8_1;
        uint32_t _iv_8_2;
        uint32_t _iv_8_3;
        uint32_t _iv_8_4;
        uint32_t _iv_8_5;
        uint32_t _iv_8_6;
        uint32_t _iv_8_7;
        asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_iv_8_0), "=r"(_iv_8_1), "=r"(_iv_8_2), "=r"(_iv_8_3), "=r"(_iv_8_4), "=r"(_iv_8_5), "=r"(_iv_8_6), "=r"(_iv_8_7) : "l"((const void*)(grad + (grow_c * K + col) + (0))) : "memory");
        vec_load_2_1[0 + 0] = (uint32_t)_iv_8_0;
        vec_load_2_1[0 + 1] = (uint32_t)_iv_8_1;
        vec_load_2_1[0 + 2] = (uint32_t)_iv_8_2;
        vec_load_2_1[0 + 3] = (uint32_t)_iv_8_3;
        vec_load_2_1[0 + 4] = (uint32_t)_iv_8_4;
        vec_load_2_1[0 + 5] = (uint32_t)_iv_8_5;
        vec_load_2_1[0 + 6] = (uint32_t)_iv_8_6;
        vec_load_2_1[0 + 7] = (uint32_t)_iv_8_7;
    }
    int gate_index_2 = grow_d * (2 * K) + col;
    unsigned int vec_load_4[8];
    {
        uint32_t _iv_9_0;
        uint32_t _iv_9_1;
        uint32_t _iv_9_2;
        uint32_t _iv_9_3;
        uint32_t _iv_9_4;
        uint32_t _iv_9_5;
        uint32_t _iv_9_6;
        uint32_t _iv_9_7;
        asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_iv_9_0), "=r"(_iv_9_1), "=r"(_iv_9_2), "=r"(_iv_9_3), "=r"(_iv_9_4), "=r"(_iv_9_5), "=r"(_iv_9_6), "=r"(_iv_9_7) : "l"((const void*)(gated_input + gate_index_2 + (0))) : "memory");
        vec_load_4[0 + 0] = (uint32_t)_iv_9_0;
        vec_load_4[0 + 1] = (uint32_t)_iv_9_1;
        vec_load_4[0 + 2] = (uint32_t)_iv_9_2;
        vec_load_4[0 + 3] = (uint32_t)_iv_9_3;
        vec_load_4[0 + 4] = (uint32_t)_iv_9_4;
        vec_load_4[0 + 5] = (uint32_t)_iv_9_5;
        vec_load_4[0 + 6] = (uint32_t)_iv_9_6;
        vec_load_4[0 + 7] = (uint32_t)_iv_9_7;
    }
    unsigned int vec_load_1_2[8];
    {
        uint32_t _iv_10_0;
        uint32_t _iv_10_1;
        uint32_t _iv_10_2;
        uint32_t _iv_10_3;
        uint32_t _iv_10_4;
        uint32_t _iv_10_5;
        uint32_t _iv_10_6;
        uint32_t _iv_10_7;
        asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_iv_10_0), "=r"(_iv_10_1), "=r"(_iv_10_2), "=r"(_iv_10_3), "=r"(_iv_10_4), "=r"(_iv_10_5), "=r"(_iv_10_6), "=r"(_iv_10_7) : "l"((const void*)(gated_input + (gate_index_2 + K) + (0))) : "memory");
        vec_load_1_2[0 + 0] = (uint32_t)_iv_10_0;
        vec_load_1_2[0 + 1] = (uint32_t)_iv_10_1;
        vec_load_1_2[0 + 2] = (uint32_t)_iv_10_2;
        vec_load_1_2[0 + 3] = (uint32_t)_iv_10_3;
        vec_load_1_2[0 + 4] = (uint32_t)_iv_10_4;
        vec_load_1_2[0 + 5] = (uint32_t)_iv_10_5;
        vec_load_1_2[0 + 6] = (uint32_t)_iv_10_6;
        vec_load_1_2[0 + 7] = (uint32_t)_iv_10_7;
    }
    unsigned int vec_load_2_2[8];
    {
        uint32_t _iv_11_0;
        uint32_t _iv_11_1;
        uint32_t _iv_11_2;
        uint32_t _iv_11_3;
        uint32_t _iv_11_4;
        uint32_t _iv_11_5;
        uint32_t _iv_11_6;
        uint32_t _iv_11_7;
        asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_iv_11_0), "=r"(_iv_11_1), "=r"(_iv_11_2), "=r"(_iv_11_3), "=r"(_iv_11_4), "=r"(_iv_11_5), "=r"(_iv_11_6), "=r"(_iv_11_7) : "l"((const void*)(grad + (grow_d * K + col) + (0))) : "memory");
        vec_load_2_2[0 + 0] = (uint32_t)_iv_11_0;
        vec_load_2_2[0 + 1] = (uint32_t)_iv_11_1;
        vec_load_2_2[0 + 2] = (uint32_t)_iv_11_2;
        vec_load_2_2[0 + 3] = (uint32_t)_iv_11_3;
        vec_load_2_2[0 + 4] = (uint32_t)_iv_11_4;
        vec_load_2_2[0 + 5] = (uint32_t)_iv_11_5;
        vec_load_2_2[0 + 6] = (uint32_t)_iv_11_6;
        vec_load_2_2[0 + 7] = (uint32_t)_iv_11_7;
    }
    unsigned int dgate[8];
    unsigned int dup[8];
    unsigned int dgate_b[8];
    unsigned int dup_b[8];
    unsigned int dgate_c[8];
    unsigned int dup_c[8];
    unsigned int dgate_d[8];
    unsigned int dup_d[8];
    __nv_bfloat162 lowest = __as_bf16x2(vec_load[0]);
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[7]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_0[0]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_0[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_0[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_0[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_0[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_0[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_0[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_0[7]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_3[0]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_3[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_3[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_3[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_3[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_3[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_3[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_3[7]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_4[0]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_4[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_4[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_4[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_4[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_4[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_4[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_4[7]));
    float2 lowest_f32 = cake_bf16x2_to_f32x2(lowest);
    float _min_13;
    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_13) : "f"(lowest_f32.x), "f"(lowest_f32.y));
    int _warp_vote_12 = __all_sync(0xFFFFFFFF, _min_13 > -87.0f);
    if (_warp_vote_12 != 0) {
        float2 x = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[0]));
        float2 u = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[0]));
        float2 g = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[0]));
        float e0 = approx_exp2(-x.x * 1.442695f);
        float e1 = approx_exp2(-x.y * 1.442695f);
        float d0 = 1.0f + e0;
        float d1 = 1.0f + e1;
        float r0 = approx_rcp(d0);
        float r1 = approx_rcp(d1);
        float _fma_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(-d0), "f"(r0), "f"(1.0f));
        float _fma_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(r0), "f"(_fma_15), "f"(r0));
        float _fma_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(-d1), "f"(r1), "f"(1.0f));
        float _fma_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(r1), "f"(_fma_17), "f"(r1));
        float2 s = make_float2(_fma_14, _fma_16);
        float2 act;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act) : "l"(*(const unsigned long long*)&x), "l"(*(const unsigned long long*)&s));
        float2 _f32x2_mul_a_18 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s), s), g);
        float2 mul;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&u));
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
        float _fma_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(-d0_0), "f"(r0_0), "f"(1.0f));
        float _fma_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(r0_0), "f"(_fma_20), "f"(r0_0));
        float _fma_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(-d1_0), "f"(r1_0), "f"(1.0f));
        float _fma_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(r1_0), "f"(_fma_22), "f"(r1_0));
        float2 s_0 = make_float2(_fma_19, _fma_21);
        float2 act_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_0) : "l"(*(const unsigned long long*)&x_0), "l"(*(const unsigned long long*)&s_0));
        float2 _f32x2_mul_a_23 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_0), s_0), g_0);
        float2 mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&u_0));
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
        float _fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(-d0_1), "f"(r0_1), "f"(1.0f));
        float _fma_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(r0_1), "f"(_fma_25), "f"(r0_1));
        float _fma_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(-d1_1), "f"(r1_1), "f"(1.0f));
        float _fma_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(r1_1), "f"(_fma_27), "f"(r1_1));
        float2 s_1 = make_float2(_fma_24, _fma_26);
        float2 act_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_1) : "l"(*(const unsigned long long*)&x_1), "l"(*(const unsigned long long*)&s_1));
        float2 _f32x2_mul_a_28 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1), s_1), g_1);
        float2 mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&u_1));
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
        float _fma_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(-d0_2), "f"(r0_2), "f"(1.0f));
        float _fma_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(r0_2), "f"(_fma_30), "f"(r0_2));
        float _fma_32;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(-d1_2), "f"(r1_2), "f"(1.0f));
        float _fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(r1_2), "f"(_fma_32), "f"(r1_2));
        float2 s_2 = make_float2(_fma_29, _fma_31);
        float2 act_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_2) : "l"(*(const unsigned long long*)&x_2), "l"(*(const unsigned long long*)&s_2));
        float2 _f32x2_mul_a_33 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_2), s_2), g_2);
        float2 mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&u_2));
        float2 mul_1_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_2) : "l"(*(const unsigned long long*)&act_2), "l"(*(const unsigned long long*)&g_2));
        __nv_bfloat162 bf16x2_4 = __float22bfloat162_rn(make_float2(mul_3.x, mul_3.y));
        __nv_bfloat162 bf16x2_1_3 = __float22bfloat162_rn(make_float2(mul_1_2.x, mul_1_2.y));
        dgate[3] = __as_u32(bf16x2_4);
        dup[3] = __as_u32(bf16x2_1_3);
        float2 x_3 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[4]));
        float2 u_3 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[4]));
        float2 g_3 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[4]));
        float e0_3 = approx_exp2(-x_3.x * 1.442695f);
        float e1_3 = approx_exp2(-x_3.y * 1.442695f);
        float d0_3 = 1.0f + e0_3;
        float d1_3 = 1.0f + e1_3;
        float r0_3 = approx_rcp(d0_3);
        float r1_3 = approx_rcp(d1_3);
        float _fma_35;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(-d0_3), "f"(r0_3), "f"(1.0f));
        float _fma_34;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(r0_3), "f"(_fma_35), "f"(r0_3));
        float _fma_37;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(-d1_3), "f"(r1_3), "f"(1.0f));
        float _fma_36;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(r1_3), "f"(_fma_37), "f"(r1_3));
        float2 s_3 = make_float2(_fma_34, _fma_36);
        float2 act_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_3) : "l"(*(const unsigned long long*)&x_3), "l"(*(const unsigned long long*)&s_3));
        float2 _f32x2_mul_a_38 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_3), s_3), g_3);
        float2 mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_38), "l"(*(const unsigned long long*)&u_3));
        float2 mul_1_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_3) : "l"(*(const unsigned long long*)&act_3), "l"(*(const unsigned long long*)&g_3));
        __nv_bfloat162 bf16x2_5 = __float22bfloat162_rn(make_float2(mul_4.x, mul_4.y));
        __nv_bfloat162 bf16x2_1_4 = __float22bfloat162_rn(make_float2(mul_1_3.x, mul_1_3.y));
        dgate[4] = __as_u32(bf16x2_5);
        dup[4] = __as_u32(bf16x2_1_4);
        float2 x_4 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[5]));
        float2 u_4 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[5]));
        float2 g_4 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[5]));
        float e0_4 = approx_exp2(-x_4.x * 1.442695f);
        float e1_4 = approx_exp2(-x_4.y * 1.442695f);
        float d0_4 = 1.0f + e0_4;
        float d1_4 = 1.0f + e1_4;
        float r0_4 = approx_rcp(d0_4);
        float r1_4 = approx_rcp(d1_4);
        float _fma_40;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(-d0_4), "f"(r0_4), "f"(1.0f));
        float _fma_39;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(r0_4), "f"(_fma_40), "f"(r0_4));
        float _fma_42;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(-d1_4), "f"(r1_4), "f"(1.0f));
        float _fma_41;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(r1_4), "f"(_fma_42), "f"(r1_4));
        float2 s_4 = make_float2(_fma_39, _fma_41);
        float2 act_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_4) : "l"(*(const unsigned long long*)&x_4), "l"(*(const unsigned long long*)&s_4));
        float2 _f32x2_mul_a_43 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_4), s_4), g_4);
        float2 mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_43), "l"(*(const unsigned long long*)&u_4));
        float2 mul_1_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_4) : "l"(*(const unsigned long long*)&act_4), "l"(*(const unsigned long long*)&g_4));
        __nv_bfloat162 bf16x2_6 = __float22bfloat162_rn(make_float2(mul_5.x, mul_5.y));
        __nv_bfloat162 bf16x2_1_5 = __float22bfloat162_rn(make_float2(mul_1_4.x, mul_1_4.y));
        dgate[5] = __as_u32(bf16x2_6);
        dup[5] = __as_u32(bf16x2_1_5);
        float2 x_5 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[6]));
        float2 u_5 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[6]));
        float2 g_5 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[6]));
        float e0_5 = approx_exp2(-x_5.x * 1.442695f);
        float e1_5 = approx_exp2(-x_5.y * 1.442695f);
        float d0_5 = 1.0f + e0_5;
        float d1_5 = 1.0f + e1_5;
        float r0_5 = approx_rcp(d0_5);
        float r1_5 = approx_rcp(d1_5);
        float _fma_45;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(-d0_5), "f"(r0_5), "f"(1.0f));
        float _fma_44;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(r0_5), "f"(_fma_45), "f"(r0_5));
        float _fma_47;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(-d1_5), "f"(r1_5), "f"(1.0f));
        float _fma_46;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(r1_5), "f"(_fma_47), "f"(r1_5));
        float2 s_5 = make_float2(_fma_44, _fma_46);
        float2 act_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_5) : "l"(*(const unsigned long long*)&x_5), "l"(*(const unsigned long long*)&s_5));
        float2 _f32x2_mul_a_48 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_5), s_5), g_5);
        float2 mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_48), "l"(*(const unsigned long long*)&u_5));
        float2 mul_1_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_5) : "l"(*(const unsigned long long*)&act_5), "l"(*(const unsigned long long*)&g_5));
        __nv_bfloat162 bf16x2_7 = __float22bfloat162_rn(make_float2(mul_6.x, mul_6.y));
        __nv_bfloat162 bf16x2_1_6 = __float22bfloat162_rn(make_float2(mul_1_5.x, mul_1_5.y));
        dgate[6] = __as_u32(bf16x2_7);
        dup[6] = __as_u32(bf16x2_1_6);
        float2 x_6 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[7]));
        float2 u_6 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[7]));
        float2 g_6 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[7]));
        float e0_6 = approx_exp2(-x_6.x * 1.442695f);
        float e1_6 = approx_exp2(-x_6.y * 1.442695f);
        float d0_6 = 1.0f + e0_6;
        float d1_6 = 1.0f + e1_6;
        float r0_6 = approx_rcp(d0_6);
        float r1_6 = approx_rcp(d1_6);
        float _fma_50;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(-d0_6), "f"(r0_6), "f"(1.0f));
        float _fma_49;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(r0_6), "f"(_fma_50), "f"(r0_6));
        float _fma_52;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(-d1_6), "f"(r1_6), "f"(1.0f));
        float _fma_51;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(r1_6), "f"(_fma_52), "f"(r1_6));
        float2 s_6 = make_float2(_fma_49, _fma_51);
        float2 act_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_6) : "l"(*(const unsigned long long*)&x_6), "l"(*(const unsigned long long*)&s_6));
        float2 _f32x2_mul_a_53 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_6), s_6), g_6);
        float2 mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_53), "l"(*(const unsigned long long*)&u_6));
        float2 mul_1_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_6) : "l"(*(const unsigned long long*)&act_6), "l"(*(const unsigned long long*)&g_6));
        __nv_bfloat162 bf16x2_8 = __float22bfloat162_rn(make_float2(mul_7.x, mul_7.y));
        __nv_bfloat162 bf16x2_1_7 = __float22bfloat162_rn(make_float2(mul_1_6.x, mul_1_6.y));
        dgate[7] = __as_u32(bf16x2_8);
        dup[7] = __as_u32(bf16x2_1_7);
        float2 x_7 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[0]));
        float2 u_7 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[0]));
        float2 g_7 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[0]));
        float e0_7 = approx_exp2(-x_7.x * 1.442695f);
        float e1_7 = approx_exp2(-x_7.y * 1.442695f);
        float d0_7 = 1.0f + e0_7;
        float d1_7 = 1.0f + e1_7;
        float r0_7 = approx_rcp(d0_7);
        float r1_7 = approx_rcp(d1_7);
        float _fma_55;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(-d0_7), "f"(r0_7), "f"(1.0f));
        float _fma_54;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(r0_7), "f"(_fma_55), "f"(r0_7));
        float _fma_57;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(-d1_7), "f"(r1_7), "f"(1.0f));
        float _fma_56;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(r1_7), "f"(_fma_57), "f"(r1_7));
        float2 s_7 = make_float2(_fma_54, _fma_56);
        float2 act_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_7) : "l"(*(const unsigned long long*)&x_7), "l"(*(const unsigned long long*)&s_7));
        float2 _f32x2_mul_a_58 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_7, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_7), s_7), g_7);
        float2 mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_58), "l"(*(const unsigned long long*)&u_7));
        float2 mul_1_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_7) : "l"(*(const unsigned long long*)&act_7), "l"(*(const unsigned long long*)&g_7));
        __nv_bfloat162 bf16x2_9 = __float22bfloat162_rn(make_float2(mul_8.x, mul_8.y));
        __nv_bfloat162 bf16x2_1_8 = __float22bfloat162_rn(make_float2(mul_1_7.x, mul_1_7.y));
        dgate_b[0] = __as_u32(bf16x2_9);
        dup_b[0] = __as_u32(bf16x2_1_8);
        float2 x_8 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[1]));
        float2 u_8 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[1]));
        float2 g_8 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[1]));
        float e0_8 = approx_exp2(-x_8.x * 1.442695f);
        float e1_8 = approx_exp2(-x_8.y * 1.442695f);
        float d0_8 = 1.0f + e0_8;
        float d1_8 = 1.0f + e1_8;
        float r0_8 = approx_rcp(d0_8);
        float r1_8 = approx_rcp(d1_8);
        float _fma_60;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(-d0_8), "f"(r0_8), "f"(1.0f));
        float _fma_59;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(r0_8), "f"(_fma_60), "f"(r0_8));
        float _fma_62;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(-d1_8), "f"(r1_8), "f"(1.0f));
        float _fma_61;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(r1_8), "f"(_fma_62), "f"(r1_8));
        float2 s_8 = make_float2(_fma_59, _fma_61);
        float2 act_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_8) : "l"(*(const unsigned long long*)&x_8), "l"(*(const unsigned long long*)&s_8));
        float2 _f32x2_mul_a_63 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_8, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_8), s_8), g_8);
        float2 mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_63), "l"(*(const unsigned long long*)&u_8));
        float2 mul_1_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_8) : "l"(*(const unsigned long long*)&act_8), "l"(*(const unsigned long long*)&g_8));
        __nv_bfloat162 bf16x2_10 = __float22bfloat162_rn(make_float2(mul_9.x, mul_9.y));
        __nv_bfloat162 bf16x2_1_9 = __float22bfloat162_rn(make_float2(mul_1_8.x, mul_1_8.y));
        dgate_b[1] = __as_u32(bf16x2_10);
        dup_b[1] = __as_u32(bf16x2_1_9);
        float2 x_9 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[2]));
        float2 u_9 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[2]));
        float2 g_9 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[2]));
        float e0_9 = approx_exp2(-x_9.x * 1.442695f);
        float e1_9 = approx_exp2(-x_9.y * 1.442695f);
        float d0_9 = 1.0f + e0_9;
        float d1_9 = 1.0f + e1_9;
        float r0_9 = approx_rcp(d0_9);
        float r1_9 = approx_rcp(d1_9);
        float _fma_65;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(-d0_9), "f"(r0_9), "f"(1.0f));
        float _fma_64;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(r0_9), "f"(_fma_65), "f"(r0_9));
        float _fma_67;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(-d1_9), "f"(r1_9), "f"(1.0f));
        float _fma_66;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(r1_9), "f"(_fma_67), "f"(r1_9));
        float2 s_9 = make_float2(_fma_64, _fma_66);
        float2 act_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_9) : "l"(*(const unsigned long long*)&x_9), "l"(*(const unsigned long long*)&s_9));
        float2 _f32x2_mul_a_68 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_9, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_9), s_9), g_9);
        float2 mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&u_9));
        float2 mul_1_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_9) : "l"(*(const unsigned long long*)&act_9), "l"(*(const unsigned long long*)&g_9));
        __nv_bfloat162 bf16x2_11 = __float22bfloat162_rn(make_float2(mul_10.x, mul_10.y));
        __nv_bfloat162 bf16x2_1_10 = __float22bfloat162_rn(make_float2(mul_1_9.x, mul_1_9.y));
        dgate_b[2] = __as_u32(bf16x2_11);
        dup_b[2] = __as_u32(bf16x2_1_10);
        float2 x_10 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[3]));
        float2 u_10 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[3]));
        float2 g_10 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[3]));
        float e0_10 = approx_exp2(-x_10.x * 1.442695f);
        float e1_10 = approx_exp2(-x_10.y * 1.442695f);
        float d0_10 = 1.0f + e0_10;
        float d1_10 = 1.0f + e1_10;
        float r0_10 = approx_rcp(d0_10);
        float r1_10 = approx_rcp(d1_10);
        float _fma_70;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(-d0_10), "f"(r0_10), "f"(1.0f));
        float _fma_69;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(r0_10), "f"(_fma_70), "f"(r0_10));
        float _fma_72;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(-d1_10), "f"(r1_10), "f"(1.0f));
        float _fma_71;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(r1_10), "f"(_fma_72), "f"(r1_10));
        float2 s_10 = make_float2(_fma_69, _fma_71);
        float2 act_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_10) : "l"(*(const unsigned long long*)&x_10), "l"(*(const unsigned long long*)&s_10));
        float2 _f32x2_mul_a_73 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_10, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_10), s_10), g_10);
        float2 mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_73), "l"(*(const unsigned long long*)&u_10));
        float2 mul_1_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_10) : "l"(*(const unsigned long long*)&act_10), "l"(*(const unsigned long long*)&g_10));
        __nv_bfloat162 bf16x2_12 = __float22bfloat162_rn(make_float2(mul_11.x, mul_11.y));
        __nv_bfloat162 bf16x2_1_11 = __float22bfloat162_rn(make_float2(mul_1_10.x, mul_1_10.y));
        dgate_b[3] = __as_u32(bf16x2_12);
        dup_b[3] = __as_u32(bf16x2_1_11);
        float2 x_11 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[4]));
        float2 u_11 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[4]));
        float2 g_11 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[4]));
        float e0_11 = approx_exp2(-x_11.x * 1.442695f);
        float e1_11 = approx_exp2(-x_11.y * 1.442695f);
        float d0_11 = 1.0f + e0_11;
        float d1_11 = 1.0f + e1_11;
        float r0_11 = approx_rcp(d0_11);
        float r1_11 = approx_rcp(d1_11);
        float _fma_75;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(-d0_11), "f"(r0_11), "f"(1.0f));
        float _fma_74;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(r0_11), "f"(_fma_75), "f"(r0_11));
        float _fma_77;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_77) : "f"(-d1_11), "f"(r1_11), "f"(1.0f));
        float _fma_76;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(r1_11), "f"(_fma_77), "f"(r1_11));
        float2 s_11 = make_float2(_fma_74, _fma_76);
        float2 act_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_11) : "l"(*(const unsigned long long*)&x_11), "l"(*(const unsigned long long*)&s_11));
        float2 _f32x2_mul_a_78 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_11, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_11), s_11), g_11);
        float2 mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_78), "l"(*(const unsigned long long*)&u_11));
        float2 mul_1_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_11) : "l"(*(const unsigned long long*)&act_11), "l"(*(const unsigned long long*)&g_11));
        __nv_bfloat162 bf16x2_13 = __float22bfloat162_rn(make_float2(mul_12.x, mul_12.y));
        __nv_bfloat162 bf16x2_1_12 = __float22bfloat162_rn(make_float2(mul_1_11.x, mul_1_11.y));
        dgate_b[4] = __as_u32(bf16x2_13);
        dup_b[4] = __as_u32(bf16x2_1_12);
        float2 x_12 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[5]));
        float2 u_12 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[5]));
        float2 g_12 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[5]));
        float e0_12 = approx_exp2(-x_12.x * 1.442695f);
        float e1_12 = approx_exp2(-x_12.y * 1.442695f);
        float d0_12 = 1.0f + e0_12;
        float d1_12 = 1.0f + e1_12;
        float r0_12 = approx_rcp(d0_12);
        float r1_12 = approx_rcp(d1_12);
        float _fma_80;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(-d0_12), "f"(r0_12), "f"(1.0f));
        float _fma_79;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(r0_12), "f"(_fma_80), "f"(r0_12));
        float _fma_82;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(-d1_12), "f"(r1_12), "f"(1.0f));
        float _fma_81;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(r1_12), "f"(_fma_82), "f"(r1_12));
        float2 s_12 = make_float2(_fma_79, _fma_81);
        float2 act_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_12) : "l"(*(const unsigned long long*)&x_12), "l"(*(const unsigned long long*)&s_12));
        float2 _f32x2_mul_a_83 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_12, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_12), s_12), g_12);
        float2 mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_83), "l"(*(const unsigned long long*)&u_12));
        float2 mul_1_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_12) : "l"(*(const unsigned long long*)&act_12), "l"(*(const unsigned long long*)&g_12));
        __nv_bfloat162 bf16x2_14 = __float22bfloat162_rn(make_float2(mul_13.x, mul_13.y));
        __nv_bfloat162 bf16x2_1_13 = __float22bfloat162_rn(make_float2(mul_1_12.x, mul_1_12.y));
        dgate_b[5] = __as_u32(bf16x2_14);
        dup_b[5] = __as_u32(bf16x2_1_13);
        float2 x_13 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[6]));
        float2 u_13 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[6]));
        float2 g_13 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[6]));
        float e0_13 = approx_exp2(-x_13.x * 1.442695f);
        float e1_13 = approx_exp2(-x_13.y * 1.442695f);
        float d0_13 = 1.0f + e0_13;
        float d1_13 = 1.0f + e1_13;
        float r0_13 = approx_rcp(d0_13);
        float r1_13 = approx_rcp(d1_13);
        float _fma_85;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(-d0_13), "f"(r0_13), "f"(1.0f));
        float _fma_84;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(r0_13), "f"(_fma_85), "f"(r0_13));
        float _fma_87;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(-d1_13), "f"(r1_13), "f"(1.0f));
        float _fma_86;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(r1_13), "f"(_fma_87), "f"(r1_13));
        float2 s_13 = make_float2(_fma_84, _fma_86);
        float2 act_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_13) : "l"(*(const unsigned long long*)&x_13), "l"(*(const unsigned long long*)&s_13));
        float2 _f32x2_mul_a_88 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_13, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_13), s_13), g_13);
        float2 mul_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_88), "l"(*(const unsigned long long*)&u_13));
        float2 mul_1_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_13) : "l"(*(const unsigned long long*)&act_13), "l"(*(const unsigned long long*)&g_13));
        __nv_bfloat162 bf16x2_15 = __float22bfloat162_rn(make_float2(mul_14.x, mul_14.y));
        __nv_bfloat162 bf16x2_1_14 = __float22bfloat162_rn(make_float2(mul_1_13.x, mul_1_13.y));
        dgate_b[6] = __as_u32(bf16x2_15);
        dup_b[6] = __as_u32(bf16x2_1_14);
        float2 x_14 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[7]));
        float2 u_14 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[7]));
        float2 g_14 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[7]));
        float e0_14 = approx_exp2(-x_14.x * 1.442695f);
        float e1_14 = approx_exp2(-x_14.y * 1.442695f);
        float d0_14 = 1.0f + e0_14;
        float d1_14 = 1.0f + e1_14;
        float r0_14 = approx_rcp(d0_14);
        float r1_14 = approx_rcp(d1_14);
        float _fma_90;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(-d0_14), "f"(r0_14), "f"(1.0f));
        float _fma_89;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(r0_14), "f"(_fma_90), "f"(r0_14));
        float _fma_92;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(-d1_14), "f"(r1_14), "f"(1.0f));
        float _fma_91;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(r1_14), "f"(_fma_92), "f"(r1_14));
        float2 s_14 = make_float2(_fma_89, _fma_91);
        float2 act_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_14) : "l"(*(const unsigned long long*)&x_14), "l"(*(const unsigned long long*)&s_14));
        float2 _f32x2_mul_a_93 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_14, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_14), s_14), g_14);
        float2 mul_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_93), "l"(*(const unsigned long long*)&u_14));
        float2 mul_1_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_14) : "l"(*(const unsigned long long*)&act_14), "l"(*(const unsigned long long*)&g_14));
        __nv_bfloat162 bf16x2_16 = __float22bfloat162_rn(make_float2(mul_15.x, mul_15.y));
        __nv_bfloat162 bf16x2_1_15 = __float22bfloat162_rn(make_float2(mul_1_14.x, mul_1_14.y));
        dgate_b[7] = __as_u32(bf16x2_16);
        dup_b[7] = __as_u32(bf16x2_1_15);
        float2 x_15 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[0]));
        float2 u_15 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[0]));
        float2 g_15 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[0]));
        float e0_15 = approx_exp2(-x_15.x * 1.442695f);
        float e1_15 = approx_exp2(-x_15.y * 1.442695f);
        float d0_15 = 1.0f + e0_15;
        float d1_15 = 1.0f + e1_15;
        float r0_15 = approx_rcp(d0_15);
        float r1_15 = approx_rcp(d1_15);
        float _fma_95;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(-d0_15), "f"(r0_15), "f"(1.0f));
        float _fma_94;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_94) : "f"(r0_15), "f"(_fma_95), "f"(r0_15));
        float _fma_97;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(-d1_15), "f"(r1_15), "f"(1.0f));
        float _fma_96;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(r1_15), "f"(_fma_97), "f"(r1_15));
        float2 s_15 = make_float2(_fma_94, _fma_96);
        float2 act_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_15) : "l"(*(const unsigned long long*)&x_15), "l"(*(const unsigned long long*)&s_15));
        float2 _f32x2_mul_a_98 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_15, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_15), s_15), g_15);
        float2 mul_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_98), "l"(*(const unsigned long long*)&u_15));
        float2 mul_1_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_15) : "l"(*(const unsigned long long*)&act_15), "l"(*(const unsigned long long*)&g_15));
        __nv_bfloat162 bf16x2_17 = __float22bfloat162_rn(make_float2(mul_16.x, mul_16.y));
        __nv_bfloat162 bf16x2_1_16 = __float22bfloat162_rn(make_float2(mul_1_15.x, mul_1_15.y));
        dgate_c[0] = __as_u32(bf16x2_17);
        dup_c[0] = __as_u32(bf16x2_1_16);
        float2 x_16 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[1]));
        float2 u_16 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[1]));
        float2 g_16 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[1]));
        float e0_16 = approx_exp2(-x_16.x * 1.442695f);
        float e1_16 = approx_exp2(-x_16.y * 1.442695f);
        float d0_16 = 1.0f + e0_16;
        float d1_16 = 1.0f + e1_16;
        float r0_16 = approx_rcp(d0_16);
        float r1_16 = approx_rcp(d1_16);
        float _fma_100;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_100) : "f"(-d0_16), "f"(r0_16), "f"(1.0f));
        float _fma_99;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(r0_16), "f"(_fma_100), "f"(r0_16));
        float _fma_102;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_102) : "f"(-d1_16), "f"(r1_16), "f"(1.0f));
        float _fma_101;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(r1_16), "f"(_fma_102), "f"(r1_16));
        float2 s_16 = make_float2(_fma_99, _fma_101);
        float2 act_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_16) : "l"(*(const unsigned long long*)&x_16), "l"(*(const unsigned long long*)&s_16));
        float2 _f32x2_mul_a_103 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_16, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_16), s_16), g_16);
        float2 mul_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_103), "l"(*(const unsigned long long*)&u_16));
        float2 mul_1_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_16) : "l"(*(const unsigned long long*)&act_16), "l"(*(const unsigned long long*)&g_16));
        __nv_bfloat162 bf16x2_18 = __float22bfloat162_rn(make_float2(mul_17.x, mul_17.y));
        __nv_bfloat162 bf16x2_1_17 = __float22bfloat162_rn(make_float2(mul_1_16.x, mul_1_16.y));
        dgate_c[1] = __as_u32(bf16x2_18);
        dup_c[1] = __as_u32(bf16x2_1_17);
        float2 x_17 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[2]));
        float2 u_17 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[2]));
        float2 g_17 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[2]));
        float e0_17 = approx_exp2(-x_17.x * 1.442695f);
        float e1_17 = approx_exp2(-x_17.y * 1.442695f);
        float d0_17 = 1.0f + e0_17;
        float d1_17 = 1.0f + e1_17;
        float r0_17 = approx_rcp(d0_17);
        float r1_17 = approx_rcp(d1_17);
        float _fma_105;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_105) : "f"(-d0_17), "f"(r0_17), "f"(1.0f));
        float _fma_104;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_104) : "f"(r0_17), "f"(_fma_105), "f"(r0_17));
        float _fma_107;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(-d1_17), "f"(r1_17), "f"(1.0f));
        float _fma_106;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_106) : "f"(r1_17), "f"(_fma_107), "f"(r1_17));
        float2 s_17 = make_float2(_fma_104, _fma_106);
        float2 act_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_17) : "l"(*(const unsigned long long*)&x_17), "l"(*(const unsigned long long*)&s_17));
        float2 _f32x2_mul_a_108 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_17, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_17), s_17), g_17);
        float2 mul_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_108), "l"(*(const unsigned long long*)&u_17));
        float2 mul_1_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_17) : "l"(*(const unsigned long long*)&act_17), "l"(*(const unsigned long long*)&g_17));
        __nv_bfloat162 bf16x2_19 = __float22bfloat162_rn(make_float2(mul_18.x, mul_18.y));
        __nv_bfloat162 bf16x2_1_18 = __float22bfloat162_rn(make_float2(mul_1_17.x, mul_1_17.y));
        dgate_c[2] = __as_u32(bf16x2_19);
        dup_c[2] = __as_u32(bf16x2_1_18);
        float2 x_18 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[3]));
        float2 u_18 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[3]));
        float2 g_18 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[3]));
        float e0_18 = approx_exp2(-x_18.x * 1.442695f);
        float e1_18 = approx_exp2(-x_18.y * 1.442695f);
        float d0_18 = 1.0f + e0_18;
        float d1_18 = 1.0f + e1_18;
        float r0_18 = approx_rcp(d0_18);
        float r1_18 = approx_rcp(d1_18);
        float _fma_110;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_110) : "f"(-d0_18), "f"(r0_18), "f"(1.0f));
        float _fma_109;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_109) : "f"(r0_18), "f"(_fma_110), "f"(r0_18));
        float _fma_112;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_112) : "f"(-d1_18), "f"(r1_18), "f"(1.0f));
        float _fma_111;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_111) : "f"(r1_18), "f"(_fma_112), "f"(r1_18));
        float2 s_18 = make_float2(_fma_109, _fma_111);
        float2 act_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_18) : "l"(*(const unsigned long long*)&x_18), "l"(*(const unsigned long long*)&s_18));
        float2 _f32x2_mul_a_113 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_18, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_18), s_18), g_18);
        float2 mul_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_113), "l"(*(const unsigned long long*)&u_18));
        float2 mul_1_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_18) : "l"(*(const unsigned long long*)&act_18), "l"(*(const unsigned long long*)&g_18));
        __nv_bfloat162 bf16x2_20 = __float22bfloat162_rn(make_float2(mul_19.x, mul_19.y));
        __nv_bfloat162 bf16x2_1_19 = __float22bfloat162_rn(make_float2(mul_1_18.x, mul_1_18.y));
        dgate_c[3] = __as_u32(bf16x2_20);
        dup_c[3] = __as_u32(bf16x2_1_19);
        float2 x_19 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[4]));
        float2 u_19 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[4]));
        float2 g_19 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[4]));
        float e0_19 = approx_exp2(-x_19.x * 1.442695f);
        float e1_19 = approx_exp2(-x_19.y * 1.442695f);
        float d0_19 = 1.0f + e0_19;
        float d1_19 = 1.0f + e1_19;
        float r0_19 = approx_rcp(d0_19);
        float r1_19 = approx_rcp(d1_19);
        float _fma_115;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_115) : "f"(-d0_19), "f"(r0_19), "f"(1.0f));
        float _fma_114;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_114) : "f"(r0_19), "f"(_fma_115), "f"(r0_19));
        float _fma_117;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_117) : "f"(-d1_19), "f"(r1_19), "f"(1.0f));
        float _fma_116;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_116) : "f"(r1_19), "f"(_fma_117), "f"(r1_19));
        float2 s_19 = make_float2(_fma_114, _fma_116);
        float2 act_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_19) : "l"(*(const unsigned long long*)&x_19), "l"(*(const unsigned long long*)&s_19));
        float2 _f32x2_mul_a_118 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_19, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_19), s_19), g_19);
        float2 mul_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_118), "l"(*(const unsigned long long*)&u_19));
        float2 mul_1_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_19) : "l"(*(const unsigned long long*)&act_19), "l"(*(const unsigned long long*)&g_19));
        __nv_bfloat162 bf16x2_21 = __float22bfloat162_rn(make_float2(mul_20.x, mul_20.y));
        __nv_bfloat162 bf16x2_1_20 = __float22bfloat162_rn(make_float2(mul_1_19.x, mul_1_19.y));
        dgate_c[4] = __as_u32(bf16x2_21);
        dup_c[4] = __as_u32(bf16x2_1_20);
        float2 x_20 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[5]));
        float2 u_20 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[5]));
        float2 g_20 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[5]));
        float e0_20 = approx_exp2(-x_20.x * 1.442695f);
        float e1_20 = approx_exp2(-x_20.y * 1.442695f);
        float d0_20 = 1.0f + e0_20;
        float d1_20 = 1.0f + e1_20;
        float r0_20 = approx_rcp(d0_20);
        float r1_20 = approx_rcp(d1_20);
        float _fma_120;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_120) : "f"(-d0_20), "f"(r0_20), "f"(1.0f));
        float _fma_119;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_119) : "f"(r0_20), "f"(_fma_120), "f"(r0_20));
        float _fma_122;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_122) : "f"(-d1_20), "f"(r1_20), "f"(1.0f));
        float _fma_121;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_121) : "f"(r1_20), "f"(_fma_122), "f"(r1_20));
        float2 s_20 = make_float2(_fma_119, _fma_121);
        float2 act_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_20) : "l"(*(const unsigned long long*)&x_20), "l"(*(const unsigned long long*)&s_20));
        float2 _f32x2_mul_a_123 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_20, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_20), s_20), g_20);
        float2 mul_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_123), "l"(*(const unsigned long long*)&u_20));
        float2 mul_1_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_20) : "l"(*(const unsigned long long*)&act_20), "l"(*(const unsigned long long*)&g_20));
        __nv_bfloat162 bf16x2_22 = __float22bfloat162_rn(make_float2(mul_21.x, mul_21.y));
        __nv_bfloat162 bf16x2_1_21 = __float22bfloat162_rn(make_float2(mul_1_20.x, mul_1_20.y));
        dgate_c[5] = __as_u32(bf16x2_22);
        dup_c[5] = __as_u32(bf16x2_1_21);
        float2 x_21 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[6]));
        float2 u_21 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[6]));
        float2 g_21 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[6]));
        float e0_21 = approx_exp2(-x_21.x * 1.442695f);
        float e1_21 = approx_exp2(-x_21.y * 1.442695f);
        float d0_21 = 1.0f + e0_21;
        float d1_21 = 1.0f + e1_21;
        float r0_21 = approx_rcp(d0_21);
        float r1_21 = approx_rcp(d1_21);
        float _fma_125;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(-d0_21), "f"(r0_21), "f"(1.0f));
        float _fma_124;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_124) : "f"(r0_21), "f"(_fma_125), "f"(r0_21));
        float _fma_127;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_127) : "f"(-d1_21), "f"(r1_21), "f"(1.0f));
        float _fma_126;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_126) : "f"(r1_21), "f"(_fma_127), "f"(r1_21));
        float2 s_21 = make_float2(_fma_124, _fma_126);
        float2 act_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_21) : "l"(*(const unsigned long long*)&x_21), "l"(*(const unsigned long long*)&s_21));
        float2 _f32x2_mul_a_128 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_21, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_21), s_21), g_21);
        float2 mul_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_128), "l"(*(const unsigned long long*)&u_21));
        float2 mul_1_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_21) : "l"(*(const unsigned long long*)&act_21), "l"(*(const unsigned long long*)&g_21));
        __nv_bfloat162 bf16x2_23 = __float22bfloat162_rn(make_float2(mul_22.x, mul_22.y));
        __nv_bfloat162 bf16x2_1_22 = __float22bfloat162_rn(make_float2(mul_1_21.x, mul_1_21.y));
        dgate_c[6] = __as_u32(bf16x2_23);
        dup_c[6] = __as_u32(bf16x2_1_22);
        float2 x_22 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[7]));
        float2 u_22 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[7]));
        float2 g_22 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[7]));
        float e0_22 = approx_exp2(-x_22.x * 1.442695f);
        float e1_22 = approx_exp2(-x_22.y * 1.442695f);
        float d0_22 = 1.0f + e0_22;
        float d1_22 = 1.0f + e1_22;
        float r0_22 = approx_rcp(d0_22);
        float r1_22 = approx_rcp(d1_22);
        float _fma_130;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_130) : "f"(-d0_22), "f"(r0_22), "f"(1.0f));
        float _fma_129;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_129) : "f"(r0_22), "f"(_fma_130), "f"(r0_22));
        float _fma_132;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_132) : "f"(-d1_22), "f"(r1_22), "f"(1.0f));
        float _fma_131;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_131) : "f"(r1_22), "f"(_fma_132), "f"(r1_22));
        float2 s_22 = make_float2(_fma_129, _fma_131);
        float2 act_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_22) : "l"(*(const unsigned long long*)&x_22), "l"(*(const unsigned long long*)&s_22));
        float2 _f32x2_mul_a_133 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_22, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_22), s_22), g_22);
        float2 mul_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_133), "l"(*(const unsigned long long*)&u_22));
        float2 mul_1_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_22) : "l"(*(const unsigned long long*)&act_22), "l"(*(const unsigned long long*)&g_22));
        __nv_bfloat162 bf16x2_24 = __float22bfloat162_rn(make_float2(mul_23.x, mul_23.y));
        __nv_bfloat162 bf16x2_1_23 = __float22bfloat162_rn(make_float2(mul_1_22.x, mul_1_22.y));
        dgate_c[7] = __as_u32(bf16x2_24);
        dup_c[7] = __as_u32(bf16x2_1_23);
        float2 x_23 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_4[0]));
        float2 u_23 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_2[0]));
        float2 g_23 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_2[0]));
        float e0_23 = approx_exp2(-x_23.x * 1.442695f);
        float e1_23 = approx_exp2(-x_23.y * 1.442695f);
        float d0_23 = 1.0f + e0_23;
        float d1_23 = 1.0f + e1_23;
        float r0_23 = approx_rcp(d0_23);
        float r1_23 = approx_rcp(d1_23);
        float _fma_135;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_135) : "f"(-d0_23), "f"(r0_23), "f"(1.0f));
        float _fma_134;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_134) : "f"(r0_23), "f"(_fma_135), "f"(r0_23));
        float _fma_137;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_137) : "f"(-d1_23), "f"(r1_23), "f"(1.0f));
        float _fma_136;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_136) : "f"(r1_23), "f"(_fma_137), "f"(r1_23));
        float2 s_23 = make_float2(_fma_134, _fma_136);
        float2 act_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_23) : "l"(*(const unsigned long long*)&x_23), "l"(*(const unsigned long long*)&s_23));
        float2 _f32x2_mul_a_138 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_23, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_23), s_23), g_23);
        float2 mul_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_138), "l"(*(const unsigned long long*)&u_23));
        float2 mul_1_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_23) : "l"(*(const unsigned long long*)&act_23), "l"(*(const unsigned long long*)&g_23));
        __nv_bfloat162 bf16x2_25 = __float22bfloat162_rn(make_float2(mul_24.x, mul_24.y));
        __nv_bfloat162 bf16x2_1_24 = __float22bfloat162_rn(make_float2(mul_1_23.x, mul_1_23.y));
        dgate_d[0] = __as_u32(bf16x2_25);
        dup_d[0] = __as_u32(bf16x2_1_24);
        float2 x_24 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_4[1]));
        float2 u_24 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_2[1]));
        float2 g_24 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_2[1]));
        float e0_24 = approx_exp2(-x_24.x * 1.442695f);
        float e1_24 = approx_exp2(-x_24.y * 1.442695f);
        float d0_24 = 1.0f + e0_24;
        float d1_24 = 1.0f + e1_24;
        float r0_24 = approx_rcp(d0_24);
        float r1_24 = approx_rcp(d1_24);
        float _fma_140;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_140) : "f"(-d0_24), "f"(r0_24), "f"(1.0f));
        float _fma_139;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_139) : "f"(r0_24), "f"(_fma_140), "f"(r0_24));
        float _fma_142;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_142) : "f"(-d1_24), "f"(r1_24), "f"(1.0f));
        float _fma_141;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_141) : "f"(r1_24), "f"(_fma_142), "f"(r1_24));
        float2 s_24 = make_float2(_fma_139, _fma_141);
        float2 act_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_24) : "l"(*(const unsigned long long*)&x_24), "l"(*(const unsigned long long*)&s_24));
        float2 _f32x2_mul_a_143 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_24, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_24), s_24), g_24);
        float2 mul_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_143), "l"(*(const unsigned long long*)&u_24));
        float2 mul_1_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_24) : "l"(*(const unsigned long long*)&act_24), "l"(*(const unsigned long long*)&g_24));
        __nv_bfloat162 bf16x2_26 = __float22bfloat162_rn(make_float2(mul_25.x, mul_25.y));
        __nv_bfloat162 bf16x2_1_25 = __float22bfloat162_rn(make_float2(mul_1_24.x, mul_1_24.y));
        dgate_d[1] = __as_u32(bf16x2_26);
        dup_d[1] = __as_u32(bf16x2_1_25);
        float2 x_25 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_4[2]));
        float2 u_25 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_2[2]));
        float2 g_25 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_2[2]));
        float e0_25 = approx_exp2(-x_25.x * 1.442695f);
        float e1_25 = approx_exp2(-x_25.y * 1.442695f);
        float d0_25 = 1.0f + e0_25;
        float d1_25 = 1.0f + e1_25;
        float r0_25 = approx_rcp(d0_25);
        float r1_25 = approx_rcp(d1_25);
        float _fma_145;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_145) : "f"(-d0_25), "f"(r0_25), "f"(1.0f));
        float _fma_144;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_144) : "f"(r0_25), "f"(_fma_145), "f"(r0_25));
        float _fma_147;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_147) : "f"(-d1_25), "f"(r1_25), "f"(1.0f));
        float _fma_146;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_146) : "f"(r1_25), "f"(_fma_147), "f"(r1_25));
        float2 s_25 = make_float2(_fma_144, _fma_146);
        float2 act_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_25) : "l"(*(const unsigned long long*)&x_25), "l"(*(const unsigned long long*)&s_25));
        float2 _f32x2_mul_a_148 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_25, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_25), s_25), g_25);
        float2 mul_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_148), "l"(*(const unsigned long long*)&u_25));
        float2 mul_1_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_25) : "l"(*(const unsigned long long*)&act_25), "l"(*(const unsigned long long*)&g_25));
        __nv_bfloat162 bf16x2_27 = __float22bfloat162_rn(make_float2(mul_26.x, mul_26.y));
        __nv_bfloat162 bf16x2_1_26 = __float22bfloat162_rn(make_float2(mul_1_25.x, mul_1_25.y));
        dgate_d[2] = __as_u32(bf16x2_27);
        dup_d[2] = __as_u32(bf16x2_1_26);
        float2 x_26 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_4[3]));
        float2 u_26 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_2[3]));
        float2 g_26 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_2[3]));
        float e0_26 = approx_exp2(-x_26.x * 1.442695f);
        float e1_26 = approx_exp2(-x_26.y * 1.442695f);
        float d0_26 = 1.0f + e0_26;
        float d1_26 = 1.0f + e1_26;
        float r0_26 = approx_rcp(d0_26);
        float r1_26 = approx_rcp(d1_26);
        float _fma_150;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_150) : "f"(-d0_26), "f"(r0_26), "f"(1.0f));
        float _fma_149;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_149) : "f"(r0_26), "f"(_fma_150), "f"(r0_26));
        float _fma_152;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_152) : "f"(-d1_26), "f"(r1_26), "f"(1.0f));
        float _fma_151;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_151) : "f"(r1_26), "f"(_fma_152), "f"(r1_26));
        float2 s_26 = make_float2(_fma_149, _fma_151);
        float2 act_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_26) : "l"(*(const unsigned long long*)&x_26), "l"(*(const unsigned long long*)&s_26));
        float2 _f32x2_mul_a_153 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_26, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_26), s_26), g_26);
        float2 mul_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_153), "l"(*(const unsigned long long*)&u_26));
        float2 mul_1_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_26) : "l"(*(const unsigned long long*)&act_26), "l"(*(const unsigned long long*)&g_26));
        __nv_bfloat162 bf16x2_28 = __float22bfloat162_rn(make_float2(mul_27.x, mul_27.y));
        __nv_bfloat162 bf16x2_1_27 = __float22bfloat162_rn(make_float2(mul_1_26.x, mul_1_26.y));
        dgate_d[3] = __as_u32(bf16x2_28);
        dup_d[3] = __as_u32(bf16x2_1_27);
        float2 x_27 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_4[4]));
        float2 u_27 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_2[4]));
        float2 g_27 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_2[4]));
        float e0_27 = approx_exp2(-x_27.x * 1.442695f);
        float e1_27 = approx_exp2(-x_27.y * 1.442695f);
        float d0_27 = 1.0f + e0_27;
        float d1_27 = 1.0f + e1_27;
        float r0_27 = approx_rcp(d0_27);
        float r1_27 = approx_rcp(d1_27);
        float _fma_155;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_155) : "f"(-d0_27), "f"(r0_27), "f"(1.0f));
        float _fma_154;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_154) : "f"(r0_27), "f"(_fma_155), "f"(r0_27));
        float _fma_157;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_157) : "f"(-d1_27), "f"(r1_27), "f"(1.0f));
        float _fma_156;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_156) : "f"(r1_27), "f"(_fma_157), "f"(r1_27));
        float2 s_27 = make_float2(_fma_154, _fma_156);
        float2 act_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_27) : "l"(*(const unsigned long long*)&x_27), "l"(*(const unsigned long long*)&s_27));
        float2 _f32x2_mul_a_158 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_27, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_27), s_27), g_27);
        float2 mul_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_158), "l"(*(const unsigned long long*)&u_27));
        float2 mul_1_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_27) : "l"(*(const unsigned long long*)&act_27), "l"(*(const unsigned long long*)&g_27));
        __nv_bfloat162 bf16x2_29 = __float22bfloat162_rn(make_float2(mul_28.x, mul_28.y));
        __nv_bfloat162 bf16x2_1_28 = __float22bfloat162_rn(make_float2(mul_1_27.x, mul_1_27.y));
        dgate_d[4] = __as_u32(bf16x2_29);
        dup_d[4] = __as_u32(bf16x2_1_28);
        float2 x_28 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_4[5]));
        float2 u_28 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_2[5]));
        float2 g_28 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_2[5]));
        float e0_28 = approx_exp2(-x_28.x * 1.442695f);
        float e1_28 = approx_exp2(-x_28.y * 1.442695f);
        float d0_28 = 1.0f + e0_28;
        float d1_28 = 1.0f + e1_28;
        float r0_28 = approx_rcp(d0_28);
        float r1_28 = approx_rcp(d1_28);
        float _fma_160;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_160) : "f"(-d0_28), "f"(r0_28), "f"(1.0f));
        float _fma_159;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_159) : "f"(r0_28), "f"(_fma_160), "f"(r0_28));
        float _fma_162;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_162) : "f"(-d1_28), "f"(r1_28), "f"(1.0f));
        float _fma_161;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_161) : "f"(r1_28), "f"(_fma_162), "f"(r1_28));
        float2 s_28 = make_float2(_fma_159, _fma_161);
        float2 act_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_28) : "l"(*(const unsigned long long*)&x_28), "l"(*(const unsigned long long*)&s_28));
        float2 _f32x2_mul_a_163 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_28, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_28), s_28), g_28);
        float2 mul_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_163), "l"(*(const unsigned long long*)&u_28));
        float2 mul_1_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_28) : "l"(*(const unsigned long long*)&act_28), "l"(*(const unsigned long long*)&g_28));
        __nv_bfloat162 bf16x2_30 = __float22bfloat162_rn(make_float2(mul_29.x, mul_29.y));
        __nv_bfloat162 bf16x2_1_29 = __float22bfloat162_rn(make_float2(mul_1_28.x, mul_1_28.y));
        dgate_d[5] = __as_u32(bf16x2_30);
        dup_d[5] = __as_u32(bf16x2_1_29);
        float2 x_29 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_4[6]));
        float2 u_29 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_2[6]));
        float2 g_29 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_2[6]));
        float e0_29 = approx_exp2(-x_29.x * 1.442695f);
        float e1_29 = approx_exp2(-x_29.y * 1.442695f);
        float d0_29 = 1.0f + e0_29;
        float d1_29 = 1.0f + e1_29;
        float r0_29 = approx_rcp(d0_29);
        float r1_29 = approx_rcp(d1_29);
        float _fma_165;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_165) : "f"(-d0_29), "f"(r0_29), "f"(1.0f));
        float _fma_164;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_164) : "f"(r0_29), "f"(_fma_165), "f"(r0_29));
        float _fma_167;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_167) : "f"(-d1_29), "f"(r1_29), "f"(1.0f));
        float _fma_166;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_166) : "f"(r1_29), "f"(_fma_167), "f"(r1_29));
        float2 s_29 = make_float2(_fma_164, _fma_166);
        float2 act_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_29) : "l"(*(const unsigned long long*)&x_29), "l"(*(const unsigned long long*)&s_29));
        float2 _f32x2_mul_a_168 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_29, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_29), s_29), g_29);
        float2 mul_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_168), "l"(*(const unsigned long long*)&u_29));
        float2 mul_1_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_29) : "l"(*(const unsigned long long*)&act_29), "l"(*(const unsigned long long*)&g_29));
        __nv_bfloat162 bf16x2_31 = __float22bfloat162_rn(make_float2(mul_30.x, mul_30.y));
        __nv_bfloat162 bf16x2_1_30 = __float22bfloat162_rn(make_float2(mul_1_29.x, mul_1_29.y));
        dgate_d[6] = __as_u32(bf16x2_31);
        dup_d[6] = __as_u32(bf16x2_1_30);
        float2 x_30 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_4[7]));
        float2 u_30 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_2[7]));
        float2 g_30 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_2[7]));
        float e0_30 = approx_exp2(-x_30.x * 1.442695f);
        float e1_30 = approx_exp2(-x_30.y * 1.442695f);
        float d0_30 = 1.0f + e0_30;
        float d1_30 = 1.0f + e1_30;
        float r0_30 = approx_rcp(d0_30);
        float r1_30 = approx_rcp(d1_30);
        float _fma_170;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_170) : "f"(-d0_30), "f"(r0_30), "f"(1.0f));
        float _fma_169;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_169) : "f"(r0_30), "f"(_fma_170), "f"(r0_30));
        float _fma_172;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_172) : "f"(-d1_30), "f"(r1_30), "f"(1.0f));
        float _fma_171;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_171) : "f"(r1_30), "f"(_fma_172), "f"(r1_30));
        float2 s_30 = make_float2(_fma_169, _fma_171);
        float2 act_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_30) : "l"(*(const unsigned long long*)&x_30), "l"(*(const unsigned long long*)&s_30));
        float2 _f32x2_mul_a_173 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_30, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_30), s_30), g_30);
        float2 mul_31;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_173), "l"(*(const unsigned long long*)&u_30));
        float2 mul_1_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_30) : "l"(*(const unsigned long long*)&act_30), "l"(*(const unsigned long long*)&g_30));
        __nv_bfloat162 bf16x2_32 = __float22bfloat162_rn(make_float2(mul_31.x, mul_31.y));
        __nv_bfloat162 bf16x2_1_31 = __float22bfloat162_rn(make_float2(mul_1_30.x, mul_1_30.y));
        dgate_d[7] = __as_u32(bf16x2_32);
        dup_d[7] = __as_u32(bf16x2_1_31);
    } else {
        int band = 0;
        float2 x_31 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[0]));
        float2 u_31 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[0]));
        float2 g_31 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[0]));
        float e0_31 = approx_exp2(-x_31.x * 1.442695f);
        float e1_31 = approx_exp2(-x_31.y * 1.442695f);
        float d0_31 = 1.0f + e0_31;
        float d1_31 = 1.0f + e1_31;
        float r0_31 = approx_rcp(d0_31);
        float r1_31 = approx_rcp(d1_31);
        float _fma_174;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_174) : "f"(-d0_31), "f"(r0_31), "f"(1.0f));
        float fma;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma) : "f"(r0_31), "f"(_fma_174), "f"(r0_31));
        float _fma_175;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_175) : "f"(-d1_31), "f"(r1_31), "f"(1.0f));
        float fma_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1) : "f"(r1_31), "f"(_fma_175), "f"(r1_31));
        float t0 = fma;
        float t1 = fma_1;
        band = band | ((fminf(t0, t1) == 0.0f) ? 1 : 0);
        float2 s_31 = make_float2(max_noftz(t0, 0.0f), max_noftz(t1, 0.0f));
        float2 act_31;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_31) : "l"(*(const unsigned long long*)&x_31), "l"(*(const unsigned long long*)&s_31));
        float2 _f32x2_mul_a_176 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_31, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_31), s_31), g_31);
        float2 mul_32;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_32) : "l"(*(const unsigned long long*)&_f32x2_mul_a_176), "l"(*(const unsigned long long*)&u_31));
        float2 mul_1_31;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_31) : "l"(*(const unsigned long long*)&act_31), "l"(*(const unsigned long long*)&g_31));
        __nv_bfloat162 bf16x2_33 = __float22bfloat162_rn(make_float2(mul_32.x, mul_32.y));
        __nv_bfloat162 bf16x2_1_32 = __float22bfloat162_rn(make_float2(mul_1_31.x, mul_1_31.y));
        dgate[0] = __as_u32(bf16x2_33);
        dup[0] = __as_u32(bf16x2_1_32);
        float2 x_31_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[1]));
        float2 u_31_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[1]));
        float2 g_31_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[1]));
        float e0_31_1 = approx_exp2(-x_31_1.x * 1.442695f);
        float e1_31_1 = approx_exp2(-x_31_1.y * 1.442695f);
        float d0_31_1 = 1.0f + e0_31_1;
        float d1_31_1 = 1.0f + e1_31_1;
        float r0_31_1 = approx_rcp(d0_31_1);
        float r1_31_1 = approx_rcp(d1_31_1);
        float _fma_177;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_177) : "f"(-d0_31_1), "f"(r0_31_1), "f"(1.0f));
        float fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_0) : "f"(r0_31_1), "f"(_fma_177), "f"(r0_31_1));
        float _fma_178;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_178) : "f"(-d1_31_1), "f"(r1_31_1), "f"(1.0f));
        float fma_1_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_0) : "f"(r1_31_1), "f"(_fma_178), "f"(r1_31_1));
        float t0_0 = fma_0;
        float t1_0 = fma_1_0;
        band = band | ((fminf(t0_0, t1_0) == 0.0f) ? 2 : 0);
        float2 s_31_1 = make_float2(max_noftz(t0_0, 0.0f), max_noftz(t1_0, 0.0f));
        float2 act_31_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_31_1) : "l"(*(const unsigned long long*)&x_31_1), "l"(*(const unsigned long long*)&s_31_1));
        float2 _f32x2_mul_a_179 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_31_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_31_1), s_31_1), g_31_1);
        float2 mul_32_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_32_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_179), "l"(*(const unsigned long long*)&u_31_1));
        float2 mul_1_31_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_31_1) : "l"(*(const unsigned long long*)&act_31_1), "l"(*(const unsigned long long*)&g_31_1));
        __nv_bfloat162 bf16x2_34 = __float22bfloat162_rn(make_float2(mul_32_1.x, mul_32_1.y));
        __nv_bfloat162 bf16x2_1_33 = __float22bfloat162_rn(make_float2(mul_1_31_1.x, mul_1_31_1.y));
        dgate[1] = __as_u32(bf16x2_34);
        dup[1] = __as_u32(bf16x2_1_33);
        float2 x_32 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[2]));
        float2 u_32 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[2]));
        float2 g_32 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[2]));
        float e0_32 = approx_exp2(-x_32.x * 1.442695f);
        float e1_32 = approx_exp2(-x_32.y * 1.442695f);
        float d0_32 = 1.0f + e0_32;
        float d1_32 = 1.0f + e1_32;
        float r0_32 = approx_rcp(d0_32);
        float r1_32 = approx_rcp(d1_32);
        float _fma_180;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_180) : "f"(-d0_32), "f"(r0_32), "f"(1.0f));
        float fma_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_2) : "f"(r0_32), "f"(_fma_180), "f"(r0_32));
        float _fma_181;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_181) : "f"(-d1_32), "f"(r1_32), "f"(1.0f));
        float fma_1_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_1) : "f"(r1_32), "f"(_fma_181), "f"(r1_32));
        float t0_1 = fma_2;
        float t1_1 = fma_1_1;
        band = band | ((fminf(t0_1, t1_1) == 0.0f) ? 4 : 0);
        float2 s_32 = make_float2(max_noftz(t0_1, 0.0f), max_noftz(t1_1, 0.0f));
        float2 act_32;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_32) : "l"(*(const unsigned long long*)&x_32), "l"(*(const unsigned long long*)&s_32));
        float2 _f32x2_mul_a_182 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_32, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_32), s_32), g_32);
        float2 mul_33;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_33) : "l"(*(const unsigned long long*)&_f32x2_mul_a_182), "l"(*(const unsigned long long*)&u_32));
        float2 mul_1_32;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_32) : "l"(*(const unsigned long long*)&act_32), "l"(*(const unsigned long long*)&g_32));
        __nv_bfloat162 bf16x2_35 = __float22bfloat162_rn(make_float2(mul_33.x, mul_33.y));
        __nv_bfloat162 bf16x2_1_34 = __float22bfloat162_rn(make_float2(mul_1_32.x, mul_1_32.y));
        dgate[2] = __as_u32(bf16x2_35);
        dup[2] = __as_u32(bf16x2_1_34);
        float2 x_33 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[3]));
        float2 u_33 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[3]));
        float2 g_33 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[3]));
        float e0_33 = approx_exp2(-x_33.x * 1.442695f);
        float e1_33 = approx_exp2(-x_33.y * 1.442695f);
        float d0_33 = 1.0f + e0_33;
        float d1_33 = 1.0f + e1_33;
        float r0_33 = approx_rcp(d0_33);
        float r1_33 = approx_rcp(d1_33);
        float _fma_183;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_183) : "f"(-d0_33), "f"(r0_33), "f"(1.0f));
        float fma_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_3) : "f"(r0_33), "f"(_fma_183), "f"(r0_33));
        float _fma_184;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_184) : "f"(-d1_33), "f"(r1_33), "f"(1.0f));
        float fma_1_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_2) : "f"(r1_33), "f"(_fma_184), "f"(r1_33));
        float t0_2 = fma_3;
        float t1_2 = fma_1_2;
        band = band | ((fminf(t0_2, t1_2) == 0.0f) ? 8 : 0);
        float2 s_33 = make_float2(max_noftz(t0_2, 0.0f), max_noftz(t1_2, 0.0f));
        float2 act_33;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_33) : "l"(*(const unsigned long long*)&x_33), "l"(*(const unsigned long long*)&s_33));
        float2 _f32x2_mul_a_185 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_33, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_33), s_33), g_33);
        float2 mul_34;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_34) : "l"(*(const unsigned long long*)&_f32x2_mul_a_185), "l"(*(const unsigned long long*)&u_33));
        float2 mul_1_33;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_33) : "l"(*(const unsigned long long*)&act_33), "l"(*(const unsigned long long*)&g_33));
        __nv_bfloat162 bf16x2_36 = __float22bfloat162_rn(make_float2(mul_34.x, mul_34.y));
        __nv_bfloat162 bf16x2_1_35 = __float22bfloat162_rn(make_float2(mul_1_33.x, mul_1_33.y));
        dgate[3] = __as_u32(bf16x2_36);
        dup[3] = __as_u32(bf16x2_1_35);
        float2 x_34 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[4]));
        float2 u_34 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[4]));
        float2 g_34 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[4]));
        float e0_34 = approx_exp2(-x_34.x * 1.442695f);
        float e1_34 = approx_exp2(-x_34.y * 1.442695f);
        float d0_34 = 1.0f + e0_34;
        float d1_34 = 1.0f + e1_34;
        float r0_34 = approx_rcp(d0_34);
        float r1_34 = approx_rcp(d1_34);
        float _fma_186;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_186) : "f"(-d0_34), "f"(r0_34), "f"(1.0f));
        float fma_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_4) : "f"(r0_34), "f"(_fma_186), "f"(r0_34));
        float _fma_187;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_187) : "f"(-d1_34), "f"(r1_34), "f"(1.0f));
        float fma_1_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_3) : "f"(r1_34), "f"(_fma_187), "f"(r1_34));
        float t0_3 = fma_4;
        float t1_3 = fma_1_3;
        band = band | ((fminf(t0_3, t1_3) == 0.0f) ? 16 : 0);
        float2 s_34 = make_float2(max_noftz(t0_3, 0.0f), max_noftz(t1_3, 0.0f));
        float2 act_34;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_34) : "l"(*(const unsigned long long*)&x_34), "l"(*(const unsigned long long*)&s_34));
        float2 _f32x2_mul_a_188 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_34, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_34), s_34), g_34);
        float2 mul_35;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_35) : "l"(*(const unsigned long long*)&_f32x2_mul_a_188), "l"(*(const unsigned long long*)&u_34));
        float2 mul_1_34;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_34) : "l"(*(const unsigned long long*)&act_34), "l"(*(const unsigned long long*)&g_34));
        __nv_bfloat162 bf16x2_37 = __float22bfloat162_rn(make_float2(mul_35.x, mul_35.y));
        __nv_bfloat162 bf16x2_1_36 = __float22bfloat162_rn(make_float2(mul_1_34.x, mul_1_34.y));
        dgate[4] = __as_u32(bf16x2_37);
        dup[4] = __as_u32(bf16x2_1_36);
        float2 x_35 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[5]));
        float2 u_35 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[5]));
        float2 g_35 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[5]));
        float e0_35 = approx_exp2(-x_35.x * 1.442695f);
        float e1_35 = approx_exp2(-x_35.y * 1.442695f);
        float d0_35 = 1.0f + e0_35;
        float d1_35 = 1.0f + e1_35;
        float r0_35 = approx_rcp(d0_35);
        float r1_35 = approx_rcp(d1_35);
        float _fma_189;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_189) : "f"(-d0_35), "f"(r0_35), "f"(1.0f));
        float fma_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_5) : "f"(r0_35), "f"(_fma_189), "f"(r0_35));
        float _fma_190;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_190) : "f"(-d1_35), "f"(r1_35), "f"(1.0f));
        float fma_1_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_4) : "f"(r1_35), "f"(_fma_190), "f"(r1_35));
        float t0_4 = fma_5;
        float t1_4 = fma_1_4;
        band = band | ((fminf(t0_4, t1_4) == 0.0f) ? 32 : 0);
        float2 s_35 = make_float2(max_noftz(t0_4, 0.0f), max_noftz(t1_4, 0.0f));
        float2 act_35;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_35) : "l"(*(const unsigned long long*)&x_35), "l"(*(const unsigned long long*)&s_35));
        float2 _f32x2_mul_a_191 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_35, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_35), s_35), g_35);
        float2 mul_36;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_36) : "l"(*(const unsigned long long*)&_f32x2_mul_a_191), "l"(*(const unsigned long long*)&u_35));
        float2 mul_1_35;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_35) : "l"(*(const unsigned long long*)&act_35), "l"(*(const unsigned long long*)&g_35));
        __nv_bfloat162 bf16x2_38 = __float22bfloat162_rn(make_float2(mul_36.x, mul_36.y));
        __nv_bfloat162 bf16x2_1_37 = __float22bfloat162_rn(make_float2(mul_1_35.x, mul_1_35.y));
        dgate[5] = __as_u32(bf16x2_38);
        dup[5] = __as_u32(bf16x2_1_37);
        float2 x_36 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[6]));
        float2 u_36 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[6]));
        float2 g_36 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[6]));
        float e0_36 = approx_exp2(-x_36.x * 1.442695f);
        float e1_36 = approx_exp2(-x_36.y * 1.442695f);
        float d0_36 = 1.0f + e0_36;
        float d1_36 = 1.0f + e1_36;
        float r0_36 = approx_rcp(d0_36);
        float r1_36 = approx_rcp(d1_36);
        float _fma_192;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_192) : "f"(-d0_36), "f"(r0_36), "f"(1.0f));
        float fma_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_6) : "f"(r0_36), "f"(_fma_192), "f"(r0_36));
        float _fma_193;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_193) : "f"(-d1_36), "f"(r1_36), "f"(1.0f));
        float fma_1_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_5) : "f"(r1_36), "f"(_fma_193), "f"(r1_36));
        float t0_5 = fma_6;
        float t1_5 = fma_1_5;
        band = band | ((fminf(t0_5, t1_5) == 0.0f) ? 64 : 0);
        float2 s_36 = make_float2(max_noftz(t0_5, 0.0f), max_noftz(t1_5, 0.0f));
        float2 act_36;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_36) : "l"(*(const unsigned long long*)&x_36), "l"(*(const unsigned long long*)&s_36));
        float2 _f32x2_mul_a_194 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_36, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_36), s_36), g_36);
        float2 mul_37;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_37) : "l"(*(const unsigned long long*)&_f32x2_mul_a_194), "l"(*(const unsigned long long*)&u_36));
        float2 mul_1_36;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_36) : "l"(*(const unsigned long long*)&act_36), "l"(*(const unsigned long long*)&g_36));
        __nv_bfloat162 bf16x2_39 = __float22bfloat162_rn(make_float2(mul_37.x, mul_37.y));
        __nv_bfloat162 bf16x2_1_38 = __float22bfloat162_rn(make_float2(mul_1_36.x, mul_1_36.y));
        dgate[6] = __as_u32(bf16x2_39);
        dup[6] = __as_u32(bf16x2_1_38);
        float2 x_37 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[7]));
        float2 u_37 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[7]));
        float2 g_37 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[7]));
        float e0_37 = approx_exp2(-x_37.x * 1.442695f);
        float e1_37 = approx_exp2(-x_37.y * 1.442695f);
        float d0_37 = 1.0f + e0_37;
        float d1_37 = 1.0f + e1_37;
        float r0_37 = approx_rcp(d0_37);
        float r1_37 = approx_rcp(d1_37);
        float _fma_195;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_195) : "f"(-d0_37), "f"(r0_37), "f"(1.0f));
        float fma_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_7) : "f"(r0_37), "f"(_fma_195), "f"(r0_37));
        float _fma_196;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_196) : "f"(-d1_37), "f"(r1_37), "f"(1.0f));
        float fma_1_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_6) : "f"(r1_37), "f"(_fma_196), "f"(r1_37));
        float t0_6 = fma_7;
        float t1_6 = fma_1_6;
        band = band | ((fminf(t0_6, t1_6) == 0.0f) ? 128 : 0);
        float2 s_37 = make_float2(max_noftz(t0_6, 0.0f), max_noftz(t1_6, 0.0f));
        float2 act_37;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_37) : "l"(*(const unsigned long long*)&x_37), "l"(*(const unsigned long long*)&s_37));
        float2 _f32x2_mul_a_197 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_37, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_37), s_37), g_37);
        float2 mul_38;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_38) : "l"(*(const unsigned long long*)&_f32x2_mul_a_197), "l"(*(const unsigned long long*)&u_37));
        float2 mul_1_37;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_37) : "l"(*(const unsigned long long*)&act_37), "l"(*(const unsigned long long*)&g_37));
        __nv_bfloat162 bf16x2_40 = __float22bfloat162_rn(make_float2(mul_38.x, mul_38.y));
        __nv_bfloat162 bf16x2_1_39 = __float22bfloat162_rn(make_float2(mul_1_37.x, mul_1_37.y));
        dgate[7] = __as_u32(bf16x2_40);
        dup[7] = __as_u32(bf16x2_1_39);
        if (band != 0) {
            unsigned int _popc_0 = (unsigned int)__popc(band);
            #pragma unroll 1
            for (int _j = 0; _j < _popc_0; _j++) {
                int _find_first_set_198 = __ffs(band);
                int j = _find_first_set_198 - 1;
                band = band & band - 1;
                unsigned long long both = band_pair(gated_input, grad, gate_index + 2 * j, grow * K + col + 2 * j, K);
                unsigned int dgate_new = (unsigned int)both;
                unsigned int dup_new = (unsigned int)(both >> 32ull);
                dgate[0] = ((j == 0) ? dgate_new : dgate[0]);
                dup[0] = ((j == 0) ? dup_new : dup[0]);
                dgate[1] = ((j == 1) ? dgate_new : dgate[1]);
                dup[1] = ((j == 1) ? dup_new : dup[1]);
                dgate[2] = ((j == 2) ? dgate_new : dgate[2]);
                dup[2] = ((j == 2) ? dup_new : dup[2]);
                dgate[3] = ((j == 3) ? dgate_new : dgate[3]);
                dup[3] = ((j == 3) ? dup_new : dup[3]);
                dgate[4] = ((j == 4) ? dgate_new : dgate[4]);
                dup[4] = ((j == 4) ? dup_new : dup[4]);
                dgate[5] = ((j == 5) ? dgate_new : dgate[5]);
                dup[5] = ((j == 5) ? dup_new : dup[5]);
                dgate[6] = ((j == 6) ? dgate_new : dgate[6]);
                dup[6] = ((j == 6) ? dup_new : dup[6]);
                dgate[7] = ((j == 7) ? dgate_new : dgate[7]);
                dup[7] = ((j == 7) ? dup_new : dup[7]);
            }
        }
        int band_b = 0;
        float2 x_38 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[0]));
        float2 u_38 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[0]));
        float2 g_38 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[0]));
        float e0_38 = approx_exp2(-x_38.x * 1.442695f);
        float e1_38 = approx_exp2(-x_38.y * 1.442695f);
        float d0_38 = 1.0f + e0_38;
        float d1_38 = 1.0f + e1_38;
        float r0_38 = approx_rcp(d0_38);
        float r1_38 = approx_rcp(d1_38);
        float _fma_199;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_199) : "f"(-d0_38), "f"(r0_38), "f"(1.0f));
        float fma_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_8) : "f"(r0_38), "f"(_fma_199), "f"(r0_38));
        float _fma_200;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_200) : "f"(-d1_38), "f"(r1_38), "f"(1.0f));
        float fma_1_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_7) : "f"(r1_38), "f"(_fma_200), "f"(r1_38));
        float t0_7 = fma_8;
        float t1_7 = fma_1_7;
        band_b = band_b | ((fminf(t0_7, t1_7) == 0.0f) ? 1 : 0);
        float2 s_38 = make_float2(max_noftz(t0_7, 0.0f), max_noftz(t1_7, 0.0f));
        float2 act_38;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_38) : "l"(*(const unsigned long long*)&x_38), "l"(*(const unsigned long long*)&s_38));
        float2 _f32x2_mul_a_201 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_38, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_38), s_38), g_38);
        float2 mul_39;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_39) : "l"(*(const unsigned long long*)&_f32x2_mul_a_201), "l"(*(const unsigned long long*)&u_38));
        float2 mul_1_38;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_38) : "l"(*(const unsigned long long*)&act_38), "l"(*(const unsigned long long*)&g_38));
        __nv_bfloat162 bf16x2_41 = __float22bfloat162_rn(make_float2(mul_39.x, mul_39.y));
        __nv_bfloat162 bf16x2_1_40 = __float22bfloat162_rn(make_float2(mul_1_38.x, mul_1_38.y));
        dgate_b[0] = __as_u32(bf16x2_41);
        dup_b[0] = __as_u32(bf16x2_1_40);
        float2 x_39 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[1]));
        float2 u_39 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[1]));
        float2 g_39 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[1]));
        float e0_39 = approx_exp2(-x_39.x * 1.442695f);
        float e1_39 = approx_exp2(-x_39.y * 1.442695f);
        float d0_39 = 1.0f + e0_39;
        float d1_39 = 1.0f + e1_39;
        float r0_39 = approx_rcp(d0_39);
        float r1_39 = approx_rcp(d1_39);
        float _fma_202;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_202) : "f"(-d0_39), "f"(r0_39), "f"(1.0f));
        float fma_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_9) : "f"(r0_39), "f"(_fma_202), "f"(r0_39));
        float _fma_203;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_203) : "f"(-d1_39), "f"(r1_39), "f"(1.0f));
        float fma_1_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_8) : "f"(r1_39), "f"(_fma_203), "f"(r1_39));
        float t0_8 = fma_9;
        float t1_8 = fma_1_8;
        band_b = band_b | ((fminf(t0_8, t1_8) == 0.0f) ? 2 : 0);
        float2 s_39 = make_float2(max_noftz(t0_8, 0.0f), max_noftz(t1_8, 0.0f));
        float2 act_39;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_39) : "l"(*(const unsigned long long*)&x_39), "l"(*(const unsigned long long*)&s_39));
        float2 _f32x2_mul_a_204 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_39, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_39), s_39), g_39);
        float2 mul_40;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_40) : "l"(*(const unsigned long long*)&_f32x2_mul_a_204), "l"(*(const unsigned long long*)&u_39));
        float2 mul_1_39;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_39) : "l"(*(const unsigned long long*)&act_39), "l"(*(const unsigned long long*)&g_39));
        __nv_bfloat162 bf16x2_42 = __float22bfloat162_rn(make_float2(mul_40.x, mul_40.y));
        __nv_bfloat162 bf16x2_1_41 = __float22bfloat162_rn(make_float2(mul_1_39.x, mul_1_39.y));
        dgate_b[1] = __as_u32(bf16x2_42);
        dup_b[1] = __as_u32(bf16x2_1_41);
        float2 x_40 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[2]));
        float2 u_40 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[2]));
        float2 g_40 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[2]));
        float e0_40 = approx_exp2(-x_40.x * 1.442695f);
        float e1_40 = approx_exp2(-x_40.y * 1.442695f);
        float d0_40 = 1.0f + e0_40;
        float d1_40 = 1.0f + e1_40;
        float r0_40 = approx_rcp(d0_40);
        float r1_40 = approx_rcp(d1_40);
        float _fma_205;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_205) : "f"(-d0_40), "f"(r0_40), "f"(1.0f));
        float fma_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_10) : "f"(r0_40), "f"(_fma_205), "f"(r0_40));
        float _fma_206;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_206) : "f"(-d1_40), "f"(r1_40), "f"(1.0f));
        float fma_1_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_9) : "f"(r1_40), "f"(_fma_206), "f"(r1_40));
        float t0_9 = fma_10;
        float t1_9 = fma_1_9;
        band_b = band_b | ((fminf(t0_9, t1_9) == 0.0f) ? 4 : 0);
        float2 s_40 = make_float2(max_noftz(t0_9, 0.0f), max_noftz(t1_9, 0.0f));
        float2 act_40;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_40) : "l"(*(const unsigned long long*)&x_40), "l"(*(const unsigned long long*)&s_40));
        float2 _f32x2_mul_a_207 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_40, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_40), s_40), g_40);
        float2 mul_41;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_41) : "l"(*(const unsigned long long*)&_f32x2_mul_a_207), "l"(*(const unsigned long long*)&u_40));
        float2 mul_1_40;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_40) : "l"(*(const unsigned long long*)&act_40), "l"(*(const unsigned long long*)&g_40));
        __nv_bfloat162 bf16x2_43 = __float22bfloat162_rn(make_float2(mul_41.x, mul_41.y));
        __nv_bfloat162 bf16x2_1_42 = __float22bfloat162_rn(make_float2(mul_1_40.x, mul_1_40.y));
        dgate_b[2] = __as_u32(bf16x2_43);
        dup_b[2] = __as_u32(bf16x2_1_42);
        float2 x_41 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[3]));
        float2 u_41 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[3]));
        float2 g_41 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[3]));
        float e0_41 = approx_exp2(-x_41.x * 1.442695f);
        float e1_41 = approx_exp2(-x_41.y * 1.442695f);
        float d0_41 = 1.0f + e0_41;
        float d1_41 = 1.0f + e1_41;
        float r0_41 = approx_rcp(d0_41);
        float r1_41 = approx_rcp(d1_41);
        float _fma_208;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_208) : "f"(-d0_41), "f"(r0_41), "f"(1.0f));
        float fma_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_11) : "f"(r0_41), "f"(_fma_208), "f"(r0_41));
        float _fma_209;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_209) : "f"(-d1_41), "f"(r1_41), "f"(1.0f));
        float fma_1_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_10) : "f"(r1_41), "f"(_fma_209), "f"(r1_41));
        float t0_10 = fma_11;
        float t1_10 = fma_1_10;
        band_b = band_b | ((fminf(t0_10, t1_10) == 0.0f) ? 8 : 0);
        float2 s_41 = make_float2(max_noftz(t0_10, 0.0f), max_noftz(t1_10, 0.0f));
        float2 act_41;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_41) : "l"(*(const unsigned long long*)&x_41), "l"(*(const unsigned long long*)&s_41));
        float2 _f32x2_mul_a_210 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_41, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_41), s_41), g_41);
        float2 mul_42;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_42) : "l"(*(const unsigned long long*)&_f32x2_mul_a_210), "l"(*(const unsigned long long*)&u_41));
        float2 mul_1_41;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_41) : "l"(*(const unsigned long long*)&act_41), "l"(*(const unsigned long long*)&g_41));
        __nv_bfloat162 bf16x2_44 = __float22bfloat162_rn(make_float2(mul_42.x, mul_42.y));
        __nv_bfloat162 bf16x2_1_43 = __float22bfloat162_rn(make_float2(mul_1_41.x, mul_1_41.y));
        dgate_b[3] = __as_u32(bf16x2_44);
        dup_b[3] = __as_u32(bf16x2_1_43);
        float2 x_42 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[4]));
        float2 u_42 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[4]));
        float2 g_42 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[4]));
        float e0_42 = approx_exp2(-x_42.x * 1.442695f);
        float e1_42 = approx_exp2(-x_42.y * 1.442695f);
        float d0_42 = 1.0f + e0_42;
        float d1_42 = 1.0f + e1_42;
        float r0_42 = approx_rcp(d0_42);
        float r1_42 = approx_rcp(d1_42);
        float _fma_211;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_211) : "f"(-d0_42), "f"(r0_42), "f"(1.0f));
        float fma_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_12) : "f"(r0_42), "f"(_fma_211), "f"(r0_42));
        float _fma_212;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_212) : "f"(-d1_42), "f"(r1_42), "f"(1.0f));
        float fma_1_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_11) : "f"(r1_42), "f"(_fma_212), "f"(r1_42));
        float t0_11 = fma_12;
        float t1_11 = fma_1_11;
        band_b = band_b | ((fminf(t0_11, t1_11) == 0.0f) ? 16 : 0);
        float2 s_42 = make_float2(max_noftz(t0_11, 0.0f), max_noftz(t1_11, 0.0f));
        float2 act_42;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_42) : "l"(*(const unsigned long long*)&x_42), "l"(*(const unsigned long long*)&s_42));
        float2 _f32x2_mul_a_213 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_42, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_42), s_42), g_42);
        float2 mul_43;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_43) : "l"(*(const unsigned long long*)&_f32x2_mul_a_213), "l"(*(const unsigned long long*)&u_42));
        float2 mul_1_42;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_42) : "l"(*(const unsigned long long*)&act_42), "l"(*(const unsigned long long*)&g_42));
        __nv_bfloat162 bf16x2_45 = __float22bfloat162_rn(make_float2(mul_43.x, mul_43.y));
        __nv_bfloat162 bf16x2_1_44 = __float22bfloat162_rn(make_float2(mul_1_42.x, mul_1_42.y));
        dgate_b[4] = __as_u32(bf16x2_45);
        dup_b[4] = __as_u32(bf16x2_1_44);
        float2 x_43 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[5]));
        float2 u_43 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[5]));
        float2 g_43 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[5]));
        float e0_43 = approx_exp2(-x_43.x * 1.442695f);
        float e1_43 = approx_exp2(-x_43.y * 1.442695f);
        float d0_43 = 1.0f + e0_43;
        float d1_43 = 1.0f + e1_43;
        float r0_43 = approx_rcp(d0_43);
        float r1_43 = approx_rcp(d1_43);
        float _fma_214;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_214) : "f"(-d0_43), "f"(r0_43), "f"(1.0f));
        float fma_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_13) : "f"(r0_43), "f"(_fma_214), "f"(r0_43));
        float _fma_215;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_215) : "f"(-d1_43), "f"(r1_43), "f"(1.0f));
        float fma_1_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_12) : "f"(r1_43), "f"(_fma_215), "f"(r1_43));
        float t0_12 = fma_13;
        float t1_12 = fma_1_12;
        band_b = band_b | ((fminf(t0_12, t1_12) == 0.0f) ? 32 : 0);
        float2 s_43 = make_float2(max_noftz(t0_12, 0.0f), max_noftz(t1_12, 0.0f));
        float2 act_43;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_43) : "l"(*(const unsigned long long*)&x_43), "l"(*(const unsigned long long*)&s_43));
        float2 _f32x2_mul_a_216 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_43, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_43), s_43), g_43);
        float2 mul_44;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_44) : "l"(*(const unsigned long long*)&_f32x2_mul_a_216), "l"(*(const unsigned long long*)&u_43));
        float2 mul_1_43;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_43) : "l"(*(const unsigned long long*)&act_43), "l"(*(const unsigned long long*)&g_43));
        __nv_bfloat162 bf16x2_46 = __float22bfloat162_rn(make_float2(mul_44.x, mul_44.y));
        __nv_bfloat162 bf16x2_1_45 = __float22bfloat162_rn(make_float2(mul_1_43.x, mul_1_43.y));
        dgate_b[5] = __as_u32(bf16x2_46);
        dup_b[5] = __as_u32(bf16x2_1_45);
        float2 x_44 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[6]));
        float2 u_44 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[6]));
        float2 g_44 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[6]));
        float e0_44 = approx_exp2(-x_44.x * 1.442695f);
        float e1_44 = approx_exp2(-x_44.y * 1.442695f);
        float d0_44 = 1.0f + e0_44;
        float d1_44 = 1.0f + e1_44;
        float r0_44 = approx_rcp(d0_44);
        float r1_44 = approx_rcp(d1_44);
        float _fma_217;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_217) : "f"(-d0_44), "f"(r0_44), "f"(1.0f));
        float fma_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_14) : "f"(r0_44), "f"(_fma_217), "f"(r0_44));
        float _fma_218;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_218) : "f"(-d1_44), "f"(r1_44), "f"(1.0f));
        float fma_1_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_13) : "f"(r1_44), "f"(_fma_218), "f"(r1_44));
        float t0_13 = fma_14;
        float t1_13 = fma_1_13;
        band_b = band_b | ((fminf(t0_13, t1_13) == 0.0f) ? 64 : 0);
        float2 s_44 = make_float2(max_noftz(t0_13, 0.0f), max_noftz(t1_13, 0.0f));
        float2 act_44;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_44) : "l"(*(const unsigned long long*)&x_44), "l"(*(const unsigned long long*)&s_44));
        float2 _f32x2_mul_a_219 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_44, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_44), s_44), g_44);
        float2 mul_45;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_45) : "l"(*(const unsigned long long*)&_f32x2_mul_a_219), "l"(*(const unsigned long long*)&u_44));
        float2 mul_1_44;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_44) : "l"(*(const unsigned long long*)&act_44), "l"(*(const unsigned long long*)&g_44));
        __nv_bfloat162 bf16x2_47 = __float22bfloat162_rn(make_float2(mul_45.x, mul_45.y));
        __nv_bfloat162 bf16x2_1_46 = __float22bfloat162_rn(make_float2(mul_1_44.x, mul_1_44.y));
        dgate_b[6] = __as_u32(bf16x2_47);
        dup_b[6] = __as_u32(bf16x2_1_46);
        float2 x_45 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[7]));
        float2 u_45 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[7]));
        float2 g_45 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[7]));
        float e0_45 = approx_exp2(-x_45.x * 1.442695f);
        float e1_45 = approx_exp2(-x_45.y * 1.442695f);
        float d0_45 = 1.0f + e0_45;
        float d1_45 = 1.0f + e1_45;
        float r0_45 = approx_rcp(d0_45);
        float r1_45 = approx_rcp(d1_45);
        float _fma_220;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_220) : "f"(-d0_45), "f"(r0_45), "f"(1.0f));
        float fma_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_15) : "f"(r0_45), "f"(_fma_220), "f"(r0_45));
        float _fma_221;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_221) : "f"(-d1_45), "f"(r1_45), "f"(1.0f));
        float fma_1_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_14) : "f"(r1_45), "f"(_fma_221), "f"(r1_45));
        float t0_14 = fma_15;
        float t1_14 = fma_1_14;
        band_b = band_b | ((fminf(t0_14, t1_14) == 0.0f) ? 128 : 0);
        float2 s_45 = make_float2(max_noftz(t0_14, 0.0f), max_noftz(t1_14, 0.0f));
        float2 act_45;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_45) : "l"(*(const unsigned long long*)&x_45), "l"(*(const unsigned long long*)&s_45));
        float2 _f32x2_mul_a_222 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_45, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_45), s_45), g_45);
        float2 mul_46;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_46) : "l"(*(const unsigned long long*)&_f32x2_mul_a_222), "l"(*(const unsigned long long*)&u_45));
        float2 mul_1_45;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_45) : "l"(*(const unsigned long long*)&act_45), "l"(*(const unsigned long long*)&g_45));
        __nv_bfloat162 bf16x2_48 = __float22bfloat162_rn(make_float2(mul_46.x, mul_46.y));
        __nv_bfloat162 bf16x2_1_47 = __float22bfloat162_rn(make_float2(mul_1_45.x, mul_1_45.y));
        dgate_b[7] = __as_u32(bf16x2_48);
        dup_b[7] = __as_u32(bf16x2_1_47);
        if (band_b != 0) {
            unsigned int _popc_0_1 = (unsigned int)__popc(band_b);
            #pragma unroll 1
            for (int _j_1 = 0; _j_1 < _popc_0_1; _j_1++) {
                int _find_first_set_223 = __ffs(band_b);
                int j_1 = _find_first_set_223 - 1;
                band_b = band_b & band_b - 1;
                unsigned long long both_1 = band_pair(gated_input, grad, gate_index_0 + 2 * j_1, grow_b * K + col + 2 * j_1, K);
                unsigned int dgate_new_1 = (unsigned int)both_1;
                unsigned int dup_new_1 = (unsigned int)(both_1 >> 32ull);
                dgate_b[0] = ((j_1 == 0) ? dgate_new_1 : dgate_b[0]);
                dup_b[0] = ((j_1 == 0) ? dup_new_1 : dup_b[0]);
                dgate_b[1] = ((j_1 == 1) ? dgate_new_1 : dgate_b[1]);
                dup_b[1] = ((j_1 == 1) ? dup_new_1 : dup_b[1]);
                dgate_b[2] = ((j_1 == 2) ? dgate_new_1 : dgate_b[2]);
                dup_b[2] = ((j_1 == 2) ? dup_new_1 : dup_b[2]);
                dgate_b[3] = ((j_1 == 3) ? dgate_new_1 : dgate_b[3]);
                dup_b[3] = ((j_1 == 3) ? dup_new_1 : dup_b[3]);
                dgate_b[4] = ((j_1 == 4) ? dgate_new_1 : dgate_b[4]);
                dup_b[4] = ((j_1 == 4) ? dup_new_1 : dup_b[4]);
                dgate_b[5] = ((j_1 == 5) ? dgate_new_1 : dgate_b[5]);
                dup_b[5] = ((j_1 == 5) ? dup_new_1 : dup_b[5]);
                dgate_b[6] = ((j_1 == 6) ? dgate_new_1 : dgate_b[6]);
                dup_b[6] = ((j_1 == 6) ? dup_new_1 : dup_b[6]);
                dgate_b[7] = ((j_1 == 7) ? dgate_new_1 : dgate_b[7]);
                dup_b[7] = ((j_1 == 7) ? dup_new_1 : dup_b[7]);
            }
        }
        int band_c = 0;
        float2 x_46 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[0]));
        float2 u_46 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[0]));
        float2 g_46 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[0]));
        float e0_46 = approx_exp2(-x_46.x * 1.442695f);
        float e1_46 = approx_exp2(-x_46.y * 1.442695f);
        float d0_46 = 1.0f + e0_46;
        float d1_46 = 1.0f + e1_46;
        float r0_46 = approx_rcp(d0_46);
        float r1_46 = approx_rcp(d1_46);
        float _fma_224;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_224) : "f"(-d0_46), "f"(r0_46), "f"(1.0f));
        float fma_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_16) : "f"(r0_46), "f"(_fma_224), "f"(r0_46));
        float _fma_225;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_225) : "f"(-d1_46), "f"(r1_46), "f"(1.0f));
        float fma_1_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_15) : "f"(r1_46), "f"(_fma_225), "f"(r1_46));
        float t0_15 = fma_16;
        float t1_15 = fma_1_15;
        band_c = band_c | ((fminf(t0_15, t1_15) == 0.0f) ? 1 : 0);
        float2 s_46 = make_float2(max_noftz(t0_15, 0.0f), max_noftz(t1_15, 0.0f));
        float2 act_46;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_46) : "l"(*(const unsigned long long*)&x_46), "l"(*(const unsigned long long*)&s_46));
        float2 _f32x2_mul_a_226 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_46, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_46), s_46), g_46);
        float2 mul_47;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_47) : "l"(*(const unsigned long long*)&_f32x2_mul_a_226), "l"(*(const unsigned long long*)&u_46));
        float2 mul_1_46;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_46) : "l"(*(const unsigned long long*)&act_46), "l"(*(const unsigned long long*)&g_46));
        __nv_bfloat162 bf16x2_49 = __float22bfloat162_rn(make_float2(mul_47.x, mul_47.y));
        __nv_bfloat162 bf16x2_1_48 = __float22bfloat162_rn(make_float2(mul_1_46.x, mul_1_46.y));
        dgate_c[0] = __as_u32(bf16x2_49);
        dup_c[0] = __as_u32(bf16x2_1_48);
        float2 x_47 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[1]));
        float2 u_47 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[1]));
        float2 g_47 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[1]));
        float e0_47 = approx_exp2(-x_47.x * 1.442695f);
        float e1_47 = approx_exp2(-x_47.y * 1.442695f);
        float d0_47 = 1.0f + e0_47;
        float d1_47 = 1.0f + e1_47;
        float r0_47 = approx_rcp(d0_47);
        float r1_47 = approx_rcp(d1_47);
        float _fma_227;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_227) : "f"(-d0_47), "f"(r0_47), "f"(1.0f));
        float fma_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_17) : "f"(r0_47), "f"(_fma_227), "f"(r0_47));
        float _fma_228;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_228) : "f"(-d1_47), "f"(r1_47), "f"(1.0f));
        float fma_1_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_16) : "f"(r1_47), "f"(_fma_228), "f"(r1_47));
        float t0_16 = fma_17;
        float t1_16 = fma_1_16;
        band_c = band_c | ((fminf(t0_16, t1_16) == 0.0f) ? 2 : 0);
        float2 s_47 = make_float2(max_noftz(t0_16, 0.0f), max_noftz(t1_16, 0.0f));
        float2 act_47;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_47) : "l"(*(const unsigned long long*)&x_47), "l"(*(const unsigned long long*)&s_47));
        float2 _f32x2_mul_a_229 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_47, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_47), s_47), g_47);
        float2 mul_48;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_48) : "l"(*(const unsigned long long*)&_f32x2_mul_a_229), "l"(*(const unsigned long long*)&u_47));
        float2 mul_1_47;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_47) : "l"(*(const unsigned long long*)&act_47), "l"(*(const unsigned long long*)&g_47));
        __nv_bfloat162 bf16x2_50 = __float22bfloat162_rn(make_float2(mul_48.x, mul_48.y));
        __nv_bfloat162 bf16x2_1_49 = __float22bfloat162_rn(make_float2(mul_1_47.x, mul_1_47.y));
        dgate_c[1] = __as_u32(bf16x2_50);
        dup_c[1] = __as_u32(bf16x2_1_49);
        float2 x_48 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[2]));
        float2 u_48 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[2]));
        float2 g_48 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[2]));
        float e0_48 = approx_exp2(-x_48.x * 1.442695f);
        float e1_48 = approx_exp2(-x_48.y * 1.442695f);
        float d0_48 = 1.0f + e0_48;
        float d1_48 = 1.0f + e1_48;
        float r0_48 = approx_rcp(d0_48);
        float r1_48 = approx_rcp(d1_48);
        float _fma_230;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_230) : "f"(-d0_48), "f"(r0_48), "f"(1.0f));
        float fma_18;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_18) : "f"(r0_48), "f"(_fma_230), "f"(r0_48));
        float _fma_231;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_231) : "f"(-d1_48), "f"(r1_48), "f"(1.0f));
        float fma_1_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_17) : "f"(r1_48), "f"(_fma_231), "f"(r1_48));
        float t0_17 = fma_18;
        float t1_17 = fma_1_17;
        band_c = band_c | ((fminf(t0_17, t1_17) == 0.0f) ? 4 : 0);
        float2 s_48 = make_float2(max_noftz(t0_17, 0.0f), max_noftz(t1_17, 0.0f));
        float2 act_48;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_48) : "l"(*(const unsigned long long*)&x_48), "l"(*(const unsigned long long*)&s_48));
        float2 _f32x2_mul_a_232 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_48, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_48), s_48), g_48);
        float2 mul_49;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_49) : "l"(*(const unsigned long long*)&_f32x2_mul_a_232), "l"(*(const unsigned long long*)&u_48));
        float2 mul_1_48;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_48) : "l"(*(const unsigned long long*)&act_48), "l"(*(const unsigned long long*)&g_48));
        __nv_bfloat162 bf16x2_51 = __float22bfloat162_rn(make_float2(mul_49.x, mul_49.y));
        __nv_bfloat162 bf16x2_1_50 = __float22bfloat162_rn(make_float2(mul_1_48.x, mul_1_48.y));
        dgate_c[2] = __as_u32(bf16x2_51);
        dup_c[2] = __as_u32(bf16x2_1_50);
        float2 x_49 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[3]));
        float2 u_49 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[3]));
        float2 g_49 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[3]));
        float e0_49 = approx_exp2(-x_49.x * 1.442695f);
        float e1_49 = approx_exp2(-x_49.y * 1.442695f);
        float d0_49 = 1.0f + e0_49;
        float d1_49 = 1.0f + e1_49;
        float r0_49 = approx_rcp(d0_49);
        float r1_49 = approx_rcp(d1_49);
        float _fma_233;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_233) : "f"(-d0_49), "f"(r0_49), "f"(1.0f));
        float fma_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_19) : "f"(r0_49), "f"(_fma_233), "f"(r0_49));
        float _fma_234;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_234) : "f"(-d1_49), "f"(r1_49), "f"(1.0f));
        float fma_1_18;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_18) : "f"(r1_49), "f"(_fma_234), "f"(r1_49));
        float t0_18 = fma_19;
        float t1_18 = fma_1_18;
        band_c = band_c | ((fminf(t0_18, t1_18) == 0.0f) ? 8 : 0);
        float2 s_49 = make_float2(max_noftz(t0_18, 0.0f), max_noftz(t1_18, 0.0f));
        float2 act_49;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_49) : "l"(*(const unsigned long long*)&x_49), "l"(*(const unsigned long long*)&s_49));
        float2 _f32x2_mul_a_235 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_49, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_49), s_49), g_49);
        float2 mul_50;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_50) : "l"(*(const unsigned long long*)&_f32x2_mul_a_235), "l"(*(const unsigned long long*)&u_49));
        float2 mul_1_49;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_49) : "l"(*(const unsigned long long*)&act_49), "l"(*(const unsigned long long*)&g_49));
        __nv_bfloat162 bf16x2_52 = __float22bfloat162_rn(make_float2(mul_50.x, mul_50.y));
        __nv_bfloat162 bf16x2_1_51 = __float22bfloat162_rn(make_float2(mul_1_49.x, mul_1_49.y));
        dgate_c[3] = __as_u32(bf16x2_52);
        dup_c[3] = __as_u32(bf16x2_1_51);
        float2 x_50 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[4]));
        float2 u_50 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[4]));
        float2 g_50 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[4]));
        float e0_50 = approx_exp2(-x_50.x * 1.442695f);
        float e1_50 = approx_exp2(-x_50.y * 1.442695f);
        float d0_50 = 1.0f + e0_50;
        float d1_50 = 1.0f + e1_50;
        float r0_50 = approx_rcp(d0_50);
        float r1_50 = approx_rcp(d1_50);
        float _fma_236;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_236) : "f"(-d0_50), "f"(r0_50), "f"(1.0f));
        float fma_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_20) : "f"(r0_50), "f"(_fma_236), "f"(r0_50));
        float _fma_237;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_237) : "f"(-d1_50), "f"(r1_50), "f"(1.0f));
        float fma_1_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_19) : "f"(r1_50), "f"(_fma_237), "f"(r1_50));
        float t0_19 = fma_20;
        float t1_19 = fma_1_19;
        band_c = band_c | ((fminf(t0_19, t1_19) == 0.0f) ? 16 : 0);
        float2 s_50 = make_float2(max_noftz(t0_19, 0.0f), max_noftz(t1_19, 0.0f));
        float2 act_50;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_50) : "l"(*(const unsigned long long*)&x_50), "l"(*(const unsigned long long*)&s_50));
        float2 _f32x2_mul_a_238 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_50, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_50), s_50), g_50);
        float2 mul_51;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_51) : "l"(*(const unsigned long long*)&_f32x2_mul_a_238), "l"(*(const unsigned long long*)&u_50));
        float2 mul_1_50;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_50) : "l"(*(const unsigned long long*)&act_50), "l"(*(const unsigned long long*)&g_50));
        __nv_bfloat162 bf16x2_53 = __float22bfloat162_rn(make_float2(mul_51.x, mul_51.y));
        __nv_bfloat162 bf16x2_1_52 = __float22bfloat162_rn(make_float2(mul_1_50.x, mul_1_50.y));
        dgate_c[4] = __as_u32(bf16x2_53);
        dup_c[4] = __as_u32(bf16x2_1_52);
        float2 x_51 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[5]));
        float2 u_51 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[5]));
        float2 g_51 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[5]));
        float e0_51 = approx_exp2(-x_51.x * 1.442695f);
        float e1_51 = approx_exp2(-x_51.y * 1.442695f);
        float d0_51 = 1.0f + e0_51;
        float d1_51 = 1.0f + e1_51;
        float r0_51 = approx_rcp(d0_51);
        float r1_51 = approx_rcp(d1_51);
        float _fma_239;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_239) : "f"(-d0_51), "f"(r0_51), "f"(1.0f));
        float fma_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_21) : "f"(r0_51), "f"(_fma_239), "f"(r0_51));
        float _fma_240;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_240) : "f"(-d1_51), "f"(r1_51), "f"(1.0f));
        float fma_1_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_20) : "f"(r1_51), "f"(_fma_240), "f"(r1_51));
        float t0_20 = fma_21;
        float t1_20 = fma_1_20;
        band_c = band_c | ((fminf(t0_20, t1_20) == 0.0f) ? 32 : 0);
        float2 s_51 = make_float2(max_noftz(t0_20, 0.0f), max_noftz(t1_20, 0.0f));
        float2 act_51;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_51) : "l"(*(const unsigned long long*)&x_51), "l"(*(const unsigned long long*)&s_51));
        float2 _f32x2_mul_a_241 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_51, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_51), s_51), g_51);
        float2 mul_52;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_52) : "l"(*(const unsigned long long*)&_f32x2_mul_a_241), "l"(*(const unsigned long long*)&u_51));
        float2 mul_1_51;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_51) : "l"(*(const unsigned long long*)&act_51), "l"(*(const unsigned long long*)&g_51));
        __nv_bfloat162 bf16x2_54 = __float22bfloat162_rn(make_float2(mul_52.x, mul_52.y));
        __nv_bfloat162 bf16x2_1_53 = __float22bfloat162_rn(make_float2(mul_1_51.x, mul_1_51.y));
        dgate_c[5] = __as_u32(bf16x2_54);
        dup_c[5] = __as_u32(bf16x2_1_53);
        float2 x_52 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[6]));
        float2 u_52 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[6]));
        float2 g_52 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[6]));
        float e0_52 = approx_exp2(-x_52.x * 1.442695f);
        float e1_52 = approx_exp2(-x_52.y * 1.442695f);
        float d0_52 = 1.0f + e0_52;
        float d1_52 = 1.0f + e1_52;
        float r0_52 = approx_rcp(d0_52);
        float r1_52 = approx_rcp(d1_52);
        float _fma_242;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_242) : "f"(-d0_52), "f"(r0_52), "f"(1.0f));
        float fma_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_22) : "f"(r0_52), "f"(_fma_242), "f"(r0_52));
        float _fma_243;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_243) : "f"(-d1_52), "f"(r1_52), "f"(1.0f));
        float fma_1_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_21) : "f"(r1_52), "f"(_fma_243), "f"(r1_52));
        float t0_21 = fma_22;
        float t1_21 = fma_1_21;
        band_c = band_c | ((fminf(t0_21, t1_21) == 0.0f) ? 64 : 0);
        float2 s_52 = make_float2(max_noftz(t0_21, 0.0f), max_noftz(t1_21, 0.0f));
        float2 act_52;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_52) : "l"(*(const unsigned long long*)&x_52), "l"(*(const unsigned long long*)&s_52));
        float2 _f32x2_mul_a_244 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_52, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_52), s_52), g_52);
        float2 mul_53;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_53) : "l"(*(const unsigned long long*)&_f32x2_mul_a_244), "l"(*(const unsigned long long*)&u_52));
        float2 mul_1_52;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_52) : "l"(*(const unsigned long long*)&act_52), "l"(*(const unsigned long long*)&g_52));
        __nv_bfloat162 bf16x2_55 = __float22bfloat162_rn(make_float2(mul_53.x, mul_53.y));
        __nv_bfloat162 bf16x2_1_54 = __float22bfloat162_rn(make_float2(mul_1_52.x, mul_1_52.y));
        dgate_c[6] = __as_u32(bf16x2_55);
        dup_c[6] = __as_u32(bf16x2_1_54);
        float2 x_53 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_3[7]));
        float2 u_53 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[7]));
        float2 g_53 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_1[7]));
        float e0_53 = approx_exp2(-x_53.x * 1.442695f);
        float e1_53 = approx_exp2(-x_53.y * 1.442695f);
        float d0_53 = 1.0f + e0_53;
        float d1_53 = 1.0f + e1_53;
        float r0_53 = approx_rcp(d0_53);
        float r1_53 = approx_rcp(d1_53);
        float _fma_245;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_245) : "f"(-d0_53), "f"(r0_53), "f"(1.0f));
        float fma_23;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_23) : "f"(r0_53), "f"(_fma_245), "f"(r0_53));
        float _fma_246;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_246) : "f"(-d1_53), "f"(r1_53), "f"(1.0f));
        float fma_1_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_22) : "f"(r1_53), "f"(_fma_246), "f"(r1_53));
        float t0_22 = fma_23;
        float t1_22 = fma_1_22;
        band_c = band_c | ((fminf(t0_22, t1_22) == 0.0f) ? 128 : 0);
        float2 s_53 = make_float2(max_noftz(t0_22, 0.0f), max_noftz(t1_22, 0.0f));
        float2 act_53;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_53) : "l"(*(const unsigned long long*)&x_53), "l"(*(const unsigned long long*)&s_53));
        float2 _f32x2_mul_a_247 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_53, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_53), s_53), g_53);
        float2 mul_54;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_54) : "l"(*(const unsigned long long*)&_f32x2_mul_a_247), "l"(*(const unsigned long long*)&u_53));
        float2 mul_1_53;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_53) : "l"(*(const unsigned long long*)&act_53), "l"(*(const unsigned long long*)&g_53));
        __nv_bfloat162 bf16x2_56 = __float22bfloat162_rn(make_float2(mul_54.x, mul_54.y));
        __nv_bfloat162 bf16x2_1_55 = __float22bfloat162_rn(make_float2(mul_1_53.x, mul_1_53.y));
        dgate_c[7] = __as_u32(bf16x2_56);
        dup_c[7] = __as_u32(bf16x2_1_55);
        if (band_c != 0) {
            unsigned int _popc_0_2 = (unsigned int)__popc(band_c);
            #pragma unroll 1
            for (int _j_2 = 0; _j_2 < _popc_0_2; _j_2++) {
                int _find_first_set_248 = __ffs(band_c);
                int j_2 = _find_first_set_248 - 1;
                band_c = band_c & band_c - 1;
                unsigned long long both_2 = band_pair(gated_input, grad, gate_index_1 + 2 * j_2, grow_c * K + col + 2 * j_2, K);
                unsigned int dgate_new_2 = (unsigned int)both_2;
                unsigned int dup_new_2 = (unsigned int)(both_2 >> 32ull);
                dgate_c[0] = ((j_2 == 0) ? dgate_new_2 : dgate_c[0]);
                dup_c[0] = ((j_2 == 0) ? dup_new_2 : dup_c[0]);
                dgate_c[1] = ((j_2 == 1) ? dgate_new_2 : dgate_c[1]);
                dup_c[1] = ((j_2 == 1) ? dup_new_2 : dup_c[1]);
                dgate_c[2] = ((j_2 == 2) ? dgate_new_2 : dgate_c[2]);
                dup_c[2] = ((j_2 == 2) ? dup_new_2 : dup_c[2]);
                dgate_c[3] = ((j_2 == 3) ? dgate_new_2 : dgate_c[3]);
                dup_c[3] = ((j_2 == 3) ? dup_new_2 : dup_c[3]);
                dgate_c[4] = ((j_2 == 4) ? dgate_new_2 : dgate_c[4]);
                dup_c[4] = ((j_2 == 4) ? dup_new_2 : dup_c[4]);
                dgate_c[5] = ((j_2 == 5) ? dgate_new_2 : dgate_c[5]);
                dup_c[5] = ((j_2 == 5) ? dup_new_2 : dup_c[5]);
                dgate_c[6] = ((j_2 == 6) ? dgate_new_2 : dgate_c[6]);
                dup_c[6] = ((j_2 == 6) ? dup_new_2 : dup_c[6]);
                dgate_c[7] = ((j_2 == 7) ? dgate_new_2 : dgate_c[7]);
                dup_c[7] = ((j_2 == 7) ? dup_new_2 : dup_c[7]);
            }
        }
        int band_d = 0;
        float2 x_54 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_4[0]));
        float2 u_54 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_2[0]));
        float2 g_54 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_2[0]));
        float e0_54 = approx_exp2(-x_54.x * 1.442695f);
        float e1_54 = approx_exp2(-x_54.y * 1.442695f);
        float d0_54 = 1.0f + e0_54;
        float d1_54 = 1.0f + e1_54;
        float r0_54 = approx_rcp(d0_54);
        float r1_54 = approx_rcp(d1_54);
        float _fma_249;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_249) : "f"(-d0_54), "f"(r0_54), "f"(1.0f));
        float fma_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_24) : "f"(r0_54), "f"(_fma_249), "f"(r0_54));
        float _fma_250;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_250) : "f"(-d1_54), "f"(r1_54), "f"(1.0f));
        float fma_1_23;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_23) : "f"(r1_54), "f"(_fma_250), "f"(r1_54));
        float t0_23 = fma_24;
        float t1_23 = fma_1_23;
        band_d = band_d | ((fminf(t0_23, t1_23) == 0.0f) ? 1 : 0);
        float2 s_54 = make_float2(max_noftz(t0_23, 0.0f), max_noftz(t1_23, 0.0f));
        float2 act_54;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_54) : "l"(*(const unsigned long long*)&x_54), "l"(*(const unsigned long long*)&s_54));
        float2 _f32x2_mul_a_251 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_54, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_54), s_54), g_54);
        float2 mul_55;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_55) : "l"(*(const unsigned long long*)&_f32x2_mul_a_251), "l"(*(const unsigned long long*)&u_54));
        float2 mul_1_54;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_54) : "l"(*(const unsigned long long*)&act_54), "l"(*(const unsigned long long*)&g_54));
        __nv_bfloat162 bf16x2_57 = __float22bfloat162_rn(make_float2(mul_55.x, mul_55.y));
        __nv_bfloat162 bf16x2_1_56 = __float22bfloat162_rn(make_float2(mul_1_54.x, mul_1_54.y));
        dgate_d[0] = __as_u32(bf16x2_57);
        dup_d[0] = __as_u32(bf16x2_1_56);
        float2 x_55 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_4[1]));
        float2 u_55 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_2[1]));
        float2 g_55 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_2[1]));
        float e0_55 = approx_exp2(-x_55.x * 1.442695f);
        float e1_55 = approx_exp2(-x_55.y * 1.442695f);
        float d0_55 = 1.0f + e0_55;
        float d1_55 = 1.0f + e1_55;
        float r0_55 = approx_rcp(d0_55);
        float r1_55 = approx_rcp(d1_55);
        float _fma_252;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_252) : "f"(-d0_55), "f"(r0_55), "f"(1.0f));
        float fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_25) : "f"(r0_55), "f"(_fma_252), "f"(r0_55));
        float _fma_253;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_253) : "f"(-d1_55), "f"(r1_55), "f"(1.0f));
        float fma_1_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_24) : "f"(r1_55), "f"(_fma_253), "f"(r1_55));
        float t0_24 = fma_25;
        float t1_24 = fma_1_24;
        band_d = band_d | ((fminf(t0_24, t1_24) == 0.0f) ? 2 : 0);
        float2 s_55 = make_float2(max_noftz(t0_24, 0.0f), max_noftz(t1_24, 0.0f));
        float2 act_55;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_55) : "l"(*(const unsigned long long*)&x_55), "l"(*(const unsigned long long*)&s_55));
        float2 _f32x2_mul_a_254 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_55, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_55), s_55), g_55);
        float2 mul_56;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_56) : "l"(*(const unsigned long long*)&_f32x2_mul_a_254), "l"(*(const unsigned long long*)&u_55));
        float2 mul_1_55;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_55) : "l"(*(const unsigned long long*)&act_55), "l"(*(const unsigned long long*)&g_55));
        __nv_bfloat162 bf16x2_58 = __float22bfloat162_rn(make_float2(mul_56.x, mul_56.y));
        __nv_bfloat162 bf16x2_1_57 = __float22bfloat162_rn(make_float2(mul_1_55.x, mul_1_55.y));
        dgate_d[1] = __as_u32(bf16x2_58);
        dup_d[1] = __as_u32(bf16x2_1_57);
        float2 x_56 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_4[2]));
        float2 u_56 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_2[2]));
        float2 g_56 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_2[2]));
        float e0_56 = approx_exp2(-x_56.x * 1.442695f);
        float e1_56 = approx_exp2(-x_56.y * 1.442695f);
        float d0_56 = 1.0f + e0_56;
        float d1_56 = 1.0f + e1_56;
        float r0_56 = approx_rcp(d0_56);
        float r1_56 = approx_rcp(d1_56);
        float _fma_255;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_255) : "f"(-d0_56), "f"(r0_56), "f"(1.0f));
        float fma_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_26) : "f"(r0_56), "f"(_fma_255), "f"(r0_56));
        float _fma_256;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_256) : "f"(-d1_56), "f"(r1_56), "f"(1.0f));
        float fma_1_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_25) : "f"(r1_56), "f"(_fma_256), "f"(r1_56));
        float t0_25 = fma_26;
        float t1_25 = fma_1_25;
        band_d = band_d | ((fminf(t0_25, t1_25) == 0.0f) ? 4 : 0);
        float2 s_56 = make_float2(max_noftz(t0_25, 0.0f), max_noftz(t1_25, 0.0f));
        float2 act_56;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_56) : "l"(*(const unsigned long long*)&x_56), "l"(*(const unsigned long long*)&s_56));
        float2 _f32x2_mul_a_257 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_56, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_56), s_56), g_56);
        float2 mul_57;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_57) : "l"(*(const unsigned long long*)&_f32x2_mul_a_257), "l"(*(const unsigned long long*)&u_56));
        float2 mul_1_56;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_56) : "l"(*(const unsigned long long*)&act_56), "l"(*(const unsigned long long*)&g_56));
        __nv_bfloat162 bf16x2_59 = __float22bfloat162_rn(make_float2(mul_57.x, mul_57.y));
        __nv_bfloat162 bf16x2_1_58 = __float22bfloat162_rn(make_float2(mul_1_56.x, mul_1_56.y));
        dgate_d[2] = __as_u32(bf16x2_59);
        dup_d[2] = __as_u32(bf16x2_1_58);
        float2 x_57 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_4[3]));
        float2 u_57 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_2[3]));
        float2 g_57 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_2[3]));
        float e0_57 = approx_exp2(-x_57.x * 1.442695f);
        float e1_57 = approx_exp2(-x_57.y * 1.442695f);
        float d0_57 = 1.0f + e0_57;
        float d1_57 = 1.0f + e1_57;
        float r0_57 = approx_rcp(d0_57);
        float r1_57 = approx_rcp(d1_57);
        float _fma_258;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_258) : "f"(-d0_57), "f"(r0_57), "f"(1.0f));
        float fma_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_27) : "f"(r0_57), "f"(_fma_258), "f"(r0_57));
        float _fma_259;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_259) : "f"(-d1_57), "f"(r1_57), "f"(1.0f));
        float fma_1_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_26) : "f"(r1_57), "f"(_fma_259), "f"(r1_57));
        float t0_26 = fma_27;
        float t1_26 = fma_1_26;
        band_d = band_d | ((fminf(t0_26, t1_26) == 0.0f) ? 8 : 0);
        float2 s_57 = make_float2(max_noftz(t0_26, 0.0f), max_noftz(t1_26, 0.0f));
        float2 act_57;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_57) : "l"(*(const unsigned long long*)&x_57), "l"(*(const unsigned long long*)&s_57));
        float2 _f32x2_mul_a_260 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_57, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_57), s_57), g_57);
        float2 mul_58;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_58) : "l"(*(const unsigned long long*)&_f32x2_mul_a_260), "l"(*(const unsigned long long*)&u_57));
        float2 mul_1_57;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_57) : "l"(*(const unsigned long long*)&act_57), "l"(*(const unsigned long long*)&g_57));
        __nv_bfloat162 bf16x2_60 = __float22bfloat162_rn(make_float2(mul_58.x, mul_58.y));
        __nv_bfloat162 bf16x2_1_59 = __float22bfloat162_rn(make_float2(mul_1_57.x, mul_1_57.y));
        dgate_d[3] = __as_u32(bf16x2_60);
        dup_d[3] = __as_u32(bf16x2_1_59);
        float2 x_58 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_4[4]));
        float2 u_58 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_2[4]));
        float2 g_58 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_2[4]));
        float e0_58 = approx_exp2(-x_58.x * 1.442695f);
        float e1_58 = approx_exp2(-x_58.y * 1.442695f);
        float d0_58 = 1.0f + e0_58;
        float d1_58 = 1.0f + e1_58;
        float r0_58 = approx_rcp(d0_58);
        float r1_58 = approx_rcp(d1_58);
        float _fma_261;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_261) : "f"(-d0_58), "f"(r0_58), "f"(1.0f));
        float fma_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_28) : "f"(r0_58), "f"(_fma_261), "f"(r0_58));
        float _fma_262;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_262) : "f"(-d1_58), "f"(r1_58), "f"(1.0f));
        float fma_1_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_27) : "f"(r1_58), "f"(_fma_262), "f"(r1_58));
        float t0_27 = fma_28;
        float t1_27 = fma_1_27;
        band_d = band_d | ((fminf(t0_27, t1_27) == 0.0f) ? 16 : 0);
        float2 s_58 = make_float2(max_noftz(t0_27, 0.0f), max_noftz(t1_27, 0.0f));
        float2 act_58;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_58) : "l"(*(const unsigned long long*)&x_58), "l"(*(const unsigned long long*)&s_58));
        float2 _f32x2_mul_a_263 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_58, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_58), s_58), g_58);
        float2 mul_59;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_59) : "l"(*(const unsigned long long*)&_f32x2_mul_a_263), "l"(*(const unsigned long long*)&u_58));
        float2 mul_1_58;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_58) : "l"(*(const unsigned long long*)&act_58), "l"(*(const unsigned long long*)&g_58));
        __nv_bfloat162 bf16x2_61 = __float22bfloat162_rn(make_float2(mul_59.x, mul_59.y));
        __nv_bfloat162 bf16x2_1_60 = __float22bfloat162_rn(make_float2(mul_1_58.x, mul_1_58.y));
        dgate_d[4] = __as_u32(bf16x2_61);
        dup_d[4] = __as_u32(bf16x2_1_60);
        float2 x_59 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_4[5]));
        float2 u_59 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_2[5]));
        float2 g_59 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_2[5]));
        float e0_59 = approx_exp2(-x_59.x * 1.442695f);
        float e1_59 = approx_exp2(-x_59.y * 1.442695f);
        float d0_59 = 1.0f + e0_59;
        float d1_59 = 1.0f + e1_59;
        float r0_59 = approx_rcp(d0_59);
        float r1_59 = approx_rcp(d1_59);
        float _fma_264;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_264) : "f"(-d0_59), "f"(r0_59), "f"(1.0f));
        float fma_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_29) : "f"(r0_59), "f"(_fma_264), "f"(r0_59));
        float _fma_265;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_265) : "f"(-d1_59), "f"(r1_59), "f"(1.0f));
        float fma_1_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_28) : "f"(r1_59), "f"(_fma_265), "f"(r1_59));
        float t0_28 = fma_29;
        float t1_28 = fma_1_28;
        band_d = band_d | ((fminf(t0_28, t1_28) == 0.0f) ? 32 : 0);
        float2 s_59 = make_float2(max_noftz(t0_28, 0.0f), max_noftz(t1_28, 0.0f));
        float2 act_59;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_59) : "l"(*(const unsigned long long*)&x_59), "l"(*(const unsigned long long*)&s_59));
        float2 _f32x2_mul_a_266 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_59, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_59), s_59), g_59);
        float2 mul_60;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_60) : "l"(*(const unsigned long long*)&_f32x2_mul_a_266), "l"(*(const unsigned long long*)&u_59));
        float2 mul_1_59;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_59) : "l"(*(const unsigned long long*)&act_59), "l"(*(const unsigned long long*)&g_59));
        __nv_bfloat162 bf16x2_62 = __float22bfloat162_rn(make_float2(mul_60.x, mul_60.y));
        __nv_bfloat162 bf16x2_1_61 = __float22bfloat162_rn(make_float2(mul_1_59.x, mul_1_59.y));
        dgate_d[5] = __as_u32(bf16x2_62);
        dup_d[5] = __as_u32(bf16x2_1_61);
        float2 x_60 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_4[6]));
        float2 u_60 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_2[6]));
        float2 g_60 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_2[6]));
        float e0_60 = approx_exp2(-x_60.x * 1.442695f);
        float e1_60 = approx_exp2(-x_60.y * 1.442695f);
        float d0_60 = 1.0f + e0_60;
        float d1_60 = 1.0f + e1_60;
        float r0_60 = approx_rcp(d0_60);
        float r1_60 = approx_rcp(d1_60);
        float _fma_267;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_267) : "f"(-d0_60), "f"(r0_60), "f"(1.0f));
        float fma_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_30) : "f"(r0_60), "f"(_fma_267), "f"(r0_60));
        float _fma_268;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_268) : "f"(-d1_60), "f"(r1_60), "f"(1.0f));
        float fma_1_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_29) : "f"(r1_60), "f"(_fma_268), "f"(r1_60));
        float t0_29 = fma_30;
        float t1_29 = fma_1_29;
        band_d = band_d | ((fminf(t0_29, t1_29) == 0.0f) ? 64 : 0);
        float2 s_60 = make_float2(max_noftz(t0_29, 0.0f), max_noftz(t1_29, 0.0f));
        float2 act_60;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_60) : "l"(*(const unsigned long long*)&x_60), "l"(*(const unsigned long long*)&s_60));
        float2 _f32x2_mul_a_269 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_60, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_60), s_60), g_60);
        float2 mul_61;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_61) : "l"(*(const unsigned long long*)&_f32x2_mul_a_269), "l"(*(const unsigned long long*)&u_60));
        float2 mul_1_60;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_60) : "l"(*(const unsigned long long*)&act_60), "l"(*(const unsigned long long*)&g_60));
        __nv_bfloat162 bf16x2_63 = __float22bfloat162_rn(make_float2(mul_61.x, mul_61.y));
        __nv_bfloat162 bf16x2_1_62 = __float22bfloat162_rn(make_float2(mul_1_60.x, mul_1_60.y));
        dgate_d[6] = __as_u32(bf16x2_63);
        dup_d[6] = __as_u32(bf16x2_1_62);
        float2 x_61 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_4[7]));
        float2 u_61 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_2[7]));
        float2 g_61 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_2[7]));
        float e0_61 = approx_exp2(-x_61.x * 1.442695f);
        float e1_61 = approx_exp2(-x_61.y * 1.442695f);
        float d0_61 = 1.0f + e0_61;
        float d1_61 = 1.0f + e1_61;
        float r0_61 = approx_rcp(d0_61);
        float r1_61 = approx_rcp(d1_61);
        float _fma_270;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_270) : "f"(-d0_61), "f"(r0_61), "f"(1.0f));
        float fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_31) : "f"(r0_61), "f"(_fma_270), "f"(r0_61));
        float _fma_271;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_271) : "f"(-d1_61), "f"(r1_61), "f"(1.0f));
        float fma_1_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_30) : "f"(r1_61), "f"(_fma_271), "f"(r1_61));
        float t0_30 = fma_31;
        float t1_30 = fma_1_30;
        band_d = band_d | ((fminf(t0_30, t1_30) == 0.0f) ? 128 : 0);
        float2 s_61 = make_float2(max_noftz(t0_30, 0.0f), max_noftz(t1_30, 0.0f));
        float2 act_61;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_61) : "l"(*(const unsigned long long*)&x_61), "l"(*(const unsigned long long*)&s_61));
        float2 _f32x2_mul_a_272 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_61, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_61), s_61), g_61);
        float2 mul_62;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_62) : "l"(*(const unsigned long long*)&_f32x2_mul_a_272), "l"(*(const unsigned long long*)&u_61));
        float2 mul_1_61;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_61) : "l"(*(const unsigned long long*)&act_61), "l"(*(const unsigned long long*)&g_61));
        __nv_bfloat162 bf16x2_64 = __float22bfloat162_rn(make_float2(mul_62.x, mul_62.y));
        __nv_bfloat162 bf16x2_1_63 = __float22bfloat162_rn(make_float2(mul_1_61.x, mul_1_61.y));
        dgate_d[7] = __as_u32(bf16x2_64);
        dup_d[7] = __as_u32(bf16x2_1_63);
        if (band_d != 0) {
            unsigned int _popc_0_3 = (unsigned int)__popc(band_d);
            #pragma unroll 1
            for (int _j_3 = 0; _j_3 < _popc_0_3; _j_3++) {
                int _find_first_set_273 = __ffs(band_d);
                int j_3 = _find_first_set_273 - 1;
                band_d = band_d & band_d - 1;
                unsigned long long both_3 = band_pair(gated_input, grad, gate_index_2 + 2 * j_3, grow_d * K + col + 2 * j_3, K);
                unsigned int dgate_new_3 = (unsigned int)both_3;
                unsigned int dup_new_3 = (unsigned int)(both_3 >> 32ull);
                dgate_d[0] = ((j_3 == 0) ? dgate_new_3 : dgate_d[0]);
                dup_d[0] = ((j_3 == 0) ? dup_new_3 : dup_d[0]);
                dgate_d[1] = ((j_3 == 1) ? dgate_new_3 : dgate_d[1]);
                dup_d[1] = ((j_3 == 1) ? dup_new_3 : dup_d[1]);
                dgate_d[2] = ((j_3 == 2) ? dgate_new_3 : dgate_d[2]);
                dup_d[2] = ((j_3 == 2) ? dup_new_3 : dup_d[2]);
                dgate_d[3] = ((j_3 == 3) ? dgate_new_3 : dgate_d[3]);
                dup_d[3] = ((j_3 == 3) ? dup_new_3 : dup_d[3]);
                dgate_d[4] = ((j_3 == 4) ? dgate_new_3 : dgate_d[4]);
                dup_d[4] = ((j_3 == 4) ? dup_new_3 : dup_d[4]);
                dgate_d[5] = ((j_3 == 5) ? dgate_new_3 : dgate_d[5]);
                dup_d[5] = ((j_3 == 5) ? dup_new_3 : dup_d[5]);
                dgate_d[6] = ((j_3 == 6) ? dgate_new_3 : dgate_d[6]);
                dup_d[6] = ((j_3 == 6) ? dup_new_3 : dup_d[6]);
                dgate_d[7] = ((j_3 == 7) ? dgate_new_3 : dgate_d[7]);
                dup_d[7] = ((j_3 == 7) ? dup_new_3 : dup_d[7]);
            }
        }
    }
    unsigned int swizzle = (unsigned int)(row >> 4 & 6 | row & 1);
    unsigned int row_base = tiles_addr + (unsigned int)(row * 128);
    unsigned int chunk = (unsigned int)((tid & 3) * 2);
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + ((chunk ^ swizzle) << 4)), "r"(dgate[0]), "r"(dgate[1]), "r"(dgate[2]), "r"(dgate[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + ((chunk + 1 ^ swizzle) << 4)), "r"(dgate[4]), "r"(dgate[5]), "r"(dgate[6]), "r"(dgate[7]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + (16384 + ((chunk ^ swizzle) << 4))), "r"(dup[0]), "r"(dup[1]), "r"(dup[2]), "r"(dup[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + (16384 + ((chunk + 1 ^ swizzle) << 4))), "r"(dup[4]), "r"(dup[5]), "r"(dup[6]), "r"(dup[7]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + 4096 + ((chunk ^ (swizzle ^ 2)) << 4)), "r"(dgate_b[0]), "r"(dgate_b[1]), "r"(dgate_b[2]), "r"(dgate_b[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + 4096 + ((chunk + 1 ^ (swizzle ^ 2)) << 4)), "r"(dgate_b[4]), "r"(dgate_b[5]), "r"(dgate_b[6]), "r"(dgate_b[7]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + 4096 + (16384 + ((chunk ^ (swizzle ^ 2)) << 4))), "r"(dup_b[0]), "r"(dup_b[1]), "r"(dup_b[2]), "r"(dup_b[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + 4096 + (16384 + ((chunk + 1 ^ (swizzle ^ 2)) << 4))), "r"(dup_b[4]), "r"(dup_b[5]), "r"(dup_b[6]), "r"(dup_b[7]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + 8192 + ((chunk ^ (swizzle ^ 4)) << 4)), "r"(dgate_c[0]), "r"(dgate_c[1]), "r"(dgate_c[2]), "r"(dgate_c[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + 8192 + ((chunk + 1 ^ (swizzle ^ 4)) << 4)), "r"(dgate_c[4]), "r"(dgate_c[5]), "r"(dgate_c[6]), "r"(dgate_c[7]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + 8192 + (16384 + ((chunk ^ (swizzle ^ 4)) << 4))), "r"(dup_c[0]), "r"(dup_c[1]), "r"(dup_c[2]), "r"(dup_c[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + 8192 + (16384 + ((chunk + 1 ^ (swizzle ^ 4)) << 4))), "r"(dup_c[4]), "r"(dup_c[5]), "r"(dup_c[6]), "r"(dup_c[7]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + 12288 + ((chunk ^ (swizzle ^ 6)) << 4)), "r"(dgate_d[0]), "r"(dgate_d[1]), "r"(dgate_d[2]), "r"(dgate_d[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + 12288 + ((chunk + 1 ^ (swizzle ^ 6)) << 4)), "r"(dgate_d[4]), "r"(dgate_d[5]), "r"(dgate_d[6]), "r"(dgate_d[7]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + 12288 + (16384 + ((chunk ^ (swizzle ^ 6)) << 4))), "r"(dup_d[0]), "r"(dup_d[1]), "r"(dup_d[2]), "r"(dup_d[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + 12288 + (16384 + ((chunk + 1 ^ (swizzle ^ 6)) << 4))), "r"(dup_d[4]), "r"(dup_d[5]), "r"(dup_d[6]), "r"(dup_d[7]) : "memory");
    __syncthreads();
    int lane_1 = tid & 31;
    int warp_1 = tid >> 5;
    int k8 = lane_1 & 7;
    int srow = (k8 >> 1) * 32 + (lane_1 >> 3) * 2 + (k8 & 1);
    unsigned int sswizzle = (unsigned int)(srow >> 4 & 6 | srow & 1);
    unsigned int cols[32];
    unsigned int strip_chunk0_1_0 = (unsigned int)(warp_1 * 2);
    unsigned int strip_src0_1_0 = tiles_addr + (unsigned int)(srow * 128) + ((strip_chunk0_1_0 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[0]), "=r"(cols[1]), "=r"(cols[2]), "=r"(cols[3])
        : "r"(strip_src0_1_0)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[4]), "=r"(cols[5]), "=r"(cols[6]), "=r"(cols[7])
        : "r"(strip_src0_1_0 + 1024)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[8]), "=r"(cols[9]), "=r"(cols[10]), "=r"(cols[11])
        : "r"(strip_src0_1_0 + 2048)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[12]), "=r"(cols[13]), "=r"(cols[14]), "=r"(cols[15])
        : "r"(strip_src0_1_0 + 3072)
        : "memory");
    unsigned int strip_chunk0_1_1 = (unsigned int)(warp_1 * 2 + 1);
    unsigned int strip_src0_1_1 = tiles_addr + (unsigned int)(srow * 128) + ((strip_chunk0_1_1 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[16]), "=r"(cols[17]), "=r"(cols[18]), "=r"(cols[19])
        : "r"(strip_src0_1_1)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[20]), "=r"(cols[21]), "=r"(cols[22]), "=r"(cols[23])
        : "r"(strip_src0_1_1 + 1024)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[24]), "=r"(cols[25]), "=r"(cols[26]), "=r"(cols[27])
        : "r"(strip_src0_1_1 + 2048)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[28]), "=r"(cols[29]), "=r"(cols[30]), "=r"(cols[31])
        : "r"(strip_src0_1_1 + 3072)
        : "memory");
    __nv_bfloat162 amax_a = __as_bf16x2(0);
    __nv_bfloat162 amax_b = __as_bf16x2(0);
    __nv_bfloat162 _hxa_275 = amax_a;
    __nv_bfloat162 _hxb_275 = __as_bf16x2(cols[0]);
    unsigned int _hxr_275;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_275) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_275)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_275)));
    __nv_bfloat162 _max_274 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_275);
    amax_a = _max_274;
    __nv_bfloat162 _hxa_277 = amax_b;
    __nv_bfloat162 _hxb_277 = __as_bf16x2(cols[16]);
    unsigned int _hxr_277;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_277) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_277)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_277)));
    __nv_bfloat162 _max_276 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_277);
    amax_b = _max_276;
    __nv_bfloat162 _hxa_279 = amax_a;
    __nv_bfloat162 _hxb_279 = __as_bf16x2(cols[1]);
    unsigned int _hxr_279;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_279) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_279)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_279)));
    __nv_bfloat162 _max_278 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_279);
    amax_a = _max_278;
    __nv_bfloat162 _hxa_281 = amax_b;
    __nv_bfloat162 _hxb_281 = __as_bf16x2(cols[17]);
    unsigned int _hxr_281;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_281) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_281)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_281)));
    __nv_bfloat162 _max_280 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_281);
    amax_b = _max_280;
    __nv_bfloat162 _hxa_283 = amax_a;
    __nv_bfloat162 _hxb_283 = __as_bf16x2(cols[2]);
    unsigned int _hxr_283;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_283) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_283)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_283)));
    __nv_bfloat162 _max_282 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_283);
    amax_a = _max_282;
    __nv_bfloat162 _hxa_285 = amax_b;
    __nv_bfloat162 _hxb_285 = __as_bf16x2(cols[18]);
    unsigned int _hxr_285;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_285) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_285)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_285)));
    __nv_bfloat162 _max_284 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_285);
    amax_b = _max_284;
    __nv_bfloat162 _hxa_287 = amax_a;
    __nv_bfloat162 _hxb_287 = __as_bf16x2(cols[3]);
    unsigned int _hxr_287;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_287) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_287)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_287)));
    __nv_bfloat162 _max_286 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_287);
    amax_a = _max_286;
    __nv_bfloat162 _hxa_289 = amax_b;
    __nv_bfloat162 _hxb_289 = __as_bf16x2(cols[19]);
    unsigned int _hxr_289;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_289) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_289)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_289)));
    __nv_bfloat162 _max_288 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_289);
    amax_b = _max_288;
    __nv_bfloat162 _hxa_291 = amax_a;
    __nv_bfloat162 _hxb_291 = __as_bf16x2(cols[4]);
    unsigned int _hxr_291;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_291) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_291)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_291)));
    __nv_bfloat162 _max_290 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_291);
    amax_a = _max_290;
    __nv_bfloat162 _hxa_293 = amax_b;
    __nv_bfloat162 _hxb_293 = __as_bf16x2(cols[20]);
    unsigned int _hxr_293;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_293) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_293)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_293)));
    __nv_bfloat162 _max_292 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_293);
    amax_b = _max_292;
    __nv_bfloat162 _hxa_295 = amax_a;
    __nv_bfloat162 _hxb_295 = __as_bf16x2(cols[5]);
    unsigned int _hxr_295;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_295) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_295)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_295)));
    __nv_bfloat162 _max_294 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_295);
    amax_a = _max_294;
    __nv_bfloat162 _hxa_297 = amax_b;
    __nv_bfloat162 _hxb_297 = __as_bf16x2(cols[21]);
    unsigned int _hxr_297;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_297) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_297)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_297)));
    __nv_bfloat162 _max_296 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_297);
    amax_b = _max_296;
    __nv_bfloat162 _hxa_299 = amax_a;
    __nv_bfloat162 _hxb_299 = __as_bf16x2(cols[6]);
    unsigned int _hxr_299;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_299) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_299)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_299)));
    __nv_bfloat162 _max_298 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_299);
    amax_a = _max_298;
    __nv_bfloat162 _hxa_301 = amax_b;
    __nv_bfloat162 _hxb_301 = __as_bf16x2(cols[22]);
    unsigned int _hxr_301;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_301) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_301)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_301)));
    __nv_bfloat162 _max_300 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_301);
    amax_b = _max_300;
    __nv_bfloat162 _hxa_303 = amax_a;
    __nv_bfloat162 _hxb_303 = __as_bf16x2(cols[7]);
    unsigned int _hxr_303;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_303) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_303)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_303)));
    __nv_bfloat162 _max_302 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_303);
    amax_a = _max_302;
    __nv_bfloat162 _hxa_305 = amax_b;
    __nv_bfloat162 _hxb_305 = __as_bf16x2(cols[23]);
    unsigned int _hxr_305;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_305) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_305)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_305)));
    __nv_bfloat162 _max_304 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_305);
    amax_b = _max_304;
    __nv_bfloat162 _hxa_307 = amax_a;
    __nv_bfloat162 _hxb_307 = __as_bf16x2(cols[8]);
    unsigned int _hxr_307;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_307) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_307)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_307)));
    __nv_bfloat162 _max_306 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_307);
    amax_a = _max_306;
    __nv_bfloat162 _hxa_309 = amax_b;
    __nv_bfloat162 _hxb_309 = __as_bf16x2(cols[24]);
    unsigned int _hxr_309;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_309) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_309)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_309)));
    __nv_bfloat162 _max_308 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_309);
    amax_b = _max_308;
    __nv_bfloat162 _hxa_311 = amax_a;
    __nv_bfloat162 _hxb_311 = __as_bf16x2(cols[9]);
    unsigned int _hxr_311;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_311) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_311)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_311)));
    __nv_bfloat162 _max_310 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_311);
    amax_a = _max_310;
    __nv_bfloat162 _hxa_313 = amax_b;
    __nv_bfloat162 _hxb_313 = __as_bf16x2(cols[25]);
    unsigned int _hxr_313;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_313) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_313)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_313)));
    __nv_bfloat162 _max_312 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_313);
    amax_b = _max_312;
    __nv_bfloat162 _hxa_315 = amax_a;
    __nv_bfloat162 _hxb_315 = __as_bf16x2(cols[10]);
    unsigned int _hxr_315;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_315) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_315)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_315)));
    __nv_bfloat162 _max_314 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_315);
    amax_a = _max_314;
    __nv_bfloat162 _hxa_317 = amax_b;
    __nv_bfloat162 _hxb_317 = __as_bf16x2(cols[26]);
    unsigned int _hxr_317;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_317) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_317)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_317)));
    __nv_bfloat162 _max_316 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_317);
    amax_b = _max_316;
    __nv_bfloat162 _hxa_319 = amax_a;
    __nv_bfloat162 _hxb_319 = __as_bf16x2(cols[11]);
    unsigned int _hxr_319;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_319) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_319)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_319)));
    __nv_bfloat162 _max_318 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_319);
    amax_a = _max_318;
    __nv_bfloat162 _hxa_321 = amax_b;
    __nv_bfloat162 _hxb_321 = __as_bf16x2(cols[27]);
    unsigned int _hxr_321;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_321) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_321)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_321)));
    __nv_bfloat162 _max_320 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_321);
    amax_b = _max_320;
    __nv_bfloat162 _hxa_323 = amax_a;
    __nv_bfloat162 _hxb_323 = __as_bf16x2(cols[12]);
    unsigned int _hxr_323;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_323) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_323)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_323)));
    __nv_bfloat162 _max_322 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_323);
    amax_a = _max_322;
    __nv_bfloat162 _hxa_325 = amax_b;
    __nv_bfloat162 _hxb_325 = __as_bf16x2(cols[28]);
    unsigned int _hxr_325;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_325) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_325)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_325)));
    __nv_bfloat162 _max_324 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_325);
    amax_b = _max_324;
    __nv_bfloat162 _hxa_327 = amax_a;
    __nv_bfloat162 _hxb_327 = __as_bf16x2(cols[13]);
    unsigned int _hxr_327;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_327) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_327)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_327)));
    __nv_bfloat162 _max_326 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_327);
    amax_a = _max_326;
    __nv_bfloat162 _hxa_329 = amax_b;
    __nv_bfloat162 _hxb_329 = __as_bf16x2(cols[29]);
    unsigned int _hxr_329;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_329) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_329)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_329)));
    __nv_bfloat162 _max_328 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_329);
    amax_b = _max_328;
    __nv_bfloat162 _hxa_331 = amax_a;
    __nv_bfloat162 _hxb_331 = __as_bf16x2(cols[14]);
    unsigned int _hxr_331;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_331) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_331)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_331)));
    __nv_bfloat162 _max_330 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_331);
    amax_a = _max_330;
    __nv_bfloat162 _hxa_333 = amax_b;
    __nv_bfloat162 _hxb_333 = __as_bf16x2(cols[30]);
    unsigned int _hxr_333;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_333) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_333)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_333)));
    __nv_bfloat162 _max_332 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_333);
    amax_b = _max_332;
    __nv_bfloat162 _hxa_335 = amax_a;
    __nv_bfloat162 _hxb_335 = __as_bf16x2(cols[15]);
    unsigned int _hxr_335;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_335) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_335)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_335)));
    __nv_bfloat162 _max_334 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_335);
    amax_a = _max_334;
    __nv_bfloat162 _hxa_337 = amax_b;
    __nv_bfloat162 _hxb_337 = __as_bf16x2(cols[31]);
    unsigned int _hxr_337;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_337) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_337)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_337)));
    __nv_bfloat162 _max_336 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_337);
    amax_b = _max_336;
    unsigned int mag_a = __as_u32(amax_a) & 2147450879;
    unsigned int mag_b = __as_u32(amax_b) & 2147450879;
    unsigned int lows_ab = mag_a & 65535 | mag_b << 16;
    unsigned int highs_ab = mag_a >> 16 | mag_b & 4294901760u;
    __nv_bfloat162 col_max = __hmax2_nan(__as_bf16x2(lows_ab), __as_bf16x2(highs_ab));
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
    int gcol_a = bx * 64 + warp_1 * 16 + (lane_1 >> 2);
    int gcol_b = gcol_a + 8;
    int out_row = by * 128 + (lane_1 & 3) * 32;
    unsigned int words_a[8];
    unsigned int words_b[8];
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_339;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_339) : "r"(bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_339) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_338;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_338) : "h"(e4m3x2_bf16x2), "h"(_cvt_satfinite_e4m3x2_bf16x2_339));
    words_a[0] = _mov_b32_pack_u16x2_338;
    uint32_t bf16x2_mul_0;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_0) : "r"(cols[16]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_0) : "r"(cols[17]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_341;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_341) : "r"(bf16x2_mul_1_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_341) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_340;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_340) : "h"(e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_341));
    words_b[0] = _mov_b32_pack_u16x2_340;
    uint32_t bf16x2_mul_4;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4) : "r"(cols[2]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_1) : "r"(cols[3]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_343;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_343) : "r"(bf16x2_mul_1_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_343) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_342;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_342) : "h"(e4m3x2_bf16x2_4), "h"(_cvt_satfinite_e4m3x2_bf16x2_343));
    words_a[1] = _mov_b32_pack_u16x2_342;
    uint32_t bf16x2_mul_5;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5) : "r"(cols[18]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_2) : "r"(cols[19]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_345;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_345) : "r"(bf16x2_mul_1_2));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_2);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_345) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_344;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_344) : "h"(e4m3x2_bf16x2_5), "h"(_cvt_satfinite_e4m3x2_bf16x2_345));
    words_b[1] = _mov_b32_pack_u16x2_344;
    uint32_t bf16x2_mul_6;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6) : "r"(cols[4]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_3) : "r"(cols[5]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_347;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_347) : "r"(bf16x2_mul_1_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_347) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_346;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_346) : "h"(e4m3x2_bf16x2_6), "h"(_cvt_satfinite_e4m3x2_bf16x2_347));
    words_a[2] = _mov_b32_pack_u16x2_346;
    uint32_t bf16x2_mul_7;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7) : "r"(cols[20]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_4) : "r"(cols[21]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_349;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_349) : "r"(bf16x2_mul_1_4));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_4);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_349) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_348;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_348) : "h"(e4m3x2_bf16x2_7), "h"(_cvt_satfinite_e4m3x2_bf16x2_349));
    words_b[2] = _mov_b32_pack_u16x2_348;
    uint32_t bf16x2_mul_8;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_8) : "r"(cols[6]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_5) : "r"(cols[7]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_351;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_351) : "r"(bf16x2_mul_1_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_351) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_350;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_350) : "h"(e4m3x2_bf16x2_8), "h"(_cvt_satfinite_e4m3x2_bf16x2_351));
    words_a[3] = _mov_b32_pack_u16x2_350;
    uint32_t bf16x2_mul_9;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_9) : "r"(cols[22]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_6) : "r"(cols[23]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_353;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_353) : "r"(bf16x2_mul_1_6));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_6);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_353) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_352;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_352) : "h"(e4m3x2_bf16x2_9), "h"(_cvt_satfinite_e4m3x2_bf16x2_353));
    words_b[3] = _mov_b32_pack_u16x2_352;
    uint32_t bf16x2_mul_10;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_10) : "r"(cols[8]), "r"(inv2));
    uint16_t e4m3x2_bf16x2_10;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_10) : "r"(bf16x2_mul_10));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_10);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_10) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_7;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_7) : "r"(cols[9]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_355;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_355) : "r"(bf16x2_mul_1_7));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_7);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_355) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_354;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_354) : "h"(e4m3x2_bf16x2_10), "h"(_cvt_satfinite_e4m3x2_bf16x2_355));
    words_a[4] = _mov_b32_pack_u16x2_354;
    uint32_t bf16x2_mul_11;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_11) : "r"(cols[24]), "r"(inv2_0));
    uint16_t e4m3x2_bf16x2_11;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_11) : "r"(bf16x2_mul_11));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_11);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_11) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_8;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_8) : "r"(cols[25]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_357;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_357) : "r"(bf16x2_mul_1_8));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_8);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_357) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_356;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_356) : "h"(e4m3x2_bf16x2_11), "h"(_cvt_satfinite_e4m3x2_bf16x2_357));
    words_b[4] = _mov_b32_pack_u16x2_356;
    uint32_t bf16x2_mul_12;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_12) : "r"(cols[10]), "r"(inv2));
    uint16_t e4m3x2_bf16x2_12;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_12) : "r"(bf16x2_mul_12));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_12);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_12) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_9;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_9) : "r"(cols[11]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_359;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_359) : "r"(bf16x2_mul_1_9));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_9);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_359) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_358;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_358) : "h"(e4m3x2_bf16x2_12), "h"(_cvt_satfinite_e4m3x2_bf16x2_359));
    words_a[5] = _mov_b32_pack_u16x2_358;
    uint32_t bf16x2_mul_13;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_13) : "r"(cols[26]), "r"(inv2_0));
    uint16_t e4m3x2_bf16x2_13;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_13) : "r"(bf16x2_mul_13));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_13);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_13) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_10;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_10) : "r"(cols[27]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_361;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_361) : "r"(bf16x2_mul_1_10));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_10);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_361) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_360;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_360) : "h"(e4m3x2_bf16x2_13), "h"(_cvt_satfinite_e4m3x2_bf16x2_361));
    words_b[5] = _mov_b32_pack_u16x2_360;
    uint32_t bf16x2_mul_14;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_14) : "r"(cols[12]), "r"(inv2));
    uint16_t e4m3x2_bf16x2_14;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_14) : "r"(bf16x2_mul_14));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_14);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_14) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_11;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_11) : "r"(cols[13]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_363;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_363) : "r"(bf16x2_mul_1_11));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_11);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_363) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_362;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_362) : "h"(e4m3x2_bf16x2_14), "h"(_cvt_satfinite_e4m3x2_bf16x2_363));
    words_a[6] = _mov_b32_pack_u16x2_362;
    uint32_t bf16x2_mul_15;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_15) : "r"(cols[28]), "r"(inv2_0));
    uint16_t e4m3x2_bf16x2_15;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_15) : "r"(bf16x2_mul_15));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_15);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_15) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_12;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_12) : "r"(cols[29]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_365;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_365) : "r"(bf16x2_mul_1_12));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_12);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_365) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_364;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_364) : "h"(e4m3x2_bf16x2_15), "h"(_cvt_satfinite_e4m3x2_bf16x2_365));
    words_b[6] = _mov_b32_pack_u16x2_364;
    uint32_t bf16x2_mul_16;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_16) : "r"(cols[14]), "r"(inv2));
    uint16_t e4m3x2_bf16x2_16;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_16) : "r"(bf16x2_mul_16));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_16);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_16) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_13;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_13) : "r"(cols[15]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_367;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_367) : "r"(bf16x2_mul_1_13));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_13);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_367) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_366;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_366) : "h"(e4m3x2_bf16x2_16), "h"(_cvt_satfinite_e4m3x2_bf16x2_367));
    words_a[7] = _mov_b32_pack_u16x2_366;
    uint32_t bf16x2_mul_17;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_17) : "r"(cols[30]), "r"(inv2_0));
    uint16_t e4m3x2_bf16x2_17;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_17) : "r"(bf16x2_mul_17));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_17);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_17) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_14;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_14) : "r"(cols[31]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_369;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_369) : "r"(bf16x2_mul_1_14));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_14);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_369) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_368;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_368) : "h"(e4m3x2_bf16x2_17), "h"(_cvt_satfinite_e4m3x2_bf16x2_369));
    words_b[7] = _mov_b32_pack_u16x2_368;
    {
        const unsigned* _raw_stv8_370 = reinterpret_cast<const unsigned*>(words_a + 0);
        asm volatile(
            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
            :: "l"((void*)(col_output + (gcol_a * M + out_row))), "r"(_raw_stv8_370[0]), "r"(_raw_stv8_370[1]), "r"(_raw_stv8_370[2]), "r"(_raw_stv8_370[3]), "r"(_raw_stv8_370[4]), "r"(_raw_stv8_370[5]), "r"(_raw_stv8_370[6]), "r"(_raw_stv8_370[7]) : "memory");
    }
    {
        const unsigned* _raw_stv8_371 = reinterpret_cast<const unsigned*>(words_b + 0);
        asm volatile(
            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
            :: "l"((void*)(col_output + (gcol_b * M + out_row))), "r"(_raw_stv8_371[0]), "r"(_raw_stv8_371[1]), "r"(_raw_stv8_371[2]), "r"(_raw_stv8_371[3]), "r"(_raw_stv8_371[4]), "r"(_raw_stv8_371[5]), "r"(_raw_stv8_371[6]), "r"(_raw_stv8_371[7]) : "memory");
    }
    int row_tiles = M >> 7;
    int rblk = by * 4 + (lane_1 & 3);
    int col_scale_a = ((gcol_a >> 7) * row_tiles + (rblk >> 2)) * 512 + (gcol_a & 31) * 16 + (gcol_a >> 5 & 3) * 4 + (rblk & 3);
    int col_scale_b = ((gcol_b >> 7) * row_tiles + (rblk >> 2)) * 512 + (gcol_b & 31) * 16 + (gcol_b >> 5 & 3) * 4 + (rblk & 3);
    *(reinterpret_cast<unsigned char*>(col_scales + col_scale_a) + (0)) = (unsigned char)(scale);
    *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b) + (0)) = (unsigned char)(scale_0);
    int lane_0 = tid & 31;
    int warp_0 = tid >> 5;
    int k8_0 = lane_0 & 7;
    int srow_0 = (k8_0 >> 1) * 32 + (lane_0 >> 3) * 2 + (k8_0 & 1);
    unsigned int sswizzle_0 = (unsigned int)(srow_0 >> 4 & 6 | srow_0 & 1);
    unsigned int cols_0[32];
    unsigned int strip_chunk0_1_2 = (unsigned int)(warp_0 * 2);
    unsigned int strip_src0_1_2 = tiles_addr + 16384 + (unsigned int)(srow_0 * 128) + ((strip_chunk0_1_2 ^ sswizzle_0) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_0[0]), "=r"(cols_0[1]), "=r"(cols_0[2]), "=r"(cols_0[3])
        : "r"(strip_src0_1_2)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_0[4]), "=r"(cols_0[5]), "=r"(cols_0[6]), "=r"(cols_0[7])
        : "r"(strip_src0_1_2 + 1024)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_0[8]), "=r"(cols_0[9]), "=r"(cols_0[10]), "=r"(cols_0[11])
        : "r"(strip_src0_1_2 + 2048)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_0[12]), "=r"(cols_0[13]), "=r"(cols_0[14]), "=r"(cols_0[15])
        : "r"(strip_src0_1_2 + 3072)
        : "memory");
    unsigned int strip_chunk0_1_3 = (unsigned int)(warp_0 * 2 + 1);
    unsigned int strip_src0_1_3 = tiles_addr + 16384 + (unsigned int)(srow_0 * 128) + ((strip_chunk0_1_3 ^ sswizzle_0) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_0[16]), "=r"(cols_0[17]), "=r"(cols_0[18]), "=r"(cols_0[19])
        : "r"(strip_src0_1_3)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_0[20]), "=r"(cols_0[21]), "=r"(cols_0[22]), "=r"(cols_0[23])
        : "r"(strip_src0_1_3 + 1024)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_0[24]), "=r"(cols_0[25]), "=r"(cols_0[26]), "=r"(cols_0[27])
        : "r"(strip_src0_1_3 + 2048)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_0[28]), "=r"(cols_0[29]), "=r"(cols_0[30]), "=r"(cols_0[31])
        : "r"(strip_src0_1_3 + 3072)
        : "memory");
    __nv_bfloat162 amax_a_0 = __as_bf16x2(0);
    __nv_bfloat162 amax_b_0 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_373 = amax_a_0;
    __nv_bfloat162 _hxb_373 = __as_bf16x2(cols_0[0]);
    unsigned int _hxr_373;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_373) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_373)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_373)));
    __nv_bfloat162 _max_372 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_373);
    amax_a_0 = _max_372;
    __nv_bfloat162 _hxa_375 = amax_b_0;
    __nv_bfloat162 _hxb_375 = __as_bf16x2(cols_0[16]);
    unsigned int _hxr_375;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_375) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_375)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_375)));
    __nv_bfloat162 _max_374 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_375);
    amax_b_0 = _max_374;
    __nv_bfloat162 _hxa_377 = amax_a_0;
    __nv_bfloat162 _hxb_377 = __as_bf16x2(cols_0[1]);
    unsigned int _hxr_377;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_377) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_377)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_377)));
    __nv_bfloat162 _max_376 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_377);
    amax_a_0 = _max_376;
    __nv_bfloat162 _hxa_379 = amax_b_0;
    __nv_bfloat162 _hxb_379 = __as_bf16x2(cols_0[17]);
    unsigned int _hxr_379;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_379) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_379)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_379)));
    __nv_bfloat162 _max_378 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_379);
    amax_b_0 = _max_378;
    __nv_bfloat162 _hxa_381 = amax_a_0;
    __nv_bfloat162 _hxb_381 = __as_bf16x2(cols_0[2]);
    unsigned int _hxr_381;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_381) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_381)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_381)));
    __nv_bfloat162 _max_380 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_381);
    amax_a_0 = _max_380;
    __nv_bfloat162 _hxa_383 = amax_b_0;
    __nv_bfloat162 _hxb_383 = __as_bf16x2(cols_0[18]);
    unsigned int _hxr_383;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_383) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_383)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_383)));
    __nv_bfloat162 _max_382 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_383);
    amax_b_0 = _max_382;
    __nv_bfloat162 _hxa_385 = amax_a_0;
    __nv_bfloat162 _hxb_385 = __as_bf16x2(cols_0[3]);
    unsigned int _hxr_385;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_385) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_385)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_385)));
    __nv_bfloat162 _max_384 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_385);
    amax_a_0 = _max_384;
    __nv_bfloat162 _hxa_387 = amax_b_0;
    __nv_bfloat162 _hxb_387 = __as_bf16x2(cols_0[19]);
    unsigned int _hxr_387;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_387) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_387)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_387)));
    __nv_bfloat162 _max_386 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_387);
    amax_b_0 = _max_386;
    __nv_bfloat162 _hxa_389 = amax_a_0;
    __nv_bfloat162 _hxb_389 = __as_bf16x2(cols_0[4]);
    unsigned int _hxr_389;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_389) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_389)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_389)));
    __nv_bfloat162 _max_388 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_389);
    amax_a_0 = _max_388;
    __nv_bfloat162 _hxa_391 = amax_b_0;
    __nv_bfloat162 _hxb_391 = __as_bf16x2(cols_0[20]);
    unsigned int _hxr_391;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_391) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_391)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_391)));
    __nv_bfloat162 _max_390 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_391);
    amax_b_0 = _max_390;
    __nv_bfloat162 _hxa_393 = amax_a_0;
    __nv_bfloat162 _hxb_393 = __as_bf16x2(cols_0[5]);
    unsigned int _hxr_393;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_393) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_393)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_393)));
    __nv_bfloat162 _max_392 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_393);
    amax_a_0 = _max_392;
    __nv_bfloat162 _hxa_395 = amax_b_0;
    __nv_bfloat162 _hxb_395 = __as_bf16x2(cols_0[21]);
    unsigned int _hxr_395;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_395) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_395)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_395)));
    __nv_bfloat162 _max_394 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_395);
    amax_b_0 = _max_394;
    __nv_bfloat162 _hxa_397 = amax_a_0;
    __nv_bfloat162 _hxb_397 = __as_bf16x2(cols_0[6]);
    unsigned int _hxr_397;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_397) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_397)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_397)));
    __nv_bfloat162 _max_396 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_397);
    amax_a_0 = _max_396;
    __nv_bfloat162 _hxa_399 = amax_b_0;
    __nv_bfloat162 _hxb_399 = __as_bf16x2(cols_0[22]);
    unsigned int _hxr_399;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_399) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_399)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_399)));
    __nv_bfloat162 _max_398 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_399);
    amax_b_0 = _max_398;
    __nv_bfloat162 _hxa_401 = amax_a_0;
    __nv_bfloat162 _hxb_401 = __as_bf16x2(cols_0[7]);
    unsigned int _hxr_401;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_401) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_401)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_401)));
    __nv_bfloat162 _max_400 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_401);
    amax_a_0 = _max_400;
    __nv_bfloat162 _hxa_403 = amax_b_0;
    __nv_bfloat162 _hxb_403 = __as_bf16x2(cols_0[23]);
    unsigned int _hxr_403;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_403) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_403)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_403)));
    __nv_bfloat162 _max_402 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_403);
    amax_b_0 = _max_402;
    __nv_bfloat162 _hxa_405 = amax_a_0;
    __nv_bfloat162 _hxb_405 = __as_bf16x2(cols_0[8]);
    unsigned int _hxr_405;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_405) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_405)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_405)));
    __nv_bfloat162 _max_404 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_405);
    amax_a_0 = _max_404;
    __nv_bfloat162 _hxa_407 = amax_b_0;
    __nv_bfloat162 _hxb_407 = __as_bf16x2(cols_0[24]);
    unsigned int _hxr_407;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_407) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_407)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_407)));
    __nv_bfloat162 _max_406 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_407);
    amax_b_0 = _max_406;
    __nv_bfloat162 _hxa_409 = amax_a_0;
    __nv_bfloat162 _hxb_409 = __as_bf16x2(cols_0[9]);
    unsigned int _hxr_409;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_409) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_409)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_409)));
    __nv_bfloat162 _max_408 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_409);
    amax_a_0 = _max_408;
    __nv_bfloat162 _hxa_411 = amax_b_0;
    __nv_bfloat162 _hxb_411 = __as_bf16x2(cols_0[25]);
    unsigned int _hxr_411;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_411) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_411)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_411)));
    __nv_bfloat162 _max_410 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_411);
    amax_b_0 = _max_410;
    __nv_bfloat162 _hxa_413 = amax_a_0;
    __nv_bfloat162 _hxb_413 = __as_bf16x2(cols_0[10]);
    unsigned int _hxr_413;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_413) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_413)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_413)));
    __nv_bfloat162 _max_412 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_413);
    amax_a_0 = _max_412;
    __nv_bfloat162 _hxa_415 = amax_b_0;
    __nv_bfloat162 _hxb_415 = __as_bf16x2(cols_0[26]);
    unsigned int _hxr_415;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_415) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_415)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_415)));
    __nv_bfloat162 _max_414 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_415);
    amax_b_0 = _max_414;
    __nv_bfloat162 _hxa_417 = amax_a_0;
    __nv_bfloat162 _hxb_417 = __as_bf16x2(cols_0[11]);
    unsigned int _hxr_417;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_417) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_417)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_417)));
    __nv_bfloat162 _max_416 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_417);
    amax_a_0 = _max_416;
    __nv_bfloat162 _hxa_419 = amax_b_0;
    __nv_bfloat162 _hxb_419 = __as_bf16x2(cols_0[27]);
    unsigned int _hxr_419;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_419) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_419)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_419)));
    __nv_bfloat162 _max_418 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_419);
    amax_b_0 = _max_418;
    __nv_bfloat162 _hxa_421 = amax_a_0;
    __nv_bfloat162 _hxb_421 = __as_bf16x2(cols_0[12]);
    unsigned int _hxr_421;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_421) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_421)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_421)));
    __nv_bfloat162 _max_420 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_421);
    amax_a_0 = _max_420;
    __nv_bfloat162 _hxa_423 = amax_b_0;
    __nv_bfloat162 _hxb_423 = __as_bf16x2(cols_0[28]);
    unsigned int _hxr_423;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_423) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_423)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_423)));
    __nv_bfloat162 _max_422 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_423);
    amax_b_0 = _max_422;
    __nv_bfloat162 _hxa_425 = amax_a_0;
    __nv_bfloat162 _hxb_425 = __as_bf16x2(cols_0[13]);
    unsigned int _hxr_425;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_425) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_425)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_425)));
    __nv_bfloat162 _max_424 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_425);
    amax_a_0 = _max_424;
    __nv_bfloat162 _hxa_427 = amax_b_0;
    __nv_bfloat162 _hxb_427 = __as_bf16x2(cols_0[29]);
    unsigned int _hxr_427;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_427) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_427)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_427)));
    __nv_bfloat162 _max_426 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_427);
    amax_b_0 = _max_426;
    __nv_bfloat162 _hxa_429 = amax_a_0;
    __nv_bfloat162 _hxb_429 = __as_bf16x2(cols_0[14]);
    unsigned int _hxr_429;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_429) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_429)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_429)));
    __nv_bfloat162 _max_428 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_429);
    amax_a_0 = _max_428;
    __nv_bfloat162 _hxa_431 = amax_b_0;
    __nv_bfloat162 _hxb_431 = __as_bf16x2(cols_0[30]);
    unsigned int _hxr_431;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_431) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_431)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_431)));
    __nv_bfloat162 _max_430 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_431);
    amax_b_0 = _max_430;
    __nv_bfloat162 _hxa_433 = amax_a_0;
    __nv_bfloat162 _hxb_433 = __as_bf16x2(cols_0[15]);
    unsigned int _hxr_433;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_433) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_433)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_433)));
    __nv_bfloat162 _max_432 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_433);
    amax_a_0 = _max_432;
    __nv_bfloat162 _hxa_435 = amax_b_0;
    __nv_bfloat162 _hxb_435 = __as_bf16x2(cols_0[31]);
    unsigned int _hxr_435;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_435) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_435)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_435)));
    __nv_bfloat162 _max_434 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_435);
    amax_b_0 = _max_434;
    unsigned int mag_a_0 = __as_u32(amax_a_0) & 2147450879;
    unsigned int mag_b_0 = __as_u32(amax_b_0) & 2147450879;
    unsigned int lows_ab_0 = mag_a_0 & 65535 | mag_b_0 << 16;
    unsigned int highs_ab_0 = mag_a_0 >> 16 | mag_b_0 & 4294901760u;
    __nv_bfloat162 col_max_0 = __hmax2_nan(__as_bf16x2(lows_ab_0), __as_bf16x2(highs_ab_0));
    unsigned int col_max_bits_0 = __as_u32(col_max_0);
    unsigned int amax_bits_1 = (col_max_bits_0 & 65535) << 16;
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
    unsigned int amax_bits_2 = (col_max_bits_0 >> 16 & 65535) << 16;
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
    int gcol_a_0 = bx * 64 + K + warp_0 * 16 + (lane_0 >> 2);
    int gcol_b_0 = gcol_a_0 + 8;
    int out_row_0 = by * 128 + (lane_0 & 3) * 32;
    unsigned int words_a_0[8];
    unsigned int words_b_0[8];
    uint32_t bf16x2_mul_18;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_18) : "r"(cols_0[0]), "r"(inv2_2));
    uint16_t e4m3x2_bf16x2_18;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_18) : "r"(bf16x2_mul_18));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_18);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_18) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_15;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_15) : "r"(cols_0[1]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_437;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_437) : "r"(bf16x2_mul_1_15));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_15);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_437) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_436;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_436) : "h"(e4m3x2_bf16x2_18), "h"(_cvt_satfinite_e4m3x2_bf16x2_437));
    words_a_0[0] = _mov_b32_pack_u16x2_436;
    uint32_t bf16x2_mul_19;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_19) : "r"(cols_0[16]), "r"(inv2_3));
    uint16_t e4m3x2_bf16x2_19;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_19) : "r"(bf16x2_mul_19));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_19);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_19) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_16;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_16) : "r"(cols_0[17]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_439;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_439) : "r"(bf16x2_mul_1_16));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_16);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_439) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_438;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_438) : "h"(e4m3x2_bf16x2_19), "h"(_cvt_satfinite_e4m3x2_bf16x2_439));
    words_b_0[0] = _mov_b32_pack_u16x2_438;
    uint32_t bf16x2_mul_20;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_20) : "r"(cols_0[2]), "r"(inv2_2));
    uint16_t e4m3x2_bf16x2_20;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_20) : "r"(bf16x2_mul_20));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_20);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_20) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_17;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_17) : "r"(cols_0[3]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_441;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_441) : "r"(bf16x2_mul_1_17));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_17);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_441) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_440;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_440) : "h"(e4m3x2_bf16x2_20), "h"(_cvt_satfinite_e4m3x2_bf16x2_441));
    words_a_0[1] = _mov_b32_pack_u16x2_440;
    uint32_t bf16x2_mul_21;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_21) : "r"(cols_0[18]), "r"(inv2_3));
    uint16_t e4m3x2_bf16x2_21;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_21) : "r"(bf16x2_mul_21));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_21);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_21) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_18;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_18) : "r"(cols_0[19]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_443;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_443) : "r"(bf16x2_mul_1_18));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_18);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_443) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_442;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_442) : "h"(e4m3x2_bf16x2_21), "h"(_cvt_satfinite_e4m3x2_bf16x2_443));
    words_b_0[1] = _mov_b32_pack_u16x2_442;
    uint32_t bf16x2_mul_22;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_22) : "r"(cols_0[4]), "r"(inv2_2));
    uint16_t e4m3x2_bf16x2_22;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_22) : "r"(bf16x2_mul_22));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_22);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_22) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_19;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_19) : "r"(cols_0[5]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_445;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_445) : "r"(bf16x2_mul_1_19));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_19);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_445) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_444;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_444) : "h"(e4m3x2_bf16x2_22), "h"(_cvt_satfinite_e4m3x2_bf16x2_445));
    words_a_0[2] = _mov_b32_pack_u16x2_444;
    uint32_t bf16x2_mul_23;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_23) : "r"(cols_0[20]), "r"(inv2_3));
    uint16_t e4m3x2_bf16x2_23;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_23) : "r"(bf16x2_mul_23));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_23);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_23) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_20;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_20) : "r"(cols_0[21]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_447;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_447) : "r"(bf16x2_mul_1_20));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_20);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_447) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_446;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_446) : "h"(e4m3x2_bf16x2_23), "h"(_cvt_satfinite_e4m3x2_bf16x2_447));
    words_b_0[2] = _mov_b32_pack_u16x2_446;
    uint32_t bf16x2_mul_24;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_24) : "r"(cols_0[6]), "r"(inv2_2));
    uint16_t e4m3x2_bf16x2_24;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_24) : "r"(bf16x2_mul_24));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_24);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_24) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_21;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_21) : "r"(cols_0[7]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_449;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_449) : "r"(bf16x2_mul_1_21));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_21);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_449) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_448;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_448) : "h"(e4m3x2_bf16x2_24), "h"(_cvt_satfinite_e4m3x2_bf16x2_449));
    words_a_0[3] = _mov_b32_pack_u16x2_448;
    uint32_t bf16x2_mul_25;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_25) : "r"(cols_0[22]), "r"(inv2_3));
    uint16_t e4m3x2_bf16x2_25;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_25) : "r"(bf16x2_mul_25));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_25);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_25) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_22;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_22) : "r"(cols_0[23]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_451;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_451) : "r"(bf16x2_mul_1_22));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_22);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_451) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_450;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_450) : "h"(e4m3x2_bf16x2_25), "h"(_cvt_satfinite_e4m3x2_bf16x2_451));
    words_b_0[3] = _mov_b32_pack_u16x2_450;
    uint32_t bf16x2_mul_26;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_26) : "r"(cols_0[8]), "r"(inv2_2));
    uint16_t e4m3x2_bf16x2_26;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_26) : "r"(bf16x2_mul_26));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_26);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_26) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_23;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_23) : "r"(cols_0[9]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_453;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_453) : "r"(bf16x2_mul_1_23));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_23);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_453) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_452;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_452) : "h"(e4m3x2_bf16x2_26), "h"(_cvt_satfinite_e4m3x2_bf16x2_453));
    words_a_0[4] = _mov_b32_pack_u16x2_452;
    uint32_t bf16x2_mul_27;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_27) : "r"(cols_0[24]), "r"(inv2_3));
    uint16_t e4m3x2_bf16x2_27;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_27) : "r"(bf16x2_mul_27));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_27);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_27) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_24;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_24) : "r"(cols_0[25]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_455;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_455) : "r"(bf16x2_mul_1_24));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_24);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_455) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_454;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_454) : "h"(e4m3x2_bf16x2_27), "h"(_cvt_satfinite_e4m3x2_bf16x2_455));
    words_b_0[4] = _mov_b32_pack_u16x2_454;
    uint32_t bf16x2_mul_28;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_28) : "r"(cols_0[10]), "r"(inv2_2));
    uint16_t e4m3x2_bf16x2_28;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_28) : "r"(bf16x2_mul_28));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_28);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_28) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_25;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_25) : "r"(cols_0[11]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_457;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_457) : "r"(bf16x2_mul_1_25));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_25);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_457) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_456;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_456) : "h"(e4m3x2_bf16x2_28), "h"(_cvt_satfinite_e4m3x2_bf16x2_457));
    words_a_0[5] = _mov_b32_pack_u16x2_456;
    uint32_t bf16x2_mul_29;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_29) : "r"(cols_0[26]), "r"(inv2_3));
    uint16_t e4m3x2_bf16x2_29;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_29) : "r"(bf16x2_mul_29));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_29);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_29) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_26;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_26) : "r"(cols_0[27]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_459;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_459) : "r"(bf16x2_mul_1_26));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_26);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_459) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_458;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_458) : "h"(e4m3x2_bf16x2_29), "h"(_cvt_satfinite_e4m3x2_bf16x2_459));
    words_b_0[5] = _mov_b32_pack_u16x2_458;
    uint32_t bf16x2_mul_30;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_30) : "r"(cols_0[12]), "r"(inv2_2));
    uint16_t e4m3x2_bf16x2_30;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_30) : "r"(bf16x2_mul_30));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_30);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_30) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_27;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_27) : "r"(cols_0[13]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_461;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_461) : "r"(bf16x2_mul_1_27));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_27);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_461) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_460;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_460) : "h"(e4m3x2_bf16x2_30), "h"(_cvt_satfinite_e4m3x2_bf16x2_461));
    words_a_0[6] = _mov_b32_pack_u16x2_460;
    uint32_t bf16x2_mul_31;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_31) : "r"(cols_0[28]), "r"(inv2_3));
    uint16_t e4m3x2_bf16x2_31;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_31) : "r"(bf16x2_mul_31));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_31);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_31) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_28;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_28) : "r"(cols_0[29]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_463;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_463) : "r"(bf16x2_mul_1_28));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_28);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_463) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_462;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_462) : "h"(e4m3x2_bf16x2_31), "h"(_cvt_satfinite_e4m3x2_bf16x2_463));
    words_b_0[6] = _mov_b32_pack_u16x2_462;
    uint32_t bf16x2_mul_32;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_32) : "r"(cols_0[14]), "r"(inv2_2));
    uint16_t e4m3x2_bf16x2_32;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_32) : "r"(bf16x2_mul_32));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_32);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_32) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_29;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_29) : "r"(cols_0[15]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_465;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_465) : "r"(bf16x2_mul_1_29));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_29);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_465) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_464;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_464) : "h"(e4m3x2_bf16x2_32), "h"(_cvt_satfinite_e4m3x2_bf16x2_465));
    words_a_0[7] = _mov_b32_pack_u16x2_464;
    uint32_t bf16x2_mul_33;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_33) : "r"(cols_0[30]), "r"(inv2_3));
    uint16_t e4m3x2_bf16x2_33;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_33) : "r"(bf16x2_mul_33));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_33);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_33) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_30;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_30) : "r"(cols_0[31]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_467;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_467) : "r"(bf16x2_mul_1_30));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_30);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_467) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_466;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_466) : "h"(e4m3x2_bf16x2_33), "h"(_cvt_satfinite_e4m3x2_bf16x2_467));
    words_b_0[7] = _mov_b32_pack_u16x2_466;
    {
        const unsigned* _raw_stv8_468 = reinterpret_cast<const unsigned*>(words_a_0 + 0);
        asm volatile(
            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
            :: "l"((void*)(col_output + (gcol_a_0 * M + out_row_0))), "r"(_raw_stv8_468[0]), "r"(_raw_stv8_468[1]), "r"(_raw_stv8_468[2]), "r"(_raw_stv8_468[3]), "r"(_raw_stv8_468[4]), "r"(_raw_stv8_468[5]), "r"(_raw_stv8_468[6]), "r"(_raw_stv8_468[7]) : "memory");
    }
    {
        const unsigned* _raw_stv8_469 = reinterpret_cast<const unsigned*>(words_b_0 + 0);
        asm volatile(
            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
            :: "l"((void*)(col_output + (gcol_b_0 * M + out_row_0))), "r"(_raw_stv8_469[0]), "r"(_raw_stv8_469[1]), "r"(_raw_stv8_469[2]), "r"(_raw_stv8_469[3]), "r"(_raw_stv8_469[4]), "r"(_raw_stv8_469[5]), "r"(_raw_stv8_469[6]), "r"(_raw_stv8_469[7]) : "memory");
    }
    int row_tiles_0 = M >> 7;
    int rblk_0 = by * 4 + (lane_0 & 3);
    int col_scale_a_0 = ((gcol_a_0 >> 7) * row_tiles_0 + (rblk_0 >> 2)) * 512 + (gcol_a_0 & 31) * 16 + (gcol_a_0 >> 5 & 3) * 4 + (rblk_0 & 3);
    int col_scale_b_0 = ((gcol_b_0 >> 7) * row_tiles_0 + (rblk_0 >> 2)) * 512 + (gcol_b_0 & 31) * 16 + (gcol_b_0 >> 5 & 3) * 4 + (rblk_0 & 3);
    *(reinterpret_cast<unsigned char*>(col_scales + col_scale_a_0) + (0)) = (unsigned char)(scale_3);
    *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b_0) + (0)) = (unsigned char)(scale_4);
}

} // extern "C"
