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
kernel_cake_gated_act_mxfp8_69f1ccc15e17f512a07b(__nv_bfloat16* __restrict__ gated_input, __nv_bfloat16* __restrict__ grad, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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
    int grow = by * 64 + row;
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
    unsigned int dgate[8];
    unsigned int dup[8];
    unsigned int dgate_b[8];
    unsigned int dup_b[8];
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
    float2 lowest_f32 = cake_bf16x2_to_f32x2(lowest);
    float _min_7;
    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_7) : "f"(lowest_f32.x), "f"(lowest_f32.y));
    int _warp_vote_6 = __all_sync(0xFFFFFFFF, _min_7 > -87.0f);
    if (_warp_vote_6 != 0) {
        float2 x = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[0]));
        float2 u = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[0]));
        float2 g = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[0]));
        float2 _f32x2_mul_b_8 = make_float2(-1.442695f, -1.442695f);
        float2 t;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t) : "l"(*(const unsigned long long*)&x), "l"(*(const unsigned long long*)&_f32x2_mul_b_8));
        float e0 = approx_exp2(t.x);
        float e1 = approx_exp2(t.y);
        float2 nd = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0, e1));
        float d0 = -nd.x;
        float d1 = -nd.y;
        float r0 = approx_rcp(d0);
        float r1 = approx_rcp(d1);
        float2 r = make_float2(r0, r1);
        float2 s_0 = fma_f32x2_rn_noftz(r, fma_f32x2_rn_noftz(nd, r, make_float2(1.0f, 1.0f)), r);
        float2 s = s_0;
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
        float2 _f32x2_mul_b_10 = make_float2(-1.442695f, -1.442695f);
        float2 t_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_0) : "l"(*(const unsigned long long*)&x_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_10));
        float e0_0 = approx_exp2(t_0.x);
        float e1_0 = approx_exp2(t_0.y);
        float2 nd_0 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_0, e1_0));
        float d0_0 = -nd_0.x;
        float d1_0 = -nd_0.y;
        float r0_0 = approx_rcp(d0_0);
        float r1_0 = approx_rcp(d1_0);
        float2 r_0 = make_float2(r0_0, r1_0);
        float2 s_2 = fma_f32x2_rn_noftz(r_0, fma_f32x2_rn_noftz(nd_0, r_0, make_float2(1.0f, 1.0f)), r_0);
        float2 s_1 = s_2;
        float2 act_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_0) : "l"(*(const unsigned long long*)&x_0), "l"(*(const unsigned long long*)&s_1));
        float2 _f32x2_mul_a_11 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1), s_1), g_0);
        float2 mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_11), "l"(*(const unsigned long long*)&u_0));
        float2 mul_1_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_0) : "l"(*(const unsigned long long*)&act_0), "l"(*(const unsigned long long*)&g_0));
        __nv_bfloat162 bf16x2_2 = __float22bfloat162_rn(make_float2(mul_0.x, mul_0.y));
        __nv_bfloat162 bf16x2_1_1 = __float22bfloat162_rn(make_float2(mul_1_0.x, mul_1_0.y));
        dgate[1] = __as_u32(bf16x2_2);
        dup[1] = __as_u32(bf16x2_1_1);
        float2 x_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[2]));
        float2 u_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[2]));
        float2 g_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[2]));
        float2 _f32x2_mul_b_12 = make_float2(-1.442695f, -1.442695f);
        float2 t_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_1) : "l"(*(const unsigned long long*)&x_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_12));
        float e0_1 = approx_exp2(t_1.x);
        float e1_1 = approx_exp2(t_1.y);
        float2 nd_1 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_1, e1_1));
        float d0_1 = -nd_1.x;
        float d1_1 = -nd_1.y;
        float r0_1 = approx_rcp(d0_1);
        float r1_1 = approx_rcp(d1_1);
        float2 r_1 = make_float2(r0_1, r1_1);
        float2 s_4 = fma_f32x2_rn_noftz(r_1, fma_f32x2_rn_noftz(nd_1, r_1, make_float2(1.0f, 1.0f)), r_1);
        float2 s_3 = s_4;
        float2 act_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_1) : "l"(*(const unsigned long long*)&x_1), "l"(*(const unsigned long long*)&s_3));
        float2 _f32x2_mul_a_13 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_3), s_3), g_1);
        float2 mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_13), "l"(*(const unsigned long long*)&u_1));
        float2 mul_1_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_1) : "l"(*(const unsigned long long*)&act_1), "l"(*(const unsigned long long*)&g_1));
        __nv_bfloat162 bf16x2_3 = __float22bfloat162_rn(make_float2(mul_2.x, mul_2.y));
        __nv_bfloat162 bf16x2_1_2 = __float22bfloat162_rn(make_float2(mul_1_1.x, mul_1_1.y));
        dgate[2] = __as_u32(bf16x2_3);
        dup[2] = __as_u32(bf16x2_1_2);
        float2 x_2 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[3]));
        float2 u_2 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[3]));
        float2 g_2 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[3]));
        float2 _f32x2_mul_b_14 = make_float2(-1.442695f, -1.442695f);
        float2 t_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_2) : "l"(*(const unsigned long long*)&x_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_14));
        float e0_2 = approx_exp2(t_2.x);
        float e1_2 = approx_exp2(t_2.y);
        float2 nd_2 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_2, e1_2));
        float d0_2 = -nd_2.x;
        float d1_2 = -nd_2.y;
        float r0_2 = approx_rcp(d0_2);
        float r1_2 = approx_rcp(d1_2);
        float2 r_2 = make_float2(r0_2, r1_2);
        float2 s_6 = fma_f32x2_rn_noftz(r_2, fma_f32x2_rn_noftz(nd_2, r_2, make_float2(1.0f, 1.0f)), r_2);
        float2 s_5 = s_6;
        float2 act_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_2) : "l"(*(const unsigned long long*)&x_2), "l"(*(const unsigned long long*)&s_5));
        float2 _f32x2_mul_a_15 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_5), s_5), g_2);
        float2 mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&u_2));
        float2 mul_1_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_2) : "l"(*(const unsigned long long*)&act_2), "l"(*(const unsigned long long*)&g_2));
        __nv_bfloat162 bf16x2_4 = __float22bfloat162_rn(make_float2(mul_3.x, mul_3.y));
        __nv_bfloat162 bf16x2_1_3 = __float22bfloat162_rn(make_float2(mul_1_2.x, mul_1_2.y));
        dgate[3] = __as_u32(bf16x2_4);
        dup[3] = __as_u32(bf16x2_1_3);
        float2 x_3 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[4]));
        float2 u_3 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[4]));
        float2 g_3 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[4]));
        float2 _f32x2_mul_b_16 = make_float2(-1.442695f, -1.442695f);
        float2 t_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_3) : "l"(*(const unsigned long long*)&x_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_16));
        float e0_3 = approx_exp2(t_3.x);
        float e1_3 = approx_exp2(t_3.y);
        float2 nd_3 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_3, e1_3));
        float d0_3 = -nd_3.x;
        float d1_3 = -nd_3.y;
        float r0_3 = approx_rcp(d0_3);
        float r1_3 = approx_rcp(d1_3);
        float2 r_3 = make_float2(r0_3, r1_3);
        float2 s_8 = fma_f32x2_rn_noftz(r_3, fma_f32x2_rn_noftz(nd_3, r_3, make_float2(1.0f, 1.0f)), r_3);
        float2 s_7 = s_8;
        float2 act_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_3) : "l"(*(const unsigned long long*)&x_3), "l"(*(const unsigned long long*)&s_7));
        float2 _f32x2_mul_a_17 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_7), s_7), g_3);
        float2 mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&u_3));
        float2 mul_1_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_3) : "l"(*(const unsigned long long*)&act_3), "l"(*(const unsigned long long*)&g_3));
        __nv_bfloat162 bf16x2_5 = __float22bfloat162_rn(make_float2(mul_4.x, mul_4.y));
        __nv_bfloat162 bf16x2_1_4 = __float22bfloat162_rn(make_float2(mul_1_3.x, mul_1_3.y));
        dgate[4] = __as_u32(bf16x2_5);
        dup[4] = __as_u32(bf16x2_1_4);
        float2 x_4 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[5]));
        float2 u_4 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[5]));
        float2 g_4 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[5]));
        float2 _f32x2_mul_b_18 = make_float2(-1.442695f, -1.442695f);
        float2 t_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_4) : "l"(*(const unsigned long long*)&x_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_18));
        float e0_4 = approx_exp2(t_4.x);
        float e1_4 = approx_exp2(t_4.y);
        float2 nd_4 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_4, e1_4));
        float d0_4 = -nd_4.x;
        float d1_4 = -nd_4.y;
        float r0_4 = approx_rcp(d0_4);
        float r1_4 = approx_rcp(d1_4);
        float2 r_4 = make_float2(r0_4, r1_4);
        float2 s_10 = fma_f32x2_rn_noftz(r_4, fma_f32x2_rn_noftz(nd_4, r_4, make_float2(1.0f, 1.0f)), r_4);
        float2 s_9 = s_10;
        float2 act_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_4) : "l"(*(const unsigned long long*)&x_4), "l"(*(const unsigned long long*)&s_9));
        float2 _f32x2_mul_a_19 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_9), s_9), g_4);
        float2 mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&u_4));
        float2 mul_1_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_4) : "l"(*(const unsigned long long*)&act_4), "l"(*(const unsigned long long*)&g_4));
        __nv_bfloat162 bf16x2_6 = __float22bfloat162_rn(make_float2(mul_5.x, mul_5.y));
        __nv_bfloat162 bf16x2_1_5 = __float22bfloat162_rn(make_float2(mul_1_4.x, mul_1_4.y));
        dgate[5] = __as_u32(bf16x2_6);
        dup[5] = __as_u32(bf16x2_1_5);
        float2 x_5 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[6]));
        float2 u_5 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[6]));
        float2 g_5 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[6]));
        float2 _f32x2_mul_b_20 = make_float2(-1.442695f, -1.442695f);
        float2 t_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_5) : "l"(*(const unsigned long long*)&x_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
        float e0_5 = approx_exp2(t_5.x);
        float e1_5 = approx_exp2(t_5.y);
        float2 nd_5 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_5, e1_5));
        float d0_5 = -nd_5.x;
        float d1_5 = -nd_5.y;
        float r0_5 = approx_rcp(d0_5);
        float r1_5 = approx_rcp(d1_5);
        float2 r_5 = make_float2(r0_5, r1_5);
        float2 s_12 = fma_f32x2_rn_noftz(r_5, fma_f32x2_rn_noftz(nd_5, r_5, make_float2(1.0f, 1.0f)), r_5);
        float2 s_11 = s_12;
        float2 act_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_5) : "l"(*(const unsigned long long*)&x_5), "l"(*(const unsigned long long*)&s_11));
        float2 _f32x2_mul_a_21 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_11), s_11), g_5);
        float2 mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&u_5));
        float2 mul_1_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_5) : "l"(*(const unsigned long long*)&act_5), "l"(*(const unsigned long long*)&g_5));
        __nv_bfloat162 bf16x2_7 = __float22bfloat162_rn(make_float2(mul_6.x, mul_6.y));
        __nv_bfloat162 bf16x2_1_6 = __float22bfloat162_rn(make_float2(mul_1_5.x, mul_1_5.y));
        dgate[6] = __as_u32(bf16x2_7);
        dup[6] = __as_u32(bf16x2_1_6);
        float2 x_6 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[7]));
        float2 u_6 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[7]));
        float2 g_6 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[7]));
        float2 _f32x2_mul_b_22 = make_float2(-1.442695f, -1.442695f);
        float2 t_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_6) : "l"(*(const unsigned long long*)&x_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_22));
        float e0_6 = approx_exp2(t_6.x);
        float e1_6 = approx_exp2(t_6.y);
        float2 nd_6 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_6, e1_6));
        float d0_6 = -nd_6.x;
        float d1_6 = -nd_6.y;
        float r0_6 = approx_rcp(d0_6);
        float r1_6 = approx_rcp(d1_6);
        float2 r_6 = make_float2(r0_6, r1_6);
        float2 s_14 = fma_f32x2_rn_noftz(r_6, fma_f32x2_rn_noftz(nd_6, r_6, make_float2(1.0f, 1.0f)), r_6);
        float2 s_13 = s_14;
        float2 act_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_6) : "l"(*(const unsigned long long*)&x_6), "l"(*(const unsigned long long*)&s_13));
        float2 _f32x2_mul_a_23 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_13), s_13), g_6);
        float2 mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&u_6));
        float2 mul_1_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_6) : "l"(*(const unsigned long long*)&act_6), "l"(*(const unsigned long long*)&g_6));
        __nv_bfloat162 bf16x2_8 = __float22bfloat162_rn(make_float2(mul_7.x, mul_7.y));
        __nv_bfloat162 bf16x2_1_7 = __float22bfloat162_rn(make_float2(mul_1_6.x, mul_1_6.y));
        dgate[7] = __as_u32(bf16x2_8);
        dup[7] = __as_u32(bf16x2_1_7);
        float2 x_7 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[0]));
        float2 u_7 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[0]));
        float2 g_7 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[0]));
        float2 _f32x2_mul_b_24 = make_float2(-1.442695f, -1.442695f);
        float2 t_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_7) : "l"(*(const unsigned long long*)&x_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_24));
        float e0_7 = approx_exp2(t_7.x);
        float e1_7 = approx_exp2(t_7.y);
        float2 nd_7 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_7, e1_7));
        float d0_7 = -nd_7.x;
        float d1_7 = -nd_7.y;
        float r0_7 = approx_rcp(d0_7);
        float r1_7 = approx_rcp(d1_7);
        float2 r_7 = make_float2(r0_7, r1_7);
        float2 s_16 = fma_f32x2_rn_noftz(r_7, fma_f32x2_rn_noftz(nd_7, r_7, make_float2(1.0f, 1.0f)), r_7);
        float2 s_15 = s_16;
        float2 act_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_7) : "l"(*(const unsigned long long*)&x_7), "l"(*(const unsigned long long*)&s_15));
        float2 _f32x2_mul_a_25 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_7, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_15), s_15), g_7);
        float2 mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_25), "l"(*(const unsigned long long*)&u_7));
        float2 mul_1_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_7) : "l"(*(const unsigned long long*)&act_7), "l"(*(const unsigned long long*)&g_7));
        __nv_bfloat162 bf16x2_9 = __float22bfloat162_rn(make_float2(mul_8.x, mul_8.y));
        __nv_bfloat162 bf16x2_1_8 = __float22bfloat162_rn(make_float2(mul_1_7.x, mul_1_7.y));
        dgate_b[0] = __as_u32(bf16x2_9);
        dup_b[0] = __as_u32(bf16x2_1_8);
        float2 x_8 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[1]));
        float2 u_8 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[1]));
        float2 g_8 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[1]));
        float2 _f32x2_mul_b_26 = make_float2(-1.442695f, -1.442695f);
        float2 t_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_8) : "l"(*(const unsigned long long*)&x_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_26));
        float e0_8 = approx_exp2(t_8.x);
        float e1_8 = approx_exp2(t_8.y);
        float2 nd_8 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_8, e1_8));
        float d0_8 = -nd_8.x;
        float d1_8 = -nd_8.y;
        float r0_8 = approx_rcp(d0_8);
        float r1_8 = approx_rcp(d1_8);
        float2 r_8 = make_float2(r0_8, r1_8);
        float2 s_18 = fma_f32x2_rn_noftz(r_8, fma_f32x2_rn_noftz(nd_8, r_8, make_float2(1.0f, 1.0f)), r_8);
        float2 s_17 = s_18;
        float2 act_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_8) : "l"(*(const unsigned long long*)&x_8), "l"(*(const unsigned long long*)&s_17));
        float2 _f32x2_mul_a_27 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_8, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_17), s_17), g_8);
        float2 mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_27), "l"(*(const unsigned long long*)&u_8));
        float2 mul_1_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_8) : "l"(*(const unsigned long long*)&act_8), "l"(*(const unsigned long long*)&g_8));
        __nv_bfloat162 bf16x2_10 = __float22bfloat162_rn(make_float2(mul_9.x, mul_9.y));
        __nv_bfloat162 bf16x2_1_9 = __float22bfloat162_rn(make_float2(mul_1_8.x, mul_1_8.y));
        dgate_b[1] = __as_u32(bf16x2_10);
        dup_b[1] = __as_u32(bf16x2_1_9);
        float2 x_9 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[2]));
        float2 u_9 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[2]));
        float2 g_9 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[2]));
        float2 _f32x2_mul_b_28 = make_float2(-1.442695f, -1.442695f);
        float2 t_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_9) : "l"(*(const unsigned long long*)&x_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_28));
        float e0_9 = approx_exp2(t_9.x);
        float e1_9 = approx_exp2(t_9.y);
        float2 nd_9 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_9, e1_9));
        float d0_9 = -nd_9.x;
        float d1_9 = -nd_9.y;
        float r0_9 = approx_rcp(d0_9);
        float r1_9 = approx_rcp(d1_9);
        float2 r_9 = make_float2(r0_9, r1_9);
        float2 s_20 = fma_f32x2_rn_noftz(r_9, fma_f32x2_rn_noftz(nd_9, r_9, make_float2(1.0f, 1.0f)), r_9);
        float2 s_19 = s_20;
        float2 act_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_9) : "l"(*(const unsigned long long*)&x_9), "l"(*(const unsigned long long*)&s_19));
        float2 _f32x2_mul_a_29 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_9, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_19), s_19), g_9);
        float2 mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&u_9));
        float2 mul_1_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_9) : "l"(*(const unsigned long long*)&act_9), "l"(*(const unsigned long long*)&g_9));
        __nv_bfloat162 bf16x2_11 = __float22bfloat162_rn(make_float2(mul_10.x, mul_10.y));
        __nv_bfloat162 bf16x2_1_10 = __float22bfloat162_rn(make_float2(mul_1_9.x, mul_1_9.y));
        dgate_b[2] = __as_u32(bf16x2_11);
        dup_b[2] = __as_u32(bf16x2_1_10);
        float2 x_10 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[3]));
        float2 u_10 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[3]));
        float2 g_10 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[3]));
        float2 _f32x2_mul_b_30 = make_float2(-1.442695f, -1.442695f);
        float2 t_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_10) : "l"(*(const unsigned long long*)&x_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_30));
        float e0_10 = approx_exp2(t_10.x);
        float e1_10 = approx_exp2(t_10.y);
        float2 nd_10 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_10, e1_10));
        float d0_10 = -nd_10.x;
        float d1_10 = -nd_10.y;
        float r0_10 = approx_rcp(d0_10);
        float r1_10 = approx_rcp(d1_10);
        float2 r_10 = make_float2(r0_10, r1_10);
        float2 s_22 = fma_f32x2_rn_noftz(r_10, fma_f32x2_rn_noftz(nd_10, r_10, make_float2(1.0f, 1.0f)), r_10);
        float2 s_21 = s_22;
        float2 act_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_10) : "l"(*(const unsigned long long*)&x_10), "l"(*(const unsigned long long*)&s_21));
        float2 _f32x2_mul_a_31 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_10, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_21), s_21), g_10);
        float2 mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_31), "l"(*(const unsigned long long*)&u_10));
        float2 mul_1_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_10) : "l"(*(const unsigned long long*)&act_10), "l"(*(const unsigned long long*)&g_10));
        __nv_bfloat162 bf16x2_12 = __float22bfloat162_rn(make_float2(mul_11.x, mul_11.y));
        __nv_bfloat162 bf16x2_1_11 = __float22bfloat162_rn(make_float2(mul_1_10.x, mul_1_10.y));
        dgate_b[3] = __as_u32(bf16x2_12);
        dup_b[3] = __as_u32(bf16x2_1_11);
        float2 x_11 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[4]));
        float2 u_11 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[4]));
        float2 g_11 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[4]));
        float2 _f32x2_mul_b_32 = make_float2(-1.442695f, -1.442695f);
        float2 t_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_11) : "l"(*(const unsigned long long*)&x_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_32));
        float e0_11 = approx_exp2(t_11.x);
        float e1_11 = approx_exp2(t_11.y);
        float2 nd_11 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_11, e1_11));
        float d0_11 = -nd_11.x;
        float d1_11 = -nd_11.y;
        float r0_11 = approx_rcp(d0_11);
        float r1_11 = approx_rcp(d1_11);
        float2 r_11 = make_float2(r0_11, r1_11);
        float2 s_24 = fma_f32x2_rn_noftz(r_11, fma_f32x2_rn_noftz(nd_11, r_11, make_float2(1.0f, 1.0f)), r_11);
        float2 s_23 = s_24;
        float2 act_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_11) : "l"(*(const unsigned long long*)&x_11), "l"(*(const unsigned long long*)&s_23));
        float2 _f32x2_mul_a_33 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_11, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_23), s_23), g_11);
        float2 mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&u_11));
        float2 mul_1_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_11) : "l"(*(const unsigned long long*)&act_11), "l"(*(const unsigned long long*)&g_11));
        __nv_bfloat162 bf16x2_13 = __float22bfloat162_rn(make_float2(mul_12.x, mul_12.y));
        __nv_bfloat162 bf16x2_1_12 = __float22bfloat162_rn(make_float2(mul_1_11.x, mul_1_11.y));
        dgate_b[4] = __as_u32(bf16x2_13);
        dup_b[4] = __as_u32(bf16x2_1_12);
        float2 x_12 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[5]));
        float2 u_12 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[5]));
        float2 g_12 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[5]));
        float2 _f32x2_mul_b_34 = make_float2(-1.442695f, -1.442695f);
        float2 t_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_12) : "l"(*(const unsigned long long*)&x_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_34));
        float e0_12 = approx_exp2(t_12.x);
        float e1_12 = approx_exp2(t_12.y);
        float2 nd_12 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_12, e1_12));
        float d0_12 = -nd_12.x;
        float d1_12 = -nd_12.y;
        float r0_12 = approx_rcp(d0_12);
        float r1_12 = approx_rcp(d1_12);
        float2 r_12 = make_float2(r0_12, r1_12);
        float2 s_26 = fma_f32x2_rn_noftz(r_12, fma_f32x2_rn_noftz(nd_12, r_12, make_float2(1.0f, 1.0f)), r_12);
        float2 s_25 = s_26;
        float2 act_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_12) : "l"(*(const unsigned long long*)&x_12), "l"(*(const unsigned long long*)&s_25));
        float2 _f32x2_mul_a_35 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_12, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_25), s_25), g_12);
        float2 mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&u_12));
        float2 mul_1_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_12) : "l"(*(const unsigned long long*)&act_12), "l"(*(const unsigned long long*)&g_12));
        __nv_bfloat162 bf16x2_14 = __float22bfloat162_rn(make_float2(mul_13.x, mul_13.y));
        __nv_bfloat162 bf16x2_1_13 = __float22bfloat162_rn(make_float2(mul_1_12.x, mul_1_12.y));
        dgate_b[5] = __as_u32(bf16x2_14);
        dup_b[5] = __as_u32(bf16x2_1_13);
        float2 x_13 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[6]));
        float2 u_13 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[6]));
        float2 g_13 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[6]));
        float2 _f32x2_mul_b_36 = make_float2(-1.442695f, -1.442695f);
        float2 t_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_13) : "l"(*(const unsigned long long*)&x_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_36));
        float e0_13 = approx_exp2(t_13.x);
        float e1_13 = approx_exp2(t_13.y);
        float2 nd_13 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_13, e1_13));
        float d0_13 = -nd_13.x;
        float d1_13 = -nd_13.y;
        float r0_13 = approx_rcp(d0_13);
        float r1_13 = approx_rcp(d1_13);
        float2 r_13 = make_float2(r0_13, r1_13);
        float2 s_28 = fma_f32x2_rn_noftz(r_13, fma_f32x2_rn_noftz(nd_13, r_13, make_float2(1.0f, 1.0f)), r_13);
        float2 s_27 = s_28;
        float2 act_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_13) : "l"(*(const unsigned long long*)&x_13), "l"(*(const unsigned long long*)&s_27));
        float2 _f32x2_mul_a_37 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_13, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_27), s_27), g_13);
        float2 mul_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_37), "l"(*(const unsigned long long*)&u_13));
        float2 mul_1_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_13) : "l"(*(const unsigned long long*)&act_13), "l"(*(const unsigned long long*)&g_13));
        __nv_bfloat162 bf16x2_15 = __float22bfloat162_rn(make_float2(mul_14.x, mul_14.y));
        __nv_bfloat162 bf16x2_1_14 = __float22bfloat162_rn(make_float2(mul_1_13.x, mul_1_13.y));
        dgate_b[6] = __as_u32(bf16x2_15);
        dup_b[6] = __as_u32(bf16x2_1_14);
        float2 x_14 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[7]));
        float2 u_14 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[7]));
        float2 g_14 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[7]));
        float2 _f32x2_mul_b_38 = make_float2(-1.442695f, -1.442695f);
        float2 t_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_14) : "l"(*(const unsigned long long*)&x_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_38));
        float e0_14 = approx_exp2(t_14.x);
        float e1_14 = approx_exp2(t_14.y);
        float2 nd_14 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_14, e1_14));
        float d0_14 = -nd_14.x;
        float d1_14 = -nd_14.y;
        float r0_14 = approx_rcp(d0_14);
        float r1_14 = approx_rcp(d1_14);
        float2 r_14 = make_float2(r0_14, r1_14);
        float2 s_30 = fma_f32x2_rn_noftz(r_14, fma_f32x2_rn_noftz(nd_14, r_14, make_float2(1.0f, 1.0f)), r_14);
        float2 s_29 = s_30;
        float2 act_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_14) : "l"(*(const unsigned long long*)&x_14), "l"(*(const unsigned long long*)&s_29));
        float2 _f32x2_mul_a_39 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_14, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_29), s_29), g_14);
        float2 mul_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_39), "l"(*(const unsigned long long*)&u_14));
        float2 mul_1_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_14) : "l"(*(const unsigned long long*)&act_14), "l"(*(const unsigned long long*)&g_14));
        __nv_bfloat162 bf16x2_16 = __float22bfloat162_rn(make_float2(mul_15.x, mul_15.y));
        __nv_bfloat162 bf16x2_1_15 = __float22bfloat162_rn(make_float2(mul_1_14.x, mul_1_14.y));
        dgate_b[7] = __as_u32(bf16x2_16);
        dup_b[7] = __as_u32(bf16x2_1_15);
    } else {
        int band = 0;
        float2 x_15 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[0]));
        float2 u_15 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[0]));
        float2 g_15 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[0]));
        float2 _f32x2_mul_b_40 = make_float2(-1.442695f, -1.442695f);
        float2 t_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_15) : "l"(*(const unsigned long long*)&x_15), "l"(*(const unsigned long long*)&_f32x2_mul_b_40));
        float e0_15 = approx_exp2(t_15.x);
        float e1_15 = approx_exp2(t_15.y);
        float2 nd_15 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_15, e1_15));
        float d0_15 = -nd_15.x;
        float d1_15 = -nd_15.y;
        float dn;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn) : "f"(e0_15), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_15 = approx_rcp(dn);
        float _fma_41;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(-dn), "f"(r_15), "f"(1.0f));
        float q;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q) : "f"(r_15), "f"(_fma_41), "f"(r_15));
        float s0;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0) : "f"(q), "f"(1.1754944e-38f));
        float err;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err) : "f"(s0), "f"(-8.507059e+37f), "f"(q));
        float y2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2) : "f"(-dn), "f"(q), "f"(1.0f));
        float nudge = __uint_as_float(__as_u32(y2) & 2147483648u | 729808896);
        float _add_42;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(err), "f"(nudge));
        float u0;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0) : "f"(_add_42), "f"(1.1754944e-38f), "f"(s0));
        float dn_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_0) : "f"(e1_15), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_15_1 = approx_rcp(dn_0);
        float _fma_43;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(-dn_0), "f"(r_15_1), "f"(1.0f));
        float q_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_0) : "f"(r_15_1), "f"(_fma_43), "f"(r_15_1));
        float s0_0;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_0) : "f"(q_0), "f"(1.1754944e-38f));
        float err_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_0) : "f"(s0_0), "f"(-8.507059e+37f), "f"(q_0));
        float y2_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_0) : "f"(-dn_0), "f"(q_0), "f"(1.0f));
        float nudge_0 = __uint_as_float(__as_u32(y2_0) & 2147483648u | 729808896);
        float _add_44;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(err_0), "f"(nudge_0));
        float u1;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1) : "f"(_add_44), "f"(1.1754944e-38f), "f"(s0_0));
        float2 s_31 = make_float2(u0, u1);
        float2 act_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_15) : "l"(*(const unsigned long long*)&x_15), "l"(*(const unsigned long long*)&s_31));
        float2 _f32x2_mul_a_45 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_15, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_31), s_31), g_15);
        float2 mul_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_45), "l"(*(const unsigned long long*)&u_15));
        float2 mul_1_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_15) : "l"(*(const unsigned long long*)&act_15), "l"(*(const unsigned long long*)&g_15));
        __nv_bfloat162 bf16x2_17 = __float22bfloat162_rn(make_float2(mul_16.x, mul_16.y));
        __nv_bfloat162 bf16x2_1_16 = __float22bfloat162_rn(make_float2(mul_1_15.x, mul_1_15.y));
        dgate[0] = __as_u32(bf16x2_17);
        dup[0] = __as_u32(bf16x2_1_16);
        float2 x_15_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[1]));
        float2 u_15_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[1]));
        float2 g_15_1 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[1]));
        float2 _f32x2_mul_b_46 = make_float2(-1.442695f, -1.442695f);
        float2 t_15_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_15_1) : "l"(*(const unsigned long long*)&x_15_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_46));
        float e0_15_1 = approx_exp2(t_15_1.x);
        float e1_15_1 = approx_exp2(t_15_1.y);
        float2 nd_15_1 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_15_1, e1_15_1));
        float d0_15_1 = -nd_15_1.x;
        float d1_15_1 = -nd_15_1.y;
        float dn_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_1) : "f"(e0_15_1), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_16 = approx_rcp(dn_1);
        float _fma_47;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(-dn_1), "f"(r_16), "f"(1.0f));
        float q_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_1) : "f"(r_16), "f"(_fma_47), "f"(r_16));
        float s0_1;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_1) : "f"(q_1), "f"(1.1754944e-38f));
        float err_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_1) : "f"(s0_1), "f"(-8.507059e+37f), "f"(q_1));
        float y2_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_1) : "f"(-dn_1), "f"(q_1), "f"(1.0f));
        float nudge_1 = __uint_as_float(__as_u32(y2_1) & 2147483648u | 729808896);
        float _add_48;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(err_1), "f"(nudge_1));
        float u0_0;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0_0) : "f"(_add_48), "f"(1.1754944e-38f), "f"(s0_1));
        float dn_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_2) : "f"(e1_15_1), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_17 = approx_rcp(dn_2);
        float _fma_49;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(-dn_2), "f"(r_17), "f"(1.0f));
        float q_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_2) : "f"(r_17), "f"(_fma_49), "f"(r_17));
        float s0_2;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_2) : "f"(q_2), "f"(1.1754944e-38f));
        float err_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_2) : "f"(s0_2), "f"(-8.507059e+37f), "f"(q_2));
        float y2_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_2) : "f"(-dn_2), "f"(q_2), "f"(1.0f));
        float nudge_2 = __uint_as_float(__as_u32(y2_2) & 2147483648u | 729808896);
        float _add_50;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_50) : "f"(err_2), "f"(nudge_2));
        float u1_0;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1_0) : "f"(_add_50), "f"(1.1754944e-38f), "f"(s0_2));
        float2 s_31_1 = make_float2(u0_0, u1_0);
        float2 act_15_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_15_1) : "l"(*(const unsigned long long*)&x_15_1), "l"(*(const unsigned long long*)&s_31_1));
        float2 _f32x2_mul_a_51 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_15_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_31_1), s_31_1), g_15_1);
        float2 mul_16_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_16_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_51), "l"(*(const unsigned long long*)&u_15_1));
        float2 mul_1_15_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_15_1) : "l"(*(const unsigned long long*)&act_15_1), "l"(*(const unsigned long long*)&g_15_1));
        __nv_bfloat162 bf16x2_18 = __float22bfloat162_rn(make_float2(mul_16_1.x, mul_16_1.y));
        __nv_bfloat162 bf16x2_1_17 = __float22bfloat162_rn(make_float2(mul_1_15_1.x, mul_1_15_1.y));
        dgate[1] = __as_u32(bf16x2_18);
        dup[1] = __as_u32(bf16x2_1_17);
        float2 x_16 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[2]));
        float2 u_16 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[2]));
        float2 g_16 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[2]));
        float2 _f32x2_mul_b_52 = make_float2(-1.442695f, -1.442695f);
        float2 t_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_16) : "l"(*(const unsigned long long*)&x_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_52));
        float e0_16 = approx_exp2(t_16.x);
        float e1_16 = approx_exp2(t_16.y);
        float2 nd_16 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_16, e1_16));
        float d0_16 = -nd_16.x;
        float d1_16 = -nd_16.y;
        float dn_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_3) : "f"(e0_16), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_18 = approx_rcp(dn_3);
        float _fma_53;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(-dn_3), "f"(r_18), "f"(1.0f));
        float q_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_3) : "f"(r_18), "f"(_fma_53), "f"(r_18));
        float s0_3;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_3) : "f"(q_3), "f"(1.1754944e-38f));
        float err_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_3) : "f"(s0_3), "f"(-8.507059e+37f), "f"(q_3));
        float y2_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_3) : "f"(-dn_3), "f"(q_3), "f"(1.0f));
        float nudge_3 = __uint_as_float(__as_u32(y2_3) & 2147483648u | 729808896);
        float _add_54;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_54) : "f"(err_3), "f"(nudge_3));
        float u0_1;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0_1) : "f"(_add_54), "f"(1.1754944e-38f), "f"(s0_3));
        float dn_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_4) : "f"(e1_16), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_19 = approx_rcp(dn_4);
        float _fma_55;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(-dn_4), "f"(r_19), "f"(1.0f));
        float q_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_4) : "f"(r_19), "f"(_fma_55), "f"(r_19));
        float s0_4;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_4) : "f"(q_4), "f"(1.1754944e-38f));
        float err_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_4) : "f"(s0_4), "f"(-8.507059e+37f), "f"(q_4));
        float y2_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_4) : "f"(-dn_4), "f"(q_4), "f"(1.0f));
        float nudge_4 = __uint_as_float(__as_u32(y2_4) & 2147483648u | 729808896);
        float _add_56;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_56) : "f"(err_4), "f"(nudge_4));
        float u1_1;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1_1) : "f"(_add_56), "f"(1.1754944e-38f), "f"(s0_4));
        float2 s_32 = make_float2(u0_1, u1_1);
        float2 act_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_16) : "l"(*(const unsigned long long*)&x_16), "l"(*(const unsigned long long*)&s_32));
        float2 _f32x2_mul_a_57 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_16, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_32), s_32), g_16);
        float2 mul_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_57), "l"(*(const unsigned long long*)&u_16));
        float2 mul_1_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_16) : "l"(*(const unsigned long long*)&act_16), "l"(*(const unsigned long long*)&g_16));
        __nv_bfloat162 bf16x2_19 = __float22bfloat162_rn(make_float2(mul_17.x, mul_17.y));
        __nv_bfloat162 bf16x2_1_18 = __float22bfloat162_rn(make_float2(mul_1_16.x, mul_1_16.y));
        dgate[2] = __as_u32(bf16x2_19);
        dup[2] = __as_u32(bf16x2_1_18);
        float2 x_17 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[3]));
        float2 u_17 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[3]));
        float2 g_17 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[3]));
        float2 _f32x2_mul_b_58 = make_float2(-1.442695f, -1.442695f);
        float2 t_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_17) : "l"(*(const unsigned long long*)&x_17), "l"(*(const unsigned long long*)&_f32x2_mul_b_58));
        float e0_17 = approx_exp2(t_17.x);
        float e1_17 = approx_exp2(t_17.y);
        float2 nd_17 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_17, e1_17));
        float d0_17 = -nd_17.x;
        float d1_17 = -nd_17.y;
        float dn_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_5) : "f"(e0_17), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_20 = approx_rcp(dn_5);
        float _fma_59;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(-dn_5), "f"(r_20), "f"(1.0f));
        float q_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_5) : "f"(r_20), "f"(_fma_59), "f"(r_20));
        float s0_5;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_5) : "f"(q_5), "f"(1.1754944e-38f));
        float err_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_5) : "f"(s0_5), "f"(-8.507059e+37f), "f"(q_5));
        float y2_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_5) : "f"(-dn_5), "f"(q_5), "f"(1.0f));
        float nudge_5 = __uint_as_float(__as_u32(y2_5) & 2147483648u | 729808896);
        float _add_60;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_60) : "f"(err_5), "f"(nudge_5));
        float u0_2;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0_2) : "f"(_add_60), "f"(1.1754944e-38f), "f"(s0_5));
        float dn_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_6) : "f"(e1_17), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_21 = approx_rcp(dn_6);
        float _fma_61;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(-dn_6), "f"(r_21), "f"(1.0f));
        float q_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_6) : "f"(r_21), "f"(_fma_61), "f"(r_21));
        float s0_6;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_6) : "f"(q_6), "f"(1.1754944e-38f));
        float err_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_6) : "f"(s0_6), "f"(-8.507059e+37f), "f"(q_6));
        float y2_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_6) : "f"(-dn_6), "f"(q_6), "f"(1.0f));
        float nudge_6 = __uint_as_float(__as_u32(y2_6) & 2147483648u | 729808896);
        float _add_62;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_62) : "f"(err_6), "f"(nudge_6));
        float u1_2;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1_2) : "f"(_add_62), "f"(1.1754944e-38f), "f"(s0_6));
        float2 s_33 = make_float2(u0_2, u1_2);
        float2 act_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_17) : "l"(*(const unsigned long long*)&x_17), "l"(*(const unsigned long long*)&s_33));
        float2 _f32x2_mul_a_63 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_17, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_33), s_33), g_17);
        float2 mul_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_63), "l"(*(const unsigned long long*)&u_17));
        float2 mul_1_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_17) : "l"(*(const unsigned long long*)&act_17), "l"(*(const unsigned long long*)&g_17));
        __nv_bfloat162 bf16x2_20 = __float22bfloat162_rn(make_float2(mul_18.x, mul_18.y));
        __nv_bfloat162 bf16x2_1_19 = __float22bfloat162_rn(make_float2(mul_1_17.x, mul_1_17.y));
        dgate[3] = __as_u32(bf16x2_20);
        dup[3] = __as_u32(bf16x2_1_19);
        float2 x_18 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[4]));
        float2 u_18 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[4]));
        float2 g_18 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[4]));
        float2 _f32x2_mul_b_64 = make_float2(-1.442695f, -1.442695f);
        float2 t_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_18) : "l"(*(const unsigned long long*)&x_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_64));
        float e0_18 = approx_exp2(t_18.x);
        float e1_18 = approx_exp2(t_18.y);
        float2 nd_18 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_18, e1_18));
        float d0_18 = -nd_18.x;
        float d1_18 = -nd_18.y;
        float dn_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_7) : "f"(e0_18), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_22 = approx_rcp(dn_7);
        float _fma_65;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(-dn_7), "f"(r_22), "f"(1.0f));
        float q_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_7) : "f"(r_22), "f"(_fma_65), "f"(r_22));
        float s0_7;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_7) : "f"(q_7), "f"(1.1754944e-38f));
        float err_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_7) : "f"(s0_7), "f"(-8.507059e+37f), "f"(q_7));
        float y2_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_7) : "f"(-dn_7), "f"(q_7), "f"(1.0f));
        float nudge_7 = __uint_as_float(__as_u32(y2_7) & 2147483648u | 729808896);
        float _add_66;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_66) : "f"(err_7), "f"(nudge_7));
        float u0_3;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0_3) : "f"(_add_66), "f"(1.1754944e-38f), "f"(s0_7));
        float dn_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_8) : "f"(e1_18), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_23 = approx_rcp(dn_8);
        float _fma_67;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(-dn_8), "f"(r_23), "f"(1.0f));
        float q_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_8) : "f"(r_23), "f"(_fma_67), "f"(r_23));
        float s0_8;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_8) : "f"(q_8), "f"(1.1754944e-38f));
        float err_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_8) : "f"(s0_8), "f"(-8.507059e+37f), "f"(q_8));
        float y2_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_8) : "f"(-dn_8), "f"(q_8), "f"(1.0f));
        float nudge_8 = __uint_as_float(__as_u32(y2_8) & 2147483648u | 729808896);
        float _add_68;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(err_8), "f"(nudge_8));
        float u1_3;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1_3) : "f"(_add_68), "f"(1.1754944e-38f), "f"(s0_8));
        float2 s_34 = make_float2(u0_3, u1_3);
        float2 act_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_18) : "l"(*(const unsigned long long*)&x_18), "l"(*(const unsigned long long*)&s_34));
        float2 _f32x2_mul_a_69 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_18, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_34), s_34), g_18);
        float2 mul_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_69), "l"(*(const unsigned long long*)&u_18));
        float2 mul_1_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_18) : "l"(*(const unsigned long long*)&act_18), "l"(*(const unsigned long long*)&g_18));
        __nv_bfloat162 bf16x2_21 = __float22bfloat162_rn(make_float2(mul_19.x, mul_19.y));
        __nv_bfloat162 bf16x2_1_20 = __float22bfloat162_rn(make_float2(mul_1_18.x, mul_1_18.y));
        dgate[4] = __as_u32(bf16x2_21);
        dup[4] = __as_u32(bf16x2_1_20);
        float2 x_19 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[5]));
        float2 u_19 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[5]));
        float2 g_19 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[5]));
        float2 _f32x2_mul_b_70 = make_float2(-1.442695f, -1.442695f);
        float2 t_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_19) : "l"(*(const unsigned long long*)&x_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_70));
        float e0_19 = approx_exp2(t_19.x);
        float e1_19 = approx_exp2(t_19.y);
        float2 nd_19 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_19, e1_19));
        float d0_19 = -nd_19.x;
        float d1_19 = -nd_19.y;
        float dn_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_9) : "f"(e0_19), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_24 = approx_rcp(dn_9);
        float _fma_71;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(-dn_9), "f"(r_24), "f"(1.0f));
        float q_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_9) : "f"(r_24), "f"(_fma_71), "f"(r_24));
        float s0_9;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_9) : "f"(q_9), "f"(1.1754944e-38f));
        float err_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_9) : "f"(s0_9), "f"(-8.507059e+37f), "f"(q_9));
        float y2_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_9) : "f"(-dn_9), "f"(q_9), "f"(1.0f));
        float nudge_9 = __uint_as_float(__as_u32(y2_9) & 2147483648u | 729808896);
        float _add_72;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_72) : "f"(err_9), "f"(nudge_9));
        float u0_4;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0_4) : "f"(_add_72), "f"(1.1754944e-38f), "f"(s0_9));
        float dn_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_10) : "f"(e1_19), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_25 = approx_rcp(dn_10);
        float _fma_73;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(-dn_10), "f"(r_25), "f"(1.0f));
        float q_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_10) : "f"(r_25), "f"(_fma_73), "f"(r_25));
        float s0_10;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_10) : "f"(q_10), "f"(1.1754944e-38f));
        float err_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_10) : "f"(s0_10), "f"(-8.507059e+37f), "f"(q_10));
        float y2_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_10) : "f"(-dn_10), "f"(q_10), "f"(1.0f));
        float nudge_10 = __uint_as_float(__as_u32(y2_10) & 2147483648u | 729808896);
        float _add_74;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_74) : "f"(err_10), "f"(nudge_10));
        float u1_4;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1_4) : "f"(_add_74), "f"(1.1754944e-38f), "f"(s0_10));
        float2 s_35 = make_float2(u0_4, u1_4);
        float2 act_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_19) : "l"(*(const unsigned long long*)&x_19), "l"(*(const unsigned long long*)&s_35));
        float2 _f32x2_mul_a_75 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_19, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_35), s_35), g_19);
        float2 mul_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_75), "l"(*(const unsigned long long*)&u_19));
        float2 mul_1_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_19) : "l"(*(const unsigned long long*)&act_19), "l"(*(const unsigned long long*)&g_19));
        __nv_bfloat162 bf16x2_22 = __float22bfloat162_rn(make_float2(mul_20.x, mul_20.y));
        __nv_bfloat162 bf16x2_1_21 = __float22bfloat162_rn(make_float2(mul_1_19.x, mul_1_19.y));
        dgate[5] = __as_u32(bf16x2_22);
        dup[5] = __as_u32(bf16x2_1_21);
        float2 x_20 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[6]));
        float2 u_20 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[6]));
        float2 g_20 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[6]));
        float2 _f32x2_mul_b_76 = make_float2(-1.442695f, -1.442695f);
        float2 t_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_20) : "l"(*(const unsigned long long*)&x_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_76));
        float e0_20 = approx_exp2(t_20.x);
        float e1_20 = approx_exp2(t_20.y);
        float2 nd_20 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_20, e1_20));
        float d0_20 = -nd_20.x;
        float d1_20 = -nd_20.y;
        float dn_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_11) : "f"(e0_20), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_26 = approx_rcp(dn_11);
        float _fma_77;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_77) : "f"(-dn_11), "f"(r_26), "f"(1.0f));
        float q_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_11) : "f"(r_26), "f"(_fma_77), "f"(r_26));
        float s0_11;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_11) : "f"(q_11), "f"(1.1754944e-38f));
        float err_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_11) : "f"(s0_11), "f"(-8.507059e+37f), "f"(q_11));
        float y2_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_11) : "f"(-dn_11), "f"(q_11), "f"(1.0f));
        float nudge_11 = __uint_as_float(__as_u32(y2_11) & 2147483648u | 729808896);
        float _add_78;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_78) : "f"(err_11), "f"(nudge_11));
        float u0_5;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0_5) : "f"(_add_78), "f"(1.1754944e-38f), "f"(s0_11));
        float dn_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_12) : "f"(e1_20), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_27 = approx_rcp(dn_12);
        float _fma_79;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(-dn_12), "f"(r_27), "f"(1.0f));
        float q_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_12) : "f"(r_27), "f"(_fma_79), "f"(r_27));
        float s0_12;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_12) : "f"(q_12), "f"(1.1754944e-38f));
        float err_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_12) : "f"(s0_12), "f"(-8.507059e+37f), "f"(q_12));
        float y2_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_12) : "f"(-dn_12), "f"(q_12), "f"(1.0f));
        float nudge_12 = __uint_as_float(__as_u32(y2_12) & 2147483648u | 729808896);
        float _add_80;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_80) : "f"(err_12), "f"(nudge_12));
        float u1_5;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1_5) : "f"(_add_80), "f"(1.1754944e-38f), "f"(s0_12));
        float2 s_36 = make_float2(u0_5, u1_5);
        float2 act_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_20) : "l"(*(const unsigned long long*)&x_20), "l"(*(const unsigned long long*)&s_36));
        float2 _f32x2_mul_a_81 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_20, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_36), s_36), g_20);
        float2 mul_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_81), "l"(*(const unsigned long long*)&u_20));
        float2 mul_1_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_20) : "l"(*(const unsigned long long*)&act_20), "l"(*(const unsigned long long*)&g_20));
        __nv_bfloat162 bf16x2_23 = __float22bfloat162_rn(make_float2(mul_21.x, mul_21.y));
        __nv_bfloat162 bf16x2_1_22 = __float22bfloat162_rn(make_float2(mul_1_20.x, mul_1_20.y));
        dgate[6] = __as_u32(bf16x2_23);
        dup[6] = __as_u32(bf16x2_1_22);
        float2 x_21 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[7]));
        float2 u_21 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[7]));
        float2 g_21 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[7]));
        float2 _f32x2_mul_b_82 = make_float2(-1.442695f, -1.442695f);
        float2 t_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_21) : "l"(*(const unsigned long long*)&x_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_82));
        float e0_21 = approx_exp2(t_21.x);
        float e1_21 = approx_exp2(t_21.y);
        float2 nd_21 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_21, e1_21));
        float d0_21 = -nd_21.x;
        float d1_21 = -nd_21.y;
        float dn_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_13) : "f"(e0_21), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_28 = approx_rcp(dn_13);
        float _fma_83;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(-dn_13), "f"(r_28), "f"(1.0f));
        float q_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_13) : "f"(r_28), "f"(_fma_83), "f"(r_28));
        float s0_13;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_13) : "f"(q_13), "f"(1.1754944e-38f));
        float err_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_13) : "f"(s0_13), "f"(-8.507059e+37f), "f"(q_13));
        float y2_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_13) : "f"(-dn_13), "f"(q_13), "f"(1.0f));
        float nudge_13 = __uint_as_float(__as_u32(y2_13) & 2147483648u | 729808896);
        float _add_84;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_84) : "f"(err_13), "f"(nudge_13));
        float u0_6;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0_6) : "f"(_add_84), "f"(1.1754944e-38f), "f"(s0_13));
        float dn_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_14) : "f"(e1_21), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_29 = approx_rcp(dn_14);
        float _fma_85;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(-dn_14), "f"(r_29), "f"(1.0f));
        float q_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_14) : "f"(r_29), "f"(_fma_85), "f"(r_29));
        float s0_14;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_14) : "f"(q_14), "f"(1.1754944e-38f));
        float err_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_14) : "f"(s0_14), "f"(-8.507059e+37f), "f"(q_14));
        float y2_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_14) : "f"(-dn_14), "f"(q_14), "f"(1.0f));
        float nudge_14 = __uint_as_float(__as_u32(y2_14) & 2147483648u | 729808896);
        float _add_86;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_86) : "f"(err_14), "f"(nudge_14));
        float u1_6;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1_6) : "f"(_add_86), "f"(1.1754944e-38f), "f"(s0_14));
        float2 s_37 = make_float2(u0_6, u1_6);
        float2 act_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_21) : "l"(*(const unsigned long long*)&x_21), "l"(*(const unsigned long long*)&s_37));
        float2 _f32x2_mul_a_87 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_21, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_37), s_37), g_21);
        float2 mul_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_87), "l"(*(const unsigned long long*)&u_21));
        float2 mul_1_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_21) : "l"(*(const unsigned long long*)&act_21), "l"(*(const unsigned long long*)&g_21));
        __nv_bfloat162 bf16x2_24 = __float22bfloat162_rn(make_float2(mul_22.x, mul_22.y));
        __nv_bfloat162 bf16x2_1_23 = __float22bfloat162_rn(make_float2(mul_1_21.x, mul_1_21.y));
        dgate[7] = __as_u32(bf16x2_24);
        dup[7] = __as_u32(bf16x2_1_23);
        int band_b = 0;
        float2 x_22 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[0]));
        float2 u_22 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[0]));
        float2 g_22 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[0]));
        float2 _f32x2_mul_b_88 = make_float2(-1.442695f, -1.442695f);
        float2 t_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_22) : "l"(*(const unsigned long long*)&x_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_88));
        float e0_22 = approx_exp2(t_22.x);
        float e1_22 = approx_exp2(t_22.y);
        float2 nd_22 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_22, e1_22));
        float d0_22 = -nd_22.x;
        float d1_22 = -nd_22.y;
        float dn_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_15) : "f"(e0_22), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_30 = approx_rcp(dn_15);
        float _fma_89;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(-dn_15), "f"(r_30), "f"(1.0f));
        float q_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_15) : "f"(r_30), "f"(_fma_89), "f"(r_30));
        float s0_15;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_15) : "f"(q_15), "f"(1.1754944e-38f));
        float err_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_15) : "f"(s0_15), "f"(-8.507059e+37f), "f"(q_15));
        float y2_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_15) : "f"(-dn_15), "f"(q_15), "f"(1.0f));
        float nudge_15 = __uint_as_float(__as_u32(y2_15) & 2147483648u | 729808896);
        float _add_90;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_90) : "f"(err_15), "f"(nudge_15));
        float u0_7;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0_7) : "f"(_add_90), "f"(1.1754944e-38f), "f"(s0_15));
        float dn_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_16) : "f"(e1_22), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_31 = approx_rcp(dn_16);
        float _fma_91;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(-dn_16), "f"(r_31), "f"(1.0f));
        float q_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_16) : "f"(r_31), "f"(_fma_91), "f"(r_31));
        float s0_16;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_16) : "f"(q_16), "f"(1.1754944e-38f));
        float err_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_16) : "f"(s0_16), "f"(-8.507059e+37f), "f"(q_16));
        float y2_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_16) : "f"(-dn_16), "f"(q_16), "f"(1.0f));
        float nudge_16 = __uint_as_float(__as_u32(y2_16) & 2147483648u | 729808896);
        float _add_92;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(err_16), "f"(nudge_16));
        float u1_7;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1_7) : "f"(_add_92), "f"(1.1754944e-38f), "f"(s0_16));
        float2 s_38 = make_float2(u0_7, u1_7);
        float2 act_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_22) : "l"(*(const unsigned long long*)&x_22), "l"(*(const unsigned long long*)&s_38));
        float2 _f32x2_mul_a_93 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_22, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_38), s_38), g_22);
        float2 mul_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_93), "l"(*(const unsigned long long*)&u_22));
        float2 mul_1_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_22) : "l"(*(const unsigned long long*)&act_22), "l"(*(const unsigned long long*)&g_22));
        __nv_bfloat162 bf16x2_25 = __float22bfloat162_rn(make_float2(mul_23.x, mul_23.y));
        __nv_bfloat162 bf16x2_1_24 = __float22bfloat162_rn(make_float2(mul_1_22.x, mul_1_22.y));
        dgate_b[0] = __as_u32(bf16x2_25);
        dup_b[0] = __as_u32(bf16x2_1_24);
        float2 x_23 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[1]));
        float2 u_23 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[1]));
        float2 g_23 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[1]));
        float2 _f32x2_mul_b_94 = make_float2(-1.442695f, -1.442695f);
        float2 t_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_23) : "l"(*(const unsigned long long*)&x_23), "l"(*(const unsigned long long*)&_f32x2_mul_b_94));
        float e0_23 = approx_exp2(t_23.x);
        float e1_23 = approx_exp2(t_23.y);
        float2 nd_23 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_23, e1_23));
        float d0_23 = -nd_23.x;
        float d1_23 = -nd_23.y;
        float dn_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_17) : "f"(e0_23), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_32 = approx_rcp(dn_17);
        float _fma_95;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(-dn_17), "f"(r_32), "f"(1.0f));
        float q_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_17) : "f"(r_32), "f"(_fma_95), "f"(r_32));
        float s0_17;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_17) : "f"(q_17), "f"(1.1754944e-38f));
        float err_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_17) : "f"(s0_17), "f"(-8.507059e+37f), "f"(q_17));
        float y2_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_17) : "f"(-dn_17), "f"(q_17), "f"(1.0f));
        float nudge_17 = __uint_as_float(__as_u32(y2_17) & 2147483648u | 729808896);
        float _add_96;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_96) : "f"(err_17), "f"(nudge_17));
        float u0_8;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0_8) : "f"(_add_96), "f"(1.1754944e-38f), "f"(s0_17));
        float dn_18;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_18) : "f"(e1_23), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_33 = approx_rcp(dn_18);
        float _fma_97;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(-dn_18), "f"(r_33), "f"(1.0f));
        float q_18;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_18) : "f"(r_33), "f"(_fma_97), "f"(r_33));
        float s0_18;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_18) : "f"(q_18), "f"(1.1754944e-38f));
        float err_18;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_18) : "f"(s0_18), "f"(-8.507059e+37f), "f"(q_18));
        float y2_18;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_18) : "f"(-dn_18), "f"(q_18), "f"(1.0f));
        float nudge_18 = __uint_as_float(__as_u32(y2_18) & 2147483648u | 729808896);
        float _add_98;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_98) : "f"(err_18), "f"(nudge_18));
        float u1_8;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1_8) : "f"(_add_98), "f"(1.1754944e-38f), "f"(s0_18));
        float2 s_39 = make_float2(u0_8, u1_8);
        float2 act_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_23) : "l"(*(const unsigned long long*)&x_23), "l"(*(const unsigned long long*)&s_39));
        float2 _f32x2_mul_a_99 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_23, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_39), s_39), g_23);
        float2 mul_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_99), "l"(*(const unsigned long long*)&u_23));
        float2 mul_1_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_23) : "l"(*(const unsigned long long*)&act_23), "l"(*(const unsigned long long*)&g_23));
        __nv_bfloat162 bf16x2_26 = __float22bfloat162_rn(make_float2(mul_24.x, mul_24.y));
        __nv_bfloat162 bf16x2_1_25 = __float22bfloat162_rn(make_float2(mul_1_23.x, mul_1_23.y));
        dgate_b[1] = __as_u32(bf16x2_26);
        dup_b[1] = __as_u32(bf16x2_1_25);
        float2 x_24 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[2]));
        float2 u_24 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[2]));
        float2 g_24 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[2]));
        float2 _f32x2_mul_b_100 = make_float2(-1.442695f, -1.442695f);
        float2 t_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_24) : "l"(*(const unsigned long long*)&x_24), "l"(*(const unsigned long long*)&_f32x2_mul_b_100));
        float e0_24 = approx_exp2(t_24.x);
        float e1_24 = approx_exp2(t_24.y);
        float2 nd_24 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_24, e1_24));
        float d0_24 = -nd_24.x;
        float d1_24 = -nd_24.y;
        float dn_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_19) : "f"(e0_24), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_34 = approx_rcp(dn_19);
        float _fma_101;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(-dn_19), "f"(r_34), "f"(1.0f));
        float q_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_19) : "f"(r_34), "f"(_fma_101), "f"(r_34));
        float s0_19;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_19) : "f"(q_19), "f"(1.1754944e-38f));
        float err_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_19) : "f"(s0_19), "f"(-8.507059e+37f), "f"(q_19));
        float y2_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_19) : "f"(-dn_19), "f"(q_19), "f"(1.0f));
        float nudge_19 = __uint_as_float(__as_u32(y2_19) & 2147483648u | 729808896);
        float _add_102;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_102) : "f"(err_19), "f"(nudge_19));
        float u0_9;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0_9) : "f"(_add_102), "f"(1.1754944e-38f), "f"(s0_19));
        float dn_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_20) : "f"(e1_24), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_35 = approx_rcp(dn_20);
        float _fma_103;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_103) : "f"(-dn_20), "f"(r_35), "f"(1.0f));
        float q_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_20) : "f"(r_35), "f"(_fma_103), "f"(r_35));
        float s0_20;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_20) : "f"(q_20), "f"(1.1754944e-38f));
        float err_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_20) : "f"(s0_20), "f"(-8.507059e+37f), "f"(q_20));
        float y2_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_20) : "f"(-dn_20), "f"(q_20), "f"(1.0f));
        float nudge_20 = __uint_as_float(__as_u32(y2_20) & 2147483648u | 729808896);
        float _add_104;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_104) : "f"(err_20), "f"(nudge_20));
        float u1_9;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1_9) : "f"(_add_104), "f"(1.1754944e-38f), "f"(s0_20));
        float2 s_40 = make_float2(u0_9, u1_9);
        float2 act_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_24) : "l"(*(const unsigned long long*)&x_24), "l"(*(const unsigned long long*)&s_40));
        float2 _f32x2_mul_a_105 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_24, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_40), s_40), g_24);
        float2 mul_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_105), "l"(*(const unsigned long long*)&u_24));
        float2 mul_1_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_24) : "l"(*(const unsigned long long*)&act_24), "l"(*(const unsigned long long*)&g_24));
        __nv_bfloat162 bf16x2_27 = __float22bfloat162_rn(make_float2(mul_25.x, mul_25.y));
        __nv_bfloat162 bf16x2_1_26 = __float22bfloat162_rn(make_float2(mul_1_24.x, mul_1_24.y));
        dgate_b[2] = __as_u32(bf16x2_27);
        dup_b[2] = __as_u32(bf16x2_1_26);
        float2 x_25 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[3]));
        float2 u_25 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[3]));
        float2 g_25 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[3]));
        float2 _f32x2_mul_b_106 = make_float2(-1.442695f, -1.442695f);
        float2 t_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_25) : "l"(*(const unsigned long long*)&x_25), "l"(*(const unsigned long long*)&_f32x2_mul_b_106));
        float e0_25 = approx_exp2(t_25.x);
        float e1_25 = approx_exp2(t_25.y);
        float2 nd_25 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_25, e1_25));
        float d0_25 = -nd_25.x;
        float d1_25 = -nd_25.y;
        float dn_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_21) : "f"(e0_25), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_36 = approx_rcp(dn_21);
        float _fma_107;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(-dn_21), "f"(r_36), "f"(1.0f));
        float q_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_21) : "f"(r_36), "f"(_fma_107), "f"(r_36));
        float s0_21;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_21) : "f"(q_21), "f"(1.1754944e-38f));
        float err_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_21) : "f"(s0_21), "f"(-8.507059e+37f), "f"(q_21));
        float y2_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_21) : "f"(-dn_21), "f"(q_21), "f"(1.0f));
        float nudge_21 = __uint_as_float(__as_u32(y2_21) & 2147483648u | 729808896);
        float _add_108;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_108) : "f"(err_21), "f"(nudge_21));
        float u0_10;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0_10) : "f"(_add_108), "f"(1.1754944e-38f), "f"(s0_21));
        float dn_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_22) : "f"(e1_25), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_37 = approx_rcp(dn_22);
        float _fma_109;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_109) : "f"(-dn_22), "f"(r_37), "f"(1.0f));
        float q_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_22) : "f"(r_37), "f"(_fma_109), "f"(r_37));
        float s0_22;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_22) : "f"(q_22), "f"(1.1754944e-38f));
        float err_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_22) : "f"(s0_22), "f"(-8.507059e+37f), "f"(q_22));
        float y2_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_22) : "f"(-dn_22), "f"(q_22), "f"(1.0f));
        float nudge_22 = __uint_as_float(__as_u32(y2_22) & 2147483648u | 729808896);
        float _add_110;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_110) : "f"(err_22), "f"(nudge_22));
        float u1_10;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1_10) : "f"(_add_110), "f"(1.1754944e-38f), "f"(s0_22));
        float2 s_41 = make_float2(u0_10, u1_10);
        float2 act_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_25) : "l"(*(const unsigned long long*)&x_25), "l"(*(const unsigned long long*)&s_41));
        float2 _f32x2_mul_a_111 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_25, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_41), s_41), g_25);
        float2 mul_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_111), "l"(*(const unsigned long long*)&u_25));
        float2 mul_1_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_25) : "l"(*(const unsigned long long*)&act_25), "l"(*(const unsigned long long*)&g_25));
        __nv_bfloat162 bf16x2_28 = __float22bfloat162_rn(make_float2(mul_26.x, mul_26.y));
        __nv_bfloat162 bf16x2_1_27 = __float22bfloat162_rn(make_float2(mul_1_25.x, mul_1_25.y));
        dgate_b[3] = __as_u32(bf16x2_28);
        dup_b[3] = __as_u32(bf16x2_1_27);
        float2 x_26 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[4]));
        float2 u_26 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[4]));
        float2 g_26 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[4]));
        float2 _f32x2_mul_b_112 = make_float2(-1.442695f, -1.442695f);
        float2 t_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_26) : "l"(*(const unsigned long long*)&x_26), "l"(*(const unsigned long long*)&_f32x2_mul_b_112));
        float e0_26 = approx_exp2(t_26.x);
        float e1_26 = approx_exp2(t_26.y);
        float2 nd_26 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_26, e1_26));
        float d0_26 = -nd_26.x;
        float d1_26 = -nd_26.y;
        float dn_23;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_23) : "f"(e0_26), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_38 = approx_rcp(dn_23);
        float _fma_113;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_113) : "f"(-dn_23), "f"(r_38), "f"(1.0f));
        float q_23;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_23) : "f"(r_38), "f"(_fma_113), "f"(r_38));
        float s0_23;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_23) : "f"(q_23), "f"(1.1754944e-38f));
        float err_23;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_23) : "f"(s0_23), "f"(-8.507059e+37f), "f"(q_23));
        float y2_23;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_23) : "f"(-dn_23), "f"(q_23), "f"(1.0f));
        float nudge_23 = __uint_as_float(__as_u32(y2_23) & 2147483648u | 729808896);
        float _add_114;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_114) : "f"(err_23), "f"(nudge_23));
        float u0_11;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0_11) : "f"(_add_114), "f"(1.1754944e-38f), "f"(s0_23));
        float dn_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_24) : "f"(e1_26), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_39 = approx_rcp(dn_24);
        float _fma_115;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_115) : "f"(-dn_24), "f"(r_39), "f"(1.0f));
        float q_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_24) : "f"(r_39), "f"(_fma_115), "f"(r_39));
        float s0_24;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_24) : "f"(q_24), "f"(1.1754944e-38f));
        float err_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_24) : "f"(s0_24), "f"(-8.507059e+37f), "f"(q_24));
        float y2_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_24) : "f"(-dn_24), "f"(q_24), "f"(1.0f));
        float nudge_24 = __uint_as_float(__as_u32(y2_24) & 2147483648u | 729808896);
        float _add_116;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_116) : "f"(err_24), "f"(nudge_24));
        float u1_11;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1_11) : "f"(_add_116), "f"(1.1754944e-38f), "f"(s0_24));
        float2 s_42 = make_float2(u0_11, u1_11);
        float2 act_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_26) : "l"(*(const unsigned long long*)&x_26), "l"(*(const unsigned long long*)&s_42));
        float2 _f32x2_mul_a_117 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_26, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_42), s_42), g_26);
        float2 mul_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_117), "l"(*(const unsigned long long*)&u_26));
        float2 mul_1_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_26) : "l"(*(const unsigned long long*)&act_26), "l"(*(const unsigned long long*)&g_26));
        __nv_bfloat162 bf16x2_29 = __float22bfloat162_rn(make_float2(mul_27.x, mul_27.y));
        __nv_bfloat162 bf16x2_1_28 = __float22bfloat162_rn(make_float2(mul_1_26.x, mul_1_26.y));
        dgate_b[4] = __as_u32(bf16x2_29);
        dup_b[4] = __as_u32(bf16x2_1_28);
        float2 x_27 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[5]));
        float2 u_27 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[5]));
        float2 g_27 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[5]));
        float2 _f32x2_mul_b_118 = make_float2(-1.442695f, -1.442695f);
        float2 t_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_27) : "l"(*(const unsigned long long*)&x_27), "l"(*(const unsigned long long*)&_f32x2_mul_b_118));
        float e0_27 = approx_exp2(t_27.x);
        float e1_27 = approx_exp2(t_27.y);
        float2 nd_27 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_27, e1_27));
        float d0_27 = -nd_27.x;
        float d1_27 = -nd_27.y;
        float dn_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_25) : "f"(e0_27), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_40 = approx_rcp(dn_25);
        float _fma_119;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_119) : "f"(-dn_25), "f"(r_40), "f"(1.0f));
        float q_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_25) : "f"(r_40), "f"(_fma_119), "f"(r_40));
        float s0_25;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_25) : "f"(q_25), "f"(1.1754944e-38f));
        float err_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_25) : "f"(s0_25), "f"(-8.507059e+37f), "f"(q_25));
        float y2_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_25) : "f"(-dn_25), "f"(q_25), "f"(1.0f));
        float nudge_25 = __uint_as_float(__as_u32(y2_25) & 2147483648u | 729808896);
        float _add_120;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_120) : "f"(err_25), "f"(nudge_25));
        float u0_12;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0_12) : "f"(_add_120), "f"(1.1754944e-38f), "f"(s0_25));
        float dn_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_26) : "f"(e1_27), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_41 = approx_rcp(dn_26);
        float _fma_121;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_121) : "f"(-dn_26), "f"(r_41), "f"(1.0f));
        float q_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_26) : "f"(r_41), "f"(_fma_121), "f"(r_41));
        float s0_26;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_26) : "f"(q_26), "f"(1.1754944e-38f));
        float err_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_26) : "f"(s0_26), "f"(-8.507059e+37f), "f"(q_26));
        float y2_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_26) : "f"(-dn_26), "f"(q_26), "f"(1.0f));
        float nudge_26 = __uint_as_float(__as_u32(y2_26) & 2147483648u | 729808896);
        float _add_122;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_122) : "f"(err_26), "f"(nudge_26));
        float u1_12;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1_12) : "f"(_add_122), "f"(1.1754944e-38f), "f"(s0_26));
        float2 s_43 = make_float2(u0_12, u1_12);
        float2 act_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_27) : "l"(*(const unsigned long long*)&x_27), "l"(*(const unsigned long long*)&s_43));
        float2 _f32x2_mul_a_123 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_27, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_43), s_43), g_27);
        float2 mul_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_123), "l"(*(const unsigned long long*)&u_27));
        float2 mul_1_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_27) : "l"(*(const unsigned long long*)&act_27), "l"(*(const unsigned long long*)&g_27));
        __nv_bfloat162 bf16x2_30 = __float22bfloat162_rn(make_float2(mul_28.x, mul_28.y));
        __nv_bfloat162 bf16x2_1_29 = __float22bfloat162_rn(make_float2(mul_1_27.x, mul_1_27.y));
        dgate_b[5] = __as_u32(bf16x2_30);
        dup_b[5] = __as_u32(bf16x2_1_29);
        float2 x_28 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[6]));
        float2 u_28 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[6]));
        float2 g_28 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[6]));
        float2 _f32x2_mul_b_124 = make_float2(-1.442695f, -1.442695f);
        float2 t_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_28) : "l"(*(const unsigned long long*)&x_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_124));
        float e0_28 = approx_exp2(t_28.x);
        float e1_28 = approx_exp2(t_28.y);
        float2 nd_28 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_28, e1_28));
        float d0_28 = -nd_28.x;
        float d1_28 = -nd_28.y;
        float dn_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_27) : "f"(e0_28), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_42 = approx_rcp(dn_27);
        float _fma_125;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(-dn_27), "f"(r_42), "f"(1.0f));
        float q_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_27) : "f"(r_42), "f"(_fma_125), "f"(r_42));
        float s0_27;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_27) : "f"(q_27), "f"(1.1754944e-38f));
        float err_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_27) : "f"(s0_27), "f"(-8.507059e+37f), "f"(q_27));
        float y2_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_27) : "f"(-dn_27), "f"(q_27), "f"(1.0f));
        float nudge_27 = __uint_as_float(__as_u32(y2_27) & 2147483648u | 729808896);
        float _add_126;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_126) : "f"(err_27), "f"(nudge_27));
        float u0_13;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0_13) : "f"(_add_126), "f"(1.1754944e-38f), "f"(s0_27));
        float dn_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_28) : "f"(e1_28), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_43 = approx_rcp(dn_28);
        float _fma_127;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_127) : "f"(-dn_28), "f"(r_43), "f"(1.0f));
        float q_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_28) : "f"(r_43), "f"(_fma_127), "f"(r_43));
        float s0_28;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_28) : "f"(q_28), "f"(1.1754944e-38f));
        float err_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_28) : "f"(s0_28), "f"(-8.507059e+37f), "f"(q_28));
        float y2_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_28) : "f"(-dn_28), "f"(q_28), "f"(1.0f));
        float nudge_28 = __uint_as_float(__as_u32(y2_28) & 2147483648u | 729808896);
        float _add_128;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_128) : "f"(err_28), "f"(nudge_28));
        float u1_13;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1_13) : "f"(_add_128), "f"(1.1754944e-38f), "f"(s0_28));
        float2 s_44 = make_float2(u0_13, u1_13);
        float2 act_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_28) : "l"(*(const unsigned long long*)&x_28), "l"(*(const unsigned long long*)&s_44));
        float2 _f32x2_mul_a_129 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_28, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_44), s_44), g_28);
        float2 mul_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_129), "l"(*(const unsigned long long*)&u_28));
        float2 mul_1_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_28) : "l"(*(const unsigned long long*)&act_28), "l"(*(const unsigned long long*)&g_28));
        __nv_bfloat162 bf16x2_31 = __float22bfloat162_rn(make_float2(mul_29.x, mul_29.y));
        __nv_bfloat162 bf16x2_1_30 = __float22bfloat162_rn(make_float2(mul_1_28.x, mul_1_28.y));
        dgate_b[6] = __as_u32(bf16x2_31);
        dup_b[6] = __as_u32(bf16x2_1_30);
        float2 x_29 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[7]));
        float2 u_29 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[7]));
        float2 g_29 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2_0[7]));
        float2 _f32x2_mul_b_130 = make_float2(-1.442695f, -1.442695f);
        float2 t_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_29) : "l"(*(const unsigned long long*)&x_29), "l"(*(const unsigned long long*)&_f32x2_mul_b_130));
        float e0_29 = approx_exp2(t_29.x);
        float e1_29 = approx_exp2(t_29.y);
        float2 nd_29 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_29, e1_29));
        float d0_29 = -nd_29.x;
        float d1_29 = -nd_29.y;
        float dn_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_29) : "f"(e0_29), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_44 = approx_rcp(dn_29);
        float _fma_131;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_131) : "f"(-dn_29), "f"(r_44), "f"(1.0f));
        float q_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_29) : "f"(r_44), "f"(_fma_131), "f"(r_44));
        float s0_29;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_29) : "f"(q_29), "f"(1.1754944e-38f));
        float err_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_29) : "f"(s0_29), "f"(-8.507059e+37f), "f"(q_29));
        float y2_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_29) : "f"(-dn_29), "f"(q_29), "f"(1.0f));
        float nudge_29 = __uint_as_float(__as_u32(y2_29) & 2147483648u | 729808896);
        float _add_132;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_132) : "f"(err_29), "f"(nudge_29));
        float u0_14;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u0_14) : "f"(_add_132), "f"(1.1754944e-38f), "f"(s0_29));
        float dn_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dn_30) : "f"(e1_29), "f"(1.1754944e-38f), "f"(1.1754944e-38f));
        float r_45 = approx_rcp(dn_30);
        float _fma_133;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_133) : "f"(-dn_30), "f"(r_45), "f"(1.0f));
        float q_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(q_30) : "f"(r_45), "f"(_fma_133), "f"(r_45));
        float s0_30;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_30) : "f"(q_30), "f"(1.1754944e-38f));
        float err_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(err_30) : "f"(s0_30), "f"(-8.507059e+37f), "f"(q_30));
        float y2_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y2_30) : "f"(-dn_30), "f"(q_30), "f"(1.0f));
        float nudge_30 = __uint_as_float(__as_u32(y2_30) & 2147483648u | 729808896);
        float _add_134;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_134) : "f"(err_30), "f"(nudge_30));
        float u1_14;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(u1_14) : "f"(_add_134), "f"(1.1754944e-38f), "f"(s0_30));
        float2 s_45 = make_float2(u0_14, u1_14);
        float2 act_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_29) : "l"(*(const unsigned long long*)&x_29), "l"(*(const unsigned long long*)&s_45));
        float2 _f32x2_mul_a_135 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_29, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_45), s_45), g_29);
        float2 mul_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_135), "l"(*(const unsigned long long*)&u_29));
        float2 mul_1_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_29) : "l"(*(const unsigned long long*)&act_29), "l"(*(const unsigned long long*)&g_29));
        __nv_bfloat162 bf16x2_32 = __float22bfloat162_rn(make_float2(mul_30.x, mul_30.y));
        __nv_bfloat162 bf16x2_1_31 = __float22bfloat162_rn(make_float2(mul_1_29.x, mul_1_29.y));
        dgate_b[7] = __as_u32(bf16x2_32);
        dup_b[7] = __as_u32(bf16x2_1_31);
    }
    unsigned int swizzle = (unsigned int)(row >> 3 & 6 | row & 1);
    unsigned int row_base = tiles_addr + (unsigned int)(row * 128);
    unsigned int chunk = (unsigned int)((tid & 3) * 2);
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + ((chunk ^ swizzle) << 4)), "r"(dgate[0]), "r"(dgate[1]), "r"(dgate[2]), "r"(dgate[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + ((chunk + 1 ^ swizzle) << 4)), "r"(dgate[4]), "r"(dgate[5]), "r"(dgate[6]), "r"(dgate[7]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + (8192 + ((chunk ^ swizzle) << 4))), "r"(dup[0]), "r"(dup[1]), "r"(dup[2]), "r"(dup[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + (8192 + ((chunk + 1 ^ swizzle) << 4))), "r"(dup[4]), "r"(dup[5]), "r"(dup[6]), "r"(dup[7]) : "memory");
    unsigned int swizzle_hi = swizzle ^ 4;
    unsigned int row_base_hi = row_base + 4096;
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base_hi + ((chunk ^ swizzle_hi) << 4)), "r"(dgate_b[0]), "r"(dgate_b[1]), "r"(dgate_b[2]), "r"(dgate_b[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base_hi + ((chunk + 1 ^ swizzle_hi) << 4)), "r"(dgate_b[4]), "r"(dgate_b[5]), "r"(dgate_b[6]), "r"(dgate_b[7]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base_hi + (8192 + ((chunk ^ swizzle_hi) << 4))), "r"(dup_b[0]), "r"(dup_b[1]), "r"(dup_b[2]), "r"(dup_b[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base_hi + (8192 + ((chunk + 1 ^ swizzle_hi) << 4))), "r"(dup_b[4]), "r"(dup_b[5]), "r"(dup_b[6]), "r"(dup_b[7]) : "memory");
    int scale_cols = K >> 6;
    int scale_col = bx * 2 + blk;
    int out_index = grow * (2 * K) + col;
    __nv_bfloat162 amax = __as_bf16x2(0);
    __nv_bfloat162 _hxa_137 = amax;
    __nv_bfloat162 _hxb_137 = __as_bf16x2(dgate[0]);
    unsigned int _hxr_137;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_137) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_137)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_137)));
    __nv_bfloat162 _max_136 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_137);
    amax = _max_136;
    __nv_bfloat162 _hxa_139 = amax;
    __nv_bfloat162 _hxb_139 = __as_bf16x2(dgate[1]);
    unsigned int _hxr_139;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_139) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_139)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_139)));
    __nv_bfloat162 _max_138 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_139);
    amax = _max_138;
    __nv_bfloat162 _hxa_141 = amax;
    __nv_bfloat162 _hxb_141 = __as_bf16x2(dgate[2]);
    unsigned int _hxr_141;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_141) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_141)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_141)));
    __nv_bfloat162 _max_140 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_141);
    amax = _max_140;
    __nv_bfloat162 _hxa_143 = amax;
    __nv_bfloat162 _hxb_143 = __as_bf16x2(dgate[3]);
    unsigned int _hxr_143;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_143) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_143)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_143)));
    __nv_bfloat162 _max_142 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_143);
    amax = _max_142;
    __nv_bfloat162 _hxa_145 = amax;
    __nv_bfloat162 _hxb_145 = __as_bf16x2(dgate[4]);
    unsigned int _hxr_145;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_145) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_145)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_145)));
    __nv_bfloat162 _max_144 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_145);
    amax = _max_144;
    __nv_bfloat162 _hxa_147 = amax;
    __nv_bfloat162 _hxb_147 = __as_bf16x2(dgate[5]);
    unsigned int _hxr_147;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_147) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_147)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_147)));
    __nv_bfloat162 _max_146 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_147);
    amax = _max_146;
    __nv_bfloat162 _hxa_149 = amax;
    __nv_bfloat162 _hxb_149 = __as_bf16x2(dgate[6]);
    unsigned int _hxr_149;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_149) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_149)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_149)));
    __nv_bfloat162 _max_148 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_149);
    amax = _max_148;
    __nv_bfloat162 _hxa_151 = amax;
    __nv_bfloat162 _hxb_151 = __as_bf16x2(dgate[7]);
    unsigned int _hxr_151;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_151) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_151)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_151)));
    __nv_bfloat162 _max_150 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_151);
    amax = _max_150;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_153;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_153) : "r"(bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_153) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_152;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_152) : "h"(e4m3x2_bf16x2), "h"(_cvt_satfinite_e4m3x2_bf16x2_153));
    words[0] = _mov_b32_pack_u16x2_152;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_155;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_155) : "r"(bf16x2_mul_1_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_155) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_154;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_154) : "h"(e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_155));
    words[1] = _mov_b32_pack_u16x2_154;
    uint32_t bf16x2_mul_4;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4) : "r"(dgate[4]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_1) : "r"(dgate[5]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_157;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_157) : "r"(bf16x2_mul_1_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_157) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_156;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_156) : "h"(e4m3x2_bf16x2_4), "h"(_cvt_satfinite_e4m3x2_bf16x2_157));
    words[2] = _mov_b32_pack_u16x2_156;
    uint32_t bf16x2_mul_5;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5) : "r"(dgate[6]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_2) : "r"(dgate[7]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_159;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_159) : "r"(bf16x2_mul_1_2));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_2);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_159) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_158;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_158) : "h"(e4m3x2_bf16x2_5), "h"(_cvt_satfinite_e4m3x2_bf16x2_159));
    words[3] = _mov_b32_pack_u16x2_158;
    reinterpret_cast<int4*>(row_output + out_index)[0] = reinterpret_cast<int4*>(words + 0)[0];
    __nv_bfloat162 amax_0 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_161 = amax_0;
    __nv_bfloat162 _hxb_161 = __as_bf16x2(dup[0]);
    unsigned int _hxr_161;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_161) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_161)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_161)));
    __nv_bfloat162 _max_160 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_161);
    amax_0 = _max_160;
    __nv_bfloat162 _hxa_163 = amax_0;
    __nv_bfloat162 _hxb_163 = __as_bf16x2(dup[1]);
    unsigned int _hxr_163;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_163) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_163)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_163)));
    __nv_bfloat162 _max_162 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_163);
    amax_0 = _max_162;
    __nv_bfloat162 _hxa_165 = amax_0;
    __nv_bfloat162 _hxb_165 = __as_bf16x2(dup[2]);
    unsigned int _hxr_165;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_165) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_165)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_165)));
    __nv_bfloat162 _max_164 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_165);
    amax_0 = _max_164;
    __nv_bfloat162 _hxa_167 = amax_0;
    __nv_bfloat162 _hxb_167 = __as_bf16x2(dup[3]);
    unsigned int _hxr_167;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_167) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_167)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_167)));
    __nv_bfloat162 _max_166 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_167);
    amax_0 = _max_166;
    __nv_bfloat162 _hxa_169 = amax_0;
    __nv_bfloat162 _hxb_169 = __as_bf16x2(dup[4]);
    unsigned int _hxr_169;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_169) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_169)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_169)));
    __nv_bfloat162 _max_168 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_169);
    amax_0 = _max_168;
    __nv_bfloat162 _hxa_171 = amax_0;
    __nv_bfloat162 _hxb_171 = __as_bf16x2(dup[5]);
    unsigned int _hxr_171;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_171) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_171)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_171)));
    __nv_bfloat162 _max_170 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_171);
    amax_0 = _max_170;
    __nv_bfloat162 _hxa_173 = amax_0;
    __nv_bfloat162 _hxb_173 = __as_bf16x2(dup[6]);
    unsigned int _hxr_173;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_173) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_173)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_173)));
    __nv_bfloat162 _max_172 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_173);
    amax_0 = _max_172;
    __nv_bfloat162 _hxa_175 = amax_0;
    __nv_bfloat162 _hxb_175 = __as_bf16x2(dup[7]);
    unsigned int _hxr_175;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_175) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_175)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_175)));
    __nv_bfloat162 _max_174 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_175);
    amax_0 = _max_174;
    unsigned int mag_0 = __as_u32(amax_0) & 2147450879;
    unsigned int peer_0 = __shfl_xor_sync(0xFFFFFFFF, mag_0, 1);
    __nv_bfloat162 both_0 = __hmax2_nan(__as_bf16x2(mag_0), __as_bf16x2(peer_0));
    unsigned int folded_0 = __as_u32(__hmax2_nan(both_0, __as_bf16x2(__as_u32(both_0) >> 16)));
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
    int scale_index_0 = ((grow >> 7) * scale_cols + (scale_col + (K >> 5) >> 2)) * 512 + (grow & 31) * 16 + (grow >> 5 & 3) * 4 + (scale_col + (K >> 5) & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + scale_index_0) + (0)) = (unsigned char)(scale_0);
    }
    unsigned int words_0[4];
    uint32_t bf16x2_mul_6;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6) : "r"(dup[0]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_3) : "r"(dup[1]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_177;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_177) : "r"(bf16x2_mul_1_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_177) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_176;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_176) : "h"(e4m3x2_bf16x2_6), "h"(_cvt_satfinite_e4m3x2_bf16x2_177));
    words_0[0] = _mov_b32_pack_u16x2_176;
    uint32_t bf16x2_mul_7;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7) : "r"(dup[2]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_4) : "r"(dup[3]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_179;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_179) : "r"(bf16x2_mul_1_4));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_4);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_179) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_178;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_178) : "h"(e4m3x2_bf16x2_7), "h"(_cvt_satfinite_e4m3x2_bf16x2_179));
    words_0[1] = _mov_b32_pack_u16x2_178;
    uint32_t bf16x2_mul_8;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_8) : "r"(dup[4]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_5) : "r"(dup[5]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_181;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_181) : "r"(bf16x2_mul_1_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_181) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_180;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_180) : "h"(e4m3x2_bf16x2_8), "h"(_cvt_satfinite_e4m3x2_bf16x2_181));
    words_0[2] = _mov_b32_pack_u16x2_180;
    uint32_t bf16x2_mul_9;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_9) : "r"(dup[6]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_6) : "r"(dup[7]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_183;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_183) : "r"(bf16x2_mul_1_6));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_6);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_183) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_182;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_182) : "h"(e4m3x2_bf16x2_9), "h"(_cvt_satfinite_e4m3x2_bf16x2_183));
    words_0[3] = _mov_b32_pack_u16x2_182;
    reinterpret_cast<int4*>(row_output + (out_index + K))[0] = reinterpret_cast<int4*>(words_0 + 0)[0];
    int out_index_hi = grow_b * (2 * K) + col;
    __nv_bfloat162 amax_1 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_185 = amax_1;
    __nv_bfloat162 _hxb_185 = __as_bf16x2(dgate_b[0]);
    unsigned int _hxr_185;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_185) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_185)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_185)));
    __nv_bfloat162 _max_184 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_185);
    amax_1 = _max_184;
    __nv_bfloat162 _hxa_187 = amax_1;
    __nv_bfloat162 _hxb_187 = __as_bf16x2(dgate_b[1]);
    unsigned int _hxr_187;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_187) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_187)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_187)));
    __nv_bfloat162 _max_186 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_187);
    amax_1 = _max_186;
    __nv_bfloat162 _hxa_189 = amax_1;
    __nv_bfloat162 _hxb_189 = __as_bf16x2(dgate_b[2]);
    unsigned int _hxr_189;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_189) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_189)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_189)));
    __nv_bfloat162 _max_188 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_189);
    amax_1 = _max_188;
    __nv_bfloat162 _hxa_191 = amax_1;
    __nv_bfloat162 _hxb_191 = __as_bf16x2(dgate_b[3]);
    unsigned int _hxr_191;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_191) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_191)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_191)));
    __nv_bfloat162 _max_190 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_191);
    amax_1 = _max_190;
    __nv_bfloat162 _hxa_193 = amax_1;
    __nv_bfloat162 _hxb_193 = __as_bf16x2(dgate_b[4]);
    unsigned int _hxr_193;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_193) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_193)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_193)));
    __nv_bfloat162 _max_192 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_193);
    amax_1 = _max_192;
    __nv_bfloat162 _hxa_195 = amax_1;
    __nv_bfloat162 _hxb_195 = __as_bf16x2(dgate_b[5]);
    unsigned int _hxr_195;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_195) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_195)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_195)));
    __nv_bfloat162 _max_194 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_195);
    amax_1 = _max_194;
    __nv_bfloat162 _hxa_197 = amax_1;
    __nv_bfloat162 _hxb_197 = __as_bf16x2(dgate_b[6]);
    unsigned int _hxr_197;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_197) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_197)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_197)));
    __nv_bfloat162 _max_196 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_197);
    amax_1 = _max_196;
    __nv_bfloat162 _hxa_199 = amax_1;
    __nv_bfloat162 _hxb_199 = __as_bf16x2(dgate_b[7]);
    unsigned int _hxr_199;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_199) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_199)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_199)));
    __nv_bfloat162 _max_198 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_199);
    amax_1 = _max_198;
    unsigned int mag_1 = __as_u32(amax_1) & 2147450879;
    unsigned int peer_1 = __shfl_xor_sync(0xFFFFFFFF, mag_1, 1);
    __nv_bfloat162 both_1 = __hmax2_nan(__as_bf16x2(mag_1), __as_bf16x2(peer_1));
    unsigned int folded_1 = __as_u32(__hmax2_nan(both_1, __as_bf16x2(__as_u32(both_1) >> 16)));
    unsigned int amax_bits_1 = (folded_1 & 65535) << 16;
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
    int scale_index_1 = ((grow_b >> 7) * scale_cols + (scale_col >> 2)) * 512 + (grow_b & 31) * 16 + (grow_b >> 5 & 3) * 4 + (scale_col & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + scale_index_1) + (0)) = (unsigned char)(scale_3);
    }
    unsigned int words_1[4];
    uint32_t bf16x2_mul_10;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_10) : "r"(dgate_b[0]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_7) : "r"(dgate_b[1]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_201;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_201) : "r"(bf16x2_mul_1_7));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_7);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_201) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_200;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_200) : "h"(e4m3x2_bf16x2_10), "h"(_cvt_satfinite_e4m3x2_bf16x2_201));
    words_1[0] = _mov_b32_pack_u16x2_200;
    uint32_t bf16x2_mul_11;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_11) : "r"(dgate_b[2]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_8) : "r"(dgate_b[3]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_203;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_203) : "r"(bf16x2_mul_1_8));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_8);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_203) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_202;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_202) : "h"(e4m3x2_bf16x2_11), "h"(_cvt_satfinite_e4m3x2_bf16x2_203));
    words_1[1] = _mov_b32_pack_u16x2_202;
    uint32_t bf16x2_mul_12;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_12) : "r"(dgate_b[4]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_9) : "r"(dgate_b[5]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_205;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_205) : "r"(bf16x2_mul_1_9));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_9);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_205) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_204;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_204) : "h"(e4m3x2_bf16x2_12), "h"(_cvt_satfinite_e4m3x2_bf16x2_205));
    words_1[2] = _mov_b32_pack_u16x2_204;
    uint32_t bf16x2_mul_13;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_13) : "r"(dgate_b[6]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_10) : "r"(dgate_b[7]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_207;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_207) : "r"(bf16x2_mul_1_10));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_10);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_207) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_206;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_206) : "h"(e4m3x2_bf16x2_13), "h"(_cvt_satfinite_e4m3x2_bf16x2_207));
    words_1[3] = _mov_b32_pack_u16x2_206;
    reinterpret_cast<int4*>(row_output + out_index_hi)[0] = reinterpret_cast<int4*>(words_1 + 0)[0];
    __nv_bfloat162 amax_2 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_209 = amax_2;
    __nv_bfloat162 _hxb_209 = __as_bf16x2(dup_b[0]);
    unsigned int _hxr_209;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_209) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_209)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_209)));
    __nv_bfloat162 _max_208 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_209);
    amax_2 = _max_208;
    __nv_bfloat162 _hxa_211 = amax_2;
    __nv_bfloat162 _hxb_211 = __as_bf16x2(dup_b[1]);
    unsigned int _hxr_211;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_211) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_211)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_211)));
    __nv_bfloat162 _max_210 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_211);
    amax_2 = _max_210;
    __nv_bfloat162 _hxa_213 = amax_2;
    __nv_bfloat162 _hxb_213 = __as_bf16x2(dup_b[2]);
    unsigned int _hxr_213;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_213) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_213)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_213)));
    __nv_bfloat162 _max_212 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_213);
    amax_2 = _max_212;
    __nv_bfloat162 _hxa_215 = amax_2;
    __nv_bfloat162 _hxb_215 = __as_bf16x2(dup_b[3]);
    unsigned int _hxr_215;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_215) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_215)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_215)));
    __nv_bfloat162 _max_214 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_215);
    amax_2 = _max_214;
    __nv_bfloat162 _hxa_217 = amax_2;
    __nv_bfloat162 _hxb_217 = __as_bf16x2(dup_b[4]);
    unsigned int _hxr_217;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_217) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_217)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_217)));
    __nv_bfloat162 _max_216 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_217);
    amax_2 = _max_216;
    __nv_bfloat162 _hxa_219 = amax_2;
    __nv_bfloat162 _hxb_219 = __as_bf16x2(dup_b[5]);
    unsigned int _hxr_219;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_219) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_219)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_219)));
    __nv_bfloat162 _max_218 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_219);
    amax_2 = _max_218;
    __nv_bfloat162 _hxa_221 = amax_2;
    __nv_bfloat162 _hxb_221 = __as_bf16x2(dup_b[6]);
    unsigned int _hxr_221;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_221) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_221)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_221)));
    __nv_bfloat162 _max_220 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_221);
    amax_2 = _max_220;
    __nv_bfloat162 _hxa_223 = amax_2;
    __nv_bfloat162 _hxb_223 = __as_bf16x2(dup_b[7]);
    unsigned int _hxr_223;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_223) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_223)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_223)));
    __nv_bfloat162 _max_222 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_223);
    amax_2 = _max_222;
    unsigned int mag_2 = __as_u32(amax_2) & 2147450879;
    unsigned int peer_2 = __shfl_xor_sync(0xFFFFFFFF, mag_2, 1);
    __nv_bfloat162 both_2 = __hmax2_nan(__as_bf16x2(mag_2), __as_bf16x2(peer_2));
    unsigned int folded_2 = __as_u32(__hmax2_nan(both_2, __as_bf16x2(__as_u32(both_2) >> 16)));
    unsigned int amax_bits_2 = (folded_2 & 65535) << 16;
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
    int scale_index_2 = ((grow_b >> 7) * scale_cols + (scale_col + (K >> 5) >> 2)) * 512 + (grow_b & 31) * 16 + (grow_b >> 5 & 3) * 4 + (scale_col + (K >> 5) & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + scale_index_2) + (0)) = (unsigned char)(scale_4);
    }
    unsigned int words_2[4];
    uint32_t bf16x2_mul_14;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_14) : "r"(dup_b[0]), "r"(inv2_3));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_11) : "r"(dup_b[1]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_225;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_225) : "r"(bf16x2_mul_1_11));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_11);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_225) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_224;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_224) : "h"(e4m3x2_bf16x2_14), "h"(_cvt_satfinite_e4m3x2_bf16x2_225));
    words_2[0] = _mov_b32_pack_u16x2_224;
    uint32_t bf16x2_mul_15;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_15) : "r"(dup_b[2]), "r"(inv2_3));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_12) : "r"(dup_b[3]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_227;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_227) : "r"(bf16x2_mul_1_12));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_12);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_227) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_226;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_226) : "h"(e4m3x2_bf16x2_15), "h"(_cvt_satfinite_e4m3x2_bf16x2_227));
    words_2[1] = _mov_b32_pack_u16x2_226;
    uint32_t bf16x2_mul_16;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_16) : "r"(dup_b[4]), "r"(inv2_3));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_13) : "r"(dup_b[5]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_229;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_229) : "r"(bf16x2_mul_1_13));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_13);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_229) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_228;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_228) : "h"(e4m3x2_bf16x2_16), "h"(_cvt_satfinite_e4m3x2_bf16x2_229));
    words_2[2] = _mov_b32_pack_u16x2_228;
    uint32_t bf16x2_mul_17;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_17) : "r"(dup_b[6]), "r"(inv2_3));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_14) : "r"(dup_b[7]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_231;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_231) : "r"(bf16x2_mul_1_14));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_14);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_231) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_230;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_230) : "h"(e4m3x2_bf16x2_17), "h"(_cvt_satfinite_e4m3x2_bf16x2_231));
    words_2[3] = _mov_b32_pack_u16x2_230;
    reinterpret_cast<int4*>(row_output + (out_index_hi + K))[0] = reinterpret_cast<int4*>(words_2 + 0)[0];
    __syncthreads();
    int lane_1 = tid & 31;
    int warp_1 = tid >> 5;
    int k8 = lane_1 & 7;
    int srow = (k8 >> 1) * 16 + (lane_1 >> 3) * 2 + (k8 & 1);
    unsigned int sswizzle = (unsigned int)(srow >> 3 & 6 | srow & 1);
    unsigned int cols[16];
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
    unsigned int strip_chunk0_1_1 = (unsigned int)(warp_1 * 2 + 1);
    unsigned int strip_src0_1_1 = tiles_addr + (unsigned int)(srow * 128) + ((strip_chunk0_1_1 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[8]), "=r"(cols[9]), "=r"(cols[10]), "=r"(cols[11])
        : "r"(strip_src0_1_1)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[12]), "=r"(cols[13]), "=r"(cols[14]), "=r"(cols[15])
        : "r"(strip_src0_1_1 + 1024)
        : "memory");
    __nv_bfloat162 amax_a = __as_bf16x2(0);
    __nv_bfloat162 amax_b = __as_bf16x2(0);
    __nv_bfloat162 _hxa_233 = amax_a;
    __nv_bfloat162 _hxb_233 = __as_bf16x2(cols[0]);
    unsigned int _hxr_233;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_233) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_233)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_233)));
    __nv_bfloat162 _max_232 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_233);
    amax_a = _max_232;
    __nv_bfloat162 _hxa_235 = amax_b;
    __nv_bfloat162 _hxb_235 = __as_bf16x2(cols[8]);
    unsigned int _hxr_235;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_235) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_235)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_235)));
    __nv_bfloat162 _max_234 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_235);
    amax_b = _max_234;
    __nv_bfloat162 _hxa_237 = amax_a;
    __nv_bfloat162 _hxb_237 = __as_bf16x2(cols[1]);
    unsigned int _hxr_237;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_237) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_237)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_237)));
    __nv_bfloat162 _max_236 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_237);
    amax_a = _max_236;
    __nv_bfloat162 _hxa_239 = amax_b;
    __nv_bfloat162 _hxb_239 = __as_bf16x2(cols[9]);
    unsigned int _hxr_239;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_239) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_239)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_239)));
    __nv_bfloat162 _max_238 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_239);
    amax_b = _max_238;
    __nv_bfloat162 _hxa_241 = amax_a;
    __nv_bfloat162 _hxb_241 = __as_bf16x2(cols[2]);
    unsigned int _hxr_241;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_241) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_241)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_241)));
    __nv_bfloat162 _max_240 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_241);
    amax_a = _max_240;
    __nv_bfloat162 _hxa_243 = amax_b;
    __nv_bfloat162 _hxb_243 = __as_bf16x2(cols[10]);
    unsigned int _hxr_243;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_243) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_243)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_243)));
    __nv_bfloat162 _max_242 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_243);
    amax_b = _max_242;
    __nv_bfloat162 _hxa_245 = amax_a;
    __nv_bfloat162 _hxb_245 = __as_bf16x2(cols[3]);
    unsigned int _hxr_245;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_245) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_245)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_245)));
    __nv_bfloat162 _max_244 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_245);
    amax_a = _max_244;
    __nv_bfloat162 _hxa_247 = amax_b;
    __nv_bfloat162 _hxb_247 = __as_bf16x2(cols[11]);
    unsigned int _hxr_247;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_247) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_247)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_247)));
    __nv_bfloat162 _max_246 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_247);
    amax_b = _max_246;
    __nv_bfloat162 _hxa_249 = amax_a;
    __nv_bfloat162 _hxb_249 = __as_bf16x2(cols[4]);
    unsigned int _hxr_249;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_249) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_249)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_249)));
    __nv_bfloat162 _max_248 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_249);
    amax_a = _max_248;
    __nv_bfloat162 _hxa_251 = amax_b;
    __nv_bfloat162 _hxb_251 = __as_bf16x2(cols[12]);
    unsigned int _hxr_251;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_251) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_251)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_251)));
    __nv_bfloat162 _max_250 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_251);
    amax_b = _max_250;
    __nv_bfloat162 _hxa_253 = amax_a;
    __nv_bfloat162 _hxb_253 = __as_bf16x2(cols[5]);
    unsigned int _hxr_253;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_253) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_253)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_253)));
    __nv_bfloat162 _max_252 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_253);
    amax_a = _max_252;
    __nv_bfloat162 _hxa_255 = amax_b;
    __nv_bfloat162 _hxb_255 = __as_bf16x2(cols[13]);
    unsigned int _hxr_255;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_255) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_255)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_255)));
    __nv_bfloat162 _max_254 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_255);
    amax_b = _max_254;
    __nv_bfloat162 _hxa_257 = amax_a;
    __nv_bfloat162 _hxb_257 = __as_bf16x2(cols[6]);
    unsigned int _hxr_257;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_257) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_257)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_257)));
    __nv_bfloat162 _max_256 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_257);
    amax_a = _max_256;
    __nv_bfloat162 _hxa_259 = amax_b;
    __nv_bfloat162 _hxb_259 = __as_bf16x2(cols[14]);
    unsigned int _hxr_259;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_259) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_259)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_259)));
    __nv_bfloat162 _max_258 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_259);
    amax_b = _max_258;
    __nv_bfloat162 _hxa_261 = amax_a;
    __nv_bfloat162 _hxb_261 = __as_bf16x2(cols[7]);
    unsigned int _hxr_261;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_261) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_261)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_261)));
    __nv_bfloat162 _max_260 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_261);
    amax_a = _max_260;
    __nv_bfloat162 _hxa_263 = amax_b;
    __nv_bfloat162 _hxb_263 = __as_bf16x2(cols[15]);
    unsigned int _hxr_263;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_263) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_263)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_263)));
    __nv_bfloat162 _max_262 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_263);
    amax_b = _max_262;
    unsigned int mag_a = __as_u32(amax_a) & 2147450879;
    unsigned int mag_b = __as_u32(amax_b) & 2147450879;
    unsigned int lows_ab = mag_a & 65535 | mag_b << 16;
    unsigned int highs_ab = mag_a >> 16 | mag_b & 4294901760u;
    __nv_bfloat162 col_max = __hmax2_nan(__as_bf16x2(lows_ab), __as_bf16x2(highs_ab));
    unsigned int peer1 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(col_max), 1);
    col_max = __hmax2_nan(col_max, __as_bf16x2(peer1));
    unsigned int col_max_bits = __as_u32(col_max);
    unsigned int amax_bits_3 = (col_max_bits & 65535) << 16;
    int scale_i32_3 = (int)(amax_bits_3 + 2031616 >> 23) - 8;
    if (scale_i32_3 < 0) {
        scale_i32_3 = 0;
    }
    if ((amax_bits_3 & 2139095040) == 2139095040) {
        scale_i32_3 = 255;
    }
    unsigned int scale_6 = (unsigned int)scale_i32_3;
    unsigned int inv_3 = 254 - scale_6 << 7;
    if (scale_6 == 255) {
        inv_3 = 32704;
    }
    unsigned int inv2_5 = inv_3 | inv_3 << 16;
    unsigned int amax_bits_4 = (col_max_bits >> 16 & 65535) << 16;
    int scale_i32_4 = (int)(amax_bits_4 + 2031616 >> 23) - 8;
    if (scale_i32_4 < 0) {
        scale_i32_4 = 0;
    }
    if ((amax_bits_4 & 2139095040) == 2139095040) {
        scale_i32_4 = 255;
    }
    unsigned int scale_7 = (unsigned int)scale_i32_4;
    unsigned int inv_4 = 254 - scale_7 << 7;
    if (scale_7 == 255) {
        inv_4 = 32704;
    }
    unsigned int inv2_6 = inv_4 | inv_4 << 16;
    int gcol_a = bx * 64 + warp_1 * 16 + (lane_1 >> 2);
    int gcol_b = gcol_a + 8;
    int out_row = by * 64 + (lane_1 & 3) * 16;
    unsigned int words_a[4];
    unsigned int words_b[4];
    uint32_t bf16x2_mul_18;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_18) : "r"(cols[0]), "r"(inv2_5));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_15) : "r"(cols[1]), "r"(inv2_5));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_265;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_265) : "r"(bf16x2_mul_1_15));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_15);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_265) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_264;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_264) : "h"(e4m3x2_bf16x2_18), "h"(_cvt_satfinite_e4m3x2_bf16x2_265));
    words_a[0] = _mov_b32_pack_u16x2_264;
    uint32_t bf16x2_mul_19;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_19) : "r"(cols[8]), "r"(inv2_6));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_16) : "r"(cols[9]), "r"(inv2_6));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_267;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_267) : "r"(bf16x2_mul_1_16));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_16);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_267) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_266;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_266) : "h"(e4m3x2_bf16x2_19), "h"(_cvt_satfinite_e4m3x2_bf16x2_267));
    words_b[0] = _mov_b32_pack_u16x2_266;
    uint32_t bf16x2_mul_20;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_20) : "r"(cols[2]), "r"(inv2_5));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_17) : "r"(cols[3]), "r"(inv2_5));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_269;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_269) : "r"(bf16x2_mul_1_17));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_17);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_269) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_268;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_268) : "h"(e4m3x2_bf16x2_20), "h"(_cvt_satfinite_e4m3x2_bf16x2_269));
    words_a[1] = _mov_b32_pack_u16x2_268;
    uint32_t bf16x2_mul_21;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_21) : "r"(cols[10]), "r"(inv2_6));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_18) : "r"(cols[11]), "r"(inv2_6));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_271;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_271) : "r"(bf16x2_mul_1_18));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_18);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_271) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_270;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_270) : "h"(e4m3x2_bf16x2_21), "h"(_cvt_satfinite_e4m3x2_bf16x2_271));
    words_b[1] = _mov_b32_pack_u16x2_270;
    uint32_t bf16x2_mul_22;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_22) : "r"(cols[4]), "r"(inv2_5));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_19) : "r"(cols[5]), "r"(inv2_5));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_273;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_273) : "r"(bf16x2_mul_1_19));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_19);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_273) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_272;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_272) : "h"(e4m3x2_bf16x2_22), "h"(_cvt_satfinite_e4m3x2_bf16x2_273));
    words_a[2] = _mov_b32_pack_u16x2_272;
    uint32_t bf16x2_mul_23;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_23) : "r"(cols[12]), "r"(inv2_6));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_20) : "r"(cols[13]), "r"(inv2_6));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_275;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_275) : "r"(bf16x2_mul_1_20));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_20);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_275) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_274;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_274) : "h"(e4m3x2_bf16x2_23), "h"(_cvt_satfinite_e4m3x2_bf16x2_275));
    words_b[2] = _mov_b32_pack_u16x2_274;
    uint32_t bf16x2_mul_24;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_24) : "r"(cols[6]), "r"(inv2_5));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_21) : "r"(cols[7]), "r"(inv2_5));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_277;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_277) : "r"(bf16x2_mul_1_21));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_21);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_277) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_276;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_276) : "h"(e4m3x2_bf16x2_24), "h"(_cvt_satfinite_e4m3x2_bf16x2_277));
    words_a[3] = _mov_b32_pack_u16x2_276;
    uint32_t bf16x2_mul_25;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_25) : "r"(cols[14]), "r"(inv2_6));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_22) : "r"(cols[15]), "r"(inv2_6));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_279;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_279) : "r"(bf16x2_mul_1_22));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_22);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_279) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_278;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_278) : "h"(e4m3x2_bf16x2_25), "h"(_cvt_satfinite_e4m3x2_bf16x2_279));
    words_b[3] = _mov_b32_pack_u16x2_278;
    reinterpret_cast<int4*>(col_output + (gcol_a * M + out_row))[0] = reinterpret_cast<int4*>(words_a + 0)[0];
    reinterpret_cast<int4*>(col_output + (gcol_b * M + out_row))[0] = reinterpret_cast<int4*>(words_b + 0)[0];
    if ((lane_1 & 1) == 0) {
        int row_tiles = M >> 7;
        int rblk = by * 2 + ((lane_1 & 3) >> 1);
        int col_scale_a = ((gcol_a >> 7) * row_tiles + (rblk >> 2)) * 512 + (gcol_a & 31) * 16 + (gcol_a >> 5 & 3) * 4 + (rblk & 3);
        int col_scale_b = ((gcol_b >> 7) * row_tiles + (rblk >> 2)) * 512 + (gcol_b & 31) * 16 + (gcol_b >> 5 & 3) * 4 + (rblk & 3);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_a) + (0)) = (unsigned char)(scale_6);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b) + (0)) = (unsigned char)(scale_7);
    }
    int lane_0 = tid & 31;
    int warp_0 = tid >> 5;
    int k8_0 = lane_0 & 7;
    int srow_0 = (k8_0 >> 1) * 16 + (lane_0 >> 3) * 2 + (k8_0 & 1);
    unsigned int sswizzle_0 = (unsigned int)(srow_0 >> 3 & 6 | srow_0 & 1);
    unsigned int cols_0[16];
    unsigned int strip_chunk0_1_2 = (unsigned int)(warp_0 * 2);
    unsigned int strip_src0_1_2 = tiles_addr + 8192 + (unsigned int)(srow_0 * 128) + ((strip_chunk0_1_2 ^ sswizzle_0) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_0[0]), "=r"(cols_0[1]), "=r"(cols_0[2]), "=r"(cols_0[3])
        : "r"(strip_src0_1_2)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_0[4]), "=r"(cols_0[5]), "=r"(cols_0[6]), "=r"(cols_0[7])
        : "r"(strip_src0_1_2 + 1024)
        : "memory");
    unsigned int strip_chunk0_1_3 = (unsigned int)(warp_0 * 2 + 1);
    unsigned int strip_src0_1_3 = tiles_addr + 8192 + (unsigned int)(srow_0 * 128) + ((strip_chunk0_1_3 ^ sswizzle_0) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_0[8]), "=r"(cols_0[9]), "=r"(cols_0[10]), "=r"(cols_0[11])
        : "r"(strip_src0_1_3)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_0[12]), "=r"(cols_0[13]), "=r"(cols_0[14]), "=r"(cols_0[15])
        : "r"(strip_src0_1_3 + 1024)
        : "memory");
    __nv_bfloat162 amax_a_0 = __as_bf16x2(0);
    __nv_bfloat162 amax_b_0 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_281 = amax_a_0;
    __nv_bfloat162 _hxb_281 = __as_bf16x2(cols_0[0]);
    unsigned int _hxr_281;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_281) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_281)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_281)));
    __nv_bfloat162 _max_280 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_281);
    amax_a_0 = _max_280;
    __nv_bfloat162 _hxa_283 = amax_b_0;
    __nv_bfloat162 _hxb_283 = __as_bf16x2(cols_0[8]);
    unsigned int _hxr_283;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_283) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_283)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_283)));
    __nv_bfloat162 _max_282 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_283);
    amax_b_0 = _max_282;
    __nv_bfloat162 _hxa_285 = amax_a_0;
    __nv_bfloat162 _hxb_285 = __as_bf16x2(cols_0[1]);
    unsigned int _hxr_285;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_285) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_285)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_285)));
    __nv_bfloat162 _max_284 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_285);
    amax_a_0 = _max_284;
    __nv_bfloat162 _hxa_287 = amax_b_0;
    __nv_bfloat162 _hxb_287 = __as_bf16x2(cols_0[9]);
    unsigned int _hxr_287;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_287) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_287)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_287)));
    __nv_bfloat162 _max_286 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_287);
    amax_b_0 = _max_286;
    __nv_bfloat162 _hxa_289 = amax_a_0;
    __nv_bfloat162 _hxb_289 = __as_bf16x2(cols_0[2]);
    unsigned int _hxr_289;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_289) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_289)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_289)));
    __nv_bfloat162 _max_288 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_289);
    amax_a_0 = _max_288;
    __nv_bfloat162 _hxa_291 = amax_b_0;
    __nv_bfloat162 _hxb_291 = __as_bf16x2(cols_0[10]);
    unsigned int _hxr_291;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_291) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_291)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_291)));
    __nv_bfloat162 _max_290 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_291);
    amax_b_0 = _max_290;
    __nv_bfloat162 _hxa_293 = amax_a_0;
    __nv_bfloat162 _hxb_293 = __as_bf16x2(cols_0[3]);
    unsigned int _hxr_293;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_293) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_293)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_293)));
    __nv_bfloat162 _max_292 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_293);
    amax_a_0 = _max_292;
    __nv_bfloat162 _hxa_295 = amax_b_0;
    __nv_bfloat162 _hxb_295 = __as_bf16x2(cols_0[11]);
    unsigned int _hxr_295;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_295) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_295)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_295)));
    __nv_bfloat162 _max_294 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_295);
    amax_b_0 = _max_294;
    __nv_bfloat162 _hxa_297 = amax_a_0;
    __nv_bfloat162 _hxb_297 = __as_bf16x2(cols_0[4]);
    unsigned int _hxr_297;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_297) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_297)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_297)));
    __nv_bfloat162 _max_296 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_297);
    amax_a_0 = _max_296;
    __nv_bfloat162 _hxa_299 = amax_b_0;
    __nv_bfloat162 _hxb_299 = __as_bf16x2(cols_0[12]);
    unsigned int _hxr_299;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_299) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_299)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_299)));
    __nv_bfloat162 _max_298 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_299);
    amax_b_0 = _max_298;
    __nv_bfloat162 _hxa_301 = amax_a_0;
    __nv_bfloat162 _hxb_301 = __as_bf16x2(cols_0[5]);
    unsigned int _hxr_301;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_301) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_301)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_301)));
    __nv_bfloat162 _max_300 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_301);
    amax_a_0 = _max_300;
    __nv_bfloat162 _hxa_303 = amax_b_0;
    __nv_bfloat162 _hxb_303 = __as_bf16x2(cols_0[13]);
    unsigned int _hxr_303;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_303) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_303)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_303)));
    __nv_bfloat162 _max_302 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_303);
    amax_b_0 = _max_302;
    __nv_bfloat162 _hxa_305 = amax_a_0;
    __nv_bfloat162 _hxb_305 = __as_bf16x2(cols_0[6]);
    unsigned int _hxr_305;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_305) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_305)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_305)));
    __nv_bfloat162 _max_304 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_305);
    amax_a_0 = _max_304;
    __nv_bfloat162 _hxa_307 = amax_b_0;
    __nv_bfloat162 _hxb_307 = __as_bf16x2(cols_0[14]);
    unsigned int _hxr_307;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_307) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_307)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_307)));
    __nv_bfloat162 _max_306 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_307);
    amax_b_0 = _max_306;
    __nv_bfloat162 _hxa_309 = amax_a_0;
    __nv_bfloat162 _hxb_309 = __as_bf16x2(cols_0[7]);
    unsigned int _hxr_309;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_309) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_309)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_309)));
    __nv_bfloat162 _max_308 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_309);
    amax_a_0 = _max_308;
    __nv_bfloat162 _hxa_311 = amax_b_0;
    __nv_bfloat162 _hxb_311 = __as_bf16x2(cols_0[15]);
    unsigned int _hxr_311;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_311) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_311)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_311)));
    __nv_bfloat162 _max_310 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_311);
    amax_b_0 = _max_310;
    unsigned int mag_a_0 = __as_u32(amax_a_0) & 2147450879;
    unsigned int mag_b_0 = __as_u32(amax_b_0) & 2147450879;
    unsigned int lows_ab_0 = mag_a_0 & 65535 | mag_b_0 << 16;
    unsigned int highs_ab_0 = mag_a_0 >> 16 | mag_b_0 & 4294901760u;
    __nv_bfloat162 col_max_0 = __hmax2_nan(__as_bf16x2(lows_ab_0), __as_bf16x2(highs_ab_0));
    unsigned int peer1_0 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(col_max_0), 1);
    col_max_0 = __hmax2_nan(col_max_0, __as_bf16x2(peer1_0));
    unsigned int col_max_bits_0 = __as_u32(col_max_0);
    unsigned int amax_bits_5 = (col_max_bits_0 & 65535) << 16;
    int scale_i32_5 = (int)(amax_bits_5 + 2031616 >> 23) - 8;
    if (scale_i32_5 < 0) {
        scale_i32_5 = 0;
    }
    if ((amax_bits_5 & 2139095040) == 2139095040) {
        scale_i32_5 = 255;
    }
    unsigned int scale_8 = (unsigned int)scale_i32_5;
    unsigned int inv_5 = 254 - scale_8 << 7;
    if (scale_8 == 255) {
        inv_5 = 32704;
    }
    unsigned int inv2_8 = inv_5 | inv_5 << 16;
    unsigned int amax_bits_6 = (col_max_bits_0 >> 16 & 65535) << 16;
    int scale_i32_6 = (int)(amax_bits_6 + 2031616 >> 23) - 8;
    if (scale_i32_6 < 0) {
        scale_i32_6 = 0;
    }
    if ((amax_bits_6 & 2139095040) == 2139095040) {
        scale_i32_6 = 255;
    }
    unsigned int scale_9 = (unsigned int)scale_i32_6;
    unsigned int inv_6 = 254 - scale_9 << 7;
    if (scale_9 == 255) {
        inv_6 = 32704;
    }
    unsigned int inv2_9 = inv_6 | inv_6 << 16;
    int gcol_a_0 = bx * 64 + K + warp_0 * 16 + (lane_0 >> 2);
    int gcol_b_0 = gcol_a_0 + 8;
    int out_row_0 = by * 64 + (lane_0 & 3) * 16;
    unsigned int words_a_0[4];
    unsigned int words_b_0[4];
    uint32_t bf16x2_mul_26;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_26) : "r"(cols_0[0]), "r"(inv2_8));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_23) : "r"(cols_0[1]), "r"(inv2_8));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_313;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_313) : "r"(bf16x2_mul_1_23));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_23);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_313) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_312;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_312) : "h"(e4m3x2_bf16x2_26), "h"(_cvt_satfinite_e4m3x2_bf16x2_313));
    words_a_0[0] = _mov_b32_pack_u16x2_312;
    uint32_t bf16x2_mul_27;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_27) : "r"(cols_0[8]), "r"(inv2_9));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_24) : "r"(cols_0[9]), "r"(inv2_9));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_315;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_315) : "r"(bf16x2_mul_1_24));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_24);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_315) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_314;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_314) : "h"(e4m3x2_bf16x2_27), "h"(_cvt_satfinite_e4m3x2_bf16x2_315));
    words_b_0[0] = _mov_b32_pack_u16x2_314;
    uint32_t bf16x2_mul_28;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_28) : "r"(cols_0[2]), "r"(inv2_8));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_25) : "r"(cols_0[3]), "r"(inv2_8));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_317;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_317) : "r"(bf16x2_mul_1_25));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_25);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_317) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_316;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_316) : "h"(e4m3x2_bf16x2_28), "h"(_cvt_satfinite_e4m3x2_bf16x2_317));
    words_a_0[1] = _mov_b32_pack_u16x2_316;
    uint32_t bf16x2_mul_29;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_29) : "r"(cols_0[10]), "r"(inv2_9));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_26) : "r"(cols_0[11]), "r"(inv2_9));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_319;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_319) : "r"(bf16x2_mul_1_26));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_26);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_319) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_318;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_318) : "h"(e4m3x2_bf16x2_29), "h"(_cvt_satfinite_e4m3x2_bf16x2_319));
    words_b_0[1] = _mov_b32_pack_u16x2_318;
    uint32_t bf16x2_mul_30;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_30) : "r"(cols_0[4]), "r"(inv2_8));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_27) : "r"(cols_0[5]), "r"(inv2_8));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_321;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_321) : "r"(bf16x2_mul_1_27));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_27);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_321) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_320;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_320) : "h"(e4m3x2_bf16x2_30), "h"(_cvt_satfinite_e4m3x2_bf16x2_321));
    words_a_0[2] = _mov_b32_pack_u16x2_320;
    uint32_t bf16x2_mul_31;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_31) : "r"(cols_0[12]), "r"(inv2_9));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_28) : "r"(cols_0[13]), "r"(inv2_9));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_323;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_323) : "r"(bf16x2_mul_1_28));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_28);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_323) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_322;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_322) : "h"(e4m3x2_bf16x2_31), "h"(_cvt_satfinite_e4m3x2_bf16x2_323));
    words_b_0[2] = _mov_b32_pack_u16x2_322;
    uint32_t bf16x2_mul_32;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_32) : "r"(cols_0[6]), "r"(inv2_8));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_29) : "r"(cols_0[7]), "r"(inv2_8));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_325;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_325) : "r"(bf16x2_mul_1_29));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_29);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_325) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_324;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_324) : "h"(e4m3x2_bf16x2_32), "h"(_cvt_satfinite_e4m3x2_bf16x2_325));
    words_a_0[3] = _mov_b32_pack_u16x2_324;
    uint32_t bf16x2_mul_33;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_33) : "r"(cols_0[14]), "r"(inv2_9));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_30) : "r"(cols_0[15]), "r"(inv2_9));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_327;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_327) : "r"(bf16x2_mul_1_30));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_30);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_327) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_326;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_326) : "h"(e4m3x2_bf16x2_33), "h"(_cvt_satfinite_e4m3x2_bf16x2_327));
    words_b_0[3] = _mov_b32_pack_u16x2_326;
    reinterpret_cast<int4*>(col_output + (gcol_a_0 * M + out_row_0))[0] = reinterpret_cast<int4*>(words_a_0 + 0)[0];
    reinterpret_cast<int4*>(col_output + (gcol_b_0 * M + out_row_0))[0] = reinterpret_cast<int4*>(words_b_0 + 0)[0];
    if ((lane_0 & 1) == 0) {
        int row_tiles_1 = M >> 7;
        int rblk_1 = by * 2 + ((lane_0 & 3) >> 1);
        int col_scale_a_1 = ((gcol_a_0 >> 7) * row_tiles_1 + (rblk_1 >> 2)) * 512 + (gcol_a_0 & 31) * 16 + (gcol_a_0 >> 5 & 3) * 4 + (rblk_1 & 3);
        int col_scale_b_1 = ((gcol_b_0 >> 7) * row_tiles_1 + (rblk_1 >> 2)) * 512 + (gcol_b_0 & 31) * 16 + (gcol_b_0 >> 5 & 3) * 4 + (rblk_1 & 3);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_a_1) + (0)) = (unsigned char)(scale_8);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b_1) + (0)) = (unsigned char)(scale_9);
    }
}

} // extern "C"
