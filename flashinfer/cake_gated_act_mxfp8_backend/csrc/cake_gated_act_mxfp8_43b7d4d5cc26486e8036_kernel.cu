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
#define SMEM_TILES_STAGE_BYTES 16384
#define SMEM_TILES_STRIDE 16384
#define SMEM_TOTAL 16384
#define THREADS 128
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
kernel_cake_gated_act_mxfp8_43b7d4d5cc26486e8036(__nv_bfloat16* __restrict__ gated_input, __nv_bfloat16* __restrict__ grad, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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
    int grow = by * 64 + row;
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
            : "=r"(_iv_2_0), "=r"(_iv_2_1), "=r"(_iv_2_2), "=r"(_iv_2_3), "=r"(_iv_2_4), "=r"(_iv_2_5), "=r"(_iv_2_6), "=r"(_iv_2_7) : "l"((const void*)(grad + (grow * K + col) + (0))) : "memory");
        _vec_load_2[0 + 0] = (uint32_t)_iv_2_0;
        _vec_load_2[0 + 1] = (uint32_t)_iv_2_1;
        _vec_load_2[0 + 2] = (uint32_t)_iv_2_2;
        _vec_load_2[0 + 3] = (uint32_t)_iv_2_3;
        _vec_load_2[0 + 4] = (uint32_t)_iv_2_4;
        _vec_load_2[0 + 5] = (uint32_t)_iv_2_5;
        _vec_load_2[0 + 6] = (uint32_t)_iv_2_6;
        _vec_load_2[0 + 7] = (uint32_t)_iv_2_7;
    }
    int grow_hi = grow + 32;
    int gate_index_hi = grow_hi * (2 * K) + col;
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
            : "=r"(_iv_3_0), "=r"(_iv_3_1), "=r"(_iv_3_2), "=r"(_iv_3_3), "=r"(_iv_3_4), "=r"(_iv_3_5), "=r"(_iv_3_6), "=r"(_iv_3_7) : "l"((const void*)(gated_input + gate_index_hi + (0))) : "memory");
        _vec_load_3[0 + 0] = (uint32_t)_iv_3_0;
        _vec_load_3[0 + 1] = (uint32_t)_iv_3_1;
        _vec_load_3[0 + 2] = (uint32_t)_iv_3_2;
        _vec_load_3[0 + 3] = (uint32_t)_iv_3_3;
        _vec_load_3[0 + 4] = (uint32_t)_iv_3_4;
        _vec_load_3[0 + 5] = (uint32_t)_iv_3_5;
        _vec_load_3[0 + 6] = (uint32_t)_iv_3_6;
        _vec_load_3[0 + 7] = (uint32_t)_iv_3_7;
    }
    unsigned int _vec_load_4[8];
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
            : "=r"(_iv_4_0), "=r"(_iv_4_1), "=r"(_iv_4_2), "=r"(_iv_4_3), "=r"(_iv_4_4), "=r"(_iv_4_5), "=r"(_iv_4_6), "=r"(_iv_4_7) : "l"((const void*)(gated_input + (gate_index_hi + K) + (0))) : "memory");
        _vec_load_4[0 + 0] = (uint32_t)_iv_4_0;
        _vec_load_4[0 + 1] = (uint32_t)_iv_4_1;
        _vec_load_4[0 + 2] = (uint32_t)_iv_4_2;
        _vec_load_4[0 + 3] = (uint32_t)_iv_4_3;
        _vec_load_4[0 + 4] = (uint32_t)_iv_4_4;
        _vec_load_4[0 + 5] = (uint32_t)_iv_4_5;
        _vec_load_4[0 + 6] = (uint32_t)_iv_4_6;
        _vec_load_4[0 + 7] = (uint32_t)_iv_4_7;
    }
    unsigned int _vec_load_5[8];
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
            : "=r"(_iv_5_0), "=r"(_iv_5_1), "=r"(_iv_5_2), "=r"(_iv_5_3), "=r"(_iv_5_4), "=r"(_iv_5_5), "=r"(_iv_5_6), "=r"(_iv_5_7) : "l"((const void*)(grad + (grow_hi * K + col) + (0))) : "memory");
        _vec_load_5[0 + 0] = (uint32_t)_iv_5_0;
        _vec_load_5[0 + 1] = (uint32_t)_iv_5_1;
        _vec_load_5[0 + 2] = (uint32_t)_iv_5_2;
        _vec_load_5[0 + 3] = (uint32_t)_iv_5_3;
        _vec_load_5[0 + 4] = (uint32_t)_iv_5_4;
        _vec_load_5[0 + 5] = (uint32_t)_iv_5_5;
        _vec_load_5[0 + 6] = (uint32_t)_iv_5_6;
        _vec_load_5[0 + 7] = (uint32_t)_iv_5_7;
    }
    unsigned int dgate[8];
    unsigned int dup[8];
    unsigned int dgate_hi[8];
    unsigned int dup_hi[8];
    __nv_bfloat162 lowest = __as_bf16x2(_vec_load_0[0]);
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[7]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_3[0]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_3[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_3[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_3[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_3[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_3[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_3[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_3[7]));
    float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(lowest);
    float _min_7;
    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_7) : "f"(_cvt_f32_0.x), "f"(_cvt_f32_0.y));
    int _warp_vote_6 = __all_sync(0xFFFFFFFF, _min_7 > -87.0f);
    if (_warp_vote_6 != 0) {
        float2 _cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 _cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 _cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float2 _f32x2_mul_b_8 = make_float2(-1.442695f, -1.442695f);
        float2 t_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_1) : "l"(*(const unsigned long long*)&_cvt_f32_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_8));
        float e0_1 = approx_exp2(t_1.x);
        float e1_1 = approx_exp2(t_1.y);
        float2 nd_1 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_1, e1_1));
        float d0_1 = -nd_1.x;
        float d1_1 = -nd_1.y;
        float r0_1 = approx_rcp(d0_1);
        float r1_1 = approx_rcp(d1_1);
        float2 r_1 = make_float2(r0_1, r1_1);
        float2 s_1 = fma_f32x2_rn_noftz(r_1, fma_f32x2_rn_noftz(nd_1, r_1, make_float2(1.0f, 1.0f)), r_1);
        float2 s_0 = s_1;
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
        float2 _f32x2_mul_b_10 = make_float2(-1.442695f, -1.442695f);
        float2 t_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_3) : "l"(*(const unsigned long long*)&_cvt_f32_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_10));
        float e0_3 = approx_exp2(t_3.x);
        float e1_3 = approx_exp2(t_3.y);
        float2 nd_3 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_3, e1_3));
        float d0_3 = -nd_3.x;
        float d1_3 = -nd_3.y;
        float r0_3 = approx_rcp(d0_3);
        float r1_3 = approx_rcp(d1_3);
        float2 r_3 = make_float2(r0_3, r1_3);
        float2 s_3 = fma_f32x2_rn_noftz(r_3, fma_f32x2_rn_noftz(nd_3, r_3, make_float2(1.0f, 1.0f)), r_3);
        float2 s_2 = s_3;
        float2 act_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_2) : "l"(*(const unsigned long long*)&_cvt_f32_4), "l"(*(const unsigned long long*)&s_2));
        float2 _f32x2_mul_a_11 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_2), s_2), _cvt_f32_6);
        float2 _mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_11), "l"(*(const unsigned long long*)&_cvt_f32_5));
        float2 _mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_3) : "l"(*(const unsigned long long*)&act_2), "l"(*(const unsigned long long*)&_cvt_f32_6));
        __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(_mul_2.x, _mul_2.y));
        __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(_mul_3.x, _mul_3.y));
        dgate[1] = __as_u32(_bf16x2_2);
        dup[1] = __as_u32(_bf16x2_3);
        float2 _cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 _cvt_f32_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 _cvt_f32_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float2 _f32x2_mul_b_12 = make_float2(-1.442695f, -1.442695f);
        float2 t_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_5) : "l"(*(const unsigned long long*)&_cvt_f32_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_12));
        float e0_5 = approx_exp2(t_5.x);
        float e1_5 = approx_exp2(t_5.y);
        float2 nd_5 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_5, e1_5));
        float d0_5 = -nd_5.x;
        float d1_5 = -nd_5.y;
        float r0_5 = approx_rcp(d0_5);
        float r1_5 = approx_rcp(d1_5);
        float2 r_5 = make_float2(r0_5, r1_5);
        float2 s_5 = fma_f32x2_rn_noftz(r_5, fma_f32x2_rn_noftz(nd_5, r_5, make_float2(1.0f, 1.0f)), r_5);
        float2 s_4 = s_5;
        float2 act_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_4) : "l"(*(const unsigned long long*)&_cvt_f32_7), "l"(*(const unsigned long long*)&s_4));
        float2 _f32x2_mul_a_13 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_4), s_4), _cvt_f32_9);
        float2 _mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_13), "l"(*(const unsigned long long*)&_cvt_f32_8));
        float2 _mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_5) : "l"(*(const unsigned long long*)&act_4), "l"(*(const unsigned long long*)&_cvt_f32_9));
        __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(_mul_4.x, _mul_4.y));
        __nv_bfloat162 _bf16x2_5 = __float22bfloat162_rn(make_float2(_mul_5.x, _mul_5.y));
        dgate[2] = __as_u32(_bf16x2_4);
        dup[2] = __as_u32(_bf16x2_5);
        float2 _cvt_f32_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 _cvt_f32_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 _cvt_f32_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float2 _f32x2_mul_b_14 = make_float2(-1.442695f, -1.442695f);
        float2 t_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_7) : "l"(*(const unsigned long long*)&_cvt_f32_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_14));
        float e0_7 = approx_exp2(t_7.x);
        float e1_7 = approx_exp2(t_7.y);
        float2 nd_7 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_7, e1_7));
        float d0_7 = -nd_7.x;
        float d1_7 = -nd_7.y;
        float r0_7 = approx_rcp(d0_7);
        float r1_7 = approx_rcp(d1_7);
        float2 r_7 = make_float2(r0_7, r1_7);
        float2 s_7 = fma_f32x2_rn_noftz(r_7, fma_f32x2_rn_noftz(nd_7, r_7, make_float2(1.0f, 1.0f)), r_7);
        float2 s_6 = s_7;
        float2 act_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_6) : "l"(*(const unsigned long long*)&_cvt_f32_10), "l"(*(const unsigned long long*)&s_6));
        float2 _f32x2_mul_a_15 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_6), s_6), _cvt_f32_12);
        float2 _mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&_cvt_f32_11));
        float2 _mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_7) : "l"(*(const unsigned long long*)&act_6), "l"(*(const unsigned long long*)&_cvt_f32_12));
        __nv_bfloat162 _bf16x2_6 = __float22bfloat162_rn(make_float2(_mul_6.x, _mul_6.y));
        __nv_bfloat162 _bf16x2_7 = __float22bfloat162_rn(make_float2(_mul_7.x, _mul_7.y));
        dgate[3] = __as_u32(_bf16x2_6);
        dup[3] = __as_u32(_bf16x2_7);
        float2 _cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 _cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float2 _cvt_f32_15 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[4]));
        float2 _f32x2_mul_b_16 = make_float2(-1.442695f, -1.442695f);
        float2 t_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_9) : "l"(*(const unsigned long long*)&_cvt_f32_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_16));
        float e0_9 = approx_exp2(t_9.x);
        float e1_9 = approx_exp2(t_9.y);
        float2 nd_9 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_9, e1_9));
        float d0_9 = -nd_9.x;
        float d1_9 = -nd_9.y;
        float r0_9 = approx_rcp(d0_9);
        float r1_9 = approx_rcp(d1_9);
        float2 r_9 = make_float2(r0_9, r1_9);
        float2 s_9 = fma_f32x2_rn_noftz(r_9, fma_f32x2_rn_noftz(nd_9, r_9, make_float2(1.0f, 1.0f)), r_9);
        float2 s_8 = s_9;
        float2 act_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_8) : "l"(*(const unsigned long long*)&_cvt_f32_13), "l"(*(const unsigned long long*)&s_8));
        float2 _f32x2_mul_a_17 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_8, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_8), s_8), _cvt_f32_15);
        float2 _mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&_cvt_f32_14));
        float2 _mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_9) : "l"(*(const unsigned long long*)&act_8), "l"(*(const unsigned long long*)&_cvt_f32_15));
        __nv_bfloat162 _bf16x2_8 = __float22bfloat162_rn(make_float2(_mul_8.x, _mul_8.y));
        __nv_bfloat162 _bf16x2_9 = __float22bfloat162_rn(make_float2(_mul_9.x, _mul_9.y));
        dgate[4] = __as_u32(_bf16x2_8);
        dup[4] = __as_u32(_bf16x2_9);
        float2 _cvt_f32_16 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 _cvt_f32_17 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float2 _cvt_f32_18 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[5]));
        float2 _f32x2_mul_b_18 = make_float2(-1.442695f, -1.442695f);
        float2 t_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_11) : "l"(*(const unsigned long long*)&_cvt_f32_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_18));
        float e0_11 = approx_exp2(t_11.x);
        float e1_11 = approx_exp2(t_11.y);
        float2 nd_11 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_11, e1_11));
        float d0_11 = -nd_11.x;
        float d1_11 = -nd_11.y;
        float r0_11 = approx_rcp(d0_11);
        float r1_11 = approx_rcp(d1_11);
        float2 r_11 = make_float2(r0_11, r1_11);
        float2 s_11 = fma_f32x2_rn_noftz(r_11, fma_f32x2_rn_noftz(nd_11, r_11, make_float2(1.0f, 1.0f)), r_11);
        float2 s_10 = s_11;
        float2 act_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_10) : "l"(*(const unsigned long long*)&_cvt_f32_16), "l"(*(const unsigned long long*)&s_10));
        float2 _f32x2_mul_a_19 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_10, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_10), s_10), _cvt_f32_18);
        float2 _mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&_cvt_f32_17));
        float2 _mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_11) : "l"(*(const unsigned long long*)&act_10), "l"(*(const unsigned long long*)&_cvt_f32_18));
        __nv_bfloat162 _bf16x2_10 = __float22bfloat162_rn(make_float2(_mul_10.x, _mul_10.y));
        __nv_bfloat162 _bf16x2_11 = __float22bfloat162_rn(make_float2(_mul_11.x, _mul_11.y));
        dgate[5] = __as_u32(_bf16x2_10);
        dup[5] = __as_u32(_bf16x2_11);
        float2 _cvt_f32_19 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 _cvt_f32_20 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float2 _cvt_f32_21 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[6]));
        float2 _f32x2_mul_b_20 = make_float2(-1.442695f, -1.442695f);
        float2 t_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_13) : "l"(*(const unsigned long long*)&_cvt_f32_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
        float e0_13 = approx_exp2(t_13.x);
        float e1_13 = approx_exp2(t_13.y);
        float2 nd_13 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_13, e1_13));
        float d0_13 = -nd_13.x;
        float d1_13 = -nd_13.y;
        float r0_13 = approx_rcp(d0_13);
        float r1_13 = approx_rcp(d1_13);
        float2 r_13 = make_float2(r0_13, r1_13);
        float2 s_13 = fma_f32x2_rn_noftz(r_13, fma_f32x2_rn_noftz(nd_13, r_13, make_float2(1.0f, 1.0f)), r_13);
        float2 s_12 = s_13;
        float2 act_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_12) : "l"(*(const unsigned long long*)&_cvt_f32_19), "l"(*(const unsigned long long*)&s_12));
        float2 _f32x2_mul_a_21 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_12, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_12), s_12), _cvt_f32_21);
        float2 _mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&_cvt_f32_20));
        float2 _mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_13) : "l"(*(const unsigned long long*)&act_12), "l"(*(const unsigned long long*)&_cvt_f32_21));
        __nv_bfloat162 _bf16x2_12 = __float22bfloat162_rn(make_float2(_mul_12.x, _mul_12.y));
        __nv_bfloat162 _bf16x2_13 = __float22bfloat162_rn(make_float2(_mul_13.x, _mul_13.y));
        dgate[6] = __as_u32(_bf16x2_12);
        dup[6] = __as_u32(_bf16x2_13);
        float2 _cvt_f32_22 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 _cvt_f32_23 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float2 _cvt_f32_24 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[7]));
        float2 _f32x2_mul_b_22 = make_float2(-1.442695f, -1.442695f);
        float2 t_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_15) : "l"(*(const unsigned long long*)&_cvt_f32_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_22));
        float e0_15 = approx_exp2(t_15.x);
        float e1_15 = approx_exp2(t_15.y);
        float2 nd_15 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_15, e1_15));
        float d0_15 = -nd_15.x;
        float d1_15 = -nd_15.y;
        float r0_15 = approx_rcp(d0_15);
        float r1_15 = approx_rcp(d1_15);
        float2 r_15 = make_float2(r0_15, r1_15);
        float2 s_15 = fma_f32x2_rn_noftz(r_15, fma_f32x2_rn_noftz(nd_15, r_15, make_float2(1.0f, 1.0f)), r_15);
        float2 s_14 = s_15;
        float2 act_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_14) : "l"(*(const unsigned long long*)&_cvt_f32_22), "l"(*(const unsigned long long*)&s_14));
        float2 _f32x2_mul_a_23 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_14, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_14), s_14), _cvt_f32_24);
        float2 _mul_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&_cvt_f32_23));
        float2 _mul_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_15) : "l"(*(const unsigned long long*)&act_14), "l"(*(const unsigned long long*)&_cvt_f32_24));
        __nv_bfloat162 _bf16x2_14 = __float22bfloat162_rn(make_float2(_mul_14.x, _mul_14.y));
        __nv_bfloat162 _bf16x2_15 = __float22bfloat162_rn(make_float2(_mul_15.x, _mul_15.y));
        dgate[7] = __as_u32(_bf16x2_14);
        dup[7] = __as_u32(_bf16x2_15);
        float2 _cvt_f32_25 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
        float2 _cvt_f32_26 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[0]));
        float2 _cvt_f32_27 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[0]));
        float2 _f32x2_mul_b_24 = make_float2(-1.442695f, -1.442695f);
        float2 t_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_17) : "l"(*(const unsigned long long*)&_cvt_f32_25), "l"(*(const unsigned long long*)&_f32x2_mul_b_24));
        float e0_17 = approx_exp2(t_17.x);
        float e1_17 = approx_exp2(t_17.y);
        float2 nd_17 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_17, e1_17));
        float d0_17 = -nd_17.x;
        float d1_17 = -nd_17.y;
        float r0_17 = approx_rcp(d0_17);
        float r1_17 = approx_rcp(d1_17);
        float2 r_17 = make_float2(r0_17, r1_17);
        float2 s_17 = fma_f32x2_rn_noftz(r_17, fma_f32x2_rn_noftz(nd_17, r_17, make_float2(1.0f, 1.0f)), r_17);
        float2 s_16 = s_17;
        float2 act_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_16) : "l"(*(const unsigned long long*)&_cvt_f32_25), "l"(*(const unsigned long long*)&s_16));
        float2 _f32x2_mul_a_25 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_16, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_16), s_16), _cvt_f32_27);
        float2 _mul_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_25), "l"(*(const unsigned long long*)&_cvt_f32_26));
        float2 _mul_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_17) : "l"(*(const unsigned long long*)&act_16), "l"(*(const unsigned long long*)&_cvt_f32_27));
        __nv_bfloat162 _bf16x2_16 = __float22bfloat162_rn(make_float2(_mul_16.x, _mul_16.y));
        __nv_bfloat162 _bf16x2_17 = __float22bfloat162_rn(make_float2(_mul_17.x, _mul_17.y));
        dgate_hi[0] = __as_u32(_bf16x2_16);
        dup_hi[0] = __as_u32(_bf16x2_17);
        float2 _cvt_f32_28 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[1]));
        float2 _cvt_f32_29 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[1]));
        float2 _cvt_f32_30 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[1]));
        float2 _f32x2_mul_b_26 = make_float2(-1.442695f, -1.442695f);
        float2 t_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_19) : "l"(*(const unsigned long long*)&_cvt_f32_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_26));
        float e0_19 = approx_exp2(t_19.x);
        float e1_19 = approx_exp2(t_19.y);
        float2 nd_19 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_19, e1_19));
        float d0_19 = -nd_19.x;
        float d1_19 = -nd_19.y;
        float r0_19 = approx_rcp(d0_19);
        float r1_19 = approx_rcp(d1_19);
        float2 r_19 = make_float2(r0_19, r1_19);
        float2 s_19 = fma_f32x2_rn_noftz(r_19, fma_f32x2_rn_noftz(nd_19, r_19, make_float2(1.0f, 1.0f)), r_19);
        float2 s_18 = s_19;
        float2 act_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_18) : "l"(*(const unsigned long long*)&_cvt_f32_28), "l"(*(const unsigned long long*)&s_18));
        float2 _f32x2_mul_a_27 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_18, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_18), s_18), _cvt_f32_30);
        float2 _mul_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_27), "l"(*(const unsigned long long*)&_cvt_f32_29));
        float2 _mul_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_19) : "l"(*(const unsigned long long*)&act_18), "l"(*(const unsigned long long*)&_cvt_f32_30));
        __nv_bfloat162 _bf16x2_18 = __float22bfloat162_rn(make_float2(_mul_18.x, _mul_18.y));
        __nv_bfloat162 _bf16x2_19 = __float22bfloat162_rn(make_float2(_mul_19.x, _mul_19.y));
        dgate_hi[1] = __as_u32(_bf16x2_18);
        dup_hi[1] = __as_u32(_bf16x2_19);
        float2 _cvt_f32_31 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[2]));
        float2 _cvt_f32_32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[2]));
        float2 _cvt_f32_33 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[2]));
        float2 _f32x2_mul_b_28 = make_float2(-1.442695f, -1.442695f);
        float2 t_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_21) : "l"(*(const unsigned long long*)&_cvt_f32_31), "l"(*(const unsigned long long*)&_f32x2_mul_b_28));
        float e0_21 = approx_exp2(t_21.x);
        float e1_21 = approx_exp2(t_21.y);
        float2 nd_21 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_21, e1_21));
        float d0_21 = -nd_21.x;
        float d1_21 = -nd_21.y;
        float r0_21 = approx_rcp(d0_21);
        float r1_21 = approx_rcp(d1_21);
        float2 r_21 = make_float2(r0_21, r1_21);
        float2 s_21 = fma_f32x2_rn_noftz(r_21, fma_f32x2_rn_noftz(nd_21, r_21, make_float2(1.0f, 1.0f)), r_21);
        float2 s_20 = s_21;
        float2 act_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_20) : "l"(*(const unsigned long long*)&_cvt_f32_31), "l"(*(const unsigned long long*)&s_20));
        float2 _f32x2_mul_a_29 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_20, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_20), s_20), _cvt_f32_33);
        float2 _mul_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&_cvt_f32_32));
        float2 _mul_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_21) : "l"(*(const unsigned long long*)&act_20), "l"(*(const unsigned long long*)&_cvt_f32_33));
        __nv_bfloat162 _bf16x2_20 = __float22bfloat162_rn(make_float2(_mul_20.x, _mul_20.y));
        __nv_bfloat162 _bf16x2_21 = __float22bfloat162_rn(make_float2(_mul_21.x, _mul_21.y));
        dgate_hi[2] = __as_u32(_bf16x2_20);
        dup_hi[2] = __as_u32(_bf16x2_21);
        float2 _cvt_f32_34 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[3]));
        float2 _cvt_f32_35 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[3]));
        float2 _cvt_f32_36 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[3]));
        float2 _f32x2_mul_b_30 = make_float2(-1.442695f, -1.442695f);
        float2 t_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_23) : "l"(*(const unsigned long long*)&_cvt_f32_34), "l"(*(const unsigned long long*)&_f32x2_mul_b_30));
        float e0_23 = approx_exp2(t_23.x);
        float e1_23 = approx_exp2(t_23.y);
        float2 nd_23 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_23, e1_23));
        float d0_23 = -nd_23.x;
        float d1_23 = -nd_23.y;
        float r0_23 = approx_rcp(d0_23);
        float r1_23 = approx_rcp(d1_23);
        float2 r_23 = make_float2(r0_23, r1_23);
        float2 s_23 = fma_f32x2_rn_noftz(r_23, fma_f32x2_rn_noftz(nd_23, r_23, make_float2(1.0f, 1.0f)), r_23);
        float2 s_22 = s_23;
        float2 act_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_22) : "l"(*(const unsigned long long*)&_cvt_f32_34), "l"(*(const unsigned long long*)&s_22));
        float2 _f32x2_mul_a_31 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_22, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_22), s_22), _cvt_f32_36);
        float2 _mul_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_31), "l"(*(const unsigned long long*)&_cvt_f32_35));
        float2 _mul_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_23) : "l"(*(const unsigned long long*)&act_22), "l"(*(const unsigned long long*)&_cvt_f32_36));
        __nv_bfloat162 _bf16x2_22 = __float22bfloat162_rn(make_float2(_mul_22.x, _mul_22.y));
        __nv_bfloat162 _bf16x2_23 = __float22bfloat162_rn(make_float2(_mul_23.x, _mul_23.y));
        dgate_hi[3] = __as_u32(_bf16x2_22);
        dup_hi[3] = __as_u32(_bf16x2_23);
        float2 _cvt_f32_37 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[4]));
        float2 _cvt_f32_38 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[4]));
        float2 _cvt_f32_39 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[4]));
        float2 _f32x2_mul_b_32 = make_float2(-1.442695f, -1.442695f);
        float2 t_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_25) : "l"(*(const unsigned long long*)&_cvt_f32_37), "l"(*(const unsigned long long*)&_f32x2_mul_b_32));
        float e0_25 = approx_exp2(t_25.x);
        float e1_25 = approx_exp2(t_25.y);
        float2 nd_25 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_25, e1_25));
        float d0_25 = -nd_25.x;
        float d1_25 = -nd_25.y;
        float r0_25 = approx_rcp(d0_25);
        float r1_25 = approx_rcp(d1_25);
        float2 r_25 = make_float2(r0_25, r1_25);
        float2 s_25 = fma_f32x2_rn_noftz(r_25, fma_f32x2_rn_noftz(nd_25, r_25, make_float2(1.0f, 1.0f)), r_25);
        float2 s_24 = s_25;
        float2 act_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_24) : "l"(*(const unsigned long long*)&_cvt_f32_37), "l"(*(const unsigned long long*)&s_24));
        float2 _f32x2_mul_a_33 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_24, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_24), s_24), _cvt_f32_39);
        float2 _mul_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&_cvt_f32_38));
        float2 _mul_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_25) : "l"(*(const unsigned long long*)&act_24), "l"(*(const unsigned long long*)&_cvt_f32_39));
        __nv_bfloat162 _bf16x2_24 = __float22bfloat162_rn(make_float2(_mul_24.x, _mul_24.y));
        __nv_bfloat162 _bf16x2_25 = __float22bfloat162_rn(make_float2(_mul_25.x, _mul_25.y));
        dgate_hi[4] = __as_u32(_bf16x2_24);
        dup_hi[4] = __as_u32(_bf16x2_25);
        float2 _cvt_f32_40 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[5]));
        float2 _cvt_f32_41 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[5]));
        float2 _cvt_f32_42 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[5]));
        float2 _f32x2_mul_b_34 = make_float2(-1.442695f, -1.442695f);
        float2 t_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_27) : "l"(*(const unsigned long long*)&_cvt_f32_40), "l"(*(const unsigned long long*)&_f32x2_mul_b_34));
        float e0_27 = approx_exp2(t_27.x);
        float e1_27 = approx_exp2(t_27.y);
        float2 nd_27 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_27, e1_27));
        float d0_27 = -nd_27.x;
        float d1_27 = -nd_27.y;
        float r0_27 = approx_rcp(d0_27);
        float r1_27 = approx_rcp(d1_27);
        float2 r_27 = make_float2(r0_27, r1_27);
        float2 s_27 = fma_f32x2_rn_noftz(r_27, fma_f32x2_rn_noftz(nd_27, r_27, make_float2(1.0f, 1.0f)), r_27);
        float2 s_26 = s_27;
        float2 act_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_26) : "l"(*(const unsigned long long*)&_cvt_f32_40), "l"(*(const unsigned long long*)&s_26));
        float2 _f32x2_mul_a_35 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_26, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_26), s_26), _cvt_f32_42);
        float2 _mul_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&_cvt_f32_41));
        float2 _mul_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_27) : "l"(*(const unsigned long long*)&act_26), "l"(*(const unsigned long long*)&_cvt_f32_42));
        __nv_bfloat162 _bf16x2_26 = __float22bfloat162_rn(make_float2(_mul_26.x, _mul_26.y));
        __nv_bfloat162 _bf16x2_27 = __float22bfloat162_rn(make_float2(_mul_27.x, _mul_27.y));
        dgate_hi[5] = __as_u32(_bf16x2_26);
        dup_hi[5] = __as_u32(_bf16x2_27);
        float2 _cvt_f32_43 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[6]));
        float2 _cvt_f32_44 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[6]));
        float2 _cvt_f32_45 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[6]));
        float2 _f32x2_mul_b_36 = make_float2(-1.442695f, -1.442695f);
        float2 t_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_29) : "l"(*(const unsigned long long*)&_cvt_f32_43), "l"(*(const unsigned long long*)&_f32x2_mul_b_36));
        float e0_29 = approx_exp2(t_29.x);
        float e1_29 = approx_exp2(t_29.y);
        float2 nd_29 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_29, e1_29));
        float d0_29 = -nd_29.x;
        float d1_29 = -nd_29.y;
        float r0_29 = approx_rcp(d0_29);
        float r1_29 = approx_rcp(d1_29);
        float2 r_29 = make_float2(r0_29, r1_29);
        float2 s_29 = fma_f32x2_rn_noftz(r_29, fma_f32x2_rn_noftz(nd_29, r_29, make_float2(1.0f, 1.0f)), r_29);
        float2 s_28 = s_29;
        float2 act_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_28) : "l"(*(const unsigned long long*)&_cvt_f32_43), "l"(*(const unsigned long long*)&s_28));
        float2 _f32x2_mul_a_37 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_28, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_28), s_28), _cvt_f32_45);
        float2 _mul_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_37), "l"(*(const unsigned long long*)&_cvt_f32_44));
        float2 _mul_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_29) : "l"(*(const unsigned long long*)&act_28), "l"(*(const unsigned long long*)&_cvt_f32_45));
        __nv_bfloat162 _bf16x2_28 = __float22bfloat162_rn(make_float2(_mul_28.x, _mul_28.y));
        __nv_bfloat162 _bf16x2_29 = __float22bfloat162_rn(make_float2(_mul_29.x, _mul_29.y));
        dgate_hi[6] = __as_u32(_bf16x2_28);
        dup_hi[6] = __as_u32(_bf16x2_29);
        float2 _cvt_f32_46 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[7]));
        float2 _cvt_f32_47 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[7]));
        float2 _cvt_f32_48 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[7]));
        float2 _f32x2_mul_b_38 = make_float2(-1.442695f, -1.442695f);
        float2 t_31;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_31) : "l"(*(const unsigned long long*)&_cvt_f32_46), "l"(*(const unsigned long long*)&_f32x2_mul_b_38));
        float e0_31 = approx_exp2(t_31.x);
        float e1_31 = approx_exp2(t_31.y);
        float2 nd_31 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_31, e1_31));
        float d0_31 = -nd_31.x;
        float d1_31 = -nd_31.y;
        float r0_31 = approx_rcp(d0_31);
        float r1_31 = approx_rcp(d1_31);
        float2 r_31 = make_float2(r0_31, r1_31);
        float2 s_31 = fma_f32x2_rn_noftz(r_31, fma_f32x2_rn_noftz(nd_31, r_31, make_float2(1.0f, 1.0f)), r_31);
        float2 s_30 = s_31;
        float2 act_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_30) : "l"(*(const unsigned long long*)&_cvt_f32_46), "l"(*(const unsigned long long*)&s_30));
        float2 _f32x2_mul_a_39 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_30, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_30), s_30), _cvt_f32_48);
        float2 _mul_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_39), "l"(*(const unsigned long long*)&_cvt_f32_47));
        float2 _mul_31;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_31) : "l"(*(const unsigned long long*)&act_30), "l"(*(const unsigned long long*)&_cvt_f32_48));
        __nv_bfloat162 _bf16x2_30 = __float22bfloat162_rn(make_float2(_mul_30.x, _mul_30.y));
        __nv_bfloat162 _bf16x2_31 = __float22bfloat162_rn(make_float2(_mul_31.x, _mul_31.y));
        dgate_hi[7] = __as_u32(_bf16x2_30);
        dup_hi[7] = __as_u32(_bf16x2_31);
    } else {
        int band = 0;
        float2 _cvt_f32_49 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 _cvt_f32_50 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 _cvt_f32_51 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float2 _f32x2_mul_b_40 = make_float2(-1.442695f, -1.442695f);
        float2 t_33;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_33) : "l"(*(const unsigned long long*)&_cvt_f32_49), "l"(*(const unsigned long long*)&_f32x2_mul_b_40));
        float e0_33 = approx_exp2(t_33.x);
        float e1_33 = approx_exp2(t_33.y);
        float2 nd_33 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_33, e1_33));
        float d0_33 = -nd_33.x;
        float d1_33 = -nd_33.y;
        float r0_33 = approx_rcp(d0_33);
        float r1_33 = approx_rcp(d1_33);
        float2 r_33 = make_float2(r0_33, r1_33);
        float2 s_33 = fma_f32x2_rn_noftz(r_33, fma_f32x2_rn_noftz(nd_33, r_33, make_float2(1.0f, 1.0f)), r_33);
        float t0_33 = s_33.x;
        float t1_33 = s_33.y;
        band = band | ((fminf(t0_33, t1_33) == 0.0f) ? 1 : 0);
        float2 s_32 = make_float2(max_noftz(t0_33, 0.0f), max_noftz(t1_33, 0.0f));
        float2 act_32;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_32) : "l"(*(const unsigned long long*)&_cvt_f32_49), "l"(*(const unsigned long long*)&s_32));
        float2 _f32x2_mul_a_41 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_32, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_32), s_32), _cvt_f32_51);
        float2 _mul_32;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_32) : "l"(*(const unsigned long long*)&_f32x2_mul_a_41), "l"(*(const unsigned long long*)&_cvt_f32_50));
        float2 _mul_33;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_33) : "l"(*(const unsigned long long*)&act_32), "l"(*(const unsigned long long*)&_cvt_f32_51));
        __nv_bfloat162 _bf16x2_32 = __float22bfloat162_rn(make_float2(_mul_32.x, _mul_32.y));
        __nv_bfloat162 _bf16x2_33 = __float22bfloat162_rn(make_float2(_mul_33.x, _mul_33.y));
        dgate[0] = __as_u32(_bf16x2_32);
        dup[0] = __as_u32(_bf16x2_33);
        float2 _cvt_f32_52 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 _cvt_f32_53 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 _cvt_f32_54 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float2 _f32x2_mul_b_42 = make_float2(-1.442695f, -1.442695f);
        float2 t_35;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_35) : "l"(*(const unsigned long long*)&_cvt_f32_52), "l"(*(const unsigned long long*)&_f32x2_mul_b_42));
        float e0_35 = approx_exp2(t_35.x);
        float e1_35 = approx_exp2(t_35.y);
        float2 nd_35 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_35, e1_35));
        float d0_35 = -nd_35.x;
        float d1_35 = -nd_35.y;
        float r0_35 = approx_rcp(d0_35);
        float r1_35 = approx_rcp(d1_35);
        float2 r_35 = make_float2(r0_35, r1_35);
        float2 s_35 = fma_f32x2_rn_noftz(r_35, fma_f32x2_rn_noftz(nd_35, r_35, make_float2(1.0f, 1.0f)), r_35);
        float t0_35 = s_35.x;
        float t1_35 = s_35.y;
        band = band | ((fminf(t0_35, t1_35) == 0.0f) ? 2 : 0);
        float2 s_34 = make_float2(max_noftz(t0_35, 0.0f), max_noftz(t1_35, 0.0f));
        float2 act_34;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_34) : "l"(*(const unsigned long long*)&_cvt_f32_52), "l"(*(const unsigned long long*)&s_34));
        float2 _f32x2_mul_a_43 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_34, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_34), s_34), _cvt_f32_54);
        float2 _mul_34;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_34) : "l"(*(const unsigned long long*)&_f32x2_mul_a_43), "l"(*(const unsigned long long*)&_cvt_f32_53));
        float2 _mul_35;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_35) : "l"(*(const unsigned long long*)&act_34), "l"(*(const unsigned long long*)&_cvt_f32_54));
        __nv_bfloat162 _bf16x2_34 = __float22bfloat162_rn(make_float2(_mul_34.x, _mul_34.y));
        __nv_bfloat162 _bf16x2_35 = __float22bfloat162_rn(make_float2(_mul_35.x, _mul_35.y));
        dgate[1] = __as_u32(_bf16x2_34);
        dup[1] = __as_u32(_bf16x2_35);
        float2 _cvt_f32_55 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 _cvt_f32_56 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 _cvt_f32_57 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float2 _f32x2_mul_b_44 = make_float2(-1.442695f, -1.442695f);
        float2 t_37;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_37) : "l"(*(const unsigned long long*)&_cvt_f32_55), "l"(*(const unsigned long long*)&_f32x2_mul_b_44));
        float e0_37 = approx_exp2(t_37.x);
        float e1_37 = approx_exp2(t_37.y);
        float2 nd_37 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_37, e1_37));
        float d0_37 = -nd_37.x;
        float d1_37 = -nd_37.y;
        float r0_37 = approx_rcp(d0_37);
        float r1_37 = approx_rcp(d1_37);
        float2 r_37 = make_float2(r0_37, r1_37);
        float2 s_37 = fma_f32x2_rn_noftz(r_37, fma_f32x2_rn_noftz(nd_37, r_37, make_float2(1.0f, 1.0f)), r_37);
        float t0_37 = s_37.x;
        float t1_37 = s_37.y;
        band = band | ((fminf(t0_37, t1_37) == 0.0f) ? 4 : 0);
        float2 s_36 = make_float2(max_noftz(t0_37, 0.0f), max_noftz(t1_37, 0.0f));
        float2 act_36;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_36) : "l"(*(const unsigned long long*)&_cvt_f32_55), "l"(*(const unsigned long long*)&s_36));
        float2 _f32x2_mul_a_45 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_36, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_36), s_36), _cvt_f32_57);
        float2 _mul_36;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_36) : "l"(*(const unsigned long long*)&_f32x2_mul_a_45), "l"(*(const unsigned long long*)&_cvt_f32_56));
        float2 _mul_37;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_37) : "l"(*(const unsigned long long*)&act_36), "l"(*(const unsigned long long*)&_cvt_f32_57));
        __nv_bfloat162 _bf16x2_36 = __float22bfloat162_rn(make_float2(_mul_36.x, _mul_36.y));
        __nv_bfloat162 _bf16x2_37 = __float22bfloat162_rn(make_float2(_mul_37.x, _mul_37.y));
        dgate[2] = __as_u32(_bf16x2_36);
        dup[2] = __as_u32(_bf16x2_37);
        float2 _cvt_f32_58 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 _cvt_f32_59 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 _cvt_f32_60 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float2 _f32x2_mul_b_46 = make_float2(-1.442695f, -1.442695f);
        float2 t_39;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_39) : "l"(*(const unsigned long long*)&_cvt_f32_58), "l"(*(const unsigned long long*)&_f32x2_mul_b_46));
        float e0_39 = approx_exp2(t_39.x);
        float e1_39 = approx_exp2(t_39.y);
        float2 nd_39 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_39, e1_39));
        float d0_39 = -nd_39.x;
        float d1_39 = -nd_39.y;
        float r0_39 = approx_rcp(d0_39);
        float r1_39 = approx_rcp(d1_39);
        float2 r_39 = make_float2(r0_39, r1_39);
        float2 s_39 = fma_f32x2_rn_noftz(r_39, fma_f32x2_rn_noftz(nd_39, r_39, make_float2(1.0f, 1.0f)), r_39);
        float t0_39 = s_39.x;
        float t1_39 = s_39.y;
        band = band | ((fminf(t0_39, t1_39) == 0.0f) ? 8 : 0);
        float2 s_38 = make_float2(max_noftz(t0_39, 0.0f), max_noftz(t1_39, 0.0f));
        float2 act_38;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_38) : "l"(*(const unsigned long long*)&_cvt_f32_58), "l"(*(const unsigned long long*)&s_38));
        float2 _f32x2_mul_a_47 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_38, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_38), s_38), _cvt_f32_60);
        float2 _mul_38;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_38) : "l"(*(const unsigned long long*)&_f32x2_mul_a_47), "l"(*(const unsigned long long*)&_cvt_f32_59));
        float2 _mul_39;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_39) : "l"(*(const unsigned long long*)&act_38), "l"(*(const unsigned long long*)&_cvt_f32_60));
        __nv_bfloat162 _bf16x2_38 = __float22bfloat162_rn(make_float2(_mul_38.x, _mul_38.y));
        __nv_bfloat162 _bf16x2_39 = __float22bfloat162_rn(make_float2(_mul_39.x, _mul_39.y));
        dgate[3] = __as_u32(_bf16x2_38);
        dup[3] = __as_u32(_bf16x2_39);
        float2 _cvt_f32_61 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 _cvt_f32_62 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float2 _cvt_f32_63 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[4]));
        float2 _f32x2_mul_b_48 = make_float2(-1.442695f, -1.442695f);
        float2 t_41;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_41) : "l"(*(const unsigned long long*)&_cvt_f32_61), "l"(*(const unsigned long long*)&_f32x2_mul_b_48));
        float e0_41 = approx_exp2(t_41.x);
        float e1_41 = approx_exp2(t_41.y);
        float2 nd_41 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_41, e1_41));
        float d0_41 = -nd_41.x;
        float d1_41 = -nd_41.y;
        float r0_41 = approx_rcp(d0_41);
        float r1_41 = approx_rcp(d1_41);
        float2 r_41 = make_float2(r0_41, r1_41);
        float2 s_41 = fma_f32x2_rn_noftz(r_41, fma_f32x2_rn_noftz(nd_41, r_41, make_float2(1.0f, 1.0f)), r_41);
        float t0_41 = s_41.x;
        float t1_41 = s_41.y;
        band = band | ((fminf(t0_41, t1_41) == 0.0f) ? 16 : 0);
        float2 s_40 = make_float2(max_noftz(t0_41, 0.0f), max_noftz(t1_41, 0.0f));
        float2 act_40;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_40) : "l"(*(const unsigned long long*)&_cvt_f32_61), "l"(*(const unsigned long long*)&s_40));
        float2 _f32x2_mul_a_49 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_40, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_40), s_40), _cvt_f32_63);
        float2 _mul_40;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_40) : "l"(*(const unsigned long long*)&_f32x2_mul_a_49), "l"(*(const unsigned long long*)&_cvt_f32_62));
        float2 _mul_41;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_41) : "l"(*(const unsigned long long*)&act_40), "l"(*(const unsigned long long*)&_cvt_f32_63));
        __nv_bfloat162 _bf16x2_40 = __float22bfloat162_rn(make_float2(_mul_40.x, _mul_40.y));
        __nv_bfloat162 _bf16x2_41 = __float22bfloat162_rn(make_float2(_mul_41.x, _mul_41.y));
        dgate[4] = __as_u32(_bf16x2_40);
        dup[4] = __as_u32(_bf16x2_41);
        float2 _cvt_f32_64 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 _cvt_f32_65 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float2 _cvt_f32_66 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[5]));
        float2 _f32x2_mul_b_50 = make_float2(-1.442695f, -1.442695f);
        float2 t_43;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_43) : "l"(*(const unsigned long long*)&_cvt_f32_64), "l"(*(const unsigned long long*)&_f32x2_mul_b_50));
        float e0_43 = approx_exp2(t_43.x);
        float e1_43 = approx_exp2(t_43.y);
        float2 nd_43 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_43, e1_43));
        float d0_43 = -nd_43.x;
        float d1_43 = -nd_43.y;
        float r0_43 = approx_rcp(d0_43);
        float r1_43 = approx_rcp(d1_43);
        float2 r_43 = make_float2(r0_43, r1_43);
        float2 s_43 = fma_f32x2_rn_noftz(r_43, fma_f32x2_rn_noftz(nd_43, r_43, make_float2(1.0f, 1.0f)), r_43);
        float t0_43 = s_43.x;
        float t1_43 = s_43.y;
        band = band | ((fminf(t0_43, t1_43) == 0.0f) ? 32 : 0);
        float2 s_42 = make_float2(max_noftz(t0_43, 0.0f), max_noftz(t1_43, 0.0f));
        float2 act_42;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_42) : "l"(*(const unsigned long long*)&_cvt_f32_64), "l"(*(const unsigned long long*)&s_42));
        float2 _f32x2_mul_a_51 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_42, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_42), s_42), _cvt_f32_66);
        float2 _mul_42;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_42) : "l"(*(const unsigned long long*)&_f32x2_mul_a_51), "l"(*(const unsigned long long*)&_cvt_f32_65));
        float2 _mul_43;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_43) : "l"(*(const unsigned long long*)&act_42), "l"(*(const unsigned long long*)&_cvt_f32_66));
        __nv_bfloat162 _bf16x2_42 = __float22bfloat162_rn(make_float2(_mul_42.x, _mul_42.y));
        __nv_bfloat162 _bf16x2_43 = __float22bfloat162_rn(make_float2(_mul_43.x, _mul_43.y));
        dgate[5] = __as_u32(_bf16x2_42);
        dup[5] = __as_u32(_bf16x2_43);
        float2 _cvt_f32_67 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 _cvt_f32_68 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float2 _cvt_f32_69 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[6]));
        float2 _f32x2_mul_b_52 = make_float2(-1.442695f, -1.442695f);
        float2 t_45;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_45) : "l"(*(const unsigned long long*)&_cvt_f32_67), "l"(*(const unsigned long long*)&_f32x2_mul_b_52));
        float e0_45 = approx_exp2(t_45.x);
        float e1_45 = approx_exp2(t_45.y);
        float2 nd_45 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_45, e1_45));
        float d0_45 = -nd_45.x;
        float d1_45 = -nd_45.y;
        float r0_45 = approx_rcp(d0_45);
        float r1_45 = approx_rcp(d1_45);
        float2 r_45 = make_float2(r0_45, r1_45);
        float2 s_45 = fma_f32x2_rn_noftz(r_45, fma_f32x2_rn_noftz(nd_45, r_45, make_float2(1.0f, 1.0f)), r_45);
        float t0_45 = s_45.x;
        float t1_45 = s_45.y;
        band = band | ((fminf(t0_45, t1_45) == 0.0f) ? 64 : 0);
        float2 s_44 = make_float2(max_noftz(t0_45, 0.0f), max_noftz(t1_45, 0.0f));
        float2 act_44;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_44) : "l"(*(const unsigned long long*)&_cvt_f32_67), "l"(*(const unsigned long long*)&s_44));
        float2 _f32x2_mul_a_53 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_44, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_44), s_44), _cvt_f32_69);
        float2 _mul_44;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_44) : "l"(*(const unsigned long long*)&_f32x2_mul_a_53), "l"(*(const unsigned long long*)&_cvt_f32_68));
        float2 _mul_45;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_45) : "l"(*(const unsigned long long*)&act_44), "l"(*(const unsigned long long*)&_cvt_f32_69));
        __nv_bfloat162 _bf16x2_44 = __float22bfloat162_rn(make_float2(_mul_44.x, _mul_44.y));
        __nv_bfloat162 _bf16x2_45 = __float22bfloat162_rn(make_float2(_mul_45.x, _mul_45.y));
        dgate[6] = __as_u32(_bf16x2_44);
        dup[6] = __as_u32(_bf16x2_45);
        float2 _cvt_f32_70 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 _cvt_f32_71 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float2 _cvt_f32_72 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[7]));
        float2 _f32x2_mul_b_54 = make_float2(-1.442695f, -1.442695f);
        float2 t_47;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_47) : "l"(*(const unsigned long long*)&_cvt_f32_70), "l"(*(const unsigned long long*)&_f32x2_mul_b_54));
        float e0_47 = approx_exp2(t_47.x);
        float e1_47 = approx_exp2(t_47.y);
        float2 nd_47 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_47, e1_47));
        float d0_47 = -nd_47.x;
        float d1_47 = -nd_47.y;
        float r0_47 = approx_rcp(d0_47);
        float r1_47 = approx_rcp(d1_47);
        float2 r_47 = make_float2(r0_47, r1_47);
        float2 s_47 = fma_f32x2_rn_noftz(r_47, fma_f32x2_rn_noftz(nd_47, r_47, make_float2(1.0f, 1.0f)), r_47);
        float t0_47 = s_47.x;
        float t1_47 = s_47.y;
        band = band | ((fminf(t0_47, t1_47) == 0.0f) ? 128 : 0);
        float2 s_46 = make_float2(max_noftz(t0_47, 0.0f), max_noftz(t1_47, 0.0f));
        float2 act_46;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_46) : "l"(*(const unsigned long long*)&_cvt_f32_70), "l"(*(const unsigned long long*)&s_46));
        float2 _f32x2_mul_a_55 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_46, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_46), s_46), _cvt_f32_72);
        float2 _mul_46;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_46) : "l"(*(const unsigned long long*)&_f32x2_mul_a_55), "l"(*(const unsigned long long*)&_cvt_f32_71));
        float2 _mul_47;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_47) : "l"(*(const unsigned long long*)&act_46), "l"(*(const unsigned long long*)&_cvt_f32_72));
        __nv_bfloat162 _bf16x2_46 = __float22bfloat162_rn(make_float2(_mul_46.x, _mul_46.y));
        __nv_bfloat162 _bf16x2_47 = __float22bfloat162_rn(make_float2(_mul_47.x, _mul_47.y));
        dgate[7] = __as_u32(_bf16x2_46);
        dup[7] = __as_u32(_bf16x2_47);
        int band_hi = 0;
        float2 _cvt_f32_73 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
        float2 _cvt_f32_74 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[0]));
        float2 _cvt_f32_75 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[0]));
        float2 _f32x2_mul_b_56 = make_float2(-1.442695f, -1.442695f);
        float2 t_49;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_49) : "l"(*(const unsigned long long*)&_cvt_f32_73), "l"(*(const unsigned long long*)&_f32x2_mul_b_56));
        float e0_49 = approx_exp2(t_49.x);
        float e1_49 = approx_exp2(t_49.y);
        float2 nd_49 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_49, e1_49));
        float d0_49 = -nd_49.x;
        float d1_49 = -nd_49.y;
        float r0_49 = approx_rcp(d0_49);
        float r1_49 = approx_rcp(d1_49);
        float2 r_49 = make_float2(r0_49, r1_49);
        float2 s_49 = fma_f32x2_rn_noftz(r_49, fma_f32x2_rn_noftz(nd_49, r_49, make_float2(1.0f, 1.0f)), r_49);
        float t0_49 = s_49.x;
        float t1_49 = s_49.y;
        band_hi = band_hi | ((fminf(t0_49, t1_49) == 0.0f) ? 1 : 0);
        float2 s_48 = make_float2(max_noftz(t0_49, 0.0f), max_noftz(t1_49, 0.0f));
        float2 act_48;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_48) : "l"(*(const unsigned long long*)&_cvt_f32_73), "l"(*(const unsigned long long*)&s_48));
        float2 _f32x2_mul_a_57 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_48, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_48), s_48), _cvt_f32_75);
        float2 _mul_48;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_48) : "l"(*(const unsigned long long*)&_f32x2_mul_a_57), "l"(*(const unsigned long long*)&_cvt_f32_74));
        float2 _mul_49;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_49) : "l"(*(const unsigned long long*)&act_48), "l"(*(const unsigned long long*)&_cvt_f32_75));
        __nv_bfloat162 _bf16x2_48 = __float22bfloat162_rn(make_float2(_mul_48.x, _mul_48.y));
        __nv_bfloat162 _bf16x2_49 = __float22bfloat162_rn(make_float2(_mul_49.x, _mul_49.y));
        dgate_hi[0] = __as_u32(_bf16x2_48);
        dup_hi[0] = __as_u32(_bf16x2_49);
        float2 _cvt_f32_76 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[1]));
        float2 _cvt_f32_77 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[1]));
        float2 _cvt_f32_78 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[1]));
        float2 _f32x2_mul_b_58 = make_float2(-1.442695f, -1.442695f);
        float2 t_51;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_51) : "l"(*(const unsigned long long*)&_cvt_f32_76), "l"(*(const unsigned long long*)&_f32x2_mul_b_58));
        float e0_51 = approx_exp2(t_51.x);
        float e1_51 = approx_exp2(t_51.y);
        float2 nd_51 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_51, e1_51));
        float d0_51 = -nd_51.x;
        float d1_51 = -nd_51.y;
        float r0_51 = approx_rcp(d0_51);
        float r1_51 = approx_rcp(d1_51);
        float2 r_51 = make_float2(r0_51, r1_51);
        float2 s_51 = fma_f32x2_rn_noftz(r_51, fma_f32x2_rn_noftz(nd_51, r_51, make_float2(1.0f, 1.0f)), r_51);
        float t0_51 = s_51.x;
        float t1_51 = s_51.y;
        band_hi = band_hi | ((fminf(t0_51, t1_51) == 0.0f) ? 2 : 0);
        float2 s_50 = make_float2(max_noftz(t0_51, 0.0f), max_noftz(t1_51, 0.0f));
        float2 act_50;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_50) : "l"(*(const unsigned long long*)&_cvt_f32_76), "l"(*(const unsigned long long*)&s_50));
        float2 _f32x2_mul_a_59 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_50, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_50), s_50), _cvt_f32_78);
        float2 _mul_50;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_50) : "l"(*(const unsigned long long*)&_f32x2_mul_a_59), "l"(*(const unsigned long long*)&_cvt_f32_77));
        float2 _mul_51;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_51) : "l"(*(const unsigned long long*)&act_50), "l"(*(const unsigned long long*)&_cvt_f32_78));
        __nv_bfloat162 _bf16x2_50 = __float22bfloat162_rn(make_float2(_mul_50.x, _mul_50.y));
        __nv_bfloat162 _bf16x2_51 = __float22bfloat162_rn(make_float2(_mul_51.x, _mul_51.y));
        dgate_hi[1] = __as_u32(_bf16x2_50);
        dup_hi[1] = __as_u32(_bf16x2_51);
        float2 _cvt_f32_79 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[2]));
        float2 _cvt_f32_80 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[2]));
        float2 _cvt_f32_81 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[2]));
        float2 _f32x2_mul_b_60 = make_float2(-1.442695f, -1.442695f);
        float2 t_53;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_53) : "l"(*(const unsigned long long*)&_cvt_f32_79), "l"(*(const unsigned long long*)&_f32x2_mul_b_60));
        float e0_53 = approx_exp2(t_53.x);
        float e1_53 = approx_exp2(t_53.y);
        float2 nd_53 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_53, e1_53));
        float d0_53 = -nd_53.x;
        float d1_53 = -nd_53.y;
        float r0_53 = approx_rcp(d0_53);
        float r1_53 = approx_rcp(d1_53);
        float2 r_53 = make_float2(r0_53, r1_53);
        float2 s_53 = fma_f32x2_rn_noftz(r_53, fma_f32x2_rn_noftz(nd_53, r_53, make_float2(1.0f, 1.0f)), r_53);
        float t0_53 = s_53.x;
        float t1_53 = s_53.y;
        band_hi = band_hi | ((fminf(t0_53, t1_53) == 0.0f) ? 4 : 0);
        float2 s_52 = make_float2(max_noftz(t0_53, 0.0f), max_noftz(t1_53, 0.0f));
        float2 act_52;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_52) : "l"(*(const unsigned long long*)&_cvt_f32_79), "l"(*(const unsigned long long*)&s_52));
        float2 _f32x2_mul_a_61 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_52, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_52), s_52), _cvt_f32_81);
        float2 _mul_52;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_52) : "l"(*(const unsigned long long*)&_f32x2_mul_a_61), "l"(*(const unsigned long long*)&_cvt_f32_80));
        float2 _mul_53;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_53) : "l"(*(const unsigned long long*)&act_52), "l"(*(const unsigned long long*)&_cvt_f32_81));
        __nv_bfloat162 _bf16x2_52 = __float22bfloat162_rn(make_float2(_mul_52.x, _mul_52.y));
        __nv_bfloat162 _bf16x2_53 = __float22bfloat162_rn(make_float2(_mul_53.x, _mul_53.y));
        dgate_hi[2] = __as_u32(_bf16x2_52);
        dup_hi[2] = __as_u32(_bf16x2_53);
        float2 _cvt_f32_82 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[3]));
        float2 _cvt_f32_83 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[3]));
        float2 _cvt_f32_84 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[3]));
        float2 _f32x2_mul_b_62 = make_float2(-1.442695f, -1.442695f);
        float2 t_55;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_55) : "l"(*(const unsigned long long*)&_cvt_f32_82), "l"(*(const unsigned long long*)&_f32x2_mul_b_62));
        float e0_55 = approx_exp2(t_55.x);
        float e1_55 = approx_exp2(t_55.y);
        float2 nd_55 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_55, e1_55));
        float d0_55 = -nd_55.x;
        float d1_55 = -nd_55.y;
        float r0_55 = approx_rcp(d0_55);
        float r1_55 = approx_rcp(d1_55);
        float2 r_55 = make_float2(r0_55, r1_55);
        float2 s_55 = fma_f32x2_rn_noftz(r_55, fma_f32x2_rn_noftz(nd_55, r_55, make_float2(1.0f, 1.0f)), r_55);
        float t0_55 = s_55.x;
        float t1_55 = s_55.y;
        band_hi = band_hi | ((fminf(t0_55, t1_55) == 0.0f) ? 8 : 0);
        float2 s_54 = make_float2(max_noftz(t0_55, 0.0f), max_noftz(t1_55, 0.0f));
        float2 act_54;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_54) : "l"(*(const unsigned long long*)&_cvt_f32_82), "l"(*(const unsigned long long*)&s_54));
        float2 _f32x2_mul_a_63 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_54, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_54), s_54), _cvt_f32_84);
        float2 _mul_54;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_54) : "l"(*(const unsigned long long*)&_f32x2_mul_a_63), "l"(*(const unsigned long long*)&_cvt_f32_83));
        float2 _mul_55;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_55) : "l"(*(const unsigned long long*)&act_54), "l"(*(const unsigned long long*)&_cvt_f32_84));
        __nv_bfloat162 _bf16x2_54 = __float22bfloat162_rn(make_float2(_mul_54.x, _mul_54.y));
        __nv_bfloat162 _bf16x2_55 = __float22bfloat162_rn(make_float2(_mul_55.x, _mul_55.y));
        dgate_hi[3] = __as_u32(_bf16x2_54);
        dup_hi[3] = __as_u32(_bf16x2_55);
        float2 _cvt_f32_85 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[4]));
        float2 _cvt_f32_86 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[4]));
        float2 _cvt_f32_87 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[4]));
        float2 _f32x2_mul_b_64 = make_float2(-1.442695f, -1.442695f);
        float2 t_57;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_57) : "l"(*(const unsigned long long*)&_cvt_f32_85), "l"(*(const unsigned long long*)&_f32x2_mul_b_64));
        float e0_57 = approx_exp2(t_57.x);
        float e1_57 = approx_exp2(t_57.y);
        float2 nd_57 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_57, e1_57));
        float d0_57 = -nd_57.x;
        float d1_57 = -nd_57.y;
        float r0_57 = approx_rcp(d0_57);
        float r1_57 = approx_rcp(d1_57);
        float2 r_57 = make_float2(r0_57, r1_57);
        float2 s_57 = fma_f32x2_rn_noftz(r_57, fma_f32x2_rn_noftz(nd_57, r_57, make_float2(1.0f, 1.0f)), r_57);
        float t0_57 = s_57.x;
        float t1_57 = s_57.y;
        band_hi = band_hi | ((fminf(t0_57, t1_57) == 0.0f) ? 16 : 0);
        float2 s_56 = make_float2(max_noftz(t0_57, 0.0f), max_noftz(t1_57, 0.0f));
        float2 act_56;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_56) : "l"(*(const unsigned long long*)&_cvt_f32_85), "l"(*(const unsigned long long*)&s_56));
        float2 _f32x2_mul_a_65 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_56, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_56), s_56), _cvt_f32_87);
        float2 _mul_56;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_56) : "l"(*(const unsigned long long*)&_f32x2_mul_a_65), "l"(*(const unsigned long long*)&_cvt_f32_86));
        float2 _mul_57;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_57) : "l"(*(const unsigned long long*)&act_56), "l"(*(const unsigned long long*)&_cvt_f32_87));
        __nv_bfloat162 _bf16x2_56 = __float22bfloat162_rn(make_float2(_mul_56.x, _mul_56.y));
        __nv_bfloat162 _bf16x2_57 = __float22bfloat162_rn(make_float2(_mul_57.x, _mul_57.y));
        dgate_hi[4] = __as_u32(_bf16x2_56);
        dup_hi[4] = __as_u32(_bf16x2_57);
        float2 _cvt_f32_88 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[5]));
        float2 _cvt_f32_89 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[5]));
        float2 _cvt_f32_90 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[5]));
        float2 _f32x2_mul_b_66 = make_float2(-1.442695f, -1.442695f);
        float2 t_59;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_59) : "l"(*(const unsigned long long*)&_cvt_f32_88), "l"(*(const unsigned long long*)&_f32x2_mul_b_66));
        float e0_59 = approx_exp2(t_59.x);
        float e1_59 = approx_exp2(t_59.y);
        float2 nd_59 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_59, e1_59));
        float d0_59 = -nd_59.x;
        float d1_59 = -nd_59.y;
        float r0_59 = approx_rcp(d0_59);
        float r1_59 = approx_rcp(d1_59);
        float2 r_59 = make_float2(r0_59, r1_59);
        float2 s_59 = fma_f32x2_rn_noftz(r_59, fma_f32x2_rn_noftz(nd_59, r_59, make_float2(1.0f, 1.0f)), r_59);
        float t0_59 = s_59.x;
        float t1_59 = s_59.y;
        band_hi = band_hi | ((fminf(t0_59, t1_59) == 0.0f) ? 32 : 0);
        float2 s_58 = make_float2(max_noftz(t0_59, 0.0f), max_noftz(t1_59, 0.0f));
        float2 act_58;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_58) : "l"(*(const unsigned long long*)&_cvt_f32_88), "l"(*(const unsigned long long*)&s_58));
        float2 _f32x2_mul_a_67 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_58, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_58), s_58), _cvt_f32_90);
        float2 _mul_58;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_58) : "l"(*(const unsigned long long*)&_f32x2_mul_a_67), "l"(*(const unsigned long long*)&_cvt_f32_89));
        float2 _mul_59;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_59) : "l"(*(const unsigned long long*)&act_58), "l"(*(const unsigned long long*)&_cvt_f32_90));
        __nv_bfloat162 _bf16x2_58 = __float22bfloat162_rn(make_float2(_mul_58.x, _mul_58.y));
        __nv_bfloat162 _bf16x2_59 = __float22bfloat162_rn(make_float2(_mul_59.x, _mul_59.y));
        dgate_hi[5] = __as_u32(_bf16x2_58);
        dup_hi[5] = __as_u32(_bf16x2_59);
        float2 _cvt_f32_91 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[6]));
        float2 _cvt_f32_92 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[6]));
        float2 _cvt_f32_93 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[6]));
        float2 _f32x2_mul_b_68 = make_float2(-1.442695f, -1.442695f);
        float2 t_61;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_61) : "l"(*(const unsigned long long*)&_cvt_f32_91), "l"(*(const unsigned long long*)&_f32x2_mul_b_68));
        float e0_61 = approx_exp2(t_61.x);
        float e1_61 = approx_exp2(t_61.y);
        float2 nd_61 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_61, e1_61));
        float d0_61 = -nd_61.x;
        float d1_61 = -nd_61.y;
        float r0_61 = approx_rcp(d0_61);
        float r1_61 = approx_rcp(d1_61);
        float2 r_61 = make_float2(r0_61, r1_61);
        float2 s_61 = fma_f32x2_rn_noftz(r_61, fma_f32x2_rn_noftz(nd_61, r_61, make_float2(1.0f, 1.0f)), r_61);
        float t0_61 = s_61.x;
        float t1_61 = s_61.y;
        band_hi = band_hi | ((fminf(t0_61, t1_61) == 0.0f) ? 64 : 0);
        float2 s_60 = make_float2(max_noftz(t0_61, 0.0f), max_noftz(t1_61, 0.0f));
        float2 act_60;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_60) : "l"(*(const unsigned long long*)&_cvt_f32_91), "l"(*(const unsigned long long*)&s_60));
        float2 _f32x2_mul_a_69 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_60, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_60), s_60), _cvt_f32_93);
        float2 _mul_60;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_60) : "l"(*(const unsigned long long*)&_f32x2_mul_a_69), "l"(*(const unsigned long long*)&_cvt_f32_92));
        float2 _mul_61;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_61) : "l"(*(const unsigned long long*)&act_60), "l"(*(const unsigned long long*)&_cvt_f32_93));
        __nv_bfloat162 _bf16x2_60 = __float22bfloat162_rn(make_float2(_mul_60.x, _mul_60.y));
        __nv_bfloat162 _bf16x2_61 = __float22bfloat162_rn(make_float2(_mul_61.x, _mul_61.y));
        dgate_hi[6] = __as_u32(_bf16x2_60);
        dup_hi[6] = __as_u32(_bf16x2_61);
        float2 _cvt_f32_94 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[7]));
        float2 _cvt_f32_95 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[7]));
        float2 _cvt_f32_96 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[7]));
        float2 _f32x2_mul_b_70 = make_float2(-1.442695f, -1.442695f);
        float2 t_63;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_63) : "l"(*(const unsigned long long*)&_cvt_f32_94), "l"(*(const unsigned long long*)&_f32x2_mul_b_70));
        float e0_63 = approx_exp2(t_63.x);
        float e1_63 = approx_exp2(t_63.y);
        float2 nd_63 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_63, e1_63));
        float d0_63 = -nd_63.x;
        float d1_63 = -nd_63.y;
        float r0_63 = approx_rcp(d0_63);
        float r1_63 = approx_rcp(d1_63);
        float2 r_63 = make_float2(r0_63, r1_63);
        float2 s_63 = fma_f32x2_rn_noftz(r_63, fma_f32x2_rn_noftz(nd_63, r_63, make_float2(1.0f, 1.0f)), r_63);
        float t0_63 = s_63.x;
        float t1_63 = s_63.y;
        band_hi = band_hi | ((fminf(t0_63, t1_63) == 0.0f) ? 128 : 0);
        float2 s_62 = make_float2(max_noftz(t0_63, 0.0f), max_noftz(t1_63, 0.0f));
        float2 act_62;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_62) : "l"(*(const unsigned long long*)&_cvt_f32_94), "l"(*(const unsigned long long*)&s_62));
        float2 _f32x2_mul_a_71 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_62, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_62), s_62), _cvt_f32_96);
        float2 _mul_62;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_62) : "l"(*(const unsigned long long*)&_f32x2_mul_a_71), "l"(*(const unsigned long long*)&_cvt_f32_95));
        float2 _mul_63;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_63) : "l"(*(const unsigned long long*)&act_62), "l"(*(const unsigned long long*)&_cvt_f32_96));
        __nv_bfloat162 _bf16x2_62 = __float22bfloat162_rn(make_float2(_mul_62.x, _mul_62.y));
        __nv_bfloat162 _bf16x2_63 = __float22bfloat162_rn(make_float2(_mul_63.x, _mul_63.y));
        dgate_hi[7] = __as_u32(_bf16x2_62);
        dup_hi[7] = __as_u32(_bf16x2_63);
        if (band != 0) {
            unsigned int _popc_0 = (unsigned int)__popc(band);
            #pragma unroll 1
            for (int _j = 0; _j < _popc_0; _j++) {
                int _find_first_set_72 = __ffs(band);
                int j = _find_first_set_72 - 1;
                band = band & band - 1;
                unsigned int _vec_load_6[1];
                {
                    _vec_load_6[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index + 2 * j) + 0);
                }
                unsigned int _vec_load_7[1];
                {
                    _vec_load_7[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index + K + 2 * j) + 0);
                }
                unsigned int _vec_load_8[1];
                {
                    _vec_load_8[0] = *reinterpret_cast<const unsigned int*>(grad + (grow * K + col + 2 * j) + 0);
                }
                float2 _cvt_f32_97 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[0]));
                float2 _cvt_f32_98 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[0]));
                float2 _cvt_f32_99 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_8[0]));
                float2 _f32x2_mul_b_73 = make_float2(-1.442695f, -1.442695f);
                float2 t_65;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_65) : "l"(*(const unsigned long long*)&_cvt_f32_97), "l"(*(const unsigned long long*)&_f32x2_mul_b_73));
                float e0_65 = approx_exp2(t_65.x);
                float e1_65 = approx_exp2(t_65.y);
                float2 nd_65 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_65, e1_65));
                float d0_65 = -nd_65.x;
                float d1_65 = -nd_65.y;
                float r0_65 = approx_rcp(d0_65);
                float r1_65 = approx_rcp(d1_65);
                float2 r_65 = make_float2(r0_65, r1_65);
                float2 s_65 = fma_f32x2_rn_noftz(r_65, fma_f32x2_rn_noftz(nd_65, r_65, make_float2(1.0f, 1.0f)), r_65);
                float t0_65 = s_65.x;
                float t1_65 = s_65.y;
                float d4_65a;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_65a) : "f"(d0_65), "f"(0.25f));
                float r4_65a = approx_rcp(d4_65a);
                float _fma_75;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(-d4_65a), "f"(r4_65a), "f"(1.0f));
                float _fma_74;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(r4_65a), "f"(_fma_75), "f"(r4_65a));
                float big_65a;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_65a) : "f"(mul_f32_rn_noftz(_fma_74, 1.2676506e+30f)), "f"(1.4073749e+14f));
                float _add_76;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(big_65a), "f"(8.388608e+06f));
                float whole_65a;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_65a) : "f"(_add_76), "f"(8.388608e+06f));
                float above_65a;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_65a) : "f"(-d4_65a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_65a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                float _sub_78;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_78) : "f"(big_65a), "f"(whole_65a));
                float _abs_77;
                asm("abs.f32 %0, %1;" : "=f"(_abs_77) : "f"(_sub_78));
                float _add_79;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(big_65a), "f"(0.5f));
                float _sub_80;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_80) : "f"(big_65a), "f"(0.5f));
                float rounded_65a = ((_abs_77 == 0.5f) ? ((above_65a > 0.0f) ? _add_79 : _sub_80) : whole_65a);
                float u0_65 = ((d0_65 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_65a, 7.888609e-31f), 1.7763568e-15f) : t0_65);
                float d4_65b;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_65b) : "f"(d1_65), "f"(0.25f));
                float r4_65b = approx_rcp(d4_65b);
                float _fma_82;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(-d4_65b), "f"(r4_65b), "f"(1.0f));
                float _fma_81;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(r4_65b), "f"(_fma_82), "f"(r4_65b));
                float big_65b;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_65b) : "f"(mul_f32_rn_noftz(_fma_81, 1.2676506e+30f)), "f"(1.4073749e+14f));
                float _add_83;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(big_65b), "f"(8.388608e+06f));
                float whole_65b;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_65b) : "f"(_add_83), "f"(8.388608e+06f));
                float above_65b;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_65b) : "f"(-d4_65b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_65b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                float _sub_85;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_85) : "f"(big_65b), "f"(whole_65b));
                float _abs_84;
                asm("abs.f32 %0, %1;" : "=f"(_abs_84) : "f"(_sub_85));
                float _add_86;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_86) : "f"(big_65b), "f"(0.5f));
                float _sub_87;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_87) : "f"(big_65b), "f"(0.5f));
                float rounded_65b = ((_abs_84 == 0.5f) ? ((above_65b > 0.0f) ? _add_86 : _sub_87) : whole_65b);
                float u1_65 = ((d1_65 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_65b, 7.888609e-31f), 1.7763568e-15f) : t1_65);
                float2 s_64 = make_float2(max_noftz(u0_65, 0.0f), max_noftz(u1_65, 0.0f));
                float2 act_64;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_64) : "l"(*(const unsigned long long*)&_cvt_f32_97), "l"(*(const unsigned long long*)&s_64));
                float2 _f32x2_mul_a_88 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_64, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_64), s_64), _cvt_f32_99);
                float2 _mul_64;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_64) : "l"(*(const unsigned long long*)&_f32x2_mul_a_88), "l"(*(const unsigned long long*)&_cvt_f32_98));
                float2 _mul_65;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_65) : "l"(*(const unsigned long long*)&act_64), "l"(*(const unsigned long long*)&_cvt_f32_99));
                __nv_bfloat162 _bf16x2_64 = __float22bfloat162_rn(make_float2(_mul_64.x, _mul_64.y));
                __nv_bfloat162 _bf16x2_65 = __float22bfloat162_rn(make_float2(_mul_65.x, _mul_65.y));
                unsigned int dgate_new = __as_u32(_bf16x2_64);
                unsigned int dup_new = __as_u32(_bf16x2_65);
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
        if (band_hi != 0) {
            unsigned int _popc_1 = (unsigned int)__popc(band_hi);
            #pragma unroll 1
            for (int _j_hi = 0; _j_hi < _popc_1; _j_hi++) {
                int _find_first_set_89 = __ffs(band_hi);
                int j_hi = _find_first_set_89 - 1;
                band_hi = band_hi & band_hi - 1;
                unsigned int _vec_load_9[1];
                {
                    _vec_load_9[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_hi + 2 * j_hi) + 0);
                }
                unsigned int _vec_load_10[1];
                {
                    _vec_load_10[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_hi + K + 2 * j_hi) + 0);
                }
                unsigned int _vec_load_11[1];
                {
                    _vec_load_11[0] = *reinterpret_cast<const unsigned int*>(grad + (grow_hi * K + col + 2 * j_hi) + 0);
                }
                float2 _cvt_f32_100 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_9[0]));
                float2 _cvt_f32_101 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_10[0]));
                float2 _cvt_f32_102 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_11[0]));
                float2 _f32x2_mul_b_90 = make_float2(-1.442695f, -1.442695f);
                float2 t_67;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_67) : "l"(*(const unsigned long long*)&_cvt_f32_100), "l"(*(const unsigned long long*)&_f32x2_mul_b_90));
                float e0_67 = approx_exp2(t_67.x);
                float e1_67 = approx_exp2(t_67.y);
                float2 nd_67 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_67, e1_67));
                float d0_67 = -nd_67.x;
                float d1_67 = -nd_67.y;
                float r0_67 = approx_rcp(d0_67);
                float r1_67 = approx_rcp(d1_67);
                float2 r_67 = make_float2(r0_67, r1_67);
                float2 s_67 = fma_f32x2_rn_noftz(r_67, fma_f32x2_rn_noftz(nd_67, r_67, make_float2(1.0f, 1.0f)), r_67);
                float t0_67 = s_67.x;
                float t1_67 = s_67.y;
                float d4_67a;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_67a) : "f"(d0_67), "f"(0.25f));
                float r4_67a = approx_rcp(d4_67a);
                float _fma_92;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(-d4_67a), "f"(r4_67a), "f"(1.0f));
                float _fma_91;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(r4_67a), "f"(_fma_92), "f"(r4_67a));
                float big_67a;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_67a) : "f"(mul_f32_rn_noftz(_fma_91, 1.2676506e+30f)), "f"(1.4073749e+14f));
                float _add_93;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(big_67a), "f"(8.388608e+06f));
                float whole_67a;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_67a) : "f"(_add_93), "f"(8.388608e+06f));
                float above_67a;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_67a) : "f"(-d4_67a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_67a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                float _sub_95;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_95) : "f"(big_67a), "f"(whole_67a));
                float _abs_94;
                asm("abs.f32 %0, %1;" : "=f"(_abs_94) : "f"(_sub_95));
                float _add_96;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_96) : "f"(big_67a), "f"(0.5f));
                float _sub_97;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_97) : "f"(big_67a), "f"(0.5f));
                float rounded_67a = ((_abs_94 == 0.5f) ? ((above_67a > 0.0f) ? _add_96 : _sub_97) : whole_67a);
                float u0_67 = ((d0_67 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_67a, 7.888609e-31f), 1.7763568e-15f) : t0_67);
                float d4_67b;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_67b) : "f"(d1_67), "f"(0.25f));
                float r4_67b = approx_rcp(d4_67b);
                float _fma_99;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(-d4_67b), "f"(r4_67b), "f"(1.0f));
                float _fma_98;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(r4_67b), "f"(_fma_99), "f"(r4_67b));
                float big_67b;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_67b) : "f"(mul_f32_rn_noftz(_fma_98, 1.2676506e+30f)), "f"(1.4073749e+14f));
                float _add_100;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_100) : "f"(big_67b), "f"(8.388608e+06f));
                float whole_67b;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_67b) : "f"(_add_100), "f"(8.388608e+06f));
                float above_67b;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_67b) : "f"(-d4_67b), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_67b, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                float _sub_102;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_102) : "f"(big_67b), "f"(whole_67b));
                float _abs_101;
                asm("abs.f32 %0, %1;" : "=f"(_abs_101) : "f"(_sub_102));
                float _add_103;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_103) : "f"(big_67b), "f"(0.5f));
                float _sub_104;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_104) : "f"(big_67b), "f"(0.5f));
                float rounded_67b = ((_abs_101 == 0.5f) ? ((above_67b > 0.0f) ? _add_103 : _sub_104) : whole_67b);
                float u1_67 = ((d1_67 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_67b, 7.888609e-31f), 1.7763568e-15f) : t1_67);
                float2 s_66 = make_float2(max_noftz(u0_67, 0.0f), max_noftz(u1_67, 0.0f));
                float2 act_66;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_66) : "l"(*(const unsigned long long*)&_cvt_f32_100), "l"(*(const unsigned long long*)&s_66));
                float2 _f32x2_mul_a_105 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_66, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_66), s_66), _cvt_f32_102);
                float2 _mul_66;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_66) : "l"(*(const unsigned long long*)&_f32x2_mul_a_105), "l"(*(const unsigned long long*)&_cvt_f32_101));
                float2 _mul_67;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_67) : "l"(*(const unsigned long long*)&act_66), "l"(*(const unsigned long long*)&_cvt_f32_102));
                __nv_bfloat162 _bf16x2_66 = __float22bfloat162_rn(make_float2(_mul_66.x, _mul_66.y));
                __nv_bfloat162 _bf16x2_67 = __float22bfloat162_rn(make_float2(_mul_67.x, _mul_67.y));
                unsigned int dgate_new_hi = __as_u32(_bf16x2_66);
                unsigned int dup_new_hi = __as_u32(_bf16x2_67);
                dgate_hi[0] = ((j_hi == 0) ? dgate_new_hi : dgate_hi[0]);
                dup_hi[0] = ((j_hi == 0) ? dup_new_hi : dup_hi[0]);
                dgate_hi[1] = ((j_hi == 1) ? dgate_new_hi : dgate_hi[1]);
                dup_hi[1] = ((j_hi == 1) ? dup_new_hi : dup_hi[1]);
                dgate_hi[2] = ((j_hi == 2) ? dgate_new_hi : dgate_hi[2]);
                dup_hi[2] = ((j_hi == 2) ? dup_new_hi : dup_hi[2]);
                dgate_hi[3] = ((j_hi == 3) ? dgate_new_hi : dgate_hi[3]);
                dup_hi[3] = ((j_hi == 3) ? dup_new_hi : dup_hi[3]);
                dgate_hi[4] = ((j_hi == 4) ? dgate_new_hi : dgate_hi[4]);
                dup_hi[4] = ((j_hi == 4) ? dup_new_hi : dup_hi[4]);
                dgate_hi[5] = ((j_hi == 5) ? dgate_new_hi : dgate_hi[5]);
                dup_hi[5] = ((j_hi == 5) ? dup_new_hi : dup_hi[5]);
                dgate_hi[6] = ((j_hi == 6) ? dgate_new_hi : dgate_hi[6]);
                dup_hi[6] = ((j_hi == 6) ? dup_new_hi : dup_hi[6]);
                dgate_hi[7] = ((j_hi == 7) ? dgate_new_hi : dgate_hi[7]);
                dup_hi[7] = ((j_hi == 7) ? dup_new_hi : dup_hi[7]);
            }
        }
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
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base_hi + ((chunk ^ swizzle_hi) << 4)), "r"(dgate_hi[0]), "r"(dgate_hi[1]), "r"(dgate_hi[2]), "r"(dgate_hi[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base_hi + ((chunk + 1 ^ swizzle_hi) << 4)), "r"(dgate_hi[4]), "r"(dgate_hi[5]), "r"(dgate_hi[6]), "r"(dgate_hi[7]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base_hi + (8192 + ((chunk ^ swizzle_hi) << 4))), "r"(dup_hi[0]), "r"(dup_hi[1]), "r"(dup_hi[2]), "r"(dup_hi[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base_hi + (8192 + ((chunk + 1 ^ swizzle_hi) << 4))), "r"(dup_hi[4]), "r"(dup_hi[5]), "r"(dup_hi[6]), "r"(dup_hi[7]) : "memory");
    int scale_cols = K >> 6;
    int scale_col = bx * 2 + blk;
    int out_index = grow * (2 * K) + col;
    __nv_bfloat162 amax = __as_bf16x2(0);
    __nv_bfloat162 _hxa_107 = amax;
    __nv_bfloat162 _hxb_107 = __as_bf16x2(dgate[0]);
    unsigned int _hxr_107;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_107) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_107)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_107)));
    __nv_bfloat162 _max_106 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_107);
    amax = _max_106;
    __nv_bfloat162 _hxa_109 = amax;
    __nv_bfloat162 _hxb_109 = __as_bf16x2(dgate[1]);
    unsigned int _hxr_109;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_109) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_109)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_109)));
    __nv_bfloat162 _max_108 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_109);
    amax = _max_108;
    __nv_bfloat162 _hxa_111 = amax;
    __nv_bfloat162 _hxb_111 = __as_bf16x2(dgate[2]);
    unsigned int _hxr_111;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_111) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_111)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_111)));
    __nv_bfloat162 _max_110 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_111);
    amax = _max_110;
    __nv_bfloat162 _hxa_113 = amax;
    __nv_bfloat162 _hxb_113 = __as_bf16x2(dgate[3]);
    unsigned int _hxr_113;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_113) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_113)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_113)));
    __nv_bfloat162 _max_112 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_113);
    amax = _max_112;
    __nv_bfloat162 _hxa_115 = amax;
    __nv_bfloat162 _hxb_115 = __as_bf16x2(dgate[4]);
    unsigned int _hxr_115;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_115) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_115)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_115)));
    __nv_bfloat162 _max_114 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_115);
    amax = _max_114;
    __nv_bfloat162 _hxa_117 = amax;
    __nv_bfloat162 _hxb_117 = __as_bf16x2(dgate[5]);
    unsigned int _hxr_117;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_117) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_117)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_117)));
    __nv_bfloat162 _max_116 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_117);
    amax = _max_116;
    __nv_bfloat162 _hxa_119 = amax;
    __nv_bfloat162 _hxb_119 = __as_bf16x2(dgate[6]);
    unsigned int _hxr_119;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_119) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_119)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_119)));
    __nv_bfloat162 _max_118 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_119);
    amax = _max_118;
    __nv_bfloat162 _hxa_121 = amax;
    __nv_bfloat162 _hxb_121 = __as_bf16x2(dgate[7]);
    unsigned int _hxr_121;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_121) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_121)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_121)));
    __nv_bfloat162 _max_120 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_121);
    amax = _max_120;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_123;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_123) : "r"(_bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_123) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_122;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_122) : "h"(_e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_123));
    words[0] = _mov_b32_pack_u16x2_122;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_125;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_125) : "r"(_bf16x2_mul_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_125) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_124;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_124) : "h"(_e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_125));
    words[1] = _mov_b32_pack_u16x2_124;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_127;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_127) : "r"(_bf16x2_mul_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_127) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_126;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_126) : "h"(_e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_127));
    words[2] = _mov_b32_pack_u16x2_126;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_129;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_129) : "r"(_bf16x2_mul_7));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_7);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_129) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_128;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_128) : "h"(_e4m3x2_bf16x2_3), "h"(_cvt_satfinite_e4m3x2_bf16x2_129));
    words[3] = _mov_b32_pack_u16x2_128;
    reinterpret_cast<int4*>(row_output + out_index)[0] = reinterpret_cast<int4*>(words + 0)[0];
    __nv_bfloat162 amax_0 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_131 = amax_0;
    __nv_bfloat162 _hxb_131 = __as_bf16x2(dup[0]);
    unsigned int _hxr_131;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_131) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_131)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_131)));
    __nv_bfloat162 _max_130 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_131);
    amax_0 = _max_130;
    __nv_bfloat162 _hxa_133 = amax_0;
    __nv_bfloat162 _hxb_133 = __as_bf16x2(dup[1]);
    unsigned int _hxr_133;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_133) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_133)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_133)));
    __nv_bfloat162 _max_132 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_133);
    amax_0 = _max_132;
    __nv_bfloat162 _hxa_135 = amax_0;
    __nv_bfloat162 _hxb_135 = __as_bf16x2(dup[2]);
    unsigned int _hxr_135;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_135) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_135)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_135)));
    __nv_bfloat162 _max_134 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_135);
    amax_0 = _max_134;
    __nv_bfloat162 _hxa_137 = amax_0;
    __nv_bfloat162 _hxb_137 = __as_bf16x2(dup[3]);
    unsigned int _hxr_137;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_137) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_137)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_137)));
    __nv_bfloat162 _max_136 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_137);
    amax_0 = _max_136;
    __nv_bfloat162 _hxa_139 = amax_0;
    __nv_bfloat162 _hxb_139 = __as_bf16x2(dup[4]);
    unsigned int _hxr_139;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_139) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_139)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_139)));
    __nv_bfloat162 _max_138 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_139);
    amax_0 = _max_138;
    __nv_bfloat162 _hxa_141 = amax_0;
    __nv_bfloat162 _hxb_141 = __as_bf16x2(dup[5]);
    unsigned int _hxr_141;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_141) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_141)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_141)));
    __nv_bfloat162 _max_140 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_141);
    amax_0 = _max_140;
    __nv_bfloat162 _hxa_143 = amax_0;
    __nv_bfloat162 _hxb_143 = __as_bf16x2(dup[6]);
    unsigned int _hxr_143;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_143) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_143)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_143)));
    __nv_bfloat162 _max_142 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_143);
    amax_0 = _max_142;
    __nv_bfloat162 _hxa_145 = amax_0;
    __nv_bfloat162 _hxb_145 = __as_bf16x2(dup[7]);
    unsigned int _hxr_145;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_145) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_145)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_145)));
    __nv_bfloat162 _max_144 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_145);
    amax_0 = _max_144;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_147;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_147) : "r"(_bf16x2_mul_9));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_9);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_147) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_146;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_146) : "h"(_e4m3x2_bf16x2_4), "h"(_cvt_satfinite_e4m3x2_bf16x2_147));
    words_11[0] = _mov_b32_pack_u16x2_146;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_149;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_149) : "r"(_bf16x2_mul_11));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_11);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_149) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_148;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_148) : "h"(_e4m3x2_bf16x2_5), "h"(_cvt_satfinite_e4m3x2_bf16x2_149));
    words_11[1] = _mov_b32_pack_u16x2_148;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_151;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_151) : "r"(_bf16x2_mul_13));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_13);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_151) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_150;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_150) : "h"(_e4m3x2_bf16x2_6), "h"(_cvt_satfinite_e4m3x2_bf16x2_151));
    words_11[2] = _mov_b32_pack_u16x2_150;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_153;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_153) : "r"(_bf16x2_mul_15));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_15);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_153) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_152;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_152) : "h"(_e4m3x2_bf16x2_7), "h"(_cvt_satfinite_e4m3x2_bf16x2_153));
    words_11[3] = _mov_b32_pack_u16x2_152;
    reinterpret_cast<int4*>(row_output + (out_index + K))[0] = reinterpret_cast<int4*>(words_11 + 0)[0];
    int out_index_hi = grow_hi * (2 * K) + col;
    __nv_bfloat162 amax_12 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_155 = amax_12;
    __nv_bfloat162 _hxb_155 = __as_bf16x2(dgate_hi[0]);
    unsigned int _hxr_155;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_155) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_155)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_155)));
    __nv_bfloat162 _max_154 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_155);
    amax_12 = _max_154;
    __nv_bfloat162 _hxa_157 = amax_12;
    __nv_bfloat162 _hxb_157 = __as_bf16x2(dgate_hi[1]);
    unsigned int _hxr_157;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_157) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_157)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_157)));
    __nv_bfloat162 _max_156 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_157);
    amax_12 = _max_156;
    __nv_bfloat162 _hxa_159 = amax_12;
    __nv_bfloat162 _hxb_159 = __as_bf16x2(dgate_hi[2]);
    unsigned int _hxr_159;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_159) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_159)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_159)));
    __nv_bfloat162 _max_158 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_159);
    amax_12 = _max_158;
    __nv_bfloat162 _hxa_161 = amax_12;
    __nv_bfloat162 _hxb_161 = __as_bf16x2(dgate_hi[3]);
    unsigned int _hxr_161;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_161) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_161)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_161)));
    __nv_bfloat162 _max_160 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_161);
    amax_12 = _max_160;
    __nv_bfloat162 _hxa_163 = amax_12;
    __nv_bfloat162 _hxb_163 = __as_bf16x2(dgate_hi[4]);
    unsigned int _hxr_163;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_163) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_163)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_163)));
    __nv_bfloat162 _max_162 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_163);
    amax_12 = _max_162;
    __nv_bfloat162 _hxa_165 = amax_12;
    __nv_bfloat162 _hxb_165 = __as_bf16x2(dgate_hi[5]);
    unsigned int _hxr_165;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_165) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_165)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_165)));
    __nv_bfloat162 _max_164 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_165);
    amax_12 = _max_164;
    __nv_bfloat162 _hxa_167 = amax_12;
    __nv_bfloat162 _hxb_167 = __as_bf16x2(dgate_hi[6]);
    unsigned int _hxr_167;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_167) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_167)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_167)));
    __nv_bfloat162 _max_166 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_167);
    amax_12 = _max_166;
    __nv_bfloat162 _hxa_169 = amax_12;
    __nv_bfloat162 _hxb_169 = __as_bf16x2(dgate_hi[7]);
    unsigned int _hxr_169;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_169) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_169)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_169)));
    __nv_bfloat162 _max_168 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_169);
    amax_12 = _max_168;
    unsigned int mag_13 = __as_u32(amax_12) & 2147450879;
    unsigned int peer_14 = __shfl_xor_sync(0xFFFFFFFF, mag_13, 1);
    __nv_bfloat162 both_15 = __hmax2_nan(__as_bf16x2(mag_13), __as_bf16x2(peer_14));
    unsigned int folded_16 = __as_u32(__hmax2_nan(both_15, __as_bf16x2(__as_u32(both_15) >> 16)));
    unsigned int amax_bits_17 = (folded_16 & 65535) << 16;
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
    int scale_index_22 = ((grow_hi >> 7) * scale_cols + (scale_col >> 2)) * 512 + (grow_hi & 31) * 16 + (grow_hi >> 5 & 3) * 4 + (scale_col & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + scale_index_22) + (0)) = (unsigned char)(scale_19);
    }
    unsigned int words_23[4];
    uint32_t _bf16x2_mul_16;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_16) : "r"(dgate_hi[0]), "r"(inv2_21));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_17) : "r"(dgate_hi[1]), "r"(inv2_21));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_171;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_171) : "r"(_bf16x2_mul_17));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_17);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_171) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_170;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_170) : "h"(_e4m3x2_bf16x2_8), "h"(_cvt_satfinite_e4m3x2_bf16x2_171));
    words_23[0] = _mov_b32_pack_u16x2_170;
    uint32_t _bf16x2_mul_18;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_18) : "r"(dgate_hi[2]), "r"(inv2_21));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_19) : "r"(dgate_hi[3]), "r"(inv2_21));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_173;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_173) : "r"(_bf16x2_mul_19));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_19);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_173) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_172;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_172) : "h"(_e4m3x2_bf16x2_9), "h"(_cvt_satfinite_e4m3x2_bf16x2_173));
    words_23[1] = _mov_b32_pack_u16x2_172;
    uint32_t _bf16x2_mul_20;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_20) : "r"(dgate_hi[4]), "r"(inv2_21));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_21) : "r"(dgate_hi[5]), "r"(inv2_21));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_175;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_175) : "r"(_bf16x2_mul_21));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_21);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_175) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_174;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_174) : "h"(_e4m3x2_bf16x2_10), "h"(_cvt_satfinite_e4m3x2_bf16x2_175));
    words_23[2] = _mov_b32_pack_u16x2_174;
    uint32_t _bf16x2_mul_22;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_22) : "r"(dgate_hi[6]), "r"(inv2_21));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_23) : "r"(dgate_hi[7]), "r"(inv2_21));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_177;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_177) : "r"(_bf16x2_mul_23));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_23);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_177) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_176;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_176) : "h"(_e4m3x2_bf16x2_11), "h"(_cvt_satfinite_e4m3x2_bf16x2_177));
    words_23[3] = _mov_b32_pack_u16x2_176;
    reinterpret_cast<int4*>(row_output + out_index_hi)[0] = reinterpret_cast<int4*>(words_23 + 0)[0];
    __nv_bfloat162 amax_24 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_179 = amax_24;
    __nv_bfloat162 _hxb_179 = __as_bf16x2(dup_hi[0]);
    unsigned int _hxr_179;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_179) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_179)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_179)));
    __nv_bfloat162 _max_178 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_179);
    amax_24 = _max_178;
    __nv_bfloat162 _hxa_181 = amax_24;
    __nv_bfloat162 _hxb_181 = __as_bf16x2(dup_hi[1]);
    unsigned int _hxr_181;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_181) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_181)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_181)));
    __nv_bfloat162 _max_180 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_181);
    amax_24 = _max_180;
    __nv_bfloat162 _hxa_183 = amax_24;
    __nv_bfloat162 _hxb_183 = __as_bf16x2(dup_hi[2]);
    unsigned int _hxr_183;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_183) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_183)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_183)));
    __nv_bfloat162 _max_182 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_183);
    amax_24 = _max_182;
    __nv_bfloat162 _hxa_185 = amax_24;
    __nv_bfloat162 _hxb_185 = __as_bf16x2(dup_hi[3]);
    unsigned int _hxr_185;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_185) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_185)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_185)));
    __nv_bfloat162 _max_184 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_185);
    amax_24 = _max_184;
    __nv_bfloat162 _hxa_187 = amax_24;
    __nv_bfloat162 _hxb_187 = __as_bf16x2(dup_hi[4]);
    unsigned int _hxr_187;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_187) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_187)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_187)));
    __nv_bfloat162 _max_186 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_187);
    amax_24 = _max_186;
    __nv_bfloat162 _hxa_189 = amax_24;
    __nv_bfloat162 _hxb_189 = __as_bf16x2(dup_hi[5]);
    unsigned int _hxr_189;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_189) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_189)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_189)));
    __nv_bfloat162 _max_188 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_189);
    amax_24 = _max_188;
    __nv_bfloat162 _hxa_191 = amax_24;
    __nv_bfloat162 _hxb_191 = __as_bf16x2(dup_hi[6]);
    unsigned int _hxr_191;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_191) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_191)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_191)));
    __nv_bfloat162 _max_190 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_191);
    amax_24 = _max_190;
    __nv_bfloat162 _hxa_193 = amax_24;
    __nv_bfloat162 _hxb_193 = __as_bf16x2(dup_hi[7]);
    unsigned int _hxr_193;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_193) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_193)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_193)));
    __nv_bfloat162 _max_192 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_193);
    amax_24 = _max_192;
    unsigned int mag_25 = __as_u32(amax_24) & 2147450879;
    unsigned int peer_26 = __shfl_xor_sync(0xFFFFFFFF, mag_25, 1);
    __nv_bfloat162 both_27 = __hmax2_nan(__as_bf16x2(mag_25), __as_bf16x2(peer_26));
    unsigned int folded_28 = __as_u32(__hmax2_nan(both_27, __as_bf16x2(__as_u32(both_27) >> 16)));
    unsigned int amax_bits_29 = (folded_28 & 65535) << 16;
    int scale_i32_30 = (int)(amax_bits_29 + 2031616 >> 23) - 8;
    if (scale_i32_30 < 0) {
        scale_i32_30 = 0;
    }
    if ((amax_bits_29 & 2139095040) == 2139095040) {
        scale_i32_30 = 255;
    }
    unsigned int scale_31 = (unsigned int)scale_i32_30;
    unsigned int inv_32 = 254 - scale_31 << 7;
    if (scale_31 == 255) {
        inv_32 = 32704;
    }
    unsigned int inv2_33 = inv_32 | inv_32 << 16;
    int scale_index_34 = ((grow_hi >> 7) * scale_cols + (scale_col + (K >> 5) >> 2)) * 512 + (grow_hi & 31) * 16 + (grow_hi >> 5 & 3) * 4 + (scale_col + (K >> 5) & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + scale_index_34) + (0)) = (unsigned char)(scale_31);
    }
    unsigned int words_35[4];
    uint32_t _bf16x2_mul_24;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_24) : "r"(dup_hi[0]), "r"(inv2_33));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_25) : "r"(dup_hi[1]), "r"(inv2_33));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_195;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_195) : "r"(_bf16x2_mul_25));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_25);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_195) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_194;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_194) : "h"(_e4m3x2_bf16x2_12), "h"(_cvt_satfinite_e4m3x2_bf16x2_195));
    words_35[0] = _mov_b32_pack_u16x2_194;
    uint32_t _bf16x2_mul_26;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_26) : "r"(dup_hi[2]), "r"(inv2_33));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_27) : "r"(dup_hi[3]), "r"(inv2_33));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_197;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_197) : "r"(_bf16x2_mul_27));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_27);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_197) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_196;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_196) : "h"(_e4m3x2_bf16x2_13), "h"(_cvt_satfinite_e4m3x2_bf16x2_197));
    words_35[1] = _mov_b32_pack_u16x2_196;
    uint32_t _bf16x2_mul_28;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_28) : "r"(dup_hi[4]), "r"(inv2_33));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_29) : "r"(dup_hi[5]), "r"(inv2_33));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_199;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_199) : "r"(_bf16x2_mul_29));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_29);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_199) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_198;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_198) : "h"(_e4m3x2_bf16x2_14), "h"(_cvt_satfinite_e4m3x2_bf16x2_199));
    words_35[2] = _mov_b32_pack_u16x2_198;
    uint32_t _bf16x2_mul_30;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_30) : "r"(dup_hi[6]), "r"(inv2_33));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_31) : "r"(dup_hi[7]), "r"(inv2_33));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_201;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_201) : "r"(_bf16x2_mul_31));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_31);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_201) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_200;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_200) : "h"(_e4m3x2_bf16x2_15), "h"(_cvt_satfinite_e4m3x2_bf16x2_201));
    words_35[3] = _mov_b32_pack_u16x2_200;
    reinterpret_cast<int4*>(row_output + (out_index_hi + K))[0] = reinterpret_cast<int4*>(words_35 + 0)[0];
    __syncthreads();
    int lane_1 = tid & 31;
    int warp_1 = tid >> 5;
    int k8 = lane_1 & 7;
    int srow = (k8 >> 1) * 16 + (lane_1 >> 3) * 2 + (k8 & 1);
    unsigned int sswizzle = (unsigned int)(srow >> 3 & 6 | srow & 1);
    unsigned int cols[16];
    unsigned int strip_chunk0 = (unsigned int)(warp_1 * 2);
    unsigned int strip_src0 = tiles_addr + (unsigned int)(srow * 128) + ((strip_chunk0 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[0]), "=r"(cols[1]), "=r"(cols[2]), "=r"(cols[3])
        : "r"(strip_src0)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[4]), "=r"(cols[5]), "=r"(cols[6]), "=r"(cols[7])
        : "r"(strip_src0 + 1024)
        : "memory");
    unsigned int strip_chunk1 = (unsigned int)(warp_1 * 2 + 1);
    unsigned int strip_src1 = tiles_addr + (unsigned int)(srow * 128) + ((strip_chunk1 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[8]), "=r"(cols[9]), "=r"(cols[10]), "=r"(cols[11])
        : "r"(strip_src1)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[12]), "=r"(cols[13]), "=r"(cols[14]), "=r"(cols[15])
        : "r"(strip_src1 + 1024)
        : "memory");
    __nv_bfloat162 amax_a = __as_bf16x2(0);
    __nv_bfloat162 amax_b = __as_bf16x2(0);
    __nv_bfloat162 _hxa_203 = amax_a;
    __nv_bfloat162 _hxb_203 = __as_bf16x2(cols[0]);
    unsigned int _hxr_203;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_203) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_203)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_203)));
    __nv_bfloat162 _max_202 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_203);
    amax_a = _max_202;
    __nv_bfloat162 _hxa_205 = amax_b;
    __nv_bfloat162 _hxb_205 = __as_bf16x2(cols[8]);
    unsigned int _hxr_205;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_205) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_205)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_205)));
    __nv_bfloat162 _max_204 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_205);
    amax_b = _max_204;
    __nv_bfloat162 _hxa_207 = amax_a;
    __nv_bfloat162 _hxb_207 = __as_bf16x2(cols[1]);
    unsigned int _hxr_207;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_207) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_207)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_207)));
    __nv_bfloat162 _max_206 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_207);
    amax_a = _max_206;
    __nv_bfloat162 _hxa_209 = amax_b;
    __nv_bfloat162 _hxb_209 = __as_bf16x2(cols[9]);
    unsigned int _hxr_209;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_209) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_209)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_209)));
    __nv_bfloat162 _max_208 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_209);
    amax_b = _max_208;
    __nv_bfloat162 _hxa_211 = amax_a;
    __nv_bfloat162 _hxb_211 = __as_bf16x2(cols[2]);
    unsigned int _hxr_211;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_211) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_211)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_211)));
    __nv_bfloat162 _max_210 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_211);
    amax_a = _max_210;
    __nv_bfloat162 _hxa_213 = amax_b;
    __nv_bfloat162 _hxb_213 = __as_bf16x2(cols[10]);
    unsigned int _hxr_213;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_213) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_213)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_213)));
    __nv_bfloat162 _max_212 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_213);
    amax_b = _max_212;
    __nv_bfloat162 _hxa_215 = amax_a;
    __nv_bfloat162 _hxb_215 = __as_bf16x2(cols[3]);
    unsigned int _hxr_215;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_215) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_215)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_215)));
    __nv_bfloat162 _max_214 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_215);
    amax_a = _max_214;
    __nv_bfloat162 _hxa_217 = amax_b;
    __nv_bfloat162 _hxb_217 = __as_bf16x2(cols[11]);
    unsigned int _hxr_217;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_217) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_217)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_217)));
    __nv_bfloat162 _max_216 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_217);
    amax_b = _max_216;
    __nv_bfloat162 _hxa_219 = amax_a;
    __nv_bfloat162 _hxb_219 = __as_bf16x2(cols[4]);
    unsigned int _hxr_219;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_219) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_219)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_219)));
    __nv_bfloat162 _max_218 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_219);
    amax_a = _max_218;
    __nv_bfloat162 _hxa_221 = amax_b;
    __nv_bfloat162 _hxb_221 = __as_bf16x2(cols[12]);
    unsigned int _hxr_221;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_221) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_221)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_221)));
    __nv_bfloat162 _max_220 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_221);
    amax_b = _max_220;
    __nv_bfloat162 _hxa_223 = amax_a;
    __nv_bfloat162 _hxb_223 = __as_bf16x2(cols[5]);
    unsigned int _hxr_223;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_223) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_223)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_223)));
    __nv_bfloat162 _max_222 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_223);
    amax_a = _max_222;
    __nv_bfloat162 _hxa_225 = amax_b;
    __nv_bfloat162 _hxb_225 = __as_bf16x2(cols[13]);
    unsigned int _hxr_225;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_225) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_225)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_225)));
    __nv_bfloat162 _max_224 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_225);
    amax_b = _max_224;
    __nv_bfloat162 _hxa_227 = amax_a;
    __nv_bfloat162 _hxb_227 = __as_bf16x2(cols[6]);
    unsigned int _hxr_227;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_227) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_227)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_227)));
    __nv_bfloat162 _max_226 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_227);
    amax_a = _max_226;
    __nv_bfloat162 _hxa_229 = amax_b;
    __nv_bfloat162 _hxb_229 = __as_bf16x2(cols[14]);
    unsigned int _hxr_229;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_229) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_229)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_229)));
    __nv_bfloat162 _max_228 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_229);
    amax_b = _max_228;
    __nv_bfloat162 _hxa_231 = amax_a;
    __nv_bfloat162 _hxb_231 = __as_bf16x2(cols[7]);
    unsigned int _hxr_231;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_231) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_231)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_231)));
    __nv_bfloat162 _max_230 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_231);
    amax_a = _max_230;
    __nv_bfloat162 _hxa_233 = amax_b;
    __nv_bfloat162 _hxb_233 = __as_bf16x2(cols[15]);
    unsigned int _hxr_233;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_233) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_233)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_233)));
    __nv_bfloat162 _max_232 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_233);
    amax_b = _max_232;
    unsigned int mag_a = __as_u32(amax_a) & 2147450879;
    unsigned int mag_b = __as_u32(amax_b) & 2147450879;
    unsigned int lows_ab = mag_a & 65535 | mag_b << 16;
    unsigned int highs_ab = mag_a >> 16 | mag_b & 4294901760u;
    __nv_bfloat162 col_max = __hmax2_nan(__as_bf16x2(lows_ab), __as_bf16x2(highs_ab));
    unsigned int peer1 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(col_max), 1);
    col_max = __hmax2_nan(col_max, __as_bf16x2(peer1));
    unsigned int col_max_bits = __as_u32(col_max);
    unsigned int amax_bits_36 = (col_max_bits & 65535) << 16;
    int scale_i32_37 = (int)(amax_bits_36 + 2031616 >> 23) - 8;
    if (scale_i32_37 < 0) {
        scale_i32_37 = 0;
    }
    if ((amax_bits_36 & 2139095040) == 2139095040) {
        scale_i32_37 = 255;
    }
    unsigned int scale_38 = (unsigned int)scale_i32_37;
    unsigned int inv_39 = 254 - scale_38 << 7;
    if (scale_38 == 255) {
        inv_39 = 32704;
    }
    unsigned int inv2_40 = inv_39 | inv_39 << 16;
    unsigned int amax_bits_41 = (col_max_bits >> 16 & 65535) << 16;
    int scale_i32_42 = (int)(amax_bits_41 + 2031616 >> 23) - 8;
    if (scale_i32_42 < 0) {
        scale_i32_42 = 0;
    }
    if ((amax_bits_41 & 2139095040) == 2139095040) {
        scale_i32_42 = 255;
    }
    unsigned int scale_43 = (unsigned int)scale_i32_42;
    unsigned int inv_44 = 254 - scale_43 << 7;
    if (scale_43 == 255) {
        inv_44 = 32704;
    }
    unsigned int inv2_45 = inv_44 | inv_44 << 16;
    int gcol_a = bx * 64 + warp_1 * 16 + (lane_1 >> 2);
    int gcol_b = gcol_a + 8;
    int out_row = by * 64 + (lane_1 & 3) * 16;
    unsigned int words_a[4];
    unsigned int words_b[4];
    uint32_t _bf16x2_mul_32;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_32) : "r"(cols[0]), "r"(inv2_40));
    uint16_t _e4m3x2_bf16x2_16;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_16) : "r"(_bf16x2_mul_32));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_32);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_16) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_33;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_33) : "r"(cols[1]), "r"(inv2_40));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_235;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_235) : "r"(_bf16x2_mul_33));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_33);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_235) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_234;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_234) : "h"(_e4m3x2_bf16x2_16), "h"(_cvt_satfinite_e4m3x2_bf16x2_235));
    words_a[0] = _mov_b32_pack_u16x2_234;
    uint32_t _bf16x2_mul_34;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_34) : "r"(cols[8]), "r"(inv2_45));
    uint16_t _e4m3x2_bf16x2_17;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_17) : "r"(_bf16x2_mul_34));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_34);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_17) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_35;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_35) : "r"(cols[9]), "r"(inv2_45));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_237;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_237) : "r"(_bf16x2_mul_35));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_35);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_237) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_236;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_236) : "h"(_e4m3x2_bf16x2_17), "h"(_cvt_satfinite_e4m3x2_bf16x2_237));
    words_b[0] = _mov_b32_pack_u16x2_236;
    uint32_t _bf16x2_mul_36;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_36) : "r"(cols[2]), "r"(inv2_40));
    uint16_t _e4m3x2_bf16x2_18;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_18) : "r"(_bf16x2_mul_36));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_36);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_18) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_37;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_37) : "r"(cols[3]), "r"(inv2_40));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_239;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_239) : "r"(_bf16x2_mul_37));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_37);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_239) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_238;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_238) : "h"(_e4m3x2_bf16x2_18), "h"(_cvt_satfinite_e4m3x2_bf16x2_239));
    words_a[1] = _mov_b32_pack_u16x2_238;
    uint32_t _bf16x2_mul_38;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_38) : "r"(cols[10]), "r"(inv2_45));
    uint16_t _e4m3x2_bf16x2_19;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_19) : "r"(_bf16x2_mul_38));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_38);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_19) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_39;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_39) : "r"(cols[11]), "r"(inv2_45));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_241;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_241) : "r"(_bf16x2_mul_39));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_39);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_241) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_240;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_240) : "h"(_e4m3x2_bf16x2_19), "h"(_cvt_satfinite_e4m3x2_bf16x2_241));
    words_b[1] = _mov_b32_pack_u16x2_240;
    uint32_t _bf16x2_mul_40;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_40) : "r"(cols[4]), "r"(inv2_40));
    uint16_t _e4m3x2_bf16x2_20;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_20) : "r"(_bf16x2_mul_40));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_40);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_20) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_41;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_41) : "r"(cols[5]), "r"(inv2_40));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_243;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_243) : "r"(_bf16x2_mul_41));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_41);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_243) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_242;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_242) : "h"(_e4m3x2_bf16x2_20), "h"(_cvt_satfinite_e4m3x2_bf16x2_243));
    words_a[2] = _mov_b32_pack_u16x2_242;
    uint32_t _bf16x2_mul_42;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_42) : "r"(cols[12]), "r"(inv2_45));
    uint16_t _e4m3x2_bf16x2_21;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_21) : "r"(_bf16x2_mul_42));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_42);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_21) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_43;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_43) : "r"(cols[13]), "r"(inv2_45));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_245;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_245) : "r"(_bf16x2_mul_43));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_43);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_245) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_244;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_244) : "h"(_e4m3x2_bf16x2_21), "h"(_cvt_satfinite_e4m3x2_bf16x2_245));
    words_b[2] = _mov_b32_pack_u16x2_244;
    uint32_t _bf16x2_mul_44;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_44) : "r"(cols[6]), "r"(inv2_40));
    uint16_t _e4m3x2_bf16x2_22;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_22) : "r"(_bf16x2_mul_44));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_44);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_22) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_45;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_45) : "r"(cols[7]), "r"(inv2_40));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_247;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_247) : "r"(_bf16x2_mul_45));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_45);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_247) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_246;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_246) : "h"(_e4m3x2_bf16x2_22), "h"(_cvt_satfinite_e4m3x2_bf16x2_247));
    words_a[3] = _mov_b32_pack_u16x2_246;
    uint32_t _bf16x2_mul_46;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_46) : "r"(cols[14]), "r"(inv2_45));
    uint16_t _e4m3x2_bf16x2_23;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_23) : "r"(_bf16x2_mul_46));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_46);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_23) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_47;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_47) : "r"(cols[15]), "r"(inv2_45));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_249;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_249) : "r"(_bf16x2_mul_47));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_47);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_249) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_248;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_248) : "h"(_e4m3x2_bf16x2_23), "h"(_cvt_satfinite_e4m3x2_bf16x2_249));
    words_b[3] = _mov_b32_pack_u16x2_248;
    reinterpret_cast<int4*>(col_output + (gcol_a * M + out_row))[0] = reinterpret_cast<int4*>(words_a + 0)[0];
    reinterpret_cast<int4*>(col_output + (gcol_b * M + out_row))[0] = reinterpret_cast<int4*>(words_b + 0)[0];
    if ((lane_1 & 1) == 0) {
        int row_tiles = M >> 7;
        int rblk = by * 2 + ((lane_1 & 3) >> 1);
        int col_scale_a = ((gcol_a >> 7) * row_tiles + (rblk >> 2)) * 512 + (gcol_a & 31) * 16 + (gcol_a >> 5 & 3) * 4 + (rblk & 3);
        int col_scale_b = ((gcol_b >> 7) * row_tiles + (rblk >> 2)) * 512 + (gcol_b & 31) * 16 + (gcol_b >> 5 & 3) * 4 + (rblk & 3);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_a) + (0)) = (unsigned char)(scale_38);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b) + (0)) = (unsigned char)(scale_43);
    }
    int lane_46 = tid & 31;
    int warp_47 = tid >> 5;
    int k8_48 = lane_46 & 7;
    int srow_49 = (k8_48 >> 1) * 16 + (lane_46 >> 3) * 2 + (k8_48 & 1);
    unsigned int sswizzle_50 = (unsigned int)(srow_49 >> 3 & 6 | srow_49 & 1);
    unsigned int cols_51[16];
    unsigned int strip_chunk0_52 = (unsigned int)(warp_47 * 2);
    unsigned int strip_src0_53 = tiles_addr + 8192 + (unsigned int)(srow_49 * 128) + ((strip_chunk0_52 ^ sswizzle_50) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_51[0]), "=r"(cols_51[1]), "=r"(cols_51[2]), "=r"(cols_51[3])
        : "r"(strip_src0_53)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_51[4]), "=r"(cols_51[5]), "=r"(cols_51[6]), "=r"(cols_51[7])
        : "r"(strip_src0_53 + 1024)
        : "memory");
    unsigned int strip_chunk1_54 = (unsigned int)(warp_47 * 2 + 1);
    unsigned int strip_src1_55 = tiles_addr + 8192 + (unsigned int)(srow_49 * 128) + ((strip_chunk1_54 ^ sswizzle_50) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_51[8]), "=r"(cols_51[9]), "=r"(cols_51[10]), "=r"(cols_51[11])
        : "r"(strip_src1_55)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_51[12]), "=r"(cols_51[13]), "=r"(cols_51[14]), "=r"(cols_51[15])
        : "r"(strip_src1_55 + 1024)
        : "memory");
    __nv_bfloat162 amax_a_56 = __as_bf16x2(0);
    __nv_bfloat162 amax_b_57 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_251 = amax_a_56;
    __nv_bfloat162 _hxb_251 = __as_bf16x2(cols_51[0]);
    unsigned int _hxr_251;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_251) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_251)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_251)));
    __nv_bfloat162 _max_250 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_251);
    amax_a_56 = _max_250;
    __nv_bfloat162 _hxa_253 = amax_b_57;
    __nv_bfloat162 _hxb_253 = __as_bf16x2(cols_51[8]);
    unsigned int _hxr_253;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_253) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_253)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_253)));
    __nv_bfloat162 _max_252 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_253);
    amax_b_57 = _max_252;
    __nv_bfloat162 _hxa_255 = amax_a_56;
    __nv_bfloat162 _hxb_255 = __as_bf16x2(cols_51[1]);
    unsigned int _hxr_255;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_255) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_255)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_255)));
    __nv_bfloat162 _max_254 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_255);
    amax_a_56 = _max_254;
    __nv_bfloat162 _hxa_257 = amax_b_57;
    __nv_bfloat162 _hxb_257 = __as_bf16x2(cols_51[9]);
    unsigned int _hxr_257;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_257) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_257)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_257)));
    __nv_bfloat162 _max_256 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_257);
    amax_b_57 = _max_256;
    __nv_bfloat162 _hxa_259 = amax_a_56;
    __nv_bfloat162 _hxb_259 = __as_bf16x2(cols_51[2]);
    unsigned int _hxr_259;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_259) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_259)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_259)));
    __nv_bfloat162 _max_258 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_259);
    amax_a_56 = _max_258;
    __nv_bfloat162 _hxa_261 = amax_b_57;
    __nv_bfloat162 _hxb_261 = __as_bf16x2(cols_51[10]);
    unsigned int _hxr_261;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_261) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_261)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_261)));
    __nv_bfloat162 _max_260 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_261);
    amax_b_57 = _max_260;
    __nv_bfloat162 _hxa_263 = amax_a_56;
    __nv_bfloat162 _hxb_263 = __as_bf16x2(cols_51[3]);
    unsigned int _hxr_263;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_263) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_263)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_263)));
    __nv_bfloat162 _max_262 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_263);
    amax_a_56 = _max_262;
    __nv_bfloat162 _hxa_265 = amax_b_57;
    __nv_bfloat162 _hxb_265 = __as_bf16x2(cols_51[11]);
    unsigned int _hxr_265;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_265) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_265)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_265)));
    __nv_bfloat162 _max_264 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_265);
    amax_b_57 = _max_264;
    __nv_bfloat162 _hxa_267 = amax_a_56;
    __nv_bfloat162 _hxb_267 = __as_bf16x2(cols_51[4]);
    unsigned int _hxr_267;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_267) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_267)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_267)));
    __nv_bfloat162 _max_266 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_267);
    amax_a_56 = _max_266;
    __nv_bfloat162 _hxa_269 = amax_b_57;
    __nv_bfloat162 _hxb_269 = __as_bf16x2(cols_51[12]);
    unsigned int _hxr_269;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_269) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_269)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_269)));
    __nv_bfloat162 _max_268 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_269);
    amax_b_57 = _max_268;
    __nv_bfloat162 _hxa_271 = amax_a_56;
    __nv_bfloat162 _hxb_271 = __as_bf16x2(cols_51[5]);
    unsigned int _hxr_271;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_271) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_271)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_271)));
    __nv_bfloat162 _max_270 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_271);
    amax_a_56 = _max_270;
    __nv_bfloat162 _hxa_273 = amax_b_57;
    __nv_bfloat162 _hxb_273 = __as_bf16x2(cols_51[13]);
    unsigned int _hxr_273;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_273) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_273)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_273)));
    __nv_bfloat162 _max_272 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_273);
    amax_b_57 = _max_272;
    __nv_bfloat162 _hxa_275 = amax_a_56;
    __nv_bfloat162 _hxb_275 = __as_bf16x2(cols_51[6]);
    unsigned int _hxr_275;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_275) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_275)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_275)));
    __nv_bfloat162 _max_274 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_275);
    amax_a_56 = _max_274;
    __nv_bfloat162 _hxa_277 = amax_b_57;
    __nv_bfloat162 _hxb_277 = __as_bf16x2(cols_51[14]);
    unsigned int _hxr_277;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_277) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_277)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_277)));
    __nv_bfloat162 _max_276 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_277);
    amax_b_57 = _max_276;
    __nv_bfloat162 _hxa_279 = amax_a_56;
    __nv_bfloat162 _hxb_279 = __as_bf16x2(cols_51[7]);
    unsigned int _hxr_279;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_279) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_279)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_279)));
    __nv_bfloat162 _max_278 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_279);
    amax_a_56 = _max_278;
    __nv_bfloat162 _hxa_281 = amax_b_57;
    __nv_bfloat162 _hxb_281 = __as_bf16x2(cols_51[15]);
    unsigned int _hxr_281;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_281) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_281)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_281)));
    __nv_bfloat162 _max_280 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_281);
    amax_b_57 = _max_280;
    unsigned int mag_a_58 = __as_u32(amax_a_56) & 2147450879;
    unsigned int mag_b_59 = __as_u32(amax_b_57) & 2147450879;
    unsigned int lows_ab_60 = mag_a_58 & 65535 | mag_b_59 << 16;
    unsigned int highs_ab_61 = mag_a_58 >> 16 | mag_b_59 & 4294901760u;
    __nv_bfloat162 col_max_62 = __hmax2_nan(__as_bf16x2(lows_ab_60), __as_bf16x2(highs_ab_61));
    unsigned int peer1_63 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(col_max_62), 1);
    col_max_62 = __hmax2_nan(col_max_62, __as_bf16x2(peer1_63));
    unsigned int col_max_bits_64 = __as_u32(col_max_62);
    unsigned int amax_bits_65 = (col_max_bits_64 & 65535) << 16;
    int scale_i32_66 = (int)(amax_bits_65 + 2031616 >> 23) - 8;
    if (scale_i32_66 < 0) {
        scale_i32_66 = 0;
    }
    if ((amax_bits_65 & 2139095040) == 2139095040) {
        scale_i32_66 = 255;
    }
    unsigned int scale_67 = (unsigned int)scale_i32_66;
    unsigned int inv_68 = 254 - scale_67 << 7;
    if (scale_67 == 255) {
        inv_68 = 32704;
    }
    unsigned int inv2_69 = inv_68 | inv_68 << 16;
    unsigned int amax_bits_70 = (col_max_bits_64 >> 16 & 65535) << 16;
    int scale_i32_71 = (int)(amax_bits_70 + 2031616 >> 23) - 8;
    if (scale_i32_71 < 0) {
        scale_i32_71 = 0;
    }
    if ((amax_bits_70 & 2139095040) == 2139095040) {
        scale_i32_71 = 255;
    }
    unsigned int scale_72 = (unsigned int)scale_i32_71;
    unsigned int inv_73 = 254 - scale_72 << 7;
    if (scale_72 == 255) {
        inv_73 = 32704;
    }
    unsigned int inv2_74 = inv_73 | inv_73 << 16;
    int gcol_a_75 = bx * 64 + K + warp_47 * 16 + (lane_46 >> 2);
    int gcol_b_76 = gcol_a_75 + 8;
    int out_row_77 = by * 64 + (lane_46 & 3) * 16;
    unsigned int words_a_78[4];
    unsigned int words_b_79[4];
    uint32_t _bf16x2_mul_48;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_48) : "r"(cols_51[0]), "r"(inv2_69));
    uint16_t _e4m3x2_bf16x2_24;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_24) : "r"(_bf16x2_mul_48));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_48);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_24) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_49;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_49) : "r"(cols_51[1]), "r"(inv2_69));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_283;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_283) : "r"(_bf16x2_mul_49));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_49);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_283) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_282;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_282) : "h"(_e4m3x2_bf16x2_24), "h"(_cvt_satfinite_e4m3x2_bf16x2_283));
    words_a_78[0] = _mov_b32_pack_u16x2_282;
    uint32_t _bf16x2_mul_50;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_50) : "r"(cols_51[8]), "r"(inv2_74));
    uint16_t _e4m3x2_bf16x2_25;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_25) : "r"(_bf16x2_mul_50));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_50);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_25) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_51;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_51) : "r"(cols_51[9]), "r"(inv2_74));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_285;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_285) : "r"(_bf16x2_mul_51));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_51);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_285) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_284;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_284) : "h"(_e4m3x2_bf16x2_25), "h"(_cvt_satfinite_e4m3x2_bf16x2_285));
    words_b_79[0] = _mov_b32_pack_u16x2_284;
    uint32_t _bf16x2_mul_52;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_52) : "r"(cols_51[2]), "r"(inv2_69));
    uint16_t _e4m3x2_bf16x2_26;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_26) : "r"(_bf16x2_mul_52));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_52);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_26) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_53;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_53) : "r"(cols_51[3]), "r"(inv2_69));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_287;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_287) : "r"(_bf16x2_mul_53));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_53);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_287) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_286;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_286) : "h"(_e4m3x2_bf16x2_26), "h"(_cvt_satfinite_e4m3x2_bf16x2_287));
    words_a_78[1] = _mov_b32_pack_u16x2_286;
    uint32_t _bf16x2_mul_54;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_54) : "r"(cols_51[10]), "r"(inv2_74));
    uint16_t _e4m3x2_bf16x2_27;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_27) : "r"(_bf16x2_mul_54));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_54);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_27) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_55;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_55) : "r"(cols_51[11]), "r"(inv2_74));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_289;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_289) : "r"(_bf16x2_mul_55));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_55);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_289) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_288;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_288) : "h"(_e4m3x2_bf16x2_27), "h"(_cvt_satfinite_e4m3x2_bf16x2_289));
    words_b_79[1] = _mov_b32_pack_u16x2_288;
    uint32_t _bf16x2_mul_56;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_56) : "r"(cols_51[4]), "r"(inv2_69));
    uint16_t _e4m3x2_bf16x2_28;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_28) : "r"(_bf16x2_mul_56));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_56);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_28) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_57;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_57) : "r"(cols_51[5]), "r"(inv2_69));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_291;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_291) : "r"(_bf16x2_mul_57));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_57);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_291) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_290;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_290) : "h"(_e4m3x2_bf16x2_28), "h"(_cvt_satfinite_e4m3x2_bf16x2_291));
    words_a_78[2] = _mov_b32_pack_u16x2_290;
    uint32_t _bf16x2_mul_58;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_58) : "r"(cols_51[12]), "r"(inv2_74));
    uint16_t _e4m3x2_bf16x2_29;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_29) : "r"(_bf16x2_mul_58));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_58);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_29) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_59;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_59) : "r"(cols_51[13]), "r"(inv2_74));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_293;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_293) : "r"(_bf16x2_mul_59));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_59);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_293) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_292;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_292) : "h"(_e4m3x2_bf16x2_29), "h"(_cvt_satfinite_e4m3x2_bf16x2_293));
    words_b_79[2] = _mov_b32_pack_u16x2_292;
    uint32_t _bf16x2_mul_60;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_60) : "r"(cols_51[6]), "r"(inv2_69));
    uint16_t _e4m3x2_bf16x2_30;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_30) : "r"(_bf16x2_mul_60));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_60);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_30) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_61;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_61) : "r"(cols_51[7]), "r"(inv2_69));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_295;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_295) : "r"(_bf16x2_mul_61));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_61);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_295) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_294;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_294) : "h"(_e4m3x2_bf16x2_30), "h"(_cvt_satfinite_e4m3x2_bf16x2_295));
    words_a_78[3] = _mov_b32_pack_u16x2_294;
    uint32_t _bf16x2_mul_62;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_62) : "r"(cols_51[14]), "r"(inv2_74));
    uint16_t _e4m3x2_bf16x2_31;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_31) : "r"(_bf16x2_mul_62));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_62);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_31) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _bf16x2_mul_63;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_63) : "r"(cols_51[15]), "r"(inv2_74));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_297;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_297) : "r"(_bf16x2_mul_63));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_63);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_297) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_296;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_296) : "h"(_e4m3x2_bf16x2_31), "h"(_cvt_satfinite_e4m3x2_bf16x2_297));
    words_b_79[3] = _mov_b32_pack_u16x2_296;
    reinterpret_cast<int4*>(col_output + (gcol_a_75 * M + out_row_77))[0] = reinterpret_cast<int4*>(words_a_78 + 0)[0];
    reinterpret_cast<int4*>(col_output + (gcol_b_76 * M + out_row_77))[0] = reinterpret_cast<int4*>(words_b_79 + 0)[0];
    if ((lane_46 & 1) == 0) {
        int row_tiles_1 = M >> 7;
        int rblk_1 = by * 2 + ((lane_46 & 3) >> 1);
        int col_scale_a_1 = ((gcol_a_75 >> 7) * row_tiles_1 + (rblk_1 >> 2)) * 512 + (gcol_a_75 & 31) * 16 + (gcol_a_75 >> 5 & 3) * 4 + (rblk_1 & 3);
        int col_scale_b_1 = ((gcol_b_76 >> 7) * row_tiles_1 + (rblk_1 >> 2)) * 512 + (gcol_b_76 & 31) * 16 + (gcol_b_76 >> 5 & 3) * 4 + (rblk_1 & 3);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_a_1) + (0)) = (unsigned char)(scale_67);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b_1) + (0)) = (unsigned char)(scale_72);
    }
}

} // extern "C"
