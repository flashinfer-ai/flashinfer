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
        float2 cvt_f32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float2 _f32x2_mul_b_8 = make_float2(-1.442695f, -1.442695f);
        float2 t;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t) : "l"(*(const unsigned long long*)&cvt_f32), "l"(*(const unsigned long long*)&_f32x2_mul_b_8));
        float e0 = approx_exp2(t.x);
        float e1 = approx_exp2(t.y);
        float2 nd = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0, e1));
        float d0 = -nd.x;
        float d1 = -nd.y;
        float r0 = approx_rcp(d0);
        float r1 = approx_rcp(d1);
        float2 r = make_float2(r0, r1);
        float2 s = fma_f32x2_rn_noftz(r, fma_f32x2_rn_noftz(nd, r, make_float2(1.0f, 1.0f)), r);
        float2 s_1 = s;
        float2 act;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act) : "l"(*(const unsigned long long*)&cvt_f32), "l"(*(const unsigned long long*)&s_1));
        float2 _f32x2_mul_a_9 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1), s_1), cvt_f32_2);
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
        float2 _f32x2_mul_b_10 = make_float2(-1.442695f, -1.442695f);
        float2 t_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_0) : "l"(*(const unsigned long long*)&cvt_f32_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_10));
        float e0_0 = approx_exp2(t_0.x);
        float e1_0 = approx_exp2(t_0.y);
        float2 nd_0 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_0, e1_0));
        float d0_0 = -nd_0.x;
        float d1_0 = -nd_0.y;
        float r0_0 = approx_rcp(d0_0);
        float r1_0 = approx_rcp(d1_0);
        float2 r_0 = make_float2(r0_0, r1_0);
        float2 s_0 = fma_f32x2_rn_noftz(r_0, fma_f32x2_rn_noftz(nd_0, r_0, make_float2(1.0f, 1.0f)), r_0);
        float2 s_1_0 = s_0;
        float2 act_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_0) : "l"(*(const unsigned long long*)&cvt_f32_0), "l"(*(const unsigned long long*)&s_1_0));
        float2 _f32x2_mul_a_11 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1_0), s_1_0), cvt_f32_2_0);
        float2 mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_11), "l"(*(const unsigned long long*)&cvt_f32_1_0));
        float2 mul_1_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_0) : "l"(*(const unsigned long long*)&act_0), "l"(*(const unsigned long long*)&cvt_f32_2_0));
        __nv_bfloat162 bf16x2_0 = __float22bfloat162_rn(make_float2(mul_0.x, mul_0.y));
        __nv_bfloat162 bf16x2_1_0 = __float22bfloat162_rn(make_float2(mul_1_0.x, mul_1_0.y));
        dgate[1] = __as_u32(bf16x2_0);
        dup[1] = __as_u32(bf16x2_1_0);
        float2 cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 cvt_f32_1_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 cvt_f32_2_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float2 _f32x2_mul_b_12 = make_float2(-1.442695f, -1.442695f);
        float2 t_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_1) : "l"(*(const unsigned long long*)&cvt_f32_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_12));
        float e0_1 = approx_exp2(t_1.x);
        float e1_1 = approx_exp2(t_1.y);
        float2 nd_1 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_1, e1_1));
        float d0_1 = -nd_1.x;
        float d1_1 = -nd_1.y;
        float r0_1 = approx_rcp(d0_1);
        float r1_1 = approx_rcp(d1_1);
        float2 r_1 = make_float2(r0_1, r1_1);
        float2 s_2 = fma_f32x2_rn_noftz(r_1, fma_f32x2_rn_noftz(nd_1, r_1, make_float2(1.0f, 1.0f)), r_1);
        float2 s_1_1 = s_2;
        float2 act_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_1) : "l"(*(const unsigned long long*)&cvt_f32_3), "l"(*(const unsigned long long*)&s_1_1));
        float2 _f32x2_mul_a_13 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1_1), s_1_1), cvt_f32_2_1);
        float2 mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_13), "l"(*(const unsigned long long*)&cvt_f32_1_1));
        float2 mul_1_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_1) : "l"(*(const unsigned long long*)&act_1), "l"(*(const unsigned long long*)&cvt_f32_2_1));
        __nv_bfloat162 bf16x2_2 = __float22bfloat162_rn(make_float2(mul_2.x, mul_2.y));
        __nv_bfloat162 bf16x2_1_1 = __float22bfloat162_rn(make_float2(mul_1_1.x, mul_1_1.y));
        dgate[2] = __as_u32(bf16x2_2);
        dup[2] = __as_u32(bf16x2_1_1);
        float2 cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 cvt_f32_1_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 cvt_f32_2_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float2 _f32x2_mul_b_14 = make_float2(-1.442695f, -1.442695f);
        float2 t_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_2) : "l"(*(const unsigned long long*)&cvt_f32_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_14));
        float e0_2 = approx_exp2(t_2.x);
        float e1_2 = approx_exp2(t_2.y);
        float2 nd_2 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_2, e1_2));
        float d0_2 = -nd_2.x;
        float d1_2 = -nd_2.y;
        float r0_2 = approx_rcp(d0_2);
        float r1_2 = approx_rcp(d1_2);
        float2 r_2 = make_float2(r0_2, r1_2);
        float2 s_3 = fma_f32x2_rn_noftz(r_2, fma_f32x2_rn_noftz(nd_2, r_2, make_float2(1.0f, 1.0f)), r_2);
        float2 s_1_2 = s_3;
        float2 act_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_2) : "l"(*(const unsigned long long*)&cvt_f32_4), "l"(*(const unsigned long long*)&s_1_2));
        float2 _f32x2_mul_a_15 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1_2), s_1_2), cvt_f32_2_2);
        float2 mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&cvt_f32_1_2));
        float2 mul_1_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_2) : "l"(*(const unsigned long long*)&act_2), "l"(*(const unsigned long long*)&cvt_f32_2_2));
        __nv_bfloat162 bf16x2_3 = __float22bfloat162_rn(make_float2(mul_3.x, mul_3.y));
        __nv_bfloat162 bf16x2_1_2 = __float22bfloat162_rn(make_float2(mul_1_2.x, mul_1_2.y));
        dgate[3] = __as_u32(bf16x2_3);
        dup[3] = __as_u32(bf16x2_1_2);
        float2 cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 cvt_f32_1_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float2 cvt_f32_2_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[4]));
        float2 _f32x2_mul_b_16 = make_float2(-1.442695f, -1.442695f);
        float2 t_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_3) : "l"(*(const unsigned long long*)&cvt_f32_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_16));
        float e0_3 = approx_exp2(t_3.x);
        float e1_3 = approx_exp2(t_3.y);
        float2 nd_3 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_3, e1_3));
        float d0_3 = -nd_3.x;
        float d1_3 = -nd_3.y;
        float r0_3 = approx_rcp(d0_3);
        float r1_3 = approx_rcp(d1_3);
        float2 r_3 = make_float2(r0_3, r1_3);
        float2 s_4 = fma_f32x2_rn_noftz(r_3, fma_f32x2_rn_noftz(nd_3, r_3, make_float2(1.0f, 1.0f)), r_3);
        float2 s_1_3 = s_4;
        float2 act_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_3) : "l"(*(const unsigned long long*)&cvt_f32_5), "l"(*(const unsigned long long*)&s_1_3));
        float2 _f32x2_mul_a_17 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1_3), s_1_3), cvt_f32_2_3);
        float2 mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&cvt_f32_1_3));
        float2 mul_1_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_3) : "l"(*(const unsigned long long*)&act_3), "l"(*(const unsigned long long*)&cvt_f32_2_3));
        __nv_bfloat162 bf16x2_4 = __float22bfloat162_rn(make_float2(mul_4.x, mul_4.y));
        __nv_bfloat162 bf16x2_1_3 = __float22bfloat162_rn(make_float2(mul_1_3.x, mul_1_3.y));
        dgate[4] = __as_u32(bf16x2_4);
        dup[4] = __as_u32(bf16x2_1_3);
        float2 cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 cvt_f32_1_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float2 cvt_f32_2_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[5]));
        float2 _f32x2_mul_b_18 = make_float2(-1.442695f, -1.442695f);
        float2 t_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_4) : "l"(*(const unsigned long long*)&cvt_f32_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_18));
        float e0_4 = approx_exp2(t_4.x);
        float e1_4 = approx_exp2(t_4.y);
        float2 nd_4 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_4, e1_4));
        float d0_4 = -nd_4.x;
        float d1_4 = -nd_4.y;
        float r0_4 = approx_rcp(d0_4);
        float r1_4 = approx_rcp(d1_4);
        float2 r_4 = make_float2(r0_4, r1_4);
        float2 s_5 = fma_f32x2_rn_noftz(r_4, fma_f32x2_rn_noftz(nd_4, r_4, make_float2(1.0f, 1.0f)), r_4);
        float2 s_1_4 = s_5;
        float2 act_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_4) : "l"(*(const unsigned long long*)&cvt_f32_6), "l"(*(const unsigned long long*)&s_1_4));
        float2 _f32x2_mul_a_19 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1_4), s_1_4), cvt_f32_2_4);
        float2 mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&cvt_f32_1_4));
        float2 mul_1_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_4) : "l"(*(const unsigned long long*)&act_4), "l"(*(const unsigned long long*)&cvt_f32_2_4));
        __nv_bfloat162 bf16x2_5 = __float22bfloat162_rn(make_float2(mul_5.x, mul_5.y));
        __nv_bfloat162 bf16x2_1_4 = __float22bfloat162_rn(make_float2(mul_1_4.x, mul_1_4.y));
        dgate[5] = __as_u32(bf16x2_5);
        dup[5] = __as_u32(bf16x2_1_4);
        float2 cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 cvt_f32_1_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float2 cvt_f32_2_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[6]));
        float2 _f32x2_mul_b_20 = make_float2(-1.442695f, -1.442695f);
        float2 t_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_5) : "l"(*(const unsigned long long*)&cvt_f32_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
        float e0_5 = approx_exp2(t_5.x);
        float e1_5 = approx_exp2(t_5.y);
        float2 nd_5 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_5, e1_5));
        float d0_5 = -nd_5.x;
        float d1_5 = -nd_5.y;
        float r0_5 = approx_rcp(d0_5);
        float r1_5 = approx_rcp(d1_5);
        float2 r_5 = make_float2(r0_5, r1_5);
        float2 s_6 = fma_f32x2_rn_noftz(r_5, fma_f32x2_rn_noftz(nd_5, r_5, make_float2(1.0f, 1.0f)), r_5);
        float2 s_1_5 = s_6;
        float2 act_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_5) : "l"(*(const unsigned long long*)&cvt_f32_7), "l"(*(const unsigned long long*)&s_1_5));
        float2 _f32x2_mul_a_21 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1_5), s_1_5), cvt_f32_2_5);
        float2 mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&cvt_f32_1_5));
        float2 mul_1_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_5) : "l"(*(const unsigned long long*)&act_5), "l"(*(const unsigned long long*)&cvt_f32_2_5));
        __nv_bfloat162 bf16x2_6 = __float22bfloat162_rn(make_float2(mul_6.x, mul_6.y));
        __nv_bfloat162 bf16x2_1_5 = __float22bfloat162_rn(make_float2(mul_1_5.x, mul_1_5.y));
        dgate[6] = __as_u32(bf16x2_6);
        dup[6] = __as_u32(bf16x2_1_5);
        float2 cvt_f32_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 cvt_f32_1_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float2 cvt_f32_2_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[7]));
        float2 _f32x2_mul_b_22 = make_float2(-1.442695f, -1.442695f);
        float2 t_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_6) : "l"(*(const unsigned long long*)&cvt_f32_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_22));
        float e0_6 = approx_exp2(t_6.x);
        float e1_6 = approx_exp2(t_6.y);
        float2 nd_6 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_6, e1_6));
        float d0_6 = -nd_6.x;
        float d1_6 = -nd_6.y;
        float r0_6 = approx_rcp(d0_6);
        float r1_6 = approx_rcp(d1_6);
        float2 r_6 = make_float2(r0_6, r1_6);
        float2 s_7 = fma_f32x2_rn_noftz(r_6, fma_f32x2_rn_noftz(nd_6, r_6, make_float2(1.0f, 1.0f)), r_6);
        float2 s_1_6 = s_7;
        float2 act_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_6) : "l"(*(const unsigned long long*)&cvt_f32_8), "l"(*(const unsigned long long*)&s_1_6));
        float2 _f32x2_mul_a_23 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1_6), s_1_6), cvt_f32_2_6);
        float2 mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&cvt_f32_1_6));
        float2 mul_1_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_6) : "l"(*(const unsigned long long*)&act_6), "l"(*(const unsigned long long*)&cvt_f32_2_6));
        __nv_bfloat162 bf16x2_7 = __float22bfloat162_rn(make_float2(mul_7.x, mul_7.y));
        __nv_bfloat162 bf16x2_1_6 = __float22bfloat162_rn(make_float2(mul_1_6.x, mul_1_6.y));
        dgate[7] = __as_u32(bf16x2_7);
        dup[7] = __as_u32(bf16x2_1_6);
        float2 cvt_f32_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
        float2 cvt_f32_1_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[0]));
        float2 cvt_f32_2_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[0]));
        float2 _f32x2_mul_b_24 = make_float2(-1.442695f, -1.442695f);
        float2 t_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_7) : "l"(*(const unsigned long long*)&cvt_f32_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_24));
        float e0_7 = approx_exp2(t_7.x);
        float e1_7 = approx_exp2(t_7.y);
        float2 nd_7 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_7, e1_7));
        float d0_7 = -nd_7.x;
        float d1_7 = -nd_7.y;
        float r0_7 = approx_rcp(d0_7);
        float r1_7 = approx_rcp(d1_7);
        float2 r_7 = make_float2(r0_7, r1_7);
        float2 s_8 = fma_f32x2_rn_noftz(r_7, fma_f32x2_rn_noftz(nd_7, r_7, make_float2(1.0f, 1.0f)), r_7);
        float2 s_1_7 = s_8;
        float2 act_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_7) : "l"(*(const unsigned long long*)&cvt_f32_9), "l"(*(const unsigned long long*)&s_1_7));
        float2 _f32x2_mul_a_25 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_7, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1_7), s_1_7), cvt_f32_2_7);
        float2 mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_25), "l"(*(const unsigned long long*)&cvt_f32_1_7));
        float2 mul_1_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_7) : "l"(*(const unsigned long long*)&act_7), "l"(*(const unsigned long long*)&cvt_f32_2_7));
        __nv_bfloat162 bf16x2_8 = __float22bfloat162_rn(make_float2(mul_8.x, mul_8.y));
        __nv_bfloat162 bf16x2_1_7 = __float22bfloat162_rn(make_float2(mul_1_7.x, mul_1_7.y));
        dgate_hi[0] = __as_u32(bf16x2_8);
        dup_hi[0] = __as_u32(bf16x2_1_7);
        float2 cvt_f32_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[1]));
        float2 cvt_f32_1_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[1]));
        float2 cvt_f32_2_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[1]));
        float2 _f32x2_mul_b_26 = make_float2(-1.442695f, -1.442695f);
        float2 t_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_8) : "l"(*(const unsigned long long*)&cvt_f32_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_26));
        float e0_8 = approx_exp2(t_8.x);
        float e1_8 = approx_exp2(t_8.y);
        float2 nd_8 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_8, e1_8));
        float d0_8 = -nd_8.x;
        float d1_8 = -nd_8.y;
        float r0_8 = approx_rcp(d0_8);
        float r1_8 = approx_rcp(d1_8);
        float2 r_8 = make_float2(r0_8, r1_8);
        float2 s_9 = fma_f32x2_rn_noftz(r_8, fma_f32x2_rn_noftz(nd_8, r_8, make_float2(1.0f, 1.0f)), r_8);
        float2 s_1_8 = s_9;
        float2 act_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_8) : "l"(*(const unsigned long long*)&cvt_f32_10), "l"(*(const unsigned long long*)&s_1_8));
        float2 _f32x2_mul_a_27 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_8, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1_8), s_1_8), cvt_f32_2_8);
        float2 mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_27), "l"(*(const unsigned long long*)&cvt_f32_1_8));
        float2 mul_1_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_8) : "l"(*(const unsigned long long*)&act_8), "l"(*(const unsigned long long*)&cvt_f32_2_8));
        __nv_bfloat162 bf16x2_9 = __float22bfloat162_rn(make_float2(mul_9.x, mul_9.y));
        __nv_bfloat162 bf16x2_1_8 = __float22bfloat162_rn(make_float2(mul_1_8.x, mul_1_8.y));
        dgate_hi[1] = __as_u32(bf16x2_9);
        dup_hi[1] = __as_u32(bf16x2_1_8);
        float2 cvt_f32_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[2]));
        float2 cvt_f32_1_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[2]));
        float2 cvt_f32_2_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[2]));
        float2 _f32x2_mul_b_28 = make_float2(-1.442695f, -1.442695f);
        float2 t_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_9) : "l"(*(const unsigned long long*)&cvt_f32_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_28));
        float e0_9 = approx_exp2(t_9.x);
        float e1_9 = approx_exp2(t_9.y);
        float2 nd_9 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_9, e1_9));
        float d0_9 = -nd_9.x;
        float d1_9 = -nd_9.y;
        float r0_9 = approx_rcp(d0_9);
        float r1_9 = approx_rcp(d1_9);
        float2 r_9 = make_float2(r0_9, r1_9);
        float2 s_10 = fma_f32x2_rn_noftz(r_9, fma_f32x2_rn_noftz(nd_9, r_9, make_float2(1.0f, 1.0f)), r_9);
        float2 s_1_9 = s_10;
        float2 act_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_9) : "l"(*(const unsigned long long*)&cvt_f32_11), "l"(*(const unsigned long long*)&s_1_9));
        float2 _f32x2_mul_a_29 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_9, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1_9), s_1_9), cvt_f32_2_9);
        float2 mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&cvt_f32_1_9));
        float2 mul_1_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_9) : "l"(*(const unsigned long long*)&act_9), "l"(*(const unsigned long long*)&cvt_f32_2_9));
        __nv_bfloat162 bf16x2_10 = __float22bfloat162_rn(make_float2(mul_10.x, mul_10.y));
        __nv_bfloat162 bf16x2_1_9 = __float22bfloat162_rn(make_float2(mul_1_9.x, mul_1_9.y));
        dgate_hi[2] = __as_u32(bf16x2_10);
        dup_hi[2] = __as_u32(bf16x2_1_9);
        float2 cvt_f32_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[3]));
        float2 cvt_f32_1_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[3]));
        float2 cvt_f32_2_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[3]));
        float2 _f32x2_mul_b_30 = make_float2(-1.442695f, -1.442695f);
        float2 t_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_10) : "l"(*(const unsigned long long*)&cvt_f32_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_30));
        float e0_10 = approx_exp2(t_10.x);
        float e1_10 = approx_exp2(t_10.y);
        float2 nd_10 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_10, e1_10));
        float d0_10 = -nd_10.x;
        float d1_10 = -nd_10.y;
        float r0_10 = approx_rcp(d0_10);
        float r1_10 = approx_rcp(d1_10);
        float2 r_10 = make_float2(r0_10, r1_10);
        float2 s_11 = fma_f32x2_rn_noftz(r_10, fma_f32x2_rn_noftz(nd_10, r_10, make_float2(1.0f, 1.0f)), r_10);
        float2 s_1_10 = s_11;
        float2 act_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_10) : "l"(*(const unsigned long long*)&cvt_f32_12), "l"(*(const unsigned long long*)&s_1_10));
        float2 _f32x2_mul_a_31 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_10, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1_10), s_1_10), cvt_f32_2_10);
        float2 mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_31), "l"(*(const unsigned long long*)&cvt_f32_1_10));
        float2 mul_1_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_10) : "l"(*(const unsigned long long*)&act_10), "l"(*(const unsigned long long*)&cvt_f32_2_10));
        __nv_bfloat162 bf16x2_11 = __float22bfloat162_rn(make_float2(mul_11.x, mul_11.y));
        __nv_bfloat162 bf16x2_1_10 = __float22bfloat162_rn(make_float2(mul_1_10.x, mul_1_10.y));
        dgate_hi[3] = __as_u32(bf16x2_11);
        dup_hi[3] = __as_u32(bf16x2_1_10);
        float2 cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[4]));
        float2 cvt_f32_1_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[4]));
        float2 cvt_f32_2_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[4]));
        float2 _f32x2_mul_b_32 = make_float2(-1.442695f, -1.442695f);
        float2 t_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_11) : "l"(*(const unsigned long long*)&cvt_f32_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_32));
        float e0_11 = approx_exp2(t_11.x);
        float e1_11 = approx_exp2(t_11.y);
        float2 nd_11 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_11, e1_11));
        float d0_11 = -nd_11.x;
        float d1_11 = -nd_11.y;
        float r0_11 = approx_rcp(d0_11);
        float r1_11 = approx_rcp(d1_11);
        float2 r_11 = make_float2(r0_11, r1_11);
        float2 s_12 = fma_f32x2_rn_noftz(r_11, fma_f32x2_rn_noftz(nd_11, r_11, make_float2(1.0f, 1.0f)), r_11);
        float2 s_1_11 = s_12;
        float2 act_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_11) : "l"(*(const unsigned long long*)&cvt_f32_13), "l"(*(const unsigned long long*)&s_1_11));
        float2 _f32x2_mul_a_33 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_11, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1_11), s_1_11), cvt_f32_2_11);
        float2 mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&cvt_f32_1_11));
        float2 mul_1_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_11) : "l"(*(const unsigned long long*)&act_11), "l"(*(const unsigned long long*)&cvt_f32_2_11));
        __nv_bfloat162 bf16x2_12 = __float22bfloat162_rn(make_float2(mul_12.x, mul_12.y));
        __nv_bfloat162 bf16x2_1_11 = __float22bfloat162_rn(make_float2(mul_1_11.x, mul_1_11.y));
        dgate_hi[4] = __as_u32(bf16x2_12);
        dup_hi[4] = __as_u32(bf16x2_1_11);
        float2 cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[5]));
        float2 cvt_f32_1_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[5]));
        float2 cvt_f32_2_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[5]));
        float2 _f32x2_mul_b_34 = make_float2(-1.442695f, -1.442695f);
        float2 t_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_12) : "l"(*(const unsigned long long*)&cvt_f32_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_34));
        float e0_12 = approx_exp2(t_12.x);
        float e1_12 = approx_exp2(t_12.y);
        float2 nd_12 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_12, e1_12));
        float d0_12 = -nd_12.x;
        float d1_12 = -nd_12.y;
        float r0_12 = approx_rcp(d0_12);
        float r1_12 = approx_rcp(d1_12);
        float2 r_12 = make_float2(r0_12, r1_12);
        float2 s_13 = fma_f32x2_rn_noftz(r_12, fma_f32x2_rn_noftz(nd_12, r_12, make_float2(1.0f, 1.0f)), r_12);
        float2 s_1_12 = s_13;
        float2 act_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_12) : "l"(*(const unsigned long long*)&cvt_f32_14), "l"(*(const unsigned long long*)&s_1_12));
        float2 _f32x2_mul_a_35 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_12, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1_12), s_1_12), cvt_f32_2_12);
        float2 mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&cvt_f32_1_12));
        float2 mul_1_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_12) : "l"(*(const unsigned long long*)&act_12), "l"(*(const unsigned long long*)&cvt_f32_2_12));
        __nv_bfloat162 bf16x2_13 = __float22bfloat162_rn(make_float2(mul_13.x, mul_13.y));
        __nv_bfloat162 bf16x2_1_12 = __float22bfloat162_rn(make_float2(mul_1_12.x, mul_1_12.y));
        dgate_hi[5] = __as_u32(bf16x2_13);
        dup_hi[5] = __as_u32(bf16x2_1_12);
        float2 cvt_f32_15 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[6]));
        float2 cvt_f32_1_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[6]));
        float2 cvt_f32_2_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[6]));
        float2 _f32x2_mul_b_36 = make_float2(-1.442695f, -1.442695f);
        float2 t_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_13) : "l"(*(const unsigned long long*)&cvt_f32_15), "l"(*(const unsigned long long*)&_f32x2_mul_b_36));
        float e0_13 = approx_exp2(t_13.x);
        float e1_13 = approx_exp2(t_13.y);
        float2 nd_13 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_13, e1_13));
        float d0_13 = -nd_13.x;
        float d1_13 = -nd_13.y;
        float r0_13 = approx_rcp(d0_13);
        float r1_13 = approx_rcp(d1_13);
        float2 r_13 = make_float2(r0_13, r1_13);
        float2 s_14 = fma_f32x2_rn_noftz(r_13, fma_f32x2_rn_noftz(nd_13, r_13, make_float2(1.0f, 1.0f)), r_13);
        float2 s_1_13 = s_14;
        float2 act_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_13) : "l"(*(const unsigned long long*)&cvt_f32_15), "l"(*(const unsigned long long*)&s_1_13));
        float2 _f32x2_mul_a_37 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_13, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1_13), s_1_13), cvt_f32_2_13);
        float2 mul_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_37), "l"(*(const unsigned long long*)&cvt_f32_1_13));
        float2 mul_1_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_13) : "l"(*(const unsigned long long*)&act_13), "l"(*(const unsigned long long*)&cvt_f32_2_13));
        __nv_bfloat162 bf16x2_14 = __float22bfloat162_rn(make_float2(mul_14.x, mul_14.y));
        __nv_bfloat162 bf16x2_1_13 = __float22bfloat162_rn(make_float2(mul_1_13.x, mul_1_13.y));
        dgate_hi[6] = __as_u32(bf16x2_14);
        dup_hi[6] = __as_u32(bf16x2_1_13);
        float2 cvt_f32_16 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[7]));
        float2 cvt_f32_1_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[7]));
        float2 cvt_f32_2_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[7]));
        float2 _f32x2_mul_b_38 = make_float2(-1.442695f, -1.442695f);
        float2 t_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_14) : "l"(*(const unsigned long long*)&cvt_f32_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_38));
        float e0_14 = approx_exp2(t_14.x);
        float e1_14 = approx_exp2(t_14.y);
        float2 nd_14 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_14, e1_14));
        float d0_14 = -nd_14.x;
        float d1_14 = -nd_14.y;
        float r0_14 = approx_rcp(d0_14);
        float r1_14 = approx_rcp(d1_14);
        float2 r_14 = make_float2(r0_14, r1_14);
        float2 s_15 = fma_f32x2_rn_noftz(r_14, fma_f32x2_rn_noftz(nd_14, r_14, make_float2(1.0f, 1.0f)), r_14);
        float2 s_1_14 = s_15;
        float2 act_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_14) : "l"(*(const unsigned long long*)&cvt_f32_16), "l"(*(const unsigned long long*)&s_1_14));
        float2 _f32x2_mul_a_39 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_14, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_1_14), s_1_14), cvt_f32_2_14);
        float2 mul_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_39), "l"(*(const unsigned long long*)&cvt_f32_1_14));
        float2 mul_1_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_14) : "l"(*(const unsigned long long*)&act_14), "l"(*(const unsigned long long*)&cvt_f32_2_14));
        __nv_bfloat162 bf16x2_15 = __float22bfloat162_rn(make_float2(mul_15.x, mul_15.y));
        __nv_bfloat162 bf16x2_1_14 = __float22bfloat162_rn(make_float2(mul_1_14.x, mul_1_14.y));
        dgate_hi[7] = __as_u32(bf16x2_15);
        dup_hi[7] = __as_u32(bf16x2_1_14);
    } else {
        int band = 0;
        float2 cvt_f32_17 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 cvt_f32_1_15 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 cvt_f32_2_15 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float2 _f32x2_mul_b_40 = make_float2(-1.442695f, -1.442695f);
        float2 t_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_15) : "l"(*(const unsigned long long*)&cvt_f32_17), "l"(*(const unsigned long long*)&_f32x2_mul_b_40));
        float e0_15 = approx_exp2(t_15.x);
        float e1_15 = approx_exp2(t_15.y);
        float2 nd_15 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_15, e1_15));
        float d0_15 = -nd_15.x;
        float d1_15 = -nd_15.y;
        float r0_15 = approx_rcp(d0_15);
        float r1_15 = approx_rcp(d1_15);
        float2 r_15 = make_float2(r0_15, r1_15);
        float2 s_16 = fma_f32x2_rn_noftz(r_15, fma_f32x2_rn_noftz(nd_15, r_15, make_float2(1.0f, 1.0f)), r_15);
        float t0 = s_16.x;
        float t1 = s_16.y;
        band = band | ((fminf(t0, t1) == 0.0f) ? 1 : 0);
        float2 s_16_1 = make_float2(max_noftz(t0, 0.0f), max_noftz(t1, 0.0f));
        float2 act_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_15) : "l"(*(const unsigned long long*)&cvt_f32_17), "l"(*(const unsigned long long*)&s_16_1));
        float2 _f32x2_mul_a_41 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_15, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_16_1), s_16_1), cvt_f32_2_15);
        float2 mul_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_41), "l"(*(const unsigned long long*)&cvt_f32_1_15));
        float2 mul_1_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_15) : "l"(*(const unsigned long long*)&act_15), "l"(*(const unsigned long long*)&cvt_f32_2_15));
        __nv_bfloat162 bf16x2_16 = __float22bfloat162_rn(make_float2(mul_16.x, mul_16.y));
        __nv_bfloat162 bf16x2_1_15 = __float22bfloat162_rn(make_float2(mul_1_15.x, mul_1_15.y));
        dgate[0] = __as_u32(bf16x2_16);
        dup[0] = __as_u32(bf16x2_1_15);
        float2 cvt_f32_17_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 cvt_f32_1_15_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 cvt_f32_2_15_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float2 _f32x2_mul_b_42 = make_float2(-1.442695f, -1.442695f);
        float2 t_15_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_15_1) : "l"(*(const unsigned long long*)&cvt_f32_17_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_42));
        float e0_15_1 = approx_exp2(t_15_1.x);
        float e1_15_1 = approx_exp2(t_15_1.y);
        float2 nd_15_1 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_15_1, e1_15_1));
        float d0_15_1 = -nd_15_1.x;
        float d1_15_1 = -nd_15_1.y;
        float r0_15_1 = approx_rcp(d0_15_1);
        float r1_15_1 = approx_rcp(d1_15_1);
        float2 r_15_1 = make_float2(r0_15_1, r1_15_1);
        float2 s_17 = fma_f32x2_rn_noftz(r_15_1, fma_f32x2_rn_noftz(nd_15_1, r_15_1, make_float2(1.0f, 1.0f)), r_15_1);
        float t0_0 = s_17.x;
        float t1_0 = s_17.y;
        band = band | ((fminf(t0_0, t1_0) == 0.0f) ? 2 : 0);
        float2 s_18 = make_float2(max_noftz(t0_0, 0.0f), max_noftz(t1_0, 0.0f));
        float2 act_15_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_15_1) : "l"(*(const unsigned long long*)&cvt_f32_17_1), "l"(*(const unsigned long long*)&s_18));
        float2 _f32x2_mul_a_43 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_15_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_18), s_18), cvt_f32_2_15_1);
        float2 mul_16_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_16_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_43), "l"(*(const unsigned long long*)&cvt_f32_1_15_1));
        float2 mul_1_15_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_15_1) : "l"(*(const unsigned long long*)&act_15_1), "l"(*(const unsigned long long*)&cvt_f32_2_15_1));
        __nv_bfloat162 bf16x2_16_1 = __float22bfloat162_rn(make_float2(mul_16_1.x, mul_16_1.y));
        __nv_bfloat162 bf16x2_1_15_1 = __float22bfloat162_rn(make_float2(mul_1_15_1.x, mul_1_15_1.y));
        dgate[1] = __as_u32(bf16x2_16_1);
        dup[1] = __as_u32(bf16x2_1_15_1);
        float2 cvt_f32_18 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 cvt_f32_1_16 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 cvt_f32_2_16 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float2 _f32x2_mul_b_44 = make_float2(-1.442695f, -1.442695f);
        float2 t_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_16) : "l"(*(const unsigned long long*)&cvt_f32_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_44));
        float e0_16 = approx_exp2(t_16.x);
        float e1_16 = approx_exp2(t_16.y);
        float2 nd_16 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_16, e1_16));
        float d0_16 = -nd_16.x;
        float d1_16 = -nd_16.y;
        float r0_16 = approx_rcp(d0_16);
        float r1_16 = approx_rcp(d1_16);
        float2 r_16 = make_float2(r0_16, r1_16);
        float2 s_19 = fma_f32x2_rn_noftz(r_16, fma_f32x2_rn_noftz(nd_16, r_16, make_float2(1.0f, 1.0f)), r_16);
        float t0_1 = s_19.x;
        float t1_1 = s_19.y;
        band = band | ((fminf(t0_1, t1_1) == 0.0f) ? 4 : 0);
        float2 s_20 = make_float2(max_noftz(t0_1, 0.0f), max_noftz(t1_1, 0.0f));
        float2 act_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_16) : "l"(*(const unsigned long long*)&cvt_f32_18), "l"(*(const unsigned long long*)&s_20));
        float2 _f32x2_mul_a_45 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_16, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_20), s_20), cvt_f32_2_16);
        float2 mul_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_45), "l"(*(const unsigned long long*)&cvt_f32_1_16));
        float2 mul_1_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_16) : "l"(*(const unsigned long long*)&act_16), "l"(*(const unsigned long long*)&cvt_f32_2_16));
        __nv_bfloat162 bf16x2_17 = __float22bfloat162_rn(make_float2(mul_17.x, mul_17.y));
        __nv_bfloat162 bf16x2_1_16 = __float22bfloat162_rn(make_float2(mul_1_16.x, mul_1_16.y));
        dgate[2] = __as_u32(bf16x2_17);
        dup[2] = __as_u32(bf16x2_1_16);
        float2 cvt_f32_19 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 cvt_f32_1_17 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 cvt_f32_2_17 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float2 _f32x2_mul_b_46 = make_float2(-1.442695f, -1.442695f);
        float2 t_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_17) : "l"(*(const unsigned long long*)&cvt_f32_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_46));
        float e0_17 = approx_exp2(t_17.x);
        float e1_17 = approx_exp2(t_17.y);
        float2 nd_17 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_17, e1_17));
        float d0_17 = -nd_17.x;
        float d1_17 = -nd_17.y;
        float r0_17 = approx_rcp(d0_17);
        float r1_17 = approx_rcp(d1_17);
        float2 r_17 = make_float2(r0_17, r1_17);
        float2 s_21 = fma_f32x2_rn_noftz(r_17, fma_f32x2_rn_noftz(nd_17, r_17, make_float2(1.0f, 1.0f)), r_17);
        float t0_2 = s_21.x;
        float t1_2 = s_21.y;
        band = band | ((fminf(t0_2, t1_2) == 0.0f) ? 8 : 0);
        float2 s_22 = make_float2(max_noftz(t0_2, 0.0f), max_noftz(t1_2, 0.0f));
        float2 act_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_17) : "l"(*(const unsigned long long*)&cvt_f32_19), "l"(*(const unsigned long long*)&s_22));
        float2 _f32x2_mul_a_47 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_17, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_22), s_22), cvt_f32_2_17);
        float2 mul_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_47), "l"(*(const unsigned long long*)&cvt_f32_1_17));
        float2 mul_1_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_17) : "l"(*(const unsigned long long*)&act_17), "l"(*(const unsigned long long*)&cvt_f32_2_17));
        __nv_bfloat162 bf16x2_18 = __float22bfloat162_rn(make_float2(mul_18.x, mul_18.y));
        __nv_bfloat162 bf16x2_1_17 = __float22bfloat162_rn(make_float2(mul_1_17.x, mul_1_17.y));
        dgate[3] = __as_u32(bf16x2_18);
        dup[3] = __as_u32(bf16x2_1_17);
        float2 cvt_f32_20 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 cvt_f32_1_18 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float2 cvt_f32_2_18 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[4]));
        float2 _f32x2_mul_b_48 = make_float2(-1.442695f, -1.442695f);
        float2 t_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_18) : "l"(*(const unsigned long long*)&cvt_f32_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_48));
        float e0_18 = approx_exp2(t_18.x);
        float e1_18 = approx_exp2(t_18.y);
        float2 nd_18 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_18, e1_18));
        float d0_18 = -nd_18.x;
        float d1_18 = -nd_18.y;
        float r0_18 = approx_rcp(d0_18);
        float r1_18 = approx_rcp(d1_18);
        float2 r_18 = make_float2(r0_18, r1_18);
        float2 s_23 = fma_f32x2_rn_noftz(r_18, fma_f32x2_rn_noftz(nd_18, r_18, make_float2(1.0f, 1.0f)), r_18);
        float t0_3 = s_23.x;
        float t1_3 = s_23.y;
        band = band | ((fminf(t0_3, t1_3) == 0.0f) ? 16 : 0);
        float2 s_24 = make_float2(max_noftz(t0_3, 0.0f), max_noftz(t1_3, 0.0f));
        float2 act_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_18) : "l"(*(const unsigned long long*)&cvt_f32_20), "l"(*(const unsigned long long*)&s_24));
        float2 _f32x2_mul_a_49 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_18, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_24), s_24), cvt_f32_2_18);
        float2 mul_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_49), "l"(*(const unsigned long long*)&cvt_f32_1_18));
        float2 mul_1_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_18) : "l"(*(const unsigned long long*)&act_18), "l"(*(const unsigned long long*)&cvt_f32_2_18));
        __nv_bfloat162 bf16x2_19 = __float22bfloat162_rn(make_float2(mul_19.x, mul_19.y));
        __nv_bfloat162 bf16x2_1_18 = __float22bfloat162_rn(make_float2(mul_1_18.x, mul_1_18.y));
        dgate[4] = __as_u32(bf16x2_19);
        dup[4] = __as_u32(bf16x2_1_18);
        float2 cvt_f32_21 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 cvt_f32_1_19 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float2 cvt_f32_2_19 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[5]));
        float2 _f32x2_mul_b_50 = make_float2(-1.442695f, -1.442695f);
        float2 t_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_19) : "l"(*(const unsigned long long*)&cvt_f32_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_50));
        float e0_19 = approx_exp2(t_19.x);
        float e1_19 = approx_exp2(t_19.y);
        float2 nd_19 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_19, e1_19));
        float d0_19 = -nd_19.x;
        float d1_19 = -nd_19.y;
        float r0_19 = approx_rcp(d0_19);
        float r1_19 = approx_rcp(d1_19);
        float2 r_19 = make_float2(r0_19, r1_19);
        float2 s_25 = fma_f32x2_rn_noftz(r_19, fma_f32x2_rn_noftz(nd_19, r_19, make_float2(1.0f, 1.0f)), r_19);
        float t0_4 = s_25.x;
        float t1_4 = s_25.y;
        band = band | ((fminf(t0_4, t1_4) == 0.0f) ? 32 : 0);
        float2 s_26 = make_float2(max_noftz(t0_4, 0.0f), max_noftz(t1_4, 0.0f));
        float2 act_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_19) : "l"(*(const unsigned long long*)&cvt_f32_21), "l"(*(const unsigned long long*)&s_26));
        float2 _f32x2_mul_a_51 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_19, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_26), s_26), cvt_f32_2_19);
        float2 mul_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_51), "l"(*(const unsigned long long*)&cvt_f32_1_19));
        float2 mul_1_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_19) : "l"(*(const unsigned long long*)&act_19), "l"(*(const unsigned long long*)&cvt_f32_2_19));
        __nv_bfloat162 bf16x2_20 = __float22bfloat162_rn(make_float2(mul_20.x, mul_20.y));
        __nv_bfloat162 bf16x2_1_19 = __float22bfloat162_rn(make_float2(mul_1_19.x, mul_1_19.y));
        dgate[5] = __as_u32(bf16x2_20);
        dup[5] = __as_u32(bf16x2_1_19);
        float2 cvt_f32_22 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 cvt_f32_1_20 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float2 cvt_f32_2_20 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[6]));
        float2 _f32x2_mul_b_52 = make_float2(-1.442695f, -1.442695f);
        float2 t_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_20) : "l"(*(const unsigned long long*)&cvt_f32_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_52));
        float e0_20 = approx_exp2(t_20.x);
        float e1_20 = approx_exp2(t_20.y);
        float2 nd_20 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_20, e1_20));
        float d0_20 = -nd_20.x;
        float d1_20 = -nd_20.y;
        float r0_20 = approx_rcp(d0_20);
        float r1_20 = approx_rcp(d1_20);
        float2 r_20 = make_float2(r0_20, r1_20);
        float2 s_27 = fma_f32x2_rn_noftz(r_20, fma_f32x2_rn_noftz(nd_20, r_20, make_float2(1.0f, 1.0f)), r_20);
        float t0_5 = s_27.x;
        float t1_5 = s_27.y;
        band = band | ((fminf(t0_5, t1_5) == 0.0f) ? 64 : 0);
        float2 s_28 = make_float2(max_noftz(t0_5, 0.0f), max_noftz(t1_5, 0.0f));
        float2 act_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_20) : "l"(*(const unsigned long long*)&cvt_f32_22), "l"(*(const unsigned long long*)&s_28));
        float2 _f32x2_mul_a_53 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_20, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_28), s_28), cvt_f32_2_20);
        float2 mul_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_53), "l"(*(const unsigned long long*)&cvt_f32_1_20));
        float2 mul_1_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_20) : "l"(*(const unsigned long long*)&act_20), "l"(*(const unsigned long long*)&cvt_f32_2_20));
        __nv_bfloat162 bf16x2_21 = __float22bfloat162_rn(make_float2(mul_21.x, mul_21.y));
        __nv_bfloat162 bf16x2_1_20 = __float22bfloat162_rn(make_float2(mul_1_20.x, mul_1_20.y));
        dgate[6] = __as_u32(bf16x2_21);
        dup[6] = __as_u32(bf16x2_1_20);
        float2 cvt_f32_23 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 cvt_f32_1_21 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float2 cvt_f32_2_21 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[7]));
        float2 _f32x2_mul_b_54 = make_float2(-1.442695f, -1.442695f);
        float2 t_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_21) : "l"(*(const unsigned long long*)&cvt_f32_23), "l"(*(const unsigned long long*)&_f32x2_mul_b_54));
        float e0_21 = approx_exp2(t_21.x);
        float e1_21 = approx_exp2(t_21.y);
        float2 nd_21 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_21, e1_21));
        float d0_21 = -nd_21.x;
        float d1_21 = -nd_21.y;
        float r0_21 = approx_rcp(d0_21);
        float r1_21 = approx_rcp(d1_21);
        float2 r_21 = make_float2(r0_21, r1_21);
        float2 s_29 = fma_f32x2_rn_noftz(r_21, fma_f32x2_rn_noftz(nd_21, r_21, make_float2(1.0f, 1.0f)), r_21);
        float t0_6 = s_29.x;
        float t1_6 = s_29.y;
        band = band | ((fminf(t0_6, t1_6) == 0.0f) ? 128 : 0);
        float2 s_30 = make_float2(max_noftz(t0_6, 0.0f), max_noftz(t1_6, 0.0f));
        float2 act_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_21) : "l"(*(const unsigned long long*)&cvt_f32_23), "l"(*(const unsigned long long*)&s_30));
        float2 _f32x2_mul_a_55 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_21, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_30), s_30), cvt_f32_2_21);
        float2 mul_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_55), "l"(*(const unsigned long long*)&cvt_f32_1_21));
        float2 mul_1_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_21) : "l"(*(const unsigned long long*)&act_21), "l"(*(const unsigned long long*)&cvt_f32_2_21));
        __nv_bfloat162 bf16x2_22 = __float22bfloat162_rn(make_float2(mul_22.x, mul_22.y));
        __nv_bfloat162 bf16x2_1_21 = __float22bfloat162_rn(make_float2(mul_1_21.x, mul_1_21.y));
        dgate[7] = __as_u32(bf16x2_22);
        dup[7] = __as_u32(bf16x2_1_21);
        int band_hi = 0;
        float2 cvt_f32_24 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
        float2 cvt_f32_1_22 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[0]));
        float2 cvt_f32_2_22 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[0]));
        float2 _f32x2_mul_b_56 = make_float2(-1.442695f, -1.442695f);
        float2 t_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_22) : "l"(*(const unsigned long long*)&cvt_f32_24), "l"(*(const unsigned long long*)&_f32x2_mul_b_56));
        float e0_22 = approx_exp2(t_22.x);
        float e1_22 = approx_exp2(t_22.y);
        float2 nd_22 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_22, e1_22));
        float d0_22 = -nd_22.x;
        float d1_22 = -nd_22.y;
        float r0_22 = approx_rcp(d0_22);
        float r1_22 = approx_rcp(d1_22);
        float2 r_22 = make_float2(r0_22, r1_22);
        float2 s_31 = fma_f32x2_rn_noftz(r_22, fma_f32x2_rn_noftz(nd_22, r_22, make_float2(1.0f, 1.0f)), r_22);
        float t0_7 = s_31.x;
        float t1_7 = s_31.y;
        band_hi = band_hi | ((fminf(t0_7, t1_7) == 0.0f) ? 1 : 0);
        float2 s_32 = make_float2(max_noftz(t0_7, 0.0f), max_noftz(t1_7, 0.0f));
        float2 act_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_22) : "l"(*(const unsigned long long*)&cvt_f32_24), "l"(*(const unsigned long long*)&s_32));
        float2 _f32x2_mul_a_57 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_22, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_32), s_32), cvt_f32_2_22);
        float2 mul_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_57), "l"(*(const unsigned long long*)&cvt_f32_1_22));
        float2 mul_1_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_22) : "l"(*(const unsigned long long*)&act_22), "l"(*(const unsigned long long*)&cvt_f32_2_22));
        __nv_bfloat162 bf16x2_23 = __float22bfloat162_rn(make_float2(mul_23.x, mul_23.y));
        __nv_bfloat162 bf16x2_1_22 = __float22bfloat162_rn(make_float2(mul_1_22.x, mul_1_22.y));
        dgate_hi[0] = __as_u32(bf16x2_23);
        dup_hi[0] = __as_u32(bf16x2_1_22);
        float2 cvt_f32_25 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[1]));
        float2 cvt_f32_1_23 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[1]));
        float2 cvt_f32_2_23 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[1]));
        float2 _f32x2_mul_b_58 = make_float2(-1.442695f, -1.442695f);
        float2 t_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_23) : "l"(*(const unsigned long long*)&cvt_f32_25), "l"(*(const unsigned long long*)&_f32x2_mul_b_58));
        float e0_23 = approx_exp2(t_23.x);
        float e1_23 = approx_exp2(t_23.y);
        float2 nd_23 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_23, e1_23));
        float d0_23 = -nd_23.x;
        float d1_23 = -nd_23.y;
        float r0_23 = approx_rcp(d0_23);
        float r1_23 = approx_rcp(d1_23);
        float2 r_23 = make_float2(r0_23, r1_23);
        float2 s_33 = fma_f32x2_rn_noftz(r_23, fma_f32x2_rn_noftz(nd_23, r_23, make_float2(1.0f, 1.0f)), r_23);
        float t0_8 = s_33.x;
        float t1_8 = s_33.y;
        band_hi = band_hi | ((fminf(t0_8, t1_8) == 0.0f) ? 2 : 0);
        float2 s_34 = make_float2(max_noftz(t0_8, 0.0f), max_noftz(t1_8, 0.0f));
        float2 act_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_23) : "l"(*(const unsigned long long*)&cvt_f32_25), "l"(*(const unsigned long long*)&s_34));
        float2 _f32x2_mul_a_59 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_23, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_34), s_34), cvt_f32_2_23);
        float2 mul_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_59), "l"(*(const unsigned long long*)&cvt_f32_1_23));
        float2 mul_1_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_23) : "l"(*(const unsigned long long*)&act_23), "l"(*(const unsigned long long*)&cvt_f32_2_23));
        __nv_bfloat162 bf16x2_24 = __float22bfloat162_rn(make_float2(mul_24.x, mul_24.y));
        __nv_bfloat162 bf16x2_1_23 = __float22bfloat162_rn(make_float2(mul_1_23.x, mul_1_23.y));
        dgate_hi[1] = __as_u32(bf16x2_24);
        dup_hi[1] = __as_u32(bf16x2_1_23);
        float2 cvt_f32_26 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[2]));
        float2 cvt_f32_1_24 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[2]));
        float2 cvt_f32_2_24 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[2]));
        float2 _f32x2_mul_b_60 = make_float2(-1.442695f, -1.442695f);
        float2 t_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_24) : "l"(*(const unsigned long long*)&cvt_f32_26), "l"(*(const unsigned long long*)&_f32x2_mul_b_60));
        float e0_24 = approx_exp2(t_24.x);
        float e1_24 = approx_exp2(t_24.y);
        float2 nd_24 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_24, e1_24));
        float d0_24 = -nd_24.x;
        float d1_24 = -nd_24.y;
        float r0_24 = approx_rcp(d0_24);
        float r1_24 = approx_rcp(d1_24);
        float2 r_24 = make_float2(r0_24, r1_24);
        float2 s_35 = fma_f32x2_rn_noftz(r_24, fma_f32x2_rn_noftz(nd_24, r_24, make_float2(1.0f, 1.0f)), r_24);
        float t0_9 = s_35.x;
        float t1_9 = s_35.y;
        band_hi = band_hi | ((fminf(t0_9, t1_9) == 0.0f) ? 4 : 0);
        float2 s_36 = make_float2(max_noftz(t0_9, 0.0f), max_noftz(t1_9, 0.0f));
        float2 act_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_24) : "l"(*(const unsigned long long*)&cvt_f32_26), "l"(*(const unsigned long long*)&s_36));
        float2 _f32x2_mul_a_61 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_24, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_36), s_36), cvt_f32_2_24);
        float2 mul_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_61), "l"(*(const unsigned long long*)&cvt_f32_1_24));
        float2 mul_1_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_24) : "l"(*(const unsigned long long*)&act_24), "l"(*(const unsigned long long*)&cvt_f32_2_24));
        __nv_bfloat162 bf16x2_25 = __float22bfloat162_rn(make_float2(mul_25.x, mul_25.y));
        __nv_bfloat162 bf16x2_1_24 = __float22bfloat162_rn(make_float2(mul_1_24.x, mul_1_24.y));
        dgate_hi[2] = __as_u32(bf16x2_25);
        dup_hi[2] = __as_u32(bf16x2_1_24);
        float2 cvt_f32_27 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[3]));
        float2 cvt_f32_1_25 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[3]));
        float2 cvt_f32_2_25 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[3]));
        float2 _f32x2_mul_b_62 = make_float2(-1.442695f, -1.442695f);
        float2 t_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_25) : "l"(*(const unsigned long long*)&cvt_f32_27), "l"(*(const unsigned long long*)&_f32x2_mul_b_62));
        float e0_25 = approx_exp2(t_25.x);
        float e1_25 = approx_exp2(t_25.y);
        float2 nd_25 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_25, e1_25));
        float d0_25 = -nd_25.x;
        float d1_25 = -nd_25.y;
        float r0_25 = approx_rcp(d0_25);
        float r1_25 = approx_rcp(d1_25);
        float2 r_25 = make_float2(r0_25, r1_25);
        float2 s_37 = fma_f32x2_rn_noftz(r_25, fma_f32x2_rn_noftz(nd_25, r_25, make_float2(1.0f, 1.0f)), r_25);
        float t0_10 = s_37.x;
        float t1_10 = s_37.y;
        band_hi = band_hi | ((fminf(t0_10, t1_10) == 0.0f) ? 8 : 0);
        float2 s_38 = make_float2(max_noftz(t0_10, 0.0f), max_noftz(t1_10, 0.0f));
        float2 act_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_25) : "l"(*(const unsigned long long*)&cvt_f32_27), "l"(*(const unsigned long long*)&s_38));
        float2 _f32x2_mul_a_63 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_25, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_38), s_38), cvt_f32_2_25);
        float2 mul_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_63), "l"(*(const unsigned long long*)&cvt_f32_1_25));
        float2 mul_1_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_25) : "l"(*(const unsigned long long*)&act_25), "l"(*(const unsigned long long*)&cvt_f32_2_25));
        __nv_bfloat162 bf16x2_26 = __float22bfloat162_rn(make_float2(mul_26.x, mul_26.y));
        __nv_bfloat162 bf16x2_1_25 = __float22bfloat162_rn(make_float2(mul_1_25.x, mul_1_25.y));
        dgate_hi[3] = __as_u32(bf16x2_26);
        dup_hi[3] = __as_u32(bf16x2_1_25);
        float2 cvt_f32_28 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[4]));
        float2 cvt_f32_1_26 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[4]));
        float2 cvt_f32_2_26 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[4]));
        float2 _f32x2_mul_b_64 = make_float2(-1.442695f, -1.442695f);
        float2 t_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_26) : "l"(*(const unsigned long long*)&cvt_f32_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_64));
        float e0_26 = approx_exp2(t_26.x);
        float e1_26 = approx_exp2(t_26.y);
        float2 nd_26 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_26, e1_26));
        float d0_26 = -nd_26.x;
        float d1_26 = -nd_26.y;
        float r0_26 = approx_rcp(d0_26);
        float r1_26 = approx_rcp(d1_26);
        float2 r_26 = make_float2(r0_26, r1_26);
        float2 s_39 = fma_f32x2_rn_noftz(r_26, fma_f32x2_rn_noftz(nd_26, r_26, make_float2(1.0f, 1.0f)), r_26);
        float t0_11 = s_39.x;
        float t1_11 = s_39.y;
        band_hi = band_hi | ((fminf(t0_11, t1_11) == 0.0f) ? 16 : 0);
        float2 s_40 = make_float2(max_noftz(t0_11, 0.0f), max_noftz(t1_11, 0.0f));
        float2 act_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_26) : "l"(*(const unsigned long long*)&cvt_f32_28), "l"(*(const unsigned long long*)&s_40));
        float2 _f32x2_mul_a_65 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_26, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_40), s_40), cvt_f32_2_26);
        float2 mul_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_65), "l"(*(const unsigned long long*)&cvt_f32_1_26));
        float2 mul_1_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_26) : "l"(*(const unsigned long long*)&act_26), "l"(*(const unsigned long long*)&cvt_f32_2_26));
        __nv_bfloat162 bf16x2_27 = __float22bfloat162_rn(make_float2(mul_27.x, mul_27.y));
        __nv_bfloat162 bf16x2_1_26 = __float22bfloat162_rn(make_float2(mul_1_26.x, mul_1_26.y));
        dgate_hi[4] = __as_u32(bf16x2_27);
        dup_hi[4] = __as_u32(bf16x2_1_26);
        float2 cvt_f32_29 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[5]));
        float2 cvt_f32_1_27 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[5]));
        float2 cvt_f32_2_27 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[5]));
        float2 _f32x2_mul_b_66 = make_float2(-1.442695f, -1.442695f);
        float2 t_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_27) : "l"(*(const unsigned long long*)&cvt_f32_29), "l"(*(const unsigned long long*)&_f32x2_mul_b_66));
        float e0_27 = approx_exp2(t_27.x);
        float e1_27 = approx_exp2(t_27.y);
        float2 nd_27 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_27, e1_27));
        float d0_27 = -nd_27.x;
        float d1_27 = -nd_27.y;
        float r0_27 = approx_rcp(d0_27);
        float r1_27 = approx_rcp(d1_27);
        float2 r_27 = make_float2(r0_27, r1_27);
        float2 s_41 = fma_f32x2_rn_noftz(r_27, fma_f32x2_rn_noftz(nd_27, r_27, make_float2(1.0f, 1.0f)), r_27);
        float t0_12 = s_41.x;
        float t1_12 = s_41.y;
        band_hi = band_hi | ((fminf(t0_12, t1_12) == 0.0f) ? 32 : 0);
        float2 s_42 = make_float2(max_noftz(t0_12, 0.0f), max_noftz(t1_12, 0.0f));
        float2 act_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_27) : "l"(*(const unsigned long long*)&cvt_f32_29), "l"(*(const unsigned long long*)&s_42));
        float2 _f32x2_mul_a_67 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_27, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_42), s_42), cvt_f32_2_27);
        float2 mul_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_67), "l"(*(const unsigned long long*)&cvt_f32_1_27));
        float2 mul_1_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_27) : "l"(*(const unsigned long long*)&act_27), "l"(*(const unsigned long long*)&cvt_f32_2_27));
        __nv_bfloat162 bf16x2_28 = __float22bfloat162_rn(make_float2(mul_28.x, mul_28.y));
        __nv_bfloat162 bf16x2_1_27 = __float22bfloat162_rn(make_float2(mul_1_27.x, mul_1_27.y));
        dgate_hi[5] = __as_u32(bf16x2_28);
        dup_hi[5] = __as_u32(bf16x2_1_27);
        float2 cvt_f32_30 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[6]));
        float2 cvt_f32_1_28 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[6]));
        float2 cvt_f32_2_28 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[6]));
        float2 _f32x2_mul_b_68 = make_float2(-1.442695f, -1.442695f);
        float2 t_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_28) : "l"(*(const unsigned long long*)&cvt_f32_30), "l"(*(const unsigned long long*)&_f32x2_mul_b_68));
        float e0_28 = approx_exp2(t_28.x);
        float e1_28 = approx_exp2(t_28.y);
        float2 nd_28 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_28, e1_28));
        float d0_28 = -nd_28.x;
        float d1_28 = -nd_28.y;
        float r0_28 = approx_rcp(d0_28);
        float r1_28 = approx_rcp(d1_28);
        float2 r_28 = make_float2(r0_28, r1_28);
        float2 s_43 = fma_f32x2_rn_noftz(r_28, fma_f32x2_rn_noftz(nd_28, r_28, make_float2(1.0f, 1.0f)), r_28);
        float t0_13 = s_43.x;
        float t1_13 = s_43.y;
        band_hi = band_hi | ((fminf(t0_13, t1_13) == 0.0f) ? 64 : 0);
        float2 s_44 = make_float2(max_noftz(t0_13, 0.0f), max_noftz(t1_13, 0.0f));
        float2 act_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_28) : "l"(*(const unsigned long long*)&cvt_f32_30), "l"(*(const unsigned long long*)&s_44));
        float2 _f32x2_mul_a_69 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_28, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_44), s_44), cvt_f32_2_28);
        float2 mul_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_69), "l"(*(const unsigned long long*)&cvt_f32_1_28));
        float2 mul_1_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_28) : "l"(*(const unsigned long long*)&act_28), "l"(*(const unsigned long long*)&cvt_f32_2_28));
        __nv_bfloat162 bf16x2_29 = __float22bfloat162_rn(make_float2(mul_29.x, mul_29.y));
        __nv_bfloat162 bf16x2_1_28 = __float22bfloat162_rn(make_float2(mul_1_28.x, mul_1_28.y));
        dgate_hi[6] = __as_u32(bf16x2_29);
        dup_hi[6] = __as_u32(bf16x2_1_28);
        float2 cvt_f32_31 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[7]));
        float2 cvt_f32_1_29 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[7]));
        float2 cvt_f32_2_29 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[7]));
        float2 _f32x2_mul_b_70 = make_float2(-1.442695f, -1.442695f);
        float2 t_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_29) : "l"(*(const unsigned long long*)&cvt_f32_31), "l"(*(const unsigned long long*)&_f32x2_mul_b_70));
        float e0_29 = approx_exp2(t_29.x);
        float e1_29 = approx_exp2(t_29.y);
        float2 nd_29 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_29, e1_29));
        float d0_29 = -nd_29.x;
        float d1_29 = -nd_29.y;
        float r0_29 = approx_rcp(d0_29);
        float r1_29 = approx_rcp(d1_29);
        float2 r_29 = make_float2(r0_29, r1_29);
        float2 s_45 = fma_f32x2_rn_noftz(r_29, fma_f32x2_rn_noftz(nd_29, r_29, make_float2(1.0f, 1.0f)), r_29);
        float t0_14 = s_45.x;
        float t1_14 = s_45.y;
        band_hi = band_hi | ((fminf(t0_14, t1_14) == 0.0f) ? 128 : 0);
        float2 s_46 = make_float2(max_noftz(t0_14, 0.0f), max_noftz(t1_14, 0.0f));
        float2 act_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_29) : "l"(*(const unsigned long long*)&cvt_f32_31), "l"(*(const unsigned long long*)&s_46));
        float2 _f32x2_mul_a_71 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_29, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_46), s_46), cvt_f32_2_29);
        float2 mul_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_71), "l"(*(const unsigned long long*)&cvt_f32_1_29));
        float2 mul_1_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_29) : "l"(*(const unsigned long long*)&act_29), "l"(*(const unsigned long long*)&cvt_f32_2_29));
        __nv_bfloat162 bf16x2_30 = __float22bfloat162_rn(make_float2(mul_30.x, mul_30.y));
        __nv_bfloat162 bf16x2_1_29 = __float22bfloat162_rn(make_float2(mul_1_29.x, mul_1_29.y));
        dgate_hi[7] = __as_u32(bf16x2_30);
        dup_hi[7] = __as_u32(bf16x2_1_29);
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
                float2 cvt_f32_32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[0]));
                float2 cvt_f32_1_30 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[0]));
                float2 cvt_f32_2_30 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_8[0]));
                float2 _f32x2_mul_b_73 = make_float2(-1.442695f, -1.442695f);
                float2 t_30;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_30) : "l"(*(const unsigned long long*)&cvt_f32_32), "l"(*(const unsigned long long*)&_f32x2_mul_b_73));
                float e0_30 = approx_exp2(t_30.x);
                float e1_30 = approx_exp2(t_30.y);
                float2 nd_30 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_30, e1_30));
                float d0_30 = -nd_30.x;
                float d1_30 = -nd_30.y;
                float r0_30 = approx_rcp(d0_30);
                float r1_30 = approx_rcp(d1_30);
                float2 r_30 = make_float2(r0_30, r1_30);
                float2 s_47 = fma_f32x2_rn_noftz(r_30, fma_f32x2_rn_noftz(nd_30, r_30, make_float2(1.0f, 1.0f)), r_30);
                float t0_15 = s_47.x;
                float t1_15 = s_47.y;
                float d4_1a;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a) : "f"(d0_30), "f"(0.25f));
                float r4_1a = approx_rcp(d4_1a);
                float _fma_75;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(-d4_1a), "f"(r4_1a), "f"(1.0f));
                float _fma_74;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(r4_1a), "f"(_fma_75), "f"(r4_1a));
                float big_1a;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a) : "f"(mul_f32_rn_noftz(_fma_74, 1.2676506e+30f)), "f"(1.4073749e+14f));
                float _add_76;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(big_1a), "f"(8.388608e+06f));
                float whole_1a;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a) : "f"(_add_76), "f"(8.388608e+06f));
                float above_1a;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a) : "f"(-d4_1a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                float _sub_78;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_78) : "f"(big_1a), "f"(whole_1a));
                float _abs_77;
                asm("abs.f32 %0, %1;" : "=f"(_abs_77) : "f"(_sub_78));
                float _add_79;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(big_1a), "f"(0.5f));
                float _sub_80;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_80) : "f"(big_1a), "f"(0.5f));
                float rounded_1a = ((_abs_77 == 0.5f) ? ((above_1a > 0.0f) ? _add_79 : _sub_80) : whole_1a);
                float u0 = ((d0_30 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a, 7.888609e-31f), 1.7763568e-15f) : t0_15);
                float d4_1a_0;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_0) : "f"(d1_30), "f"(0.25f));
                float r4_1a_0 = approx_rcp(d4_1a_0);
                float _fma_82;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(-d4_1a_0), "f"(r4_1a_0), "f"(1.0f));
                float _fma_81;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(r4_1a_0), "f"(_fma_82), "f"(r4_1a_0));
                float big_1a_0;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_0) : "f"(mul_f32_rn_noftz(_fma_81, 1.2676506e+30f)), "f"(1.4073749e+14f));
                float _add_83;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(big_1a_0), "f"(8.388608e+06f));
                float whole_1a_0;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_0) : "f"(_add_83), "f"(8.388608e+06f));
                float above_1a_0;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_0) : "f"(-d4_1a_0), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_0, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                float _sub_85;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_85) : "f"(big_1a_0), "f"(whole_1a_0));
                float _abs_84;
                asm("abs.f32 %0, %1;" : "=f"(_abs_84) : "f"(_sub_85));
                float _add_86;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_86) : "f"(big_1a_0), "f"(0.5f));
                float _sub_87;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_87) : "f"(big_1a_0), "f"(0.5f));
                float rounded_1a_0 = ((_abs_84 == 0.5f) ? ((above_1a_0 > 0.0f) ? _add_86 : _sub_87) : whole_1a_0);
                float u1 = ((d1_30 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_0, 7.888609e-31f), 1.7763568e-15f) : t1_15);
                float2 s_48 = make_float2(max_noftz(u0, 0.0f), max_noftz(u1, 0.0f));
                float2 act_30;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_30) : "l"(*(const unsigned long long*)&cvt_f32_32), "l"(*(const unsigned long long*)&s_48));
                float2 _f32x2_mul_a_88 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_30, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_48), s_48), cvt_f32_2_30);
                float2 mul_31;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_88), "l"(*(const unsigned long long*)&cvt_f32_1_30));
                float2 mul_1_30;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_30) : "l"(*(const unsigned long long*)&act_30), "l"(*(const unsigned long long*)&cvt_f32_2_30));
                __nv_bfloat162 bf16x2_31 = __float22bfloat162_rn(make_float2(mul_31.x, mul_31.y));
                __nv_bfloat162 bf16x2_1_30 = __float22bfloat162_rn(make_float2(mul_1_30.x, mul_1_30.y));
                unsigned int dgate_new = __as_u32(bf16x2_31);
                unsigned int dup_new = __as_u32(bf16x2_1_30);
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
                float2 cvt_f32_33 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_9[0]));
                float2 cvt_f32_1_31 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_10[0]));
                float2 cvt_f32_2_31 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_11[0]));
                float2 _f32x2_mul_b_90 = make_float2(-1.442695f, -1.442695f);
                float2 t_31;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_31) : "l"(*(const unsigned long long*)&cvt_f32_33), "l"(*(const unsigned long long*)&_f32x2_mul_b_90));
                float e0_31 = approx_exp2(t_31.x);
                float e1_31 = approx_exp2(t_31.y);
                float2 nd_31 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_31, e1_31));
                float d0_31 = -nd_31.x;
                float d1_31 = -nd_31.y;
                float r0_31 = approx_rcp(d0_31);
                float r1_31 = approx_rcp(d1_31);
                float2 r_31 = make_float2(r0_31, r1_31);
                float2 s_49 = fma_f32x2_rn_noftz(r_31, fma_f32x2_rn_noftz(nd_31, r_31, make_float2(1.0f, 1.0f)), r_31);
                float t0_16 = s_49.x;
                float t1_16 = s_49.y;
                float d4_1a_1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_1) : "f"(d0_31), "f"(0.25f));
                float r4_1a_1 = approx_rcp(d4_1a_1);
                float _fma_92;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(-d4_1a_1), "f"(r4_1a_1), "f"(1.0f));
                float _fma_91;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(r4_1a_1), "f"(_fma_92), "f"(r4_1a_1));
                float big_1a_1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_1) : "f"(mul_f32_rn_noftz(_fma_91, 1.2676506e+30f)), "f"(1.4073749e+14f));
                float _add_93;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(big_1a_1), "f"(8.388608e+06f));
                float whole_1a_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_1) : "f"(_add_93), "f"(8.388608e+06f));
                float above_1a_1;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_1) : "f"(-d4_1a_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                float _sub_95;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_95) : "f"(big_1a_1), "f"(whole_1a_1));
                float _abs_94;
                asm("abs.f32 %0, %1;" : "=f"(_abs_94) : "f"(_sub_95));
                float _add_96;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_96) : "f"(big_1a_1), "f"(0.5f));
                float _sub_97;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_97) : "f"(big_1a_1), "f"(0.5f));
                float rounded_1a_1 = ((_abs_94 == 0.5f) ? ((above_1a_1 > 0.0f) ? _add_96 : _sub_97) : whole_1a_1);
                float u0_1 = ((d0_31 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_1, 7.888609e-31f), 1.7763568e-15f) : t0_16);
                float d4_1a_1_1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_1a_1_1) : "f"(d1_31), "f"(0.25f));
                float r4_1a_1_1 = approx_rcp(d4_1a_1_1);
                float _fma_99;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(-d4_1a_1_1), "f"(r4_1a_1_1), "f"(1.0f));
                float _fma_98;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(r4_1a_1_1), "f"(_fma_99), "f"(r4_1a_1_1));
                float big_1a_1_1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_1a_1_1) : "f"(mul_f32_rn_noftz(_fma_98, 1.2676506e+30f)), "f"(1.4073749e+14f));
                float _add_100;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_100) : "f"(big_1a_1_1), "f"(8.388608e+06f));
                float whole_1a_1_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_1a_1_1) : "f"(_add_100), "f"(8.388608e+06f));
                float above_1a_1_1;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_1a_1_1) : "f"(-d4_1a_1_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_1a_1_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                float _sub_102;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_102) : "f"(big_1a_1_1), "f"(whole_1a_1_1));
                float _abs_101;
                asm("abs.f32 %0, %1;" : "=f"(_abs_101) : "f"(_sub_102));
                float _add_103;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_103) : "f"(big_1a_1_1), "f"(0.5f));
                float _sub_104;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_104) : "f"(big_1a_1_1), "f"(0.5f));
                float rounded_1a_1_1 = ((_abs_101 == 0.5f) ? ((above_1a_1_1 > 0.0f) ? _add_103 : _sub_104) : whole_1a_1_1);
                float u1_1 = ((d1_31 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_1a_1_1, 7.888609e-31f), 1.7763568e-15f) : t1_16);
                float2 s_50 = make_float2(max_noftz(u0_1, 0.0f), max_noftz(u1_1, 0.0f));
                float2 act_31;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act_31) : "l"(*(const unsigned long long*)&cvt_f32_33), "l"(*(const unsigned long long*)&s_50));
                float2 _f32x2_mul_a_105 = mul_f32x2_noftz(fma_f32x2_rn_noftz(act_31, sub_f32x2_noftz(make_float2(1.0f, 1.0f), s_50), s_50), cvt_f32_2_31);
                float2 mul_32;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_32) : "l"(*(const unsigned long long*)&_f32x2_mul_a_105), "l"(*(const unsigned long long*)&cvt_f32_1_31));
                float2 mul_1_31;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1_31) : "l"(*(const unsigned long long*)&act_31), "l"(*(const unsigned long long*)&cvt_f32_2_31));
                __nv_bfloat162 bf16x2_32 = __float22bfloat162_rn(make_float2(mul_32.x, mul_32.y));
                __nv_bfloat162 bf16x2_1_31 = __float22bfloat162_rn(make_float2(mul_1_31.x, mul_1_31.y));
                unsigned int dgate_new_hi = __as_u32(bf16x2_32);
                unsigned int dup_new_hi = __as_u32(bf16x2_1_31);
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_123;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_123) : "r"(bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_123) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_122;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_122) : "h"(e4m3x2_bf16x2), "h"(_cvt_satfinite_e4m3x2_bf16x2_123));
    words[0] = _mov_b32_pack_u16x2_122;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_125;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_125) : "r"(bf16x2_mul_1_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_125) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_124;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_124) : "h"(e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_125));
    words[1] = _mov_b32_pack_u16x2_124;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_127;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_127) : "r"(bf16x2_mul_1_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_127) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_126;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_126) : "h"(e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_127));
    words[2] = _mov_b32_pack_u16x2_126;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_129;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_129) : "r"(bf16x2_mul_1_2));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_2);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_129) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_128;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_128) : "h"(e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_129));
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_147;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_147) : "r"(bf16x2_mul_1_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_147) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_146;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_146) : "h"(e4m3x2_bf16x2_3), "h"(_cvt_satfinite_e4m3x2_bf16x2_147));
    words_11[0] = _mov_b32_pack_u16x2_146;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_149;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_149) : "r"(bf16x2_mul_1_4));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_4);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_149) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_148;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_148) : "h"(e4m3x2_bf16x2_4), "h"(_cvt_satfinite_e4m3x2_bf16x2_149));
    words_11[1] = _mov_b32_pack_u16x2_148;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_151;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_151) : "r"(bf16x2_mul_1_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_151) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_150;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_150) : "h"(e4m3x2_bf16x2_5), "h"(_cvt_satfinite_e4m3x2_bf16x2_151));
    words_11[2] = _mov_b32_pack_u16x2_150;
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_153;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_153) : "r"(bf16x2_mul_1_6));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_6);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_153) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_152;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_152) : "h"(e4m3x2_bf16x2_6), "h"(_cvt_satfinite_e4m3x2_bf16x2_153));
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
    unsigned int amax_bits_1 = (folded_16 & 65535) << 16;
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
    int scale_index_22 = ((grow_hi >> 7) * scale_cols + (scale_col >> 2)) * 512 + (grow_hi & 31) * 16 + (grow_hi >> 5 & 3) * 4 + (scale_col & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + scale_index_22) + (0)) = (unsigned char)(scale_1);
    }
    unsigned int words_23[4];
    uint32_t bf16x2_mul_8;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_8) : "r"(dgate_hi[0]), "r"(inv2_1));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_7) : "r"(dgate_hi[1]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_171;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_171) : "r"(bf16x2_mul_1_7));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_7);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_171) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_170;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_170) : "h"(e4m3x2_bf16x2_7), "h"(_cvt_satfinite_e4m3x2_bf16x2_171));
    words_23[0] = _mov_b32_pack_u16x2_170;
    uint32_t bf16x2_mul_9;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_9) : "r"(dgate_hi[2]), "r"(inv2_1));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_8) : "r"(dgate_hi[3]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_173;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_173) : "r"(bf16x2_mul_1_8));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_8);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_173) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_172;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_172) : "h"(e4m3x2_bf16x2_8), "h"(_cvt_satfinite_e4m3x2_bf16x2_173));
    words_23[1] = _mov_b32_pack_u16x2_172;
    uint32_t bf16x2_mul_10;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_10) : "r"(dgate_hi[4]), "r"(inv2_1));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_9) : "r"(dgate_hi[5]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_175;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_175) : "r"(bf16x2_mul_1_9));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_9);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_175) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_174;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_174) : "h"(e4m3x2_bf16x2_9), "h"(_cvt_satfinite_e4m3x2_bf16x2_175));
    words_23[2] = _mov_b32_pack_u16x2_174;
    uint32_t bf16x2_mul_11;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_11) : "r"(dgate_hi[6]), "r"(inv2_1));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_10) : "r"(dgate_hi[7]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_177;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_177) : "r"(bf16x2_mul_1_10));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_10);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_177) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_176;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_176) : "h"(e4m3x2_bf16x2_10), "h"(_cvt_satfinite_e4m3x2_bf16x2_177));
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
    unsigned int amax_bits_2 = (folded_28 & 65535) << 16;
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
    int scale_index_34 = ((grow_hi >> 7) * scale_cols + (scale_col + (K >> 5) >> 2)) * 512 + (grow_hi & 31) * 16 + (grow_hi >> 5 & 3) * 4 + (scale_col + (K >> 5) & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + scale_index_34) + (0)) = (unsigned char)(scale_2);
    }
    unsigned int words_35[4];
    uint32_t bf16x2_mul_12;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_12) : "r"(dup_hi[0]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_11) : "r"(dup_hi[1]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_195;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_195) : "r"(bf16x2_mul_1_11));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_11);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_195) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_194;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_194) : "h"(e4m3x2_bf16x2_11), "h"(_cvt_satfinite_e4m3x2_bf16x2_195));
    words_35[0] = _mov_b32_pack_u16x2_194;
    uint32_t bf16x2_mul_13;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_13) : "r"(dup_hi[2]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_12) : "r"(dup_hi[3]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_197;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_197) : "r"(bf16x2_mul_1_12));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_12);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_197) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_196;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_196) : "h"(e4m3x2_bf16x2_12), "h"(_cvt_satfinite_e4m3x2_bf16x2_197));
    words_35[1] = _mov_b32_pack_u16x2_196;
    uint32_t bf16x2_mul_14;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_14) : "r"(dup_hi[4]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_13) : "r"(dup_hi[5]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_199;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_199) : "r"(bf16x2_mul_1_13));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_13);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_199) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_198;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_198) : "h"(e4m3x2_bf16x2_13), "h"(_cvt_satfinite_e4m3x2_bf16x2_199));
    words_35[2] = _mov_b32_pack_u16x2_198;
    uint32_t bf16x2_mul_15;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_15) : "r"(dup_hi[6]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_14) : "r"(dup_hi[7]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_201;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_201) : "r"(bf16x2_mul_1_14));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_14);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_201) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_200;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_200) : "h"(e4m3x2_bf16x2_14), "h"(_cvt_satfinite_e4m3x2_bf16x2_201));
    words_35[3] = _mov_b32_pack_u16x2_200;
    reinterpret_cast<int4*>(row_output + (out_index_hi + K))[0] = reinterpret_cast<int4*>(words_35 + 0)[0];
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
    unsigned int amax_bits_3 = (col_max_bits & 65535) << 16;
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
    unsigned int amax_bits_4 = (col_max_bits >> 16 & 65535) << 16;
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
    int gcol_a = bx * 64 + warp_1 * 16 + (lane_1 >> 2);
    int gcol_b = gcol_a + 8;
    int out_row = by * 64 + (lane_1 & 3) * 16;
    unsigned int words_a[4];
    unsigned int words_b[4];
    uint32_t bf16x2_mul_16;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_16) : "r"(cols[0]), "r"(inv2_3));
    uint16_t e4m3x2_bf16x2_15;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_15) : "r"(bf16x2_mul_16));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_16);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_15) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_15;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_15) : "r"(cols[1]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_235;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_235) : "r"(bf16x2_mul_1_15));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_15);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_235) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_234;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_234) : "h"(e4m3x2_bf16x2_15), "h"(_cvt_satfinite_e4m3x2_bf16x2_235));
    words_a[0] = _mov_b32_pack_u16x2_234;
    uint32_t bf16x2_mul_17;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_17) : "r"(cols[8]), "r"(inv2_4));
    uint16_t e4m3x2_bf16x2_16;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_16) : "r"(bf16x2_mul_17));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_17);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_16) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_16;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_16) : "r"(cols[9]), "r"(inv2_4));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_237;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_237) : "r"(bf16x2_mul_1_16));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_16);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_237) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_236;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_236) : "h"(e4m3x2_bf16x2_16), "h"(_cvt_satfinite_e4m3x2_bf16x2_237));
    words_b[0] = _mov_b32_pack_u16x2_236;
    uint32_t bf16x2_mul_18;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_18) : "r"(cols[2]), "r"(inv2_3));
    uint16_t e4m3x2_bf16x2_17;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_17) : "r"(bf16x2_mul_18));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_18);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_17) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_17;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_17) : "r"(cols[3]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_239;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_239) : "r"(bf16x2_mul_1_17));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_17);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_239) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_238;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_238) : "h"(e4m3x2_bf16x2_17), "h"(_cvt_satfinite_e4m3x2_bf16x2_239));
    words_a[1] = _mov_b32_pack_u16x2_238;
    uint32_t bf16x2_mul_19;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_19) : "r"(cols[10]), "r"(inv2_4));
    uint16_t e4m3x2_bf16x2_18;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_18) : "r"(bf16x2_mul_19));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_19);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_18) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_18;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_18) : "r"(cols[11]), "r"(inv2_4));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_241;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_241) : "r"(bf16x2_mul_1_18));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_18);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_241) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_240;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_240) : "h"(e4m3x2_bf16x2_18), "h"(_cvt_satfinite_e4m3x2_bf16x2_241));
    words_b[1] = _mov_b32_pack_u16x2_240;
    uint32_t bf16x2_mul_20;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_20) : "r"(cols[4]), "r"(inv2_3));
    uint16_t e4m3x2_bf16x2_19;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_19) : "r"(bf16x2_mul_20));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_20);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_19) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_19;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_19) : "r"(cols[5]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_243;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_243) : "r"(bf16x2_mul_1_19));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_19);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_243) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_242;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_242) : "h"(e4m3x2_bf16x2_19), "h"(_cvt_satfinite_e4m3x2_bf16x2_243));
    words_a[2] = _mov_b32_pack_u16x2_242;
    uint32_t bf16x2_mul_21;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_21) : "r"(cols[12]), "r"(inv2_4));
    uint16_t e4m3x2_bf16x2_20;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_20) : "r"(bf16x2_mul_21));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_21);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_20) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_20;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_20) : "r"(cols[13]), "r"(inv2_4));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_245;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_245) : "r"(bf16x2_mul_1_20));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_20);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_245) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_244;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_244) : "h"(e4m3x2_bf16x2_20), "h"(_cvt_satfinite_e4m3x2_bf16x2_245));
    words_b[2] = _mov_b32_pack_u16x2_244;
    uint32_t bf16x2_mul_22;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_22) : "r"(cols[6]), "r"(inv2_3));
    uint16_t e4m3x2_bf16x2_21;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_21) : "r"(bf16x2_mul_22));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_22);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_21) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_21;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_21) : "r"(cols[7]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_247;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_247) : "r"(bf16x2_mul_1_21));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_21);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_247) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_246;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_246) : "h"(e4m3x2_bf16x2_21), "h"(_cvt_satfinite_e4m3x2_bf16x2_247));
    words_a[3] = _mov_b32_pack_u16x2_246;
    uint32_t bf16x2_mul_23;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_23) : "r"(cols[14]), "r"(inv2_4));
    uint16_t e4m3x2_bf16x2_22;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_22) : "r"(bf16x2_mul_23));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_23);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_22) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_22;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_22) : "r"(cols[15]), "r"(inv2_4));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_249;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_249) : "r"(bf16x2_mul_1_22));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_22);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_249) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_248;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_248) : "h"(e4m3x2_bf16x2_22), "h"(_cvt_satfinite_e4m3x2_bf16x2_249));
    words_b[3] = _mov_b32_pack_u16x2_248;
    reinterpret_cast<int4*>(col_output + (gcol_a * M + out_row))[0] = reinterpret_cast<int4*>(words_a + 0)[0];
    reinterpret_cast<int4*>(col_output + (gcol_b * M + out_row))[0] = reinterpret_cast<int4*>(words_b + 0)[0];
    if ((lane_1 & 1) == 0) {
        int row_tiles = M >> 7;
        int rblk = by * 2 + ((lane_1 & 3) >> 1);
        int col_scale_a = ((gcol_a >> 7) * row_tiles + (rblk >> 2)) * 512 + (gcol_a & 31) * 16 + (gcol_a >> 5 & 3) * 4 + (rblk & 3);
        int col_scale_b = ((gcol_b >> 7) * row_tiles + (rblk >> 2)) * 512 + (gcol_b & 31) * 16 + (gcol_b >> 5 & 3) * 4 + (rblk & 3);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_a) + (0)) = (unsigned char)(scale_3);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b) + (0)) = (unsigned char)(scale_4);
    }
    int lane_46 = tid & 31;
    int warp_47 = tid >> 5;
    int k8_48 = lane_46 & 7;
    int srow_49 = (k8_48 >> 1) * 16 + (lane_46 >> 3) * 2 + (k8_48 & 1);
    unsigned int sswizzle_50 = (unsigned int)(srow_49 >> 3 & 6 | srow_49 & 1);
    unsigned int cols_51[16];
    unsigned int strip_chunk0_0 = (unsigned int)(warp_47 * 2);
    unsigned int strip_src0_0 = tiles_addr + 8192 + (unsigned int)(srow_49 * 128) + ((strip_chunk0_0 ^ sswizzle_50) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_51[0]), "=r"(cols_51[1]), "=r"(cols_51[2]), "=r"(cols_51[3])
        : "r"(strip_src0_0)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_51[4]), "=r"(cols_51[5]), "=r"(cols_51[6]), "=r"(cols_51[7])
        : "r"(strip_src0_0 + 1024)
        : "memory");
    unsigned int strip_chunk0_1 = (unsigned int)(warp_47 * 2 + 1);
    unsigned int strip_src0_1 = tiles_addr + 8192 + (unsigned int)(srow_49 * 128) + ((strip_chunk0_1 ^ sswizzle_50) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_51[8]), "=r"(cols_51[9]), "=r"(cols_51[10]), "=r"(cols_51[11])
        : "r"(strip_src0_1)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols_51[12]), "=r"(cols_51[13]), "=r"(cols_51[14]), "=r"(cols_51[15])
        : "r"(strip_src0_1 + 1024)
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
    unsigned int amax_bits_5 = (col_max_bits_64 & 65535) << 16;
    int scale_i32_5 = (int)(amax_bits_5 + 2031616 >> 23) - 8;
    if (scale_i32_5 < 0) {
        scale_i32_5 = 0;
    }
    if ((amax_bits_5 & 2139095040) == 2139095040) {
        scale_i32_5 = 255;
    }
    unsigned int scale_5 = (unsigned int)scale_i32_5;
    unsigned int inv_5 = 254 - scale_5 << 7;
    if (scale_5 == 255) {
        inv_5 = 32704;
    }
    unsigned int inv2_5 = inv_5 | inv_5 << 16;
    unsigned int amax_bits_6 = (col_max_bits_64 >> 16 & 65535) << 16;
    int scale_i32_6 = (int)(amax_bits_6 + 2031616 >> 23) - 8;
    if (scale_i32_6 < 0) {
        scale_i32_6 = 0;
    }
    if ((amax_bits_6 & 2139095040) == 2139095040) {
        scale_i32_6 = 255;
    }
    unsigned int scale_6 = (unsigned int)scale_i32_6;
    unsigned int inv_6 = 254 - scale_6 << 7;
    if (scale_6 == 255) {
        inv_6 = 32704;
    }
    unsigned int inv2_6 = inv_6 | inv_6 << 16;
    int gcol_a_75 = bx * 64 + K + warp_47 * 16 + (lane_46 >> 2);
    int gcol_b_76 = gcol_a_75 + 8;
    int out_row_77 = by * 64 + (lane_46 & 3) * 16;
    unsigned int words_a_78[4];
    unsigned int words_b_79[4];
    uint32_t bf16x2_mul_24;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_24) : "r"(cols_51[0]), "r"(inv2_5));
    uint16_t e4m3x2_bf16x2_23;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_23) : "r"(bf16x2_mul_24));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_24);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_23) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_23;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_23) : "r"(cols_51[1]), "r"(inv2_5));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_283;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_283) : "r"(bf16x2_mul_1_23));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_23);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_283) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_282;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_282) : "h"(e4m3x2_bf16x2_23), "h"(_cvt_satfinite_e4m3x2_bf16x2_283));
    words_a_78[0] = _mov_b32_pack_u16x2_282;
    uint32_t bf16x2_mul_25;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_25) : "r"(cols_51[8]), "r"(inv2_6));
    uint16_t e4m3x2_bf16x2_24;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_24) : "r"(bf16x2_mul_25));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_25);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_24) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_24;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_24) : "r"(cols_51[9]), "r"(inv2_6));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_285;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_285) : "r"(bf16x2_mul_1_24));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_24);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_285) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_284;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_284) : "h"(e4m3x2_bf16x2_24), "h"(_cvt_satfinite_e4m3x2_bf16x2_285));
    words_b_79[0] = _mov_b32_pack_u16x2_284;
    uint32_t bf16x2_mul_26;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_26) : "r"(cols_51[2]), "r"(inv2_5));
    uint16_t e4m3x2_bf16x2_25;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_25) : "r"(bf16x2_mul_26));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_26);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_25) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_25;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_25) : "r"(cols_51[3]), "r"(inv2_5));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_287;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_287) : "r"(bf16x2_mul_1_25));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_25);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_287) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_286;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_286) : "h"(e4m3x2_bf16x2_25), "h"(_cvt_satfinite_e4m3x2_bf16x2_287));
    words_a_78[1] = _mov_b32_pack_u16x2_286;
    uint32_t bf16x2_mul_27;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_27) : "r"(cols_51[10]), "r"(inv2_6));
    uint16_t e4m3x2_bf16x2_26;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_26) : "r"(bf16x2_mul_27));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_27);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_26) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_26;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_26) : "r"(cols_51[11]), "r"(inv2_6));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_289;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_289) : "r"(bf16x2_mul_1_26));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_26);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_289) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_288;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_288) : "h"(e4m3x2_bf16x2_26), "h"(_cvt_satfinite_e4m3x2_bf16x2_289));
    words_b_79[1] = _mov_b32_pack_u16x2_288;
    uint32_t bf16x2_mul_28;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_28) : "r"(cols_51[4]), "r"(inv2_5));
    uint16_t e4m3x2_bf16x2_27;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_27) : "r"(bf16x2_mul_28));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_28);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_27) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_27;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_27) : "r"(cols_51[5]), "r"(inv2_5));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_291;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_291) : "r"(bf16x2_mul_1_27));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_27);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_291) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_290;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_290) : "h"(e4m3x2_bf16x2_27), "h"(_cvt_satfinite_e4m3x2_bf16x2_291));
    words_a_78[2] = _mov_b32_pack_u16x2_290;
    uint32_t bf16x2_mul_29;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_29) : "r"(cols_51[12]), "r"(inv2_6));
    uint16_t e4m3x2_bf16x2_28;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_28) : "r"(bf16x2_mul_29));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_29);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_28) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_28;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_28) : "r"(cols_51[13]), "r"(inv2_6));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_293;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_293) : "r"(bf16x2_mul_1_28));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_28);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_293) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_292;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_292) : "h"(e4m3x2_bf16x2_28), "h"(_cvt_satfinite_e4m3x2_bf16x2_293));
    words_b_79[2] = _mov_b32_pack_u16x2_292;
    uint32_t bf16x2_mul_30;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_30) : "r"(cols_51[6]), "r"(inv2_5));
    uint16_t e4m3x2_bf16x2_29;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_29) : "r"(bf16x2_mul_30));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_30);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_29) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_29;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_29) : "r"(cols_51[7]), "r"(inv2_5));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_295;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_295) : "r"(bf16x2_mul_1_29));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_29);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_295) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_294;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_294) : "h"(e4m3x2_bf16x2_29), "h"(_cvt_satfinite_e4m3x2_bf16x2_295));
    words_a_78[3] = _mov_b32_pack_u16x2_294;
    uint32_t bf16x2_mul_31;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_31) : "r"(cols_51[14]), "r"(inv2_6));
    uint16_t e4m3x2_bf16x2_30;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_30) : "r"(bf16x2_mul_31));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_31);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_30) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_30;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_30) : "r"(cols_51[15]), "r"(inv2_6));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_297;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_297) : "r"(bf16x2_mul_1_30));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_30);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_297) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_296;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_296) : "h"(e4m3x2_bf16x2_30), "h"(_cvt_satfinite_e4m3x2_bf16x2_297));
    words_b_79[3] = _mov_b32_pack_u16x2_296;
    reinterpret_cast<int4*>(col_output + (gcol_a_75 * M + out_row_77))[0] = reinterpret_cast<int4*>(words_a_78 + 0)[0];
    reinterpret_cast<int4*>(col_output + (gcol_b_76 * M + out_row_77))[0] = reinterpret_cast<int4*>(words_b_79 + 0)[0];
    if ((lane_46 & 1) == 0) {
        int row_tiles_1 = M >> 7;
        int rblk_1 = by * 2 + ((lane_46 & 3) >> 1);
        int col_scale_a_1 = ((gcol_a_75 >> 7) * row_tiles_1 + (rblk_1 >> 2)) * 512 + (gcol_a_75 & 31) * 16 + (gcol_a_75 >> 5 & 3) * 4 + (rblk_1 & 3);
        int col_scale_b_1 = ((gcol_b_76 >> 7) * row_tiles_1 + (rblk_1 >> 2)) * 512 + (gcol_b_76 & 31) * 16 + (gcol_b_76 >> 5 & 3) * 4 + (rblk_1 & 3);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_a_1) + (0)) = (unsigned char)(scale_5);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b_1) + (0)) = (unsigned char)(scale_6);
    }
}

} // extern "C"
