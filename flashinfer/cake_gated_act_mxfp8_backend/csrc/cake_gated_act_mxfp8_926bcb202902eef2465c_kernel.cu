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

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_gated_act_mxfp8_926bcb202902eef2465c(__nv_bfloat16* __restrict__ gated_input, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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
        float2 cvt_f32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 _f32x2_mul_b_6 = make_float2(-1.442695f, -1.442695f);
        float2 t;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t) : "l"(*(const unsigned long long*)&cvt_f32), "l"(*(const unsigned long long*)&_f32x2_mul_b_6));
        float e0 = approx_exp2(t.x);
        float e1 = approx_exp2(t.y);
        float2 nd = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0, e1));
        float d0 = -nd.x;
        float d1 = -nd.y;
        float r0 = approx_rcp(d0);
        float r1 = approx_rcp(d1);
        float2 r = make_float2(r0, r1);
        float2 s = fma_f32x2_rn_noftz(r, fma_f32x2_rn_noftz(nd, r, make_float2(1.0f, 1.0f)), r);
        float2 _f32x2_mul_a_7 = mul_f32x2_noftz(cvt_f32, s);
        float2 mul;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul) : "l"(*(const unsigned long long*)&_f32x2_mul_a_7), "l"(*(const unsigned long long*)&cvt_f32_1));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(mul.x, mul.y)));
        float2 cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 cvt_f32_1_0 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 _f32x2_mul_b_8 = make_float2(-1.442695f, -1.442695f);
        float2 t_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_0) : "l"(*(const unsigned long long*)&cvt_f32_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_8));
        float e0_0 = approx_exp2(t_0.x);
        float e1_0 = approx_exp2(t_0.y);
        float2 nd_0 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_0, e1_0));
        float d0_0 = -nd_0.x;
        float d1_0 = -nd_0.y;
        float r0_0 = approx_rcp(d0_0);
        float r1_0 = approx_rcp(d1_0);
        float2 r_0 = make_float2(r0_0, r1_0);
        float2 s_0 = fma_f32x2_rn_noftz(r_0, fma_f32x2_rn_noftz(nd_0, r_0, make_float2(1.0f, 1.0f)), r_0);
        float2 _f32x2_mul_a_9 = mul_f32x2_noftz(cvt_f32_0, s_0);
        float2 mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_9), "l"(*(const unsigned long long*)&cvt_f32_1_0));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_0.x, mul_0.y)));
        float2 cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 cvt_f32_1_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 _f32x2_mul_b_10 = make_float2(-1.442695f, -1.442695f);
        float2 t_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_1) : "l"(*(const unsigned long long*)&cvt_f32_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_10));
        float e0_1 = approx_exp2(t_1.x);
        float e1_1 = approx_exp2(t_1.y);
        float2 nd_1 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_1, e1_1));
        float d0_1 = -nd_1.x;
        float d1_1 = -nd_1.y;
        float r0_1 = approx_rcp(d0_1);
        float r1_1 = approx_rcp(d1_1);
        float2 r_1 = make_float2(r0_1, r1_1);
        float2 s_1 = fma_f32x2_rn_noftz(r_1, fma_f32x2_rn_noftz(nd_1, r_1, make_float2(1.0f, 1.0f)), r_1);
        float2 _f32x2_mul_a_11 = mul_f32x2_noftz(cvt_f32_2, s_1);
        float2 mul_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_11), "l"(*(const unsigned long long*)&cvt_f32_1_1));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_1.x, mul_1.y)));
        float2 cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 cvt_f32_1_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 _f32x2_mul_b_12 = make_float2(-1.442695f, -1.442695f);
        float2 t_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_2) : "l"(*(const unsigned long long*)&cvt_f32_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_12));
        float e0_2 = approx_exp2(t_2.x);
        float e1_2 = approx_exp2(t_2.y);
        float2 nd_2 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_2, e1_2));
        float d0_2 = -nd_2.x;
        float d1_2 = -nd_2.y;
        float r0_2 = approx_rcp(d0_2);
        float r1_2 = approx_rcp(d1_2);
        float2 r_2 = make_float2(r0_2, r1_2);
        float2 s_2 = fma_f32x2_rn_noftz(r_2, fma_f32x2_rn_noftz(nd_2, r_2, make_float2(1.0f, 1.0f)), r_2);
        float2 _f32x2_mul_a_13 = mul_f32x2_noftz(cvt_f32_3, s_2);
        float2 mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_13), "l"(*(const unsigned long long*)&cvt_f32_1_2));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_2.x, mul_2.y)));
        float2 cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 cvt_f32_1_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float2 _f32x2_mul_b_14 = make_float2(-1.442695f, -1.442695f);
        float2 t_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_3) : "l"(*(const unsigned long long*)&cvt_f32_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_14));
        float e0_3 = approx_exp2(t_3.x);
        float e1_3 = approx_exp2(t_3.y);
        float2 nd_3 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_3, e1_3));
        float d0_3 = -nd_3.x;
        float d1_3 = -nd_3.y;
        float r0_3 = approx_rcp(d0_3);
        float r1_3 = approx_rcp(d1_3);
        float2 r_3 = make_float2(r0_3, r1_3);
        float2 s_3 = fma_f32x2_rn_noftz(r_3, fma_f32x2_rn_noftz(nd_3, r_3, make_float2(1.0f, 1.0f)), r_3);
        float2 _f32x2_mul_a_15 = mul_f32x2_noftz(cvt_f32_4, s_3);
        float2 mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&cvt_f32_1_3));
        packed[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_3.x, mul_3.y)));
        float2 cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 cvt_f32_1_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float2 _f32x2_mul_b_16 = make_float2(-1.442695f, -1.442695f);
        float2 t_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_4) : "l"(*(const unsigned long long*)&cvt_f32_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_16));
        float e0_4 = approx_exp2(t_4.x);
        float e1_4 = approx_exp2(t_4.y);
        float2 nd_4 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_4, e1_4));
        float d0_4 = -nd_4.x;
        float d1_4 = -nd_4.y;
        float r0_4 = approx_rcp(d0_4);
        float r1_4 = approx_rcp(d1_4);
        float2 r_4 = make_float2(r0_4, r1_4);
        float2 s_4 = fma_f32x2_rn_noftz(r_4, fma_f32x2_rn_noftz(nd_4, r_4, make_float2(1.0f, 1.0f)), r_4);
        float2 _f32x2_mul_a_17 = mul_f32x2_noftz(cvt_f32_5, s_4);
        float2 mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&cvt_f32_1_4));
        packed[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_4.x, mul_4.y)));
        float2 cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 cvt_f32_1_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float2 _f32x2_mul_b_18 = make_float2(-1.442695f, -1.442695f);
        float2 t_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_5) : "l"(*(const unsigned long long*)&cvt_f32_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_18));
        float e0_5 = approx_exp2(t_5.x);
        float e1_5 = approx_exp2(t_5.y);
        float2 nd_5 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_5, e1_5));
        float d0_5 = -nd_5.x;
        float d1_5 = -nd_5.y;
        float r0_5 = approx_rcp(d0_5);
        float r1_5 = approx_rcp(d1_5);
        float2 r_5 = make_float2(r0_5, r1_5);
        float2 s_5 = fma_f32x2_rn_noftz(r_5, fma_f32x2_rn_noftz(nd_5, r_5, make_float2(1.0f, 1.0f)), r_5);
        float2 _f32x2_mul_a_19 = mul_f32x2_noftz(cvt_f32_6, s_5);
        float2 mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&cvt_f32_1_5));
        packed[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_5.x, mul_5.y)));
        float2 cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 cvt_f32_1_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float2 _f32x2_mul_b_20 = make_float2(-1.442695f, -1.442695f);
        float2 t_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_6) : "l"(*(const unsigned long long*)&cvt_f32_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
        float e0_6 = approx_exp2(t_6.x);
        float e1_6 = approx_exp2(t_6.y);
        float2 nd_6 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_6, e1_6));
        float d0_6 = -nd_6.x;
        float d1_6 = -nd_6.y;
        float r0_6 = approx_rcp(d0_6);
        float r1_6 = approx_rcp(d1_6);
        float2 r_6 = make_float2(r0_6, r1_6);
        float2 s_6 = fma_f32x2_rn_noftz(r_6, fma_f32x2_rn_noftz(nd_6, r_6, make_float2(1.0f, 1.0f)), r_6);
        float2 _f32x2_mul_a_21 = mul_f32x2_noftz(cvt_f32_7, s_6);
        float2 mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&cvt_f32_1_6));
        packed[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_6.x, mul_6.y)));
        float2 cvt_f32_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float2 cvt_f32_1_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
        float2 _f32x2_mul_b_22 = make_float2(-1.442695f, -1.442695f);
        float2 t_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_7) : "l"(*(const unsigned long long*)&cvt_f32_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_22));
        float e0_7 = approx_exp2(t_7.x);
        float e1_7 = approx_exp2(t_7.y);
        float2 nd_7 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_7, e1_7));
        float d0_7 = -nd_7.x;
        float d1_7 = -nd_7.y;
        float r0_7 = approx_rcp(d0_7);
        float r1_7 = approx_rcp(d1_7);
        float2 r_7 = make_float2(r0_7, r1_7);
        float2 s_7 = fma_f32x2_rn_noftz(r_7, fma_f32x2_rn_noftz(nd_7, r_7, make_float2(1.0f, 1.0f)), r_7);
        float2 _f32x2_mul_a_23 = mul_f32x2_noftz(cvt_f32_8, s_7);
        float2 mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&cvt_f32_1_7));
        packed_hi[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_7.x, mul_7.y)));
        float2 cvt_f32_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float2 cvt_f32_1_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[1]));
        float2 _f32x2_mul_b_24 = make_float2(-1.442695f, -1.442695f);
        float2 t_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_8) : "l"(*(const unsigned long long*)&cvt_f32_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_24));
        float e0_8 = approx_exp2(t_8.x);
        float e1_8 = approx_exp2(t_8.y);
        float2 nd_8 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_8, e1_8));
        float d0_8 = -nd_8.x;
        float d1_8 = -nd_8.y;
        float r0_8 = approx_rcp(d0_8);
        float r1_8 = approx_rcp(d1_8);
        float2 r_8 = make_float2(r0_8, r1_8);
        float2 s_8 = fma_f32x2_rn_noftz(r_8, fma_f32x2_rn_noftz(nd_8, r_8, make_float2(1.0f, 1.0f)), r_8);
        float2 _f32x2_mul_a_25 = mul_f32x2_noftz(cvt_f32_9, s_8);
        float2 mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_25), "l"(*(const unsigned long long*)&cvt_f32_1_8));
        packed_hi[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_8.x, mul_8.y)));
        float2 cvt_f32_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float2 cvt_f32_1_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[2]));
        float2 _f32x2_mul_b_26 = make_float2(-1.442695f, -1.442695f);
        float2 t_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_9) : "l"(*(const unsigned long long*)&cvt_f32_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_26));
        float e0_9 = approx_exp2(t_9.x);
        float e1_9 = approx_exp2(t_9.y);
        float2 nd_9 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_9, e1_9));
        float d0_9 = -nd_9.x;
        float d1_9 = -nd_9.y;
        float r0_9 = approx_rcp(d0_9);
        float r1_9 = approx_rcp(d1_9);
        float2 r_9 = make_float2(r0_9, r1_9);
        float2 s_9 = fma_f32x2_rn_noftz(r_9, fma_f32x2_rn_noftz(nd_9, r_9, make_float2(1.0f, 1.0f)), r_9);
        float2 _f32x2_mul_a_27 = mul_f32x2_noftz(cvt_f32_10, s_9);
        float2 mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_27), "l"(*(const unsigned long long*)&cvt_f32_1_9));
        packed_hi[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_9.x, mul_9.y)));
        float2 cvt_f32_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float2 cvt_f32_1_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[3]));
        float2 _f32x2_mul_b_28 = make_float2(-1.442695f, -1.442695f);
        float2 t_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_10) : "l"(*(const unsigned long long*)&cvt_f32_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_28));
        float e0_10 = approx_exp2(t_10.x);
        float e1_10 = approx_exp2(t_10.y);
        float2 nd_10 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_10, e1_10));
        float d0_10 = -nd_10.x;
        float d1_10 = -nd_10.y;
        float r0_10 = approx_rcp(d0_10);
        float r1_10 = approx_rcp(d1_10);
        float2 r_10 = make_float2(r0_10, r1_10);
        float2 s_10 = fma_f32x2_rn_noftz(r_10, fma_f32x2_rn_noftz(nd_10, r_10, make_float2(1.0f, 1.0f)), r_10);
        float2 _f32x2_mul_a_29 = mul_f32x2_noftz(cvt_f32_11, s_10);
        float2 mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&cvt_f32_1_10));
        packed_hi[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_10.x, mul_10.y)));
        float2 cvt_f32_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[4]));
        float2 cvt_f32_1_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[4]));
        float2 _f32x2_mul_b_30 = make_float2(-1.442695f, -1.442695f);
        float2 t_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_11) : "l"(*(const unsigned long long*)&cvt_f32_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_30));
        float e0_11 = approx_exp2(t_11.x);
        float e1_11 = approx_exp2(t_11.y);
        float2 nd_11 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_11, e1_11));
        float d0_11 = -nd_11.x;
        float d1_11 = -nd_11.y;
        float r0_11 = approx_rcp(d0_11);
        float r1_11 = approx_rcp(d1_11);
        float2 r_11 = make_float2(r0_11, r1_11);
        float2 s_11 = fma_f32x2_rn_noftz(r_11, fma_f32x2_rn_noftz(nd_11, r_11, make_float2(1.0f, 1.0f)), r_11);
        float2 _f32x2_mul_a_31 = mul_f32x2_noftz(cvt_f32_12, s_11);
        float2 mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_31), "l"(*(const unsigned long long*)&cvt_f32_1_11));
        packed_hi[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_11.x, mul_11.y)));
        float2 cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[5]));
        float2 cvt_f32_1_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[5]));
        float2 _f32x2_mul_b_32 = make_float2(-1.442695f, -1.442695f);
        float2 t_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_12) : "l"(*(const unsigned long long*)&cvt_f32_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_32));
        float e0_12 = approx_exp2(t_12.x);
        float e1_12 = approx_exp2(t_12.y);
        float2 nd_12 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_12, e1_12));
        float d0_12 = -nd_12.x;
        float d1_12 = -nd_12.y;
        float r0_12 = approx_rcp(d0_12);
        float r1_12 = approx_rcp(d1_12);
        float2 r_12 = make_float2(r0_12, r1_12);
        float2 s_12 = fma_f32x2_rn_noftz(r_12, fma_f32x2_rn_noftz(nd_12, r_12, make_float2(1.0f, 1.0f)), r_12);
        float2 _f32x2_mul_a_33 = mul_f32x2_noftz(cvt_f32_13, s_12);
        float2 mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&cvt_f32_1_12));
        packed_hi[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_12.x, mul_12.y)));
        float2 cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[6]));
        float2 cvt_f32_1_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[6]));
        float2 _f32x2_mul_b_34 = make_float2(-1.442695f, -1.442695f);
        float2 t_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_13) : "l"(*(const unsigned long long*)&cvt_f32_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_34));
        float e0_13 = approx_exp2(t_13.x);
        float e1_13 = approx_exp2(t_13.y);
        float2 nd_13 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_13, e1_13));
        float d0_13 = -nd_13.x;
        float d1_13 = -nd_13.y;
        float r0_13 = approx_rcp(d0_13);
        float r1_13 = approx_rcp(d1_13);
        float2 r_13 = make_float2(r0_13, r1_13);
        float2 s_13 = fma_f32x2_rn_noftz(r_13, fma_f32x2_rn_noftz(nd_13, r_13, make_float2(1.0f, 1.0f)), r_13);
        float2 _f32x2_mul_a_35 = mul_f32x2_noftz(cvt_f32_14, s_13);
        float2 mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&cvt_f32_1_13));
        packed_hi[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_13.x, mul_13.y)));
        float2 cvt_f32_15 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[7]));
        float2 cvt_f32_1_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[7]));
        float2 _f32x2_mul_b_36 = make_float2(-1.442695f, -1.442695f);
        float2 t_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_14) : "l"(*(const unsigned long long*)&cvt_f32_15), "l"(*(const unsigned long long*)&_f32x2_mul_b_36));
        float e0_14 = approx_exp2(t_14.x);
        float e1_14 = approx_exp2(t_14.y);
        float2 nd_14 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_14, e1_14));
        float d0_14 = -nd_14.x;
        float d1_14 = -nd_14.y;
        float r0_14 = approx_rcp(d0_14);
        float r1_14 = approx_rcp(d1_14);
        float2 r_14 = make_float2(r0_14, r1_14);
        float2 s_14 = fma_f32x2_rn_noftz(r_14, fma_f32x2_rn_noftz(nd_14, r_14, make_float2(1.0f, 1.0f)), r_14);
        float2 _f32x2_mul_a_37 = mul_f32x2_noftz(cvt_f32_15, s_14);
        float2 mul_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_37), "l"(*(const unsigned long long*)&cvt_f32_1_14));
        packed_hi[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_14.x, mul_14.y)));
    } else {
        int band = 0;
        float2 cvt_f32_16 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 cvt_f32_1_15 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 _f32x2_mul_b_38 = make_float2(-1.442695f, -1.442695f);
        float2 t_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_15) : "l"(*(const unsigned long long*)&cvt_f32_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_38));
        float e0_15 = approx_exp2(t_15.x);
        float e1_15 = approx_exp2(t_15.y);
        float2 nd_15 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_15, e1_15));
        float d0_15 = -nd_15.x;
        float d1_15 = -nd_15.y;
        float r0_15 = approx_rcp(d0_15);
        float r1_15 = approx_rcp(d1_15);
        float2 r_15 = make_float2(r0_15, r1_15);
        float2 s_15 = fma_f32x2_rn_noftz(r_15, fma_f32x2_rn_noftz(nd_15, r_15, make_float2(1.0f, 1.0f)), r_15);
        float t0 = s_15.x;
        float t1 = s_15.y;
        band = band | ((fminf(t0, t1) == 0.0f) ? 1 : 0);
        float2 _f32x2_mul_a_39 = mul_f32x2_noftz(cvt_f32_16, make_float2(max_noftz(t0, 0.0f), max_noftz(t1, 0.0f)));
        float2 mul_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_39), "l"(*(const unsigned long long*)&cvt_f32_1_15));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_15.x, mul_15.y)));
        float2 cvt_f32_16_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 cvt_f32_1_15_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 _f32x2_mul_b_40 = make_float2(-1.442695f, -1.442695f);
        float2 t_15_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_15_1) : "l"(*(const unsigned long long*)&cvt_f32_16_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_40));
        float e0_15_1 = approx_exp2(t_15_1.x);
        float e1_15_1 = approx_exp2(t_15_1.y);
        float2 nd_15_1 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_15_1, e1_15_1));
        float d0_15_1 = -nd_15_1.x;
        float d1_15_1 = -nd_15_1.y;
        float r0_15_1 = approx_rcp(d0_15_1);
        float r1_15_1 = approx_rcp(d1_15_1);
        float2 r_15_1 = make_float2(r0_15_1, r1_15_1);
        float2 s_15_1 = fma_f32x2_rn_noftz(r_15_1, fma_f32x2_rn_noftz(nd_15_1, r_15_1, make_float2(1.0f, 1.0f)), r_15_1);
        float t0_0 = s_15_1.x;
        float t1_0 = s_15_1.y;
        band = band | ((fminf(t0_0, t1_0) == 0.0f) ? 2 : 0);
        float2 _f32x2_mul_a_41 = mul_f32x2_noftz(cvt_f32_16_1, make_float2(max_noftz(t0_0, 0.0f), max_noftz(t1_0, 0.0f)));
        float2 mul_15_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_15_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_41), "l"(*(const unsigned long long*)&cvt_f32_1_15_1));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_15_1.x, mul_15_1.y)));
        float2 cvt_f32_17 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 cvt_f32_1_16 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 _f32x2_mul_b_42 = make_float2(-1.442695f, -1.442695f);
        float2 t_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_16) : "l"(*(const unsigned long long*)&cvt_f32_17), "l"(*(const unsigned long long*)&_f32x2_mul_b_42));
        float e0_16 = approx_exp2(t_16.x);
        float e1_16 = approx_exp2(t_16.y);
        float2 nd_16 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_16, e1_16));
        float d0_16 = -nd_16.x;
        float d1_16 = -nd_16.y;
        float r0_16 = approx_rcp(d0_16);
        float r1_16 = approx_rcp(d1_16);
        float2 r_16 = make_float2(r0_16, r1_16);
        float2 s_16 = fma_f32x2_rn_noftz(r_16, fma_f32x2_rn_noftz(nd_16, r_16, make_float2(1.0f, 1.0f)), r_16);
        float t0_1 = s_16.x;
        float t1_1 = s_16.y;
        band = band | ((fminf(t0_1, t1_1) == 0.0f) ? 4 : 0);
        float2 _f32x2_mul_a_43 = mul_f32x2_noftz(cvt_f32_17, make_float2(max_noftz(t0_1, 0.0f), max_noftz(t1_1, 0.0f)));
        float2 mul_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_43), "l"(*(const unsigned long long*)&cvt_f32_1_16));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_16.x, mul_16.y)));
        float2 cvt_f32_18 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 cvt_f32_1_17 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 _f32x2_mul_b_44 = make_float2(-1.442695f, -1.442695f);
        float2 t_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_17) : "l"(*(const unsigned long long*)&cvt_f32_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_44));
        float e0_17 = approx_exp2(t_17.x);
        float e1_17 = approx_exp2(t_17.y);
        float2 nd_17 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_17, e1_17));
        float d0_17 = -nd_17.x;
        float d1_17 = -nd_17.y;
        float r0_17 = approx_rcp(d0_17);
        float r1_17 = approx_rcp(d1_17);
        float2 r_17 = make_float2(r0_17, r1_17);
        float2 s_17 = fma_f32x2_rn_noftz(r_17, fma_f32x2_rn_noftz(nd_17, r_17, make_float2(1.0f, 1.0f)), r_17);
        float t0_2 = s_17.x;
        float t1_2 = s_17.y;
        band = band | ((fminf(t0_2, t1_2) == 0.0f) ? 8 : 0);
        float2 _f32x2_mul_a_45 = mul_f32x2_noftz(cvt_f32_18, make_float2(max_noftz(t0_2, 0.0f), max_noftz(t1_2, 0.0f)));
        float2 mul_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_45), "l"(*(const unsigned long long*)&cvt_f32_1_17));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_17.x, mul_17.y)));
        float2 cvt_f32_19 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 cvt_f32_1_18 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float2 _f32x2_mul_b_46 = make_float2(-1.442695f, -1.442695f);
        float2 t_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_18) : "l"(*(const unsigned long long*)&cvt_f32_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_46));
        float e0_18 = approx_exp2(t_18.x);
        float e1_18 = approx_exp2(t_18.y);
        float2 nd_18 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_18, e1_18));
        float d0_18 = -nd_18.x;
        float d1_18 = -nd_18.y;
        float r0_18 = approx_rcp(d0_18);
        float r1_18 = approx_rcp(d1_18);
        float2 r_18 = make_float2(r0_18, r1_18);
        float2 s_18 = fma_f32x2_rn_noftz(r_18, fma_f32x2_rn_noftz(nd_18, r_18, make_float2(1.0f, 1.0f)), r_18);
        float t0_3 = s_18.x;
        float t1_3 = s_18.y;
        band = band | ((fminf(t0_3, t1_3) == 0.0f) ? 16 : 0);
        float2 _f32x2_mul_a_47 = mul_f32x2_noftz(cvt_f32_19, make_float2(max_noftz(t0_3, 0.0f), max_noftz(t1_3, 0.0f)));
        float2 mul_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_47), "l"(*(const unsigned long long*)&cvt_f32_1_18));
        packed[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_18.x, mul_18.y)));
        float2 cvt_f32_20 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 cvt_f32_1_19 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float2 _f32x2_mul_b_48 = make_float2(-1.442695f, -1.442695f);
        float2 t_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_19) : "l"(*(const unsigned long long*)&cvt_f32_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_48));
        float e0_19 = approx_exp2(t_19.x);
        float e1_19 = approx_exp2(t_19.y);
        float2 nd_19 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_19, e1_19));
        float d0_19 = -nd_19.x;
        float d1_19 = -nd_19.y;
        float r0_19 = approx_rcp(d0_19);
        float r1_19 = approx_rcp(d1_19);
        float2 r_19 = make_float2(r0_19, r1_19);
        float2 s_19 = fma_f32x2_rn_noftz(r_19, fma_f32x2_rn_noftz(nd_19, r_19, make_float2(1.0f, 1.0f)), r_19);
        float t0_4 = s_19.x;
        float t1_4 = s_19.y;
        band = band | ((fminf(t0_4, t1_4) == 0.0f) ? 32 : 0);
        float2 _f32x2_mul_a_49 = mul_f32x2_noftz(cvt_f32_20, make_float2(max_noftz(t0_4, 0.0f), max_noftz(t1_4, 0.0f)));
        float2 mul_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_49), "l"(*(const unsigned long long*)&cvt_f32_1_19));
        packed[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_19.x, mul_19.y)));
        float2 cvt_f32_21 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 cvt_f32_1_20 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float2 _f32x2_mul_b_50 = make_float2(-1.442695f, -1.442695f);
        float2 t_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_20) : "l"(*(const unsigned long long*)&cvt_f32_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_50));
        float e0_20 = approx_exp2(t_20.x);
        float e1_20 = approx_exp2(t_20.y);
        float2 nd_20 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_20, e1_20));
        float d0_20 = -nd_20.x;
        float d1_20 = -nd_20.y;
        float r0_20 = approx_rcp(d0_20);
        float r1_20 = approx_rcp(d1_20);
        float2 r_20 = make_float2(r0_20, r1_20);
        float2 s_20 = fma_f32x2_rn_noftz(r_20, fma_f32x2_rn_noftz(nd_20, r_20, make_float2(1.0f, 1.0f)), r_20);
        float t0_5 = s_20.x;
        float t1_5 = s_20.y;
        band = band | ((fminf(t0_5, t1_5) == 0.0f) ? 64 : 0);
        float2 _f32x2_mul_a_51 = mul_f32x2_noftz(cvt_f32_21, make_float2(max_noftz(t0_5, 0.0f), max_noftz(t1_5, 0.0f)));
        float2 mul_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_51), "l"(*(const unsigned long long*)&cvt_f32_1_20));
        packed[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_20.x, mul_20.y)));
        float2 cvt_f32_22 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 cvt_f32_1_21 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float2 _f32x2_mul_b_52 = make_float2(-1.442695f, -1.442695f);
        float2 t_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_21) : "l"(*(const unsigned long long*)&cvt_f32_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_52));
        float e0_21 = approx_exp2(t_21.x);
        float e1_21 = approx_exp2(t_21.y);
        float2 nd_21 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_21, e1_21));
        float d0_21 = -nd_21.x;
        float d1_21 = -nd_21.y;
        float r0_21 = approx_rcp(d0_21);
        float r1_21 = approx_rcp(d1_21);
        float2 r_21 = make_float2(r0_21, r1_21);
        float2 s_21 = fma_f32x2_rn_noftz(r_21, fma_f32x2_rn_noftz(nd_21, r_21, make_float2(1.0f, 1.0f)), r_21);
        float t0_6 = s_21.x;
        float t1_6 = s_21.y;
        band = band | ((fminf(t0_6, t1_6) == 0.0f) ? 128 : 0);
        float2 _f32x2_mul_a_53 = mul_f32x2_noftz(cvt_f32_22, make_float2(max_noftz(t0_6, 0.0f), max_noftz(t1_6, 0.0f)));
        float2 mul_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_53), "l"(*(const unsigned long long*)&cvt_f32_1_21));
        packed[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_21.x, mul_21.y)));
        int band_hi = 0;
        float2 cvt_f32_23 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float2 cvt_f32_1_22 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
        float2 _f32x2_mul_b_54 = make_float2(-1.442695f, -1.442695f);
        float2 t_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_22) : "l"(*(const unsigned long long*)&cvt_f32_23), "l"(*(const unsigned long long*)&_f32x2_mul_b_54));
        float e0_22 = approx_exp2(t_22.x);
        float e1_22 = approx_exp2(t_22.y);
        float2 nd_22 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_22, e1_22));
        float d0_22 = -nd_22.x;
        float d1_22 = -nd_22.y;
        float r0_22 = approx_rcp(d0_22);
        float r1_22 = approx_rcp(d1_22);
        float2 r_22 = make_float2(r0_22, r1_22);
        float2 s_22 = fma_f32x2_rn_noftz(r_22, fma_f32x2_rn_noftz(nd_22, r_22, make_float2(1.0f, 1.0f)), r_22);
        float t0_7 = s_22.x;
        float t1_7 = s_22.y;
        band_hi = band_hi | ((fminf(t0_7, t1_7) == 0.0f) ? 1 : 0);
        float2 _f32x2_mul_a_55 = mul_f32x2_noftz(cvt_f32_23, make_float2(max_noftz(t0_7, 0.0f), max_noftz(t1_7, 0.0f)));
        float2 mul_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_55), "l"(*(const unsigned long long*)&cvt_f32_1_22));
        packed_hi[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_22.x, mul_22.y)));
        float2 cvt_f32_24 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float2 cvt_f32_1_23 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[1]));
        float2 _f32x2_mul_b_56 = make_float2(-1.442695f, -1.442695f);
        float2 t_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_23) : "l"(*(const unsigned long long*)&cvt_f32_24), "l"(*(const unsigned long long*)&_f32x2_mul_b_56));
        float e0_23 = approx_exp2(t_23.x);
        float e1_23 = approx_exp2(t_23.y);
        float2 nd_23 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_23, e1_23));
        float d0_23 = -nd_23.x;
        float d1_23 = -nd_23.y;
        float r0_23 = approx_rcp(d0_23);
        float r1_23 = approx_rcp(d1_23);
        float2 r_23 = make_float2(r0_23, r1_23);
        float2 s_23 = fma_f32x2_rn_noftz(r_23, fma_f32x2_rn_noftz(nd_23, r_23, make_float2(1.0f, 1.0f)), r_23);
        float t0_8 = s_23.x;
        float t1_8 = s_23.y;
        band_hi = band_hi | ((fminf(t0_8, t1_8) == 0.0f) ? 2 : 0);
        float2 _f32x2_mul_a_57 = mul_f32x2_noftz(cvt_f32_24, make_float2(max_noftz(t0_8, 0.0f), max_noftz(t1_8, 0.0f)));
        float2 mul_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_57), "l"(*(const unsigned long long*)&cvt_f32_1_23));
        packed_hi[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_23.x, mul_23.y)));
        float2 cvt_f32_25 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float2 cvt_f32_1_24 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[2]));
        float2 _f32x2_mul_b_58 = make_float2(-1.442695f, -1.442695f);
        float2 t_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_24) : "l"(*(const unsigned long long*)&cvt_f32_25), "l"(*(const unsigned long long*)&_f32x2_mul_b_58));
        float e0_24 = approx_exp2(t_24.x);
        float e1_24 = approx_exp2(t_24.y);
        float2 nd_24 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_24, e1_24));
        float d0_24 = -nd_24.x;
        float d1_24 = -nd_24.y;
        float r0_24 = approx_rcp(d0_24);
        float r1_24 = approx_rcp(d1_24);
        float2 r_24 = make_float2(r0_24, r1_24);
        float2 s_24 = fma_f32x2_rn_noftz(r_24, fma_f32x2_rn_noftz(nd_24, r_24, make_float2(1.0f, 1.0f)), r_24);
        float t0_9 = s_24.x;
        float t1_9 = s_24.y;
        band_hi = band_hi | ((fminf(t0_9, t1_9) == 0.0f) ? 4 : 0);
        float2 _f32x2_mul_a_59 = mul_f32x2_noftz(cvt_f32_25, make_float2(max_noftz(t0_9, 0.0f), max_noftz(t1_9, 0.0f)));
        float2 mul_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_59), "l"(*(const unsigned long long*)&cvt_f32_1_24));
        packed_hi[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_24.x, mul_24.y)));
        float2 cvt_f32_26 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float2 cvt_f32_1_25 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[3]));
        float2 _f32x2_mul_b_60 = make_float2(-1.442695f, -1.442695f);
        float2 t_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_25) : "l"(*(const unsigned long long*)&cvt_f32_26), "l"(*(const unsigned long long*)&_f32x2_mul_b_60));
        float e0_25 = approx_exp2(t_25.x);
        float e1_25 = approx_exp2(t_25.y);
        float2 nd_25 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_25, e1_25));
        float d0_25 = -nd_25.x;
        float d1_25 = -nd_25.y;
        float r0_25 = approx_rcp(d0_25);
        float r1_25 = approx_rcp(d1_25);
        float2 r_25 = make_float2(r0_25, r1_25);
        float2 s_25 = fma_f32x2_rn_noftz(r_25, fma_f32x2_rn_noftz(nd_25, r_25, make_float2(1.0f, 1.0f)), r_25);
        float t0_10 = s_25.x;
        float t1_10 = s_25.y;
        band_hi = band_hi | ((fminf(t0_10, t1_10) == 0.0f) ? 8 : 0);
        float2 _f32x2_mul_a_61 = mul_f32x2_noftz(cvt_f32_26, make_float2(max_noftz(t0_10, 0.0f), max_noftz(t1_10, 0.0f)));
        float2 mul_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_61), "l"(*(const unsigned long long*)&cvt_f32_1_25));
        packed_hi[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_25.x, mul_25.y)));
        float2 cvt_f32_27 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[4]));
        float2 cvt_f32_1_26 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[4]));
        float2 _f32x2_mul_b_62 = make_float2(-1.442695f, -1.442695f);
        float2 t_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_26) : "l"(*(const unsigned long long*)&cvt_f32_27), "l"(*(const unsigned long long*)&_f32x2_mul_b_62));
        float e0_26 = approx_exp2(t_26.x);
        float e1_26 = approx_exp2(t_26.y);
        float2 nd_26 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_26, e1_26));
        float d0_26 = -nd_26.x;
        float d1_26 = -nd_26.y;
        float r0_26 = approx_rcp(d0_26);
        float r1_26 = approx_rcp(d1_26);
        float2 r_26 = make_float2(r0_26, r1_26);
        float2 s_26 = fma_f32x2_rn_noftz(r_26, fma_f32x2_rn_noftz(nd_26, r_26, make_float2(1.0f, 1.0f)), r_26);
        float t0_11 = s_26.x;
        float t1_11 = s_26.y;
        band_hi = band_hi | ((fminf(t0_11, t1_11) == 0.0f) ? 16 : 0);
        float2 _f32x2_mul_a_63 = mul_f32x2_noftz(cvt_f32_27, make_float2(max_noftz(t0_11, 0.0f), max_noftz(t1_11, 0.0f)));
        float2 mul_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_63), "l"(*(const unsigned long long*)&cvt_f32_1_26));
        packed_hi[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_26.x, mul_26.y)));
        float2 cvt_f32_28 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[5]));
        float2 cvt_f32_1_27 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[5]));
        float2 _f32x2_mul_b_64 = make_float2(-1.442695f, -1.442695f);
        float2 t_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_27) : "l"(*(const unsigned long long*)&cvt_f32_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_64));
        float e0_27 = approx_exp2(t_27.x);
        float e1_27 = approx_exp2(t_27.y);
        float2 nd_27 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_27, e1_27));
        float d0_27 = -nd_27.x;
        float d1_27 = -nd_27.y;
        float r0_27 = approx_rcp(d0_27);
        float r1_27 = approx_rcp(d1_27);
        float2 r_27 = make_float2(r0_27, r1_27);
        float2 s_27 = fma_f32x2_rn_noftz(r_27, fma_f32x2_rn_noftz(nd_27, r_27, make_float2(1.0f, 1.0f)), r_27);
        float t0_12 = s_27.x;
        float t1_12 = s_27.y;
        band_hi = band_hi | ((fminf(t0_12, t1_12) == 0.0f) ? 32 : 0);
        float2 _f32x2_mul_a_65 = mul_f32x2_noftz(cvt_f32_28, make_float2(max_noftz(t0_12, 0.0f), max_noftz(t1_12, 0.0f)));
        float2 mul_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_65), "l"(*(const unsigned long long*)&cvt_f32_1_27));
        packed_hi[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_27.x, mul_27.y)));
        float2 cvt_f32_29 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[6]));
        float2 cvt_f32_1_28 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[6]));
        float2 _f32x2_mul_b_66 = make_float2(-1.442695f, -1.442695f);
        float2 t_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_28) : "l"(*(const unsigned long long*)&cvt_f32_29), "l"(*(const unsigned long long*)&_f32x2_mul_b_66));
        float e0_28 = approx_exp2(t_28.x);
        float e1_28 = approx_exp2(t_28.y);
        float2 nd_28 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_28, e1_28));
        float d0_28 = -nd_28.x;
        float d1_28 = -nd_28.y;
        float r0_28 = approx_rcp(d0_28);
        float r1_28 = approx_rcp(d1_28);
        float2 r_28 = make_float2(r0_28, r1_28);
        float2 s_28 = fma_f32x2_rn_noftz(r_28, fma_f32x2_rn_noftz(nd_28, r_28, make_float2(1.0f, 1.0f)), r_28);
        float t0_13 = s_28.x;
        float t1_13 = s_28.y;
        band_hi = band_hi | ((fminf(t0_13, t1_13) == 0.0f) ? 64 : 0);
        float2 _f32x2_mul_a_67 = mul_f32x2_noftz(cvt_f32_29, make_float2(max_noftz(t0_13, 0.0f), max_noftz(t1_13, 0.0f)));
        float2 mul_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_67), "l"(*(const unsigned long long*)&cvt_f32_1_28));
        packed_hi[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_28.x, mul_28.y)));
        float2 cvt_f32_30 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[7]));
        float2 cvt_f32_1_29 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[7]));
        float2 _f32x2_mul_b_68 = make_float2(-1.442695f, -1.442695f);
        float2 t_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_29) : "l"(*(const unsigned long long*)&cvt_f32_30), "l"(*(const unsigned long long*)&_f32x2_mul_b_68));
        float e0_29 = approx_exp2(t_29.x);
        float e1_29 = approx_exp2(t_29.y);
        float2 nd_29 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_29, e1_29));
        float d0_29 = -nd_29.x;
        float d1_29 = -nd_29.y;
        float r0_29 = approx_rcp(d0_29);
        float r1_29 = approx_rcp(d1_29);
        float2 r_29 = make_float2(r0_29, r1_29);
        float2 s_29 = fma_f32x2_rn_noftz(r_29, fma_f32x2_rn_noftz(nd_29, r_29, make_float2(1.0f, 1.0f)), r_29);
        float t0_14 = s_29.x;
        float t1_14 = s_29.y;
        band_hi = band_hi | ((fminf(t0_14, t1_14) == 0.0f) ? 128 : 0);
        float2 _f32x2_mul_a_69 = mul_f32x2_noftz(cvt_f32_30, make_float2(max_noftz(t0_14, 0.0f), max_noftz(t1_14, 0.0f)));
        float2 mul_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_69), "l"(*(const unsigned long long*)&cvt_f32_1_29));
        packed_hi[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_29.x, mul_29.y)));
        if (band != 0) {
            unsigned int _popc_0 = (unsigned int)__popc(band);
            #pragma unroll 1
            for (int _j = 0; _j < _popc_0; _j++) {
                int _find_first_set_70 = __ffs(band);
                int j = _find_first_set_70 - 1;
                band = band & band - 1;
                unsigned int _vec_load_4[1];
                {
                    _vec_load_4[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index + 2 * j) + 0);
                }
                unsigned int _vec_load_5[1];
                {
                    _vec_load_5[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index + K + 2 * j) + 0);
                }
                float2 cvt_f32_31 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[0]));
                float2 cvt_f32_1_30 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[0]));
                float2 _f32x2_mul_b_71 = make_float2(-1.442695f, -1.442695f);
                float2 t_30;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_30) : "l"(*(const unsigned long long*)&cvt_f32_31), "l"(*(const unsigned long long*)&_f32x2_mul_b_71));
                float e0_30 = approx_exp2(t_30.x);
                float e1_30 = approx_exp2(t_30.y);
                float2 nd_30 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_30, e1_30));
                float d0_30 = -nd_30.x;
                float d1_30 = -nd_30.y;
                float r0_30 = approx_rcp(d0_30);
                float r1_30 = approx_rcp(d1_30);
                float2 r_30 = make_float2(r0_30, r1_30);
                float2 s_30 = fma_f32x2_rn_noftz(r_30, fma_f32x2_rn_noftz(nd_30, r_30, make_float2(1.0f, 1.0f)), r_30);
                float t0_15 = s_30.x;
                float t1_15 = s_30.y;
                float d4_8a;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a) : "f"(d0_30), "f"(0.25f));
                float r4_8a = approx_rcp(d4_8a);
                float _fma_73;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(-d4_8a), "f"(r4_8a), "f"(1.0f));
                float _fma_72;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(r4_8a), "f"(_fma_73), "f"(r4_8a));
                float big_8a;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a) : "f"(mul_f32_rn_noftz(_fma_72, 1.2676506e+30f)), "f"(1.4073749e+14f));
                float _add_74;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_74) : "f"(big_8a), "f"(8.388608e+06f));
                float whole_8a;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a) : "f"(_add_74), "f"(8.388608e+06f));
                float above_8a;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a) : "f"(-d4_8a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                float _sub_76;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_76) : "f"(big_8a), "f"(whole_8a));
                float _abs_75;
                asm("abs.f32 %0, %1;" : "=f"(_abs_75) : "f"(_sub_76));
                float _add_77;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(big_8a), "f"(0.5f));
                float _sub_78;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_78) : "f"(big_8a), "f"(0.5f));
                float rounded_8a = ((_abs_75 == 0.5f) ? ((above_8a > 0.0f) ? _add_77 : _sub_78) : whole_8a);
                float u0 = ((d0_30 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a, 7.888609e-31f), 1.7763568e-15f) : t0_15);
                float d4_8a_0;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_0) : "f"(d1_30), "f"(0.25f));
                float r4_8a_0 = approx_rcp(d4_8a_0);
                float _fma_80;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(-d4_8a_0), "f"(r4_8a_0), "f"(1.0f));
                float _fma_79;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(r4_8a_0), "f"(_fma_80), "f"(r4_8a_0));
                float big_8a_0;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_0) : "f"(mul_f32_rn_noftz(_fma_79, 1.2676506e+30f)), "f"(1.4073749e+14f));
                float _add_81;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_81) : "f"(big_8a_0), "f"(8.388608e+06f));
                float whole_8a_0;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_0) : "f"(_add_81), "f"(8.388608e+06f));
                float above_8a_0;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_0) : "f"(-d4_8a_0), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_0, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                float _sub_83;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_83) : "f"(big_8a_0), "f"(whole_8a_0));
                float _abs_82;
                asm("abs.f32 %0, %1;" : "=f"(_abs_82) : "f"(_sub_83));
                float _add_84;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_84) : "f"(big_8a_0), "f"(0.5f));
                float _sub_85;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_85) : "f"(big_8a_0), "f"(0.5f));
                float rounded_8a_0 = ((_abs_82 == 0.5f) ? ((above_8a_0 > 0.0f) ? _add_84 : _sub_85) : whole_8a_0);
                float u1 = ((d1_30 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_0, 7.888609e-31f), 1.7763568e-15f) : t1_15);
                float2 _f32x2_mul_a_86 = mul_f32x2_noftz(cvt_f32_31, make_float2(max_noftz(u0, 0.0f), max_noftz(u1, 0.0f)));
                float2 mul_30;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_86), "l"(*(const unsigned long long*)&cvt_f32_1_30));
                unsigned int pair = __as_u32(__float22bfloat162_rn(make_float2(mul_30.x, mul_30.y)));
                packed[0] = ((j == 0) ? pair : packed[0]);
                packed[1] = ((j == 1) ? pair : packed[1]);
                packed[2] = ((j == 2) ? pair : packed[2]);
                packed[3] = ((j == 3) ? pair : packed[3]);
                packed[4] = ((j == 4) ? pair : packed[4]);
                packed[5] = ((j == 5) ? pair : packed[5]);
                packed[6] = ((j == 6) ? pair : packed[6]);
                packed[7] = ((j == 7) ? pair : packed[7]);
            }
        }
        if (band_hi != 0) {
            unsigned int _popc_1 = (unsigned int)__popc(band_hi);
            #pragma unroll 1
            for (int _j_hi = 0; _j_hi < _popc_1; _j_hi++) {
                int _find_first_set_87 = __ffs(band_hi);
                int j_hi = _find_first_set_87 - 1;
                band_hi = band_hi & band_hi - 1;
                unsigned int _vec_load_6[1];
                {
                    _vec_load_6[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_hi + 2 * j_hi) + 0);
                }
                unsigned int _vec_load_7[1];
                {
                    _vec_load_7[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index_hi + K + 2 * j_hi) + 0);
                }
                float2 cvt_f32_32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_6[0]));
                float2 cvt_f32_1_31 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_7[0]));
                float2 _f32x2_mul_b_88 = make_float2(-1.442695f, -1.442695f);
                float2 t_31;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_31) : "l"(*(const unsigned long long*)&cvt_f32_32), "l"(*(const unsigned long long*)&_f32x2_mul_b_88));
                float e0_31 = approx_exp2(t_31.x);
                float e1_31 = approx_exp2(t_31.y);
                float2 nd_31 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_31, e1_31));
                float d0_31 = -nd_31.x;
                float d1_31 = -nd_31.y;
                float r0_31 = approx_rcp(d0_31);
                float r1_31 = approx_rcp(d1_31);
                float2 r_31 = make_float2(r0_31, r1_31);
                float2 s_31 = fma_f32x2_rn_noftz(r_31, fma_f32x2_rn_noftz(nd_31, r_31, make_float2(1.0f, 1.0f)), r_31);
                float t0_16 = s_31.x;
                float t1_16 = s_31.y;
                float d4_8a_1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_1) : "f"(d0_31), "f"(0.25f));
                float r4_8a_1 = approx_rcp(d4_8a_1);
                float _fma_90;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(-d4_8a_1), "f"(r4_8a_1), "f"(1.0f));
                float _fma_89;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(r4_8a_1), "f"(_fma_90), "f"(r4_8a_1));
                float big_8a_1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_1) : "f"(mul_f32_rn_noftz(_fma_89, 1.2676506e+30f)), "f"(1.4073749e+14f));
                float _add_91;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_91) : "f"(big_8a_1), "f"(8.388608e+06f));
                float whole_8a_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_1) : "f"(_add_91), "f"(8.388608e+06f));
                float above_8a_1;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_1) : "f"(-d4_8a_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                float _sub_93;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_93) : "f"(big_8a_1), "f"(whole_8a_1));
                float _abs_92;
                asm("abs.f32 %0, %1;" : "=f"(_abs_92) : "f"(_sub_93));
                float _add_94;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_94) : "f"(big_8a_1), "f"(0.5f));
                float _sub_95;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_95) : "f"(big_8a_1), "f"(0.5f));
                float rounded_8a_1 = ((_abs_92 == 0.5f) ? ((above_8a_1 > 0.0f) ? _add_94 : _sub_95) : whole_8a_1);
                float u0_1 = ((d0_31 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_1, 7.888609e-31f), 1.7763568e-15f) : t0_16);
                float d4_8a_1_1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_1_1) : "f"(d1_31), "f"(0.25f));
                float r4_8a_1_1 = approx_rcp(d4_8a_1_1);
                float _fma_97;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(-d4_8a_1_1), "f"(r4_8a_1_1), "f"(1.0f));
                float _fma_96;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(r4_8a_1_1), "f"(_fma_97), "f"(r4_8a_1_1));
                float big_8a_1_1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_1_1) : "f"(mul_f32_rn_noftz(_fma_96, 1.2676506e+30f)), "f"(1.4073749e+14f));
                float _add_98;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_98) : "f"(big_8a_1_1), "f"(8.388608e+06f));
                float whole_8a_1_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_1_1) : "f"(_add_98), "f"(8.388608e+06f));
                float above_8a_1_1;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_1_1) : "f"(-d4_8a_1_1), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_1_1, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                float _sub_100;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_100) : "f"(big_8a_1_1), "f"(whole_8a_1_1));
                float _abs_99;
                asm("abs.f32 %0, %1;" : "=f"(_abs_99) : "f"(_sub_100));
                float _add_101;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_101) : "f"(big_8a_1_1), "f"(0.5f));
                float _sub_102;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_102) : "f"(big_8a_1_1), "f"(0.5f));
                float rounded_8a_1_1 = ((_abs_99 == 0.5f) ? ((above_8a_1_1 > 0.0f) ? _add_101 : _sub_102) : whole_8a_1_1);
                float u1_1 = ((d1_31 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_1_1, 7.888609e-31f), 1.7763568e-15f) : t1_16);
                float2 _f32x2_mul_a_103 = mul_f32x2_noftz(cvt_f32_32, make_float2(max_noftz(u0_1, 0.0f), max_noftz(u1_1, 0.0f)));
                float2 mul_31;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_103), "l"(*(const unsigned long long*)&cvt_f32_1_31));
                unsigned int pair_hi = __as_u32(__float22bfloat162_rn(make_float2(mul_31.x, mul_31.y)));
                packed_hi[0] = ((j_hi == 0) ? pair_hi : packed_hi[0]);
                packed_hi[1] = ((j_hi == 1) ? pair_hi : packed_hi[1]);
                packed_hi[2] = ((j_hi == 2) ? pair_hi : packed_hi[2]);
                packed_hi[3] = ((j_hi == 3) ? pair_hi : packed_hi[3]);
                packed_hi[4] = ((j_hi == 4) ? pair_hi : packed_hi[4]);
                packed_hi[5] = ((j_hi == 5) ? pair_hi : packed_hi[5]);
                packed_hi[6] = ((j_hi == 6) ? pair_hi : packed_hi[6]);
                packed_hi[7] = ((j_hi == 7) ? pair_hi : packed_hi[7]);
            }
        }
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
    __nv_bfloat162 amax = __as_bf16x2(0);
    __nv_bfloat162 _hxa_105 = amax;
    __nv_bfloat162 _hxb_105 = __as_bf16x2(packed[0]);
    unsigned int _hxr_105;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_105) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_105)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_105)));
    __nv_bfloat162 _max_104 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_105);
    amax = _max_104;
    __nv_bfloat162 _hxa_107 = amax;
    __nv_bfloat162 _hxb_107 = __as_bf16x2(packed[1]);
    unsigned int _hxr_107;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_107) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_107)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_107)));
    __nv_bfloat162 _max_106 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_107);
    amax = _max_106;
    __nv_bfloat162 _hxa_109 = amax;
    __nv_bfloat162 _hxb_109 = __as_bf16x2(packed[2]);
    unsigned int _hxr_109;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_109) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_109)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_109)));
    __nv_bfloat162 _max_108 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_109);
    amax = _max_108;
    __nv_bfloat162 _hxa_111 = amax;
    __nv_bfloat162 _hxb_111 = __as_bf16x2(packed[3]);
    unsigned int _hxr_111;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_111) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_111)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_111)));
    __nv_bfloat162 _max_110 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_111);
    amax = _max_110;
    __nv_bfloat162 _hxa_113 = amax;
    __nv_bfloat162 _hxb_113 = __as_bf16x2(packed[4]);
    unsigned int _hxr_113;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_113) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_113)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_113)));
    __nv_bfloat162 _max_112 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_113);
    amax = _max_112;
    __nv_bfloat162 _hxa_115 = amax;
    __nv_bfloat162 _hxb_115 = __as_bf16x2(packed[5]);
    unsigned int _hxr_115;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_115) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_115)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_115)));
    __nv_bfloat162 _max_114 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_115);
    amax = _max_114;
    __nv_bfloat162 _hxa_117 = amax;
    __nv_bfloat162 _hxb_117 = __as_bf16x2(packed[6]);
    unsigned int _hxr_117;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_117) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_117)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_117)));
    __nv_bfloat162 _max_116 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_117);
    amax = _max_116;
    __nv_bfloat162 _hxa_119 = amax;
    __nv_bfloat162 _hxb_119 = __as_bf16x2(packed[7]);
    unsigned int _hxr_119;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_119) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_119)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_119)));
    __nv_bfloat162 _max_118 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_119);
    amax = _max_118;
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
    int scale_col = bx * 4 + blk;
    int row_scale_index = ((grow >> 7) * (K >> 7) + (scale_col >> 2)) * 512 + (grow & 31) * 16 + (grow >> 5 & 3) * 4 + (scale_col & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + row_scale_index) + (0)) = (unsigned char)(scale);
    }
    unsigned int row_words[4];
    uint32_t bf16x2_mul;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul) : "r"(packed[0]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1) : "r"(packed[1]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_121;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_121) : "r"(bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_121) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_120;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_120) : "h"(e4m3x2_bf16x2), "h"(_cvt_satfinite_e4m3x2_bf16x2_121));
    row_words[0] = _mov_b32_pack_u16x2_120;
    uint32_t bf16x2_mul_0;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_0) : "r"(packed[2]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_0) : "r"(packed[3]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_123;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_123) : "r"(bf16x2_mul_1_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_123) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_122;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_122) : "h"(e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_123));
    row_words[1] = _mov_b32_pack_u16x2_122;
    uint32_t bf16x2_mul_2;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2) : "r"(packed[4]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_1) : "r"(packed[5]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_125;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_125) : "r"(bf16x2_mul_1_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_125) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_124;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_124) : "h"(e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_125));
    row_words[2] = _mov_b32_pack_u16x2_124;
    uint32_t bf16x2_mul_3;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3) : "r"(packed[6]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_2) : "r"(packed[7]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_127;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_127) : "r"(bf16x2_mul_1_2));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_2);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_127) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_126;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_126) : "h"(e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_127));
    row_words[3] = _mov_b32_pack_u16x2_126;
    reinterpret_cast<int4*>(row_output + (grow * K + col))[0] = reinterpret_cast<int4*>(row_words + 0)[0];
    __nv_bfloat162 amax_0 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_129 = amax_0;
    __nv_bfloat162 _hxb_129 = __as_bf16x2(packed_hi[0]);
    unsigned int _hxr_129;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_129) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_129)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_129)));
    __nv_bfloat162 _max_128 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_129);
    amax_0 = _max_128;
    __nv_bfloat162 _hxa_131 = amax_0;
    __nv_bfloat162 _hxb_131 = __as_bf16x2(packed_hi[1]);
    unsigned int _hxr_131;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_131) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_131)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_131)));
    __nv_bfloat162 _max_130 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_131);
    amax_0 = _max_130;
    __nv_bfloat162 _hxa_133 = amax_0;
    __nv_bfloat162 _hxb_133 = __as_bf16x2(packed_hi[2]);
    unsigned int _hxr_133;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_133) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_133)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_133)));
    __nv_bfloat162 _max_132 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_133);
    amax_0 = _max_132;
    __nv_bfloat162 _hxa_135 = amax_0;
    __nv_bfloat162 _hxb_135 = __as_bf16x2(packed_hi[3]);
    unsigned int _hxr_135;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_135) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_135)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_135)));
    __nv_bfloat162 _max_134 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_135);
    amax_0 = _max_134;
    __nv_bfloat162 _hxa_137 = amax_0;
    __nv_bfloat162 _hxb_137 = __as_bf16x2(packed_hi[4]);
    unsigned int _hxr_137;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_137) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_137)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_137)));
    __nv_bfloat162 _max_136 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_137);
    amax_0 = _max_136;
    __nv_bfloat162 _hxa_139 = amax_0;
    __nv_bfloat162 _hxb_139 = __as_bf16x2(packed_hi[5]);
    unsigned int _hxr_139;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_139) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_139)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_139)));
    __nv_bfloat162 _max_138 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_139);
    amax_0 = _max_138;
    __nv_bfloat162 _hxa_141 = amax_0;
    __nv_bfloat162 _hxb_141 = __as_bf16x2(packed_hi[6]);
    unsigned int _hxr_141;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_141) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_141)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_141)));
    __nv_bfloat162 _max_140 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_141);
    amax_0 = _max_140;
    __nv_bfloat162 _hxa_143 = amax_0;
    __nv_bfloat162 _hxb_143 = __as_bf16x2(packed_hi[7]);
    unsigned int _hxr_143;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_143) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_143)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_143)));
    __nv_bfloat162 _max_142 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_143);
    amax_0 = _max_142;
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
    int scale_col_0 = bx * 4 + blk;
    int row_scale_index_0 = ((grow_hi >> 7) * (K >> 7) + (scale_col_0 >> 2)) * 512 + (grow_hi & 31) * 16 + (grow_hi >> 5 & 3) * 4 + (scale_col_0 & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + row_scale_index_0) + (0)) = (unsigned char)(scale_0);
    }
    unsigned int row_words_0[4];
    uint32_t bf16x2_mul_4;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4) : "r"(packed_hi[0]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_3) : "r"(packed_hi[1]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_145;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_145) : "r"(bf16x2_mul_1_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_145) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_144;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_144) : "h"(e4m3x2_bf16x2_3), "h"(_cvt_satfinite_e4m3x2_bf16x2_145));
    row_words_0[0] = _mov_b32_pack_u16x2_144;
    uint32_t bf16x2_mul_5;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5) : "r"(packed_hi[2]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_4) : "r"(packed_hi[3]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_147;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_147) : "r"(bf16x2_mul_1_4));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_4);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_147) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_146;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_146) : "h"(e4m3x2_bf16x2_4), "h"(_cvt_satfinite_e4m3x2_bf16x2_147));
    row_words_0[1] = _mov_b32_pack_u16x2_146;
    uint32_t bf16x2_mul_6;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6) : "r"(packed_hi[4]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_5) : "r"(packed_hi[5]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_149;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_149) : "r"(bf16x2_mul_1_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_149) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_148;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_148) : "h"(e4m3x2_bf16x2_5), "h"(_cvt_satfinite_e4m3x2_bf16x2_149));
    row_words_0[2] = _mov_b32_pack_u16x2_148;
    uint32_t bf16x2_mul_7;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7) : "r"(packed_hi[6]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_6) : "r"(packed_hi[7]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_151;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_151) : "r"(bf16x2_mul_1_6));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_6);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_151) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_150;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_150) : "h"(e4m3x2_bf16x2_6), "h"(_cvt_satfinite_e4m3x2_bf16x2_151));
    row_words_0[3] = _mov_b32_pack_u16x2_150;
    reinterpret_cast<int4*>(row_output + (grow_hi * K + col))[0] = reinterpret_cast<int4*>(row_words_0 + 0)[0];
    __syncthreads();
    int lane_1 = tid & 31;
    int warp_1 = tid >> 5;
    int k8 = lane_1 & 7;
    int srow = (k8 >> 1) * 16 + (lane_1 >> 3) * 2 + (k8 & 1);
    unsigned int sswizzle = (unsigned int)(srow >> 3 & 6 | srow & 1);
    unsigned int cols[16];
    unsigned int strip_chunk0_1_0 = (unsigned int)(warp_1 * 2);
    unsigned int strip_src0_1_0 = tile_addr + (unsigned int)(srow * 256) + ((strip_chunk0_1_0 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[0]), "=r"(cols[1]), "=r"(cols[2]), "=r"(cols[3])
        : "r"(strip_src0_1_0)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[4]), "=r"(cols[5]), "=r"(cols[6]), "=r"(cols[7])
        : "r"(strip_src0_1_0 + 2048)
        : "memory");
    unsigned int strip_chunk0_1_1 = (unsigned int)(warp_1 * 2 + 1);
    unsigned int strip_src0_1_1 = tile_addr + (unsigned int)(srow * 256) + ((strip_chunk0_1_1 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[8]), "=r"(cols[9]), "=r"(cols[10]), "=r"(cols[11])
        : "r"(strip_src0_1_1)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[12]), "=r"(cols[13]), "=r"(cols[14]), "=r"(cols[15])
        : "r"(strip_src0_1_1 + 2048)
        : "memory");
    __nv_bfloat162 amax_a = __as_bf16x2(0);
    __nv_bfloat162 amax_b = __as_bf16x2(0);
    __nv_bfloat162 _hxa_153 = amax_a;
    __nv_bfloat162 _hxb_153 = __as_bf16x2(cols[0]);
    unsigned int _hxr_153;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_153) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_153)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_153)));
    __nv_bfloat162 _max_152 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_153);
    amax_a = _max_152;
    __nv_bfloat162 _hxa_155 = amax_b;
    __nv_bfloat162 _hxb_155 = __as_bf16x2(cols[8]);
    unsigned int _hxr_155;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_155) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_155)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_155)));
    __nv_bfloat162 _max_154 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_155);
    amax_b = _max_154;
    __nv_bfloat162 _hxa_157 = amax_a;
    __nv_bfloat162 _hxb_157 = __as_bf16x2(cols[1]);
    unsigned int _hxr_157;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_157) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_157)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_157)));
    __nv_bfloat162 _max_156 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_157);
    amax_a = _max_156;
    __nv_bfloat162 _hxa_159 = amax_b;
    __nv_bfloat162 _hxb_159 = __as_bf16x2(cols[9]);
    unsigned int _hxr_159;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_159) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_159)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_159)));
    __nv_bfloat162 _max_158 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_159);
    amax_b = _max_158;
    __nv_bfloat162 _hxa_161 = amax_a;
    __nv_bfloat162 _hxb_161 = __as_bf16x2(cols[2]);
    unsigned int _hxr_161;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_161) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_161)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_161)));
    __nv_bfloat162 _max_160 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_161);
    amax_a = _max_160;
    __nv_bfloat162 _hxa_163 = amax_b;
    __nv_bfloat162 _hxb_163 = __as_bf16x2(cols[10]);
    unsigned int _hxr_163;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_163) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_163)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_163)));
    __nv_bfloat162 _max_162 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_163);
    amax_b = _max_162;
    __nv_bfloat162 _hxa_165 = amax_a;
    __nv_bfloat162 _hxb_165 = __as_bf16x2(cols[3]);
    unsigned int _hxr_165;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_165) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_165)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_165)));
    __nv_bfloat162 _max_164 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_165);
    amax_a = _max_164;
    __nv_bfloat162 _hxa_167 = amax_b;
    __nv_bfloat162 _hxb_167 = __as_bf16x2(cols[11]);
    unsigned int _hxr_167;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_167) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_167)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_167)));
    __nv_bfloat162 _max_166 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_167);
    amax_b = _max_166;
    __nv_bfloat162 _hxa_169 = amax_a;
    __nv_bfloat162 _hxb_169 = __as_bf16x2(cols[4]);
    unsigned int _hxr_169;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_169) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_169)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_169)));
    __nv_bfloat162 _max_168 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_169);
    amax_a = _max_168;
    __nv_bfloat162 _hxa_171 = amax_b;
    __nv_bfloat162 _hxb_171 = __as_bf16x2(cols[12]);
    unsigned int _hxr_171;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_171) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_171)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_171)));
    __nv_bfloat162 _max_170 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_171);
    amax_b = _max_170;
    __nv_bfloat162 _hxa_173 = amax_a;
    __nv_bfloat162 _hxb_173 = __as_bf16x2(cols[5]);
    unsigned int _hxr_173;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_173) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_173)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_173)));
    __nv_bfloat162 _max_172 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_173);
    amax_a = _max_172;
    __nv_bfloat162 _hxa_175 = amax_b;
    __nv_bfloat162 _hxb_175 = __as_bf16x2(cols[13]);
    unsigned int _hxr_175;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_175) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_175)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_175)));
    __nv_bfloat162 _max_174 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_175);
    amax_b = _max_174;
    __nv_bfloat162 _hxa_177 = amax_a;
    __nv_bfloat162 _hxb_177 = __as_bf16x2(cols[6]);
    unsigned int _hxr_177;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_177) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_177)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_177)));
    __nv_bfloat162 _max_176 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_177);
    amax_a = _max_176;
    __nv_bfloat162 _hxa_179 = amax_b;
    __nv_bfloat162 _hxb_179 = __as_bf16x2(cols[14]);
    unsigned int _hxr_179;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_179) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_179)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_179)));
    __nv_bfloat162 _max_178 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_179);
    amax_b = _max_178;
    __nv_bfloat162 _hxa_181 = amax_a;
    __nv_bfloat162 _hxb_181 = __as_bf16x2(cols[7]);
    unsigned int _hxr_181;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_181) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_181)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_181)));
    __nv_bfloat162 _max_180 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_181);
    amax_a = _max_180;
    __nv_bfloat162 _hxa_183 = amax_b;
    __nv_bfloat162 _hxb_183 = __as_bf16x2(cols[15]);
    unsigned int _hxr_183;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_183) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_183)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_183)));
    __nv_bfloat162 _max_182 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_183);
    amax_b = _max_182;
    unsigned int mag_a = __as_u32(amax_a) & 2147450879;
    unsigned int mag_b = __as_u32(amax_b) & 2147450879;
    unsigned int lows_ab = mag_a & 65535 | mag_b << 16;
    unsigned int highs_ab = mag_a >> 16 | mag_b & 4294901760u;
    __nv_bfloat162 col_max = __hmax2_nan(__as_bf16x2(lows_ab), __as_bf16x2(highs_ab));
    unsigned int peer1 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(col_max), 1);
    col_max = __hmax2_nan(col_max, __as_bf16x2(peer1));
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
    unsigned int scale_3 = (unsigned int)scale_i32_2;
    unsigned int inv_2 = 254 - scale_3 << 7;
    if (scale_3 == 255) {
        inv_2 = 32704;
    }
    unsigned int inv2_2 = inv_2 | inv_2 << 16;
    int gcol_a = bx * 128 + warp_1 * 16 + (lane_1 >> 2);
    int gcol_b = gcol_a + 8;
    int out_row = by * 64 + (lane_1 & 3) * 16;
    unsigned int words_a[4];
    unsigned int words_b[4];
    uint32_t bf16x2_mul_8;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_8) : "r"(cols[0]), "r"(inv2_1));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_7) : "r"(cols[1]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_185;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_185) : "r"(bf16x2_mul_1_7));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_7);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_185) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_184;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_184) : "h"(e4m3x2_bf16x2_7), "h"(_cvt_satfinite_e4m3x2_bf16x2_185));
    words_a[0] = _mov_b32_pack_u16x2_184;
    uint32_t bf16x2_mul_9;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_9) : "r"(cols[8]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_8) : "r"(cols[9]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_187;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_187) : "r"(bf16x2_mul_1_8));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_8);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_187) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_186;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_186) : "h"(e4m3x2_bf16x2_8), "h"(_cvt_satfinite_e4m3x2_bf16x2_187));
    words_b[0] = _mov_b32_pack_u16x2_186;
    uint32_t bf16x2_mul_10;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_10) : "r"(cols[2]), "r"(inv2_1));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_9) : "r"(cols[3]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_189;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_189) : "r"(bf16x2_mul_1_9));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_9);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_189) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_188;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_188) : "h"(e4m3x2_bf16x2_9), "h"(_cvt_satfinite_e4m3x2_bf16x2_189));
    words_a[1] = _mov_b32_pack_u16x2_188;
    uint32_t bf16x2_mul_11;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_11) : "r"(cols[10]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_10) : "r"(cols[11]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_191;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_191) : "r"(bf16x2_mul_1_10));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_10);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_191) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_190;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_190) : "h"(e4m3x2_bf16x2_10), "h"(_cvt_satfinite_e4m3x2_bf16x2_191));
    words_b[1] = _mov_b32_pack_u16x2_190;
    uint32_t bf16x2_mul_12;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_12) : "r"(cols[4]), "r"(inv2_1));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_11) : "r"(cols[5]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_193;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_193) : "r"(bf16x2_mul_1_11));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_11);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_193) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_192;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_192) : "h"(e4m3x2_bf16x2_11), "h"(_cvt_satfinite_e4m3x2_bf16x2_193));
    words_a[2] = _mov_b32_pack_u16x2_192;
    uint32_t bf16x2_mul_13;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_13) : "r"(cols[12]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_12) : "r"(cols[13]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_195;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_195) : "r"(bf16x2_mul_1_12));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_12);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_195) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_194;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_194) : "h"(e4m3x2_bf16x2_12), "h"(_cvt_satfinite_e4m3x2_bf16x2_195));
    words_b[2] = _mov_b32_pack_u16x2_194;
    uint32_t bf16x2_mul_14;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_14) : "r"(cols[6]), "r"(inv2_1));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_13) : "r"(cols[7]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_197;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_197) : "r"(bf16x2_mul_1_13));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_13);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_197) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_196;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_196) : "h"(e4m3x2_bf16x2_13), "h"(_cvt_satfinite_e4m3x2_bf16x2_197));
    words_a[3] = _mov_b32_pack_u16x2_196;
    uint32_t bf16x2_mul_15;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_15) : "r"(cols[14]), "r"(inv2_2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_14) : "r"(cols[15]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_199;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_199) : "r"(bf16x2_mul_1_14));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_14);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_199) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_198;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_198) : "h"(e4m3x2_bf16x2_14), "h"(_cvt_satfinite_e4m3x2_bf16x2_199));
    words_b[3] = _mov_b32_pack_u16x2_198;
    reinterpret_cast<int4*>(col_output + (gcol_a * M + out_row))[0] = reinterpret_cast<int4*>(words_a + 0)[0];
    reinterpret_cast<int4*>(col_output + (gcol_b * M + out_row))[0] = reinterpret_cast<int4*>(words_b + 0)[0];
    if ((lane_1 & 1) == 0) {
        int row_tiles = M >> 7;
        int rblk = by * 2 + ((lane_1 & 3) >> 1);
        int col_scale_a = ((gcol_a >> 7) * row_tiles + (rblk >> 2)) * 512 + (gcol_a & 31) * 16 + (gcol_a >> 5 & 3) * 4 + (rblk & 3);
        int col_scale_b = ((gcol_b >> 7) * row_tiles + (rblk >> 2)) * 512 + (gcol_b & 31) * 16 + (gcol_b >> 5 & 3) * 4 + (rblk & 3);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_a) + (0)) = (unsigned char)(scale_1);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b) + (0)) = (unsigned char)(scale_3);
    }
}

} // extern "C"
