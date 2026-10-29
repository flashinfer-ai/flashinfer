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
#define SMEM_TILE_STAGE_BYTES 8192
#define SMEM_TILE_STRIDE 8192
#define SMEM_TOTAL 8192
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 8

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
kernel_cake_gated_act_mxfp8_ff6e1cf61c61f4b595fa(__nv_bfloat16* __restrict__ gated_input, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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
    int grow = by * 32 + row;
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
    unsigned int packed[8];
    __nv_bfloat162 lowest = __as_bf16x2(_vec_load_0[0]);
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[7]));
    float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(lowest);
    float _min_3;
    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_3) : "f"(_cvt_f32_0.x), "f"(_cvt_f32_0.y));
    int _warp_vote_2 = __all_sync(0xFFFFFFFF, _min_3 > -87.0f);
    if (_warp_vote_2 != 0) {
        float2 cvt_f32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 _f32x2_mul_b_4 = make_float2(-1.442695f, -1.442695f);
        float2 t;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t) : "l"(*(const unsigned long long*)&cvt_f32), "l"(*(const unsigned long long*)&_f32x2_mul_b_4));
        float e0 = approx_exp2(t.x);
        float e1 = approx_exp2(t.y);
        float2 nd = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0, e1));
        float d0 = -nd.x;
        float d1 = -nd.y;
        float r0 = approx_rcp(d0);
        float r1 = approx_rcp(d1);
        float2 r = make_float2(r0, r1);
        float2 s = fma_f32x2_rn_noftz(r, fma_f32x2_rn_noftz(nd, r, make_float2(1.0f, 1.0f)), r);
        float2 _f32x2_mul_a_5 = mul_f32x2_noftz(cvt_f32, s);
        float2 mul;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul) : "l"(*(const unsigned long long*)&_f32x2_mul_a_5), "l"(*(const unsigned long long*)&cvt_f32_1));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(mul.x, mul.y)));
        float2 cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 cvt_f32_1_0 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 _f32x2_mul_b_6 = make_float2(-1.442695f, -1.442695f);
        float2 t_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_0) : "l"(*(const unsigned long long*)&cvt_f32_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_6));
        float e0_0 = approx_exp2(t_0.x);
        float e1_0 = approx_exp2(t_0.y);
        float2 nd_0 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_0, e1_0));
        float d0_0 = -nd_0.x;
        float d1_0 = -nd_0.y;
        float r0_0 = approx_rcp(d0_0);
        float r1_0 = approx_rcp(d1_0);
        float2 r_0 = make_float2(r0_0, r1_0);
        float2 s_0 = fma_f32x2_rn_noftz(r_0, fma_f32x2_rn_noftz(nd_0, r_0, make_float2(1.0f, 1.0f)), r_0);
        float2 _f32x2_mul_a_7 = mul_f32x2_noftz(cvt_f32_0, s_0);
        float2 mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_7), "l"(*(const unsigned long long*)&cvt_f32_1_0));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_0.x, mul_0.y)));
        float2 cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 cvt_f32_1_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 _f32x2_mul_b_8 = make_float2(-1.442695f, -1.442695f);
        float2 t_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_1) : "l"(*(const unsigned long long*)&cvt_f32_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_8));
        float e0_1 = approx_exp2(t_1.x);
        float e1_1 = approx_exp2(t_1.y);
        float2 nd_1 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_1, e1_1));
        float d0_1 = -nd_1.x;
        float d1_1 = -nd_1.y;
        float r0_1 = approx_rcp(d0_1);
        float r1_1 = approx_rcp(d1_1);
        float2 r_1 = make_float2(r0_1, r1_1);
        float2 s_1 = fma_f32x2_rn_noftz(r_1, fma_f32x2_rn_noftz(nd_1, r_1, make_float2(1.0f, 1.0f)), r_1);
        float2 _f32x2_mul_a_9 = mul_f32x2_noftz(cvt_f32_2, s_1);
        float2 mul_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_9), "l"(*(const unsigned long long*)&cvt_f32_1_1));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_1.x, mul_1.y)));
        float2 cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 cvt_f32_1_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 _f32x2_mul_b_10 = make_float2(-1.442695f, -1.442695f);
        float2 t_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_2) : "l"(*(const unsigned long long*)&cvt_f32_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_10));
        float e0_2 = approx_exp2(t_2.x);
        float e1_2 = approx_exp2(t_2.y);
        float2 nd_2 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_2, e1_2));
        float d0_2 = -nd_2.x;
        float d1_2 = -nd_2.y;
        float r0_2 = approx_rcp(d0_2);
        float r1_2 = approx_rcp(d1_2);
        float2 r_2 = make_float2(r0_2, r1_2);
        float2 s_2 = fma_f32x2_rn_noftz(r_2, fma_f32x2_rn_noftz(nd_2, r_2, make_float2(1.0f, 1.0f)), r_2);
        float2 _f32x2_mul_a_11 = mul_f32x2_noftz(cvt_f32_3, s_2);
        float2 mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_11), "l"(*(const unsigned long long*)&cvt_f32_1_2));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_2.x, mul_2.y)));
        float2 cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 cvt_f32_1_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float2 _f32x2_mul_b_12 = make_float2(-1.442695f, -1.442695f);
        float2 t_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_3) : "l"(*(const unsigned long long*)&cvt_f32_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_12));
        float e0_3 = approx_exp2(t_3.x);
        float e1_3 = approx_exp2(t_3.y);
        float2 nd_3 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_3, e1_3));
        float d0_3 = -nd_3.x;
        float d1_3 = -nd_3.y;
        float r0_3 = approx_rcp(d0_3);
        float r1_3 = approx_rcp(d1_3);
        float2 r_3 = make_float2(r0_3, r1_3);
        float2 s_3 = fma_f32x2_rn_noftz(r_3, fma_f32x2_rn_noftz(nd_3, r_3, make_float2(1.0f, 1.0f)), r_3);
        float2 _f32x2_mul_a_13 = mul_f32x2_noftz(cvt_f32_4, s_3);
        float2 mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_13), "l"(*(const unsigned long long*)&cvt_f32_1_3));
        packed[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_3.x, mul_3.y)));
        float2 cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 cvt_f32_1_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float2 _f32x2_mul_b_14 = make_float2(-1.442695f, -1.442695f);
        float2 t_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_4) : "l"(*(const unsigned long long*)&cvt_f32_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_14));
        float e0_4 = approx_exp2(t_4.x);
        float e1_4 = approx_exp2(t_4.y);
        float2 nd_4 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_4, e1_4));
        float d0_4 = -nd_4.x;
        float d1_4 = -nd_4.y;
        float r0_4 = approx_rcp(d0_4);
        float r1_4 = approx_rcp(d1_4);
        float2 r_4 = make_float2(r0_4, r1_4);
        float2 s_4 = fma_f32x2_rn_noftz(r_4, fma_f32x2_rn_noftz(nd_4, r_4, make_float2(1.0f, 1.0f)), r_4);
        float2 _f32x2_mul_a_15 = mul_f32x2_noftz(cvt_f32_5, s_4);
        float2 mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&cvt_f32_1_4));
        packed[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_4.x, mul_4.y)));
        float2 cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 cvt_f32_1_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float2 _f32x2_mul_b_16 = make_float2(-1.442695f, -1.442695f);
        float2 t_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_5) : "l"(*(const unsigned long long*)&cvt_f32_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_16));
        float e0_5 = approx_exp2(t_5.x);
        float e1_5 = approx_exp2(t_5.y);
        float2 nd_5 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_5, e1_5));
        float d0_5 = -nd_5.x;
        float d1_5 = -nd_5.y;
        float r0_5 = approx_rcp(d0_5);
        float r1_5 = approx_rcp(d1_5);
        float2 r_5 = make_float2(r0_5, r1_5);
        float2 s_5 = fma_f32x2_rn_noftz(r_5, fma_f32x2_rn_noftz(nd_5, r_5, make_float2(1.0f, 1.0f)), r_5);
        float2 _f32x2_mul_a_17 = mul_f32x2_noftz(cvt_f32_6, s_5);
        float2 mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&cvt_f32_1_5));
        packed[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_5.x, mul_5.y)));
        float2 cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 cvt_f32_1_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float2 _f32x2_mul_b_18 = make_float2(-1.442695f, -1.442695f);
        float2 t_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_6) : "l"(*(const unsigned long long*)&cvt_f32_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_18));
        float e0_6 = approx_exp2(t_6.x);
        float e1_6 = approx_exp2(t_6.y);
        float2 nd_6 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_6, e1_6));
        float d0_6 = -nd_6.x;
        float d1_6 = -nd_6.y;
        float r0_6 = approx_rcp(d0_6);
        float r1_6 = approx_rcp(d1_6);
        float2 r_6 = make_float2(r0_6, r1_6);
        float2 s_6 = fma_f32x2_rn_noftz(r_6, fma_f32x2_rn_noftz(nd_6, r_6, make_float2(1.0f, 1.0f)), r_6);
        float2 _f32x2_mul_a_19 = mul_f32x2_noftz(cvt_f32_7, s_6);
        float2 mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&cvt_f32_1_6));
        packed[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_6.x, mul_6.y)));
    } else {
        int band = 0;
        float2 cvt_f32_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 cvt_f32_1_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 _f32x2_mul_b_20 = make_float2(-1.442695f, -1.442695f);
        float2 t_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_7) : "l"(*(const unsigned long long*)&cvt_f32_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
        float e0_7 = approx_exp2(t_7.x);
        float e1_7 = approx_exp2(t_7.y);
        float2 nd_7 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_7, e1_7));
        float d0_7 = -nd_7.x;
        float d1_7 = -nd_7.y;
        float r0_7 = approx_rcp(d0_7);
        float r1_7 = approx_rcp(d1_7);
        float2 r_7 = make_float2(r0_7, r1_7);
        float2 s_7 = fma_f32x2_rn_noftz(r_7, fma_f32x2_rn_noftz(nd_7, r_7, make_float2(1.0f, 1.0f)), r_7);
        float t0 = s_7.x;
        float t1 = s_7.y;
        band = band | ((fminf(t0, t1) == 0.0f) ? 1 : 0);
        float2 _f32x2_mul_a_21 = mul_f32x2_noftz(cvt_f32_8, make_float2(max_noftz(t0, 0.0f), max_noftz(t1, 0.0f)));
        float2 mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&cvt_f32_1_7));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_7.x, mul_7.y)));
        float2 cvt_f32_8_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 cvt_f32_1_7_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 _f32x2_mul_b_22 = make_float2(-1.442695f, -1.442695f);
        float2 t_7_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_7_1) : "l"(*(const unsigned long long*)&cvt_f32_8_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_22));
        float e0_7_1 = approx_exp2(t_7_1.x);
        float e1_7_1 = approx_exp2(t_7_1.y);
        float2 nd_7_1 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_7_1, e1_7_1));
        float d0_7_1 = -nd_7_1.x;
        float d1_7_1 = -nd_7_1.y;
        float r0_7_1 = approx_rcp(d0_7_1);
        float r1_7_1 = approx_rcp(d1_7_1);
        float2 r_7_1 = make_float2(r0_7_1, r1_7_1);
        float2 s_7_1 = fma_f32x2_rn_noftz(r_7_1, fma_f32x2_rn_noftz(nd_7_1, r_7_1, make_float2(1.0f, 1.0f)), r_7_1);
        float t0_0 = s_7_1.x;
        float t1_0 = s_7_1.y;
        band = band | ((fminf(t0_0, t1_0) == 0.0f) ? 2 : 0);
        float2 _f32x2_mul_a_23 = mul_f32x2_noftz(cvt_f32_8_1, make_float2(max_noftz(t0_0, 0.0f), max_noftz(t1_0, 0.0f)));
        float2 mul_7_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_7_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&cvt_f32_1_7_1));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_7_1.x, mul_7_1.y)));
        float2 cvt_f32_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 cvt_f32_1_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
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
        float t0_1 = s_8.x;
        float t1_1 = s_8.y;
        band = band | ((fminf(t0_1, t1_1) == 0.0f) ? 4 : 0);
        float2 _f32x2_mul_a_25 = mul_f32x2_noftz(cvt_f32_9, make_float2(max_noftz(t0_1, 0.0f), max_noftz(t1_1, 0.0f)));
        float2 mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_25), "l"(*(const unsigned long long*)&cvt_f32_1_8));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_8.x, mul_8.y)));
        float2 cvt_f32_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 cvt_f32_1_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
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
        float t0_2 = s_9.x;
        float t1_2 = s_9.y;
        band = band | ((fminf(t0_2, t1_2) == 0.0f) ? 8 : 0);
        float2 _f32x2_mul_a_27 = mul_f32x2_noftz(cvt_f32_10, make_float2(max_noftz(t0_2, 0.0f), max_noftz(t1_2, 0.0f)));
        float2 mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_27), "l"(*(const unsigned long long*)&cvt_f32_1_9));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_9.x, mul_9.y)));
        float2 cvt_f32_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 cvt_f32_1_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
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
        float t0_3 = s_10.x;
        float t1_3 = s_10.y;
        band = band | ((fminf(t0_3, t1_3) == 0.0f) ? 16 : 0);
        float2 _f32x2_mul_a_29 = mul_f32x2_noftz(cvt_f32_11, make_float2(max_noftz(t0_3, 0.0f), max_noftz(t1_3, 0.0f)));
        float2 mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&cvt_f32_1_10));
        packed[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_10.x, mul_10.y)));
        float2 cvt_f32_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 cvt_f32_1_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
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
        float t0_4 = s_11.x;
        float t1_4 = s_11.y;
        band = band | ((fminf(t0_4, t1_4) == 0.0f) ? 32 : 0);
        float2 _f32x2_mul_a_31 = mul_f32x2_noftz(cvt_f32_12, make_float2(max_noftz(t0_4, 0.0f), max_noftz(t1_4, 0.0f)));
        float2 mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_31), "l"(*(const unsigned long long*)&cvt_f32_1_11));
        packed[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_11.x, mul_11.y)));
        float2 cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 cvt_f32_1_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
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
        float t0_5 = s_12.x;
        float t1_5 = s_12.y;
        band = band | ((fminf(t0_5, t1_5) == 0.0f) ? 64 : 0);
        float2 _f32x2_mul_a_33 = mul_f32x2_noftz(cvt_f32_13, make_float2(max_noftz(t0_5, 0.0f), max_noftz(t1_5, 0.0f)));
        float2 mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&cvt_f32_1_12));
        packed[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_12.x, mul_12.y)));
        float2 cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 cvt_f32_1_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
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
        float t0_6 = s_13.x;
        float t1_6 = s_13.y;
        band = band | ((fminf(t0_6, t1_6) == 0.0f) ? 128 : 0);
        float2 _f32x2_mul_a_35 = mul_f32x2_noftz(cvt_f32_14, make_float2(max_noftz(t0_6, 0.0f), max_noftz(t1_6, 0.0f)));
        float2 mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&cvt_f32_1_13));
        packed[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_13.x, mul_13.y)));
        if (band != 0) {
            #pragma unroll 1
            for (int i = 0; i < 8; i++) {
                if ((band >> i & 1) != 0) {
                    unsigned int _vec_load_2[1];
                    {
                        _vec_load_2[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index + 2 * i) + 0);
                    }
                    unsigned int _vec_load_3[1];
                    {
                        _vec_load_3[0] = *reinterpret_cast<const unsigned int*>(gated_input + (gate_index + K + 2 * i) + 0);
                    }
                    float2 cvt_f32_15 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
                    float2 cvt_f32_1_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
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
                    float t0_7 = s_14.x;
                    float t1_7 = s_14.y;
                    float d4_8a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a) : "f"(d0_14), "f"(0.25f));
                    float r4_8a = approx_rcp(d4_8a);
                    float _fma_38;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(-d4_8a), "f"(r4_8a), "f"(1.0f));
                    float _fma_37;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(r4_8a), "f"(_fma_38), "f"(r4_8a));
                    float big_8a;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a) : "f"(mul_f32_rn_noftz(_fma_37, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_39;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39) : "f"(big_8a), "f"(8.388608e+06f));
                    float whole_8a;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a) : "f"(_add_39), "f"(8.388608e+06f));
                    float above_8a;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a) : "f"(-d4_8a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_41;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41) : "f"(big_8a), "f"(whole_8a));
                    float _abs_40;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_40) : "f"(_sub_41));
                    float _add_42;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(big_8a), "f"(0.5f));
                    float _sub_43;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43) : "f"(big_8a), "f"(0.5f));
                    float rounded_8a = ((_abs_40 == 0.5f) ? ((above_8a > 0.0f) ? _add_42 : _sub_43) : whole_8a);
                    float u0 = ((d0_14 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a, 7.888609e-31f), 1.7763568e-15f) : t0_7);
                    float d4_8a_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_0) : "f"(d1_14), "f"(0.25f));
                    float r4_8a_0 = approx_rcp(d4_8a_0);
                    float _fma_45;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(-d4_8a_0), "f"(r4_8a_0), "f"(1.0f));
                    float _fma_44;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(r4_8a_0), "f"(_fma_45), "f"(r4_8a_0));
                    float big_8a_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_0) : "f"(mul_f32_rn_noftz(_fma_44, 1.2676506e+30f)), "f"(1.4073749e+14f));
                    float _add_46;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(big_8a_0), "f"(8.388608e+06f));
                    float whole_8a_0;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_0) : "f"(_add_46), "f"(8.388608e+06f));
                    float above_8a_0;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_0) : "f"(-d4_8a_0), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_0, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
                    float _sub_48;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_48) : "f"(big_8a_0), "f"(whole_8a_0));
                    float _abs_47;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_47) : "f"(_sub_48));
                    float _add_49;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(big_8a_0), "f"(0.5f));
                    float _sub_50;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_50) : "f"(big_8a_0), "f"(0.5f));
                    float rounded_8a_0 = ((_abs_47 == 0.5f) ? ((above_8a_0 > 0.0f) ? _add_49 : _sub_50) : whole_8a_0);
                    float u1 = ((d1_14 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_0, 7.888609e-31f), 1.7763568e-15f) : t1_7);
                    float2 _f32x2_mul_a_51 = mul_f32x2_noftz(cvt_f32_15, make_float2(max_noftz(u0, 0.0f), max_noftz(u1, 0.0f)));
                    float2 mul_14;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_51), "l"(*(const unsigned long long*)&cvt_f32_1_14));
                    unsigned int pair = __as_u32(__float22bfloat162_rn(make_float2(mul_14.x, mul_14.y)));
                    packed[0] = ((i == 0) ? pair : packed[0]);
                    packed[1] = ((i == 1) ? pair : packed[1]);
                    packed[2] = ((i == 2) ? pair : packed[2]);
                    packed[3] = ((i == 3) ? pair : packed[3]);
                    packed[4] = ((i == 4) ? pair : packed[4]);
                    packed[5] = ((i == 5) ? pair : packed[5]);
                    packed[6] = ((i == 6) ? pair : packed[6]);
                    packed[7] = ((i == 7) ? pair : packed[7]);
                }
            }
        }
    }
    unsigned int swizzle = (unsigned int)(row >> 2 & 6 | row & 1);
    unsigned int row_base = tile_addr + (unsigned int)(row * 256);
    unsigned int chunk = (unsigned int)((tid & 7) * 2);
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + ((chunk ^ swizzle) << 4)), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + ((chunk + 1 ^ swizzle) << 4)), "r"(packed[4]), "r"(packed[5]), "r"(packed[6]), "r"(packed[7]) : "memory");
    __syncthreads();
    int lane_1 = tid & 31;
    int warp_1 = tid >> 5;
    int k8 = lane_1 & 7;
    int srow = (k8 >> 1) * 8 + (lane_1 >> 3) * 2 + (k8 & 1);
    unsigned int sswizzle = (unsigned int)(srow >> 2 & 6 | srow & 1);
    unsigned int cols[8];
    unsigned int strip_chunk0_1_0 = (unsigned int)(warp_1 * 2);
    unsigned int strip_src0_1_0 = tile_addr + (unsigned int)(srow * 256) + ((strip_chunk0_1_0 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[0]), "=r"(cols[1]), "=r"(cols[2]), "=r"(cols[3])
        : "r"(strip_src0_1_0)
        : "memory");
    unsigned int strip_chunk0_1_1 = (unsigned int)(warp_1 * 2 + 1);
    unsigned int strip_src0_1_1 = tile_addr + (unsigned int)(srow * 256) + ((strip_chunk0_1_1 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[4]), "=r"(cols[5]), "=r"(cols[6]), "=r"(cols[7])
        : "r"(strip_src0_1_1)
        : "memory");
    __nv_bfloat162 amax_a = __as_bf16x2(0);
    __nv_bfloat162 amax_b = __as_bf16x2(0);
    __nv_bfloat162 _hxa_53 = amax_a;
    __nv_bfloat162 _hxb_53 = __as_bf16x2(cols[0]);
    unsigned int _hxr_53;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_53) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_53)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_53)));
    __nv_bfloat162 _max_52 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_53);
    amax_a = _max_52;
    __nv_bfloat162 _hxa_55 = amax_b;
    __nv_bfloat162 _hxb_55 = __as_bf16x2(cols[4]);
    unsigned int _hxr_55;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_55) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_55)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_55)));
    __nv_bfloat162 _max_54 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_55);
    amax_b = _max_54;
    __nv_bfloat162 _hxa_57 = amax_a;
    __nv_bfloat162 _hxb_57 = __as_bf16x2(cols[1]);
    unsigned int _hxr_57;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_57) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_57)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_57)));
    __nv_bfloat162 _max_56 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_57);
    amax_a = _max_56;
    __nv_bfloat162 _hxa_59 = amax_b;
    __nv_bfloat162 _hxb_59 = __as_bf16x2(cols[5]);
    unsigned int _hxr_59;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_59) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_59)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_59)));
    __nv_bfloat162 _max_58 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_59);
    amax_b = _max_58;
    __nv_bfloat162 _hxa_61 = amax_a;
    __nv_bfloat162 _hxb_61 = __as_bf16x2(cols[2]);
    unsigned int _hxr_61;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_61) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_61)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_61)));
    __nv_bfloat162 _max_60 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_61);
    amax_a = _max_60;
    __nv_bfloat162 _hxa_63 = amax_b;
    __nv_bfloat162 _hxb_63 = __as_bf16x2(cols[6]);
    unsigned int _hxr_63;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_63) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_63)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_63)));
    __nv_bfloat162 _max_62 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_63);
    amax_b = _max_62;
    __nv_bfloat162 _hxa_65 = amax_a;
    __nv_bfloat162 _hxb_65 = __as_bf16x2(cols[3]);
    unsigned int _hxr_65;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_65) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_65)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_65)));
    __nv_bfloat162 _max_64 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_65);
    amax_a = _max_64;
    __nv_bfloat162 _hxa_67 = amax_b;
    __nv_bfloat162 _hxb_67 = __as_bf16x2(cols[7]);
    unsigned int _hxr_67;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_67) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_67)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_67)));
    __nv_bfloat162 _max_66 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_67);
    amax_b = _max_66;
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
    int gcol_a = bx * 128 + warp_1 * 16 + (lane_1 >> 2);
    int gcol_b = gcol_a + 8;
    int out_row = by * 32 + (lane_1 & 3) * 8;
    unsigned int words_a[2];
    unsigned int words_b[2];
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_69;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_69) : "r"(bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_69) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_68;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_68) : "h"(e4m3x2_bf16x2), "h"(_cvt_satfinite_e4m3x2_bf16x2_69));
    words_a[0] = _mov_b32_pack_u16x2_68;
    uint32_t bf16x2_mul_0;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_0) : "r"(cols[2]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_0) : "r"(cols[3]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_71;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_71) : "r"(bf16x2_mul_1_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_71) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_70;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_70) : "h"(e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_71));
    words_a[1] = _mov_b32_pack_u16x2_70;
    uint32_t bf16x2_mul_2;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2) : "r"(cols[4]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_1) : "r"(cols[5]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_73;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_73) : "r"(bf16x2_mul_1_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_73) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_72;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_72) : "h"(e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_73));
    words_b[0] = _mov_b32_pack_u16x2_72;
    uint32_t bf16x2_mul_3;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3) : "r"(cols[6]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_2) : "r"(cols[7]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_75;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_75) : "r"(bf16x2_mul_1_2));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_2);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_75) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_74;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_74) : "h"(e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_75));
    words_b[1] = _mov_b32_pack_u16x2_74;
    reinterpret_cast<int2*>(col_output + (gcol_a * M + out_row))[0] = reinterpret_cast<int2*>(words_a + 0)[0];
    reinterpret_cast<int2*>(col_output + (gcol_b * M + out_row))[0] = reinterpret_cast<int2*>(words_b + 0)[0];
    if ((lane_1 & 3) == 0) {
        int row_tiles = M >> 7;
        int col_scale_a = ((gcol_a >> 7) * row_tiles + (by >> 2)) * 512 + (gcol_a & 31) * 16 + (gcol_a >> 5 & 3) * 4 + (by & 3);
        int col_scale_b = ((gcol_b >> 7) * row_tiles + (by >> 2)) * 512 + (gcol_b & 31) * 16 + (gcol_b >> 5 & 3) * 4 + (by & 3);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_a) + (0)) = (unsigned char)(scale);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b) + (0)) = (unsigned char)(scale_0);
    }
}

} // extern "C"
