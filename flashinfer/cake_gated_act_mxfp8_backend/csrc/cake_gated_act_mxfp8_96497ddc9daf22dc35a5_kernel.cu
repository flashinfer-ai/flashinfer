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

__device__ __noinline__ unsigned int band_pair(__nv_bfloat16* src, int index, int k)
{
    unsigned int _vec_load_0[1];
    {
        _vec_load_0[0] = *reinterpret_cast<const unsigned int*>(src + index + 0);
    }
    unsigned int _vec_load_1[1];
    {
        _vec_load_1[0] = *reinterpret_cast<const unsigned int*>(src + (index + k) + 0);
    }
    float2 cvt_f32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
    float2 cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
    float e0 = approx_exp2((-cvt_f32.x) * 1.442695f);
    float e1 = approx_exp2((-cvt_f32.y) * 1.442695f);
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
    float d4_8a;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a) : "f"(d0), "f"(0.25f));
    float r4_8a = approx_rcp(d4_8a);
    float _fma_3;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(-d4_8a), "f"(r4_8a), "f"(1.0f));
    float _fma_2;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(r4_8a), "f"(_fma_3), "f"(r4_8a));
    float big_8a;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a) : "f"(mul_f32_rn_noftz(_fma_2, 1.2676506e+30f)), "f"(1.4073749e+14f));
    float _add_4;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(big_8a), "f"(8.388608e+06f));
    float whole_8a;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a) : "f"(_add_4), "f"(8.388608e+06f));
    float above_8a;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a) : "f"(-d4_8a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
    float _sub_6;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6) : "f"(big_8a), "f"(whole_8a));
    float _abs_5;
    asm("abs.f32 %0, %1;" : "=f"(_abs_5) : "f"(_sub_6));
    float _add_7;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(big_8a), "f"(0.5f));
    float _sub_8;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_8) : "f"(big_8a), "f"(0.5f));
    float rounded_8a = ((_abs_5 == 0.5f) ? ((above_8a > 0.0f) ? _add_7 : _sub_8) : whole_8a);
    float u0 = ((d0 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a, 7.888609e-31f), 1.7763568e-15f) : t0);
    float d4_8a_0;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_0) : "f"(d1), "f"(0.25f));
    float r4_8a_0 = approx_rcp(d4_8a_0);
    float _fma_10;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(-d4_8a_0), "f"(r4_8a_0), "f"(1.0f));
    float _fma_9;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(r4_8a_0), "f"(_fma_10), "f"(r4_8a_0));
    float big_8a_0;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_0) : "f"(mul_f32_rn_noftz(_fma_9, 1.2676506e+30f)), "f"(1.4073749e+14f));
    float _add_11;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11) : "f"(big_8a_0), "f"(8.388608e+06f));
    float whole_8a_0;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_0) : "f"(_add_11), "f"(8.388608e+06f));
    float above_8a_0;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_0) : "f"(-d4_8a_0), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_0, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
    float _sub_13;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_13) : "f"(big_8a_0), "f"(whole_8a_0));
    float _abs_12;
    asm("abs.f32 %0, %1;" : "=f"(_abs_12) : "f"(_sub_13));
    float _add_14;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(big_8a_0), "f"(0.5f));
    float _sub_15;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15) : "f"(big_8a_0), "f"(0.5f));
    float rounded_8a_0 = ((_abs_12 == 0.5f) ? ((above_8a_0 > 0.0f) ? _add_14 : _sub_15) : whole_8a_0);
    float u1 = ((d1 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_0, 7.888609e-31f), 1.7763568e-15f) : t1);
    float2 _f32x2_mul_a_16 = mul_f32x2_noftz(cvt_f32, make_float2(max_noftz(u0, 0.0f), max_noftz(u1, 0.0f)));
    float2 mul;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul) : "l"(*(const unsigned long long*)&_f32x2_mul_a_16), "l"(*(const unsigned long long*)&cvt_f32_1));
    unsigned int band_pair_out = __as_u32(__float22bfloat162_rn(make_float2(mul.x, mul.y)));
    return band_pair_out;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_gated_act_mxfp8_96497ddc9daf22dc35a5(__nv_bfloat16* __restrict__ gated_input, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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
    unsigned int _vec_load_2[8];
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
        _vec_load_2[0 + 0] = (uint32_t)_iv_0_0;
        _vec_load_2[0 + 1] = (uint32_t)_iv_0_1;
        _vec_load_2[0 + 2] = (uint32_t)_iv_0_2;
        _vec_load_2[0 + 3] = (uint32_t)_iv_0_3;
        _vec_load_2[0 + 4] = (uint32_t)_iv_0_4;
        _vec_load_2[0 + 5] = (uint32_t)_iv_0_5;
        _vec_load_2[0 + 6] = (uint32_t)_iv_0_6;
        _vec_load_2[0 + 7] = (uint32_t)_iv_0_7;
    }
    unsigned int _vec_load_3[8];
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
        _vec_load_3[0 + 0] = (uint32_t)_iv_1_0;
        _vec_load_3[0 + 1] = (uint32_t)_iv_1_1;
        _vec_load_3[0 + 2] = (uint32_t)_iv_1_2;
        _vec_load_3[0 + 3] = (uint32_t)_iv_1_3;
        _vec_load_3[0 + 4] = (uint32_t)_iv_1_4;
        _vec_load_3[0 + 5] = (uint32_t)_iv_1_5;
        _vec_load_3[0 + 6] = (uint32_t)_iv_1_6;
        _vec_load_3[0 + 7] = (uint32_t)_iv_1_7;
    }
    int grow_hi = grow + 32;
    int gate_index_hi = grow_hi * (2 * K) + col;
    unsigned int _vec_load_4[8];
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
            : "=r"(_iv_2_0), "=r"(_iv_2_1), "=r"(_iv_2_2), "=r"(_iv_2_3), "=r"(_iv_2_4), "=r"(_iv_2_5), "=r"(_iv_2_6), "=r"(_iv_2_7) : "l"((const void*)(gated_input + gate_index_hi + (0))), "l"(0x12F0000000000000ULL) : "memory");
        _vec_load_4[0 + 0] = (uint32_t)_iv_2_0;
        _vec_load_4[0 + 1] = (uint32_t)_iv_2_1;
        _vec_load_4[0 + 2] = (uint32_t)_iv_2_2;
        _vec_load_4[0 + 3] = (uint32_t)_iv_2_3;
        _vec_load_4[0 + 4] = (uint32_t)_iv_2_4;
        _vec_load_4[0 + 5] = (uint32_t)_iv_2_5;
        _vec_load_4[0 + 6] = (uint32_t)_iv_2_6;
        _vec_load_4[0 + 7] = (uint32_t)_iv_2_7;
    }
    unsigned int _vec_load_5[8];
    {
        uint32_t _iv_3_0;
        uint32_t _iv_3_1;
        uint32_t _iv_3_2;
        uint32_t _iv_3_3;
        uint32_t _iv_3_4;
        uint32_t _iv_3_5;
        uint32_t _iv_3_6;
        uint32_t _iv_3_7;
        asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
            : "=r"(_iv_3_0), "=r"(_iv_3_1), "=r"(_iv_3_2), "=r"(_iv_3_3), "=r"(_iv_3_4), "=r"(_iv_3_5), "=r"(_iv_3_6), "=r"(_iv_3_7) : "l"((const void*)(gated_input + (gate_index_hi + K) + (0))), "l"(0x12F0000000000000ULL) : "memory");
        _vec_load_5[0 + 0] = (uint32_t)_iv_3_0;
        _vec_load_5[0 + 1] = (uint32_t)_iv_3_1;
        _vec_load_5[0 + 2] = (uint32_t)_iv_3_2;
        _vec_load_5[0 + 3] = (uint32_t)_iv_3_3;
        _vec_load_5[0 + 4] = (uint32_t)_iv_3_4;
        _vec_load_5[0 + 5] = (uint32_t)_iv_3_5;
        _vec_load_5[0 + 6] = (uint32_t)_iv_3_6;
        _vec_load_5[0 + 7] = (uint32_t)_iv_3_7;
    }
    unsigned int packed[8];
    unsigned int packed_hi[8];
    __nv_bfloat162 lowest = __as_bf16x2(_vec_load_2[0]);
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_2[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_2[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_2[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_2[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_2[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_2[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_2[7]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_4[0]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_4[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_4[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_4[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_4[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_4[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_4[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_4[7]));
    float2 _cvt_f32_2 = cake_bf16x2_to_f32x2(lowest);
    float _min_5;
    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_5) : "f"(_cvt_f32_2.x), "f"(_cvt_f32_2.y));
    int _warp_vote_4 = __all_sync(0xFFFFFFFF, _min_5 > -87.0f);
    if (_warp_vote_4 != 0) {
        float2 cvt_f32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float2 cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
        float e0 = approx_exp2((-cvt_f32.x) * 1.442695f);
        float e1 = approx_exp2((-cvt_f32.y) * 1.442695f);
        float d0 = 1.0f + e0;
        float d1 = 1.0f + e1;
        float r0 = approx_rcp(d0);
        float r1 = approx_rcp(d1);
        float _fma_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(-d0), "f"(r0), "f"(1.0f));
        float _fma_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(r0), "f"(_fma_7), "f"(r0));
        float _fma_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(-d1), "f"(r1), "f"(1.0f));
        float _fma_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(r1), "f"(_fma_9), "f"(r1));
        float2 _f32x2_mul_a_10 = mul_f32x2_noftz(cvt_f32, make_float2(_fma_6, _fma_8));
        float2 mul;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul) : "l"(*(const unsigned long long*)&_f32x2_mul_a_10), "l"(*(const unsigned long long*)&cvt_f32_1));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(mul.x, mul.y)));
        float2 cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float2 cvt_f32_1_0 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[1]));
        float e0_0 = approx_exp2((-cvt_f32_0.x) * 1.442695f);
        float e1_0 = approx_exp2((-cvt_f32_0.y) * 1.442695f);
        float d0_0 = 1.0f + e0_0;
        float d1_0 = 1.0f + e1_0;
        float r0_0 = approx_rcp(d0_0);
        float r1_0 = approx_rcp(d1_0);
        float _fma_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(-d0_0), "f"(r0_0), "f"(1.0f));
        float _fma_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(r0_0), "f"(_fma_12), "f"(r0_0));
        float _fma_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(-d1_0), "f"(r1_0), "f"(1.0f));
        float _fma_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(r1_0), "f"(_fma_14), "f"(r1_0));
        float2 _f32x2_mul_a_15 = mul_f32x2_noftz(cvt_f32_0, make_float2(_fma_11, _fma_13));
        float2 mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&cvt_f32_1_0));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_0.x, mul_0.y)));
        float2 cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float2 cvt_f32_1_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[2]));
        float e0_1 = approx_exp2((-cvt_f32_2.x) * 1.442695f);
        float e1_1 = approx_exp2((-cvt_f32_2.y) * 1.442695f);
        float d0_1 = 1.0f + e0_1;
        float d1_1 = 1.0f + e1_1;
        float r0_1 = approx_rcp(d0_1);
        float r1_1 = approx_rcp(d1_1);
        float _fma_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(-d0_1), "f"(r0_1), "f"(1.0f));
        float _fma_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(r0_1), "f"(_fma_17), "f"(r0_1));
        float _fma_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(-d1_1), "f"(r1_1), "f"(1.0f));
        float _fma_18;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(r1_1), "f"(_fma_19), "f"(r1_1));
        float2 _f32x2_mul_a_20 = mul_f32x2_noftz(cvt_f32_2, make_float2(_fma_16, _fma_18));
        float2 mul_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_20), "l"(*(const unsigned long long*)&cvt_f32_1_1));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_1.x, mul_1.y)));
        float2 cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float2 cvt_f32_1_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[3]));
        float e0_2 = approx_exp2((-cvt_f32_3.x) * 1.442695f);
        float e1_2 = approx_exp2((-cvt_f32_3.y) * 1.442695f);
        float d0_2 = 1.0f + e0_2;
        float d1_2 = 1.0f + e1_2;
        float r0_2 = approx_rcp(d0_2);
        float r1_2 = approx_rcp(d1_2);
        float _fma_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(-d0_2), "f"(r0_2), "f"(1.0f));
        float _fma_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(r0_2), "f"(_fma_22), "f"(r0_2));
        float _fma_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(-d1_2), "f"(r1_2), "f"(1.0f));
        float _fma_23;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(r1_2), "f"(_fma_24), "f"(r1_2));
        float2 _f32x2_mul_a_25 = mul_f32x2_noftz(cvt_f32_3, make_float2(_fma_21, _fma_23));
        float2 mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_25), "l"(*(const unsigned long long*)&cvt_f32_1_2));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_2.x, mul_2.y)));
        float2 cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[4]));
        float2 cvt_f32_1_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[4]));
        float e0_3 = approx_exp2((-cvt_f32_4.x) * 1.442695f);
        float e1_3 = approx_exp2((-cvt_f32_4.y) * 1.442695f);
        float d0_3 = 1.0f + e0_3;
        float d1_3 = 1.0f + e1_3;
        float r0_3 = approx_rcp(d0_3);
        float r1_3 = approx_rcp(d1_3);
        float _fma_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(-d0_3), "f"(r0_3), "f"(1.0f));
        float _fma_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(r0_3), "f"(_fma_27), "f"(r0_3));
        float _fma_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(-d1_3), "f"(r1_3), "f"(1.0f));
        float _fma_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(r1_3), "f"(_fma_29), "f"(r1_3));
        float2 _f32x2_mul_a_30 = mul_f32x2_noftz(cvt_f32_4, make_float2(_fma_26, _fma_28));
        float2 mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&cvt_f32_1_3));
        packed[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_3.x, mul_3.y)));
        float2 cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[5]));
        float2 cvt_f32_1_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[5]));
        float e0_4 = approx_exp2((-cvt_f32_5.x) * 1.442695f);
        float e1_4 = approx_exp2((-cvt_f32_5.y) * 1.442695f);
        float d0_4 = 1.0f + e0_4;
        float d1_4 = 1.0f + e1_4;
        float r0_4 = approx_rcp(d0_4);
        float r1_4 = approx_rcp(d1_4);
        float _fma_32;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(-d0_4), "f"(r0_4), "f"(1.0f));
        float _fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(r0_4), "f"(_fma_32), "f"(r0_4));
        float _fma_34;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(-d1_4), "f"(r1_4), "f"(1.0f));
        float _fma_33;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(r1_4), "f"(_fma_34), "f"(r1_4));
        float2 _f32x2_mul_a_35 = mul_f32x2_noftz(cvt_f32_5, make_float2(_fma_31, _fma_33));
        float2 mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&cvt_f32_1_4));
        packed[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_4.x, mul_4.y)));
        float2 cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[6]));
        float2 cvt_f32_1_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[6]));
        float e0_5 = approx_exp2((-cvt_f32_6.x) * 1.442695f);
        float e1_5 = approx_exp2((-cvt_f32_6.y) * 1.442695f);
        float d0_5 = 1.0f + e0_5;
        float d1_5 = 1.0f + e1_5;
        float r0_5 = approx_rcp(d0_5);
        float r1_5 = approx_rcp(d1_5);
        float _fma_37;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(-d0_5), "f"(r0_5), "f"(1.0f));
        float _fma_36;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(r0_5), "f"(_fma_37), "f"(r0_5));
        float _fma_39;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(-d1_5), "f"(r1_5), "f"(1.0f));
        float _fma_38;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(r1_5), "f"(_fma_39), "f"(r1_5));
        float2 _f32x2_mul_a_40 = mul_f32x2_noftz(cvt_f32_6, make_float2(_fma_36, _fma_38));
        float2 mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_40), "l"(*(const unsigned long long*)&cvt_f32_1_5));
        packed[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_5.x, mul_5.y)));
        float2 cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[7]));
        float2 cvt_f32_1_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[7]));
        float e0_6 = approx_exp2((-cvt_f32_7.x) * 1.442695f);
        float e1_6 = approx_exp2((-cvt_f32_7.y) * 1.442695f);
        float d0_6 = 1.0f + e0_6;
        float d1_6 = 1.0f + e1_6;
        float r0_6 = approx_rcp(d0_6);
        float r1_6 = approx_rcp(d1_6);
        float _fma_42;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(-d0_6), "f"(r0_6), "f"(1.0f));
        float _fma_41;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(r0_6), "f"(_fma_42), "f"(r0_6));
        float _fma_44;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(-d1_6), "f"(r1_6), "f"(1.0f));
        float _fma_43;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(r1_6), "f"(_fma_44), "f"(r1_6));
        float2 _f32x2_mul_a_45 = mul_f32x2_noftz(cvt_f32_7, make_float2(_fma_41, _fma_43));
        float2 mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_45), "l"(*(const unsigned long long*)&cvt_f32_1_6));
        packed[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_6.x, mul_6.y)));
        float2 cvt_f32_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[0]));
        float2 cvt_f32_1_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[0]));
        float e0_7 = approx_exp2((-cvt_f32_8.x) * 1.442695f);
        float e1_7 = approx_exp2((-cvt_f32_8.y) * 1.442695f);
        float d0_7 = 1.0f + e0_7;
        float d1_7 = 1.0f + e1_7;
        float r0_7 = approx_rcp(d0_7);
        float r1_7 = approx_rcp(d1_7);
        float _fma_47;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(-d0_7), "f"(r0_7), "f"(1.0f));
        float _fma_46;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(r0_7), "f"(_fma_47), "f"(r0_7));
        float _fma_49;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(-d1_7), "f"(r1_7), "f"(1.0f));
        float _fma_48;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(r1_7), "f"(_fma_49), "f"(r1_7));
        float2 _f32x2_mul_a_50 = mul_f32x2_noftz(cvt_f32_8, make_float2(_fma_46, _fma_48));
        float2 mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_50), "l"(*(const unsigned long long*)&cvt_f32_1_7));
        packed_hi[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_7.x, mul_7.y)));
        float2 cvt_f32_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[1]));
        float2 cvt_f32_1_8 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[1]));
        float e0_8 = approx_exp2((-cvt_f32_9.x) * 1.442695f);
        float e1_8 = approx_exp2((-cvt_f32_9.y) * 1.442695f);
        float d0_8 = 1.0f + e0_8;
        float d1_8 = 1.0f + e1_8;
        float r0_8 = approx_rcp(d0_8);
        float r1_8 = approx_rcp(d1_8);
        float _fma_52;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(-d0_8), "f"(r0_8), "f"(1.0f));
        float _fma_51;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(r0_8), "f"(_fma_52), "f"(r0_8));
        float _fma_54;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(-d1_8), "f"(r1_8), "f"(1.0f));
        float _fma_53;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(r1_8), "f"(_fma_54), "f"(r1_8));
        float2 _f32x2_mul_a_55 = mul_f32x2_noftz(cvt_f32_9, make_float2(_fma_51, _fma_53));
        float2 mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_55), "l"(*(const unsigned long long*)&cvt_f32_1_8));
        packed_hi[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_8.x, mul_8.y)));
        float2 cvt_f32_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[2]));
        float2 cvt_f32_1_9 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[2]));
        float e0_9 = approx_exp2((-cvt_f32_10.x) * 1.442695f);
        float e1_9 = approx_exp2((-cvt_f32_10.y) * 1.442695f);
        float d0_9 = 1.0f + e0_9;
        float d1_9 = 1.0f + e1_9;
        float r0_9 = approx_rcp(d0_9);
        float r1_9 = approx_rcp(d1_9);
        float _fma_57;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(-d0_9), "f"(r0_9), "f"(1.0f));
        float _fma_56;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(r0_9), "f"(_fma_57), "f"(r0_9));
        float _fma_59;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(-d1_9), "f"(r1_9), "f"(1.0f));
        float _fma_58;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(r1_9), "f"(_fma_59), "f"(r1_9));
        float2 _f32x2_mul_a_60 = mul_f32x2_noftz(cvt_f32_10, make_float2(_fma_56, _fma_58));
        float2 mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&cvt_f32_1_9));
        packed_hi[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_9.x, mul_9.y)));
        float2 cvt_f32_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[3]));
        float2 cvt_f32_1_10 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[3]));
        float e0_10 = approx_exp2((-cvt_f32_11.x) * 1.442695f);
        float e1_10 = approx_exp2((-cvt_f32_11.y) * 1.442695f);
        float d0_10 = 1.0f + e0_10;
        float d1_10 = 1.0f + e1_10;
        float r0_10 = approx_rcp(d0_10);
        float r1_10 = approx_rcp(d1_10);
        float _fma_62;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(-d0_10), "f"(r0_10), "f"(1.0f));
        float _fma_61;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(r0_10), "f"(_fma_62), "f"(r0_10));
        float _fma_64;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(-d1_10), "f"(r1_10), "f"(1.0f));
        float _fma_63;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(r1_10), "f"(_fma_64), "f"(r1_10));
        float2 _f32x2_mul_a_65 = mul_f32x2_noftz(cvt_f32_11, make_float2(_fma_61, _fma_63));
        float2 mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_65), "l"(*(const unsigned long long*)&cvt_f32_1_10));
        packed_hi[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_10.x, mul_10.y)));
        float2 cvt_f32_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[4]));
        float2 cvt_f32_1_11 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[4]));
        float e0_11 = approx_exp2((-cvt_f32_12.x) * 1.442695f);
        float e1_11 = approx_exp2((-cvt_f32_12.y) * 1.442695f);
        float d0_11 = 1.0f + e0_11;
        float d1_11 = 1.0f + e1_11;
        float r0_11 = approx_rcp(d0_11);
        float r1_11 = approx_rcp(d1_11);
        float _fma_67;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(-d0_11), "f"(r0_11), "f"(1.0f));
        float _fma_66;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(r0_11), "f"(_fma_67), "f"(r0_11));
        float _fma_69;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(-d1_11), "f"(r1_11), "f"(1.0f));
        float _fma_68;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(r1_11), "f"(_fma_69), "f"(r1_11));
        float2 _f32x2_mul_a_70 = mul_f32x2_noftz(cvt_f32_12, make_float2(_fma_66, _fma_68));
        float2 mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&cvt_f32_1_11));
        packed_hi[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_11.x, mul_11.y)));
        float2 cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[5]));
        float2 cvt_f32_1_12 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[5]));
        float e0_12 = approx_exp2((-cvt_f32_13.x) * 1.442695f);
        float e1_12 = approx_exp2((-cvt_f32_13.y) * 1.442695f);
        float d0_12 = 1.0f + e0_12;
        float d1_12 = 1.0f + e1_12;
        float r0_12 = approx_rcp(d0_12);
        float r1_12 = approx_rcp(d1_12);
        float _fma_72;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(-d0_12), "f"(r0_12), "f"(1.0f));
        float _fma_71;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(r0_12), "f"(_fma_72), "f"(r0_12));
        float _fma_74;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(-d1_12), "f"(r1_12), "f"(1.0f));
        float _fma_73;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(r1_12), "f"(_fma_74), "f"(r1_12));
        float2 _f32x2_mul_a_75 = mul_f32x2_noftz(cvt_f32_13, make_float2(_fma_71, _fma_73));
        float2 mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_75), "l"(*(const unsigned long long*)&cvt_f32_1_12));
        packed_hi[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_12.x, mul_12.y)));
        float2 cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[6]));
        float2 cvt_f32_1_13 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[6]));
        float e0_13 = approx_exp2((-cvt_f32_14.x) * 1.442695f);
        float e1_13 = approx_exp2((-cvt_f32_14.y) * 1.442695f);
        float d0_13 = 1.0f + e0_13;
        float d1_13 = 1.0f + e1_13;
        float r0_13 = approx_rcp(d0_13);
        float r1_13 = approx_rcp(d1_13);
        float _fma_77;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_77) : "f"(-d0_13), "f"(r0_13), "f"(1.0f));
        float _fma_76;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(r0_13), "f"(_fma_77), "f"(r0_13));
        float _fma_79;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(-d1_13), "f"(r1_13), "f"(1.0f));
        float _fma_78;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_78) : "f"(r1_13), "f"(_fma_79), "f"(r1_13));
        float2 _f32x2_mul_a_80 = mul_f32x2_noftz(cvt_f32_14, make_float2(_fma_76, _fma_78));
        float2 mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_80), "l"(*(const unsigned long long*)&cvt_f32_1_13));
        packed_hi[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_13.x, mul_13.y)));
        float2 cvt_f32_15 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[7]));
        float2 cvt_f32_1_14 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[7]));
        float e0_14 = approx_exp2((-cvt_f32_15.x) * 1.442695f);
        float e1_14 = approx_exp2((-cvt_f32_15.y) * 1.442695f);
        float d0_14 = 1.0f + e0_14;
        float d1_14 = 1.0f + e1_14;
        float r0_14 = approx_rcp(d0_14);
        float r1_14 = approx_rcp(d1_14);
        float _fma_82;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(-d0_14), "f"(r0_14), "f"(1.0f));
        float _fma_81;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(r0_14), "f"(_fma_82), "f"(r0_14));
        float _fma_84;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(-d1_14), "f"(r1_14), "f"(1.0f));
        float _fma_83;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(r1_14), "f"(_fma_84), "f"(r1_14));
        float2 _f32x2_mul_a_85 = mul_f32x2_noftz(cvt_f32_15, make_float2(_fma_81, _fma_83));
        float2 mul_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_85), "l"(*(const unsigned long long*)&cvt_f32_1_14));
        packed_hi[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_14.x, mul_14.y)));
    } else {
        int band = 0;
        float2 cvt_f32_16 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[0]));
        float2 cvt_f32_1_15 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[0]));
        float e0_15 = approx_exp2((-cvt_f32_16.x) * 1.442695f);
        float e1_15 = approx_exp2((-cvt_f32_16.y) * 1.442695f);
        float d0_15 = 1.0f + e0_15;
        float d1_15 = 1.0f + e1_15;
        float r0_15 = approx_rcp(d0_15);
        float r1_15 = approx_rcp(d1_15);
        float _fma_86;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(-d0_15), "f"(r0_15), "f"(1.0f));
        float fma;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma) : "f"(r0_15), "f"(_fma_86), "f"(r0_15));
        float _fma_87;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(-d1_15), "f"(r1_15), "f"(1.0f));
        float fma_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1) : "f"(r1_15), "f"(_fma_87), "f"(r1_15));
        float t0 = fma;
        float t1 = fma_1;
        band = band | ((fminf(t0, t1) == 0.0f) ? 1 : 0);
        float2 _f32x2_mul_a_88 = mul_f32x2_noftz(cvt_f32_16, make_float2(max_noftz(t0, 0.0f), max_noftz(t1, 0.0f)));
        float2 mul_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_88), "l"(*(const unsigned long long*)&cvt_f32_1_15));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_15.x, mul_15.y)));
        float2 cvt_f32_16_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[1]));
        float2 cvt_f32_1_15_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[1]));
        float e0_15_1 = approx_exp2((-cvt_f32_16_1.x) * 1.442695f);
        float e1_15_1 = approx_exp2((-cvt_f32_16_1.y) * 1.442695f);
        float d0_15_1 = 1.0f + e0_15_1;
        float d1_15_1 = 1.0f + e1_15_1;
        float r0_15_1 = approx_rcp(d0_15_1);
        float r1_15_1 = approx_rcp(d1_15_1);
        float _fma_89;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(-d0_15_1), "f"(r0_15_1), "f"(1.0f));
        float fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_0) : "f"(r0_15_1), "f"(_fma_89), "f"(r0_15_1));
        float _fma_90;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(-d1_15_1), "f"(r1_15_1), "f"(1.0f));
        float fma_1_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_0) : "f"(r1_15_1), "f"(_fma_90), "f"(r1_15_1));
        float t0_0 = fma_0;
        float t1_0 = fma_1_0;
        band = band | ((fminf(t0_0, t1_0) == 0.0f) ? 2 : 0);
        float2 _f32x2_mul_a_91 = mul_f32x2_noftz(cvt_f32_16_1, make_float2(max_noftz(t0_0, 0.0f), max_noftz(t1_0, 0.0f)));
        float2 mul_15_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_15_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_91), "l"(*(const unsigned long long*)&cvt_f32_1_15_1));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_15_1.x, mul_15_1.y)));
        float2 cvt_f32_17 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[2]));
        float2 cvt_f32_1_16 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[2]));
        float e0_16 = approx_exp2((-cvt_f32_17.x) * 1.442695f);
        float e1_16 = approx_exp2((-cvt_f32_17.y) * 1.442695f);
        float d0_16 = 1.0f + e0_16;
        float d1_16 = 1.0f + e1_16;
        float r0_16 = approx_rcp(d0_16);
        float r1_16 = approx_rcp(d1_16);
        float _fma_92;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(-d0_16), "f"(r0_16), "f"(1.0f));
        float fma_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_2) : "f"(r0_16), "f"(_fma_92), "f"(r0_16));
        float _fma_93;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_93) : "f"(-d1_16), "f"(r1_16), "f"(1.0f));
        float fma_1_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_1) : "f"(r1_16), "f"(_fma_93), "f"(r1_16));
        float t0_1 = fma_2;
        float t1_1 = fma_1_1;
        band = band | ((fminf(t0_1, t1_1) == 0.0f) ? 4 : 0);
        float2 _f32x2_mul_a_94 = mul_f32x2_noftz(cvt_f32_17, make_float2(max_noftz(t0_1, 0.0f), max_noftz(t1_1, 0.0f)));
        float2 mul_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_94), "l"(*(const unsigned long long*)&cvt_f32_1_16));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_16.x, mul_16.y)));
        float2 cvt_f32_18 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[3]));
        float2 cvt_f32_1_17 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[3]));
        float e0_17 = approx_exp2((-cvt_f32_18.x) * 1.442695f);
        float e1_17 = approx_exp2((-cvt_f32_18.y) * 1.442695f);
        float d0_17 = 1.0f + e0_17;
        float d1_17 = 1.0f + e1_17;
        float r0_17 = approx_rcp(d0_17);
        float r1_17 = approx_rcp(d1_17);
        float _fma_95;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(-d0_17), "f"(r0_17), "f"(1.0f));
        float fma_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_3) : "f"(r0_17), "f"(_fma_95), "f"(r0_17));
        float _fma_96;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(-d1_17), "f"(r1_17), "f"(1.0f));
        float fma_1_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_2) : "f"(r1_17), "f"(_fma_96), "f"(r1_17));
        float t0_2 = fma_3;
        float t1_2 = fma_1_2;
        band = band | ((fminf(t0_2, t1_2) == 0.0f) ? 8 : 0);
        float2 _f32x2_mul_a_97 = mul_f32x2_noftz(cvt_f32_18, make_float2(max_noftz(t0_2, 0.0f), max_noftz(t1_2, 0.0f)));
        float2 mul_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_97), "l"(*(const unsigned long long*)&cvt_f32_1_17));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_17.x, mul_17.y)));
        float2 cvt_f32_19 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[4]));
        float2 cvt_f32_1_18 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[4]));
        float e0_18 = approx_exp2((-cvt_f32_19.x) * 1.442695f);
        float e1_18 = approx_exp2((-cvt_f32_19.y) * 1.442695f);
        float d0_18 = 1.0f + e0_18;
        float d1_18 = 1.0f + e1_18;
        float r0_18 = approx_rcp(d0_18);
        float r1_18 = approx_rcp(d1_18);
        float _fma_98;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(-d0_18), "f"(r0_18), "f"(1.0f));
        float fma_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_4) : "f"(r0_18), "f"(_fma_98), "f"(r0_18));
        float _fma_99;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(-d1_18), "f"(r1_18), "f"(1.0f));
        float fma_1_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_3) : "f"(r1_18), "f"(_fma_99), "f"(r1_18));
        float t0_3 = fma_4;
        float t1_3 = fma_1_3;
        band = band | ((fminf(t0_3, t1_3) == 0.0f) ? 16 : 0);
        float2 _f32x2_mul_a_100 = mul_f32x2_noftz(cvt_f32_19, make_float2(max_noftz(t0_3, 0.0f), max_noftz(t1_3, 0.0f)));
        float2 mul_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_100), "l"(*(const unsigned long long*)&cvt_f32_1_18));
        packed[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_18.x, mul_18.y)));
        float2 cvt_f32_20 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[5]));
        float2 cvt_f32_1_19 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[5]));
        float e0_19 = approx_exp2((-cvt_f32_20.x) * 1.442695f);
        float e1_19 = approx_exp2((-cvt_f32_20.y) * 1.442695f);
        float d0_19 = 1.0f + e0_19;
        float d1_19 = 1.0f + e1_19;
        float r0_19 = approx_rcp(d0_19);
        float r1_19 = approx_rcp(d1_19);
        float _fma_101;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(-d0_19), "f"(r0_19), "f"(1.0f));
        float fma_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_5) : "f"(r0_19), "f"(_fma_101), "f"(r0_19));
        float _fma_102;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_102) : "f"(-d1_19), "f"(r1_19), "f"(1.0f));
        float fma_1_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_4) : "f"(r1_19), "f"(_fma_102), "f"(r1_19));
        float t0_4 = fma_5;
        float t1_4 = fma_1_4;
        band = band | ((fminf(t0_4, t1_4) == 0.0f) ? 32 : 0);
        float2 _f32x2_mul_a_103 = mul_f32x2_noftz(cvt_f32_20, make_float2(max_noftz(t0_4, 0.0f), max_noftz(t1_4, 0.0f)));
        float2 mul_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_103), "l"(*(const unsigned long long*)&cvt_f32_1_19));
        packed[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_19.x, mul_19.y)));
        float2 cvt_f32_21 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[6]));
        float2 cvt_f32_1_20 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[6]));
        float e0_20 = approx_exp2((-cvt_f32_21.x) * 1.442695f);
        float e1_20 = approx_exp2((-cvt_f32_21.y) * 1.442695f);
        float d0_20 = 1.0f + e0_20;
        float d1_20 = 1.0f + e1_20;
        float r0_20 = approx_rcp(d0_20);
        float r1_20 = approx_rcp(d1_20);
        float _fma_104;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_104) : "f"(-d0_20), "f"(r0_20), "f"(1.0f));
        float fma_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_6) : "f"(r0_20), "f"(_fma_104), "f"(r0_20));
        float _fma_105;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_105) : "f"(-d1_20), "f"(r1_20), "f"(1.0f));
        float fma_1_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_5) : "f"(r1_20), "f"(_fma_105), "f"(r1_20));
        float t0_5 = fma_6;
        float t1_5 = fma_1_5;
        band = band | ((fminf(t0_5, t1_5) == 0.0f) ? 64 : 0);
        float2 _f32x2_mul_a_106 = mul_f32x2_noftz(cvt_f32_21, make_float2(max_noftz(t0_5, 0.0f), max_noftz(t1_5, 0.0f)));
        float2 mul_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_106), "l"(*(const unsigned long long*)&cvt_f32_1_20));
        packed[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_20.x, mul_20.y)));
        float2 cvt_f32_22 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_2[7]));
        float2 cvt_f32_1_21 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_3[7]));
        float e0_21 = approx_exp2((-cvt_f32_22.x) * 1.442695f);
        float e1_21 = approx_exp2((-cvt_f32_22.y) * 1.442695f);
        float d0_21 = 1.0f + e0_21;
        float d1_21 = 1.0f + e1_21;
        float r0_21 = approx_rcp(d0_21);
        float r1_21 = approx_rcp(d1_21);
        float _fma_107;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(-d0_21), "f"(r0_21), "f"(1.0f));
        float fma_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_7) : "f"(r0_21), "f"(_fma_107), "f"(r0_21));
        float _fma_108;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_108) : "f"(-d1_21), "f"(r1_21), "f"(1.0f));
        float fma_1_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_6) : "f"(r1_21), "f"(_fma_108), "f"(r1_21));
        float t0_6 = fma_7;
        float t1_6 = fma_1_6;
        band = band | ((fminf(t0_6, t1_6) == 0.0f) ? 128 : 0);
        float2 _f32x2_mul_a_109 = mul_f32x2_noftz(cvt_f32_22, make_float2(max_noftz(t0_6, 0.0f), max_noftz(t1_6, 0.0f)));
        float2 mul_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_109), "l"(*(const unsigned long long*)&cvt_f32_1_21));
        packed[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_21.x, mul_21.y)));
        int band_hi = 0;
        float2 cvt_f32_23 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[0]));
        float2 cvt_f32_1_22 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[0]));
        float e0_22 = approx_exp2((-cvt_f32_23.x) * 1.442695f);
        float e1_22 = approx_exp2((-cvt_f32_23.y) * 1.442695f);
        float d0_22 = 1.0f + e0_22;
        float d1_22 = 1.0f + e1_22;
        float r0_22 = approx_rcp(d0_22);
        float r1_22 = approx_rcp(d1_22);
        float _fma_110;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_110) : "f"(-d0_22), "f"(r0_22), "f"(1.0f));
        float fma_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_8) : "f"(r0_22), "f"(_fma_110), "f"(r0_22));
        float _fma_111;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_111) : "f"(-d1_22), "f"(r1_22), "f"(1.0f));
        float fma_1_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_7) : "f"(r1_22), "f"(_fma_111), "f"(r1_22));
        float t0_7 = fma_8;
        float t1_7 = fma_1_7;
        band_hi = band_hi | ((fminf(t0_7, t1_7) == 0.0f) ? 1 : 0);
        float2 _f32x2_mul_a_112 = mul_f32x2_noftz(cvt_f32_23, make_float2(max_noftz(t0_7, 0.0f), max_noftz(t1_7, 0.0f)));
        float2 mul_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_112), "l"(*(const unsigned long long*)&cvt_f32_1_22));
        packed_hi[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_22.x, mul_22.y)));
        float2 cvt_f32_24 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[1]));
        float2 cvt_f32_1_23 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[1]));
        float e0_23 = approx_exp2((-cvt_f32_24.x) * 1.442695f);
        float e1_23 = approx_exp2((-cvt_f32_24.y) * 1.442695f);
        float d0_23 = 1.0f + e0_23;
        float d1_23 = 1.0f + e1_23;
        float r0_23 = approx_rcp(d0_23);
        float r1_23 = approx_rcp(d1_23);
        float _fma_113;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_113) : "f"(-d0_23), "f"(r0_23), "f"(1.0f));
        float fma_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_9) : "f"(r0_23), "f"(_fma_113), "f"(r0_23));
        float _fma_114;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_114) : "f"(-d1_23), "f"(r1_23), "f"(1.0f));
        float fma_1_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_8) : "f"(r1_23), "f"(_fma_114), "f"(r1_23));
        float t0_8 = fma_9;
        float t1_8 = fma_1_8;
        band_hi = band_hi | ((fminf(t0_8, t1_8) == 0.0f) ? 2 : 0);
        float2 _f32x2_mul_a_115 = mul_f32x2_noftz(cvt_f32_24, make_float2(max_noftz(t0_8, 0.0f), max_noftz(t1_8, 0.0f)));
        float2 mul_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_115), "l"(*(const unsigned long long*)&cvt_f32_1_23));
        packed_hi[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_23.x, mul_23.y)));
        float2 cvt_f32_25 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[2]));
        float2 cvt_f32_1_24 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[2]));
        float e0_24 = approx_exp2((-cvt_f32_25.x) * 1.442695f);
        float e1_24 = approx_exp2((-cvt_f32_25.y) * 1.442695f);
        float d0_24 = 1.0f + e0_24;
        float d1_24 = 1.0f + e1_24;
        float r0_24 = approx_rcp(d0_24);
        float r1_24 = approx_rcp(d1_24);
        float _fma_116;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_116) : "f"(-d0_24), "f"(r0_24), "f"(1.0f));
        float fma_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_10) : "f"(r0_24), "f"(_fma_116), "f"(r0_24));
        float _fma_117;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_117) : "f"(-d1_24), "f"(r1_24), "f"(1.0f));
        float fma_1_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_9) : "f"(r1_24), "f"(_fma_117), "f"(r1_24));
        float t0_9 = fma_10;
        float t1_9 = fma_1_9;
        band_hi = band_hi | ((fminf(t0_9, t1_9) == 0.0f) ? 4 : 0);
        float2 _f32x2_mul_a_118 = mul_f32x2_noftz(cvt_f32_25, make_float2(max_noftz(t0_9, 0.0f), max_noftz(t1_9, 0.0f)));
        float2 mul_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_118), "l"(*(const unsigned long long*)&cvt_f32_1_24));
        packed_hi[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_24.x, mul_24.y)));
        float2 cvt_f32_26 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[3]));
        float2 cvt_f32_1_25 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[3]));
        float e0_25 = approx_exp2((-cvt_f32_26.x) * 1.442695f);
        float e1_25 = approx_exp2((-cvt_f32_26.y) * 1.442695f);
        float d0_25 = 1.0f + e0_25;
        float d1_25 = 1.0f + e1_25;
        float r0_25 = approx_rcp(d0_25);
        float r1_25 = approx_rcp(d1_25);
        float _fma_119;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_119) : "f"(-d0_25), "f"(r0_25), "f"(1.0f));
        float fma_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_11) : "f"(r0_25), "f"(_fma_119), "f"(r0_25));
        float _fma_120;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_120) : "f"(-d1_25), "f"(r1_25), "f"(1.0f));
        float fma_1_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_10) : "f"(r1_25), "f"(_fma_120), "f"(r1_25));
        float t0_10 = fma_11;
        float t1_10 = fma_1_10;
        band_hi = band_hi | ((fminf(t0_10, t1_10) == 0.0f) ? 8 : 0);
        float2 _f32x2_mul_a_121 = mul_f32x2_noftz(cvt_f32_26, make_float2(max_noftz(t0_10, 0.0f), max_noftz(t1_10, 0.0f)));
        float2 mul_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_121), "l"(*(const unsigned long long*)&cvt_f32_1_25));
        packed_hi[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_25.x, mul_25.y)));
        float2 cvt_f32_27 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[4]));
        float2 cvt_f32_1_26 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[4]));
        float e0_26 = approx_exp2((-cvt_f32_27.x) * 1.442695f);
        float e1_26 = approx_exp2((-cvt_f32_27.y) * 1.442695f);
        float d0_26 = 1.0f + e0_26;
        float d1_26 = 1.0f + e1_26;
        float r0_26 = approx_rcp(d0_26);
        float r1_26 = approx_rcp(d1_26);
        float _fma_122;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_122) : "f"(-d0_26), "f"(r0_26), "f"(1.0f));
        float fma_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_12) : "f"(r0_26), "f"(_fma_122), "f"(r0_26));
        float _fma_123;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_123) : "f"(-d1_26), "f"(r1_26), "f"(1.0f));
        float fma_1_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_11) : "f"(r1_26), "f"(_fma_123), "f"(r1_26));
        float t0_11 = fma_12;
        float t1_11 = fma_1_11;
        band_hi = band_hi | ((fminf(t0_11, t1_11) == 0.0f) ? 16 : 0);
        float2 _f32x2_mul_a_124 = mul_f32x2_noftz(cvt_f32_27, make_float2(max_noftz(t0_11, 0.0f), max_noftz(t1_11, 0.0f)));
        float2 mul_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_124), "l"(*(const unsigned long long*)&cvt_f32_1_26));
        packed_hi[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_26.x, mul_26.y)));
        float2 cvt_f32_28 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[5]));
        float2 cvt_f32_1_27 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[5]));
        float e0_27 = approx_exp2((-cvt_f32_28.x) * 1.442695f);
        float e1_27 = approx_exp2((-cvt_f32_28.y) * 1.442695f);
        float d0_27 = 1.0f + e0_27;
        float d1_27 = 1.0f + e1_27;
        float r0_27 = approx_rcp(d0_27);
        float r1_27 = approx_rcp(d1_27);
        float _fma_125;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(-d0_27), "f"(r0_27), "f"(1.0f));
        float fma_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_13) : "f"(r0_27), "f"(_fma_125), "f"(r0_27));
        float _fma_126;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_126) : "f"(-d1_27), "f"(r1_27), "f"(1.0f));
        float fma_1_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_12) : "f"(r1_27), "f"(_fma_126), "f"(r1_27));
        float t0_12 = fma_13;
        float t1_12 = fma_1_12;
        band_hi = band_hi | ((fminf(t0_12, t1_12) == 0.0f) ? 32 : 0);
        float2 _f32x2_mul_a_127 = mul_f32x2_noftz(cvt_f32_28, make_float2(max_noftz(t0_12, 0.0f), max_noftz(t1_12, 0.0f)));
        float2 mul_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_127), "l"(*(const unsigned long long*)&cvt_f32_1_27));
        packed_hi[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_27.x, mul_27.y)));
        float2 cvt_f32_29 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[6]));
        float2 cvt_f32_1_28 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[6]));
        float e0_28 = approx_exp2((-cvt_f32_29.x) * 1.442695f);
        float e1_28 = approx_exp2((-cvt_f32_29.y) * 1.442695f);
        float d0_28 = 1.0f + e0_28;
        float d1_28 = 1.0f + e1_28;
        float r0_28 = approx_rcp(d0_28);
        float r1_28 = approx_rcp(d1_28);
        float _fma_128;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_128) : "f"(-d0_28), "f"(r0_28), "f"(1.0f));
        float fma_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_14) : "f"(r0_28), "f"(_fma_128), "f"(r0_28));
        float _fma_129;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_129) : "f"(-d1_28), "f"(r1_28), "f"(1.0f));
        float fma_1_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_13) : "f"(r1_28), "f"(_fma_129), "f"(r1_28));
        float t0_13 = fma_14;
        float t1_13 = fma_1_13;
        band_hi = band_hi | ((fminf(t0_13, t1_13) == 0.0f) ? 64 : 0);
        float2 _f32x2_mul_a_130 = mul_f32x2_noftz(cvt_f32_29, make_float2(max_noftz(t0_13, 0.0f), max_noftz(t1_13, 0.0f)));
        float2 mul_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_130), "l"(*(const unsigned long long*)&cvt_f32_1_28));
        packed_hi[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_28.x, mul_28.y)));
        float2 cvt_f32_30 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_4[7]));
        float2 cvt_f32_1_29 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_5[7]));
        float e0_29 = approx_exp2((-cvt_f32_30.x) * 1.442695f);
        float e1_29 = approx_exp2((-cvt_f32_30.y) * 1.442695f);
        float d0_29 = 1.0f + e0_29;
        float d1_29 = 1.0f + e1_29;
        float r0_29 = approx_rcp(d0_29);
        float r1_29 = approx_rcp(d1_29);
        float _fma_131;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_131) : "f"(-d0_29), "f"(r0_29), "f"(1.0f));
        float fma_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_15) : "f"(r0_29), "f"(_fma_131), "f"(r0_29));
        float _fma_132;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_132) : "f"(-d1_29), "f"(r1_29), "f"(1.0f));
        float fma_1_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1_14) : "f"(r1_29), "f"(_fma_132), "f"(r1_29));
        float t0_14 = fma_15;
        float t1_14 = fma_1_14;
        band_hi = band_hi | ((fminf(t0_14, t1_14) == 0.0f) ? 128 : 0);
        float2 _f32x2_mul_a_133 = mul_f32x2_noftz(cvt_f32_30, make_float2(max_noftz(t0_14, 0.0f), max_noftz(t1_14, 0.0f)));
        float2 mul_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_133), "l"(*(const unsigned long long*)&cvt_f32_1_29));
        packed_hi[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_29.x, mul_29.y)));
        if (band != 0) {
            unsigned int _popc_0 = (unsigned int)__popc(band);
            #pragma unroll 1
            for (int _j = 0; _j < _popc_0; _j++) {
                int _find_first_set_134 = __ffs(band);
                int j = _find_first_set_134 - 1;
                band = band & band - 1;
                unsigned int pair = band_pair(gated_input, gate_index + 2 * j, K);
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
                int _find_first_set_135 = __ffs(band_hi);
                int j_hi = _find_first_set_135 - 1;
                band_hi = band_hi & band_hi - 1;
                unsigned int pair_hi = band_pair(gated_input, gate_index_hi + 2 * j_hi, K);
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
    __nv_bfloat162 _hxa_137 = amax_a;
    __nv_bfloat162 _hxb_137 = __as_bf16x2(cols[0]);
    unsigned int _hxr_137;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_137) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_137)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_137)));
    __nv_bfloat162 _max_136 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_137);
    amax_a = _max_136;
    __nv_bfloat162 _hxa_139 = amax_b;
    __nv_bfloat162 _hxb_139 = __as_bf16x2(cols[8]);
    unsigned int _hxr_139;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_139) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_139)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_139)));
    __nv_bfloat162 _max_138 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_139);
    amax_b = _max_138;
    __nv_bfloat162 _hxa_141 = amax_a;
    __nv_bfloat162 _hxb_141 = __as_bf16x2(cols[1]);
    unsigned int _hxr_141;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_141) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_141)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_141)));
    __nv_bfloat162 _max_140 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_141);
    amax_a = _max_140;
    __nv_bfloat162 _hxa_143 = amax_b;
    __nv_bfloat162 _hxb_143 = __as_bf16x2(cols[9]);
    unsigned int _hxr_143;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_143) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_143)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_143)));
    __nv_bfloat162 _max_142 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_143);
    amax_b = _max_142;
    __nv_bfloat162 _hxa_145 = amax_a;
    __nv_bfloat162 _hxb_145 = __as_bf16x2(cols[2]);
    unsigned int _hxr_145;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_145) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_145)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_145)));
    __nv_bfloat162 _max_144 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_145);
    amax_a = _max_144;
    __nv_bfloat162 _hxa_147 = amax_b;
    __nv_bfloat162 _hxb_147 = __as_bf16x2(cols[10]);
    unsigned int _hxr_147;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_147) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_147)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_147)));
    __nv_bfloat162 _max_146 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_147);
    amax_b = _max_146;
    __nv_bfloat162 _hxa_149 = amax_a;
    __nv_bfloat162 _hxb_149 = __as_bf16x2(cols[3]);
    unsigned int _hxr_149;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_149) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_149)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_149)));
    __nv_bfloat162 _max_148 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_149);
    amax_a = _max_148;
    __nv_bfloat162 _hxa_151 = amax_b;
    __nv_bfloat162 _hxb_151 = __as_bf16x2(cols[11]);
    unsigned int _hxr_151;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_151) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_151)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_151)));
    __nv_bfloat162 _max_150 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_151);
    amax_b = _max_150;
    __nv_bfloat162 _hxa_153 = amax_a;
    __nv_bfloat162 _hxb_153 = __as_bf16x2(cols[4]);
    unsigned int _hxr_153;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_153) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_153)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_153)));
    __nv_bfloat162 _max_152 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_153);
    amax_a = _max_152;
    __nv_bfloat162 _hxa_155 = amax_b;
    __nv_bfloat162 _hxb_155 = __as_bf16x2(cols[12]);
    unsigned int _hxr_155;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_155) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_155)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_155)));
    __nv_bfloat162 _max_154 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_155);
    amax_b = _max_154;
    __nv_bfloat162 _hxa_157 = amax_a;
    __nv_bfloat162 _hxb_157 = __as_bf16x2(cols[5]);
    unsigned int _hxr_157;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_157) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_157)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_157)));
    __nv_bfloat162 _max_156 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_157);
    amax_a = _max_156;
    __nv_bfloat162 _hxa_159 = amax_b;
    __nv_bfloat162 _hxb_159 = __as_bf16x2(cols[13]);
    unsigned int _hxr_159;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_159) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_159)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_159)));
    __nv_bfloat162 _max_158 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_159);
    amax_b = _max_158;
    __nv_bfloat162 _hxa_161 = amax_a;
    __nv_bfloat162 _hxb_161 = __as_bf16x2(cols[6]);
    unsigned int _hxr_161;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_161) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_161)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_161)));
    __nv_bfloat162 _max_160 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_161);
    amax_a = _max_160;
    __nv_bfloat162 _hxa_163 = amax_b;
    __nv_bfloat162 _hxb_163 = __as_bf16x2(cols[14]);
    unsigned int _hxr_163;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_163) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_163)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_163)));
    __nv_bfloat162 _max_162 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_163);
    amax_b = _max_162;
    __nv_bfloat162 _hxa_165 = amax_a;
    __nv_bfloat162 _hxb_165 = __as_bf16x2(cols[7]);
    unsigned int _hxr_165;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_165) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_165)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_165)));
    __nv_bfloat162 _max_164 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_165);
    amax_a = _max_164;
    __nv_bfloat162 _hxa_167 = amax_b;
    __nv_bfloat162 _hxb_167 = __as_bf16x2(cols[15]);
    unsigned int _hxr_167;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_167) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_167)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_167)));
    __nv_bfloat162 _max_166 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_167);
    amax_b = _max_166;
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
    int out_row = by * 64 + (lane_1 & 3) * 16;
    unsigned int words_a[4];
    unsigned int words_b[4];
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
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_169;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_169) : "r"(bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_169) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_168;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_168) : "h"(e4m3x2_bf16x2), "h"(_cvt_satfinite_e4m3x2_bf16x2_169));
    words_a[0] = _mov_b32_pack_u16x2_168;
    uint32_t bf16x2_mul_0;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_0) : "r"(cols[8]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_0) : "r"(cols[9]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_171;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_171) : "r"(bf16x2_mul_1_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_171) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_170;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_170) : "h"(e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_171));
    words_b[0] = _mov_b32_pack_u16x2_170;
    uint32_t bf16x2_mul_2;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2) : "r"(cols[2]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_1) : "r"(cols[3]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_173;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_173) : "r"(bf16x2_mul_1_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_173) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_172;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_172) : "h"(e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_173));
    words_a[1] = _mov_b32_pack_u16x2_172;
    uint32_t bf16x2_mul_3;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3) : "r"(cols[10]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_2) : "r"(cols[11]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_175;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_175) : "r"(bf16x2_mul_1_2));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_2);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_175) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_174;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_174) : "h"(e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_175));
    words_b[1] = _mov_b32_pack_u16x2_174;
    uint32_t bf16x2_mul_4;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4) : "r"(cols[4]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_3) : "r"(cols[5]), "r"(inv2));
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
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_176) : "h"(e4m3x2_bf16x2_3), "h"(_cvt_satfinite_e4m3x2_bf16x2_177));
    words_a[2] = _mov_b32_pack_u16x2_176;
    uint32_t bf16x2_mul_5;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5) : "r"(cols[12]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_4) : "r"(cols[13]), "r"(inv2_0));
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
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_178) : "h"(e4m3x2_bf16x2_4), "h"(_cvt_satfinite_e4m3x2_bf16x2_179));
    words_b[2] = _mov_b32_pack_u16x2_178;
    uint32_t bf16x2_mul_6;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6) : "r"(cols[6]), "r"(inv2));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_5) : "r"(cols[7]), "r"(inv2));
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
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_180) : "h"(e4m3x2_bf16x2_5), "h"(_cvt_satfinite_e4m3x2_bf16x2_181));
    words_a[3] = _mov_b32_pack_u16x2_180;
    uint32_t bf16x2_mul_7;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7) : "r"(cols[14]), "r"(inv2_0));
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
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_6) : "r"(cols[15]), "r"(inv2_0));
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
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_182) : "h"(e4m3x2_bf16x2_6), "h"(_cvt_satfinite_e4m3x2_bf16x2_183));
    words_b[3] = _mov_b32_pack_u16x2_182;
    reinterpret_cast<int4*>(col_output + (gcol_a * M + out_row))[0] = reinterpret_cast<int4*>(words_a + 0)[0];
    reinterpret_cast<int4*>(col_output + (gcol_b * M + out_row))[0] = reinterpret_cast<int4*>(words_b + 0)[0];
    if ((lane_1 & 1) == 0) {
        int row_tiles = M >> 7;
        int rblk = by * 2 + ((lane_1 & 3) >> 1);
        int col_scale_a = ((gcol_a >> 7) * row_tiles + (rblk >> 2)) * 512 + (gcol_a & 31) * 16 + (gcol_a >> 5 & 3) * 4 + (rblk & 3);
        int col_scale_b = ((gcol_b >> 7) * row_tiles + (rblk >> 2)) * 512 + (gcol_b & 31) * 16 + (gcol_b >> 5 & 3) * 4 + (rblk & 3);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_a) + (0)) = (unsigned char)(scale);
        *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b) + (0)) = (unsigned char)(scale_0);
    }
}

} // extern "C"
